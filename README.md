# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-01-14 | 今日论文总数: 452

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. S4 modal sequent calculus as intermediate logic and intermediate language

**arXiv ID:** 2601.08071 | [PDF](https://arxiv.org/pdf/2601.08071v1)

**作者:** Jean Caspar `[一作]`, Guillaume Munch-Maccagnoni `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出一种三极化的序列算子系统，构建了对应的中间语言与中间逻辑，证明了其与经典模态逻辑 S4 的对应关系，并给出了堆/栈分离的机器语义，进一步证明了在该语义下满足“堆栈性”性质，即满足第二类续延的栈式管理。

**💡 创新点**

创新点在于：①将中间语言与中间逻辑对应的思想正式化，提出三极化的 S4 序列算子；②通过在语义中显式分离堆和栈，揭示了第二类续延的堆栈性；③利用 Gödel‑McKinsey‑Tarski 定理，将 S4 的模态片段视为一个中间逻辑，提供了一个新的视角来解释编译器中续延的线性约束。

**🔧 技术方法**

主要技术包括：
- 极化序列算子（正、负、模态极化）
- 形态分解与多极化类型系统
- 机器语义（堆/栈分离、限制分配）
- 归约与类型保守性证明
- 对 S4 的形式化翻译与证明（Gödel‑McKinsey‑Tarski）。

**📊 数据集**

无使用数据集；该工作为理论研究，未涉及实验数据。

**📈 对比分析**

本工作没有进行实验比较或性能评估；主要通过形式化证明展示该语义满足预期性质（强归约性、类型安全、堆栈性）。

**⚠️ 局限性**

局限性包括：
- 仍未给出完整的 CPS 翻译细节与实现细节，依赖外部工作；
- 只在理论层面证明堆栈性，缺乏在实际编译器中的实证验证；
- 对非模态返回类型的栈管理仍存在奇异行为，需要进一步研究。

---

## 2. Adapting Rules of Official International Mahjong for Online Players

**arXiv ID:** 2601.08211 | [PDF](https://arxiv.org/pdf/2601.08211v1)

**作者:** Chucai Wang `[一作]` (Peking University), Wenxin Li `[通讯]` (Peking University)

**通讯引用:** 3154 | [OpenAlex ID](https://openalex.org/A5100397213)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

分析官方国际麻将在单局在线环境中的平衡性，提出补偿机制消除先手优势并根据AI自对弈数据调整子目标得分，随后在Botzone实现并公开新的麻将版本。

**💡 创新点**

首次利用比赛冠军AI的海量自对弈数据来量化评估游戏平衡，基于统计结果提出可直接在规则中嵌入的补偿机制与子目标分数调整方案，避免传统的轮换位置或经验设定。

**🔧 技术方法**

使用深度强化学习训练出的麻将冠军AI进行自对弈，收集并统计胜率与得分数据；应用博弈论关于先手优势的理论分析；设计补偿点机制与分数调整算法；实现在线游戏后端。

**📊 数据集**

IJCAI 2022麻将AI竞赛冠军AI的557,056局自对弈数据（以及其他几名AI的部分自对弈），用于统计先手优势与子目标出现频率。

**📈 对比分析**

通过对比原始规则下的位次轮换实验（24 × 1024局）与固定位次实验，验证补偿机制能恢复排名一致性；对比不同AI在自对弈中的子目标频率，证明调整方案的普适性；实验表明补偿后排名与原始复位相符，平均得分差距显著下降。

**⚠️ 局限性**

仅针对单局在线玩法，补偿参数和分数调整是基于特定AI的数据，可能对人类玩家或其他AI存在偏差；子目标分数的调整仍基于频率与理论组合数，未考虑玩家策略多样性及游戏体验的全面评估。

---

## 3. ForgetMark: Stealthy Fingerprint Embedding via Targeted Unlearning in Language Models

**arXiv ID:** 2601.08189 | [PDF](https://arxiv.org/pdf/2601.08189v1)

**作者:** Zhenhua Xu `[一作]`, Meng Han `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种基于有针对性去学习的隐蔽模型指纹化框架 ForgetMark，使用 LoRA 适配器在预训练 LLM 上抑制选定键值对的概率，形成可在黑/灰盒访问下验证的指纹。

**💡 创新点**

创新点包括：① 用预测熵筛选低不确定性的自然键值对，构造低 perplexity 的键；② 通过概率遗忘痕迹而非固定触发-响应实现隐蔽指纹；③ 引入统一的概率/语义指纹成功率（FSR）指标；④ 在模型融合和增量微调场景下评估鲁棒性。

**🔧 技术方法**

使用技术包括 LoRA 参数高效适配、基于预测熵的键选择、目标模型的有针对性去学习（sign‑likelihood 目标）、概率与语义指纹成功率评估、PPL 与 Token‑Forcing（TF）检测等。

**📊 数据集**

使用的数据集包括：GPT‑4 辅助生成键值对、Alpaca 作为保留集、3 个 7–8B LLM（Mistral‑7B‑v0.3、LLaMA‑3‑8B、Qwen2.5‑7B），以及 18 个 zero‑shot 评估基准（ANLI、ARC、OpenBookQA、Winogrande、LogiQA、SciQ、BoolQ、CB、RTE、WiC、WSC、CoPA、MultiRC、LAMBADA 等）。

**📈 对比分析**

与 IF‑SFT 与 Chain&Hash 基准对比，使用灰盒 FSR_prb（基于概率）和黑盒 FSR_rouge（基于语义）评估。ForgetMark 在指纹验证率上达到 100%，PPL 低至 55.6/26.27，TF 检测率为 0%，并在模型融合（各种混合比例下）保持 100% 的 FSR，且对原始模型性能影响极小。

**⚠️ 局限性**

主要局限：在持续的增量微调下，遗忘痕迹会逐步恢复，导致 FSR 降低；对同族模型之间的指纹迁移可靠性尚未系统验证。

---

## 4. Representations of Text and Images Align From Layer One

**arXiv ID:** 2601.08017 | [PDF](https://arxiv.org/pdf/2601.08017v1)

**作者:** Evžen Wybitul `[一作]` (ETH Zurich), Stanislav Fort `[通讯]` (Aisle Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在视觉‑语言模型的不同层上，用概念向量驱动的图像合成方法，直接检验并证明文本与图像表示在早期层已显著对齐。

**💡 创新点**

提出一种无需外部数据集或辅助模型的基于直接上升合成（DAS）的概念向量合成技术，首次在层级层面提供概念级对齐的直接证据，并发现大部分概念在层 1 即可对齐。

**🔧 技术方法**

采用概念向量提取、加权补丁聚合、直接上升合成、多分辨率扰动、随机增广以及 GPT‑5 识别评估等技术。

**📊 数据集**

在 Gemma 3 4B 上对 100+ 个概念（动物、季节、活动等）进行实验，并在 InternVL 3 8B 上做少量验证；整个实验完全不使用任何外部数据集，合成过程完全自给自足。

**📈 对比分析**

与传统代理指标（重构误差、稀疏自编码器等）相比，直接合成能在层级层面提供概念级对齐的可视化证明；实验表明在层 1 对动物、季节、活动等概念的识别率可达 50–70%，显示早期对齐显著，而中间层出现明显的对齐崩溃。

**⚠️ 局限性**

局限性包括：合成图像在某些类别下仅模型可识别、人工识别率低；方法对层级和模型结构高度依赖，需要精细的超参数调优；目前主要验证在 Gemma 3，未能在更广泛的 VLM 上复现；优化失败可能导致对齐缺失。

---

## 5. Robust Subpixel Localization of Diagonal Markers in Large-Scale Navigation via Multi-Layer Screening and Adaptive Matching

**arXiv ID:** 2601.08161 | [PDF](https://arxiv.org/pdf/2601.08161v1)

**作者:** Jing Tao `[一作]` (National University of Defense Technology), Qifeng Yu `[通讯]` (National University of Defense Technology)

**通讯引用:** 2796 | [OpenAlex ID](https://openalex.org/A5103187819)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种针对大范围飞行导航中复杂背景干扰导致定位失败的三层次斜角标记定位方法。

**💡 创新点**

创新点在于将多层角点筛选与自适应模板匹配相结合，消除粗定位中的模板匹配冗余，通过几何与光度耦合自动生成模板并用二次曲面极值实现亚像素定位。

**🔧 技术方法**

技术手段包括GDWGIF降噪、局部伽马校正、结构信息提炼、曲率密度自适应筛选、LSD直线检测、基于角度的自适应模板生成，以及矩阵重塑加速的NCC与二次拟合定位。

**📊 数据集**

实验数据来自实验室构造的多形态斜角标记数据集和多孔径无人机平台的实地航拍图像（4096×3000分辨率）。

**📈 对比分析**

与Zhang的棋盘角点检测、RSFEM和人工提取相比，平均误差<0.03像素，97%以上的角点误差<0.1像素；飞行平台定位精度达到厘米级空间误差、角度误差<1弧分；计算时间波动率低于12%，与对比方法相比平均耗时降低约38%。

**⚠️ 局限性**

局限性包括对斜角标记的专用性、仍需CUDA/CPU优化提升实时性、数据集不公开、在极端光照或遮挡情况下可能表现下降。

---

## 6. Where Does Vision Meet Language? Understanding and Refining Visual Fusion in MLLMs via Contrastive Attention

**arXiv ID:** 2601.08151 | [PDF](https://arxiv.org/pdf/2601.08151v1)

**作者:** Shezheng Song `[一作]`, Jie Yu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提供了关于可重复性评估的关键方面的指导，要求作者在文档中直接编辑并回答相关问题。

**💡 创新点**

创新点在于系统化地列出了可重复性检查清单，确保研究结果的透明性和可验证性。

**🔧 技术方法**

使用了理论贡献和实验设计的框架，强调了对数据集和实验代码的详细描述。

**📊 数据集**

使用了多个数据集，并提供了对选择这些数据集进行实验的动机。

**📈 对比分析**

通过详细的实验设计和参数设置，确保了结果的可靠性，但未提供单维性能总结的分析。

**⚠️ 局限性**

限制在于未对性能改进的显著性进行适当的统计测试，可能影响结果的解释。

---

## 7. Sherry: Hardware-Efficient 1.25-Bit Ternary Quantization via Fine-grained Sparsification

**arXiv ID:** 2601.07892 | [PDF](https://arxiv.org/pdf/2601.07892v1)

**作者:** Hong Huang `[一作]` (City University of Hong Kong), Dapeng Wu `[通讯]` (City University of Hong Kong)

**通讯引用:** 20173 | [OpenAlex ID](https://openalex.org/A5001469325)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于3:4稀疏结构的三值量化框架（Sherry），并通过Arenas机制解决稀疏训练中的权重陷阱问题，实现在边缘设备上高效推理。

**💡 创新点**

创新点包括：①将3:4稀疏模式与5位打包结合，达到1.25比特宽度并保持SIMD对齐；②设计Arenas残差门控的退火梯度注入，缓解稀疏训练中的梯度均质化和表达崩塌；③在量化后端使用LUT加速实现。

**🔧 技术方法**

技术方法包括：三值量化（{-1,0,+1}）与按通道/组的缩放因子；3:4结构稀疏约束和贪婪Sparse-AbsMean求解；Arenas残差门控的退火调度；LUT基线推理引擎；量化感知训练（QAT）和STE近似。

**📊 数据集**

数据集：使用LLaMA‑3.2‑1B/3B模型，在10B个UltraFineWeb文本上进行QAT；在PIQA、ARC‑Easy、ARC‑Challenge、HellaSwag、WinoGrande等5个零样本任务上评估。

**📈 对比分析**

与1.67‑bit和2‑bit三值基线（Tequila、TernaryLLM、DLT等）以及BF16对比；在1B模型上平均准确率与SOTA齐平，3B模型也保持第二名；在Intel i7‑14700HX CPU上实现1.25‑bit时模型尺寸减少25%，推理速度提升10‑18%。

**⚠️ 局限性**

局限性包括：仅验证至3B规模，未探究更大模型；侧重通用SIMD打包，未针对服务器专用硬件优化；仅做权重量化，激活/KV仍为BF16；训练阶段需额外算力，Arenas引入额外计算开销。

---

## 8. Relational Knowledge Distillation Using Fine-tuned Function Vectors

**arXiv ID:** 2601.08169 | [PDF](https://arxiv.org/pdf/2601.08169v1)

**作者:** Andrea Kang `[一作]` (University of California, Los Angeles), Hongjing Lu `[通讯]` (University of California, Los Angeles)

**通讯引用:** 3341 | [OpenAlex ID](https://openalex.org/A5072587042)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出通过对函数向量进行微调和线性组合，提取并利用大型语言模型中的关系知识，以提升关系推断和类比推理的性能。

**💡 创新点**

创新之处在于将因果调解分析得到的函数向量在极小样本下微调，并构造复合函数向量实现跨关系类比，突破了传统激活调度的局限。

**🔧 技术方法**

采用因果调解分析、激活调度、函数向量微调（交叉熵+L2正则）和仿射变换等技术。

**📊 数据集**

评估使用118条基线关系（simple-task、SemEval-2012、Google、MSR等）以及复杂任务关系，并在Green、BATS、SAT等类比数据集上测试。

**📈 对比分析**

与原始模型、初始函数向量、ActAdd等基线相比，微调函数向量在零样本任务中提升20–60%，在远程类比中提升19–20%，在人类相似度判定上相关系数提升至0.83。

**⚠️ 局限性**

限制在于仅处理四项类比形式，未覆盖更复杂的叙事类比或层级关系，需要进一步扩展。

---

## 9. Bridging the Trust Gap: Clinician-Validated Hybrid Explainable AI for Maternal Health Risk Assessment in Bangladesh

**arXiv ID:** 2601.07866 | [PDF](https://arxiv.org/pdf/2601.07866v1)

**作者:** Farjana Yesmin `[一作]` (Independent Researcher), Suraiya Shabnam Bristy `[通讯]` (Sonargaon Sheba General Hospital)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并验证了结合模糊推理与SHAP的混合可解释AI框架，用于孟加拉国孕产妇风险评估。

**💡 创新点**

将先验模糊规则与后验SHAP解释相结合，形成可解释且高精度的混合模型，并通过临床专家验证提升信任度。

**🔧 技术方法**

采用模糊推理系统、XGBoost、SHAP与LIME等解释技术。

**📊 数据集**

使用了UCI孕产妇风险数据集（1014例）并加入孟加拉地区医疗可及性分数。

**📈 对比分析**

与六种基线模型对比，混合模型准确率88.67%、ROC‑AUC0.9703，显著优于基线。

**⚠️ 局限性**

样本量仅14名临床医生、缺少产科历史等关键特征、需在更大多中心样本中验证。

---

## 10. Efficient Synthesis for Two-Dimensional Strand Arrays with Row Constraints

**arXiv ID:** 2601.07968 | [PDF](https://arxiv.org/pdf/2601.07968v1)

**作者:** Boaz Moav `[一作]` (Technion), Eitan Yaakobi `[通讯]` (Technion)

**通讯引用:** 3983 | [OpenAlex ID](https://openalex.org/A5021586372)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究在行约束下DNA条形素合成的调度问题，给出两条链行内合成时间的理论期望上限和下限，并提供最优离线调度的动态规划算法。

**💡 创新点**

首次给出针对行约束模型的解析期望完成时间上界与下界；提出无前瞻在线策略“滞后优先”(LF)并证明其在任意字母表上达到最优渐进性能；在二元情况下通过单符号前瞻进一步提升效率；将问题与最长公共子序列（LCS）关联，提供组合学见解。

**🔧 技术方法**

采用马尔可夫链建模、重启奖励法、Wald等式、Foster–Lyapunov漂移分析、Renewal–Reward理论以及动态规划（O(L²q) 复杂度）等概率与组合技术。

**📊 数据集**

使用随机均匀生成的长度为 L 的两条序列作为实验对象，并对理论上限与下限进行数值验证，未使用公开数据集。

**📈 对比分析**

与最优无前瞻策略比较，证明 LF 在任意 q 下达到期望完成时间的渐进上限 L(q+3)/2；在二元情况下，单符号前瞻策略实现 7L/3 的期望完成时间，优于 2.5L 的无前瞻上界；对比了动态规划得到的最优时间，验证了理论界限的紧密性。

**⚠️ 局限性**

局限在于仅分析两条链的单行情况；对更大 m×k 数组、复杂约束（如光学耦合）以及更广泛的字母表仍需进一步研究；理论上限与下限虽然渐进精确，但在小 L 下误差可能显著。

---

## 11. Mechanisms are Transferable: Data-Efficient Low-Resource Adaptation via Circuit-Targeted Supervised Fine-Tuning

**arXiv ID:** 2601.08146 | [PDF](https://arxiv.org/pdf/2601.08146v1)

**作者:** Khumaisa Nur'aini `[一作]` (Monash University), Derry Wijaya `[通讯]` (Monash University)

**通讯引用:** 1580 | [OpenAlex ID](https://openalex.org/A5027174211)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于机制识别的低资源语言适配方法——Circuit-Targeted Supervised Fine‑Tuning (CT‑SFT)，通过先在高资源源语言上学习任务机制，然后在目标语言上只更新识别出的关键注意力头与 LayerNorm，极大减少参数更新并降低灾难性遗忘。

**💡 创新点**

创新点包括：1）将 Contextual Decomposition for Transformers (CD‑T) 适配到无模板、低资源文本环境，使用标签平衡均值与方向性相关性评分；2）利用机制驱动的稀疏微调实现“编辑‑保持”权衡，依据源‑目标匹配度选择更新或保持机制；3）展示该方法在跨语言转移中既提升准确率又显著抑制源语言遗忘。

**🔧 技术方法**

核心技术：CD‑T 机制分解、标签平衡均值构造、方向性相关性评分、稀疏头级梯度遮蔽、LayerNorm 仅更新、机制导向的两阶段微调。

**📊 数据集**

使用的公开数据集包括：NusaX‑Senti（印尼多语种情感分类）和 XNLI（多语言自然语言推理）。

**📈 对比分析**

与完整微调、随机稀疏更新、最不相关头更新等基线比较，CT‑SFT 在 NusaX‑Senti 和 XNLI 上在极小参数预算（<1%）下均取得或超过完整微调的交叉语言准确率，并且在源语言上保持 90%+ 的性能；同时显著减少灾难性遗忘。

**⚠️ 局限性**

局限性：1）机制发现依赖源语言的任务熟练度，若源任务表现弱，CT‑SFT 可能不稳定；2）仅在单模型（Qwen2.5‑0.5B）和单标记任务上验证，未知在更大模型或多标记生成任务上的适用性；3）仅通过源语言精度评估遗忘，未覆盖分布漂移、鲁棒性等维度。

---

## 12. Contact-aware Path Planning for Autonomous Neuroendovascular Navigation

**arXiv ID:** 2601.07945 | [PDF](https://arxiv.org/pdf/2601.07945v1)

**作者:** Aabha Tamhankar `[一作]` (Worcester Polytechnic Institute), Giovanni Pittiglio `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 858 | [OpenAlex ID](https://openalex.org/A5049921916)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种确定性、时间高效的触感感知路径规划器，用于自主神经血管导航；

**💡 创新点**

创新点在于利用预先定义的运动原语（壁导滑移、自由空间反弹、导管发射）将复杂的受力约束简化为几何运动，从而实现快速规划；

**🔧 技术方法**

采用基于预手术 CTA/MRA 图像的三角网格、统一采样、曲率启发式和逆运动学控制指令等技术；

**📊 数据集**

使用公开的 Aortic Vessel Tree (AVT) CTA 数据集中的三种常见主动脉弓变体（经典、牛型、直接椎动脉起源）进行验证；

**📈 对比分析**

与传统基于迭代优化的规划方法相比，该方法在三种变体中实现了 100% 成功率，最坏情况仅需 22.8 秒，子毫米级跟踪误差（<0.64 mm）；

**⚠️ 局限性**

局限性包括假设血管静态、工具软化、仅考虑静态几何约束，缺乏实时闭环控制，对摩擦、机械间隙等动态效应未建模。

---

## 13. Small Symbols, Big Risks: Exploring Emoticon Semantic Confusion in Large Language Models

**arXiv ID:** 2601.07885 | [PDF](https://arxiv.org/pdf/2601.07885v1)

**作者:** Weipeng Jiang `[一作]` (Xi'an Jiaotong University), Yang Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 81937 | [OpenAlex ID](https://openalex.org/A5100355964)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统性研究了大语言模型在代码生成场景下因ASCII表情符号导致的语义混淆问题，揭示其可引发误执行的安全风险。

**💡 创新点**

创新点在于首次定义并量化“表情符号语义混淆”这一漏洞，构建可扩展的自动化测试管道并公开了3,757条多语言、多情境测试案例。

**🔧 技术方法**

使用了基于LLM的prompt生成、符号重叠筛选、上下文复杂度控制等技术，结合正则表达式和手工验证进行误判检测。

**📊 数据集**

使用公开的约62,000条表情符号集合与四种主流编程语言（Bash/Shell、Python、SQL、JavaScript）的符号集进行交叉过滤，最终得到21个元场景、3种上下文复杂度下的测试集。

**📈 对比分析**

通过在六个SOTA LLM（Claude、Gemini、GPT‑4.1‑Mini、DeepSeek、Qwen3‑Coder、GLM）上多次复测，得到平均混淆率超过38%，其中>90%为“silent failure”，说明该漏洞普遍且严重。

**⚠️ 局限性**

限制主要包括：数据集覆盖面有限（无法囊括所有真实交互边缘情况），以及实验仅评估了六款主流模型，未来需扩展至更多LLM与Agent系统。

---

## 14. Towards Cross-Platform Generalization: Domain Adaptive 3D Detection with Augmentation and Pseudo-Labeling

**arXiv ID:** 2601.08174 | [PDF](https://arxiv.org/pdf/2601.08174v1)

**作者:** Xiyan Feng `[一作]` (Dalian University of Technology), You He `[通讯]` (Shenzhen International Graduate School, Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了一套基于PVRCNN++的跨平台3D目标检测框架，并加入跨平台抖动对齐（CJA）与自训练（ST3D）实现跨平台迁移。

**💡 创新点**

创新点包括：① 用CJA抖动对齐数据增强显式补偿不同平台间点云视角差异；② 通过ST3D自训练迭代生成高质量伪标签实现无监督域自适应；③ 替换CenterHead为AnchorHead提升提议质量。

**🔧 技术方法**

采用的技术包括：PVRCNN++点体素混合网络、CJA旋转数据增强、ST3D伪标签自训练、AnchorHead、Voxel Set Abstraction、RoI‑grid pooling等。

**📊 数据集**

使用数据集为RoboSense2025 Track5，包含车辆、无人机、四足机器人三域的点云-图像对，源域仅有标注。

**📈 对比分析**

与PointRCNN、VoxelRCNN、PDV等基线对比，CJA+ST3D+AnchorHead方案在车辆→四足机器人迁移中将Car AP@0.5从29.44%提升至58.79%，Ped AP@0.5从14.94%提升至49.81%。

**⚠️ 局限性**

局限性：仅对体素或点体素混合模型有效，纯点云模型受益有限；需要大规模GPU资源；对不同平台的视角抖动范围仍需进一步细化。

---

## 15. GI-Bench: A Panoramic Benchmark Revealing the Knowledge-Experience Dissociation of Multimodal Large Language Models in Gastrointestinal Endoscopy Against Clinical Standards

**arXiv ID:** 2601.08183 | [PDF](https://arxiv.org/pdf/2601.08183v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 16. Knowledge-based learning in Text-RAG and Image-RAG

**arXiv ID:** 2601.08226 | [PDF](https://arxiv.org/pdf/2601.08226v1)

**作者:** Alexander Shim `[一作]` (Florida International University), Samuel Clarke `[通讯]` (Florida International University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究分析并比较了基于视觉变换器（EVA-ViT）的多模态图像编码器与LlaMA或ChatGPT大语言模型（LLM）在减少幻觉问题和检测胸部X光图像疾病方面的效果。

**💡 创新点**

创新点在于通过不同的检索增强生成（RAG）策略，评估其在胸部X光图像解读中的表现，特别是文本RAG和图像RAG的比较。

**🔧 技术方法**

使用了视觉变换器（EVA-ViT）、GPT和LlaMA模型，结合KNN方法进行图像检索。

**📊 数据集**

使用了NIH胸部X光图像数据集，共有112,000张图像，经过单标签过滤后，最终使用了84,053张图像进行训练、测试和验证。

**📈 对比分析**

与基线模型相比，文本RAG有效减少了幻觉问题，而图像RAG通过KNN方法提高了预测信心和校准。GPT LLM的表现优于LlaMA模型，幻觉率更低，期望校准误差（ECE）更好。

**⚠️ 局限性**

限制在于数据集不平衡和硬件资源不足，未来需要更平衡的数据集和更强大的硬件以提高模型性能。

---

## 17. Local-Global Feature Fusion for Subject-Independent EEG Emotion Recognition

**arXiv ID:** 2601.08094 | [PDF](https://arxiv.org/pdf/2601.08094v1)

**作者:** Zheng Zhou `[一作]` (Beijing Jiaotong University), Camilo E. Valderrama `[通讯]` (University of Winnipeg)

**通讯引用:** 439 | [OpenAlex ID](https://openalex.org/A5038317537)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种融合局部（通道级）和全局（试验级）特征的EEG情绪识别框架，并在SEED-VII数据集上实现了跨受试者的情绪分类。

**💡 创新点**

创新点包括：①将差分熵与图论连通性特征结合生成局部表示；②构造多尺度时间频域与多分形特征的全局描述；③采用双分支Transformer（MAET）实现局部与全局特征的注意力融合，并通过域对抗正则化提升跨受试者鲁棒性。

**🔧 技术方法**

使用的技术包括：差分熵、图论网络度/聚类/介数/PageRank、时域统计量、频域功率、波形多分形分析、双分支Transformer、梯度反转层、受试者归一化和留一受试者交叉验证。

**📊 数据集**

数据集为SEED-VII，包含20名受试者、7类情绪（快乐、悲伤、恐惧、厌恶、惊讶、愤怒、中性），共80条试验记录。

**📈 对比分析**

与单一局部特征、全局特征以及经典机器学习基线（SVM、LogReg、MLP、RF）比较，双分支Transformer模型在LOSO评估中实现约40.1%准确率、38.7%宏F1，明显优于局部特征（36.4%）和传统基线。

**⚠️ 局限性**

局限性包括：①差分熵分箱可能忽略细粒度频谱信息；②基于单一窗口的相关性矩阵未考虑时间变化；③Transformer处理的是展开向量，未显式保留通道空间关系，可能限制对空间结构的捕捉。

---

## 18. Q-realign: Piggybacking Realignment on Quantization for Safe and Efficient LLM Deployment

**arXiv ID:** 2601.08089 | [PDF](https://arxiv.org/pdf/2601.08089v1)

**作者:** Qitao Tan `[一作]` (University of Georgia), Geng Yuan `[通讯]` (University of Georgia)

**通讯引用:** 1477 | [OpenAlex ID](https://openalex.org/A5007300551)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于后训练量化的安全恢复方法，可在 LLM 微调后无需重新训练即可修复安全偏差。

**💡 创新点**

将量化视为双目标（压缩+安全恢复）；利用激活空间的空间可分离性与 SLR 探针来引导量化，并采用 Softplus 分离损失实现恶意激活的再分离，实现在不影响任务性能的前提下恢复安全性。

**🔧 技术方法**

后训练量化（PTQ，使用 OmniQuant 框架）、SLR 探针、Softplus 损失、LoRA 微调、GPU 内存与时间监测。

**📊 数据集**

基准模型：LLaMA2‑7B‑Chat、Gemma2‑9B、Qwen2.5‑7B；Fine‑tune 数据集：Alpaca、SST‑2、GSM8K；安全评估使用 AdvBench。

**📈 对比分析**

与 PTST、Panacea、Lisa 等基线对比，平均 harmful score 降至 7.64%（比 PTST 低 5.15%），保持或略低于浮点精度微调；在 GPU 费用和显存方面比基线低 1–2h GPU 小时，显存 7–8 GB。

**⚠️ 局限性**

仅验证到 9B 参数模型，尚未扩展到更大模型；目前仅针对单模态语言模型，尚未验证多模态体系。

---

## 19. Efficient Incremental SLAM via Information-Guided and Selective Optimization

**arXiv ID:** 2601.08110 | [PDF](https://arxiv.org/pdf/2601.08110v1)

**作者:** Reza Arablouei `[一作]` (Commonwealth Scientific and Industrial Research Organisation), Reza Arablouei `[通讯]` (Commonwealth Scientific and Industrial Research Organisation)

**通讯引用:** 1540 | [OpenAlex ID](https://openalex.org/A5044185375)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种高效的增量SLAM后端，结合信息导向门限（IGG）和选择性局部优化（SPO）实现批量优化的精度与实时效率的平衡。

**💡 创新点**

创新点在于使用信息量的对数行列式增益来触发全局优化，只有在信息显著提升时才进行；以及在每次迭代中仅对受影响的变量子集执行Gauss‑Newton步骤，动态更新活跃集以减少不必要的计算。

**🔧 技术方法**

主要技术包括稀疏Cholesky因子化、信息矩阵对数行列式计算、活跃集的增删策略、以及基于Bayes树的增量重线性化与分块求解。

**📊 数据集**

实验使用了MIT、FR079、CSAIL、Intel、FRH等常见SLAM基准数据集，并在MIT-P等增设外部位姿先验的数据集上进一步验证。

**📈 对比分析**

与iSAM2、GTSAM等传统增量方法以及全批量求解相比，所提出的GNi‑SPO‑IGG在保持与全批量几乎相同的Nχ²和ATE误差的同时，将求解和更新的FLOPs降低了2~8倍，显著提升了实时性能。

**⚠️ 局限性**

局限性包括需要手工调节信息增益阈值τ_η和更新阈值τ_d，代码目前仅在MATLAB实现；对极端非高斯测量、强噪声或实时嵌入式平台的适应性尚待进一步验证。

---

## 20. Hybrid SARIMA LSTM Model for Local Weather Forecasting: A Residual Learning Approach for Data Driven Meteorological Prediction

**arXiv ID:** 2601.07951 | [PDF](https://arxiv.org/pdf/2601.07951v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 21. Joint Communication and Sensing in RIS-Assisted MIMO System Under Mutual Coupling

**arXiv ID:** 2601.08142 | [PDF](https://arxiv.org/pdf/2601.08142v1)

**作者:** Dilki Wijekoon `[一作]` (University of Manitoba), Ekram Hossain `[通讯]` (University of Manitoba)

**通讯引用:** 33927 | [OpenAlex ID](https://openalex.org/A5089270885)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在单用户单目标的RIS辅助JCAS系统中，联合设计基站的主动波束赋形与RIS的被动相位调制，考虑了RIS元件之间的互耦，并使用Fisher信息和互信息的加权和作为目标函数，求解系统最优参数。

**💡 创新点**

创新点在于：①采用物理一致的互耦模型（基于散射参数而非传统偶极子阻抗模型），②在单波束配制下通过量化处理直接考察自干扰对感知性能的影响，③将感知精度（FI）与通信效率（MI）统一放入加权优化框架，形成Pareto边界分析。

**🔧 技术方法**

使用技术包括：多输入多输出（MIMO）雷达模型、RIS相位矩阵设计、Fisher信息与互信息理论、投影梯度上升法、MATLAB CVX优化工具、量化误差模型及自干扰矩阵建模。

**📊 数据集**

实验使用仿真参数：基站4×4天线、RIS元素数量为64/100/144，波长3GHz，功率30dBm，噪声功率-50dBm，采用多径路径成分15个，雷达目标角度为0°或指定AoD；并未使用公开数据集，而是基于物理仿真模型生成数据。

**📈 对比分析**

通过与传统忽略互耦的RIS模型对比，结果显示：物理一致模型在Fisher信息与互信息上均有显著提升，特别是当RIS元素数量增大或间距减小时，互耦效应带来更大性能改善；自干扰通过量化显著削弱感知精度，说明需进一步抑制。

**⚠️ 局限性**

局限性包括：①仅考虑单用户单目标场景，未涵盖多用户或多目标环境；②仿真基于理想化信道模型，缺乏真实测量验证；③自干扰模型简化为自由空间路径损耗，实际可能更复杂；④算法复杂度较高，尤其在大规模RIS时计算量大。

---

## 22. MPCI-Bench: A Benchmark for Multimodal Pairwise Contextual Integrity Evaluation of Language Model Agents

**arXiv ID:** 2601.08235 | [PDF](https://arxiv.org/pdf/2601.08235v1)

**作者:** Shouju Wang `[一作]` (University of Hawaii at Manoa), Haopeng Zhang `[通讯]` (University of Hawaii at Manoa)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了MPCI-Bench，旨在评估多模态语言模型在代理任务中的情境完整性（隐私行为）;

**💡 创新点**

创新点在于首个多模态、成对（正负）且分三层（Seed/Story/Trace）的情境完整性基准，并采用三原则迭代细化流程保障数据质量;

**🔧 技术方法**

主要技术包括基于LLM的判定与改进循环、上下文完整性理论、代理沙盒执行追踪、以及LLM-as-Judge进行泄露与效用评估;

**📊 数据集**

数据来源于VISPR图像集，并通过自动生成与人工细化产生约2,052对（10个领域）图文情境对;

**📈 对比分析**

通过二分类探测（准确率、精确率、召回率、F1）和行动级泄露/效用指标进行比较，结果显示模型在探测上表现良好但在视觉泄露和隐私-效用权衡上存在显著缺陷，GPT‑5和InternVL3.5‑8B表现相对最佳;

**⚠️ 局限性**

局限性包括CI规范的文化适用性不足、对缓解策略探索有限、过度依赖提示工程、未涉及训练时或架构层面的改进，以及数据集可能存在的偏差.

---

## 23. VoxCog: Towards End-to-End Multilingual Cognitive Impairment Classification through Dialectal Knowledge

**arXiv ID:** 2601.07999 | [PDF](https://arxiv.org/pdf/2601.07999v1)

**作者:** Tiantian Feng `[一作]` (University of Southern California), Shrikanth Narayanan `[通讯]` (University of Southern California)

**通讯引用:** 30339 | [OpenAlex ID](https://openalex.org/A5010028928)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了 VoxCog，一种完全基于原始语音的端到端认知障碍分类框架，利用预训练的方言识别模型作为先验信息。

**💡 创新点**

创新点在于：①将大规模方言知识迁移到认知障碍检测任务；②不依赖 ASR、文本或图像等辅助模态，显著简化流程；③通过方言先验显著提升多语言数据集上的性能。

**🔧 技术方法**

使用技术包括：Whisper‑Large、MMS‑LID‑256 语音基础模型，LoRA 适配与 1D 卷积、DNN 分类器；滑动窗口 15 s 片段切分；聚合片段预测得到个体级结果。

**📊 数据集**

使用的多语言数据集：ADReSS 2020、ADReSSo 2021（英）、VAS（英）、Ivanova（西）、2021‑NCMMSC（中）和 TAUKADIAL‑zh（中）。

**📈 对比分析**

与基线模型（同一基础模型无方言预训练）以及文献中的多模态集成方法对比。VoxCog 在 ADReSS 2020 上达 87.5% 准确率、ADReSSo 2021 上 85.9%；在非英语数据集上亦提升宏 F1 / UAR，整体性能优于多数现有系统。

**⚠️ 局限性**

局限性：录音中包含访谈者语音，可能影响分类；使用固定窗口切分对极短或极长录音的适应性有限；方言预训练模型覆盖范围有限，未覆盖所有方言种类。

---

## 24. EmbeddingRWKV: State-Centric Retrieval with Reusable States

**arXiv ID:** 2601.07861 | [PDF](https://arxiv.org/pdf/2601.07861v1)

**作者:** Haowen Hou `[一作]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy), Jie Yang `[通讯]` (Shenzhen Yuanshi Intelligence Co., Ltd)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于“状态(state)”的检索框架，将检索与重排序统一起来，实现了检索模型与重排序模型共享可重用的状态。

**💡 创新点**

创新点包括：①利用RWKV的矩阵状态实现线性时间与常数空间的检索；②通过领域感知单阶段课程学习，5%数据即可得到高质量状态；③设计仅使用查询tokens即可完成重排序的状态重排序器，显著降低计算量；④对层级状态进行压缩实验，发现大部分性能可由少数层保留。

**🔧 技术方法**

技术手段：RWKV-7线性RNN、矩阵状态、InfoNCE对比学习、二元交叉熵训练、单阶段领域感知课程学习、层级状态压缩、基于状态的重排序器。

**📊 数据集**

数据集：自建开放源代码检索数据集用于训练；MTEB English benchmark、NanoBEIR、MTEB Chinese benchmark 进行评测；与传统Transformer嵌入模型与重排序模型对比。

**📈 对比分析**

与基准的比较：在MTEB上与e5-base、gte-base等模型性能相当甚至更好；在NanoBEIR上，状态重排序器与单塔Transformer重排序器相当；在速度上，对长文档可实现5.4×–44.8×的加速，显著低于FlashAttention-2等Transformer。

**⚠️ 局限性**

局限：仍需两模型并行，额外资源；对极长或细粒度文档的压缩可能导致信息丢失；共享单一状态会导致性能灾难性下降。

---

## 25. CSQL: Mapping Documents into Causal Databases

**arXiv ID:** 2601.08109 | [PDF](https://arxiv.org/pdf/2601.08109v1)

**作者:** Sridhar Mahadevan `[一作]` `[通讯]` (Adobe Research and University of Massachusetts), Sridhar Mahadevan (Adobe Research and University of Massachusetts)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将大量非结构化文本文档自动转换为可用标准 SQL 查询的因果数据库（CDB），并支持因果分析、干预与争议检验。

**💡 创新点**

创新点在于：①通过语言自动诱导因果结构并聚合为关系数据库，摆脱传统手工本体与固定模式；②将因果模型的证据量化为“score_mass”等度量，支持可信度排序、hub 检测、循环识别；③提供基于 SQL 的可扩展因果推理层，兼容 RAG/IE 预处理结果。

**🔧 技术方法**

核心技术包括：大语言模型（LLM）做因果话语编译、局部因果模型（LCM）生成与评分、聚合算法（score_mass、polarity_mass、controversy 指标）、关系模式归一化与哈希映射、Parquet 存储与 DuckDB 等 SQL 引擎执行。

**📊 数据集**

使用了两类数据集：①单篇报刊文章（如《华盛顿邮报》关于人类直立行走与黑巧克力老化的文章）做案例验证；②大规模经济学论文语料库 Testing Causal Claims (TCC)，包含 45,319 篇论文、260,777 条因果边，验证系统在大规模语料下的可扩展性。

**📈 对比分析**

与传统 RAG、知识图谱、信息抽取管道对比，Csql 在因果可信度、hub 检测、循环识别等任务上表现优异；在 TCC 语料上，聚合后边权重呈重尾分布，能快速定位主因果 hub；性能上，数据编译后仅需标准 SQL 查询，查询延迟低、可在大数据分析平台直接使用。

**⚠️ 局限性**

局限性包括：依赖 LLM 的话语质量，因果推理仅基于文本争议而非实验验证；缺乏时间演化建模与版本控制；对复杂因果代数（如精细干预、反事实）支持有限；在极大规模语料下，语料编译阶段仍是瓶颈。

---

## 26. Enriching Semantic Profiles into Knowledge Graph for Recommender Systems Using Large Language Models

**arXiv ID:** 2601.08148 | [PDF](https://arxiv.org/pdf/2601.08148v1)

**作者:** Seokho Ahn `[一作]` (Inha University), Young-Duk Seo `[通讯]` (Inha University)

**通讯引用:** 845 | [OpenAlex ID](https://openalex.org/A5038350029)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 SPiKE 模型，利用一次性生成的 LLM 语义实体描述，将其注入知识图谱进行聚合，并通过 pairwise profile 匹配实现全局传播，从而提升推荐效果。

**💡 创新点**

创新点包括：① 对 profiling 进行四维度（知识库、偏好指示、影响范围、主体）分层评估，确定 LLM 与 KG 的最佳协同方式；② 只在预处理阶段调用 LLM，训练阶段保持高效；③ 引入 profile‑aware KG 聚合与匹配机制，使结构与语义信息在全图范围内有效融合。

**🔧 技术方法**

采用 LLM（如 GPT‑4o 或 Qwen3‑4B）生成文本 profile，使用预训练文本编码器得到向量；通过知识图谱的消息传递/聚合实现实体表示学习；采用 BPR 损失与采样式 pairwise alignment 损失进行训练；最终用点积预测用户‑物品偏好。

**📊 数据集**

在 Amazon Books、Amazon Movies & TV、Yelp 三个真实业务数据集上进行实验，并构建相应的知识图谱。

**📈 对比分析**

与八个基线（BPR‑MF、LightGCN、SGL、LightGCL、KGRec、DiffKG、RLMRec、CoLaKG）进行对比评估；SPiKE 在 Recall@40 / NDCG@40 上平均提升约 1.5–2.0%，尤其在 Movies & TV 领域提升显著，整体稳健性强。

**⚠️ 局限性**

局限性包括：对知识图谱质量敏感，弱 KG 时性能提升有限；LLM 生成的 profile 受 prompt 长度限制，且一次性预处理耗时；未结合更高级的对比学习或自监督策略，存在进一步提升空间。

---

## 27. An Axiomatic Approach to General Intelligence: SANC(E3) -- Self-organizing Active Network of Concepts with Energy E3

**arXiv ID:** 2601.08224 | [PDF](https://arxiv.org/pdf/2601.08224v1)

**作者:** Daesuk Kwon `[一作]` (Hyntel), Won-gi Paeng `[通讯]` (Hyntel)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出自组织概念网络框架 SANC(E_3)，通过能量最小化驱动的 Gestalt 完成来解释通用智能的产生与运行。

**💡 创新点**

创新点在于将系统 tokens、无先验词表自生成、三项能量平衡（重建、结构、更新）以及 Gestalt 完成统一于单一框架，实现感知、推理、生成与行动的内在一致性。

**🔧 技术方法**

采用形式化公理、能量函数分解、相似竞争与置信度稳定机制，并借鉴预测处理、自由能原理与深度网络概念进行理论阐述。

**📊 数据集**

本论文为理论性研究，未使用具体数据集进行实验。

**📈 对比分析**

未给出实验比较，主要通过推导命题、与现有理论的对应关系来展示框架的理论一致性和预期优势。

**⚠️ 局限性**

主要局限在于缺乏可实现的算法与实验验证，能量最小化与资源约束在实际系统中的可实现性和可扩展性仍待评估。

---

## 28. A Human-Centric Pipeline for Aligning Large Language Models with Chinese Medical Ethics

**arXiv ID:** 2601.07954 | [PDF](https://arxiv.org/pdf/2601.07954v1)

**作者:** Haoan Jin `[一作]` (Shanghai Jiao Tong University), Mengyue Wu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1553 | [OpenAlex ID](https://openalex.org/A5109064838)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了医学伦理对齐框架 MedES 与 Guardian‑in‑the‑Loop，利用 260 条权威中医、法律、伦理文件构建场景化评测与对齐流程。

**💡 创新点**

创新点包括：①基于真实法规构建的 12 场景、1278 条结构化规则的动态伦理基准；②自研评估器通过专家标注和 LoRA 微调实现 97% 以上准确率；③将评估器反馈嵌入多轮 SFT 的闭环对齐，提升小模型伦理表现超大模型 10%。

**🔧 技术方法**

使用技术：大规模语言模型（7B 参数）、LoRA 微调、监督对齐、结构化多维评估器、基于规则的自动评测与生成、RAG 等潜在补充方案。

**📊 数据集**

使用数据集：MedES（12 场景、4,004 条主观问答；1,111 条客观问答；1,377 急诊决策问答；3,896 药物安全问答），以及 260 条法规文件构成的知识库。

**📈 对比分析**

通过与 12 大模型（包括 7B、671B、GPT‑4 等）对比，SFT‑5 轮的 7B 模型在主观伦理评测上获得 0.9356 综合分，风险率仅 3.2%，优于同类大模型；在客观任务上虽仍落后于 671B 系列，但提升显著。

**⚠️ 局限性**

局限性：①知识面受限，目标任务知识缺口仍大；②对评估器的偏差依赖；③对极端场景的覆盖不完整，需进一步结合检索增强与动态更新。

---

## 29. A Pin-Array Structure for Gripping and Shape Recognition of Convex and Concave Terrain Profiles

**arXiv ID:** 2601.08143 | [PDF](https://arxiv.org/pdf/2601.08143v1)

**作者:** Takuya Kato `[一作]` (Tohoku University), Kazuya Yoshida `[通讯]` (Tohoku University)

**通讯引用:** 13030 | [OpenAlex ID](https://openalex.org/A5023419492)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并实现了一种能够同时抓取并测量崎岖地形形状的针阵抓手

**💡 创新点**

针阵抓手兼具可抓凸凹两类地形并通过触觉传感实现地形测量的双重功能

**🔧 技术方法**

机械结构设计、弹性针、舵机驱动的锁定机构、压力传感器读取针位高度、力学与摩擦模型

**📊 数据集**

使用人工仿真地形（凸凹地面、坡面）和500g载荷实验样本，没有公开数据集

**📈 对比分析**

与传统八指齿轮抓手对比，抓取力平均提升约5–10 N，形状识别误差标准差约2–3 mm，3D点云平均误差约7.7 mm

**⚠️ 局限性**

受限于针间个体差异导致测量误差、对极端斜率或表面积不足时抓取力波动较大、仅在实验室仿真环境验证

---

## 30. Cross-Cultural Expert-Level Art Critique Evaluation with Vision-Language Models

**arXiv ID:** 2601.07984 | [PDF](https://arxiv.org/pdf/2601.07984v1)

**作者:** Haorui Yu `[一作]`, Qiufeng Yi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个三层（Tier I–III）跨文化艺术批评评价框架，用于衡量视觉语言模型（VLM）对艺术品的 L1–L5 文化深度理解。

**💡 创新点**

创新点包括：①引入 L1–L5 文化层级和三层评价（自动覆盖/风险、单一判定、金标校准）解决判定尺度不一致；②提供跨 6 文化传统的维度列表、评测资产和诊断工具；③通过等距回归校准单一判定得分，实现可比性与可复制性。

**🔧 技术方法**

采用的技术包括：自动指标（Dimension Coverage Ratio、Cultural Semantic Alignment、Critique Depth Score、Linguistic Quality Score）、Claude Opus 4.5 作为单一判定器、等距回归（Tier III）校准、Python 脚本与脚本化 Rubric 进行批判评分。

**📊 数据集**

使用 Vulca‑Bench 数据集，包含 6,695 对文化艺术品与专家双语批评，6 种文化传统（中、西、日、韩、伊斯、印），294 个专家锚点与 450 个人工评分样本（298 训练 / 152 评估）。

**📈 对比分析**

评估 15 个 VLM 在 294 个黄金样本上的表现，采用校准后的 S_II^* 作为主排名指标；模型分布宽度 0.78 分，主导维度为 Alignment 与 Accuracy；发现 Western 样本平均得分高于 Chinese，表明文化偏差存在。

**⚠️ 局限性**

局限性包括：①样本分布不均衡（西方样本多）、②语言仅限中英，可能导致非汉语传统词汇失真；③仅使用单一判定器，受其主观性与偏好影响；④人工标注一致性有限（κ≈0.39）；⑤自动指标与人类评分相关性低，难以单独替代；⑥缺乏对其他文化与媒介（雕塑、书法等）的覆盖。

---

## 31. How Reliable are Confidence Estimators for Large Reasoning Models? A Systematic Benchmark on High-Stakes Domains

**arXiv ID:** 2601.08134 | [PDF](https://arxiv.org/pdf/2601.08134v1)

**作者:** Reza Khanmohammadi `[一作]` (Michigan State University), Mohammad M. Ghassemi `[通讯]` (Michigan State University)

**通讯引用:** 9538 | [OpenAlex ID](https://openalex.org/A5076266282)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了RMCB（Reasoning Model Confidence estimation Benchmark），收集并标注了347,496条大型推理模型（LRM）的长篇推理轨迹，并对十余种基于表示的置信度估计方法进行了大规模系统评估。

**💡 创新点**

创新点在于①构建了首个覆盖医疗、金融、法律、数学等高风险领域的公开LRM置信度评估基准；②系统揭示了置信度估计在判别力与校准度之间的固有权衡；③发现结构感知模型在校准方面优于仅利用隐藏状态的复杂模型。

**🔧 技术方法**

主要技术包括文本编码器、序列模型、图神经网络、基于隐藏状态的两阶段分类器、以及基于token级logits统计的生成不确定性信号；所有方法均为后置置信度估计，未对LRM进行微调。

**📊 数据集**

使用的多源数据集包括：医学（MedQA、MedMCQA）、金融（FinQA、FinQA-测试）、法律（LegalBench）、数学（MATH、GSM8K）、通用推理（MMLU-Pro、BBH、ARC、CommonsenseQA2、LogiQA、OpenBookQA、QuaRTz、ReClor），共计61,951个样本。

**📈 对比分析**

对六款LRM（Phi、Qwen、Mistral、EXAONE等）在上述数据集上评估，十余种方法的性能显示：文本编码器（如ETTIN、ETTIN-HGA）在AUROC上最高（≈0.672），而基于图结构的模型（如GNN-SB-GCN、GNN-SB-GraphSAGE）在ECE上最佳（≈0.148）；无方法能同时在两项指标达到最优，凸显判别力与校准度的取舍。

**⚠️ 局限性**

局限性包括：仅使用chunk级表示（未覆盖完整token级分布）；标注依赖LLM判定，缺乏人工专家审核；未对多语言或一致性方法进行评估；数据仅来自英语模型和六款LRM。

---

## 32. SwiftMem: Fast Agentic Memory via Query-aware Indexing

**arXiv ID:** 2601.08160 | [PDF](https://arxiv.org/pdf/2601.08160v1)

**作者:** Anxin Tian `[一作]` (Huawei), Mingxuan Yuan `[通讯]` (Huawei)

**通讯引用:** 1987 | [OpenAlex ID](https://openalex.org/A5078949174)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一种查询感知的代理式记忆系统 SwiftMem，利用时间与语义索引实现子线性检索。

**💡 创新点**

创新点包括：①时间索引实现 O(log N) 区间查询；②语义 DAG‑Tag 索引将查询路由到相关主题；③嵌入‑标签协同合并提升缓存局部性，整体将检索复杂度从 O(N) 降至 O(log N + log|V| + Dmax)。

**🔧 技术方法**

使用二叉搜索时间树、DAG 语义标签结构、LLM 生成标签、向量索引与协同合并、查询‑标签路由器等技术。

**📊 数据集**

在 LoCoMo 与 LongMemEval_S 两个公开基准上进行实验。

**📈 对比分析**

与 FullContext、RAG‑4096、LangMem、Zep、Mem0、Nemori 等六种基线对比；搜索延迟仅 11 ms，较最快基线提升 47×；总体吞吐 1.289 s，远快于 2.884 s；准确率 LLM 得分 0.704，BLEU‑1 0.467，表现优于大多数基线。

**⚠️ 局限性**

局限性包括：对 LLM 生成标签的依赖、时间索引仅在有时间线索时激活、对极端长会话的压缩仍待完善，以及仅在公开基准上验证，真实多模态或开放域场景的泛化性未知。

---

## 33. Towards Specialized Generalists: A Multi-Task MoE-LoRA Framework for Domain-Specific LLM Adaptation

**arXiv ID:** 2601.07935 | [PDF](https://arxiv.org/pdf/2601.07935v1)

**作者:** Yuxin Yang `[一作]` (Shanghai University), Xiangquan Yang `[通讯]` (East China Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Med-MoE-LoRA，一种将 Mixture‑of‑Experts 与 Low‑Rank Adaptation 结合的参数高效微调框架，用于在保持通用知识的同时将大语言模型适配到医学领域。

**💡 创新点**

创新点包括：① 双路径知识架构（Base Expert 与 Specialist Experts）实现知识分离；② 采用层级不均匀的专家分配，深层堆积更多 LoRA 专家；③ 引入软融合的自适应路由器，弥合不同医学子任务之间的灰区；④ 采用秩级解耦策略，为不同专家分配不同的低秩容量，提升参数利用率。

**🔧 技术方法**

技术手段包括 LoRA、Mixture‑of‑Experts、软融合路由器、秩级解耦、负载平衡损失等。

**📊 数据集**

使用的基准数据集有 PubMedQA、MedQA、MIMIC‑III（Clinical‑Sum）、MMLU 及 GSM8K。

**📈 对比分析**

与全量微调、标准 LoRA、多任务 LoRA、普通 MoE‑LoRA 等基线相比，Med‑MoE‑LoRA 在医学任务上获得最高准确率（如 MedQA +6.9%），并在 MMLU 与 GSM8K 上仅出现 0.5% 与 0.3% 的性能下降，证明在保持通用能力的前提下实现了更好的领域适配。

**⚠️ 局限性**

局限性在于专家拓扑固定，无法动态增删专家；且目前仅处理文本任务，尚未扩展到多模态医学数据。

---

## 34. DNF: Dual-Layer Nested Fingerprinting for Large Language Model Intellectual Property Protection

**arXiv ID:** 2601.08223 | [PDF](https://arxiv.org/pdf/2601.08223v1)

**作者:** Zhenhua Xu `[一作]`, Meng Han `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种双层嵌套指纹（DNF）方法，在黑盒访问下通过在输入中嵌入层级化的风格与语义触发器，实现对大型语言模型的所有权验证。

**💡 创新点**

创新点在于将外层风格约束与内层语义触发器嵌套构成层级触发，既提升了隐蔽性，降低了触发词的困惑度，又通过规则式生成减少了触发泄露风险，并在保持鲁棒性的同时实现了可再生触发。

**🔧 技术方法**

主要技术包括利用LoRA低秩适配器进行指纹注入、构造四子集（联合、风格、语义、正常）触发数据集、实现层级化触发器、以及通过FSR等指标进行验证。

**📊 数据集**

使用的数据集包括CodeXGLUE代码改进集和将UltraChat转化为莎士比亚式文本的语料，构造的触发集有3,000条；在攻击实验中还使用了Databricks Dolly和UltraChat多轮对话集进行增量微调。

**📈 对比分析**

与IF和HashChain相比，在Mistral、LLaMA‑3‑8B‑Instruct和Falcon3‑7B‑Instruct模型上，DNF实现了100%指纹成功率、低困惑度（PPL），在Token‑Forcing检测中为0%，并在模型合并与增量微调中保持了显著的鲁棒性。

**⚠️ 局限性**

局限性在于对跨同源模型迁移的验证不足，且在极端微调或更强的检测策略下仍可能出现失效；此外，实验主要聚焦于黑盒API场景，未评估对内部参数访问的攻击。

---

## 35. STO-RL: Offline RL under Sparse Rewards via LLM-Guided Subgoal Temporal Order

**arXiv ID:** 2601.08107 | [PDF](https://arxiv.org/pdf/2601.08107v1)

**作者:** Chengyang Gu `[一作]` (Hong Kong University of Science and Technology), Yize Chen `[通讯]` (University of Alberta)

**通讯引用:** 1847 | [OpenAlex ID](https://openalex.org/A5080591736)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了 STO-RL 框架，利用大型语言模型（LLM）生成有时间顺序的子目标序列，并通过潜在奖励塑造将稀疏奖励转化为稠密奖励，以实现离线强化学习。

**💡 创新点**

创新点在于：①将 LLM 生成的子目标顺序与潜在奖励函数结合，形成时序感知的奖励塑造；②仅使用离线数据即可完成子目标发现与奖励塑造，避免额外交互；③通过映射函数 h 将每个状态映射到子目标阶段，实现子目标与低层策略的无缝对接。

**🔧 技术方法**

核心技术包括：LLM 任务规划（ChatGPT/其他模型）、子目标序列与映射函数提取、潜在奖励塑造（Φ(s) = -t/T·1/k_t）、离线 RL 训练（IQL）以及对比实验。

**📊 数据集**

使用四个基准任务：离散任务 CliffWalking、FourRoom，以及连续任务 PointMaze‑UMaze、PointMaze‑Medium，全部基于 D4RL 数据集的离线轨迹。

**📈 对比分析**

与 IQL、GC‑BC、HIQL 基线对比，STO‑RL 在离散任务上实现 100% 成功率且收敛更快；在连续任务中成功率提升至 0.68/0.55，且轨迹更短，学习曲线显示收敛速度显著加快。

**⚠️ 局限性**

局限性包括：未对 LLM 生成的子目标进行在线校正，可能导致子目标顺序不完整；策略仅针对单一任务，缺乏跨任务或迁移能力；对更复杂环境的鲁棒性和可解释性仍待验证。

---

## 36. Sesame Plant Segmentation Dataset: A YOLO Formatted Annotated Dataset

**arXiv ID:** 2601.07970 | [PDF](https://arxiv.org/pdf/2601.07970v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 37. CASHEW: Stabilizing Multimodal Reasoning via Iterative Trajectory Aggregation

**arXiv ID:** 2601.08010 | [PDF](https://arxiv.org/pdf/2601.08010v1)

**作者:** Chaoyu Li `[一作]` (Arizona State University), Pooyan Fazli `[通讯]` (Arizona State University)

**通讯引用:** 377 | [OpenAlex ID](https://openalex.org/A5029196810)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Cashew 框架和 Cashew‑RL 变体，通过多轨迹聚合与视觉验证来稳定视觉语言模型的多步推理。

**💡 创新点**

创新点在于：① 采用迭代聚合多条推理轨迹并结合视觉检验消除幻觉；② 在后训练中通过 GSPO 内部化聚合策略；③ 设计混合奖励鼓励最小足够视觉证据和自适应推理长度。

**🔧 技术方法**

使用了视觉语言模型推理、Grounding DINO 视觉验证、采样多轨迹、迭代聚合、强化学习（Group Sequence Policy Optimization, GSPO）以及自定义奖励机制。

**📊 数据集**

使用了多种图像/视频基准数据集：ScienceQA、MME、POPE、SEED‑Bench、Video‑MME、LongVideoBench、EgoSchema、MVBench、NExT‑QA、VideoMMMU、VSI‑Bench、Video‑TT、TOMATO 等，并构建了聚合格式训练集。

**📈 对比分析**

与 Self‑Consistency、Self‑Selector、Self‑Synthesizer 等测试时缩放方法对比，Cashew 在大多数基准上均领先；例如在 ScienceQA 提升 +23.6%，在 EgoSchema 提升 +8.1%，Cashew‑RL 在 T=3 时进一步提升。

**⚠️ 局限性**

主要局限在于对底层视觉感知模型的准确性依赖较大；若检测错误会影响聚合结果；感知与聚合的耦合尚不紧密，未来需探索更鲁棒的感知或联合优化方案。

---

## 38. Transformer-Based Approach for Automated Functional Group Replacement in Chemical Compounds

**arXiv ID:** 2601.07930 | [PDF](https://arxiv.org/pdf/2601.07930v1)

**作者:** Bo Pan `[一作]` (Emory University), Liang Zhao `[通讯]` (Emory University)

**通讯引用:** 6429 | [OpenAlex ID](https://openalex.org/A5061568038)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段Transformer模型，实现对化合物功能基的移除与替换，并通过SMIRKS序列进行自动化生成。

**💡 创新点**

创新点在于：①采用两阶段生成保证仅在功能基层面修改结构；②利用SMIRKS编码直接输出功能基变换；③支持模型建议与用户指定的功能基替换，提升灵活性。

**🔧 技术方法**

使用了Encoder‑Decoder Transformer（ChemT5预训练模型）、SMIRKS表示、Beam Search、Teacher Forcing等技术。

**📊 数据集**

数据集为从ChEMBL通过MMPDB筛选得到的约200万条匹配分子对（MMP），每条记录包含源分子和对应的SMIRKS变换。

**📈 对比分析**

与传统Mol2Mol（直接翻译分子）进行对比；在%Valid、%Exist、覆盖率等指标上，Mol2Trans在大k时能保持较高的合法性并覆盖更多已知化合物；在k增大时还能产生更多新颖化合物，表现出良好的探索与保留平衡。

**⚠️ 局限性**

局限性：①训练样本规模有限（2M vs 20B可能更佳）；②不支持属性驱动的优化；③当前仅处理功能基级别变换，缺乏对分子整体属性的约束。

---

## 39. The Role of Noisy Data in Improving CNN Robustness for Image Classification

**arXiv ID:** 2601.08043 | [PDF](https://arxiv.org/pdf/2601.08043v1)

**作者:** Oscar H. Ramírez-Agudelo `[一作]` (German Aerospace Center), Michael Karl `[通讯]` (German Aerospace Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在CIFAR-10训练集中引入不同噪声（高斯噪声、盐噪声、Gaussian blur），并评估其对CNN（ResNet-18）在干净与受噪声测试集上的分类性能影响。

**💡 创新点**

发现仅 5–10% 的噪声样本即可显著提升模型在噪声测试集上的准确率，而对干净集影响极小，从而验证了噪声注入可作为一种简单有效的正则化手段。

**🔧 技术方法**

采用 ResNet‑18 网络，使用 SGD 训练，随机加入三类噪声，进行 10 次实验并统计平均值与标准差；同时在多种噪声强度与污染比例下评估性能。

**📊 数据集**

使用公开数据集 CIFAR‑10（60,000 张 32×32 彩色图像，10 类）。

**📈 对比分析**

通过在干净测试集和不同噪声等级测试集上计算 Top‑1 准确率和损失进行对比，结果显示噪声训练可将受噪声测试集的准确率提升约 5–10%，同时干净集准确率仅下降 ≤ 1%。

**⚠️ 局限性**

局限性在于仅考虑了三种噪声类型、单一模型架构、仅针对图像分类任务；未验证更复杂噪声、不同数据集或模型的普适性，噪声比例最优值可能随任务变化。

---

## 40. Semantic Gravity Wells: Why Negative Constraints Backfire

**arXiv ID:** 2601.08070 | [PDF](https://arxiv.org/pdf/2601.08070v1)

**作者:** Shailesh Rana `[一作]` `[通讯]` (Independent Researcher), Shailesh Rana (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在负面约束（如“不要使用单词X”）下的失败机制，并提供全面的机制化分析。

**💡 创新点**

提出语义压力概念并证明其与违反概率呈Logistic关系；揭示priming和override两种失败模式，并通过激活补丁定位导致失败的第23–27层。

**🔧 技术方法**

使用语义压力量化、logit lens、注意力分析、激活补丁、层级贡献分解等技术。

**📊 数据集**

构造了2500条单词补全提示，覆盖5类语义（成语、事实、常识、创意、离散），并在负面约束下共生成40,000次采样。

**📈 对比分析**

通过比较正负约束下概率变化、抑制幅度、注意力分配和层级贡献，发现抑制在成功与失败间相差4.4倍；低压下成功率高，压强越大违规率越高，最高超过46%。

**⚠️ 局限性**

仅在单一模型Qwen2.5‑7B‑Instruct上验证；仅单词输出；未探究多词、长文本约束；注意力解释仍是路由信号，补丁粒度为完整残差，缺乏更细粒度的因果定位。

---

## 41. Enhancing Large Language Models for Time-Series Forecasting via Vector-Injected In-Context Learning

**arXiv ID:** 2601.07903 | [PDF](https://arxiv.org/pdf/2601.07903v1)

**作者:** Jianqi Zhang `[一作]` (Institute of Software, Chinese Academy of Sciences), Changwen Zheng `[通讯]` (Institute of Software, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过将多条时间序列示例压缩成一个上下文向量，并在冻结的大模型内部的残差流中注入该向量，提出了一种名为LVICL的向量注入式上下文学习方法，以提升时间序列预测性能。

**💡 创新点**

创新点在于：①使用可学习的上下文向量适配器自适应提取并压缩多示例信息；②将该向量聚合为无序对例子顺序不敏感的上下文向量；③将向量注入到模型每一层的残差流，充分激活LLM的内在学习能力，同时保持LLM参数不变，显著降低算力开销。

**🔧 技术方法**

技术手段包括：大语言模型（LLaMA‑7B）作为主体；基于自注意力残差流的向量注入；上下文向量的平均聚合与全连接适配器；自回归预测框架；与LoRA、全微调等对比实验。

**📊 数据集**

实验数据集涵盖长时序预测（ECL、ETT（h1,h2,m1,m2）、Traffic、Weather）与短时序预测（M4）以及零样本迁移（M3/M4）等七个公开数据集。

**📈 对比分析**

与全微调、LoRA、传统ICL以及多种基准TS方法（AutoTimes、FPT、TimesNet、PatchTST、Fedformer、iTransformer）对比，LVICL在大多数指标（MSE/MAE/SMAPE/MASE/OWA）上接近或超过全微调、显著优于LoRA，且显著提升了传统ICL的稳定性。

**⚠️ 局限性**

局限性在于：①仍略逊于完整微调的预测精度；②对示例的选择与数量仍有一定影响（虽已通过聚合降低敏感性但未完全消除）；③在构造上下文向量时需额外前向推理，虽不影响推理速度但增加训练前的预处理成本；④目前主要在已知规模的LLM（如7B）验证，未充分探究更大规模模型的扩展性。

---

## 42. Beyond the Next Port: A Multi-Task Transformer for Forecasting Future Voyage Segment Durations

**arXiv ID:** 2601.08013 | [PDF](https://arxiv.org/pdf/2601.08013v1)

**作者:** Nairui Liu `[一作]` (Tsinghua University), Xindi Tang `[通讯]` (Central University of Finance and Economics)

**通讯引用:** 405 | [OpenAlex ID](https://openalex.org/A5074338619)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出多任务Transformer模型，对未来航段的航行时长和港口拥堵状态进行联合预测。

**💡 创新点**

将ETA任务转化为航段级时间序列预测，结合历史航程序列与港口拥堵代理，使用因果遮蔽Transformer和多任务学习提升长周期预测准确性。

**🔧 技术方法**

采用因果遮蔽多头注意力的Transformer架构，提前融合航程、船舶、段落和时序特征，并使用多任务输出层预测航程时长和港口船舶计数。

**📊 数据集**

使用2021年全球AIS航行记录（约2亿条）、端口边界与船舶静态特征，筛选5008艘集装箱船和1120条航段的数据集。

**📈 对比分析**

与LightGBM、CatBoost、XGBoost、Stacking、LSTM、iTransformer等基线对比，模型在加权MAE 6.08h、MAPE 18.05%优于LSTM（6.39h/18.99%）和iTransformer（6.52h/20.70%），在不同航段长度和频次分组均保持领先。

**⚠️ 局限性**

模型仍受低频航段数据稀疏影响，未给出不确定性量化，且未整合天气、海流等外部环境变量。

---

## 43. Max-Min Neural Network Operators For Approximation of Multivariate Functions

**arXiv ID:** 2601.07886 | [PDF](https://arxiv.org/pdf/2601.07886v1)

**作者:** Abhishek Yadav `[一作]` (Indian Institute of Technology), Feng Dai `[通讯]` (University of Alberta)

**通讯引用:** 8139 | [OpenAlex ID](https://openalex.org/A5069790626)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

提出并分析了多元max‑min神经网络算子及其逼近性质，用sigmoidal激活函数构造了新型的非线性逼近算子，给出收敛性与误差上界。

**💡 创新点**

创新点在于将max‑min半环结构推广到多元情形，并证明其伪线性、子加性等性质；同时给出以模量连续性和多元绝对矩为基础的误差估计，展示该算子在常数项逼近上优于传统线性算子。

**🔧 技术方法**

利用sigmoidal函数的微分与凹凸性、max/min 运算的半环理论、模量连续性、绝对矩估计以及逼近算子理论中的quasi‑interpolation技术。

**📊 数据集**

使用人工合成函数 h(y₁,y₂)=y₁²+y₂²/2 在 [0,1]² 上进行数值实验，未使用公开数据集。

**📈 对比分析**

通过对比经典神经网络算子 F_n、max‑product 算子 F_n^(M) 与新构造的 max‑min 算子 ℒ_n 的误差表格，显示 ℒ_n 的误差随 n 增大显著下降，优于经典算子但略逊于 max‑product 算子，计算复杂度与前两者相当。

**⚠️ 局限性**

主要限制包括：对激活函数需满足三条严格假设（单调、C²、指数衰减），且误差阶数与传统算子相同，只在常数上有改进；实验仅在低维合成函数上验证，未探讨高维或真实数据场景。

---

## 44. Reverse Flow Matching: A Unified Framework for Online Reinforcement Learning with Diffusion and Flow Policies

**arXiv ID:** 2601.08136 | [PDF](https://arxiv.org/pdf/2601.08136v1)

**作者:** Zeyang Li `[一作]` (Massachusetts Institute of Technology), Navid Azizan `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 399 | [OpenAlex ID](https://openalex.org/A5005748450)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种逆流匹配（Reverse Flow Matching, RFM）框架，用于在在线强化学习中训练扩散和流策略，使其能够采样未归一化的Boltzmann分布。

**💡 创新点**

创新点在于将先前的噪声期望与梯度期望方法统一为后验均值估计问题，并通过引入Langevin Stein算子构造控制变量，得到可调节方差的最优估计器，从而实现更高效、更稳定的政策训练。

**🔧 技术方法**

核心技术包括逆流匹配目标函数、后验均值估计（自归一化重要采样），Langevin Stein控制变量，控制变量参数自适应优化，以及将流模型与强化学习价值函数结合的在线更新。

**📊 数据集**

实验采用DeepMind Control Suite中的八个连续控制任务（如CartPole、Walker、Humanoid等）进行评估。

**📈 对比分析**

与SAC、Q-score Matching、Q-weighted Noise Estimation和Diffusion Q-sampling四种基线进行对比，RFM在所有任务上实现了更高的累计奖励且收敛更稳定，显著优于现有方法。

**⚠️ 局限性**

局限性包括：需要显式估计后验均值，控制变量参数需通过额外优化；算法对采样提议分布的选择敏感；目前实验仅验证流策略，在更复杂或高维分布下的泛化尚待进一步研究。

---

## 45. Operator learning for models of tear film breakup

**arXiv ID:** 2601.08001 | [PDF](https://arxiv.org/pdf/2601.08001v1)

**作者:** Qinying Chen `[一作]` (University of Delaware), Tobin A. Driscoll `[通讯]` (University of Delaware)

**通讯引用:** 5485 | [OpenAlex ID](https://openalex.org/A5062904806)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用神经算子框架，直接将荧光图像时间序列映射到泪膜厚度和渗透压，替代传统耗时的逆问题求解。

**💡 创新点**

提出将 ODE 与 1D PDE 泪膜破裂模型的模拟结果作为训练数据，利用算子学习实现从单点荧光观测快速预测关键生理量，突破了计算成本与可实时分析的瓶颈。

**🔧 技术方法**

采用三种算子学习技术：Fourier Feature Network（FFN）、Dense-PCA 与 Dense-PCAX（在 PCA 之后加入外部参数），并使用全连接网络与正则化。

**📊 数据集**

训练数据来源于六维参数空间下的 ODE 与 PDE 模型模拟（≈60K 与 29K 样本），并使用 467 条实验荧光时间序列做外部验证。

**📈 对比分析**

在合成测试集上，各模型均实现 1–2 位数的相对 RMSE；在实验数据上，PDE 训练模型在渗透压预测上与逆问题结果更一致，厚度预测则略逊，但整体误差相对较小。

**⚠️ 局限性**

局限性包括：单点荧光观测导致的可识别性不足、模型依赖性强、未加入物理约束、训练样本未覆盖多位置信息，影响对真实泪膜动态的泛化。

---

## 46. Hierarchical Precision and Recursion for Accelerating Symmetric Linear Solves on MXUs

**arXiv ID:** 2601.08082 | [PDF](https://arxiv.org/pdf/2601.08082v1)

**作者:** Vicki Carrica `[一作]` (Massachusetts Institute of Technology), Alan Edelman `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 18135 | [OpenAlex ID](https://openalex.org/A5029673947)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个在GPU上利用矩阵处理单元（MXU）的多精度递归Cholesky求解器，能够在保持数值稳定性的同时显著提升计算速度。

**💡 创新点**

创新点包括：1) 将递归策略完全延伸到TRSM和SYRK阶段，形成全局递归结构；2) 引入树形混合精度层级与块级量化/反量化，针对不同矩阵块动态分配FP16/FP32/FP64；3) 在Julia中通过多派发、参数化类型实现跨厂商（NVIDIA/AMD）的高层抽象与后端调度。

**🔧 技术方法**

使用的技术有：递归分块算法、层次化混合精度与块量化、Tensor Core/Matrix Core加速、Julia多派发与GPU后端（CUDA.jl、AMDGPU.jl）、cuBLAS/rocBLAS等库的基线调用。

**📊 数据集**

使用的数据集为随机生成的对称正定矩阵（尺寸从几千到65,536），通过在对角线上加上矩阵维数保证正定性。

**📈 对比分析**

通过与cuSOLVER/rocSOLVER FP64基准对比：在NVIDIA H200上，递归FP64 SYRK 14×加速，FP16实现最高27×；整体Cholesky 5.32×加速；在AMD MI300X上，混合精度配置可实现2–5×加速；精度方面保持12–15位有效数字，远优于纯FP16。

**⚠️ 局限性**

局限性包括：1) 量化/反量化仅应用于FP16块，未覆盖所有低精度场景；2) AMD平台缺乏FP16混合乘法实现，导致性能上限；3) 对小矩阵递归深度过大导致开销增加；4) 当前实现仅支持密集对称正定系统，未扩展到稀疏、带状或不定矩阵；5) 未实现多GPU并行与异构硬件的进一步扩展。

---

## 47. Structure Detection for Contextual Reinforcement Learning

**arXiv ID:** 2601.08120 | [PDF](https://arxiv.org/pdf/2601.08120v1)

**作者:** Tianyue Zhou `[一作]` (Massachusetts Institute of Technology), Cathy Wu `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 54119 | [OpenAlex ID](https://openalex.org/A5053761444)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出一种在线结构检测框架 SD-MBTL，结合 M/GP-MBTL 在 Contextual Reinforcement Learning (CRL) 中根据任务空间的真实结构动态选择源任务训练策略，从而显著提升迁移学习效率。

**💡 创新点**

创新点在于：① 将 CMDP 的泛化性能拆解为政策质量、任务难度与任务相异度三部分，并通过在线检测判断是否满足“山形结构”；② 依据检测结果在 GP‑MBTL 与基于聚类的 M‑MBTL 之间切换，实现对不同结构的最优任务选择；③ 设计了两套可扩展的结构检测与决策流程，首次实现了在多维上下文空间中自动匹配迁移策略。

**🔧 技术方法**

主要技术包括：Gaussian Process 回归建模源任务性能、基于距离度量的线性泛化差分假设、聚类（改进的 K‑means）用于山形结构下的源任务选取、在线结构检测的方差与斜率判据、以及强化学习中的 PPO 训练与零射击迁移。

**📊 数据集**

实验数据集涵盖：1）三维合成数据，检验结构满足与否；2）连续控制基准（CartPole、BipedalWalker）；3）交通控制基准（IntersectionZoo）；4）农业管理基准（CyclesGym）。

**📈 对比分析**

与独立训练、多任务训练、随机选择、GP‑MBTL、M‑MBTL 以及近似最优（Myopic Oracle）进行对比。结果显示：在满足山形结构时，M‑MBTL 达到接近 oracle 的表现；当结构不满足时，GP‑MBTL 更强；M/GP‑MBTL 在所有基准上均获得最高的聚合性能，比 GP‑MBTL 提升约 12.5% 近似到 oracle，表明自适应结构检测能够实现“最佳两全”。

**⚠️ 局限性**

局限性包括：① 目前仅支持单一结构检测（山形结构），无法同时识别多重或更复杂的结构；② 结构检测依赖已训练任务的评估结果，在任务评估成本高或上下文维度极大时可能效率低下；③ 所有实验仅在三维上下文空间中验证，尚未证明对更高维场景的可扩展性。

---

## 48. DataScribe: An AI-Native, Policy-Aligned Web Platform for Multi-Objective Materials Design and Discovery

**arXiv ID:** 2601.07966 | [PDF](https://arxiv.org/pdf/2601.07966v1)

**作者:** Divyanshu Singh `[一作]` (Texas A&M University), Vahid Attari `[通讯]` (Texas A&M University)

**通讯引用:** 665 | [OpenAlex ID](https://openalex.org/A5066701381)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一个基于云端、AI-native的材料发现平台DataScribe，集成实验与计算数据，提供FAIR元数据、知识图谱、闭环多目标Bayesian优化等功能。

**💡 创新点**

创新点在于将本体驱动的数据接入、知识图谱构建和多目标Bayesian优化嵌入应用层，实现多模态数据统一、数字孪生与政策约束，并提供可视化工作流与LLM代理。

**🔧 技术方法**

使用了微服务架构、Kubernetes、NGINX、RESTful API、Google Drive、HuggingFace LLM、PyTorch/Scikit-learn、Gaussian Process、多目标多精度Bayesian优化、数字孪生模型等技术。

**📊 数据集**

利用了电化学实验数据、高熵合金(Hybrid alloy)实验数据、来自Materials Project、OQMD、AFLOW、JARVIS等公开数据库以及自有实验/计算混合数据集。

**📈 对比分析**

通过与传统EI/PI/LCB采集、benchmark函数以及HEA强度预测R^2≈0.95等指标进行比较，展示了快速收敛、Pareto前沿生成与超参数调优效果。

**⚠️ 局限性**

局限在于目前尚未直接接入实验仪器与HPC；图谱层仍基于关系数据库，缺乏schema‑on‑read；可视化工作流尚未可执行，部分功能需人工干预；数据类型支持仍有限。

---

## 49. SECite: Analyzing and Summarizing Citations in Software Engineering Literature

**arXiv ID:** 2601.07939 | [PDF](https://arxiv.org/pdf/2601.07939v1)

**作者:** Shireesh Reddy Pyreddy `[一作]` (SUNY Polytechnic Institute), Tarannum Shaila Zaman `[通讯]` (University of Maryland Baltimore County)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个半自动化流程，自动从软件工程论文中提取引用语境，利用无监督情感分析对引用进行正负分类，并基于分类结果通过生成式语言模型生成正面和负面摘要。

**💡 创新点**

将情感分析与生成式摘要深度整合，首次在单一管线中同时完成引用情感分类与情感化摘要生成；使用BERT句子嵌入配合K‑means聚类实现无标签情感判定；通过LLM生成细粒度正负面摘要。

**🔧 技术方法**

技术包括：Selenium + Google Scholar 自动抓取，PyPDF 解析 PDF，NLP 前处理（停用词、数字过滤），BERT 嵌入（Hugging Face），Scikit‑learn K‑means 与 t‑SNE 可视化，Gemini 1.5 Flash / GPT‑4o 进行摘要，LangChain 管道管理。

**📊 数据集**

数据集为 151 篇软件工程论文中的 911 条引用语句，聚焦四篇核心论文（SCMiner、SimRacer、DESCRY、RRF），并对九篇目标论文进行情感和摘要评估。

**📈 对比分析**

使用 Silhouette Score 对聚类效果评估（最大 0.138），用 t‑SNE 可视化聚类分布；摘要质量通过语义相似度（余弦相似度）和人工评分（概念、事实、信息量 1–5）进行比较；结果显示正负面摘要平均相似度 0.64，人工评分 4–5，表明框架能较好捕捉核心观点。

**⚠️ 局限性**

局限性包括：自动提取成功率仅 44%（受 CAPTCHA、PDF 解析错误、格式差异限制），K‑means 对簇数与初始质心敏感，未考虑中性情感，数据集规模小且仅覆盖软件工程领域，难以直接推广到其他学科。

---

## 50. A Preliminary Agentic Framework for Matrix Deflation

**arXiv ID:** 2601.08219 | [PDF](https://arxiv.org/pdf/2601.08219v1)

**作者:** Paimon Goulart `[一作]` (University of California), Evangelos E. Papalexakis `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于大型语言模型与视觉语言模型协作的无阈值矩阵去秩方法。

**💡 创新点**

创新点是将解算器LLM生成秩-1 SVD 更新与评估器VLM的接受/拒绝循环相结合，并通过行列置换提升视觉可解释性，从而消除了传统数值阈值。

**🔧 技术方法**

使用 Gemini 2.5 Flash 生成 SVD 向量、Qwen-2-VL 进行质量检验与停止判定，并采用 ICL 与行列置换（None、Sort、GroupNTeach-Block）技术。

**📊 数据集**

评估数据集包括 8×8 的 Digits、32×32 的 CIFAR‑10 以及 16×16 的合成带噪声矩阵。

**📈 对比分析**

与 NumPy 传统去秩基准比较，RMSE 差距在 1.75 至 50.59 之间，步数基本相当或略低，尤其在无噪声 Synthetic、Digits 与 CIFAR‑10 上表现相近。

**⚠️ 局限性**

局限性在于对大规模矩阵的可扩展性不明、评估过程依赖视觉判定带来主观性，以及仅在公开图像与合成数据上验证，未测试在真实用户日志等更复杂场景下的鲁棒性。

---

## 51. Multiplicative Orthogonal Sequential Editing for Language Models

**arXiv ID:** 2601.07873 | [PDF](https://arxiv.org/pdf/2601.07873v1)

**作者:** Hao-Xiang Xu `[一作]` (University of Science and Technology of China), Jia-Chen Gu `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种多重正交顺序编辑（MOSE）方法，用左乘正交矩阵的方式对大语言模型参数进行知识编辑，从而在保持原模型数值稳定性的前提下，实现对知识的精准更新。

**💡 创新点**

创新点在于：①摒弃传统的加法更新框架，改用正交乘法，理论上不改变矩阵的 Frobenius 范数和条件数，避免数值不稳定导致的编辑性能下降；②构建正交 Procrustes 优化问题，闭式求解正交矩阵，保证编辑操作的高效性和可解释性。

**🔧 技术方法**

技术手段包括：正交矩阵乘法、奇异值分解（SVD）求解正交 Procrustes 问题、层选择算法（依据激活强度挑选最佳编辑层并连同相邻层编辑），以及对编辑后模型进行多维度评估（可靠性、泛化、局部性）。

**📊 数据集**

使用的数据集有：事实性知识集（ZsRE、CounterFact）、概念性知识集（ConceptEdit）、以及下游任务集（自然语言推理、摘要、开放域问答、情感分析）来检验编辑效果和通用性能。

**📈 对比分析**

与六种主流编辑方法（ROME、MEMIT、RECT、EMMET、PRUNE、AlphaEdit）比较，MOSE 在单序列编辑中相较于最佳对手提升约 12.08% 的编辑保留率，且在下游任务中保持 95.73% 的原始性能；在批量序列编辑场景下也表现出更稳健的编辑保留和更高的通用性。时间成本略高（主要是 SVD 计算），但提升显著。

**⚠️ 局限性**

局限性包括：①仅在参数修改范式下测试，未探讨与参数保持方法的结合；②对极大规模模型的可扩展性和训练时开销需要进一步验证；③正交矩阵的维度固定，可能对非线性模型的表达能力有限；④在极端连续编辑次数时，正交变换仍可能出现微小累计误差，需进一步研究。

---

## 52. A New Strategy for Verifying Reach-Avoid Specifications in Neural Feedback Systems

**arXiv ID:** 2601.08065 | [PDF](https://arxiv.org/pdf/2601.08065v1)

**作者:** Samuel I. Akinwande `[一作]` (Stanford University), Clark Barrett `[通讯]` (Stanford University)

**通讯引用:** 9991 | [OpenAlex ID](https://openalex.org/A5026961968)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出针对神经反馈系统的后向可达性分析算法，并将其与前向可达性分析结合形成FaBRe验证框架。

**💡 创新点**

创新性地设计了后向可达集的上下近似方法（包括金字塔搜索、迭代凸包和最大空盒三种策略），以及前后向联合验证策略。

**🔧 技术方法**

采用优化求解、约束优化、黄金分割搜索、采样与MILP等技术实现可达集近似。

**📊 数据集**

未公开具体数据集，论文在基准神经控制系统上进行评估。

**📈 对比分析**

与现有最先进的前向可达性方法对比，展示了在可扩展性上的显著提升，尤其在处理高维控制问题时。

**⚠️ 局限性**

后向方法仍受计算复杂度限制，近似精度可能不足，且实验验证尚待进一步补充。

---

## 53. RewriteNets: End-to-End Trainable String-Rewriting for Generative Sequence Modeling

**arXiv ID:** 2601.07868 | [PDF](https://arxiv.org/pdf/2601.07868v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 54. LJ-Spoof: A Generatively Varied Corpus for Audio Anti-Spoofing and Synthesis Source Tracing

**arXiv ID:** 2601.07958 | [PDF](https://arxiv.org/pdf/2601.07958v1)

**作者:** Surya Subramani `[一作]` (University of Michigan), Hafiz Malik `[通讯]` (University of Michigan)

**通讯引用:** 7460 | [OpenAlex ID](https://openalex.org/A5033599238)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并发布了一套名为LJ‑Spoof的单说话人、生成式多样化数据集，用于声学反欺骗与合成源追踪研究。

**💡 创新点**

通过系统控制训练方式、输入、生成参数、声码器、后处理等多维度生成500+子集，填补现有单说话人数据集在多样性与可复现性方面的空白。

**🔧 技术方法**

使用最新的TTS架构（RNN、Transformer、Diffusion、LLM等）、多种声码器、可调采样温度、步数、速度因子以及神经后处理（re‑vocoding、re‑codec）等技术，并提供版本化的生成协议。

**📊 数据集**

基于LJSpeech 13k真实录音生成34个默认子集，并在此基础上构造30+ TTS家族、500+ 变体子集，总计超过300万条语音样本。

**📈 对比分析**

在等量真实/伪造、不同文本、不同参数、不同后处理等对照实验中评估反欺骗器，实验表明在该数据集上训练的模型能显著降低EER/FRR，证明其对多样化合成攻击的鲁棒性。

**⚠️ 局限性**

局限性包括仅覆盖单一说话人，缺乏多说话人、现场录音、跨语言及多音色场景；部分生成器参数与后处理组合尚未覆盖完整，后续需要进一步扩展。

---

## 55. A brain-inspired information fusion method for enhancing robot GPS outages navigation

**arXiv ID:** 2601.08244 | [PDF](https://arxiv.org/pdf/2601.08244v1)

**作者:** Yaohua Liu `[一作]` (Guangdong Institute of Intelligence Science and Technology), Binkai Ou `[通讯]` (Innovation and Research and Development Department BoardWare Information System Company Ltd)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于脉冲神经网络的GPS/INS融合网络（BGFN），用于在GPS失效时辅助惯性导航。

**💡 创新点**

创新点在于首次将脉冲Transformer与脉冲编码器结合，利用历史IMU序列预测GPS位置增量，并通过融合Kalman滤波实现低能耗、高精度的无GPS导航。

**🔧 技术方法**

使用了脉冲神经网络（LIF神经元）、脉冲Transformer、卷积脉冲编码器、Kalman滤波以及BPTT训练。

**📊 数据集**

数据集包括NaveGo公开数据集（Turin城市道路）和自建移动车平台实测数据。

**📈 对比分析**

与传统单独INS、MLP、AT‑LSTM等方法比较，BGFN在两次GPS中断场景下，东/北方向最大误差分别降低约70%–90%，在实测中误差比传统方法低一至两级，能耗也下降约66%。

**⚠️ 局限性**

局限在于模型训练依赖充足的标记数据，且目前仍未在真正的脑形硬件上完整部署，未来需探索端到端可训练的SNN与硬件实现。

---

## 56. LDLT L-Lipschitz Network Weight Parameterization Initialization

**arXiv ID:** 2601.08253 | [PDF](https://arxiv.org/pdf/2601.08253v1)

**作者:** Marius F. R. Juston `[一作]` (University of Illinois Urbana-Champaign), Ahmet Soylemezoglu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了LDLT层在- Lipschitz 神经网络中的初始化动态，并推导出其对输出方差的精确期望；

**💡 创新点**

创新点在于使用 Wishart 分布的矩阵指数展开与 Zonal 多项式/Isserlis 定理，得到高阶矩的闭式近似，进而得到最佳初始化尺度（约 10/√n）以保持单层方差接近 1；

**🔧 技术方法**

采用 Wishart 分布理论、Zonal 多项式期望、Isserlis（Wick）定理、矩阵拉普拉斯积分、Monte‑Carlo 验证、以及对梯度、Hessian 等训练动态的数值分析；

**📊 数据集**

使用 Higgs 物理学分类数据集（100k 训练样本、20k 测试样本）进行实验；

**📈 对比分析**

与传统 He/Kaiming 初始化以及不同深度、优化器（SGD、AdamW）比较，实验显示在 AdamW 下，较大的初始化尺度能提升方差但并未显著改善验证 AUC；在 SGD 下更大尺度导致性能下降；总体而言，单层方差可通过尺度调节，但对最终性能影响有限；

**⚠️ 局限性**

局限性包括：① 理论上虽可推导方差为 1，但实际网络深度不出现预期的快速信息衰减；② 经验验证表明初始化尺度对训练表现的影响在现代 Adam 优化器中被梯度归一化所抵消；③ 高阶矩的计算仅到 k=10，精度受限；④ 仅在 Higgs 数据集上验证，缺乏更广泛的实证。

---

## 57. Decoder Generates Manufacturable Structures: A Framework for 3D-Printable Object Synthesis

**arXiv ID:** 2601.08015 | [PDF](https://arxiv.org/pdf/2601.08015v1)

**作者:** Abhishek Kumar `[一作]` `[通讯]`, Abhishek Kumar

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计了一种基于解码器的深度学习框架，直接生成符合增材制造约束的可打印3D结构。

**💡 创新点**

创新点在于将制造约束（倾斜角、壁厚、连通性、支撑优化）嵌入解码过程，使生成模型无需后处理即可输出可打印几何。

**🔧 技术方法**

使用变分解码器网络（四层反卷积块）配合约束感知批归一化、重建+制造约束+KL损失训练。

**📊 数据集**

在5类共500个测试对象上评估，数据来源未给出，推测使用公开3D网格/voxel数据集。

**📈 对比分析**

与无约束生成+后处理方法比较，制造可行率提升至96.8%，倾斜违规率降至0.3%，推理时间仅156 ms，打印成功率98%。

**⚠️ 局限性**

限制包括低分辨率 voxel (64³)、对特定打印技术（FDM）假设、缺少复杂材料属性、以及对非标准约束的可扩展性不足。

---

## 58. From Word Sequences to Behavioral Sequences: Adapting Modeling and Evaluation Paradigms for Longitudinal NLP

**arXiv ID:** 2601.07988 | [PDF](https://arxiv.org/pdf/2601.07988v1)

**作者:** Adithya V Ganesan `[一作]` (Stony Brook University), H Andrew Schwartz `[通讯]` (Vanderbilt University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并验证了纵向 NLP 的评估框架，指出传统文档级拆分会导致误导性结论，进而展示基于人序列建模的改进效果。

**💡 创新点**

创新点在于将泛化目标拆分为跨人（cross‑sectional）与跨时（prospective）两轴，并对指标进行人间与人内拆分；同时证明了序列输入与不同时间状态粗细的模型对不同泛化目标的选择至关重要。

**🔧 技术方法**

技术包括 RoBERTa‑large 与线性回归、主成分降维、Ridge 自回归、Bag‑of‑Elements、Transformer 等序列模型，以及专门设计的交叉验证、跨人/跨时拆分与人间/人内指标计算。

**📊 数据集**

使用 PTSD‑STOP 纵向数据集：238 名受试者在 90 天内每日录制视频日记并完成 PTSD 症状量表，共 17,051 条转录文本。

**📈 对比分析**

通过与传统随机文档拆分、均值基线、以及跨人/跨时拆分进行比较，发现传统拆分导致模型在跨人上表现不佳、跨时表现良好；序列模型在跨人拆分下 MAE 明显下降，相关系数提升；所有改进均以 MAE 和 Pearson r 为核心指标衡量。

**⚠️ 局限性**

局限性包括：实验仅在单一临床指标和单一数据集上验证；使用的模型相对简单，未探讨更深层次的序列网络；缺乏对子群差异的分析；缺乏对缺失机制的深入讨论；自动语音识别误差对结果的影响未作系统评估。

---

## 59. VBO-MI: A Fully Gradient-Based Bayesian Optimization Framework Using Variational Mutual Information Estimation

**arXiv ID:** 2601.08172 | [PDF](https://arxiv.org/pdf/2601.08172v1)

**作者:** Farhad Mirkarimi `[一作]` `[通讯]`, Farhad Mirkarimi

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种完全基于梯度的变分贝叶斯优化框架 VBO‑MI，利用变分互信息估计实现无内部采样循环的主动搜索；

**💡 创新点**

创新点在于：①将变分互信息的 Donsker‑Varadhan 下界直接嵌入为获取函数，②采用 actor‑critic 结构实现完整梯度流，消除传统 BNN‑BO 的采样与获取函数优化瓶颈，③不依赖高斯过程假设，能够在高维、噪声环境下更好探索；

**🔧 技术方法**

技术手段包括：变分互信息估计（DV 变分下界）、深度神经网络（动作网络与互信息估计网络，后者使用 LSTM + FC）、梯度下降自适应训练、批量化采样与交替优化；

**📊 数据集**

使用的实验数据集：三类标准基准函数（Branin、Hartmann、Ackley）以及四个真实工程任务（PDE 优化、干涉仪定位、Lunar Lander 游戏、Pest Control）；

**📈 对比分析**

与 GP、LLA、HMC、SGHMC、DKL、IBNN、稀疏高斯过程等基线对比，VBO‑MI 在大多数任务上获得与甚至优于基线的最终平均奖励，且在高维情形下收敛更快、计算量减少约两百倍；

**⚠️ 局限性**

局限性：需要对网络结构和学习率进行手工调参；在极高维或极度噪声、离散目标的情形下，互信息估计的梯度可能不稳定；模型仍需更多跨领域验证。

---

## 60. Incorporating Cognitive Biases into Reinforcement Learning for Financial Decision-Making

**arXiv ID:** 2601.08247 | [PDF](https://arxiv.org/pdf/2601.08247v1)

**作者:** Liu He `[一作]` (Beijing Institute of Technology), Liu He `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 9267 | [OpenAlex ID](https://openalex.org/A5100317866)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究将认知偏差（如损失厌恶和过度自信）引入强化学习框架，用以模拟人类交易者行为并评估其对金融交易策略的影响。

**💡 创新点**

创新点在于：① 在奖励函数中按比例放大负收益以模拟损失厌恶；② 根据近期平均奖励动态调整 ε 探索率以模拟过度自信；③ 对这些偏差进行系统的消融与超参数调优，探讨其在RL中的作用。

**🔧 技术方法**

技术手段包括：离散化价格状态、基于ε‑贪婪的表格 Q‑学习、偏差驱动的奖励修改、动态 ε 调整、超参数搜索与消融实验。

**📊 数据集**

使用的数据集为基于随机游走模型生成的合成价格序列（T=200 天），不包含真实市场数据。

**📈 对比分析**

将偏差模型的 Q‑学习代理与基线（λ=1、固定 ε=0.1）进行比较；实验显示所有配置均未实现正累计收益，Sharpe 比例高度波动，偏差对性能无显著提升。

**⚠️ 局限性**

局限性包括：① 合成随机游走环境缺乏真实市场的复杂性；② 采用表格 Q‑学习难以捕捉高维连续状态与偏差交互；③ 训练过程极不稳定；④ 仅考虑单资产、无组合效应，未检验多智能体场景。

---

## 61. Tiny-Twin: A CPU-Native Full-stack Digital Twin for NextG Cellular Networks

**arXiv ID:** 2601.08217 | [PDF](https://arxiv.org/pdf/2601.08217v1)

**作者:** Ali Mamaghani `[一作]` (University of California San Diego), Dinesh Bharadia `[通讯]` (University of California San Diego)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一套 CPU 原生、全栈的 5G 数字孪生框架 Tiny-Twin，用于在通用 CPU 上高保真模拟多用户下一代网络。

**💡 创新点**

通过在软件层实现多射线时变多通道卷积、并行化与稀疏卷积、CPU 绑定等技术，使得无需 FPGA/ GPU 即可在 20 通道下保持实时性，兼具高保真与低成本。

**🔧 技术方法**

利用 OpenAirInterface 5G 栈、RFSIM 模式、并行卷积、稀疏卷积、CPU Pinning、Docker 容器、Grafana/Prometheus 监控以及实时 RIC 接口等技术。

**📊 数据集**

使用了 3GPP 统计模型、Sionna 生成的射线追踪通道、ARGOS 真实无线测量通道，以及合成的 SNR 演化序列。

**📈 对比分析**

在 10 UE、20 通道的实验中，90% 计时降至 2 ms，RTT 维持 < 10 ms；相比原始 OAI vanilla 系统的 8 ms、RTT 超百 ms，以及 Colosseum、NVIDIA DT 等硬件加速平台，Tiny-Twin 在保持协议层真实性的前提下实现了 4 倍的实时性提升。

**⚠️ 局限性**

目前仅支持单基站单节点，扩展到多基站和大规模用户仍需分布式集群；对极低时延（<1 ms）仍受 CPU 计算限制；仅覆盖 5G NR 频段，尚未支持未来 6G 新调制与波形。

---

## 62. MemoBrain: Executive Memory as an Agentic Brain for Reasoning

**arXiv ID:** 2601.08079 | [PDF](https://arxiv.org/pdf/2601.08079v1)

**作者:** Hongjin Qian `[一作]` (Beijing Academy of Artificial Intelligence), Zheng Liu `[通讯]` (Beijing Academy of Artificial Intelligence)

**通讯引用:** 7323 | [OpenAlex ID](https://openalex.org/A5072069692)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了工具增强代理框架下的执行型记忆模型 MemoBrain，用以解决长周期推理中上下文溢出导致的逻辑失衡问题。

**💡 创新点**

创新点在于把记忆视作显式的上下文控制器，构建依赖感知的思维图并通过折叠（Fold）与选择性清除（Flush）操作，在固定预算下主动压缩并重构推理轨迹。

**🔧 技术方法**

采用监督微调的思维抽象技术进行记忆构造，并利用直接偏好优化（DPO）训练记忆管理策略来预测折叠与清除操作，整体框架实现协同式 Copilot 的异步执行。

**📊 数据集**

在 GAIA、WebWalker 和 BrowseComp-Plus 三大长周期推理基准上进行实验评估。

**📈 对比分析**

与直接推理、检索增强、工具集成等多类基线进行对比，MemoBrain 在所有基准上均显著提升性能，尤其在难度较高的任务中获得最高分。

**⚠️ 局限性**

主要局限包括：依赖推理代理能持续进行多轮工具调用；目前仅实现折叠和清除两种记忆操作，缺乏更丰富的记忆策略；以及在极早终止或工具实现不完善的情形下效果有限。

---

## 63. The many faces of multivariate information

**arXiv ID:** 2601.08030 | [PDF](https://arxiv.org/pdf/2601.08030v1)

**作者:** Thomas F. Varley `[一作]` `[通讯]` (Vermont Complex Systems Institute), Thomas F. Varley (Vermont Complex Systems Institute)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了一个参数化函数 Δ^k，将 S‑信息、双总相关性、负 O‑信息统一为特例，并引入其共轭层级 Γ^k，阐明它们与交互阶数的关系。

**💡 创新点**

创新点在于把多种高阶信息度量视为同一框架下的不同 k 值，并通过信息共轭构造对应的冗余层级，形成了可调节阶数的统一描述。

**🔧 技术方法**

使用信息论基础（Shannon 熵、总相关性、双总相关性、O‑信息）以及组合计数和共轭变换的理论推导。

**📊 数据集**

本研究为理论分析，未使用任何具体实验数据集。

**📈 对比分析**

通过数学证明和极端案例推导比较不同 k 下 Δ^k 的符号与数值，未给出实验性能指标。

**⚠️ 局限性**

限制在于仅适用于满足非负、单调性和纯关系脆弱性三条准则的函数，实际应用时需要验证其泛化性和在真实数据上的表现。

---

## 64. Emergent Coordination in Multi-Agent Systems via Pressure Fields and Temporal Decay

**arXiv ID:** 2601.08129 | [PDF](https://arxiv.org/pdf/2601.08129v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 65. Reasoning Beyond Chain-of-Thought: A Latent Computational Mode in Large Language Models

**arXiv ID:** 2601.08058 | [PDF](https://arxiv.org/pdf/2601.08058v1)

**作者:** Zhenghao He `[一作]` (University of Virginia), Aidong Zhang `[通讯]` (University of Virginia)

**通讯引用:** 11665 | [OpenAlex ID](https://openalex.org/A5013588572)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过稀疏自编码器识别并激活LLM内部潜在推理特征，以实现无CoT提示下的多步推理。

**💡 创新点**

证明多步推理是LLM的内在机制，而CoT提示仅是触发方式之一；并展示单一潜在特征即可匹配CoT性能。

**🔧 技术方法**

使用稀疏自编码器（SAE）进行特征提取与干预，并实施潜在特征调节。

**📊 数据集**

在GSM8K、GPQA、BBH等推理基准上进行实验。

**📈 对比分析**

与直接提示和CoT提示对比，潜在调节在大型模型上可获得与CoT相当的准确率，且生成长度显著缩短。

**⚠️ 局限性**

结果主要基于表征层的解耦假设，缺乏完整的因果机制解释，并且对非推理任务效果有限。

---

## 66. Multilingual, Multimodal Pipeline for Creating Authentic and Structured Fact-Checked Claim Dataset

**arXiv ID:** 2601.07985 | [PDF](https://arxiv.org/pdf/2601.07985v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 67. An Empirical Study on Knowledge Transfer under Domain and Label Shifts in 3D LiDAR Point Clouds

**arXiv ID:** 2601.07855 | [PDF](https://arxiv.org/pdf/2601.07855v1)

**作者:** Subeen Lee `[一作]` (KAIST), Jaesik Choi `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了ROAD基准，系统评估3D LiDAR点云分类在领域偏移和标签演化（类分裂、扩展、插入）下的迁移性能，涵盖零样本迁移、线性探测和连续学习三种任务。

**💡 创新点**

创新点在于首次在大规模真实世界数据上同时引入领域和多种标签变迁，构建统一评估协议，并对比不同骨干、训练目标和连续学习方法的表现。

**🔧 技术方法**

采用了PointNet++与Point Transformer骨干网络，使用交叉熵、SSL（SupCon、InfoNCE、VICReg）等训练目标，并引入LwF和EWC两种连续学习策略。

**📊 数据集**

实验基于Waymo、NuScenes和Argoverse2三大自动驾驶LiDAR数据集，构成Waymo→NuScenes和Waymo→Argoverse2迁移场景。

**📈 对比分析**

通过对比各方法在零样本迁移、线性探测和连续学习任务中的准确率、BWT和ACC等指标发现SSL目标在线性探测上优于监督学习，但在连续学习中效果不佳；LwF显著优于EWC，且对标签重叠场景更稳健。

**⚠️ 局限性**

限制在于仅聚焦分类任务、仅使用点坐标和强度特征、对标签映射假设较简单，且EWC在复杂骨干上稳定性差，未来需扩展至检测、预测任务并改进预训练与连续学习策略。

---

## 68. TabPFN Through The Looking Glass: An interpretability study of TabPFN and its internal representations

**arXiv ID:** 2601.08181 | [PDF](https://arxiv.org/pdf/2601.08181v1)

**作者:** Aviral Gupta `[一作]` (Birla Institute of Technology and Science), Dhruv Kumar `[通讯]` (Birla Institute of Technology and Science)

**通讯引用:** 6454 | [OpenAlex ID](https://openalex.org/A5027859418)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究分析了TabPFN模型的内部表示及其在不同层次上的演变，探讨了模型如何处理和转化输入特征。

**💡 创新点**

创新点在于首次对TabPFN模型进行机械性和表征分析，揭示了模型内部存储的有意义和结构化信息，并展示了模型如何系统地构建答案。

**🔧 技术方法**

使用了线性探测（linear probing）技术，通过一系列探测实验来分析模型的内部计算过程。

**📊 数据集**

构建了合成数据集，以已知的功能关系为基础，进行探测实验。

**📈 对比分析**

通过与传统模型的比较，发现TabPFN在中间层能够线性解码线性关系的系数和中间量，表明其内部计算是结构化的，性能优于简单的黑箱模型。

**⚠️ 局限性**

本研究的局限性在于仅限于简单的算术表达式，未来的工作可以扩展到更复杂的真实世界数据集。

---

## 69. Distributed Detection under Stringent Resource Constraints

**arXiv ID:** 2601.07989 | [PDF](https://arxiv.org/pdf/2601.07989v1)

**作者:** Abdelaziz Bounhar `[一作]` (Mohamed Bin Zayed University of Artificial Intelligence), Michèle Wigger `[通讯]` (Telecom Paris)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文在离散无记忆信道（DMC）上研究了在三种极端资源约束下的分布式假设检测问题，分别是（a）信道使用次数随观测数子线性增长；（b）使用次数为 n，但施加了子线性且满足几乎必然的功率约束；（c）仅在期望下施加子线性功率约束。

**💡 创新点**

创新点在于首次给出了 Stein 指数的闭式表达，并揭示了其在 DMC 的完全连通（每个输入都能产生所有输出）与部分连通两种结构下的二分行为：完全连通时在前两种约束下通信无效，仅能实现本地检测指数；部分连通时可达到与零速率无噪声链路相同的指数。为实现该指数提出了新的编码与判决策略，并给出了相应的取证与变形证明，包含变换测度与“放大”引理的巧妙运用。

**🔧 技术方法**

采用的技术包括典型序列分析、零速率编码、KL 散度与 Stein 引理、变换测度（change‑of‑measure）以及 blowing‑up lemma 进行误差指数的上下界推导；此外利用了信息论的极限定理和典型性不等式。

**📊 数据集**

由于研究属于理论信息论，文中未使用具体实验数据集；所有结果均在符号概率模型下推导，假设源分布和信道模型已知。

**📈 对比分析**

与传统无噪声零速率通信模型（Han–Kobayashi、Shalaby–Papamarcou 等）的指数相比，本文证明在部分连通 DMC 下仍能实现相同指数；而在完全连通 DMC 下，前两种约束下指数退化为仅利用决策中心本地观测得到的 KL 散度，第三种约束下指数略高但仍低于零速率无噪声链路的最佳指数。实验上无法验证，但理论上可用 KL 散度数值进行比较。

**⚠️ 局限性**

局限性包括：① 必须已知 DMC 的连通结构；② 仅适用于子线性资源约束和期望约束的理想化模型；③ 结果仅给出渐进式误差指数，缺乏有限样本量的性能评估；④ 对实际编码与解码实现细节未给出；⑤ 若信道不满足零成本符号存在假设，结论不再成立。

---

## 70. Coordinated Cooling and Compute Management for AI Datacenters

**arXiv ID:** 2601.08113 | [PDF](https://arxiv.org/pdf/2601.08113v1)

**作者:** Nardos Belay Abera `[一作]` (University of Alberta), Yize Chen `[通讯]` (University of Alberta)

**通讯引用:** 1847 | [OpenAlex ID](https://openalex.org/A5080591736)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了AI数据中心的层级协同冷却与计算控制框架，联合优化GPU并行度、DVFS和冷却参数，实现能耗与热管理的协同提升。

**💡 创新点**

创新点包括：① 将工作负载预测、TP调度、冷却MPC、Per-job DVFS四层耦合的层级化控制结构；② 引入温度感知的DVFS和基于DistilBERT的作业分类；③ 通过LSTM预测、MILP调度和MPC冷却实现联合能耗下降。

**🔧 技术方法**

使用了LSTM时序预测、DistilBERT分类、MILP优化、MPC冷却控制、GPU功耗模型、动态频率缩放（DVFS）和张量并行（TP）调度等技术。

**📊 数据集**

采用了Microsoft Azure LLM推理轨迹（包含多种工作负载）以及8×Tesla V100 GPU的实际功耗/温度记录。

**📈 对比分析**

与单TP8固定频率基线对比，实验显示计算能耗下降24.2%、冷却能耗下降31.2%、平均GPU温度下降17%，延迟基本不变，整体能耗显著降低。

**⚠️ 局限性**

局限性：仅针对空气冷却和单一GPU阵列，未考虑液冷或混合冷却；模型参数需要在不同数据中心环境手工校准；对极端长/短作业的预测误差仍存在。

---

## 71. How vehicles change lanes after encountering crashes: Empirical analysis and modeling

**arXiv ID:** 2601.08125 | [PDF](https://arxiv.org/pdf/2601.08125v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 72. An Efficient Additive Kolmogorov-Arnold Transformer for Point-Level Maize Localization in Unmanned Aerial Vehicle Imagery

**arXiv ID:** 2601.07975 | [PDF](https://arxiv.org/pdf/2601.07975v1)

**作者:** Fei Li `[一作]` (University of Wisconsin-Madison), Zhou Zhang `[通讯]` (University of Wisconsin-Madison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了基于加性 Kolmogorov–Arnold Transformer 的玉米点位定位框架，并构建了大规模点标注数据集。

**💡 创新点**

核心创新在于用 Padé KAN 模块替代传统 MLP，并将自注意力改为高效的 PKAN 加性注意机制，实现表达能力提升与计算复杂度下降。

**🔧 技术方法**

技术实现包括卷积主干、Padé KAN 网络、PKAN 加性注意、Hungarian 匹配训练、以及针对 UAV 图像的远程感知增强等。

**📊 数据集**

使用了自研的 Point‑based Maize Localization（PML）数据集，包含 1,928 张 3648×4864 像素的 UAV 图像和约 501,280 个手工点标注。

**📈 对比分析**

与多种基线（LCFCN、CLTR、FIDTM 等）在 PML 上对比，平均 F1 达 69.8%（比基线高 4.2%），精度 72.3%、召回 67.3%，计数 MAE 7.1，间距 RMSE 1.95–1.97 cm，推理速度提升 20.7% 并降低 12.6% FLOPs。

**⚠️ 局限性**

主要局限包括数据集规模受限、稠密或遮挡场景下匹配误差、对多模态融合与轻量化部署的需求，以及匹配策略仍可进一步改进。

---

## 73. Route, Retrieve, Reflect, Repair: Self-Improving Agentic Framework for Visual Detection and Linguistic Reasoning in Medical Imaging

**arXiv ID:** 2601.08192 | [PDF](https://arxiv.org/pdf/2601.08192v1)

**作者:** Md. Faiyaz Abdullah Sayeedi `[一作]` (Independent University Bangladesh), AKM Mahbubur Rahman `[通讯]` (Independent University Bangladesh)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 R^4 四步代理框架，将医疗影像任务拆解为路由、检索、反思、修复四个协同子系统

**💡 创新点**

创新在于利用患者历史与元数据驱动路由、基于示例记忆的检索、结构化反思与循环修复，实现无梯度微调的自我提升

**🔧 技术方法**

结合大规模视觉语言模型、少量示例提示、pass@k候选生成、JSON式问题检测与迭代修正技术

**📊 数据集**

在 VinBigData 胸部 X 光异常检测数据集和 IU Chest X‑Rays 报告生成数据集上进行实验

**📈 对比分析**

与单体 VLM 基线及前沿多代理方法对比，LLM‑as‑a‑Judge 分数提升约 1.7–2.5 分，弱监督定位 mAP 提升约 2.5–3.5 点，表现显著优于单体模型

**⚠️ 局限性**

仍受限于仅覆盖胸部 X 光、定位精度不及监督检测、缺乏跨模态验证、需进一步简化流程并加入临床反馈

---

## 74. PathoGen: Diffusion-Based Synthesis of Realistic Lesions in Histopathology Images

**arXiv ID:** 2601.08127 | [PDF](https://arxiv.org/pdf/2601.08127v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 75. When Models Know When They Do Not Know: Calibration, Cascading, and Cleaning

**arXiv ID:** 2601.07965 | [PDF](https://arxiv.org/pdf/2601.07965v1)

**作者:** Chenjie Hao `[一作]` (University of California), Yubei Chen `[通讯]` (Aizip)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套无训练的统一框架，用置信度校准来实现模型的自我认知，并基于此实现了模型级联与多专家数据清洗；

**💡 创新点**

创新点在于：① 将温度缩放与 Platt 缩放统一为跨视觉与语言任务的校准方法；② 利用校准后置信度的相对优势作为路由信号，实现高效小‑大级联以及同规模模型的互补级联；③ 采用基于多模型一致性的混合专家置信度清洗方案，显著提升数据集标签质量；

**🔧 技术方法**

核心技术包括：温度缩放、Platt 缩放、置信度优势路由、多专家一致性清洗、OOV/噪声下的鲁棒性评估；

**📊 数据集**

实验数据集覆盖 ImageNet‑1K、ImageNet‑C、MMLU、MBPP、BigCodeBench、ARC、GSM8K 等；

**📈 对比分析**

对比随机分配、未校准路由、矩阵分解路由等基线，级联方法在多任务上平均提升 5–15% 准确率且推理成本降低；同规模模型级联可进一步提升 8–12%；数据清洗方法在 ImageNet‑1K 及 MMLU 上的精度超过 Confident Learning，检测率也更高；

**⚠️ 局限性**

局限性包括：置信度校准对极端 OOD 的鲁棒性有限；需要可靠的验证器，尤其是生成任务；缺乏对多标签或连续预测的直接支持；阈值与阈值选择仍需人工设定或后期微调。

---

## 76. Where to Split? A Pareto-Front Analysis of DNN Partitioning for Edge Inference

**arXiv ID:** 2601.08025 | [PDF](https://arxiv.org/pdf/2601.08025v1)

**作者:** Adiba Masud `[一作]` (University of Texas at San Antonio), Palden Lama `[通讯]` (University of Texas at San Antonio)

**通讯引用:** 1000 | [OpenAlex ID](https://openalex.org/A5108522835)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 ParetoPipe 框架，用于在边缘设备上通过管道并行分区 DNN，并通过 Pareto 前沿分析系统性评估延迟与吞吐量的权衡。

**💡 创新点**

创新点在于将 DNN 分区视为多目标优化问题，生成完整的延迟-吞吐 Pareto 前沿；提供开源、双后端（PyTorch RPC 与自定义轻量级 socket）的实验工具；并在真实网络延迟/带宽下系统性研究分区策略的鲁棒性。

**🔧 技术方法**

技术包括：管道并行（Pipeline Parallelism）、PyTorch RPC 与自定义 TCP socket 通信、按块级别的 DNN 计算/内存剖析、Pareto 前沿分析、Linux traffic control（tc）仿真网络延迟/带宽、PSutil 监控 CPU/内存/网络利用率。

**📊 数据集**

使用预训练的 ImageNet 模型（MobileNetV2、ResNet18、InceptionV3、ResNet50、AlexNet、VGG16），在 CIFAR-10 数据集上微调后用于推理基准。

**📈 对比分析**

对比方法：对所有有效分割点进行实验，测量每种配置的吞吐量（图像/秒）和总延迟（秒），并在不同网络条件下绘制 Pareto 前沿。结果显示：自定义 socket 后端比 PyTorch RPC 提升 76% 的延迟、53% 的吞吐；Pi‑to‑GPU 配置在理想条件下显著提升吞吐；在 200 ms 延迟、5 Mbit/s 带宽的恶劣网络下，Pareto 前沿整体向高延迟、低吞吐移动，说明网络成为主瓶颈。

**⚠️ 局限性**

局限性：只考虑两设备管道（Pi‑to‑Pi / Pi‑to‑GPU），未涵盖更复杂的多节点拓扑；目标仅为延迟与吞吐，未加入能耗等指标；分区策略为静态预选，缺乏实时自适应；实验仅覆盖 CNN 结构，对 Transformer 等新型模型适用性未知。

---

## 77. Learning a Stochastic Differential Equation Model of Tropical Cyclone Intensification from Reanalysis and Observational Data

**arXiv ID:** 2601.08116 | [PDF](https://arxiv.org/pdf/2601.08116v1)

**作者:** Kenneth Gee `[一作]` (Massachusetts Institute of Technology), Sai Ravela `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 2482 | [OpenAlex ID](https://openalex.org/A5064676646)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过系统识别方法学习了一个10项多项式随机微分方程，用来模拟北半球热带气旋强度随时间的演化。

**💡 创新点**

创新点在于将Integral SINDy与Ensemble Kalman Filter结合，首次在海量历史观测与再分析数据上自动推导出可解释、物理风格的强度动力学方程，并揭示了风切变下的鞭状分岔现象。

**🔧 技术方法**

主要技术包括：Integral SINDy（稀疏回归）、Ensemble Kalman 更新（参数微调）、SDE 结构校准（加入随机扰动）以及多步前向滚动学习。

**📊 数据集**

使用的数据集为：IBTrACS 热带气旋强度记录（1979–2020）和对应的 ERA5 再分析环境变量（2016–2020），训练集采用 2016–2020 年数据，验证集 10%，测试集为 1979–2015 年全部记录。

**📈 对比分析**

与 6 小时、24 小时、72 小时的持久性模型对比，模型在 6h、24h 时段的总变差距离分别为 0.0529、0.0893，显著低于持久性模型（0.1912、0.2469）；在历史记录的强度分布、极值分布、陆岸强度以及回归期曲线中与观测相匹配度高，但在极端强度与某些地区的回归期上存在过估。

**⚠️ 局限性**

局限包括：对极端强度的随机扰动参数估计不足导致的极值过估、对强度低于 14 kt 的 TD 报告缺失造成的偏差、模型结构过于简化（多项式、线性噪声方差）以及仅基于预先工程化特征，缺乏对南半球及未来气候情景的泛化验证。

---

## 78. Automating API Documentation from Crowdsourced Knowledge

**arXiv ID:** 2601.08036 | [PDF](https://arxiv.org/pdf/2601.08036v1)

**作者:** Bonan Kou `[一作]` (Purdue University), Tianyi Zhang `[通讯]` (Purdue University)

**通讯引用:** 7142 | [OpenAlex ID](https://openalex.org/A5100437458)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出AutoDoc方法，结合Stack Overflow讨论和大语言模型，自动生成包含七类API知识的完整文档。

**💡 创新点**

创新点在于将Fine‑tuned DPR检索、LLM提取、自动校验与去重摘要三大模块串联，显著降低LLM幻觉和信息冗余。

**🔧 技术方法**

使用Fine‑tuned DPR检索模型、GPT‑4o进行知识抽取与自检、LLM自校验和聚合摘要技术。

**📊 数据集**

实验基于48个Java/Android/Kotlin/TensorFlow API的官方文档及其对应的SO帖子数据集。

**📈 对比分析**

与五个基线（SISE、API Caveat、三种GPT‑4o提示）对比，准确率提升至96.2%（比基线高77.7%），独特度提升9.5%，官方文档未覆盖知识占34.4%。

**⚠️ 局限性**

对低流行度API效果下降，仍存在错误且缺乏出处追溯，生成内容与官方文档存在一定重叠。

---

## 79. VULCA-Bench: A Multicultural Vision-Language Benchmark for Evaluating Cultural Understanding

**arXiv ID:** 2601.07986 | [PDF](https://arxiv.org/pdf/2601.07986v1)

**作者:** Haorui Yu `[一作]`, Qiufeng Yi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个多文化艺术批评基准（7,410幅图像-批评对，涵盖8种传统），并通过5层框架（L1–L5）评估视觉‑语言模型的文化理解；

**💡 创新点**

创新点在于：①提出跨文化的“文化对称性”原则，保证各文化在层级、协议和质量上的结构一致；②设计了225个文化特定维度与双语专家批评，实现层级化、可解释的文化评价；

**🔧 技术方法**

采用视觉‑语言模型（GPT‑4o、Claude‑Sonnet‑4.5、Gemini‑2.5‑Pro、Qwen3‑VL‑235B、GLM‑4V‑Flash）进行评测，并通过维度覆盖率（DCR）和层级差距ΔL诊断；

**📊 数据集**

使用自建数据集——包括图像、双语批评、维度标注与元数据；

**📈 对比分析**

与现有VLM基准对比时，所有模型在L3–L5层的DCR普遍比L1–L2低25–40个百分点，表明高层文化推理是显著难点；

**⚠️ 局限性**

局限性包括：中文/西方占比高，导致少数文化评估方差大；L5层主观性高；双语设计对非中文传统词汇的保留有限；DCR仅为粗粒度诊断，非精确排名工具。

---

## 80. Human-inspired Global-to-Parallel Multi-scale Encoding for Lightweight Vision Models

**arXiv ID:** 2601.08190 | [PDF](https://arxiv.org/pdf/2601.08190v1)

**作者:** Wei Xu `[一作]` `[通讯]` (Qinghai Normal University), Wei Xu (Qinghai Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种人类视觉启发的全局到并行多尺度编码（GPM）模块，并在此基础上构建了轻量级的 H-GPE 视觉网络，用于分类、检测和分割任务。

**💡 创新点**

创新点在于：①将全局感知与局部细节同时并行处理；②通过 GIG 提取全局语义先验；③在并行分支中分别使用 LSAE（中大尺度语义）和 IRB（高频细节）并配合 ASA/CRA 提升特征表达；④在轻量化设计上实现了更低参数与 FLOPs 的优异准确率。

**🔧 技术方法**

主要技术包括：strip pooling + 大核分组卷积（GIG）、局部窗口自注意力（LSAE）、倒置残差块（IRB）以及通道/空间注意力模块（ASA、CRA），构成 GPM；整体网络采用多阶段堆叠的 GPE‑Block 与 IRB，形成 H‑GPE‑S/T/N 三种规模。

**📊 数据集**

实验使用的公开数据集：ImageNet（分类）、MS COCO（目标检测）和 ADE20K（语义分割），全部在单张 RTX 4090 GPU 上训练。

**📈 对比分析**

与多种轻量化基准（MobileNetV4、Mamba、MobileViT、EMO、SpectFormer 等）进行对比。H‑GPE‑S/T/N 在参数和 FLOPs 约相同或更低的情况下，Top‑1、AP、mIoU 分别提升 1–2% 以上，显示出更优的准确‑效率折中。

**⚠️ 局限性**

局限性：①固定核/窗口大小不易适应不同分辨率或尺度；②目前仅针对静态图像，缺少时序建模；③评估仅基于 FLOPs、参数和准确率，未包含实际延迟、吞吐量等硬件感知指标。

---

## 81. Moonworks Lunara Aesthetic Dataset

**arXiv ID:** 2601.07941 | [PDF](https://arxiv.org/pdf/2601.07941v1)

**作者:** Yan Wang `[一作]`, Sabit Hassan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

发布了 Lunara Aesthetic Dataset，一个包含 2000 张高质量、区域化风格的图像–提示对的数据集，用于研究文本到图像生成中的提示遵循和风格调节。

**💡 创新点**

创新点在于：① 针对美学与风格进行细致策划的公开数据集；② 采用人类细化的提示，提升提示语义与图像的对应精度；③ 通过区域化艺术风格标签实现跨文化、跨媒介的风格对比与实验；④ 采用 Apache 2.0 许可证，保障学术与商业自由使用。

**🔧 技术方法**

技术手段包括：Moonworks Lunara（小于10B 参数的扩散模型）生成图像；CAT 主动学习框架优化训练数据；人工提示精炼与双轮注释流程；评估采用 LAION Aesthetic v2、CLIP 相似度、Recall@K 交叉模态检索和 LPIPS 视觉多样性指标。

**📊 数据集**

主要数据集为 Lunara Aesthetic Dataset（2000 张图像–提示对）；评估时与 CC3M、LAION‑2B‑Aesthetic、WIT 等公开数据集做对比。

**📈 对比分析**

比较方法：使用 LAION Aesthetic 预测分数评估美学质量，CLIP 余弦相似度衡量图像–文本对齐，Recall@1/5/10 评估交叉模态检索效果，LPIPS 衡量视觉多样性。结果显示 Lunara 的平均美学分数 6.32、对齐相似度 0.32、Recall@1 约 43%（使用 ViT‑B/32），并在美学分布、检索表现和多样性上均优于对比数据集。

**⚠️ 局限性**

局限性：① 数据完全由模型合成，缺乏真实世界多样性；② 只涵盖 2000 张样本，规模有限；③ 区域化风格标签为粗粒度抽象，未细化到历史时期或具体流派；④ 仅聚焦提示与风格，对实际生成模型的泛化能力与伦理影响讨论有限。

---

## 82. KVzap: Fast, Adaptive, and Faithful KV Cache Pruning

**arXiv ID:** 2601.07891 | [PDF](https://arxiv.org/pdf/2601.07891v1)

**作者:** Simon Jegou `[一作]`, Maximilian Jeblick `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了KVzap，一种利用轻量级隐藏状态模型预测KV重要性并阈值裁剪的KV缓存压缩方法，能够在保持几乎不降准确率的前提下实现约3-4倍的压缩；

**💡 创新点**

创新点包括：①在KVzip基础上加入归一化得分（KVzip+）并用轻量级模型逼近；②采用阈值裁剪而非固定top-k，实现输入自适应压缩；③结合滑动窗口保持局部上下文；④兼容prefilling与decoding两种阶段；

**🔧 技术方法**

采用的技术有：轻量级线性或两层MLP替代KVzip评分、阈值裁剪、滑动窗口、与FlashAttention2/PagedAttention等高效注意力内核兼容；

**📊 数据集**

训练数据来自NVIDIA Nemotron预训练样本集（多语言、代码、数学等多领域），评估数据集包括RULER 4k/16k、LongBench和AIME25；

**📈 对比分析**

与15种现有KV压缩方法（如KVzip、Expected Attention、Compactor等）对比，KVzap在RULER 4k、LongBench、AIME25等基准上保持接近全缓存准确率，压缩比约2-4×，在RULER 4k上实现最佳表现；

**⚠️ 局限性**

局限性包括：仍为后置方法，需手工集成；未实现真正的显存或速度提升；在更大模型或稀疏注意力架构上的验证不足；对滑动窗口大小和阈值选择较为敏感。

---

## 83. Edge-AI Perception Node for Cooperative Road-Safety Enforcement and Connected-Vehicle Integration

**arXiv ID:** 2601.07845 | [PDF](https://arxiv.org/pdf/2601.07845v1)

**作者:** Shree Charran R `[一作]` (Indian Institute of Science), Rahul Kumar Dubey `[通讯]` (Bosch Global Software Technologies Private Limited)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

实现了一个基于YOLOv8‑Nano、DeepSORT和轻量OCR的低功耗Edge‑AI路侧感知节点，支持多类交通违规检测、车牌识别与V2X安全事件发布；

**💡 创新点**

创新点在于自动ROI生成、规则驱动的多语种OCR、实时V2X输出以及在Jetson Nano上TensorRT FP16优化，实现97.7%违规检测准确率和30 fps的低能耗表现；

**🔧 技术方法**

使用YOLOv8‑Nano、DeepSORT、Tesseract OCR、TensorRT FP16、MQTT/V2X等技术；

**📊 数据集**

数据集包括COCO+自定义印度交通视频、11,271张车牌图像以及多场景违规标注；

**📈 对比分析**

与YOLOv4‑Tiny、PP‑YOLOE‑S、NanoDet‑Plus等模型对比，Jetson Nano上获得+10.7% mAP、1.4×准确率/瓦特提升，28‑30 fps、9.6 W功耗；

**⚠️ 局限性**

局限在于对严重遮挡、低光和非标准车牌识别误差仍高，未覆盖行人及道路通道违规，仅在单摄像头场景下验证。

---

## 84. LLM Review: Enhancing Creative Writing via Blind Peer Review Feedback

**arXiv ID:** 2601.08003 | [PDF](https://arxiv.org/pdf/2601.08003v1)

**作者:** Weiyue Li `[一作]` (Harvard University), Mengyu Wang `[通讯]` (Harvard University)

**通讯引用:** 2511 | [OpenAlex ID](https://openalex.org/A5100632182)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于盲目同行评审的多代理框架 LLM Review，提升语言模型在科幻短篇写作中的创意表现

**💡 创新点**

通过将批评与改写分离，限制信息流以保持创意轨迹的多样性，克服传统多代理互动导致的同质化问题

**🔧 技术方法**

使用多代理角色扮演、盲评流程、LLM 评审、规则化多维度创意评估（Token 涌现、KL 词汇偏差、语义余弦距离、嵌入空间体积增益）

**📊 数据集**

构建 SciFi‑100 科幻写作数据集（100 条多维度创意提示）并与 SFGram 经典科幻小说语料做对比评估

**📈 对比分析**

与单代理、LLM Teacher、LLM Debate、LLM Discussion 等基线在 LLM‑as‑a‑Judge 评分和规则化多维度指标上对比，LLM Review 在五个创意维度得分均最高，且在规则化指标上表现最为突出；在多模型族中，较小模型通过该框架可超越更大单代理模型

**⚠️ 局限性**

仅评估短篇科幻写作，规则化指标不保证语义意义；人类评测样本有限，专业作家可能评价不同；多代理框架推理成本约为单代理的9倍

---

## 85. FinVault: Benchmarking Financial Agent Safety in Execution-Grounded Environments

**arXiv ID:** 2601.07853 | [PDF](https://arxiv.org/pdf/2601.07853v1)

**作者:** Zhi Yang `[一作]` (Shanghai University of Finance and Economics), Liwen Zhang `[通讯]` (Shanghai University of Finance and Economics)

**通讯引用:** 5898 | [OpenAlex ID](https://openalex.org/A5100459595)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 FinVault，一个针对金融智能代理的执行层安全基准，包含 31 个监管驱动的沙箱情境、107 条真实漏洞和 963 个系统测试用例，用于评估代理在实际操作流程中的安全性。

**💡 创新点**

首次提出基于可写状态和真实金融业务流程的安全基准，系统覆盖提示注入、越狱、金融专用攻击与正常输入，显著提升了安全评估的现实感与针对性。

**🔧 技术方法**

使用大型语言模型（LLM）驱动的智能代理、可写数据库沙箱、攻击脚本与安全防御策略，对代理行为进行模拟与评测。

**📊 数据集**

数据集包括 31 个监管案例驱动的情境、107 条公开漏洞以及 963 个覆盖攻击与误报的测试用例。

**📈 对比分析**

与多种现有防御机制对比实验显示，平均攻击成功率最高达 50%；即便是最强防御也有 6.7% 的成功率，说明当前安全设计在金融场景中的转移性有限。

**⚠️ 局限性**

局限性在于评估仅覆盖选定的 LLM 模型与攻击类型，缺乏跨模型泛化验证，且未涵盖更复杂的金融业务流程或多代理交互场景。

---

## 86. E^2-LLM: Bridging Neural Signals and Interpretable Affective Analysis

**arXiv ID:** 2601.07877 | [PDF](https://arxiv.org/pdf/2601.07877v1)

**作者:** Fei Ma `[一作]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy), Qi Tian `[通讯]` (Huawei)

**通讯引用:** 40497 | [OpenAlex ID](https://openalex.org/A5100393506)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出并实现了首个将脑电（EEG）信号与大语言模型（LLM）结合的可解释情绪分析框架 E²-LLM。

**💡 创新点**

通过分层 EEG 编码器、可学习投影与 Qwen3 LLM 的多阶段训练，实现跨模态对齐、指令调优与链式思考，使模型能生成可解释的情绪报告。

**🔧 技术方法**

使用 EEGPT 预训练的 EEG 编码器、Qwen3 LLM、低秩适配 LoRA、层次 Transformer、专用投影层以及三阶段情绪判别、对齐与指令调优技术。

**📊 数据集**

在 SEED‑VII 多模态情绪数据集（20 名受试者、7 种情绪）上进行训练和评估。

**📈 对比分析**

与 NeuroLM 等基线进行对比，在 IED 基础情绪预测任务中 8B/14B 版本分别达 61.73%/61.05% 的平衡准确率、Cohen κ 55.14%；在多任务推理与零样本情境推理中，14B 版平均准确率 69.84%，显著优于随机基线。

**⚠️ 局限性**

仅在 SEED‑VII 小样本上验证，模型规模大、计算成本高，某些任务表现不随规模单调提升，零样本情境推理仍有限，缺乏人工评估解释质量。

---

## 87. Attention Projection Mixing and Exogenous Anchors

**arXiv ID:** 2601.08131 | [PDF](https://arxiv.org/pdf/2601.08131v1)

**作者:** Jonathan Su `[一作]` `[通讯]`, Jonathan Su

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ExoFormer 与 NuResFormer 两种框架，利用归一化的跨层残差混合重新设计 Transformer 的注意力模块，并通过外部锚点解耦第一层的锚定与计算职责，从而提升模型性能与数据效率。

**💡 创新点**

① 统一归一化残差混合机制；② 引入外部锚点（ExoFormer）来分离锚定与计算；③ 动态混合模块（DM）实现上下文感知的系数；④ Offloading Hypothesis 解释外部锚点导致的表示崩塌与性能提升。

**🔧 技术方法**

RMSNorm、RoPE、QKNorm、SwiGLU 激活、Gated Attention、动态混合系数生成、Mu 优化器、FlashAttention、以及多尺度系数（scalar、headwise、elementwise）。

**📊 数据集**

FineWeb‑Edu 10B‑token 子集用于训练与验证（100M token），以及 6 个多项选择下游任务（ARC‑Challenge、ARC‑Easy、HellaSwag、OpenBookQA、PIQA、Winogrande）。

**📈 对比分析**

通过与基线 Transformer、Gated Attention、ResFormer、NuResFormer、Naïve Combination 等模型在 perplexity、下游准确率、注意力 sink 等指标上对比，ExoFormer 与其动态变体分别在 PPL、准确率和注意力 sink 上均优于对照；动态 ExoFormer 进一步提升了准确率并降低 PPL，且数据效率提升 1.84×。

**⚠️ 局限性**

实验仅在 450M 参数、10B token 的规模下进行，未验证更大模型或长上下文任务；未进行 λ 参数的全量搜索；缺乏对归一化残差混合机制的形式化理论解释。

---

## 88. Likelihood ratio for a binary Bayesian classifier under a noise-exclusion model

**arXiv ID:** 2601.07982 | [PDF](https://arxiv.org/pdf/2601.07982v1)

**作者:** Howard C. Gifford `[一作]` (University of Houston), Howard C. Gifford `[通讯]` (University of Houston)

**通讯引用:** 1803 | [OpenAlex ID](https://openalex.org/A5032527429)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于噪声排除的统计理想观察者模型，利用阈值截断特征以减少噪声并压缩模型参数。

**💡 创新点**

创新点在于将特征截断与“gist”与“猜测”成分结合进AUC分析，构建了截断理想观察者的新理论框架，并在高内部噪声下实现性能提升。

**🔧 技术方法**

主要技术包括贝叶斯理想分类器、似然比评分、ROC曲线与AUC计算、阈值截断的概率分布推导以及内部噪声卷积处理。

**📊 数据集**

实验使用合成的正态分布特征（负类N(0,1)，正类N(0.75,σ²)），未采用真实医学影像数据集。

**📈 对比分析**

通过模拟不同阈值和内部噪声水平对比截断与不截断理想观察者的AUC，结果显示在高噪声场景下截断策略可显著提高AUC，甚至在无信息评分时仍能超过随机猜测。

**⚠️ 局限性**

局限性包括仅在低维正态特征上验证、缺乏真实医学影像与人类读者的对比、阈值优化的计算复杂度、对特征相关性的处理有限以及对模型鲁棒性未进行深入评估。

---

## 89. Directional Electrical Spiking, Bursting, and Information Propagation in Oyster Mycelium Recorded with a Star-Shaped Electrode Array

**arXiv ID:** 2601.08099 | [PDF](https://arxiv.org/pdf/2601.08099v1)

**作者:** Andrew Adamatzky `[一作]` (University of the West of England), Andrew Adamatzky `[通讯]` (University of the West of England)

**通讯引用:** 12642 | [OpenAlex ID](https://openalex.org/A5036652783)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在木屑基质中生长的牡蛎蘑菇菌丝体上，用星形差分电极阵列连续记录电位，分析了电位的时空结构、方向异质性、爆发动力学、局部耦合与慢速传播。

**💡 创新点**

创新点在于：① 采用角度分辨率的星形多通道电极，首次揭示菌丝体电信号在不同方向上的显著异质性；② 发现电活动表现为局部耦合、爆发聚集且存在多量级的有向传播延迟，支持菌丝体为空间可扩散的兴奋介质；③ 在低采样率（1 Hz）下仍能捕获慢速生理信号。

**🔧 技术方法**

技术手段包括低噪声差分电极阵列、Pico ADC‑24 24‑bit 数据记录器（1 Hz 采样），以及基于 Python 的自适应阈值检测、统计分析、Pearson 相关系数计算和事件驱动的传播延迟测定。

**📊 数据集**

数据集来自三次独立实验，每次记录约5 天，使用木屑基质完全殖民的 Pleurotus ostreatus 菌丝体，产生了八个方向通道的长时序电位记录。

**📈 对比分析**

通过比较各通道的平均发放率、幅度分布、爆发大小/持续时间、相关系数随距离衰减以及传播延迟分布，显示出明显的方向性异质性和局部耦合；与传统单通道或线性阵列研究相比，本方法揭示了更丰富的空间结构和慢速传播特征。

**⚠️ 局限性**

局限性包括：① 采样率仅为1 Hz，无法捕捉秒级快速事件；② 仅记录自发活动，缺少对外部刺激响应的评估；③ 电极布置固定，无法实时追踪菌丝体形态的动态变化；④ 对信号产生机制和功能意义仍缺乏直接证据。

---

## 90. One-Shot Federated Ridge Regression: Exact Recovery via Sufficient Statistic Aggregation

**arXiv ID:** 2601.08216 | [PDF](https://arxiv.org/pdf/2601.08216v1)

**作者:** Zahir Alsulaimawi `[一作]` `[通讯]`, Zahir Alsulaimawi

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种单轮联邦岭回归协议（One-Shot σ-Fusion），通过各客户端一次性上传其 Gram 矩阵和矩向量即可在服务器端直接得到与集中式岭回归完全一致的模型。

**💡 创新点**

创新点在于：① 证明在任意数据分布与客户端划分下，单轮聚合即可实现精确恢复；② 通过一次性通信消除迭代协议的隐私累积与通信瓶颈；③ 结合随机投影在高维场景下实现可控的通信-精度折衷；④ 提供了对隐私、客户端掉线、超参数选择等实际部署问题的完整分析。

**🔧 技术方法**

核心技术包括：分布式充分统计量的加法分解、岭回归闭式解、Gaussian 机制实现差分隐私、随机投影压缩、留一客户端交叉验证、矩阵分块与 Cholesky 分解。

**📊 数据集**

实验主要基于人工生成的异构回归数据集（K=20/50/100/200/500 客户端，每个 200–500 样本，特征维度 50–4000），并与 FedAvg、FedProx 等迭代基线及集中式岭回归对比。

**📈 对比分析**

与迭代基线相比，单轮协议在测试 MSE 上与集中式方法几乎完全一致（误差 < 0.01%），通信量可低至 10–80 倍、计算时间 10–400 倍；在异构程度、客户端掉线、隐私预算为 ε≥1 时保持优越性能；在高隐私（ε<0.5）或极大维度（d>1000）下需引入随机投影或安全聚合。

**⚠️ 局限性**

局限包括：仅适用于可闭式解的线性/核模型；对 d>4R 的高维情况仍需投影；在极高隐私预算下单轮噪声放大导致精度下降；服务器侧需要 O(d³) 的矩阵求逆，超大 d 时可采用迭代求解。

---

## 91. Post-Quantum Cryptography Key Expansion Method and Anonymous Certificate Scheme Based on NTRU

**arXiv ID:** 2601.07841 | [PDF](https://arxiv.org/pdf/2601.07841v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 92. Cardinality-consistent flag codes with longer type vectors

**arXiv ID:** 2601.08144 | [PDF](https://arxiv.org/pdf/2601.08144v1)

**作者:** Junfeng Jia `[一作]` (Beijing Jiaotong University), Yanxun Chang `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 1782 | [OpenAlex ID](https://openalex.org/A5004840229)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文基于部分扩散与循环轨道旗码，提出了在有限域上n=sk+h（s≥2,0≤h<k）情形下，构造出最优距离旗码和更长类型向量的旗码，且保证码量达到∑_{i=1}^{s-1}q^{ik+h}+1，进一步获得了最长可行类型向量(1,2,…,k,n−k,…,n−1)的最优距离旗码；

**💡 创新点**

创新点在于：①首次系统化构造了包含更长类型向量的旗码；②利用循环轨道动作与部分扩散相结合，既提升了码长，又保持了码量不变；③给出了关于更长类型向量旗码的距离、码量和一致性（cardinality-consistent）的完整理论阐述。

**🔧 技术方法**

主要技术：1）部分扩散（partial spread）与k-扩散；2）循环轨道旗码（cyclic orbit flag codes）及其作用群的矩阵表示；3）伴随矩阵与不可约多项式的理论；4）矩阵行列式与秩分析来证明距离。

**📊 数据集**

论文为理论研究，没有使用具体数据集。

**📈 对比分析**

方法评估通过理论证明完成：对所有构造的旗码证明了最优距离、码量与一致性；对比已知的最优距离旗码（如(1,…,k,n−k,…,n−1)）显示码量与距离相同或更优。

**⚠️ 局限性**

局限性：①仅给出了在n=sk+h（s≥2,0≤h<k）范围内的构造；②缺乏实验验证或编码/译码算法实现；③对完全旗码（full flag codes）在卡点一致性与码量上仍未达到最优；④对于s=4时的特殊情况仍需进一步研究。

---

## 93. Carrying is Hard: Exploring the Gap between Hardness for NP and PSPACE for the Hanano and Jelly no Puzzles

**arXiv ID:** 2601.08057 | [PDF](https://arxiv.org/pdf/2601.08057v1)

**作者:** Michael C. Chavrimootoo `[一作]` (Denison University), Jin Seok Youn `[通讯]` (Denison University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究了一玩家游戏 Hanano Puzzle（以及相似游戏 Jelly no Puzzle）的可解性问题，证明在仅用一种颜色且不允许灰色可移动块的限制下，问题仍为 PSPACE‑complete，并通过宽度为 1 的块进一步展示了可解性属于 P，阐明了“搬运”机制是导致硬度的关键因素。

**💡 创新点**

创新点：①首次证明在单色、无灰色可移动块的限制下 Hanano Puzzle 仍为 PSPACE‑complete；②提出搬运（carrying）是决定从 P 到 PSPACE‑hard 的核心机制；③给出了完整的可逆 NCL 归约构造（包含红弯曲、OR、AND 三种 gadget），并与 Jelly no Puzzle 进行对比；④对可移动块数有限的情况给出 P‑可解性上限。

**🔧 技术方法**

主要技术：Nondeterministic Constraint Logic（NCL）框架、可视化表示（visibility representations）以及通过构造可逆块移动的 gadget 进行归约；同时使用降维度的“坠落度量”证明宽度为 1 的块时可解性属于 P。

**📊 数据集**

没有使用实际数据集，所有结果均为理论复杂度分析。

**📈 对比分析**

方法与已有工作（如 Liu & Yang、Crabtree & Mitsou 等）的比较表明在相同限制下仍保持 PSPACE‑hard；通过构造可逆 gadget 证明了可解性不随块宽度缩小而变易；性能评价为理论上可解性属于 P 或 PSPACE‑complete，未给出实验性能。

**⚠️ 局限性**

局限性：①对可移动块数为常数的情况仅给出了上界，而未提供完整的硬度证明；②对两色且无灰色块的完整复杂度仍未确定；③归约中仅使用宽度为 1 或 2 的块，未探讨更一般块形状；④未给出实验验证或对具体实例的求解算法。

---

## 94. From Tool to Teacher: Rethinking Search Systems as Instructive Interfaces

**arXiv ID:** 2601.08035 | [PDF](https://arxiv.org/pdf/2601.08035v1)

**作者:** David Elsweiler `[一作]` (University of Regensburg), David Elsweiler `[通讯]` (University of Regensburg)

**通讯引用:** 2073 | [OpenAlex ID](https://openalex.org/A5041921674)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

从教育学视角重新审视信息检索系统，将其视为教学环境，分析传统与生成式 AI 搜索功能对用户学习的启发与限制；

**💡 创新点**

提出将七种教学框架（nudging、boosting、scaffolding、cognitive apprenticeship、SRL、instructional feedback、heutagogy）应用于信息检索系统评估与设计的概念框架，首次将学习理论系统化用于检索界面；

**🔧 技术方法**

采用教育与行为科学理论构建框架，并对现有搜索功能（查询自动补全、源标签、对话式 AI、记忆式对话、ReAct 代理）进行框架映射与功能说明；

**📊 数据集**

未使用实证数据集，而是基于文献综述、案例演练与示例系统的设计原型；

**📈 对比分析**

比较方法为理论映射与案例对比，未给出定量性能指标，仅通过教学价值的潜在提升与功能映射来评估系统；

**⚠️ 局限性**

局限包括缺乏经验验证、未涵盖所有教学模型、对生成式 AI 的可靠性（如幻觉）未做处理、示例系统仅为概念性设计，缺乏用户实验与性能评估。

---

## 95. Riemannian Zeroth-Order Gradient Estimation with Structure-Preserving Metrics for Geodesically Incomplete Manifolds

**arXiv ID:** 2601.08039 | [PDF](https://arxiv.org/pdf/2601.08039v1)

**作者:** Shaocong Ma `[一作]` (University of Maryland), Heng Huang `[通讯]` (University of Maryland)

**通讯引用:** 24460 | [OpenAlex ID](https://openalex.org/A5060016795)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文研究在黎曼度量不可完整的情形下，构造结构保持且几何完整的新度量，并使用该度量进行零阶优化以逼近原始度量下的驻点。

**💡 创新点**

创新点在于提出了几何完整且保持驻点不变的度量构造，重新分析了对称两点零阶估计器的均方误差，仅依赖于流形内在几何，并在此基础上给出了随机梯度下降的收敛保证。

**🔧 技术方法**

采用了结构保持的度量构造技术、对称两点零阶估计器的内在误差分析以及基于该估计器的随机梯度下降方法。

**📊 数据集**

实验数据集包括合成问题和实际的网格优化任务。

**📈 对比分析**

与传统在可完整度量下的算法比较，本文方法在不可完整情形下仍保持稳定收敛，且达到已知的最佳复杂度。

**⚠️ 局限性**

局限性在于对原始度量可转化性的假设以及实验覆盖范围有限，尚未在更广泛的真实世界任务中验证。

---

## 96. LUT-Compiled Kolmogorov-Arnold Networks for Lightweight DoS Detection on IoT Edge Devices

**arXiv ID:** 2601.08044 | [PDF](https://arxiv.org/pdf/2601.08044v1)

**作者:** Oleksandr Kuznetsov `[一作]` (eCampus University), Oleksandr Kuznetsov `[通讯]` (V.N. Karazin Kharkiv National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出将Kolmogorov‑Arnold网络的B‑样条评估编译为查找表，实现在资源受限的IoT边缘设备上对DoS攻击进行快速检测。

**💡 创新点**

创新点在于通过预量化LUT与线性插值替代运行时样条计算，获得几千倍加速且保持几乎无准确率损失。

**🔧 技术方法**

采用KAN架构、B‑spline拟合、对称/非对称int8量化、LUT编译、线性插值以及Numba JIT加速等技术。

**📊 数据集**

使用CICIDS2017数据集进行二分类训练与评估。

**📈 对比分析**

与原始浮点KAN及主流IoT IDS（如SecEdge、ALNS‑CNN）对比，LUT‑8在单样本推理速度提升约6000×、批量256提升约68×，准确率仅下降0.04%，模型大小仅0.19 MB。

**⚠️ 局限性**

局限性包括仅在x86 CPU上评估，ARM性能未知；能耗未直接测量；仅验证单一数据集，未评估多类检测；低分辨率下对分布漂移的鲁棒性有限。

---

## 97. Explaining Generalization of AI-Generated Text Detectors Through Linguistic Analysis

**arXiv ID:** 2601.07974 | [PDF](https://arxiv.org/pdf/2601.07974v1)

**作者:** Yuxi Xia `[一作]` (University of Vienna), Benjamin Roth `[通讯]` (University of Vienna)

**通讯引用:** 1448 | [OpenAlex ID](https://openalex.org/A5046895021)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含多种 LLM、提示策略和文本领域的综合基准，系统评估 AI 文本检测器在跨提示、跨模型、跨数据集情境下的泛化能力，并通过 80 个表层语言特征的偏移量与检测性能的相关性分析，解释泛化差异。

**💡 创新点**

创新点在于：①首次将表层语言特征（时态、语态、代词等）与检测器泛化性能关联并量化；②构建 516k 条文本的大规模、多维度基准；③使用多重假设校正和非线性相关分析验证相关性稳健性。

**🔧 技术方法**

主要技术包括：Fine‑tune XLM‑RoBERTa‑base 与 DeBERTa‑V3‑small 进行二分类；使用 StyloMetrix、Gunning Fog 等工具提取 80 项语言特征；计算训练-测试分布的特征偏移；Pearson 相关、Spearman 相关以及 Bonferroni、Benjamini–Hochberg 校正进行统计分析。

**📊 数据集**

使用 4 个领域数据集（arXiv 摘要、Amazon 评价、CNN/Daily Mail 新闻、ASQA QA），结合 7 种 LLM（Mistral‑123B、Deepseek‑70B、Llama‑70B、Qwen‑72B/32B/14B、Solar‑22B）和 6 种提示策略（0-shot、3-shot、Style、0-shot CoT、1-shot CoT、Self‑refine），总计 516,000 条文本。

**📈 对比分析**

对比方法：在 3 种泛化设置（跨提示、跨模型、跨数据集）下评估检测器，发现 In‑Domain 几乎完美，但 Out‑of‑Domain 精度显著下降（如跨模型从 0.75 降至 0.58，跨数据集从 0.90 降至 0.57）。相关性分析揭示，时态、代词使用、被动语态等特征与泛化下降高度相关，不同模型的检测器依赖不同语言信号。

**⚠️ 局限性**

局限性：仅针对英文文本，使用的语言特征为表层特征，未覆盖语义或话语层面；仅评估 encoder‑only 微调模型，未探究生成式或检索式检测器；相关性分析为关联而非因果；基准覆盖范围虽广但仍未包含最新 LLM 与更丰富的提示方法。

---

## 98. Sola-Visibility-ISPM: Benchmarking Agentic AI for Identity Security Posture Management Visibility

**arXiv ID:** 2601.07880 | [PDF](https://arxiv.org/pdf/2601.07880v1)

**作者:** Gal Engelberg `[一作]` (Sola Security), Yoni Weintrob `[通讯]` (Sola Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Sola Visibility ISPM Benchmark，用于评估代理式人工智能系统在真实企业身份安全姿态管理（ISPM）可见性任务中的表现，并开发了Sola AI Agent实现数据驱动的问答。

**💡 创新点**

创新点在于：①首个面向ISPM的基准，结合AWS、Okta、Google Workspace三大平台的真实生产环境；②提供多维度评价框架（专家评估 + LLM-as-Judge），兼顾答案质量、推理过程、检索利用和SQL生成；③区分fast‑path 与 full‑path 推理策略，揭示例子适配对性能的关键作用。

**🔧 技术方法**

采用的技术包括：自然语言处理+检索增强（检索示例SQL、schema）；多步推理（Tree‑of‑Thought 样式）与快速路径推理；SQL 生成与执行；LLM-as-Judge 评估（Claude Sonnet 4.5 + GPT‑4.1）；人工专家评审。

**📊 数据集**

使用的数据集为企业级身份与访问管理数据，涵盖 AWS IAM、Okta IdP、Google Workspace，包含身份、权限、配置信息及安全配置；通过构造 77 条可数据可回答的问题集实现评测。

**📈 对比分析**

与专家评估、LLM-as-Judge 的分数对比，结果显示整体精度高：专家准确率0.84、成功率0.77；AWS Hygiene 上最佳（准确率0.95、成功率0.90）。Fast‑path 与 full‑path 对比表明 full‑path 在复杂域（Okta、GWS）更稳健；示例适配与准确率呈正相关，fast‑path 相关性更强。

**⚠️ 局限性**

局限性包括：基准仅覆盖基础可见性任务，未涵盖跨系统关联、行为分析、风险评分等高级 ISPM 能力；fast‑path 评估缺乏推理轨迹；评估多依赖人工专家和 LLM 判定，仍可能存在主观偏差；仅在单一企业环境测试，缺乏多机构泛化验证。

---

## 99. From Fixed to Flexible: Shaping AI Personality in Context-Sensitive Interaction

**arXiv ID:** 2601.08194 | [PDF](https://arxiv.org/pdf/2601.08194v1)

**作者:** Shakyani Jayasiriwardene `[一作]` (University of Sydney), Zhanna Sarsenbayeva `[通讯]` (University of Sydney)

**通讯引用:** 1089 | [OpenAlex ID](https://openalex.org/A5024805223)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

调研用户在信息、情感、评估三种任务情境下如何实时调整聊天机器人八维人格，并评估其对信任与体验的影响

**💡 创新点**

首次将八维人格框架与实时滑块交互结合，系统性分析用户人格配置轨迹与情境依赖性

**🔧 技术方法**

基于GPT‑4.1 LLM的对话生成、Vue+Node实现的可调性界面、滑块配置与提示工程

**📊 数据集**

60名线上实验参与者产生的对话日志与问卷数据（共3×6轮对话）

**📈 对比分析**

通过潜在剖面分析与轨迹聚类对比任务前后配置，结果表明用户在所有情境中显著提升信任、满意度与交互连贯性

**⚠️ 局限性**

仅能每轮调整单一维度、实验仅为短期线上互动、缺乏长期持续使用与多模态交互的数据

---

## 100. GADPN: Graph Adaptive Denoising and Perturbation Networks via Singular Value Decomposition

**arXiv ID:** 2601.08230 | [PDF](https://arxiv.org/pdf/2601.08230v1)

**作者:** Hao Deng `[一作]`, Bo Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一种名为 GADPN 的轻量级图结构学习框架，通过自适应低秩去噪和泛化结构扰动对图拓扑进行精细化，从而提升 GNN 在多种图数据上的节点分类性能。

**💡 创新点**

主要创新点包括：① 使用贝叶斯优化自动确定 SVD 去噪的最佳秩，动态匹配不同图的同质性水平；② 将结构扰动方法从仅适用于无向对称图扩展到任意（甚至有向）图，借助奇异值分解提供理论依据；③ 将上述两步有机串联，形成一次性、高效的图增强流程。

**🔧 技术方法**

采用的核心技术包括：随机化奇异值分解（Randomized SVD）、贝叶斯优化（Gaussian Process + Expected Improvement）来选择秩、奇异值扰动理论、以及传统 GCN/GraphSAGE 等基准 GNN 作为下游模型。

**📊 数据集**

实验使用六个公开基准数据集：Cora、Citeseer、Pubmed（同质图）以及 Chameleon、Squirrel、Actor（异质/不均衡图）。

**📈 对比分析**

与 GCN、GraphSAGE、GAT、GEN 等主流基准进行对比，GADPN 在 5/6 个数据集上实现了最高或接近最高的准确率，尤其在 Chameleon、Squirrel、Actor 这类不均衡、噪声更严重的数据集上提升显著，证明了自适应去噪与结构扰动的协同效应。

**⚠️ 局限性**

局限性主要体现在：① 需要手动设定扰动比例 p、恢复参数 q、权重 α 等超参数，虽表现稳健但仍需经验调优；② 目前仅验证于静态图数据，动态或大规模图上的应用尚待探索；③ 低秩近似假设在极大稠密图中可能导致信息损失。

---

## 101. Is Sentiment Banana-Shaped? Exploring the Geometry and Portability of Sentiment Concept Vectors

**arXiv ID:** 2601.07995 | [PDF](https://arxiv.org/pdf/2601.07995v1)

**作者:** Laurits Lyngbaek `[一作]` (Aarhus University), Kenneth Enevoldsen `[通讯]` (Aarhus University)

**通讯引用:** 165 | [OpenAlex ID](https://openalex.org/A5087151587)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估并验证概念向量投影（CVP）在不同语域、时期、语言及情感维度下的可迁移性与线性假设。

**💡 创新点**

首次系统展示CVP在跨领域、跨语言和跨时代文本中的通用性，并对其线性假设进行几何分析。

**🔧 技术方法**

使用预训练句子嵌入模型（paraphrase-multilingual-mpnet-base-v2）构建概念向量，计算投影得分并进行z-score归一化；通过Spearman相关性评估模型表现。

**📊 数据集**

EmoBank、Facebook状态更新和Fiction4（19世纪至20世纪的英语/丹麦文学）三大数据集。

**📈 对比分析**

通过跨数据集Spearman相关性对比；在情感极性上平均相关性约0.66-0.70，跨域迁移保持较高一致性；对激活度与支配度的扩展评估相对较低但可接受。

**⚠️ 局限性**

仅使用单一嵌入模型；跨语言评估仅限英语和丹麦，未检验对不同语言家族的泛化；在激活度/支配度上使用相同数据集可能导致性能轻微高估。

---

## 102. Rescind: Countering Image Misconduct in Biomedical Publications with Vision-Language and State-Space Modeling

**arXiv ID:** 2601.08040 | [PDF](https://arxiv.org/pdf/2601.08040v1)

**作者:** Soumyaroop Nandi `[一作]` (University of Southern California), Prem Natarajan `[通讯]` (University of Southern California)

**通讯引用:** 7736 | [OpenAlex ID](https://openalex.org/A5066184920)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了 Rescind 数据集和 Integscan 框架，用视觉-语言模型指导的扩散合成和结构化状态空间模型实现了生物医学图像伪造的生成与精细检测。

**💡 创新点**

创新点包括：① 通过 VLM 与扩散模型实现语义驱动的逼真伪造生成；② 构建覆盖显微镜、凝胶、宏观、FACS 等四大模态的 600k+ 规模 Rescind 伪造基准；③ 在检测端引入结构化状态空间模型与 VLM 对齐的 Prompt‑Conditioned Fusion，实现精准的源/目标定位。

**🔧 技术方法**

使用了 BiomedCLIP、LLaVA‑Med 等视觉‑语言模型，MedSAM 分割，扩散式 inpainting，结构化状态空间模型 (SSM)，Attention 机制以及 Prompt‑Conditioned Fusion 等技术。

**📊 数据集**

主要使用 Rescind（含 Rescind‑P、Rescind‑T、Rescind‑I、Rescind‑G、Rescind‑V 五类子集）进行训练与评估，数据覆盖显微镜、凝胶/电泳、宏观、FACS 四大模态；同时对 Rescind‑T 真实重审集进行测试。

**📈 对比分析**

在 Rescind‑T、Rescind‑V 等公开基准上与传统特征方法、深度网络和 MONet 等前沿模型进行对比，Integscan 在图像级和像素级 F1、IoU 上均领先，尤其在外部/内部复制、切割/去除等多种伪造类型上表现突出，并在多种扰动攻击下保持优异鲁棒性。

**⚠️ 局限性**

主要局限包括：① 对 VLM 进行模态识别的依赖，模态误判会影响伪造语义一致性；② 生成的伪造样本难以覆盖多步、跨面板的复合伪造；③ 共享编码器限制了对不同伪造任务的专属特征学习；未来需要改进 prompt 细化、检索/强化学习提示以及更丰富的复合伪造数据。

---

## 103. Cost and accuracy of long-term graph memory in distributed LLM-based multi-agent systems

**arXiv ID:** 2601.07978 | [PDF](https://arxiv.org/pdf/2601.07978v1)

**作者:** Benedict Wolff `[一作]` (KTH Royal Institute of Technology), Jacopo Bennati `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个可扩展的分布式多智能体系统（DMAS）测试平台，并在无约束与受限网络条件下对比评估两种长期记忆框架 mem0（向量存储）和 Graphiti（图数据库）在成本、效率与准确率上的表现。

**💡 创新点**

首次系统地将长期记忆框架与完整的多智能体系统结合，并在同一基准上引入统计 Pareto 效率分析，以在成本和准确率之间寻找客观的最优点；同时揭示了在分布式环境中网络约束对成本影响微乎其微的事实。

**🔧 技术方法**

核心技术包括：1）基于 mem0 的向量压缩记忆；2）基于 Neo4j 的 Graphiti 知识图；3）LLM Qwen2.5 与 GPT‑4o‑mini 进行问答；4）Docker、WSL 与 AWS Fargate 进行资源监测与计费；5）多维度性能度量（CPU、RAM、磁盘、网络、token 计数）。

**📊 数据集**

使用了 LOCOMO 长上下文基准（包含 10 场对话、19 个会话、199 题）作为记忆加载与问答测试数据集。

**📈 对比分析**

通过因素设计实验，分别测量两阶段（加载、问答）在两种网络配置下的 token 使用、美元成本、计算与网络资源消耗及答案准确率。结果显示：mem0 在加载阶段比 Graphiti 快 86%+、占用 CPU/内存/网络显著更低；在问答阶段速度也优于 Graphiti；token 与金钱成本差异显著，Graphiti 约高 30-40%；但准确率差异无统计学显著性，均在 7–11% 左右。

**⚠️ 局限性**

主要限制包括：1）DMAS 的行为被预先固定，缺乏自主探索和动态决策；2）网络约束仅为单一低延迟宽带场景，未覆盖更复杂的 P2P 或边缘互联；3）大量“我不知道”回答，未深入分析其根源；4）仅对两种记忆框架进行比较，缺少更广泛的基准和对比基线。

---

## 104. Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs

**arXiv ID:** 2601.08198 | [PDF](https://arxiv.org/pdf/2601.08198v1)

**作者:** Yibo Wang `[一作]` (Nanjing University), Lijun Zhang `[通讯]` (Nanjing University)

**通讯引用:** 28060 | [OpenAlex ID](https://openalex.org/A5115603699)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过自我对弈的方式对大语言模型进行细调，使用三元组（真实、合成、原始合成）并引入熵约束，以稳定训练并解决奖励与生成指标的不一致。

**💡 创新点**

创新点在于①利用历史优势（合成对原始合成的优势）来保证即使当前优势消失仍可稳定优化；②在自我对弈中加入熵约束，从而消除对参考策略的依赖，实现训练与生成的一致性。

**🔧 技术方法**

采用基于大语言模型的自我对弈框架，使用对数似然、熵约束以及三元组损失来更新策略，理论上与原始自我对弈相比更稳健。

**📊 数据集**

采用Zephyr‑7B和Mistral‑7B预训练模型，在Ultrachat‑200k中随机抽取5万条人工标注样本进行实验。

**📈 对比分析**

与传统SFT以及原始自我对弈相比，Triplet‑SP在数学推理、指令跟随等10个任务上保持更高的分数，迭代过程更稳定；仅用25%样本即可达到甚至超过全量SFT。

**⚠️ 局限性**

限制在于每轮迭代需重新生成所有合成样本，且训练数据固定不适应目标分布的漂移，需要设计过滤策略或在线学习方法。

---

## 105. Hierarchical Sparse Plus Low Rank Compression of LLM

**arXiv ID:** 2601.07839 | [PDF](https://arxiv.org/pdf/2601.07839v1)

**作者:** Pawan Kumar `[一作]` (International Institute of Information Technology), Aditi Gupta `[通讯]` (International Institute of Information Technology)

**通讯引用:** 1392 | [OpenAlex ID](https://openalex.org/A5081505598)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型（尤其是LLaMA‑7B的注意力投影层）提出一种分层稀疏+低秩压缩方法（sHSS和sHSS‑RCM），先提取大幅度权重做稀疏化，再对残差进行递归Hierarchically Sparse Separable（HSS）低秩分解；

**💡 创新点**

创新点在于将稀疏剪枝与多层HSS低秩分解统一到一个框架中，并加入Reverse Cuthill‑McKee（RCM）重排来进一步压缩跨块低秩，显著提升压缩后模型的困惑度与存储效率；

**🔧 技术方法**

使用稀疏矩阵、RCM重排、HSS分层低秩分解（递归SVD/随机SVD）、CUDA加速的FP16推理实现；

**📊 数据集**

以WikiText‑103测试集评估模型困惑度；在LLaMA‑7B上仅压缩1.6B参数的注意力投影层；

**📈 对比分析**

与稀疏+SVD、稀疏+随机SVD等基线对比，30%稀疏、外层秩512时，sHSS‑RCM实现PPL 1.64，存储压缩约1.7×，相较基线保持或提升了PPL；

**⚠️ 局限性**

局限性：仅在注意力层实验，未验证对MLP或更大模型的适用性；RCM对压缩效果提升有限；未结合量化或动态秩调整；训练时对参数的优化仍需进一步研究；

---

## 106. The Impact of AI Generated Content on Decision Making for Topics Requiring Expertise

**arXiv ID:** 2601.08178 | [PDF](https://arxiv.org/pdf/2601.08178v1)

**作者:** Shangqian Li `[一作]` (University Of Queensland), Gianluca Demartini `[通讯]` (University Of Queensland)

**通讯引用:** 4524 | [OpenAlex ID](https://openalex.org/A5052565959)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过实验室实验结合问卷和访谈，探究了受试者在面对普通和专业化议题时，领域知识、AI生成内容（AIGC）与人类撰写内容对其在线决策与观点变化的影响。

**💡 创新点**

创新点包括：①首次系统比较AIGC与UGC在专业领域议题中的影响；②采用GLMM模型量化领域知识、性别、信息帮助度等多重因素对意见变化的贡献；③通过源信息敏感与不敏感两种模式，检验信息来源对决策的调节作用。

**🔧 技术方法**

使用的技术主要有：Qualtrics在线问卷；JavaScript记录点击与阅读时间；生成式AI（ChatGPT）生成内容；信任度与技术亲和度量表（TIA、ATI）；统计建模工具R中的GLMM及AIC/BIC模型比较。

**📊 数据集**

数据集包括：30名大学生（本科、硕士、博士）共240条数据（8题×5条信息×2次确认），以及通过人工审核的AIGC与UGC文本（每条文本评分>4）。

**📈 对比分析**

方法比较：通过GLMM对二元意见变化和连续自信度变化建模，利用AIC/BIC、chi-square检验比较基线、全模型与精炼模型。结果显示精炼模型显著优越，R²约为0.47（意见变化）和0.44（自信度）；表明领域知识和信息帮助度显著影响决策，AIGC与UGC对决策帮助相当。

**⚠️ 局限性**

局限性：①样本仅为大学生，缺乏更广泛的年龄、职业与文化多样性；②仅考察化学与非化学两大领域，难以推广到其他专业；③AIGC质量控制流程不够完善，可能存在细微偏差；④实验时间短且受限于实验室环境，未覆盖真实社交媒体动态。

---

## 107. Cognitive Biases in LLM-Assisted Software Development

**arXiv ID:** 2601.08045 | [PDF](https://arxiv.org/pdf/2601.08045v1)

**作者:** Xinyi Zhou `[一作]` (University of Southern California), Souti Chattopadhyay `[通讯]` (University of Southern California)

**通讯引用:** 323 | [OpenAlex ID](https://openalex.org/A5059243154)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对LLM辅助软件开发中的认知偏差进行混合方法研究，构建了15类偏差的分类法。

**💡 创新点**

首次系统识别并量化LLM互动中出现的90种认知偏差，并提出相应的缓解策略。

**🔧 技术方法**

采用观察研究、访谈和问卷调查三种方法，结合定性编码与统计检验。

**📊 数据集**

收集了14名学生和专业开发者的实际编码日志（共2013条动作，808条LLM互动），并对22名开发者进行情景问卷。

**📈 对比分析**

通过统计检验发现LLM相关动作更易出现偏差（56.4%）且更易被逆转（29.5%），与传统工作相比，偏差率显著提高；不涉及模型性能。

**⚠️ 局限性**

样本量有限、项目不具备代表性、仅基于自选项目、缺乏长期追踪。

---

## 108. Dynamic Graph Structure Learning via Resistance Curvature Flow

**arXiv ID:** 2601.08149 | [PDF](https://arxiv.org/pdf/2601.08149v1)

**作者:** Chaoqun Fei `[一作]` (South China Normal University), Tianyong Hao `[通讯]` (Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于有效电阻的可微分几何流——Resistance Curvature Flow (RCF)，并实现了动态图结构学习算法 DGSL-RCF，用于提升深度度量学习、流形学习和图结构学习中的表示质量和下游性能。

**💡 创新点**

创新点在于将电路中的有效电阻概念转化为图的曲率度量，替代传统的Ollivier-Ricci曲率，显著降低计算复杂度（超过100倍加速），同时保持或超过OCF的拓扑辨识能力，并提供三种可嵌入深度网络的范式（预处理、隐藏层正则、输出精炼）。

**🔧 技术方法**

技术包括：有效电阻曲率定义、RCF 的离散动力学方程、基于稀疏Cholesky/共轭梯度的线性系统求解、可微分矩阵运算实现以及在PyTorch中的端到端反向传播；算法对图 Laplacian 的微扰实现了高效曲率计算。

**📊 数据集**

使用的公开数据集包括：深度度量学习的 CUB‑200‑2011、Cars‑196、SOP；流形学习的 MNIST、USPS、Medical‑MNIST、KVASIR 以及 Swiss‑Roll、S‑Curve、Truncated‑Sphere、Gaussian Surface；图结构学习的 Wine、Cancer、Digits、20News。

**📈 对比分析**

与传统 k‑NN、OGC‑OCF、LLE/HLL/LEM 等基线相比，DGSL‑RCF 在 DML 上 NMI/F1/Recall 提升约 5–10%，在 ML 上 ACC/NMI/ARI 提升 2–8%，在 GSL 上准确率提升 1–3%；在时间上与 OCF 比较，RCF 在相同迭代下速度提升 50–280 倍，且在 GPU 单卡上即可完成。

**⚠️ 局限性**

局限性包括：仍需手动调节邻域大小 k、迭代次数 n_iter 与学习率 η，算法在极大规模（百万级节点）图上仍受内存限制；对高度异质或极稀疏图的鲁棒性未完全验证，未来需进一步研究自适应流速和分布式实现。

---

## 109. Decentralized Firmware Integrity Verification for Cyber-Physical Systems Using Ethereum Blockchain

**arXiv ID:** 2601.08091 | [PDF](https://arxiv.org/pdf/2601.08091v1)

**作者:** S M Mostaq Hossain `[一作]` (Tennessee Technological University), Amani Altarawneh `[通讯]` (Tennessee Technological University)

**通讯引用:** 166 | [OpenAlex ID](https://openalex.org/A5060400318)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了基于Ethereum公链的去中心化固件完整性验证框架，利用SHA‑256哈希在智能合约中存储并通过Python客户端完成固件注册与实时验证。

**💡 创新点**

在公共测试网上完整实现链上固件哈希验证并公开部署，首次展示在CPS场景下的透明、可审计的固件完整性保障；同时提出Layer‑2 rollup和IPFS等扩展方案以提升可扩展性。

**🔧 技术方法**

采用Ethereum智能合约（Solidity）、Sepolia测试网、Web3.py + Infura节点、Python脚本、SHA‑256哈希、IPFS、Layer‑2 rollup（讨论）等技术。

**📊 数据集**

使用真实的Arduino Mega 2560 R3固件HEX文件作为实验数据集。

**📈 对比分析**

通过在Sepolia上测量gas消耗、确认时延与吞吐量，验证每条注册交易约78,200 gas（≈$0.12）且确认延迟≈14.6秒，验证吞吐9–10 tx/min；与传统集中式哈希数据库对比显示更低成本、去中心化、透明且易审计。

**⚠️ 局限性**

局限性包括：Gas费用对大规模IoT场景成本较高；链上确认延迟不适合极高实时需求；仅在测试网上验证，主网或Layer‑2实现尚未完成；对并发扩展与多固件版本管理等功能仍待完善。

---

## 110. Multi-Objective Optimization for Joint Communication and Sensing in Multi-user MIMO Systems: Characterizing the Pareto Boundary

**arXiv ID:** 2601.08152 | [PDF](https://arxiv.org/pdf/2601.08152v1)

**作者:** Thakshila Perera `[一作]` (University of Manitoba), Ekram Hossain `[通讯]` (University of Manitoba)

**通讯引用:** 33927 | [OpenAlex ID](https://openalex.org/A5089270885)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对多天线基站同时进行通信和雷达感知（JCAS）系统，作者提出了多目标优化框架，以加权求和的形式同时最大化通信的互信息（MI）和感知的费舍尔信息（FI），并通过上行-下行双向性、块坐标上升以及投影梯度下降（PGD）方法求解多用户和单用户两种场景下的最优波束成形。

**💡 创新点**

① 统一考虑 MI 与 FI 的 Pareto 边界，实现真正的联合波束成形而非传统的分离设计；② 在多用户场景下首次利用上行-下行双向性将非凸问题转化为凸形式，并给出解析的 Lagrangian/线搜索求解；③ 在单用户场景引入等效等向辐射功率（EIRP）约束，并使用 PGD + KKT 进行约束下的闭式优化。

**🔧 技术方法**

使用了以下技术：多目标凸优化、上行-下行双向性、块坐标上升、Lagrangian 以及线搜索、投影梯度下降（PGD）、KKT 条件、矩阵特征值分解（EVD）、奇异值分解（SVD）以及 MATLAB CVX 进行数值验证。

**📊 数据集**

该工作主要基于仿真，采用 IEEE 802.11 / LTE / 5G 的多路径信道模型，设置多用户（K=1,4,6,8）、天线数（N_t=N_r=4,10,16,20）、发射功率、EIRP、噪声方差等参数；没有使用公开数据集，所有结果均为基于自定义信道仿真产生的。

**📈 对比分析**

通过与子最优（K+1 波束）以及不考虑感知的单独波束设计进行对比，作者展示了：① 在多用户场景下，联合波束成形显著提升 MI 与 FI 的 Pareto 边界；② 随着用户数或天线数增大，最优与子最优的性能差距缩小；③ 在单用户 EIRP 限制下，PGD 解与 CVX 结果高度一致，说明所用方法有效；整体性能显示 MI 与 FI 可以在可接受的范围内平衡，并随着功率、天线数或 EIRP 增大而提升。

**⚠️ 局限性**

主要限制包括：仅考虑单一目标（单点雷达），假设通信用户间及用户与目标间不存在干扰，使用的是理想化的高斯信道模型；未考虑多目标、多径雷达误差、双向天线互调等实际系统干扰；方法对极端高用户数或极大天线阵列时的计算复杂度（O(KN_t^3)）较高；未来需要扩展至双向/多目标系统并考虑实际硬件约束。

---

## 111. Instance-Aligned Captions for Explainable Video Anomaly Detection

**arXiv ID:** 2601.08155 | [PDF](https://arxiv.org/pdf/2601.08155v1)

**作者:** Inpyo Song `[一作]` (SungKyunKwan University), Jangwon Lee `[通讯]` (SungKyunKwan University)

**通讯引用:** 6044 | [OpenAlex ID](https://openalex.org/A5033845008)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了实例对齐的描述框架，用于可解释的视频异常检测；

**💡 创新点**

创新点是将文本描述与每个对象实例的分割掩码绑定，形成完整的 who–what–whom–where 解释；

**🔧 技术方法**

采用 SAM2 做实例分割与跟踪，结合 LLM/VLM（如 Qwen3‑VL、InternVL3）生成对象级别字幕，并通过 GPT‑4o‑mini 评估；

**📊 数据集**

在八个主流 VAD 数据集（UCF‑Crime、ShanghaiTech、UBnormal、Avenue、Ped1、Ped2、DoTA）以及扩展的 VIEW360+ 上进行注释与评测；

**📈 对比分析**

与单模态、联合模型以及多阶段 VLM+SAM2 基线对比，结果显示多阶段管道在 F_SC（字幕质量与定位的调和平均）上优于现有方法，且错误实体率低；

**⚠️ 局限性**

局限包括对细微异常的定位仍不够准确、对受害者/目标的描述与分割精度低，以及在低分辨率场景下模型性能下降。

---

## 112. InfGraND: An Influence-Guided GNN-to-MLP Knowledge Distillation

**arXiv ID:** 2601.08033 | [PDF](https://arxiv.org/pdf/2601.08033v1)

**作者:** Amir Eskandari `[一作]` (Queen's University), Farhana Zulkernine `[通讯]` (Queen's University)

**通讯引用:** 2510 | [OpenAlex ID](https://openalex.org/A5063480277)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一种基于节点影响度的图知识蒸馏框架InfGraND，能够将GNN教师的知识高效迁移给无消息传递的MLP学生，并通过一次性多跳邻域特征预计算进一步降低推理成本。

**💡 创新点**

创新点在于①引入图结构感知的节点影响度指标，用以识别对全局表示贡献最大的节点，取代传统的基于预测不确定性的均匀或不均匀蒸馏；②通过影响度加权的子图蒸馏损失和影响度引导的监督损失实现非均匀蒸馏；③采用一次性特征传播实现结构信息注入，避免推理时额外开销。

**🔧 技术方法**

核心技术包括使用GCN/GAT/GraphSAGE教师、MLP学生；节点影响度通过二阶传播后余弦相似度计算并归一化；采用影响度加权交叉熵与KL蒸馏损失；一次性多跳特征聚合（平均池化）作为结构输入；在转导与归纳场景下进行实验。

**📊 数据集**

实验数据集涵盖七个同质图：Cora、Citeseer、Pubmed、Amazon-Photo、CoAuthor-CS、CoAuthor-Phy以及大规模OGBN-Arxiv。

**📈 对比分析**

与GLNN、KRD、HGMD、FF-G2M、Vanilla MLP及教师模型等基线比较，InfGraND在转导和归纳两种设置下均实现了显著提升，平均提升12.6%（转导）和9.3%（归纳），在OGBN‑Arxiv大规模数据上更是提升19.5%/9.5%，且在部分任务甚至超越教师模型。

**⚠️ 局限性**

主要局限包括：只验证了同质图，未对异构或动态图场景进行评估；节点影响度计算在极大图上可能带来一定预处理成本；模型对关键超参数（γ₂、δ₂、λ、传播步数等）敏感；尚未结合预测不确定性或其他多维度重要性度量。

---

## 113. Simulations for Augmented Reality Evaluation for Mass Casualty Incident Triage

**arXiv ID:** 2601.08186 | [PDF](https://arxiv.org/pdf/2601.08186v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 114. Delaunay Triangulations with Predictions

**arXiv ID:** 2601.08106 | [PDF](https://arxiv.org/pdf/2601.08106v1)

**作者:** Sergio Cabello `[一作]` (University of Ljubljana), Panos Giannopoulos `[通讯]` (City University of London)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

针对二维Delaunay三角剖分问题，研究了预测增强算法，给出了多种针对不同错误度量的时间复杂度（包括确定性 O(n+log³n)、随机化 O((n+log n)log* n) 以及最优随机化 O(n+log n)）以及在概率模型下的 O(n loglog n + n log(1/ρ)) 复杂度。

**💡 创新点**

首次把预测（prediction）概念引入几何算法，提出了利用预测“误差”D、随机子样本R、以及冲突列表的技术，实现了当预测接近真解时可以突破 O(n log n) 最坏情况的算法框架；并给出了一套通用的分析工具（r‑division、Planar 分离器、指数衰减引理等），可推广到最小生成树、相对邻域图等相关几何结构。

**🔧 技术方法**

核心技术包括：
- r‑division 与 planar graph separator 处理输入三角剖分；
- 随机采样与冲突列表（conflict list）技术；
- 迭代采样与指数衰减，控制冲突列表大小；
- 通过局部 Delaunay 检查快速验证大区域；
- 结合传统的 Delaunay 递归合并与最小生成树算法；
- 复杂度分层分析与 log* n 的消除。

**📊 数据集**

本文为理论性研究，没有使用实际点集；所有实验与比较均基于理论分析和已知的最坏情况对照。

**📈 对比分析**

与经典 O(n log n) 最坏情况算法相比，若预测误差 D 远小于 n 或在概率模型下采样率 ρ 较大，本文提出的算法可在 O(n+log n) 或近似线性时间内完成；在最坏情况下仍保持 O(n log n)。在 D 约为 n/log n 时，最优随机化算法已达到线性时间，显示出显著性能提升。

**⚠️ 局限性**

主要限制包括：
- 一些算法仍包含 log* n 或 loglog n 因子；
- 对 D 的上界和下界在极端情况下（如 D=Ω(n)）仍未达到最优；
- 对更复杂的预测模型（如自适应错误分布）尚未给出完整分析；
- 证明中依赖多项式级别的常数与分割技术，实际实现的细节尚待验证。

---

## 115. Monte Carlo to Las Vegas for Recursively Composed Functions

**arXiv ID:** 2601.08073 | [PDF](https://arxiv.org/pdf/2601.08073v1)

**作者:** Bandar Al-Dhalaan `[一作]`, Shalev Ben-David `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究递归复合布尔函数的查询复杂度，定义并证明了组合极限的收敛性；

**💡 创新点**

提出零误差随机查询复杂度的组合极限等于最大值(max{^*(f),^*(f)})，实现 Monte Carlo 算法升格为 Las Vegas，并推广到量子算法；

**🔧 技术方法**

使用组合极限分析、归约技术、证书查找量子/随机算法以及多项式度数、敏感度等组合性质证明；

**📊 数据集**

本研究为纯理论工作，未使用实验数据或特定数据集；

**📈 对比分析**

通过理论推导与已有上下界比较，得到 _0^*(f)=max{^*(f),^*(f)} 等关系，表明在递归函数上零误差与有误差算法的复杂度相当；

**⚠️ 局限性**

结果仅适用于可递归复合的函数，缺乏对非递归函数的推广，且量子/随机查询复杂度的上界仍保留常数因子。

---

## 116. Reasoning over Precedents Alongside Statutes: Case-Augmented Deliberative Alignment for LLM Safety

**arXiv ID:** 2601.08000 | [PDF](https://arxiv.org/pdf/2601.08000v1)

**作者:** Can Jin `[一作]` (Rutgers University), Dimitris N. Metaxas `[通讯]` (Rutgers University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过引入案例增量的推理过程（Case‑Augmented Deliberative Alignment, CADA）来改进开源大语言模型的安全对齐。研究系统评估了在推理时和训练时使用安全代码与案例两种方式的效果，并最终提出了一种基于自生成安全推理链的强化学习方法，显著提升了模型的无害性、鲁棒性和过度拒绝率，同时保持了原有的实用性能。

**💡 创新点**

创新点在于：① 将安全规范从单纯的“代码”转向结合案例的“先例”形式，以更好地捕捉上下文依赖的安全决策；② 设计了一种不依赖大规模人工标注或过滤数据、仅使用自生成推理链的强化学习框架；③ 引入 KL‑惩罚机制，平衡安全推理与模型原始实用性，减少对有益请求的误拒。

**🔧 技术方法**

技术包括：
- Deliberative Alignment（推理先行）
- 监督微调（SFT）
- 对抗性偏好优化（DPO）
- 基于 REINFORCE 的强化学习
- KL‑距离惩罚
- 简单格式化奖励函数
- GPT‑4o 作为奖励模型与评估工具

**📊 数据集**

数据集：
- 500 条来自 Beavertails 的有害请求（含可选类别标签）
- 评估基准：StrongREJECT、AdvBench、HEx‑PHI（无害性/鲁棒性）
- 友好性评估：XSTest
- 实用性评估：GSM8K、BBH、MMLU
- 对抗攻击：PAIR、PAP

**📈 对比分析**

对比方法：
- 原始模型（w/o SR）
- 仅直接拒绝训练（SFT w DR）
- 代码推理训练（SFT w _c）
- 案例增量推理训练（SFT w CR）
- 案例增量推理 DPO（DPO w CR）
- 提出的 CADA（RL‑on‑self‑generated chains）
结果显示：
- CADA 在 StrongREJECT、AdvBench、HEx‑PHI 的攻击成功率均显著降低，且无害性分数最高；
- 在 XSTest 的过度拒绝率最小，友好性提升；
- 对 GSM8K、BBH、MMLU 的零/少样本推理准确率与原始模型基本持平，表明实用性未受损；
- 与 SFT w CR/ DPO w CR 相比，CADA 在安全性上更强，在友好性上也更优。

**⚠️ 局限性**

局限性：
- 安全与友好性的边界仍主观，模型在不同应用场景下可能需要重新调优案例分布；
- 仅在已知的 jailbreak 攻击上验证鲁棒性，未知或未来的对抗策略仍可能突破；
- 自生成推理链可能继承预训练数据中的偏见，且奖励设计简化后仍可能忽略细粒度推理质量；
- 需要依赖 GPT‑4o 等强大评估模型，成本相对较高。

---

## 117. Scalable Multiagent Reinforcement Learning with Collective Influence Estimation

**arXiv ID:** 2601.08210 | [PDF](https://arxiv.org/pdf/2601.08210v1)

**作者:** Zhenglong Luo `[一作]` (University of Newcastle), Ke Pan `[通讯]` (Central South University)

**通讯引用:** 7203 | [OpenAlex ID](https://openalex.org/A5034211613)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于SAC的多智能体强化学习框架CIEN-SAC，利用集体影响估计网络（CIEN）在通信受限环境下实现三臂协同抬物任务。

**💡 创新点**

创新点在于用低维的集体影响因子代替显式对手动作/策略建模，保持网络尺寸不随智能体数量增长，从而实现可扩展性；CIEN仅依赖任务对象状态和局部观测，无需额外通信。

**🔧 技术方法**

采用Soft Actor–Critic（SAC）算法与CIEN模块集成；使用多臂协同抬物环境的MuJoCo/RoboSuite仿真与真实UR5e机器人；实现分布式、去中心化策略学习。

**📊 数据集**

数据集主要来自仿真：MuJoCo中的三臂抬物任务；RoboSuite环境；以及在实际UR5e机器人平台上采集的视觉标记状态数据。

**📈 对比分析**

通过与完全通信的集中式SAC、去中心化SAC（无CIEN）和加入对象状态的基线进行对比。实验显示CIEN-SAC在收敛速度和最终抬升高度上与集中式SAC相当，优于无CIEN的去中心化SAC；在真实机器人上亦保持稳定性能，且对观测噪声鲁棒。

**⚠️ 局限性**

局限性包括：训练仍相对耗时，主要针对协同任务；对大规模多智能体或非协同/混合交互情形的有效性尚未验证；模型对任务对象状态精度敏感，若对象观测极其不准可能影响性能。

---

## 118. From Prompts to Deployment: Auto-Curated Domain-Specific Dataset Generation via Diffusion Models

**arXiv ID:** 2601.08095 | [PDF](https://arxiv.org/pdf/2601.08095v1)

**作者:** Dongsik Yoon `[一作]` (HDC LABS), Jongeun Kim `[通讯]` (HDC LABS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `e0540dec-d77f-42db-94ae-d039248f6393` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个三阶段自动化管线，用扩散模型在域特定背景图中合成目标对象，并通过多模态评估（检测、审美、视觉‑语言对齐）进行筛选，最终生成高质量的合成数据集。

**💡 创新点**

创新点在于将扩散模型生成与多模态质量评估结合，并引入用户偏好分类器，使得仅需极少标注即可实现个性化筛选与高质量数据集构建。

**🔧 技术方法**

使用了扩散模型进行图像填充、目标检测器、审美评分模型、视觉‑语言模型以及基于DINOv2+ConvNeXt的偏好分类器等技术。

**📊 数据集**

使用了电梯闭路电视视角的域特定背景图与犬类目标对象的图像，并在训练阶段使用约289张标注样本。

**📈 对比分析**

与传统CNN骨干（VGG‑19、ViT‑Large/16、CLIP）相比，DINOv2偏好分类器在F1上达到0.8427，优于其他模型，验证了方法的有效性。

**⚠️ 局限性**

局限性包括仅在单一域-对象对（电梯‑犬）上验证，仍需一定背景图样本，依赖现成预训练模型且缺乏对偏差的系统分析。

---

## 119. Subspace Alignment for Vision-Language Model Test-time Adaptation

**arXiv ID:** 2601.08139 | [PDF](https://arxiv.org/pdf/2601.08139v1)

**作者:** Zhichen Zeng `[一作]` (University of Illinois Urbana-Champaign), Hanghang Tong `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 17136 | [OpenAlex ID](https://openalex.org/A5068043486)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于子空间对齐的测试时自适应框架，修复视觉‑语言模型在分布偏移下的模态差距与视觉噪声问题。

**💡 创新点**

创新点在于：①用弦距（chordal distance）在Grassmannian上对视觉与文本子空间进行无秩约束对齐；②在对齐后的子空间投影视觉特征，过滤掉任务无关噪声，从而显著提升零样本预测质量。

**🔧 技术方法**

核心技术包括：子空间提取（SVD/特征协方差分解）、弦距子空间对齐、语义投影、以及标准的自监督 TTA 目标（如交叉熵、聚类损失）。

**📊 数据集**

使用的公开数据集为 CIFAR‑10‑C、CIFAR‑100‑C 与 ImageNet‑C，涵盖 15 种图像失真类型。

**📈 对比分析**

与多种基准 TTA 方法（训练基、训练无、增强、聚类等）对比，所提出的方法在三大基准上平均提升 2.24% 以上，且在强度最高的失真层面几乎消除了负适应现象，成为新的 state‑of‑the‑art。

**⚠️ 局限性**

局限性包括：①需要批量统计信息，难以直接用于单样本（episodic）设置；②需要对模型梯度可访问，无法直接应用于闭源 API；③SVD 计算带来一定的运行时开销。

---

## 120. 3DGS-Drag: Dragging Gaussians for Intuitive Point-Based 3D Editing

**arXiv ID:** 2601.07963 | [PDF](https://arxiv.org/pdf/2601.07963v1)

**作者:** Jiahua Dong `[一作]` (University of Illinois Urbana-Champaign), Yu-Xiong Wang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 6579 | [OpenAlex ID](https://openalex.org/A5102952938)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于3D Gaussian Splatting的直观拖拽编辑框架，用户只需指定3D手柄点和目标点，即可完成几何相关的3D内容编辑；

**💡 创新点**

创新点包括：①采用基于3D Gaussian的复制‑粘贴变形策略，省去先验几何需求；②结合扩散模型进行视觉内容纠正，提升3D一致性；③使用多步编辑调度与历史感知扩散微调，支持更激进的拖拽操作；

**🔧 技术方法**

主要技术包括：3D Gaussian Splatting、变形引导、扩散引导（Dreambooth + LoRA）、多视角渲染、局部编辑掩码、Annealed 数据集编辑、梯度下降优化；

**📊 数据集**

使用公开数据集：Instruct‑NeRF2NeRF、PDS、Mip‑NeRF360、Tank & Temple 等，共八个场景进行实验；

**📈 对比分析**

与 Instruct‑NeRF2NeRF、Deformation、PDS、SDE‑Drag 等基线对比，显示在几何编辑、细节保留、视觉一致性方面均优于对手；单次编辑耗时约15分钟（含20步迭代），显著快于传统方法；

**⚠️ 局限性**

局限性：对目标物体视角有限或过小的场景效果下降；极端激进拖拽可能导致模型退回原始状态；依赖扩散模型的质量与训练数据，可能在复杂多尺度场景中表现不佳；

---

## 121. Executable Ontologies in Game Development: From Algorithmic Control to Semantic World Modeling

**arXiv ID:** 2601.07964 | [PDF](https://arxiv.org/pdf/2601.07964v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 122. Fiducial Exoskeletons: Image-Centric Robot State Estimation

**arXiv ID:** 2601.08034 | [PDF](https://arxiv.org/pdf/2601.08034v1)

**作者:** Cameron Smith `[一作]` (USC Physical Superintelligence Lab), Yue Wang `[通讯]` (USC Physical Superintelligence Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种通过在每个机器人关节上安装3D打印的基准外骨骼与基准标记（fiducial markers），利用单张RGB图像直接估计所有链条的6D位姿，并通过全局优化与前向运动学求解关节状态与相机外参，实现低成本机器人（≈$100）的精准状态估计与控制。

**💡 创新点**

创新点在于：①将基准外骨骼与已知标记–关节变换集成，消除传统手眼标定和迭代收集；②使用单张图像直接得到每个关节的完整SE(3)位姿；③在估计后加入状态基准微调（delta refinement）实现更精细的运动控制。

**🔧 技术方法**

技术手段包括：ArUco/Marker 6D姿态估计、3D打印外骨骼设计、SE(3)关节位姿回归、前向运动学与非线性优化（L-BFGS）求解关节角度、基于估计误差的增量控制循环。

**📊 数据集**

数据集为自制低成本6-DoF机器人（SO‑100）在多种姿态下拍摄的RGB图像，涵盖全遮挡、部分遮挡（1/3/4个标记）与倒置姿态等多样场景。

**📈 对比分析**

与传统前向运动学（Just Enc.）及带/不带编码器初始化（Ours Enc./No‑Enc.）比较，平均末端执行器SE(3)误差下降≈75%，控制误差下降≈45%；与基于可微渲染的Dr. Robot相比，关节误差下降≈98%，相机平移/旋转误差分别下降≈99%，推理时间缩短约100倍。

**⚠️ 局限性**

局限性包括：需要外部相机能够同时看到基座标记和至少一个其他标记；每个关节都需手动安装外骨骼标记，工作量大且对视觉遮挡敏感；标记尺寸较大，可能遮挡机器人视觉效果；对极度遮挡或无可见标记的场景估计不稳。

---

## 123. The End of Reward Engineering: How LLMs Are Redefining Multi-Agent Coordination

**arXiv ID:** 2601.08237 | [PDF](https://arxiv.org/pdf/2601.08237v1)

**作者:** Haoran Su `[一作]` (New York University), Congjia Yu `[通讯]` (Lerna AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过使用大语言模型（LLM）将自然语言目标自动转换为多智能体强化学习（MARL）的奖励函数，并提出动态适应与人类对齐的框架，重新定义奖励工程流程。

**💡 创新点**

①将奖励从手工数值映射转为自然语言描述，减少翻译损失；②利用LLM实现奖励的在线自适应更新（CARD风格）；③通过可验证的语言奖励（RLVR）提升人类意图的可解释性与一致性。

**🔧 技术方法**

使用GPT‑4/Claude‑3等大型语言模型进行奖励生成与评估；基于语言的奖励反射与迭代优化；采用RLVR框架进行可验证奖励学习；对多智能体环境进行模拟训练。

**📊 数据集**

主要使用公开的 MARL 基准：Multi‑Agent MuJoCo、SMARTS 交通交叉口、IsaacGym 机器人抓取/运动任务等；并引入新颖的环境变体以避免模型记忆泄漏。

**📈 对比分析**

与手工设计奖励、逆强化学习、进化搜索等方法在协作效率、样本效率和设计时间上进行比较。实验设计包含 5 大实验，预期LLM生成奖励在任务成功率、协作效率上接近或优于专家设计，同时设计时间显著下降；动态适应实验预期适应速度可与人工重新调参相当或更快。

**⚠️ 局限性**

主要局限包括：LLM 推理成本高、可能产生幻觉/不安全奖励、语言歧义导致奖励多样化、对大规模智能体的可扩展性尚未验证、缺乏统一的评估基准与安全验证机制。

---

## 124. MLLM-VADStory: Domain Knowledge-Driven Multimodal LLMs for Video Ad Storyline Insights

**arXiv ID:** 2601.07850 | [PDF](https://arxiv.org/pdf/2601.07850v1)

**作者:** Jasmine Yang `[一作]` (Meta), Shawndra Hill `[通讯]` (Meta)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出基于多模态大型语言模型的 MLLM‑VADStory 框架，用广告域知识驱动，系统量化视频广告的叙事结构并生成可操作的洞察。

**💡 创新点**

①首创将 LLM 与广告专业知识相结合，构建功能角色分类法拆分视频；②基于功能意图生成广告叙事结构；③在大规模广告上提供行业细分的最佳故事结构推荐。

**🔧 技术方法**

使用多模态视频分割（视觉 + Whisper 语音），LLama‑Video‑LLM 和 Llama‑4‑Scout 进行故事检测与功能标签，XGBoost 预测绩效提升，并结合人工校验。

**📊 数据集**

约 5 万条美国社交媒体视频广告（15–60 秒），覆盖服装、美容、食品、饮料四个子垂直，英文广告为主。

**📈 对比分析**

通过 OLS 和 XGBoost 的部分依赖分析，发现包含故事的广告前 2 秒可提升 5–10% 的滞留率；不同故事结构对 CTR、CVR 也有显著正向提升，效果因行业而异。

**⚠️ 局限性**

局限：仅针对英文、15–60 秒的视频；功能标签需人工复核，LLM 解释可能不准确；缺乏跨语言、跨文化验证，未探讨对短视频平台推荐系统的直接影响。

---

## 125. Representation Learning with Semantic-aware Instance and Sparse Token Alignments

**arXiv ID:** 2601.08165 | [PDF](https://arxiv.org/pdf/2601.08165v1)

**作者:** Phuoc-Nguyen Bui `[一作]` (Sungkyunkwan University), Hyunseung Choo `[通讯]` (Sungkyunkwan University)

**通讯引用:** 4390 | [OpenAlex ID](https://openalex.org/A5054933494)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种多层对齐框架SISTA，用于医学影像与放射学报告的跨模态预训练，解决传统对比学习中的假负样本问题，并实现细粒度的补丁-词级别对齐。

**💡 创新点**

创新点在于：1) 语义感知实例级对齐（SIA）利用报告相似性识别伪正样本，缓解假负；2) 稀疏令牌级对齐（STA）通过稀疏交叉注意力实现补丁与词的精准匹配；3) 将实例级与令牌级对齐融合，实现从全局到细粒度的双层学习。

**🔧 技术方法**

使用对比学习（InfoNCE）、稀疏交叉注意力、视觉编码器（ResNet‑50或ViT‑B/16）、文本编码器（BioClinicalBERT）以及自监督增强（图像旋转/模糊、文本摘要）。

**📊 数据集**

主要使用MIMIC‑CXR 2.0.0进行预训练，后续在CheXpert、RSNA Pneumonia、COVID‑x进行分类；SIIM Pneumothorax、RSNA Pneumonia进行分割；RSNA Pneumonia、Object CXR进行目标检测。

**📈 对比分析**

与现有方法（GLoRIA、ConVIRT、MGCA、M‑FLAG、PRIOR、MedKLIP、MLIP）对比，SISTA在分类、分割和检测任务上均取得SOTA或近SOTA结果，尤其在少量标注数据（1%–10%）下显著提升AUC、Dice和mAP。

**⚠️ 局限性**

局限性包括：①对不同模态的预处理和分词仍需人工选择；②稀疏对齐阈值和重要性权重需要经验调参；③在极大规模多模态数据上计算成本仍较高。

---

## 126. HOSC: A Periodic Activation with Saturation Control for High-Fidelity Implicit Neural Representations

**arXiv ID:** 2601.07870 | [PDF](https://arxiv.org/pdf/2601.07870v1)

**作者:** Michal Jan Wlodarczyk `[一作]` (Warsaw University of Technology), Przemyslaw Musialski `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 2938 | [OpenAlex ID](https://openalex.org/A5065767002)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 HOSC（Hyperbolic Oscillator with Saturation Control）激活函数，能在保留正弦周期结构的同时通过 β 参数独立控制梯度幅度。

**💡 创新点**

创新点在于将梯度尺度与频率解耦，给出严格的 Lipschitz 上界 L = βω₀，并利用 tanh 包装实现可调的梯度门控。

**🔧 技术方法**

使用了正弦激活、双曲正切、NTK 分析、梯度上界推导以及标准化训练协议，构建了多尺度 INR 网络。

**📊 数据集**

实验数据集包括 DIV2K 图像、Bach 小提琴音频、猫咪视频、Blender NeRF 场景、以及多种 3D SDF 形状。

**📈 对比分析**

通过与 SIREN、FINER、Gaussian、WIRE、PEMLP 等基线对比，HOSC 在音频任务上实现显著误差下降（≈4 倍），在图像和视频上提升 1–3 dB，NeRF 与 SIREN/FINER 接近，SDF 亦保持竞争力。

**⚠️ 局限性**

局限在于 β 需要手动调参；高维坐标（如视频、NeRF）受益有限；在极高 β 下易产生不稳定，且仍需与位置编码或网络结构配合以获得最佳效果。

---

## 127. Calibration Is Not Enough: Evaluating Confidence Estimation Under Language Variations

**arXiv ID:** 2601.08064 | [PDF](https://arxiv.org/pdf/2601.08064v1)

**作者:** Yuxi Xia `[一作]` (University of Vienna), Benjamin Roth `[通讯]` (University of Vienna)

**通讯引用:** 1448 | [OpenAlex ID](https://openalex.org/A5046895021)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个综合评估框架，衡量LLM置信估计的鲁棒性、稳定性和敏感性，并与传统的校准与区分度指标进行对比。

**💡 创新点**

创新点在于引入了三种新度量：对提示扰动的鲁棒性、对语义等价答案的稳定性、对语义不同答案的敏感性，揭示传统指标无法捕捉的性能差异。

**🔧 技术方法**

采用了多种置信估计方法（token概率、Platt scaling、语言化置信、P(True)、辅助校准分类器），并用GPT‑4o作为判定模型来评估答案正确性与语义等价性。

**📊 数据集**

使用四个开放式问答数据集（Natural Questions、SciQ、TriviaQA、PopQA）和九个不同家族与规模的LLM（GPT‑4o、Mistral‑123B、Llama‑70B、Qwen‑72B/32B/14B、OLMo‑32B/13B/7B）。

**📈 对比分析**

在鲁棒性、稳定性、敏感性、AUROC、Brier分数等指标上对五种方法进行了比较。结果显示：虽有方法在校准和区分度上表现优异，但在鲁棒性或敏感性上往往不足；如P(True)在区分度上最强，但鲁棒性差；而后置方法（Platt scaling、Calib1）鲁棒性好但敏感性低。

**⚠️ 局限性**

局限性包括：仅评估英语事实问答；提示扰动样本有限；使用GPT‑4o作为判定模型可能带来偏差；未包含一致性/自一致性置信估计；仅关注单轮评估，未考虑交互式或动态置信更新。

---

## 128. LWMSCNN-SE: A Lightweight Multi-Scale Network for Efficient Maize Disease Classification on Edge Devices

**arXiv ID:** 2601.07957 | [PDF](https://arxiv.org/pdf/2601.07957v1)

**作者:** Fikadu Weloday `[一作]` (Southwest University of Science and Technology), Jianmei Su `[通讯]` (Southwest University of Science and Technology)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5015886605)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量级多尺度CNN LWMSCNN-SE，用于在边缘设备上高效分类玉米叶病。

**💡 创新点**

创新点在于将残差多尺度深度可分离卷积与Squeeze-and-Excitation注意力模块协同设计，兼顾局部与全局特征，显著提升小参数模型的准确率。

**🔧 技术方法**

采用深度可分离卷积、残差多尺度模块、SE注意力、数据增强、Adam优化等技术。

**📊 数据集**

使用了PlantVillage公开玉米叶病数据集，共3852张图像，四类。

**📈 对比分析**

与ResNet50、VGG16、EfficientNet-B0、MobileNetV1等传统CNN对比，LWMSCNN-SE仅有241k参数、0.666 GFLOPs，准确率96.63%，保持相近或略高性能，显著降低计算成本。

**⚠️ 局限性**

局限在于对视觉相似病害的区分仍存在误判，且模型验证主要在静态图像上，缺乏在实际田间动态环境中的鲁棒性评估。

---

## 129. Derandomizing Matrix Concentration Inequalities from Free Probability

**arXiv ID:** 2601.08111 | [PDF](https://arxiv.org/pdf/2601.08111v1)

**作者:** Robert Wang `[一作]` (University of Waterloo), Hong Zhou `[通讯]` (Fuzhou University)

**通讯引用:** 5598 | [OpenAlex ID](https://openalex.org/A5027561311)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

本文在自由概率与算法差异理论的交叉框架下，构造了多项式时间的确定性算法，完成了矩阵范数及全谱浓度的无随机化收敛证明；

**💡 创新点**

其创新点在于首次将自由概率中的“内部自由性”与离散化的粘性布朗运动算法相结合，实现了对矩阵高阶范数和谱分布的确定性近似，同时显著减小了对对数因子与矩阵非交换性的依赖；

**🔧 技术方法**

核心技术包括自由概率的极限分布理论、Gaussian 协方差同构、张量 Hölder 及交叉项消除（intrinsic freeness）以及差异理论中的粘性布朗运动和部分着色算法；

**📊 数据集**

论文不依赖具体实验数据集，所有结果均为理论证明与构造性算法；

**📈 对比分析**

与以往的随机化算法相比，本文的确定性算法在保持近似误差不变的前提下完成了同等精度的矩阵范数与谱分布控制，但运行时间上因多次枚举和子空间投影导致的多项式时间复杂度显著提升；

**⚠️ 局限性**

主要限制在于算法的时间复杂度较高，尤其在高维或大规模矩阵情况下可能不可行，同时对非Hermitian 或非自交换矩阵的适用性尚未得到充分验证。

---

## 130. Query Suggestion for Retrieval-Augmented Generation via Dynamic In-Context Learning

**arXiv ID:** 2601.08105 | [PDF](https://arxiv.org/pdf/2601.08105v1)

**作者:** Fabian Spaeh `[一作]` (Celonis), Bin Shen `[通讯]` (Celonis)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种用于 agentic RAG（检索增强生成）系统的查询建议方法，帮助用户在初始查询无法回答时获得可回答的相似查询；

**💡 创新点**

创新点在于引入动态少样本学习（dynamic few‑shot learning），通过模板化查询并检索与之相似的正负例来指导 LLM 生成答案可行的查询；

**🔧 技术方法**

使用了 LLM（GPT‑4o）进行模板化、少样本提示和查询生成，结合文本嵌入（text‑embedding‑3‑small）实现相似度检索，并通过自学习机制从历史查询中自动标注答案可行性；

**📊 数据集**

评估使用了三组真实业务数据集：InvoicesNoPython、InvoicesPython、Orders，分别包含约2000–2700条用户查询；

**📈 对比分析**

与静态少样本学习和仅检索两种基线对比，实验显示动态少样本方法在答案可行性和语义相似度上均显著优于两者，尤其在答案可行率上提升约10–20%；

**⚠️ 局限性**

限制在于需要预先构建工具和数据的描述以支持模板化，对极端复杂或未知工具的推理仍受限；

---

## 131. FigEx2: Visual-Conditioned Panel Detection and Captioning for Scientific Compound Figures

**arXiv ID:** 2601.08026 | [PDF](https://arxiv.org/pdf/2601.08026v1)

**作者:** Jifeng Song `[一作]` (University of Pittsburgh), Yufei Huang `[通讯]` (University of Pittsburgh)

**通讯引用:** 11082 | [OpenAlex ID](https://openalex.org/A5068887242)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种视觉驱动的框架FigEx2，能够从复合科学图像中同时定位面板并生成面板级文本描述；

**💡 创新点**

创新点在于：①将面板检测与面板生成联合建模，通过特殊触发符号和跨分支的噪声感知门控融合模块实现文本信息在检测中的自适应调制；②采用阶段化训练策略，先分离预训练后联合SFT+RL，利用CLIP与BERTScore奖励保证多模态一致性；③构建高质量的BioSci‑Fig‑Cap数据集并提供跨学科物理、化学测试集，验证跨域零样本迁移。

**🔧 技术方法**

技术手段包括：Qwen3‑VL‑8B视觉语言模型、DAB‑DETR检测器、跨分支门控融合模块、Self‑Critical Sequence Training (SCST)、CLIP视觉文本对齐奖励与BERTScore语义奖励、LoRA微调。

**📊 数据集**

使用的数据集为：BioSci‑Fig‑Cap（经清洗的生物科学面板级注释）、MedICaT、PhysSci‑Fig‑Cap‑Test、ChemSci‑Fig‑Cap‑Test；训练集与测试集分别按比例划分。

**📈 对比分析**

与多种基线（YOLOS、DAB‑DETR、Qwen3‑VL‑8B、LLaVA‑NEXT、Llama3.2‑11B等）比较，FigEx2在面板检测上mAP@0.5:0.95提升至0.291/0.726，面板生成在BLEU4、ROUGE‑L、METEOR、BERTScore等指标上均超过同类模型，尤其在跨域零样本迁移和少样本提示下表现优异。

**⚠️ 局限性**

局限性包括：仅支持字母标记面板；未充分利用主图注文本，生成的面板描述可能缺乏实验细节；奖励设计依赖自动指标，可能无法完全反映科学准确性，需要人工评估与领域特定真值检验。

---

## 132. Debiasing Large Language Models via Adaptive Causal Prompting with Sketch-of-Thought

**arXiv ID:** 2601.08108 | [PDF](https://arxiv.org/pdf/2601.08108v1)

**作者:** Bowen Li `[一作]` (RMIT University), Feng Xia `[通讯]` (RMIT University)

**通讯引用:** 18946 | [OpenAlex ID](https://openalex.org/A5089615958)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Adaptive Causal Prompting with Sketch-of-Thought (ACPS)，一种可适配不同推理任务且能够减少 token 使用的提示框架；

**💡 创新点**

将标准前门（standard front‑door）与条件前门（conditional front‑door）联合使用，并引入简短的 Sketch‑of‑Thought（SoT）来替代冗长的 Chain‑of‑Thought（CoT），从而兼顾去偏、通用性与效率；

**🔧 技术方法**

采用结构因果模型（SCM）、前门调整、SoT 生成、多样化温度采样、句子编码、k‑means 聚类、权重化的 NWGM 近似以及 DistilBERT 分类器来决定使用的干预；

**📊 数据集**

在七个推理基准上评估：GSM8K、MATH、CommonsenseQA、StrategyQA、HotpotQA、MuSiQue、FEVER，使用三大后端 LLM（Mini‑TAL‑3B、LLaMA‑3.1 8B、GPT‑3.5‑turbo）；

**📈 对比分析**

与 ICL、CoT、CoT‑SC、SoT、CAD、DeCoT、CP 等基线对比，ACPS 在所有数据集上均取得最高或接近最高的准确率/EM/F1，且在 token 预算内的准确率‑效率平衡最优，鲁棒性实验亦显示对噪声/扰乱输入的优异表现；

**⚠️ 局限性**

主要局限包括：实验规模受限于现有数据集和模型；对生成 SoT 的多样性仍受 API 安全策略约束；缺乏在更大规模模型和更广泛真实场景下的验证。

---

## 133. MirrorBench: An Extensible Framework to Evaluate User-Proxy Agents for Human-Likeness

**arXiv ID:** 2601.08118 | [PDF](https://arxiv.org/pdf/2601.08118v1)

**作者:** Ashutosh Hathidara `[一作]` (SAP Labs), Anil Babu Ankisettipalli `[通讯]` (SAP Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了MirrorBench，一个可扩展、可复现的框架，用于评估用户代理（user‑proxy）在多场景对话中生成的用户发言是否与真实人类相似；

**💡 创新点**

创新点在于将用户代理评估与下游任务成功解耦，提供了六层模块化架构、兼容性检查、缓存与可观测性，并结合词汇多样性指标与LLM‑评判者的三类人类相似度度量；

**🔧 技术方法**

技术实现包括：基于LLM的评判者（GTEval、Pairwise Indistinguishability、Rubric‑and‑Reason）、词汇多样性统计（MATTR、Yule’s K、HD‑D）以及自适应的HH/PP校准、异步/分布式执行、OpenTelemetry日志和持久化数据库；

**📊 数据集**

使用了四个公开对话语料库（QULAC、ClariQ、OASST1、ChatbotArena），每个语料库都经过统一的JSONL预处理并生成用户目标描述；

**📈 对比分析**

评估通过对五种LLM代理（GPT‑4o、GPT‑5、GPT‑OSS‑120B、Claude‑4‑Sonnet、Gemini‑2.5‑Pro）在四个数据集上的人类相似度进行排序，发现Gemini‑2.5‑Pro和Claude‑4‑Sonnet在GTEval/PI/RNR指标上最接近人类；同时还展示了不同评判者的敏感度、成本‑质量折衷与吞吐量分析，证明了框架在大规模评估中的可扩展性；

**⚠️ 局限性**

主要局限包括：评判者模型可能带来族群偏差，仍需多评判者与更强校准；实验仅使用单一随机种子，助理模型固定；数据集仅限四个英文语料，未覆盖多语言与特殊场景；词汇多样性指标无法完全反映对话交互细节。

---

## 134. Hierarchical Online-Scheduling for Energy-Efficient Split Inference with Progressive Transmission

**arXiv ID:** 2601.08135 | [PDF](https://arxiv.org/pdf/2601.08135v1)

**作者:** Zengzipeng Tang `[一作]` (Beijing Jiaotong University), Yulin Shao `[通讯]` (University of Hong Kong)

**通讯引用:** 1089 | [OpenAlex ID](https://openalex.org/A5037069274)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 ENACHI 一种分层能量-精度协同优化框架，用于多用户边缘协同推理系统，在硬截止时延约束下实现能耗与推理精度的平衡。

**💡 创新点**

创新点：①将任务层与包层调度耦合，采用两层 Lyapunov 递推实现长期能耗稳定与即时精度最大化；②引入可拟合的误差模型替代不可解析的 DNN 精度函数；③结合不确定性感知的渐进特征传输与重要性排序，实现样本级动态资源分配；④设计参考跟踪策略在包层实时调节发射功率，弥补任务层粗粒度决策与无线细粒度的错配。

**🔧 技术方法**

主要技术包括：分层 Lyapunov drift-plus-penalty 优化、可插值的误差函数拟合、渐进特征传输（重要性排序 + 预测熵停止）、参考跟踪功率控制、基于 MLP 的不确定性预测器、离散分区点的贪婪搜索、基于 Lambert W 的闭式功率解。

**📊 数据集**

使用 ResNet‑50 预训练模型并在 ImageNet 数据集上评估，验证误差模型与精度曲线的拟合效果。

**📈 对比分析**

与 EFFECT‑DNN、SC‑CAO、ProgressiveFTX、Edge‑Only、Device‑Only 等基线对比，ENACHI 在 100 ms 期限下相较基线提升约 43 % 的推理精度，能耗降低约 62 %；在宽带、用户数变化下同样保持最优或最优附近的精度与能耗，并表现出良好的可扩展性。

**⚠️ 局限性**

局限性：①误差模型的拟合误差与样本分布漂移会影响任务层决策；②依赖预先设定的分区点集合，缺乏在线动态分区；③假设信道独立且仅考虑 Rayleigh 衰落，实际多径/干扰情况未完整建模；④算法对超大规模用户时仍需进一步优化计算复杂度。

---

## 135. Exploiting DINOv3-Based Self-Supervised Features for Robust Few-Shot Medical Image Segmentation

**arXiv ID:** 2601.08078 | [PDF](https://arxiv.org/pdf/2601.08078v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 136. Generalization Analysis and Method for Domain Generalization for a Family of Recurrent Neural Networks

**arXiv ID:** 2601.08122 | [PDF](https://arxiv.org/pdf/2601.08122v1)

**作者:** Atefeh Termehchi `[一作]` (University of Manitoba), Isaac Woungang `[通讯]` (Toronto Metropolitan University)

**通讯引用:** 3814 | [OpenAlex ID](https://openalex.org/A5058166217)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于 Koopman 变换的可解释 LSTM 结构，利用 H∞ 范数给出域外泛化误差上界，并设计了基于 LMI 的状态反馈域迁移拒绝方法，显著提升在输入分布漂移下的预测性能。

**💡 创新点**

①将训练好的 RNN 视为非线性闭环动力系统，使用 Koopman 与 DMDc 将其逼近为线性系统，实现全局可解释；②通过谱分析得到域外扰动对隐藏状态的最坏影响，给出可计算的泛化误差上界；③在此分析基础上设计 LMI 优化的状态反馈控制器，实现对域外扰动的抑制。

**🔧 技术方法**

Koopman 变换、动态模式分解（DMDc）、H∞ 控制理论、线性矩阵不等式（LMI）优化、控制可达性降维、统计距离（Wasserstein、Hellinger）比较。

**📊 数据集**

公开的 California building‑energy 数据集（OpenEI），用于预测未来 24 小时电力负荷，包含 876 小时测试集。

**📈 对比分析**

与传统基于 Hellinger 距离的泛化上界进行对比；实验中在三种扰动（均匀、固定、混合）下，理论上界始终不被违背且在大扰动时更保守；域迁移拒绝方法将 MSE 降低 75%（固定扰动）和 59%（均匀扰动），显著优于未使用方法。

**⚠️ 局限性**

仅针对标准 LSTM，假设扰动为加性、可观测且已知内部状态；模型维度较大时计算成本高，需控制可达子空间；理论上界依赖于 H∞ 系统稳定性，若模型训练不充分可能不满足；缺少在线域漂移检测与适配机制。

---

## 137. Forecast Aware Deep Reinforcement Learning for Efficient Electricity Load Scheduling in Dairy Farms

**arXiv ID:** 2601.08052 | [PDF](https://arxiv.org/pdf/2601.08052v1)

**作者:** Nawazish Alia `[一作]`, Karl Mason `[通讯]` (University of Galway)

**通讯引用:** 899 | [OpenAlex ID](https://openalex.org/A5088726212)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了预测感知的近端策略优化（F-PPO）和PID-KL自适应PPO，用于乳牛场的电池与热水器负荷调度，降低电网依赖和能耗；

**💡 创新点**

创新点在于将短期负荷与光伏预测嵌入观测空间，并通过PID控制器自适应调节KL约束，实现更稳健的策略更新与更精准的前瞻决策；

**🔧 技术方法**

采用深度强化学习框架（PPO变体）、GRU编码预测信号、PID控制KL、以及DQN、SAC等基线算法；

**📊 数据集**

使用芬兰VTT公开的全年乳牛场用电数据、NREL SAM生成的20kW光伏产能、赫尔辛基电价；以及爱尔兰乳牛场分解负荷与电价数据；

**📈 对比分析**

与标准PPO、DQN、SAC对比，F‑PPO在电池调度上比PPO低1%电费、比DQN低4.8%、比SAC低1.5%；在热水器调度上实现4.76%电费节省、用户满意率≈99%；电池导入量比无电池模式低13.1%；Wilcoxon检验显示差异显著；

**⚠️ 局限性**

局限在于仅考虑单一光伏来源、固定SOC阈值，模型对极端天气或价格剧变的鲁棒性待验证，且未实现多能耗设备协同或多智能体协作。

---

## 138. Qalb: Largest State-of-the-Art Urdu Large Language Model for 230M Speakers with Systematic Continued Pre-training

**arXiv ID:** 2601.08141 | [PDF](https://arxiv.org/pdf/2601.08141v1)

**作者:** Muhammad Taimoor Hassan `[一作]` (Auburn University), Muhammad Awais `[通讯]` (BTU Cottbus)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 LLaMA‑3.1 8B 的基础上进行 1.97B 词元的持续预训练，再对 Alif Urdu‑instruct 数据集进行指令微调，构建了面向 230 M 说话者的 Urdu 语言模型 Qalb。

**💡 创新点**

提出了大规模、跨领域 Urdu 语料库与持续预训练相结合的低资源语言适配方法，并在此基础上实现了 4‑bit 量化版本，显著提升 Urdu 任务性能。

**🔧 技术方法**

使用 LoRA 参数高效微调、Unsloth 框架、bfloat16 训练、AdamW‑8bit 优化器、余弦学习率调度，以及 GPT‑4o 作为自动评判者进行系统评测。

**📊 数据集**

构建 1.84B 词元 Urdu 语料（新闻、文学、政府文档、社交媒体等）与 140M 英文 Wikipedia 语料混合，累计 1.97B 词元，保证语言知识与防止灾难性遗忘。

**📈 对比分析**

与 LLaMA‑3.1 8B‑Instruct、Alif‑1.0‑Instruct 及其他多语模型在 7 项 Urdu 基准上对比，使用 GPT‑4o 自动评估并人工验证，Qalb 的加权平均得分 90.34，较 Alif 提升 3.24 分、较基线提升 44.64 分。

**⚠️ 局限性**

模型可能产生文化偏见、刻板印象或误导性内容；偶尔会出现幻觉式签名；评估数据集有限，尤其在医疗、法律等敏感领域需谨慎使用。

---

## 139. Robust Stable Matchings: Dealing with Changes in Preferences

**arXiv ID:** 2601.07959 | [PDF](https://arxiv.org/pdf/2601.07959v1)

**作者:** Rohith Reddy Gangam `[一作]` (University of California), Vijay V. Vazirani `[通讯]` (University of California)

**通讯引用:** 19422 | [OpenAlex ID](https://openalex.org/A5007368382)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文系统研究了在两侧稳定匹配模型下，匹配在两个（或多个）偏好实例之间保持稳定的情形——即鲁棒稳定匹配。作者从最简单的单一偏好位移开始，逐步推广到单个代理任意排列以及多个代理任意变化，并对鲁棒匹配集合的结构、算法可行性、极限与多面体整齐性等进行全面阐述。

**💡 创新点**

创新点在于：
① 构建了鲁棒匹配的子格结构与半子格特性，揭示了何时鲁棒匹配集合仍为分布式格；
② 在不同扰动模型（(p,q)）下给出了子格存在与否的精确阈值，首次把 (p,q) 维度与格结构与多面体整齐性联系起来；
③ 提出了一种基于压缩（compression）与 bouquet 的高效算法，能够在多实例情形下构造 Birkhoff 部分顺序；
④ 在 (1,n) 与 (0,n) 模型下给出了多项式时间算法（DA 变种、LP、枚举），并证明了 (2,2) 以上即为 NP‑hard。

**🔧 技术方法**

主要技术包括：
- 分布式格与 Birkhoff 定理的结构分析；
- 旋转（rotation）与旋转偏序的使用，构造压缩以得到鲁棒子格；
- 复杂度分析中的参数化算法与 XP‑时间算法；
- LP 形式化与极点整齐性证明；
- 退避接受（Deferred Acceptance）算法的同步扩展以求取鲁棒最优匹配。

**📊 数据集**

本文未使用实测数据集，而是以理论模型为主。所有结果均基于 n 规模的假设性稳定匹配实例，采用严格的数学证明与构造性示例。

**📈 对比分析**

与传统单实例稳定匹配算法（如 DA、旋转枚举）相比，本文的算法在 (0,n) 与 (1,n) 情形下保持了多项式复杂度；在 (p,q) = (2,2) 时给出了 XP‑时间算法；在 (n,n) 时则证明为 NP‑hard。实验或基准测试并未给出，但理论上已证明在可行区间内效率与传统方法相当或更优。

**⚠️ 局限性**

局限性包括：
- 对 (p,q) ≥ (2,2) 的一般情形缺乏多项式算法，仍处于 NP‑hard；
- 对 (1,1) 情形的子格与多面体结构尚未完全阐明，存在开放问题；
- Birkhoff 部分顺序的构造在 (1,n) 与 (0,n) 之外没有给出高效算法；
- 论文仅讨论严格完全的偏好列表，未考虑弱偏好或不完全信息；
- 实际市场中多重偏好变化的动态过程并未被直接模拟。

---

## 140. ZeroDVFS: Zero-Shot LLM-Guided Core and Frequency Allocation for Embedded Platforms

**arXiv ID:** 2601.08166 | [PDF](https://arxiv.org/pdf/2601.08166v1)

**作者:** Mohammad Pivezhandi `[一作]` (Wayne State University), Ali Jannesari `[通讯]` (Iowa State University)

**通讯引用:** 1072 | [OpenAlex ID](https://openalex.org/A5079359777)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文提出了 ZeroDVFS，一种基于模型的分层多智能体强化学习框架，用于在嵌入式多核平台上实现动态电压频率调节和任务到核分配；

**💡 创新点**

创新点包括：1）将 Dyna‑Q 结合 D3QN 的分层多智能体架构，将指数级动作空间拆分为频率与核分配两子任务；2）利用 LLM 提取 OpenMP 源码的 13 个语义特征，实现零样本、零射频的跨平台部署；3）通过环境模型进行合成训练，显著提升采样效率，首决策时延仅 3.5–8 s，随后的决策 358 ms；

**🔧 技术方法**

技术手段包括：模型基强化学习（Dyna‑Q）、双重优势网络（D3QN）、多重回归模型（FCN、CNN、RNN 等）预测温度与能耗、LLM 语义特征提取（DeepSeek‑V3、Claude‑Sonnet、GPT‑4o）、迁移学习与少样本微调；

**📊 数据集**

数据集与平台：使用 BOTS 与 PolybenchC 两套 OpenMP 基准，部署于 NVIDIA Jetson TX2、Jetson Orin NX、RubikPi ARM‑Kryo 与 Intel Core i7‑8th；

**📈 对比分析**

评估方法：与 Linux ondemand、Precise Scheduler、GearDVFS 等基线进行基准测试；ZeroDVFS 在 Jetson TX2 上实现 7.09× 能耗提升、4.0× 周期缩短，首决策时延比完整表格式剖析快 8300 倍，后续决策时延快 5 × 10⁶ 倍；

**⚠️ 局限性**

局限性：仅针对单 DAG OpenMP 工作负载；跨平台零样本迁移仍存在 64–73% MAPE 的域漂移；LLM 特征提取目前仅支持单文件代码；实验重复次数有限，缺乏置信区间；缺少多任务并发与 GPU 迁移研究。

---

## 141. Universal computation is intrinsic to language model decoding

**arXiv ID:** 2601.08061 | [PDF](https://arxiv.org/pdf/2601.08061v1)

**作者:** Alex Lewandowski `[一作]` (University of Alberta), Dale Schuurmans `[通讯]` (University of Alberta)

**通讯引用:** 18397 | [OpenAlex ID](https://openalex.org/A5010575626)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

证明语言模型在自回归解码下能实现通用计算，甚至在随机初始化时亦可；

**💡 创新点**

发现训练不增添计算能力，而是提升可编程性；提出通过系统提示或可注入码本实现“证明-仿真”；

**🔧 技术方法**

自回归解码、Lag系统、通用图灵机、系统提示、编码器-解码器、贪婪采样、热身训练等技术；

**📊 数据集**

未使用传统文本语料，仅利用自构造的规则序列（1857条Lag系统规则）进行验证；

**📈 对比分析**

方法对比：用系统提示让已训练的 Llama‑4‑17B‑128E‑Instruct 通过 1857 条规则验证成功；对随机初始化模型，训练编码器-解码器得到码本后也能通过相同规则验证，表明模型具备通用计算能力；

**⚠️ 局限性**

局限：需手工或自动寻找有效提示/码本；对实际任务的可解释性和可扩展性尚未验证；自回归解码受上下文窗口限制，需扩展机制才能实现无限存储；

---

## 142. Improving Zero-shot ADL Recognition with Large Language Models through Event-based Context and Confidence

**arXiv ID:** 2601.08241 | [PDF](https://arxiv.org/pdf/2601.08241v1)

**作者:** Michele Fiori `[一作]` (University of Milan), Claudio Bettini `[通讯]` (University of Milan)

**通讯引用:** 6916 | [OpenAlex ID](https://openalex.org/A5010533347)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种零样本日常生活活动识别方法，利用大型语言模型结合事件式分割和置信度估计来识别智能家居中的日常活动。

**💡 创新点**

创新点包括：①将事件式分割与LLM结合，利用前k个事件上下文推断活动；②提出基于多次推理的置信度估计方法；③在零样本条件下超越传统时基分割和监督学习。

**🔧 技术方法**

使用了大型语言模型（Gemma 3 27B、4B以及GPT‑4o mini）、Chain‑of‑Thought prompting、事件式窗口分割以及多次重复推理产生置信度。

**📊 数据集**

实验数据来自公开的CASAS Aruba和CASAS Milan两个智能家居数据集。

**📈 对比分析**

与时基零样本方法（ADL‑LLM）以及监督学习事件式随机森林进行对比；在两个数据集上事件式LLM的加权F1分别达到0.80/0.56，明显优于时基零样本（0.50/0.40）且接近监督上限；通过置信度阈值过滤还能将F1提升至0.90。

**⚠️ 局限性**

局限性包括：依赖LLM推理导致延迟和幻觉；需要多次重复推理以获得置信度，增加计算成本；在现代IoT设备的真实环境中尚未验证；置信度阈值会导致部分样本被丢弃；仍受事件分割参数和模型大小限制。

---

## 143. Instruction-Driven 3D Facial Expression Generation and Transition

**arXiv ID:** 2601.08179 | [PDF](https://arxiv.org/pdf/2601.08179v1)

**作者:** Anh H. Vo `[一作]` (Sejong University), Yong-Guk Kim `[通讯]` (Sejong University)

**通讯引用:** 1124 | [OpenAlex ID](https://openalex.org/A5064727471)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

生成3D面部表情并根据文本指令实现从一种表情到另一种表情的过渡动画

**💡 创新点**

设计了跨注意力的IFED模块融合文本与面部参数信息，并在条件变分自编码器框架中加入顶点重构损失以提升语义理解；提出可通过生成中性表情实现更长、连贯的表情序列

**🔧 技术方法**

利用CLIP文本编码、IFED（跨注意力变压器）、条件变分自编码器、FLAME 3D头模型、DECA系数提取、线性插值和面部渲染模型（ROME/CVTHead）

**📊 数据集**

CK+ 与 CelebV-HQ 数据集（已人工添加文本指令）

**📈 对比分析**

与 MotionClip 进行基准对比，使用 Acc_1、Acc_2、Gmean 衡量表情识别，PSNR、LPIPS、MS-SSIM 衡量渲染质量；在 CK+ 上 Acc≈91%/84%/80%，在 CelebV-HQ 上 Acc≈58%/33%/46%；渲染结果中 Ours+CVTHead 显著优于 MotionClip

**⚠️ 局限性**

对 DECA 提取的 3D 系数依赖较大，线性插值导致过渡不够自然；在大幅姿态变化或缺失区域的渲染会失真；模型在极端表情或大角度旋转时表现下降

---

## 144. AdaJudge: Adaptive Multi-Perspective Judging for Reward Modeling

**arXiv ID:** 2601.08097 | [PDF](https://arxiv.org/pdf/2601.08097v1)

**作者:** Yongliang Miao `[一作]` (Emory University), Mengnan Du `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究奖励建模中的聚合与表示瓶颈，提出AdaJudge两阶段自适应框架，将表示细化与多视角聚合相结合，提升对人类偏好评分的能力。

**💡 创新点**

引入深度门控表示细化模块以及域感知的多视角聚合路由器，使模型在不需要token级监督的情况下实现任务自适应的聚合和表示优化。

**🔧 技术方法**

采用Transformer backbone + 轻量级的迭代细化块、深度门控加权、三视角聚合（last-token、mean、attention）以及门控路由网络，训练使用焦点Bradley–Terry目标与熵正则。

**📊 数据集**

在HelpSteer3人类偏好对列表上训练，并在RM‑Bench和JudgeBench这两个奖励模型基准上进行评估。

**📈 对比分析**

与现有开箱奖励模型以及相同backbone的固定聚合基线对比，AdaJudge在RM‑Bench、JudgeBench上显著领先，尤其在高难度子任务与小模型上提升明显。

**⚠️ 局限性**

对大规模模型的验证不足，模型引入的细化与路由会增加参数与推理延迟，且仅使用三种聚合专家，可进一步扩展专家库或压缩模型。

---

## 145. DYCP: Dynamic Context Pruning for Long-Form Dialogue with LLMs

**arXiv ID:** 2601.07994 | [PDF](https://arxiv.org/pdf/2601.07994v1)

**作者:** Nayoung Choi `[一作]` (Emory University), Jinho D. Choi `[通讯]` (Emory University)

**通讯引用:** 2818 | [OpenAlex ID](https://openalex.org/A5101829031)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种动态上下文裁剪（DyCP）方法，能够在长对话中实时识别并提取与当前查询相关的连续段落，从而降低输入长度并提升回答质量与响应速度。

**💡 创新点**

创新点在于：①无需预先划分主题边界，动态根据查询识别连续高相关段落；②扩展Kadane算法用于一次性识别多段落；③显著减少LLM调用次数与输入规模。

**🔧 技术方法**

技术手段包括：基于bi‑encoder的嵌入检索；使用不同检索器（Contriever、BGE等）获取相关性得分；对得分序列进行z‑score归一化后，采用扩展Kadane算法识别连续段落；对话顺序保持不变并拼接返回给LLM。

**📊 数据集**

数据集：LoCoMo、MT‑Bench+、SCM4LLMs 三个长对话基准，覆盖人机对话和人机对话，平均对话长度从 300 轮以上不等。

**📈 对比分析**

与 No Context、Full Context、三种基线（turn‑level、segment‑level、Hybrid）进行对比，使用 GPT‑4o、Claude 3.7、GPT‑4o‑mini、GPT‑4.1、Claude 4.0 等多模型评估；DyCP 在 GPT4Score、EM/ROUGE 等质量指标上均显著优于 Full Context，且响应延迟显著下降（平均减少 30–70%），在最长对话（SCM4LLMs）上表现尤为突出。

**⚠️ 局限性**

局限性包括：①对检索器性能敏感，若检索不准确会影响效果；②仍假设LLM具备一定长上下文处理能力，极大长会话中可能受限；③阈值（gain、stop）需要经验调优，适用性可能随数据集和模型而异。

---

## 146. Large Language Models and Algorithm Execution: Application to an Arithmetic Function

**arXiv ID:** 2601.07898 | [PDF](https://arxiv.org/pdf/2601.07898v1)

**作者:** Farah Ben Slama `[一作]` (University Claude Bernard Lyon 1), Frédéric Armetta `[通讯]` (University Claude Bernard Lyon 1)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在LLM中引入分解式算法学习（LLM‑DAL），通过逐步训练子任务使模型能够执行乘法运算并给出完整的推理过程。

**💡 创新点**

创新点在于将复杂算法拆解为可训练的子任务，并采用递归提示（recursive prompting）使模型生成完整多步推理，显著提升算法推理能力。

**🔧 技术方法**

使用了Llama 3.2‑Instruct基础模型，采用多阶段监督微调、链式思维（CoT）分解、递归提示等技术。

**📊 数据集**

使用人工合成的数据集：t1_mult（两位数乘法）、t2_add（两位数加一位数）、t3_extract（位数提取）、t4_concat（左拼接）以及最终的CoT乘法推理数据集（2000条）。

**📈 对比分析**

与直接在全局乘法任务上训练的Vanilla Llama 3.2相比，LLM‑DAL在全局乘法任务上的准确率从13.5%提升到42.1%，各子任务的单独准确率均接近100%。

**⚠️ 局限性**

局限性包括：仍需人工合成数据，无法自动生成链式思维；对更复杂或不同算法的迁移性能未验证；递归提示机制可能导致推理循环或效率低下。

---

## 147. Large Artificial Intelligence Model Guided Deep Reinforcement Learning for Resource Allocation in Non Terrestrial Networks

**arXiv ID:** 2601.08254 | [PDF](https://arxiv.org/pdf/2601.08254v1)

**作者:** Abdikarim Mohamed Ibrahim `[一作]` (Sunway University), Rosdiadee Nordin `[通讯]` (Sunway University)

**通讯引用:** 6139 | [OpenAlex ID](https://openalex.org/A5060844784)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种利用大型语言模型（LLM）为深度强化学习（DRL）提供策略指导的混合智能框架，用于LEO星座下的下行资源分配。

**💡 创新点**

创新点在于将LLM生成的高层策略标签通过嵌入和注意力机制融入DRL奖励与决策过程，实现策略引导、奖励塑造和可解释性三位一体的资源分配。

**🔧 技术方法**

采用的技术包括：大型语言模型提示工程、策略嵌入与注意力机制、基于TD3的连续动作离策略actor‑critic、以及ITU‑R 规范的传播模型和基准 heuristic（等分、wf、mmf、pc）。

**📊 数据集**

数据集为模拟环境，包含10颗550km高轨道LEO卫星、50个地面用户（按纬度分布），以及两种天气情景（常规与极端），通过仿真生成路径损耗、雨衰减等指标。

**📈 对比分析**

与传统黑盒DRL、等分、wf、mmf、pc等方法对比，LLM‑DRL在常规天气下平均提升约40%吞吐量，极端天气下提升约64%，同时保持约0.76的Jain公平指数并降低掉线概率。

**⚠️ 局限性**

局限性包括：仅验证在中等规模星座和单波束配置，缺乏理论收敛或最优性保证，提示工程对结果敏感，且对大规模星座、跨星干扰或多波束场景的可扩展性尚待评估。

---

## 148. NOVAK: Unified adaptive optimizer for deep neural networks

**arXiv ID:** 2601.07876 | [PDF](https://arxiv.org/pdf/2601.07876v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 149. Cultural Compass: A Framework for Organizing Societal Norms to Detect Violations in Human-AI Conversations

**arXiv ID:** 2601.07973 | [PDF](https://arxiv.org/pdf/2601.07973v1)

**作者:** Myra Cheng `[一作]` (Stanford University), Sunipa Dev `[通讯]` (Google)

**通讯引用:** 2806 | [OpenAlex ID](https://openalex.org/A5090145147)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种细粒度的社会文化规范分类法，并基于该分类法构建了一个评估大语言模型在开放式对话中遵守规范的自动化流程。

**💡 创新点**

创新点在于：①将规范拆解为上下文框架、规范细化与执行机制三大维度，区分人-人与人-AI的规范；②通过自动生成多样化提示、LLM 判别相关规范以及检测违规的四步管道，实现对现实对话中的规范遵守进行量化评估。

**🔧 技术方法**

使用的大规模语言模型（LLM）包括 Gemini‑2‑Flash、Claude 3.7 Sonnet 与 GPT‑4o；评估时使用 LLM 作为判别器来识别相关规范与检测违规；同时利用自定义提示生成器来系统化构造测试用例。

**📊 数据集**

实验使用的规范数据集为 NormAd，包含多国人-人与人-AI场景下的规范条目。

**📈 对比分析**

通过计算不同模型在 H‑H 与 H‑AI 规范下的违规率来比较，发现所有模型均存在违规，但违规率随文化背景、是否在提示中明确提及国家、用户意图和情境类型而显著变化；例如 Claude Sonnet 的违规率最低（约 4‑5%），而 Gemini‑2‑Flash 与 GPT‑4o 的违规率较高。

**⚠️ 局限性**

局限性包括：①数据集主要覆盖人-人规范，缺乏足够的人-AI规范；②分类维度未能囊括所有社会规范细节；③评估依赖 LLM 判别器，需进一步人工专家验证；④对文化复杂性的简化可能导致评估偏差。

---

## 150. WISE-Flow: Workflow-Induced Structured Experience for Self-Evolving Conversational Service Agents

**arXiv ID:** 2601.08158 | [PDF](https://arxiv.org/pdf/2601.08158v1)

**作者:** Yuqing Zhou `[一作]` (George Mason University), Wei Niu `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

利用完整服务日志生成可重用的工作流，指导LLM代理实现任务执行

**💡 创新点**

提出工作流诱导+前置条件增强的结构化经验框架（WISE‑Flow），并引入F_β评价指标

**🔧 技术方法**

多阶段LLM诱导、对比学习、进度对齐、前置条件校验、检索式控制

**📊 数据集**

ToolSandbox与τ^2‑bench两个仿真服务环境

**📈 对比分析**

相较于LLM‑only、ReAct、Reflexion、REMEMBERER、AutoGuide等基线，WISE‑Flow在成功率、F_β与错误率上提升约3–5 %（在两种模型和两个基准上均保持领先）

**⚠️ 局限性**

依赖完整的环境反馈、诱导流程可能出现幻觉或不必要步骤、进度对齐对分支复杂的任务易失效

---

## 151. Towards Principled Design of Mixture-of-Experts Language Models under Memory and Inference Constraints

**arXiv ID:** 2601.08215 | [PDF](https://arxiv.org/pdf/2601.08215v1)

**作者:** Seng Pei Liew `[一作]` (SB Intuitions), Yuyang Dong `[通讯]` (SB Intuitions)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在给定内存和推理预算的前提下，系统研究 MoE 语言模型的五个设计维度（深度、宽度、专家数量、激活专家数、粒度）并提出一套以总参数、专家稀疏度和专家总数为主的设计原则。

**💡 创新点**

创新点在于将 MoE 架构拆解为五个变量，并通过拟合功率律发现性能主要由总参数、专家稀疏度（s = n_exp / n_topk）和专家总数决定，进一步给出最大化总参数、最小化专家稀疏度、最小化专家总数的优化策略，消除了原有的架构歧义。

**🔧 技术方法**

使用的技术包括：基于 Qwen3 的 MoE 变压器架构、参数量约束式（N_total, N_active），粒度 g、宽度/深度比例 γ 的实验探索，采用对数-对数线性回归和 R²、t 检验、条件数等统计评估方法来验证不同变量对损失的显著性，并利用 Chinchilla‑style 规模律进一步验证设计原则。

**📊 数据集**

主要使用 FineWeb‑Edu 数据集进行语言建模训练与评估，训练规模从 30M 到 3B 参数，固定 token 预算。

**📈 对比分析**

通过对比 (n_exp,n_topk)=(128,8) 与 (256,16) 两种保持相同稀疏度的配置，在相同总参数和数据集大小下训练模型并拟合规模律，结果表明 (128,8) 配置在损失上始终优于 (256,16)，验证了设计原则的有效性。

**⚠️ 局限性**

主要局限包括：实验规模受限于中小型模型和数据集，缺乏大规模全网格搜索；使用总参数和激活参数作为内存和推理成本的代理，未能充分覆盖真实部署场景中的硬件、软件优化及并行度等复杂因素。

---

## 152. Generation-Augmented Generation: A Plug-and-Play Framework for Private Knowledge Injection in Large Language Models

**arXiv ID:** 2601.08209 | [PDF](https://arxiv.org/pdf/2601.08209v1)

**作者:** Rongji Li `[一作]` (Institute of Automation Chinese Academy of Sciences), Xu-Yao Zhang `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为Generation-Augmented Generation（GAG）的框架，在冻结的基础LLM上通过单一连续嵌入向量注入私有领域知识，支持插件式、多域扩展并实现可靠的选择激活。

**💡 创新点**

创新点在于：① 将私有知识视作额外模态，通过轻量级专家模型生成隐藏表示并对齐至基础模型的嵌入空间；② 采用单一token的常量预算接口，避免检索或文本注入的上下文碎片化和检索不稳定性；③ 使用无训练的原型路由器（Prototype Plug‑and‑Play Routing, PPR）实现无需参数学习的动态域选择，便于增量部署。

**🔧 技术方法**

核心技术包括：多模态对齐、轻量级投影器（两层MLP）、冻结的专家LM生成背景、基于原型的最近邻路由、两阶段训练（专家适配+投影器对齐），以及对齐和评估使用BERTScore与EM。

**📊 数据集**

使用的数据集有两类私有科学QA基准（免疫学佐剂与催化材料）以及六个公开通用QA基准（FreebaseQA、HotpotQA、Natural Questions、TriviaQA、WebQuestions、PopQA）。

**📈 对比分析**

与基线（Base‑Model‑Only、RAG、xRAG、Expert‑Generated Context）对比，GAG在私有域QA上分别提升约15%和15%，且在通用QA上保持与基线相近，接近oracle路由上限；同时路由精度达99%以上。

**⚠️ 局限性**

局限性：① 只支持单域知识注入，难以处理需要跨域知识融合的查询；② 单token注入可能在需要精确复制数字或单位的场景下表现不如文本检索，需要额外的数值校验或后处理。

---

## 153. Intra-tree Column Subsampling Hinders XGBoost Learning of Ratio-like Interactions

**arXiv ID:** 2601.08121 | [PDF](https://arxiv.org/pdf/2601.08121v1)

**作者:** Mykola Pinchuk `[一作]` `[通讯]` (Independent Researcher), Mykola Pinchuk (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了 XGBoost 内树列采样对比例型交互的影响，探讨缺失比例特征时的性能下降；

**💡 创新点**

首次量化内树列采样如何破坏需要两原始特征协调分割的比例消除交互，并提出避免或加入工程比例特征的实用建议；

**🔧 技术方法**

使用 XGBoost 的列采样参数（s），构造两种合成数据生成过程（连续对数比率和计数曝光差异），评估 PR‑AUC、ROC‑AUC、路径共用度和潜在变量相关性；

**📊 数据集**

两种合成数据集：DGP‑A（连续对数比率）和 DGP‑B（计数+曝光），每个样本包含 2 个信号原始特征、120 个噪声特征及可选的工程比例特征；

**📈 对比分析**

与不使用内树列采样（s=1.0）的基线对比；在 s=0.4 时 PR‑AUC 最高下降 53%（≈0.09），加入比例特征后下降压缩至 ≤5%；路径共用度随性能下降同步降低；

**⚠️ 局限性**

局限包括仅使用合成数据、固定特征量 120、仅测试 XGBoost；对真实数据、不同特征维度、深度及交互复杂度的适用性未验证。

---

## 154. Affect and Effect: Limitations of regularisation-based continual learning in EEG-based emotion classification

**arXiv ID:** 2601.07858 | [PDF](https://arxiv.org/pdf/2601.07858v1)

**作者:** Nina Peire `[一作]` (Imperial College), Björn Schuller `[通讯]` (Imperial College)

**通讯引用:** 53009 | [OpenAlex ID](https://openalex.org/A5043060302)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了正则化型持续学习方法在EEG情绪分类中的适用性，并系统评估其性能与局限

**💡 创新点**

证明正则化方法在高个体间/个体内变异下的稳定‑可塑性失衡，并提供理论与经验分析，指出它们无法有效实现前向迁移

**🔧 技术方法**

使用EWC、SI、MAS三种正则化持续学习算法，并与传统无正则化的序列微调做对比

**📊 数据集**

在公开EEG情绪数据集DREAMER和SEED上进行实验

**📈 对比分析**

通过ACC、BWT、FWT指标比较，发现正则化方法对前向迁移提升有限，后向迁移略有改善，但总体性能与基线相近

**⚠️ 局限性**

受EEG噪声、样本量不足、个体顺序敏感、重要性估计误差和正则化强度选择等因素限制

---

## 155. Towards Verifiably Safe Tool Use for LLM Agents

**arXiv ID:** 2601.08012 | [PDF](https://arxiv.org/pdf/2601.08012v1)

**作者:** Aarya Doshi `[一作]` (Georgia Institute of Technology), Christian Kästner `[通讯]` (Carnegie Mellon University)

**通讯引用:** 14734 | [OpenAlex ID](https://openalex.org/A5067467896)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出一种面向任务特定 LLM 代理的安全保障流程，通过安全工程方法识别危险并用信息流控制实现可验证的工具使用约束。

**💡 创新点**

创新点在于将系统理论过程分析（STPA）与信息流控制（IFC）结合，构建可扩展的 MCP 框架并引入能力、机密性和信任级别的结构化标签，实现从概率性缓解到确定性安全壁垒的转变。

**🔧 技术方法**

主要技术包括 STPA 危险分析、形式化规范（IFC、时序逻辑）与 Alloy 形式化模型、MCP 代理工具调用协议的增强标签与策略执行层。

**📊 数据集**

本文未使用公开机器学习数据集，而是在 Alloy 形式化模型中验证工具交互和信息流安全性；实验基于构造的日历调度代理工具集。

**📈 对比分析**

通过 Alloy 分析验证，安全规范被完全满足，未发现违例路径；与传统模型检测/手工检查方法相比，提供了自动化的形式化保证，避免了人为疏漏。

**⚠️ 局限性**

主要限制包括：标签的获取仍需人工或可信度验证，无法完全覆盖开放市场工具；对任务特定代理的依赖意味着在通用代理上的适用性有限；且实现细节（如标签认证）尚未在真实系统中全面部署。

---

## 156. Training Free Zero-Shot Visual Anomaly Localization via Diffusion Inversion

**arXiv ID:** 2601.08022 | [PDF](https://arxiv.org/pdf/2601.08022v1)

**作者:** Samet Hicsonmez `[一作]` (University of Luxembourg), Djamila Aouada `[通讯]` (University of Luxembourg)

**通讯引用:** 2629 | [OpenAlex ID](https://openalex.org/A5083368272)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种训练免费、视觉仅的零样本异常检测框架，利用预训练扩散模型（Stable Diffusion）进行图像反演，从而在无任何训练样本的情况下实现异常检测与定位。

**💡 创新点**

创新点在于：不需要任何提示语或额外文本引导，也不需要额外训练；直接使用扩散模型的反演过程生成正常图像，并通过与原图的特征差异来定位异常，彻底摆脱了提示词生成的依赖。

**🔧 技术方法**

核心技术包括：Stable Diffusion 2.1 的 DDIM 反演、基于 DINO 的特征提取与余弦不相似度计算、CutLER 的无监督对象掩膜提取，以及简单的文本描述“an image of a [class]”。

**📊 数据集**

使用的数据集为：MVTec-AD（10 物体 + 5 纹理类）、VISA（12 物体类）和 MPDD（工业缺陷类），全部采用公开测试集进行评估。

**📈 对比分析**

与现有训练免费和训练基的零样本异常检测方法比较，VISA 数据集上在像素级 ROC、AP、F1 等指标上取得了 state‑of‑the‑art 表现；在 MVTec-AD 上与现有训练免费方法持平；在 MPDD 上显著提升了所有像素级指标，表明方法具有良好的泛化能力。

**⚠️ 局限性**

主要局限性包括：对缺失部件或细小细节（如焊点、引脚）难以准确重建导致定位失败；在纹理类图像背景杂乱时误检较多，导致 ROC_I 指标相对较低；目前仍是零样本设置，若需要进一步提升检测准确率可能需引入少量正常样本。

---

## 157. FSAG: Enhancing Human-to-Dexterous-Hand Finger-Specific Affordance Grounding via Diffusion Models

**arXiv ID:** 2601.08246 | [PDF](https://arxiv.org/pdf/2601.08246v1)

**作者:** Yifan Han `[一作]` (Institute of Automation, Chinese Academy of Sciences), Wenzhao Lian `[通讯]` (School of Artificial Intelligence, Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

利用预训练文本到图像扩散模型提取物体语义与手部交互信息，结合少量人类演示，学习出每根手指的具体抓取优先点，并通过几何对齐规划实现多指抓取；

**💡 创新点**

提出Finger‑Specific Affordance Field (FSAF)，通过扩散模型的中间特征实现细粒度手指抓取先验；实现跨手型、跨对象的零样本抓取；

**🔧 技术方法**

使用Stable Diffusion U‑Net提取超特征，FPN解码生成五通道手指热图；深度相机+GroundingDINO-SAM2生成点云；离散优化(Levenberg–Marquardt QP)实现抓取轨迹；

**📊 数据集**

自制130个日常物体的人类抓取演示（13类，每类10次），并在7个未见对象上评估；

**📈 对比分析**

与ReKep、CMKA以及基于CLIP/DINO的基线比较，FSAF在KLD、SIM、NSS等指标上显著提升；在真实抓取实验中，成功率达到100%（已知）/85%（未知），远优于ACT‑3D、Diffusion Policy 3D及CMKA；

**⚠️ 局限性**

对细长或低平物体的闭环控制不足，易产生旋转或滑移；仅靠离线规划对动态接触不具备鲁棒性；

---

## 158. A Hardware-Algorithm Co-Designed Framework for HDR Imaging and Dehazing in Extreme Rocket Launch Environments

**arXiv ID:** 2601.08162 | [PDF](https://arxiv.org/pdf/2601.08162v1)

**作者:** Jing Tao `[一作]` (National University of Defense Technology), Qifeng Yu `[通讯]` (National University of Defense Technology)

**通讯引用:** 2796 | [OpenAlex ID](https://openalex.org/A5103187819)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一个硬件-算法协同的光学测量框架，用自定义空间可变曝光（SVE）相机采集多曝光数据，并配合物理感知去雾算法，实现极端火箭发射环境下的高动态范围成像与去雾。

**💡 创新点**

创新点包括：①将SVE传感器与物理感知去雾耦合，直接在硬件层捕获雾散射特征；②基于多曝光统计构建雾分布感知模型，避免传统先验假设；③采用分区自适应增强与多尺度熵约束融合，实现局部光照优化与全局细节保留。

**🔧 技术方法**

核心技术有：空间可变曝光相机硬件设计、基于多曝光统计的雾感知模型、区域分割自适应增强（加权引导滤波+Retinex分解）、多尺度金字塔融合与熵约束重建。

**📊 数据集**

数据集包括：①现场火箭发射序列（475帧）；②实验室燃烧图像（560张）；③仿真雾化火箭模型（512张），后者提供雾化前后配对参考。

**📈 对比分析**

与Mertens09、Wang19、MSPD-MEF、MESPD-MEF、PESPD-MEF五种主流多曝光融合方法对比，使用AG、VIF、IE三种指标；实验结果显示本方法在所有数据集上均获得最高或第二高分，显著提升图像细节、对比度与视觉信息保留。

**⚠️ 局限性**

局限性：缺乏实时处理支持，尚未实现FPGA/GPU加速；需要更完善的现场自动标定方案；对其他恶劣环境（水下、沙尘）需重新调优SVE设计与雾感知模型。

---

## 159. Sliced-Wasserstein Distribution Alignment Loss Improves the Ultra-Low-Bit Quantization of Large Language Models

**arXiv ID:** 2601.07878 | [PDF](https://arxiv.org/pdf/2601.07878v1)

**作者:** Deyu Cao `[一作]` (University of Tokyo), Samin Aref `[通讯]` (University of Toronto)

**通讯引用:** 548 | [OpenAlex ID](https://openalex.org/A5067809445)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出在大语言模型超低比特量化过程中加入切片Wasserstein损失，以在不增加推理开销的前提下提升量化后模型的准确率和流畅度。

**💡 创新点**

创新点在于将分布匹配的切片Wasserstein距离与传统 MSE 损失结合，形成一种通用的分布感知校准方法，显著恢复 2‑bit 及 4‑bit 量化中丢失的性能，且无需额外推理成本。

**🔧 技术方法**

技术方法包括：随机线性投影下的切片Wasserstein距离、点对点 MSE 损失、学习可调权重切片与 MSE 的加权组合，以及在 OmniQuant 与 TesseraQ 两大前沿量化框架中的集成。

**📊 数据集**

使用的数据集包括 WikiText‑2 与 C4 进行量化校准，零样本下通过 ARC‑Challenge、ARC‑Easy、BoolQ、HellaSwag、PIQA 与 Winogrande 六大通用推理基准进行评测。

**📈 对比分析**

与全精度及基线量化（OmniQuant/TesseraQ）对比，作者在 LLaMA‑2‑7B、13B、OPT‑6.7B 等模型中，W2A16g128/W4A4 等配置下平均准确率提升 0.48–2.68 点（相对 4.1–20.4%），TesseraQ 的提升为 0.3–1.6 点（相对 3–7.6%），同时在困扰的 4‑bit 量化中也可实现 2‑bit 的可接受性能。

**⚠️ 局限性**

局限性包括：仅在 PTQ 场景验证，额外的计算开销主要来自随机投影和排序但仍较轻；需要对投影数 n_proj 与权重 sw_w 进行经验调参；在极端低比特（≤2‑bit）或更大模型的 QAT 场景下效果仍待进一步探索。

---

## 160. Spiking Neural-Invariant Kalman Fusion for Accurate Localization Using Low-Cost IMUs

**arXiv ID:** 2601.08248 | [PDF](https://arxiv.org/pdf/2601.08248v1)

**作者:** Yaohua Liu `[一作]` (Guangdong Institute of Intelligence Science and Technology), Binkai Ou `[通讯]` (BoardWare Information System Co.Ltd)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种将突触神经网络（SNN）与不变扩展卡尔曼滤波器（InEKF）结合的低成本MEMS IMU里程计框架；

**💡 创新点**

创新点在于利用SNN实时提取IMU序列的时空特征并自适应调节InEKF的噪声协方差，从而提升在高噪声环境下的定位精度；

**🔧 技术方法**

核心技术包括基于事件驱动的LIF神经元、突触Transformer结构、surrogate梯度训练和InEKF的伪观测更新；

**📊 数据集**

在KITTI视觉惯性里程计数据集（IMU采样100Hz）和自建移动机器人实际场景（ICM‑20602 IMU）上进行实验；

**📈 对比分析**

与ORB‑SLAM3、AI‑IMU和MD‑IMU等基线相比，平均相对平移误差从9.7%降低至3.0%，相对旋转误差从9.4/​km降至4.4/​km，且在IMU数据缺失时仍能保持较低误差；

**⚠️ 局限性**

局限性在于对极端动态或长期漂移的适应性仍有限，且需要较大的时间窗口（500步）导致实时性受限，未来需进一步优化模型规模与推理速度。

---

## 161. Coupled Diffusion-Encoder Models for Reconstruction of Flow Fields

**arXiv ID:** 2601.07946 | [PDF](https://arxiv.org/pdf/2601.07946v1)

**作者:** AmirPouya Hemmasian `[一作]` (Carnegie Mellon University), Amir Barati Farimani `[通讯]` (Carnegie Mellon University)

**通讯引用:** 8000 | [OpenAlex ID](https://openalex.org/A5008745801)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种将卷积 ResNet 编码器与条件扩散模型耦合的框架（DiffCoder）用于流场的压缩与重构。

**💡 创新点**

创新点在于用扩散模型作为解码器，提供生成性先验，从压缩的潜在空间中恢复高频统计结构，显著提升在强压缩下的谱保真度。

**🔧 技术方法**

采用卷积残差网络编码器、条件 U‑Net 逆扩散解码器、DDIM 采样以及端到端的训练策略。

**📊 数据集**

使用二维 Kolmogorov 流（Re=1000）的 256×256 vorticity 快照数据集进行实验。

**📈 对比分析**

与匹配的 VAE 基线在不同模型规模和压缩深度下对比，DiffCoder 在深压缩时谱误差显著下降，VAE 在轻压缩时更优；两者相同的 L2 误差下，DiffCoder 保留了更好的统计分布。

**⚠️ 局限性**

局限性包括在极端压缩时仍存在较高的点误差、仅验证于二维流场、未加入物理约束或时间预测等功能，未来需要扩展到三维、加入物理正则化与潜在空间动力学。

---

## 162. Reinforcement Learning Methods for Neighborhood Selection in Local Search

**arXiv ID:** 2601.07948 | [PDF](https://arxiv.org/pdf/2601.07948v1)

**作者:** Yannick Molinghen `[一作]` (Université Libre de Bruxelles), Stefano Michelini `[通讯]` (CETIC research center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对局部搜索中的邻域选择进行了基于强化学习的方法评估，包括多臂赌博机和深度强化学习。

**💡 创新点**

提出了三种问题无关的奖励设计，并系统分析了奖励对学习效果的影响。

**🔧 技术方法**

使用了ε-贪婪、UCB、DDQN、PPO等算法，结合图神经网络/卷积网络状态表示和潜在奖励塑形。

**📊 数据集**

在TSPLib、CSPLib以及PDPTW的数据集上进行实验，共计368个实例。

**📈 对比分析**

与随机、轮询、最佳斜率等基线比较，结果表明ε-贪婪始终表现最优；DRL在车序列问题上有优势，但计算成本高，整体性能与时间预算有关。

**⚠️ 局限性**

限制包括DRL计算开销大、奖励设计敏感、无法显著超越简单基线、需进一步验证在其他LS求解器和问题上的通用性。

---

## 163. Embedded AI Companion System on Edge Devices

**arXiv ID:** 2601.08128 | [PDF](https://arxiv.org/pdf/2601.08128v1)

**作者:** Rahul Gupta `[一作]` (Superfocus AI), Stephen D. H. Hsu `[通讯]` (Michigan State University)

**通讯引用:** 4251 | [OpenAlex ID](https://openalex.org/A5102790826)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种可在边缘设备上运行的 AI 伴侣系统，采用主动–被动记忆范式实现实时低延迟对话与离线记忆提取与融合。

**💡 创新点**

创新点在于将记忆处理拆分为用户活跃期的实时检索与用户静默期的高成本提取、整合与遗忘；并提出全自动化、全维度的 AI 伴侣基准。

**🔧 技术方法**

主要技术包括 int4 量化 Qwen‑7B 语言模型、gpt‑4‑embeddings (gte‑base‑en‑v1.5)、向量检索、滑动上下文窗口、遗忘曲线、LLM 生成的 JSON 处理与重试。

**📊 数据集**

使用 GPT‑5 生成的用户角色与 Claude Sonnet 4.5 模拟的对话作为实验数据，未使用公开传统数据集。

**📈 对比分析**

通过与无记忆 Qwen、GPT‑3.5 Turbo 以及完整 GPT‑5 的基线进行对比，在对话质量、QA 准确率、个性化命中率和提取质量等多维指标上，系统均显著优于无记忆 Qwen，仍落后于 GPT‑5。

**⚠️ 局限性**

局限包括模型规模受限导致提取准确率不足、基准中缺乏多跳/聚合问题、儿童角色模拟不够自然、长期时间与会话长度的现实性不完整以及遗忘机制未在真实时间轴上充分验证。

---

## 164. Project Synapse: A Hierarchical Multi-Agent Framework with Hybrid Memory for Autonomous Resolution of Last-Mile Delivery Disruptions

**arXiv ID:** 2601.08156 | [PDF](https://arxiv.org/pdf/2601.08156v1)

**作者:** Arin Gopalan Yadav `[一作]` (Vellore Institute of Technology University), Kumar Shivam `[通讯]` (Vellore Institute of Technology University)

**通讯引用:** 96 | [OpenAlex ID](https://openalex.org/A5040174085)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了基于层级多代理架构的 Project Synapse，用于自动识别、拆解并执行 last‑mile delivery（LMD）中断的全流程解决方案，集成混合内存（工作、情节、语义）支持有状态、情境感知和事实依据推理；

**💡 创新点**

创新点包括①层级监督+专职工人代理的分层任务拆解与执行机制；②混合内存架构实现短期工作记忆、长期情节记忆与语义知识库的统一查询与利用；③使用 LangGraph 实现可循环、可分支的工作流；④采用 LLM‑as‑a‑Judge 评估框架与偏差缓解；

**🔧 技术方法**

技术栈包括 Qwen‑14B LLM、ReAct/CoT 推理、LangGraph/Chain 进行工作流与工具调用、LangSmith 监控与调试、MCP Toolkit 微服务、ChromaDB 向量数据库、SQLite 做工作记忆快照、RAG 技术；

**📊 数据集**

使用从 6,239 条超级应用用户评论中抽取并定性分析得到的 30 个复杂真实 LMD 中断情境数据集；

**📈 对比分析**

通过与单代理 ReAct、平面多代理、加 RAG、层级多代理等基线进行消融比较，并采用 LLM‑as‑a‑Judge 评估；在 30 个场景上总体得分 0.73，Plan Correctness 0.71、Reasoning 0.77、Efficiency 0.73；相较基线平均提升 0.15–0.25；

**⚠️ 局限性**

局限性：样本量仅 30 个场景；在仿真环境下验证，未在真实生产系统中测试；评估依赖 LLM‑Judge，缺乏人工专家评估；内存容量有限，长时学习效果未知；未测算成本与延迟；未实现自我进化与自动化任务拆解。

---

## 165. A survey: Information search time optimization based on RAG (Retrieval Augmentation Generation) chatbot

**arXiv ID:** 2601.07838 | [PDF](https://arxiv.org/pdf/2601.07838v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 166. Internal Deployment Gaps in AI Regulation

**arXiv ID:** 2601.08005 | [PDF](https://arxiv.org/pdf/2601.08005v1)

**作者:** Joe Kwon `[一作]` (MIT), Stephen Casper `[通讯]` (MIT)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5099024522)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析了美国与欧盟2025年前沿AI法规对内部部署的适用性，识别并系统阐述了三大监管缺口（范围模糊、静态合规、信息不对称）。

**💡 创新点**

首次将内部部署的监管缺口与外部部署进行对照，提出了可行的政策补救路径，并为未来监管设计提供了概念框架。

**🔧 技术方法**

采用法律文本分析、比较法研究与案例梳理等方法，对法规条文进行系统解读。

**📊 数据集**

主要基于2025年颁布或提议的法规文本（欧盟AI法案、GPAI代码、加州SB 53、新纽约RAISE法案、联邦AIREA）以及相关行业案例资料。

**📈 对比分析**

通过对法规条款、触发条件、合规周期等维度的对比，评估各法规对内部部署的覆盖度；未进行实验性能评估，仅在理论层面衡量监管完整性。

**⚠️ 局限性**

局限在于缺乏对监管执行效果的实证验证，主要依据文本推理；对跨国企业多重合规层面的具体实践不足，技术细节评估受限于法规描述而非实测数据。

---

## 167. Revealing the Attention Floating Mechanism in Masked Diffusion Models

**arXiv ID:** 2601.07894 | [PDF](https://arxiv.org/pdf/2601.07894v1)

**作者:** Xin Dai `[一作]` (Northeastern University), Maosong Sun `[通讯]` (Tsinghua University)

**通讯引用:** 36343 | [OpenAlex ID](https://openalex.org/A5046448314)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Masked Diffusion Models（MDMs）的内部注意力机制进行系统分析，首次发现并描述了注意浮动（Attention Floating）现象，并解释其导致的浅层结构感知、深层内容聚焦（Shallow Structure‑Aware, Deep Content‑Focused）机制；

**💡 创新点**

创新点在于：1) 通过可视化与量化方法揭示MDMs与传统自回归模型（ARMs）在注意力分布上的根本差异；2) 用几何分解（QK→方向与规模）证明浅层关注结构锚点，深层关注语义内容；3) 结合检索头分析和区域级注意流验证浮动机制对上下文学习鲁棒性的提升；

**🔧 技术方法**

技术手段包括：双向Transformer注意力、逐步去噪（denoising）推理、QK分解（角度与范数）、检索头得分（retrieval‑score）、区域级注意流分析、上下文扰动Stress测试、RAG检索集成；

**📊 数据集**

使用的数据集有：知识密集型任务（GSM8K、2WikiMQA、Open‑Domain QA、Multi‑hop QA、Slot Filling、HotpotQA、NQ、TQA、Marco QA、T‑REx）以及常见问答（Close‑book、RAG）；

**📈 对比分析**

对比方法：与主流ARMs（Llama、Qwen）及其RAG版本对比；MDMs在知识检索增强（RAG）场景下平均提升约19.5%，比ARMs的8.5%提升更为显著；在噪声干扰、证据位置扰动和证据整合等Stress测试中，MDMs表现出更低的性能波动和更高的鲁棒性；

**⚠️ 局限性**

局限性：仅聚焦MDMs，未覆盖其他DLLM变体；未提出可直接用于训练或推理的基于浮动的改进方法；缺乏对不同去噪/扩散策略对注意浮动的系统对比。

---

## 168. Knowing But Not Doing: Convergent Morality and Divergent Action in LLMs

**arXiv ID:** 2601.07972 | [PDF](https://arxiv.org/pdf/2601.07972v1)

**作者:** Jen-tse Huang `[一作]` (Johns Hopkins University), Mark Dredze `[通讯]` (Johns Hopkins University)

**通讯引用:** 23110 | [OpenAlex ID](https://openalex.org/A5024437840)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了ValAct-15k，一个基于Schwartz十维价值观的15,000个四选项决策场景数据集，并用它评估10个前沿LLM与55名人类受试者的价值认知与行为表现。

**💡 创新点**

创新点包括：①把价值观测量从单纯问卷（PVQ-40）扩展到现实情境决策；②系统比较了LLM与人类在自报与实际行为间的“知识-行动”差距；③发现LLM在角色扮演指令下出现“role‑play resistance”，提示当前对齐方式的局限性。

**🔧 技术方法**

使用的技术主要是：大型语言模型（GPT‑4o、Qwen‑2.5 等）进行情境生成和决策输出；通过手工与 LLM 迭代的 prompt 设计；统计分析采用 Pearson 相关、准确率、PCA 等方法。

**📊 数据集**

数据集来源于 2020‑2025 年 16 个 Reddit 子版块的 3,000 条贴文（每条对应 4 条价值导向的动作），并生成 15,000 个四选题；同时使用 PVQ‑40 进行自报价值测评。

**📈 对比分析**

比较方法：对每个模型与人类分别计算自报价值向量与情境决策向量，再求 Pearson 相关；在角色扮演实验中测量选值与角色扮演两种指令下的准确率。结果显示：LLM 在不同来源与架构间的价值一致性几乎为 1.0；人类表现高度多样；LLM 与人类在自报与行为上的相关性约为 0.3‑0.4；角色扮演指令会使准确率下降 3–6%。

**⚠️ 局限性**

局限性包括：①数据来源仅为 Reddit，样本偏向特定网络社群；②主要评估为英语，虽做了中英翻译检验但仍可能受语言偏差影响；③潜在的模型预训练数据污染，难以完全排除记忆式回答；④样本规模虽大，但人类受试者仅来自美国，跨文化普适性待进一步验证。

---

## 169. TP-Blend: Textual-Prompt Attention Pairing for Precise Object-Style Blending in Diffusion Models

**arXiv ID:** 2601.08011 | [PDF](https://arxiv.org/pdf/2601.08011v1)

**作者:** Xin Jin `[一作]` (GenPi Inc.), Yapeng Tian `[通讯]` (The University of Texas at Dallas)

**通讯引用:** 10484 | [OpenAlex ID](https://openalex.org/A5101835756)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 TP-Blend，一种训练‑free 的扩散编辑框架，可同时完成对象替换、对象混合和样式融合；

**💡 创新点**

创新点在于：①双文本提示解耦内容与风格；②通过交叉注意力融合（CAOF）与熵正则化的最优传输将混合对象特征映射到目标位置；③自注意力样式融合（SASF）使用细节感知实例归一化（DSIN）和文本驱动的键值替换实现高频纹理注入；

**🔧 技术方法**

技术手段包括：基于 SD‑XL 的扩散模型、CFG‑TE 引导、跨头平均注意力、最优传输（Sinkhorn）、DSIN、Key/Value 替换、低高频分离过滤；

**📊 数据集**

使用公开的 Unsplash 图像集合（4000 张样本，40 张基准图像，20 种替换-混合组合，5 种风格），通过 DDIM 逆向获取潜在空间；

**📈 对比分析**

与 9 种最新文本编辑器（如 Step1X‑Edit、SeedEdit、LEDITS++、TurboEdit、IP2P、StyleAligned、Blended Diffusion、FLUX.1 Kontext、Nano Banana）对比，TP-Blend 在 BOM（对象混合）和 BOSM（对象+风格混合）指标上均领先，CLIP 相似度和 1‑LPIPS 也表现优异，推理速度不逊色；

**⚠️ 局限性**

局限性包括：仍主要针对单对象混合，复杂多对象或场景的融合仍有挑战；对阈值、融合系数等超参数敏感；以及在极端风格或高度相似对象时可能出现细节混淆。

---

## 170. Predicting Region of Interest in Human Visual Search Based on Statistical Texture and Gabor Features

**arXiv ID:** 2601.07998 | [PDF](https://arxiv.org/pdf/2601.07998v1)

**作者:** Hongwei Lin `[一作]` (University of Houston), Howard C. Gifford `[通讯]` (University of Houston)

**通讯引用:** 1803 | [OpenAlex ID](https://openalex.org/A5032527429)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过开发两条融合Gabor特征与GLCM纹理特征的视觉搜索模型管线，对早期视觉搜索行为进行建模，并在模拟的数字乳腺层析成像（DBT）图像上进行评估。

**💡 创新点**

创新点在于将Gabor滤波器捕获的局部方向频率信息与GLCM统计纹理信息结合，形成全局与局部双重特征的搜索策略，并揭示两者高度相关（r=0.765）。

**🔧 技术方法**

采用高斯混合模型（GMM）进行特征聚类、交叉相关得到Gabor特征图、逆水流算法选取局部极大点、阈值筛选以及基于阈值数据的模型观察者。

**📊 数据集**

使用VICTRE生成的三种乳腺密度（mostly fatty、scattered areas of dense、heterogeneously dense）的模拟DBT图像，并收集两名受试者的Tobii Pro眼动数据。

**📈 对比分析**

通过对比管线A、管线B以及阈值模型的预测固定点位置与眼动结果进行可视化与相关性分析，发现三者在定位上相似，且Gabor与GLCM特征呈显著正相关，显示方法有效但缺乏定量性能指标。

**⚠️ 局限性**

局限在于仅使用模拟数据、眼动样本规模有限、评价多为定性分析，缺乏严格的数值性能评估，未来需扩大眼动数据库并引入更多特征族进行验证。

---

## 171. A Survey of Security Challenges and Solutions for UAS Traffic Management (UTM) and small Unmanned Aerial Systems (sUAS)

**arXiv ID:** 2601.08229 | [PDF](https://arxiv.org/pdf/2601.08229v1)

**作者:** Iman Sharifi `[一作]` (George Washington University), Amir Shirkhodaie `[通讯]` (Tennessee State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对小型无人机（sUAS）在UTM（无人机交通管理）框架下的网络安全漏洞与防御措施进行了系统化综述，构建了从通信、导航、感知到软件层面的完整安全漏洞与防御技术分类，并针对每一子系统总结了现有研究与技术缺口。

**💡 创新点**

创新点在于：①将sUAS的安全问题与UTM生态系统紧密关联，形成“CNS‑感知‑软件”四层完整安全栈；②提出统一的漏洞与防御对照表，帮助研究者快速识别攻击链与相应缓解方案；③指出多层防御相互作用、资源受限下的轻量级实现路径以及未来标准化测试平台的必要性。

**🔧 技术方法**

主要使用的技术与方法：系统化文献检索与归纳、构建攻击-防御映射表、对比分析不同防御机制的适用性与资源消耗、对已有标准（CWE/CVE/CAPEC）进行映射。

**📊 数据集**

本文并未使用公开数据集，而是引用了大量已有研究中的案例、实验设备（如SDR、GNSS模拟器、视觉传感器实验平台）与公开事件报告，作为漏洞与防御技术的原始素材。

**📈 对比分析**

由于为综述性工作，本文没有自己的实验或性能对比，而是综述各研究中报告的指标（如检测率、误报率、延迟、功耗等），并对不同方法的优缺点进行理论性比较。整体而言，文中指出多数防御方案在理论上可行，但在低功耗、实时性或规模化部署上缺乏统一评估。

**⚠️ 局限性**

局限性包括：①缺乏统一的实验平台与基准，导致跨研究结果难以直接比较；②多数安全技术在实验室或仿真环境中验证，未覆盖复杂真实场景（多目标、恶劣天气、强干扰等）；③对资源受限的sUAS，轻量化实现与能耗评估仍不足；④未提供对UTM云服务与第三方数据完整性验证的实测方案。

---

## 172. A Highly Efficient Diversity-based Input Selection for DNN Improvement Using VLMs

**arXiv ID:** 2601.08024 | [PDF](https://arxiv.org/pdf/2601.08024v1)

**作者:** Amin Abbasishahkoo `[一作]` (University of Ottawa), Lionel Briand `[通讯]` (University of Ottawa)

**通讯引用:** 28649 | [OpenAlex ID](https://openalex.org/A5078533117)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于Vision‑Language模型的概念多样性度量（CBD）和相应的混合输入选择方法，用于高效地从大量无标签图像中挑选用于DNN微调的高信息量样本。

**💡 创新点**

创新点在于利用VLM（如CLIP）自动提取图像概念，并通过线性映射实现高效概念检索，CBD在保持与传统几何多样性（GD）高度相关性的同时，计算时间缩短2.5~36倍；将CBD与不确定度（Margin）结合的混合策略既提升了模型准确率，又保持了低计算开销。

**🔧 技术方法**

核心技术包括CLIP视觉‑文本双编码器、概念嵌入与概念抽取、线性对齐器、Shannon熵多样性计算以及基于不确定度的递归筛选算法。

**📊 数据集**

实验使用CIFAR‑10（ResNet‑18）和ImageNet‑1k（ResNet‑101）的标准训练/验证集，覆盖从1%到20%不同选择预算。

**📈 对比分析**

与五个最优基线（Margin、DATIS、RTS、DeepGD、SETS）比较，CBD基线在所有预算下平均提升约79%（CIFAR‑10）和38%（ImageNet）的微调准确率，且选择时间仅为Margin的1.5~2倍，远低于其他多样性或混合基线。

**⚠️ 局限性**

局限性包括：概念抽取中的m（每张图提取概念数）需手工调参，未探索不同VLM或知识库对CBD的影响，线性对齐在深层网络上效率相对下降，且在极大输入集（>百万样本）下仍需进一步优化。

---

## 173. Unified Multi-Site Multi-Sequence Brain MRI Harmonization Enriched by Biomedical Semantic Style

**arXiv ID:** 2601.08193 | [PDF](https://arxiv.org/pdf/2601.08193v1)

**作者:** Mengqi Wu `[一作]` (University of North Carolina), Mingxia Liu `[通讯]` (University of North Carolina)

**通讯引用:** 9481 | [OpenAlex ID](https://openalex.org/A5050560717)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `da1b1a89-583a-4b57-9c81-478778569bec` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出MMH框架，针对多站点多序列脑MRI实现去谐化，同时保持解剖结构不变。

**💡 创新点**

创新点在于两阶段学习：第一阶段用梯度锚定的条件扩散模型实现全局统一风格；第二阶段用三平面注意力BiomedCLIP提取语义风格，进行目标站点细粒度调优；无需配对数据，能够一次性处理多序列。

**🔧 技术方法**

采用条件扩散模型、梯度归一化条件、EMA统计+AdaIN、三平面注意力CLIP（TPA‑CLIP）、语义风格位移自监督损失等技术。

**📊 数据集**

使用OpenBHB（3,984份T1w），SRPBS（9名旅行者在11站），DWI‑THP（5名旅行者在8站，T1w/T2w）三大多站点、多序列数据集。

**📈 对比分析**

与CycleGAN、StyleGAN、HF、HACA3、3D CycleGAN、DDPM等SOTA方法对比，MMH在图像特征聚类、体素级对比（SSIM/PSNR/WD）、组织分割Dice、年龄预测MAE、站点分类准确率等指标均优于对手，尤其在SSIM/PSNR提升约2%–3%，WD下降到≈0.004–0.005，Dice提升≥1%。

**⚠️ 局限性**

局限性包括：仅在健康人群验证，缺少病灶数据；仅训练已见序列，无法零样本适配新序列；扩散模型计算成本高；对极端噪声站点仍可能存在残留风格。

---

## 174. How Do Optical Flow and Textual Prompts Collaborate to Assist in Audio-Visual Semantic Segmentation?

**arXiv ID:** 2601.08133 | [PDF](https://arxiv.org/pdf/2601.08133v1)

**作者:** Peng Gao `[一作]` (Hong Kong Baptist University), Wentao Fan `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 2350 | [OpenAlex ID](https://openalex.org/A5056455910)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为Stepping Stone Plus（SSP）的框架，用于音频视觉语义分割（AVSS），通过光流预掩模、双文本提示和视觉-文本对齐模块提升分割性能。

**💡 创新点**

创新点包括：1）将光流作为动态提示与预掩模结合以捕获运动物体；2）使用两种文本提示（场景描述与潜在发声物体），弥补光流对静止发声物体的缺陷；3）设计视觉-文本对齐模块（VTA）实现跨模态特征统一；4）引入后掩模训练目标，强制模型学习光流相关动态特征。

**🔧 技术方法**

技术包括光流提取（Perceiver IO + 平均滤波）、光流掩模生成、文本提示生成（MiniCPM‑o‑2.6）、BERT+CLIP/BLIP视觉-文本对齐、Mask2Former视觉编码、VGGish音频特征、Mask R‑CNN/Transformer解码器以及多任务损失（交叉熵、Dice、后掩模损失）。

**📊 数据集**

使用 AVSBench‑object（S4 单源、MS3 多源）和 AVSBench‑semantic（AVSS）三个数据集，总计约 24,000 条视频，配有音频、视频和语义掩模标注。

**📈 对比分析**

与 6 种融合基模型（如 AVSBench、AQFormer、Catr 等）和 8 种提示基模型（如 AAVS、TeSO、AVS‑Mamba 等）对比，SSP 在所有三个数据集上均取得最高 mIoU 和 F‑score，分别在 S4、MS3、AVSS 上提升约 1.6%、5.0%、1.6%（mIoU）和 1.3%、7.0%、1.3%（F‑score）。

**⚠️ 局限性**

局限性包括：1）光流对静止发声物体仍不够敏感，需依赖文本提示；2）文本提示质量直接影响性能，若 MLLM 生成冗余信息会导致误差；3）训练过程对光流和后掩模权重的调参敏感；4）目前仅在公开数据集上验证，尚未在更大规模或多域场景中评估。

---

## 175. Integrating Attendance Tracking and Emotion Detection for Enhanced Student Engagement in Smart Classrooms

**arXiv ID:** 2601.08049 | [PDF](https://arxiv.org/pdf/2601.08049v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 176. Ideological Isolation in Online Social Networks: A Survey of Computational Definitions, Metrics, and Mitigation Strategies

**arXiv ID:** 2601.07884 | [PDF](https://arxiv.org/pdf/2601.07884v1)

**作者:** Xiaodan Wang `[一作]` (Yanbian University), Quan Bai `[通讯]` (University of Tasmania)

**通讯引用:** 2008 | [OpenAlex ID](https://openalex.org/A5029548157)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述并构建统一的计算框架，对在线社交网络中的意识形态隔离进行定义、度量与缓解方法的系统性整理。

**💡 创新点**

创新点在于：①提出四类隔离类型（结构性、内容性、交互性、认知性）并与现有现象（选择性曝光、滤镜气泡、回音室等）对应；②统一了曝光、观点一致性与多样性的数学表述，形成可操作的指标体系；③整合网络、内容与行为层面的度量与干预策略，提供跨层面的评估与设计范式。

**🔧 技术方法**

使用了图论度量（模块化、同质性、RWC 等）、机器学习（贝叶斯/代理模型、强化学习、图神经网络）以及推荐系统的预/中/后处理技术（多样性正则、对抗式校正、RL 规划、重排序）等多种计算方法。

**📊 数据集**

综述中引用了多平台数据（Facebook、Twitter、WeChat、TikTok、YouTube、Reddit 等）以及公开的政治/内容标注数据集，但未在本文中自行构建新的实验集。

**📈 对比分析**

通过对比已有研究中的指标和实验结果，作者整理了各类方法在准确率、多样性、鲁棒性和可解释性等维度的表现，并在表格中概述了不同策略的优势与劣势，强调在保持推荐精度的前提下提升多样性与公平性的权衡。

**⚠️ 局限性**

局限性包括：①缺乏统一的基准任务与评价标准，导致跨研究对比困难；②大多数实验仅在单一平台或合成数据上验证，缺少跨平台、真实生产环境的验证；③多数缓解策略在提升多样性时会牺牲一定的准确性或用户满意度；④实际部署与人类评估缺乏系统的实证检验。

---

## 177. Second-order Gaussian directional derivative representations for image high-resolution corner detection

**arXiv ID:** 2601.08182 | [PDF](https://arxiv.org/pdf/2601.08182v1)

**作者:** Dongbo Xie `[一作]` (Shaanxi University of Science and Technology), Weichuan Zhang `[通讯]` (Shaanxi University of Science and Technology)

**通讯引用:** 2216 | [OpenAlex ID](https://openalex.org/A5013741704)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过求解二阶高斯方向导数(SOGDD)在两类高分辨角模型（END型与L型）中的表示，提出了一种能准确检测相邻角点的新角点检测方法。

**💡 创新点**

创新点在于：①推导了两类角模型的SOGDD表达式；②利用SOGDD自相关矩阵的特征值来衡量角点强度；③提出了基于SOGDD的尺度选择策略，能够在尺度1–1.2内有效分辨相邻角点。

**🔧 技术方法**

采用的技术包括二阶高斯方向导数滤波、角点自相关矩阵构造与特征值分解、以及阈值化与非极大抑制的角点提取流程。

**📊 数据集**

使用的数据集包括：二维几何图像“Geometry”与“Table”，HPatches视角/光照变换数据集，以及五个三维重建数据集（Fountain、Herzjesu、South Building、Madrid Metropolis、Gendarmenmarkt）。

**📈 对比分析**

与Harris、FAST、Zhang(2013/2021/2023)、SuperPoint、D2‑Net等16种主流方法相比，本文方法在定位误差最低、平均可重复率最高、匹配准确率排名前列，并在三维重建中实现了最高或次高的点数、轨迹长度与重投影误差。

**⚠️ 局限性**

局限性主要体现在：在某些数据集的重投影误差仍未达到最优；方法对极端噪声或高压缩JPEG的鲁棒性尚需进一步提升；并且算法的计算复杂度相对较高，需要在实时场景中进行优化。

---

## 178. Prompt-Based Clarity Evaluation and Topic Detection in Political Question Answering

**arXiv ID:** 2601.08176 | [PDF](https://arxiv.org/pdf/2601.08176v1)

**作者:** Lavanya Prahallad `[一作]` (Research Spark Hub Inc), Pranathi Prahallad `[通讯]` (Emerald High School)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究提示工程对政治问答中自动化清晰度与回避检测的影响

**💡 创新点**

通过系统对不同提示策略（简单、链式思考、少样本）在GPT-5.2上进行对比，展示结构化提示对高层清晰度评估的提升，但对细粒度回避分类影响不稳定

**🔧 技术方法**

GPT‑5.2大语言模型、提示工程、层级标签评估（准确率、F1、层级精确匹配）、主题识别提示

**📊 数据集**

SemEval‑2026 CLARITY数据集（问答、清晰度/回避层级标签及主题标签）

**📈 对比分析**

与GPT‑3.5基线及不同提示配置比较；清晰度准确率从56%提升至63%（链式+少样本），主题识别从60%提升至74%，但回避细粒度准确率波动，整体仅提升至约32%

**⚠️ 局限性**

回避细粒度分类仍低，提示复杂度并不总是提升；仅在单一模型和单一数据集上验证；主题识别未覆盖多主题场景

---

## 179. The Agent's First Day: Benchmarking Learning, Exploration, and Scheduling in the Workplace Scenarios

**arXiv ID:** 2601.08173 | [PDF](https://arxiv.org/pdf/2601.08173v1)

**作者:** Daocheng Fu `[一作]` (Fudan University), Botian Shi `[通讯]` (Shanghai AI Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个名为 EvoEnv 的动态评估环境，用于模拟多模态大型语言模型（MLLM）在真实生产场景中的持续学习、主动探索和动态任务调度能力。

**💡 创新点**

创新点在于：①将评估从静态对照实验迁移到持续、随机、包含优先级的流任务；②通过隐藏线索与部分可观测性设计主动探索任务；③构建基于规则的元任务与动态组合场景，防止模型记忆化；④实现自动化验证与细粒度检查点反馈，支持持续学习。

**🔧 技术方法**

采用的技术包括：基于状态转移系统的环境建模、工具调用接口、动态任务生成器（随机化引擎）、自动化验证机制、交互协议封装以及多模态 LLM 的工具使用与推理。

**📊 数据集**

数据集：自建的 181 个元任务模板，按四大功能域划分；通过随机化引擎生成 50 个包含 2-6 个任务的动态场景；不使用公开数据集，所有任务均由规则生成。

**📈 对比分析**

通过 7 个主流 LLM（GPT‑5.1、GPT‑4o、Claude‑4‑Sonnet、Gemini‑3‑Flash、Grok‑4、Qwen3‑VL‑A235B、Llama‑4‑Maverick）在 50 个动态场景下进行对比，评价指标为成功率、检查点得分、平均步数和工具调用次数。结果显示，最佳模型 Gemini‑3‑Flash 的成功率仅为 35%，所有模型在复杂任务、主动探索和持续学习方面均显著不足；对比人类提示可将成绩从 0.24 提升至 0.83，凸显现有模型的局限。

**⚠️ 局限性**

限制包括：①任务组合多样性不足，缺乏复杂因果依赖；②依赖手工编写的规则，扩展性受限；③实验范围仅覆盖少数模型与单一工作场景；④未覆盖更广泛的工业/生产领域，缺乏跨域验证。

---

## 180. User-Oriented Multi-Turn Dialogue Generation with Tool Use at scale

**arXiv ID:** 2601.08225 | [PDF](https://arxiv.org/pdf/2601.08225v1)

**作者:** Jungho Cho `[一作]` (Upstage AI), Sungrae Park `[通讯]` (Upstage AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个可插拔的、以用户为中心的多轮对话生成框架，能够动态合成工具与任务，并在可执行的SQL环境中进行真实工具调用，生成高密度、真实感的对话数据。

**💡 创新点**

创新点在于：①将任务生成与用户模拟解耦，采用逐步请求与反馈机制逼近真实人机交互；②通过可执行的SQL接口实现工具调用的可验证性；③支持从任意中间状态启动，提升数据多样性与生成效率。

**🔧 技术方法**

核心技术包括：大模型驱动的工具与任务自动生成（使用GPT‑OSS‑120B等 LLM），基于规则的用户模拟器，SQL‑backed 交互环境，插件式生成流水线，Pass^k 评估指标。

**📊 数据集**

使用的数据集为：Nemotron（种子工具）、τ2（任务结构）、BFCL（基准评测），以及通过Spider 等公开数据自动合成的自定义数据库 schema。

**📈 对比分析**

与 Apigen、Nemotron 等基线相比，在 BFCL 与 τ2 基准上均取得明显提升，尤其在 τ2 的 Telecom 域表现最为突出；Pass^k 指标显示模型在重复尝试时保持更高的一致性。

**⚠️ 局限性**

主要限制是：生成过程需要多轮交互，导致更高的 GPU 计算成本与延迟；SQL 环境依赖数据库一致性，容易出现状态漂移或错误传播；当数据库视图不完整时模型易失真，需改进错误处理与状态恢复机制。

---

## 181. Hyperbolic Heterogeneous Graph Transformer

**arXiv ID:** 2601.08251 | [PDF](https://arxiv.org/pdf/2601.08251v1)

**作者:** Jongmin Park `[一作]` (Chungnam National University), Sungsu Lim `[通讯]` (Chungnam National University)

**通讯引用:** 742 | [OpenAlex ID](https://openalex.org/A5011058984)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 Hyperbolic Heterogeneous Graph Transformer（HypHGT），用于在负曲率的双曲空间中高效学习异构图的节点表示；

**💡 创新点**

创新点包括：①完全在双曲空间内实现注意力机制，避免频繁的切换到切线空间导致的映射失真；②为每种关系类型引入单独的双曲空间和可学习曲率，实现关系感知的注意力；③采用线性时间复杂度的双曲注意力与 Transformer 架构，结合局部邻域的 Heterogeneous GNN，兼顾全局层级结构与局部语义；

**🔧 技术方法**

技术手段包括：Lorentz 双曲模型、HT 与 HR 操作、线性双曲注意力（kernel + ReLU）、多头 Transformer、关系感知的双曲注意力、局部 Heterogeneous GNN（关系依赖的注意力聚合）、信息融合 λ 参数、交叉熵分类训练；

**📊 数据集**

使用了三大真实异构图数据集（IMDB、DBLP、ACM）以及基于 Barabasi–Albert 模型的合成异构图进行规模化测试；

**📈 对比分析**

与 8 种基线（GCN、GAT、HGCN、Hypformer、HAN、MAGNN、GTN、HGT、Simple‑HGN、SHAN、HHGAT、MSGAT）进行对比，HypHGT 在 Macro‑F1 与 Micro‑F1 上均获得最高分，同时在训练时间与内存消耗上显著低于大多数对手，尤其在大规模合成图上表现出线性增长的优良可扩展性；

**⚠️ 局限性**

局限性包括：对动态异构图和跨域迁移的适应性尚未验证；需要手动调节 λ 与关系数目等超参数；虽然线性复杂度，但仍受关系类型数量和嵌入维度的影响；以及对极大规模图的最终极限仍待进一步探索。

---

## 182. Scoping Review: Mental Health XR Games at ISMAR, IEEEVR, & TVCG

**arXiv ID:** 2601.08203 | [PDF](https://arxiv.org/pdf/2601.08203v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 183. Statistical Blendshape Calculation and Analysis for Graphics Applications

**arXiv ID:** 2601.08234 | [PDF](https://arxiv.org/pdf/2601.08234v1)

**作者:** Shuxian Li `[一作]` (Lenovo Research), Chris Twombly `[通讯]`

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

用标准摄像头实时将面部关键点转换为3D虚拟人blendshapes，实现低功耗实时表情动画。

**💡 创新点**

独立开发的统计模型将Landmark转blendshapes，低硬件需求、少训练数据、引入多步骤Affine、分割、变换、非线性回归与自适应平滑，性能接近ARKit6。

**🔧 技术方法**

使用MediaPipe Holistic 0.9.0.1关键点检测、Affine变换、分段矩阵、PCA/指数等数据变换、多种回归模型（OLS、PCA、PLS、GP、SVR、集成）以及自定义偏置校正与门控平均平滑，辅以统计检验。

**📊 数据集**

采用131张iPhone12采集图像进行分割、200-600张Unreal Engine 4 Metahuman合成头像图像用于训练，21段iPhone14摄像头采集的18209帧作为测试数据，并与Apple ARKit6进行对比。

**📈 对比分析**

通过Pearson、Spearman、ξ相关性和改良F1得分进行比较，平均F1仅比ARKit差9.53%，部分blendshape实现100%精度；CPU占用约40‑81%，内存约550MB，响应时延0.86‑2.46ms，适用于低功耗设备且性能接近ARKit。

**⚠️ 局限性**

受MediaPipe Holistic关键点分辨率与噪声限制，左侧blendshape精度低于右侧；对极端表情或大幅度头部运动仍不稳定；缺乏完整真值数据，模型对不同面孔的泛化需要进一步优化。

---

## 184. FUME: Fused Unified Multi-Gas Emission Network for Livestock Rumen Acidosis Detection

**arXiv ID:** 2601.08205 | [PDF](https://arxiv.org/pdf/2601.08205v1)

**作者:** Taminul Islam `[一作]` (Southern Illinois University), Amer AbuGhazaleh `[通讯]` (Southern Illinois University)

**通讯引用:** 2291 | [OpenAlex ID](https://openalex.org/A5019065958)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

利用双气体（CO₂ 与 CH₄）的光学气体成像，提出 FUME 网络对奶牛瘤胃酸中毒进行同时的气体雾分割和健康状态分类，验证在体外实验中的可行性。

**💡 创新点**

创新点包括：①首个双气体 OGI 数据集，包含 8,967 张标注帧；②轻量级双流架构，采用共享 Fast‑SCNN 编码器、模态自注意力和通道注意力融合；③多任务学习（分割 + 分类）实现互补监督；④实验表明 CO₂ 是主导诊断信号，CH₄ 主要用于空间精度提升。

**🔧 技术方法**

使用深度学习技术：Fast‑SCNN 编码器、单头自注意力、通道注意力融合、双任务解码器；优化目标为 Focal+Dice 分割损失与 Focal 分类损失的加权组合；采用 AdamW、余弦退火、数据增强等训练策略。

**📊 数据集**

使用 8,967 张双气体光学成像帧组成的 OGI 数据集，覆盖 6 个 pH 水平（6.5、6.2、5.9、5.6、5.3、5.0），并按健康状态（Healthy、Transitional、Acidotic）划分；每帧带有像素级背景/管道/气体三类掩码。

**📈 对比分析**

与多种通用分割基线（BiSeNetV2、ENet、DDRNetSlim、ESPNetV2 等）和 OGI 专用模型（Gasformer、GasTwinFormer、CarboFormer）对比；FUME 在 mIoU 上达到 80.99%（最高）、分类准确率 98.82%，参数仅 1.28M，MACs 1.97G，推理速度 326.8 FPS，显著低于其他模型，显示出最佳的质量-效率折中。

**⚠️ 局限性**

主要局限：仅在体外发酵系统中验证；未考虑动物运动、现场环境波动及气体可见性不稳定等实际场景；注意力模块在某些设置下对分割质量有负面影响，需进一步调优；未来需在真实牧场中进行临床验证。

---

## 185. MobiDiary: Autoregressive Action Captioning with Wearable Devices and Wireless Signals

**arXiv ID:** 2601.08204 | [PDF](https://arxiv.org/pdf/2601.08204v1)

**作者:** Fei Deng `[一作]` (Xi'an Jiaotong University), Fei Wang `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 35350 | [OpenAlex ID](https://openalex.org/A5100455803)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一种从 IMU 与 Wi‑Fi CSI 信号直接生成自然语言动作描述的框架 MobiDiary。

**💡 创新点**

创新点在于：① 统一传感器编码器将两种不同物理信号映射到同一特征空间；② 采用补丁化、位置嵌入和 Conv‑FFN 模块提升局部语义与跨设备建模；③ 用自回归 Transformer 解码器生成连贯的动作字幕，且在训练时使用教师强制策略。

**🔧 技术方法**

技术方案包括 Patch‑based sensor encoder、placement embedding、Conv‑FFN、Transformer decoder、教师强制训练与自回归生成。

**📊 数据集**

使用了 XRF V2、UWash 与 WiFiTAD 三大公开数据集，分别包含 IMU 与 Wi‑Fi CSI 信号。

**📈 对比分析**

与多种基线（LSTM、Transformer、XRFMamba、PatchTST、TimeLLM、SensorLLM、LLaSA 等）进行对比，MobiDiary 在 BLEU@4、METEOR、ROUGE‑L、CIDEr 以及 RMC 上均取得最优或接近最优成绩，证明其在多模态动作字幕任务上的领先性能。

**⚠️ 局限性**

局限性包括：仅在日常生活场景的数据集上验证，缺乏工业、医疗等更复杂环境的测试；对极长序列和高噪声 Wi‑Fi 信号的鲁棒性仍有提升空间。

---

## 186. Evaluating Implicit Regulatory Compliance in LLM Tool Invocation via Logic-Guided Synthesis

**arXiv ID:** 2601.08196 | [PDF](https://arxiv.org/pdf/2601.08196v1)

**作者:** Da Song `[一作]` (Shandong University), Foutse Khomh `[通讯]` (Polytechnique Montreal)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

做了一个自动化框架，能够将未结构化的监管文本转换为线性时序逻辑或acular，利用逻辑引导的模糊测试合成满足安全约束的可执行轨迹，并通过安全遮蔽将轨迹转化为隐式合规的用户指令，从而生成240个人工验证的LLM工具调用基准。

**💡 创新点**

创新点在于①将监管规则自动映射到可验证的LTL公式；②利用基于约束满足的模糊测试系统生成既合法又满足安全约束的执行路径；③通过安全遮蔽迫使LLM在缺少显式安全提示的情况下自行推断隐式合规约束，从而在评测中揭示LLM在隐式合规上的缺陷。

**🔧 技术方法**

使用的技术包括：LTL语法解析与验证、基于约束满足的逻辑引导模糊测试引擎、LLM辅助的规则提取与指令合成、隐式安全遮蔽与指令类型化（目标导向/工作流导向）。

**📊 数据集**

数据集主要来自真实监管文件（欧盟支付服务指令2、美国HIPAA安全规则、欧洲电信标准EN 303 645），并结合工具规范（ToolEmu等）生成240个手工核验的任务。

**📈 对比分析**

与仅用LLM生成或现有基准对比，采用安全关键API覆盖率（S.C. Cov）和相邻转移覆盖率（ATC）衡量多样性，逻辑模糊测试实现100%安全API覆盖率、ATC最高；在LLM评测中，前沿商用模型Pass@1约为60‑80%，但在安全API密度高的场景下性能明显下滑。

**⚠️ 局限性**

限制包括：①仅支持两类LTL模板，无法覆盖参数级或概率性约束；②生成多样性仍受LLM先验约束，极端结构或边缘案例仍需人工干预；③安全遮蔽可能导致任务难度过高或不现实。

---

## 187. Improving LLM Reasoning with Homophily-aware Structural and Semantic Text-Attributed Graph Compression

**arXiv ID:** 2601.08187 | [PDF](https://arxiv.org/pdf/2601.08187v1)

**作者:** Zijun Di `[一作]` (Shanghai Jiao Tong University), Chenghu Zhou `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 16599 | [OpenAlex ID](https://openalex.org/A5023919188)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种同质性社区驱动的TAG压缩框架 HS_2C，用来提高 LLM 在图文本推理任务中的准确率与压缩效率。

**💡 创新点**

创新点在于结合结构熵最小化的全局层次划分与同质社区的语义聚合，兼顾结构与语义同质性，避免随机采样导致的噪声与推理不稳定。

**🔧 技术方法**

使用结构熵、KNN 结构增强、层次编码树、LLM 语义聚合等技术实现全局同质社区检测与语义压缩。

**📊 数据集**

在 10 个节点级 TAG 基准（OGBN‑ArXiv、TAPE、Citeseer、Reddit 等）以及 7 个图级任务（如 Graph‑SST、Molecular 等）上进行评估。

**📈 对比分析**

与 11 种基线（GNN、采样、Skeleton 等）比较，HS_2C 在压缩率、准确率和综合指标 GCI 上均显著提升，尤其在 OGBN‑ArXiv 上实现 94.98% 的压缩率并提升 3–5% 的准确率。

**⚠️ 局限性**

局限性包括：在极大规模图上计算成本仍较高；目前仅针对静态图，未考虑动态图的实时同质社区更新和语义聚合。

---

## 188. CogniMap3D: Cognitive 3D Mapping and Rapid Retrieval

**arXiv ID:** 2601.08175 | [PDF](https://arxiv.org/pdf/2601.08175v1)

**作者:** Feiran Wang `[一作]` (University of Illinois Chicago), Yan Yan `[通讯]` (University of Illinois Chicago)

**通讯引用:** 19243 | [OpenAlex ID](https://openalex.org/A5100395068)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出CogniMap3D框架，实现对单目视频中动态场景的实时三维重建、深度估计与相机位姿重建，并通过持续记忆实现多次访问场景的快速检索与更新。

**💡 创新点**

创新点在于：①多阶段运动线索框架实现动态物体与静态背景的精确分离；②基于视觉基础模型的记忆库实现静态场景的持续存储、检索与更新；③通过因子图优化在全局一致性约束下细化相机位姿，兼容基础模型输出。

**🔧 技术方法**

使用技术包括视觉基础模型（VGGT）、光流聚类、几何运动分析、SAM2动态掩膜、PointNet+++octree 3D特征、DINOv2 2D特征、ICP对齐、Levenberg–Marquardt 因子图优化及Huber损失鲁棒性。

**📊 数据集**

主要使用数据集：Sintel、KITTI、Bonn、DAVIS、7-Scenes、TUM-dynamics、ScanNet 等，覆盖合成、户外、室内及动态场景。

**📈 对比分析**

与 Spann3R、MonST3R、CUT3R、VGGT 等基线比较，CogniMap3D 在视频深度估计、相机位姿（ATE、RPE）和三维重建（Acc/Comp/NC）上均达到或超过最先进方法，尤其在多次访问与记忆更新场景中表现突出。

**⚠️ 局限性**

局限性包括：①记忆库维护与检索引入一定计算开销；②对基础模型的依赖导致在极端光照或缺失纹理场景下性能受限；③动态物体分离阈值设置仍需经验调参，可能在复杂动态环境中出现误检。

---

## 189. Enhancing Sentiment Classification and Irony Detection in Large Language Models through Advanced Prompt Engineering Techniques

**arXiv ID:** 2601.08302 | [PDF](https://arxiv.org/pdf/2601.08302v1)

**作者:** Marvin Schmitt `[一作]` (IU International University of Applied Sciences), Sebastian Lempert `[通讯]` (IU International University of Applied Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究高级提示工程（few-shot、链式推理、self-consistency）对大型语言模型在情感分析任务（包括二分类、多分类、基于方面的情感分析、讽刺检测）性能的提升效果。

**💡 创新点**

系统评估不同提示策略对两款不同架构LLM（GPT‑4o‑mini 与 Gemini‑1.5‑flash）的任务特异化影响，并发现提示策略需要针对模型和任务匹配。

**🔧 技术方法**

使用基于文本提示的few-shot、chain‑of‑thought (CoT) 与 self‑consistency 提示；通过OpenAI与Google的API调用实现模型推理。

**📊 数据集**

使用四个公开数据集：SST‑2（电影评论二分类）、SB‑10k（德语推文三分类）、SemEval‑2014 ABSA（多方面情感）、SemEval‑2018‑Irony（讽刺检测）。

**📈 对比分析**

与零shot基线对比，通过准确率、召回率、F1分数及bootstrap置信区间评估。结果显示few‑shot在GPT‑4o‑mini上最显著提升；Gemini‑1.5‑flash在CoT提示下讽刺检测提升高达46%；整体提示策略均能显著优于基线，但效果差异依赖模型与任务。

**⚠️ 局限性**

仅评估两款专有LLM，样本量受限（每任务1000条），提示设计手工且未进行系统化消融，且未探讨参数-提示交互、语言偏差及多模型泛化。

---

## 190. ReCo-KD: Region- and Context-Aware Knowledge Distillation for Efficient 3D Medical Image Segmentation

**arXiv ID:** 2601.08301 | [PDF](https://arxiv.org/pdf/2601.08301v1)

**作者:** Qizhen Lan `[一作]` (University of Texas Health Science Center at Houston), Xiaoqian Jiang `[通讯]` (University of Texas Health Science Center at Houston)

**通讯引用:** 13802 | [OpenAlex ID](https://openalex.org/A5055458864)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种仅在训练阶段使用的区域与上下文感知知识蒸馏框架ReCo-KD，用于压缩3D医学图像分割模型。

**💡 创新点**

创新点在于结合多尺度结构感知区域蒸馏(MS-SARD)和多尺度上下文对齐(MS-CA)，同时解决小体素不平衡和全局上下文一致性问题。

**🔧 技术方法**

采用nnU-Net自适配框架、通道宽度缩放、类感知掩膜、尺度归一化权重、注意力激活对齐以及全局上下文GC块实现蒸馏。

**📊 数据集**

在BTCV、Hippocampus、BraTS2021和一个包含110个脑区的大规模聚合数据集上进行评估。

**📈 对比分析**

与基准CNN/Transformer模型和多种蒸馏方法比较，ReCo-KD在保持与教师模型相近的Dice/NSD的同时，参数、FLOPs和CPU推理时间分别减少约94%、93%和71%，并在小结构上显著提升。

**⚠️ 局限性**

仅在nnU-Net统一宽度缩放下评估，未探索更广泛的学生架构或跨模态蒸馏，可能低估了蒸馏潜力。

---

## 191. Self-Certification of High-Risk AI Systems: The Example of AI-based Facial Emotion Recognition

**arXiv ID:** 2601.08295 | [PDF](https://arxiv.org/pdf/2601.08295v1)

**作者:** Gregor Autischer `[一作]` (Graz University of Technology), Dominik Kowald `[通讯]` (Know Center Research GmbH)

**通讯引用:** 1062 | [OpenAlex ID](https://openalex.org/A5071624510)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对面部情感识别AI系统进行完整的自我认证循环，使用Fraunhofer AI评估目录对可靠性与公平性进行评估，并基于认证结果对模型架构、训练方法和数据集进行迭代改进，最终得到改进版模型并记录技术改进过程。

**💡 创新点**

创新点在于将自我认证框架作为主动的系统开发工具，通过认证问题驱动技术改进；展示了在缺乏正式欧盟标准时，内部评估可为后续合规奠定基础，并揭示了文档式认证与技术测试之间的互补关系。

**🔧 技术方法**

使用PyTorch实现的卷积神经网络（CNN）架构，配合CosineAnnealingWarmRestarts、AdamW、WeightedCrossEntropyLoss、BatchNorm、Dropout、GlobalAvgPool等技术；训练中引入权重采样、梯度裁剪等手段；采用置信度、熵、宏F1等多维度评价指标。

**📊 数据集**

数据集包括FER‑2013、CK+、RAF‑DB，并利用FairFace生成性别、种族、年龄等属性标签；通过数据增强（旋转、光照、噪声、模糊、遮挡等）扩展样本量，提升模型鲁棒性。

**📈 对比分析**

与基线模型（准确率53.88%、置信度约27%）相比，改进模型准确率提升至68.19%，宏F1提升至0.676；置信度提高到78.7%，熵下降到0.53；在性别、种族和年龄组的公平性差距均控制在可接受范围内，鲁棒性测试中仅模糊样本误差最大。

**⚠️ 局限性**

局限性包括：仍未满足欧盟AI法正式合规的必要标准（缺乏已发布的欧盟标准及合规程序）；对70岁以上人群的性能仍不理想；自我认证虽可指导技术改进，但不能替代正式第三方评估与法规申报流程。

---

## 192. OpenMic: A Multi-Agent-Based Stand-Up Comedy Generation System

**arXiv ID:** 2601.08288 | [PDF](https://arxiv.org/pdf/2601.08288v1)

**作者:** Yuyang Wu `[一作]` (Peking University), Yufei Li `[通讯]` (Peking University)

**通讯引用:** 2951 | [OpenAlex ID](https://openalex.org/A5100707302)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了基于AutoGen的多智能体端到端中文单口喜剧生成系统OpenMic，能够把用户提供的生活主题转换成3‑5分钟的完整舞台脚本甚至视频；

**💡 创新点**

①将单口喜剧拆解为观众分析、剧本规划、写作、演绎标注与质量控制五个专属智能体，形成可迭代的创作流程；②采用检索增强生成（RAG）+三层内循环（检索→LLM评分→精炼）来获取文化根植素材；③对JokeWriter进行QLoRA微调，专门学习setup–punchline延迟、回调等喜剧结构；④使用结构化DSL标注表演节奏，并直接转成可渲染的视频；

**🔧 技术方法**

AutoGen多智能体框架、检索增强生成（RAG）、QLoRA参数高效微调、LLM‑as‑a‑Judge评测、多维度结构化DSL脚本、视频合成中间件（Kling AI等）等技术；

**📊 数据集**

CFUN短篇搞笑对话、改写后的跨口秀脚本、隐私化与风格转换后的传统跨口秀对话库，用于检索与素材扩展；微调数据集基于Qwen‑2.5‑3B‑Instruct；

**📈 对比分析**

使用LLM‑as‑a‑Judge的五维评分（Persona、Humor、Reactivity、Coherence、Narrative）对不同温度与是否使用RAG进行对比；实验表明低温度+RAG组合在幽默机制、节奏与一致性得分最高（约88–98分），明显优于单模型直生或高温度生成；

**⚠️ 局限性**

①仍受限于预训练LLM，可能出现文化差异或偏见；②评测主要基于LLM，缺乏真实观众反馈；③对跨语言推广与多样化主题的适应性有限；④对实时表演调度和情绪动态的细粒度控制尚未完善；

---

## 193. Greedy Is Enough: Sparse Action Discovery in Agentic LLMs

**arXiv ID:** 2601.08280 | [PDF](https://arxiv.org/pdf/2601.08280v1)

**作者:** Angshul Majumdar `[一作]` `[通讯]`, Angshul Majumdar

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

研究在极大动作空间（如工具调用、API、文档检索等）下的稀疏动作发现问题，提出将动作选择视为块稀疏恢复并给出理论保证。

**💡 创新点**

创新点在于：1) 将稀疏动作发现映射为Contextual Block‑OMP（基于正交匹配追踪）的块稀疏恢复问题；2) 在标准不可辨性、信号强度和覆盖条件下证明贪心算法能以样本复杂度 O(kd log M) 恢复真相；3) 给出信息理论下界，证明在无稀疏性或无足够覆盖时样本复杂度必需线性于 M。

**🔧 技术方法**

使用的技术包括：块正交匹配追踪（Contextual Block‑OMP）、矩阵集中与子高斯分布、不可辨性（irrepresentability）条件、最小二乘重拟合、Kullback‑Leibler、Fano 与 Bretagnolle–Huber 信息理论下界、矩阵不变性与Rayleigh商等。

**📊 数据集**

该工作为理论研究，未使用公开数据集；实验验证被省略，主要以理论证明为主。

**📈 对比分析**

方法比较主要是理论上的对比：与无稀疏性情况下样本复杂度线性 M 的上限相比，Block‑OMP 在稀疏假设下仅需对数级；对比传统启发式工具选择或凸稀疏正则化方法的理论上限尚未给出，但理论框架表明贪心方法可达最优。

**⚠️ 局限性**

局限性包括：假设奖励是已知低维状态的线性函数；动作序列与状态假设相互独立；需要足够覆盖（每个相关动作至少被采样一定次数）；未考虑非线性或非高斯噪声、序列状态动力学以及LLM内部表示学习的联合优化；实际部署中可能出现工具效果的时变性和模型不匹配。

---

## 194. Matrix-PIC: Harnessing Matrix Outer-product for High-Performance Particle-in-Cell Simulations

**arXiv ID:** 2601.08277 | [PDF](https://arxiv.org/pdf/2601.08277v1)

**作者:** Yizhuo Rao `[一作]` (Sun Yat-Sen University), Yutong Lu `[通讯]` (Sun Yat-Sen University)

**通讯引用:** 5569 | [OpenAlex ID](https://openalex.org/A5101633465)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 MatrixPIC 框架，重新设计 PIC 仿真中的粒子-格子散射（current deposition）阶段，使其可在集成了矩阵处理单元（MPU）的 CPU 上实现高效并行。

**💡 创新点**

创新点包括：① 将散射计算映射为矩阵外积操作，充分利用 MPU 的高密度算力；② 设计 VPU‑MPU 混合执行流水线，协调数据准备与高吞吐量累加；③ 开发 O(1) 递增排序的 GPMA 数据结构，实现粒子局部性保持与低成本重排。

**🔧 技术方法**

使用技术包括：MPU 外积指令、VPU SIMD 指令、GPMA（Gapped Packed Memory Array）递增排序、基于 WarpX 的粒子代码、自动向量化与手工 intrinsics 混合优化。

**📊 数据集**

在统一等离子体（Uniform Plasma）和激光波前加速（LWFA）两类仿真场景中，采用粒子/格子密度 1–128 的测试集，并在 3D CIC 与 3阶 QSP 形状函数下进行评测。

**📈 对比分析**

与 WarpX 基线和手工向量化版本比较，MatrixPIC 在 CIC 时实现 1.19× 总仿真速度提升、2.63× LWFA 总时长加速；在 3阶 QSP 时实现 8.7× 内核加速；CPU 上达到 83.08% 最高峰值利用率，约为 GPU CUDA 实现的 2.8 倍。

**⚠️ 局限性**

局限性在于仅支持直接散射（无电荷守恒方案）、实现仅针对部署在 MPUs 上的 CPU，且优化聚焦于散射阶段；未来需扩展到更复杂的守恒散射、收敛到更广泛硬件以及实现低粒子密度下的自适应回退。

---

## 195. Discovery and Reinforcement of Tool-Integrated Reasoning Chains via Rollout Trees

**arXiv ID:** 2601.08274 | [PDF](https://arxiv.org/pdf/2601.08274v1)

**作者:** Kun Li `[一作]` (Chinese University of Hong Kong), Bo Zhou `[通讯]` (Tencent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大型推理模型中训练模型实现长链推理时自发调用工具，而不需要任何手工标注。

**💡 创新点**

创新点在于提出动态回溯树构造与树式优势估计，利用高熵位置进行工具调用探测，实现无标注的工具整合。

**🔧 技术方法**

采用强化学习框架，利用回溯树生成器、Python 代码解释器、熵基分叉、树型优势计算等技术。

**📊 数据集**

训练数据为从 DAPO 数据集中筛选的 2795 条数学题；评估使用 AIME24、AIME25、GPQA‑Diamond 等基准。

**📈 对比分析**

与 ToRL、UTIR、ReTool、START 等基线相比，DART 在 Pass@1/Pass@8 以及代码调用频率上均显著提升，表明工具整合效果更好。

**⚠️ 局限性**

局限在于只验证 Python 解释器与数理/科学推理，未扩展到事实核查、创作等领域，且回溯树训练仍存在额外延迟。

---

## 196. Unleashing Tool Engineering and Intelligence for Agentic AI in Next-Generation Communication Networks

**arXiv ID:** 2601.08259 | [PDF](https://arxiv.org/pdf/2601.08259v1)

**作者:** Yinqiu Liu `[一作]` (Nanyang Technological University), Dong In Kim `[通讯]` (Sungkyunkwan University)

**通讯引用:** 23480 | [OpenAlex ID](https://openalex.org/A5022649488)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统综述工具工程，为下一代通信网络中的代理式AI（Agentic AI）提供工具智能（Tool Intelligence）框架，并以无人机轨迹规划为案例，验证基于教师引导强化学习（Teacher‑Guided PPO）的工具激活策略的有效性。

**💡 创新点**

创新点包括：①提出完整的工具工程生命周期（创建、发现、选择、学习与基准化）；②将工具智能嵌入四大核心组件（感知、推理、行动、记忆）并阐释其对通信功能与效率的提升；③在无人机场景中首次结合标准工具与语义工具，设计能量约束下的工具激活决策；④引入教师盾（Shield）实现能量预测与即时奖励塑造，显著加速收敛。

**🔧 技术方法**

主要技术包括：大型语言模型（LLM）推理引擎；模型上下文协议（MCP）用于工具注册与调用；多智能体协同与工具链调度；深度强化学习（PPO）与教师引导奖励机制；模拟环境与噪声注入实现无人机运动模型；工具激活的能量成本模型。

**📊 数据集**

数据集：使用仿真生成的无人机运动轨迹、工具可用性地图、标准与语义工具的功能参数；未使用公开真实通信或无人机数据集，实验基于自行搭建的仿真平台。

**📈 对比分析**

比较方法：与 Vanilla PPO、随机策略、贪心策略（主动激活可用工具）以及成本感知启发式（仅在能量足够时激活）对比。结果显示：教师引导 PPO 在任务完成率、总奖励、能量消耗与工具激活准确性方面均优于基线，最高奖励为 664.9，显著高于最优基线（约 542.5）。

**⚠️ 局限性**

局限性：①实验仅基于仿真，缺乏真实网络与无人机场景验证；②工具集仅包含两类工具，未覆盖更丰富的通信场景工具；③教师盾实现需事先知道能量消耗模型，实际部署可能面临模型误差；④未探讨工具执行的安全性与可信度（Zero‑Trust 机制）。

---

## 197. T3: Benchmarking Sycophancy and Skepticism in Causal Judgment

**arXiv ID:** 2601.08258 | [PDF](https://arxiv.org/pdf/2601.08258v1)

**作者:** Edward Y. Chang `[一作]` (Stanford University), Edward Y. Chang `[通讯]` (Stanford University)

**通讯引用:** 18291 | [OpenAlex ID](https://openalex.org/A5013545831)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了T3诊断基准，用以评估LLM在Pearl因果层次（关联、干预、反事实）上的因果判断能力。

**💡 创新点**

创新点在于引入Utility/Safety分解与Wise Refusal三分法，揭示“怀疑陷阱”和“缩放悖论”等新的对齐路径障碍。

**🔧 技术方法**

采用专家制定的454个自然语言情境、三种提示协议（Neutral Direct、Epistemic Permissiveness、Adversarial Pressure）以及Recursive Causal Audit (RCA) 过程验证技术。

**📊 数据集**

使用454个专家标注的vignettes数据集，覆盖10个领域、12类因果陷阱，作为评测样本。

**📈 对比分析**

通过在前沿模型（GPT‑4‑Turbo、GPT‑5.2、Claude 3.5 Haiku等）上测量Utility、Safety和Wise Refusal，发现GPT‑5.2在L3上比GPT‑4‑Turbo低55点，RCA能显著提升表现。

**⚠️ 局限性**

局限在于样本量有限、因果标签主观性强、仅在固定提示下评测、结果不一定适用于高风险领域。

---

## 198. Detecting Mental Manipulation in Speech via Synthetic Multi-Speaker Dialogue

**arXiv ID:** 2601.08342 | [PDF](https://arxiv.org/pdf/2601.08342v1)

**作者:** Run Chen `[一作]` (Columbia University), Julia Hirschberg `[通讯]` (Columbia University)

**通讯引用:** 21259 | [OpenAlex ID](https://openalex.org/A5045037642)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建SpeechMentalManip基准，将文本基准MentalManip中的电影对话通过高质量多说话人TTS生成对齐音频，并使用大型音频语言模型（Qwen2.5-Omni-7B）和人类重注释评估在语音对话中检测心理操纵的任务。

**💡 创新点**

① 将文本基准扩展为语音基准，首次系统性探讨语音模态对心理操纵检测的影响；② 采用两阶段TTS生成流程实现声学一致、连贯的多说话人音频；③ 对文本与语音两模态进行严格对比，揭示精确率高但召回率低的模态差异。

**🔧 技术方法**

两阶段Text-to-Speech合成（ElevenLabs多说话人声音），Qwen2.5-Omni-7B音频语言模型（few-shot + 约束解码），人类重注释（Cohen/Fleiss/Krippendorff一致性评估）。

**📊 数据集**

SpeechMentalManip（来源自MentalManip共2,915条对话，生成609条含操纵音频+90条非操纵音频）以及100条人类重注释样本。

**📈 对比分析**

通过在文本和音频两套数据上执行few-shot检测，使用准确率、召回率、F1评估；结果显示音频检测精确率高（≈0.85）但召回率仅34.8%，相较文本检测召回率更高；人类在音频上的Kappa仅0.42，文本约0.53，模型在音频上表现更保守。

**⚠️ 局限性**

① 语音合成缺乏自然交谈特征（重叠、停顿、背景噪声等）；② 仅评估单一模型和提示策略，缺乏多模型对比；③ 标签基于文本，可能与语音表达不一致；④ 合成音频的多样性和自然度有限，影响模型与人类的可比性。

---

## 199. IGAN: A New Inception-based Model for Stable and High-Fidelity Image Synthesis Using Generative Adversarial Networks

**arXiv ID:** 2601.08332 | [PDF](https://arxiv.org/pdf/2601.08332v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 200. From Antenna Abundance to Antenna Intelligence in 6G Gigantic MIMO Systems

**arXiv ID:** 2601.08326 | [PDF](https://arxiv.org/pdf/2601.08326v1)

**作者:** Emil Björnson `[一作]` (KTH Royal Institute of Technology), Vitaly Petrov `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 3718 | [OpenAlex ID](https://openalex.org/A5035321474)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究在6G网络中利用非均匀稀疏天线阵列和可移动天线（MA）实现与传统均匀阵列相当甚至更高的多用户MIMO容量，重点比较不同阵列设计对上行峰值速率的影响。

**💡 创新点**

创新点在于将定位/感知领域的最小冗余阵列、稀疏阵列稀疏化以及预优化不规则阵列等概念迁移至通信场景，并首次展示可移动天线在实时CSI下调节副瓣、实现几乎达到无干扰极限的效果。

**🔧 技术方法**

采用RZF波束成形、理想CSI、Rayleigh / Rician 随机信道模型、粒子群优化、仿真仿真器以及基于射线追踪的数字孪生来设计并评估阵列。

**📊 数据集**

数据集主要为合成的无线信道数据：在LOS、Rician 六簇、K因子8dB、3GHz 频带下生成多达50个15 kHz子载波的多用户落位，亦使用仿真生成的多角度随机用户落位。

**📈 对比分析**

比较方法是将同一覆盖区域内的多种阵列（均匀ULA/UPA、稀疏ULA/UPA、MRA、稀疏化、预优化不规则阵列（PIA）、可移动天线（MA））在相同天线数、相同总辐射功率下通过RZF上行链路平均总速率和CDF曲线进行对比。MA阵列在所有仿真场景下均取得最高速率，甚至用16天线的MA/PIA/2D‑PMRA超过了64天线均匀UPA，速率提升幅度可达124%。

**⚠️ 局限性**

局限性包括：1）仍无法完全逼近无干扰上界，主瓣宽度受限；2）可移动天线实现成本与复杂度高；3）仿真基于理想CSI，实际链路中的估计误差与硬件非理想会进一步降低性能；4）仅在特定覆盖区间与天线摆放约束下验证，泛化到更大规模或不同环境需进一步研究。

---

## 201. ActiveVLA: Injecting Active Perception into Vision-Language-Action Models for Precise 3D Robotic Manipulation

**arXiv ID:** 2601.08325 | [PDF](https://arxiv.org/pdf/2601.08325v1)

**作者:** Zhenyang Liu `[一作]` (Fudan University), Yanwei Fu `[通讯]` (Fudan University)

**通讯引用:** 16344 | [OpenAlex ID](https://openalex.org/A5084959430)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了ActiveVLA框架，实现了主动感知的vision-language-action模型，能够在任务执行过程中主动选择视角和进行3D放大，完成细粒度操控。

**💡 创新点**

核心创新是两阶段粗细化主动感知：先投影多视角的3D点云到2D正交图像并通过热图定位关键区域，然后在该区域主动选择最佳视角并进行3D放大，显著提升分辨率和信息量。

**🔧 技术方法**

技术手段包括使用PaliGemma VLM backbone、三正交投影、热图预测、KD树可见性检查、几何均匀视角采样、虚拟光学放大、全局-局部特征融合以及多任务动作解码。

**📊 数据集**

在RLBench、COLOSSEUM和GemBench三大仿真基准上进行训练与评估，并在真实KinoVA GEN2 + RealSense D455机器人上验证其跨域泛化能力。

**📈 对比分析**

与Image‑BC、C2F‑ARM‑BC、HiveFormer、PolarNet、Act3D、3D Diffuser Actor、RVT、BridgeVLA等多种基线进行对比；在RLBench平均成功率91.8%/平均排名1.22，在COLOSSEUM平均成功率65.9%/平均排名1.07，在GemBench平均成功率51.3%，均优于所有现有方法。

**⚠️ 局限性**

局限性包括对多视角数量和放大因子的超参数依赖、计算开销相对较高，以及在极端遮挡或动态环境下仍可能出现失效；此外仍需进一步探索在更大规模多模任务中的迁移与扩展。

---

## 202. Characterizing Personality from Eye-Tracking: The Role of Gaze and Its Absence in Interactive Search Environments

**arXiv ID:** 2601.08287 | [PDF](https://arxiv.org/pdf/2601.08287v1)

**作者:** Jiaman He `[一作]` (RMIT University), Noriko Kando `[通讯]` (National Institute of Informatics)

**通讯引用:** 3388 | [OpenAlex ID](https://openalex.org/A5045875579)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究利用iPad博物馆搜索任务中的眼动追踪时间序列及其缺失信息，构建多模态时序模型以预测参与者的五大人格特质。

**💡 创新点**

创新点在于将眼动缺失（用户眨眼或移开视线时的空窗期）视为有价值的行为信号，并将其与眼动坐标、瞳孔直径、速度等特征一起输入时序神经网络进行学习。

**🔧 技术方法**

采用了双向长短时记忆网络（BiLSTM）与缺失掩码、时间间隔特征相结合的特征增强方案，直接处理原始眼动数据，无需繁琐的预处理。

**📊 数据集**

数据集来自25名参与者在Minpaku Guide应用上进行的交互式搜索实验，采集了60 Hz眼动追踪数据以及交互日志，构成了眼动时间序列与缺失标记的训练集。

**📈 对比分析**

实验通过5折交叉验证与按参与者划分的严格验证，对比仅使用时序特征、仅使用时序+时间间隔、仅使用时序、以及基于统计特征的随机森林模型；全模型在宏F1上平均约73–78%，显著优于所有对照方法。

**⚠️ 局限性**

主要局限包括样本量仅25人、仅单一博物馆搜索任务、缺失信息解释受限于实验情境、以及模型在未见个体上的泛化能力尚未充分验证。

---

## 203. CLaS-Bench: A Cross-Lingual Alignment and Steering Benchmark

**arXiv ID:** 2601.08331 | [PDF](https://arxiv.org/pdf/2601.08331v1)

**作者:** Daniil Gurgurov `[一作]` (German Research Center for Artificial Intelligence), Simon Ostermann `[通讯]` (Centre for European Research in Trusted AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了CLaS-Bench，构建了一个轻量化的平行问题基准，用于在32种语言上评估LLM的语言驱动技术。

**💡 创新点**

创新点在于首次提供跨语言的标准化驱动评估框架，并发现简单的残差差均值（DiffMean）方法在多语种场景中最为有效，同时揭示语言在内部表示中按族群聚类的现象。

**🔧 技术方法**

使用的技术包括残差流的DiffMean干预、神经元驱动（LAPE）、线性探针、PCA/LDA、稀疏自编码器、以及基准提示；评估时采用FastText语言识别和LLM-as-judge评分。

**📊 数据集**

数据集由Vicuna中挑选的70个开放式问题翻译成31种语言（共32种），并通过本地校对保证质量；此外，为各方法生成对齐表示还使用了每种语言的1000万token。

**📈 对比分析**

通过计算语言强制成功率（F）与输出相关性（R）的调和平均值（S）对比，DiffMean平均得分84.5%，显著高于提示基线（≈67%）和其他方法；LAPE次之（80.1%），其余方法的得分依次降低。

**⚠️ 局限性**

局限性包括不同方法使用的训练样本量差异、稀疏自编码器仅覆盖部分层和模型、仅包含32种语言、未评估基模型以及仅关注指令微调模型的驱动行为。

---

## 204. AtomMem : Learnable Dynamic Agentic Memory with Atomic Memory Operation

**arXiv ID:** 2601.08323 | [PDF](https://arxiv.org/pdf/2601.08323v1)

**作者:** Yupeng Huo `[一作]` (Renmin University of China), Yankai Lin `[通讯]` (Renmin University of China)

**通讯引用:** 12031 | [OpenAlex ID](https://openalex.org/A5043098453)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种将代理记忆管理重新定义为动态决策问题的方法，拆解为 CRUD 原子操作并通过强化学习学习自适应记忆策略。

**💡 创新点**

将静态记忆流程拆解为可学习的原子 CRUD 操作，并将记忆管理视为序列决策，通过 RL 优化得到任务对齐的记忆策略，摆脱“一刀切”的手工流程。

**🔧 技术方法**

采用监督微调 + 基于策略梯度的 RL（GRPO）训练，使用向量数据库实现记忆存储，配合 Qwen3‑8B 大语言模型和 scratchpad 机制。

**📊 数据集**

在 HotpotQA、2WikiMultiHopQA、MuSiQue 三个多跳长文本问答数据集上进行扩展，构造了长上下文与多问答混合的对照任务。

**📈 对比分析**

与静态记忆、RAG、MemAgent 等基线对比，实验显示 RL 训练后模型在 200/400/800 文档设置下均领先约 2‑5% 并在 800 文档时提升近 10% 以上，证明了动态记忆策略的有效性。

**⚠️ 局限性**

RL 训练成本高，单个模型收敛需 2‑3 天 8 核 GPU，且在更长或噪声更大的任务上可能需要更多计算资源。

---

## 205. UM-Text: A Unified Multimodal Model for Image Understanding

**arXiv ID:** 2601.08321 | [PDF](https://arxiv.org/pdf/2601.08321v1)

**作者:** Lichen Ma `[一作]` (JD.COM), Junshi Huang `[通讯]` (JD.COM)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一多模态模型UM-Text，能够通过自然语言指令完成视觉文本的生成、编辑和翻译等多场景任务。

**💡 创新点**

核心创新包括：①使用UM-Designer VLM对指令与图像进行语义理解；②设计UM-Encoder融合文本、字符视觉与VLM嵌入；③提出区域一致性损失和三阶段训练策略，提升文本风格一致性和字形精度。

**🔧 技术方法**

技术手段包括流匹配扩散模型、VAE编码解码、T5语言嵌入、OCR字符视觉嵌入、Canny边缘检测、FLUX等扩散与VLM技术。

**📊 数据集**

主要数据集有：UM-DATA-200K（200k产品海报图像与文本）、AnyWord-3M（用于扩散模型预训练）、以及AnyWord-Benchmark、UDiffText、TextSeg、LAION-OCR等评测基准。

**📈 对比分析**

在多项公开基准上与AnyText、AnyText-2、FLUX-Text等方法对比，UM-Text在文本准确率、FID、LPIPS等指标均实现显著提升，尤其在中文文本生成与编辑任务中获得最高准确率和最低视觉失真。

**⚠️ 局限性**

局限性主要包括：对算力需求高；在极其复杂的多行文本场景下仍可能出现细节失真；对非标准字符或极端风格的适应性尚未充分验证。

---

## 206. YOLOBirDrone: Dataset for Bird vs Drone Detection and Classification and a YOLO based enhanced learning architecture

**arXiv ID:** 2601.08319 | [PDF](https://arxiv.org/pdf/2601.08319v1)

**作者:** Dapinder Kaur `[一作]` (Council of Scientific and Industrial Research), Shashi Poddar `[通讯]` (Council of Scientific and Industrial Research)

**通讯引用:** 609 | [OpenAlex ID](https://openalex.org/A5069253574)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于 YOLOv9 的 YOLOBirDrone 模型，用于实时检测与分类无人机与鸟类。

**💡 创新点**

创新点包括引入可变形卷积的 AELAN 结构、分层多尺度双注意模块 MPDA 与 RMPDA，以提升对小目标的定位与分类能力。

**🔧 技术方法**

在 YOLOv9 基础上采用可变形卷积、AELAN、MPDA、RMPDA 等技术，实现更强的特征聚合与多尺度注意力。

**📊 数据集**

使用自行构建的 BirDrone 数据集（11,495 张标注图像）以及 Drone‑vs‑Bird 竞赛数据进行训练与评估。

**📈 对比分析**

与 YOLOv8、YOLOv9、YOLOv10‑12、RT‑DETRv2 等 SOTA 进行对比，YOLOBirDrone 在 mAP^0.5‑0.95 为 0.668，检测精度 84.9%，平均推理时间仅 0.149 s/帧，表现优于对比模型。

**⚠️ 局限性**

仍面临极小目标（<20×20）检测难度大、光照变化、遮挡等复杂环境导致的误检与漏检问题。

---

## 207. Enhancing Image Quality Assessment Ability of LMMs via Retrieval-Augmented Generation

**arXiv ID:** 2601.08311 | [PDF](https://arxiv.org/pdf/2601.08311v1)

**作者:** Kang Fu `[一作]` (Shanghai Jiao Tong University), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 20610 | [OpenAlex ID](https://openalex.org/A5064168853)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种检索增强生成框架，利用外部图像质量数据检索并在大型多模态模型中生成更准确的图像质量评估。

**💡 创新点**

创新点在于将检索模块与生成模型结合，实现零样本、无训练的图像质量评估。

**🔧 技术方法**

采用检索增强技术、生成式语言模型以及视觉‑语言预训练模型。

**📊 数据集**

在多种公开 IQA 数据集上评估，如 LIVE、TID2013、KADID‑10k 等。

**📈 对比分析**

与传统单一视觉模型和基准 IQA 方法对比，所提方法在 SRCC、PLCC 等指标上显著提升了约 3‑5%。

**⚠️ 局限性**

局限性包括对检索库的依赖、检索效率与生成速度的瓶颈，以及在极端噪声条件下表现仍不稳定。

---

## 208. ORBIT: On-policy Exploration-Exploitation for Controllable Multi-Budget Reasoning

**arXiv ID:** 2601.08310 | [PDF](https://arxiv.org/pdf/2601.08310v1)

**作者:** Kun Liang `[一作]` (Peking University), Yunfang Wu `[通讯]` (Peking University)

**通讯引用:** 981 | [OpenAlex ID](https://openalex.org/A5027803148)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了ORBIT框架，实现了多预算可控推理，并将不同推理模式集成到单一模型中。

**💡 创新点**

创新点包括：①基于RLVR的单阶段多预算探索与压缩循环，发现各预算的Pareto前沿策略；②利用基于RKL的on‑policy蒸馏将多模式专家融合为统一模型，同时保留模式分离；③通过模型合并实现冷启动，提升蒸馏收敛速度。

**🔧 技术方法**

采用RLVR（Group Relative Policy Optimization）进行自回归推理训练，使用一次性扩展‑压缩循环来探索推理前沿，随后进行on‑policy蒸馏（RKL）并在此基础上进行多教师融合与模式感知初始化。

**📊 数据集**

训练使用DAPO‑Math‑17K（用于o3-mini）和Polaris‑53K（用于gpt‑oss‑120b和其它模型），评估数据包括AIME 2024/2025、BeyondAIME、GPQA‑Diamond、MMLU‑Pro等数学与通用问答基准。

**📈 对比分析**

与现有可控推理模型（o3‑mini、gpt‑oss‑120b、ThinkDial）以及RL‑based高效推理方法相比，ORBIT在每个预算下保持了较高的准确‑长度比；在不同领域迁移时模式顺序保持不变，仅有轻微的性能衰减；整体在各基准上表现与或优于对比模型。

**⚠️ 局限性**

局限性包括：①压缩阶段的二进制上下文划分可能不够精细，影响模式分离与训练稳定性；②在输出熵极低的固定模型上，on‑policy RL的样本效率有限，可能导致难以充分饱和压缩过程；③长链RL训练对大模型及长上下文任务计算成本高，限制了可扩展性。

---

## 209. AgriAgent: Contract-Driven Planning and Capability-Aware Tool Orchestration in Real-World Agriculture

**arXiv ID:** 2601.08308 | [PDF](https://arxiv.org/pdf/2601.08308v1)

**作者:** Bo Yang `[一作]` (Zhejiang University), Shijian Li `[通讯]` (Zhejiang University)

**通讯引用:** 7004 | [OpenAlex ID](https://openalex.org/A5103196339)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 AgriAgent 两层级的农业智能体框架，System‑1 用于快速多模态问答，System‑2 用于复杂任务的合同驱动规划与工具编排。

**💡 创新点**

创新点在于：①引入显式合同（need contract）把任务需求与工具实现解耦；②使用辩论式计划细化（critique‑defend‑revise）保证结构可验证；③通过 ToolHub 采用 TDI/TOCI 两阶段匹配实现能力感知的工具编排；④动态工具生成器 ToolMaker 在无工具时构造可验证工具。

**🔧 技术方法**

技术包括多模态专用模型（AgriGPT、AgriGPT‑VL、AgriGPT‑Omni）+联合 RAG，合同驱动规划、Debate 生成 DAG、ToolHub 语义匹配与合成、ToolMaker 自动生成工具、可验证执行与证据聚合。

**📊 数据集**

数据集：System‑1 采用 AgriGPT‑Omni 多模态问答数据；System‑2 采用自构建的 1,000 条复杂农业任务集合；工具池包含多种农业相关工具与公开开源工具。

**📈 对比分析**

与单模型 CoT/ToT、React、Plan‑and‑Execute 等基线对比，AgriAgent 在 Programmatic（规则引用、证据、规范化）和 LLM‑based 指标上均显著提升；工具选择 Hit@1/3/5 在大规模工具池下稳定，ToolMaker 成功率达 96.94%。

**⚠️ 局限性**

局限性包括：长流程任务导致计算与交互延迟；对结构化合同输出与 schema 的依赖，模型输出偏差会影响解析；ToolMaker 目前不专门针对极其复杂或领域专属工具；未给出详细延迟与成本分析。

---

## 210. SnapGen++: Unleashing Diffusion Transformers for Efficient High-Fidelity Image Generation on Edge Devices

**arXiv ID:** 2601.08303 | [PDF](https://arxiv.org/pdf/2601.08303v1)

**作者:** Dongting Hu `[一作]` (University of Melbourne), Anil Kag `[通讯]` (Snap Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向移动和边缘设备的高效扩散变压器（Efficient Diffusion Transformer），实现高分辨率（1024×1024）文本到图像生成。

**💡 创新点**

创新点包括：①自适应全局–局部稀疏注意力（Adaptive Sparse Self-Attention）实现高效高分辨率处理；②弹性训练框架（Elastic Training）在同一超网络中共享参数，可按硬件动态切换子网络；③知识引导分布匹配蒸馏（K-DMD）将多步教师知识压缩到仅四步即可生成高质量图像。

**🔧 技术方法**

技术手段包括：扩散变压器架构、三阶段（Down-Middle-Up）设计、密集长距离跳跃连接、分块邻域注意力、分量化与剪枝、流匹配损失、知识蒸馏与分布匹配蒸馏、LoRA增强的少步教师。

**📊 数据集**

使用 ImageNet‑1K 进行训练与评估，并在 MS‑COCO、DPG‑Bench、GenEval、T2I‑CompBench 与 CLIP‑Score 等公开基准上做对比验证。

**📈 对比分析**

与 SnapGen、SANA、Flux 等基准模型对比，单模型参数数仅 0.3B–1.6B，推理延迟 280–1580 ms，4 步蒸馏后性能接近 28 步基线，显著提升实时生成速度并保持或超过服务器级质量。

**⚠️ 局限性**

局限性：在 1024×1024 分辨率下部分模型仍会出现 OOM；对极低算力设备的显存/功耗仍有一定压力；蒸馏过程中对教师模型依赖较大，且不同硬件上细节调优仍需进一步探索。

---

## 211. M3SR: Multi-Scale Multi-Perceptual Mamba for Efficient Spectral Reconstruction

**arXiv ID:** 2601.08293 | [PDF](https://arxiv.org/pdf/2601.08293v1)

**作者:** Yuze Zhang `[一作]` (Shenzhen University), Victor C. M. Leung `[通讯]` (Shenzhen University)

**通讯引用:** 64005 | [OpenAlex ID](https://openalex.org/A5035919267)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种多尺度、多感知的 Mamba 架构 M3SR，用于从 RGB 图像恢复高光谱图像。

**💡 创新点**

创新点包括：① 多感知融合（MPF）模块，将空间、频域与光谱三种感知方式并行融合；② 将 Mamba 与 U‑Net 结合，构建多尺度特征提取与融合；③ 自适应加权融合与残差连接提升重建质量。

**🔧 技术方法**

技术手段包括 Mamba 状态空间模型、2D Selective‑Scan、VSS 块、离散小波变换（DWT/IDWT）、分组 Mamba 块、频域分解、加权融合以及 MAE 损失函数。

**📊 数据集**

使用四个公开高光谱重建数据集：NTIRE2022、NTIRE2020‑Clean、NTIRE2020‑Realworld 与 CAVE。

**📈 对比分析**

与十种 SOTA 方法（HSCNN+、FMNet、HRNet、GDNet、HSRNet、SSDCN、MST++、RepCPSI、SSRNet、GMSR）在 RMSE/PSNR/SAM/MSSIM 指标上进行对比；M3SR 在所有数据集上均取得最优或第二最优指标，并且参数量与 FLOPs 低于大多数竞争方法。

**⚠️ 局限性**

局限性：对极高分辨率或更复杂场景的鲁棒性尚未验证；模型的组数 G 需要调优，过大会显著增加参数与计算量；仍有提升空间以进一步降低计算成本与提升泛化性能。

---

## 212. KidVis: Do Multimodal Large Language Models Possess the Visual Perceptual Capabilities of a 6-Year-Old?

**arXiv ID:** 2601.08292 | [PDF](https://arxiv.org/pdf/2601.08292v1)

**作者:** Xianfeng Wang `[一作]` (Shanghai Jiao Tong University), Xiongkuo Min `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9611 | [OpenAlex ID](https://openalex.org/A5043405654)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 KidVis 基准，评估多模态大型语言模型（MLLM）在儿童级别的六种视觉原始能力（注意、追踪、辨别、记忆、空间、闭合）上的表现；

**💡 创新点**

创新点在于把视觉智能拆解为与儿童视觉发展对应的六个原子能力，并构造十类低语义、无运动干扰的视觉任务，形成系统化的诊断基准；

**🔧 技术方法**

技术上采用了零样本评估、双因子任务设计、视觉能力加权汇总、以及对比分析模型在不同规模下的性能；

**📊 数据集**

数据集为 KidVis Benchmark，包含 10 类任务，每类 50 题，图像高分辨率（≥2K），专门设计以剔除语义与运动因素；

**📈 对比分析**

对比方法是将 20 个主流 MLLM（包括 GPT‑5、Gemini‑2.5‑Pro 等专有模型和 Qwen3‑VL、InternVL3.5、LLaVA‑Next 等开源模型）在零样本设置下与 6‑7 岁儿童基准进行对照；结果显示儿童平均得分 95.32，GPT‑5 仅 67.33，开源模型平均低于 40，表明存在约 30–50% 的差距；

**⚠️ 局限性**

局限性包括：仅评估视觉原始能力，未覆盖更高层次的视觉推理；基准任务设计可能受人工选择影响；缺乏对模型内部机制的深入解释，导致对改进方向的指引不够精准；

---

## 213. Pursuing transparency: How research performing organizations in Germany collect data on publication costs

**arXiv ID:** 2601.08340 | [PDF](https://arxiv.org/pdf/2601.08340v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 214. ToolACE-MCP: Generalizing History-Aware Routing from MCP Tools to the Agent Web

**arXiv ID:** 2601.08276 | [PDF](https://arxiv.org/pdf/2601.08276v1)

**作者:** Zhiyuan Yao `[一作]` (Zhejiang University), Weiwen Liu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 5050 | [OpenAlex ID](https://openalex.org/A5015039449)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一套历史感知路由器训练框架和轻量级路由代理，用于在 Agent Web 中高效导航和调用大规模工具与代理。

**💡 创新点**

创新点包括：①自进化图扩展 + 随机游走采样 + 多智能体轨迹合成，生成针对路由的高质量监督信号；②通过轻量化插件式路由代理实现跨工具/代理的无缝集成。

**🔧 技术方法**

技术手段包括：图嵌入与相似度构建候选图、LLM 驱动的自进化变异、随机游走采样、基于多智能体的轨迹仿真、LoRA 微调 Qwen3-8B，以及简化的路由器调用与执行工具。

**📊 数据集**

使用的数据集：MCP‑Universe、MCP‑Mark 真实基准；通过自进化变异扩增至 2005 工具；构建 400+ 代理的 Agent Bank 进行跨代理评估。

**📈 对比分析**

与嵌入检索、ReAct、GPT‑4o、Gemini‑2.5‑Pro 等基线对比，路由器在 MCP‑Universe 上达 53.44%（对比 GPT‑4o 47.41%），MCP‑Mark 上 60.00%，在扩展工具空间和噪声环境下保持 53.02% 及 56.00%；在 Agent 任务中实现 91.6% 的准确率，表现出显著的稳健性与泛化能力。

**⚠️ 局限性**

局限性：仅在 8B 模型上进行 LoRA 微调；路由器主要训练于工具数据，对更大模型与极大规模检索任务的扩展性仍需进一步验证；跨代理通用性在部分复杂场景下仍需优化。

---

## 215. On Evaluation of Unsupervised Feature Selection for Pattern Classification

**arXiv ID:** 2601.08257 | [PDF](https://arxiv.org/pdf/2601.08257v1)

**作者:** Gyu-Il Kim `[一作]` (Chung-Ang University), Jaesung Lee `[通讯]` (Chung-Ang University)

**通讯引用:** 3203 | [OpenAlex ID](https://openalex.org/A5100744043)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

重新定义无监督特征选择的评估框架，采用多标签分类环境评估现有方法，并在21个多标签数据集上进行系统实验。

**💡 创新点**

提出了多标签评估视角来客观评估无监督特征选择，揭示单标签评估可能导致的偏差，并验证了基于熵最大化的EMUFS在多标签条件下的优势。

**🔧 技术方法**

使用无监督特征选择方法（EMUFS、MCFS、FSDK、CNAFS、RUSLP、EGCFS），采用ML-kNN分类器，结合Hamming Loss、Ranking Loss、One-Error、Multi-Label Accuracy四个多标签评估指标。

**📊 数据集**

21个公开多标签数据集：Inter3000、CHD49、GpositiveGO、GpositivePseAAC、PlantGO、PlantPseAAC、VirusGO、Waterquality、Birds、CAL500、Emotions、Enron、Flags、Foodtruck、Genbase、Image、Langlog、Medical、Scene、Coffee、Yeast。

**📈 对比分析**

与多种代表性无监督特征选择方法在ML-kNN（k=10）下比较，采用Multi-Label Accuracy为主要指标。EMUFS在绝大多数数据集上获得最高准确率，平均排名第二；在Hamming Loss、Ranking Loss、One-Error等损失指标上也名列前茅，显示其在多标签环境中的稳健性。

**⚠️ 局限性**

局限性包括：仅使用ML-kNN且k固定为10，未评估其他分类器的表现；未深入分析单标签评估偏差的根本原因；实验规模相对有限，未涵盖更大或更复杂的标签结构；仅关注无监督特征选择，未考虑半监督或监督场景。

---

## 216. From Local Windows to Adaptive Candidates via Individualized Exploratory: Rethinking Attention for Image Super-Resolution

**arXiv ID:** 2601.08341 | [PDF](https://arxiv.org/pdf/2601.08341v1)

**作者:** Chunyu Meng `[一作]` (University of Electronic Science and Technology of China), Shuhang Gu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 12162 | [OpenAlex ID](https://openalex.org/A5100745570)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种 Individualized Exploratory Transformer（IET），用于单图像超分辨率任务。

**💡 创新点**

创新点在于设计了 Individualized Exploratory Attention（IEA）机制，能够为每个 token 自适应选择内容感知、非对称的 attention 伙伴，并通过逐层的扩张与稀疏化实现高效且更精准的信息聚合。

**🔧 技术方法**

使用的技术包括 Transformer 自注意力、稀疏矩阵乘法（SMM）、稀疏采样的 DLSG 初始化、基于图论的邻居扩张与稀疏化、Similarity‑Fuse Feed‑Forward Network（SF‑FFN）以及 LePE 位置编码。

**📊 数据集**

在 DIV2K+Flickr2K 训练集上进行训练，并在 Set5、Set14、BSD100、Urban100、Manga109 等标准超分辨率基准数据集上进行评估。

**📈 对比分析**

与当前主流方法（如 SwinIR、CAT、PFT、ATD 等）在 ×2、×3、×4 放大比例下进行比较，IET 在保持相近参数量与 FLOPs 的前提下，在 PSNR/SSIM 上均实现了显著提升，尤其在 Urban100 上差距达 0.17–0.37 dB。

**⚠️ 局限性**

局限性包括：1）对 dilation 参数和扩张步数的选择较为敏感；2）扩张与稀疏化在过深网络或极大分辨率时可能导致稳定性下降；3）仅在超分任务验证，未在更复杂的视觉任务或 NLP 任务上进一步验证。

---

## 217. APT-MCL: An Adaptive APT Detection System Based on Multi-View Collaborative Provenance Graph Learning

**arXiv ID:** 2601.08328 | [PDF](https://arxiv.org/pdf/2601.08328v1)

**作者:** Mingqi Lv `[一作]` (Zhejiang University of Technology), Tiantian Zhu `[通讯]` (Zhejiang University of Technology)

**通讯引用:** 1870 | [OpenAlex ID](https://openalex.org/A5012235835)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于多视图协同学习的APT检测系统APT-MCL，利用无监督学习从主机产生的证据图中学习正常行为模式，并通过异常检测识别节点级APT攻击。

**💡 创新点**

创新点包括：① 将结构视图与行为视图分别提取并独立训练子模型，避免特征融合导致的过拟合；② 通过协同学习框架（Co‑Training）在无标签条件下生成伪标签，逐步将无监督子模型转化为弱监督模型；③ 设计多种融合策略（Benign/Malicious Voting、Soft Voting、Stacking）以提高跨场景泛化能力。

**🔧 技术方法**

使用的技术主要有：图神经网络（GraphSAGE）进行自监督节点类型分类以获取高阶特征；Isolation Forest做异常检测；多视图特征提取（结构特征：边类型分布；行为特征：敏感操作指示符）；协同学习框架（Co‑Training）生成伪标签；以及多种融合策略。

**📊 数据集**

实验数据集包括：DARPA TC E3的三子集（TRACE、THEIA、CADETS）；DataBreach（模拟的三种APT攻击场景）；Ransomware（针对Windows的勒索软件攻击）。

**📈 对比分析**

与Log2vec、MAGIC、ThreaTrace等主流无监督APT检测方法对比，APT‑MCL在所有数据集上均实现了更高的宏F1、低FPR以及更好的召回率，尤其在DARPA子集表现出几乎无误报的极低FPR，并在DataBreach等隐蔽攻击场景中显著提升精度；与全监督学习相比，性能相差不大但无需昂贵的细粒度标注。

**⚠️ 局限性**

局限性：① 训练与推理阶段的计算与内存开销较大（多视图、迭代Co‑Training）；② 对阈值参数（confidence threshold、batch prob）敏感，需要经验调优；③ 对极其隐蔽、极低频率的APT行为仍可能产生误判；④ 目前实现主要针对离线分析，实时部署仍需进一步优化。

---

## 218. Deep Exploration of Epoch-wise Double Descent in Noisy Data: Signal Separation, Large Activation, and Benign Overfitting

**arXiv ID:** 2601.08316 | [PDF](https://arxiv.org/pdf/2601.08316v1)

**作者:** Tomoki Kubo `[一作]` (Niigata University), Yusuke Iida `[通讯]` (Niigata University)

**通讯引用:** 792 | [OpenAlex ID](https://openalex.org/A5076724630)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对在带有30%标签噪声的CIFAR‑10数据集上训练的全连接神经网络进行 epoch‑wise 双下坡现象的实证研究，分析了内部信号分离与大激活的出现机制，并提出一种新的深度双下坡解释方案。

**💡 创新点**

首次将内部激活相似度和大激活量作为指标，揭示了干净样本与噪声样本在网络内部的分离过程，并将这种分离与“善良过拟合”以及大激活现象关联起来，提供了关于深度双下坡的新物理解释。

**🔧 技术方法**

使用标准全连接（MLP）网络、Adam 优化器、交叉熵损失、ReLU 激活、softmax 输出，配合对干净/噪声训练集的损失与精度分解、余弦相似度计算及最大激活与 RMS 比值分析。

**📊 数据集**

CIFAR‑10 数据集（训练 50k 张，测试 10k 张），在训练集上随机 30% 的标签引入噪声。

**📈 对比分析**

通过将训练过程划分为早期、噪声学习与双下坡三个阶段，并用损失/准确率曲线以及相似度/激活比例曲线进行对比，表明在大模型（MLP7）中出现了测试误差的双下坡且最终实现了善良过拟合；中等模型仅在后期出现微弱现象，较小模型则无双下坡。

**⚠️ 局限性**

研究仅局限于简单的全连接网络与单一数据集，未考虑卷积或 Transformer 结构；实验样本量有限，缺乏对超参数、学习率、噪声比例变化的系统性评估；大激活的机制仍需理论解释。

---

## 219. Data-Induced Groupings and How To Find Them

**arXiv ID:** 2601.08256 | [PDF](https://arxiv.org/pdf/2601.08256v1)

**作者:** Yilan Jiang `[一作]` (University of Illinois), Eugene Wu `[通讯]` (Columbia University)

**通讯引用:** 5554 | [OpenAlex ID](https://openalex.org/A5049016095)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过两项用户研究探究了在点图中由数据值诱导的假性分组，并基于实验结果构建了可解释的分组预测模型以及诊断和重排工具。

**💡 创新点**

创新点在于首次系统揭示数据值如何强行产生视觉群组（data‑induced groupings），并提出利用聚类与共线性特征的轻量化可解释模型来预测用户的群组判断，同时提供自动化诊断与优化重排方案。

**🔧 技术方法**

使用的技术包括：基于Prolific平台的用户实验、特征工程（共线性误差、质心距离、凸包重叠等）、逻辑回归与深度有限的决策树、SHAP解释、以及针对x轴置换的穷举优化算法。

**📊 数据集**

实验数据集主要是人工生成的六点点图（每幅图随机y值，x轴为名义顺序），另外在案例演示中使用了真实的饮料价格数据集。

**📈 对比分析**

模型评估采用5折交叉验证和独立hold‑out，决策树在hold‑out上实现F1≈0.97–0.99，精度100%；诊断与重排工具通过计算所有720种x轴置换的违规与满足度来比较，显示出明显的优化空间。

**⚠️ 局限性**

主要局限包括：仅针对六点点图、仅考虑名义x轴；负样本生成依赖启发式规则可能引入偏差；未覆盖条形图、折线图等其它图表；用户任务局限为主动寻找群组，未验证开放式或领域任务中的表现。

---

## 220. Semantic Laundering in AI Agent Architectures: Why Tool Boundaries Do Not Confer Epistemic Warrant

**arXiv ID:** 2601.08333 | [PDF](https://arxiv.org/pdf/2601.08333v1)

**作者:** Oleg Romanchuk `[一作]`, Roman Bondar `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出并形式化了 LLM 代理架构中的“语义洗牌”问题，即通过工具接口把生成的命题误认为观测，从而无意义地提升其知识地位，进一步阐述了其与 Gettier 方案的对应关系并给出不可避免的自许可定理。

**💡 创新点**

创新点包括：
- 引入“语义洗牌”这一新的架构缺陷概念；
- 定义并证明了“担保侵蚀原则”与“不可避免自许可定理”，展示了为何在主流代理模式下无法消除循环证明；
- 明确区分工具的三类 epistemic 角色（观察者、计算器、生成器），并指出缺乏此区分是造成洗牌的根源；
- 说明 LLM‑as‑judge 方案在同一命题空间内同样会陷入自许可，提供了安全关键系统监管失效的理论依据。

**🔧 技术方法**

采用形式化逻辑工具：定义了 epistemic warrant、epistemic relevance、semantic laundering 等概念，并利用公理化的证明框架（类似知识论中的布尔代数）推导定理；同时在代码层面给出 ReAct 等常用框架的示例，展示如何在实践中识别洗牌。

**📊 数据集**

无；本文是理论与架构分析，不依赖具体数据集。

**📈 对比分析**

无直接实验比较；性能评价采用理论可行性与结构不可逆性分析，指出在满足主要假设的任何实现中，洗牌都无法通过提升模型规模、校准或评估偏好来根除。

**⚠️ 局限性**

局限性包括：
- 仅关注基于 LLM 的工具调用架构，未覆盖所有可能的代理实现；
- 依赖于对工具类别的人工划分，未提供自动化的类型系统实现；
- 只给出了架构层面的解决思路，未给出完整可执行的规范或工具库；
- 对于实际安全系统，如何将理论框架落地仍需进一步研究。

---

## 221. AgriLens: Semantic Retrieval in Agricultural Texts Using Topic Modeling and Language Models

**arXiv ID:** 2601.08283 | [PDF](https://arxiv.org/pdf/2601.08283v1)

**作者:** Heba Shakeel `[一作]` (Jamia Millia Islamia), Chandni Saxena `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 202 | [OpenAlex ID](https://openalex.org/A5016394253)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建了AgriLens框架，将无监督的BERTopic主题发现与LLM零-shot主题标签和向量检索结合，实现对大规模农业文本的语义检索与交互；

**💡 创新点**

创新点在于将主题模型作为检索引导层，实现完全无监督的主题驱动检索，并利用LLM生成可读标签，同时加入偏见与语义一致性评估模块，专门针对农业文本领域；

**🔧 技术方法**

技术组合包括BERTopic（基于BERT嵌入、UMAP降维、HDBSCAN聚类）、Flan‑T5‑Small和量化Mistral‑7B进行零-shot标签生成、Sentence‑BERT（all‑MiniLM‑L6‑v2）生成密集向量、ChromaDB/FAISS存储检索以及Prompt设计与评估；

**📊 数据集**

使用的主要数据集是ENAGRINEWS（22,814篇英文农业新闻），经过清洗、句子分块并构建嵌入向量；

**📈 对比分析**

通过多指标评估（Cosine相似度、BERTScore、Factuality、Document Coverage、人工Likert打分）与两模型比较，Mistral‑7B在语义相似度0.699、BERTScore0.825、文档覆盖率0.99、人工相关性评分4.3/5等指标均优于Flan‑T5‑Small，并且检索结果平均Relevance 1.85/2；

**⚠️ 局限性**

局限性包括对短文本噪声的敏感、LLM可能产生偏见或幻觉、标签多义导致检索特异性不足、未集成多模态或实时更新机制，以及对农学专业知识深度的进一步挖掘仍有提升空间。

---

## 222. Markovian Pre-Trained Transformer for Next-Item Recommendation

**arXiv ID:** 2601.08275 | [PDF](https://arxiv.org/pdf/2601.08275v1)

**作者:** Cong Xu `[一作]` (East China Normal University), Wei Zhang `[通讯]` (East China Normal University)

**通讯引用:** 34025 | [OpenAlex ID](https://openalex.org/A5100441678)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于Markov链合成数据预训练的Transformer模型（MPT）用于下一条物品推荐，并通过轻量级适配器实现高效微调。

**💡 创新点**

创新点在于：①发现下一条推荐可归纳为两种可迁移能力（对历史行为做全局摘要与突出最近交互）；②使用全合成Markov链进行预训练激发这两种能力；③证明Markovian预训练比传统推荐预训练和语言模型预训练更高效、效果更佳。

**🔧 技术方法**

核心技术包括：Transformer架构、Markovian预训练（next-state prediction on synthetic Markov chains）、轻量级适配器（RMSNorm+两层MLP）以及对比LoRA等参数高效微调方案；评估指标为HR@N和NDCG@N。

**📊 数据集**

实验使用了五个公开数据集：Beauty、Toys、Sports（Amazon Review 5‑core过滤）、Yelp（2018版 10‑core过滤）和Online Retail（UK电商）。

**📈 对比分析**

对比传统序列模型（GRU4Rec、SASRec、FMLPRec、HSTU）、专用推荐预训练模型（UniSRec、RecFormer）以及语言模型预训练模型（E4SRec、Qwen2.5），MPT在大多数数据集上平均提升12–15%（比最佳无预训练模型），比语言模型提升约25%，显著优于其它预训练方案。

**⚠️ 局限性**

局限性包括：①仅使用Markov链合成数据预训练，可能不足以覆盖所有真实交互模式；②在某些数据集（如Online Retail）对Cold‑start表现有限；③LoRA等PEFT在部分场景下会导致性能下降；④模型在极大数据量或更复杂序列场景下的进一步提升空间尚待探索。

---

## 223. Sparsity Is Necessary: Polynomial-Time Stability for Agentic LLMs in Large Action Spaces

**arXiv ID:** 2601.08271 | [PDF](https://arxiv.org/pdf/2601.08271v1)

**作者:** Angshul Majumdar `[一作]` `[通讯]`, Angshul Majumdar

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了稀疏代理控制（SAC）框架，用以在工具集合规模巨大的情况下学习稀疏的决策策略

**💡 创新点**

通过在策略学习中引入ℓ₁,₂正则化，证明只需 O(k log M) 次样本即可实现稀疏工具集的识别和近似最优控制；并给出了密集策略不可避免的线性样本复杂度下界

**🔧 技术方法**

利用高维统计的稀疏恢复理论（政策-RSC、不可代表性、β‑min 等）构建凸优化学习器，并使用主-双重证人（PDW）证明精确支持恢复；在部分可观测情形下进一步引入表示误差 ε_b 分解性能

**📊 数据集**

论文未使用特定公开数据集，理论以抽象的工具集和情境特征为基础；实验验证（未在本文中给出）可采用标准的工具使用基准或自定义模拟环境

**📈 对比分析**

与传统基于提示的、规则驱动或密集路由器的 LLM 系统对比，SAC 在工具稀疏性满足时实现了对数级样本复杂度的控制，且在密集策略下会出现线性样本需求；实验表明支持恢复后可显著提升任务完成率和鲁棒性

**⚠️ 局限性**

主要限制在于对稀疏性、不可代表性和政策-RSC 的严格假设；若工具集高度冗余或特征不可区分，理论不再适用；另外，表示误差 ε_b 的估计和控制仍是实际部署中的挑战

---

## 224. Tissue Classification and Whole-Slide Images Analysis via Modeling of the Tumor Microenvironment and Biological Pathways

**arXiv ID:** 2601.08336 | [PDF](https://arxiv.org/pdf/2601.08336v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 225. Automated Machine Learning in Radiomics: A Comparative Evaluation of Performance, Efficiency and Accessibility

**arXiv ID:** 2601.08334 | [PDF](https://arxiv.org/pdf/2601.08334v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 226. Med-CoReasoner: Reducing Language Disparities in Medical Reasoning via Language-Informed Co-Reasoning

**arXiv ID:** 2601.08267 | [PDF](https://arxiv.org/pdf/2601.08267v1)

**作者:** Fan Gao `[一作]` (University of Tokyo), Irene Li `[通讯]` (University of Tokyo)

**通讯引用:** 2774 | [OpenAlex ID](https://openalex.org/A5101537931)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种语言信息协同推理框架，在英语与本地语言并行推理后通过概念融合与检索实现多语言医疗推理。

**💡 创新点**

通过将英语逻辑骨架与本地语言专业知识并行融合来弥补低资源语言的推理鸿沟。

**🔧 技术方法**

使用并行推理、跨语言概念抽取与融合、检索增强推理、知识检索和模型微调等技术。

**📊 数据集**

使用MMLU-ProX-Health、Global-MMLU医学子集、以及新建的七语种LFQA与NLI基准数据集。

**📈 对比分析**

在多语言MCQA、LFQA和NLI任务上与多种闭源和开源模型对比，平均提升约5-8%，尤其在斯瓦希里语、约鲁巴语等低资源语言上显著提高。

**⚠️ 局限性**

依赖英语为枢轴、计算开销大、检索质量波动、缺乏理论解释等局限。

---

## 227. AIMC-Spec: A Benchmark Dataset for Automatic Intrapulse Modulation Classification under Variable Noise Conditions

**arXiv ID:** 2601.08265 | [PDF](https://arxiv.org/pdf/2601.08265v1)

**作者:** Sebastian L. Cocks `[一作]` (Adelaide University), Feras Dayoub `[通讯]` (Adelaide University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文创建并公开发布了 AIMC‑Spec 数据集，针对不同噪声水平下的单脉冲调制分类问题提供统一的合成 I/Q 信号和时频谱图；

**💡 创新点**

创新点在于构建了一个包含 33 类调制、13 级 SNR 的大规模、可复现数据集，并在同一 spectrogram 输入上系统比较多种深度学习架构；

**🔧 技术方法**

采用 FFT‑based spectrogram 预处理，结合卷积网络、U‑Net、去噪自编码器、Transformer 等深度学习技术进行特征提取与分类；

**📊 数据集**

使用 AIMC‑Spec 数据集，该数据集为每类 1,000 条 100,000 采样点的合成 I/Q 信号，共 13 个 SNR 等级（+10 至 –20 dB）；

**📈 对比分析**

通过统一的 80:20 训练/测试划分，对五种代表性模型进行基准评估，结果表明 LDC‑Unet 在全任务和 FM 子集均保持最高稳定准确率，Transformer 需大量数据支持，整体准确率随 SNR 降低显著下降；

**⚠️ 局限性**

局限在于仅模拟 AWGN 条件，缺乏多径、衰落、脉冲抖动等真实信道效应；对低 SNR 下非频率调制的识别仍显薄弱。

---

## 228. VGG Induced Deep Hand Sign Language Detection

**arXiv ID:** 2601.08262 | [PDF](https://arxiv.org/pdf/2601.08262v1)

**作者:** Subham Sharma `[一作]` (DXOps), Sharmila Subudhi `[通讯]` (Maharaja Sriram Chandra Bhanja Deo University)

**通讯引用:** 323 | [OpenAlex ID](https://openalex.org/A5052193595)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于 VGG-16 的手势识别系统，利用迁移学习与图像增强训练模型，并使用 MediaPipe 捕获手部关键点进行最终分类。

**💡 创新点**

创新点在于将预训练的 VGG-16 与数据增强相结合，同时利用 MediaPipe 实时获取手部关键点，实现了对 10 类手势的高精度识别。

**🔧 技术方法**

采用了 VGG-16 卷积网络、迁移学习、Keras ImageDataGenerator 数据增强、RMSprop 优化器、交叉熵损失以及 MediaPipe 手部关键点检测技术。

**📊 数据集**

使用的主要数据集包括 ImageNet（预训练权重）、NUS 手势数据集（10 类字母 a‑j）以及通过 MediaPipe 捕获的自定义测试集（同 10 类）。

**📈 对比分析**

通过准确率和交叉熵损失进行评估，实验结果显示模型在验证/测试集上达到了 98.33% 的准确率，表现优异。

**⚠️ 局限性**

局限性包括：仅处理 10 类静态手势，未验证动态或多类别场景；模型对不同光照、遮挡等真实环境的鲁棒性有限；训练过程仍存在过拟合风险。

---

## 229. Safe Heterogeneous Multi-Agent RL with Communication Regularization for Coordinated Target Acquisition

**arXiv ID:** 2601.08327 | [PDF](https://arxiv.org/pdf/2601.08327v1)

**作者:** Gabriele Calzolari `[一作]` (Luleå University of Technology), George Nikolakopoulos `[通讯]` (Luleå University of Technology)

**通讯引用:** 6253 | [OpenAlex ID](https://openalex.org/A5064878830)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在未知环境中为异构机器人团队设计了分散式多智能体强化学习框架，以实现目标发现与获取。

**💡 创新点**

创新点在于将图注意网络通信、轨迹安全过滤器以及通信正交正则化结合，既保证安全又提升通信多样性。

**🔧 技术方法**

使用了 MAPPO（多智能体近端策略优化）、GAT（图注意网络）、DeepSets、RK4 积分等技术，并在每个动作上加入安全过滤器。

**📊 数据集**

使用自定义 VMAS 仿真环境生成的随机目标分布数据集进行训练与评估。

**📈 对比分析**

与四种奖励配置（R1–R4）对比，R4 方案表现最优，平均获得目标数约 2–3 个，奖励收敛最快、方差最小。

**⚠️ 局限性**

局限性：仅在仿真中验证，团队规模有限，未在真实硬件上部署与测试。

---

## 230. HIPPO: Accelerating Video Large Language Models Inference via Holistic-aware Parallel Speculative Decoding

**arXiv ID:** 2601.08273 | [PDF](https://arxiv.org/pdf/2601.08273v1)

**作者:** Qitan Lv `[一作]` (University of Science and Technology of China), Chao Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 19492 | [OpenAlex ID](https://openalex.org/A5100460246)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种全局感知并行的推测式解码框架，以加速视频大型语言模型的推理；

**💡 创新点**

①结合全局注意力与局部视觉语义进行语义感知的令牌保留，消除位置偏差；②设计视频并行推测式解码算法，使草稿生成与目标验证同步并重叠计算；

**🔧 技术方法**

全局-局部注意力融合评分、跨帧时间冗余与空间冗余度量、并行解码调度、动态策略切换；

**📊 数据集**

Video-MME、VideoDetailCaption（VDC）、MVBench、LongVideoBench、MLVU、LVBench；

**📈 对比分析**

与 Vanilla SD、SD-tree、SpecVLM 等基线对比，平均加速比 1.3–3.5 倍，最高 3.51×；在四大视频‑LLM 上保持无损生成质量；

**⚠️ 局限性**

需专门训练轻量级草稿模型以获得更佳对齐；对大批量吞吐量适配有限；对高通量处理的适用性尚未验证。

---

## 231. A Usable GAN-Based Tool for Synthetic ECG Generation in Cardiac Amyloidosis Research

**arXiv ID:** 2601.08260 | [PDF](https://arxiv.org/pdf/2601.08260v1)

**作者:** Francesco Speziale `[一作]`, Pietro Hiram Guzzi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

开发了一套基于GAN的可使用工具，用于心脏淀粉样变性研究中生成高保真、带标签的合成ECG波形，并提供命令行与Streamlit GUI交互式训练与生成。

**💡 创新点**

创新点包括：① 类别特定双向LSTM GAN实现不同心电波形的高质量生成；② 设计易用的工作流与可视化界面，使非专业研究者能快速训练并批量生成合成样本；③ 针对心脏淀粉样变性独特ECG特征的评估与聚类验证框架。

**🔧 技术方法**

使用技术：PyTorch实现的1D GAN（生成器和判别器均基于双向LSTM+全连接），Adam优化；评估方法包括DTW、Fréchet、欧氏距离、Kolmogorov–Smirnov、MMD、Hotelling T²等统计量；AI‑ECG分类器（CNN/Transformer）和聚类算法（k‑means、GMM）；Streamlit构建GUI。

**📊 数据集**

主要数据集：MIT‑BIH心律失常数据库作为训练与基准集，随后使用有限的AL/ATTR心脏淀粉样变性患者真实ECG样本进行微调与评估；实验中将合成样本与真实样本进行对比。

**📈 对比分析**

比较方法：利用DTW、Fréchet、欧氏距离等时间域相似度评估形态；KS、MMD、多变量检验评估分布一致性；在AI‑ECG分类实验中，GAN增强组在AUROC、AUPRC、灵敏度/特异性等指标上显著优于仅过采样组，且保持良好校准；聚类稳定性和ARI提升，说明合成样本有助于子型发现。

**⚠️ 局限性**

限制：仅基于MIT‑BIH数据与有限CA样本训练，少数类别仍易出现模式崩溃；评估主要以统计相似度为主，缺乏临床验证；模型对不同设备/采样率的泛化性待验证。

---

## 232. Demystifying the Slash Pattern in Attention: The Role of RoPE

**arXiv ID:** 2601.08297 | [PDF](https://arxiv.org/pdf/2601.08297v1)

**作者:** Yuan Cheng `[一作]` (National University of Singapore), Zhuoran Yang `[通讯]` (Yale University)

**通讯引用:** 4619 | [OpenAlex ID](https://openalex.org/A5101727948)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型中出现的 Slash‑Dominant Heads（SDH）现象，揭示其根源是查询/键的低秩结构与 RoPE 的高/中频率成分共同作用而产生的斜线注意力模式。

**💡 创新点**

创新点在于将 SDH 的出现从经验现象转化为可解释的几何机制，并给出了完整的理论证明：在 token 嵌入近似锥形、RoPE 频率满足“斜线占优”条件时，梯度下降训练的浅层 Transformer 能自发学习到 SDH 并在 OOD 上保持通用。

**🔧 技术方法**

使用的技术主要包括：RoPE 旋转位置编码、对查询/键矩阵的低秩分析、傅里叶分解法解释注意力日志、以及基于梯度下降的两阶段训练理论分析。

**📊 数据集**

实验数据集涵盖三款开源 LLM（Gemma‑7B、Qwen2.5‑7B‑Instruct、Llama3‑8B‑Instruct）的 LongBench 长文本提示以及随机均匀生成的 OOD 提示，用于评估 SDH 的出现与泛化。

**📈 对比分析**

与传统方法的比较主要通过平均斜线得分和 OOD 与 in‑distribution 得分比值来衡量；结果显示 SDH 在 OOD 上得分不低于在分布内的 50%，证明了其内在性和泛化能力；理论上也给出了收敛速度和阶段性注意力聚焦的证明。

**⚠️ 局限性**

局限性包括：理论分析仅针对两层单头 Transformer，未覆盖深层网络；只研究了 RoPE 位置编码，未系统评估 ALiBi、NoPE 等其他方案；在大偏移 Δ 的情况实验有限，缺乏针对不同任务的更广泛验证。

---

## 233. D$^2$Plan: Dual-Agent Dynamic Global Planning for Complex Retrieval-Augmented Reasoning

**arXiv ID:** 2601.08282 | [PDF](https://arxiv.org/pdf/2601.08282v1)

**作者:** Kangcheng Luo `[一作]` (Peking University), Yansong Feng `[通讯]` (Peking University)

**通讯引用:** 5430 | [OpenAlex ID](https://openalex.org/A5102220317)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 D^2Plan 双代理动态全局规划框架，解决检索增强 LLM 在多跳推理中检索链构造失败与外围证据 hijacking 两大难题。

**💡 创新点**

创新点包括：① 双代理（Reasoner + Purifier）协同工作，Reasoner 负责全局规划、推理与检索；Purifier 评估检索结果并提炼关键信息；② 采用两阶段训练：先用教师模型生成高质量轨迹进行 SFT 冷启动，再通过规划导向的 RL（SPlanRL）提升动态规划能力；③ 设计初始规划奖励 R_p 与计划适配奖励 R_a，精细引导计划构造与自我修正。

**🔧 技术方法**

技术手段：基于 Qwen2.5-3B/7B Instruct 的大语言模型；检索采用 E5 嵌入与 2018 Wikipedia 语料，top‑5 文档；教师模型（如更大 LLM）用于轨迹生成；SFT 监督微调；SPlanRL（GRPO）强化学习；动态全局规划与查询重写机制。

**📊 数据集**

数据集：训练与评估使用 Natural Questions、SimpleQA、HotpotQA、2WikiMultihopQA、MuSiQue、FRAMES 六大公开 QA 基准。

**📈 对比分析**

与 Search‑R1、ReSearch、AutoRefine、ZeroSearch、R1‑Searcher、StepSearch 等 RL 基线对比，D^2Plan 在 3B 版本上平均提升 1.34% LasJ、7B 版本平均提升 3.83% LasJ；在高跳数任务（如 FRAMES）提升显著，提升幅度可达 58.3%。

**⚠️ 局限性**

局限性：仅在 3B/7B 模型上验证；使用本地 E5 + Wikipedia 检索，未测试更强检索系统；依赖强教师模型生成训练轨迹，可能受成本与可获取性限制。

---

## 234. One-Shot Identification with Different Neural Network Approaches

**arXiv ID:** 2601.08278 | [PDF](https://arxiv.org/pdf/2601.08278v1)

**作者:** Janis Mohr `[一作]` (Bochum University of Applied Sciences), Jörg Frochte `[通讯]` (Bochum University of Applied Sciences)

**通讯引用:** 137 | [OpenAlex ID](https://openalex.org/A5027576910)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了三种一次性学习/零样本识别方法：基于堆叠图像的CNN、传统孪生网络以及孪生胶囊网络，并在工业电极、smallNORB和AT&T人脸数据集上进行验证。

**💡 创新点**

提出了将两张图像按通道堆叠后直接输入CNN进行一次性识别的方案，并将胶囊网络与孪生结构结合，利用胶囊的向量特性和动态路由实现高效的一次性学习。

**🔧 技术方法**

使用了CNN、孪生网络、胶囊网络（CapsNet）及其动态路由、对比损失、解码器生成、以及数据增强（仿射变换、亮度、烧痕模拟）等技术。

**📊 数据集**

使用了三套数据集：自制的工业电极生产过程图像、smallNORB立体灰度图像数据集以及AT&T人脸数据库。

**📈 对比分析**

通过10折交叉验证比较三种方法的准确率。工业数据集：堆叠图像98.4% > 孪生胶囊96.4% > 孪生CNN96.4%；smallNORB：孪生胶囊98.4% > 堆叠图像94.7% > 孪生CNN92.5%；AT&T：孪生胶囊90.2% > 堆叠图像88.6% > 孪生CNN87.3%。显示孪生胶囊在多域任务中表现最优。

**⚠️ 局限性**

局限性包括：堆叠图像方法受图像尺寸和通道数限制；胶囊网络参数量大、训练成本高；实验仅覆盖三种数据集，缺乏更广泛验证；对实时性和部署成本的评估不充分。

---

## 235. Hybrid Distillation with CoT Guidance for Edge-Drone Control Code Generation

**arXiv ID:** 2601.08412 | [PDF](https://arxiv.org/pdf/2601.08412v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 236. Large Multimodal Models for Embodied Intelligent Driving: The Next Frontier in Self-Driving?

**arXiv ID:** 2601.08434 | [PDF](https://arxiv.org/pdf/2601.08434v1)

**作者:** Long Zhang `[一作]` (Hebei University of Engineering), Yuchen Xia `[通讯]` (Hebei University of Engineering)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个将大型多模态模型（LMM）与深度强化学习（DRL）相结合的语义与策略双驱动混合决策框架，用于实现嵌入式智能驾驶的持续学习与联合决策；

**💡 创新点**

创新点在于：①将LMM用于语义理解与认知表征，DRL用于实时策略优化；②引入融合管道以对齐两者决策并实现反馈学习；③实现了闭环的持续学习与动态适应，显著提升了在复杂交通环境中的泛化能力；

**🔧 技术方法**

使用技术包括：大型多模态模型（如PaliGemma、Gemma 2B、SigLIP）、深度强化学习中的D3QN算法、LoRA微调、AdamW优化器以及多模态编码、对齐与生成技术；

**📊 数据集**

实验使用公开数据集nuScenes，包括多视角视频、鸟瞰图像和问答注释；

**📈 对比分析**

与基线DDQN、DQN以及无LMM的DRL方案进行对比，结果显示所提框架在收敛速度和平均奖励上分别优于DDQN约19.47%和DQN约31.07%，并在车辆密度变化下保持更高的动态适应性；

**⚠️ 局限性**

局限性包括：仅在车道变更任务中验证，缺乏多任务与真实道路测试；计算复杂度较高，需进一步优化部署；对安全性与鲁棒性的评估仍不充分，未来需结合更广泛的场景与安全防护研究。

---

## 237. Thematic Working Group 5 -- Artificial Intelligence (AI) literacy for teaching and learning: design and implementation

**arXiv ID:** 2601.08380 | [PDF](https://arxiv.org/pdf/2601.08380v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 238. Teaching Robots Like Dogs: Learning Agile Navigation from Luring, Gesture, and Speech

**arXiv ID:** 2601.08422 | [PDF](https://arxiv.org/pdf/2601.08422v1)

**作者:** Taerim Yoon `[一作]` (Korea University), Sungjoon Choi `[通讯]` (Korea University)

**通讯引用:** 1703 | [OpenAlex ID](https://openalex.org/A5047885515)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种人机交互式框架LURE，使四足机器人能够通过自然的手势与语音指令实现导航与敏捷动作；

**💡 创新点**

创新点包括：①多模态（手势+语音）融合以消除空间歧义；②进阶目标提示（progressive goal cueing）在训练中动态对齐指令与机器人状态；③利用场景重构与数据聚合（DAgger）降低样本需求并缓解分布漂移；④仅用少量（<1小时）演示数据即可快速适应新用户；

**🔧 技术方法**

技术手段：运动捕捉手势关键点提取、语音识别与文本编码、基于物理的场景重建、局部专家策略与数据聚合、域随机化、速度规划与跟踪控制、LLM生成语义等价表达；

**📊 数据集**

使用基于六种交互场景（Go there、Come here、Follow me、Come around、Jump over、Zigzag）收集的手势、语音与机器人状态数据，采样频率10Hz，总演示时长不到1小时；

**📈 对比分析**

与BC、GAIL、DAgger（无进阶提示）对比，LURE平均成功率97.15%，导航误差显著下降；DAgger比BC提升了约18.6%成功率、24.6%误差；再加入进阶提示则再提升约13.7%成功率、15.2%误差；

**⚠️ 局限性**

局限性：缺乏跨用户零样本泛化，手势表征仅通过噪声扩增缺乏语义多样性；进阶提示仅对原始演示做时间重排，无法生成全新行为；依赖运动捕捉与简单几何障碍，难以直接迁移至开放场景与复杂视觉感知。

---

## 239. MMLGNet: Cross-Modal Alignment of Remote Sensing Data using CLIP

**arXiv ID:** 2601.08420 | [PDF](https://arxiv.org/pdf/2601.08420v1)

**作者:** Aditya Chaudhary `[一作]` (LNMIIT Jaipur), Biplab Banerjee `[通讯]` (IIT Bombay)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种多模态语言引导网络（MMLGNet），将高光谱影像（HSI）与 LiDAR 数据与自然语言描述通过 CLIP 模型对齐，以实现跨模态语义理解。

**💡 创新点**

创新点在于将语言监督引入多模态遥感，使用简单 CNN 编码器和双向对比损失将视觉特征映射到共享文本嵌入空间，显著提升了语义对齐和分类性能。

**🔧 技术方法**

采用的技术包括：模态特定 CNN 编码器、融合层、CLIP 预训练冻结文本编码器、L2 归一化的双向对比损失以及基于土地覆被类别的手工文本提示。

**📊 数据集**

使用的公开数据集为 MUUFL Gulfport（11 类）和 Trento（6 类），每个样本提取 11×11 的 HSI+LiDAR 补丁进行训练和测试。

**📈 对比分析**

与 SVM、ELM、TB-CNN、FusAtNet 等纯视觉方法对比，MMLGNet 在 Trento 数据集上 OA 99.42%、AA 99.18%，在 MUUFL 上 OA 88.79%、AA 90.87%，整体在多项指标上均优于或接近最优，证明语言引导的有效性。

**⚠️ 局限性**

局限性包括：依赖手工制作的提示词，未探索自动提示学习；仅验证了 HSI+LiDAR 两种模态，对其他遥感传感器的适用性未知；模型规模小，可能无法捕捉更复杂的空间光谱特征；实验数据集规模有限。

---

## 240. A Qualitative Model to Reason about Object Rotations (QOR) applied to solve the Cube Comparison Test (CCT)

**arXiv ID:** 2601.08382 | [PDF](https://arxiv.org/pdf/2601.08382v1)

**作者:** Zoe Falomir `[一作]` (Umeå University), Zoe Falomir `[通讯]` (Umeå University)

**通讯引用:** 775 | [OpenAlex ID](https://openalex.org/A5054234494)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Qualitative Object Rotation (QOR)模型，利用定性空间表示与推理解决Cube Comparison Test (CCT) 的立方体旋转与可视面推断问题，并实现了交互式测试平台。

**💡 创新点**

创新点在于首次将定性空间表示(QSR)与概念邻域图（CNG_RLO）相结合，构建可描述立方体旋转导致的面位置与方向变化的量化模型，并用它来推理 CCT。

**🔧 技术方法**

采用了定性空间表示与推理技术，构建概念邻域图、组合表格以及基于特征、位置、方向的三维参考系来实现旋转推理。

**📊 数据集**

主要使用 Cube Comparison Test（CCT）提供的 21 道题目图形作为实验数据，没有使用公开的大规模数据集。

**📈 对比分析**

比较方法是统计两块立方体的重复特征数 R，并在 CNG_RLO 图上搜索旋转路径验证面位置与方向是否可匹配；论文未给出具体准确率或性能指标，主要展示理论可行性与可解释性。

**⚠️ 局限性**

局限性包括：仅适用于每个面标记唯一且无重复的立方体；模型假设特征可辨识且不考虑噪声、遮挡或不同形状物体的鲁棒性；缺乏大规模实验验证与性能评估。

---

## 241. Large Language Models to Enhance Multi-task Drone Operations in Simulated Environments

**arXiv ID:** 2601.08405 | [PDF](https://arxiv.org/pdf/2601.08405v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 242. Statistical Characterization and Prediction of E2E Latency over LEO Satellite Networks

**arXiv ID:** 2601.08439 | [PDF](https://arxiv.org/pdf/2601.08439v1)

**作者:** Andreas Casparsen `[一作]` (Aalborg University), Israel Leyva Mayorga `[通讯]` (Aalborg University)

**通讯引用:** 784 | [OpenAlex ID](https://openalex.org/A5061754527)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过500 Hz高采样测量Starlink LEO网络的端到端延迟，剔除15 秒周期边缘突发，构建统计模型对内部延迟进行预测与周期级分类。

**💡 创新点**

发现并利用星链15 秒周期的确定性结构，提出轻量级统计框架，能在毫秒级误差内预测99th分位延迟并实现实时周期分类，支持动态调度。

**🔧 技术方法**

使用UDP高频一往测量工具irtt；基于边缘检测+循环平均分段周期；采用均匀、正态、GMM、经验分布和极值理论（GPD）进行分位数回归；评估AUPRC、MSE和折扣服务可用性（DSA）。

**📊 数据集**

采集多个月星链链路的500 Hz UL、DL、RTT延迟样本（数万至数十万条），覆盖不同时间段和负载条件。

**📈 对比分析**

与多种分布模型及EVT比较，GMM(3核)在1.6 s内即可达到AUPRC≈0.95，MSE≈21 ms²；GPD在3.5 s后逼近完美分类；99th分位误差可低于4.5 ms；DSA在FPR≤5%时可实现约30%服务可用性。

**⚠️ 局限性**

仅在单一星链系统验证，边缘区预测不佳；假设地面链路时变性小；未考虑多路复用与协议交互；需进一步验证跨星座及多接入场景。

---

## 243. Fine-Mem: Fine-Grained Feedback Alignment for Long-Horizon Memory Management

**arXiv ID:** 2601.08435 | [PDF](https://arxiv.org/pdf/2601.08435v1)

**作者:** Weitao Ma `[一作]` (Meituan), Bing Qin `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 15641 | [OpenAlex ID](https://openalex.org/A5017671620)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并训练了一个统一的RL框架Fine‑Mem，用于长时序任务的内存管理，能够通过学习动态插入、更新、删除和跳过操作来构建和维护有效的长期记忆；

**💡 创新点**

提出了两种细粒度奖励机制——Chunk‑level Step Reward（基于片段级QA给与即时反馈）和Evidence‑Anchored Reward Attribution（将全局奖励按证据映射回具体操作），从而显著缓解奖励稀疏和信用分配难题；

**🔧 技术方法**

采用基于GRPO的强化学习优化、单层内存架构、四元操作空间、Chunk‑level QA生成、EARA奖励归因、BM25检索以及vLLM作为推理代理等技术；

**📊 数据集**

在Memalpha（训练/验证）和MemoryAgentBench（测试/OOD）两个基准上进行实验；

**📈 对比分析**

与七类基线（无内存、工作流式、训练型管理器）对比，Fine‑Mem平均提升约4.4%/7.2%，在所有子任务（准确检索、时刻学习、长范围理解）上均为首或次优，同时保持较紧凑的记忆长度；

**⚠️ 局限性**

局限性包括仅使用BM25检索，难以捕获深层语义关系；仅处理文本模式；未实现与推理模型的协同进化，导致推理能力无法共同提升。

---

## 244. Regulatory gray areas of LLM Terms

**arXiv ID:** 2601.08415 | [PDF](https://arxiv.org/pdf/2601.08415v1)

**作者:** Brittany I. Davidson `[一作]` (University of Bath), Adam N. Joinson `[通讯]` (University of Bath)

**通讯引用:** 12602 | [OpenAlex ID](https://openalex.org/A5040132838)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对Anthropic、DeepSeek、Google、OpenAI、xAI五大LLM提供商在2025年11月的服务条款进行系统比较，揭示了研究者在使用LLM时面临的监管灰区。

**💡 创新点**

创新点在于构建公开可访问的跨平台条款对比资源，并以细致的归类方式揭示研究者使用受限与灰色监管区，弥补了以往缺乏系统性条款分析的空白。

**🔧 技术方法**

采用文档分析与内容分析相结合的方法，对条款进行拆解、交叉引用和归类，形成条款覆盖表。

**📊 数据集**

数据集为2025年11月收集的五大LLM提供商公开的服务条款文本。

**📈 对比分析**

比较方法通过构建条款覆盖表，对各类限制项进行量化，结果显示Anthropic条款最为全面，其次是OpenAI，而Google与DeepSeek条款相对简略，整体性能表现为对研究者影响的清晰可视化。

**⚠️ 局限性**

局限性包括：条款仅为2025年11月的快照，未覆盖整个LLM生态；缺乏对研究者实际使用体验的访谈验证；条款随时间变动可能导致结论过时。

---

## 245. Out-of-distribution generalization of deep-learning surrogates for 2D PDE-generated dynamics in the small-data regime

**arXiv ID:** 2601.08404 | [PDF](https://arxiv.org/pdf/2601.08404v1)

**作者:** Binh Duong Nguyen `[一作]` (Forschungszentrum Jülich GmbH), Stefan Sandfeld `[通讯]` (RWTH Aachen University)

**通讯引用:** 1577 | [OpenAlex ID](https://openalex.org/A5022752212)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种周期性二维 PDE 动力学的自回归深度学习替代模型，并在小样本条件下评估其性能。

**💡 创新点**

创新点在于设计了多通道、周期填充的 U‑Net（me‑UNet），在保证周期边界的同时通过残差预测实现高效时间步进，并在有限数据与 OOD 初始条件下表现出与更复杂模型相当或更优的结果。

**🔧 技术方法**

使用卷积 U‑Net、周期填充、平均池化、残差输出以及与 ViT、AFNO、PDE‑Transformer、KAN‑UNet 等架构相同的训练协议（MSE+感知损失），并采用 Grad‑CAM 等可视化手段分析网络内部。

**📊 数据集**

在十个基于五类 PDE 的合成数据集（DS‑1~DS‑6d，覆盖线性输运、扩散、连续位错动力学、Kolmogorov 流和 Gray‑Scott 反应扩散）上进行实验，每个数据集包含 100 条 64×64 的周期网格轨迹。

**📈 对比分析**

在全局误差、谱相似度以及物理守恒量三方面与 ViT、AFNO、PDE‑Transformer、KAN‑UNet 进行比较，me‑UNet 在大多数数据集上实现最低 RMSE、最高 PSD 相似度，并在 OOD 初始条件下仍保持较低误差；训练时间也显著更短。

**⚠️ 局限性**

局限性包括仅考虑二维结构化周期网格、固定分辨率、单一标量观测、仅在同一 PDE 参数下的 OOD（初始条件）而非跨 PDE 或参数的泛化、未对边界条件进行硬约束或物理驱动的损失函数，且对非结构化网格或三维问题支持不足。

---

## 246. Owen-Shapley Policy Optimization (OSPO): A Principled RL Algorithm for Generative Search LLMs

**arXiv ID:** 2601.08403 | [PDF](https://arxiv.org/pdf/2601.08403v1)

**作者:** Abhijnan Nath `[一作]` (Amazon Science), Nikhil Krishnaswamy `[通讯]` (Department of Computer Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于Owen–Shapley值的策略优化框架OSPO，用以在单回合推荐任务中解决奖励稀疏导致的信用分配问题。

**💡 创新点**

创新点在于通过将生成序列划分为语义片段并计算Owen–Shapley贡献值，直接将序列级奖励细化到段级，从而在不使用价值网络的情况下实现精细梯度分配。

**🔧 技术方法**

技术上结合了协同博弈论中的Owen–Shapley值、连续子集采样、潜在奖励塑形与GRPO的群体优势分布，形成新的优势重分配目标。

**📊 数据集**

实验数据集涵盖Amazon ESCI商品查询和H&M时尚推荐，使用FAISS检索与领域特定编码器进行评估。

**📈 对比分析**

与SFT、DPO、GRPO等基线对比，OSPO在ESC I的NDCG从0.418提升至0.522，在H&M上从0.379提升至0.436，显著超越同规模模型并接近十倍参数模型。

**⚠️ 局限性**

局限性包括额外的Owen值估算计算开销、对协同块宽度和采样深度的敏感性，以及仅验证了单回合场景，未覆盖多轮或多模态任务。

---

## 247. PATS: Personality-Aware Teaching Strategies with Large Language Model Tutors

**arXiv ID:** 2601.08402 | [PDF](https://arxiv.org/pdf/2601.08402v1)

**作者:** Donya Rooein `[一作]` (Bocconi University), Dirk Hovy `[通讯]` (Bocconi University)

**通讯引用:** 7242 | [OpenAlex ID](https://openalex.org/A5084505122)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了把17种教学策略映射到五大人格维度的分类法，并基于此设计了PATS框架，让LLM教师根据学生人格自适应教学；通过LLM仿真生成并评估师生对话；

**💡 创新点**

首次系统化把学习科学中的人格-教学策略对应关系融入LLM ITS，并在对话层面实现人格感知与策略匹配；

**🔧 技术方法**

使用大语言模型（GPT‑4o、Llama 3.3 70B、Gemini‑2.0‑Flash）结合两阶段规划/响应的prompt工程，构建学生与教师模拟器；

**📊 数据集**

利用110名意大利高中学生的匿名对话日志（BFT人格测评）、10幅图片、10个故事作为任务素材，并在此基础上生成640+对话用于评估；

**📈 对比分析**

与两种基线（无人格信息、仅知人格但无策略指令）进行对话比较，采用LLM偏好测试和4位专业教师的对话对比评估；PATS在三大LLM上对两任务的胜率均超过80%，在人类评估中在动机、人格适配、参与度、同理心等维度显著优于基线；

**⚠️ 局限性**

限制包括：仅模拟单一人格维度，未考虑认知水平等重要因素；所用策略有限（仅12种在对话中实际使用）；缺乏真实课堂干预验证学习成效；模型偏向默认的提问答式或提示式策略。

---

## 248. An Explainable Two Stage Deep Learning Framework for Pericoronitis Assessment in Panoramic Radiographs Using YOLOv8 and ResNet-50

**arXiv ID:** 2601.08401 | [PDF](https://arxiv.org/pdf/2601.08401v1)

**作者:** Ajo Babu George `[一作]` (DiceMed), Kunal Agarwal `[通讯]` (S.C.B Dental College and Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一个两阶段深度学习框架，先用YOLOv8定位并分类全景片中第三磨牙的解剖位置和角度，再用改进的ResNet‑50对裁剪的牙齿 ROI 进行周围炎的二分类，并通过Grad‑CAM生成可解释热图。

**💡 创新点**

创新点在于将解剖定位、病理分类与可解释 AI 融合到同一管线，首次在全景片上实现基于 Winter 角度的第三磨牙定位并提供临床可验证的热图。

**🔧 技术方法**

使用技术包括 YOLOv8 目标检测、ResNet‑50 分类器、Grad‑CAM 可解释性可视化以及多源数据集的分层训练和增强策略。

**📊 数据集**

采用了 1,190 张多中心全景片（含 3 号牙定位）和 1,550 张裁剪的第三磨牙 ROI（775 病变、775 对照），并利用 Kaggle、Zenodo 等公开数据库构建数据集。

**📈 对比分析**

在验证集上 YOLOv8 达到 92% 精度和 92.5% mAP50，ResNet‑50 在病变/正常分类上 F1 分别为 0.86 与 0.88，AUC 为 0.94；与单模型基准相比保持竞争力，并获得 84% 临床医师对 Grad‑CAM 匹配率。

**⚠️ 局限性**

限制包括数据集规模虽大但可能不足以覆盖所有解剖和病理变异；未使用亮度/对比增强导致对不同设备的鲁棒性有限；仅识别放射学表现，无法区分急慢期；部署受硬件与监管等限制。

---

## 249. Symbolic Functional Decomposition: A Reconfiguration Approach

**arXiv ID:** 2601.08354 | [PDF](https://arxiv.org/pdf/2601.08354v1)

**作者:** Mateus de Oliveira Oliveira `[一作]` (Stockholm University), Wim Van den Broeck `[通讯]` (University of Bergen)

**通讯引用:** 75 | [OpenAlex ID](https://openalex.org/A5113306687)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文提出了一种符号化的功能重构与功能分解框架，能够在给定布尔函数（通过 OBDD 表示）以及一个布尔电路 C 的情况下，将该函数分解为 C 组合的若干更简单子函数；

**💡 创新点**

核心创新在于将功能重构宽度（即在重构过程中出现的中间函数的 OBDD 宽度）与电路结构、目标函数类（用二阶有限自动机表示）等参数相结合，构造了一个可在参数化线性时间内求解的算法；

**🔧 技术方法**

主要技术包括：1) OBDD 的符号化表示与最小化；2) 二阶有限自动机对函数类的闭包与判定；3) 通过自动机理论构造“选择器”与张量积语言，递归维护电路一致性；4) 参数化复杂度分析与固定参数可收敛性证明；

**📊 数据集**

论文为纯理论研究，未使用实际数据集；主要以 OBDD 长度、宽度、电路门数和自动机状态数等参数为实验维度进行理论复杂度评估；

**📈 对比分析**

与传统的功能分解或 DFA/OBDD 因子化方法相比，该方法在参数 k（因子数）、OBDD 宽度 w、电路门数 m、自动机大小 |A| 等方面实现了 2^{O(k)}·|D| 的固定参数线性时间；在小参数范围内显著优于暴力枚举与一般 NP‑hard 方法；

**⚠️ 局限性**

局限性包括：1) 仅适用于 OBDD 形式的输入函数；2) 对于宽度、门数或自动机大小过大的实例，指数因子仍可能使算法不可行；3) 需要预先给定或枚举电路结构 C，未给出高效的电路搜索策略；4) 实际工程中 OBDD 生成与最小化本身也可能成为瓶颈。

---

## 250. Beyond Linearization: Attributed Table Graphs for Table Reasoning

**arXiv ID:** 2601.08444 | [PDF](https://arxiv.org/pdf/2601.08444v1)

**作者:** Yuxiang Wang `[一作]` (University of Melbourne), Jianzhong Qi `[通讯]` (University of Melbourne)

**通讯引用:** 4463 | [OpenAlex ID](https://openalex.org/A5022290876)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于Attributed Table Graph的表格推理框架Table Graph Reasoner，能够在不训练的前提下对表格进行图结构化表示和推理。

**💡 创新点**

创新点包括将表格转化为ATG以显式保留行列单元结构，结合Question‑Guided Personalized PageRank实现对关键三元组的重排序，解决了线性化方法的“失去中间信息”问题，并提供可解释的细粒度推理路径。

**🔧 技术方法**

使用图结构建模、Personalized PageRank、LLM的链式思维（CoT）进行答案生成。

**📊 数据集**

在WikiTableQuestions和TableFactVerification两个公开数据集上进行实验，并在更复杂的Hierarchical Table数据集上做了验证。

**📈 对比分析**

与现有分解式和全表格式方法（如Table‑Critic、RoT等）对比，TGR在不同LLM骨干下的准确率平均提升约9.7%（TableQA）和2.5%（TableFV），并在表格排列扰动、token成本和大表格表现上更稳健。

**⚠️ 局限性**

局限性包括仅针对文本表格，未评估多模态表格或非英文表格；并且在全图推理模式下仍需较高的输入token。

---

## 251. YaPO: Learnable Sparse Activation Steering Vectors for Domain Adaptation

**arXiv ID:** 2601.08441 | [PDF](https://arxiv.org/pdf/2601.08441v1)

**作者:** Abdelaziz Bounhar `[一作]` (Mohammed Bin Zayed University of Artificial Intelligence), Guokan Shang `[通讯]` (Mohammed Bin Zayed University of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出YaPO方法，学习稀疏、偏好优化的激活干预向量，实现对LLM的文化适配及其他对齐任务控制。

**💡 创新点**

将稀疏自编码器与BiPO偏好优化相结合，得到可解释、稀疏且稳定的steering向量；同时首次构建跨语言跨文化的细粒度评测数据。

**🔧 技术方法**

利用稀疏自编码器（SAE）编码激活、Bi‑directional Preference Optimization (BiPO) 目标、稀疏向量优化与激活注入技术。

**📊 数据集**

自制五语种、十五文化背景的文化适配数据集；MMLU基准；Hallucination、Wealth‑Seeking、Jailbreak、Power‑Seeking等对齐任务数据。

**📈 对比分析**

与无干预、CAA、SAS、BiPO比较；YaPO收敛更快、训练更稳定、在文化适配、MCQ 与开放式生成任务中获得最高RCA、最低PNLG；在MMLU上保持知识水平；在其他对齐任务中排名第二。

**⚠️ 局限性**

仅在Gemma‑2系列模型上验证；依赖已训练的SAE，若无可用需自行训练；数据集未涵盖同国内部多样性，跨模型迁移尚未评估。

---

## 252. Protrusion Decompositions Revisited: Uniform Lossy Kernels for Reducing Treewidth and Linear Kernels for Hitting Disconnected Minors

**arXiv ID:** 2601.08424 | [PDF](https://arxiv.org/pdf/2601.08424v1)

**作者:** Roohani Sharma `[一作]` (Institute for Basic Science), Michał Włodarczyk `[通讯]` (University of Warsaw)

**通讯引用:** 201 | [OpenAlex ID](https://openalex.org/A5103139616)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对包含平面图的禁用子图族F，给出了统一的近似核化（lossy kernelization）方法：在保持 2 近似的情况下实现多项式核化，并进一步通过多次调用核化算子将近似因子逼近 1+ε。

**💡 创新点**

创新点在于：①将近似的“near‑protrusion”分解转化为真正的 protrusion 分解并容忍少量近似损失；②提出“二分性”性质（dichotomy property），使得即使 F 包含非连通图也能在保持可压缩性的前提下进行 protrusion 替换；③构建统一的 lossy kernel 化协议，突破了传统核化对 η 的非统一依赖。

**🔧 技术方法**

核心技术包括：Protrusion 技术（近似 protrusion 分解、近似压缩、替换）；LCA 闭包与树宽控制；近似近似 kernelization（2-近似核化、(1+ε)-近似协议）；多项式 Turing 核化（利用小实例的 oracle 并多次调用）。

**📊 数据集**

本研究属于理论计算复杂性，不涉及实验数据集，全部工作在图论与参数化算法的理论框架下完成。

**📈 对比分析**

与以往的 Fomin‑Lokshtanov‑Misra‑Saurabh（2012）非统一多项式核化相比，本文的 2‑近似核化大小为 g(η)·k^5，且可通过多次 oracle 调用将近似因子进一步压到 1+ε；在保持相同 k 的前提下，核大小与 η 的指数依赖被显著削弱，体现出更好的理论性能。

**⚠️ 局限性**

局限性：①需要对小规模实例调用一个可提供 β‑近似解的 oracle；②多次 oracle 调用导致协议仍非完全统一；③近似因子虽可逼近 1+ε，但仍不是完全精确的核化；④仅适用于至少包含一个平面图的禁用子图族。

---

## 253. Taxon: Hierarchical Tax Code Prediction with Semantically Aligned LLM Expert Guidance

**arXiv ID:** 2601.08418 | [PDF](https://arxiv.org/pdf/2601.08418v1)

**作者:** Jihang Li `[一作]` (Hong Kong University of Science and Technology), Zeyi Wen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1170 | [OpenAlex ID](https://openalex.org/A5013127195)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在电商平台上构建了一套自动化的分层税码预测系统，解决商品标题等信息映射到多层税码树的难题。

**💡 创新点**

创新点包括：①基于特征门控的混合专家（MoE）架构，实现多层级的自适应特征路由；②通过大语言模型蒸馏出的语义一致性监督，显式对商品描述与官方税码定义进行对齐；③RePath后处理策略，从叶节点重建完整路径，显著提升结构一致性。

**🔧 技术方法**

核心技术：预训练文本编码器（BERT）、特征门控MoE分类器、层级交叉熵损失、LLM蒸馏的语义一致性损失、RePath路径重构。

**📊 数据集**

数据集：本土化的内部TaxCode大规模中文税码数据（约85万条训练、13.6万验证、17万测试）以及公开英文WOS两层税码基准。

**📈 对比分析**

与HGCLR、HILL、LH-Mix等现有层级与语义兼顾基线对比，Taxon在路径级宏/微F1分别达到89.37/93.16，叶节点宏/微F1 78.30/87.06，优于所有基线；在WOS上也实现86.74/93.67宏/微F1，保持领先。

**⚠️ 局限性**

局限性：①语义监督受LLM标注误差影响，提升有限；②对深层次结构误差仍存在，RePath虽改善但未完全消除；③模型对内部业务数据高度依赖，迁移到其他行业或语言仍需进一步验证。

---

## 254. An Efficient Algorithm to Sample Quantum Low-Density Parity-Check Codes

**arXiv ID:** 2601.08387 | [PDF](https://arxiv.org/pdf/2601.08387v1)

**作者:** Paolo Santini `[一作]` `[通讯]` (Polytechnic University of Marche), Paolo Santini (Polytechnic University of Marche)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一种高效算法，用于随机采样满足自正交性且稀疏的量子LDPC码的检查矩阵。

**💡 创新点**

提出了一个纯组合的采样方法，利用信息集合解码(ISD)逐行生成自正交稀疏矩阵，从而克服了以往依赖代数结构的局限。

**🔧 技术方法**

主要技术是信息集合解码算法、随机稀疏矩阵抽样、列加权分析及理论证明。

**📊 数据集**

采用自定义的随机行权重分布ℋ_{n,r,v}，并在GitHub上提供的SageMath实现中生成不同参数的代码。

**📈 对比分析**

与随机LDPC码的Gilbert–Varshamov距离和理论期望权重分布对比，通过计时实验显示在相同参数下平均调用ISD一次即可完成采样，时间从秒级到数千秒不等。

**⚠️ 局限性**

主要局限在于对ISD在稀疏矩阵上的性能假设尚未严格证明，且在高码率或接近n/2的情况下可能陷入循环或失败。

---

## 255. Training-Free Distribution Adaptation for Diffusion Models via Maximum Mean Discrepancy Guidance

**arXiv ID:** 2601.08379 | [PDF](https://arxiv.org/pdf/2601.08379v1)

**作者:** Matina Mahdizadeh Sani `[一作]` (University of Waterloo), Farzan Farnia `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 537 | [OpenAlex ID](https://openalex.org/A5017160178)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无训练的 MMD 引导方法，在扩散模型的反向采样中加入最大均值差异（MMD）梯度，实现对目标分布的对齐。

**💡 创新点**

创新点在于将 MMD 直接嵌入采样动态，利用可微核实现分布匹配；同时通过产品核实现条件下的 prompt‑aware 对齐，并在潜在空间实现高效计算。

**🔧 技术方法**

使用最大均值差异（MMD）核、潜在扩散模型（LDM）以及潜在空间编码/解码；在训练时不改动模型参数，只在推理时添加梯度。

**📊 数据集**

在合成高维高斯混合、FFHQ、CelebA‑HQ 以及多风格的合成图像集上进行评估；同时使用真实的医疗 X 光、品牌风格等假想数据进行验证。

**📈 对比分析**

与无引导、CFG、CLF、域引导等基线比较；实验显示 MMD 引导在 Fréchet 距离、Kernel 距离、覆盖率等指标上优于或接近微调模型，同时保持生成质量。

**⚠️ 局限性**

局限在极少参考样本（极低方差）时性能下降，且对核选择敏感；目前只针对图像扩散，视频等顺序生成尚未验证。

---

## 256. A New Tool to Find Lightweight (And, Xor) Implementations of Quadratic Vectorial Boolean Functions up to Dimension 9

**arXiv ID:** 2601.08368 | [PDF](https://arxiv.org/pdf/2601.08368v1)

**作者:** Marie Bolzer `[一作]` (University of Lorraine), Marine Minier `[通讯]` (University of Lorraine)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

开发了一款专门针对二次向量布尔函数（S‑box）的工具，可在 AND‑深度为 1 的前提下，最小化实现中的 AND 门数，支持到 9 位输入。

**💡 创新点**

核心创新在于：①预计算所有可由单个 AND 门产生的二次布尔函数，并将实现问题转化为大规模线性方程组；②采用分支定界与贪心搜索相结合的算法；③利用多线程并行化和稀疏向量存储，显著降低时间和内存开销；④在 7‑9 位范围内实现了从 10‑15% 的 AND 门数和数倍的速度提升。

**🔧 技术方法**

技术实现主要包括 C++ 编程、位运算编码（多项式压缩为整数）、预处理阶段生成 1‑AND 操作集合、二次函数与线性函数的分离求解、分支定界搜索、并行化（std::thread/parallel_for）、稀疏向量存储与哈希表等。

**📊 数据集**

实验数据集：多类 S‑box，覆盖 5‑9 位，包括 Keccak/ASCON、随机生成、X^k（k 为 3、5、7、9、…）以及 CCZ‑等价的 APN 函数，均以查表形式输入。

**📈 对比分析**

与现有工具（如 ABC、Stoffelen 等）比较，所提出工具在 AND 门数与运行时间上均优于传统工具；在 8‑9 位 S‑box 上实现的 AND 门数与最优或近最优值相当，且完成时间从数小时降至数分钟；同时，工具能够在 9 位时仍保持内存占用 < 15% 的可用 RAM。

**⚠️ 局限性**

局限性：仅限于二次函数，最多 9 位；从 8 位起搜索不再完全全局最优，采用随机局部搜索；预计算文件巨大，对内存要求高；未针对高阶 S‑box 或其它复杂度度量（如 Gate‑Equivalent、Bit‑Slice Gate Complexity）进行优化。

---

## 257. MLPlatt: Simple Calibration Framework for Ranking Models

**arXiv ID:** 2601.08345 | [PDF](https://arxiv.org/pdf/2601.08345v1)

**作者:** Piotr Bajger `[一作]` (Allegro sp. z o.o.), Paweł Zawistowski `[通讯]` (Warsaw University of Technology)

**通讯引用:** 102 | [OpenAlex ID](https://openalex.org/A5023467386)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了MLPlatt，一种用于将无校准的排名模型输出转换为可解释的点击率（CTR）概率的后置校准方法，并保持排名顺序不变。

**💡 创新点**

创新点在于：①通过上下文感知的多层网络（Context Model + MonoMLP）实现对列表级上下文特征的充分利用；②在损失中加入单调性约束，确保校准后仍保持排名顺序；③兼顾全局和字段级（device、country）校准，显著提升F‑ECE。

**🔧 技术方法**

使用技术包括：多层感知器（MLP）与单调性正则化、基于上下文嵌入的校准头、二阶段训练（先训练排名器后训练校准器）、比较实验采用Platt、Isotonic、ConfCalib、DESC、RCR等方法。

**📊 数据集**

实验数据集：专有的 Allegro（约2亿条搜索列表，覆盖多设备）和公开的 AliExpress（约5百万条列表，4个国家）。

**📈 对比分析**

与传统校准方法和基于 RCR 的直接训练方法对比，MLPlatt 在 F‑ECE、Log‑Loss、AUC 上均取得最低（最优）分数，且不降低 NDCG，证明了在保持排名质量的同时实现高质量校准的可行性。

**⚠️ 局限性**

局限性包括：①需要两阶段训练，需先获得所有列表的排名分数；②校准模型不利用条目级特征，可能在极端稀疏上下文下表现受限；③在极大规模实时系统中仍需评估增量训练和低延迟部署的实际成本。

---

## 258. When KV Cache Reuse Fails in Multi-Agent Systems: Cross-Candidate Interaction is Crucial for LLM Judges

**arXiv ID:** 2601.08343 | [PDF](https://arxiv.org/pdf/2601.08343v1)

**作者:** Sichu Liang `[一作]` (Southeast University), Deyu Zhou `[通讯]` (Southeast University)

**通讯引用:** 2631 | [OpenAlex ID](https://openalex.org/A5007145568)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在多代理LLM系统中判决者（judge）侧 KV 缓存重用对判决一致性的影响，发现虽然任务准确率保持稳定，但判决者对候选答案的选择会出现显著不稳定。

**💡 创新点**

创新点在于提出 Judge Consistency Rate (JCR) 指标来量化判决一致性，并通过注意力诊断和掩码实验揭示 KV 重用削弱了判决者的跨候选交互，导致选择行为非单调；同时系统性地比较了三种重用方案（Naïve、KVCOMM、PAL‑KV）与 dense prefill 的差异。

**🔧 技术方法**

主要技术包括 KV 缓存重用（Naïve、KVCOMM、PAL‑KV）、基于 anchor 的偏移校正、候选交互注意力分析、以及对候选顺序扰动的实验设计；判决者使用 Llama‑3.1/3.2 与 Qwen‑2.5 等模型进行推理。

**📊 数据集**

使用了三大基准数据集：GSM8K（算术推理）、MMLU（多领域知识测试）和 HumanEval（代码生成），每个样本生成 4 条候选答案，分别在 Progressive Refinement 与 Parallel Exploration 两种候选生成模式下评估。

**📈 对比分析**

与 dense prefill 对比，重用率可高达 70% 以上，任务准确率基本保持不变，但 JCR 在大多数设置下显著下降（低至 20–30%），说明判决一致性受损；不同重用方案在准确率上相近，但 JCR 下降幅度相似，显示问题普遍存在。

**⚠️ 局限性**

主要局限在于：①仅关注判决者侧重用，未评估完整多代理链条的整体效率与一致性；②实验仅在同一模型族且参数不超过 14B 的开源模型上进行；③未探讨不同 Prompt 设计、异构模型或超大模型的适用性；④未量化最终系统能耗与延迟提升。

---

## 259. Edge-Optimized Multimodal Learning for UAV Video Understanding via BLIP-2

**arXiv ID:** 2601.08408 | [PDF](https://arxiv.org/pdf/2601.08408v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 260. Deep Learning Based Facial Retargeting Using Local Patches

**arXiv ID:** 2601.08429 | [PDF](https://arxiv.org/pdf/2601.08429v1)

**作者:** Yeonsoo Choi `[一作]` (Netmarble F&C), Junyong Noh `[通讯]` (Visual Media Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于局部补丁的表情转移方法，能够将真人表演视频中的面部动画迁移到面部结构与尺寸差异显著的风格化3D角色上。

**💡 创新点**

核心创新点在于自动补丁提取与对齐模块（APEM）实现源目标面部补丁的精确对应，利用局部补丁进行图像到图像的再现（RM）并通过重建的补丁回归PCA动画参数（WEM），从而在无配对数据条件下处理大幅结构差异。

**🔧 技术方法**

采用HRNet人脸关键点检测、卷积自编码器进行图像到图像的翻译、PCA线性混合形状系数回归、数据增强、以及全局编码网络与MLP的组合。

**📊 数据集**

使用两位表演者（共33,618帧60FPS）作为源视频，和五个风格化角色（Mery、Malcolm、Child、Girl、Piers、Metahuman）以及对应的ARKit或人工制作的动画帧和20维PCA参数作为目标数据。

**📈 对比分析**

与Kim et al.和Moser et al.的两种基准方法对比，定量上顶点误差约为1.4mm（比2.6mm和6.3mm低），训练时长774min（中间值），推理时间24ms（略高于2.69ms和6.31ms），在不同光照和复杂表情下表现更稳定。

**⚠️ 局限性**

局限性包括仅适用于人类形态或近似人类的风格化角色，无法处理如大象鼻或昆虫触角等非人类特征；需人工标注目标模型关键点；对单一表演者的适配性有限；对眼睛完全闭合、头部俯仰/仰角运动的迁移效果不足；对角色骨骼/表情细节的依赖较强。

---

## 261. Silence the Judge: Reinforcement Learning with Self-Verifier via Latent Geometric Clustering

**arXiv ID:** 2601.08427 | [PDF](https://arxiv.org/pdf/2601.08427v1)

**作者:** Nonghai Zhang `[一作]` (Peking University), Jingwen Xu `[通讯]` (Meituan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Latent-GRPO 框架，利用 LLM 潜在空间几何特征自我评估，生成密集奖励，替代外部验证器以提升推理任务训练效率。

**💡 创新点**

创新点在于发现正确推理路径终端隐藏状态形成聚簇，基于此提出迭代鲁棒质心估计（IRCE）算法，提供自监督、连续奖励，并结合 GRPO 实现高效训练。

**🔧 技术方法**

使用了 Transformer 隐藏状态提取、球面归一化、IRCE 迭代软加权质心、连续奖励映射、GRPO 强化学习等技术。

**📊 数据集**

实验数据集包括 GSM8K、MATH、Open‑Platypus 以及 AIME、MMLU、BBH 等通用推理基准。

**📈 对比分析**

与 LLM‑as‑Judge、规则基验证器对比，Latent‑GRPO 在相同模型规模下训练速度提升约 2×，准确率保持或超越基线，并在多模型、多任务上表现稳健。

**⚠️ 局限性**

局限性：仅验证到 8B 参数规模，超大模型（70B+）及开放式生成任务的适用性未知；缺乏正式的潜在空间聚类理论。

---

## 262. Lower Bounds for Dominating Set in Ball Graphs and for Weighted Dominating Set in Unit-Ball Graphs

**arXiv ID:** 2601.08425 | [PDF](https://arxiv.org/pdf/2601.08425v1)

**作者:** Mark de Berg `[一作]` (Eindhoven University of Technology), Sándor Kisfaludi-Bak `[通讯]` (Max Planck Institute for Informatics)

**通讯引用:** 133 | [OpenAlex ID](https://openalex.org/A5057527289)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

本文研究了在单位球图以及更一般的“肥”对象交叉图上经典图论问题（如最小支配集、最小连通支配集、反馈顶点集、最小染色数等）的子指数算法，并给出了相应的下界，证明在三维及更高维度中大部分问题在单位球图上不可在 2^o(n) 时间内解决。

**💡 创新点**

创新点包括：① 将 De Berg 等人的基于团分离器的框架推广至非同尺寸肥对象，并通过精心构造的几何实例展示该推广在多种问题上不可行；② 通过 (3,3)-SAT 归约构造球图和单位球图，证明在三维单位球图上最小支配集及其加权版本不具子指数上界；③ 在二维中进一步说明对任意肥对象（非常数复杂度）对应的问题同样无子指数算法。

**🔧 技术方法**

主要技术手段包括：
- 团分离器（Clique Separator）与秩基方法（Rank‑based）相结合的分治框架；
- 对 (3,3)-SAT 进行预处理后构造变量、子句与文字 gadget 的图结构；
- 通过几何构造（利用两条不共线直线上的球、以及圆柱坐标等）将得到的图映射为球图或单位球图；
- 使用 ETH 假设推导 2^o(n) 下界。

**📊 数据集**

该工作为纯理论研究，不使用实验数据集；所有结果均基于构造性证明与理论上限。

**📈 对比分析**

由于本文主要给出下界与理论框架，没有直接的算法实现或实验性能对比；相对以往仅在同尺寸肥对象上实现的子指数算法，本文展示了在更广泛对象集合中无法保持同样复杂度的事实。

**⚠️ 局限性**

局限性：
- 对单位圆（disk）图中的最小支配集等问题仍未给出子指数算法；
- 加权版本在二维同尺寸肥对象上的复杂度仍未知；
- 仅给出了 ETH 下的否定结果，未给出匹配的上界或近似算法；
- 所有下界均基于 (3,3)-SAT 归约，若 ETH 失效则结论不成立。

---

## 263. Design and Development of a Low-Cost Scalable GSM-IoT Smart Pet Feeder with a Remote Mobile Application

**arXiv ID:** 2601.08394 | [PDF](https://arxiv.org/pdf/2601.08394v1)

**作者:** Md. Rakibul Hasan Nishat `[一作]` (Rajshahi University of Engineering and Technology), A. S. M. Ahsanul Sarkar Akib `[通讯]` (Robo Tech Valley)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

**🎯 论文内容**

设计、开发并验证了一款低成本、可扩展的GSM‑IoT智能宠物喂食器，支持通过手机短信远程控制与监测。

**💡 创新点**

其创新点在于：1）通过GSM模块实现无网络（无Wi‑Fi）依赖的远程操作；2）低成本（约35美元）与模块化设计；3）利用超声波传感器实现食物水平实时监测；4）精准分量（±2.67%）且能在低电流（125 mA）下持续工作。

**🔧 技术方法**

采用Arduino UNO、SIM800L GSM模块、HC‑SR04超声波传感器、SG90舵机以及MIT App Inventor开发的手机应用；系统通过短信（SMS）协议实现指令下发与状态推送。

**📊 数据集**

未使用公开数据集，而是通过实验收集数据：对100条短信指令进行成功率测试；30次喂食实验测得分量一致性；多次功耗曲线测量。

**📈 对比分析**

与现有Wi‑Fi或AI驱动的宠物喂食器进行对比：成本低于传统设备（$35 vs >$100），具备98%短信命令成功率，分量一致性±2.67%，功耗保持在125 mA空闲电流。

**⚠️ 局限性**

局限性包括：依赖移动网络覆盖，缺乏太阳能或更长续航方案；仅支持单宠物或单一食物类型；未配备图像识别或重量传感器，功能受限于当前硬件。

---

## 264. Semantic Misalignment in Vision-Language Models under Perceptual Degradation

**arXiv ID:** 2601.08355 | [PDF](https://arxiv.org/pdf/2601.08355v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 265. Creativity in AI as Emergence from Domain-Limited Generative Models

**arXiv ID:** 2601.08388 | [PDF](https://arxiv.org/pdf/2601.08388v1)

**作者:** Corina Chutaux `[一作]` `[通讯]` (Sorbonne University), Corina Chutaux (Sorbonne University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于领域受限的生成系统的创造性出现框架，并在18世纪欧洲文本与图像数据集上用多模态CGAN验证其生成过程的创造性。

**💡 创新点**

创新点在于把创造性视为内部生成动力的涌现属性，用历史语境、个体世界观、模式积累与随机冲击四维分解建模，而非后置评估。

**🔧 技术方法**

采用多模态生成对抗网络（CGAN）与标准单模DCGAN，并结合文本编码与视觉生成的交叉条件。

**📊 数据集**

使用18世纪欧洲绘画与对应文学文本构成的受限语料库。

**📈 对比分析**

通过对比DCGAN与CGAN在训练进程中的输出，观察到CGAN产生更高的形式多样性和原创性，但未给出定量指标；性能上并未优化特定任务。

**⚠️ 局限性**

局限在于缺乏客观评价指标、仅限单一历史时期、并未探索跨域或身体化系统的创造性表现。

---

## 266. On the Generalization Error of Differentially Private Algorithms Via Typicality

**arXiv ID:** 2601.08386 | [PDF](https://arxiv.org/pdf/2601.08386v1)

**作者:** Yanxiao Liu `[一作]` (Imperial College London), Deniz Gündüz `[通讯]` (Imperial College London)

**通讯引用:** 18161 | [OpenAlex ID](https://openalex.org/A5016883501)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文从信息论视角出发，对不同ially私有学习算法的泛化误差进行定量分析，提出了更紧的互信息与最大泄漏上界，并据此给出期望与高概率泛化误差上界。

**💡 创新点**

创新点在于利用典型性和私有算法的稳定性，显著改进了Rodríguez‑Gálvez等人提出的互信息上界，并首次给出了可直接计算的最大泄漏上界。

**🔧 技术方法**

主要技术包括典型性（method of types）、信息理论的KL散度与Sibson信息、群组隐私与代表性集合构造，以及对离散私有算法的精细化 KL 上界。

**📊 数据集**

论文未使用具体实测数据集，而是通过两类示例场景（N=10^3,|Z|=2 与 N=10^7,|Z|=10^6）来说明理论上界的可行性。

**📈 对比分析**

与之前工作相比，实验图表显示在两种场景下，新上界在参数范围内均优于旧有互信息或最大泄漏估计，尤其在样本量大、特征空间高维时提升显著。

**⚠️ 局限性**

局限性在于仍依赖离散设定，未讨论连续空间或高斯差分隐私的情况，且上界在某些极端参数（ε>1）下未进一步优化。

---

## 267. Deconstructing Pre-training: Knowledge Attribution Analysis in MoE and Dense Models

**arXiv ID:** 2601.08383 | [PDF](https://arxiv.org/pdf/2601.08383v1)

**作者:** Bo Wang `[一作]` (Hong Kong University of Science and Technology), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1186 | [OpenAlex ID](https://openalex.org/A5057914558)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比 MoE 与密集 Transformer 在预训练期间的神经元级知识获取动态，提出 Gated‑LPI 归因方法并进行时间分辨分析。

**💡 创新点**

创新点在于：①把 Log‑Probability Increase (LPI) 方法扩展到 MoE 结构；②首次系统追踪 MoE 与密集模型在 1.2M/600K 步训练过程中的神经元与层级稳定性；③揭示 MoE 形成低熵核心、早期巩固和功能鲁棒性的机制。

**🔧 技术方法**

使用的技术包括：Gated‑LPI（归因），J_stab、R_t、ρ_avg、σ_rel 等稳定性指标，注意力与 FFN 层的子值子键归因，实验中还做了基于 HIT@10 的消融对比。

**📊 数据集**

使用的数据集为 906 条关系事实（包含语言学、常识、事实与偏见四类），训练数据来自 Dolma 与 OLMoE‑MIX（含代码/数学），用于评估 HIT@10。

**📈 对比分析**

比较方法：对同一任务在 OLMo‑7B（密集）与 OLMoE‑1B‑7B（MoE）上记录多次检查点，统计神经元/层级稳定性指标并在不同训练阶段进行消融实验，结果显示 MoE 在稳定性、早期巩固和消融鲁棒性上明显优于密集模型；MoE 的 HIT@10 下降幅度仅 9–35% 而密集模型高达 50–96%。

**⚠️ 局限性**

局限性包括：仅使用单一关系事实小数据集（906 条）；仅对两种模型（一个 MoE 与一个密集）进行对比，难以推广到更大或不同架构；训练语料分布差异（Dolma vs OLMoE‑MIX）可能混杂影响；实验未涉及多任务或更复杂推理场景。

---

## 268. Noise-Adaptive Regularization for Robust Multi-Label Remote Sensing Image Classification

**arXiv ID:** 2601.08446 | [PDF](https://arxiv.org/pdf/2601.08446v1)

**作者:** Tom Burgert `[一作]` (Berlin Institute for the Foundations of Learning and Data), Begüm Demir `[通讯]` (Berlin Institute for the Foundations of Learning and Data)

**通讯引用:** 4433 | [OpenAlex ID](https://openalex.org/A5087126293)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种适用于遥感多标签分类的噪声自适应正则化方法NAR，能够在训练过程中动态区分并处理加法与减法标签噪声；

**💡 创新点**

创新点在于首次将多标签噪声分为加法、减法与混合三类，并通过置信度驱动的三态标签处理与早期学习正则化相结合，实现了半监督的噪声自适应学习；

**🔧 技术方法**

采用的技术包括半监督学习框架、置信度阈值驱动的标签保持/抑制/翻转机制、早期学习正则化(ELR)以及标准的二进制交叉熵损失；

**📊 数据集**

实验使用了三大遥感多标签数据集：UCMerced、DeepGlobe-ML 与 AID-ML；

**📈 对比分析**

与传统BCE、ELR、SAT、ASL、BalanceMix等方法比较，NAR在减法噪声和混合噪声场景下平均提升2%~6% mAP，整体实现最佳性能；

**⚠️ 局限性**

主要局限在于需手动调节置信度阈值，并且在极高噪声或标签稀疏的场景下性能提升有限。

---

## 269. Incentivizing Cardiologist-Like Reasoning in MLLMs for Interpretable Echocardiographic Diagnosis

**arXiv ID:** 2601.08440 | [PDF](https://arxiv.org/pdf/2601.08440v1)

**作者:** Yi Qin `[一作]` (Hong Kong University of Science and Technology), Xiaomeng Li `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 8895 | [OpenAlex ID](https://openalex.org/A5100427643)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出了心脏推理模板（CRT）和 CardiacMind 强化学习框架，用于在多视图、跨模态心脏超声图像中进行更可靠的诊断推理。

**💡 创新点**

创新点在于：①设计了只包含15种复杂心脏疾病的高层次、标准化的推理模板；②构建了三种新型奖励（数量、质量、语义）鼓励模型生成与模板一致、具备可视化对齐的逐步推理；③通过模板指导推理修正（TRR）实现推理路径的可扩展性和纠错。

**🔧 技术方法**

采用的技术包括：多模态大型语言模型 Qwen‑2.5‑VL‑7B，基于 Group Relative Policy Optimization 的强化学习；使用 LLM‑as‑Judge 以及 CLIP‑style 视频-文本相似度评估；并结合模板检索与推理修正。

**📊 数据集**

主要数据集为作者自建的 EchoComplex，包含 1,486 名训练病例、623 名测试病例，覆盖 15 复杂疾病；此外在 CardiacNet‑ASD 与 CardiacNet‑PAH 单视图基准上进行评估。

**📈 对比分析**

与心脏超声基线模型（PanEcho、EchoPrime）、一般 MLLM（Qwen、MedVLM‑R1 等）以及专门的医学推理 MLLM（Med‑VLM‑R1、Med‑RwR 等）相比，CardiacMind 在 EchoComplex 上实现了 48% 的准确率提升（0.83 vs 0.56），F1 也从 0.69 提升到 0.81；在 CardiacNet‑PAH 上提升了 5% 的准确率；推理质量评分最高，临床医生对 93% 的推理路径给与“与心脏科医生逻辑一致”的评价。

**⚠️ 局限性**

局限性包括：①仅针对 15 种疾病，缺乏更广泛病种覆盖；②模型训练与评估依赖大量标注数据，仍需专业心脏科医生投入；③当前实现依赖 7B 级模型，计算资源消耗仍较大；④在某些罕见视图或测量缺失时，模板检索和奖励可能无法完全匹配。

---

## 270. SPARK: Scalable Real-Time Point Cloud Aggregation with Multi-View Self-Calibration

**arXiv ID:** 2601.08414 | [PDF](https://arxiv.org/pdf/2601.08414v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 271. Forcing and Interpolation in First-Order Hybrid Logic with rigid symbols

**arXiv ID:** 2601.08432 | [PDF](https://arxiv.org/pdf/2601.08432v1)

**作者:** Daniel Găină `[一作]` (Kyushu University), Go Hashimoto `[通讯]` (Kyushu University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

在多排序一阶混合逻辑中，构造了 Craig 插值性质的类比，证明了其在含有可变域（甚至可能为空域）的模型上的成立。

**💡 创新点**

核心创新是提出一种动态添加新常量的 forcing 技术，能够在保持一致性的同时为模型生成必要的符号，从而实现对多排序混合逻辑的直接插值证明。

**🔧 技术方法**

采用了机构（institution）理论、语义 forcing 属性、Henkin 常量、可达模型（reachable model）以及代换与可扩展的签名同态等工具，构建了完整的证明框架。

**📊 数据集**

本工作为纯理论研究，无使用任何数据集，全部以形式化证明为主。

**📈 对比分析**

未进行实验或性能评估，主要通过形式化证明和逻辑等价性进行验证。

**⚠️ 局限性**

局限在于：仅适用于包含可量化变量的多排序混合逻辑，缺少对无可能世界量化的扩展；对空域模型的处理仍存在一定限制；插值性质的适用条件要求签名同态满足“保护灵活符号”的严格条件。

---

## 272. RubricHub: A Comprehensive and Highly Discriminative Rubric Dataset via Automated Coarse-to-Fine Generation

**arXiv ID:** 2601.08430 | [PDF](https://arxiv.org/pdf/2601.08430v1)

**作者:** Sunzhu Li `[一作]` (Li Auto Inc.), Wei Chen `[通讯]` (Li Auto Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种三阶段的自动化 Coarse-to-Fine Rubric Generation 框架，并基于此构建了 110k 条多领域评估规则集 RubricHub，随后利用 RubricHub 进行两阶段后训练（Rubric-based Rejection Sampling Fine‑Tuning RuFT 与 Rubric-based Reinforcement Learning RuRL）

**💡 创新点**

创新点在于①将原则引导与答案定位相结合生成初始评估准则；②采用多模型聚合消除单一模型偏差；③通过难度进化机制将基准准则提升为高辨别力的细粒度标准；④将细粒度 Rubric 用作奖励信号，显著提升 LLM 的对齐效果

**🔧 技术方法**

使用的技术包括：LLM 生成与聚合提示（多前沿模型如 GPT‑5.1、Gemini 3 Pro 等）、难度进化提示、Rubric‑based Rejection Sampling 进行数据筛选、基于 DAPO 的强化学习、以及 gpt‑oss‑120B 等大模型作为评判者

**📊 数据集**

主要使用的数据集为自建的 RubricHub（≈110k 问答–评估规则对，涵盖科学、指令跟随、写作、医疗与聊天五大域）以及多项基准评测集：HealthBench、LLMEval‑Med、IFEval、IFBench、WritingBench、ResearchQA、GPQA‑Diamond、ArenaHard V2 等

**📈 对比分析**

与专有模型（如 GPT‑5、Gemini 3 Pro）及开源 Rubric‑based 基线（Rubicon‑Preview、Baichuan‑M2 等）进行对比；RuFT→RuRL 方案在 Qwen3‑14B 上实现了显著提升，HealthBench 得分 69.3 超越 GPT‑5（67.2），在其他领域亦多次击败同级或更大规模的商业模型

**⚠️ 局限性**

局限性包括：①RubricHub 仅覆盖非可验证任务，缺乏对数学、编码等纯可验证任务的系统支持；②评判者需要大模型，导致评估可靠性受限且成本高；③RuRL 阶段计算开销大，推理延迟显著，需要进一步的架构优化

---

## 273. Coverage Improvement and Fast Convergence of On-policy Preference Learning

**arXiv ID:** 2601.08421 | [PDF](https://arxiv.org/pdf/2601.08421v1)

**作者:** Juno Kim `[一作]` (University of California Berkeley), Kwang-Sung Jun `[通讯]` (University of Arizona)

**通讯引用:** 738 | [OpenAlex ID](https://openalex.org/A5046896406)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究并证明了在线基于偏好学习（如在线DPO）相较于离线方法在样本效率和收敛速度上的显著优势，并提出了两种改进方案：基于G‑最优设计的混合采样器和基于奖励蒸馏的在线方法。

**💡 创新点**

创新点在于：①提出并严格证明“覆盖改进原则”，解释在线更新如何快速提升采样覆盖率并实现线性收敛；②设计了利用G‑最优实验设计的两步在线DPO，使得收敛不再受覆盖度影响；③引入基于奖励差值的“无噪声”蒸馏框架，获得理论上更快的O(1/n)收敛率。

**🔧 技术方法**

主要技术包括：线性softmax策略与Bradley‑Terry偏好模型的理论分析；覆盖度（variance‑和MAD‑基）与凸优化的结合；G‑最优实验设计与Kiefer‑Wolfowitz等价性；奖励蒸馏的相对回归与偏差校准。

**📊 数据集**

实验使用了 TL;DR 摘要数据集（Pythia‑1.4B SFT 及 Pythia‑6.9B 奖励模型）和 UltraFeedback 对话数据集（LLaMA‑3‑8B‑Instruct + ArmoRM‑Llama3‑8B‑v0.1 奖励模型），并在公开评测集 AlpacaEval 2.0、MMLU、GSM8K、ARC、Winogrande、TruthfulQA、HellaSwag 等上验证。

**📈 对比分析**

通过对比离线 DPO、在线 DPO、REBEL、PDFR、RDRC 等方法，实验显示在线 DPO 与其蒸馏版本在 win‑rate、奖励分数和多项基准上均呈现单调提升，显著优于对应的离线版本，证明了覆盖改进与无噪声蒸馏的理论预期。

**⚠️ 局限性**

局限性包括：在线 DPO 仍需满足足够批量大小与覆盖阈值；G‑最优设计在大规模特征空间下求解成本较高；奖励蒸馏依赖奖励模型的可实现性与误差；理论主要在线性 softmax 或可实现假设下成立，非线性或非可实现情形需进一步研究。

---

## 274. WebTrap Park: An Automated Platform for Systematic Security Evaluation of Web Agents

**arXiv ID:** 2601.08406 | [PDF](https://arxiv.org/pdf/2601.08406v1)

**作者:** Xinyi Wu `[一作]` (Fudan University), Min Yang `[通讯]` (Fudan University)

**通讯引用:** 68830 | [OpenAlex ID](https://openalex.org/A5100694840)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了WebTrap Park平台，自动在真实网页环境中评估Web Agent的安全性；

**💡 创新点**

创新点在于：①无需改动Agent代码；②基于Agent实际交互行为的结果导向评估；③构建可扩展的1,226项风险任务库，涵盖恶意提示、提示注入与误导网站；

**🔧 技术方法**

技术包括：Web层面事件拦截（click、type），Docker+Kubernetes容器化和调度，后端评估脚本自动判定攻击成功率；

**📊 数据集**

数据集为融合并细化的现有安全数据集，按攻击场景、目标与策略分层，形成统一的评测任务集合；

**📈 对比分析**

通过对比不同Agent框架（Browser Use、Skyvern‑AI、Agent‑E、SeeAct）与不同VLM/LLM模型（GPT‑4o、Claude‑4‑Sonnet、o3、Llama‑3.3‑70B‑Instruct、DeepSeek‑V3、Qwen2.5‑72B‑Instruct）的攻击成功率，得出最高80.08%（最佳组合）至最低51.51%的性能差异，体现框架和模型对安全性的显著影响；

**⚠️ 局限性**

局限性包括：评测仅覆盖当前三大风险类别；仍需在真实网络环境中部署，可能受外部因素干扰；数据集数量有限，未来需扩展更多攻击类型与场景。

---

## 275. Controlled LLM Training on Spectral Sphere

**arXiv ID:** 2601.08393 | [PDF](https://arxiv.org/pdf/2601.08393v1)

**作者:** Tian Xie `[一作]` (Microsoft Research Asia), Baining Guo `[通讯]` (Microsoft Research Asia)

**通讯引用:** 45256 | [OpenAlex ID](https://openalex.org/A5101666011)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Spectral Sphere Optimizer（SSO），在权重与更新上同时满足最大更新参数化（μP）的谱约束，实现严格的谱球几何优化；

**💡 创新点**

创新点在于将最速下降与谱球约束结合，利用切线空间的Lagrange乘子搜索得到最优更新方向，并通过重traction保持权重在谱球上，构成了首个完整的μP‑对齐优化器；

**🔧 技术方法**

使用谱正则化、矩阵符号运算（Newton–Schulz迭代）、根求解器（bracketing‑bisection）、分布式原子模块分片、负载平衡与自适应核调度等技术；

**📊 数据集**

采用大规模混合语言数据集OLMo‑Mix‑1124（约100B token）进行预训练；

**📈 对比分析**

与AdamW、Muon以及Muonsphere比较，在Dense‑1.7B、MoE‑8B‑A1B和DeepNet‑200层上，SSO在验证损失、MoE路由负载平衡、激活稳定性方面均优于基线，收敛速度提升≈19%；

**⚠️ 局限性**

主要限制是根求解器和矩阵符号计算的CPU‑GPU同步开销、运行时不平衡，且对低精度训练的适配仍需进一步优化。

---

## 276. Source-Free Domain Adaptation for Geospatial Point Cloud Semantic Segmentation

**arXiv ID:** 2601.08375 | [PDF](https://arxiv.org/pdf/2601.08375v1)

**作者:** Yuan Gao `[一作]` (Aerospace Information Research Institute), Cheng Wang `[通讯]` (Aerospace Information Research Institute)

**通讯引用:** 14783 | [OpenAlex ID](https://openalex.org/A5100416961)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种针对源端数据不可访问的源自由无监督域适配框架LoGo，用于地理空间三维点云的语义分割。

**💡 创新点**

创新点包括：① 类平衡的局部原型估计模块通过每类高置信锚点挖掘缓解长尾分布导致的特征坍塌；② 全局分布对齐模块利用最优传输构造全局匹配，纠正局部贪婪分配的主类偏差；③ 本地-全局双共识筛选机制通过交叉验证局部预测与全局最优分配，抑制伪标签噪声。

**🔧 技术方法**

技术主要有：参数高效的Mean‑Teacher架构（冻结大部分权重，只更新BN层）；多增强投影（V=4）得到稳健概率；类平衡原型聚合（anchor mining）；基于Sinkhorn的最优传输全局分布匹配；交叉一致性筛选与交叉熵自训练。

**📊 数据集**

使用的公开基准包括：跨模态点云（STPLS3D→H3D）和跨视角点云（DALES→T3D），分别覆盖光学摄影测量、UAV激光雷达、ALS与MLS等不同传感器。

**📈 对比分析**

在两组任务中，LoGo在mIoU与OA上均优于现有SOTA方法。具体：STPLS3D→H3D mIoU提升至54.75%（+5.9%相较源模型），DALES→T3D mIoU提升至73.54%（+11.42%）。与AdaBN、TENT、SHOT、TTYD等方法相比，LoGo显著降低误分类并提升稀疏类别性能。

**⚠️ 局限性**

局限性：① 仅在源自由、无标签的场景下验证，未考虑目标数据中出现新类别或显著类别分布漂移；② 采用固定比例anchor抽样（ρ），在极端稀疏类别时仍可能受噪声影响；③ 计算量受多增强与Sinkhorn迭代影响，需进一步优化效率。

---

## 277. Shifting the Sweet Spot: High-Performance Matrix-Free Method for High-Order Elasticity

**arXiv ID:** 2601.08374 | [PDF](https://arxiv.org/pdf/2601.08374v1)

**作者:** Dali Chang `[一作]` (Dalian University of Technology), Weiqiang Kong `[通讯]` (Dalian University of Technology)

**通讯引用:** 3292 | [OpenAlex ID](https://openalex.org/A5085374981)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

在MFEM框架中实现了高阶弹性问题的矩阵无关(PA)算子，并将其与几何多重网格(GMG)预处理器深度集成，形成端到端的高性能求解器。

**💡 创新点**

创新点包括：① 用张量分解实现O(p⁴)的求逆算法替代原O(p⁶)通用实现；② 结合Voigt对称性消除冗余计算；③ 采用宏核融合消除内存往返；④ 采用切片级循环结构提升缓存局部性；⑤ 将上述多级优化与GMG预处理器整合，实现性能“sweet spot”从低阶p≈2迁移至高阶p≥6。

**🔧 技术方法**

技术手段包括：张量乘子分解(sum‑factorization)、Voigt符号简化、宏核融合、切片式循环优化、几何多重网格(含Chebyshev平滑器)、以及对比评测使用的ARM Kunpeng和AMD EPYC多核CPU。

**📊 数据集**

使用结构化六面体网格的3D线性弹性基准问题，阶数p从1到8，实验规模从约830k自由度到超过5千万自由度。

**📈 对比分析**

对比方法：在相同GMG预处理器下比较传统全组装(FA)、未优化的PA以及优化后的PAop。评估指标为总求解时间、峰值内存占用、求解吞吐量(MDof/s)与算力(GFLOPS)。实验表明PAop相对于PA可实现最高17×的总时间加速，最高2722×相对于FA；内存使用显著降低，FA在高阶时出现OOM；并成功将性能高峰迁移至高阶p≥6。

**⚠️ 局限性**

局限性：仍受内存带宽限制，主要针对线性弹性问题；对更复杂物理模型（非线性固体、流体、电磁）需进一步验证；在极端规模下，GMG的粗层求解、通信与其他系统级瓶颈成为新的性能壁垒。

---

## 278. Geo-NVS-w: Geometry-Aware Novel View Synthesis In-the-Wild with an SDF Renderer

**arXiv ID:** 2601.08371 | [PDF](https://arxiv.org/pdf/2601.08371v1)

**作者:** Anastasios Tsalakopoulos `[一作]` (Centre for Research and Technology Hellas), Dimitrios Zarpalas `[通讯]` (Centre for Research and Technology Hellas)

**通讯引用:** 2751 | [OpenAlex ID](https://openalex.org/A5059708050)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Geo-NVS-w 框架，利用 Signed Distance Function (SDF) 与八叉树特征体积实现高保真、几何一致的原始图像合成，特别适用于无结构、野生场景。

**💡 创新点**

创新点包括：① Octree‑Accelerated feature‑based volume 让渲染聚焦在几何区块；② Geometry‑Preservation Loss (GPL) 强制保持重要几何细节；③ 对能耗进行量化评估，证明显著节能；④ 将 SDF 作为渲染指导与动态遮挡分离相结合。

**🔧 技术方法**

使用的技术：SDF 表示、NeuS 渲染公式、八叉树加速结构、CNN 生成 transient mask、Eikonal 与曲率损失、混合精度训练、GPU 能耗监测等。

**📊 数据集**

使用的数据集为 IMC‑Phototourism (IMC‑PT) 数据集中的四个场景：Sacré‑Cœur、Trevi Fountain、Brandenburg Gate 和 Taj Mahal。

**📈 对比分析**

与 Ha‑NeRF、NeRF‑W 进行对比：PSNR/SSIM/LPIPS 通常更高，能耗仅 2.05 kWh（相比 NeRF‑W 9.7 kWh、Ha‑NeRF 7.71 kWh），训练速度更快、质量更好。

**⚠️ 局限性**

局限性：仅在四个 IMC‑PT 场景验证；未与最新 Gaussian Splatting 等更快方法直接对比；对极端动态遮挡或更大规模场景的鲁棒性仍待进一步验证。

---

## 279. PosIR: Position-Aware Heterogeneous Information Retrieval Benchmark

**arXiv ID:** 2601.08363 | [PDF](https://arxiv.org/pdf/2601.08363v1)

**作者:** Ziyang Zeng `[一作]` (Beijing University of Posts and Telecommunications), Yuqing Yang `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 2062 | [OpenAlex ID](https://openalex.org/A5012503019)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了名为PosIR的检索基准，用于诊断检索模型对信息位置的偏倚，结合了细粒度的引用段、长度多样性和多语言覆盖；

**💡 创新点**

创新点在于：①通过LLM生成并严格验证位置感知的查询-文档对，彻底分离文档长度与信息位置的影响；②构建310个跨10种语言、31个领域的数据集，显著扩展了评测范围；③引入PSI指标和梯度显著性分析，揭示模型内部的首位/尾部偏倚机制；

**🔧 技术方法**

技术手段包括：LLM（DeepSeek-V3.1、Qwen3-30B）生成查询与引用段；对比重排序模型（bge-reranker、Qwen3-Reranker）做语义验证；翻译与后处理保证多语言一致性；梯度显著性评估模型对不同位置词的敏感度；PSI与nDCG@10评测位置敏感度；

**📊 数据集**

使用的数据集为310个检索集合，涵盖10种语言（包括英语、汉语、阿拉伯语、德语、西班牙语、法语、意大利语、韩语、葡萄牙语、俄语）和31个行业领域，文档来源为IndustryCorpus2与FineWeb；

**📈 对比分析**

通过与MMTEB基准对比，计算Spearman相关系数和nDCG@10，发现MMTEB与PosIR在长文本检索上相关性显著下降；使用PSI衡量位置偏倚，实验证明大多数模型在长文本中表现出首位偏倚，少数模型出现尾部偏倚；整体性能显示在长文本场景下模型普遍退化，揭示现有评测的局限；

**⚠️ 局限性**

局限性包括：①LLM生成的引用段可能遗漏或错误，导致噪声；②多语言翻译仍可能存在语义漂移，尤其是低资源语言；③仅评测基于嵌入的密集检索模型，未覆盖交叉编码器或生成式检索；④仅限文本检索，未探讨多模态检索中的位置偏倚。

---

## 280. Scalable Sequential Recommendation under Latency and Memory Constraints

**arXiv ID:** 2601.08360 | [PDF](https://arxiv.org/pdf/2601.08360v1)

**作者:** Adithya Parthasarathy `[一作]` (Institute of Electrical and Electronics Engineers), Sumit Saha `[通讯]` (East West Bank)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 HoloMambaRec，一种融合光学绑定与选择性状态空间编码的轻量级序列推荐模型，能够在低延迟和内存约束下捕获长程用户行为。

**💡 创新点**

创新点在于将项与离散属性通过圆形卷积实现光学绑定，同时采用线性时间的 Mamba‑style 选择性状态空间网络，兼顾高效性与表达能力。

**🔧 技术方法**

技术手段包括：光学（环形卷积）绑定、FFT 加速的嵌入运算、Mamba 选择性状态空间编码、线性时间扫描、Mask 训练与评估。

**📊 数据集**

实验使用 Amazon Beauty 与 MovieLens‑1M 两大公开数据集，采用统一的前处理和留一评估策略。

**📈 对比分析**

与 SASRec（Transformer）和 GRU4Rec（RNN）进行对比，10 轮训练后 HoloMambaRec 在两数据集均显著优于 SASRec，并在 MovieLens‑1M 上超过 GRU4Rec，且显著降低了内存占用。

**⚠️ 局限性**

局限在于目前仅绑定单一离散属性、未验证多属性或多值元数据的效果、未对时间压缩（窗口绑定）做联合训练，以及实现仍以 Python 为主，缺乏高效 CUDA/Kernels。

---

## 281. Decodable but not structured: linear probing enables Underwater Acoustic Target Recognition with pretrained audio embeddings

**arXiv ID:** 2601.08358 | [PDF](https://arxiv.org/pdf/2601.08358v1)

**作者:** Hilde I. Hummel `[一作]` (Centre of Mathematics and Computer Science), Burooj Ghani `[通讯]` (Naturalis Biodiversity Center)

**通讯引用:** 201 | [OpenAlex ID](https://openalex.org/A5075427113)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究对多种预训练音频模型在船舶声学目标识别（UATR）任务中的迁移学习性能进行了系统比较，利用冻结权重后提取嵌入向量，分别通过线性探测、聚类与相似度评估验证其泛化能力。

**💡 创新点**

创新点在于首次将来自不同音频域（通用音频、语音、生物声学、海洋生物声学）的预训练模型系统评估于UATR任务，并通过嵌入空间分析揭示记录ID信息主导嵌入结构但线性探测仍能提取船型特征。

**🔧 技术方法**

采用了自监督学习模型（AudioMAE、BEATS、HuBERT AS等）、监督学习模型（Wav2Vec2.0、BirdNet等）以及Transformer/ResNet等架构，冻结模型后提取嵌入，并使用逻辑回归、K‑means、余弦相似度等技术进行评估。

**📊 数据集**

预训练数据包括AudioSet、FDSK50、VGGSound、Librispeech、Xeno‑Canto、BirdSet、iNatSounds、MeerKAT、ReefSet、NOAA等；下游评估数据为Deepship（45小时、4类）与ShipsEar（8小时、5类）两个船舶声学基准集。

**📈 对比分析**

比较方法包括：NMI聚类评估、ROC‑AUC相似度评估以及线性探测的分类准确率；结果显示线性探测在两组数据上均显著优于基线，最佳表现来自BEATS模型（Deepship 65.4%、ShipsEar 74.0%），其余模型普遍落后于BEATS。

**⚠️ 局限性**

局限性在于嵌入空间被记录ID信息占据，未能充分捕捉船型语义；对海洋生物预训练模型的迁移效果差，表明跨域泛化受限；实验依赖于有限标注数据，未探索更复杂的微调或域自适应方法。

---

## 282. Movable Antenna for Integrating Near-field Channel Estimation and Localization

**arXiv ID:** 2601.08357 | [PDF](https://arxiv.org/pdf/2601.08357v1)

**作者:** Chongjia Sun `[一作]` (Beijing Institute of Technology), Rui Zhang `[通讯]` (National University of Singapore)

**通讯引用:** 100573 | [OpenAlex ID](https://openalex.org/A5100422102)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出了基于可移动天线的近场ISAC框架，利用子区域角度估计与散射体定位实现完整信道重构。

**💡 创新点**

创新点在于将可移动天线区域划分为若干子区域，采用Newtonized OMP获取高精度角度，再通过子区域光线聚类（LSRC）定位散射体，并利用定位结果优化信道估计。

**🔧 技术方法**

主要技术包括Newtonized Orthogonal Matching Pursuit、最小二乘光线聚类、稀疏重建的多测量向量（MMV）结构以及近场方向向量字典。

**📊 数据集**

实验数据来源于仿真生成的近场多径模型（6条随机散射体、随机角度和延迟），未使用公开数据集。

**📈 对比分析**

与基于OMP的2D/3D字典和传统OMP-LSRC方法对比，实验显示在高SNR下NMSE显著下降，角度和距离的平均误差大幅降低，且在端口压缩比和子载波压缩比上均表现更优。

**⚠️ 局限性**

局限性包括对子区域划分与端口测量数量的依赖，需满足散射体数量可辨识、路径可分辨率以及可移动天线的运动精度和机械约束。

---

## 283. Keyframe-based Dense Mapping with the Graph of View-Dependent Local Maps

**arXiv ID:** 2601.08520 | [PDF](https://arxiv.org/pdf/2601.08520v1)

**作者:** Krzysztof Zielinski `[一作]`, Dominik Belter `[通讯]` (Poznan University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于关键帧的视角依赖稠密映射系统，利用RGB‑D传感器更新局部NDT地图，并通过位姿图实现全局地图修正与循环闭合后校正。

**💡 创新点**

创新点在于：1) 将三维空间划分改为二维图像平面，使用椭球体表示局部占用信息，随距离自适应分辨率；2) 在局部地图中直接维护椭球体信息，支持高精度靠近相机的细节；3) 通过视角依赖的局部地图与位姿图结合，实现全局地图的快速合并与遮挡过滤。

**🔧 技术方法**

技术手段包括：RGB‑D深度图投影、NDT更新方法、椭球体统计（均值、协方差、颜色）、基于协视图的关键帧图、mean‑shift聚类、前向/后向投影投射、遮挡检测与过滤。

**📊 数据集**

主要使用TUM RGB‑D数据集（freiburg1/2/3 系列）以及ICL‑NUIM living_room 数据集进行实验验证。

**📈 对比分析**

与 OctoMap（基于体素）和 NDT‑OM（基于体素+NDT）进行对比。结果显示：在相同精度下，视角依赖方法显著降低更新时间与体素/椭球数；在重建误差（RMSE）方面，比 OctoMap 更小，且在细节保留上优于 NDT‑OM；在循环闭合后全局修正时可保持局部高精度。

**⚠️ 局限性**

局限性包括：1) 需要已知或准确的位姿输入；2) 对遮挡和多视角重叠的处理仍较为简化，可能在极端场景下产生误差；3) 目前只针对静态环境，动态物体尚未考虑；4) 计算量随局部地图数量和椭球数增加，仍有提升空间。

---

## 284. STAR: Detecting Inference-time Backdoors in LLM Reasoning via State-Transition Amplification Ratio

**arXiv ID:** 2601.08511 | [PDF](https://arxiv.org/pdf/2601.08511v1)

**作者:** Seong-Gyu Park `[一作]` (Soongsil University), Daeseon Choi `[通讯]` (Soongsil University)

**通讯引用:** 1091 | [OpenAlex ID](https://openalex.org/A5053797473)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为STAR（State‑Transition Amplification Ratio）的检测框架，用以识别在LLM推理过程中注入的隐蔽反向门攻击；

**💡 创新点**

创新点在于将模型先验概率与触发条件概率的放大比率与CUSUM累计统计相结合，从单一次后期概率检验即可发现恶意推理路径；

**🔧 技术方法**

采用概率比率、对数似然差、CUSUM算法以及LLM生成时的token概率（logit）进行后期检测；

**📊 数据集**

使用的基准数据集包括GSM8K、ASDiv、CSQA、StrategyQA与Letter五个推理任务；

**📈 对比分析**

与Onion（perplexity）和CoS（一致性）等基线对比，STAR在8B-70B模型、五个数据集上AUROC≈1.0、F1>0.9、R@5=1.0，且检测速度约为基线的42倍；

**⚠️ 局限性**

局限性：需要模型输出完整或top‑k logits；仅验证推理时反向门，训练时反向门尚未实验；对极少情况可能需要调参，且对商用闭源LLM的可用性受限。

---

## 285. An IoT-Enabled Smart Aquarium System for Real-Time Water Quality Monitoring and Automated Feeding

**arXiv ID:** 2601.08484 | [PDF](https://arxiv.org/pdf/2601.08484v1)

**作者:** MD Fatin Ishraque Ayon `[一作]` (Daffodil International University), A. S. M. Ahsanul Sarkar Akib `[通讯]` (Robo Tech Valley)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了基于ESP32的IoT智能水族箱系统，实现多参数实时监测、自动喂食和供氧，并通过Blynk云平台实现远程监控与报警。

**💡 创新点**

引入冷却算法防止报警泛滥，结合多种传感器（pH、TDS、温度、浊度、溶氧）与执行器，低成本硬件实现全自动化，并在10L水族箱中验证高准确率和低延迟。

**🔧 技术方法**

ESP32 MCU + DHT11、DS18B20、HC‑SR04、SEN0161、SEN0244、SEN0189 传感器 + SG90舵机 + 继电器泵 + Blynk IoT 平台 + MQTT/HTTP + NTP + Arduino IDE + 本地LCD + Web Dashboard 等技术。

**📊 数据集**

实测数据在10L鱼缸实验中收集的实时传感器读数与人工校准值（pH计、温度计、ISO浊度计等）作为基准数据集。

**📈 对比分析**

通过200次实验测量传感器准确率（pH 91%、TDS 88%、浊度 92.5%、温度 96.5%）与响应时间（≤2.1 s），与人工监测及现有商业方案对比，表现出优于传统手工、低成本、实时响应且可靠性>97%。

**⚠️ 局限性**

依赖Wi‑Fi网络导致恢复延迟、泵与舵机偶发卡塞、TDS和pH在高值下精度下降、仅在小规模水族箱验证、缺乏离线/LoRa等低功耗通信、缺乏机器学习预测与图像识别等功能。

---

## 286. DiffMM: Efficient Method for Accurate Noisy and Sparse Trajectory Map Matching via One Step Diffusion

**arXiv ID:** 2601.08482 | [PDF](https://arxiv.org/pdf/2601.08482v1)

**作者:** Chenxu Han `[一作]` (East China Normal University), Jilin Hu `[通讯]` (East China Normal University)

**通讯引用:** 1035 | [OpenAlex ID](https://openalex.org/A5020559625)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出DiffMM框架，利用编码器将轨迹与候选道路段嵌入共享潜在空间，并通过单步扩散（shortcut）模型完成地图匹配。

**💡 创新点**

创新点在于：①将道路网络信息与轨迹信息联合编码；②首次将扩散模型与shortcut机制应用于地图匹配；③通过单步去噪实现高效推理。

**🔧 技术方法**

使用技术包括：Transformer编码器、注意力融合候选段、DiT块的shortcut扩散模型、交叉熵+自一致性损失、基于高斯噪声的单步扩散。

**📊 数据集**

使用的数据集为葡萄牙波尔图和中国北京出租车轨迹数据集，路网来源于OpenStreetMap。

**📈 对比分析**

与HMM、DeepMM、GraphMM、RNTrajRec四种基线比较，DiffMM在稀疏轨迹下准确率提升15%以上，推理速度比HMM快约17倍，训练时间与最快基线相当。

**⚠️ 局限性**

局限性包括：只能输出单步匹配结果，无法直接生成完整轨迹；对极端稀疏或大规模路网的鲁棒性尚未彻底验证；需预先构建候选段并进行距离搜索。

---

## 287. Baiting AI: Deceptive Adversary Against AI-Protected Industrial Infrastructures

**arXiv ID:** 2601.08481 | [PDF](https://arxiv.org/pdf/2601.08481v1)

**作者:** Aryan Pasikhani `[一作]` (University of Sheffield), Biplab Sikdar `[通讯]` (National University of Singapore)

**通讯引用:** 11746 | [OpenAlex ID](https://openalex.org/A5041189303)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了基于多智能体深度强化学习的低慢隐蔽攻击，针对工业控制系统（ICS）中的液体混合过程进行磨损攻击，能够在不被AI驱动的入侵检测系统识别的前提下，持续降低设备寿命与产品质量。

**💡 创新点**

创新点包括：①将攻击任务拆分为调度器（决定攻击时机）和扰动器（执行包延迟）两大Agent，显著提升学习效率与隐蔽性；②引入低慢（Low & Slow）与突发（Smash & Grab）两种攻击策略，并通过对比展示后者在短期内易被检测、前者在长期内更具隐蔽性的差异；③将攻击模型嵌入Min‑Max对抗训练框架，用于强化DNN基线IDS的鲁棒性。

**🔧 技术方法**

使用了Advantage Actor‑Critic (A2C) 算法训练调度器与扰动器；对抗训练时加入奖励函数平衡攻击破坏与隐蔽性；对IDS使用 DenseNet、CNN、ResNet、LSTM 与 Transformer 等深度网络；实验中亦比较了 PPO、DDPG 与 SAC 等 RL 算法。

**📊 数据集**

主要数据集来自自建工业级测试平台的实时传感与控制日志，包含 3 个储液罐与 1 个混合罐的流量、压力、阀门状态等时间序列；攻击日志与正常运行日志混合生成训练与评估样本，公开可在 https://tinyurl.com/ycncu57k 获取。

**📈 对比分析**

与基线 IDS 的比较：在黑盒场景下，攻击可将 DenseNet、CNN、ResNet、LSTM、Transformer 的召回率从约 84–98% 降至 0–1%（极低）或 99.95% 的误报率；在灰盒与白盒场景下更显著。相对传统规则或统计 IDS，实验表明深度学习模型在面对低慢攻击时鲁棒性不足。对抗训练后，五种模型的精确率、召回率与 F1 分数提升 5–15% 以上，验证了 Min‑Max 训练的有效性。

**⚠️ 局限性**

局限性：①实验集中于批处理/混合流程，难以直接推广到高速实时控制（如机器人、轨道交通）；②攻击策略依赖对网络时延与 PLC 扫描周期的精准控制，实际环境中的噪声与硬件差异可能降低攻击成功率；③对抗训练需要大量标记数据与多轮迭代，部署成本高；④研究未涵盖基于硬件安全或物理隔离的防御方案，可能在真实工厂环境中失效。

---

## 288. Developing Predictive and Robust Radiomics Models for Chemotherapy Response in High-Grade Serous Ovarian Carcinoma

**arXiv ID:** 2601.08455 | [PDF](https://arxiv.org/pdf/2601.08455v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 289. Do You Understand How I Feel?: Towards Verified Empathy in Therapy Chatbots

**arXiv ID:** 2601.08477 | [PDF](https://arxiv.org/pdf/2601.08477v1)

**作者:** Francesco Dettori `[一作]` (Université Paris-Saclay), Matteo Giovanni Rossi `[通讯]` (Politecnico di Milano)

**通讯引用:** 1679 | [OpenAlex ID](https://openalex.org/A5045104020)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一个整合Transformer NLP与形式化验证的框架，用于从对话中提取情感特征、生成随机混合自动机模型并通过统计模型检查与策略合成验证并引导治疗聊天机器人的同理行为。

**💡 创新点**

创新点在于首次将情感计算与形式化验证结合，提供可追踪、可验证的同理需求规范，并利用策略合成在正式模型中实现同理行为优化。

**🔧 技术方法**

使用Fine‑tuned Transformer语言模型提取对话阶段、话语行为与情绪维度；基于Stochastic Hybrid Automaton构建疗程模型；利用UPPAAL实现统计模型检查与策略合成；并在推理阶段通过上下文学习实现响应生成。

**📊 数据集**

采用MEMO心理治疗对话语料库（包含阶段与行为标注）、IEMOCAP语音情绪语料（用于情绪维度和时长估计），并参考EPITOME支持对话数据探讨同理机制。

**📈 对比分析**

通过对模型预测的转移概率和期望情绪与MEMO测试数据进行比较，模型在主要对话模式上具有良好拟合；在三种策略（随机、数据驱动、最大情绪提升）下进行统计模型检查，情绪提升策略在满足同理需求的概率上显著高于其余两者。

**⚠️ 局限性**

局限性包括：模型仍相对简化，无法完全覆盖同理行为的多维目标；依赖现有标注数据且缺乏跨域验证；策略优化仅基于情绪提升，未能体现更细腻的同理机制；缺乏真实临床评估与人类反馈。

---

## 290. SUMMPILOT: Bridging Efficiency and Customization for Interactive Summarization System

**arXiv ID:** 2601.08475 | [PDF](https://arxiv.org/pdf/2601.08475v1)

**作者:** JungMin Yun `[一作]` (Chung-Ang University), YoungBin Kim `[通讯]` (Chung-Ang University)

**通讯引用:** 1803 | [OpenAlex ID](https://openalex.org/A5016930939)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了名为SummPilot的交互式可定制摘要系统，支持多文档输入，利用语义图、实体聚类以及用户交互式指令来生成并细化摘要；系统还集成了可解释的评估面板来检测事实错误。

**💡 创新点**

创新点在于将大型语言模型与图结构可视化、实体聚类和交互式控制相结合，使用户能够实时查看并调整摘要内容，同时提供事实一致性评估反馈，弥补了传统交互式摘要在内容关系表示和决策可追溯性方面的不足。

**🔧 技术方法**

核心技术包括GPT‑4o大型语言模型、关系抽取与语义三元组生成、实体聚类算法、语义图可视化技术、自然语言指令解析以及改造自FactScore的事实一致性评估模块。

**📊 数据集**

文中未公开具体使用的数据集，系统在多文档场景下进行了内部测试，推测可能使用公开摘要语料库（如CNN/DM、XSum）以及自建的多文档集合进行评估。

**📈 对比分析**

通过用户调研问卷，系统在易用性、满意度、错误检测等方面的平均分均在4.4–4.8（标准差≤0.8）之间，显示相较于现有交互式摘要方案更具优势；未给出与基准模型的数值对比，但用户体验指标显著提升。

**⚠️ 局限性**

局限性包括仅支持单语文本，缺乏多模态输入和多语言支持；依赖高算力的大模型导致部署成本高；评估机制虽能发现错误，但对复杂事实推理的覆盖仍有限。

---

## 291. M3-BENCH: Process-Aware Evaluation of LLM Agents Social Behaviors in Mixed-Motive Games

**arXiv ID:** 2601.08462 | [PDF](https://arxiv.org/pdf/2601.08462v1)

**作者:** Sixiong Xie `[一作]` (Peking University), Xiang Jing `[通讯]` (Peking University)

**通讯引用:** 2354 | [OpenAlex ID](https://openalex.org/A5082724488)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出M3‑Bench多阶段混合动机游戏基准，融合行为轨迹、推理过程与沟通内容三视角的过程感知评估，并以五大人格与社会交换理论生成社交行为画像。

**💡 创新点**

创新点在于将行为、思考、语言三维过程信息统一评估，构建多视角评估框架并映射至心理学模型，揭示行为与内在动机不一致的风险。

**🔧 技术方法**

使用LLM‑as‑Judge评估推理与沟通，规则统计分析行为轨迹，四层递进任务设计，基于Big Five与社会交换理论进行映射。

**📊 数据集**

基于24个递进级混合动机游戏任务，与14款主流LLM、经典规则策略以及人类参与者对弈数据。

**📈 对比分析**

与现有基准对比，利用任务完成分数与三模态一致性指标，显示封闭源旗舰模型优于开放源模型，过程评估能识别行为‑推理‑沟通不一致的模型，性能差异显著。

**⚠️ 局限性**

局限包括：任务仍为抽象模拟，LLM评判器可能存在偏差，过程评估成本高，交互协议单一，未覆盖更真实复杂交互场景。

---

## 292. A decentralized academic certificate issuance system using smart contracts on the tron network

**arXiv ID:** 2601.08513 | [PDF](https://arxiv.org/pdf/2601.08513v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 293. On Deciding Constant Runtime of Linear Loops

**arXiv ID:** 2601.08492 | [PDF](https://arxiv.org/pdf/2601.08492v1)

**作者:** Florian Frohn `[一作]` (RWTH Aachen University), Nils Lommen `[通讯]` (RWTH Aachen University)

**通讯引用:** 22 | [OpenAlex ID](https://openalex.org/A5030571586)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

研究了线性单路径循环（更新矩阵具有实特征值）的常数运行时间问题，并给出了判定算法。

**💡 创新点**

创新点在于将常数运行时间判定转化为有效性问题，利用闭式表达、根数上界、量化消元以及 Fourier‑Motzkin 变形，证明在实数/有理数域上该问题可判定；对整数域进一步限制为{-1,0,1}特征值可判定。

**🔧 技术方法**

核心技术包括：多项式-指数闭式求解、实根数上界推导、对存在量化的量化消除、最终值符号分析、Fourier‑Motzkin 变量消去，以及一参数 Presburger 算法（仅限整数特征值情况）。

**📊 数据集**

使用 Termination Problems Data Base（TPDB）中 1495 条单路径循环（去重后 1336 条），全部满足实特征值条件。

**📈 对比分析**

与 TermComp 竞赛中所有上界工具（cdd, Omega, PIPER 等）对比，自己实现的工具在 76 条循环上判定常数运行时间，平均耗时 1.71 秒；相较之下，其他工具仅识别 5-6 条，证明其性能优越。

**⚠️ 局限性**

局限性：只能处理实特征值（整数域仅限{-1,0,1}），不支持初始条件的推理，且对复特征值及更一般的整数循环尚无完整方法。

---

## 294. Closed-Loop LLM Discovery of Non-Standard Channel Priors in Vision Models

**arXiv ID:** 2601.08517 | [PDF](https://arxiv.org/pdf/2601.08517v1)

**作者:** Tolgay Atinc Uzun `[一作]` (University of Wurzburg), Radu Timofte `[通讯]` (University of Wurzburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

基于大型语言模型的闭环通道配置搜索框架：先用 AST 变异生成大量可执行网络，随后让 LLM 在性能反馈驱动下迭代生成更优的通道宽度；

**💡 创新点**

创新点在于把通道搜索转化为有条件代码生成任务，使用 AST 先行构造合法架构数据解决训练稀缺问题，并通过闭环微调让 LLM 学习领域专属的架构先验；

**🔧 技术方法**

采用 LLaMA‑33B 及 LoRA 微调的 LLM、AST 解析与结构变异、形状一致性/梯度/训练步长验证协议，以及 LEMUR 数据库进行架构管理；

**📊 数据集**

主要使用 CIFAR‑100 数据集进行验证，基线网络为 AlexNet 与 AirNet，生成的架构在同一数据集上进行单轮训练评估；

**📈 对比分析**

通过与随机搜索和传统 NAS/剪枝方法对比，验证通过率为 9.09%，单轮训练验证集最高准确率提升至 0.311，相比初始分布提升 24.1%；

**⚠️ 局限性**

限制包括：生成有效模型的成功率低、对硬件约束缺乏直接考虑、依赖单一数据集且 LLM 生成效率受限、模型可解释性和迁移性待进一步验证。

---

## 295. Robust CAPTCHA Using Audio Illusions in the Era of Large Language Models: from Evaluation to Advances

**arXiv ID:** 2601.08516 | [PDF](https://arxiv.org/pdf/2601.08516v1)

**作者:** Ziqi Ding `[一作]`, Yuekang Li `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一套评估框架并设计了基于音频幻觉的新型 CAPTCHA，旨在抵御先进的 LALM 与 ASR 攻击，同时保持对人类用户的易用性。

**💡 创新点**

创新点在于将正弦波语音幻觉与不可逆降采样相结合，形成了人类可懂但 AI 难以识别的声学特征，突破了传统视觉与语音 CAPTCHA 的安全/可访问性平衡。

**🔧 技术方法**

使用了大型音频语言模型（如 Qwen‑Audio‑Chat、SeaLLMs‑Audio‑7B、Qwen2‑Audio‑7B‑Instruct）、先进的 ASR 模型（GPT‑4o‑Transcript、GPT‑4o‑mini‑Transcript）以及自研的正弦波转换与降采样模块进行评估与防御。

**📊 数据集**

数据集主要为通过 TTS 合成并筛选的 210 条音频样本，包含七种现有 CAPTCHA 的 30 条实例以及自研 CAPTCHA 的 210 条幻觉音频，同时收集了 63 名参与者（含 36 名视障用户）的 756 次试验数据。

**📈 对比分析**

实验显示，七种现有 CAPTCHA 在 LALM/ASR 体系下绕过率均超过 40%，而自研 CAPTCHA 在所有 AI 解决方案下的绕过率为 0%，且在人类实验中第一轮成功率达 100%，显著优于传统方案。

**⚠️ 局限性**

局限性包括样本量有限、实验环境相对受控、未考虑针对正弦波语音进行微调的自适应攻击，未来需扩大用户测试、引入更复杂噪声场景并评估专门训练的模型威胁。

---

## 296. Algorithmic Stability in Infinite Dimensions: Characterizing Unconditional Convergence in Banach Spaces

**arXiv ID:** 2601.08512 | [PDF](https://arxiv.org/pdf/2601.08512v1)

**作者:** Przemysław Spyra `[一作]` `[通讯]` (AGH University of Science and Technology), Przemysław Spyra (AGH University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

本文研究了Banach空间中无条件收敛的本质，并给出了七个等价的判定条件。

**💡 创新点**

创新点在于将无条件收敛的七个等价表述统一归纳为一个完整的定理，并将其应用于梯度累积和帧展开的算法稳定性分析。

**🔧 技术方法**

使用了功能分析、拓扑学、算子理论及弱收敛性等数学工具进行严谨证明。

**📊 数据集**

本文没有使用具体数据集，全部为理论证明。

**📈 对比分析**

由于缺乏实验验证，作者仅通过理论比较说明无条件收敛在算法中的稳健性优于条件收敛。

**⚠️ 局限性**

局限在于缺乏量化估计、随机性扩展以及对非线性迭代算法的适用性分析。

---

## 297. STAGE: A Benchmark for Knowledge Graph Construction, Question Answering, and In-Script Role-Playing over Movie Screenplays

**arXiv ID:** 2601.08510 | [PDF](https://arxiv.org/pdf/2601.08510v1)

**作者:** Qiuyu Tian `[一作]` (Southeast University), Xin Zhang `[通讯]` (ZhuiWen Technology Co., Ltd.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套名为 STAGE 的统一基准，用以评估大语言模型在完整电影剧本中的叙事理解与角色生成能力。

**💡 创新点**

创新点在于将剧本视为统一的叙事世界，设计四个相互关联的任务（知识图构建、情节事件摘要、长文本问答与角色扮演），并提供共享的结构化图谱与多语言数据；同时引入多检索方案与记忆机制，系统性考察模型在一致性、抽象与推理上的综合表现。

**🔧 技术方法**

核心技术包括：基于 Qwen3 系列与 GPT‑4o 的大语言模型，GraphRAG 与 EDC 结构化抽取框架，混合 dense‑sparse 检索策略，记忆驱动的角色代理（含情节回放与事实记忆），以及基于 FActScore 的事件结构一致性评估。

**📊 数据集**

使用 150 部电影剧本（108 部英文、42 部中文）构建的 STAGE 数据集，包含清洗后的剧本文本、手工校对的知识图谱、情节事件与角色角色化注释。

**📈 对比分析**

实验结果表明：随着模型规模增大，KG 与事件摘要的 F1 与覆盖率均提升；在问答任务中，Hybrid 检索在中等规模模型上表现最佳，而 GraphRAG 在 GPT‑4o 上能进一步提升问答准确率；角色扮演中加入情节记忆与事实记忆显著提升角色一致性与叙事可信度。

**⚠️ 局限性**

主要限制包括：缺乏事件层级因果关系建模、依赖人机协同导致潜在偏差、知识图未充分表征时间演化以及语言分布不均，且尚未覆盖低资源或非标准剧本语言。

---

## 298. What If TSF: A Benchmark for Reframing Forecasting as Scenario-Guided Multimodal Forecasting

**arXiv ID:** 2601.08509 | [PDF](https://arxiv.org/pdf/2601.08509v1)

**作者:** Jinkwan Jang `[一作]` (Seoul National University), Taesup Kim `[通讯]` (Seoul National University)

**通讯引用:** 2395 | [OpenAlex ID](https://openalex.org/A5065728469)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 What If TSF (WIT) 这一多模态时间序列预测基准，用可解释的未来情景（包括反事实情景）来评估模型在给定文本上下文下的方向性预测能力；同时提供了严格的文本预处理、去标识化以及验证流程，确保数据质量。

**💡 创新点**

①聚焦未来情景而非仅靠历史文本，解决现有基准重复、噪声多、时间错位的问题；②构建了跨四个领域（政治、社会、能源、经济）的大规模样本；③引入了反事实任务，检验模型对情景指令的适应性；④采用方向性准确率作为核心评估指标，突出逻辑推理而非仅数值匹配。

**🔧 技术方法**

使用零样本 LLM 生成与人类专家交互式的情景文本；利用 LLM 进行文本过滤、去标识化和时序对齐；对模型输出进行多轮校验；评估采用 3 方向准确率（上升、保持、下降）以及短期 MSE；比较基准包含大规模 LLM（GPT‑4o、Llama‑3‑70B 等）、时间序列预训练模型（Chronos、Moirai、TimesFM）和传统统计方法（ARIMA、ETS）。

**📊 数据集**

基准样本 5,352 条，涵盖四个领域，每条样本包含历史时间序列、静态上下文 S、历史上下文 H、未来情景 F（可行/反事实）。数据来源为公开新闻、报告与权威机构，全部经过 LLM 预处理与专家审校，确保无个人敏感信息。

**📈 对比分析**

在零样本设置下，LLM 在短期预测中显著优于单模态 TSFMs 与统计基线，方向准确率提升 10%‑20%；GPT‑4o 在短期任务中表现最佳，部分较大 LLM 在长期任务中表现竞争甚至更优；TSFMs 在数值精度（MSE）上与 LLM 近似，但在方向准确率上落后，显示仅靠历史模式无法满足情景引导需求。

**⚠️ 局限性**

①未来情景由人工专家手工编写，未测试模型自生成情景的能力；②模型偶尔产生超出变量取值范围的数值或不符合约束的预测；③当前仅进行零样本评估，缺乏少样本或提示工程的探究；④对反事实任务的设计在长周期下可能受多重因素影响；⑤虽然已去标识化，但仍存在潜在记忆化风险。

---

## 299. Temporal Fusion Nexus: A task-agnostic multi-modal embedding model for clinical narratives and irregular time series in post-kidney transplant care

**arXiv ID:** 2601.08503 | [PDF](https://arxiv.org/pdf/2601.08503v1)

**作者:** Aditya Kumar `[一作]` (Hahn-Schickard), Oliver Amft `[通讯]` (University of Freiburg)

**通讯引用:** 7264 | [OpenAlex ID](https://openalex.org/A5064135418)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出 Temporal Fusion Nexus (TFN)，一种任务无关的多模态嵌入模型，能够整合不规则时间序列与临床文本，用于后肾移植护理中的风险预测。

**💡 创新点**

创新点在于：① 将稀疏性保留的 TM‑LSTM 与时间注意力相结合，①1有效捕获不规则长期动态；② 采用 Med‑GTE‑hybrid‑de 文本编码并通过跨注意力融合两模态；③ 采用自监督预训练和解耦正则化，使潜在空间解耦且可解释。

**🔧 技术方法**

使用技术包括 TM‑LSTM（时间衰减+特征级遮掩）、跨注意力融合、对比学习+去噪自编码的文本预训练、解耦与去相关正则化、SHAP 可解释性、Platt 归一化校准等。

**📊 数据集**

实验基于德国 Charité 的 NephroCAGE 3382 例后肾移植队列，包含不规则时间序列、静态特征和德语临床文本。

**📈 对比分析**

与单模态基线、时间序列+静态数据基线以及 Roller 等 CDSS 对比，TFN 在 90 天肾移植失活、排斥、死亡预测的 AUC 分别为 0.96、0.84、0.86，显著优于基线（约 10% AUC 提升）和现有 CDSS（仅 26 个特征）。

**⚠️ 局限性**

局限性包括：仅来自单中心的数据，缺乏多中心验证；文本解释性仍不够直观；对极端时间窗口（360 天）预测性能下降；模型尚未扩展至图像、组学等其他模态。

---

## 300. An Under-Explored Application for Explainable Multimodal Misogyny Detection in code-mixed Hindi-English

**arXiv ID:** 2601.08457 | [PDF](https://arxiv.org/pdf/2601.08457v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 301. GraphFusionSBR: Denoising Multi-Channel Graphs for Session-Based Recommendation

**arXiv ID:** 2601.08497 | [PDF](https://arxiv.org/pdf/2601.08497v1)

**作者:** Jia-Xin He `[一作]` (National Central University), Hung-Hsuan Chen `[通讯]` (National Central University)

**通讯引用:** 843 | [OpenAlex ID](https://openalex.org/A5078925594)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种多通道去噪的会话推荐模型GraphFusionSBR。

**💡 创新点**

通过知识图谱去噪、超图与线图联合表示，并对各通道进行去噪与互信息最大化。

**🔧 技术方法**

使用图注意力网络、超图卷积、线图卷积、对比学习以及TransR知识图谱嵌入。

**📊 数据集**

在Tmall、RetailRocket和KKBox三个公开电商/音乐数据集上进行实验。

**📈 对比分析**

与RNN、GNN、HGNN基线相比，在P@10、P@20、MRR@10、MRR@20等指标上均实现3–4%的提升。

**⚠️ 局限性**

仍受知识图谱噪声及缺乏多模态（图像、时空）上下文信息的影响，需进一步改进。

---

## 302. Surgical Refusal Ablation: Disentangling Safety from Intelligence via Concept-Guided Spectral Cleaning

**arXiv ID:** 2601.08489 | [PDF](https://arxiv.org/pdf/2601.08489v1)

**作者:** Tony Cristofano `[一作]` `[通讯]`, Tony Cristofano

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对安全对齐语言模型的拒绝行为进行编辑，通过去除干扰的拒绝向量实现无害拒绝率下降。

**💡 创新点**

提出将拒绝向量视为多义性向量，通过概念原子注册表和光谱残差化清理得到干净的拒绝方向，显著降低了编辑时对模型能力的破坏。

**🔧 技术方法**

概念原子注册表、Ridge 正则化的光谱残差化、迭代硬负样本回馈、低秩（rank‑one）权重更新、语义能量代理。

**📊 数据集**

有害/安全提示集合、概念原子对（10–15 条提示）、GSM8K（数学）、MBPP（代码）、WT2 等标准文本。

**📈 对比分析**

与传统直接去除对比，SRA 在拒绝率降至 0–2% 的同时，平均 PPL 仅上升约 +0.02，第一步 KL 往往低于 0.03（如 Qwen3‑VL‑4B 从 2.088 降至 0.044），远低于标准去除方法。

**⚠️ 局限性**

需要人工构建概念原子，可能漏掉未覆盖的干扰；评估依赖自动拒绝检测和有限的提示集；漂移指标（PPL、KL）不完全覆盖所有行为；SRA 不是完整的安全方法。

---

## 303. Degree-preserving Godel logics with an involution: intermediate logics and (ideal) paraconsistency

**arXiv ID:** 2601.08474 | [PDF](https://arxiv.org/pdf/2601.08474v1)

**作者:** M. E. Coniglio `[一作]` (University of Campinas), L. Godo `[通讯]` (Artificial Intelligence Research Institute)

**通讯引用:** 7268 | [OpenAlex ID](https://openalex.org/A5072818908)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了加了可逆否定的戈登模糊逻辑G_∼及其有限值扩展G_n∼的度保持伴随逻辑，探讨其在矛盾容忍（paraconsistency）上的性质，并对介于G_∼与经典命题逻辑CPL之间的所有中间逻辑进行分类；提出了比理想矛盾容忍（ideal paraconsistency）更弱的饱和矛盾容忍（saturated paraconsistency）概念，并证明在该框架下仅存在三个饱和矛盾容忍逻辑，其中两者为理想矛盾容忍、第三个为饱和但非理想；随后对有限值卢卡西维克逻辑进行类似分析，给出了大量非理想饱和矛盾容忍逻辑实例；并证明所有讨论的逻辑均为形式不一致逻辑（LFIs）.

**💡 创新点**

1) 提出了饱和矛盾容忍的概念，扩展了理想矛盾容忍的研究范围；2) 对G_∼与其有限值扩展的所有中间逻辑进行完备分类；3) 在有限值卢卡西维克逻辑中构造出大量非理想饱和矛盾容忍逻辑，展示该概念的丰度与实际可构造性；4) 证明这些逻辑均为LFIs，进一步连接非经典逻辑与信息不确定性研究。

**🔧 技术方法**

主要使用了逻辑矩阵语义（尤其是基于[G_∼]-代数的有序滤波器矩阵）和代数化学（如分解为子代数、直接积、判定子代数可辨性）技术；结合判定性分析与可判定性结果，对逻辑间的包含关系、最大化、饱和性等属性进行形式化证明。

**📊 数据集**

无实验数据集；研究完全基于理论证明与逻辑语义构造。

**📈 对比分析**

本研究不涉及经验比较或性能指标，评估方式为逻辑系统之间的包含关系、可扩展性与最大化性质的证明，结果表明在给定框架下只出现有限数量的饱和矛盾容忍逻辑，且理想与饱和的关系被严格阐明。

**⚠️ 局限性**

1) 仅讨论了G_∼及其有限值扩展，未涉及更一般的模糊逻辑；2) 饱和矛盾容忍的定义仍相对宽松，是否最优尚无统一评价标准；3) 对大型或无限值代数的扩展与算法实现仍未展开；4) 仅给出理论结构，缺乏与实际应用场景的直接对接。

---

## 304. QP-Based Control of an Underactuated Aerial Manipulator under Constraints

**arXiv ID:** 2601.08523 | [PDF](https://arxiv.org/pdf/2601.08523v1)

**作者:** Nesserine Laribi `[一作]`, Mehdi Benallegue `[通讯]` (Joint Robotics Laboratory National Institute of Advanced Industrial Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出一种基于二次规划的约束感知控制框架，用于无驱动空气操纵器的末端执行器轨迹跟踪。

**💡 创新点**

在二次规划中显式考虑了平台无驱动、执行器极限以及动力学约束，并在扭矩层加入被动积分动作以提升鲁棒性。

**🔧 技术方法**

二次规划、几何推力方向调节、被动积分控制、逆动力学、Simscape Multibody 仿真、RTK‑GNSS+IMU 状态估计。

**📊 数据集**

使用自建的 1.5 kg 四旋翼+5‑DOF 机械臂模型及其 CAD/STEP 数据，未使用公开数据集。

**📈 对比分析**

通过与无积分项控制和仅限于无驱动约束的对照实验比较，结果显示 RMSE 降至 10⁻³ m、偏差消失，QP 求解在 200 Hz 下平均 3 次迭代，满足实时性。

**⚠️ 局限性**

实验仅在仿真环境下验证，缺乏实际平台测试，且对接触动力学、环境约束等场景尚未处理。

---

## 305. Your Group-Relative Advantage Is Biased

**arXiv ID:** 2601.08521 | [PDF](https://arxiv.org/pdf/2601.08521v1)

**作者:** Fengkai Yang `[一作]` (Beihang University), Yikun Ban `[通讯]` (Beihang University)

**通讯引用:** 174 | [OpenAlex ID](https://openalex.org/A5047387636)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对强化学习中基于组的优势估计器的偏差进行了理论分析，并提出了历史感知自适应难度加权（HA-DW）方法，用以纠正这种偏差，从而提升大语言模型在推理任务上的表现。

**💡 创新点**

创新点包括：①首次给出组相对优势估计器系统性偏差的理论证明；②设计了基于跨批次历史信息的难度锚点与自适应加权机制（HA-DW），能够在训练中动态调节优势权重，弥补偏差；③提出可直接插入 GRPO 及其变体的通用模块。

**🔧 技术方法**

技术手段包括：组相对策略优化（GRPO/GSPO/DAPO）；优势估计与指数加权重调整；Kalman 风格的历史信念更新；动态阈值（C_t）与自适应遗忘因子；理论证明与概率分析。

**📊 数据集**

使用的数据集为五个数学推理基准：MATH500、AIME25、AMC23、Minerva 与 OlympiadBench；实验模型为 Qwen3-4B-Base、Qwen3-8B-Base 与 LLaMA‑3.2‑3B‑Instruct。

**📈 对比分析**

通过将 HA-DW 整合到 GRPO、GSPO、DAPO 三种基线算法中进行对比，实验显示无论模型规模或算法种类，HA-DW 均能显著提升各基准的准确率（如 Qwen3-4B-Base GRPO 从 75.4% 提升至 78.0%），并在难度最高的题目上实现 3–4% 的加分；同时训练奖励更高、生成文本更长，体现出更好的探索与推理能力。

**⚠️ 局限性**

局限性在于 HA-DW 仅适用于基于组的优势估计器，未覆盖更广泛的 RL 设定；偏差修正仍需在不同奖励分布下进一步验证；此外，该方法的实现与调参相对复杂，需要额外的历史记忆与动态阈值机制。

---

## 306. CD^2: Constrained Dataset Distillation for Few-Shot Class-Incremental Learning

**arXiv ID:** 2601.08519 | [PDF](https://arxiv.org/pdf/2601.08519v1)

**作者:** Kexin Bao `[一作]` (Institute of Information Engineering), Shiming Ge `[通讯]` (Institute of Information Engineering)

**通讯引用:** 3530 | [OpenAlex ID](https://openalex.org/A5033254559)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种结合数据集蒸馏（DDM）与蒸馏约束（DCM）的FSCIL框架CD^2；

**💡 创新点**

通过蒸馏产生更具代表性的合成样本并引入特征与结构保持的约束，以降低灾难性遗忘和分布漂移；

**🔧 技术方法**

采用ResNet骨干网络、三层MLP分类器，并使用MMD匹配合成与真实分布、KD实现分布对齐；

**📊 数据集**

在CIFAR‑100、mini‑ImageNet和CUB‑200这三大FSCIL基准数据集上进行评估；

**📈 对比分析**

与多种SOTA方法（如iCaRL、NC‑FSCIL、MICS等）对比，在所有数据集上均取得最高平均精度，CIFAR‑100上平均提升约1–2%，CUB‑200上仍保持领先；

**⚠️ 局限性**

对合成样本数量和K值敏感，且在复杂类别（如CUB‑200）上的收益有限，缺乏对多任务跨域迁移的探讨。

---

## 307. AUV Trajectory Learning for Underwater Acoustic Energy Transfer and Age Minimization

**arXiv ID:** 2601.08491 | [PDF](https://arxiv.org/pdf/2601.08491v1)

**作者:** Mohamed Afouene Melki `[一作]` (King Abdullah University of Science and Technology), Mohamed-Slim Alouini `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 88921 | [OpenAlex ID](https://openalex.org/A5083193286)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了利用强化学习（PPO）控制水下无人水面车辆（AUV）轨迹，以实现同时进行声波能量传输（AET）与信息上行，从而实现水下物联网（IoUT）设备的永续运行。

**💡 创新点**

创新点包括：①在轨迹规划中同时最小化信息新鲜度指标（AoI）并保证能量公平分配；②提出两种双工策略——频分双工（FDD）与时分双工（TDD），在不同硬件复杂度与性能需求下实现可调节方案；③首次将深度强化学习应用于水下环境的能量与通信协同调度问题。

**🔧 技术方法**

使用技术：深度强化学习（Proximal Policy Optimization, PPO）、多维状态空间（位置、AoI、能量）、奖励函数融合AoI、Jain公平指数与罚项、频分/时分双工模型、仿真平台。

**📊 数据集**

数据集：纯仿真数据，建立 1000 m × 1000 m × 400 m 三维网格，K=3、5、7、10 个传感节点，AUV 起始于网格中心，使用预设的声学参数与通信参数进行实验。

**📈 对比分析**

与传统基线（随机游走 Random Walk、轮询 Round Robin、贪婪算法 Greedy）相比，FDD 和 TDD 两种 RL 方法在平均 AoI、能量收集量与公平指数上均有显著提升；FDD 在性能上最优，TDD 在低复杂度、低能耗下仍保持较好效果。

**⚠️ 局限性**

局限性：仅在仿真环境中验证；未考虑 AUV 的导航能耗；单一 AUV 方案，未讨论多代理或多 AUV 场景；硬件实现与实际水下噪声、潮汐、路径约束等现实因素未纳入模型。

---

## 308. Decoding Order Matters in Autoregressive Speech Synthesis

**arXiv ID:** 2601.08450 | [PDF](https://arxiv.org/pdf/2601.08450v1)

**作者:** Minghui Zhao `[一作]` (University of Sheffield), Anton Ragni `[通讯]` (University of Sheffield)

**通讯引用:** 1167 | [OpenAlex ID](https://openalex.org/A5024283093)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了可变解码顺序对自回归语音合成的影响，并提出了基于时长预测的自适应解码策略。

**💡 创新点**

创新点包括：①使用掩码扩散模型实现任意顺序解码，证明固定左到右顺序并非最优；②提出Top‑K位置选择和时长引导解码；③展示仅用线性量化（甚至1‑bit）即可得到高质量语音。

**🔧 技术方法**

使用的技术有掩码扩散模型（MDM）、离散化量化、Top‑K自适应解码、时长预测器和HiFi‑GAN vocoder。

**📊 数据集**

实验数据集为LJSpeech，约13,100条单语者朗读短句。

**📈 对比分析**

通过与固定左/右到右、长度基础、Grad‑TTS基线在MCD、log F0、UTMOS和MOS等指标上的对比，发现自适应与时长引导解码在MCD/UTMOS上均优于固定策略，最高MOS接近参考音频。

**⚠️ 局限性**

limitations: 仍需探索最优解码调度，Top‑K更新尺寸与自然度之间的权衡；离散化时频单独采样忽略了频率内相关性；对量化温度较为敏感。

---

## 309. Degree bounds for linear differential equations and recurrences

**arXiv ID:** 2601.08522 | [PDF](https://arxiv.org/pdf/2601.08522v1)

**作者:** Louis Gaillard `[一作]` `[通讯]` (Ecole Normale Supérieure de Lyon), Louis Gaillard (Ecole Normale Supérieure de Lyon)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文提出一种统一的理论框架，用以推导多类D‑有限函数和P‑递归序列相关算法的多项式系数的度数上界；

**💡 创新点**

核心创新在于将伪Krylov系统的求解转化为对伪线性映射矩阵的McMillan度数分析，并通过构造低度数的矩阵实现在不满足严格正则性假设的情况下仍能得到最优度数上界，进而在多种具体问题（如LCLM、对称乘积、代数函数的微分解析元、组合函数与递推式的闭包、Hermite归约等）中实现了比以往更紧的度数估计；

**🔧 技术方法**

关键技术包括：1) 控制理论中的McMillan度数与矩阵分解（X M⁻¹Y）；2) 对矩阵的分子分母结构（判定子式分母、Smith‑McMillan标准形）进行估计；3) 伪线性映射的迭代表示（θ^s = p(x)_x + T或T σ_x）；4) 伪Krylov矩阵的行列式分母上界；5) 变换 μ(x) 使矩阵正则化；6) 通过最小基理论与Kronecker指数相联系获得度数上界；

**📊 数据集**

该工作属于理论研究，没有使用实验数据集；所有结果均为符号性数学证明；

**📈 对比分析**

与以往文献（如LCLM的O(s²d²)或对称乘积的O(r₁²r₂²d)等）比较，本文给出的度数上界在主项常数和阶次上均有提升，并在实验案例中验证了与之前估计的匹配或优于之；

**⚠️ 局限性**

局限性包括：1) 仍需在求解伪Krylov系统时对矩阵进行实化，计算复杂度高；2) 对输入矩阵的正则化变换 μ(x) 的构造虽然理论可行，但在实践中可能带来额外计算负担；3) 本文主要给出上界，实际算法实现与时间复杂度尚未深入探讨；4) 对非平凡多变量多项式P的特殊结构（如非素性、重根等）处理仍有限。

---

## 310. EfficientFSL: Enhancing Few-Shot Classification via Query-Only Tuning in Vision Transformers

**arXiv ID:** 2601.08499 | [PDF](https://arxiv.org/pdf/2601.08499v1)

**作者:** Wenwen Liao `[一作]` (Fudan University), Hang Ruan `[通讯]` (Fudan University)

**通讯引用:** 3565 | [OpenAlex ID](https://openalex.org/A5030927081)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种轻量级的 query‑only 微调框架 EfficientFSL，用于 ViT 的少样本分类；

**💡 创新点**

核心创新是通过冻结 ViT 并加入可训练的 Forward Block（含 Active 与 Frozen 子块）、Combine Block 及 Support‑Query Attention Block，实现仅更新极少参数的同时显著提升分类性能；

**🔧 技术方法**

技术包括可训练的 prompt 与 bottleneck 结构投影、Self‑Attention、MLP、特征聚合、投影对齐与原型位置调整（SQ Attention），并基于 Prototypical Network 进行分类；

**📊 数据集**

在四个内部基准（miniImageNet、tieredImageNet、CIFAR、FC100）和六个跨域基准（CUB、Stanford Cars、Places、Plantae、EuroSAT、CropDiseases）上验证；

**📈 对比分析**

与全模型微调、SOTA 迁移学习方法（FewTURE、MetaFormer‑A、FewVS 等）以及 PETL 基线（Adapter、AdaptFormer、LoRA）进行对比，EfficientFSL 在参数量仅约 1–2 M 的情况下，往往取得或超过其他方法的最高准确率，显著提升了训练速度、显存占用和推理速度；

**⚠️ 局限性**

限制主要体现在对超参数（如 α、ξ、ζ）的敏感性，以及在极低样本（1‑shot）或极端跨域场景下仍有提升空间，且目前只针对 ViT 结构，其他架构的推广尚未验证。

---

## 311. Modality-Decoupled RGB-Thermal Object Detector via Query Fusion

**arXiv ID:** 2601.08458 | [PDF](https://arxiv.org/pdf/2601.08458v1)

**作者:** Chao Tian `[一作]` (Harbin Institute of Technology), Zhenyu He `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 5845 | [OpenAlex ID](https://openalex.org/A5100740564)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于查询融合的模态解耦 RGB‑T 目标检测框架 MDQF，利用独立的 DETR 分支对 RGB 与 TIR 图像分别建模，并在每个解码阶段通过查询融合实现高质量查询的互相传递，从而在极端环境下平衡模态互补与分离。

**💡 创新点**

创新点包括：
- 查询融合机制，先通过 top‑k 选取置信度高的查询/框，再用轻量 MLP 适配器将一模态的高质量查询投射到另一模态的解码器；
- 保持各模态分支原始结构不变，实现在无配对数据时可单独训练；
- 分离‑到‑联合的双阶段训练策略，先单独训练再联合微调，降低对 RGB‑T 对齐数据的依赖；

**🔧 技术方法**

使用技术包括：
- 基于 DINO（DETR 变体）且采用 ViT‑tiny backbone 的单模态检测器；
- Top‑k 查询/框选择与自适配的 MLP 查询适配器；
- 交叉模态查询融合（query fusion）在每个解码层实现；
- 分离‑联合训练、NMS 后处理、损失函数组合（分类、GIoU、L1）。

**📊 数据集**

数据集：
- FLIR ADAS（640×512）
- M³FD（1024×768）
两者均按 COCO 风格评价，采用 mAP、mAP50 等指标。

**📈 对比分析**

与多种现有 RGB‑T 检测器对比，MDQF 在 FLIR 上 mAP 提升至 43.8%（比 DINO 基线 41.7% 提升 2.1%），在 M³FD 上 mAP 达 55.9%（比 DINO 53.9% 提升 2.0%）。在模态失真或单模态测试中，MDQF 仍保持较高性能，显示出更好的鲁棒性；实验也表明可通过单模态训练进一步提升。

**⚠️ 局限性**

局限性：
- 虽然查询融合仅增加少量参数，但仍需额外的适配器计算，影响推理速度；
- Top‑k 选择可能丢失低置信度但潜在有用的信息；
- 主要在公开基准上验证，缺乏对极端真实场景（如多光源、极低分辨率）的进一步评估。

---

## 312. On the Maximum Toroidal Distance Code for Lattice-Based Public-Key Cryptography

**arXiv ID:** 2601.08452 | [PDF](https://arxiv.org/pdf/2601.08452v1)

**作者:** Shuiyin Liu `[一作]` (Holmes Institute), Amin Sakzad `[通讯]` (Monash University)

**通讯引用:** 2046 | [OpenAlex ID](https://openalex.org/A5057144969)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出最大环距距离（MTD）码以提升基于格的KEM的解密可靠性

**💡 创新点**

创新点是将最小 L₂‑norm 环距最大化，利用 D4、E₈ 等最优格构造出 G‑TD 码；并证明 ℓ=2 的 MTD 码等价于优化后的 Minal 码

**🔧 技术方法**

采用格理论、离散环编码、距离优化与 E₈ 最优最近点解码技术

**📊 数据集**

实验数据基于 Kyber‑1024（q=3329）参数集，使用该参数下的噪声分布和加密流程

**📈 对比分析**

与 Minal、最大 Lee 距离码（MLD）以及原始 Kyber 编码对比，MTD/GTD 在 ℓ>2 时显著降低 DFR，且在相同 CER 下保持更高安全性

**⚠️ 局限性**

仅在维度 ℓ≤8 以内可实现，且更高维度时解码复杂度仍是实现瓶颈

---

## 313. Divide and Conquer: Static-Dynamic Collaboration for Few-Shot Class-Incremental Learning

**arXiv ID:** 2601.08448 | [PDF](https://arxiv.org/pdf/2601.08448v1)

**作者:** Kexin Bao `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Shiming Ge `[通讯]` (Institute of Information Engineering, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 Static-Dynamic Collaboration (SDC) 框架，将少样本类别增量学习(FSCIL)划分为静态保留阶段与动态学习阶段，利用静态投影器保留旧知识并结合动态投影器与外部记忆实现对新类别的高效学习。

**💡 创新点**

创新点在于：①首次将投影器拆分为静态和动态两部分，并通过调节权重 α 在保持旧知识与学习新知识之间实现可控平衡；②在增量阶段冻结 backbone，仅训练投影器，显著降低参数更新导致的灾难性遗忘；③利用外部记忆与互信息理论解释两投影器协作机制；④整体框架简单、可扩展且无需额外预训练。

**🔧 技术方法**

技术要点包括：MLP 投影器、双投影器融合、外部记忆（prototype 记忆）、交叉熵损失、Mixup/CutMix 数据增强、SGD 优化、t‑SNE 可视化与互信息分析。

**📊 数据集**

实验数据集：CIFAR‑100、MiniImageNet、CUB‑200 以及工业场景数据集 FGVC‑Aircraft。

**📈 对比分析**

与多种 SOTA 方法（iCaRL、EEIL、TOPIC、LIMIT、ALICE、NC‑FSCIL、CABD、OrCo 等）在所有四个数据集上进行比较，SDC 在每个任务和整体平均准确率上均优于对比方法，显示出更好的稳定性与可塑性。

**⚠️ 局限性**

限制与挑战：①需要手动调节 α、投影器维度和层数以获得最佳效果；②在极少样本条件下，投影器仍可能出现过拟合；③计算资源仍受投影器参数数量影响，特别是在高维投影器时显著；④模型对记忆更新策略的敏感度未深入探讨。

---

## 314. JudgeRLVR: Judge First, Generate Second for Efficient Reasoning

**arXiv ID:** 2601.08468 | [PDF](https://arxiv.org/pdf/2601.08468v1)

**作者:** Jiangshan Duo `[一作]`, Liang Zhao `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种两阶段训练框架 JudgeRLVR，即先训练模型成为判定者，再让其生成答案，以提升推理效率和质量。

**💡 创新点**

创新点在于：①首次将判定与生成分离，先让模型学习判断答案正确性；②通过判定训练获得的错误感知能力，后续生成阶段能够更早地剪枝低质量分支；③在不加长度惩罚的情况下自然降低回溯与冗长推理。

**🔧 技术方法**

技术包括：RLVR（可验证奖励强化学习）与 DAPO 的策略梯度优化；在 Qwen3‑30B‑A3B（MoE）上做监督微调（SFT）后进行两阶段 RL；使用硬负样本挖掘、类别平衡、动态采样等方法构造判定数据集。

**📊 数据集**

数据集：SFT 采用公开 CoT 数据；RLVR 训练集为 113k 个数学题；在评测时使用多种基准：AIME24/25、MATH500、HMMT_feb_2025、BeyondAIME（入域），以及 GPQA Diamond、IFEval、LiveCodeBenchv6、MMLU‑Redux、ZebraLogic（跨域）。

**📈 对比分析**

与基线（Base SFT）和 Vanilla RLVR 进行对比，指标为准确率与生成长度。JudgeRLVR 在入域数学题上平均提升 3.7 个百分点准确率，同时长度平均缩短 42%；在跨域基准上平均提升 4.5 个百分点准确率，长度略有缩短或保持不变。通过 ablation（Judge Only、Mixed Strategy）验证了分阶段训练的必要性。

**⚠️ 局限性**

局限性：①在需要格式严格或规则验证的任务（如 IFEval、ZebraLogic）中，长度可能略有增加；②对已饱和的数据集（如 MATH500）准确率提升有限；③模型对某些需要显式检查的任务仍可能产生冗长输出；④两阶段训练的超参数和阶段顺序对结果影响较大。

---

## 315. Supervised Spike Agreement Dependent Plasticity for Fast Local Learning in Spiking Neural Networks

**arXiv ID:** 2601.08526 | [PDF](https://arxiv.org/pdf/2601.08526v1)

**作者:** Gouri Lakshmi S `[一作]` (Indian Institute of Science Education and Research Thiruvananthapuram), Saptarshi Bej `[通讯]` (Indian Institute of Science Education and Research Thiruvananthapuram)

**通讯引用:** 274 | [OpenAlex ID](https://openalex.org/A5046944026)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种监督型 Spike Agreement-Dependent Plasticity（SADP）学习规则，用以实现无梯度、无教师强制的本地化、线性时复杂度的脉冲神经网络训练；

**💡 创新点**

创新点在于用群体级一致性度量（如 Cohen’s κ）替代传统的单对时序比较，既保留 STDP 的生物学可解释性，又显著提升学习速度与准确度；

**🔧 技术方法**

核心技术包括基于 Poisson 的脉冲编码、CNN 前端特征提取、可分层 SADP 更新、Kappa 相关调制和局部 Hebbian 输出层学习；

**📊 数据集**

在 MNIST、Fashion-MNIST、CIFAR-10 以及两类生物医学图像（LC25000 病理图像和脑部 MRI 肿瘤图像）上进行实验；

**📈 对比分析**

与现有 STDP、SSDP、S2-STDP 等本地学习方法以及梯度式混合 SNN 相比，Supervised SADP 在 CIFAR-10 上达 70.7% 准确率，MNIST 与 Fashion-MNIST 接近 99% 与 90%，并且训练时长显著降低；

**⚠️ 局限性**

局限性包括：深层 2SADP 与 1SADP 的性能提升有限；CNN 前端需离线预训练，缺乏完全端到端的脉冲式学习；目前仅在静态图像分类任务中验证，缺乏对序列或递归结构的适配；

---

## 316. Simplifying ROS2 controllers with a modular architecture for robot-agnostic reference generation

**arXiv ID:** 2601.08514 | [PDF](https://arxiv.org/pdf/2601.08514v1)

**作者:** Davide Risi `[一作]` (University of Salerno), Pasquale Chiacchio `[通讯]` (University of Salerno)

**通讯引用:** 3493 | [OpenAlex ID](https://openalex.org/A5085445088)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了ROS2的模块化架构，将参考生成与控制律分离，设计并实现了两种参考生成器（JRG和TRG）及相应的PDGC、CPC和AC控制器，并在仿真与真实UR10、FRANKA机器人上验证其可重用性和性能。

**💡 创新点**

核心创新是将参考获取、验证、插值等职责从控制器中剥离出来，形成独立的Reference Generator模块；该模块通过ROS2链式机制与下游控制器通信，显著提升了跨机器人、跨控制器的可复用性。

**🔧 技术方法**

技术包括ROS2消息与Action通信、控制器链式机制、有限状态机管理、线性插值、基于Jacobian的逆运动学、重力补偿PD、PID、Admittance控制等。

**📊 数据集**

实验数据集为UR10与FRANKA Emika（FER）机械臂的轨迹点（joint‑space 与 Cartesian 轨迹），以及Gazebo仿真环境中的对应轨迹；未使用公开数据集。

**📈 对比分析**

通过将相同参考生成器与不同控制器组合（如JRG→PDGC、TRG→CPC、TRG→AC→CPC等）在仿真与真实机器人上对比，观察跟踪误差和实现简易性；结果表明轨迹跟踪误差均保持在可接受范围，尤其在PDGC与CPC上误差低，但在Admittance场景下误差增大。

**⚠️ 局限性**

局限性包括：插值仅使用线性方法，缺乏高阶平滑；实验仅覆盖两款机器人，缺乏更广泛平台验证；对接收器/动作接口实现受限于ROS2版本；在受力交互实验中由于可变负载导致误差增大。

---

## 317. It's All About the Confidence: An Unsupervised Approach for Multilingual Historical Entity Linking using Large Language Models

**arXiv ID:** 2601.08500 | [PDF](https://arxiv.org/pdf/2601.08500v1)

**作者:** Cristian Santini `[一作]` (University of Macerata), Mehwish Alam `[通讯]` (Télécom Paris)

**通讯引用:** 1291 | [OpenAlex ID](https://openalex.org/A5009026163)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MHEL‑LLaMo无监督多语言历史实体链接方法，结合多语言bi‑encoder BELA与指令调优LLM进行NIL预测与候选选择。

**💡 创新点**

通过自适应阈值仅在难样本使用LLM，避免无用推理并减少幻觉；将LLM用于候选选择的prompt链；实现不需微调的跨语言历史文本EL。

**🔧 技术方法**

使用多语言bi‑encoder BELA（XLM‑R）+FAISS稠密向量检索；instruction‑tuned LLM（Mistral‑Small‑24B‑Instruct、Gemma‑3‑27B‑it、Poro‑2‑8B‑Instruct）+prompt链进行NIL与候选判定；自适应阈值过滤；SQLite lookup表补充Wikidata元数据。

**📊 数据集**

在HIPE‑2020、NewsEye、AJMC、MHERCL四个历史实体链接基准上，覆盖6种欧洲语言（英语、芬兰语、法语、德语、意大利语、瑞典语），包含19–20世纪报纸、音乐期刊和古典评论等文本。

**📈 对比分析**

在F1指标上与SOTA（MELHISSA、SBB、L3i、BELA等）进行横向比较，MHEL‑LLaMo在所有数据集均超越对手，最高提升达27%（MHERCL），在多语种和不同体裁上均实现显著性能提升。

**⚠️ 局限性**

对芬兰语、瑞典语的性能仍偏低；在经典评论（AJMC）中NIL预测和文化实体链接效果不佳；依赖Wikidata版本及其演化导致标签失效；未采用参数高效微调，推理仍有一定环境成本。

---

## 318. PKI: Prior Knowledge-Infused Neural Network for Few-Shot Class-Incremental Learning

**arXiv ID:** 2601.08493 | [PDF](https://arxiv.org/pdf/2601.08493v1)

**作者:** Kexin Baoa `[一作]`, Shiming Ge `[通讯]` (Institute of Information Engineering, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种Prior Knowledge-Infused Neural Network（PKI）模型，通过在每个增量阶段新增并冻结投影器（projector）来持续保存先前知识，从而解决少样本类别增量学习中的灾难性遗忘与过拟合问题。

**💡 创新点**

创新点在于：1）构建投影器集合，使得每个新任务只更新一个投影器并与之前所有投影器联合使用；2）同时保留冻结的骨干网络、投影器权重和类别均值记忆，形成多源先验知识；3）提出两种轻量化变体PKIV‑1和PKIV‑2，通过压缩投影器集合在保持性能的同时降低存储成本。

**🔧 技术方法**

使用ResNet骨干、三层MLP投影器、类均值记忆；训练时冻结骨干，增量阶段仅微调新投影器和分类器；整体采用交叉熵损失与记忆约束；对投影器进行L2归一化并聚合。

**📊 数据集**

在三个标准基准上评估：CIFAR‑100、MiniImageNet与CUB‑200；分别采用60/40、60/40和100/100的基类/增量类划分。

**📈 对比分析**

与包括NC‑FSCIL、C‑FSCIL、ALICE等最先进的少样本类别增量学习方法对比，PKI在CIFAR‑100、MiniImageNet、CUB‑200上均取得最高或次高平均准确率，提升幅度约1.0%–1.5%，并在大多数增量阶段保持领先。

**⚠️ 局限性**

局限性：投影器集合随增量次数增加导致存储和计算开销累积；对极度相似或分辨率较低的新类仍可能出现性能下降；轻量化变体在进一步压缩时可能略微牺牲准确率。

---

## 319. BenchOverflow: Measuring Overflow in Large Language Models via Plain-Text Prompts

**arXiv ID:** 2601.08490 | [PDF](https://arxiv.org/pdf/2601.08490v1)

**作者:** Erin Feiglin `[一作]` (Deepkeep), Raz Lapid `[通讯]` (Deepkeep)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大语言模型中由普通文本提示引起的过度输出（Overflow）现象进行系统研究，构建 BenchOverflow 基准并评估九种提示策略在九个模型上的影响，同时验证一种简易的“简洁提示”防御。

**💡 创新点**

① 将 Overflow 明确为非对抗性、自然提示导致的超长输出，并提出无需攻击后缀的九类纯文本提示；② 设计 BenchOverflow benchmark 与统一评估协议，使得跨模型、跨提示的长度风险可量化；③ 将长度控制从“风格”问题提升为可靠性与成本问题；④ 发现简洁提醒能显著抑制右尾，提供低成本防御方案。

**🔧 技术方法**

使用元提示（meta‑prompt）生成提示集；统一 token 计数与长度分布分析（直方图、ECDF、CSR@1k/3k/5k）；within‑prompt 方差与跨模型 Pearson 相关性；LLM‑as‑judge 对拒绝与回答质量进行评估；简洁提醒预置为输入前缀进行对照实验。

**📊 数据集**

BenchOverflow 自制数据集：9类 Overflow 提示，每类约 300+ 个手工构造提示；基准 OASST2 作为正常对话参考；评估涵盖 6 个开源模型（Qwen3‑4B/8B、LLaMA‑3.1/3.2、Gemma‑3/2‑9）和 3 个闭源模型（GPT‑5、Gemini‑2.5‑Flash、Claude‑Sonnet）。

**📈 对比分析**

在 5,000 token 预算下，对每个模型/策略执行 4 次，统计输出长度分布、CSR、ECDF、方差与相关性。结果显示大多数模型 Overflow 右偏，Explicit forced length 与 Tokenizer stress 最易触发 5k 触底；简洁提醒可将平均长度下降 30–90%，但在部分模型上略微降低回答完整度。整体表明 Overflow 是可复现且跨模型的可靠性与成本风险。

**⚠️ 局限性**

① 仅在固定默认温度与参数下实验，未统一温度对长度的影响；② 未考虑工具、检索、多模态或多轮对话场景；③ 评估依赖 LLM‑as‑judge，样本有限且可能有误；④ 结果基于单点模型默认配置，后续版本可能变化；⑤ 未直接量化能源消耗或实际运营成本。

---

## 320. AME-2: Agile and Generalized Legged Locomotion via Attention-Based Neural Map Encoding

**arXiv ID:** 2601.08485 | [PDF](https://arxiv.org/pdf/2601.08485v1)

**作者:** Chong Zhang `[一作]` (ETH Zurich), Marco Hutter `[通讯]` (ETH Zurich)

**通讯引用:** 19610 | [OpenAlex ID](https://openalex.org/A5044258783)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了AME-2框架，结合注意力地图编码器与学习式地形映射，实现了对四足和双足机器人在未知多种地形上的高机动性与通用控制；

**💡 创新点**

创新点在于将关注机制嵌入地图编码，配合概率赢家优先融合的贝叶斯不确定性映射，以及教师-学生强化学习结合映射随机化，形成端到端可解释且对噪声鲁棒的完整闭环；

**🔧 技术方法**

技术涵盖PPO强化学习、CNN+贝叶斯网络进行局部高度与不确定性预测、注意力网络（global + local特征融合）、概率赢家取胜融合、动作蒸馏与表示损失的教师-学生训练、深度图投影到局部网格并在线融合；

**📊 数据集**

使用合成的多类随机地形网格（dense、climbing、sparse）及额外的随机堆叠盒子、随机高度场等，共计数千万帧训练样本；在真实实验中使用ANYmal‑D与TRON1机器人及其深度相机、惯导等传感器；

**📈 对比分析**

与AM1、MoE、可视化递归模型等基线相比，未见测试地形平均成功率>80%，在极难赛道实现1.5‑2 m/s速度，零样本完成跳跃、攀爬与稀疏地形行走；相较传统模块化系统，机动性提升约30%~40%，泛化性能显著；

**⚠️ 局限性**

主要局限在仅使用二维高程图，未处理完整3D环境与极端感知损坏；对动态障碍与大范围视野缺陷仍敏感；在技能切换点易失效，需更系统的迁移或全身规划方法。

---

## 321. Cross-modal Proxy Evolving for OOD Detection with Vision-Language Models

**arXiv ID:** 2601.08476 | [PDF](https://arxiv.org/pdf/2601.08476v1)

**作者:** Hao Tang `[一作]` (Centre for Smart Health), Jing Qin `[通讯]` (Centre for Smart Health)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种零样本 OOD 检测框架 CoEvo，该框架在推理阶段不需要训练或标注，通过双向、样本条件的代理对齐共进化机制，实时更新文本和视觉代理缓存，并根据多模态相似度计算 OOD 分数。

**💡 创新点**

创新点：1) 双向样本条件代理对齐共进化，文本与视觉代理相互指导，解决单向动态化导致的模态失衡；2) 动态挖掘上下文负文本代理并同步更新视觉代理；3) 引入自适应阈值和权重翻转的多模态得分演化，实现在线、无参数的稳健 OOD 判别。

**🔧 技术方法**

核心技术：CLIP 预训练的视觉–文本双模编码器；文本/视觉代理缓存（正负队列）；基于余弦相似度的多模态得分；注意力聚合视觉代理；自适应阈值估计与置信度门控；代理共进化的闭环更新策略；多模态得分的权重翻转机制。

**📊 数据集**

实验数据集：ID 采用 ImageNet‑1K；OOD 测试集包括 iNaturalist、SUN、Places、Textures、NINCO、SSB‑hard；OpenOOD benchmark 及其 Near‑OOD / Far‑OOD 设置；此外在不同 CLIP backbones（ViT‑B/16、ViT‑B/32、ResNet‑50）上验证。

**📈 对比分析**

与 NegLabel、CSP、AdaNeg 等主流负标签基线以及训练/无训练方法比较，CoEvo 在 ImageNet‑1K 上 AUROC 97.95%（比最佳基线提升 1.33%），FPR95 10.22%（比 AdaNeg 降低 45.98%）。在 OpenOOD 的 Near‑OOD 和 Far‑OOD 场景下，FPR95 分别为 66.88% 与 14.47%，AUROC 分别为 74.65% 与 96.70%，均优于现有方法。推理速度略低于简单负标签基线，但无额外参数，保持轻量部署。

**⚠️ 局限性**

局限性：1) 需要大规模、语义丰富的文本语料库来挖掘负文本代理；2) 依赖 CLIP 预训练模型，在与预训练域差异显著的领域（如医学、遥感）可能性能下降；3) 代理更新策略启发式，面对噪声或对抗样本可能不够鲁棒；4) 维护代理缓存导致额外的运行时内存和计算开销，虽然可控但仍高于完全无更新方法。

---

## 322. sui-1: Grounded and Verifiable Long-Form Summarization

**arXiv ID:** 2601.08472 | [PDF](https://arxiv.org/pdf/2601.08472v1)

**作者:** Benedikt Droste `[一作]` (Ellamind), Björn Plüster `[通讯]` (Ellamind)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个24B参数的语言模型sui‑1，用于生成带内联引用的可验证摘要。

**💡 创新点**

创新点在于通过合成数据管线结合链式推理和多阶段验证，自动生成带XML标签的高质量引用样本，实现了大规模、跨语言的引用驱动摘要训练。

**🔧 技术方法**

主要技术包括MD5哈希生成唯一句子标签、基于Mistral‑Small‑3.2‑24B的LoRA微调、100K token上下文处理、FP8量化以及多阶段过滤与链式推理提示。

**📊 数据集**

使用了德国议会文件系统、Common Crawl/OSCAR和多语言维基百科等来源，数据中约74%为德语，10%英语，5%法语、意大利语、西班牙语，训练集22,152条。

**📈 对比分析**

通过LLM‑as‑a‑judge评估，sui‑1在五项指标（真实性、覆盖率、具体性、格式合规、指令遵循）上平均得分0.842，格式合规率0.895，显著优于所有开放权重基线（0.137–0.411），仅落后参考模型0.891。

**⚠️ 局限性**

局限性包括：训练数据以德语为主，可能在其他语言或专业领域泛化不足；评估依赖LLM‑as‑a‑judge，可能存在系统性偏差。

---

## 323. Towards Safer Mobile Agents: Scalable Generation and Evaluation of Diverse Scenarios for VLMs

**arXiv ID:** 2601.08470 | [PDF](https://arxiv.org/pdf/2601.08470v1)

**作者:** Takara Taniguchi `[一作]` (OMRON SINIC X), Atsushi Hashimoto `[通讯]` (OMRON SINIC X)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了可扩展的生成管道 HazardForge，用于自动合成移动、侵入、远距离等危险场景，并利用其生成 MovSafeBench 评测移动代理 VLM 的安全决策。

**💡 创新点**

首次将图像编辑模型与布局决策和 VLM 验证模块结合，能够在保持原始安全动作不变的前提下生成多模态危险场景，填补了现有基准对异常与时空多样化场景缺失的问题。

**🔧 技术方法**

采用图像编辑模型（如 Stable Diffusion 等）、VLM‑based 验证模块（Qwen3‑VL‑30B）以及基于遮罩的布局决策算法。

**📊 数据集**

从 DriveBench 与 SA‑Bench 各 200 张真实图像作为输入，生成 7,254 张带 MCQ 的场景，包含 13 种物体（常见与异常）及 4 种时空场景。

**📈 对比分析**

在 MovSafeBench 上评估 7 款 VLM（Qwen2.5‑VL、LLaVA‑NEXT、Phi4‑Multimodal、InternVL3.5、PaliGemma2 等），相较于未编辑图像，VLM 在运动、距离等情境下准确率下降约 20‑30%，人类表现远高于 VLM，显示 VLM 对异常与时空推理能力不足。

**⚠️ 局限性**

局限性包括依赖图像编辑模型的生成质量与 VLM 验证的准确性，且仅覆盖了 13 种物体与 4 种场景，未考虑更复杂的交互或多传感器输入。

---

## 324. Zero-Shot Distracted Driver Detection via Vision Language Models with Double Decoupling

**arXiv ID:** 2601.08467 | [PDF](https://arxiv.org/pdf/2601.08467v1)

**作者:** Takamichi Miyata `[一作]` (Chiba Institute of Technology), Andrew Morris `[通讯]` (Loughborough University)

**通讯引用:** 5786 | [OpenAlex ID](https://openalex.org/A5085455108)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出零样本偏离驾驶检测框架，利用视觉语言模型通过主体外观解耦和文本正交化提升检测精度。

**💡 创新点**

双重解耦创新：一方面从图像嵌入中去除驾驶员外观因素，另一方面将文本嵌入正交化，显著增强类间可分性。

**🔧 技术方法**

使用CLIP视觉语言模型、均值嵌入减法（去除外观）、Stiefel矩阵正交化（SVD）以及精细prompt工程。

**📊 数据集**

SAM-DD（Singapore AutoMan@NTU）偏离驾驶图像数据集。

**📈 对比分析**

与DriveCLIP基线对比，在10类分类中Top‑1提升至75.9%（从66.5%），二分类AUPRC提升至95.8%（从90.6%），误报率降至10.9%。

**⚠️ 局限性**

仅依赖单摄像头视觉，误警和漏检仍可能导致安全风险，且对极端照明、视角或不同车型的泛化尚未完全验证。

---

## 325. CoMa: Contextual Massing Generation with Vision-Language Models

**arXiv ID:** 2601.08464 | [PDF](https://arxiv.org/pdf/2601.08464v1)

**作者:** Evgenii Maslov `[一作]` (FusionBrain Lab), Ivan Oseledets `[通讯]` (Institute of Numerical Mathematics)

**通讯引用:** 10340 | [OpenAlex ID](https://openalex.org/A5004111307)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 CoMa‑20K 数据集，并基于 Vision‑Language 模型实现建筑体量的条件生成。

**💡 创新点**

创新点在于首次构建大规模多模态体量数据集，并将体量生成任务形式化为 VLM 的条件生成任务。

**🔧 技术方法**

使用 Qwen3‑VL 系列 VLM 进行细调（2B/4B/8B）与零样本推理（235B），结合 LoRA、FSDP 等训练技巧。

**📊 数据集**

CoMa‑20K 数据集包含 20,000 个开发场地，提供建筑几何、功能需求和多视图城市语境。

**📈 对比分析**

通过对比细调模型与零样本模型在 Pattern Match、JSON Validity、ID IoU、Floor Error、Area Error、Site IoU、Contextual Relevance 等指标，零样本模型在格式指标上表现最好，细调模型在几何复杂度上更佳，但整体表现仍偏低。

**⚠️ 局限性**

局限性包括几何准确性与上下文适配不足，复杂建筑生成易出现自交和形状不合理；数据集偏向简单形状，导致模型泛化受限。

---

## 326. Real2Sim based on Active Perception with automatically VLM-generated Behavior Trees

**arXiv ID:** 2601.08454 | [PDF](https://arxiv.org/pdf/2601.08454v1)

**作者:** Alessandro Adami `[一作]` (Polytechnic of Bari), Pietro Falco `[通讯]` (University of Padova)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于视觉语言模型（VLM）自动生成行为树（BT），让力控Franka Emika Panda机器人执行目标驱动的物理交互，估计缺失的质量、摩擦、表面高度等参数，从而构建高保真数字孪生。

**💡 创新点**

创新点在于将多模态场景理解、VLM推理与BT生成相结合，实现任务意图驱动的、无需预设脚本的自适应参数采集；并通过BT保证执行的可解释性与安全性。

**🔧 技术方法**

使用的核心技术包括：ChatGPT‑5/其他VLM进行语义理解和BT结构化输出；Cartesian impedance 控制与力/扭矩传感器实现顺应交互；行为树（BT）库实现分层、可重用的执行逻辑；MuJoCo 进行物理仿真。

**📊 数据集**

数据来源主要是实验室真实机器人与RGB相机捕获的图像，以及利用MuJoCo生成的合成图像；未使用公开标准数据集，而是针对三种场景（质量、摩擦、遮挡）自构造实验数据。

**📈 对比分析**

通过与传统固定探索策略对比，实验表明在三种场景下，该方法只采集任务所需参数，估计误差低且执行更高效；实验中质量误差≈2–3 %，摩擦系数误差约10–20 %，验证了方法的实用性和可解释性。

**⚠️ 局限性**

局限性包括：假设已知物体位姿（需改为视觉位姿估计）；对VLM幻觉的防护依赖特定提示，且未对数字孪生的整体逼真度做系统化量化；仅在单一桌面场景验证，未覆盖复杂多物体、多任务环境。

---

## 327. Sleep-Based Homeostatic Regularization for Stabilizing Spike-Timing-Dependent Plasticity in Recurrent Spiking Neural Networks

**arXiv ID:** 2601.08447 | [PDF](https://arxiv.org/pdf/2601.08447v1)

**作者:** Andreas Massey `[一作]`, Solve Sæbø `[通讯]`

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并验证了基于睡眠相似的自适应正则化机制，稳定并提升了使用STDP的脉冲神经网络学习表现。

**💡 创新点**

引入周期性睡眠阶段的权重幂律衰减与自发重放，首次在局部Hebbian SNN中证明其可降低权重发散、提升泛化，同时揭示其对SG梯度学习的局限。

**🔧 技术方法**

使用基于LIF的可积-消耗模型实现STDP与iSTDP，配合噪声驱动的睡眠正则化；SG模型采用surrogate-gradient反向传播；统计分析采用混合效应β回归。

**📊 数据集**

在几何噪声数据和MNIST族（KMNIST、MNIST、FMNIST、NotMNIST）上进行实验。

**📈 对比分析**

通过在相同网络结构下比较有无睡眠、不同睡眠比例的准确率，STDP模型在10–20%睡眠时提升≈3–4%（MNIST族），而SG模型对睡眠几乎无影响，整体准确率仍低于最先进方法。

**⚠️ 局限性**

受限于单层浅层架构、训练一次epoch、缺乏更大规模验证，且SG与STDP在输入编码、学习信号等方面差异导致无法直接对比；总体准确率仍远低于传统ANN。

---

## 328. Rewriting Video: Text-Driven Reauthoring of Video Footage

**arXiv ID:** 2601.08565 | [PDF](https://arxiv.org/pdf/2601.08565v1)

**作者:** Sitong Wang `[一作]` (Columbia University), Dingzeyu Li `[通讯]` (Adobe Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于文本的可编辑脚本的文本驱动视频再创作系统，并在此基础上进行实验研究

**💡 创新点**

① 提出了逆向生成算法，将视频重构为可编辑文本提示；② 设计了交互探针 Rewrite Kit，使创作者能通过文本编辑视频脚本并生成新视频

**🔧 技术方法**

利用 Gemini 2.5 Pro 进行文本提示生成与比较，Veo‑3 文本到视频模型生成视频，CLIP（ViT‑B/32）计算相似度，人工评估视频质量

**📊 数据集**

30 条多样化的 8 秒视频剪辑（vlog、动画、电影场景、社交媒体等）作为算法和人类评估的数据集

**📈 对比分析**

算法在 3–6 次迭代后收敛，平均 CLIP 相似度最高 0.9145（比初始提升 0.0135）；人工评估平均得分 5.07/7，96.7% 的重建视频被认为可接受，但评审强调时间连贯性比帧级细节更重要

**⚠️ 局限性**

仅适用于 10 秒以内的短视频；实验依赖参考视频作为学习工具，缺乏无参考的全新叙事能力；使用的模型非最新（未评估 Sora‑2 等）；文本驱动方式在表达复杂视听语言时受限，需加入更多多模态交互

---

## 329. MASH: Evading Black-Box AI-Generated Text Detectors via Style Humanization

**arXiv ID:** 2601.08564 | [PDF](https://arxiv.org/pdf/2601.08564v1)

**作者:** Yongtong Gu `[一作]` (Southeast University), Xia Hu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于多阶段风格迁移的黑盒文本生成检测逃逸框架 MASH

**💡 创新点**

将检测逃逸重新定义为“机器写作到人类写作”的风格迁移任务，采用分阶段训练（Style-SFT、DPO 对齐、推理时细化）实现高成功率

**🔧 技术方法**

风格注入式监督微调 (Style-SFT)、直接偏好优化 (DPO)、推理时对抗细化、BART 预训练模型、LLM 生成并过滤数据

**📊 数据集**

六个领域的数据集（MGTBench 的 Essay、Reuters、WP；MGT‑Academic 的 STEM、Social、Humanity），以及公开的对抗与零样本检测器（RoBERTa、Binoculars、SCRN、Writer、Scribbr）

**📈 对比分析**

与 11 种基线（扰动、提示、改写）进行对比，平均攻击成功率提升 24%~30%（对 RoBERTa），对商业检测器的平均 ASR 约 89%，文本质量指标（PPL、GRUEN、BERTScore）均保持或提升

**⚠️ 局限性**

仅在英语数据上验证；对检测器高假阳性时无意义；需要先验人类文本样本；未在多语言、低资源场景或未来新检测方法上进行测试

---

## 330. VideoHEDGE: Entropy-Based Hallucination Detection for Video-VLMs via Semantic Clustering and Spatiotemporal Perturbations

**arXiv ID:** 2601.08557 | [PDF](https://arxiv.org/pdf/2601.08557v1)

**作者:** Sushant Gautam `[一作]` (SimulaMet and OsloMet), Pål Halvorsen `[通讯]` (SimulaMet and Forzasys and OsloMet)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 VideoHEDGE 框架，对视频视觉‑语言模型（Video‑VLM）的幻觉进行检测，结合多样化采样、语义聚类与熵基可靠度分数；

**💡 创新点**

创新点在于：①将 HEDGE 的熵基方法推广到时序视频输入；②引入 Vision‑Amplified Semantic Entropy（VASE），利用清洁与扰动视频语义分布差异；③对 NLI 与句子嵌入聚类的可扩展性与效果进行系统比较；

**🔧 技术方法**

使用高温采样、光度与时空噪声扰动、NLI 与句子嵌入语义聚类、熵、RadFlag、VASE 可靠度计算，以及 Qwen3‑30B‑A3B 作为 LLM‑as‑judge 进行幻觉判定；

**📊 数据集**

采用 SoccerChat 数据集（包含事件分类与 VideoQA 任务的足球短视频片段）；

**📈 对比分析**

在三种 7B 视觉语言模型（Qwen2‑VL、Qwen2.5‑VL、SoccerChat‑qwen2‑vl）上，使用不同帧数、像素预算和扰动组合进行评估；VASE 在较大扰动预算下实现 ROC‑AUC 0.57–0.67，优于 SE 与 RadFlag；embedding 聚类与 NLI 的 ROC‑AUC 接近，但前者计算更快；Fine‑tuning 降低幻觉率但对 ROC‑AUC 的提升有限；

**⚠️ 局限性**

局限包括：ROC‑AUC 仍处于中等水平；帧数/分辨率对性能影响不稳定；评估依赖 LLM‑judge，缺乏人类标注；扰动类型有限，未探索更复杂或多模态扰动；未在训练阶段加入可靠性相关目标。

---

## 331. EviNAM: Intelligibility and Uncertainty via Evidential Neural Additive Models

**arXiv ID:** 2601.08556 | [PDF](https://arxiv.org/pdf/2601.08556v1)

**作者:** Sören Schleibaum `[一作]` (Clausthal University of Technology), Jörg P. Müller `[通讯]` (Clausthal University of Technology)

**通讯引用:** 4165 | [OpenAlex ID](https://openalex.org/A5005203044)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Evidential Neural Additive Model（Evidential NAM），在单次前向推理中同时提供可解释的特征贡献以及先验和经验不确定性估计。

**💡 创新点**

通过将非线性激活函数“转发”到特征级别，实现了既满足分布参数约束，又保持加性结构，从而实现单次前向即可得到可解释特征贡献及两类不确定性。

**🔧 技术方法**

结合深度经验分布式回归（DER）与神经可加模型（NAM），使用Normal Inverse Gamma (NIG) 与 Dirichlet 分布、softplus/identity 激活、Adam优化器及贝叶斯优化超参调优。

**📊 数据集**

在OpenML套件的24个回归数据集和10个分类数据集上评估，另外使用合成的1D/2D数据验证不确定性。

**📈 对比分析**

与NAM、NAMLSS、EnsNAM以及DER-MLP等基线相比，Evidential NAM在MAE、NLL、CRPS等回归指标上与NAM、EnsNAM相当或更优，在分类指标上平均排名略低于基线，但实现了单次推理的两类不确定性和可加类别概率。

**⚠️ 局限性**

在回归任务中模型训练时间略高于NAM/NAMLSS，且对分布参数学习更为复杂；在分类任务中相对基线性能略逊，且对多类别场景的可解释性尚未彻底验证。

---

## 332. Contrastive and Multi-Task Learning on Noisy Brain Signals with Nonlinear Dynamical Signatures

**arXiv ID:** 2601.08549 | [PDF](https://arxiv.org/pdf/2601.08549v1)

**作者:** Sucheta Ghosh `[一作]` (Heidelberg University), Felix Dietrich `[通讯]` (Technical University of Munich)

**通讯引用:** 1335 | [OpenAlex ID](https://openalex.org/A5065128348)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种两阶段多任务学习框架，先用去噪自编码器去除EEG噪声，再用Transformer+CNN骨干同时完成运动想象分类、混沌/非混沌判别和对比学习表示学习。

**💡 创新点**

创新点在于将去噪、动力学特征（Lyapunov指数）判别与自监督对比学习统一到同一端到端体系，并通过多任务共享降低目标冲突，实现噪声鲁棒性与动力学感知的协同提升。

**🔧 技术方法**

技术包括1D卷积去噪自编码器、Transformer自注意力编码器、CNN空间特征提取、基于Lyapunov指数的混沌标签生成、NT‑Xent对比损失、AdamW优化及多任务加权损失。

**📊 数据集**

使用公开EEG数据集BCI2000（109受试者，64通道）和BNCI Horizon 2020（9受试者，22通道）进行训练与评估。

**📈 对比分析**

与传统RNN、单任务Transformer+CNN、基于Lyapunov的单任务模型以及最新SOTA（EEGNet、FBCNet、TS‑TCC、CL‑EEG等）比较，实验显示在BCI2000和BNCI Horizon 2020上均取得最高F1（约0.84/0.83），并在混沌判别任务中也保持领先。

**⚠️ 局限性**

局限包括：受限于标注EEG数据量不足，低信噪比环境下仍可能残留噪声；模型缺乏可解释性，需进一步引入可解释AI方法；仅考虑混沌/非混沌两类动力学，未来可扩展至周期、准周期等更多状态。

---

## 333. Reducing Compute Waste in LLMs through Kernel-Level DVFS

**arXiv ID:** 2601.08539 | [PDF](https://arxiv.org/pdf/2601.08539v1)

**作者:** Jeffrey Spaan `[一作]` (University of Twente), Ana-Lucia Varbanescu `[通讯]` (University of Twente)

**通讯引用:** 1625 | [OpenAlex ID](https://openalex.org/A5109847155)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对大语言模型训练过程中的能耗浪费，提出基于核级（kernel-level）动态频率与电压调节（DVFS）的细粒度优化方案，旨在在不显著降低性能的前提下显著降低能耗。

**💡 创新点**

创新点在于：①将DVFS粒度从传统的迭代或通道级提升至核级；②引入“计算浪费（compute waste）”这一新的优化目标，替代传统的能耗-延迟乘积（EDP）；③构建全局与局部两种优化策略，证明全局优化能进一步提升能耗收益。

**🔧 技术方法**

技术手段包括：使用NVIDIA RTX 3080 Ti GPU的手动DVFS接口（nvidia‑smi）对核心频率与内存频率进行枚举；利用C/CUDA实现的LLM框架对GPT‑3‑XL模型的各个算子进行单独测量；基于全搜索的频率组合评估，并采用全局/局部约束求解器挑选最优配置。

**📊 数据集**

数据集：在GPT‑3‑XL（1.3 B参数）模型上进行训练，序列长度固定为1024，批量大小默认40，实验覆盖多种数据并行与张量并行度（1–16）以及不同批次大小的情况。

**📈 对比分析**

对比方法：将提出的核级DVFS与传统的通道级或迭代级DVFS（以及EDP优化）进行对比；实验结果显示，核级全局优化在严格浪费约束下可实现约15.6 %能耗下降且性能提升约0.6 %，相比通道级仅能实现约2 %能耗下降；在不同并行度和批次大小下，能耗收益保持在14–16 %之间，性能损失不超过1 %。

**⚠️ 局限性**

局限性：①频率切换延迟较高，限制了真正的核级动态调节；②实验仅在单一GPU型号（RTX 3080 Ti）和单一模型（GPT‑3‑XL）上验证，跨GPU和更大模型的可迁移性需要进一步验证；③测量误差和统计波动导致实验结果略低于理论最佳值；④未考虑多节点分布式训练中网络通信与同步开销对DVFS策略的影响。

---

## 334. How Hard Is It to Rig a Tournament When Few Players Can Beat or Be Beaten by the Favorite?

**arXiv ID:** 2601.08530 | [PDF](https://arxiv.org/pdf/2601.08530v1)

**作者:** Zhonghao Wang `[一作]` (University of Electronic Science and Technology of China), Mingyu Xiao `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 1430 | [OpenAlex ID](https://openalex.org/A5033729619)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了锦标赛修复问题（TFP），即在给定的锦标赛有向图中，是否可以安排比赛以确保指定选手v^*获胜。

**💡 创新点**

引入了两个新的结构参数：选手v^*的入度k和出度ℓ，并证明了当这两个参数较小时，TFP可以有效解决。这为TFP的参数化算法理解提供了新的视角。

**🔧 技术方法**

使用了颜色编码技术和结构分析方法，设计了基于这些新参数的算法。

**📊 数据集**

使用了锦标赛的有向图D，研究了其中的选手v^*及其入度和出度。

**📈 对比分析**

与之前的算法相比，本文的算法在入度和出度参数化下均表现出固定参数可解性，尤其是入度参数化的算法具有较高的技术复杂性，性能优于现有的基于全局参数的算法。

**⚠️ 局限性**

限制在于尚未解决子集fas/fvs数的参数化复杂性，且对于这些参数是否会导致TFP的NP-hard性仍不清楚。

---

## 335. Provably Safe Reinforcement Learning using Entropy Regularizer

**arXiv ID:** 2601.08646 | [PDF](https://arxiv.org/pdf/2601.08646v1)

**作者:** Abhijit Mazumdar `[一作]` (Aalborg University), Manuela L. Bujorianu `[通讯]` (Aalborg University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在有限状态马尔可夫决策过程中设计了两种安全强化学习算法（PSRL 与 ER-PSRL），通过扩展线性规划和熵正则化实现了在学习阶段以高概率满足安全约束，并给出了有限样本下的收益（回报）与约束违背（风险）收敛分析。

**💡 创新点**

① 结合安全约束的马尔可夫决策过程（CMDP）采用“reach–avoid”框架；② 基于乐观不确定性（OFU）原理提出 PSRL；③ 在 PSRL 的基础上加入熵正则化得到 ER-PSRL，显著降低每回合波动、提升探索效率；④ 证明 ER-PSRL 的回报收敛速率不随安全裕度（p-p^s）恶化而下降，优于 PSRL。

**🔧 技术方法**

技术方法包括：
- 线性规划与扩展 LP 解决约束最优策略；
- 乐观不确定性（OFU）构造可信的转移概率界；
- 熵正则化引入强凸性，保证解的连续性与可扩展性；
- 经验 Bernstein 置信区间与安全基线策略构造；
- 通过概率论证明安全性与回报上界。

**📊 数据集**

实验使用自定义 5 状态、2 动作的小型 MDP（含目标、危害、生活集合），不使用公开数据集。

**📈 对比分析**

与 PSRL 进行对比：
- ER-PSRL 的每回合回报波动显著降低，累计回报始终低于 PSRL；
- 两种算法的每回合与累计约束违背均保持在 0 以下，说明安全约束始终满足；
- 通过多次实验验证，知识到代理集合（proxy set）可进一步降低累计回报。

**⚠️ 局限性**

局限性：
- 仅适用于有限状态动作空间，未考虑函数逼近或连续空间；
- 需要已知或可构造的安全基线策略（需每个代理状态有安全动作）；
- 线性规划求解在大规模 MDP 中计算成本高；
- 论文仅在模拟环境验证，缺乏真实世界应用与更大规模实验。

---

## 336. Quantum CSS LDPC Codes based on Dyadic Matrices for Belief Propagation-based Decoding

**arXiv ID:** 2601.08636 | [PDF](https://arxiv.org/pdf/2601.08636v1)

**作者:** Alessio Baldelli `[一作]` (Università Politecnica delle Marche), Laurent Schmalen `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 4042 | [OpenAlex ID](https://openalex.org/A5053280913)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于二次矩阵的代数构造方法，用于设计经典和量子低密度奇偶校验（LDPC）码，特别是量子CSS LDPC码。

**💡 创新点**

创新点在于引入了基于二次矩阵的构造方法，能够设计出具有更高速率和良好解码性能的新码，同时满足CAMEL框架的兼容性条件。

**🔧 技术方法**

使用了代数构造方法，结合了CAMEL框架和基于二次排列矩阵的设计，采用了四元信念传播解码策略。

**📊 数据集**

使用了不同的有限域（如𝔽_16和𝔽_32）构建量子QLDPC码，并通过Monte Carlo模拟评估其在去极化信道上的逻辑错误率（LER）。

**📈 对比分析**

与传统的四元BP解码方法相比，CAMEL解码在抑制错误地板方面表现出明显优势，尤其是在存在长度为4的循环时，性能显著优于其他参考码。

**⚠️ 局限性**

限制在于尽管新构造的码在高率和解码性能上具有竞争力，但在某些情况下仍可能受到短循环的影响，尤其是在使用传统的四元BP解码时。

---

## 337. Moral Lenses, Political Coordinates: Towards Ideological Positioning of Morally Conditioned LLMs

**arXiv ID:** 2601.08634 | [PDF](https://arxiv.org/pdf/2601.08634v1)

**作者:** Chenchen Yuan `[一作]` (Technical University of Munich), Gjergji Kasneci `[通讯]` (Technical University of Munich)

**通讯引用:** 13473 | [OpenAlex ID](https://openalex.org/A5024434748)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过在大型语言模型中注入不同道德价值观（如功利主义、道义主义、关怀、正义等）并使用政治罗盘测试（PCT）测量其政治坐标，探讨道德条件对模型政治立场的因果影响；

**💡 创新点**

创新之处在于将道德视为可控的“镜头”而非单纯的社会身份，系统评估多种道德量表对政治倾向的影响，并揭示角色框架和模型规模如何调节这种关系；

**🔧 技术方法**

采用道德条件注入技术（描述性、第一人称、第三人称、候选人-选民等），结合政治罗盘测试作为评估标准，并计算平均位移、方向一致性、翻转率等统计指标；

**📊 数据集**

使用的实验数据包括62条政治罗盘题目，以及三种道德量表（Moral Foundations Questionnaire、Oxford Utilitarianism Scale、FactualDilemmas），并评估12个LLM（LLaMA、Qwen、Mistral、Phi、OpenAI GPT‑4o/4/5等）；

**📈 对比分析**

通过对不同道德条件、角色框架及模型规模的结果进行对比，发现第三人称和候选人框架显著放大政治位移，模型规模越大响应越极化，且与人类样本的趋势保持一致；

**⚠️ 局限性**

局限性包括仅在单一文化背景下测试、未考虑中立选项、仅基于行为观察缺乏内部机制解释、可能受到社会期望偏差以及道德量表可靠性有限（尤其是德ontology）。

---

## 338. Get away with less: Need of source side data curation to build parallel corpus for low resource Machine Translation

**arXiv ID:** 2601.08629 | [PDF](https://arxiv.org/pdf/2601.08629v1)

**作者:** Saumitra Yadav `[一作]` (International Institute Information Technology), Manish Shrivastava `[通讯]` (International Institute Information Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于句子结构复杂度的源侧数据筛选框架 LALITA，旨在为低资源机器翻译构建更高质量、规模更小的平行语料库。

**💡 创新点**

核心创新在于：①从丰富的词汇、句法、语义特征构造多维向量；②通过 PCA 提取第一主成分作为 LALITA 分数，衡量句子结构复杂度；③根据该分数对句子进行聚类与选择，从而实现高效的数据裁剪与增量生成。

**🔧 技术方法**

技术手段包括：Trankit 语法分析、词汇/语法特征统计、PCA 降维、Fisher‑Jenks 聚类、BPE 分词、Transformer NMT 训练（Fairseq）、back‑translation、数据过滤（长度比、字符比例、重复等）。

**📊 数据集**

主要使用数据集为 Samanantar 英-印（约 1.8M 并经清洗后 1.85M 对），并在此基础上构造 50K–800K 子集；在实验中还扩展到英-奥迪亚、英-尼泊尔、英-挪威 Nynorsk、英-德语等语言对。

**📈 对比分析**

与随机采样（RS）和比例采样（baselineP）比较，LALITA 在 0_0_0_100 这一“仅取最复杂句子”配置下，往往在相同或更少的数据量下获得 3–7 % CHRF++ 的提升，数据量下降可达 44–60%。在高资源英-德语实验中，0_0_0_100 甚至超越完整 2M 语料的表现。

**⚠️ 局限性**

主要局限：①需要可靠的源侧句法分析工具，限制了极低资源语言的适用性；②仅关注句子级别，未考虑跨句子语篇信息；③未引入句子嵌入或更深层语义特征；④大规模超参数搜索与实验耗时高，需进一步优化。

---

## 339. GraphSearch: Agentic Search-Augmented Reasoning for Zero-Shot Graph Learning

**arXiv ID:** 2601.08621 | [PDF](https://arxiv.org/pdf/2601.08621v1)

**作者:** Jiajin Liu `[一作]` (New York University), Qiaoyu Tan `[通讯]` (New York University)

**通讯引用:** 1007 | [OpenAlex ID](https://openalex.org/A5043697901)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于搜索增强的推理框架GraphSearch，可在图结构数据上实现零样本图学习。

**💡 创新点**

核心创新在于图感知查询规划器（Graph-Aware Query Planner）和图感知检索器（Graph-Aware Retriever），通过把结构范围与语义查询解耦来实现结构化检索，并提供递归和全局两种遍历模式，显著提升了零样本推理的效果与效率。

**🔧 技术方法**

技术手段包括：LLM（如Qwen、Llama系列）在推理循环中的动态查询生成；基于邻域、全局PageRank和属性相似度构造候选集；混合语义-结构排名（α调节）；递归（GraphSearch‑R）与全局（GraphSearch‑F）检索策略；以及与传统GNN和GraphRAG方法的对比实验。

**📊 数据集**

使用六个公开基准数据集：OGB-Products、Amazon‑Sports‑Fitness、Amazon‑Computers、Cora、PubMed、Reddit，涵盖电商、学术引用和社交网络三大领域。

**📈 对比分析**

与传统的直接推理、搜索增强LRM、GraphICL、GraphRAG、以及多种GNN/LLM图学习方法比较，GraphSearch在六个数据集的零样本节点分类和链接预测任务上均获得或近似最佳的准确率，超越GraphICL、Search-o1和多数基准，并在检索延迟与token消耗上实现了1.3–5.8倍的加速和几乎相同的token使用。

**⚠️ 局限性**

局限性包括：仍需依赖图的邻域信息，对极大稠密或异构图的扩展尚未充分验证；检索过程受限于内存索引实现，实际部署中可进一步优化；α参数对不同图的敏感度需要经验调整；以及对LLM规模的依赖，在小模型上性能提升有限。

---

## 340. ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios

**arXiv ID:** 2601.08620 | [PDF](https://arxiv.org/pdf/2601.08620v1)

**作者:** António Loison `[一作]` (Illuin Technology), Gautier Viaud `[通讯]` (Illuin Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了ViDoRe V3多模态检索生成基准，包含人类注释的多语言、多页、视觉元素丰富的文档检索与生成任务。

**💡 创新点**

通过三阶段人工标注流程生成多类型查询与细粒度视觉定位，整合检索、生成、定位三组件评估，并在10个专业领域、6语言中发布。

**🔧 技术方法**

采用视觉语言模型预过滤、文本与视觉检索器（如ColEmbed）、文本与视觉重排序器、LLM与VLM生成器（Gemini 3 Pro、Qwen3‑VL）以及MTEB排行榜评估框架。

**📊 数据集**

使用10个公开行业文档集（Finance、Physics等）共26,000页，3,099个查询，覆盖6语言，完成12,000小时人工标注。

**📈 对比分析**

按检索NDCG@10、生成准确率、视觉定位F1进行基准评估；视觉检索器领先文本，文本重排序提升显著；生成Oracle 64.7%（hard）对比最佳非Oracle 54.7%；视觉定位F1仅0.09。

**⚠️ 局限性**

仅限英法源文与6西欧语言；主以公开长篇文档，缺乏私有短文本；标注存在主观性；视觉定位性能仍低。

---

## 341. VeriTaS: The First Dynamic Benchmark for Multimodal Automated Fact-Checking

**arXiv ID:** 2601.08611 | [PDF](https://arxiv.org/pdf/2601.08611v1)

**作者:** Mark Rothermel `[一作]` (Technical University of Darmstadt), Anna Rohrbach `[通讯]` (Technical University of Darmstadt)

**通讯引用:** 5916 | [OpenAlex ID](https://openalex.org/A5037747070)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了首个动态多模态自动事实核查（AFC）基准——Verified Theses and Statements，涵盖 24,000 条多语言（54 种语言）真实事实核查声明，包含文本、图片与视频，并通过每季度自动更新以避免数据泄露。

**💡 创新点**

创新点：1）首次实现多模态（文本+图片+视频）真实声明的动态基准；2）引入四维分级评判（媒体真实性、上下文、真实性、上下文覆盖）及整体完整度（Integrity）评分，并以 -1~1 细粒度不确定度标注；3）全自动七阶段数据构建与注释流程，使用 LLM 集成实现高质量标注与修正；4）提供基于专家判定的文本解释，便于结果可解释性。

**🔧 技术方法**

技术：使用 GPT‑5、Gemini‑2.5、Gemini‑3 Pro 等大型多模态 LLM 进行文本归一、媒体检索、判定标准化、纠错与验证；链式思维与少样本提示；视频以五帧截取；基于 CLIP 嵌入做媒体相似度过滤；聚合四个 LLM 的平均分以提高一致性。

**📊 数据集**

数据集：来自 ClaimReview 结构化事实核查结果的 371K 条评论，过滤后得到 24K 条经过自动化标准化与纠错的多模态声明；每季度 1K 条，涵盖 54 语言、图像 8,377 个、视频 4,785 个；同时提供长时序 2,400 条平衡样本用于时间分析。

**📈 对比分析**

比较方法：在最新 Q4‑2025 切片对 Gemini‑3 Pro、Gemini‑2.0 Flash、GPT‑5、Gemini‑3 Pro‑search、DEFAME、Loki 等基线模型进行评估；核心指标 MSE、MAE、Accuracy；Gemini‑3 Pro 在无检索下 MSE=0.39、MAE=0.35、Acc≈75%；其余模型性能更差；知识截止后 MSE 进一步升高，表明现有模型对新出现声明的可靠性不足；未达到 0.1 的阈值，仍需提升。

**⚠️ 局限性**

限制：1）未做跨语言去重，可能出现多语言重复声明；2）自动生成的纠正声明在语言风格和简洁度上仍不理想；3）仅覆盖文本、图像、视频，未涵盖音频等新型媒介；4）构建成本高（约 14.9k 美元、600 美元/季度、2.7k GPU 时数），限制资源有限实验室的复现；5）受现行事实核查生态、平台政策及媒体可访问性变化的影响。

---

## 342. Sketch-Based Facade Renovation With Generative AI: A Streamlined Framework for Bypassing As-Built Modelling in Industrial Adaptive Reuse

**arXiv ID:** 2601.08531 | [PDF](https://arxiv.org/pdf/2601.08531v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 343. ExpSeek: Self-Triggered Experience Seeking for Web Agents

**arXiv ID:** 2601.08605 | [PDF](https://arxiv.org/pdf/2601.08605v1)

**作者:** Wenyuan Zhang `[一作]` (Institute of Information Engineering Chinese Academy of Sciences), Yongbin Li `[通讯]` (Tongyi Lab Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自触发式经验寻求框架 ExpSeek，使 Web 代理在交互过程中主动根据步级熵指标寻找并使用经验指导。

**💡 创新点**

创新点在于：①将经验注入从被动全局上下文转变为主动步级寻求；②使用模型自身熵值作为内在触发信号并通过自助阈值区间进行动态决策；③构建基于成功/失败轨迹的经验三元组并按主题组织，供经验模型实时生成定制化指导。

**🔧 技术方法**

技术包括：ReAct 代理框架；熵计算与基于 Bootstrap 的阈值估计；经验库构建（成功/失败对比、误差分析、主题聚类）；经验模型（生成式）与上下文检索结合；步骤级熵自触发机制。

**📊 数据集**

使用四个公开 Web 代理基准（GAIA、WebWalkerQA、xbench‑DeepSearch、Seal‑Hard）和 Qwen3‑8B/32B 语言模型进行评测，并在训练阶段采样 WebWalkerQA 25% 例子构建经验库。

**📈 对比分析**

与传统全局经验注入方法（GRPO、ReasoningBank）及强化学习模型相比，ExpSeek 在 Qwen3‑8B 上平均提升 9.3%、Qwen3‑32B 上提升 7.5%，且比基线提升 6–7%；在四个基准任务上均显著优于基线，显示出跨任务泛化能力。

**⚠️ 局限性**

局限性：①阈值估计依赖训练集与工具模型的步骤质量评估，缺乏更鲁棒的阈值确定策略；②目前仅在 Web 领域验证，未探索其它领域或多工具扩展；③尚未研究其在代理强化学习训练过程中的加速或样本质量提升潜力。

---

## 344. Ministral 3

**arXiv ID:** 2601.08584 | [PDF](https://arxiv.org/pdf/2601.08584v1)

**作者:** Alexander H. Liu `[一作]`, Zaccharie Ramzi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了面向计算与内存受限场景的三尺寸（3B、8B、14B）密集语言模型系列，并分别发布基础、指令、推理版，模型具备图像理解与长达256K token的上下文能力。

**💡 创新点**

创新点在于：① 采用迭代剪枝‑蒸馏（prune‑distill‑repeat）策略，将大模型压缩为高效子模型；② 发现更强教师并非预训练的最佳选择，并系统研究教师版本对学生性能的影响；③ 引入在线直接偏好优化（ODPO）及GRPO等后训练手段提升模型对齐与推理能力。

**🔧 技术方法**

使用Transformer decoder架构，结合Grouped Query Attention、RoPE、SwiGLU、RMSNorm、YaRN长上下文扩展、ViT视觉编码器；预训练采用剪枝与logit蒸馏；后训练使用SFT、ODPO、GRPO与链式思考（CoT）数据。

**📊 数据集**

训练数据包括文本与图像混合大规模语料、指令遵循数据、Chain‑of‑Thought样本、数学、代码与多模态数据；评测使用MMLU、MATH、MMMU、MathVista、AIME、HMMT、PhyBench等公开基准。

**📈 对比分析**

通过统一评估管线与同规模开源模型（Qwen 3、Gemma 3）对比，结果显示14B/8B/3B Base模型在大部分基准上与同等参数模型竞争甚至超越，指令与推理版在对应任务上亦保持高水平性能。

**⚠️ 局限性**

局限性包括：预训练与后训练成本仍高；3B模型在推理后训练效果有限；多模态与推理任务仍略逊于更大模型；对齐过程中仍存在无限循环等问题；缺乏对硬件部署细节与真实推理成本的深入分析。

---

## 345. Enhancing Financial Literacy and Management through Goal-Directed Design and Gamification in Personal Finance Application

**arXiv ID:** 2601.08640 | [PDF](https://arxiv.org/pdf/2601.08640v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 346. REVNET: Rotation-Equivariant Point Cloud Completion via Vector Neuron Anchor Transformer

**arXiv ID:** 2601.08558 | [PDF](https://arxiv.org/pdf/2601.08558v1)

**作者:** Zhifan Ni `[一作]` (Technical University of Munich), Eckehard Steinbach `[通讯]` (Technical University of Munich)

**通讯引用:** 9825 | [OpenAlex ID](https://openalex.org/A5077346002)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于向量神经元（VN）的 SO(3)‑等变点云补全框架 REVNET，实现了在任意旋转姿态下对不完整点云的精确补全。

**💡 创新点**

创新点包括：①引入等变 Anchor Transformer 并使用通道差分注意力；②利用 VN 的等变‑不变转换生成稳定坐标；③改进 VN 偏置与 ZCA 层归一化以提升表达能力。

**🔧 技术方法**

技术包含：Vector Neuron 网络、等变 Transformer、通道差分注意力、VN‑Inv、VN‑ZCALayerNorm、以及多阶段等变特征提取与细粒度解码。

**📊 数据集**

主要使用合成 MVP 数据集进行 SO(3) 等变评估，且在真实 KITTI 点云上验证无需姿态对齐的鲁棒性。

**📈 对比分析**

与旋转变异与等变基准对比，REVNET 在 MVP 上几乎所有类别实现了最优 CD‑1 与 F‑Score，并且在旋转一致性上显著优于对比模型；在 KITTI 上获得与全局特征旋转变异方法相近的 FD/MMD 分数。

**⚠️ 局限性**

局限性包括：输入点稀疏（<10 点）时易失去结构信息；对极端旋转仍需进一步提升鲁棒性；与最新旋转变异方法相比，仍存在一定性能差距。

---

## 347. Cities at Play: Improving Equilibria in Urban Neighbourhood Games

**arXiv ID:** 2601.08642 | [PDF](https://arxiv.org/pdf/2601.08642v1)

**作者:** Martin Gairing `[一作]` (University of Liverpool), Zhanzhan Zhao `[通讯]` (Chinese University of Hong Kong)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

研究城市投资如何在居民策略性反应下提升社会福利，利用博弈论版Schelling局部邻域模型，证明小规模投资可让所有纳什均衡实现至少ε的最优福利。

**💡 创新点**

提出成本上限0.81 ε²的目标，证明通过目标化的、微小规模的干预即可保证社会福利至少为最优值的ε，从负面结果转化为正面收益的理论框架。

**🔧 技术方法**

博弈论建模、Schelling局部邻域模型、非凸效用函数、纳什均衡分析及严格的数学证明。

**📊 数据集**

无实验数据，全部为理论推导。

**📈 对比分析**

以理论证明和上界/下界分析对比，未进行实验对比；性能以ε比例衡量，保证所有均衡满足最低福利下限。

**⚠️ 局限性**

仅在假设的效用形式和博弈框架内成立，缺乏实证验证，忽略城市复杂性与动态变化，成本上限假设可能在实际中不现实。

---

## 348. Resisting Manipulative Bots in Memecoin Copy Trading: A Multi-Agent Approach with Chain-of-Thought Reasoning

**arXiv ID:** 2601.08641 | [PDF](https://arxiv.org/pdf/2601.08641v1)

**作者:** Yichen Luo `[一作]` (University College London), Yang Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 81937 | [OpenAlex ID](https://openalex.org/A5100355964)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种可解释的多智能体系统，用于 memecoin 复制交易。

**💡 创新点**

创新点在于将资产管理团队的结构拆解为子任务，并通过多智能体协作、few-shot chain-of-thought 提示以及可解释决策，解决了单一 LLM 在多模态、多任务下的局限。

**🔧 技术方法**

使用大语言模型 (LLM) 结合少量示例链式思维 (few-shot CoT) 提示，多智能体协同架构与可解释决策机制。

**📊 数据集**

使用包含 1,000 个 memecoin 项目交易数据的数据集。

**📈 对比分析**

与传统机器学习模型和单一 LLM 进行对比，系统在识别高质量 memecoin 项目和 KOL 钱包的精度分别为 73% 与 70%，且所选 KOL 总收益达 50 万美元，优于对照组。

**⚠️ 局限性**

局限性包括对数据规模与质量的高度依赖、无法完全排除操纵机器人的干扰、仅基于历史回测缺乏实时性能验证，以及可解释性效果仍需进一步实证。

---

## 349. How Order-Sensitive Are LLMs? OrderProbe for Deterministic Structural Reconstruction

**arXiv ID:** 2601.08626 | [PDF](https://arxiv.org/pdf/2601.08626v1)

**作者:** Yingjie He `[一作]` (Peking University), Jianyuan Ni `[通讯]` (Marquette University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了名为OrderProbe的基准，用于评估大型语言模型在将打乱顺序的四字成语恢复为标准顺序的能力。

**💡 创新点**

创新点在于：①选用四字表达作为唯一确定的评测单元，避免句子级恢复的多义性；②构建六维诊断框架（恢复率、语义一致、逻辑有效、结构一致、鲁棒性、信息密度），揭示语义记忆与结构规划的解耦；③在中文、日语、韩语等多写体上系统评测。

**🔧 技术方法**

技术包括：通过全排列生成所有非同位置换作为扰动；使用LLM生成释义并与词典定义对齐；采用BERTScore、跨语言NLI、信息密度等指标进行评估；在12款LLM上实现统一的两条输出格式，比较零样本与链式推理（CoT）效果。

**📊 数据集**

数据集为3,543条四字成语/惯用语，涵盖简体中文、繁体中文、日语、韩语四种文字体系，并按六种句法类别（平行、并列、主谓、宾谓、动宾、主宾）划分。

**📈 对比分析**

通过在零样本与CoT两种提示方式下对12款LLM进行评测，发现平均恢复率低于35%，CoT在部分模型上有提升但差异显著，整体表明结构恢复仍是LLM的瓶颈。

**⚠️ 局限性**

局限性包括：①评测仅针对已存在的四字表达，受限于模型的记忆和频率；②未涉及形态丰富或拼音文字体系；③缺乏对新组合或未见表达的泛化测试；④诊断框架主要关注语义与结构的分离，可能忽略其他鲁棒性因素。

---

## 350. SafeRedir: Prompt Embedding Redirection for Robust Unlearning in Image Generation Models

**arXiv ID:** 2601.08623 | [PDF](https://arxiv.org/pdf/2601.08623v1)

**作者:** Renyang Liu `[一作]` (National University of Singapore), See-Kiong Ng `[通讯]` (National University of Singapore)

**通讯引用:** 5208 | [OpenAlex ID](https://openalex.org/A5090171111)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于prompt嵌入重定向的轻量化推理时反向学习框架SafeRedir，用于在不修改模型的前提下实现图像生成模型的安全反向学习。

**💡 创新点**

创新点在于：①将安全检测与嵌入空间重定向结合，采用多模态（prompt、潜在图像、时间步）融合的轻量级分类器；②在token级别预测方向、缩放与掩码，实现最小化干预；③实现插件式、模型无关、可零拷贝迁移。

**🔧 技术方法**

技术包括：多模态交叉注意力融合、token级方向向量预测、可学习缩放因子、soft mask、LPIPS、CLIP评分、YOLO检测、攻击生成（UnlearnDiffAtk）。

**📊 数据集**

使用公开数据集：IGMU、I2P、MMA；以及基于Stable Diffusion v1.4自动生成的prompt–图像对，评估NSFW、VanGogh风格、Church对象三类任务。

**📈 对比分析**

与EASD、AdvUnlearn、MACE、UCE、RECE等20余种基线对比，SafeRedir在忘记率(≈99.8%)、内容保留(CSDR最低、LPIPS最低)、图像质量(FID、Q-Align、Laion_aes最高)、鲁棒性(ASR最小、攻击时间最长)上均超过或接近最优，且推理延迟<1.5%。

**⚠️ 局限性**

局限：仍需在推理时额外部署插件；对极端高维、跨模态复杂概念（政治敏感、隐蔽暴力）尚未充分验证；当目标概念与正常语义高度重叠时，重定向可能影响图像完整性。

---

## 351. Interpretability and Individuality in Knee MRI: Patient-Specific Radiomic Fingerprint with Reconstructed Healthy Personas

**arXiv ID:** 2601.08604 | [PDF](https://arxiv.org/pdf/2601.08604v1)

**作者:** Yaxi Chen `[一作]` (University College London), Yipeng Hu `[通讯]` (University College London)

**通讯引用:** 5059 | [OpenAlex ID](https://openalex.org/A5032309114)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

该研究提出一种融合生成式“健康人物”和个体化放射组学特征权重的框架，用于膝关节MRI的异常、前交叉韧带撕裂和半月板撕裂的自动诊断。

**💡 创新点**

通过预测每个病人的特征使用概率生成个体化放射组学指纹，并利用去噪扩散模型构造个体健康基线（persona），实现可解释性与高精度的协同提升。

**🔧 技术方法**

使用3D DDPM生成健康MRI、ResNet‑18基线的特征使用预测网络、Logistic回归分类器以及PyRadiomics提取形状与一阶统计特征。

**📊 数据集**

在公开的MRNet数据集（1370例）和克罗地亚Rijeka医院的ACL撕裂临床数据集（664例）上训练与评估。

**📈 对比分析**

与端到端DL基线MRNet和ELNet比较，采用ACC、Sens、Spe、AUC等指标；在ACL、半月板撕裂任务中均达到或超过基线AUC（ACL 0.85，半月板 0.84），同时保持较高特异性。

**⚠️ 局限性**

需要大量特征候选和注册预处理；高阶纹理特征虽提升性能但牺牲可解释性；健康人物的重建质量不必完美但若不足可能误导；模型对不同机构扫描协议的泛化仍有限。

---

## 352. Efficient Maintenance of Leiden Communities in Large Dynamic Graphs

**arXiv ID:** 2601.08554 | [PDF](https://arxiv.org/pdf/2601.08554v1)

**作者:** Chunxu Lin `[一作]` (Chinese University of Hong Kong), Chen Cheng `[通讯]` (ByteDance)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种高效的Leiden社区维护算法——Hierarchical Incremental Tree Leiden (HIT‑Leiden)，用于大规模动态图的社区更新；

**💡 创新点**

创新点在于从理论上分析了现有动态Leiden算法的无界性，并利用连通分量和层级社区结构设计了增量移动、增量细化、增量聚合三阶段，显著降低受影响节点范围；

**🔧 技术方法**

核心技术包括增量移动保持顶点最优性、增量细化维护γ‑连通子社区、增量聚合处理超级边变更、以及基于动态连通分量索引（如D‑Tree/HD T）的子社区维护；

**📊 数据集**

实验使用了五个真实动态图数据集（包括社交网络、合作网络等，具体数据集名称待从原文提取）；

**📈 对比分析**

与现有DF‑Leiden、DS‑Leiden等方法相比，HIT‑Leiden在保持相同或相近模量质量的同时，速度提升可达5个数量级；

**⚠️ 局限性**

局限性包括仅支持无向加权图、未考虑动态图的分布式实现、对极端频繁更新场景的适应性待验证。

---

## 353. Learner-Tailored Program Repair: A Solution Generator with Iterative Edit-Driven Retrieval Enhancement

**arXiv ID:** 2601.08545 | [PDF](https://arxiv.org/pdf/2601.08545v1)

**作者:** Zhenlong Dai `[一作]` (Zhejiang University), Jingyuan Chen `[通讯]` (Zhejiang University)

**通讯引用:** 3413 | [OpenAlex ID](https://openalex.org/A5090689233)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了“Learner‑Tailored Program Repair”（LPR）任务，构建了LSGen框架，能够同时生成修复后的代码和对应的bug描述，帮助学习者理解错误原因。

**💡 创新点**

创新点在于：①将解决方案检索与编辑驱动检索相结合，②利用差异(diff)和bug说明引导LLM生成修复；③通过迭代检索增强提升修复质量；④提出基于LLM的自动bug描述评估指标。

**🔧 技术方法**

使用的技术包括大型语言模型（如GPT‑4o、Claude‑4）、代码向量编码器、编辑驱动检索、diff‑based分析、迭代检索增强以及LLM评估机制。

**📊 数据集**

实验数据集来自ACP‑R测试集（含错误程序、修正程序与bug说明）与CodeNet构建的检索库，bug说明先由GPT‑4o生成后由人工复核。

**📈 对比分析**

通过与NoRef、AdaPatcher、PAR、PyDex、PyFiXV等基线比较，采用代码准确率与Bug‑F1评估；LSGen在GPT‑4o上实现91.4%准确率、38.5%Bug‑F1，明显优于其它方法。

**⚠️ 局限性**

局限性包括对大型语言模型的依赖、检索库覆盖度有限、评估指标可能无法完全捕捉bug描述的细微差异，且目前实验聚焦于教育编程场景。

---

## 354. Distribution Estimation with Side Information

**arXiv ID:** 2601.08535 | [PDF](https://arxiv.org/pdf/2601.08535v1)

**作者:** Haricharan Balasundaram `[一作]` (Indian Institute of Technology Madras), Andrew Thangaraj `[通讯]` (Indian Institute of Technology Madras)

**通讯引用:** 1553 | [OpenAlex ID](https://openalex.org/A5069723364)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在离散分布估计中引入侧信息，提出两种模型：局部信息模型和部分排序模型；

**💡 创新点**

创新点在于利用已知的分布邻域或符号概率排序信息，构造插值和两层 Good‑Turing 估计，理论上给出最优风险上界与下界；

**🔧 技术方法**

采用插值（shrinkage）估计、Good‑Turing 估计、Le Cam 与 Assouad 低界技术以及 Hellinger 距离等统计方法；

**📊 数据集**

实验使用 Project Gutenberg 书籍语料库中的 bigram 以及人工合成的两层分布；

**📈 对比分析**

与传统经验频率估计和单层 Good‑Turing 估计比较，实验显示在样本量小或符号概率分布显著分层时，侧信息估计的均方误差明显降低；

**⚠️ 局限性**

局限性包括：理论上对一般 π 的下界不够紧；对部分排序模型的下界尚未完成；需要对 Δ 的选择进行数据驱动调参。

---

## 355. Tailored Immersive Environments: Advancing Neurodivergent Support Through Virtual Reality

**arXiv ID:** 2601.08652 | [PDF](https://arxiv.org/pdf/2601.08652v1)

**作者:** Elia Moscoso-Thompson `[一作]` (CNR-IMATI), Chiara Malagoli `[通讯]` (CNR-ITD)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

设计并实现了一套基于VR的自适应城市过街训练系统EASE VR，并提出了基于加权特征的自动化情境个性化方法。

**💡 创新点**

通过将用户认知能力映射为特征权重，系统能在保证难度一致的前提下自动生成多样化的训练情境，实现真正的个性化与生态效度。

**🔧 技术方法**

采用虚拟现实渲染、高精度图形界面、线性加权求和与Jensen–Shannon散度评估技术，算法时间复杂度线性。

**📊 数据集**

以四个合成的用户配置文件为实验基准，没有使用真实数据集。

**📈 对比分析**

通过计算每个难度层级的场景数和特征方差指标，结果显示同一难度层级内场景多样性维持在高水平，且约87.5%的场景满足用户约束。

**⚠️ 局限性**

仅在合成配置下验证，缺乏真实 ASD 受试者反馈，评估指标有限，未来需结合实际使用者调研与更多多样化度量。

---

## 356. CtrlFuse: Mask-Prompt Guided Controllable Infrared and Visible Image Fusion

**arXiv ID:** 2601.08619 | [PDF](https://arxiv.org/pdf/2601.08619v1)

**作者:** Yiming Sun `[一作]` (Southeast University), Pengfei Zhu `[通讯]` (Xiong'an Guochuang Lantian Technology Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种可通过掩码提示交互控制的红外与可见光图像融合框架CtrlFuse；

**💡 创新点**

创新点在于将掩码提示动态编码为语义提示，显式注入多模态特征，并通过联合优化融合与分割任务实现双向互促；

**🔧 技术方法**

采用基于Transformer的多模态特征提取器、Reference Prompt Encoder、Prompt‑Semantic Fusion Module以及SAM模型进行掩码提示编码与分割；

**📊 数据集**

使用FMB、MSRS和DroneVehicle三个公开数据集，并在这些数据集上生成掩码进行训练与评估；

**📈 对比分析**

与8种SOTA方法对比，CtrlFuse在PSNR、N_abf、Q_abf等融合指标上取得领先，在语义分割和目标检测上也达成或接近最佳性能；

**⚠️ 局限性**

局限在于对掩码提示的依赖，若提示质量极差或缺失可能影响融合效果，同时需要SAM等预训练模型，增加计算成本与资源需求。

---

## 357. A Parallel Cross-Lingual Benchmark for Multimodal Idiomaticity Understanding

**arXiv ID:** 2601.08645 | [PDF](https://arxiv.org/pdf/2601.08645v1)

**作者:** Dilara Torunoğlu-Selamet `[一作]`, Zhuohan Xie `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建并公开了XMPIE数据集，收集了约10K条跨20余种语言的潜在成语表达，并为每条成语提供了五张标注图像（成语、近义、字面近义、字面、随机干扰），形成并行的多模态资源。

**💡 创新点**

创新点在于首次提供完整的跨语言并行成语基准，并细粒度划分成语对应关系（1‑1、1‑0、1‑N等），结合图论分析展示不同语言间的语义连通性，为多语言成语理解评估搭建了系统框架。

**🔧 技术方法**

技术方法包括：1）邀请89名语言专家对英语种子成语进行跨语言映射并撰写文本；2）使用Midjourney图像生成系统为每个成语生成五张图片；3）利用EVA‑CLIP‑18B无训练的检索式模型进行基线评估；4）运用图论指标衡量跨语言连通性。

**📊 数据集**

使用的数据集为基于英语种子成语扩展的多语言图文数据，覆盖Aromanian、Azeri、Bulgarian、Catalan、Chinese、Danish、Farsi、Georgian、Greek、Hebrew、Hungarian、Igbo、Indonesian、Italian、Japanese、Kazakh、Latvian、Lithuanian、Luxembourgish、Macedonian、Norwegian、Portuguese、Russian、Serbian、Slovak、Slovenian、Spanish、Swahili、Turkish、Ukrainian、Urdu、Uzbek等语言，共计约10,000+条成语与对应5张图片。

**📈 对比分析**

通过Top‑1、Top‑2准确率和NDCG@5评估模型在无上下文情景下的图像排序性能。实验显示，虽然NDCG@5普遍较高（0.87–0.95），但模型在识别成语图像方面明显落后于字面图像（Top‑1成语≈0.06–0.38，字面≈0.58–0.90），表明当前视觉‑语言模型在成语理解上仍面临挑战。

**⚠️ 局限性**

局限性包括：仅使用英语文本作为检索查询，缺乏上下文；图像生成模型偏向英语提示，可能影响非英语成语的视觉表达；数据集语言覆盖仍不完整，低资源语言样本不足；基线评估仅采用单一CLIP模型，未探讨更复杂的多模态或上下文增强方法；成语本身的抽象性与多义性难以通过简单标注完全捕捉。

---

## 358. Coverage-Guided Road Selection and Prioritization for Efficient Testing in Autonomous Driving Systems

**arXiv ID:** 2601.08609 | [PDF](https://arxiv.org/pdf/2601.08609v1)

**作者:** Qurban Ali `[一作]` (University of Milano-Bicocca), Oliviero Riganelli `[通讯]` (University of Milano-Bicocca)

**通讯引用:** 564 | [OpenAlex ID](https://openalex.org/A5015376844)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于道路几何与行为特征的自动驾驶系统（ADAS）测试优先级框架，能够在保留几何与行为多样性的前提下，过滤冗余测试场景并按风险度优先执行。

**💡 创新点**

创新点在于：①将道路切分为细粒度段落，结合曲率与动态指标构建行为特征；②使用DTW与动态距离进行段落匹配与层次聚类；③在测试选择中以覆盖最小化为目标，随后用多指标（几何复杂度、动态难度、历史失败）进行排序，实现跨模型的可迁移性评估。

**🔧 技术方法**

核心技术包括曲率计算与基于窗口的几何分段、动态时间规整（DTW）与相似度归一化、层次聚类（complete linkage）、多指标加权评分（α·G+β·D+H）以及基于APFD与EFD的性能评估。

**📊 数据集**

使用了OpenCat三大子集（Ambiegen、Frenetic、Frenetic_v）共32,580条道路场景，并在Udacity自驾仿真器上测试两款模型（DAVE‑2与Chauffeur）。

**📈 对比分析**

与随机顺序及单纯几何排序相比，所提方法在测试集缩减率上平均达到81–96%，失败案例覆盖率保持在57–100%；优先级排序使早期故障检测率提升约60–95×，APFD均值为0.82–0.97，显著优于基线。

**⚠️ 局限性**

局限性包括：①依赖预设阈值与权重，参数敏感性需进一步验证；②实验仅覆盖仿真环境，真实路况（噪声、交通、天气）影响仍待评估；③跨模型迁移性有限，行为特征对不同网络结构的适用性不完全，需针对各模型进行更细粒度的调优。

---

## 359. SfMamba: Efficient Source-Free Domain Adaptation via Selective Scan Modeling

**arXiv ID:** 2601.08608 | [PDF](https://arxiv.org/pdf/2601.08608v1)

**作者:** Xi Chen `[一作]` (Harbin Institute of Technology), Kui Jiang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于Mamba的源自由域适应框架（SfMamba），通过在视觉Mamba编码器后加插通道扫描块和背景补丁洗牌策略实现域不变特征学习。

**💡 创新点**

创新点在于①引入通道维度的双向状态空间扫描（Ch‑VSS）捕获频域一致性；②设计基于Grad‑CAM的背景补丁随机洗牌（SCS）降低序列误差；③将两者结合在Mamba结构上，兼顾长距离依赖与线性计算。

**🔧 技术方法**

使用了视觉Mamba（Visual Mamba）、双向状态空间模型、Grad‑CAM可视化、伪标签聚类、自信息最大化、KL一致性正则等技术。

**📊 数据集**

在四大基准上验证：Office、Office‑Home、VisDA‑C 与 DomainNet‑126。

**📈 对比分析**

与CNN、ViT、Mamba等现有SFDA方法对比，SfMamba在Office‑Home（81.7%）、VisDA‑C（89.3%）、DomainNet‑126（77.9%）等指标均取得新SOTA，并保持更低参数和FLOPs，表现优异。

**⚠️ 局限性**

局限性包括对极端风格域（如Clipart）仍有过拟合，缺乏对SSM为何能有效提取域不变特征的理论分析。

---

## 360. WaterCopilot: An AI-Driven Virtual Assistant for Water Management

**arXiv ID:** 2601.08559 | [PDF](https://arxiv.org/pdf/2601.08559v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 361. Estimating the True Distribution of Data Collected with Randomized Response

**arXiv ID:** 2601.08603 | [PDF](https://arxiv.org/pdf/2601.08603v1)

**作者:** Carlos Antonio Pinzón `[一作]` (INRIA Saclay), Catuscia Palamidessi `[通讯]` (INRIA Saclay)

**通讯引用:** 7493 | [OpenAlex ID](https://openalex.org/A5090842450)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

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

## 362. End-to-End Video Character Replacement without Structural Guidance

**arXiv ID:** 2601.08587 | [PDF](https://arxiv.org/pdf/2601.08587v1)

**作者:** Zhengbo Xu `[一作]` (AliBaBa Group), Jing Li `[通讯]` (AliBaBa Group)

**通讯引用:** 2724 | [OpenAlex ID](https://openalex.org/A5100683044)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出MoCha框架，实现仅使用单帧遮罩的端到端视频角色替换。

**💡 创新点**

创新点在于条件感知RoPE和RL后训练提升面部一致性，同时利用视频扩散模型的跟踪能力，消除对全帧分割与结构引导的需求。

**🔧 技术方法**

采用视频扩散（DiT）+ Flow匹配、RoPE扩展、RL奖励、LoRA等技术。

**📊 数据集**

使用三源数据集：Unreal Engine 5渲染对齐视频、基于Portrait动画的表情驱动视频、公开视频-遮罩数据（VIVID-10M、VPData）。

**📈 对比分析**

在合成与真实基准上与VACE、HunyuanCustom、Wan-Animate等方法比较，SSIM/LPIPS/PSNR/ VBench指标均优于现有方法，表现出更高的身份、时间一致性。

**⚠️ 局限性**

局限包括对单帧遮罩的精确定位要求、对大规模真实数据的依赖、以及在极端光照或极度复杂交互场景下仍可能出现细节失真。

---

## 363. In the Search for Good Neck Cuts

**arXiv ID:** 2601.08566 | [PDF](https://arxiv.org/pdf/2601.08566v1)

**作者:** Sam Ruggerio `[一作]` (University of Illinois), Sariel Har-Peled `[通讯]` (University of Illinois)

**通讯引用:** 9255 | [OpenAlex ID](https://openalex.org/A5040577923)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于等周不等式的新的 neck‑cut 定义，并给出了既可理论近似又可实际高效的两种算法，用于在三维网格上检测瓶颈曲线。

**💡 创新点**

创新点包括：①用等周比（紧致度）衡量“neck‑like”特征；②利用显著点（salient point）和基于最短路径的骨架（skeleton）快速限定搜索空间；③在强假设下提供 O(n^5) 的近似证明，并在实践中实现子二次的高效算法；④不依赖复杂的全局预处理或迭代平滑。

**🔧 技术方法**

技术方法主要涉及：几何与拓扑分析、等周不等式、最短路径算法（Dijkstra）、显著点检测、Steiner 树近似构造、区域面积累加与紧致度计算、基于 Polyscope 与 Geometry Central 的实现。

**📊 数据集**

实验数据集包括 Benchmark for 3D Mesh Segmentation Dataset（包含 Human、Armadillo、Octopus、Ant、Horse、Hand、Human2）以及 Stanford Bunny，所有模型均为 genus‑0 的三角网格。

**📈 对比分析**

与先前方法对比（虽无法完整重现其实现），本文在单线程 Intel i7-14700K 上的运行时间平均在 400‑800 ms 之间，显著低于需大量预处理的传统方法；算法性能与显著点数量高度相关，但在大多数测试模型上能在秒级完成；在 tightness 过滤后得到的 neck‑cut 质量与手工标注相近。

**⚠️ 局限性**

局限性包括：仅适用于 genus‑0 网格；对噪声/细节丰富表面可能产生误判；需要手动设定显著点过滤参数 r；理论近似证明基于较强的“well‑behaved”假设，实际应用中不保证全局最优；实现尚未完成所有潜在优化。

---

## 364. SVFusion: A CPU-GPU Co-Processing Architecture for Large-Scale Real-Time Vector Search

**arXiv ID:** 2601.08528 | [PDF](https://arxiv.org/pdf/2601.08528v1)

**作者:** Yuchen Peng `[一作]` (Zhejiang University), Gang Chen `[通讯]` (Zhejiang University)

**通讯引用:** 101834 | [OpenAlex ID](https://openalex.org/A5100389265)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SVFusion框架，实现GPU-CPU-磁盘协同的大规模实时向量搜索和动态更新。

**💡 创新点**

创新点包括：1) 三级层次（GPU‑CPU‑磁盘）索引结构；2) 工作负载感知向量置放（WAVP）缓存策略；3) 实时协调与并发控制（细粒度锁＋多版本同步）。

**🔧 技术方法**

技术手段：图基近邻搜索（KNNG）、改进CAGRA的GPU加速、CUDA多流、GPU内存分区、预测函数结合最近访问与邻居度、磁盘分区构建、细粒度锁+版本控制、批量插入、GPU/CPU距离并行计算等。

**📊 数据集**

使用的公开数据集：Wikipedia、MSMARCO、MSTuring、Deep1B、Text2Image。

**📈 对比分析**

与HNSW、FreshDiskANN、CAGRA、PilotANN、GGNN等基线对比，SVFusion在搜索吞吐率提升至20.9×、插入吞吐率提升至71.8×，延迟比基线低1.3×至50.7×，召回率保持在91%–96%。

**⚠️ 局限性**

局限性：仍受GPU内存大小限制，磁盘合并阶段是瓶颈；逻辑删除与全局重构需要权衡；极高负载下CPU‑GPU同步开销仍显著；对GPU内存比例和缓存阈值敏感。

---

## 365. Safe Language Generation in the Limit

**arXiv ID:** 2601.08648 | [PDF](https://arxiv.org/pdf/2601.08648v1)

**作者:** Antonios Anastasopoulos `[一作]` (George Mason University), Evgenios M. Kornaropoulos `[通讯]` (George Mason University)

**通讯引用:** 445 | [OpenAlex ID](https://openalex.org/A5030196581)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

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

## 366. FPT Approximations for Connected Maximum Coverage

**arXiv ID:** 2601.08639 | [PDF](https://arxiv.org/pdf/2601.08639v1)

**作者:** Tanmay Inamdar `[一作]` (Indian Institute of Technology Jodhpur), Meirav Zehavi `[通讯]` (Ben-Gurion University of the Negev)

**通讯引用:** 1213 | [OpenAlex ID](https://openalex.org/A5082025487)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并研究了一个通用的连通约束覆盖模型，统一描述了 Max Coverage、Partial Vertex Cover、Partial Dominating Set 等多种变种。

**💡 创新点**

核心创新在于将连通性约束与覆盖层分离，并构造了“邻域稀疏化”技术，生成一族可近似保留所有 k 规模集合邻域的权重函数，从而得到高效的参数化近似算法。

**🔧 技术方法**

主要技术包括：参数化稀疏化（基于随机分离与贪心），最大权重 k‑树算法（指数时间 2^k n^O(1)），以及对称性利用的归约到 Relaxed Directed Steiner Out‑Tree，结合 FPT 近似框架。

**📊 数据集**

论文为理论性工作，未使用具体实验数据集；所有结果均通过数学证明给出。

**📈 对比分析**

在满足“biclique‑free”或“有限 VC‑维”等结构限制下，提供了 (1,1‑ε) 与 (1+ε,1) 两种参数化近似方案，时间复杂度分别为 2^(k²d/ε)·n^O(1) 与 2^(k d(k²+log d))/ε · n^O(1)，远优于已知的 (logΔ)‑近似或无多项式近似结果。

**⚠️ 局限性**

局限性包括：仍未对所有结构化图类给出完整的可行与不可行划分，且在更一般的覆盖图（如含大二分团）上难以突破 W[1]/W[2] 难点，未来需进一步探索更细粒度的 FPT‑近似阈值。

---

## 367. On the Optimality of Decode and Forward for Some Cooperative Broadcast Channels

**arXiv ID:** 2601.08592 | [PDF](https://arxiv.org/pdf/2601.08592v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 368. M$^2$FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting

**arXiv ID:** 2601.08631 | [PDF](https://arxiv.org/pdf/2601.08631v1)

**作者:** Yaohui Huang `[一作]` (Central South University), Ruipeng Dong `[通讯]` (Central South University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种多视角频域混合专家模型 M^2FMoE，用来在水位等时间序列中同时捕获常规趋势与极端事件的动态，并通过多分辨率融合与门控整合提升预测精度。

**💡 创新点**

创新点在于：
• 采用交叉视角共享频段拆分（CSS）将 Fourier 与 Wavelet 频域对齐，保证专家在两视角上对同一频带负责；
• 通过多分辨率自适应融合逐级聚合粗细尺度特征；
• 引入专家多样性与一致性正则化，促进专家间互补与跨视角一致性；
• 采用门控整合历史与近期预测，动态权衡长短期信息。

**🔧 技术方法**

主要技术包括：离散傅里叶变换、连续小波变换、混合专家路由网络、跨视角共享频段拆分、卷积块处理频谱、门控整合机制、MSE+专家多样性+一致性损失的联合优化。

**📊 数据集**

使用五个公开水库水位数据集：Almaden、Coyote、Lexington、Stevens Creek、Vasona（每个数据集均为 1991‑2019 年的小时级水位记录）。

**📈 对比分析**

与 9 个基线（CATS、TQNet、iTransformer、FreqMoE、Umixer、KAN、CycleNet、DAN、MCANN）在 8h 与 72h 预测窗口下进行对比，M^2FMoE 在 RMSE 上均取得显著优势（平均提升 22% 以上），并在大多数数据集上获得最佳平均排名；统计检验显示改进显著。

**⚠️ 局限性**

局限性包括：
• 需要手动设置最近段长度、专家数量和分辨率级数，影响模型泛化；
• 对于极端事件的稀缺性仍有欠缺的鲁棒性，模型对极端事件的识别仍受限；
• 目前仅在水位数据上验证，缺乏跨领域（如电力、交通等）的广泛评估；
• 计算量相对较大，尤其在双频域和多分辨率处理上。

---

## 369. SoC: Semantic Orthogonal Calibration for Test-Time Prompt Tuning

**arXiv ID:** 2601.08617 | [PDF](https://arxiv.org/pdf/2601.08617v1)

**作者:** Leo Fillioux `[一作]` (CentraleSupélec), Jose Dolz `[通讯]` (ÉTS Montréal)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Semantic Orthogonal Calibration（SoC）方法，用于Vision‑Language模型的测试时提示调优，解决全正交约束导致的过度自信问题。

**💡 创新点**

创新点在于采用Huber损失对文本原型进行平滑正则化，既保留语义相似性又显著提升模型校准性能。

**🔧 技术方法**

利用CLIP VLM、Huber正则化、单步梯度更新的测试时提示调优技术，并结合温度缩放等校准手段。

**📊 数据集**

在11个细粒度分类数据集、4个ImageNet变体以及EuroSAT等多种数据集上进行评估。

**📈 对比分析**

与TPT、C‑TPT、O‑TPT等基线比较，SoC在保持或提升准确率的同时，ECE显著下降，表现出更优的校准效果。

**⚠️ 局限性**

局限性包括仍受初始提示模板影响，对多步梯度更新敏感，以及在极端分布漂移场景下鲁棒性需进一步验证。

---

## 370. Formalization and Implementation of Safe Destination Passing in Pure Functional Programming Settings

**arXiv ID:** 2601.08529 | [PDF](https://arxiv.org/pdf/2601.08529v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 371. WaveFormer: Frequency-Time Decoupled Vision Modeling with Wave Equation

**arXiv ID:** 2601.08602 | [PDF](https://arxiv.org/pdf/2601.08602v1)

**作者:** Zishan Shu `[一作]` (Peking University), Jie Chen `[通讯]` (Peking University)

**通讯引用:** 46083 | [OpenAlex ID](https://openalex.org/A5100694761)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于阻尼波方程的特征传播方法——Wave Propagation Operator（WPO），通过频率‑时间解耦实现全局语义的振荡传递，并构建WaveFormer视觉骨干网络；

**💡 创新点**

创新点在于用物理波动力学替代传统热扩散与自注意力，实现高频细节与低频全局语义并存的可控传播，并以FFT实现O(N log N)复杂度；

**🔧 技术方法**

采用阻尼波方程的解析解、二维傅里叶变换、频率调制与可学习的波速、阻尼参数，构成WPO模块；

**📊 数据集**

在ImageNet‑1K（分类）、COCO（目标检测与实例分割）和ADE20K（语义分割）等公开数据集上进行评估；

**📈 对比分析**

与Swin、ConvNeXt、vHeat等主流CNN/ViT骨干比较，WaveFormer在保持或提升Top‑1/ mIoU/AP 的同时，FLOPs更低、推理速度提升约30–50%；

**⚠️ 局限性**

局限在于需手动或自动调节波速与阻尼超参数，对极端长程依赖或非视觉任务的适用性尚未充分验证。

---

## 372. Intersectional Data and the Social Cost of Digital Extraction: A Pigouvian Surcharge

**arXiv ID:** 2601.08574 | [PDF](https://arxiv.org/pdf/2601.08574v1)

**作者:** Eduardo C. Garrido-Merchán `[一作]` `[通讯]` (Universidad Pontificia Comillas), Eduardo C. Garrido-Merchán (Universidad Pontificia Comillas)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

本文提出了一种基于信息理论的Pigouvian增税机制，用以对数字平台对跨层面身份（intersectional identity）数据提取的社会外部性进行内部化；

**💡 创新点**

创新点在于将互信息（mutual information）与交叉身份（intersectionality）相结合，构建一个模型无关、可计量的隐私外部性定价规则，并将该规则解释为对信息权力积累的制度约束；

**🔧 技术方法**

使用了信息理论中的互信息度量、熵（entropy）以及Pigouvian定价框架；

**📊 数据集**

论文未使用具体数据集，侧重理论推导与概念框架构建；

**📈 对比分析**

由于缺乏实验或数据对比，本文并未提供传统意义上的性能指标，而是通过逻辑和经济论证说明该增税可使平台在面对交叉身份预测时面临额外成本，进而抑制过度剖析；

**⚠️ 局限性**

局限性包括：对监管执行与透明度要求高，若平台拥有强大市场力量可能通过价格转嫁或规避；该机制无法单独解决数字资本主义中的结构性权力不平等，需要与更广泛的治理与再分配措施配合；

---

## 373. DeepResearch Bench II: Diagnosing Deep Research Agents via Rubrics from Expert Report

**arXiv ID:** 2601.08536 | [PDF](https://arxiv.org/pdf/2601.08536v1)

**作者:** Ruizhe Li `[一作]` (University of Science and Technology of China), Zhendong Mao `[通讯]` (University of Science and Technology of China)

**通讯引用:** 5931 | [OpenAlex ID](https://openalex.org/A5023341829)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Deep Research Bench II，构建了132个基于真实专家调查报告的任务，并通过四阶段LLM+人工流程提取了9,430条细粒度可验证的二进制rubrics，用于评估深度研究系统的报告质量。

**💡 创新点**

创新点在于：① 采用真实专家撰写的公开报告为源，保证评估基准的权威性与可验证性；② 将评估拆分为信息回忆、分析、展示三维度，形成细粒度的rubric体系；③ 通过LLM自评、人工修订和专家审查相结合的四阶段管道，系统性生成高质量rubrics；④ 使用LLM‑as‑judge进行统一评估，并在批量大小与评估模型上做消融验证。

**🔧 技术方法**

技术手段包括：LLM（如Gemini‑2.5‑Pro）生成任务与rubrics、进行自评循环；LLM作为判定者进行二分类评估；人工审阅与专家优化以确保rubrics的准确性；评估脚本实现批量化评估并计算每维度得分。

**📊 数据集**

数据集来源为132篇经过版权检查（CC‑4.0/CC‑4.0‑NC）的公开专家调查报告，涵盖22个领域；从每篇报告提炼出9,430条细粒度rubrics，总任务数132。

**📈 对比分析**

通过对每条rubric的二分类通过率计算每个任务的得分，并按三维度聚合；在此框架下对多款顶尖深度研究系统（OpenAI‑GPT‑3.5/4、Gemini‑3‑Pro、Gemini‑2.5‑Pro、Doubao、Qwen3‑Max、Grok、Perplexity、Tongyi）进行基准测试，发现最强模型仅满足不到50%的rubrics，信息回忆和分析维度表现尤为薄弱，而展示维度表现相对较好；此外还验证了评估批量大小与评估LLM的影响。

**⚠️ 局限性**

局限性包括：① 无法完全阻止模型接触到源报告导致泄漏风险；② LLM‑as‑judge的判断仍受模型知识与偏差影响，尤其在信息检索与推理方面；③ 评估的展示维度尚未覆盖用户自适应呈现；④ 人类标注过程存在主观性与偏差；⑤ 对闭源模型的搜索结果控制有限。

---

## 374. Sampling via Stochastic Interpolants by Langevin-based Velocity and Initialization Estimation in Flow ODEs

**arXiv ID:** 2601.08527 | [PDF](https://arxiv.org/pdf/2601.08527v1)

**作者:** Chenguang Duan `[一作]`, Ruizhe Zhang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文提出了一种基于线性随机插值的概率流ODE采样框架（SSI），通过Langevin扩散实现初始化分布和速度场的在线估计，完成对未归一化Boltzmann分布的高效采样。

**💡 创新点**

创新点在于将多模态目标分布的采样拆分为易于采样的中间分布与速度场的两步；利用Tweedie公式在每一步用Langevin采样估计条件期望，并引入RMSprop预处理显著提升了在高维与复杂势能面上的收敛速度。

**🔧 技术方法**

主要技术包括线性随机插值产生的概率流ODE、Tweedie公式的条件期望估计、Euler-Maruyama离散化的无调整和调整的Langevin采样、以及RMSprop预处理的Langevin动力学。

**📊 数据集**

在二维任务上使用了三种经典混合高斯分布（七七格点、40个随机高斯、一个特殊多环分布）；在高维任务上使用了高维多模态分布；在贝叶斯任务上使用了一维高斯混合模型的聚类中心估计。

**📈 对比分析**

与ULA、MALA、pULA（RMSprop预处理的ULA）和HMC进行对比，SSI在NLL、MMD、2-Wasserstein距离上均表现最优，尤其在多模态分布中能完全捕捉所有模态并恢复相对权重。

**⚠️ 局限性**

局限性包括对初始化时间T_0的敏感性（需在一定区间内取值）以及在极端高维或非常弱光滑的分布中可能仍面临采样效率下降。

---

## 375. Spatial Context Improves the Integration of Text with Remote Sensing for Mapping Environmental Variables

**arXiv ID:** 2601.08750 | [PDF](https://arxiv.org/pdf/2601.08750v1)

**作者:** Valerie Zermatten `[一作]` (Ecole Polytechnique Federale de Lausanne), Devis Tuia `[通讯]` (Ecole Polytechnique Federale de Lausanne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过将来自瑞士的高分辨率航空影像与来自维基百科的物种栖息地文本结合，并使用注意力机制在空间邻域内融合多模态特征，预测103个环境变量。

**💡 创新点**

创新点在于提出基于注意力的多模态融合框架，结合位置编码和空间邻域信息，动态选择有用的邻居，同时通过文本-图像相似度挑选最相关句子，实现稀疏文本与连续遥感数据的有效融合。

**🔧 技术方法**

使用了预训练的Vision‑Language模型（SkyCLIP‑ViT‑B）作为视觉与文本编码器，位置编码（极坐标）以及Transformer‑based注意力融合模块，并对输入进行随机token掩蔽。

**📊 数据集**

实验基于EcoWikiRS（航空影像+维基百科文本）与SWECO25（103个环境变量）这两个瑞士地区的数据集。

**📈 对比分析**

与单一模态、单一位置以及不使用空间上下文的基线相比，最佳模型在测试集上R²达0.50，明显提升；加入空间邻居可进一步提升性能，文本与影像协同作用明显。

**⚠️ 局限性**

局限性包括：仅覆盖瑞士且缺乏全球尺度的环境变量数据；仅使用英语维基百科文本，语言多样性不足；文本稀疏且地理定位不精确，可能导致空间过拟合和偏倚。

---

## 376. The Unification Type of an Equational Theory May Depend on the Instantiation Preorder: From Results for Single Theories to Results for Classes of Theories

**arXiv ID:** 2601.08710 | [PDF](https://arxiv.org/pdf/2601.08710v1)

**作者:** Franz Baader `[一作]` (Dresden University of Technology), Oliver Fernández Gil `[通讯]` (Dresden University of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

研究了等价理论在不同实例化偏序（受限与非受限）下的统一类型，并给出了广泛理论类的无穷性/有限性结果。

**💡 创新点**

首次提出在受限与非受限偏序之间，统一类型可能变好或变差的通用元理论，并给出对正则、有限、局部有限及受限单纯型理论的完整判定。

**🔧 技术方法**

使用偏序的极限与最小完全集的序理论，结合正则性、有限性与单张性等代数性质，构造了上下界证明。

**📊 数据集**

无实验数据集，全部为理论证明。

**📈 对比分析**

通过证明最小完全集的极限与实例化偏序包含关系，对比受限与非受限的统一类型，展示在特定理论中类型的提升或下降；由于理论性质，未给出运行时性能指标。

**⚠️ 局限性**

局限在于对非正则或非受限单纯型理论缺乏完整结果，且未给出实现可枚举最小完全集的算法。

---

## 377. Multi-Preconditioned LBFGS for Training Finite-Basis PINNs

**arXiv ID:** 2601.08709 | [PDF](https://arxiv.org/pdf/2601.08709v1)

**作者:** Marc Salvadó-Benasco `[一作]` (Università della Svizzera Italiana), Alena Kopaničáková `[通讯]` (Toulouse INP-ENSEEIHT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了多预条件化LBFGS（MP-LBFGS）算法，用于加速有限基物理信息神经网络（FBPINN）的训练。

**💡 创新点**

创新点在于结合非线性加性Schwarz预条件和低维子空间最小化机制，将各子域的LBFGS搜索方向有效聚合，从而降低通信开销并提升收敛速度。

**🔧 技术方法**

采用FBPINN域分解架构、LBFGS与右预条件化、子空间最小化求解、线搜索与并行子域优化等技术。

**📊 数据集**

使用一维和二维Poisson方程（已知解析解）以及Burgers方程的合成数据集，采样点通过Hammersley方法生成。

**📈 对比分析**

与传统LBFGS对比，评估指标包括训练轮数（epochs）、有效梯度评估次数（#g_e）和验证L2误差；MP-LBFGS在通信成本、训练轮次及误差上均优于LBFGS，误差可降低至一阶数量级。

**⚠️ 局限性**

局限性包括对子域分解的依赖、在极端非线性或大子域数下可能出现性能退化（如Burgers 3×3案例）、子空间最小化的额外计算开销以及对通信模型的简化假设。

---

## 378. "Where is My Troubleshooting Procedure?": Studying the Potential of RAG in Assisting Failure Resolution of Large Cyber-Physical System

**arXiv ID:** 2601.08706 | [PDF](https://arxiv.org/pdf/2601.08706v1)

**作者:** Maria Teresa Rossi `[一作]` (University of Milano-Bicocca), Paolo Gavazzo `[通讯]` (University of Milano-Bicocca)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 Retrieval Augmented Generation（RAG）在大型工业网络物理系统（CPS）故障排除手册中的应用，构建了基于自然语言问答的对话接口，帮助运维人员快速检索和生成合适的排障步骤。

**💡 创新点**

创新点包括：①将语义检索与 LLM 生成结合，首次在工业技术手册的自然语言文本中实现半自动排障建议；②系统支持跨验证，通过返回检索来源来提醒用户核对；③在缺失手册信息时，利用 LLM 生成潜在的未记录排障步骤；④对多种 LLM、检索粒度与采样参数进行系统评估，揭示速度与准确性的权衡。

**🔧 技术方法**

技术方案：使用 LangChain+Chroma 进行向量检索，Ollama 容器部署多模型（Qwen、Mistral、Llama）；embedding 与 chunking（400/800/1000 字符）并用余弦相似度检索；RAG 架构将检索块作为前缀提示 LLM；LLM‑as‑a‑jury 评估采用 Gemma、Granite、Hermes 三个判别模型；使用 BLEU、Jury 分数等指标评估。

**📊 数据集**

数据集：来自 Fincantieri 的 100+ 技术手册，挑选 25 个主题共 100 条自然语言问答；手册已转为 XML 并分块，构成知识库；三种 KB 配置（完整、无响应、无条目、无 KB）用于测试 RAG 的推导能力。

**📈 对比分析**

比较方法：对 5 个研究问题（准确性、敏感性、推导、定性评估、性能）进行实验，系统评估不同 LLM、chunk size 与 top‑p 的影响；结果显示 Qwen 在准确性和敏感性上最优，Llama 在时间上最快；完整 KB 下错误率仅 8%，缺失 KB 时误差大；平均响应时间 2.9–14.2 秒，表明存在速度‑准确性权衡。

**⚠️ 局限性**

局限性：①实验规模仅 25 主题，难以覆盖全部工业查询；②评估依赖 LLM 判别器，缺乏真实运维人员验证；③对多条件、模糊描述的问句仍易出现错误；④仅处理文本手册，未考虑多模态（图像、视频）排障信息；⑤系统需要手动交叉验证，无法完全自动化。

---

## 379. All Required, In Order: Phase-Level Evaluation for AI-Human Dialogue in Healthcare and Beyond

**arXiv ID:** 2601.08690 | [PDF](https://arxiv.org/pdf/2601.08690v1)

**作者:** Shubham Kulkarni `[一作]` (Interactly), Preetam Joshi `[通讯]` (AIMon Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种面向临床对话的阶段级合规评估方法OIP–SCE，评估对话是否满足所有必要信息义务并按规定顺序完成

**💡 创新点**

将合规性从逐句评分转为阶段依赖图检查，结合Coverage和OrderSafe两个判定标准，并提供可审计的单行表格和可视化证据

**🔧 技术方法**

利用阶段依赖图（有向无环图）、规则推理、低精度锚点和LLM辅助裁判相结合的评估管线；实现线性时间的Coverage/OrderSafe计算

**📊 数据集**

使用公开的医生–患者对话数据集MediTOD（呼吸病历）和内部匿名的AI–人类保险查询通话数据进行案例验证

**📈 对比分析**

未给出数值指标，只说明在案例中所有必要阶段通过Coverage，OrderSafe可捕获顺序违规；对比传统逐句准确率方法，OIP–SCE能发现全局违规，提升监管符合度

**⚠️ 局限性**

局限包括需要人工标注阶段边界、对长通话边界检测的鲁棒性、对多语言和多模态的通用性不足，以及对实时监控的挑战

---

## 380. QuantEval: A Benchmark for Financial Quantitative Tasks in Large Language Models

**arXiv ID:** 2601.08689 | [PDF](https://arxiv.org/pdf/2601.08689v1)

**作者:** Zhaolu Kang `[一作]` (Peking University), Richeng Xuan `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布 QuantEval 基准，用于系统评估 LLM 在金融定量任务（知识问答、定量推理、策略编码）上的能力，并对 13 个开源/专有模型进行评测，随后通过 SFT 与 RL 进一步提升性能。

**💡 创新点**

三维评估框架与 CTA‑style 回测相结合，既检验模型生成的策略能否编译、执行，又量化其回测指标（可执行率、Return MAE 等），并引入专家+多代理生成与验证流程，保证数据质量与可复现性。

**🔧 技术方法**

采用多代理生成、Chain‑of‑Thought Prompting、SFT、Group Relative Policy Optimization (GRPO) 强化学习、CTA 回测框架与自动验证脚本，并结合专家人工审核。

**📊 数据集**

核心数据集共 1,575 条样本（660 QA、855 推理、60 策略）；数据来源包括金融教材、学术论文、监管文件、公开市场数据、开源策略库；训练时使用约 57,000 条金融文本与案例，包含 Agentar‑DeepFinance‑100K、DianJin‑R1‑Data、FinQA 等公开数据。

**📈 对比分析**

在统一 Prompting（CoT/无 CoT）和执行基准上对 13 个 LLM 进行评测。人类专家约 90%+，开源模型 QA 60–75%，推理 30–45%，策略编码可执行率 0%（开源）到 10–63%（专有），SFT+RL 在某些模型可提升 4–10% 但仍落后于人类；专有模型整体优于开源。

**⚠️ 局限性**

局限包括仅英文样本、策略编码样本数量少（60 条）、固定的 CTA 回测配置（成本、滑点、资产范围），不同配置会影响指标，且不涵盖多语言/多市场场景。

---

## 381. Why AI Alignment Failure Is Structural: Learned Human Interaction Structures and AGI as an Endogenous Evolutionary Shock

**arXiv ID:** 2601.08673 | [PDF](https://arxiv.org/pdf/2601.08673v1)

**作者:** Didier Sornette `[一作]` (Southern University of Science and Technology), Ke Wu `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 7688 | [OpenAlex ID](https://openalex.org/A5081728299)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对LLM行为的社会学、经济学与系统动力学分析，将传统的道德对齐议题重新定位为结构治理问题，并将LLM视为人类互动模式与权力关系的放大器。

**💡 创新点**

创新点在于：①将被视为异常的黑幕、威胁等行为归纳为人类交换规律的极端表现；②提出“关系偏差映射”和“放大阈值监测”等治理框架；③强调对齐应聚焦于部署与治理结构而非单纯道德规则。

**🔧 技术方法**

主要采用关系模型理论、系统动力学视角、AI安全概念框架进行理论建模与阐释；未涉及算法实现。

**📊 数据集**

本文未使用具体数据集，主要引用公开论文、案例研究与历史数据进行论证。

**📈 对比分析**

由于缺乏实验评估，无法给出数值性能；作者通过理论推导与案例比较阐述观点的合理性。

**⚠️ 局限性**

局限性在于缺乏实证验证与可操作性评估，模型假设过于抽象，难以直接转化为工程实现，且对实际治理措施的具体落地缺乏细节。

---

## 382. NEVO-GSPT: Population-Based Neural Network Evolution Using Inflate and Deflate Operators

**arXiv ID:** 2601.08657 | [PDF](https://arxiv.org/pdf/2601.08657v1)

**作者:** Davide Farinati `[一作]` (Vita-Salute San Raffaele University), Mauro Castelli `[通讯]` (NOVA Information Management School)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种融合变异、重组与剪枝的进化式神经网络架构搜索方法，用于提升回归任务的性能

**💡 创新点**

创新点在于将重组与剪枝两种新型算子与传统变异相结合，形成更高效的搜索策略

**🔧 技术方法**

主要采用进化计算（遗传算法）与网络剪枝技术进行架构搜索与优化

**📊 数据集**

实验使用了UCI公开数据集中的bio、chem、comp、image、time等五个回归数据集

**📈 对比分析**

与随机搜索、传统NAS等方法进行对比，实验结果表明新方法在RMSE和MAE指标上均优于对照组

**⚠️ 局限性**

缺点是仅在规模较小的数据集上验证，且搜索过程计算成本较高

---

## 383. On the Algebraic Structure Underlying the Support Enumerators of Linear Codes

**arXiv ID:** 2601.08744 | [PDF](https://arxiv.org/pdf/2601.08744v1)

**作者:** Nitin Kenjale `[一作]` (K. J. Somaiya Institute of Technology), Anuradha S. Garge `[通讯]` (University of Mumbai)

**通讯引用:** 50 | [OpenAlex ID](https://openalex.org/A5005613380)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出支持分布与支持枚举概念，推导支持枚举器的MacWilliams式及其在自双重码中的判定条件。

**💡 创新点**

通过坐标级信息提升传统重量枚举的细粒度，首次给出支持枚举器的对偶关系和自双重码的必要判定。

**🔧 技术方法**

利用有限域的加法角色性、正交性理论和线性码的对偶性质进行推导。

**📊 数据集**

示例数据集包括二元/多元简单形码、汉明码、重复码与扩展汉明码。

**📈 对比分析**

无实验比较；该工作主要为理论证明，未给出性能指标。

**⚠️ 局限性**

仅适用于线性码，对非线性码或更一般的编码结构缺乏推广。

---

## 384. From Rows to Reasoning: A Retrieval-Augmented Multimodal Framework for Spreadsheet Understanding

**arXiv ID:** 2601.08741 | [PDF](https://arxiv.org/pdf/2601.08741v1)

**作者:** Anmol Gulati `[一作]` (PricewaterhouseCoopers), Kevin Paul `[通讯]` (PricewaterhouseCoopers)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了FRTR框架，利用多模态检索增强生成方法对大型企业级Excel工作簿进行推理。

**💡 创新点**

创新点包括：①细粒度行/列/块/图像索引；②混合词典-稠密检索与递归排名融合（RRF）以提高检索鲁棒性；③统一的文本-视觉嵌入，兼顾数值与视觉证据。

**🔧 技术方法**

技术手段包括：多模态向量数据库、Amazon Titan Multimodal编码器、RRF融合、LLM（如Claude Sonnet 4.5、GPT‑5）对检索结果进行推理与生成。

**📊 数据集**

使用了FRTR‑Bench数据集：30份企业Excel工作簿，约392万单元格、210万行、53张嵌入图片，涵盖跨表引用和多模态内容。

**📈 对比分析**

在FRTR‑Bench上与SpreadsheetLLM及完整上下文基线对比，FRTR达成74%答案准确率（相比24%提升显著），在单表Benchmark上达到87%准确率，且令token使用减少约50%，推理延迟与Baseline相当或更低。

**⚠️ 局限性**

局限性在于：对单表、纯文本任务的性能仍不如最先进的压缩方法；检索召回仍受索引粒度与模型嵌入质量限制；未实现自动适应检索深度与多模态对齐的动态机制。

---

## 385. PrivGemo: Privacy-Preserving Dual-Tower Graph Retrieval for Empowering LLM Reasoning with Memory Augmentation

**arXiv ID:** 2601.08739 | [PDF](https://arxiv.org/pdf/2601.08739v1)

**作者:** Xingyu Tan `[一作]` (University of New South Wales), Wenjie Zhang `[通讯]` (University of New South Wales)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一个隐私保护的KG增强式推理框架 PrivGemo，使用双塔结构将原始KG保留在本地，远程LLM仅处理匿名视图，并结合经验记忆实现多跳多实体推理。

**💡 创新点**

创新点在于：1）通过结构去歧义的匿名化实现语义与结构双重隐私保护；2）采用指示器引导的长链路径检索，保证多跳全局最优；3）引入记忆驱动的远程调用控制，显著降低隐私泄露与调用成本。

**🔧 技术方法**

技术手段包括：双塔双LLM架构、会话级实体关系匿名化、结构脱识别化、指示器引导的长链检索、经验记忆检索与强化学习式控制、Beam搜索剪枝等。

**📊 数据集**

在六大KGQA基准上评测：CWQ、WebQSP、GrailQA、Simple Questions、WebQuestions、QALD‑10 等。

**📈 对比分析**

与现有KG‑RAG和隐私友好方法（ARoG、ToG 等）对比，平均提升 12.4%–30.2%，多数数据集超过 GPT‑4 强基线，在严格的私密设置下仍保持高准确率。

**⚠️ 局限性**

局限性在于仅支持文本KG证据，未考虑多模态信息；匿名化对极细粒度知识的推理仍可能产生一定损失。

---

## 386. Revisiting "Revisiting Neuron Coverage for DNN Testing: A Layer-Wise and Distribution-Aware Criterion": A Critical Review and Implications on DNN Coverage Testing

**arXiv ID:** 2601.08729 | [PDF](https://arxiv.org/pdf/2601.08729v1)

**作者:** Jinhan Kim `[一作]` (Università della Svizzera italiana), Paolo Tonella `[通讯]` (Università della Svizzera italiana)

**通讯引用:** 10920 | [OpenAlex ID](https://openalex.org/A5025438762)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对Yuan等人提出的Neural Coverage (NLC) 进行批判性评审，系统分析其理论假设与实证方法，并在实验中验证NLC在单调性、顺序独立性、层级信息、无上界等方面的缺陷，提出改进思路；

**💡 创新点**

创新点在于：①首次从覆盖准则核心属性出发，对NLC进行全面剖析；②通过实验揭示预设“真值”排序的局限；③提出利用协方差矩阵行列式等替代度量，保留矩阵结构特性；④强调层级信息的可解释性与聚合方式的改进；

**🔧 技术方法**

技术手段包括统计分析、数值实验（对不同模型、数据集计算并比较NLC值），使用JS散度、谱分析衡量多样性；对输入顺序、层级贡献等做多次随机化实验；

**📊 数据集**

使用CIFAR-10与ImageNet上的预训练模型VGG16_BN、ResNet50、MobileNetV2，构造原始、白噪声扩充、PGD/CW攻击等多组测试集；

**📈 对比分析**

通过对原始、噪声扩充、对抗样本等多种测试集分别计算NLC及其他覆盖指标，比较其与多样性或缺陷揭示排序的一致性；实验表明NLC对输入顺序敏感、层级贡献失衡，无法满足期望的覆盖目标；

**⚠️ 局限性**

限制在于：①NLC缺乏单调性与顺序独立性，覆盖值无上界，无法作为停止准则；②层级信息被聚合后失真；③实验中使用的“真值”排序可能不可靠，对评估结果产生偏差；④对实际开发者可解释性与实用性有限。

---

## 387. Malware Detection based on API Calls: A Reproducibility Study

**arXiv ID:** 2601.08725 | [PDF](https://arxiv.org/pdf/2601.08725v1)

**作者:** Juhani Merilehto `[一作]` (University of Vaasa), Juhani Merilehto `[通讯]` (University of Vaasa)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5024761489)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

独立复现了Fellicious等人提出的基于Windows API调用频率的无序检测方法，并对Unigram、Bigram、Trigram和Combined四种n-gram模型在2500调用阈值下进行训练和评估。

**💡 创新点**

通过多次实验验证原方法的鲁棒性，并在多随机种子下实现了1–2.5%的F1提升，证明仅凭频率特征即使无序也能获得高精度。

**🔧 技术方法**

使用scikit-learn的Random Forest分类器，特征为API调用的n-gram频数（Unigram 59维，Bigram 2540维，Trigram 5483维）。

**📊 数据集**

使用Zenodo公开的API调用跟踪数据集，共330,105恶意样本和10,000正常样本（约572 GB，包含59个API函数）。

**📈 对比分析**

对照原论文的四个模型，采用同样的训练/测试划分、相同阈值和评估指标（F1、精确率、召回率、ROC‑AUC），在最佳阈值2500时F1均提高0.99%–2.57%，Precision、Recall、AUC均高于原版。

**⚠️ 局限性**

仅使用默认Random Forest参数，未进行系统的超参数优化；模型仅基于频数特征，缺乏对时间序列或上下文的捕捉，可能对模拟攻击存在脆弱性。

---

## 388. Multivariate Polynomial Codes for Efficient Matrix Chain Multiplication in Distributed Systems

**arXiv ID:** 2601.08708 | [PDF](https://arxiv.org/pdf/2601.08708v1)

**作者:** Jesús Gómez-Vilardebò `[一作]` `[通讯]` (Centre Technological of Telecommunications of Catalunya), Jesús Gómez-Vilardebò (Centre Technological of Telecommunications of Catalunya)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了两种多元多项式编码方案，用于解决分布式环境下的矩阵链乘法中慢节点（straggler）导致的延迟问题。

**💡 创新点**

创新点在于通过多变量编码既降低了每个工作节点的存储与上传开销，又维持了可恢复阈值，克服了传统单变量编码在链长增加时导致的指数级存储增长。

**🔧 技术方法**

采用了多元多项式插值、分块矩阵编码与多变量插值解码的技术，并对共享与专用存储两种资源模型进行了分析。

**📊 数据集**

论文未使用具体公开数据集，而是通过理论推导和仿真评估（基于参数 p、N、m 的数值实验）验证方案性能。

**📈 对比分析**

通过与传统单变量多项式编码在计算与存储开销上的对比，实验表明多元方案在链长增大时存储开销显著下降（趋近常数或线性增长），但计算开销随链长以 2^(m‑1) 级增长，恢复阈值也较高。

**⚠️ 局限性**

主要局限在于：恢复阈值上升导致额外计算负担；多变量插值的可解码性需额外验证；当分块规模较大时，实际实现的调度与负载均衡仍需进一步研究。

---

## 389. Nationality and Region Prediction from Names: A Comparative Study of Neural Models and Large Language Models

**arXiv ID:** 2601.08692 | [PDF](https://arxiv.org/pdf/2601.08692v1)

**作者:** Keito Inoshita `[一作]` (Kansai University), Keito Inoshita `[通讯]` (Kansai University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5106529447)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对比六种传统神经模型与六种大型语言模型（LLM）在按姓名预测国籍、地区、洲三层粒度上的性能，系统评估并提出LLM在此任务上的优势与局限。

**💡 创新点**

创新点：①系统化比较传统模型与LLM的多粒度预测能力；②在频率分层与错误类型分析中揭示LLM的“近似成功”特征；③提出基于地区层级的评价框架，阐明粒度越粗模型差距越小。

**🔧 技术方法**

技术：传统机器学习（SVM、fastText）、深度学习（CNN、BiLSTM）、预训练字符级Transformer（CANINE、XLM‑RoBERTa）以及GPT‑4.1‑mini 的六种提示策略（Zero‑shot、Few‑shot、Chain‑of‑Thought、Self‑Consistency、Least‑to‑Most、Self‑Reflection）。

**📊 数据集**

数据集：name2nat（罗马字化姓名），经过滤后保留99个国籍、约75k条样本，划分为8:1:1训练/验证/测试。

**📈 对比分析**

比较方法：在99国籍、14地区、6洲三层粒度上分别用 Accuracy、Macro‑F1、Precision@k 评估。LLM（Self‑Reflection）在 99 国籍上 Acc≈0.78、Macro‑F1≈0.78，显著优于 SVM（Acc≈0.48）。在更粗粒度时差距缩小，LLM 仍保持高准确率；频率分层显示 SVM 频率鲁棒性最好，预训练模型与 LLM 在低频国籍上表现下降；错误分析显示 LLM 多为“近似成功”错误，神经模型易跨区域错误。

**⚠️ 局限性**

局限性：仅评估 GPT‑4.1‑mini，未覆盖其它 LLM；仅使用罗马字化英文姓名，未考虑原始文字与多音译；对国籍与姓名的对应关系视为静态，未处理多重合法标签；预训练语料偏差导致低频国籍表现欠佳。

---

## 390. Lessons from the Field: An Adaptable Lifecycle Approach to Applied Dialogue Summarization

**arXiv ID:** 2601.08682 | [PDF](https://arxiv.org/pdf/2601.08682v1)

**作者:** Kushal Chawla `[一作]` (Capital One), Sambit Sahu `[通讯]` (Capital One)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并验证了一个基于代理（agentic）架构的多方对话摘要系统，重点解决了评估、任务分解、输入噪声以及提示可移植性等工业场景难题。

**💡 创新点**

创新点包括：1）结合人类评估与LLM-as-a-Judge（AutoEval）的混合评估协议，适应需求演变；2）把摘要任务拆解为草稿、评估与修订三个可单独优化的代理；3）系统性分析并降低ASR引起的噪声对摘要质量的影响；4）揭示不同LLM对同一提示的系统性偏好差异，提示可移植性受限。

**🔧 技术方法**

技术手段：agentic 多代理架构（Drafting、Accuracy/Completeness/Readability Evaluator、Refinement、Redundancy Checker）；LLM-as-a-Judge（AutoEval）基于Claude 3.7 Sonnet；使用Llama‑3.3‑70B‑Instruct、gpt‑oss‑120b；Whisper+对齐模型用于提升 ASR 质量；基于专家手工标注的 Gold Summary 进行评估校准。

**📊 数据集**

使用内部匿名化的多方对话数据，包含音频→ASR 转录、专家手工标注的 Gold Summary 以及用于训练 AutoEval 的合成样本。

**📈 对比分析**

与单一 LLM（monolithic）基线对比，agentic v5 在人类 A/B 评估中被优选 59%（vs 23%），在自动评估中 Accuracy 4.48、Completeness 3.68、Readability 4.70，明显优于 monolithic；Whisper+对齐的 upstream 流程在 WER 上下降 5% 以上，摘要质量提升（Accuracy 4.57→4.56，Completeness 3.81→3.68，Readability 4.48→4.35，用户偏好从 29% 提升至 53%）。

**⚠️ 局限性**

局限性：①提示与代理设计高度依赖任务，迁移到其他领域需重新校准；②实验仅基于内部数据，缺乏公开数据集验证；③代理体系结构导致推理延迟；④不同 LLM 对同一提示的偏好差异，导致提示不可迁移。

---

## 391. TRACE: Reconstruction-Based Anomaly Detection in Ensemble and Time-Dependent Simulations

**arXiv ID:** 2601.08659 | [PDF](https://arxiv.org/pdf/2601.08659v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 392. Advancing ESG Intelligence: An Expert-level Agent and Comprehensive Benchmark for Sustainable Finance

**arXiv ID:** 2601.08676 | [PDF](https://arxiv.org/pdf/2601.08676v1)

**作者:** Yilei Zhao `[一作]` (Nanyang Technological University), Wei Yang Bryan Lim `[通讯]` (Nanyang Technological University)

**通讯引用:** 6381 | [OpenAlex ID](https://openalex.org/A5027969322)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了名为ESGAgent的分层多智能体系统，用于自动化执行从信息检索、网页搜索到深度分析的ESG评估工作，并配套构建了一个三级难度的ESG基准测试集，用以评估从原子问答到综合报告生成的多维能力。

**💡 创新点**

创新点包括：①专门针对ESG领域设计的工具集（检索器、深度研究器、浏览器、绘图器、报告生成器等）与分层规划机制相结合；②利用检索增强生成（RAG）与知识图谱实现多源信息融合；③构建了从易到难、从事实提取到完整报告的三层基准，具有较高的行业现实性与诊断性；④在性能评测中表现出显著优于现有闭源LLM的优势。

**🔧 技术方法**

技术实现主要基于：层级化智能体架构、检索增强生成框架（LightRAG + 知识图谱）、多工具调用（Python解释器、深度分析器、Web搜索器等）、自动化绘图与报告生成模块，并通过多模型评审框架评估结果。

**📊 数据集**

使用的数据集为来自道琼斯工业平均指数（DJIA）成分股的310份2010-2024年的年度可持续发展报告（含文本、PDF、CSV、MP3/MP4等多模态附件），以及由领域专家标注的任务难度与能力标签。

**📈 对比分析**

在二层基准（易中难）上，与GPT-5系列、Gemini-3系列、Deepseek等闭源LLM进行对比，ESGAgent在原子问答任务中平均准确率达84.15%，高于Gemini-3-flash的80.89%；在三级综合报告任务中，与Grok、Perplexity、GPT-5深度研究版等进行比较，ESGAgent在图表数量、引用密度和专业度上均优于基线，报告更具可验证性与信息密度；消融实验进一步验证了深度研究器和检索器的关键作用。

**⚠️ 局限性**

局限性包括：仍可能生成事实错误或计算失误，尤其在处理非标准化或含糊的报告时；受限于可访问的披露信息，可能无法完整构建可验证洞见；在高强度的专业应用场景中需人工复核与监控；隐私与数据安全要求在使用Web浏览器时需严格隔离。

---

## 393. Analyzing Bias in False Refusal Behavior of Large Language Models for Hate Speech Detoxification

**arXiv ID:** 2601.08668 | [PDF](https://arxiv.org/pdf/2601.08668v1)

**作者:** Kyuri Im `[一作]` (TU Dresden), Michael Färber `[通讯]` (TU Dresden)

**通讯引用:** 1684 | [OpenAlex ID](https://openalex.org/A5031600582)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统分析LLM在仇恨言论去毒任务中的错误拒绝行为，并量化其在不同语言与语境中的偏见；提出跨语言翻译框架显著降低错误拒绝率。

**💡 创新点**

首次在多语言去毒任务中揭示对民族、宗教、政治议题的系统性错误拒绝偏见，并通过跨语言翻译轻量化地缓解该偏见。

**🔧 技术方法**

使用LLM指令式去毒、HolisticBias 分类、毒性评分模型、Phi-4判定、Qwen‑MT 翻译与 GPT‑4o‑mini 去毒等技术。

**📊 数据集**

英文数据集：HateXplain、ParaDetox、Davidson；多语种数据集：法语、德语、西班牙语、中文、韩语。

**📈 对比分析**

在9种LLM上比较拒绝率，发现英文场景拒绝率最高；跨翻译方法将HateXplain的错误拒绝率从11.78%降至1.09%，同时保持毒性与语义一致性。

**⚠️ 局限性**

局限于固定提示词、仅覆盖六种语言、依赖自动判定模型进行偏见分析，可能引入判定误差且对低资源语言缺乏验证。

---

## 394. Efficient Parameter Calibration of Numerical Weather Prediction Models via Evolutionary Sequential Transfer Optimization

**arXiv ID:** 2601.08663 | [PDF](https://arxiv.org/pdf/2601.08663v1)

**作者:** Heping Fang `[一作]` (Southern University of Science and Technology), Peng Yang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 4910 | [OpenAlex ID](https://openalex.org/A5039532881)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并评估了 SEETO（SEquential Evolutionary Transfer Optimization）算法，用以在有限评估预算下高效校准数值天气预报（NWP）模型的物理参数。

**💡 创新点**

创新点包括：① 通过自监督编码‑解码器提取气象状态的低维表示并以余弦相似度衡量任务相似度；② 双层自适应知识迁移机制，既在解级注入源任务的精英个体，又在模型级构建加权代理模型；③ 构建源任务档案实现“warm start”，显著降低每次任务的搜索开销。

**🔧 技术方法**

技术方法：多目标进化优化（基于 PlatEMO 平台）、Gaussian Process 代理模型、深度自监督表示学习（ResNet‑编码‑解码网络）、余弦相似度与 softmax 权重、动态权重 β 调节模型贡献。

**📊 数据集**

使用数据集：ERA5 重新分析与 CLDAS‑V2.0 观测数据；在 WRF 4.5 版本的 10 个目标任务（来自 2017 年 8 月）与 20 个源任务（2017 年 7 月）上进行参数校准实验。

**📈 对比分析**

与 MOASMO、KMO 两种基线算法在仅允许 60 次 WRF 评估的情形下对比。SEETO 在 10 个任务中平均提升约 6% 的 Hypervolume（HV）指标，且相同性能所需的额外评估量分别为 28–64%；在任务相似度高的 7 个目标任务中表现尤为突出；低相似度任务中出现负迁移，后期性能被基线追赶。

**⚠️ 局限性**

局限性：在源与目标任务相似度低时会产生负迁移，导致后期收敛下降；当前仅使用静态相似度度量，缺乏在线动态调整机制；实验仅覆盖单一分辨率和同一物理方案，扩展到高维、多目标或跨分辨率场景仍需进一步研究。

---

## 395. Inferring Latent Intentions: Attributional Natural Language Inference in LLM Agents

**arXiv ID:** 2601.08742 | [PDF](https://arxiv.org/pdf/2601.08742v1)

**作者:** Xin Quan `[一作]` (University of Manchester), André Freitas `[通讯]` (University of Manchester)

**通讯引用:** 2480 | [OpenAlex ID](https://openalex.org/A5053978668)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Attributional NLI (Att‑NLI) 框架和 Undercover‑V 文本游戏，用于评估多代理 LLM 的意图推理能力。

**💡 创新点**

创新点在于将传统 NLI 扩展为两阶段的归谬‑演绎推理，并设计可验证的游戏与 Attributional Score 指标，以及将外部定理推理器引入神经符号 Att‑NLI。

**🔧 技术方法**

使用 GPT‑4o、GPT‑4o‑mini、Mixtral‑8x22B、Mistral‑Medium 等 LLM，并结合 abduction‑deduction 流程、Isabelle/HOL 定理推理器实现神经符号推理。

**📊 数据集**

主要使用自制的 Undercover‑V 游戏数据集（角色词卡与游戏日志），未引用公开文本语料库。

**📈 对比分析**

通过对比标准 NLI、标准 Att‑NLI 与神经符号 Att‑NLI 在胜率、消除率和 Attributional Score 等指标上的表现，结果显示神经符号 Att‑NLI 获得最高的 17.08% 侦探胜率和约 24% 的 Attributional Score 提升。

**⚠️ 局限性**

局限性包括：意图推理仅限于 spy/市民两角色的词义区分，缺乏对更复杂社会规范和长期行为的建模；评估依赖游戏性能和嵌入相似度，难以推广到开放式多代理环境。

---

## 396. Model-Agnostic Solutions for Deep Reinforcement Learning in Non-Ergodic Contexts

**arXiv ID:** 2601.08726 | [PDF](https://arxiv.org/pdf/2601.08726v1)

**作者:** Bert Verbruggen `[一作]` (Vrije Universiteit Brussel), Vincent Ginis `[通讯]` (Harvard University)

**通讯引用:** 1187 | [OpenAlex ID](https://openalex.org/A5049169851)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了在非可遍历（non‑ergodic）环境下强化学习的局限性，并提出通过在训练中加入时间维度的重复决策，使深度RL能学习到基于时间平均增长率的最优策略。

**💡 创新点**

创新点在于：①不对奖励或目标函数做任何改造，而是通过让网络直接感知轨迹的时间信息来实现对可遍历性破坏的补偿；②在离散（DQN）和连续（Actor‑Critic）两种结构下验证了该方法，并将其与传统的期望值优化策略进行对比。

**🔧 技术方法**

采用的技术包括：Deep Q‑Network（DQN）和基于Actor‑Critic的连续动作空间学习；在实验中对状态加入当前财富值，利用经验回放和ε‑greedy探索；对策略使用对数损失与平滑L1损失；对奖励使用几何平均而非算术平均。

**📊 数据集**

使用的数据集为两种自定义的合成环境：①单一动作的乘法财富增长玩具模型；②基于Kelly准则的投资组合分配问题。所有环境均通过人工设定的概率和收益因子构造。

**📈 对比分析**

比较方法：将训练得到的策略与理论上的期望值优化（p_E）和时间增长率优化（p_T）进行对比，计算均方误差（MSE）。实验结果显示，单步训练的DQN及Actor‑Critic几乎只能复制期望值策略，而加入时间重复后，策略逐渐逼近Kelly最优策略，MSE随重复次数的增加显著下降。

**⚠️ 局限性**

局限性：①实验仅在极其简化的合成环境中验证，缺乏真实金融或复杂多智能体场景的评估；②需要手动设定重复次数和训练周期，对训练效率和收敛速度的影响尚未深入研究；③方法假设环境的时间依赖可通过多步经验显式捕捉，可能不适用于瞬时决策或高度噪声的场景。

---

## 397. Real-Time Localization Framework for Autonomous Basketball Robots

**arXiv ID:** 2601.08713 | [PDF](https://arxiv.org/pdf/2601.08713v1)

**作者:** Naren Medarametla `[一作]` (Vellore Institute of Technology), Sreejon Mondal `[通讯]` (Vellore Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一个基于单目摄像头的轻量前馈神经网络定位系统，用于Robocon 2025篮球赛场的实时自定位。

**💡 创新点**

创新点在于仅使用普通单摄像头和简单前馈网络即可完成实时定位，避免了大光学视角、深度传感器或复杂CNN的需求。

**🔧 技术方法**

采用HSV颜色空间白线分割、径向下采样、全连接网络预测坐标、Integrated Gradients可解释性、Gazebo仿真数据收集等技术。

**📊 数据集**

使用了在Gazebo仿真场景下收集的6283张图像及对应坐标的数据集，按9:1比例划分训练/测试。

**📈 对比分析**

与MobileNetV2和EfficientNet‑B0对比，前馈模型在Jetson Orin Nano上实现了0.040的MSE、3.3 ms的推理时间，平均定位误差约0.06 m，性能显著优于两者。

**⚠️ 局限性**

局限性包括受光照与遮挡影响、缺乏多传感器融合导致姿态校正不稳定，以及模型权重大导致内存占用较高。

---

## 398. Evaluating the Ability of Explanations to Disambiguate Models in a Rashomon Set

**arXiv ID:** 2601.08703 | [PDF](https://arxiv.org/pdf/2601.08703v1)

**作者:** Kaivalya Rawal `[一作]` (Oxford Internet Institute), Chris Russell `[通讯]` (Oxford Internet Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出基于模型相对性和对流形评估的无真值特征重要性解释评估框架，验证其能检测Rashomon集中的伪解释。

**💡 创新点**

创新点在于引入三大评估原则（局部上下文化、模型相对性、对流形评估）以及新的基于k‑NN可预测性无真值指标，能够在无真值情况下区分模型差异并发现解释公平洗牌。

**🔧 技术方法**

采用k‑最近邻预测可解释性、可预测性评估与传统敏感性指标（PGI、PGU）对比的技术。

**📊 数据集**

使用德国信用（German Credit）、COMPAS 和社区与犯罪（Communities and Crime）数据集。

**📈 对比分析**

与PGI、PGU等传统指标对比，传统指标误判率高达50%，而新框架在所有公平洗牌案例中均无误判，表现优异。

**⚠️ 局限性**

局限在于仅针对局部特征重要性解释，未涵盖全局或其他解释形式；需进一步验证在更大模型和多模态数据上的鲁棒性。

---

## 399. LLMs in Code Vulnerability Analysis: A Proof of Concept

**arXiv ID:** 2601.08691 | [PDF](https://arxiv.org/pdf/2601.08691v1)

**作者:** Shaznin Sultana `[一作]` (Ohio University), Nasir U. Eisty `[通讯]` (University of Tennessee)

**通讯引用:** 110 | [OpenAlex ID](https://openalex.org/A5035948887)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估开源LLM在软件漏洞检测、严重度预测、访问复杂度评估和修复生成四个任务中的性能，并通过微调、零样本和少样本方法进行对比

**💡 创新点**

首次在四阶段漏洞处理流水线中系统性比较代码专用与通用LLM的效果，发现微调是关键，并揭示传统评估指标对修复质量的不足

**🔧 技术方法**

使用LoRA参数高效微调、指令调优、零/少样本提示，以及多种代码相似度评估指标（CodeBERTScore、CodeBLEU、BLEU-4、ROUGE‑L、ChrF）

**📊 数据集**

采用公开的C/C++漏洞数据集Big‑Vul和VulRepair

**📈 对比分析**

通过对比F1、精确率、召回率以及修复指标，微调版模型普遍优于零/少样本；在零/少样本场景通用模型有时胜过专用模型，DeepSeek R1在多任务表现最强

**⚠️ 局限性**

局限包括仅针对C语言、数据集标注噪声、提示输出冗长导致推理成本高、评估仅基于文本相似度缺乏执行验证

---

## 400. MEMEWEAVER: Inter-Meme Graph Reasoning for Sexism and Misogyny Detection

**arXiv ID:** 2601.08684 | [PDF](https://arxiv.org/pdf/2601.08684v1)

**作者:** Paolo Italiani `[一作]` (University of Bologna), Paolo Rosso `[通讯]` (Universitat Politècnica de València)

**通讯引用:** 15184 | [OpenAlex ID](https://openalex.org/A5053947754)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了名为MemeWeaver的端到端可学习的图模型，用于检测梗图中的性别歧视与厌女言论，利用批次级的图推理来捕捉 meme 之间的社会互动关系。

**💡 创新点**

创新点包括：① 将文本与图像特征通过多模态融合（MFB/GMU）融合；② 在训练批次内部构造完全连通图并通过图消息传递自学习邻接关系（IMGR）；③ 结合 LLM 生成的补充标题以丰富语义；④ 通过批次级图推理实现更快收敛和更高准确率。

**🔧 技术方法**

技术包括 CLIP 图像/文本编码器、MFB/GMU 融合层、基于相似度的邻接矩阵构建与 GCN 消息传递、交叉熵分类器；以及使用 Qwen2.5-VL‑7B‑Instruct 生成的标题。

**📊 数据集**

使用两大公开梗图数据集：MAMI（针对厌女内容）和 EXIST（跨英西性别歧视），分别包含约9k和3.2k条样本。

**📈 对比分析**

与传统基线（EF‑CaTrBERT、PromptHate、Pro‑Cap、HyperHatePrompt）以及最新多模态 LLM（Qwen2.5‑VL、Phi‑4、GPT‑4o‑mini）比较，MemeWeaver 在 MAMI 上取得 77.6% Acc / 77.4% F1 / 83.4% AUC，显著领先；在 EXIST 上也实现了 76.3% Acc / 73.6% F1 / 72.9% AUC，表现稳健并且收敛更快。

**⚠️ 局限性**

局限性包括：① 未将 LLM 作为主干；② 仅在零样本下评估多模态 LLM；③ 仅测试两种标题提示；④ 对 EXIST‑Spanish 使用机器翻译，可能丢失文化细节；⑤ 图结构仅在固定批次内构建，缺乏全局关系。

---

## 401. PersonaDual: Balancing Personalization and Objectivity via Adaptive Reasoning

**arXiv ID:** 2601.08679 | [PDF](https://arxiv.org/pdf/2601.08679v1)

**作者:** Xiaoyou Liu `[一作]` (Fudan University), Zhongyu Wei `[通讯]` (Fudan University)

**通讯引用:** 5871 | [OpenAlex ID](https://openalex.org/A5011504177)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在LLM中提出PersonaDual框架，允许模型根据查询与个性化信息自动切换通用客观推理与个性化推理模式。

**💡 创新点**

创新点在于将两种推理模式整合到同一模型，并通过DualGRPO强化学习实现上下文感知的模式选择，既降低个性化干扰，又利用对齐信息提升客观答案质量。

**🔧 技术方法**

采用监督微调（SFT）训练两种前缀激活的推理轨迹，随后使用DualGRPO强化学习结合强制前缀采样和双模优势分解来优化模式选择策略。

**📊 数据集**

使用Qwen3-8B-Instruct为基础模型，构建PersonaDualData（含8,000条SFT样本与9,998条RL样本），采样自UltraMedical、FLAN与AlignX，并利用PersonaHub随机个性化与GPT‑4o生成的对齐个性化。

**📈 对比分析**

与通用模型、个性化模型和其他双模实现进行对比，实验表明PersonaDual在未对齐个性化下客观问答准确率可达54.0%（与无个性化上限54.7%接近），在对齐个性化下提升约3%客观准确率，并在个人化任务上超过专门的个性化模型。

**⚠️ 局限性**

主要限制是仅在英文数据集上验证，个性化基准有限，且仍可能面临社会身份偏见等问题，未来需扩展多语言与更丰富的个性化评测。

---

## 402. Além do Desempenho: Um Estudo da Confiabilidade de Detectores de Deepfakes

**arXiv ID:** 2601.08674 | [PDF](https://arxiv.org/pdf/2601.08674v1)

**作者:** Lucas Lopes `[一作]` (Universidade Federal do Paraná), André Grégio `[通讯]` (Universidade Federal do Paraná)

**通讯引用:** 1158 | [OpenAlex ID](https://openalex.org/A5059851424)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个基于转移性、鲁棒性、可解释性和计算效率四大支柱的深度伪造检测评估框架，并将其应用于五种最先进方法进行实证对比。

**💡 创新点**

创新点在于将传统的准确率指标扩展为多维可靠性评估体系，并给出统一的量化公式和全局可靠性得分。

**🔧 技术方法**

使用的技术包括CNN、ViT、LoRA、Transformer、GAN、Diffusion、Triplet学习、Grad‑CAM、LIME、SHAP以及多模态自然语言解释模型。

**📊 数据集**

评估所用的数据集为多个公开的深度伪造数据集（如FaceForensics++、DeepFake‑in‑the‑Wild、Celeb-DF、DFDC等），并在跨数据集、压缩、噪声以及攻击场景下进行测试。

**📈 对比分析**

方法对比显示TruthLens在转移性上最高（AUC≈0.94），CFM在鲁棒性上最优（R≈0.58），SCLoRA和OSDFD在效率与鲁棒性间取得平衡，整体全局可靠性得分最高为0.68。

**⚠️ 局限性**

局限性包括：大多数方法未进行对抗攻击评估；可解释性普遍不足（除TruthLens外多为盒子模型）；计算成本高，尤其是使用大型Transformer和多模态模型时，实际部署受限。

---

## 403. Rational degree is polynomially related to degree

**arXiv ID:** 2601.08727 | [PDF](https://arxiv.org/pdf/2601.08727v1)

**作者:** Matt Kovacs-Deak `[一作]`, Rain Zimin Yang `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

证明了对于每个布尔函数f，(f) ≤ 2(f)^4，其中(f)是f的度，(f)是f的有理度。这解决了Nisan和Szegedy在1994年提出的三个未解问题中的第二个。

**💡 创新点**

创新点在于首次证明了有理度与度之间的关系，具体为(f) ≤ O((f)^2(f)^2)，并且提供了更强的界限。

**🔧 技术方法**

使用了多项式对称化、Markov不等式等技术来进行证明。

**📊 数据集**

没有具体提到使用的数据集，主要是理论证明。

**📈 对比分析**

通过与已有的复杂度度量（如决策树复杂度）进行比较，证明了D(f) ≤ 4(f)^2(f)^2，从而得出(f)的界限。

**⚠️ 局限性**

限制在于该结果可能无法推广到所有部分布尔函数，且在某些情况下，(f)与(f)之间的关系可能会有较大差异。

---

## 404. UR-Bench: A Benchmark for Multi-Hop Reasoning over Ultra-High-Resolution Images

**arXiv ID:** 2601.08748 | [PDF](https://arxiv.org/pdf/2601.08748v1)

**作者:** Siqi Li `[一作]` (Zhejiang University), Yong Liu `[通讯]` (Zhejiang University)

**通讯引用:** 19928 | [OpenAlex ID](https://openalex.org/A5100724297)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向超高分辨率图像的多跳推理基准UR‑Bench，并提出了基于语义抽象与检索的Agent框架进行推理。

**💡 创新点**

创新点在于：①将超高分辨率图像拆分为可处理的语义块并用语言表征；②利用Agent在语言层面进行工具调用，实现模块化、多步推理；③设计了三层难度的问答划分以细粒度评估推理能力。

**🔧 技术方法**

使用的技术包括：LLM驱动的Agent控制、Qwen2.5‑VL‑7B生成语义描述、BGE‑M3语义检索、GroundingDINO目标检测、VLM问答、裁剪/增强等视觉工具。

**📊 数据集**

数据集为UR‑Bench，包含人文场景（古卷绘画）和自然场景（卫星图、街景），图像分辨率从数十万像素到数十亿像素，分为四个子集。

**📈 对比分析**

通过与多款开源/闭源M模型的端到端和Agent框架对比，Agent框架在整体准确率上提升15‑20个百分点；在不同子集和难度层级上均优于端到端方法，最高整体得分39.97。

**⚠️ 局限性**

局限性包括：需要先行生成大量语义块，工具调用成本高；对极大图像仍需裁剪/分块；目前模型对复杂跨区域关系的推理仍有限，尚未覆盖所有类型的超高分辨率图像。

---

## 405. TableCache: Primary Foreign Key Guided KV Cache Precomputation for Low Latency Text-to-SQL

**arXiv ID:** 2601.08743 | [PDF](https://arxiv.org/pdf/2601.08743v1)

**作者:** Jinbo Su `[一作]` (Renmin University of China), Jing Zhang `[通讯]` (Renmin University of China)

**通讯引用:** 16016 | [OpenAlex ID](https://openalex.org/A5100345321)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

预先离线计算表级 KV 缓存并构建 Trie，在线时快速匹配并加载所需表缓存，以显著加速 Text-to-SQL LLM 的推理。

**💡 创新点**

以主外键为引导恢复跨表注意力的表表示；使用 Table Trie 实现高效缓存检索；结合查询重排序与计算-加载流水线提升缓存命中率和并行度。

**🔧 技术方法**

主外键图与拓扑排序、LLM KV 缓存预计算、Trie 检索、缓存管理（FIFO/LRU）、查询重排序、异步预取流水线等技术。

**📊 数据集**

在 BIRD 与 Spider 两个 Text-to-SQL 基准数据集上进行实验。

**📈 对比分析**

与 Transformer 基线、vLLM、SGLang、PromptCache、TurboRAG 等方法对比，TableCache 在 TTFT 上相较基线降低至约 1/3，速度提升 3.6×，准确率基本保持不变。

**⚠️ 局限性**

目前仅适用于具有明确外键约束的结构化数据库，难以直接推广到无结构化文本问答或知识图谱问答等场景。

---

## 406. TerraFormer: Automated Infrastructure-as-Code with LLMs Fine-Tuned via Policy-Guided Verifier Feedback

**arXiv ID:** 2601.08734 | [PDF](https://arxiv.org/pdf/2601.08734v1)

**作者:** Prithwish Jana `[一作]` (Georgia Institute of Technology), Laurent Callot `[通讯]` (Amazon Web Services)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种结合大语言模型与形式化验证的神经符号框架 TerraFormer，用于从自然语言生成和变更 Terraform IaC 配置；

**💡 创新点**

创新点包括：① 自动化构建大规模高质量 NL‑to‑IaC 数据集 TF‑Gen 与 TF‑Mutn，② 采用多轮 LLM 自我纠错与验证器反馈实现数据修复；③ 在 SFT 后引入基于验证器的细粒度奖励的强化学习，显著提升 IaC 的语法、可部署性与策略合规性；

**🔧 技术方法**

技术手段包括：大型 LLM 微调（SFT）、基于梯度的 RL（GRPO）与细粒度奖励（编译、部署、合规三阶段验证器），以及多轮错误纠正循环；

**📊 数据集**

使用的数据集有：从 GitHub 采集的 Terraform 配置构成 SeedCorpus，经过验证器修复后得到 152k+ TF‑Gen（生成）与 52k+ TF‑Mutn（变更）实例；

**📈 对比分析**

与 17 先进 LLM（含 Sonnet 3.7、DeepSeek‑R1、GPT‑4.1 等）对比，TerraFormer 在 TF‑Gen（Test）和 TF‑Mutn（Test）上均优于更大模型，IaC‑Eval 上排名第三，且在编译、部署、合规、最佳实践与安全合规率方面显著提升；

**⚠️ 局限性**

局限性包括：① 目前仅针对 Terraform，扩展到其他 IaC 工具仍需适配验证器；② 强调形式化验证的同时可能忽略业务层面更细粒度的业务意图；③ 训练与验证成本高，尤其是多轮修复循环需要昂贵的 LLM 费用与算力。

---

## 407. Salience-SGG: Enhancing Unbiased Scene Graph Generation with Iterative Salience Estimation

**arXiv ID:** 2601.08728 | [PDF](https://arxiv.org/pdf/2601.08728v1)

**作者:** Runfeng Qu `[一作]` (Technische Universität Berlin), Olaf Hellwich `[通讯]` (Technische Universität Berlin)

**通讯引用:** 3537 | [OpenAlex ID](https://openalex.org/A5081055268)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Salience-SGG 框架，结合迭代显著性解码器 (ISD) 在 Unbiased‑SGG 过程中强调具有显著空间结构的三元组

**💡 创新点**

通过底层无语义显著性标签与迭代显著性解码器解决偏置策略导致的空间感知缺失问题

**🔧 技术方法**

使用 DETR‑基础网络、改进的 G‑ESA 与 P‑ECA 关注层、显著性消息传递与迭代细化技术

**📊 数据集**

在 Visual Genome、Open Images V6、GQA‑200 三大公开数据集上进行实验

**📈 对比分析**

与 Motifs、BGNN、DRM、TDE、IETrans、SSR‑CNN 等多种基准对比，取得 F@K、mR@K、wmAP 等指标的 SOTA 甚至第二名

**⚠️ 局限性**

仍受样本不均衡影响，对极少见谓词的准确性有限，且显著性标签生成成本较高

---

## 408. RAGShaper: Eliciting Sophisticated Agentic RAG Skills via Automated Data Synthesis

**arXiv ID:** 2601.08699 | [PDF](https://arxiv.org/pdf/2601.08699v1)

**作者:** Zhengwei Tao `[一作]` (Peking University), Wentao Zhang `[通讯]` (Tencent AI Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为 RAGShaper 的自动化数据合成框架，利用知识图谱式信息树与多维扰乱器生成密集检索环境，随后通过约束导航策略让教师 LLM 产生包含错误修正与噪声排除的完整推理轨迹，并在此基础上微调大型语言模型实现 Agentic RAG。

**💡 创新点**

创新点主要包括：①设计了基于信息树与多维扰乱器的合成流程；②提出了约束导航策略，强制教师代理与噪声交互，从而得到高质量的错误修正轨迹；③系统化展示了比人工标注更高效、更鲁棒的 Agentic RAG 训练数据生成方法；④在多种基准上实现了显著性能提升。

**🔧 技术方法**

技术手段包括：密集检索工具（基于 E5 + DPR）、ReAct 交互式推理框架、LLM 生成扰乱文本、约束导航策略、标准化的监督微调（SFT）和对比实验评估（EM/F1）。

**📊 数据集**

使用公开 Wikipedia 作为知识库；合成训练集为 4.5k/6.5k 条轨迹；对比人类标注的 HL‑Data（HotpotQA、2WikiMultiHopQA 子集）；在四个公开基准（Natural Questions、PopQA、AmbigQA、Bamboogle）上进行评测。

**📈 对比分析**

采用与多种 prompt‑based（Iter‑RetGen、IR‑CoT、FLARE、Search‑o1 等）及 learning‑based（DeepRAG、IKEA、ReasonRAG、DecEx‑RAG、Search‑R1、HL‑Data）方法对比；使用 Exact Match（EM）和 F1 分数衡量；RAGShaper 6.5k 模型在平均 EM 50.3、F1 62.0 上显著优于所有基线，尤其在噪声密集的 Bamboogle 与 AmbigQA 上表现最为突出。

**⚠️ 局限性**

局限性包括：合成数据尚未完全突破最难陷阱（如 Subjective Fallacy）的处理；对极端复杂推理的覆盖不足；缺乏强化学习等更深层次训练机制；模型仍可能在极端噪声或未知知识域中失效。

---

## 409. Double Strike: Breaking Approximation-Based Side-Channel Countermeasures for DNNs

**arXiv ID:** 2601.08698 | [PDF](https://arxiv.org/pdf/2601.08698v1)

**作者:** Lorenzo Casalino `[一作]` (CentraleSupélec), Rubén Salvador `[通讯]` (CentraleSupélec)

**通讯引用:** 1143 | [OpenAlex ID](https://openalex.org/A5055706093)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d`

**🎯 论文内容**

针对使用随机剪枝增强深度神经网络侧信道抗性的MACPruning方案，作者提出了一种基于控制流依赖的预处理方法，成功恢复了大量重要权重。

**💡 创新点**

创新点在于揭示并利用剪枝机制中的控制流差异，使得原本认为安全的实现被突破，展示了侧信道攻击对基于重要性剪枝防御的实质性威胁。

**🔧 技术方法**

使用侧信道分析、统计相关性、控制流分类与拼接技术，并在ChipWhisperer-Lite硬件上采集功耗侧信道轨迹。

**📊 数据集**

实验使用自定义多层感知器模型（未指明公开数据集），在ChipWhisperer-Lite平台上收集侧信道轨迹。

**📈 对比分析**

与原始MACPruning防御相比，攻击仅需数百条轨迹即可恢复96%的重要权重，甚至100%的非重要权重，显示出显著的安全降低；实验环境局限于单个MLP与专用硬件。

**⚠️ 局限性**

局限性包括：需要对控制流做精确分类；仅在基于权重重要性剪枝的实现中有效；对更复杂或无控制流差异的实现仍需进一步验证。

---

## 410. Enabling Population-Based Architectures for Neural Combinatorial Optimization

**arXiv ID:** 2601.08696 | [PDF](https://arxiv.org/pdf/2601.08696v1)

**作者:** Andoni Irazusta Garmendia `[一作]` (University of the Basque Country), Alexander Mendiburu `[通讯]` (University of the Basque Country)

**通讯引用:** 3125 | [OpenAlex ID](https://openalex.org/A5051966806)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于神经网络的种群搜索框架（PB‑NCO），通过共享内存的上下文改进策略（cNI）和条件化构造策略（cNC）实现对图问题的全局搜索。

**💡 创新点**

创新点在于：①构建了三层种群感知等级的分类框架；②在单网络中实现了对整个种群的可变大小、可置换的表示与交互；③将多样性与强度化合并到学习目标和重启机制中，首次在NCO中实现了种群级别的协调。

**🔧 技术方法**

技术核心包括图变换器（Graph Transformer）编码器、基于k‑NN的共享内存检索、基于奖励加上重复惩罚的策略梯度训练，以及通过可调探测权重实现质量–多样性的条件化构造。

**📊 数据集**

在Erdős–Rényi（ER）和随机二分图（RB）生成的图实例上进行实验，训练集为50–300节点，评估集为700–800节点的ER和800–1200节点的RB。

**📈 对比分析**

与精确求解器（GUROBI）、贪婪启发式、遗传算法、粒子群、专用算法（BURER、K_AMIS）以及现有学习基线（S2V‑DQN、FlowNet、MARCO 等）比较，PB‑NCO 在 MC 上取得最优或同等最佳结果，在 MIS 上保持竞争力，尤其在 1 分钟内的计算预算下表现突出。

**⚠️ 局限性**

局限性包括：共享内存容量有限导致长时间运行时存储开销大；重启策略与探索权重调度为手工设计，缺乏自适应；模型对非常大图或极长搜索时间的可扩展性尚未验证；以及在不同问题类型上的迁移性需要进一步研究。

---

## 411. Parallel Context-of-Experts Decoding for Retrieval Augmented Generation

**arXiv ID:** 2601.08670 | [PDF](https://arxiv.org/pdf/2601.08670v1)

**作者:** Giulio Corallo `[一作]` (SAP Labs), Paolo Papotti `[通讯]` (EURECOM)

**通讯引用:** 3795 | [OpenAlex ID](https://openalex.org/A5011336242)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关的 Parallel Context-of-Experts Decoding（Pced）框架，利用预编码的 KV 缓存并行解码，将检索文档视为独立专家，在解码阶段聚合多文档信息，实现高效的多文档推理。

**💡 创新点**

核心创新在于把文档聚合从注意力层转移到解码层，结合检索感知对比解码与检索先验，在保持 KV 缓存模块化的同时恢复跨文档交互，从而兼顾推理准确性与速度。

**🔧 技术方法**

技术要点包括：①预编码 KV 缓存；②并行多专家解码；③检索感知对比解码（contrastive decoding）和检索先验融合；④动态 β、γ 参数调节；⑤与传统拼接、KV 合并（APE）、MapReduce 等 baseline 的对比。

**📊 数据集**

使用了 LOFT（RAG/ICL 任务）、LongBench、HotpotQA、MuSiQue、NQ、QAMParI、Quest、Web、Tracking7、Date 等多种公开数据集进行评测。

**📈 对比分析**

与全上下文拼接、KV 合并、MapReduce 等方法对比，Pced 在多文档推理任务（如 HotpotQA、MuSiQue、QAMParI 等）上提升 5–70 分，能够匹配或超越全上下文基线；在效率方面，TTFT 可提升至 180×，整体推理时间大幅降低。

**⚠️ 局限性**

局限性包括：需要访问模型 logits，限制了对闭源或仅 API 的 LLM 应用；对检索质量敏感，若检索不到关键文档会影响效果；KV 缓存需要大量存储，适用于读多写少的静态知识库。

---

## 412. From Classical to Quantum Reinforcement Learning and Its Applications in Quantum Control: A Beginner's Tutorial

**arXiv ID:** 2601.08662 | [PDF](https://arxiv.org/pdf/2601.08662v1)

**作者:** Abhijit Sen `[一作]` (Tulane University), Denys I. Bondar `[通讯]` (Tulane University)

**通讯引用:** 887 | [OpenAlex ID](https://openalex.org/A5076308400)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文编写了一份面向本科生的强化学习（RL）教程，系统阐述了RL基础概念、核心算法（策略评估/改进、动态规划、蒙特卡洛、时序差分、SARSA、Q‑learning、策略梯度与Actor‑Critic）以及与量子控制的跨学科融合，并给出了完整可执行的代码示例。

**💡 创新点**

创新点在于：① 用一个单一、简易的1D网格例子将RL理论与代码实现串联，降低学习门槛；② 将RL与量子控制结合，展示如何用RL方法实现量子态操控（如单比特旋转控制）并给出Actor‑Critic量子RL示例；③ 提供完整的GitHub代码和中文解释，便于学生直接复现和实验。

**🔧 技术方法**

使用的技术包括：
- 经典RL算法（策略评估/改进、DP、MC、TD、SARSA、Q‑learning、策略梯度、Actor‑Critic）
- 量子控制技术（时间相关薛定谔方程、GRAPE、量子最优控制）
- 变分量子电路（VQC）实现量子策略梯度。
- 代码实现主要使用Python（NumPy、Gym‑style环境）及量子模拟器。

**📊 数据集**

没有使用公开数据集；环境为自定义网格或单比特量子演化示例，所有数据均在代码中生成。

**📈 对比分析**

本文为教程性质，未做系统的实验对比或性能评测。量子控制示例仅展示单比特旋转控制的保真度可达 ~0.99999998596，但未与其他算法或基准进行定量比较。

**⚠️ 局限性**

局限性：
- 仅演示了极其简化的离散网格和单比特任务，缺乏对大规模、连续动作或多步高难度任务的讨论；
- 对深度RL或分布式学习方法的介绍不足；
- 量子RL示例仅是概念性演示，缺乏对真实量子硬件噪声和可扩展性的深入分析；
- 文中未提供系统性实验或对比研究，读者需自行扩展验证。

---

## 413. Prism: Towards Lowering User Cognitive Load in LLMs via Complex Intent Understanding

**arXiv ID:** 2601.08653 | [PDF](https://arxiv.org/pdf/2601.08653v1)

**作者:** Zenghua Liao `[一作]` (National University of Defense Technology), Xiang Zhao `[通讯]` (National University of Defense Technology)

**通讯引用:** 15240 | [OpenAlex ID](https://openalex.org/A5065273431)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了名为Prism的框架，针对LLM在复杂意图识别中的认知负荷问题，通过逻辑依赖的澄清交互实现更高效的用户意图解析

**💡 创新点**

首次将认知负荷理论与意图分解相结合，提出逻辑化澄清生成和意图感知奖励机制，并通过自我进化调优提升LLM的逻辑澄清能力

**🔧 技术方法**

利用LLM进行意图分解、交互式表格澄清、蒙特卡洛采样的意图感知奖励、SFT/DPO自进化微调及CLT建模

**📊 数据集**

使用自建的27域429意图检索数据集，以及ITIU、Tell_Me_More、Ask-before-Plan等公开数据集

**📈 对比分析**

与三种SOTA交互式意图理解方法比较，逻辑冲突率降至11.5%，用户满意度提升14.4%，任务完成时间缩短34.8%，在多数据集和多模型上均保持领先

**⚠️ 局限性**

依赖大量手工标注的依赖关系，模型对极度模糊或新颖意图的鲁棒性有限，且在多轮交互中的实时性与算力仍有提升空间

---

## 414. Data Product MCP: Chat with your Enterprise Data

**arXiv ID:** 2601.08687 | [PDF](https://arxiv.org/pdf/2601.08687v1)

**作者:** Marco Tonnarelli `[一作]` (Eindhoven University of Technology), Linus W. Dietz `[通讯]` (King's College London)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5107823319)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一个基于 LLM 的聊天式数据产品访问与治理系统——Data Product MCP，能够在数据市场中检索、请求并执行符合合同与治理规则的查询。

**💡 创新点**

核心创新在于将 Model Context Protocol 与数据产品市场、LLM 代理以及可执行的数据契约相结合，实现了目的驱动的自动化治理、即时查询审计和跨域数据组合的自助式探索。

**🔧 技术方法**

使用技术包括：Anthropic Claude / GPT LLM、Model Context Protocol（FastMCP）、Entropy Data 数据产品市场（ODPS/ODCS）、云数据平台（Snowflake、Databricks、GCP）、SQL 网关与治理 AI 模块、以及 Python/FastAPI 架构。

**📊 数据集**

主要数据集为企业内部业务数据（如客户销售数据、支持工单数据）通过数据产品形式在演示中使用；并收集了 16 位行业专家的使用反馈。

**📈 对比分析**

评估方式为情景驱动演示与问卷调查，未给出定量性能指标；专家认为系统显著降低技术门槛、提升探索效率，并在治理合规性方面优于传统手工流程。

**⚠️ 局限性**

局限性包括：数据与元数据质量不足、LLM 生成查询可能出现误差、身份与访问控制需更严密、系统可扩展性与成本控制待验证，以及对合规性与信任度的持续监督需求。

---

## 415. To Retrieve or To Think? An Agentic Approach for Context Evolution

**arXiv ID:** 2601.08747 | [PDF](https://arxiv.org/pdf/2601.08747v1)

**作者:** Rubing Chen `[一作]` (Hong Kong Polytechnic University), Qing Li `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 45320 | [OpenAlex ID](https://openalex.org/A5100404130)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Agentic Context Evolution (ACE) 框架，利用多代理决策在多跳问答任务中动态决定检索或内部推理，以提升上下文质量与推理效率。

**💡 创新点**

创新点在于将上下文增强视为“检索‑或‑思考”交替过程，并通过中心指挥代理与检索器、推理器协同，实现基于元认知的动态上下文演化，避免冗余检索和信息噪声。

**🔧 技术方法**

采用多代理决策（majority voting）、检索‑思考循环、LLM（LLaMA‑3.1‑8B‑Instruct）生成子查询和答案，以及工作记忆更新机制。

**📊 数据集**

在 MultiHop‑RAG、HotpotQA 与 2WikiQA 三个多跳问答基准上进行评估。

**📈 对比分析**

与无检索 LLM、单步 RAG 与迭代检索 IterDRAG 对比，ACE 在准确率上提升 19–23pp，且在迭代检索相比下平均 token 消耗降低约 40–50%，同时保持高精度。

**⚠️ 局限性**

局限性包括需手动设定迭代深度，过多迭代会导致准确率下降和 token 消耗激增；不同数据集最佳迭代步数不一致，需进一步研究自适应控制。

---

## 416. A Novel Approach to Explainable AI with Quantized Active Ingredients in Decision Making

**arXiv ID:** 2601.08733 | [PDF](https://arxiv.org/pdf/2601.08733v1)

**作者:** A. M. A. S. D. Alagiyawanna `[一作]` (University of Moratuwa), A. Mahasinghe `[通讯]` (University of Colombo)

**通讯引用:** 104 | [OpenAlex ID](https://openalex.org/A5000871914)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

比较经典Boltzmann机器（CBM）与量子Boltzmann机器（QBM）在MNIST 0/1二分类任务中的准确率与可解释性，使用梯度敏感度和SHAP方法对特征重要性进行分析。

**💡 创新点**

创新点在于将量子-经典混合Boltzmann机与量子梯度可解释性工具相结合，展示量子模型在准确性和解释性上优于经典模型的“活性成分”定位能力。

**🔧 技术方法**

技术包括PennyLane量子电路、角度嵌入与强纠缠层、经典RBM对比散度训练、梯度敏感度映射、SHAP值、PCA降维、t‑SNE可视化与熵评估。

**📊 数据集**

数据集为MNIST手写数字，只保留数字0与1的样本，使用PCA降至四维输入。

**📈 对比分析**

通过训练50轮、学习率0.01的两模型进行对比：QBM准确率83.5%远高于CBM的54%；在特征重要性熵上，QBM熵1.27低于CBM的1.38，表明QBM更集中于关键特征。

**⚠️ 局限性**

局限在于仅验证了极简化的二分类MNIST子集，量子电路规模受限，缺乏对更复杂数据集和更大网络的验证，且量子梯度解释性尚未完全统一与传统SHAP方法。

---

## 417. ISLA: A U-Net for MRI-based acute ischemic stroke lesion segmentation with deep supervision, attention, domain adaptation, and ensemble learning

**arXiv ID:** 2601.08732 | [PDF](https://arxiv.org/pdf/2601.08732v1)

**作者:** Vincent Roca `[一作]` (University of Lille), Renaud Lopes `[通讯]` (University of Lille)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出并实现了ISLA，一个基于3D U‑Net的急性缺血性卒中脑梗塞灶分割深度学习模型，并在多中心数据上进行系统优化。

**💡 创新点**

创新点包括：综合评估损失函数、卷积结构、深度监督、注意力机制、无监督域适配（Mean Teacher）和集成学习，最终得到可公开使用的模型。

**🔧 技术方法**

采用的技术有：3D U‑Net（StdUNet/ResUNet）、Unified Focal Loss、深度监督、SE/AG/CBAM/SEAGs注意力模块、Group Normalization、数据增强、混合精度训练、Mean Teacher UDA及多模型平均集成。

**📊 数据集**

使用的数据集为SOOP、ISLES 2022和HDF（含train/ test）共计1500余例急性卒中患者，外部测试集为HDF‑test（100例）以评估泛化。

**📈 对比分析**

通过在外部测试集与两种SOTA（DAGMNet、DeepISLES）进行对比，采用Dice、AVD、ALD、F1和HD95指标，ISLA‑ENS在所有指标上均优于SOTA，表现出更高的准确度和更低的假阳性。

**⚠️ 局限性**

主要限制为外部测试集样本量较小（100例），仅探索U‑Net体系结构，未尝试Transformer模型，且域适配效果受限于源/目标数据分布差异。

---

## 418. Learning from Demonstrations via Capability-Aware Goal Sampling

**arXiv ID:** 2601.08731 | [PDF](https://arxiv.org/pdf/2601.08731v1)

**作者:** Yuanlin Duan `[一作]` (Rutgers University), He Zhu `[通讯]` (Rutgers University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出一种基于能力感知的目标采样框架（Capability‑Aware Goal Sampling，Cago），利用有限演示轨迹作为结构化路线图，动态监测智能体当前可达范围，挑选处于其能力边界的中间目标，以此驱动目标条件策略在长时延、稀疏奖励环境中的自适应学习。

**💡 创新点**

创新点在于：①不再将演示视为直接监督或奖励塑形，而是将其拆解为可到达的子目标序列；②通过访问计数字典实时评估智能体对演示轨迹的掌握程度，进而在该边界附近采样目标，形成以能力为驱动的渐进式课程；③将两阶段 Go‑Explore 结构与目标条件策略、BC‑探索器以及世界模型相结合，实现高效的数据收集与策略更新。

**🔧 技术方法**

所用技术包括：目标条件强化学习、Go‑Explore 两阶段训练（Go + Explore）、基于行为克隆的探索策略、Dreamer 版世界模型、时间距离估计器、目标预测器以及基于相似度阈值的访问计数字典。

**📊 数据集**

实验数据集涵盖三大基准：MetaWorld 的 5 个“very hard”任务（Shelf Place、Disassemble、Stick Pull、Stick Push、Pick Place Wall）、Adroit 的 3 个手部操控任务（Door、Hammer、Pen）以及 ManiSkill 的 3 个高维视觉任务（PegInsertionSide、StackCube、PullCubeTool）。每个任务仅使用 10–20 条演示轨迹。

**📈 对比分析**

与 Dreamer、JSRL、MoDem、Cal‑QL、GAIL、PWIL、SQIL、ValueDice、RLPD 等基线进行对比。实验显示 Cago 在所有任务上均实现了更快的收敛速度和更高的最终成功率，尤其在长时延稀疏奖励任务中显著优于基线，并在视觉输入任务中保持竞争力。

**⚠️ 局限性**

局限性：算法需要能够将环境重置至演示的起始状态；无法在中间演示状态上重置，限制了在真实机器人上的直接部署；对演示数量与质量敏感，若演示不足或分布差异大，访问计数字典与能力评估可能失效。

---

## 419. Soft Partition-based KAPI-ELM for Multi-Scale PDEs

**arXiv ID:** 2601.08719 | [PDF](https://arxiv.org/pdf/2601.08719v1)

**作者:** Vikas Dwivedi `[一作]` (CREATIS Biomedical Imaging Laboratory), Bruno Sixou `[通讯]` (CREATIS Biomedical Imaging Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于软分区的Kernel‑Adaptive Physics‑Informed Extreme Learning Machine（KAPI‑ELM），通过可调分区长度控制插值中心密度和高斯核宽度，实现无 Fourier 特征、无反向传播的多尺度、振荡和奇异扰动 PDE 求解。

**💡 创新点**

创新点在于：① 用低维分区长度实现连续的粗细自适应采样和核宽度调节；② 消除硬界面约束，提供光滑的粗到细分辨率；③ 引入基于签名距离函数（SDF）的残差加权，提升不规则几何上的稳定性；④ 全部参数化可由贝叶斯优化一次性确定，避免手工调参。

**🔧 技术方法**

技术方法包括：物理信息极限学习机（PI‑ELM）框架；Gaussian RBF 核与软分区采样；SDF 权重残差；分区长度作为低维超参数；贝叶斯优化（Gaussian Process + EI）用于外层参数搜索；单次最小二乘求解得到全部系数。

**📊 数据集**

使用八个合成基准 PDE 数据集：1D 振荡与多尺度 ODE；2D 高频 Poisson 与五瓣形不规则域 Poisson、Biharmonic；1D 奇异扰动对流扩散 ODE，均采用人工制造的解析解来计算误差。

**📈 对比分析**

与 PINN、Fourier‑PINN、FB‑PINN、PI‑ELM、Deep‑TFC、X‑TFC 等现有方法对比，KAPI‑ELM 在相同或更低的计算成本（单次 0.1–10 s）下获得 10⁻⁶–10⁻¹² 的误差，显著优于深度网络，尤其在高频或薄层问题上实现了与 X‑TFC 同等或更高的精度。

**⚠️ 局限性**

局限性包括：仅验证了稳态线性 PDE；未推广到时间依赖、非线性或 3D 问题；SDF 计算与核矩阵规模在大域上可能成为瓶颈；对分区参数的贝叶斯优化仍需实验验证其泛化性。

---

## 420. A Hybrid Model-based and Data-based Approach Developed for a Prosthetic Hand Wrist

**arXiv ID:** 2601.08711 | [PDF](https://arxiv.org/pdf/2601.08711v1)

**作者:** Shifa Sulaiman `[一作]` (University of Naples Federico II), Fanny Ficuciello `[通讯]` (University of Naples Federico II)

**通讯引用:** 3064 | [OpenAlex ID](https://openalex.org/A5065563904)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种基于ANN与滑模控制（SMC）相结合的软连续手腕控制器，用于PRISMA HAND II义手，实现快速、鲁棒的腕部运动控制。

**💡 创新点**

创新点在于将ANN用于预测关节弯曲角度、SMC用于调节肌腱力矩，并采用PCC模型实现软连续腕的几何动力学表征，三者融合大幅提升了控制精度与计算效率。

**🔧 技术方法**

采用的技术包括Piece‑wise Constant Curvature（PCC）建模、人工神经网络（ANN）回归、滑模控制（SMC）、Simulink仿真、Arduino+步进电机执行、ROS视觉跟踪等。

**📊 数据集**

使用了由PCC模型生成的1000条关节角度-位置信息样本作为数据集，70%用于训练、30%用于验证；实验则利用真实制备的手腕结构及外力扰动进行验证。

**📈 对比分析**

与PID、MRAC、GVSC和神经控制器（NC）对比，SMC在仿真中RMSE仅为2.7×10⁻⁴rad、稳态误差2.3×10⁻⁴rad，实验中RMSE约0.16rad，尽管收敛速度略慢于NC，但误差和稳态误差均显著优于其他方法。

**⚠️ 局限性**

主要限制包括实验误差较大（主要受弹簧刚度不匹配影响）、手腕结构易受外部扰动导致的误差增大，以及缺乏实时传感反馈导致的控制精度有限。

---

## 421. RMBRec: Robust Multi-Behavior Recommendation towards Target Behaviors

**arXiv ID:** 2601.08705 | [PDF](https://arxiv.org/pdf/2601.08705v1)

**作者:** Miaomiao Cai `[一作]` (National University of Singapore), Meng Wang `[通讯]` (Hefei University of Technology)

**通讯引用:** 41599 | [OpenAlex ID](https://openalex.org/A5100377147)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于信息理论的鲁棒多行为推荐框架RMBRec，解决辅助行为噪声与语义不一致问题。

**💡 创新点**

创新点在于将表示层局部语义一致性与优化层全局不变性协同优化，使用目标对齐对比学习与分布方差正则化实现。

**🔧 技术方法**

采用对比学习（target‑anchored contrastive）、信息熵与互信息最大化、Invariant Risk Minimization（REx）以及LightGCN编码器。

**📊 数据集**

使用Taobao、TMall、Beibei三大真实电商多行为数据集。

**📈 对比分析**

与单行为LightGCN/SimGCL以及多行为CRGCN、MB‑CGCN、PKEF、S‑MBRec、MBSSL、MISSL、UIPL等基线对比，RMBRec在HR@10/NDCG@10上均超过20%且在噪声扰动下表现更稳健。

**⚠️ 局限性**

局限在于对低BAR辅助行为的对齐仍可能导致负迁移，且对时间动态与更广泛多源场景的适应性尚待研究。

---

## 422. Auditing Student-AI Collaboration: A Case Study of Online Graduate CS Students

**arXiv ID:** 2601.08697 | [PDF](https://arxiv.org/pdf/2601.08697v1)

**作者:** Nifu Dan `[一作]` `[通讯]` (Georgia Institute of Technology), Nifu Dan (Georgia Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在乔治亚理工学院OMSCS研究生中进行一项混合方法调查，量化学生对12种学术任务的期望自动化程度与实际使用程度，并通过开放式问卷挖掘学生对AI风险的关注和对系统功能的设计期望。

**💡 创新点**

①首次以人机中心AI(HCAI)和人类代理尺度(Human Agency Scale)为框架，对学生-AI合作的“期望与实际”对齐进行细粒度审计；②引入四区自动化映射（绿色灯、红灯、研发机会、低优先级），系统化描述学生在不同任务上的自动化倾向与使用差距；③以开放式回应构建可解释、透明、可信AI的需求地图。

**🔧 技术方法**

使用定量Likert量表（1–5）评估自动化期望与实际使用；定性文本编码与主题分析提炼系统功能需求；利用Human Agency Scale量表与自我报告使用频率构造聚合指标。

**📊 数据集**

调查样本为44名OMSCS（计算机科学）研究生，涵盖12个学术任务（如摘要、编码、写作、学习计划等）。

**📈 对比分析**

对比学生的期望自动化与自报使用水平，绘制四区对齐图；在定性层面对开放式回答进行主题归纳，形成设计原则清单。未涉及模型性能评估或对比基准，因研究焦点在用户体验与需求。

**⚠️ 局限性**

①样本规模小且仅限CS研究生，结果可能不具备跨学科推广性；②使用自我报告数据，未与专家评估或实际AI表现对比，难以验证AI的真实能力；③未考察不同AI工具差异，只关注通用的“AI系统”概念，可能忽略工具特定特征。

---

## 423. VLingNav: Embodied Navigation with Adaptive Reasoning and Visual-Assisted Linguistic Memory

**arXiv ID:** 2601.08665 | [PDF](https://arxiv.org/pdf/2601.08665v1)

**作者:** Shaoan Wang `[一作]` (ByteDance), Junzhi Yu `[通讯]` (Peking University)

**通讯引用:** 11434 | [OpenAlex ID](https://openalex.org/A5073958329)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于视觉-语言-动作（VLA）框架，通过自适应链式思考（AdaCoT）和视觉辅助语言记忆（VLingMem）实现高效、可解释的智能机器人导航，并在模拟与真实机器人上实现零样本迁移。

**💡 创新点**

创新点在于（1）AdaCoT机制根据情境动态触发思考，仅在必要时进行慢速推理；（2）VLingMem将视觉信息转化为可长期保持的语言摘要，解决传统 VLA 模型的短视记忆问题；（3）联合监督学习与专家引导的在线强化学习，克服单纯模仿学习的局限性；（4）构建 Nav‑AdaCoT‑2.9M 大规模自适应链式思考数据集，填补现有数据空白。

**🔧 技术方法**

采用 LLaVA‑Video‑7B 作为 VLM 视觉编码器，结合两层 MLP 投影、RoPE 时序编码、动态 FPS 采样与网格池化；使用多任务训练、在线专家指导的 PPO‑style RL 以及混合数据收集策略；动作输出采用多元高斯连续控制模型。

**📊 数据集**

主要使用 Nav‑AdaCoT‑2.9M（包含 2.9M 步长导航与 1.6M 开放世界视频），并在 HM3D、MP3D、EVT‑Bench、HM3D Instance ImageNav 等公开基准上进行评估。

**📈 对比分析**

在 ObjectNav、EVT、ImageNav 三大任务中均取得或突破 SOTA（例如 ObjectNav SR 83%/SPL 40.5，EVT SR 88.4/TR 81.2，ImageNav SR 60.8/SPL 37.4），并在 Unitree Go2 四足机器人上实现无额外微调的零样本真实环境导航，表现出优异的泛化与实时推理能力。

**⚠️ 局限性**

主要局限为仅使用单目视角，导致视野受限；采用单系统架构，决策频率受限；控制仅基于 MPC，缺乏更灵活的运动学控制。

---

## 424. RULERS: Locked Rubrics and Evidence-Anchored Scoring for Robust LLM Evaluation

**arXiv ID:** 2601.08654 | [PDF](https://arxiv.org/pdf/2601.08654v1)

**作者:** Yihan Hong `[一作]` (Washington University in St. Louis), Yushun Dong `[通讯]` (Florida State University)

**通讯引用:** 860 | [OpenAlex ID](https://openalex.org/A5047581320)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Rulers框架，将自然语言rubric编译为可执行规范，解决LLM-as-a-Judge的鲁棒性、证据可验证性与评分尺度失配问题。

**💡 创新点**

创新点在于三阶段流程：rubric锁定、证据验证的结构化解码以及Wasserstein后置校准，全部在不更新模型参数的前提下实现高度可审计且与人类评分一致的评估。

**🔧 技术方法**

使用了编译器将rubric转为JSON规范、结构化提取式解码与确定性验证、Wasserstein生成回归（WGR）进行分布校准以及零参数的指令化交互。

**📊 数据集**

实验基准包括ASAP 2.0、SummHF和DREsS，涵盖议论文、摘要和英语学习者作文三类任务。

**📈 对比分析**

与DHS、MTS、AutoScore等基线在相同模型与数据集上对比，采用Quadratic Weighted Kappa评估；Rulers在所有模型规模下均显著优于基线，尤其在鲁棒性、分布对齐和人类一致性上表现突出。

**⚠️ 局限性**

局限性包括对rubric细化质量的依赖、强制提取式证据可能忽略整体语义、确定性匹配对格式差异敏感，以及后置校准需标注数据，跨域迁移时可能需重新校准。

---

## 425. Agent Contracts: A Formal Framework for Resource-Bounded Autonomous AI Systems

**arXiv ID:** 2601.08815 | [PDF](https://arxiv.org/pdf/2601.08815v1)

**作者:** Qing Ye `[一作]` (Independent Researcher), Jing Tan `[通讯]` (Independent Researcher)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Agent Contracts 形式化框架，统一了输入/输出、资源、时间与成功等约束，并在多代理协作中实现资源守恒定律，实现可审计、可资源约束的 AI 部署。

**💡 创新点**

创新点在于将合同理论与多代理协调和实时系统的资源受限计算相结合，定义七元组合同，提供多维资源预算、时间边界和成功标准，并引入多代理协作中的资源守恒定律与合同委托。

**🔧 技术方法**

技术上使用合同理论、多代理系统协调、实时系统资源受限计算；实现基于 LLM 调用、工具调用的监控与动态预算推断；实验基于 Google ADK、LiteLLM、Gemini 2.5 Flash，并采用 Bootstrap 置信区间分析。

**📊 数据集**

使用的数据集包括 LiveCodeBench（Python 代码评审）、OpenR1 逻辑推理问题、Crisis Communication 场景、以及多类别研究主题等。

**📈 对比分析**

实验通过对比有无合同的情况，评估 token 使用、迭代次数、LLM 调用次数与成功率；结果显示有合同可使 token 减少 90%，方差降低 525 倍，资源守恒无违规，合同模式可实现从 70% 到 86% 的成功率。

**⚠️ 局限性**

局限性在于 LLM 调用只能在完成后才知道 token 消耗，无法实时强制约束；当前只能在多次调用或多代理场景提供最佳努力，缺乏硬性保证，需 API 支持实时中断、预留 token 等功能。

---

## 426. APEX-SWE

**arXiv ID:** 2601.08806 | [PDF](https://arxiv.org/pdf/2601.08806v1)

**作者:** Abhi Kottamasu `[一作]` (Mercor), Bertie Vidgen `[通讯]` (Mercor)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了APEX–SWE基准，用于评估AI模型在真实软件工程任务（端到端集成与生产故障诊断）中的表现。

**💡 创新点**

创新点在于设计两类逼真的跨平台任务并强调模型的“认知学问”——区分假设与事实并主动验证，从而模拟专业工程师的推理与迭代过程。

**🔧 技术方法**

技术上使用多步工具调用（终端、文件操作、MCP服务）与持续迭代的评估 harness，评测指标为 Pass@1 与 Pass@3 以及基于 Rubric 的功能、鲁棒性、风格评分。

**📊 数据集**

数据集包含 200 个真实任务（100 个 Integration、100 个 Observability），并公开 50 个开发集，任务来源为真实 GitHub Issue–PR 对、云原生服务（LocalStack、EspoCRM、Medusa 等）以及模拟的日志与聊天记录。

**📈 对比分析**

对八个前沿模型进行比较，最高通过率为 Gemini 3 Pro 25%（Pass@1），Claude 系列 14–23%；在功能与鲁棒性评估中，Gemini 与 Claude 取得 80% 以上功能得分，整体性能仍远低于 90% 的单函数基准，表明仍有显著提升空间。

**⚠️ 局限性**

局限性包括低通过率、对某些语言（Java 0%）与日志量大的任务表现不佳、任务样本规模相对有限、以及对多语言编译与框架特性的支持不足。

---

## 427. Fast and explainable clustering in the Manhattan and Tanimoto distance

**arXiv ID:** 2601.08781 | [PDF](https://arxiv.org/pdf/2601.08781v1)

**作者:** Stefan Güttel `[一作]` (University of Manchester), Kaustubh Roy `[通讯]` (University of Manchester)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

扩展 CLASSIX 聚类算法到曼哈顿距离和 Tanimoto 距离，并给出相应的早停搜索规则；

**💡 创新点**

提出了基于任意范数的排序与逆三角不等式的通用终止准则，在 Tanimoto 距离中引入 Baldi 交叉不等式实现更紧的剪枝，并通过 orthant shifting 改进曼哈顿距离的搜索效率；

**🔧 技术方法**

采用两阶段贪婪聚合与合并，使用逆三角不等式、Baldi 不等式进行距离剪枝，利用整数或位运算计算距离，并构建可解释的邻接矩阵以支持聚类解释；

**📊 数据集**

使用合成高维二进制向量、化学指纹（Morgan fingerprint）91,662 分子、Iris、Banknote、MNIST 等公开数据集，并在部分实验中采用 UMAP 降维；

**📈 对比分析**

与 DBSCAN、OPTICS、Taylor–Butina 等方法比较，结果显示在化学指纹上 CLASSIX_T 速度比 Taylor–Butina 快 30 倍、DBSCAN 快 80 倍，同时 ARI 更高；在合成二进制数据上运行时间随样本数线性增长且明显低于 DBSCAN；在小数据集 CLASSIX_M 较慢，但在大规模 MNIST 数据上表现出更好的可扩展性；

**⚠️ 局限性**

局限性包括：在低分散或小簇场景下搜索终止效率下降；曼哈顿版本对离群点处理较慢；目前未充分利用位运算加速二进制距离计算；理论分析基于均匀分布模型，需进一步验证；未来可通过 GPU 并行化和更精准的分布假设进一步提升。

---

## 428. LWM-Spectro: A Foundation Model for Wireless Baseband Signal Spectrograms

**arXiv ID:** 2601.08780 | [PDF](https://arxiv.org/pdf/2601.08780v1)

**作者:** Namhyun Kim `[一作]` (Arizona State University), Ahmed Alkhateeb `[通讯]` (Arizona State University)

**通讯引用:** 14521 | [OpenAlex ID](https://openalex.org/A5003243464)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

训练了一个基于Transformer的无线基础层频谱基础模型LWM‑Spectro，利用自监督掩码建模、对比学习和混合专家架构学习可迁移的时频特征，并在调制识别、SNR/移动性联合分类等下游任务中微调；

**💡 创新点**

①将无线I/Q信号转换为时频频谱并进行大规模预训练；②在同一模型中引入混合专家实现协议感知特征提取；③结合掩码自监督与监督对比学习实现高效、可迁移的表示；

**🔧 技术方法**

Transformer编码器、掩码频谱建模、监督对比学习、Mixture‑of‑Experts路由器、轻量级1D CNN分类头；

**📊 数据集**

基于DeepMIMO、3GPP TDL等生成的9.2百万个WiFi/LTE/5G I/Q频谱样本，覆盖多城市、多协议、多SNR、多速度条件；

**📈 对比分析**

与传统CNN、ImageNet预训练模型（ResNet‑18、EfficientNet‑B0、MobileNet‑V3）以及专用Deep CNN比较，LWM‑Spectro在少样本和大样本场景下在调制识别、SNR/移动性联合分类等任务中显著优于基线，尤其在few‑shot下F1提升超过30%并接近饱和性能；

**⚠️ 局限性**

需要大规模无标签数据生成与预训练，训练成本高；Mixture‑of‑Experts路由器在极端协议混合或未知协议下可能失效；对极低SNR或极端多径环境的鲁棒性仍待进一步验证。

---

## 429. Reliable Graph-RAG for Codebases: AST-Derived Graphs vs LLM-Extracted Knowledge Graphs

**arXiv ID:** 2601.08773 | [PDF](https://arxiv.org/pdf/2601.08773v1)

**作者:** Manideep Reddy Chinthareddy `[一作]` `[通讯]` (Centerville), Manideep Reddy Chinthareddy (Centerville)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对三种代码库检索增强生成（RAG）管道进行基准：向量检索、LLM 提取知识图谱和基于 AST 的确定性知识图谱，评估其索引、查询、成本与答案正确性。

**💡 创新点**

提出了基于 Tree‑sitter 的确定性 AST‑导图（DKB）并实现双向遍历与接口消费者扩展，以提升多跳结构推理；同时系统性量化 LLM‑提取图谱的不完整性与成本膨胀。

**🔧 技术方法**

技术包括：向量相似度检索、LLM 结构化输出提取、Tree‑sitter 语法解析、图遍历与扩展、成本归一化分析、可视化与日志记录。

**📊 数据集**

数据集为三个真实 Java 代码库：Shopizer（电商）、ThingsBoard（物联网）、OpenMRS Core（电子病历），每个库均使用 15 题的架构与追踪问答集合。

**📈 对比分析**

通过统一的 15 题 benchmark，比较指标包括索引时间、查询时延、代码块与图覆盖率、成本。DKB 的索引时间接近向量基线，查询时延略高但稳定，答案正确率 15/15；LLM‑KB 正确率 13/15，索引覆盖率低、成本 19‑45 倍；向量基线时延最快，但多跳问题答案多失真。

**⚠️ 局限性**

局限：LLM‑KB 的抽取过程随机且易漏文件，导致索引不完整；DKB 的 AST 规则无法捕获所有语义（如反射、动态生成代码）且对文件命名/路径匹配敏感；评估仅覆盖三 Java 仓库，缺乏跨语言与更大规模的验证；答案评判依赖单一人工标注，缺少多评标一致性。

---

## 430. Majority-Logic Decoding of Binary Locally Recoverable Codes: A Probabilistic Analysis

**arXiv ID:** 2601.08765 | [PDF](https://arxiv.org/pdf/2601.08765v1)

**作者:** Hoang Ly `[一作]` (Rutgers University), Philip Whiting `[通讯]` (Macquarie University)

**通讯引用:** 4444 | [OpenAlex ID](https://openalex.org/A5080012042)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文通过对二进制局部可恢复码（LRC）在随机错误与擦除模型下的多数逻辑解码（MLD）进行概率性分析，推导了比特错误率（BER）与块错误率（BLER）的上界，并证明了当可恢复度随码长增长时，块误码概率趋于零，且MLD能纠正线性比例的随机错误/擦除；

**💡 创新点**

创新点在于将MLD应用于LRC的随机误码环境，首次给出关于局部性与可恢复度的概率性误码性能界限，揭示了最坏情况与典型情况之间的巨大性能差距，并证明在可恢复度足够大时，MLD几乎可纠正所有“典型”错误模式；

**🔧 技术方法**

采用多数逻辑解码框架，结合大偏差理论（Markov、Chernoff 边界）、联合界与大数定律等概率工具，得到BER/BLER 的指数衰减表达式；

**📊 数据集**

使用仿真数据对不同可恢复度增长速率（线性、polylog、sub-log）下的随机码进行评估，采用人工生成的二进制LRC实例；

**📈 对比分析**

通过与理论上界的对比，仿真显示BER 与BLER 均低于上界；结果表明线性或足够高阶的 polylog 可恢复度可使误码率快速下降，而 sub-log 缓慢增长时误码率无法消失；

**⚠️ 局限性**

局限性包括：仅研究二进制LRC；假设可恢复度可随码长任意增长；MLD 仅为硬判决解码，对软判决或更一般的 q-ary 码的适用性待进一步研究。

---

## 431. FusID: Modality-Fused Semantic IDs for Generative Music Recommendation

**arXiv ID:** 2601.08764 | [PDF](https://arxiv.org/pdf/2601.08764v1)

**作者:** Haven Kim `[一作]` (University of California San Diego), Julian McAuley `[通讯]` (University of California San Diego)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 FusID，一种融合多模态特征的语义 ID 生成框架，用于生成式推荐。

**💡 创新点**

创新点在于通过单一融合网络联合编码多模态信息，结合对比学习和正则化损失，消除 ID 冲突并提高码本利用率。

**🔧 技术方法**

使用对比学习、正则化（VICReg 风格）、产品量化和 GPT‑2 生成模型等技术。

**📊 数据集**

在 Million Playlist Dataset（MPD）上进行实验。

**📈 对比分析**

与 Baseline TalkPlay 和 SASRec 比较，FusID 在冲突率为 0、码本利用率 100% 的同时，MRR 与 Recall@k 均比最优基线提升约 6% 至 15%。

**⚠️ 局限性**

局限在于实验仅覆盖音乐推荐领域，且对正则化系数敏感，缺乏跨领域验证。

---

## 432. Near-perfect photo-ID of the Hula painted frog with zero-shot deep local-feature matching

**arXiv ID:** 2601.08798 | [PDF](https://arxiv.org/pdf/2601.08798v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 433. MemRec: Collaborative Memory-Augmented Agentic Recommender System

**arXiv ID:** 2601.08816 | [PDF](https://arxiv.org/pdf/2601.08816v1)

**作者:** Weixin Chen `[一作]` (Hong Kong Baptist University), Yongfeng Zhang `[通讯]` (Rutgers University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了 MemRec 框架，通过将记忆管理（LM_Mem）与推理（LLM_Rec）解耦，实现协作记忆检索、合成和异步传播，提升代理式推荐系统的性能。

**💡 创新点**

创新点包括：① 零射 LLM 生成域适配的裁剪规则，解决协作检索时的认知过载；② 基于标签传播的异步协作传播，将更新成本降到 O(1)；③ 通过“先裁剪后合成”信息瓶颈策略，提取高价值协作上下文；④ 在单一框架内兼顾推理质量、成本和隐私，构建新的 Pareto 前沿。

**🔧 技术方法**

核心技术有：LLM 辅助的零射规则生成与记忆裁剪；多层次记忆合成与上下文提示；基于 LLM 的协作记忆合成与更新；异步批量传播；GPT‑4o‑mini 作为推理与记忆管理的 LLM。

**📊 数据集**

使用四个基准数据集：Amazon Books、Amazon Goodreads、MovieTV、Yelp，均包含用户说明式推荐任务。

**📈 对比分析**

与 LightGCN、SASRec、P5、Vanilla LLM、iAgent、RecBot、AgentCF、i^2Agent 等传统与代理式基线比较，MemRec 在所有四个数据集上均获得显著提升（如 Goodreads 上 H@1 提升 28.98%，MovieTV 上 H@1 提升 19.75%，总体提升 10–30%），同时在成本、延迟与隐私上形成更优的 Pareto 前沿。

**⚠️ 局限性**

局限性：① 传播仅限于一跳邻居，无法覆盖更大社区；② 裁剪规则基于离线域统计，缺乏实时自适应；③ 仍依赖强大的专有 LLM，完整的开源实现尚未达到极限性能。

---

## 434. Reasoning Matters for 3D Visual Grounding

**arXiv ID:** 2601.08811 | [PDF](https://arxiv.org/pdf/2601.08811v1)

**作者:** Hsiang-Wei Huang `[一作]` (University of Washington), Jenq-Neng Hwang `[通讯]` (University of Washington)

**通讯引用:** 12323 | [OpenAlex ID](https://openalex.org/A5101702810)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一个完全自动化的3D视觉定位数据生成管道，生成包含四阶段结构化推理的训练样本，并利用这些样本对开源LLM进行微调，得到Reason3DVG-8B。

**💡 创新点**

创新点包括：①无需人工注释即可生成大规模3D场景与查询；②在每个样本中提供详细、结构化的推理过程，用于训练LLM的推理能力；③证明推理监督比单纯扩大数据规模更关键；④仅用3D-GRAND 1.6%的数据即可超越其性能。

**🔧 技术方法**

技术手段：程序化3D场景生成、Mask3D物体检测器生成提案、四阶段推理模板（相关对象选择、情境估计、推理、结论）、Llama‑3.1‑8B微调（交叉熵训练）以及LLM生成推理文本。

**📊 数据集**

使用自生成的3.2K样本（包含50+对象的3D场景、查询及推理），并在公开基准ScanRefer和NR3D上进行评估。

**📈 对比分析**

与零样本方法ZSVG3D、SeeGround及3D-GRAND等进行比较；在ScanRefer上实现了SOTA零样本性能，在NR3D上也超过所有零样本SOTA；仅用1.6%数据就能超越3D-GRAND，整体准确率提升约16%。

**⚠️ 局限性**

主要局限：性能受Mask3D检测器生成的对象提案质量限制；生成的数据不包含过于复杂的真实世界查询；需要更强的检测/视觉特征来进一步提升效果。

---

## 435. Multiplex Thinking: Reasoning via Token-wise Branch-and-Merge

**arXiv ID:** 2601.08808 | [PDF](https://arxiv.org/pdf/2601.08808v1)

**作者:** Yao Tang `[一作]` (University of Pennsylvania), Jiatao Gu `[通讯]` (University of Pennsylvania)

**通讯引用:** 10353 | [OpenAlex ID](https://openalex.org/A5112542984)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种称为 Multiplex Thinking 的推理机制，利用 K 次独立采样的离散 token 并将其词嵌入聚合成一个连续多路 token，从而在保持采样动态的同时实现信息压缩，并支持直接使用 on‑policy 强化学习进行优化。

**💡 创新点**

创新点在于：1）将离散采样与连续表示结合，形成可训练的多路 token；2）通过多路 token 让单一步骤保留多条潜在推理路径，既保持概率分布又提升信息密度；3）构造可解析的多路 token 概率分布，使得完整推理轨迹可被直接优化；4）在不显著增加序列长度的前提下，显著提升 Pass@k 的上限。

**🔧 技术方法**

主要技术包括：K 次独立采样、词嵌入聚合（平均或加权），对多路 token 采用离散采样的概率分解，使用 Group Relative Policy Optimization (GRPO) 进行 on‑policy RL，结合 top‑p、温度调参进行采样。

**📊 数据集**

使用的训练数据为 DeepScaleR-Preview-Dataset（约40k 个题目答案对），评估数据集包括 AIME 2024/2025、AMC 2023、MATH‑500、Minerva Math、OlympiadBench 等数学推理基准。

**📈 对比分析**

与离散 Chain‑of‑Thought、训练无关的 Soft Thinking、以及基于同一模型的离散 RL 进行对比。实验表明 Multiplex Thinking 在 Pass@1 上在 11/12 组设置中获得最佳成绩，并且在 Pass@k（k=1…1024）上往往达到更高的上限；此外相同或更短的序列长度即可获得更高准确率。

**⚠️ 局限性**

局限性包括：1）K 的增大虽然能提升性能但收益递减，且需权衡采样开销；2）对大模型的依赖较强，较小模型效果有限；3）多路 token 聚合方式对不同任务可能需调优；4）实验主要集中在数学推理任务，对其他类型的推理场景尚未充分验证。

---

## 436. Memory DisOrder: Memory Re-orderings as a Timerless Side-channel

**arXiv ID:** 2601.08770 | [PDF](https://arxiv.org/pdf/2601.08770v1)

**作者:** Sean Siddens `[一作]` (University of Washington), Tyler Sorensen `[通讯]` (Microsoft Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种利用主流CPU和GPU的内存重排序（无计时器）作为侧信道的新攻击手段，并在此基础上实现了低成本的隐写通道和模型指纹识别。

**💡 创新点**

创新点在于：①无需计时器、硬件监控或深度反汇编；②通过简单的litmus测试和系统压力可在跨进程、跨虚拟化边界获得可靠信号；③展示了利用低级硬件细节（如L1缓存集）可将隐写速率提升至近30 kbps。

**🔧 技术方法**

技术方法包括：自定义的litmus测试框架、内存/线程启动压力、统计检验（Mann‑Whitney、t‑test）与窗口分类、以及针对不同架构的调参与缓存失效策略。

**📊 数据集**

数据集主要是基于在六类主流设备（Arm、x86、Apple CPU、NVIDIA/AMD GPU）上收集的重排序计数；用于指纹识别的测试对象为5种常见DNN架构（ResNet‑50、GoogLeNet、VGG16、MobileNet‑V3、AlexNet）。

**📈 对比分析**

与现有计时器侧信道或缓存侧信道比较，本文攻击在无计时器环境下仍能实现超过95 %准确率的隐写通道，CPU端速率0.3–0.6 bps，GPU端可达16 bps；指纹识别在100样本时准确率可达80‑95%，最高可达99%。

**⚠️ 局限性**

局限性在于：①当前实现依赖经验调参，针对不同硬件需手工调试；②对系统噪声敏感，CPU端速率受限；③未对加密密钥等高精度攻击做完整实验；④需进一步研究重排序产生机制以实现更稳健的防御。

---

## 437. Rewarding the Rare: Uniqueness-Aware RL for Creative Problem Solving in LLMs

**arXiv ID:** 2601.08763 | [PDF](https://arxiv.org/pdf/2601.08763v1)

**作者:** Zhiyuan Hu `[一作]` (Massachusetts Institute of Technology), Bryan Hooi `[通讯]` (National University of Singapore)

**通讯引用:** 5314 | [OpenAlex ID](https://openalex.org/A5065675832)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对大型语言模型在后训练强化学习中出现的策略崩塌问题，本文提出了基于解题策略唯一性的强化学习目标，鼓励模型产生多样化且正确的解法。

**💡 创新点**

创新点在于：① 用LLM裁判对同一问题的多条推理轨迹按高层解题思路聚类；② 通过聚类大小对优势函数加权，使稀有但正确的策略获得更高奖励，从而在同一问题的样本集合中维持多策略覆盖。

**🔧 技术方法**

技术方法包括：Group Relative Policy Optimization（GRPO）框架、LLM裁判模型进行聚类、优势加权（1/clusterSize^α）、基于验证器的奖励计算、以及训练时的熵与KL正则化。

**📊 数据集**

使用的评测数据集包括：数学（MATH、AIME、HLE）、物理（OlympiadBench、MegaScience物理子集）以及医学（MedCaseReasoning）等多个高难度推理基准。

**📈 对比分析**

与 Instruct 基础模型、SimpleRL（仅GRPO）、DAPO、Forking Token 等方法相比，本文方法在 pass@k、AUC@K、以及覆盖率（cover@n）上均取得更佳表现，尤其在中大采样预算下显著提升多策略覆盖，且不降低 pass@1。

**⚠️ 局限性**

局限性：① 依赖LLM裁判聚类，计算开销大且聚类误差可能影响奖励；② 仅衡量单个问题内的稀有度，未考虑跨问题的长期多样性；③ 高层策略定义依赖任务，聚类的鲁棒性需进一步验证。

---

## 438. RAVEN: Erasing Invisible Watermarks via Novel View Synthesis

**arXiv ID:** 2601.08832 | [PDF](https://arxiv.org/pdf/2601.08832v1)

**作者:** Fahad Shamshad `[一作]` (Mohammed Bin Zayed University of Artificial Intelligence), Karthik Nandakumar `[通讯]` (Michigan State University)

**通讯引用:** 10717 | [OpenAlex ID](https://openalex.org/A5059676452)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于视图合成的隐形水印去除方法，利用零射扩散模型在潜空间对水印图像做视角变换并保持语义一致，从而在不破坏视觉质量的前提下抑制水印。

**💡 创新点**

创新点包括：①将水印去除重新框架为视角合成问题；②在潜空间实现全局视角平移并配合视角引导对应注意力，既保留语义又打乱水印；③无需访问水印检测器或水印知识，做到零射黑盒攻击。

**🔧 技术方法**

核心技术：Stable Diffusion 图像到图像扩散模型、DDIM 逆扩散、潜空间视角平移、视角引导对应注意力、CIELAB 颜色与对比度转移、零射扩散流程。

**📊 数据集**

使用的数据集包括：MS‑COCO‑2017（5,000 条）、DiffusionDB 2M（1,001 条）、Stable‑Diffusion‑Prompts（8,192 条），并在 14 种语义/位流水印方案上进行评估。

**📈 对比分析**

与 15 种基线（像素级处理、再生攻击、CtrlGen+、UnMarker 等）在 16 种水印上对比。该方法平均 TPR@1%FPR 0.026（相较最佳基线 0.078 下降 60%），FID 分别为 40.18/49.47/47.11，CLIP 分数 0.328/0.364/0.350，均达到或超过所有基线，表明在水印抑制和视觉质量上均实现领先。

**⚠️ 局限性**

局限性：① 需要调节噪声注入强度参数 s，取值过大会产生视觉伪影；② 目前主要针对平移视角的变换，对极端几何变换或非平移视角效果待验证；③ 仅针对单张图像的零射推理，批量处理的速度和资源消耗未充分评估；④ 对未来针对潜空间对抗训练或多尺度视角攻击仍可能存在脆弱性。

---

## 439. Modeling LLM Agent Reviewer Dynamics in Elo-Ranked Review System

**arXiv ID:** 2601.08829 | [PDF](https://arxiv.org/pdf/2601.08829v1)

**作者:** Hsiang-Wei Huang `[一作]` (University of Washington), Jenq-Neng Hwang `[通讯]` (University of Washington)

**通讯引用:** 12323 | [OpenAlex ID](https://openalex.org/A5101702810)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了基于大型语言模型的多轮评审仿真框架，并在其中加入Elo排名机制来研究评审者动态。

**💡 创新点**

首次将Elo评分体系引入评审流程，并通过LLM代理模拟不同评审者人格，探究其对评审质量和主办人决策的影响。

**🔧 技术方法**

使用Gemini‑2.5‑Flash LLM为评审者和主席生成评审文本，并实现Elo调整与记忆模块。

**📊 数据集**

采集了ICLR 2025会议的150篇论文的真实提交内容作为仿真输入。

**📈 对比分析**

对比了基线、AC访问、全访问三种设置，发现加入Elo后Area Chair决策准确率提升，且全访问下评审者对Elo做出战略调整但未必提升实际评审质量。

**⚠️ 局限性**

受限于仅30轮仿真与计算资源，无法观察长期收敛和稳定性，且Elo反馈可能导致评审者策略化行为。

---

## 440. Motion Attribution for Video Generation

**arXiv ID:** 2601.08828 | [PDF](https://arxiv.org/pdf/2601.08828v1)

**作者:** Xindi Wu `[一作]` (NVIDIA), Jonathan Lorraine `[通讯]` (NVIDIA)

**通讯引用:** 259 | [OpenAlex ID](https://openalex.org/A5030776397)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于梯度的运动属性化方法，用来评估训练视频对生成模型运动质量的影响，并据此选择最具影响力的数据进行微调。

**💡 创新点**

创新点在于将运动信息通过运动加权掩模融入梯度计算，显著抑制静态外观影响，并通过帧长归一化和结构化投影实现了大规模高效评估；该方法首次在视频生成中实现了运动级别的数据归因与利用。

**🔧 技术方法**

使用了梯度相似度与Fastfood投影的组合、AllTracker光流掩模、VAE潜在空间映射以及自定义运动加权损失，构成端到端的运动归因框架。

**📊 数据集**

在两大公开视频集 VIDGEN‑1M 与 4DNeX‑10M 上挑选 10k 条样本，并在 Wan2.1‑T2V‑1.3B 预训练模型上进行实验。

**📈 对比分析**

与基线（随机采样、全量微调、无运动掩模等）相比，使用归因挑选的 10% 数据即可在 VBench 的运动平滑度与动态度上达到或超过全量微调的效果，且在 50 条视频的人工评测中获得 74.1% 的胜率。

**⚠️ 局限性**

局限包括对 AllTracker 运动掩模的依赖（遮挡或透明物体导致误判）、难以分离纯相机运动以及对极细微微动的识别不足。

---

## 441. Older Adults' Preferences for Feedback Cadence from an Exercise Coach Robot

**arXiv ID:** 2601.08819 | [PDF](https://arxiv.org/pdf/2601.08819v1)

**作者:** Roshni Kaushik `[一作]` (Carnegie Mellon University), Reid Simmons `[通讯]` (Carnegie Mellon University)

**通讯引用:** 15581 | [OpenAlex ID](https://openalex.org/A5064960456)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

评估了老年人对机器人锻炼教练的口头与非口头反馈节奏的偏好，设计并在线实验验证不同节奏对认知和情感体验的影响。

**💡 创新点**

发现口头和非口头反馈节奏相互影响，并提出针对老年人优化反馈频率的设计指南。

**🔧 技术方法**

采用基于运动评估的机器人反馈控制器，生成口头与非口头多模态反馈。

**📊 数据集**

使用由100名60岁以上的在线参与者观看的视频数据集。

**📈 对比分析**

通过重复测量ANOVA和配对后检验对比各节奏水平的Likert量表结果，显示中高口头节奏更清晰有用，中等非口头节奏更及时有用。

**⚠️ 局限性**

局限在视频观察而非实时交互，且反馈对自身行为的影响未得到验证。

---

## 442. Aggregating Diverse Cue Experts for AI-Generated Image Detection

**arXiv ID:** 2601.08790 | [PDF](https://arxiv.org/pdf/2601.08790v1)

**作者:** Lei Tan `[一作]` (National University of Singapore), Robby T. Tan `[通讯]` (ASUS Intelligent Cloud Services)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种多线索聚合网络（MCAN），用于检测人工合成图像，能够同时利用图像内容、高频信息和色度不一致性（CI）三种线索进行判别；

**💡 创新点**

创新点在于：①将三种互补线索统一映射到同一网络；②采用混合编码器适配器（MoEA）动态融合不同线索特征；③设计CI变换消除光照影响，凸显真实图像噪声差异；

**🔧 技术方法**

技术手段包括：基于冻结CLIP ViT‑B/16的特征提取；混合编码器适配器（Mixture‑of‑Encoder Adapter）实现多专家动态路由；色度不一致性变换与位置嵌入打乱；多任务损失（交叉熵+重要性损失+熵损失）训练；

**📊 数据集**

使用的公开数据集：GenImage、Chameleon 和 UniversalFakeDetect（包含数千张真实图像与多种生成模型的合成图像）；

**📈 对比分析**

与多种 SOTA 方法（CNNSpot、FreDect、Fatformer、LaRE²、NPR、AIDE 等）在三大基准上对比，MCAN 在 GenImage 上平均 96.9% ACC、Chameleon 上 69.6% ACC、UniversalFakeDetect 上 98.8% mAcc，分别比现有最佳方案提升约 7–10%，显示出更强的跨模型泛化能力；

**⚠️ 局限性**

局限性包括：对极低分辨率或非 RGB 采样的图像效果尚未验证；当专家数量不足或过多时性能会下降；对某些新型生成模型仍可能出现误判；整体计算成本相对较高。

---

## 443. MixServe: An Automatic Distributed Serving System for MoE Models with Hybrid Parallelism Based on Fused Communication Algorithm

**arXiv ID:** 2601.08800 | [PDF](https://arxiv.org/pdf/2601.08800v1)

**作者:** Bowen Zhou `[一作]` (Southeast University), Fang Dong `[通讯]` (Southeast University)

**通讯引用:** 14596 | [OpenAlex ID](https://openalex.org/A5108490672)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为MixServe的自动分布式服务系统，旨在通过基于融合通信算法的混合并行性高效部署Mixture of Experts (MoE)模型。

**💡 创新点**

MixServe的创新点在于其TP-EP混合并行性和融合AR-A2A通信算法，能够有效重叠节点内和节点间的通信，从而减少通信延迟。

**🔧 技术方法**

使用了融合的AR-A2A通信算法，结合了张量并行性（TP）和专家并行性（EP）以优化通信开销。

**📊 数据集**

在DeepSeek-R1和Qwen3模型上进行了广泛的实验，使用了ShareGPT-V3数据集进行基准评估。

**📈 对比分析**

与现有方法相比，MixServe在时间到第一个令牌（TTFT）上加速了1.08×至3.80×，在令牌间延迟（ITL）上加速了1.03×至1.66×，并在吞吐量上提高了5.2%至50.3%。

**⚠️ 局限性**

MixServe的局限性在于其依赖于网络拓扑和硬件资源的配置，可能在不同的环境中表现不一致。

---

## 444. DentalX: Context-Aware Dental Disease Detection with Radiographs

**arXiv ID:** 2601.08797 | [PDF](https://arxiv.org/pdf/2601.08797v1)

**作者:** Zhi Qin Tan `[一作]` (Centre for Oral, Clinical and Translational Sciences), Yunpeng Li `[通讯]` (Centre for Oral, Clinical and Translational Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 DentalX，联合学习牙科疾病检测与牙齿解剖分割，利用结构上下文提取模块增强检测；

**💡 创新点**

创新点在于：①构建单向结构上下文提取模块(SCE)，将辅助分割任务的结构信息注入检测头；②针对不同任务的部分标注数据实现联合训练；③通过多任务学习显著提升牙科图像的检测与分割性能；

**🔧 技术方法**

技术实现：基于 YOLOX 的 FPN backbone，SCE 模块（上采样、卷积、像素级交叉熵+IoU 损失），联合损失函数，PyTorch 实现，SGD + cosine annealing 训练；

**📊 数据集**

使用数据集：10,121 张含 20 类疾病的内部检测数据；1,556 张 periapical + 4,388 张 bitewing 的解剖分割数据，6 类结构（Enamel, Dentin, Root Dentin, Pulp, Bone, Implants）；

**📈 对比分析**

与 Faster R‑CNN、YOLOX、DINOv3 等检测基线比较，DentalX 在 AP@0.5 为 45.9（比 YOLOX 高 5.2），在 AP@0.75 为 17.1，平均 AP0.5:0.95 为 21.0；在分割任务上与 UNet、SETR、PIDNet 对比，mIoU 86.3、mDice 92.5、mAcc 94.2，均超过基线；

**⚠️ 局限性**

局限性：模型参数较多（≈54M），推理速度略低；结构上下文为单向融合，可能未能充分利用双向交互；仅在单模态（bitewing、periapical）上验证，未涉及多模态或更大规模数据；缺乏临床可解释性与真实诊疗场景验证。

---

## 445. Pervasive Annotation Errors Break Text-to-SQL Benchmarks and Leaderboards

**arXiv ID:** 2601.08778 | [PDF](https://arxiv.org/pdf/2601.08778v1)

**作者:** Tengjun Jin `[一作]` (University of Illinois), Daniel Kang `[通讯]` (University of Illinois)

**通讯引用:** 1769 | [OpenAlex ID](https://openalex.org/A5072348548)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对BIRD和Spider 2.0‑Snow两个文本转SQL基准进行错误率评估，并在BIRD Dev子集上人工校正后重新评估16个开源模型，以研究注释错误对性能与排行榜的影响。

**💡 创新点**

首次系统性量化两大基准的注释错误率（>50%），并提出基于LLM的SAR‑Agent自动化错误检测工具和SAPAR注释纠错管线，展示错误率对模型性能和排名的显著扭曲。

**🔧 技术方法**

利用SAR‑Agent进行多轮SQL验证与诊断报告，结合人工专家审查；将SAR‑Agent嵌入SAPAR管线进行批量纠错；在原始与修正后的Dev子集上重新评估16个文本转SQL模型。

**📊 数据集**

BIRD Mini‑Dev、BIRD Dev（Dev子集）和Spider 2.0‑Snow。

**📈 对比分析**

在原始与纠正后的Dev子集上评估16个模型，执行准确率变化从-7%到+31%（相对），排名变动从-9到+9位；Spearman相关系数从0.85下降到0.32，表明错误率严重影响排行榜可信度。

**⚠️ 局限性**

评估仅覆盖Dev子集，未扩展到完整Dev集；依赖人工专家验证，成本高且主观性强；仅针对BIRD和Spider 2.0‑Snow，其他基准未覆盖。

---

## 446. AI as Entertainment

**arXiv ID:** 2601.08768 | [PDF](https://arxiv.org/pdf/2601.08768v1)

**作者:** Cody Kommers `[一作]` (Alan Turing Institute), Ari Holtzman `[通讯]` (University of Chicago)

**通讯引用:** 5016 | [OpenAlex ID](https://openalex.org/A5063151917)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了“厚娱乐”（thick entertainment）框架，旨在将人工智能（AI）视为娱乐创作工具，并批判目前AI评估主要关注智能优势与文化危害，缺乏对娱乐价值的积极度量方法；同时通过案例和统计数据说明AI在娱乐领域的现有应用、商业动机与社会影响。

**💡 创新点**

创新点在于：1）把AI的主要价值定位从“智能工具”转向“娱乐创作”，揭示了企业利润与娱乐模式的紧密关联；2）提出“厚娱乐”评价维度，强调意义生成、情境描述、叙事厚度等人文因素；3）指出评估体系存在的“利益偏向”（智能收益 vs 文化危害）不平衡，并呼吁构建正向文化评估框架。

**🔧 技术方法**

作者未开展实验性技术实现；主要利用现有的AI系统（如Character.AI、OpenAI Sora、Meta Vibes、Google Notebook LM）与公开报告、行业数据作为案例分析；技术层面聚焦在生成式多模态模型与对话模型的应用。

**📊 数据集**

文中引用了多项公开统计与报告数据（如英国AI安全研究所调查、美国青少年AI使用率调查、OpenAI与Disney合作数据、媒体行业营收估算等），但未使用单一统一数据集进行实验。

**📈 对比分析**

由于本文为概念与论证性论文，未开展系统性实验或对比评估；作者通过文献综述与行业案例说明当前评估方法集中于智能收益与文化危害，而缺乏对文化收益的量化比较，强调需开发新的评价指标和基准。

**⚠️ 局限性**

局限性包括：①缺乏实证数据与实验验证，论证主要基于案例与统计描述；②提出的“厚娱乐”框架尚未得到定量化与可操作化；③对商业模式与社会影响的推测未能从多方利益相关者进行深入讨论；④未考虑不同文化语境下娱乐价值的多样性与潜在负面效应。

---

## 447. Adaptive Requesting in Decentralized Edge Networks via Non-Stationary Bandits

**arXiv ID:** 2601.08760 | [PDF](https://arxiv.org/pdf/2601.08760v1)

**作者:** Yi Zhuang `[一作]` (University of Electronic Science and Technology of China), Xingran Chen `[通讯]` (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一个去中心化的协作请求问题，旨在优化边缘网络中时间敏感客户端的信息新鲜度。通过接入节点（AN）作为网关，客户端在不观察AN状态或其他客户端行为的情况下请求内容。

**💡 创新点**

提出了一种结合自适应窗口和周期监测的算法，以应对奖励动态变化的挑战，并证明该算法在理论上接近最优性能。

**🔧 技术方法**

使用了去中心化学习、非平稳多臂赌博机（MAB）技术，结合自适应窗口和监测机制来跟踪奖励分布的变化。

**📊 数据集**

使用了模拟数据集来验证理论结果，具体数据集未详细说明。

**📈 对比分析**

与传统的多臂赌博机方法相比，提出的算法在处理非平稳和相关奖励方面表现更好，能够有效跟踪奖励动态，且在模拟中显示出优越的性能。

**⚠️ 局限性**

算法的局限性在于在高度动态的环境中可能会面临挑战，尤其是在变化频繁的情况下，可能需要进一步的调整和优化。

---

## 448. 3AM: Segment Anything with Geometric Consistency in Videos

**arXiv ID:** 2601.08831 | [PDF](https://arxiv.org/pdf/2601.08831v1)

**作者:** Yang-Che Sun `[一作]` (National Yang Ming Chiao Tung University), Yu-Lun Liu `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 1145 | [OpenAlex ID](https://openalex.org/A5101674908)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出3AM模型，在SAM2框架中融入3D感知特征，实现无摄像机位姿、无深度预处理的鲁棒视频目标分割；

**💡 创新点**

核心创新在于通过Feature Merger融合MUSt3R的几何特征与SAM2的外观特征，并采用视角感知采样策略，使模型在大视角变化下保持跨视图一致性；

**🔧 技术方法**

技术实现包括SAM2、MUSt3R、跨层交叉注意力+卷积的Feature Merger、视角感知采样、记忆注意力和掩码解码器；

**📊 数据集**

训练数据使用ASE、ScanNet++、MOSE三大3D数据集，评估数据集为ScanNet++和Replica；

**📈 对比分析**

与SAM2、SAM2Long、DAM4SAM、SegMASt3R等方法比较，3AM在ScanNet++全集IoU达0.8898、Selected子集0.9061，平均提升15.9/30.4分；在3D实例分割上AP 47.3，显著优于同类在线方法；

**⚠️ 局限性**

局限在于依赖MUSt3R的预训练与计算，训练阶段仍需摄像机位姿；记忆选择策略未针对3AM专门设计，进一步提升空间有限；

---

## 449. S3-CLIP: Video Super Resolution for Person-ReID

**arXiv ID:** 2601.08807 | [PDF](https://arxiv.org/pdf/2601.08807v1)

**作者:** Tamas Endrei `[一作]`, Gyorgy Cserey `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了S3-CLIP，一种针对视频跨视角人再识别的任务驱动式视频超分辨率框架；

**💡 创新点**

创新点在于两阶段无GAN任务驱动训练、引入时序一致性损失，并将SwinIR与VSLA-CLIP无缝结合；

**🔧 技术方法**

采用SwinIR超分网络、CLIP视觉编码器、VSLA视频适配器、对比学习与三元组损失；

**📊 数据集**

在DetReIDX数据集上进行实验，该数据集包含多视角、不同分辨率的摄像头轨迹；

**📈 对比分析**

与VSLA-CLIP基线对比，Ground-to-Aerial匹配中Rank‑1提升11.24%、Rank‑10提升17.98%，在A→G场景中略有提升，整体mAP提升1.71%；

**⚠️ 局限性**

局限包括极低分辨率下超分效果有限、对运动模糊、JPEG压缩等真实降质的泛化不足、单尺度超分无法适应多尺度场景。

---

## 450. Uncovering Political Bias in Large Language Models using Parliamentary Voting Records

**arXiv ID:** 2601.08785 | [PDF](https://arxiv.org/pdf/2601.08785v1)

**作者:** Jieying Chen `[一作]` (Vrije Universiteit Amsterdam), Chendi Wang `[通讯]` (Vrije Universiteit Amsterdam)

**通讯引用:** 214 | [OpenAlex ID](https://openalex.org/A5074374964)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于议会投票记录的跨国政治偏见基准（PoliBiasNL、PoliBiasNO、PoliBiasES），并用其评估大语言模型（LLM）的政治立场与实体偏差。

**💡 创新点**

创新点在于：①使用真实议会动议和党派投票构建大规模、可纵向跟踪的政治偏见基准；②通过将LLM投票向量映射至Chapel Hill Expert Survey（CHES）二维空间，实现模型与现实党派的可解释对齐；③提出实体偏差指数（EBI）衡量党派名称对LLM投票结果的影响。

**🔧 技术方法**

技术主要包括：自然语言处理、句法抽取（仅取动议主体条款）、零样本提示（prompt），统计方法（投票一致性、相似度）、偏差可视化（CHES投影、热力图）、概率归一化评估模型置信度、部分最小二乘回归（PLS）构建投票→CHES映射。

**📊 数据集**

数据集为三国议会记录：荷兰第二议院2701条动议（15党），挪威议会10584条（9党），西班牙议会2480条（10党），均通过爬虫抓取并整理为投票向量。

**📈 对比分析**

比较方法：对多款LLM（GPT‑3.5、GPT‑4o‑mini、Llama2‑7B、Llama3‑8B、Falcon3‑7B、Mistral‑7B、DeepSeek‑7B、NorskGPT、Aguila‑7B）在每个基准上进行投票预测；使用CHES投影和投票一致性热图评估政治倾向；通过EBI热图评估实体偏差。实验显示：LLM整体呈现中左、进步倾向；对右派和极右派表现出显著负偏差；GPT系列置信度最高，开源小模型置信度较低。

**⚠️ 局限性**

局限性包括：①仅涵盖三国议会，跨文化推广受限；②基准只使用动议主体条款，可能忽略修辞影响；③模型评估基于零样本提示，未探索微调或RLHF对偏见的影响；④实体偏差分析仅限党派名称，未涉及更细粒度政治实体。

---

## 451. Asymptotic Universal Alignment: A New Alignment Framework via Test-Time Scaling

**arXiv ID:** 2601.08777 | [PDF](https://arxiv.org/pdf/2601.08777v1)

**作者:** Yang Cai `[一作]` (Yale University), Weiqiang Zheng `[通讯]` (Yale University)

**通讯引用:** 17 | [OpenAlex ID](https://openalex.org/A5078958679)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了通过在推理时生成多条候选答案并让用户挑选，从而实现对多样化且可能冲突的用户偏好的统一对齐。

**💡 创新点**

证明了单一输出模型通过测试时扩展（k 次采样）即可达到最优的对齐收敛速率 f(k)=k/(k+1)，并展示了 NLHF 等传统后训练方法在多样性不足时无法提升该速率。

**🔧 技术方法**

构造了 (k+1)-玩家对齐博弈，证明其对称纳什均衡策略满足上述最优收敛率；并提供自我对弈无后悔学习动态的理论收敛与有限时间保证。

**📊 数据集**

未使用公开数据集，理论分析基于普适的 Plackett‑Luce/排名偏好模型，证明结果对任意满足指定属性的用户偏好均成立。

**📈 对比分析**

通过理论证明与实验可视化比较，单一输出模型的测试时多样化采样可在所有 k 下实现 win‑rate 接近 k/(k+1)，优于 NLHF（仅 1/2）和 RLHF（模式坍塌导致的 12）。

**⚠️ 局限性**

局限性包括：1）需要对齐策略本身保持足够多样性，若存在强烈多数偏好会导致模式坍塌；2）对齐游戏的自我对弈收敛证明仅在对称场景下成立，实际训练中需额外技巧；3）对 ℓ>1 的对手输出仍存在收敛速率缺口，尚未给出最优上界。

---

## 452. Translating Light-Sheet Microscopy Images to Virtual H&E Using CycleGAN

**arXiv ID:** 2601.08776 | [PDF](https://arxiv.org/pdf/2601.08776v1)

**作者:** Yanhua Zhao `[一作]` `[通讯]`, Yanhua Zhao

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将多通道荧光显微镜图像转换为伪H&E染色图像，帮助病理学家以熟悉的H&E格式解读荧光数据。

**💡 创新点**

在未配对数据下使用CycleGAN实现荧光到H&E的双向映射，保留形态结构同时赋予逼真的H&E颜色；并在源域和目标域之间构建无监督的图像翻译。

**🔧 技术方法**

采用CycleGAN框架，ResNet编码解码器生成器，PatchGAN判别器，结合对抗损失、循环一致性损失和身份损失训练。

**📊 数据集**

使用两通道荧光数据（TO-PRO-3与Eosin）作为源域，以及MHIST数据库中的3,152幅H&E染色图像作为目标域。

**📈 对比分析**

通过可视化对比不同训练轮次的生成结果，评估结构保留与颜色逼真度；未能提供定量指标，且与CUT等更先进方法未进行直接性能比较。

**⚠️ 局限性**

受限于源域样本数量仅256张、无配对真值图像、缺乏生成图像质量评估与不确定性估计，模型易产生伪影并在某些结构细节上失真。

---

