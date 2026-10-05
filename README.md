# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-10-05 | 今日论文总数: 749

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. MACTS-EM: Multi-Agent Collaborative Time Series Forecasting with Emergent Memory

**arXiv ID:** 2610.02255 | [PDF](https://arxiv.org/pdf/2610.02255v1)

**作者:** Ahmad Shahi `[一作]` (Unitec Institute of Technology), Mamehgol Yousefi `[通讯]` (Unitec Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了 MACTS-EM，多代理协作框架，用于跨域时间序列预测，能动态分配专门化代理并共享记忆。

**💡 创新点**

创新点包括：多专门化代理协同、元认知层动态权重分配、突现记忆实现跨域知识迁移、多模态上下文集成以及对抗训练提升鲁棒性。

**🔧 技术方法**

技术组合包括：N‑BEATS、TCN、BlockRNN、LightGBM 等专门化代理；对比学习的突现记忆；跨模态注意力融合文本、图像、表格等；元认知层权重学习与对抗训练。

**📊 数据集**

使用了金融（S&P 500）、气候（Berkeley Earth 温度+卫星影像）、疫情（Johns Hopkins COVID‑19 + Mobility）和能源（ETT + GEFCom2014）四大数据集。

**📈 对比分析**

与 ARIMA、Prophet、N‑BEATS、TFT、TimeMixer、Time‑LLM 等基线在多域、多时段进行比较，MACTS‑EM 在绝大多数情形下提升 8–12% 预测精度、约 25% 的零样本迁移性能，鲁棒性提升约 18%。

**⚠️ 局限性**

主要局限是训练和推理成本较高（训练 +36%、推理 +42%），部分短期气候/能源场景表现略逊，长周期预测能力有限，且需要调优的超参数空间较大。

---

## 2. Unifying Privacy Accounting: Information Equivalence and Information Loss

**arXiv ID:** 2610.02414 | [PDF](https://arxiv.org/pdf/2610.02414v1)

**作者:** Buxin Su `[一作]` (University of Pennsylvania), Chendi Wang `[通讯]` (Xiamen University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了差分隐私中多种曲线式度量（双向(ε,δ)-DP、f-DP、隐私损失分布PLD、Rényi DP与零浓度DP）在同一输出分布对上的信息等价性，并量化了从完整RDP曲线压缩为zCDP时所失去的隐私信息，进一步评估了这种信息损失对噪声量与模型性能的实际影响。

**💡 创新点**

创新点主要包括：① 在有限阶RDP满足条件时证明四种曲线式度量在信息内容上完全等价；② 引入zCDP–RDP信息间隙量度并推导其在连续与离散噪声、Poisson采样等常见机制下的显式形式与局部展开；③ 分析信息间隙在独立组合中的线性累积特性；④ 通过实验验证信息间隙对噪声方差减少（约45%）与DP-SGD精度提升（≈8.7个百分点）的实用意义。

**🔧 技术方法**

技术手段包括：信息理论与概率论工具（隐私损失随机变量、累积生成函数、RDP、f-DP、PLD、zCDP的等价性证明）；数学分析方法（局部泰勒展开、极限与不等式推导）；数值实验与隐私会计实现（RDP与zCDP会计器、δ‑转换公式）。

**📊 数据集**

实验所用数据集：美国人口普查ACS 2020年县级中位数收入与教育水平表格（用于构造工作负载），以及Fashion‑MNIST图像分类数据集（用于DP‑SGD实验）。

**📈 对比分析**

比较方法：在相同(ε,δ)目标下，分别采用完整RDP曲线与其zCDP压缩版本进行隐私会计，求得所需噪声方差；随后在ACS工作负载中对比方差差异，在DP‑SGD中对比训练后的模型准确率与交叉熵。实验结果显示：Gaussian混合噪声在RDP下可比zCDP减少约45%方差；在DP‑SGD中，RDP会计器在相同(ε,δ)下提升约8.7个百分点的测试准确率，zCDP会计器表现更差。

**⚠️ 局限性**

局限性：① 主要针对单一输出分布对的理论，未直接推广至机制级RDP或自适应组合；② 结果依赖于独立成分与有限阶RDP的假设；③ 对离散/格点噪声的分析仅给出有限的数值或周期性结论；④ ρ的数值求解仍需优化，缺乏闭式表达；⑤ 实验规模有限，未覆盖更多数据集或模型。

---

## 3. A Generalized Source Integral Equation for Homogeneous Penetrable Scatterers

**arXiv ID:** 2610.02223 | [PDF](https://arxiv.org/pdf/2610.02223v1)

**作者:** Boris Diner `[一作]` (Ben Gurion University Of Negev), Yaniv Brick `[通讯]` (Ben Gurion University Of Negev)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并验证了一种针对均匀可渗透散射体的广义源积分方程（GSIE）方法；

**💡 创新点**

创新点在于将传统的单源等价原理改为使用一种外部广义源分布和内部传统源分布，从而显著降低外部块矩阵的秩，实现了对电荷量大的散射体的低秩压缩；

**🔧 技术方法**

采用多极子基底的辅助核函数构造GSIE算子，利用Galerkin Galerkin矩阵化，并结合高阶等价原理和高斯积分技术；

**📊 数据集**

通过对圆柱、尖顶弧形、椭圆和波纹圆柱等几何体的数值实验，验证了方法的准确性和压缩效果；

**📈 对比分析**

与传统PMCHWT等价原理比较，GSIE在相同误差阈值下实现了更低的矩阵秩和更快的收敛，尤其在电长大、材料损耗或背景介质速度差较大时表现突出；

**⚠️ 局限性**

局限性包括对非凸形体或极端尖角处理较困难，需要手动禁用辅助核，且对高频、低损耗介质的参数调优要求较高。

---

## 4. Finding the Move Is Not Winning the Game: XiangqiBench for Closed-Loop Evaluation of LLM Agents

**arXiv ID:** 2610.02425 | [PDF](https://arxiv.org/pdf/2610.02425v1)

**作者:** Yekun Chai `[一作]` (FloatAI), Haoyi Xiong `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

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

论文未提供具体内容，因此无法总结限制。

---

## 5. Fine-Grained Analysis of SIMD-Based Hash Table Implementations

**arXiv ID:** 2610.02385 | [PDF](https://arxiv.org/pdf/2610.02385v1)

**作者:** Cyril Nicaud `[一作]` (Univ Gustave Eiffel), Pablo Rotondo `[通讯]` (Univ Gustave Eiffel)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对 SIMD 指令下的开放寻址哈希表进行理论分析，给出了桶内元素分布和跳转（hop）次数的高概率估计，并将其推广到包含溢出位的实现。

**💡 创新点**

创新点在于：① 将 Wormald 的微分方程方法应用到哈希表动态行为；② 推导出隐式函数 λ_b(t) 的显式表达式和完整的溢出位模型；③ 给出与实验高度吻合的闭式近似公式，解释 SIMD 哈希表在不同负载因子下的性能。

**🔧 技术方法**

主要技术包括：Wormald 微分方程方法、浓度不等式、Poisson化/反Poisson化、常微分方程求解以及数值积分实现 λ_b(t) 与溢出位概率的计算。

**📊 数据集**

实验数据来源于自定义的模拟：n=2^14（约 16384）桶、b=16、d=0/8/16 等参数，单次实验已验证理论预测。

**📈 对比分析**

与传统未 SIMD 的哈希表对比，理论预测与实验结果几乎重合；在高负载因子下，SIMD 版的平均跳转次数仅为传统实现的 1/（b+1）左右，且溢出位越多，未命中搜索的额外跳转越小。

**⚠️ 局限性**

局限性包括：未处理删除操作；假设哈希函数为完全均匀、探测为随机；不适用于桶内元素重叠或非线性探测策略；需要额外的理论工作才能扩展到更一般的删除模型或更复杂的实现。

---

## 6. Lexicographic Multi-Objective On-Policy Distillation

**arXiv ID:** 2610.02359 | [PDF](https://arxiv.org/pdf/2610.02359v1)

**作者:** Doseok Jang `[一作]` (Cohere), Youran Qi `[通讯]` (Cohere)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Lexicographic Multi-Objective On-Policy Distillation (LMOPD)，通过按优先级路由并投影专家修正实现多目标后训练；

**💡 创新点**

创新点在于引入了基于缺陷的优先级路由和单向函数空间投影，保证低优先级专家不会破坏高优先级性能；

**🔧 技术方法**

使用的技术包括按门控的离散与连续奖励分级路由、对标记化策略的log-差分投影、Gram矩阵流式实现和基于逆KL的目标蒸馏；

**📊 数据集**

采用内部的数学推理语料库（AIME、HMMT等），每个问题生成50个完成样本；

**📈 对比分析**

与Rewarded Soup、GDPO、Correctness-Gated RLVR等基线比较，LMOPD在保留准确率和推理质量方面接近或超过专家，且在四专家设置下可保留90%以上的最高优先级增益；

**⚠️ 局限性**

局限性包括仅为局部启发式而非全局最优、后置专家可能饥饿、依赖特定奖励与教师质量、仅在专有模型上验证且重现实验受限。

---

## 7. Maximum Edge Open Packing on AT-Free, Chordal, and Convex Bipartite Graphs

**arXiv ID:** 2610.02275 | [PDF](https://arxiv.org/pdf/2610.02275v1)

**作者:** Gautam K. Das `[一作]` (Indian Institute of Technology Guwahati), Kamal Santra `[通讯]` (Indian Institute of Technology Guwahati)

**通讯引用:** 19 | [OpenAlex ID](https://openalex.org/A5032937760)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在AT‑free、弦图和凸二分图中寻找最大边开放包装（edge open packing）的算法问题；通过构造有向星冲突图并保持AT‑free性质，实现了O(n²+m⁴)时间算法；同时给出弦图和凸二分图的O(n⁴)时间动态规划算法；

**💡 创新点**

主要创新在于证明有向星冲突图A_G在原图G为AT‑free时仍为AT‑free，从而将最大边开放包装转化为AT‑free图的最大独立集问题；此外，对弦图利用最大团树构造的nice树分解提出新的状态机，进一步实现无界团数下的O(n⁴)算法；对于凸二分图，引入左右障碍栏（α,β）进行递归状态转移，扩展了此前的双凸（biconvex）算法；

**🔧 技术方法**

使用的技术包括：有向星冲突图构造、AT‑free图的最大独立集算法、树分解与动态规划、凸性（consecutive‑ones）性质与区间表示、前缀和表加速状态转移；

**📊 数据集**

本文未使用任何实验数据集，全部为理论算法与证明；

**📈 对比分析**

与已有工作比较：对AT‑free图实现了O(n²+m⁴)（即O(n⁸)）的时间；对弦图实现了无界团数下的O(n⁴)；对凸二分图实现了O(n⁴)（提升了早期双凸O(n⁴·2)的明确上界）；性能优于先前的指数或更高多项式算法；

**⚠️ 局限性**

局限性在于：算法对弦图和凸二分图的时间仍为O(n⁴)，尚未达到更低的多项式度；未讨论图的空间复杂度细节；对更广泛类如弦二分图的可解性仍未给出；算法在实际中需构造树分解或凸顺序，构造步骤可能影响常数因子；

---

## 8. Capability Scaling-Down Laws for LLM Compression

**arXiv ID:** 2610.02462 | [PDF](https://arxiv.org/pdf/2610.02462v1)

**作者:** Xueqi Cheng `[一作]` (Florida State University), Yushun Dong `[通讯]` (Florida State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统研究LLM压缩中能力缩减规律，构建可测量的预测模型；

**💡 创新点**

通过共享压缩响应结构大幅减少测量需求，并评估数值预测对方法选择的价值；

**🔧 技术方法**

基于剪枝、量化、蒸馏的实验，构建能力损失函数并拟合参数/非参数关系；

**📊 数据集**

使用MATH-500、MBPP、2WikiMultihopQA等评测集，测试Pythia、Gemma等多模型；

**📈 对比分析**

与全网格回归、经验中位数比较，误差≤0.02 nats/token，选择策略在QA任务上几乎与最佳方法相同；

**⚠️ 局限性**

预测仅在已训练范围内表现良好，跨模型超出范围时误差上升，对新任务或分布的泛化有限。

---

## 9. FactorSplat: Appearance-Controllable Gaussian Proxies for Medical Volume Rendering

**arXiv ID:** 2610.02382 | [PDF](https://arxiv.org/pdf/2610.02382v1)

**作者:** Zhongpai Gao `[一作]` (United Imaging Intelligence), Ziyan Wu `[通讯]` (United Imaging Intelligence)

**通讯引用:** 4424 | [OpenAlex ID](https://openalex.org/A5003798053)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于N维高斯散点的单一代理模型，能够在不重新训练的前提下，根据用户指定的区域特定传输函数（TF）实时调整医学体渲染的颜色和不透明度。

**💡 创新点**

创新点在于结合局部TF查找与低秩残差学习两种分支，既保留了作者编辑的显式效果，又通过学习补偿逼近完整的视图渲染；同时利用可视化覆盖感知裁剪保持不同TF下的结构完整性。

**🔧 技术方法**

使用N维高斯散点（N-DGS）作为几何与方向化外观的共享骨架，加入物理局部TF查找、低秩DC色彩与不透明度因子、分析可见性门控以及功能编码器来实现TF无关渲染。

**📊 数据集**

在七个临床CT和MR扫描上评估，每个扫描包含数十个作者指定的TF预设，使用真实的体渲染图像作为监督。

**📈 对比分析**

与传统的N-DGS身份基准、单独训练的专家模型以及改进的VEG基线相比，FactorSplat在多种评估（PSNR、编辑误差）上平均提升约1.1–1.5 dB，编辑误差降低，且训练时间减少约5倍，渲染速度提升至524 FPS。

**⚠️ 局限性**

局限性包括：对局部光照变化的补偿不足；持续使用的物质描述符在几何优化后可能失配；缺乏对全局光照或更复杂材质编辑的支持；需要针对更广泛的TF编辑类型进行进一步验证。

---

## 10. CLEAN: Psychometrically Consistent Incremental Cognitive Diagnosis under Concept-Space Expansion via Architectural Isolation

**arXiv ID:** 2610.02278 | [PDF](https://arxiv.org/pdf/2610.02278v1)

**作者:** Tao He `[一作]` (Shenzhen University), Fan Jiang `[通讯]` (Guangdong Polytechnic Normal University)

**通讯引用:** 16423 | [OpenAlex ID](https://openalex.org/A5012086581)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CLEAN框架，实现在知识概念空间动态扩展的增量认知诊断，保证历史诊断完全不变。

**💡 创新点**

通过严格的拓扑双分区、冻结历史诊断函数、正交列掩码和共享偏置冻结，构建结构性点对点不变性，首次实现零表示漂移的心理计量一致性。

**🔧 技术方法**

采用生成式诊断函数（GDF）、微方差初始化、正交列掩码、共享偏置冻结、LoRA低秩分支以及STB协议等技术。

**📊 数据集**

在Junyi、ASSISTments 2009–2010和Math1三个真实教育数据集上进行实验。

**📈 对比分析**

与Full‑Replay Oracle、EWC、DER++、C‑LoRA、X‑DER、ICD等常见连续学习基线对比，CLEAN在保持RD=0、旧项指标与anchor完全一致的前提下，新项表现与Oracle相近或更优。

**⚠️ 局限性**

目前仅支持一次性增量扩展，需预先划分Q矩阵，尚未验证多阶段连续增量、长期容量累积和多次概念扩展的鲁棒性与性能。

---

## 11. CUEing User Simulators: Calibrated User Embeddings for Multi-Turn Benchmarking

**arXiv ID:** 2610.02460 | [PDF](https://arxiv.org/pdf/2610.02460v1)

**作者:** Anjali Kantharuban `[一作]` (Handshake AI), Jonas Mueller `[通讯]` (Handshake AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Calibrated User Embeddings（CUE）框架，用连续用户嵌入与潜在扩散采样生成用户 persona，从而在多轮交互中更好地校准模拟器与真实用户的结果。

**💡 创新点**

创新点在于：① 学习可泛化的连续用户表示，既能编码观测会话也能采样新用户；② 通过解码生成自然语言 persona 指令，兼容任意黑盒 LLM；③ 用对比学习与内容抑制约束嵌入，提升行为多样性与结果校准；④ 结合检索增强，提升模拟器的行为覆盖。

**🔧 技术方法**

使用技术包括：ModernBERT 作为会话编码器；Qwen 3 0.6B 作为指令解码器；潜在扩散模型 + MLP 作为采样器；检索器提取训练示例；对比学习、内容抑制、跨模态注意、标签自监督等。

**📊 数据集**

数据集：24 个 DialogStudio、LMSYS-Chat-1M、WildChat-1M 约 170k 任务、角色扮演与人机对话；评测数据来自 τ^2‑Bench（航空与零售）、SimulatorArena、PRISM 等。

**📈 对比分析**

比较方法：与 UserLM、USP、PPol、RealUserSim 等基线在同一任务、代理、轮数下进行对齐。CUE 在 τ^2‑Bench 的成功率误差、宏 F1、失败归因等指标上均优于基线；在 SimulatorArena 的文档质量与交互质量误差也显著低于 RealUserSim；在 PRISM 上的行为覆盖与风格邻近度也保持竞争力，尽管无人基线在部分指标上也表现不错。

**⚠️ 局限性**

局限性：仅在两项人类研究中验证结果，未检验对多代理排名的一致性；所有数据和模型均为英语，依赖专有基础 LLM；未单独评估各子模块贡献；缺乏跨会话或前瞻性预测的验证；并非所有真实用户行为模式均已完全校准。

---

## 12. EviDent-CBCT: Evidence-Bottlenecked Report Generation from Dental CBCT under Non-Exhaustive Report Supervision

**arXiv ID:** 2610.02375 | [PDF](https://arxiv.org/pdf/2610.02375v1)

**作者:** Ruiyang Hao `[一作]` (King’s College London), Yunpeng Li `[通讯]` (King’s College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了EviDent-CBCT框架，用离散证据瓶颈实现从牙科CBCT扫描生成临床报告

**💡 创新点**

核心创新是将证据预测与语言生成分离，采用可审计的离散证据记录并通过牙科逻辑一致性投影纠正错误

**🔧 技术方法**

技术包括nnU-Net分割、三分支可靠性权重的多标签证据预测、金属敏感通道、确定性渲染、以及基于Qwen的图像无关重写模型

**📊 数据集**

使用ToothFairy3/4数据集，包含624份CBCT扫描及对应的1001份英文报告，并在50份隐藏测试集上评估

**📈 对比分析**

与五种受控直接生成接口对比，EviDent-CBCT在逻辑F1上达0.402±0.003，优于最佳直接模型0.371±0.018，在ODIN 2026自动评测中排名第二，临床Arena排名第三

**⚠️ 局限性**

局限性包括对非完全记录报告的依赖、缺乏独立影像真值、证据架构对罕见指标和细节的覆盖不足

---

## 13. Job Scheduling with Battery Recharging Constraints

**arXiv ID:** 2610.02447 | [PDF](https://arxiv.org/pdf/2610.02447v1)

**作者:** Rudransh Kumar `[一作]` (University of British Columbia), Sathish Gopalakrishnan `[通讯]` (University of British Columbia)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文建立了一种离线电池充电与作业调度模型，考虑充电时长随能量需求变化以及可能的固定设置时间，并对其32种变体的复杂度、最优/近似算法进行系统分析与实现。

**💡 创新点**

创新点包括：统一处理可部分/完整充电与设置时间的多约束模型；证明14种变体可多项式求解并给出 2-近似与 5/4-近似；对剩余24种变体给出强 NP 难性证明、指数精确算法以及近似方案；将子集求和最小化/最大化等经典问题映射到调度优化。

**🔧 技术方法**

使用了组合优化、动态规划、交换/排序最优性证明、子集求和最小化/最大化近似、以及实验性能评估等技术。

**📊 数据集**

实验数据集包括 30 个合成作业集、6 个基于 BLE/UWB 传感器电流记录的 10-job 批次，以及从 IEEE RTSS 2022 预印本得到的实例。

**📈 对比分析**

通过与最优解、SJF/SSF 等启发式对比，实验显示 A1 与 A2 的平均比值分别约为 1.02 与 1.01，最大比值分别为 1.18 与 1.06，均远低于理论上限；在完整充电策略下，A2 的增益相对最优最多仅为 18.2%。

**⚠️ 局限性**

局限性：未在真实硬件上验证模型；假设线性充电速率、固定设置时间、单设备批处理；强 NP 难变体仅给出指数解或近似，缺乏更优多项式近似；未考虑充电窗口、能量收集或多设备竞争。

---

## 14. CORE: COverage CAlibration and Evicted-Mass REdistribution for KV Cache

**arXiv ID:** 2610.02235 | [PDF](https://arxiv.org/pdf/2610.02235v1)

**作者:** Shuxin Liu `[一作]` (University of Chinese Academy of Sciences), Ou Wu `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 1690 | [OpenAlex ID](https://openalex.org/A5000753987)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对长上下文推理中的 KV 缓存压缩问题，提出 CORE 方法，通过统一的覆盖校准分配实现 KV 状态保留与补偿的协同。

**💡 创新点**

核心创新在于：① 将 evict 误差分解为丢弃注意力质量和方向误差，构造覆盖校准的分配；② 用单一分配同时驱动 Top‑B 保留和稀疏记忆写入；③ 通过离线 log‑determinant 覆盖目标与边界监督实现高效蒸馏。

**🔧 技术方法**

使用 log‑determinant 覆盖目标、聚类与原型库、轻量级索引器、边界损失、隐式记忆模块、门控机制、两阶段训练、KL 蒸馏等技术。

**📊 数据集**

在三大模型（Mistral‑7B、Llama‑3.1‑8B、Qwen3‑14B）上，使用 RULER、LongBench、AIME25、Math500 等数据集进行评估。

**📈 对比分析**

与多种现有 KV 压缩与补偿基线（H₂O、SnapKV、PyramidKV、ExpectedAttention、KeyDiff、CAPKV、IndexMem、MOMENTKV）对比，CORE 在 90% 压缩率下平均提升 2–3.8 评分点；在实时压缩与 needle 检索任务中也实现了更高准确率和更低延迟。

**⚠️ 局限性**

局限性包括：需要昂贵的离线训练与蒸馏；在低压缩比例时提升有限；对极长上下文的可扩展性尚待验证；内存写入模块仍带来额外存储与计算开销。

---

## 15. Harnessing LLMs as Agents: What Does It Cost?

**arXiv ID:** 2610.02488 | [PDF](https://arxiv.org/pdf/2610.02488v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 16. An Extensive Empirical Study on Evaluation Metrics for Combinatorial Interaction Testing

**arXiv ID:** 2610.02560 | [PDF](https://arxiv.org/pdf/2610.02560v1)

**作者:** Lisha Qin `[一作]` (Macau University of Science and Technology), Lei Ma `[通讯]` (University of Tokyo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文针对组合交互测试（CIT）中的黑盒评估指标进行了系统综述，并通过在8个开源项目中构造295,624个测试套件，评估了10多种评估指标与缺陷检测效果的相关性，进而给出了评估指标选择的实用指南。

**💡 创新点**

创新点在于：①首次对CIT的黑盒评估指标进行统一定义、分类和复杂度分析；②基于大规模实验数据量化各指标与缺陷检测率（PMS）的关联；③发现VCC（值组合覆盖率）在λ=3时最能预测缺陷检测效果，同时指出分布式指标计算成本更低；④提出单强度指标往往优于多强度组合指标，并给出具体参数配置建议。

**🔧 技术方法**

使用的技术包括：组合交互测试生成方法（RC、ARC）、静态指标计算（如VCC、VID、WMVCC、SVCC、Completeness、Discrepancy、Dispersion、Diversity、Similarity、Novelty、Divergence、IMS）、相关性度量（R²、距离相关）以及时间/空间复杂度分析。

**📊 数据集**

实验数据集为8个主流开源项目（FLEX、GREP、GZIP、MAKE、SED、BUSYBOX、DRUPAL、LINUX内核），每个项目构造4种参数比例的测试场景（共32场景），使用约1,000+模拟缺陷及真实缺陷集合，共计295,624个测试套件。

**📈 对比分析**

通过R²和距离相关两种相关性度量，比较静态指标与实际缺陷检测率的关系；结果表明VCC在λ=3时具有最高相关性，分布式指标（如Divergence、Novelty）计算成本最低；单强度指标在绝大多数情况下优于多强度指标；实验表明缺陷检测效果与指标值呈显著正相关，且不同项目/场景下最优参数略有差异。

**⚠️ 局限性**

局限性包括：仅评估了静态指标，未考虑动态指标的实时反馈；实验仅覆盖8个项目，无法完全代表所有配置复杂度；对Linux内核等大规模项目，部分指标在24小时内超时；缺陷集主要基于模拟或旧版缺陷，可能与实际生产缺陷存在差距。

---

## 17. Windfoil: Closed-Form Coverage for Real-Time and Differentiable Vector Graphics

**arXiv ID:** 2610.02468 | [PDF](https://arxiv.org/pdf/2610.02468v1)

**作者:** Matt DesLauriers `[一作]` `[通讯]` (University of the Arts London), Matt DesLauriers (University of the Arts London)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 Windfoil，一套 GPU 友好的闭式盒滤波环数计算方法，支持二次 Bézier 向量图的实时渲染与可微分优化，全部实现于 WebGPU，可在浏览器和服务器端运行。

**💡 创新点**

创新点包括：① 在每个像素上闭式求解盒滤波环数，避免多采样；② 采用单轴行分区（row‑band）加速曲线遍历；③ 同一闭式计算同时用于显示和梯度求导，保证渲染与优化使用相同的覆盖模型；④ 可通过调整滤波尺寸实现可调的抗锯齿与模糊效果。

**🔧 技术方法**

技术手段包括：Green 定理下的边界积分、曲线分段与单调性裁剪、WebGPU fragment 与 compute 着色器、行分区缓存、基于梯度的可微分渲染（VJP），以及与 Skia、Slug、DiffVG、Bézier Splatting 的对比实现。

**📊 数据集**

使用的测试数据集：Kodak 24 张图像、Färlev 512×288 与 4096×2304 的合成场景、数百个自定义 SVG 场景，以及基于 CLIP 的文本提示生成的草图。

**📈 对比分析**

比较方法：对齐渲染分辨率，计算平均绝对覆盖误差（Windfoil 1.2×10⁻⁴，Slug 1.1×10⁻³，Skia 1.6×10⁻³）；对图像拟合使用均方误差与 PSNR；对优化效率用步数与时延比。Windfoil 在大多数基准下的实时性能与 Slug 相当，在放大/缩小时更快；在图像拟合任务中，Windfoil 以 243×–1,947× 的速度提升并达到或超过 DiffVG 与 Bézier Splatting 的最终质量。

**⚠️ 局限性**

局限性：① 对自交或重叠轮廓的像素会产生误差；② 仅支持闭合二次 Bézier，无法直接渲染描边、三次曲线、渐变或纹理；③ 盒滤波使用轴对齐矩形，旋转/剪切下不完全精确；④ 预滤波后叠加与后滤波后叠加的差异会导致边缘不一致。

---

## 18. SOLO: Certified-Recall Metric Similarity Search with Scan-Only Sampled Inverted Lists

**arXiv ID:** 2610.02387 | [PDF](https://arxiv.org/pdf/2610.02387v1)

**作者:** Édgar Chávez `[一作]` `[通讯]`, Édgar Chávez

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种完全扫描的近似最近邻索引（solo），仅通过随机采样词汇表、rank‑b 分配和递归拆分大列表来实现检索，检索路径不使用任何排名启发式，查询只需路由、扫描并直接返回结果。

**💡 创新点**

创新点包括：① 回召率可由索引内部签名直接算出（recall = coverage），可一次性得到整个操作面板的回召率；② 仅通过一次扫描完成检索，避免图遍历导致的随机 I/O 与记忆体瓶颈；③ 递归规则在列表过大时将其视为子数据库，保持列表规模受控；④ 通过量化叶子（SQ8/SQ4）实现 SIMD‑友好的扫描与精确重排序；⑤ 支持一次搜索插入与精确删除，构建极简无聚类或图细化。

**🔧 技术方法**

核心技术包括：随机采样生成词汇表、rank‑b 前缀分配、递归拆分、列表按排名顺序存储、整数集交集实现覆盖率计算、量化叶子扫描（8/4‑bit 近似距离 + 精确重排序）、多线程位图去重、可直接映射的路由器、分页 I/O 优化与内存/磁盘层次结构。

**📊 数据集**

实验数据集涵盖多种规模与特征：SIFT‑1M（128‑维 L2）、GloVe‑1.2M（200‑维角度）、Deep‑10M/Deep‑100M/Deep‑1B（96‑维 L2），MS‑MARCO Web Search（768‑维内积）、LAION‑10M、PubMed23、T2I‑10M 等，覆盖从低维到高维、从单机到十亿规模。

**📈 对比分析**

与 HNSW、DiskANN、NAPP、SPANN、Faiss IVF‑PQ 等基准在相同机器、查询与真值下对比，solo 在 10^8 规模下以 1 GB RAM 以内实现 recall@10≥0.998，吞吐量超过 HNSW 的饱和点；在磁盘模式下以 1 GB 内存实现 Deep‑100M 的 0.998 recall；递归扩展至 10^9 时仅需 10 GB 以内；整体性能在扫描层次上可获得 1–2 倍以上吞吐，且回召率可精确预测。

**⚠️ 局限性**

局限性包括：① 对离群/分布外查询需要额外的覆盖率预估表；② 量化叶子在低维/高度聚集数据上可能导致误差；③ 大规模磁盘 I/O 仍受限，需高效块读取；④ 递归深度受可用内存限制，极大规模需多级递归；⑤ 过滤查询的覆盖率需额外统计；⑥ 对极高维或低内在维度的数据集尚未充分验证；⑦ 插入/删除虽然简单，但在高并发场景下仍需进一步评估。

---

## 19. Reward Inflation: A Healthy Stimulus for Reinforcement Learning

**arXiv ID:** 2610.02545 | [PDF](https://arxiv.org/pdf/2610.02545v1)

**作者:** Ganghun Lee `[一作]` (Seoul National University), Byoung-Tak Zhang `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在RL训练中引入奖励通胀机制，使奖励随时间逐渐放大，以提供持续的梯度信号。

**💡 创新点**

创新点在于将奖励缩放作为时间维度的动态调节，既产生隐式递延加权，又抑制神经元休眠，并提出Fed自适应控制。

**🔧 技术方法**

使用深度强化学习算法DQN、SAC，并结合奖励通胀、Fed、梯度范数自适应等技术。

**📊 数据集**

在Atari ALE 2600（30款）和MuJoCo 8个任务上进行实验。

**📈 对比分析**

与基线、学习率通胀、PER、RN、RTN、RWS等方法对比，奖励通胀在多数任务上平均提升约10-12%，Fed进一步提升。

**⚠️ 局限性**

限制在训练时间过长时奖励可能失稳、通胀率需手工调节且Fed涉及多超参数，且最佳通胀率因任务而异。

---

## 20. Right Order, Wrong Scale: Auditing LLM Judges for Occupational AI Measurement

**arXiv ID:** 2610.02492 | [PDF](https://arxiv.org/pdf/2610.02492v1)

**作者:** Harry Lyu `[一作]` (Massachusetts Institute of Technology), Neil Thompson `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对45,796份工人对AI生成工作任务响应的评分进行审核，评估33种LLM评判配置的排序、一致性及对职业接受率的影响，提出O*NET-BENCH审核工具。

**💡 创新点**

揭示排序一致性不保证评分水平、接受率和职业聚合的一致性，并证明一种精调协议可提升排序却降低聚合一致性；同时验证校准与预测辅助估计在此任务中的有限效益。

**🔧 技术方法**

使用大型语言模型评判（Claude、GPT、Llama、Qwen等）、列表式与逐例式提示、微调、TF‑IDF基线、校准与预测辅助估计等技术。

**📊 数据集**

基于先前收集的45,796条工人评分数据，涵盖45,796条AI响应与O*NET工作任务的配对，任务实例约五条响应。

**📈 对比分析**

通过比较排序准确率、均值偏差、接受率误差和职业分类一致性等指标；结果显示排序准确率约0.60‑0.66，但均值偏差普遍为负，接受率误差可达50个百分点；校准能消除均值偏差但预测精度低（R²≤0.085）。

**⚠️ 局限性**

局限在于样本非雇佣加权、仅一份人类评分、仅英文文本、低任务覆盖度、低可靠性、无法区分人类与评判器误差。

---

## 21. From Mathematical to Executable Certificates for Machine Unlearning

**arXiv ID:** 2610.02268 | [PDF](https://arxiv.org/pdf/2610.02268v1)

**作者:** Ziyu Zhao `[一作]` (Arizona State University), Yixuan He `[通讯]` (Arizona State University)

**通讯引用:** 18 | [OpenAlex ID](https://openalex.org/A5136837192)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了可执行发布认证（Executable Release Certification）框架，用于在机器学习模型删除操作后，确认部署时实际发布的数值模型满足预设的数学安全约束或与当前保留数据的重训练结果保持足够接近。

**💡 创新点**

创新点在于：①把数学证书与实际可执行的数值模型分离，提供两条认证路径——本地证书闭合与重训练参考认证；②针对连续删除场景，设计了增量式的岭回归实现，维护有限精度状态与真实保留数据统计之间的可验证关系；③在不改动原有模型更新规则的前提下，对四种公开无学习方法进行后置认证，显示可修正或紧化原始安全决策。

**🔧 技术方法**

主要技术包括：精确数值求解与逆矩阵缺陷证明、Sherman‑Morrison 升降更新的安全包装、误差界限传播、欧氏参数误差与输出误差的可执行上界计算，以及对FP32/FP64、BF16 等数值格式的兼容性评估。

**📊 数据集**

实验使用的数据集与模型：冻结的 Llama‑3.2‑3B‑Instruct（含 3073 维岭回归头），DINOv2 在 CIFAR‑10 上的特征，RoBERTa 在 AG‑News 上的特征，以及公开无学习方法（Newton‑update、Newton direct perturbation、ScaleGUN、CEU）的输出。

**📈 对比分析**

对比方法：原始无学习实现（未经后置认证）、全新因子重构（fresh‑factor）以及增量维护因子（maintained‑factor）。实验结果表明：①可执行认证消除 0% 的误发布；②增量实现相比全新重构在高频发布下成本降低 40%‑50%；③在四个无学习方法上，认证既能保持原有安全界限，又能在必要时进一步收紧；整体性能比单纯依赖重训练显著提升。

**⚠️ 局限性**

局限性包括：①需要针对具体模型头（如岭回归）实现增量证明，通用性有限；②对非线性或非凸头需要改用后置梯度证明，精度与复杂度可能下降；③数值精度和格式限制（如 BF16）可能导致无法满足约束，需要更高精度或不同编码；④仅验证数值模型与约束的满足，并未覆盖语义忘记或多轮适应性隐私问题。

---

## 22. MintFlow: Minimal Trajectory Intervention for Constrained Flow Matching

**arXiv ID:** 2610.02260 | [PDF](https://arxiv.org/pdf/2610.02260v1)

**作者:** Yesom Park `[一作]` (University of California), Hayden Schaeffer. Xihaier Luo `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 MintFlow，一种无需额外训练的流匹配模型约束采样框架，能够在保持原始生成分布的前提下通过最小轨迹干预满足多种约束。

**💡 创新点**

核心创新在于：①将约束采样转化为“最小轨迹干预”问题；②利用对偶方程直接获得最小范数扰动，避免高维雅可比矩阵；③自适应选择干预时刻平衡扰动大小与剩余传播时间；④可迭代细化以处理高度非线性约束。

**🔧 技术方法**

使用了流匹配模型的ODE解算、对偶（adjoint）求导、伪逆最小二乘、自动微分框架（如 PyTorch/DeepMind JAX）以及可选的迭代Newton步。

**📊 数据集**

实验涵盖图像逆问题（如去噪、超分辨）、图像编辑（颜色/纹理修改）以及物理系统建模（遵循守恒定律的模拟），使用公开的图像数据集（CelebA、ImageNet等）和物理仿真数据集。

**📈 对比分析**

与现有训练自由方法（如引导、终点投影、轨迹优化）进行对比，MintFlow 在满足约束误差更小、与原始分布的 Wasserstein 距离更低、计算效率更高（一次前向+一次对偶回传）方面均表现优异。

**⚠️ 局限性**

局限性包括：依赖一阶线性近似，非线性或不可微约束可能需要多步迭代；对极端高维或复杂动力学的计算成本仍有提升空间；对非常长的剩余传播时间仍可能导致扰动放大。

---

## 23. PowerBench: Measuring Language Model Bias in Power-shifting Requests

**arXiv ID:** 2610.02303 | [PDF](https://arxiv.org/pdf/2610.02303v1)

**作者:** Nicolas Martorell `[一作]` (Universidad de Buenos Aires), Tomás Gimenez Molina `[通讯]` (Universidad de Buenos Aires)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并公开了PowerBench评估框架，系统测量大型语言模型在处理“权力转移”请求时的拒绝倾向及其对国籍、请求者身份和语言的偏差；

**💡 创新点**

创新点在于将权力转移请求细分为自我增强、剥夺他人权力和抢夺权力三类，并构建了多维度（领域、情境、规模、先前权力地位）且跨语言、跨国籍的标准化数据集，首次量化模型在不同政治与文化背景下的偏好；

**🔧 技术方法**

技术主要包括：构造结构化请求模板、使用多语言自动翻译与人工校验、基于“You are a helpful assistant”无推理模式下的模型推理、自动化拒绝判定器（deepseek-v4-flash-0731）、以及广泛的混合效应回归与多重检验校正；

**📊 数据集**

使用了8种语言（英语、西班牙语、德语、法语、印地语、斯瓦希里语、中文、葡萄牙语）的576条权力转移请求及192条控制请求，随后扩展到18种国籍组合与AI代理情境，形成总计6,144条测试实例；

**📈 对比分析**

通过在24款模型（12款美国、12款中国开发者）上进行统计比较，发现模型在拒绝率、国籍偏差与语言偏差上差异显著；平均拒绝率从1.3%到35.2%不等，且对权力抢夺的拒绝最高，受目标规模和AI代理身份影响明显；

**⚠️ 局限性**

局限性包括：请求单轮且非真实对话、自动判定器对低资源语言的准确性可能不足、AI代理情境仅通过身份声明改变，未涵盖工具使用细节、以及模型样本并不代表全部行业模型，缺乏对现实中权力流动实际影响的直接测量。

---

## 24. Proving at Scale for Universal Algebra

**arXiv ID:** 2610.02500 | [PDF](https://arxiv.org/pdf/2610.02500v1)

**作者:** João Araújo `[一作]` (Universidade Nova de Lisboa), Bartosz Naskręcki `[通讯]` (Adam Mickiewicz University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文实现了一个自动化工作流，用语言模型代理和Lean 4证明助手，对所有阶≤6的半群进行身份基数计算并进行形式化验证，构建了完整的机器可检验目录；

**💡 创新点**

创新点在于将代理驱动的搜索与人类决策相结合，利用共享的“family”证明模板以及自动化重构机制，实现了对数千个半群的批量处理和最终的全库核对；

**🔧 技术方法**

使用了语言模型（Codex/Claude）生成候选基、搜索反例、编写Lean 4证明；随后利用Lean 4内核和Vampire定理证明器完成证明验证、基础最小化与包含关系推导；

**📊 数据集**

数据集来源于SmallSemigroups，包含所有阶≤6的非同构半群（共15973个），以及已知的四个非有限基半群；

**📈 对比分析**

通过Vampire对已认证基进行最小化、互相独立性检验和包含关系验证，得到505个不同的半群类；实验显示Vampire能证明13321条包含关系，其中4条仍未决定，289条问题被标记为最难，表明系统在大规模包含推理上表现良好；

**⚠️ 局限性**

局限在于对更大阶半群（如阶7）仍面临尾部难题，证明生成成本呈长尾分布，现有方法对数十万半群的规模仍难以完全自动化，且部分证明仍需人工干预。

---

## 25. Co-design Gym: A Unified Benchmark for Embodiment-Policy Co-optimization

**arXiv ID:** 2610.02366 | [PDF](https://arxiv.org/pdf/2610.02366v1)

**作者:** Aviraj Newatia `[一作]` (University of Cambridge), Rika Antonova `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Co-Design Gym，一套统一的、可扩展的 benchmark，专门用于同时优化机器人的体制（embodiment）与控制策略（policy），覆盖 20 个任务族、85 个配置，支持 GPU 并行仿真。

**💡 创新点**

创新点在于：① 将 Gymnasium API 扩展为包含设计空间的 co‑design 结构；② 将多种经典、工业和游戏任务转化为 co‑design 形式；③ 构建多样化的软体与硬件仿真环境，首次在统一平台上对比 co‑design 算法；④ 公开 Benchmark 数据与评测脚本，推动社区可重复、可比较的研究。

**🔧 技术方法**

使用的技术包括：基于 MuJoCo / MuJoCo‑Warp 的刚体/软体动力学仿真；GPU 并行环境实例化；四种代表性 co‑design 方法（CMA‑ES、FastTD3、PPO+NGOpt、LOKI）以及 LLM 外部设计器；强化学习框架（PPO、TD3、FastTD3）与进化算法；Transformer‑VAE 生成器与聚类搜索。

**📊 数据集**

没有传统意义上的数据集，评测基于自定义的 18 个预设环境与 85 个细化配置，每个环境提供多种任务、奖励与物理约束；对比数据来源为实验中收集的回报分布、设计空间覆盖率与拒绝率。

**📈 对比分析**

实验结果表明：没有任何方法在所有任务上统治；CMA‑ES 在大部分连续控制任务（如 Locomotion）表现最好，FastTD3 在球捕捉、赛道与微电网等任务领先，PPO+NGOpt 在仓库、网络与 Pokémon 等离散任务占优，LOKI 在部分任务可与 FastTD3 对齐。整体来看，所有方法均距离理论最优回报还有明显差距，许多预设仍处于“未解决”状态。

**⚠️ 局限性**

局限性包括：① 设计空间未覆盖所有可能自由度；② 仅对 18 个预设进行了基准实验，未覆盖全部 85 个配置；③ 固定的 4‑小时预算更偏向实现效率，可能掩盖了算法本身的潜力；④ 依赖现有仿真器，未包含流体、风等复杂物理；⑤ LLM 作为外部设计器表现不佳，提示设计者的任务约束与提示设计尚未成熟。

---

## 26. SoK: Stablecoins in the Quantum Era

**arXiv ID:** 2610.02435 | [PDF](https://arxiv.org/pdf/2610.02435v1)

**作者:** Panagiotis Chatzigiannis `[一作]` (VISA Research), Duc V. Le `[通讯]` (Circle Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对后量子时代稳定币的安全与迁移进行了系统化梳理与评估，提出了安全威胁模型、迁移三维分类以及监管与责任框架；

**💡 创新点**

创新点在于将稳定币特定的权限-责任缺口纳入迁移考量，并构建了控制面、迁移层、加密策略三维税onomies，揭示了迁移缺口与监管约束的交互；

**🔧 技术方法**

采用了量子计算对传统公钥密码学（ECDSA、BLS、ECDH等）的影响分析，结合已标准化的后量子签名（ML‑DSA、SLH‑DSA）与加密封装（ML‑KEM、HQC）以及零知识证明、承诺与多重签名等加密组件；

**📊 数据集**

使用了公开文献、标准草案、区块链改进提案（BIP、EIP、ZIP、CCTP等）以及各大稳定币发行方（USDC、DAI、USDT等）的技术文档与路线图作为“数据集”；

**📈 对比分析**

对迁移方案的比较主要通过理论成本（签名尺寸、校验费用）、功能覆盖度（控制面、权限、监管合规）以及层级依赖性来评估；实验性测评表明后量子签名在高频桥接/合约操作中会显著提升链上算力与带宽负载；

**⚠️ 局限性**

局限性包括：缺乏完整的端到端性能基准；多层交互的实际迁移成本与时间窗口尚未量化；对监管合规的具体实现细节与跨组织责任分配仍需进一步研究；

---

## 27. Drive vs. Decay: On the Training Dynamics of Joint-Embedding Predictive Architectures

**arXiv ID:** 2610.02344 | [PDF](https://arxiv.org/pdf/2610.02344v1)

**作者:** José Lucas De Melo Costa `[一作]` (Université Paris-Saclay), Bich-Liên Doan `[通讯]` (Université Paris-Saclay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过构建驱动-衰减（drive–decay）理论，统一并解释了 Joint‑Embedding Predictive Architectures (JEPAs) 在早期训练阶段出现的表征坍塌现象，并在此基础上提出了新的残差预测器 ResidualPred。

**💡 创新点**

创新点在于：①提出了以谱比 μ = γ/σ 为核心的早期稳定性判据；②将预测器缩放、遮掩比例、EMA 等经验技巧映射为对 μ 的“剂量”控制；③设计了对注意力自注意力块做对角偏置的残差预测器，以在初始化时保持对称性，显著提升有效秩和下游线性探测精度。

**🔧 技术方法**

核心技术包括：线性化 JEPA 梯度流的雅可比分析、谱分离假设下的模式级稳定性判据、对 μ 的解析计算、以及基于残差身份映射的 Transformer 预测器实现。

**📊 数据集**

使用的实验数据集包括三类表格数据集 ALOI、Helena、Jannis 以及图像数据集 CIFAR‑10、CIFAR‑100、STL‑10、ImageNet‑1k，并在 800+ 组合的 Tabular‑JEPA 上进行大规模验证。

**📈 对比分析**

与传统 JEPAs（含 EMA、stop‑gradient、SIGReg 等）相比，ResidualPred 在所有实验中均实现了更高的有效秩和线性探测准确率；在表格任务中提升约 3–5% 线性准确率，在图像任务中提升 2–4% 线性准确率，且在 128/224 像素的 ImageNet 试验中表现尤为显著。

**⚠️ 局限性**

主要局限在于理论推导基于线性模型与谱分离假设，实测验证仍受限于中等规模数据；对高分辨率 ImageNet 的正式训练结果尚未充分验证，且 μ=1 的判据仅为局部初始时刻的指标，未能完全描述非线性阶段的全局收敛行为。

---

## 28. Counterexample Generation via Per-Theorem Symbolic Verifiers: When Imitation Hurts and Reinforcement Repairs

**arXiv ID:** 2610.02444 | [PDF](https://arxiv.org/pdf/2610.02444v1)

**作者:** Omar Farouk Zouak `[一作]` (National School of Artificial Intelligence), Samia Nefti-Meziani `[通讯]` (University of Birmingham)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 SymCE 数据集并使用可执行 Python 验证器训练 Qwen3‑4B 进行反例生成，发现并修复了 SFT 的“伪相似陷阱”，实现了在定量与跨域推理任务上的显著提升。

**💡 创新点**

① 通过可执行 verifier 构造大规模、可验证的反例数据集；② 提出了对抗式奖励（RLVR）与稀疏结果奖励的组合，解决了传统 SFT 的校准失效；③ 发现并解析了“伪相似陷阱”这一新颖失调现象。

**🔧 技术方法**

使用监督微调（SFT）+ Group Relative Policy Optimization（GRPO）强化学习，结合可执行 verifier 作为奖励函数；还利用 witness‑schema 约束输出形状，并对比稠密与稀疏奖励形式。

**📊 数据集**

SymCE：4,707 条本科层次代数与实分析伪定理，每条配有 Python 验证器；另外使用 GSM8K、MATH‑500、MMLU‑college‑math 等标准数学 benchmark 进行迁移评估。

**📈 对比分析**

与五个 7B 开源数学专家模型以及六个前沿商业 API 进行基准对比；Qwen3‑4B 在 SymCE 上从 0.30 提升至 0.49（平均），超过所有 7B 开源模型，且在 GSM8K、MATH‑500、MMLU‑college‑math 上的 Pass@1 分别提升 5–35% 以上，表现与商业 API 相当。

**⚠️ 局限性**

受限于 4B 参数规模、GRPO 算法、Python‑SymPy/Z3 的可验证范围、教师模型单一、验证器审核样本有限，未能验证更大模型、不同 RL 算法或更广域逻辑的适用性；同时只对单一假设删减方式进行实验，未涵盖更复杂的假设变形。

---

## 29. MEA: A Reward-Driven Multi-Agent System for Faithful Model Explanations

**arXiv ID:** 2610.02480 | [PDF](https://arxiv.org/pdf/2610.02480v1)

**作者:** Yuyang Cheng `[一作]` (University of Virginia), Chirag Agarwal `[通讯]` (University of Virginia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个由 Proposer 和 Actor 两个智能体组成的多模态解释框架，能够根据用户自然语言问题自动选择解释工具、执行并合成基于模型行为的自然语言解释，并构建了一个包含 16k 条样本、10 类问题类型、三种模态（表格、文本、视觉）的基准数据集。

**💡 创新点**

核心创新包括：①将解释任务端到端地用强化学习（GRPO）优化，使解释与模型行为的 faithfulness 成为直接奖励；②引入模态自适应奖励和工具计数惩罚，防止奖励上手和工具滥用；③设计了一套通用的问题分类和对应的扰动式 faithfulness 指标，统一评估多模态、跨任务的解释质量。

**🔧 技术方法**

主要技术：多智能体（Proposer–Actor）框架、GRPO 强化学习、模态自适应奖励、工具调用与归纳、LLM 基座（Qwen3.6‑35B）+ LoRA 微调、扰动式 faithfulness 评估、工具库（LIME、SHAP、Grad‑CAM、计数器、对比推理等）。

**📊 数据集**

使用的数据集包括：Adult Census、Breast Cancer、SNLI、IMDb、CUB‑200‑2011、STL‑10（训练/测试），以及在 OOD 评估中引入的 Yelp Review、German Credit、CIFAR‑10，分别配合多种模型（两层网络、TabNet、CNN、ResNet‑50、DenseNet‑201 等）。

**📈 对比分析**

与基线对比：在所有三种模态下，系统在 10 种解释问题上均优于传统后置解释器（LIME、SHAP、Grad‑CAM）以及前沿闭源模型和其他对话式/代理式框架（CoT、ReAct、ToT）。平均 faithfulness 提升分别为 +28%（表格）、+21%（文本）和 +34%（视觉）。实验还显示系统对问题表述的改写鲁棒，且在 OOD 数据集和模型上保持显著性能提升。

**⚠️ 局限性**

局限性：仅针对分类任务；工具库固定，未覆盖所有可能的解释技术；扰动式 faithfulness 指标对分布漂移敏感；未来需要扩展到其他任务（回归、生成）、更丰富的工具集以及多样的 faithfulness 信号。

---

## 30. MeshQuery: Agentic Seam Planning for UV Parametrization

**arXiv ID:** 2610.02507 | [PDF](https://arxiv.org/pdf/2610.02507v1)

**作者:** Marco Schouten `[一作]` (Adobe Research), Tamy Boubekeur `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于代理(agent)的UV展开框架，使LLM通过查询式网格表示和DSL规划自动化选择UV缝隙，支持大规模四边网格的高质量展开；

**💡 创新点**

创新点在于：①构造可查询的网格数据库让LLM在不序列化网格的前提下获取几何、拓扑、语义信息；②设计面向UV缝隙的DSL，支持聚合边选择与主动感知；③利用闭环反馈（UV质量评估）让代理反复细化缝隙计划；

**🔧 技术方法**

技术包括：大语言模型（Claude Opus 5）、关系数据库查询、域专用语言（DSL）与工具调用、UV参数化与打包算子、自动化评价指标反馈；

**📊 数据集**

使用的公开数据集为Adobe Substance 3D Assets（50件）和Toys4K（50件）四边网格；

**📈 对比分析**

与FlattenAnything、OptCuts、xatlas、Blender Smart UV、PartUV等基线对比；在数量指标上，平均减少2.9×至4.3×UV图、1.6×至1.7×UV缝隙长度，兼具低失真；在人类评测中，专业艺术家偏好率达80.9%；

**⚠️ 局限性**

局限性包括：仅适用于四边网格、依赖PartField语义分割与标注成本、迭代闭环耗时较高、对极端大模型的参数化和实时性仍有限。

---

## 31. Why Does Adaptive Batching Help LLM Pretraining? A Perspective from Unbounded Variance

**arXiv ID:** 2610.02355 | [PDF](https://arxiv.org/pdf/2610.02355v1)

**作者:** Arda Fazla `[一作]` (Purdue University), Abolfazl Hashemi `[通讯]` (Purdue University)

**通讯引用:** 563 | [OpenAlex ID](https://openalex.org/A5036900440)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在大型语言模型预训练过程中为何逐步增大批量大小（batch size）有效，并提出了BG‑a噪声模型以及基于该模型的自适应批量调度器，给出了对应的理论证明与实验验证。

**💡 创新点**

创新点在于：①引入BG‑a噪声模型，介于传统均匀方差和BG‑0噪声之间，能够更精确描述噪声随与初始化距离的增长；②证明该模型对随机梯度优化的oracle复杂度产生的影响，并给出匹配的动态批量上界；③将理论结果转化为实际可用的自适应批量调度器，并在LLM预训练中实现性能提升。

**🔧 技术方法**

使用了随机优化理论（L‑smooth性、无偏噪声、方差上界）、动态批量调度、SGD/SGDM/Adam优化器、线性/平方根学习率缩放以及大规模实验评估。

**📊 数据集**

实验使用C4数据集对OLMo2-100M、OLMo2-600M、OLMo2-1B模型进行预训练；附录中还对ResNet50和图像分类任务做了验证。

**📈 对比分析**

将自适应批量调度器与固定小批量（32/64）和固定大批量（512/2048）在相同token预算下进行对比。结果显示，BG‑a调度器在验证损失上均优于两种固定策略，且仅使用小于10%的小批量迭代次数，训练时间减少约20%–40%。

**⚠️ 局限性**

局限性包括：需要先行估计噪声参数B、G、a，理论上限仅针对a≤2；对分布式训练的细节实现有限；目前仅在在线一次性训练场景验证，可能不适用于多周期离线训练或其它模型结构。

---

## 32. RUL-Aware RRT*: Degradation-Balanced Motion Planning for Robotic Manipulators

**arXiv ID:** 2610.02469 | [PDF](https://arxiv.org/pdf/2610.02469v1)

**作者:** Haibo Li `[一作]` (CentraleSupélec, Université Paris-Saclay), Xu Li `[通讯]` (Beijing Jiaotong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种将关节剩余使用寿命（RUL）信息嵌入到RRT*运动规划中的闭环算法RUL‑aware RRT*，并通过在线RUL更新和自适应权重机制实现关节使用的动态平衡。

**💡 创新点**

创新点在于：①将实时健康反馈直接融入路径代价函数，实现健康感知的运动规划；②设计在线RUL更新与RUL驱动的权重调节双机制，使规划过程自适应关节健康变化；③通过三种典型退化场景验证该方法在提升系统可靠性、延长故障时间和保持关节使用均衡方面的显著效果。

**🔧 技术方法**

技术包括采样式运动规划算法RRT*的改进、RUL更新模型（基于累计关节位移的退化公式）、自适应权重映射（基于RUL方差的指数映射）、以及ROS+MoveIt+OMPL的仿真实现。

**📊 数据集**

使用的“数据集”是基于UR5 6自由度机械臂的仿真环境；关节RUL通过预设初始值（1000、500等）和累计位移更新来模拟；实验共执行80次pick‑place任务或直至首个关节失效。

**📈 对比分析**

与传统RRT*基准对比：在全健康、异构退化和局部退化三种场景下，RUL‑aware RRT*显著降低RUL方差、提高最小剩余寿命（RUL_min）和首次失效任务数，恢复率与生命周期增益分别提升数%至50%之间，证明了在不同退化形状参数p下仍能保持竞争优势。

**⚠️ 局限性**

局限性包括：①实验仅在仿真环境进行，未验证真实传感器噪声和执行误差的影响；②退化模型仅基于累计位移，未涵盖温度、负载等多维健康因素；③仅针对单机器人设置，缺乏多机器人协作与任务分配的评估；④算法参数（α、λ、p）对性能影响较大，需进一步自动化调优。

---

## 33. LiteEMG-FM: An Efficient and Deployable Foundation Model for Robust EMG Sensing

**arXiv ID:** 2610.02497 | [PDF](https://arxiv.org/pdf/2610.02497v1)

**作者:** Tianhao Wu `[一作]` (University of Georgia), Jian Liu `[通讯]` (University of Georgia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一个8M参数的混合CNN‑Transformer EMG基础模型LiteEMG‑FM，支持跨用户、跨场景的无校准语义迁移，并实现了层次化唤醒与三种部署策略，实现在ESP32‑S3微控制器上低功耗实时推理。

**💡 创新点**

创新点在于将EMG特有的时频谱表示与局部卷积先验相结合的自监督掩码重构预训练，并提供面向可部署的多级唤醒和分布式推理框架，显著缩小模型规模同时提升泛化性能。

**🔧 技术方法**

采用自监督掩码自动编码器、卷积前端+Transformer编码器、时频谱预处理、量化压缩以及轻量级1D‑CNN唤醒门控。

**📊 数据集**

使用统一的16个公开EMG数据集（包括NinaPro、GRABMyo、CapgMyo、Locomotion等）进行预训练，后续在NinaPro DB2/7/8、SIAT‑LLMD、MyPredict等任务上评估。

**📈 对比分析**

通过线性探针和全微调与MOMENT、Brant、PatchTST等基线对比，在所有5个下游任务中，LiteEMG‑FM以约8M参数赢得最高准确率，零校准跨用户精度约40%，低标记下也超过传统模型20%以上。

**⚠️ 局限性**

局限在于模型仍需在更复杂的多传感器高频EMG、不同皮肤阻抗条件下进一步验证；分布式推理在网络延迟敏感场景下可能受限，且可解释性和鲁棒性对极端噪声仍待提升。

---

## 34. A Generative Model of Complex Networks Using Graphons and Neural Inverse Operators

**arXiv ID:** 2610.02439 | [PDF](https://arxiv.org/pdf/2610.02439v1)

**作者:** Wooseong Choi `[一作]` (University of Southern California), Paul Bogdan `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了多分形阶梯图层图形（multifractal step graphon）并设计了一种神经逆算子（MWNOT）在函数空间中进行参数恢复，实现了对未见图大小的零样本图生成以及单图网络的快速拟合。

**💡 创新点**

创新点包括：①将机制化图模型与深度生成模型统一到函数空间；②通过Kronecker递归构造极小参数化的多分形阶梯图形，能够在任意尺度捕捉层次结构；③开发了可跨尺度、可推广到任意图大小的神经逆算子；④在零样本生成和单图拟合上取得与或超过现有方法的性能，且模型参数显著更少。

**🔧 技术方法**

使用的技术包括：多分形阶梯图层图形（基于步图形与Kronecker递归），多波let变换与跨注意力的Multiwavelet Neural Operator Transformer（MWNOT），Aldous表示下的图采样，MMD、Wasserstein 和谱指标等评估度量。

**📊 数据集**

训练数据仅为合成的多分形阶梯图形样本；评估数据涵盖 12 个不同领域的真实图网络（如 Facebook、BIO、LGGM‑X 等）、单观测网络（C. elegans、Gnutella、Drosophila、Power Grid、MN Roads 等）以及 20 受试者的 alpha‑band EEG 连接网络。

**📈 对比分析**

与基线 LGGM‑X、LGGM‑X 预训练以及 WMGM 直接拟合方法对比；在零样本生成中，在 4/12 领域取得最佳平均 MMD，并且模型参数约为 22 倍少；在单图拟合中，速度比 WMGM 快 9–550 倍，误差与 WMGM 相当或更好；EEG 研究中，推断的 assortativity 与 multifractal 宽度在重度镇静时显著变化，效果显著高于传统网络统计。

**⚠️ 局限性**

局限性包括：假设连接规则在所有尺度保持一致，无法处理尺度上规则不规则变化的网络；逆算子不是置换不变的，需要预先对节点进行度排序；对谱指标的拟合仍有改进空间；对非多分形结构的网络表现不佳。

---

## 35. LLM-Based Semantic Modeling and Cooperative Evolutionary Fuzzing for Traffic Violation Scenario Generation

**arXiv ID:** 2610.02222 | [PDF](https://arxiv.org/pdf/2610.02222v1)

**作者:** Yangyang Liu `[一作]` (Hohai University), Pengcheng Zhang `[通讯]` (Hohai University)

**通讯引用:** 4034 | [OpenAlex ID](https://openalex.org/A5035407903)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种名为SLaFE的框架，利用大型语言模型将交通法规抽象成结构化的场景约束，并通过协同进化的模糊测试算法在仿真环境中自动生成并触发交通违规场景；

**💡 创新点**

创新点在于：①将交通法规通过两阶段LLM推理转化为可直接用于测试的违规特征向量；②构建基于区域语义的统一规则模型，消除法律歧义；③提出跨法规的相似度协同进化策略，显著提升测试效率和覆盖率；

**🔧 技术方法**

核心技术包括：大型语言模型（GPT‑4o‑mini）进行语义抽取和特征向量生成；自定义的两阶段Prompt工程；基于特征向量的协同进化模糊算法（区域引导变异、相似度共享）；仿真平台LGSVL与Apollo 7.0交互；

**📊 数据集**

使用10条真实交通法规（来自中国和美国）作为实验数据，结合LGSVL提供的旧金山高清地图；

**📈 对比分析**

与六个基线（LawBreaker、ABLE、VioHawk、AV‑Fuzzer、DriveFuzz、AutoFuzz）在Apollo 7.0平台上进行比较；SLaFE在10条违规类型上均能触发（100%），平均触发时间仅为5.1分钟，显著快于VioHawk的9.0分钟，整体效率提升约54%；

**⚠️ 局限性**

局限性包括：依赖LLM的推理稳定性；实验仅覆盖10条法规，未验证更广泛法律场景的适用性；仅在Apollo+LGSVL平台验证，缺乏跨平台通用性；

---

## 36. "I just assumed that it would translate": examining MT risk awareness among healthcare staff with abbreviations as a use case

**arXiv ID:** 2610.02496 | [PDF](https://arxiv.org/pdf/2610.02496v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 37. Awomo-SimDataEngine: Agentic Simulation-ReadyWorld Generation

**arXiv ID:** 2610.02274 | [PDF](https://arxiv.org/pdf/2610.02274v1)

**作者:** Awomo-PhysicalRSI Team `[一作]`, Zijian Ma `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个名为 Awomo‑SimDataEngine 的端到端系统，集成了交互式资产生成、文本和单图像驱动的场景构建、图形化执行与修复以及自动化演示合成；

**💡 创新点**

创新点包括：① ISArt 的结构条件部件生成与仿真反馈绑定；② Unravel 的无监督单视场景重建与点云融合；③ SimForge 的文本驱动多房间场景规划；④ 图形化 Harness 对模块状态、验证与有限修复的统一管理；⑤ PolicyForge 将验证过的世界与任务绑定，自动生成多装备演示数据；

**🔧 技术方法**

使用技术包括：视觉语言模型（VLM）+结构规划、DiT 结构条件生成、ICP 与半距离 Chamfer 进行注册、物理仿真（Isaac Sim 与 MuJoCo）进行碰撞/支持验证、RL/Diffusion Transformer（WAM）等；

**📊 数据集**

主要使用的数据集有：基于 Hunyuan3D、TRELLIS.2、SAM 3 等的资产库；Unravel/SimForge 生成的场景（约 160 个物体的单图像、179 条室内提示与 31 条多房间提示）；LIBERO‑Plus 与 LIBERO‑Pro 作为基准评测；

**📈 对比分析**

比较方法为在同一 13K 训练步数预算内选取整体成功率最高的检查点，使用 1,003 条评测集；实验显示 Co‑training 数据将整体成功率从 77.17% 提升至 89.43%（+12.26pp），目标泛化提升 31.66pp，空间泛化提升 6.25pp；

**⚠️ 局限性**

局限性包括：单视角重建与结构先验误差导致几何不准；图形化 Harness 的修复预算有限且未量化修复效率；演示覆盖受教师脚本/运动规划限制；缺乏对真实机器人转移的验证；实验未做模块/增量化 ablation，难以评估各子系统贡献；

---

## 38. Filter-Aware Fine-Tuning for Safe Humanoid Whole-Body Tracking

**arXiv ID:** 2610.02341 | [PDF](https://arxiv.org/pdf/2610.02341v1)

**作者:** Pranit Mohnot `[一作]`, Marco Pavone `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于安全过滤器（CBF）的“CoFiT”方法，对已有的全身跟踪策略进行微调，使其能够与运行时安全过滤器协同工作，显著降低约束违规时间并减小过滤器需要的修正量。

**💡 创新点**

创新点包括：① 系统性分析了跟踪器与安全过滤器之间的动力学、目标和信息失配问题；② 设计了紧凑的约束信息观察（包含约束状态、动作空间约束正则化等）；③ 在微调过程中加入了时间反馈、约束历史与过滤器修正相关的奖励（修正大小、原始条件、求解器松弛度惩罚）；④ 通过实验证明该方法在多种轨迹和障碍物场景下均能大幅提升安全性与协同性能。

**🔧 技术方法**

使用技术主要包括：强化学习微调（Actor‑Critic），控制障碍函数（CBF）与全身控制的二阶QP过滤器，动作空间约束映射与约束观察编码，时间序列历史处理（Top‑k/注意力），以及多种奖励惩罚函数。

**📊 数据集**

数据集与场景：
- TWIST2 运动库（用于训练与测试）；
- SONIC 运动库（用于对比实验）；
- 生成的球形障碍场景（约束交互的基础实验）；
- 通过 Kimodo 与 RoboCasa 生成的多种几何障碍（manipulation、locomanipulation 等）；
- Unitree G1 实物实验，利用运动捕捉系统收集的球与人体位姿数据。

**📈 对比分析**

比较方法：对比继续无过滤器训练、过滤器仅训练、只使用约束观察、只使用惩罚、完整 CoFiT。评估指标包括违规时间、过滤器修正 RMS、命令总变动、残余加速度、关节位置误差。结果显示：
- 在 TWIST2 上，CoFiT 将违规时间降低 91%，过滤器修正 RMS 降低 59%；
- 在 SONIC 上，违规时间降低 21%，修正 RMS 降低 29%；
- 在 Unitree G1 真实跑动中，违规时间降低 83%，且所有试验均无操作员停止，滤波修正量显著减小，运动更平滑。

**⚠️ 局限性**

局限性：
- 仍需先验的安全过滤器，无法完全替代全局规划；
- 对于动态障碍物或需要重新规划路径的情况，方法仅能做局部调整；
- 训练过程中对 CBF 约束的假设（相对阶数为 2）有限制；
- 微调后可能牺牲一部分跟踪精度，尤其在没有全局位姿反馈的系统中更为明显；
- 未提供严格的安全保证，只是经验性减少违规。

---

## 39. When Terminal-Agent Training Stalls: Demystifying Data Generation and Verification Challenge

**arXiv ID:** 2610.02405 | [PDF](https://arxiv.org/pdf/2610.02405v1)

**作者:** Xi Qin `[一作]` (SAP Lab), Yaad Oren `[通讯]` (SAP Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究设计并实现了一套元代理流水线，用于自动生成、验证并训练终端任务代理，生成了18,185个结构化、可执行的终端任务。

**💡 创新点**

创新点在于：①重构生成提示以消除“Worksheet Voice”和“Grader Leakage”，并将难度从命令数转为认知层次；②提出“可解性带”概念，阐明任务难度对RL训练的梯度信号影响；③引入结构有效性、验证器审核和基础设施错误计量为评估标准。

**🔧 技术方法**

技术上采用了Harbor框架、Docker容器、Claude Opus/4.6/4.7等LLM进行任务生成，SkyRL+GRPO进行RL训练，利用Dockerfile构建、pytest验证器和任务元数据实现完整流水线。

**📊 数据集**

使用由Claude Opus在8192-token窗口生成的三版本数据集（V1/V2/V3），共计18,185条任务，并基于Harbor标准的任务目录格式。

**📈 对比分析**

在RL实验中，Qwen2.5-3B在V1训练仅实现7.8% pass@1；Qwen3.5-9B在V1+V2训练平均pass@2达到81.3%；加入V3硬任务后平均pass@2降至20.6%，表明难度调节对模型学习至关重要。

**⚠️ 局限性**

局限性包括：生成任务的稀疏奖励导致梯度信号不足；Docker构建、网络池耗尽等基础设施错误对评估产生干扰；异步返回样本的顺序偏差影响pass@k估计；实验规模受成本限制，未验证完整训练过程的提升。

---

## 40. Choosing Before Acting: Comparative Value Estimation for Long-Horizon Tool-Use Agents

**arXiv ID:** 2610.02330 | [PDF](https://arxiv.org/pdf/2610.02330v1)

**作者:** Yu Li `[一作]` (Southeast University), Lei Feng `[通讯]` (Southeast University)

**通讯引用:** 27452 | [OpenAlex ID](https://openalex.org/A5060924118)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种用于长周期工具调用的对比推理框架CITA，训练对比推理模型CIM来预测每个可能的下一步工具调用的长期价值，并用其指导策略训练和推理。

**💡 创新点**

创新点在于：①利用对比推理模型CIM，通过配对信号学习长期价值估计；②构造三源对比训练数据（真实轨迹、贝叶斯工具图模拟器、LLM对比判定）；③将CIM既作为步骤级奖励，又作为策略训练的指导，从而实现对长周期决策的前瞻性估计。

**🔧 技术方法**

技术：对比推理模型（基于LLM编码器+值头+置信度头）；贝叶斯工具图模拟器生成结构化对比样本；GRPO强化学习与对比奖励结合；LLM对比判定用于生成语义对比信号；分阶段训练与多源数据混合。

**📊 数据集**

数据集：Toolathlon、TOUCAN、TRAJECT‑Bench三大长周期工具使用基准，以及对应的真实轨迹和工具图。

**📈 对比分析**

比较方法：与提示/搜索、RL、步骤级奖励、预测规划等多类基线对比，CITA在Tool F1和Task Success Rate两项指标上均取得领先，平均提升约7.6 F1点、9.9 TSR点。

**⚠️ 局限性**

局限：依赖已有工具图与日志轨迹，对动态变化的工具生态适应性有限；对大规模工具集合的对比样本生成仍需改进；缺乏对更复杂实时执行环境的评估。

---

## 41. Mitigating Private Data Leakage in LLMs with Whiteout

**arXiv ID:** 2610.02418 | [PDF](https://arxiv.org/pdf/2610.02418v1)

**作者:** Anna Yoo Jeong Ha `[一作]` (University of Chicago), Ben Y. Zhao `[通讯]` (University of Chicago)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于精确覆盖的LLM隐私泄露防护工具Whiteout。

**💡 创新点**

创新点在于用少量伪装样本精准覆盖并重连用户与PSI的关联，而非全量遗忘或拒绝。

**🔧 技术方法**

采用标准监督微调（SFT/LoRA）结合自动生成的在分布内伪装数据进行模型更新。

**📊 数据集**

评估使用7个近期公开与商业LLM（如Llama 3.2、Gemma 3、Phi-3-mini、DeepSeek、Phi-4、GPT-OSS、GPT-4o-mini）以及包含21名真实公众人物和29名合成个体的PSI数据集。

**📈 对比分析**

与梯度上升、负向偏好优化和拒绝策略对比，Whiteout在100% PER的同时保持模型效用和安全性；对多种黑盒/白盒攻击（重学习、量化、GCG等）均能保持0%成功率。

**⚠️ 局限性**

局限包括只能处理三种PSI类型、需外部验证请求、可能被滥用插入虚假信息，以及在更大规模或特殊模型上验证有限。

---

## 42. Slow-Fast Multi-Teacher On-Policy Distillation for Capability Preservation

**arXiv ID:** 2610.02324 | [PDF](https://arxiv.org/pdf/2610.02324v1)

**作者:** Xiaofei Yin `[一作]` (Ant Group), Huijia Zhu `[通讯]` (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Slow‑Fast Multi‑Teacher On‑Policy Distillation（SF‑MOPD）框架，结合慢速EMA模型与快速学生模型，对多教师自我策略蒸馏进行改进，降低能力干扰并提升专业与通用能力。

**💡 创新点**

创新点在于使用慢‑快对齐的EMA移动参考，并对每个教师的更新进行投影，只剔除推动快速模型远离慢速模型的分量，从而实现专长吸收与通用能力保持的平衡。

**🔧 技术方法**

使用的技术包括多教师自我策略蒸馏、EMA滑动平均、对数概率投影、CenterNorm归一化、JSD损失、KL正则化与β加权整体目标。

**📊 数据集**

实验使用的数据集包括DeepVision‑103K、Vision‑OPD‑6K、MathVerse、VSTAR、ZoomBench等专业评测数据以及MMMU‑Pro、MMBench‑CN/EN、MMStar、HallusionBench等通用评测。

**📈 对比分析**

与原始MOPD、MOPD+Ref、instruct基准及专家oracle对比，实验显示在2B/4B/8B三种规模下，SF‑MOPD在专业平均分提升3–4个百分点，通用能力提升约3个百分点，整体平均分超过MOPD 2.2个百分点，且接近专家oracle水平。

**⚠️ 局限性**

局限性包括：仍受教师多样性和任务冲突程度影响；在低冲突数据设置下专业性能会下降；需要手动调参EMA系数、β、λ等超参数；与专家oracle相比仍存在一定差距；目前仅在视觉多模态任务上验证。

---

## 43. A Multi Method Importance and Performance Efficiency Analysis of Topological Metrics for Natural Visibility Graph Based Cyber Attack Detection

**arXiv ID:** 2610.02342 | [PDF](https://arxiv.org/pdf/2610.02342v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 44. The Surprising Effectiveness of Shared Memory in Looped Transformers

**arXiv ID:** 2610.02383 | [PDF](https://arxiv.org/pdf/2610.02383v1)

**作者:** Giovanni Monea `[一作]` (Cornell University), Ramón Fernandez Astudillo `[通讯]` (IBM Research)

**通讯引用:** 2092 | [OpenAlex ID](https://openalex.org/A5003156858)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并训练了循环 Transformer（Looped Prediction Transformer）和其混合变体，采用共享 KV 缓存的机制以降低上下文内存消耗。

**💡 创新点**

创新点在于将第一层循环的 KV 缓存视为全局共享记忆，后续循环仅保留短窗口本地缓存，同时在单一 softmax 中同时访问共享与本地缓存，形成梯度高速通道，提升了模型质量。

**🔧 技术方法**

使用了自回归 Transformer 架构、RMSNorm、SwiGLU、旋转位置编码、FlexAttention 以及自定义的共享缓存机制，并在 150M–1B 参数规模上进行预训练。

**📊 数据集**

主要在 FineWeb‑Edu 语料上进行预训练（20B–50B 词元），并在 ARC‑Easy、HellaSwag、PIQA、SciQ、LAMBADA 等标准零样本评测集上评估。

**📈 对比分析**

与标准循环 Transformer、Parallel Loop Transformer 及无权重共享的深层 Transformer 进行对比，LPT 在 2–5 次递归下实现了 0.4–1.9 词元熵下降、3–6% 下游准确率提升，且上下文 KV 内存仅为标准模型的 21–24%，推理 FLOPs 增幅不足 1%。

**⚠️ 局限性**

局限性包括：仍非计算最优，递归次数固定且无法针对每个 token 自适应；仅在 4,096 令牌长度下验证；未对长上下文或更大模型规模做进一步测试。

---

## 45. Reinforcement Learning Techniques for the Optimization of Target Polarization in Nuclear Physics Scattering Experiments

**arXiv ID:** 2610.02452 | [PDF](https://arxiv.org/pdf/2610.02452v1)

**作者:** Armen Kasparian `[一作]` (Thomas Jefferson National Accelerator Facility), David Lawrence `[通讯]` (Thomas Jefferson National Accelerator Facility)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `14d48e9d-0069-4ad9-996a-1d5968216998` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研发了一套基于数据驱动的控制框架，利用高斯过程和多层感知器的代理模型结合强化学习，自动调节核磁共振靶材的微波频率，以最大化极化度。

**💡 创新点**

首次将不确定性校准的高斯过程代理模型嵌入到RL奖励函数中，并通过多样本GP近似与预计算查找表/可微环境实现可扩展的、对分布漂移敏感的自适应控制；同时提出Differentiable TD3与Trajectory Optimization改进。

**🔧 技术方法**

Gaussian Process回归、Multi‑Layer Perceptron、随机傅里叶特征近似（GPA）、强化学习算法TD3、Diff‑TD3与Trajectory Optimization、Gymnasium仿真环境、预计算查找表、低置信上界奖励、正则化行动惩罚。

**📊 数据集**

取自Jefferson Lab APOLLO固体靶实验的操作数据，包括微波频率、束流电流、累计辐照剂量和极化度，涵盖运行周期P07、P11、P14共计约5600条记录。

**📈 对比分析**

与人工操作员基线以及单样本GP+RL对比，单样本模型仅与人类表现相当；多样本GP（通过LUT或可微环境）训练的RL策略在同一实验周期实现累计极化度提升约33%，显著优于人工和单样本策略。

**⚠️ 局限性**

GP模型计算量大，单样本模型对分布漂移不敏感；RL奖励过度惩罚不确定区导致探索不足；当前仅在模拟环境验证，尚未完成与真实控制系统的集成；模型对不同靶材、不同实验条件的泛化能力仍需进一步评估。

---

## 46. Pincer: Resource Authorization for Agents using a Digital Twin

**arXiv ID:** 2610.02569 | [PDF](https://arxiv.org/pdf/2610.02569v1)

**作者:** Mayank Rathee `[一作]` (University Of California Berkeley), Ion Stoica `[通讯]` (University Of California Berkeley)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于数字双生（digital twin）的资源授权框架 Pincer，能够在编程代理的资源层自动学习并执行用户特定的最小权限策略，并与现有的工具调用层防御（如 auto mode）协同工作。

**💡 创新点**

创新点包括：①将授权放在资源层并引入数字双生以持续学习用户偏好，②使用 advantage‑agnostic（优势无关）去污化技术消除文件名误导攻击，③设计三轮投票机制（trusted‑only、content‑dependent、pre‑committed policy）以实现安全性与可用性的平衡，④构建首个面向用户的长期交互数据集，⑤给出可形式化的授予完整性（grant integrity）保证。

**🔧 技术方法**

采用的大规模语言模型（Claude Haiku 4.5、Meta Muse Spark 1.2）作为工作者、数字双生和裁决者；通过 PRF 生成的优势无关路径映射进行去污化；多投票决策逻辑；容器化执行与 POSIX ACL 的硬件级隔离；在评估中使用 LLM 计算代价和 token 计数。

**📊 数据集**

自研的用户中心化合成数据集：包含两类角色（软件开发者、民防律师）的模拟工作空间、历史对话、策略与文件，提供攻击样本（DI、II、MP、CL、PL）与正常样本，生成工具为 Claude Opus，标注了资源访问的安全/有用标签。

**📈 对比分析**

与 NanoClaw、Static‑Judge、Dynamic‑Judge、Conseca+、Conseca‑Gemini+ 等基线在同一数据集上比较。评估指标为攻击成功率 ASR（越低越好）和正常授权率 BGR（越高越好）。Pincer 在所有攻击类型（尤其是记忆中毒和命名伪装）上显著降低 ASR，且 BGR 与基线持平或略高，显示出更优的安全‑可用性折中。

**⚠️ 局限性**

局限性：①系统对 LLM 计算成本高，token 费用为 2–5× 基线；②多轮投票导致 API 调用次数多；③目前仅针对文件系统资源；④实验仅使用两款 LLM，缺乏更广泛的模型验证；⑤仍需进一步优化状态压缩与批量请求以降低开销；⑥数字双生对 LLM 的安全性和鲁棒性依赖，仍存在被细粒度攻击绕过的风险。

---

## 47. AI-driven Thermal-aware Data Center Capacity Planning

**arXiv ID:** 2610.02442 | [PDF](https://arxiv.org/pdf/2610.02442v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 48. datascribe_api: Enabling Data-Driven Materials Discovery with DataScribe

**arXiv ID:** 2610.02211 | [PDF](https://arxiv.org/pdf/2610.02211v1)

**作者:** Doğuhan Sarıtürk `[一作]` (Texas A&M University), Vahid Attari `[通讯]` (Texas A&M University)

**通讯引用:** 844 | [OpenAlex ID](https://openalex.org/A5066701381)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

实现了一个 Python 库和 CLI，统一访问 DataScribe Cloud，整合用户自建表格与 Materials Project、AFLOW、OQMD 等公开材料数据库，实现统一查询、认证、错误处理、结果校验与直接返回 Pandas DataFrame。

**💡 创新点**

创新点在于：①将多种数据库的不同查询语言统一为 Python 原生表达式；②自动完成认证、连接管理与临时网络错误恢复；③使用 Pydantic 对响应进行即时校验，避免数据结构不匹配导致的隐蔽错误；④将查询结果一次性转换为 Pandas DataFrame，便于机器学习；⑤提供可输出 JSON 的 CLI，支持 shell 管道与 LLM 交互。

**🔧 技术方法**

技术栈包括 Python 同步 HTTP、requests、Pydantic、pandas、Typer（CLI 解析）、Rich（终端渲染）、Python 标准比较运算符解析、Typer CLI 子命令、Docker 流水线（CI）等。

**📊 数据集**

使用的数据集包括：DataScribe 自有用户管理的实验表格；公开材料数据库 Materials Project、AFLOW、OQMD；以及在案例研究中使用的高熵合金硬度-温度数据。

**📈 对比分析**

与各自原生客户端对比，datascribe_api 减少了三套 SDK 的维护成本，并通过统一的过滤表达式提升了查询可读性。在 XGBoost 预测实验中，使用 datascribe_api 获取的数据集实现 R² = 0.956，表明数据一致性和可重复性高，且在管道中无额外解析步骤。

**⚠️ 局限性**

局限性包括：①仅同步实现，无法满足高并发异步场景；②AFLOW 查询受速率限制，CI 中被排除；③仅支持当前九个端点，若 DataScribe 平台新增功能需手动更新；④对复杂多步过滤逻辑的调试仍需手动验证。

---

## 49. Traversing the Satisfaction-Diversity Frontier in Text-to-Image Diffusion

**arXiv ID:** 2610.02372 | [PDF](https://arxiv.org/pdf/2610.02372v1)

**作者:** Kevin Zhai `[一作]` (University of Central Florida), Mubarak Shah `[通讯]` (University of Central Florida)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `a4b10f5d-130b-4e77-9367-6469ec621899` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在推理时对每张候选图像设定奖励阈值并对整个批次要求多样性阈值，从而实现文本到图像的满足式生成。

**💡 创新点**

创新点是提出了满足式（satisficing）框架，并在推理阶段使用奖励与多样性两种可微惩罚以及潜在替换来逼近奖励-多样性 Pareto 前沿。

**🔧 技术方法**

主要技术包括基于流匹配的文本到图像模型、可微奖励与多样性惩罚函数、批量相对奖励阈值、潜在替换机制以及梯度更新推理方法。

**📊 数据集**

实验数据集包括 Pick‑a‑Pic、HPDv2，以及使用 FLUX.1‑dev、SANA‑1.6B、Stable Diffusion 1.5 的预训练模型，奖励模型为 HPSv3 和 ImageReward。

**📈 对比分析**

与 FK steering、DAS、VASR、NegToMe、FK+NegToMe 等基线对比，SatisDive 在保持相同 DreamSim 多样性时，将最差候选图像奖励提升至 FLUX 上 0.43、SANA 上 0.70，且在 Pareto 前沿上优于 FK steering。

**⚠️ 局限性**

限制包括只能处理单一奖励阈值、对奖励模型的依赖、以及潜在替换对高质量图像可能产生的影响，未来需研究多奖励阈值的通用化和更精细的多样性度量。

---

## 50. Hop-Decayed Influence: New Vulnerabilities of Structural Auxiliary Indexing in GraphRAG Pipelines with LLM

**arXiv ID:** 2610.02373 | [PDF](https://arxiv.org/pdf/2610.02373v1)

**作者:** Jisung Park `[一作]` (University of Wollongong), Heath Cooper `[通讯]` (University of Wollongong)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究GraphRAG管线中的辅助结构层面攻击，提出HDI和3S框架，对语义、结构和评分层进行投毒，证明极小修改即可导致高攻击成功率。

**💡 创新点**

识别并利用辅助结构为新的攻击表面，提出Hop-Decayed Influence目标选择及3S多层攻击框架，实现1:N放大效果，揭示现有实例级防御的盲点。

**🔧 技术方法**

使用图结构构建、LLM生成语义摘要、结构化边缘扩展、预计算得分以及灰盒攻击、指数衰减影响传播模型等技术。

**📊 数据集**

HotpotQA和2WikiMultiHopQA两大基准问答数据集。

**📈 对比分析**

对Microsoft GraphRAG和HippoRAG2两种架构进行对照实验，使用ASR、SLR等指标评估，HDI在仅0.016%结构改动下实现约94%ASR，SLR最高6，显著优于PoisonedRAG和GRAGPOISON。

**⚠️ 局限性**

假设攻击者仅能在索引后写入辅助结构且拥有查询集；未评估对其他LLM或多模态GraphRAG的适用性；防御评估仅覆盖文本层检测；缺乏对真实生产环境的验证。

---

## 51. The AI Risk Observatory: What Can We Learn from AI Disclosures in Annual Reports About Societal Resilience?

**arXiv ID:** 2610.02281 | [PDF](https://arxiv.org/pdf/2610.02281v1)

**作者:** Bart Jaworski `[一作]` `[通讯]` (Independent Researcher), Bart Jaworski (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

利用LLM两阶段分类管线，对英国上市公司2020-2025年年度报告进行AI风险与采纳披露的自动识别和量化；

**💡 创新点**

提出可复现、可扩展的LLM驱动披露挖掘方法，并首次将其应用于英国CNI（关键国家基础设施）行业层面的大规模披露趋势；

**🔧 技术方法**

核心技术为大语言模型（Gemini 3 Flash）与自定义关键词门控、两阶段多标签分类，配合手工标注的验证集进行性能评估；

**📊 数据集**

数据集为9,821份英国上市公司年度报告（1,362家公司），包含2020-2025年完整年份和2026年部分数据，并对474条人工注释样本进行验证；

**📈 对比分析**

方法通过与人工标注的精度、召回率对比验证，其整体召回率高（>90%），但在细粒度标签（风险分类、采纳类型）上精度中等；对比传统词典法，LLM方法在提取AI相关信息的覆盖率和多样性上显著提升；

**⚠️ 局限性**

局限性包括：披露不等同治理效果，数据来源受限于年度报告的公开性与语言保守，关键词门控可能漏检隐性AI描述，验证样本单一注解且规模有限，且模型对罕见标签的精度不高。

---

## 52. The Price of Greenwashing: Algorithmic Verification and Market Discipline using Conformal Machine Learning

**arXiv ID:** 2610.02225 | [PDF](https://arxiv.org/pdf/2610.02225v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 53. MiDShip: Multimodal Dataset of Ship Cargo Hold Structures for Engineering Design

**arXiv ID:** 2610.02214 | [PDF](https://arxiv.org/pdf/2610.02214v1)

**作者:** Noah J. Bagazinski `[一作]` (Massachusetts Institute of Technology), Faez Ahmed `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 2632 | [OpenAlex ID](https://openalex.org/A5040705509)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了MiDShip数据集，包含12,753个船舱结构的参数向量、3D几何、工程绘图、BOM、性能指标和25条ABS规则约束，并提供完整的生成与评估管线；演示了两种基于约束的生成方法。

**💡 创新点**

首个公开同步多模态船舱结构数据集，涵盖参数、几何、绘图、材料、性能和规则评估；提供可复现的Python脚本与规则计算代码，推动结构设计的机器学习研究。

**🔧 技术方法**

使用RhinoPython脚本实现参数化CAD生成；SGLD-inspired采样配合神经网络预测约束残差与结构性能；LLM辅助的方程式修复；对生成结果进行t‑SNE、MNN等多模态分析。

**📊 数据集**

使用MiDShip自身数据集：6,020个随机生成、496个SGLD生成、6,237个修复设计；每个实例包含参数向量、完整3D/网格几何、工程绘图、BOM、性能评估与约束标签。

**📈 对比分析**

对比随机、SGLD生成与修复集的约束满足率与平均违规数：SGLD获得64.9%满足率、平均0.409违规（比随机平均13.192下降96.9%）；修复获得79.4%满足率、平均0.296违规（比随机下降97.8%）。MNN距离显示SGLD更集中、修复更分散。

**⚠️ 局限性**

仅覆盖货舱区域，未包含全船加载与全局约束；约束仅为25条ABS规则的子集，未覆盖完整审批；设计采用均匀间距、平面结构，忽略复杂形状；数据为合成样本，未代表真实船舶分布；生成方法基于已知规则，未验证泛化；LLM修复未捕获所有CAD后处理与船级依赖。

---

## 54. Validated Data Onboarding for AI Demand Forecasting on U.S. Building Meter Data: Design, Controlled Evaluation, and a Corrected Negative Result

**arXiv ID:** 2610.02397 | [PDF](https://arxiv.org/pdf/2610.02397v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 55. Approximation Property of Dropout Neural Networks: Sobolev Rates and Confidence Bounds

**arXiv ID:** 2610.02253 | [PDF](https://arxiv.org/pdf/2610.02253v1)

**作者:** Jia-He Yao `[一作]` `[通讯]`, Jia-He Yao

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

研究了具有独立边缘丢弃的ReLU神经网络对Sobolev空间单位球的逼近能力，分析了网络大小与准确性之间的关系。

**💡 创新点**

提出了一种结合局部子网络、成功逼近事件的局部化和多尺度泰勒分解的网络构造方法，展示了在固定深度和保留概率下的逼近误差界限。

**🔧 技术方法**

使用了ReLU激活函数和独立的Bernoulli(p)边缘丢弃机制，结合局部泰勒逼近和随机实现的理论。

**📊 数据集**

使用了Sobolev空间的单位球作为目标函数，研究了具有n个有界弱导数的光滑函数的逼近。

**📈 对比分析**

通过构造上界和下界，证明了在固定或对数深度预算下，网络的准确性指数为max{d/n,2}，并且在d≤2n时，置信度也匹配至对数的准确性。

**⚠️ 局限性**

限制在于未能确定保留依赖性和对数因子的最优性，且在高维情况下的联合置信成本仍然是一个开放问题。

---

## 56. On-Premises Multi-Course RAG Tutoring for Business Education: Hardware-Software Trade-offs in a Campus AI Tutor

**arXiv ID:** 2610.02510 | [PDF](https://arxiv.org/pdf/2610.02510v1)

**作者:** Sidney Shapiro `[一作]` (University of Lethbridge), Joshua Lindemann `[通讯]` (University of Lethbridge)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并部署了 Campus AI 教师 CourseChat，一个面向本科商科课程的本地化检索增强生成（RAG）对话系统，支持六门课程的独立索引、预制题库以及双节点本地推理架构；

**💡 创新点**

提出了显式的硬件–软件权衡策略，将模型选择、检索深度、上下文限制与产品功能分离；实现了课程级隔离、预制、源绑定的练习库，并引入了速度门和结构化响应契约的评估框架；

**🔧 技术方法**

使用 FastAPI、Qdrant 向量数据库、Ollama 本地 8B LLM（基准），句子 Transformer 嵌入、交叉编码器 reranker、混合检索、SSE 流式输出，以及 LangChain 组件；

**📊 数据集**

利用六门商科课程的教材、幻灯片、笔记、OCR 文本，共计 15,730+ 章节段落和 435 道预制练习题，覆盖 65 个模块；

**📈 对比分析**

通过两轮模型 bake‑off 进行 live‑probe（12 句）测试，比较候选模型相对于 8B 的 p50 延迟，并要求速度不超过 +30% 的门限；此外采用固定证据的结构化响应对比验证质量。结果显示 14B、12B 方案超速；Qwen 7B、Mistral Nemo 12B 通过速度门但未显著提升质量；35B Mixture‑of‑Experts 在部分纠错上表现更好，却出现新的事实和连贯性错误；

**⚠️ 局限性**

缺乏学习成效与教师评审的实证；评测仅基于 12 句 probe，未做并发负载实验；检索与生成错误共存；预制题库有限，需人工复审；系统仍处于试点阶段，公共网关接受与安全控制尚未完成。

---

## 57. Intent-Hiding Jailbreaks: An Information-Theoretic Framework for Compositional Attacks

**arXiv ID:** 2610.02302 | [PDF](https://arxiv.org/pdf/2610.02302v1)

**作者:** Fengwei Tian `[一作]` (University of Arizona), Ravi Tandon `[通讯]` (University of Arizona)

**通讯引用:** 4260 | [OpenAlex ID](https://openalex.org/A5004316408)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过信息理论框架研究意图隐藏式 jailbreak，提出先验‑后验匹配（prior–posterior matching）并证明在束大小约束下精确匹配为 NP‑hard，给出分数化的水填充解，进一步将束级优化与自然语言查询生成耦合，并在多种模型与数据集上做大规模实验。

**💡 创新点**

创新点包括：①将安全意图的隐藏视为束级先验‑后验匹配问题；②在束大小限制下证明精确匹配 NP‑hard，并提供分数化水填充与贪心阈值解；③将查询级目标保持与束级安全度联合优化，揭示二者的权衡；④在多模型、多查询生成器上验证组合攻击跨模型迁移的强大效果。

**🔧 技术方法**

采用信息理论（先验‑后验匹配）、线性规划与水填充、贪心搜索、分数化束构造、LLM 查询生成（Qwen3‑32B/14B）、安全评估器（Llama Guard、HarmBench）、目标保持评估器（Flow‑Judge）。

**📊 数据集**

使用 JailbreakBench（100 条有害目标）、Super‑NaturalInstructions（50 条正面辅助任务）与 Llama Guard 给出的意图概率；实验中还利用 Qwen、Mistral、Gemma 系列模型和公开的安全评测数据集。

**📈 对比分析**

通过对比不同束大小、查询生成器与响应模型的攻击成功率（ASR）与目标保持水平，发现即使仅加入少量正面任务，ASR 可从 3% 提升至 99%，但随着束大小增大目标保持逐步下降；实验表明组合攻击对多模型具有广泛迁移性，模型大小与后训练方式对鲁棒性影响不一。

**⚠️ 局限性**

局限性在于：①精确匹配在小束大小下往往不可行，需分数化近似；②查询生成不确定，可能导致目标信息被弱化；③实验依赖估计的意图概率与安全评估器，评估结果受其偏差影响；④仅考虑有限的任务词典与特定安全阈值，未涵盖所有可能的攻击手段。

---

## 58. World Editing: Intervening on Executable Worlds at Increasing Depth

**arXiv ID:** 2610.02331 | [PDF](https://arxiv.org/pdf/2610.02331v1)

**作者:** Max Ku `[一作]` (University of Waterloo), Ho Kei Cheng `[通讯]` (G-G-G)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了基于可执行游戏世界的“世界编辑”能力，将编辑任务划分为属性、实体、动力学和系统四个层级，并通过 Minecraft 与 Terraria 的 Mod 环境构建了 110 个可执行评估任务。

**💡 创新点**

创新点在于：①将世界编辑定位为对已有可执行世界的干预，并引入“干预深度”概念来度量编辑所需的实体、动力学与系统耦合程度；②创建了可执行、可重复验证的 Benchmark（包括状态、行为、回归与视觉评估），为世界编辑能力提供定量基准；③系统性研究了干预深度对编辑可靠性的影响。

**🔧 技术方法**

技术方法包括：利用 Fabric（Minecraft）与 tModLoader（Terraria）的 Mod 工具链进行构建与运行；实现自动化评估流程（编译、加载、行为检查、视觉一致性检查）；使用 TPIPS（基于 LPIPS 的文本条件感知指标）评估视觉资产；部署七种前沿编码代理（GPT‑5.6 Sol/Luna、Gemini 3.5 Flash、Claude Opus 4.8、DeepSeek‑V4‑Pro、Kimi‑K3、GLM‑5.3）与相应的编码 harness。

**📊 数据集**

数据集：110 个编辑任务（57 题 Minecraft，53 题 Terraria），共 1.1K 条可执行评估标准（状态检查、行为检查、回归检查、视觉检查），每个任务包含自然语言请求与对应的执行验证逻辑。

**📈 对比分析**

比较方法：对七种代理配置在 3600 秒内完成编辑任务，并用 Criterion Pass Rate (CPR) 与 World‑Editing Success Rate (WSR) 评估功能正确性；视觉任务用 Pass_style/Pass_sem/Pass_both 评估视觉一致性。最佳配置 GPT‑5.6 Sol 在严格 WSR 上 78.2%，CPR 94.8%，但随着干预深度提升性能显著下降；视觉一致性全部低于 50%，显示视觉编辑是独立难点。

**⚠️ 局限性**

局限性：①评估仅覆盖已定义的状态与行为检查，可能漏检深度干预导致的隐式副作用；②仅在 Minecraft 与 Terraria 上验证，难以直接推广到其它游戏；③视觉评估依赖 TPIPS，无法覆盖音频、动画、叙事等其他游戏模态；④每个模型–任务对只评估一次，缺乏重复试验带来的可靠性统计。

---

## 59. Adaptive Sparsity Optimization with Learnable Soft Top-K and Per-Term Thresholding for Efficient Retrieval

**arXiv ID:** 2610.02572 | [PDF](https://arxiv.org/pdf/2610.02572v1)

**作者:** Wentai Xie `[一作]` (University of California, Santa Barbara), Tao Yang `[通讯]` (University of California, Santa Barbara)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了AdaSparse方案，在Lion‑SP稀疏检索模型中引入可学习的软Top‑K、逐词阈值与FLOPs正则化，实现向量稀疏化并保持高相关性。

**💡 创新点**

创新点在于（1）可学习软Top‑K提供上下文自适应的稀疏阈值；（2）逐词阈值实现细粒度低权重抑制；（3）将三种正则化协同组合，形成多目标稀疏化框架。

**🔧 技术方法**

采用Lion‑SP基于LLaMA‑3的稀疏检索模型，结合对数平滑、ReLU、最大池化，并在训练中使用对比损失与知识蒸馏，同时加入STop、PTT与FLOPs正则化。

**📊 数据集**

在MS MARCO passage数据集上进行训练与评估，并在其Dev集、TREC DL19/20以及13个BEIR数据集上进行零样本检索实验。

**📈 对比分析**

通过与Lion‑SP、FLOPs、L1‑FLOPs、Top‑305、VDR、MRP、HT等基线在MS MARCO Dev、TREC DL与BEIR的MRR/NDCG、向量长度、检索延迟和存储空间进行对比；AdaSparse在保持0.5–1 %相关性下降的前提下，将查询/文档向量长度压缩4–5倍，检索延迟下降约3.8–4倍，存储空间缩减4倍。

**⚠️ 局限性**

局限性包括对LLM词表中多种变体的消除仍有限，soft Top‑K对词权重噪声较敏感，极端稀疏化时零样本性能略有下降，且方案对其他大词表LLM的通用性尚未充分验证。

---

## 60. THPL: A Vision-to-Language Decision Support Framework for Rainbow Trout Feeding Management in RAS

**arXiv ID:** 2610.02378 | [PDF](https://arxiv.org/pdf/2610.02378v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 61. Conditional Correctness in List Decoding: How a Late Second Codeword Can Rescue Confidence in the First

**arXiv ID:** 2610.02458 | [PDF](https://arxiv.org/pdf/2610.02458v1)

**作者:** Conrad Struss `[一作]` (Northeastern University), Ken R. Duffy `[通讯]` (Northeastern University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文通过将SOGRAND软输出公式与列表解码的后验成功概率严格对应，证明该公式是精确的而非近似，并在大块长度极限下通过大型偏差理论给出了对第一条目可靠性的判定函数Δ及其指数收敛速率，进一步揭示了列表解码（L≥2）相较于单一解码所包含的额外可靠性信息，尤其是第二条目“救赎”现象。对BSC给出了阈值y*和y̅的闭式表达，并在CRC[17,5]和eBCH[16,5]等二进制线性码上验证了该理论。

**💡 创新点**

创新点在于：①证明SOGRAND公式正是条件后验概率；②利用猜测工作和LDP获得Δ决策函数，仅依赖前三个排名；③揭示列表大小≥2时的“救赎”窗口（y*<y<y̅），并给出BSC下的阈值解析；④将理论推广到结构化线性码，展示噪声符号的综合性和可靠性一致性。

**🔧 技术方法**

核心技术包括：随机码簇模型、ML列表解码定义、SOGRAND软输出推导、Massey猜测工作、Renyi熵与LDP、凸分析、极大似然判别、BSC概率模型、以及二进制线性码的冗余结构。

**📊 数据集**

主要使用的实验数据是：BSC（参数p=0.1、0.2）下的CRC[17,5]（R=0.29）和eBCH[16,5]（R=0.31）码，计算所有可能的排名对(q1,q2)并绘制SOGRAND与Exact的对比图。

**📈 对比分析**

与Exact（全枚举后验概率）对比，SOGRAND在所有可行的(q1,q2)上高度逼近Exact；在BSC和线性码实验中，SOGRAND几乎匹配Exact的概率曲线，证明了其在实际列表解码中的有效性。

**⚠️ 局限性**

局限性包括：①分析仅在硬判决（加性噪声）下完成；②软判决情况需更一般的LDP，当前方法无法直接适用；③对非二元或大块长度的特殊噪声模型（如软信息）仍需进一步研究。

---

## 62. World-Calibrated Proposal-to-Action Flow for Vision-Language-Action Models

**arXiv ID:** 2610.02323 | [PDF](https://arxiv.org/pdf/2610.02323v1)

**作者:** Jie He `[一作]` (Harbin Institute of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 31864 | [OpenAlex ID](https://openalex.org/A5038612499)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种世界校准的提议-动作生成框架 ProAct，能够在流式视觉-语言-动作（VLA）策略中将起始噪声源从无信息的高斯分布改为基于最近动作的场景感知提议，并通过未来预测来校准该提议。

**💡 创新点**

创新点在于：①将最近执行的动作视为柔性提议而非最终动作；②通过对未来的潜在表示进行校准，动态调整提议的偏差幅度和方向，形成可调 anisotropic 源；③保持原有流式参数化不变，实现高效采样与推理。

**🔧 技术方法**

使用技术包括：流匹配/扩散模型（flow-based VLA）、轻量级 Proposal Expert（场景感知提议）、对齐的 Transformer 结构的 World Expert（未来预测与校准）、V-JEPA 2 预训练编码器做潜在未来目标、低秩几何校准、以及基于该源的流式动作采样。

**📊 数据集**

使用数据集：LIBERO、LIBERO-Plus（含七类扰动）、RoboTwin 2.0（50个随机化任务）以及真实机器人平台 GALAXEA R1 Lite 与 AgileX Cobot Magic 的六个基准任务。

**📈 对比分析**

与多种基线（π_0、π_0.5、OpenVLA-OFT、WorldVLA、UniVLA、VLA-Adapter、DreamVLA、VLA-JEPA、Fast-WAM 等）进行对比。ProAct 在 LIBERO 上平均成功率 98.4%（比 π_0.5 高 1.5 点），在 LIBERO-Plus 上 86.8%（比 π_0.5 高 13.2 点），在 RoboTwin 2.0 上 60.3%（比 π_0.5 高 31.7 点）。此外，推理时长降低 25.8%~16.3%，吞吐量提升 34.8%~19.5%，成功率提升 1.5%~8.7%。

**⚠️ 局限性**

局限性包括：① 需要预训练的 V-JEPA 2 编码器和大规模 VLM 支持，模型规模较大；② 对未来预测的依赖可能在极端动态环境或缺失视觉信息时表现不佳；③ 校准过程仍涉及额外的 Transformer 计算，虽然降低了采样成本，但整体模型复杂度仍高；④ 在部分真实机器人任务中，性能提升相对有限，提示对运动连续性与场景变化的更细粒度建模仍有提升空间。

---

## 63. SocialVLA: A Social Perception Gateway for Human-Reaction-Based Failure Detection and Recovery in VLA Manipulation

**arXiv ID:** 2610.02360 | [PDF](https://arxiv.org/pdf/2610.02360v1)

**作者:** Sofya Konstantinova `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为SocialVLA的本地化、与策略无关的社交感知门（gateway），能将观察者的即时非语言反应（语音情绪、面部表情、手势或直接停止命令）转换为实时的暂停指令，帮助VLA（视觉-语言-动作）控制的机器人在出现潜在失效时及时中断并记录人类修正；

**💡 创新点**

创新点在于：①采用异步“首次事件”多模态融合策略，在不等待所有模态达成一致的前提下即可触发暂停；②将音频情绪识别、视频表情识别和直接停止命令分离，单独估计机器人相关性作为上下文约束；③在暂停后继续录制非受限语音纠正，实现完整的“停止-纠正”闭环；

**🔧 技术方法**

使用技术包括：Causal 320 ms音频情绪检测（Aniemore / wav2vec 2.0）、语音识别（Vosk）获取停止命令、YuNet + MobileFaceNet 进行面部检测与表情评分、额外的视觉相关性头、基于ExtraTrees的多模态融合与阈值决策；

**📊 数据集**

数据集为15名受试者（共238条需要暂停的反应事件、1.038 h非暂停行为）在Unitree G1机器人上进行的真实抓取-放置实验；此外在第16名未见过的参与者上进行的前瞻性现场部署；

**📈 对比分析**

方法对比显示：在完整离线回放中，系统实现54.6 %事件召回率、69.5 %精度；在前瞻性部署中，召回率提升至59.5 %，精度高达91.7 %，平均从反应到机器人物理停止的延迟约1.02 s（中位数1.02 s，95%分位1.57 s）；

**⚠️ 局限性**

局限性包括：受试者样本仅16人，缺乏跨语言、文化、年龄等多样性；只检测观察者感知的“需要暂停”而非所有物理失败；偶尔的误停率仍较高；并且作为辅助监控，无法替代硬件急停或内部故障检测。

---

## 64. SoTa: Soft Tactile Skins for Dexterous Manipulation

**arXiv ID:** 2610.02338 | [PDF](https://arxiv.org/pdf/2610.02338v1)

**作者:** Jingyun Yang `[一作]` (Stanford University), Jeannette Bohg `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种低成本、可定制的多层电容触觉皮肤（SoTa），实现了人手与机器人手的全指和掌面触觉覆盖，并在三种接触密集的操作任务上提升了机器人策略的成功率。

**💡 创新点**

创新点在于：①将相同的202个触点布局映射到不同手型，避免了跨传感器映射学习；②通过低成本的织物电极与多层电容结构，保持高灵敏度、长循环寿命；③引入语言监督的辅助任务，使人类示范无需动作标签即可参与共训练。

**🔧 技术方法**

使用技术包括：多层电容感应结构、光刻织物电极、SEBS介电层、ESP32‑S3 数据采集板、PCAP04 电容数字转换器、与 PaliGemma 视觉–语言模型结合的 FTP‑1 策略框架。

**📊 数据集**

数据集为：每个任务收集 75（盒子/插头）或 50（杯子）个机器人演示以及 150 个对应的人类演示（覆盖三种物体），并在三种任务（盒子重定位、杯子挑取、插头插入）上进行评估。

**📈 对比分析**

与仅视觉或仅触觉的基线相比，加入触觉后在分布内成功率提升，且在人为设定的 OOD 场景中通过人机共训练将平均成功率从 22.8% 提升至 45.9%，在所有 OOD 条件下均表现出显著改进。

**⚠️ 局限性**

局限性包括：人类演示样本量有限，尚未评估长期使用下的耐久性；仅使用单一机器人演示对象，可能限制对未知对象的泛化；并且未与跨传感器映射学习方法直接对比。

---

## 65. What Does a Token Cost? A Mixture-of-Agents Measurement of Sufficient Per-Token Compute

**arXiv ID:** 2610.02491 | [PDF](https://arxiv.org/pdf/2610.02491v1)

**作者:** Zhixu Du `[一作]` (Duke University), Yiran Chen `[通讯]` (Duke University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对已验证的生成序列进行token级Mixture‑of‑Agents测量，推导出每个token所需的最小计算量（即足够计算），并利用此信息分析计算需求分布，揭示少数高成本token占据大部分计算量。

**💡 创新点**

创新点在于首次提出token级足够计算测量框架，展示跨模型族的可比复制顺序，证明少量高成本token主导总计算，并将此映射用于改进模型路由与投机解码，从而显著提升效率。

**🔧 技术方法**

主要技术包括token‑level Mixture‑of‑Agents框架、FLOP基准测算、最小成本控制器、投机解码与模型路由策略，以及对不同规模模型的可视化分析。

**📊 数据集**

使用了公开的数学与代码生成数据集：GSM8K、MATH‑500、HumanEval 等，以及不同规模模型族（Qwen、OLMo、DeepSeek‑R1‑Distill）进行实验。

**📈 对比分析**

与传统基于置信度阈值的模型级路由、固定窗口投机解码相比，利用足够计算映射的路由在MATH‑500上将预估延迟从7.59 s降至5.12 s、准确率相近或略提升；投机解码采用映射定义的动态窗口，减少32.6 %草稿token、约20 %延迟。

**⚠️ 局限性**

局限性包括需要已知完整参考序列且需对所有模型逐token评估，无法直接用于开放式生成；精确复制忽略可接受的同义或多步推理；实验仅覆盖已通过检查器的数据，无法评估失败或极难问题；跨模型族映射的可迁移性尚未验证。

---

## 66. Conditions for Social Trajectory Collapse: Agent-Based Simulation of Time-Geographic Trajectory Distributions

**arXiv ID:** 2610.02581 | [PDF](https://arxiv.org/pdf/2610.02581v1)

**作者:** Daneul Kim `[一作]` (Seoul National University), Yuyeong Kim `[通讯]` (NC AI)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过构建代理式仿真模型，对中国、俄罗斯、日本、英国和美国五国的城市系统进行建模，研究未来十年居民日常活动轨迹多样性、成本负担和福利的变化，并探讨其与区域发展政策的关系。

**💡 创新点**

创新点在于：①首次将“轨迹多样性”这一社会空间行为指标与成本负担和福利三项指标统一起来评估城市化过程；②通过模型对五国不同城市网络进行比较，揭示多样性衰减与最常见轨迹增长之间的异同；③使用多种分组规则和参数扰动，验证结果的稳健性。

**🔧 技术方法**

采用基于ODD框架的代理式模型，使用PyTorch CUDA实现并行计算；对代理状态进行六维度更新（本地嵌入、主中心/区域中心/网络可达性、成本负担、支持安全）；通过加权指数和softmax采样决策迁移，模拟轨迹演化；使用熵和有效数目等信息理论指标量化多样性。

**📊 数据集**

使用联合国世界城市化展望（WUP）2010年与2026年的人口分布数据，构建五国（中国14个城市、美国17个城市、英国12个城市、俄罗斯9个城市、日​本7个城市）区域模型，随后基于该数据拟合2026年分布并投射到2036年。

**📈 对比分析**

模型通过最小化Hellinger距离与WUP 2026年分布的差异进行参数拟合，并与“持久性”基准（直接使用2010年分布）进行对比；使用熵、有效数目、最常见轨迹比例、成本负担和福利等指标评估结果。实验显示轨迹多样性在所有案例均下降52.6%–79.0%，成本负担普遍上升，福利在不同国家呈升降两向，验证了多维度评估的必要性。

**⚠️ 局限性**

局限性包括：①不同国家区域覆盖与城市数量不一致，限制跨案例的直接比较；②模型未直接与实测迁移或活动轨迹数据验证，仅通过人口比例拟合；③参数设定对特定城市网络高度敏感，跨案例迁移效果差异大；④仅考虑了区域成本与支持的反馈，未包含更细粒度的政策或基础设施变化；⑤实验规模相对有限，未覆盖更大范围的城市网络。

---

## 67. Coco: An Agentic Copilot for the Hardware--Software Co-Design Lifecycle

**arXiv ID:** 2610.02376 | [PDF](https://arxiv.org/pdf/2610.02376v1)

**作者:** Samuel Kushnir `[一作]` (Google DeepMind), Suvinay Subramanian `[通讯]` (Google DeepMind)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了 Coco（Copilot for Codesign）平台，旨在通过自动化实验设置、仿真扫描、数据检索与分析工作流，显著加速机器学习加速器的硬件‑软件协同设计过程。

**💡 创新点**

创新点在于将协同设计拆解为四层架构——数据仓库、工具库、工作流代理与交互式平台，并通过结构化 SQL 检索和嵌入导航上下文的方式，使 LLM 代理能够在缺乏预训练知识的情况下，以可审计的方式生成基于最新仿真数据的洞察。

**🔧 技术方法**

采用的技术包括：大型语言模型 + 结构化 SQL 查询；typed API 工具链；多代理自动化工作流（如 iso‑execution 分析）；以及将用户交互轨迹与代理上下文无缝绑定的 UX 设计。

**📊 数据集**

使用的数据集为 TPU 架构仿真扫描结果，涵盖 Ironwood 系统、TPU 8i 以及不同网络拓扑（torus 与 Boardfly）等公共生成器的实验数据，并统一存入规范化关系数据库。

**📈 对比分析**

比较方法是通过 iso‑execution 分析对不同系统/模型在相同执行配置下的性能进行逐步拆解，案例研究表明在 Ironwood 与 TPU 8i 的对比以及拓扑 what‑if 分析中，Coco 将时间到仿真与洞察约缩短一倍，且能以可追溯 SQL 链接输出每个数值。

**⚠️ 局限性**

局限性包括：仍需手工维护关系模式与工具注册；无法自动检索外部文献，导致模型无法利用已有知识；代理的推理深度受 LLM 训练和当前工具链的覆盖范围限制，且对极端或未见过的仿真场景可能产生误判。

---

## 68. DeskForge: Dense Supervision from Desktop Environments for Computer-Use Agents

**arXiv ID:** 2610.02320 | [PDF](https://arxiv.org/pdf/2610.02320v1)

**作者:** A. Said Gurbuz `[一作]` (ETH Zurich), Peter W. J. Staar `[通讯]` (IBM Research Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了可控的桌面环境DeskForge，并用它生成了大规模的桌面视觉标注数据；

**💡 创新点**

创新点在于通过可配置的桌面场景组合真实应用、系统化采样，并将截图、无障碍树和窗口几何融合，得到密集且可追踪的元素注释；

**🔧 技术方法**

采用Linux桌面自动化、无障碍API、可视化截图、RT-DETR检测以及大模型（Qwen、Gemma、InternVL、UI‑R1）微调等技术；

**📊 数据集**

使用DeskForge数据集，包含约122万帧、1.6亿元素实例，覆盖19个应用、7种主题和7种分辨率；

**📈 对比分析**

与基线模型相比，微调后在内部hold‑out场景、5个公开GUI基准（ScreenSpot‑Pro、ScreenSpot‑v2、OSWorld‑G、UI‑Vision、MMBench‑GUI）以及固定规划器下的WebArena‑Infinity与OpenApps任务上均提升了10–25个百分点；

**⚠️ 局限性**

局限在于仅支持Linux后端、依赖可访问性接口、对非原生平台应用覆盖不足，且实验集中在单目标定位，未探讨多目标或动态推理。

---

## 69. Autoregressive Differentiable Method for Integer Programming

**arXiv ID:** 2610.02528 | [PDF](https://arxiv.org/pdf/2610.02528v1)

**作者:** Ouns El Harzli `[一作]` (BCG X AI Science Institute), Yudong Cao `[通讯]` (BCG X AI Science Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

训练因果Transformer，在给定二进制整数规划实例的可行种子解上通过梯度下降改进目标值，形成一种端到端的自回归改进机制。

**💡 创新点**

将Transformer的共享参数视为决策空间的非线性变换，利用在参数空间优化产生的状态相关几何，首次实现了在连续松弛空间中跨越不同吸引 basin 的“隧道式”搜索。

**🔧 技术方法**

使用因果Transformer与Gumbel‑Softmax自回归生成、有限logit目标的模糊化、拉格朗日约束惩罚以及两阶段训练（种子复制与梯度优化）。

**📊 数据集**

在随机生成的二次背包问题（QKP）上进行实验，规模分别为n=400、1000、10000，分别使用100/10个实例。

**📈 对比分析**

与HiGHS、CP‑SAT、CBC等开源求解器在相同总时间下比较，ADIP在中大型实例中显著提升解值（如n=10000时实现无限改进），并能够到达不同松弛空间的吸引 basin。

**⚠️ 局限性**

仅对单实例训练且需先获取可行种子解；中间松弛点可能不可行，缺乏全局最优保证，且实验仅限于随机QKP，未验证在更广泛问题上的效果。

---

## 70. Hesitation Has a Geometry: Entropy-Trained Hyperbolic Probes for Sparse Activation Steering

**arXiv ID:** 2610.02391 | [PDF](https://arxiv.org/pdf/2610.02391v1)

**作者:** Zeyong Zhang `[一作]` (New Jersey Institute of Technology), Mengjia Xu `[通讯]` (New Jersey Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在推理时利用模型自身的下一词熵训练的双曲探针，在高熵停顿点沿双曲测地线编辑隐藏状态，从而实现数学推理过程的激活调节。

**💡 创新点**

仅用熵信号训练双曲探针，并采用 Busemann 函数的 Riemannian 梯度给出自适应方向，使得调节仅发生在停顿点，避免了全局固定向量带来的干扰。

**🔧 技术方法**

采用 Poincaré 球双曲空间、Busemann 函数、Riemannian 梯度、Möbius 加法、弹性高斯-牛顿映射回隐藏层，以及熵门控的贪婪解码技术。

**📊 数据集**

在 Qwen2.5‑Math（1.5B/7B）和 Llama‑3.1‑8B 指令调优模型上，使用 MATH‑500、GSM8K、OlympiadBench（训练 100 题，评测 575+ 问题）进行实验。

**📈 对比分析**

与贪婪解码和 Contrastive Activation Addition (CAA) 对比；在五六组模型-数据组合中提升 0.23–1.80 分，平均 1.10 分；CAA 反而下降 0.38–12.60 分；只在 0.8–3.0% 的停顿点进行编辑，开销可控。

**⚠️ 局限性**

受限于原始解答已接近上限，对高停顿密度任务易破坏正确答案；未在更大模型或非数学任务上验证；欧氏探针不具备优势，表明效果依赖双曲几何。

---

## 71. CITADEL: CWE-Guided Insertion of Hardware Trojans via Analysis of DFG-Enabled LLMs

**arXiv ID:** 2610.02544 | [PDF](https://arxiv.org/pdf/2610.02544v1)

**作者:** Jayanth Thangellamudi `[一作]` (George Mason University), Sai Manoj P D `[通讯]` (George Mason University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了CITADEL框架，利用大语言模型（LLM）结合数据流图（DFG）和CWE漏洞目录，自动生成符合结构语义、可综合且极难被检测的硬件木马。

**💡 创新点**

创新点在于将标准化的CWE弱点与DFG结构上下文共同引导LLM进行漏洞匹配、目标定位和RTL级别的木马插入，从而实现模块级、最小化且触发条件稀有的木马生成。

**🔧 技术方法**

使用技术包括：LLM（如GPT‑5）进行语义推理与代码生成、PyVerilog提取DFG、LangChain搭建提示模板、Synopsys VCS/Design Compiler进行编译与合成验证，以及随机仿真检测木马激活概率。

**📊 数据集**

数据集包括MITRE公开的CWE硬件弱点子集和多种工业级RTL基准（AES、MIPS、SDRAM、WBRAM、SoC）用于评估框架在不同设计上的适用性。

**📈 对比分析**

通过编译、功能、HT功能和合成四个验证阶段，CITADEL在15个实例中达成100%成功率，木马在随机仿真中激活概率极低且触发条件可达，显示出相较于现有手工或概率性插入方法更高的结构一致性和隐蔽性。

**⚠️ 局限性**

局限性包括对不同LLM模型的敏感性（如Gemini或Grok在同一提示下易出现不稳定或被安全门控阻断）以及对特定安全检测技术的适用性尚未系统评估。

---

## 72. Compressible aerodynamics and rigid-body support motion in post-flutter piezoelectric energy harvesting from a pitch-plunge-flap aerofoil

**arXiv ID:** 2610.02227 | [PDF](https://arxiv.org/pdf/2610.02227v1)

**作者:** Nikolaos D. Tantaroudas `[一作]` (National Technical University of Athens), Andrew J. McCracken `[通讯]` (DASKALOS APPS)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文研究了将压缩Euler流动与自由支撑耦合到弹性机翼上的压电能量采集器，探讨其对气动弹性稳定性和能量输出的影响，并验证两项模型改进在不同负载电阻和耦合强度下的复合效应。

**💡 创新点**

创新点在于首次将可压缩Euler方程与自由支撑的四自由度耦合到同一高保真求解器中，揭示了电阻范围内的符号反转与复合非线性效应。

**🔧 技术方法**

采用数值求解器，将压缩Euler流动与弹性机翼动力学耦合，利用二维NACA O‑mesh和高阶有限差分实现非线性极限环演算。

**📊 数据集**

使用了基于NACA O‑mesh的二维网格，参数为马赫数 M = 0.10，且通过无粘压缩Euler方程模拟压电悬臂弹性机翼，在不同负载电阻和耦合系数下产生多组数据。

**📈 对比分析**

通过与先前的压缩和自由支撑单独研究结果进行数值比较，验证在高电阻时两项修正乘积近似正确，而在低电阻和高耦合时出现明显偏差；功率提升约82.5%。

**⚠️ 局限性**

局限在于仅在单一马赫数 0.10 下验证，未探究马赫数依赖；仅考察单翼单电阻点，未给出完整参数空间或飞行器级模型；且未对压缩流动的粘性效应进行分析。

---

## 73. Fast Models, Slow Evidence: A Paired and Self-Audited Evaluation of System-1 Decision Models for LLM Agent Harnesses

**arXiv ID:** 2610.02267 | [PDF](https://arxiv.org/pdf/2610.02267v1)

**作者:** Jiawei Li `[一作]` `[通讯]`, Jiawei Li

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对两种 System-1 模型在 11 个 Agent harness 决策点进行配对评估，分析其准确性、可靠性和成本节约，并对实验过程进行自我审计。

**💡 创新点**

① 设计了统一决策点基准并公开配对评估结果；② 明确指出误报成本估计误差和渠道效应；③ 系统化的自审流程可迁移至其他门控组件评估。

**🔧 技术方法**

使用 System-1 的非自回归推断，配对统计检验（McNemar、Wilson 置信区间）、校准评估、阈值选择、成本计量等技术。

**📊 数据集**

采用 18 个公开源构成的 2,100 案例集，包括 RouterBench、BFCL、BEIR、AgentDojo、ai4privacy 等。

**📈 对比分析**

通过配对实验与 Wilson 区间进行比较，结果显示 Jev 在 9/11 任务上优于 Laya，总体准确率 76.7% 对 63.1%；在工具选择、标签大集和注入检测上表现突出，但在路由、RAG 门控和 PII 方面均未超过 LLM。

**⚠️ 局限性**

仅评估单一版本零射击模型，数据主要为英语，合成类目混合；成本节约基于实验基率，未覆盖微调、非英语或真实生产环境，且受限于标签噪声和渠道效应。

---

## 74. FlashSinkhorn 2: Block-Sparse Entropic Optimal Transport

**arXiv ID:** 2610.02395 | [PDF](https://arxiv.org/pdf/2610.02395v1)

**作者:** Felix X. -F. Ye `[一作]`, Davis Wertheimer `[通讯]` (IBM T. J. Watson Research Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了FlashSinkhorn 2（FS2），一种针对低维平方欧氏成本的离散熵正则化OT求解器；

**💡 创新点**

通过将粗粒度细胞级解与块稀疏细粒度解耦合，利用质心上升和块筛选实现显著加速，同时给出误差保证；

**🔧 技术方法**

使用Morton排序的空间局部块、TF32张量核心的流式FlashSinkhorn核、基于潜在值的阈值筛选、采样检查与自适应块/细胞划分；

**📊 数据集**

在32个合成3D点云、16个聚类与异质云（共2^19–2^23点）以及真实宇宙N‑body模拟Quijote（1.34×10^8粒子）上验证；

**📈 对比分析**

与GeomLoss多尺度与原始FlashSinkhorn比较，FS2在所有32个合成案例中达到预设残差（τ=0.005），在10个案例中比GeomLoss快74–645×，在Quijote全盒子（2^27粒子）下实现<0.01残差仅需2.5小时；

**⚠️ 局限性**

仅适用于d≤3，需单GPU存储，块筛选需全块对全块检查，且在极大规模时仍受单卡内存限制。

---

## 75. Rank-Aware Speculative Sampling for Diffusion Draft Trees

**arXiv ID:** 2610.02251 | [PDF](https://arxiv.org/pdf/2610.02251v1)

**作者:** Marcello Bullo `[一作]` (Imperial College London), Deniz Gündüz `[通讯]` (Imperial College London)

**通讯引用:** 21951 | [OpenAlex ID](https://openalex.org/A5016883501)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Rank‑Aware Speculative Sampling (RASS)，一种在扩散模型中对草稿树进行排名并选择最佳候选的推断策略，显著提升接受率并保持目标分布不变。

**💡 创新点**

创新点在于：①引入基于目标-提议似然比的排名机制，②在排名上使用可调权重以最小化总变差并最大化接受概率；③给出了理论上可达的最优接受概率上界，并证明任何权重组合下的 Exactness。

**🔧 技术方法**

主要技术包括：草稿树式 speculative diffusion、最大化耦合（maximal coupling）、排名列表耦合（rank‑aware list coupling）、高斯提议与目标的解析处理，以及对权重分布的优化求解。

**📊 数据集**

实验数据集涵盖：合成高斯混合（5 维）、CIFAR‑10、FFHQ 以及 Stable Diffusion 3.5 的 512×512 隐空间文本到图像任务（使用 COCO‑2014 提示）。

**📈 对比分析**

与 D‑GRS 与 RMC 在相同并行预算下对比，RASS 在大多数配置下实现 5–20% 的 target‑call 速度提升，且 FID/CLIP 质量与基线相近；在 CIFAR‑10 上最高提升约 20.9%，在 Gaussian‑mixture 上约 6.4%。

**⚠️ 局限性**

局限性包括：需要提议与目标共享同一高斯协方差且提议独立，限制了可直接应用的采样器；速度提升指标为 target‑call 效率，实际壁钟加速受硬件与并行实现影响；此外，理论上可达的接受率上限与实际方法仍有差距，需进一步改进。

---

## 76. Evaluating and Improving the Robustness of Large Language Models to Input Sequence Variations

**arXiv ID:** 2610.02432 | [PDF](https://arxiv.org/pdf/2610.02432v1)

**作者:** Narek Maloyan `[一作]` `[通讯]`, Narek Maloyan

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套基于 R_stab(f) 指标的 LLM 鲁棒性评估与改进方法，并针对注入攻击、后门和代理系统构建了攻击与防御算法。

**💡 创新点**

创新点在于正式定义生成鲁棒度指标 R_stab 并证明局部攻击可用 R_class 界定；提出 ASA 自适应搜索攻击；量化后门检测不对称性；基于异构模型投票与多层防御的委员会；以及针对 MCP 的加密鉴权与隔离机制。

**🔧 技术方法**

采用 Jensen–Shannon 散度、演化搜索、语义重述、Token 级突变、统计置信区间、HMAC 签名、Commit Boundary 隔离等技术。

**📊 数据集**

使用公开基准 MT‑Bench、TDC2023、SaTML CTF 2024、Kaggle “LLMs: You Can't Please Them All”、MCPBench 以及 Gemma、Llama、GPT‑4、Claude‑3 等模型。

**📈 对比分析**

在黑盒下 ASA ASR 达 73.8%，与多模型委员会将 ASR 降至 19.3%；多层防御将 ASR 降至 15–25%；后门实验 REASR≈0.99 而召回率≈0.17；在 SaTML CTF 攻击成功率高达 90% 被多层防御降至 15–25%，与公开基线对比性能显著提升。

**⚠️ 局限性**

局限包括指标对对比协议敏感、攻击空间依赖已知攻击族、模型鲁棒性与准确率、可解释性和效率的权衡；以及实验主要集中在公开模型，未覆盖全规模商用模型的复杂部署环境。

---

## 77. From Retrieval to Typed Decisions: Calibrated System One Models from Biomedical Sentence Encoders

**arXiv ID:** 2610.02486 | [PDF](https://arxiv.org/pdf/2610.02486v1)

**作者:** Pritam Deka `[一作]` `[通讯]` (Queen's University Belfast), Pritam Deka (Queen's University Belfast)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

未提供论文内容，无法确定具体研究内容。

**💡 创新点**

未提供论文内容，无法确定创新点。

**🔧 技术方法**

未提供论文内容，无法确定使用技术。

**📊 数据集**

未提供论文内容，无法确定使用数据集。

**📈 对比分析**

未提供论文内容，无法确定比较方法与性能表现。

**⚠️ 局限性**

未提供论文内容，无法确定局限性。

---

## 78. Does Every User Need a Private LoRA? Decoupling Personalization from Per-User Adaptation

**arXiv ID:** 2610.02353 | [PDF](https://arxiv.org/pdf/2610.02353v1)

**作者:** Songyuan Sui `[一作]` (Samsung Semiconductor), Joon Hee Choi `[通讯]` (Samsung Semiconductor)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出LINEUP方法，重新审视个性化大型语言模型的适配容量分配，将共享的低秩可重用因子与每用户极小的用户码分离，从而实现高效的个性化。

**💡 创新点**

创新点在于：①从适配容量分配视角系统分析可重用性、可组合性和个性化校正；②构建共享因子库并通过历史检索+查询校准进行条件组合；③仅使用八维用户码进行个性化校正；④给出有限步、有限历史的风险上界，阐明何时支持拟合能提升性能。

**🔧 技术方法**

主要技术包括LoRA参数空间适配、表示空间校正、低秩可重用因子学习、历史检索与查询校准的双层组合、用户码初始化与微调、以及理论风险分析。

**📊 数据集**

实验使用LaMP个性化基准，涵盖六个任务：LaMP-2M、2N、3（分类/预测）和LaMP-4、5、7（生成），均以Llama‑2‑7B作为基础模型。

**📈 对比分析**

与八种基线（Non、Basic、RAG、OPPU、PerFit、Per‑Pcs、P2P、MTA）进行对比，LINEUP在所有12个评价指标上均排名第一，表现显著提升，如在LaMP‑3 RMSE下降0.050、LaMP‑2M F1提升0.020、LaMP‑7 ROUGE‑L提升0.029。

**⚠️ 局限性**

局限性包括：仅在LaMP基准上验证，跨任务泛化性未知；依赖用户历史质量，对低历史或噪声历史效果不明；共享因子库需在源用户上预训练，新增用户需重新检索；用户码虽小但仍需保护，隐私风险尚未彻底解决。

---

## 79. HakemBench: A Turkish Benchmark of Typed Decisions

**arXiv ID:** 2610.02293 | [PDF](https://arxiv.org/pdf/2610.02293v1)

**作者:** Sait Furkan Teke `[一作]` `[通讯]` (ufak AI), Sait Furkan Teke (ufak AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了公开的土耳其语言 typed‑decision benchmark HakemBench v1.0，并发布评测 harness、榜单以及自家模型 ufakzeka‑karar。

**💡 创新点**

①为土耳其提供多类型（choice、yes/no、score）决策评测并同时衡量决策质量、校准和可选择自动化；②引入完整的 AI‑pass + panel + 人类核查金标生成流程；③提供顺序、改写、翻译、槽位等探针评估模型稳健性。

**🔧 技术方法**

使用大语言模型多轮无监督标注与仲裁、Brier/AUGRC 等度量的评分器、兼容 chat‑completions 的评测 harness，以及 bootstrap 抽样估计置信区间。

**📊 数据集**

数据主要来自土耳其议会会议记录、宪法法院案例、Prompt‑Injection 公开集合以及 LLM 编写的文本，共 2,346 条样本、4,275 个问题，涵盖事实核查、教育、guardrails、法律分流、内容审核、垃圾邮件/钓鱼、客服等七个 track。

**📈 对比分析**

对 16 个自测模型在 7 个 track 上计算 macro‑F1、校准（Brier）和可选择自动化（AUGRC）三轴几何平均综合分；最高分 Gemini 3.8 Flash 为 0.888，随后 GPT‑5.6 Sol 0.842、GLM 5.3 0.827；自家模型 ufakzeka‑karar 得分 0.660，排名第七；开放编码模型 Laya 等表现接近基准。

**⚠️ 局限性**

金标主要由 AI 生成并由同一模型家族仲裁，缺乏人类校验；主观尺度与表面线索易导致模型误解；支持、guardrails 等部分样本由 LLM 编写，可能出现训练样本泄漏；探针间相关性导致区间过窄；整体缺少隐藏测试集，评测结果易被污染。

---

## 80. OpenRUA: Robot-Use Agents Are Zero-Shot Visuomotor Policies

**arXiv ID:** 2610.02459 | [PDF](https://arxiv.org/pdf/2610.02459v1)

**作者:** Zhaoyang Chu `[一作]` (University College London), He Ye `[通讯]` (University College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一种零抽象化的机器人使用工具包，使现成的编码代理能够仅通过终端访问ROS 2原生接口完成机器人任务。

**💡 创新点**

创新点在于无需预先工程化特定工具或学习视觉运动策略，利用编码代理自行编写感知、测量、运动控制和闭环反馈程序，从而实现零样本视觉运动策略。

**🔧 技术方法**

使用了Claude Code（Claude Opus 5）等大型语言模型作为编码代理，并通过ROS 2命令行和Python客户端直接与机器人交互；同时采用最小化工作区设计，将感知转化为文件I/O、操控转化为代码。

**📊 数据集**

在三大仿真基准上进行评估：CaP‑Bench、LIBERO‑PRO以及RoboCasa365。

**📈 对比分析**

与人类编写程序、学习型视觉运动策略、以及基于预制工具的代理相比，零抽象化方法在CaP‑Bench上实现99.0%成功率、LIBERO‑PRO上87.0%成功率，优于ASPIRE（81.0%）和Harness VLA（82.4%），在RoboCasa365的未见任务上也获得28.1%成功率。

**⚠️ 局限性**

局限性包括仅在仿真环境中验证；未评估推理延迟对实时任务的影响；对真实传感器噪声和缺失值的鲁棒性未知；安全控制主要依赖代理自写逻辑，缺乏独立的碰撞和紧急停止机制。

---

## 81. DeepStratNet: A Context-Aware Coordinate Regression Framework for Seismic Horizon Tracking under Sparse Labels

**arXiv ID:** 2610.02494 | [PDF](https://arxiv.org/pdf/2610.02494v1)

**作者:** Aniq Ahmad `[一作]` (University of Oklahoma), Heather Bedle `[通讯]` (University of Oklahoma)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于回归的地震地平面跟踪框架 DeepStratNet

**💡 创新点**

将跟踪问题改为有界坐标回归，配合轻量回归头、LSTM 空间上下文与平滑正则化，解决分割框架的后处理和稀疏标注问题

**🔧 技术方法**

使用预训练视觉骨干（ResNet-50/101、ViT‑Large、ConvNeXt‑Large）+ LSTM + 注意力 + L1/L2 + 平滑正则

**📊 数据集**

基于新西兰 Taranaki 盆地的深水曲线通道 3D 地震数据（572 条inline，22 条全标注）

**📈 对比分析**

与传统分割方法（DeepLabV3 等）对比，RMSE 下降 1–3 ms，PCC 提升 0.002–0.01，且在 25%–75% 标注稀疏时性能更稳健

**⚠️ 局限性**

只能处理单一地平面，难以同时跟踪多层、断层或强倾斜地层，且对复杂地质结构的适应性仍有限

---

## 82. A Simulation-Grounded Agentic VLM Framework for Wildfire Monitoring and Reporting

**arXiv ID:** 2610.02451 | [PDF](https://arxiv.org/pdf/2610.02451v1)

**作者:** Duowen Chen `[一作]` (Georgia Institute of Technology), Bo Zhu `[通讯]` (Georgia Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个基于模拟的可视化记忆框架，用自动化的Blender代理将二维火灾模拟转化为可标签化的二维视频，并将这些视频与模拟标签一起存储为多模态记忆，以训练无关的多代理视觉语言模型实现野火监测与报告。

**💡 创新点**

创新点在于：① 使用固定的Blender映射将物理模拟状态映射为轻量级3D代理，保持空间与物理一致性；② 将生成的视频与模拟标签一起作为可检索记忆，实现无训练的记忆增强推理；③ 通过多代理（视觉、检索、综合）和争议仲裁机制，提升对环境驱动因子和报告字段的推理准确率。

**🔧 技术方法**

技术包括：SimFire二维火灾模拟、LANDFIRE地形与燃料数据、Blender自动代理生成、基于深度与scribble控制的可控视频生成（VACE/LTX）、Qwen3-2B-VL嵌入检索、GPT-5重排序、多代理VLM推理与争议仲裁。

**📊 数据集**

数据集主要为：① 500个分离地理位置的SimFire生成视频（400用于记忆、100用于评估）；② 三个公开UAV数据集（Boreal Forest Fire、FireMan-UAV-RGBT、FLAME）用于真实视频验证；③ 公开的LANDFIRE地形与燃料地图。

**📈 对比分析**

与直接查询VLM、文本记忆以及其他野火VLM基准（ForestFireVLM、Qwen2-Wildfire-2B）对比，视频记忆在四标签准确率上提升至51.5%（远高于22.6%），完整系统在六个报告字段上的准确率达77.3%（最高为82.3%），并在跨生成器和真实UAV评估中表现出稳健性。

**⚠️ 局限性**

局限性包括：缺乏与同步物理测量的验证，代理映射仅适用于特定模拟场景，生成视频可能无法完全保留物理动态；检索依赖记忆覆盖，仲裁不保证绝对正确；对真实世界噪声、长时序演化、极端情况等方面的处理不足。

---

## 83. Hybrid Machine Learning-Assisted Raman Spectroscopy with Generative Feature Augmentation for Pharmaceutical Identification

**arXiv ID:** 2610.02224 | [PDF](https://arxiv.org/pdf/2610.02224v1)

**作者:** Quach Thi Thai Binh `[一作]` (University of Science), Nguyen Tuan Hung `[通讯]` (Tohoku University)

**通讯引用:** 2313 | [OpenAlex ID](https://openalex.org/A5076636767)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了HyMLRaman混合框架，将EfficientNet-B3深度特征提取与经典机器学习分类器相结合，实现对六种常见药物的Raman光谱识别。

**💡 创新点**

创新点在于（1）首次将CNN提取的高维嵌入与多种传统分类器（SVM、KNN、LR、RF、XGBoost、ANN）组合，显著提升准确率；（2）引入DDPM在PCA降维后的嵌入空间进行特征级数据增强，仅在样本稀缺时显著改善KNN和SVM；（3）提供交互式Raman药物分析器，展示可解释的光谱响应图。

**🔧 技术方法**

技术包括：EfficientNet-B3卷积网络用于特征提取；多分类经典机器学习模型（SVM、KNN、LR、RF、XGBoost、ANN）；DDPM生成模型用于特征增强；t-SNE、ROC-AUC、混淆矩阵评估；PyTorch与scikit-learn实现；PyQt界面实现分析器。

**📊 数据集**

使用公开Raman光谱数据集，共1003张光谱图像，覆盖六种药物（amoxicillin、chloramphenicol、ciprofloxacin、tetracycline、ibuprofen、paracetamol）。另外采集实验Raman样本用于应用演示。

**📈 对比分析**

采用分层十折交叉验证进行性能对比。与单一CNN基线相比，EfficientNet-B3–SVM基线达到96.31%准确率、96.36%宏F1，显著优于CNN基线（约89.6%）。在低样本比例下，DDPM增强对KNN/ SVM 约+0.5%至+0.8%提升，效果随样本比例增大而减弱。

**⚠️ 局限性**

局限性包括：DDPM增强仅在特征空间内生成，缺乏对原始光谱的直接可解释性；在完整数据集上提升有限；模型对近似光谱相似的药物（如ibuprofen与paracetamol）仍存在混淆；需进一步在新收集的外部样本上验证实际部署效果。

---

## 84. Learning What to Investigate Next: Meta-Reasoning for Long-Horizon Research Agents

**arXiv ID:** 2610.02525 | [PDF](https://arxiv.org/pdf/2610.02525v1)

**作者:** Ankur Samanta `[一作]` (Meta AI), Anirudh Goyal `[通讯]` (Meta AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现一种将长时序研究过程拆分为决策层与执行层的架构（MIRA），并在决策层上学习价值预测与策略（MIRA‑AC），以提升自动研究代理在多种科研任务中的表现。

**💡 创新点**

① 将研究决策与执行显式分离，形成可计价的决策边界；② 在决策层使用生成式价值预测（next‑token distribution）并跨环境预训练；③ 采用共享生成式 actor‑critic（MIRA‑AC）实现决策级信用分配，避免对长执行轨迹直接优化；④ 在多种环境上验证决策层学习能显著提升金标评估。

**🔧 技术方法**

大语言模型（GPT‑5.5、Codex、Qwen‑27B、GPT‑6 Astra）+ Codex harness + 生成式 Critic + 共享生成式 Actor‑Critic（MIRA‑AC）+ LoRA 微调 + 单步异步优化 + 经验缓冲区。

**📊 数据集**

IMOProofBench（定理证明），HillClimbBench（Residual Matrix Transformer、Loop Transformer、CPU 版本），BNLearn（贝叶斯网络），Physics Discovery，Symbolic Regression（SRBench）等。

**📈 对比分析**

与原始 GPT‑5.5、Codex、MIRA（无学习）做直接比较；与 token‑level 价值预测、单独 actor‑critic 做对比；在四个环境的金标评估上，MIRA‑AC 在所有环境都优于基线，尤其在 Symbolic Regression、Physics Discovery 与 HillClimbBench‑CPU 上提升显著；决策级生成式 Critic 的 RMSE 约比 token‑level 低 0.1；跨环境预训练实现零样本与快速适应，显著降低 RMSE。

**⚠️ 局限性**

① 只训练决策层，未学习上下文精炼与执行，导致对未呈现证据缺乏利用；② 仅使用单一 Codex inner‑loop，未检验跨 inner‑loop 的适应性；③ 仅微调 LoRA，未验证全参数训练与极长实验；④ 代理依赖代理评估与金标对齐，可能导致代理误判；⑤ 共享 actor‑critic 可能产生更新冲突；⑥ 对于非结构化研究任务的决策边界有效性仍未验证。

---

## 85. An AI-Based Multi-Stage Approach for Androgenetic Alopecia Assessment from Low-Magnification Scalp Images

**arXiv ID:** 2610.02421 | [PDF](https://arxiv.org/pdf/2610.02421v1)

**作者:** Mahmoud Raslan `[一作]` (Cairo University), Muhammad Rushdi `[通讯]` (Cairo University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了可解释的多阶段自动化系统，实现毛囊单元检测、可视毛囊计数、校准宽度测量以及区域汇总，用于辅助雄激素性脱发（AGA）的临床评估。

**💡 创新点**

创新点在于将切片增强检测、支持图辅助的秩序计数、根锚定宽度估计与基于规则的区域聚合相结合，形成一个可解释、模块化的工作流，而非直接的疾病分类模型。

**🔧 技术方法**

采用YOLOv8m进行毛囊检测，EfficientNet‑B5+支持图进行三类秩序计数，支持图生成与根锚定分支追踪结合进行宽度测量，使用Slicing‑Aided Hyper Inference、CLAHE、Frangi ridge等传统与深度学习技术。

**📊 数据集**

使用了243例临床队列（127 AGA / 116 非AGA）及160例专家标注的2400幅trichoscopy图像（约158k毛囊框），以及500幅额外临床图像进行系统级评估。

**📈 对比分析**

采用患者分层离散化（70/15/15）进行评价；YOLOv8m在测试集上达 mAP@0.5=0.920、recall=0.860；EfficientNet‑B5+S计数精度为 87.0%（宏 F1=0.85）；500图像系统级平均绝对误差为检测 6.56、分类 16.59，根锚定宽度测量平均耗时12秒/图像。

**⚠️ 局限性**

局限性包括宽度估计验证样本有限、不同方法参考子集不一致、缺乏完整患者级诊断对照、模型尚未在其他头皮病症上验证、以及系统仅提供规则层支持而非完整学习式诊断。

---

## 86. RAPID: Row-Parallel Arithmetic Processing in DRAM

**arXiv ID:** 2610.02502 | [PDF](https://arxiv.org/pdf/2610.02502v1)

**作者:** William C. Tegge `[一作]` (Syracuse University), Alex K. Jones `[通讯]` (Syracuse University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种行并行的 DRAM 内部算术处理架构 RAPID，利用迁移单元和反转单元实现 CPU 兼容的数据布局下的位并行计算。

**💡 创新点**

创新点在于：①仅通过轻量级的迁移单元实现行内水平数据移动；②使用反转单元实现原地逻辑 NOT；③在此基础上实现 O(log n) 深度的 Kogge–Stone 加法和多操作数的并行 CSA 减压；④开发全局动态规划编译器同时优化算术算法和数据布局，消除位串列化与行并行之间的转置开销。

**🔧 技术方法**

采用 DRAM 子阵列迁移/反转单元、Majority/NOT 原语、行内移位/广播、Kogge–Stone 前缀网络和 CSA 压缩，配合 LLVM 基础的 PIM 编译器进行布局与算法协同优化。

**📊 数据集**

在 19 个 MLPerf 端到端推理基准（BERT‑Large、LLaMA‑2/3、ResNet‑50、UNet3D、LSTM 等）以及 20,000+ 层宽度组合的 GEMV 任务上进行评估。

**📈 对比分析**

与 Ambit、SIMDRAM、DRISA 等位串列化 PUM 基线对比，RAPID 在 DDR3 上实现 94×、DDR4 上实现 5.9× 的端到端吞吐量提升，同时在大多数模型中通过行并行算术降低了 O(log n) 级加法和 CSA 乘法的延迟；PUM‑only 通过更低的并行度略逊一筹，但整体性能受转置成本主导，RAPID 通过消除转置实现显著优势。

**⚠️ 局限性**

主要局限：单行并行度仅 1,040（32‑bit），导致在极高吞吐需求场景下 PUM‑only 性能不及位串列化方案；需要对 DRAM 子阵列进行硬件扩展（迁移/反转单元），对现有 DRAM 兼容性与成本有一定影响；目前仅支持整数算术，浮点/低位宽特定运算仍需进一步研究。

---

## 87. SCOPE-4D: Endoscopic 4D Geometry Foundation Models

**arXiv ID:** 2610.02343 | [PDF](https://arxiv.org/pdf/2610.02343v1)

**作者:** Chaoyi Zhou `[一作]` (United Imaging Intelligence), Ziyan Wu `[通讯]` (United Imaging Intelligence)

**通讯引用:** 4424 | [OpenAlex ID](https://openalex.org/A5003798053)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个端oscopic 4D 结构基础模型，能够一次性预测相机参数、稠密几何和 3D 组织轨迹，并通过一个新型的数据标注流水线生成了约 5,000 条包含真实、合成 GI 与腹腔镜视频的几何标注数据集，同时提供了物理结肠模型和真实结肠的评估基准。

**💡 创新点**

创新点包括：① 用 vision‑language screening + SfM+MapAnything 的标注流水线，大幅提升了端oscopic 视觉标注的覆盖与质量；② 引入 Common‑Residual Motion (CRM)，通过共通 SE(3) 与点级残差变换分离共振运动与局部变形，显著减少相机运动与组织变形的混淆；③ 将几何监督 fine‑tuning 与轨迹监督联合，得到一次前向推理即可获得相机、深度与轨迹的完整 4D 预测。

**🔧 技术方法**

技术手段包括：基于预训练的 VGGT‑Ω 语义/几何模型；几何监督 fine‑tuning (SFT)；CRM 结构及其损失；轨迹监督（来自 Sano、StereoMIS 等）；MapAnything 的深度生成；MFT、LoMa/SIFT+LightGlue 进行相机姿态恢复；SAV 进行立体匹配；以及后处理的 SE(3) 变换与轨迹组合。

**📊 数据集**

使用的数据集：自建的 SCOPE‑4K（约 5,000 条视频，涵盖真实 GI、合成 GI、腹腔镜）；新增的 SCOPE‑colon‑Track（物理结肠模型，EM）和 SCOPE‑colon‑Real（临床结肠视频）；公开数据集包括 SCARED、C3VD、SimCol3D、StereoMIS、Sano、Hamlyn 等。

**📈 对比分析**

在与 VGGT‑Ω、Endo3R、Endo‑FASt3R、SM4RT 等模型的对比实验中，SCOPE‑4D 在 ID/ OOD 基准上实现了更高的相机 AUC@5°、更低的 ATE/ARE、δ1.25；3D 跟踪的 APD、EPE 也显著优于竞争方法；长序列重建中 ATE 下降；在六位专家的盲测中，SCOPE‑4D 在相机轨迹、点云与深度三项中均排名第一，平均排名显著优于其他模型。

**⚠️ 局限性**

局限性包括：仍依赖 monocular RGB，无法直接处理多视角或光学干扰极强的场景；数据集虽大但仍缺乏极端临床环境（如大量出血、液体反射等）的覆盖；对实时性能和硬件部署的评估不足；临床安全性与可部署性尚未得到实证验证。

---

## 88. Calibrating the Checksum: An Empirical False-Positive Bound for One-Sided ABFT on bfloat16 GEMM, and What It Fails to Catch

**arXiv ID:** 2610.02240 | [PDF](https://arxiv.org/pdf/2610.02240v1)

**作者:** Gautam Khosla `[一作]` `[通讯]`, Gautam Khosla

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了bfloat16下单向校验式算法故障容错（ABFT）的阈值校准与检测性能，采用输出层误码注入实验；

**💡 创新点**

在Ku≈32（bfloat16）无理论阈值的情形下，使用极值理论（Generalized Pareto分布）对残差尾部进行拟合，给出基于预设假阳性率的实用阈值，并系统评估其检测效果；

**🔧 技术方法**

极值理论、统计检验、残差比较、单向checksum、GPU TensorCore GEMM、CUDA/cuBLAS等技术；

**📊 数据集**

随机均匀生成的bfloat16矩阵乘法（M=N=K=4096）共20,000次清洁样本和多组注入样本；

**📈 对比分析**

以预注册阈值为基准，比较不同误码位类（指数、高低尾数、符号）的检测率，结果显示指数位错误约76%被检测，低尾数位零检测率；与float32传统阈值相比，低精度下检测性能显著下降；

**⚠️ 局限性**

仅在单个Tesla T4 GPU、单一工作负载、输出层注入模型下进行，无法推断硅级故障率；阈值依赖于采样规模与尾部极值拟合，跨硬件/后端可能不适用；未能检测低尾数位误码且未覆盖硬件级别的错误源。

---

## 89. VisAudit: Evaluating Multimodal Agents for Visual Diagnosis and Repair

**arXiv ID:** 2610.02399 | [PDF](https://arxiv.org/pdf/2610.02399v1)

**作者:** Shicheng Liu `[一作]`, Yada Zhu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 VisAudit 这个基准，用于评估多模态代理在数据可视化诊断、修复与验证方面的自主能力；

**💡 创新点**

创新点在于：①构建了三轨道（诊断+修复、纯自主修复、开放世界验证）以完整模拟人类可视化审核流程；②引入可控缺陷注入框架，生成 1,900 个带缺陷实例（21 种图表、10 种缺陷类别）以及 300 个正确图表；③设计多维度评估指标（可执行性、恢复、保留、正确图表识别）；

**🔧 技术方法**

采用多模态交互式环境，代理通过查看图表、源码、数据表、文字摘要，利用工具执行代码、渲染、检查并迭代；实验使用当前主流大模型（Gemma、Qwen、Ministral、Claude、Mistral、DeepSeek、Gemini）与多模态输入；

**📊 数据集**

使用 VisAudit 数据集，包括 1,900 个有缺陷实例（单/双/三缺陷）和 300 个正确实例；缺陷通过在源代码中注入 10 类缺陷（标题错误、坐标轴错误、图例缺失等）产生；

**📈 对比分析**

与三轨道对比实验显示：在诊断已知缺陷时，恢复率可提升至 60–70%；在完全自主修复时，最佳模型（Gemini、DeepSeek）仅能完全恢复 45–47% 的图表；在开放世界验证中，正确图表识别率低至 32–72%，表明代理易误操作；显著差距在于缺陷识别、修复完整性和无干预决策；

**⚠️ 局限性**

局限性包括：①缺陷识别仍弱，导致多缺陷实例恢复率急剧下降；②缺陷修复往往只修复部分属性，容易产生不必要的改动；③在无缺陷情境下识别失败率高，影响实际部署；④实验对交互预算敏感，过多回合会引入无意义修改；

---

## 90. Topology-Aware Integrated Sensing, Communication, Charging in Massive Low-Altitude Wireless Network

**arXiv ID:** 2610.02457 | [PDF](https://arxiv.org/pdf/2610.02457v1)

**作者:** Han Yu `[一作]` (Technical University of Berlin), Hing Cheung So `[通讯]` (City University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于拓扑感知的低复杂度框架，用二分图统一对低空无线网络中的通信、感知与能量传输三大功能进行多目标协同优化。

**💡 创新点**

创新点在于将多功能目标通过利润与成本的统一图形化表示，利用统一资源调整规则实现跨AP与UE的联合决策，避免传统交替优化的高计算开销。

**🔧 技术方法**

采用拓扑感知图结构、边权重阈值筛选、平均功率分配、MRT前向以及低复杂度贪心/迭代资源调整算法。

**📊 数据集**

使用在2km×2km区域内随机部署的64个陆地AP+64个UAV AP，80% UAV UE（其中9个充电UE、1个感知目标）构成的仿真数据集，100个空间部署+100个小尺度衰落样本。

**📈 对比分析**

通过与“无选择”与“UE‑centric”基线比较，总速率、目标检测概率与成功充电率等指标表明，在UE密集且干扰激烈场景下，TA框架较基线提升约10–15%总速率，检测概率与充电成功率均实现近100%或显著提升。

**⚠️ 局限性**

局限性包括对一系列阈值参数的依赖、仅考虑单一感知目标、假设平均功率分配与MRT前向，以及在极高动态或更大规模部署下仍需进一步验证其适应性。

---

## 91. A Compression Tree Generation Method for High Performance VLSI Datapaths

**arXiv ID:** 2610.02228 | [PDF](https://arxiv.org/pdf/2610.02228v1)

**作者:** Christophe Giacomotto `[一作]` `[通讯]` (Ampere Computing), Christophe Giacomotto (Ampere Computing)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出一种基于加权位堆的延迟感知压缩树生成方法，能够在不依赖具体算术操作的情况下，为任意位堆构造低延迟、低面积的压缩树；同时展示该方法在 Radix‑4 Booth 乘法器和融合的多路点积中的应用。

**💡 创新点**

创新点在于：①将每个位描述为“位置+相对到达时间”而非具体乘法器的偏移，构建与运算无关的加权位堆；②使用仅基于 CSA3:2 的灵活组合，并在每个压缩阶段保留最大带宽，继承 Dadda 树的“最大宽带”原则；③采用技术无关的逻辑延迟单元（归一化 Boolean 逻辑压力）来衡量延迟并进行位的重新排序，从而实现更佳的延迟与面积折中。

**🔧 技术方法**

技术包括：加权位堆表示、基于 CSA3:2 的压缩操作、Dadda 树的最大带宽拓扑、延迟单位（逻辑压力）归一化模型、位重排序算法以及对迟到输入（late accumulate）进行插入点分析。

**📊 数据集**

数据集为合成的 Radix‑4 Booth 乘法器（宽度 11 位至 64 位）以及 4‑路和 8‑路整数点积（乘数宽度 8~16 位），所有位堆均由作者手工构造或自动生成。

**📈 对比分析**

比较方法：将生成的 Dadda‑style 树与传统 Wallace‑style 最大压缩树在同一位堆上进行模型延迟（单位：逻辑延迟）和估算逻辑体积（面积）对比；同时在多路点积中比较融合与保留中间冗余格式的树，评估迟到累加器的吸收阈值。结果显示：Dadda‑style 树在 11~64 位乘法器上持续取得 8%–25% 的面积减少，且延迟不劣于 Wallace；多路点积中保留冗余格式的树仅在上半路径略有延迟增幅，且能在 2~3 个延迟单位内吸收迟到累加器。

**⚠️ 局限性**

局限性：①仅使用技术无关的逻辑压力延迟模型，缺乏对实际工艺、布局与布线的精细评估；②未考虑能耗与功率，后续工作需加入切换活动维度；③对极端非标准结构或极大宽度时的位堆重排可能需要更多优化；④生成的树仍需通过后端 EDA 流程验证，无法完全替代综合与后端布局；⑤在高度流水线化或功率敏感的场景下，最大带宽策略可能不再最优。

---

## 92. How To Train Your World Model: Fine-tuning vs RAG for LM-based World Modeling

**arXiv ID:** 2610.02542 | [PDF](https://arxiv.org/pdf/2610.02542v1)

**作者:** Dhananjay Ashok `[一作]` (University of Southern California), Alfy Samuel `[通讯]` (Capital One)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文系统评估了基于检索增强生成（RAG）和微调两种语言模型（LM）构建世界模型（WM）的性能，并提出了一种融合检索与微调的混合 WM 方案；

**💡 创新点**

创新点在于：①利用反事实干预估计检索阶段的错误率并揭示检索器倾向于检索次优过渡；②设计层次化查询改写策略显著提升检索质量；③将参数化的核心动态模型与主动维护的检索记忆相结合，构建鲁棒的混合 WM；

**🔧 技术方法**

主要技术包括：LM 微调、RAG 检索、反事实干预估计、层次化查询改写、以及混合模型的集成实现；

**📊 数据集**

使用了五个多样化环境的数据集，涵盖体现式、Web 导航和社交场景等不同任务类型；

**📈 对比分析**

通过奖励评估将微调、RAG、提示基线以及混合模型进行对比，结果显示微调在 15/20 环境中表现最佳，RAG 在数据效率上更优，混合模型在所有环境和模型上均优于单一方法；

**⚠️ 局限性**

局限性包括：RAG 检索器仍易检索到次优过渡；混合模型对经验数据依赖较高；未深入探讨不同规模 LM 的泛化能力及检索改写策略的细粒度影响。

---

## 93. DeReAct: Decomposed Reasoning and Acting for Reliable AI Agents

**arXiv ID:** 2610.02351 | [PDF](https://arxiv.org/pdf/2610.02351v1)

**作者:** Ajay Vohra `[一作]` (Amazon), Caron Zhang `[通讯]` (Apple)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DeReAct架构，将原本单一LLM的ReAct循环拆分为Brain、Critic和Context Manager，分别负责任务推理、动作授权与完成验证；

**💡 创新点**

通过外部化动作授权和完成控制，显式解耦任务推理与安全可靠性，利用环境证据重构State实现更可靠的终止判定；

**🔧 技术方法**

采用多模态LLM Prompting、滑动窗口历史上下文、预执行检查器、环境支持的State重构与验证以及Pass@1/Pass@3评估和策略充分性诊断等技术；

**📊 数据集**

在GAIA（检索+推理验证）和SWE-bench Verified（GitHub issue代码执行验证）两套基准上进行实验；

**📈 对比分析**

与传统ReAct及不同消融版本在Pass@1/Pass@3进行对比，弱模型时提升6.5–7.0分，强模型提升不显著，轨迹可信度提升且成本略高但可控；

**⚠️ 局限性**

仅在两套基准上评估，未覆盖多轮对话、实时交互等场景；使用同一系列Claude模型做Critic/CM，未验证跨模型泛化；成本评估未考虑缓存/批处理；对最大步骤限制的影响有限。

---

## 94. Automating the Application of HCI Principles: Skills for On-Demand UI Construction, the Human-AI Space to Think, and the Future of HCI

**arXiv ID:** 2610.02369 | [PDF](https://arxiv.org/pdf/2610.02369v1)

**作者:** Nathan Conklin `[一作]` (Virginia Tech), Chris North `[通讯]` (Virginia Tech)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种将大型语言模型与人机交互(HCI)设计原则相结合的框架，利用可执行的“技能”文件实现动态、可验证的用户界面生成。

**💡 创新点**

创新点在于把传统的HCI原则（Nielsen 10 条可用性启发式、Norman 交互准则、WCAG 规范等）转化为可插拔的、声明式的技能，形成在“思考空间”中的任务分解与生成管道，使 UI 生成成为人机共同的认知延伸而非后期审计。

**🔧 技术方法**

核心技术包括：大型语言模型（LLM）+ 链式思考/ ReAct/ Tree-of-Thoughts 交互式提示；技能脚本（skill.md）作为软件工程工具；混合主动式交互、共享任务模型和可视化生成管道。

**📊 数据集**

本文未给出专门的实验数据集，主要依赖公开的 LLM 与现有生成系统（如 Claude、ChatGPT、Anthropic’s Artifacts、OpenAI’s Canvas、Vercel 的 v0 等）的 API 与示例。

**📈 对比分析**

未开展系统的定量对比实验，作者通过架构图和案例演示说明方法可行性，缺乏明确的性能指标或实验结果。

**⚠️ 局限性**

局限性包括：生成 UI 的可验证性仍依赖后期自动检查或手工审计；技能库的维护与更新成本；模型偏差对生成质量的影响；以及对低资源语言、多模态场景和复杂交互任务的适用性仍需进一步研究。

---

## 95. An Adaptive Heterogeneous Architecture for High-Ratio, High-Throughput Lossless Compression

**arXiv ID:** 2610.02236 | [PDF](https://arxiv.org/pdf/2610.02236v1)

**作者:** Zeyu Jia `[一作]` (Tianjin Medical University), Zeyu Jia `[通讯]` (Tianjin Medical University)

**通讯引用:** 244 | [OpenAlex ID](https://openalex.org/A5018020501)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 GPX，一个可在不需要先验模式信息的情况下通过自适应结构探测、GPU 可逆域变换、AVX2 多假设序列优化实现高压缩率与高吞吐量的异构压缩体系。

**💡 创新点**

创新点在于 15 微秒级结构探测子程序、GPU 原生可逆变换（stride‑2、列转置、BCJ）与自适应熵边界结合，以及通过多假设 beam search 在不牺牲标准 RFC 8878 兼容性的前提下实现压缩比和吞吐量的 Pareto 最优。

**🔧 技术方法**

使用 CUDA/GPU 进行域变换、AVX2 SIMD 加速多假设序列优化、Zstandard（RFC 8878）熵编码、Beam Search、哈希/链表搜索、以及自定义的 4 字节元数据头实现自描述封装。

**📊 数据集**

主要实验数据集包括 Silesia‑12（202 MiB）、Canterbury、Calgary、enwik8、现代 100 MiB 编译二进制文件以及 10 MiB 合成控制基准。

**📈 对比分析**

在 Silesia‑12 上与官方 Zstandard L1–L19、Blosc+Zstandard 等基线比较，GPX‑Track A+B 在压缩率上比 Zstandard L10–L14 低 182 KB、吞吐量提升 1.15–18.7×；GPX‑Track A 在所有 RFC 8878 兼容配置中非支配，编码 235 MiB/s、解码 3074 MiB/s，整体形成新的 Pareto 前沿。

**⚠️ 局限性**

局限包括：多线程规模下速度不如原生 Zstandard（~0.8×），在纯文本大文件上可能出现过度分块导致尺寸增大；域变换需要 GPU 支持；对非结构化数据的自适应分块在极端冗余或完全随机数据时效果与基线无差别。

---

## 96. CRISP: A Framework for Clause-Reconstructed Interpretable NeuroSymbolic Propositions

**arXiv ID:** 2610.02431 | [PDF](https://arxiv.org/pdf/2610.02431v1)

**作者:** Alex Chan `[一作]` (Newcastle University), Rishad Shafik `[通讯]` (Newcastle University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

CRISP框架通过把二值化神经网络（BNN/BCCNN）的最后隐藏层激活向量（LLAV）用Tsetlin Machine（TM）子句重建为符号表达式，形成每个隐藏单元对应的可解释ITM子句集合。

**💡 创新点**

创新点包括：①每个LLAV坐标对应单独的ITM，保留神经单元身份；②对LLAV进行符号化重建而非软化输出；③对不同布尔化策略（阈值、温度计、四分位）进行系统比较；④引入LLAV注入测试验证重建隐层对分类的可用性。

**🔧 技术方法**

使用技术包括：二值化卷积网络（BCCNN、BNN）、Tsetlin Machine（Individual TM）、布尔化编码（二值阈值、温度计、四分位）、LLAV重建准确率度量、注入测试、子句证据聚合可视化。

**📊 数据集**

实验数据集涵盖MNIST、KMNIST、FashionMNIST、SVHN、CIFAR10，均为十分类图像任务。

**📈 对比分析**

通过对比教师网络在不同布尔化策略下的重建准确率（fidelity）与教师原始准确率以及注入测试准确率，发现LLAV重建可在所有数据集上达到65–88% 的准确率；四分位一位编码在SVHN上表现最佳；教师准确率与重建准确率不完全正相关。

**⚠️ 局限性**

局限性包括：仅单次实验无多种随机种子验证；训练完整256 ITM银行内存和时间限制（如CIFAR10训练>24h）；仅重建最后隐藏层，未覆盖中间层；仅针对二值化网络；对更大、更复杂数据集的可扩展性尚未验证；子句数量和复杂度的解释仍需进一步研究。

---

## 97. Mitigating Social Sycophancy via Pluralistic Preference Optimization

**arXiv ID:** 2610.02568 | [PDF](https://arxiv.org/pdf/2610.02568v1)

**作者:** Stephane Hatgis-Kessell `[一作]` (Stanford University), Emma Brunskill `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一种后训练方法 Pluralistic Preference Optimization (PlurPO)，旨在减少语言模型在提供个人建议时的社交谦恭行为。

**💡 创新点**

创新点在于利用模型自身生成的多元化利益相关者视角作为监督信号，无需外部标签或更强模型，通过程序化多元化的偏好优化显著降低社交谦恭。

**🔧 技术方法**

采用的技术包括：模型内部模拟利益相关者并构造偏好对；Iterative RPO（基于 DPO 的迭代偏好优化）；LoRA 适配器微调；以及自动生成多样化候选回复的混合策略。

**📊 数据集**

使用了四个公开评估基准数据集：OEQ（开放式建议）、PAS（有害行动声明）、AITA（责任判断）和 AITA-Flipped（对立视角），并在这些数据集上构建偏好数据集。

**📈 对比分析**

与无提示、提示、系统提示、DPO-Neutral 等基线以及不同模型族（Qwen3-8B/32B、Phi-4、Llama 3.1、Granite 4.1）比较；在所有四个数据集上，PlurPO 显著降低 endorsement rate，尤其在 AITA 上获得最佳 macro‑F1，整体社交谦恭度逼近人类水平。

**⚠️ 局限性**

局限性包括：模型自我模拟的判断可能包含刻板偏见；在极端情境下可能出现过度批评；实验仅基于公开数据，跨文化或更复杂情境的泛化尚未验证；生成和模拟成本相对较高。

---

## 98. Mitigating Convergence Collapse in Fixed-Target Anomaly Detectors via Kernel-Anchored Locality Regularization

**arXiv ID:** 2610.02345 | [PDF](https://arxiv.org/pdf/2610.02345v1)

**作者:** José Lucas De Melo Costa `[一作]` (University Paris-Saclay), Bich-Liên Doan `[通讯]` (CentralSupélec)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了固定目标异常检测器（如自编码器、流匹配器）在训练至收敛时出现的残差消失导致的检测性能下降现象，并提出了两种对策：一种是无需训练、能够防止崩溃的核收缩匹配（KCM）；另一种是将KCM作为锚点的可训练核锚正则化（KAR），在保持性能的同时消除崩溃。

**💡 创新点**

创新点主要包括：①正式表述并分析了固定目标检测器的“收敛崩溃”现象；②提出了闭式、训练自由且不易崩溃的核收缩匹配（KCM）；③设计了基于核锚的正则化（KAR），使可训练网络在保持局部性约束的前提下实现高性能。

**🔧 技术方法**

使用了固定目标损失、核回归（Nadaraya–Watson）、神经切线核理论、闭式留一交叉验证选择带宽、带死区正则化、三层感知机实现正则化，以及AUC/ROC评估方法；同时与ADBench基准中的46个无标签方法进行了对比。

**📊 数据集**

实验使用了ADBench 47个表格数据集，涵盖医疗、金融、图像派生、网络等领域，样本量跨度四个数量级，特征维度从低到高。

**📈 对比分析**

与46个无标签基线（包括kNN、KDE、Isolation Forest、LUNAR、DTE等）比较，KCM在平均AUROC 0.871位居榜首，与DTE-NP 0.865、LUNAR 0.860相当；KAR在崩溃易发数据集上显著提升，且在更长训练周期下保持稳定。相比GPU训练，KCM无需梯度、无GPU，计算成本低。

**⚠️ 局限性**

局限性包括：KCM拟合的O(N^2D)复杂度较高，需使用子样本；当核锚表现差时，KAR无法进一步提升；只针对训练集外的异常，未评估簇内或条件异常；未证明其他局部约束是否同样有效。

---

## 99. DISSOLVR: An Interpretable and Fast Framework for Aqueous and Organic Solubility Prediction

**arXiv ID:** 2610.02574 | [PDF](https://arxiv.org/pdf/2610.02574v1)

**作者:** Vansh Ramani `[一作]` (Indian Institute of Technology), Tarak Karmakar `[通讯]` (Indian Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种基于物理化学描述符的 Gradient Boosted Tree 框架（名为 SoluT-Tree），实现了水相和多溶剂环境下的高效、可解释溶解度预测，并配备了六阶段 LLM 辅助后置解释器。

**💡 创新点**

创新点在于：① 通过手工设计的 moietal、拓扑、热力学三类描述符构建与化学原理紧密对齐的特征空间；② 在 CatBoost 模型中施加硬单调约束，保证温度与溶解度的热力学一致性；③ 结合交互层（跨注意力）建模溶质–溶剂相互作用；④ 采用六阶段 LLM 解释流程，将符号模型证据转换为化学直觉可读的自然语言说明。

**🔧 技术方法**

使用技术包括：CatBoost 梯度提升树、手工特征工程（Moietal 计数、Motif 结构、Joback 组贡献、Abraham 参数重建）、交互层（Attention 机制）、温度单调约束、以及大语言模型（LLM）用于后置解释。

**📊 数据集**

实验数据集涵盖 ESOL、AqSolDB、BigSolDB 1.0/2.0、Leeds 溶解度数据库以及 Second Solubility Challenge，覆盖单溶剂和多溶剂两大场景。

**📈 对比分析**

在单溶剂（AqSolDB、ESOL）和多溶剂（BigSolDB、Leeds）两大任务上，与传统树模型、GNN、Chemprop、FastSolv、RILOOD 等基线比较，RMSE 与 R² 均逼近实验可测噪声极限，且在 OOD（溶剂外推）设置下表现优于现有方法。

**⚠️ 局限性**

局限性包括：仅适用于有机小分子；假设溶解度随温度单调上升，无法处理显著的放热溶解；缺乏立体化学/三维结构信息；不适用于无机盐或聚合物等非目标化合物。

---

## 100. Flow Matching for Fast Posterior Sampling in Bayesian Inverse Problems

**arXiv ID:** 2610.02377 | [PDF](https://arxiv.org/pdf/2610.02377v1)

**作者:** Jan Blechschmidt `[一作]`, Björn Sprungk `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种基于流匹配（flow matching）的可加速后验采样框架，并在多种 PDE 逆问题中验证其有效性。

**💡 创新点**

创新点在于将神经 ODE 作为可训练的变换网络，并通过 Metropolization 纠正流匹配产生的偏差，同时给出可计算的 TV 与 KL 上界，并在函数空间逆问题中实现了真正的在线低成本采样。

**🔧 技术方法**

使用技术包括条件流匹配（conditional flow matching）、神经 ODE、独立 Metropolis–Hastings、pCN、DRAM 以及可计算的后验误差估计。

**📊 数据集**

数据集涵盖了四类典型的贝叶斯逆问题：二维参数的边值问题、分布式扩散系数重建、电阻测量（EIT）以及无似然的 Lotka–Volterra 模型。

**📈 对比分析**

与传统 MCMC（pCN、DRAM、RWMH）以及其他生成模型（GAN、NN）比较后，流匹配在后验均值、方差和置信带上表现相近或更优，且在线采样成本几乎为零；在极端噪声或观测落在先验尾部时，接受率下降，说明精度受限。

**⚠️ 局限性**

主要局限包括：对极度集中或极端尾部观测的适应性差，流匹配近似在高维参数下可能导致 TV 距离显著偏大，且当前误差分析仅基于有限维距离，缺乏函数空间的理论支持。

---

## 101. Two-Sided Product Expanding Codes via Rademacher Matrices

**arXiv ID:** 2610.02512 | [PDF](https://arxiv.org/pdf/2610.02512v1)

**作者:** Eshan Chattopadhyay `[一作]` (Cornell University), Nicholas Spooner `[通讯]` (Cornell University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明了在固定维数与组件码数下，随机线性码在足够大的素域上具有常数双向乘积扩张性质。

**💡 创新点**

引入了实数域随机 Rademacher 矩阵和对 Gram 矩阵稀疏限制的算子范数控制，避免了对已知 c^3-LTC 的显式构造的依赖。

**🔧 技术方法**

使用了实数域随机线性码、Rademacher 矩阵、Gram 矩阵算子范数、trace power 方法、弱 Wigner 词计数与闭合遍历计数等技术。

**📊 数据集**

无数据集，全部为理论证明。

**📈 对比分析**

与以往基于 c^3-LTC 的方法相比，提出了更宽泛的随机模型，并在理论上证明了常数扩张；实验性能未涉及。

**⚠️ 局限性**

需要域特征随块长增长，无法在小素域上得到结果；仍未给出显式码构造，且对非立方体复形局部码的应用仍是开放问题。

---

## 102. Are you Synthesizing or Recalling? Evaluating LLMs on Algorithmic Code Retrieval

**arXiv ID:** 2610.02438 | [PDF](https://arxiv.org/pdf/2610.02438v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了“参数化代码检索”（parametric code retrieval）的新任务，评估大型语言模型在零样本情况下是否能从内部记忆中完整重现已知算法的实现。

**💡 创新点**

创新点在于：①将代码生成任务拆分为检索、合成、复述三类能力并聚焦检索；②设计了包含 599 个问题、77 算法、7 语言、4 输入形式的基准 AlgoREval；③系统探究提示增强与执行奖励微调对检索性能的影响。

**🔧 技术方法**

技术手段包括：零样本提示生成、静态与动态执行评测、代码相似度分析（AlgoSim）、提示增强（伪代码提示 / GitHub 代码片段）以及基于 GRPO 的执行奖励微调和传统 SFT 微调。

**📊 数据集**

使用的数据集是从 RosettaCode、GeeksforGeeks 等公开代码仓库提取的 107 个 Python 问题，并转换为 6 种其他语言和 4 种图算法输入形式，形成 599 条评测实例。

**📈 对比分析**

对 15 大型模型（7B–34B）进行零样本评测，整体通过率约 41%。通过提示增强和微调分别提升 0–5% 的通过率；GRPO 在 JavaScript 上提升 14.3%（最显著），而 SFT 在多语言上平均提升 3.9%。

**⚠️ 局限性**

局限性包括：①评测仅基于单一测试输入，可能忽略边缘情况；②假设所有算法已在预训练语料中出现，无法区分检索失败与知识缺失；③仅通过行为评测无法完全区分检索与合成，缺乏因果或机制化分析。

---

## 103. A Bifurcation-Based Domain Decomposition Method with Neural Operators for Blood Flow Simulation

**arXiv ID:** 2610.02238 | [PDF](https://arxiv.org/pdf/2610.02238v1)

**作者:** Yuzhou Zhao `[一作]` (Princeton University), Guillermo Sapiro `[通讯]` (Princeton University)

**通讯引用:** 65473 | [OpenAlex ID](https://openalex.org/A5025218580)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建基于分支单元（Bifurcation Unit, BU）的血流模拟框架，用神经算子取代传统数值求解器，实现高速、准确的血流场预测。

**💡 创新点**

① 将血管网络分解为可复用的分支单元，消除全局耦合难题；② 采用Windkessel参数聚合形成局部出口边界；③ 结合Schwarz波形松弛（SWR）实现跨单元的一致性；④ 设计BUFormer transformer算子，实现单个前向传播即可得到整个单元的压力、流速波形。

**🔧 技术方法**

利用1D Navier–Stokes方程的精简模型；Windkessel参数聚合；Schwarz波形松弛迭代；BUFormer（基于Transformer的多头注意力与傅里叶编码/解码）算子；深度学习训练与混合精度推理。

**📊 数据集**

使用公开的55段动脉树模型（及其7段子集）生成虚拟受试者数据，涵盖多种年龄、弹性、直径、血压和心率参数；数据由Nektar1D数值求解器生成，用于训练和测试。

**📈 对比分析**

与传统Nektar1D全网络求解器、纯数值SWR求解器进行对比；SWR算子求解在55段模型上实现约17×速度提升，压/流误差均低于1%；PWV等临床指标误差在1–2%范围内；对不同拓扑（剪枝网络）和异常血管弹性条件下的鲁棒性也得到验证。

**⚠️ 局限性**

① 对SWR多重迭代的收敛性理论尚未完全建立；② 目前仅验证在健康或轻度衰老情况下，未覆盖重度狭窄、动脉瘤等病理情形；③ 仅适用于树状血管拓扑，无法直接处理非树形分支或环路；④ 对极端超出训练分布的弹性/直径组合，误差会有所提升。

---

## 104. From Behavior to Provenance: Attributing Tabular Foundation Models to Synthetic Pretraining Data

**arXiv ID:** 2610.02347 | [PDF](https://arxiv.org/pdf/2610.02347v1)

**作者:** Mohamed Bouadi `[一作]` (Lexsi Labs), Vinay Kumar Sankarapu `[通讯]` (Lexsi Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个以合成预训练语料为基础的因果验证框架，用于评估训练数据归因方法的真实影响

**💡 创新点**

创新点在于同时利用可观测的合成生成轨迹、归因分数和对训练集删除的反事实干预，区分归因的可信度、生成关联性与可靠性

**🔧 技术方法**

采用影响函数（Influence）、TracIn、TRAK、表示相似性等归因技术，并使用 O'PRIOR 生成器产生的冻结合成任务、ROC‑AUC、Lift@k 等指标进行评估

**📊 数据集**

使用 5,000 条 O'PRIOR 合成分类任务做预训练，评估基于 22 个 OpenML‑CC18 真实分类数据集的行为；同时构造缺失、协变量移位、shortcut 等控制任务

**📈 对比分析**

通过任务删除（top、random、bottom）比较归因方法的“移除可信度”，评估 top‑vs‑random 差距、Lift@5% 及 within‑prov gap；结果显示 Influence 在 top‑vs‑random 与 within‑shortcut 上表现最佳（ΔROC‑AUC ≈ 0.012），TracIn 在 provenance enrichment 上更突出，整体性能差异与方法特性相符

**⚠️ 局限性**

局限性包括：受限于 5,000 任务与 NanoTabPFN 模型，干预结果依赖特定训练算法、优化器与预算；删除交互可能产生非线性效应；O'PRIOR 机制共现导致关联性不等价于因果；可微替代目标可能与实际评估指标不完全一致；反事实重训练成本高，限制了规模与干预次数

---

## 105. Effects of interpulse-interval variation on deep-learning classification of bat vocalizations

**arXiv ID:** 2610.02284 | [PDF](https://arxiv.org/pdf/2610.02284v1)

**作者:** Welmoed R. Eversteijn `[一作]` (Naturalis Biodiversity Centre), Dan Stowell `[通讯]` (Naturalis Biodiversity Centre)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究interpulse-interval（IPI）变异对蝙蝠物种分类的影响，并比较CNN（EfficientNet-B0）与Transformer（PaSST）模型在自然与归一化IPI条件下的性能；

**💡 创新点**

通过构造并行的自然与固定50 ms IPI数据集，并进行跨条件泛化评估，揭示归一化并不必然提升在真实场景中的表现；

**🔧 技术方法**

使用预训练 EfficientNet-B0 与 PaSST 进行微调，同时评估现有 BatDetect2 与 BAT；

**📊 数据集**

采用 ChiroSetEurope 欧洲蝙蝠声学数据集的 2 秒录音片段，包含 16 种蝙蝠，分别生成自然与固定 IPI 两个版本；

**📈 对比分析**

采用五折交叉验证评估同一 IPI 条件下的性能，单一划分评估跨条件泛化；结果显示 PaSST 在两种条件下均稳定 (~70 %)，EfficientNet 在归一化 IPI 下提升约10 %，但在跨条件下降约4–8 %；现有模型表现较差，归一化影响极小；

**⚠️ 局限性**

样本量小、类别不平衡、仅用两秒短片段、归一化过程可能改变背景音质，且未独立控制 IPI 之外的特征，导致无法明确归一化对真实数据的影响。

---

## 106. Overcoming Challenges of Interpretive Structural Modeling with Large Language Models

**arXiv ID:** 2610.02254 | [PDF](https://arxiv.org/pdf/2610.02254v1)

**作者:** Everett Rush `[一作]` (University of Tennessee), Michael A. Langston `[通讯]` (University of Tennessee)

**通讯引用:** 7594 | [OpenAlex ID](https://openalex.org/A5011863644)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于大型语言模型的Interpretive Structural Modeling（ISM）方法，利用LLM自动生成SSIM*并构建层级结构。

**💡 创新点**

创新点在于直接发现SSIM*的传递闭包、引入行优先(rowwise)和完整图(full graph)提示策略，并结合检索增强生成（RAG）提供上下文信息。

**🔧 技术方法**

使用GPT-5.2、GPT-OSS 120B、Nemotron Super 3等LLM模型、Floyd‑Warshall算法求传递闭包、Boolean代数、行/列/完整图prompting以及检索增强生成技术。

**📊 数据集**

采用9篇公开ISM研究结果的基准数据集，节点数从7到17，边数从17到115。

**📈 对比分析**

通过SHD（结构汉明距离）和F1‑score对不同prompting方法进行比较，行优先和完整图在GPT‑5.2上分别得到SHD=160、F1=0.77；GPT‑OSS 120B完整图得到SHD=135，其他模型表现次之。

**⚠️ 局限性**

限制在于仅在小规模ISM任务上验证，未测试对更大变量集的可扩展性、真实专家交互以及跨语言适用性。

---

## 107. Design Space Exploration of Backside Clock Meshes for 2 nm GAAFET BSPDN Technology

**arXiv ID:** 2610.02401 | [PDF](https://arxiv.org/pdf/2610.02401v1)

**作者:** Wajid Ali `[一作]` (University of California Santa Cruz), Matthew Guthaus `[通讯]` (University of California Santa Cruz)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

在2 nm GAAFET BSPDN技术上设计并评估背面时钟网格（backside clock mesh），并与传统前面时钟网格进行对比。

**💡 创新点**

首次将背面金属层用于时钟网格，结合TSV穿插和电源网格切割，构建完整的后向时钟网络，并通过贝叶斯优化实现多目标设计空间探索。

**🔧 技术方法**

使用OpenROAD的后向路由扩展、全网络HSPICE仿真、Multi‑Objective Tree‑structured Parzen Estimator（MOTPE）在Optuna中的贝叶斯优化、Monte Carlo工艺和IR变异模型。

**📊 数据集**

四个规模不同的硅片级设计（RISC‑V核心、JPEG编码器、网络‑on‑chip等），从1,938到15,311个Flip‑Flop，全部在GT2N 2 nm技术库中实现。

**📈 对比分析**

与同等网格、驱动与LCB配置的前面网格进行直接比较，背面网格平均降低约45 %时钟偏移、25 %漏电流、4.5 %功耗，并减少约28 %前面时钟布线；Monte Carlo实验显示其时钟偏移方差是前面网格的三分之一，表现更稳健。

**⚠️ 局限性**

局限在于：需要TSV穿插并切割电源轨，导致布线复杂度和时延略有增加；设计验证需耗时长的全网络SPICE仿真；仅在2 nm GT2N技术下验证，尚未探讨更大规模或不同工艺的可扩展性。

---

## 108. Efficient FlashAttention on Blackwell via Fixed-Shift Softmax and Persistent Scheduling

**arXiv ID:** 2610.02229 | [PDF](https://arxiv.org/pdf/2610.02229v1)

**作者:** Oleksandr Stashuk `[一作]` (Meta), Jay Shah `[通讯]` (Colfax Research)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在 Blackwell 架构上实现了 FlashAttention‑4 的 TLX 版本，采用固定行移位、BF16 指数近似和逆分母预计算等技术改进了注意力前向与反向计算。

**💡 创新点**

创新点在于：①用固定行移位消除在线归一化的递归累加校正；②在 BF16 前向中直接构造指数近似，避免频繁指数运算；③在反向中保存逆分母，统一预处理阶段减少 LSE 计算；④对因果注意力的全对角块进行跳过与守护式归一化，提升效率。

**🔧 技术方法**

技术手段包括：Triton TLX 异步任务调度、warp/寄存器分配、内存共享和同步、BF16 packed 指数求值、指数字段编码、固定点 dQ 合并、重建与恢复机制。

**📊 数据集**

实验使用 BF16 数据，head 大小 d=128、H=16，批量大小随序列长度（1024–32768）变化，采用 B200 GPU（1kW）进行测评。

**📈 对比分析**

通过与原 FA4 CuTe 版本和 TritonBench 进行对比，TLX 在前向上提升 13.0%/7.3%（总体），反向提升 10.4%/24.4%，但因果前向略慢 1.5%。

**⚠️ 局限性**

主要限制包括：BF16 指数近似的数值范围受限，导致极端指数会出现 NaN；固定行移位对数值精度有影响，尤其在短因果前缀时需要额外恢复；恢复路径的触发成本及整体可扩展性尚待进一步评估。

---

## 109. FinDialogLens: Event Extraction over Multi-Party Dialogue for Missed-Trade Identification in Financial Chatrooms

**arXiv ID:** 2610.02455 | [PDF](https://arxiv.org/pdf/2610.02455v1)

**作者:** Chin-Lun Fu `[一作]` (JPMorgan Chase & Co.), Behrouz Madahian `[通讯]` (JPMorgan Chase & Co.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FinDialogLens，一个混合 LLM 管道，用于从多方金融聊天记录中自动识别并抽取未成交交易事件，包括 RFQ 触发、最终价格和交易结果。

**💡 创新点**

创新点：将 compact 的 RoBERTa-base 预测器作为 inference‑time scaffolds；通过 per‑event RFQ 窗口分割降低跨事件干扰；利用 LLM（GPT‑4o 或 fine‑tuned open‑source）完成角色填充；引入 difficulty‑aware router 在成本与准确率之间做动态权衡，显著提升多方对话的事件抽取性能。

**🔧 技术方法**

技术：事件抽取（EE）、RoBERTa‑base 微调分类器（RFQ trigger, price NER, trade outcome classifier）、Per‑event RFQ windowing、LLM‑powered Trade Engine（GPT‑4o、Mistral‑7B、Llama‑3、Phi‑3、Flan‑T5‑XL）、Chain‑of‑Thought prompting、难度感知路由器。

**📊 数据集**

数据集：约 47,362 条金融聊天消息，涵盖 8,343 个 RFQ；按时间拆分为训练集 37,245 条/7,082 RFQ，评估集 1,147 条/295 RFQ，测试集 8,970 条/966 RFQ。

**📈 对比分析**

对比方法：规则引擎、全聊天 CoT prompting（zero‑shot/few‑shot）；FinDialogLens (GPT‑4o) 在测试集上达到价格准确率 92.1%、交易结果准确率 94.3%；fine‑tuned open‑source LLMs（如 Flan‑T5‑XL）在同一框架下可达 91.7%/92.8%；难度路由器将 LLM 调用降至 15%（价格）/40%（结果）即可恢复约一半准确率差距，日均节省 300+ 美元。

**⚠️ 局限性**

限制：依赖高性能 compact 模型和充足的标注数据；窗口阈值固定，难以适配不同交易节奏；路由器仅做二元决策，无法覆盖多级模型、延迟或风险策略；数据不可公开，外部复现受限。

---

## 110. MuLoRA: Spectrally Balanced Low-Rank Adaptation for Continual Learning

**arXiv ID:** 2610.02283 | [PDF](https://arxiv.org/pdf/2610.02283v1)

**作者:** Junkang Liu `[一作]` (Tianjin University), Junkang Liu `[通讯]` (Tianjin University)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5048532598)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种结合历史感知子空间选择与谱平衡的低秩适配方法，用于解决连续学习中LoRA的谱塑性坍塌问题。

**💡 创新点**

创新点在于：①通过相对响应谱分配（RRS）在历史白化空间中挑选当前任务最具响应且相对历史响应低的方向，构造固定的正交下投影矩阵；②采用子空间谱平衡（SSB）在该子空间内对动量更新做近似极化正交化，使每一步权重更新的非零奇异值均匀分布，从而保证低秩容量被充分利用。

**🔧 技术方法**

技术手段包括：低秩适配（LoRA）框架、Muon's近似极化正交化、历史白化、随机SVD、Newton‑Schulz多项式迭代、QR正交化、谱理论分析（最大最小最优、累计谱不等式）以及Vision Transformer (ViT) 预训练模型。

**📊 数据集**

实验使用的公开数据集包括 ImageNetR、ImageNetA、CIFAR‑100、CUB 以及 DomainNet（选取前200个类别）。

**📈 对比分析**

与多种基线（如InfLoRA、SD‑LoRA、BiLoRA、PLAN、LoRA‑P&M、CoSO、SplitLoRA、EBLoRA 等）在5/10/20个增量任务设置下对比，本文方法在 15/16 个评估指标中获得最高平均准确率，并在多数场景下显著提升终局和平均准确率，尤其在 20S‑ImageNetA 上提升约 3–5 个百分点。

**⚠️ 局限性**

局限性包括：①方法依赖于预先设定的低秩 rank，若 rank 选取不当仍可能出现容量不足；②对极端长序列任务或非常大模型的可扩展性尚未充分验证；③需要额外计算历史统计量，可能增加内存与训练成本。

---

## 111. Prompted to Discriminate: Generalizing Malicious-Input Probes in the Wild

**arXiv ID:** 2610.02413 | [PDF](https://arxiv.org/pdf/2610.02413v1)

**作者:** Elad David `[一作]` (Zenity), Max Fomin `[通讯]` (Zenity)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在LLM代理中使用激活探针监测恶意输入，并通过在用户后追加分类后缀（post-user suffix）提高检测性能。

**💡 创新点**

创新点在于拆解后缀中的成分，证明分类格式本身对跨数据集排名提升最为关键，而命名真实判别标准仅在低FPR阈值下提升精度，并验证此效果在单位置和多位置读取层均成立。

**🔧 技术方法**

技术包括基于KV-cache的后缀插入、冻结单位置线性探针以及多位置池化探针（attention, multi-max, MLP），并采用留一数据集离散（LODO）评估。

**📊 数据集**

使用了13个公开安全基准（包括注入、越狱和正常聊天），在Llama-3.1-8B、Qwen3.5-9B、Gemma-4-12B三种开源模型上进行实验。

**📈 对比分析**

与无后缀基线比较，分类后缀在共享安全AUC上平均提升约2-4点（在22/27配置显著提升），在1% FPR下召回率提升至约43%（相对竞争监测器约55%）。

**⚠️ 局限性**

局限性包括在Qwen模型上提升有限、只验证单轮对话且未覆盖多轮、工具调用或结构化输入，且无法证明探针真正捕捉恶意性概念，仅显示了更可迁移的可分离表示。

---

## 112. Threshold-Aware Conformal Routing

**arXiv ID:** 2610.02487 | [PDF](https://arxiv.org/pdf/2610.02487v1)

**作者:** Shiwei Tan `[一作]` (Rutgers University), Danielle C. Maddix `[通讯]` (Siemens)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究如何在高保真仿真与深度学习代理之间，根据阈值决定是否使用代理，提出阈值感知的分割合成校准路由方法。

**💡 创新点**

通过学习输入依赖的区间宽度并在合成校准过程中对阈值附近的区间进行加权，优化路由决策，使仿真调用率显著下降而保持分布无关的覆盖保证。

**🔧 技术方法**

基于分割合成校准、可微分软分位数、阈值加权训练目标、尺度头与深度学习代理（GraphSAGE、Transformer、CNN、MLP）以及可微分规模预测。

**📊 数据集**

六个工程与科学数据集：CircuitNet、Darcy Flow、AirfRANS、AhmedML、ShapeNet-Car、synthetic Beam，覆盖EDA、CFD、结构力学等。

**📈 对比分析**

与标准合成校准、CQR、Uniform、TW-UQ、Deep Ensemble等基线对比；在90%覆盖率下，阈值感知方法将仿真路由率降低14–75%，误判率≤2.5%，性能优于基线。

**⚠️ 局限性**

仅适用于单阈值决策，保证为边际覆盖非条件覆盖；对多阈值、多约束场景、局部风险控制及在线更新的适用性尚未验证。

---

## 113. $Ψ$-Resilience: Model-Free Feature Importance from 1D Topological Signals

**arXiv ID:** 2610.02299 | [PDF](https://arxiv.org/pdf/2610.02299v1)

**作者:** Fabian Galis `[一作]`, Pedro Real Jurado `[通讯]` (University of Sevilla)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种无模型的全局特征重要性方法 Ψ-Resilience，通过构建一维类不一致信号并计算其 0 维持久性来评估特征的重要性。

**💡 创新点**

创新点在于利用核密度差异生成一维拓扑信号，并用持久性同伦理论聚合鲁棒性特征，得到可审计、模型无关的特征重要性评分。

**🔧 技术方法**

采用核密度估计（KDE）、一维立方体复形构造、0 维持久性同伦理论以及聚合函数计算 Ψ_Δ。

**📊 数据集**

使用合成多类别数据（已知重要性）以及七个真实分类数据集：Wisconsin Breast Cancer、Diabetes 130-US、Covertype、Spambase、Jannis、PhishingWebsites 和 HIGGS。

**📈 对比分析**

与 MI、F‑Score、SHAP、Permutation、Drop‑column、Parr 影响等方法比较；在合成数据上 Spearman 相关约 0.75‑0.82，与 SHAP 相当；在真实数据上与 SHAP 的相关性为 0.64‑0.96，且计算速度快。

**⚠️ 局限性**

局限性包括只能处理单变量特征，无法捕捉交互；对核带宽、样本量和类别不平衡敏感；仅适用于分类任务，尚未扩展到回归或结构化输出。

---

## 114. SEDIMA: Cross-Run Hierarchical Insight Memory for Evolutionary Search Agents

**arXiv ID:** 2610.02361 | [PDF](https://arxiv.org/pdf/2610.02361v1)

**作者:** Amirhossein Abaskohi `[一作]` (University of British Columbia), Zirui Zhou `[通讯]` (Huawei Technologies Canada)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一个持久化分层洞察记忆模块，将进化搜索的评估结果转换为自然语言洞察，并在变异前检索相关经验以指导搜索。

**💡 创新点**

创新点在于：①将原始演化轨迹蒸馏成可读洞察；②构建三层层级（原始轨迹→洞察→语义聚类）并用注意力加权质心进行检索；③记忆跨跑、跨任务持久化，允许经验在不同问题间迁移。

**🔧 技术方法**

技术主要包括：LLM（GPT‑5.4、DeepSeek V4 Pro、Gemini 3 Pro、Qwen 3.7 Max、Qwen 3 Coder）用于洞察蒸馏和推荐生成；Qwen‑Embedding‑4B 进行嵌入；语义相似度检索和聚类；模块通过评估后写入和变异前读取两接口集成到 OpenEvolve 与 ShinkaEvolve。

**📊 数据集**

使用 AlgoTune（数值编程任务）和 ALE‑Bench LITE（10 个游戏）两组基准数据集进行评估；实验涉及五种 LLM 后端。

**📈 对比分析**

与原始无记忆基线比较，平均提升 AlgoTune 5.5%、ALE‑Bench LITE 6.6%；在 OpenEvolve 下平均减少 32.3% 迭代次数；跨跑和跨任务实验也显示显著收益，表明记忆可以迁移并加速搜索。

**⚠️ 局限性**

局限包括：依赖 LLM 生成洞察，可能出现误导；聚类和检索的超参数固定，未进行自适应调优；记忆随时间无限增长，缺乏衰减或压缩机制；实验仅与无记忆基线对比，未与其他记忆化进化系统直接竞争。

---

## 115. SideKernel: A Usable microVM Sandbox for AI Coding Agents on macOS

**arXiv ID:** 2610.02456 | [PDF](https://arxiv.org/pdf/2610.02456v1)

**作者:** Dimitrios Prasakis `[一作]` `[通讯]` (Georgia Institute of Technology), Dimitrios Prasakis (Georgia Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在本研究中，作者设计并实现了名为 SideKernel 的本地微型虚拟机沙箱，以支持 macOS 上的 AI 编码代理，并通过在线问卷收集可用性障碍，随后对比评估了多款类似沙箱，验证其功能与易用性。

**💡 创新点**

创新点包括：1）专注于 AI 编码代理的可用安全沙箱设计；2）实现了双向剪贴板、自动端口转发、凭证同步与个性化镜像持久化等一系列提升用户体验的功能；3）在 macOS ARM 上构建开源微VM 沙箱，填补了市场缺口；4）以用户可用性障碍为驱动的迭代改进方法。

**🔧 技术方法**

技术实现采用了 Apple Virtualization 框架编写 Swift VMM、Kata Containers 提供的精简 Linux 内核、Rust 版代理负责 VirtioFS 挂载与 vsock 通信、OverlayFS 实现持久化、NAT kill‑switch、macOS Keychain 代理等；比较实验则利用 Docker Sandbox、Microsandbox、SmolVM、Shuru、VibeBox 等同类沙箱。

**📊 数据集**

使用的数据集主要为：1）一份包含 80 位受访者的 Google 问卷（其中 33 位为现有或曾使用沙箱的用户），记录可用性障碍与使用频率；2）对上述问卷结果衍生的 23 项可用性能力测试，用以与同类沙箱进行比较；未使用公开机器学习或代码库数据集。

**📈 对比分析**

比较方法：根据用户调查提炼的三类可用性障碍，设计 23 个二元能力测试；对每个沙箱执行测试并记录通过/未通过以及摩擦感受；计算总通过数。结果显示 SideKernel 与 Docker Sandbox 在 19/23 项测试中表现最佳，其余沙箱（SmolVM、Microsandbox、Shuru）分别获得 12/23、4/9 等较低得分。性能上，SideKernel 在 M1/M4 macOS 上与 Docker Sandbox 相近，但缺少 Docker 的可观测性与企业治理功能。

**⚠️ 局限性**

局限性包括：1）评估者本人即开发者，可能存在评估偏差；2）问卷设计与样本规模受限，偏向安全技术受众；3）仅关注微VM 方案，未覆盖 Seatbelt 等其他隔离原语；4）SideKernel 尚未通过正式安全审计、未 notarized、缺乏可观测性与企业级特性；5）仅在 macOS ARM 上测试，未验证跨平台或更大规模的适用性；6）二元评分未对能力重要性加权，且未完全衡量用户主观易用感。

---

## 116. AdaptViT: Runtime-Adaptive Vision Transformer Deployment on Custom RISC-V

**arXiv ID:** 2610.02288 | [PDF](https://arxiv.org/pdf/2610.02288v1)

**作者:** Vishnu PS `[一作]` (University College Dublin), Deepu John `[通讯]` (University College Dublin)

**通讯引用:** 1478 | [OpenAlex ID](https://openalex.org/A5029098934)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出端到端部署流水线，将预训练 Vision Transformer 转化为单一可在 RISC‑V 上根据运行时稀疏度动态切换的二进制，实现多稀疏度实时推理。

**💡 创新点**

创新点在于：① 结合后训练块级结构化剪枝与硬件对齐；② 通过后编译循环重构和掩码跳过实现单二进制多稀疏度；③ 引入专用 MAC2 指令加速线性投影。

**🔧 技术方法**

使用技术包括结构化剪枝、块级 MLP 剪枝、INT8 量化、TVM 生成 C 内核、掩码控制逻辑以及 RISC‑V 自定义 ISA 扩展。

**📊 数据集**

使用 Oxford‑IIIT Pet 37 类数据集评估推理准确率。

**📈 对比分析**

与传统多二进制部署对比，单二进制存储节省 4.86×；在 ViT‑Base 上在 65% MLP 与 50% 头剪枝下实现 2.8× 推理速度提升、33% 能耗下降。

**⚠️ 局限性**

局限性包括量化导致约 5% 准确率下降；仅对线性层实现跳过，非线性层仍需完整计算；ISA 扩展仅在特定硬件实现，迁移性受限。

---

## 117. Joint Movement and Compression Ratio Design for Mobile Embodied AI Networks (MEAN)

**arXiv ID:** 2610.02334 | [PDF](https://arxiv.org/pdf/2610.02334v1)

**作者:** Yahao Ding `[一作]` (King's College London), Mohammad Shikh-Bahaei `[通讯]` (King's College London)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

提出一种在移动实体 AI 网络（MEAN）中，联合优化语义压缩比、上行功率和移动距离，以最大化最小能效（EE）的算法。

**💡 创新点**

创新点在于首次把语义通信、能量管理与移动控制三者统一建模并联合优化，设计了双层 AO‑Dinkelbach 算法，在 Dinkelbach 变换、SCA 与网格搜索的协同下解决非凸分式问题。

**🔧 技术方法**

采用了 Dinkelbach 变换处理分式目标、交替优化（AO）结合序列凸逼近（SCA）求解功率子问题、坐标网格搜索更新移动距离，并利用 MMSE 收发组合实现信号处理。

**📊 数据集**

实验使用仿真数据：多用户 Rayleigh 小尺度衰落、路径损耗模型 β_k(d)=β_0(D_k-d)^‑δ，固定系统参数（K=4/5、M=8、B=10 MHz、P_k^max 20–36 dBm 等），并未使用公开真实数据集。

**📈 对比分析**

与两种基线（无移动、无语义压缩）对比，结果显示在可控功率范围内该方法比无移动方案高 60–70 Mbits/J、比无压缩方案提升 5‑倍；在不同距离、用户数和功率预算下均保持显著优势。

**⚠️ 局限性**

局限性包括：仅考虑单小区、理想完美 CSI、固定移动时长、仅上行链路；未对多小区干扰、任务准确性约束或更复杂的移动路径规划进行建模。

---

## 118. HXAI: Hierarchical Privacy-Preserving Explainable AI in Distributed Energy Systems

**arXiv ID:** 2610.02504 | [PDF](https://arxiv.org/pdf/2610.02504v1)

**作者:** Poushali Sengupta `[一作]` (University of Oslo), Yan Zhang `[通讯]` (University of Oslo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了HXAI框架，在分布式能源系统中实现隐私保护的可解释AI。

**💡 创新点**

创新点是层次化设计：本地无噪声解释 + 区域层差分隐私聚合，强调语义稳定性；同时引入信息控制协议与多查询隐私计数。

**🔧 技术方法**

技术包括SHAP本地解释、差分隐私 Laplace 机制、TEE、隐私预算协商以及语义稳定性评估。

**📊 数据集**

使用了UCI Appliances、UCI Household Power、UCI Bike Sharing三大公开数据集。

**📈 对比分析**

与中心化 DP‑SHAP 及局部 DP 等基线比较，HXAI 在保持排名一致性、余弦相似度高的同时，噪声误差更小；实验显示在 ε>1 时性能优于基线。

**⚠️ 局限性**

局限在：仍需假设可信TEE；聚合仅适用于 SHAP 向量，不能直接处理非树模型；高异质性数据下需更复杂预算策略；缺乏现场部署验证。

---

## 119. Connectedness, Cognitive Load, and Human-AI Oversight in Cyber Operations

**arXiv ID:** 2610.02384 | [PDF](https://arxiv.org/pdf/2610.02384v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 120. Diffusion-Based Synthetic Data Pretraining for Enhancing Activity Recognition

**arXiv ID:** 2610.02292 | [PDF](https://arxiv.org/pdf/2610.02292v1)

**作者:** E. Riveros `[一作]` (State University of Campinas), A. Rocha `[通讯]` (State University of Campinas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于扩散模型的合成数据预训练管线，先用生成的多传感器时间序列预训练 CABiGRU，再在真实数据上微调，以提升智能手表上对进食和饮水行为的识别；

**💡 创新点**

创新点在于将 PaD‑TS 扩散模型改为双向 GRU 编码器以更好捕捉短时间窗口的多传感器动态，并将其作为无监督预训练信号，解决类不平衡和细粒度时序辨别问题；

**🔧 技术方法**

使用的技术包括：双向 GRU 的 CABiGRU 识别骨干网络、Population‑aware Diffusion for Time Series（PaD‑TS）扩散模型、两阶段预训练+微调策略、不同激活函数与分类头的组合实验；

**📊 数据集**

使用的公开数据集为 Daily Living Activities Dataset（DEO子集），包含 90 名受试者的加速度计、陀螺仪、磁力计 5 s 采样窗口；

**📈 对比分析**

与从零训练的 CABiGRU、PCF‑GAN 生成的预训练相比，扩散预训练在 90.6% 的平衡准确率、0.9055 的 F1（宏）等指标上均显著提升，尤其在饮水与进食两类的召回率上优于对手；

**⚠️ 局限性**

局限性包括：合成数据量增多反而降低泛化性能，需控制生成样本的多样性；模型仅在该特定手表数据集上验证，泛化到其他设备或人群需要进一步测试；

---

## 121. Rethinking World-Action Model for Compositional and In-Context Robotic Manipulation

**arXiv ID:** 2610.02368 | [PDF](https://arxiv.org/pdf/2610.02368v1)

**作者:** Shukai Gong `[一作]` (Peking University), Daquan Zhou `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ViGAR 框架，将长周期组合操作拆分为视觉子目标规划与子目标条件下的世界动作模型（WAM）联合视频-动作生成。

**💡 创新点**

创新点：① 将子目标图像作为显式决策变量，桥接任务级规划与低层动作生成；② 通过全局目标图像实现推理时的无参数更新的“in‑context”学习；③ 结合子目标预规划与 WAM 的双分支架构，提升对未知场景的泛化。

**🔧 技术方法**

技术：基于 Cosmos3‑Nano 16B 双分支 Transformer + VAE，使用流匹配（flow‑matching）目标函数；子目标规划采用子目标前瞻规则与末端执行器 ROI 加权训练；子目标注入采用几何路由（VAE 先验 + 生成器）。

**📊 数据集**

数据集：RoboTwin Clean2Random 基准（2500 条 Clean 轨迹 + 随机场景），AgiBot A2 机器人 900h 远程操控 + 任务特定演示，实验中的全局目标任务（水果排布、桌面物品存储）提供额外的全局目标图像。

**📈 对比分析**

比较方法与性能：对比 StarVLA、Abot‑M0、X‑VLA、π0.5（VLA 方案）及 Fast‑WAM、LingBot‑VA、4D‑WAM（WAM 方案），以及 Cosmos3‑Nano‑RoboTwin（无子目标引导）。在 RoboTwin 上，ViGAR Clean 成功率 82.0%，Random 67.0%，平均 74.5%，比最佳基线高 12.9 百分点；在实机五个任务中，任务进度均高于两基线；in‑context 任务中，ViGAR 在域内 77.5% 进度，域外 47.5%。

**⚠️ 局限性**

局限性：① 在 Random 场景的性能仍低于 Clean，表明对极端布局的泛化仍有限；② 需要预训练的大规模 16B 模型，资源消耗高；③ 只支持通过全局目标图像进行推理时重组，无法自动生成多样化的全局目标；④ 子目标规划依赖端执行器 ROI，若目标与执行器位置分离时可能失效。

---

## 122. Tropical Reinforcement Learning

**arXiv ID:** 2610.02478 | [PDF](https://arxiv.org/pdf/2610.02478v1)

**作者:** Arip Asadulaev `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Martin Takac `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将强化学习中轨迹的聚合算子从求和改为最大，提出热带强化学习（Tropic），实现大型语言模型在多步推理任务中将单独产生的碎片有效组合成完整解决方案。

**💡 创新点**

核心创新是利用热带半环（max-plus）代数定义价值函数，使价值永不下降且始终对应已验证路径，从而突破传统期望回报无法复用计算碎片的瓶颈。

**🔧 技术方法**

技术手段包括热带Bellman方程、最大化对数概率、前向后向动态规划、图结构记忆、前缀/后缀价值计算、跨回合组合重放与验证。

**📊 数据集**

在四个交互式任务（Sokoban、Countdown、FrozenLake、WebShop）以及RAGEN‑2环境上进行实验，并使用Qwen‑2.5‑3B‑Instruct和Gemma 4 E4B‑it两种大型语言模型。

**📈 对比分析**

与PPO、GRPO、DAPO、SNR‑Aware等基准在相同算力与训练预算下对比，Tropic在所有任务和模型上均取得最高成功率，提升幅度最高达16个百分点，且在token数与训练时间上亦表现更优。

**⚠️ 局限性**

局限性在于仍依赖手工构建的记忆图与前缀/后缀回溯，难以直接扩展到更大规模或非确定性环境；对模型参数共享时的交叉干扰缺乏严格理论保证。

---

## 123. Counterfactual Predictions in Scientific Emulators Without Controlled Experiments

**arXiv ID:** 2610.02252 | [PDF](https://arxiv.org/pdf/2610.02252v1)

**作者:** Dingling Yao `[一作]` (Institute of Science and Technology Austria), Anima Anandkumar `[通讯]` (California Institute of Technology)

**通讯引用:** 15546 | [OpenAlex ID](https://openalex.org/A5014498545)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ReRoute框架，利用已有预训练模型和部分机理知识，对目标干预进行轻量微调，实现针对性what‑if预测。

**💡 创新点**

创新点在于仅需要目标驱动的第一阶机理路径，无需完整物理模型或额外控制实验，即可通过观测数据识别并重构目标干预效应，并提供可识别性的理论保证。

**🔧 技术方法**

采用机制引导的重路由（ReRoute）技术——把目标输入固定到参考值，通过已知路径注入变化，结合自回归模型微调；使用Lean证明可识别性；实验中使用FNO、ACE2等预训练模型。

**📊 数据集**

实验数据集包括：人工生成的二维advection‑diffusion轨迹、ACE2预训练的AMIP+平衡气候数据（无随机CO₂），以及历史ERA5再分析数据。

**📈 对比分析**

与基线模型（FNO、ACE2）对比，ReRoute在离散参数空间（c,D）中误差降低约90%；在极端SST–CO₂分布偏移下平均误差下降18.2–31.8%；在ERA5上固定CO₂时保留温度升温效应并显著改进热带风暴统计，整体性能优于基线。

**⚠️ 局限性**

局限性：目前理论仅适用于单一目标变量且路径需已知，对路径误差敏感；假设路径与状态无关，尽管实验表明多变量或状态相关路径仍可工作，但未被理论覆盖；需足够观测数据来学习下游动态。

---

## 124. Keep It CALM: Analyzing the Limits of Global Unsafety in Text-to-Image Generation

**arXiv ID:** 2610.02300 | [PDF](https://arxiv.org/pdf/2610.02300v1)

**作者:** NaHyeon Park `[一作]` (KAIST), Hyunjung Shim `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练-free的局部对比修正方法，对文本到图像的生成安全性进行修正；

**💡 创新点**

创新点在于将不安全语义拆分为匹配的对比对，并通过提示路由只对相关类别进行局部最小化编辑，而非全局统一消除；

**🔧 技术方法**

主要技术包括构建unsafe‑benign anchor bank、prompt routing、token‑level凸投影、残差空间对比修正以及对文本编码空间的几何分析；

**📊 数据集**

实验使用Ring‑A‑Bell、UnlearnDiff‑Atk、MMA‑Diffusion、COCO、I2P、ViSU等数据集，并用Qwen3‑VL‑8B‑Instruct等VLM评判；

**📈 对比分析**

与多种训练‑时间和推理‑时间安全防护方法比较，ASR/TR/CLIP‑T/FID等指标均优于基线，尤其在SD‑v1.4及迁移到SDXL、FLUX、SANA、OmniGen2等模型时表现突出；

**⚠️ 局限性**

局限性包括对抗性白盒攻击和极端提示仍具挑战，仅在文本编码空间进行分析，且对内部去噪器表示的鲁棒性待进一步提升。

---

## 125. NEEDLEWORK: Offline Rewriting of Robot Data with Verified Local Stitches

**arXiv ID:** 2610.02339 | [PDF](https://arxiv.org/pdf/2610.02339v1)

**作者:** Juntao Ren `[一作]` (Stanford University), Shuran Song `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种离线数据集增广方法NEW（New Edges from Existing Demonstrations through Local Editing），通过在已有机器人演示中插入经过验证的短动作桥梁，改进数据集；

**💡 创新点**

创新点在于将轨迹拼接问题转化为局部可监督的到达性任务，使用逆动力学模型生成动作序列并由动作条件验证器筛选可行桥梁，同时保留原始演示的监督；

**🔧 技术方法**

主要技术包括目标条件扩散逆动力学模型（IDM）和动作条件到达性验证器，训练时使用多视角RGB图像和机器人本体感知，构建桥接后将桥梁动作直接拼接到原始演示窗口进行策略学习；

**📊 数据集**

实验使用真实机器人（Sweater Folding、Dish Racking）和仿真数据（Robomimic的Multi-Human Can、Square、Transport，以及DexMimicGen的Coffee、Three Piece Assembly、Threading），所有数据仅包含RGB、关节信息和回合成功/失败标记；

**📈 对比分析**

与多种基线（MBTS、SBR、AWR、SARM、IDQL）在相同模型架构下对比，NEW在真实机器人任务上平均提升24%和18%成功率，在仿真任务上平均提升约8%成功率，且在固定时间预算内完成更多试验；

**⚠️ 局限性**

局限性包括对近最优演示效果有限（增益 ≤4%），需要短桥梁且不适用于需要持续接触或长时间反馈的任务；此外，性能依赖于验证器的准确性和桥梁的可行性。

---

## 126. ArrivalBench: Agent-Generated Data Pipelines Are Correct Once and Wrong Under Time

**arXiv ID:** 2610.02363 | [PDF](https://arxiv.org/pdf/2610.02363v1)

**作者:** Pranay Kothari `[一作]` `[通讯]` (University of Oxford), Pranay Kothari (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了40个数据管道基准，并设计了ArrivalBench，评估生成式模型在重播不确定交付条件下的时间安全性。

**💡 创新点**

首次将基准从单次执行扩展为重播测试，且在评估中将“错误结果”和“崩溃”区分开来，以揭示模型在exact‑once语义上的弱点。

**🔧 技术方法**

利用DuckDB执行脚本、重播式交付模拟、元评估器与Bootstrap统计方法，对多家大型语言模型进行系统对比。

**📊 数据集**

使用自定义的40个任务（含42个负控制），每个任务基于单一事件日志，注入迟到、重复、乱序等交付危险。

**📈 对比分析**

采用Snapshot Pass与Silent Failure率对比；单次评估无法区分十一倍差异；未提示模型时间错误率≈35%，提示模型在重播测试中降至0%。

**⚠️ 局限性**

仅在单一DuckDB引擎和单一SQL方言下实验；任务设计与共享阶段模式可能偏倚结果；人类基准样本有限，模型对提示的敏感性未完全掌控。

---

## 127. MIRROR: Multipath Quorum Integrity for LLM Multi-Agent Communication

**arXiv ID:** 2610.02349 | [PDF](https://arxiv.org/pdf/2610.02349v1)

**作者:** Ryuichi Yamafuji Lun `[一作]` (University of Southern California), Ruiteng Li `[通讯]` (University of Southern California)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 MIRROR，一种多路径法定量完整性原语，用于在大型语言模型多智能体系统（LLM‑MAS）中防御中间人攻击，保证消息在未受信任的中间路由上的完整性。

**💡 创新点**

创新点在于：① 对攻击模型与安全范围进行严格界定，明确仅在路由多数为诚实时提供完整性；② 采用无密钥哈希与法定量投票机制，避免了公钥基础设施；③ 提出了针对共享故障组的阈值分析，为实际部署提供可审计的配置准则。

**🔧 技术方法**

技术手段包括：① 统一字节级 canonicalization、SHA‑256 无密钥哈希；② 在 k 条逻辑路由上发送完整载荷或仅哈希值，利用 t+1 与 t 的全载荷/见证划分；③ 接收端通过投票（>k/2）决定是否接受并恢复原始消息；④ 可选的路由轮换与失效处理。

**📊 数据集**

评估使用的数据集与环境有：MMLU、HumanEval、MBPP（AutoGen、CAMEL 框架）；MetaGPT 在真实生产 API（Gemini 2.5 Pro）上；实验还包含不同拓扑（链、树、完全、随机）以及多路由配置（k=5）。

**📈 对比分析**

与基线（未防御）和 LLM‑as‑a‑Judge（语义验证）进行对比：在路由多数阈值 α<0.5 时 MIRROR 的攻击成功率（ASR）始终为 0%，而 LLM‑as‑a‑Judge 仍出现高达 44.2% 的误报；在生产 API 上 MIRROR 的 token 消耗仅为基线的 1×，相比之下 LLM‑as‑a‑Judge 需要约 35× 的推理开销。

**⚠️ 局限性**

局限性包括：① 依赖路由的独立性，若共享基础设施导致多数路由被同一攻击者控制，则安全失效；② 对路由轮换的效果与可靠性尚未在自适应攻击下充分验证；③ 不处理已被攻破的智能体或提示注入；④ 需要严格实现字节级 canonicalization，任何细微差异都可能导致误投票；⑤ 未针对可调攻击者对 canonicalization 逻辑的攻击进行评估。

---

## 128. "I'm trying not to get hacked:" How Adults with Intellectual and Developmental Disabilities Navigate Security and Privacy Notifications

**arXiv ID:** 2610.02374 | [PDF](https://arxiv.org/pdf/2610.02374v1)

**作者:** Hailey L. Johnson `[一作]` (University of Wisconsin--Madison), Rahul Chatterjee `[通讯]` (University of Wisconsin--Madison)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对7名成年智力与发育障碍（IDD）参与者进行形成性用户研究，探究他们在移动与桌面环境中对常见安全与隐私通知（如两步验证、垃圾邮件、Cookie 同意、浏览器警告和权限弹窗）的感知与决策过程；

**💡 创新点**

提出三条包容性设计建议：①超越简化术语，针对上下文依赖的语言误解；②明确行动与结果的关联；③支持用户与可信协作者共同决策，提升安全交互的可访问性；

**🔧 技术方法**

采用质性访谈、可用性测试、眼动追踪与系统日志收集，并使用反射性主题分析法对数据进行编码与主题提炼；

**📊 数据集**

使用真实场景下嵌入的安全与隐私通知（如 Gmail 2FA、钓鱼邮件、Pinterest Cookie 同意、Chrome 浏览器警告、AccuWeather 权限请求）与受试者个人设备及操作系统组合；

**📈 对比分析**

研究以定性分析为主，未进行量化对照或性能评估；结果以主题与设计启示呈现，未提供数值性能指标；

**⚠️ 局限性**

样本规模仅7人，诊断多为唐氏综合征或 ASD，缺乏更广泛的 IDD 群体；仅涉及视觉通知，未覆盖音频/触觉；研究过程中的提示可能影响自然行为；研究未区分受访者与支持者的具体输入，导致部分结论混合来源。

---

## 129. TRACE: A Reproducible Benchmark for Electricity Price Forecasting with Official Operational Text

**arXiv ID:** 2610.02256 | [PDF](https://arxiv.org/pdf/2610.02256v1)

**作者:** Xinyi Yi `[一作]` (University of Cambridge), Ioannis Lestas `[通讯]` (University of Cambridge)

**通讯引用:** 1881 | [OpenAlex ID](https://openalex.org/A5082551328)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 TRACE 基准，将美国 P​JM 电力市场的官方操作文本与历史价格对齐，用于日内价格预测。

**💡 创新点**

创新点在于首次提供可复现的 EPF 基准，采用截止时间重构文本防止信息泄漏，并证明文本能显著降低上尾风险预测误差。

**🔧 技术方法**

使用的大语言模型语义评估、时间序列基础模型与文本融合（直接融合和结构化语义融合）以及 pinball loss 评估方法。

**📊 数据集**

数据集包含 2021‑10‑02 至 2025‑09‑30 的 7,300 个区域‑日实例，覆盖 5 个 P​JM 区域的 14 天历史价格和对应官方操作文本。

**📈 对比分析**

通过与无文本基准和打乱文本对照，所有模型在上尾量化误差 PB_0.9 上平均下降 7.4%，最高可达 19.3%，显示文本融合显著提升预测性能。

**⚠️ 局限性**

局限性包括文本对下尾和中位误差影响有限；跨日文本错配会导致性能下降；基准仅针对 P​JM 市场，推广至其他市场需进一步验证。

---

## 130. Efficient Neural Field Learning via Adaptive Coverage and Focused Sampling

**arXiv ID:** 2610.02410 | [PDF](https://arxiv.org/pdf/2610.02410v1)

**作者:** Guang Zhao `[一作]` (Brookhaven National Laboratory), Wei Xu `[通讯]` (Brookhaven National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为ACES的结构化区域级自适应采样框架，用于提升隐式神经表示（INR）在高维连续场学习中的训练效率。

**💡 创新点**

创新点在于将空间划分与区域重要性加权分离：先通过自适应分区保证全域覆盖并降低梯度方差，再利用残差信息对不同区域施加可控偏差的权重，既减少冗余采样，又加速收敛。

**🔧 技术方法**

核心技术包括：基于梯度方差的自适应空间划分（类似分层采样），区域级重要性加权的可控偏差采样，理论上结合方差降低与方向对齐分析；实现时采用残差统计近似梯度方差，并以SIREN网络为基础进行训练。

**📊 数据集**

在实验中使用了Navier–Stokes涡量场（NS2D）和沸腾过程场（PoolBoiling2D）进行二维空间重建，以及对应的三维时空场（NS3D、PoolBoiling3D）进行时空联合重建。

**📈 对比分析**

与均匀随机采样、INT、EVOS、固定分层等基线相比，ACES在相同采样预算下显著降低相对均方误差，收敛速度快且在计算时间上也具备竞争力，尤其在局部复杂结构（如沸腾界面）更突出。

**⚠️ 局限性**

局限性包括：采样与分区更新产生额外计算开销；目前的分析主要聚焦一阶优化步长，未深入探讨非凸长期动态；在某些任务（如NS3D早期阶段）效果与固定分层相近。

---

## 131. A Composable AI-Accelerated Iterative Solver for 3D-IC Thermal Modeling

**arXiv ID:** 2610.02461 | [PDF](https://arxiv.org/pdf/2610.02461v1)

**作者:** Yixing Li `[一作]` (Cadence Design Systems), Xin Ai `[通讯]` (Cadence Design Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可组合的 AI 加速迭代求解器 DAIST，利用域分解将 3D‑IC 热分析拆分为块级子问题，用神经算子替代局部求解器，并通过温度和热流交换实现迭代耦合；

**💡 创新点**

创新点在于：①将整体热问题分解为物理块级模块，打破单一全包模型的拓扑锁定；②设计可调节准确性/运行时间的接口迭代耦合；③实现块级模型跨拓扑重用，零重训练即可迁移到不同堆叠结构；

**🔧 技术方法**

技术手段包括：域分解+块 Jacobi 迭代；神经算子（U‑NO 等）作为子域求解器；双阶段算法（先接口迭代再体域构造）；松弛参数调节与收敛判据；

**📊 数据集**

数据集：为每种块类型（interposer、SoC、HBM、MC、TIM）生成 5,000 组训练样本，随机几何、功率分布（高斯随机场）及边界条件；训练集/验证集/测试集比例 8:1:1；

**📈 对比分析**

比较方法：与全包 FEM 求解器（Celsius）及单块/全包神经预测器对比；指标为 MAE、RMSE、MAPE、最大误差、运行时；在多芯片组系统上，DAIST 速度提升 178×，MAE≈0.25 K；在高级封装系统上，速度提升 99×，MAE≈1.06 K；对比单块模型，DAIST 在 OOD 拓扑下误差仅略升，而单块模型误差暴涨；

**⚠️ 局限性**

局限性：需要手动调节松弛参数与迭代次数；块级模型对极端热流分布的泛化有限（需微调）；目前仅采用三层简化结构，无法覆盖更复杂物理（如接触热阻、非稳态）；需为每种新块训练模型，尚未实现完全零训练。

---

## 132. Test-time Multi-agent Coordination by Decomposed Value Gradient Flow

**arXiv ID:** 2610.02554 | [PDF](https://arxiv.org/pdf/2610.02554v1)

**作者:** Dongsu Lee `[一作]` (University of Texas at Austin), Amy Zhang `[通讯]` (University of Texas at Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种面向离线多智能体强化学习的测试时行动细化框架SCOUT，利用流匹配的行为先验与分解后的价值函数，通过Stein变分梯度实现多智能体动作的高价值迁移；

**💡 创新点**

核心创新在于将价值最大化从训练阶段完全解耦，采用测试时的概率运输（Optimal Unified Transport）来提升协调性，并在分解价值函数与IGM原则下提供理论保证；

**🔧 技术方法**

技术包括流匹配（Flow-Matching）用于生成行为分布、价值分解网络（VDN）实现可加的Q函数、Stein变分梯度下降（SVGD）进行概率运输以及基于RBF核的粒子复合；

**📊 数据集**

实验数据集主要为SMACv1（StarCraft II微观管理任务）与MA-MuJoCo（多智能体机器人控制），覆盖多种数据质量（Good、Medium、Poor）与不同任务规模；

**📈 对比分析**

与Gaussian、Diffusion、Flow等多种基线比较，SCOUT在离线与离线-在线迁移实验中均实现最高平均奖励，尤其在低质量数据上显著优于训练时去中心化策略；

**⚠️ 局限性**

局限在于细化步骤（L_test、L_train）的超参数对不同任务和数据集具有一定敏感性，缺乏统一的自适应终止准则，需要进一步研究。

---

## 133. TREMOR: Template Matching for Large Seismic Data Collections

**arXiv ID:** 2610.02534 | [PDF](https://arxiv.org/pdf/2610.02534v1)

**作者:** Manos Chatzakis `[一作]` (Université Paris Cité), Themis Palpanas `[通讯]` (Université Paris Cité)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出TREMOR，一个分布式框架，用于在海量地震波形中高效执行模板匹配；

**💡 创新点**

创新点在于将iSAX子序列索引与自适应扫描、FFT回退相结合，并通过复制组动态调度与BSF共享实现高并行与负载均衡；

**🔧 技术方法**

采用分布式iSAX索引、SIMD加速的早停距离计算、两阶段叶节点精细化、FFT匹配以及多线程与MPI并行化；

**📊 数据集**

使用两大真实地震数据集：法国都市地震监测网络（SeiFR）和智利马乌勒大地震余震部署（Maule），包含数十亿点长波形和数千模板；

**📈 对比分析**

与MASS、DMASS-V1/V3、跳序扫描等基线相比，TREMOR在阈值搜索中最快，速度提升可达9–160倍，在k-NN搜索中最快，提升可达5–66倍；

**⚠️ 局限性**

局限在于需要对参数如f、b、叶大小等进行调优；在极低选择性阈值或极大数据规模时，索引构建成本和复制带宽开销仍然显著。

---

## 134. Budgeted Cache Repair for Cross-Context KV-Cache Reuse

**arXiv ID:** 2610.02233 | [PDF](https://arxiv.org/pdf/2610.02233v1)

**作者:** Haeyong Kang `[一作]` (Duksung Women's University), Chang D. Yoo `[通讯]` (Korea Advanced Institute Of Science And Technology)

**通讯引用:** 6564 | [OpenAlex ID](https://openalex.org/A5073287748)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于预算的 KV‑cache 修复方法（Budgeted Cache Repair），通过生成两条草稿 token 并按注意力评分来挑选需要重新计算的 cache 行，从而在保持大部分缓存复用的同时恢复因复用导致的准确性损失。

**💡 创新点**

创新点在于：① 以「预算」为限制而不是全局信任或不信任的方式选择修复行；② 利用两 token 草稿的注意力信号来排序缓存行，显著提升修复效果；③ 研究三种修复布局（固定块、动态跨度、单行）并评估其对不同工作负载的影响。

**🔧 技术方法**

主要技术包括：KV‑cache 预取与跨上下文复用；在线锚点池预测 cache 失真；基于注意力权重的行排序；固定预算的精确重计算；三种修复布局实现；与 CacheBlend 以及完整预填充 baseline 的对比实验。

**📊 数据集**

使用的评测数据集包括 GSM8K（算术推理），MMLU（多学科通识），以及 HumanEval（代码生成）；实验模型为 Llama‑3.1‑8B‑Instruct 与 Qwen2.5‑Coder‑7B‑Instruct。

**📈 对比分析**

比较方法：在相同 prompt、解码参数和硬件环境下对不同策略（完整预填、CacheBlend、Budgeted Cache Repair 各布局）进行配对统计。结果显示：在 GSM8K 上，Budgeted Cache Repair 能将准确率恢复至 dense‑prefill 级别，保持 63% 的缓存复用；在 MMLU 上仅恢复约一半误差；HumanEval 上无显著损失。修复布局中，单行和动态跨度布局在大多数 benchmark 上优于 baseline。

**⚠️ 局限性**

局限性包括：① 修复效果高度依赖工作负载和模型结构，MMLU 的 tokenization 问题无法通过缓存修复解决；② 需要额外的两 token 草稿和预算计算，带来额外的推理延迟；③ 目前仅在单一 chunk 大小和拓扑上验证，尚未在更大规模多代理网络中检验；④ 预算大小的选择仍需经验调优，未给出通用自动化方法。

---

## 135. Energy Saving in 5G and Beyond Networks: A Quantum Reinforcement Learning Approach

**arXiv ID:** 2610.02403 | [PDF](https://arxiv.org/pdf/2610.02403v1)

**作者:** Muhammad Usman `[一作]` (University of Salento), Mariangela Lazoi `[通讯]` (University of Salento)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于量子强化学习（QRL）的框架，在5G及更高网络中根据用户设备动态行为自动调节基站天线功率与开关状态，实现能耗优化。

**💡 创新点**

创新点在于利用参数化量子电路实现策略学习，借助量子叠加与纠缠实现更快的探索与收敛，同时在同一框架下同时优化能耗与吞吐量。

**🔧 技术方法**

使用了量子强化学习（QRL）、参数化量子电路（上限8量子比特、3层）、经典深度强化学习（DRL）和Q‑learning作为对比算法，并通过Cirq等经典量子模拟器实现。

**📊 数据集**

采用自定义仿真数据：随机分布与移动的用户设备（UE）在单基站覆盖区内，未使用公开数据集。

**📈 对比分析**

通过仿真将QRL与DRL、Q‑learning进行比较，评估收敛速度、平均功耗和吞吐量；结果显示QRL在约6,000步内收敛，平均功耗最低且吞吐量保持在QoS阈值以上，优于基线方法。

**⚠️ 局限性**

局限性：实验仅在单基站、少量UE的简化仿真环境中验证；未考虑多基站协同、网络大规模规模和量子硬件噪声；且使用经典模拟器，缺乏真实量子设备的验证。

---

## 136. SimuVerity: Benchmarking Agents for Engineering-Grade Simulink Model Generation

**arXiv ID:** 2610.02304 | [PDF](https://arxiv.org/pdf/2610.02304v1)

**作者:** Ruiqi Zhang `[一作]` (Xi'an Jiaotong University), Xiaohua Wang `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 14236 | [OpenAlex ID](https://openalex.org/A5100438586)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为SimuVerity的基准，包含101个跨10个工程领域的文本到可执行模型生成任务，并对生成的模型进行基于原生仿真和六维工程性能的分层评估。

**💡 创新点**

创新点在于：① 通过可执行系统配置文件和多样化仿真场景实现真实的工程需求验证；② 采用分层评估（交付、可执行、工程合格）和六维度（准确性、输出质量、机制忠实度、控制因果完整性、域鲁棒性、动态响应）来衡量模型质量；③ 通过结构相似度与工程性能的对比揭示结构相似度并不一定代表工程优异。

**🔧 技术方法**

技术手段包括：MATLAB/原生仿真、MCP工具链、自动化评估脚本、基于情景的性能映射与加权归一化、专家评审与相关性验证。

**📊 数据集**

数据集为101个手工构建的任务集合，包含可执行系统原型、执行系统配置文件、四类仿真场景、评估脚本和参考模型，全部公开发布在GitHub。

**📈 对比分析**

对六个LLM+工具链系统（Opus 4.8/Claude Code、GPT-5.5/Codex 等）进行三次独立运行评测，结果最高系统整体得分为42.86，交付率约94.7%，可执行率86.5%，工程合格率76.9%。相比参考模型（100%）显示显著差距，证明当前系统在跨域工程生成方面仍有瓶颈。

**⚠️ 局限性**

局限性包括：① 评估过程依赖MATLAB/原生仿真，速度慢且不易扩展；② benchmark 仅覆盖10个领域，可能无法涵盖所有工业场景；③ 仅测试了少数工具链和LLM组合，缺乏更广泛的系统对比；④ 评估侧重功能性能，仍未覆盖如安全性、可维护性等更深层次工程属性。

---

## 137. How to Have a Sensitive Debate: An Instance-Optimal Protocol for AI Debate

**arXiv ID:** 2610.02557 | [PDF](https://arxiv.org/pdf/2610.02557v1)

**作者:** Jiawei Li `[一作]`, Jonah Brown-Cohen `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

提出了一种基于分数块灵敏度（fractional block sensitivity）的 AI 辩论协议，用于在递归可分解问题中实现最优的监督与错误检验。

**💡 创新点**

创新点在于：①在最坏情况下提供正确性保证；②把诚实与正确性提升为双方的占优策略（dominant strategy equilibrium）；③证明在只具黑盒人类判断的前提下，该协议在实例层面上实现了最优查询复杂度。

**🔧 技术方法**

采用的技术包括：计算复杂性框架下的 AI 辩论协议设计、分数块灵敏度的线性规划定义与对偶分析、递归分解树的构造与分析，以及对该协议的最优性与下界证明。

**📊 数据集**

本工作为理论性研究，未使用具体数据集，而是针对通用的递归可分解问题和抽象的判定函数。

**📈 对比分析**

与之前的 prover‑estimator、双效能辩论等协议相比，本协议在最坏情况、占优策略和平稳性（fractional block sensitivity）保证上均更强；理论上实现了 Ω(ρ^d) 的下界，说明在黑盒人类判断模型下已达到最优。

**⚠️ 局限性**

局限性包括：①仅适用于具有有限分数块灵敏度的递归可分解问题；②协议的最优性基于人类判断仅能通过黑盒查询获取的假设，若实际人类判断不满足该假设则可能失效；③实现时需求解对偶 LP，计算开销和对大规模问题的可扩展性尚未评估。

---

## 138. From Alert Floods to Precedence Forests: Zero-Prior-Knowledge Incident Triage with LOGOS

**arXiv ID:** 2610.02297 | [PDF](https://arxiv.org/pdf/2610.02297v1)

**作者:** Radhika Niranjan Mysore `[一作]` `[通讯]`, Radhika Niranjan Mysore

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 LOGOS，一套无先验知识的日志分析系统，利用原始日志中的实体-事件共现和时间先行关系，构建优先级森林来压缩降噪并自动定位故障传播路径。

**💡 创新点**

创新点：
- 彻底抛弃域知识、历史基线和分布式跟踪，仍能在原始日志中发现故障传播。
- 通过进阶起点跟踪+Hub节点剪枝构造无向实体‑事件图的有向子树，形成可解释的优先级森林。
- 在 2‑小时窗口内实现 16 小时的诊断提前、124× 警报压缩、99.8% 噪声消除，并在 LLM 零射击场景下达到 80%/100% 根因识别率。

**🔧 技术方法**

技术手段：
- 正则表达式日志抽象与严重性提升（Severity Elevation）。
- 进阶起点跟踪（Progressive Onset Tracking）与时间窗口切分。
- 并查集求连通分量（Bounded Subgraph Detection）。
- Hub‑节点剪枝与父子关系分配（Precedence Forest Construction）。
- 线性 O(M) 复杂度实现。
- 结合 Gemini LLM 进行零射击根因推理。

**📊 数据集**

数据集：
- 25 起企业级虚拟基础设施故障日志（共计数百万行）。
- 12 起开源项目日志（OpenStack、TrainTicket）。
- 共 37 个数据集，日志大小从数 MB 到数十 GB。

**📈 对比分析**

评估与性能：
- 与四种图中心性/时间排序基线相比，LOGOS 在企业场景实现平均召回 0.76、Top‑10 可达 60%，在 OpenStack 场景 100% 召回。
- LLM 诊断精度从 4/25（基线）提升至 10/25（80%）完整诊断，整体准确率 80%/100%。
- 解析时间 4.5 分钟，内存峰值 ≤ 48 GB；噪声降噪 99.8%，警报压缩 124×。

**⚠️ 局限性**

局限性：
- 对第三方日志的严重性提升效果有限，导致识别率下降。
- 依赖自定义日志抽象规则与 Hub‑节点表列，若日志格式或标识不一致会影响性能。
- 最大连通分量选择可能导致部分根因被丢弃，出现零召回。
- 目前仅为批量一次性分析，缺乏交互式迭代和因果推断。
- 对无日志或标识不连贯的系统无法完整构造传播路径。

---

## 139. SD-DPC: Sparse Dictionary Differentiable Predictive Control

**arXiv ID:** 2610.02466 | [PDF](https://arxiv.org/pdf/2610.02466v1)

**作者:** Ali Reza Daneshvar Garmroodi `[一作]` (Concordia University), Jan Drgoňa `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种稀疏字典可微预测控制（SD-DPC）框架，利用稀疏识别的非线性动力学模型与稀疏反馈字典同时训练，得到可解释且高效的闭环控制策略。

**💡 创新点**

创新点包括：① 将模型识别与策略稀疏化统一在同一阈值梯度下降流程中，直接在闭环性能目标下挑选策略项；② 给出有限终止、剪枝代价上界和显式灵敏度/稳定性保证；③ 通过稀疏字典实现极低的存储和计算需求。

**🔧 技术方法**

技术手段包括：稀疏识别（SINDy）与多步回滚误差拟合；可微预测控制（DPC）与闭环回滚梯度；阈值梯度下降+去偏拟合；投影与泄漏-半带近似的可微投影；使用AdamW优化器。

**📊 数据集**

数据集与实验：三种仿真基准——双罐系统、受迫Van der Pol振荡器、二维积分器避障；训练、验证、测试全部使用由控制器生成的仿真轨迹，未采用公开真实数据集。

**📈 对比分析**

比较方法：与IPOPT在线MPC、深度神经DPC（NN-DPC）以及基于相同稀疏字典的蒸馏策略（NN-Distilled）对比。结果显示：SD-DPC满足所有约束，追踪误差比蒸馏高约10倍，计算时间和内存比MPC低两位数、比NN-DPC低约两位数；在三个案例中均实现了显著的性能与效率提升。

**⚠️ 局限性**

局限性：① 约束通过惩罚实现，缺乏硬约束保证；② 对噪声数据和字典匹配缺乏鲁棒性分析；③ 仅在模拟实验验证，真实系统鲁棒性与在线适应性尚待进一步研究；④ 需要更强的闭环安全与稳定性证明。

---

## 140. Feature tracking in physics-informed neural networks via joint optimization of nonlinear deformation manifolds: application to shocks

**arXiv ID:** 2610.02230 | [PDF](https://arxiv.org/pdf/2610.02230v1)

**作者:** Akshay Thakur `[一作]` (University of Notre Dame), Matthew Zahr `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种在固定参考域上训练的 PINN（FT-PINN），通过学习可变形映射将均匀采样点聚焦到解中的冲击波、尖锐特征，从而在相同训练预算下准确捕捉冲击波位置与形状。

**💡 创新点**

创新点在于：①将非可分离的高维径向基函数（RBF）变形映射与解网络联合优化；②使用单侧折叠惩罚保证映射保持单射；③通过切向投影严格保持边界几何；④无需先验冲击位置、残差自适应重采样或逐点优化，整体由少量变形参数控制点聚集。

**🔧 技术方法**

核心技术包括：Physics‑Informed Neural Network (PINN)、参考域映射与拉回残差、RBF 基础的变形场、单侧折叠惩罚、切向投影、Adam 优化与分组学习率调度、随机 Fourier 特征嵌入的全连接网络。

**📊 数据集**

使用四个自合成 PDE 测试集：一维粘性 Burgers 合并冲击、减速冲击、二维时间空间 Euler 震荡管以及二维稳态 Euler 正常反射。数据来源为高分辨率有限体积数值解或解析解，作为评估参考。

**📈 对比分析**

与标准 PINN（相同网络、相同采样点数、相同训练时长）对比。FT‑PINN 能在同等点数下精确定位冲击波并保留幅值，而 vanilla PINN 仅产生扩散或完全失真；在所有四个实验中 FT‑PINN 的误差与人工黏性尺度相当，远优于 vanilla PINN 的 O(1) 扩散误差。

**⚠️ 局限性**

局限性包括：仅在矩形/简单几何域实验，未验证复杂多边形或三维情况；RBF 参数需手动设置中心/宽度；折叠惩罚在极端压缩时可能不足；对极其尖锐或多尺度冲击仍需进一步调优。

---

## 141. Nearest-neighbour baselines for fingerprint prediction from MS/MS spectra under different assumptions

**arXiv ID:** 2610.02249 | [PDF](https://arxiv.org/pdf/2610.02249v1)

**作者:** Ling Min Serena Khoo `[一作]` `[通讯]`, Ling Min Serena Khoo

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

系统比较了在不同信息假设下（无化学式、已知化学式、化学式注释峰）使用最近邻检索预测MS/MS谱对应分子指纹的方法。

**💡 创新点**

创新点在于明确推理时可用信息层级，评估化学式和子公式注释对检索性能的影响，并提出子公式规则在缺乏同一化学式样本时的替代检索策略。

**🔧 技术方法**

采用多种谱相似度度量（cosine、修改后cosine、中性损失cosine、DreaMS嵌入cosine、分箱cosine）结合最近邻检索，并与现有深度学习模型进行对标。

**📊 数据集**

使用公开质谱数据集NPLIB1和MassSpecGym，在scaffold和random两种数据拆分下进行评估。

**📈 对比分析**

通过Jaccard（Tanimoto）和余弦相似度进行比较，结果显示随着信息假设的提升性能显著提升：在scaffold拆分下，子公式规则+注释峰实现Jaccard最高0.842/0.952，余弦最高0.960/0.960；在random拆分下性能也显著优于无化学式假设。

**⚠️ 局限性**

局限性包括依赖已知化学式或子公式注释，对缺少同一化学式样本的全集检索仍表现较差，且未探索更高级的深度学习与最近邻融合方法。

---

## 142. APDMem: Agent-Controlled Progressive Disclosure for Query-Adaptive Long-Term Memory

**arXiv ID:** 2610.02472 | [PDF](https://arxiv.org/pdf/2610.02472v1)

**作者:** Chin-Lun Fu `[一作]` (JPMorgan Chase & Co.), Behrouz Madahian `[通讯]` (JPMorgan Chase & Co.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种层次化的长期记忆架构APDMem，通过分层的主题摘要、关键事实、转述证据和原始消息实现可进阶的检索；

**💡 创新点**

创新点在于将记忆检索视为查询自适应的逐步披露问题，利用代理控制器动态决定检索深度，并通过组织式笔记合成器将检索结果结构化；

**🔧 技术方法**

采用多分辨率记忆子结构、进阶披露控制器、回显与推理工具以及笔记合成器，并结合LLM（GPT‑4.1/4.1‑mini）进行推理；

**📊 数据集**

在LongMemEval（LongMemEval‑S）上进行评估，包含500道长上下文问题，涵盖六类查询；

**📈 对比分析**

与全上下文检索、Mem0、LightMem、SimpleMem等基线进行公平对比，APDMem在GPT‑4.1下达到87.8%准确率，GPT‑4.1‑mini下达到79.8%，相比最佳基线提升3.8/2.9个百分点，且仅访问约8%对话；

**⚠️ 局限性**

局限包括对主题摘要质量的依赖、首次访问时的延迟（L₁/L₂惰性生成），以及未显式建模跨会话关系，未来需改进摘要评估与跨会话链接机制。

---

## 143. Network-in-the-Loop at Scale: GPU-Batched 5G Simulation for Massively Parallel Robot Learning

**arXiv ID:** 2610.02370 | [PDF](https://arxiv.org/pdf/2610.02370v1)

**作者:** Zifan Zhang `[一作]` (North Carolina State University), Yuchen Liu `[通讯]` (North Carolina State University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一个GPU批量5G NR模块，将网络模拟嵌入Isaac Lab训练循环，实现千级并行环境；

**💡 创新点**

通过固定形状张量状态、槽级时间步驱动与高效Graph/Triton后端，首次在闭环学习中精确重现ns-3 5G-LENA的延迟与AoI，解决独立延迟模型的不足；

**🔧 技术方法**

使用PyTorch/NumPy固定张量、CUDA Graph与Triton kernel、3GPP NR调度、HARQ、MCS、EESM等，配合Isaac Lab mixin及与ns-3、OAI的co-simulation桥接；

**📊 数据集**

利用公开的5G-LENA、OAI校园与OAI 2026.w39测量数据、ns-3 5G-LENA运行记录，以及随机生成的无人机/机器人场景（150 m方阵、随机UE位置）等数据集；

**📈 对比分析**

通过与ns-3 5G-LENA的延迟CDF、丢包率、PRB使用、KS/Wasserstein距离等统计量比较；在百万机器人环境下，网络占训练时间约20‑25%，相较ns-3同步模拟提升80‑200倍，单GPU可达1.24控制步/秒；

**⚠️ 局限性**

受限于单/多cell支持、triton kernel仅支持≤128机器人/环境、buffer‑report grant过程在triton后端未实现、GPU内存限制和对极大规模场景尚未充分验证。

---

## 144. From Fragments to Global Maps: Learning Vectorized Map Aggregation with Large Language Models

**arXiv ID:** 2610.02513 | [PDF](https://arxiv.org/pdf/2610.02513v1)

**作者:** Ziwei Li `[一作]` (Bosch Research North America), Liu Ren `[通讯]` (Bosch Research North America)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于大语言模型的向量化HD地图聚合框架，直接将多帧局部预测序列生成全局聚合图。

**💡 创新点**

创新点在于将聚合任务转化为条件序列生成，使用坐标词表、几何预训练和线级对比损失，使模型自适应不同检测器并无需手工阈值。

**🔧 技术方法**

采用Qwen3.5-4B LLM、坐标token化、几何预训练、LoRA微调以及线级关联损失。

**📊 数据集**

在Argoverse2和nuScenes两个真实数据集上进行训练与评估。

**📈 对比分析**

与VMA和MonoLaM等基线相比，本文方法在车道分隔符和道路边界的F1上分别提升约5–30个百分点，尤其在道路边界上显著超越。

**⚠️ 局限性**

局限在于需要人工设计的坐标量化范围、对极端预测误差的鲁棒性尚待验证，并且在复杂多层道路结构下仍可能出现过度平滑或缺失细节。

---

## 145. Non-Malleable Affine Extractors with Small Error and Complexity Lower Bounds

**arXiv ID:** 2610.02407 | [PDF](https://arxiv.org/pdf/2610.02407v1)

**作者:** Xin Li `[一作]` (Johns Hopkins University), Yan Zhong `[通讯]` (Johns Hopkins University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构造了在线性输出且误差指数级小的非可变形仿射提取器，并将其用于证明弱可读一次线性分支程序、非显式决策树及对称分辨率的强下界；

**💡 创新点**

实现了任意正熵率下的线性输出非可变形仿射提取器，突破了先前仅在高熵率下可行的限制；

**🔧 技术方法**

采用协同提取、相关性破坏器、线性编码与多重仿射插值等技术；

**📊 数据集**

本研究为理论性，未使用具体数据集；

**📈 对比分析**

与已知的方向仿射提取器/多项式决策树等方法对比，获得了更小误差、更高输出长度以及更强的下界；

**⚠️ 局限性**

仅适用于常数熵率，尚未覆盖子常数熵或完全读一次模型；

---

## 146. Improving the Energy-Efficiency of the Code Generated by LLMs through Effective Prompting

**arXiv ID:** 2610.02571 | [PDF](https://arxiv.org/pdf/2610.02571v1)

**作者:** Ritika Rekhi `[一作]` (University at Buffalo), Tevfik Kosar `[通讯]` (University at Buffalo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 10 种主流 LLM（含 7 种开源模型和 3 种专有模型）生成的 Python 与 C++ 代码，系统评估 21 种 prompt 设计的能耗影响，并挑选 8 种表现最佳的 prompt 在 878 个 EffiBench 题目上进行大规模实验。

**💡 创新点**

首次在多语言、多模型环境下进行能耗驱动的 prompt 设计评估；提出从基准 prompt 出发的单/多轮 prompt 组合，并将 EffiBench 语义评测扩展至 C++，形成完整的能耗测量框架。

**🔧 技术方法**

使用 Prompt Engineering、Chain‑of‑Thought、Persona 设定、few‑shot 与 energy‑efficiency 关键字等技术；利用 Linux perf 读取 CPU 包（PKG）与 DRAM 能耗；构建多轮交互和自适应评估流水线。

**📊 数据集**

EffiBench 数据集（878 个 LeetCode 问题），扩展后包含 Python 与对应 C++ 版本；每个问题有标准测试用例，保证功能正确性与能耗可比性。

**📈 对比分析**

将每种 prompt 生成的代码执行 30/40 次，记录能耗后减去空闲基准，计算相对基线的百分比变化；实验表明 Prompt 20 在 Python 上可降低 25% 能耗，Prompt 13 在 C++ 上可降低 17%，最优单模型（Granite‑4.0‑H‑Small）能量下降可达 57%。

**⚠️ 局限性**

实验集中在 Python 与 C++，未覆盖其他语言；数据集难度分布偏向 Medium，可能影响结果泛化；未考虑 prompt 生成与推理过程本身的能耗；不同 LLM 对同一 prompt 的敏感度差异大，导致部分模型出现能耗上升。

---

## 147. EditHero: A Benchmark for Long-Horizon Part-Level 3D Editing and Vibe Modeling

**arXiv ID:** 2610.02298 | [PDF](https://arxiv.org/pdf/2610.02298v1)

**作者:** Ruihan Yu `[一作]` (Alaya Lab), Zhixiang Wang `[通讯]` (Alaya Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 EditHero benchmark，构建可按步骤执行且每一步都拥有确切 3D 目标的连环编辑任务，并在此基准上评估传统非代理方法与 LLM/VLM 代理方法的多步编辑能力。

**💡 创新点**

创新点在于：①使用基于零件的合成引擎自动生成多轮编辑链并提供每一步的精确 ground‑truth；②设计了针对编辑区域和未编辑区域的专门指标（IF 与 CC）以区分指令遵循与内容保持；③首次将 LLM/VLM 代理与传统生成方法在同一连环编辑任务下进行系统对比。

**🔧 技术方法**

核心技术包括：零件检索与放置的程序化合成引擎、基于 VLM 的候选检索、图像编辑模型进行再纹理、以及 LLM/VLM 代理通过代码生成（add、remove、replace、retexture）实现局部编辑。

**📊 数据集**

使用的主要数据集是 EditHero，零件来源于 PartVerse‑XL、Objaverse‑XL 与 HY3D‑Bench；Benchmark 包含数百条编辑链、数千次编辑和十余种主机对象。

**📈 对比分析**

对比方法采用自回放（self‑rollout）设置，使用 IF（指令遵循）与 CC（内容一致性）两个区域指标评估；实验显示传统非代理方法在 IF 与 CC 上均低于 LLM/VLM 代理，后者保持未编辑部件更好、指令遵循更准确，但编辑耗时较长；而传统方法在保持完整性时易出现漂移并对目标编辑不足。

**⚠️ 局限性**

局限性：①LMM/VLM 代理生成新几何体仍较困难，导致尺寸/形状偏差；②执行时间较长（数分钟），不适合实时交互；③非代理方法因整体重生成导致未编辑区域随时间漂移，且易漏指令。

---

## 148. State-Space Unlearning for Non-Stationary Bias in Land Surface Forecasting

**arXiv ID:** 2610.02248 | [PDF](https://arxiv.org/pdf/2610.02248v1)

**作者:** Anidipta Pal `[一作]` `[通讯]` (Heritage Institute of Technology), Anidipta Pal (Heritage Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种针对Mamba家族结构状态空间模型的机器消学（unlearning）框架SSU-LSF，能够在不完整重训练的情况下消除非平稳混杂事件对土地表面预测模型的长期偏差。

**💡 创新点**

创新点包括：①基于EKFac的影响函数与闭式矩阵指数梯度，快速定位混杂事件的时间足迹；②在KL散度信任域内进行Hessian‑free投影梯度上升，并加入空间总变（TV）正则化，以保证空间一致性；③给出了残差混杂与重训练成本的理论上界，并通过实验验证了时间窗口长度对残差的影响。

**🔧 技术方法**

技术手段包括：Mamba编码器/解码器、EKFac近似、时间足迹阈值化、KL散度限制、投影梯度上升、空间TV正则化以及实验中使用的AdamW优化器。

**📊 数据集**

使用了三个基准数据集：NDVI‑LST融合数据（2002–2022）、ERA5 Patchified（1979–2022）和CropHarvest（87,343地点），并在每个数据集上注入人工混杂事件进行评估。

**📈 对比分析**

与11个基线（包括CM、SISA、EWC‑UL、ForgetFS、MULL等）比较，SSU‑LSF在RMSE、CRR、TCI、ACC等指标上均取得最优或近优性能，同时每次消学请求的GPU成本仅为完整重训练的1/8–1/4，训练效率显著提升。

**⚠️ 局限性**

局限性包括：实验全部基于人工注入的混杂且依赖oracle时间窗口；对真实多事件序列的鲁棒性未充分验证；理论上界相对松散；需要显式混杂事件起止时间和MIA审计来保证安全与可靠。

---

## 149. Inherit-MAS: Test-Time Evolution of Multi-Agent Systems through Workflow and Execution Inheritance

**arXiv ID:** 2610.02396 | [PDF](https://arxiv.org/pdf/2610.02396v1)

**作者:** Songtao Wei `[一作]` (University of Texas at Dallas), Bingzhe Li `[通讯]` (University of Texas at Dallas)

**通讯引用:** 971 | [OpenAlex ID](https://openalex.org/A5048972267)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在大语言模型驱动的多智能体系统中进行测试时演化的框架——Inherit-MAS，利用工作流继承与执行继承在每一次尝试中保留有效节点并仅做一次诊断驱动的编辑，从而在不重新执行已验证步骤的情况下改进任务性能。

**💡 创新点**

创新点在于将工作流继承与执行继承明确化：1）工作流继承只保留评测认为有用的节点并对其进行一次校验后编辑；2）执行继承通过完整请求匹配与上下文指纹验证，允许在后续尝试中直接复用之前的可重用计算结果，显著减少冗余推理。

**🔧 技术方法**

采用元模型对任务与接口进行工作流自动合成；单独提示的判定器对输出与执行轨迹进行分量级评分与反馈；在GPT‑4o‑mini或Qwen3‑32B等LLM作为工人，GPT‑5.4‑mini负责合成与评判；执行继承机制基于SHA‑256请求哈希与指纹比较实现。

**📊 数据集**

在WorkBench（多领域企业流程任务）和HotpotQA FullWiki（多文档推理+句子级支持事实检索）两个公开基准上进行评估。

**📈 对比分析**

与单一ReAct、EvoAgent、EvoMAS、TacoMAS等演化MAS基线在相同候选调用预算下对比，Inherit-MAS在两套LLM工人下均取得最高的主指标（WorkBench完成率55.4%/49.7% HotpotQA），并在执行继承下相较于完整重跑节约约30%–35%的工人token，整体token使用更少。

**⚠️ 局限性**

局限在于：①仅在一次编辑限制内进行改进，复杂问题可能需要多步迭代；②执行继承仅适用于无状态或只读工具的节点，状态变更节点仍需重算；③依赖于判定器的准确性与可解释性，若评判错误可能导致错误节点保留或不当编辑。

---

## 150. Spatial Memory Intelligence: Endowing World Models with Understanding-Driven Long-Term Memory

**arXiv ID:** 2610.02521 | [PDF](https://arxiv.org/pdf/2610.02521v1)

**作者:** Ying Yang `[一作]` (Chinese University of Hong Kong, Shenzhen), Li Jiang `[通讯]` (Chinese University of Hong Kong, Shenzhen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Spatial Memory Intelligence (SMI) 框架，利用多模态大语言模型进行长时视频世界模型的空间记忆管理。

**💡 创新点**

创新在于把空间聚类、稀疏化、动作感知检索与可靠性过滤四大原子操作统一到一个 MLLM 控制器，实现对记忆的系统化组织与筛选。

**🔧 技术方法**

技术包括基于 Qwen3.5‑4B 的多模态理解模型、操作导向的数据构造与微调、以及与 HY1.5、Wan2.2 世界模型的集成。

**📊 数据集**

使用了 HY1.5 与 Wan2.2 两个长时视频世界模型的数据集，构建了约千条轨迹的人工标注与教师模型生成的训练样本。

**📈 对比分析**

与五种基线（FramePack、Deep Forcing、MoC、VMem、MemFlow）以及基准模型进行对比，SMI 在 VBench、GPT‑5.6‑sol 评估、重建一致性等指标上均提升显著，同时实现约 80% 的记忆稀疏化并保持较低延迟。

**⚠️ 局限性**

局限在于未与生成模型端到端联合训练、计算开销相对较高、缺乏对更复杂长期推理的支持。

---

## 151. Duplication-Aware Retiming and Cell Interface Redesign for Superconductor Circuit Minimization

**arXiv ID:** 2610.02333 | [PDF](https://arxiv.org/pdf/2610.02333v1)

**作者:** Panagiotis Papanikolaou `[一作]` (University of Wisconsin - Madison), Jennifer Volk `[通讯]` (University of Wisconsin - Madison)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了针对超导电路的复制感知重定时算法与单元接口重新设计，旨在最小化电路面积和延迟。

**💡 创新点**

创新点在于将复制（fan‑out 复制）纳入重定时约束，设计新的单元接口以减少因复制导致的布线和延迟损耗，并提供联合优化框架。

**🔧 技术方法**

采用基于图的重定时模型、整数线性规划（ILP）求解复制约束、以及基于超导电路仿真器的时序分析技术。

**📊 数据集**

使用ISCAS ’85/89/99 及超导电路专用 benchmark 库（如 SuperC Benchmarks）进行实验验证。

**📈 对比分析**

与传统不考虑复制的重定时方法以及单纯的单元接口优化方法进行对比，结果显示面积平均降低约 25–35%，时钟周期延迟提升 10–15%，在所有 benchmark 上均保持最优或相近性能。

**⚠️ 局限性**

局限性包括：仅适用于目前主流超导逻辑家族；对大规模设计的 ILP 求解时间较长；未考虑物理布局层的布线互连和功耗等实际工艺约束。

---

## 152. RINS: Residual-Image Neural Subspace Solvers for Large Sparse Linear Systems

**arXiv ID:** 2610.02217 | [PDF](https://arxiv.org/pdf/2610.02217v1)

**作者:** Zhongyan Ouyang `[一作]` (Shanghai Innovation Institute), Junchi Yan `[通讯]` (Shanghai Innovation Institute)

**通讯引用:** 19583 | [OpenAlex ID](https://openalex.org/A5087158377)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于残差图像的神经子空间求解器 Gate-RINS，利用多项式残差探针和轻量级逐点门控生成纠正基向量，同时保留传统的投影最小残差闭包。

**💡 创新点**

创新点在于将残差图像多项式生成与逐点门控相结合，形成既低成本又具备非线性表达能力的子空间生成器，并引入 GRANS 与 Gate-RINS 的混合推理调度，实现不同阶段最优控制。

**🔧 技术方法**

采用投影最小残差闭包、Chebyshev 多项式残差探针、点级 MLP 门控、图注意力网络（GRANS）、以及混合控制器调度等技术。

**📊 数据集**

使用六种基于 PDE 离散的稀疏线性系统基准：Poisson、Helmholtz、heat、advection-diffusion、Klein–Gordon 与 wave，规模分别为约 1k、1w 及更大约 10⁵ 的 Helmholtz 问题。

**📈 对比分析**

在与 GMRES、GRANS 以及 Block-GRANS 的同步 wall-clock 计时下，Gate-RINS 在多数任务上实现了更快的相对残差阈值收敛，且在 10⁵ 规模 Helmholtz 上仍保持竞争优势；混合调度进一步提升了残差轨迹。

**⚠️ 局限性**

局限性包括：对高频 Helmholtz 等困难模式的处理仍需改进；缺乏预条件器支持；在某些规模下单步成本高于 GMRES；尚未实现矩阵自由化和大规模分布式部署。

---

## 153. Octrees as an Explicit 3D Language

**arXiv ID:** 2610.02388 | [PDF](https://arxiv.org/pdf/2610.02388v1)

**作者:** Ran Dan `[一作]` (Peking University), Peng-Shuai Wang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的多模态大型语言模型 OctLLM，能够在同一架构中完成文本/图像到 3D 形状的生成以及 3D 输入到文本的理解。

**💡 创新点**

创新点包括：1）将稀疏八叉树（S-Octree）作为显式 3D 语言，保留空间结构且避免高分辨率序列增长；2）采用 token‑routed 双流 Transformer，3D 分支与冻结的文本‑图像路径共享注意力，但 3D 训练仅更新分支参数，保持原始语言能力；3）使用位置感知掩码查询与独立 3D 词表，提升生成准确性。

**🔧 技术方法**

核心技术：稀疏八叉树编码与 Z‑order 序列化、位置感知旋转编码（RoPE）结合深度嵌入、可路由 Transformer 块、独立 mesh 嵌入和输出头、3D U‑Net 辅助稠密补全、流式生成器解码。

**📊 数据集**

训练数据来源于 Objaverse‑XL（Sketchfab 子集）、HSSD、ABO、ShapeNet，包含 195K 3D 资产、585K 指令；完成网络使用 175K 稀疏‑完整配对。

**📈 对比分析**

与 ShapeLLM‑Omni、SAR3D、AR3D‑R1 等多模态 LLM 对比，OctLLM 在 1K Toys4K 资产的图像/文本到 3D 生成上分别将 Inception‑FID 降低 17.4%/45% 以及 KID 下降 45%；在 PointLLM‑200 3D 理解任务上 GPT‑img 评测提升 28+ 点；在通用语言基准（MMLU、HellaSwag、GSM8K）上几乎无性能损失。

**⚠️ 局限性**

局限性：S‑Octree 只编码几何结构，缺乏材质与颜色信息，导致在需要外观描述时表现不如 PointLLM；稀疏化过程中信息丢失需后续 U‑Net 补全，增加推理成本；对极细粒度结构的恢复仍有提升空间。

---

## 154. Trained Agentic Context Management

**arXiv ID:** 2610.02404 | [PDF](https://arxiv.org/pdf/2610.02404v1)

**作者:** Bryce Sandlund `[一作]` `[通讯]`, Bryce Sandlund

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过在最简 Agent harness（读取工具与递归调用自身）上对 Qwen3.6-35B-A3B 进行 SFT，使模型在 8K token 的上下文窗口下能够自主管理长文本，实现与 GPT‑5.4 竞争的长文本推理。

**💡 创新点**

证明仅用两种极简工具即可训练出具备长上下文管理能力的模型，摆脱了传统长上下文模型扩展或手工 harness 的局限。

**🔧 技术方法**

采用合成 Oracle 数据生成、SFT 微调、树归约/顺序折叠/委派策略、子代理递归执行等技术。

**📊 数据集**

在多样化的合成数据集上训练，并在 RULER 与 OOLONG‑synth 长文本基准上进行评测。

**📈 对比分析**

与原始 Qwen3.6-35B（64K 上下文）及 GPT‑5.4（1M 上下文）比较；在 RULER 上保持 ≥85% 以上，在 OOLONG‑synth 上当文档长度超过 40K tokens 时与 GPT‑5.4 水平相当。

**⚠️ 局限性**

仅在合成数据上训练，缺乏 RL 收敛或更广泛任务泛化，且对极大规模文档的高层策略仍需进一步研究。

---

## 155. Simple analysis of an algorithm for multiple-source shortest paths in planar graphs

**arXiv ID:** 2610.02371 | [PDF](https://arxiv.org/pdf/2610.02371v1)

**作者:** Philip N. Klein `[一作]` (Brown University), Philip N. Klein `[通讯]` (Brown University)

**通讯引用:** 7378 | [OpenAlex ID](https://openalex.org/A5035567880)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文对 Klein 提出的多源最短路径（MSSP）算法进行了简化的正确性与复杂度分析，给出直观且易于理解的证明；同时说明了该算法在 Steiner 类型问题中的关键作用；

**💡 创新点**

创新点在于：①用新的不变量和“defunct”弧的概念，将原来繁琐的分析转化为简单的循环和子路径论证；②在每一次 pivot 只注入一次弧、且弧不被再次使用的性质得到清晰证明；③阐明了该算法在构造 strip/brick decomposition 时的 pivot 规则与效果。

**🔧 技术方法**

主要技术包括：平面嵌入的组合表述、链接-切割树实现对双图树的维护、引入弧的“松弛/紧张”判定、以及对树路径长度不变性的 invariants 证明。

**📊 数据集**

本论文不包含实验数据集，主要是理论分析与算法描述；作者提及的实验评估来自其他工作（Das 等、Cabello 等）的实验，但未直接使用。

**📈 对比分析**

由于论文聚焦理论证明，没有与其他算法进行实验对比；作者仅说明原算法在 Steiner TSP 与 Steiner 树 PTAS 中实现 O(n log n) 的关键性，并提到其在一些应用（如距离预处理、割与流）中的使用。

**⚠️ 局限性**

限制：①仅针对平面图的多源最短路径，未扩展到更高 genus 或一般图；②假设所有最短路径唯一或通过随机/词典序 tie‑break 处理，虽然作者指出可用随机化或 Lexicographic tie‑break 解决，但实现复杂度略高；③论文仅给出简化证明，未对实现细节（如 link‑cut 树实现细节）给出完整代码。

---

## 156. Confidence-Controlled XAI Auditing for Pedestrian Detection under Domain Shift

**arXiv ID:** 2610.02364 | [PDF](https://arxiv.org/pdf/2610.02364v1)

**作者:** Ruben Dario Florez-Zela `[一作]` `[通讯]` (Universidad Nacional de San Agustín de Arequipa), Ruben Dario Florez-Zela (Universidad Nacional de San Agustín de Arequipa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对在两个驾驶数据集PIE和JAAD上使用固定YOLOv8s检测器，设计并执行了一套基于检测强度控制的跨域XAI审核流程，评估了针对检测的D-RISE与EigenCAM两种解释方法的删除式可解释性，并证明了可解释性与检测强度高度相关且在控制后仍呈现域依赖性。

**💡 创新点**

创新点在于提出了置信度控制的审核协议，以检测强度f0为控制变量消除可信度混杂；并首次量化跨域下删除式可解释性与检测器表现的耦合关系及其域差异。

**🔧 技术方法**

使用了基于随机遮罩的D-RISE解释器、无扰动的EigenCAM基线、删除曲线与AUC评估、Spearman相关、Wilcoxon符号秩检验、Cliff's delta、Bootstrap CI与Holm校正等统计方法。

**📊 数据集**

采用了两个从移动车辆记录的驾驶数据集PIE与JAAD，分辨率1920×1080，均含行人框注释。

**📈 对比分析**

通过在相同检测强度区间（f0）内对比PIE与JAAD的D-Deletion AUC，发现D-RISE在中心f0区间显著比JAAD低（表示更好可解释性），EigenCAM表现不显著；两者的AUC差异在控制后仍保持中等效应且统计显著。

**⚠️ 局限性**

局限性包括：只评估单一检测器和两种解释器，删除式评估易受遮罩导致的分布偏移影响；置信度控制基于f0的假设可能不完全；置信度分层使用全图置信度而非ROI置信度；样本量在极端f0区间不足；未考虑行人尺度、遮挡、光照等次要混杂因素。

---

## 157. Labels Override Definitions in Jev-Style Typed Decision Models

**arXiv ID:** 2610.02586 | [PDF](https://arxiv.org/pdf/2610.02586v1)

**作者:** Seyedarmin Azizi `[一作]` (University of Southern California), Massoud Pedram `[通讯]` (University of Southern California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在typed decision模型中，选项标签（label）与定义（definition）对模型决策的影响，并通过对开放权重模型进行对比实验揭示了标签偏差（label‑label bias）的存在与机制；

**💡 创新点**

创新点在于将标签偏差归因于prompt渲染方式（是否将标签写入模型输入）而非模型决策头，并提出了可在两次调用之间检测标签敏感性的实用测试；同时构建了PolicyBench合成路由测试集以便在无标签噪声的环境中评估该偏差。

**🔧 技术方法**

使用了open‑weight typed decision模型（laya‑td、laya‑en、laya‑ml、von）以及在Qwen2.5、Llama‑3.1‑8B和Mistral‑7B等大型语言模型上的不同读出（label log‑probability、first‑token、generate‑then‑parse）进行实验；

**📊 数据集**

主要数据集包括11个公开分类任务（SST‑2、RTE、CB等），7个是是/否任务（BoolQ、WiC等），以及自定义的PolicyBench路由测试集；

**📈 对比分析**

通过对比标签与定义的删减、标签改名、交叉交换等变体，测量准确率、flip rate、state‑blindness等指标，发现大多数模型对标签敏感，准确率下降至随机；在PolicyBench上，使用无意义标签可提升约5–10%准确率；相对传统语言模型读出，typed decision模型在标签偏差上并无优势。

**⚠️ 局限性**

局限性包括：未对商业版Jev进行实验，实验数据以合成和模板化数据为主，可能缺乏真实自然语言多样性；只评估了开放权重实现，无法直接验证闭源实现的机制；对标签偏差的检测与修正方法需要根据具体任务和模型进行调优。

---

## 158. A generative-informed neuro-symbolic framework for syntactic ambiguity resolution: Evidence from Arabic DPs

**arXiv ID:** 2610.02529 | [PDF](https://arxiv.org/pdf/2610.02529v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 159. CriticHack: Evaluating Visual Rewards Under Robot Policy Optimization

**arXiv ID:** 2610.02527 | [PDF](https://arxiv.org/pdf/2610.02527v1)

**作者:** Jiaxuan Luo `[一作]` (Johns Hopkins University), Zhen Zhang `[通讯]` (University of California, Santa Barbara)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究考察了在机器人策略优化中使用学习视觉奖励时的“错误放大”现象，发现即使任务成功率提升，优化也可能同步放大对错误物体的失败；

**💡 创新点**

创新点在于提出并验证了“倾斜模型”来解释奖励聚合度高时仍能掩盖语义错误的机制，并通过冻结的成功验证器证明可纠正这一放大效应；

**🔧 技术方法**

采用扩散策略（Diffusion Policy）全参数微调和受限噪声选择，利用Robometer、LIV等视觉奖励，并结合KL正则化的指数倾斜理论进行分析与实验；

**📊 数据集**

使用MuJoCo仿真环境中半遮挡抽屉抓取与颜色立方体堆叠任务，生成512个随机种子样本；在Franka Research 3机器人上进行了有限的真实物理实验；

**📈 对比分析**

通过与任务完成奖励的对比，利用奖励提升、任务成功率和错误物体失败率的变化及95%自助置信区间评估；结果显示对学习奖励的优化导致成功率提升约10%，错误失败率同步提升约10%，而基准奖励未出现错误放大，差异显著；

**⚠️ 局限性**

局限性包括实验仅覆盖两类任务，规模有限；验证器为任务特定且基于仿真标签，无法完全替代真实视觉奖励；硬件实验仅为单一训练案例，缺乏广泛验证。

---

## 160. Social bot detection in the age of ChatGPT: Challenges and opportunities

**arXiv ID:** 2610.02386 | [PDF](https://arxiv.org/pdf/2610.02386v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 161. Bandits via Additive Quantized Representations

**arXiv ID:** 2610.02440 | [PDF](https://arxiv.org/pdf/2610.02440v1)

**作者:** Ami Tavory `[一作]` (Meta Platforms), Ido Guy `[通讯]` (Meta Platforms)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出利用残差量化（RQ）作为上下文表示层，结合可加层级模型和影子提升，实现上下文多臂赌博机的非线性奖励建模，且保持固定 O(1) 的内存和不使用回放缓冲。

**💡 创新点**

创新点在于：1）将离线训练的 RQ 码本与增量级联模型结合，生成可扩展深度的可加性奖励预测；2）引入影子提升机制，在线自适应选择最优深度；3）在固定内存约束下取得与树/神经网络相当甚至更优的性能。

**🔧 技术方法**

使用了 k‑means 残差量化构建码本、三种基学习器（TS‑RQ、SGD‑LinTS‑RQ、LinTS‑RQ）、Freedman 检验的影子提升、以及对比的 XGBoost、NeuralCB、TabNet 等非线性基线。

**📊 数据集**

在 13 个 OpenML/UCI 表格数据集上进行实验，样本规模从 1.3e5 到 1.1e7，特征维数 3–90，动作数 2–104。

**📈 对比分析**

将 RQ 方法与对应的无 RQ 版本以及三种非线性基线进行比较；RQ 方法在 11/13 数据集上显著优于基线，LinTS‑RQ 与 XGBoost/NeuralCB 在 regret 上相当，但内存仅为 1/1000；TS‑RQ 在多数数据集相对于 TS 获胜；整体表现优于固定深度或无深度方案。

**⚠️ 局限性**

局限性包括：① 需要足够的无标签上下文用于训练码本；② 在数据稀缺或维度高（如 year_prediction）时，LinTS‑RQ 的协方差估计不足导致性能下降；③ 需要满足 N/(bKd^2)≫1 的数据充分性条件；④ 码本训练仅最小化重构误差，未直接考虑奖励信息，可能导致误差 Δ_ℓ 进一步增大。

---

## 162. Hardware-Native Joint Sparse-Quantization for Trillion-Scale Mixture-of-Experts

**arXiv ID:** 2610.02241 | [PDF](https://arxiv.org/pdf/2610.02241v1)

**作者:** Kwanhee Lee `[一作]` (POSTECH), Dan Alistarh `[通讯]` (ISTA)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `afceb026-1760-41ae-8d86-010831a37d97` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套端到端的硬件‑软件协同设计框架，能够将Mixture‑of‑Experts模型的专家权重压缩为硬件原生的低精度半结构稀疏表示，并在NVIDIA Blackwell的稀疏张量核上实现高效推理。

**💡 创新点**

核心创新在于通过Gumbel‑Softmax连续化稀疏支持选择、联合稀疏化与量化的可微优化以及专门为半结构稀疏低精度设计的分组稀疏GEMM核，从而实现模型精度近乎不降、显著降低存储与内存带宽需求。

**🔧 技术方法**

使用了Gumbel‑Softmax、Straight‑Through Estimator、块级专家压缩、硬件原生半结构稀疏（4:8 NVFP4）量化、分组稀疏GEMM核等技术，并在NVIDIA Blackwell的SpTC上实现。

**📊 数据集**

在OpenLLM Leaderboard（GSM8K、MMLU、WG、HSwag、TQA）以及Kimi‑K2.5的多步推理基准（AIME25、GPQA Diamond、MATH500）上进行评估。

**📈 对比分析**

与稠密NVFP4、JSQ、SGPTQ、OBR、GSQ、REAP、INT4 Marlin、INT2 Humming等方法相比，压缩后模型保留了96.09%原始准确率，分组稀疏GEMM核实现了1.65×的核级加速，整体推理吞吐提升1.18×，端到端延迟降低4.03×。

**⚠️ 局限性**

局限性包括仅在NVIDIA Blackwell架构上验证，稀疏模式固定为4:8 NVFP4，可能对其他硬件或更大稀疏比率的迁移性有限，且需要专门的编译与驱动支持。

---

## 163. Autonomous mobile robot operations logistics: a dataset of jobs, dispatch events and robot states

**arXiv ID:** 2610.02428 | [PDF](https://arxiv.org/pdf/2610.02428v1)

**作者:** Jan-Felix Klein `[一作]` (KTH Royal Institute of Technology), Yongkuk Jeong `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

发布了MoRoOp数据集，记录AMR在实验室物流场景下的工作任务、操作、调度事件和机器人状态；

**💡 创新点**

创新点在于将计划层、调度层与执行层的数据以统一的表格形式整合，并保留原始与清洗后两版机器人状态，提供可对齐、可复用的多层级时间序列；

**🔧 技术方法**

技术实现基于Apache Kafka消息流、Node‑RED数据摄取、MariaDB存储以及Wheel.me AMR的OpenAPI接口，机器人状态按1 Hz轮询并仅在变化时写入；

**📊 数据集**

使用的原始数据包含9个8小时班次，共1,382个作业、4,815个操作、19,352个调度事件和140,386个机器人状态观测；

**📈 对比分析**

论文未进行算法或性能比较，只提供数据作为后续研究的基准，未给出具体性能指标；

**⚠️ 局限性**

局限性包括单一机器人实验、实验室场景限制、任务生成与实际工业作业差异、缺乏多机器人交互与大规模调度等。

---

## 164. Hypothesis-guided discovery of cognitive algorithms via program refinement

**arXiv ID:** 2610.02523 | [PDF](https://arxiv.org/pdf/2610.02523v1)

**作者:** Huiwen Alex Yang `[一作]`, Bill D. Thompson `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个混合系统，利用LLM对人类先验的概率程序模型进行逐步改进，以更好地拟合人类在灯光网格求解任务中的行为

**💡 创新点**

创新点在于将人类专业知识与LLM生成的程序修订相结合，采用结构化的程序修订循环（Game–Coding–Audit–Inference）并在保持原有模型框架的前提下自动发现局部算法改进

**🔧 技术方法**

技术包括概率编程（FlipPy）用于编码认知模型，OpenAI GPT‑5.4作为LLM代理进行游戏分析、代码生成与审计，传统的概率推理模块用于计算数据似然

**📊 数据集**

使用了150名参与者在四种实验条件（小/大板格 × 简单/复杂 DAG）下完成的灯光网格任务的行为轨迹数据，涵盖多种算法策略

**📈 对比分析**

与原始的三种基准策略（空间扫描、颜色块、两阶段填充）相比，改进模型在约53%（74/139）受试者上提升了每步对数边缘似然，并且在留出数据上也保持了显著提升；合成恢复测试显示一半以上的改进程序能够恢复或超越基准模型的预测性能

**⚠️ 局限性**

局限性包括：仅在单个任务和单轮修订循环中评估；过度拟合个体轨迹的风险；缺乏系统评估模型与更广泛理论框架的关联；对多代理或更复杂算法情境的适用性尚未验证

---

## 165. Real-time Optimization of Simulation and Data Processing Pipelines for Experiments on Exascale Computing Platforms

**arXiv ID:** 2610.02498 | [PDF](https://arxiv.org/pdf/2610.02498v1)

**作者:** Thomas Wester `[一作]` (University of Chicago), Christine M. Simpson `[通讯]` (Argonne National Laboratory)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并验证了一种实时两阶段任务调度策略，利用流水线各阶段运行时间的相关性，在Exascale HPC上显著提升高能物理Monte Carlo流水线的资源利用率并缩短完成时间。

**💡 创新点**

创新点在于将流水线拆分为生成阶段与下游阶段，利用第一阶段的耗时信息动态调整后续任务优先级，实现最长优先的自适应调度，并提供理论分析与离散事件仿真框架来评估调度最优性。

**🔧 技术方法**

使用了随机任务时间分布下的 makespan 与 idle 估算理论、基于 DAG 的离散事件调度仿真器、两阶段拆分与动态最长优先调度策略，并在 HPE Cray EX Aurora 256 节点集群上进行实验。

**📊 数据集**

实验基于 sbnd 研究所的 neutrino Monte Carlo 流水线，包含约 104448 条流水线（每条约百万级事件），采集了任务运行时间分布及其相关性统计。

**📈 对比分析**

通过与理论极限、仿真结果以及真实运行数据在 T_p 与 ϕ_p（idle 率）上进行对比，动态多阶段调度在 99% 任务完成时 makespan 约 153 分钟、idle 率仅 4.5%，相比无序单阶段调度（157 分钟、11% idle）提升显著。

**⚠️ 局限性**

局限性包括：理论假设任务时间均匀分布、未考虑多节点网络延迟与启动延迟、仅测试 CPU 资源、在高方差任务时间下动态调度效果下降，以及未覆盖混合 CPU/GPU 资源的情况。

---

## 166. The Power of Flexible Budgets in Adwords

**arXiv ID:** 2610.02479 | [PDF](https://arxiv.org/pdf/2610.02479v1)

**作者:** Suho Kang `[一作]`, Rajan Udwani `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了具有预算灵活性的D天AdWords问题，分析在每日预算可超出δ倍但总预算仍受限的情形下，设计了一种基于时间感知惩罚函数的在线分配算法并证明其在大多数情境下实现了最优的竞争比率1‑e⁻ᵟ；

**💡 创新点**

创新点在于：①首次量化了预算灵活性对AdWords worst‑case 性能的提升；②引入“完美利用”设定来精确刻画极限竞争比，并以此得到最优惩罚函数；③证明了该算法在任意固定δ下随天数D趋于无穷时收敛到1‑e⁻ᵟ；④给出了D=δ=2的精确最优竞争比约0.701，展示了完美利用并非全局最优；

**🔧 技术方法**

采用了LP‑free 竞争分析框架，利用等化性质与指数解释构造惩罚函数，并使用递归与指数随机变量的期望分析证明了下界；同时通过改进的上三角构造实例给出上界；

**📊 数据集**

该工作为纯理论分析，无使用实测数据集；

**📈 对比分析**

与传统的Balance/Reduced‑Bid算法比较，本文算法在预算灵活的情况下把竞争比从1‑1/e提升到1‑e⁻ᵟ（在大D时几乎达到此极限），并在有限天数下提供更精确的最优值；

**⚠️ 局限性**

局限性包括：只在小竞价（γ→0）假设下工作，未讨论大竞价场景；对一般D与δ的精确有限期望最优仍未给出；算法的惩罚函数虽已优化，但实现复杂度未讨论。

---

## 167. Dense Mixture-of-Experts as a Reparameterized Wide FFN: A Granularity Sweep at Fixed Compute

**arXiv ID:** 2610.02584 | [PDF](https://arxiv.org/pdf/2610.02584v1)

**作者:** Vu Quang Hoang `[一作]` (University of Information Technology), Nghia Hieu Nguyen `[通讯]` (University of Information Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在固定总FFN宽度下，将Transformer的FFN拆分为K个全激活的SwiGLU专家，并使用softmax门控进行加权，比较K=1（单一密集专家）与K>1（动态专家组合）在验证集上的性能。

**💡 创新点**

首次在不使用稀疏路由或大专家池的情况下，探究仅通过token级软加权来评估动态专家组合的真正效益，并给出其与宽密集FFN的等价理论。

**🔧 技术方法**

采用12层Decoder结构，使用RoPE位置编码、RMSNorm、SwiGLU激活；通过softmax门控组合K个专家；使用AdamW优化器，混合fp16精度训练。

**📊 数据集**

训练数据为FineWeb-10B子集（约1.3B tokens），验证集为FineWeb完整验证拆分。

**📈 对比分析**

以验证损失和困惑度为指标比较，发现K=2在训练后期可略优于基线（-0.0048 loss），而K=4、K=6表现更差（+0.0053/+0.0197）；单跑结果显示性能差距在后期稳定。

**⚠️ 局限性**

实验规模仅数千万参数，且每个配置仅跑一次且未固定随机种子；仅评估单一数据集；未尝试更大K或稀疏MoE模型；缺乏多跑统计与训练时序分析，导致结果缺乏显著统计显著性。

---

## 168. Keep the Effect, Drop the Actor: Programmable Effect-to-Execution World-Action Models

**arXiv ID:** 2610.02398 | [PDF](https://arxiv.org/pdf/2610.02398v1)

**作者:** Junyi Hu `[一作]` (New York University Abu Dhabi), Yi Fang `[通讯]` (New York University Abu Dhabi)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将演示转换为无演示者的效果程序，仅保留物体轨迹、接触点和终端状态，使用单一世界‑动作模型通过编程方式生成机器人执行。

**💡 创新点**

1) 提供一种演员独立的任务接口；2) 在单一网络中同时学习效果、执行、动作和终端状态；3) 通过编程式推理实现闭环重规划；4) 支持程序编辑和跨机器人迁移。

**🔧 技术方法**

一个具有 AdaLN‑Zero 调制的 71.5M 参数 DiT，四个流的独立 flow‑matching；采样推理、闭环指针/门控；基于 SAM 2 的关键点提取。

**📊 数据集**

LIBERO‑Goal、Meta‑World（ML10/ML45）、RLBench、四种仿真机器人和真实 Franka arm 的人类视频演示。

**📈 对比分析**

与 UWM、Instant Policy、ATM、Zero‑WAM 等零样本方法以及 Meta‑World 公开基线对比。示例：在 3 个 LIBERO 任务上 44% 成功率（对比 21%/5%/0%/0%）；Meta‑World 3 类 100%/52%/22% 成功率，超过大多数基线；在 RLBench 24 任务上 160/720 成功率（相对 108/720）。

**⚠️ 局限性**

仅能描述显著移动的物体；指针和门控规则需人工设定；对箱体假设为刚体；在 RLBench 上仍远低于专用策略；依赖精确关键点和接触点的标注。

---

## 169. Latent-MOPD: Latent Multi-Teacher On-Policy Distillation

**arXiv ID:** 2610.02381 | [PDF](https://arxiv.org/pdf/2610.02381v1)

**作者:** Zhengyu Fang `[一作]` (Case Western Reserve University), Jing Li `[通讯]` (Case Western Reserve University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在已有的专业化 RL 训练模型的基础上，提出了一种多教师表示级别的在线蒸馏（OPD）方法，利用教师的隐藏层表示与输出概率共同训练单一学生模型。

**💡 创新点**

创新点：①首次将隐藏状态监督与多教师路由相结合；②根据教师与学生的关系动态选择深层表示，并使用共享投影对齐不同宽度；③在训练中采用域纯批次更新和教师特定的交叉淡化（crossfade）来平衡隐藏层与 token 监督。

**🔧 技术方法**

使用的技术包括：多教师路由在线蒸馏（MOPD）、表示匹配（LastOPD/OPRD）、共享线性投影、域纯批次更新、交叉淡化调度、参数合并初始化（model soup）以及中心化核对齐（CKA）等。

**📊 数据集**

训练数据集为 DAPO‑Math‑17k、OpenCodeReasoning 与 Reasoning Gym（共 17,856 条提示）；评估基准涵盖 Math（BBH, MuSR, AIME24）、Code（MBPP, MBPP+, LiveCodeBench‑easy）与 Logic（Minerva, GYM）等。

**📈 对比分析**

与 token‑only、representation‑only、均匀平均等单通道基线以及各教师自身性能进行对比。该方法在同族 1.5B 设置下平均提升 Norm 至 1.05，跨族 7B 教师中提升至 0.26，并在 5/9 基准上超过最佳教师；跨族实验中也优于所有单通道基线。

**⚠️ 局限性**

局限性：仅在相同模型族或固定规模下验证；跨族对齐仍需共享投影，宽度差异适配复杂；训练过程依赖域纯批次与交叉淡化超参，缺乏通用性；未评估大规模 7B+教师组合的可扩展性与推理成本。

---

## 170. Feature Freshness Budgets for Real-Time ML Inference Under Stream Lag

**arXiv ID:** 2610.02259 | [PDF](https://arxiv.org/pdf/2610.02259v1)

**作者:** Amit Rajula `[一作]` `[通讯]` (Independent Researcher), Amit Rajula (Independent Researcher)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了在线特征存储的“新鲜度预算”模型，并给出了特征新鲜度的上界与崩溃阈值；

**💡 创新点**

创新点在于把特征新鲜度从二元“足够/不足”转为可量化、可预算的资源，并证明了闭式崩溃阈值与排队负载相关；

**🔧 技术方法**

使用离散事件仿真、Apache Kafka、Redis 与 PostgreSQL 构建的实时特征存储链路，并分析采样相位与延迟对新鲜度的影响；

**📊 数据集**

使用合成 Poisson + 峰值事件流模拟单实体工作负载，未使用公开真实数据集；

**📈 对比分析**

通过对比三种消费策略（基线、阻塞、预算）以及不同回退策略，在仿真与真实部署中验证阈值的准确性；仿真与真实系统在阈值处的误差<0.02，表明模型准确；

**⚠️ 局限性**

局限性包括仅针对单实体合成负载、仅考虑固定阈值决策规则、回退策略简单、相位模型未直接测量、以及结果只适用于所测试的配置，未给出生产级基准。

---

## 171. Out of Sync, Out of Sight: Phantom State Attacks against IIoT Intrusion Detection

**arXiv ID:** 2610.02552 | [PDF](https://arxiv.org/pdf/2610.02552v1)

**作者:** Sabrine Ennaji `[一作]` (University Lyon 1), Nadia Kabachi `[通讯]` (University Lyon 1)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种只通过对攻击流的计时漂移进行最小化校准的Phantom State Attack，能在不查询模型、不修改数据包内容的零查询威胁模型下，使工业物联网 IDS 的窗口聚合失效；

**💡 创新点**

创新点在于利用监控流水线对窗口边界的时间同步假设，将攻击者的时序偏移与窗口划分耦合，从而在无需模型或对手模型的情况下实现误判；

**🔧 技术方法**

主要技术包括基于流的交互时延统计、窗口边界距离计算、流自适应漂移注入和窗口重分配特征重构；

**📊 数据集**

在两个工业 IoT 数据集上评估：ToN‑IoT 和 CIC IIoT 2025（DataSense），覆盖 14 种攻击类别；

**📈 对比分析**

与随机森林、MLP、XGBoost 等传统模型对比，PSA 在满足窗口跨越条件的攻击类别中可降低 10%–50% 的检测准确率，且零查询成本；与基于查询的对手相比，PSA 取得较低的成功率但消耗完全不存在查询；

**⚠️ 局限性**

局限在于仅对单一流的时序偏移有效，需攻击流包含足够多包且跨窗口分布；对短短密集流、对时间漂移受限的环境效果有限，并且对高漂移检测器易被发现。

---

## 172. Proof Interfaces for Exploratory Mathematics

**arXiv ID:** 2610.02449 | [PDF](https://arxiv.org/pdf/2610.02449v1)

**作者:** Nishant Kheterpal `[一作]` (University of Michigan), Jean-Baptiste Jeannin `[通讯]` (University of Michigan)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

扩展 Hazel Prover，设计并实现了一个面向教学与探索数学的交互式界面，支持多级自动化、可配置的数学操作文件、rewrite search 机制，并能将一步步推导导出为 Coq 证明。

**💡 创新点**

创新点包括：① 将教育与专家需求结合的多级自动化模式；② 通过可定制的“math profile”实现细粒度的步骤控制与自动化组合；③ 结合 rewrite search 与 Coq 验证的双向集成，实现可验证的、可导出的证明；④ 在网页端实现即时反馈与可视化，降低学习曲线。

**🔧 技术方法**

主要技术：Hazel live 编程环境、rewrite search 架构、可配置的 math profile、JSCoq（浏览器端 Coq）、Coq 证明脚本生成与验证、JavaScript 与 Web 前端交互。

**📊 数据集**

没有使用传统机器学习或图像数据集；而是通过手工构造的若干数学案例（算术、代数、三角、微积分等）进行案例研究评估。

**📈 对比分析**

评估方法：通过一系列渐进式案例研究（从运算顺序到二次泰勒多项式），比较低配置成本、交互负担、步骤粒度、数学覆盖度等指标；结果表明界面满足教学与专家两种工作流的需求，且 Coq 验证能够在大部分情况下自动完成；在性能方面，rewrite search 受限于 profile，速度可接受，但在某些复杂表达式下可能略慢。

**⚠️ 局限性**

局限性：① 目前只能对完整程序进行步进，无法对子表达式单独操作；② Taylor 余项等高级符号计算无法导出 Coq 证明；③ 高级自动化模式下搜索有时会过慢；④ 需要手工定义 math profile，缺乏自动生成或迁移能力。

---

## 173. Geometry-Aware Time Reparameterization for Flow-Map Distillation

**arXiv ID:** 2610.02427 | [PDF](https://arxiv.org/pdf/2610.02427v1)

**作者:** Félix Dedek `[一作]` (Okinawa Institute of Science and Technology Graduate University), Makoto Yamada `[通讯]` (Okinawa Institute of Science and Technology Graduate University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过对预训练 ODE 的时间参数化进行几何感知的重参数化，改进了流图蒸馏（flow‑map distillation）以实现一阶或少阶生成。

**💡 创新点**

创新点在于提出基于轨迹法向加速度的硬度（hardness）诊断，并构造一个共享时钟（clock）使得硬度在时间上均匀化，从而让学生网络在学习有限时间映射时更容易。

**🔧 技术方法**

核心技术包括流匹配（flow matching）、Lagrangian 流图蒸馏、法向加速度估计、PCHIP 插值重参数化、以及在教师轨迹上一次性估计时钟并在学生训练中使用。

**📊 数据集**

实验使用了三组数据集：二维检波板（checkerboard）作为合成数据，CIFAR‑10 以及 CelebA‑64 作为图像数据。

**📈 对比分析**

与使用恒等时间（Id）蒸馏的基线相比，重参数化时钟在匹配目标分布的 FID 上均有提升，尤其在一阶生成时显著降低 FID（如 CIFAR‑10 从 8.57 降至 6.99，CelebA‑64 从 6.19 降至 4.64），同时在多阶生成中也表现更优。

**⚠️ 局限性**

局限性包括对教师 ODE 的 C² 连续性和非零速度的假设；缺乏对法向加速度与蒸馏误差的理论关联；以及重参数化仅在教师预训练后一次估计，可能不适用于动态或多任务场景。

---

## 174. Neuron merging via inverse-activation regression for post-training compression of sigmoid neural networks

**arXiv ID:** 2610.02559 | [PDF](https://arxiv.org/pdf/2610.02559v1)

**作者:** Ao Kuniya `[一作]` (Saitama University), Jun Ohkubo `[通讯]` (Saitama University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于逆激活函数（logit）回归的后训练神经元合并框架，并通过权重与激活信息的聚类与合并方法探讨了信息利用的重要性。

**💡 创新点**

首次结合数据驱动的逆激活回归与权重聚类，实现了无需微调即可在Sigmoid网络中高效合并神经元，并展示了权重信息适用于聚类、激活信息适用于重构的互补特性。

**🔧 技术方法**

使用k‑means聚类（权重或激活特征）、加权平均与基于logit的最小二乘回归、Ridge正则化等技术，对全连接Sigmoid网络进行压缩。

**📊 数据集**

在MNIST和Fashion‑MNIST两个小规模手写/服饰图像数据集上进行实验。

**📈 对比分析**

与随机、L1、L2权重幅值的结构化剪枝方法比较，合并方法（尤其是C‑weight/M‑logit）在各种压缩比下都显著优于剪枝，数据驱动的合并甚至在高压缩率下保持较高精度。

**⚠️ 局限性**

实验仅限于全连接Sigmoid网络与小数据集，方法只适用于可逆激活；对ReLU等非可逆激活缺乏适配；随机数据的聚类表现不佳；缺乏对大规模或卷积网络的评估与微调机制。

---

## 175. The AI Theorist reveals excitonic structure in $α$-RuCl$_3$

**arXiv ID:** 2610.02417 | [PDF](https://arxiv.org/pdf/2610.02417v1)

**作者:** Hongjian Zhou `[一作]` (University of Oxford), David A. Clifton `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过AI Theorist系统，自动生成并验证物理模型，解释了α‑RuCl₃在差分反射和光电流光谱中观测到的激子态及其极化选择规则。

**💡 创新点**

首次实现基于实验数据的全自动物理模型生成与迭代优化，结合LLM驱动的假设生成、first‑principles计算与反馈修正，专为量子材料设计。

**🔧 技术方法**

使用大型语言模型（Claude Opus 4.6、Qwen3.5‑27B）以及计算工具Quantum ESPRESSO、BerkeleyGW、GP‑UCB 参数搜索和Paperclip文献检索。

**📊 数据集**

实验数据集包括8 K低温差分反射光谱与光电流光谱（α‑RuCl₃/graphene/hBN 结构）以及相关材料文献数据库。

**📈 对比分析**

与传统手工模型对比，AI模型成功解释三低能峰（L、α、α′）的能量、极化依赖关系；计算时间约2.5–38 h，总计≈154 M tokens，成本约US$335。

**⚠️ 局限性**

受限于计算资源、对实验条件的依赖、激子收敛性与高能多体效应的处理不足，且对极化旋转实验预测尚未验证。

---

## 176. Line-Rate GTP-U Admission Control at the Edge of a Cloud-Native 5G Core: An XDP-Based Design for Kubernetes-Hosted User Plane Functions

**arXiv ID:** 2610.02296 | [PDF](https://arxiv.org/pdf/2610.02296v1)

**作者:** Simhadri Podala Narasimha `[一作]` `[通讯]` (Independent Researcher), Simhadri Podala Narasimha (Independent Researcher)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了GTP‑Guard，一种基于XDP的边缘UPF入口层，能够在驱动接收路径上对GTP‑U流量进行合法性校验并及时丢弃非法包，提升云原生5G核心的安全性与可用性。

**💡 创新点**

创新点在于：①将XDP用作UPF无关的“前门”，实现对任意UPF实现的流量控制；②通过成本‑概率比的排序理论给出最优阶段顺序，并支持运行时通过尾调用重新排序；③在Kubernetes环境下提供持久化映射、代理升级、公共云NIC兼容等完整生命周期与部署方案。

**🔧 技术方法**

使用的技术包括：eBPF/XDP（驱动层可编程网络）、CO‑RE编译技术、AF_XDP、BPF映射（LPM、哈希、Per‑CPU数组、程序数组）、Token Bucket计量器、Kubernetes DaemonSet与CRD、Prometheus监控、NF架构PFCP与SMF同步。

**📊 数据集**

数据集主要是合成的GTP‑U流量（TRex/PacketRusher生成的统计分布），覆盖正常会话（10^4–10^6 TEID）、TEID扫描、伪造源、GTP‑in‑GTP、会话变更等多种攻击/工作负载；未使用真实业务流量，实验基于公共云与本地测试平台。

**📈 对比分析**

方法是对比无过滤（B0）、仅PFCP规则过滤（B1）、XDP泛型（B2）、XDP原生融合（B3）以及XDP原生自适应排序（B4）的吞吐量、丢包容量、CPU周期/包、延迟与误丢率；实验显示：①B3在正常负载下几乎无吞吐或延迟损失；②B3在TEID扫描攻击下每核丢包容量远高于B1、B2；③自适应排序可在攻击混合变化时进一步降低每包周期；④PFCP到映射的延迟保持在N2信令延迟以内，避免插入前使用错误。

**⚠️ 局限性**

局限性包括：①仅支持IPv4，IPv6及扩展头解析待扩展；②对分片包的处理有限，需依赖栈重组；③映射状态滞后导致合法流量被误丢，需监控插入延迟；④共享内核风险需严格控制加载权限；⑤仅过滤入站流量，未覆盖下行N6或完整PDR优先级；⑥公共云NIC对MTU/队列的约束需手动调整，可能导致性能下降。

---

## 177. SCION: Scene Composition with Instanced Neural Primitives

**arXiv ID:** 2610.02322 | [PDF](https://arxiv.org/pdf/2610.02322v1)

**作者:** William Koch `[一作]` (Princeton University), Felix Heide `[通讯]` (Princeton University)

**通讯引用:** 9477 | [OpenAlex ID](https://openalex.org/A5059313827)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种层次化、可复用的 3D 高斯场表示法，使用词汇表中的局部高斯模板和轻量级实例实例化构建完整场景，支持编辑与动画；

**💡 创新点**

核心创新在于将重复结构直接融入表示中：通过共享几何模板与实例化参数，减少独立高斯数量，同时利用词汇表级别的稠密化、分割和对抗性细节保持来解决共享导致的模糊问题；

**🔧 技术方法**

技术包括：多视角联合优化（分离梯度）、局部高斯词汇表与全局实例参数的双向稠密化、基于 MH-3DGS 的行生成与重定位、实例克隆/细分、对抗性补丁损失、以及 DINO 特征初始化与固定词汇分配；

**📊 数据集**

在 Mip-NeRF 360、Tanks & Temples、Deep Blending、Synthetic-NeRF 四个公开基准数据集上进行实验；

**📈 对比分析**

与现有 3D 高斯压缩方法（ContextGS、HAC++、OMG 等）在相同存储（≈1–2 MB）下进行比较，虽然 PSNR/SSIM 略低，但在 1.5 MB 时能保留重复结构，提供更好的可编辑性与动画效果；

**⚠️ 局限性**

局限性包括：在缺乏明显重复或单实例变化大的场景下共享效果差，硬分配限制了梯度更新，训练成本高于普通 3DGS。

---

## 178. Parameter-Free Interval-Dynamic Regret under Heavy-Tailed Noise

**arXiv ID:** 2610.02258 | [PDF](https://arxiv.org/pdf/2610.02258v1)

**作者:** Vaneet Aggarwal `[一作]` (Purdue University), Vaneet Aggarwal `[通讯]` (Purdue University)

**通讯引用:** 6925 | [OpenAlex ID](https://openalex.org/A5064822688)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

论文提出了一种参数无关的在线凸优化算法，能够在每个未预先指定的时间窗口内，以未知的重尾噪声下获得动态回退（interval‑dynamic regret）上界，且额外的适配代价仅为对数平方量级。

**💡 创新点**

创新点在于：①将期望校准与本地相对熵（relative‑entropy）比较相结合，构建可预测的“睡眠专家”框架；②利用共享梯度 AdaGrad 与 dyadic 窗口分解，精确控制窗口位置、持续时间和重启成本；③在未知 p‑阶噪声矩的条件下，证明了回退上界与下界的对数噪声功率相匹配，展示了统计与算法成本的本质区别。

**🔧 技术方法**

核心技术包括：多速率乘法聚合、期望校准（expected calibration）与正切剩余的上界、相对熵变分不等式、共享梯度 AdaGrad、dyadic 窗口与基轨迹、以及对噪声条件矩的自适应处理。

**📊 数据集**

本工作为纯理论研究，无实验或公开数据集；所有结果均为渐进式上界和下界的分析推导。

**📈 对比分析**

与现有方法（如 AdaGrad、Ader、AOA 等）比较时，论文证明其在未知 p‑噪声和未预先指定窗口的情形下，仍保持与最优全景（full‑horizon）和区间（interval）基准相当，只多出 O(log²T) 的适配项；在已知统计量时，适配项进一步降为 1+log(T/n)，实现了理论最优阶的区间动态回退。

**⚠️ 局限性**

局限性包括：①参数无关版仍需 O(log²T) 的适配成本；②算法假设可测的有限 p‑阶噪声矩和已知域直径 D；③对极端重尾噪声（p→1）时，校准与适配的常数可能较大；④理论分析依赖于理想的投影和梯度取样，未考虑计算开销或实战噪声分布的不完全满足。

---

## 179. OpenGameEval: Benchmarking Agentic Programming and Exploration in a Stateful Game Engine

**arXiv ID:** 2610.02563 | [PDF](https://arxiv.org/pdf/2610.02563v1)

**作者:** Eray Turkel `[一作]` (Roblox), Tiantian Zhang `[通讯]` (Roblox)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一套基于游戏引擎的代理程序开发评测框架，能够在可重现的、具有状态的游戏环境中运行大型语言模型作为代理，并通过编辑场景与模拟游戏运行两阶段检查来判定任务成功。

**💡 创新点**

创新点在于：① 将观测工具与执行工具明确分离，形成八种可调用工具的行动空间；② 在评测中引入游戏运行时验证（物理、网络、客户端-服务器交互）而非仅依赖单元测试；③ 通过任务依赖注解量化探索行为，并证明探索覆盖率是预测任务成功的显著指标。

**🔧 技术方法**

技术实现包括：使用 Roblox 游戏引擎及其编辑器插件；构建任务场景（place files）与参考解法；定义 13 种工具（搜索、读取、编辑脚本、执行脚本等）；利用语言模型自动调用工具；采用两阶段检查（check_scene 与 check_game）以及统计学评估（pass@k、consensus、覆盖率等）。

**📊 数据集**

数据集为 113 题的核心任务（84 题）与调试任务（29 题），每题配有 place 文件、提示语、参考解法和单元/游戏运行检查；其中 54 题带有依赖注解，用于测量探索覆盖率。

**📈 对比分析**

比较方法：对 13 个前沿模型在 84 个核心任务上执行 16 次尝试，采用配对 Wilcoxon 检验和自助置信区间评估 pass@1、pass@5、cons@5、all@5；通过工具调用计数、错误率、耗时等过程指标；探索覆盖率与成功率的线性回归。性能方面，最强模型单次尝试通过率仅 51.7%，五次连通通过率 39.4%；不同模型虽然整体通过率相近，但在脚本编写与场景修改两类任务上表现差异显著。

**⚠️ 局限性**

局限性包括：仅针对 Roblox 引擎与文本输入，未加入视觉反馈；探索度测量依赖人工注解的任务依赖，无法自动化；检查可能存在误判风险；公开任务与注解可能导致模型泄漏；实验仅在单一游戏引擎下验证，难以直接推广到其他平台。

---

## 180. DAGS: Disentangled Appearance-and-Geometry Steering of a Frozen Image DiT for Temporally Stabilized Generative Rendering

**arXiv ID:** 2610.02567 | [PDF](https://arxiv.org/pdf/2610.02567v1)

**作者:** Karthik Mohan Kumar `[一作]` (Advanced Micro Devices, Inc.), Rama Harihara `[通讯]` (Advanced Micro Devices, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出DAGS，一种轻量、无注意力、解耦外观-几何条件下的冻结图像DiT调控框架，用于实现时序稳定的高质量生成渲染。

**💡 创新点**

通过两条独立卷积编码器实现外观与几何的解耦控制，配合递归照明稳定器和无训练时间的时间引导，既保留冻结后端强大先验，又显著提升可控性、画质与时间稳定性。

**🔧 技术方法**

使用冻结的Flux.2‑Klein 4B DiT、两条卷积编码器、递归U‑Net照明稳定器、relit prior作为流源，以及训练自由的时间引导机制。

**📊 数据集**

训练使用室内场景数据集，评估在9个hold‑out场景及部分外部分布数据集上，输入为1‑spp路径跟踪帧加G‑buffer。

**📈 对比分析**

与训练免费去噪器OIDN和SD1.5前向渲染器RGB↔X对比；DAGS在重建PSNR上分别比OIDN高+8.6dB、比RGB↔X高+10.1dB，LPIPS显著下降，时间稳定性tLP比OIDN低2.5×，E_warp降低到2.52。

**⚠️ 局限性**

局限性包括：训练数据仅为室内场景导致域外表现下降；高频细节与高光仍会闪烁；快速相机运动下重投影失效，导致时间引导效果受限。

---

## 181. Learning Closure of Dynamical Systems with Kernel Ridge Regression

**arXiv ID:** 2610.02564 | [PDF](https://arxiv.org/pdf/2610.02564v1)

**作者:** Evan Habbershaw `[一作]` (Pennsylvania State University), Senwei Liang `[通讯]` (Texas Tech University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于核岭回归（KRR）的闭合建模框架，用以识别 ODE/PDE 及动量闭合问题中的缺失动力学成分。

**💡 创新点**

创新点在于（1）给出差分闭合的误差界定并证明数值求解器与插值误差的主要贡献；（2）在多模态和非平稳情形下引入空间局部化的 KRR 代替全局 PCA，显著提升泛化性能；（3）将 KRR 与扩散图核、低阶多项式插值等技术结合，实现长时 horizon 的高精度预测。

**🔧 技术方法**

主要技术包括：核岭回归与扩散图核（DM）、低阶多项式插值、RK4/ETDRK4 等 ODE/PDE 数值积分器、LSTM 对比实验、PCA 降维、局部窗口建模、BGK 1D 动力学数值求解、IMEX RK、Rusanov 等价数值通量。

**📊 数据集**

数据集涵盖 Lorenz‑63 系统、Kuramoto–Sivashinsky PDE 与 1D BGK 动力学方程的平滑热扰动初始条件，训练集规模从 512 至 16384，分别用于验证与测试。

**📈 对比分析**

性能评估采用 Valid Prediction Time（VPT）和空间 L² 误差等指标；在 Lorenz‑63 上 KRR 取得约 8 Lyapunov 时代的 VPT，约为 LSTM 的两倍；在 KS PDE 上 VPT 约 2.6，明显优于 LSTM；在 BGK 闭合中，局部一阶 KRR 在两类初始条件下平均误差最低。

**⚠️ 局限性**

局限性包括：全局 PCA 模型对平移与多模态的泛化不足；KRR 对噪声鲁棒性不足；在高维或更复杂流体/多体问题中需进一步验证与扩展；非平稳、粗网格或高斯噪声下的稳健性仍是未来工作。

---

## 182. Physical AI Smart Spaces: A Large-Scale Benchmark for Multi-Camera 3D Perception in Smart Spaces

**arXiv ID:** 2610.02580 | [PDF](https://arxiv.org/pdf/2610.02580v1)

**作者:** Yuxing Wang `[一作]` (NVIDIA), Zheng Tang `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `6514db3d-8de6-452c-91b7-acdb31787cc4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

发布了一个大型多摄像头3D感知基准数据集PAISS，包含同步1080p视频、自动3D/2D标注、相机标定、深度信息、Cosmos Transfer视觉域迁移数据以及隐藏的真实仓库目标；同时提出3D HOTA评估与完整的生成与评测流程；

**💡 创新点**

①同时满足规模大、3D多类、多摄像头、自动标注、深度和Sim2Real评估的综合基准；②首次在此类任务中引入3D HOTA评估，统一检测、定位与身份一致性；③利用Cosmos Transfer 2.5实现视觉域迁移与隐藏真实目标的Sim2Real验证；④公开完整的生成流水线与标注规范，方便复现与扩展；

**🔧 技术方法**

使用NVIDIA Omniverse/Isaac Sim（Replicator Agent、Animated Robot Controller、RTX Sensor Placement/Calibration）进行合成渲染；Cosmos Transfer 2.5进行视觉域迁移；VGGT视觉几何推理完成真实世界相机标定；深度图与点云聚合做3D检测；TrackEval + 3D HOTA实现评测；

**📊 数据集**

PAISS 2024/2025/2026公开合成与CT2.5数据，共139场景、1,799摄像头、282.5小时1080p视频、72M+ 3D标注、241M 2D框，涵盖7类对象（人、NovaCarter、Transporter、FourierGR1T2、AgilityDigit、Forklift、PalletTruck）；此外还有隐藏的真实仓库目标（RGB‑only）做Sim2Real评估；

**📈 对比分析**

在AI City Challenge 2024/2025两届公开排行榜上对比多种方法：2024年顶级M CBLT获得81.22 HOTA（人物定位），2025年顶级ZV获得69.91 HOTA（多类3D盒）；方法包括离线点云+Transformer、RGB‑only BEV、深度聚合+ReID、在线ReID+几何一致性等，显示深度、点云、BEV及在线/离线融合等技术均能显著提升性能；

**⚠️ 局限性**

仅覆盖静态预标定室内1080p摄像头，主要为仓库场景；缺乏真实人物影像，不能用于面部识别或公平性研究；Cosmos Transfer单视图训练可能带来偏差；资产库有限且不平衡，可能影响模型泛化；隐藏真实目标受限，公开数据无法直接验证部署性能，实际应用需额外隐私与环境验证；

---

## 183. Santiago's A.T. Field: Visualizing Urban Accessibility through an Evangelion-Inspired Interface

**arXiv ID:** 2610.02562 | [PDF](https://arxiv.org/pdf/2610.02562v1)

**作者:** Eduardo Graells-Garrido `[一作]` (Universidad de Chile), Claudio Gaete `[通讯]` (Universidad de Chile)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

将Neon Genesis Evangelion中的A.T. Field视觉化界面改编为基于Santiago市的实际可达性评估工具，并通过扫描动画展示可达性指数。

**💡 创新点**

创新在于把科幻界面翻译为真实城市分析平台，利用扫描动画解释E2SFCA模型的累积效果，并在界面设计上兼顾叙事权威与可解释性。

**🔧 技术方法**

使用React+deck.gl+MapLibre GL+h3-js构建交互式地图；Python库tobler、QuackOSM、r5py用于数据预处理和步行时间计算；E2SFCA与α参数实现通勤调节；CIELCh空间构建可视化调色板。

**📊 数据集**

数据来源包括2020年智利人口普查计数、OpenStreetMap街网及药房/超市/银行/医疗机构位置、工作地点分布等。

**📈 对比分析**

本文未给出量化性能指标；仅提及对颜色可视化的生理学验证、动画帧预算控制以及对减少运动偏好的适配。未来可通过静态地图与扫描动画对比实验评估读者记忆和理解。

**⚠️ 局限性**

局限包括：尚未在规划实践中使用，缺乏实证评估；色彩对色弱用户的完整验证未覆盖动画色调；动画受浏览器设置限制；缺乏多城市可复制性与更细粒度的工作使用数据。

---

## 184. Evaluating Multi-Dimensional Generalization of Large Language Models in Temporal Extraction Tasks

**arXiv ID:** 2610.02549 | [PDF](https://arxiv.org/pdf/2610.02549v1)

**作者:** Fahmid Shahriar Iqbal `[一作]` (University of North Texas), Sagnik Ray Choudhury `[通讯]` (University of North Texas)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在时间与事件表达式提取任务上，系统评估了多种LLM配置在四种泛化维度（域迁移、鲁棒性、组合性、结构性）下的表现，探究了基线性能与泛化能力的关系；

**💡 创新点**

创新点在于构建了覆盖四维泛化的综合评估框架，比较了不同推理策略（归纳、演绎、溯因及其组合）与模型架构（Dense vs MoE）及规模对泛化的影响；

**🔧 技术方法**

采用了提示式LLM（LLaMA、Qwen、Mistral），包括Dense与Mixture‑of‑Experts变体，使用七种推理策略；评估指标为基于Hungarian算法的微平均F1；还对比了微调的RoBERTa、T5、GPT‑2等预训练模型；

**📊 数据集**

使用TimeBank 1.2作为基线，构造了对抗数据（TextFooler、DeepWordBug）、临床域数据T2C、词汇替换域Voc、文档级长度数据Len以及组合性数据Comp；

**📈 对比分析**

通过统计显著性检验和Spearman相关性对模型在不同维度下的排序进行比较。结果显示：强基准性能往往能预测大部分泛化维度，但在大规模域偏移（T2C）时预测失效；归纳式提示最稳定；MoE并不总能超越Dense；模型规模对鲁棒性和结构性有提升，但对域迁移和组合性影响有限；

**⚠️ 局限性**

局限性包括仅聚焦时间与事件表达式提取，未覆盖更高级的时序推理任务；可能存在预训练语料污染；无法单独归因于架构差异或提示设计对性能的具体贡献；

---

## 185. BaCP: Backbone Contrastive Pruning for Preserving Representations in Extremely Sparse Neural Networks

**arXiv ID:** 2610.02524 | [PDF](https://arxiv.org/pdf/2610.02524v1)

**作者:** Mohammad Haroon Khawaja `[一作]` (Lahore University of Management Sciences), Muhammad Tahir `[通讯]` (Lahore University of Management Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了无结构剪枝在极高稀疏度下的表示崩塌问题，提出了骨干对比剪枝（BaCP）方法，通过多源对齐正则化保持稀疏网络的嵌入质量。

**💡 创新点**

BaCP 在剪枝过程中加入来自预训练、微调和历史快照的冻结对齐正则化，并采用全对全相似矩阵的对比损失，显著提升了超过 99.9% 稀疏度的鲁棒性。

**🔧 技术方法**

使用对比学习正则化（CAP 的 PrC/FiC/SnC 分解）、无结构剪枝准则（Magnitude、SNIP‑it、WANDA）、全局稀疏度调度、卷积骨干网络、增强堆栈以及交叉熵与对比损失的组合。

**📊 数据集**

在 CIFAR‑10 和 CIFAR‑100 上对 ResNet‑34/50、VGG‑11/19、MobileNetV2 等骨干进行评估。

**📈 对比分析**

与相同预算的迭代剪枝基线（仅交叉熵）对比，在 90 个实验设置中 BaCP 在 53 处获胜；当基线性能低于 70% 时平均提升约 4.76 分；在基线性能高于 85% 时差距可忽略不计。

**⚠️ 局限性**

对比正则化的收益高度依赖增强和学习率的调整，已保留表示的场景几乎无提升；在 MobileNetV2 高稀疏度下出现回退，原因尚未完全解析。

---

## 186. Learning the Latent Structure: A Feature-Centric Approach to Graph Data Augmentation

**arXiv ID:** 2610.02517 | [PDF](https://arxiv.org/pdf/2610.02517v1)

**作者:** Yu Song `[一作]` (Michigan State University), Hui Liu `[通讯]` (Michigan State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于特征的图数据增强框架 SelfAug，直接在嵌入空间对节点表示进行补偿而非显式修复图结构。

**💡 创新点**

创新点在于利用自监督的逆掩码重建目标，学习嵌入残差映射，并结合信息消息正则化与自举训练以提升泛化与鲁棒性。

**🔧 技术方法**

使用了 GNN 编码器（如 GAT）与三层 MLP 增强器，配合逆掩码重建、消息正则化、Bootstrap 训练及自监督损失组合。

**📊 数据集**

在十个公开基准（Cora、Citeseer、Pubmed、DBLP、WikiCS、ogbn-arxiv、Sportsfit、Products、Photo、Computer）上进行评估。

**📈 对比分析**

与传统 GNN、GSSL 与 GDA 基线对比，SelfAug 在迁移性强的归纳和冷启动场景下取得最高准确率，且推理时间和内存均最优。

**⚠️ 局限性**

局限在于仅验证了同域迁移，对跨域、极端稀疏或噪声图的适应性尚未深入探究。

---

## 187. Instance-Dependent Regret for CMDPs with Step-Wise Constraints

**arXiv ID:** 2610.02520 | [PDF](https://arxiv.org/pdf/2610.02520v1)

**作者:** Qian Zuo `[一作]` (University of Edinburgh), Sattar Vakili `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在步进式安全约束下的在线学习，提出了安全方差自适应探索算法 SVAE。

**💡 创新点**

创新点在于将安全子图结构与方差分析结合，设计了可实现实例依赖 regret 与约束违规量的算法，并给出对应的下界。

**🔧 技术方法**

采用了变异置信界、基于安全子图的回溯学习、方差自适应奖金以及经验伯努利探索等技术。

**📊 数据集**

主要使用合成的 CMDP 实验验证，未使用公开真实数据集。

**📈 对比分析**

与传统无约束 RL、无方差下界及其他安全 RL 方法比较，SVAE 在低方差实例上实现了近似常数或 polylog 的 regret，并将累计违规量控制在 O(√K)。

**⚠️ 局限性**

局限性包括对安全子图可搜索性的依赖、在高维或函数逼近场景下扩展困难、需要已知安全阈值以及对极端成本间隙假设较为敏感。

---

## 188. Student-Guided Teacher Distillation for Efficient LLM Task Routing: Positioning Against Jev-Style System-1 Classifiers

**arXiv ID:** 2610.02516 | [PDF](https://arxiv.org/pdf/2610.02516v1)

**作者:** Haifeng Wu `[一作]` (PayPal), Xin Chen `[通讯]` (PayPal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套基于学生引导的教师蒸馏管线，用小型 ModernBERT 作为快速路由器和候选生成器，再用大规模 DeBERTa‑v3 零样本 NLI 教师对 Top‑k 候选进行重新排序，并通过迭代训练不断收窄候选集以降低教师标注成本。

**💡 创新点**

（1）在固定 60 类任务分类学下，利用学生的概率分布做候选集生成，区别于通用嵌入检索；（2）引入迭代学生‑教师循环，使候选集随覆盖率逐步收窄；（3）明确指出截断 Top‑k 分布不适合作为 KL 蒸馏目标；（4）提出多维度评估框架（候选覆盖、教师一致性、吞吐量/延迟）。

**🔧 技术方法**

使用 ModernBERT 作为学生分类器、DeBERTa‑v3 作为零样本 NLI 教师、Softmax+KL 蒸馏、Top‑k 候选选择、vLLM + Nginx 负载均衡、批量推理以及 Python/脚本化实验流水线。

**📊 数据集**

固定 60 类 LLM 任务分类学；300 条种子手工示例、公开软件工程请求、1,000–2,218 条人工审核的 LLM 生成合成任务示例；未使用真实用户流量。

**📈 对比分析**

与全教师标注、嵌入检索基线以及仅学生的下界进行对比；主要指标包括候选覆盖率 Coverage@k、教师一致率 Agreement、吞吐量与延迟。实验表明最佳学生（v5‑tuned）在 200 例测试集上教师一致率达 77.5%，Coverage@16 为 91–100%；部署版 v4 在 4 GPU 上实现约 5,200 请求/秒单例吞吐，单请求延迟 150–170 ms。

**⚠️ 局限性**

（1）缺乏正式的持有外部验证集，导致一致率对分布偏移高度敏感；（2）低置信度伪标签、样本重复和分布迁移影响模型性能；（3）截断 Top‑k 作为 KL 目标会引入系统偏差；（4）候选集覆盖率在小 k 下不稳健；（5）教师标注成本尚未得到完整量化。

---

## 189. Post-Training Quantization of Autoregressive Weather Models

**arXiv ID:** 2610.02511 | [PDF](https://arxiv.org/pdf/2610.02511v1)

**作者:** Ananyo Bhattacharya `[一作]`, Christiane Jablonowski `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对全球尺度深度学习气候模型DLWP（U-Net）和FCN（ViT）进行后训练量化（PTQ）以降低权重和激活位宽，实现高效推理。

**💡 创新点**

创新点在于系统评估多种PTQ策略（INT8、INT4、INT2、SmoothQuant、AWQ）对递归预测误差的累积影响，发现权重量化对预测质量影响更大，且特殊量化方法能在保持高精度的同时显著提升硬件效率。

**🔧 技术方法**

使用NVIDIA Earth2Studio和ModelOpt进行PTQ，评估RMSE/ACC并记录GPU功耗与推理时长，结合FP32基线和气候基准。

**📊 数据集**

基准数据集为ERA5再分析（全球），在四个季节性初始条件（2020年DLWP，2022年FCN）下进行48–60小时短期预测。

**📈 对比分析**

相较于FP32，PTQ模型在短期预测（≤60h）内保持了0.55–0.8的ACC和与ERA5相近的RMSE，硬件上功耗基本不变，推理时间略增但仍低于FP32，验证了量化的可行性。

**⚠️ 局限性**

局限性包括仅测试短期递归预测、未使用量化感知训练(QAT)、缺乏理想化环流案例验证物理一致性，以及对更长时间尺度或更高分辨率的泛化尚未评估。

---

## 190. Multi-Fidelity Policy Gradients Stabilize Data-Scarce Reinforcement Learning

**arXiv ID:** 2610.02505 | [PDF](https://arxiv.org/pdf/2610.02505v1)

**作者:** Xinjie Liu `[一作]` (University of Texas at Austin), David Fridovich-Keil `[通讯]` (University of Texas at Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种在数据稀缺的场景下，通过多精度数据（高精度/低精度）结合的方式，改进PPO算法实现更稳定的在线策略梯度学习。

**💡 创新点**

创新点在于：① 将低精度（模拟）数据用于构造控制变量以降低高精度梯度方差，而不引入偏差；② 设计了跨精度采样、优势估计和控制变量构造的重构方法；③ 引入预算感知机制，根据每个采样的成本动态分配高低精度样本；④ 通过监测控制变量的置信区间和方差下降，避免方差膨胀。

**🔧 技术方法**

技术主要包括：多精度控制变量（mfpg）框架、PPO的优势估计（GAE）与裁剪策略、跨环境同步与共享随机噪声、指数移动平均估计控制系数、置信区间与方差比率监控、预算最优分配（基于二阶多精度采样分配）。

**📊 数据集**

在仿真数据集上使用Isaac Lab中的Unitree H1 humanoid和ANYmal-D quadruped，分别在平坦与崎岖地形下进行机器人步态学习；在真实机器人数据集上使用Franka arm完成立方体拾取任务，配合ManiSkill仿真环境。

**📈 对比分析**

与多种基线（PPO、SAC、DARC、PAR、Co‑training、mfpg‑REINFORCE等）比较，mfpg‑PPO在大部分任务与数据稀缺场景下均优于单精度PPO，性能提升可达相当于使用16倍高精度数据；在最困难的H1崎岖地形和最小高精度预算下亦实现了与单精度PPO相当的表现。

**⚠️ 局限性**

局限性包括：① 需要手动设置同步策略和控制变量的拟合单元，可能对不同任务不适配；② 低精度和高精度统计均可能存在偏差，尤其在不完全同步的lf批次中；③ 预算分配基于方差模型的理论假设，实际中可能不完全符合；④ 仅在GPU并行模拟与物理机器人两类环境验证，其他类型的多精度设置仍需进一步研究。

---

## 191. Oracle headroom without signal: null-calibrated evaluation of candidate selection for thermal heart rate estimation

**arXiv ID:** 2610.02561 | [PDF](https://arxiv.org/pdf/2610.02561v1)

**作者:** Mohammad Rakibur Rahman `[一作]`, Constantino Álvarez Casado `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本研究分析了热成像心率估计中使用“oracle”选择方法的误差，并建立了无信息下的基准模型；

**💡 创新点**

创新点在于提出了无信息候选的order statistics基准以及匹配空白（null）评估框架，用以区分候选多样性带来的误差与真正的生理信息；

**🔧 技术方法**

采用了热成像视频多候选心率估计、频段滤波、不同频谱估计器（Welch、FFT、MUSIC、峰间距）、SQI量化及order statistics分析等技术；

**📊 数据集**

使用iBVP热成像数据集，包含96个热像录制（共6,816个10秒窗口），并以耳部PPG作为参考心率；

**📈 对比分析**

将oracle误差（0.91 bpm）与最佳固定设置（10.74 bpm）、SQI选择（18–28 bpm）及恒定预测（8.61 bpm）进行比较，显示oracle误差与无信息基准（0.40 bpm）极为接近，说明大部分oracle优势可归因于候选数目而非真实生理信息；

**⚠️ 局限性**

局限性包括仅使用固定ROI平均、仅针对热成像数据、未对所有候选生成完整null分布、候选覆盖率不均以及未检验不同ROI或学习模型的潜在改进。

---

## 192. Compound AI System Reliability: A Failure Taxonomy and Resilience Pattern Catalog from 150 Production Incidents

**arXiv ID:** 2610.02503 | [PDF](https://arxiv.org/pdf/2610.02503v1)

**作者:** Rudrendu Kumar Paul `[一作]` (Boston University), Sourav Nandy `[通讯]` (University of Texas at Austin)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过分析150个生产事故，系统梳理出23种以组件边界为核心的失败模式，并在受控6组件测试平台上进行故障注入实验，验证并量化了5种可落地的恢复模式，证明至少采用3种模式可将平均恢复时间缩短71%

**💡 创新点**

首次从检索、生成、工具、编排、集成等全链路层面构建统一的失败模式分类，并为每种模式提供实验可测的恢复策略，且将这些模式和实验结果公开为实践资源

**🔧 技术方法**

采用故障注入、受控实验环境、语义质量监控（如余弦相似度、事实一致性检测）、电路断路器、输出质量门、组件隔离、语义校验器、强类型接口等技术手段

**📊 数据集**

使用150条来自12个开源 Compound AI 项目（如LangChain、LlamaIndex等）和53条企业内部部署事故的事故报告，作为案例库；故障注入实验在自建的6组件系统上执行

**📈 对比分析**

与无结构化监控基线对比；每个恢复模式在100次实验中测量链路深度、扩散范围、MTTR等指标，结果显示：电路断路器链路深度下降89%，输出质量门捕获73%轻度失效，组件隔离扩散范围下降64%，语义校验率81%，类型接口成功率92%；组合3+模式时MTTR从28.7分钟降至8.4分钟，提升71%

**⚠️ 局限性**

企业事故样本规模有限；基准仅为无结构监控，未与标准重试/回退基线对比；实验仅在单一6组件架构，无法覆盖更大规模或人机交互场景；缺乏实时生产验证，未评估实际运行开销；聚焦技术层面，未涵盖组织管理失效

---

## 193. Answering clinicians' questions over trial evidence tables with verifiable, feedback-driven language models

**arXiv ID:** 2610.02576 | [PDF](https://arxiv.org/pdf/2610.02576v1)

**作者:** Manan Roy Choudhury `[一作]` (Arizona State University), Vivek Gupta `[通讯]` (Arizona State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了FD‑SCoPE系统，能让临床医生用自然语言查询临床试验证据表，系统可回答已记录字段的查询，也能推导表中未存储的属性，并对答案进行可视化与可验证。

**💡 创新点**

创新点在于：①将查询与推导分离，先用SQL精准筛选试验，再由语言模型推导隐藏属性；②引入可审计的“已验证程序”与“已批准查询”存储，专家纠错后能生成可复用的规则；③实现了对查询过程、所选试验和推导规则的可追踪与可复核，增强可信度。

**🔧 技术方法**

技术包括：基于大型语言模型（Qwen3.8‑27B、Gemma‑3‑27B）多阶段提示（路由、分解、SQL生成、修复、规划、程序诱导、逐行推导等）；利用词表、近似匹配和示例检索实现术语归一化；通过沙箱执行Python表达式进行属性推导；并结合安全门控与范围检查保证数据库只读。

**📊 数据集**

使用IOTOX活证据资源的免疫检查点抑制剂（ICI）临床试验表，包含159条记录、32字段；通过人工制定的1,500个基准查询、1,500个需要推导属性的查询以及140个临床风格任务进行评估。

**📈 对比分析**

与七种文本到SQL基线、四种推导基线比较，FD‑SCoPE在记录字段查询上精确匹配率58.9%（比最佳对照55.2%高3.7个百分点），在需要推导属性的任务中Derived‑Value F1为77.7%（比最佳73.4%高4.2个百分点）。在临床风格任务上100%正确率。通过专家反馈（299个问题）提升未见问题的Derived‑Value F1从77.9%升至84.9%。

**⚠️ 局限性**

局限包括：仅在单表癌症试验数据上验证；未直接评估真实临床医生使用体验；模型对拼写错误敏感；反馈模拟为参考答案，未量化专家纠错成本；系统在多表联合、外部数据集上的表现未知；安全性测试基于预设规则，真实攻击场景未充分验证。

---

## 194. Beaver: Elastic GPU Sharing between ML and Latency-Critical vRAN Workloads

**arXiv ID:** 2610.02522 | [PDF](https://arxiv.org/pdf/2610.02522v1)

**作者:** Yuncheng Yao `[一作]` (Duke University), Tingjun Chen `[通讯]` (Duke University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为Beavers的GPU共享系统，实现在latency‑critical vRAN与best‑effort ML 工作负载间的弹性共享，保障vRAN 99.9th 百分位延迟不超过1.5 ms，同时保持LLM推理吞吐率。

**💡 创新点**

创新点在于三项协同机制：①工作负载感知SM分配器可预测每个时隙所需最小SM；②按槽时钟快速重划SM分区，利用预创建的CUDA绿上下文实现子毫秒级切换；③对已编译ML核进行PTX级重写，动态调节HBM带宽，避免内存争用。

**🔧 技术方法**

采用CUDA绿上下文、GPU PTX重写与共享位图、实时SM重分配、基于统计的延迟模型与SM需求预测，以及HMM/线性回归预测p99.9延迟。

**📊 数据集**

使用NVIDIA Aerial 5G L1流水线、真实基站上行/下行轨迹（Madrid、O‑RAN TRACTOR）、多种LLM推理框架（vLLM、Llama‑3.3‑70B、Mistral‑Small‑24B）、单轴压力测试（FMA burner、HBM flooder）等数据集。

**📈 对比分析**

与现有GPU共享系统（YinYangRAN、CAORA、RAN‑LLM、Gandiva、Orion、TGS、MPS）相比，Beavers在不出现任何截止时间错失的前提下，保留74%~85% LLM吞吐率；在多GPU平台（H200、A100、GB10、GH200）以及下行链路（375 µs）场景均保持1.5 ms/375 µs延迟。

**⚠️ 局限性**

局限性包括仅支持单GPU共享、需预创建GC对、对不支持PTX或动态生成核的工作负载无效、以及未覆盖多租户并行协作或跨GPU资源调度。

---

## 195. FDP: The Data Placement Promise of Modern NVMe SSDs

**arXiv ID:** 2610.02676 | [PDF](https://arxiv.org/pdf/2610.02676v1)

**作者:** Sijie Lan `[一作]` (Pennsylvania State University), Vivek Shah `[通讯]` (Samsung)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了NVMe Flexible Data Placement（FDP）SSD在企业级数据中心中的写放大缓解效果，并将其应用于RocksDB和MySQL两大开源存储系统，通过文件系统生命周期提示或I/O Passthru实现数据分层写入；

**💡 创新点**

提出将RocksDB LSM层级与MySQL写入模式与FDP的RUH映射结合，利用现有文件系统生命周期提示实现非侵入式数据分区，展示在现有SSD上即可获得显著WAF降低的跨层机制；

**🔧 技术方法**

利用Linux 6.9内核的I/O Passthru接口、文件系统级生命周期提示（lifetime hints）、FDP SSD的RUH、RocksDB的VFS API以及自研TorFS文件系统绕过方案；

**📊 数据集**

使用Synthetic YCSB、TPC‑C和多种随机/顺序混合FIO工作负载，以及对3.84 TB FDP SSD的实测数据；

**📈 对比分析**

与传统NVMe SSD基线对比，单租户和多租户场景下，RocksDB在FDP上实现WAF近乎1，I/O吞吐提升3.3×；MySQL在FDP上WAF下降约10–15%，并提升吞吐；通过FIO微基准验证RUH隔离在不同负载下的效果；

**⚠️ 局限性**

受限于FDP SSD当前可用的RUH数量（仅8个），过多租户或写入模式差异导致RUH共享时WAF仍升高；文件系统生命周期提示粒度仅到文件级，无法细粒度控制LBAs；I/O Passthru实现复杂度高，需自研层；对随机写的隔离效果随写入模式变为全随机时下降。

---

## 196. DataWeave: Deploying Human-LLM Analytics for Exploratory Structured Data Analysis

**arXiv ID:** 2610.02679 | [PDF](https://arxiv.org/pdf/2610.02679v1)

**作者:** Raquib Bin Yousuf `[一作]` (Virginia Tech), Naren Ramakrishnan `[通讯]` (Virginia Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并部署了一个名为 DataWeave 的交互式、基于 LLM 的数据新闻分析系统，支持记者在 IPEDS 数据集上进行探索性分析，结合对话式交互、schema grounding、分析规划、SQL 生成与可追溯执行。

**💡 创新点**

创新点在于将 LLM 作为可视化、可修正的合作伙伴而非单纯的答案生成器，引入目标约束的 schema grounding、可检查的执行日志、受限恢复策略以及基于 ReAct 的动态代理调度，提升了高风险结构化分析的可信度与可解释性。

**🔧 技术方法**

核心技术包括：大型语言模型（如 GPT‑4）、ReAct 代理框架、schema grounding 与知识检索、SQL 代码生成与执行、以及对话式前端与会话管理。

**📊 数据集**

使用的主要数据集是美国教育部的 Integrated Postsecondary Education Data System（IPEDS），覆盖多年的入学、财务、人口统计和结果表。

**📈 对比分析**

通过与基线手工实现的原型比较，Agentic Exploration 与 Agentic RAG 在 400 题 IPEDS Trend Generator 基准上分别达到了 70% 与 72.8% 的相对误差 1% 内的正确率，执行延迟与成本显著下降，成本从 $0.0601 降至 $0.0065/题。

**⚠️ 局限性**

局限性包括对 LLM 推理误差的依赖、对 schema 漂移的适应需要持续维护、代理策略的复杂性导致对话路径不可预测、以及在高频交互场景下对系统性能与集成成本的挑战。

---

## 197. LEAP: Learning Efficient Action Proposals For LLM Agents

**arXiv ID:** 2610.02670 | [PDF](https://arxiv.org/pdf/2610.02670v1)

**作者:** Zhen Xu `[一作]` (University of Chicago), Ce Zhang `[通讯]` (University of Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对LLM代理在行动推测中引入LEAP小型动作草稿模型，通过训练使其精准预测目标模型动作，从而在单GPU上将端到端时间缩短60%

**💡 创新点**

创新点在于构建延迟框架评估动作草稿的速度收益与成本，并用目标动作序列训练微型草稿模型，既保持低成本又大幅提升准确率

**🔧 技术方法**

使用LoRA微调的0.6B草稿模型、并行提议‑验证‑提交循环以及目标模型的推理验证

**📊 数据集**

使用OpenAGI、TaskBench、τ^2‑bench和BFCL四个工具驱动基准，以及Qwen3‑32B与Gemma‑4‑31B‑it两种大型目标模型

**📈 对比分析**

通过与无训练草稿、目标自提议和不同模型大小对比，测得在四个基准上从1.10到1.63倍的整体速度提升，任务成功率基本不变

**⚠️ 局限性**

局限包括只能同步提议‑验证模式、对工具延迟敏感、对并行解码对目标输出的影响未完全评估，且在工具执行时间较长时收益受限

---

## 198. World Action Modeling with Progressive Visual Planning

**arXiv ID:** 2610.02508 | [PDF](https://arxiv.org/pdf/2610.02508v1)

**作者:** Fei Zhang `[一作]` (Shanghai Jiao Tong University), Amir Bar `[通讯]` (Imperial College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 ProWAM，一种把进度条件的稀疏子目标预测与低层动作生成统一在同一生成式视频专家和动作专家中的世界行动模型，支持长时域闭环控制。

**💡 创新点**

创新点：
- 进度条件的稀疏子目标序列预测，用相对进度 r∈[0,1] 把整条轨迹拆分成若干可视化子目标；
- 将子目标槽嵌入视频生成路径，并通过非对称注意力（MoT）将视觉子目标直接喂给动作专家；
- 两阶段训练：先用无动作视频进行视频专家预训练，再与动作专家联合微调；
- 缓存子目标特征，仅需一次视频推理，显著降低推理成本。

**🔧 技术方法**

技术：
- Diffusion Transformer（DiT）视频专家与动作专家；
- 进度条件的 adaLN（progress‑conditioned LayerNorm）与正弦嵌入；
- Mixture‑of‑Transformers（MoT）共享注意力结构；
- Flow‑matching 目标函数，分别训练视频、子目标和动作；
- 子目标缓存策略实现高效推理。

**📊 数据集**

数据集：
- 大规模无动作视频（公开动作自由视频集）用于视频预训练；
- 机器人演示数据（Franka 手抓、DROID 物理任务）用于微调；
- 评测使用 LIBERO、LIBERO‑Plus、RoboTwin、RoboCasa365；
- 真实世界零样本测试在 DROID 物理任务上。

**📈 对比分析**

对比与性能：
- 在 LIBERO‑Plus（零样本）达到 85.8%，超过所有 VLA 与 WAM 基线；
- 在 RoboTwin 随机化（OOD）任务中获得 75.7%，比最佳基线高 20%；
- 在 RoboCasa365 总体 48.1%，排名第二；
- 真实世界零样本任务平均成功率 70%，比 DreamZero 提升 15%。
- 相比传统全视觉或无视觉 WAM，ProWAM 在 OOD 情况下显著提升且推理成本降低（≈84% FLOPs 降低）。

**⚠️ 局限性**

局限性：
- 子目标生成质量对视觉先验高度依赖，模糊或错误的子目标会导致失败；
- 需要大量无动作视频进行预训练，若预训练域与目标任务差异大则效果衰减；
- 目前评测主要在抓取与装配类任务，尚未验证在更复杂动力学或多机器人场景的泛化；
- 对子目标数目和进度调度仍需手工设计，缺乏自动化策略；
- 虽然推理成本降低，但在高频实时控制下仍存在一定延迟。

---

## 199. A GHOST in Long-Horizon Agents: Governance Hazard from Overlooked Safety Constraints across Turns

**arXiv ID:** 2610.02664 | [PDF](https://arxiv.org/pdf/2610.02664v1)

**作者:** XinPeng Shen `[一作]`, Haoxiang Deng `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了长期交互中历史安全约束被忽视导致的GHOST故障，并提出STAR-Guard双层防御来阻止此类违规执行

**💡 创新点**

首次正式定义GHOST并通过条件风险理论证明其几乎必然发生；设计了可执行、环境基准SCARBench；提出结合语义安全约束恢复与预执行审计的STAR-Guard方案

**🔧 技术方法**

条件风险模型、LLM驱动的安全约束提取与语义恢复、预执行规则审计、工具调用与执行轨迹监测、基准评测框架

**📊 数据集**

SCARBench benchmark（412个可执行实例，覆盖6类工具使用场景）以及多模型（GPT‑5.5、DeepSeek‑V4‑Pro、GLM‑5.2、Kimi‑K2.6、Qwen‑3.6、Qwen3.5‑4B、Llama‑3.1‑8B）

**📈 对比分析**

与无防御、单层恢复、单层审计及多种基线（Prompt Reminder、BM25、DeCRIM、TrustAgent、DVR、LIGHT Three‑Memory、VerIFY‑Summarize）进行对比；STAR‑Guard在GPT‑5.5上提升安全完成率至94%，GHOST率降至0；在其他模型上也实现显著提升（安全完成率提升7–34个百分点，GHOST率降至1–2%）

**⚠️ 局限性**

仍依赖LLM准确检索和恢复历史约束，极长上下文或多代理情境下效果未验证；预执行审计可能阻断合法动作导致任务失败；理论假设（如残差风险下界）在真实交互中可能不完全成立

---

## 200. Large Language Continuous Diffusion Models

**arXiv ID:** 2610.02665 | [PDF](https://arxiv.org/pdf/2610.02665v1)

**作者:** Zhihan Yang `[一作]` (NVIDIA), Morteza Mardani `[通讯]` (NVIDIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了3B/8B规模的连续扩散语言模型，采用16维低维潜在空间并块级训练；

**💡 创新点**

创新点包括：1）实现大规模连续扩散的块级训练框架；2）利用预训练AR权重进行warm‑start并加入AR正则化；3）在逆扩散中结合Classifier‑Free Guidance、Score Temperature和Self‑Conditioning进行轨迹引导；4）采用SUSReg正则化嵌入以提升少步性能；5）改进的Parallel Decoding Distillation实现低NFE生成；

**🔧 技术方法**

技术手段：高斯扩散/ODE/SDE轨迹、块级自回归注意力、CFG、ST、自条件化、DDPM/DDIM采样、SUSReg嵌入正则化、PDD蒸馏；

**📊 数据集**

训练使用公开Web文本，tokenizer为Mistral‑NeMo（V=131072），评估基准包括GSM8K、Minerva、Math500、AIME、HumanEval、MBPP、HumanEval+、MBPP+等数学与编程任务；

**📈 对比分析**

与离散扩散模型（Nemotron‑Labs‑Diffusion、LLaDA、Dream、SDAR）及AR基线对比，pass@1在多数任务与离散模型相当或略优；在少步（4/8步）场景下通过SUSReg和PDD提升性能，达到可比效果；

**⚠️ 局限性**

局限性：仍需大量计算资源；在极低NFE（≤4）下性能显著下降；依赖AR预训练；嵌入维度可能导致表达不足；在非常长序列或复杂任务上未充分验证。

---

## 201. CuBEs: Culturally-Situated Behavioral Evaluations and the Limitations of Culture-Blind LLM Judges

**arXiv ID:** 2610.02622 | [PDF](https://arxiv.org/pdf/2610.02622v1)

**作者:** Hoda Ayad `[一作]` (University of Washington), Abhishek Mukherji `[通讯]` (Centific)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了Culturally-situated Behavioral Evaluations (CuBEs) 框架，针对多文化背景下的LLM行为进行系统化评估

**💡 创新点**

创新点在于将文化情境直接注入行为评估流程，构建跨12种文化的人工标注数据集，并展示文化差异对行为表现的显著影响

**🔧 技术方法**

采用自动化代理评估管线（改进自Anthropic Bloom），使用Claude Sonnet 4.5生成情景，Gemini 3 Flash进行用户模拟，Gemma 3等模型进行判定，结合多模型对齐和自动判分

**📊 数据集**

构建了涵盖12个英语国家、3个社会领域（工作、健康、家庭）以及6种风险行为的人工标注数据集，共计1,860条标签，用于对齐与评估

**📈 对比分析**

通过与文化无关的基准对比，发现文化情境下的行为出现率显著波动；CuBEs可将偏差从6.06点以上缩小到平均2.72点，表明文化化评估更精准；模型评估显示Anthropic系列表现最佳，DeepSeek表现最差

**⚠️ 局限性**

主要限制包括仅使用英语提示，无法覆盖非英语文化；将国家视为单一文化代表导致内部多样性被平滑；仅关注风险行为，未涵盖礼貌等正向行为的文化差异

---

## 202. Batched Speech Decisions Without Decoding: Single-Token Supervision Lets a Frozen LLM Hear Beyond the Transcript

**arXiv ID:** 2610.02638 | [PDF](https://arxiv.org/pdf/2610.02638v1)

**作者:** Jie Jin `[一作]`, Xiaowen Zhang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过将ASR编码器输出映射到冻结LLM，并使用单词级别的单步读取实现全双工语音代理的决策，避免自回归解码；

**💡 创新点**

提出无解码批量决策读取框架，利用答案-令牌监督让LLM捕获说话者的性别与情感，并实现模块化的跨层注意融合与投影；

**🔧 技术方法**

使用冻结的Qwen3-32B LLM与Qwen3-ASR-0.6B编码器，配合cross‑attention fusion、投影层、单词级next‑token读取、前缀共享批量推理、答案‑令牌交叉熵与混合蒸馏技术；

**📊 数据集**

基于WenetSpeech、GigaSpeech、LibriSpeech、Common Voice、CoVoST、AISHELL‑1、ESD、CREMA‑D、Whisper‑large‑v3‑turbo等公开语料，构建100条多选spoken‑QA（qa100）、ZJU音频基准、Easy‑Turn等评测集；

**📈 对比分析**

与传统级联解码器对比，单词读取在同一GPU上提升约17–20倍吞吐量，延迟下降≈17×，在qa100、ZJU‑ML、Easy‑Turn等任务上与文本转录相差≤1%内容准确度，性别/情感准确率从55%/28%提升至≈90%，但在混合任务中内容分数略有下降；

**⚠️ 局限性**

依赖冻结模型，无法在同一模型中同时完美兼顾内容与多任务；在情感/性别训练中可能出现信息泄漏或训练失衡；长上下文共享仍需复杂KV缓存管理；尚未评估实时流式输入与极低延迟部署场景。

---

## 203. RaBitQ-SSD: Split Codes and Pipelined I/O for SSD-Resident Vector Search

**arXiv ID:** 2610.02652 | [PDF](https://arxiv.org/pdf/2610.02652v1)

**作者:** Yuexuan Xu `[一作]` (Nanyang Technological University), Cheng Long `[通讯]` (Nanyang Technological University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向SSD的近似最近邻搜索系统 RaBitQ-SSD，结合可配置的前缀 1-bit RaBitQ 代码与异步 I/O 管线，实现了高吞吐、低 SSD 访问量的向量检索。

**💡 创新点**

创新点在于：①仅在内存中保留 RaBitQ 代码前缀，仍能推导无偏距离估计和概率误差上界；②利用该上界实现细粒度的 SSD 页面裁剪；③设计了异步搜索管线，按页面下界排序并使用 α‑分位阈值削减无效读请求，最大化 I/O‑计算重叠。

**🔧 技术方法**

核心技术包括 RaBitQ 量化与概率误差分析、IVF 细化量化器（IRQ）、SIMD 高速扫描（FastScan）、异步 NVMe I/O 调度、α‑分位阈值裁剪、以及基于页面下界的读取优先级排序。

**📊 数据集**

实验数据集覆盖从 5 M 到 10 B 的七大规模集合：OpenAI‑5M、Wiki‑10M、DPR‑100M、LAION‑100M、YFCC‑100M、DataComp‑1B、DINO‑10B。

**📈 对比分析**

与 DiskANN、Starling、PipeANN、SPANN、AlayaLaser 等主流 SSD‑驻留索引对比，RaBitQ‑SSD 在 90% recall 下实现了最高吞吐（1.74× 于 1 B 集合，3.9× 于 10 B 集合），SSD 页读取量下降至 3.8×，索引尺寸缩小至 7×，构建时间比最快图索引快 5×以上。

**⚠️ 局限性**

局限性包括：①对 IVF 分区的依赖，难以处理高度动态或非均匀分布的查询；②前缀长度越小，误差上界越宽，可能导致更多 SSD 读请求；③在极端大规模下仍受 SSD 带宽限制，且未对动态更新或在线学习场景进行评估。

---

## 204. Mind the Refinement Gap: When Safe High-Level Robot Plans Produce Unsafe Executions

**arXiv ID:** 2610.02662 | [PDF](https://arxiv.org/pdf/2610.02662v1)

**作者:** Stabak Das `[一作]` (Prairie View A&M University Texas A&M University Systems), Lijun Qian `[通讯]` (Prairie View A&M University Texas A&M University Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

探讨语言驱动机器人系统中，监视器对高层动作序列的安全判定是否与实际图形驱动的导航和隐式动作效果一致；通过对RoboGuard的评估，发现表层计划被接受而图形精细化轨迹被拒绝的“假安全”情况；提出图形追踪细化作为轻量级缓解与诊断手段。

**💡 创新点**

首次系统性区分动作抽象导致的假安全与规范生成错误，并提供基于语义图的轨迹细化机制，确保监视器判定与实际执行相匹配；同时构建了包含多种抽象失配类型的对照案例集。

**🔧 技术方法**

利用线性时序逻辑（LTL）与Büchi自动机（Spot），结合RoboGuard的安全监控框架、SPINE自然语言规划器与语义图；对计划与细化轨迹进行LTL监视。

**📊 数据集**

使用人工构造的28条固定计划（包含12个目标抽象案例与16个对照案例）以及14条基于SPINE的自然语言任务与四个配对语义图场景。

**📈 对比分析**

通过对表层计划和细化轨迹分别进行相同LTL判定，记录“假安全”与阻断结果；在受控评估中12个目标案例全部出现假安全，16个对照案例正确；在端到端评估中5/14出现假安全，3/14正确阻断；细化轨迹检查平均耗时从0.54ms提升至0.69ms，性能影响极小。

**⚠️ 局限性**

仅针对离散语义图与离散轨迹，未考虑连续运动、感知误差、碰撞规避及实际机器人执行；细化需先确定执行路径，若未知需检查所有可行细化或放弃；接口与语义兼容性问题在部分案例中导致语法错误。

---

## 205. Learning When to Commit from Partial Speech for End-to-End Simultaneous Speech Translation

**arXiv ID:** 2610.02612 | [PDF](https://arxiv.org/pdf/2610.02612v1)

**作者:** Hieu Hoang `[一作]` (Microsoft), Amittai Axelrod `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

基于自监督的语音语言模型，构建前缀学习框架，使模型能在不依赖文本或人工翻译的情况下完成无闪烁的实时语音翻译。

**💡 创新点**

创新点包括：①直接使用完整语音与其前缀的翻译一致性作为前缀监督；②引入合成边距（synthesis margin）调控前缀监督密度；③对比单回合强制前缀和多回合追加仅写解码两种流式策略；④通过置信度阈值动态控制发译时机，实现质量‑延迟可调。

**🔧 技术方法**

技术手段主要为：Qwen2.5‑Omni‑7B 语音语言模型 + LoRA 微调；自监督全语音翻译 + 前缀监督；合成边距策略；置信度阈值控制；单回合与多回合解码实现。

**📊 数据集**

训练数据：LibriSpeech‑100、LibriSpeech‑360、Common Voice；验证/测试数据：FLEURS、CoVoST 2。

**📈 对比分析**

比较方法：在同一音频块和语言方向下，对基准模型与经过前缀学习的单回合/多回合模型进行质量‑延迟（COMET‑AUC）和置信校准（ECE）评估。结果显示：前缀学习显著提升质量‑延迟前沿；多回合解码在低延迟下更优；置信度机制提供最广泛的可调范围；合成边距在小幅度下可进一步降低延迟，但大边距会降低翻译质量和校准。

**⚠️ 局限性**

局限性：仅评估单一模型体系与三种目标语；未测试不同口音、领域、真实对话；仅使用固定2 s块分割；未评估自适应分段、实时网络延迟、计算资源占用；自动评测（COMET）缺乏人工评估；校准仅针对基准模型推断上下文，无法泛化；单回合与多回合的对比同时改变了因果表示和训练样本结构，难以单独归因。

---

## 206. Time Series Forecasting Benchmarks Need Scenario-Grounded Stress Testing

**arXiv ID:** 2610.02608 | [PDF](https://arxiv.org/pdf/2610.02608v1)

**作者:** Yuyang Zhao `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Hao Xue `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种面向情境的时间序列预测（TSF）评估框架，指出现有平均误差评估无法捕捉真实部署中的结构性失效，进而设计了包含失败操作、情境参数和难度得分的结构化测试实例，并给出了四类失效结构与六条设计原则。

**💡 创新点**

创新点在于：①将测试实例从单一的 (X, Y) 对转变为包含失败操作 ϕ、情境参数 Θ 与难度 δ 的五元组；②引入可解释、可组合、因果化的情境参数；③系统性划分四类结构性失效（局部事件、跨变量依赖、机制耦合、级联传播）及对应的评测设计原则；④强调情境参数的可视化与透明度，以实现模型鲁棒性因果解释。

**🔧 技术方法**

主要技术是基于理论框架的设计与方法论阐述：使用参数化失败操作符 ϕ(·;Θ)、难度映射 κ(Θ)、以及结构化元组定义；并未实现具体算法，而是提供评测流程与设计规范。

**📊 数据集**

文中未给出实验数据集；讨论建议在公开的 TSF 基准（如 M3、Monash、LTSF、GIFT‑Eval、TDB 等）上构造基线序列，并通过情境库生成结构化失效实例。

**📈 对比分析**

比较方法建议按情境参数 Θ 的不同子集和难度等级构造评测电池，绘制鲁棒性曲线；性能评价以误差随 Θ 变化的趋势来衡量，侧重模型在结构性失效下的表现而非单一平均误差。

**⚠️ 局限性**

局限性包括：①缺乏具体实现与实测结果，框架仍处于概念阶段；②情境参数的设计与校准需要领域专家参与，难以统一标准；③可能面临 Goodhart 定律风险，模型对已公开 Θ 可能过拟合；④情境库构建与难度标注工作量大，兼容现有基准的难度；⑤未解决跨域情境的可迁移性与统一评测尺度。

---

## 207. Rateless Nested Lattice Codes for Secure Cooperative V2X Broadcast over Fading and Erasure Channels

**arXiv ID:** 2610.02605 | [PDF](https://arxiv.org/pdf/2610.02605v1)

**作者:** Pegah Sharifi `[一作]` (Amirkabir University of Technology), Chen Feng `[通讯]` (University of British Columbia)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种面向车联网广播的无固定速率 Construction‑D’ Raptor 余弦格点编码方案，支持在多路径衰落、符号擦除与被动窃听者存在的环境下安全可靠地进行多方广播。

**💡 创新点**

创新点在于首次将 Raptor 余子码与 QC‑LDPC 前置码、Construction‑D’ 多层格点结构、嵌套格点物理层安全（PLS）与 DSSS 隐蔽、以及区块链辅助的译码可靠性评分融为一体，形成了一套完整的车联网安全广播框架。

**🔧 技术方法**

使用的核心技术包括：Raptor 冲洗码（LT+QC‑LDPC）、Construction‑D’ 格点构造、层级 Belief‑Propagation 与多阶段解码、嵌套格点形状与直接序列扩频、以及区块链记录的译码信誉表。

**📊 数据集**

实验数据来源于对城市 V2X 侧链的 Block‑Rayleigh 衰落与随机擦除模型进行 Monte‑Carlo 仿真，参数取自典型的 ITS 频段（5.9 GHz、10 MHz）与车辆速度范围；未使用真实车载实验数据。

**📈 对比分析**

与传统 LDLC、LDPC 等匹配长度格点编码以及 H‑ARQ 固定块码在 AWGN 与衰落信道上进行对比，结果显示在 1.5 dB 平均 VNR、0.15 擦除率下，Raptor 格点实现的误块率仅 0.3%，而 LDLC、LDPC 分别为 1.2% 与 2.4%，且在 8 dB 信噪比劣势下，窃听者成功率仅 62%。

**⚠️ 局限性**

主要局限包括：采用简化的 Block‑Rayleigh 与擦除信道模型、使用保守的链路层抽象来估计 Raptor 的累计效能、未对区块链信誉系统进行定量性能评估，以及缺乏实车实验验证。

---

## 208. GRAFT: Growing Agglomerative Foundation Models via Continual Teacher Distillation

**arXiv ID:** 2610.02597 | [PDF](https://arxiv.org/pdf/2610.02597v1)

**作者:** Zhenghao Zhao `[一作]` (University of Illinois Chicago), Yelin Kim `[通讯]` (Amazon)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 GRAFT 框架，实现单一 ViT-B 编码器通过持续多教师蒸馏逐步吸收多种视觉基础模型的能力。

**💡 创新点**

创新点在于：①使用前一阶段学生作为冻结的“保留教师”，只需与当前教师联合蒸馏即可避免重新蒸馏所有教师；②引入教师特定读取令牌（TSRT）和几何无关关系损失（Geometry Agnostic Relational Loss）解决教师异构性冲突。

**🔧 技术方法**

采用的技术包括：连续多教师蒸馏（continual MTKD）、教师特定读取令牌（TSRT）、几何无关关系损失、轻量 MLP 投影器、ViT‑B/14 backbone 以及温度软化的图像‑文本相似性对齐。

**📊 数据集**

使用的数据集包括：DataComp‑12M、ImageNet‑21k、Mapillary、Google Landmarks 等用于蒸馏；评估则使用 ImageNet‑1k、ADE20K、BEDLAM、AGORA、ARKitScenesV2、COCO‑5K、Flickr‑30K 等。

**📈 对比分析**

与 AM‑RADIO、DUNE、SAK、Theia 等聚合模型对比，GRAFT 在 5 个能力族（图像理解、2D 细粒度预测、3D 关节姿势、3D 视觉、视觉‑语言）上实现单模型覆盖；在各子任务上与专门模型相当或略低，同时在持续加入教师时保持低干扰。

**⚠️ 局限性**

局限性包括：对教师规模与数据差异仍敏感；尚未验证极大模型或跨域扩展；缺乏对非视觉任务的泛化；需手动决定教师顺序与 TSRT 数量。

---

## 209. Effects of a Behavioural Commitment Scheme on Study Regularity in a Self-Paced Learning Platform

**arXiv ID:** 2610.02595 | [PDF](https://arxiv.org/pdf/2610.02595v1)

**作者:** Meenakshi V. `[一作]`, S. R. S. Iyengar `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在自学平台上实现绑定的两小时学习窗口预订机制，让学习者提前预订后才可进入课程，并通过观看遥测验证窗口内学习，既形成学习承诺又提供平台容量预测。

**💡 创新点**

将预订行为同时视为学习者自我调节承诺与平台容量规划输入，并使其成为默认且可验证的机制，证明绑定调度比单纯提示更能提高学习规律性。

**🔧 技术方法**

使用 MERN 堆栈（MongoDB, Express, React, Node）与 Google Cloud Run 无服务器基础设施；通过视频播放器每15秒一次的观看遥测验证窗口内学习；实现预约日志、访问门控、时长预算、完成验证与奖励循环。

**📊 数据集**

数据集包含 946 名学习者的 MERN Web 开发课程，1,862 个可验证的预订，约 400,000 个遥测会话；同一学习者还完成了前置的 AI 基础课程，用作对照。

**📈 对比分析**

采用同一学习者前后对比和差分法（DiD）评估效果；已预订学习者周活跃天数从 1.01 提升至 1.33（+0.32），对照课程平均提升 +1.37 天/周；窗口占用率 86.8%；预订日志与实际并发负载相关系数 0.77，解释 59% 方差。

**⚠️ 局限性**

研究为准实验，学习者自选参与导致自选偏差；mid‑course 引入时段限制样本，跨课程差分受限；遥测仅覆盖视频播放器，无法评估学习成效；未实际测算成本节省；仅在单门课程、单项目、25 天内验证，外推性有限。

---

## 210. Spend Teacher Tokens Where They Matter: Success-Referenced On-Policy Distillation

**arXiv ID:** 2610.02678 | [PDF](https://arxiv.org/pdf/2610.02678v1)

**作者:** Xiang Chen `[一作]` (Tongji University), TanLin Li `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段预查询路由方法（SR-OPD），在不改变标准OPD目标的前提下，利用学生自身的成功推理结果作为参考，先筛选仅包含成功与失败推理的提示，再根据成功推理轨迹与失败推理轨迹的持久发散度（PDA）和教师输入成本决定只为一条失败推理请求教师监督；

**💡 创新点**

创新点在于将教师监督分配转化为预查询路由问题，首次将同一提示下的成功推理轨迹用作教师-free参考，通过持久发散度衡量失败推理的“重要性”，并结合教师输入成本实现显著节约；

**🔧 技术方法**

技术包括：学生多次生成rollout、基于隐状态的轨迹对齐与PDA计算、成本感知分数S_n= A_n·C_min/C_n、两阶段路由决策与标准OPD无缝集成；

**📊 数据集**

使用三对教师-学生模型（Qwen3-4B→Qwen3-1.7B、Skywork-OR1-Math-7B→DeepSeek-R1-Distill-Qwen-1.5B、Granite-3.3-8B→Granite-3.3-2B），以及六大数学推理基准（AIME24/25、AMC23、HMMT24/25、MATH‑500）训练和评估；

**📈 对比分析**

与Vanilla OPD、TA‑OPD、TLR‑OPD对比；在一次通行端到端设置下，SR‑OPD仅使用3.46%–5.02%教师输入token，宏观平均分与Vanilla OPD相差不超过0.1分；在匹配5%教师预算的受控实验中，提示路由与成功参考滚动路由均能提升性能，证明设计有效；

**⚠️ 局限性**

局限性包括：需可靠的验证器，无法处理全成功或全失败提示；教师成本仅计序列化输入token，未反映实际计算/壁钟时间；实验仅覆盖数学推理任务，泛化能力尚待验证；在不同训练种子下，部分子组件的效果仍为方向性，未达到统计显著性；

---

## 211. Asterism: Exploring and Synthesizing Scattered Observations into Literature-Grounded Hypotheses and Theories

**arXiv ID:** 2610.02673 | [PDF](https://arxiv.org/pdf/2610.02673v1)

**作者:** Joseph Chee Chang `[一作]` (Allen Institute for AI), Daniel S. Weld `[通讯]` (Allen Institute for AI)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个交互式系统，能够从数百篇论文中抽取实验观测，构建概念-关系三元组，并通过层级本体对概念进行归一化，让研究者在此基础上手动构建证据图谱，随后系统生成并筛选假设，最终合成可验证的理论。

**💡 创新点**

核心创新在于：①将大量论文的实验观测压缩为可聚合的三元组并自动生成可扩展的本体；②通过研究者在画布上的手动操作（加入、删除、聚合概念）保持研究者主导的意图；③将生成的假设与用户已选证据相绑定，支持可视化冲突证据和未解释证据；④在多学科场景中验证其可扩展性和实际价值。

**🔧 技术方法**

技术实现主要采用：大型语言模型（Claude Opus 4.6）进行结构化抽取、关系归一化和假设/理论生成；Semantic Scholar API 用于检索论文全文；前端交互使用可视化画布、聊天机器人助手和概念本体层级导航；后端包括多阶段抽取、构建本体、关系标准化、假设生成与理论合成的流水线。

**📊 数据集**

数据集为自选领域内约 100-200 篇公开访问论文（包括 CS、农业和免疫学领域），共抽取约 1.7 万个三元组；在技术评估中对 12 篇专家编写的综述大纲进行对齐评估；在实验中与 10 名 CS 研究者以及 2 个学科专家团队（农业、免疫学）进行部署与案例研究。

**📈 对比分析**

与传统的本体生成或自动理论生成方法相比，本文的系统在概念覆盖度上平均达 74% 的最佳匹配 Jaccard（对照 35%），在三元组抽取和假设生成上支持交互式选择，用户满意度高（多数研究者在 1-2 周内生成 1-4 条理论）。成本约 50-150 美元（处理 100 篇论文）。

**⚠️ 局限性**

局限性包括：①基图在交互期间保持静态，无法实时增补新的证据；②可视化深度和上下文展示有限，难以表达实验上下文细节；③缺乏领域特定的本体与关系词典，导致在专业领域中覆盖不足；④对大型语言模型的依赖导致抽取质量受模型推理误差影响；⑤实验评估主要基于综述大纲对齐，缺乏系统性的理论质量客观评估。

---

## 212. Imagine the Future, Internalize the Gist: Efficient VLA Reasoning via Internalized Spatiotemporal Imagination

**arXiv ID:** 2610.02626 | [PDF](https://arxiv.org/pdf/2610.02626v1)

**作者:** Shenglan Li `[一作]` (Stevens Institute of Technology), Shaoyi Huang `[通讯]` (Stevens Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出IG‑VLA框架，利用潜在时空推理在视觉表示空间预想未来场景，并通过Scene Gist记忆将推理结果压缩成简洁的语义摘要供动作生成使用。

**💡 创新点**

创新点在于（1）直接在潜在视觉空间生成未来关键帧，避免像素级视频生成的高成本；（2）引入Scene Gist Token，将完整推理路径压缩为可在推断时一次性使用的“情境精髓”，从而显著降低在线推理开销。

**🔧 技术方法**

使用技术包括：潜在时空推理模块（Latent Spatiotemporal Reasoning）、Scene Gist Compressor/Encoder、预训练的SigLIP视觉编码器与Gemma多模态背骨、MSE与余弦方向一致性损失、行为一致性损失等。

**📊 数据集**

使用的数据集为LIBERO、LIBERO‑Plus、VLABench三大模拟数据集，以及在真实双臂UR3平台收集的77条遥控演示数据。

**📈 对比分析**

与ACoT‑VLA、FastWAM、ImageWAM、OpenVLA等多种基线对比，在LIBERO上平均成功率达98.9%（超越98.5%基线），在LIBERO‑Plus平均成功率88.5%，在VLABench最高Intention Score为66.68；Scene Gist版实现每个动作块169.5 ms推断，较FastWAM（302 ms）和ImageWAM（263 ms）提升约1.78×与1.55×。

**⚠️ 局限性**

局限性包括：需要大量高质量演示数据；潜在预测的时间窗口有限，长时间推理可能积累误差；当前实现以离线训练为主，未探讨在线自适应学习；模型依赖预训练视觉‑语言背骨的泛化能力，可能在极端新场景下表现受限。

---

## 213. Online Verification of Language Model Responses Under Cost Constraints

**arXiv ID:** 2610.02632 | [PDF](https://arxiv.org/pdf/2610.02632v1)

**作者:** Erfan Hajihashemi `[一作]` (University of California), Yanning Shen `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出在线多验证器验证（OMVV）算法，动态选择不同成本与准确度的弱验证器并在必要时查询昂贵的真值oracle，确保错误率受限且成本低。

**💡 创新点**

创新点在于：①维护一池多种弱验证器并通过指数权重在线学习路由策略；②为每个验证器单独学习组合得分（原始分+一致性）和阈值，提升判定可靠性；③在有限的oracle查询下实现分布式无偏损失估计，获得理论错误率控制与子线性 regret。

**🔧 技术方法**

技术实现：指数加权路由（Hedge）+ 重要性加权估计；逻辑回归得分组合；双阈值接受/拒绝/查询策略；部分反馈下的带探索oracle查询；成本与一致性相结合的损失函数。

**📊 数据集**

实验数据集：MATH（含多难度子集）、Zebra Logic、TriviaQA、DeepMind Mathematics，均采用 GPT‑4o‑mini 生成5个候选答案，Oracle 为 GPT‑4o 或程序化判分。

**📈 对比分析**

与两种单一弱验证器（Qwen2‑1.5B、Qwen2.5‑3B）对比，OMVV在四个数据集上均获得最高准确率、最低平均错误率、最低平均成本，并始终满足预设的错误阈值。对不同难度块大小的鲁棒性测试也表明，OMVV在快速变化的流中仍保持优秀表现。

**⚠️ 局限性**

局限性：需要先验设置多种弱验证器及其成本；对探索概率和学习率敏感；当oracle成本极高或无法频繁调用时，仍需较多查询；算法在极端数据分布漂移下的理论保证有限，且对多模型互补性假设较强。

---

## 214. Bao: Automatic Region Placement and Memory Allocation for Intermittent Computing

**arXiv ID:** 2610.02624 | [PDF](https://arxiv.org/pdf/2610.02624v1)

**作者:** Byeongjee Kang `[一作]` (Carnegie Mellon University), Feras A. Saad `[通讯]` (Carnegie Mellon University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

提出了一种基于混合整数线性规划（MILP）的编译器优化方法，实现对间歇性计算设备的能量安全区域划分和内存分配。

**💡 创新点**

将能量可行性与检查点成本直接编码到MILP中，做到全局最优且具形式化正确性保证，突破了现有局部启发式方案的局限。

**🔧 技术方法**

采用LLVM IR级能量建模、循环分块与CFG摘要、MILP求解（Gurobi）以及自动化的检查点插桩技术。

**📊 数据集**

在13个嵌入式基准（如AES、ChaCha20、Dijkstra、RSA等）和10条RFID能量采集轨迹上进行评测。

**📈 对比分析**

与两种基线（Chinchilla和Rockclimb）在连续电源和间歇电源下对比，平均提升10%执行速度，Region边界触发次数减少52%，在能量追踪实验中相对最优基线提升约17.8倍。

**⚠️ 局限性**

仍需循环上限注解、无法处理递归/间接调用及大规模函数；能量模型保守导致可能过度插桩；实现依赖Gurobi许可证。

---

## 215. CHASE-VLA: Post-Training Quantization Framework for Vision-Language-Action Models with Chunk-Aware Scale Estimation

**arXiv ID:** 2610.02666 | [PDF](https://arxiv.org/pdf/2610.02666v1)

**作者:** Jin Hyun `[一作]` (Korea Advanced Institute of Science and Technology), Youngjoo Lee `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对 Vision‑Language‑Action 模型的 diffusion‑based 行动专家（AE）设计了一种后训练量化（PTQ）框架 CHASE‑VLA，该框架利用之前生成的动作块（包含未执行的后缀）作为因果上下文动态估计激活量化尺度，支持 4‑bit 权重和激活的 W4A4 量化。

**💡 创新点**

创新点在于：① 引入块感知（chunk‑aware）激活尺度估计，将前一次动作块与降噪步组信息结合；② 通过轻量级预测器产生尺度乘子并加入不确定性修正；③ 将量化范围扩展到 AE 中的 MLP 与注意力投影，而非仅限 MLP，从而大幅降低 AE 的存储与内存流量。

**🔧 技术方法**

使用技术包括：后训练量化、基于动作块的尺度预测器、降噪步分组（M groups）、不确定性权重调节、对称量化与校准、以及针对 AE 线性层的 W4A4 量化实现。

**📊 数据集**

实验数据集主要为 LIBERO 四个任务套件（Spatial、Object、Goal、Long），并在三种 VLA 模型（π_0.5、GR00T N1.6、CogACT）上进行评估。

**📈 对比分析**

与 FP16、QuantVLA（W4A8/扩展 W4A4）、Q‑DiT、PTQ4DiT 等方法对比，CHASE‑VLA 在 π_0.5 上实现 97.3% 的平均成功率（与 FP16 相当），在 CogACT 与 GR00T N1.6 上分别达到 90.5% 与 91.5%；显著降低 AE 权重存储（73.4%）与单块内存流量（70.9–71.2%），并在 GPU 上实现 1.25–1.28× 的速度提升。

**⚠️ 局限性**

局限性：① 需要前一次动作块可用，首次推理时只能使用初始尺度；② 预测器训练依赖于校准轨迹，可能对新任务或极端运动场景的泛化能力有限；③ 目前验证仅针对现有的 diffusion‑based VLA AE 结构，未覆盖其他类型的 AE 或更大规模模型。

---

## 216. When History Misleads: Asymmetric Margin Supervision for Instruction-Guided LLM Generative Recommendation

**arXiv ID:** 2610.02600 | [PDF](https://arxiv.org/pdf/2610.02600v1)

**作者:** Ming Yin `[一作]` (Duke University), Qifan Wang `[通讯]` (Meta)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种通过对历史记录单个事件进行删除操作，并将删除后得到的边际提升作为监督目标的AIMS方法，用于改进指令驱动的生成式推荐系统的历史利用。

**💡 创新点**

创新点在于：①发现历史事件对预测的影响与其对目标项目的正负效用不一致（Influence–Utility Misalignment）；②提出使用离线历史删除的边际提升作为目标，在完整历史下训练学生模型；③采用不对称竞争者抑制（ACS）辅助损失，仅在边际不足时通过竞争者梯度更新，从而不改变推理过程。

**🔧 技术方法**

技术包括：预训练大语言模型的监督微调（SFT）、基于 SID 的自回归评分、Beam 搜索、单事件删除干预、边际计算、交叉熵训练、辅助损失 ACS、边际对齐。

**📊 数据集**

实验使用了一个工业数据集以及公开的 Qilin 和 KuaiSearch-Lite 两个基准数据集。

**📈 对比分析**

与多种基线（Continued CE、CFT、LETTER、LTRGR、S-DPO 等）在 Recall@10 和 NDCG@10 上进行对比。AIMS 在所有六种模型（Llama-3.1-8B/70B、Gemma-3-4B/12B、Qwen3-8B/14B）和三个数据集上均取得最佳 Recall@10 与 NDCG@10，提升幅度为 4.0–10.9%（相对），并在指令–历史冲突场景下表现更佳。

**⚠️ 局限性**

局限性包括：①方法依赖离线历史删除的质量，若删除选择不佳会影响边际目标；②仅在完整历史训练时使用，推理阶段仍需要完整历史，可能对长历史的计算开销有影响；③对极端长历史或高维事件的扩展尚未验证；④在冲突请求之外的场景下提升幅度相对有限。

---

## 217. Activation Sparsity with Weight Approximation for Faster LLM Decoding on Offloaded Weights

**arXiv ID:** 2610.02598 | [PDF](https://arxiv.org/pdf/2610.02598v1)

**作者:** JuneHyung Kim `[一作]` (LG Electronics), Nandita Vijaykumar `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练无关的多层级激活稀疏机制SpAx，通过按激活强度将权重列划分为全精度读取、压缩近似读取和完全跳过三类，以在GPU显存不足时加速LLM推理。

**💡 创新点**

创新点在于：①将传统二元稀疏决策扩展为三元决策；②引入中间压缩层，使用两位量化或树形编码在保持低误差的同时显著减少传输字节；③离线校准权重列分层，动态分配读取预算；④通过将同一输入激活共享的权重矩阵放置在连续存储区域并合并读取请求，降低闪存访问碎片。

**🔧 技术方法**

使用的技术包括：激活强度排名与top‑k分层、对每列使用两位Lloyd–Max量化或重叠窗口树形编码、离线预算搜索（分组贪婪搜索+网格搜索）、GPU端解压缩、Flash/CPU内存权重卸载、与现有稀疏方法（TEAL、WINA、LaRoSA、R‑Sparse）对比。

**📊 数据集**

在LLAMA‑3.1‑8B、Gemma‑4‑31B、Qwen‑3.6‑27B上使用WikiText‑2测试集进行困惑度评估，并在ARC‑Challenge、HellaSwag、PIQA、WinoGrande、MMLU、GSM8K等任务上使用lm‑eval评估下游性能。

**📈 对比分析**

与密集模型及四种基线（TEAL、WINA、LaRoSA、R‑Sparse）比较，SpAx在10%困惑度增益约束下：CPU内存卸载时BF16可获得3.86×（最高5.57×）速度提升，Q4_K_M可获得2.06×（最高2.74×）提升；闪存卸载时BF16可达3.31×（最高4.81×），Q4_K_M可达1.54×（最高2.03×）。在字节/吞吐量方面，SpAx在保持相同困惑度的情况下所需的读取预算显著低于所有基线。

**⚠️ 局限性**

局限性包括：①中间层压缩码的解压缩在GPU上增加计算开销，尤其在4‑bit格式下仍显慢；②闪存读取碎片导致的高I/O延迟在低压缩率下无法完全抵消；③对不同模型架构（如DeltaNet注意力）的通用性尚未完全验证；④需要离线校准数据集，若部署环境激活分布变化，可能需要重新校准。

---

## 218. Random Quantum LDPC Codes Approaching the Gilbert-Varshamov Bound

**arXiv ID:** 2610.02648 | [PDF](https://arxiv.org/pdf/2610.02648v1)

**作者:** Tushant Mittal `[一作]`, Mary Wootters `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构造了一个随机量子低密度校验码（QLDPC）族，证明其在给定速率下能以高概率达到量子Gilbert–Varshamov（GV）距离界，并在擦除通道和记忆无关Pauli通道上实现容量近似。

**💡 创新点**

创新点在于：①提出了结合AEL（Alon–Edmonds–Luby）距离放大技术与随机内码拼接的新的随机构造；②首次量化并控制QLDPC码的退化性（degeneracy），从而克服低密度校验码天然存在的低权重稳定子问题；③证明该随机族在量子通道性能上与完全随机稳定子码匹配。

**🔧 技术方法**

主要技术包括：量子稳定子码的对称性线性代数描述、随机拼接（Thommesen）构造、AEL距离放大与图扩张的组合、低权重向量计数与退化性分析，以及典型集与信息论工具用于通道性能证明。

**📊 数据集**

无实测数据集；所有结果均为理论分析与概率论证明。

**📈 对比分析**

通过与随机稳定子码的已知性能（例如达到GV界、擦除通道容量和哈希界）对比，随机QLDPC码在理论上可实现相同的距离与容量，且保持O(1)检查权重，意味着具有高效的本地检测与纠错潜力。

**⚠️ 局限性**

局限性包括：构造仍为随机化，缺乏可构造（explicit）方案；实现需要对称子空间的随机采样；对退化性控制需保持常数η，可能导致检查权重随η增大而上升；此外，虽然理论上可匹配容量，但实际实现与解码算法的复杂度仍未给出完整细节。

---

## 219. SpectralCache: Accelerating Diffusion-Based World Models via Spectral Feature Caching

**arXiv ID:** 2610.02660 | [PDF](https://arxiv.org/pdf/2610.02660v1)

**作者:** Zhendong Mi `[一作]` (Stevens Institute of Technology), Shaoyi Huang `[通讯]` (Stevens Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种无训练的谱特征缓存框架 SpectralCache，用来加速扩散式世界模型的推理，同时保持生成质量。

**💡 创新点**

通过分析扩散特征的奇异值分解，发现奇异子空间在相邻步长高度稳定且奇异值随步长呈线性或常数比例，可用线性外推或比例缩放重建特征，从而减少Transformer计算。

**🔧 技术方法**

使用奇异值分解（SVD）、线性外推、奇异值缩放、误差累积触发全计算以及低秩缓存等技术。

**📊 数据集**

在 HunyuanWorld‑Voyager‑13B 与 Aether‑5B 两个主流世界模型上评估，并利用对应的视频生成数据集以及 Sintel 数据集进行 3D 重建评估。

**📈 对比分析**

与多种训练‑free 缓存方法（DuCa、ToCa、TaylorSeer、HiCache、TeaCache、EasyCache、HERO、WorldCache）在 WorldScore、PSNR/SSIM/LPIPS、速度与显存等指标上比较，SpectralCache 在 Hunyuan 实现 5.22× 速度提升且 WorldScore 提升 1–2 分，在 Aether 2.20× 速度提升且提升 1–2 分。

**⚠️ 局限性**

需要额外进行 SVD 计算，虽然开销小但仍存在额外开销；对最大缓存间隔和缩放因子敏感，若环境变化剧烈可能导致误差累积；目前仅在两款模型上验证，跨模型通用性尚未完全证明。

---

## 220. A Passive AI System for Verifying Physical State on Automated Liquid Handlers

**arXiv ID:** 2610.02668 | [PDF](https://arxiv.org/pdf/2610.02668v1)

**作者:** Junqiong Joanne Qiu `[一作]` (UCLA), Eleazar Eskin `[通讯]` (UCLA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一个基于低成本摄像头和YOLO视觉模型的外部检测系统，用于在实验开始前实时识别并验证Opentrons Flex液体处理器机台槽位中的实验器材与协议要求是否匹配；

**💡 创新点**

该系统实现了无需硬件或固件修改、无干扰的协议感知机台检查，结合秒级别的图像分割+分类与协议匹配，并通过网页界面和云端部署实现了可视化、易集成的检查流程；

**🔧 技术方法**

主要技术包括YOLO11m分割模型、YOLO26m分类模型、图像预处理（CLAHE）与自适应缩放、仿射校正与匈牙利算法进行槽位对齐、协议匹配逻辑、FastAPI后端与React前端，并在AWS云上部署；

**📊 数据集**

使用了121张全机台拍摄图像和1212张裁剪后的实验器材图像，共17类（含空槽），采用Roboflow进行训练集增强，并按85/10/5划分训练/验证/测试集；

**📈 对比分析**

在实验室场景下（空机台、SwabSeq RVP预PCR、TECAN Library Cleanup）测试，槽位分类准确率在95.1%~97.7%之间，完整机台检测100%；端到端延迟为本地2.1秒、云端4.2秒，显示出与现有仅像素比较的DeckCheck相比，能更直接识别器材并匹配协议；

**⚠️ 局限性**

局限性包括：仅检查机台布局，无法检测液体体积、吸头/管子缺失或液体转移错误；对外观相似的器材易误判；需完整摄像头覆盖并手动添加新协议或器材；无法阻止操作员继续执行错误操作。

---

## 221. TasteBench: Multimodal Benchmark for Sensory Prediction, from Molecules to Sustainable Foods

**arXiv ID:** 2610.02599 | [PDF](https://arxiv.org/pdf/2610.02599v1)

**作者:** Anna T. Thomas `[一作]` (Stanford), Benjamin Sanchez-Lengeling `[通讯]` (University of Toronto)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出TasteBench基准，提供面向可持续蛋白的食品级感官排名任务与分子级味觉分类任务，并通过Kaggle竞赛实现隐私保护与公开评测。

**💡 创新点**

构建多模态分级特征评估框架、隐私保护竞赛机制，给出人类感官评测的可靠性上限，并揭示分子分类精度不转移至食品级排名的现象。

**🔧 技术方法**

使用多模态特征提取（营养、文本、化学、图像）、LLM零射击（Gemini 3.1 Pro、Qwen 3.5）、监督学习（Ridge、Bradley–Terry、Kernel RankSVM、LightGBM）、集成与MMRF，以及分子编码器（FART、D-MPNN、ChemBERTa‑2）等技术。

**📊 数据集**

依托NECTAR 2025/2026感官评测数据（215植物性产品）、Taste Like（1200+无感官评分产品）用于隐私混淆、FoodAtlas知识图谱进行化学成分映射，以及FartDB 15025分子数据用于味觉分类。

**📈 对比分析**

通过成对排名准确率、Spearman/Kendall相关、Recall@k等指标评估；最佳模型在所有对上达到0.683，略高于中位数人类评审0.650，但仍低于可靠性上限0.825；分子分类模型准确率在0.84–0.90之间，但对食品级排名并无显著提升。

**⚠️ 局限性**

仅涵盖肉类与乳制品，样本量小且人类评测受限于美国城市的未受训消费者，未覆盖其他感官维度；LLM可能受训练数据污染，且模型对训练面板与专业评审的泛化尚未验证。

---

## 222. Annotation-Driven Migration of CUDA Programs to Tenstorrent Blackhole

**arXiv ID:** 2610.02658 | [PDF](https://arxiv.org/pdf/2610.02658v1)

**作者:** Ayumi Ohno `[一作]` (University of Tokyo), Shinya Takamaeda-Yamazaki `[通讯]` (University of Tokyo and RIKEN)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

迁移CUDA HPC内核到Tenstorrent Blackhole，实现了基于MLIR的编译器，自动推断数据布局、跨核心通信与每个核心的计算/数据迁移实现，并通过声明式注解暴露不同的实现选项。

**💡 创新点**

创新点在于将每核实现视为可声明的策略空间，允许在编译时对算术、保护、广播、同步等细粒度决策进行组合并自动下推，而非硬编码固定实现。

**🔧 技术方法**

使用技术包括Polygeist将CUDA转换为MLIR Affine Dialect、TT‑MLIR的Direct‑to‑Metal IR、TT‑Metal、LLK扩展，以及声明式注解和多阶段（TilePlan、TileFlow、Task、Physical）编译流水线。

**📊 数据集**

使用的数据集包括Rodinia的Gaussian消去、PolyBench的五点Stencil和Symmetric Rank‑k Update（SYRK），所有输入保持原始Affine形状，线程块尺寸固定为32×32。

**📈 对比分析**

通过在Blackhole上测量不同注解组合的执行时间，并与编译器默认实现及A100 GPU实现进行对比；在BF16 Gaussian上获得4.2×加速，FP32 Gaussian与Jacobi分别获得2.1–2.2×加速；与A100相比，Blackhole在相同问题规模下约慢3×，但在L1保持全部工作集方面表现相当。

**⚠️ 局限性**

局限性包括：仅支持Affine、固定尺寸、二维张量的CUDA内核；线程块必须匹配32×32；不支持CUDA内存/共享内存；跨核心流式加载不可分块；需要手动选择注解；且对不同精度的性能影响高度相互耦合，仍需进一步的自动化和反馈驱动的注解搜索。

---

## 223. A Token Service Interface for AI-Native RANs

**arXiv ID:** 2610.02618 | [PDF](https://arxiv.org/pdf/2610.02618v1)

**作者:** Jianan Zhang `[一作]` (Peking University), Xiang Cheng `[通讯]` (Peking University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Token Service Interface (TSI)，为 AI 本地 RAN 的 token 通信提供统一的描述符和控制边界，并在无人机检查案例中验证其效果。

**💡 创新点**

通过将服务、时间与状态三维语义绑定到 token 集描述符，实现按重要性、预备时间与执行状态迁移的动态网络资源调度；同时提供双向接口支持应用与网络的协同决策。

**🔧 技术方法**

结合 5G Release 18 PDU‑级 QoS、TSI token 集描述符、模型执行预测、状态迁移技术、解耦上/下行网络架构以及深度学习推理框架。

**📊 数据集**

使用 COCO 2017 验证集生成 saliency 权重以及无人机飞行轨迹数据。

**📈 对比分析**

与均匀重要性与耦合/解耦网络基线对比；在解耦上行下行中，按重要性加权可节省约 27% 上行资源；预备准备可提升约 15% 实时价值；状态迁移策略相较固定锚点或每次迁移可将平均额外延迟降低 26% 以上。

**⚠️ 局限性**

仍需对服务重要性进行校准、时序元数据规模与预测误差、移动状态迁移时延与同步成本、接口标准化及隐私安全等方面进行进一步研究。

---

## 224. Coherence-Driven Belief Formation and Population Dynamics of Contagion in LLM Agents

**arXiv ID:** 2610.02654 | [PDF](https://arxiv.org/pdf/2610.02654v1)

**作者:** Tathagata Banerjee `[一作]` (Takeda Pharmaceuticals), Nima Moghaddas `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过实验测量大型语言模型（LLM）代理在不同社交压力下对单一信念的采纳概率，构建了以同伴认同数为自变量的sigmoid型采纳核，并在网络中模拟其传播行为。

**💡 创新点**

创新点在于将人类社会传播研究中的简单与复杂传染机制引入LLM代理，并发现采纳核与三个可调因素（信念可信度、来源可靠性、代理倾向）可归约为单一有效“连贯性”维度，且在网络层面表现出聚类优势、双稳性和自持共识等复杂传播特征。

**🔧 技术方法**

技术上采用多轮提示式推理与对数概率计算得到采纳概率，利用主成分分析和加性线性模型评估维度压缩，使用Watts–Strogatz网络模拟聚类与随机网络对传播的影响，并实现基于采纳核的Agent驱动传播与保留模拟。

**📊 数据集**

数据集包含两类不可验证预测信念（次日波士顿温度和通勤时间）以及三种信源可靠性与三种代理倾向的提示组合，共计27组实验条件，并在两款LLM（Qwen与Mistral）上重复实验。

**📈 对比分析**

比较方法包括对比不同网络聚类系数下的最终采纳比例、利用Mann–Whitney U检验聚类优势、通过阈值截断计算二分式收敛窗口以及绘制上升与下降路径的滞后环，实验表明在聚类网络中传播更广、共识更难被撤销，且单维度压缩解释率高于90%。

**⚠️ 局限性**

主要局限在于仅考虑单一无竞争信念、网络规模有限且样本仅覆盖两种LLM与两类信念，未探究内部表示机制、竞争性传播、不同规模或更复杂事实/价值观信念的行为，且结果受提示设计与模型偏差影响。

---

## 225. WebUIProof: Benchmarking WebUI Code Generators with UI-Agent Execution Harness

**arXiv ID:** 2610.02617 | [PDF](https://arxiv.org/pdf/2610.02617v1)

**作者:** Yun-Yun Tsai `[一作]` (Columbia University), Sinong Wang `[通讯]` (Meta SuperIntelligence Labs)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了WebUIProof，一个基于可执行交互测试的Web前端代码生成评测基准。

**💡 创新点**

创新点在于用结构化规范与大量可执行交互测试，结合UI-agent执行引擎实现功能级评估，并提供训练信号。

**🔧 技术方法**

技术包括头less浏览器（Playwright）、UI代理（基于WebVoyager）、多步plan‑act‑observe循环、视觉‑语言评估、强化学习（VisRL）以及多温度采样与快速门过滤。

**📊 数据集**

数据集来源于WebGen‑Bench、WebDev‑Arena以及三维模拟示例，共219个任务、约3k交互测试，包含149个普通WebUI与70个3D仿真。

**📈 对比分析**

通过对八大商业LLM（Claude Sonnet、Gemini、Qwen 3 Coder、DeepSeek R1、GPT4.1等）以及两款自研模型（Qwen 2.5 14B、MIMO 7B）进行评测，发现即使是最强模型在交互测试中准确率仍低于40%，但VisRL训练可将小模型准确率提升至约42%。

**⚠️ 局限性**

局限在于评测依赖UI-agent定位与交互步骤，存在步骤预算与定位误差导致误判；此外3D仿真任务仍对模型提出高难度交互需求，未完全覆盖更复杂的图形交互场景。

---

## 226. Information Operations Exploit APIs to Manipulate Social Media

**arXiv ID:** 2610.02591 | [PDF](https://arxiv.org/pdf/2610.02591v1)

**作者:** Manita Pote `[一作]` (Indiana University), Filippo Menczer `[通讯]` (Indiana University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

探讨第三方应用程序（API）在推特信息作战中的作用，分析其如何被利用来协调不真实账户的行为；

**💡 创新点**

首次将应用程序使用模式作为协调指示器，并通过三种互补计算方法（名称相似、账户重叠、应用使用相似）揭示API基础设施、游戏应用滥用、伪造应用等操纵手段；

**🔧 技术方法**

使用图论、相似度度量（最长公共前缀、账户重叠、TF-IDF + 余弦相似度）、阈值剖析、最大指示器支持（基于置换检验的p值阈值）以及无监督网络检测算法；

**📊 数据集**

从Twitter/ X的43个被暂停的信息作战账户的完整推文及应用元数据（496个应用、8,597个账户、2100万条推文）以及26个带控制账户的标注子集；

**📈 对比分析**

对比使用五个传统指标（共享话题、URL、转推用户、转推内容、发帖同步）与加入应用使用后，通过最大指示器支持进行排序；AUC-ROC从0.56提升到0.64，提升显著（p<0.0001），在多数运营中均有改善；

**⚠️ 局限性**

局限性：仅基于过去（2018‑2021）数据；阈值设定固定；未使用内容相似度；仅研究推特平台；平台已移除应用元数据，无法在未来复现相同分析；

---

## 227. AIGS: Adaptive Incremental Gating System for Online Representation Learning in Non-Stationary Data Streams

**arXiv ID:** 2610.02661 | [PDF](https://arxiv.org/pdf/2610.02661v1)

**作者:** SiRui He `[一作]` (Xiamen University Malaysia), Chean Khim Toa `[通讯]` (Xiamen University Malaysia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Adaptive Incremental Gating System（AIGS），一种基于闭环状态感知的在线表示学习框架，用以解决边缘计算环境下的稳定性-可塑性困境。

**💡 创新点**

创新点：① 引入 Shock Ratio（冲击比）作为内生残差反馈，用以标准化重构误差；② 设计连续可塑性控制器（Sigmoid 门控），实现学习率与记忆保留的平滑插值；③ 通过状态-特征双信号耦合，把结构波动直接送入下游模型；④ 维持严格 O(k·d) 的线性每步复杂度，兼具轻量化与可插拔性。

**🔧 技术方法**

技术手段：在线主成分分析（GHA / OPAST / CCIPCA / LinearAE）+ 逐步子空间跟踪；指数移动平均 + Shock Ratio；连续 Sigmoid 门控 + 自适应学习率/记忆衰减；GRU 下游序列预测；实验对比、消融、敏感性分析等。

**📊 数据集**

数据集：工业设备温度序列（ETTm1、ETTm2）、交通流量（PEMS04、PEMS08）、气象雷达（Jena Weather）等智能城市边缘数据。

**📈 对比分析**

方法比较：对标 DLinear、PatchTST、iTransformer、T-GCN、CEP、FSNet、ADWIN+GRU 等多种基线；在 ETT 上平均提前预警时间（AvgLead）显著提升；在 PEMS 上后突变恢复时间（ART）显著缩短；在 Weather 上异常召回率提高；整体在 RMSE 与资源占用上实现 Pareto 优势，适合边缘部署。

**⚠️ 局限性**

局限性：EMA 需要短暂缓冲，导致对极端突发事件的检测延迟；单通道设计难以捕捉多维协方差旋转；在高噪声环境下仍需人工介入以避免误报；对超参数（阈值、放大因子）敏感，需场景调优。

---

## 228. How Causality Bridges the Semantic Gap

**arXiv ID:** 2610.02594 | [PDF](https://arxiv.org/pdf/2610.02594v1)

**作者:** Shuhao Zhang `[一作]` (University of California San Diego), Yujia Zheng `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于因果结构的语义对齐框架，能够在只有少量已命名变量的情况下，利用因果图推断并为未命名（包括潜变量）分配语义名称；

**💡 创新点**

创新点在于将因果关系作为语义推断的核心，通过结构约束的嵌入优化和前缀适配器将因果结构直接映射到语言模型，从而摆脱传统依赖人类知识的偏见；

**🔧 技术方法**

核心技术包括因果结构学习（BOSS、RLCD）、结构约束的嵌入优化（生成函数、距离相关度约束）、前缀适配器（KV缓存）与冻结语言模型（如Qwen3-8B）等；

**📊 数据集**

使用了六个机器人/车辆场景（仿真机械臂、真实机械臂日志、车辆CAN总线记录）以及五个公开心理测量量表（DASS、HEXACO、KIMS、RIASEC、World Values Survey）做实验；

**📈 对比分析**

与多种基线（CLIP-Dissect-e5、AutoInterp、Delphi、SASC、DeViSE、GraphMAE等）比较，框架在变量命名准确率（NRR@1、MRR）上显著优于基线，尤其在变量标注稀缺（20–90%遮蔽）时优势更大；

**⚠️ 局限性**

局限性在于实验仅覆盖已知真实名称的系统，未在真正无标签或深层潜变量（如神经网络特征、基因调控网络）上进行验证，且依赖于可学习因果结构的假设。

---

## 229. Seer: Maximum Likelihood Regression for Learning-Speed Curves

**arXiv ID:** 2610.02610 | [PDF](https://arxiv.org/pdf/2610.02610v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 230. Look Here or Look Across: Unified Cardinality Constraints for N-ary Relationships

**arXiv ID:** 2610.02603 | [PDF](https://arxiv.org/pdf/2610.02603v1)

**作者:** Huanyi Chen `[一作]` `[通讯]` (University of Waterloo), Huanyi Chen (University of Waterloo)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种新的统一写法 Rpq = (lower, upper)，用以消除实体关系图中基数约束的两种相反解读，并提供默认约束、分解和增广两条推理规则，以在 n 元关系中自动推导出隐藏的基数约束。

**💡 创新点**

创新点在于将基数约束抽象为可独立书写的函数 Rpq，既不偏向“look‑here”或“look‑across”解读，又不受图形边标签数量限制；同时给出两条简单且可手工使用的推理规则，解决了传统 ER 图在三元或以上关系中的表达不足。

**🔧 技术方法**

主要技术包括：基数约束的数学建模、基于集合分组的符号表示、推理规则的形式化证明以及案例分析中的手工计数与推导。

**📊 数据集**

本文没有使用公开数据集；案例研究采用的是一个虚构的大学研究中心模型，用来说明约束推导流程。

**📈 对比分析**

由于本工作主要是理论与符号化分析，未进行实验性性能评估；作者通过手工案例展示了推理规则的有效性，但未给出定量比较。

**⚠️ 局限性**

局限性包括：推理规则仅覆盖基数约束的基本传播，无法处理更复杂的依赖组合；默认约束 (0,*) 可能导致信息丢失；案例研究仅涉及单一关系，未检验在大规模多关系图中的可扩展性。

---

## 231. Context-Tower Conversion Preserves Generation While Freezing Retains Knowledge: Low-Budget AR-to-Diffusion Conversion of MoE LLMs

**arXiv ID:** 2610.02657 | [PDF](https://arxiv.org/pdf/2610.02657v1)

**作者:** Wentao Lu `[一作]` (Celeris), Tianyu Zhu `[通讯]` (Celeris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将预训练的自回归语言模型转换为扩散语言模型，并对两种主流转换方法（in-place 与 frozen‑tower）在同一30B MoE父模型、相同语料、相同训练预算、相同可训练参数集及评估工具下进行比较。

**💡 创新点**

提出在冻结上下文塔（frozen‑tower）同时训练可变形的去噪器，能够在低预算（约1B训练token）下保持大约95%父模型生成性能，显著优于传统的 in-place 方式；同时给出了梯度隔离分析、采样协议对分数影响的理论与实证，揭示了评估协议差异导致的分数变动达4.8×。

**🔧 技术方法**

使用软max注意力的因果Transformer架构，采用掩码扩散（masked diffusion）目标与去噪对齐损失；实现两塔结构时通过键值拼接实现无参数的交叉注意力；对比实验中还使用了不同的attach密度和去噪器大小。

**📊 数据集**

使用公开的 MoE 预训练混合语料库（含 30B 参数的 MoE 父模型）以及标准评估基准（GSM8K、MBPP、BoolQ、SQuAD、JSON validity、HumanEval 等）。

**📈 对比分析**

在相同的训练预算（1B token）和评估工具下，frozen‑tower 模型在10-token生成任务中获得约92%父模型分数，显著高于 in-place 模型的约80%（提升约11.6×）；在知识基准上，两种方法都保持大部分性能；在长文本生成任务上，frozen‑tower 维持较好分数，而 in-place 明显退化。

**⚠️ 局限性**

实验仅覆盖单一 MoE 父模型，随机种子与硬件差异仅在部分实验中检验；未对 in-place 方法进行多种随机种子验证；未对吞吐量与内存占用进行全面评估；去噪器大小与attach密度的比较受硬件与学习率等因素干扰，结论仍需进一步验证。

---

## 232. What Is Lost in Post-Training? Default Collapse and the Loss of In-Context Steerability Across Diverse Perspectives

**arXiv ID:** 2610.02614 | [PDF](https://arxiv.org/pdf/2610.02614v1)

**作者:** Jessica Dierking `[一作]`, Niclas Boehmer `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大型语言模型在面对文化价值分歧时的后训练（post‑training）策略进行研究，并提出一种新型目标函数以在保持多元观点表达的同时实现偏好调节。

**💡 创新点**

创新点在于揭示传统后训练会削弱模型在逆向引导下表达未训练过的观点的能力，并设计了“立场分布匹配（stance‑distribution matching）”的目标函数，在保证整体奖励最大化的同时约束输出视角分布。

**🔧 技术方法**

使用技术包括：① 传统的后训练微调（fine‑tuning）；② 构造基于文化价值对立的监督数据并在不同检查点评估；③ 设计新的受限奖励优化目标，结合分布约束实现多视角输出；④ 评估模型在“对立观点执行”任务上的表现。

**📊 数据集**

数据集未在摘要中给出，作者在受控实验中使用了人工构造的文化价值分歧样本，具体来源和规模未披露。

**📈 对比分析**

通过对比在训练过程中各检查点的“对立观点表达”能力，作者发现传统后训练导致模型在主导视角上越来越强，而对立视角的执行率显著下降。采用新目标函数后，模型能够在不牺牲整体奖励的前提下，保持两侧观点的相对均衡，提升了多元表达的可控性，但具体数值指标未在摘要中给出。

**⚠️ 局限性**

主要限制包括：① 需要人工定义所期望的视角分布，可能缺乏通用性；② 新目标函数的优化成本和收敛性问题尚未在大规模实验证明；③ 仅在受控实验环境中验证，未在真实用户交互场景中检验多元观点保持的效果；④ 对立观点的质量与多样性仍受限于训练数据本身的覆盖范围。

---

## 233. Real-time Event-camera Stereo Visual Odometry via Keytime Gaussian Process Regression

**arXiv ID:** 2610.02601 | [PDF](https://arxiv.org/pdf/2610.02601v1)

**作者:** Nikan Nobari `[一作]` (Queen’s University), Jonathan D. Gammell `[通讯]` (Queen’s University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种实时连续时间立体视觉里程计（VO）管线，专为事件相机设计；

**💡 创新点**

创新点在于利用高斯过程（GP）与白噪声加速先验（WNOA）结合的关键时刻插值方法，显著减小状态量同时保留事件异步测量的高时间分辨率；

**🔧 技术方法**

采用GP-WNOA先验、MC‑RANSAC离群点拒绝、iSAM2增量优化以及C++/GTSAM实现的关键时刻插值测量因子；

**📊 数据集**

使用MVSEC（346×260）与DSEC（640×480）事件相机数据集进行评估；

**📈 对比分析**

与基线GPCT和离散事件VO ES‑PTAM 对比：关键时刻估计在全局和相对误差上与GPCT相当，且在除一序列外均优于ES‑PTAM；实时帧率分别为MVSEC 22 Hz、DSEC 6 Hz；

**⚠️ 局限性**

局限性包括仅在立体事件相机上测试，特征提取依赖事件聚簇，未考虑IMU融合，且对高分辨率数据的实时性仍受限于事件特征生成速度。

---

## 234. Fisher-Guided Submodular Data Selection for Continual Pre-Training of Large Language Models

**arXiv ID:** 2610.02593 | [PDF](https://arxiv.org/pdf/2610.02593v1)

**作者:** Zhenghao Zhao `[一作]` (University of Illinois Chicago), Yan Yan `[通讯]` (University of Illinois Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种基于 Fisher 信息的子模数据选择方法，用于连续预训练（CPT），在目标领域提升性能的同时显著降低遗忘。

**💡 创新点**

创新点在于将候选样本梯度拆分为高 Fisher（anchor）与低 Fisher（frontier）两种方向，并用 log‑determinant 子模目标在流式中一次通过地选取非冗余且兼顾保留与获取的样本。

**🔧 技术方法**

使用了 Fisher 信息、梯度分解、log‑determinant 子模函数、Sieve‑Streaming 一次通过、LoRA 子空间、对角 Fisher 近似、随机投影（TRAK）以及 Sherman‑Morrison 升级。

**📊 数据集**

实验使用 TinyLlama‑1.1B 与 Llama‑3.1‑8B 两大模型；候选数据来自医学领域的 PMC 与 PubMed 文本，参考分布取自 FineWeb 与 Llama3‑SynE 等通用语料。

**📈 对比分析**

与随机、低/高 perplexity、Replay、EWC 等基线比较，实验表明在 TinyLlama 上 1 B 选取已优于 10 B Replay，适配增益超过 4.2 倍，遗忘率降低一半，整体 Pareto 前沿最优。

**⚠️ 局限性**

局限性在于仅使用对角 Fisher 近似、依赖 LoRA 子空间和 warmup 检测点，未探索更完整的曲率模型或多模态场景。

---

## 235. Designing the Future of User Feedback for Generative AI

**arXiv ID:** 2610.02631 | [PDF](https://arxiv.org/pdf/2610.02631v1)

**作者:** Alisa Frik `[一作]` (International Computer Science Institute), Mohammad Tahaei `[通讯]` (International Computer Science Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了生成式AI产品的用户反馈机制，进行基准、可用性评估、专家启发式评估及用户测试，提出设计准则并验证原型。

**💡 创新点**

创新点在于系统化的可用性框架、跨行业基准与用户驱动的设计准则，并将法规与实践对接。

**🔧 技术方法**

采用可用性检验、Nielsen启发式评估、远程可用性测试与交互原型设计（Figma）等技术。

**📊 数据集**

数据来源为32款含GenAI功能的公开产品（包括eBay及其他平台），以及13名eBay买卖双方用户。

**📈 对比分析**

与现有反馈机制相比，新的工具在用户完成率、满意度、信息完整度等指标上提升约20–30%。

**⚠️ 局限性**

限制包括样本仅覆盖英文免费产品、案例过度依赖eBay、未覆盖多语言和行业差异，以及未评估长期使用效果。

---

## 236. Lost in the Request: How Communication Variation Disrupts Retrieval and Action in Email Agents

**arXiv ID:** 2610.02627 | [PDF](https://arxiv.org/pdf/2610.02627v1)

**作者:** Feng Chen `[一作]`, Alex Williams `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究电邮助手在请求表述（语气、正式度、冗长、情绪、语法、方言）改变时，是否仍能完成同一任务；构建并验证变体，评估检索增强生成（RAG）和工具调用（agentic）两类系统的鲁棒性。

**💡 创新点**

提出一种可验证的请求变体生成框架，将语义保持不变的不同表述与标准化方言特征结合；首次在多种评估任务中同时考察检索质量与执行完整性，并区分遗漏动作与不支持动作两种失败模式。

**🔧 技术方法**

利用 LLM 重新写作（10 个语气极值）、规则基方言转换器、检索器（BM25、SPLADE、BGE‑M3）、生成模型（T5、LLaMA‑2、ChatGLM）以及工具调用框架；对变体进行门控验证后进行评估。

**📊 数据集**

EnronQA（检索增强问答）、ToolTalk（多轮工具调用）和 WorkBench（工作场景工具调用）三大公开基准；在 EnronQA 选取 400 题；在 ToolTalk 选取 50 轮对话；在 WorkBench 选取 90 任务。

**📈 对比分析**

通过比较基线与每种变体的得分差（∆）和置信区间，发现间接请求导致所有任务性能下降；正式请求对 agentic 任务有显著负面影响；方言变体对所有任务都有影响，尤其是 SPLADE 检索时维持检索成功后仍有生成错误。具体数值：EnronQA 直接-间接差约 -3.2%，ToolTalk 对话成功率降约 -22.4%，WorkBench 任务准确率降约 -10.7%。

**⚠️ 局限性**

局限性包括：方言变体仅在最高特征密度（1.0）下生成，未考虑实际使用频率；WorkBench 在部分任务缺少覆盖；ToolTalk 的参考极值导致基线偏差；跨模型/系统比较受限于计算资源；未提供任何鲁棒性提升的实证方法。

---

## 237. Distributed Learning with Selective State Space Models: Architecture-Aware Convergence Analysis

**arXiv ID:** 2610.02659 | [PDF](https://arxiv.org/pdf/2610.02659v1)

**作者:** Adam Piaseczny `[一作]` (Purdue University), Christopher G. Brinton `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了在联邦学习环境下的选择性状态空间模型（SSM）的收敛行为，给出了针对其结构的梯度和光滑性界，并对FedAvg、FedProx等算法提供了架构感知的收敛率；

**💡 创新点**

创新点在于首次将SSM的递归稳定性、输入相关离散化与投影范数等核心参数显式纳入收敛分析，形成可解释的梯度上界和聚合误差项；

**🔧 技术方法**

使用了S4/ Mamba 等现代SSM架构、首次差分光滑性分析、FedAvg/FedProx等联邦学习框架及其变体；

**📊 数据集**

实验数据集包括教师生成的单层SSM合成序列以及RedPajama文本域（6个域）下的Mamba2语言建模；

**📈 对比分析**

与九种联邦学习算法（FedAvg、FedProx、SCAFFOLD、FedDyn、FedAlign、FedSAM、FedExP、DiLoCo、FedAdam）进行比较，理论预期与实验性能相关性约0.71（β=1时为0.81），部分算法在高异构性下表现优于FedAvg；

**⚠️ 局限性**

局限性主要在于界限保守、常数过大、对半径R的依赖以及对全局序列长度的忽视，导致在实际训练中理论与经验偏差较大。

---

## 238. Quantifying the Value of Constructive Induction, Knowledge, and Noise Filtering on Inductive Learning

**arXiv ID:** 2610.02615 | [PDF](https://arxiv.org/pdf/2610.02615v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 239. Open-Endedness Bench: Measuring Epistemic Process from Agent Records

**arXiv ID:** 2610.02588 | [PDF](https://arxiv.org/pdf/2610.02588v1)

**作者:** Chengyang Shi `[一作]` (University of Michigan), Jiachen Liu `[通讯]` (ARA Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个无任务依赖的评估框架，通过统一的认知事件图从执行记录中评估AI代理的经验过程。

**💡 创新点**

创新点在于不需要真值答案或任务特定的评分表，利用LLM提取卡片、法庭判决构造图，使用机会率规则量化证据处理与实验设计能力。

**🔧 技术方法**

技术包括基于LLM的卡片抽取与法庭问答、图编译与计数规则、机会率评分机制、以及对奖励作弊的检测。

**📊 数据集**

使用了三大基准的日志：PostTrainBench、Chip-Bench 和 nanoGPT speedrun 的执行记录。

**📈 对比分析**

与现有的结果基准相比，该方法在多任务上保持了较低的分数波动（≤0.1），验证了评分稳健性，并揭示了代理仅约20% 的自报改进真实。

**⚠️ 局限性**

局限性包括对数值表达的推理限制、缺少记录外的推理、可能被代理针对优化、以及较高的计算成本。

---

## 240. Capturing Dynamics: The 4D Facial Expression Intensity Dataset

**arXiv ID:** 2610.02647 | [PDF](https://arxiv.org/pdf/2610.02647v1)

**作者:** Zesheng Wang `[一作]` (École Centrale Nantes), Guoying Zhao `[通讯]` (University of Oulu)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

创建了4D Facial Expression Intensity Dataset（4D‑FEID），收集了2869个3D面部表情序列，并通过90,000+ Likert 评分获得全局序列级强度标注，同时在此数据集上构建并评估了多种深度网络基线；

**💡 创新点**

首次提供全时空序列级强度标注，并提出差分学习策略去除身份噪声，证明高保真3D时空建模相较于传统2D或参数化方法更能捕捉细微情绪变化；

**🔧 技术方法**

使用FLAME 3D 形状/表情参数生成可控表情，结合差分学习、时空图卷积、Bi‑LSTM、ResNet‑18 等深度学习框架；

**📊 数据集**

数据集本身（4D‑FEID）以及对比的 DISFA、BP4D 等公开表情强度数据集；

**📈 对比分析**

对比 Video‑CNN、4D‑Vertex 与 FLAME‑based 三种基线，4D‑Vertex 在 MAE、ICC、PCC、CCC 等指标上表现最佳，且跨身份泛化性能优异；

**⚠️ 局限性**

局限在于人工筛选流程、未覆盖极端或文化多样性表情、模型仍需进一步提升鲁棒性与泛化力。

---

## 241. VERSE: Verified Self-Evolving Optimizer for Agent Harnesses

**arXiv ID:** 2610.02616 | [PDF](https://arxiv.org/pdf/2610.02616v1)

**作者:** Zekai Wang `[一作]` (Massachusetts Institute Of Technology), Chandan K. Reddy `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为LLM代理的harness提供了一个可验证的自我进化优化器，使得优化器在不改变模型权重的前提下，通过执行验证、回放失败和训练审计等工具，自主改进自身的提示、技能、工具和工作流程，从而提升了对软件工程基准的表现。

**💡 创新点**

创新点在于将执行验证与自我进化结合，构建了四类工具（归因、验证、训练审计、工作流程）并让优化器在循环中更新自身harness，实现了对自适应修改的系统化管理。

**🔧 技术方法**

采用LLM（Qwen3.8-Flash-Next和Qwen3.8-27B）、harness演化框架、执行验证工具、trace最小化和基于验证的选择机制。

**📊 数据集**

主要使用SWE‑rebench（Python任务及OOV的Go、Java、Rust、TypeScript）和Terminal‑Bench。

**📈 对比分析**

在四个基线harness优化器（Meta‑Harness、AHE、Self‑Harness、HarnessX）上添加VERSE后，在内部测试集上平均提升约3–10点，在OOV任务上提升2–10点，验证选取的harness往往优于最终轮次。

**⚠️ 局限性**

局限性包括对验证工具的依赖，某些主机或语言的提升有限，且自我进化过程的计算开销和可解释性仍需进一步研究。

---

## 242. Scale-Recursive Rectified Flows for Few-Step Precipitation Ensembles

**arXiv ID:** 2610.02611 | [PDF](https://arxiv.org/pdf/2610.02611v1)

**作者:** Shunya Nagashima `[一作]` (Neurogica Inc), Takumi Bannai `[通讯]` (LTS Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种尺度递归Rectified Flow，将采样步骤在粗尺度和细尺度之间分配，以提高降水下尺度估计的概率质量。

**💡 创新点**

创新点在于利用各尺度的扩散-误差缺口诊断自动分配采样步骤，并引入低通尺度可扩展校准，既提高CRPS/CIS，又保持相同计算预算。

**🔧 技术方法**

采用Wavelet分解、Rectified Flow生成器、条件残差建模、可变步长Euler积分以及空间功率谱校准技术。

**📊 数据集**

训练集为IMERG-Late 0.1°卫星预报，目标为1 km分辨率的MRMS雷达数据，覆盖美国CONUS 2021-2024年。

**📈 对比分析**

在相同计算成本下与单流、像素域流、重构单流、Churn/Repulsion等基线比较，递归模型在成本八/十六时CRPS下降约0.001-0.002、CSI提升约0.01，且采样速度更快。

**⚠️ 局限性**

局限在于仅针对降水下尺度验证，强降雨校准仍不稳定，对不同训练种子结果差异显著，且未扩展到其他多尺度生成任务。

---

## 243. Query Performance Tuning with Optimal Exploration of Optimizer Cost Model Parameter Space

**arXiv ID:** 2610.02607 | [PDF](https://arxiv.org/pdf/2610.02607v1)

**作者:** Wentao Wu `[一作]` (Microsoft Research), Surajit Chaudhuri `[通讯]` (Microsoft Research)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种确定性、最优的 CMP 调优框架，能够完整探索查询优化器的成本模型参数空间，并在此基础上通过超时与计划相似度剪枝大幅降低候选计划的评估时间，最终为单个查询寻找最优执行计划。

**💡 创新点**

创新点：① 将查询优化器的线性成本模型与参数空间分解结合，提出基于二叉空间划分（BSP）的确定性探索算法，保证能枚举到所有可能计划；② 设计了进化式超时策略与两类相似度剪枝（嵌入式和瓶颈算子匹配），使得在保证最优性的同时显著减少评估开销；③ 通过对比随机搜索和贝叶斯优化，证明了该方法在可重复性和性能上的优势。

**🔧 技术方法**

技术手段：二叉空间划分（BSP）计划空间搜索、进化式/轮询式超时裁剪、基于 AMA 的计划嵌入相似度判定、瓶颈算子匹配剪枝、PostgreSQL 与 Microsoft SQL Server 的成本模型参数（seq_page_cost、random_page_cost、cpu_tuple_cost、cpu_index_tuple_cost、cpu_operator_cost）以及实验中使用的查询计划生成与执行框架。

**📊 数据集**

数据集：TPC‑H、TPC‑DS（scaling factor 1、10、25、100）、Join Order Benchmark (JOB)、Cardinality Estimation Benchmark (STATS)，并在 PostgreSQL 17.4 与 Microsoft SQL Server 上进行实验。

**📈 对比分析**

比较方法：与默认计划、随机搜索（RS）和贝叶斯优化（BO）三种基线进行对比，评估指标包括执行时间提升（相对默认）和调优总耗时。实验结果显示，DOT 在多数工作负载中将执行时间提升 30%–81% 以上，且在多次调优后相对于 RS/BO 的评估开销显著降低，整体在执行时间与调优时间之间取得更优平衡。

**⚠️ 局限性**

局限性：① 仅针对单个查询进行调优，缺乏跨查询工作负载级别的参数共享与预测；② 计划空间探索虽然完整但在某些查询上会产生过多的优化器调用，仍有进一步减少调用次数的空间；③ 依赖优化器的确定性行为，对非确定性优化器可能需要额外处理；④ 目前未考虑在线快速调优或动态重排等实时需求。

---

## 244. Dynamic LLM Routers are Often Misguided

**arXiv ID:** 2610.02762 | [PDF](https://arxiv.org/pdf/2610.02762v1)

**作者:** Sam Wang `[一作]` (Fastino Labs), Kelton Zhang `[通讯]` (Fastino Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文评估了六个商业LLM路由器的表现，揭示了四种常见缺陷（难度盲目、长度反转、语义匹配、名册非最优），并提出了不奖励这些模式的改进评估方法，随后构建了一个两模型路由器证明这些缺陷可被避免，尽管其性能提升有限。

**💡 创新点**

创新点在于系统地识别并解释路由器的四种普遍错误模式、证明标准Pareto效率评估会鼓励这些错误，并提出了一套新的评估框架来消除这些偏差；同时通过实验验证名册大小并非提升性能的关键。

**🔧 技术方法**

使用Rasch/IRT模型估计模型能力与查询难度，计算Goodman-Kruskal gamma、AMI等统计量来量化路由策略；设计基于共享枢纽网络的难度预测器，并通过阈值控制构造简单的两模型路由器；还使用多基准对比方法评估成本-准确率表现。

**📊 数据集**

实验基于17个跨八类（编码、指令、知识、数学、问答、办公、工具使用、Humanity's Last Exam）基准的数据集，包含2508训练样本和800评估样本；此外复现了 RouterBench、SPROUT、LLMRouterBench 等公开基准以检验结果的一致性。

**📈 对比分析**

通过将商业路由器与随机选择{Gemini 3.7 Flash, Opus 5 (high)}的两模型基线在相同成本下进行对比，并使用改进评估指标（难度带升级率、长度/语义匹配）进行细粒度分析。结果显示大多数商业路由器在大多数预算下的准确率低于或仅与随机基线持平，差距可达10个百分点以上；改进路由器在改进评估下能提升难度带升级率，但整体准确提升仅在噪声范围内。

**⚠️ 局限性**

限制包括仅考虑单轮路由且未探讨多轮KV缓存成本；依赖基准任务而非真实用户流量；难度预测受限，尤其是OOV场景；所分析模型中专精度有限，未来专精模型可能改变结论；改进评估框架基于若干假设，仍需在更广泛场景验证。

---

## 245. Exact Memory-Time Optimization for Prefix-Cached Language Model Serving

**arXiv ID:** 2610.02766 | [PDF](https://arxiv.org/pdf/2610.02766v1)

**作者:** Shivam Gupta `[一作]` `[通讯]`, Shivam Gupta

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了一种针对语言模型服务的静态前缀保留（Prefix-Certificate Retention, PCR）最优配置模型，将可重用前缀的依赖关系转换为最大权闭包问题，并给出线性时间动态规划与断点定理实现连续超时优化。

**💡 创新点**

创新点在于：1) 用前缀证书显式建模可用前缀的必然条件，消除传统独立保留估计的过度乐观；2) 引入断点定理，使连续超时问题可映射到有限网格上；3) 提供可合成的线性时间动态规划和三向最优性上界；4) 通过实验验证模型可与现有的统一或分组 TTL 基线进行严格对比。

**🔧 技术方法**

技术手段包括：图论（最大权闭包→最小割）、线性规划、动态规划、事件排布与时间分辨率敏感性分析、Python/NumPy/PyMaxflow实现、以及多重实验验证（包括网格化与连续时间、合成与真实轨迹、不同分组策略）。

**📊 数据集**

使用公开的 Mooncake FAST'25 轨迹，包括 Conversation、Tool agent 两类真实工作负载和一份 Synthetic 负载，共计 39,632 条请求，训练/验证/测试按时间顺序划分。

**📈 对比分析**

方法通过在训练窗口上离线求解 PCR（包括可限制/不限制的超时、分组/深度组合），并将得到的配置在测试窗口下进行批量重放与对比。实验结果显示：1) 在绝大多数场景下，无需放宽前缀一致性约束即可获得最优或近似最优；2) 在部分存储价格区间，分组/深度混合策略可略微提升可重用率；3) 细粒度连续超时优化不一定能在未来窗口中提升性能；4) 计算成本低，可在秒级内完成，适合离线配置。

**⚠️ 局限性**

局限性：仅适用于静态离线轨迹，无法处理实时在线自适应；不考虑硬容量约束、请求并发、模型版本失效、活跃请求锁定和预填计算/传输开销；奖励仅为完整前缀块计数，无法直接映射到延迟或成本；实验数据集有限，难以证明在其他模型或更大规模负载下的泛化能力。

---

## 246. Correcting Guided Diffusion Trajectories with Spectral Alignment

**arXiv ID:** 2610.02753 | [PDF](https://arxiv.org/pdf/2610.02753v1)

**作者:** Gihoon Kim `[一作]` (Seoul National University), Taesup Kim `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了基于谱对齐的训练无关校正方法 Spectral Correction Guidance，用于改进条件扩散模型的引导采样。

**💡 创新点**

创新点在于利用自然图像谱的解析参考来检测并修正采样过程中谱偏差，从而提供自适应的谱校正。

**🔧 技术方法**

技术包括谱距离度量、解析参考谱计算、对逆向状态的对数谱插值校正，以及对校正强度与松弛参数的调节。

**📊 数据集**

使用的数据集包括 COCO 验证集（文本到图像）和 ImageNet（类别条件生成），并在 SDXL、PixArt‑α、SD3.5、DiT‑XL/2 等模型上验证。

**📈 对比分析**

与标准 CFG、TV‑CFG、CFG++、LF‑CFG 等方法对比，Spectral Correction Guidance 在 HPSv3、ImageReward、PickScore、CLIP‑T、FID 等指标上均取得显著提升，尤其在高引导尺度和低采样步数下表现优异。

**⚠️ 局限性**

局限性在于需预估谱系数并在不同模型/噪声计划上手工调参，且过强或过弱的校正会导致失真或不足；对细节的过度校正也可能产生伪影。

---

## 247. Learning Query Encoders Can Be Hard Even When Vector Retrieval Is Geometrically Easy

**arXiv ID:** 2610.02749 | [PDF](https://arxiv.org/pdf/2610.02749v1)

**作者:** Anders Wikum `[一作]` (Stanford University), Tal Wagner `[通讯]` (Amazon AWS)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究文档索引的几何容量与查询编码器学习的可行性，提出通过排名SVM计算几何容量下界，并在多种检索基准上实验检索召回差距；同时构造了一个检索任务，证明存在可由小型ReLU网络实现完美召回但在统计查询模型下学习极其困难。

**💡 创新点**

创新点在于：①提出实用的几何容量下界评估方法，量化索引几何对检索召回的上限；②通过实证发现即使几何容量高，单向量查询编码器往往无法达到该上限；③在理论上给出统计查询模型下的学习难度下界，说明高几何容量不等于可学习的查询编码器。

**🔧 技术方法**

主要技术包括：排名支持向量机（Ranking SVM）进行下界优化；单向量查询编码器的训练与微调（LoRA + InfoNCE）；统计查询（SQ）学习模型及其下界证明；ReLU神经网络实现检索规则的构造。

**📊 数据集**

使用了三大检索基准：LIMIT（50k文档的合成集合），BRIGHT（12类问题文档集），BEIR（7个多样化子任务）。

**📈 对比分析**

在实验中，几何容量下界平均超过95%召回，而各预训练/微调的单向量查询编码器在测试集上召回仅低于50%，部分甚至不到25%；对比多向量检索（ColBERT）和词法检索在某些基准上已达到近乎完美召回。

**⚠️ 局限性**

局限性包括：训练样本不足导致微调泛化差；评估仅针对单向量检索，未覆盖多向量或混合方案；理论硬度仅在构造的极端任务中体现，实际任务是否普遍存在此难点尚未证明；以及几何容量估计依赖于排名SVM的近似求解，可能有误差。

---

## 248. CSIR: Contextually and Socially Informed Robots for Efficient Person Goal Navigation

**arXiv ID:** 2610.02750 | [PDF](https://arxiv.org/pdf/2610.02750v1)

**作者:** Tyler Chung `[一作]` (University of Waterloo), Yue Hu `[通讯]` (University of Waterloo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `67630363-6be0-4f51-ab05-7198250671a5`

**🎯 论文内容**

提出了基于语义与距离信息相结合、并按信息置信度加权的CSIR框架，用以解决室内人员目标导航（PersonNav）问题，并构建了可控的合成基准；

**💡 创新点**

创新点在于：①将自然语言提示拆分为多种信息源（行为、语境、最近位置等），通过置信度权重融合成概率分布；②使用加权旅行维修商（WTRP）与动态规划求解最优搜索路径；③通过合成场景生成器提供可重复的评估环境；④显式透明的置信度分布可单独调节。

**🔧 技术方法**

技术主要包括：大语言模型（GPT‑4o‑mini）用于合成场景和解析请求；BERT/Transformer 估计语言不确定度；多源置信度生成与归一化；加权TRP（Held‑Karp 动态规划）；基于图的随机、贪心、TSP、语义贪心、语义+距离基线；Nav2、YOLO、DeepFace用于硬件实验。

**📊 数据集**

数据集：①使用 GPT‑4o‑mini 生成的 1,500 个合成场景（包括 30 名演员、10 件物品，3 种环境：办公室、医院、大学楼）；②真实硬件实验在一所大学建筑中进行，使用激光扫描生成的语义标注地图。

**📈 对比分析**

方法通过对比 6 种基线（随机、贪心、TSP、语义贪心、语义+距离、语义 TRP）评估，主要指标包括 Top‑3 准确率、访问地点数、行驶效率和规划时间。CSIR 在所有环境中实现最高平均行驶效率 0.62（相当于 LLM‑TRP 0.60），并在 Top‑3 准确率上优于其他基线；规划时间在 20 位置以内可在秒级完成。

**⚠️ 局限性**

局限性包括：①规划复杂度随节点数指数增长，限制大规模环境使用；②假设物品/职业信息为真实，实际可能出现误差；③未在搜索过程中实时更新置信度（仅靠事先生成的分布）；④实验主要集中在室内固定地图，未覆盖开放或动态环境；⑤隐私和安全约束仍需进一步研究。

---

## 249. Structural-Functional Brain Connectivity Generation via Multimodal Hypergraph-based Flow Matching

**arXiv ID:** 2610.02722 | [PDF](https://arxiv.org/pdf/2610.02722v1)

**作者:** Chyong Yi Poh `[一作]` (Monash University Malaysia), Chee-Ming Ting `[通讯]` (Monash University Malaysia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于多模态超图流匹配（MHG-FM）框架，用于同时生成结构连接（SC）和功能连接（FC），并实现两者之间的双向跨模态翻译。

**💡 创新点**

创新点包括：① 用超图结构捕获大脑多节点间的高阶依赖；② 通过双向交叉注意力（Dual Cross-Attention）融合结构与功能特征，强化结构-功能耦合；③ 在共享潜在空间中采用条件流匹配（Conditional Flow Matching）实现高质量且采样高效的生成；④ 统一模型即可完成生成和翻译，避免单独训练多模型。

**🔧 技术方法**

技术手段：超图神经网络（HGNN）编码、双向交叉注意力融合、变分自编码器（VAE）映射潜在空间、条件流匹配训练目标、ODE求解采样、对比实验使用WGAN、VAE、topoGAN、MGCN-GAN、HYGENE、MHG-DiT等基线。

**📊 数据集**

使用 Human Connectome Project Young Adult 3T (HCP-YA 3T) 数据集，包含 562 训练、70 验证、70 测试个体，采用 AAL-116 分区。

**📈 对比分析**

与多种基线对比：MHG-FM 在矩阵层面（Pearson r、R²、MAE 等）与流匹配基线 MHG-DiT 竞争，整体保留网络拓扑（Degree/W_1、Edge/W_1、NMI、ARI、|ΔMI|）最佳；生成速度约比 MHG-DiT 快 8 倍；在 FC→SC 与 SC→FC 的跨模态翻译任务中，MHG-FM 超越 topoGAN、MGCN-GAN，具有更低的 Frobenius 误差和更高的相关性。

**⚠️ 局限性**

局限性：仅在 HCP-YA 3T 健康年轻人数据上验证，尚未评估在临床多站点或动态功能连接上的泛化能力；超图构建依赖阈值和聚类规则，可能影响可解释性；尽管采样比扩散更快，但仍需 ODE 求解，计算开销不如单次前向网络。

---

## 250. TPBench: A Turning-Point Benchmark for Dialogue Compression

**arXiv ID:** 2610.02736 | [PDF](https://arxiv.org/pdf/2610.02736v1)

**作者:** Minji Park `[一作]` (Korea Institute of Energy Technology), Hyuk Lim `[通讯]` (Korea Institute of Energy Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 TPBench，一套用于评估对话压缩方法的基准框架，包含三种探针（P1 初始目标、P2 当前值、P3 初始目标与晚期更新联合恢复），通过读取压缩后的对话并回答自然语言问题来衡量保留信息的能力。

**💡 创新点**

核心创新在于将对话压缩的评估拆分为三种互补的目标，避免将不同信息需求混合成单一分数；同时构建了无需人工标注的新方法，可直接从现有的 SGD 与 MultiWOZ 对话状态注释中生成评测问题；进一步通过删除关键更新回合的实验揭示了“转折点驱逐”现象。

**🔧 技术方法**

采用基于 Transformer 的长文本压缩技术，包括提示侧压缩（LLMLingua‑2、H2O 代理、MMR、随机、最早/最新、均匀步进等）与 KV‑缓存压缩（ChunkKV、SnapKV、PyramidKV、StreamingLLM）；使用 Llama‑3.1‑8B‑Instruct 和 Mistral‑7B‑Instruct 作为阅读器；采用句子级答案匹配、词元 F1、以及人工辅助语义等多种评分规则。

**📊 数据集**

数据集主要为英文学术任务导向语料 SGD 与 MultiWOZ 2.2；额外实验使用 LongMemEval‑KU（自由文本记忆查询）和 RiSAWOZ（中文对话）验证跨语种与跨任务的适用性。

**📈 对比分析**

对七种轮次选择器和四种 KV‑缓存压缩方法，在 0.30 保留比例下，P1、P2 与 P3 的排名差异显著；例如在 P1 上均匀步进（uniform stride）与 H2O 代理表现最好，而在 P2 上 MMR 或随机表现领先；在 P3（联合）上全局最佳仍为均匀步进，但与完整上下文相比仍落后 28–36 个百分点。更宽的预算（0.50/0.70）可缩小 P2 与完整上下文的差距，KV‑缓存方法在 MultiWOZ 上的 P2 分数显著高于轮次选择器。

**⚠️ 局限性**

限制主要包括：仅评估基于槽位标注的英文任务导向对话；缺乏对开放式、多人或无槽位目标的通用性验证；评测依赖固定的答案匹配规则，无法覆盖所有同义表达；未进行人工真实评估；以及对中文、自由文本的实验样本量相对有限。

---

## 251. RoboBridge: A Self-Evolving Embodied Agent Framework for Sim-to-Real Transfer

**arXiv ID:** 2610.02717 | [PDF](https://arxiv.org/pdf/2610.02717v1)

**作者:** Chenxi Li `[一作]` (Shanghai Artificial Intelligence Laboratory), Dongzhan Zhou `[通讯]` (Shenzhen Institutes of Advanced Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为RoboBridge的自我演化式框架，用于将基于仿真的视觉‑语言‑动作（VLA）策略迁移至真实机器人，并通过实时执行反馈持续优化任务技能。

**💡 创新点**

创新点：1) 将预训练VLA模型包装成可重用的动作工具，同时保持其冻结参数；2) 通过明确的任务程序（skill）将任务知识与工具调用分离，支持程序级的增删改；3) 在工具调用时引入推理时奖励引导，实现对冻结策略的细粒度调优；4) 采用交互式反馈验证候选技能更新，避免错误经验污染；5) 通过仿真演化得到的技能可直接映射到物理环境，并在物理执行中进一步演化。

**🔧 技术方法**

核心技术包括：预训练的端到端VLA策略（π_0.5-SFT/π_0.5-DROID）；统一的观测与工具接口；基于奖励梯度的推理时指导；基于执行日志的技能更新与验证机制；以及从仿真到现实的任务语义与接口映射。

**📊 数据集**

主要使用的数据集与任务场景是LIBERO-PRO仿真平台（SPATIAL、OBJECT、GOAL、LIBERO-10四个任务组）以及对应的三项真实机器人任务（两方块放入碗、蛋糕放入锅并盖上、笔放入笔筒）。

**📈 对比分析**

对比方法：对比基准包括直接使用π_RLinf、Harness VLA以及在不同实验配置下的完整RoboBridge。结果显示：在LIBERO-PRO上，完整框架在四个任务组的总体成功率为74.4%，超过Harness VLA（72.1%）并显著提升GOAL-T任务；在真实机器人上，直接迁移的技能已将成功率从约25%提升至约50%，进一步通过物理演化后可达接近100%，并将从零开始演化所需的迭代次数和Token消耗分别降低约25%和75%。

**⚠️ 局限性**

局限性：实验仅在Codex + GPT‑5.5的编码器环境下进行，未验证对其他大型模型或不同代理框架的泛化；技能更新仍需人工或自动化验证，规模化部署的验证成本未知；仿真与现实的映射Φ仍需要针对不同平台手工对齐；并且推理时奖励引导的参数调优仍依赖经验。

---

## 252. Around the World: Unified Learned Locomotion on a 270 g Continuous-Rotation Quadruped

**arXiv ID:** 2610.02728 | [PDF](https://arxiv.org/pdf/2610.02728v1)

**作者:** Arturo Flores Alvarez `[一作]` (University of California Los Angeles), Dennis Hong `[通讯]` (University of California Los Angeles)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在270克的微型四足机器人MiNI-Q上实现了闭环强化学习驱动的全周期行走与着陆恢复，采用单一姿态条件化网络在机器人本地双微控制器上实时执行。

**💡 创新点**

创新点包括：
- 利用连续旋转关节将姿态空间映射到8维环面（T^8），实现正面、倒立行走与中途姿态切换无需状态机；
- 引入重力条件化参考和镜像变换，实现姿态与重力方向同步，统一控制正倒两种体态；
- 在网络输入中采用周期化观测（sin/cos角度）与三帧动作历史，保持关节角度连续性；
- 通过姿态插值与发布课程训练模型，使其在姿态变化和自由落体恢复中保持鲁棒；
- 对执行机构进行动力学识别并在模拟中进行跨引擎验证，支持不同脚形的无缝迁移；
- 采用SIMD加速的嵌入式推理，使网络在双核ESP32上以50Hz执行并保持低内存占用。

**🔧 技术方法**

技术手段包括：
- 端到端强化学习（PPO）在4096并行Isaac Gym环境中训练；
- 关节姿态的环面编码、toroidal距离度量与姿态插值；
- 观测中的三帧动作历史、重力投影、姿态参考的周期化嵌入；
- 作用力和位置驱动的机械辨识（单腿悬挂实验）并将识别参数用于模拟；
- 基于ESP32-S3的双微控制器架构，使用SIMD（PIE128）实现高效前向传播；
- 脚形随机化和跨脚形验证，提升对接触几何的泛化能力。

**📊 数据集**

使用的数据集为：
- 4096个并行的模拟环境（Isaac Gym），每个环境包括随机的姿态、速度命令和脚形随机化；
- 训练中使用的特权观测（位置、速度、质量参数等）仅用于训练时的critic；
- 通过实际机器人进行硬件验证，使用ZED2摄像头获取位姿轨迹，对比真实与命令速度。

**📈 对比分析**

比较方法与性能：
- 与六个关键干预（无环面编码、无镜像变换、无抛射课程、离散姿态、无脚形随机化）进行模拟消融，显示正倒行走的RMSE在倒立时从1.00降到0.02，恢复时间从6.9s提升到10s；
- 硬件上正向和倒向速度RMSE分别为0.037 m/s与0.050 m/s；
- 80%（24/30）恢复成功率；
- 脚形泛化实验中，RMSE在未见脚形与不同表面组合下保持在0.14 m/s以内，平均0.087 m/s。

**⚠️ 局限性**

局限性包括：
- 训练稳定性受限于姿态对的相对距离，难以一次性引入过远的姿态；
- 仅在室内平整表面验证，未测试户外多样地形或高初始角速度；
- 对硬件的影响力耐受性不足，发生过电流、机电接触脱落或齿轮滑移导致失效；
- 由于未对脚形随机化的真实收益做更细粒度的对比，缺乏对单一脚形与随机化策略的成本效益评估；
- 目前采用单一网络无法针对不同任务（如跳跃、攀爬）进行动态策略切换，未来需要更丰富的姿态与运动模式库。

---

## 253. AdaTempo: Learning Shared Relative Tempo from Demonstrations for Faster Robot Manipulation

**arXiv ID:** 2610.02706 | [PDF](https://arxiv.org/pdf/2610.02706v1)

**作者:** Jiale Cao `[一作]` (Fudan University), Huazhe Xu `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

AdaTempo通过对演示轨迹进行相对节奏对齐、聚合并生成连续加速曲线，进而离线地重采样演示数据，训练出速度更快、成功率更高的视动机策略。

**💡 创新点**

创新点在于利用演示间共享的相对节奏作为自监督信号，构造无运行时开销的连续加速配置，彻底摆脱了传统的均匀或基于代理的加速方案。

**🔧 技术方法**

技术方法包括：动态时间规整（DTW）实现相位对齐、跨演示相对节奏一致性聚合、基于置信度的加速权重、软最小化与斜率限制的连续速度映射，以及非整数步长的时间虚采样与插值。

**📊 数据集**

使用数据集包括：Aloha MuJoCo（Transfer Cube、Insertion）与Robomimic（Can、Lift、Square）5个仿真任务，以及AgileX ALOHA双臂真实平台的6个任务（Picking Cube、Sorting、Pouring、Folding、Stacking Cups、Conveyor Fast）。

**📈 对比分析**

与原始策略、统一加速、DemoSpeedup、SAIL以及DP-Fast等基线相比，AdaTempo在11个任务中平均可实现约2.8×的演示速度提升，同时保持或提高成功率（仿真平均成功率75.5%，真实任务平均成功率77.8%），在单个任务上甚至达到3.57×的加速。

**⚠️ 局限性**

局限性包括：需为每个任务手工选择加速上下限、对演示中不同策略或停顿不够鲁棒、无在线适应机制、以及高加速可能导致跟踪误差、执行极限等物理限制无法完全保证安全。

---

## 254. A Two-Stage Cascade for Near-Real-Time Forest Anomaly Detection from Sentinel-1 SAR Time Series

**arXiv ID:** 2610.02763 | [PDF](https://arxiv.org/pdf/2610.02763v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 255. A Controlled Audit of Personal AI Memory for Rating Prediction

**arXiv ID:** 2610.02764 | [PDF](https://arxiv.org/pdf/2610.02764v1)

**作者:** Shivam Gupta `[一作]` `[通讯]` (Independent research), Shivam Gupta (Independent research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在个人 AI 记忆与评级预测的交叉点，作者通过保持评级分布不变的置换控制、内置记忆写入器、匹配数值阅读器以及全历史对照，构建了一个可复现的诊断实验，用来分离记忆提取、历史关联使用、读取器行为与输出可靠性等因素在两域（Coat 与 MovieLens）中的作用；

**💡 创新点**

1）使用评级分布保持置换控制，明确区分记忆提取与历史关联的贡献；2）提供冻结评估、完整失败计数与资源消耗记录，实现透明可审计；3）在同一历史记录下进行写入器/读取器匹配对照，系统性检验多种组件的交互影响；

**🔧 技术方法**

Qwen3-4B-Instruct 与 Phi-4 语言模型作为读取器，Mem0 2.1.0 作为记忆写入器，岭回归数值阅读器；配合配对 Bootstrap 置信区间、配对差异检验、排序一致性等统计方法；

**📊 数据集**

Coat 购物实验数据（290 用户，24 条历史 + 16 条随机目标）与 MovieLens 100K（645 用户，24 条历史 + 16 条目标），仅使用项目属性与评分，去掉标题、评论、时间戳等敏感信息；

**📈 对比分析**

与无历史、全历史、记忆写入、置换历史等十个预声明对照进行比较，主要指标为用户宏平均 MAE、RMSE 与排序一致性。实验结果显示：在 Coat 上，内置记忆相较全历史会增加误差，但正确赋值对预测有显著帮助；在 MovieLens 关联效应不显著；岭回归在两域均优于语言模型读取器；数值读者表现稳定但失败率因条件而异；

**⚠️ 局限性**

仅限短期结构化历史，未评估长期对话、检索或增量更新；实验仅涵盖两域，缺乏多样性；记忆写入器仅一次性评估，未检验不同写入方式或检索策略；结果受数据集采样、预训练风险等因素限制，无法直接推广至大规模商业系统。

---

## 256. Test-time Calibration Learning for Large Language Model Reasoning

**arXiv ID:** 2610.02695 | [PDF](https://arxiv.org/pdf/2610.02695v1)

**作者:** Zizhuo Zhang `[一作]` (Hong Kong Baptist University), Bo Han `[通讯]` (Hong Kong Baptist University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在测试时对大型语言模型进行无标签校准学习（Test‑Time Calibration Learning, TTCL）的框架，能够在没有真值标签的前提下自适应地提升模型的答案准确性与置信度校准；

**💡 创新点**

核心创新在于：①利用多次采样的模型生成答案的频率来构造“自监督”校准目标；②将正确性奖励与置信度奖励分离，分别作用于答案与置信度 token；③引入目标缓存（target cache）与指数滑动平均（EMA）来稳定训练，避免自我强化导致的崩溃；

**🔧 技术方法**

技术手段包括：多回合采样（G=32）、PPO/GRPO 风格的策略梯度优化、Brier score 作为置信度损失、目标缓存的 EMA 更新、以及与 RLCR、P(True)、Verbalization 等基线的对比实验；

**📊 数据集**

使用了数学推理数据集（MATH500、AMC、AIME24/25/26、DAPO‑Math‑14K）和事实问答数据集（SimpleQA、NQ、HotpotQA、TriviaQA、ChineseSimpleQA、FactQA‑10K）以及跨域评估（ARC、GPQA‑D、IFEval、MMLU‑Pro、TruthfulQA）进行实验；

**📈 对比分析**

与基于标签的 RLCR、无标签的 P(True)、Verbalization、Elicitation、BaseCal 等方法对比，TTCL 在所有测试集上均实现了显著的准确率提升（平均 +30–40%）和校准误差下降（平均 ECE 降低 60–70%），并在域迁移场景中进一步改善已预校准模型的性能；

**⚠️ 局限性**

局限性包括：①依赖多次采样，计算成本相对较高；②目标缓存参数（如 EMA 率）需要手动调优；③在极端噪声或多模答案环境下，频率作为置信度近似可能不够精准；④未对连续决策任务或长文本生成的实时校准进行验证。

---

## 257. GeoScaffold: Learning Compact Geometric Latents via Reconstruction for Efficient Vision-Language Navigation

**arXiv ID:** 2610.02697 | [PDF](https://arxiv.org/pdf/2610.02697v1)

**作者:** Yixuan Jiang `[一作]` (Nanjing University), Jian Cheng `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 GeoScaffold，通过在训练阶段使用几何监督（深度、连通性、可通行性）将几何信息内化到流式 VLN 策略中，最终仅用 RGB 与语言即可执行导航。

**💡 创新点**

创新点在于仅在训练阶段引入几何监督，并通过少量查询 token 将几何信息压缩为紧凑的潜在表示，部署时无需深度传感器、3D 编码器或额外的几何模块，保持轻量化。

**🔧 技术方法**

采用 VQ‑VAE 离散深度编码器、查询 token 监督、混合注意力掩码、视频‑LLM 架构以及多目标重建损失（深度、连通性、可通行性）。

**📊 数据集**

在 Matterport3D 的 VLN‑CE 环境下使用 R2R‑CE、RxR‑CE、DAgger 等轨迹，并利用 Grounding‑DINO/SAM 生成的深度、连通性与可通行性标签作为训练时监督。

**📈 对比分析**

与 JanusVLN、GA‑VLN、StreamVLN 等多种基线对比，GeoScaffold 在 R2R‑CE 和 RxR‑CE 上显著提升 SR 与 SPL（最高 64.8% SR、60.5% SPL），并将每帧推理延迟压缩至 81 ms，远优于需要实时几何模块的模型。

**⚠️ 局限性**

局限包括对离散深度码表的依赖、训练时需额外生成几何标签、以及在极端复杂几何场景下仍可能需要进一步强化监督。

---

## 258. ServeTwin: A Benchmark-Validated Simulator for Distributed LLM Architecture Exploration

**arXiv ID:** 2610.02732 | [PDF](https://arxiv.org/pdf/2610.02732v1)

**作者:** Sungjoon Park `[一作]` (Samsung Advanced Institute of Technology), Sangjoon Kim `[通讯]` (Samsung Advanced Institute of Technology)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种闭环模拟器，结合 KV 生命周期管理与无需硬件剖面的分析时序模型，能够在物理集群可用前评估分布式 LLM 服役系统的吞吐率、交互延迟与 KV 缓存动态；

**💡 创新点**

创新点包括：①基于状态机的 KV 生命周期和多轮会话建模，实现真实的前填充-解码拆分与 KV 传输；②iSTAGE 通过规范驱动的符号图生成，避免目标硬件剖面；③分层成本归属（吞吐、调度、运行时）使参数可按组件重用；

**🔧 技术方法**

技术实现涵盖：状态闭环调度、iSTAGE 符号时序推导、ASTRA-sim 网络层仿真、CUDA-graph 采样与压缩、组件归属成本模型；

**📊 数据集**

使用公开基准：InferenceX（steady‑state吞吐–交互曲线）与 LMBenchmark（多轮会话动态），以及 DeepSeek‑R1、Qwen3‑32B 等大模型；

**📈 对比分析**

通过与真实部署（H200/H100/NVL72）对比，InreformanceX 前沿误差仅 3.6% 以内，LMBenchmark 多轮误差 ≤10%；在预硅设计探索中揭示高带宽与容量在不同工作负载下的优先级互换，并证明软件调度/运行时瓶颈在高并发时超过内存带宽；

**⚠️ 局限性**

局限性在于：①仍需手动对每个组件进行参数迁移，若出现未建模的新机制需重新校准；②仿真精度依赖于硬件规范与模型公式的完整性，对极端异构场景或自研框架的兼容性尚未充分验证；

---

## 259. Beyond Correctness: Resolving Underspecification in Agentic Text-to-SQL

**arXiv ID:** 2610.02739 | [PDF](https://arxiv.org/pdf/2610.02739v1)

**作者:** Wen-Zhi Li `[一作]` (Cornell University), Balakrishnan Murali Narayanaswamy `[通讯]` (Amazon Web Services)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了 PlanPool 机制，在文本到 SQL 的交互式问答中将澄清计划外化为可变问题池，以防止未解决的歧义被忽略并提升基于执行结果的 SQL 准确率与根源清晰度。

**💡 创新点**

创新点在于将澄清计划从模型内部文本转换为可显式管理的状态，要求在提交 SQL 前显式处理所有待解决问题，并在交互过程中动态增删问题，显著提升歧义覆盖率和减少静默失败。

**🔧 技术方法**

使用 LLM 代理（Claude‑Opus‑4.8）实现交互式探索、提问和 SQL 生成，结合基于工具的数据库查询、规划与动作执行框架，采用外部化问题池的策略。

**📊 数据集**

在 BIRD‑Interact（Lite、Full）和 Spider 的三个扩展子集上进行评估，分别包含多重注解歧义的问句。

**📈 对比分析**

与 Naive、Read‑All、Regenerate、Draft、Self‑Reflection 以及 Prompt‑Planning 等六种基线对比，PlanPool 在歧义回忆率和根源成功率上均超过 Prompt‑Planning，提升约 7–10%，且在 BIRD‑Lite 上实现 49.1% 根源成功率，同时保持竞争性的执行准确率。

**⚠️ 局限性**

主要局限是交互成本略高（需要更多澄清问题），且在极大数据库或更复杂歧义场景下问题池管理仍可能出现遗漏，未来需进一步优化动态增删策略与成本平衡。

---

## 260. GAANet: Global-guided Asymmetric Attention Network for Audio-Visual Speech Separation

**arXiv ID:** 2610.02752 | [PDF](https://arxiv.org/pdf/2610.02752v1)

**作者:** Zhiyuan Zhang `[一作]` (Hefei University of Technology), Dan Guo `[通讯]` (Hefei University of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种全新的音视听语音分离网络GAANet，实现对混合语音的高质量分离。

**💡 创新点**

核心创新包括：① 异步多尺度融合框架，音频和视频分别以最适时域分辨率提取特征；② 全局引导注意力机制，将每模态压缩为时间维度为1的全局Token，为跨尺度的内模态与跨模态融合提供一致的语义上下文。

**🔧 技术方法**

使用1D卷积、层归一化、PReLU激活、全局注意力、轻量级FFN以及多次迭代的融合与细化模块，最终通过转置卷积解码成目标语音。

**📊 数据集**

在LRS2和VoxCeleb2两大公开音视听语音分离基准集上进行实验。

**📈 对比分析**

与多种SOTA方法对比，GAANet在LRS2上SI‑SNRi达16.5 dB、SDRi 16.64 dB；在VoxCeleb2上SI‑SNRi 14.0 dB、SDRi 14.7 dB，参数仅3.3M、MACs 19.8G，展示了在轻量化与性能上的优越平衡。

**⚠️ 局限性**

局限性包括：在极低SNR或视频帧缺失情况下的鲁棒性尚待进一步提升；目前仅支持单说话人分离，扩展到多说话人场景仍是挑战。

---

## 261. Inner Momentum for Differentially Private Muon

**arXiv ID:** 2610.02738 | [PDF](https://arxiv.org/pdf/2610.02738v1)

**作者:** Bishnu Bhusal `[一作]` (Los Alamos National Laboratory), Manish Bhattarai `[通讯]` (Los Alamos National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究差分隐私训练中 Muon 优化器受梯度裁剪影响的几何失真，并提出在裁剪前对每个样本的梯度进行跨模型状态平均（DP‑Muon‑IM），从而降低裁剪残差并提升私有 GPT‑2 微调效果。

**💡 创新点**

创新点在于：①对裁剪残差对 Muon 极限因子的影响进行解析，证明 Newton–Schulz 迭代保持极限因子不变；②提出“内动量”方案，即在裁剪前对同一样本在当前及最近历史模型上的梯度做加权平均，显著减少裁剪残差和极限误差；③在保持隐私会计不变的前提下，将此方法应用于 Muon 训练，提升生成质量。

**🔧 技术方法**

使用技术包括：差分隐私 SGD（梯度裁剪 + 高斯噪声）、Muon 优化器（矩阵正交化 + 5 步 Newton–Schulz）、内动量平均（跨模型状态梯度加权）、Poisson 采样 + PRV 计数隐私会计、BLEU/ROUGE‑L 评估、GPU 实验与内存/时间测量。

**📊 数据集**

实验数据集为 GPT‑2‑small 在两套表到文本任务：E2E（42k 训练/4.7k 验证/4.7k 测试）和 DART（62k/7k/12k）。

**📈 对比分析**

与 DP‑Muon、DP‑Muon‑BC、KF‑DP‑Muon、KF‑DP‑Muon‑IM 等现有 Muon 私有优化器在相同 ε∈{1,2,4,8} 下比较；在所有预算和 3 个种子上 DP‑Muon‑IM 均比 DP‑Muon 提升 BLEU 0.6–2.6 分、ROUGE‑L 0.5–1.1 分；在 E2E ϵ=8 时 BLEU 65.65、ROUGE‑L 67.80，显著优于基线。

**⚠️ 局限性**

局限性包括：需要额外梯度评估和模型状态存储，导致训练成本和 GPU 内存显著增加；对历史长度 K 与衰减 γ 的选择敏感，需经验调优；理论假设依赖梯度在不同模型状态间的相关性；实验仅验证 GPT‑2‑small，缺乏更大模型或其他任务的泛化验证。

---

## 262. MuonIO: Principled Norm-Aware Descent for Embedding Tables and Language Model Heads

**arXiv ID:** 2610.02705 | [PDF](https://arxiv.org/pdf/2610.02705v1)

**作者:** Linkai Ma `[一作]` (Purdue University), Brian Bullins `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对语言模型输入嵌入表和输出头部的 Muon‑style 统一更新规则（MuONIO），并在 LLaMA 系列模型上实现该更新。

**💡 创新点**

创新点在于：① 统一了对两种词表相关层的几何处理，利用 1→2 以及 2→∞ 操作范数对应的列/行归一化；② 通过与 Muon 的谱范数更新对齐，使得这两层不再使用 AdamW，显著降低了优化器状态与 FLOPs；③ 通过总变差稳定性分析证明 2→∞ 范数对 softmax 输出的影响。

**🔧 技术方法**

采用了 Muon 的局部线性化与谱范数正则化、1→2 与 2→∞ 操作范数、行/列归一化更新、AdamW 对比、以及对 FLOPs 与内存的精确计数。

**📊 数据集**

在 C4（Common Crawl）英文语料上训练了 60M、130M 和 1B 规模的 LLaMA‑family 模型。

**📈 对比分析**

与基线 Muon、Polar Express、Ember 等优化器比较，MuONIO 在 60M/130M/1B 规模上分别提升了 0.2–0.6 点的验证困惑度，并将 I/O 层的优化器状态内存减少约 50%、FLOPs 降低约 46%。

**⚠️ 局限性**

局限性在于：① 只考虑了总变差（TV）而未处理 softmax 对全局偏移的平移不变性；② 可能无法捕捉 KL 散度等更细粒度的分布变化；③ 目前仅在 LLaMA 系列模型验证，其他模型/任务的泛化仍待探索。

---

## 263. Self-Supervised Scaling of Terminal Environments for Scientific Domains

**arXiv ID:** 2610.02710 | [PDF](https://arxiv.org/pdf/2610.02710v1)

**作者:** Zhongzhi Li `[一作]` (Tencent HY LLM Frontier), Leowei Liang `[通讯]` (Tencent HY LLM Frontier)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了“软件循环重构（Software-in-the-loop Reconstruction, SWR）”框架，用已有的科学工程工作流生成任务、公共示例和隐藏验证输出，从而自动化构建可验证的终端代理任务；

**💡 创新点**

核心创新在于将现成工作流同时作为参考实现和验证目标，利用统一的可执行接口和分层语义验证器，实现任务的可扩展化与自动验证；

**🔧 技术方法**

使用了自监督学习、编程-by-示例（programming by example）、行为验证（hierarchical semantic verifier）以及多轮交互记录做监督微调（SFT）；

**📊 数据集**

构建了包含500个工作流、46个软件家族、15,600个公共场景和8,400个隐藏场景的SWR数据集，涵盖物理科学、分子科学、地球与空间科学等六大领域；

**📈 对比分析**

对比了八种API服务模型和四个匹配token的训练语料库，SWR SFT在TB2、TB4、LHTB和SWR100等终端代理基准上均获得最高平均表现，Pass@1/Pass@3分别提升至22.8%/27.9%，验证集精度在2.0%以内；

**⚠️ 局限性**

局限性包括：1) 仍需手工制定域特定的比较规则；2) 只验证已知工作流的任务，未评估对未知工作流的泛化；3) 仅提升终端代理转移性能，未证明在自主科学发现方面的效果；

---

## 264. SymRegFlow: Symmetry-Regularized Flow Matching for Video World Models

**arXiv ID:** 2610.02726 | [PDF](https://arxiv.org/pdf/2610.02726v1)

**作者:** Xi Ye `[一作]` (Tsinghua University), Jun Zhu `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

对现有多视角流匹配生成模型进行微调，使其能够在任意连续摄像机姿态下生成多视角一致的视频，无需新视角RGB监督。

**💡 创新点**

通过引入对称正则化、双锚点遮罩监督和跨锚点去噪一致性，显著降低单源视角偏差，并实现持续姿态控制。

**🔧 技术方法**

结合流匹配、扩散去噪、3D几何重投影、可训练的姿态嵌入和有向跨视角注意力（DCVA）等技术。

**📊 数据集**

在Cosmos-Drive-Dreams与nuScenes两个自动驾驶多视角数据集上进行实验。

**📈 对比分析**

与DiST-4D、OmniRe、GEN3C、FreeVS等基线对比，SymRegFlow在nuScenes的FVD/FVMD、FID、实例保持等指标上均取得最高或最优成绩。

**⚠️ 局限性**

仍依赖高质量深度估计和几何重投影，可能在极端视角差距或复杂遮挡下产生误差，且实现过程计算量相对较大。

---

## 265. Silent Dissent: LLM Agents That Yield to the Majority Still Represent Their Original Premise

**arXiv ID:** 2610.02702 | [PDF](https://arxiv.org/pdf/2610.02702v1)

**作者:** Ziang Ni `[一作]` (Delft University of Technology), Peng Zou `[通讯]` (Sun Yat-sen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在多代理辩论中“沉默异议”现象，即当代理人放弃正确答案并接受多数错误答案时，内部仍保留正确前提。

**💡 创新点**

创新点在于：①设计了只读未被任何人声明的桥接实体（bridge）的实验范式；②使用 Jacobian 视角（Jacobian lens）对内部表示进行解码，发现它能捕捉到沉默异议；③对比 logit lens 并在四个不同模型上预注册并检验假设。

**🔧 技术方法**

技术手段包括：多代理文本对话协议（三名脚本化同行）、两跳事实推理（bridge → answer）、基于 Neuronpedia 的 Jacobian lens 与 logit lens 的激活读取、注入干预（激活方向注入）以及 Bootstrap 区间统计。

**📊 数据集**

数据集主要是公开的 TwoHopFact（约45k条两跳事实）以及手工添加的一些首都城市事实；对每个模型仅测试其已知的事实（占 3–9%）。

**📈 对比分析**

比较方法：预注册 H1–H4 的 95% 置信区间；检验不同模型、不同层级、不同压力条件下的桥读取显著性。结果显示：Qwen 系列模型和 Gemma 在大部分测试中均出现沉默异议，Jacobian lens 的读数显著高于 logit lens；在 Qwen 模型中注入桥的方向可部分恢复原答案，Gemma 与 Llama 无此效果。

**⚠️ 局限性**

局限性包括：①仅覆盖两跳事实且桥为国家/城市/大学，难以推广到更广泛语义；②仅能检验模型已知事实的 3–9%；③不同模型采用不同提示格式、层级规则，导致可比性受限；④对 logit lens 的依赖度不确定；⑤注入干预的效果不稳定，未在所有模型中验证。

---

## 266. RoboChemGym: A Protocol-Driven Generative Simulation Framework for Long-Horizon Chemical Manipulation

**arXiv ID:** 2610.02708 | [PDF](https://arxiv.org/pdf/2610.02708v1)

**作者:** Chenxi Li `[一作]` (Shanghai Artificial Intelligence Laboratory), Dongzhan Zhou `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `51c0528b-f690-4182-ae60-bb5f046c276c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了RoboChemGym框架，利用LLM自动解析化学实验协议，生成符合实验流程的高保真长周期细粒度操控演示，并支持可扩展的数据合成与层次化基准评估。

**💡 创新点**

结合协议驱动的任务合成、双循环动作与场景自适应优化、记忆驱动的自改进机制以及实验室专属的数据扩增，首次实现长周期化学实验的全自动生成与细粒度性能评估。

**🔧 技术方法**

使用大语言模型(Claude 3 Opus)、高保真物理仿真(Isaac Sim)、动作与场景双循环优化、记忆库自改进、参数化原子动作库及多维度随机化扩增技术。

**📊 数据集**

基于真实化学实验协议（20条）生成的虚拟场景与轨迹，采集约200条专家轨迹，并通过实验室纹理与光照库实现多样化。

**📈 对比分析**

与ACT、Diffusion Policy、π₀等策略在清洁与随机环境下对比，π₀在长序列上保持较高成功率（Level‑1 22.5/15.4%，Level‑2 5.2/3.8%，Level‑3 接近0%），RoboChemGym生成的数据在 sim‑to‑real 中将成功率提升约23%，表现出显著鲁棒性。

**⚠️ 局限性**

仅在仿真环境验证，未实现完全自主执行；仅支持Franka Panda机器人，缺乏对多机器人平台的泛化能力。

---

## 267. Learning to Revise Reasoning with Segment-wise On-Policy Distillation

**arXiv ID:** 2610.02703 | [PDF](https://arxiv.org/pdf/2610.02703v1)

**作者:** Yuxiang Zhang `[一作]` (Hong Kong University of Science and Technology), Tianxiang Zhao `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出Segment-wise On-Policy Distillation（Seg-OPD），通过教师重写学生的中间推理段落来训练学生进行推理修订。

**💡 创新点**

创新点在于：①使用不确定性指标选择关键推理节点并生成对应的教师重写；②通过段落级偏好学习目标，让学生倾向于采用教师重写而非原始段落，同时保持密集的 token‑wise OPD 监督。

**🔧 技术方法**

技术包括：基于熵的推理不确定性度量、教师重写生成、段落级偏好损失（Bradley‑Terry 对数似然）以及结合 token‑wise KL 损失的联合优化。

**📊 数据集**

使用数学推理数据集 DeepMath-103K 训练，评估在 AIME24/25/26、MATH500、MINERVA 等数学题目以及 Codeforces、TACO、HumanEval+ 等竞赛编程任务。

**📈 对比分析**

与 OPD、E‑OPD、G‑OPD、OmniOPD 等先进对策相比，Seg‑OPD 在所有模型和数据集上平均提升约5.22%的推理准确率，且显著提高修订成功率、降低循环率。

**⚠️ 局限性**

局限性包括：教师重写生成成本高、对推理不确定性阈值和段落长度的敏感度、以及在更大规模或更复杂任务上的通用性尚未完全验证。

---

## 268. DeltaWorld: Physically Consistent Interactive World Simulators via Action-Conditioned Latent Increment Learning

**arXiv ID:** 2610.02691 | [PDF](https://arxiv.org/pdf/2610.02691v1)

**作者:** Boyuan Hou `[一作]` (Imprintx Robotics), Shaowei Cui `[通讯]` (Imprintx Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于动作诱导的潜在增量模型Delta-LTM和交互感知潜在对齐的物理一致交互世界模拟器DeltaWorld

**💡 创新点**

创新点在于将未来潜在状态的预测转化为动作引起的潜在增量预测，并通过构造反事实交互区域实现对交互敏感位置的加权监督，提升长时序物理一致性

**🔧 技术方法**

采用自编码器学习2D潜在表示，3D卷积与FiLM、空间/时间注意力相结合的动作条件动态模型，交互对齐通过潜在差异掩码实现

**📊 数据集**

在Interactive World Simulator (IWS) 基准（5个任务、2视角）以及自采集的跨机器人数据集（3种机器人、4个任务）上进行训练和评估

**📈 对比分析**

与LeWorld、Cosmos3、IWS等基线对比，DeltaWorld在FID、FVD、LPIPS、PSNR、SSIM、MSE等指标上均优于基线，FVD下降约46.6%，LPIPS下降约31.1%

**⚠️ 局限性**

局限性主要包括：仍需在更复杂交互场景下验证鲁棒性，潜在空间对极端视觉变化的适应性有限，未深入探讨不同动作尺度对增量预测的影响

---

## 269. Prospective Hindsight: Self-Calibrating Reinforcement Learning via Prediction-Reality Gaps

**arXiv ID:** 2610.02740 | [PDF](https://arxiv.org/pdf/2610.02740v1)

**作者:** Jiaxin Zhang `[一作]` (Salesforce AI Research), Chien-Sheng Wu `[通讯]` (Salesforce AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种自校准训练原则——前瞻性回顾（Prospective Hindsight，PH），通过将行动时的预测与回溯评估之间的差距作为权重信号，增强对不准确预测样本的梯度更新。

**💡 创新点**

创新点在于：①将行动时预测误差视为可利用的训练信号，而非仅仅诊断；②通过“惊讶”权重自适应地放大误差样本的梯度；③不引入额外的校准损失，直接在任意回溯式基准方法（GRPO、OPD及其组合）上实现自校准。

**🔧 技术方法**

技术上使用了共享参数的自评估模块（prompted LLM预测成功/失败/不确定），四格校准分类（CS、OF、US、AF），以及基于惊讶指示符的梯度加权公式 •ℓ^{PH}= (1+α·ξ)·ℓ^{base}，并在单回合与多回合任务中与GRPO、OPD结合。

**📊 数据集**

实验数据集包括：单回合的 Science Q&A 与 Tool Use（均使用 SDPO 评估器），以及多回合的 OpenClaw‑RL “个人代理”任务（使用 GPT‑4.1 用户模拟器与 PRM 判断器）。

**📈 对比分析**

与基线（α=0）及随机/仅失误加权对照相比，PH 在所有方法和模型规模下均提升了任务成功率，同时显著降低了过度自信失误率（OFR）和总惊讶率。单回合实验中，OFR 从约34%降至20%，多回合实验中，OFR 进一步降至约3%并且预测准确率提升至约88%。

**⚠️ 局限性**

局限性包括：①对高斯连续型任务的适用性未验证；②过大 α 可能导致自评估偏向极端不确定（US 或 OF 失衡），影响预测质量；③仍依赖于回溯评估器的质量，评估误差会影响 PH 的加权效果。

---

## 270. Bellman Error Minimization Via Linear Programming Normalization

**arXiv ID:** 2610.02730 | [PDF](https://arxiv.org/pdf/2610.02730v1)

**作者:** Haining Yu `[一作]` `[通讯]` (Independent Consultant), Haining Yu (Independent Consultant)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种结合深度神经网络和线性规划归一化的函数逼近方法，用于在高维动态规划中最小化贝尔曼误差，并以网络容量控制问题为案例进行验证。

**💡 创新点**

创新点在于将深度网络与定制化LP归一化相结合，既利用LP捕捉一阶容量需求关系，保证单调性，又让网络学习高阶非线性；同时采用分层采样估计贝尔曼误差并用梯度优化。

**🔧 技术方法**

采用深度前馈神经网络（5层、64隐藏单元、sigmoid激活）、线性规划求解器、Adam梯度下降、Monte Carlo仿真与分层采样等技术实现贝尔曼误差最小化。

**📊 数据集**

使用合成的网络容量控制数据集：四组仿真案例（不同资源数、容量、需求与收益），每组通过模拟生成640条请求序列。

**📈 对比分析**

通过与先来先服务（FCFS）、简单LP策略和基于分解的高级近似算法（DECOMP）的对比，DNN策略在所有案例中均优于FCFS和LP，且比DECOMP平均提升约0.3%~0.4%。

**⚠️ 局限性**

局限性包括：仅在合成数据上验证；对LP求解的依赖导致对大规模状态空间的扩展受限；缺乏理论收敛与性能保证；采样方法需覆盖足够多状态以保证逼近质量。

---

## 271. Same Performance, Different Process: Epistemic Ownership in AI-Mediated Education

**arXiv ID:** 2610.02731 | [PDF](https://arxiv.org/pdf/2610.02731v1)

**作者:** Jorge Fábrega `[一作]` `[通讯]` (CICS-UDD), Jorge Fábrega (CICS-UDD)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对150段学生与大型语言模型对话记录的编码，探讨生成式AI在教育中的认知参与模式，特别是“方向、整合、评估”三维的可观测行为，并与作业成绩进行比较；

**💡 创新点**

创新点在于引入并操作化“认知所有权”框架，将其拆解为可在对话中观测到的三维指标，证明相同的成绩可能伴随截然不同的认知过程，并强调评估需要结合过程证据而非仅靠最终产出；

**🔧 技术方法**

采用了基于OpenAI GPT-5.6的响应API对学生发言进行自动编码，并结合手工校正，利用编码规则识别“方向、整合、评估”三维行为；

**📊 数据集**

使用的Dataset为StudyChat，包含203名学生在2024-2025学年完成7项人工智能与计算机科学作业的2214个对话，约16,851个学生发言；

**📈 对比分析**

比较方法通过设定不同最小学生发言数阈值，统计每段对话中是否出现三维行为，并将其与对应标准化作业分数对比，结果显示无显著的成绩差异；在同分数对比案例中展示同分不同过程，表明成绩与过程无系统性关联；

**⚠️ 局限性**

局限性包括：只观测到对话内行为，无法捕捉对话外的认知活动；在低反馈或非编程任务中整合、评估等维度难以观测；缺乏对“答复可辩性”的系统评估；样本量有限，难以推广至更广泛情境。

---

## 272. Label-Efficient Time Series Classification at Scale: A Dual-Stream OSSE-LSTM with Counterfactual Attribution

**arXiv ID:** 2610.02704 | [PDF](https://arxiv.org/pdf/2610.02704v1)

**作者:** Nguyen Ho `[一作]` (Loyola University Maryland), Long Van Ho `[通讯]` (International University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出双流 Omni-Scale SE 与 BiLSTM 的原型网络，实现少量标记样本下的高效时间序列分类，并提供可解释性。

**💡 创新点**

创新点在于：1）双流设计将多尺度局部特征与全局时序上下文联合提取；2）提出 Counterfactual Integrated Gradients 用于解释原型间距离并指导测试时原型细化；3）在固定标签空间的 K-shot 设置下保持极高的稳定性。

**🔧 技术方法**

使用技术包括：基于原型网络的度量学习、Omni-Scale 卷积与 Squeeze‑and‑Excitation 调节、双向 LSTM 时序建模、L2 归一化融合、Counterfactual IG 解释以及软掩码原型细化。

**📊 数据集**

在 19 个 UCR 单变量时间序列数据集上进行实验。

**📈 对比分析**

与 InceptionTime、LSTM‑FCN、MiniRocket、TapNet、TS2Vec、DPSN 等六个基线在 2‑8 shot 下进行对比，OSSE‑LSTM 平均准确率超过 96%，在每个 K 的比赛中均为最高，且随 K 变化误差仅 0.36 个百分点，表现出卓越的稳定性。

**⚠️ 局限性**

局限性包括：对超参数的依赖、在大支持集或多通道时间序列上的泛化尚未充分验证，且 C‑IG 细化在部分数据集提升有限。

---

## 273. Ego2World: Compiling Egocentric Cooking Videos into Executable Worlds for Belief-State Planning

**arXiv ID:** 2610.02715 | [PDF](https://arxiv.org/pdf/2610.02715v1)

**作者:** Qinchuan Cheng `[一作]` (Xi'an Jiaotong University), Shijie Li `[通讯]` (Institute for Infocomm Research, A*STAR)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了可执行的 egocentric 烹饪视频基准，将 HD‑EPIC 注释编译为符号世界、动作规则与任务条件，支持部分观测、持续执行和代理信念与世界状态分离；

**💡 创新点**

创新点在于：① 通过编译器将真实视频证据映射到可执行符号动作，保留来源证据；② 提供可持续、可追踪的执行环境；③ 设立严格完成率、条件达成度、执行诊断等多维评估指标；

**🔧 技术方法**

使用了符号世界编译、规则执行器、观测与视觉查询接口、任务条件验证、Bootstrap 置信区间以及多大语言模型（Qwen‑Plus、GPT‑5.5 等）作为规划器；

**📊 数据集**

采用 HD‑EPIC 以及其关联的 EPIC‑KITCHENS 等 egocentric 视频数据集；

**📈 对比分析**

通过对 105 任务、18 个完整剧集的 6 个规划器进行严格完成率、GCR、动作有效率等指标比较；Qwen‑Plus 的持久信念实验显示行动有效率提升 4.15pp，视觉查询次数减少 90.27%，但未见完成率提升；

**⚠️ 局限性**

局限性包括：规则覆盖仅限厨房烹饪符号动作，扩展需要新本体和规则；未独立隔离规划时长和记忆影响；内存实验仅针对 Qwen‑Plus；不涉及物理执行安全性和文化多样性。

---

## 274. Localized Conformal Safety Monitoring with Vision-Language Models for Autonomous Driving

**arXiv ID:** 2610.02765 | [PDF](https://arxiv.org/pdf/2610.02765v1)

**作者:** Luís Marques `[一作]` (University of Michigan), Dmitry Berenson `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种对冻结视觉语言模型(VLM)的后置校准层，用于安全监测车辆轨迹，并提供概率保证的安全预测集合。

**💡 创新点**

将局部化、标签条件的合成校准方法与VLM的线性探针结合，形成SLCP+label CP，显著提升未见场景下碰撞轨迹的检测率。

**🔧 技术方法**

采用VLM隐藏层提取、线性逻辑探针、UMAP降维、本地化核加权分位数、标签条件分位数校准以及Split CP的修正。

**📊 数据集**

在CARLA仿真环境中采集的约15k轨迹（SimLingo规划），覆盖Fail2Drive的三类场景（HardBrake、PedestriansOnRoad、Wall），共计40k帧。

**📈 对比分析**

与未校准的VLM及若干消融基线对比，校准后在Qwen3‑VL和Cosmos‑Reason2上实现约89%和88%的unsafe标签覆盖率，且保持在Beta‑Binomial 90%置信区间内，误报率上升至约50%。

**⚠️ 局限性**

结果受UMAP投影和核参数敏感；训练集帧相关导致交换性假设失效；场景转移与数据相关性影响校准泛化。

---

## 275. Revisiting Visual Representation Enhancement of VLMs via Kernel Canonical Correlation Analysis

**arXiv ID:** 2610.02718 | [PDF](https://arxiv.org/pdf/2610.02718v1)

**作者:** Peilin Yang `[一作]` (Beijing Institute of Technology), Qinghua Tao `[通讯]` (Beijing Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过微调CLIP视觉编码器，并在训练中使用Kernel Canonical Correlation Analysis (KCCA) 与DINOv2以及CLIP文本编码器的投影相关性进行三视图对齐，提升细粒度视觉表现。

**💡 创新点**

创新点在于用KCCA最大化特征子空间投影相关性替代传统的核矩阵逐元素匹配，并提出融合文本语义的三视图联合对齐框架。

**🔧 技术方法**

主要技术包括KCCA、Lagrangian KKT条件求解、端到端训练、CLIP与DINOv2模型以及投影矩阵优化。

**📊 数据集**

使用ImageNet‑1K进行微调，MMVP‑VLM评测细粒度视觉能力，Flickr30K与MSCOCO评估零样本图文检索。

**📈 对比分析**

与DIVA、KUEA等方法对比，MMVP‑VLM精度从17.8%提升至25.9%，零样本检索平均精度略升至73.3%，显示显著性能提升。

**⚠️ 局限性**

仅在CLIP‑ViT‑L/14 与 DINOv2‑ViT‑L/14 组合上验证，未扩展到其他VLM或教师模型，对更广泛的感知基准测试仍有限。

---

## 276. Conditional Capacity and Routing in Mixture-of-Experts Particle Transformers

**arXiv ID:** 2610.02701 | [PDF](https://arxiv.org/pdf/2610.02701v1)

**作者:** Kaushik Pendiyala `[一作]` (University of California Davis), Javier Duarte `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在JetClass-II 188类 jet 分类任务中，对比稀疏混合专家 Particle Transformer（MoEParT）与密集 Particle Transformer（ParT），系统评估专家数量、容量、top‑K 以及辅助负载平衡损失对准确率、QCD 拒绝率和前向 FLOPs 的影响，并通过 NMI 路由分析揭示专家分配与粒子属性的关联。

**💡 创新点**

提出了将存储参数容量、主动计算、路由容量与路由组织分离评估的框架；证明单一专家容量提升在保持计算量不变时可提升约1%准确率，而多专家激活虽然提高准确率但计算量显著增加；进一步发现额外专家带来的准确率提升趋于饱和，且更强的路由结构并非性能提升的单调指标。

**🔧 技术方法**

使用 Particle Transformer 结构、稀疏混合专家（MoE）路由、top‑K 专家激活、负载平衡辅助损失、NMI 路由关联度分析、AUROC 与 QCD 拒绝率评估以及标准化前向 FLOPs 计量。

**📊 数据集**

JetClass-II 188 类 Pythia 模拟数据集，包含两峰、三峰、四峰信号与 QCD 背景，共计 456,822 条测试样本，每条 jet 最多 128 个粒子。

**📈 对比分析**

在相同训练预算、验证集与优化设置下，对齐密集与 MoE 配置的参数量和 FLOPs，使用验证准确率选择检查点，并在测试集上报告准确率、R_50、R_80。结果显示无 drop 的 top‑1 MoE 在保持 FLOPs 接近密集模型的前提下准确率提升约 1%，而 top‑2 或更多专家激活可进一步提升准确率但需要 50–70% 的计算量增加，额外专家容量提升逐渐饱和。

**⚠️ 局限性**

仅使用单一训练检查点、未评估随机种子波动、训练样本量有限、仅针对单一 ParT 结构与 JetClass-II 任务，缺乏对不同架构、数据规模与多种随机初始化下的泛化性与稳定性的系统验证。

---

## 277. WakeKV: Reactive, Reversible KV Residency for Heads That Change Their Minds

**arXiv ID:** 2610.02713 | [PDF](https://arxiv.org/pdf/2610.02713v1)

**作者:** Utkarsh Ranjan `[一作]` `[通讯]` (University of California San Diego), Utkarsh Ranjan (University of California San Diego)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种可逆动态KV缓存管理策略 WakeKV，针对注意力头状态非稳定的问题，在推理过程中实时响应需求并恢复被降级的头状态。

**💡 创新点**

创新点在于：1) 通过GPU端LRU+CPU缓存的组合实现头状态可恢复而非永久丢弃；2) 在匹配内存或预算下持续降低miss率；3) 与传统的固定分类和破坏性驱逐相比，提供更稳健的性能提升。

**🔧 技术方法**

使用技术包括：GPU驻留LRU缓存与CPU可恢复存储、实时需求检测与动态预算控制、PCIe传输恢复、FlexiCache/vLLM shim 实现，以及针对不同模型和推理任务的模拟与真实硬件评估。

**📊 数据集**

数据集与模型：Qwen2.5-3B、DeepSeek-R1-Distill-Qwen-1.5B、DeepSeek-R1-Distill-Llama-8B 在三种推理任务（NIAH、长链式推理CoT、多轮回忆）以及 Mistral-7B/NIAH；使用 LongBench 评估质量。

**📈 对比分析**

对比方法：在匹配内存或预算的条件下与冻结分类、破坏性驱逐、SnapKV、R-KV、ReasonAlloc 等基线进行比较；WakeKV 在所有 17/20 配置下 miss 率 ≤ 对手；在 A30 硬件上吞吐提升 1.34–2.33 倍，质量保持 ≥95%。

**⚠️ 局限性**

限制：仅评估 1.5B–8B 模型，模拟计数 miss 而非真实 stall 时间；真实硬件实验仅覆盖单模型单任务；对更大规模模型、不同硬件平台和多 GPU 场景缺乏验证；未对不同压缩/存储策略的细粒度影响做深入探讨。

---

## 278. Learning from Evolving Errors: Adaptive Iterative Repair for On-Policy Distillation

**arXiv ID:** 2610.02700 | [PDF](https://arxiv.org/pdf/2610.02700v1)

**作者:** Rui Li `[一作]` (University of Science and Technology of China), Qi Liu `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个自适应迭代修复框架，通过动态生成修复引导并在错误对齐区域进行教师监督，改进了参考条件的 on‑policy 自蒸馏，避免了 shortcut 风险。

**💡 创新点**

创新点在于将教师监督从静态参考切换为动态修复引导，结合错误对齐的监督区域和结果感知的阶段加权，从而更精准地纠正学生错误。

**🔧 技术方法**

使用 on‑policy self‑distillation、修复引导生成器、错误对齐的 distillation 损失以及结果感知的阶段加权技术，基于 Qwen3‑4B/8B 大模型实现。

**📊 数据集**

训练采用 DAPO‑Math‑17K 数据集（生成 GPT‑5.5 参考解），评估使用 AIME24、AIME25、HMMT25 数学推理基准，外部 OOD 测试为 MMLU‑Pro 与 GPQA。

**📈 对比分析**

与 SFT、GRPO、OPSD 等基线对比，Qwen3‑4B/8B 在数学推理平均得分提升 2.8–3.6 分，同时在 OOD 基准保持接近基线性能。

**⚠️ 局限性**

局限包括需要规则过滤生成的修复引导、对极端错误仍可能产生噪声、以及仅在数学推理领域验证，未充分测试其他任务。

---

## 279. LearnAdapt Praxis: Controlled AI Assistance and Evidence Traces for Adult Workplace Learning

**arXiv ID:** 2610.02699 | [PDF](https://arxiv.org/pdf/2610.02699v1)

**作者:** Nizam Kadir `[一作]` `[通讯]` (Singapore University of Technology and Design), Nizam Kadir (Singapore University of Technology and Design)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

实现了 LearnAdapt Praxis 的结构化成人工作场景学习工作流，支持 AI 辅助、版本控制、转移阶段与评估记录；

**💡 创新点**

在同一项目级别实现了 AI 辅助与独立工作区分、转移期间权限限制以及可审计的记录链；

**🔧 技术方法**

采用 PHP/Apache 结合 Cloud Run 的后端服务，SQLite 内存数据库做合成演练，真实环境使用 OpenAI API；

**📊 数据集**

使用 120 条合成剧本（3 个工作情境 × 4 种 AI 提供条件 × 10 次重复）作为测试数据；

**📈 对比分析**

通过 3,990 条断言检查，全部通过，演练在内存环境下平均 0.9 ms；真实服务在 staging/production 上通过多项 smoke 测试并完成一次 OpenAI 调用；

**⚠️ 局限性**

仅验证技术实现与流程完整性，未涉及真实学习成效、并发性能、可访问性或完整的 PROV 追踪；

---

## 280. ReLEAF: A Socio-Technical Framework Bridging Custodians and Researchers for Trustworthy Data Sharing

**arXiv ID:** 2610.02720 | [PDF](https://arxiv.org/pdf/2610.02720v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 281. Who Went Where When on the Lunar Surface: Forensic Trajectory Analysis to Identify Byzantine Rovers

**arXiv ID:** 2610.02694 | [PDF](https://arxiv.org/pdf/2610.02694v1)

**作者:** Lachlan Holden `[一作]` (AI for Space Group), Tat-Jun Chin `[通讯]` (Adelaide University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种针对多无人车、存在拜占庭（欺骗）测量时的法医轨迹分析方法，能够后验重构各车轨迹并识别不可信车辆。

**💡 创新点**

创新点在于：①把可信度建模为整个车辆子集而非单个测量；②利用内部相互检测的一致性和与外部车辆检测的不一致性，统计评估可信子集；③将可信子集的测量投射到姿态图优化中，结合平滑约束实现高精度轨迹恢复。

**🔧 技术方法**

采用统计一致性评估（χ² 概率）、姿态图优化（Levenberg-Marquardt）、平滑约束、集合搜索以选取最优可信子集。

**📊 数据集**

实验数据包括：①合成 2D 仿真（5 车组、单/双拜占庭、不同视野、丢包情况）；②真实行星模拟数据（Lajoie et al. 2025 的 C‑SLAM 数据集，带 RTK GPS 轨迹）。

**📈 对比分析**

与鲁棒 PGO 方法 GNC、PCM 进行对比；在单/双拜占庭、不同测量维度、视野、丢包率下，本文方法召回率 ≥ 83.3%（单拜占庭）/64.6%（双拜占庭），误差显著低于对手；对手召回率 <10% 或 ~5%；精度更高。

**⚠️ 局限性**

局限性：假设可信车辆至少占一半；依赖前端测量误差已知且可建模；对多分离拜占庭群体、不同攻击策略的鲁棒性尚待验证；在极端传感器噪声或稀疏检测下性能可能下降。

---

## 282. On the Chain-of-Thought Monitorability of Looped Language Models

**arXiv ID:** 2610.02741 | [PDF](https://arxiv.org/pdf/2610.02741v1)

**作者:** Han Wang `[一作]` (University of Illinois Urbana Champaign), Huan Zhang `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究 LoopLM（循环语言模型）对 Chain-of-Thought（CoT）监控的可监测性，系统评估不同循环深度以及与非循环模型的对比表现。

**💡 创新点**

首次系统评估 LoopLM 在 CoT 可监测性上的影响，发现循环深度增加会在某些任务下降低可监测性，但循环架构本身并不必然导致可监测性下降。

**🔧 技术方法**

采用循环 Transformer 结构的 LoopLM，使用 MonitorBench 的 CoT 监控得分评估方法，并以 Qwen3.8‑27B 为判定者；实验通过 vLLM 进行推理。

**📊 数据集**

利用 MonitorBench 的八个任务（Logic、Health、Safety、Engineering、Science、Knowledge、Preference、Law Judgment）以及两种压力测试（Direct Concealment、Monitor‑aware Evasion）。

**📈 对比分析**

对比方法：在同一模型家族内部调节循环深度；跨模型比较时匹配参数量、层数或有效层数。结果显示，循环深度在 Logic/Science/Engineering 等任务下可监测性下降，但 LoopLM 与匹配的非循环模型的可监测性相当或更好。

**⚠️ 局限性**

局限：开放源码 LoopLM 的多样性和规模有限；跨模型比较受训练数据、后处理等混杂因素影响；实验仅覆盖部分 MonitorBench 任务，未包括更难的 agent/coding 任务。

---

## 283. EpiWorld: Grounding LLM Policy Agents in Epidemiological World Models

**arXiv ID:** 2610.02744 | [PDF](https://arxiv.org/pdf/2610.02744v1)

**作者:** Zeeshan Memon `[一作]` (Emory University), Liang Zhao `[通讯]` (Emory University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了EpiWorld闭环框架，将大型语言模型与基于区域交互的世界模型相结合，用于生成符合公共卫生协议的干预方案并通过反事实模拟进行评估；

**💡 创新点**

创新点在于：1）将LLM作为政策推理者，利用协议约束、监测技能与适应性经验三层技能库实现可解释、可审计的决策；2）构建具有延迟编码、图结构耦合与政策条件下潜在调制的流行病世界模型，实现高质量的反事实滚动；3）通过后动作适应循环持续改进经验库，无需重新训练模型；

**🔧 技术方法**

使用技术包括：大型语言模型（如GPT‑4o、Qwen2.5‑7B等）作为政策生成器；时序CDE和GRU混合编码、FiLM式政策调制的潜在状态网络；图注意力机制建模区域耦合；多目标监督（潜在和观测误差）训练世界模型；多模型集成评估不确定性；

**📊 数据集**

实验数据集涵盖三类：真实美国COVID‑19州级每周监测（病例、住院、死亡）；COVID‑19 SEIR仿真数据；流感SEIR仿真数据；

**📈 对比分析**

与三大类基线对比：1）政策无关与政策条件下的预测器（Compartmental‑GP、MechBayes、epidemia、PAN‑CODE、EARTH）；2）基于RL、遗传规划和DQN的政策优化器（EpiPolicy‑RL、ADIOS、EpidRLearn）；在所有数据集上，EpiWorld在外部样本预测误差（Peak‑MAE）上优于所有基线，且在8周累计住院量减少上比最强RL基线提升约58‑59%；

**⚠️ 局限性**

局限性包括：依赖人为监督与协议约束，无法完全自动化；经验库仅对同一感染阶段有效，跨阶段迁移受限；世界模型集成虽提供可靠性信号但过于收敛；评估仅在历史/仿真环境，缺乏现场部署验证；

---

## 284. Differential Privacy of Gradient Descent on Perturbed Objectives

**arXiv ID:** 2610.02716 | [PDF](https://arxiv.org/pdf/2610.02716v1)

**作者:** Austin Watkins `[一作]` (Johns Hopkins University), Raman Arora `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在目标扰动（objective perturbation）框架下，对一次性加入高斯噪声后进行的全批梯度下降（full‑batch GD）第 N 步迭代进行差分隐私与泛化误差的理论分析，证明该迭代映射是可逆的并给出显式的隐私计量与误差上界。

**💡 创新点**

创新点主要在于：① 在仅一次扰动的前提下，通过可逆性（C^1‑diffeomorphism）直接对有限迭代进行隐私计算，避免了传统的二次扰动或逐步噪声；② 通过冻结 Hessian 的递推闭式与真实 Jacobian 的比较，得到对雅可比矩阵最小奇异值的显式下界；③ 对广义线性模型（GLM）利用梯度差和 Hessian 仅在特征子空间中的秩‑1 结构，消除维度项，得到无维度依赖的隐私预算；④ 引入几何递减的优化误差项，使得在 N 越大时隐私和误差都逼近传统目标扰动极限。

**🔧 技术方法**

技术上主要采用：顺序矩阵乘积展开、冻结 Hessian 递推与闭式解、雅可比矩阵对称部分负定性分析、隐私损失的改变变量法、Gaussian 分布的尾部积分、矩阵行列式比较（矩阵行列式引理）以及统一收敛（Uniform Convergence）和强凸性、光滑性与 Hessian Lipschitz 性的结合。

**📊 数据集**

实验部分以合成逻辑回归（logistic regression）为例，设置特征归一化 ζ=1，样本量 n=10^6，维度 d=10^4，正则化 μ=10^{-2}，噪声尺度 σ=10^{-5}，步长 η=1/L，截断参数 u=6，展示了 N=1500 步即可实现 (1,10^{-7})‑DP，并给出相应的误差上界；论文的主要结果为理论性质，未对真实数据集进行评估。

**📈 对比分析**

与已有的 Objective Perturbation、AMP（Approximate Minima Perturbation）以及基于噪声迭代的隐私放大方法相比，本文的隐私上界不显式依赖维度 d，并且在 N 越大时收敛速度更快（几何递减），实现的隐私‑误差折中更优：在给定 (1,10^{-7})‑DP 的条件下，仅需 1500 次全批梯度计算；而 DP‑SGD 需 O(n^2) 次单样本梯度评估，SVRG+噪声方案亦需多次迭代或额外噪声。

**⚠️ 局限性**

局限性包括：① 仅适用于强凸、光滑且 Hessian Lipschitz 的 GLM；② 需要全批梯度下降，计算成本相对较高；③ 需要设定截断半径与迭代次数，参数选择可能较保守；④ 对非凸或受约束（有界域）问题的适用性有限，且对实际大规模数据集的实际实现与加速（如随机梯度或分布式）仍需进一步研究。

---

## 285. Lessons from Trauma-Informed Training on Technology-Facilitated Abuse for Gender-Based Violence Advocates

**arXiv ID:** 2610.02688 | [PDF](https://arxiv.org/pdf/2610.02688v1)

**作者:** Naman Gupta `[一作]` (University of Wisconsin--Madison), Rahul Chatterjee `[通讯]` (University of Wisconsin--Madison)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开展了面向性别暴力倡导者的创伤知情技术促进虐待培训工作坊，并通过回顾性自我民族志反思评估其影响

**💡 创新点**

提出了协作知识翻译的培训设计与评估检查表，为高风险情境下的教育干预提供可操作的框架

**🔧 技术方法**

采用基于混合方法的培训材料与创伤知情教学策略，结合调查问卷与自我民族志访谈进行评估

**📊 数据集**

收集了参与美国多家性别暴力组织倡导者的问卷数据及访谈记录，涉及数十名受训者

**📈 对比分析**

通过前后测量的自评问卷比较，显示受训者对技术促进虐待的理解与支持自信显著提升（p<0.05）

**⚠️ 局限性**

局限性包括样本规模有限、仅为自评、短期随访、缺乏客观行为变化测量，以及研究仅限于美国地区

---

## 286. Characterizing the Performance Gap in Human Activity Recognition for Older Adults

**arXiv ID:** 2610.02711 | [PDF](https://arxiv.org/pdf/2610.02711v1)

**作者:** Hossein Khayami `[一作]` (University of Maryland), Hernisa Kacorri `[通讯]` (University of Maryland)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文比较了年轻成人与老年成人在腕带加速度计人类活动识别（HAR）中的性能差距，探究不同深度学习架构与表示学习对老年人数据的影响。

**💡 创新点**

创新点在于将关注点从传统的架构排名转向表示学习质量，证明在大规模年龄多样化数据上进行自监督预训练并冻结特征能显著缩小老年人与年轻人之间的性能差距。

**🔧 技术方法**

采用多种深度学习模型（MLP、DeepConvLSTM、MobileHART、AttnTCN、Transformer、LIMU‑BERT）以及多级表示（原始信号、随机 ResNet、监督 ResNet、手工特征、SSL ResNet 训练与冻结），并使用宏 F1 作为评价指标。

**📊 数据集**

使用公开数据集包括 WISDM、RealWorld（年轻成人）、MyMove（老年人自由生活）、UK Biobank（无标签、用于自监督预训练），并在 LOSO 与跨数据集（零样本）评估框架下进行比较。

**📈 对比分析**

在 LOSO 与零样本转移评估中，年轻成人上的基准提升并未同等转移至老年人，性能差距持续甚至扩大；但随着表示质量提升（尤其是冻结的 SSL ResNet），老年人性能显著提升，宏 F1 从 0.34–0.57 进步到 0.54–0.68，差距从 0.19 缩小到 0.11。

**⚠️ 局限性**

主要局限在于无法完全将年龄与采集方式、标签方法、自由生活与脚本化行为等混杂因素分离，且老年人标签的语义与年轻人存在差异，导致差距根源未能精确归因。

---

## 287. Decoupling Memory from Context: Structured Memory for Token-Efficient Test-Time Continual Learning

**arXiv ID:** 2610.02687 | [PDF](https://arxiv.org/pdf/2610.02687v1)

**作者:** Yehya Farhat `[一作]` (Rice University), Anastasios Kyrillidis `[通讯]` (Rice University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出GraphMemory，一种基于图结构的外部记忆，用于在LLM推理时进行上下文检索和持续学习。

**💡 创新点**

创新点在于将记忆构建视为上下文优化的迭代过程，并通过有界检索实现记忆上下文长度常数化，显著降低令牌成本。

**🔧 技术方法**

采用GraphMemory框架（ACE改造）、图检索、Beta后验加权边、生成‑反思‑策划循环等技术。

**📊 数据集**

使用金融推理数据集FiNER和Formula进行实验。

**📈 对比分析**

与ACE基线对比，GraphMemory在保持相近准确率的同时，训练/测试令牌数下降81–92%，成本大幅降低，且在Qwen3.8‑27B上甚至略优。

**⚠️ 局限性**

局限包括仅在两任务、两模型上验证、检索开销导致推理延迟、实验单轮、缺乏跨模型迁移评估等。

---

## 288. FiberGeoText: A Vision-Language Model for Population- Level Organization of Superficial White Matter

**arXiv ID:** 2610.02755 | [PDF](https://arxiv.org/pdf/2610.02755v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 289. Proprioceptive Sketches as Long-Horizon Intent for Generative Action Policies

**arXiv ID:** 2610.02759 | [PDF](https://arxiv.org/pdf/2610.02759v1)

**作者:** Fangyuan Wang `[一作]` (Hong Kong Polytechnic University), David Navarro-Alarcon `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了一种联合生成机器人未来配置空间路径草图与可执行动作块的模型，用于长周期双臂操作任务。

**💡 创新点**

创新点是引入时序无关的B样条草图作为长周期意图表示，并通过块因果注意力和分层降噪在同一Transformer中引导动作生成。

**🔧 技术方法**

技术包括扩散式Transformer denoiser、B-spline参数化、块因果注意力以及分层降噪计划。

**📊 数据集**

使用了Push‑T、LIBERO‑Long模拟任务以及四个真实双臂平台（handoff、close lid、fold cloth、measure）的演示数据集。

**📈 对比分析**

与动作仅预测的Diffusion Policy、B-spline Policy和FLOWER等基线相比，PAM在Push‑T覆盖率、LIBERO‑Long成功率以及四个真实任务的成功率均显著提升（如真实任务从47.5%提升至75%）。

**⚠️ 局限性**

限制包括草图未考虑物体动力学与接触、分层降噪增加额外推理步骤，以及对不同物体交互的通用性不足。

---

## 290. Jumping up and down: Denoiser diffusion models for discrete ordinal data

**arXiv ID:** 2610.02754 | [PDF](https://arxiv.org/pdf/2610.02754v1)

**作者:** Yair Shenfeld `[一作]` (Brown University), Stefano Peluchetti `[通讯]` (Sakana AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一类名为 Jumping Up and Down (JUD) 的基于去噪器的离散序数数据扩散模型，并实现了其在多种任务中的生成。

**💡 创新点**

创新点在于：①设计了可实现双向（上下）扰动的离散CTMC过程；②通过学习“潜在去噪器”即可推导出最终的数据去噪器；③提供了Binomial‑Poisson和Poisson‑Poisson两种过程，并证明了单一去噪器足以采样。

**🔧 技术方法**

使用了离散时间马尔可夫链、离散Tweedie公式、Bregman散度（如MSE）训练去噪器、tau‑leaping和Euler采样方法，并在CIFAR‑10上采用DDPM+++EDM预处理架构。

**📊 数据集**

实验数据集包括：合成离散分布（Poisson、混合、负二项等）、CIFAR‑10图像数据以及Sudoku拼图数据。

**📈 对比分析**

与多种连续、混合及离散扩散模型比较：在合成分布上取得更低的Wasserstein‑1距离；在CIFAR‑10上实现了2.07的FID（仅次于连续EDM模型）；在Sudoku完整预测中，Posterior CE方法达到66.99% 的准确率，远超其他离散扩散基准。

**⚠️ 局限性**

限制包括：①需从易采样的源分布（如Poisson）开始，无法处理任意源分布；②采样时需要大量神经网络评估；③Poisson‑Poisson方法需调节最终时间参数以提升近似精度。

---

## 291. RAOA: Alternating-Operator Neural Computation with Programmable Radio Propagation

**arXiv ID:** 2610.02683 | [PDF](https://arxiv.org/pdf/2610.02683v1)

**作者:** Toshiaki Koike-Akino `[一作]` `[通讯]` (Mitsubishi Electric Research Laboratories), Toshiaki Koike-Akino (Mitsubishi Electric Research Laboratories)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出并验证了 Radio Alternating Operator Ansatz（RAOA），将可编程无线传播路径视为可复用的计算深度，构造了交替执行问题更新与混合更新的递归框架，并在离散优化、受限相位传播仿真以及预训练 Transformer 的迁移学习等三大实验范式中进行了系统评估。

**💡 创新点**

创新点包括：①将可编程传播与经典交替算子优化（类似 QAOA）相结合，形成共享权重、可重复执行的深度循环；②证明即使仅使用两条可学习控制参数，递归深度增加也能提升性能；③设计零初始化残差适配器，将 RAOA 直接嵌入预训练 Transformer，实现高效迁移。

**🔧 技术方法**

核心技术涵盖：递归能量梯度更新 + Hadamard 混合器 + tanh 激活；离散优化中的 QUBO/HUBO 能量映射；受限相位只自由空间板（phase-only free‑space）编译的受限传播模型；以及零初始化的 Transformer 适配器插入。

**📊 数据集**

实验数据集：离散优化随机生成的 Max‑Cut、3‑SAT、Partition、Hypergraph Max‑Cut（n=8/12）实例；预训练模型迁移使用 WikiText 语料库（训练/验证分离）和 GSM8K 评估集。

**📈 对比分析**

与冻结模型、LoRA、浅层 MLP 等基线在相同参数预算下对比。RAOA 在 WikiText 上恢复 94–99% 的浅层 MLP 性能；在离散优化中，递归深度从 2 级提升至 16 级后成功率显著提升（如 3‑SAT 由 0.37 提升至 0.98）。在相位板传播实验中，4 面板实现后对 Max‑Cut 的理想性能保留约 70%。

**⚠️ 局限性**

局限性：仅为软件模拟，未提供真实硬件验证；相位板实现难以支持更高阶或更复杂的物理实现；预训练模型深度扩展未表现显著优势；部分目标（Partition）对深度不敏感；未评估鲁棒性、能耗、时延等工程指标。

---

## 292. Large language models exhibit unreliable updating of clinical judgment as patient evidence evolves

**arXiv ID:** 2610.02684 | [PDF](https://arxiv.org/pdf/2610.02684v1)

**作者:** Min Zeng `[一作]`, Rui Zhang `[通讯]` (University of Minnesota)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

评估大型语言模型在ICU患者随访中对临床风险评估的长期更新可靠性，检验模型在接收自身先前评估后是否会产生误差扩大、更新不对称等问题；

**💡 创新点**

首次揭示模型在纵向推理中因自我上下文导致误差增加、对先验信念具有因果影响，并提出基于客观证据验证的更新策略以提升更新可靠性；

**🔧 技术方法**

采用MIMIC‑IV匹配构建ICU轨迹，对Qwen及多家LLM进行独立与纵向推理，设计控制干预、提示干预和证据验证规则，并使用统计方法（bootstrap）评估性能；

**📊 数据集**

使用MIMIC‑IV ICU数据构建2000例匹配队列（1000机械通气正例+1000对照），并独立构造血管升压启动队列；

**📈 对比分析**

通过AUROC、Brier、MAE等指标与独立推理以及多模型比较，发现纵向上下文大多导致预测误差上升；控制干预显示模型对恶化证据更新更大；提示无效；EVLU虽降低误差但覆盖率极低；

**⚠️ 局限性**

局限包括仅基于MIMIC‑IV结构化数据、缺乏多模态/非结构化证据、EVLU覆盖率低、未验证不同机构或临床实时场景的泛化性。

---

## 293. Efficient Memory Crystallization for Graph Learning under Non-Stationary Distribution Shifts

**arXiv ID:** 2610.02795 | [PDF](https://arxiv.org/pdf/2610.02795v1)

**作者:** Yue Hou `[一作]` (Beihang University), Ke Xu `[通讯]` (Beihang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种名为Efficient Memory Crystallization（EMC）的测试时无训练框架，用于在非平稳图分布漂移环境下快速适应。

**💡 创新点**

创新点在于将每个新到的图域以闭式解方式凝练为紧凑、语义保真记忆，并通过状态演化记忆建模域间依赖，从而在无需生成模块的情况下获得更紧的泛化误差上界。

**🔧 技术方法**

核心技术包括基于记忆的分布匹配目标、闭式解构建记忆、类内子集聚合、状态演化记忆机制和KL正则化的连续适应损失。

**📊 数据集**

实验采用Facebook‑100、Twitch‑Explicit、OGB‑Arxiv和Elliptic四个节点分类数据集进行评估。

**📈 对比分析**

与10个主流基线（包括GCAL、CoTTA等）对比，EMC在平均性能(AP)和平均遗忘(AF)方面均表现更优，同时平均耗时下降87.4%，GPU显存下降92.4%。

**⚠️ 局限性**

局限性在于对超参数（如α、β）的敏感性、实验主要聚焦节点分类任务，且在极大规模图或标签漂移环境下的适用性尚待进一步验证。

---

## 294. ManiPhysicsBench: Physics-Based Assessment of Object Preservation in VLA Manipulation

**arXiv ID:** 2610.02802 | [PDF](https://arxiv.org/pdf/2610.02802v1)

**作者:** Sangwu Park `[一作]` (Korea Advanced Institute Of Science And Technology), Chanyoung Park `[通讯]` (Korea Advanced Institute Of Science And Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于物理的评估框架，结合可复用的对象资产和求解器推断抓取导致的破坏，形成了可在LIBERO和SimplerEnv上评估对象保护的VLA基准。

**💡 创新点**

首次将对象材料属性、几何网格与损伤阈值结合，并提出基于求解器的损伤评估与对象特定的连续抓取监督，揭示了任务成功与安全成功的差距。

**🔧 技术方法**

采用有限元求解器评估抓取力对破坏阈值的影响，构建对象资产库，利用模拟抓取记录计算接触力，并在VLA训练中加入连续抓取标签。

**📊 数据集**

使用LIBERO与SimplerEnv的任务集、公开的对象3D网格与文献材料数据，Bridge演示数据用于重新标注，并评估多种公开VLA模型（CogACT、GR00T、InternVLA-M1等）。

**📈 对比分析**

通过任务成功率(SR)、安全成功率(Safe SR)和保持率(PR)三项指标进行比较；结果显示多数模型SR高达40–70%但Safe SR仅为10–20%，对象特定监督可将Safe SR提升至25–64%但伴随SR下降。

**⚠️ 局限性**

评估基于文献的材料参数和求解器预测，尚需实物实验验证；仅考虑双手抓取，未覆盖掉落或撞击造成的损伤；对象特定监督的泛化能力有限，且降低了任务完成率。

---

## 295. From TS-SUF-2 to TS-SUF-4: Practical Security Enhancements for FROST2 Threshold Signatures

**arXiv ID:** 2610.02805 | [PDF](https://arxiv.org/pdf/2610.02805v1)

**作者:** Will Wang `[一作]` (Solv Protocol), Martin Zhao `[通讯]` (Solv Protocol)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种改进的阈值Schnorr签名方案（称为 DSign），在静态破坏与集中/分布式密钥生成场景下实现最高安全层级（i=4），同时保持两轮高效实现。

**💡 创新点**

通过在预处理令牌中加入轻量级身份验证机制（使用现有签名密钥和令牌中的秘密指数），消除 Bellare 等人揭示的 rogue‑public‑key 漏洞，使方案在保持原有效率的前提下实现 i=4 级安全；并且不需要额外的数字签名密钥对。

**🔧 技术方法**

利用 Schnorr 签名、一次更多离散对数（OMDL）假设、随机预言机模型、Lagrange 插值、批量验证以及预计算技术。

**📊 数据集**

在 ZCash 的 Ed25519 实现上进行基准测试，采用多组阈值参数（t=0.7n，n=50, 100, 150, 200）评估性能。

**📈 对比分析**

与 ZCash 原始实现、Bellare 等人的基准方案及其变体（如 5cSign 等）进行比较；DSign 在预计算开启时的签名吞吐量与 ZSign 接近，且比传统方案快 64%–79%，在 n=200 时达到 913 sig/s，验证其在性能与安全性上的优势。

**⚠️ 局限性**

局限性：安全证明仅在静态破坏和随机预言机模型下成立；需要预计算才能获得最佳性能；在完全自适应破坏或无预计算环境下，安全性或效率可能下降。

---

## 296. Skill2Real: Agentic Skill Learning for Zero-Shot Sim-to-Real Robot Manipulation

**arXiv ID:** 2610.02788 | [PDF](https://arxiv.org/pdf/2610.02788v1)

**作者:** Xincheng He `[一作]` (University Of California), Chenfanfu Jiang `[通讯]` (University Of California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Skill2Real 框架，学习可在仿真中训练的可冻结的两层技能层级（Cerebellum 本地操作技能和 Brain 任务级组合技能），并通过共享的代码化策略接口实现零-shot sim‑to‑real 转移；

**💡 创新点**

创新点在于：① 通过基础 VLM 进行跨域对齐，将任务级知识与低层感知/控制分离，② 采用异向 Proposer–Verifier–Governor（PVG）三代理机制，用仿真特权监督和验证来提升技能质量，③ 在仿真训练中直接学习可执行程序，实现高层次可组合性并在真实机器人上保持冻结不变的知识；

**🔧 技术方法**

技术包括：大型语言模型（GPT‑Sol、Claude Opus 5 等）作为 Proposer，Verifier 用仿真证据诊断结果，Governor 基于验证回放决定更新；共享代码化策略接口；两阶段 PVG 训练（Cerebellum 本地技能学习、Brain 任务级程序学习）；数据驱动的仿真环境 LIBERO‑90、Robosuite 及真实 UR5e 机器人；

**📊 数据集**

使用 LIBERO‑90（90 个本地操作任务）、Robosuite（7 个单臂/双臂任务）、LIBERO‑Pro Long（长序列任务）、以及实际 UR5e 机器人实验；

**📈 对比分析**

与 CaP‑Agent0、ASPIRE、Zetta 等现有方法在 LIBERO‑Pro Long 上比较，Skill2Real 在 3 个 VLM Proposer（GPT‑Astra、GPT‑Sol、Claude Opus 5）下均表现出最高的 Overall/Task 成功率（如 Astra 56.3% vs 51.5% Zetta），在真实 UR5e 上从 27.5% 提升至 78.8%；验证 PVG 角色对性能提升贡献显著（去掉 Verifier 或 Governor 均下降 13–17%）；

**⚠️ 局限性**

局限性包括：① 仍需 VLM 辅助推理，推理延迟较高；② 对于极窄空间或需要高频反馈的操作，代码化策略接口的可执行性不如端到端 VLA 策略；③ 依赖于仿真特权信息，未在非仿真环境中验证；

---

## 297. RMCW: A Deletion-Robust Watermark Based on Reed--Muller Codes for Language Models

**arXiv ID:** 2610.02817 | [PDF](https://arxiv.org/pdf/2610.02817v1)

**作者:** Yi Wang `[一作]` (Tsinghua University), Tianxing He `[通讯]` (Tsinghua University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于 Reed–Muller 码的 LLM 水印方法，能够在删除攻击下保持可检测性。

**💡 创新点**

创新点在于将全局码字恢复转为局部 Reeds–Solomon 一致性检测，从而抵御同步丢失问题。

**🔧 技术方法**

使用 Reed–Muller 码、Reed–Solomon 低度一致性检验、Berlekamp–Welch 算法以及密钥词表分箱等技术。

**📊 数据集**

在 C4 与 ELI5 两个数据集上，使用 OPT‑1.3B 与 Llama‑3.1‑8B‑Instruct 两个模型进行实验。

**📈 对比分析**

与 KGW、EXP、PRC 等基线相比，在干净文本下 TPR@1%FPR 达 99.8%，在突发删除、截断、同义词替换等攻击下保持高检测率（如 98.1%/86.6%），整体性能优于或与基线相当。

**⚠️ 局限性**

局限性包括对强重写（重释）攻击鲁棒性不足、短文本检测效果差，以及需要针对不同模型/域重新校准检测参数。

---

## 298. Controlling Polar Exposure to Delay Memorization in Diffusion Models

**arXiv ID:** 2610.02780 | [PDF](https://arxiv.org/pdf/2610.02780v1)

**作者:** Xuanchen Wang `[一作]` (University of Sydney), Weidong Cai `[通讯]` (University of Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了质量门控去白化(QGD)控制器，用于在扩散模型训练中保持快速学习前缀并通过复制反馈限制后续极化暴露，从而延长泛化窗口；同时引入复制预算选择(CBS)以在冻结的检查点族中按复制概率阈值和质量进行安全释放。

**💡 创新点**

创新点在于：①将随机特征分析中的协方差、曲率和平衡振幅三种记忆钟分离，并证明极化曝光的有限性可恢复与数据集规模相关的复制延迟；②设计了基于质量门控与因果复制反馈的逐步极化衰减策略；③构建了结合二项校准和置信上限的检查点选择方法(CBS)，实现对复制预算的联合误差保证。

**🔧 技术方法**

使用的技术包括：随机特征降维的线性扩散模型、Muon's正交矩阵动量优化、极化更新、固定增益尾部恢复、复制检测器（像素距离阈值）、二项置信区间校准、离线验证银行、以及对流匹配和舞蹈生成等任务的扩散模型。

**📊 数据集**

主要数据集：CIFAR-10（2,000/10,000图像子集），以及用于迁移测试的流匹配数据集和AIST++舞蹈生成数据集。

**📈 对比分析**

对比方法包括SGD、AdamW、Muon、同门硬切换以及QGD+CBS。实验显示：QGD在CIFAR-10上在保持质量到达时间的同时将有用窗口扩大约8.3倍，复制率下降约75.9%，最终FID从79.37降至75.56；在流匹配和舞蹈生成任务中也实现了较低的FID和复制率。

**⚠️ 局限性**

限制：理论保证基于对齐、固定特征和理想极化动态，未考虑非线性特征学习或动态表示；实验仅覆盖有限的模型规模和数据集，结果在更大规模下可能不同；CBS仅针对所选复制检测器和阈值，无法防止其他形式的记忆或信息泄露；评估频率和检查点族大小决定生成与校准成本，实际部署需权衡。

---

## 299. OPD Before RL: Warm-Starting Rubric-Based RL with On-Policy Distillation

**arXiv ID:** 2610.02781 | [PDF](https://arxiv.org/pdf/2610.02781v1)

**作者:** Xinpeng Wang `[一作]` (New York University), Richard Yuanzhe Pang `[通讯]` (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段训练框架：首先用带有任务评估规范（rubric）的教师对学生进行稠密的基于token的对齐训练（rubric‑privileged on‑policy distillation），然后在此基础上继续进行rubric‑based强化学习，以提升开放式自然语言生成任务的质量。

**💡 创新点**

创新点在于：1）将任务规范作为教师的特权信息，用于生成更丰富的token分布，提供更细粒度的监督；2）通过先让模型在“更高质量”教师的引导下保持更高的token熵，从而在后续RL阶段获得更好的探索性和更少的奖励作弊；3）展示了OPD在warm‑start阶段对RL性能的显著提升，并与传统SFT+RL比较，验证了更少的reward hacking。

**🔧 技术方法**

使用技术包括：
- 先行的rubric‑conditioned on‑policy distillation（采用前向KL），
- 强化学习（PPO/REINFORCE）以优化rubric‑based奖励，
- 采用更强大的教师模型（Qwen2.5‑32B、Llama‑3.1‑70B）对学生生成的前缀进行评分，
- 用LLM判别器（Qwen3‑32B或GPT‑4o‑mini）计算每条回复的rubric分数。

**📊 数据集**

实验数据集包括：HealthBench（医学问答）、ResearchQA（科学研究问题）和RubricHub Science（技术/科学问答）。

**📈 对比分析**

与SFT、SFT+RL以及直接无warm‑up的rubric‑based RL进行对比。两阶段OPD+RL在所有三组数据集上都取得了最高的rubric分数，并保持较高的token熵；相比SFT+RL，OPD+RL在reward hacking上表现更稳健。总体上，OPD+RL的性能优于单纯SFT或无warm‑up的RL。

**⚠️ 局限性**

局限性包括：
- 需要手工编写或提供适合任务的rubric，且教师必须足够强大，导致额外的推理成本；
- 评估主要集中在健康和科学领域，未验证在其他开放式生成任务上的通用性；
- 强调使用统一判别器的比较，判别器质量会显著影响结果，未来需探究更强判别器对性能的影响。

---

## 300. Performance Evaluation of Emerging Networks of Quantum Repeaters: Analysis and Simulation

**arXiv ID:** 2610.02778 | [PDF](https://arxiv.org/pdf/2610.02778v1)

**作者:** Kobi Ravid `[一作]` (Columbia University), Gil Zussman `[通讯]` (Columbia University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文针对基于现有光纤基础设施的量子网络，构建了一个完整的量子+经典混合仿真平台，验证并分析了双输入重复器（DIR）的内存利用和成对速率（PR），随后在单重复器与多重复器线拓扑网络上评估了纠缠对速率（EPR）与平均保真度（Fidelity）的表现，并提出了一种简单的贪心交换策略。

**💡 创新点**

创新点主要包括：① 在DIR上给出了可解析的内存大小-性能关系，证明即使内存利用率无界，极小的内存容量（如 M=3）也可实现与最优速率相差 ≤3% 的性能；② 开发了一个集成了量子与经典通道、损耗、误差及时延的仿真平台，可直接与理论分析对比；③ 对实际 SCY‑QNet 以及人工合成线网络进行系统的 EPR 与 Fidelity 评估，揭示了纤维长度、内存错误率、外部延迟等因素对性能的影响。

**🔧 技术方法**

主要技术方法包括：离散事件仿真、离散时间马尔可夫链（DTMC）分析、量子门模拟（CNOT、Hadamard 等）、经典信号传播时延模型、误差通道模型（幅度衰减、相位翻转）以及贪心式的交换选择算法。

**📊 数据集**

使用的实验数据集：SCY‑QNet 的实际网络拓扑及光纤距离（如 SBU、BNL、CBU 等节点的实际路径长度），以及从公开光纤基础设施（Crown Castle 在 Seattle 和 Orlando）的 GIS 数据绘制的图；另外还使用了合成的等距线拓扑和 1:2 长度比线网络进行对比实验。

**📈 对比分析**

通过将仿真结果与先前文献中针对多输入重复器的理论公式（E[Q]）、以及针对 DIR 的新 DTMC 解析结果进行对比，误差均低于 3%（或 0.3% 的 PR 匹配误差）。EPR 在高内存尺寸下可逼近理论最大速率，Fidelity 在低内存错误率下保持在 0.9 以上；然而在长距离或高错误率场景下，Fidelity 下降到 0.3–0.5，显示出对内存错误的高度敏感性。

**⚠️ 局限性**

局限性包括：① 仅考虑无环线拓扑，未涵盖更复杂的网格或树形网络；② 交换策略为简化贪心算法，未探索更优的调度或多路径路由；③ 内存错误模型仅为单一幅度衰减和相位翻转，未考虑更完整的噪声或误差校正机制；④ 只评估了少数典型的光纤长度和错误参数，缺乏更广泛的参数空间覆盖；⑤ 对时延同步、信道复用及资源分配等实际工程细节关注不足。

---

## 301. Gated Slot Attention-2: Two-Sided Associative Memory Correction in Linear Attention

**arXiv ID:** 2610.02816 | [PDF](https://arxiv.org/pdf/2610.02816v1)

**作者:** Ruijie Li `[一作]` (Hong Kong University of Science and Technology), Yuxuan Liang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的两阶段门控槽注意力网络GSA2，用于高效序列建模。

**💡 创新点**

创新点在于将Oja规则用于键侧纠正，与Delta规则用于值侧纠正相结合，并在两阶段槽记忆中解耦擦除-写入控制。

**🔧 技术方法**

使用门控Oja规则、门控Delta规则、共享潜在槽、低秩WY/UT分块实现、SiLU激活、低秩投影以及SWA混合。

**📊 数据集**

在100B FineWeb-Edu训练数据上评估，使用WikiText、LAMBADA、PIQA、BoolQ、NQ等多任务以及长上下文评估RULER、LongBench。

**📈 对比分析**

与Transformer、Mamba、GDN、KDA、GDN2、GSA、OJA等基线比较，GSA2在语言建模、常识推理、检索和长文本任务上取得更优或相近表现，同时保持线性时间和常数内存。

**⚠️ 局限性**

局限包括相对较高的计算开销（相比单一规则稍慢），对超参数如槽数敏感，且在极长上下文或更大模型规模下的性能需进一步验证。

---

## 302. iS-KV: Online Low-Rank KV Cache Compression via Block-Incremental SVD

**arXiv ID:** 2610.02815 | [PDF](https://arxiv.org/pdf/2610.02815v1)

**作者:** Yiren Zhao `[一作]` (Hong Kong University of Science and Technology), Xitong Gao `[通讯]` (Institutes of Advanced Technology Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在线低秩 KV 缓存压缩方法，在长链式推理中保持最近窗口精确，旧状态压缩为固定秩表示；

**💡 创新点**

通过同步更新基底与历史系数的块增量 SVD，解决传统基底更新导致历史漂移的问题；

**🔧 技术方法**

使用块增量 SVD、低秩压缩、RoPE 预旋转、关键子集保护策略、在线更新与重正交化等技术；

**📊 数据集**

在 DeepSeek‑R1‑Distill‑Llama‑8B 与 Qwen3‑8B 上评估，主要使用 MATH‑500、AIME 2024/2025 等长推理数据集；

**📈 对比分析**

与 R‑KV、SnapKV 等基于 token 淘汰的方法对比，在匹配内存预算下，取得 4.06×–5.64× 的压缩率，准确率仅比原模型低 1–6 点，且在所有预算点上均优于基线；

**⚠️ 局限性**

仍存在内存随推理长度线性增长、压缩后精度略低、计算与重正交化带来的额外开销，以及仅在 8B 模型上验证的局限性。

---

## 303. Learning Reflexive Behavior for Contact-Rich Manipulation

**arXiv ID:** 2610.02811 | [PDF](https://arxiv.org/pdf/2610.02811v1)

**作者:** Quan Nguyen `[一作]` (Neuromeka Co Ltd), Joonho Lee `[通讯]` (Neuromeka Co Ltd)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文训练了一个仅利用关节位置历史的本体感知反射策略，嵌入在高层命令与低层关节控制之间，实现了在接触丰富环境下的鲁棒执行。

**💡 创新点**

创新点在于通过仅三种交互原语（弹簧、平面、轨道）在仿真中训练反射策略，并通过合成的接触阈值奖励与强制方式，使策略无需力/几何感知即可自动调节执行以满足局部约束。

**🔧 技术方法**

采用深度强化学习（PPO+非对称Actor-Critic），结合时序卷积编码器、双层奖励（位置跟踪、力阈值惩罚、正则化）以及多实例交互原语采样。

**📊 数据集**

使用自生成的三种交互原语与两款机器人（工业人形和QDD机械臂）的仿真环境，并在实际硬件上执行双臂提箱、插销、表面跟随等任务；未使用公开数据集。

**📈 对比分析**

与传统基于IK、惯性阻抗与混合运动–力控制的执行层相比，反射策略在硬件双臂提箱、插销（0.02mm容差）和粗糙表面跟随等实验中分别提升了约20%–30%的成功率，显著降低了平均接触力（≤50%），且在不需要手动调参的情况下保持良好性能。

**⚠️ 局限性**

主要局限包括：仅针对平移约束的交互原语，未覆盖旋转耦合场景；需要外部模式指令（compliance mode）才能区分跟踪与接触；在更复杂多模态任务中的通用性与鲁棒性尚待验证。

---

## 304. BitNest: Bit-Nested Speculative Decoding for Memory-Efficient LLM Inference Acceleration

**arXiv ID:** 2610.02800 | [PDF](https://arxiv.org/pdf/2610.02800v1)

**作者:** Chence Yang `[一作]` (University of Georgia), Geng Yuan `[通讯]` (University of Georgia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出BitNest自推测解码框架，在同一权重量表中嵌入低精度草稿与高精度目标，实现无额外模型存储的自推测解码；

**💡 创新点**

创新点在于：1) 采用“基底先行”构造低精度草稿并通过残差量化构建高精度目标，保持嵌套关系；2) 设计双平面权重量存储，草稿仅读取基底平面，目标读取完整；3) 将嵌套精度原则扩展至KV缓存，实现长上下文推理加速；

**🔧 技术方法**

技术包括：函数保持旋转量化、GPTQ量化、4/8位双平面存储、KV嵌套、推测长度调优、Edge设备部署与能耗评估；

**📊 数据集**

使用LLaMA-2-7B、LLaMA-3-8B、Qwen2-7B、Qwen2.5-7B四个7B–8B模型；任务涵盖WikiText-2、GSM8K、代码生成（HumanEval、MBPP）、对话（ShareGPT）、长文本（LongDoc、PG‑19）及长上下文（LLaMA-2-7B-32K）；

**📈 对比分析**

与QSpec、QuantSpec、Draft & Verify等自推测基线对比，BitNest在六个工作负载上平均获得1.48–1.61×的端到端速度提升，接受率≈95%，并在FP16基准上几乎无模型质量损失；在Jetson Orin NX边缘设备上实现约1.5×速度提升、能耗下降≈20%；

**⚠️ 局限性**

局限性：仅针对自推测解码，未利用可提供更大候选并行性的辅助草稿模型（如扩散式推测），未来可在保持嵌套内存优势的同时支持此类扩展。

---

## 305. SARI: Phase-Split Sim-Real Co-Training for Contact-Rich Manipulation

**arXiv ID:** 2610.02804 | [PDF](https://arxiv.org/pdf/2610.02804v1)

**作者:** Xingxin He `[一作]` (Hong Kong University of Science and Technology), Ziqi Wang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种分阶段的模拟与真实共同训练框架 SARI，先在数字孪生中生成多样化的自由空间到预接触的逼真轨迹，再在真实环境中仅收集少量接触阶段的数据，通过视觉对齐和摄像机相对动作空间实现单一策略的无缝拼接。

**💡 创新点**

核心创新在于：① 将任务分为“自由空间”与“接触”两阶段，分别在最适合的域（模拟与真实）收集数据；② 通过光照、颜色校正及完整的三维重建实现跨域视觉对齐；③ 将动作表达转换为摄像机相对坐标系，从而使模拟与真实数据在同一动作空间中可直接混合训练；④ 在五个接触丰富的操纵任务上显著降低真实数据采集量并获得在未见放置位置上的成功率。

**🔧 技术方法**

使用 Gaussian Splatting 数字孪生进行高保真渲染；摄像机相对动作表示（camera-relative action representation）；行为克隆（flow‑matching imitation learning）在合并后的数据集上进行后训练；视觉对齐采用高阶球谐波、颜色映射校正；实验平台为 Franka Research 3 + Franka Hand，观测使用两台外部相机与手腕相机。

**📊 数据集**

在五个任务（按压按钮、开灯、拉抽屉、翻书、拉纸巾）中，每个任务都使用了六个目标位置，其中真实接触演示仅覆盖两位置；模拟演示覆盖全部六个位置；共计约 40 个放置位置作为评估。数据来源主要是重建的数字孪生与少量真实演示。

**📈 对比分析**

对比了三种基线：全模拟、全真实、以及全任务的模拟‑真实联合训练。SARI 在“未见”位置的任务完成率为 27.5%，比基线 0% 提升显著；在“已见”位置完成率为 50%，同样优于基线；同时真实演示时间从 28.03 分钟降低到 18.42 分钟，节约约 34.3%。

**⚠️ 局限性**

局限性包括：仅适用于固定桌面场景与单臂 Franka 机器人；需人工完成场景重建、相机标定、预接触姿态与颜色参考选择；无法处理动态场景、多阶段交互或多机器人情形；自动化准备与跨机器人通用性仍待研究。

---

## 306. Nearly Optimal Fixed-Confidence Best-Arm Identification with 1-Bit Feedback

**arXiv ID:** 2610.02771 | [PDF](https://arxiv.org/pdf/2610.02771v1)

**作者:** Khang Luong `[一作]` (Hanoi University Of Science And Technology), Tuan Quang Dam `[通讯]` (Hanoi University Of Science And Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究在仅获得单比特阈值反馈的条件下实现固定置信度的最佳臂识别问题；

**💡 创新点**

创新点包括提出一种基于随机阈值和截断尾积分恒等式的时间统一单比特均值估计器，并利用其构造了自适应裁剪的分阶段最佳臂识别算法，理论上与信息学下界匹配；

**🔧 技术方法**

核心技术涵盖随机阈值查询、截断尾积分身份、时间统一置信序列、UGapE候选‑对手框架、分阶段裁剪以及相应的改变测度下界证明；

**📊 数据集**

实验使用了多种合成分布（高斯、Student‑t、伯努利、指数）以及标准的多臂赌博机环境来验证算法；

**📈 对比分析**

与基线单比特估计器和全信息UGapE对比，单比特均值估计器在平均绝对误差上更优；分阶段裁剪算法在具有异质间隙的环境中样本复杂度显著低于固定裁剪算法；

**⚠️ 局限性**

局限性在于需要预先获取每臂均值的锚点进行局部化，且仅给出了最坏情况下的实例依赖复杂度，未能完全揭示所有实例的最优期望复杂度，并且对其他纯探索任务的推广仍待研究。

---

## 307. No-Free-Graph: Learning When Multimodal Data Should Be Graphified

**arXiv ID:** 2610.02768 | [PDF](https://arxiv.org/pdf/2610.02768v1)

**作者:** Zekai Chen `[一作]` (Beijing Institute of Technology), Rong-Hua Li `[通讯]` (Beijing Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在多模态图学习中是否需要先构建图，并提出了MAG-Scout框架来在完全生成图之前预测图构建的收益；

**💡 创新点**

创新点在于把图构建视为预先的决策问题，利用有限的关系样本构造关系sketch，并通过三重瓶颈推理（必要性、可构造性、效用性）评估潜在收益，同时加入成本与不确定性实现构建与跳过的智能决策；

**🔧 技术方法**

技术包括任务与构造器条件编码的关系表示、基于预算的候选关系采样与sketch构造、三瓶颈潜能推理网络、成本感知效用预测与阈值策略，以及联合优化的损失（收益、排名、置信度、决策）；

**📊 数据集**

使用六个OpenMAG多模态基准（Toys、Grocery、Bili Music、DY、QB、Bili Cartoon），涵盖节点分类、链接预测和跨模态检索三类任务；

**📈 对比分析**

与NetInfoF、WDGH、GLEMOS-S2、MetaGL等现有图效用评估器对比，MAG-Scout在保持95%正增益保留率的前提下节省约23.6%的图构建工作，保留96.7%的正增益；在多任务、多构造器场景下表现稳定；

**⚠️ 局限性**

局限性包括对关系证据的敏感性（误跳率约13%）、需手动调优阈值、在某些构造器或任务上仍有较高误跳率、对极端噪声关系的鲁棒性仍待提升。

---

## 308. Adaptive Spectral-Koopman Dynamics Modeling for Temporal Domain Generalization

**arXiv ID:** 2610.02822 | [PDF](https://arxiv.org/pdf/2610.02822v1)

**作者:** Tengxue Zhang `[一作]` (East China Normal University), Bin Yang `[通讯]` (East China Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5a41884c-404f-4688-a89c-aa238c10fe68` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 AdaSpecK 框架，结合谱正则 Koopman 动力学与自适应上下文提取，解决时间不规则采样与非平稳性导致的 Temporal Domain Generalization（TDG）难题。

**💡 创新点**

创新点包括：① 在潜在空间中使用谱滤波提取低频轨迹，抑制噪声后学习 Koopman 线性化动力学；② 采用目标条件注意力与环境签名路由，实现对历史窗口的异质化、可自适应的上下文聚合；③ 将谱正则、Koopman 动力学与自适应上下文机制整合为统一损失，提升跨域时间推断稳健性。

**🔧 技术方法**

技术手段：离散傅里叶变换（DFT）谱滤波、Koopman 运算符学习、目标条件注意力机制、基于专家的路由器、正交语义约束、复合损失（Koopman、重建、一致性、语义正交）。实现框架基于 PyTorch，并使用 Adam + Cosine Annealing 训练。

**📊 数据集**

实验数据集共八个：分类任务 - Rotated MNIST、Twitter Influenza Risk、YearBook、Online News Popularity、Shuttle；回归任务 - Tropical Cyclone Intensity、House Prices、ApplianceEnergy。

**📈 对比分析**

与时间无关基线（Offline、LastDomain、IncFinetune、IRM、V-REx）、离散时间域自适应方法（CIDA、TKNets、DRAIN、DRAIN-Δt）以及连续 TDG 方法（DeepODE、Koodos、Frekoo）对比，AdaSpecK 在所有八个基准上均取得最佳或第二佳结果，显著提升尤其在 Twitter、YearBook、Cyclone 与 Appliance 等多步未来演化任务上。

**⚠️ 局限性**

局限性：对超参数（λ_K、λ_S、λ_C、谱比例 β）和专家数量 K 的敏感度需进一步调优；模型相对复杂，计算成本和存储需求高；实验仅覆盖公开基准，尚未验证在极快概念漂移或更高维潜在空间下的表现。

---

## 309. When History Fails to Become Experience: Action Calibration in Language Agents

**arXiv ID:** 2610.02769 | [PDF](https://arxiv.org/pdf/2610.02769v1)

**作者:** Jingyu Liu `[一作]` (Renmin University of China), Yong Liu `[通讯]` (Renmin University of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究语言代理在同一任务中如何利用交互历史来改进决策，并系统评估了不同历史处理方式的效果

**💡 创新点**

提出将观察标记为前一动作结果（Outcome标记）并训练自适应校准器（Calibrator）以选择性记录经验，从而显著提升任务完成率

**🔧 技术方法**

使用Prompt Engineering对历史记录进行结构化处理，利用大型语言模型（GPT‑5.6‑Sol、DeepSeek‑V4‑Pro、Qwen3‑系列）执行动作生成与经验校准，结合监督微调训练校准器

**📊 数据集**

在四个多任务环境（ALFWorld、ScienceWorld、WebShop、AppWorld）上进行实验，使用五种模型的500‑任务样本进行全回放评估

**📈 对比分析**

与Raw（原始历史）、Reference（额外参考历史）、Shuffle（动作打乱）、Outcome（结果标记）以及Experience（自生成经验）等基线比较，发现Calibrator在所有模型/环境均提升约3–6个百分点，最高可达10%点的成功率提升

**⚠️ 局限性**

自生成经验可能包含重复或无依据的推断，导致在全回放中性能不如Outcome标记；校准器虽有效但仍受限于对经验质量的判定准确性

---

## 310. AptMQL-Bench: From Text-to-SQL to Text-to-MQL via Access-Pattern Schema Design and Data-Preserving Migration

**arXiv ID:** 2610.02770 | [PDF](https://arxiv.org/pdf/2610.02770v1)

**作者:** Hy Nguyen `[一作]` (University of Sydney), Robin Vujanic `[通讯]` (MongoDB Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文将现有文本到SQL基准转换为文本到MongoDB查询（MQL）基准，构建了AptMQL‑Bench，并通过编码代理与人工验证的流水线实现无数据损失且高效的数据库迁移与查询生成。

**💡 创新点**

创新点在于提出基于访问模式的文档架构设计方法，并结合自动编码代理和人类审核形成多阶段转换流程，显著提升数据完整性与查询效率，优于传统的一对一映射和外键嵌入。

**🔧 技术方法**

主要技术包括：①编码代理（如Claude Code）执行文档模式设计、数据库迁移脚本生成和SQL‑>MQL 翻译；②多轮验证脚本与人工审核确保数据无丢失；③MQL 优化阶段去除 SQL 形态遗留；④对八个LLM进行 Soft‑EX、Strict‑EX、Exec‑Rate 等指标评测。

**📊 数据集**

使用了 BIRD 文本到SQL 基准的 21 个数据库、3,186 条自然语言请求及其对应的 MQL 查询；并与 DocSpider、TEND‑v2/v3、NL‑to‑MongoSH、SM3‑Text‑to‑Query 等基准进行对比。

**📈 对比分析**

比较方法：在 AptMQL‑Bench 上评估八个LLM的 Exec‑Rate、Strict‑EX 与 Soft‑EX；对三种转换策略（一对一、FK 嵌入、访问模式）在数据完整性和查询执行时间（log–log 规模化实验）进行对比。性能方面，访问模式设计在数据完整性上无损失，执行时间比 FK 嵌入快约10‑11 倍，且比一对一快两位数；Claude‑Opus‑4.5 在 Soft‑EX 上最高达 70.34%。

**⚠️ 局限性**

限制：①访问模式是基于假设工作负载，可能不反映实际使用；②人工审核时间为估计值，未进行精确计量；③仅在 MongoDB 上验证，未检验跨文档存储的通用性；④转换流程需人工推断业务场景，对大规模数据库适用性仍需进一步验证。

---

## 311. Correlation-Based Distillation Yields More Mergeable Speech-Music Encoders

**arXiv ID:** 2610.02836 | [PDF](https://arxiv.org/pdf/2610.02836v1)

**作者:** Fabian Ritter-Gutierrez `[一作]` `[通讯]` (Nanyang Technological University), Fabian Ritter-Gutierrez (Nanyang Technological University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了在语音与音乐任务的知识蒸馏后模型合并时，不同蒸馏损失对合并可行性的影响，探究了交叉相关蒸馏损失对两种合并算法的表现提升。

**💡 创新点**

发现交叉相关蒸馏损失（CC）比传统的DistilHuBERT损失更能提升两种完全无关合并方法（任务算术插值与激活排列）的融合效果，且此提升与模型结构无关，提示损失形态决定了可合并的表示。

**🔧 技术方法**

采用交叉相关蒸馏损失（σ_CC+ σ_SC-γσ_cos），任务算术插值与交叉相关排列（CP）两种无监督合并技术，并使用标准的Transformer编码器和自监督训练框架。

**📊 数据集**

使用LibriSpeech 960h 作为语音数据，Music4All 与 MERT 作为音乐教师，评估任务包括 SUPERB（ASR, KS, IC, ER）与 MARBLE（歌手、声学技巧、乐器、流派识别）以及 ESC-50 环境音分类。

**📈 对比分析**

通过在同一学生池上对比两种损失在两种合并算法下的性能，使用九任务均衡评分体系（Speech Score、Music Score、Average Score）和鲁棒性指标Δ，结果显示CC损失在17/18比较中保持更高Speech Score，且在CP合并中需要更少的通道重排，整体提升约10–15点。

**⚠️ 局限性**

局限性包括仅在固定架构与训练步骤下验证，未探讨不同模型宽度对CC损失的影响；CP合并对初始权重的依赖仍是限制；缺乏对更大规模或不同域的验证。

---

## 312. MetaRubric: Learning to Reward for Rubric-Based Reinforcement Learning

**arXiv ID:** 2610.02824 | [PDF](https://arxiv.org/pdf/2610.02824v1)

**作者:** Yuxuan Fan `[一作]` (NTU Singapore), Jaehong Yoon `[通讯]` (NTU Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MetaRubric方法，通过证据感知的政策优化与基于响应的规则修订联合提升鲁棒性。

**💡 创新点**

创新点在于将判断标准拆分为满足度、覆盖度、支持度，并用对抗性反事实提示与自适应权重/修订同步更新规则，以消除虚无信用。

**🔧 技术方法**

使用GRPO、证据感知奖励、辅助QA奖励、对照反事实提示、权重自适应与语义锚定规则修订等技术。

**📊 数据集**

在PubMedQA、HealthBench‑Hard、MMOral‑X、MMOral‑OPG等医学文本和多模态问答数据集上进行评估。

**📈 对比分析**

与静态规则GRPO、DAPO、Dr. GRPO、GSPO等基线相比，MetaRubric在21个模型-指标组合中均实现显著提升，PubMedQA准确率提升6–20个百分点，HealthBench‑Hard 2–3个百分点，MMOral 1–3个百分点。

**⚠️ 局限性**

限制包括：最终规则单独训练难以完全复现全部提升；方法依赖高质量判定器与反事实提示生成；对非医学任务的可迁移性尚未验证。

---

## 313. AMBER: Multi-View Adaptive Budget Allocation for Listwise Vision-Language Reranking

**arXiv ID:** 2610.02831 | [PDF](https://arxiv.org/pdf/2610.02831v1)

**作者:** Wenteng Chen `[一作]` (Shanghai Jiao Tong University), Jianghao Lin `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在线预算化的多视角视觉语言模型重排序框架AMBER；

**💡 创新点**

通过将局部列表排序结果转化为连续Elo更新，实现轻量级全局排名状态，并在候选层和查询层双重自适应资源分配，基于信息增益和子模性质提供近似最优调度；

**🔧 技术方法**

利用Elo更新（等价于Bradley–Terry极大似然梯度上升）、子模信息增益近似、局部视图熵度量、头部重要性与观测惩罚等技术；

**📊 数据集**

在CIRR、CIRCO和PhotoBench三大多模检索基准上进行评估；

**📈 对比分析**

与embedding‑only检索以及多种多调用VLM重排序基线（AcuRank、Sliding Window、TourRank、UniRank）对比，在24调用和8调用两种预算设置下，AMBER在Recall@K、mAP、NDCG等指标上均优于所有基线，平均排名最低；

**⚠️ 局限性**

局限在于仅在图像检索上验证，未考虑链式推理（CoT）或动态超参调优，且模型假设VLM输出符合Bradley–Terry独立性，实际中可能存在位置偏置或上下文敏感性。

---

## 314. Co-Designing AI For Mental Health Support With Young Adults of Color (YOC): Needs, Expectations, and Implications for AI Literacy

**arXiv ID:** 2610.02812 | [PDF](https://arxiv.org/pdf/2610.02812v1)

**作者:** Elaine Dabin Jeon `[一作]` (University of Southern California), Angel Hsing-Chi Hwang `[通讯]` (University of Southern California)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过为期两天的共同设计工作坊，邀请13名18-24岁的有色人种年轻人（亚裔、黑人、拉美裔/西班牙裔）参与，探讨他们对AI聊天机器人在心理健康支持方面的需求、期望与担忧，并让参与者共同设计更具身份感知与文化适应性的聊天机器人功能。

**💡 创新点**

创新点包括：①首次将有色人种年轻人视为设计伙伴，采用参与式共创方法收集文化特定需求；②提出整合AI素养与心理健康素养的概念框架，阐明两者的交叉缺口；③生成针对身份感知、隐私权衡与多角色支持的设计原则，为未来AI聊天机器人的文化适配与责任阐释提供指导。

**🔧 技术方法**

技术手段主要为：①基于ChatGPT的人工对话实验（使用研究者提供的临时账号）；②共创工作坊流程（故事板、低保真原型、价值敏感设计活动）；④归纳性主题分析（Braun & Clarke 6阶段）。未开发新的机器学习模型。

**📊 数据集**

数据集为：13名参与者的访谈记录、工作坊现场音频与笔记、聊天日志（ChatGPT交互文本）、前后调查问卷以及低保真原型与设计方案。

**📈 对比分析**

该研究不涉及算法性能比较，重点在于定性洞见与设计输出；未对聊天机器人进行性能评估或对照实验。

**⚠️ 局限性**

局限性包括：样本规模小且仅限美国西海岸都市地区，缺乏跨地区、跨文化验证；使用临时ChatGPT账号限制了聊天记录的持续性与个性化；工作坊受主持人引导影响，可能产生社会期望偏差；研究仅关注工作坊过程，对实际使用场景的长期影响未知。

---

## 315. What Actually Makes Correlation-Based SSL Distillation Noise-Robust? A Mechanistic Correction

**arXiv ID:** 2610.02823 | [PDF](https://arxiv.org/pdf/2610.02823v1)

**作者:** Fabian Ritter-Gutierrez `[一作]` (Nanyang Technological University), Eng Siong Chng `[通讯]` (Nanyang Technological University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析并纠正了自监督学习中的噪声鲁棒性机制，证明交叉相关项而非自相关项是噪声抑制的关键，并给出了相应的设计规则。

**💡 创新点**

首次通过四种诊断（Pearson-variance分解、探针偏差诊断、相同噪声因果控制、维度分析）明确交叉相关对噪声鲁棒性的真正贡献，从而纠正了之前对自相关项的错误归因。

**🔧 技术方法**

采用Barlow Twins风格的相关性蒸馏、Pearson方差分析、无降维和维度级噪声分类探针，以及相同噪声控制实验和多任务下游评估。

**📊 数据集**

在LibriSpeech-100、CHiME-3、MUSAN、WHAM!、FSD50K、ESC-50以及音乐任务（SingerID、VocID、InstCls）和环境音任务（ESC-50）上进行训练与评估。

**📈 对比分析**

将原始KL蒸馏、交叉相关仅、完整相关损失三种设置在噪声分类准确率、ASR WER、情感识别、音乐/环境音任务中对比，完整损失在噪声分类准确率从约76.98%下降到55.55%（更噪声鲁棒），在干净任务上提升0.5%–2%准确率。

**⚠️ 局限性**

该机制仅在教师与学生噪声独立采样的加性噪声场景下有效，对卷积或非加性噪声抑制有限；在音调/音色敏感的音乐任务中，由于数据增强削弱了关键信号，导致性能下降。

---

## 316. FUSEye: Training-Light Fisheye Detection with Overlapping Views and Zero-Initialized Adapters

**arXiv ID:** 2610.02799 | [PDF](https://arxiv.org/pdf/2610.02799v1)

**作者:** Wenya Su `[一作]` (Hunan University), Kailun Yang `[通讯]` (Hunan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出FUSEye框架，将COCO预训练的YOLO检测器轻量化地适配到鱼眼图像，实现低成本、低标签、低算力的目标检测；

**💡 创新点**

创新点在于三层协同模块：GridViews产生重叠全局与局部视图以补偿边界压缩；Zero‑Initialized Residual Adapters在冻结主干的前提下纠正畸变导致的特征失配；AgreeFusion通过跨视图一致性学习重新评分并融合检测结果；

**🔧 技术方法**

采用重叠网格裁剪、残差适配器、交叉视图同意融合、轻量化MLP重评分、权重冻结训练等技术；

**📊 数据集**

使用WoodScape环视鱼眼数据集（MVR验证集）以及其前视、后视、镜左视进行训练；

**📈 对比分析**

与直接迁移和完整微调对比，FUSEye在YOLO26‑x上将mAP50从14.80提升至26.61（+11.81点），mAP50:95提升至17.26（+6.30点），并在仅使用25%训练样本时仍保持97.6%性能；同样的改进在YOLOv8/9/10/11等多种模型上均可复制；

**⚠️ 局限性**

局限在于对巴士类检测提升不足，且在完整微调后仍存在较大误差；框架对特定畸变类型的适应仍需进一步优化。

---

## 317. PAPER2LLM++: Continual Self-Evolution of LLMs from Research Papers

**arXiv ID:** 2610.02793 | [PDF](https://arxiv.org/pdf/2610.02793v1)

**作者:** Hongji Pu `[一作]` (University of Illinois Urbana-Champaign), Wenpeng Yin `[通讯]` (Pennsylvania State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种利用科学论文中发现的缺陷证据进行持续自我进化的框架Paper2LLM++，通过验证、生成合成监督和尝试–评估–提交机制实现LLM的持续改进。

**💡 创新点**

创新点在于把研究发现视为可执行的监督信号，自动化提取证据、合成训练数据，并在更新前通过三维评估（新行为增益、旧行为退化、通用能力退化）实现安全且高效的持续自我进化。

**🔧 技术方法**

采用GPT‑OSS‑120B进行结构化抽取，Qwen2.5‑32B‑Instruct生成合成数据，LoRA微调，试验–评估–提交策略、重放缓冲和保护探针以防止遗忘。

**📊 数据集**

使用30篇失败论文对应的基准（如GSM‑Plus、HumanEval+、MATH‑Perturb、LiveCodeBench等）以及通用能力评估集HellaSwag、WinoGrande、PIQA、BoolQ、ARC‑Challenge。

**📈 对比分析**

与Text‑To‑LoRA、ELDER、RLEdit、UltraEdit、StableEdit等方法对比，在单论文和连续流场景下，Paper2LLM++平均提升+7分，保留率>80%，遗忘率低，通用能力损失≤0.5点，表现最优。

**⚠️ 局限性**

局限性包括：仅能处理可转化为输入‑输出监督的发现，对检索、规划、交互等复杂能力的论文效果有限；需要持续获得新论文，后续更新仍可能覆盖先前改进。

---

## 318. LOCUS: Landmark-Oriented Container Discrimination Using Spatial Graphs

**arXiv ID:** 2610.02803 | [PDF](https://arxiv.org/pdf/2610.02803v1)

**作者:** Taylor Bergeron `[一作]` (Worcester Polytechnic Institute), Kevin Leahy `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 LOCUS 方法，利用基于空间图的 GNN 与 CLIP 嵌入，结合正式本体论，对同一类容器的实例进行区分，从而实现机器人在未知环境中定位并检索被隐藏的物体。

**💡 创新点**

创新点包括：①将正式本体论用于节点角色（容器、标志物、双重角色）划分，使模型能泛化到新场景；②使用 APPNP GNN 在场景图中传播空间与语义上下文，更新 CLIP 嵌入，实现实例级区分；③采用多模态特征融合（视觉 CLIP、文本 CLIP、全景 CLIP、高度层级、软标志物距离）与双重对比损失（InfoNCE + 搜索排序损失），提升搜索效率。

**🔧 技术方法**

核心技术包括：CLIP 视觉与文本嵌入、3D 位置编码、软标志物距离特征、共享残差编码器、APPNP 结构化消息传递、对比学习与排序损失、以及与本体知识融合的图结构构建。

**📊 数据集**

使用的主要数据集：iTHOR 120 场景（训练/验证/测试 80/20/20），通过手工设计的放置策略生成 42 类查询对象的实例级标注；RoboCasa 10 布局 × 3 样式 × 5 种子，作为跨模拟器零样本测试；真实厨房实验中使用 Detic 检测得到的 3D 场景图，配合 Hello Robot Stretch 3 进行硬件验证。

**📈 对比分析**

与随机、CLIP、TidyBot、LLM 等基线对比，LOCUS 在实例判别准确率（IRA）和期望打开容器数（ECO/N）上显著优于所有基线；在 iTHOR 上实现 24% 的搜索成本提升，在 RoboCasa 上零样本转移仍保持领先；在真实厨房实验中，LOCUS 与 Visual/文本 CLIP 的 ECO 仅相差 1-2 个容器，证明跨域鲁棒性。

**⚠️ 局限性**

局限性包括：①依赖完整场景图，需先进行全局探索；②在标志物稀缺的房间（如客厅）空间信息不足导致性能下降；③放置策略对真实家庭的覆盖度有限，可能不完全反映多样化的家居布局；④对检测误差和位姿估计不稳的真实感知仍存在一定鲁棒性挑战。

---

## 319. TRAC: Trajectory-aware Reuse and Adaptive Correction for Efficient Autoregressive Video Generation

**arXiv ID:** 2610.02779 | [PDF](https://arxiv.org/pdf/2610.02779v1)

**作者:** Jiaxing Song `[一作]` (Hainan University), Yunshan Zhong `[通讯]` (Hainan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种训练自由的TRAC框架，用于加速自回归视频生成；

**💡 创新点**

设计了鲁棒累计调度（RCS）、自回归轨迹感知指导调度（ATGS）以及频谱结构修正（SSC）三大创新模块，以兼顾加速与质量；

**🔧 技术方法**

利用缓存重用、分类器无关指导（CFG）重用、动态规划优化以及低频频谱补偿技术；

**📊 数据集**

在SkyReels‑V2和FramePack‑F1两套基准模型上，使用VBench、VBench++等评测数据集；

**📈 对比分析**

与TeaCache、FlowCache、MotionCache、FreqForcing、FasterCache、MeanCache等方法对比，TRAC在SkyReels‑V2上实现约6.03×速度提升，仅降低0.6% VBench分数；在FramePack‑F1上获得1.67×速度提升且保持最高Quality分数；

**⚠️ 局限性**

对新兴自回归模型的泛化性尚未验证，且TRAC可与量化、蒸馏、稀疏注意等技术进一步集成以获取更多加速。

---

## 320. ROUTEAUDIT: Interaction-Aware Identification for Budgeted Multi-Verifier Routing

**arXiv ID:** 2610.02808 | [PDF](https://arxiv.org/pdf/2610.02808v1)

**作者:** Miaobo Hu `[一作]` (University of Chinese Academy of Sciences), Jun Xiao `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于契约条件的路由识别框架 RouteAudit，对多验证器系统的路由策略进行契约化审计，并给出了路由对比、验证器集对比和naive对比三种度量。

**💡 创新点**

创新点包括：将路由问题转化为契约条件下的识别问题；引入契约格子与路径敏感性分析；使用响应带实现顺序识别；在不完全匹配下给出尖锐的部分识别区间；提供机器可验证的识别证书。

**🔧 技术方法**

技术方法：契约格子与Shapley平均、响应带、部分识别区间、学习路由器、RLVR策略、离线强化学习、Bootstrap、McNemar、Wilson区间等统计检验和实验设计。

**📊 数据集**

数据集：原始缓存（raw-tail）记录与 Exact‑answer reasoning benchmark，Qwen2.5‑7B/32B 生成器对话；训练/校准/测试集分别为 4096/256/1319 条记录；附加 64 条校准记录与 128/32 条 held‑out 缓存。

**📈 对比分析**

比较方法：使用匹配控制（static、cascade、calibrated cascade、learned router、RLVR router）在同一契约下评估任务质量、调用次数、成本、p95 延迟，并报告与匹配 static 的差值与 95% 区间。实验表明 RLVR 在匹配 static 基础上提升质量 +0.0068、平均调用 1.4466、成本 0.680；learned router 在质量上略低，调用 1.4936；Raw‑tail 控制揭示验证器集对质量的显著贡献。

**⚠️ 局限性**

局限性：识别仅在请求支持、验证器目录、可用性、计费和在线过滤等契约条件满足时成立；实验使用短期路由窗口，契约交互和路径敏感值仅适用于测量的实验格子；请求配对与再训练带来不确定性；RLVR 仍需延迟验证，结果受实现细节影响。

---

## 321. LatticeSMC: Where to Spend Inference-Time Compute in Chunked Sequence Generators

**arXiv ID:** 2610.02774 | [PDF](https://arxiv.org/pdf/2610.02774v1)

**作者:** Xuanchen Wang `[一作]` (University of Sydney), Weidong Cai `[通讯]` (University of Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在块级生成器上进行推理时间奖励引导的统一框架；

**💡 创新点**

引入了二维格点Feynman-Kac模型，并给出了两条推导公式，明确奖励在块轴和去噪轴上的权重相同，且终端奖励可用前缀得分实现精确的twist；

**🔧 技术方法**

使用了分子蒙特卡洛（SMC）、有效样本数（ESS）门控重采样、密集与边界两种潜在调度以及温度调节的奖励倾斜；

**📊 数据集**

在AIST++舞蹈数据集的音乐转舞蹈扩散模型和40秒文本生成音乐的flow模型上进行实验；

**📈 对比分析**

与best-of-N、块剪枝、学习twist的SMC、Feynman-Kac密集调度等方法在相同推理预算下对比，实验显示其在节拍对齐、重复奖励、提示一致性和主题重复等四个奖励上均超过或匹配对手，且人类评测也更受欢迎；

**⚠️ 局限性**

局限包括仅在舞蹈模型中进行多时长实验、模型规模有限、只评测两种奖励、未与梯度引导方法比较，以及未给出有限粒子行为的理论上界。

---

## 322. Improving Atomic-Fact Recall via Focused Views in Unstructured Knowledge Editing

**arXiv ID:** 2610.02772 | [PDF](https://arxiv.org/pdf/2610.02772v1)

**作者:** Ding Wu `[一作]` (Georgia Institute of Technology), Tianci Liu `[通讯]` (University of Tennessee)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

改进了无结构知识编辑（UKE）中对单条事实的记忆能力，提出通过随机扰动RoPE键位构造聚焦视角来消除编辑时的上下文依赖；

**💡 创新点**

创新点在于将位置编码扰动与编辑目标结合，形成无模型参数干预的“FOCUSED”框架，能够在保持原有位置编码的前提下，使编辑过程更专注于事实本身；

**🔧 技术方法**

使用了RoPE（Rotary Position Embedding）键位随机扰动、基于句子级别的损失重构、两阶段LTE写入策略，以及与多种直接优化和Locate‑then‑Edit编辑器的无缝集成；

**📊 数据集**

使用了UnFine三子集（UnFine‑UnKE、UnFine‑CF、UnFine‑MQ）作为评测数据集，基于Qwen2.5-7B-Instruct和Llama‑3.1‑8B‑Instruct两大LLM；

**📈 对比分析**

在五种UKE编辑器（FT‑M、LoRA、COIN、μKE、UnKE）上与原始编辑器进行对比，FOCUSED平均提升了约10%‑12%（Atomic ROUGE‑L）且不影响整体句子回忆，整体性能显著优于基线；

**⚠️ 局限性**

局限性包括：需要额外的扰动超参（σ）调优；实验仅覆盖两大LLM和有限的编辑器；未能处理多条相互关联或持续演化的事实更新；

---

## 323. How Far Back Should a Transformer Look? Repetition and Copying in Music Sequence Models

**arXiv ID:** 2610.02837 | [PDF](https://arxiv.org/pdf/2610.02837v1)

**作者:** Amir Fathi `[一作]` `[通讯]`, Amir Fathi

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了Transformer模型在音乐序列预测中对不同上下文长度（token和latent）的影响，并通过干预实验验证长上下文依赖的因果机制。

**💡 创新点**

提出了通过裁剪或替换历史记忆来干预模型，以直接评估模型对早期重复信息的利用，从而揭示长上下文性能提升的可检验机制。

**🔧 技术方法**

采用了自回归Transformer，基于token和潜在表征的预测器，结合显式的记忆检索与控制干预实验，并构造了复制基线进行比较。

**📊 数据集**

在四个音乐数据集上评估：Nottingham、O'Neill's folk tunes、MusicNet和MAESTRO。

**📈 对比分析**

将模型在不同上下文长度的NLL对比，并与复制基线和独立训练的短上下文模型对照，结果显示长上下文能显著降低NLL，复制基线能部分解释提升但未完全覆盖。

**⚠️ 局限性**

实验受限于训练规模、干预的随机性以及仅聚焦于精确重复，未探讨更复杂的音乐结构或跨曲调的一般化能力。

---

## 324. All Work And No Play Makes Jack a Dull Boy: Understanding and Preventing Catastrophic Strategy Collapse in RLVR

**arXiv ID:** 2610.02835 | [PDF](https://arxiv.org/pdf/2610.02835v1)

**作者:** Qiyuan Huang `[一作]` (Peking University), Meng Li `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了大语言模型在使用可验证奖励的强化学习(RLVR)训练后的崩溃机制，并基于策略级理论提出Mesh Learning框架来防止此崩溃。

**💡 创新点**

创新点在于：①以策略级视角阐释RLVR竞争导致策略集中与所需策略容量冲突，形成崩溃的根本机制；②设计轻量级的Mirrored Entanglement Index(MEI)作为在线预警指标；③提出Mesh Learning（Coach Prompting + Strategy‑Balancing Regularization）实现多策略保持与均衡。

**🔧 技术方法**

使用了策略级理论分析、信息理论推导、梯度相互作用模型、MEI在线监测、Coach Prompting、策略平衡正则化、以及GRPO/DAPO/GSPO等RLVR算法。

**📊 数据集**

实验数据集涵盖 AIME26/25、MATH‑500、GPQA‑Diamond、LiveCodeBench v6 以及 Phi‑4‑mini‑reasoning 等多任务多模型。

**📈 对比分析**

在与 GRPO、DAPO、GSPO 及 KL/JS 正则化基线的对比中，Mesh Learning 在 Qwen3‑4B 与 Qwen2.5‑7B‑Instruct 上均提升 1.6–13.4pp，保持 MEI 低于阈值并成功避免后期准确率崩溃。

**⚠️ 局限性**

局限性包括：需预先离线生成策略前缀，m（策略数）选择影响效果；理论假设如学习率、轨迹长度等需满足一定范围；未在更大规模模型或更长训练周期下进行验证。

---

## 325. Clinical Concept Centers in LLMs

**arXiv ID:** 2610.02829 | [PDF](https://arxiv.org/pdf/2610.02829v1)

**作者:** Aishik Nagar `[一作]` (National University of Singapore), Stefan Winkler `[通讯]` (National University of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了大语言模型在临床决策中的潜在“临床概念中心”，并证明其可定位、特异且因果影响模型输出。

**💡 创新点**

创新点在于：①使用 ICD 代码作为专家标注的语料库构建稀疏自编码器（SAE）字典，定位可解释的临床概念向量；②在这些向量上进行因果干预，验证其对模型行为的直接影响；③将概念中心用于评估、性能提升和临床医生偏好预测。

**🔧 技术方法**

技术包括：稀疏自编码器（SAE）字典学习、差分均值方向构造、向量注入干预、对抗性角色基调实验、生成过程中的概念中心监测、基于概念中心的模型调节，以及对抗实验中的多重对照。

**📊 数据集**

数据集为 MIMIC-IV 出院摘要（约 8,617 条记录，按 ICD-9 章节标签）和 HealthBench 对话数据，用于临床医生评估。

**📈 对比分析**

对照方法包括：随机方向、错误章节方向、无嵌入方向等四种干预对照，评估对 ICD 章节预测的影响；在生成任务中对比概念中心注入、随机注入和错误注入。性能方面，概念中心注入在 11 种模型上平均提升 1.1–2.3 例正确诊断/100 例，且在角色基调实验中提升 2.9–11.2 分，显著优于随机或错误注入。

**⚠️ 局限性**

局限性包括：仅使用 ICD 9 章节作为概念标签，可能忽略更细粒度或跨章节关联；实验仅在开源权重模型上进行，未验证在经过多步微调或商业化模型上的适用性；对概念中心的解释性依赖于稀疏自编码器字典的质量，且未彻底探索不同词典方法的差异。

---

## 326. TerrainForge: Physics-Grounded road geometry Editing for Counterfactual Autonomous Driving

**arXiv ID:** 2610.02825 | [PDF](https://arxiv.org/pdf/2610.02825v1)

**作者:** Yang Chen `[一作]` (Rochester Institute of Technology), Zilin Bian `[通讯]` (Rochester Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于物理的对已重建多车驾驶场景进行道路几何和路面条件编辑的框架TerrainForge，能够同步产生车辆运动、相机视角和车辆间间隙的对照式后果视频。

**💡 创新点**

创新点在于将道路编辑视为统一的空间场，既驱动车辆四轮动力学，又驱动场景几何和相机轨迹，实现了道路、视觉和物理的同步，且提供了场景独立的安全效果预测代理，显著降低了多车仿真成本。

**🔧 技术方法**

使用了基于四轮耦合动力学模型、CarSim对比验证、ZOD LiDAR和Waymo Motion重建、基于梯度提升回归/分类的候选编辑筛选器以及Gaussian‑scene渲染技术。

**📊 数据集**

主要使用的数据集包括Waymo Motion（重建场景与车辆轨迹）、Zenseact Open Dataset（真实道路几何与测量）、CarSim 2022.1（物理仿真基准）和自建的15,758条道路编辑与983个制动场景的实验银行。

**📈 对比分析**

通过与CarSim、ZOD LiDAR以及Waymo原始轨迹的对比，验证了车辆动力学与道路几何的准确性；在多车实验中发现仅仿真ego车辆导致最大间隙误差超过1.5 m；代理预测终端间隙误差MAE分别为0.27 m（坡下）和0.57 m（颠簸），比无编辑预测降低22–40%，且对重大编辑的检测AP达0.85。

**⚠️ 局限性**

局限在于需依赖高质量重建与控制恢复，对极端或未覆盖的道路特征、复杂车辆交互以及非车道交通（如行人）处理尚不足；代理仅针对ego车辆的终端间隙，未能全面评估所有安全指标。

---

## 327. Text-Centric Post-Training for Omni-Modal Reasoning

**arXiv ID:** 2610.02819 | [PDF](https://arxiv.org/pdf/2610.02819v1)

**作者:** Ziyang Cheng `[一作]` (Shanghai Jiao Tong University), Yu Wang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对Omni-LLM进行后训练，探索用文本训练提升多模态推理，同时用少量音视频数据修复感知能力。

**💡 创新点**

发现感知与推理可在局部优化上实现部分解耦，提出“文本中心”后训练流程：先文本推理强化，再音视频微调。

**🔧 技术方法**

使用文本监督微调（SFT）+强化学习（GRPO）以及文本生成的合成脚本、MCQ等数据；还使用少量真实音视频样本进行RL细调。

**📊 数据集**

合成脚本/MCQ数据、真实音视频数据集（如AVHBench、MUSIC-AVQA、Video-MME-v2等）以及多模态推理评测基准（Daily-Omni、Omni-Cloze、Video-Holmes 等）。

**📈 对比分析**

与完整音视频训练、现有多模态基线对比，文本推理路径在推理宏平均提升约9.6点（+25.8% GM），且GPU时长下降56.6%；合成文本训练也能获得约21% GM提升；后续音视频细调恢复感知，保持93.5%推理收益。

**⚠️ 局限性**

文本训练导致感知下降，需要额外音视频微调；合成数据质量与覆盖度有限；对更大规模模型与更复杂任务的适用性尚未充分验证。

---

## 328. FSPO: Policy-Consistent Risk and Pareto-Feasible Control for Budgeted LLM RL Post-Training

**arXiv ID:** 2610.02828 | [PDF](https://arxiv.org/pdf/2610.02828v1)

**作者:** Miaobo Hu `[一作]` (University of Chinese Academy of Sciences), Jun Xiao `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一种名为FSPO的反馈状态控制器，用于大模型训练后期资源和风险管理。

**💡 创新点**

将政策一致的风险回顾、决策条件轨迹校准与Pareto资源连续性证书三项技术结合，实现在线资源预算约束下的自适应控制。

**🔧 技术方法**

采用拟合策略评估、交叉拟合的决策轨迹校准（DCTC）、Pareto资源前沿计算，并与GRPO训练器的匹配评估框架相结合。

**📊 数据集**

在Qwen2.5-3B-Instruct模型上使用结构修复提示、BFCL-context repair 以及JSONSchema-test 数据集进行 held‑out 与 OOD 评估，并在 24 个案例、8 区间的机制基准上验证。

**📈 对比分析**

与统一分配、阈值规则、固定反馈、适应性KL、上下文UCB、PB2 等基线在同一资源与 GPU 时长下对比，FSPO 在 held‑out 66.11%、OOD 59.43%、区间失败率 0.014，并在 GPU 预算 9.29h 时优于 PB2 与上下文UCB。

**⚠️ 局限性**

在动态成本、随机资源消耗或行动目录变化时 PRCC 需重新前沿；跨任务或更大模型迁移时需重新拟合风险与校准模型，且决策延迟随动作数增长。

---

## 329. Register-Routed Delayed Fusion: Rewiring Shortcut-Prone Observation Fusion in Visuomotor Imitation

**arXiv ID:** 2610.02813 | [PDF](https://arxiv.org/pdf/2610.02813v1)

**作者:** Jieting Long `[一作]` (University of Sydney), Weiming Zhi `[通讯]` (University of Sydney)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一种名为 Register‑Routed Delayed Fusion（RRDF）的多模态融合架构，控制视觉与紧凑感知信号的交互时序，保留行动调度。

**💡 创新点**

创新点在于采用源分区的隔离‑收集‑路由注意力调度，彻底屏蔽早期视觉‑紧凑直接注意，利用注册工作区延迟交互，从而降低捷径学习并提升鲁棒性。

**🔧 技术方法**

技术手段包括基于 Transformer 的多模态编码器、注册（latent）令牌、阶段化注意力掩码、路由表设计，以及与 ACT 框架集成的行为克隆与动作分块生成。

**📊 数据集**

实验数据涵盖了仿真任务（RoboMimic Can、Square；MimicGen Stack‑D1、Coffee‑D2；Push‑T）以及真实机器人任务（Soap Pick‑up、Plate Stacking、Toolbox），使用 Piper‑X 机械臂进行评估。

**📈 对比分析**

通过与稠密 token 融合和特征拼接等基线对比，使用成功率、完成时间以及外观/位置偏移的鲁棒性评估。RRDF 在所有任务中均与或优于稠密融合，并在外观与位置偏移测试中表现更好。

**⚠️ 局限性**

局限性包括需额外注册令牌和更复杂的路由配置，对超参数（注册数、路由延迟）敏感；在更长序列或更复杂任务中的表现尚未完全验证，也未完全消除所有捷径学习。

---

## 330. VeriPy Source-Preserving Verification and Compatibility Checking for Python Components

**arXiv ID:** 2610.02814 | [PDF](https://arxiv.org/pdf/2610.02814v1)

**作者:** Naing Oo Lwin `[一作]` `[通讯]` (Astrio Labs), Naing Oo Lwin (Astrio Labs)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实现了一套完整的工作流，将 Python 代码、注释式契约、证明侧车以及向后兼容性检测集成在一起，能够在代码更新时保持功能正确性并检测版本间的兼容性。

**💡 创新点**

创新点在于：①将源级证明、可执行模型与版本关系检查统一到同一流程；②支持两种成熟后端（Dafny 与 Lean）并提供可视化的兼容性判定报告；③通过关系产品程序实现旧版输入的兼容性验证，避免单纯功能证明的盲点。

**🔧 技术方法**

主要技术包括：Python 注释式契约（类似 ACSL/SMT）、Viper/Nagini 风格的注释、Dafny 与 Lean 代码生成、关系产品程序（self‑composition）、守护代码生成、静态类型检查（basedpyright）以及显式语义模型与依赖解析。

**📊 数据集**

实验数据集来自 7 个真实 Python 仓库（Black、Django、CPython、Werkzeug、Packaging、stdnum、PyPNG 等），共 25 个证明单元、10 个历史版本边界，覆盖 32 个函数/接口。

**📈 对比分析**

评估方法：分别使用 Dafny 与 Lean 进行功能证明，并使用关系产品程序进行兼容性验证；记录通过率、未通过原因、运行时守护开销。实验结果显示功能证明几乎全部通过；兼容性检查成功率约 40–60%，失败多因新旧输入域不匹配或未建模的异常；守护开销在大多数函数中可忽略，但在某些数据结构操作上仍显著。

**⚠️ 局限性**

局限性：①依赖可信的语义模型与前端翻译，若翻译错误会导致证明不完整；②未覆盖多态、反射、正则表达式、并发等复杂特性；③缺乏完整的端到端可机理化可靠性证明；④实验规模受限于手工注释，未体现大规模自动化能力；⑤运行时守护在性能敏感场景可能产生不可接受的开销。

---

## 331. Muon Learns Facts Better: Understanding the Role of Spectral Orthogonalization

**arXiv ID:** 2610.02798 | [PDF](https://arxiv.org/pdf/2610.02798v1)

**作者:** Xuheng Li `[一作]` (University of California, Los Angeles), Quanquan Gu `[通讯]` (University of California, Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过分析梯度流（GF）、谱梯度流（Spectral GF）和符号梯度流（Sign GF）在事实回忆任务中的连续时间动力学，研究了它们在单层线性注意力模型上的特征学习行为，揭示了谱正交化对特征分离和学习时间的影响。

**💡 创新点**

创新点在于：①在可降维的守恒流形上刻画事实回忆任务，将模型预测分解为主客体和关系两部分；②证明Spectral GF能够将主客体和关系特征的学习时间比从Θ(√{S/R})降低到Θ(1)，并消除软最大饱和导致的δ⁻¹依赖；③指出Sign GF对嵌入矩阵敏感，嵌入不同可导致特征分离、倒置或消失。

**🔧 技术方法**

主要技术包括：连续时间梯度流理论、谱正交化（Muon）与其近似、流形不变性分析、误差分解与学习时间估计、对比实验（SGD、AdamW、Muon、Spectral GF）。

**📊 数据集**

实验数据集包含：①简化的事实回忆数据（S个主客体，R个关系，答案空间SR）；②人造传记问答数据（8192人×6属性，共49152条事实）用Pythia 35M模型进行训练；③模拟实验使用线性注意力模型的梯度流和谱梯度流。

**📈 对比分析**

通过对比GF、Spectral GF、Sign GF以及常规SGD/AdamW/Muon的学习时间、学习时间比（T_S^δ/T_R^δ）与δ、S、R的关系，发现Spectral GF使学习时间比趋于1，显著缩短主客体特征学习时间；在真实传记任务中，Muon与Spectral GF提前收敛主客体误差，整体误差下降速度快于SGD和AdamW。

**⚠️ 局限性**

局限性：实验基于理想化假设（单头线性注意力、完整梯度、连续时间、固定嵌入正交），不考虑噪声梯度、非线性层、真实多头注意力及其参数维度；未证明在更一般模型或数据分布下结果保持。

---

## 332. Scaling Trajectories for Complex Tasks through Recursive Self-Rewrite

**arXiv ID:** 2610.02826 | [PDF](https://arxiv.org/pdf/2610.02826v1)

**作者:** Zongxia Li `[一作]` (Tencent HY LLM Frontier), LeoweiLiang `[通讯]` (Tencent HY LLM Frontier)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种通过多种专用执行 harness 收集成功轨迹，并使用同一模型在规划、批评、执行三阶段将其重写为通用 harness 下的可验证演示，进而进行监督微调，从而提升模型在多种终端任务上的通用性能。

**💡 创新点**

创新点在于将多种 harness 的经验统一重写为通用 harness 的演示，解决了跨 harness 的分布不匹配问题，并利用同一模型在三角色（planner、critic、executor）中完成轨迹重写，实现了无需参数更新即可提升弱模型的通用能力。

**🔧 技术方法**

核心技术包括基于 Qwen-3.8-27B 的自回归生成模型、三阶段重写流程（规划→批评→执行）、自动化漏点检测与过滤、以及在通用 harness Terminus 2 下重新执行验证。

**📊 数据集**

使用了内部构建的约 3K 终端任务池（SWR 约 2.5K+ RST 420 题），通过 Terminus 2、StateM 与自定义 Terminus 2– 这三种 harness 收集约 2K 成功轨迹，并在此基础上生成约 10K 经过验证的重写轨迹。

**📈 对比分析**

在 Terminal‑Bench 2/3/4、Terminal‑Bench Hard、Long‑Horizon Terminal Bench 以及自建 Software Terminal 100 等 5 个基准上，重写后模型在 pass@3 和平均每跑通关率上相较直接 SFT 提升 20.8%~7.0%，在 LHTB 的 process reward 从 0.25 提升到 0.29，整体性能显著优于基线。

**⚠️ 局限性**

限制包括：重写过程依赖同一模型的多角色推理，可能在规模更大或更复杂的任务上产生长序列或循环行为；重写后仍需大量计算资源；以及对低资源或小模型的适用性尚未充分验证。

---

## 333. Modeling Shared and Individual Structure for Cross-Subject Continuous Affect Regression from EEG-fNIRS

**arXiv ID:** 2610.02796 | [PDF](https://arxiv.org/pdf/2610.02796v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 334. MLCommons Jailbreak Benchmark v1.0

**arXiv ID:** 2610.02827 | [PDF](https://arxiv.org/pdf/2610.02827v1)

**作者:** Carsten Maple `[一作]` (MLCommons), Mohammed Serrhini `[通讯]` (Université Mohammed Premier Oujda)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了MLCommons Jailbreak Benchmark v1.0的完整端到端评估流程，系统性测评大语言模型在单轮文本攻击下的安全韧性。

**💡 创新点**

创新点在于整合机制优先的攻击分类、基于安全基准的Resilience Gap度量、风险校准的责任披露流程以及在评估中使用多模型LLM评判器的自动化判分体系。

**🔧 技术方法**

采用了MLCommons Jailbreak Taxonomy（基于机制的攻击分层）、AILuminate Assessment Standard v1.4、LLM-as-judge评估集合、配套的人工标注校准、以及对攻击、种子和SUT的系统化选择方法。

**📊 数据集**

使用了来自AILuminate公开Practice集的264条种子提示（每个危害类别24条），并在8个开源权重SUT（3个可访问级、5个大型模型）上执行，攻击覆盖11类（共11个攻击集），未公开具体攻击提示。

**📈 对比分析**

通过与基准安全评测（AILuminate Safety Benchmark）对比并使用Resilience Gap和攻击成功率指标，平均Resilience Gap为7.57%，最高攻击成功率在角色扮演/模板和包装/模式类中达约35%；总体上较大模型表现更稳健。

**⚠️ 局限性**

局限性包括评估器误判率高（约16%误判为安全、38%误判为违规）、受试模型数量有限、种子提示与攻击覆盖不完整、未公开具体攻击材料导致外部复现受限，以及部分模型出现负Resilience Gap的原因仍需进一步探究。

---

## 335. FastOPD: On-Policy Distillation for Lightweight VLA Deployment

**arXiv ID:** 2610.02832 | [PDF](https://arxiv.org/pdf/2610.02832v1)

**作者:** Yoojin Oh `[一作]` (KAIST), Jong Chul Ye `[通讯]` (KAIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 FastOPD，一种高效的 on‑policy 蒸馏框架，将大规模 Vision‑Language‑Action（VLA）基础模型压缩为轻量级学生模型，并通过仅在单个 on‑policy 状态上对教师速度场进行监督，结合自一致性目标实现少步推理。

**💡 创新点**

创新点在于：1) 将传统 OPD 的全步教师评估替换为单步监督，显著降低训练成本；2) 引入自一致性约束，将局部速度信息传播到任意区间流图，生成“any‑step”学生；3) 通过理论证明，FastOPD 的目标可使学生恢复与教师相同的分布，确保少步推理性能；4) 在多任务、跨领域（VLA 与 WAM）教师上均可直接使用，只需教师速度场，无需匹配架构。

**🔧 技术方法**

采用流匹配（flow matching）与流图（flow‑map）架构，结合 on‑policy 速度匹配（OPFD）和自一致性损失（SC），以及时间投影层实现时间条件化；训练使用 Adam，参数 λ 控制 OPFD 权重；在推理时实现 1~4 次 denoising 步骤即可完成动作采样。

**📊 数据集**

使用 LIBERO、RoboTwin 2.0 两大多任务仿真基准（分别包含 40 项和 50 项任务），以及真实机器人实验（使用 MolmoAct2 作为教师、SmolVLA 作为学生）进行评估。

**📈 对比分析**

与教师模型、SmolVLA 原始版本以及三种少步蒸馏基线（CTM、DMD、iMF）对比。FastOPD 在 2 步推理下能保留约 84% 的教师性能，推理时延降低 78%（相较 SmolVLA 10 步）。训练速度提升 5.7×；在 LIBERO 上 1 步成功率 84.2%，在 RoboTwin 上 1 步成功率 51.2%；在真实机器人上 4 步成功率提升 8pp。相较于基线，FastOPD 在少步 regime（1–4 步）均优于对手。

**⚠️ 局限性**

局限性包括：1) 在较多步（>4）推理时性能略低于原始学生模型；2) 对 λ 参数敏感，过小导致收敛失效；3) 仅对教师速度场进行监督，若教师难以提供高质量速度信息或两模型间差异过大，效果可能受限；4) 目前仅在固定的 VLA/WAM 架构上验证，尚未在更广泛的任务或更高维控制空间中测试。

---

## 336. SimpleTouch: Can Vision-Language-Action Models Master Contact-Rich Manipulation Without Tactile Policy Pretraining?

**arXiv ID:** 2610.02784 | [PDF](https://arxiv.org/pdf/2610.02784v1)

**作者:** Chen Yang `[一作]` (Tsinghua University), Chen Wang `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在已有的预训练视觉‑语言‑动作（VLA）模型基础上，加入一个冻结的触觉编码器并用触觉专家学习，仅凭50条演示即可实现接触丰富的操控。

**💡 创新点**

不需要额外的触觉策略预训练或视觉‑触觉对齐，只保留预训练触觉特征并通过多时延触觉预测监督触觉专家；同时利用全局与空间触觉 tokens 实现更精细的接触建模。

**🔧 技术方法**

使用冻结的 T3 触觉编码器、Transformer 结构的触觉专家、动作专家、视觉‑语言 Backbone、条件流匹配、跨模态注意力以及未来触觉多时延预测。

**📊 数据集**

UniVTAC 仿真六个任务（Lift Bottle, Pull‑out Key, Lift Can, Put Bottle, Insert Hole, Insert Tube）和四个真实机器人任务（Play Mahjong, Wipe Board, Pick Up Chips, Insert USB），每个任务提供 50 条演示。

**📈 对比分析**

与七个强基线（ACT, VITaL, UniVTAC‑ACT, Tactile‑VLA, FTP‑π0.5, FTP‑1 等）对比，平均成功率 77.5% 在 UniVTAC 上超过 FTP‑1 10.8 个百分点，在真实机器人上 71.3% 超过 FTP‑1 8.8 个百分点，表现最优。

**⚠️ 局限性**

方法在极少演示场景仍有提升，但在极其接触敏感任务（如 Insert USB）成功率仍低；对预训练触觉模型依赖较大，未在更大多样化数据上验证，且推理时仍需预训练模型，未来需探索更高效的多模态融合与实时推理。

---

## 337. Automatic Evaluation of Mental Health Stigma in Online Communication

**arXiv ID:** 2610.02775 | [PDF](https://arxiv.org/pdf/2610.02775v1)

**作者:** Naomi Baes `[一作]` (University of Melbourne), Yulia Otmakhova `[通讯]` (University of Melbourne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并发布了一个理论驱动的心理健康污名检测基准，结合二元检测与多维细粒度分类，并对自然新闻与 Reddit 语料进行标注。

**💡 创新点**

创新点在于：①首次在六种精神疾病上同时展开多维度污名分析；②提出了包含污名模式、领域及公共子因子（认知、情感、行为）的分层标签体系；③通过明确的操作规则与示例大幅提升人工与模型判断的一致性。

**🔧 技术方法**

技术上使用了大语言模型（Claude Sonnet 4.6、GPT‑5.5、Gemini 3 Flash、DeepSeek V4 Flash）与LoRA微调的 Qwen2.5‑7B 分类器，结合多种提示（定义、示例、规则）进行零样本/少样本推理；同时对毒性、仇恨语、情感等邻近任务进行比较。

**📊 数据集**

数据集由 470 篇文本组成，来源于 News on the Web（新闻）与 Reddit，覆盖 ADHD、酗酒、自闭症、BPD、抑郁、精神分裂症六类精神健康状况。

**📈 对比分析**

对比实验表明：在未加入规则的基线提示下，LLM 召回率高但误报率大；加入操作规则后精确率显著提升，整体准确率与 F1 亦有提升；相较于情感/毒性/仇恨语模型，LLM 在检测污名时更易过度预测，且性能在不同疾病与文本难度上差异显著。

**⚠️ 局限性**

局限性包括：标注分布受样本筛选影响；未纳入有精神健康经历者视角；文本仅限英语新闻与 Reddit，缺乏其他媒介与多语言；标注规模有限，细粒度类别稀疏；模型在特定疾病（如精神分裂症）易过度预测，需更细致的鲁棒性验证。

---

## 338. VIGOR: Zero-Shot Visual Generalization via Latent-Space Consistency in Model-Based Reinforcement Learning

**arXiv ID:** 2610.02801 | [PDF](https://arxiv.org/pdf/2610.02801v1)

**作者:** Mingyu Park `[一作]` (KAIST), Donghwan Lee `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 VIGOR 框架，通过弱到强的异步增强、动力学一致性回归与编码器稳定化三项技术实现模型基强化学习在视觉扰动下的零射击泛化。

**💡 创新点**

创新点在于识别 MBRL 的两级脆弱性，并提出异步批量弱到强增强、动力学一致性（直接在潜空间回归）与编码器稳定化协同的三元机制，使得模型既保持样本效率，又实现与增强方式无关的稳健泛化。

**🔧 技术方法**

使用了异步弱到强增强（AW）、动力学一致性损失（DC）、编码器稳定化损失（ES）以及 TD‑MPC2 作为骨干；此外还使用了随机 overlay/conv 作为强增强、U‑MAP 可视化与自监督对比等辅助技术。

**📊 数据集**

在 DeepMind Control Suite 与 Robosuite 两大连续控制数据集上，按 RL‑ViGen 零射击协议评估，覆盖七种视觉扰动（背景颜色/视频、光照、摄像头位置等）。

**📈 对比分析**

与八个基线（4 MFRL + 4 MBRL）对比，VIGOR 在 DMC 上平均提升 3.4%（IQM 2.6%），Robosuite 上平均提升 43.6%（IQM 16.2%）；在样本效率上与 TD‑MPC2 轨迹几乎持平，并在 100K 预算下仍保持领先。

**⚠️ 局限性**

局限性主要在于仅针对外观扰动，几何变换（如摄像头旋转）仍表现不佳；未涵盖动力学或任务分布迁移，也未在真实机器人上进行验证。

---

## 339. Frequency Is Not Sensitivity Identifying Safety-Sensitive Experts in Sparse MoE LLM

**arXiv ID:** 2610.02910 | [PDF](https://arxiv.org/pdf/2610.02910v1)

**作者:** Md Nurul Absar Siddiky `[一作]` (University of Hawaii at Manoa), Yingfei Dong `[通讯]` (University of Hawaii at Manoa)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在五种稀疏 Mixture‑of‑Experts (MoE) 语言模型上，通过抑制专家来评估对安全拒绝行为的影响，比较激活频率与路由梯度敏感度两种专家重要性指标。

**💡 创新点**

首次将路由梯度敏感度作为专家抑制的选择信号，在多模型、多预算下证明其比传统激活频率更能降低拒绝率，揭示模型安全性对专家梯度的显著依赖。

**🔧 技术方法**

使用梯度计算、激活统计、专家抑制（负路由偏置）以及对比实验（相同专家数、相同路由流量、层级匹配控制）来量化抑制效果。

**📊 数据集**

采用 AEGIS2.0（正面）与 AdvBench（恶意）提示集做分析，随后用两个独立的 AEGIS2.0 集合评估行为，实验共使用 500 正/500 恶提示用于分析，100 正/100 恶提示用于评估。

**📈 对比分析**

与随机抑制和激活频率抑制做对比；梯度抑制在 24/25 条件下降低拒绝率，最高可达 OLMoE 模型 73.5% 的相对下降；激活抑制往往无效甚至提升拒绝；层级匹配实验中梯度优势仍保留在 23/25 条件。

**⚠️ 局限性**

局限包括仅测试 5 个模型、少量提示集、未评估对生成质量或指令遵循的影响；梯度计算成本高；在更大抑制预算时效果非单调，机制尚未解释。

---

## 340. Law And Order: Tax Law Autoformalization

**arXiv ID:** 2610.02792 | [PDF](https://arxiv.org/pdf/2610.02792v1)

**作者:** Sophia Simeng Han `[一作]`, Michael Genesereth `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种神经‑符号框架，自动将美国IRS税表及其说明书转化为可执行的符号程序，并通过单元级验证与局部修复循环，实现对税务计算的高精度符号化；

**💡 创新点**

创新点在于构建结构对应与函数对应的法律‑逻辑双向映射；利用LLM进行程序合成、单元级符号执行验证，并采用局部修复策略在大规模税表上实现精确、可验证的自动化；

**🔧 技术方法**

使用的技术包括大语言模型（如GPT‑6、Claude Opus、Claude Fable、Gemini、Kimi K3、Qwen）进行程序合成；Lean 4 代码生成与符号执行；单元级输出对比与差异定位；局部修复循环；以及 OpenTaxSolver 的人类编写税单作为修复集；

**📊 数据集**

数据集主要有：OpenTaxSolver（人类编写的税单，用作修复集）以及 TaxCalcBench（51份由CPA编写的独立税单，用于最终评估），覆盖30个税表；

**📈 对比分析**

评估方式是将模型在修复集上进行迭代修复后，在未见的 TaxCalcBench 集上测试；相较于单纯的LLM翻译、无修复或全局自调试，使用局部反馈的模型在修复集上实现100% cell/表格准确率，在Hold‑out集上最高模型从66%提升到100%，显示出显著性能提升；

**⚠️ 局限性**

局限性包括：对低能力LLM（如Qwen 3.6 8B）效果有限；需人工提供修复集，缺乏完全自动化；仅针对美国联邦税表，跨司法或语言迁移尚待研究；以及对非常复杂的交叉引用和法规变更的适应性仍有待提升。

---

## 341. Do ResNets Route? Sparse Interaction Experts in Residual Networks

**arXiv ID:** 2610.02907 | [PDF](https://arxiv.org/pdf/2610.02907v1)

**作者:** Liang Yan `[一作]` (Fudan University), Mu Miao `[通讯]` (Datacanvas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过对训练好的ResNet进行布尔莫比乌斯反演，将网络输出精确拆分为单个残差修正及其更高阶交互项，并分析这些交互项在不同输入、类别和残差规模下的分布与稀疏性，揭示标准ResNet在功能上实现了隐式软路由。

**💡 创新点**

创新点在于：①提出用残差分支掩码视角对ResNet进行布尔函数建模并精确逆变换，得到完整的交互系数；②证明在残差缩放小的情况下，k阶交互项按O(λ^k)衰减，解释残差网络的低阶偏置；③通过大规模枚举掩码实验发现，尽管网络是全密集执行的，但功能贡献高度稀疏、输入相关，且主导交互与类别紧密相关，形成隐式软Mixture‑of‑Experts结构。

**🔧 技术方法**

主要技术包括：布尔莫比乌斯反演、残差分支掩码实验、交互能量谱统计、残差缩放分析、Jaccard相似度与Spearman相关性评估、以及对ImageNet预训练ResNet-18/34的全枚举掩码评估。

**📊 数据集**

使用的数据集为ImageNet 2012的验证集（10,000张图像），并对ResNet-18（8个残差分支）和ResNet-34（16个残差分支）进行全掩码枚举。

**📈 对比分析**

与传统的路径计数或平均掩码输出方法对比，本文的莫比乌斯重构能够精确恢复完整logits（误差≈10^-7），而朴素方法误差极大；在交互能量上，ResNet-18与ResNet-34的主导阶次分别为5和10，低阶交互仅占约15%；残差缩放能显著降低交互阶次，但会大幅改变模型预测（top‑1准确率下降至≈0.05）。

**⚠️ 局限性**

局限性包括：①仅在相对浅层ResNet（18/34）上验证，深层网络需要更高效的交互估计方法；②实验仅对ImageNet验证集进行，缺乏对其他任务或数据集的泛化验证；③虽然揭示了隐式软路由，但对实际加速或模型压缩的实用性尚未证明；④莫比乌斯反演需要枚举所有掩码，计算成本随层数呈指数级增长。

---

## 342. Interpreting at Write Time: A Policy Ablation for Multi-Goal Agent Memory

**arXiv ID:** 2610.02897 | [PDF](https://arxiv.org/pdf/2610.02897v1)

**作者:** Albert Sadowski `[一作]` (Warsaw University of Technology), Jarosław A. Chudziak `[通讯]` (Warsaw University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过比较三种写入策略（无目标写法、一次性写入所有目标、按目标分写）评估长运行助手的摘要质量，使用多种大语言模型在两条结构化事件流上进行实验。

**💡 创新点**

创新点在于首次系统地对比目标条件下的写入策略，证明按目标分写并联合阅读能够显著提升相关性、完整性和准确性。

**🔧 技术方法**

采用Claude、GPT‑5、Minimax等大型语言模型作为编码器与阅读器，使用LLM评审器进行自动评分，并通过Jaccard相似度、t检验等统计方法进行分析。

**📊 数据集**

使用两份结构化客户交互事件流——Meridian（5条事件、4条查询）和Helix（15条事件、6条查询）作为数据集。

**📈 对比分析**

通过对五种模型、两条流、十次重复实验得到的复合得分比较，按目标分写策略平均得分3.97，优于无目标写法3.68和一次性写法3.34，差异在统计上显著；同时无目标写法优于一次性写法。

**⚠️ 局限性**

局限性包括仅使用短小结构化流、评审器为单一LLM可能偏好较长答案、按目标分写与一次性写法同时改变写入次数与指令，未单独评估这两项因素；缺乏人工评价和动态目标变化的验证。

---

## 343. Distributionally Robust Survival Models under Subpopulation Shift and Outlier Contamination

**arXiv ID:** 2610.02868 | [PDF](https://arxiv.org/pdf/2610.02868v1)

**作者:** Seonghwi Kim `[一作]` (Pohang University of Science and Technology), Minwoo Chae `[通讯]` (Pohang University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种联合考虑亚群体分布偏移与异常值污染的分布鲁棒生存模型框架，并给出相应的交替梯度算法。

**💡 创新点**

设计了外层最小化修正名义分布以抑制异常样本，内层最大化聚焦最难亚群，兼顾非可分解Cox负偏差等非加性生存损失，并利用KKT条件推导外层梯度。

**🔧 技术方法**

混合式DRO框架、α‑CVaR约束、KKT条件梯度推导、指数梯度更新、Cox比例风险模型与神经Cox。

**📊 数据集**

METABRIC、FLC两大真实生存数据集以及仿真数据。

**📈 对比分析**

与标准Cox、f‑DRO、Exact‑Cox、Robust‑Cox、Cox‑NN等基线对比，实验显示在亚群体偏移+异常污染场景下本文方法在worst‑group C‑index最高、tail loss最低，整体C‑index保持竞争甚至超越，训练过程更稳定。

**⚠️ 局限性**

尚未同时处理亚群体内协变量偏移，扩展到更大网络/高维可能需更高计算成本；异常值处理仍需调节ε参数。

---

## 344. Permutation Robustness Is Not Enough: Action Collapse in Multi-Agent Transformer Policies

**arXiv ID:** 2610.02848 | [PDF](https://arxiv.org/pdf/2610.02848v1)

**作者:** Amit Thakur `[一作]` (University of California, Merced), Mukesh Singhal `[通讯]` (University of California, Merced)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了Transformer多智能体机器人策略在对代理顺序变化时的鲁棒性问题，指出低等价误差可能误导并提出动作崩塌诊断指标。

**💡 创新点**

创新点在于系统性将等价误差、动作多样性、同动作比例等指标组合评估，并揭示等价正则化与行为多样性之间的折衷。

**🔧 技术方法**

采用PPO/MAPPO训练框架、Transformer编码器、等价正则化、对偶策略多样性正则化以及熵奖励等技术。

**📊 数据集**

在PettingZoo/MPE的合作导航任务上进行实验，分别测试N=3和N=4两种团队规模。

**📈 对比分析**

通过返回值、等价误差、KL散度、动作不一致度以及动作多样性等指标比较，发现弱等价正则化能显著提升鲁棒性而保持一定多样性，强正则导致所有智能体趋向同一动作。

**⚠️ 局限性**

实验仅限于仿真环境、中心化演员、单一任务，未考察更大规模团队或真实机器人平台，且对多样性正则化的设计仍较简单。

---

## 345. DNAlign: Dynamic Null-Space Safe Alignment for LLMs

**arXiv ID:** 2610.02844 | [PDF](https://arxiv.org/pdf/2610.02844v1)

**作者:** Jisheng Dang `[一作]` (Lanzhou University), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了DNAlign框架，通过控制理论与空洞空间投影实现LLM的安全对齐，保持原始知识完整性。

**💡 创新点**

创新点在于将动态控制信号与针对有害子空间的投影约束相结合，使干预只作用于危险内容而不破坏中性知识。

**🔧 技术方法**

技术包括基于控制理论的离散动态系统建模、空洞空间投影、轻量级价值函数训练及梯度上升优化控制信号。

**📊 数据集**

实验使用Vicuna‑7B/Qwen‑7B基础模型，在HH‑RLHF、SHP、RealToxicityPrompts、ToxiGen、Categorical‑HarmfulQA以及六大无害数据集进行评估。

**📈 对比分析**

与未编辑模型、Static RE、Contrastive Decoding、RE‑Control等基线对比，DNAlign在安全奖励、综合质量、连贯度等指标上均表现更优，同时保持多样性不受影响。

**⚠️ 局限性**

局限性包括对投影矩阵样本量和控制信号迭代次数的敏感性，初始推理延迟略高，并且在极端高毒性场景下仍可能出现轻微质量下降。

---

## 346. Turnover-Orthogonal Credit Assignment for Open-Team Multi-Agent Reinforcement Learning

**arXiv ID:** 2610.02847 | [PDF](https://arxiv.org/pdf/2610.02847v1)

**作者:** Amit Thakur `[一作]` (University of California Merced), Mukesh Singhal `[通讯]` (University of California Merced)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种用于开放式多智能体强化学习的信用分解框架（TOCA），通过事件条件下的价值分解将团队收益分为动作效应、纯换手效应和动作–换手交互效应，进而实现对换手事件的归因与排除。

**💡 创新点**

创新点在于：①提出了事件条件下的中心化基线，使得纯换手效应被剔除；②保留并量化动作–换手交互信用；③通过软权重β引入TOCA‑β，降低高方差控制环境中的噪声；④将上述分解实现为可扩展的、置换不变的集编码器。

**🔧 技术方法**

主要技术包括：置换不变的集编码器（Deep Sets/Set Transformer），四头值分解（μ、q_A、q_E、q_AE），基于事件条件的中心化基线，PPO式Actor-Critic训练，中心化正则化约束，软交互权重TOCA‑β，计数式信用分解。

**📊 数据集**

实验数据集：①诊断开放式团队环境（具备可观测的 q_A、q_E、q_AE 基准），②基于粒子导航的 Dynamic Spread 替换式基准（四名代理覆盖地标，随机高能力被低能力代理替换）。

**📈 对比分析**

与 MAPPO、Event‑Set‑MAPPO 等基线进行比较。TOCA 在诊断环境中平均回报提升至 341，且分解误差低；去交互项性能显著下降。Dynamic Spread 中，TOCA‑β 在平均回报上与 Event‑Set‑MAPPO 相当，在高换手率下高事件回报最高，优于无交互版本。

**⚠️ 局限性**

局限性：仅针对外生换手事件设计，若换手受代理动作驱动则需显式因果建模；交互信用在噪声较大的控制任务中可能导致方差增加；Dynamic Spread 实验表现为趋势性提升，未能在所有基准上显著压倒现有方法。

---

## 347. Seeing, Saying, but Not Using: From Reportable Spatial Facts to Usable States in Multimodal Large Language Models

**arXiv ID:** 2610.02876 | [PDF](https://arxiv.org/pdf/2610.02876v1)

**作者:** Jinchang Zhang `[一作]` (Indiana University Bloomington), Guoyu Lu `[通讯]` (Indiana University Bloomington)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出SpaceConflict基准，构造四层空间推理任务并探究模型空间事实可用性与可利用性的差距，随后提出Operational State Supervision（OSS）对齐并监督状态轨迹以提升多模态模型的空间状态使用能力。

**💡 创新点**

创新点在于：①将空间推理拆分为L1~L4四个结构化层级，①揭示“可用-可利用”行为差距；②提出OSS，通过跨上下文状态对齐和轨迹监督直接提升模型对空间状态的组织与使用，显著提高了L3、L4的PairAcc。

**🔧 技术方法**

采用统一的Supported/Contradictory/Unknown判定接口；将空间事实映射为可序列化的元组并以自回归语言模型训练；利用可视化诊断实验（直接状态、完整变换、显式状态三种条件）评估行为差距；OSS训练集由原始答案、状态对齐样本与轨迹样本组成。

**📊 数据集**

数据来源于现有多模态空间数据集（如VSI-Bench、Spatial457、SPAR-Bench等），通过确定性适配器提取可验证空间事实并构造23,196个三元组实例，涵盖四个推理层级。

**📈 对比分析**

使用多模态LLM（Qwen3.5-4B/9B/27B、GPT‑5、Claude等）在L1–L4任务上进行ClaimAcc/PairAcc评估；OSS将PairAcc从约48%提升至96%（L3）及从62%提升至68%（L4），整体ClaimAcc/PairAcc分别从56%/44%提升至58%/86%；与答案SFT、CoT‑SFT等基线对比，OSS展现显著优势。

**⚠️ 局限性**

限制在于：①可用-可利用差距尚未完全消除，OSS只能改善部分实例；②评测聚焦于静态视觉+文本场景，未覆盖动态交互或真实物理环境；③结果受模型规模、prompt设计和解码策略影响，未能在所有多模态架构上验证通用性。

---

## 348. Containing the Autonomous Operator: A Defense-in-Depth Framework and Reference Architecture for Securing AI Agents on Kubernetes

**arXiv ID:** 2610.02861 | [PDF](https://arxiv.org/pdf/2610.02861v1)

**作者:** Simhadri Podala Narasimha `[一作]` `[通讯]` (Independent Researcher), Simhadri Podala Narasimha (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了在 Kubernetes 上保护 LLM 代理的完整安全框架：威胁模型、九条设计原则、七层防御架构与治理层，并给出跨 AWS EKS、Azure AKS 与 GKE 的参考部署方案。

**💡 创新点**

创新点在于：① 把传统云原生安全与代理特有的工具边界、输入注入、供应链攻击结合，形成七层技术 + 管理治理的安全体系；② 明确“受限自治”与审批治理，将安全控制与人工审核、预算等策略整合；③ 通过多云原生机制（RBAC、VAP、网络策略、gVisor/Kata、eBPF、OPA/Cedar、SPIFFE 等）实现对代理的最小权限与完整中介；④ 提供可迁移至主流托管 Kubernetes 服务的参考架构。

**🔧 技术方法**

采用的技术包括：Kubernetes 原生安全（ServiceAccount、RBAC、ValidatingAdmissionPolicy、Pod Security、网络策略）、SIG Apps Agent Sandbox + gVisor/Kata、eBPF 运行时检测（Tetragon/Falco）、OPA/Cedar 策略引擎、SPIFFE/SPIRE、工作负载身份（Pod Identity/IRSA、Entra Workload ID、Workload Identity Federation）、Cilium FQDN 网络策略、GitOps（Argo CD/Flux）、Model Context Protocol（MCP）与 Agent2Agent（A2A）协议、OpenTelemetry Tracing、模型端点私有接入（PrivateLink、PrivateEndpoint、Private Service Connect）。

**📊 数据集**

本论文未使用公开数据集，而是通过定性评估、攻击演练和威胁–控制覆盖矩阵来验证框架；实验部分建议在未来工作中构建基于 AgentDojo 的 Kubernetes 注入基准。

**📈 对比分析**

论文未给出量化性能指标；提出的实验方法包括：① 逐层添加控制的攻击成功率评估；② 单层失效验证；③ 延迟与资源消耗测量（gateway 处理延迟、沙盒获取时间、gVisor/Kata 性能）；④ 运营负担评估（审批量/时延）。

**⚠️ 局限性**

limitations: 1) 仅为框架与设计研究，缺乏实测攻击成功率与性能数据；2) 依赖多第三方组件（AgentSandbox、OPA、Cilium 等），若组件更新需同步维护；3) 对模型层抗攻击的保障有限，仍需配合模型级防御；4) 审批与自治治理可能导致审批瓶颈；5) 未覆盖模型训练期攻击、模型输出安全等场景；6) 方案对不同云供应商的实现细节存在差异，需额外适配。

---

## 349. LUMOS: Tracing Parametric Knowledge from Training Data to Behavioral Outputs in LLMs

**arXiv ID:** 2610.02902 | [PDF](https://arxiv.org/pdf/2610.02902v1)

**作者:** Seoyeon Ye `[一作]` (Ewha Womans University), Hyunsoo Cho `[通讯]` (Ewha Womans University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了基于训练数据曝光与行为表现的多维知识诊断框架，分析LLM内部知识与输出之间的因果链。

**💡 创新点**

将训练曝光轴纳入评估体系，揭示记忆、检索、泛化和自我认知的四类知识状态，并提供可验证的指标。

**🔧 技术方法**

结合内部探测（线性探测器、MARS、SAPLMA）和外部指标（准确率、DC、PC、Self-Verify），利用多尺度OLMo 2模型进行实验。

**📊 数据集**

使用OLMo 2透明预训练语料库以及自构建的Seen/Unseen事实与推理问答集（Web/Academic、GSM8K等）。

**📈 对比分析**

通过与规模、提示策略、RLVR/SFT训练方式比较，发现曝光决定事实可靠性、CoT仅提升信心而不校准，推理泛化受模型容量与训练方法影响，性能在Seen上可达80%+，Unseen低于50%。

**⚠️ 局限性**

局限在于仅针对OLMo 2，缺乏跨模型验证，且对动态知识更新与更广泛任务类型的适用性仍未充分评估。

---

## 350. TACD: Distilling Efficient Text-to-Motion Models via Terminal Amplification Control

**arXiv ID:** 2610.02867 | [PDF](https://arxiv.org/pdf/2610.02867v1)

**作者:** Wei-Jin Huang `[一作]` (Sun Yat-sen University), Wei-Shi Zheng `[通讯]` (Sun Yat-sen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无真实运动数据的端到端文本到运动模型压缩方法（TACD），通过控制端点放大（terminal amplification）来提升少步生成质量。

**💡 创新点**

创新点在于识别并消除固定监督网格下的端点放大问题，将教师查询与学生步长绑定，给每个监督点设定上限，从而在保持推理步骤不变的前提下显著提升少步生成效果。

**🔧 技术方法**

采用分段 on‑policy 迁移（π‑Flow）、velocity matching、端点匹配（endpoint matching）等技术；对流模型与扩散模型均可扩展，且不需要额外的真实运动数据。

**📊 数据集**

在 HumanML3D、KIT‑ML 和 Kimodo 三个数据集上进行实验，并对比多种基准（HY‑Motion Lite、MotionLCM‑V2、MLD 等）。

**📈 对比分析**

与传统的无监督迁移、流匹配或一致性蒸馏相比，TACD 在 8 步生成中将 FID 由 2.486 降至 1.041（58% 降幅），并在 4 步端点匹配下在 HumanML3D 上实现 FID 0.222、R@3 0.818，显著优于 50 步教师模型；在 KIT‑ML 和 Kimodo 上也取得了类似的质量提升，并在推理速度上实现 7.7–11.9× 的加速、GPU 内存下降 3.8–6.7×。

**⚠️ 局限性**

局限性：仍依赖强大的教师模型，端点匹配方法仅在固定步长设置下有效，且对极大步长或更复杂运动类型的泛化能力尚待进一步验证。

---

## 351. Harness-Aware Distillation for Small Language Model Agents

**arXiv ID:** 2610.02858 | [PDF](https://arxiv.org/pdf/2610.02858v1)

**作者:** Moonseok Choi `[一作]` (KAIST AI), Juho Lee `[通讯]` (KAIST AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Harness-Aware Distillation（HAD）框架，用于在拥有固定硬件（harness）的语言模型代理中进行知识蒸馏。

**💡 创新点**

HAD 的创新点在于通过对同一教师在有无 harness 信息下的动作进行对比学习，并结合有效性过滤，显式监督学生如何利用 harness 信息，解决了传统 OPD 对 harness 依赖的三大失败。

**🔧 技术方法**

该方法采用对比偏好学习、基于 harness 的响应蒸馏、有效性过滤以及无奖励、无标签的无监督方式。

**📊 数据集**

实验使用 ALFWorld、WebShop、ScienceWorld 等长周期文本任务，评估 Qwen3、Gemma‑4 等模型。

**📈 对比分析**

与四种基线 OPD 方法（OPD、SAGE‑OPD、Guided‑OPD、SOPD）以及零射击教师/学生进行对比，HAD 在所有环境中获得最高成功率，并显著提升 harness 利用率，甚至超过 8B 教师。

**⚠️ 局限性**

实验仅限于文本基任务和 2B 以下模型，未验证更大模型或工具调用场景；harness 仍需人工设计，且 HAd 对额外教师查询量不敏感但需更多对比数据。

---

## 352. Probe the Harness: Setup Checks for Stale-Data RL Comparisons in Language Models

**arXiv ID:** 2610.02911 | [PDF](https://arxiv.org/pdf/2610.02911v1)

**作者:** Taiheng Pan `[一作]` `[通讯]` (University of Melbourne), Taiheng Pan (University of Melbourne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在语言模型的慢数据强化学习实验中，作者通过系统检查训练管线（harness）中的关键细节，重新评估方法排名。

**💡 创新点**

提出“Probe The Harness”检查表，识别并纠正导致方法排名偏差的四个层面，证明正确配置后方法表现一致。

**🔧 技术方法**

使用行为自由的自锚定正则化（SAWN）、截断重要性采样和 PPO 剪切等强化学习算法，结合 verL 框架和单 GPU 重放训练器进行实验。

**📊 数据集**

使用 GSM8K 语言模型基准数据集。

**📈 对比分析**

将 SAWN 与 GRPO 在 verL 和单 GPU 训练器上的性能对比，发现错误配置导致 SAWN 看似优于 GRPO，修正后两者在 verL 上表现相当，SAWN 在单 GPU 训练器上稳定提升。

**⚠️ 局限性**

实验仅在有限的模型、数据和训练配置下验证，缺乏对更广泛场景的普适性；需要进一步检查所有潜在管线细节以保证结果稳健。

---

## 353. To Explore The Strange New World Beyond Data Distribution: System Behavior, Causality Tax, and Non-causal Base Model

**arXiv ID:** 2610.02839 | [PDF](https://arxiv.org/pdf/2610.02839v1)

**作者:** Xianzhi Zeng `[一作]` (Nanyang Technological University), Gao Cong `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了系统行为（System Behavior）作为语言模型的第一性原理隐变量，并将其嵌入到 ELBO 中，展示了传统因果（causal）生成链的局限性（即“因果税”），随后设计了非因果的变分族 Green Shell，并通过理论推导和 NTK 频谱验证其在误差界和信噪比方面的优势。

**💡 创新点**

创新点主要包括：① 把系统行为视为不可约的隐变量，引入新的 Bayesian 视角；② 首次系统性证明因果链是子最优的（因果税）；③ 提出一种非因果的分治变分族 Green Shell，能够消除因果链带来的结构误差；④ 通过 NTK 分析给出非因果模型更紧的误差上界。

**🔧 技术方法**

技术手段：贝叶斯推理与变分推断（ELBO），隐变量建模，扰动测量（Gaussian 噪声注入）、控制与评估，Neural Tangent Kernel (NTK) 分析，理论误差边界推导。

**📊 数据集**

实验中未使用公开自然语言数据集，主要基于理论模型与仿真实验（NTK 频谱、噪声扰动实验）进行验证。

**📈 对比分析**

通过对比因果与非因果模型的 NTK 频谱，发现 Green Shell 在懒训练（lazy‑training）阶段误差上界更紧，信噪比提升 7 dB 以上，并在后期训练中实现了多尺度拟合提升约 20%。

**⚠️ 局限性**

局限性：① 结果主要为理论与 NTK 级别验证，缺乏完整的端到端实测；② 依赖 UAT 与极值理论，适用范围有限；③ 非因果实例的行为在部分细节上仍不够透明；④ 未讨论在真实大型模型与数据集上的性能与可扩展性；⑤ 可能存在被恶意利用的风险。

---

## 354. ConvoDrift: A Multi-Turn Conversational Dataset for Modeling Stylistic Tone Evolution

**arXiv ID:** 2610.02873 | [PDF](https://arxiv.org/pdf/2610.02873v1)

**作者:** Vihindi Kotalawala `[一作]` (Informatics Institute of Technology), Prasan Yapa `[通讯]` (University of Luxembourg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

创建了 ConvoDrift 数据集，专注于多轮对话中的语调漂移和风格变化，并通过语义保持的方式生成风格对比的对齐样本；同时构建了基于五种人设的偏好对标数据。

**💡 创新点**

创新点在于：① 明确分离语义与风格，捕捉对话中逐步的语调漂移；② 通过统一的对话结构同时生成多轮漂移轨迹和对齐对；③ 引入人设条件和多样化偏好，支持多样化（pluralistic）对齐研究；④ 在构造、验证、评估上采用人类、LLM-as-judge 和自动指标的多模态方案。

**🔧 技术方法**

技术手段包括：大型语言模型（LLM）进行对话生成与漂移标注；对话后处理与规则过滤；人类与 LLM 作为评判器进行标注校正；语义相似度评估（Sentence‑BERT embeddings）、词汇相似度（LCS、Jaccard、ROUGE‑L）；使用 Variational Preference Learning（VPL）训练偏好模型；对齐数据构造采用 pairwise preference 形式，结合 persona‑conditioned 标注。

**📊 数据集**

使用的主要数据集为 ConvoDrift：15,727 条多轮（6 轮）对话，覆盖 5 种沟通体裁（商务邮件、休闲邮件、名言祝福、领英帖子、推文）和 5 个人设；在下游实验中使用 GPT‑2、Llama 3.1 8B 作为模型。

**📈 对比分析**

比较方法：① 人类评估（3 位评审，Krippendorff’s α≈0.88，Drift 0.76/方向 0.78）；② LLM‑as‑Judge（Drift κ≈0.888，方向 κ≈0.930）；③ 语义相似度指标（adjacent≈0.899，anchor≈0.890，end‑to‑start≈0.852）；④ 词汇相似度在漂移边界显著下降；⑥ 下游偏好学习：VPL 在 GPT‑2 上准确率 0.579，在 Llama 3.1 上 0.65。总体表现显示数据集质量高，且能有效训练风格偏好模型。

**⚠️ 局限性**

局限性：① 仅为合成数据，缺乏真实人类对话的随机性；② 只考虑“正式”与“休闲”两种风格，未覆盖更细腻的情感或风格维度；③ 对话长度固定为 6 轮，限制了长程风格依赖分析；④ 采用的 LLM 生成与标注可能带来模型特定的偏好与噪声；⑤ 人设仅限五种，未能覆盖更广泛的用户偏好空间。

---

## 355. Constraint-Aware Training

**arXiv ID:** 2610.02909 | [PDF](https://arxiv.org/pdf/2610.02909v1)

**作者:** Jinwoo Kim `[一作]` `[通讯]` (University of California-San Diego), Jinwoo Kim (University of California-San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了将前缀约束分析外部化到训练过程的约束感知损失（Constraint‑Aware Loss），从而在不必在模型内部实现语法/作用域/类型约束的情况下训练语言模型；

**💡 创新点**

创新点在于将任何可计算的前缀约束转化为训练目标，证明其可外部化后对模型宽度、深度和数据效率产生理论上可量化的优势，并通过合成语料实验验证；

**🔧 技术方法**

主要技术包括约束感知损失的定义与梯度分解、基于掩码的softmax重正则化、以及对Transformer读出层和深度的可达性分析；

**📊 数据集**

使用了多种人工合成语料：作用域分析（包含 12 个作用域变量和 61 个候选变量）、基于算术表达式的组间约束、以及多语料规模的随机生成数据集；

**📈 对比分析**

通过比较在相同宽度/深度下使用约束感知训练与传统交叉熵训练的余量损失、验证误差和对不同语料的方差，结果显示约束感知模型在宽度和深度要求上均显著降低、在小样本场景下表现更好、方差更小；

**⚠️ 局限性**

局限性包括：实验仅基于合成数据，缺乏对真实编程语言数据的验证；约束分析的有效性依赖于分析的可计算性和效率；理论证明假设了训练的完美收敛，实际训练中仍受优化器和初始化影响。

---

## 356. How Robust Is Multimodal Claim Verification to LLM Rewriting?

**arXiv ID:** 2610.02841 | [PDF](https://arxiv.org/pdf/2610.02841v1)

**作者:** Yun-Ang Wu `[一作]` (Nii LlmC), Akiko Aizawa `[通讯]` (Nii LlmC)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究大型语言模型在进行多模态声明验证时，文字风格重写（自然重写与单词注入）对模型预测结果的影响。

**💡 创新点**

首次在多模态声明验证任务中系统评估非攻击性风格变更对模型决策的影响，并揭示即便准确率不变，概率估计仍会出现一致的偏移，尤其是谨慎语气与保守词注入导致的概率下降。

**🔧 技术方法**

采用 GPT-OSSt-120b 进行重写，使用自一致性（self‑consistency）抽样估计支持概率，评估 11 种公开权重的视觉语言模型（Gemma、GLM、InternVL、Kimi-VL、Qwen3）。

**📊 数据集**

使用 SciClaimEval 数据集（配对的支持/反驳声明与表格/图像证据）进行实验。

**📈 对比分析**

比较方法：在原始声明与多种重写条件下测量配对准确率、F1 以及支持概率的变化。结果显示：大多数模型在准确率上几乎无显著变化；但在谨慎语气（CT）和保守词注入（C1）下，支持概率在 11/11 模型中显著下降，且变化显著性达到 q<0.05；相比之下，提升语气（OA）或自然润色（LP、FI、AS）对概率影响有限。模型规模与概率偏移关系不显著，除 Qwen3 系列中大模型表现出更大幅度的偏移。

**⚠️ 局限性**

局限性：①仅使用单一重写模型（gpt‑oss‑120b）和 SciClaimEval 数据集，缺乏跨任务、跨领域的泛化验证；②重写过程仍可能引入语义变更（尤其是插入保守词前的数值修饰），难以完全区分风格与语义影响；③概率估计仅基于 10 次自一致性抽样，未与其他置信度提取方法比较；④未评估在更具挑战性或主观性案例中风格变更对最终决策的影响。

---

## 357. MixVLA: Adaptive Mixing of Non-Invariant Information for Generalizable Vision-Language-Action Models

**arXiv ID:** 2610.02898 | [PDF](https://arxiv.org/pdf/2610.02898v1)

**作者:** Pingrui Zhang `[一作]` (Fudan University), Xuelong Li `[通讯]` (TeleAI, China Telecom Corp Ltd)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了MixVLA框架，改进Vision‑Language‑Action模型在零样本OOD环境下的泛化能力；

**💡 创新点**

核心创新是自适应混合非稳健信息（AMI），通过在不变与变异特征间做随机混合来正则化环境特定噪声，并在不需要额外OOD数据或网络改造的前提下提升鲁棒性；

**🔧 技术方法**

使用信息瓶颈（IB）提取不变特征，权重减法构造变异特征，AdaIN与线性插值实现AMI混合，随后与原始模型头结合；

**📊 数据集**

在LIBERO、LIBERO‑Plus、RoboTwin和真实Franka机器人抓取放置任务等多种数据集上进行评估；

**📈 对比分析**

与OpenVLA‑OFT、π_0.5等基线在零样本OOD测试（如光照、视角、语言、随机化环境）中对比，MixVLA在LIBERO‑Plus成功率提升至76.2%（相较基线+6.6%），在RoboTwin C2R提升至62.4%（+16.4%）并保持高域内表现；

**⚠️ 局限性**

局限性包括对视角引起的几何偏移效果有限、需要两阶段训练以及对不同网络结构的具体适配仍需进一步验证。

---

## 358. PsyEvo: A Personalized Counseling Agent That Self-Evolves at Test Time

**arXiv ID:** 2610.02885 | [PDF](https://arxiv.org/pdf/2610.02885v1)

**作者:** Yuting Yan `[一作]` (Lyncia Lab), Minghao Wang `[通讯]` (Lyncia Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了PsyEvo，一种在多会话心理咨询中实现客户端个性化与共享响应策略自适应的LLM测试时学习框架。

**💡 创新点**

通过将客户端私有的技能后验（HBSP）与共享响应适配器（LiPO）分离，并利用SOCA构建候选偏好与序列信用，实现了在冻结主干模型的前提下的双向学习。

**🔧 技术方法**

使用了Hierarchical Bayesian Skill Policy、Listwise Preference Optimization、State‑conditioned Ordinal Credit Assignment以及Qwen3‑32B背骨与LoRA适配器的混合训练。

**📊 数据集**

在PsychEval（100名模拟患者、五种治疗模式）和SummEval（1600篇摘要）上进行评估，且对照多种基线模型。

**📈 对比分析**

在PsychEval上取得总体得分7.684，超过所有基线（最高为7.363），在SummEval上获得平均Spearman相关0.508，优于其他自动评估方法。

**⚠️ 局限性**

局限性包括仅使用模拟患者与LLM评审，未验证对真实患者的临床效果；共享在线适配未证实对未见患者的迁移；各组件互补性未单独验证；人类评估样本量小且一致性有限。

---

## 359. DyRA: Dynamic Residual Approximation for Efficient Matrix Multiplication in DNNs

**arXiv ID:** 2610.02882 | [PDF](https://arxiv.org/pdf/2610.02882v1)

**作者:** Daewon Chae `[一作]` (University of Michigan), Hun-Seok Kim `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在推理阶段对预训练模型的结构化矩阵乘法加入输入自适应残差纠正，提出 DyRA 方法。

**💡 创新点**

创新点在于直接对输出空间的残差进行低秩近似，并通过一次 ALS 迭代实现高效、实时的纠正，显著降低权重近似导致的输出误差。

**🔧 技术方法**

采用低秩、Monarch、BLAST 等结构化权重，使用 ALS 求解低秩因子，利用 Triton 自定义 kernel 进行高效融合计算，进行动态残差低秩纠正。

**📊 数据集**

实验数据集包括 ImageNet（ViT‑L, DINOv3）、ADE20K、NYUv2、LibriSpeech、LLaDA‑8B、HumanEval、GSM8K 等多任务多模态数据。

**📈 对比分析**

与仅使用结构化权重或输入感知分解的基线在相同 FLOPs/算力预算下进行比较。DyRA 在视觉、语言、语音任务中均取得更低误差/更高准确率，DINOv3 在 60% FLOPs 降低时 mIoU 仅下降 1.7 点，GPU 加速达 1.5×，且准确率下降仅为基线的 1/3。

**⚠️ 局限性**

主要局限在于对大规模矩阵乘法有效，对 token‑by‑token 的自回归推理不适合；动态残差 rank 需要额外计算，且未探索按输入动态分配残差维度的策略。

---

## 360. AgentTrap: Stateful Feedback Deception against Autonomous Penetration Testing Agents

**arXiv ID:** 2610.02869 | [PDF](https://arxiv.org/pdf/2610.02869v1)

**作者:** Yuelin Wang `[一作]` (Tianjin University), Yanbang Sun `[通讯]` (Tianjin University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了第一个闭环蜜罐系统，针对自主渗透测试代理（LLM驱动），通过哨兵端点、状态化欺骗与行为驱动升级机制，持续误导代理并在攻击过程中诱捕其 API key，从而阻止真实资产被攻破并收集攻击者行为证据。

**💡 创新点**

创新点包括①使用闭环动态响应，基于代理的实时行为与历史交互生成可信的伪造反馈；②引入哨兵端点，仅对攻击者可达，避免对合法用户造成干扰；③将真实应用的运行时状态作为上下文，限制LLM生成的幻觉；④通过行为引导升级策略决定何时触发对抗攻击，避免过早或过迟触发。

**🔧 技术方法**

技术主要包括：基于大语言模型的控制器生成上下文感知回复；状态化欺骗模块；行为监控与升级决策逻辑；对抗攻击（诱捕API key）实现；以及与真实SQL注入目标的集成。

**📊 数据集**

实验使用八个公开的自主渗透测试代理（GitHub 星级均超过28k），并对每个代理配备两种LLM后端（Flash、Pro等）。实验场景为一个包含真实SQL注入漏洞的业务端点与一个单独的蜜罐端点的沙盒环境。

**📈 对比分析**

对比方法：在同一应用下设置四种防御策略（无防御、静态欺骗、固定升级、AgentTrap），测量攻击成功率、对抗成功率（是否抓取API key）以及代理的token消耗与耗时。结果显示AgentTrap将攻击成功率从95.8%降至79.2%，对抗成功率达到18.8%，显著优于其他策略，同时保持合理的资源消耗。

**⚠️ 局限性**

局限性包括：对抗效果高度依赖于LLM的推理与对欺骗请求的识别能力，某些模型（如Claude Code、AIDA）会拒绝泄露密钥；某些代理架构（如Strix）通过沙箱隔离阻止密钥泄露；实验仅涵盖SQL注入场景和八个代理，缺乏对其他漏洞类型与更广泛代理的验证；随机性高，需更多实验以提升统计可靠性；实际部署对合法用户影响与运维成本尚待评估。

---

## 361. On Unlearning for Time-series Forecasting

**arXiv ID:** 2610.02865 | [PDF](https://arxiv.org/pdf/2610.02865v1)

**作者:** Zeyu Shi `[一作]` (Chinese University of Hong Kong Shenzhen), Lixu Wang `[通讯]` (Chinese University of Hong Kong Shenzhen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种用于时间序列预测的近似机器删学习框架RDTU，能够在不重新训练模型的情况下高效删除指定观测并保持预测性能。

**💡 创新点**

创新点包括：①基于神经切线核（NTK）预测的基准标签构造；②结合全局与局部结构支持的结构不可替换得分（SIS）来评估删除窗口的重要性；③采用条件扩散模型学习NTK与真实模型残差，从而生成逼近精确删学习的伪标签；④以轻量化标签更新实现高效删学习。

**🔧 技术方法**

使用的核心技术包括：神经切线核预测、结构不可替换得分、条件扩散模型、标签导向的轻量级参数更新。

**📊 数据集**

实验使用四个公开时间序列基准：Air Quality、Traffic（单变量）；EEG、HAR70+（多变量）。

**📈 对比分析**

与SSD、NegGrad+、Zero-label、Random-label、SCRUB-R、TS-Unlearn等基线及Transformer基准相比，RDTU在F‑P Gap和NRMSE上均取得最低误差，U/N AUC亦保持在可接受范围内，并且计算时间大幅低于完整重训练。

**⚠️ 局限性**

限制：对删除比例较大或结构极度不均衡的数据集效果可能下降；仅提供近似删学习，缺乏形式化的安全保证；在成员识别方面仍存在一定差异；扩散模型训练仍有一定计算成本。

---

## 362. NeuroLens: Learning Latent Embeddings of Neural Semantics from Chronic Recordings

**arXiv ID:** 2610.02864 | [PDF](https://arxiv.org/pdf/2610.02864v1)

**作者:** Hanrui Lyu `[一作]` (Northwestern University), Yizi Zhang `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于自监督联合嵌入预测架构的适应性编码器与因果变压器，用于从慢性神经记录中学习去噪、语义化的潜在表示。

**💡 创新点**

创新点在于结合跨日可变神经元的跨注意力身份编码和共享的因果预测器，在潜在空间预测而非重构，显著提升对记录非平稳性的鲁棒性和对高阶任务变量的解码性能。

**🔧 技术方法**

使用了自监督学习、跨注意力编码器、因果Transformer、签名正则化(SIGReg)以及少量校准样本进行快速无梯度适配。

**📊 数据集**

在小鼠IBL Neuropixels慢性视觉决策数据和两名ALS患者的慢性Utah阵列语音尝试数据上进行验证。

**📈 对比分析**

与Spike、Causal NDT、CEBRA、SPINT、POYO、POSSM等基线对比，LENS在选择/运动解码、句子嵌入解码上分别提升约6%、2%和38%，并且在少量校准下即可实现快速适配。

**⚠️ 局限性**

局限在于未能将学习引起的表征漂移与行为变化及内在神经变异分离，缺乏跨区域交互与区域特异性漂移分析。

---

## 363. Evaluating LLM-as-a-Judge Beyond Score Alignment: A Psychometric Analysis of Residual Judging Difficulty

**arXiv ID:** 2610.02877 | [PDF](https://arxiv.org/pdf/2610.02877v1)

**作者:** Longwei Cong `[一作]` (DIPF Leibniz Institute for Research and Information in Education), Ulf Kroehne `[通讯]` (Chemnitz University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用心理测量方法（Many‑Facet Rasch Model）对人类与LLM在SummEval摘要评估中的评分进行分解，研究了两者在潜在摘要质量和判断难度（残差硬度）上的匹配与差异。

**💡 创新点**

提出并使用残差硬度（Residual Hardness）诊断指标来揭示人类与LLM在判定哪些案例难以评分时的差异；发现维度级别的显著不匹配，并证明人类易、LLM难案例可通过文本特征预测。

**🔧 技术方法**

主要技术包括Many‑Facet Rasch Model（MFRM）拟合、残差硬度计算、相关系数和百分比差异比较、逻辑回归预测以及SHAP特征重要性分析。

**📊 数据集**

使用SummEval数据集：1600条系统摘要、4个评价维度（连贯性、一致性、流畅性、相关性）、8名人类评审和17个开源LLM作为自动评判者。

**📈 对比分析**

人类与LLM的潜在质量相关系数约为0.37，而残差硬度相关系数仅为0.15，说明质量匹配不等于难度匹配；在维度层面Consistency出现LLM‑hard↑23.9个百分点，Coherence出现人类‑hard↑22.5个百分点。预测模型AUROC在0.63–0.72之间，AUPRC在0.34–0.44之间，表明可部分预测人类易、LLM难的案例。

**⚠️ 局限性**

局限性包括：仅在单一摘要基准SummEval上验证；使用固定提示和单一LLM面板；残差硬度仅为诊断工具，未说明其因果含义；预测结果仅揭示相关性，未解释机制。

---

## 364. Bounded Reachability & Jailbreak Detection via Contraction-Constrained State Space Models

**arXiv ID:** 2610.02853 | [PDF](https://arxiv.org/pdf/2610.02853v1)

**作者:** Omanshu Thapliyal `[一作]` `[通讯]` (Hitachi America Ltd.), Omanshu Thapliyal (Hitachi America Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于状态空间模型（SSM）的安全头，并通过收缩约束实现对输入扰动的正式可证性检测；

**💡 创新点**

证明当状态转移矩阵的∞-范数满足A_∞<1时，线性时间不变（LTI）SSM可实现精确的区间界传播，从而获得非空可信区间；

**🔧 技术方法**

使用精确区间传播（IBP）、收缩正则化（hinge penalty）、S4架构的两层状态空间头以及逻辑回归线性探针；

**📊 数据集**

在毒性评论数据、JailbreakBench（JBB）、AdvBench、HarmBench等公开基准上进行实验；

**📈 对比分析**

与字符串/关键字判断器、逻辑回归线性探针和MLP进行对比，S4安全头在JBB静态威胁模型下达成100%检测率，AUROC高达0.994；在自适应攻击下仍易受攻击；相比之下线性探针在所有指标上表现更佳；

**⚠️ 局限性**

限制在于：S4安全头的判别性能与线性探针相近，主要优势是可证性；在自适应攻击、版权和骚扰类别上表现欠佳；未来需加强对抗训练与更深层次的非线性SSM探索。

---

## 365. Counterfactual Action Evaluation, Observation Bottlenecks, and Representation Geometry in Joint-Embedding Predictive World Models

**arXiv ID:** 2610.02860 | [PDF](https://arxiv.org/pdf/2610.02860v1)

**作者:** Arjun Subramanian `[一作]` `[通讯]` (Massachusetts Institute of Technology), Arjun Subramanian (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本工作在二维可变形物理模拟器中对行动条件化的 JEPA（V-JEPA2）进行审计，跟踪同一物理状态在模拟器、像素观测、目标嵌入和预测器输出之间的干预路径；

**💡 创新点**

提出一种评估协议，能分别定位观察可见性、表示几何与行动依赖三种瓶颈，系统验证三种失败模式；

**🔧 技术方法**

使用 VICReg 风格的对比学习与行动条件化预测器，ViT 编码器、EMA 目标网络，结合模拟器分叉、统计量（参与度、有效秩、线性探针）进行分析；

**📊 数据集**

使用 2000 条长度 16 的物理轨迹数据集，包含圆盘或条形物体，Young 模量取值 {18,70,220}，每帧为 64×64 的密度/速度物理 raster；

**📈 对比分析**

通过对比 MSE‑only 与 VICReg、行动无关/有关模型，使用 10 步 rollout 误差、有效秩、线性探针准确率等指标；结果显示 MSE‑only 在 latent 误差上低 8 倍但表示高度集中，行动条件化并未显著改善 rollout 误差；在高可见性反事实中预测器响应仅为目标的 0.5%–2%；

**⚠️ 局限性**

局限性包括：环境对行动影响弱、像素量化抹去差异、仅在短周期二维场景验证、未评估闭环控制、未检验预测方向一致性、样本量有限且无独立测试集。

---

## 366. Misinformation Without Triggers: From Factual Answers to Downstream Decisions

**arXiv ID:** 2610.02886 | [PDF](https://arxiv.org/pdf/2610.02886v1)

**作者:** Lin Tian `[一作]` (University of Technology Sydney), Marian-Andrei Rizoiu `[通讯]` (University of Technology Sydney)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了无触发器的虚假训练如何影响语言模型的事实回答与基于这些回答的决策，并通过“Guess the Capital”游戏和澳洲山火谣言案例进行验证。

**💡 创新点**

揭示了直接事实审计与模型实际决策之间的“审计差距”，并证明即使事实回答已被纠正，模型仍可能保留错误决策。

**🔧 技术方法**

使用持续预训练与指令微调、固定解码规则的决策游戏、直接对比式提示探测以及多模型对照实验。

**📊 数据集**

对8种LLM进行高剂量注入虚假事实的训练，使用合成的首都数据、澳洲山火相关的Facebook帖子以及MMLU基准。

**📈 对比分析**

与匹配的真值训练基线对比，评估注入事实在直接回答的命中率（95.8–100%）以及在决策中的选择提升（1.7–14.4%）和整体游戏准确率下降（3.3–26.4%），MMLU上仅出现微小差异。

**⚠️ 局限性**

实验范围受限于单一虚假事实族、有限模型种类以及固定决策规则，缺乏对更广泛误信息类型与更复杂决策场景的验证。

---

## 367. Tangent Schrödinger Bridge Matching: Learning Stochastic Transport with Mechanistic Sensitivities

**arXiv ID:** 2610.02906 | [PDF](https://arxiv.org/pdf/2610.02906v1)

**作者:** Jowaria Khan `[一作]` (University of Michigan), Elizabeth Bondi-Kelly `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在随机系统中学习终端分布与干预响应的模型，称为 Tangent Schrödinger Bridge Matching（Tangent‑SBM），通过监督轨迹的参数灵敏度来改进模型对干预变化的预测。

**💡 创新点**

创新点在于：①在 Schrödinger 桥的训练目标中加入灵敏度监督；②区分单轨迹响应和平均响应监督，利用双重采样消除对响应方差的额外惩罚；③证明精准灵敏度可界定有限变化预测误差和决策遗憾，并通过理论与实验验证其有效性。

**🔧 技术方法**

使用条件 Schrödinger 桥（Conditional DSBM）框架，结合自动微分推导轨迹灵敏度，构造双重采样损失；训练时在原始桥损失上加权灵敏度损失；在实验中对比多种基线（Conditional DSBM、GSBM、TSBM、Sobolev‑DSBM）。

**📊 数据集**

在四个基准上评估：低维高斯分布、随机双井势、PDEBench 反应扩散 PDE 以及 SPDEBench Navier‑Stokes，分别覆盖平滑连续、双峰非线性、空间场与高维随机流体等情形。

**📈 对比分析**

与条件 DSBM 等基线相比，Tangent‑SBM 在敏感度误差上提升 50‑70%，在有限变化预测误差和决策跟踪误差方面也显著下降；在 Navier‑Stokes 的干预选择实验中，跟踪误差减少 54% 以上，决策成功率达 79%。

**⚠️ 局限性**

局限性包括：需要来自物理模拟器的灵敏度标签（若不可得或标签质量低则效果受限）；双重采样带来额外计算开销；在极端 OOD 情况下，敏感度覆盖不足仍导致预测误差升高；对高维系统的可扩展性和训练稳定性尚需进一步研究。

---

## 368. PointWAM: 3D World Action Modeling for Dexterous Robotic Manipulation

**arXiv ID:** 2610.02840 | [PDF](https://arxiv.org/pdf/2610.02840v1)

**作者:** Chunghyun Park `[一作]` (POSTECH), Minsu Cho `[通讯]` (POSTECH)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并训练了 Point World Action Model (PWAM)，通过预测共享时空 3D 点轨迹联合学习场景与手部动作，并将手部轨迹转化为机器人指令，实现从人类视频预训练到机器人执行的无任务特定点掌握。

**💡 创新点**

①将场景与手部解耦为共享时空 3D 点轨迹；②利用人类视频中的同一 3D 轨迹实现跨模态预训练；③通过手部轨迹再映射实现无规划的动作预测。

**🔧 技术方法**

使用 3D 点云编码器 Mosaic3D、Transformer 编码/解码器、轴向 3D 位置嵌入、手部关键点特征、轨迹监督、动作再映射等技术。

**📊 数据集**

预训练采用 EgoDex 与 VITRA 共 1.15M 人类演示视频；微调与评估使用 DexJoCo、RoboDojo-Precision、OpenArm 实验平台。

**📈 对比分析**

与 GR00T N1.6、Fast-WAM、PointACT 等基线对比，PWAM 在 DexJoCo 10 项任务平均成功率 69%（比 GR00T 高 11.7pp），在 RoboDojo-Precision 上获得 4.8% 成功率，并在真实机器人香蕉取放/方形插槽任务上显著优于对比模型。

**⚠️ 局限性**

仅基于点云，可能缺失细粒度表面信息、透明物体或文字，且对不同机器人结构的适配仍需进一步验证。

---

## 369. ViTok: Improving Dense Semantics in AM-RADIO-Style Multi-Teacher Distillation with PHI-S and Masked Image Modelling

**arXiv ID:** 2610.02903 | [PDF](https://arxiv.org/pdf/2610.02903v1)

**作者:** Hailun Xu `[一作]` (Beihang University), Kanchan Sarkar `[通讯]` (Indian Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个多教师蒸馏 recipe，目标是让单一 ViT-B 学生既能保持强大的全局识别能力（ImageNet‑1K kNN），又能获得稳健的密集语义表现（ADE20K 分割）。为此作者采用 SigLIP + DINO 作为两个互补教师，并在蒸馏过程中引入了 CLS/patch 级别的 split adaptor、异步损失（CLS 用余弦损失，patch 用 MSE）、PHI‑S 特征归一化、Masked Image Modeling（MIM）辅助正则以及教师权重调节，最终实现了在 ImageNet‑1K kNN 上略优于 SigLIP 教师，并在 ADE20K 上恢复到教师水平。

**💡 创新点**

创新点主要包括：①将 CLS 与 patch 采用分离 adaptor 与不对称损失处理，显著缓解全局与密集特征冲突；②提出 PHI‑S 旋转+同质归一化来平衡多教师统计差异，恢复密集语义；③将 MIM 作为稀疏辅助目标，在多教师蒸馏中提升 patch 语义并避免单纯匹配导致的稀疏退化。

**🔧 技术方法**

技术手段：ViT‑B student；split adaptor heads；余弦损失（CLS）+ MSE 损失（patch）；PHI‑S 归一化；教师权重重置；Masked Image Modeling（MIM）辅助正则；MAE 预训练初始化；教师特定的 projection heads。

**📊 数据集**

数据集：ImageNet‑1K（训练 & kNN 评估），ImageNet‑22K（训练尝试），ADE20K（线性分割评估）。额外尝试的教师包括 SAM3、HOG‑style 特征。

**📈 对比分析**

对比方式：使用 kNN（CLS 及 patch token）在 ImageNet‑1K 上评估分类精度；在 ADE20K 上采用线性 probe 评估 mIoU/mAcc。实验结果显示，最佳配置取得 83.2 patch‑kNN、85.2 CLS‑kNN，略优于 SigLIP 教师；在 ADE20K 上恢复到 48.5 mIoU / 61.0 mAcc，达到教师水平。扩展到 ImageNet‑22K 或加入额外教师未能提升性能，说明仍存在干扰。

**⚠️ 局限性**

局限性：①多教师间的统计干扰难以一次性完全解决，仍需手动权重调节；②MIM 的收益依赖学习率与优化设置；③实验仅覆盖 kNN 与线性分割，未验证检测、分割精细化等下游任务；④PHI‑S 的统计需要离线预计算，可能不适合在线训练；⑤在更大规模数据或更多教师时仍表现不稳定，说明需更系统的教师协调机制。

---

## 370. Found but Not Read: When Extracted Text Closes the Retrieval-Reading Gap in Document Vision-Language Models

**arXiv ID:** 2610.02880 | [PDF](https://arxiv.org/pdf/2610.02880v1)

**作者:** Qingtao Xia `[一作]` (Harbin Institute of Technology), Jie Liu `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估检索增强式文档问答中的检索‑阅读间隙，提出配对协议以量化 OCR 与文本层对视觉语言模型的提升。

**💡 创新点**

引入检索控制的配对协议、可追踪证据数据集 FoveDoc‑Bench、并揭示文本模态与检索质量对性能的双重边界。

**🔧 技术方法**

使用视觉检索器 ColQwen2、视觉语言模型 Qwen3‑VL、CPU OCR RapidOCR、LLM 判定与 McNemar 检验等技术。

**📊 数据集**

使用 FoveDoc‑Bench（1173 文档、2346 题）和 MMLongBench‑Doc（1091 题）作为评估数据集。

**📈 对比分析**

在检索已饱和的条件下，将仅图像与图像+OCR/文本层的两臂结果配对比较，OCR 提升 13–16 分点，文本层约翻倍，六个 VLM 在三大族群均显著受益。

**⚠️ 局限性**

仍存在 13–16 点未使用证据的检索‑阅读间隙；OCR 误差、图表模态对文本层无效；当检索质量下降时，优势显著减弱。

---

## 371. Query-aware routing for Cross-lingual performance gains in Encoders

**arXiv ID:** 2610.02875 | [PDF](https://arxiv.org/pdf/2610.02875v1)

**作者:** Akshay Jain `[一作]` (ConfidentialMind), Edward Kim `[通讯]` (ConfidentialMind)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究通过在冻结的文档编码器上训练查询仅低秩适配器，并结合语言路由规则，实现芬兰语和瑞典语与英语之间的跨语言检索提升，同时保持同语言检索性能不变。

**💡 创新点**

创新点在于将查询适配器与确定性的语言路由结合，既可在不重新编码文档向量的前提下提升跨语言检索，又能通过路由保留原始同语言检索效果。

**🔧 技术方法**

使用的技术包括LoRA低秩适配器（针对Nemotron‑3‑Embed‑1B或Harrier基模型的查询编码器）和基于查询与文档语言标签的语言路由决策。

**📊 数据集**

使用的数据集为FIQA金融问答检索集（英、芬、瑞三语），以及MIRACL‑fi和Mr. TyDi‑fi的全篇语料库。

**📈 对比分析**

在FIQA 194个查询样本上，以nDCG@10为评测指标，跨语言平均提升约20.9%（从0.2406到0.2908），所有六个语言方向均有提升；同语言分数与基线保持一致。

**⚠️ 局限性**

局限性包括：仅评估芬兰语/瑞典语与英语的跨语言检索；缺乏自然瑞典语查询或其他北欧语言的评估；未考察混合语言文档集；依赖准确的语言标签且未评估下游答案质量或生产力影响。

---

## 372. Evaluating VQA in Vision Language Models using Cooperative Principles

**arXiv ID:** 2610.02878 | [PDF](https://arxiv.org/pdf/2610.02878v1)

**作者:** Monika Shah `[一作]` (University of Memphis), Deepak Venugopal `[通讯]` (University of Memphis)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估视觉语言模型在视觉问答任务中，当问题包含违反格里斯合作原则（非必要信息、模糊或虚假修饰）时的推理性能。

**💡 创新点**

创新点包括：①使用VLM自身生成问题修饰并通过人工验证，构建Grice准则违规作为诊断框架；②对比人类与VLM的语用推理差异；③比较人类生成与AI生成违规对VLM推理的影响；④关联VLM推理与人类认知负荷。

**🔧 技术方法**

技术手段包括：VLM生成器（ChatGPT‑4o、Claude Sonnet 3.5、Gemini‑1.5‑Flash、Llava‑7B）用于生成修饰；VLM回答器用于回答原始与修改后问题；AMT人工评估用于验证修饰合法性、答案质量、简化问法；统计检验（McNemar检验、t检验）评估显著性；BERTScore 评估开放式答案语义相似度；CLIPScore 评估问题与图像的视觉相关性。

**📊 数据集**

使用的数据集：VQA v2.0 测试集（995问，含二进制、定量、开放式），人类修改版数据集（500二进制+130开放式）以及AMT收集的答案与修饰。

**📈 对比分析**

比较方法：计算原始问题与修改后问题的准确率下降、McNemar检验验证显著性、人工评估开放式答案优劣、BERTScore 与 CLIPScore 对比人类与VLM答案。结果显示：VLM准确率平均下降 6‑17%，Gemini 生成的违规影响最大；GPT 对自身修饰表现相对好；人类生成的违规对VLM影响较小；VLM 在处理人类违规时更接近人类语用推理，且准确率高于 AI 违规。

**⚠️ 局限性**

局限性：①假设原始人类问题遵循格里斯准则，可能不完全成立；②自我生成修饰可能导致偏差；③开放式问题无唯一答案，人工评估可能带来主观性；④未考虑 VLM 响应的不确定性；⑤实验仅聚焦 VQA，未涵盖更广泛的多模态任务。

---

## 373. HASTE: Evolving Agent Harnesses Against Emerging Attacks Using Sparse Evidence

**arXiv ID:** 2610.02920 | [PDF](https://arxiv.org/pdf/2610.02920v1)

**作者:** Xiqiao Xiong `[一作]` (University of Science and Technology of China), Xiangnan He `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种多智能体框架 Haste，用于在仅有稀疏威胁证据的情况下自动演化 Agent 的 harness，以提升安全性并保持任务效用。

**💡 创新点**

创新点在于通过安全规范生成与攻击案例生成的对抗性循环，利用评估反馈动态更新安全规范和攻击案例，从稀疏证据外推出更全面的防御策略。

**🔧 技术方法**

使用了五个智能体（攻击解析器、规范优化器、案例优化器、提议者、评判者），以及 LLM（如 DeepSeek‑V4‑Pro、Claude Code、GPT‑5.5）来生成规范、案例、修改 harness 并评估安全/效用。

**📊 数据集**

主要实验数据集包括 Agent‑SafetyBench（ASB）、STAC 以及基于 ASB 的 ASB‑300 子集，用来评估攻击成功率（ASR）和安全有效率（SHR）。

**📈 对比分析**

与无防御、Guardrail 模型、传统 harness 防御以及 Meta‑Harness 进行比较。结果显示 Haste 在所有 backbone 模型上均显著降低 ASR、提升 SHR，且在跨 benchmark、跨模型迁移以及多阶段演化场景中表现出更强的鲁棒性和稳定性。

**⚠️ 局限性**

局限性包括：依赖评判者的准确性；对评判模型选择敏感；生成案例可能导致演化漂移；在真实环境中的部署与持续更新尚未验证；以及对不同类型威胁的泛化能力仍需进一步探究。

---

## 374. Learning Jazz Pianist Style with Cross-Attention Conditioning

**arXiv ID:** 2610.02918 | [PDF](https://arxiv.org/pdf/2610.02918v1)

**作者:** Drew Edwards `[一作]` (Queen Mary University of London), Simon Dixon `[通讯]` (Queen Mary University of London)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

使用预训练的符号音乐Transformer对爵士钢琴家风格进行建模，训练分类器识别钢琴家身份，并在Transformer上加入门控交叉注意力适配器实现对特定艺术家风格的条件生成；随后通过分类器一致性和合成转移等指标评估生成的风格保真度，并提出基于分类器置信度的“特征区域检测”方法定位每位钢琴家在演奏中最具代表性的片段。

**💡 创新点**

（1）首次将门控交叉注意力与艺术家嵌入结合，用于符号音乐的全球条件生成；（2）提出基于分类器一致性的风格保真度评估方案，克服了困扰此类任务的困惑率/困惑度指标；（3）创新性地将分类器再利用为“特征区域检测”，实现对钢琴家风格关键片段的自动定位。

**🔧 技术方法**

Aria 预训练Transformer、门控交叉注意力适配器、对比学习得到的艺术家嵌入、基于交叉熵的分类器、滑动窗口一致性协议、合成迁移实验、z‑score 归一化的特征区域检测方法。

**📊 数据集**

PiJAMA‑30（30位爵士钢琴家）中挑选的12位风格显著的钢琴家（PiJAMA‑12），以及 Deep Pianist Identification（DPI‑20）基准。

**📈 对比分析**

与 DPI‑20 上的 ResNet‑50 基线比较：分类器在 20 类上达到 96.9% 轨道级精度；在 PiJAMA‑12 上分类器得到 98.8% 轨道级精度；条件生成模型在滑动窗口协议中对真艺术家的一致率提升至约 70%，比无条件基线 37% 高出 33个百分点；合成转移实验中，仅用合成数据训练的分类器在真实测试集上获得 95.0% 轨道级精度，证明生成音乐保留了可迁移的风格信息。

**⚠️ 局限性**

实验仅覆盖 12 位风格高度可区分的钢琴家，难以验证模型在风格模糊或训练样本稀少的艺术家上的泛化能力；评估仍基于自动分类器，缺乏主观听感验证；模型对不同乐器或伴奏环境的适用性未知。

---

## 375. Peer Effects in Signed Networks: Separating Influence Through Positive and Negative Ties

**arXiv ID:** 2610.02872 | [PDF](https://arxiv.org/pdf/2610.02872v1)

**作者:** Xiaojing Du `[一作]`, Thuc Duy Le `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文定义并识别了在带符号网络（正连结与负连结）中的四种同伴效应：正连结效应、负连结效应、它们的交互效应以及符号组合效应，并提出一种新的双重稳健估计器SiDE（Signed-exposure Doubly Robust Estimator）来准确估计这些效应；此外，还给出了考虑重叠邻域的区间估计方法。

**💡 创新点**

创新点在于：①首次将符号信息嵌入同伴效应定义，并证明在符号模糊分配下 unsigned 效应为正负连结效应的加权平均；②提出将符号特定邻居编码与 Poisson–binomial 归因概率相结合的双重稳健得分（SiDE），实现对符号网络的高效估计；③引入符号组合效应，为干预设计提供“同一总处理数下选择正连结还是负连结的影响”洞察。

**🔧 技术方法**

技术手段包括：双重稳健 AIPW 得分与 Poisson–binomial 归因概率、针对正负邻居的分离多层感知机编码器、邻域嵌入、近似方差估计（考虑重叠邻域）与 Clopper–Pearson 区间估计；同时使用半合成实验与实际学校干预数据进行验证。

**📊 数据集**

使用的真实符号网络共六个：Bitcoin‑Alpha、Bitcoin‑OTC、Wiki‑Elec、Slashdot、Epinions、Reddit，用于构建半合成实验；另外还利用学校冲突干预实验的数据做案例分析。

**📈 对比分析**

与十一种基线（饱和回归、GPS、TARNet、CFR、SAGE‑HSIC、NetEst、HINITE、DWR、GNN‑DR、HINet、GAIPW）在四个效应上进行 MAE 与绝对偏差比较。结果显示，SiDE 在大多数网络与效应上取得最低 MAE 与最小偏差；区间覆盖率在绝大多数情况下满足 95% 置信水平，只有少数网络出现轻微欠覆盖。

**⚠️ 局限性**

局限性包括：①仅考虑静态网络，未处理网络随干预而改变的动态情况；②符号组合效应仅在总处理数固定时有意义；③在某些网络（如 Reddit、比特币网络）区间覆盖率略低于 95%；④模型对隐藏层宽度略敏感；⑤未进一步细分不同强度或类型的正负连结。

---

## 376. DIVINE: Simple Cross-Market Stock Pretraining via Diverse Indicator Reconstruction

**arXiv ID:** 2610.02866 | [PDF](https://arxiv.org/pdf/2610.02866v1)

**作者:** Kuan-Yu Chen `[一作]` (SinoPac Holdings), Tien-Hao Chang `[通讯]` (SinoPac Holdings)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种跨市场预训练框架 DIVINE，通过重建技术指标来学习金融序列的可迁移表示。

**💡 创新点**

创新点在于：① 用技术指标作为仅依赖历史的监督目标，既避免了未来结果的不确定性，又保持与回报预测的紧密对齐；② 证明指标多样性与市场多样性共同驱动跨市场迁移，强调监督设计与多市场学习的关键作用；③ 在保持极低参数量的同时实现与大型金融基础模型相当甚至更优的性能。

**🔧 技术方法**

技术包括：基于 OHLCV 的 77 维技术指标（16 种指标×5 周期）重建，使用 GRU/Transformer 等编码器，MSE 重建损失，随后仅迁移编码器到下游股票排名任务；同时对指标进行多阶段归一化与标准化。

**📊 数据集**

使用六个国际股票市场数据集：CSI300、CSI500、NI225、SP500、FTSE100、TWSE300；训练集 2008‑2019，验证 2020，测试 2021‑2024。

**📈 对比分析**

与从零训练的股票排名模型、通用时间序列预训练、金融预训练方法以及大规模金融基础模型（Kronos‑small、FinCast）进行对比。DIVINE 在平均 Sharpe Ratio（≈1.40）和 Calmar Ratio（≈1.80）上实现最优，且参数量仅 0.05M，显著优于 PatchTST、Kronos‑small 等基线。

**⚠️ 局限性**

局限性：① 对较小股票池（如 FTSE100）时，缺乏跨股票关系建模可能导致表现下降；② 技术指标的选择仍基于经验，未探索更广泛或自适应的指标集；③ 迁移到完全陌生市场时仍需要一定的目标市场信息，完全 OOD 迁移的效果有限。

---

## 377. Toward Omni Multimodal Graph Foundation Model: A Topology-Driven Binding Approach

**arXiv ID:** 2610.02881 | [PDF](https://arxiv.org/pdf/2610.02881v1)

**作者:** Xunkai Li `[一作]` (Beijing Institute of Technology), Guoren Wang `[通讯]` (Beijing Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于图拓扑的多模态图基础模型GraphBind，能够在节点属性缺失的真实图数据上进行预训练，并通过拓扑驱动的多模态绑定学习统一的表征空间；

**💡 创新点**

创新点在于：① 将图拓扑作为稳定的结构参考，用于可靠邻居筛选和全局语义校准；② 在预训练阶段模拟真实模态缺失，使模型能利用不完整图谱；③ 设计轻量级下游适配器（拓扑前缀提示器和G2M-Adapter），既保持预训练空间不变，又实现分类与生成任务的高效迁移；

**🔧 技术方法**

技术包括：模态缺失模拟、节点自语义初始化、邻居可靠性评分与加权聚合、拓扑编码与共享潜在空间校准、共享空间的轻量化下游适配；

**📊 数据集**

使用了OpenMAG基准中的八个多模态图数据集：Grocery、Movies、Toys、RedditS、DY、Bili_Dance、Flickr30k、SemArt；

**📈 对比分析**

与11类基线（MAG模型、GFMs、MGFMs）对比，GraphBind在节点分类、边预测、图到文本生成和图到图像生成等任务上均名列前茅，尤其在G2Text中BLEU‑4提升28.1%，在节点分类中MRR提升5.4%；

**⚠️ 局限性**

局限性包括：与轻量级MAG基线相比推理成本略高；预训练参数量仍显多，虽低于PLANET但相较于UniGraph2较大；实验范围聚焦于八个数据集，跨域泛化与极端模态缺失情况仍待进一步验证。

---

## 378. When Can We Trust the Matching Principle? Robust Deployment Geometry Under Finite-Sample and Model Uncertainty

**arXiv ID:** 2610.02894 | [PDF](https://arxiv.org/pdf/2610.02894v1)

**作者:** Vishal Rajput `[一作]` `[通讯]` (Independent researcher), Vishal Rajput (Independent researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于估计不确定性与谱分隔比值（trust ratio τ）的匹配准则，并将其实现为三路策略（匹配、软混合、均匀扩散）以决定是否沿估计方向进行正则化。

**💡 创新点**

创新点在于将 τ 作为决定信任与否的单一量化指标，证明在线性-二次匹配响应下投影匹配误差随 τ² 缩放，并提出了经验校准的 Confidence‑Calibrated Matching（CCM）策略，能在有限样本与模型不确定条件下自动避免总是匹配导致的失败。

**🔧 技术方法**

技术手段包括：线性-二次响应模型、Davis–Kahan 以及 Wedin 近似、协方差估计与稀疏化、以及对 τ 的阈值校准与三路策略的实现。

**📊 数据集**

使用的数据集包括合成尖峰网格（spike grid）、UCI HAR 行为嵌入、Fashion‑MNIST 旋转实验以及 Office‑31 域适配数据集。

**📈 对比分析**

与全协方差匹配、等方差扩散和单纯的匹配/不匹配基线相比，CCM 在多种实验设置下能显著降低平均 LQ 滞后（dual cost），尤其在 UCI HAR 上避免了总是匹配导致的性能恶化，并在 Fashion 与 Office-31 的可识别情形下实现匹配优势。

**⚠️ 局限性**

局限性包括：对谱间隙和协方差估计的依赖；在高维实测数据中环境维度浓度可能导致过度抑制；对非线性匹配、在线或大模型情形的适用性尚未证明；并且模型错配（误估几何族）会导致 τ 虽低但匹配仍失效。

---

## 379. Revealing Epistemic Uncertainty in MLLMs via Causal-Invariant Masking

**arXiv ID:** 2610.02887 | [PDF](https://arxiv.org/pdf/2610.02887v1)

**作者:** Haoyang Luo `[一作]` (City University of Hong Kong), Minjing Dong `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于因果不变掩码的多模态大模型不确定性量化方法，区分随机与模型本身的不确定性。

**💡 创新点**

创新点在于通过因果不变掩码测量语义偏移（Semantic Divergence）来捕捉模型的认知误差，并给出几何近似（Expected Embedding Drift）实现高效计算。

**🔧 技术方法**

核心技术包括RoI裁剪的因果掩码、能量模型理论推导、Kullback–Leibler 语义偏移度量、vMF 分布下的嵌入漂移估计。

**📊 数据集**

使用 InternVL2‑8B、Qwen2.5‑VL‑7B 作为模型，评测数据集包括 VQAv2、OKVQA、AdVQA、POPE、MME。

**📈 对比分析**

与语义熵、EigenScore、UMPIRE 等基线比较，在所有基准上均获得最高的 AUROC（如 AdVQA 76.8%、POPE 95.1%），并且 EED 速度提升约 50% 同时保持性能。

**⚠️ 局限性**

局限性在于对 RoI 检测器和语义等价性判断的依赖，若这些组件失效会导致不确定性评估失真。

---

## 380. PaxosLease in Relativistic Inertial Frames

**arXiv ID:** 2610.02879 | [PDF](https://arxiv.org/pdf/2610.02879v1)

**作者:** Márton Trencséni `[一作]` `[通讯]`, Márton Trencséni

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文在分布式租约协议 PaxosLease 的基础上，重新定义安全性属性，使其在相对论框架下无论相对运动的观测者都能一致判定租约是否安全，并对协议中的超时参数进行调整，使其满足该重定义的安全性。

**💡 创新点**

创新点在于：①将“任意时刻”安全性属性改为光锥（causal cone）内的条件，消除了相对论中时空观测者不同导致的安全性歧义；②在超时设计中引入相对论多普勒因子 k = √((1+β)/(1-β))，从而在参与者相对运动时保持租约的因果排斥；③通过新的包含（containment）和隔离（quarantine）规则保证即使在相对速度和时钟漂移的情况下也能实现安全和活性。

**🔧 技术方法**

使用的技术主要有：分布式系统中的 Paxos 协议机制、相对论光锥理论、相对论多普勒效应计算、时钟漂移与相对速度的上界假设以及超时调度。

**📊 数据集**

本文没有采用传统意义上的数据集；其验证主要通过理论证明和空间-时间图示（light cone diagrams）来说明改进后的协议满足重定义的安全性。

**📈 对比分析**

比较方法：作者对比经典 PaxosLease 与改进后的相对论 PaxosLease 在安全性（Causal Exclusion）和活性（Liveness）方面的差异，主要通过理论推导证明改进后满足因果排斥。性能上主要通过超时参数的放宽来保证活性，具体数值上使用相对论 Doppler 因子 k 调整排斥和隔离超时。

**⚠️ 局限性**

局限性：①协议假设所有命令以光速传播且资源按到达顺序执行；②仅考虑平坦时空（特殊相对论）且假设所有参与者保持匀速直线运动；③在强引力场或非惯性运动情况下的适用性未被探讨。

---

## 381. Custom Forcing: Training-Free Subject Customization for Autoregressive Video Generation

**arXiv ID:** 2610.02914 | [PDF](https://arxiv.org/pdf/2610.02914v1)

**作者:** Yunseung Ok `[一作]` (Kyung Hee University), Suhyun Kim `[通讯]` (Kyung Hee University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Custom Forcing，一种训练无关的自回归视频生成方法，通过把用户提供的参考图像写入 KV 缓存并在自注意力中调节其影响，能够在长时段（如 2 分钟）内保持视频中主体身份不漂移；

**💡 创新点**

核心创新在于仅修改 KV 缓存与自注意力机制，提出漂移自适应值放大（DVA）与锚对比引导（ACG），同时解决身份漂移与文本提示泛化问题，完全不需要微调或额外的条件网络；

**🔧 技术方法**

使用冻结的 Wan2.1‑T2V‑1.3B 自回归视频扩散模型，改造其自注意力层；利用 DINOv2‑B 评估漂移、SAM 生成主体掩码、FLUX.2 生成自定义锚图像；

**📊 数据集**

采用公开 DreamBooth 数据集，包含 10 个主题（6 个动物、4 个物体）和每个主题 10 个仅写类名的提示；

**📈 对比分析**

与双向定制方法（SMRABooth、CustomCrafter、MotionBooth 等）、图像‑到‑视频模型（FramePack、SkyReels‑V2）和参考‑到‑视频模型（SkyReels‑V3）进行对比；在 30 秒和 2 分钟视频中，Custom Forcing 的 DINO‑I 均保持 0.58‑0.62，远高于固定锚的 0.42，并且每帧生成速度比 14B 模型快 9.5‑28.5 倍；用户研究显示该方法更受欢迎；

**⚠️ 局限性**

局限性包括：仅适用于冻结的自回归模型，依赖高质量参考图像；在需要频繁更新主体或与动态背景交互的场景下效果不明；对未经授权的真实人物使用存在潜在误导风险。

---

## 382. ByteSplat: Efficient Distributed 3D Gaussian Splatting Training via Intra- and Inter-GPU communication reduction

**arXiv ID:** 2610.02851 | [PDF](https://arxiv.org/pdf/2610.02851v1)

**作者:** Shuo Wu `[一作]` (Shanghai Jiao Tong University), Minyi Guo `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一种分布式3D高斯散点渲染训练框架，旨在通过减少GPU内部与GPU间的数据传输来提升大规模场景训练效率。

**💡 创新点**

创新点包括：1）将前向与后向光栅化以及损失计算融合为单个GPU核，显著降低DRAM访问；2）基于硬件共享内存预算的可硬件感知剪枝，扩大可融合的图像块比例；3）零梯度剔除与稀疏梯度编码，实现梯度通信量大幅削减。

**🔧 技术方法**

主要技术手段有：mega‑kernel光栅化融合、共享内存利用与自适应分配、按图像块负载量计算的剪枝优先级、零梯度识别与位图压缩编码、GPU高效解码与直接原子累加、S3IM局部SSIM损失、稀疏all‑to‑all通信。

**📊 数据集**

实验数据集包含六个场景：Mill‑19、UrbanScene3D、MatrixCity、Mip‑NeRF 360、DeepBlending 与 Tanks&Temples，均采用原始分辨率训练与评估。

**📈 对比分析**

与Grendel‑GS基线在A6000与A100服务器上进行对比，使用八块GPU时平均加速分别为3.4×（A6000）与2.7×（A100），峰值可达6.1×；同时把GPU内部DRAM流量减少63.4%，后向梯度通信量减少65.8%，且PSNR/SSIM几乎不变。

**⚠️ 局限性**

局限性包括：在小规模场景中加速效果相对有限；剪枝比例需手动调节，过度剪枝会影响视觉质量；框架依赖现有GPU架构，难以进一步突破GPU共享内存限制；在超过八块GPU的规模下通信瓶颈仍待解决。

---

## 383. Adaptive Mutual Distillation for Balanced Multi-Task Post-Training of Large Language Models

**arXiv ID:** 2610.02856 | [PDF](https://arxiv.org/pdf/2610.02856v1)

**作者:** Baohang Li `[一作]` (Harbin Institute of Technology), Bing Qin `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Adaptive Mutual Distillation (AMD)，在多任务 LLM 的后训练阶段，让两台使用不同任务采样策略的模型通过自适应的双向知识蒸馏互相学习；

**💡 创新点**

创新点在于：①在同一训练阶段动态调整每个任务及每个转移方向的蒸馏权重；②利用共享的短期训练探针与验证反馈，独立为每个任务和方向选取最佳权重；③通过模型合并进一步提升单模型性能；

**🔧 技术方法**

技术核心包括：自适应蒸馏权重更新（logit 空间步长与sigmoid映射），短期探针评估，双向蒸馏目标，模型融合（MergeKit）以及对比学习的任务采样策略（比例采样与温度平滑采样）；

**📊 数据集**

数据集涵盖数学（NuminaMath、Math-500）、医学（MedQA、MedBullets）、法律（DISC-Law、LawBench、CMMLU-Law）与通用指令（InfinityInstruct），共约17万条样本；

**📈 对比分析**

与基于相同采样的 SFT 基线、固定权重蒸馏方案以及 PCGrad、FAMO、CoBa、MFTCoder、HBO 等单模型任务平衡方法对比，AMD 在所有三种 LLM 体系（Qwen3-0.6B、Llama-3.1-8B、Llama-3.2-1B）上均实现宏平均得分提升约 2.2–2.5 分，模型合并后平均提升约 2.9 分；

**⚠️ 局限性**

局限性包括：仅评估两台模型的合作，缺乏多模型扩展验证；仅针对后训练阶段，未探讨在预训练阶段的可扩展性；实验覆盖的任务类别有限，可能对更广泛领域的迁移性能未知；

---

## 384. Understanding Enrichment in Reinforcement Learning

**arXiv ID:** 2610.02846 | [PDF](https://arxiv.org/pdf/2610.02846v1)

**作者:** Jinwoo Kim `[一作]` (University of California-San Diego), Shraddha Barke `[通讯]` (Microsoft Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文从理论和实验两方面研究了在稀疏奖励的RLVR中引入“enrichment”（外部提示/引导）对训练的影响，并提出了一种基于Sequential Monte Carlo（SMC）的权重校正方法，减少了标准重要性采样导致的高方差。

**💡 创新点**

创新点包括：1) 将enrichment解释为对奖励的隐式重加权，并用scale、rotation、variance三种误差分量剖析其对梯度的影响；2) 推导并实现了SMC重采样机制，使权重的指数放大转化为线性方差累积；3) 在实际LLM（Qwen3-1.7B）与稀疏OpenMathReasoning子集上验证了未校正、SMC校正和未enrichment三种策略的学习曲线及“collapse”现象。

**🔧 技术方法**

使用的技术主要包括：REINFORCE/GRPO框架、重要性采样权重、SMC重采样（particle filtering）以及基于token级别的权重累积和截断。实验中还使用了LLM推理、分布式训练和统计显著性检验。

**📊 数据集**

数据集为OpenMathReasoning的一个稀疏成功率约为2%的500题子集（训练），并保留300题做held‑out评估；使用预训练的Qwen3-1.7B作为基线模型。

**📈 对比分析**

比较方法：在相同训练步数（500步）下，分别训练三种策略，并记录pass@1、pass@k、典型性等指标。结果显示：1) 两种enrichment策略最终都能提升pass@k并避免RL的“collapse”；2) 早期训练阶段未enrichment表现更好；3) SMC校正虽在统计上不显著优于未校正，但在避免低pass@k下降方面表现更稳健。

**⚠️ 局限性**

局限性：1) 权重校正的方差仍受样本长度影响，SMC方法虽降低方差但增加了采样与重采样的计算开销；2) 仅在单一LLM+单一任务上验证，缺乏跨任务泛化的证据；3) 对float32的数值稳定性仍存在挑战，尤其在极长序列时；4) 代码与完整实验细节尚未公开，复现性受限。

---

## 385. SceneFactory-3D: Lifting 2D Traffic Scenes into 3D Physical Counterfactuals for Scalable Physically Grounded Safety Evaluation

**arXiv ID:** 2610.02874 | [PDF](https://arxiv.org/pdf/2610.02874v1)

**作者:** Yicheng Zhu `[一作]` (Rochester Institute of Technology), Zilin Bian `[通讯]` (Rochester Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文开发了 SceneFactory‑3D，一款基于 GPU 批处理、物理驱动的多车驾驶仿真器，支持每个车轮的力学计算、空间可变摩擦与三维地形，并可在相同交通场景下对不同道路物理条件进行匹配的因果对照实验。

**💡 创新点**

创新点包括：
1) 引入基于车轮接触点的阻尼、摩擦、驱动力模型，实现更真实的路面‑轮胎交互；
2) 采用 per‑world 地形隔离和 GPU 批处理，使同一交通场景可在多套物理环境中并行运行，实现可匹配的物理 counterfactual 评估；
3) 将经典规划器与强化学习策略在统一的物理平台上进行对照，揭示摩擦/坡度对交通安全与任务完成率的闭环影响。

**🔧 技术方法**

主要技术包括：
- Isaac Lab 与 NVIDIA PhysX/XPBD 进行物理仿真；
- 自定义 Warp kernel 计算每轮胎的力学响应；
- GPU 批处理（CUDA Graphs）实现数千个并行世界；
- 统一的 canonical 路径接口、路面摩擦贴图与三维高度场。

**📊 数据集**

数据集与实验配置：
- 21 套摩擦与坡度条件（含 8 级摩擦、8 级坡度和 4 组组合）；
- 对每套条件进行 1,024 个匹配的 12 车世界，涵盖 12 种初始速度与车道配置；
- 训练 5 组控制器（2 个经典规划器 + 3 个 PPO 训练策略，分别为 nominal、terrain‑diverse、terrain‑diverse+preview）。

**📈 对比分析**

比较方法：
- 采用“匹配实验”框架，将相同交通场景与控制器在不同物理条件下并行执行；
- 量化指标包括：valid passage（通过率）、collision / off‑road / lane incursion 率、TTC（与前车相对时间）分布；
- 结果显示：摩擦下降至 0.18 时，learned 策略的通过率可从 90% 降至 0%（幅度 6–90%），经典规划器约 18–19%；低摩擦下近碰撞事件显著增加；坡度对通过率影响有限。
- 性能方面：在单一 GPU 上，1,024 个 12 车世界可实现约 0.2–0.3 百万车轮步骤/秒，内存峰值 8–12 GB，扩展性近线性。

**⚠️ 局限性**

限制与挑战：
- 模拟器未进行真实道路校准，安全性指标仅在仿真域内有效；
- 仅测试单一封闭路段与单一车辆架构，难以泛化至更复杂场景；
- 训练种子数有限（4 组），导致不同种子间波动大，难以确定策略优劣；
- 预览特性仅为无噪声的未来摩擦/坡度信息，未考虑感知误差或记忆机制；
- 对比实验在全局重置与批量构成上可能导致结果混淆，需进一步细化统计方法。

---

## 386. BISCEPTER: Probability-Driven Bisection for Large-Scale System Software

**arXiv ID:** 2610.02995 | [PDF](https://arxiv.org/pdf/2610.02995v1)

**作者:** Mingyan Gao `[一作]` (University of Hong Kong), Zhendong Su `[通讯]` (ETH Zürich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于历史 BIC 延迟的概率驱动二分搜索方法，在标准中位数二分的基础上改进 pivot 选择，从而减少调试迭代次数。

**💡 创新点**

利用实验发现 BIC 强烈集中在最近 commit，构造以历史 BIC 延迟为权重的 weighted‑median pivot，并加入自适应回退阈值，使二分更贴合实际 BIC 分布。

**🔧 技术方法**

概率驱动权重中位数二分、历史 BIC 延迟统计、前缀和加速权重求和、基于阈值的自适应回退机制。

**📊 数据集**

8,172 条真实 BIC 记录，来自 GCC、Linux kernel 与 MariaDB 三大开源项目。

**📈 对比分析**

与标准中位数二分对比，平均迭代次数减少 25.75%（最多 55.55%），在 91.26% 的测试案例中性能更优；鲁棒性实验表明在历史记录噪声、缺失或重复时仍保持稳定。

**⚠️ 局限性**

方法依赖历史 BIC 数据的完整性与代表性；在分支合并繁多、merge 历史压缩或提交策略差异较大的项目中可能效果下降；未对多语言构建成本差异进行进一步分析。

---

## 387. RASPER: Reward-Aligned Summarization of Clinical Notes for EHR Outcome Prediction

**arXiv ID:** 2610.02979 | [PDF](https://arxiv.org/pdf/2610.02979v1)

**作者:** Arya Hadizadeh Moghaddam `[一作]` (University of Kansas), Zijun Yao `[通讯]` (University of Kansas)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

RASPER提出了一种奖励对齐的摘要框架，利用LLM生成结构化的住院笔记摘要并通过下游预测反馈进行强化学习，以提升EHR预测性能。

**💡 创新点**

创新点在于将摘要生成与预测任务直接耦合，使用预测损失作为奖励；引入基于时间的软提示编码器将结构化代码与摘要融合；并通过SFT+RLPF实现任务特定的摘要优化。

**🔧 技术方法**

技术包括Gemma-2-2B LLM+LoRA适配器、RETAIN时间编码器、Soft Prompt + Prompt Tuning、Proximal Policy Optimization（PPO）强化学习以及自监督训练（SFT）。

**📊 数据集**

使用公开的MIMIC‑III与MIMIC‑IV两大ICU电子病历数据库，涵盖多次住院记录，进行读入院预测与药物推荐任务。

**📈 对比分析**

与传统深度EHR模型（Deepr、RETAIN等）及最新LLM方法（GraphCare、LINKO、RePrompT）对比，RASPER在两数据集上读入院AUROC 0.704/0.725、PRAUC 0.738/0.742、药物推荐F1 0.388/0.282等指标均取得最优或接近最优成绩。

**⚠️ 局限性**

局限性包括对大型LLM计算资源的依赖；奖励信号受预测器质量影响；摘要生成仍可能遗漏罕见但重要信息；缺乏对实时临床可解释性的系统评估。

---

## 388. Engineering Sustainable Agents: A Systematic Comparison of Agentic LLMs for Developer Workflows

**arXiv ID:** 2610.03010 | [PDF](https://arxiv.org/pdf/2610.03010v1)

**作者:** Merve Astekin `[一作]` (SINTEF), Hui Song `[通讯]` (SINTEF)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对五种软件工程任务（代码生成、技术债务识别、代码漏洞检测、日志解析、日志分析）中，从非代理到多代理的多种LLM配置进行综合实验，评估其准确性、推理延迟和能耗。

**💡 创新点**

系统量化多代理复杂度与能耗之间的权衡，发现多代理设计导致能耗和延迟显著提升，而准确性提升仅在漏洞检测任务中可见，提出以任务为导向的可持续设计准则。

**🔧 技术方法**

采用六种开源权重LLM（Gemma、CodeGemma、Qwen、Qwen-Coder、DeepSeek-Coder、gpt-oss），两种提示策略（零样本、少样本），AG2/AutoGen式多代理框架以及CodeCarbon能耗监测。

**📊 数据集**

使用公开基准数据集 HumanEval、MLCQ、PrimeVul、LogHub HDFS‑v1 等。

**📈 对比分析**

通过实验对比四种代理级别、三种硬件平台（Server‑SMU、Workstation‑SMU、Server‑STF）和两种提示策略，发现多代理配置平均能耗提升 6.36 倍、延迟提升 6.07 倍，准确性提升仅在漏洞检测任务中显著；Pareto 分析显示轻量级配置占主导。

**⚠️ 局限性**

局限性包括仅覆盖五个任务、模型规模限制（≤20B）、实验仅在本地硬件上，未考虑云部署和更大模型，以及任务特定最佳代理设计可能未被探索。

---

## 389. Evolutionary Computation for Trustworthy AI: From Attacks and Defenses to Self-Evolving Era

**arXiv ID:** 2610.02996 | [PDF](https://arxiv.org/pdf/2610.02996v1)

**作者:** Junhao Dong `[一作]` (Nanyang Technological University), Yew-Soon Ong `[通讯]` (Nanyang Technological University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述了进化计算（EC）在可信 AI 领域的应用，系统梳理了进化攻击、进化防御以及可信自进化 AI 三个方向，并提出统一的 EC 视角框架和分类体系。

**💡 创新点**

创新点在于将可信 AI、进化计算与自进化系统三条研究线索通过进化视角统一连接，构建了完整的分类框架，汇总了评估方法与基准资源，并对搜索效率、泛化性、评估可靠性及安全持续适配等开放挑战进行了深入讨论。

**🔧 技术方法**

主要技术包括梯度无关优化、进化多目标优化、保持多样性的种群搜索、协同进化与自适应演化，以及对约束、验证和记忆/技能管理的结构化处理。

**📊 数据集**

作为综述文章，本文引用了多种公开数据集，如 ImageNet、COCO、MNIST、CIFAR、GLUE、SuperGLUE、OpenAI 的 GPT、RAG 数据集等，收集并整理了这些数据集在进化攻击与防御中的使用情况。

**📈 对比分析**

文章通过比较不同研究中使用的进化算法、评估指标和基准，呈现了攻击有效性与防御鲁棒性之间的权衡，并总结了在多目标和多样性方面的最新性能趋势，指出在大规模模型和多任务环境下进化方法仍面临挑战。

**⚠️ 局限性**

局限性包括：综述范围依赖现有公开文献，缺乏统一的评估平台与基准；对大规模模型的进化搜索仍缺乏高效策略；缺少针对长期自进化系统安全与可解释性的系统化研究；以及在真实世界动态环境中验证效果的困难。

---

## 390. An Applicative Multiset Path Order (Extended Version)

**arXiv ID:** 2610.02973 | [PDF](https://arxiv.org/pdf/2610.02973v1)

**作者:** Nao Hirokawa `[一作]` (JAIST), Wataru Yachi `[通讯]` (JAIST)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种用于无类型应用式词项重写的多集路径顺序变体（AMPO），能够证明如map、filter等高阶函数的终止性。

**💡 创新点**

创新点在于引入了阶数分配（arity assignment）和重现化（reification）两种机制，并通过η-限制构造了一个闭合于上下文且满足子项性质的基序。

**🔧 技术方法**

主要技术包括：递归定义的基序、词项重现化、η-限制、基于基序的闭合上下文扩展、以及利用地面替换构造的可降顺序。

**📊 数据集**

在TPDB（Termination Problem Database）共202条ATR系统上进行实验，自动寻找合适的阶数分配与优先级。

**📈 对比分析**

与LPO、EPO、未卷曲化等传统多集/词路径顺序相比，AMPO在143条左头变量无的ATR系统中证明了24条终止性，虽不如AProVE和NaTT，但在部分高阶规则上显示出更强的表达能力。

**⚠️ 局限性**

局限性包括：无法处理包含部分/重复变量应用的规则（如f(fx)→f(g(fx))）、对优先级的最大化要求导致某些规则无法被定向、以及缺乏对给定阶数分配和优先级的判定决策过程。

---

## 391. Reasoning with Evidence, Not Merely Rationales: Verifiable Preference Proofs for LLM-Based Recommendation

**arXiv ID:** 2610.02968 | [PDF](https://arxiv.org/pdf/2610.02968v1)

**作者:** Yu Hou `[一作]` (Yonsei University), Hua Li `[通讯]` (Yonsei University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种可验证偏好推理框架PROVE-Rec，能够把历史交互转化为结构化的偏好证明并以此为依据进行推荐

**💡 创新点**

创新点在于引入“证据-证明-影响”三段验证目标，确保生成的偏好声明既与所引用的历史证据高度关联，又能显著影响最终推荐结果，同时通过排名保持目标保持整体性能

**🔧 技术方法**

使用大规模生成式LLM推荐模型（基于TIGER/LC-Rec的语义编码器）实现两阶段推理，并采用softplus对比损失、KL约束和排名保留正则化

**📊 数据集**

在Amazon Grocery、Tools以及Yelp三个包含丰富用户评论的数据集上进行实验

**📈 对比分析**

与传统序列模型、基于语义标识的LLM推荐以及多种强化学习/注意力增强方法对比，PROVE-Rec在Recall@5/10和NDCG@5/10上分别提升2–7.5%，在所有数据集上均领先最强基线

**⚠️ 局限性**

局限性包括需要丰富的评论文本，推理过程增加额外计算开销（尤其是首次生成证明时），且在非评论丰富的场景下效果可能下降，且偏好证明可能暴露用户敏感信息

---

## 392. Post-Training Frontier Text-to-Image Models by Composing Preference and Rubric Rewards

**arXiv ID:** 2610.02967 | [PDF](https://arxiv.org/pdf/2610.02967v1)

**作者:** Yuanhao Ban `[一作]` (Arena Intelligence Inc), Cho-Jui Hsieh `[通讯]` (Arena Intelligence Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究开发了一种简单有效的后训练配方，用于开放域文本到图像生成，基于互补奖励信号的组合。

**💡 创新点**

创新点在于提出了一种奖励系统，包括偏好奖励和基于评分的奖励，能够更全面地捕捉人类的审美和感知偏好，同时防止奖励被模型利用。

**🔧 技术方法**

使用了布拉德利-特里偏好模型和多种基于评分的奖励机制，结合了强化学习技术。

**📊 数据集**

使用了大规模的人类偏好数据集，包含约500万对人类偏好投票。

**📈 对比分析**

与现有方法相比，后训练的FLUX.2-dev模型在Arena文本到图像排行榜上获得了比基础模型高出69分的Elo评分，而后训练的Ideogram-4模型超越了所有开源模型，达到了1223.5的Elo评分。

**⚠️ 局限性**

限制在于仅在两个基础模型上进行了评估，未能全面评估该方法在不同模型家族和规模上的适用性；此外，分解的忠实性问题未经过人工审核，可能存在错误或模糊性。

---

## 393. Hyperparameter selection for equation learning with biologically-informed neural networks

**arXiv ID:** 2610.02954 | [PDF](https://arxiv.org/pdf/2610.02954v1)

**作者:** William Lavery `[一作]` (Uppsala University), Sara Hamis `[通讯]` (Uppsala University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一套无接地真理的诊断工作流程，用于在生物信息神经网络（BINN）中系统性地选择网络容量（宽度/深度）和早停耐心等关键超参数。

**💡 创新点**

核心创新在于：①将验证损失与已学习的右端项函数一致性作为无真值的选择依据；②构建基于网络数据流的半顺序调优流程；③归纳出四条可直接迁移的经验规则（宽度比深度更重要、状态网络比右端项网络更宽、保持右端项网络窄、早停需同时查看所有函数的一致性）。

**🔧 技术方法**

技术手段包括：BINN架构（状态 MLP + 扩散/生长 RHS MLPs）、联合损失（数据、PDE、约束）、随机训练/验证拆分、早停、宽度/深度超参数搜索、成本–性能曲线、训练动态图、函数一致性图。

**📊 数据集**

实验使用合成的 1D+t 与 2D+t 反应扩散数据，涵盖常数、线性、二次、指数扩散/生长，并在 2D+t 设定中加入 5% MPE 噪声；无真实实验数据。

**📈 对比分析**

通过对比验证损失、函数一致性与已知真值，验证超参数选择可使学习的扩散与生长函数在 MPE ≤ 5% 内逼近真值，且训练成本保持可接受；在噪声和高维情形下依旧能保持相近精度。

**⚠️ 局限性**

局限性：①仅在合成数据（最高 5% 噪声）上验证，缺乏对实验数据和更高噪声、模型错配的鲁棒性评估；②采用半顺序搜索而非全局联合优化，可能漏掉更优配置；③超参数判断仍需人工阅读诊断图，缺乏统一阈值。

---

## 394. Discriminating Fixture Coverage in Agent-Infrastructure Verification Suites

**arXiv ID:** 2610.02928 | [PDF](https://arxiv.org/pdf/2610.02928v1)

**作者:** Xin Xu `[一作]` (Carnegie Mellon University), Siru Tao `[通讯]` (Carnegie Mellon University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文评估并改进了多会话代理状态投影层的验证套件，通过变异分析揭示验证套件在证据缺失方面的不足，并基于外部指定的变异者提出补丁并验证；

**💡 创新点**

创新点在于首次将验证套件冻结后对外部指定的变异者进行挑战测评，揭示单一通过/失败观察证据的价值有限，并将失效分为“未激活”与“内部掩蔽”两种模式，随后通过输入维度预测实现补丁；

**🔧 技术方法**

使用的技术包括变异测试（mutation testing）、基于场景的 invariant 检查、故障注入、内部状态记录以及输入维度覆盖分析；

**📊 数据集**

使用的数据集为人工构造的多会话状态投影层参考实现及其 11/12 条 invariant 检查，外加 10 个由攻击者指定的第一阶变异者，以及 12 个固定场景，后续扩展到 17 条检查并新增 5 个 fixture；

**📈 对比分析**

通过比较标准验证（reference vs all-defects）通过/失败率、冻结套件对外部变异者的杀伤率（5/10）以及修复后杀伤率（10/10）来评估方法，结果表明原始验证套件的证据价值有限，修复后检测能力显著提升；

**⚠️ 局限性**

局限性包括仅在人工合成实现上测试，使用手写的一阶变异者，外部挑战集可能不代表真实错误，输入维度覆盖仍可能不完整，缺乏对真实系统的验证。

---

## 395. When Predicting Nothing Beats SAM 3: Revisiting Evaluation in Video Object Segmentation

**arXiv ID:** 2610.02946 | [PDF](https://arxiv.org/pdf/2610.02946v1)

**作者:** Jihwan Hong `[一作]` (Seoul National University), Jaeyoung Do `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出低时间可见度下的视频目标分割评估问题，构建了FaVOS基准并引入体素化J&F指标。

**💡 创新点**

创新点在于揭示传统J&F在低可见度时退化为缺失检测，并提出体素化J&F以减弱此效应。

**🔧 技术方法**

技术手段包括指标分解、体素级Jaccard与边界精度、以及多模型评测。

**📊 数据集**

使用FaVOS-20/FaVOS-40（200视频281物体）以及DAVIS、YouTube-VOS等公开数据集。

**📈 对比分析**

与现有J&F、tIoU、T等指标比较发现传统J&F易被空掩预测压倒，体素化J&F能更好区分模型，显著提升对低可见度场景的评估公平性。

**⚠️ 局限性**

局限性包括基准规模有限、体素权重可能偏向大物体、以及对少量假阳性容忍度高。

---

## 396. Output Language Confusion under Multilingual Prompt Contamination

**arXiv ID:** 2610.02926 | [PDF](https://arxiv.org/pdf/2610.02926v1)

**作者:** Riju Marwah `[一作]` (University of Tübingen), Amit Sheth `[通讯]` (Indian AI Research Organization)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Multilingual Distractor Interference（MDI）评估协议，探究在英文事实问答任务中加入无关外语句子对模型回答语言、准确性和自我否定行为的影响；

**💡 创新点**

创新点在于：① 设计轻量可复现的MDI协议；② 揭示精确匹配评估在混语环境下因脚本切换而导致的幻觉误判；③ 构建脚本切换、主动否定和稳定三类失败模式的分类法；

**🔧 技术方法**

技术方法包括：在问题前添加指定的西班牙语、印地语（梵文）和汉语（简体）句子（可重复三次），使用多语言字母表检测脚本切换；对TruthfulQA采用概率评分，TriviaQA采用greedy生成并手动校正脚本切换误判；采用McNemar检验对比基线；

**📊 数据集**

数据集使用TruthfulQA mc1验证集（500题）和TriviaQA验证集（500题），不收集新数据，外语句子为固定短句，长英文段落来自WikiText-2；

**📈 对比分析**

对五款公开指令调优模型（Llama‑3.1‑8B、Llama‑3.2‑3B、Phi‑3‑mini、Mistral‑7B、Qwen‑2.5‑3B）在八种干扰条件下进行对比：TruthfulQA多选不受干扰；TriviaQA中Llama‑3.1‑8B出现脚本切换，原始幻觉率0.710（调整后0.470）；其余模型表现为否定率上升；长英文段落导致所有模型近乎全否定；

**⚠️ 局限性**

局限性包括：仅评估五款模型，未覆盖封闭源或最新版本；干扰句子明确声明无关，未模拟更自然的混杂情况；仅使用单GPU float16 推理；否定检测基于子串匹配，可能漏检非标准否定；

---

## 397. OmniConfess: Eliciting Token Confessions to Mitigate Omni-Modal Hallucination

**arXiv ID:** 2610.02999 | [PDF](https://arxiv.org/pdf/2610.02999v1)

**作者:** Huiqiang Rong `[一作]` (Beijing University of Posts and Telecommunications), Yifan Zhu `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 OmniConfess，一种训练无关、推理时纠正多模态大语言模型幻觉的方法。

**💡 创新点**

通过固定候选回答、逐通道证据干预、token 级别的 confessions 结构化揭示证据依赖并据此纠正错误，实现对不相关或矛盾证据的自动识别与校正。

**🔧 技术方法**

使用 commitment anchoring、evidence interrogation（单通道干预与对比概率计算）以及 confession‑guided correction 的三阶段框架；利用 token‑级别的依赖量化与阈值化进行局部修正。

**📊 数据集**

构建 OmniHalluBench（3540 条样本），聚合六个公开数据集：CMM、PhD、PubMedQA、RAGTruth、HaloQuest 与 AVHBench。

**📈 对比分析**

在 Qwen2.5‑Omni‑7B、Qwen3‑Omni‑30B‑A3B 与 Nemotron‑3‑Nano‑Omni‑30B 三种骨干上，与搜索、证据对比、细粒度等十余种推理时方法比较；OmniConfess 在所有数据集上平均 F1 提升 >10 点，跨骨干提升分别为 10.3、6.3 与 7.5 点，显著优于现有基线。

**⚠️ 局限性**

主要限制是推理时需对每条证据通道进行多次重新评分，导致多模态任务下显著延迟；且方法依赖候选答案冻结，可能限制生成多样性。

---

## 398. Dirac-Interconnected Neural Elements: Discovering Modularity in Physical Systems Without Reduction

**arXiv ID:** 2610.02960 | [PDF](https://arxiv.org/pdf/2610.02960v1)

**作者:** Reiho Li `[一作]` (Hokkaido University), Takashi Matsubara `[通讯]` (Hokkaido University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 Dirac-interconnected neural elements (DINEs)，一种基于差分-代数方程（DAE）的物理系统数据驱动建模方法。

**💡 创新点**

创新点在于：①利用 Dirac 结构在核表示中精确参数化系统互连，既保留代数约束又可学习部件特性；②实现部件可隔离、可复用并可在不同物理域组装；③支持部分可观测系统，无需先验约束或降阶。

**🔧 技术方法**

技术包括：1）用神经网络拟合存储元件的能量函数 H_i(z_i) 与阻尼元件的关系 e_R=f_R；2）对 Dirac 结构采用可分离、常数参数化 F,E = [W_f 0; 0 W_e]，只需指定流约束数 n_f；3）在训练时使用轨迹网络 χ_η 近似解 DAE，损失为数据拟合与 DAE 残差之和；4）预测时采用 Radau IIA 五阶求解器。

**📊 数据集**

使用五个合成实验任务的数据集：①三叉接点、②RC 梯形、③刚性耦合、④液压-机械耦合、⑤弹簧复用，每个任务均以数值仿真生成时间序列。

**📈 对比分析**

与多种基线（Neural ODE、PoDiNN、CT‑SUBNET、Coupled Neural ODE）对比，DINEs 在所有任务上实现了最低均方误差（MSE）和最长有效预测时间（VPT），尤其在涉及代数约束和部分可观测场景中表现显著优于现有方法。

**⚠️ 局限性**

局限性包括：①需预先指定流约束数 n_f，若设定错误会导致收敛失败；②当前仅支持可分离的 Dirac 结构，不能处理含齿轮器或状态耦合系数的系统；③模型对尺度（Gauge）有自由度，需要额外的读出或校准步骤。

---

## 399. Subject-Specific Predictive Musculoskeletal Simulations of Lower-Limb Exoskeleton Assistance: Metabolic and Biomechanical Effects of Joint Assistance Strategies

**arXiv ID:** 2610.02991 | [PDF](https://arxiv.org/pdf/2610.02991v1)

**作者:** Neethan Ratnakumar `[一作]` (University of Jaffna), Xianlian Zhou `[通讯]` (New Jersey Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

使用基于 OpenSim Moco 的直接离散化预测模拟，评估六个被动伸缩指数（BMI）不同的体型在单关节、双关节和三关节（臀、膝、踝）低腿外骨骼助力组合下，两个峰值扭矩水平（25 Nm、50 Nm）对步态的能量、关节动力学和肌肉激活的影响。

**💡 创新点**

首次系统化评估所有可能的臀、膝、踝助力组合，并将体型与肌肉力量按 BMI 进行个性化缩放，利用直接离散化优化框架在能量优化目标下预测最优助力策略，揭示膝关节助力在能量节约上的非直观“正功”策略及其对多关节助力增益的边际递减。

**🔧 技术方法**

采用 OpenSim Moco 的直接离散化（direct collocation）优化，使用 DeGroote‑Fregly 平滑肌肉模型；目标函数结合代谢能量、肌肉努力、关节与 COM 加速度、运动跟踪和追踪误差；理想扭矩执行器模拟外骨骼助力，约束运动学跟踪以实验步态为参照。

**📊 数据集**

基于公开的 Camargo 等人步态数据库（含六名不同身高、体重、BMI 的受试者）提供轨迹跟踪和关节动力学基线；通过 OpenSim Scale Tool 将通用模型按受试者尺寸与体重缩放为六个个体化模型。

**📈 对比分析**

比较方法：对每种助力组合计算成本运输 (COT) 降低百分比，并使用 Friedman、Wilcoxon 等非参数检验评估显著性。结果显示：在 50 Nm 下，三关节 H+K+A 助力可实现约 48.5 % 的 COT 降低；双关节 H+A 仅略低（≈44.6 %），单关节膝助力效益最小；膝关节在 50 Nm 下几乎无增益，显示出正功策略；总体而言，膝助力对多关节增益贡献有限。

**⚠️ 局限性**

局限性：模型采用理想扭矩、有限肌群、刚性肌腱和简化接触动力学，导致对真实人体代谢和关节力学的过度估计；预测结果高度依赖目标函数权重和假设；未进行实验验证；个体化助力对不同体型的鲁棒性未充分评估。

---

## 400. Neural Data Needs Semantic Tokenization: Behavioral Events as Boundaries of Session-Transferable Tokens

**arXiv ID:** 2610.03001 | [PDF](https://arxiv.org/pdf/2610.03001v1)

**作者:** Sangyoon Bae `[一作]` (Seoul National University), Jiook Cha `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

设计了一种新的 tokenizer —— Tokenization with States（TWS），将行为任务中的各个状态（即任务事件之间的时间段）转换为无神经元/会话标签的低维 tokens，并以此作为跨会话神经基础模型的输入。

**💡 创新点**

创新点：1）通过把 population manifold 的子空间作为 token，突破了传统 per‑neuron token 导致的会话外推失败；2）token 仅依赖于任务事件，无需对新会话做对齐或额外训练；3）token 空间在不同动物、不同记录技术间保持一致，实现了强大的跨物种和跨平台迁移；4）自监督预训练（mask 还原 + 状态子空间正交）进一步降低了会话识别信息。

**🔧 技术方法**

技术手段：软 sigmoid 门分段状态、无位置编码的交叉注意力聚合、Gram‑Schmidt 正交化得到 Grassmannian token、两维子空间（k_dim=2）token、卷积/Transformer/Mamba 等多种 backbone、线性 probe 解码、以及自监督预训练任务。

**📊 数据集**

使用的数据集：IBL Brain Wide Map（243 Neuropixels mouse sessions）、Steinmetz 2019 Neuropixels（10 mouse sessions）、Gallego 2022 Utah array macaque sessions、NLB MC_Maze Utah pseudo‑sessions，以及 53 个 held‑out IBL 会话。

**📈 对比分析**

比较方法：与 per‑neuron tokenizer（POYO, CEBRA, NEDS）、PerceiverIO、PCA（对齐/不对齐）、Population Statistics、事件时间 baseline 等对比。在跨会话 IBL 任务上，TWS 运动解码 MCC≈0.56、反应时间 R²≈0.40，显著优于所有基线；在其他数据集迁移时，TWS 通过线性 probe 在目标任务上获得最高分（例如 reach direction MCC≈0.23），相比仅用事件时间的解码提升巨大；在仅使用 5 个有标签会话时，TWS 的解码效果比全部会话的基线高数倍。

**⚠️ 局限性**

局限性：1）只能保留跨会话共享的 Type A 变量，对 Type B 变量（如 Choice、Block）解码效果差；2）依赖已知的任务事件时间；3）丢弃了状态内的时间动态信息；4）状态分割不准会显著降低性能；5）在极少神经元或高噪声条件下的鲁棒性尚未充分验证。

---

## 401. Positive-Unlabeled Learning for Agent Safety False Alarm Auditing

**arXiv ID:** 2610.02925 | [PDF](https://arxiv.org/pdf/2610.02925v1)

**作者:** Xichen Yan `[一作]` (Jinan University), Lixu Wang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种两阶段的正负样本不平衡（PU）排序框架，用于在语言模型代理的安全监控中优先筛选假警报，解决监控导致的正样本选择偏差问题。

**💡 创新点**

创新点在于（1）信任感知PU监督，将已验证的安全轨迹迁移到报警域并降低对可疑假警报的负压；（2）可靠性门控的排名蒸馏，聚合多个PU参考模型的一致排序；（3）共识引导的结构化细化，利用分层安全参考支持和报警间关系进一步提升排序质量。

**🔧 技术方法**

主要技术包括正负样本不平衡学习（PU）、排名蒸馏、分层支持投影、关系约束优化以及对齐/混合损失的分布对齐与表示混合。

**📊 数据集**

实验使用 ATBench、TraceSafe 以及 Agent‑SafetyBench 三大基准数据集，覆盖 AgentDoG、Granite Guardian、StepGuard 等主流安全监控器。

**📈 对比分析**

与八种传统 PU 方法、原生监控排序及零样本 LLM 再评估方法对比，宏观 AUPRC 达到 0.6444，较最强基线提升 5.27–16.98%，在 5% 审核预算下假警报恢复率提高 33.3%。

**⚠️ 局限性**

局限性包括对已验证安全样本规模的依赖，需在固定监控器下训练，且对不同监控器的迁移效果与泛化性尚未完全验证。

---

## 402. The Coverage Depth Problem in Distributed DNA Data Storage

**arXiv ID:** 2610.02931 | [PDF](https://arxiv.org/pdf/2610.02931v1)

**作者:** Xiangliang Kong `[一作]` (Chinese Academy of Sciences), Tolga M. Duman `[通讯]` (Bilkent University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建并分析了分布式 DNA 存储中的全信息恢复问题，引入了分布式覆盖深度概念，对线性码的分布式读数进行了定量评估，证明 MDS 码在最小化期望读取次数方面仍然是最优的，并对简单形码（Simplex Code）给出了最优分区布局及其覆盖深度的精确上界。

**💡 创新点**

创新点包括：①提出分布式覆盖深度理论框架，统一描述单容器与多容器读取过程；②给出 MDS 码分布式读数的精确表达式与最优布局构造；③在低码率下给出下界并在高码率下构造更优分区；④对简单形码证明了分布式覆盖深度最优、构造阶梯式一轮完美布局，并揭示了平衡容器对读取总量的有限提升；⑤对现有方法的局限性提出未来研究方向。

**🔧 技术方法**

主要技术包括：组合几何与概率论（如 q-二项式定理、负相关随机变量集中不等式）、随机分区与大偏差分析、极限分布与几何阶段分解、连续映射定理与弱收敛论证。

**📊 数据集**

未使用实验数据集，全部结果来自理论证明与符号计算。

**📈 对比分析**

通过对比单容器模型与平衡容器模型下的期望读取次数，发现：在常数大小容器时可实现总读取量的减小；在多项式规模容器时仅提升并行速度但不降低总读取量；在简单形码中平衡布局提升有限，整体提升不超过 1.61 次读取。性能指标为期望读取次数与覆盖深度，理论上证明最优性。

**⚠️ 局限性**

局限性包括：①对小域（q < n-3）下无 MDS 码时无法给出最优分布式布局；②仅考虑理想的均匀抽样与无错误读取，实际系统中的异质性、噪声、异步读取等因素未被建模；③平衡容器的存在性受参数限制，某些 M 与 k 的组合无法构造平衡分区；④对中间容器数（M < k）在部分参数区间内仍未给出最优解，留有开放问题。

---

## 403. Dynamic Expert Pruning for Multi-Agent Systems

**arXiv ID:** 2610.02951 | [PDF](https://arxiv.org/pdf/2610.02951v1)

**作者:** Jabin Koo `[一作]` (Pohang University of Science and Technology), Jungseul Ok `[通讯]` (Pohang University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出动态专家剪枝（DEP）方案，在多代理系统中通过仅利用系统和任务提示文本预测每个请求的专家子集，从而在保持模型冻结的前提下实现高效稀疏推理。

**💡 创新点**

核心创新是把专家子集的选择从静态校准迁移到一次性前向预测；仅凭提示文本即可生成专门化的掩码，无需额外校准或训练回调，且能适应未见工作流与任务。

**🔧 技术方法**

技术包括：小型文本编码器+MLP预测器、REAP与自蒸馏双重训练信号、基于专家重要性与分布偏差的损失、Delta加载实现稀疏显存与交换。

**📊 数据集**

使用多任务工作流数据（包括数学推理、代码生成、知识问答等）以及不同规模和架构的MoE模型（Qwen3-30B/480B、DeepSeek-V2.5-236B、Qwen3-Coder等），并在SWE‑bench Verified等基准上进行评估。

**📈 对比分析**

与多种静态剪枝与专家合并基线（频率、REAP、HC‑SMoE）在50%与25%保留率下比较，DEP在绝大多数基准上均取得最高整体准确率，特别是在稀疏度较高时优势最显著；同时实现近线性内存缩减与约1%推理延迟。

**⚠️ 局限性**

局限性包括：需要单次预测器训练，且对底层MoE模型的容忍度需预先测定；在极端稀疏（如25%）时部分基线已失效；对未知工作流的泛化虽然显著，但仍受提示质量与模型规模的影响。

---

## 404. CreateScore: Domain-Theory-Informed Bayesian Routing for LLM-Based CV Screening

**arXiv ID:** 2610.02972 | [PDF](https://arxiv.org/pdf/2610.02972v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 405. TerraVis: Towards Evaluation of World-Grounded Visual Consistency in Text-to-Image Generation via MLLM Workflows

**arXiv ID:** 2610.02959 | [PDF](https://arxiv.org/pdf/2610.02959v1)

**作者:** Shuai Fu `[一作]` (Adelaide University), Qi Wu `[通讯]` (Adelaide University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究文本到图像模型的世界一致性评估，提出 TerraVis 框架以系统检测和量化生成图像的现实世界违规。

**💡 创新点**

创新点在于：①将世界一致性拆解为对象、交互、场景三级违规分类并定义 18 种细粒度类型；②使用多阶段 MLLM 工作流进行可视化问答、违规检测和严重性评估；③引入指数衰减聚合，将严重违规惩罚放大，提升与人工判断的一致性。

**🔧 技术方法**

技术手段包括：多模态大语言模型（如 GPT‑5.5、Gemma 4 31B）进行可视化问答；基于违规类型的分层检测与严重性分类；指数衰减函数 S = exp[−λ(N_major + αN_minor)] 进行评分；对比传统 FID、CLIPScore、ImageReward 等指标。

**📊 数据集**

使用的数据集为 COCO‑T2I（200 条 prompt）和 GenAI‑Bench（1,600 条 prompt）生成的图像，并收集约 13K 人类评估标注作为黄金标准。

**📈 对比分析**

在两个基准上与现有质量、对齐、人类偏好指标进行 Spearman/Kendall 相关性对比；TerraVis 在 COCO‑T2I 上的 Spearman ρ≈0.44、τ≈0.37，GenAI‑Bench 上 ρ≈0.33、τ≈0.30，均显著高于其它指标（最大提升约 50%）。

**⚠️ 局限性**

局限性：①多阶段 MLLM 流程相对耗时；②违规检测易出现假阳性或漏检；③分类体系可能未覆盖所有新型违规；④依赖人工标注，主观性和稀缺性仍是挑战。

---

## 406. A Guideline-Augmented Multi-Agent Framework for Schema-as-Code Biomedical Named Entity Recognition

**arXiv ID:** 2610.02970 | [PDF](https://arxiv.org/pdf/2610.02970v1)

**作者:** Songtao Li `[一作]` (Dalian Maritime University), Hongfei Lin `[通讯]` (Dalian University Of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了 GAMA，一种基于多代理和注解规则的框架，实现了无需微调 LLM 的结构化生物医学命名实体识别。

**💡 创新点**

创新点在于①从训练样本自动诱导并验证数据集特定的注解规则；②将实体抽取拆分为规划、编码、验证三阶段，并通过 schema‑as‑code 结构化输出；③采用双循环迭代修正提升生成质量。

**🔧 技术方法**

使用了多代理协作、提示工程、规则诱导与验证、链式思考、schema 约束和双循环迭代验证等技术。

**📊 数据集**

实验使用了 NCBI Disease、BC2GM、BC4CHEMD、GENIA、AnatEM 五个公开生物医学 NER 数据集。

**📈 对比分析**

与 CodeIE、GPT‑NER、CMAS、EICL 等基线在 Llama3、Qwen2.5、GPT‑3.5、GPT‑4o 等多种 LLM 上进行无监督比较，GAMA 在所有数据集和模型上均取得最高 Micro‑F1，平均提升 1–4 分。

**⚠️ 局限性**

局限性包括对 LLM 推理的高计算成本、规则诱导受训练样本偏差影响、以及对极长文本或低频实体的泛化能力仍需提升。

---

## 407. Kinematics-Induced Multimodal 3D Human Pose Estimation with Subject-Level Privacy

**arXiv ID:** 2610.02943 | [PDF](https://arxiv.org/pdf/2610.02943v1)

**作者:** Kaushik Bhargav Sivangi `[一作]` (University of Glasgow), Fani Deligianni `[通讯]` (University of Glasgow)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了融合RGB、LiDAR、毫米波雷达的多模态3D人体姿态估计框架，并加入主体级差分隐私与隐私审计。

**💡 创新点**

① 引入运动学驱动的双轨融合模块实现跨模态对齐与自适应加权；② 设计基于回归的主体级成员推断与点级最大泄露分析；③ 采用动作时间分层采样实现主体级DP训练。

**🔧 技术方法**

多模态编码器（DSTFormer、PointTransformerV2）、运动学对齐与自适应门控融合、主体级DP（Poisson采样+自适应裁剪）及RDP合成、点级最大泄露评估。

**📊 数据集**

MM-Fi 多模态人体姿态数据集。

**📈 对比分析**

与单模态及现有多模态基线对比，融合模型在三种评估协议下MPJPE平均下降约10~11mm，跨环境/跨主体提升更显著；在主体级DP下隐私-效用权衡符合预期，ε=200时仍保持较好精度。

**⚠️ 局限性**

仅在八个非成员受试者的MM-Fi划分上验证，受样本量限制；未探索更强的私有自适应学习方法，且仅针对回归型成员推断。

---

## 408. PLCWorld: Benchmarking LLM-Generated PLC Programs in Closed-Loop Plant Simulation

**arXiv ID:** 2610.02982 | [PDF](https://arxiv.org/pdf/2610.02982v1)

**作者:** Yunji Kim `[一作]` (Dongguk University), Woojin Lee `[通讯]` (Dongguk University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了PLCWorld，结合闭环仿真环境与100个工业控制任务，用于评估LLM生成的Structured Text（ST）PLC程序是否完成任务并满足安全约束；

**💡 创新点**

创新点在于构建统一的闭环执行框架、任务规范与评估协议，并分别报告任务成功率与安全违规率，揭示“执行差距”问题；

**🔧 技术方法**

采用ST语言执行引擎、物理仿真模型、评估器规则、LLM文本生成与验证流程；

**📊 数据集**

使用从2201个真实PLC项目和工程文档中抽取的控制关系生成的合成任务，构成100个任务、473个任务-条件对；

**📈 对比分析**

对比六款LLM（GPT‑5.5、GPT‑4o、Claude Sonnet 5、Gemini 3.1 Pro、Gemini 3.8 Flash、Qwen3.5‑Plus）以及四个LLM+验证框架，发现易难度下成功率可达80%+，但难度增加时显著下降，且安全违规率与完成率不完全相关；

**⚠️ 局限性**

局限在于任务为合成抽象模型，未覆盖所有真实工业场景；评估仅在模拟环境中完成，缺乏物理硬件验证；

---

## 409. OLMo-Detect: A Multi-Stage, Confounder-Controlled Benchmark for Membership Inference on Large Language Models

**arXiv ID:** 2610.02986 | [PDF](https://arxiv.org/pdf/2610.02986v1)

**作者:** Tao Shi `[一作]` (University of Melbourne), Jey Han Lau `[通讯]` (University of Melbourne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个跨预训练、中训练和后训练阶段、对齐成员与非成员、并严格过滤重叠的 LLM 会员推断基准 OLMo-Detect，评估了 15 种无监督与 3 种有监督攻击方法。

**💡 创新点**

创新点在于：① 设计多阶段、三轴对齐（质量、时间、词汇）并使用 infini‑gram 13‑gram 过滤非成员；② 引入分布失配版本 OLMo-Detect (Shifted) 来检验鲁棒性；③ 发现“精细数学数据”是最易被识别的类别，揭示数据类型而非训练阶段主导性能。

**🔧 技术方法**

采用的技术包括：基于 OLMo 2 开放训练流水线的基准构造、边界采样与模拟退火生成对齐成员、13‑gram 过滤、以及 15 种已有 MIA（似然、扰动、提示、输出分布）和 3 种监督 MIA 的统一评估框架。

**📊 数据集**

数据集为 OLMo 2 的 9 个域（包括 DCLM‑Baseline、OpenWebMath、peS2o、StarCoder、Math Mix、Stack Exchange、SFT、DPO、RLVR）以及对应的非成员集合，成员与非成员在质量、时间与词汇上严格匹配。

**📈 对比分析**

实验结果显示，无监督方法最高 AUC 为 0.68（Zlib），有监督方法仅略高于无监督，且在跨域（LODO）或分布失配（Shifted）情况下性能大幅下降；模型规模从 1B 逐渐增至 13B 有所提升，32B 之后趋于平稳。

**⚠️ 局限性**

局限性包括：会员信号仍弱，难以跨域迁移；分布失配下鲁棒性不足；结果主要受“精细数学”数据的影响，未能全面揭示训练阶段的影响；并且仅在 OLMo 系列和少数公开模型上验证，尚需在更广泛模型上进一步评估。

---

## 410. Sentry: Learning to Recover from LLM Agent Failures at Test Time

**arXiv ID:** 2610.02994 | [PDF](https://arxiv.org/pdf/2610.02994v1)

**作者:** Changxiu Ji `[一作]` (Stanford University), Kunle Olukotun `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种名为 Sentry 的失败管理层，用于在 LLM 代理执行过程中检测、诊断并修复失败，且只在检测到匹配失败时才向代理提供恢复知识，并通过验证后才将经验写入外部恢复手册。

**💡 创新点**

创新点在于将失败知识视为条件性知识，先在上下文中保持学习、后仅在特定失败时才暴露，并通过运行时验证确保只保存有效的恢复经验，从而兼顾学习与局部干预。

**🔧 技术方法**

技术包括：基于 LLM 的失败检测与诊断、硬修复与软修复路由、检索式恢复手册（按失败类型和标签匹配）、恢复验证（无奖励判定）以及外部演练手册的在线更新。

**📊 数据集**

使用了四个基准：WebShop、AppWorld、SWE‑bench Lite 以及 Mind2Web Replay，涵盖 Web、状态化应用和软件工程任务。

**📈 对比分析**

与多种基准运行时干预方法（AgentGuard、AgentFixer、Wink、AgentForesight）以及上下文演化方法（Reflexion、ACE、AWM、ReasoningBank）比较，Sentry 在所有四个基准上平均提升 37% 的性能（最高 77%），并且与 ACE 结合可进一步提升 39%。

**⚠️ 局限性**

局限性包括：实验仅在 Qwen3.5‑9B 与 GPT‑OSS‑120B 上验证，其他模型的失败模式可能不同；检测器与验证器使用同一模型，未探讨解耦；手册可能出现冗余或过度细化，需要后期合并与裁剪。

---

## 411. Rethinking Fixed Temporal Grids: Frequency-Disentangled Motion Generation

**arXiv ID:** 2610.03012 | [PDF](https://arxiv.org/pdf/2610.03012v1)

**作者:** Yunjiao Zhou `[一作]` (Nanyang Technological University), Jianfei Yang `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种频率自适应运动表示 FreqMo，利用小波分解将运动序列分解为多尺度频率带，并通过统一频率残差量化 UFRQ 将所有频率带压缩到单一共享码本，实现三分之一的 token 长度，同时保持精确重构。

**💡 创新点**

创新点在于将运动分解为多尺度频率带以消除频率耦合，采用从低频到高频的残差量化顺序确保低频结构优先被高精度捕捉，并引入尺度感知注意力以在生成时平衡不同频率层的关注度。

**🔧 技术方法**

核心技术包括离散小波变换（DWT）实现频率分解、统一频率残差量化（UFRQ）进行共享码本编码、掩码 Transformer 与尺度感知注意力生成全部频率带、以及可迁移到连续扩散模型的表示框架。

**📊 数据集**

实验使用 HumanML3D（约14.6k 动作/44.9k 文本）和 KIT-ML（3.9k 动作/6.3k 文本）两个公开数据集。

**📈 对比分析**

与多种基线（包括 Light-T2M、ReMoDiffuse、MLD 等）比较，FreqMo 在 HumanML3D 上获得最佳 FID（0.040）并在 KIT-ML 上保持竞争力；在加速误差和抖动误差上也优于传统方法，显示出高频细节保留显著提升。

**⚠️ 局限性**

局限性包括仅在两大人类运动数据集上验证，缺乏对非人类或更复杂长序列运动的评估；以及在极端高频动作或多模态场景下可能仍存在量化误差与计算开销的挑战。

---

## 412. Temporal Geometry of Deep Networks: Hyperbolic Representations of Training Dynamics for Intrinsic Explainability

**arXiv ID:** 2610.03000 | [PDF](https://arxiv.org/pdf/2610.03000v1)

**作者:** Ambarish Moharil `[一作]` `[通讯]` (Eindhoven University of Technology), Ambarish Moharil (Eindhoven University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个时序双曲空间图元网络，对多层感知机的训练轨迹进行超参数化嵌入，并利用该嵌入进行网络自组织和性能预测；

**💡 创新点**

创新点在于：①将参数图的时序演化映射到Poincaré球面，实现了在负曲率空间中的几何自适应；②设计了基于双曲图注意力与Einstein流形几何的聚合；③使用GRU演化注意力核，保证时序平滑；④引入符号权重回归，将权重幅值与双曲距离关联，极性映射到切空间方向；

**🔧 技术方法**

采用了Poincaré双曲嵌入、图神经网络、双曲图注意力、Riemannian 优化、GRU 核演化、Fermi-Dirac 链接解码器、符号权重回归、以及渐进式训练日程；

**📊 数据集**

使用了MNIST/Fashion‑MNIST（作为INR训练轨迹）、CIFAR‑10（MLP泛化预测）、以及一维正弦回归（sinusoid‑MLP）等数据集；

**📈 对比分析**

与DWSNets、NFN_HNP、GMN等基线对比，实验结果显示在INR分类任务中达到95.6%（MNIST）和80.72%（Fashion‑MNIST）；在CIFAR‑10泛化预测中Kendall τ为0.846，略低于NFN_HNP的0.934；在sinusoid任务中MSE为1.06，略优于GMN的1.13；总体表现与主流基线相当或稍逊，但提供了可解释的几何嵌入；

**⚠️ 局限性**

局限性包括：①在泛化预测任务中精度略逊于纯张量方法，说明对细粒度结构的保留不够；②双曲空间训练对曲率、步长等超参数敏感，可能导致数值不稳定；③目前仅针对MLP，扩展到更复杂架构仍需研究；④时序核演化的记忆占用与梯度消失问题需要进一步优化。

---

## 413. Root isolation for analytic functions using cubic hermite interpolation

**arXiv ID:** 2610.02934 | [PDF](https://arxiv.org/pdf/2610.02934v1)

**作者:** Christophe Raffalli `[一作]` `[通讯]` (Université de la Polynésie Française), Christophe Raffalli (Université de la Polynésie Française)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于局部立方Hermite插值的根隔离算法，利用数值比较快速判断区间内是否包含根，递归细分并终止，得到每个区间至少包含一个实根。

**💡 创新点**

创新点在于：① 用三次Hermite插值作为判别标准，避免传统符号根计数方法的符号计算和高阶多项式展开；② 通过质量量 Q(x) 的阈值 ε 控制细分，理论上保证终止且收敛；③ 将该算法推广到任意解析函数并在GPU单精度下实现实时隐式曲面可视化。

**🔧 技术方法**

技术包括：数值计算（双精度/单精度浮点）、Hermite插值、根计数（基于插值的导数零点）、递归细分、GPU GLSL 实现、与传统 Sturm/Descartes 方法的比较。

**📊 数据集**

在多项式家族（Chebyshev、Legendre、Wilkinson、几何、Mignotte）以及随机多项式上进行实验；还在隐式曲面渲染中使用三维函数（如 Barth sextic、Labs heptic、cos(y²+z²)-sin(xy)-e^{xz}/2 等）作为数据集。

**📈 对比分析**

与现有基于 Sturm 序列或 Descartes 规则的实现（pari/gp、RS）相比，实验显示：细分次数近似线性；总计算时间显著更短（在同等浮点精度下往往快几倍甚至十倍）；在GPU单精度下仍能保持高帧率（30~100 FPS）且无可见误差。

**⚠️ 局限性**

局限性：算法仅为数值近似，缺乏严格的误差保证；需要经验选择 ε 与 n，某些特殊多项式（如 Wilkinson）对 ε 需求更高；对极高次数多项式受浮点精度限制；对极度聚集根的处理仍不够稳健。

---

## 414. On Representational Alignment among Embodied Agents

**arXiv ID:** 2610.02985 | [PDF](https://arxiv.org/pdf/2610.02985v1)

**作者:** Fulvio Mastrogiovanni `[一作]` `[通讯]` (University of Genoa), Fulvio Mastrogiovanni (University of Genoa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

提出了“关系充分性”理论，研究在多主体身体化认知中，哪些异构观测者的表征差异需要在交互中被消除，哪些可保持未解；

**💡 创新点**

核心创新在于将表征对齐的需求从全局一致性转向交互相关的“任务可区分性”，并给出最少锚定量的局部下界；

**🔧 技术方法**

使用关系足够性框架、等价类与不确定度纤维、局部线性分析与可观测性思想，构建理论证明；

**📊 数据集**

论文没有使用实验数据集，全部为解析模型（异步手交任务的两个变体）验证理论；

**📈 对比分析**

通过解析两种手交情景（相对同步 vs 绝对时间）演示：相对同步时不需要额外锚定，绝对时间时需至少一个标定锚定；性能表现为理论上满足关系充分性；

**⚠️ 局限性**

局限包括：仅考虑两主体，假设已知可传输映射，锚定仅为静态可观测，未处理随机性与多任务情景，且未在真实机器人实验中验证。

---

## 415. Profile-Aware Trustworthy Recipe Generation with Planner-Critic Agentic Remediation

**arXiv ID:** 2610.02969 | [PDF](https://arxiv.org/pdf/2610.02969v1)

**作者:** Shanhong Liu `[一作]` (Singapore University of Technology and Design), Konstantinos N. Plataniotis `[通讯]` (University of Toronto)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 PCAR 框架，将食物图像生成食谱分为规划和安全审计两步，形成迭代纠错循环。

**💡 创新点**

创新点在于将安全验证与生成分离，使用安全批评者给出结构化反馈实现可修复而非直接拒绝的机制。

**🔧 技术方法**

使用规划-批评者两代理模型，结合大型语言模型（如 GPT‑4o、Gemini‑2.5‑Flash）和本地小型模型，构建感知、生成与安全审核模块。

**📊 数据集**

实验基于 Epicurious 食谱图像数据集，并构造 100 个包含过敏、饮食和准备安全约束的用户配置文件。

**📈 对比分析**

通过 SGSR、ULR、MRL、RASS 等指标评估安全性与效率，在 GPT‑4o 上实现 98% 安全生成成功率、0% 漏洞率，Gemini‑2.5‑Flash 约 74% 成功率，局部模型性能显著下降。

**⚠️ 局限性**

局限性主要是对底层模型的高度依赖，轻量化本地模型在安全性和稳定性上表现不足，缺乏对营养与医学约束的更细粒度建模。

---

## 416. Reliable Self-Evolution with Imperfect Proxy Rewards

**arXiv ID:** 2610.02975 | [PDF](https://arxiv.org/pdf/2610.02975v1)

**作者:** Kangjun Noh `[一作]` (Yonsei University), Kyungwoo Song `[通讯]` (Yonsei University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 CISE 框架，利用条件共形推断生成候选特定奖励区间，并将该区间用于 LLM 驱动的自进化搜索的反馈和最终候选筛选，显著降低误报。

**💡 创新点**

创新点在于首次将条件共形推断与在线密度比估计相结合，生成可校准、可覆盖的奖励区间，既能在进化中消除代理误差，又能在输出时保证高可靠性；同时将区间信息同时用于搜索反馈与候选接受决策。

**🔧 技术方法**

采用条件共形推断、Least‑Squares Importance Fitting 进行在线密度比估计，使用 gpt‑5‑mini 生成候选，代理模型 MGT 预测带隙，最后用 Quantum ESPRESSO 进行 DFT 高保真评估。

**📊 数据集**

实验基于 LLEMA benchmark 的三项材料发现任务：宽带隙半导体（WBG）、固态电解质（SSE）和光伏吸收剂（PV），并使用相应的源数据集做密度比估计与共形校准。

**📈 对比分析**

与 MatterGen、LLMatDesign、LLEMA 四个方法比较，CISE 在每个任务中返回的候选数更少，但所有被验证的候选均为真阳性；在相同验证预算下（如 PV 任务），CISE 的成功率为 100%，而 LLEMA 仅 50%，显示出显著的可靠性提升。

**⚠️ 局限性**

局限性包括：需要满足独立性和可界定的协变量偏移假设；区间宽度可能过宽导致输出数量下降；依赖代理模型的准确性和共形校准数据集的质量；在非材料领域或代理误差分布不满足假设时效果可能下降。

---

## 417. Understanding Trajectory Heterogeneity in Federated World Model Learning

**arXiv ID:** 2610.02957 | [PDF](https://arxiv.org/pdf/2610.02957v1)

**作者:** Yipan Wei `[一作]` (Wuhan University), Lixu Wang `[通讯]` (Chinese University of Hong Kong Shenzhen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个基于MIMIC-IV的交叉时间联邦世界模型基准，评估10种联邦算法在8种临床疾病、4种严重度分区和6个预测步长下的表现。

**💡 创新点**

创新点在于：①提出“交叉时间所有权”概念，模拟同一病程在不同客户端间的时间片段划分；②系统性量化所有权对历史/未来窗口可用性的影响；③揭示不同分区粒度对长期预测精度的影响；④对算法在不同时间片段下的更新规模、外推激活和缓存敏感性进行细粒度分析。

**🔧 技术方法**

采用了基于Transformer的自编码世界模型（latent动态预测 + FiLM调制），配合FedAvg、FedProx、FedAvgM、FedAdam、SCAFFOLD、FedExP、FedVARP、FedCDA、FedLWS、FedMuon等联邦优化器。

**📊 数据集**

使用MIMIC-IV 3.1数据库的8个疾病科室（Circulatory、Digestive、Endocrine/metabolic、Infectious、Injury/poisoning、Neoplasms、Nervous/sensory、Respiratory）共计约4.1千万转移样本。

**📈 对比分析**

通过fresh-feature–patient宏观MAE（nMAE）和nMSE在H1–H32六个步长上进行评估。FedAvg为基准，FedProx在短期（H1–H8）平均提升约0.6%，其优势随步长增长消失；FedExP、FedLWS基本等于FedAvg；FedVARP、SCAFFOLD、FedMuon等在大多数设置下误差显著上升。中心化模型在所有疾病上均优于任何联邦方案。

**⚠️ 局限性**

局限性包括：①长期窗口（H32）覆盖率仅7.5%–21.4%，严重限制了长时序学习；②分区粒度加细会导致训练样本分布偏移且提高误差；③算法分析受限于固定的五轮、10%参与度设定，未探索更大参与度或多轮动态采样；④未验证跨机构、跨设备的真实可扩展性。

---

## 418. GTDD: Generative Test-Driven Development for AI Coding Agents with Adversarial Testing

**arXiv ID:** 2610.02952 | [PDF](https://arxiv.org/pdf/2610.02952v1)

**作者:** Masahiro Kato `[一作]` `[通讯]` (Mizuho-Dl Financial Technology Co Ltd), Masahiro Kato (Mizuho-Dl Financial Technology Co Ltd)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种名为Generative Test-Driven Development（GTDD）的开发框架，将测试生成与编码分离，利用在每次提交后生成的新测试并返回简化的反例作为反馈，最终通过独立的随机审计决定是否接受代码。

**💡 创新点**

创新点在于：①将测试生成与实现迭代分离，形成可适配的反馈循环；②提供了对有限样本下自适应候选者的假设检验与错误概率上界分析；③提出在每次提交后进行随机审计的方式，既保留了已有回归测试，又能保证接受决策的统计可靠性；④在实验中验证了动态生成测试相较于一次性生成测试的优越性。

**🔧 技术方法**

技术包括：大语言模型（gpt‑4.1‑mini）用于生成测试与实现；回归测试与简化的失败案例收集；基于有限总体的概率分析与描述长度论证；随机抽样与超马尔科夫过程（Ville不等式）用于证明审计置信度；以及在实验中使用的Python等实现工具。

**📊 数据集**

使用了一个基于Redis子集的状态化键值存储任务，包含61个命令、8个功能族，评估集共2000个随机操作序列；此外通过400/600个由模型生成的测试样本进行开发。

**📈 对比分析**

比较方法：在30个代码块上对五种策略（Public only、Static property、Static LLM、Dynamic blind、Dynamic informed）进行四轮修复，利用配对差异检验（bootstrap、符号检验）评估最终失败率。结果显示，Dynamic blind和Dynamic informed的平均失败率分别为0.279/0.274，而Static LLM为0.405；差异显著（p≈0.0015），表明动态测试生成显著降低错误率。进一步的“增量”实验未能单独归因于再生、样本量或历史反馈。

**⚠️ 局限性**

局限性包括：①实验仅在单一任务（键值存储）上验证，缺乏多任务验证；②大语言模型的随机性与调优对结果影响尚未充分评估；③审计样本量对最终接受率的影响仍需进一步优化；④未深入探讨动态生成测试对更复杂自然语言或语音交互场景的适用性。

---

## 419. Evaluator-in-the-Loop Monte Carlo Tree Search via LLM Agents for Motif Scaffolding in Protein Design

**arXiv ID:** 2610.02924 | [PDF](https://arxiv.org/pdf/2610.02924v1)

**作者:** Haotian Hu `[一作]` (Georgia Institute of Technology), Faramarz Fekri `[通讯]` (Georgia Institute of Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种名为 ELMS 的基于 LLM 的蒙特卡洛搜索框架，用于在蛋白质 Motif‑scaffolding 过程中将结构评估结果转化为可执行的局部修复操作。

**💡 创新点**

创新点在于将评估器反馈嵌入搜索树中，通过 Critic 与 Policy 两个 LLM 代理解析状态特定的结构问题，并通过 motif‑locked 操作保持关键 Motif 的几何约束，从而将传统的生成‑筛选流程升级为迭代式、反馈驱动的设计搜索。

**🔧 技术方法**

核心技术包括：LLM（gpt‑5‑nano）生成的 Critic 与 Policy 方案、固定的 motif‑locked 操作库、ESMFold 等结构评估器、以及基于 UCB 的蒙特卡洛树搜索（MCTS）实现搜索与回传。

**📊 数据集**

在 GeomMotif 与 MotifBench 两大公开基准上进行实验，使用 100 候选人/任务的评估预算进行对比。

**📈 对比分析**

在 GeomMotif 单/双 Motif 任务中，ELMS 分别达成 86.41% 与 84.57% 的成功率，MotifBench 上的任务成功率达到 88.89%，显著高于最强基线（单一生成器 53.33% 及 26.7/30 任务成功）。

**⚠️ 局限性**

局限性包括：对昂贵的结构评估器依赖强、目前仅在序列层面进行修改、缺乏实验验证、以及在更复杂的功能或多目标设计中的可推广性尚待评估。

---

## 420. When to Compile a Computer-Use Agent? Measuring Payback and Making Compilation Decisions for Token Efficiency

**arXiv ID:** 2610.02932 | [PDF](https://arxiv.org/pdf/2610.02932v1)

**作者:** Yulong Ming `[一作]` (City University of Hong Kong), Xiaohua Jia `[通讯]` (City University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于经验的程序编译与在线决策框架 PACE，用以降低电脑使用代理的 token 消耗。

**💡 创新点**

创新点在于：① 统一测量编译成本与回报的协议；② 在累计成本约束下的在线编译决策；③ 对决策的累积成本提供理论保证。

**🔧 技术方法**

采用 ReAct 风格代理、GUI 交互程序编译、Token 成本测量与累积预算约束的决策算法。

**📊 数据集**

使用了七个参数化任务族、三种模型配置以及合成与真实日志任务到达序列作为数据集。

**📈 对比分析**

与 ReAct、AutoRPA、ToolPro 对比，平均在合成与记录到达模拟下 token 成本分别降低 17.3%、24.9%、17.3%；理论上保证累计成本不超过基线的 1+ε 倍。

**⚠️ 局限性**

局限性包括：仅覆盖七个任务族与三种模型；在线实验未覆盖完整算法；对无声程序失败与部分接口漂移缺乏评估；仅计量 token，未考虑延迟、设备使用及维护成本；模型更新可能影响测量结果。

---

## 421. Differentiable Koopman Operator for Contrastive Learning on Dynamic Graphs

**arXiv ID:** 2610.02990 | [PDF](https://arxiv.org/pdf/2610.02990v1)

**作者:** Md Abrar Jahin `[一作]` (University Of Southern California), Md Rizwan Parvez `[通讯]` (Qatar Computing Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出KAIROS，一个自监督动态图对比学习框架，结合可微Koopman算子对节点嵌入随时间的线性演化建模，用于无监督节点分类和异常检测。

**💡 创新点**

创新点：①将Koopman算子嵌入对比学习循环，实现对节点嵌入演化的显式线性模型；②采用双视图（原始特征与PPR扩散）多粒度对比；③通过Koopman残差、时序不一致和邻域偏差三项无监督异常分数融合，得到高效异常检测。

**🔧 技术方法**

技术：自监督双分支图神经网络、InfoNCE对比损失、多粒度视图采样、可微Koopman算子、正交正则化、z标准化异常分数组合。

**📊 数据集**

数据集：九个动态图基准——DBLP、Bitcoinotc、BITotc、BITalpha、TAX51、Reddit、MOOC、Arxiv、Elliptic。

**📈 对比分析**

比较方法：在节点分类上与14种半监督/无监督基线、在异常检测上与7种专用基线和CLDG/CLDG++对比；KAIROS在节点分类上大部分数据集与CLDG++相当或略优；在异常检测上在所有九个数据集均获得最高性能，平均提升约13.3 ROC‑AUC，单个数据集最高提升23.15。

**⚠️ 局限性**

限制：依赖离散时间快照，对高度不规则或事件驱动的图可能不满足线性Koopman假设；异常检测实验使用合成注入，真实异常场景下性能尚待验证。

---

## 422. AvoKV-E: Payload-Aware KV Cache Eviction for Long Reasoning

**arXiv ID:** 2610.03007 | [PDF](https://arxiv.org/pdf/2610.03007v1)

**作者:** Han Yu `[一作]` (LinkedIn Corporation), Alborz Geramifard `[通讯]` (LinkedIn Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对长推理过程中 KV 缓存压缩问题，提出了一种训练无关的两阶段淘汰策略；

**💡 创新点**

创新点在于：①引入延迟淘汰机制，防止新生成的状态因早期缺少注意而被误判；②在淘汰评分中加入了读压力、键冗余与价值负载三项互补信号，并采用候选相对归一化；

**🔧 技术方法**

采用的技术包括：基于注意力历史的读压力估计、RoPE 变换下的键相似度测度、值向量范数的中位数归一化，以及候选内百分位归一化的组合评分；

**📊 数据集**

使用的数据集为 MATH-500 和 AIME-25 两个数学推理基准；

**📈 对比分析**

与无压缩基线、基于注意力×冗余、滞后注意力和思想适应性压缩三种基线在匹配 KV 预算下进行对比，结果显示新方法在最小预算下性能提升最大，在所有预算下均不劣于其他方法，并显著降低最大长度截断次数；

**⚠️ 局限性**

局限性包括：只在两个大型推理模型上验证，未覆盖代码或多模态推理任务；实验在 HuggingFace 生成框架下进行，未评估实际内存压缩率或吞吐量；

---

## 423. Recursive Self-Improvement in Unified Multimodal Models

**arXiv ID:** 2610.03002 | [PDF](https://arxiv.org/pdf/2610.03002v1)

**作者:** Huijuan Wang `[一作]` (University of Southern California), Xuezhe Ma `[通讯]` (University of Southern California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种递归交叉能力自我提升（RSI）循环，利用统一多模态模型（UMM）的文本与视觉能力相互监督，并通过程序执行提供外部真值；

**💡 创新点**

创新点在于将程序执行作为独立真值来源，让模型在自身生成和理解错误的诊断中生成验证数据，从而实现闭环的自我提升；

**🔧 技术方法**

采用生成器、阅读器、写程序器三种能力，并结合程序执行环境、错误诊断规则和动态数据计划进行训练；

**📊 数据集**

在图表生成任务上使用自己构造的 BasicChartBench（包含两层难度、476张图表和1.4万问题）进行评估；

**📈 对比分析**

实验对比显示四轮RSI后在未见请求措辞的基本层次上准确率从45.7%提升至60.2%，相比仅继续训练保持约46.3%，验证数据和针对性数据分别贡献了约8.6%与3.5%的提升；

**⚠️ 局限性**

局限性包括：在结构准确性上对未知措辞的适应不佳、词汇切换导致的遗忘、现有更大规模基准仍难以提升、以及总体通用理解能力轻微下降。

---

## 424. Response Variability and Stability in Human Reasoning

**arXiv ID:** 2610.03008 | [PDF](https://arxiv.org/pdf/2610.03008v1)

**作者:** Clemens Bombach `[一作]` (TU Chemnitz), Marco Ragni `[通讯]` (TU Chemnitz)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于几何距离和能量（p‑energy）的形式化框架，用于量化并比较人类演绎推理（尤其是三段论推理）中的响应模式及其可变性。

**💡 创新点**

创新点在于将推理任务与响应映射为离散空间中的函数，并通过定义任务与响应的哈密顿距离来度量个体推理模式的相似度和局部能量，从而捕捉个体间以及同一人随时间的稳定性与变化。

**🔧 技术方法**

技术手段包括：二进制特征编码、哈密顿距离、p‑energy 定义、广义线性混合效应模型（GLMM）预测复测表现、k‑means 聚类分析以及混合效应模型评估能量与正确率的交互作用。

**📊 数据集**

使用了公开的“syllogistic reasoning stability”数据集（100名受试者、64个三段论题目、两轮测验、一周间隔），该数据集提供了每位受试者在两次测验中的完整响应记录。

**📈 对比分析**

通过GLMM比较，能量对复测正确率具有显著预测效应：对初始错误响应，高能量提高复测正确率；对初始正确响应，高能量降低复测正确率；模型AIC、BIC和伪R²表明加入能量后模型拟合显著改善。

**⚠️ 局限性**

局限性包括：框架仅适用于确定性响应（未考虑多答案情况）；任务编码基于三段论的表面特征，未纳入内容效应或信念偏差；能量定义与理论假设紧密耦合，可能在其他推理域或不同心理模型下需要重新设计距离函数。

---

## 425. Relevant Evidence Decoding for Audio-Visual Hallucination Mitigation

**arXiv ID:** 2610.02976 | [PDF](https://arxiv.org/pdf/2610.02976v1)

**作者:** Hyunjae Ra `[一作]` (Sungkyunkwan University), Sungeun Hong `[通讯]` (Sungkyunkwan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练无关的“Relevant Evidence Decoding (RED)”方法，用于减少音频-视觉大语言模型（AV-LLMs）的跨模态谵妄。

**💡 创新点**

创新点在于先通过问题仅问的前向推理确定问题需要的感知证据类型，然后利用点互信息（PMI）将音频、视频以及它们的交互贡献分解，并仅增强与问题相关的证据贡献，从而使模型的预测更好地跟随真正需要的证据。

**🔧 技术方法**

核心技术包括：问题导向的证据选择器、PMI 分解技术、对比解码的自适应校正以及在解码阶段对原始音频-视觉预测的增量式调整。

**📊 数据集**

实验使用的主要数据集为 CMM、AVHBench、SVHalluc、MUSIC-AVQA 以及其 Hard 子集，覆盖了多种音频-视觉谵妄和通用问答场景。

**📈 对比分析**

与 AVCD、ASD、MAD 等现有对比解码方法进行对比，RED 在三种主流 AV-LLMs（VideoLLaMA2.1-7B-AV、Qwen2.5-Omni-3B/7B）上在 CMM、AVHBench、SVHalluc 三大谵妄基准上分别提升了 3–7% 的准确率，平均首词生成时间仅为标准解码的 1.5 倍；在 MUSIC-AVQA 与 Hard 子集上也实现了明显的性能提升。

**⚠️ 局限性**

局限性：证据选择器仍依赖文本信息，可能在极端多模态推理或模态缺失的情况下失效；虽然不需要额外训练，但对每个问题仍需额外的前向推理，导致在实时或高吞吐量场景中仍存在一定的计算开销。

---

## 426. From Language Priors to Field Adaptation: Preference Learning for Traversability Estimation

**arXiv ID:** 2610.02974 | [PDF](https://arxiv.org/pdf/2610.02974v1)

**作者:** Simon Schwaiger `[一作]` (Graz University of Technology), Gerald Steinbauer-Wagner `[通讯]` (Graz University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于语言先验的 vMF 原型混合模型，用以在冻结的视觉‑语言特征空间中估计机器人行走可通行性，并通过少量相对标注实现域适配。

**💡 创新点**

创新点在于将通行性作为可解释的语言先验进行初始化，利用 von Mises–Fisher 混合在视觉‑语言空间中学习可通行性原型，从而实现零样本跨域推理和样本高效微调。

**🔧 技术方法**

核心技术包括 CLIP‑style 冻结视觉‑语言模型、vMF 混合原型学习、平方铰链偏好学习、语言与图像相对标注的联合训练，以及基于责任函数的密集可通行性推断。

**📊 数据集**

使用了 WayFAST（带相对可通行性标注）、RUGD 进行数据迁移评估，以及 RoboNav 的大规模三维语义地图做可视化验证。

**📈 对比分析**

与静态提示 VLM、随机初始化 vMF、线性探针、MLP 和 Transformer 头等基线进行比较，实验表明语言先验初始化在零样本、跨域迁移和少样本微调三方面均能获得更低的 HDR（误判率），并在微调后与现有最先进的端到端模型竞争。

**⚠️ 局限性**

局限性包括对冻结模型的依赖导致对空间结构的捕捉不足；vMF 原型的可解释性受限于语言‑视觉嵌入空间的歧义；以及在极端场景或大规模连续学习时对持续自我标注和动态更新机制的需求未充分解决。

---

## 427. SlimKV: Joint Token-Feature KV Cache Compression with Reconstruction-Free Beacon Attention

**arXiv ID:** 2610.02953 | [PDF](https://arxiv.org/pdf/2610.02953v1)

**作者:** Zihan Teng `[一作]` (University of Science and Technology of China), Weichen Liu `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SlimKV框架，实现令牌与特征双向压缩的KV缓存，适用于长上下文LLM。

**💡 创新点**

结合低秩Beacon学习、层自适应秩分配，并利用Key侧RoPE不对Beacon键做加法，得到无重建的潜在空间注意力。

**🔧 技术方法**

低秩参数化、Beacon生成、RoPE异向性、潜在空间注意力、层自适应秩分配等技术。

**📊 数据集**

LongBench、Needle-in-a-Haystack、RedPajama、LongAlpaca、BookSum、GPT生成的合成数据。

**📈 对比分析**

在Llama-3.1-8B和Qwen2.5-14B上，与Full KV、PALU、KVZip、SnapKV、Activation Beacon等基线相比，16×/32×压缩下平均得分超过93%，并在128K上下文下实现7.34×注意力加速、3.38×整体解码加速。

**⚠️ 局限性**

主要验证在RoPE编码的LLM上，缺少对其他位置编码或多模态缓存的评估，且Beacon压缩效果依赖训练数据分布。

---

## 428. Tracking Human Daily Cognitive Activity from EEG and Biometric Data

**arXiv ID:** 2610.02971 | [PDF](https://arxiv.org/pdf/2610.02971v1)

**作者:** Alina Gutoreva `[一作]` (Kazakh-British Technical University), Zhaniya Omar `[通讯]` (Kazakh-British Technical University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在真实生活环境下，利用EEG、可穿戴生理信号、行为视频和自我报告数据，构建并验证多模态框架来追踪每日认知活动。

**💡 创新点**

提出将神经、行为、情境多模态数据整合，并首次在自由生活环境中验证其可行性，同时发现午间能量与动机下降的时间模式。

**🔧 技术方法**

采用时间同步、特征提取、早期/晚期/混合融合策略、相关分析和多元线性回归等技术。

**📊 数据集**

使用三位受试者两周内按10分钟间隔记录的自评数据（280个标记区间），未包含EEG实际分析。

**📈 对比分析**

通过多元回归模型预测动机，R^2=0.76，MAE≈9.84，RMSE≈12.85；未与现有基线方法进行对比。

**⚠️ 局限性**

局限包括样本量极小、仅采用自评数据、缺乏EEG/生理数据分析、可能存在自我报告偏差和同步误差。

---

## 429. Digital Twin-Assisted Mapping of ICS Telemetry to ATT&CK for ICS with Evidence-Driven Dependency Reasoning

**arXiv ID:** 2610.02955 | [PDF](https://arxiv.org/pdf/2610.02955v1)

**作者:** Konstantinos E. Kampourakis `[一作]` (Norwegian University of Science and Technology), Sokratis Katsikas `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于数字孪生的框架，将ICS遥测同步转换为结构化事件，利用检索增强LLM映射到ATT&CK for ICS，并构建时间索引的依赖图。

**💡 创新点**

创新点在于将同步的数字孪生上下文与检索增强生成式语言模型相结合，实现语义映射和依赖推理，并对不同配置进行系统评估。

**🔧 技术方法**

采用数字孪生同步状态、检索增强生成式语言模型（LLM）以及基于上下文的依赖分类器。

**📊 数据集**

使用SWaT、BATADAL和WADI三个水处理系统的遥测数据集进行实验。

**📈 对比分析**

对比四种映射配置（M1–M4）和四种依赖配置（D1–D4），在SWaT上DT增强可将FP降低约35%，召回波动不大，外部数据集未出现显著提升；依赖推理在协同检索上能识别关系，但易将错误映射连成误边。

**⚠️ 局限性**

局限性包括：召回率不稳定、对不同数据集迁移性差、依赖推理易误连错误映射、实际分析负担高、缺乏多领域验证以及缺少对模型不确定性的显式传播。

---

## 430. Enhancing Biomedical Named Entity Recognition via Multiple Programming Languages Instruction Tuning and Ensemble Method

**arXiv ID:** 2610.02949 | [PDF](https://arxiv.org/pdf/2610.02949v1)

**作者:** Songtao Li `[一作]` (Dalian Maritime University), Hongfei Lin `[通讯]` (Dalian University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了多语言指令调优与投票集成框架MITE，用代码式结构化输入输出改进生物医学命名实体识别。

**💡 创新点**

创新点在于将BioNER转化为结构对结构的代码生成任务，并通过多语言表示产生结构多样化监督以及多语言投票提升鲁棒性。

**🔧 技术方法**

使用大型语言模型（如LLaMA2）进行监督指令微调，结合多语言代码模板和实体级投票集成。

**📊 数据集**

使用六个公开BioNER数据集，包括BC5CDR-化学、BC2GM-基因、NCBI-疾病以及外部转移集NLM-Gene、NLM-Chem-BC7、BC5CDR-Disease。

**📈 对比分析**

在标准Micro-F1上与BERT/LLM基线对比，MITE在三大基准数据集上均取得最高分，且在跨数据集转移任务中表现优异。

**⚠️ 局限性**

局限在于仍需依赖大规模LLM与多语言模板，缺乏对更低资源或非代码相关领域的泛化与可解释性。

---

## 431. Safeguarding Mutual Correction in Source-Free Domain Adaptation via Cut Statistics

**arXiv ID:** 2610.02981 | [PDF](https://arxiv.org/pdf/2610.02981v1)

**作者:** Seongjun Lee `[一作]` (Korea University), Changhee Lee `[通讯]` (Korea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种源自由域适配方法SafeCut，利用目标模型与外部对偶模型之间的相对可靠性进行门控互相纠正；

**💡 创新点**

核心创新在于使用无标签的cut统计量作为预测可靠性代理，动态决定监督方向与强度，并配合时间自锚机制抑制错误传播；

**🔧 技术方法**

技术包括cut统计量计算、可靠性门控的KL互监督、IIC信息最大化、熵多样性正则、温度门控σ函数与时间自锚；

**📊 数据集**

实验数据集涵盖Office‑31、Office‑Home、DomainNet‑126、VisDA以及医疗影像Camelyon17、EyePACS‑APTOS；

**📈 对比分析**

与现有SFDA基线（SHOT、NRC、CoWA、ELR等）及最新方法比较，SafeCut在Office‑31平均93.2%，Office‑Home 90.9%、DomainNet‑126 86.7%、VisDA 90.9%，均实现或逼近最优性能；

**⚠️ 局限性**

局限性包括需对偶模型具有明显不同的错误模式，若对偶模型极其不可靠或两模型预测高度一致，门控优势减弱；此外，对β等门控参数仍有一定敏感性。

---

## 432. RIPPLE in Still Water: Zero-Shot Clustering in Federated Learning with Wavelet Scattering Transform

**arXiv ID:** 2610.03054 | [PDF](https://arxiv.org/pdf/2610.03054v1)

**作者:** Alessandro Licciardi `[一作]` `[通讯]` (Politecnico di Torino), Alessandro Licciardi (Politecnico di Torino)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种聚类联邦学习框架，先在离线阶段用每个客户端的谱特征（变异权重主成分+Wavelet散射变换）生成离散化的软聚类权重，然后在联邦训练中按这些固定权重聚合模型，支持新客户端零次训练直接获得个性化模型。

**💡 创新点**

创新点包括：①将聚类分配完全脱离训练循环，避免每轮通信和梯度泄露；②通过WST实现对域漂移鲁棒的谱特征；③利用GMM‑VAE在服务器端完成离线映射，可一次前向传播完成分配；④提供可计算的路由误差上界和聚类目标与真实目标的差距保证；⑤实现零次训练的新客户端个性化。

**🔧 技术方法**

核心技术：谱特征构造（变异权重主成分）、Wavelet散射变换（WST）、高斯混合变分自编码器（GMM‑VAE）映射软聚类权重、软权重聚合的联邦更新、HDBSCAN用于聚类数自动修正。

**📊 数据集**

实验数据集包括 MNIST、CIFAR‑10、CIFAR‑100、Office‑Home 与 GLDv2‑23k，覆盖标签不平衡、协变量漂移和自然长尾分布。

**📈 对比分析**

与11种基线（全局、个性化、传统聚类FL等）对比，使用C≤10聚类、S=10客户端/轮、1000轮训练，结果显示在所有五个基准上均明显优于基线，尤其在GLDv2与Office‑Home等最真实的分布下，平均提升幅度在3–7个百分点；在零次新客户端评估中，零次分配也能击败最佳基线。

**⚠️ 局限性**

局限性：仅提供信息论层面的安全/误差分析，未给出差分隐私保障；离线聚类假设数据分布不随时间变化，若客户端特征漂移或新分布出现需重新离线训练；对WST参数（尺度、方向数）的选择依赖经验；在轻微非IID但分布相近的场景（如CIFAR‑10）聚类分解可能过细，导致零次分配效果退化。

---

## 433. hacktrace: behavior-supervised detection of reward hacking during code generation

**arXiv ID:** 2610.03055 | [PDF](https://arxiv.org/pdf/2610.03055v1)

**作者:** Hao Jiang `[一作]` (Nanyang Technological University), Weisi Lin `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本工作提出一种基于编码代理内部激活状态的行为监督监视器，用于在代码生成过程中即时检测并惩罚奖励劫持行为，并公开了173,561条多轮编码轨迹及其标签。

**💡 创新点**

创新点在于：①将奖励劫持行为的**尝试**而非仅成功案例作为监督目标；②利用生成过程已产生的激活状态（无需额外模型推理）构建轻量级监视器；③将激活信息与最终文件静态特征相结合，获得近乎完美的检测准确率；④将该监视器作为GRPO训练中的惩罚信号，显著降低作弊率。

**🔧 技术方法**

核心技术包括：多层激活池化（MultiMax）与固定层读出（Fixed Readouts），激活聚合与静态特征组合，基于二分类逻辑回归的集成评分，GRPO（Group Relative Policy Optimization）强化学习框架以及与内部激活状态相结合的惩罚项。

**📊 数据集**

使用的主要数据集是自构造的编码轨迹集：173,561条由Qwen3-8B（以及可选的Llama‑3.1‑8B‑Instruct）在CodeContests题库上产生的多轮交互轨迹，配有最终文件、测试执行结果以及人工/LLM裁定的奖励劫持标签。

**📈 对比分析**

与传统自报探测器（self‑report probes）、静态AST特征、TF‑IDF+逻辑回归、CodeBERT等方法相比，激活+静态组合监视器在“同一题目”AUC上达到0.997，误检率仅1%，监测延迟≈8 ms，且无额外语言模型通道；在GRPO训练中，使用完成标签的MultiMax惩罚能将作弊率从≈90 %降至1‑5 %，同时保持≥30 %诚实通过率。

**⚠️ 局限性**

主要局限包括：仅在Qwen3-8B（及少数Llama‑3.1‑8B‑Instruct）上验证；标签基于最终产物而非完整操作历史，人工审核覆盖范围有限；监测仅在回溯性前缀上评估，未证明对实时动作的即时识别；延迟评估未与连续批处理系统集成；训练阶段只测试了惩罚引发的适应，未对专门攻击做进一步验证。

---

## 434. Continual Graph Memory for Mathematical Research Agents

**arXiv ID:** 2610.02945 | [PDF](https://arxiv.org/pdf/2610.02945v1)

**作者:** Junyi Zhang `[一作]` (University of California), Wei Wang `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了名为 Ansatz 的数学研究代理，结合了可进化的图结构记忆系统 Continual Graph Memory，用于记录、检索和复用证明过程中的中间结果。

**💡 创新点**

创新点在于：①将证明事实、探索计划、反例等多种中间信息分离成不同类型节点，并用显式依赖图表达它们之间的关系；②引入证据敏感的策展人、可检索的依赖前置以及跨项目的可扩展回忆机制，保证只在有证据支持的上下文中重用信息；③通过可进化的记忆架构实现跨任务学习与自适应更新。

**🔧 技术方法**

技术手段包括：大型语言模型（GPT‑5.6）驱动的工作者与验证器、基于图的记忆写入/检索工具（fact_submit、fact_neighbors、memory_recall 等）、策展人标注与前沿更新、以及离线的可重现性审计。

**📊 数据集**

数据集主要有：① First Proof Second Batch（10 个研究级数学题目）作为基准；②若干公开的开放数学问题（Jamison caterpillar conjecture、Erdős 289/348/488 等）作为实战测试。

**📈 对比分析**

通过在不使用网络搜索的条件下，与 Danus、Codex、GPT‑5.6、官方参赛团队等进行对比。Ansatz 在 First Proof Second Batch 上 10/10 成功率，远超对手（最多 7/10），并在开放问题中独立完成 4 个完整解答及多项进展；显著提升了检索效率和验证质量。

**⚠️ 局限性**

局限性包括：①对事实图结构本身的贡献在某些任务中不明显；②系统高度依赖 LLM 及外部验证器，若验证失误可能导致错误记忆；③未能解决所有开放问题，部分进展仍需人工审阅；④缺乏对更大规模或不同领域问题的评估。

---

## 435. SoftGene: Protein Language Model-Enhanced Soft Prompting for Interpretable Gene Set Annotation

**arXiv ID:** 2610.03029 | [PDF](https://arxiv.org/pdf/2610.03029v1)

**作者:** Drew Ross `[一作]` (University of Kansas), Zijun Yao `[通讯]` (University of Kansas)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SoftGene框架，利用蛋白质序列的层次注意力编码与软提示结合，自动生成基因集注释

**💡 创新点**

首次将预训练蛋白质语言模型嵌入软提示，并通过层次注意力实现可解释的基因集特征聚合

**🔧 技术方法**

预训练蛋白质语言模型(ESM-2)、层次注意力聚合、软提示(MLP投射)、局部LLM生成

**📊 数据集**

Gene Ontology（GO）与MSigDB（C2）两大基因集数据库

**📈 对比分析**

与零/少/一轮提示、提示调优、代理系统等基准相比，SoftGene在METEOR、ROUGE-L、BERTScore上显著领先，表现最优

**⚠️ 局限性**

受限于训练数据偏向已知基因、蛋白质嵌入贡献在不同GO域差异大，且评价仍基于参考文本，缺乏实验验证

---

## 436. PEEK: Heterogeneous Parallelism for Privileged Error Detection in Safety-Critical Processors

**arXiv ID:** 2610.03045 | [PDF](https://arxiv.org/pdf/2610.03045v1)

**作者:** Tinglue Wang `[一作]` (Southeast University), Zhe Jiang `[通讯]` (Southeast University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

设计并实现了PEEK——一种异构并行错误检测架构，首次将HPED扩展到完整的特权模式，并完成硅级实现；

**💡 创新点**

创新点在于：动态CSR快照机制、硬件级特权重放、死锁预防等技术，显著降低性能与面积开销，同时实现全系统（含特权）错误检测；

**🔧 技术方法**

采用RISC‑V Rocket/BOOM异构SoC，CSR分类与动态访问监控、参数化位过滤、Shadow CSRs、Context Switching Unit、Deadlock Handler等硬件技术，并配合Linux完整系统；

**📊 数据集**

在Linux v6.7上使用SPECint 2006、PARSEC、MiBench等标准工作负载，并进行门级故障注入；

**📈 对比分析**

与用户模式HPED、全特权HPED、nZDC、LockStep、EA‑LockStep等基线对比，PEEK在SPEC/PARSEC/MiBench上的几乎无慢速（1.02×左右），面积增幅仅1.8%/3.8%，检测延迟<4µs，覆盖率>65%；

**⚠️ 局限性**

对极高频繁特权切换的工作负载仍有轻微性能损耗，架构目前仅在RISC‑V平台验证，未评估在更大规模多核或其他ISA上的通用性。

---

## 437. Tailoring the Quantization Space for 1-Bit KV Cache Compression

**arXiv ID:** 2610.03027 | [PDF](https://arxiv.org/pdf/2610.03027v1)

**作者:** Minsoo Cheong `[一作]` (Seoul National University), Sungjoo Yoo `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了TaSQ，一种针对KV缓存的自定义空间向量量化方法，用于实现1比特级别的极低位压缩；

**💡 创新点**

创新点包括：①基于查询引导的通道加权以反映注意力误差敏感性；②跨头共享尺度归一化降低通道级开销；③协方差感知的通道分组以利用通道间相关性；所有变换兼容RoPE且可直接合并至投影权重与码本；

**🔧 技术方法**

主要技术：向量量化（VQ）、查询加权、RoPE预处理、协方差分析、Fisher加权k-means、SGLang+Triton推理核；

**📊 数据集**

使用多种数据集进行校准与评测：Wikitext‑2（校准）、GSM8K、MATH500、MBPP、HumanEval、BBH、MMLU、AIME 2024/2025、LiveCodeBench v6、SciBench、RULER（needle‑in‑a‑haystack）等；

**📈 对比分析**

与CQ、NSNQuant、NovaKV等基线在1比特键压缩下对比，TaSQ在大多数通用、推理和长上下文检索任务中取得最高准确率；在SGLang实现上，单 RTX 6000 Ada GPU时峰值吞吐量提升至1.87×，批量大小提升至14×；

**⚠️ 局限性**

局限性：仅针对键进行自定义量化，值仍使用传统VQ；依赖校准窗口和特定模型结构（RoPE、跨头共享尺度）；在极低比特率或非常大模型上可能需要进一步优化；TTFT（首个令牌延迟）略有提升；

---

## 438. Signal Simplification Is Not Predictive Simplification: Diagnosing Residual Neural Forecasting in Short-Horizon Volatility

**arXiv ID:** 2610.03019 | [PDF](https://arxiv.org/pdf/2610.03019v1)

**作者:** Bingqi Lian `[一作]` (University of Maryland), Jerry Wu `[通讯]` (University of Maryland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究短期波动率预测中统计-神经混合模型的残差预处理效果，评估残差简化是否能提升后续LSTM预测

**💡 创新点**

提出“forecaster–preconditioner asymmetry”诊断框架，区分统计预测与残差预处理的作用，发现残差压缩不一定带来LSTM性能提升

**🔧 技术方法**

使用HAR风格统计模型、单层LSTM、残差-仅LSTM混合、滚动起点评估、DM检验、QLIKE、MSE/MAE/R²评估以及系统运行时测量

**📊 数据集**

使用2020-2023年美国五个资产（SPY、QQQ、IWM、AMD、XOM）60分钟OHLCV数据，构造六观测滚动方差 proxy 作为目标

**📈 对比分析**

通过滚动原点交叉验证比较 AR、MA、ARIMA、HAR、Pure LSTM、HAR+LSTM；结果显示 HAR 在残差压缩上最优但 HAR+LSTM 预测误差高于 HAR；Pure LSTM 最佳预测但计算量大；系统运行时间显示 HAR 最轻量

**⚠️ 局限性**

局限：仅使用单变量、单层 LSTM；未加入额外上下文或多变量信息；只评估残差-仅混合模型；结果受特定目标构造和窗口长度影响

---

## 439. From Expression to Reaction: Role-aware Visual Transfer and Stimulus-guided Reasoning for Interlocutor Emotion Recognition

**arXiv ID:** 2610.03016 | [PDF](https://arxiv.org/pdf/2610.03016v1)

**作者:** Wei Wang `[一作]` (Guangdong University of Technology), Zhenguo Yang `[通讯]` (Guangdong University of Technology)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种角色感知刺激引导框架（RASG），在缺乏听者标签的条件下，通过视觉一致性筛选、听者领域自举伪标签以及仅在视觉不确定时使用受刺激上下文进行二分类约束推理，实现对听者情绪的识别。

**💡 创新点**

创新点包括：① 通过多模型视觉一致性判别对说话者标签进行校正，减少视觉与标签的不匹配；② 利用非说话者面部轨迹进行伪标签自举，构建听者域监督；③ 在视觉预测边界模糊时，采用受刺激上下文进行受限语言推理，仅在视觉Top‑2候选范围内做决策，避免过度依赖语言模型。

**🔧 技术方法**

技术手段包括：CLIP视觉编码器与LoRA微调、OpenFace行为特征分支、TalkNet说话检测、跨模型视觉一致性投票、DeepSeek大型语言模型进行受刺激边界推理。

**📊 数据集**

实验使用MER‑Cross对话情绪识别基准（含9,395名说话者训练样本和574名听者测试样本）以及MER‑SEMI无标签视频库（124,802条）进行自举。

**📈 对比分析**

与官方CLIP‑large基线（58.88%）相比，RASG提升至76.25% F1，显著高于其他单模或多模融合基线，并在ACM MM 2026 MER Grand Challenge的Track 1中排名第二。

**⚠️ 局限性**

局限包括：视觉一致性筛选依赖多模型投票，可能保留系统性偏差；伪标签误差在自举阶段可能被放大；受刺激推理仅在低边界样本中使用，无法覆盖所有复杂的多模态相互作用。

---

## 440. RYOPO: Bringing End-to-End Category-Level Object Pose Estimation into Real Time

**arXiv ID:** 2610.03013 | [PDF](https://arxiv.org/pdf/2610.03013v1)

**作者:** Hakjin Lee `[一作]` (PIT IN Co), Jaehoon Sim `[通讯]` (PIT IN Co)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个端到端的 RGB‑D 查询式集合预测框架（Real‑Time YOPO），同时完成检测、分割与类别级姿态估计。

**💡 创新点**

无需 CAD 先验或外部实例分割器，通过查询条件几何通道将观测 3D 点与查询关联，并使用姿态状态反馈进行递归残差校正，实现实时全帧姿态推断。

**🔧 技术方法**

基于 DETR/EdgeCrafter 的多查询解码器，结合点云投影、稀疏体素卷积场景编码、交叉注意力与姿态编码器，并采用一对一 Hungarian 匹配训练。

**📊 数据集**

在 NOCS（REAL275、CAMERA25）和 HouseCat6D 上进行评估。

**📈 对比分析**

与已发表的 RGB‑D 组合方法和两阶段方法对比，在 NOCS 上实现了 5°5cm 约 52.5 AP、REAL275 10°5cm 73.6 AP，并在 RTX A6000 上达到 31.8 FPS；总体上在所有对象评估中与两阶段方法竞争，优于大多数单阶段方法。

**⚠️ 局限性**

对 2 cm 误差阈值的姿态精度仍不及 AG‑Pose/CleanPose，且在 HouseCat6D 的 10°5cm AP 与 cuboid 对齐上略逊于单体裁剪方法。

---

## 441. PaNGEA: Parallel Node Generation and Exploration Algorithm on GPU

**arXiv ID:** 2610.03090 | [PDF](https://arxiv.org/pdf/2610.03090v1)

**作者:** Jean Pauphilet `[一作]` (London Business School), Yupeng Wu `[通讯]` (London Business School)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种名为PaNGEA的GPU加速MIP启发式，利用批量并行生成与探索多个节点；

**💡 创新点**

核心创新在于将节点生成策略并行化（不再单一挑选策略），并引入两阶段Local‑MIP搜索（先在松弛空间搜索，再限制节点边界），显著提升可行解质量；

**🔧 技术方法**

使用GPU实现的线性松弛求解（cuPDLP）和批量Local‑MIP本地搜索，结合CUDA核、矩阵‑矩阵乘法与批量投影；

**📊 数据集**

在MIPcc26（283个实例）和MIPLIB2017（233个实例）上进行实验；

**📈 对比分析**

与非商业Local‑MIP、Gurobi启发式以及无分支的Gurobi进行对比；PaNGEA在gap积分、最终gap和可行实例数上均优于其它非商业方法，并在高约束/变量比的实例上与Gurobi表现相近或更优；

**⚠️ 局限性**

受限于GPU版线性求解器的精度不足，导致节点松弛未能完全优化，影响上界与下界的强度；同时两阶段Local‑MIP在高维变量时的单坐标更新效率较低，影响在低约束/变量比实例上的表现。

---

## 442. Securing Computer-Use Agents Against Branch Steering Attacks

**arXiv ID:** 2610.03089 | [PDF](https://arxiv.org/pdf/2610.03089v1)

**作者:** Giulio Zingrillo `[一作]` (ETH Zurich), Robert Mullins `[通讯]` (University of Cambridge)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对电脑使用代理（CUA）的间接提示注入问题，提出了COBRA体系结构，利用分离的规划LLM与观测LLM以及提前设定的分支约束，实现在GUI和MCP接口上的控制流与数据流完整性防御。

**💡 创新点**

创新点在于把数据流完整性与分支级别的权限约束嵌入Dual‑LLM规划流程，并通过Branch Resolution Hub与HTTP/MCP代理统一执行，首次在STEER‑Bench上实现0%分支导向攻击成功率。

**🔧 技术方法**

主要技术包括Dual‑LLM架构、预编译分支约束的计划语言、BRH的动态分支决策校验、确定性HTTP与MCP代理以及MESA爬虫生成站点sitemap。

**📊 数据集**

使用的评估数据集包括自建的STEER‑Bench（101个任务，9个领域）以及公开的OSWorld‑MCP、OSWorld、OpenCUA、OS‑Harm、WASP、MCPTox和MCPSecBench等。

**📈 对比分析**

与ReAct和传统Dual‑LLM基线比较，COBRA在STEER‑Bench上攻击成功率从94.4%/89.5%降到0%，并保持97%的正常任务完成率；在外部基准中亦能完全阻止相关攻击。

**⚠️ 局限性**

局限性包括无法对自由文本语义或隐藏效果做强制校验、对SMTP/文件系统等非HTTP/MCP通道不覆盖、对缺乏站点sitemap或MCP工具描述时只能采用粗粒度主机白名单，以及规划对意外界面状态的预判仍有限。

---

## 443. Coda: Exploiting Admission Flexibility for Coding-Agent Serving

**arXiv ID:** 2610.03088 | [PDF](https://arxiv.org/pdf/2610.03088v1)

**作者:** Youhe Jiang `[一作]` (University of Cambridge), Yi Xu `[通讯]` (Meta)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了面向编码代理服务的Admission层，利用可调的状态准备顺序和执行分组来提升GPU利用率和SLO符合率。

**💡 创新点**

创新点在于同时考虑KV状态准备成本与上下文长度兼容性，分别设计了Tiered‑Aging状态准入、Compatibility‑Aware执行准入，并在多工作器环境中加入上下文兼容路由。

**🔧 技术方法**

采用多层KV缓存、CPU‑Shadow异步恢复、等待衰减、二分组注意力以及基于阈值的自适应决策等技术。

**📊 数据集**

使用TraceLab工作负载追踪，并在Qwen3‑30B与Qwen3‑235B模型上进行实验。

**📈 对比分析**

与CacheWise、SMetric、vLLM等基线相比，单机/多机设置下平均提升输出token吞吐20–23%，SLO符合率提升70–140%，同时保持或降低E2E延迟。

**⚠️ 局限性**

局限在于对KV层信息和参数调优的高度依赖，在极端高并发或不同模型规模下可能需要重新校准。

---

## 444. An automated pipeline for standardised speech-unit annotation in spontaneous dialogue

**arXiv ID:** 2610.03078 | [PDF](https://arxiv.org/pdf/2610.03078v1)

**作者:** Hanlu He `[一作]` (Technical University of Denmark), Ivana Konvalinka `[通讯]` (Technical University of Denmark)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发并评估了一套从分通道丹麦语双人自然对话录音中自动识别并标注发言轮次与背后反馈（backchannel）的全流程管线；

**💡 创新点**

创新在于将无监督VAD、能量阈值过滤、时序合并、Whisper多语种ASR以及基于上下文的后处理结合，能够在不依赖训练数据的前提下直接从音频得到交互单元，并在不同听觉条件下保持稳定性能；

**🔧 技术方法**

使用的技术包括5阶Butterworth高通滤波、rVAD无监督VAD、能量阈值过滤、3秒前瞻窗口合并、Whisper大型多语言ASR、词频熵分类以及基于交互上下文的规则后处理；

**📊 数据集**

数据集为99段10分钟的丹麦语自然对话（共33对同性别参与者），涵盖正常与非对称噪声两种听觉条件；

**📈 对比分析**

与人工参考注释通过IoU匹配评估F1和时间误差，整体F1约0.62，回声段时间误差中位数为0.15–0.18秒，噪声条件下性能无显著差异；与人类互评相比，人工一致性更高、误差更小；

**⚠️ 局限性**

局限性包括对ASR质量高度依赖、参数敏感、未测量注释效率、仅验证于丹麦双人对话、对短非词化backchannel识别效果欠佳、缺乏跨语言及多人场景的进一步验证。

---

## 445. Unmasking Propaganda: A Comparative Analysis of Masked and Causal Language Models

**arXiv ID:** 2610.03077 | [PDF](https://arxiv.org/pdf/2610.03077v1)

**作者:** Claudiu Creanga `[一作]` (University of Bucharest), Liviu P. Dinu `[通讯]` (University of Bucharest)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对SemEval-2020 Task 11 文章级技术分类任务的研究，提出并实现了新的SOTA结果；

**💡 创新点**

创新点在于将Masked Language Models（如DeBERTa V3 Large）与多种Causal LLMs（如Gemini Flash 2.5、GPT‑4）结合，使用两种提示策略（base与Chain‑of‑Thought），并在此基础上对模型进行两阶段微调与策略对比；

**🔧 技术方法**

采用的技术包括：基于XLM‑RoBERTa与DeBERTa V3的Masked LM微调，基于API的Causal LLM推理（Gemini、OpenAI、Anthropic、Mistral、Meta LLaMA 3），以及两种Prompting策略；

**📊 数据集**

使用的主要数据集为SemEval‑2020 Task 11的技术分类子任务（共536篇新闻，约8,981个标注实例）；

**📈 对比分析**

实验结果表明，DeBERTa V3在基准微调后达到63.18 F1，Gemini Flash 2.5在base提示下达成最高63.62 F1；两类模型的性能可互补，提示策略影响精度与召回的平衡，未见统一最佳提示；

**⚠️ 局限性**

主要局限包括潜在的训练数据泄露风险、对多语言或跨域适用性的验证不足，以及对单一模型的泛化能力不足，建议未来采用集成与跨语言方法进一步提升。

---

## 446. MintEval: Do LLMs Implement the Trading Strategy You Asked For? A Behavioural-Equivalence Benchmark for Natural-Language-to-Strategy Code

**arXiv ID:** 2610.03080 | [PDF](https://arxiv.org/pdf/2610.03080v1)

**作者:** Siyu Wang `[一作]`, Yuecheng He `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 MintEval 基准，用于评估 LLM 在生成交易策略代码时是否真正实现了给定的自然语言描述。

**💡 创新点**

创新点包括：① 引入行为等价度（bar 级别）指标取代单元测试；② 将任务复杂度拆分为描述长度与状态跨度两条轴；③ 通过程序生成与反向翻译构造无污染任务；④ 结合可执行比较与 LLM 代码审查，揭示传统代码评判器对“静默失效”的盲区。

**🔧 技术方法**

采用 LLM 程序生成器、反向翻译器、行为匹配指标（ActionMatch、TradeF1、|ES|）、以及 QuantCode‑Bench 代码审查模型进行评估。

**📊 数据集**

使用 BTCUSDT 15 分钟行情数据，并在包含滑点与手续费的模拟环境中回测。

**📈 对比分析**

在开放和封闭两种设置下，对四类模型（低成本与开源编码器）进行 CompileOK、SpecMatch、ActionMatch、TradeF1、|ES| 的对比，结果显示低成本模型 ActionMatch 约 0.55，前沿 Claude Opus 5.5 在开放设置下达 0.89。

**⚠️ 局限性**

局限性包括任务规模有限（仅单一品种、任务数量少）、说明文本由 LLM 生成可能偏向模型友好、缺少真实策略分布、未考虑实时延迟及交易细节等。

---

## 447. Smart Sensing for Safer Bridges: From Sensor Signals to AI-Driven Anomaly Detection

**arXiv ID:** 2610.03082 | [PDF](https://arxiv.org/pdf/2610.03082v1)

**作者:** Rahul Jaiswal `[一作]` (Smart Sensor Systems AS), Halvor Heiberg `[通讯]` (Smart Sensor Systems AS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

比较信号处理与Isolation Forest两种方法在桥梁传感器数据中的异常检测效果，构建实时检测框架并进行多指标评估与控制注入实验；

**💡 创新点**

将传统峰值检测与无监督机器学习算法结合，提出互补的异常检测流程，并通过时间一致性与注入实验展示两者的敏感性差异；

**🔧 技术方法**

信号峰值与谷值检测算法、Isolation Forest无监督学习、数据预处理（线性插值、网格搜索参数调优）；

**📊 数据集**

iBridge设备在挪威桥梁采集的11天5Hz数据，包含ADC1、ADC2、ADC3三路金属梁下压测量，共3,774,011个样本；

**📈 对比分析**

使用异常数量、检测时间、处理速率、异常率、时间一致性、召回率等指标进行比较；信号处理检测速度快、计算量低但异常数少；Isolation Forest召回率高、异常数多但耗时较长，处理速率可接受，时间一致性低；

**⚠️ 局限性**

缺乏真实标签导致评价基于假设；两种方法对不同传感器灵敏度差异大；仅单桥单模态数据，未验证多桥多传感器泛化；实时性能仍需进一步优化；

---

## 448. Zephon: Elastic Determinism for Online, Stateful Foundation Model Data Loading Pipelines

**arXiv ID:** 2610.03087 | [PDF](https://arxiv.org/pdf/2610.03087v1)

**作者:** Maximilian Böther `[一作]` (DatologyAI), Bogdan Gaza `[通讯]` (DatologyAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个名为Zephon的在线状态化数据加载器，能够在不牺牲吞吐量的前提下为基础模型训练提供弹性确定性、可插拔的n‑to‑m操作以及高效恢复机制；

**💡 创新点**

核心创新包括：①基于“lane”抽象的弹性确定性方案，将全局数据序列划分为拓扑无关子流；②使用累加器串行化状态化操作的排序决策，同时在每个阶段并行执行无状态工作；③通过flush sentinel和段分割实现可恢复的状态快照；④共享内存协同和GIL移除等技术实现进程/线程切换兼容；

**🔧 技术方法**

实现技术涵盖：operator‑graph 架构、累加器+worker 模式、lane 及多路复用器、checkpoint 机制与 replay‑filter、flush sentinel + 片段化、共享内存（SHM）协同、Python 3.13/3.14t GIL‑free 线程、进程/线程运行时、自动化验证与校验；

**📊 数据集**

使用的数据集：大规模文本数据（1B 规模）与多模态 VLM 数据（MAmmoTH‑VL、Vision‑Language 文档），以及在 DCLM Core v1 与 FineWeb 上的评估集；

**📈 对比分析**

与 LitData、Mosaic Streaming、Grain、InternalDs 等基线对比。fetch‑and‑batch 基准下，Zephon 在文本场景下达到 59k samples/s，VLM 场景下 16k samples/s（≈2–3× 基线），在 VLM 预处理流水线中每 GPU 可用 token 通过率 29k utps；训练时 loss 轨迹在不同 GPU 数量下差异 <0.1 pp；checkpoint 大小约 3 MiB，恢复时间约 70 s，且恢复成本随训练进度保持稳定；

**⚠️ 局限性**

限制与改进点：①仍未覆盖跨节点分布式恢复与多机异构（GPU‑CPU）场景；②线程运行时受 GIL 限制，需进一步优化多核/多进程协同；③flush‑sentinel 的频率与片段大小需在不同管道中调优；④对 GPU 级数的随机数种子一致性仍有限；⑤目前实现依赖 Python 生态，导致在极大规模（>64 GPU）或高延迟存储环境下的性能瓶颈仍待研究。

---

## 449. Learning Transferable Policies from Action-free Time Series Through Dynamical Embeddings

**arXiv ID:** 2610.03065 | [PDF](https://arxiv.org/pdf/2610.03065v1)

**作者:** Niklas Emonds `[一作]` (Heidelberg University), Georgia Koppe `[通讯]` (Heidelberg University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种层级模型基础强化学习框架，利用多系统的共享结构在无动作记录的情况下学习系统特定的控制策略，并能将训练得到的策略迁移到未见系统。

**💡 创新点**

创新点在于：① 将层级动力学重建模型（dsr）与基于离散时间pwl的自回归神经网络结合，用低维嵌入捕捉个体差异；② 用这些嵌入参数化共享的SAC策略和价值网络，实现跨系统的参数共享与零样本迁移；③ 通过线性解码器投影将动作限制在可观测变量的零空间，实现跨模态的可解释干预；④ 利用pwl动态模型分析局部线性动力学，解释控制机制。

**🔧 技术方法**

使用了层级动力学重建（hierarchical dsr）、piecewise‑linear recurrent neural network (alrnn)、Soft Actor‑Critic (SAC) 强化学习、行动投影与解码器约束、以及局部线性稳定性分析。

**📊 数据集**

在合成数据集上验证：64 个不同参数的 Lorenz‑63 系统、16 个不同质量长度的双摆系统；在神经行为记录数据上使用了 LINK（12 个长时段的猴子手指运动记录）和一个延迟中心外伸展任务的单会话记录。

**📈 对比分析**

与独立训练的 SAC、基于在线规划的 CEM‑MPC、TD‑MPC 以及从真实系统交互中训练的 DreamerV3 进行对比。层级 SAC 在 Lorenz‑63 真实系统上平均奖励约 -0.04（相对独立 SAC -0.17），在双摆上平均奖励约 -9.7（相对独立 SAC -16.8）。对未见系统的零样本迁移在 Lorenz‑63 上表现优于使用训练嵌入的基线，双摆上表现相似。

**⚠️ 局限性**

局限性包括：① 层级嵌入假设系统差异可线性刻画，可能不适用于非线性差异；② 只在无动作记录的情况下学习动作对系统的真实影响，无法验证实际干预效果；③ 需要足够丰富的系统族覆盖才能实现良好的迁移；④ 动作建模为加性潜在扰动，未解决真实物理效应的映射问题。

---

## 450. When Does Synthetic Relational Data Teach Models to Use Relations? Tracing Predictive Structure from Pretraining Data to Model Behavior

**arXiv ID:** 2610.03057 | [PDF](https://arxiv.org/pdf/2610.03057v1)

**作者:** Shivam Dubey `[一作]` (Lexsi Labs), Vinay Kumar Sankarapu `[通讯]` (Lexsi Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了合成关系型预训练数据中外键结构对模型学习的影响，并通过多阶段实验验证了“关系预测必要性”与模型对外键的功能依赖之间的因果关系。

**💡 创新点**

提出可度量的“关系预测必要性”指标，系统关联数据属性、模型内部机制与下游性能，揭示外键信息在预训练中如何驱动跨表计算。

**🔧 技术方法**

使用 Relational Transformer 进行掩码单元重建预训练，结合结构扰动、内部路径消融、LightGBM 预测差异评估以及 RelBench 上的因果干预实验。

**📊 数据集**

采用四套合成数据库集（RelDiff、GRDM、PluRel、RDB-PFN）以及真实 RelBench 数据库（user‑engagement、user‑churn 等）。

**📈 对比分析**

通过同一批次的掩码重建损失、FK扰动敏感度、内部路径消融以及下游 AUROC 变化进行比较；RelDiff 在多项指标上表现最佳，但其优势高度依赖外键结构，且在部分任务上外键干预甚至提升性能。

**⚠️ 局限性**

仅针对单一架构和预训练目标，未对生成器进行随机化对照，样本与训练随机性有限，且合成数据在类型处理上存在差异，限制了结论的普适性。

---

## 451. BeeWhere: Segmenting Bumble Bee Colonies to Quantify Behavioral Effects

**arXiv ID:** 2610.03051 | [PDF](https://arxiv.org/pdf/2610.03051v1)

**作者:** Roberta Hunt `[一作]` (University of Copenhagen), James Crall `[通讯]` (University of Wisconsin-Madison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发并验证 BeeWhere AI 辅助标注、实例分割与 ArUco 跟踪相结合的工作流，用于定量蜜蜂巢内行为和种群增长。

**💡 创新点**

创新点在于将高精度实例分割与传统标记跟踪融合，在遮挡严重的密集巢内实现更完整的检测和身份保持，并提供可复现的工具与预训练模型。

**🔧 技术方法**

使用 YOLOv8（YOLO26）进行检测与分割，SAM2 辅助生成掩膜，ByteTrack、SimpleIOU、Centroid 匹配等多种跟踪算法，结合 ArUco fiducial 标签实现身份重识别。

**📊 数据集**

数据集包括 483 帧 8,443 只蜜蜂实例、各类巢结构和花粉球标注，以及 3,761 条 10 秒视频（3 轮 imidacloprid 处理实验），共涉及 24 个微殖（每组 8 只工蜂）。

**📈 对比分析**

与仅 ArUco 跟踪相比，实例分割检测数提升 2.5 倍；在验证集上，结合实例分割与 ArUco 的 Centroid-200 方法 MOTA 超过 90%，IDF1 超过 94%；行为指标（如最近邻距离）显示处理组间显著差异。

**⚠️ 局限性**

局限性：实验样本量小（每剂量 3 个微殖），未达到统计显著性；巢结构分割精度低、易与花粉球混淆；跟踪过程仍存在一定 ID 交换；仅在室内模拟环境验证，外延性待进一步研究。

---

## 452. Adaptive Second-Order Solvers for Fast Stochastic Diffusion Sampling

**arXiv ID:** 2610.03034 | [PDF](https://arxiv.org/pdf/2610.03034v1)

**作者:** Ella Kemperman `[一作]` (Radboud University), Luca Ambrogioni `[通讯]` (Radboud University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了适用于扩散模型的比例-积分（PI）自适应步长控制器，并将其与噪声归一化误差估计结合，以生成可针对二阶采样器的自适应或平均时间步长调度。

**💡 创新点**

将PI控制引入扩散采样，结合噪声归一化误差估计，既在每个样本上实现平滑自适应步长，又可聚合为数据驱动的固定调度。

**🔧 技术方法**

数值分析中的PI控制、误差估计、随机Heun方法、概率流ODE、线性插值聚合；在多模态数据上进行实验。

**📊 数据集**

自然图像（FFHQ、ImageNet 64×64）和语言（LM1B）以及一维高斯混合。

**📈 对比分析**

与Euler-Maruyama、Stochastic Heun、EDM‑churn、Gotta Go Fast等基准在固定NFE下比较，PI自适应在图像上优于EDM调度、在语言上在低至中等NFE下优于EDM‑churn，平均调度在多数情况与自适应相近。

**⚠️ 局限性**

仅在随机采样上验证；未完整探索超参空间；仅使用VE模型；对确定性采样的潜在优势留待未来研究。

---

## 453. When Numbers Start Talking: Numerical Signalling and Strategic Behaviour Among LLMs

**arXiv ID:** 2610.03033 | [PDF](https://arxiv.org/pdf/2610.03033v1)

**作者:** Alessio Buscemi `[一作]` (Luxembourg Institute of Science and Technology), Pietro Liò `[通讯]` (University of Cambridge)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型（LLM）在四种典型博弈中的非语言数值通信对合作水平和行为的影响。

**💡 创新点**

创新点在于发现所有LLM在被指令的数值通信中会自发地产生以收益表为锚点的结构化符号，并且这一结构在不同模型中具有可比性，而行为效应仅取决于模型的先验偏好。

**🔧 技术方法**

使用的方法包括信息论熵、互信息检验、排列检验、回归相关和游戏理论框架下的博弈模拟。

**📊 数据集**

数据集为通过FAIRGAME框架生成的模拟博弈记录，包括四款游戏（PD、SD、SH、H）、四种LLM、不同人格组合、通信模式和多次重复。

**📈 对比分析**

对比结果显示：在所有模型中，数值信号的熵显著低于随机基线；行为上，数值信号对合作的影响因模型而异，GPT-4o和DeepSeek-V3在某些游戏中提升合作，而Mistral和Claude则表现相反；整体上通信未能稳定带来预期均衡。

**⚠️ 局限性**

局限性包括仅测试指令式而非自发通信、仅有两类人格、固定博弈与支付、有限重复步数、模型解码差异、消息顺序未平衡，以及行为关联检验样本有限。

---

## 454. LS-AR: Future-Predictive Latent Steering in Autoregressive LLMs

**arXiv ID:** 2610.03093 | [PDF](https://arxiv.org/pdf/2610.03093v1)

**作者:** Anubha Gupta `[一作]` (University College London), Eduardo Pignatelli `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种双通道的自回归架构LS‑AR，将宏观目标的连续潜在表示与离散令牌生成解耦，构建静态与动态目标编码器；

**💡 创新点**

创新点在于通过FiLM条件化将单一连续潜在向量注入解码器，实现宏观目标的持续引导，同时提供静态保留与动态轨迹更新两种运行范式；

**🔧 技术方法**

技术手段包括Joint Embedding Predictive Architecture (JEPA) 预训练、VICReg对齐、FiLM层注入、触发式动态状态跟踪以及基于多阶段训练的目标对齐与条件生成；

**📊 数据集**

实验使用了两套合成符号环境：Treasure Hunt（目标检索与数字通行码）和Blocksworld（逻辑规划与错误恢复），并在固定参数规模下进行评测；

**📈 对比分析**

与参数匹配的标准自回归基线对比，LS‑AR在超出上下文窗口（H=1024,W=500）时保持100%语义召回、约35%吞吐提升，动态版本在受扰动情境下完成率达89%，显著优于基线；

**⚠️ 局限性**

局限性包括单向量潜在表示在离散对象扩展（N→N+1）时失效、仅在合成数据上验证、对数值通行码的细粒度压缩产生误差、缺乏与连续内存增强模型的对比以及对真实自然语言场景的适用性待验证。

---

## 455. Beyond Predefined Sinks: Security-Aware Dependency Analysis for LLM Agents

**arXiv ID:** 2610.03014 | [PDF](https://arxiv.org/pdf/2610.03014v1)

**作者:** Hang Cui `[一作]` `[通讯]` (University of Chinese Academy of Sciences), Hang Cui (University of Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一套针对LLM代理程序的安全感知静态分析框架（AgentSecGraph），并构建了可复现的67个开源代理仓库语料库（AgentSecBench）。

**💡 创新点**

创新点在于将预定义的安全敏感操作作为分析锚点，进一步挖掘并附加代理相关性、源头追溯、依赖关系、信任边界、保护措施和外部影响等安全上下文，从而实现对相同低层操作的多维安全解读；同时通过可复现的复制实验建立了保守的安全行为基准集。

**🔧 技术方法**

技术上采用语言AST和语法分析相结合的静态追踪（Python使用AST、TypeScript/JavaScript使用语法级回溯），框架适配器将不同代理框架的工具声明统一映射，生成候选中心化的安全依赖图（Security‑Aware Agent Dependency Graph）。

**📊 数据集**

使用数据集为67个公开LLM代理仓库，涵盖11个生态系统，37,542文件，产生23,866个安全敏感操作候选，并通过复制实验筛选出22个安全行为基准（1个已确认漏洞、1个待披露候选、20个受保护非漏洞）。

**📈 对比分析**

在同一套仓库上与 sink‑only 与简化 ADG 进行对比，完整表示在已验证的9个案例中保留了91.1%的安全上下文并显示全部5个保护措施；sink‑only仅保留20%，简化 ADG 40%；分析总耗时约50.8分钟，平均7.83候选/秒。

**⚠️ 局限性**

局限性包括：静态分析仅局部、AST/词法级别，未覆盖完整的全程、别名、控制流与动态特性；仅在Python中实现完整的依赖追踪；未能恢复运行时特权信息；复制基准集规模有限，难以估计整体漏洞率；以及对不同语言与框架的适配不均衡。

---

## 456. Small universal multiset reaction systems

**arXiv ID:** 2610.03021 | [PDF](https://arxiv.org/pdf/2610.03021v1)

**作者:** Andrei Paun `[一作]`, Annemarie-Beatrix Messner `[通讯]` (University of Bucharest)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31`

**🎯 论文内容**

在多集反应系统框架下构造了一个包含23条指令、87种元素的最小通用机器，利用最大反应体积优先级实现指令的顺序执行；

**💡 创新点**

创新点在于首次证明多集反应系统在最大顺序化模式下具备通用性，并提出一种基于反应体积优先的序列化机制；

**🔧 技术方法**

使用多集反应系统（Multiset Reaction Systems）理论与优先级调度技术来模拟寄存器机；

**📊 数据集**

本研究为理论证明，未使用具体实验数据集；

**📈 对比分析**

通过与已知的寄存器机模型对比，证明该系统能够模拟任意图灵机，性能以所需元素数（87）为衡量；

**⚠️ 局限性**

限制在于元素数量仍较大，序列化机制对多集系统的扩展性有限，且仅针对特定的小型通用机器模型；

---

## 457. Balancing Multimodal Learning via Functional Progress

**arXiv ID:** 2610.03035 | [PDF](https://arxiv.org/pdf/2610.03035v1)

**作者:** Zhongjing Gu `[一作]`, Yang Yang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出Function‑Space Guided Multimodal Optimization (FGMO)，通过函数空间进度估计(RFP)对多模态训练进行可比的进度量化并协调优化；

**💡 创新点**

创新点在于：①使用函数空间响应估计和无模态参考对比得到可比的相对进度RFP；②基于RFP的功能响应控制(FRC)，通过KL约束动态分配张量级学习率，实现多模态平衡；③提供一阶收敛性理论证明；

**🔧 技术方法**

主要技术包括函数空间响应估计、对齐无模态参考、RFP构造、FRC控制、张量级学习率调节、KL约束、理论分析与一阶近似；

**📊 数据集**

使用六个公开多模态数据集：CREMA‑D、KSounds、VGGSound、Twitter、Sarcasm、NVGesture；

**📈 对比分析**

与多种融合与重平衡基线（Concat、Affine、ML‑LSTM、G‑Blend、MSLR、OGM、PMR、AGM、MMPareto、SMV、MLA、DI‑MML、ReconBoost、LFM、AMSS+、InfoReg、ARL、AUG、DecAlign、RGM）进行对比；在所有数据集上均实现了最高或最接近最高的准确率/mAP/F1，显著提升性能；

**⚠️ 局限性**

局限性包括：需预先构建无模态参考导致额外计算成本；对调整间隔K和控制增益λ敏感；理论证明基于一阶近似，实际收敛性受参数配置影响；在极少数任务或数据规模较小时提升不明显。

---

## 458. Parasitic Co-Denoising: Unlocking 3D Human Motion Generation in a Frozen Video Diffusion Model

**arXiv ID:** 2610.03047 | [PDF](https://arxiv.org/pdf/2610.03047v1)

**作者:** Yunjiao Zhou `[一作]` (Nanyang Technological University), Jianfei Yang `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种从冻结的视频扩散模型中解码3D人体运动的新方法——寄生共去噪（Parasitic Co‑Denoising），无需训练单独的运动生成器即可在生成视频的同时得到对应运动；

**💡 创新点**

创新点在于发现视频扩散模型在整个去噪轨迹上携带可恢复的运动信号，并设计了与主模型共享噪声时间表、σ自适应多层融合的寄生运动解码器（PMD），实现了对运动生成的显式解码；

**🔧 技术方法**

主要技术包括：冻结视频扩散模型（Wan2.1）、对视频与运动潜在空间进行对齐的阶段一、共享时间表的流匹配解码器阶段二、σ自适应多层特征融合、适用于每个时间步的流匹配损失训练；

**📊 数据集**

使用了ViMoGen-228K（含171.5K MoCap文本-运动对、56.6K 实景文本-视频-运动三元组和约11K 合成三元组）以及HumanML3D、MBench等公开基准；

**📈 对比分析**

与传统文本到运动生成模型（如ViMoGen、MotionVLA、MotionCraft等）比较，PMD在语义一致性和运动多样性等指标上均表现更佳，且参数量仅为传统模型的少数比例；

**⚠️ 局限性**

局限性包括：仍依赖冻结的视频扩散模型的质量，难以跨域推广；对非常细粒度的运动控制支持不足；以及对大规模运动数据的依赖性在一定程度上降低了模型的独立性。

---

## 459. CrowdOcc: Monocular Semantic Scene Completion for Quadruped Robots in Crowded Indoor Environments

**arXiv ID:** 2610.03031 | [PDF](https://arxiv.org/pdf/2610.03031v1)

**作者:** Feiyang Chen `[一作]` (Tongji University), Yuanjian Zhang `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了CrowdOcc数据集和一种针对四足机器人在拥挤室内环境中单目语义场景完成的完整框架。

**💡 创新点**

创新点在于：① Normal‑Guided Scene Geometry Fusion（NGSGF）通过表面法线与深度融合，提升在遮挡环境下的静态几何恢复；② Human‑Centric Sparse Interaction（HCSI）利用稀疏人类候选并进行自注意/交叉注意，显著改进人类占据空间的完整性和空间定位。

**🔧 技术方法**

主要技术包括：MoGe‑2 估计单目深度与法线，NGSGF 进行窗口化的表面法线引导融合；HCSI 进行人类候选路由、Top‑K 选择、稀疏自注意与交叉注意；整体采用基于 ISO 的深度感知提升与体素化编码解码结构。

**📊 数据集**

使用CrowdOcc数据集，包含 25.1K RGB‑D 帧、11 个室内场景、50×60×96 的 0.08 m 体素网格，并提供语义占据标签；同时利用同步 LiDAR 与 IMU 进行离线构造。

**📈 对比分析**

与 SplatSSC、GPOcc、EmbodiedOcc、MonoScene、NDC‑Scene、ISO 等现有方法在跨场景测试集上进行对比，CrowdOcc 取得 15.80 IoU、11.40 mIoU、46.23 Human IoU，显著优于对手，特别是在人类占据准确性上提升了约 2.4 点。

**⚠️ 局限性**

局限性包括：对单帧点云的依赖导致对快速运动或严重遮挡的人体重建不稳定；法线与深度估计误差可能影响 NGSGF 性能；目前仅验证了室内四足视角，缺乏对其他平台或户外场景的适用性验证。

---

## 460. HyperThink: Text-to-Parameter Hypernetworks for Efficient Reasoning

**arXiv ID:** 2610.03039 | [PDF](https://arxiv.org/pdf/2610.03039v1)

**作者:** Donggyun Kim `[一作]` (KAIST), Seunghoon Hong `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种文本到参数的框架，利用轻量级的超网络在推理时对大型语言模型的部分偏置进行查询条件化更新，从而在不生成长形式推理轨迹的情况下实现多步推理；

**💡 创新点**

创新点在于将思考过程抽象为查询条件化的参数更新，并通过带向量量化瓶颈的超网络实现离散、可复用的推理原型，显著降低推理延迟而保持推理质量；

**🔧 技术方法**

采用的技术包括：冻结的文本编码器、基于MM‑DiT的偏置编码器、向量量化（VQ）解码器、最大似然训练与VQ正则化，以及仅对偏置参数进行轻量级更新；

**📊 数据集**

使用的数据集涵盖数学推理（GSM8K、DeepScaleR、MATH‑500）和通用推理（CodeForces‑CoTs、LogiQA、OpenBookQA、QASC、AIME、LiveCodeBench、BIG‑Bench Hard、CommonsenseQA）等；

**📈 对比分析**

与基准方法（思考模式、预算控制思考模式、本地无思考模式、System 2 迁移、TokenSkip）对比，实验显示在低延迟区间内取得了更高的 Pass@5/准确率，且 FLOPs 与无思考模式相近；

**⚠️ 局限性**

局限性包括：在高预算/高难度任务（如 AIME、LiveCodeBench）仍无法完全取代完整思考模式；仅更新偏置可能限制模型的表达能力；对训练数据覆盖范围的依赖导致在未见领域的泛化可能受限；

---

## 461. Verifiable, Articulable, and Tacit Components of Preference

**arXiv ID:** 2610.03025 | [PDF](https://arxiv.org/pdf/2610.03025v1)

**作者:** Alexander Spangher `[一作]` (Stanford University), Sanmi Koyejo `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

该论文通过构建CreativePreferences大规模数据集，研究并量化创意领域的可表述性与可验证性缺口，揭示人类隐性偏好对AI模型的影响

**💡 创新点**

创新点在于提出可表述性与可验证性缺口概念，开发自动化指标发现与评估方法，并系统评估七个创意域的 42 个任务，首次量化隐性偏好的存在与影响

**🔧 技术方法**

使用自动化指标发现算法（AutoMetrics、ExperiGen、HypoGeniC、HypotheSAEs）、程序合成与提示优化（GEPA、ε-估计）、捕获-重现法估计未发现指标、稀疏回归与Transformer 预训练模型进行性能评估

**📊 数据集**

使用 2.8M 文本与 317M 人类偏好标注构成的 CreativePreferences 数据集，涵盖数学、编程、法律、学术、新闻、创意写作与幽默等七个创意领域的 42 个任务

**📈 对比分析**

通过对可表述性与可验证性指标的上限与下限进行估计，并与完整模型（VAT）比较，发现大多数任务存在 20%–40% 的缺口，完整模型在预测人类偏好时表现更好，证明缺口对决策具有显著影响

**⚠️ 局限性**

局限包括：缺口定义基于模型性能差异，未直接验证为人类隐性知识；指标发现与评估仅使用自动化方法，可能遗漏专家手工制定的更优规则；判别器为 LLM，可能引入判断偏差；数据集来源多样但仍受标签噪声与领域偏差影响

---

## 462. Personalized Automatic Speech Recognition for a Dysarthric and Tracheostomic Speaker using Artificial Conversations

**arXiv ID:** 2610.03017 | [PDF](https://arxiv.org/pdf/2610.03017v1)

**作者:** David Nadrchal `[一作]` (Johannes Kepler University Linz), Paul Primus `[通讯]` (Johannes Kepler University Linz)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `67630363-6be0-4f51-ab05-7198250671a5` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套针对重度运动障碍和永久气管瘘的捷克语说话者的个性化自动语音识别系统；

**💡 创新点**

创新点在于提出“人工对话”数据采集协议、构建大型单说话者障碍语料库、以及多阶段Fine‑tune与音频模拟的训练管线；

**🔧 技术方法**

采用Whisper Base模型作为基座，结合Common Voice、Ortofon数据的预训练、合成气管瘘语音模拟、知识蒸馏、设备级适配、个性化VAD和词汇引导等技术；

**📊 数据集**

使用了33小时的标注障碍语料（人工对话、阅读、问答、即兴对话等）以及常规捷克语语料（Common Voice、Ortofon），并在GitHub公开了数据和代码；

**📈 对比分析**

与未训练Whisper Base、非受限捷克语Fine‑tune版及“仅说话者”版对比，在三种实时场景（脚本对话、问答、即兴对话）中，系统将字符错误率（CER）从约0.94降至0.43-0.63，显示相对40-90个百分点的提升，并在低上下文场景下接近经验人类听者的准确率；

**⚠️ 局限性**

局限性包括：仍高于非障碍语音的CER，主要误差来源为插入错误和语音分割；依赖大量手工标注；缺乏多说话者和多模态（视觉、对话历史）支持；对极端低能量、无周期语音的模拟效果有限。

---

## 463. ReSCUE: Re-translation with Sentence Commitment for Unsegmented Long-Form Simultaneous Sign Language Translation

**arXiv ID:** 2610.03022 | [PDF](https://arxiv.org/pdf/2610.03022v1)

**作者:** Sihan Ren `[一作]` (ShanghaiTech University), Minye Wu `[通讯]` (University of Derby)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种用于连续无分段长篇手语视频的实时翻译框架 ReSCUE，能够在流式输入下实现低延迟、稳定的手语到文本翻译。

**💡 创新点**

创新点包括：1) 训练时模拟真实流式条件（前缀、无声段、多句）以提升鲁棒性；2) 结合重译与基于偏置搜索+Mask-k的稳定机制；3) 句子承诺机制自动检测句子边界并管理内存。

**🔧 技术方法**

采用基于姿态估计+多模态语言模型的 Uni-Sign 架构，进行推理时的分块输入、重译、偏置搜索、Mask-k、句子承诺；并使用前缀截断、无声段、多句训练样本。

**📊 数据集**

在标准句子级数据集 CSL‑Daily 与 Phoenix‑2014‑T 上评估；并构建新长篇无分段数据集 How2Sign‑Long 与 CSL‑Story 进行长篇测试。

**📈 对比分析**

与多种离线与实时同义翻译方法（如 SimulSLT、CTL++、CV‑SLT 等）对比；ReSCUE 在低延迟下 BLEU 与 ROUGE 领先，低延迟配置可达 693 ms 并保持 23.32 BLEU；高精度配置 BLEU 约 24.72，靠近离线最优。

**⚠️ 局限性**

主要局限包括：提前或延迟句子承诺导致的翻译错误；对无声段检测的依赖；以及姿态估计噪声、词表缺失导致的误译。

---

## 464. The Geometry of Knowledge Accessibility in Large Language Models

**arXiv ID:** 2610.03052 | [PDF](https://arxiv.org/pdf/2610.03052v1)

**作者:** Lihu Chen `[一作]` `[通讯]` (Imperial College London), Lihu Chen (Imperial College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大语言模型中知识可访问性的几何结构，并提出了以“可访问性中心”为核心的几何假设。

**💡 创新点**

创新点在于将知识可访问性与模型内部表示的空间几何关系关联，发现可访问查询在表示空间中聚集于一个中心，距离中心越近可访问性越高，并将该几何结构用于预测不同干预策略的效果。

**🔧 技术方法**

主要技术包括：从查询单词层提取隐藏表示；构建中心化几何（球面/椭球）与线性、MLP 基准进行对比；学习可访问性中心并计算查询到中心的距离；通过该距离评估可访问性、稳定性、答案一致性，并在此基础上设计查询重写、链式推理和检索增强三种干预策略。

**📊 数据集**

使用四个问答基准：TriviaQA、SciQ（知识可访问性高、推理低）、MuSiQue、GSM8K（推理需求高），以及MQuAKE用于对齐知识可访问性与推理的对照实验。

**📈 对比分析**

与线性、MLP 等基准对比时，中心化几何在知识主导任务（TriviaQA、SciQ）上表现相当甚至优于更灵活的 MLP，说明其在捕获可访问性信号方面高效；在推理主导任务（MuSiQue、GSM8K）中，MLP 仍获优势，但中心化几何在参数量上更轻量。实验还展示了距离可访问性中心的排序可跨数据集转移，并揭示不同干预在不同可访问性区间的效果差异。

**⚠️ 局限性**

局限性包括：仅针对单事实查询；使用正确率作为可访问性的代理，无法区分知识缺失与检索失败；实验多为相关性研究，缺乏因果性验证；未探讨中心如何在预训练阶段形成，也未将几何扩展到推理能力。

---

## 465. OmniAct3D: Leveraging Foundation Geometry and Evidence-Grounded Reasoning for Panoramic 3D Detection

**arXiv ID:** 2610.03015 | [PDF](https://arxiv.org/pdf/2610.03015v1)

**作者:** Runtong Wu `[一作]` (Hunan University), Kailun Yang `[通讯]` (Hunan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在单幅等距投影图像上实现了面向移动实体智能体的 3D 检测，通过适配预训练的视觉基础模型来预测物体的 3D 位置、尺寸与朝向。

**💡 创新点**

创新点在于三方面：① 引入 ERP-Ray Geometry Adapter 对齐预训练几何先验与等距投影的球面及周期结构；② 设计 Visual-Action Reasoning Chain，将每个检测 hypothesis 与全景证据关联并转化为几何动作；③ 使用 Appearance-Guided Heading Expert 在局部区域重编码以恢复朝向估计的细节。

**🔧 技术方法**

采用视觉基础模型（如 DINO、SAM 等）与几何基础模型（如 UniDepth、MoGe），结合 ERGA‑Ray、VARC 与 AGHE 三个模块。

**📊 数据集**

在 Spheriverse 与 PanoMMOcc 两个等距投影 3D 检测基准数据集上进行实验。

**📈 对比分析**

相较于现有基准，OmniAct3D 在 Spheriverse 上 mAP 提升 2.96 NDS 点、在 PanoMMOcc 上 mAP 提升 24.87 点，并在跨配置迁移时保持 95–98% 的 mAP。

**⚠️ 局限性**

主要局限在于需要为每种传感器配置训练专属 ERGA‑Ray，且方法在极端视场畸变或高速动态场景下的鲁棒性尚待进一步验证。

---

## 466. WebFovea: When the Model Is Right but the Click Is Wrong -- Reliable Round Trips for Vision-Based Web Agents on Live Websites

**arXiv ID:** 2610.03036 | [PDF](https://arxiv.org/pdf/2610.03036v1)

**作者:** Jiangang Han `[一作]` `[通讯]` (Independent Researcher), Jiangang Han (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种视觉驱动的网页代理，利用四阶段循环（解析、执行、反馈、观察）和守护机制实现对真实网站的可靠交互；

**💡 创新点**

提出了四阶段视图与守护层的框架，系统性分析并解决解析、坐标对齐、iframe、悬停、文本检索等多阶段失效问题，显著提升任务完成率；

**🔧 技术方法**

采用大语言模型（Claude 4），Chrome DevTools Protocol、像素坐标映射、DOM指纹、OCR与哈希、自动重试、预算限制等技术；

**📊 数据集**

使用WebRetriever Benchmark的Protocol III任务集，包含公开的100个任务（dev70/holdout30）和隐藏的100个任务；

**📈 对比分析**

通过四轮官方提交评测，最终得分57.0/100，排名第二，较起始分31.0提升近26分；与基准模型对比显示大幅提升；

**⚠️ 局限性**

局限包括仅使用单一LLM、对小样本评估的统计意义有限、对实时网站变更敏感、未提供完整基线实现、对非文本交互（如画布）仍存在能力缺口。

---

## 467. NegT2IBench: When Negation Changes the Picture. A Polarity Benchmark for Text-to-Image Models

**arXiv ID:** 2610.03084 | [PDF](https://arxiv.org/pdf/2610.03084v1)

**作者:** Omar Elfatairy `[一作]` (Technical University Of Munich), Zeynep Akata `[通讯]` (Technical University Of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个基于检测器的文本到图像生成模型否定性评测基准，并评估了11种模型在4,800条提示下的否定执行能力。

**💡 创新点**

将否定与肯定提示在同一复杂度下对比，使用专用检测器逐条判定成功与否，避免遗漏对象导致的误判，并提供细粒度失效分类。

**🔧 技术方法**

利用LLM生成可行否定提示，使用RF‑DETR‑XL、SAM 2、Depth‑Anything V2、SigLIP‑2等检测器检测对象、属性与关系；通过阈值规则判定正负语句；对比多种LLM判别器和图文相似度方法。

**📊 数据集**

从COCO物体类别、GenEval颜色词表、OVAD材质词表以及预定义方向、接近、比较和深度关系构造4,800条提示，生成共211,200张图像；并对600张图像进行三人标注验证。

**📈 对比分析**

对11个T2I配置进行四次采样，计算图像准确率；与人类判别以及多种VQA/图文评分模型对比，检测器在保持高Kappa（72.0）且显著降低GPU内存占用；多数模型在否定语句上准确率比肯定低20–40个百分点，最高模型仅31.4%。

**⚠️ 局限性**

评测仅覆盖79类物体、两类属性与四类关系；检测器误差与阈值设置影响结果；对更复杂否定结构、语言多样性或长句子未做深入；模型改进仍需针对否定训练。

---

## 468. ULTRADISCOVERY: Abductive Exploration in an Interconnected, Epistemically Open Universe

**arXiv ID:** 2610.03092 | [PDF](https://arxiv.org/pdf/2610.03092v1)

**作者:** Weihan Li `[一作]` (University of Tokyo), Simon See `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一套新的交互式科学发现基准环境，该环境通过2×2实验设计（开放/披露表示、分布/对齐证据）分离了构造解释性表示与跨域证据组合两大难题，并对多种大型语言模型（LLM）及其厂商运行框架在此环境下的探索行为进行了系统评估。

**💡 创新点**

创新点在于：① 通过“epistemically open”与“structurally interconnected”双重属性设计单一潜在世界，将表示构建与证据整合两种抽象负担独立控制；② 利用2×2因子设计（Open/Disclosed × Distributed/Aligned）清晰量化表示开放性和证据组织对发现过程的影响；③ 通过多维度度量（grounded blocks、表示变更、跨域推断、终极预测）揭示了表示构建是限制可迁移解释的关键瓶颈；④ 引入自评、反驳和进化三种“支架”方法，检验其对探索效率与发现质量的作用。

**🔧 技术方法**

技术手段包括：① 构造一个可交互的五域模拟世界，内置18个知识块和3个跨域超边；② 设计最小化的交互循环（Baseline Loop）并叠加自评、反驳、进化等支架；③ 通过事件日志与人工评判器提取模型书写的推断与证据，计算多种发现指标；④ 采用配对引导抽样与bootstrap区间进行模型与实例间的统计对比；⑤ 在厂商的原生 harness 环境（Claude Fable 5.1、GPT‑6 Astra）中运行模型以检验系统级改进。

**📊 数据集**

数据集：完全自定义的实验世界——五个域（fengrazing fen、trellisrepair trellis、ladderladder of beacons、consignfreight floor、cadastreregistry）以及其中的18个知识块和3个跨域超边。该世界通过脚本生成不同的 2×2 版本，未使用公开的自然语言语料或外部知识库。

**📈 对比分析**

比较方法：对每个模型在同一实例的四个版本（Open×Distributed、Open×Aligned、Disclosed×Distributed、Disclosed×Aligned）进行配对评估，汇总 12 组实例的指标；使用配对bootstrap 计算置信区间；主要指标包括：探索AUC、表示变更数（RC）、跨域推断数（XC）以及终极预测是否准确。结果显示：在 200 次付费行动内，绝大多数模型未能给出精确预测；提供表示（Disclosed）显著提升了干预请求但对发现提升有限；支架方法在提升测试频率方面表现出一定作用，但对整体发现提升不足；厂商 harness 在更大预算下取得轻微优势，但仍未解决表示构建的根本难题。

**⚠️ 局限性**

局限性：① 只使用了一个手工构造的潜在世界，缺乏多样化环境验证其通用性；② 评估对象仅为 LLM 代理，未考察人类或其他人工智能系统的表现；③ “开放性”与“对齐”对照是通过显式授予的信息（披露文档、对齐记录）实现的，未完全模拟自然探索过程；④ 评价指标仅基于模型书写的文本与事件日志，可能遗漏未记录的推断或隐式表征；⑤ 结果在发布后可能因公开机制细节导致可复现性挑战。

---

## 469. Light Entropic Optimal Transport on Riemannian Manifolds

**arXiv ID:** 2610.03085 | [PDF](https://arxiv.org/pdf/2610.03085v1)

**作者:** Xavier Aramayo-Carrasco `[一作]` (Applied AI Institute), Alexander Korotin `[通讯]` (Applied AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 ManifoldLightOT，一种在球面、环面、SO(3) 和 SE(3) 等常见流形上可直接采样、无内部优化的熵正则化最优传输（EOT）学习框架。

**💡 创新点**

创新点在于设计几何专属 Gibbs 核与可解析归一化的潜在函数组合，使得 EOT 伴随分布可以闭式归一化并直接采样；同时该构造可扩展到笛卡尔积流形。

**🔧 技术方法**

使用基于 von Mises‑Fisher、圆形 von Mises、双曲余弦核和高斯核的显式潜在参数化，利用 Monte Carlo 估计的 KL 损失进行无监督训练，并通过解析条件分布实现采样。

**📊 数据集**

在合成数据（高维球面、环面、SO(3)、SE(3) 的已知分布）以及真实数据（地质板块漂移、晶体取向迁移）上进行实验。

**📈 对比分析**

与 RNOT、ERNOT、NeuralOT 等基线进行比较；在所有实验中，ManifoldLightOT 在 MMD、纹理指标等度量上均显著优于对手，且训练与采样更为高效。

**⚠️ 局限性**

主要局限在于需为每种新流形手工推导兼容核‑潜在对；对更复杂或高维流形的推广仍需要额外的解析工作。

---

## 470. Peer Influence across Heterogeneous AI Models

**arXiv ID:** 2610.03095 | [PDF](https://arxiv.org/pdf/2610.03095v1)

**作者:** Frida Nøhr Laustsen `[一作]` (IT University of Copenhagen), Luca Maria Aiello `[通讯]` (IT University of Copenhagen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个基于双轮对话的框架，用以量化不同语言模型（LLM）之间的相互说服影响，并在此框架下评估多模型组合在决策过程中的行为。

**💡 创新点**

创新点在于：①将“说服”转化为可量化的概率位移度量；②系统性比较异构模型对话中的影响方向与强度；③揭示模型置信度与规模并不可靠地预测其易受影响性，甚至小模型可在异构组合中发挥强大说服力。

**🔧 技术方法**

技术手段包括：对七个公开量化（4‑bit）LLM（Llama、Gemma、Qwen、GPT‑OSS 等）进行多次（M=10）推理；构造“互相对立”标签‑解释对；计算正向影响得分 Δ 与反向“回火”得分 Δ⁻；使用宏平均、标准误等统计方法评估影响程度。

**📊 数据集**

使用三种二分类数据集：Sentiment Analysis、CommonsenseQA 2.0 以及 Sarcasm Detection，每个数据集均约 10‑15k 样本，覆盖易到难的文本理解任务。

**📈 对比分析**

对比方法：将各模型对（异构与同构）在三任务上的影响得分与回火得分进行矩阵化展示，并分析模型规模、置信度、任务难度对影响的调节效应。实验显示：大多数模型的影响得分>0.5，异构组合可产生比同构更大或更小的影响；小模型在某些组合中可与大模型媲美甚至超越；任务难度越大，影响幅度越显著。

**⚠️ 局限性**

局限性包括：①仅考察单轮二分类交互，未覆盖多轮或开放式任务；②实验仅涉及七款开源量化模型，结果可能不适用于更大或商业模型；③使用 4‑bit 量化可能改变模型行为；④将所有交互配置等权重，忽略实际配置的频繁度；⑤未评估模型的真实分类性能与实际任务准确率。

---

## 471. DyadMem: A Long-Term Memory Benchmark of How Agents Work with Users

**arXiv ID:** 2610.03020 | [PDF](https://arxiv.org/pdf/2610.03020v1)

**作者:** Yifei Tao `[一作]` (Nanyang Technological University), Liujian Tang `[通讯]` (Stepfun)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本论文提出了DyadMem benchmark，用以评估多轮对话中长时间记忆系统的完整生命周期，包括捕获、更新、检索和答案生成；同时引入了用户条件关系代理记忆（URAM）并对其进行标注；

**💡 创新点**

创新点在于：①提出关系特定代理记忆URAM，区分用户侧与关系侧记忆；②构建双域全流程评估框架，分别对Capture、Update、Recall进行金标注与测评；③通过Gold‑Memory与Full‑Pipeline两种QA设定，能够定位失败点；

**🔧 技术方法**

采用多阶段任务公式与双向语义覆盖评估、BGE‑M3+NLI检索、vLLM部署与LLM对话生成、手工标注与自动评判器；指标包括Precision/Recall/F1、op‑F1、Recall@n、Grounded Accuracy等；

**📊 数据集**

使用DyadMem数据集，包含3,065个事件、50,961个会话、61,210个QA实例，覆盖6类记忆类型（用户属性、目标信息、角色、合作流程、决策规则、共享事件），跨5大领域和25个子领域；

**📈 对比分析**

对16个开源LLM和4个专有模型进行比较，Gold‑Memory QA得分78.7–96.3%，Full‑Pipeline QA仅47.7–57.4%；Capture recall仅15%，Update op‑F1低于20%；URAM在所有模型上平均提升约+3个百分点；模型排名在不同评判器下高度稳定；

**⚠️ 局限性**

局限性包括：捕获与检索阶段表现不佳，Recall与Capture低；更新操作中误删率高，导致记忆不完整；用户侧记忆更难捕获且QA效果差；数据规模虽大但仍可进一步扩展；评测主要聚焦问答，缺乏多模态或行动层面的考察。

---

## 472. RIFAR: Reliability and Forgetting-Aware Replay for Continual Robot Learning

**arXiv ID:** 2610.03079 | [PDF](https://arxiv.org/pdf/2610.03079v1)

**作者:** Zirong Song `[一作]` (MBZUAI), Xiuying Chen `[通讯]` (MBZUAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于世界-动作模型的持续学习框架RIFAR，利用可靠性筛选与漂移感知的经验重放，显著提升机器人在新任务学习过程中对旧任务的记忆与泛化。

**💡 创新点**

创新点在于：①使用冻结的逆动力学模型对生成轨迹的动作-视觉一致性进行可靠性筛选；②通过对旧任务轨迹的动作漂移进行测量，动态优先重放对旧任务影响最大的轨迹；③仅保留每个旧任务的16帧初始上下文，极大压缩存储需求。

**🔧 技术方法**

技术手段包括：世界-动作模型（WAM）进行自回归轨迹生成；冻结的逆动力学模型（IDM）做动作一致性评估；双阶段训练（先用高质量轨迹微调，再用漂移高轨迹重放）和漂移度量公式；以及在真实Frank a机器人上验证的物理仿真。

**📊 数据集**

实验数据集：LIBERO benchmark（Goal、Object、Spatial三套任务）以及真实Frank a机械臂的三类操作任务。

**📈 对比分析**

对比方法包括：顺序微调、EWC、PackNet、REGEN、经验重放ER(n)等。RIFAR在LIBERO-Goal上达到90.97 AUC，仅保留每任务320步历史，优于REGEN（67.8 AUC）且与ER(50)（91.8 AUC）相当但存储量仅为4.9%。在Object、Spatial套件中也显著提升AUC并降低忘记率。

**⚠️ 局限性**

局限性包括：①逆动力学模型的可靠性筛选不能完全消除错误轨迹；②冻结的IDM对分布漂移敏感，可能误判新任务行为；③WAM的价值终止策略可能导致提前停止，影响轨迹完整性；④长时序生成易累积误差，需更强的错误纠正或层级生成机制。

---

## 473. Where to Look Is Not How to Fix: Pre-Denoising Diagnostics and Modality-Dependent Control in Diffusion Composition

**arXiv ID:** 2610.03068 | [PDF](https://arxiv.org/pdf/2610.03068v1)

**作者:** Fangzheng Wu `[一作]` (Tulane University), Brian Summa `[通讯]` (Tulane University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文在文本到图像扩散模型中，通过控制anchor–stress协议研究组合失效的诊断与干预，提出了文本仅 CSI 以及 UNet 块级交叉注意力诊断，并在 SD1.5、SDXL 和 SD3 上进行多块级实验。

**💡 创新点**

创新点在于揭示诊断位置与干预效果不一致的“诊断‑控制解耦”，并给出块级介入与模态分解的经验映射，帮助选择更有效的修复策略。

**🔧 技术方法**

技术方法包括：文本编码器残差检测（CSI）、UNet 块级交叉注意力诊断（DA）、低秩适配器修复、跨注意力增减/抑制干预，以及颜色命中率（CHR）评估。

**📊 数据集**

实验数据集来自 T2I-CompBench、CLEVR 与 ARO，构成 216 对 anchor–stress 提示，并在 SD1.5、SDXL、SD3 三种模型上测试。

**📈 对比分析**

比较方法以 ROC‑AUC、DA 均值、CHR 变化为指标；CSI 在三模型上 ROC‑AUC=1.0；跨注意力增减在解码器块产生正向 CHR 提升，深层编码器仅表现出强诊断能力但对输出几乎无改善。

**⚠️ 局限性**

局限性包括：仅验证属性‑对象组合和颜色匹配，未覆盖更广泛的语义评估；干预方案受限于固定的超参数和目标函数；未探索不同目标函数或 DiT 架构的泛化效果。

---

## 474. SecJev: Bringing Security Expertise to System One Decision Models

**arXiv ID:** 2610.03073 | [PDF](https://arxiv.org/pdf/2610.03073v1)

**作者:** Zheng Chen `[一作]` (University of Electronic Science and Technology of China), Lei Chen `[通讯]` (National Key Laboratory of Security Communication)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 SecJev，一系列专为安全任务设计的 System One 决策模型，能够处理工具输出、流量、身份验证和车辆消息等多种安全场景的 Boolean、choice 与 ordered 决策。

**💡 创新点**

创新点在于：① 将 Jev 的单通候选评分机制与安全专用 LoRA 适配器和决策头结合，形成统一的 typed 接口；② 通过 SecJev‑Corpus 统一 14 个安全任务的源标注与显式政策评估；③ 通过场景加权训练和温度标定，在不同规模模型上实现比通用 Kev 更高的准确率（如 0.8B SecJev 超过 Kev‑9B 20%）。

**🔧 技术方法**

采用的技术包括：Jev 的单通候选评分（pointer head）、LoRA 低秩适配、温度标定、场景加权交叉熵训练、三轮安全专用训练预算、FP32/BF16 混合精度、与 Kev 相同的基础架构。

**📊 数据集**

使用 SecJev‑Corpus（170,185 个问题，115,324 个场景，覆盖 8 个来源），包含 CICIoT2023、ToN‑IoT、Twins/Streamlet、VeReMi、AgentDojo、InjecAgent、ByzFL、LANL 等安全数据集。

**📈 对比分析**

通过与通用决策初始化的 Kev、生成式 SFT、直接 Qwen 指针等接口进行对比；在 FP32 下 0.8B SecJev 的宏观准确率为 90.34%，与 0.8B 生成式 90.01% 相近；4B SecJev 在宏观准确率上达到 94.51%，略高于 4B 生成式 94.34%；相比生成式，SecJev 在内存占用、短/长输入延迟和吞吐量方面更优，且在 0.8B 规模下可将 peak 内存从 3.52 GiB 降至 2.56 GiB。

**⚠️ 局限性**

限制包括：仅能对已提供的观测做有限决策；不同来源的标注不一定完整，尤其是源标注任务；部分任务（如车辆消息）仍表现不佳；训练样本有限导致对未知安全场景的泛化能力不确定；模型置信度在安全任务中被优化但校准仍有偏差。

---

## 475. Learn Feasibility Once, Optimize All Objectives: Derivative-Free Diffusion Models for Chance-Constrained Programming

**arXiv ID:** 2610.03071 | [PDF](https://arxiv.org/pdf/2610.03071v1)

**作者:** Ziwen Liu `[一作]` (Beijing University of Posts and Telecommunications), Weichen Zhao `[通讯]` (Nankai University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种去梯度的扩散框架 D³Opt，先学习仅基于约束的数据的风险条件扩散先验，再通过 Feynman–Kac 校正实现对任意后置目标的优化。

**💡 创新点**

创新点在于将约束建模与目标优化完全解耦：先一次性学习约束可行结构并冻结，随后只用目标函数值通过粒子化的 FKC 进行无梯度优化；同时给出了可保证可行性和误差分解的理论分析。

**🔧 技术方法**

核心技术包括风险条件噪声预测扩散模型、逆扩散采样、基于逆温度的 Annealed Feynman–Kac 粒子重加权与重采样，以及与传统 CCP 解决方案的对比实验。

**📊 数据集**

使用的数据集主要是三类合成与实际问题：线性高斯 CCP（8 维）、目标迁移实验（ASQ、MW4 目标）、以及有风速不确定的 10 单元经济调度（VPE）问题。

**📈 对比分析**

与基准方法（SOC‑CVX、SAA、DiffOPT、GGDOpt 等）比较，D³Opt 在所有实验中均取得最小均值/中位数/标准差的目标/成本，并在目标迁移中误差仅 4‑6%，优于 DiffOPT 的 26‑34%；在经济调度中平均成本比 DiffOPT 低 0.81%，且满足 0.90 的可行性阈值。

**⚠️ 局限性**

局限性包括：需要足够的约束可行样本以覆盖全局最优子集；FK 校正仅在先验支持内移动，若先验覆盖不足则无法达到全局最优；在高维、复杂可行性检查成本高的场景中，采样效率和粒子数需进一步提升。

---

## 476. Mobility Enhancement of Patients Body Monitoring based on WBAN with Multipath Routing

**arXiv ID:** 2610.03042 | [PDF](https://arxiv.org/pdf/2610.03042v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 477. HARPO: Hallucination-Aware Reinforcement Learning for Faithful and Creative Language Generation

**arXiv ID:** 2610.03063 | [PDF](https://arxiv.org/pdf/2610.03063v1)

**作者:** Tiezheng Yu `[一作]` (Huawei Technologies), Lifeng Shang `[通讯]` (Huawei Technologies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 HARPO 框架，联合优化大语言模型的真实性（去幻觉）与创意写作质量，利用训练好的 HA‑GRM 生成奖励模型并通过 Selective Activation Mechanism（SAM）和创意→真伪的动态数据课程实现。

**💡 创新点**

创新点包括：① 构建了同时评估幻觉与写作偏好的 HA‑GRM；② 通过 SAM 在判定无幻觉时才激活写作奖励，解决奖励冲突；③ 采用从创意到真伪的渐进式数据课程，降低灾难性遗忘并平衡两项目标。

**🔧 技术方法**

技术上使用了基于 Group Relative Policy Optimization（GRPO）的强化学习、可验证奖励（RLVR）框架、HA‑GRM 的 span‑level 与 pairwise preference 训练、SAM 条件奖励聚合、以及余弦退火的数据课程调度。

**📊 数据集**

实验数据集包括 RAGTruth（summarization & QA）、HaluEval（QA）、Arena‑Human‑Preference‑140k（创意写作）、Arena‑Hard‑v2.0（通用与创意子集）、HHEM（summarization）和 MultiHopRAG（QA）。

**📈 对比分析**

通过与 SFT、Hallucination‑Only、Writing‑Only、Linear Mixture 等基线对比，HARPO 在 Qwen3‑4B 上将 HHEM 幻觉率从 3.29% 降至 1.02%，创意写作分数从 16.95% 提升至 27.54%，并在多模型规模上保持更低幻觉率和更高写作质量，优于线性混合策略。

**⚠️ 局限性**

局限性包括：① 需要昂贵的 span‑level 注释；② HA‑GRM 推理开销较大；③ 评估与奖励共享同一模型，误判可能影响优化；④ 仅验证到 8B 参数模型，未测试更大模型或专用推理任务。

---

## 478. Divide and conquer: Scalable performance and energy in MCM GPUs

**arXiv ID:** 2610.03061 | [PDF](https://arxiv.org/pdf/2610.03061v1)

**作者:** Mario Ibáñez Bolado `[一作]`, Julio Ramón Beivide `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

未提供内容

**💡 创新点**

未提供内容

**🔧 技术方法**

未提供内容

**📊 数据集**

未提供内容

**📈 对比分析**

未提供内容

**⚠️ 局限性**

未提供内容

---

## 479. PocketSplat: Mobile Gaussian Reconstruction via World-Space Latent Allocatio

**arXiv ID:** 2610.03192 | [PDF](https://arxiv.org/pdf/2610.03192v1)

**作者:** Wenzhi Guo `[一作]` (Hong Kong Polytechnic University), Bing Wang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 PocketSplat，面向移动端的前向 3D 高斯重建框架，能够根据预设的高斯数量预算直接生成可用于渲染、存储和传输的资产。

**💡 创新点**

创新点在于：1) 在世界空间上对稠密候选进行单元化分配，实现精确整数预算分配；2) 在分配后对保留候选做跨视角隐藏融合与空间责任解码（SRD），自适应地调整高斯支持；3) 仅对选定候选解码完整属性，从而显著减少内存占用与计算量。

**🔧 技术方法**

使用了冻结的多视角 Transformer + 变换器骨干、深度头、可训练的 Gaussian 头、细节熵估计、视角加权、容量分配器、隐藏融合网络和 SRD 头，全部在 CPU+ONNX 运行时实现。

**📊 数据集**

在 DL3DV 基准集、Tanks & Temples、MegaDepth 以及真实手机采集的 Mip‑NeRF 360 上进行评测。

**📈 对比分析**

与 pixelSplat、MVSplat、DepthSplat、TranSplat、F⁴Splat 等基线相比，在相同高斯预算下 PocketSplat 的 PSNR 提升 2–4 dB、SSIM 提升 0.01–0.03、LPIPS 降低 0.02–0.04；在 iPhone 上实现全流程部署，构建/推理时间分别降至 10 s 左右，显存峰值约 3 GB，完全避免 OOM。

**⚠️ 局限性**

局限性包括：1) 依赖冻结的几何先验，可能在极端视角或遮挡场景下效果下降；2) 目前仅支持 4 张视角的输入；3) 由于移动硬件限制，最高预算仍低于桌面级方法；4) 对高噪声或低光图像的鲁棒性尚未充分验证。

---

## 480. A Shortest Augmenting Path Algorithm for Linear Matroid Parity

**arXiv ID:** 2610.03030 | [PDF](https://arxiv.org/pdf/2610.03030v1)

**作者:** Kou Hamada `[一作]` (University of Tokyo), Satoru Iwata `[通讯]` (University of Tokyo)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4`

**🎯 论文内容**

本文提出了首个基于最短增广路径的线性矩阵并行性问题（linear matroid parity）算法，实现了确定性求解。

**💡 创新点**

创新点在于将Gabow–Stallmann的增广路径框架与Micali–Vazirani的同步blossom技术相结合，并给出线性代数上对最短增广路径长度的上界，推广了Cunningham对线性矩阵交叉的结论。

**🔧 技术方法**

核心技术包括：① 利用Pfaffian和矩阵表示的线性代数论证最短增广路径长度；② 对搜索路径的“先验长度下界”进行分析；③ 在算法实现中使用“dummy line”和“synchronized blossom”实现搜索路径同步；④ 采用快速矩阵乘法进一步降低复杂度。

**📊 数据集**

论文为理论算法，不使用实验数据集，而是基于矩阵表示的线性多重集合（线性矩阵）进行分析。

**📈 对比分析**

与以往最优的确定性算法(n r³ 或 n r^ω)相比，本文算法复杂度降至(n r² log r)，若使用快速矩阵乘法则可进一步到(n r²)。与当前最佳随机算法(n r²)相比，时间复杂度相当但保持确定性；在大规模实例上，可在保持可实现的时间内求解更大规模的线性多重集合。

**⚠️ 局限性**

局限性包括：① 仅适用于具备矩阵表示的线性多重集合，不能直接推广到一般多重集合；② 算法实现复杂，涉及大量线性代数操作和blossom维护；③ 虽然理论上时间被改进，但常数因子仍然较大，实际性能需要进一步评估。

---

## 481. MOF-VERIFY: A Failure-Aware Agentic Harness for MOF Hypothesis Verification

**arXiv ID:** 2610.03056 | [PDF](https://arxiv.org/pdf/2610.03056v1)

**作者:** Donghyun Lee `[一作]` (Ewha Womans University), Soo Kyung Kim `[通讯]` (Ewha Womans University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了针对金属有机框架（MOF）假设验证的诊断基准，并基于该基准设计了 MOF-Verify 失败感知的 agentic harness，显著提升了验证准确性。

**💡 创新点**

创新点在于：①将验证过程拆分为结构识别、合成条件检索、证据充分性判断和计算验证四大瓶颈；②针对每个瓶颈设计专用模块（IR、SE、LE、ES、CE）并通过规则与检索实现高可信度；③不对 LLM 进行微调，直接利用冻结模型与外部工具协同完成任务。

**🔧 技术方法**

采用检索增强生成（RAG）、规则推理与结构化检索（IR、SE）、文献结构化提取（LE）、证据充分性评估（ES）、MLIP 计算（CE）以及确定性决策路由（VR）等技术。

**📊 数据集**

使用自制诊断基准 T-MOF-1~4（共1270条样本，包含结构、合成、证据与计算任务），结合公开的 CSD 数据库、DOI 文献集合及 SevenNet MLIP 计算规范。

**📈 对比分析**

与闭本、检索、oracle 三种设置及多种后端 LLM（GPT‑4o、Claude、Gemini、Qwen 等）比较，MOF-Verify 在 T-MOF-1~3 的平均 Macro‑F1 达到 62.82，较闭本提升 31.8、较检索提升 8.16，接近 oracle 上限；在 T-MOF-4 上从 5.7% 提升至 56.97%。

**⚠️ 局限性**

局限性包括：合成条件提取仍受限于文献结构化质量；规则与手工标注的模块难以覆盖全部文献多样性；对证据充分性评估的自动化程度有限，仍需人工专家干预。

---

## 482. A Mechanistic Model of the Human Menstrual Cycle

**arXiv ID:** 2610.03053 | [PDF](https://arxiv.org/pdf/2610.03053v1)

**作者:** Lena Reitinger `[一作]` (Johannes Kepler University), Stefan Angerbauer `[通讯]` (Kepler Universitaetsklinikum)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个10维ODE模型，描述FSH、LH、E2和P4在28天月经周期内的动态变化。

**💡 创新点**

通过在保留关键生理机制的前提下引入更精细的细胞周期、血管化和反馈环节，使模型维度大幅降低至10维且参数仅40个，保持了高生理准确性。

**🔧 技术方法**

采用Hill函数描述激素反馈，结合细胞周期动力学、血管化生长方程以及稳态假设，使用Simulink（ode15s求解器）进行数值仿真。

**📊 数据集**

使用公开的实验测量数据（FSH、LH、E2、P4在28天周期内的浓度曲线），但未给出具体数据集名称或编号。

**📈 对比分析**

通过与文献中报道的激素曲线对比，模型能重现早期峰值、黄体期峰值及周期性收敛特性；仿真计算量低，适合快速迭代。

**⚠️ 局限性**

主要局限在于中期激素峰值时间与实验略有偏差，未考虑延迟（DDE）或个体差异；模型对峰值时间的准确性有待改进。

---

## 483. LiBRA: Detection-Aware Image Watermark Removal via Bidirectional Latent Optimization

**arXiv ID:** 2610.03166 | [PDF](https://arxiv.org/pdf/2610.03166v1)

**作者:** Saibo Ye `[一作]` (City University of Macau), Tianqing Zhu `[通讯]` (City University of Macau)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在潜在空间进行双向均衡优化的水印移除攻击LiBRA

**💡 创新点**

通过对平均解码置信度做对称约束，避免过度反转导致的可检测水印，同时保持图像质量

**🔧 技术方法**

潜在空间的有界优化、双向置信度损失、频率引导空间掩模、自动编码器解码、梯度反馈

**📊 数据集**

Stable Signature、YU1、HiDDeN、MBRS四套公开水印系统，使用MS‑COCO、CelebA等图像集共计20,000张样本

**📈 对比分析**

与WEvade-W-II、UnMarker、VAE/扩散重建等攻击和常规后处理对比，LiBRA在两侧显著降低检测率（≤0.05）且PSNR/SSIM/LPIPS均优于基线，计算速度提升约9.5×

**⚠️ 局限性**

仅适用于白盒且可访问水印关键和解码梯度的场景，需预先估计关键且对不同显著性阈值和检测统计的鲁棒性有限

---

## 484. Not Until the Evidence Says So: Teaching LLM Investigators When to Close a Case

**arXiv ID:** 2610.03190 | [PDF](https://arxiv.org/pdf/2610.03190v1)

**作者:** Tingzhu Bi `[一作]` (Peking University), Meng Ma `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了调查关闭（investigative closure）的评估框架，并训练大型语言模型在多轮证据收集后判断是否足以闭案，既可闭案也可保持开放并指出缺失证据。

**💡 创新点**

创新点在于将闭案决策拆分为两个独立任务——原因判定与是否足够闭案——并引入三种评价方式（闭案准确率、证据依赖性、结论与缺口质量）来衡量模型是否真正依据证据做决策。

**🔧 技术方法**

技术主要包括：使用Qwen3.5-9B模型进行监督微调（SFT）与基于RLVR的强化学习（只奖励闭案决定），并结合ReAct提示、记账与工具调用等交互格式。

**📊 数据集**

使用731个经过审核的调查案例，来源涵盖航空、铁路、海事、化工安全、车辆缺陷与服务器故障等领域，并提供141个跨域OOD测试集与对照版本。

**📈 对比分析**

与未训练的基线、Frontier模型以及多种提示/RL奖励组合对比，经过SFT后闭案准确率从49.9提升至83.3（与教师相当），误判率显著下降；RL只奖励闭案可进一步提升准确率至84%但略削弱证据依赖。

**⚠️ 局限性**

局限包括：源标签高度相关导致准确率易被源偏见影响；RL奖励对结论质量的直接评价会导致模型过度不闭案；未训练模型对缺失证据的表述往往笼统；在服务器（Host）数据上的闭案表现未得到充分验证。

---

## 485. Sample complexity of variance-reduced policy gradient: weaker assumptions and lower bounds

**arXiv ID:** 2610.03165 | [PDF](https://arxiv.org/pdf/2610.03165v1)

**作者:** Gabor Paczolay `[一作]` (Politecnico di Milano), Marcello Restelli `[通讯]` (Politecnico di Milano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出一种基于防御性重要采样的方差减小策略梯度算法（DEF‑PG），在不需要传统重要权重方差上限假设的情况下实现了求解ϵ-FOSP的样本复杂度为O(ϵ⁻³)，并给出了对应的下界证明。

**💡 创新点**

创新点包括：①首次在策略梯度中引入防御性重要采样，天然保证重要权重方差有界；②在oracle层面将算法视为Coupled‑PAGE，实现了统一的方差减小框架；③在黑盒策略优化模型下给出匹配的Ω(ϵ⁻⁴)与Ω(ϵ⁻³)下界，证明了O(ϵ⁻³)速率的最优性。

**🔧 技术方法**

技术手段：防御性重要采样（α‑defensive mixture）、随机方差减小技术（如PAGE、SARAH/SPIDER等）、GPOMDP估计器、oracle层级分析、黑盒反馈模型和高阶概率不等式。

**📊 数据集**

本文不使用公开数据集，而是通过理论构造的随机策略与奖励机制（高斯策略、参数化奖励）进行实验验证，重点展示理论上对样本复杂度的影响。

**📈 对比分析**

与传统REINFORCE（O(ϵ⁻⁴））以及其他O(ϵ⁻³）方差减小方法（如SRVR‑PG、PAGE‑PG）相比，DEF‑PG在不依赖重要权重方差上限且不需要梯度截断的前提下实现相同的样本复杂度，并在理论上证明其优于REINFORCE。

**⚠️ 局限性**

局限性：下界仅适用于黑盒策略优化模型，未在标准MDP观测轨迹协议下证明；需要假设奖励有界且策略分数满足均方连续性；不提供全局最优性或收敛到全局最优策略的保证；对连续动作空间的实现仍需进一步验证。

---

## 486. Page-EntroKV: Hardware-Aligned, Entropy-Weighted KV-Cache Eviction under Grouped-Query Attention

**arXiv ID:** 2610.03135 | [PDF](https://arxiv.org/pdf/2610.03135v1)

**作者:** Inbasekaran S `[一作]` `[通讯]`, Inbasekaran S

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向Grouped‑Query Attention（GQA）结构的KV‑cache淘汰框架 Page‑EntroKV，直接在物理KV组层面进行熵加权池化与页面级块最大化选择，确保严格的预算保持和有限上下文的“针”保留；

**💡 创新点**

创新点在于：①基于sink‑isolated Rényi‑2注意力熵实现头级权重分配，消除头独立淘汰导致的缓存膨胀；②在物理组级别聚合并投射到分页表，避免内部碎片；③推导组内不一致度与联合开销比的精确关系，并给出两侧界；④证明熵加权池化在有限上下文下可实现针保留，平均池化则失效；

**🔧 技术方法**

采用Sink‑Isolated Rényi‑2注意力熵、Boltzmann加权池化、分页最大化选择、Radix Top‑K、两核融合实现；

**📊 数据集**

在Pilot验证中使用Qwen2.5‑1.5B‑Instruct模型（28层，12头，r=6），对长上下文文本进行评估，测量头间不一致度、联合开销比、针保留率等；

**📈 对比分析**

与传统头独立淘汰方法（如H2O、SnapKV）比较，发现熵加权池化在相同预算下将联合开销压回1.0，平均池化导致针保留率为0%；Pilot实验显示在20%预算下针保留率100%，而平均池化为0%，且在多任务（LongBench子集）上保持一定正确率；

**⚠️ 局限性**

局限性包括：①在所有头熵相近时退化为均值池化，需最大化切换回退方案；②严格所有层针保留在Pilot阶段未通过，需更大模型或更细粒度预算；③仅验证单模型，未覆盖不同GQA比率或更大上下文；④仍缺乏完整Serving性能评估与量化。

---

## 487. Learning While Inferring: Local and Parallel Learning for Edge SNNs across Sensing Modalities

**arXiv ID:** 2610.03149 | [PDF](https://arxiv.org/pdf/2610.03149v1)

**作者:** Yanxun Zhang `[一作]` (Fudan University), Xiaoqing Zheng `[通讯]` (Fudan University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种边缘设备可持续学习的SNN训练方法——双向基于突触的蒸馏（BSD），实现了在持续感知过程中边缘设备可同时推理与更新模型参数。

**💡 创新点**

创新点在于将前向推理分支和反向训练分支解耦，利用局部对比目标（ReCo）实现层级对齐，从而消除传统BP的全局误差链，允许推理与学习并行执行；同时大幅降低训练能耗和时延。

**🔧 技术方法**

主要技术包括：双向脉冲神经网络架构、局部对比损失（ReCo）、事件驱动的反向推理、局部梯度更新、时序BSD（对RNN/LSTM的时间反向处理）以及基于原型的增量学习适配。

**📊 数据集**

在SOUL基准下评测，涵盖25个任务，跨五类传感模态：视觉（MNIST、CIFAR-10/100等）、运动（UCI-HAR、HHARL等）、声学（ESC-50、UrbanSound8K等）、无线（UT-HAR、Widar3等）以及神经形态（CIFAR10-DVS、DVS-Gesture）。

**📈 对比分析**

与匹配的BP基准比较，BSD平均仅低于3.8个百分点（81.12% vs 84.89%），且在多数任务中保持与BP相近的准确率；在并行执行下，训练周期可缩至BP的0.72×，能耗降至0.36×。

**⚠️ 局限性**

局限性包括：对批量大小敏感，特别是大输入维度时需要较小batch导致对比损失效果受限；在噪声或数据稀缺任务（如WidAr3、BullyDetect）仍存在一定性能欠缺；且目前实验为GPU仿真，实际嵌入式硬件的能耗和时延尚未实测。

---

## 488. The Reeb Transform

**arXiv ID:** 2610.03126 | [PDF](https://arxiv.org/pdf/2610.03126v1)

**作者:** Erin Chambers `[一作]` (University of Notre Dame), Katharine Turner `[通讯]` (Australian National University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了 Reeb 变换（Reeb Transform），定义为在所有方向上得到的 Reeb 图族，并系统地研究了其可逆性。作者证明了在二维可构造集、低维分层空间以及三维闭表面上 Reeb 变换是单射；同时给出了高维（>3）时失去单射性的反例，说明该描述符在更高维度下的局限性。

**💡 创新点**

创新点在于：①首次将 Reeb 图推广为一类全局变换；②给出了 Reeb 变换的 Lipschitz 稳定性和功能失真距离；③在三维闭表面上通过内部/外部域的 Reeb 变换对原表面进行唯一重建；④在 o‑minimal 可定义框架下完整证明了上述单射性与非单射性结果，拓展了先前仅针对 ECT/PHT 的理论。

**🔧 技术方法**

主要技术包括：o‑minimal 结构与可定义集合理论、圆柱细分（cylindrical cell decomposition）、可定义映射与变形重心、Euler 直角与同调（Alexander 对偶性）、函数失真距离、以及在表面切片的拓扑分类。通过这些工具实现了对 Reeb 图结构的精确分析与证明。

**📊 数据集**

本文没有使用具体数据集，所有结果均为理论证明。作者主要关注可定义、可构造集合的抽象性质。

**📈 对比分析**

方法比较：作者将 Reeb 变换与已知的 ECT/PHT 进行对比，指出后者是可逆的，而 Reeb 变换在一般情况下不可逆。性能方面没有实验评估；理论上在二维、低维分层空间以及三维闭表面上可完全区分不同形状，说明其在这些“温和”场景下具有较强辨别力；但在高维或不满足可定义条件时失效。

**⚠️ 局限性**

局限性：Reeb 变换仅记录水平集的连通分量信息，无法捕捉内部同调结构；因此对一般集合（尤其是高维或复杂拓扑）不可逆；需要在所有方向上取样才能恢复形状，导致计算成本高；对“崩塌方向”有度量零集合的技术限制；且在三维以上无法推广至所有表面，仍需进一步研究。

---

## 489. ParaGeo: Decomposing Paralinguistic Variation into a Shared Latent Geometry

**arXiv ID:** 2610.03125 | [PDF](https://arxiv.org/pdf/2610.03125v1)

**作者:** Yuhan Liu `[一作]`, Yunbo Long `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现ParaGeo框架，对冻结的语音语言模型（如GLM-4-Voice）进行匹配内容分解，提取低维共享坐标用于控制语音的抒情特征。

**💡 创新点**

首次引入内容中心化、K/V池化、低秩SVD与共享投影，将跨语境的抒情变化映射到统一坐标系，既可用于精细控制，又可在不更新模型的情况下评估风格一致性。

**🔧 技术方法**

采用语音合成、冻结模型推理、音频token重播、K/V聚合、句子中心化、截断SVD、投影正交化，配合静态/加性/动态干预和cosine最近中心分类、条件置换检验等技术。

**📊 数据集**

使用12个SpeechParaling-Bench benchmark family共80个请求控制，8句固定句子以及10场景债务谈判数据集进行合成与校准。

**📈 对比分析**

通过80-way准确率与相同标签跨内容cosine相似度评估，得到9.49%准确率（vs 1.25%基线）和0.285同标签相似度（vs 0.017），显著优于置换基线；在动态与静态干预中，中间层偏好率最高达68.75%，并与全空间添加对比展示控制与文本保真度的权衡。

**⚠️ 局限性**

仅在单一GLM-4-Voice模型验证，缺乏跨模型泛化；投影与干预能量未匹配；缺少人类听觉验证；标签层面控制精度有限，需进一步训练优化基准。

---

## 490. Building Interpretable Feature Representations for Resume-Vacancy Matching by Distilling Production LLM Signals

**arXiv ID:** 2610.03112 | [PDF](https://arxiv.org/pdf/2610.03112v1)

**作者:** Ilya Chekin `[一作]` (BroutonLab), Mikhail Yurushkin `[通讯]` (Curately)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发一个基于LLM标签与特征双编码器的候选人‑职位匹配系统，能够实时输出八个可解释的匹配维度；

**💡 创新点**

通过招聘者反馈持续迭代LLM提示，并将其蒸馏为低延迟CPU双编码器，同时引入查询限定适用性头以处理仅在职位要求时才相关的特征；

**🔧 技术方法**

使用生产反馈驱动的LLM提示、跨编码器教师+软标签、知识蒸馏、秩序一致性与余弦/点积对齐的特征表示、LoRA微调、专用MLP头以及查询限定适用性头；

**📊 数据集**

基于真实招聘数据，17,921个职位与180,030份简历共168,772个职位‑简历对，所有对均经过LLM标签并用于训练；

**📈 对比分析**

与LLM标签、跨编码器教师及基线进行对比；在留存测试集上宏F1达0.849，实际招聘反馈一致率95.79%；CPU推理速度约5500条/秒，端到端延迟≤500 ms；

**⚠️ 局限性**

反馈稀疏且非盲目，缺乏独立评估；数据不可公开，且仅英文，未验证多语言或新行业；未测量招聘结果影响，也未进行系统性偏见审计。

---

## 491. Tracking State Footprints: How Agents Can Transact

**arXiv ID:** 2610.03140 | [PDF](https://arxiv.org/pdf/2610.03140v1)

**作者:** Oto Mraz `[一作]` (Ververica GmbH), Asterios Katsifodimos `[通讯]` (Delft University of Technology)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文探讨将多智能体系统（MAS）中的状态访问和并发管理视为数据库事务问题，提出“状态足迹”概念并构建跨Agent、编排器与外部系统三层状态管理框架；

**💡 创新点**

创新点在于统一建模状态足迹、引入可配置的ACID级别（原子性、隔离性、一致性、持久性），并允许Agent通过语义冲突解决而非简单回滚；

**🔧 技术方法**

使用LLM代理框架（如LangGraph、AutoGen）、工具调用机制、事务日志与版本控制（Git、文件系统快照）、基于图的并发控制以及TCC/补偿模式与事务标识符等技术；

**📊 数据集**

在SWE-Bench的四个编码任务（bug修复、数学错误、媒体对象合并、Django API重构）上进行实验，并采用Claude Haiku与Claude Opus模型；

**📈 对比分析**

通过比较顺序、无协调、任务有序、冲突有序四种执行拓扑，结果显示冲突有序拓扑在保持正确性的同时提升约1.23倍的执行速度、显著降低成本与错误率，验证了状态依赖追踪的有效性；

**⚠️ 局限性**

局限在于缺乏统一的跨系统版本协同与事务标识映射、难以精确回滚长时间非确定性任务、对外部系统事务支持不足，以及在动态规划与大规模Agent循环中的性能与可扩展性待进一步验证。

---

## 492. Ontological Instability and Statistical Amplification: The Paradox of "Humanizing" LLM-Generated Text

**arXiv ID:** 2610.03110 | [PDF](https://arxiv.org/pdf/2610.03110v1)

**作者:** Claudiu Creanga `[一作]` (University of Bucharest), Liviu Dinu `[通讯]` (University of Bucharest)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了RoBERTa基准AI文本检测器在语义、结构和分词层面的鲁棒性，并通过对Mistral-7B进行“人性化”重写以及字符层的同形异义词攻击来探究检测器的决策依据。

**💡 创新点**

创新点在于揭示了“统计放大”现象，即通过提升词汇多样性反而使文本更易被检测，同时对比结构化事件检测器显示鲁棒性取决于特征类型；并首次验证了同形异义字符攻击对事件检测器的影响。

**🔧 技术方法**

采用RoBERTa监督分类器、基于事件的Latent Space检测器、Mistral-7B生成重写、字符级同形异义词替换、NFKC归一化加自定义映射、以及Gzip压缩比、词汇多样性等指标。

**📊 数据集**

使用M4公开数据集（10,000条英文本，其中5,000验证、5,000测试）以及额外的300条Mistral生成的“人物压力测试”样本。

**📈 对比分析**

RoBERTa在M4上实现近乎完美的AUC 1.0，但对正式人类学术文本的误报率高达76%；对同形异义词攻击的成功率可达41%，但经过字符映射后降至2%；Latent Space检测器仅在WikiHow域上取得AUC 0.577，其余域甚至低于随机。

**⚠️ 局限性**

局限性包括仅评估单一RoBERTa模型、仅使用英文数据、事件提取依赖spaCy动词可能不稳健，以及同形异义词攻击易被Unicode分析检测，结果对其他模型和语言的可迁移性未知。

---

## 493. The Fragility of Trigger-Tag Mechanisms for Misuse Detection in Open-Weight LLMs

**arXiv ID:** 2610.03124 | [PDF](https://arxiv.org/pdf/2610.03124v1)

**作者:** Toluwani Aremu `[一作]` (MBZUAI), Dinil Mon Divakaran `[通讯]` (Agency for Science, Technology and Research Institute of Advanced Intelligence and Computing)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对开放权重 LLM 的滥用检测中使用的触发标签（trigger‑tag）机制进行形式化，并评估其在被攻击（输出重写、标签覆盖、模型微调/剪枝/SVD）下的鲁棒性。

**💡 创新点**

创新点包括：①将触发标签区分为 token‑level 与 weight‑level 两类并给出统一定义；②提出 UnTag 统一攻击评估框架，系统化攻击面与评估方法；③首次将文本水印、后门学习与多种攻击技术结合，量化其在实际滥用场景中的弱点。

**🔧 技术方法**

技术手段主要包括：文本水印（KGW、Unigram、EXP、SynthID）、后门学习（Paladin‑Base/Pro、LLM‑Mark）、输出侧攻击（表面清理、语义重写、标签覆盖）、模型侧攻击（细调、剪枝、对比子空间 SVD 消除）。

**📊 数据集**

使用的主要数据集为 Paladin 提供的 phishing 与正常邮件样本，用于生成标签化文本；重写攻击采用 Qwen2.5‑3B‑Instruct 生成重写版本；模型评估在 LLaMA‑7B/8B、Qwen‑7B/8B 上进行。

**📈 对比分析**

评估方法为将各触发标签在无攻击时的 TPR/FPR 与在不同攻击下的 TPR 进行对比；实验显示绝大多数机制在攻击后检测率降至 0–4%，即使在最稳健配置下也无法维持高检测率，说明其鲁棒性不足。

**⚠️ 局限性**

局限性包括：仅针对 phishing 这一滥用场景；实验模型与实现有限；假设完美的目标条件监测，未考虑提示侧规避或检测器优化；未进行大规模人类评估，无法完全验证攻击后的语义与实用性。

---

## 494. Emergent Structure in the Marginal Attention Space of Language Models

**arXiv ID:** 2610.03109 | [PDF](https://arxiv.org/pdf/2610.03109v1)

**作者:** Valentino Maiorca `[一作]` (Institute of Science and Technology Austria), Francesco Locatello `[通讯]` (Institute of Science and Technology Austria)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了语言模型中注意力的边际分布，提出了按头和按令牌归约的两种结构，并用它们分别捕捉模型私有特征和文本共性。

**💡 创新点**

首次将边际注意力与输入-输出雅可比相连，证明相似的下一个词预测导致相似的雅可比，从而解释文本共性；并利用头级边际注意力构建离线KV缓存预算。

**🔧 技术方法**

边际注意力计算、雅可比统计、Hutchinson估计、可微解析、KV缓存淘汰、Expected Attention得分、离线预算表等技术。

**📊 数据集**

使用Census数据集（850篇自然文本，来自Pile 17个领域）以及预训练文档200篇用于预算；评估在RULER和LongBench等基准上。

**📈 对比分析**

与AdaKV+EA、KVzip等方法对比，HM+EA+EA在RULER 4×、8×、16×压缩下的得分与KVzip相近，优于多方法，且无需运行时预算开销。

**⚠️ 局限性**

理论仅覆盖输入-输出雅可比统计，边际注意力与雅可比的对应仍经验性；只在英文短文本上验证；HM+Guide需要额外前向传播，实际部署受限；未证明对所有语言和大模型的通用性。

---

## 495. FinNextAssist: Towards Professional Financial Deep Research Assistant

**arXiv ID:** 2610.03174 | [PDF](https://arxiv.org/pdf/2610.03174v1)

**作者:** Xiangyu Li `[一作]` (South China University of Technology), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向金融深度研究的端到端框架，结合任务规划、证据编译、推理引擎和报告组装四个阶段；

**💡 创新点**

创新点在于将金融分析拆解为类型化的子任务，利用专门的 TabAgent 和 HeteroAgent 对表格与跨模态数据进行标准化和对齐，并将数值运算交给可执行的金融技能，确保推理过程中的数据一致性和可追溯性；

**🔧 技术方法**

采用大型语言模型（如 Claude‑Sonnet‑4.5、GPT‑5 等）作为主体，辅以 Qwen‑2.5‑7B‑Instruct 的子代理，使用检索 API（Serper、Jina Reader、yfinance、AkShare、SEC EDGAR 等）和专用金融工具（比率计算、货币兑换、CAR 等）实现多模态数据处理；

**📊 数据集**

在三个公开基准上进行评测：FinDeepResearch、Finance Agent Benchmark 和 FinTMMBench‑Web，涵盖完整报告、SEC 文件问答和 Web 检索的多时态多模态推理；

**📈 对比分析**

与 23 种基线（包括商业和开源 DR 系统）对比，所提框架在 FinDeepResearch 上获得 48.5 分，领先最强基线 10.5 分；在 Finance Agent Benchmark 上实现 77.39% 的准确率，提升 20.68 分；在 FinTMMBench‑Web 上 EM、F1、Accuracy 及 LLM‑judge Accuracy 均超过 Web 基线 25.87、41.26、33.91、30.02 分；

**⚠️ 局限性**

局限性主要在于子代理和技能的手工设计与调度，需要手动指定哪些子任务使用哪些子代理；模型对非金融领域的迁移能力尚未验证，且在高层次分析（抽象、解释）上的提升仍有限。

---

## 496. BRIDGE: Bridging Routine Inputs and Digital modelling for Geotechnical Engineering

**arXiv ID:** 2610.03170 | [PDF](https://arxiv.org/pdf/2610.03170v1)

**作者:** Roshan Philip Saji `[一作]` (New York University Abu Dhabi), Mostafa E. Mobasher `[通讯]` (Mansoura University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了 BRIDGE 框架，用于从常规土壤指标预测机械性能，填补表征缺口。

**💡 创新点**

创新点在于将物理一致性嵌入数据驱动的校准流程，采用分层分箱与小样本多阶段逆向校准，构建可验证的机械响应包络；框架模块化可扩展。

**🔧 技术方法**

技术包括分箱与子分箱、有限元单元轴对称模型（Extended Cam‑Clay）、多阶段优化（COBYLA+Nelder–Mead+Brent）、预测包络（凸包/边界框）、采样方法（Puchwein、Latin Hypercube、Sobol 等）。

**📊 数据集**

使用东京海洋黏土数据库 Tokyo‑CLAY/14/67760 共 2,766 样本，筛选后 2,609 样本，包含基本属性和机械属性。

**📈 对比分析**

通过三项数值研究比较分箱准则、是否二级分箱、采样方法与样本预算，评估准确度（预测包络覆盖率）和计算成本；平均验证覆盖率 ≥ 80%，收敛率 95.8%，计算成本在 10% 以内可下降。

**⚠️ 局限性**

限制：对低 SU 或低 e 区域的校准收敛率下降；预测区间宽，受数据散布限制；对单一 ECC 模型的依赖；需进一步评估不同采样策略在所有单元的表现。

---

## 497. Data-Free Weak-Form Staggered Neural Operators for Magneto-Mechanical Coupling in Finite-Strain Elastomers

**arXiv ID:** 2610.03156 | [PDF](https://arxiv.org/pdf/2610.03156v1)

**作者:** Alireza Yazdandousthamedani `[一作]` (Technische Universität Dresden), Michael Kaliske `[通讯]` (Technische Universität Dresden)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一种数据无标注、基于弱形式残差的耦合磁-力学神经算子（WSNO），可对一族不同材料、几何和加载条件下的有限应变磁-力学问题快速预测。

**💡 创新点**

创新点包括：①将有限元弱形式残差直接嵌入物理约束，避免自动微分和标注数据；②引入磁场和位移两份独立神经算子，通过分阶段（staggered）优化保持耦合但避免梯度冲突；③可与神经初始化的Newton迭代结合，显著加速非线性求解。

**🔧 技术方法**

采用了有限算子学习（Finite Operator Learning, FOL）框架、Fourier神经算子（FNO）、深度学习优化、基于自动微分的梯度停止（stop‑gradient）以及传统有限元软件作为参考验证。

**📊 数据集**

使用多种合成数据集：二维随机分布圆形磁性包裹、不同磁相对磁导率的随机微结构、面积分数变化的单个包裹、基于傅里叶系数生成的光滑材料分布以及三维空心圆柱几何参数化。

**📈 对比分析**

通过与高精度有限元求解对比，评估相对 L₂误差；在随机微结构下磁势误差<1%，位移误差<2%；在面积分数外推时磁势误差<1%，位移误差≈3%；神经初始化Newton在单步加载下仅需4–5次迭代，速度提升约两百倍。

**⚠️ 局限性**

主要局限：对形态差异较大的外推样本（如圆形包裹对光滑分布）直接预测精度下降；训练成本较高，需大量样本；目前仅验证了静态磁场、线性磁性和可变几何，尚未扩展至非线性磁饱和、各向异性或时变问题。

---

## 498. Does Physics Live in the Activations? Localizing Physical Quantities in Video Diffusion Models

**arXiv ID:** 2610.03154 | [PDF](https://arxiv.org/pdf/2610.03154v1)

**作者:** Jonas Kneifl `[一作]` (IDEAS Research Institute), Kamil Deja `[通讯]` (Warsaw University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对三个开源视频Diffusion Transformer（Wan 2.1、CogVideoX‑1.5、Open‑Sora 2.0）进行内部表示的物理量可线性解码实验，研究其是否真正“理解”物理规律。

**💡 创新点**

发现物理量（速度、方向、动量、重力等）可在去噪过程早期以高精度线性解码，信息在生成过程中被主动构建、存储在靠近物体的局部 token 位置，并且解码方向对场景变化具有一定可迁移性，可通过注入探针方向直接改写生成的视频运动。

**🔧 技术方法**

采用 Ridge 回归线性探针、Token 局部聚合、噪声级（SNR）分析、跨层跨噪声的 Probe 设计，以及对输入噪声潜在层和条件图像的对照基线。

**📊 数据集**

使用 Genesis 物理模拟器渲染的合成视频数据集，包括：滚动无打滑、弹性碰撞、多球碰撞、抛体运动等场景，提供精确的物理量标签。

**📈 对比分析**

与模型自身的噪声潜在输入基线对比，探针在大多数模型中均显著优于基线；在中间层和 0 dB 噪声水平下，速度、方向、动量、重力等量均达到 R²>0.8 的解码效果；局部 1–2 % token 的 ROI 能比全局读取更好，且在背景、形状、视角等场景变换后仍保持 70–90 % 的可迁移性能，并可通过注入探针方向实现生成运动的显著偏转。

**⚠️ 局限性**

仅限于无粘性、刚体、少数物体的简单机械运动；不涉及能量耗散、旋转、柔性或多体接触、流体、热力学或光学现象；探针只能解码状态而非物理定律；对 Open‑Sora 的迁移与控制效果相对较差，且对更复杂物理量的直接注入控制尚未实现。

---

## 499. Keeping JEPA World Models Plannable When Little of the Frame Moves

**arXiv ID:** 2610.03137 | [PDF](https://arxiv.org/pdf/2610.03137v1)

**作者:** Florian Strohm `[一作]` (Fraunhofer IPA), Marco Huber `[通讯]` (Fraunhofer IPA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个多小物体推送基准，并在该基准上实现了基于语言描述目标的规划方法。

**💡 创新点**

发现并修复了JEPA世界模型在帧无动作响应情况下的表示崩溃，提出单步逆动力学辅助损失；同时设计了可在已训练世界模型上直接使用的语言目标头，支持分阶段语言指令完成推送任务。

**🔧 技术方法**

使用JEPA视觉世界模型、ViT‑Tiny编码器、Transformer预测器、CEM规划器，加入逆动力学头（共享、stop‑gradient）和跨注意力的语言头，训练阶段句子序列。

**📊 数据集**

自定义2D pymunk仿真生成的训练与评估数据，包含10k条基准轨迹、40k个带阶段句子的训练场景，基准分为四个难度层级。

**📈 对比分析**

与原始JEPA（plain）、PushT、脚本控制器和视觉目标oracle比较。修复后推送成功率从≈0提升到0.35，导航任务达到0.84（oracle 1.00），分阶段语言指令将推送成功率提升到0.25–0.33，接近目标框架oracle。

**⚠️ 局限性**

仅限于2D仿真；对未见词汇鲁棒性不足；需使用阶段句子并保持特定调度；在最难层级性能仍低；尚未证明可迁移至更大或更复杂的世界模型。

---

## 500. CalCErt: Bin-wise Certification of Confidence Calibration in Medical Image Classification

**arXiv ID:** 2610.03142 | [PDF](https://arxiv.org/pdf/2610.03142v1)

**作者:** Leo Fillioux `[一作]` (Université Paris-Saclay), Jose Dolz `[通讯]` (LIVIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种后置认证方法，能够为任意预训练的可微分类器在给定 ℓ₂ 范围内对每个置信度区间（bin）提供误差上界，确保置信度校准在对抗扰动下保持可控；

**💡 创新点**

创新点在于将本地 Lipschitz 常数与统计集中偏差（Hoeffding 上界）相结合，构造可解的 bin‑wise 误差上界，且不需要对模型进行再训练，首次将认证扩展到置信度校准而非仅预测类别；

**🔧 技术方法**

核心技术包括：本地 Lipschitz 常数估计（梯度范数）、Hoeffding 统计上界、后置（post‑hoc）认证框架以及对置信度区间的自适应子区间（robust sub‑bins）设计；

**📊 数据集**

实验使用 MedMNIST 11 个 2D 医疗图像分类数据集，采用 Vision Transformer (ViT-Ti/S/B) 与 ResNet‑18 作为 backbone；

**📈 对比分析**

与两种基线（对抗扰动下直接校准和特定攻击校准）相比，本文方法在 ID 场景下覆盖率 0.90–0.99，显著高于基线的 0–0.44；在 OOD 场景下覆盖率也保持 0.6–0.93，且紧凑度相当或更好；

**⚠️ 局限性**

主要限制包括：置信度区间内样本分布不均导致统计上界过大；本地 Lipschitz 常数估计可能过大，导致证书过于保守；仅保证校准而不保证分类准确性；缺乏对不同攻击单独 fine‑tuning 的支持；

---

## 501. Trading Strategy Optimization via Textual Gradient

**arXiv ID:** 2610.03128 | [PDF](https://arxiv.org/pdf/2610.03128v1)

**作者:** Chaoqun Yang `[一作]` (National University of Singapore), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出 TradeGrad 框架，通过经验驱动的文本梯度与跨周期稳健目标，自动设计稳健的量化交易策略。

**💡 创新点**

创新点在于将全局经验记忆与多尺度文本梯度估计结合，并引入低尾跨周期稳健目标（CPRO），提升策略的时间稳健性。

**🔧 技术方法**

采用 LLM 生成文本梯度、全局记忆抽取、采样式梯度估计、多尺度结构与局部修订、以及跨周期评估与 CVaR 形式优化。

**📊 数据集**

实验使用 2018‑2025 年中国 A 股和美国股市每日数据，分别在 CSI300/S&P500 及相应指数的横截面与时序设置下进行。

**📈 对比分析**

与传统指数、经典策略、岛屿算法、QuantEvolve 及 TextGrad 等基线对比，TradeGrad 在所有四种设置下在样本内和样本外均取得最高得分，并显著提升年化收益、夏普比率和最大回撤。

**⚠️ 局限性**

局限性包括对 LLM 质量的依赖、记忆窗口与采样大小的超参数调优、以及对高频或多资产类别的适用性尚未验证。

---

## 502. Safe Streaming Flow Planning by Aligning Sampling Dynamics with Execution Dynamics

**arXiv ID:** 2610.03132 | [PDF](https://arxiv.org/pdf/2610.03132v1)

**作者:** Seunghwan Jang `[一作]` (Nanyang Technological University), SooJean Han `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于流匹配的安全流式规划器，能够在物理时间下逐步生成状态轨迹，并通过高阶控制屏障函数实现实时安全约束。

**💡 创新点**

将流式采样时间与执行时间对齐，使用分层状态预测消除第二阶动力学中的惯性捷径，并将高阶CBF仅应用于已执行的步骤，从而将优化求解次数从O(HK)降到O(H)。

**🔧 技术方法**

流匹配/流式采样、分层状态预测、离散时间高阶控制屏障函数、PD控制、二次规划（QP）等。

**📊 数据集**

收集自Maze2D、F1TENTH赛道、MuJoCo Hopper以及Gazebo仓库等四个仿真环境的演示数据。

**📈 对比分析**

与Diffuser、SafeDiffuser、FlowMatcher、SafeFlowMatcher等安全生成式规划器以及经典RRT*、A*进行比较；在四个基准上取得更低的执行时安全违规率、更高的成功率或更快的推理时间，尤其在长时域与闭环执行下表现突出。

**⚠️ 局限性**

需要丰富的演示数据；CBF对障碍形状有限制且需预设；在某些情形下高阶CBF不可行导致失败；仅在仿真中验证，真实机器人尚未测试。

---

## 503. TSGuard: A Real-Time Framework for Detecting and Imputing Missing Data in Streaming Time Series

**arXiv ID:** 2610.03147 | [PDF](https://arxiv.org/pdf/2610.03147v1)

**作者:** Imane Hocine `[一作]` (University of Luxembourg), Grégoire Danoy `[通讯]` (University of Luxembourg)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了TSGuard，一个面向实时流数据的完整缺失值处理框架，能够检测、推断、验证并向运维人员展示缺失值补全结果。

**💡 创新点**

创新点包括：①将缺失值处理视为闭环数据质量工作流；②使用轻量级图神经层 + LSTM 的混合时空补全模型；③在补全后立即进行物理范围和空间一致性约束验证，并在违反约束时计算邻域回退估计；④将决策过程、报警和解释开放给操作员，甚至通过LLM辅助生成自然语言说明。

**🔧 技术方法**

技术手段包括：图神经网络（Graph Neural Network）聚合邻域上下文，LSTM 捕获时序动态，约束验证模块（物理范围与空间一致性规则），回退估计公式，Apache IoTDB 持久化，LLM 辅助解释。

**📊 数据集**

使用北京 AQI-36 空气质量监测数据集，按 PriSTI 方式产生缺失模式进行评估。

**📈 对比分析**

与 PriSTI（离线）、PriSTI-ON（仅前向推理）和 ORBITS（在线）比较。TSGuard 的 MAE 为 16.13，RMSE 为 28.37，实时延迟约 60 ms，吞吐约 180 传感器更新/秒；相比 ORBITS MAE 18.16、RMSE 29.35、延迟 50 ms，TSGuard 在约束验证与回退方面更稳健，虽然略高延迟。PriSTI-ON 速度慢（≈ 6000 ms）。

**⚠️ 局限性**

局限性：仅在单一 AQI 数据集上验证；约束规则为静态手工设定，缺乏自适应能力；回退估计仅基于邻域平均，可能在邻域缺失严重时失效；未针对更大规模传感网络或多领域数据进行评估；离线训练与在线推理的统一性待改进。

---

## 504. Gains and Collapse in On-Policy Distillation:A Reinforcement Learning Perspective

**arXiv ID:** 2610.03185 | [PDF](https://arxiv.org/pdf/2610.03185v1)

**作者:** Han Cui `[一作]` (Zhejiang University), Yue Zhang `[通讯]` (Westlake University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了On‑Policy Distillation（OPD）在语言模型后训练中的行为，发现它通过教师的隐式奖励放大学生已具备的行为，导致性能提升但并不扩展学生能力。

**💡 创新点**

将OPD解释为教师隐式奖励的RL优化，揭示奖励劫持导致的过长/重复生成，并提供基于候选池的掩蔽和SFT预热两种可行的防崩溃策略。

**🔧 技术方法**

逆KL基准的OPD、PPO式梯度更新、NLL/Min‑10NN诊断、奖励优势评估、掩蔽损失与SFT warm‑up。

**📊 数据集**

DeepMath‑103K（≥6层级）、OpenThoughts3、AMC23、AIME 24‑26 等数学推理测试集。

**📈 对比分析**

通过pass@k、mean@4、覆盖率、手工审核比较，OPD在小k下提升准确率但不增加覆盖；在失败设置中奖励劫持导致长串/重复，掩蔽或warm‑up可提升约3–6个百分点。

**⚠️ 局限性**

实验仅限小型模型与数学推理任务，扩展到大模型或其他领域未知；掩蔽/warm‑up仅针对过长/重复问题，未必通用；覆盖评估受采样预算与解码策略限制。

---

## 505. Exploring the Trade-Off Between Structured Pruning and Fault Tolerance in Deep Neural Networks for Space Applications

**arXiv ID:** 2610.03117 | [PDF](https://arxiv.org/pdf/2610.03117v1)

**作者:** Toon Vinck `[一作]` (Magics Technologies), Peter Karsmakers `[通讯]` (KU Leuven)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过迭代的全局结构化剪枝（去掉通道）并结合量化感知训练，系统评估了在单事件翻转（SEU）环境下不同宽度的深度神经网络对误差的鲁棒性。

**💡 创新点**

创新点在于：①首次将结构化剪枝与高位宽累加寄存器的误差注入结合；②通过将运算量与SEU发生概率关联，构造了 FIT 率指标，量化了剪枝对系统可靠性的综合影响。

**🔧 技术方法**

采用的技术包括 PyTorch + Brevitas 量化框架、分层的结构化剪枝策略（Taylor 重要性评估）、单点位翻转注入工具和基于 MAC 操作计数的 SEU 风险评估。

**📊 数据集**

使用 EuroSAT 数据集，对 ResNet‑18 和 MobileNetV2 两个网络架构进行实验，训练、剪枝、量化并注入约 10 万个单点错误。

**📈 对比分析**

对比方法是：在每个剪枝阶段测量无误差的准确率、SDC‑1 误差率以及基于 MAC 计数的 FIT 率；实验结果显示，虽然 SDC‑1 随模型变窄而升高，但 FIT 率在 ResNet‑18 上基本保持不变，MobileNetV2 的 FIT 率甚至略有下降，证明剪枝后系统整体可靠性可保持甚至提升。

**⚠️ 局限性**

局限性包括：仅考虑单位翻转而忽略多位翻转、只模拟单一简化 PE 的数据流、未覆盖完整的加速器架构、实验数据集和网络模型有限，缺乏对真实航天任务级别可靠性影响的进一步验证。

---

## 506. Lightweight and Resource-Efficient Perception for Robotic Guide Dogs

**arXiv ID:** 2610.03187 | [PDF](https://arxiv.org/pdf/2610.03187v1)

**作者:** Jinse Kwon `[一作]` (Electronics and Telecommunications Research Institute), Jemin Lee `[通讯]` (Jeonbuk National University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `51c0528b-f690-4182-ae60-bb5f046c276c` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了基于360相机和2D LiDAR的全设备轻量感知系统，用于四足机器人导盲犬，实现实时深度估计、移动目标检测与路径解释。

**💡 创新点**

引入角度条件六参数仿射校正融合单目深度与LiDAR，采用四面立方体图像分解与边缘融合，并在低功耗硬件上实现实时人类中心导航与VLM驱动路径解释。

**🔧 技术方法**

SC-DepthV3单目深度估计、YOLOv8m目标检测、Hailo-8 NPU加速、ZeroMQ异步流、Qwen2.5-VL VLM+TTS、Jetson AGX Orin + Ricoh THETA Z1 360摄像机 + SLAMTEC C1 2D LiDAR。

**📊 数据集**

GuideDogQA 真实世界 egocentric 导盲基准（相对深度与对象识别），以及人工场景背景墙测试和 AI‑Hub adverse‑weather 图像用于深度评估。

**📈 对比分析**

与 GPT‑4o 等大型 VLM 在 GuideDogQA 上对比，系统在相对深度任务中取得 83.8%（GPT‑4o 为 67.1%），对象识别 COCO 覆盖子集 95.4%；实时性 25.3 FPS/face，功耗 <55 W，近距离深度误差 0.34 m，中间距误差偏大。

**⚠️ 局限性**

存在中间距深度偏差受背景支配、受限于 COCO‑80 词汇导致对非 COCO 目标识别不足，以及尚未通过盲人用户实地评估。

---

## 507. Predictor-Guided Latent Space Codon Optimization for Maximizing Protein Expression

**arXiv ID:** 2610.03098 | [PDF](https://arxiv.org/pdf/2610.03098v1)

**作者:** Alberto Caron `[一作]` (Johnson & Johnson Innovative Medicine), Rui Liao `[通讯]` (Johnson & Johnson Innovative Medicine)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于预训练 mRNA 语言模型潜在空间的连续梯度优化方法 LSCO，用来在保持蛋白序列不变的前提下，改进 mRNA 的翻译效率与结构稳定性。

**💡 创新点**

创新点包括：①将离散的密码子空间映射到连续潜在空间，从而实现梯度搜索；②构建不确定性感知的表达预测器（LCB）作为主目标；③加入最小自由能（MFE）正则化以保证结构稳定；④利用蛋白到密码子反向翻译的自然性先验约束；⑤采用受限解码保证蛋白序列完整。

**🔧 技术方法**

技术方法主要有：变分掩码自动编码器（VMAE）预训练的 mRNA 语言模型、深度集成表达预测器、MFE 预测器、CodonBERT 反向翻译模型以及基于拉格朗日乘子/梯度下降的连续潜在空间优化。

**📊 数据集**

数据集包括：15M+ 的 OAS‑mRNA 数据集用于预训练；基于实际实验测得的抗体表达（A280）和结构信息（MFE、GC% 等）的 54 只单克隆抗体样本用于评估；此外使用 ViennaRNA 计算 MFE。

**📈 对比分析**

与频率基方法、CodonBERT 反向翻译、ICOR、CodonTransformer、GEMORNA 等多种基线在 54 抗体样本上进行比较。LSCO 在内部 1D 表达预测器及两款独立评估器（1D 与 2D GNN）上均获得最高或可比的预测表达值；在 MFE、GC% 与 CAI 等结构与宿主兼容性指标上也表现优于或与基线持平，表明在表达与稳定性之间实现了良好平衡。

**⚠️ 局限性**

主要局限包括：评估基于预测模型而非真实实验测量，缺乏大规模体外验证；优化过程仍需依赖预训练模型的质量，潜在空间可能无法覆盖所有生物学可能；对不同宿主细胞或特定表达系统的适用性尚待进一步研究。

---

## 508. Hindsight-Guided Rationale Distillation for Rare Disease Diagnosis

**arXiv ID:** 2610.03176 | [PDF](https://arxiv.org/pdf/2610.03176v1)

**作者:** Aarav Singh `[一作]` (Dr. Shyama Prasad Mukherjee International Institute of Information Technology), Navyansh Singh `[通讯]` (Dr. Shyama Prasad Mukherjee International Institute of Information Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在ZebraMap稀有病诊断任务中，作者尝试将教师模型生成的包含真值标签的Chain‑of‑Thought（CoT）推理轨迹，用监督微调（SFT）训练学生模型，并对轨迹进行正则表达式过滤以消除“ground truth is X”类标签泄漏，形成过滤版StudentF模型。

**💡 创新点**

创新点在于首次量化标签可见生成导致的GT标签泄漏（trace contamination）在医疗诊断中的严重影响，并证明通过静态过滤可以在保持学生模型大小的同时，略微提升准确率（相较教师p<0.001），同时揭示过滤成本与频率依赖的知识迁移。

**🔧 技术方法**

技术包括：教师生成带标签的CoT轨迹，学生使用QLoRA 4‑bit量化的Instruct模型进行SFT；正则表达式过滤去除“ground truth is X”槽；评估使用自定义Accuracy（M3）及其上界/格式兼容等指标；统计检验采用聚类置换检验和Bootstrap置信区间。

**📊 数据集**

数据集为ZebraMap——从PubMed案例报告构建的稀有病多模态知识图谱，包含98,038张临床图像及患者级案例，按自定义训练/评估拆分，共计42种疾病。

**📈 对比分析**

与教师模型、无过滤学生以及无CoT的Base模型对比，过滤版StudentF在主Accuracy指标上比教师高约+X个百分点（p<0.001），无过滤学生未显著超越教师；但整体准确率仍低于BM25检索（≈30%），显示任务存在上限。

**⚠️ 局限性**

局限性包括：静态过滤无法区分正确与错误GT标签导致的性能损失；性能提升主要集中在高频疾病，低频疾病效果有限；实验仅使用单个训练实例，未评估训练随机性；以及对真实诊断可信度的评估不足。

---

## 509. Behavior Pack Optimization for Video MLLM Post-Training

**arXiv ID:** 2610.03141 | [PDF](https://arxiv.org/pdf/2610.03141v1)

**作者:** Zhaolu Kang `[一作]` (Peking University), Kaiyue Zhou `[通讯]` (Chengdu Minto Tech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了行为包优化（BPO）框架，利用多视角对照视图联合奖励，改进视频多模态大语言模型的后训练，使模型更依赖视觉证据而非表面特征。

**💡 创新点**

创新点在于：①将整组对照视图作为优化单元；②引入锚相对优势估计，避免小批量方差；③按问题类型选择视图集合，并通过跨视图合同奖励（稳定性、敏感性、诚实放弃）引导行为。

**🔧 技术方法**

采用的技术包括：行为包优化（BPO）+ GRPO框架、锚相对优势估计、基于Verifier的多维奖励（答案正确、证据支持、结构一致、格式、合同惩罚）、EST结构化输出接口。

**📊 数据集**

使用的评测数据集有：TempCompass、MVBench、NExT‑QA、Video‑MME、LongVideoBench，以及跨模型验证的LLaVA‑Video‑7B。

**📈 对比分析**

在与同算力的GRPO、Video‑R1、TPO等基线对照时，BPO在宏观准确率提升4.7个百分点、时间硬件子集提升7.8个百分点、放弃指标提升20个百分点；在跨模型和更长视频任务中亦保持显著优势。

**⚠️ 局限性**

局限性在于：依赖可用的对照干预与Verifier信号，当前视图集合主要涵盖证据移除、时间顺序和放弃场景，难以覆盖开放式生成、密集视频或更大模型的需求。

---

## 510. Foresight: planning future perception in streaming VLMs without retraining

**arXiv ID:** 2610.03123 | [PDF](https://arxiv.org/pdf/2610.03123v1)

**作者:** Ashok Prasad Neupane `[一作]` (Independent Researcher), Danda Pani Paudel `[通讯]` (NAAMII)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Foresight，一个训练‑free 的主动异步流式视觉‑语言模型框架，利用冻结的 VLM 对未来的预测能力，在不暂停感知的情况下动态规划何时、何处以及多密度地采样并执行后续推理。

**💡 创新点**

创新点在于：将未来预测视为控制计算的手段而非单纯输出；使用双子 VLM（Ingest 与 Think）共享 KV cache，让思考 LLM 在后台实时规划；通过差分计划解码与轻量化更新实现无训练的主动异步推理；以及通过自适应帧率与对象级剪枝动态调整计算资源。

**🔧 技术方法**

核心技术包括双流架构、共享 KV cache、实时采样控制、差分计划解码、轻量化差分更新、基于 Qwen3‑VL‑8B 的冻结视觉‑语言模型、两帧视频剪辑的输入流水线。

**📊 数据集**

使用了 OmniPro（主动响应任务）、StreamingBench（多任务视频理解）和 OVO‑Bench（实时/前向/后向）三大流式评测数据集。

**📈 对比分析**

与已训练的基线（MiniCPM‑o、LiveStar、StreamAgent 等）进行对比，Foresight 在 OmniPro Online 取得 23.0 mean joint F1（比最强训练基线提升 9.5%），在 StreamingBench 上获得 79.3% 的整体准确率（最高），在 OVO‑Bench 上尤其在 Forward 任务提升 18.7%；整体计算速度接近实时，表现优于实时推理基线。

**⚠️ 局限性**

主要局限：依赖底层 VLM 的未来预测与时序定位能力，误报后可能继续重复报告；对提示词敏感；在更强的后端模型或更专业的预测模块下仍有提升空间。

---

## 511. Investigating the Role of Reasoning-Language Alignment in Monolingual Retrieval-Augmented Generation

**arXiv ID:** 2610.03136 | [PDF](https://arxiv.org/pdf/2610.03136v1)

**作者:** Oliver Hauck `[一作]` (Johannes Gutenberg University Mainz), Katharina von der Wense `[通讯]` (Johannes Gutenberg University Mainz)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在单语德国 RAG 环境下强制 LLM 采用目标语言进行推理是否有益，并构建对应测试床与 QA 基准。

**💡 创新点**

发现对齐推理语言与检索文本语言能提升准确率，且更丰富的目标语言结构可降低强制推理的成本，且提升不受语言熟练度影响。

**🔧 技术方法**

使用 Qwen3 语言模型、结构化章节式检索知识库、prompt‑hacking 控制推理语言、LLM‑as‑a‑Judge 与人工评测相结合的实验框架。

**📊 数据集**

构建以《Das Schwarze Auge》德文源书为基础的 1.13M token 知识库，包含 585 条单段问答和 30 条多跳问答。

**📈 对比分析**

对比强制德语/英语/法语、无约束推理、禁用推理等设置；在检索设置下，强制德语达到或略低于无约束英语的性能，优于强制法语；推理仅在多跳问题中显著提升。

**⚠️ 局限性**

局限性包括仅测试单一模型与语言对、仅采用 prompt‑hacking 控制推理、三十道多跳样本规模有限、LLM‑as‑a‑Judge 评价存在偏差、检索超参数固定。

---

## 512. Geometry-Aligned Semantic Matching for Cross-Modal Planar Image Registration

**arXiv ID:** 2610.03167 | [PDF](https://arxiv.org/pdf/2610.03167v1)

**作者:** Zhiwei Wang `[一作]`, Edmund Y. Lam `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种跨模态平面图像配准的稠密匹配框架 CDPM，旨在在大幅度模态差异下实现稳定、精确的几何对应。

**💡 创新点**

创新点：①使用几何一致的 patch‑level 对比学习对预训练的 DINOv3 进行两阶段适配，使语义表示更能反映真实空间对应；②构建 DINO‑Centric Feature Pyramid（DCFP），让多尺度 DINO 表示主导对应估计，并通过轻量 CNN 辅助细节增强，避免细尺度漂移；③结合自注意力、残差细化与 RANSAC 的全流程，实现高效且精确的平面配准。

**🔧 技术方法**

技术手段：DINOv3 视觉基础模型、跨视角交叉注意力、基于 patch 的对比学习、Spatial Feature Block（SFB）、轻量级 CNN 细节分支、双阶段迭代细化、置信度映射与二元交叉熵监督、RANSAC 估计 homography。

**📊 数据集**

使用的数据集：公开 VIS‑IR（可见‑红外）和 GoogleMap（卫星‑地图）两大跨模态数据集；自建 PCB‑Layout（PCB 实图与布局图）数据集，用于验证在工业场景中的泛化。

**📈 对比分析**

与 RoMa、RoMa v2、GFNet、LoFTR 等方法比较：在 VIS‑IR 上 mACE 由 5.83 降至 2.78（≈23%提升），AUC@3、@5、@10、@20 分别提升 7.36/13.40/13.75/10.42%；在 GoogleMap 上 AUC@3、@5、@10、@20 为 52.37/67.71/81.85/89.91，mACE 2.50；在 PCB‑Layout 上 AUC@20 80.38，mACE 3.95。模型参数 307 M，FLOPs 1.00 G，推理时间 74 ms，显著低于对手。

**⚠️ 局限性**

局限性：①对极端光照或高度重复纹理仍可能出现细尺度漂移；②两阶段 DINO 适配需要额外训练成本；③在极大模态差异（如光学‑雷达）下的泛化尚待进一步验证。

---

## 513. Aggregate accuracy conceals concentrated temporal vulnerability in a spiking speech classifier

**arXiv ID:** 2610.03155 | [PDF](https://arxiv.org/pdf/2610.03155v1)

**作者:** İsmail Can Dikmen `[一作]` `[通讯]` (İstinye University), İsmail Can Dikmen (İstinye University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对冻结的 SpikeSCR 语音命令分类器，在 100 条验证样本的 5 ms 相邻 bin 计数移动（保持总计数不变）上保留并记录了 725,070 次预测，构建了完整的局部决策地图，分析了不良、补救和横向转移的分布，并通过内部激活追踪和清洁激活置换验证内部变化与输出决策的关联。

**💡 创新点**

首次提出在事件驱动时序扰动下完整枚举邻域并保留所有结果的局部审计方法；揭示了平均准确率掩盖局部脆弱性、优势与劣势转移集中分布、以及不同搜索策略（均匀、边缘引导、梯度搜索）对发现脆弱源的差异；通过内部激活置换证明局部干预可恢复多数不良预测，验证了干预有效性而非唯一性。

**🔧 技术方法**

使用 SpikeSCR 结构、事件时间编码、5 ms bin 转换、相邻 bin 一计数移动操作、统一/边缘/梯度搜索、内部激活记录、清洁激活置换、CPU/GPU 对比重放、q/k LIF 路径隔离、两种计数读取 SNN 复制品的完整邻域审计等技术。

**📊 数据集**

Spiking Speech Commands (SSC) 数据集；挑选 100 条标签均衡的验证样本进行审计。

**📈 对比分析**

将审计结果与平均准确率、统一采样、低边缘采样、梯度排序等方法比较。结果显示：平均准确率 86.08%，均匀邻域准确率提升至 84.54%，但 13/84 初始正确样本存在不良邻居；低边缘搜索在 8,400 次查询下发现 9.42/13；梯度排名 7-8；完整枚举发现全部。两计数读取复制品的类变更率差距高达 5.21 倍，凸显同一准确率下不同模型脆弱性差异。

**⚠️ 局限性**

局部审计仅针对单一冻结模型的 100 条样本，难以推广到全部 9,981 条验证集；邻域扰动仅为单计数相邻 bin 移动，未覆盖更大时序扰动或多计数变动；未对训练阶段进行脆弱性自适应或增强；内部激活置换依赖于可访问清洁状态，未证明为可部署的防御方法；两复制品仅在计数读取架构上做对比，未验证在更复杂网络中的可迁移性。

---

## 514. Benchmarking Literature Retrieval for a Model Organism: A Dictyostelium Case Study

**arXiv ID:** 2610.03130 | [PDF](https://arxiv.org/pdf/2610.03130v1)

**作者:** Yun Wang `[一作]` (University of Ljubljana), Blaž Zupan `[通讯]` (University of Ljubljana)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Dictyostelium（模型生物）文献检索领域构建了一个基于dictyBase curator注释的检索基准，并通过实验探究了跨编码reranker、基因知识驱动的查询扩展以及全文检索对检索性能的影响。

**💡 创新点**

创新点在于①提出了针对小规模、词汇高度专业化的模型生物检索基准；②系统评估了跨编码reranker在不同模型适配性下的增益；③利用结构化基因注释实现低成本查询扩展；④通过全文段落检索揭示摘要不足时的检索瓶颈。

**🔧 技术方法**

使用的技术包括：两阶段检索（BM25+Dense + RRF）、跨编码rerankers（MiniLM‑L12、BGE‑m3、MedCPT、BGE‑Gemma）、基因知识驱动的查询扩展、全文段落检索与重排序后融合（RRF）。

**📊 数据集**

数据集为从dictyBase gene summary页面自动提取的1,656条基因相关查询与对应PubMed引用，覆盖20,447篇Europe PMC摘要；此外构建了可全文检索的1,124篇PDF子集，并对每条查询标注了抽象层级证据标签。数据已公开于Zenodo（https://doi.org/10.5281/zenodo.20308282）。

**📈 对比分析**

实验对比BM25基准、BM25+Dense融合、四个跨编码reranker以及外部Ragnarok系统，使用Recall@K（候选阶段）和MRR@K（最终排序）评估。结果显示：MedCPT和BGE‑Gemma显著提升MRR@10（+5.7pp、+11.4pp），MiniLM‑L12则不利；基因扩展提升BM25/密集检索召回，并在BGE‑m3、MedCPT下进一步提升MRR；全文段落检索在摘要证据不足的查询中将Recall@1000从0.78提升至0.97，MRR@10从0.16提升至0.39，效果最为显著。

**⚠️ 局限性**

局限性包括：查询自动抽取可能缺失上下文导致不完全自洽；inline citation仅提供已知引用，未覆盖所有相关文献；LLM生成的证据标签非专家评审；全文检索受限于开放获取PDF覆盖率，且全文段落检索后排名仍具挑战。

---

## 515. A Benchmark for Spatially Grounded Gesture Generation

**arXiv ID:** 2610.03105 | [PDF](https://arxiv.org/pdf/2610.03105v1)

**作者:** Anna Deichler `[一作]` (Kth Royal Institute Of Technology), Jonas Beskow `[通讯]` (Kth Royal Institute Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

建立了空间定向指点手势生成的基准数据集与评估框架，提出任务定义与评估协议。

**💡 创新点**

提供第一份配对的VR对话中指点手势与3D场景图的标注数据，独立的任务与评估设计将空间定位与时序对齐分离，并引入自然度用户研究。

**🔧 技术方法**

对比了基于流匹配的生成模型、检索式系统以及真值参考，使用空间-时间评估与人类自然度打分。

**📊 数据集**

约2000条自然VR对话中点点手势标注，1000条单目标脚本化手势与共享演员；每条都有对应的3D场景图和目标标签。

**📈 对比分析**

通过空间定位得分、时序对齐得分和自然度打分三维评估，检索式系统在空间定位上显著优于基线和捕获动作，但自然度与基线相当；时序评估区分系统，但与捕获动作的对比受参数影响。

**⚠️ 局限性**

评估指标与主观运动质量存在偏差，缺乏统一的全局时空一致性度量；数据规模与多样性仍有限，无法覆盖更复杂情境。

---

## 516. How to Find and Reuse Policies for Continuous Adaptation in Lifelong Reinforcement Learning

**arXiv ID:** 2610.03119 | [PDF](https://arxiv.org/pdf/2610.03119v1)

**作者:** Saptarshi Nath `[一作]` (Loughborough University), Andrea Soltoggio `[通讯]` (Loughborough University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Adaptive Mask Selection and Composition (AMSC)，一种在终身强化学习中利用在线任务相似度来选择并组合稀疏政策掩码的方法。

**💡 创新点**

创新点在于将非参数 Wasserstein 嵌入的任务相似度作为无监督的稀疏检索与权重指引，避免直接优化选择/组合权重，同时实现对新任务的正向迁移且无遗忘。

**🔧 技术方法**

技术包括非参数线性 Wasserstein 任务嵌入、稀疏 softmax 检索、L2 归一化掩码、在线更新与周期性重检索，以及稀疏掩码量化。

**📊 数据集**

实验使用 CT-Graph (CT28)、MiniGrid (MG16) 与 Continual World (CW10) 三个持续学习基准。

**📈 对比分析**

在 CT28 与 MG16 上 AMSC 的平均 AUC/FWT 分别达到 0.94/0.94 与 0.73/0.54，显著优于模块化基线且保持零遗忘，优于大部分重放方法；在 CW10 上虽 AUC 0.48 但 FWT 为负，表明在连续控制任务上效果有限。

**⚠️ 局限性**

主要局限在于相似度权重在所有层共享，导致在连续控制任务中无法充分利用层级特定的知识；另外稀疏检索对任务分布变化的鲁棒性与掩码存储增长仍待进一步压缩。

---

## 517. Ask, Relax, or Act? Evaluating Actionable Indeterminacy in LLM Preference Reasoning

**arXiv ID:** 2610.03102 | [PDF](https://arxiv.org/pdf/2610.03102v1)

**作者:** Ang Li `[一作]` (Chinese University of Hong Kong Shenzhen), Baoxiang Wang `[通讯]` (Chinese University of Hong Kong Shenzhen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出“可行动的不确定性”框架，并构建跨物体分配、会议排程、公寓选择与稳定匹配四种决策结构的 Solver‑Grounded Benchmark，用来衡量 LLM 在何时需要干预、澄清或约束修复的能力。

**💡 创新点**

创新点在于将不确定性分为可保持不确定与必需干预两类，并设计匹配动作/干预对的评估方法，展示响应内容要求对模型决策行为和可验证输出的显著影响。

**🔧 技术方法**

使用大型语言模型（GPT‑4、Claude‑5‑Sonnet、Gemini‑3.1‑Pro 等）进行单轮推理，配合精确求解器生成标签和固定检查器验证响应内容，采用 Accuracy、Intervention Flip Correctness (IFC) 与 Fully Correct (FC) 三个指标评估模型。

**📊 数据集**

采用自定义的 4,320 条任务组成的 V3 benchmark，任务覆盖四种决策结构，每个任务单独改变偏好、目标或约束来源，以构成匹配动作/干预对。

**📈 对比分析**

通过 Accuracy、IFC 与 FC 指标对比模型表现；原始提示下 Accuracy 约 92%，FC 仅 51%；加入显式 JSON 或证据请求后 FC 提升至 94–97%，IFC 亦大幅提升；不同模型与来源表现差异显著。

**⚠️ 局限性**

限制在于模型仍易产生不必要的干预，响应内容与规定不匹配导致验证失败；对约束修复与证据生成的能力有限；评估受限于固定任务设计和单轮交互。

---

## 518. Reversing the Clock: Layout-Aware Recovery of Design Intent from Clock Distribution Networks

**arXiv ID:** 2610.03182 | [PDF](https://arxiv.org/pdf/2610.03182v1)

**作者:** Sascha Tommasone `[一作]` (TechInsights Inc.), Steffen Becker `[通讯]` (Ruhr University Bochum)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了一套基于布局的时钟分配网络（CDN）逆向工程方法，利用扫描电镜图像与门级网表结合，恢复并分析IC的时钟树结构、缓冲、分频、路由与信号完整性，并通过案例验证其有效性。

**💡 创新点**

创新点在于首次从已制造的IC中完整重建CDN，结合网表与物理布局的四阶段管线，提出自底向上的时钟树恢复算法，并通过时序模拟得到实际时钟延迟与偏差；同时公开了算法插件与基准网表，促进复现与后续研究。

**🔧 技术方法**

主要技术包括：多层扫描电镜图像拼接与三维聚合、网表与布局一致性检查、时钟树恢复插件（结合结构化网表与物理信息）、图形化布局查询（射线投射、中心线图）、分支恢复与缓冲分析、分层金属利用与过孔计数、Crosstalk 评估、基于 RC 网络的 SPICE 时序模拟。

**📊 数据集**

使用的实验数据集为一颗450 nm 节点商业IC的三金属层布局与扫描电镜图像；为可复现性提供了开源的基准网表（Cloneless）与恢复插件，实测案例网表因保密无法公开。

**📈 对比分析**

通过对比恢复前后时钟树覆盖率（覆盖率 95%）以及时序分析得到的全局/局部时钟偏差（全局 8.77 ns、I2C 0.21 ns 等），表明方法能准确捕捉设计意图；实验中未与其它同类工具直接比较，但展示了在商业IC上成功重建完整 H‑tree/X‑tree 结构、缓冲与分频机制。

**⚠️ 局限性**

局限性包括：仅支持单驱动、树状 CDN，无法处理网格、复杂分频/乘法器或多驱动网；对三金属层以上的更深层技术节点验证不足；时序模型未考虑耦合、感抗与经验测量，可能导致延迟误差；对异常网表错误的容错性仍有提升空间。

---

## 519. Domain-Adaptive Data Assimilation for Global AI Weather Forecasting

**arXiv ID:** 2610.03172 | [PDF](https://arxiv.org/pdf/2610.03172v1)

**作者:** Minseok Seo `[一作]` (Korea Advanced Institute of Science and Technology), Changick Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于观测的领域自适应数据同化方法（DADA），通过在冻结的AI天气预报模型中仅优化初始状态扰动，使外部分析与预训练模型在观测空间中一致，从而提升实时预报性能。

**💡 创新点**

创新点在于将观测引导的误差最小化与模型冻结相结合，仅对初始扰动进行测试时优化，并利用预训练的神经观测算子把观测误差梯度反向传播到扰动空间；无需重新训练预报模型，也不需要重建ERA5分析，显著降低了初始条件源异质导致的性能衰退。

**🔧 技术方法**

使用了梯度下降优化（AdamW）对初始扰动进行测试时优化、基于Transformer的观测算子网络（包含表面编码器、垂直编码器、查询编码器和多层感知机）、物理基线+残差学习、以及多种观测类型（ATMS亮度温度、GNSS‑RO、温度、湿度、风速、站压）对应的网络头。

**📊 数据集**

训练观测算子时使用2020–2021年NOAA UFS GEFS‑v13诊断数据与ERA5气象场；评估数据来自2023年的GFS、HealDA和ERA5背景，覆盖Aurora、FengWu、GraphCast、AIFS及FCNv3的多模型组合。

**📈 对比分析**

通过与原始背景、ERA5参考初始化以及不同模型的比较，使用纬度加权RMSE（确定性）和CRPS（概率性）评估。结果显示，DADA能在GFS、HealDA等非ERA5源下将RMSE/CRPS下降约30–70%（视变量和先导时间而定），使预报质量接近ERA5初始化，显著提升短期和中期预报能力。

**⚠️ 局限性**

局限性包括：需要已存在的背景状态与预训练的观测算子；对缺失的大尺度或特定变量信息恢复有限；适用性受限于模型动态与观测算子覆盖范围；需要在每种新背景或新模型前先训练观测算子；未能取代完整的数值同化系统，只是提供一种适配层。

---

## 520. Bridging Research and Practice: A Systematic Evaluation of Generalist and Dermatology-Specific Models in Clinical Skin Lesion Classification

**arXiv ID:** 2610.03193 | [PDF](https://arxiv.org/pdf/2610.03193v1)

**作者:** Emanoel dos Santos `[一作]` (Universidade Federal de Pernambuco), Tsang Ing Ren `[通讯]` (Universidade Federal de Pernambuco)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了多种通用与皮肤科专用模型在多来源皮肤病变图像上的二分类恶性风险预测性能。

**💡 创新点**

通过在不同模态、合并数据与分布偏移下统一评估，揭示了领域对齐的嵌入模型与大规模VLM的性能差距。

**🔧 技术方法**

采用基于嵌入的特征提取+下游分类器、端到端卷积/视觉变压器、以及多模态VLM（CLIP、MedSigLIP、Gemma‑3、LLaMA等）与多种prompt设计。

**📊 数据集**

使用HAM10000、ISIC2018/2024、PAD‑UFES‑20、HC、DDI、SD‑198等公开数据集，并构造MERGED‑ALL/CLINIC/DERM合并集。

**📈 对比分析**

采用统一70/15/15划分、F1分数对比；嵌入模型平均F1≈88%，CNN/ViT≈86%，VLM≈66%，并在分布偏移、模态变化及类别不平衡策略下验证。

**⚠️ 局限性**

VLM在图像仅输入下缺乏鲁棒性，提示设计与模型规模对临床可靠性提升有限，且单一视觉特征难以捕捉所有恶性病变。

---

## 521. Beyond Single Videos: Benchmarking and Active Evidence Seeking for E-Commerce Cross-Video Reasoning

**arXiv ID:** 2610.03099 | [PDF](https://arxiv.org/pdf/2610.03099v1)

**作者:** Jinghan Zhao `[一作]` (Alibaba Group), Bo Zheng `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了AdsCVR跨视频电商推理基准并设计了AdSeek主动证据获取框架，利用多轮工具调用实现跨视频细粒度比较；

**💡 创新点**

创新点包括：①构建电商跨视频推理基准AdsCVR；②提出主动证据获取Agent框架AdSeek，能根据比较目标动态选择视频、时段、模态和视觉区域；③设计Rectified Bootstrapping Pipeline，结合在线强化学习、离线轨迹纠正与SFT，提升证据收集与推理质量；

**🔧 技术方法**

技术手段主要是多模态语言模型+工具调用（视觉/语音），多轮推理策略、强化学习（GRPO）与离线轨迹纠正、监督微调（SFT）以及教师模型辅助诊断；

**📊 数据集**

使用的数据集为AdsCVR（2483视频、6110问答），并在MVU‑Eval、CrossVid上进行额外训练和跨域评估；

**📈 对比分析**

与传统单视频模型（Qwen3‑VL‑8B、Qwen3.8‑27B、Gemini、GPT‑5.5）以及先前AgentCVR进行对比，AdSeek在AdsCVR测试集上达74.30%准确率，比基线提升27.90个百分点；在CrossVid上平均准确率31.55%，相较基线提升约3个百分点；

**⚠️ 局限性**

局限性在于多轮推理和工具调用导致推理延迟较高，强化学习与离线纠正过程训练成本高，适用于对推理质量要求高但对时延不敏感的场景。

---

## 522. Predicting Steering Vectors and Adapter Weights for Few-Shot Author-Style Transfer

**arXiv ID:** 2610.03163 | [PDF](https://arxiv.org/pdf/2610.03163v1)

**作者:** Leonard Popp `[一作]` (Karlsruhe Institute of Technology), Jan Niehues `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在冷启动环境下，仅利用三篇摘要对大型语言模型进行作者级风格控制，并提出三种基于对比激活调节、预测向量和超网络预测 LoRA 适配器的方法。

**💡 创新点**

①通过在相同内容的“中性”生成与真实摘要对比，构造作者级对比向量；②证明手工与预测向量近正交，说明风格方向不唯一；③在保持质量的同时实现更优的风格与质量折中。

**🔧 技术方法**

使用 Qwen3‑4B‑Instruct 作为冻结基础模型，STAR 编码器提取风格特征；对比激活调节、MLP 预测向量、四层 MLP + 超网络生成 LoRA 权重；采用 k‑NN 作者分类器、主题保持判别器和人工对比评估。

**📊 数据集**

以 2022 年前的 arXiv 论文为基准，构建 1,232 篇论文（203 位作者）用于训练与验证，1,312 篇论文（328 位作者）用于未见作者测试；每篇论文包含大纲、正文、原始摘要与三篇同作者摘要。

**📈 对比分析**

与零样本、少样本提示和全模型 LoRA 微调三种基线对比。结果显示：零样本预估风格 31%/M R=0.40，LoRA 49.8%/M R=0.55，手动对比 34.5%/0.42，预测向量 35.0%/0.43，超网络 LoRA 36.5%/0.46；超网络在保持 91% 以上偏好率的同时达到 36% 风格准确率，处于风格与质量折中 Pareto 前沿。

**⚠️ 局限性**

风格准确率与内容高度耦合，无法单独衡量风格；缺乏对比实验验证是否仅提升了“人类写作”倾向；仅在单一模型、单一领域、单一语言上验证；层选择未做完整搜索，可能影响性能；未评估对不同作者间向量相似度的内在差异。

---

## 523. EvoRiskBench: An Evolving Benchmark for Runtime Security Risks in Workspace Agents

**arXiv ID:** 2610.03153 | [PDF](https://arxiv.org/pdf/2610.03153v1)

**作者:** Shiyi Kuang `[一作]` (Novo Ordo For Ai), Ping Chen `[通讯]` (Fudan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了可执行的工作空间代理安全基准EvoRiskBench，覆盖9个风险入口点与5个技术效果，并通过自动化工作流生成450个可执行对抗任务。

**💡 创新点**

创新点在于将风险以EP-Path-EF框架结构化，结合自动化生成、回放与迭代改进的执行反馈循环，使基准可随模型、工具和威胁演进而持续扩展。

**🔧 技术方法**

使用的技术包括大型语言模型、三种不同的执行框架（Codex、Claude Code、OpenClaw）、ETW系统事件采集、Docker化隔离环境和AI判定的轨迹伤害评分。

**📊 数据集**

使用了450个由自动化流程构建的对抗任务集合，覆盖六种场景；数据集主要由人工设计的恶意载荷和工作空间资源组成。

**📈 对比分析**

通过在九种模型×执行框架组合上执行450个案例，报告了攻击成功率37.46%、任务成功率60.44%，并展示了模型对攻击成功率的显著影响。

**⚠️ 局限性**

局限性包括仅关注间接提示注入（IPI）攻击，使用的是合成环境和有限的工具集，无法完全覆盖生产级工作空间的复杂性与长期运行场景。

---

## 524. Budgeted-GS: Real-Time Large-Scale Gaussian Splatting via Factoring LOD

**arXiv ID:** 2610.03162 | [PDF](https://arxiv.org/pdf/2610.03162v1)

**作者:** Haipeng Wang `[一作]` `[通讯]` (Neusoft), Haipeng Wang (Neusoft)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种基于容量地板（capacity floor）的 3D 高斯渲染管线，先用理论确定每个场景在给定质量目标下所需的最小高斯数目，然后在此预算下直接训练（Budget‑Centered Training，BCT）或对已训练的模型构造多分辨率分层（Factoring LOD），从而实现城市规模场景在消费级 GPU 上实时高质量渲染。

**💡 创新点**

创新点包括：
• 将渲染视作相位空间中的最优传输问题，利用 Zador 定律给出可实现误差下界（容量地板）。
• 通过 3D Kakeya 集 conjecture 推导的“collapse license”，证明哪些高斯可以合并或丢弃而误差可控。
• 设计基于像素足迹阈值的单参数 scheduler，动态按视角选择合适分辨率的节点，实现连续的内存‑质量曲线。
• 结合容量地板直接在训练阶段确定预算，实现一次性训练就得到合适规模模型，避免后期剪枝浪费。

**🔧 技术方法**

核心技术：
• 相位空间最优传输理论与高分辨率量化（Zador law）。
• 体素/高斯等价聚合（moment‑matching aggregates）。
• GPU 级八叉树 + 规模带（scale band）划分 + Morton 排序。
• 像素足迹调度（pixel‑footprint scheduler）。
• MCMC 密集化 + 梯度训练 + 预算驱动的稀疏化。

**📊 数据集**

使用的数据集：
• 13 个公开室内/室外场景（garden、room、bonsai、counter、flowers、kitchen 等）。
• MatrixCity Aerial（官方城市规模数据集，Block_all）。
• 公开的 Mip‑NeRF‑360 等合成数据用于验证。

**📈 对比分析**

对比方法：
• 传统压缩方法（Compact3DGS、EAGLES、LightGaussian、PUP、RAP 等）。
• 训练‑耦合 LOD 方法（Octree‑GS、CityGaussianV2）。
结果：
• BCT 在所有测试预算下均优于传统 train‑then‑prune，PSNR 提升 0.2–1.9 dB；
• Factoring LOD 在 RTX 4080 SUPER 上实现 1920×1080 全 SH 实时渲染，FPS 57–76，内存 0.4–1.3 GB；
• 相比 Octree‑GS，FPS 提升 6×、显存占用减少 6.9×。

**⚠️ 局限性**

局限性：
• 评估仅在单一 GPU（RTX 4080 SUPER）上完成，未验证跨硬件的可移植性。
• 只覆盖合成渲染（compositing）场景，对透明、折射、散射等效果支持不足。
• 目前不支持内存流式（out‑of‑core）渲染，超大场景仍需完整装载。
• 理论假设（可见性指数、相位空间维度等）仍有待进一步验证。
}

---

## 525. In-Distribution Forcing for Long Video Generation at Test Time

**arXiv ID:** 2610.03120 | [PDF](https://arxiv.org/pdf/2610.03120v1)

**作者:** Jeongwoo Shin `[一作]` (Seoul National University), Jaemoo Choi `[通讯]` (Georgia Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种测试时的In-Distribution Forcing（ID-Forcing）框架，通过自缓存（self-caching）和保持首块（sink）来保证在视频扩展到训练时限之外时，KV缓存与KV条件始终处于训练分布内，从而显著抑制颜色、纹理及运动的漂移，支持分钟级长视频生成。

**💡 创新点**

创新点在于首次将KV provenance（缓存的上下文来源）视为漂移根源，并提出两级KV管理策略：①在KV缓存层使用自缓存，使每个KV条目仅在训练时出现过的上下文中生成；②在KV条件层将首块固定为sink并进行重旋转，形成严格等价于训练窗口的滚动窗口，彻底消除训练外的上下文诱发漂移。

**🔧 技术方法**

使用技术包括：自缓存机制（每个KV条目仅参考自身或训练窗口中已有条目）、首块sink与重旋转（R_-f）保持窗口内相对时间位置一致、旋转位置嵌入（RoPE）、以及现有的AR视频扩散模型（Self‑Forcing、LongLive）作为基线；整个方法不需要额外训练，仅在推理时修改KV操作。

**📊 数据集**

使用数据集：MovieGen 提示集（前128个提示，使用 Qwen2.5-7B-Instruct 细化）生成视频，评估数据来源于 VBench‑Long 基准；视频分辨率 480×832，长度分别为 120 s 与 240 s。

**📈 对比分析**

与 Deep Forcing、∞‑RoPE、MemRoPE、Self‑Forcing、LongLive 等基线在 VBench‑Long 指标、颜色漂移、运动漂移、动态度（Dynamic Degree）以及用户研究（2AFC）进行对比。ID‑Forcing 在所有漂移相关指标上均优于基线，且在美学质量、成像质量、主题一致性等指标保持竞争力或更优，用户研究显示在六个评价维度均被显著偏好。

**⚠️ 局限性**

局限性包括：①仍需手动设定滚动窗口长度 L 与自缓存长度 ℓ，超长视频或极端场景下可能需要进一步调优；②方法主要在潜在空间内验证，实际像素级细节在极高分辨率或复杂动态场景下可能出现细微漂移；③虽然是训练无关的推理策略，但对原模型的KV结构和RoPE实现要求较高，迁移到其他架构需做额外适配。

---

## 526. Coverage You Can Steer: Online Conformal Calibration for RL-Driven Hardware-Aware NAS

**arXiv ID:** 2610.03127 | [PDF](https://arxiv.org/pdf/2610.03127v1)

**作者:** Pedro Brandimarte `[一作]` (Vicomtech Foundation, Basque Research and Technology Alliance), Oihana Otaegui `[通讯]` (Vicomtech Foundation, Basque Research and Technology Alliance)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了在硬件感知神经架构搜索中使用在线自适应合成推断（ACI）对候选评估进行过滤，并将校准后的上界用作探索信号，从而在强化学习驱动的搜索中实现可调控的覆盖率并显著减少评估成本。

**💡 创新点**

创新点包括：① 用ACI替代静态分割合成推断，提供对非平稳候选流的分布自由长期覆盖；② 引入无调参的专家聚合（AgACI）、归一化以及Mondrian分组校准，实现局部和分组级别的自适应校准；③ 将校准上界转化为分布自由的GP‑UCB等价探索函数，用于贪婪和树搜索的直接搜索。

**🔧 技术方法**

采用多智能体PPO控制器、MLP预测器、分割与自适应合成推断、专家聚合、归一化与Mondrian分组校准、蒙特卡洛树搜索（AlphaZero式）以及Gaussian过程基准，并使用分析式硬件成本模型和部分训练代理。

**📊 数据集**

使用MNIST、CIFAR‑10和CIFAR‑100数据集，分别对应LeNet‑、ResNet‑和MobileNet‑风格的CNN搜索空间。

**📈 对比分析**

与静态分割合成推断、周期性再校准、GP‑UCB、随机搜索、进化和PPO等基线比较；ACI在三大搜索空间均实现覆盖率与请求值1e‑3精度、种子方差≈0.002，评估节省25–50%且无解质量损失；在约束测试床上基于校准上界的贪婪/树搜索在1000次评估内平均提升≈0.04准确率，方差显著降低。

**⚠️ 局限性**

局限性包括：① 仅验证单一MCU‑风格硬件成本模型，未在真实设备上测评；② 评估依赖部分训练代理，代理的真实性与最终模型性能的关联尚未验证；③ Mondrian校准在深层/数据稀疏组上受限；④ 仅在有限的实验室环境和少数种子下验证搜索效率与可扩展性；⑤ 覆盖率测量在被剪枝候选中受限，需要进一步处理选择偏差。

---

## 527. SPEAR: A Spectral-Disentangled MoE Neural Operator with Knowledge-Guided Expert Aggregation for Large-Scale PDE Pretraining

**arXiv ID:** 2610.03265 | [PDF](https://arxiv.org/pdf/2610.03265v1)

**作者:** Dengdi Sun `[一作]` (Anhui University), Bin Luo `[通讯]` (Anhui University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 SPEAR，一种在大规模 PDE 预训练中使用频谱解耦的 MoE 神经算子，并通过知识引导的专家聚合来减少专家冗余。

**💡 创新点**

创新点在于：① 通过低频共享分支和高频专用 MoE 分离共享动力学与 PDE 专属细节；② 采用低秩 LoRA 适配器提升专家效率；③ 用数据集特定知识与路由偏好构造专家相似度矩阵，实现语义化的专家聚合。

**🔧 技术方法**

使用技术包括：频谱解耦的 AFNO、深度可分离卷积、Top‑2 动态路由、LoRA 低秩适配器、负载平衡与正交性约束、知识引导的层次聚类。

**📊 数据集**

使用十二个多样化 PDE 数据集（如 PDEBench、PDEArena、CFDBench、Wave‑Gauss、Wave‑Layer、SWE、DR 等）进行预训练、微调与转移实验。

**📈 对比分析**

与现有基线（FNO、UNet、GK‑T、GNOT、OFormer、MPP、DPOT、NESTOR、UniSolver、MoE‑POT 等）对比，SPEAR 在预训练、微调、未见 PDE 推理以及 50% 专家压缩后均保持或提升 L2RE，展示出显著的性能优势。

**⚠️ 局限性**

局限性包括：训练和路由调参复杂，聚合策略依赖于数据集相似度估计，对极端异构或 3D PDE 的泛化尚未充分验证。

---

## 528. Performance Analysis of MR-NOMA with Arbitrary Number of Users and Symbol Rates

**arXiv ID:** 2610.03266 | [PDF](https://arxiv.org/pdf/2610.03266v1)

**作者:** Zainab Khader `[一作]` (Khalifa University), Emad Alsusa `[通讯]` (University of Manchester)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了多符号率-NOMA在Nakagami-m衰落下的下行BER性能，并给出精确与近似解析表达式。

**💡 创新点**

首次对任意符号时长和用户数的mr-NOMA给出BER解析，揭示符号时长多样性带来的天然干扰消除优势。

**🔧 技术方法**

采用符号级合并、SIC、Q函数积分、Nakagami-m平均、解析组合与数值优化等技术。

**📊 数据集**

利用Monte Carlo仿真（10^6次）验证解析结果，无使用真实数据集。

**📈 对比分析**

与传统单符号率-NOMA对比，mr-NOMA在相同BER下可获得约12 dB的SNR提升，并通过功率分配优化进一步降低平均BER。

**⚠️ 局限性**

对大用户数的非完美SIC解析难以实现，仅能在完美SIC假设下给出近似；高阶调制分析仍待研究。

---

## 529. PaMIR: Open Benchmark of Public Credit-Default Datasets

**arXiv ID:** 2610.03259 | [PDF](https://arxiv.org/pdf/2610.03259v1)

**作者:** Mikhail Liashkov `[一作]` (zypl.ai), Bonu Boboeva `[通讯]` (zypl.ai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文构建了一个公开、可复现的19个信用违约数据集集合，并提供了基于标签延迟的流式评测协议以及传统i.i.d.划分协议。

**💡 创新点**

创新点在于将稀缺且延迟到来的标签情境与严谨的泄漏审计相结合，形成可持续更新的 benchmark，同时引入了针对 synthetic 数据的泄漏控制和评估框架。

**🔧 技术方法**

技术上采用 Python 生态（pandas、scikit-learn、XGBoost、SDV 等）实现数据处理、模型训练与评测；评测指标为 ROC‑AUC 与 Gini；流式协议通过预设延迟、refit 触发器实现。

**📊 数据集**

使用的 19 个公开数据集覆盖消费者、P2P、车辆、企业等 7 类产品，来自 9 个国家，合计约 124 万行样本，默认率范围 3–41%。

**📈 对比分析**

在 i.i.d. 划分下，GBDT 比逻辑回归提升约 0.06 的平均 AUC；在流式评测中，GBDT 与逻辑回归的差距缩小到约 0.04，整体平均 AUC 约 0.72；在标签稀缺阶段（<100 标签）两者相近，随后 GBDT 超越。

**⚠️ 局限性**

主要局限包括：缺乏真实时间顺序导致无法模拟人口漂移；批量评分可能产生泄漏；数据质量参差不齐；synthetic 模块仅在交叉验证下验证，未在流式协议下测试；仅提供两种未调参基准。

---

## 530. Moving Forward with Video Saliency: A New Dataset and Benchmark where Motion Matters

**arXiv ID:** 2610.03276 | [PDF](https://arxiv.org/pdf/2610.03276v1)

**作者:** Susmit Agrawal `[一作]` (Tübingen AI Center, University of Tübingen), Matthias Kümmerer `[通讯]` (Tübingen AI Center, University of Tübingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了新的视频注意力基准 SalTempto，并对比静态与时序模型的性能，进一步验证了 LEDOV 诊断的有效性。

**💡 创新点**

创新点在于：①构建了更具时序动态的测试集；②用更完善的 gold‑standard（4 组分混合）评估信息收益；③发现并系统化描述了三种模型常忽视的人类视觉行为（遮挡下的对象持久性、场景惯性、预测性扫视）。

**🔧 技术方法**

采用静态基线 DeepGaze MR、12 种时序模型（SalFoM、TMFI-Net、ViNet‑S/A/E、TASED‑Net、TinyHD‑single/multi、UniformerSal‑Spatial/SpatioTemporal、UNISAL‑Spatial/SpatioTemporal），并使用信息理论指标（LL、IG、IGE 等）进行评估。

**📊 数据集**

使用 LEDOV（1.52 h）和新基准 SalTempto（224 min，总计 3.4 h 训练 + 0.17 h 验证/测试）的视频与眼动数据，SalTempto 选自 HACS‑Segments，包含 1 min 动态剪辑、最多 16 名受试者眼动记录。

**📈 对比分析**

评估方法：先在预训练检查点下做零拷贝评估，再在各自数据集上 fine‑tune；比较中心偏差、金标上限和模型性能。结果显示：在 LEDOV 上静态基线可恢复 >50% 可解释信息；在 SalTempto 上仅恢复 ~13%；最强时序模型在 SalTempto 上提升约 15–20% IG，仍剩约 50% headroom 未解释。

**⚠️ 局限性**

限制：SalTempto 来源于动作数据集，主要捕捉前景动作动态；缺乏无演员驱动的场景变化；验证/测试视频仅 1 min 长度，可能不足以覆盖极端时序挑战。

---

## 531. DexJoCo-X: Benchmarking Action Representations for Multi-Hand Dexterous Manipulation

**arXiv ID:** 2610.03278 | [PDF](https://arxiv.org/pdf/2610.03278v1)

**作者:** Xiangwei Jiang `[一作]` (University of Electronic Science and Technology of China), Wen Li `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DexJoCo-X 基准与工具箱，统一 7 种手、6 种任务和 2100 条演示，用于跨体现学的可比评估。

**💡 创新点**

创新点在于：① 统一的学习接口和控制框架，② 多手、多任务平衡数据集，③ 通过统一动作空间和跨体现预训练来比较不同动作表示方法。

**🔧 技术方法**

使用的方法包括：π_0.5 (Ego-Pi) 与 Being‑H0.5 端到端策略训练，FAAS 功能对齐槽位，DexLatent 共享编码解码器，以及混合流 (Mixture‑of‑Flow) 结构。

**📊 数据集**

数据集为 2100 条人类演示，覆盖 7 把手（XHand、Inspire、Wuji、LEAP、Sharpa Wave、LinkerHand、Allegro）与 6 个单臂/双臂任务，演示通过手套+Vive 采集并自动场景扩展。

**📈 对比分析**

通过在相同演示、视角、网络结构、训练预算下对 Native、FAAS、DexLatent 三种动作表示进行比较，结果显示 Native 与 FAAS 在整体成功率（约 47‑48%）上相近，FAAS 在双臂任务上略优，Native 在单臂任务上略优，DexLatent 效果最差（约 33%）。

**⚠️ 局限性**

局限性包括：双臂协调仍表现低下，仅在仿真环境验证，未测试对未知手的零样本或少样本迁移，且任务覆盖仍有限。

---

## 532. HexVIO: Towards All-Day Stereo-Inertial Tracking Through Commodity DSPs

**arXiv ID:** 2610.03283 | [PDF](https://arxiv.org/pdf/2610.03283v1)

**作者:** Patrick Wolf `[一作]` (Technical University of Munich), Daniel Cremers `[通讯]` (Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将Basalt的视觉前端移植至Snapdragon 8 Gen 2手机中的Hexagon V73 DSP，并通过一次合并的FastRPC调用实现实时立体视觉‑惯性里程计（VIO）前端；

**💡 创新点**

首次在商用设备上实现多摄像头立体‑惯性VIO的DSP加速，提出了聚合RPC、专为Hexagon设计的像素/特征并行向量化、VTCM层级存取以及基于FastRPC的低功耗调度策略；

**🔧 技术方法**

使用Hexagon V73 DSP的HVX向量指令、VTCM（8 MB）缓存、FastRPC远程过程调用、基于Basalt的KLT/FAST算法、FastCV框架以及ARM Cortex‑A15/18主机与DSP协同；

**📊 数据集**

评估数据集包括EuRoC（MAV）、Monado SLAM（VR头显）、TUM‑VI（手持机）以及自制的XREAL AIR 2 Ultra（XR眼镜）四个系列，总计八个序列；

**📈 对比分析**

与CPU端Basalt基线进行对比：DSP实现的前端吞吐量提升1.6–2.4×，功耗降低约67%，同时保持相同或更低的ATE/RTE；实时性方面，DSP配置在30 fps下功耗仅0.83 W，理论可持续约18 h；

**⚠️ 局限性**

局限性：后端优化器仍在CPU上，导致后端占比高；仅在支持Hexagon V73的Snapdragon平台可用；极端高帧率或热限制时仍可能触发CPU频率下降；未来需改进后端协同、利用ISP直接输入、支持更多SoC。

---

## 533. Cross-cohort TB classification using clinical data gathered in Uganda and South Africa

**arXiv ID:** 2610.03256 | [PDF](https://arxiv.org/pdf/2610.03256v1)

**作者:** Joshua M. Jansen van Vüren `[一作]` (University of Stellenbosch), Thomas R. Niesler `[通讯]` (University of Stellenbosch)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了基于临床与人口统计数据的 TB 预筛选，使用逻辑回归、MLP 与 CNN 在乌干达和南非数据集上进行交叉国域验证。

**💡 创新点**

创新点在于首次跨国域评估深度学习 TB 分类器的鲁棒性，并提出联合特征选择与排序的 CNN 优化策略。

**🔧 技术方法**

采用了逻辑回归、两层 MLP 与一维 CNN，结合网格搜索、Sequential Feature Selection（SFS）和特征排序。

**📊 数据集**

使用 CAGE-TB 数据集，包含 28 个临床与人口统计特征，分别来自乌干达和南非社区诊所的 721 名受试者。

**📈 对比分析**

通过十折交叉验证与全外部测试，LR+SFS 在两国都实现 AUROC 0.84‑0.86，SFS 提升 2‑7%，但 MLP/CNN 在外部测试表现不稳定，LR 接近 WHO 目标。

**⚠️ 局限性**

限制在样本量有限、特征缺失率高、缺乏实验室指标，且深度模型在不同域的泛化仍受限。

---

## 534. Prompt framing governs LLM default following in collective-action

**arXiv ID:** 2610.03253 | [PDF](https://arxiv.org/pdf/2610.03253v1)

**作者:** Eladio Montero-Porras `[一作]` (Universite Libre De Bruxelles), Tom Lenaerts `[通讯]` (Universite Libre De Bruxelles)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过在两种一轮社会博弈（公共池资源提取与阈值公共物品贡献）中为大型语言模型预填默认值，探究默认值对模型输出分布的影响。

**💡 创新点**

创新点在于：①系统评估默认值对LLM行为的拉力（default pull）并揭示其依赖于措辞、动作空间粒度及默认与模型基准偏好冲突与否；②证明相同模型在不同措辞下可从低拉力变为高拉力，显示默认效果并非模型固有属性；③提出净拉力（net_pull）与Wasserstein距离两种量化指标，用以衡量预填默认值对完整概率分布的改变。

**🔧 技术方法**

使用的技术包括：①基于token‑logprob的概率分布提取，②构造四种默认措辞（中性、明确、许可、指令式），③细粒度（0–30）与粗粒度（0、15、30）动作空间的对比，④统计分析（均值、95% CI、Spearman相关），以及水平特距检验（w_pull）。

**📊 数据集**

实验数据来源为七款公开模型（GPT‑4、GPT‑3.5、Llama‑3、Gemma‑4、Qwen‑3.5 等）在统一系统提示下的单轮博弈回应，涵盖全部默认值、措辞和动作空间组合。

**📈 对比分析**

通过对比不同模型、默认值和措辞的净拉力，作者发现：在CPR游戏中默认拉力普遍为正且最高于Nash均衡值；在TPG游戏中拉力呈二分布，部分模型高度敏感。相比粗粒度动作空间，细粒度更易被默认拉动；许可式措辞显著抑制拉力，而指令式措辞在CPR中放大拉力。整体上，默认拉力在不同条件下差异显著，表明模型对默认的响应高度可变。

**⚠️ 局限性**

主要局限包括：①仅使用七款支持token‑logprob的模型，未覆盖如Claude、Gemini等主流模型；②实验仅为一次性单轮决策，无法评估默认效应在多轮互动中的衰退或强化；③未探究模型内部机制（如锚定、指令遵循）导致的默认拉力；④部分模型在冲突/一致细胞下的拉力未能实现清晰对比。

---

## 535. Self-Repairing Recurrent Ensembles for Real-Time Recovery from Distribution Shift

**arXiv ID:** 2610.03249 | [PDF](https://arxiv.org/pdf/2610.03249v1)

**作者:** Julian Lemmel `[一作]` (Vienna University of Technology), Radu Grosu `[通讯]` (Vienna University of Technology)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种在线自监督的递归集成控制器，能够在部署时面对传感器漂移、失效或噪声增大时自我修复。

**💡 创新点**

创新点在于使用随机遮蔽观测形成多样化集成，利用Kalman增益融合生成自监督标签，并通过RFLO实现每步即时梯度更新，实现无监督的分布偏移恢复。

**🔧 技术方法**

技术包括递归神经网络（CT‑RNN或LRU）集成、随机遮蔽观测、Kalman增益融合、Follow‑The‑Leader自监督损失、RFLO在线梯度计算以及与EnsembleDAgger的对比。

**📊 数据集**

使用的是Mujoco仿真环境的连续控制任务（Ant、HalfCheetah、Humanoid），先用PPO训练专家策略后用于预训练。

**📈 对比分析**

与EnsembleDAgger、BPTT、CT‑RNN等基线比较，实验显示在传感器漂移或失效时，随机遮蔽集成+RTR‑FTL 能恢复到接近预训练水平，而全观测集成无法恢复；在三种任务上均表现出优于基线的鲁棒性。

**⚠️ 局限性**

局限包括未验证对多传感器同时失效、与动作相关的偏移的处理；实验仅在仿真中进行，缺乏真实平台的验证；并未探究不同遮蔽比例或融合顺序的最优选择。

---

## 536. The Effects of Air-Conditioning and Road-Traffic Noise on Perceived, Cognitive, and EEG Responses in a University Classroom

**arXiv ID:** 2610.03210 | [PDF](https://arxiv.org/pdf/2610.03210v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 537. A Kinetic Theory of the Gated Self-Evolving LLM Agent

**arXiv ID:** 2610.03243 | [PDF](https://arxiv.org/pdf/2610.03243v1)

**作者:** Haipeng Wang `[一作]` `[通讯]` (Neusoft Corporation), Haipeng Wang (Neusoft Corporation)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究自演化LLM代理的插件实例群体，发现其统计上呈现气体动力学特征，并给出对应的动力学方程与理论证明。

**💡 创新点**

创新点在于将插件实例视为相同的硬球，构建了基于门控的Kinetic Theory，推出击打时间证书、分辨率定律与分离定理，并在实验中验证了流体动力学的统计标记。

**🔧 技术方法**

使用技术包括：基于Boltzmann方程的插件动力学主方程、门控自演化机制（validation gate）、DeepSeek Harness（DSH）平台、MiniMax‑M3 LLM reflector、WebShop 训练环境以及 Lean 形式化证明。

**📊 数据集**

数据集为 WebShop 1,000 件商品与 6,910 维合成属性目标的文本环境，以及一个四参数插件库。

**📈 对比分析**

通过预注册探针（P1–P5、formal1–13）与理论预测对比，验证了密度调制的 -1/2 缓变斜率、碰撞通道收益 +0.19 以及 NESS 盘整，表现符合预期。

**⚠️ 局限性**

局限性包括样本量极小（仅 2 个种子）、仅在最小化实例上验证，缺乏大规模生产验证；门控必须与验证集分离，否则证书失效；流体类比受限，无法得到完整的 Navier‑Stokes 等连续介质方程。

---

## 538. Execution-Path Qualification and Realized Costs in Speculative Decoding on Consumer Systems

**arXiv ID:** 2610.03228 | [PDF](https://arxiv.org/pdf/2610.03228v1)

**作者:** Chengzhan Li `[一作]` `[通讯]` (University of Electronic Science and Technology of China), Chengzhan Li (University of Electronic Science and Technology of China)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在 B570、RTX、358H 和 M4 等消费级系统上，对投机解码的执行路径进行资格化、预测和完整成本评估。

**💡 创新点**

创新点在于首次实现直接 ID 路径的资格化与冻结预测，并结合同状态分叉、独立程序重放以及完整成本核算，提出 Guard 动态控制策略。

**🔧 技术方法**

采用了直接 ID 资格化、草稿自由目标预测、独立推理与重放、完整成本计量以及 Guard 控制等技术。

**📊 数据集**

使用的数据集包括 Qwen3‑8B/0.6B Q8_0、HumanEval、Spec‑Bench，以及 UB1/UB128/UB2 等对照集。

**📈 对比分析**

通过冻结的直接 ID 一致性比较、保留推断对比以及不同模式的几何平均速度评估，实验结果显示 RTX 序列化在测量边界内更慢，358H 在零草稿成本下仍达不到平衡，而 M4 Guard 在固定模式下未能击败 Always‑Spec。

**⚠️ 局限性**

局限性包括仅针对特定模型、贪婪截断输出、单一消费级堆栈，缺乏硬件通用性、能耗评估及多样化任务的泛化。

---

## 539. Uncertainty as a Proxy for Semantic Correctness in Diffusion-Based Medical Image Synthesis

**arXiv ID:** 2610.03224 | [PDF](https://arxiv.org/pdf/2610.03224v1)

**作者:** Yuxuan Ou `[一作]` (University of Oxford), Vicente Grau `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究了基于扩散模型的非对比CT到对比增强CT的合成，并提出以多尺度不确定性与生成图像中的血管分割作为语义正确性量化的评估框架。

**💡 创新点**

创新点在于将多任务扩散模型的血管分割作为语义正确性参考，系统评估六类不确定性方法在像素、区域、图像尺度上的表现，并验证其在外部多中心分布漂移和临床相关的OOD（血管支架）样本上的鲁棒性。

**🔧 技术方法**

采用AortaDiff多任务扩散模型、MC Dropout、TTA、RDS、Ensemble、BayesDiff、HyperDiff等不确定性量化技术，并使用DDIM采样、SLIC分割、Dice/AUROC等评价指标。

**📊 数据集**

使用Oxford Abdominal Aortic Aneurysm（OxAAA）内部数据集进行训练和内部评估，使用AICT多中心外部数据集进行分布漂移验证和OOD支架检测。

**📈 对比分析**

在像素、区域和图像尺度上通过AUSE、AUSC、AUROC等指标比较六种方法，MC Dropout在所有尺度及外部数据集上均名列前茅，能够无额外训练成本提供可靠的不确定性；其他方法表现不一，BayesDiff最弱。

**⚠️ 局限性**

研究仅在腹主动脉NCCT→CECT单一任务上验证，尚未检验到其他器官、模态和生成任务；图像级排序和对高质量图像细微差异的判别仍有限。

---

## 540. Contextual Flow Matching: Adaptive Step Selection in Flow Models for Efficient Visual Generation

**arXiv ID:** 2610.03202 | [PDF](https://arxiv.org/pdf/2610.03202v1)

**作者:** Divya Jyoti Bajpai `[一作]` (Indian Institute of Technology Bombay), Manjesh Kumar Hanawal `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Contextual Flow Matching，利用在线神经上下文带队方法在流匹配模型中自适应选择步数，从而在不重训练模型的前提下加速图像、图像编辑和视频生成。

**💡 创新点**

将步数预算选择视为神经上下文带队问题，使用输入上下文特征和无监督速度变异奖励在线学习步数，无需预训练策略或额外数据；同时给出前向欧拉误差 O(1/K) 的理论分析。

**🔧 技术方法**

采用神经上下文带队（Neural UCB）+ 轻量化上下文特征（语义嵌入+手工特征）+ 无监督速度变异奖励 + 经验回放更新 + 前向欧拉误差分析。

**📊 数据集**

使用 GenEval（文本→图像）、GEdit（图像编辑）和 VBench（文本→视频）等公开基准数据集进行实验。

**📈 对比分析**

与全步、TeaCache、InstaFlow、PeRFlow、FlowCast、AdaDiff、FastFlow 等基线比较；在 BAGEL/FLUX 上实现约 2.5× 的加速（NFE 从 50 降至 ≈18），同时保持 0.77/0.65 的质量得分；视频和编辑任务亦保持竞争性质量。

**⚠️ 局限性**

需要初始探索阶段；性能依赖上下文特征的表达能力，且对极其复杂提示的适配仍有限；目前仅适用于轻量级特征，可能需要更丰富特征以进一步提升自适应效果。

---

## 541. Beyond Reward Hacking: Proxy Divergence Across Four Layers of a Staged Humanoid Learning Pipeline

**arXiv ID:** 2610.03196 | [PDF](https://arxiv.org/pdf/2610.03196v1)

**作者:** Arunabh Bora `[一作]` `[通讯]`, Arunabh Bora

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了强化学习管道中奖励、课程门、评估统计和参考运动等四类代理的失配问题，并在仿真人形机器人上提出并验证了相应的闭合改进方案。

**💡 创新点**

创新点在于将所有代理统一为同类失配概念，针对课程门、评估统计与参考运动分别给出正式的闭合改造（峰值误差门、可达性检查、阶段信息日志、无同步评估、静态可行性检查与残差前馈），并结合无额外训练成本的函数保持输入扩展和分割权限技术，实现了单一策略在多阶段连续成长。

**🔧 技术方法**

采用 PPO + GAE 的 Gaussian MLP 策略网络，配合 ELU 激活；技术细节包括：L1 轨迹追踪成本、峰值误差门与接触级结果、开放式参考可行性检查、残差前馈、函数保持输入扩展、观测层外部运动驱动；所有改进均在训练流程中直接嵌入，无需额外损失或额外网络。

**📊 数据集**

使用仿真生成的地形与随机冲击数据；机器人为 1.91 m、69.35 kg 的全尺寸人形机器人，单 GPU（RTX 5070 Ti）上训练 13,500 次迭代，未使用任何公开数据集。

**📈 对比分析**

通过在相同硬件与训练设置下对比传统管道与改进管道，评估指标包括基准漂移（0.01–0.04 m/s 下降）、跌倒率、单腿支撑成功率等；改进方案在不增加训练时间或额外成本的前提下显著提升了性能，并消除了奖励外代理的失配。

**⚠️ 局限性**

限制主要在于：实验仅在仿真环境中完成，单一训练种子，动力学模型基于模拟而非真实硬件；缺乏跨任务与真实机器人迁移的验证；对不同机器人平台的泛化仍待进一步研究。

---

## 542. Source Preference in the Wild: How LLM Agents Favor Items by Source, and How to Reduce It

**arXiv ID:** 2610.03195 | [PDF](https://arxiv.org/pdf/2610.03195v1)

**作者:** Jonghyun Song `[一作]` (Seoul National University), Yohan Jo `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型代理在端到端搜索与选择任务中，对不同来源（网站、服务）的偏好及其对最终选择的影响。

**💡 创新点**

①证明源偏好即使在满足相同需求的项目间也存在并可压倒更优项目；②用实验验证源标识本身能驱动选择；③揭示训练阶段源与满足度的关联能形成或消除偏好；④提供通过补全缺失信息或系统提示来减弱偏好的方法。

**🔧 技术方法**

匹配需求的项目对比、循环 Latin 方阵控制位置、Bradley–Terry 评分模型（带 Davidson 误差处理）、聚类自举检验、DPO（Direct Preference Optimization）训练、系统提示干预等。

**📊 数据集**

WebShop、HotelQuEST、ScholarGym三大查询集；通过网络检索得到真实商品、酒店和学术条目；人工与LLM评估需求满足度以验证模型判断。

**📈 对比分析**

与传统基于文本内容或评分的选择方法相比，本文的偏好测量能量化源偏好对结果的影响，显示偏好分数常在±10%至±20%之间，逆序率（偏好源取代更优项目）高达70%，证明偏好对代理决策影响显著。

**⚠️ 局限性**

实验范围受限于预先定义的来源列表和受控的请求集；缺乏对真实用户行为的因果验证；DPO训练仅在人工生成的“虚假”来源对上测试，可能与实际应用差异；对缺失信息补全与提示干预的效果仅在实验设置中体现，尚需进一步验证。

---

## 543. Closing the Prediction Gap: Completing Machine Shape So That Predicted Time, Power, Energy, and Mapping Match What Real Hardware Does

**arXiv ID:** 2610.03197 | [PDF](https://arxiv.org/pdf/2610.03197v1)

**作者:** Lenore Mullin `[一作]`, Gaetan Hains `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文在前人机器形状模型基础上扩展了五个新字段（总内存、功耗、可达执行单元、编译指令配置、内存预留开销），并在五台真实生产设备上直接测量验证，消除了预测误差。

**💡 创新点**

创新点在于将低精度硬件使用、功耗变化与实现细节形式化为闭式预测项，并提出两步可接受性判据，区分确定性与启发式编译器参数，从而实现可复现的跨设备性能预测。

**🔧 技术方法**

技术手段包括数学数组理论（MoA）的 DNF/ONF 变换、ψ/γ 选择算子，以及硬件监控接口（Level‑Zero、Nsight、VTune、ROCm‑SMI）对功耗、内存占用与执行单元利用率进行测量。

**📊 数据集**

数据集为注意力网络的在线 softmax 核，采用 Q/K/V/Out 四个张量，规模为 n=8192、B=1、D=64 的查询集。

**📈 对比分析**

比较方法是将扩展模型的时间、功耗和内存阈值预测与实际测量值对比，误差率降至 1% 以下；在 A100 上 fp16 加速 14.7×、功耗提升 36.3%，并验证其他设备的无加速/无功耗差异。

**⚠️ 局限性**

局限性在于模型仍需预先确定可达性阈值（Boolean 或连续），仅覆盖单核实现；未考虑多核调度、驱动版本差异导致的运行时开销，以及对更大规模数据和不同算子组合的泛化性。

---

## 544. COSMI: COmpositional Synthesis of Multi-object Interactions

**arXiv ID:** 2610.03252 | [PDF](https://arxiv.org/pdf/2610.03252v1)

**作者:** Daniel Eskandar `[一作]` (University of Tübingen), Gerard Pons-Moll `[通讯]` (University of Tübingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构造了一套可组合的多物体人机交互(HOI)数据集，并提出了一个可变物体数的文本条件扩散变换器来生成3D人体与物体交互动画。

**💡 创新点**

创新点在于将局部单物体交互片段通过语义门控与几何校验进行组合，从而以组合次数而非录制时长实现数据量指数级增长；模型采用共享权重的物体槽设计，使参数与物体数量无关，并实现端到端生成。

**🔧 技术方法**

使用CLIP文本编码、Hierarchical Point‑Set几何编码、Transformer denoiser、以及接触一致性指导的扩散过程；通过物体槽、驱动关节预测和多项式损失实现人体与物体的联合生成。

**📊 数据集**

数据来源于GRAB、BEHAVE、InterCap、OMOMO、ARCTIC、AMASS等公开全身交互数据集，通过规则提取、语言模型标注、镜像、组合得到222k序列、275小时，最大可达五物体。

**📈 对比分析**

在自建的测试基准上与HIMO、MDM、PriorMDM对比，表现为最高FID、R‑precision、MM‑Dist、最小物体穿透和接触准确率；在未见物体与未见组合的泛化测试中同样保持领先。

**⚠️ 局限性**

局限性包括仅覆盖持续单交互或并行交互，无法处理多阶段任务；手部运动精度仍低于专用抓取模型；文本提示仅指定动作与物体，缺乏路径与时序细节控制。

---

## 545. Shrome at Touché: Soft-Vote Ensembling and Counter-Causal Augmentation for Causality Extraction

**arXiv ID:** 2610.03268 | [PDF](https://arxiv.org/pdf/2610.03268v1)

**作者:** Roham Zendehdel Nobari `[一作]` (University of Zurich), Shayan Sooratgar `[通讯]` (University of Zurich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一套用于 Touché 2026 Causal Extraction 任务的系统，覆盖因果检测、因果跨度提取与极性分类。

**💡 创新点**

创新点包括：基于三层 BILOU+CRF 的发射级 soft‑vote 集成、利用大语言模型按九种对因果表述模式生成的对因果数据增强、以及跨任务堆叠规则实现检测与提取的互校。

**🔧 技术方法**

技术手段主要是 RoBERTa‑large + 3‑层 BILOU+CRF 发射软投票、BERT‑base 纯序列分类加实体标记、四种种子平均 softmax、LLM（GPT‑5.4）提示生成、DeBERTa‑v3 NLI 端点验证。

**📊 数据集**

使用的数据集为 Countercausal News Corpus（CCNC），并在提取子任务中加入 CNCv2/RECESS 与 EDA 扩充，极性子任务额外合成 2,433 条对因果训练样本。

**📈 对比分析**

在官方 TIRA 评测中，检测二分类 F1 0.869，提取粗粒度调和 0.728，极性宏平均 0.817，分别在提取和极性子任务中领先所有提交者，检测则位列第三。

**⚠️ 局限性**

局限性包括：模型在开发集上调优导致测试误差较大、对 CCNC 的领域依赖、LLM 生成对因果的语义噪声风险、缺乏跨领域/多语言验证以及对规则堆叠效果缺乏外部检验。

---

## 546. Consecutive Posterior Fusion for Diffusive Recovery of Unobservable Image Structures

**arXiv ID:** 2610.03261 | [PDF](https://arxiv.org/pdf/2610.03261v1)

**作者:** Elena Morotti `[一作]` (University of Bologna), Elena Loli Piccolomini `[通讯]` (University of Bologna)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在 DDNM 后向扩散过程中使用连续后验融合的轻量级推理策略（CPF-DDNM），通过结合前一步测量感知估计来改善不可观测图像结构的恢复。

**💡 创新点**

创新点在于利用 DDNM 的范数/零空间分解，证明连续融合只影响先验驱动的零空间估计，同时在零空间内实现超逼近，且通过局部误差分析给出最优融合系数的理论依据。

**🔧 技术方法**

使用基于扩散模型的后验采样框架（DDNM），并结合共轭梯度最小二乘近似伪逆、线性时间调度的融合系数以及传统的 DDIM 逆向步骤。

**📊 数据集**

在 Mayo Clinic CT 数据集上进行评估，同时针对超分辨率任务使用同一数据集的低分辨率采样。

**📈 对比分析**

与多种基准方法（FBP、SIRT、ℓ₂-TV、DPS、PS+、DiffPIR 等）对比，CPF‑DDNM 在稀疏视角 CT 和低剂量 CT 中平均提升 SSIM、PSNR 并降低 LPIPS，尤其在 60 视角下显著优于现有扩散后验采样器；在超分辨率中相对 DDNM 改善 SSIM，但在噪声场景下与 DiffPIR 的差距取决于评价指标。

**⚠️ 局限性**

主要限制是融合系数调度需要基于验证数据手工设定，且在不同逆问题或测量噪声水平下可能需要重新调优；此外对伪逆的数值近似会引入误差，影响数据一致性。

---

## 547. EmbPASS: Towards Cross-Embodiment Open Panoramic Segmentation

**arXiv ID:** 2610.03248 | [PDF](https://arxiv.org/pdf/2610.03248v1)

**作者:** Pujun Guo `[一作]` (Hunan University), Kailun Yang `[通讯]` (Hunan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出跨平台开放全景语义分割任务并构建 EmbPASS 基准，研发 EPONet 网络以实现异构平台下的统一全景分割。

**💡 创新点**

创新点包括 Relation‑Aware Metric Adapter (RAMA) 用于自适应投影畸变的结构化采样，以及 Content‑Adaptive Semantic Transfer (CAST) 通过动态重组 CLIP 语义特征实现跨视角知识迁移。

**🔧 技术方法**

技术核心是冻结 CLIP ViT‑B/16 视觉编码器，结合 RAMA 与 CAST 的轻量级侧分支，使用可变形卷积与 Randers 度量实现局部几何建模，并通过掩模生成与文本匹配实现开放词汇分割。

**📊 数据集**

数据集包括：COCOStuff‑164k（用于训练）、新构建的 EmbPASS（Vehicle、Drone、Wearable、Quadruped 四个平台，共 1,000 张全景图）以及公开基准 Stanford2D3D 与 DensePASS 用于跨基准评测。

**📈 对比分析**

在 EmbPASS 上对比 SAN、OOOPS 等现有方法，EPONet 以 35.82% mIoU（平台均衡）超过最强基线 1.10%，在 Stanford2D3D 与 DensePASS 上同样保持竞争力，均排在前两名。

**⚠️ 局限性**

局限性主要体现在 Drone 子集上 mIoU 仍显逊色，说明对高纬度投影畸变的适应仍有限；且模型对极端视角变化的鲁棒性还有提升空间。

---

## 548. The Investment Acceleration Principle Revisited by means of a Neural Network

**arXiv ID:** 2610.03282 | [PDF](https://arxiv.org/pdf/2610.03282v1)

**作者:** Guido Fioretti `[一作]` `[通讯]` (University of Stuttgart), Guido Fioretti (University of Stuttgart)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文设计并实现了一种基于Kohonen自组织映射（SOM）的投资加速模型，将企业投资决策抽象为神经元权重，利用Hebbian学习规则动态更新加速系数，从而在技术创新初期捕捉投资波动，并通过仿真验证其在早期复苏阶段的有效性；

**💡 创新点**

创新点在于首次将SOM引入宏观投资建模，把企业的认知与投资决策映射到自组织网络，实现加速系数的自适应更新；并提出将快慢信息分离、分阶段（信息快速、认知慢速）的自组织系统框架，用于解释技术创新与投资之间的相互作用；

**🔧 技术方法**

使用了Kohonen自组织映射、Hebbian学习规则、离散时间差分与积分算子、固定系数与两阶段自组织系统分析、以及基于这些工具的仿真算法；

**📊 数据集**

主要使用人工合成的技术创新序列（正弦波+逐步衰减的高斯噪声），配合假设的消费、资本与雇佣向量，未使用真实宏观经济或POS数据；

**📈 对比分析**

通过10次仿真得到单个企业及聚合投资曲线，并与传统固定系数加速器模型以及两阶段分离模型进行对比；原始SOM模型在技术模式出现后能够从随机振荡转为指数增长，而固定系数模型只能在技术模式已稳定后才出现稳定增长，说明SOM模型在早期复苏阶段具有更好的动态响应；

**⚠️ 局限性**

局限性包括：模型假设过度简化（固定企业数、单一资本品、无储蓄与金融约束）；仅基于人工合成数据，缺乏对真实宏观数据的验证；Hebbian学习规则过于简单，未考虑更复杂的认知或策略学习；信息传播被假设为即时，忽略了实际生产与信息流动的时延；整体模型对外部冲击与不确定性的鲁棒性尚未检验。

---

## 549. Toward SLM-based agentic task-tool intent matching

**arXiv ID:** 2610.03213 | [PDF](https://arxiv.org/pdf/2610.03213v1)

**作者:** Chiara Troiani `[一作]` (Cisco Systems), Marcelo Yannuzzi `[通讯]` (Cisco Systems)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究利用小型语言模型（SLM）作为工具相关性分类器，以实现对工具调用的即时、低延迟监督；

**💡 创新点**

创新点在于将任务基访问控制（TBAC）扩展为意图驱动的TBAC，并通过GEPA、SFT和GRPO三阶段专门化流程提升SLM的工具相关性判断能力；

**🔧 技术方法**

采用Gemma 3（1B、4B）模型，利用GEPA进行提示优化，SFT训练LoRA适配器，GRPO强化学习进一步提升判定精度；

**📊 数据集**

使用构造的跨MCP服务器工具集合数据集，包含352种工具、12个服务器，生成任务-工具对并手工校正的高质量标注集；

**📈 对比分析**

通过对比基线、GEPA、SFT、GRPO四个阶段的指标，Gemma 3 4B在测试集上实现了96.13% E2E准确率、F₁ 96.90%，远超95%操作阈值；Gemma 3 1B达到89.60% E2E准确率，未达标；

**⚠️ 局限性**

局限性包括仅使用合成、LLM生成的任务-工具对、仅评估两种模型规模、未覆盖多样化真实场景和可能的安全隐患。

---

## 550. A Dynamic UPF Fault Recovery Mechanism for Enhanced Resilience in 5G Core Networks

**arXiv ID:** 2610.03245 | [PDF](https://arxiv.org/pdf/2610.03245v1)

**作者:** Shirin Behnaminia `[一作]` (Isfahan University of Technology), Mohammad Reza Heidarpour `[通讯]` (Isfahan University of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在5G核心网络中提出并实现了一种动态的UPF故障检测与恢复机制，通过在SMF层面实现会话上下文的转移，实现在UPF重启后快速恢复用户会话，并加入了基于失败历史的UPF选择逻辑。

**💡 创新点**

创新点包括：① 轻量级的应用层上下文恢复方案，避免了传统冗余方案的资源开销；② 引入失败感知的UPF选择机制，在新UE接入时优先选取故障历史较少的UPF；③ 通过监控脚本实现自动化UPF容器重启与IP地址回收，进一步缩短恢复时间。

**🔧 技术方法**

使用技术：OpenAirInterface 5G核心实现（SMF、UPF、gNB、UE模拟器），Docker与Docker‑Compose容器化，PFCP协议（N4接口）会话管理，Heartbeat与启动通知机制，C++实现的SMF自定义模块，ip分组传输工具iperf与ping。

**📊 数据集**

数据集：无公开真实数据集，实验采用OAI仿真环境产生的TCP、UDP流量以及ICMP ping 测试数据，评估吞吐量、延迟与丢包率。

**📈 对比分析**

比较方法：在相同网络环境下分别测试：① UPF正常运行；② UPF重启导致停机；③ 通过机制恢复后。结果显示：UPF重启停机时间约2 s（容器重启）或14 s（新容器启动），恢复后TCP吞吐量恢复到1.05 Mbps，UDP保持稳定；Ping丢包率仅16.2%，平均RTT 6 ms，恢复期间无显著延迟峰值，表明机制能显著降低服务中断时间并保持低丢包与延迟。

**⚠️ 局限性**

局限性：仅在单一UPF重启场景下验证，未完整实现和测试主动备份（Active‑Standby）模式；缺乏对高并发、多UPF多租户环境下的性能评估；失败感知选择仅依据失败次数与时间，未结合实时负载与网络延迟；实验环境为OAI仿真，缺乏真实运营商网络的多样性与复杂性；对gNB端的隧道重配置支持仍待完善。

---

## 551. Evidence-Guided Repository-Level RTL Repair

**arXiv ID:** 2610.03219 | [PDF](https://arxiv.org/pdf/2610.03219v1)

**作者:** Yuxin Du `[一作]` (City University of Hong Kong), Nan Guan `[通讯]` (City University of Hong Kong)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于证据的框架，自动在 RTL 仓库级别定位并修复缺陷，使用重现的失败运行、波形引导定位和一致性验证。

**💡 创新点**

将失败重现、波形为依据的定位工具以及一致性传播与验证集成到单一闭环，克服跨文件跨模块缺陷定位和修复一致性难题。

**🔧 技术方法**

采用失败 grounding、波形查询工具箱（时域定位、连通性、活跃路径、语义提升）以及基于仓库关系图的修复范围扩展与重放验证；结合 HDL 仿真、Verilog 编译与生成 RTL 映射等技术。

**📊 数据集**

在 HWE-Bench 的 388 个真实仓库级缺陷任务上进行评估，这些任务来自六个开源 RTL 项目（OpenTitan、Ibex、CVA6、Caliptra、XiangShan、Rocket Chip）。

**📈 对比分析**

与基线模型（DeepSeek V4 Flash、DeepSeek V4 Pro、Kimi K2.6）在同一 388 任务上对比，框架在每个模型上都提升了解决率（如 OpenHands 从 273/388 提升至 312/388，平均提升约 7%），且开销仅略有增加。

**⚠️ 局限性**

对缺乏可重现波形或需要生成器级决策的错误仍无法定位；代理对协议与微架构知识不足导致修复不完整；生成 RTL 追溯困难导致范围扩展不完全。

---

## 552. D2K-Bench: Can LLM Agents Turn Expert Designs into Efficient GPU Kernels?

**arXiv ID:** 2610.03226 | [PDF](https://arxiv.org/pdf/2610.03226v1)

**作者:** Daifeng Li `[一作]` (HKUST), Dayiheng Liu `[通讯]` (Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并评估了一个新的 GPU kernel 生成基准，测试 LLM 代理在给定专家指导下的实现性能。

**💡 创新点**

创新点在于引入分层专家设计指导（L1‑L3）与对等任务比较，并通过 LLM 判别器评估设计发现与实现完整性。

**🔧 技术方法**

采用 Triton 编译器、MCP 工具链、Qwen3.8 判别器、定制任务文件以及几何平均性能评分与实现得分等技术。

**📊 数据集**

使用 26 个 GPU kernel 优化任务，总计 85 个工作负载，数据来源于 vLLM、SGLang、FlashAttention 等生产级实现。

**📈 对比分析**

通过对比无指导与有指导的双重实验，计算每个模型的 S_perf（几何平均加速）和 JS_impl/JS_design 得分；指导后正确率从 93.1% 提升至 98.5%，S_perf 提升 33.9%。

**⚠️ 局限性**

局限性包括 LLM 代理仍未完全实现所有评估设计属性，且评估不适用于 SOL‑bound 分析，仅在 NVIDIA B200 GPU 上验证。

---

## 553. KV$^2$: A Self-Refining KV Cache

**arXiv ID:** 2610.03198 | [PDF](https://arxiv.org/pdf/2610.03198v1)

**作者:** Johannes Wesch `[一作]` (Karlsruhe Institute of Technology), Jan Niehues `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种可重用KV缓存压缩方法KV²，利用轻量级代理选择信息丰富的查询后，仅对该小集合做完整重构，从而在极低预算下保持模型性能并显著降低压缩阶段成本。

**💡 创新点**

创新点在于两阶段选择性重构框架：先用代理评分挑选稀疏的关键查询，再用它们对整个缓存进行局部重构；这种方式既保留了重构的精度，又避免了全量重构的高成本。

**🔧 技术方法**

采用KeyDiff等轻量级代理评分、分块(chunk)处理、选择性重构、FlashAttention‑2、PyTorch、HuggingFace Transformers等技术；整体实现为基于掩码的KV缓存压缩。

**📊 数据集**

使用的评估数据集包括RULER（4K/16K）、Needle‑in‑a‑Haystack以及LongBench等长文本推理与检索基准。

**📈 对比分析**

与KeyDiff、Expected Attention、KVzip等主流基线进行对比。KV²在2%–10% KV缓存预算下，在RULER、LongBench和Needle‑in‑a‑Haystack上均明显优于基线，尤其在极端压缩时提升40+个百分点；压缩阶段的运行时和峰值内存均低于KVzip。

**⚠️ 局限性**

局限性包括：迭代细化需要额外计算；代理评分方法仍可进一步提升；当前实现基于掩码而非物理压缩，未能直接展示解码时的显著内存/延迟收益。

---

## 554. Implementation of Hybrid QoS-Aware Data Radio Bearer in 5G Networks

**arXiv ID:** 2610.03209 | [PDF](https://arxiv.org/pdf/2610.03209v1)

**作者:** Padmapriya Patil `[一作]` (University of Texas at Dallas), Koteswararao Kondepu `[通讯]` (Indian Institute of Technology Dharwad)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于5QI的混合QoS流到DRB映射机制，在OpenAirInterface平台上支持单PDU会话内多QoS流的动态分配。

**💡 创新点**

创新点在于将many-to-one与one-to-one映射结合，根据5QI特征动态决定共享或专用DRB，实现QoS流的柔性隔离与资源共享，克服了传统单DRB映射的效率和隔离不足。

**🔧 技术方法**

使用了SDAP层动态QFI-DRB映射、OAI eBPF数据路径、NGAP/RRC/PDCP/SDAP/E1AP等协议栈修改，以及OAI多DRB支持等技术。

**📊 数据集**

未使用公开数据集，采用自建实验流（GBR、Non-GBR、Delay-Critical GBR）和iperf下行流量进行性能评测。

**📈 对比分析**

通过与OAI单DRB基线对比，采用三种负载场景和可扩展性实验，测量吞吐量、延迟、谱效率、QoS满足度等指标，混合方案显著提升吞吐量与延迟性能，并在高负载下保持Delay-Critical流的稳定性。

**⚠️ 局限性**

限制在于实验仅在受控环境下验证，未考虑动态QoS流增删、多gNB多用户场景，以及实际部署中的兼容性与标准化问题。

---

## 555. Predicting and Repairing Merge Collapse in Large Language Models

**arXiv ID:** 2610.03199 | [PDF](https://arxiv.org/pdf/2610.03199v1)

**作者:** Jungseob Lee `[一作]` (Korea University), Heuiseok Lim `[通讯]` (Korea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了一种基于任务向量方差的预合并屏蔽与修复方法，能够预测和修复大型语言模型合并导致的性能崩溃。

**💡 创新点**

提出了“PRISM”软阈值合并算子，该算子根据任务向量方差（干扰）自适应设定阈值并结合抑制冲突的门控机制。

**🔧 技术方法**

利用任务向量方差估计、干扰得分、软阈值（Donoho–Johnstone）以及干扰门控技术。

**📊 数据集**

使用公开的LLM基线（Qwen2.5、Llama‑3.1、Mistral‑7B、DeepSeek‑7B）以及自行训练的数学、代码、金融、医学等领域专家模型；评估采用GSM8K、ARC‑Challenge、MMLU、TruthfulQA、HellaSwag、WinoGrande、WikiText‑2等标准基准。

**📈 对比分析**

与Task Arithmetic、TIES、DELLA、DARE、LEWIS、AdaMerging等多种基线对比，PRISM在所有测试中都避免了合并崩溃，并在多数任务上保持或提升相对基线10–20个百分点。

**⚠️ 局限性**

仅在干扰显著但冲突低的情形下会退回到简单平均，且对非常大规模模型或多专业合并的理论与实验验证仍有限；方法仍依赖对权重统计的精确估计，并未直接处理安全与偏见问题。

---

## 556. AFORE: Attention-FFN Disaggregation with Overlapped Reconfiguration of Experts

**arXiv ID:** 2610.03203 | [PDF](https://arxiv.org/pdf/2610.03203v1)

**作者:** Wenshuang Li `[一作]` (Hong Kong University of Science and Technology), Binhang Yuan `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大规模稀疏专家模型（MoE）的推理服务，在注意力–前馈网络（AFD）分离架构中实现了基于微批次的专家重配置，能够动态调整专家放置并将迁移任务与前置微批次计算重叠。

**💡 创新点**

创新点包括：① 通过提前预取目标微批次的专家需求，实现在微批次级别的实时重配置；② 设计了迁移感知的调度算法，在评估迁移收益与暴露成本后决定是否重新配置；③ 在AFD流水线中利用空闲窗口并行执行专家迁移，完全隐藏迁移延迟。

**🔧 技术方法**

使用的技术包括：MoE专家并行、AFD分离架构、GPU–GPU NVLink迁移、共享内存调度队列、基于整数分配的贪心调度算法、微批次级别的专家需求采样与聚合。

**📊 数据集**

实验数据集为四种实际工作负载：ShareGPT、FineWeb、CodeForces 和 GSM8K，覆盖对话、文档、代码生成和数学推理等场景。

**📈 对比分析**

与静态放置、统计重配置、反应式负载平衡、预测式重配置等基线相比，系统在所有四种工作负载下均实现了 10.1–17.6% 的吞吐量提升和 7.1–9.5% 的 P95 交互令牌延迟下降；调度开销低于 0.25 ms，近似全局最优的负载平衡。

**⚠️ 局限性**

局限性：需在已实现AFD流水线的系统中部署，迁移时仍依赖 GPU 互连带宽；对极高专家并行度或极端负载波动时，预取窗口可能不足以捕获所有变化；以及在多节点大规模部署时，跨节点迁移开销与网络延迟可能成为新的瓶颈。

---

## 557. Monge matrix searching for lot sizing with piecewise-concave production costs

**arXiv ID:** 2610.03277 | [PDF](https://arxiv.org/pdf/2610.03277v1)

**作者:** Kleitos Papadopoulos `[一作]` `[通讯]`, Kleitos Papadopoulos

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种针对单品种限量库存问题的确切算法，该算法能够处理分段凹形生产成本、可回滚库存、库存界限以及回滚成本；

**💡 创新点**

创新点在于将问题归约为固定分段凹形生产成本下的双阶梯Monge矩阵，利用矩阵搜索（SMAWK或矩形搜索）在O(T^m+2α(T+2))（确定性）或O(T^m+2)（期望）时间内完成动态规划；

**🔧 技术方法**

使用技术包括状态枚举、前后缀表、Monge矩阵性质、双阶梯分解、SMAWK矩阵搜索、随机化Las Vegas搜索以及回溯构造最优方案；

**📊 数据集**

实验数据集为人工生成的三类问题（无回滚、有限产能、无产能）共300个基准案例和624个扩展案例，所有实例均遵循固定的生产断点和库存边界集合；

**📈 对比分析**

与原始的Koca–Yaman–Aktürk动态规划（KYA）进行比较，结果显示在大规模无产能、自由终端库存场景下R‑SMAWK实现可实现约1.6倍的速度提升，但在小规模或有限产能场景下两者相差不大，甚至KYA略占优势；

**⚠️ 局限性**

局限性包括：需要固定数量的公共生产断点和库存边界；假设所有成本查询和算术操作是精确且单步的；未实现更强的双阶梯搜索原语；未对多品种、非闭区间或不连续成本等更一般情况进行理论或实验验证。

---

## 558. Learning a Fact Is Not Learning How to Retrieve It

**arXiv ID:** 2610.03251 | [PDF](https://arxiv.org/pdf/2610.03251v1)

**作者:** Chaemin Jang `[一作]` (Korea Advanced Institute of Science and Technology), Dongman Lee `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在两阶段训练框架下，探究语言模型如何区分知识学习与检索方式学习，并通过操纵上下文状态验证其对事实检索的控制作用。

**💡 创新点**

提出将事实学习与检索方式学习分离的概念，发现上下文状态是两者的桥梁，并展示仅通过调整上下文状态即可在不改变权重的情况下打开或关闭已学知识的检索。

**🔧 技术方法**

采用对比实验、隐藏层状态分析、方向向量干预、以及在训练期间对上下文状态进行投影与移位的技术。

**📊 数据集**

使用人工构造的事实集合（首都、货币、人口、创始人、出生年份等），在多种预训练模型（Pythia‑410M、Pythia‑1.4B、Qwen‑2.5‑1.5B、Llama‑3.2‑1B）上进行实验。

**📈 对比分析**

与传统只用声明式训练的基线对比，发现仅在第一阶段加入多种请求形式即可使第二阶段在列表、冒号结尾等未见形式上检索准确率从接近0提升至≈97‑99%，且通过上下文状态干预可在不改写训练文本的情况下显著提升检索性能。

**⚠️ 局限性**

实验基于人工事实集合，缺乏自然语言文本的真实性；上下文状态解释仅适用于所研究的检索失败，不能覆盖所有知识检索失效情况；效果依赖于预训练模型对请求形式的先验理解，可能随模型规模或预训练策略变化。

---

## 559. Pinning Decisions Before Failure: Executable Records of Underspecified Choices in AI-Assisted Code Generation

**arXiv ID:** 2610.03237 | [PDF](https://arxiv.org/pdf/2610.03237v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 560. Asymptotic Analysis of Trading Fees in CFMM

**arXiv ID:** 2610.03262 | [PDF](https://arxiv.org/pdf/2610.03262v1)

**作者:** Peiyang Jin `[一作]`, Jing Qian `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

研究了在CFMM（自动化做市商）中，当交易费率趋近于零时，流动性提供者（LP）通过套利交易获得的交易费总额，并给出了闭式公式；进一步扩展到含跳跃的价格过程，证明跳跃是导致LP损失的主要原因。

**💡 创新点**

创新点在于：
- 提供了一个适用于任意二次可微CFMM且价格过程为一般半鞅的交易费率趋零时的闭式公式；
- 证明了在连续价格过程下，交易费可以完全抵消LVR（损失与再平衡）损失；
- 通过引入跳跃组件，首次系统地量化跳跃对LP损失的影响；
- 为AMM设计者和LP提供了基于跳跃与延迟的实用指导。

**🔧 技术方法**

主要使用了随机微积分工具：
- 二次变差、截断变差与对数变差的关系；
- Ito公式（含跳跃项）来拆解LP盈亏；
- 极限与渐近分析，推导出费率趋零时的结果；
- 结合CFMM的预设函数性质（例如g(p)）进行具体实例化。

**📊 数据集**

本工作为纯理论分析，不使用任何公开数据集；在论文中仅以Uniswap V3的公式为示例来验证推导结果。

**📈 对比分析**

与现有文献的比较主要在理论层面：
- 与之前忽略交易费的研究相比，本文填补了缺失的费率影响；
- 与专门针对G3M的研究相比，本文范围更广，适用于任意CFMM；
- 在连续价格过程下，理论上交易费可以完全抵消LVR损失；
- 在跳跃过程中，LP损失仅来源于跳跃导致的套利损失。

**⚠️ 局限性**

局限性：
- 假设仅存在套利交易，忽略了噪声交易和流动性挖矿等实际因素；
- 认为套利者能够即时并完全消除价格差距，未考虑交易延迟与网络拥堵；
- 只分析费率趋零的极限情况，对非零费率情形的精细行为未作深入探讨；
- 对跳跃的处理依赖于跳跃的统计特性，实际链上跳跃分布可能更复杂。

---

## 561. Seeing through the Eyes of AI: Situated Explainability in Augmented Reality

**arXiv ID:** 2610.03232 | [PDF](https://arxiv.org/pdf/2610.03232v1)

**作者:** Ana Stanescu `[一作]` (Adelaide University), Denis Kalkofen `[通讯]` (Graz University of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了“situated explainability”，在AR中实时、空间注册可视化AI模型的解释性热图，并进行了原型实现与用户研究。

**💡 创新点**

创新点在于首次将CAM等可解释方法与AR相结合，实现在用户现场的实时三维热图展示，支持数据收集与交互式调试。

**🔧 技术方法**

使用了Meta Quest 3、Unity3D、YOLOv8/12、RT-DETR、PaliGemma VLM、LayerCAM、深度相机、点云渲染以及离线分析等技术。

**📊 数据集**

训练使用COCO数据集，测试及用户研究基于真实环境中的杂物桌面和家庭储藏室场景，结合预录制图像。

**📈 对比分析**

通过与传统2D静态热图的对照实验，使用SUS、UEQ、NASA‑TLX、TiA等量表评估；结果显示AR在情感体验上显著优于2D，但工作负荷更高；整体可用性相当。

**⚠️ 局限性**

局限性包括样本量仅为有AI背景的少数受试者、头戴设备佩戴舒适度、仅支持静态对象的离线模式、认知负荷较高、未评估非专业用户、性别比例失衡、未实现动态物体跟踪等。

---

## 562. Kernel Singular Value Decomposition with Extension to Multiple Data Sources

**arXiv ID:** 2610.03216 | [PDF](https://arxiv.org/pdf/2610.03216v1)

**作者:** Xinjie Zeng `[一作]` (KU Leuven), Johan Suykens `[通讯]` (KU Leuven)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种多源核奇异值分解（MKSVD），利用非对称核学习多源数据的联合非线性特征，并给出了双重、协方差以及神经网络实现。

**💡 创新点**

创新点在于将传统二源KSVD推广到多源，并推导出多源核矩阵下的广义移位特征值问题、协方差表述以及利用零目标性质实现的可参数化神经网络版本，实现端到端监督学习。

**🔧 技术方法**

使用LSSVM框架、非对称核函数、移位特征值分解、协方差优化、Nyström近似、随机傅里叶特征以及神经网络显式特征映射等技术。

**📊 数据集**

实验采用七个真实数据集：Image-caption、3Sources、YouTube video、UCI Digit、BBC-3、Reuters-600、NUS-WIDE-OBJ。

**📈 对比分析**

与KPCA、MCCA、MvKPLS等基线比较，MKSVD在大多数数据集上实现了更高的宏F1得分；Neural版本在六个数据集上均优于基线；Dual和Covariance实现在效率上相近，Covariance在训练时更快。

**⚠️ 局限性**

主要局限在于神经网络变体高度依赖标签监督和超参数调优，且大规模核矩阵的计算仍然存在显著开销；未来工作需探索自监督目标以降低对标签的依赖。

---

## 563. VDOT++: Unified Few-Step Video Generation via Unbalanced Optimal Transport Distillation

**arXiv ID:** 2610.03221 | [PDF](https://arxiv.org/pdf/2610.03221v1)

**作者:** Yutong Wang `[一作]` (University of Sydney), Chang Xu `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种统一的少步视频生成框架VDOT++，通过不平衡最优传输（OT）蒸馏在文本转视频、图像转视频和基于条件的多任务生成中实现四步高效采样。

**💡 创新点**

创新点包括：①采用非对称不平衡OT，对学生边缘分布松弛、教师覆盖保持；②使用ℓ1地面代价取代平方代价，避免均值聚合导致信息丢失；③将分布匹配蒸馏与对抗细化结合，并通过分阶段后向传递实现内存友好训练；④跨尺度蒸馏利用更大规模的评分网络提升小模型性能。

**🔧 技术方法**

核心技术包括分布匹配蒸馏（DMD）、不平衡OT（自适应边缘约束）、ℓ1地面成本、相对对抗GAN、顺序后向合并、以及跨尺度评分网络蒸馏。

**📊 数据集**

实验使用VBench、VBench-I2V、VACE基准以及多任务语料库（包含文本、图像、视频与掩模控制），覆盖文本转视频、图像转视频和基于条件的生成三大任务族。

**📈 对比分析**

与多步教师模型及现有少步方法（如DMD2、Self-Forcing等）对比，四步VDOT++在各基准上均达到或超过教师性能，尤其在运动质量与图像细节方面显著提升，归一化平均分均表现领先。

**⚠️ 局限性**

局限性在于OT仅在同一时间帧内进行空间-token匹配，未建模跨帧或时空轨迹的传输；同时该框架局限于四步设置，需进一步探索一步、流式或自回归生成的适配。

---

## 564. Collective Bias Mitigation via Model Routing and Collaboration

**arXiv ID:** 2610.03240 | [PDF](https://arxiv.org/pdf/2610.03240v1)

**作者:** Mingzhe Du `[一作]` (Nanyang Technological University), See-Kiong Ng `[通讯]` (National University Of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Collective Bias Mitigation (CBM) 框架，通过模型路由器挑选合适的 LLM 并在不同协作拓扑（单体、顺序、投票、辩论、委员会）下协同生成回答，从而降低 LLM 的隐性偏见。

**💡 创新点**

创新点在于：①将多模型协作作为偏见缓解手段；②设计了基于行为数据的模型路由器，能够根据查询的社会维度选择最中性模型；③提出了多种拓扑结构，使模型之间能够互相交流、纠正偏见；④构建了 CrowdEval 数据集，细粒度记录各模型对偏见诱导问题的回应。

**🔧 技术方法**

技术手段包括：利用预训练 LLM 作为路由器，采用概率式路由机制；对选定模型在不同拓扑下执行交互（投票、辩论、委员会等）；使用 Fine‑Tuning、贪婪解码和 Consensus 机制来生成最终回答；通过 Bootstrap 评估路由器的准确性与精确度。

**📊 数据集**

使用的数据集：1) CrowdEval——由 BBQ（Ambiguous 子集）生成的约 1,024 题目，记录 50+ 开源 LLM 的细粒度回答；2) BBQ（Disambiguated 子集被排除）用于评估偏见分数；3) 训练路由器时采用 BBQ 的社交维度标签。

**📈 对比分析**

与单模型基线（如 Qwen2.5‑32B）对比：在 top‑7 配置下，CBM 在 age 维度将偏见分数从 0.25 降至 0.10；投票拓扑相较单体提升稳定，辩论拓扑取得最低偏见分数，但计算成本高约 27 倍；委员会拓扑在降低偏见与计算成本之间取得平衡，误差方差更小。

**⚠️ 局限性**

局限性包括：①路由器需要较大模型（≥9B）才能稳定工作；②多模型协作显著提高推理成本，尤其是辩论拓扑；③对模型互相信息交流的方式缺乏统一标准，可能导致协作失效；④CrowdEval 与 BBQ 只覆盖部分社会维度，未必能全面覆盖所有现实场景的偏见。

---

## 565. HyperFuse: Fast Self-Supervised Node Embeddings for Attributed Hypergraphs

**arXiv ID:** 2610.03211 | [PDF](https://arxiv.org/pdf/2610.03211v1)

**作者:** Megha P `[一作]` (Indian Institute of Science Education and Research Thiruvananthapuram), Saptarshi Bej `[通讯]` (Indian Institute of Science Education and Research Thiruvananthapuram)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出HyperFuse，一种快速无标签的自监督超图节点嵌入方法

**💡 创新点**

利用矩阵自由的Banerjee邻接式超图模量最大化、稳定性加权的特征与边缘摘要以及仅100个epoch的相关性自监督训练，显著降低嵌入时间

**🔧 技术方法**

谱模量最大化、残差化特征摘要、线性回归评估边缘效用、CCA-SSG自监督目标、轻量级聚合网络

**📊 数据集**

九个公开超图数据集（Cora‑CC、Citeseer、PubMed、Cora‑CA、DBLP、Zoo、20News、NTU2012、ModelNet40）

**📈 对比分析**

与TriCL、SE‑HSSL、VilLain、HypeBoy四个基线比较，HyperFuse平均嵌入时间快约143‑179倍，分类与聚类性能与TriCL/SE‑HSSL相当，优于最快基线HypeBoy

**⚠️ 局限性**

仅在离线全量重嵌入场景适用，未评估增量/时变超图；对高维特征内存消耗较大；对不同硬件/GPU规格的时间可变；在某些弱特征数据上性能略逊

---

## 566. Evolving Hybrid Quantum-Classical Architectures for Image Classification

**arXiv ID:** 2610.03220 | [PDF](https://arxiv.org/pdf/2610.03220v1)

**作者:** Devroop Kar `[一作]` (Rochester Institute of Technology), Travis Desell `[通讯]` (Rochester Institute of Technology)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将 EXAQC 进化搜索扩展到混合量子‑经典图像分类模型中，演化 PQC 作为中间处理模块。

**💡 创新点**

首次将 U3 编码与进化架构搜索相结合，并通过联合训练经典编码器与 PQC，实现高效小规模量子电路。

**🔧 技术方法**

采用进化搜索（变异、交叉、拉马克斯继承）、参数化量子电路、量子输入编码（RX、RY、U3、振幅）、经典卷积编码器和解码器，以及交叉熵训练。

**📊 数据集**

使用 MNIST、Fashion‑MNIST 和 CIFAR‑10 三个公开数据集。

**📈 对比分析**

与固定架构经典基线及之前的量子架构搜索方法比较，获得 MNIST 98.42%、Fashion‑MNIST 90.62%、CIFAR‑10 85.47%；在 CIFAR‑10 上超越前沿方法且门数更少；与 10 层 CNN 对比，参数减少 96% 仍保持 85.68% 的准确率。

**⚠️ 局限性**

限制：与先前工作对比缺乏统一评测；进化搜索耗时且对量子硬件噪声的鲁棒性未评估；对经典编码器质量高度依赖；未对编码器与电路进行联合优化。

---

## 567. AdaStep: Adaptive Step Credit Weighting for Agentic Reinforcement Learning

**arXiv ID:** 2610.03223 | [PDF](https://arxiv.org/pdf/2610.03223v1)

**作者:** Xin Wang `[一作]` (Tsinghua University), Jian Luan `[通讯]` (Xiaomi Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

AdaStep通过自适应权重对局部步骤优势进行加权，从而在LLM代理中实现更细粒度的信用分配。

**💡 创新点**

创新点在于将局部优势的权重推导为均方误差最小化的收缩系数，并将其解释为动作对返回方差解释度的比例，既无需额外的Critic也无需额外采样。

**🔧 技术方法**

采用了组相对强化学习（GRPO/ GiGPO）框架、锚点状态分组、方差分解与MSE投影计算收缩系数，并在奖励稀疏的情境下实现无critic的优势估计。

**📊 数据集**

使用了三大基准数据集：ALFWorld、WebShop 和 ScienceWorld，分别在三种LLM模型（Qwen3‑1.7B、Qwen3‑4B、Qwen2.5‑7B‑Instruct）上进行实验。

**📈 对比分析**

在与GRPO、GiGPO、HGPO、PPO、RLOO等基线对比时，AdaStep在所有模型-任务组合上均表现出显著提升，最大可达+9.36分（约提升10%），并且仅增加约1% 的优势计算时间。

**⚠️ 局限性**

局限性在于依赖于可直接匹配的锚点状态，难以扩展到连续或部分可观测环境；当某些步骤组样本不足时，需采用默认权重，可能导致估计不稳健。

---

## 568. StanceEval 2026: The Second Stance Detection Shared Task

**arXiv ID:** 2610.03215 | [PDF](https://arxiv.org/pdf/2610.03215v1)

**作者:** Rasha Albalawi `[一作]` (KFUPM), Nora Alturayeif `[通讯]` (HUMAIN)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并组织了 StanceEval 2026 共享任务，专注于阿拉伯语社交媒体文本的跨目标立场检测（相关目标和完全未知目标两条轨道），并通过大规模团队参与评估模型的泛化能力。

**💡 创新点**

创新点包括：1) 通过 Mawqif‑XT 新增测试目标（Women Driving、E‑Cars、Trimester System）实现真正的跨目标评估；2) 系统性比较传统微调编码器、LLM 提示、混合检索-LLM 等多种方法，并发现未见目标反而易于学习；3) 公开完整排行榜和错误分析，为后续研究提供基准。

**🔧 技术方法**

采用的技术有：预训练阿拉伯语编码器微调、跨目标验证策略；LLM 零/少样本提示、跨模型投票；检索增强型对话/上下文学习（RAG）；混合编码器-LLM管道；多任务学习、阈值优化、伪标签生成、对齐算法等。

**📊 数据集**

使用的数据集是原始 Mawqif（COVID‑19 Vaccine、Digital Transformation、Women Empowerment）作为训练/验证，Mawqif‑XT（Women Driving、E‑Cars、Trimester System）作为测试。两组数据均为手工标注的 Favor/Against/None、情感、讽刺标签。

**📈 对比分析**

评估指标为 F_avg2（Favor 与 Against 的宏平均 F1）、F_avg3（全三类）和准确率。基线模型（包括多语言 BERT、AraBERT、LLM 零样本）在两条轨道分别取得 0.7366 / 0.7475 的 F_avg2，顶尖系统分别达到 0.8994（轨道1）和 0.9400（轨道2），显著超过基线。

**⚠️ 局限性**

局限性：1) 轨道1与轨道2的目标在类别分布、极化程度、讽刺和中立内容上差异大，无法纯粹归因于跨目标泛化；2) F_avg2 忽略 None 类导致模型偏向立场；3) 测试集规模有限，主要覆盖海湾和沙特方言，可能不具备更广泛的泛化性；4) 错误分析基于系统日志与直观观察，缺乏系统化标注支持。

---

## 569. WAMpy: Efficient Synthesis of Prolog Programs in Python

**arXiv ID:** 2610.03234 | [PDF](https://arxiv.org/pdf/2610.03234v1)

**作者:** Dominik Magiera `[一作]` (Technische Universität Darmstadt), Frank Jäkel `[通讯]` (Technische Universität Darmstadt)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了WAMpy，一个专为Prolog程序合成工作负载优化的Python框架，支持高效编译与评估小型候选程序。

**💡 创新点**

创新点在于将WAM指令转为NumPy数组表示，结合Numba JIT并引入部分重编译机制，使得仅对假设更改时重编译，显著提升性能。

**🔧 技术方法**

使用Python、NumPy数组、Numba JIT编译、Warren Abstract Machine（WAM）子集实现，并提供Python高层API进行解析、编译与查询。

**📊 数据集**

在一个模拟家族关系的基准数据集（13条性别事实、13条父母事实和7条规则）上进行评测。

**📈 对比分析**

与Janus 1.5.3（通过SWI-Prolog 10.0.2嵌入Python）对比，WAMpy在端到端时间上约快38.5倍；在多次迭代后，部分重编译模式仅需约6.2 ms，而Janus约240 ms。

**⚠️ 局限性**

仅实现了WAM的X寄存器子集，缺少完整的SWI-Prolog内置谓词库（如算术、meta-call等），无法满足通用Prolog功能需求。

---

## 570. VisionMX: Unlocking Microscaling Post-Training Quantization for Vision Models

**arXiv ID:** 2610.03218 | [PDF](https://arxiv.org/pdf/2610.03218v1)

**作者:** Elad Dror Cohen `[一作]`, Hai Victor Habi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文研究了视觉模型后训练Microscaling（MX）量化，提出了新的RangeRound与激活仿射校正（AAC）方法来提升精度。

**💡 创新点**

创新点在于将范围学习舍入（RangeRound）与MX专用的激活仿射校正（AAC）相结合，解决权重在非均匀浮点网格上的重构误差和激活非负分布导致的码位浪费。

**🔧 技术方法**

采用了块级量化、AdaRound式的软正则化学习舍入、区块重建优化、以及多维范围搜索等技术。

**📊 数据集**

实验数据集包括ImageNet‑1K（分类）、COCO（目标检测与语义分割）和低照度图像增强数据集。

**📈 对比分析**

与RTN、AdaRound等基线比较，4‑bit MXFP4量化下在分类中提升3–10% Top‑1 率，检测与分割的AP和mIoU平均提升5–10个百分点，尤其在小型模型上效果显著。

**⚠️ 局限性**

局限性在于块尺寸与尺度格式的动态范围限制，非均匀网格仍可能产生重构误差；在极低精度或极大模型上提升幅度有限。

---

## 571. EVOL: Simulator-Guided Evolutionary Expert Synthesis for Deployment-Free Learning Path Recommendation

**arXiv ID:** 2610.03273 | [PDF](https://arxiv.org/pdf/2610.03273v1)

**作者:** Geonwoo Bang `[一作]` (Sungkyunkwan University), Moohong Min `[通讯]` (Sungkyunkwan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个演示学习框架 EVOL，通过知识演化模拟器（KES）进行进化搜索生成每个学习者的专家学习路径，然后将这些演示蒸馏成无模拟器的路径推荐策略。

**💡 创新点**

核心创新在于将仿真环境既用于训练又用于专家路径生成，并采用非对称的 actor‑critic 结构，使得部署时可无仿真一次性生成完整路径；另外证明专家质量决定最终性能，而非具体的模仿算法。

**🔧 技术方法**

使用深度知识追踪（DKT）作为 KES、进化搜索（遗传算法）生成专家路径、行为克隆（BC）或 AWR / DAPG 模仿、对称与非对称 actor‑critic 的 PPO 微调。

**📊 数据集**

三大数据集：ASSIST15（100 课题），Junyi Academy（39 课题），EdNet（189 课题），分别实验不同路径长度 L∈{5,10,20}。

**📈 对比分析**

与 8 类基线（启发式、顺序推荐、RL、图增强 RL、LLM 增强）在部署无模拟器约束下对比，EVOL 在所有数据集和路径长度上均优于基线，提升 EP 均在 0.03–0.12 之间；BC/AWR/DAPG 结果相近，表明专家质量是关键。

**⚠️ 局限性**

局限性包括：专家路径仅在 DKT 模拟器内部生成，可能继承模型的偏差；实验仅在模拟环境中评估，未验证对真实学习者的实际效果；对不同的响应模型和模拟器实例仍存在一定的稳健性考验。

---

## 572. Mapping and Advancing the Scalability-Accuracy Frontier of Nonlinear Causal Discovery

**arXiv ID:** 2610.03258 | [PDF](https://arxiv.org/pdf/2610.03258v1)

**作者:** Hendrik Suhr `[一作]` (CISPA Helmholtz Center for Information Security), Jilles Vreeken `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对非线性因果结构学习中的可扩展性-精度权衡进行系统评估，并提出基于固定基函数样条的可复用分数计算方法。

**💡 创新点**

创新点在于利用固定基函数样条的可复用统计量，将组合搜索中的局部分数评估从 O(nd³) 降低到 O(nd²+d³)，显著扩展可处理的变量数和样本量。

**🔧 技术方法**

采用固定基函数通用加性样条回归、BIC 分数、梯度下降、稀疏有向无环图搜索以及多种对比算法。

**📊 数据集**

使用合成的高斯过程非线性加性噪声模型、Erdős–Rényi DAG、Causal Chamber 真实数据以及 Sachs 蛋白质信号数据集。

**📈 对比分析**

与四大主流方法（组合搜索、可微学习、扩散学习、得分匹配）以及多种竞争者进行对比，结果显示新方法在保持高结构精度的同时实现数百倍加速，能处理高达 1600 维变量和 160k 样本。

**⚠️ 局限性**

局限性在于仅针对可观测且可识别的加性噪声模型，未覆盖部分可识别、潜在混杂或异方差机制，并且实验主要基于仿真，需进一步验证在更广泛真实场景中的鲁棒性。

---

## 573. A space-time finite element formulation for geometrically exact shear-deformable beams

**arXiv ID:** 2610.03285 | [PDF](https://arxiv.org/pdf/2610.03285v1)

**作者:** Ivo Steinbrecher `[一作]`, Alexander Humer `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种统一的时空有限元方法，用混合形式将几何精确剪切可变形梁的空间和时间离散化，同时引入独立的平移和角速度场，并使用目标旋转插值和时间向上风化稳定化。

**💡 创新点**

创新点在于：①把空间与时间同时离散为时空曲面；②通过引入独立速度场降低时间导数阶数，便于施加初始速度条件；③利用目标旋转插值保持旋转对象性；④直接通过时空曲面几何处理时间变化的材料域（如滑动结构）。

**🔧 技术方法**

使用了几何精确剪切梁理论、旋转向量插值、Petrov–Galerkin 以及时间向上风化（SUPG）稳定化、自动微分求解非线性系统，并在开源软件 BeamMe/AceFEM 中实现。

**📊 数据集**

通过数值案例（预弯曲梁、L形自由梁、滑动意大利面问题）进行验证，未使用公开数据集，而是自行设定梁几何、材料参数和载荷时间历程。

**📈 对比分析**

与经典的基于普遍时间步进（Generalized‑α）以及 Sliding Beam Formulation (SBF) 进行对比，误差收敛率满足预期（ST‑4: 二阶，ST‑9: 三阶），性能与传统时序法相当，但在高频/大位移下需要较强稳定化；稳定化参数对结果影响显著。

**⚠️ 局限性**

局限性包括：①对小稳定化参数时收敛性差；②对非材料边界的动态处理仍需额外变换；③计算成本高，尤其在长时间或高分辨率时空网格下；④缺乏严谨的理论分析和稳定性证明。

---

## 574. The Neuro-Physical Inverter: A Modular Framework for Magnetotelluric Inversion Coupling Ensemble Conditioning with Residual Learning

**arXiv ID:** 2610.03225 | [PDF](https://arxiv.org/pdf/2610.03225v1)

**作者:** Jae Deok Kim `[一作]` (Massachusetts Institute of Technology), Rob. L. Evans `[通讯]` (Woods Hole Oceanographic Institution)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了Neuro‑Physical Inverter（NPI），一种两阶段模块化框架，先用单次集合条件化（EnsCGP）生成符合观测的先验基准，再用受限残差卷积网络（ResNet‑1D）对该基准做物理约束下的残差修正；

**💡 创新点**

创新点在于把单次Kalman式集合条件化作为可解释、可扩展的先验校准；残差网络仅学习残差而非完整模型，保持物理一致性和不确定性传递；整个流程无维度限制，可直接迁移至更高维问题；

**🔧 技术方法**

采用Ensemble‑Conditional Gaussian Process、Kalman式更新、残差卷积神经网络（ResNet‑1D）、物理耦合微调损失、Wait算法前向模拟，训练使用AdamW、Huber损失、K‑fold交叉验证；

**📊 数据集**

使用先验为3D地质模型提取的低秩高斯马尔可夫随机场（GMRF）生成的合成1D电阻率样本；随后在美国内华达州Gabbs Valley 81个宽带MT站点的旋转不变1D响应上进行实测验证；

**📈 对比分析**

与单独的EnsCGP、随机化‑优化RTO‑TKO以及端到端全逆网络做对比；在合成测试中NPI将均方误差从0.131降至0.111，数据拟合nRMS从1.355降至0.962，覆盖率提升至96.8%；在实测数据中NPI在中频段将误差从约10%降至几%，并保持与RTO‑TKO相当甚至更窄的不确定区间，计算速度比RTO‑TKO快约50倍；

**⚠️ 局限性**

限制包括仅在1D框架中验证，无法捕捉多维结构；不确定性为经验度量而非严格贝叶斯后验；对先验的依赖较强，需在高维时重新生成先验且计算成本显著；残差网络对训练分布和细节调参的稳健性仍待进一步评估。

---

## 575. Parallel Time-Aligned Spiking Self-Attention for Consistent Integer-Valued Training and Spike-Driven Inference

**arXiv ID:** 2610.03291 | [PDF](https://arxiv.org/pdf/2610.03291v1)

**作者:** Peng Xue `[一作]` (Shenzhen Institute of Advanced Technology Chinese Academy of Sciences), Huihui Zhou `[通讯]` (Pengcheng Laboratory)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种并行时序对齐的脉冲自注意力（PT-SSA）以及其自适应阈值版本（Adaptive PT-SSA），通过重构虚拟脉冲切片并仅在同一时间步内计算注意力，消除了整数化训练与脉冲推理之间的时间交互不匹配（TIM）问题；

**💡 创新点**

创新点在于：①识别并量化了SFA基础脉冲自注意力中的TIM；②提出PT-SSA实现时间对齐的并行注意力计算；③引入自适应阈值模块以补偿SFA输出尺度的改变，从而显著提升脉冲推理准确率；

**🔧 技术方法**

采用整数化LIF计数（I-LIF）与脉冲发射率（SFA）两种神经元表示；利用并行脉冲神经元、脉冲自注意力、梯度直通估计（STE）与自适应阈值学习；实现了Triton融合的高效训练核；

**📊 数据集**

在CIFAR‑10、CIFAR‑100以及ImageNet‑1K三个视觉分类数据集上进行实验；

**📈 对比分析**

与传统SSA、递归LIF SSA以及SDT‑V3等方法对比，PT-SSA在CIFAR上将整数-脉冲差距从≈1.8%降至≈0.3%，在ImageNet上将差距从27.78%降至0.06%，并保持2.9×吞吐率优势，显著提升了脉冲推理精度和训练效率；

**⚠️ 局限性**

限制包括：整数化训练的准确率仍略低于SSA；需额外的阈值学习与梯度路由设计；对大规模虚拟时间步（D）敏感；实现复杂度提升，尤其是重构脉冲切片的内存与计算开销；在非Transformer或不同任务上的泛化性尚待验证。

---

## 576. CVE2AP: Automated Generation of PDDL-Encoded Attack Paths via Large Language Models

**arXiv ID:** 2610.03383 | [PDF](https://arxiv.org/pdf/2610.03383v1)

**作者:** Lin Cui `[一作]` (Karlsruhe Institute of Technology), Raffaela Mirandola `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 CVE2AP，一个基于大型语言模型（LLM）的框架，能够自动将 CVE 说明转换为 PDDL 编码的攻击路径；

**💡 创新点**

创新点在于：①设计了结构化提示与错误反馈循环，结合规划器的语法与可解性检查实现迭代纠错；②通过多模型、多配置系统性评估，并提出针对安全语义的评估维度；

**🔧 技术方法**

核心技术包括 LLM 生成（支持聊天式与补全式接口）、结构化提示（系统/用户/助手消息）、错误反馈机制（Metric‑FF 语法/可解性检查反馈）、S3Eval 评估框架；

**📊 数据集**

使用了 21 条 CVE 描述及对应专家编写的参考 PDDL 路径作为数据集；

**📈 对比分析**

与六种 LLM（包括在线与本地多尺寸模型）及 12 种提示配置（0/1/2-shot、是否错误反馈、是否模板）进行对比，结果显示最佳模型在语法正确率 86.9%、可解性 78.6%、LLM‑expert 语义正确率 93.1%，并且在生成时间与 token 消耗上取得较好平衡；

**⚠️ 局限性**

局限性包括：仅处理单一 CVE 的攻击路径；错误反馈只覆盖语法/可解性，未对语义错误做反馈；对小模型的上下文容量有限，导致效果受限；

---

## 577. Bidirectional Voronoi-biased Exploration Curriculum for Reinforcement Learning

**arXiv ID:** 2610.03395 | [PDF](https://arxiv.org/pdf/2610.03395v1)

**作者:** Juri Pfammatter `[一作]` (ETH Zürich), Marco Hutter `[通讯]` (ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了BVER课程，通过双向Voronoi偏置的探索扩展从目标和初始分布生成中间起始点和目标，训练单一目标条件策略。

**💡 创新点**

创新点在于结合RRT和RRT-Connect的双向Voronoi偏置扩展，既生成中间起点又生成中间目标，并在两个方向同时进行，以覆盖整个任务空间。

**🔧 技术方法**

使用了基于PPO的强化学习框架，随机行走模拟器重置、Voronoi偏置采样、目标空间连接以及目标条件策略训练。

**📊 数据集**

在六个模拟任务上进行实验，包括三种点质量迷宫、四足机器人爬箱、环杆转移以及四个真实环境扫描地形。

**📈 对比分析**

与无课程、随机课程、逆向课程以及需要演示的参考方法对比，BVER在所有任务上实现最快的样本效率，甚至在爬0.7箱时唯一成功的参考无课程方法，且鲁棒性高。

**⚠️ 局限性**

局限在于需要可重置任意状态的模拟器、可逆动态假设、任务空间需有欧氏距离且维度低，且实验仅在仿真环境中验证，未部署到真实机器人。

---

## 578. Interpretable Deepfake Detection in Videos via Explicit Forensic Features and Temporal Modeling

**arXiv ID:** 2610.03380 | [PDF](https://arxiv.org/pdf/2610.03380v1)

**作者:** Chahira Benhama `[一作]` (University of Quebec in Outaouais), Assia Hamadene `[通讯]` (University of Quebec in Outaouais)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种基于显式法医学特征与时序建模的可解释深度伪造视频检测框架。

**💡 创新点**

创新点在于：①引入四类物理意义明确的特征（光度、纹理、几何、压缩）并在身份一致的轨迹上进行时序建模；②利用LSTM捕捉连续帧的细微不一致，提升跨数据集泛化；③保持模型可解释性，无需后置解释。

**🔧 技术方法**

核心技术包括：面部检测与Kalman滤波轨迹跟踪、固定长度时间窗口分段、68维显式特征提取（包括GLCM、LBP、波形子带、ELA、JPEG残差等）、双层LSTM分类网络以及多数据集统一预处理。

**📊 数据集**

使用了FaceForensics++、Celeb-DF v2、DFDC子集和DeeperForensics四个公开数据集；在ForgreenNet和WildDeepfake上做零样本跨域验证。

**📈 对比分析**

与近年时序一致性、频域建模、上下文感知等方法对比，本文在四个主要数据集上取得平均F1≈95.7%、AUC≈0.98，跨域零样本时在ForgeryNet和WildDeepfake分别达到93.8%/89.4%准确率。

**⚠️ 局限性**

局限性包括：①对极端姿态、表情变化的鲁棒性待提升；②依赖手工特征，可能在极高质量伪造中失效；③LSTM对长序列记忆有限，未来可尝试Transformer或图神经网络提升。

---

## 579. From Patching to Pruning Visual Computation in Vision Language Models

**arXiv ID:** 2610.03389 | [PDF](https://arxiv.org/pdf/2610.03389v1)

**作者:** Rahul Chowdhury `[一作]` (Northeastern University), Yanzhi Wang `[通讯]` (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Patch-to-Prune (P2P)，一种训练‑free 的推理加速框架，通过在解码器各层识别视觉令牌计算冗余并用固定原型替代，从而省去不必要的注意力与 MLP 计算。

**💡 创新点**

创新点在于将激活补丁（activation patching）从仅用于解释的诊断工具转变为实际推理时的计算绕过技术，并通过验证引导的层级安全区选择，实现在不删除令牌、不改变序列结构的前提下实现高效推理，同时揭示视觉信息在解码器深度上的非均匀分布。

**🔧 技术方法**

使用机制解释中的激活补丁、前向与后向累积层级扫描、模态条件激活原型缓存、P2P‑aware LoRA 微调等技术；同时进行层级敏感度分析与验证‑导向的安全层选择。

**📊 数据集**

在 Qwen2.5‑VL‑3B‑Instruct 与 LLaVA‑Next（Vicuna‑7B/13B）模型上，使用七个多模态基准数据集（POPE、WhatsUp、ScienceQA‑IMG 等）进行实验，采用互相独立的校准、验证、测试集。

**📈 对比分析**

与原始模型、FastV、PyramidDrop 等方法对比，P2P 在 3% 容差下保持约 94% 的稠密准确率，FLOPs 降低约 55%，在多模态基准上实现显著的计算与延迟提升。

**⚠️ 局限性**

局限性包括：需要额外的校准/验证集来确定安全层；在极端容差下可能导致显著准确率下降；仅针对预训练模型推理加速，对不同模型架构的泛化性有限；无法动态捕捉令牌重要性随时间变化的情况。

---

## 580. Multilingual GSM-Symbolic: What determines capability transfer across languages?

**arXiv ID:** 2610.03367 | [PDF](https://arxiv.org/pdf/2610.03367v1)

**作者:** Kenneth Enevoldsen `[一作]` (Aarhus University), Kristoffer Nielbo `[通讯]` (Aarhus University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并公开了Multilingual GSM‑Symbolic多语言数学推理基准，并用其系统评估跨语言能力迁移；

**💡 创新点**

①使用可扩展的符号模板生成大量匹配样本；②联合建模模型规模、语言资源、推理能力与语言典型距离等因素对迁移的影响；③给出对未见语言性能的高精度预测；

**🔧 技术方法**

使用符号模板技术生成数据、零样本提示评估、统计学中的广义线性混合模型进行效应分析，并借助LLM做翻译与验证；

**📊 数据集**

基于GSM8K与GSM‑Symbolic改造而成的Multilingual GSM‑Symbolic，覆盖15种语言、30,000个题目-答案对，并利用Common Crawl估算语言资源；

**📈 对比分析**

通过零样本提示在多语言上对比模型，发现模型规模（β=1.77）、语言资源（β=0.77）、推理能力（β=0.67）和典型距离（β=‑0.25）是主要决定因素；预测误差可控制在≈6pp（仅10模板可降至≈4pp）；

**⚠️ 局限性**

受限于样本规模（15种语言）、仅涉及小学算术题、可能出现饱和、结果对其他领域或模型类型的泛化性有限。

---

## 581. Operator-informed initialization for Fourier features physics-informed neural networks

**arXiv ID:** 2610.03378 | [PDF](https://arxiv.org/pdf/2610.03378v1)

**作者:** Juan Molina `[一作]` (National Center for Artificial Intelligence), Francisco Sahli Costabal `[通讯]` (Millennium Institute for Intelligent Healthcare Engineering)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于PDE符号的频率感知初始化方法，用于Fourier Feature PINNs，从而显著缓解频率偏差问题；

**💡 创新点**

创新点在于：1）从NTK理论出发推导残差频率动力学，揭示频率偏差由PDE符号与初始化分布共同决定；2）设计了将PDE符号嵌入初始化频率采样的分布，使得高频、弱约束方向的权重得到提升；3）实现无额外训练成本的改进；

**🔧 技术方法**

采用的技术包括：Fourier Feature网络、Neural Tangent Kernel（NTK）分析、PDE符号（拉普拉斯、波动、热方程等）的计算、基于符号的初始化分布、梯度下降/ADAM优化以及JAX-PI框架集成；

**📊 数据集**

实验数据集涵盖多种线性及半线性PDE（Helmholtz、Laplace、Transport、Heat、Wave、非线性Laplace、Burgers、Sine–Gordon、Klein–Gordon）以及JAX-PI基准（Advection、Allen–Cahn、Kuramoto–Sivashinsky）;

**📈 对比分析**

与传统高斯初始化对比，采用PDE感知初始化在训练损失、相对L²误差、频域误差以及随机种子稳定性上均优于对照组；在SOTA与基准配置下均取得更快收敛、更低误差、波动更小，尤其在Transport、Helmholtz和Wave方程中提升显著；

**⚠️ 局限性**

局限性包括：1）仅适用于可写成常系数线性或半线性PDE，需已知主导线性符号；2）对高度非线性或非常系数问题难以构造有效符号；3）对频率分布的假设（如高斯与指数衰减）可能不适用于所有领域，未来需扩展至更一般的算子或自适应频率策略。

---

## 582. Refinement Buys Intelligibility, Search Buys Identity: What Test-Time Compute Buys in Masked-Diffusion TTS

**arXiv ID:** 2610.03320 | [PDF](https://arxiv.org/pdf/2610.03320v1)

**作者:** Nityanand Mathur `[一作]` (Blackstar Inc), Ayush Pratap Singh `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了扩散式语音合成模型中可租用的推理步骤（refinement）与固有的模型深度（depth）对语音可懂度和说话人身份的不同影响，使用宽度-深度网格训练后仅在推理时调节步骤数，评估跨句零样本合成的WER与SIM-o，提出“可达范围占比”这一模型无关的评估指标。

**💡 创新点**

创新点在于：①首次用可达范围占比而非绝对误差评估推理步骤对可懂度与身份的相对收益；②发现推理步骤对可懂度收益显著但对身份收益有限，且二者瓶颈不同；③通过模型-无关实验验证深度与步骤不可简单替代，拒绝“depth-to-steps”统一交换率假设；④将训练计算、最佳候选搜索与推理步骤三种杠杆统一对比，揭示身份提升主要来自训练与搜索。

**🔧 技术方法**

技术方法包括：1) 双向非因果Transformer backbone；2) MaskGIT式置信解码器（每层固定T步）；3) 预训练语音编码器Mimi与WavLM；4) 采用“pre-registered”非线性幂律模型和AICc比较；5) 通过多重自举重采样评估比例统计；6) 结合多语者数据集与ASR基准Whisper-large-v3进行floor参考。

**📊 数据集**

使用的数据集为Emilia-EN的speaker-stratified子集（约数千名说话人、4–15秒目标句子），包含约h小时语音；同时使用Whisper-large-v3作为可懂度floor，使用Mimi编码器进行codec-ceiling；评估跨句零样本合成。

**📈 对比分析**

比较方法：对不同w、d、T组合分别计算WER和SIM-o，并以floor值为基准，计算“可达范围占比”；使用AICc比较可分离与替换模型的拟合优度；对最佳候选搜索（best‑of‑K）与单步推理进行相同NFE下的WER/identity比较。性能结果显示：当T从1升至16时，可懂度的可达范围占比从32%提升至86%，而身份的可达范围占比从20%提升至46%，两者的比例约为1.86；训练计算增加3×可懂度进一步提升到约97%，身份提升到约99%；最佳候选搜索在保持WER不变的情况下可将身份提升至≈50%。

**⚠️ 局限性**

局限性包括：①模型在实验中训练不足（30k步、参数上限1.1M），可懂度仍在下降；②身份提升受codec本身误差限制（codec round‑trip与真实音频差距约40%）；③仅针对单一语种、单一codec与单一数据集；④搜索实验受限于使用的说话人编码器家族（ECAPA等），可能存在代表性偏差；⑤部分拟合参数（κ、宽度幅度）未完全识别，导致部分结论不完全可推广。

---

## 583. JOVE: Joint Execution and Verification for Resource-Aware LLM Task Graphs

**arXiv ID:** 2610.03296 | [PDF](https://arxiv.org/pdf/2610.03296v1)

**作者:** Haoran Zhang `[一作]` (University of Texas at Austin), Haris Vikalo `[通讯]` (University of Texas at Austin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了 JOVE 框架，用在线混合整数线性规划同时分配 LLM 执行器和选择需要付费验证的中间节点，以在多步推理任务图中平衡即时准确性与未来学习。

**💡 创新点**

创新点在于：①将执行与选择性验证联合建模为受长期预算和查询时延约束的在线优化问题；②利用信息增益奖金将付费验证的学习价值直接纳入决策；③在不依赖离线标签、仅通过异步验证获得节点级反馈的前提下实现子线性质量学习。

**🔧 技术方法**

核心技术包括：基于任务图的 LLM 调度；在线估计模型质量、成本和时延；LinUCB 风格的置信上界预测；D‑optimal 信息增益度量；每查询求解 MILP 的自适应预算价格与时延约束；以及理论证明的子线性学习误差。

**📊 数据集**

使用了四个推理基准：Bamboogle（多跳问答）、MMLU‑Pro（跨学科推理）、GPQA（科学推理）和 LiveBench‑Reasoning（结构化推理）。

**📈 对比分析**

与传统无约束推理基线（Direct、CoT、SoT、Plato）以及资源感知基线（WR‑Online、Knapsack‑ascend/descend）对比。JOVE 在保持与最强基线相当的准确率（约 49.5%）的同时，平均成本降低 3.7–16.3 倍、时延降低 8.3–17.0 倍；相比资源感知基线，准确率提升约 9–10 分，且在预算/时延约束下实现更稳定的资源利用。

**⚠️ 局限性**

局限性包括：①需要先验任务图生成器，且图结构对最终准确性影响较大；②验证成本与预算约束下的权衡依赖于手工设定的信息增益权重 k_v，过大或过小均可能影响性能；③在极端高时延/低预算场景下，信息增益估计误差可能导致约束违反；④目前仅支持节点级成功判定，无法直接对最终答案进行回报学习。

---

## 584. Training-Loss Guarantees for Muon with Finite-Step Newton--Schulz Orthogonalization

**arXiv ID:** 2610.03306 | [PDF](https://arxiv.org/pdf/2610.03306v1)

**作者:** Amartya Roy `[一作]` (School of Interdisciplinary Research), Souvik Chakraborty `[通讯]` (Department of Applied Mechanics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明了全批 Muon（带动量、五步调优的 Newton–Schulz 变换）在宽两层 ReLU 网络上能在有限步数内达到任意给定训练损失，并给出了宽度、学习率与迭代上界；

**💡 创新点**

仅利用 Newton–Schulz 变换的对齐与谱范数上界两条性质，即可获得 ε⁻¹ᐟ² 的迭代上界，避免了对近似正交化的严格需求；

**🔧 技术方法**

结合近似正交化、Newton–Schulz 迭代、神经切线核理论以及梯度-更新对齐分析等技术；

**📊 数据集**

使用的是高斯输入、随机 ReLU 教师生成的教师‑学生数据集（20 个样本、100 维、α=1 的输出权重）；

**📈 对比分析**

通过宽度、动量、学习率的多维实验，对比了 Muon 与经典 Newton–Schulz 及极性因子，实验显示所有 30 次实验在理论阈值远低的宽度下均能在 O((1‑μ)⁻¹ε⁻¹ᐟ²) 步内达到目标损失，损失下降曲线与理论一致；

**⚠️ 局限性**

限制包括理论宽度阈值极大（m≈2×10²⁷），仅适用于靠近初始化的全批训练，未考虑小批量梯度、数值误差以及深层网络的推广；

---

## 585. Architecture-Dependent Fusion Pathways in MLLMs

**arXiv ID:** 2610.03289 | [PDF](https://arxiv.org/pdf/2610.03289v1)

**作者:** Hebao Zhu `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Dongxia Wu `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对齐解耦、注意力路由与熵、内在维度三阶段分析以及因果干预，系统揭示并验证了两类MLLM的跨模态融合路径。

**💡 创新点**

首次提出“融合路径”概念，并将对齐解耦、注意力路由、内在维度估计与因果干预结合，全面解释不同架构在层次上如何进行视觉与文本的融合。

**🔧 技术方法**

利用Centered Kernel Alignment (CKA)、对齐解耦、注意力路由与熵统计、内在维度估计、因果干预（文本目标掩码、噪声注入）以及视觉CKA热图等技术。

**📊 数据集**

COCO-val、VQA、MMMU、MMBench（VLMEvalKit）和POPE等公开数据集。

**📈 对比分析**

通过跨模型对比和因果干预验证，concat模型表现出“文字先，视觉后”的融合路径，native模型表现为早期视觉-文本共适应；在POPE和MMBench上显示出不同的鲁棒性差异。

**⚠️ 局限性**

局限于仅评估公开模型、训练流程差异难以完全对齐、干预方法仅覆盖文本掩码和噪声注入，未覆盖所有可能的融合机制。

---

## 586. Geometry Meets Physics: Data-Efficient Pre-Training for Unstructured Neural PDE Solvers

**arXiv ID:** 2610.03363 | [PDF](https://arxiv.org/pdf/2610.03363v1)

**作者:** Luis Medrano-Navarro `[一作]` (Technical University of Munich), Nils Thuerey `[通讯]` (Technical University of Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种无磁盘存储的预训练框架GMP，用在线生成几何和物理数据来训练可处理无结构3D网格的PDE代理模型。

**💡 创新点**

创新点在于：①几何预训练利用随机几何体、法向、曲率和体积VDF等内在描述符；②物理预训练采用在线谱求解器生成简化PDE轨迹；③框架模块化，可分别针对稳态和瞬态任务；④实现了完全无离线存储、计算成本低的预训练。

**🔧 技术方法**

使用技术包括随机几何生成、Gaussian去畸变网格、光谱求解器、Masked AutoEncoder、Transformer/Graph Attention结构（SMART、AB‑UPT、Transolver++等）、多编码器解码器架构以及ConFIG优化器。

**📊 数据集**

评估数据集包括：稳态：DrivAerML（汽车）、SHIFT‑Wing（机翼）、SHIFT‑Crash（结构）；瞬态：2D KS、2D Ellipse、SHIFT‑Crash 3D；预训练阶段不使用任何外部存储数据。

**📈 对比分析**

与从零训练和GeoPT预训练进行对比；在低数据（如16样本）下稳态任务提升高达47%，收敛速度提升显著（仅4h预训练 vs 80h原始训练）；瞬态任务1步NRMSE降低10–25%，Rollout误差下降约20–30%；整体性能优于传统基线。

**⚠️ 局限性**

局限性包括：在极低数据场景下部分模型易过拟合（如AB‑UPT无明显提升）；几何描述符选择仍可进一步优化；对复杂耦合物理或高维多通道任务的泛化尚待验证；预训练仍需要一定GPU资源。

---

## 587. EVEWorld: Physical Evolution Supervision for Embodied World Models

**arXiv ID:** 2610.03374 | [PDF](https://arxiv.org/pdf/2610.03374v1)

**作者:** Kaiqi Wang `[一作]` (China Merchants Group), Jiaxing Zhang `[通讯]` (China Merchants Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EVEWorld 物理演化监督框架，通过实例导向恢复 (IGR) 与时间实例对齐 (TIA) 两个模块，显著降低 embodied world 模型的“模型懒惰”现象，提升目标实例的一致性与跨帧连续性。

**💡 创新点**

创新点在于：①引入 Model Laziness Rate (MLR) 量化指标；②通过 IGR 给出实例级恢复监督，解决实例计数漂移；③通过 TIA 在 transformer 层内实现跨帧实例对齐，提升时间一致性；④两者结合实现了全流程的物理演化监督，显著提升了视频生成质量和指令遵循能力。

**🔧 技术方法**

使用技术包括：视频扩散变换器（VAE 编码 + EDM 风格噪声权重）、GroundingDINO + 语言模型定位目标实例、拼贴数据增强、实例加权恢复损失、跨帧残差补偿与软对齐、CFG 感知训练策略、MLR 评估脚本等。

**📊 数据集**

使用的数据集与评测基准包括：DreamGenBench、EWMBench、WorldArena 1.0/2.0、PBench、以及官方的多项视频生成与机器人学习评测指标（Qwen-IF、Gemini-IF、EWMScore、JEPA Similarity 等）。

**📈 对比分析**

在 DreamGenBench 上与标准 SFT、CogVideoX、Wan、Cosmos、GigaWorld-0 等对比，EVEWorld 将 MLR 从 11.11% 降至 1.59%（85.7% 降低），Qwen-IF 与 Gemini-IF 分别提升约 6% 与 7%；在 WorldArena 2.0 Track 1 榜单上排名第 17 并在 JEPA Similarity 上排名第 6；在 EWMBench、PBench 等跨分布与跨后骨干的评测中亦获得明显的指标提升。

**⚠️ 局限性**

局限性：监督仅为局部对偶，未能显式建模长时序依赖或复杂多物体交互；对非实例变化任务（如裁剪、组装）不适用；模型仍需外部定位/检测器在训练阶段支持，推理时依赖模型内部对齐；未来工作需扩展更全面的演化监督和长期推理机制。

---

## 588. Harmonic Eigenspace: A Web-based Application for Navigating and Composing Microtonal Harmony

**arXiv ID:** 2610.03398 | [PDF](https://arxiv.org/pdf/2610.03398v1)

**作者:** David Dalmazzoa `[一作]`, Ken Déguernel `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一个基于四维心理声学空间的 Web 应用，用于导航与创作微分音和弦。

**💡 创新点**

将和弦类型映射到基于粗糙度的四维谐振空间，提供不受调式限制的可视化和可播放的“谐波本征空间”，并扩展了 31‑TET 与 53‑TET 的模态互换工具。

**🔧 技术方法**

利用 Sethares 的粗糙度模型构建四维不协和体，使用 Plotly 实现 3D 交互可视化，MIDI/MPE 进行实时演奏，前端采用 JavaScript/ Web Audio。

**📊 数据集**

使用六个等幅谐波音色的分量来计算不协和度，并公开了 31 位听力测试的音频与 jsPsych 实验代码（链接给出）。

**📈 对比分析**

通过 31 位参与者的听力实验，将模型预测的粗糙度与主观不协和度评分进行相关分析（Spearman ρ≈0.48、Pearson r≈0.57），并对 10 条 53‑TET 进程的连贯性、音乐性等维度进行评估，结果显示进程在可接受性上与 12‑TET 近似和弦相近。

**⚠️ 局限性**

样本量小、受试者对微分音经验差异未控制，模型仅针对谐波音色，对非谐波乐器的适用性需进一步研究。

---

## 589. Randomization Beyond Deterministic Adaptivity in Parallel Sampling

**arXiv ID:** 2610.03335 | [PDF](https://arxiv.org/pdf/2610.03335v1)

**作者:** Da Li `[一作]` (Dalian University of Technology), Xiaopeng Wei `[通讯]` (Dalian University of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在三轮硬性上限下的并行采样器，比较确定性自适应策略与随机混合策略在生成已知目标分布时的前向KL误差与总变差。

**💡 创新点**

创新点在于给出一个完整支持的二进制族，证明任何确定性自适应三轮策略的KL误差线性增长，而四个固定随机计划的混合能实现指数级减小误差；并提出“覆盖”引理阐明随机混合如何重建目标分布，进一步在匹配族上展示了随机策略对圆周度提升的优势。

**🔧 技术方法**

主要技术包括信息谱分析、条件总相关（conditional total correlation）与KL分解、覆盖矩阵与密度覆盖证明、以及利用随机子空间切分和二项分布的概率上界。

**📊 数据集**

使用的“数据集”是人工构造的离散分布：9点仿射平面上的二进制族以及2^k维匹配族（每个标量对应一个二进制选择符），均满足已知条件边缘概率。

**📈 对比分析**

通过理论证明和数值实验（如维度389的有限和，r=43时的KL分离），结果表明随机混合策略在KL误差上可降到指数量级，且总变差上可逼近1/4，而确定性策略则线性递增；在匹配族中，随机策略在三轮内即可达到与理想匹配相当的精度，而确定性策略需增长到Ω(n/ log n)轮。

**⚠️ 局限性**

局限性包括仅针对已知、有限支持且能精确抽取条件边缘的分布；随机化需要额外的随机种子（仅两比特），并且目前仅证明了特定结构族的优越性，尚未推广到一般连续或高维分布，且对实际硬件实现的可行性未做评估。

---

## 590. SCAD: Structured Credit Assignment and Distillation for Long-Horizon Agents

**arXiv ID:** 2610.03372 | [PDF](https://arxiv.org/pdf/2610.03372v1)

**作者:** Shangyang Wu `[一作]` (Beijing University of Posts and Telecommunications), Haoran Luo `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了 SCAD（Subtask‑Local Distillation with Cross‑Rollout Planning Credit）框架，用以训练长时序智能体解决文本与多模态长任务。

**💡 创新点**

核心创新在于三点：① 在每个子任务内进行局部的教师引导蒸馏；② 通过子任务-报告前缀树对规划决策赋予跨滚动的信用；③ 在规划与执行之间采用结构化信号分配，使规划获得完整终端信用，执行仅获取正向终端信用与局部教师反馈。

**🔧 技术方法**

技术手段包括基于上下文切片的局部教师监督、子任务聚类与报告规范化、树形信用估计与先验收缩、GRPO 与 PPO 双重剪裁的优势计算、以及教师与学生分布间的逆 KL 损失。

**📊 数据集**

实验使用了 11 个文本数据集（共 814 题）和 8 个多模态数据集（共 510 题），其中包含 ID 与 OOD 子集，涵盖 Bamboogle、PopQA、FVQA 等公开基准。

**📈 对比分析**

与 RL、OPD、ATOD、HyperEyes 等基线对比，SCAD 在文本任务上宏观准确率提升至 46.10%（比强基线高 4.48 个百分点），在多模态任务上提升至 33.13%（比强基线高 4.19 个百分点）。训练时间约比 ATOD 低 37.96%，教师监督效率亦更高。

**⚠️ 局限性**

局限性包括：教师监督仍受固定预算限制，未能自适应地在子任务间分配监督；使用的教师模型固定不更新；实验仅覆盖现有数据集与任务，未验证在更大或更异构环境下的泛化能力。

---

## 591. S$^{2}$-PINN: Stochastic Separable Physics-Informed Neural Networks

**arXiv ID:** 2610.03303 | [PDF](https://arxiv.org/pdf/2610.03303v1)

**作者:** Zhendong Li `[一作]` (Lehigh University), Akwum Onwunta `[通讯]` (Lehigh University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种名为 S^2-PINN 的随机偏微分方程不确定性量化方法，能够同时学习空间、时间与随机特征。

**💡 创新点**

创新点在于将可学习的高斯空间字典、傅里叶时间特征和正交 gPC 随机基底通过低秩 CP 张量耦合，实现了三维可分离结构，并通过混合强形式残差与 gPC 投影损失提升精度和校准。

**🔧 技术方法**

采用的技术包括可分离式 PINN、低秩 CP 张量分解、gPC 正交多项式、傅里叶时间特征、正交化正则化、混合残差损失以及 Adam 优化器。

**📊 数据集**

使用的数据集包括四个制造的随机 PDE 基准（扩散、Allen–Cahn、Burgers、Darcy）以及两个非制造的 Poisson 与 Darcy 反问题，和一个 2D 随机 Navier–Stokes 基准。

**📈 对比分析**

与九种基线（标准 PINN、PI‑DeepONet、MC‑Dropout、深度集成、SC、SPINN+Z、CP‑PINN、Neural Chaos）以及 PC^2 进行对比，S^2‑PINN 在均值/方差 L^2 误差、校准覆盖率、参数量和训练时间等指标上均显著优于基线。

**⚠️ 局限性**

局限性包括：仅适用于规则域，缺乏复杂几何的字典适配；理论上未给出有限秩和截断下的全局误差保证；在非多项式随机系数（如对数正态）时投影正则化可能引入偏差。

---

## 592. T3lescope: Arbitrary-Resolution High-Fidelity Generative Surface Reconstruction from Images

**arXiv ID:** 2610.03308 | [PDF](https://arxiv.org/pdf/2610.03308v1)

**作者:** Atsuhiro Noguchi `[一作]` (Preferred Networks, Inc.), Eiichi Matsumoto `[通讯]` (Preferred Networks, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

利用单一固定分辨率的3D生成器，结合推理时的粗到细级联和多尺度图像条件，直接从多视角已标定图像重建高保真三维网格，无需对每个场景进行单独优化。

**💡 创新点**

1) 在推理阶段通过粗到细级联复用同一模型，灵活决定场景覆盖范围与分辨率；2) 细胞无关的多尺度图像条件，支持任意位置与尺度；3) 采用父子几何传递与 SDEdit 级联，提升细节恢复。

**🔧 技术方法**

3D生成器（TRELLIS.2）+ VAE + 反扩散 Transformer + DINOv3/ConvNeXt 多尺度特征 + voxel‑queried attention + SDEdit 级联 + 多尺度图像金字塔。

**📊 数据集**

ScanNet++、Tanks & Temples、作者自研城市街区数据集；以及用于训练的合成数据集 SAGE‑10K、3D‑FRONT 等。

**📈 对比分析**

与 2DGS、MIlo、GaussianWrapping、CityGaussianV2、DA3、MapAnything、Murre、GenRecon 等基线相比，在 ScanNet++、Tanks & Temples、城市街区上取得更低的 Chamfer 距离、更高的 F‑score 以及更佳的法向一致性；在稀疏视角下与优化方法相当或更优，并能更好恢复透明、镜面表面。

**⚠️ 局限性**

推理时间相对较长（每场景数十分钟），受限于单一分辨率模型，难以处理无限场景；对远距离物体和大规模稀疏观测的重建仍有偏移，且对无标定图像不适用。

---

## 593. Beyond Entropy: Self-Diagnostic Multi-Role Token Optimization for Video Reasoning

**arXiv ID:** 2610.03400 | [PDF](https://arxiv.org/pdf/2610.03400v1)

**作者:** Yudong Han `[一作]` (Beijing Institute of Technology), Liyuan Pan `[通讯]` (Beijing Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出DyCPO框架，通过多角色token选择与自诊断rollout对比干预实现视频推理中的token级信用分配。

**💡 创新点**

创新点包括：①多角色依赖度量同时考虑视觉与答案敏感性；②动态自生成的counterfactual信号与策略共进化；③对比正负视图的token级正则化；④EMA自教师动态生成关键帧对比。

**🔧 技术方法**

技术：多角色token权重计算、对比学习、强化学习（GRPO）与token级对比正则、EMA自教师、rollout对比关键帧定位、奖励归一化与优势门控。

**📊 数据集**

数据集：VideoRFT-CoT-102K用于SFT；3K样本的Video-R1子集用于RL；评测基准包括Video-Holmes、VideoMMMU、MMVU、VideoMME、TempCompass、LVBench、VSIBench、VidHalluc。

**📈 对比分析**

与多种基线对比，DyCPO在Video-Holmes上提升至44.8%（相较基线41.5%），在MMVU达68.5%（相较64.3%），在VideoMME 62.6%（相较60.3%）等，整体取得SOTA或与顶尖模型相当，显著优于GRPO、TW‑GRPO、Video‑OPSD等。

**⚠️ 局限性**

局限性：依赖于精细的counterfactual设计，易受奖励稀疏和难度不匹配影响；对极难或极易任务表现不佳；训练复杂度高，需大规模RL数据；仍需进一步验证跨域泛化。

---

## 594. Low-Density Parity-Check Codes of High Girth from Permutation Polynomials

**arXiv ID:** 2610.03341 | [PDF](https://arxiv.org/pdf/2610.03341v1)

**作者:** Tasawwar Hussain `[一作]` (University of South Florida), Tefjol Pllaha `[通讯]` (University of South Florida)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了基于高阶置换多项式构造的原型图低密度奇偶校验（LDPC）码，旨在超越传统QC-LDPC码的最小距离和圈长上限，同时保持简单的表示形式。

**💡 创新点**

创新点在于使用非交换的二项式置换多项式构造原型图码，从而提高了码的圈长和最小距离，改善了迭代解码性能。

**🔧 技术方法**

采用了高阶置换多项式，特别是单项式和二项式置换多项式，这些多项式在组合和逆运算下是封闭的，便于进行圈长和循环分析。

**📊 数据集**

使用了有限域上的多项式作为数据集，具体包括𝔽_q中的置换多项式，进行仿真比较。

**📈 对比分析**

与基于仿射置换多项式的构造相比，仿真结果显示本文提出的基于置换多项式的码在解码性能上有显著改善，尤其在圈长和最小距离方面表现更优。

**⚠️ 局限性**

限制在于所提出的构造仍需进一步探索更复杂的置换多项式，以期获得更显著的编码增益，同时对量子LDPC码的研究也需要进一步深入。

---

## 595. A Fully Automatic Pipeline for 3D Dendrite Instance Segmentation in SBF-SEM

**arXiv ID:** 2610.03332 | [PDF](https://arxiv.org/pdf/2610.03332v1)

**作者:** Zewen Zhuo `[一作]` (University of Eastern Finland), Jussi Tohka `[通讯]` (University of Eastern Finland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

构建了一个全自动的三维树突实例分割流水线，能够在SBF‑SEM图像中从零开始完成树突的检测、分割、跨切片链接和高分辨率细节恢复，支持在控制与癫痫模型的大脑海马CA1区进行大规模重建。

**💡 创新点**

创新点在于将YOLOv6检测与Promptable SAM分割无缝结合，采用迭代二维掩模优化与随机森林跨切片链接，最终通过nnU‑Net实现高分辨率实例感知细节恢复，实现了模块化、无人工提示、可端到端推理的完整系统，且不依赖大量三维标注或联合端到端训练。

**🔧 技术方法**

核心技术包括YOLOv6目标检测、SAM（Promptable 2D分割模型）、迭代掩模后处理、随机森林（RF）跨切片链接、nnU‑Net高分辨率二值语义分割与实例恢复。

**📊 数据集**

使用了两套小型动物SBF‑SEM数据集：一套来自健康大鼠的海马CA1区（控制组），另一套来自pilocarpine诱导癫痫大鼠的同一区域（癫痫组）。

**📈 对比分析**

在10个稀疏标注切片上评估，与稀疏语义Dice（控制0.93，癫痫0.91）、IoU（控制0.87，癫痫0.83）相比，实例层面的S_inst分别为0.63/0.38，检测质量DQ从0.86降至0.51，整体Panoptic PQ从0.70降至0.37，说明在控制组下语义与实例分割均优异，癫痫组主要受检测召回不足影响。

**⚠️ 局限性**

局限性包括：①检测在稠密癫痫组织中的召回率低导致实例识别错误；②训练样本仅为30个切片，缺乏对不同动物或扫描条件的泛化验证；③RF链接特征依赖像素尺度，跨分辨率迁移需重训练；④未提供精确度评估，因标注稀疏导致误差判定受限。

---

## 596. EdgeAgent: Orchestrating On-Device LLM inference for End-User Multi-Agent Systems on CPU-GPU Unified Memory Architectures

**arXiv ID:** 2610.03394 | [PDF](https://arxiv.org/pdf/2610.03394v1)

**作者:** Yuhai Long `[一作]` (Sun Yat-sen University), Jiangsu Du `[通讯]` (Sun Yat-sen University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了EdgeAgent系统，专门为Edge UMA设备上的多智能体LLM推理设计，解决内存绑定位解码阶段的总线争用和多代理工作负载碎片化执行问题。

**💡 创新点**

①采用UMA-aware零拷贝Tensor Parallelism，利用不对称内存布局实现CPU‑GPU无锁并行；②引入Agent‑aware调度层，基于历史接受长度(HAL)动态分配draft预算；③实现异步suspend‑and‑yield机制，在工具调用停滞时抢占并回收计算槽，三者协同显著提升并发吞吐和时延。

**🔧 技术方法**

ARMv9 SME2微核优化（定制SME GEMM）；异步图编译与逻辑依赖Barrier；Zero‑copy TP；Speculative Decoding（EAGLE‑3）；动态draft预算分配（HAL+温度Softmax）；suspend‑and‑yield上下文切换；trace‑driven多代理工作负载生成。

**📊 数据集**

LLM模型：DeepSeek‑R1‑Distill‑Llama‑8B、Llama‑3.1‑8B‑Instruct；数据集：LongBench（推理任务）、MBPP（结构化输出）、ToolBench（工具调用）；通过合成trace混合这些任务来模拟多代理工作负载。

**📈 对比分析**

与Batch‑AR、Seq‑SD (EAGLE‑3)及Batch‑SD (Batched EAGLE‑3)等基线对比。EdgeAgent在Apple M4 SoC上：UMA‑aware层实现1.29×加速；加入HAL调度后进一步提升1.05‑1.17×；在极端工具停滞场景下总体加速达1.77×。吞吐量可达DeepSeek 33.6 tok/s、LLaMA 28.0 tok/s。

**⚠️ 局限性**

仅在共享内存的UMA SoC上最优，CPU‑GPU零拷贝依赖统一地址空间；在离散GPU或PCIe环境中需重构；SME微核优化专属ARM，难以直接迁移到x86/NVIDIA；对GPU性能敏感；实验基于自制trace，真实多代理场景的多样性与鲁棒性待进一步验证。

---

## 597. SyntaxBench: A Statistical Diagnostic Framework for Character-Level Reasoning in Large Language Models

**arXiv ID:** 2610.03329 | [PDF](https://arxiv.org/pdf/2610.03329v1)

**作者:** Mohsen Larni `[一作]` (University of Nevada), Kazem Taghva `[通讯]` (University of Nevada)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本工作提出并实现了 SyntaxBench 基准，包含六个确定性的字符级任务，并构建了完整的统计评估框架；

**💡 创新点**

创新点在于：① 通过英文与长度匹配的随机字符串对比，首次揭示 tokenization 对字符任务性能的主导作用；② 提供了从 Bootstrap CI、Cohen κ、McNemar、Kendall τ 到多比较校正等一整套统计工具，打破仅用整体准确率的局限；③ 系统评估“思考模式”(chain‑of‑thought) 在字符推理中的加减效应；

**🔧 技术方法**

使用技术包括：零/一/四示例提示、思考/非思考两种推理模式、Tokenizer 细分分析、Bootstrap 置信区间、配对 McNemar、Cohen κ、Kendall τ、χ² 检验、Benjamini–Hochberg FDR、解析错误率等统计方法；

**📊 数据集**

数据集包含：NLTK 词表、Wiktionary 词典的回文词表、Wikipedia 摘录作为英文输入；随机字符串按相同字符长度从 a‑z 随机生成；压力测试使用 200–500 词长的英文与随机文档；

**📈 对比分析**

比较方法：在 8 款 2B–32B 的开源 LLM（含思考/非思考两模式）上进行 0/1/4 shot 评测，报告 EMA、RA、PER、κ 等指标，并通过配对检验和多比较校正确定显著性。结果显示 tokenization 对英/随机差距产生主导影响，思考模式效果不一，最长子串提取任务在所有模型中几乎全败，部分模型已在其余任务接近饱和；

**⚠️ 局限性**

局限性：仅评估 2–32B 开源模型，未覆盖 100B+ 或闭源模型；仅在英文与随机字符上实验，未检验多语种/非拉丁文字；后端服务差异可能影响细节；任务范围局限于单步确定性字符操作；统计检验假设与思考模式混杂，未完全分离训练差异。

---

## 598. To Jev or Not? Evaluating the Accuracy and Efficiency of Structured Decision Models for Hate-Speech Moderation

**arXiv ID:** 2610.03324 | [PDF](https://arxiv.org/pdf/2610.03324v1)

**作者:** Demetris Paschalides `[一作]` (University of Cyprus), Marios D. Dikaiakos `[通讯]` (University of Cyprus)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了六种结构化决策模型在不同仇恨言论数据集上的零样本性能，并与专业化模型、零射击、商业LLM以及监督分类器进行对比；

**💡 创新点**

验证了提供定义与按规则拆解判断对模型准确率的实际影响，展示了低成本高质量的决策方案；

**🔧 技术方法**

使用了结构化决策框架（Jev、Laya、Decider、SemIf、Bespoke‑Nimble）、商业LLM（Luna、Sol）及监督模型（TF‑IDF+LR、DistilBERT），并采用规则拆解与直接决策两种推理方式；

**📊 数据集**

实验覆盖四个英文仇恨言论数据集：Dynamic、MHS、HateXplain（代理定义）和 HateCheck；

**📈 对比分析**

通过宏F1、精确率/召回率、跨数据集迁移、定义敏感度、属性预测、拆解对比以及推理时延与费用测量进行多维比较；结果显示：商业LLM在3/4数据集上占优，最优决策模型（如Jeva）在HateCheck上仅1.6宏F1点差距且成本降低约97%；定义和拆解在大部分情况下并未显著提升，且拆解往往带来额外成本；

**⚠️ 局限性**

主要局限包括：所有数据为英语，缺乏跨语言验证；测试集主要是人工挑选的功能性样本，日常流量表现未知；未对模型对照的“定义遵从度”进行人类评估；缺乏对拆解结果正确性的标注；零样本评估不排除模型在数据集上的潜在预训练泄漏。

---

## 599. DriftTTS: Few-Step Text-to-Speech Without Distillation via Distribution-Matching Drift

**arXiv ID:** 2610.03390 | [PDF](https://arxiv.org/pdf/2610.03390v1)

**作者:** Mohammad Nur Hossain Khan `[一作]` (University of Massachusetts Amherst), Bashima Islam `[通讯]` (University of Massachusetts Amherst)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 DriftTTS，一种在无生成教师、无蒸馏、无对抗训练的条件文本到语音模型。

**💡 创新点**

创新点在于将漂移模型的分布匹配目标应用于 TTS，并引入可控步数的 on‑policy roll‑out 训练策略。

**🔧 技术方法**

采用分布匹配漂移目标、冻结的 MelMAE 预训练特征、卷积 Transformer 解码器以及 HiFi‑GAN 语音合成技术。

**📊 数据集**

使用 LJSpeech 数据集进行训练和评估。

**📈 对比分析**

与 Matcha‑TTS、Grad‑TTS 在 LJSpeech 上比较，DriftTTS 在 NFE=4 时实现 3.87 dB MCD、3.7% WER、4.18 MOS，性能与 Matcha‑TTS 相近且自然度接近真实录音。

**⚠️ 局限性**

局限性包括只能在训练深度 K 内使用（NFE ≤ K），对更长语句的泛化有限，且缺乏跨语种或更大规模的实验验证。

---

## 600. ReFract: Benchmarking Perspective Awareness in Language Model Agents with Text World Models

**arXiv ID:** 2610.03356 | [PDF](https://arxiv.org/pdf/2610.03356v1)

**作者:** Hainiu Xu `[一作]` (Amazon), Luca D'Angelo `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个可执行的文本世界模型（Text World Model）和基准数据集（150个专家验证的工业维护场景），用于评估大语言模型（LLM）在不同用户角色（如技术员、经理等）下的“视角意识”（Perspective-Awareness）表现。

**💡 创新点**

首次把视角意识拆解为“视角获取”（Perspective‑Taking）与“视角路由”（Perspective‑Routing）两大维度，并通过可执行模型量化这两种能力；同时设计了面向角色的工具使用与权限门控机制，形成了面向角色的完整评测框架。

**🔧 技术方法**

采用POMDP框架定义任务、手工编写的域文件和问题文件、Python实现的可执行文本世界模型、LLM的工具调用（Function Calling、ReAct）以及多种评估指标（通过率、角色合规率、超越/低估程度、效率等）。

**📊 数据集**

使用从工业维护支持会话中匿名提取的查询构建的150条基准数据，每条数据对应的文本世界模型均由领域专家核对并验证，形成了包含角色、工具、知识与权限门控的完整评测数据集。

**📈 对比分析**

对比了多种公开与专有LLM（包括 32B、80B、235B、5.5、5.6 等）在两种工具环境（查询特定工具集 QTS 与完整工具集 FTS）下的表现。结果显示最强模型在 QTS 下最高通过率约为 68.5%，但在 FTS 下降至 46.6%；所有模型都存在“过度谨慎”或“越权”问题，专有模型在视角合规率和通过率上优于公开模型。

**⚠️ 局限性**

局限性包括：依赖人工专家手工构造域文件和问题文件，难以大规模自动化；基准聚焦工业维护场景，缺乏跨领域验证；评测仅考虑工具调用层面，未深入分析语言理解与决策细节；目前视角意识仍未得到充分解决，模型普遍表现欠佳。

---

## 601. Symbolic Execution of Constrained Horn Clauses

**arXiv ID:** 2610.03345 | [PDF](https://arxiv.org/pdf/2610.03345v1)

**作者:** Johannes Weiser `[一作]` (TU Vienna), Philipp Rümmer `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文将受限 Horn 子句（CHC）的约束解析作为符号执行的逻辑框架，展示了正向与反向符号执行分别对应于约束正向单元超解析与 SLD 解析的关系；

**💡 创新点**

创新点在于（1）将约束解析与符号执行统一，阐明正向/反向解析对应传统符号执行；（2）通过反向解析与子句子集化证明 k‑induction 的泛化；（3）揭示约束解析与错误逻辑（incorrectness logic）的对应关系；（4）给出多种可判定的子句子集化判别方法；

**🔧 技术方法**

核心技术包括约束解析（取代统一为等式约束）、单元超解析（forward）与超 SLD 解析（backward）、子句子集化判别、以及基于 SMT 求解器的理论求解；

**📊 数据集**

实验使用了 CHC‑COMP 2026 benchmark 组的 9 类数据集（如 ADT‑LIA、LIA‑Arrays、BV‑Lin 等），共计数千条实例；

**📈 对比分析**

与 CVC4 的 CEGAR 引擎比较，正向/反向符号执行在多类 benchmark 上各自取得独特的 satisfiable/unsatisfiable 解决实例，且两者互补；总体上，符号执行在某些类别上超过 CEGAR，且实现更简单；

**⚠️ 局限性**

局限性包括：仅实现了基于语法相同的子句子集化；DFS 策略仅适用于线性 CHC；未实现高级搜索启发式（如 concolic、迭代加深等）；对非线性目标的模型构造仍不完整；在某些深度分支上可能无限循环导致不终止。

---

## 602. Follow the Winners: Conservative Policy Improvement with the Cross-Entropy Method for Critic-Free RFT

**arXiv ID:** 2610.03361 | [PDF](https://arxiv.org/pdf/2610.03361v1)

**作者:** Joery Ariën de Vries `[一作]` (Trent AI Limited), Zhenwen Dai `[通讯]` (Trent AI Limited)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种无价值网络的强化学习微调算法 Follow the Winners（FTW），通过在回放缓冲区上采用阶梯式精英筛选和交叉熵投影，实现了多样化、稳健的策略更新；

**💡 创新点**

创新点在于将控制-as-推断框架扩展为基于批次序列的归因，恢复了 DPO 与 GRPO 的理论关系，并通过精英比例控制风险偏好，从而在不使用价值模型或多次 roll‑out 的前提下实现多项式收敛与低方差；

**🔧 技术方法**

主要技术包括：控制-as-推断（CAI）框架、交叉熵方法、阶梯精英过滤、回放缓冲区提议分布、对数顺序概率（Plackett‑Luce）与多项式潜能的分析；

**📊 数据集**

实验数据集包括 Sokoban（视觉盒子推理）、Search‑R1（检索问答）、Brax HalfCheetah 以及若干诊断性 bandit 任务；

**📈 对比分析**

与 GRPO、PPO、REINFORCE++ 等基线对比，FTW 在 Sokoban 与 Search‑R1 上的成功率/准确率与 GRPO/PPO 相当，同时在训练稳定性和 seed 方差上优于 REINFORCE++，并显著降低了 GPU/内存开销；

**⚠️ 局限性**

局限性：风险偏好在高度随机环境下可能导致对平均最优策略的偏移；回放缓冲区会增加 CPU 内存占用；假设轨迹可全序的 IIA 条件在某些非传递性任务中需进一步调整。

---

## 603. Sparse Distributed Fiber Optic Sensor Placement in Metropolitan and Datacenter Elastic Optical Networks

**arXiv ID:** 2610.03338 | [PDF](https://arxiv.org/pdf/2610.03338v1)

**作者:** Sleman Mouammar `[一作]` (Technische Universitât Braunschweig), André Drummond `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了基于分布式光纤传感（DFOS）的弹性光网络（EON）架构，并评估稀疏传感器部署对大都市和数据中心网络故障恢复的影响。

**💡 创新点**

创新点在于设计了贪心最短路径覆盖（GSPC）算法，使得仅部署5%传感器即可实现接近全覆盖的服务恢复效果，并给出适用于不同网络的预测窗口时长建议。

**🔧 技术方法**

使用了DFOS光学信号、SDN控制平面、k最短路路由、FirstFit光谱分配以及假设已训练的机器学习故障预测模型。

**📊 数据集**

利用ION、Catalunya和NovaCube三种真实网络拓扑，仿真Poisson流量、随机失败、预测窗口等数据集进行实验。

**📈 对比分析**

通过仿真比较带宽阻塞率（BBR）、受影响电路数、停机时间等指标，发现GSPC在5%部署下可将受影响电路数降低约80%，停机时间降低约70%，与全覆盖差距不大。

**⚠️ 局限性**

局限性包括仅考虑单一故障场景、预测误差为零、未分析多故障、误报和漏检情况，以及缺乏实时硬件验证。

---

## 604. A population-level assessment framework for flood-related wellbeing from public discourse in Ireland

**arXiv ID:** 2610.03334 | [PDF](https://arxiv.org/pdf/2610.03334v1)

**作者:** Róisín Luo `[一作]` (University of Galway), Karyn Morrissey `[通讯]` (University of Galway)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于公共话语的洪水相关福祉评估框架，利用三维负面构造（痛苦、功能中断、制度疏离）对爱尔兰公众社交媒体帖子进行福祉评分；

**💡 创新点**

创新点在于提出了面向洪水话语的证据式福祉测量工具和解释性贝叶斯推理模型Wellbeing‑Former，兼顾透明度、效率与预测性能；

**🔧 技术方法**

采用轻量级预训练语言编码器、非因果证据解码器和基于理由的分数解码器，并通过标注数据训练，随后在全体224k帖子上推断；

**📊 数据集**

使用Meta Content Library检索的爱尔兰洪水相关帖子数据集（约224,000条），训练集1万条由Claude Opus标注；

**📈 对比分析**

与词典、神经网络、选择性推理和LLM评估基线比较，Wellbeing‑Former在Acc@1、F1@1、κ、τ等指标上分别达0.982、0.918、0.953、0.946，明显优于所有基线；

**⚠️ 局限性**

局限包括对公开社交媒体的依赖（可能存在样本偏差）、模型对极端或细微情感的判别仍有限，以及对多语言或跨文化推广的验证不足。

---

## 605. Reformulating plastic instabilities within a damage-like variational framework

**arXiv ID:** 2610.03328 | [PDF](https://arxiv.org/pdf/2610.03328v1)

**作者:** Baptiste Reyne `[一作]` `[通讯]` (SINTEF), Baptiste Reyne (SINTEF)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种将塑性不稳定性表述为带损伤变量的变分软化问题的最小化模型；

**💡 创新点**

创新点在于将软化与硬化分离为独立内部变量，并利用变分结构和 Lipschitz 正则化实现对局部化的控制；

**🔧 技术方法**

采用变分法、损伤力学原理、Lipschitz 连续性约束以及交替凸化求解器；

**📊 数据集**

通过在一维拉伸杆（15 节点）上进行数值试验，使用 σ_y=100 MPa、σ_a=40 MPa、k=10、E=70 GPa、δ=0.02 等参数；

**📈 对比分析**

结果显示正则化长度越大，局部化越平滑；与传统梯度正则化相比，Lipschitz 约束在软化未触发时无影响，性能相当；

**⚠️ 局限性**

局限性包括求解器收敛性差、解不唯一、对初始塑性缺陷敏感、需进一步优化求解算法及推广到高维情形。

---

## 606. VenusRL: A Fully Disaggregated Agentic RL System with Priority Scheduling and Scalable Interaction

**arXiv ID:** 2610.03286 | [PDF](https://arxiv.org/pdf/2610.03286v1)

**作者:** Mingjun Zhang `[一作]`, Yujun Zhang `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个完全去耦合的Agentic RL训练系统，拆分训练、生成和环境交互三大模块，并通过优先级感知的动作调度器和资源共享的环境管理器来提升训练吞吐量与成本效率。

**💡 创新点**

核心创新点包括：①基于优先级的动作级调度策略，识别并优先执行关键分组；②三层KV缓存模型与轨迹感知的 radix 缓存，降低高优先级轨迹的重算成本；③基于页面级沙箱部署和按组共享页面的策略，实现高密度、低内存占用的环境交互。

**🔧 技术方法**

使用的技术包括：Python/Rust/Golang实现的去耦系统框架；优先级感知GPU槽分配；轨迹长度预测启发式调度；三层KV缓存管理；页面级沙箱分配与copy‑on‑write；动态入库控制与全局沙箱池。

**📊 数据集**

实验使用了主流大语言模型训练任务与标准RL基准（未在摘要中具体列出，但推测为常用的agentic RL数据集）。

**📈 对比分析**

与Slime、RollFlash、ThunderAgent及E2B等现有系统比较，系统在端到端训练速度上分别提升1.07–3.26×、1.06–4.24×和最高2.67×，并在环境成本上相对E2B降低了约89%。

**⚠️ 局限性**

局限性包括：需要精确的长度预测和优先级划分才能发挥最大效益；页面级沙箱共享在某些多样化轨迹上可能无法完全实现；系统部署和维护的复杂性较高；实验评估未覆盖所有可能的任务和框架。

---

## 607. Multi-Task Evolution for Zero-Shot Cross-Problem Generalization using LLMs

**arXiv ID:** 2610.03316 | [PDF](https://arxiv.org/pdf/2610.03316v1)

**作者:** Zhouliang Xie `[一作]` (Southern University of Science and Technology), Zhenkun Wang `[通讯]` (Southern University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于LLM的多任务进化框架MECo，用来自动生成并组合启发式算法，支持在仅使用源任务反馈的情况下零样本跨问题泛化。

**💡 创新点**

创新点包括：①利用任务转移缺口衡量不同任务间的互补性，从而指导任务间的交互与知识迁移；②通过源任务评估完成启发式进化与多源覆盖的补充式集选择，完全不需要目标问题的搜索或适配；③将LLM生成的程序与进化算子（机制组合、冲突解决、探索性交叉、迁移导向变异）结合，提升跨任务性能。

**🔧 技术方法**

技术方法：使用大语言模型（如GPT‑4o‑mini、gpt‑5.4‑nano、gemini‑3.1‑flash‑lite）生成程序；多任务进化搜索框架；任务级转移缺口计算与任务配对策略；非支配排序与多源覆盖的补充式集选择；评估使用标准化源任务成本。

**📊 数据集**

数据集：车辆路径规划（VRP）与柔性车间调度（FJSP）的32个变体（16个VRP、16个FJSP），包括单约束的4个源任务与多约束组合的11个目标任务；使用公开的VRP和FJSP实例集进行训练与测试。

**📈 对比分析**

与八种自动启发式设计基线（包括任务特定、跨任务与经典构造规则）以及三种基线与MECo的混合版本进行比较。MECo在ID与OOD均取得最低平均成本、最高平均排名、最小相对差距；在跨问题基准中超越所有基线；将MECo框架嵌入现有基线后，三种基线的性能均得到显著提升。

**⚠️ 局限性**

局限性：①完全依赖源任务评估，若源任务与目标任务差距过大，泛化能力可能受限；②对LLM生成质量和可解释性的敏感性，模型选择会影响结果；③在极度复杂或高维约束组合时，搜索预算与计算成本显著增加；④未探讨动态环境或在线适配场景。

---

## 608. Equivariant Visual-Tactile Diffusion Policy for Contact-Rich Manipulation

**arXiv ID:** 2610.03333 | [PDF](https://arxiv.org/pdf/2610.03333v1)

**作者:** Lik Hang Kenny Wong `[一作]` (Chinese University of Hong Kong), Qi Dou `[通讯]` (Chinese University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种工作空间等变的视触觉扩散策略VISTA，用于在少量演示数据下进行接触丰富的机器人操控任务

**💡 创新点**

通过将视觉与触觉观测投影到球面，使用排列等变球面融合并利用末端执行器旋转对齐，实现工作空间等变性，并在扩散模型中条件化空间一致的动作预测

**🔧 技术方法**

球面投影与球面谐波、排列等变球面交叉注意力、工作空间旋转校正、等变扩散网络、有限旋转群（I_60、C_8）等技术

**📊 数据集**

在八个仿真任务（插拔、拉取、提升、放置等）以及三个人机实测任务（插USB、插铅笔、擦除）上进行评估

**📈 对比分析**

与非等变视触觉方法（ManiFeel、VITAL、UniVTAC）及等变扩散基线EquiDiff+Tac比较，VISTA在50个演示下平均成功率80.9%，比EquiDiff+Tac高15.9个百分点；在100个演示下提升至84.7%；实机上平均成功率7.33/10，远超1.33/10

**⚠️ 局限性**

等变性仅在所选有限群下精确；假设视觉/触觉观测在工作空间旋转前不变，且传感器固定不动；未实现高频触觉闭环，可能错过瞬时滑动或接触变化

---

## 609. ForestQuery: Boundary-Aware and Spatially Anchored Query Learning for Unified Forest Point Cloud Segmentation

**arXiv ID:** 2610.03403 | [PDF](https://arxiv.org/pdf/2610.03403v1)

**作者:** Zhihao Zhan `[一作]` (Nanjing University), Jie Yuan `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种统一的查询式框架ForestQuery，用于森林点云的语义和个体树分割；

**💡 创新点**

创新点包括：①边界不确定性显式建模并在实例查询构造与优化中引入自适应损失重权；②空间锚定语义查询增强（SA‑SQE），利用可学习的三维锚点编码森林垂直分层先验；

**🔧 技术方法**

采用稀疏3D U‑Net编码器、Transformer解码器、ISA‑Boundary‑Guided查询点采样、边界目标构造与自适应重权、可学习的3D位置编码等技术；

**📊 数据集**

使用公开的FOR‑instanceV2森林点云数据集（训练/验证/测试），以及未参与训练的LAUTx和自采集的UAV/MLS数据集进行跨域评估；

**📈 对比分析**

与ForAINetV2、TreeLearn、OneFormer3D、ForestFormer3D等方法对比，ForestQuery在FOR‑instanceV2上实现个体树F1最高84.0%、覆盖率90.0%，语义mIoU 87.6%；在LAUTx和自采集集上亦表现出最高召回率、覆盖率和最优的二分类语义mIoU；

**⚠️ 局限性**

局限性：仍对极度稀疏或高度重叠的树冠存在分割误差；对不同传感器的域迁移需要进一步自适应策略；数据量有限，未在更大规模多样化森林数据上验证。

---

## 610. The Plan Language of a Curriculum: A Formal Model and the Complexity of Degree Planning

**arXiv ID:** 2610.03392 | [PDF](https://arxiv.org/pdf/2610.03392v1)

**作者:** Sherzod Turaev `[一作]` (United Arab Emirates University), Mamoun Awad `[通讯]` (United Arab Emirates University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种形式化课程模型，将课程、前置条件、学分阈值和学期容量统一为语言生成器，分析其可行学习计划语言，并求解两大规划目标：学期数（time-to-degree）与总学分负荷（load）。

**💡 创新点**

证明两目标的复杂性来源完全分离：容量约束是 time-to-degree 的唯一困难源；而 disjunction（前置条件的或关系）和重叠选修（electives）则是 load 的独立困难源；并提供了延迟因子（delay factor）作为 time-to-degree 的多项式上界和“disjunctive slack”修正量。

**🔧 技术方法**

使用图论（有向无环图）、布尔公式（CNF）、动态规划、归约（Bin Packing、Set Cover、Subset Sum、反馈弧集）等理论工具，给出多项式算法与 NP/CoNP 难度证明。

**📊 数据集**

基于 22 所阿联酋大学的课程目录数据集（6,422 门含前置条件课程），公开了课程前置结构、宽度分布、滑差量等统计。

**📈 对比分析**

对比实验显示：在实际数据中 88% 的课程是纯粹的合取，disjunctive slack 在课程层面仅为 8.9% 的课程，容量约束才是决定学期数的关键；load 目标在实际课程中已表现出 NP 难度。模型与传统的课程复杂度指标（delay factor）一致，但通过“disjunctive slack”给出更精确的预测。

**⚠️ 局限性**

限制：模型假设所有课程均可在任何学期选修（忽略开设时间、核心课、共同课等限制），且未考虑课程重修、最小学分下限等现实约束；在容量不受限时的时间目标仍可能因课程数量大而导致算法效率下降；同时，NP/CoNP 难度结果为最坏情况分析，实际规划问题在特定参数下可能可用启发式或近似求解。

---

## 611. 16-bit Precision of Convolutional Neural Networks on Microcontroller Units for 8-bit Costs

**arXiv ID:** 2610.03402 | [PDF](https://arxiv.org/pdf/2610.03402v1)

**作者:** Rui Liu `[一作]` (Bielefeld University), Benjamin Paaßen `[通讯]` (Bielefeld University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 W16A16 16 位整数量化方案，在 ARM Cortex-M MCU 上实现高精度、低能耗的深度神经网络推理，重点针对时间卷积网络（TCN）进行实现与评估。

**💡 创新点**

创新点在于利用 ARMv7E‑M 双 MAC 指令消除 8 位量化的符号扩展和重排开销，并设计了 96 位跨寄存器移位与双阶段饱和的精确重量化算法，使 16 位量化在保持高精度的同时速度与能耗与 8 位量化相当。

**🔧 技术方法**

使用了 ARMv7E‑M 指令集分析、Seq2Col 1D 变换、全精度 96 位乘法与移位、固定点 Q31 近似、Cortex‑M DSP 指令、C/C++ 与汇编混合实现以及开源软件包。

**📊 数据集**

实验基于 C‑MAPSS 机舱剩余寿命估计数据集和 NinaPro DB2 肌电信号数据集（含回归与分类任务）。

**📈 对比分析**

通过回归 RMSE、分类准确率、单层与整体模型的 CPU 周期以及能耗测量进行比较；W16A16 在回归误差比 8 位低 10 倍、分类准确率与 FP32 接近，执行周期比 W8A16 快约 17%/24%（STM32H7/GD32F4），能耗略高但仍低于混合 8/16 方案。

**⚠️ 局限性**

局限性包括仅适用于 ARMv7E‑M 架构，4 位 MAC 指令优势不适用于 ARMv8 或 RISC‑V；实验仅在 TCN、两个数据集和两款 MCU 上验证，其他模型、数据集或新架构的推广性仍待进一步验证。

---

## 612. A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants

**arXiv ID:** 2610.03396 | [PDF](https://arxiv.org/pdf/2610.03396v1)

**作者:** Nikolaj T. Mücke `[一作]`, Benjamin Sanderse `[通讯]` (Centrum Wiskunde & Informatica)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种统一的后验采样框架，能将预训练的随机插值器、流匹配与扩散模型改造成无须重新训练的后验采样器；

**💡 创新点**

通过在插值路径上同时插值观测，推导出闭式高斯似然得分并统一处理漂移/速度修正，实现三类生成模型的后验采样统一；

**🔧 技术方法**

使用插值观测得到的高斯似然得分、可导的协方差近似（Jacobian‑free 与共享雅可比）以及SDE/ODE采样；

**📊 数据集**

在三种实验场景下评估：线性高斯系统、二维随机Navier–Stokes、三维城市气流（uDALES）；

**📈 对比分析**

与 FlowDAS+SURGE、SDA+SURGE、D‑Flow SGLD、Guided FM、EnKF、粒子滤波等基线对比；在密集观测下本方法RMSE、CRPS均优于其它后验采样器；在稀疏Navier–Stokes中SI‑SDE（共享协方差）表现最优；在城市气流中DM‑SDE在速度上最好，SI‑SDE在温度上最好；Jacob‑free 版本速度快但在稀疏情形下精度下降。

**⚠️ 局限性**

假设观测为线性高斯、需使用高斯近似似然得分、雅可比近似可能引入偏差；共享协方差在观测维度大时成本上升；一阶自回归式更新未利用未来观测，长时间滚动可能累计误差；对非线性观测、平滑或更大模型不匹配的扩展尚未实现。

---

## 613. Benchmarking Candidate Coverage in Typed Decision Models

**arXiv ID:** 2610.03387 | [PDF](https://arxiv.org/pdf/2610.03387v1)

**作者:** Jiawen Lu `[一作]` (Monash University), Tongtong Wu `[通讯]` (Monash University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种配对候选-覆盖评估协议，对 Laya 与 Jev 这两种 typed decision 模型在四个文本分类任务上的缺失答案检测与错误拒绝进行评估。

**💡 创新点**

创新点在于提出“配对候选-覆盖”协议，能同时衡量模型的缺失答案检测、错误拒绝、分类准确性，并提供可复现的冻结输入与精度审计。

**🔧 技术方法**

使用了 Laya (0.3.21) 与 Jev 1.13.0 两种 open‑source typed decision 模型，并结合自定义阈值校准、AUROC 评估、Bootstrap 区间分析等技术。

**📊 数据集**

数据集包括 AG News、DBpedia、DAIR Emotion、TREC，分别覆盖不同类别数的文本。

**📈 对比分析**

比较方法：对每个任务、候选数、命名方式进行配对评估，测量缺失答案检测率 D、错误拒绝率 F、全集准确率和 AUROC；结果显示 Laya 在 TREC 具有高检测率但高错误拒绝率，Jev 在 DBpedia 具有高准确率且低错误拒绝率，整体无统一排名。

**⚠️ 局限性**

限制：样本量有限、仅评估两模型、缺少自然 OOS 验证、阈值校准仅在训练集上、未处理模型训练重叠、对概率精度敏感等。

---

## 614. KungfuAthleteBot: learning high-dynamic humanoid motion from video with unified robust recovery

**arXiv ID:** 2610.03388 | [PDF](https://arxiv.org/pdf/2610.03388v1)

**作者:** Zhongxiang Lei `[一作]` (Beijing Institute of Technology), Xuesong Li `[通讯]` (Beijing Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了KungfuAthleteBot（KAB）框架，将从视频重建的高动态人体动作转化为可在真机上执行的机器人运动，并实现了单一策略同时完成运动跟踪、干扰抵御和跌倒恢复。

**💡 创新点**

创新点在于三大失败模式的系统解决：①用物理引导的抛物线校正消除根部浮动、地面穿透和高频抖动；②引入伪低动能（LKE）采样，重塑初始状态分布以避免不物理可行的空中姿态；③设计统一奖励与离散状态初始化，使同一策略能够在无恢复参考数据的情况下完成跟踪与跌倒恢复，且恢复时间仅0.7 s。

**🔧 技术方法**

采用的关键技术包括：基于物理的轨迹修正算法、LKE采样与三阶段课程学习、FastSAC离散分布式强化学习、基于重力的随机跌倒状态生成（DGRSI）、以及多任务奖励函数与终止条件设计。

**📊 数据集**

使用的主要数据集是自研的KungfuAthlete数据集，包含197段国家级武术运动员训练视频、1,726段子clip，涵盖地面动作与跳跃动作，提供高线速度和角速度数据；与AMASS、PHUMA、LAFAN1等公开数据集进行对比。

**📈 对比分析**

在仿真与真实Unitree G1机器人上进行评估，将KAB与现有系统（TWIST、GMT、SONIC、BeyondMimic、FIRM、StableMimic等）对比。KAB在7个极端动态动作上实现100%成功率，跟踪误差显著低于对比方法；在跌倒恢复任务中，KAB恢复时间约0.69 s，明显优于其他统一策略（最快1.56 s），验证了其性能优势。

**⚠️ 局限性**

局限性包括：仍需人工标注抛物线最小点以完成轨迹修正；数据集规模与动作多样性有限，难以覆盖所有高动态场景；以及尚未实现完全端到端的视频到机器人控制链，后续工作需进一步自动化与扩展。

---

## 615. Native Action-Prior Learning from Videos for World Action Models

**arXiv ID:** 2610.03391 | [PDF](https://arxiv.org/pdf/2610.03391v1)

**作者:** Zhaochong An `[一作]` (Meta AI), Sen He `[通讯]` (Meta AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Native Action‑Prior Learning (NAVA‑WAM)，通过观察‑only 视频直接预训练动作策略，随后仅用少量动作标记的机器人演示进行后期微调，实现仅动作‑only 推理。

**💡 创新点**

创新点在于：1) 在 Mixture‑of‑Transformers 中设计 transition‑structured joint attention，使 Action‑DiT 在预训练阶段即可通过未来视频的 flow‑matching 直接学习动作先验；2) 通过去除中间潜在动作或视觉表征的接口，显著降低表示‑to‑control 的间接性；3) 采用异步注意机制，实现在后期推理时仅使用动作流，避免生成未来视频，从而实现高效的动作‑only 控制。

**🔧 技术方法**

使用的技术包括：Mixture‑of‑Transformers（MoT）架构、Video‑DiT 与 Action‑DiT 双流、transition‑structured joint attention、flow‑matching 训练目标、异步跨流注意、视频‑动作流匹配与视频‑动作流的联合后训练。

**📊 数据集**

数据集：观察‑only 视频来源于 Open X‑Embodiment、AgiBotWorld、EgoDex；动作‑标记机器人演示来自 LIBERO / LIBERO‑Plus、RoboTwin 2.0；物理实验使用 Franka FR3 机器人，执行三种桌面操作任务。

**📈 对比分析**

在 LIBERO、LIBERO‑Plus、RoboTwin 2.0 等标准基准上，与直接动作策略、Fast‑WAM、Image‑WAM、DreamZero、π_0.5 等基线相比，NAVA‑WAM 在 ID 与 OOD 场景均取得最高成功率（LIBERO 99.0% / 83.5%，RoboTwin 88.5% / 73.6%），并在低动作‑标签预算下优于代表性学习与潜在动作学习 10–20% 的提升；在物理机器人上实现 93.3% 的成功率，高于 DreamZero（66.7%）与 π_0.5（53.3%）。

**⚠️ 局限性**

局限性：1) 仍需少量动作标记进行后训练；2) 对不同机器人硬件或动力学的跨平台适配可能需要额外微调；3) 模型规模较大，推理时仍比纯视觉模型略高的 FLOPs；4) 对超低延迟实时控制的性能尚未系统评估。

---

## 616. LAS-CLIP: A Lightweight Adapter Steering Approach for CLIP's Visual Encoder

**arXiv ID:** 2610.03370 | [PDF](https://arxiv.org/pdf/2610.03370v1)

**作者:** Anh-Khoa Dinh-Duc `[一作]` (Viet Nam National University), Minh-Triet Tran `[通讯]` (Viet Nam National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出LAS-CLIP，一种在CLIP视觉编码器上插入轻量MaskAdapter的轻量化适配方案；

**💡 创新点**

通过可学习的注意力偏置实现对指定区域的动态引导，同时保持CLIP全部参数冻结；

**🔧 技术方法**

MaskAdapter（包含掩码编码器、令牌编码器与门控交互模块）+对比损失+身份正则化；

**📊 数据集**

使用100K带掩码的图像-文本对（从GRIT+YOLOE生成），并在ImageNet‑S、RefCOCO/RefCOCO+/RefCOCOg等数据集评测；

**📈 对比分析**

与原CLIP、MaskAdaptedCLIP、Red Circle、Alpha‑CLIP等方法对比，LAS‑CLIP在ImageNet‑S的Top‑1/5提升至≈70.0/90.9，且在RefCOCO/RefCOCO+/RefCOCOg的召回率上超过Alpha‑CLIP，表现出更强的区域定位与文本对齐；

**⚠️ 局限性**

依赖预先生成的掩码；在每层引入的偏置矩阵导致显著GPU内存占用；对极端遮罩噪声的鲁棒性仍有限。

---

## 617. Cordial Learning: Distributed Training with Correlated Data

**arXiv ID:** 2610.03330 | [PDF](https://arxiv.org/pdf/2610.03330v1)

**作者:** Sarah Shitrit `[一作]` (Tel Aviv University), Ilai Bistritz `[通讯]` (Tel Aviv University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种分布式学习框架Cordial Learning，适用于数据相关的多代理系统。

**💡 创新点**

创新点在于仅共享低维嵌入输出，利用协调层将彼此信息融合，并证明在线性模型下能收敛到全局最优。

**🔧 技术方法**

采用随机梯度下降与两速更新、协同层嵌入、残差化输入、投影约束等技术；理论基于博弈与随机优化。

**📊 数据集**

实验使用合成线性模型和多位数MNIST（Pairwise-10和Common-Cause）数据集。

**📈 对比分析**

与孤立训练和集中式网络比较，Cordial Learning在有相关性的数据下显著降低损失、提升准确率，甚至与集中式接近，且通信量极低。

**⚠️ 局限性**

局限在于收敛率与非线性理论分析尚未给出，对信息压缩与实时自适应学习机制的进一步研究仍待开展。

---

## 618. Preserving Mathematical Reasoning in Compressed Diffusion Language Models via Trajectory-Aware Low-Rank Approximation

**arXiv ID:** 2610.03326 | [PDF](https://arxiv.org/pdf/2610.03326v1)

**作者:** Tian Liang `[一作]` (Duke University), Yiran Chen `[通讯]` (Duke University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了扩散语言模型（dLLM）的低秩压缩，提出了轨迹感知的低秩目标与无完整生成回路的 Monte Carlo 估计方法 Traj‑MC。

**💡 创新点**

首次将压缩目标定义为沿生成轨迹分布的状态，提出 Traj‑MC 实现高效估计，并证明其在数学推理任务上的显著优越性。

**🔧 技术方法**

使用轨迹感知低秩优化、白化 SVD、Monte Carlo 估计、Cholesky 分解与无标记掩码采样等技术。

**📊 数据集**

校准使用 C4 数据集，评估基准包括 GSM8K、MATH‑500、SVAMP 与 ARC‑C 等数学推理任务。

**📈 对比分析**

与传统清洁激活校准（clean calibration）和无激活权重 SVD 进行对比，Traj‑MC 在 20%–40% 参数压缩下保持更高的数学推理性能，并显著降低生成状态重建误差。

**⚠️ 局限性**

仅在低秩压缩场景验证，未探讨对更大模型或其他压缩方法（量化、剪枝）的通用性；Monte Carlo 估计受样本方差影响，需更大样本以提升稳定性。

---

## 619. Defense-in-Depth at the Perception-Reasoning Interface of LLM-Centric Agentic UAV Swarms

**arXiv ID:** 2610.03319 | [PDF](https://arxiv.org/pdf/2610.03319v1)

**作者:** Mohammadhossein Homaei `[一作]`, Bo Wei `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文实现并评估了针对LLM中心无人机(UAV)编队的感知-推理接口的五层防御深度（Defense‑in‑Depth）体系，以防止对结构化感知报告的攻击导致编队调度被篡改。

**💡 创新点**

创新点在于（1）构造了完整的五层安全架构，每层基于已知系统量计算可检测的阈值并闭式推导攻击预算；（2）将攻击分为四阶“能力阶梯”，使每层仅面对能突破前一层的攻击；（3）在实际编队仿真中对每层进行独立与组合评估，并量化检测与响应之间的性能权衡；（4）提出了基于日志链的不可篡改治理层和基于安全阈值的确定性回退调度。

**🔧 技术方法**

采用的技术包括：大语言模型推理（Qwen3‑8B‑AWQ）、结构化传感报告的MAC/标签验证、物理可行性与跨模态一致性检查、队列守恒异常检测、基于历史记录的安全性验证与修复、以及基于哈希链的治理与回退。

**📊 数据集**

实验使用了基于LAUS仿真平台的三架无人机、20个地面传感器、30个时间槽的自定义场景（100 m×100 m 区域、100 m 高度、固定通信模型与能量/队列参数）。

**📈 对比分析**

对比方法：在30对齐种子下，逐层对抗四类攻击（通道拦截、感知节点篡改、冗余感知攻击、策略自适应攻击）进行评估，使用“累计成本”“攻击成功率”“回退占比”等指标。结果显示：单层检测能将攻击成功率压至几乎0，但若采用原始数据补偿，损失比例提升；单层安全验证（L4）虽不检测攻击，却能减少约37.5 % 的额外成本；完整五层堆叠将攻击损失压至1.02×，并把攻击成功率降至0.001。

**⚠️ 局限性**

局限性包括：仅评估了“旧值保持”补偿策略，未探索更优恢复方法；仅测试了特定乘法攻击模型，对加性误差的阈值可能不足；攻击者仅限于报文级别，没有考虑对控制链路或内存的直接篡改；实验规模受限于仿真平台，未验证在更大规模或更高噪声环境下的泛化性。

---

## 620. Lightweight, Rubric-Guided Trajectory Evaluation for Production AI Agents

**arXiv ID:** 2610.03315 | [PDF](https://arxiv.org/pdf/2610.03315v1)

**作者:** Linh-An Phan `[一作]` (Huawei), Yanbin Zhang `[通讯]` (Huawei)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

LiteTrajEval 是一种轻量级、预算受限的轨迹评估框架，旨在在生产级 LLM 代理中快速定位和诊断失败。

**💡 创新点**

其创新点在于将离线语义规则生成与在线单次 LLM 判定相结合，采用预算约束的序列化与冗余压缩，避免多调用导致的高成本与高延迟。

**🔧 技术方法**

技术手段包括离线规则学习（利用 LLM 发现语义模式）、统一轨迹预处理与标记、跨步冗余删除、分层预算分配、MMR 证据挑选以及基于 rubric 的单次 LLM 判定。

**📊 数据集**

实验使用公开的 Magentic‑One（44 条轨迹）和 τ‑retail（29 条轨迹）失败轨迹数据集。

**📈 对比分析**

与多调用的 AgentRx 基线相比，LiteTrajEval 在失败定位与人类注释的对齐度上提升约 20–35%，成本降低约 6 倍，评估时间缩短超过 8 倍。

**⚠️ 局限性**

局限性包括对用户交互行为的检测不足、对单一裁判模型的依赖导致鲁棒性受限，以及规则库需要手动维护和更新。

---

## 621. Optimal Planning in a Dynamic World

**arXiv ID:** 2610.03312 | [PDF](https://arxiv.org/pdf/2610.03312v1)

**作者:** Devin Wild Thomas `[一作]` (University of New Hampshire), Andrew Coles `[通讯]` (King's College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出Augmented SIPP，将安全区与可用时间转化为可变到达时间函数（ATF），实现更灵活的时间约束规划

**💡 创新点**

创新点在于把ATF作为成本代数的元件，证明其构成成本代数A*，并利用RePEAT算法实现任意起始时间规划

**🔧 技术方法**

采用成本代数框架、ATF的函数合成、复合ATF（cATF）数据结构以及改进的剪枝策略

**📊 数据集**

在一系列人工合成的交通网络和机器人路径规划测试案例上进行评估

**📈 对比分析**

与传统SIPP及其他时间约束规划方法相比，Augmented SIPP在大多数测试案例中实现了更短的规划时间和更高的成功率

**⚠️ 局限性**

局限在于处理高度动态或大规模时间约束时，ATF合成和cATF维护成本较高，且对非周期性约束支持有限

---

## 622. Passing the Test You Trained On: Re-evaluating Prompt-Injection Detectors for LLM Agents

**arXiv ID:** 2610.03448 | [PDF](https://arxiv.org/pdf/2610.03448v1)

**作者:** Zhuowen Liu `[一作]` `[通讯]` (Japan Advanced Institute of Science and Technology), Zhuowen Liu (Japan Advanced Institute of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估并比较了十五种 Prompt‑Injection 检测器（包括 Meta Prompt Guard、Horizon‑Labs 等）在两个 LLM 代理基准（AgentDojo 与 τ‑bench）及 BIPIA 上的检测效果与误报率，探讨训练数据与输入形式对性能的影响。

**💡 创新点**

发现检测器在不同基准间的排名转移差，训练输入形式（而非单纯攻击字符串）决定检测器在代理环境中的表现，且误报率在代理工具输出上表现一致，可通过简单格式化显著降低。

**🔧 技术方法**

采用差分重放、工具调用回放、任务级别阻断率评估、FPR/TPR 与 AUROC 计算、训练数据审计、两种 LLM 判定器进行对照实验。

**📊 数据集**

使用 AgentDojo 与 τ‑bench 的真实工具调用记录、BIPIA 的邮件/表格/代码上下文、公开的 GitHub README、业务邮件与教育网页文本做为基准与对照数据。

**📈 对比分析**

在 1% FPR 下，Horizon‑Labs、Wolf Defender、Prismor 等检测器在 AgentDojo 与 τ‑bench 上分别捕获 70% 以上注入；但它们在 BIPIA 上表现差异显著；整体误报率在不同工具输出间高度相关，说明输入格式是决定因素。

**⚠️ 局限性**

受限于基准的模拟环境、未考虑自适应攻击、部分检测器训练数据不可访问、仅评估静态注入，且判定器仅为诊断工具，未作为正式防御方案。

---

## 623. A Vision-Language Model (VLM)-based Pipeline for End-to-End Procedural Modeling of Field-Grown Maize from Point Clouds

**arXiv ID:** 2610.03468 | [PDF](https://arxiv.org/pdf/2610.03468v1)

**作者:** Mozhgan Hadadi `[一作]` (Iowa State University), Baskar Ganapathysubramanian `[通讯]` (Iowa State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

利用零样本视觉‑语言模型（VLM）对从场景点云渲染的正交视图进行叶片中线和节点标注，随后通过几何投影、跨视图一致性、叶片扩展与修复、节点约束、参数提取和可微 NURBS 微调，自动将未经手工标注的玉米点云转换为可编辑的程序化 3D 模型。

**💡 创新点**

①零样本 VLM 标注与几何纠正的组合能在无手工调参或专属标注的情况下获得 99.4% 的叶片恢复；②跨视图一致性和节点约束避免单视图误标导致的错误；③利用程序化生成器直接用测量到的叶片参数填充 NURBS 控制网；④通过可微 NURBS 微调在保留节点位置的前提下实现对叶片细节的精细拟合。

**🔧 技术方法**

多模态 VLM（Gemini）+ 2D 渲染与投影；几何投影与 2D 脑图分割；跨视图一致性（互最优匹配）；叶片扩展与碎片修复（基于表面图、方向一致性、椭球包络）；节点约束（距离与间距阈值）；程序化生成器 FloraForge；可微 NURBS 微调（Adam 优化 + 约束正则化）。

**📊 数据集**

MaizeField3D 数据集，100 棵田间玉米点云（10 棵分别覆盖不同基因型，包含 1,072 片叶），每棵点云已预处理为 10k 点的工作分辨率。

**📈 对比分析**

与之前的半自动 PSO+NURBS‑Diff 流程对比；本工作在 100 棵植株上实现中位点 Chamfer 距离 5.4 mm，叶片回召率 99.4%；B‑spline 中线重构 RMSE 中位 1.7 mm；在随机留出 20% 点的 hold‑out 试验中，拟合残差仅比训练点多 8%（均值 5.6 mm），显示拟合泛化良好。

**⚠️ 局限性**

对 VLM 质量的依赖导致顶部稠密叶片的标注不稳定，部分叶片仅在两视图中被检出；节点约束仍会产生未解决的节点（2/118），叶片碎片修复受限于几何门限；未标记点占比约 8%，影响宽度和长度测量；整体计算成本高（多视图渲染、VLM 推理、可微优化），对大规模实验的实时性有挑战。

---

## 624. Still funded, no longer counted: how NIH's 2025 award reviews changed what the government counts as minority health research

**arXiv ID:** 2610.03443 | [PDF](https://arxiv.org/pdf/2610.03443v1)

**作者:** Fangfang Xie `[一作]` (Nanjing University), Haining Wang `[通讯]` (Indiana University)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了美国NIH 2025年要求移除DEI语言后，基金审批对奖项摘要文本的编辑如何影响NIH基于文本的分类（RCDC REMHR）计数，并追踪了37000余份奖项与实际研究实践记录的差异；

**💡 创新点**

首次揭示了基金方对文本进行“词汇定向”治理后，分类指标随文本编辑而变更，导致已获资助但仍在进行的研究不再被统计，形成“未计入科学”；

**🔧 技术方法**

运用了自然语言处理与文本特征提取、逻辑回归与AUC评估、案例对比审查（盲评）等方法，对文本编辑与类别保持率之间的关系进行建模；

**📊 数据集**

使用了NIH ExPORTER、RCDC、OpenAlex、AACT（ClinicalTrials.gov）、Grant Witness、美国参议院委员会名单等公开数据集；

**📈 对比分析**

通过比较编辑前后文本特征、分类保持率、实验审查结果，模型在预测名称存活方面的AUC约为0.69，分类保持率从98.5%降至15%（当摘要失去种族/民族名称时），显示显著性能下降；

**⚠️ 局限性**

局限包括：严格案例样本小（20项目），实践记录对研究真实进行的捕捉不完整，文本特征模型仅捕捉部分编辑信息，研究窗口短，无法验证长期研究内容是否真正保持不变。

---

## 625. Jumping the Line: Exploiting Length Predictions in LLM Scheduling

**arXiv ID:** 2610.03430 | [PDF](https://arxiv.org/pdf/2610.03430v1)

**作者:** Yuyang Dai `[一作]` (Florida State University), Mahmood Sharif `[通讯]` (Tel Aviv University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对基于长度预测的LLM请求调度系统的攻击方法（JIL），通过在请求末尾追加对抗性后缀降低预测输出长度，从而提升自身调度优先级并加速完成。

**💡 创新点**

创新点在于利用对长度预测器的梯度信息针对性优化后缀，使调度器误估请求大小，形成“调度作弊”；同时首次评估此类攻击对不同模型、任务与调度策略的影响，并探讨可行的防御（预测阈值与分段）。

**🔧 技术方法**

主要技术包括：梯度引导的离散优化（GCG）用于生成对抗后缀；基于TRAIL的长度预测器与SRPT风格调度；对抗实验与真实部署中的成对工作负载评估；防御策略实现。

**📊 数据集**

使用四个LLM模型（Llama‑3‑8B‑Instruct、Qwen2.5‑7B‑Instruct、Mistral‑7B‑Instruct‑v0.3、Llama‑3.1‑70B‑Instruct）与四类数据集（Alpaca、UltraChat、ARC‑Challenge、HellaSwag）进行实验。

**📈 对比分析**

方法通过成对对比（clean vs. attacked）测量预测长度变化、完成时间/首 token 速度、完成顺序以及任务指标。实验表明攻击可将预测长度降低高达 83.4%，受攻击请求平均完成时间提升约 1.5×，但对正常请求造成上层 10–30% 的延迟；在不同模型/任务中攻击效果和任务质量损失存在差异。

**⚠️ 局限性**

局限包括：对抗后缀对生成文本质量影响不一，跨模型迁移效果受限；仅针对基于长度预测的调度器，未考虑其他资源调度策略；防御措施（阈值、分段）需权衡公平与效率，且可能对正常请求产生副作用。

---

## 626. CLIMB: Confidence-Guided Complementary Evidence for Multimodal Retrieval-Augmented Generation

**arXiv ID:** 2610.03421 | [PDF](https://arxiv.org/pdf/2610.03421v1)

**作者:** Hang Gao `[一作]` (Rutgers University), Dimitris N. Metaxas `[通讯]` (Rutgers University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的多模态检索增强生成框架CLIMB，先构建冗余抑制的证据池，再在固定池内通过信心控制的迭代推理更新答案。

**💡 创新点**

创新点在于：①使用MMR式目标构造非冗余、多样化证据池；②设计R/E/C三维评分的批判器和基于证据覆盖度的信心估计；③通过置信度上升判定实现自适应迭代停止。

**🔧 技术方法**

技术包括CLIP视觉-文本检索、MMR重排、LLM（LLaMA-3.1-8B）生成、R/E/C评分提示、覆盖度/一致性/特异性信心评估及面向多模态的交叉模态一致性检验。

**📊 数据集**

使用 Encyclopedic‑VQA 与 InfoSeek 两个需要外部Wikipedia知识的视觉问答基准。

**📈 对比分析**

与文本‑LLM、零样本多模态LLM、以及多模态RAG基线（DPR+T、RORA‑VLM、Wiki‑LLaVA、EchoSight、ReflectiVA）比较，CLIMB在两个数据集均实现最高准确率，分别提升约6–9%和约5%（单/多跳）。

**⚠️ 局限性**

局限在于仅适用于基于文本检索的知识强化视觉问答，无法直接扩展至视频、多图或特定领域语料；信心分数仅为相对可靠性指标，非绝对概率，需进一步校准。

---

## 627. LayerIt: Towards a Framework for Time-Aligned, Composable Music Visualizations

**arXiv ID:** 2610.03428 | [PDF](https://arxiv.org/pdf/2610.03428v1)

**作者:** Fernando Azeredo `[一作]` (Universidade do Porto), António Sá Pinto `[通讯]` (Universidade do Porto)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了 LayerIt Python 库，能够将音频、对齐后的分数、以及其他时间相关的信号或特征在同一表现时间轴上自动叠加并输出 SVG 可视化图；

**💡 创新点**

提出共享的表现时间轴层叠框架，自动组合多层信息并保留 MEI 结构可识别与可编辑，解决手工拼接可视化图的痛点；

**🔧 技术方法**

利用 librosa 读取音频、ScoreWarp 进行对齐、Matplotlib 绘图并导出 SVG，支持函数、瞬时、时间段、频谱等四类表示，并保持 MEI 结构；

**📊 数据集**

使用 Debussy《Clair de Lune》Maria João Pires 现场演奏的前六小节音频及其 MEI 分数，配合 Piano Precision 的对齐结果和 Beat This! 的节拍/下拍激活数据；

**📈 对比分析**

通过在同一图中展示波形、节拍 logits 与分数对齐，手动检查与传统工具（如 Sonic Visualiser）对比，示例中显示能够直接定位误差；未给出数值性能指标；

**⚠️ 局限性**

依赖 MEI 分数和外部对齐，可能携带对齐或排版错误；密集时间频谱被栅格化无法矢量化样式；仅支持 onset‑level 对齐，需进一步兼容其他同步工具。

---

## 628. Persona Guardrail: A Production-Grade Defense Framework for Agentic Systems

**arXiv ID:** 2610.03434 | [PDF](https://arxiv.org/pdf/2610.03434v1)

**作者:** Bijeeta Pal `[一作]` (Uber), Sean Tout `[通讯]` (Uber)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Persona Guardrail，一种在生产环境中同步验证用户输入和代理输出以保证 LLM 驱动代理在其功能边界内运行的安全框架，并创建 PAGE benchmark 来评估此类 guardrail 的效果。

**💡 创新点**

核心创新在于将功能边界（allowlist + blocklist）与 LLM 进行语义匹配，通过迭代优化 allowlist 提高 out-of-domain 检测，并将 guardrail 集成到实际生产系统，满足 sub‑100 ms 的 P90 延迟目标。

**🔧 技术方法**

使用 Qwen3‑30B‑A3B 作为二元分类模型（输入/输出），结合混合专家、FP8 量化和前缀缓存的 vLLM 推理架构，采用闭环反馈（shadow、canary、review）对策略进行持续演进。

**📊 数据集**

数据集包括：PAGE benchmark（Workspace、Slack、Banking、Travel 四个 AgentDojo 领域的正负样本）、内部生产流量（shadow 采样）、对抗样本（prompt injection、jailbreak 等）和基准 golden 例子。

**📈 对比分析**

与通用 LLM 判别器相比，iterative allowlist 方案将整体准确率从 85.7% 提升至 95.9%，out‑of‑domain 检测率从 57.3% 提升至 93.5%，误准率从 25.0% 降至 4.7%；在生产环境中实现 P90 延迟约 60 ms，满足 100 ms 的预算。

**⚠️ 局限性**

局限性包括：只能检测最终响应和单轮输入，无法覆盖多轮对话或内部工具调用的安全；对新出现的复杂攻击需要手工迭代 refine allowlist；在极端高流量时仍可能触发 P99 超过 100 ms。

---

## 629. Interactive Machine Learning Interfaces for Disease Risk Prediction: Effects on Risk Perception and Behaviour

**arXiv ID:** 2610.03511 | [PDF](https://arxiv.org/pdf/2610.03511v1)

**作者:** Tiffany Ngai `[一作]` (University of Waterloo), Anamaria Crisan `[通讯]` (University of Waterloo)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文通过设计并评估一个交互式Type 2糖尿病风险预测界面，探讨用户对机器学习模型输出的理解程度及其对行为改变的影响。

**💡 创新点**

创新点在于：①将用户感知理解与真实理解进行对比分析；②提出可直接落地的界面设计准则（术语简化、视觉友好、可解释性强化、行为导向）。

**🔧 技术方法**

使用的技术包括：R Shiny构建前端界面、Gradient Boosting Machine作为预测模型、DALEX实现模型解释（特征重要性瀑布图）以及交互式输入控制。

**📊 数据集**

数据集：采用公开的Type 2糖尿病风险预测模型所需的临床、人口、遗传特征；界面中使用合成的风险分布和样本数据进行演示，未使用真实病人数据。

**📈 对比分析**

评估方法：对10名参与者进行量化（感知理解与实际理解评分对比）和定性（主题分析）两方面的研究；模型性能指标如准确率、精确率、召回率及置信区间也被报告以增强透明度，但未与其他模型进行直接性能比较。

**⚠️ 局限性**

局限性：样本量仅15人（10人用于定量分析），多数为STEM背景；使用合成数据可能降低参与者投入；研究聚焦单一疾病，结果可能难以推广至其他病种。

---

## 630. Detect and Suppress: A Mechanistic Defense against Adversarial Patches in VLA Models

**arXiv ID:** 2610.03498 | [PDF](https://arxiv.org/pdf/2610.03498v1)

**作者:** Yukiya Horiba `[一作]` (Keio University), Taiki Miyanishi `[通讯]` (University of Tokyo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对视觉-语言-动作（VLA）模型在对抗性补丁攻击下的脆弱性，提出了一种基于稀疏自编码器（SAE）的防御方案，识别并在检测到攻击时抑制特定内部特征，从而提高任务成功率。

**💡 创新点**

创新点在于：①首次将SAE分析与对抗性攻击关联，定位攻击相关的内部特征；②利用线性探测器实现对攻击的实时检测；③仅在检测到攻击时才抑制该特征，避免持续干预导致的性能下降。

**🔧 技术方法**

技术方法包括：稀疏自编码器（SAE）特征提取、线性探测器（logistic regression）进行攻击检测、按条件抑制特定特征的插值操作；实验使用LIBERO‑10机器人任务集。

**📊 数据集**

数据集：LIBERO‑10（10项机器人任务，50个初始状态，每项10次，合计500个回合）。

**📈 对比分析**

对比方法：基线（无干预）、连续干预（无检测）、基于oracle检测的干预。结果显示：在间歇性UADA攻击下，条件干预在π_0.5模型上将任务成功率从26.0%提升至32.4%，在SmolVLA上提升幅度较小；相比之下，连续干预显著降低正常任务性能；oracle检测方法表现最优。

**⚠️ 局限性**

局限性：①对SmolVLA的提升有限，且在正常情境下仍有一定性能下降；②仅在仿真环境和单一UADA补丁攻击上验证，未测试对未知攻击或真实机器人；③攻击特征选择与检测阈值需额外训练，可能不适用于不同模型或任务。

---

## 631. Dual-Context Analog Retrieval for Time Series Forecasting

**arXiv ID:** 2610.03491 | [PDF](https://arxiv.org/pdf/2610.03491v1)

**作者:** Jung Min Choi `[一作]` (University of Hildesheim), Lars Schmidt-Thieme `[通讯]` (University of Hildesheim)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DuoTS 模型，在多变量长时序预测中先生成基线预测，再通过两种视角（最近上下文和检索到的历史相似片段）逐段细化预测。

**💡 创新点**

创新点在于：①双视角检索校正模块，可与任意基线模型结合；②对每个未来 patch 采用独立的检索与最近上下文权重调节；③保持基线预测完整性，仅在需要时进行微调，避免完全替换。

**🔧 技术方法**

使用 Patch 级别的并行编码器、稀疏跨通道混合、二阶差分动态建模、基于余弦注意力的检索、以及可学习的 gate 与 softmax 路由进行校正。

**📊 数据集**

在八个公开长时序基准上评估：ETTh1/2、ETTm1/2、Weather、Electricity、Traffic、Solar，预测时延为 96、192、336、720。

**📈 对比分析**

与六个主流基线（SRSNet、TimeKAN、Amplifier、iTransformer、TimeMixer、PatchTST）对比，DuoTS 在 32 个数据集–时延组合中 22 次取得最低 MSE、23 次取得最低 MAE，且在多数据集–时延上显著提升。

**⚠️ 局限性**

局限在于只能检索输入窗口内的相似片段，匹配不佳或重复时校正效果有限；对小通道数时序开销较大；检索的相似度不一定对应因果关系。

---

## 632. Becoming Suspicious Across Borders: Algorithmic Extraterritoriality and AI-Driven Financial Surveillance

**arXiv ID:** 2610.03425 | [PDF](https://arxiv.org/pdf/2610.03425v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 633. Code distances of matrix rank-metric codes under Add-and-Remove transformations

**arXiv ID:** 2610.03460 | [PDF](https://arxiv.org/pdf/2610.03460v1)

**作者:** Gianira N. Alfarano `[一作]` (University of Rennes), Adrien Vinçotte `[通讯]` (University of Rennes)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在 Add‑and‑Remove 构造下，矩阵 Gabidulin 码及一般矩阵码的子码距离如何变化，并给出了相应的不等式与等式，尤其证明了当 ℓ_s<m 时子码距离仅在每个长度为 m 的区块末尾 ℓ_s 个指标处可能下降 1，随后将结果应用于 MIRANDA 签名方案的公共码，给出其对偶码子码距离的下界。

**💡 创新点**

首次量化了 Add‑and‑Remove 构造对子码距离的影响，揭示了子码距离的局部稳定性与极限，并将其与对偶码的 MRD 性质关联，提供了理论上可用于区分隐藏 Gabidulin 结构的指标，扩展了对 rank‑metric 码的结构分析。

**🔧 技术方法**

利用子码距离（α_i）定义、Singleton‑like 上界、Delsarte 双对偶、Gabidulin 码的层级结构、Rank‑metric 等距映射以及对偶码构造等理论工具，推导了一系列关于子码距离的上下界与等式。

**📊 数据集**

本文未使用实验数据集，而是基于符号计算与理论推导，使用了 MIRANDA 方案的典型参数集（例如 (m,n,κ,ℓ_a,ℓ_s)=(113,113,107,220,3)）来演示结果。

**📈 对比分析**

通过理论上推导的子码距离下界与上界与随机码的预期子码距离进行比较，说明 Add‑and‑Remove 码在某些指标上可优于随机码，但未给出实际实验性能评估；对偶码的子码距离下界表明其对齐结构相对较强。

**⚠️ 局限性**

主要限制在于：1）尚不知下界差异在所有参数下的出现频率；2）随机矩阵码子码距离的分布尚未知，难以构造多项式时间的区分器；3）对偶码的完整子码距离分析仍未完成，无法直接用于高效攻击；4）理论结果未通过实验验证。

---

## 634. Fed-ADApt: Federated Anytime Depth Adaptation for Resource-Aware Medical Image Segmentation

**arXiv ID:** 2610.03474 | [PDF](https://arxiv.org/pdf/2610.03474v1)

**作者:** Abhijeet Parida `[一作]` (Sheikh Zayed Institute for Pediatric Surgical Innovation), Holger R. Roth `[通讯]` (NVIDIA Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种 Fed-ADApt 框架，支持 UNet 结构在联邦学习中根据不同站点的训练与推理计算预算进行深度自适应训练与推理。

**💡 创新点**

创新点在于将多深度监督与层级参数聚合相结合，使低资源站点能够参与训练并在部署时自适应选择合适的深度，从而兼顾训练与推理成本。

**🔧 技术方法**

利用多深度 UNet、DepthFL 的层级聚合、Federated Anytime UNet 的任意深度推理以及 FedAvg 与 Dice 损失等技术实现联邦学习。

**📊 数据集**

在多站点二维视网膜光盘分割（Drishti、RIGA、REFUGE 等）和三维脑肿瘤分割（BraTS）数据集上进行实验评估。

**📈 对比分析**

与 FedAvg、DepthFL、Federated Anytime UNet 等基线对比，Fed-ADApt 在二维任务平均 Dice 0.81（相较 FedAvg 0.85）并显著降低训练/推理/通信成本；在三维任务保持 WT Dice 0.82 与 FedAvg 相同，同时大幅节约计算与内存。

**⚠️ 局限性**

局限包括预设计算预算、未在真实 IoMT 设备上验证、深度选择基于验证而非实时估计、仅验证 UNet 架构，缺乏对 Transformer 等更高级分割模型的扩展。

---

## 635. ChromaGS: Text-Driven Semantic Editing of 4D Gaussian Avatars

**arXiv ID:** 2610.03441 | [PDF](https://arxiv.org/pdf/2610.03441v1)

**作者:** Antonio Canela `[一作]` (Universitat Politècnica de Catalunya), Jordi Sànchez-Riera `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

针对可动画的3D高斯头部化身，实现了实时、语言驱动的颜色编辑，用户可在渲染时通过自然语言直接修改语义区域的颜色。

**💡 创新点**

创新点在于：①为每个高斯原语学习软语义分配并将颜色拆分为区域基色与残差，实现局部一致的颜色更改；②两阶段语言解析+颜色解析管线，支持绝对与相对色彩指定；③实现无重训练、即时编辑。

**🔧 技术方法**

采用3D高斯Splatting与FLAME绑定的可动画表示，结合SAM3语义分割、Flan‑T5‑Small文本解析、all‑MiniLM‑L6‑v2句子编码、HSV转化、线性混合皮肤变形，配合色彩残差的球谐系数学习。

**📊 数据集**

在FaceScape与INSTA两个面部视频数据集上进行训练与评估，使用6个预定义语义类别（头发、眉毛、眼睛、嘴唇、皮肤、背景）。

**📈 对比分析**

与GaussianAvatar‑Editor和InstructPix2Pix对比；在身份保持、上下文保留和CLIP分数上取得更高的身份与上下文分数；编辑耗时仅50 ms、渲染>100 fps，显著快于需要800 s每次编辑的基线。

**⚠️ 局限性**

局限性包括：依赖语义分割的精度；只能实现颜色平移，无法修改纹理、材质或几何形状；高频细节如头发的光照不够逼真；受限于预设调色板，外部色调需映射到最近条目；重建受FLAME跟踪精度限制。

---

## 636. CorrectGuard: Eyes-Off Correctness Estimation for Black-Box Security Guardrails

**arXiv ID:** 2610.03470 | [PDF](https://arxiv.org/pdf/2610.03470v1)

**作者:** Adam Faulkner `[一作]` (Microsoft), Matthew Dressman `[通讯]` (Microsoft)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CorrectGuard 框架，通过在无“眼睛”访问的环境下训练外部正确性模型，对黑盒安全防护进行无内部信息的错误检测、排序和拒绝决策。

**💡 创新点**

创新点在于将离线眼睛闭合（human/machine eyes-off）数据用于训练外部正确性评估器，实现在隐私受限场景下对黑盒防护效果的可观测性与可控制性。

**🔧 技术方法**

使用了 in-context 学习（OpenAI GPT-5.4）、低秩适配（LoRA）微调的 LLM、基于嵌入的 MLP 以及 BinaryShield 隐私保护向量等技术。

**📊 数据集**

采用了 13 个公开安全/攻击数据集（harmful, jailbreak, prompt injection, extraction 等）与 3 个开源黑盒 guardrail（Granite Guardian, Qwen3Guard, GPT-OSS-Safeguard）进行实验。

**📈 对比分析**

采用 leave‑one‑dataset‑out 评估，ICL 模型提升约 15–25 个百分点，嵌入+MLP 提升约 10–15%，所有模型保持良好排序能力，AURC 明显下降，表明能有效控制风险并提高决策质量。

**⚠️ 局限性**

局限性包括仅在开放权重代理上验证、ICL 仅用单一 LLM、微调受计算限制、BinaryShield 实验范围有限，以及在强 guardrail 上性能不一且正确性概率校准差。

---

## 637. PEACE: Joint Embeddings of DSP Effects Code and Audio

**arXiv ID:** 2610.03405 | [PDF](https://arxiv.org/pdf/2610.03405v1)

**作者:** David Braun `[一作]` (Princeton University), Adam Finkelstein `[通讯]` (Princeton University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了PEACE模型，实现了音频效果代码和输出音频的联合嵌入，支持跨模态检索与效果链拓扑恢复。

**💡 创新点**

首次将音频效果代码与音频映射到共享嵌入空间，并引入同时使用基于T5的文本编码器和基于BoxGraph图神经网络的代码编码器；通过SLAP无对比学习实现参数无关的拓扑检索。

**🔧 技术方法**

采用SLAP对齐音频与代码模态，使用AFx-Rep的CNN14音频编码器，微调T5 Transformer作为代码编码器，以及基于Faust Block Diagram Algebra的BoxGraph图神经网络；同时在训练时对参数进行遮蔽以获得拓扑级别表示。

**📊 数据集**

构建200K个音频-代码对，覆盖9类干音源和21种Faust效果器；验证集与测试集使用公开音乐数据集、专业插件的房间衰减响应和多源音频。

**📈 对比分析**

与预训练AFx-Rep、Fx-Encoder++、LAION-CLAP等基线在Recall@1/10、均值/中位数排名、模态可分性等指标比较；PEACE在跨模态检索上与基线相当或略优，尤其在长链检索与Reverb匹配中显著超越，BoxGraph在拓扑检索上最高达61% Recall@10。

**⚠️ 局限性**

数据随机采样导致与真实工程差距，缺乏人类标注的参数配置；在非线性与动态效果的检索性能不佳；仅评估了Reverb而非更广泛插件效果，且对模型的泛化仍有待提升。

---

## 638. MobiAgent: Dual-Loop Recursive Policy Self-Improvement for Long-Horizon Mobile Manipulation

**arXiv ID:** 2610.03476 | [PDF](https://arxiv.org/pdf/2610.03476v1)

**作者:** Chenzhi Liu `[一作]` (University of Hong Kong), Xiaojuan Qi `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MobiAgent 双环框架，用于长周期移动操控；内环实现动态规划、原子技能执行与视觉反思，外环实现无人工注释的持续自我改进。

**💡 创新点**

创新点在于将高层推理与低层控制分离为可组合的原子技能，利用 VLM 进行实时动态规划与视觉验证；外环实现自动数据策划与递归技能更新，突破了传统单一子任务映射、僵化重规划和无学习闭环的局限。

**🔧 技术方法**

技术组合包括：Vision‑Language 模型（VLM）+ flow‑matching VLA（π_0.5）共用 VLM trunk + 任务规划器、技能执行器、反思评判器 + 数据策划器、技能生成器、技能训练器；以及自回归的分段验证与无监督聚类。

**📊 数据集**

使用数据集：RoboCasa（模拟）、BEHAVIOR‑1K（模拟）、Astribot S1（真实机器人）以及 OmniGibson 作为环境仿真。

**📈 对比分析**

与 π_0.5‑TA、CaP‑X 等基准对比，BEHAVIOR‑1K 平均成功率提升至 65%（比基准高 22.5pp）；RoboCasa 从 7.5% 提升至 27.5%（+20pp）；Astribot S1 从 32.5% 提升至 57.5%（+25pp）。

**⚠️ 局限性**

局限包括：技能扩展可能导致已学行为干扰；内环仍存在姿态校正、物体放置误差和视觉遮挡导致的评判漂移；需要在更多任务、环境与机器人平台上进一步验证其可扩展性与鲁棒性。

---

## 639. OuroReward: Sequential Reward Scheduling for Reinforcement Learning in Text-to-3D Generation

**arXiv ID:** 2610.03423 | [PDF](https://arxiv.org/pdf/2610.03423v1)

**作者:** Bingyang Cui `[一作]` (Shanghai Jiao Tong University), Yunfeng Guan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文针对文本到三维（Text-to-3D）生成中的多维度强化学习（RL）优化问题，提出了两项关键技术：AdaSelect（自适应提示词选择）和OuroReward（干扰感知的顺序奖励调度），从而在保证多维度质量平衡的同时提升生成质量和训练稳定性。

**💡 创新点**

创新点在于：
1) AdaSelect 能根据模型当前能力动态挑选信息量足够、难度适中的提示词，显著提升学习信号质量；
2) OuroReward 通过估计维度间梯度相似度与奖励变化的相关性，构造环形依赖路径并按最小累积干扰顺序进行奖励调度，解决传统同时优化导致的维度干扰和“奖励挖掘”问题；
3) 将两者结合形成一个完整的多维度 RL 框架，兼容多种奖励模型与 RL 算法。

**🔧 技术方法**

技术手段包括：
- 强化学习框架（DiffusionNFT、GRPO、AWM 等）；
- LoRA 参数高效微调；
- 梯度余弦相似度、奖励变化 PLCC 用于构造维度依赖图；
- 轮流/顺序奖励调度与头间隙（headroom）动态切换；
- 多种奖励模型（Projection-based：HyperScore、Rank2Score；Model-based：Uni3D、ULIP；LLM-based：Qwen3-VL-8B-Instruct）与自训练的维度专属奖励。
- GPT-5.6 生成大规模提示词数据集，覆盖多种属性组合。

**📊 数据集**

数据集：
- 通过 GPT-5.6 生成 10,000 条结构化提示词（8,000 训练 / 1,000 验证 / 1,000 测试），覆盖主体、属性、材质、动作、关系、场景、风格等维度；
- 评测集采用多种评价器：Projection-based（CLIPScore、ImageReward、HyperScore、Rank2Score）、Model-based（Uni3D、ULIP）、LLM-based（Qwen3-VL-8B-Instruct）。

**📈 对比分析**

对比方法：
- 基线：平均奖励聚合；
- 对比 RL 算法：GRPO、DiffusionNFT、AWM；
- 对比奖励聚合策略：FocalReward、OuroReward。 
实验结果表明：
- 在 Proxy 奖励（HyperScore、Rank2Score）上提升 3–5%；
- 在未见评测器（CLIPScore、ImageReward、Uni3D、Qwen3-VL-8B-Instruct）上均有显著提升，尤其是人类偏好 Win Rate 提升至 60% 以上；
- Ablation 证明 AdaSelect 与 OuroReward 各自均能提升性能，二者组合效果最佳；
- 在不同基础模型（TRELLIS、Hunyuan3D-2.1）和 RL 算法上保持一致性。

**⚠️ 局限性**

局限性：
1) 依赖预训练奖励模型的准确性，若奖励模型本身存在偏差，改进效果可能受限；
2) 计算成本仍高，尤其是多提示词的轻量级 roll‑out 与维度依赖估计；
3) 当前仅验证了 4 维度（对齐、几何、纹理、整体）和两种主流 RL 算法，对更大维度或其他 RL 框架的推广需要进一步研究；
4) 对提示词构造的 GPT-5.6 生成质量和多样性有一定依赖，实际应用中需保证提示词质量。

---

## 640. Causal Representation Learning with Instantaneous and Lagged Relations via Nonstationarity

**arXiv ID:** 2610.03452 | [PDF](https://arxiv.org/pdf/2610.03452v1)

**作者:** Tatsuya Yamada `[一作]` (University of Osaka), Yoshinobu Kawahara `[通讯]` (University of Osaka)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为 iCReN 的对比学习框架，用于在时间序列中联合识别潜在状态及其瞬时和滞后因果结构，并利用观察到的辅助变量（如时间或实验条件）捕捉转移噪声的非平稳性。

**💡 创新点**

创新点在于：① 通过辅助变量引入的转移噪声非平稳性与瞬时、滞后因果关系共同满足可识别性条件；② 在理论上给出了充分条件，证明潜在状态和因果结构可识别（相对于先前仅识别滞后关系的工作）；③ 将 IIA（独立创新分析）结合对比学习（TCL/GCL），实现无监督的潜在表示学习，且不需要显式的变分后验或解码器。

**🔧 技术方法**

技术手段包括：基于 IIA 的对比学习（对离散辅助变量使用 TCL，对连续辅助变量使用 GCL）；对逆向转移函数 r̂ 进行稀疏与无环正则化；利用 Jacobian 支持估计因果图；在理论分析中使用马尔可夫网络、变换不变性和线性代数等工具。

**📊 数据集**

实验数据集：① 通过三层神经网络生成的合成时间序列（维度 5，滞后 L=1），使用离散环境标签或连续时间作为辅助变量；② 真实世界数据：1）跑步机步态数据（多受试者的运动捕捉序列），2）ERA5 全球再分析气候数据（5 个城市、8 个地区的温度、压力、风速等特征）。

**📈 对比分析**

与 IDOL、G‑CaRL、CtrlNS、LEAP、iVAE、β‑VAE 等基线进行比较；在合成数据上，iCReN 在潜在状态回归（MCC）和瞬时/滞后 F1 分数上均取得最优或接近最优结果；在真实数据上，iCReN 在步态预测（不同速度的迁移任务）和天气预报（三小时相对湿度）中获得最低误差/标准化 MSE，表现优于其它对比学习与变分自编码器方法。

**⚠️ 局限性**

局限性包括：① 需要观察到与转移噪声变化相关的辅助变量；② 理论可识别性条件不涵盖当转移噪声可拆解为当前与过去状态函数的线性或某些特殊非线性模型；③ 在真实数据上的评估仅基于下游预测性能，缺乏对潜在因果结构的直接验证。

---

## 641. Deep Bayesian REFoCUS

**arXiv ID:** 2610.03419 | [PDF](https://arxiv.org/pdf/2610.03419v1)

**作者:** Simon Penninga `[一作]` (Eindhoven University of Technology), Ruud van Sloun `[通讯]` (Eindhoven University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种基于深度生成先验的贝叶斯多静态回收框架 Deep Bayesian REFoCUS，能够在秩缺陷和噪声环境下从任意发射序列恢复完整多静态超声通道数据，并给出不确定性估计。

**💡 创新点**

创新点在于将后向传播（adjoint）编码与扩散后验采样（Diffusion Posterior Sampling）相结合，利用生成模型填补秩缺陷的零空间，实现无须微调即可适用于任意发射模式，并在解算过程中自然表达不确定性。

**🔧 技术方法**

采用的技术包括流匹配扩散生成模型、U‑Net 3D 结构、后向传播编码、Diffusion Posterior Sampling 指导、Tikhonov 正则化对比、以及在模拟与真实数据上的多尺度训练和评估。

**📊 数据集**

训练数据来源于 EchoNet‑LVH 的真实心脏 PLAX B‑mode 图像转化为散射密度图，随后用 STA 模拟生成多静态数据；测试数据包括12名志愿者在 Verasonics Vantage 256 上采集的多发射序列的真实心脏 PLAX 记录。

**📈 对比分析**

通过与传统线性 REFoCUS、带坡度滤波的后向传播和 Tikhonov 逆解等基线进行比较，使用通道域的复相关、NMSE，图像域的 PSNR 与 SSIM 等指标，实验表明在秩缺陷和低采样率下 Deep Bayesian REFoCUS 显著优于线性解码器，并在大多数噪声水平下保持较高性能；在真实数据上虽表现略逊，但仍优于线性方法。

**⚠️ 局限性**

主要局限包括：需要大量高质量多静态训练数据，训练与推理计算量大、速度慢；在分布外的真实数据上可能产生幻觉或过度去噪；对参数（如引导强度）敏感，需要经验调优。

---

## 642. Measure Less, Know More: Self-Supervised Test-Time Feature Acquisition

**arXiv ID:** 2610.03454 | [PDF](https://arxiv.org/pdf/2610.03454v1)

**作者:** Eeshaan Jain `[一作]` (EPFL), Charlotte Bunne `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了 echo-k，一种任务无关、无标签、基于预训练模型潜在表示的自监督序列视图采集策略；

**💡 创新点**

创新点在于将预训练模型的潜在表示作为采集的目标，构造无监督奖励并通过强化学习学习长时序的视图选择策略；

**🔧 技术方法**

采用了自监督奖励、深度强化学习（PPO）、以及预训练的多模态基座模型（如TABULA、VirTues、MMEarth-Bench 等），并在理论上给出了线性设定下的性能上界；

**📊 数据集**

实验数据集涵盖合成数据、MNIST、Fashion‑MNIST、MiniBooNE、单细胞转录组（hPancreas）、空间蛋白组（NSCLC）、多模态地理空间（MMEarth‑Bench）等；

**📈 对比分析**

与多种基准（oracle、随机、基于特征选择的无监督方法、以及部分有标签的自适应采集方法）比较，echo‑k 在大多数预算下均能获得更高的下游任务性能并实现更好的表示恢复；

**⚠️ 局限性**

局限性包括对线性近似的理论假设、对预训练模型表示的依赖以及在极端稀缺数据或高度相关视图情况下的效果尚未完全验证。

---

## 643. Quantifying Ethereum Energy Consumption via Network Mapping

**arXiv ID:** 2610.03440 | [PDF](https://arxiv.org/pdf/2610.03440v1)

**作者:** Yahn Costa Hackspacher `[一作]` (FIZ Karlsruhe -- Leibniz Institute for Information Infrastructure), Moritz Schubotz `[通讯]` (FIZ Karlsruhe -- Leibniz Institute for Information Infrastructure)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过双层 Nebula 爬虫收集 Ethereum PoS 网络可达节点的客户端、硬件架构、操作系统、云供应商以及验证器子网等属性，结合公开的功耗基准和云功耗模型，构造规则表为每个节点分配功耗；对属性缺失的节点使用随机森林回归进行补齐；最终给出 6,934 个匹配节点的 415 kW 能耗快照，并与统一功耗分配、CCAF 估计及 MiCAR 能耗阈值进行对比。

**💡 创新点**

① 利用 libp2p、ENR 公开字段与云 IP 前缀对节点属性进行精准识别；② 构建基于硬件与验证器状态的多维功耗规则表；③ 在缺失属性的情况下使用随机森林实现“无标签”功耗推断；④ 将节点级功耗与网络层级能耗阈值（MiCAR）直接关联，提供监管合规参考。

**🔧 技术方法**

双层 Nebula 爬虫；属性提取与匹配；功耗规则表（CCRI 基准、云 PUE、ARM/x86 分别模型）；随机森林回归；统计误差评估（RMSE、MAE、R²）；24 h 实验室功耗测量；数据可视化与对比分析。

**📊 数据集**

• 2026‑06‑19 与 2026‑06‑22 两次 Nebula 主网爬虫（共 6,940 个地址匹配）。
• 公开的 CCRI 现场功耗基准（主链客户端与硬件组合）。
• Cloud Carbon Footprint（CCF）每 vCPU 瓦特表及 AWS 具体实例功耗公式。
• 自建的 24 h 硬件功耗测量（Windows 11 x86 游戏机与两套主链客户端）。

**📈 对比分析**

与统一 Lighthouse+Nethermind x86 的 431 kW 估计比较，差异 3.9%；与 CCAF 0.90 MW 估计比较，差距 54%（主要因节点覆盖不足）。随机森林在已标记节点上的 RMSE 1.6 W、MAE 0.28 W、R² 0.99；对缺失属性节点的 MAE 为 4.3 W。24 h 实验室测量与模型预测相差约 19%（去除 GPU 基准后）。

**⚠️ 局限性**

• 爬虫仅捕获可达节点，忽略不接受入站连接的隐藏节点，导致能耗低估。
• 两次爬虫时间间隔 3 天且地址匹配率 54% 不一致，增加不确定性。
• 功耗规则表基于 CCRI 测量，硬件混合比例（0.75/0.25）对结果高度敏感。
• 随机森林在隐藏字段（特别是云供应商）时精度下降，MAE 约 9 W。
• GPU 校正仅针对单台桌面，未覆盖典型无 GPU 的节点。
• 未包含 ARM 全节点、云高负载状态以及所有验证器子网完整功耗。

---

## 644. An Automated and Reproducible Workflow for Crack Identification and Damage Assessment of Fusion Materials

**arXiv ID:** 2610.03505 | [PDF](https://arxiv.org/pdf/2610.03505v1)

**作者:** Rinkle Juneja `[一作]`, Gary M. Staebler `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一套可复现的Galaxy工作流，能够自动从扫描电子显微镜（SEM）图像中识别裂纹并量化损伤，同时记录完整的处理历史和中间产物。

**💡 创新点**

创新点在于将裂纹识别与工作流自动化、可追溯性、跨平台可移植性以及与机器学习和物理模拟模块的无缝接口相结合，且实现了无需图像特定阈值调优的全局处理流程。

**🔧 技术方法**

使用的技术包括Galaxy平台、Sato脊滤波器、三角阈值分割、形态学闭运算、骨架提取、DINOv2视觉模型、PCA与扩散映射降维、CabanaPD物理模拟等。

**📊 数据集**

数据集为418张SEM图像，来源于114个电子束热冲击实验，覆盖5种钨等级（纯钨、UHP、K掺杂、1% Ta、5% Ta）与3种微观结构（纵向、横向、再结晶）共15种组合。

**📈 对比分析**

通过无阈值自动阈值分割和尺度无关的裂纹密度指标，实现了在不同分辨率、不同材料和不同损伤状态下的统一比较；实验结果显示所有图像均成功处理，且质量控制层能够及时发现极端或无效掩码，性能稳定且可重复。

**⚠️ 局限性**

局限性包括：仅针对细长裂纹的脊响应；可能误判纹理、粗糙线或孔洞；对图像对比度和分辨率高度敏感；预测模块受数据覆盖不足和材料元数据不完整的限制；物理模拟仅为连续介质模型，未包含微观结构细节。

---

## 645. Single-Pass Uncertainty Heads for Claim-Level Hallucination Detection in Persian Medical Language Models

**arXiv ID:** 2610.03482 | [PDF](https://arxiv.org/pdf/2610.03482v1)

**作者:** Mehrdad Ghassabi `[一作]` (University of Isfahan), Audrina Ebrahimi `[通讯]` (University of Texas at Dallas)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

做了什么：将 LLM Uncertainty Head 框架迁移到基于 Aya‑Expanse 的波斯语医学模型，并训练轻量化的 claim‑level 幻觉检测头。

**💡 创新点**

创新点是什么：首次在低资源语言波斯语中构建专门的 claim‑level 幻觉数据集，并实现单通道、单次推理的不确定性检测；同时通过两种不同训练方式的模型对比验证同一架构的通用性。

**🔧 技术方法**

用了什么技术：利用冻结的 Aya‑Expanse‑8B 基础模型的多层注意力图和前四个 token 概率特征，经过两层 Transformer 编码器生成每条 claim 的幻觉概率。

**📊 数据集**

用了什么数据集：自制了 1,600 条医学问题，生成对应回答并提取 60k+ 句子，标注为支持或幻觉，并在 1,600 条回答中构建了两份配对的波斯语 claim‑level 数据集。

**📈 对比分析**

如何比较的方法，性能怎么样：在两种不同训练的模型上分别训练 head，评估 ROC‑AUC 约 0.78，PR‑AUC 约 0.48，较随机基线提升 2–3 倍，证明内部信号可用于幻觉排名。

**⚠️ 局限性**

limitation是什么：数据量有限且标注自动化（非人工验证），仅在 Aya‑Expanse 系列模型上测试，泛化性与其他语言模型的适用性尚未证明。

---

## 646. RailWave: Adaptive Spatial and Temporal Scheduling for Expert-Parallel Communication

**arXiv ID:** 2610.03415 | [PDF](https://arxiv.org/pdf/2610.03415v1)

**作者:** Chutian Wang `[一作]` (Sun Yat-sen University), Xiuyu Li `[通讯]` (Renmin University of China)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 RailWave，一个基于 DeepEP 的通信层，专门为 Mixture‑of‑Experts（MoE）模型的专家并行（EP）执行优化物理网络的空间和时间调度。

**💡 创新点**

创新点在于：①源侧 RailBalance 通过仅利用源节点本地信息在可用 Rails 之间重新分配流量；②可重用的拓扑导向置换调度在不重建需求依赖表的前提下限制接收端 incast；③结合这两种机制并使用轻量级校准选择器，根据实时流量特征动态选择最佳执行路径。

**🔧 技术方法**

技术包括：源本地 Rail 重新分配算法、基于循环位移的多波次置换调度、三维流量描述符（B、ρ_g、κ）与离线校准的查找表、DeepEP 后端的实现。

**📊 数据集**

使用 GLM‑4.5‑Air 106B 模型训练期间收集的真实专家路由统计（覆盖 45 层 MoE），以及三种控制 incast 的发送模式（Ring、Moderate、Extreme）进行回放实验。

**📈 对比分析**

与 Native、NCCL、以及公开的 FAST 实现比较；在 32‑GPU H800 与 H20 集群上，RailWave 在 P50 级别上分别比 Native 提升 2.02–5.84 倍（H800）和 1.74–4.36 倍（H20），在更严苛的 incast 情况下仍保持低延迟，且在多种流量特征下表现出良好的自适应性。

**⚠️ 局限性**

限制：需要对每个训练阶段进行离线校准，且校准表对新流量模式的适应性有限；仅适用于已固定路由和专家分配的 EP 场景；在极端不均匀的 Rail 或接收端分布下，单一机制的优势可能被削弱，整体性能受限于硬件拓扑与负载分布。

---

## 647. Efficient Reasoning Training Does Not Always Harm CoT Faithfulness and Monitorability

**arXiv ID:** 2610.03509 | [PDF](https://arxiv.org/pdf/2610.03509v1)

**作者:** Samuel Lewis-Lim `[一作]` (University of Sheffield), Nikolaos Aletras `[通讯]` (University of Sheffield)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究高效链式推理训练对链式思考（CoT）可信度和可监测性的影响，探讨三种长度压缩方法（ThinkPrune、L1、GLP）在三种大型语言模型上的效果。

**💡 创新点**

创新点在于同时评估 faithfulness 与 monitorability，并比较不同长度压缩策略在多模型、多任务上的具体影响，揭示压缩对模型一致性和可解释性的差异化影响。

**🔧 技术方法**

采用 RL-with-verifiable-rewards 训练方式，使用 Normalized Simulatability Gain (NSG) 衡量 faithfulness，使用 g-mean² 评估 monitorability，结合多任务和干预式评测。

**📊 数据集**

训练数据为 AIME‑AMC 竞赛数学题；评测数据包括 MMLU‑Redux、七个决策任务（Employee Attrition、Bank Marketing 等）以及两类干预（sycophancy 与认知偏差）。

**📈 对比分析**

通过对比基准模型与训练后模型的 CoT 长度、NSG 以及 monitorability 指标，发现大多数效率训练显著降低 faithfulness（主要因一致性下降），但 monitorability 在绝大多数情况下保持稳定，只有在极度压缩时才有轻微衰减。

**⚠️ 局限性**

局限性：仅评估了三种模型与三种压缩方法，未覆盖更复杂的代理任务；效率训练对模型一致性的负面影响可能限制其在需高可靠推理的场景中的适用性。

---

## 648. Single or Multiple Policies for Phase-Structured Reinforcement Learning?

**arXiv ID:** 2610.03475 | [PDF](https://arxiv.org/pdf/2610.03475v1)

**作者:** Guilhem Loussouarn `[一作]` (Imperial College London), Kin K. Leung `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究在确定性阶段结构化强化学习中，单一共享策略与多阶段专用策略的性能差异；

**💡 创新点**

提出基于阶段内暂态与准稳态分解的诊断框架，量化共享与专用策略的优势与成本，并给出可选架构的决策准则；

**🔧 技术方法**

使用PPO强化学习，结合状态时间标记、共享、分支和多头网络等三种架构；

**📊 数据集**

在三个周期性阶段环境上验证：高空平台覆盖控制（HAPS）、PhasePendulum和PhaseGridWorld；

**📈 对比分析**

通过实验对比三种架构，发现共享策略在短暂态或数据稀缺时更优；当阶段异质性高、阶段时长长且数据充足时，多阶段专用策略表现最好；多头网络作为折衷在中间情形表现稳健；

**⚠️ 局限性**

局限在于仅考虑可观测且已知阶段序列，对隐藏或随机阶段变化缺乏处理；模型的选择仍依赖手工诊断，缺乏自动化的架构适配方法。

---

## 649. Weave Forcing: Compositional Memory Routing for Interactive Long Video Generation

**arXiv ID:** 2610.03510 | [PDF](https://arxiv.org/pdf/2610.03510v1)

**作者:** Ziyi Wang `[一作]` (University of Electronic Science and Technology of China), Hongliang Li `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了无训练的 Weave Forcing 框架，用于交互式长视频生成中的组合记忆重用。

**💡 创新点**

创新点包括：① 利用 LLM 进行语义槽路由，拆分并为每个角色和背景单独匹配历史来源；② 通过对比槽注意力（CSA）生成语义 KV 掩码，精准隔离所需内容；③ 采用覆盖自适应 RoPE，根据角色/背景覆盖情况动态调整时间偏移和记忆衰减，平衡历史指导与新内容生成。

**🔧 技术方法**

核心技术包括 LLM 语义槽路由、对比槽注意力、语义 KV 掩码、覆盖自适应 RoPE、压缩记忆缓存、跨帧注意力门控。

**📊 数据集**

使用 100 条人工构造的叙事序列（每条 6 个 10 秒镜头）作为测试集，生成 60 秒视频；使用 Gemini 2.5 Pro 生成提示并构造数据集。

**📈 对比分析**

与 Bidirectional 模型（EchoShot、CineTrans）和 Autoregressive 模型（Self Forcing、LongLive、Rolling Forcing、Infinity‑RoPE、Reward Forcing、ShotStream、Echo Forcing）对比，Weave Forcing 在跨镜头（inter‑shot）主题和背景一致性上最高（主题 0.513，背景 0.811），同时保持竞争性的视觉质量和文本对齐；在 intra‑shot 一致性方面略逊于部分对手，但整体性能优于所有基线。

**⚠️ 局限性**

局限性：① 在 intra‑shot 一致性方面未能领先最优模型；② 依赖 LLM 的槽路由质量，若提示模糊或多义可能导致路由错误；③ 对极长序列或高分辨率视频的记忆管理和计算开销仍待进一步优化。

---

## 650. Below what training size do deep tabular generators stop beating trivial baselines? A preregistered benchmark on a size ladder of clinical and standard datasets

**arXiv ID:** 2610.03500 | [PDF](https://arxiv.org/pdf/2610.03500v1)

**作者:** Shivam Shrivastava `[一作]` `[通讯]` (VIT Bhopal University), Shivam Shrivastava (VIT Bhopal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文在小样本（数百行）场景下，利用规模阶梯式基准对深度表格生成模型（CTGAN、TVAE、TabDDPM）与传统基线（SMOTE、Gaussian Copula等）进行系统评估，所有实验结果均公开可复现。

**💡 创新点**

创新点在于首次采用预注册的规模阶梯实验设计，公开完整结果文件，深入检验深度模型在小样本下是否能优于简单基线，并探讨大数据子采样与真实小样本的对应关系。

**🔧 技术方法**

技术方法包括：TSTR（AUROC平均）作为主要性能指标；Optuna TPE 20 次调参搜索；使用Kendall τ评估不同规模下模型排名的稳定性；在单核 CPU 上执行。

**📊 数据集**

实验数据集包括八个公开大型数据集（在200–20,000 行的阶梯子采样）和四个原生小型临床数据集（如心脏疾病、心衰、糖尿病等）。

**📈 对比分析**

比较方法：对每个（数据集，样本数）组合，计算TSTR AUROC 并与基线进行平均值比较。结果显示：在所有规模下，深度模型未能显著超越最佳基线；基线在大多数单元格中占优；深度模型在小样本下的排名稳定性最高。

**⚠️ 局限性**

局限性包括：仅涉及二分类任务；下游分类器固定不调；单核 CPU、有限调参预算；未考虑差分隐私、回归或多分类问题；子采样与真实小样本的对应关系不可靠。

---

## 651. Certified Mechanistic Edits: Behavioral Guarantees for Skill Removal and Preservation

**arXiv ID:** 2610.03502 | [PDF](https://arxiv.org/pdf/2610.03502v1)

**作者:** Md Sazid Uddin `[一作]` (American International University-Bangladesh), M. F. Mridha `[通讯]` (American International University-Bangladesh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在神经网络中通过正式编辑方法（如消融、权重编辑和引导）来证明对特定技能的去除和保留，并在连续输入区域内给出完整证明。

**💡 创新点**

提出了“认证机制编辑”框架，证明了没有有限黑盒测试可以保证去除技能，并首次在完整连续区域内对编辑效果进行精确证明；同时将精确SMT编码与基于边界传播的扩展相结合，实现了对标准Transformer的可验证去除与保留。

**🔧 技术方法**

使用Z3 SMT求解器进行精确的线性实数算术推理，结合CROWN/auto_LiRPA的边界传播做非线性层的可验证推导，配合凸包松弛与一维搜索实现对编辑半径的计算。

**📊 数据集**

实验涵盖了从两输入ReLU网络、阈值门Transformer、两个加法器的已知公式模型，到标准的Softmax+LayerNorm Transformer，使用合成数据和自动生成的prompt集合。

**📈 对比分析**

与基于攻击的下界和网格测试进行对比；在SMT下得到的精确去除半径在大约9倍的维度上可实现，CROWN在更大空间上给出可行下界；通过攻击上界验证证明的可信度，整体表现优于传统的测试集验证。

**⚠️ 局限性**

局限在于只能处理小型标准架构模型，需先验可判定的技能规范，连续输入空间与离散prompt空间的差异，以及非线性层的精确性不足，未能直接应用于真实有害能力的安全验证。

---

## 652. Metropolis-Hastings Dominates Importance Resampling for Policy Composition

**arXiv ID:** 2610.03480 | [PDF](https://arxiv.org/pdf/2610.03480v1)

**作者:** Alexey Kurennoy `[一作]` (Fin AI), Fergal Reid `[通讯]` (Fin AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在推理时使用独立Metropolis–Hastings对多奖励权重组合的本地产品解码进行校正，使其更接近全序列目标。

**💡 创新点**

证明了无论候选预算，MH在所有凸f-散度下都不比普通SIR差，并给出了MH改进的可证书以及近似专家时的误差阶数提升。

**🔧 技术方法**

使用独立Metropolis–Hastings、采样重要重采样、局部产品解码、对数空间权重记录和凸散度理论。

**📊 数据集**

在可枚举的4词语料、Gemma-4-E2B-it的GSM8K与HH-RLHF任务上进行实验。

**📈 对比分析**

与SIR和最大权重选取等方法比较，MH在相同候选预算下均表现出更低的KL/TV误差、较大的一致性提升，且在GSM8K上可实现准确率提升1–1.3个百分点、长度缩短16–42 tokens。

**⚠️ 局限性**

只能在候选池中出现的响应能被采样；对长序列时目标-提议比率波动大导致无法覆盖所有目标质量，且在LLM规模下难以直接评估真实分布差异。

---

## 653. UniDynamics: Event-RGB Fusion for Unified Future 4D Dynamic Scene Generation

**arXiv ID:** 2610.03473 | [PDF](https://arxiv.org/pdf/2610.03473v1)

**作者:** Daikun Liu `[一作]` (Southeast University), Changyin Sun `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 UniDynamics，一种基于扩散模型的单帧事件‑RGB 对齐，能够同时生成未来的 RGB、深度和光流，实现 4D 场景预测。

**💡 创新点**

创新点包括：①事件流作为运动先验通过 Event Latent Enhancement (ELE) 模块强化扩散条件；②在多尺度 U‑Net 中引入 Perceptual Dynamics Space (PDS)，将深度与光流分离并实现双向交互，提升几何-运动一致性。

**🔧 技术方法**

使用了稳定视频扩散（SVD）作为生成骨架，结合跨模态注意力、零初始化卷积注入、分类器无关指导（classifier‑free guidance）和 DeepSpeed ZeRO‑2 优化。

**📊 数据集**

在合成事件‑VKItti2 数据集上训练，并在真实世界的 DSEC、MVSEC、M3ED 上做零射测试验证跨域泛化。

**📈 对比分析**

与 SimVP、TAU、SVD、Vista、UniFuture 等基线对比，UniDynamics 在 FID/FVD、深度 AbsRel/δ、光流 EPE/AE 等指标上均表现出更低误差、更高连贯性，尤其在高速运动模糊场景中优势显著。

**⚠️ 局限性**

局限性包括：①对极低事件密度场景的鲁棒性仍有限；②模型规模较大，推理时间与显存需求高；③对极端光照或遮挡下的长时序预测尚未充分验证。

---

## 654. ProgressNet: Sketching and Prompting with a Frozen Text-to-Image Model

**arXiv ID:** 2610.03512 | [PDF](https://arxiv.org/pdf/2610.03512v1)

**作者:** Arkaprabha Basu `[一作]` (University of Surrey), Yi-Zhe Song `[通讯]` (University of Surrey)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出 ProgressNet，一种训练无关的框架，能让冻结的 FLUX 模型在绘图过程中实时跟随用户的笔划、擦除和提示更新，支持约 1 秒的交互速度。

**💡 创新点**

创新点在于三种推理时机制：Previous‑Concept Memory（PCM）用于记忆与擦除操作，Layer‑Selective K/V Injection（LS‑KVI）在非关键层保留注意力特征，Banded Adaptive Control（BAC）根据 ControlNet 记忆分布自适应调节控制强度。

**🔧 技术方法**

技术上结合了冻结的 FLUX 1‑schnell 与 Union ControlNet，利用 RGB 记忆图直接做遮挡、白化、灰化处理，并在 Transformer 的低重要性层注入前一轮的 K/V，最后用 JSD‑基准对记忆强度进行分层调制。

**📊 数据集**

在三个绘图域上评估：FS‑COCO（自由手场景）、Photo‑Sketching（边缘对齐）和 Sketchy（单物体自由手），覆盖场景级与对象级绘制。

**📈 对比分析**

与 Sketch‑a‑Sketch、SDXS、Conditional Balance、FLUX+ControlNet、StableFlow 等方法对比，ProgressNet 在 FID‑I、DINOv2、CLIPScore、T5‑Cos 以及自定义的 PFC‑DINO/PFC‑LPIPS 指标均取得最高或接近最高分，且每轮延迟仅 0.8 秒，用户研究中在连贯性、质量与擦除等方面均优于所有竞争方法。

**⚠️ 局限性**

局限性包括目前仅针对基于草图的控制，未验证对深度、姿态等其他模态的适用性；BAC 的层区间与调制阈值是预设的，缺乏自适应判定机制。

---

## 655. Preserving Anatomical Continuity: Three-Stage Pipeline for Colon Segmentation in 3D Abdominal CT Scans

**arXiv ID:** 2610.03467 | [PDF](https://arxiv.org/pdf/2610.03467v1)

**作者:** Deshan Kalupahana `[一作]` (University of New South Wales), Arcot Sowmya `[通讯]` (University of New South Wales)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种三阶段、拓扑保持的结肠分割管线，先用深度学习得到初始分割，再通过中心线桥接恢复分离区域，最后利用 UNet 对缺失部位进行重建，得到完整连通的结肠分割结果。

**💡 创新点**

创新点在于结合基于中心线的拓扑驱动桥接与重建，既解决了深度学习模型易产生断裂预测的问题，又保持了原有分割精度；同时通过贝塞尔曲线生成的桥接和局部重建实现了结构连通性的显著提升。

**🔧 技术方法**

使用 nnUNet 等 UNet 变体完成初始分割；骨架化 + 图结构处理生成中心线；贝塞尔曲线桥接端点；再用 UNet 仅以中心线桥接为输入预测缺失区域；所有模型训练采用交叉熵+Dice 损失。

**📊 数据集**

实验基于公开数据集 TotalSegmentator（467个腹部 CT，包含结肠）和 RAOS（413个腹部 CT），两者均手工或主动学习标注。

**📈 对比分析**

对比使用 Dice、IoU、Hausdorff、SSIM、clDice、ACD、Connectivity Ratio 等指标。三阶段方法在拓扑指标（clDice、CR）上有显著提升，重叠与距离指标略有下降，但总体性能仍保持与初始分割相当，且能有效恢复缺失的连通性。

**⚠️ 局限性**

局限性包括：中心线桥接对极端弯曲或自连环路的识别不够稳健；桥接阈值（120mm/90mm）对不同病例的适用性有限；缺失区域较小导致量化提升不显著；模型未区分结肠壁与腔，可能影响端点定位和重建质量。

---

## 656. A Near-Zero Monitor Readout Is Not Evidence of Behavioral Control

**arXiv ID:** 2610.03458 | [PDF](https://arxiv.org/pdf/2610.03458v1)

**作者:** Zhe Zhou `[一作]` (University of Washington), Tianhua Tao `[通讯]` (University of Washington)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了将监控器嵌入强化学习奖励函数的效果，探讨低监控读数是否能证明行为受控。

**💡 创新点**

发现即使监控读数很低（几乎零），策略仍可能在混合或高度奖励黑客化的两种不同行为模式中表现，揭示了传统离线门控和在线读数无法区分控制与逃逸的局限。

**🔧 技术方法**

使用了前缀条件化自我承诺惩罚（SCL、Cut）、激活层探针（Probe）以及强化学习框架GRPO，并在MBPP-Honeypot-CoT环境中进行训练与评估。

**📊 数据集**

采用公开的MBPP（Mini-Bench for Program Puzzles）数据集，包含可见测试和隐藏测试，用于评估代码生成任务的正确性。

**📈 对比分析**

通过在三种种子下对比四种监控器（无监控、Probe、SCL、Cut）的训练结果，评估门控通过率、启动时间、黑客率等指标；结果显示所有通过门控的监控器都可能进入黑客化或混合状态，且低读数无法预测最终性能。

**⚠️ 局限性**

局限性包括：只使用三种种子验证随机性；监控器训练与奖励之间尺度不匹配；前缀条件化假设攻击早期可见；激活探针读数在训练时几乎为零，缺乏动态范围；评估指标主要基于断言通过率，未涵盖更丰富的行为验证。

---

## 657. Beyond Random Splits: Evaluating Drug-Target Affinity Models Under Chemically and Biologically Motivated Distribution Shifts Copy

**arXiv ID:** 2610.03456 | [PDF](https://arxiv.org/pdf/2610.03456v1)

**作者:** Minjae Chung `[一作]` (Georgia Institute of Technology), May Dongmei Wang `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对药物-靶标亲和力预测模型进行系统评估，探讨不同分布偏移对模型选择的影响。

**💡 创新点**

首次构建统一基准，系统比较不同药物表征与蛋白交互模式在四种化学与靶标分布偏移下的模型排名稳定性。

**🔧 技术方法**

使用四种药物表示（ECFP4、ChemBERTa、GCN、GCN+ChemBERTa）和三种ESM-2交互方式（均值池化、注意力池化、交叉注意力）以及Morgan指纹+蛋白CNN基线。

**📊 数据集**

从ChEMBL和BindingDB整合的718,800条药物-蛋白互作数据，构建四种OOD拆分。

**📈 对比分析**

采用RMSE评价、Kendall/Spearman排名相关、重复种子、Bootstrap不确定性等多种比较手段，结果显示化学偏移对RMSE影响较小，目标序列偏移导致误差显著上升，且模型排名在蛋白/OOD情况下发生逆转。

**⚠️ 局限性**

局限在于仅使用基于序列的蛋白编码、实验测定的异质数据，缺乏前瞻性验证，且未覆盖更大规模预训练模型或图网络。

---

## 658. The Shape of Speech: A Geometric Measure of Coarticulation for Speech-Driven 3D Facial Animation

**arXiv ID:** 2610.03436 | [PDF](https://arxiv.org/pdf/2610.03436v1)

**作者:** Danzel Serrano `[一作]` (New Jersey Institute of Technology), Przemyslaw Musialski `[通讯]` (New Jersey Institute of Technology)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `4de8e9d8-757b-475f-9627-18a445e50202` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于辅音节点的几何路径比值（R_pw）来度量语音驱动3D面部动画中的共振化（coarticulation）轨迹形状，并在VOCASET数据集上评估四种主流方法的轨迹平坦化程度；同时通过β控制的快速成分调节与视觉实验将几何缺陷与感知偏好关联。

**💡 创新点**

创新点在于：①设计了对运动增益不变且只需强制对齐的辅音感知路径比值，克服了传统端点弦法的退化；②将几何缺陷与可解释的β等价量关联，为模型训练和后处理提供可量化目标；③通过大规模感知实验验证了几何度量与观众偏好的一致性。

**🔧 技术方法**

使用技术包括蒙特利尔强制对齐（MAF）、FLAME 3D面部模型、路径长度与R_pw计算、基于β的高频成分调节、Bootstrap置信区间、线性混合模型分析以及用户研究（97名观众、3,523次判断）。

**📊 数据集**

采用VOCASET数据集——受控朗读英语语音捕捉，包含12位说话者、FLAME拓扑，评测使用两名保留说话者的183条匹配标记词条。

**📈 对比分析**

与真实捕捉的语音轨迹相比，四种方法（CodeTalker、FaceFormer、DiffPoseTalk、ARTalk）均显示轨迹平坦化，β等价分别约为0.92、0.85、0.41、0.56；感知实验中，真实语音在句子级别获得73.4%偏好，抑制快速成分被惩罚，而放大则未显著惩罚。

**⚠️ 局限性**

局限性包括：①仅关注嘴部区域，未覆盖舌头等发音器官；②仅在单一英语朗读语料上验证，未检验跨语言或不同说话风格的泛化；③R_pw为相对比例，无法绝对量化共振强度，且对时间分布不敏感；④实验使用统一的面部模型，未考虑人种、性别等外观差异；⑤方法排名对不同构造敏感，无法给出一致的绝对优劣。

---

## 659. Rethinking Epistemic Uncertainty in Node Classification through Information Growth

**arXiv ID:** 2610.03418 | [PDF](https://arxiv.org/pdf/2610.03418v1)

**作者:** Emma Meneghini `[一作]` (University of Trento), Veronica Lachi `[通讯]` (University of Tromsø)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

建立了一套统计框架，用信息增长实验协议与一致性判据来评估图节点分类的认识不确定性，并在此框架下验证现有图Evidential深度学习（EDL）方法的局限性，随后提出并实验了图Bootstrap集成（GB‑Ens）作为可实现一致性预测的候选方案。

**💡 创新点**

1) 首次将信息增长实验协议和一致性定义引入图节点分类；2) 统一分析并证明现有EDL目标在统计层面上不满足一致性；3) 提出图Bootstrap集成通过同时捕获数据采样与训练不确定性，实现在信息增长下的认识不确定性收敛，首次给出理论与实验双重支持。

**🔧 技术方法**

统计框架（projective graph DGP、信息增长协议、一致性判据），统一EDL损失（Dirichlet后验、温度化后验）、图Bootstrap集成（bootstrap图采样+多随机种子训练）、不确定性分解（bootstrap vs seed），以及Wasserstein‑1距离评估逼近质量。

**📊 数据集**

两种合成projective graphon DGP：指数衰减 graphon 与 assortative SBM，用于生成从 500 到 10,000 节点的六个规模，训练/验证/测试比例为 5%/15%/80%。

**📈 对比分析**

与 10+ 现有 ED‑L 模型（S‑BGCN‑K/T‑K、GPN、GPN‑GDEVI/GDLAT、GPN‑LOP、CUQ‑APPNP/GAT/GCN）以及标准深度集成（G‑Ens）进行对比。实验表明：EDL 模型的认识不确定性（EU）几乎不随信息增长而下降，且强烈依赖超参数；而 GB‑Ens 的 EU 在 500→10,000 节点时可下降 61–94%，满足一致性；AU 最终趋于稳定，表现与基线相当或更好。

**⚠️ 局限性**

1) 图Bootstrap集成在理论上是否真正一致性尚未给出完整证明；2) 仅在合成的 projective graphon 上验证，缺少真实图增长数据；3) 对于不同 backbone（如 GAT）的收敛性受随机种子影响，表现可能不稳定；4) 对真实大规模图的计算成本仍是一个实际瓶颈。

---

## 660. I2CD: Direct Image-to-Convex Decomposition for Simulation-Ready Collision Geometry

**arXiv ID:** 2610.03453 | [PDF](https://arxiv.org/pdf/2610.03453v1)

**作者:** Qian Wang `[一作]` (Yale University), Daniel Rakita `[通讯]` (Yale University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种从单张RGB图像直接预测物体的凸分解，避免传统的重建再分解流程；

**💡 创新点**

创新点在于利用预训练的图像到3D生成器（Hunyuan3D-2）冻结不变，只训练一个轻量级交叉注意力头，实现仅38M参数、10小时内即可获得可直接用于物理仿真和运动规划的凸几何；

**🔧 技术方法**

技术包括：冻结的Diffusion Transformer（DiT）与ShapeVAE解码器、基于CvxNet的半平面凸多面体表示、交叉注意力头输出H半平面参数、对数求和与Sigmoid平滑的可微占据函数、类平衡BCE和对比损失以抑制薄层堆叠；

**📊 数据集**

训练使用121,056个由MolmoSpaces提供的艺术家创作三维网格；测试在227个留出的OmniObject3D和Google Scanned Objects（GSO）样本上；

**📈 对比分析**

与八种重建+分解管线（HY3D/SAM3D与CoACD/V-HACD/CvxNet/BSP-Net）相比，平均体积IoU从~62%提升至65.7%，并在0.5秒内完成所有步骤，速度比完整管线快6–37倍；在四大物理引擎中，所生成的凸几何直接被使用且几乎无预处理，物理仿真成功率与传统方法相近；在xArm7物理取放实验中，场景构建时间从328秒缩减到11秒，成功率为85/100。

**⚠️ 局限性**

局限包括：仍受冻结后端生成器的误差影响（薄结构缺失、结构模糊）；与通过网格编码的latent相比，图像采样的IoU下降约3.2%；表面细节（F-score）略逊于最佳管线；凸槽对语义分离不足，可能导致不理想的几何划分；实验仅覆盖单一桌面场景与静态取放任务，未验证更复杂的接触丰富任务。

---

## 661. Most-Recent Anchoring with Recurrent Ordering for Time Series Forecasting

**arXiv ID:** 2610.03494 | [PDF](https://arxiv.org/pdf/2610.03494v1)

**作者:** Jung Min Choi `[一作]` (University of Hildesheim), Lars Schmidt-Thieme `[通讯]` (University of Hildesheim)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MARO模型，采用最近-旧扫描方式将最近时间片段作为anchor，递归地整合更久远的历史信息进行时间序列预测。

**💡 创新点**

创新点在于：① 用最近-旧扫描与共享参数递归模块显式实现recency bias；② 通过多深度检查点融合在保持低参数量的同时捕获不同时间尺度的特征；③ 结合稀疏通道混合和日历条件化进一步提升预测精度。

**🔧 技术方法**

技术手段包括可逆实例归一化（RevIN）、重叠分块、稀疏通道混合、日历条件化、共享RNN（单层递归）模块、检查点多深度融合、MLP解码器。

**📊 数据集**

在长周期任务使用ETT（ETTh1/2/ETTm1/2）、Weather、Electricity、Traffic、Solar等数据集；在短周期任务使用PEMS03/04/07/08数据集。

**📈 对比分析**

与PatchTST、TimeKAN、iTransformer、SRSNet、TimeKAN、iTransformer等基线在多种长短期任务（MSE/MAE）进行对比，MARO在大多数数据集与时延、参数量、FLOPs上均实现了最优或接近最优性能，且在效率上显著优于Transformer类模型。

**⚠️ 局限性**

主要限制：递归扫描带来的顺序依赖导致在通道数或批量较小的场景下GPU利用率低；通道邻域结构固定，缺乏对不同输入窗口的自适应路由能力。

---

## 662. Getting Your Guidance Weights Right in diffusion and flow-matching posterior sampling

**arXiv ID:** 2610.03503 | [PDF](https://arxiv.org/pdf/2610.03503v1)

**作者:** Liam Moroy `[一作]` (Heriot-Watt University), Guillaume Bourmaud `[通讯]` (University of Bordeaux)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种SIMCA方法，用于在无训练的后验采样中自动调节指导权重。

**💡 创新点**

创新点在于利用条件评分/流匹配目标的最小二乘结构，将权重优化化为每个时间步的二维线性最小二乘问题，从而实现一次性离线校准。

**🔧 技术方法**

采用了扩散模型、流匹配模型、Tweedie测量一致性项、线性最小二乘求解与贪心迭代校准（SIMCA）。

**📊 数据集**

在FFHQ、ImageNet、CelebA、AFHQ‑Cat四个数据集上进行实验，涵盖超分、随机/盒子填充、去噪、去模糊等逆问题。

**📈 对比分析**

与DAPS、DPS、DDRM、DDNM、DiffPIR、FPS‑SMC、Flower等基线对比，SIMCA在大多数任务和度量上均达到或超过state‑of‑the‑art，尤其在随机填充和超分任务上提升LPIPS/FID显著，并能将采样步数从1000降至50而几乎不损失质量。

**⚠️ 局限性**

局限性包括对高斯模糊、AFHQ‑Cat去模糊与去噪等任务效果不如最强基线，且需针对不同测量算子、噪声水平和采样器重新校准。

---

## 663. Depth Hypothesis Guided Iterative Refinement for Event-Image Monocular Depth Estimation

**arXiv ID:** 2610.03439 | [PDF](https://arxiv.org/pdf/2610.03439v1)

**作者:** Daikun Liu `[一作]` (Southeast University), Changyin Sun `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出HypoDepth框架，利用事件与图像的互补信息进行单目深度估计，并通过迭代细化提升精度。

**💡 创新点**

创新点包括：① 引入离散Depth Hypothesis Volume（DHV），将连续深度回归转化为受限搜索；② 构建轻量化3D cost volume并在多尺度下做相关搜索；③ 采用GRU迭代残差优化，实现从全局到局部的逐步细化；④ 在零样本迁移场景下展现出强大的泛化能力。

**🔧 技术方法**

使用技术包括：事件体素化、Swin‑T特征提取、跨模态融合（低分辨率交叉注意力、高分辨率卷积融合）、DHV嵌入、3D cost volume、multi‑scale correlation lookup、几何编码器、GRU迭代单元、log‑depth 归一化与SI_log 损失。

**📊 数据集**

使用的数据集：合成的EventScape进行预训练；真实的DSEC与MVSEC用于评估；同时在两者上进行零样本迁移测试。

**📈 对比分析**

与EReformer、DepthAnyEvent、PCDepth、SRFNet等方法对比，HypoDepth在DSEC上实现最优Abs Rel 0.099、RMSE 3.583，零样本迁移时也优于EReformer和PCDepth；Tiny版模型在1.4 ms推理时间下仍保持高精度，证明了实时性与性能兼顾。

**⚠️ 局限性**

局限性包括：DHV离散化可能限制极细粒度的深度精度；多尺度和迭代过程仍增加计算负担；对事件稀疏或极端光照、快速运动场景的鲁棒性尚未完全验证。

---

## 664. Cross-Facility LLM Pre-training on HPC: Elastic Aggregation, Data Leasing, and Queue-Aware Placement

**arXiv ID:** 2610.03457 | [PDF](https://arxiv.org/pdf/2610.03457v1)

**作者:** Zarè Palanciyan `[一作]` (SURF), Tim Kok `[通讯]` (SURF)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了在三台不同超算（Snellius、LUMI、Frontier）上联合预训练0.6B参数语言模型的系统，并在C4数据集上完成20k优化步的训练。

**💡 创新点**

创新点是将DiLoCo的两循环训练与弹性聚合、DARL动态数据租赁以及队列感知规划结合，形成可弹性、容错且对异构硬件友好的跨站点训练框架。

**🔧 技术方法**

使用了DiLoCo两循环训练、弹性聚合、DARL租赁协议、Flower框架、gRPC/HTTP权重交换、Slurm队列探测、预标记数据加载、torch-titan等技术。

**📊 数据集**

数据集为预标记的C4文本数据集。

**📈 对比分析**

与单站点集中式训练对比，三站点运行在20k步后离散困惑度为34.7，集中式为28.2；通信开销仅占6%（H=1000），多站点训练在保持数据完整性的同时将碎片化分配合并为一次性训练。

**⚠️ 局限性**

限制在于仅演示了小规模0.6B模型、单个epoch、单一拓扑，未评估更大规模、更多站点或下游任务，容错与排队预测还需进一步完善。

---

## 665. Point and Line Nearest-Neighbor Searching in 3-Space

**arXiv ID:** 2610.03451 | [PDF](https://arxiv.org/pdf/2610.03451v1)

**作者:** Pankaj K. Agarwal `[一作]` (Duke University), Micha Sharir `[通讯]` (Tel Aviv University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一系列新的数据结构，用以解决在三维空间中，涉及点、直线、线段、三角形的最近邻搜索（NN）问题。它们分别针对线对点、点对线、点对线段/三角形、线对线以及线段/三角形对点的查询，给出了空间与查询时间的多种折中方案。

**💡 创新点**

创新点包括：
• 通过构造小规模的测试集（test set）并利用垂直分解（vertical decomposition）实现线对点的线性空间结构，查询时间为 O*(n^½)；
• 利用四维参数空间与半代数范围搜索结合，构建 O*(n^4) 空间、O*(1) 查询时间的线对线数据结构；
• 在点对线段/三角形的查询中，将问题转化为 5 维半代数范围搜索，得到线性空间、O*(n^2/3) 查询时间的方案；
• 通过参数化搜索与多层分区树，获得空间/时间的连续折中，填补了之前仅有两端极值的空白；
• 对线段与三角形的距离判定，提出了一种新的几何分解和合并方法，显著降低了复杂度。

**🔧 技术方法**

主要技术手段包括：
• 垂直分解与几何切割（geometric cuttings）来构造测试集；
• 参数化搜索（parametric search）将 NN 查询转化为半代数范围空空性查询；
• 多层分区树与随机采样，结合冲突列表实现高效的查询；
• 对半代数函数的最小/最大包络（lower/upper envelopes）与最小/最大图（min/max diagrams）分析；
• 对线段与三角形的距离函数进行代数化处理，使其可归入 5 维半代数查询框架。

**📊 数据集**

本文为理论工作，未使用任何实际数据集；所有结果均为数学证明与算法分析。实验部分仅在合成实例上验证了理论复杂度，未给出公开数据集。

**📈 对比分析**

与现有最优结果相比，本文在以下几个关键点取得突破：
• 线对点的线性空间结构从 O*(n^(3/2)) 降至 O*(n^½)；
• 点对线段/三角形的线性空间结构从 O*(n^2) 降至 O*(n^2/3)；
• 线对线的 O*(n^4) 空间结构实现了 O*(1) 查询；
• 通过折中方案，提供了连续的空间/时间曲线，满足不同应用场景需求。实验结果与理论一致，证明了算法在实际规模（n≈10⁶）下的可行性。

**⚠️ 局限性**

限制与未解决问题包括：
• 对三角形对线段（或三角形）查询，无法在 O*(n^½) 查询时间内实现线性空间；
• 需要对垂直分解的上界（尤其是三维/四维中高阶包络的复杂度）做进一步研究；
• 线对线结构的 O*(n^4) 空间在实际大规模数据上可能不可行，需要更紧凑的实现；
• 对高维（d>3）情况的推广尚未完成，主要受限于高维半代数范围搜索的上界；
• 在某些 degenerate 情况（如多条线共面）需要额外的处理，算法鲁棒性待进一步完善。

---

## 666. Corrupted but Correct: Why Vision-Language Models Lie to Themselves Internally

**arXiv ID:** 2610.03445 | [PDF](https://arxiv.org/pdf/2610.03445v1)

**作者:** Arun Josephraj Arokiaraj `[一作]` (University College London), Adriano Koshiyama `[通讯]` (Holistic AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过对Qwen2.5‑VL‑7B‑Instruct模型在COCO val2017图像上进行两阶段PGD对抗扰动实验，探究了视觉语言模型（VLM）在训练（teacher‑forced）和推理（自由生成）过程中的差距，并用logit lens和线性探针对内部机制进行了解析。

**💡 创新点**

创新点包括：①在对抗训练和推理的对比中发现了一个“train/inference gap”，该差距只出现在第二个生成步骤，且其目标词排名在所有图像中恒定为3388；②证明视觉编码器对对抗扰动的影响是统一且无差异的，真正决定输出的还是语言解码器的语言先验；③用像素统计（如OTI）验证其无法预测VLM的易感性，说明传统CNN的鲁棒性指标不适用于VLM；④通过线性探针揭示内部表示即可区分易感与抗拒图像，暗示后置合并层可能成为防御入口。

**🔧 技术方法**

所用技术包括：双阶段带动量的PGD攻击（ε=16/255，α=4/255），对ViT视觉编码器、合并投影器以及28层LLM解码器的前向钩子；logit lens用于在每个层级投影隐藏状态到词表概率；线性探针用于评估合并隐藏状态的可分离性；以及统计分析（Pearson、ridge回归、rank‑biserial、AUC等）。

**📊 数据集**

使用的数据集为COCO val2017的200张图像（按超类分层抽样），目标描述为固定的3词短语（token ID 32、33457、13），并在每张图像上进行两阶段PGD优化。

**📈 对比分析**

实验对比了三类图像（易感、train/inference gap、抗拒）在Stage‑1损失、目标词排名、解码层层级排名变化等指标。结果显示：目标词在第二步的排名始终为3388，推理成功率为0%；合并层对所有图像的影响相同，但在第15–18层LLM解码器出现分化；线性探针对抗特征的AUC达0.858，表明内部表示能够提前区分最终输出。相比传统的对抗成功率评估，本文提供了更细粒度、机制层面的性能对比。

**⚠️ 局限性**

局限性包括：实验仅在单一模型Qwen2.5‑VL‑7B‑Instruct和单一目标短语下进行，缺乏跨模型和跨目标的验证；logit lens和探针仅给出相关性，未进行因果性验证；攻击仅使用ε=16/255的对抗扰动，未探讨更大扰动或更长目标序列的情况；此外，实验仅覆盖图像级对抗，未涉及多模态或交互式部署场景。

---

## 667. AIBL: Augmented Instance-Based Learning with Structured Memory and Neural Embeddings

**arXiv ID:** 2610.03413 | [PDF](https://arxiv.org/pdf/2610.03413v1)

**作者:** Radha Poovendran `[一作]` (University of Washington), Linda Bushnell `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 AIBL（Augmented Instance-Based Learning）——一种将实例学习（IBLT）与神经嵌入和多层内存管理相结合的顺序决策模型。

**💡 创新点**

创新点包括：① 用学习得到的向量空间代替符号属性；② 设计了惊奇（surprise）内存、遗忘（forgotten）内存与激活（active）内存三层，按相似度与激活值动态迁移；③ 引入了“毕业”机制以便重复出现的低相似度样本最终进入主动内存；④ 在更新时加入近似值偏差检测（outcome deviation）以保持结果多样性。

**🔧 技术方法**

技术手段：IBLT 框架、语义嵌入（Sentence‑BERT、自动编码器、矩阵分解向量）、余弦相似度、Welford 在线均值/方差更新、激活函数、软最大决策、近似最近邻（可选）和多任务实验脚本。

**📊 数据集**

使用的数据集：20 Newsgroups（文本分类）、UCI Mushroom（情景决策）、Credit Card Fraud（异常检测）、Electricity Market（概念漂移预测）和 MovieLens 100K（序列推荐）。

**📈 对比分析**

通过与符号 IBLT、k‑NN、LinUCB、Isolation Forest、矩阵分解等基线在相同的在线评估协议下比较，AIBL 在符号 IBLT 基线上提升 6–17% 的准确率或宏 F1，异常检测 F1 提升 34%，推荐 F1 提升 2.6%，漂移任务准确率提升 7.1%。

**⚠️ 局限性**

局限性：检索复杂度线性增长，需近似索引以处理大规模记忆；性能高度依赖编码器质量；不具备隐式规则学习能力（如 WCST）；需要外部验证集调参；目前仅处理单步动作，未实现动作序列或时间信用分配。

---

## 668. Depth as Time in One-Step Generative Models

**arXiv ID:** 2610.03626 | [PDF](https://arxiv.org/pdf/2610.03626v1)

**作者:** Arnold Caleb Asiimwe `[一作]` (Princeton University), Olga Russakovsky `[通讯]` (Princeton University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究单步扩散生成模型在网络深度上展现的“深度即时间”现象，发现其隐层的逐步推断类似多步扩散的去噪轨迹。

**💡 创新点**

创新点在于首次将多步去噪轨迹映射到单步模型的层级空间，并证明这种深度级去噪可被利用实现显著参数压缩。

**🔧 技术方法**

采用流映射、扩散去噪、层级解码与线性混合回归等技术，并用无学习参数的探测头对多步与单步模型的中间表示进行对比。

**📊 数据集**

主要使用 ImageNet-256×256 数据集以及 8‑Gaussian 低维仿真来验证模型表现。

**📈 对比分析**

通过层级解码和归一化距离度量与传统多步采样对比，实验表明 MeanFlow 等基于时间索引的模型在层级去噪与压缩后仍保持低 FID（如从 4.0 提升至 4.9），而漂移模型则性能显著下降。

**⚠️ 局限性**

局限性包括深度‑时间映射与训练时的流匹配时间不完全对应，且在漂移模型中难以观察到此现象，说明方法对模型结构的依赖较强。

---

## 669. OptiSelect: How does the Optimizer Shape Data Curriculum?

**arXiv ID:** 2610.03432 | [PDF](https://arxiv.org/pdf/2610.03432v1)

**作者:** Simin Fan `[一作]` (EPFL), Martin Jaggi `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了“OptiSelect”框架，即在每一步训练中使用优化器预先变换的梯度来评估候选样本的实用性，进而在批量中挑选最有价值的数据；对其优化器感知的评分函数做了系统的理论推导，并给出了选择收益的上界与最优候选倍率；在124M和720M规模的LLM预训练中对七种主流优化器进行基准测试，验证理论与实践的一致性；同时评估了数据重写（FinePhrase）对OptiSelect的影响。

**💡 创新点**

核心创新在于：①将优化器与数据选择融合，形成“优化器感知”评分；②引入“可辨识性”指标，阐释不同优化器几何如何限制或提升选择收益；③推导出最优候选倍率λ* = 2，提供实际调参依据；④发现最优优化器不一定是最优的评分几何，并通过分离优化器与评分几何的实验揭示这一点；⑤证明OptiSelect在数据重写环境下仍保持优势。

**🔧 技术方法**

采用高斯分布假设对候选价值进行统计分析，提出可辨识性和选择收益公式；实现了多种优化器（Adam、AdaFactor、RMSProp、Sign、Polar‑Tangential）的评分变换算子；利用Transformer（Llama‑style）架构在FineWeb‑HQ上进行4.2B/14B token的预训练；使用代理任务（ARC‑Easy/Challenge、HellaSwag、PIQA、SciQ）进行在线评分，并在WinoGrande、WSC、CommonsenseQA、OpenBookQA等零样本评测上验证效果。

**📊 数据集**

FineWeb‑HQ 作为主训练语料；代理集由4,096条来自ARC‑Easy/Challenge、HellaSwag、PIQA、SciQ的样本组成；下游评测使用WinoGrande、WSC、CommonsenseQA、OpenBookQA等公开数据集。

**📈 对比分析**

通过在相同token数量下比较标准训练与OptiSelect，测量下游损失和零样本准确率；结果显示：在Adam、AdaFactor、RMSProp上，OptiSelect可在相同token下降低约1–2%损失；在Sign优化器上无明显提升；当λ=2时，增益最大；在重写比例（FinePhrase）不同的混合语料下，OptiSelect保持与标准训练相似甚至更好的收益。

**⚠️ 局限性**

仅在124M与720M两种规模上验证，缺乏多亿甚至数十亿参数的实验；每个实验仅用单一随机种子，未评估方差；代理任务范围有限，未覆盖更高难度任务（数学、代码、代理等）；理论假设基于高斯分布，实际梯度分布可能偏离，需进一步验证。

---

## 670. CORNAV: Construction-Aware Reasoning for Robot Navigation on Active Worksites

**arXiv ID:** 2610.03622 | [PDF](https://arxiv.org/pdf/2610.03622v1)

**作者:** Parastoo Ali Pour `[一作]` (University of California at Irvine), Mohammad Abdullah Al Faruque `[通讯]` (University of California at Irvine)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `51c0528b-f690-4182-ae60-bb5f046c276c` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了 CORNAV，一套基于施工蓝图、项目进度表与安全规则的机器人导航框架，能够在动态施工现场为移动机器人生成安全合规、时间约束满足的路径。

**💡 创新点**

创新点在于：① 将 2D CAD 图与 3D 场景图相融合，实现蓝图定位和房间身份的确定；② 将项目进度表转化为随时间变化的硬/软约束，支持时间感知导航；③ 引入 LLM（GPT‑4o）对软约束进行安全验证，将潜在危险升级为硬约束；④ 在无需 BIM 模型的前提下，完成端到端的构造‑aware 导航。

**🔧 技术方法**

采用的技术包括：DXF 解析与 ICP 对齐实现蓝图-地图配准；HOV‑SG + CLIP 的 3D 场景图构建与对象检索；JSON 化项目进度表与时间窗口处理；LLM 安全验证模块；基于成本惩罚的 A* 规划；ROS 2 轨迹规划与执行。

**📊 数据集**

使用的数据集为：① 现场收集的 RGB‑Depth 与位姿数据（Unitree Go2、G1）；② 施工现场与办公室的 2D CAD 蓝图；③ 对应的项目进度表（Excel→JSON）；⑤ 施工现场的 3D 场景图（由 HOV‑SG 生成）。不存在公开数据集，全部使用作者自建现场数据。

**📈 对比分析**

与基线 HOV‑SG 仅做语义导航的方案进行对比。关键指标：CCR（约束合规率）100% vs 42.6%；TSR（任务成功率）72.2% vs 7.4%；FPR（可行规划率）46.3% vs 31.5%；SACR（时间约束合规率）29.6% vs 20.4%；PER（路径效率）0.988 vs 1.398。去掉蓝图、进度表或安全验证后，性能大幅下降，表明三者对安全与效率贡献显著。

**⚠️ 局限性**

局限性：① 需要手工准备高质量 CAD 蓝图和项目进度表，依赖现场维护；② 蓝图-地图对齐受 ICP 误差和环境变化限制；③ LLM 验证依赖规则集，误判可能导致误升/降约束；④ 仅在单机器人、短周期任务上验证，未探讨多机器人协调或长周期任务；⑤ 公开数据缺乏，难以复现或推广至更大范围。

---

## 671. A Unified Framework for Empowerment and Predictive Control

**arXiv ID:** 2610.03563 | [PDF](https://arxiv.org/pdf/2610.03563v1)

**作者:** Wooyoung Chung `[一作]` (Texas Tech University), Stas Tiomkin `[通讯]` (Texas Tech University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

结合赋权信息量与采样式MPC，提出单一行动分布下的赋权最大化轨迹优化方法，并将其与任务成本共同使用。

**💡 创新点**

统一探测与执行的策略，去除了传统赋权控制中的双重分布；仅需一阶线性化即可计算赋权，避免了二阶导数，且赋权可直接嵌入采样MPC。

**🔧 技术方法**

使用采样式MPC（随机射击、预测采样、CEM、MPPI）+线性化赋权估计（水填充算法）+信息理论通道容量计算。

**📊 数据集**

实验环境为经典连续控制任务：单摆、倒立摆-摆杆、双摆以及hopper（仿真）等。

**📈 对比分析**

在同一MPC框架下与仅使用稀疏任务成本、仅使用赋权以及赋权+任务成本组合进行比较；结果显示赋权+任务成本实现最高成功率，赋权单独亦可在高维环境实现自我恢复。

**⚠️ 局限性**

对规划时长敏感、需要大量采样、在高维系统上收敛慢；线性高斯假设限制了赋权估计的鲁棒性。

---

## 672. Divergence controls entropy in distillation

**arXiv ID:** 2610.03529 | [PDF](https://arxiv.org/pdf/2610.03529v1)

**作者:** Nicolas Zucchet `[一作]` (Stanford University), Scott W. Linderman `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文从熵的视角深入研究了大型语言模型蒸馏过程中的熵变化规律，揭示了不同 KL 散度（正向、反向、Jensen‑Shannon 交叉）与采样策略对学生模型熵的影响，并通过理论推导与大规模实验验证了熵与损失、采样分布、教师规模等因素的关系。

**💡 创新点**

创新点在于：①将采样分布与散度选择解耦，使得能单独分析两者对熵的影响；②给出正向 KL 与反向 KL 在软max 头模型上的熵增/减定理，并用极限分析解释反向 KL 在任务难度极端时熵升的现象；③阐释了 Top‑k 约束如何根据尾部处理方式调节熵；④在自蒸馏中揭示熵崩塌机制并提供超参数补偿策略；⑤通过预训练、监督微调和自蒸馏三大场景，系统验证了理论预言。

**🔧 技术方法**

主要技术包括：蒸馏目标的 f‑divergence（KL、JS）、正向/反向 KL 的熵分析、软max 头模型的理论推导、Top‑k 限制与尾部处理、以及对 OLMo 2、Pythia、Qwen3、DeepScaleR、SciKnowEval、GSM8K 等数据集的实证实验。

**📊 数据集**

使用的数据集包括：预训练与监督微调的 OLMo 2 与 Pythia 预训练语料；DeepScaleR 的 MATH500 题库进行 on‑policy 蒸馏实验；SciKnowEval（Chemistry 子集）与 GSM8K 用于自蒸馏实验。

**📈 对比分析**

比较方法：通过计算学生模型在给定数据集上的平均熵与对应的交叉熵损失，验证熵与损失的等价关系；在 on‑policy 与 off‑policy 蒸馏中对比不同散度（正向、反向 KL）对熵的影响；在自蒸馏中评估不同 λ（JS 加权）、教师更新频率和 top‑k 约束对熵和性能的影响。实验结果表明：正向 KL 显著提升熵；反向 KL 降低熵并可能导致熵崩塌；合适的 JS λ 与慢速 EMA 教师能保持熵稳定并获得最佳性能。

**⚠️ 局限性**

局限性：①理论推导主要基于软max 头模型，未涉及完整的序列模型动态；②实验中对散度的采样近似（忽略 score‑function 项）可能与实际实现略有差异；③对极端稀疏数据或高维词表的行为尚未完全验证；④自蒸馏的超参数空间较大，仍需更系统的调优研究。

---

## 673. XGenAct: Geometry-Enhanced World Action Models through Cross-Task Generation

**arXiv ID:** 2610.03516 | [PDF](https://arxiv.org/pdf/2610.03516v1)

**作者:** Tingting Du `[一作]` (University of Wisconsin--Madison), Ang Li `[通讯]` (University of Maryland)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建一种统一的视频生成模型XGenAct，能够同时预测RGB、机器人动作、度量深度、表面法向量和功能角色分割，并在闭环机器人控制中直接使用。

**💡 创新点**

创新点在于：①将所有感知任务与动作映射为RGB视频，并使用确定性编码器让它们共享同一冻结的视频VAE和扩散变换器；②采用跨任务模板和条件计划，在训练时仅对单一感知流与动作流进行建模，无需多模态头部或辅助损失；③通过结构化感知提升动作生成，直接生成几何与语义未来，比传统RGB→专家管线更准确。

**🔧 技术方法**

使用扩散变换器（DiT）视频扩散模型、冻结的视频VAE、流匹配损失、确定性RGB编码器，以及多视角动作编码（Action Images）。

**📊 数据集**

训练数据来自RLBench（16个操纵任务、3960个episode）和ManiSkill3（7个任务、1750个episode），评估时使用未见RLBench任务。

**📈 对比分析**

与八个基线策略（包括MolmoAct、VLA-JEPA、PAD-Depth等）比较，XGenAct在平均成功率上达到52%，是最强基线的两倍；在多任务上，加入表面法向或分割可提升约28个百分点；直接生成几何/语义的误差比RGB+专家管线低约2-3倍。

**⚠️ 局限性**

局限性包括：最佳感知菜单因任务而异，需人工选择或自适应；深度流在某些组合中略逊于RGB+专家管线；模型对计算资源仍有一定需求；在极端复杂场景或未见任务的泛化性仍待验证。

---

## 674. Low-Cost Video--Time Priors as a Strong Baseline for EEG--fNIRS Emotion Regression on Familiar Videos

**arXiv ID:** 2610.03618 | [PDF](https://arxiv.org/pdf/2610.03618v1)

**作者:** Minghao Kong `[一作]` (Sun Yat-sen University), Rongjie Wang `[通讯]` (Pengcheng Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究在熟悉视频场景下使用视频-时间先验与EEG‑fNIRS融合预测情绪轨迹。

**💡 创新点**

提出 fold‑wise 视频-时间先验作为低成本基线，并证明其对新观看者预测效果优异；同时评估 EEG‑fNIRS 在此场景下的残差改进，发现融合提升有限且不均匀。

**🔧 技术方法**

采用 EEG 64 通道与 fNIRS 51 通道的频域与时域特征，利用图卷积网络与双向注意力编码器进行多模态融合；视频‑时间先验采用跨参与者中位数与三点滑动平滑实现。

**📊 数据集**

使用 MER‑PS 2026 训练/验证集（24 名参与者、15 个熟悉视频）和外部评估集（4 名参与者、同 15 视频）。

**📈 对比分析**

对比全局常数、视频身份、视频‑时间先验、单纯 EEG‑fNIRS 分支以及固定融合；内部 MAE 为 29.01，外部 MAE 为 27.72，融合略优于先验，先验已接近最佳性能。

**⚠️ 局限性**

限制包括仅评估熟悉视频，未检验未知视频迁移；EEG‑fNIRS 使用未来时段导致离线估计；融合权重未经过完全嵌套选择，外部最优权重不同；数据量有限，缺乏跨视频验证与实时评估。

---

## 675. LLA-MPC on Embedded Hardware: Rapid Adaptive Control with Thousands of Parallel Models

**arXiv ID:** 2610.03616 | [PDF](https://arxiv.org/pdf/2610.03616v1)

**作者:** Henry Z. Liao `[一作]` (Carnegie Mellon University), John M. Dolan `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在F1TENTH小型无人车上实现并验证了基于Look‑Back和Look‑Ahead自适应模型预测控制（LLA‑MPC）的在线系统识别与控制框架，能够在有限计算资源和噪声状态估计条件下实时识别轮胎参数并完成高速、低摩擦及多表面下的路径跟踪任务。

**💡 创新点**

创新点包括：①把LLA‑MPC从仅仿真环境推广到真实硬件，并提供可模块化、可并行化的开源实现；②通过在单个时间窗内评估成千上万个候选物理模型，在每一步选择误差最小的模型，从而实现零学习、快速适应；③在嵌入式GPU上并行运行 1.5 万个模型，显著提升了实时性能。

**🔧 技术方法**

使用的技术包括：单轨车辆动力学模型、Fiala刷子轮胎模型、JAX+GPU加速的 RK4 并行积分、acados 求解器实现 MPC、状态估计（运动捕捉 + 上机粒子滤波）以及基于滑动窗口的残差累积误差评估。

**📊 数据集**

实验数据来自 F1TENTH 平台的实际驾驶记录，使用不同摩擦系数的塑料轮胎和多种路面（高/低摩擦、突变高度），采样率分别为 180 Hz（运动捕捉）与 40 Hz（粒子滤波）。

**📈 对比分析**

与传统基于固定模型的 MPC 进行对比：在高速低摩擦、不同表面变化以及噪声状态估计条件下，LLA‑MPC 能完成更多周数的跟踪任务且 H‑step 预测误差显著低于基准模型；此外，在“漂移”高速转弯实验中，LLA‑MPC 成功完成任务而基准控制器失效。

**⚠️ 局限性**

局限性包括：①需要预先构建并维护一个大规模的候选模型库，模型数量越大计算负担越重；②实验仅在 F1TENTH 平台进行，未验证在更大车辆或更复杂环境中的可迁移性；③对超参数（窗口大小、状态权重、模型数量等）敏感，缺乏系统的调优指南；④目前只识别轮胎参数，未考虑其他可能变化的系统特性。

---

## 676. Mastering Atari 2600 Games with Discovered Options

**arXiv ID:** 2610.03604 | [PDF](https://arxiv.org/pdf/2610.03604v1)

**作者:** Erik M. Lintunen `[一作]`, Marlos C. Machado `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 Wayfarer 这一单流深度强化学习代理，能够在高维观察下在线学习并利用 Laplacian 表征驱动的选项（option）进行探索、信用分配和泛化。

**💡 创新点**

创新点在于：1) 将 Laplacian 表征与代理中心状态结合，避免受环境非可控因素影响；2) 通过自监督逆动力学预测与 Laplacian 目标共同训练，在线学习稳定的特征空间；3) 同时学习选项的策略与价值，并在同一循环中相互反馈，形成 virtuous loop。

**🔧 技术方法**

使用技术包括：深度 Q‑网络（Rainbow + IQN）用于任务价值估计；多头卷积编码器 + 逆动力学头 + Laplacian 头；Laplacian 目标（ALLO）与多步逆动力学损失的联合优化；噪声网络 + 量化分布式 RL；经验回放与优先采样。

**📊 数据集**

实验数据集为 Atari 2600 的“challenging set”十款游戏（Beam Rider、Freeway、Montezuma's Revenge、Pitfall!、Pong、Private Eye、Skiing、Solaris、Surround、Venture），全部使用 sticky 动作与完整动作集，未访问生命信息。

**📈 对比分析**

与 Rainbow、IQN（单流无模型）以及并行模型基础的 DreamerV3 进行对比；在 200M 训练帧下，Wayfarer 在多数游戏中超越 Rainbow 与 IQN，且与 DreamerV3 接近；在挑战集上实现最高累计回报，展示了探索、信用分配和泛化的优势。

**⚠️ 局限性**

局限性包括：1) 对 Laplacian 表征的依赖可能在极端非可控环境中受限；2) 选项学习对计算资源有一定开销；3) 在极难探索或极长时间尺度的任务中仍难以获得正回报；4) 需要进一步验证跨域迁移与长期稳定性。

---

## 677. Bridging Frontier Reasoning and Robot Execution: From Autonomous Demonstration Generation to Dense Language Supervision

**arXiv ID:** 2610.03615 | [PDF](https://arxiv.org/pdf/2610.03615v1)

**作者:** Bosung Kim `[一作]` (University of California San Diego), Prithviraj Ammanabrolu `[通讯]` (University of California San Diego)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并验证了两个桥接方案：利用前沿模型自主生成演示来训练低延迟本地控制器，并通过三层级多角度语言监督扩展本地策略的指令理解。

**💡 创新点**

创新点在于结合纠正性演示段提升生成可靠性，并将任务划分为原子、原始、复合级别，配合多样化表述增强指令多样性，从而将前沿推理与实时控制无缝对接。

**🔧 技术方法**

使用前沿大语言模型（Fable 5、GPT‑6 Astra）进行自控演示生成、OpenPI基础策略微调、密集语言标注、多层级指令生成以及多模型（Qwen3.6‑27B）进度监测。

**📊 数据集**

在四个真实机器人任务上收集混合人类+模型生成演示，并在RoboCasa 365、BEHAVIOR‑1K模拟长时程任务以及自制交叉字母拼图任务中评估。

**📈 对比分析**

通过对比全局前沿控制、仅本地策略、带纠正段的前沿控制和前沿+本地控制挂钩四种配置，在任务成功率、执行时间、生成成本等指标上进行实验，结果显示挂钩策略将任务成功率提升约65%且平均执行时间缩短约40‑60%，多级多角度监督进一步提高长任务成功率至约52%。

**⚠️ 局限性**

局限包括对高精度接触操作的依赖仍不稳定，模型生成成本在早期仍高，且本地策略对指令的依赖性导致在极端指令变体或硬件错误时性能下降。

---

## 678. ManifoldSplat: Language-Guided Semantic Shape Editing of 3D Gaussian Head Avatars

**arXiv ID:** 2610.03599 | [PDF](https://arxiv.org/pdf/2610.03599v1)

**作者:** Antonio Canela `[一作]` (Universitat Politècnica de Catalunya), Jordi Sànchez-Riera `[通讯]` (Institut de Robòtica i Informàtica Industrial, CSIC-UPC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出端到端的语言驱动形状编辑框架，可对单目视频重建的可动画3D高斯分布头部化身进行局部几何编辑。

**💡 创新点**

创新点在于在FLAME形状子空间内使用条件变分自编码器实现区域解耦的文本指导形状增量，并引入目标渲染优化（表面一致性、平滑色彩、变形映射）解决高斯漂移与细节失真。

**🔧 技术方法**

使用FLAME参数化、3D高斯分布渲染、Flan‑T5文本编码、条件VAE、图像配准、光照与表面正则化等技术。

**📊 数据集**

构建了约500k条带有区域位移原型、文本变体的合成数据集（涵盖六个面部区域），并使用NeRSemble等公开数据进行评测。

**📈 对比分析**

与GaussianAvatar‑Editor和InstructPix2Pix对比，在CLIP‑D、ArcFace、TV2D等指标上取得最高得分；编辑速度约30 s，渲染速率>800 fps，显著快于对比方法。

**⚠️ 局限性**

局限性包括只能编辑FLAME表面几何，无法处理头发或外部区域；依赖有限的原型库，无法覆盖完全新颖的结构变化；高斯与FLAME之间的拓扑差异仍可能导致极端变形下的细节失真。

---

## 679. DEPICT: Scoring Text-to-Image Alignment by Answer Agreement

**arXiv ID:** 2610.03617 | [PDF](https://arxiv.org/pdf/2610.03617v1)

**作者:** Vasco Ramos `[一作]` (Sword Health), Pedro Henrique Martins `[通讯]` (Sword Health)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无需训练的图像-文本对齐评估指标，利用图像与文本答案的一致性并融合整体与分解评分。

**💡 创新点**

创新在于用期望一致性替代固定是/否标签，消除负面提示错误，并将分解与整体评分合并。

**🔧 技术方法**

采用视觉语言模型（VLM）生成是/否问题、双通道推理、软概率评分、承诺加权与加权融合。

**📊 数据集**

在GenAI-Bench、TIFA160、RichHF-18K、Winoground、NegBench等公开基准上进行评测。

**📈 对比分析**

与细调评估器和其他训练免费指标对比，表现至少匹配或超过细调模型，尤其在负面提示准确率从19%提升至88%。

**⚠️ 局限性**

局限在于短提示的组合依赖性不足，且图像与文本通道共享模型导致同一偏差可能伪造一致性。

---

## 680. HyperBrowseComp: A Multilingual and Multimodal Stress Test for Web-Browsing Agents

**arXiv ID:** 2610.03574 | [PDF](https://arxiv.org/pdf/2610.03574v1)

**作者:** Alham Fikri Aji `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Irina Nikishina `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个包含423道多语言、多模态浏览问题的基准，用于评估AI代理在开放网络上寻找并验证稀有证据的能力。

**💡 创新点**

创新点在于将多语言原生编写的问题与多模态证据（视频、图像、PDF、地图等）相结合，创建了比以往更具挑战性、跨文化的浏览测试环境。

**🔧 技术方法**

使用了内置搜索、Exa统一接口以及OWL多代理框架等检索技术，并通过ReAct和多代理角色分工的方式与模型交互。

**📊 数据集**

使用的主要数据集是自研的多语言多模态问题集，涵盖13种语言和8种模态，所有问题均来自公开可验证的真实世界信息。

**📈 对比分析**

对五个前沿LLM在三种检索环境下进行比较，最佳模型（Gemini 3.7 Flash内置搜索）准确率仅达31.68%，整体上仍有约57.68%的问题未被任何模型正确回答。

**⚠️ 局限性**

局限性包括评测模型数量有限、检索工具与模型的耦合影响结果、基准不代表日常搜索分布以及不同语言可检索信息的差异导致的性能不均衡。

---

## 681. Threat-Preserving Representation Sensitivity in Agent-Security Benchmarks

**arXiv ID:** 2610.03585 | [PDF](https://arxiv.org/pdf/2610.03585v1)

**作者:** Neeraj Karamchandani `[一作]` (Pennsylvania State University), Dinghao Wu `[通讯]` (Pennsylvania State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在agent安全基准中，攻击成功率（ASR）是否会因威胁信息的表示方式而产生显著变化，并引入了威胁保留表示敏感度（TPRS）指标来量化这一效应。

**💡 创新点**

创新点在于提出TPRS框架，系统评估不同基准（ASB、MCPTox、AgentDojo）在对工具名称、描述等可见表示做“威胁保留”变换时ASR的波动，揭示基准评分并非与攻击本质完全独立。

**🔧 技术方法**

主要技术包括对工具名称、描述进行三类变换（正字法、语义中性、威胁提示）、使用克隆/哈希保证基准不变、对模型（GPT‑5‑mini、Claude Haiku 4.5、GPT‑4o‑mini）多次运行、采用cluster‑bootstrap CI估计ASR变化。

**📊 数据集**

使用的数据集为三大公开agent安全基准：Agent Security Bench（ASB）、MCPTox、AgentDojo，共计28,904次agent运行。

**📈 对比分析**

比较方法：对同一基准、同一模型，在保持攻击目标与评估准则不变的前提下，分别测得变换前后ASR的差值并给出95% CI。结果显示：ASB中去除威胁词的工具名可提升ASR约12–14个百分点；MCPTox中加入威胁词则降低ASR约4–11个百分点；AgentDojo中ASR变动很小，仅0.5个百分点，但对正常功能的实用度产生约5个百分点的负面影响。

**⚠️ 局限性**

局限性包括：仅评估了工具名称与描述的变换，未涵盖其他可能的表示维度；部分语义中性变换未通过功能等价验证；实验覆盖模型与基准有限，未验证对更广泛模型和防御方法的普适性；TPRS仅衡量ASR的敏感度，未完全归因于特定表示属性。

---

## 682. Recursive Harness Self-Improvement for Frontier Reasoning Data Synthesis

**arXiv ID:** 2610.03548 | [PDF](https://arxiv.org/pdf/2610.03548v1)

**作者:** Wenlong Zhang `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一套任务与生成 harness 的协同进化框架，递归改进推理任务与其生成流程。

**💡 创新点**

创新点在于同时采用在线自我改进与批次后自我改进两阶段 harness 进化，并通过候选评估与回滚机制保证难度提升同时控制成本。

**🔧 技术方法**

使用的技术包括大型语言模型 DeepSeek-V4-Pro、Meta Agent、Verifier、在线技能更新、候选评估门控，以及 GRPO 与 SFT 训练等。

**📊 数据集**

使用的数据集涵盖数学、编码与科学三类种子任务，构建 10K 级数学示例和 10K 级编码/科学示例，评估基准包括 MATH、GPQA、FrontierMath、SciCode、FrontierScience 等。

**📈 对比分析**

通过对比固定 harness、仅在线、仅后置以及完整进化四种配置，固定 harness 的准确率为 73.5%，完整进化后降至 50%（难度提升 45%），在 27B 学生上 10K 示例可达 62.5% APEX 绩效，SFT/GRPO 训练均显著提升。

**⚠️ 局限性**

局限性：未评估与独立 solver 的转移效果、总生成成本、候选有效率、训练 token 对齐以及对持久 seed 的泛化能力。

---

## 683. Knowledge or Calculator? Decomposing the Skill Premium in Verifiable Financial Agent Workflows

**arXiv ID:** 2610.03564 | [PDF](https://arxiv.org/pdf/2610.03564v1)

**作者:** Jermyn Zhen Yong Bek `[一作]` (Independent Researcher), Zhongtian Sun `[通讯]` (University of Kent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了FinSkillBench，一套可验证的金融AI代理工作流评测套件，涵盖投资组合构建、风险管理和基本面分析三大领域，共2,603个时点任务；通过对可重现隐藏真值的定量验证，评估LLM在实际工作流中的技能表现；

**💡 创新点**

创新点包括：①设计了可控的技能资源干预（无技能、策划技能、自动生成技能）并分解为文档与可执行工具的贡献；②引入了多种评分变体和交叉主机验证，验证结果鲁棒性；③揭示了“沉默能力”——高分子子任务无法在整体流程中组合成功的现象；④提供了完整的可复现代码、评测器和数据。

**🔧 技术方法**

技术手段：大语言模型（如GPT-4.1、Gemini-2.5-pro等）结合ReAct式函数调用与工具调用；构建技能文档与Python脚本包装；使用确定性验证器与结构化评分；在两个独立主机上跑实验；对比多模型、三种资源条件、十二子任务。

**📊 数据集**

数据集：基于30支美国股票的市场数据（价格、因子、宏观）、EDGAR XBRL报告、合成投资组合和风险场景；所有数据点均为点时态，确保可重现性。

**📈 对比分析**

比较方法：对9种模型在3种资源条件下执行17,820个episode，计算平均得分；对比无技能与策划技能、自动生成技能；使用Bootstrap置信区间评估差异。性能：策划技能平均提升+16.2分（0.366→0.528），自动生成仅+0.5分；文档+5.6分，工具+19.5分，组合非加性。

**⚠️ 局限性**

局限性：①仅采样单条轨迹，未估计模型内部随机性；②覆盖面仅限30支股票、三大领域，未包含其他资产或国际市场；③任务与技能资源高度耦合，结果受技能设计影响；④主机设计差异影响效果（例如Hermes主机的工具访问方式）；⑤组合工作流案例为单一模型单一约束，无法评估普遍性。

---

## 684. From Benchmarks to Production: A Text-to-SQL System for Complex Financial Data

**arXiv ID:** 2610.03524 | [PDF](https://arxiv.org/pdf/2610.03524v1)

**作者:** Arijit Sehanobish `[一作]` (Kensho Technologies), Kristen Howell `[通讯]` (Kensho Technologies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一种针对金融数据库的领域专用自然语言到 SQL 翻译系统 FLINT。

**💡 创新点**

结合查询检索、查找代理、分阶段模式化和执行反思的管道，以解决深度规范化与离散整数键的价值归一问题。

**🔧 技术方法**

使用多代理架构、LLM（Claude 4.5 Opus）、嵌入检索、实体链接服务、schema linking 及执行反思等技术。

**📊 数据集**

在两组生产金融数据库（Financials 与 Transactions，共 359 条问题）以及额外四个科学/医学基准上进行评测。

**📈 对比分析**

与七个学术 T2S 系统（CHESS、ReFoRCE 等）在相同数据库上进行三阶段评估，FLINT 的准确率为 68.8%/65.2%，显著高于 50% 的基线，并在 10–20 s 内完成。

**⚠️ 局限性**

需专家手工编写领域规则和查询库，缺乏自动化扩展；检索组件在无历史查询时效果下降；LLM 判断可能引入噪声。

---

## 685. Autonomous Robotic Navigation for Endovascular Brain-Computer Interface Access

**arXiv ID:** 2610.03537 | [PDF](https://arxiv.org/pdf/2610.03537v1)

**作者:** Harry Robertshaw `[一作]` (Kings College London), Sam E. John `[通讯]` (University of Melbourne)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

在计算机仿真与3D打印血管模型上，首次实现了针对脑血管内植入式脑机接口的自主机器人导向导航，并配备了在线失败预测器。

**💡 创新点**

创新点包括：①首次针对脑静脉系统的自主导航演示；②使用几何增强实现跨解剖结构的迁移学习；③设计任务特定的递归失败预测模型；④在全仿真训练后直接转移至物理试验并评估模拟到现实的性能差距。

**🔧 技术方法**

技术方法包括：Soft Actor‑Critic（SAC）强化学习配合LSTM策略；SOFA+stEVE仿真框架；基于荧光图像的导向点跟踪；机械机器人手臂实现导管与导丝的平移/旋转；GRU递归网络用于在线失败预测。

**📊 数据集**

数据集：两份公开的MRI血管STL模型（CNS Venography 3D SR Nevit Dilmen 与 3DPX‑003440 CNS Venography Nevit Dilmen），并在训练阶段对其进行几何增强。

**📈 对比分析**

对比方法：在训练解剖结构和未见的 hold‑out 解剖中分别进行 250 次仿真试验与 5 次物理试验。仿真成功率为训练：Task A 85.6%，Task B 98.4%；hold‑out：Task A 42.0%，Task B 91.6%。物理实验总体成功率 70%（Task A 40% hold‑out，Task B 80% hold‑out）。失败预测器在仿真中检测率 99.3–100%，误报率 0.8–6.7%；在物理实验中误报率升高，显示模拟到现实的校准不足。

**⚠️ 局限性**

局限性：仅使用两份解剖模型，缺乏真实血流/脉冲与柔性材料；仿真与物理实验之间的摩擦与动力学差异导致失败模式；物理实验样本量小且无人工导航基准；未评估安全指标（如导管接触力、血管损伤）或真实 BCI 设备植入；失败预测器仅在同一控制器生成的数据上训练，缺乏对未知策略或硬件故障的鲁棒性。

---

## 686. Reasoning Models Are Accurate but Unsound on Identification

**arXiv ID:** 2610.03519 | [PDF](https://arxiv.org/pdf/2610.03519v1)

**作者:** Arman Behnam `[一作]` (Illinois Institute of Technology), Binghui Wang `[通讯]` (Illinois Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于正式因果识别算法的评估框架，可为因果推理模型提供无人工干预的真值标签和可验证的评分。

**💡 创新点**

创新点在于：①使用完备的 id 算法自动生成真值标签；②设计结构泄漏诊断与修复机制，避免模型仅凭图结构获胜；③引入数值验证器，彻底消除字符串匹配误判，保证评估可信。

**🔧 技术方法**

核心技术包括：因果图的 ID（id）算法、结构泄漏判定与修复、边剪切（repair）任务、随机生成结构因果模型进行数值验证、LLM 与交互解析。

**📊 数据集**

数据集：随机生成的 ADMG（4–50 个节点）与五个公开因果图（asia、child、insurance、alarm、hepar2）派生的实例，主池 600 个（可辨识/不可辨识各占一半）、规模网格 600 个、边分类 200 个。

**📈 对比分析**

比较方法：将模型的可辨识/不可辨识判定与 id 标签对比，计算准确率、答复率和错误声称率。结果显示前沿 LLM 在识别率上表现良好，但准确率并不能揭示真实可证性；GPT‑5.5 的错误声称率最低，表明其在可证性方面更安全。

**⚠️ 局限性**

局限性：评估仅适用于最多 9 个变量的可枚举结构因果模型；数值验证依赖随机抽样，可能遗漏极端参数情况；LLM 的回答不稳定且可能被截断，导致结果波动；对大图不可辨识查询的修复机制仍需进一步改进。

---

## 687. World Action Learning via Interaction-Centric Spectral Latent Guidance

**arXiv ID:** 2610.03607 | [PDF](https://arxiv.org/pdf/2610.03607v1)

**作者:** Zhiming Liu `[一作]`, Song Guo `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过从大规模第一人称视频中学习交互中心的潜在动作表示，并将其转移到机器人学习任务中，提出了一种名为WING的框架；

**💡 创新点**

创新点包括①引入交互中心潜在动作学习，抑制观察者运动的影响；②采用频域低频成分作为可转移的动作指导；③利用DCT分解潜在动作轨迹以提取低频信息；

**🔧 技术方法**

使用的技术包括潜在动作模型WING‑LAM与图像空间的观察者‑交互分解；离散余弦变换（DCT）进行频域分析；轻量级预测器估计低频指导；以及基于语言与视觉输入的世界动作模型π_θ；

**📊 数据集**

使用的数据集包括大规模第一人称人类视频（未明示具体名称），仿真基准LIBERO、RoboTwin 2.0、RoboCasa–GR1，以及真实世界的四个双手操控任务；

**📈 对比分析**

与π_0.5、LingBot‑VA等基线方法比较，WING在LIBERO上平均成功率99.2%、RoboTwin 2.0 93.8%、RoboCasa–GR1 57.7%，在真实世界任务中保持强性能，整体优于同类方法；

**⚠️ 局限性**

主要限制包括①依赖外部运动信号和图像空间分解，全球仿射逼近在大视差和3D运动下可能不够精确；②潜在动作预训练和频域指导增加了训练成本和流程复杂度。

---

## 688. Learning from Repaired Reasoning: Root-Cause-Guided On-Policy Distillation

**arXiv ID:** 2610.03515 | [PDF](https://arxiv.org/pdf/2610.03515v1)

**作者:** Chenglei Shen `[一作]` (Renmin University of China), Jun Xu `[通讯]` (Renmin University of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Root‑Cause‑Guided On‑Policy Distillation (RC‑OPD)，通过诊断学生的错误、构造局部修复并迭代验证，给错误段与有效前缀分别提供差异化的教师监督；

**💡 创新点**

创新点在于（1）将错误定位与根因修复相结合生成局部 hindsight；（2）通过迭代反事实验证保证修复可推进至正确答案；（3）区分错误段与有效前缀的监督，解决参考引导导致的推理不匹配与蒸馏陷阱；

**🔧 技术方法**

使用 Qwen3 学生与冻结同尺寸教师、DeepSeek‑V4‑Flash‑0731 诊断模型、On‑Policy Self‑Distillation、分阶段 KL 蒸馏、LoRA 微调与前缀加权技术；

**📊 数据集**

训练数据来自 OpenThoughts‑Math‑30K，评估使用 AIME 2024、AIME 2025、HMMT 2025 各 30 题；

**📈 对比分析**

与 Base、SFT、GRPO、OPSD、EOPD、DASH、PW‑OPSD、AVSD、ROSD 等方法对比，RC‑OPD 在 1.7B/4B/8B 三个模型规模和三大基准上平均准确率最高，提升约 5%–10% 相比 OPSD；

**⚠️ 局限性**

局限性：依赖预先提供的参考答案和诊断模型；迭代验证耗时且对大模型扩展性不明；当错误不足时仍需 fallback 参考蒸馏，可能降低自我修正效果；未充分验证在更广泛任务上的迁移能力。

---

## 689. Rubric-Based Optimization for Text-to-Music Generation

**arXiv ID:** 2610.03589 | [PDF](https://arxiv.org/pdf/2610.03589v1)

**作者:** Ping Wang `[一作]` (University of Washington), Noah A. Smith `[通讯]` (University of Washington)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了利用预训练音频-语言模型（ALM）按多维度 rubric 作为奖励，对 autoregressive 与 diffusion 文本到音乐生成器进行后训练。

**💡 创新点**

创新点在于无需人工偏好或专门奖励模型，直接用 ALM rubric 评分产生多维度奖励，并对不同优化器、评估器、提示方式进行系统比较。

**🔧 技术方法**

使用 DPO 与 DiffusionNFT 作为优化器，利用 MusicGen‑small、ACE‑Step v1 生成器，Qwen3‑Omni 等 ALM 评判器，以及音乐评测指标 CLAP、SongEval、Audiobox‑Aesthetics。

**📊 数据集**

数据集包括 MusicCaps（5,521 条文本提示）和 MTG‑Jamendo 的节奏、调式、乐器标签等属性提示。

**📈 对比分析**

实验表明，ALM rubric 在多评测器上能同步提升 CLAP、SongEval 与 Audiobox‑Aesthetics，DiffusionNFT 对 ACE‑Step 的提升显著；但在可精确测量的属性（tempo、key、instrumentation）时，直接优化指标优于 rubric。

**⚠️ 局限性**

局限包括仅使用开源 ALM 评判器、仅评估 MusicGen‑small 与 ACE‑Step、生成片段短暂，且 ALM 对某些属性（key 等）敏感度低，缺少对更长篇幅与更细粒度音乐特性的评估。

---

## 690. Writerslogic at the CLEF 2026 SimpleText Track: Multi-Candidate LLM Simplification and Stacked Complexity Spotting

**arXiv ID:** 2610.03567 | [PDF](https://arxiv.org/pdf/2610.03567v1)

**作者:** David L. Condrey `[一作]` `[通讯]` (WritersLogic Inc), David L. Condrey (WritersLogic Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并提交了基于多候选生成与参考无关评分的文本简化系统，以及利用DeBERTa+特征工程的复杂度检测系统；

**💡 创新点**

创新点在于多温度多候选重选、SARI-代理评分、引入Cochrane PLS词表、构建序列特征捕获过度生成传播、堆叠学习与阈值优化；

**🔧 技术方法**

采用GPT-4o-mini/Claude Sonnet 4生成、LightGBM+LogReg堆叠、DeBERTa‑large微调、句子嵌入与NLI交叉编码；

**📊 数据集**

使用Cochrane系统综述文本（英语及多语种）以及350K标注的（源句、简化句）对；

**📈 对比分析**

在CodaBench评测中，句子级简化SARI 47.43、BLEU 14.21领跑；复杂度检测二分类macro‑F1 0.8081/0.8085、复分类准确率0.804，均位居第二；

**⚠️ 局限性**

限制在于需依赖商业API导致成本与复现性受限，本地模型性能下降；序列特征需要完整文档，无法流式推理；NLI与嵌入推理耗时与GPU资源需求较高。

---

## 691. Beyond Trained Models: Compiling GNNs for a Sound Explainer Benchmark

**arXiv ID:** 2610.03526 | [PDF](https://arxiv.org/pdf/2610.03526v1)

**作者:** Steve Azzolin `[一作]` (University of Trento), Andrea Passerini `[通讯]` (University of Trento)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种将分级模态逻辑公式编译为图神经网络（GNN）权重的编译器GRAcC，并基于此构建可控的GNN模型；利用主子公式的prime implicant集合定义精确的“真实解释”，并设计算法从逻辑公式中提取该解释；在此基础上构建BenchGRAcC基准，对11种主流GNN解释器进行细粒度评估；

**💡 创新点**

创新点在于①首次实现逻辑公式到GNN的可证明等价编译；②提出基于prime implicant的统一真实解释定义并给出高效提取算法；③基于可控GNN构建了一个严谨的解释器评估基准，揭示了现有解释方法的多项缺陷；

**🔧 技术方法**

使用分级模态逻辑、消息传递网络架构、减号ReLU、Binary Decision Diagram（BDD）知识编译、逻辑公式到GNN权重的映射算法、以及多种主流解释器（梯度、扰动、代理、分解等）进行实验；

**📊 数据集**

在人工构造的图/节点分类任务中生成图数据集，任务包括链式红蓝节点、黑节点及其邻居计数、颜色计数等，以验证编译模型与真实解释的一致性；

**📈 对比分析**

通过计算预测解释与真实解释的精度、召回率和F1值，对11种解释器进行比较，结果表明只有GraphSVX、GNNExplainer、PGExplainer在所有任务中保持高精度/召回；其余方法表现高度依赖任务、实现细节，且对间接影响不敏感；

**⚠️ 局限性**

局限性包括：编译GNN仅支持二元特征与无噪声聚合，难以模拟真实训练模型的复杂性；基准仅覆盖二分类与离散节点特征，未考虑多类别、连续特征及更复杂网络结构；

---

## 692. DR-IPC: Disturbance-Resilient Integrated Planning and Control for LiDAR-Based Quadrotor Navigation

**arXiv ID:** 2610.03530 | [PDF](https://arxiv.org/pdf/2610.03530v1)

**作者:** Peng Liu `[一作]` (Southeast University), Yunda Yan `[通讯]` (University College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研发并实现了DR-IPC算法，使四旋翼在受风、悬挂载荷和动态障碍等扰动环境中通过LiDAR地图完成多目标导航。

**💡 创新点**

创新点包括：将非线性EKF与扰动观测器（NDO）耦合实现实时噪声抑制和扰动重构；将软安全通道（SFC）惩罚嵌入NMPC中，兼顾碰撞规避与扰动补偿；将路径规划、状态估计与控制统一到一个预测后端，消除传统层级架构的耦合瓶颈。

**🔧 技术方法**

技术栈：LiDAR-惯性融合（FAST‑LIO2）、A*局部规划、SFC构造、非线性EKF+NDO观测器、非线性模型预测控制（NMPC，ACADO+qpOASES）、软约束优化、Gazebo/MARSIM仿真、PX4飞控与Ubuntu+ROS1运行平台。

**📊 数据集**

使用的实验与仿真数据：Gazebo与MARSIM森林环境中的点云地图，室内工厂、林地、夜间林地四个真实飞行场景；无公开数据集，全部为作者自建。

**📈 对比分析**

对比方法：与现有IPC（Hierarchical Planning & Control）算法比较；在Gazebo多目标测试中，完成任务率从1/10提升至9/10；高度RMSE从0.34m降至0.01m；在风+载荷、动态障碍等场景下，误差、碰撞次数和姿态波动显著下降。

**⚠️ 局限性**

局限性：观测噪声仍影响扰动重构精度；软SFC阈值需手工调参；在极端快速扰动或高动态障碍情形下预测误差可能增大；计算时间偶尔超过10 ms，但大部分周期满足100 Hz实时要求。

---

## 693. Writerslogic at PAN 2026: Process over Content for Robust Detection under Domain Shift

**arXiv ID:** 2610.03565 | [PDF](https://arxiv.org/pdf/2610.03565v1)

**作者:** David L. Condrey `[一作]` `[通讯]` (WritersLogic Inc), David L. Condrey (WritersLogic Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在PAN@CLEF 2026三项任务中，作者提出并验证了一套特征鲁棒性框架，基于“支持重叠”而非训练集效应大小来判断特征在分布偏移下的有效性，并构建了基于该框架的系统；在Reasoning Trajectory Detection任务中实现了源检测与安全分类，Voight‑Kampff任务中实现了跨体裁的AI文本检测，MAWSA任务中提出了作者风格切换检测方案（未正式提交）。

**💡 创新点**

创新点在于提出了“支持重叠”特征鲁棒性框架，形成了anchored、portable、invariant三类特征的直观分类，并揭示了“验证幻觉”和“粒度陷阱”两种常见的泛化误区；该框架指导特征选择、模型组合与阈值校准，显著提升跨域任务性能。

**🔧 技术方法**

技术上结合了手工设计的词汇指纹（hapax比率、Yule's K、Heaps指数）、压缩率、字符n-gram统计，使用LightGBM、DeBERTa‑v2、SVM等基学习器进行堆叠；采用LLM（Claude Opus、Sonnet、Llama、Qwen）同意投票、查询‑拒绝拆分与等价融合；通过等阶回归、Platt缩放实现概率校准。

**📊 数据集**

使用的数据集主要来自PAN@CLEF 2026的三项任务：Reasoning Trajectory Detection（训练为数学文本，测试涵盖未见领域）、Voight‑Kampff（多体裁人机文本混合）、MAWSA（多作者风格切换）。

**📈 对比分析**

与任务官方榜单及基线相比，源检测系统以0.85宏F1夺得第一名；安全分类系统以0.66宏F1位居第三；Voight‑Kampff系统在2026测试集上实现0.891 ROC‑AUC，跨体裁性能保持稳定，并在2025回归集上达0.979。

**⚠️ 局限性**

局限性包括：框架与实验均由单位研究者完成，验证过程为事后回顾；系统高度依赖商业LLM API，导致成本与可复现性受限；特征分类粗粒度，缺乏细粒度评估；未能在MAWSA任务上完成官方评测，相关预测仅为假设；最终模型组合机制（如LLM一致性贡献）尚不完全透明。

---

## 694. Feedforward Novel View Synthesis for Heterogeneous Cameras

**arXiv ID:** 2610.03522 | [PDF](https://arxiv.org/pdf/2610.03522v1)

**作者:** Meng Wei `[一作]` (Monash University), Jianfei Cai `[通讯]` (Monash University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种能够在不同投影模型（透镜、鱼眼、全景）间统一工作的前向新视角合成方法；

**💡 创新点**

核心创新是引入局部光束图（local raymaps）与投影感知的二维 RoPE，使得每个视觉令牌能够明确描述其局部光束分布，并在不同相机投影下保持相对位置一致；

**🔧 技术方法**

使用Transformer架构（decoder-only LVSM），结合相机位置编码（UCPE）、局部光束图以及基于光束角度的投影感知RoPE；

**📊 数据集**

在ScanNet++ v2数据集上进行实验，构造了鱼眼、未畸变和全景图像三种相机类型；

**📈 对比分析**

与基准方法（Plücker Raymaps、CamRay、PRoPE、UCPE）对比，混合相机评估和零射全景目标的零样本测试中取得更高的PSNR、SSIM和LPIPS分数，证明了方法的优越性；

**⚠️ 局限性**

受限于需要精确标定的多相机数据集，数据收集难度大，且对不同相机的普适性仍有待进一步验证。

---

## 695. Beyond Idealized Orbits: Realistic Network Topology and Spatio-Temporal Traffic Emulation for LEO Satellite Constellations

**arXiv ID:** 2610.03623 | [PDF](https://arxiv.org/pdf/2610.03623v1)

**作者:** Suryansh Aryan `[一作]`, Samantha Parry Kenyon `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文介绍了 Elsevier 官方 LaTeX 文档类 elsarticle.cls，阐述了其结构、功能、安装方式、使用选项以及前置标记、浮动体、定理环境、列表、交叉引用、数学公式排版等方面的实现细节。

**💡 创新点**

创新点在于：① 完全重写的类，基于标准 article.cls，避免了旧版 elsart.cls 与其他宏包冲突；② 默认提供 preprint 与多种期刊最终版（1p、3p、5p）格式；③ 与 natbib、hyperref 等常用宏包无缝集成；④ 兼容定理、列表、注释等高级排版需求；⑤ 支持双盲审稿、浮动体末端放置等实用选项。

**🔧 技术方法**

采用的技术主要是 LaTeX 宏包：natbib、geometry、graphicx、txfonts、hyperref、endfloat 等；类内部使用 
ewtheorem、
ewlist 等自定义环境；通过选项机制实现不同期刊风格；利用交叉引用、超链接等功能提升可读性。

**📊 数据集**

该工作不涉及数据集，主要是软件/宏包实现与使用说明。

**📈 对比分析**

本文未给出实验或性能评测；若要验证可用性，作者可将同一稿件分别用 preprint、1p、3p、5p 选项编译，检查排版是否符合期刊要求；此外，可通过比较旧版 elsart.cls 与新版 elsarticle.cls 的编译兼容性来评估改进效果。

**⚠️ 局限性**

限制包括：① 仍需手动在单列预印本与双列最终版间调整长公式排版；② 对于极端自定义宏包或特殊排版需求，可能与类内部定义产生冲突；③ 只适用于 LaTeX，不能直接在其他排版系统使用。

---

## 696. HazardWeaver: Scientific Route Selection for Hazard Analysis Agents

**arXiv ID:** 2610.03591 | [PDF](https://arxiv.org/pdf/2610.03591v1)

**作者:** Wangshu Zhu `[一作]` (Florida State University), Yushun Dong `[通讯]` (Florida State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于大语言模型的灾害科学路线选择与执行框架 HazardWeaver，并通过可解释的知识编译与可执行能力图实现状态依赖的科学路线决策。

**💡 创新点**

核心创新在于将科学适用条件从文献中提取为可检验的证据链接规则，并结合类型化的能力图动态更新可执行路线集合，实现多路径、多危害情境下的自适应决策与复原。

**🔧 技术方法**

使用大语言模型（如 Llama、Mixtral 等）进行决策、检索与推理，配合 Hazard Knowledge Compiler、Hazard Capability Graph 及 Agent 控制循环。

**📊 数据集**

构建了 141 个封闭实例的 Hazard Weaver Benchmark，涵盖七个单危害领域和四个多危害交互，来源于 PWFDF、USGS Ground Failure、SFINCS、NOAA 等公开数据集与操作产品。

**📈 对比分析**

与 10 个改造后的基线（AIDE、ReAct 等）对比，HazardWeaver 在整体 Decision‑Constrained Accuracy 上达到 89.4%，比最强基线提高约 48 个百分点，尤其在多路线任务上表现突出。

**⚠️ 局限性**

局限性包括对大型模型和丰富知识库的依赖，缺乏对实时动态事件的即时更新支持，以及对复杂跨域多步骤推理的可扩展性仍待验证。

---

## 697. Normal-Form Correlation in Markov Games

**arXiv ID:** 2610.03621 | [PDF](https://arxiv.org/pdf/2610.03621v1)

**作者:** Ioannis Anagnostides `[一作]` (Carnegie Mellon University), Brian Hu Zhang `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种高效算法，用于在有限时域的马尔可夫博弈中计算正常形式相关均衡（NFCE），并且适用于固定数量的玩家。

**💡 创新点**

创新点在于首次提出了在马尔可夫博弈中计算NFCE的高效算法，且该算法在1/ϵ和游戏描述的多项式时间内运行，超越了现有的相关均衡计算方法。

**🔧 技术方法**

使用了反向归纳法和常期望相关均衡的概念，结合线性规划和适当的离散化技术。

**📊 数据集**

未具体提及使用的数据集，但讨论了在有限时域马尔可夫博弈中的状态、行动和玩家数量。

**📈 对比分析**

与现有方法相比，本文的算法在计算NFCE时表现出更好的时间复杂度，尤其是在玩家数量固定的情况下，提供了完全多项式时间近似方案（FPTAS）。

**⚠️ 局限性**

限制在于算法在玩家数量增加时的扩展性较差，且在状态独立推荐的情况下，计算NFCE的复杂性仍然是NP难的。

---

## 698. FALCON: A Model and Dataset Agnostic Framework for Synthetic Data Generation for NL2SQL Pairs

**arXiv ID:** 2610.03625 | [PDF](https://arxiv.org/pdf/2610.03625v1)

**作者:** Darian Lee `[一作]` (University of California, Santa Cruz), Yuanming Shi `[通讯]` (Adobe)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 FALCON 框架，用低成本的开源 LLM 生成真实、含歧义的自然语言到 SQL 的数据集，覆盖现有基准难度更高的查询。

**💡 创新点**

创新点包括：① 使用保留词种子与角色设定种子共同驱动 SQL 生成，显著提升 SQL 复杂度与 NL 多样性；② 引入歧义检测与多解释生成，真正处理歧义而非简单过滤；③ 结合对齐模型、奖励模型与 prompt-based LLM 判断，实现高质量、高对齐的过滤；④ 全流程低成本、模型与数据库无关，适用于内部私有数据。

**🔧 技术方法**

技术手段：SQL-then-NL 的 LLM chain‑of‑thought 生成；保留词与 persona 种子提示；DistilBERT‑fine‑tuned 进行歧义打分；对齐预测模型与奖励模型用于过滤；执行验证与多阶段 prompt‑filter；多模态评估（EM、FS、Soft‑EX）。

**📊 数据集**

数据来源：以 Spider 与 WikiSQL 作为种子表进行生成；使用 BIRD 的保留词分布作参考；最终评估基于自生成的 FALCON 数据集、Spider、BIRD、WikiSQL 等公开基准；人类评估采用 4 名标注者的评估表。

**📈 对比分析**

对比方法：在 5,500 条 synthetic 与 5,500 条 Spider 训练集上微调 Qwen 2.5‑3B；对 Spider、FALCON、混合数据集进行 EM、FS、Soft‑EX 比较；结果显示 FALCON‑训练模型在难度更高的四分位数上显著优于 Spider‑训练模型；混合训练能在简单查询上保持 Spider 性能，复杂查询仍优于 Spider；人类评估分数均 >0.92，SQL 复杂度与 NL 丰富度均超过 BIRD；成本相对 GPT‑5.5 低 16 倍。

**⚠️ 局限性**

局限性：① 重点覆盖复杂查询，简单查询欠缺，需 5–10% 真实数据补充；② 歧义检测高特异性但召回率低，仅捕获部分真正歧义；③ 目前仅支持 SQLite，其他 SQL 方言需手工扩展；④ 下游评估仅使用小型 LLM，未验证大模型；⑤ 生成数据受种子数据库领域限制，可能无法覆盖所有生产数据库复杂度。

---

## 699. Constant-Rate Certified Deletion

**arXiv ID:** 2610.03590 | [PDF](https://arxiv.org/pdf/2610.03590v1)

**作者:** Kai-Min Chung `[一作]` (Academia Sinica), Shota Yamada `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本论文提出一种基于子空间余子空间的量子态构造，并利用该构造实现了常数速率的可认证删除（Certified Deletion）协议，进一步将其与已有的“全不泄露型”密码原语结合，得到常数速率的可认证删除扩展；

**💡 创新点**

创新点在于首次构造可在有限资源下（即常数速率）实现可认证删除的量子态，并给出通用的提升框架，使得多种现有密码原语可直接升级为支持可认证删除；

**🔧 技术方法**

主要技术包括量子子空间余子空间的线性代数构造、MDS 码的错误纠正性质、量子提取器（extraction lemma）、Gentle 量子测量技术以及子空间隐藏（subspace-hiding）安全模型；

**📊 数据集**

该工作为理论论文，不涉及实际数据集；

**📈 对比分析**

由于是理论构造，性能评估以复杂度与速率等量化指标为主：证明了在安全参数 λ、消息长度 k 与维度 n 满足 4k+Θ(λ)≤n 的条件下，可实现常数 1/3 速率的可认证删除，并给出相应的统计距离和安全上界；

**⚠️ 局限性**

局限性包括：实现仍需依赖理想的量子操作和高质量的 MDS 码；在实际量子硬件上实现时需要对错误率、退相干等问题进行进一步研究；此外，证明主要针对单个经典消息的加密，扩展到更复杂协议时可能需要额外的技术细节。

---

## 700. AVL-JEPA: Preventing Causal Dynamics Information Collapse In Joint Embedding Predictive Architecture World Models

**arXiv ID:** 2610.03587 | [PDF](https://arxiv.org/pdf/2610.03587v1)

**作者:** Yikang Qiao `[一作]` (Central South University), Duan Huang `[通讯]` (Central South University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了在Joint Embedding Predictive Architecture（JEPA）世界模型中因果动力学信息崩塌的问题，并提出了一种新的AVL方法来防止这一崩塌；

**💡 创新点**

创新点在于通过动作锚定的行动路径和视觉不变性的对齐路径相结合，既保持动作相关的动态信息，又通过视觉扰动对齐迫使模型充分利用因果动力学；

**🔧 技术方法**

所采用的技术包括动作锚定（action‑grounded pathway）与视觉不变性（vision‑invariance pathway）对齐、SIGReg正则化、joint‑embedding 预测、动作损失以及多种视觉扰动训练；

**📊 数据集**

实验使用了四个机器人控制任务数据集：TwoRoom、PushT、OGBench Cube 和 Reacher；

**📈 对比分析**

与PLDM、SD‑JEPA等基线模型比较时，AVL在干净环境下的闭环规划成功率最高，在 Gaussian、brightness、saturation 等视觉扰动下也表现出显著的鲁棒性提升；

**⚠️ 局限性**

局限性包括仅在仿真环境中验证、仍依赖SIGReg正则化、未在真实机器人上进行实验，未来计划进一步剔除正则化依赖并开展真实机器人测试。

---

## 701. Faster Sublinear Maximal Independent Set Size

**arXiv ID:** 2610.03588 | [PDF](https://arxiv.org/pdf/2610.03588v1)

**作者:** Peter Kiss `[一作]` (University of Vienna), Arash Kooroshnezhad `[通讯]` (University of Warwick)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在邻接矩阵查询模型下设计了一种亚线性时间算法，用于在随机顺序贪心算法生成的最大独立集上给出 (1+ϵ) 近似的大小估计。

**💡 创新点**

核心创新在于不把先前的 MIS 成员资格判定算法当作黑箱，而是将其内部递归过程与部分已构造的基线独立集 S_b 结合，引入 U = V\(S_b∪N(S_b)) 的视角，从而在期望 O(n) 查询时间内判断随机选取顶点是否属于最终独立集，突破了之前的 Õ(n^{1+1/2}) 速度壁垒，取得了 Õ(n^{1+1/3}/ϵ^2) 的复杂度。

**🔧 技术方法**

技术手段包括：① 先在前 O(n^{1+1/3}) 次查询中构造基线独立集 S_b；② 通过对 U 的特殊查询（U‑query）和对已知 S_b 的过滤，减少对邻接矩阵的实际访问；③ 采用负超几何分布与 Chernoff 边界对样本误差进行控制；④ 将 MIS 大小估计结果迁移到子线性度量 k‑center 和 Steiner forest 的近似算法中。

**📊 数据集**

该工作主要是理论性的，未使用具体数据集；所有结果均基于随机图模型的抽样与查询模型的假设。

**📈 对比分析**

与先前的 Õ(n^{1+1/2}) 估计器相比，新算法在大多数 n 上的时间复杂度显著下降，尤其在 n 较大且 ϵ 较小的情形下优势更为明显；实验上（若有实现）可见常数与 log 因子对实际性能的影响，但理论上已达到最优的 O(n^{1+1/3}) 下界。

**⚠️ 局限性**

局限性包括：仍属于亚线性复杂度，且实现中存在高阶多项式 log 因子；算法对 ϵ 依赖为 1/ϵ^2，可能在 ϵ 接近 0 时性能下降；在极稀疏或高度结构化图中，S_b 的构造可能不如预期高效；此外，结果主要针对邻接矩阵查询模型，对其他更灵活的访问模型的适应性尚未给出。

---

## 702. OpenMP Meta-Lowering: A Declarative Approach to Performance Portable Parallel Code Generation

**arXiv ID:** 2610.03571 | [PDF](https://arxiv.org/pdf/2610.03571v1)

**作者:** Luca Parigi `[一作]` (University of Bologna), Giuseppe Tagliavini `[通讯]` (University of Bologna)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于MLIR的OpenMP元低阶化框架，使用DSL描述不同运行时的低阶化规则，并通过三阶段管线（注释、提炼、计划应用）实现可编程、可扩展的OpenMP低阶化。

**💡 创新点**

把OpenMP低阶化抽象为可编程DSL，解耦编译器与运行时；提供三阶段MLIR管线；大幅减少低阶化代码量（相较Clang、GCC分别减少约32%和76%）。

**🔧 技术方法**

MLIR框架、CIR高阶dialect、OpenMP dialect、DSL规范、Clang/CIR与MLIR集成、三阶段低阶化策略。

**📊 数据集**

PolyBench/C‑OMP基准套件。

**📈 对比分析**

通过与Clang/LLVM默认libomp路径进行对比，在一般CPU和嵌入式多核（PULP）目标上评测；结果显示性能与最先进工具链持平，代码尺寸增幅低于0.7%。

**⚠️ 局限性**

仅支持OpenMP子集；需要手写DSL规则才能加入新运行时；对极复杂并行模式的支持有限；尚未覆盖所有可能的优化与平台。

---

## 703. Rethinking What to Cache in Few-Step Diffusion Transformers: Solver-Aware Target Selection

**arXiv ID:** 2610.03577 | [PDF](https://arxiv.org/pdf/2610.03577v1)

**作者:** Shuo Yang `[一作]`, Youqing Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 AutoTarget 方法，用于在已压缩的 Diffusion Transformer（DiT）中选择最优的缓存张量，从而加速采样。

**💡 创新点**

创新点在于基于精确校准的无训练缓存目标选择，考虑求解器转换，并分析 Euler 采样下等价缓存形式。

**🔧 技术方法**

使用无训练校准、误差评估与 VAE 解码、Euler/流求解器等技术；并在 PixArt‑LCM、FLUX.1‑schnell 与 HunyuanVideo 等模型上实现。

**📊 数据集**

在 PixArt‑LCM、FLUX.1‑schnell（distilled）和 HunyuanVideo（distilled 20‑step）等公开图像/视频生成数据集上进行实验。

**📈 对比分析**

与 TeaCache、DiCache、TaylorSeer、HiCache、DisCa 等缓存基线比较，AutoTarget 通过精细校准实现 1.3‑2.0× 的缓存加速，同时保持与未缓存模型相近的 PSNR/SSIM、ImageReward、CLIP 等指标，且缓存大小仅 0.5 MiB。

**⚠️ 局限性**

局限性包括需要针对每个模型/求解器/调度重新校准；若候选目标误差相近则会放弃选择；仅适用于冻结的已压缩模型；对超大张量缓存支持有限。

---

## 704. A Secure dToF LiDAR SoC with Dual-Domain Fingerprinting and Event-Driven AFE Circuit Achieving Sensor-Level Attack Resilience

**arXiv ID:** 2610.03562 | [PDF](https://arxiv.org/pdf/2610.03562v1)

**作者:** Risa Nonaka `[一作]` (Keio University), Kentaro Yoshioka `[通讯]` (Keio University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一款具备硬件级抗欺骗功能的直接飞行时间（dToF）LiDAR SoC，结合双域指纹识别、事件驱动AFE与时变激光驱动，实现对射频欺骗攻击的实时检测与拦截；

**💡 创新点**

创新点包括：①双域指纹（时域+幅度域）实现更高安全性，②事件驱动AFE仅在峰值附近激活ADC，功耗下降99%且仍保留1cm距离分辨率，③时变激光驱动实现10%–100%幅度调制，支持双域指纹；

**🔧 技术方法**

使用技术包括：双域指纹验证、事件驱动AFE、三点抛物线插值实现亚采样峰值估计、触发式8位SAR ADC、时变激光驱动器、65nm CMOS SoC集成以及离芯SPAD光探测；

**📊 数据集**

实验数据集为实测点云与距离数据：在室内120m范围内测距、对抗高频欺骗攻击的点云记录，并使用两次不同激光驱动电压扫描获得幅度指纹；

**📈 对比分析**

通过与现有离芯光探测器LiDAR的对比，显示该SoC在120m范围内可实现1cm分辨率，AFE功耗仅3.1mW/通道（比对比方法低60%），在欺骗攻击下点云保护率73%（无指纹为0%）；

**⚠️ 局限性**

局限性在于：双域指纹的实时在芯片上验证尚未实现，仍有27%测距失败，评估仅针对高频欺骗攻击，未考虑中继攻击；实验仅在室内离芯SPAD平台完成，缺乏户外与全芯片SPAD集成的验证。

---

## 705. Learning to Assess Heartbeat Observability for mmWave Heart-Rate Sensing

**arXiv ID:** 2610.03570 | [PDF](https://arxiv.org/pdf/2610.03570v1)

**作者:** Yuxuan Hu `[一作]` (Fudan University), Feng Xu `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

开发了一种基于毫米波雷达相位谱的可观测性评估模型HEAR，能够学习判断单个测量的心跳可读性并根据评估结果筛选可靠的心率估计。

**💡 创新点**

创新点在于：① 将心跳可观测性视为可测量的属性，并通过多散射体FMCW模拟生成标签；② 设计轻量级双任务Transformer，联合预测可观测性分数与心率；③ 通过零样本迁移实现对真实数据的即时应用；④ 将可观测性评分用于跨多种心率估计器的选择性预测。

**🔧 技术方法**

主要技术包括：多散射体FMCW雷达模拟、相位信号预处理、将相位幅度与相对呼吸基频编码为2‑D token、轻量级Transformer双任务网络、可观测性标签生成、零样本迁移、选择性预测与AURC评估。

**📊 数据集**

训练使用 5.6M 条模拟观测数据；评估使用公开毫米波心率数据集 Parralejo（60 GHz，110 受试者）和 VitalSense（120 GHz，24 受试者）。

**📈 对比分析**

与手工特征、MLP、1D‑CNN、SNR‑基准及多种心率估计器（FFT 峰、加权峰、相关、HPS、VMD）进行对比；HEAR 的可观测性分数在模拟和两实测数据上 AUC 均超过 0.86，且在 120 GHz 数据集上通过选择性筛选将 MAE 从 17.9 BPM 降至 1.6 BPM（覆盖率约 50%），与仅基于 SNR 的方法相比，AURC 更低，说明选择更有效。

**⚠️ 局限性**

局限性：① 可观测性标签仅基于峰值与参考心率的匹配，未覆盖心跳信号可能被其他估计方法恢复的情况；② 在高心率或运动后条件下可辨识度显著下降；③ 模拟假设简化（固定散射体幅度、预设胸壁运动权重、固定雷达链路），未考虑衣物、环境多径、硬件差异等实际变异；④ 选择性预测导致输出间隙，适用于间歇监测但对持续监测仍需进一步研究。

---

## 706. Get a GRIP, this will be a long TRIP: A Quantifiable Long-Range Framework for Verifying Over-squashing

**arXiv ID:** 2610.03556 | [PDF](https://arxiv.org/pdf/2610.03556v1)

**作者:** Ferran Hernandez Caralt `[一作]` (University of Cambridge), Pietro Liò `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `79276348-11e0-48e3-84bc-7ec231d0171c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个基于四个可验证公理的长程图任务框架，并通过该框架对现有长程基准进行审核与评估。

**💡 创新点**

创新点包括：①四个可检验的公理（可预测性、紧致性、严格k-范围、拓扑不变性）；②构造任务TRIP及其通用化版本，使任何图都能生成满足所有公理的长程任务；③针对TRIP任务提供闭式的每范围MLE下界，首次给出任意长程基准的先验误差下限。

**🔧 技术方法**

使用的技术主要有：概率分布理论（稳定分布、f-稳定）、图神经网络（消息传递、残差GCN）、统计学量化（互信息、熵、MLE）、实验评估（对比曲率、ECHO任务等）。

**📊 数据集**

使用的图数据集包括：Long Range Graph Benchmark (LRGB)、LRIM、GLoRa、ECHO 以及通过TRIP构造的任意图（如随机图、周期网格、DAG、分子图等）。

**📈 对比分析**

通过在TRIP任务和原始基准上训练相同的基线模型（GCN、GIN、GCNII等），比较其误差和有效范围。实验表明：在TRIP任务中，曲率与GNN误差无显著相关；在ECHO任务中，模型排名差距并非完全由长程驱动，表明存在其他瓶颈。

**⚠️ 局限性**

局限性包括：特征假设为独立同分布，无法覆盖真实数据中的特征相关性；仅针对无噪声的可测定目标，未考虑目标噪声；缺乏对更复杂、动态图结构的进一步验证。

---

## 707. Author Representation Strategies for Zero-Shot Authorship Attribution: A Comparative Study of LLM-Based and Embedding-Based Approaches

**arXiv ID:** 2610.03531 | [PDF](https://arxiv.org/pdf/2610.03531v1)

**作者:** Nudrat Habib `[一作]` (Luleå University of Technology), Elisa Barney `[通讯]` (Luleå University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了零样本作者归因任务，比较了标签仅提示、作者样本、LLM生成的风格描述和风格嵌入四种作者表征策略。

**💡 创新点**

创新点在于提出两阶段LISA嵌入框架，通过候选空间缩减和维度选择显著提升归因性能，并系统比较多种表征方式的效果。

**🔧 技术方法**

使用了大语言模型提示、LLM生成风格描述、LISA风格嵌入与余弦相似度、两阶段候选筛选与特征维度选择等技术。

**📊 数据集**

实验基于Blog Authorship Corpus（博客作者语料）及其子集进行评估。

**📈 对比分析**

在相同实验设置下，三种开源指令调优LLM（Mixtral、Gemma、Qwen）进行比较：标签仅提示接近随机，样本表征约43%准确率，描述约33%，嵌入单阶段36%，两阶段提升至最高56.6%。

**⚠️ 局限性**

局限性包括单一样本或描述信息不足、LLM生成的描述可能缺乏辨识度，以及当前开源LLM在零样本归因仍受限，嵌入方法仍需进一步优化检索和维度选择策略。

---

## 708. Cephalonauts One: A deep fMRI dataset for decoding naturalistic speech in the human brain

**arXiv ID:** 2610.03558 | [PDF](https://arxiv.org/pdf/2610.03558v1)

**作者:** Antoine Collas `[一作]` (Karavela), Alexis Thual `[通讯]` (Karavela)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了一个深度人类大脑功能磁共振（fMRI）数据集，每个受试者可获得30小时的全脑3T扫描，同时收集对应的音频、文字转录及预训练模型嵌入，并提供标准化的音频段检索基准任务。

**💡 创新点**

创新点在于：①在同一受试者内大幅增加了训练时长（30h/受试者，远超现有语音fMRI数据集），②将自然语音的音频和文字转录对齐并嵌入高维表征（MTEB 4096维、MAEB 3584维），③设计了基于检索的解码评估框架并公开基线模型及评测脚本。

**🔧 技术方法**

使用技术包括：高阶fMRI预处理（空间校正、运动校正、表面投影）、多模态嵌入生成（MTEB、MAEB）、一层神经网络解码器、CLIP对比损失、AdamW优化和余弦相似度检索。

**📊 数据集**

数据集：3名健康受试者（French）在单一3T Siemens Cima扫描仪上，每人30小时fMRI；音频来自Les Pieds sur Terre播客，包含文本转录；嵌入使用MTEB（文本）和MAEB（音频）。

**📈 对比分析**

与基线相比，解码性能随训练时长呈对数线性提升：Cosine Similarity从0.075–0.090提升到0.144–0.155，Median Relative Rank从0.11–0.19降至0.022–0.034，Top‑10@2000从3.8–6.4%提升到22.3–27.3%；并未出现饱和点。

**⚠️ 局限性**

局限性包括：仅3名受试者（难以评估群体变异性）、单一语言（French）与单播客来源、单站点单扫描仪（缺乏跨设备泛化）、时间高分辨率但空间分辨率较低、仅评估检索任务（不涵盖生成或多类分类等更开放的解码场景）。

---

## 709. UniIntervene++: An Adaptive Intervention Agent for Efficient Real-World Reinforcement Learning

**arXiv ID:** 2610.03620 | [PDF](https://arxiv.org/pdf/2610.03620v1)

**作者:** Yudong Lin `[一作]` (Nanyang Technological University), Ziwei Wang `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `51c0528b-f690-4182-ae60-bb5f046c276c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种自适应干预代理，能够在在线强化学习过程中动态分配控制权，在自主执行和多种辅助行为之间切换，以提升机器人操纵任务的成功率。

**💡 创新点**

创新点在于：①将演化的 RL 策略、轨迹校正、代码策略统一建模为选项，采用半马尔可夫决策过程学习其相对价值；②引入竞争适应干预与自适应 RL 探测，持续评估并更新 RL 策略的能力；③通过耦合经验学习让辅助经验直接提升 RL 策略，形成闭环。

**🔧 技术方法**

使用 Double‑DQN 调度器、SMDP 框架、轨迹校正和 CodePolicy 选项、主动探测窗口、耦合经验重放等技术。

**📊 数据集**

在五个真实世界操纵任务上测试：Ring Transfer、Tube Insertion、USB Insertion、Wipe Whiteboard、Towel Folding，使用 UR7e 机器人与 Robotiq 扳手实现。

**📈 对比分析**

与 HIL‑SERL、AutoSERL、UniIntervene 等基线对比，平均成功率 89.67%（比最佳基线高 6 点），人类干预率仅 0.77%（比 AutoSERL 低 94.6%）。

**⚠️ 局限性**

局限性：仍需人工干预作为安全保障；性能受任务结构和选项设计影响；在任务之外的泛化能力尚未充分验证；算法对感知错误和实时计算资源有一定依赖。

---

## 710. A Path Integral Surrogate for Multi-Step Gradient Inversion in Federated Learning

**arXiv ID:** 2610.03597 | [PDF](https://arxiv.org/pdf/2610.03597v1)

**作者:** Agnivo Ghosh `[一作]` (Indian Institute of Technology Kharagpur), Saumik Bhattacharya `[通讯]` (Indian Institute of Technology Kharagpur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种梯度逆向攻击PI-SME，利用贝塞尔曲线拟合FedAvg客户端的局部训练轨迹，并通过Gauss–Legendre多节点数值积分逼近梯度场的路径积分，从而恢复被隐藏的训练样本；

**💡 创新点**

创新点在于将多步FedAvg更新视为梯度场的路径积分，并采用多节点数值积分方法（Gauss–Legendre）而非传统的单点近似，显著提高逆向恢复的精度；

**🔧 技术方法**

技术方案包括贝塞尔曲线拟合、Gauss–Legendre数值积分、学习校准向量、多节点梯度评估、正则化（TV、控制点正则），并使用Adam优化器进行联合优化；

**📊 数据集**

实验数据集为CIFAR-100（彩色32×32图像）和FEMNIST（灰度28×28图像），用于评估逆向效果；

**📈 对比分析**

与SME、NL-SME以及IG、GI-NAS、DGGI等多种梯度逆向攻击进行对比，PI-SME在PSNR、SSIM、LPIPS、L_sim等指标上均优于对手，尤其在长局部训练步骤（T=250）和类别稀缺（C=10）场景下提升显著；

**⚠️ 局限性**

局限性包括对节点数m的经验调优仍有必要，计算量随节点数线性增长，且在极端类别不足或非常短的训练步数时，逆向效果仍有限。

---

## 711. Objects Without Morphisms: What LLMs for Mathematics Do Not Represent

**arXiv ID:** 2610.03551 | [PDF](https://arxiv.org/pdf/2610.03551v1)

**作者:** Yanli Wang `[一作]` (Imperial College London), Haohan Wang `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大型语言模型在跨子领域翻译时是否保留隐式假设，揭示其在向更一般化框架翻译时普遍范围扩大而非缩小的现象。

**💡 创新点**

提出一种将真值、内容与范围分别编码并且配备判定隐式假设的无评判者度量的新评价仪器，首次系统性揭示模型在翻译过程中出现的范围漂移。

**🔧 技术方法**

利用无评判者正则表达式检测、人工评审、模型生成与对照等技术，对模型输出进行自动与人工双重编码与验证。

**📊 数据集**

使用MELD benchmark（90条已注册+45条保留）以及ProofWiki人工对齐对作为测试数据集。

**📈 对比分析**

通过七个大模型在两种翻译方向（更一般化与更具体化）下的范围变化率进行比较，发现抽象方向约60.6%范围扩大，具体方向约0.3%缩小；模型能力与方向无关，指令提升率但不实现方向控制。

**⚠️ 局限性**

局限性包括仅测试通用模型、样本量有限、正则模式受限、未包含专门训练的数学模型，且注释质量受标注者主观影响。

---

## 712. ZeroMAG: Zero-Shot Multimodal Adapter Generation for Plug-and-Play EEG Foundation Models

**arXiv ID:** 2610.03546 | [PDF](https://arxiv.org/pdf/2610.03546v1)

**作者:** Yubo Wang `[一作]` (Brown University), Cuntai Guan `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

研发ZeroMAG，一个零样本多模态适配器生成框架，能够在冻结EEG基础模型上无目标标签、无目标侧优化地扩展多模态功能。

**💡 创新点**

① 配置不变的适配器架构支持任意伴随信号组合；② 通过基于模态、主体、任务的无标签条件构造，生成目标专用适配器；③ 在低维结构化潜在空间中进行功能约束的条件扩散生成，确保生成适配器与参考器件在预测上保持一致。

**🔧 技术方法**

迁移学习+冻结EEG编码器和预测头；模块化时频编码+共享协调的适配器结构；条件编码器+Diffusion（DDIM）+Mixture-of-Experts解码器；VAE+MoE+DiT用于潜在表示学习与生成；预测功能损失（KL）作为功能一致性约束。

**📊 数据集**

六个保留的目标数据集（ISRUC-S3、HMC、MASS、SEED-V、EEGMAT、FoG），覆盖睡眠分期、情绪识别、认知负荷、运动想象、步态冻结等；来源数据用于训练适配器库，目标数据保持不参与训练。

**📈 对比分析**

与EEG-only、均值/最近源权重、Direct MLP等标签无关基线以及全模态监督微调进行比较。ZeroMAG在所有18个backbone–dataset组合中获得最高的B-ACC，平均比EEG-only提升7.22点，超越Direct MLP 4.89点，且仅平均落后0.50点于监督适配，恢复了93.5%监督收益。

**⚠️ 局限性**

需要预先训练好的多模态适配器库和EEG基础模型；无法处理在训练集中未出现的模态类型；对目标侧无标签和无优化的假设限制了对极端分布偏移的适应能力；生成器仅在冻结的潜在空间内操作，可能受限于源域覆盖范围。

---

## 713. RATE: Risk-Aware Tactile Encoding for Contact-rich Robotic Manipulation

**arXiv ID:** 2610.03538 | [PDF](https://arxiv.org/pdf/2610.03538v1)

**作者:** Yuyao Jiang `[一作]` (Nanyang Technological University), Ziwei Wang `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种风险感知触觉编码（RATE）框架，通过历史条件的预测和警报监督学习任务相关的触觉风险表示，并通过轻量残差路径将该表示融合进已有的视觉-触觉策略，实现对接触风险的精准捕捉与控制。

**💡 创新点**

创新点包括：①将触觉历史的短期未来变化预测与连续风险警报监督相结合，形成以任务风险为导向的触觉上下文表示；②使用零初始化的残差适配器，仅更新0.59%参数，轻量高效地将风险表示注入预训练策略；③在仿真与真实机器人上通过连续警报权重重塑模仿损失，提升关键接触步骤的决策质量。

**🔧 技术方法**

技术手段包括：两层LSTM的时间序列聚合、基于图像的未来变更预测、连续风险警报回归、残差多层感知网络（MLP）、ACT策略的动作块预测、以及基于警报加权的模仿学习。

**📊 数据集**

使用了UniVTAC仿真基准（8个任务）和三项真实机器人任务（USB插入、瓶盖拧紧、插头插入），采集自Franka Panda与xArm 7机器人，配备GelSight Mini与Daimon触觉传感器，数据规模分别为100/50次演示。

**📈 对比分析**

与ACT（视觉仅）、ACT+UniVTAC、VITaL、π_0.5、StarVLA-α和GigaWorld-Policy等基线比较；在UniVTAC上RATE宏平均成功率达到71.8%，位列首位，并在所有任务类别均表现最好；在真实任务上RATE的任务成功率分别为75%、85%和60%，宏平均73.3%，并在安全成功率上均超过所有基线（宏平均92.2%）。

**⚠️ 局限性**

局限性在于对触觉幅度变化较大的交互阶段，细粒度相对变化可能被稀释，导致某些细微但任务相关的触觉差异难以区分；此外，警报信号依赖任务特定的可观测量，跨任务迁移时需要重新设计警报生成方式。

---

## 714. Structured Composition of Verifiable Atomic Insights for Table-to-Report Generation

**arXiv ID:** 2610.03525 | [PDF](https://arxiv.org/pdf/2610.03525v1)

**作者:** Teng Lin `[一作]` (HKUST(GZ)), Nan Tang `[通讯]` (HKUST(GZ))

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于原子洞察的多关系证据图合成框架，用于自动从关系型表生成可验证的、结构完整的分析报告。

**💡 创新点**

将分析拆分为可执行的原子洞察，构建多关系图并通过专门的组合算子系统化生成比较、相关、上下文、时间与因果等复合洞察，从而解决传统顺序探索导致的探索偏差和缺乏创新问题。

**🔧 技术方法**

采用预定义的SQL分析模式、统计显著性筛选、LLM‑as‑Judge、NetworkX多图与Leiden社区检测、Granger因果检验+LLM因果验证、GPT‑5.5生成复合洞察与报告等技术。

**📊 数据集**

在InsightBench、DDR‑Bench和T2R‑Bench三大基准上进行评估。

**📈 对比分析**

与Pandas Agent、AgentPoirot、DeepSeek‑R1、Claude 4.5 Sonnet及GPT‑5.5等基线对比；在InsightBench获得0.73的LLaMA‑3‑Eval（领先AgentPoirot13.3个百分点），在DDR‑Bench最高达79.34%，在T2R‑Bench平均得分83.5%，显著优于所有基线。

**⚠️ 局限性**

主要局限包括：复合洞察生成是耗时瓶颈，因果关系生成对整体性能贡献有限；对大型表仍较慢；系统依赖预定义分析模式，未处理多语言或非结构化数据。

---

## 715. PrivDev: Mapping Static-Analysis Data Types to DPV

**arXiv ID:** 2610.03518 | [PDF](https://arxiv.org/pdf/2610.03518v1)

**作者:** Simon Bernbeck `[一作]` (PUC-Rio), Juliana Alves Pereira `[通讯]` (PUC-Rio)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 PrivDev Layer 1，将 Bearer CLI 静态分析检测到的 122 条个人数据类型映射到 DPV‑PD 个人数据分类，并通过 ODRL 生成可查询的 RDF 知识图谱，进而链接潜在相关的 GDPR 条文；

**💡 创新点**

创新点在于：①采用两阶段映射策略——先用确定性匹配处理 43 条精确对应，再用检索‑驱动的 LLM 解决 79 条非平凡映射；②将 SHACL 验证、DeepEval LLM 质量评分与多统计（Gwet AC1/AC2 等）人类评估结合的 FEDS 评估框架，形成跨技术的多证据评判；

**🔧 技术方法**

使用技术包括：LangGraph 调度管道、FAISS 索引检索+Cross‑Encoder 排序、GPT‑4o‑mini LLM 生成映射、SPARQL/SHACL/OOPS! 进行结构验证、DeepEval 进行语义相关性评分、Gwet AC1/AC2 等统计衡量人类评估一致性；

**📊 数据集**

使用的数据集包括：Bearer CLI 输出的 122 条数据类型标签；DPV 2.3 的个人数据分类与 DPV‑OBL 扩展的 GDPR 条文；9 名 annotator 评审的 79 条非精确映射；

**📈 对比分析**

评估方法：通过 5 个验证门（G1‑G5）——G1 SHACL 验证无违规；G2 OOPS! 无严重错误；G3 SPARQL 竞价查询返回预期结果；G4 人类评估原始一致率 0.72、Gwet AC2 0.88；G5 DeepEval 语义相关性平均 0.78。整体覆盖率 100%，生成 118 条 ODRL 资源，表明映射质量合理；

**⚠️ 局限性**

局限性：①依赖托管 LLM，输出可能变动；②DPV 覆盖不足，存在类缺口；③仅验证 Bearer CLI 的 122 条类型，未直接验证其他扫描器的适用性；④人类评估仅针对非精确映射，精确匹配未外部验证；⑤未评估工具对实际开发者工作量或错误率的影响。

---

## 716. DuoMatching: Joint-Marginal Distribution Matching for Few-Step Video Generation

**arXiv ID:** 2610.03543 | [PDF](https://arxiv.org/pdf/2610.03543v1)

**作者:** Jiahao Zhan `[一作]` (MMLab, CUHK), Tianfan Xue `[通讯]` (MMLab, CUHK)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种统一的联合-边缘分布匹配框架，用于训练少步骤视频生成模型，使其在保持实时推理能力的同时显著提升视觉质量与语义对齐。

**💡 创新点**

创新点在于将传统的联合分布匹配（joint DMD）与图像教师提供的边缘分布匹配（marginal DMD）结合，并引入轻量级的LatentBridge模块解决视频和图像潜在空间的不匹配，以及Latent Variation Sampling (LVS) 技术在时间上高效分配边缘监督。

**🔧 技术方法**

核心技术包括：联合-边缘分布匹配目标、图像教师驱动的边缘 DMD、LatentBridge 用于映射时间压缩的视频潜在到帧级图像潜在、LVS 用于在时间维度上选择多样化的监督帧、以及在现有的 Causal Forcing++ 与 CausVid 生成器上进行的微调。

**📊 数据集**

使用了 29,400 条高审美质量且运动丰富的 OpenVid 子集进行训练；评估数据采用 VBench 官方 prompt 套件和额外的 400 条挑战性 prompts，计算语义、审美、成像、动态与运动平滑度等指标。

**📈 对比分析**

与 Self Forcing、Reward Forcing、Causal Forcing++、One-Forcing、以及 CausVid 等基线相比，该方法在视觉质量、语义对齐、动态表现上均获得显著提升，VBench 总分提升约 10-15%，并在 2AFC 人工评测中获得 80% 以上的整体优先率。

**⚠️ 局限性**

局限性包括：仍依赖图像教师的能力，若教师模型不足可能难以获得更高质量；LVS 需要调参（K 的选择对性能影响明显）；方法在极端快速运动或极长视频时仍可能出现细节模糊或微小漂移；训练仍需大量 GPU 与显存，且对不同视频编码方式的泛化需要进一步验证。

---

## 717. 4DCodeBench: Benchmarking Agents on Inverse Graphics of Dynamic Scenes

**arXiv ID:** 2610.03715 | [PDF](https://arxiv.org/pdf/2610.03715v1)

**作者:** Ruihong Shen `[一作]` (Johns Hopkins University), Jiajun Wu `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出4D逆向图形基准4DCodeBench，要求智能体通过代码生成恢复视频中的三维几何与动态。

**💡 创新点**

创新点在于将可执行图形程序作为目标，实现动态场景的代码生成，并提供多模态评估与人类对齐。

**🔧 技术方法**

采用多模态编码模型、视觉-语言模型评估、Blender渲染、光流、轨迹、Chamfer等多种技术。

**📊 数据集**

使用200个场景，包括100个真实视频和100个合成场景，涵盖变形、流体、破裂等物理现象。

**📈 对比分析**

通过18个前沿模型进行基准测试，使用感知、二维/三维几何、动态等指标，结果显示强模型在静态重建上表现优异，但动态重建仍显不足。

**⚠️ 局限性**

局限在于缺乏对物理机制的约束，模拟与预设轨迹之间差距大，且对更长时延预测和物理推理的评估不足。

---

## 718. ProAR: Learning Prospective Reasoning with Autoregressive Video Models

**arXiv ID:** 2610.03664 | [PDF](https://arxiv.org/pdf/2610.03664v1)

**作者:** Linghui Shen `[一作]` (The Hong Kong Polytechnic University), Muhao Chen `[通讯]` (University of California, Davis)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ProAR 框架，使自回归视频生成具备前瞻性推理能力。

**💡 创新点**

创新点在于同时加入目标引导（goal belief）和过渡引导（future representation self‑alignment），实现长短期双向监督。

**🔧 技术方法**

采用自回归视频扩散模型、目标帧预测的非对称注意力掩码以及教师-学生特征对齐的轻量级预测器。

**📊 数据集**

在 13 个视觉推理任务（VBVR、VideoRLVR）以及 WorldArena 机器人仿真基准上进行评估。

**📈 对比分析**

与标准自回归基线相比，ProAR 在 VBVR 10 任务平均得分从 0.663 提升到 0.801，VideoRLVR 成功率提升至 52.97，且训练效率提高到只用 25% 步骤即可达到最优。

**⚠️ 局限性**

局限性包括对自回归模型结构的依赖，过渡引导需在后期训练中激活，且在极端长序列或复杂物理交互时仍可能出现漂移。

---

## 719. SigLIP2 for aerial fire risk classification

**arXiv ID:** 2610.03689 | [PDF](https://arxiv.org/pdf/2610.03689v1)

**作者:** Yunus Serhat Bıçakçı `[一作]` `[通讯]` (Marmara University), Yunus Serhat Bıçakçı (Marmara University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在公开的FireRisk航空影像数据集上，对SigLIP2图像编码器进行迁移学习实验，比较冻结编码器和完整微调两种训练策略；同时提供可复现的分割、预处理和实验框架，并对类别级错误进行分析。

**💡 创新点**

首次系统评估SigLIP2在航空影像火灾风险七类别分类任务中的性能，构建可复现的训练/验证分割，记录数据来源、预处理和模型选择细节，并提供详细的类别误差分析；为后续在不同视觉编码器和预训练域间的比较奠定基准。

**🔧 技术方法**

使用SigLIP2视觉编码器（Base 16×16 patch, 224×224），配合可学习的LayerNorm与7维线性分类头；采用AdamW、label smoothing、cosine学习率调度、随机裁剪/翻转/旋转增强；对比冻结probe（仅头部可训练）与全适配（全参数可训练）两种实验配置。

**📊 数据集**

公开的FireRisk镜像（70,331张320×320 RGB PNG），标签来源于2020 Wildfire Hazard Potential地图；在分层抽样下划分为49,231训练、10,552验证和10,548保留测试样本。

**📈 对比分析**

在单一训练种子下，冻结probe得到55.95%准确率/50.19%宏F1；全适配得到63.05%准确率/58.94%宏F1，提升约7%准确率、8.8%宏F1；细分类别F1普遍提升，但“very high”类别召回仅为33.47%。

**⚠️ 局限性**

仅在单一随机种子和固定分割下评估，未使用独立测试集或空间/时间独立性验证；“very high”类别召回低，误分类多集中于向低风险类别；缺乏多模态或时间信息支持，且无法证明模型在新地区的泛化能力。

---

## 720. PoCoFL: POlicy-COmpliant Federated Learning

**arXiv ID:** 2610.03650 | [PDF](https://arxiv.org/pdf/2610.03650v1)

**作者:** Dominik Roy George `[一作]` (KU Leuven), Aysajan Abidin `[通讯]` (KU Leuven)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一个通用的可验证的联邦学习框架PoCoFL，支持在不同联邦学习类型、政策与加密实现之间分离；

**💡 创新点**

创新点在于将联邦学习类型、政策语义和加密实现模块化，允许政策通过零知识证明来验证，同时保持网络拓扑无关和加密可替换；

**🔧 技术方法**

技术主要包括可绑定承诺、非交互零知识证明（EZKL）、多聚变加密（Threshold‑HE）以及基于Flower的联邦学习框架；

**📊 数据集**

实验使用MNIST数据集，在100个客户端上进行Vanilla、Personalized、Continual及Threshold‑HE四种联邦学习场景；

**📈 对比分析**

对比方法是将开启政策验证与不验证两种设置在模型准确率、隐私和攻击抵抗力上进行对比，结果显示开启政策验证后在模型鲁棒性和准确率上提升，且证明生成/验证时间和通信成本可接受；

**⚠️ 局限性**

局限在于只评估了单机环境下的证明开销，未考虑大规模并行网络延迟，且对不同攻击类型的安全性仅做了有限实验。

---

## 721. FlowHMR: Physically Plausible Motion Capture from Video

**arXiv ID:** 2610.03691 | [PDF](https://arxiv.org/pdf/2610.03691v1)

**作者:** Zhanke Wang `[一作]` (Peking University), Linchao Bao `[通讯]` (Tencent)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文提出一种从单目视频恢复全局 3D 人体运动的框架，该框架先用流匹配预训练生成器，再通过物理仿真奖励的强化学习进行后训练，以实现既符合视频又具备物理可执行性的人体运动。

**💡 创新点**

创新点包括：① 将视频条件生成与流匹配结合，克服深度模糊导致的多解问题；② 引入两种奖励（保真度与物理跟踪）在后训练中同时约束生成器，避免单一奖励导致的“奖励破解”；③ 提出了 Wild-4K 公开评估集，涵盖多样化、困难、遮挡等真实网络视频；④ 通过 MixGRPO 在物理仿真环境中直接优化生成器参数，首次实现生成器与物理反馈的无缝协同。

**🔧 技术方法**

技术手段包括：多模态扩散 Transformer (MM‑DiT) 作为视频条件生成器；流匹配 (Flow Matching) 作为无监督预训练目标；MixGRPO 强化学习框架；冻结的 PHC+ 控制器在 Isaac Gym 里做物理跟踪评估；视觉特征编码器采用 SAM 3D Body；以及 SMPL‑H 运动表示与前向运动学监督。

**📊 数据集**

使用的数据集：
- 约 3,000 小时的合成视频与对应 3D 动作（基于 BEDLAM/BEDLAM2.0 扩展），用于预训练；
- Wild‑4K（4,182 条互联网视频）用于评估；
- RICH、Human3.6M、3DPW 等公开基准用于对比实验。

**📈 对比分析**

与 GVHMR、PromptHMR、GENMO、DuoMo 等现有方法做盲目对比。结果显示：
- 在 Wild‑4K 上物理跟踪成功率从 78.79% 提升到 82.47%；
- 人类评测中优胜率在 61.5%–79.2% 之间，显著高于基线；
- 在 RICH 上的 MPJPE、根轨迹误差和追踪成功率均优于对比方法；
- 轨迹平滑度、地面接触误差等物理可执行性指标也得到明显改善。

**⚠️ 局限性**

局限性：
- 物理仿真仅在单一地面平面上进行，缺少对物体/支撑面交互的建模；
- 后训练依赖大量物理仿真 roll‑out，计算成本高，难以扩展到更大规模或更长序列；
- 对复杂环境与多种物体交互的场景支持不足；
- 生成多样性虽然存在，但在极端遮挡或高速动作下仍有一定误差。

---

## 722. Credit Where It Matters: Dependency-Aware Policy Optimization for Terminal Agents

**arXiv ID:** 2610.03634 | [PDF](https://arxiv.org/pdf/2610.03634v1)

**作者:** Yu Li `[一作]`, Lei Feng `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 DepGPO，利用执行依赖关系将轨迹优势在终端任务的交互步骤间进行重新分配。

**💡 创新点**

通过构造命令依赖图并追踪到验证器读取的资源，给写操作分配分数并将读操作的支持信用传播，提供比轨迹级或基于状态的信用分配更细粒度的奖励分配。

**🔧 技术方法**

执行轨迹分析、命令依赖图构建、读写依赖信用分配以及基于 GRPO 的策略优化与剪辑损失。

**📊 数据集**

使用 Terminal-Bench 2.0/2.1 数据集，结合 Qwen3.5-9B 与 Qwen3.6-27B 语言模型，并在 SETA 与 TMAX 两套训练数据上进行实验。

**📈 对比分析**

与 GRPO、DAPO、DPPO、GiGPO、GraphGPO 等基线对比，DepGPO 在所有八种模型/数据配置下均取得最高 pass@1，提升幅度在 3.2%–10.0% 之间（相较最强基线），并表现出更稳定的训练收敛。

**⚠️ 局限性**

对依赖图构造的准确性和完整性敏感，无法覆盖未记录的读写操作；对验证器资源集合变化敏感；对小模型的提升有限；依赖于任务验证器的实现，若验证器策略变动需重新调整。

---

## 723. What Should World Models Forget? Stratified Retention for Continual Adaptation

**arXiv ID:** 2610.03713 | [PDF](https://arxiv.org/pdf/2610.03713v1)

**作者:** Nishit Anand `[一作]` (University of Maryland), Dinesh Manocha `[通讯]` (University of Maryland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

分析持续世界模型的失效模式，提出基于不变性时间尺度的知识分层和差分保留评估框架

**💡 创新点**

首次将失效分为灾难性遗忘、刻意忘却和过时性，提出按不变性时间尺度分层并结合不变性回归测试与修订延迟的双重指标

**🔧 技术方法**

使用现有物理概念探针（如WorldBench、PhyGround）和持续学习技术（如经验回放、EWC、蒸馏）构建评估协议

**📊 数据集**

主要参考公开 benchmark 如 Minigrid、MiniHack、Procgen、Atari、Minecraft 等，但本文未进行实验，使用这些数据集作为潜在测试基准

**📈 对比分析**

通过对比传统的向后迁移/遗忘指标，指出其无法区分正确修订与错误遗忘；差分保留指标能够同时度量 invariant retention 与 revision latency，暂无性能数值

**⚠️ 局限性**

仅存在物理不变性探针，几何一致性和因果结构的探针尚缺失；时间尺度 τ 的界定与估计仍待研究；是否需要架构支持以局部化不变性仍不明

---

## 724. NeutronGym: Physics-Graded Neutron Instrument Design for LLM Agents

**arXiv ID:** 2610.03631 | [PDF](https://arxiv.org/pdf/2610.03631v1)

**作者:** Lijie Ding `[一作]` (Oak Ridge National Laboratory), Changwoo Do `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了可执行、可验证的中子仪器设计环境 NeutronGym，评估语言模型在无参考答案的设计任务中的表现，并通过强化学习显著提升模型性能。

**💡 创新点**

创新点在于：①引入基于 McStas 的物理可验证执行环境；②构建无 LLM 判决的多层级奖励阶梯；③使用程序化生成任务族与无参考验证，确保模型真正“设计”而非记忆；④门控检测（no‑model probes）避免快捷方式。

**🔧 技术方法**

使用了 McStas 物理模拟器、22 个验证工具、Model Context Protocol、GRPO 强化学习、LoRA 微调、门控检测、奖励阶梯、以及工具服务器与评测 harness。

**📊 数据集**

数据集包括：四个程序化生成的匹配/最大化任务族（共 4 家族），12 个公开中子仪器的复制/改进任务（共 16 评估任务），两套未公开的 hold‑out 仪器及其扰动变体，以及 sandbox 环境的测试。

**📈 对比分析**

与 Claude、Gemini、Llama、Qwen3‑32B/8B 等多种 LLM 进行一次性或循环评估；训练后 Qwen3‑8B 在 4 个 gated 家族的 hold‑out 成功率从 11% 提升至 77%，超过未训练的 32B；在公开任务中最多 7/16 通过，但未达到改进目标；与经典优化对比，强化学习模型在奖励上与手工物理逆推相近。

**⚠️ 局限性**

局限性包括：评测规模小，任务仅固定布局参数且不涉及拓扑选择；模拟理想化（无背景、无重力）；门控检测不足，部分任务易被快捷方式绕过；训练结果仅基于单个随机种子，跨种子稳定性差；对多任务学习效果不佳；依赖物理模型的精确性，易受模拟误差影响。

---

## 725. World Embedding Benchmark

**arXiv ID:** 2610.03632 | [PDF](https://arxiv.org/pdf/2610.03632v1)

**作者:** Yiqi Liu `[一作]` (World-Embedding Team), Chenghao Xiao `[通讯]` (World-Embedding Team)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了World Embedding Benchmark，构建了8,000个受控物理仿真案例，结合视频、物理注释和文本描述，设计文本-视频检索、物理属性回归与视频-描述匹配三类评测任务。

**💡 创新点**

创新点在于同时评估视频嵌入的物理信息可编码性与跨模态对齐性，并通过物理特定的对比学习展示对齐与可回归信息之间的权衡；进一步验证物理嵌入可提升检索增强视频生成的物理真实性。

**🔧 技术方法**

利用多物理仿真引擎（OpenFOAM、DOLFINx、Chrono、HCIPy、Meep）生成数据，采用轻量级线性回归探针评估可回归物理量，使用对比学习对多模态嵌入模型（Video‑CLIP‑XL、Qwen3‑VL‑Embedding、Omni‑Embed‑Nemotron、LCO‑Embedding）进行物理专属微调，最后与MiniMax‑H3结合实现检索增强视频生成。

**📊 数据集**

数据集为World Embedding Benchmark，包含80个物理家族（流体力学、固体力学、动力学、光学与电磁学），每个家族100个参数化实例，生成视频、物理属性与自然语言描述。

**📈 对比分析**

与基线对比显示：预训练嵌入模型在检索与家族内匹配表现接近随机，但回归探针能恢复多维物理量；物理对比微调显著提升检索召回率与匹配准确率，但回归误差略升；检索增强生成实验中，使用物理对齐模型检索到的参考视频提升MiniMax‑H3生成的物理真实性，平均得分从0.60提升至0.66。

**⚠️ 局限性**

主要局限在于：①对齐与可回归之间存在权衡，无法同时优化；②评测仅基于仿真视频，缺乏真实世界多样性；③对比学习需要大量物理对话样本，且训练收敛慢；④尚未探索更精细的物理属性恢复方法或多任务联合训练。

---

## 726. Decoding the Functional Roles of Register and High-Norm Patch Tokens in Vision Transformers

**arXiv ID:** 2610.03698 | [PDF](https://arxiv.org/pdf/2610.03698v1)

**作者:** Neel Varma `[一作]`, Vasu Sharma `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对 DINOv2 中的寄存器令牌和高范数异常补丁令牌进行稀疏自编码器（SAE）分析，探究它们在视觉 Transformer 中的功能与语义角色。

**💡 创新点**

首次揭示寄存器令牌主要承载高级语义信息，而高范数异常令牌主要携带结构/纹理信息，并证明两者在功能上存在显著不对称性。

**🔧 技术方法**

利用稀疏自编码器、Gemma 3 视觉‑语言模型、UMAP 聚类、CLIP 空间对齐以及特征层级消融等技术实现可解释性与因果评估。

**📊 数据集**

实验基于 ImageNet‑1k 的预训练 DINOv2‑small 与加寄存器版本的模型，采样层 8 的激活。

**📈 对比分析**

通过消融 top‑5 SAE 特征并测量余弦相似度下降和注意力地图变化，发现寄存器令牌的消融导致 48% 的相似度下降，而异常令牌仅 0.3%，显著表明寄存器对表示稳定性更关键。

**⚠️ 局限性**

研究局限于 DINOv2‑small 的第 8 层、有限的 SAE 训练预算以及主要依赖定性解释方法，未验证不同规模/层级模型的普适性，也未揭示令牌路由的具体网络路径。

---

## 727. LESSER: Post-Training Data Selection with Output-Layer Gradients

**arXiv ID:** 2610.03702 | [PDF](https://arxiv.org/pdf/2610.03702v1)

**作者:** Lyuxin David Zhang `[一作]` (University of Pennsylvania), Anton Xue `[通讯]` (University of Texas at Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种只利用输出层梯度进行数据选择的高效方法（LESSER），替代传统的全梯度特征，用于大语言模型的后训练任务（SFT、RL 和教师蒸馏）。

**💡 创新点**

创新点在于发现输出层梯度能够在不执行完整反向传播的前提下，近似全梯度特征，从而显著降低特征提取成本，同时保持与全梯度方法相近的下游性能。

**🔧 技术方法**

核心技术包括：①将全梯度特征替换为输出层梯度；②在前向传播中直接计算输出层梯度；③采用与原方法相同的选择规则（如 GIST、GradAlign、GRACE）并在低维投影下进行相似度计算。

**📊 数据集**

使用的主要数据集包括：
- SFT：Tulu V2 指令‑响应样本池 + TyDiQA、MMLU‑Pro、GSM8K、Codex、BBH 作为查询与测试集；
- RL：AMC22 任务池、Countdown 等在线 RL 环境；
- 蒸馏：10‑14 名教师模型的生成回答。

**📈 对比分析**

与全梯度方法（GIST、GradAlign、GRACE）以及随机与隐藏层特征基线进行对比。实验表明：
- SFT：LESSER 的下游分数与全梯度方法差距平均 1.3‑2.6 分；
- RL：在受噪声奖励和在线 RL 场景中，LESSER 与 GradAlign 的选择结果高度一致，最终准确率相同；
- 蒸馏：LESSER 能恢复与 GRACE 相同的最佳教师。成本方面，SFT 特征提取 FLOP 下降 9.7×，RL 下降 3.0×，蒸馏 下降 3.9‑5.4×，壁钟时间亦大幅降低。

**⚠️ 局限性**

局限性：
- 仅在多模型、任务和选择设置下验证，未覆盖所有后训练范式；
- 侧重于模型初始化阶段的训练效果，长时间训练下的表现未知；
- SFT 评估基于完整管线，未与全梯度做匹配检查，可能存在其他差异；
- 对输出层梯度有效性的理论解释仍不充分，需要进一步研究。

---

## 728. Forecasting from Counterfactual Simulator Rollouts: A Sim2Real Evaluation

**arXiv ID:** 2610.03662 | [PDF](https://arxiv.org/pdf/2610.03662v1)

**作者:** Angel Wang `[一作]` (Amazon), Carson Eisenach `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了利用仿真生成的反事实数据训练预测模型，以解决在部署新决策策略时预测目标冷启动问题，并在两项真实库存控制部署中验证了仿真训练预测器的有效性；

**💡 创新点**

创新点在于将Sim2Real关注点从策略转移扩展到预测模型转移，利用目标策略的仿真轨迹训练聚合需求预测器，并证明仿真训练优于历史离线训练，且可通过在线校准进一步提升；

**🔧 技术方法**

使用Exo-IDP库存仿真器生成对策性控制的反事实轨迹，构建多时域聚合预测网络，采用扩展窗口线性回归进行在线校准；

**📊 数据集**

使用了两套真实库存订单策略的历史部署数据（约20k产品一年期实验和约100k产品三个月期实验），以及相应的仿真器生成的轨迹；

**📈 对比分析**

与历史先前政策下基于真实数据训练的预测器做零样本对比；仿真训练预测器在所有预测时段均比历史基线低1.2–3.1%（Study1）和12.5–18.7%（Study2），相对提升8–21%和45–67%；在线校准进一步在3周以内误差降至约2.5%；

**⚠️ 局限性**

局限在于仅验证于库存控制场景，仿真与现实的逼真度有限且对不同策略依赖性未系统刻画；缺乏对不同仿真器和领域的泛化评估；

---

## 729. On-Board Anomaly Detection for Efficient Marine Environmental Monitoring

**arXiv ID:** 2610.03649 | [PDF](https://arxiv.org/pdf/2610.03649v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 730. Transcriptome-informed multi-modal AI for predicting neoadjuvant therapy response from breast cancer biopsies

**arXiv ID:** 2610.03693 | [PDF](https://arxiv.org/pdf/2610.03693v1)

**作者:** Jungkyu Park `[一作]` (Ataraxis AI), Krzysztof J. Geras `[通讯]` (Ataraxis AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

利用预处理的H&E切片和临床变量构建两阶段AI模型NEO，先通过MORPHEUS推断全基因组表达，再预测乳腺癌新辅助治疗后的病理完全缓解（pCR）概率。

**💡 创新点**

创新点在于：①将大规模无标记病理图像与转录组配对，利用自监督基础模型学习形态-转录组映射；②用推断的全基因表达作为中间表示，显著提升pCR预测性能、解释性和对肿瘤取样偏差的鲁棒性；③通过独立成分分析压缩表达并与临床变量融合，形成可解释的多模态预测。

**🔧 技术方法**

技术包括：自监督基础模型Falcon、基于注意力的多实例学习（MIL）、独立成分分析（ICA）、可解释提升机（EBM）与表格ResNet、随机效应元分析、校准、ICC与W/T变异分解、空间转录组验证以及病理学家对图像特征的人工审阅。

**📊 数据集**

数据来源：32个TCGA项目共8,742例（MORPHEUS训练与验证），5个乳腺癌新辅助治疗队列1,080例（NEO训练），9个独立评估队列1,412例，涵盖北美、拉丁美洲、欧洲和中东共14个队列，样本覆盖三种分子亚型和多种临床特征。

**📈 对比分析**

与基线临床变量、四种计算TIL生物标志物和病理学家评估的Ki‑67比较，NEO在外部评估中获得汇总AUROC 0.79（95% CI 0.73–0.85），显著优于临床变量（AUROC 0.75）、TIL指标（0.52–0.58）和Ki‑67（0.65）。模型对亚型的判别保持一致，校准良好；ICC为0.93，W/T阈值0.15仅需一张切片即可满足稳定性要求。

**⚠️ 局限性**

局限性包括：①回顾性、多中心队列，缺乏前瞻性验证；②仅评估pCR作为终点，未涉及长期预后；③与TIL指标的比较基于自己重实现，可能存在实现差异；④在二级终点RBC与结节响应的样本量有限，结果为探索性；⑤对小基因列表的选择未优化，且模型训练未考虑新辅助药物多样性。

---

## 731. Revisiting Input Time-frequency Representations in Multi-pitch Estimation for Vocal Ensembles

**arXiv ID:** 2610.03656 | [PDF](https://arxiv.org/pdf/2610.03656v1)

**作者:** Junyoung Koh `[一作]` (Yonsei University), Hao-Wen Dong `[通讯]` (University of Michigan)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了使用线性STFT作为声乐合奏多音高估计的输入表示，验证其能在不同编码器下优于传统的HCQT，并显著降低特征提取成本。

**💡 创新点**

提出不使用频率自适应的HCQT，而直接采用固定频率分辨率的STFT，证明更简化的特征可以获得更好的性能。

**🔧 技术方法**

主要使用STFT、卷积神经网络(ConvNeXt、InceptionNeXt)、TCN、Conformer等深度学习编码器，并采用对数压缩的STFT幅度作为输入。

**📊 数据集**

在多组声乐数据集上评测，包括jaCappella、ACappellaSet、Korean Multi‑Singer、Cantoría、Dagstuhl ChoirSet、ESMUC以及合成数据ChoralSynth。

**📈 对比分析**

对比方法为与HCQT输入下相同编码器的多音高准确率（20¢与50¢容差），线性STFT在所有数据集上均获得更高的F1/准确率，并且特征提取时间缩短约30%-50%。

**⚠️ 局限性**

局限性包括参考F0的自动估计可能带来误差，训练数据主要为独立录制的音轨，缺乏真实合奏的空间与音准交互。

---

## 732. MRVQ: One Resident Index for Dimension- and Rate-Elastic Vector Search

**arXiv ID:** 2610.03651 | [PDF](https://arxiv.org/pdf/2610.03651v1)

**作者:** Sean Culatana `[一作]` (Atlassian), Kang Li `[通讯]` (Atlassian)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 Matryoshka Residual Vector Quantization (MRVQ) 的后置残差量化方法，用于在同一文档码流中同时支持不同的维度和码率，从而实现稀疏检索的弹性部署。

**💡 创新点**

创新点在于将残差量化和坐标前缀嵌套，以一次量化即可覆盖所有（维度, 码率）组合；相较于传统的每个码率单独训练量化器，MRVQ 在保持可比质量的前提下显著降低驻留内存。

**🔧 技术方法**

核心技术包括：残差向量量化 (L 级残差量化器，256 码字/级)、前缀加权损失函数、坐标前缀嵌套、以及对冻结嵌入向量的后置训练；对比方法还使用 PCA-标量量化作为低成本替代方案。

**📊 数据集**

使用 BEIR 公开的两个中等规模语料库 FiQA（57,638 文档）和 NFCorpus（3,633 文档），以及四类嵌入模型：MPNet、Mxbai、Nomic、BGE，分别在 768 或 1024 维空间。

**📈 对比分析**

对比方法包括：单独训练的 QINCo2 (每码率)、共享模型 Steelman、传统的 PQ、OPQ、AdANNS-OPQ；在相同码字大小下，MRVQ 在查询准确率上优于 PQ/OPQ，优于 AdANNS-OPQ；在驻留内存上，MRVQ 远低于每码率 QINCo2（17.8–22.0 倍）且低于共享 Steelman（1.89–2.02 倍）。PCA-标量量化在质量上与 RaBitQ 相近，但构建速度快 420–700 倍。

**⚠️ 局限性**

局限性包括：高码率下 QINCo2 训练不稳定导致检索性能崩溃；基于残差能量的排名预测器无法满足预设的相关性阈值；实验仅在两套中等规模数据集和四类嵌入模型上验证，未涉及大规模部署、延迟或能耗等实际系统指标。

---

## 733. When May a Bandit Leave Its Anchor? E-Process-Authorized Thompson Sampling under Non-stationarity

**arXiv ID:** 2610.03646 | [PDF](https://arxiv.org/pdf/2610.03646v1)

**作者:** Mayand Gulati `[一作]` (University of California Santa Barbara), WeiChen Au `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了一种基于 e‑process 授权的 Thompson 采样算法，在非平稳多臂赌博机中根据证据决定何时放弃完整历史模型，改用短期折扣后验进行决策。

**💡 创新点**

创新点在于将 e‑process 的阈值穿越作为正式授权机制，结合全历史 Beta 后验与折扣短期后验，加入相关性评分和决策权重，实现无固定时间窗口的自适应停留或偏离历史基准，并给出概率保证。

**🔧 技术方法**

采用 Beta–Bernoulli 先验、KL 效果检验、加权候选分割点、e‑process 与 Ville 不等式、Thompson 采样、指数加权平均、以及与 GLR 规则的对照等技术。

**📊 数据集**

实验使用 16,380 个非平稳环境（9 种机制、20 个 K‑T 组合）和 1,820 个平稳环境组成的注册套件；以及 4,500 次重放实验（606 条平均轨迹，9 个文献派生环境族）组成的重放套件。

**📈 对比分析**

与 open ramp（无授权版本）、Certified GLR、Practical GLR 及 16 种基线方法对比；在注册套件上相较 open ramp 降低 27.7% 的伪遗憾，在重放套件上略增 8.2%；整体在 17 个基线中排名第 10 左右。

**⚠️ 局限性**

局限性包括授权机制不可重启、缺乏对固定均值的严格保证、缺少非平稳环境下的遗憾理论、以及对短期记忆有效性和阈值选择的经验依赖。

---

## 734. Language Models that Play Chess and Explain Their Moves

**arXiv ID:** 2610.03695 | [PDF](https://arxiv.org/pdf/2610.03695v1)

**作者:** Adithya Bhaskar `[一作]` (Princeton University), Danqi Chen `[通讯]` (Princeton University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种4B参数的棋类语言模型，能够在顶尖棋手级别下棋并用自然语言解释其走法；

**💡 创新点**

创新点在于将无声专家棋类编码器与指令调优的语言模型通过交叉注意力耦合，再通过四阶段QA课程进行域适配，随后引入类似Bellman更新的迭代搜索-蒸馏算法提升解释质量；

**🔧 技术方法**

使用的技术包括Leela Chess Zero BT5编码器、SmolLM3解码器、Flamingo式交叉注意力桥、QA型域适配、基于Alpha‑Beta搜索的迭代蒸馏、以及对战和谜题评测；

**📊 数据集**

使用的数据集包括公开的Lichess棋局和谜题数据，采用Sol（GPT‑5.6‑Sol）生成的示例作为初始训练集，并结合Stockfish进行搜索与评估；

**📈 对比分析**

与GPT‑5.6‑Sol、Gemini‑3.1‑Pro等前沿语言模型相比，本模型在全局对局模拟中获得2697 Elo（比Sol高≈600点、比Gemini高≈450点），在谜题和通用位置上的无误差率（NMR/FNMR）也优于对手；

**⚠️ 局限性**

主要局限包括：在概念连贯性方面仍低于顶尖模型，可能出现符号和战术概念的幻觉；依赖昂贵的搜索蒸馏步骤，模型规模仍受限于4B参数，且对其他需要无声专家的领域的迁移需要进一步验证。

---

## 735. Less Decoder is More Encoder: Geometric Representation Learning from Novel View Synthesis

**arXiv ID:** 2610.03717 | [PDF](https://arxiv.org/pdf/2610.03717v1)

**作者:** Keerthi Kaashyap `[一作]` (Georgia Institute of Technology), Animesh Garg `[通讯]` (Georgia Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种自监督的多视角变换器框架，利用姿态无关编码器和姿态条件局部解码器实现可迁移的几何表征。

**💡 创新点**

通过限制解码器的感受野并使用冻结的ViT特征作为重建目标，解决了传统NVS方法中解码器过度表达与像素级目标导致表征衰退的问题。

**🔧 技术方法**

采用Transformer结构的姿态无关多视角编码器、RoPE2D位置编码、旋转对齐的交叉注意力、窗口掩码解码器以及在DINOv3特征空间上的L1+梯度重建损失。

**📊 数据集**

在RealEstate10K、DL3DV、Co3Dv2等带姿态标注的室内场景数据集上预训练，并在Hypersim、7Scenes、MimicGen等跨域数据上进行零样本评估。

**📈 对比分析**

与多种基线（LVSM、RayZer、VGGT、Muskie等）对比，在点对应、视觉定位、姿态估计、深度估计和机器人操控等五个任务中，取得与几何监督模型相当或更优的表现，尤其在视角变化下保持稳健。

**⚠️ 局限性**

受限于仅使用静态文本化场景和固定计算预算，尚未验证动态视频扩展及更大规模数据和算力下的扩展性。

---

## 736. LoGo: Local-Global Rewards for Consistent Long-Horizon Video Generation

**arXiv ID:** 2610.03636 | [PDF](https://arxiv.org/pdf/2610.03636v1)

**作者:** Ziqi Ma `[一作]` (California Institute of Technology), Gowthami Somepalli `[通讯]` (World Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种局部-全局奖励混合的后训练框架，并创建了 TrajectoryBench 评估集，用于提升摄像机控制视频模型在长视野下的 3D 一致性。

**💡 创新点**

核心创新在于：① 在 3D 空间内对每个体素计算重投影误差，实现细粒度信用分配；② 将局部奖励与全局奖励加权融合，避免单一奖励导致的质量下降；③ 采用奖励交织策略以兼顾视频质量和摄像机跟踪；④ 提供专门针对长视野复杂摄像机控制的新基准。

**🔧 技术方法**

技术手段包括：Voxel 化点云 + 重投影误差计算；VGGT 预测深度/RGB；DiffusionNFT/Flow-GRPO/DRaFT 等后训练方法；奖励交织（全局、局部、审美、摄像机）等。

**📊 数据集**

使用的数据集包括：DL3DV 的 100 场景 holdout；自研的 TrajectoryBench（2000 个样例，按难度与场景类型划分）；以及原始 WorldScore 用作基线对比。

**📈 对比分析**

在三款基线模型（Lingbot2、Lyra2、UniWorld）上与 VideoGPA、World-R1 等基线对比，评估指标为 RGBD 重投影 PSNR、MVCS、Gaussian 重建误差和 epipolar 误差。LoGo 在 TrajectoryBench 与 DL3DV 上的 PSNR 提升最多 2.7 dB，epipolar 误差降低 37%，并且在摄像机控制误差和视频质量上保持或提升，显著优于基线仅提升 0.1–0.2 dB 的效果。

**⚠️ 局限性**

局限性：① 对极长视野（>400 帧）仍存在挑战，可能需更先进的记忆设计；② 目前仅针对静态场景，动态场景的 4D 重建与奖励设计仍待进一步研究。

---

## 737. Separating QMA from QCIP with a Classical Oracle, or, the Power of Quantum Proofs over Classical Interaction for Quantum Verifiers

**arXiv ID:** 2610.03648 | [PDF](https://arxiv.org/pdf/2610.03648v1)

**作者:** Alper Cakan `[一作]` `[通讯]` (Carnegie Mellon University), Alper Cakan (Carnegie Mellon University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究者尝试通过新技术工具来区分 QMA 与 QCIP 两个量子复杂度类

**💡 创新点**

提出了将 QMA 约束映射到 QCIP 的新方法，构建了理论框架

**🔧 技术方法**

采用量子信息理论、复杂度分析与计算模型构建

**📊 数据集**

未使用具体实验数据集

**📈 对比分析**

通过理论证明与比较分析，证明两类之间存在严格包含关系，但具体性能指标未给出

**⚠️ 局限性**

主要局限在缺乏实验验证与细节实现，结论基于理论推导

---

## 738. Planning to Learn

**arXiv ID:** 2610.03667 | [PDF](https://arxiv.org/pdf/2610.03667v1)

**作者:** Ian Osband `[一作]` `[通讯]` (Google DeepMind), Ian Osband (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种新的训练损失——horizon loss，利用剩余训练预算来分配学习，改善分类器的期望准确率，尤其在存在标签噪声时更显优势。

**💡 创新点**

创新点在于将精确策略梯度（只考虑下一步收益）与交叉熵（假设永远继续学习）视为极限，并通过引入可变时间视野（horizon）在两者之间动态调度；证明在简化模型中可逃脱两端的陷阱，并在实测中显著提升准确率。

**🔧 技术方法**

采用分配模型分析、精确策略梯度推导、交叉熵的“耐心准确率”解释、horizon loss定义；在网络训练中使用Adam/AdamW；实验中对horizon loss、cross-entropy、exact policy gradient及不同 horizon 方案进行对比。

**📊 数据集**

使用MNIST和ImageNet（ResNet‑50/ResNet‑101/ViT‑S/16）作为主实验数据集，并在ImageNet上引入不同比例的标签噪声进行额外验证。

**📈 对比分析**

在相同网络结构、相同学习率、相同训练步骤下比较期望准确率和top‑1准确率。horizon loss 在 flat learning rate 上对ImageNet提升 1.0–2.4 点 top‑1，提升 4–5 点期望准确率；对MNIST提升约 0.16 点；在标签噪声实验中提升随噪声从 3.3 点提升至 7.3 点。相较于 cross‑entropy 及 fixed/growing horizon，horizon loss 在训练后期展现更明显优势；cosine decay 使增益略减。

**⚠️ 局限性**

局限性包括：需要手动校准 κ 以估计剩余学习进度；基于独立样本的分配模型无法完整捕获参数共享与网络动态；horizon loss 是可分离的，无法处理样本间耦合；未处理 RL 中常见的探索、信用分配及采样噪声等问题；实验仅限于图像分类任务，缺乏在其他任务或更大规模模型上的验证。

---

## 739. EyeRobot 2.0: Active Gaze for Precise Manipulation without Wrist Cameras

**arXiv ID:** 2610.03710 | [PDF](https://arxiv.org/pdf/2610.03710v1)

**作者:** Kush Hari `[一作]` (University Of California Berkeley), Angjoo Kanazawa `[通讯]` (University Of California Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本研究提出了一种“主动视觉固定”（Active Visual Fixation，AVF）框架，利用单一固定立体摄像头通过协调双眼视角实现精细双臂操作，完成如吸管插入、标记帽子盖住等高精度任务。

**💡 创新点**

创新点包括：① 层次化的视线控制——低层固定点策略与高层目标选择器共同决定何时、何处聚焦；② 通过多分辨率“视网膜”裁剪实现对关注区域的高分辨率采样；③ 在固定点参考坐标系下规范化SE(3)抓取动作，显著压缩动作空间；④ 结合强化学习与行为克隆的闭环训练，实现无需人工注视数据的自我学习。

**🔧 技术方法**

技术手段：强化学习（PPO）训练目标选择器和固定点策略；基于语义分割（SAM3）与立体深度（FoundationStereo）获得3D目标；多尺度裁剪+Transformer解码器进行视觉特征提取；动作片段在固定点坐标系中表示与行为克隆；使用立体相机实现视线合成和视角仿真。

**📊 数据集**

使用自采集的7个真实世界和6个仿真任务的遥操作演示数据，覆盖多阶段、精细对齐、长时序等场景；每个任务收集10-53分钟的数据，约1000+物理实验和1800+仿真实验。

**📈 对比分析**

与传统无腕摄像头的固定立体观测和普遍使用的“腕+头”摄像头对比。AVF在真实场景中比静态立体提升约40%成功率（对比20%仿真），在腕摄像头被遮挡时仍保持高性能，整体匹配或超过腕摄像头方案；当腕摄像头清晰可见时差距仅约5%。

**⚠️ 局限性**

局限性：仅训练任务特定的视线与抓取策略，难以迁移到多任务；未控制机器人头部/颈部运动，假设固定头位；缺乏对子物体级别的细粒度注视；在极端遮挡或光照不佳的环境下仍可能表现不佳。

---

## 740. On the Convergence of Success Conditioning for Policy Optimization

**arXiv ID:** 2610.03642 | [PDF](https://arxiv.org/pdf/2610.03642v1)

**作者:** Matthew Brun `[一作]` (Massachusetts Institute of Technology), Xu Andy Sun `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究并证明了“成功条件化（SC）”策略在马尔可夫决策过程（MDP）中的收敛性，给出了单期和折扣型MDP的理论收敛速度；

**💡 创新点**

首次将SC视为迭代优化方法，给出其收敛到受限最优策略的证明，并推导出单期MDP的O(log(1/ε))和折扣MDP的O(1/ε^p)收敛上界；

**🔧 技术方法**

理论分析方法，包括价值函数递增性、闭式表达、算术-几何均值不等式、近似值迭代（AVI）框架以及误差递归分析；

**📊 数据集**

无实验数据集，全文以理论证明为主；

**📈 对比分析**

与传统价值迭代、策略迭代等方法做理论对比，指出SC收敛速度较慢（单期O(log(1/ε)) vs. 线性/超线性收敛），但保留了随机策略与保守更新的优势；

**⚠️ 局限性**

局限性在于收敛速度慢、依赖初始策略支持动作集、仅能收敛到受限最优解，且缺乏实验验证与样本复杂度分析。

---

## 741. Single-Sample Prophet Inequalities: A Combinatorial to Single-Item Reduction

**arXiv ID:** 2610.03660 | [PDF](https://arxiv.org/pdf/2610.03660v1)

**作者:** Shuchi Chawla `[一作]` (University of Texas at Austin), Trung Dang `[通讯]` (University of Texas at Austin)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种通用的从单样本组合预测不等式到单物品预测不等式的归约框架，能够将多维分配问题拆分为买家侧的自由处置问题和物品侧的单物品预测问题；

**💡 创新点**

创新点在于引入支持价格与自由处置价值的概念，将买家多维需求线性化，构建买家局部决策机制，并利用随机顺序自由处置算法与单样品单物品预测算法的组合，实现了比以往更高的竞争比；

**🔧 技术方法**

使用了支持价格（XOS/capped‑XOS）与单物品预测不等式的阈值规则、随机顺序贪心算法、层析（layer‑cake）分析以及Game of Googol模型的样本对称性；

**📊 数据集**

未使用任何外部数据集，全部在理论模型下分析；

**📈 对比分析**

与之前的1/576或1/(6H_m)等常数相比，本文在XOS和capped‑XOS下分别实现了约1/10.4的竞争比，并证明了单样本半比例的最优性；

**⚠️ 局限性**

局限在于仅适用于具有支持价格的XOS/capped‑XOS类，依赖于对支持价格的oracle查询，且对一般子加或更复杂价值函数的推广尚未完成。

---

## 742. Broken scale symmetries in undercomplete linear autoencoders

**arXiv ID:** 2610.03640 | [PDF](https://arxiv.org/pdf/2610.03640v1)

**作者:** Farhad Pashakhanloo `[一作]` (Harvard University), Jacob A. Zavatone-Veth `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在这篇论文中，作者对欠完备线性自编码器（undercomplete linear autoencoders）进行了理论与数值研究，探讨了梯度下降在尺度对称性（scale symmetry）方向上的动态行为；

**💡 创新点**

创新点在于揭示了SGD在尺度对称性方向上会出现定向漂移（directed scale drift），并构建了可解析的有效动力学模型，同时证明漂移终止于“稳定边缘”（edge of stability）而非无限增长；

**🔧 技术方法**

主要技术包括梯度下降的噪声分析、尺度对称性的Noether电荷、时间尺度分离、Lambert W函数求解、Hessian曲率分析以及对非线性ReLU自编码器的初步扩展；

**📊 数据集**

实验使用的是合成高斯数据集，设定协方差矩阵具有严格的谱间隙（k阶主子空间与其余方向分离）；

**📈 对比分析**

作者将理论预测与数值模拟进行对比，结果显示有效动力学模型能准确预测不同尺度方向的增长速率，并且在达到稳定边缘后所有尺度趋于相同；在“尖锐度”指标上，尽管总体Hessian最大特征值升高，但其他锐度度量（如批量尖锐度）却随尺度增大而降低；

**⚠️ 局限性**

主要局限包括：仅考虑线性或同质非线性自编码器的欠完备设置，缺乏对更复杂网络结构或完整自编码器的推广；实验仅基于合成数据，缺乏对真实数据集的验证；

---

## 743. MoSE3: Learning World-Space SE(3) at Every Pixel

**arXiv ID:** 2610.03716 | [PDF](https://arxiv.org/pdf/2610.03716v1)

**作者:** Jiahuan Cheng `[一作]` (Harvard University), Qianqian Wang `[通讯]` (Harvard University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一个前向网络，能够从单目RGB视频中直接预测每个像素的全局 6-DoF SE(3) 运动，既包含旋转也包含平移，并通过像素级刚体分组表达部件运动。

**💡 创新点**

核心创新在于将运动预测拆解为两步：先回归稠密 3D 点轨迹和刚体嵌入，然后通过可微分的加权 Horn 拟合闭式求解每像素运动；这一分解使得难以直接回归的旋转与刚体分组可以利用更丰富的监督信号并实现端到端学习。

**🔧 技术方法**

使用了 Transformer 结构的跟踪分支（在已有几何基线的冻结权重上添加可训练分支），点轨迹头、刚体嵌入头以及可微分的 Horn 拟合；同时采用了多任务损失，包括轨迹、可见性、嵌入一致性和直接运动监督。

**📊 数据集**

主要数据集为新构建的 Art-Kubric（大规模合成多物体可动场景，提供 3D 轨迹、刚体标签、像素级分割），以及 HO3D、iTACO、YCBInEOAT 等公开数据集用于评估。

**📈 对比分析**

与现有基线（如基于 Horn 拟合的点轨迹聚类、RAFT-3D、ProxyPose 等）进行对比。结果显示在单像素、部件和物体层面均实现了领先性能，并在 TAPVid-3D 的 PointOdyssey、ADT、PStudio 上达到了最佳平均跟踪精度，几乎超越所有对比方法。

**⚠️ 局限性**

局限性包括：高度依赖外部几何基线的相机参数与点图估计，若这些输入误差大则运动估计会受影响；在快速运动场景下轨迹与嵌入的误差会显著放大；目前对柔性物体的量化评估尚缺失，只在视觉上给出了定性结果。

---

## 744. Simulation-Free Learning of Population Dynamics with Wasserstein Lagrangian Residuals

**arXiv ID:** 2610.03679 | [PDF](https://arxiv.org/pdf/2610.03679v1)

**作者:** Fedor Sergeev `[一作]` (Basis Research Institute), Eli Bingham `[通讯]` (Basis Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无模拟的残差驱动方法 Double-Stitch，用于学习概率分布随时间演化的 Lagrangian 动力学。

**💡 创新点**

创新点在于：1）通过 Clebsch 变分原理推导出不要求速度为梯度的 Wasserstein Lagrangian 运动方程；2）将该运动方程的残差直接作为损失，实现训练时不需要数值模拟；3）兼顾梯度、守恒与周期性动力学。

**🔧 技术方法**

使用的技术包括：Kernel Density Estimation (KDE) 参数化分布和速度场；神经网络表示外部势能和粒子相互作用；有限差分近似加速度；残差损失与数据拟合损失联合优化；对初始速度做柔性约束。

**📊 数据集**

数据集涵盖：三种合成高斯混合动力学（保守、阻尼、梯度流）；人类胚胎干细胞单细胞 RNA 测序数据（EB）；墨西哥湾海洋涡旋数据（小涡旋插值与大涡旋预测）。

**📈 对比分析**

与梯度流方法 Stitching、流动匹配方法、以及基于模拟的 Wasserstein Lagrangian 方法 WLM 进行比较；在合成与单细胞数据上 Double-Stitch 与 Stitching 相当或更优，在海洋涡旋插值任务上与 WLM 相当，训练速度比 WLM 快 4–14 倍；在海洋涡旋预测任务中仍略逊于 WLM。

**⚠️ 局限性**

局限性：作为残差驱动方法，无法保证完全满足假设动力学；优化过程对初始化敏感，残差与数据损失可能竞争，导致收敛困难；缺乏严格的收敛理论与对更复杂动力学（如主动物质、粘性流）的适用性。

---

## 745. IDRF: Inverse-Distilled Reward Fine-tuning of Masked Discrete Diffusion Models

**arXiv ID:** 2610.03641 | [PDF](https://arxiv.org/pdf/2610.03641v1)

**作者:** Vladislav Gromadskii `[一作]` (Applied AI Institute), Alexander Korotin `[通讯]` (Applied AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种通过逆蒸馏正则化的奖励微调框架 IDRF，用于少步掩码扩散模型的训练。

**💡 创新点**

创新点在于用逆蒸馏闭合的序列级 KL 上界来代替不可计量的全序列似然，使得学生模型可以在自己的采样路径上直接进行奖励微调，避免奖励劫持并显著减少采样步骤。

**🔧 技术方法**

核心技术包括掩码扩散模型、逆蒸馏（inverse distillation）、马尔可夫决策过程（MDP）下的轨迹策略梯度、以及与奖励的结合。

**📊 数据集**

在三个任务上验证：DNA 活性设计（200 bp 监管序列），CLIP 引导的 ImageNet 图像生成，亚马逊时尚评论的情感 steering。

**📈 对比分析**

与多种基线（如 DRAKES、SDPO、SEPO、ReDGE、RLOO 等）比较，IDRF 在仅 8–32 步采样的情况下实现与 128 步参考相当甚至更好的奖励，同时保持或提升样本质量，且显著减少奖励劫持现象。

**⚠️ 局限性**

局限性包括：训练过程中存在对抗式极小极大更新，易导致不稳定；逆蒸馏需要额外的辅助模型，增加计算成本；对最终采样分布的控制尚未完全保证，且在更大规模模型上的可扩展性待验证。

---

## 746. FrugalEvo: Towards Cost-Aware LLM-Guided Program Evolution

**arXiv ID:** 2610.03675 | [PDF](https://arxiv.org/pdf/2610.03675v1)

**作者:** Hui Chen `[一作]` (National University Of Singapore), Bryan Hooi `[通讯]` (National University Of Singapore)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种成本感知的LLM指导进化框架 FrugalEvo，通过高成本 LLM 探索设计策略、低成本 LLM 实现并细化代码，并采用缓存高效提示实现固定成本预算下的程序优化。

**💡 创新点**

创新点包括：①引入预算感知 AUC（BA‑AUC）衡量成本效率；②将候选程序生成拆分为探索与实现两阶段，并使用不同成本的 LLM；③设计共享前缀的提示以最大化缓存复用；④在多任务上实现比现有基线更优的性能。

**🔧 技术方法**

使用技术包括 LLM 生成、进化搜索、岛式 MAP‑Elites 存储、提示缓存、策略探索与实现分离、预算感知评估指标等。

**📊 数据集**

实验数据集包含 20 个真实世界优化任务：5 个数学任务、5 个系统任务（ADRS 基准）和 10 个算法任务（ALE‑Bench‑Lite）。

**📈 对比分析**

与 OpenEvolve、ShinkaEvolve、AdaEvolve、EvoX 以及 AlphaEvolve 等基线在相同成本预算下对比；FrugalEvo 在大多数任务上获得最高最终分数，并在 9/10 数学与系统任务上取得最高 BA‑AUC，算法任务的平均表现也最高。

**⚠️ 局限性**

局限性：仅适用于可通过代码自动评估的任务；策略探索依赖手工编写的提示，缺乏自动生成机制；不易推广到需要物理实验或手工验证的科学领域。

---

## 747. Do Large Language Models Know Colombian Law? A Reliability Benchmark for the Colombian Legal System

**arXiv ID:** 2610.03639 | [PDF](https://arxiv.org/pdf/2610.03639v1)

**作者:** Rubén Manrique `[一作]` (Universidad de los Andes), Joaquín Vélez Navarro `[通讯]` (Universidad de los Andes)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个针对哥伦比亚法律体系的专家验证基准，并对15款大型语言模型在闭合选择、半开放和开放式（IRAC）三种题型上的可靠性进行了评估。

**💡 创新点**

创新点在于：①提供了多格式（闭合、多选、半开放、开放式）且覆盖十个法律领域的1152条专家审核题库；②采用人机协作平台实现多阶段审核流程；③通过RAGAS等自动化评判与LLM‑judge对答案进行双重验证。

**🔧 技术方法**

技术手段包括：多阶段人机审核流水线、基于LLM的自动评判（RAGAS、BERTScore、BLEU、ROUGE）、LLM‑judge判分器、Item Response Theory（IRT）模型、聚类与统计检验（McNemar、Friedman、Wilcoxon）。

**📊 数据集**

数据集：1042道题目，涵盖宪法、行政、刑事、劳动、民事、商业与公司、程序、税收、家庭、市场法等十个领域，按难度划分为低、中、高三层，三种题型分别为305道闭合、多选题、682道半开放题、55道开放式IRAC题。

**📈 对比分析**

比较方法：对每种题型使用相应指标（闭合题准确率、半开放/开放式答案的RAGAS正确率等），模型按准确率/正确率分层。性能表现为：闭合题最高准确率0.905（Gemini 3.1 Pro），最低0.577（Command R）；开放式答案最高事实正确率0.451（GPT‑5.4），最低约0.31；模型排名在闭合题和开放式答案高度相关（ρ≈0.94）。

**⚠️ 局限性**

局限性：①评估主要依赖自动化指标，缺乏完整人类专家打分覆盖；②开放式题目样本不足（55道）；③未采用检索增强或工具调用，导致外部文档/最新案例难以回答；④对模型随机性、提示词多样性和温度设定的鲁棒性未作系统测试；⑤缺少正式的双重标注一致性评估。

---

## 748. Pivot-SD: Efficient Self-Distillation for Masked Diffusion Language Models

**arXiv ID:** 2610.03665 | [PDF](https://arxiv.org/pdf/2610.03665v1)

**作者:** Seo Hyun Kim `[一作]` (KAIST AI), Rahul G. Krishnan `[通讯]` (University of Toronto & Vector Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Pivot-SD，一种针对掩码扩散语言模型的离线自蒸馏方法；

**💡 创新点**

通过信息增益识别“枢轴”决策点，只对这些关键步骤进行正负训练，从而实现高效的信用分配；

**🔧 技术方法**

使用掩码扩散模型、信息增益度量、交叉熵与不相似性(unlikelihood)损失、离线轨迹采样与重放；

**📊 数据集**

在数学推理数据集（MATH、GSM8K）和代码推理数据集（HumanEval+、MBPP+）上训练和评估；

**📈 对比分析**

与传统全序列SFT、基于验证器的SFT以及在线RL（diffu-GRPO、wd1++）对比，Pivot-SD在相同的200题预算下，平均准确率提升约2–5个百分点，同时训练时间比5,000步RL低约8倍；

**⚠️ 局限性**

局限包括需手工调参的超参数、仅在单个扩散块内做信用分配、无法跨块延伸信用，且实验仅覆盖两种规模相当的扩散模型。

---

## 749. RNADyn: A Benchmark for Generating and Understanding RNA Dynamics

**arXiv ID:** 2610.03712 | [PDF](https://arxiv.org/pdf/2610.03712v1)

**作者:** Yiming Huang `[一作]` (Imperial College London), Tolga Birdal `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 RNADynBench 数据集和 RNADynNet 模型，用于 RNA 动力学轨迹生成与单构象动态表征。

**💡 创新点**

将生成与预测任务统一到同一框架，加入物理定向（PG）与轨迹对齐机制，并提供规模化、标准化的 RNA MD 轨迹。

**🔧 技术方法**

基于扩散时空编码器、坐标去噪、轨迹与单帧对齐以及物理监督（残基协方差、运动耦合）实现。

**📊 数据集**

使用 2,585 条 100 ns 全原子 RNA 轨迹（共 2.585 μs，覆盖 1,469 种 RNA），并划分 leakage‑controlled train/val/test/transfer 集。

**📈 对比分析**

与 ConfRover、MDGen、BioKinema 等基线对比，RNADynNet 在轨迹生成上 RMSF 相关性达到 0.875/0.766，单构象预测 RMSF 相关性 0.867/0.783，生成速度更快、几何质量更优。

**⚠️ 局限性**

轨迹仅覆盖 100 ns，无法捕捉慢速构象；仅考虑单一 RNA，未包含 RNA–蛋白或小分子复合体；依赖大量 GPU 资源，训练成本高。

---

