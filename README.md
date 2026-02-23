# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-02-23 | 今日论文总数: 336

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Assigning Confidence: K-partition Ensembles

**arXiv ID:** 2602.18435 | [PDF](https://arxiv.org/pdf/2602.18435v1)

**作者:** Aggelos Semoglou `[一作]` (Athens University of Economics and Business), John Pavlopoulos `[通讯]` (Athens University of Economics and Business)

**通讯引用:** 3510 | [OpenAlex ID](https://openalex.org/A5033894687)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于聚类集成的实例级置信度评估框架CAKE，用于量化每个样本在聚类中的可靠性。

**💡 创新点**

创新点在于同时结合跨轮聚类结果的分配稳定性和局部几何拟合度，形成一个介于0到1的可解释置信分数，并给出了理论误差和噪声下的性能保障。

**🔧 技术方法**

技术包括多次随机初始化的k‑means（或其他聚类算法）生成集成，使用匈牙利算法对标签进行对齐，计算Silhouette统计量的均值与标准差，以及基于成对一致性的U‑统计量来衡量稳定性，最终采用乘积或调和平均法融合两种分量。

**📊 数据集**

实验在七个合成数据集和八个公开真实数据集（Iris、Breast Cancer、Pendigits、Letter、Digits、Fashion‑MNIST、Satimage、20Newsgroups）上进行，覆盖不同维度、类别数和数据结构。

**📈 对比分析**

与传统的内部验证指标、熵一致性、Bootstrap稳定性等基线相比，CAKE在大多数数据集上在ARI、AMI、ACC以及错误发现任务中均取得更高或相近的性能，且在不同模型（k‑means、GMM、谱聚类）和缺失k值情形下也保持鲁棒。

**⚠️ 局限性**

局限性包括对聚类算法的依赖（需多次运行相同或同类算法以构建集成），对高维稀疏数据中Silhouette近似的准确性可能受限，以及在极大规模数据时仍需较多计算资源，尽管已使用基于质心的近似来降低成本。

---

## 2. PRISM-FCP: Byzantine-Resilient Federated Conformal Prediction via Partial Sharing

**arXiv ID:** 2602.18396 | [PDF](https://arxiv.org/pdf/2602.18396v1)

**作者:** Ehsan Lari `[一作]` (Norwegian University of Science and Technology), Stefan Werner `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 8014 | [OpenAlex ID](https://openalex.org/A5059938646)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 PRISM-FCP 框架，通过在联邦学习中使用部分参数共享来实现跨训练和校准阶段的拜占庭鲁棒联邦合适预测。

**💡 创新点**

将部分共享（通信压缩）与拜占庭鲁棒聚合相结合，既抑制训练阶段攻击，又提升校准阶段的异常检测，实现端到端鲁棒性。

**🔧 技术方法**

采用部分共享 PSO‑Fed、Conformal Prediction、距离基恶意分数检测、分位数估计以及安全聚合技术。

**📊 数据集**

在合成数据和 UCI 超导性数据集上进行实验。

**📈 对比分析**

与全共享的 FCP 和 Rob‑FCP 对比，PRISM‑FCP 在拜占庭攻击下保持 90% 覆盖率，区间宽度显著缩小（比 Rob‑FCP 约 1.8 倍），同时通信量降低约 70%。

**⚠️ 局限性**

仅针对线性回归模型给出理论证明，对非线性模型和自适应攻击的鲁棒性尚未充分分析。

---

## 3. Detecting PowerShell-based Fileless Cryptojacking Attacks Using Machine Learning

**arXiv ID:** 2602.18285 | [PDF](https://arxiv.org/pdf/2602.18285v1)

**作者:** Said Varlioglu `[一作]` (University of Cincinnati), John M. Emmert `[通讯]` (University of Cincinnati)

**通讯引用:** 854 | [OpenAlex ID](https://openalex.org/A5031863054)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建包含500条PowerShell文件无加密挖矿脚本的专用数据集，结合AST（抽象语法树）和机器学习模型，对文件无Cryptojacking脚本进行检测与分类。

**💡 创新点**

创新点在于将AST集成到预训练的CodeBERT模型中进行微调，既不需要手工特征提取，也不依赖繁琐的混淆/解混淆流程，却能实现高召回率（96.6%）且提升模型效率。

**🔧 技术方法**

使用的技术包括：PowerShell内置AST解析器、词/字符级别分词、LSTM、BiLSTM、Transformer（CodeBERT）以及Fine‑Tuning、AdamW优化器、K‑fold交叉验证等。

**📊 数据集**

使用的数据集为：① 500条来自Purple Fox、Lemon Duck、Tor2Mine的恶意文件无脚本；② 另一个公开数据集 1,770 条恶意脚本和 4,819 条正样本，合计约 6,589 条脚本。

**📈 对比分析**

实验中将AST与非AST版本的 LSTM、BiLSTM 和 CodeBERT 进行对比。AST‑based CodeBERT 的准确率 96.2%、召回率 96.6%、F1 分数 96.3%，在保持高召回率的同时相较于传统特征提取方法实现了更高的计算效率。

**⚠️ 局限性**

主要限制包括：收集恶意脚本的难度导致样本数量有限，缺乏更丰富的变体；实验未涵盖混淆/解混淆过程，可能影响在更复杂威胁场景中的泛化能力。

---

## 4. Computer Vision in Tactical AI Art

**arXiv ID:** 2602.18189 | [PDF](https://arxiv.org/pdf/2602.18189v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 5. Influence-Preserving Proxies for Gradient-Based Data Selection in LLM Fine-tuning

**arXiv ID:** 2602.17835 | [PDF](https://arxiv.org/pdf/2602.17835v1)

**作者:** Sirui Chen `[一作]` (University of Illinois Urbana-Champaign), Jingrui He `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 4199 | [OpenAlex ID](https://openalex.org/A5073158087)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种两阶段框架（IProX），用于从目标大模型直接构建影响力保留的小型代理，以实现高效的梯度基数据选择，随后在大模型上进行微调。

**💡 创新点**

创新点在于：①提出了影响力保持的低秩分解（IPSVD），通过自适应加权的SVD保留梯度影响相关的矩阵分量；②引入梯度对齐与输出对齐的结合，使代理在保持影响力的同时保持与目标模型的梯度和输出一致；③通过可调节的稀疏比例实现代理规模与计算成本的灵活控制。

**🔧 技术方法**

使用技术包括：梯度相关的影响力度量（TracIn、影响函数）、低秩矩阵分解（IPSVD）、自适应矩阵重加权、梯度对齐损失、KL蒸馏对齐、Probe集收集、稀疏矩阵求逆与平方根的“skinny SVD”加速实现。

**📊 数据集**

主要使用了 Dolly 作为候选训练数据集，并在 MMLU、BBH、TyDiQA 三个下游评估任务上验证模型性能；在实验中对 Llama3.2‑3B、Gemma3‑4B、Qwen3‑4B、Qwen2‑7B 四大模型家族进行了评估。

**📈 对比分析**

与同族的预置代理（如 1B、1.7B）以及两种基线（Layer Extraction、Influence Scorer）进行对比。IProX 在所有模型和任务上均优于预置代理，平均提升 1.5–3%（在 Qwen3‑4B 的 1.5B 代理甚至超过 1.7B 代理）。同时在计算量上，代理压缩可减少至 50% 以上的 FLOPs，影响力计算时间从 90 分钟降至 40–45 分钟，整体耗时约 43–51 分钟。

**⚠️ 局限性**

局限性包括：①代理压缩到过低的秩（<10%）会导致性能急剧下降；②嵌入层和 LM 头无法压缩，限制了进一步的规模减小；③对 Probe 集的选择和多样性敏感，过小或过少多样性会影响影响力保留效果；④对齐阶段增加了训练成本和实现复杂度；⑤在极端稀疏情况下，梯度对齐仍可能不足，导致对某些任务的性能下滑。

---

## 6. Diff2DGS: Reliable Reconstruction of Occluded Surgical Scenes via 2D Gaussian Splatting

**arXiv ID:** 2602.18314 | [PDF](https://arxiv.org/pdf/2602.18314v1)

**作者:** Tianyi Song `[一作]`, Francisco Vasconcelos `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种两阶段框架 Diff2DGS，通过扩散模型对手术器械遮挡区域进行时间一致性重建，再利用可学习的 2D 高斯展开与可变形模型实现高精度实时的可变形手术场景三维重建。

**💡 创新点**

创新点包括：①将扩散模型与时序注意力相结合的手术视频重建模块；②引入可学习变形模型 (LDM) 以在 2D 高斯分布中捕捉组织变形；③自适应深度损失权重策略，动态平衡图像质量与几何精度。

**🔧 技术方法**

使用的技术包括：稳定扩散 (Stable Diffusion v1.5) + Phased Consistency Model、2D 高斯展开 (2D Gaussian Splatting)、可学习变形模型、深度自适应损失、RAFT 作为深度参考。

**📊 数据集**

在 EndoNeRF、StereoMIS、SCARED 三个公开数据集上进行评估，并对比 RAFT 及其他几何基准。

**📈 对比分析**

与现有方法相比，Diff2DGS 在 EndoNeRF 上 PSNR 38.02 dB、StereoMIS 上 PSNR 34.72 dB，遮挡区域的 PSNR 和 RMSE 明显优于 Deform3DGS、EndoGaussian、SurgicalGS；同时保持与 3DGS 相近的渲染速度，显著提升遮挡区域与几何精度。

**⚠️ 局限性**

局限性在于目前主要针对相对静止摄像机视角，未充分考虑显著相机运动的场景，未来需加入摄像机运动建模以增强鲁棒性。

---

## 7. Efficient Filtered-ANN via Learning-based Query Planning

**arXiv ID:** 2602.17914 | [PDF](https://arxiv.org/pdf/2602.17914v1)

**作者:** Zhuocheng Gan `[一作]` (University of Hawaii Manoa), Yifan Wang `[通讯]` (University of Hawaii Manoa)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于学习的查询规划框架，动态在预过滤与后过滤之间选择，以高效执行过滤的近似最近邻（ANN）查询。

**💡 创新点**

创新点在于通过轻量级的选择率估计与两层MLP决策模型，实现对每条查询的自适应执行策略，显著提升了不同数据集与过滤负载下的性能。

**🔧 技术方法**

使用了频率表、直方图、梯度提升回归模型进行选择率估计，结合两层MLP分类器进行策略决策，并与预过滤、后过滤及ACORN‑1基线对比；实现了无GPU的轻量化部署。

**📊 数据集**

实验数据集包括四个真实/合成集合：ArXiv（2.14M 384‑维）、Wolt（1.72M 512‑维）、GloVe‑200（1.18M 200‑维）和SIFT（1M 128‑维），每个集合都包含向量、元数据与过滤查询。

**📈 对比分析**

与预过滤、后过滤和ACORN‑1相比，学习规划在多数场景下达成≥90%召回率且加速达4×，在ArXiv上相较ACORN‑1加速4×、在SIFT上表现最优；在高选择率或单一最优策略场景下提升有限。

**⚠️ 局限性**

局限性包括需要离线训练模型、对极低选择率时可能误判、对极其适合单一策略的工作负载提升有限，以及对更复杂过滤条件（多属性多范围）的支持尚未完善。

---

## 8. Grassmannian Mixture-of-Experts: Concentration-Controlled Routing on Subspace Manifolds

**arXiv ID:** 2602.17798 | [PDF](https://arxiv.org/pdf/2602.17798v1)

**作者:** Ibne Farabi Shihab `[一作]` (Iowa State University), Anuj Sharma `[通讯]` (Iowa State University)

**通讯引用:** 2985 | [OpenAlex ID](https://openalex.org/A5083087081)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于Grassmann流形的Mixture-of-Experts路由框架GrMoE，利用Matrix Bingham分布的浓度参数实现可控稀疏性，并通过变分推断得到不确定感知的专家分配。

**💡 创新点**

创新点包括：1) 将专家视为子空间，用Grassmann子空间来做分类；2) 用Bingham浓度矩阵Λ作为单一连续可调节的稀疏度控制阈值；3) 给出浓度谱与路由熵、top‑k质量、专家崩溃之间的严格理论界限；4) 设计稀疏正则化和自回归变分推断实现不确定性路由；5) 实现无需重训练即可按需调整稀疏度的后期调优。

**🔧 技术方法**

技术包括Grassmann子空间表示、Matrix Bingham分布、Saddle‑point 归一化近似、Riemannian Adam优化子空间参数、采样对齐正则化、两层MLP自回归变分网络、稀疏路由的可控参数α。

**📊 数据集**

使用合成路由任务数据（子空间混合噪声生成）进行控制实验；在语言建模任务上使用OpenWebText数据集训练350M、1.3B和2.7B规模的MoE模型。

**📈 对比分析**

与Softmax Top‑k、Switch、Expert Choice、Hash、Soft MoE和vMF‑Gate等现有路由方法在相同规模模型上对比。GrMoE在所有规模下实现0%专家崩溃、显著改善负载均衡（CV下降）、略优的困惑度（PPL 下降0.2–0.4点），并在α调节下实现平滑的稀疏度–性能曲线，避免了需要重新训练的Top‑k模型。

**⚠️ 局限性**

局限性：1) 仅在至1.3B规模下验证，7B+规模及64+专家的可扩展性仍待探索；2) Bingham归一化常数需近似，极端浓度下误差增大；3) 路由秩k_r为新超参数，需经验调优；4) 自回归变分网络仅带来轻微提升，是否能在更大规模显著提高尚未验证；5) 训练中α固定为1，未尝试自适应或逐步提升策略。

---

## 9. Zero-shot Interactive Perception

**arXiv ID:** 2602.18374 | [PDF](https://arxiv.org/pdf/2602.18374v1)

**作者:** Venkatesh Sripada `[一作]` (University of Surrey), Amir Ghalamzan `[通讯]` (University of Sheffield)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个零样本交互感知框架ZS-IP，能够通过视觉语言模型引导机器人进行推、抓、提等多策略操作，解决部分可见场景下的遮挡与模糊查询。

**💡 创新点**

创新点在于引入“pushlines”二维视觉增强、记忆驱动的动作决策以及与VLM闭环的多策略交互，能够在不依赖训练的前提下实现零样本推理与操作。

**🔧 技术方法**

使用了视觉语言模型（GPT‑4o）、Grounded SAM分割、PCA生成推线、ArUco 2D网格投影、OMPL/Pilz运动规划以及记忆检索机制。

**📊 数据集**

实验数据集采用YCB物体集合，并通过Frankia Panda 7‑DOF机械臂和RealSense RGB‑D相机在八个遮挡/复杂度不同的交互感知任务中收集。

**📈 对比分析**

与MOKA等基线对比，ZS‑IP在八个任务中的成功率平均提升至约80%，在推送任务上显著优于基线；同时在轨迹长度、位置误差和Oracle成功率等指标上也表现更优。

**⚠️ 局限性**

局限性包括低分辨率深度信息导致精细操作受限、VLM推理的幻觉与常识不足、动作表示受限于SE(3)的部分维度，以及缺乏触觉/本体反馈等多模态感知。

---

## 10. Two Calm Ends and the Wild Middle: A Geometric Picture of Memorization in Diffusion Models

**arXiv ID:** 2602.17846 | [PDF](https://arxiv.org/pdf/2602.17846v1)

**作者:** Nick Dodson `[一作]` (University of Missouri), Zhengchao Wan `[通讯]` (University of Missouri)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究扩散模型在不同噪声尺度下的记忆化行为，提出基于后验权重浓度和高斯壳层覆盖的几何框架，并发现存在一个高风险的中间噪声区；

**💡 创新点**

创新点在于用几何视角解释记忆化与泛化的分离，识别出危险区并提出针对该区的欠训练（gap training）策略以显著降低记忆化；

**🔧 技术方法**

采用后验权重分析、Gaussian shell覆盖理论、对中间噪声区的实验交换与欠训练，以及基于EDM的训练框架；

**📊 数据集**

主要数据集为CIFAR‑10（2k车子子集）和CelebA（1024灰度图像），并在这些数据上进行实验；

**📈 对比分析**

与标准EDM基线、随机噪声对照（Dummy）以及gap训练三种方法对比，gap训练将记忆化率从78.8%降至0.7%，FID从3.72提升至2.35；

**⚠️ 局限性**

局限性包括实验仅针对小规模数据集，理论分析对高维实测情况的可推广性尚未充分验证，且缺乏对更复杂分布的深入探讨。

---

## 11. SuiteEval: Simplifying Retrieval Benchmarks

**arXiv ID:** 2602.18107 | [PDF](https://arxiv.org/pdf/2602.18107v1)

**作者:** Andrew Parry `[一作]` (University of Glasgow), Sean MacAvaney `[通讯]` (University of Glasgow)

**通讯引用:** 1687 | [OpenAlex ID](https://openalex.org/A5014199889)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SuiteEval框架，统一实现跨benchmark套件的端到端评估，自动化数据加载、索引、排名、指标计算及结果聚合。

**💡 创新点**

创新点在于动态索引复用减少磁盘占用、通过单一pipeline生成器统一配置、支持一行代码注册新套件，并标准化指标与聚合规则。

**🔧 技术方法**

使用Python管道（PyTerrier生态）结合PisaIndex、T5重排器、Electra评分器等，采用ir_measures实现官方评估指标，并通过Workspace管理临时文件。

**📊 数据集**

覆盖BEIR、LoTTE、MS MARCO、NanoBEIR、BRIGHT等公开套件，并演示如何自定义MS MARCO passage套件；示例实验在BEIR上进行。

**📈 对比分析**

通过统一的度量（如nDCG@10）和几何平均聚合进行比较，支持基线系统进行显著性检验；示例实验显示SuiteEval在磁盘使用上可减少数十倍，但论文未给出模型性能提升。

**⚠️ 局限性**

局限性包括：依赖PyTerrier等现有工具，尚未覆盖所有索引器或GPU加速方案；自定义指标和管道需要手工实现；注册新套件需代码修改，无法完全零代码配置。

---

## 12. BLM-Guard: Explainable Multimodal Ad Moderation with Chain-of-Thought and Policy-Aligned Rewards

**arXiv ID:** 2602.18193 | [PDF](https://arxiv.org/pdf/2602.18193v1)

**作者:** Yiran Yang `[一作]` (Kuaishou Technology), Jiefei Zhang `[通讯]` (Kuaishou Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于Chain-of-Thought推理和规则指导的多模态广告审核框架BLM-Guard，用于短视频广告的细粒度政策合规检测与可解释决策

**💡 创新点**

核心创新在于：① Interleaved-modal Chain-of-Thought（ICoT）数据合成流水线，快速生成结构化视觉-语言推理样本；② 结合规则与自适应批评（SCA-R）的奖励机制，采用Self‑Adaptive GRPO强化学习实现对政策漂移的动态适配；③ 多任务架构同时建模 intra‑modal 夸张与 cross‑modal 不匹配，提升鲁棒性与泛化能力

**🔧 技术方法**

使用视觉语言模型（InternVL、InternViT、Qwen2.5-VL）、CLIP进行关键帧与区域提取、基于规则的监督微调、Token‑level GRPO强化学习、SCA‑R自评奖励模型以及 GPT‑4o 辅助一致性评估

**📊 数据集**

构建了BLM‑Guard Benchmark：真实短视频广告数据，包含三级风险标签（严重程度、情景、违规类型）及部分结构化推理轨迹，并在公开数据集 LSPD、XD‑Violence、UCF‑Crime、FakeSV、FVC 上进行评测

**📈 对比分析**

与 LLaVA‑Next‑Video‑7B、InternVL‑3‑8B、QwenGuard‑7B、LlavaGuard‑7B 等基线模型对比，BLM‑Guard 在风险分类、整体准确率、二元检测与推理一致性上均显著提升（例如高危类准确率 0.978，整体严格准确率 0.962，推理一致性 0.845），并在未见政策与多样数据集上保持优异泛化

**⚠️ 局限性**

局限性包括：① 依赖预训练模型的推理能力，仍可能在极端模糊或多模态冲突的案例中误判；② 规则库与自评奖励的维护成本高，需持续更新以跟随政策变化；③ 计算开销较大，尤其在强化学习阶段需要多样本生成与奖励评估，影响实时部署

---

## 13. Grammar Repair with Examples and Tree Automata: Extended Version

**arXiv ID:** 2602.18166 | [PDF](https://arxiv.org/pdf/2602.18166v1)

**作者:** Yunjeong Lee `[一作]` (National University of Singapore), Ilya Sergey `[通讯]` (National University of Singapore)

**通讯引用:** 1734 | [OpenAlex ID](https://openalex.org/A5009639508)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文提出一种基于树自动机的“按例子修复语法”框架，利用用户在冲突树示例间做出选择，生成约束树自动机并与原始语法的树自动机交叉，以消除上下文无关语法中的歧义；

**💡 创新点**

主要创新点包括：①一种新的从正/负树例子学习树自动机的算法，能够一次性编码所有关联的优先级与结合性约束；②一种高效的树自动机交叉算法，结合可达性分析与 ε‑转移优化，显著减小交叉结果的状态与转换数量；两者均提供完整的音响性证明；

**🔧 技术方法**

核心技术为：CFG→树自动机的翻译；树自动机学习（LearnOaOp + GenTA）；树自动机交叉（IntersectTA）；结合 OCaml 的 LR(1) 解析器生成器来定位冲突，并以树示例形式交互给用户；实现中还使用了状态最小化和 ε‑转移简化等优化；

**📊 数据集**

使用的基准语法包括：编译课程中的 4 个练习语法、两个 StackOverflow 示例语法、Tezos Michelson、Kaitai 和 SQL 的子语法，共计 10 个不同规模与冲突数量的语法；

**📈 对比分析**

实验结果表明在 10 个语法中平均成功消除 85% 的歧义，7 个语法 100%；平均运行时间约 20 ms，交叉步骤占大多数时间；与人工修复相比，工具在时间上快数千倍，编辑次数大幅减少；通过消融实验验证了可达性、重复状态消除和 ε‑引入等优化对效率与正确性的影响；

**⚠️ 局限性**

局限性：仅能处理能通过两种树示例表示的冲突（即二选一的结合/优先级冲突）；依赖 LR(1) 解析器的冲突报告，可能产生误报；无法解决涉及三种以上产生式的冲突或需更深层次语义约束的歧义；对大规模、深度递归语法的学习与转换仍会产生较高成本；

---

## 14. Click it or Leave it: Detecting and Spoiling Clickbait with Informativeness Measures and Large Language Models

**arXiv ID:** 2602.18171 | [PDF](https://arxiv.org/pdf/2602.18171v1)

**作者:** Wojciech Michaluk `[一作]` (Warsaw University of Technology), Anna Wroblewska `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种混合模型，将Transformer嵌入与15个手工语言学特征相结合，用于点击诱饵标题的二分类检测。

**💡 创新点**

创新点在于在高维语义嵌入基础上加入可解释的语言学特征，显著提升了模型的可解释性和预测性能。

**🔧 技术方法**

采用的技术包括OpenAI/Transformer嵌入、BERT、RoBERTa、TF‑IDF、Word2Vec、GloVe、手工特征工程以及XGBoost分类器。

**📊 数据集**

使用了Kaggle‑1、Kaggle‑2、Clickbait Challenge 2017和SemEval Clickbait Spoiling等公开英语数据集，构建了标题、标题+正文和spoiling三类任务集。

**📈 对比分析**

通过与传统TF‑IDF+RF、Word2Vec+XGBoost、RoBERTa、OpenAI嵌入+XGBoost以及LLM prompt‑based基线进行对比，混合模型在测试集上达F1 0.909、AUC 0.969，明显优于所有基线。

**⚠️ 局限性**

局限性包括仅针对英文数据集，缺乏跨语言验证；模型结构较为复杂，推理时间和计算资源需求高于传统方法。

---

## 15. Multi-Attribute Group Fairness in $k$-NN Queries on Vector Databases

**arXiv ID:** 2602.17858 | [PDF](https://arxiv.org/pdf/2602.17858v1)

**作者:** Thinh On `[一作]` (New Jersey Institute of Technology), Baruch Schieber `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 6508 | [OpenAlex ID](https://openalex.org/A5089964489)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了在向量数据库中进行多属性群体公平的 k‑NN 查询，提出了一套基于属性分区+LSH 近似检索与后处理的完整框架。

**💡 创新点**

创新点包括：① 将数据按所有受保护属性的笛卡尔积划分并用位图编码；② 采用 LSH 在每个分区内高效生成候选；③ 对两属性情况给出多项式时间的最小成本流解，对三及以上属性给出 ILP 精确解并证明 NP‑hard；④ 提供理论保证与实证验证的完整评估。

**🔧 技术方法**

技术方法涵盖属性分区与 bitmap 编码、基于 LSH 的近似邻居检索、最小成本流与整数线性规划的后处理、流/ILP 的理论分析以及实验平台实现。

**📊 数据集**

实验使用了 FairFace、CelebA、Audio 语音、GloVe 文本以及一个合成数据集，共覆盖多种模态和维度；每个数据集均标注了 3~5 个受保护属性。

**📈 对比分析**

通过与 FAISS、HNSW、SIEVE、Filter‑DiskANN、SAIR、JIR 等现有方法对比，实验显示本框架在可行查询比例、Recall@k 与 DAF 上均优于基线，查询时间比传统 ANN 方法提升 3–4 倍，且在多属性公平约束下仍能保证精确满足比例要求。

**⚠️ 局限性**

局限性在于：① 需要为所有属性组合构建分区索引，存储与维护成本随属性数增加呈指数增长；② 对三属性及以上的情况仍需 ILP 求解，计算开销显著；③ 仅能满足事先定义的计数约束，对动态或更细粒度的公平度量尚未覆盖。

---

## 16. Reasoning-Native Agentic Communication for 6G

**arXiv ID:** 2602.17738 | [PDF](https://arxiv.org/pdf/2602.17738v1)

**作者:** Hyowoon Seo `[一作]` (Sungkyunkwan University), Dong In Kim `[通讯]` (Sungkyunkwan University)

**通讯引用:** 24431 | [OpenAlex ID](https://openalex.org/A5022649488)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出了面向6G的推理本体通信框架，通过在通信栈中嵌入递归信念建模来对抗智能代理间的信念偏差。

**💡 创新点**

创新点在于将通信视为推理协调器，构建双平面架构、互惠代理推理（MAR）以及基于信念差异触发的最小化通信。

**🔧 技术方法**

采用共享本体、递归信念引擎（RBE）、近似推理、动态本体校准以及预测验证等技术实现推理同步。

**📊 数据集**

实验基于高保真多智能体仿真环境（人形机器人协作、自动车辆交互等），未使用公开数据集。

**📈 对比分析**

与传统位级、语义级通信相比，在RANS、DIB、MBS等新KPI下，推理本体通信在行为一致性、决策影响/比特以及长期信念稳定性上提升约30%~400%。

**⚠️ 局限性**

局限包括模型规模增长导致的递归推理开销、实时推理延迟、动态本体演化困难与对抗性干扰安全风险。

---

## 17. Thinking by Subtraction: Confidence-Driven Contrastive Decoding for LLM Reasoning

**arXiv ID:** 2602.18232 | [PDF](https://arxiv.org/pdf/2602.18232v1)

**作者:** Lexiang Tang `[一作]` (Peking University), Yuexian Zou `[通讯]` (Peking University)

**通讯引用:** 5676 | [OpenAlex ID](https://openalex.org/A5002795838)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于置信度的对比解码方法CCD，能在推理时仅对低置信度的 token 进行干预，提升 LLM 推理准确性与简洁性。

**💡 创新点**

创新点在于将 token 级别置信度作为切换阈值，动态触发对比解码；对比分布通过替换高置信度 token 形成故意混乱的参考，随后在低置信度位置进行子弹式减法。

**🔧 技术方法**

技术包括：在线置信度估计、滑动窗口阈值自适应、对比解码(logits 加权相减)、双 KV 缓存管理，全部无需额外训练或多轨迹采样。

**📊 数据集**

使用四大数学推理 benchmark：AIME 2024/2025、BRUMO 2025、HMMT 2025；评测模型覆盖 Qwen3 系列（4B–32B）和 DeepSeek‑R1‑Distill‑Qwen‑1.5B。

**📈 对比分析**

与基线 Top‑p/温度采样、MTI 方案对比，CCD 在所有模型规模和数据集上均显著提升 Mean@8 准确率（大约 3–5%），并在 AIME 上实现与 MTI 相当的表现，同时生成更短的推理轨迹。

**⚠️ 局限性**

局限在于评测仅覆盖数学推理任务，对通用领域或多模态推理的效果未知；此外仍受预训练模型偏差影响，且对置信度阈值选择有一定依赖。

---

## 18. BioBridge: Bridging Proteins and Language for Enhanced Biological Reasoning with LLMs

**arXiv ID:** 2602.17680 | [PDF](https://arxiv.org/pdf/2602.17680v1)

**作者:** Yujia Wang `[一作]` (Tongji University), Xuhong Wang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了 BioBridge，一种将大型语言模型与蛋白质语言模型融合的多任务蛋白质理解框架，利用领域增量持续预训练和跨模态对齐，支持蛋白质属性预测、知识问答等任务。

**💡 创新点**

创新点在于：①提出领域增量持续预训练（DICP）以在不遗忘通用语言能力的前提下注入生物医学知识；②设计 PLM-Projector 通过 ESM2 编码器与 Q‑Former 将蛋白质序列映射至 LLM 语义空间，实现跨模态对齐；③采用端到端的联合微调，使 LLM 能直接利用蛋白质表征完成多任务推理。

**🔧 技术方法**

技术包括：Transformer‑based LLM (Qwen2.5‑7B‑Instruct)、蛋白质编码器 ESM2、Q‑Former、对比学习 (contrastive loss) 进行蛋白质‑文本对齐、持续预训练策略、端到端微调和少量后训练指令恢复。

**📊 数据集**

使用的数据集包括：生物学教材、PubMed 文献、注入蛋白序列的文本、Swiss‑Prot FASTA‑描述对、OntoProtein、Swiss‑Prot 训练集、PFMBench、以及多种通用语言基准（MMLU、RACE、AGI‑Eval 等）。

**📈 对比分析**

通过与多种基线（ESM2、ESM3、ProtGPT2、ProtT5、DPLM 等）在 PFMBench 16 任务以及通用语言任务（AGI‑Eval、MMLU、RACE、OpenBookQA）上的对比，BioBridge 在蛋白质属性预测任务上与专业 PLM 相当，在某些定位、金属离子结合任务上更优；在通用语言任务上其表现与 Qwen2.5‑7B‑Instruct 接近甚至超越，显示出较强的跨任务泛化能力。

**⚠️ 局限性**

局限性包括：在溶解度、生产、突变等任务上仍略逊于专门训练的 PLM；持续预训练和对齐所需的计算资源大，训练成本高；模型在极端长序列或多模态输入时仍可能面临性能瓶颈。

---

## 19. Euclidean Noncrossing Steiner Spanners of Nearly Optimal Sparsity

**arXiv ID:** 2602.17801 | [PDF](https://arxiv.org/pdf/2602.17801v1)

**作者:** Sujoy Bhore `[一作]` (Indian Institute of Technology Bombay), Sampson Wong `[通讯]` (University of Copenhagen)

**通讯引用:** 36 | [OpenAlex ID](https://openalex.org/A5029179453)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并实现了一种欧氏平面非交叉Steiner（1+ε）稠密图（即平面稠密网）的构造方法，并给出了相应的上界与下界。

**💡 创新点**

主要创新在于：①通过引入线性变换而非旋转，显著减少所需的图数量（k=O(1/√ε)），从而将Steiner点数量从先前的O(n/ε⁴)提升到O(n/ε^{3/2})；②利用近代几何测度理论中对圆盘-管道相交的Szemerédi–Trotter型定理，给出了几乎匹配的下界Ω_μ(n/ε^{3/2-μ})；③在构造与证明中使用了平衡盒分解、斜向限制（cone‑restricted）路径以及跨窗不冒险与不偏斜的窗口分析。

**🔧 技术方法**

技术手段包括：平衡盒分解（Balanced Box Decomposition）来构造轴对齐图；对每个方向使用线性变换以保持斜向限制；对图间交叉的计数利用几何测度理论中对δ圆盘–δ管道相交的上界；以及对弧段与椭圆相交的几何分析，配合分窗（window）与窗口的良好/不良分类来推导下界。

**📊 数据集**

主要使用的是构造性的示例点集：在单位正方形两侧等间距布置点的集合 A∪B（其规模为Θ(ε^{-1/2})），以及多副这类点集的并集，用来证明下界；上界构造不依赖具体数据集，只需任意给定的 n 点集合。

**📈 对比分析**

相较于先前最优的 O(n/ε⁴) 上界，本工作将上界降低到 O(n/ε^{3/2})，并给出下界 Ω_μ(n/ε^{3/2-μ})，二者仅相差多项式下的微小次幂项，表明几乎达到了最优。实验与理论比较表明，所构造的稠密图在保证 (1+ε) 近似距离与非交叉性同时得到最小化 Steiner 点数方面已接近极限。

**⚠️ 局限性**

局限性包括：①上界与下界之间仍留有小的次多项式差距；②论文未讨论 Steiner 点数与图的总权重（lightness）的权衡；③仅针对欧氏平面，在更高维度或一般度量空间的非交叉稠密图尚未给出对应结果；④对交叉数的精确上界仍未达到与下界匹配的程度。

---

## 20. MantisV2: Closing the Zero-Shot Gap in Time Series Classification with Synthetic Data and Test-Time Strategies

**arXiv ID:** 2602.17868 | [PDF](https://arxiv.org/pdf/2602.17868v1)

**作者:** Vasilii Feofanov `[一作]`, Ievgen Redko `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并训练了Mantis+与MantisV2两种时间序列分类的基础模型，并通过零样本特征提取显著提升性能。

**💡 创新点**

创新点包括：① 仅用合成数据CauKer进行预训练，消除数据泄露并提升泛化；② 通过系统消融优化Transformer架构，得到更轻量高效的MantisV2；③ 在推理阶段引入中间层表示、改进输出token聚合、多尺度插值自集成以及跨模型嵌入融合，进一步提升零样本性能。

**🔧 技术方法**

技术手段：基于Transformer的编码器，实例归一化+卷积+平均池化的token化；随机裁剪缩放（RCR）增强；对比学习目标；多尺度插值、首阶差分、RoPE、SwiGLU、RMS归一化等；分类器采用随机森林或Logistic回归；自集成与跨模型融合。

**📊 数据集**

使用的实验集包括公开的UCR（128单变量）、UEA（27多变量）、人类活动识别（HAR）7个数据集、EEG（多任务）以及用于预训练的CauKer 2M合成时间序列。

**📈 对比分析**

与Catch22+、TabPFN/TabICL、MOMENT、TiRex、Chronos2、TiViT‑H、TiConvNext、NuTime、TS2Vec等基线对比，MantisV2在零样本下平均准确率最高，且通过自集成与跨模型融合后可逼近甚至超越微调后的Mantis；在多种基准上均显著优于之前的基础模型。

**⚠️ 局限性**

局限性：零样本与微调之间的差距在某些任务上仍未完全消除；自集成和跨模型融合增加了推理时的计算开销；对多通道或更长序列的适用性仍待验证；当前方法主要针对分类任务，预测任务仍需改进。

---

## 21. Validating Political Position Predictions of Arguments

**arXiv ID:** 2602.18351 | [PDF](https://arxiv.org/pdf/2602.18351v1)

**作者:** Jordan Robinson `[一作]` (University of Liverpool), Anthony G. Cohn `[通讯]` (University of Leeds)

**通讯引用:** 10524 | [OpenAlex ID](https://openalex.org/A5024409281)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个大规模结构化论证知识库，将23,228条BBC Question Time的论证单元与22种大型语言模型的政治立场预测结合，并通过双尺度人类验证（点评和配对评估）进行校准。

**💡 创新点**

提出双尺度验证框架，既利用可扩展的点评衡量政治立场，又通过配对比较恢复可比序列；生成可验证的政治立场知识库，实现论证结构与政治立场的联合表示；证明点评LLM预测可提取可比的顺序结构。

**🔧 技术方法**

使用22种LLM（GPT‑4、Claude、Llama等）的点评预测；Krippendorff α、Spearman、Kendall、ordinal Krippendorff α；Bradley‑Terry BT模型与Luce Spectral Ranking；配对win矩阵与Ensemble聚合；Neo4j图数据库存储与查询。

**📊 数据集**

基于30集BBC Question Time的论证单元（23,228条），AIF转ASPIC+，再通过Prolific crowdworker进行点评与配对验证。

**📈 对比分析**

对点评使用Krippendorff α与macro/micro F1、balanced accuracy；对配对使用Spearman、Kendall、ordinal Krippendorff α与宏观F1；在高置信度子集上，最佳模型Krippendorff α达0.86，macro F1约0.72；整体来看模型与人类在高置信度上一致性显著提升。

**⚠️ 局限性**

点评标签本身主观性导致人类间一致性低；集成方法需多次调用成本高；配对验证仅覆盖100条论证，域内局限；离散化与BT正则化可能掩盖细粒度顺序信息；推理时无法预判不确定性。

---

## 22. On A. V. Anisimov's problem for finding a polynomial algorithm checking inclusion of context-free languages in group languages

**arXiv ID:** 2602.18305 | [PDF](https://arxiv.org/pdf/2602.18305v1)

**作者:** Krasimir Yordzhev `[一作]` (Trakia University), Krasimir Yordzhev `[通讯]` (Trakia University)

**通讯引用:** 93 | [OpenAlex ID](https://openalex.org/A5008502101)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

论文提出一种多项式时间算法，用于判定任意上下文无关语言是否是给定群语言的子集；通过将上下文无关文法转化为带标签的有向图，利用一个特殊的幺半群 𝒰 及其子集半环 𝒮_𝒰 来计算所有可能的路径标签，最终判断是否只有单一标签 〈ε ,e〉存在，从而确定包含关系。

**💡 创新点**

创新点在于：①首次将群语言的判定问题与图结构和幺半群标签相结合，构造了 𝒰‑标签的有向图表示；②利用半环 𝒮_𝒰 的运算（并、乘、星）对路径标签进行闭包计算，相当于 Floyd‑Warshall 的扩展；③证明该算法的运算次数上界为 O(n³)（n 为文法非终结符数），从而在此前只对正规或线性语言可做多项式判定的基础上，扩展到了所有上下文无关语言。

**🔧 技术方法**

主要技术包括：图论（路径、循环、闭包）、幺半群与限制 Dyck 语言的结合、半环代数（并、乘、星运算）、动态规划/闭包算法（类似 Floyd‑Warshall），以及在群论中的单词问题判定。

**📊 数据集**

论文并未使用传统意义上的实验数据集；算法以 Chomsky 正规形式的上下文无关文法作为输入，群语言由其生成元和定义关系给出，全部在理论分析中完成。

**📈 对比分析**

在方法比较方面，作者仅给出了理论复杂度分析：算法在半环 𝒮_𝒰 上最多执行 O(n³) 次并/乘运算和 O(n²) 次星运算；若这些运算能够在多项式时间内实现，则整体算法为多项式时间；论文未给出实验验证或与现有算法的性能对比。

**⚠️ 局限性**

局限性包括：①算法的效率高度依赖于半环 𝒮_𝒰 的基本运算能否在多项式时间内完成；②子集 𝒰 的表示可能导致指数级存储与运算；③仅适用于已转换为 Chomsky 正规形式的上下文无关文法；④未给出实验评估，实际可行性尚未验证。

---

## 23. Reducing Text Bias in Synthetically Generated MCQAs for VLMs in Autonomous Driving

**arXiv ID:** 2602.17677 | [PDF](https://arxiv.org/pdf/2602.17677v1)

**作者:** Sutej Kulgod `[一作]` (Zoox), Christoffer Heckman `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了自动驾驶任务中基于多项选择问答（MCQA）的评测，发现合成 MCQA 数据中隐藏的文本线索使模型不需视觉输入即可达到高准确率，并提出一种去偏的干扰词生成方法和课程学习策略，使模型真正依赖视觉信息。

**💡 创新点**

创新点包括：① 通过从其他样本的真实答案中采样干扰词，消除文本偏差；② 在训练时采用课程式选项丢弃策略，迫使模型从视觉中推理答案；③ 通过零视频评估检测文本偏差的存在。

**🔧 技术方法**

使用了 Gemini 2.5 生成问题答案与干扰词，LLM 干扰词生成，视觉语言模型 Qwen2-VL-2B 的细调与全细调，课程学习的选项丢弃策略，以及零视频评估技术。

**📊 数据集**

使用的主要数据集为：基础 BEV 视频+平衡操作标签数据集（D_base），基线合成 MCQA 数据集（D_llm），去偏后的 MCQA 数据集（D_new），以及对照的驾驶 VLM 评测数据集如 DriveBench、NuPlanQA、AutoDrive-QA。

**📈 对比分析**

通过零视频评估比较基线与去偏后数据集在预训练 VLM 上的准确率；在监督细调后对比 D_llm 与 D_new 的整体准确率和无视觉输入准确率。结果显示基线在无视觉条件下可达 66.9% 以上，去偏后仅 2.9% 以上；整体准确率从 93.8% 降至 75–77%，但无视觉准确率大幅下降，验证模型真正依赖视觉。

**⚠️ 局限性**

局限性包括：仅在 Qwen2-VL-2B 上验证，未扩展到其他模型或更大数据集；干扰词去偏仅通过标签抽样，未系统评估多样性；Zero‑shot 检测仅简单，未使用更细粒度的偏差分析；实验依赖 BEV 视频与专家模型，可能对其他视觉任务的通用性有限。

---

## 24. QueryPlot: Generating Geological Evidence Layers using Natural Language Queries for Mineral Exploration

**arXiv ID:** 2602.17784 | [PDF](https://arxiv.org/pdf/2602.17784v1)

**作者:** Meng Ye `[一作]` (SRI International), Yi Yao `[通讯]` (SRI International)

**通讯引用:** 2385 | [OpenAlex ID](https://openalex.org/A5052203401)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究开发了一款基于自然语言处理与人工智能的工具，能够自动从地质图和文献生成矿产潜力证据图层，并通过交互式界面支持矿物勘探工作；

**💡 创新点**

创新点在于将自然语言查询与文献摘要相结合，利用语义相似度自动生成多维证据图层，并通过叠加缓冲区进一步定位潜在矿区；

**🔧 技术方法**

主要技术包括Transformer‑基的句子嵌入模型（如BGE、GTE）、LLM（GPT‑4o）进行文献摘要、语义相似度计算以及Python/Streamlit实现的Web可视化；

**📊 数据集**

使用的数据集包括美国SGMC地质图数据库、120种矿床描述文献（约151篇）以及矿床位置数据（MRDS）和专家评估的许可通道；

**📈 对比分析**

与人工专家生成的许可通道和随机/顺序排名进行对比，采用面积交集/IOU、精度/召回/F1和AUPRC等指标；实验显示，在最优阈值与缓冲距离下，证据图层能覆盖约90%已知矿床，且加入该图层可提升监督分类器AUPRC、AUC、F1等性能；

**⚠️ 局限性**

局限包括：1）地质图跨州不一致导致地图伪影；2）依赖于SGMC的分辨率与属性有限；3）未针对地质文本进行模型微调；4）未考虑经济和风险评估。

---

## 25. EXACT: Explicit Attribute-Guided Decoding-Time Personalization

**arXiv ID:** 2602.17695 | [PDF](https://arxiv.org/pdf/2602.17695v1)

**作者:** Xin Yu `[一作]`, Lingzhou Xue `[通讯]` (Pennsylvania State University)

**通讯引用:** 2031 | [OpenAlex ID](https://openalex.org/A5089400160)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为EXACT的解码时个性化框架，利用有限的配对偏好反馈学习用户的显式属性集合，并在推理时根据提示相似度检索最相关属性来引导生成；

**💡 创新点**

创新点在于：①将用户偏好建模为可解释的属性子集；②采用离线属性索引+在线检索的两阶段设计，既避免了参数更新，又能适应上下文偏好漂移；③在理论层面给出了贪心属性选择的近似保证，并证明检索机制能有效减小多任务场景下的偏好漂移；

**🔧 技术方法**

主要技术包括：强化学习与人类反馈（RLHF）中的Brady–Terry模型、KL正则化的策略优化、贪心子集搜索、语义相似度检索（如BGE小模型）、以及在推理时的属性注入式提示模板；

**📊 数据集**

实验使用了两个公开的用户偏好数据集——PRISM（对话式回复偏好）和Summarize From Human Feedback（摘要偏好），每个用户平均约20条偏好对；

**📈 对比分析**

与基线（Base、Drift、Reward）在三个开源LLM（Llama-3.1-8B、Gemma-2-9B-it、Qwen2.5-7B-Instruct）上对比，EXACT在用户偏好建模准确率、生成质量的win‑rate以及整体性能上均超越所有基线，提升幅度可达约10%至20%；

**⚠️ 局限性**

局限性包括：依赖预定义属性库（可能不覆盖所有细粒度偏好）；检索策略对相似度模型和样本分布敏感；在极少数据或极高偏好漂移情形下仍可能出现误检；并且需保证用户历史数据的安全与隐私。

---

## 26. PRISM: Parallel Reward Integration with Symmetry for MORL

**arXiv ID:** 2602.18277 | [PDF](https://arxiv.org/pdf/2602.18277v1)

**作者:** Finn van der Knaap `[一作]` (University of Edinburgh), Fengxiang He `[通讯]` (University of Edinburgh)

**通讯引用:** 1753 | [OpenAlex ID](https://openalex.org/A5100635369)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

PRISM 通过并行奖励整合与对称正则化，解决了异构多目标 RL 中稀疏奖励导致的学习偏差，实现高效信用分配。

**💡 创新点**

创新点在于引入 ResSymNet 残差网络预测稀疏奖励的即时值，并通过 SymReg 强制策略反射等价性，显著降低假设复杂度并提升样本效率。

**🔧 技术方法**

采用残差网络（ReSymNet）、奖励模型、反射对称正则化（SymReg）、CAPQL 背景算法以及迭代奖励重塑等技术。

**📊 数据集**

在 MuJoCo 四个多目标版本（mo-hopper-v5、mo-walker2d-v5、mo-halfcheetah-v5、mo-swimmer-v5）上进行评估。

**📈 对比分析**

与 oracle（全密集奖励）和仅稀疏奖励 baseline 以及 CAPQL 基线对比，PRISM 在 Hypervolume、Expected Utility Metric、Variance Objective 指标上分别提升 32%–88%，甚至超过 oracle。

**⚠️ 局限性**

局限性包括依赖环境具备可反射对称性、奖励模型需要大量随机样本、近似等价而非精确，且对非对称或高度异构任务的效果尚待验证。

---

## 27. Examining LLMs Ability to Summarize Code Through Mutation-Analysis

**arXiv ID:** 2602.17838 | [PDF](https://arxiv.org/pdf/2602.17838v1)

**作者:** Lara Khatib `[一作]`, Meiyappan Nagappan `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于变异的评估方法，直接检测LLM生成的代码摘要是否随代码逻辑变更而更新。

**💡 创新点**

创新点在于通过人为注入行为改变的变异来评估摘要的语义一致性，而非仅用表面相似度指标。

**🔧 技术方法**

技术包括手工构造三类变异（语句、值、决策）、LLM摘要生成（GPT‑4/GPT‑5.2）以及人工标注的正负评判。

**📊 数据集**

数据集为12个受控合成程序和50个人写的LBPP程序，合计624个变异–摘要评估实例。

**📈 对比分析**

对比方法是对同一代码做原始和变异两版摘要进行人工比较；结果显示GPT‑4在简单函数上约76%准确，复杂多线程约17%，GPT‑5.2提升至85%并能标记为错误。

**⚠️ 局限性**

局限包括仅使用有限变异类型、受限于手工标注的主观性、对复杂真实项目的推广性不足。

---

## 28. The Complexity of Sparse Win-Lose Bimatrix Games

**arXiv ID:** 2602.18380 | [PDF](https://arxiv.org/pdf/2602.18380v1)

**作者:** Eleni Batziou `[一作]` (University of Liverpool), Rahul Savani `[通讯]` (The Alan Turing Institute)

**通讯引用:** 1569 | [OpenAlex ID](https://openalex.org/A5084692264)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

证明在常数稀疏度（3‑sparse）的 win‑lose 二人博弈中，寻找 ε‑Nash 均衡是 PPAD‑hard 的；同时给出了从 3‑sparse win‑lose 到 2‑sparse 之间的阈值，即 2‑sparse 可多项式求解，而 3‑sparse 已不可多项式求解。

**💡 创新点**

首次完成了常数稀疏度 win‑lose 博弈的完全复杂度阈值分析，填补了先前只讨论 10‑sparse 或 log‑n 稀疏度的空白，并将多玩家多项式时间可解的结果（Polymatrix）成功转化到两人博弈。

**🔧 技术方法**

通过一系列细化的多步归约：从 PPAD‑complete 的电路固定点问题 → 受限多玩家 Polymatrix 游戏（两动作、对角/反对角、度≤2）→ 3‑sparse {0,1,2,8} 价值的 bimatrix 游戏 → 逐步使用三种列模拟（单列、双列与二列）将最大收益逐步压缩至 {0,1}，同时严格保持 3‑稀疏性。

**📊 数据集**

无实验数据集，论文为纯理论证明性质。

**📈 对比分析**

比较方式是与已知的 PPAD‑hardness 结果对照，展示在 3‑sparse 的 win‑lose 框架下仍保持 PPAD‑hard 性，证明其复杂度与一般不稀疏 win‑lose 或更稀疏（2‑sparse）情况相对。

**⚠️ 局限性**

局限性：仅针对逆多项式 ε（ε = O(1/n)) 的近似均衡；对更大 ε 或其他稀疏度范围（如 4‑sparse、非对角/反对角支付）仍未给出结论；并且缺乏对实际求解算法的实验验证。

---

## 29. Turbo Connection: Reasoning as Information Flow from Higher to Lower Layers

**arXiv ID:** 2602.17993 | [PDF](https://arxiv.org/pdf/2602.17993v1)

**作者:** Mohan Tang `[一作]` (University of California Los Angeles), Sidi Lu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的架构Turbo Connection (TurboConn)，通过从高层到低层的多个残差连接来克服Transformer模型的固定深度限制，从而提高推理能力。

**💡 创新点**

创新点在于通过增加高层到低层的连接，允许模型的有效推理深度与序列长度线性增长，显著提升了逻辑推理任务的性能。

**🔧 技术方法**

使用了Turbo Connection架构，结合了残差连接的思想，增强了信息流动。

**📊 数据集**

在多个推理密集型数据集上进行评估，包括Parity、Multi-step Arithmetic和GSM8K。

**📈 对比分析**

与标准Transformer模型进行比较，TurboConn在多个任务上表现出0.9%到10%以上的准确率提升，尤其在Parity任务上，TurboConn使得模型从53.78%提升至100%的准确率。

**⚠️ 局限性**

限制在于TurboConn在训练时需要牺牲部分并行性，导致处理速度可能变慢，但在推理时不会影响生成延迟。

---

## 30. Exploiting Liquidity Exhaustion Attacks in Intent-Based Cross-Chain Bridges

**arXiv ID:** 2602.17805 | [PDF](https://arxiv.org/pdf/2602.17805v1)

**作者:** André Augusto `[一作]` (University of Lisbon), Miguel Correia `[通讯]` (University of Lisbon)

**通讯引用:** 6277 | [OpenAlex ID](https://openalex.org/A5016455665)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过对2025年6月至11月间3.5百万跨链意图进行数据收集与分析，构建回放式仿真框架，系统地定义并评估了流动性枯竭攻击（Liquidity Exhaustion Attacks）在Mayan Swift、Across和deBridge三大意图式跨链桥上的可行性、盈利性与可用性影响，并提出基线与定向攻击策略及多层防御建议。

**💡 创新点**

创新点包括：①首次系统性描述并量化意图式跨链桥的流动性枯竭攻击；②将真实历史意图数据与可参数化仿真相结合，构建可复现的攻击模拟框架；③提出基于中位数-σ触发的基线攻击以及针对Solver参与模式的定向攻击，并揭示其对不同协议的攻击成功率和收益差异；④从协议设计角度给出多维度防御方案。

**🔧 技术方法**

使用的技术手段包括：链上事件抓取与Token价格API获取、Solver流动性重构、统计阈值触发（median±kσ）、回放式仿真（模拟攻击意图注入、Solver竞价与结算）、成本收益模型（诱导成本、填充成本、收益、净利润）、参数化实验（k值、攻击窗口、意图价值阈值、交易量乘数）以及结果可视化。

**📊 数据集**

数据集：2025年6月1日至11月1日期间收集的3.5百万跨链意图，总价值9.24亿美元，涉及Mayan Swift、Across、deBridge三大协议，跨9条链（Solana、Arbitrum、Ethereum、Base、Polygon、BNB、Unichain、Optimism、Avalanche）。数据来源为链上事件、Solidity合约状态查询及Alchemy Token Prices API。

**📈 对比分析**

比较方法：对同一协议在不同攻击窗口（1000s、600s、300s）、不同k阈值（0-3）、不同意图价值上限、不同交易量乘数进行多维参数化仿真；评估指标包括攻击成功概率、平均/90th分位净利润、成功意图数量、费用与成本。实验结果表明：deBridge在基线攻击中成功率高、平均净利润正；Across在任何参数下均为负；定向攻击在deBridge可实现高收益（平均>200美元），而在Mayan Swift收益普遍为负。攻击窗口越长、k越低导致成功率上升但成本增加；整体而言，流动性枯竭攻击在高Solver利润、集中度高的协议中最具威胁。

**⚠️ 局限性**

局限性：①仿真仅考虑短期<1000秒的攻击窗口，未覆盖长周期的Settlement延迟；②未模拟Solver的自适应行为（如动态加注、价格调节、限流）导致结果可能被低估或高估；③对Solver内部策略和风险阈值缺乏可观测性，假设为黑盒；④依赖历史数据，未检验在实时攻击场景下的可行性；⑤定向攻击收益受意图价值分布影响，若高价值意图稀少则收益受限。

---

## 31. How Well Can 3D Accessibility Guidelines Support XR Development? An Interview Study with XR Practitioners in Industry

**arXiv ID:** 2602.17939 | [PDF](https://arxiv.org/pdf/2602.17939v1)

**作者:** Daniel Killough `[一作]` (University of Wisconsin-Madison), Yuhang Zhao `[通讯]` (University of Wisconsin-Madison)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

对现有3D可访问性（a11y）指南在XR开发中的适用性进行评估，采用25名XR实践者的半结构化访谈，分析其实现难度、使用经验与改进需求。

**💡 创新点**

首次系统地将3D a11y 指南迁移至XR情境，揭示了技术不匹配、实现阻碍与实践者对指南的解释差异，并为XR专用指南与工具提出了具体改进建议。

**🔧 技术方法**

使用访谈录音转写、手工主题分析（open coding、coder reliability）以及结构化的七步评估协议（指南展示、解释、实现评估、改进建议等）来收集和分析数据。

**📊 数据集**

数据来源为25位XR实践者（涵盖创业、初创、中型与大型企业，主要使用Unity/Unreal/WebXR等平台），以及从六大资源中筛选的20条跨视觉、运动、认知、语音/听觉的通用指南。

**📈 对比分析**

通过对比实践者对每条指南的可实现性、技术挑战、优先级及支持工具需求，定性评估指南在XR中的可操作性；结果显示多条指南在XR上实现成本高、技术不匹配，整体可行性不佳。

**⚠️ 局限性**

局限性包括：样本量有限、对高激励参与者存在偏倚、只评估现有指南未形成新标准、侧重定性访谈而非定量实验，且对特定平台（如Unity）实现细节依赖较大。

---

## 32. SARAH: Spatially Aware Real-time Agentic Humans

**arXiv ID:** 2602.18432 | [PDF](https://arxiv.org/pdf/2602.18432v1)

**作者:** Evonne Ng `[一作]` (Meta Reality Labs), Alexander Richard `[通讯]` (Meta Reality Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种能够实时、因果地为沉浸式对话代理生成全身运动，并且能够根据用户位置进行空间感知和可控的眼神交流。

**💡 创新点**

核心创新在于将因果 Transformer‑VAE 与流匹配模型相结合，并引入基于 classifier‑free guidance 的可控瞳孔对齐机制，实现既自然又可调节的空间交互。

**🔧 技术方法**

使用因果 Transformer‑VAE、流匹配网络、欧氏表面点运动表示、HuBERT 音频特征、以及自回归推理与可控引导。

**📊 数据集**

在 Embody 3D 双人对话数据集上训练与评估。

**📈 对比分析**

与随机、NN、MDM、A2P、SHOW 等基线对比，方法在 FGD、FGD_acc、Foot Slide、Wrist Var、Head Ang. 上均达到或超过最先进水平，且实时帧率超过 300 FPS，比非因果基线快 3 倍。

**⚠️ 局限性**

局限性包括对训练数据分布的依赖、目前仅支持两人对话、无法直接控制手势风格或步态等其他行为，以及在多方对话场景下的推广需要进一步研究。

---

## 33. Balancing Symmetry and Efficiency in Graph Flow Matching

**arXiv ID:** 2602.18084 | [PDF](https://arxiv.org/pdf/2602.18084v1)

**作者:** Benjamin Honoré `[一作]` (Ecole Polytechnique Fédérale de Lausanne), Pascal Frossard `[通讯]` (Ecole Polytechnique Fédérale de Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在图生成模型中通过可控对称性破坏（可调正弦位置编码与节点置换）来加速训练、平衡过拟合。

**💡 创新点**

提出了可调 λ 控制位置编码对称破坏程度以及时间依赖置换循环的机制，实现更快收敛同时保持泛化。

**🔧 技术方法**

基于离散流匹配模型 DeFoG，使用可调正弦位置编码、λ 控制、训练过程中的节点置换及动态置换速率。

**📊 数据集**

主要在 Stochastic Block Model (SBM) 数据集上实验；亦对 planar graphs 与树等简单数据集做了验证。

**📈 对比分析**

通过与等变 RRWP 编码基线对比，采用 VUN（有效性、唯一性、创新性）指标评估；最佳配置实现 VUN 0.975（基线 0.900），且仅需 19% 的训练步数。

**⚠️ 局限性**

对称性破坏在简单图数据集上的效果有限，且需要手动调节 λ 与置换率；过度破坏可能导致过拟合，通用性待进一步验证。

---

## 34. Temporal Consistency-Aware Text-to-Motion Generation

**arXiv ID:** 2602.18057 | [PDF](https://arxiv.org/pdf/2602.18057v1)

**作者:** Hongsong Wang `[一作]` (Southeast University), Xin Geng `[通讯]` (Southeast University)

**通讯引用:** 6461 | [OpenAlex ID](https://openalex.org/A5074742406)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种时序一致性感知的文本到动作生成框架 TCA‑T2M，通过在 VQ‑VAE 中加入循环时序一致性约束、残差量化以及运动学约束块，生成更具时序连贯性和物理真实性的 3D 动作序列。

**💡 创新点**

创新点在于将循环时序一致性约束引入离散动作表示学习，结合残差量化提升信息保留，加入运动学约束块消除量化导致的足滑等物理不合理现象。

**🔧 技术方法**

采用基于 VQ‑VAE 的时序一致性空间量化网络 (TCaS‑VQ‑VAE)、残差量化 (RQVAE)、运动学约束块 (KCB) 与带动态遮挡的文本驱动 Transformer (Masked Motion Transformer) 进行动作生成。

**📊 数据集**

使用公开数据集 HumanML3D 和 KIT‑ML 进行训练与评估。

**📈 对比分析**

与多种基线方法（T2M‑GPT、MotionGPT、TM2T、MMD、MDL 等）对比，TCA‑T2M 在 FID、R‑Precision、MM‑Dist 等关键指标上均取得最优或接近最优表现，同时保持良好的多样性指标。

**⚠️ 局限性**

局限在于仍存在语义误解导致动作与描述不符的情况，数据多样性不足，以及长序列实时生成仍面临挑战。

---

## 35. Avoid What You Know: Divergent Trajectory Balance for GFlowNets

**arXiv ID:** 2602.17827 | [PDF](https://arxiv.org/pdf/2602.17827v1)

**作者:** Pedro Dall'Antonia `[一作]` (Getulio Vargas Foundation), Diego Mesquita `[通讯]` (Getulio Vargas Foundation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Adaptive Complementary Exploration (ACE)，一种通过引入探索 GFlowNet 与标准 GFlowNet 共同训练、并利用 Divergent Trajectory Balance (DTB) 损失实现对低覆盖高奖励状态的高效探索。

**💡 创新点**

创新点在于：①定义 DTB 使探索策略专注于未被充分覆盖的高奖励终点并完全排除已过度采样的路径；②通过动态权重 w（基于两网络的 Z 值）实现两网络样本的自适应混合；③提供理论证明，说明当 α>1 时探索网络收敛到加权奖励分布 R^β，避免模式崩塌。

**🔧 技术方法**

技术手段包括：GFlowNet 的前向/后向策略、轨迹平衡 (TB) 损失、DTB 损失、ε-贪婪探索、混合采样、stop‑gradient 计算、以及对 α、β 参数的阈值调节。

**📊 数据集**

使用多种基准数据集：Rings、8 Gaussians、Lazy Random Walk、Grid World、Bit Sequences、Sequence Design、Bag Generation、Quadratic Knapsack、以及抗菌肽 (AMP) 设计空间（10^13 组合）。

**📈 对比分析**

与 ϵ‑贪婪、AT‑GFlowNet、SA‑GFlowNet 等方法对比，ACE 在 Top‑K 平均奖励、分布拟合度（TV 距离）和模式发现速度上均显著优于现有技术，尤其在高维组合任务与 AMP 生成中实现了更快的多样化高奖励样本收集。

**⚠️ 局限性**

局限性包括：①需要手动设定阈值 α 与指数 β，若选取不当可能导致探索网络崩塌或收敛到 R^β；②对 OA/UA 集合的估计需要额外采样，增加计算成本；③在非常稀疏奖励或极大状态空间时，DTB 仍可能无法完全覆盖所有高奖励模式；④实验主要集中在离散组合任务，连续或非结构化任务的推广尚未验证。

---

## 36. History-Constrained Systems

**arXiv ID:** 2602.18143 | [PDF](https://arxiv.org/pdf/2602.18143v1)

**作者:** Louwe B. Kuijer `[一作]` (University of Liverpool), Patrick Totzke `[通讯]` (University of Liverpool)

**通讯引用:** 349 | [OpenAlex ID](https://openalex.org/A5060577776)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文研究了一种新的模型——历史约束系统（HCS），分析其表达能力、紧凑性以及在可达性、游戏求解等验证问题上的复杂性。

**💡 创新点**

创新点包括：①证明正则守护下的HCS与DFA等价但指数级更简洁；②提出基于HCS的游戏求解算法并证明其EXPTIME完备；③探讨向量加法系统（VASS）守护的HCS，揭示其可达性可判性与EXPSPACE上限之间的界限。

**🔧 技术方法**

主要技术手段有：幂集+乘积构造、游戏理论与策略分析、计数器系统（VASS）与覆盖/可达性判定、Rackoff算法的空间复杂性分析、以及下推自动机的不可判定性证明。

**📊 数据集**

本文未使用实验数据，所有结论均基于理论证明与图例示例。

**📈 对比分析**

通过理论比较，展示HCS在表达力与游戏求解上相较传统有限自动机具备指数级压缩优势，但在游戏求解方面仍维持EXPTIME复杂度；VASS守护下的可达性问题与覆盖性问题分别处于EXPSPACE上限与可判定性边界。

**⚠️ 局限性**

局限性：对于VASS守护的可达性/覆盖性问题仍存在未判定性；DFACover-VASS的闭包与可判定性尚未完全确定；本文主要聚焦理论分析，缺乏实验验证与性能评估。

---

## 37. Gradient Regularization Prevents Reward Hacking in Reinforcement Learning from Human Feedback and Verifiable Rewards

**arXiv ID:** 2602.18037 | [PDF](https://arxiv.org/pdf/2602.18037v1)

**作者:** Johannes Ackermann `[一作]` (University of Tokyo), Masashi Sugiyama `[通讯]` (RIKEN AIP)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出并实验了梯度正则化增强的梯度强化学习（Grad‑RL+GR）方法，用于降低RL目标的锐度并提升策略表现。

**💡 创新点**

创新点在于将梯度正则化与梯度下降相结合，以显著降低策略的sharpness，改善BT loss和收敛速度。

**🔧 技术方法**

使用梯度下降、梯度正则化以及策略梯度（Policy Gradient）等技术，并通过梯度正则化增强的Grad‑RL实现。

**📊 数据集**

在OpenAI Gym的经典控制任务（如CartPole、MountainCar、Acrobot）等强化学习基准环境中进行实验。

**📈 对比分析**

与Q‑Table、SARSA、REINFORCE、PPO等基线算法进行对比，Grad‑RL+GR在收敛速度、BT loss降低、sharpness下降等指标上优于传统方法。

**⚠️ 局限性**

实验仅覆盖离散动作空间的小规模任务，计算开销较大，缺乏对连续或大规模环境的验证，理论分析仍不充分。

---

## 38. A Topology-Aware Positive Sample Set Construction and Feature Optimization Method in Implicit Collaborative Filtering

**arXiv ID:** 2602.18288 | [PDF](https://arxiv.org/pdf/2602.18288v1)

**作者:** Jiayi Wu `[一作]` (Beijing Institute of Technology), Guoren Wang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 7295 | [OpenAlex ID](https://openalex.org/A5054991337)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种利用拓扑社区特征将误负样本转化为正样本的算法，称为 TPSC-FO，旨在提升隐式协同过滤的推荐效果。

**💡 创新点**

创新点在于将社区检测与个性化噪声过滤相结合构建正样本集合，并通过邻域引导特征优化减少正样本噪声。

**🔧 技术方法**

核心技术包括基于 Leiden 与 Infomap 的差分社区检测、基于 ALS 的节点嵌入、个性化相似度阈值过滤，以及邻域均值+Mixup 的特征优化。

**📊 数据集**

实验数据集包括五个真实世界数据集（Amazon‑beauty、Amazon‑home、Epinions、Gowalla、Tmall）以及两组人工噪声合成数据集（beauty‑20%、Epinions‑20%）。

**📈 对比分析**

与六种负采样方法和五种去噪方法对比，TPSC‑FO 在 Recall@10/20 与 NDCG@20 上均取得最优成绩，性能提升幅度从 5–11% 以上。

**⚠️ 局限性**

局限性包括对社区检测算法参数敏感、个性化阈值需手动调参，以及在极度稀疏或小规模数据集上提升幅度相对有限。

---

## 39. A Deep Surrogate Model for Robust and Generalizable Long-Term Blast Wave Prediction

**arXiv ID:** 2602.18168 | [PDF](https://arxiv.org/pdf/2602.18168v1)

**作者:** Danning Jing `[一作]` (National University of Defense Technology), Jie Liu `[通讯]` (National University of Defense Technology)

**通讯引用:** 94269 | [OpenAlex ID](https://openalex.org/A5100454174)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种名为RGD-Blast的深度替代模型，专门用于高保真、长时程的爆炸波传播预测。

**💡 创新点**

创新点包括：1）多尺度卷积模块与ConvGRU递归单元的结合，有效捕捉全局流场与局部边界相互作用，降低自回归预测中的误差累积；2）动态-静态特征耦合机制，将时间变化的压强场与静态的源信息、建筑布局等融合，显著提升对分布外场景的泛化能力；3）结合物理约束的梯度损失，进一步提升预测精度。

**🔧 技术方法**

技术手段涵盖：多尺度卷积模块、CBAM注意力机制、ConvGRU时间建模、动态-静态特征耦合、物理约束损失（梯度约束）、滑动窗口自回归推理、Mish激活函数、PyTorch实现。

**📊 数据集**

使用基于CFD数值仿真的二维截面数据，共145个案例（随机布局50个、可变源50个、可变弹头45个），每个案例包含290个时间步，训练集与测试集按9:1划分。

**📈 对比分析**

与FNO、E3D-LSTM等基线模型在随机布局、可变源、可变弹头三种OOV场景下进行对比，评估指标为RMSE、MAPE、R²。RGD-Blast在280步长预测中平均RMSE仅0.0048、MAPE 1.93%，R²最小值0.8928；与传统数值方法相比，推理速度提升约189倍。

**⚠️ 局限性**

局限性在于目前仅验证二维平面场景，未扩展至三维城市网格；缺乏真实实验数据验证；在极端高能量或特殊气象条件下的泛化尚未充分测试；自回归推理仍存在末期误差累积。

---

## 40. DesignAsCode: Bridging Structural Editability and Visual Fidelity in Graphic Design Generation

**arXiv ID:** 2602.17690 | [PDF](https://arxiv.org/pdf/2602.17690v1)

**作者:** Ziyuan Liu `[一作]` (Peking University), Jiang Bian `[通讯]` (Microsoft)

**通讯引用:** 13740 | [OpenAlex ID](https://openalex.org/A5030951014)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了将图形设计视为可执行 HTML/CSS 代码合成的框架，并通过 Plan‑Implement‑Reflect 三阶段流水线实现高视觉保真度与结构可编辑性的图形设计生成。

**💡 创新点**

①采用可编辑的 HTML/CSS 代码作为表示，突破传统栅格或占位符的局限，支持丰富的视觉效果；②引入语义规划器从专业设计语料中提炼布局与内容规划，生成动态层次结构；③构建视觉感知反射机制，通过渲染反馈迭代优化代码，解决视觉冲突与可读性问题。

**🔧 技术方法**

使用大语言模型（如 Qwen3‑8B、GPT‑5、GPT‑4o）进行语义规划与代码生成；结合文本到图像模型与检索生成获取视觉资产；利用图像编辑模型与视觉评估网络（CLIP、MLLM）实现迭代视觉反射；通过 HTML/CSS 渲染实现可执行输出。

**📊 数据集**

以 Crello 设计数据集（含层级、文本属性等）训练模型，测试时使用 Crello 与 Broad（涵盖更广泛的设计类型）两套数据集，并使用公开的图像检索库进行资产获取。

**📈 对比分析**

在 Crello 与 Broad 两个基准上与 DeepSeek‑R1、Qwen3‑8B/30B、GPT‑5、OpenCOLE 等基线进行客观（Validity、Alignment、Readability、Clip）和主观（文本、图像、布局、颜色）指标对比。实验表明 DesignAsCode 在大多数指标上显著优于基线，尤其在视觉美感、布局与颜色方面领先；在用户研究中也获得最佳平均排名。

**⚠️ 局限性**

受限于当前 LLM 的规划精度，极高信息密度或极端视觉创意场景仍可能出现布局失衡或视觉冲突；对极端尺寸比例的自适应重排效果有限；实际部署需考虑算力与推理延迟；以及对训练数据中存在的文化偏见和创作版权风险的潜在影响。

---

## 41. Faster Training, Fewer Labels: Self-Supervised Pretraining for Fine-Grained BEV Segmentation

**arXiv ID:** 2602.18066 | [PDF](https://arxiv.org/pdf/2602.18066v1)

**作者:** Daniel Busch `[一作]`, Tobias Meisen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种两阶段训练策略，先在无BEV标签的条件下进行自监督预训练，再在半量BEV标签上进行细调，实现精细道路标记语义BEV分割；

**💡 创新点**

创新点在于使用可微分投影将BEV预测重新投射到图像平面，利用Mask2Former生成的相机视角伪标签进行自监督，并加入时间一致性损失以提升连续帧的鲁棒性；

**🔧 技术方法**

采用BEVFormer骨干网络、可微分渲染与重投影模块、交叉熵和时间一致性损失，辅以Mask2Former生成的伪标签；

**📊 数据集**

在nuScenes数据集上进行实验，仅使用其提供的道路边界、车道分隔符和人行横道三类语义地图；

**📈 对比分析**

与全监督基线对比，双阶段方法在miou上提升约+2.5个百分点，同时将BEV标签需求减半、训练时间缩至三分之一；

**⚠️ 局限性**

局限性包括伪标签与真实BEV标注的分布差异导致的误差，以及时间一致性损失可能引起的过度平滑现象。

---

## 42. Predicting Contextual Informativeness for Vocabulary Learning using Deep Learning

**arXiv ID:** 2602.18326 | [PDF](https://arxiv.org/pdf/2602.18326v1)

**作者:** Tao Wu `[一作]`, Adam Kapelner `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究构建了一个深度学习系统，用于自动识别有助于高中生词汇学习的上下文片段。

**💡 创新点**

创新点在于提出“Retention Competency Curve”评估指标，并结合监督学习与手工特征提升上下文信息质量预测。

**🔧 技术方法**

使用了 MPNet、Qwen3 Transformer 及多层感知机回归头等技术。

**📊 数据集**

数据集来自 DictionarySquared，包含约 67,807 条 42-65 字的上下文，并由 10 名 MTurk 工作者提供标注。

**📈 对比分析**

与以往的 Random Forest 方法相比，监督+手工特征模型在 70% 抛弃率下 Good-to-Bad 比率提升至约 440 倍，AUC 约 308。

**⚠️ 局限性**

限制包括对稀有词汇的泛化仍有限，手工特征增益有限且计算成本较高。

---

## 43. Visual Anthropomorphism Shifts Evaluations of Gendered AI Managers

**arXiv ID:** 2602.17919 | [PDF](https://arxiv.org/pdf/2602.17919v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 44. Flexible Coupler Array with Reconfigurable Pattern: Mechanical Beamforming and Digital Agent

**arXiv ID:** 2602.17710 | [PDF](https://arxiv.org/pdf/2602.17710v1)

**作者:** Xiaodan Shao `[一作]` (University of Waterloo), Shen `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出一种可移动被动耦合器的柔性耦合阵列，并设计了双时尺度的机械波束成形与天线位置优化框架；

**💡 创新点**

通过仅移动被动耦合器实现机械波束成形，提供额外辐射自由度，结合EM映射与深度学习“数字代理”在慢时尺度上快速预测最优sum‑rate，从而显著降低在线计算复杂度；

**🔧 技术方法**

使用柔性耦合阵列硬件、EM映射（channel knowledge map）、深度前馈网络（多层感知机）作为性能代理、凸松弛+分组四舍五入的机械波束成形、投影梯度上升求解天线位置；

**📊 数据集**

采用基于射线追踪的环境模型构建EM映射，生成10⁵条训练样本（含多路径统计与机械波束成形结果）作为离线数据，并用少量实时测量数据进行微调；

**📈 对比分析**

与传统两时尺度的内层-外层优化对比，数字代理实现95%最终sum‑rate，完成时间从约10.9分钟降至约32.4秒，且在集群核心分布漂移时仍保持鲁棒性；

**⚠️ 局限性**

对真实环境的依赖度高，需要预先建立精确EM映射，离线训练量大，对快速变化的小尺度统计仍需实时微调，且方案假设单轨道滑动，扩展到更复杂部署仍需验证；

---

## 45. Diffusing to Coordinate: Efficient Online Multi-Agent Diffusion Policies

**arXiv ID:** 2602.18291 | [PDF](https://arxiv.org/pdf/2602.18291v1)

**作者:** Zhuoran Li `[一作]` (Institute for Interdisciplinary Information Sciences), Longbo Huang `[通讯]` (Institute for Interdisciplinary Information Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种在线离策略多智能体强化学习框架 OMAD，利用扩散模型生成多模态动作，解决传统高斯策略表达受限的问题。

**💡 创新点**

核心创新在于：1）推导出可计算的联合熵下界以实现最大熵探索；2）构建中心化分布式价值网络以同步更新分布式扩散策略；3）实现温度自适应调节，自动平衡探索与利用。

**🔧 技术方法**

采用扩散策略（SDE反向采样）、中心化训练与分散执行（CTDE）框架、分布式 Q‑学习（CrossQ）和自监督熵下界估计。

**📊 数据集**

在多智能体粒子环境（MPE）与多智能体 MuJoCo（MAMuJoCo）两大基准数据集上进行实验。

**📈 对比分析**

与 HATD3、HASAC、MADPMD、MASDAC 等最先进基线对比，OMAD 在 10 个任务中实现 2.5–5× 的样本效率提升，且在最终性能上持续领先，展现出更快的收敛速度和更高的最终奖励。

**⚠️ 局限性**

局限性包括：1）扩散采样需要多步迭代，计算开销相对较高；2）在极大规模代理数或高维观察空间下的可扩展性尚未验证；3）对超参数（如步数、分布式 Q 支持区间）的敏感性仍需进一步研究。

---

## 46. Faster Parallel Batch-Dynamic Algorithms for Low Out-Degree Orientation

**arXiv ID:** 2602.17811 | [PDF](https://arxiv.org/pdf/2602.17811v1)

**作者:** Guy Blelloch `[一作]` (Carnegie Mellon University), Chase Hutton `[通讯]` (University of Maryland)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出三种并行批量动态低出度定向算法：摊销工作效率最优的并行算法；两阶段算法；重插算法，分别在工作量、最坏情况出度和时间深度上实现改进。

**💡 创新点**

创新点包括：
1) 首个摊销工作效率最优的并行批量动态低出度定向算法；
2) 通过天际线（skyline）与批量静态定向结合，设计两阶段和重插两种最坏情况算法，显著降低了之前 log⁹ 等大常数；
3) 设计批量计数器游戏和 (H,T)-bounded 框架，用于证明最坏情况的出度上界。

**🔧 技术方法**

采用的技术：并行工作-深度模型、随机半排序、Barenboim‑Elkin 静态定向、近似排序列表（roughly sorted list）、袋（bag）与双袋（pannier）数据结构、天际线选择、批量计数器游戏、潜能分析与 (H,T)-bounded 调用框架。

**📊 数据集**

论文主要以理论分析为主，未给出具体实验数据集。

**📈 对比分析**

与之前工作对比，摊销工作量提升到 O(b)（比 Liu 等人多了对数因子），最坏情况工作量从 O(b log⁹n) 降到 O(b√log n) 或 O(b log²n)；时间深度保持 poly‑log；最大出度从 O(c)→O(c log n) 或 O(c+log n)，比之前的 O(c log n) 更好或更接近理想值。

**⚠️ 局限性**

局限性：
1) 需要预先知道整个更新序列的最大 arboricity 上界；
2) 若使用确定性排序，工作量会乘上 O(log n) 因子；
3) 最坏情况的工作量仍高于摊销，且在极稀疏图上与 O(c) 仍有常数因子差距；
4) 批量计数器游戏未考虑 arboricity 约束，可能存在进一步改进空间；
5) 论文仅提供理论证明，未进行实验验证。

---

## 47. ScaleBITS: Scalable Bitwidth Search for Hardware-Aligned Mixed-Precision LLMs

**arXiv ID:** 2602.17698 | [PDF](https://arxiv.org/pdf/2602.17698v1)

**作者:** Xinlin Li `[一作]` (University of California, Los Angeles), Christina Fragouli `[通讯]` (University of California, Los Angeles)

**通讯引用:** 7327 | [OpenAlex ID](https://openalex.org/A5056591688)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为ScaleBITS的混合精度权重量化框架，用以在不增加推理开销的前提下，在LLM中实现极低位宽（2~4bit）压缩，显著提升模型精度。

**💡 创新点**

创新点包括：1）利用进阶量化参考点的梯度估计构建敏感度指标，保留了层级权重的相对重要性；2）在权重矩阵上做双向通道重排序，使高敏感度权重聚集在矩阵左上角；3）采用块级硬件对齐划分与分块精度分配，兼顾表达力与矩阵乘法效率；4）设计可扩展的贪心搜索近似算法，利用敏感度排序和批量更新实现全局精度分配，避免昂贵的逐步评估。

**🔧 技术方法**

技术手段包括：基于量化参考点的一阶Taylor展开的敏感度估计、双向通道重排序、块级硬件对齐划分、可扩展贪心搜索（包含批量增减与接受检查）、RTN统一量化、Triton实现的混合精度 GEMM。

**📊 数据集**

使用RedPajama数据集进行敏感度估计与校准，评估数据包括WikiText-2（perplexity）、WinoGrande、PiQA、HellaSwag、ARC-easy、ARC-challenge、BoolQ、MMLU（5-shot）、GSM8K、MBPP。

**📈 对比分析**

与统一精度RTN、GPTQ、GuidedQuant、SlimLLM等方法对比，ScaleBITS在2~4bit区间内平均提升0-36%（相对RTN）和0-13%（相对SOTA），在多种LLM（LLaMA-2/3、Gemma-2）上均表现出色；搜索时间比SlimLLM快3–6倍；混合精度块级实现无显著推理延迟。

**⚠️ 局限性**

局限性包括：仅针对权重量化，未考虑激活或向量量化；目前实现基于标量量化，向量量化需进一步集成；在极低位宽下仍可能出现精度下降，需要进一步微调或训练时适配；对硬件实现的依赖使得跨平台推广受限。

---

## 48. Causality by Abstraction: Symbolic Rule Learning in Multivariate Timeseries with Large Language Models

**arXiv ID:** 2602.17829 | [PDF](https://arxiv.org/pdf/2602.17829v1)

**作者:** Preetom Biswas `[一作]` (Arizona State University), K. Selçuk Candan `[通讯]` (Arizona State University)

**通讯引用:** 5664 | [OpenAlex ID](https://openalex.org/A5003070145)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种利用大型语言模型（LLM）在黑盒时间序列模拟器中提取可解释的因果规则的闭环框架，能够生成并验证符号时序规则；

**💡 创新点**

①将因果推断转化为含时间延迟的符号规则生成；②在无先验知识的黑盒模拟器上通过LLM自动抽取可解释规则；③采用闭环生成-评估-修正循环提升规则一致性与可验证性；

**🔧 技术方法**

大语言模型（GPT‑4.1）与提示工程；树结构Parzen估计器（TPE）求解多样化输入；UMAP+K‑Means降维聚类；符号时序逻辑规则语言；

**📊 数据集**

①COVID‑19 流行病学模拟器（测试率→感染数，2020‑03‑22 至 2020‑06‑30）；②EnergyPlus 商场能耗模拟（温度+太阳辐射→电力，1997‑01‑01 至 1997‑01‑05）；

**📈 对比分析**

通过三类实验（输入重构、规则因果性消融、规则泛化）与无规则/仅规则/仅上下文的基线对比；结果显示规则集显著降低输出误差，误差随迭代递减，且在不同基线上保持较好重构性能，证明了模型的有效性和泛化能力；

**⚠️ 局限性**

①不尝试恢复模拟器的真实生成机制，只提供符号近似规则；②未与现有符号学习器或可解释预测器进行对比；③对规则语言和上下文的依赖可能导致过拟合；④LLM提示与手工约束限制可扩展性与自动化程度。

---

## 49. Optimality Analysis of RSMA Degenerating to SDMA Under Imperfect SIC

**arXiv ID:** 2602.18077 | [PDF](https://arxiv.org/pdf/2602.18077v1)

**作者:** Xuejun Cheng `[一作]` (Shandong University), Bruno Clerckx `[通讯]` (Imperial College London)

**通讯引用:** 16157 | [OpenAlex ID](https://openalex.org/A5070530952)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在考虑了硬件失真和不完美 SIC 的情境下，严谨证明了当 SIC 效果变差至极限（残余干扰系数趋近于 1）时，RSMA 的最优方案会将公共信号波束向量设为零，进而退化为 SDMA。

**💡 创新点**

创新点在于从最优性角度提供了 RSMA 退化到 SDMA 的理论证明，而非仅凭经验或仿真；并阐明了残余干扰对 RSMA 结构选择的决定性影响。

**🔧 技术方法**

主要技术包括：系统建模（RIS‑辅助 RSMA 与硬件失真模型）、SINR 表达式推导、利用不完美 SIC 的残余干扰项、基于最优性假设的反证法证明。

**📊 数据集**

本文不涉及实验或仿真数据集，仅在理论上给出证明；若要验证，可采用标准的 MIMO 通信仿真平台（如 MATLAB/Simulink）配合随机 Rayleigh 信道。

**📈 对比分析**

比较方法为：在 δ_SIC 从 0 逐步升至 1 的过程中，观察 RSMA 与 SDMA 的总速率曲线。理论结果表明，随着 δ_SIC 接近 1，RSMA 的性能将逼近 SDMA，最终完全等价。

**⚠️ 局限性**

局限性包括：仅在理想化的残余干扰模型下给出证明，未考虑实际系统中其他干扰源；且仅证明了最优解退化的情况，未给出非最优策略的详细分析。

---

## 50. Aurora: Neuro-Symbolic AI Driven Advising Agent

**arXiv ID:** 2602.17999 | [PDF](https://arxiv.org/pdf/2602.17999v1)

**作者:** Lorena Amanda Quincoso Lugones `[一作]` (Florida International University), Janki Bhimani `[通讯]` (Florida International University)

**通讯引用:** 571 | [OpenAlex ID](https://openalex.org/A5011474644)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Aurora，一个结合检索增强生成、符号推理与正规化课程数据库的神经-符号学术指导系统，能够在保持政策合规的前提下，提供自然语言课程推荐及其可追溯的解释。

**💡 创新点**

创新点在于将检索增强的 LLM 与 Prolog 规则引擎、BCNF 规范化的课程知识库以及结构化的 5W+1H 链式思考提示进行模块化耦合，使得推荐既具备生成模型的语言流畅性，又具备符号推理的准确性、可解释性和可审计性；同时通过显式回退机制实现对超出范围请求的安全响应。

**🔧 技术方法**

技术细节包括：
- 采用 DeepSeek‑R1‑Distill‑Qwen‑7B（量化 4‑bit NF4）作为指令调优的 LLM；
- 使用 PostgreSQL 的 BCNF 关系模式存储课程、技能、先修、学位要求等数据；
- SWI‑Prolog 进行先修、学分上限和合规性检查；
- SQL 过滤器先把候选课程范围缩小；
- Retrieval‑Augmented Generation (RAG) 仅检索与查询相关的少量课程信息；
- 5W+1H 结构化提示作为链式思考模板，约束 LLM 只表述已验证的事实。

**📊 数据集**

实验使用合成的四个学位项目（CS、Data Science、IT、Product Management）的学生档案和机构课程目录，构造了 20 条评测查询；通过人工专家标注的答案作为基准。数据集已公开托管在 GitHub。

**📈 对比分析**

对比基准 Raw‑LLM（同一 LLM 但无检索和符号推理）得到：
- 语义一致度（余弦相似度）从 0.68 提升到 0.93（+36%）；
- 课程级别精确率、召回率和 F1 分别为 0.81‑0.83，Raw‑LLM 接近 0；
- 平均响应时间从 59.2 s 降至 0.71 s，亚秒级；
- 对超出范围请求始终触发回退模板，保证了安全性。

**⚠️ 局限性**

局限性包括：
- 评测基于合成学生数据，未覆盖真实用户行为、情感与动机；
- 仅验证了单一机构的课程结构，缺乏跨校泛化性；
- 未评估人机交互的信任感、易用性和实际学习效果；
- 只处理先修、学分和核心课程，未覆盖选修、多专业、学术例外等复杂情况；
- 对公平性与潜在偏差的系统性分析尚未完成。

---

## 51. Optimizing Graph Causal Classification Models: Estimating Causal Effects and Addressing Confounders

**arXiv ID:** 2602.17941 | [PDF](https://arxiv.org/pdf/2602.17941v1)

**作者:** Simi Job `[一作]` (University of Southern Queensland), Xin Wang `[通讯]` (University of Calgary)

**通讯引用:** 29086 | [OpenAlex ID](https://openalex.org/A5100327957)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种名为CCAGNN的混合因果图神经网络，能够通过注意力引导的特征解耦、互信息正则化以及对抗性干预来实现对图数据中的因果特征进行分离与强化，从而提升节点分类的鲁棒性与准确性。

**💡 创新点**

创新点在于：1) 引入可学习门控的注意力解耦机制，将节点特征拆分为因果与非因果两部分；2) 采用互信息最小化和条件互信息约束来促使两类特征解耦；3) 在训练中使用对抗性干预和多分支学习实现对因果关系的对抗性验证；4) 通过门控融合实现自适应特征组合，进一步提升鲁棒性。

**🔧 技术方法**

技术手段包括：图注意力网络（GATv2）用于特征编码；门控学习与注意力引导的噪声注入、特征遮掩和结构变更的图增强；双编码器互信息估计（对抗式正则化）；对抗性干预与多分支分类器；以及组合损失函数。

**📊 数据集**

实验使用了六个公开图数据集：Cora、Citeseer、PubMed（引用网络），Twitch（社交网络），Amazon-Computers 与 Amazon-Photo（同购网络）。

**📈 对比分析**

与基线方法（GCN、GAT、GraphSAGE）、因果方法（GCN-CAL、GAT-CAL、HebCGNN、GCN-ICL、GAT-ICL）以及基于干预的因果方法（ACE、ACE-GAT、CaNet）进行了比较。CCAGNN在所有六个数据集上均实现了最高F-score，尤其在Cora、Citeseer、PubMed上显著提升（约10–20%）。

**⚠️ 局限性**

局限性包括：1) 对大规模图的可扩展性尚未充分验证；2) 干预设计的手工设置可能需要领域知识；3) 互信息估计的样本效率与计算成本较高；4) 仅在节点分类任务上验证，其他图任务（如链接预测）尚未评估。

---

## 52. KPM-Bench: A Kinematic Parsing Motion Benchmark for Fine-grained Motion-centric Video Understanding

**arXiv ID:** 2602.17768 | [PDF](https://arxiv.org/pdf/2602.17768v1)

**作者:** Boda Lin `[一作]` (Kuaishou Technology), Meng Wang `[通讯]` (Kuaishou Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于运动分解的自动标注流水线，生成了 KPM‑Bench 数据集，并提出 MoPE 算法与 Mo‑Hall 指标来评估和降低视频描述中的运动幻觉。

**💡 创新点**

创新点包括：① 用三维姿态与运动学计算自动提取细粒度运动属性；② 设计 PaMoR 结构化语言表示；③ 引入 MoPE 的双语义句法解析实现精准运动属性抽取；④ 在 GRPO 训练中加入运动感知奖励；⑤ 创建包含 75k 细粒度视频-字幕对、38k 运动 QA 以及 215 条幻觉评估样本的综合基准。

**🔧 技术方法**

技术栈包括 RTMPose3D 进行 3D 姿态估计、基于速度/角速度与 FFT 的运动学分析、AMR+依存句法解析、GPT‑4.1 生成 PaMoR 与密集字幕、GRPO 强化学习、以及 GPT‑Score、Mo‑Hall 等评估指标。

**📊 数据集**

使用自建 KPM‑Bench（视频‑字幕、QA、幻觉集）作为主数据集，同时在 MoVid、Dream‑1K、MVBench、MotionBench 与 FAVOR 等公开基准上进行跨模型比较。

**📈 对比分析**

通过与闭源 VLM（GPT‑4.1、Gemini‑2.5Pro 等）以及开源 VLM（Tarsier2‑Recap、InternVideo‑2.5 等）在 GPT‑Score、BLEU、GPT‑Hall 与 Mo‑Hall 等指标上进行评测，KPM‑Bench 在 GPT‑Score 上领先约5–6个百分点，幻觉率显著降低，任务整体准确率达到 94.05%。

**⚠️ 局限性**

局限性包括：① 视角与镜头推理仍然是最难的维度；② MoPE 奖励虽能降低幻觉但对整体 NLG 评价略有轻微影响；③ 对极端运动类型或长时序视频的覆盖仍不充分；④ 需要大量算力进行全参数 SFT。

---

## 53. Predict to Skip: Linear Multistep Feature Forecasting for Efficient Diffusion Transformers

**arXiv ID:** 2602.18093 | [PDF](https://arxiv.org/pdf/2602.18093v1)

**作者:** Hanshuai Cui `[一作]` (School of Artificial Intelligence, Beijing Normal University), Weijia Jia `[通讯]` (School of Artificial Intelligence, Beijing Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了一种训练‑free 的 PrediT 框架，用于加速 Diffusion Transformers 的推理，采用线性多步方法预测特征并避免重复计算。

**💡 创新点**

创新点在于将特征预测视作线性多步数值积分问题，使用 Adams‑Bashforth 预测器 + Adams‑Moulton 校正器，并结合动态步长调节机制，显著降低高动态区域误差。

**🔧 技术方法**

核心技术包括 Adams‑Bashforth / Adams‑Moulton 数值方法、特征预测与纠正、动态步长调节、FlashAttention 等。

**📊 数据集**

在 FLUX.1（ImageNet、DiffusionDB）、HunyuanVideo（VBench）、DiT‑XL/2（ImageNet）等数据集上进行实验。

**📈 对比分析**

与现有缓存与预测方法（Δ‑DiT、FORA、TeaCache、TaylorSeer、AB‑Cache 等）比较，PrediT 在保持或提升生成质量的前提下，分别实现 FLUX 4.28×、HunyuanVideo 3.28×、DiT‑XL/2 2.48× 的最高加速，比对手多出 2–5 倍。

**⚠️ 局限性**

仍需手动调节阈值/参数，对极高动态区域或长序列的预测误差可能仍存在累积，且仅为训练‑free 框架，无法进一步压缩模型体积。

---

## 54. Symfrog-512: High-Capacity Sponge-Based AEAD Cipher (1024-bit State)

**arXiv ID:** 2602.17900 | [PDF](https://arxiv.org/pdf/2602.17900v1)

**作者:** Victor Duarte Melo `[一作]` `[通讯]` (Independent Researcher), Victor Duarte Melo (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出并实现了 SymFrog‑512，一种基于 1024 位自定义置换和双工 sponge 的 AEAD 加密方案，支持 256 位密钥、Nonce 和标签。

**💡 创新点**

创新点在于：① 采用 512 位速率/容量的双工模式并结合自定义 1024 位置换；② 通过自定义输出变换（含 SplitMix64 终端化）增强扩散；③ 在文件头加入键控身份认证标签，实现早期拒绝和完整性校验；④ 在理想置换模型下给出完整安全分析。

**🔧 技术方法**

技术手段包括：双工 sponge 与域分离；自定义置换（混合层、Chi、Kick、旋转+P‑box）；Argon2id 密钥派生；SplitMix64 输出终端化；理想置换模型安全证明；基准测试与扩散实验。

**📊 数据集**

使用的“数据集”是多长度测试向量（0~1,048,583 字节），并提供 SHA‑256 校验和作为回归文件；未涉及真实数据集。

**📈 对比分析**

性能对比：置换 435.1 ns/次（200k 次迭代），AEAD 加密 131.7 MiB/s（64 MiB 缓冲区，忽略 I/O 与 KDF）。实验表明 4~5 轮即可达到 512‑bit 的平均 Hamming 距离，表明扩散良好。

**⚠️ 局限性**

局限性：仅在理想置换模型下给出安全界限，未对置换本身做严格分析；不具备 nonce‑误用抵抗；实现安全高度依赖代码质量；性能受平台差异影响，需进一步多架构评估。

---

## 55. Hardware-Friendly Input Expansion for Accelerating Function Approximation

**arXiv ID:** 2602.17952 | [PDF](https://arxiv.org/pdf/2602.17952v1)

**作者:** Hu Lou `[一作]` (Northwest Institute of Nuclear Technology), Jia-Rui Zhang `[通讯]` (Northwest Institute of Nuclear Technology)

**通讯引用:** 298 | [OpenAlex ID](https://openalex.org/A5100363743)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出通过在输入空间加入常数扩展的方式打破神经网络参数对称性，从而加速函数逼近。

**💡 创新点**

创新点在于使用硬件友好的输入维度扩展实现对称性破坏，既不增加参数也无推理成本。

**🔧 技术方法**

技术方法为常数填充扩展输入向量、标准MLP、LBFGS优化、均方误差评估。

**📊 数据集**

使用十个一维函数基准集，包括光滑、分段、尖峰、高频和分形函数。

**📈 对比分析**

对比标准单维输入模型和宽度扩展模型，5维常数π扩展在平均12%迭代下降、66.3% MSE下降上表现最佳。

**⚠️ 局限性**

局限在于对不连续或不可微函数的收敛速度提升有限，且需针对不同域挑选常数。

---

## 56. PINEAPPLE: Physics-Informed Neuro-Evolution Algorithm for Prognostic Parameter Inference in Lithium-Ion Battery Electrodes

**arXiv ID:** 2602.18042 | [PDF](https://arxiv.org/pdf/2602.18042v1)

**作者:** Karkulali Pugalenthi `[一作]` (Singapore Institute of Manufacturing Technology, Agency for Science, Technology and Research), Chinchun Ooi `[通讯]` (Institute of High Performance Computing, Agency for Science, Technology and Research)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了PINEAPPLE框架，用于从电压-时间曲线无创、实时地推断锂离子电池内部状态参数（如扩散系数、几何系数等）并跟踪其随循环的演变。

**💡 创新点**

创新点在于将元学习的物理信息神经网络（PINN）与进化搜索相结合，形成“Baldwinian”元学习PINN，可实现零射击快速预测，并通过CMA-ES对参数进行高效、鲁棒的逆推理；该框架实现了大幅加速（≈10倍）且保持高精度的物理一致性。

**🔧 技术方法**

使用的技术包括：物理信息神经网络（PINN）、Baldwinian元学习、Tikhonov正则化的线性最小二乘微调、进化搜索算法（CMA-ES）以及JAX后端的高性能自动微分。

**📊 数据集**

利用公开的CALCE LCO‑Graphite电池数据集（CX2-34/36/37/38四组循环曲线），并在合成数据上进行验证。

**📈 对比分析**

与传统数值求解器（PyBaMM）和基于等效电路模型的对比表明，PINEAPPLE在相同网格下误差相当但计算时间缩短≈10倍；逆推结果在合成与真实曲线上都能恢复参数随循环的趋势，平均误差低于0.01%且能实时完成。

**⚠️ 局限性**

局限性包括：仅使用单粒子模型（SPM），忽略电解质、电荷分布、热效应等；逆推问题在晚期循环易出现多解，识别度下降；对不同化学体系和更复杂工况的验证尚不足。

---

## 57. Stop Saying "AI"

**arXiv ID:** 2602.17729 | [PDF](https://arxiv.org/pdf/2602.17729v1)

**作者:** Nathan G. Wood `[一作]` (Institute of Air Transportation Systems), Daniel Kloock-Schreiber `[通讯]` (Institute of Air Transportation Systems)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性梳理并提出军事领域人工智能的松散分类，阐述不同AI系统的功能、应用场景及其面临的独特挑战，并主张在讨论中应避免使用泛化的“AI”术语，而是针对具体系统展开分析

**💡 创新点**

强调人工智能不是单一技术，而是一类通用赋能技术；提出通过对军用AI系统进行细分和针对性讨论，可减少概念混淆、误判与不必要的法规制定，从而提升决策的准确性与可操作性

**🔧 技术方法**

文献综述与理论分析，无具体算法实现；采用逻辑梳理、案例对比与风险评估等方法论

**📊 数据集**

无公开使用的数据集，主要依赖公开报道、官方文件、学术论文与行业案例等二手资料

**📈 对比分析**

未进行实验或性能评估；比较主要基于案例描述与已知技术特性进行定性讨论，未给出量化指标

**⚠️ 局限性**

1）讨论范围有限，未覆盖所有可能的AI系统与应用；2）依赖公开信息，可能忽略机密或最新技术；3）缺乏实证验证，结论多为理论性或主观推测；4）未系统评估不同技术对军事伦理、法律与操作的具体影响

---

## 58. ROCKET: Residual-Oriented Multi-Layer Alignment for Spatially-Aware Vision-Language-Action Models

**arXiv ID:** 2602.17951 | [PDF](https://arxiv.org/pdf/2602.17951v1)

**作者:** Guoheng Sun `[一作]` (University of Maryland), Ang Li `[通讯]` (University of Maryland)

**通讯引用:** 4370 | [OpenAlex ID](https://openalex.org/A5100413657)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种残差导向的多层表示对齐框架（ROCKET），通过共享投影器将3D视觉基础模型的空间知识迁移到2D预训练的视觉‑语言‑动作模型上，显著提升机器人操纵任务中的空间推理能力。

**💡 创新点**

核心创新在于①将多层对齐视为残差流的对齐，使用单一共享投影器以降低梯度冲突；②引入马特罗什卡式稀疏激活机制，让深层对齐拥有更大容量，平衡浅层与深层的对齐损失；③给出了梯度冲突的理论分析并验证共享投影器能够提升梯度一致性。

**🔧 技术方法**

技术上实现了多层对齐的共享投影器（两层MLP），马特罗什卡式稀疏激活，基于残差动态的梯度冲突分析，采用LoRA微调和损失权重λ调节对齐与任务学习的平衡。

**📊 数据集**

主要使用LIBERO、LIBERO‑Plus、RoboTwin 2.0等机器人操纵基准数据集进行评估，同时在不同规模的VLA骨干（OpenVLA‑7B、PI0、PI0.5等）上进行实验。

**📈 对比分析**

与单层对齐、显式3D输入以及先前最先进的VLA模型相比，ROCKET在LIBERO上达到了98.5% 的平均成功率，仅需约4%的计算预算；在LIBERO‑Plus和RoboTwin 2.0等更具挑战性的基准上也表现出优于或相当于现有最优方法的性能。

**⚠️ 局限性**

局限性包括：需要预先训练好的3D视觉基础模型作为教师；对齐层的选取仍需经验或启发式策略；在极端数据稀缺或高噪声环境下对齐效果可能受限，且共享投影器在非常深的网络中可能无法完全捕捉细粒度空间特征。

---

## 59. Operational Agency: A Permeable Legal Fiction for Tracing Culpability in AI Systems

**arXiv ID:** 2602.17932 | [PDF](https://arxiv.org/pdf/2602.17932v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 60. Certified Learning under Distribution Shift: Sound Verification and Identifiable Structure

**arXiv ID:** 2602.17699 | [PDF](https://arxiv.org/pdf/2602.17699v1)

**作者:** Chandrasekhar Gokavarapu `[一作]` (Government College), S. R. Bhargava `[通讯]` (Government College)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种统一框架，将分布偏移下的风险上界与可验证的神经网络安全性通过对偶证书关联起来，并通过可识别的可解释结构约束来降低风险与验证复杂度。

**💡 创新点**

创新点在于：① 给出可计算的分布偏移风险上界；② 证明对偶证书既能验证模型安全性，又能给出风险上界；③ 引入可识别的可解释类，实现结构化的敏感性校正与维度约简。

**🔧 技术方法**

使用 Wasserstein DRO 对偶、凸松弛（CROWN/LiRPA）求解验证的 LP、对偶约束推导风险界限、可解释模型的可识别性定理。

**📊 数据集**

论文以理论推导为主，实验示例采用一维线性、2 层 ReLU 分类器和稀疏可加模型的合成数据；未给出真实数据集。

**📈 对比分析**

论文未给出与现有方法的数值对比；理论上指出不完整验证可在多层网络上实现线性时间，完整验证则呈指数复杂度。

**⚠️ 局限性**

局限性包括：需要满足协变量偏移假设、Lipschitz 常数可证性；对偶证书仅在可识别的可解释类内可获得；完整验证在最坏情况下指数复杂；缺乏实测性能评估。

---

## 61. VeriSoftBench: Repository-Scale Formal Verification Benchmarks for Lean

**arXiv ID:** 2602.18307 | [PDF](https://arxiv.org/pdf/2602.18307v1)

**作者:** Yutong Xin `[一作]` (University of Texas at Austin), Işil Dillig `[通讯]` (University of Texas at Austin)

**通讯引用:** 5100 | [OpenAlex ID](https://openalex.org/A5006424908)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建并评估了一个500条Lean4证明任务的仓库级基准，涵盖23个开源验证项目。

**💡 创新点**

首次将验证任务放入与实际代码仓库相同的跨文件依赖上下文中，评估了模型对仓库级上下文的适应性。

**🔧 技术方法**

采用了大型语言模型（GPT‑5.2、Claude‑Opus‑4.5、Gemini‑3‑Pro）和专业定理证明器（Goedel‑Prover‑V2、Aristotle），并使用生成‑检查‑修复循环。

**📊 数据集**

基准数据集为VeriSoftBench，包含从23个Lean仓库提取的500个任务，分为完整仓库和精炼上下文两种配置。

**📈 对比分析**

对照在精炼上下文和完整仓库上下文下的成功率，Gemini‑3‑Pro在精炼上下文下达41%成功率，在完整仓库下34.8%，Aristotle在子集下达到69%。

**⚠️ 局限性**

局限性在于当前模型仍难以处理多跳依赖链和大量本地定义，精炼上下文的收益有限，需进一步改进检索与层级推理能力。

---

## 62. Detecting Contextual Hallucinations in LLMs with Frequency-Aware Attention

**arXiv ID:** 2602.18145 | [PDF](https://arxiv.org/pdf/2602.18145v1)

**作者:** Siya Qi `[一作]` (King's College London), Lin Gui `[通讯]` (King's College London)

**通讯引用:** 7125 | [OpenAlex ID](https://openalex.org/A5062168574)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出频率感知的注意力分析方法，用高频分量检测LLM生成中的语境幻觉（hallucination）

**💡 创新点**

创新点在于把注意力分布视作离散信号，利用频域/小波/拉普拉斯等高通滤波器提取局部快速波动的高频能量作为幻觉指标，弥补了传统基于注意力分配或熵的粗粒度度量不足

**🔧 技术方法**

核心技术包括离散傅里叶变换（DFT）、离散小波变换（DWT）、离散拉普拉斯算子等高通滤波操作，随后用L2能量度量高频分量并作为特征输入轻量化线性分类器

**📊 数据集**

在RAGTruth（QA、Data-to-Text、Summarization）和HallucRAG（QA）两个检索式生成幻觉基准上评估，覆盖LLaMA-7B、LLaMA-13B、Mistral-7B三大模型

**📈 对比分析**

与验证式、内部表征式以及传统注意力式基线（如SelfCheckGPT、RefChecker、EigenScore、ReDeEP、Lookback-Lens、注意力方差/熵）对比，频率感知方法在token级别和span级别均实现AUROC提升，最高可达比Lookback-Lens提升约10%（在summarization任务上）

**⚠️ 局限性**

局限性包括：只利用单层线性分类器可能无法捕获更复杂关系；高频特征对模型架构或训练数据的泛化性尚未完全验证；方法仍受检索质量影响，且在多模态或长文本场景下表现未知

---

## 63. Collaborative Processing for Multi-Tenant Inference on Memory-Constrained Edge TPUs

**arXiv ID:** 2602.17808 | [PDF](https://arxiv.org/pdf/2602.17808v1)

**作者:** Nathan Ng `[一作]` (University of Massachusetts Amherst), Prashant Shenoy `[通讯]` (Indian Institute of Science)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于分析排队模型的自适应多租户TPU‑CPU协同推理系统，针对内存受限的Edge TPU设备降低推理延迟。

**💡 创新点**

创新点在于将模型分区、CPU核心分配与内存交换开销统一到一个非线性整数排队模型，并通过贪婪梯度算法在线自适应调整，首次针对多租户场景全局优化。

**🔧 技术方法**

使用排队理论（M/G/1、M/D/k）、权重缺失概率建模、TensorFlow Lite、Edge TPU编译器、GraphSurgeon、量化与一次性性能概型等技术。

**📊 数据集**

采用九个常用卷积模型（SqueezeNet、MobileNetV2、EfficientNet、MnasNet、GPUNet、DenseNet201、ResNet50V2、Xception、InceptionV4），主要基于ImageNet预训练权重。

**📈 对比分析**

与Edge TPU编译器、阈值分区及无交换版基线进行对比，实验在单租户和多租户、不同TPU利用率下评估；该系统平均降低单租户延迟63.8%，多租户77.4%，动态工作负载下提升至75.1%。

**⚠️ 局限性**

局限性包括：仅针对单个Edge TPU设备、对权重缺失概率采用保守近似、主要针对卷积网络、需预先剖分分区且动态切换有限、未考虑跨多TPU的协同与更复杂的内存置换策略。

---

## 64. Pricing with a Hidden Sample

**arXiv ID:** 2602.18038 | [PDF](https://arxiv.org/pdf/2602.18038v1)

**作者:** Zhihao Gavin Tang `[一作]` (Shanghai University of Finance and Economics), Shixin Wang `[通讯]` (Georgia Institute of Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在卖家仅观察到一个隐藏样本、买家知道完整价值分布的情形下，如何实现鲁棒定价。作者提出“隐藏定价机制”，利用买家的信息优势，将传统需要分布统计量的价格策略（如均值、L^η-范数、CVaR）转化为仅凭单一样本即可实现的策略，并对α-正则（尤其是MHR）分布给出了最优的单调隐藏定价方案及其近似比。

**💡 创新点**

创新点包括：1）证明单一隐藏样本足以实现所有凹性价格策略的性能保证；2）提出将最坏情况分布降维到仅两个参数的技术，使得无限维的最坏分布优化变为可解的参数化优化；3）得到MHR分布下约0.79的最优近似比，并给出对所有凹性及更一般先验无关机制的上界；4）将基于统计量的鲁棒定价与基于样本的鲁棒定价统一为一个框架。

**🔧 技术方法**

主要技术手段包括：先验无关机制设计、正确评分规则的凹函数表示、单调随机占优、两参数分布族的构造、线性规划与对偶性、Lambert W函数的解析表达、数值优化与计算机辅助验证。

**📊 数据集**

本文完全为理论研究，没有使用任何实测数据集；所有结论均通过数学推导与数值实验（计算机辅助求解）得到。

**📈 对比分析**

通过与以往基于单样本的定价（最大约0.5-0.68）和基于分布统计的鲁棒定价（已知均值等）进行对比。隐藏定价在MHR分布下实现约0.79的近似比，超过之前已知的单样本确定性比例，并且其对凹性价格策略的上界为0.801，对所有先验无关机制的上界为0.838，说明其性能接近最优。

**⚠️ 局限性**

局限性包括：1）仅考虑单一买家单件商品的情形；2）要求买家完全了解价值分布，实际可能不满足；3）仅证明对单调凹性策略的性能；4）对多样本、多买家或多件商品的推广仍待研究；5）部分结果需数值计算，闭式解析有限。

---

## 65. "How Do I ...?": Procedural Questions Predominate Student-LLM Chatbot Conversations

**arXiv ID:** 2602.18372 | [PDF](https://arxiv.org/pdf/2602.18372v1)

**作者:** Alexandra Neagu `[一作]` (Imperial College London), Rhodri Nelson `[通讯]` (Imperial College London)

**通讯引用:** 137 | [OpenAlex ID](https://openalex.org/A5003534520)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文分析了学生在与大型语言模型驱动的教育聊天机器人对话中提出的问题类型，并评估了使用LLM作为分类器的可行性。

**💡 创新点**

创新点在于将多种既有问题分类方案迁移到LLM对话情境，并开发可扩展的LLM评估流程，以大规模自动标注学生问题。

**🔧 技术方法**

采用了11种不同规模的LLM（OpenAI、Google Gemini、Llama等）和三名人工评标者进行多方案分类，评估使用Fleiss κ、Gwet AC1等一致性指标。

**📊 数据集**

使用了两组 STEM 学习场景的数据：41名工程二年级学生在自学平台上的对话（基于gemini-2.0-flash）和203名计算机科学学生在课程作业中使用GPT‑4o‑mini的公开对话。

**📈 对比分析**

通过交叉验证的 leave‑one‑out 分析、内部一致性检验以及与人工“真值”比较，LLM评标在各方案间达到了中等到较好的一致性（κ≈0.55–0.57，AC1≈0.60–0.64），但相较人类评标一致性略低。

**⚠️ 局限性**

局限在于现有分类方案无法完整覆盖学生与LLM的多元交互，且单标签约束导致对复合式问题的歧义和低认知负荷请求被误判，未能深入探讨对话上下文对意图识别的影响。

---

## 66. Improved Algorithms for Clustering with Noisy Distance Oracles

**arXiv ID:** 2602.18389 | [PDF](https://arxiv.org/pdf/2602.18389v1)

**作者:** Pinki Pradhan `[一作]` (National Institute of Science Education and Research), Ragesh Jaiswal `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 802 | [OpenAlex ID](https://openalex.org/A5075959430)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了在弱强oracle模型下的k‑means和k‑center聚类算法，显著降低强oracle查询量。

**💡 创新点**

关键创新是将k‑means++适配到弱强oracle，并用中位数估计距离；k‑center采用球切割法。

**🔧 技术方法**

使用弱强oracle、Chernoff界、中位数估计、D^2分布、球切割以及基于弱oracle的距离上界技术。

**📊 数据集**

实验数据包括合成的SBM数据和MNIST（SVD、t‑SNE）嵌入。

**📈 对比分析**

与Bateni等先前方法对比，查询量减少至少61%（k‑means）和99%（t‑SNE k‑means），聚类质量保持常数近似。

**⚠️ 局限性**

限制包括δ≤1/3的假设、对球半径选择的依赖，以及仅实现常数近似而非最优解。

---

## 67. A Simple yet Effective Negative Sampling Plugin for Constructing Positive Sample Pairs in Implicit Collaborative Filtering

**arXiv ID:** 2602.18206 | [PDF](https://arxiv.org/pdf/2602.18206v1)

**作者:** Jiayi Wu `[一作]` (Beijing Institute of Technology), Guoren Wang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 7295 | [OpenAlex ID](https://openalex.org/A5054991337)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在隐式协同过滤中，作者提出一种负采样插件PSP-NS，旨在通过构造高质量的正样本对来增强正向监督信号，从而提升模型性能。

**💡 创新点**

创新点在于将负采样视角转向正样本构造：使用SVD提取全局交互模式与本地信号融合生成加权二分图，利用边权进行复制式重加权构造正样本对；并引入基于用户活跃度的对数加权方案，显著缓解用户活跃度偏差。

**🔧 技术方法**

核心技术包括：随机SVD降维、适应性邻居选择、加权二分图构建、复制式正样本重加权、基于用户活跃度的对数加权，以及标准的BPR/LightGCN等负采样框架。

**📊 数据集**

在Pinterest、Yelp、MovieLens-1M和Epinions四个真实隐式数据集上进行实验。

**📈 对比分析**

与五种负采样方法（RNS、DNS、MixGCF、DNS(M,N)、AHNS）和五种正样本去噪方法（T-CE、R-CE、DeCA、DCF、PLD）进行对比，PSP-NS在Recall@k、Precision@k等指标上普遍提升，最高可达约30%提升（如Yelp Recall@30+32%）。同时与MF、GNN等模型集成后亦保持性能提升。

**⚠️ 局限性**

主要局限包括：需要预先离线计算SVD和加权图，增加一次性开销；对超参数（q、s、a）的敏感性需要调参；在极度稀疏或极大规模数据集上，SVD与邻居选择可能导致额外计算负担，且在极度极端用户活跃度分布时对数加权效果可能受限。

---

## 68. It does not matter how you define locally checkable labelings

**arXiv ID:** 2602.18188 | [PDF](https://arxiv.org/pdf/2602.18188v1)

**作者:** Antonio Cruciani `[一作]` (Aalto University), Jukka Suomela `[通讯]` (Aalto University)

**通讯引用:** 2462 | [OpenAlex ID](https://openalex.org/A5025555126)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文证明了 LCL（局部可检验标记）问题在极度受限的 RE（节点-边可检验）形式上与一般 LCL 等价，且在所有相关分布式模型中保持相同的时间复杂度（除非进入对称性破除区域）。

**💡 创新点**

创新点在于：①展示了 LCL 的定义对极端限制（无输入、3‑正则图、半边标签、r=½ 检查半径）几乎不影响其表达能力；②证明了所有“反直觉”性质（量子优势、共享随机/量子状态、可计算性依赖、非标准时间复杂度、不可判定性）在 RE 形式下仍然成立；③提供了一套完整的本地归约链（A→B→D→E），展示了如何从一般 LCL 逐步转化为 RE 形式。

**🔧 技术方法**

采用了多种技术：本地归约、对称性破除（距离‑c 颜色）、PN‑可检验问题的构造、输入去除技巧（使用 gadget 序列与距离‑c 颜色），以及半边标签编码半径‑r 视图以实现 RE 形式。论文中还利用了图的覆盖映射、树展开视图和局部视图一致性等概念。

**📊 数据集**

本文为理论性研究，不涉及实验或数据集；所有结论均来自严格的形式化证明与构造。

**📈 对比分析**

通过证明 LCL 与 RE 之间的等价，论文展示了在所有考虑的分布式模型（LOCAL、CONGEST、SLOCAL、ONLINE‑LOCAL、动态‑LOCAL、量子‑LOCAL、非信号分布等）中，算法在 LCL 与 RE 间的时间复杂度相差至多一个 O(log* n) 的附加项，表明两者在实际表现上基本相同。

**⚠️ 局限性**

局限性包括：①在对称性破除的“极限”下（如 o(log* n) 轮）是否存在差异尚未解决；②对树图或非简单图（含多重边/自环）的处理结果尚未给出；③对更强限制（如输入标签或不同正则度）的完整表达性仍是开放问题。

---

## 69. GPU Memory and Utilization Estimation for Training-Aware Resource Management: Opportunities and Limitations

**arXiv ID:** 2602.17817 | [PDF](https://arxiv.org/pdf/2602.17817v1)

**作者:** Ehsan Yousefzadeh-Asl-Miandoab `[一作]` (IT University of Copenhagen), Pınar Tözün `[通讯]` (IT University of Copenhagen)

**通讯引用:** 806 | [OpenAlex ID](https://openalex.org/A5061990490)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了深度学习训练任务的 GPU 内存和利用率估计方法，并提出基于深度学习的 GPUMemNet 与 GPUUtilNet。

**💡 创新点**

创新点在于将估计任务离散化为分类问题，构建大规模合成数据集，并公开 GPUMemNet/GPUUtilNet 框架，实现对多种网络架构的高精度预测。

**🔧 技术方法**

采用深度神经网络（MLP 与 Transformer）以及传统的分类模型，配合符号张量 FakeTensor 与 Horus 分析公式进行对比。

**📊 数据集**

使用自定义合成数据集（MLP、CNN、Transformer）以及公开的真实模型（ResNet、EfficientNet、BERT 等）进行评估。

**📈 对比分析**

实验显示 GPUMemNet 在 MLP 上达到 97% 以上精度，CNN/Transformer 在 8 GB 级别 81–88% 之间，整体优于 Horus 与 FakeTensor；利用率估计准确率相对较低。

**⚠️ 局限性**

主要限制包括对未见网络层的泛化能力不足、对 GPU 体系结构的依赖、以及利用率估计缺乏可区分性和加法性假设。

---

## 70. Downwash-aware Configuration Optimization for Modular Aerial Systems

**arXiv ID:** 2602.18344 | [PDF](https://arxiv.org/pdf/2602.18344v1)

**作者:** Mengguang Li `[一作]` (Technische Universität Darmstadt), Heinz Koeppl `[通讯]` (Technische Universität Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种生成和优化多模块无人机组装配置的框架，并通过非同构拓扑枚举、非线性规划与下洗流碰撞约束实现任务特定的最优配置。

**💡 创新点**

创新点在于：①将下洗流约束显式引入组装设计；②利用非同构树拓扑枚举并在大规模情况下采用采样加速；③在单配置层面用非线性优化同时优化连接角与控制输入，最小化整体控制量。

**🔧 技术方法**

采用的技术包括：基于矩阵的模块连接拓扑枚举、BFS 组装几何求解、利用胶囊模型约束下洗流、非线性优化（MATLAB fmincon/Ipopt）求解最小控制成本、几何控制器实现跟踪。

**📊 数据集**

数据集主要是物理仿真随机产生的 10 个任务力矩向量（在 0.9–1.1 nmg 的 z 方向范围内），以及基于标准四旋翼模块（质量 0.24 kg、推力常数 3.87e-7、转矩常数 1.06e-8）的参数；实验使用真实的 2–60 模块组装在 VICON 轨迹捕捉平台上进行。

**📈 对比分析**

通过在每个非同构配置中求解控制成本，比较所有可行配置并选取成本最小者；在仿真中 30/60 模块组装能在 3° 内跟踪圆轨迹；实验中的 2 模块组装成功完成水平推力任务，证明方法可行且性能良好。

**⚠️ 局限性**

局限性包括：非线性规划局部最优；在大规模模块数下仍需采样近似，完整性下降；下洗流模型简化为胶囊，未考虑复杂气动耦合；仅考虑无环树结构，无法处理含闭环的连通网络。

---

## 71. G-LoG Bi-filtration for Medical Image Classification

**arXiv ID:** 2602.18329 | [PDF](https://arxiv.org/pdf/2602.18329v1)

**作者:** Qingsong Wang `[一作]` (Jilin University), Cailing Yao `[通讯]` (Jilin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了基于高斯-拉普拉斯(LoG)双滤波的 G-LoG bi-filtration，用于医学影像分类。

**💡 创新点**

创新点在于将 Gaussian 与 Laplacian-of-Gaussian 组合成双参数滤波器，能够捕获更丰富的边界和纹理信息，并证明其相干性对最大范数的稳定性。

**🔧 技术方法**

采用多参数持久同调、卷积与 Gaussian 平滑、LoG 变换、近似多参数持久模块、向量化为多参数持久图像，再用 MLP 训练分类。

**📊 数据集**

使用 MedMNIST（v2）2D 与 3D 医学图像数据集进行实验。

**📈 对比分析**

与深度学习基线（ResNet、AutoML、AutoKeras 等）以及单参数持久同调进行对比，G-LoG bi-filtration 在多数数据集上显著优于单参数方法，性能可与复杂深度模型相当。

**⚠️ 局限性**

局限性包括对参数（σ）的选择敏感，三维卷积生成时间较长，且在某些数据集（如 ChestMNIST、DermaMNIST）仍未超过最优深度模型。

---

## 72. SeedFlood: A Step Toward Scalable Decentralized Training of LLMs

**arXiv ID:** 2602.18181 | [PDF](https://arxiv.org/pdf/2602.18181v1)

**作者:** Jihun Kim `[一作]` (POSTECH), Namhoon Lee `[通讯]` (POSTECH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于种子可重构零阶梯度的去中心化训练框架 SeedFlood，通过全局洪泛传播实现对大模型的高效协同学习；

**💡 创新点**

创新点在于将通信成本降为近零（仅传输随机种子和标量），并用洪泛而非传统 gossip 机制实现全局一致性，同时通过共享低秩子空间加速梯度聚合；

**🔧 技术方法**

核心技术包括共享随机数生成器实现种子重构、洪泛式全局传播、低秩子空间的零阶梯度估计（Subspace LOZO）以及延迟洪泛调优；

**📊 数据集**

实验使用 OPT 系列语言模型（1B、125M、2.7B 等）在 SuperGLUE、SST‑2、RTE、BoolQ 等 NLP 任务的数据集上；

**📈 对比分析**

与 DSGD、DZSGD、ChocoSGD、LoRA 等基线相比，SeedFlood 在通信量上几乎不依赖模型大小（仅几百 KB），在稀疏网络上性能优于传统 gossip，且在 128 节点大规模场景下接近甚至超过一阶方法；

**⚠️ 局限性**

主要局限是相对一阶方法存在 4‑6% 的准确率差距，且对子空间维度和刷新周期敏感；在极低洪泛步数（k=1,2）时易出现信息延迟导致性能退化。

---

## 73. Strange Undercurrents: A Critical Outlook on AI's Cultural Influence

**arXiv ID:** 2602.17841 | [PDF](https://arxiv.org/pdf/2602.17841v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 74. Dichotomy for Axiomatising Inclusion Dependencies on K-Databases

**arXiv ID:** 2602.18390 | [PDF](https://arxiv.org/pdf/2602.18390v1)

**作者:** Miika Hannula `[一作]` (Institute of Computer Science, University of Tartu), Jonni Virtema `[通讯]` (University of Glasgow)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

本文研究了在元组标注为正交换半群（即正交换单体）的注释型数据库上，包含依赖（IND）的蕴含问题，并给出了完整的公理化分类；

**💡 创新点**

创新点在于提出了一个新的⊕-追踪（chase）变体，能够在单体自然序非全序或非正时进行推理，并利用该追踪实现对弱可吸收和弱可消除单体两种情况的完备性证明；

**🔧 技术方法**

主要技术包括：单体的代数性质（弱可吸收、弱可消除、可吸收序、周期性）分析、⊕-追踪算法、归结与公理化推理（弱对称、公理化归结、平衡公理等）；

**📊 数据集**

论文为理论性质研究，并未使用具体实验数据集；

**📈 对比分析**

没有实验比较，论文的“性能”体现在证明的完备性与可终止性：在弱可消除单体上，⊕-追踪必然终止，且产生唯一的标准模型；在弱可吸收单体上则可通过传统追踪得到满足性模型；

**⚠️ 局限性**

局限性包括：仅讨论正交换单体；⊕-追踪在自然序非全序时需要额外假设；对非可吸收且非可计数可吸收单体的情况仍需特殊处理；实际应用中需验证单体性质是否满足理论假设。

---

## 75. Programming Backpropagation with Reverse Handlers for Arrows

**arXiv ID:** 2602.18090 | [PDF](https://arxiv.org/pdf/2602.18090v1)

**作者:** Takahiro Sanada `[一作]` (Fukui Prefectural University), Shin-ya Katsumata `[通讯]` (Kyoto Sangyo University)

**通讯引用:** 723 | [OpenAlex ID](https://openalex.org/A5041431787)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出并实现了一种可微分箭头计算机语言（Differentiable Arrow Calculus with Operations and Reverse Handlers），为神经网络设计与实现提供了形式化语义与安全类型系统；

**💡 创新点**

创新点在于将箭头、运算符、效应处理器与逆向自动微分（Backpropagation）结合，构建基于逆向差分限制范畴（RDRC）的强势promonad模型，并通过逆向处理器实现编码器-解码器对、QAT等功能；

**🔧 技术方法**

主要技术包括强势promonad、逆向差分限制范畴、算子效应处理器、字符串图示与形式化语义；

**📊 数据集**

本文未使用具体机器学习数据集，而是通过示例网络（如MLP、残差网络、自动编码器、CNN、U‑Net）演示理论框架；

**📈 对比分析**

性能比较采用形式化证明（Soundness 与 Adequacy）与示例推导，显示语义与操作语义一致，且在手工梯度下降或STE实现中得到相同结果，未给出实验基准；

**⚠️ 局限性**

局限性包括缺乏对递归/循环网络的支持（需延迟追踪），以及对大型工业框架的完整集成与性能评估尚待进一步研究。

---

## 76. AndroWasm: an Empirical Study on Android Malware Obfuscation through WebAssembly

**arXiv ID:** 2602.18082 | [PDF](https://arxiv.org/pdf/2602.18082v1)

**作者:** Diego Soi `[一作]` (University Of Cagliari), Giorgio Giacinto `[通讯]` (Consorzio Interuniversitario Nazionale per l’Informatica)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

研究了WebAssembly在Android应用中的执行方式，并用PoC演示了恶意代码利用Wasm实现的混淆与逃逸技术。

**💡 创新点**

创新点在于首次将Wasm作为Android恶意软件的隐藏载体，系统阐述三种执行模式，并证明其能绕过VirusTotal和MobSF等主流检测。

**🔧 技术方法**

采用WebView、JavaScript引擎以及WasmEdge原生运行时，结合JNI和WASI接口进行Wasm模块加载与调用。

**📊 数据集**

使用来自公开恶意软件仓库的两款toy样本（勒索软件与间谍软件），并在实验中对其Wasm化版本与原始版本进行对比。

**📈 对比分析**

通过提交到VirusTotal比较检测率，原始样本被66个引擎标记27个，Wasm化版本仅被7个；MobSF静态分析同样未检测到Wasm中的恶意逻辑，证明Wasm能显著降低检测率。

**⚠️ 局限性**

局限性包括样本规模有限、缺乏自动化的Wasm混淆检测流水线，以及未深入研究Wasm自身的安全缺陷与利用路径。

---

## 77. Robust Temporal Guarantees in Budgeted Sequential Auctions

**arXiv ID:** 2602.17916 | [PDF](https://arxiv.org/pdf/2602.17916v1)

**作者:** Giannis Fikioris `[一作]` (Cornell University), Eva Tardos `[通讯]` (Cornell University)

**通讯引用:** 35272 | [OpenAlex ID](https://openalex.org/A5025175846)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

设计了一种极其简单的基于预算的学习算法，在每轮拍卖中根据上一轮支付的价格更新出价，目标是最大化获胜次数。该算法在对手为任意预算约束的优化者时，仍能保证每个代理至少获得与其预算比例相匹配的获胜次数（误差为 O(√T)）。在所有代理都使用该算法的自演练场景下，研究了获胜次数随时间的分布，证明在短时间后各代理的获胜次数与预算比例极其接近，且在任意长度为 O(√T) 的时间窗口内的偏差不超过 1。

**💡 创新点**

创新点在于：
1) 仅用一条简单的基元更新规则（不需要投影或随机化），就能在预算约束下提供类似“无后悔”且对对手攻击具有鲁棒性的保证；
2) 在自演练中证明了低偏差（低失配）性质，显示该算法自然实现了“几乎均匀”分配，远优于传统的无后悔或快照算法；
3) 通过将学习动态视为对凸不光滑目标函数的子梯度下降，借助二次增长和平均值收敛分析，首次给出从 O(√η) 到 O(η) 的收敛阶梯，实现了更精细的时间空间分布控制。

**🔧 技术方法**

主要技术：
- 预算约束下的单参数学习率 η，更新规则 b_{t+1}=b_t+η(ρ_i - p_i^t)。
- 将多轮拍卖的动态转化为子梯度下降问题，目标函数 f(b)=½max_i b_i^2 - ρ^T b + ½，利用其凸性、唯一最小点为 1⃗。
- 通过拉格朗日松弛、整数规划和二次增长性质分析，证明对手（优化者）在任意策略下赢得的轮次受限。
- 细化收敛分析：先证明 O(√η) 的误差收敛，再利用平均值逼近与一致性证明所有代理的出价在 O(η) 范围内收敛。
- 通过严格的区间偏差分析与轮盘规则证明在等预算情形下最终实现循环赢取。

**📊 数据集**

该工作为理论研究，没有使用实验数据集；所有结果均来自严谨的数学证明与理论分析。

**📈 对比分析**

与现有基于预算的自适应节拍（Adaptive Pacing）或无后悔/无交换后悔算法相比，本文提供了:
- 对抗任意预算约束对手的强保证（至少 ρT - 3√T 次胜利）。
- 在自演练场景下的低偏差（误差 ≤ 1）和短期收敛（O(√T)logT）。
- 与传统算法相比不需要随机化或投影操作，简洁易实现。由于缺乏实验评估，性能在实际广告平台中的表现仍待进一步验证。

**⚠️ 局限性**

限制与待改进之处：
- 仅针对目标为最大化获胜次数的单位价值假设；不考虑每轮的价值差异或效用最大化。 
- 只证明了对 first-price 和 second-price 拍卖的结果，其他拍卖形式未覆盖。 
- 结果依赖于 η 的选择（常设 η=1/√T 以获得最佳常数）；在不同预算规模或动态预算下的适应性尚未探讨。 
- 由于理论性质，缺乏对实际广告系统中噪声、非均匀价值、延迟反馈等因素的稳健性分析。 
- 对于非常不均匀的预算分配（如 1/2, 1/3, 1/6），仍无法保证严格的周期性分配，只能得到总量比例的近似。

---

## 78. Hilbert's Nullstellensatz is in the Counting Hierarchy

**arXiv ID:** 2602.17904 | [PDF](https://arxiv.org/pdf/2602.17904v1)

**作者:** Robert Andrews `[一作]` (University of Waterloo), Éric Schost `[通讯]` (University of Waterloo)

**通讯引用:** 2599 | [OpenAlex ID](https://openalex.org/A5041679295)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

论文展示了Hilbert的Nullstellensatz问题，即决定多元多项式方程组在基础域的代数闭包中是否有解，属于计数层次。

**💡 创新点**

创新点在于证明了在计数层次中可以多项式时间内计算方程组的解的数量，并构造了一个均匀的常深度算术电路族来计算多元结果式。

**🔧 技术方法**

使用了算术电路和计数层次的相关技术，特别是通过Poisson公式来计算多元结果式。

**📊 数据集**

使用了具有有理数或有限域系数的多项式，具体数据集未明确提及。

**📈 对比分析**

与之前已知的复杂性界限相比，提出的方法在多项式时间内解决了Nullstellensatz问题，性能显著提升。

**⚠️ 局限性**

限制在于对正特征域的复杂性理解较少，且在有限域上没有已知的更强下界或更好的算法。

---

## 79. Generative Model via Quantile Assignment

**arXiv ID:** 2602.18216 | [PDF](https://arxiv.org/pdf/2602.18216v1)

**作者:** Georgi Hrusanov `[一作]` (University of Lausanne), Julien S. Bodelet `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为NeuroSQL的深度生成模型，利用固定的量化格子和线性分配算法直接学习潜在变量并训练生成器，从而消除了传统生成模型所需的编码器和判别器。

**💡 创新点**

创新点在于用最佳传输理论构建多维量化点并通过匈牙利算法求解线性分配，取代了昂贵的编码器/判别器，实现了无监督、稳定、高效的潜在空间学习。

**🔧 技术方法**

采用了最优传输（Optimal Transport）构造多维分位点、线性分配（Hungarian/贪婪算法）、深度神经网络生成器以及动量更新的训练循环。

**📊 数据集**

实验使用了手写数字MNIST、人脸CelebA、动物人脸AFHQ以及脑成像OASIS四个不同域的数据集。

**📈 对比分析**

与VAE、GAN和匹配参数的扩散模型对比，NeuroSQL在FID、SSIM、LPIPS等指标上表现更好，训练速度最快、参数最少，且在低数据、低计算预算场景下性能领先。

**⚠️ 局限性**

局限性包括仅在低分辨率和有限样本的实验环境验证，尚未评估高分辨率、大规模数据以及跨模态的可扩展性，且对极端噪声或复杂结构的生成能力仍待进一步研究。

---

## 80. Evaluating Graphical Perception Capabilities of Vision Transformers

**arXiv ID:** 2602.18178 | [PDF](https://arxiv.org/pdf/2602.18178v1)

**作者:** Poonam Poonam `[一作]`, Timo Ropinski `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

介绍了 elsarticle.cls 文档类的功能与使用方法

**💡 创新点**

改进了前身 elsart.cls，兼容性更好并提供了多种排版选项

**🔧 技术方法**

使用 LaTeX 宏包 natbib、geometry、graphicx 等

**📊 数据集**

无数据集

**📈 对比分析**

无实验对比，主要说明配置与使用方法

**⚠️ 局限性**

仅适用于 Elsevier 期刊，缺乏对其他期刊的通用性

---

## 81. Asking Forever: Universal Activations Behind Turn Amplification in Conversational LLMs

**arXiv ID:** 2602.17778 | [PDF](https://arxiv.org/pdf/2602.17778v1)

**作者:** Zachary Coalson `[一作]` (Oregon State University), Sanghyun Hong `[通讯]` (Oregon State University)

**通讯引用:** 720 | [OpenAlex ID](https://openalex.org/A5102751625)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并演示了在对话式LLM中通过诱导模型持续进行澄清询问，从而导致对话回合数显著增加、算力成本膨胀的“回合放大”攻击；

**💡 创新点**

首次系统化回合放大这一新型成本放大攻击，提出交互层审计框架、激活空间中普适的澄清方向，并证明该攻击可通过激活调节、LoRA微调和位翻转实现；

**🔧 技术方法**

激活调节(activation steering)、低秩微调LoRA、梯度引导位翻转、LLM‑as‑a‑Judge评估框架、多轮对话模拟等技术；

**📊 数据集**

Alpaca（单轮QA）和GSM8K（数学题）数据集，以及基于这两者合成的10轮对话；

**📈 对比分析**

与无攻击、前缀/系统消息等基线对比，在四个LLM（Qwen2.5‑3B、Llama3‑8B、Falcon3‑10B、Mistral‑22B）上评估，激活调节导致回合数提升约9.9×、输入/输出tokens分别提升约200×/6.4×，整体成本提升约9.6×；LoRA微调与位翻转亦可显著放大；

**⚠️ 局限性**

实验仅限于开放源模型与受控环境，缺乏真实用户交互验证；防御措施有限；攻击实现需要对模型内部表示或权重的白盒访问；对模型质量影响虽有限，但仍存在一定误差。

---

## 82. WHED: A Wearable Hand Exoskeleton for Natural, High-Quality Demonstration Collection

**arXiv ID:** 2602.17908 | [PDF](https://arxiv.org/pdf/2602.17908v1)

**作者:** Mingzhang Zhu `[一作]` (University of California Los Angeles), Dennis W. Hong `[通讯]` (University of California Los Angeles)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了 WHED，可穿戴手外骨骼系统，用于在真实环境中高质量、自然地采集多指手的演示数据。

**💡 创新点**

创新点包括：①以可穿戴性为首要目标的整体架构；②自由移动拇指耦合机制，保持拇指自然工作空间并实现与机器人拇指的可学习映射；③链接驱动的指关节与被动滑动补偿的组合，兼顾精准度与跨用户适配；④完整的端到端同步数据采集与重放管线。

**🔧 技术方法**

使用了四杆链接、自由伸缩拇指关节、编码器（RDC506018A）、STM32F042 MCU、Apple ARKit（iPhone）姿态跟踪、Intel RealSense 视觉、双速缓冲、Butterworth 低通滤波、姿态变换等技术。

**📊 数据集**

数据集为作者自行采集的手部演示序列，包含对橙子、USB、干擦笔和水瓶等不同物体的抓取与搬运操作；未使用公开公共数据集。

**📈 对比分析**

通过将采集的演示与机器人重放结果进行对齐，比较轨迹相似度与抓取成功率。重放结果与原始操作高度一致，误差低且抓取稳定性与人类演示相当，表明数据质量高。

**⚠️ 局限性**

局限性：缺乏大规模跨用户实验验证、对多种复杂物体与环境的泛化评估、对长期持续使用的舒适性与耐久性进一步探测，以及更精细的运动学可执行性与稳定性分析。

---

## 83. Solving and learning advective multiscale Darcian dynamics with the Neural Basis Method

**arXiv ID:** 2602.17776 | [PDF](https://arxiv.org/pdf/2602.17776v1)

**作者:** Yuhe Wang `[一作]` (Asia Pacific Technology), Min Wang `[通讯]` (University of Houston)

**通讯引用:** 11999 | [OpenAlex ID](https://openalex.org/A5100340933)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出神经基方法（Neural Basis Method, NBM），将物理约束嵌入投影式最小化框架中，使用预定义的物理符合神经基空间求解耦合的多尺度达尔文-输运偏微分方程，并进一步构建参数化学习版本（NBM‑OL）实现快速多查询推断。

**💡 创新点**

创新点在于：
• 把传统的损失权重调参转变为算子诱导的残差度量，使残差成为可解释的误差指示器；
• 采用双层神经网络生成全局物理符合基底（通过Helmholtz分解得到散度自由与无旋向量基），在投影中保持局部质量守恒；
• 引入能量一致的加权最小二乘投影，显著提升数值稳定性；
• 在参数化学习阶段直接对残差最小化，获得基于残差的自监督目标，残差与解误差呈线性关系。

**🔧 技术方法**

技术手段包括：双层ResNet型神经基生成、Helmholtz分解构造向量基、能量一致的混合加权最小二乘投影、上风控制体卷的稳化（upwind control‑volume）用于输运；使用POD压缩生成低维特征，再用MLP预测基底系数；在训练中自监督最小化残差。

**📊 数据集**

数据集与基准：
• CO₂地下储存模型，均匀与随机多尺度渗透率场（对比系数~500）
• 通过随机块化（5×5）生成多尺度渗透率样本；
• 制造解基准（分离多尺度渗透率，提供解析解）；
• 多参数化实验：块化渗透率、顶边界通量、左侧注入浓度。

**📈 对比分析**

比较方法与性能：
• 单例求解与传统FVM、PINN对比，NBM在压力、速度、浓度的相对L2误差分别低于0.001%、0.16%/0.30%、5.17%，显著优于PINN；
• 在多尺度渗透率下，能量一致权重提升误差约7.7倍；
• 参数化学习（NBM‑OL）在多查询场景中实现10³–10⁴倍速度提升；
• 残差与误差呈线性相关，可作为训练监控指标。

**⚠️ 局限性**

局限性：
• 基底采用全局神经网络，可能在存在局部尖锐不连续或非凸几何时产生Gibbs现象；
• 需手工设计加权权重，尚未实现最优自适应策略；
• 目前实验仅在二维平面，三维扩展及大规模并行实现仍待验证；
• 对高维参数空间仍受限，无法完全规避维数灾难。

---

## 84. PsihoRo: Depression and Anxiety Romanian Text Corpus

**arXiv ID:** 2602.18324 | [PDF](https://arxiv.org/pdf/2602.18324v1)

**作者:** Alexandra Ciobotaru `[一作]` (University of Bucharest), Liviu P. Dinu `[通讯]` (University of Bucharest)

**通讯引用:** 1397 | [OpenAlex ID](https://openalex.org/A5057738797)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了第一个公开的罗马尼亚心理健康文本语料库PsihoRo，收集开放式问答文本并结合PHQ-9和GAD-7评分。

**💡 创新点**

在罗马尼亚语言上首次构建可公开获取的抑郁与焦虑文本数据集，并通过LIWC、情感检测和主题建模等多方法探索语言特征。

**🔧 技术方法**

使用罗马尼亚版LIWC进行语言特征分析，LightGBM+SHAP进行二分类评估，RoBERTa fine‑tuned on REDv2做情感多标签分类，结构化主题模型STM进行主题发现。

**📊 数据集**

PsihoRo（205条问答文本及对应PHQ-9/GAD-7评分），并利用REDv2训练情感模型。

**📈 对比分析**

采用LightGBM在LIWC特征上分别预测抑郁和焦虑，准确率分别为0.85和0.83；情感模型在REDv2测试集上的F1为66.85%；主题模型设定5主题并通过GPT-5生成可解释主题。

**⚠️ 局限性**

样本量有限，缺乏人口学信息，无法建立稳定的回归模型；语言特征在罗马尼亚语中对“I”等词不显著，导致部分传统指标失效。

---

## 85. Context-Aware Mapping of 2D Drawing Annotations to 3D CAD Features Using LLM-Assisted Reasoning for Manufacturing Automation

**arXiv ID:** 2602.18296 | [PDF](https://arxiv.org/pdf/2602.18296v1)

**作者:** Muhammad Tayyab Khana `[一作]`, Seung Ki Moon `[通讯]` (Nanyang Technological University)

**通讯引用:** 5056 | [OpenAlex ID](https://openalex.org/A5082988770)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个确定性优先、基于上下文的框架，将二维工程图注释映射到三维 CAD 特征，从而生成统一的制造规范。

**💡 创新点**

结合 VLM 语义增强、可解释的对应评分、工程启发式与受限 LLM 级联推理，形成可审计、透明的映射流程。

**🔧 技术方法**

采用视觉语言模型（GPT‑4o）进行语义丰富与多模态推理，规则驱动的对应评分、工程启发式调整，以及单步人工审阅。

**📊 数据集**

使用 20 组真实机械部件的 CAD–绘图对，包含约 101 个特征/注释，主要为尺寸注释。

**📈 对比分析**

通过与仅确定性、去掉启发式、去掉 LLM 等消融版本对比，最终平均 F1 达 86.3%，精度 83.7%，召回 90.5%，证明完整管线显著优于简化方案。

**⚠️ 局限性**

数据集规模小、GD&T 注释稀缺、对 VLM 的依赖高、对前置 AFR/提取质量敏感，且推理时延较大。

---

## 86. [Re] Benchmarking LLM Capabilities in Negotiation through Scoreable Games

**arXiv ID:** 2602.18230 | [PDF](https://arxiv.org/pdf/2602.18230v1)

**作者:** Jorge Carrasco Pollo `[一作]` (University of Amsterdam), John Hua Yao `[通讯]` (University of Amsterdam)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

复现并扩展了Scoreable Games基准，评估多种LLM在复杂多代理谈判中的表现，并在此过程中修复原始实现中的漏洞，提供更透明的评估指标。

**💡 创新点**

提出更完整的可重复实验框架、额外的社会福利评估指标、基准模型的可复制实现，并系统验证原始论文三项核心声明的可靠性。

**🔧 技术方法**

采用Python + HuggingFace Transformers、bitsandbytes量化、OpenAI GPT-4、以及自定义的泄漏检测与评估脚本，对多轮对话生成与评估进行自动化。

**📊 数据集**

使用Scoreable Games提供的五个预设游戏（Base、Base_rewritten、Game1-3）以及通过Prompt生成的额外游戏，共计超过10种设置；模型涵盖GPT‑4o mini、Qwen2.5‑72B、Mistral‑Small、Llama系列、DeepSeek等多种公开与闭源模型。

**📈 对比分析**

通过“5‑way / 6‑way”接受率、任意可达性、错误交易比例、泄漏率等指标进行模型比较；结果显示高阶模型在多数游戏中表现优异，但不同游戏对同一模型的难度差异大，模型间比较不稳定，尤其在ablation与游戏多样性实验中表现出显著波动。

**⚠️ 局限性**

局限性包括：基准游戏仍缺乏足够多样性（大多为建筑项目场景）；评估指标过度聚焦接受率，缺乏正式的纳什均衡分析；原始代码中的泄漏与阈值错误需手动修复；不同模型对设置、阈值、玩家数的敏感度高，导致跨模型比较不可靠；对弱模型的量化和泄漏检测仍存在偏差。

---

## 87. Evaluating Text-based Conversational Agents for Mental Health: A Systematic Review of Metrics, Methods and Usage Contexts

**arXiv ID:** 2602.17669 | [PDF](https://arxiv.org/pdf/2602.17669v1)

**作者:** Jiangtao Gong `[一作]` (Institute of AI Research, Tsinghua University), Yangrong Tang `[通讯]` (Institute of AI Research, Tsinghua University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过PRISMA流程对613篇文献进行系统筛选与分析，纳入132篇研究，梳理并归纳文本型心理健康对话代理的评估指标、方法与评估时空背景。

**💡 创新点**

首次构建了包含CA属性、用户体验、用户变化三大维度的评价框架，并将评估方法分为自动化指标、标准量表、定性探究三类，强调时间维度对评估效度的重要性；提出了整合评估路径和未来改进方向。

**🔧 技术方法**

采用系统综述技术：PRISMA检索、双人编码、交叉验证（κ=0.77–0.92）、频数统计与主题分析。

**📊 数据集**

数据来源为三大数据库（ACM Digital Library、Scopus、PsycINFO），检索得到613条记录，最终筛选出132篇满足条件的实证研究。

**📈 对比分析**

对比了不同评估维度与方法的使用频率与覆盖范围，但未给出具体性能指标；指出自动化指标与用户心理健康结果之间关联薄弱，需进一步验证。

**⚠️ 局限性**

局限性包括样本规模普遍偏小、干预时间短暂、主要使用西方开发的量表且缺乏跨文化适配、自动评估指标与实际用户福祉缺乏直接映射，以及专业能力评估不足。

---

## 88. MePoly: Max Entropy Polynomial Policy Optimization

**arXiv ID:** 2602.17832 | [PDF](https://arxiv.org/pdf/2602.17832v1)

**作者:** Hang Liu `[一作]` (University of Michigan), Maani Ghaffari `[通讯]` (University of Michigan)

**通讯引用:** 1180 | [OpenAlex ID](https://openalex.org/A5046777734)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出一种基于多项式能量函数的显式分布策略 MePoly，用于最大熵强化学习和模仿学习，能够在非凸、多模态的动作空间中直接优化概率密度和熵。

**💡 创新点**

创新点包括：① 将多项式指数族与最大熵原理结合，理论上实现任意分布的全局逼近；② 使用正交勒让德基底提升数值稳定性和频谱正则化；③ 通过数值积分获得可微的对数分区函数，实现单步采样与可计算的熵，避免了扩散模型的多步推理与无显式似然问题。

**🔧 技术方法**

核心技术包括：最大熵策略优化（Soft Actor-Critic / PPO）、能量基模型、勒让德多项式基、多维数值积分/网格化正则化、对数分区函数求导、低维潜在空间学习（VAE + AdaLN）用于模仿学习。

**📊 数据集**

实验数据集包括：1）Smooth World 2D 导航环境；2）Contextual Bandit 任务（lemniscate 与 two moons 目标分布）；3）ManiSkill 机器人操作基准（PushCube、PickCube、StackCube、PushT）以及其公开演示数据。

**📈 对比分析**

与传统单峰高斯、Gaussian Mixture、流匹配（FPO）等强化学习基线比较，MePoly 在多模态任务中显著减少模式崩溃并获得更高的成功率；在模仿学习上与 Diffusion Policy 在多数任务上表现相当，部分任务（PushCube、StackCube）甚至略优。总体表现优于现有多模态策略，但在部分高维任务上仍不如扩散模型。

**⚠️ 局限性**

局限性包括：① 归一化（log‑partition）采用网格积分，对高维动作空间的可扩展性受限；② 模仿学习中潜在空间的构建仍易受约束失配影响，导致性能波动；③ 需要较大计算资源（数值积分+二叉搜索）以实现实时采样。

---

## 89. Message-Oriented Middleware Systems: Technology Overview

**arXiv ID:** 2602.17774 | [PDF](https://arxiv.org/pdf/2602.17774v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 90. IRPAPERS: A Visual Document Benchmark for Scientific Retrieval and Question Answering

**arXiv ID:** 2602.17687 | [PDF](https://arxiv.org/pdf/2602.17687v1)

**作者:** Connor Shorten `[一作]` (Weaviate), Bob van Luijt `[通讯]` (Weaviate)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了IRPAPERS视觉文档检索与问答基准，并对文本与图像两种文档表示进行系统评估。

**💡 创新点**

提出了多模态混合检索方案、MUVERA编码的效率‑性能权衡、以及针对于视觉与文本互补性的失败模式分析。

**🔧 技术方法**

使用ColModernVBERT、ColPali、ColQwen2等多向量图像嵌入；Arctic 2.0与BM25混合检索；RSF、RRF融合策略；MUVERA压缩；RAG+GPT‑4.1生成模型与LLM‑as‑Judge评测。

**📊 数据集**

IRPAPERS数据集：166篇信息检索论文、3,230页图像与OCR文本、180个“针眼”式查询答案对。

**📈 对比分析**

实验表明：文本检索Recall@1 46%，图像检索 43%；多模态融合Recall@1提升至49%，Recall@5 81%，Recall@20 95%；闭源Cohere Embed v4图像嵌入Recall@1 58%，显著优于开源模型；问答层面文本RAG在k=5时对齐率0.82，图像RAG为0.71，二者均受检索深度提升。

**⚠️ 局限性**

局限性包括：图像表示缺乏精确文本匹配能力；文本转录已能捕获大部分图表信息；视觉元素需人工筛选，样本稀缺；问答评测受LLM判定噪声；仅覆盖IR领域，难以推广至更广泛学科。

---

## 91. Lend me an Ear: Speech Enhancement Using a Robotic Arm with a Microphone Array

**arXiv ID:** 2602.17818 | [PDF](https://arxiv.org/pdf/2602.17818v1)

**作者:** Zachary Turcotte `[一作]` (Université de Sherbrooke), François Grondin `[通讯]` (Université de Sherbrooke)

**通讯引用:** 4484 | [OpenAlex ID](https://openalex.org/A5045901453)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种将麦克风阵列固定在机械臂上，并通过机械臂关节动态调整阵列几何，实现工业环境下的语音增强与自动语音识别提升。

**💡 创新点**

创新点在于将可重构麦克风阵列与机械臂耦合，使阵列几何随声源位置实时变化，从而显著提升参考信号质量，并将改进的SRP‑PHAT、深度学习IRMA和MVDR beamformer集成到同一增强流水线。

**🔧 技术方法**

使用技术包括：改进的SRP‑PHAT声源定位、MediaPipe面部检测、Kinova Gen3 机械臂逆向运动学、16个全向麦克风、BLSTM网络估计理想比率掩码（IRM）以及MVDR beamformer。

**📊 数据集**

训练数据来自DNS Challenge语音数据集以及SLR28房间冲击响应；实验数据为多种噪声（真空泵、钻机、发动机等）在不同方向与SNR（-5、0、5、10 dB）下的人工混合语音。

**📈 对比分析**

与四种静态麦克风阵列和狙击麦克风进行对比，使用SI‑SDR和WER作为评估指标。实验表明，优化阵列在SI‑SDR上提升约3–5 dB，WER降至约5–7%，显著优于所有静态配置。

**⚠️ 局限性**

局限性包括：实验仅在受控环境下验证，实时实现与轻量化网络仍待完善；对面部检测与逆向运动学的依赖可能在遮挡或定位误差时影响性能；参考通道选择虽以第16声道为准，但并未实现自适应最佳通道选择。

---

## 92. A Case Study of Selected PTQ Baselines for Reasoning LLMs on Ascend NPU

**arXiv ID:** 2602.17693 | [PDF](https://arxiv.org/pdf/2602.17693v1)

**作者:** Yuchen Luo `[一作]` (Wuhan University), Wei Shao `[通讯]` (Huawei)

**通讯引用:** 7942 | [OpenAlex ID](https://openalex.org/A5077862962)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了多种后训练量化（PTQ）方法在华为 Ascend NPU 上对推理模型的性能与稳定性。

**💡 创新点**

系统地揭示了低位量化在 Ascend NPU 的平台敏感性，提出针对 NPU 的校准策略并验证了真实 INT8 推理加速瓶颈。

**🔧 技术方法**

对比 AWQ、GPTQ、SmoothQuant、FlatQuant 等四种 PTQ 算法，采用层级 MSE 校准、针对性学习率调优以及 INT8×INT8 矩阵乘法核。

**📊 数据集**

在 DeepSeek‑R1‑Distill‑Qwen 系列、QwQ‑32B 等推理模型上评测 AIME‑120、MATH‑500、GSM8K、LiveCodeBench、GPQA‑Diamond 等数学推理与编程基准；并用 Llama‑3 在 ARC‑Easy/Challenge、HellaSwag、LAMBADA、PIQA、WinoGrande 等 QA 基准。

**📈 对比分析**

通过 fake‑quantization 与真实 INT8 推理对比，使用 perplexity 与任务准确率衡量；发现 4‑bit 权重仅量化可行，4‑bit 权重+激活在长上下文推理上易失真，8‑bit 量化保持稳定，真实 INT8 加速受动态量化开销限制。

**⚠️ 局限性**

受限于 Ascend NPU 对低位量化的数值实现与算子缺失，导致 4‑bit W‑A‑KV 方案不稳；动态量化开销高、缺乏完整算子支持，需进一步低层优化。

---

## 93. Unifying approach to uniform expressivity of graph neural networks

**arXiv ID:** 2602.18409 | [PDF](https://arxiv.org/pdf/2602.18409v1)

**作者:** Huan Luo `[一作]` (University of Glasgow), Jonni Virtema `[通讯]` (University of Glasgow)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

引入模板图神经网络（Template GNN）框架，定义模板 Weisfeiler–Leman 算法、模板 bisimulation 与相应的格子模态逻辑，并证明它们与模板 GNN 的表达能力等价。

**💡 创新点**

提出统一的模板 GNN 元理论，可将多种扩展 GNN（如 k‑hop 子图、子图计数等）归入同一框架，并给出对应的逻辑表达式，提供了对 GNN 表达力的系统化描述。

**🔧 技术方法**

通过模板嵌入、颜色细化、格子模态逻辑与有限计数的组合，构建了模板 GNN 与逻辑之间的双向转换，实现了形式化证明。

**📊 数据集**

本研究为理论分析，未使用任何具体数据集，而是在形式化证明中考虑任意有限图。

**📈 对比分析**

通过证明模板 GNN 的分类器与对应的逻辑公式等价，展示了该框架能够捕捉所有受限计数的 GNN 表达能力，提供了统一的可解释性与上限分析。

**⚠️ 局限性**

仅适用于计数上有上界的模板 GNN；对无上界的非限制计数 GNN 的表达能力尚未完整刻画，递归结构的扩展仍需进一步研究。

---

## 94. Learning Compact Video Representations for Efficient Long-form Video Understanding in Large Multimodal Models

**arXiv ID:** 2602.17869 | [PDF](https://arxiv.org/pdf/2602.17869v1)

**作者:** Yuxiao Chen `[一作]` (Amazon AGI), Davide Modolo `[通讯]` (Amazon AGI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种端到端的长视频理解框架，包含信息密度自适应采样器（AVS）和基于自编码器的时空视频压缩器（SVC），并将其与多模态大语言模型（MLLM）集成。

**💡 创新点**

创新点在于：① 通过信息密度和镜头边界检测实现自适应帧采样，显著减少冗余；② 采用残差约束的自编码器压缩时空特征，实现64×压缩率同时保持判别信息；③ 只用视频自监督预训练，无需视频-文本对；④ 结合MLLM实现全长视频的单一推理。

**🔧 技术方法**

主要技术包括：ViT视频编码器、3D卷积自编码器、镜头边界检测与非极大值抑制、残差潜在空间约束、投影器映射至LLM、Qwen2大语言模型、四阶段训练策略。

**📊 数据集**

使用的数据集：训练阶段使用ShutterStock（≈3M）和多种视频数据（Ego4D、Breakfast、AVA、Vatex、Something‑somethingV2、Kinetics）；微调阶段加入NextQA、CLEVRER、PerceptionTest、Egoschema；评测集包括PerceptionTest、ActivityNet‑QA、MVBench、NextQA、EgoSchema、MLVU。

**📈 对比分析**

与现有长视频方法（VideoAgent、LLoVi、VideoINSTA、LLaMA‑VID、Movie‑Chat）和SOTA MLLM（LLava‑OV）对比，平均提升 2.6%–3.3%（EgoSchema/PerceptionTest），在ActivityNet‑QA提高 4.8%，并将视觉令牌量降低 80%（约 1,440 令牌对比 6,000）。

**⚠️ 局限性**

局限性：对镜头切换检测的依赖导致在无明显镜头变化的视频上提升有限；压缩比与性能权衡仍需进一步优化；当前仅在ViT/三维卷积框架下验证，缺乏对实时或更大模型的探索。

---

## 95. Learning Smooth Time-Varying Linear Policies with an Action Jacobian Penalty

**arXiv ID:** 2602.18312 | [PDF](https://arxiv.org/pdf/2602.18312v1)

**作者:** Zhaoming Xie `[一作]` (Robotics AI Institute), Jessica Hodgins `[通讯]` (Robotics AI Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出利用动作雅可比矩阵惩罚与线性策略网络（LPN）联合训练，实现平滑时间变线性控制策略，用于模拟人物和真实四足机器人动作模仿；

**💡 创新点**

创新点在于：①以动作雅可比矩阵惩罚取代传统任务依赖的高频抑制奖励，消除高频控制信号；②设计LPN只输出反馈矩阵与前馈动作，显著降低雅可比矩阵计算开销，并保持学习速度；③在多种动态与环境交互任务中验证其普适性；

**🔧 技术方法**

使用技术包括：深度强化学习（PPO）训练，动作雅可比矩阵惩罚通过自动微分实现，LPN结构（两层MLP生成反馈矩阵和前馈动作），低秩线性反馈矩阵近似，模拟与真实Spot四足机器人的仿真-实机迁移，以及策略蒸馏与技能组合；

**📊 数据集**

数据集主要为DeepMimic风格的参考动作，包括步行、跑步、后空翻、侧翻、手倒立、台球步伐、壁面攀爬、双孔跳以及足球运球等；此外使用Spot机器人手臂与腿部的手工周期运动数据；

**📈 对比分析**

与四种基线方法（FF+Jac惩罚、无正则化、动作变化惩罚、多种权重、Lipschitz约束）进行比较。评估指标为动作平滑度、高频比率与运动颤动；实验显示LPN+Jac惩罚在收敛速度最快、平滑度最优或相当，FF+Jac惩罚收敛慢，动作变化惩罚需手工调参，Lipschitz约束无法产生平滑策略；

**⚠️ 局限性**

局限性包括：①对极端动态动作（如后空翻）雅可比矩阵惩罚效果有限；②线性策略在需要更高阶控制的复杂动作中表现受限；③仅针对DeepMimic式模仿任务，难以直接迁移至无运动捕捉或对抗式任务；④技能间的无缝切换尚未完全实现；⑤主要验证短周期动作，长序列的可扩展性尚待探索。

---

## 96. TempoNet: Slack-Quantized Transformer-Guided Reinforcement Scheduler for Adaptive Deadline-Centric Real-Time Dispatchs

**arXiv ID:** 2602.18109 | [PDF](https://arxiv.org/pdf/2602.18109v1)

**作者:** Rong Fu `[一作]` (University of Macau), Simon James Fong `[通讯]` (University of Macau)

**通讯引用:** 11928 | [OpenAlex ID](https://openalex.org/A5086422507)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 TempoNet，一种基于强化学习的实时调度框架，利用紧迫度分词器与轻量级 Transformer Q‑网络实现低延迟、多核任务分配。

**💡 创新点**

将连续滑动余量量化为可学习的紧迫度 token，结合稀疏块 Top‑k 与局部分块注意力实现全局推理，同时保持子毫秒推理时间；并设计可插拔的多核映射层。

**🔧 技术方法**

深度 Q‑学习、可学习的滑动余量分词器、稀疏 Transformer、块级 Top‑k 与局部分块注意力、迭代掩码贪婪或可微匹配的多核映射。

**📊 数据集**

工业级混合关键性工作负载、100–600 任务规模的多核仿真集以及标准周期任务配置的基准集。

**📈 对比分析**

与经典 RM/EDF、FF‑DQN、DIOS 等方法对比，在单核、工业多核和大规模多核情景下均实现最高的准时率（≈90%）和约 25–35% 的平均响应时间提升，且推理时间维持在子毫秒级。

**⚠️ 局限性**

仅在同构多核环境验证，缺乏异构硬件和多目标（能耗、能效）评估；对极端高负载或任务极端不确定性时的鲁棒性仍待进一步研究。

---

## 97. It's Not Just Timestamps: A Study on Docker Reproducibility

**arXiv ID:** 2602.17678 | [PDF](https://arxiv.org/pdf/2602.17678v1)

**作者:** Oreofe Solarin `[一作]` (Case Western Reserve University), Oreofe Solarin `[通讯]` (Case Western Reserve University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5118596158)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究构建了 Docker 测量管道，对 2000 个 GitHub 仓库的 Dockerfile 进行可重建性评估，揭示原始可重现率仅 2.7%，硬化基础设施后提高到 18.6%，并通过差分分析定位开发者行为导致的非重现根因，提出可行的 Dockerfile 指南。

**💡 创新点**

创新点在于系统量化 Dockerfile 可重现性、揭示基础设施与开发者选择对非重现性的相对贡献，并基于差分报告提出针对性的重现性改进建议。

**🔧 技术方法**

使用 Docker BuildKit 进行清洁构建、SHA‑256 哈希对比、diffoscope 逐文件差异分析，并通过脚本自动化抓取、构建与分析流程。

**📊 数据集**

使用了 2000 个基于 GitHub Archive 的流行开源项目（至少 5 星），并在 2024 年度活跃仓库中按星数分层抽样。

**📈 对比分析**

方法为两次清洁构建对比、硬化后构建对比以及 diffoscope 逐文件对比，结果显示可重现率从 2.7% 提升到 18.6%，但 78.7% 的构建仍不可重现。

**⚠️ 局限性**

局限在于仅检测单一 Dockerfile、单平台构建、未考虑多架构与不同镜像源的差异，且根因标注依赖于 diffoscope 的近似标签。

---

## 98. Distribution-Free Sequential Prediction with Abstentions

**arXiv ID:** 2602.17918 | [PDF](https://arxiv.org/pdf/2602.17918v1)

**作者:** Jialin Yu `[一作]` (Georgia Institute of Technology), Moïse Blanchard `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

研究了在半对抗环境下的在线预测问题，提出了一种无需先验分布知识的分布无关的拒绝预测算法。

**💡 创新点**

创新点在于将弱学习者的投票与 Boosting 结合，并通过估计打散概率实现误分类与拒绝误差之间的多项式权衡，从而在一般 VC 类下实现子线性误差。

**🔧 技术方法**

采用 U 统计量估计打散概率、极值分布中值估计、弱学习者组合、删除策略的 Boosting、以及基于减少维数（reduction dimension）的理论分析。

**📊 数据集**

论文未使用实证数据集，全部为理论分析与合成实验。

**📈 对比分析**

与已知分布前提下的算法相比，该方法在无先验分布的情况下仍保持子线性误分类和拒绝误差，且对适应性攻击者在有限减少维数的 VC 类（如线性分类器）也能取得类似性能。

**⚠️ 局限性**

局限性：对适应性攻击者仅适用于有限减少维数的 VC 类；在一般 VC 类下上界与下界仍存在多项式差距；弱学习者需要少量干净样本，实际实现难度和常数较大；缺乏实验验证。

---

## 99. Homotopic information gain for sparse active target tracking

**arXiv ID:** 2602.17926 | [PDF](https://arxiv.org/pdf/2602.17926v1)

**作者:** Jennifer Wakulicz `[一作]` (University of Sydney), Robert Fitch `[通讯]` (University of Technology Sydney)

**通讯引用:** 6593 | [OpenAlex ID](https://openalex.org/A5053830227)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于同伦信息增益的主动目标跟踪方法，在同伦信念空间内规划感知轨迹，显著降低搜索复杂度。

**💡 创新点**

创新点在于定义同伦信息增益并证明其为度量信息增益下界，利用离散稀疏的同伦信念空间进行高效规划，并采用热图+MCTS解决带时间窗口的最优行程问题。

**🔧 技术方法**

核心技术包括同伦签名（h‑signature）、变量阶Markov过程(VOMP)估计同伦信念、同伦高斯混合模型、KL散度计算、信息增益热图、时间窗口最优行程（OPTW）和蒙特卡洛树搜索（MCTS）。

**📊 数据集**

实验使用三组数据集：模拟行人数据（3~7个障碍物）、ATC商场真实行人数据（5个障碍物）以及THÖR真实+增广行人数据（3个障碍物）。

**📈 对比分析**

与传统低层度量信息增益（最大化后验GMM熵）方法对比，使用均方位移误差（ADE）、KL散度、互信息和运行时等指标评估。结果显示，同伦方法在维持或略优的ADE的同时，测量次数更少、计算时间更短，成功率仅在障碍物数量大幅增加时略低。

**⚠️ 局限性**

局限性包括：对不穿过障碍物射线的路径（空同伦签名）信息增益为零；对目标到达时间不确定性处理不足；同伦签名仅捕获射线穿越信息，低层轨迹复杂度高时效果下降；需要更通用的同伦不变量或更鲁棒的时间窗口处理。

---

## 100. PHAST: Port-Hamiltonian Architecture for Structured Temporal Dynamics Forecasting

**arXiv ID:** 2602.17998 | [PDF](https://arxiv.org/pdf/2602.17998v1)

**作者:** Shubham Bhardwaj `[一作]` (University of Texas at Austin), Chandrajit Bajaj `[通讯]` (University of Texas at Austin)

**通讯引用:** 9788 | [OpenAlex ID](https://openalex.org/A5030896164)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种统一的端到端端口-哈密顿网络（PHAST），可在仅观测位置的情况下学习并预测耗散动力学，兼顾长期稳定性和物理可解释的参数恢复。

**💡 创新点**

创新点在于将端口-哈密顿结构与低秩正定参数化、阻尼强度上限、Strang 拆分积分器相结合，形成可插拔的已知/部分/未知三种知识模式，并引入双轴评估（预测与可识别性）来揭示两者的权衡。

**🔧 技术方法**

使用端口-哈密顿网络框架、低秩 PSD/SPD 参数化、Strang 划分数值积分、TCN 速度观测器、以及多项式损失函数（数据拟合、能量守恒、滚动误差）等技术。

**📊 数据集**

在十三个仅观测位置的基准任务上评估，包括机械（摆杆、双摆、摆杆+电路）、非机械（RLC 电路、Lennard‑Jones 粒子、热交换、引力三体、洛特卡‑沃尔泰拉）等多领域系统。

**📈 对比分析**

与 GRU、S5、LinOSS、D‑LinOSS、Transformer 和 VPT 等基线比较，PHAST 在所有任务中实现了最优的长时 horizon 预测误差，并在已知/部分模式下实现了高达 0.99+ 的阻尼回归 R²，显著优于传统结构保留网络和序列模型。

**⚠️ 局限性**

主要局限包括：仅适用于 q‑only 设定，需在短上下文窗口内估计动量，噪声和高自由度系统下表现不稳定；在部分/未知模式下若无物理先验或阻尼上限，参数可识别性仍有限；离散时间稳定性依赖步长与积分子选择。

---

## 101. JAX-Privacy: A library for differentially private machine learning

**arXiv ID:** 2602.17861 | [PDF](https://arxiv.org/pdf/2602.17861v1)

**作者:** Ryan McKenna `[一作]`, Mikhail Pravilov `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

构建了一个基于 JAX 的差分隐私机器学习库（JAX-Privacy），集成了批次采样、梯度裁剪、噪声注入、隐私核算与审计，并提供 Keras 级别的易用 API。

**💡 创新点**

将核心 DP 组件抽象为可插拔、可验证的函数，支持多种噪声策略（如高斯、矩阵分解）、微批裁剪和分布式噪声生成；利用 JAX 纯函数式编程实现高效梯度裁剪与可扩展的隐私核算，兼容多机训练；提供统一的审计框架验证实现正确性。

**🔧 技术方法**

采用 JAX、jax.vmap/jax.grad/jax.lax.scan 微批技术、NumPy、矩阵分解噪声、分布式训练（pmap、jax.random）、TensorFlow Privacy、Opacus 等库做基准比较，结合 PyTorch/AdamW。

**📊 数据集**

论文主要在单机 NVIDIA RTX 3060 上使用空白/虚拟数据进行吞吐率基准；在内部项目中对 Gemma、VaultGemma 等大模型进行微调，但未公开具体训练数据集。

**📈 对比分析**

通过在 CNN、State Space、Transformer 三类模型（1M、10M、100M 参数）上测量吞吐率（样本/秒），与 PyTorch 普通训练和 Opacus 对比；结果显示 DP 训练在某些模型（如中大型 State Space）可达 1.04× 的非 DP 速度，CNN 开销约 2×，整体比 TensorFlow Privacy 的 10×+ 开销更低。

**⚠️ 局限性**

局限性包括：仅在单机基准上评估，缺乏公开数据集实验；对大型模型的性能与可扩展性仅在内部案例中验证；批次裁剪和噪声参数调优仍需经验；审计框架虽提供，但实际攻击成本与安全性仍需进一步研究。

---

## 102. Learning Without Training

**arXiv ID:** 2602.17985 | [PDF](https://arxiv.org/pdf/2602.17985v1)

**作者:** Ryan O'Dowd `[一作]` (Claremont Graduate University), Ryan O'Dowd `[通讯]` (Claremont Graduate University)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5009984435)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9`

**🎯 论文内容**

研究了三类机器学习问题：监督学习中的函数逼近与流形学习、跨域迁移学习中的函数提升、以及基于信号分离思想的主动学习分类。

**💡 创新点**

提出了新的基于流形的逼近方法、将逆问题与迁移学习联系、并将信号分离技术用于分类的统一框架。

**🔧 技术方法**

采用了流形学习、逆问题理论、正交多项式、Clenshaw 算法及深度网络实现等数学工具。

**📊 数据集**

文中未给出具体公开数据集，实验多采用合成或模拟数据验证。

**📈 对比分析**

与现有监督学习、迁移学习与主动学习算法进行对比，取得相当甚至更优的准确率，并显著加快计算速度。

**⚠️ 局限性**

方法依赖于流形、光滑性与噪声假设，实际大规模数据和非光滑情况下效果可能受限。

---

## 103. Snapping Actuators with Asymmetric and Sequenced Motion

**arXiv ID:** 2602.18421 | [PDF](https://arxiv.org/pdf/2602.18421v1)

**作者:** Xin Li `[一作]` (University of Freiburg), Edoardo Milana `[通讯]` (University of Freiburg)

**通讯引用:** 610 | [OpenAlex ID](https://openalex.org/A5013363595)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文设计并实现了一种偏心圆顶型快变形驱动器，并将其集成到四足软机器人中，以单一气压实现异步行走和跳跃运动。

**💡 创新点**

创新点在于通过在圆顶结构中引入几何偏心和厚度调节，形成可控的非对称快变形路径，实现无需电子控制的物理时序；同时利用窄通道产生气压延迟，实现多驱动器的时序协同。

**🔧 技术方法**

采用有限元数值模拟（Abaqus/CAE）进行压力-体积及位移分析；实验方面使用真空泵、压力传感器、高速相机记录运动轨迹；气动网络设计基于RC电路类比。

**📊 数据集**

未使用公开数据集，所有实验数据均为本文自行采集的压力曲线、位移轨迹和机器人运动速度。

**📈 对比分析**

与传统单向快变形或需要多阀控制的软机器人相比，本文实现了仅靠单一气压即可完成四足协调运动，在7.5 Hz频率下达到最大72.78 mm s⁻¹的速度，速度比1 Hz下提升约14倍，显示出高效能与简化控制的优势。

**⚠️ 局限性**

局限性包括：偏心设计对制造误差敏感，导致快变形阈值偏差；气压延迟控制仅适用于低频操作，频率升高时前腿动作被抑制；缺乏内置压气源，仍需外部泵源；未来需进一步优化几何参数并实现全无绳化系统。

---

## 104. Robo-Saber: Generating and Simulating Virtual Reality Players

**arXiv ID:** 2602.18319 | [PDF](https://arxiv.org/pdf/2602.18319v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 105. MIDAS: Mosaic Input-Specific Differentiable Architecture Search

**arXiv ID:** 2602.17700 | [PDF](https://arxiv.org/pdf/2602.17700v1)

**作者:** Konstanty Subbotko `[一作]` `[通讯]` (Independent Researcher), Konstanty Subbotko (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于自注意力的输入特定可微架构搜索方法MIDAS，动态生成每个输入的架构参数。

**💡 创新点**

创新点在于将静态架构参数替换为基于局部补丁的自注意力计算得到的输入特定参数，并通过无参数拓扑搜索实现节点连接选择。

**🔧 技术方法**

使用轻量级点积注意力、Patch-wise自注意力、参数无关拓扑搜索空间和DARTS等基础NAS框架。

**📊 数据集**

在NAS-Bench-201、DARTS空间、RDARTS四个搜索空间以及CIFAR-10、CIFAR-100和ImageNet-16-120数据集上进行实验。

**📈 对比分析**

与DARTS、AGNAS、GDAS等方法比较，MIDAS在CIFAR-10上取得97.42% top-1，CIFAR-100 83.38%，在NAS-Bench-201中多次发现全局最优架构，并在ImageNet上获得约75.4% top-1，整体性能优于或持平于现有最佳方法。

**⚠️ 局限性**

局限性包括解码过程需对输入特定参数进行平均，可能因多峰分布导致解码不确定；需要手动设置补丁大小；计算开销和显存略高；未在更大规模任务或多任务场景中验证。

---

## 106. SOMtime the World Ain$'$t Fair: Violating Fairness Using Self-Organizing Maps

**arXiv ID:** 2602.18201 | [PDF](https://arxiv.org/pdf/2602.18201v1)

**作者:** Joseph Bingham `[一作]` (Technion-Israel Institute of Technology), Dvir Aran `[通讯]` (Technion-Israel Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过实验证明，即使在无监督表示学习中，若将敏感属性（如年龄、收入）从输入中排除，模型仍能在嵌入空间中重建这些属性，且会形成全局的单调排序；并基于此提出一种无监督公平性审计工具SOMtime；

**💡 创新点**

创新点在于揭示无监督表示可能隐含全局敏感属性结构，并通过高容量自组织映射(SOM)捕捉并可视化这一结构；提供一种可直接检视嵌入层面公平风险的审计方法；

**🔧 技术方法**

技术核心为高容量Self‑Organizing Map（SOM），将样本映射到二维网格并通过激活能量扩展为三维嵌入；随后通过Spearman相关系数评估敏感属性与嵌入轴的对齐；并与PCA、UMAP、t‑SNE、自动编码器等基线方法对比；

**📊 数据集**

使用的真实数据集为：World Values Survey（WVS）五国子集（加拿大、罗马尼亚、德国、中国、美国），以及KDD Census‑Income（约27k有效记录）; 两者均为表格型数据；

**📈 对比分析**

对比方法包括PCA、UMAP、t‑SNE、AE等；SOMtime在年龄属性上Spearman最高可达0.85、收入0.83，而其他基线最高仅约0.31；在WVS与Census上SOMtime均显著优于基线，显示其在泄露敏感信息方面更为敏感；

**⚠️ 局限性**

局限性：仅针对顺序型敏感属性（年龄、收入）；未涵盖分类属性（性别、种族）且缺少相应评估指标；仅在表格数据上验证，未扩展至文本、图像、图结构；未给出公平性缓解方案，需进一步研究。

---

## 107. SimVLA: A Simple VLA Baseline for Robotic Manipulation

**arXiv ID:** 2602.18224 | [PDF](https://arxiv.org/pdf/2602.18224v1)

**作者:** Yuankai Luo `[一作]`, Zhenguo Li `[通讯]` (Frontier Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 SimVLA——一种简单的 Vision‑Language‑Action 基线模型，核心思想是将感知与控制完全解耦，配合严格统一的训练方案。

**💡 创新点**

创新点在于：1）采用最小化的模块化架构（VLM 编码器 + 轻量级动作头）实现高性能；2）系统阐释并标准化了“无声”实现细节（数据打乱、动作归一化、学习率调度等），展示这些细节对性能的决定性影响；3）证明 0.5B 参数模型即可在多大模型上取得 SOTA 结果。

**🔧 技术方法**

主要技术包括：预训练的 Vision‑Language Backbone（如 SmolVLM‑0.5B 或 Florence‑2）、Transformer 动作头、条件流匹配（Flow Matching）进行连续动作生成、动作块（action chunk）机制、以及严格的优化调度（学习率、多步 warm‑up、学习率衰减）。

**📊 数据集**

使用的数据集涵盖：LIBERO（Spatial、Object、Goal、Long）、LIBERO‑PRO（对抗性扰动）、SimplerEnv（SimFractal、SimBridge）、Galaxea Open‑World 数据集以及在 Galaxea R1 Lite、WidowX、Google Robot 上的真实机器人任务。

**📈 对比分析**

与 OpenVLA、MemoryVLA、π_0.5、X‑VLA 等众多基线在相同输入/训练条件下对比，SimVLA 在 LIBERO 的平均成功率达到 98.6%，在 LIBERO‑PRO 语义鲁棒性几乎为 100%，在 WidelyX 与 Google Robot 的任务上也取得 95.8% 和 76.1% 的平均成功率，整体性能与大模型持平或超越，且显著降低了参数量和显存占用。

**⚠️ 局限性**

局限性包括：1）对空间（位置）扰动仍易失效；2）依赖精细调参（数据打乱、学习率、归一化）才能实现最佳表现；3）缺乏显式记忆机制，可能在极长序列或多目标任务中表现不如专门设计的长记忆模型；4）对高度多模态动作分布的建模能力有限。

---

## 108. VIRAASAT: Traversing Novel Paths for Indian Cultural Reasoning

**arXiv ID:** 2602.18429 | [PDF](https://arxiv.org/pdf/2602.18429v1)

**作者:** Harshul Raj Surana `[一作]` (AI Institute, University of South Carolina), Amit Sheth `[通讯]` (AI Institute, University of South Carolina)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向印度文化的多跳问答基准数据集VIRAASAT，并提出基于符号化图操作的SCoM推理框架；

**💡 创新点**

创新点在于：①使用专家构建的700+节点知识图谱自动生成多跳问题；②引入符号验证器和原子操作的SCoM训练，显著提升多跳推理准确率；

**🔧 技术方法**

技术主要包括：知识图谱构建、模板化问题生成、链式思考（CoT）与符号链式操作（SCoM）以及参数高效微调（PEFT）；

**📊 数据集**

数据集为VIRAASAT，包含约3.2k个2跳问题，覆盖印度28邦及8联邦领地，基于13个文化属性；

**📈 对比分析**

与零射击、CoT微调等基准相比，SCoM微调可在全匹配率上提升约20%（如Qwen2.5-7B从36%提升至57%），性能显著优于传统CoT；

**⚠️ 局限性**

局限性包括：仅支持2跳模板；知识图谱稀疏且仅覆盖选定属性；仅英文版本；未覆盖更长跳或多桥结构。

---

## 109. On the scaling relationship between cloze probabilities and language model next-token prediction

**arXiv ID:** 2602.17848 | [PDF](https://arxiv.org/pdf/2602.17848v1)

**作者:** Cassandra L. Jacobs `[一作]` (University at Buffalo), Morgan Grobol `[通讯]` (Université Paris Nanterre)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在填空任务（cloze）中的预测性能，比较不同规模模型与人类填空概率的匹配度，并探讨模型容量、训练预算、去重等因素的影响。

**💡 创新点**

发现更大模型在语义适配和概率估计上更符合人类填空行为，并揭示去重能显著提升模型与人类答案的一致性，提出对模型记忆与语义泛化关系的新洞察。

**🔧 技术方法**

使用Pythia系列大规模自回归语言模型，结合logit变换、Luce选择法、n-gram统计、Spearman秩相关、代表性相似度分析（RSA）和邻域重叠度量等技术进行评估。

**📊 数据集**

使用Peelle的3085句子填空规范数据集作为人类答案来源，模型训练采用Pile（以及去重版本）以及维基百科用于n-gram基线。

**📈 对比分析**

通过Spearman秩相关、log-odds回归、对比n-gram统计以及语义空间相似度（RSA）等多重指标评估；结果显示最大Pythia-2.8B模型与人类答案的秩相关最高（≈0.48），但仍显著低估中间概率，并且去重训练可提升约20%。

**⚠️ 局限性**

实验仅涵盖Pythia系列模型，未覆盖其它架构或RLHF训练，且仅在英语单词级别评估，难以推广到多形态或低资源语言；子词预测可能导致语义邻域不一致。

---

## 110. Probabilistic NDVI Forecasting from Sparse Satellite Time Series and Weather Covariates

**arXiv ID:** 2602.17683 | [PDF](https://arxiv.org/pdf/2602.17683v1)

**作者:** Irene Iele `[一作]` (Università Campus Bio-Medico di Roma), Matteo Tortora `[通讯]` (University of Genoa)

**通讯引用:** 143 | [OpenAlex ID](https://openalex.org/A5076081587)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种针对稀疏、间歇性云遮蔽下清晰天空 Sentinel‑2 NDVI 的概率预测框架，利用 Transformer 对历史植被序列和未来气象协变量进行分支建模，生成多步量化预测。

**💡 创新点**

创新点包括：①分离历史与未来信息的双分支 Transformer 架构；②针对间歇采样引入时间距离加权的分位数损失；③通过累积与极端天气特征工程捕捉延迟气象影响；④利用稀疏时间选择保持与真实卫星观测时序一致。

**🔧 技术方法**

使用 Transformer 编码器、时间距离加权的分位数损失、多分位数预测头、周期性编码、二值掩码处理缺失值、气象协变量的累积/极端特征工程以及未来协变量的扰动模拟。

**📊 数据集**

GreenEarthNet 数据集：涵盖 2017‑2022 年欧盟 24,861 份数据立方体，每个立方体包含 30 张 5 天间隔的云遮蔽 Sentinel‑2 NDVI 与 150 天气变量（风速、湿度、短波辐射、降水、气压、温度）。

**📈 对比分析**

与 AutoARIMA、LSTMPlus、RNNPlus、DeepAR、InceptionTimePlus、PatchTST、TimeLLM 等统计、循环、卷积、Transformer 和大语言模型基线对比；在 RMSE、MAE、WMAPE、MASE、CRPS、Pinball 等指标上均取得显著最优（如 RMSE 0.0821，MAE 0.0509）。

**⚠️ 局限性**

局限性：仅在欧洲农田数据上验证，缺乏对其他地区、作物类型和气候区的泛化评估；未来需加入气候区和作物类型条件以提升跨区域鲁棒性。

---

## 111. Going Down Memory Lane: Scaling Tokens for Video Stream Understanding with Dynamic KV-Cache Memory

**arXiv ID:** 2602.18434 | [PDF](https://arxiv.org/pdf/2602.18434v1)

**作者:** Vatsal Agarwal `[一作]` (Work completed during internship at TikTok), Abhinav Shrivastava `[通讯]` (University of Maryland)

**通讯引用:** 7594 | [OpenAlex ID](https://openalex.org/A5101614443)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 MemStream 框架，利用稀疏滑动窗口注意力和训练无关的混合专家检索，以在视频流理解中更高效地存储和检索关键帧信息。

**💡 创新点**

创新点包括（1）自适应关键选择（AKS）在编码阶段消除时空冗余；（2）引入训练无关的外部模型混合专家检索（RRF 融合）提升检索稳定性。

**🔧 技术方法**

使用了 KV 缓存、稀疏滑动窗口注意力、Patch‑wise/Frame‑wise 动态压缩、Reciprocal Rank Fusion、动态视频编码与多模态大语言模型 Qwen2.5‑VL。

**📊 数据集**

在 CG‑Bench、LVBench、VideoMME Long 以及在线 RVS‑Ego、RVS‑Movie 等长视频问答基准上进行评估。

**📈 对比分析**

与 ReKV 等前沿方法对比，MemStream 在 CG‑Bench 提升 +8.0%、LVBench +8.5%、VideoMME Long +2.4%；在线 VQA 中相较 ReKV 实现 +3.6% 的精度提升，且延迟保持相同或更低。

**⚠️ 局限性**

在极长视频或高帧率场景下，过度压缩可能导致关键帧信息缺失；外部检索对全局语义的覆盖仍有限。

---

## 112. Interacting safely with cyclists using Hamilton-Jacobi reachability and reinforcement learning

**arXiv ID:** 2602.18097 | [PDF](https://arxiv.org/pdf/2602.18097v1)

**作者:** Aarati Andrea Noronha `[一作]` (Carnegie Mellon University), Jean Oh `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3002 | [OpenAlex ID](https://openalex.org/A5019807694)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一个将 Hamilton–Jacobi 可达性分析与深度 Q 学习相结合的框架，用以实现自动驾驶车辆与自行车手交互时的安全与最优平衡，计算安全度量并将其融入奖励信号，同时通过自编码器建模自行车手的舒适度。

**💡 创新点**

创新点包括：1）将单智能体的安全 Bellman 与 Q 学习扩展到双智能体差分游戏；2）将自行车手的舒适度作为扰动输入，利用自编码器进行稀有事件分类；3）将可达性分析得到的安全度量直接嵌入强化学习奖励。

**🔧 技术方法**

采用 Hamilton–Jacobi 可达性分析、Hamilton–Jacobi–Bellman PDE 求解、双代理零和差分游戏建模、稀有事件自编码器、深度 Q 网络（DQN）与经验回放等技术。

**📊 数据集**

使用安全驾驶马里兰数据库（Safety Pilot Michigan Database）中的自行车事件数据。

**📈 对比分析**

通过与人类驾驶者和 Fisac 2019 单智能体方法的对比，利用 Unsafe states%、Safety factor、Time、Collisions、Goal Reached 等指标评估。结果显示，本框架在安全性上优于两者，时间略高于 Fisac 2019 但快于人类。

**⚠️ 局限性**

局限性在于可达性分析的计算复杂度高、扰动模型依赖自编码器且稀有事件样本有限，且实验仅覆盖双智能体场景，尚未验证多智能体扩展。

---

## 113. CityGuard: Graph-Aware Private Descriptors for Bias-Resilient Identity Search Across Urban Cameras

**arXiv ID:** 2602.18047 | [PDF](https://arxiv.org/pdf/2602.18047v1)

**作者:** Rong Fu `[一作]` (University of Macau), Simon Fong `[通讯]` (University of Macau)

**通讯引用:** 11928 | [OpenAlex ID](https://openalex.org/A5086422507)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出CityGuard框架，结合拓扑感知注意力、分散适应的边缘学习和差分隐私嵌入，实现城市规模分布式摄像头下的隐私保护人身份检索。

**💡 创新点**

①将粗略几何先验映射为摄像头拓扑图，注入自注意力以提升跨视角一致性；②采用实例级自适应边缘（ACT）损失根据身份特征分散动态调节阈值；③使用可校准的高斯机制和敏感度裁剪实现可量化的差分隐私索引。

**🔧 技术方法**

拓扑感知图注意力、时间图网络(TGN)、分散适应的ACT损失、熵正则化最优传输、差分隐私嵌入、近似最近邻索引(HNSW/PQ)。

**📊 数据集**

Market‑1501、MARS、MSMT17、RegDB、SYSU‑MM01、Occluded‑REID、Partial‑REID等公共人再识别基准。

**📈 对比分析**

在所有基准上与多种现有方法对比，CityGuard在Rank‑1、mAP、mINP等指标上均超越对手；在对抗攻击、遮挡、域迁移等场景下保持高鲁棒性；并在大规模检索时实现低延迟与高吞吐。

**⚠️ 局限性**

依赖粗略几何先验时仍受定位误差影响；差分隐私噪声对极低预算下会导致性能下降；对极端遮挡或相近服装仍易误检；当前模型训练复杂度高，需多阶段调参。

---

## 114. DeepSVU: Towards In-depth Security-oriented Video Understanding via Unified Physical-world Regularized MoE

**arXiv ID:** 2602.18019 | [PDF](https://arxiv.org/pdf/2602.18019v1)

**作者:** Yujie Jin `[一作]` (Soochow University), Guodong Zhou `[通讯]` (Soochow University)

**通讯引用:** 10100 | [OpenAlex ID](https://openalex.org/A5012794465)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种针对深度安全视频理解（DeepSVU）任务的 Unified Physical-world Regularized MoE（UPRM）模型，能够精准识别、定位并归因视频中的威胁。

**💡 创新点**

创新点包括：①将粗细层次的物理世界信息融合到MoE框架中，形成三种细粒度专家（姿态、物体关系、背景）与一类粗粒度专家；②设计了物理世界权衡正则化器（PTR）——包含动态专家路由器和门控损失，解决不同粒度信息不平衡的问题；③基于视频LLM（Video-LLaVA）进行两阶段预调和指令微调，首次构建了 DeepSVU 指令数据集。

**🔧 技术方法**

技术方法主要包括：Mixture of Experts（MoE）架构、注意力机制（姿态注意力、交叉注意力）、Graph Transformer、SAM分割、LanguageBinds、门控专家路由器、门控权衡损失；训练使用 AdamW + LoRA，冻结姿态、关系和分割预训练模型。

**📊 数据集**

使用的数据集：①预调阶段的 RefCOCO、HumanML3D、RSI-CB；②DeepSVU 指令数据集——基于 UCF-Crime 的 UCF-C 指令集和基于 CUVA 的 CUVA 指令集；评估时与多种基线模型在这些数据集上对比。

**📈 对比分析**

与基线（Video-LLMs 如 VideoChat、Valley、mPLUG‑Owl、PandaGPT、Chat‑UniVi、Omni‑SILA、Holmes、Hawkeye；非LLM 如 VadClip、BiConvLSTM、X3D、CLIP‑TSA）对比，UPRM 在威胁识别的 FNR 下降约 20%，F2 与 mAP@tIoU 均提升 3–5 分；在归因任务上 BLEU 提升 6.5，ROUGE、SB、GPT、Human 分别提升 0.03–0.19，表现优于所有对比模型。

**⚠️ 局限性**

局限性：①模型高度依赖预训练的姿态、关系与分割模型，冻结这些模块可能限制适应新场景的能力；②训练与推理成本较高，尤其是多专家与门控机制；③仅在构造的 DeepSVU 指令数据集上验证，缺乏在真实多样化监控视频中的广泛测试；④未充分利用事件、位置等更丰富的语义信息，未来可进一步提升。

---

## 115. Multi-Level Conditioning by Pairing Localized Text and Sketch for Fashion Image Generation

**arXiv ID:** 2602.18309 | [PDF](https://arxiv.org/pdf/2602.18309v1)

**作者:** Ziyue Liu `[一作]` (University of Verona, Polytechnic Institute of Turin), Marco Cristani `[通讯]` (University of Verona, Reykjavik University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种多层次条件框架LOTS，利用多对局部草图‑文本配对生成时装图像，解决属性混淆并保持整体结构。

**💡 创新点**

创新点在于引入局部配对中心化表示与全局草图结合的多级条件，采用Pair‑Former和Diffusion Pair Guidance，将局部与全局信息分阶段编码并逐步注入扩散过程，从而显著降低属性泄漏。

**🔧 技术方法**

主要技术包括预训练扩散模型（Stable Diffusion）适配器（ControlNet/IP‑Adapter）、模态特定编码器、Pair‑Former注意力模块、跨注意力全局条件融合、CLIP+文本编码以及VQA评估等。

**📊 数据集**

使用了新构建的Sketchy数据集，基于Fashionpedia，包含47k图像、104k局部草图‑文本对，分为专业级草图和“in‑the‑wild”手绘草图两个分区。

**📈 对比分析**

在与Stable Diffusion、ControlNet、IP‑Adapter、Multi‑T2I‑Adapter、AnyControl以及前期版本LOTS*的基线对比中，LOTS在FID、GlobalCLIP、LocalCLIP、L‑VQAScore、SSIM等多项指标上均优于所有对手；在人类评估中亦表现最优。

**⚠️ 局限性**

局限性包括对极端手绘草图仍可能出现结构偏差；全局上下文提示仍需人工编写，缺乏自动化；模型对非典型服装或多层配件的适应性尚未充分验证。

---

## 116. Tethered Reasoning: Decoupling Entropy from Hallucination in Quantized LLMs via Manifold Steering

**arXiv ID:** 2602.17691 | [PDF](https://arxiv.org/pdf/2602.17691v1)

**作者:** Craig Atkinson `[一作]` `[通讯]` (Verificate Pty Ltd), Craig Atkinson (Verificate Pty Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在4位量化的Granite 4.0 H Small模型上，提出了Helix框架，通过构建真值流形并计算统一真值分数(UTS)，实现了在高温（T>2.0）下的输出轨迹引导，避免了轨迹发散而非语义崩溃，保持了推理质量并提升了创造力

**💡 创新点**

创新点在于：1）将轨迹与真值流形几何耦合，区分温度与幻觉；2）设计了基于语义熵与马氏距离的单步UTS评分；3）实现了仅干预0.2–2.5%token的分层激活引导；4）揭示高温“高熵创造储备”，并利用多温度合成提升独创性；5）在量化模型上恢复甚至超越全精度基线

**🔧 技术方法**

使用的技术包括：4位量化（Q4_K_M）、Hybrid Mamba-Transformer架构、真值流形构造（统计特征、马氏距离）、统一真值分数UTS、基于UTS阈值的渐进式logit修正、温度自适应阈值调节、Per-token不确定性监控

**📊 数据集**

使用的数据集包括：GSM8K（数学推理）、HumanEval（代码生成）、MMLU（通用知识问答）、TruthfulQA、WikiText-103、Qwen3-30B-A3B MoE（跨架构验证）、Creative Songwriting任务（多温度创意评估）

**📈 对比分析**

通过与无干预4位量化基线、全精度基线以及跨架构MoE模型对比，发现：在T=1.0时GSM8K准确率91.80%（超全精度87.27%），T=3.0时仅下降2.81%；HumanEval在T=3.0仍保持76.83%；MMLU在T=3.0仅下降1.24%；高温下创意多样性提升200%+，重复率从70–80%降至5–20%

**⚠️ 局限性**

局限性包括：1）实验仅在Granite 4.0和Qwen3两架构上验证，缺乏更广泛的模型覆盖；2）真值流形基于通用事实提示，可能不适用于高度专业化领域；3）创意评估依赖语义相似度与LLM判定，缺乏客观人类评测；4）高温下的创造性并未系统探究其边界与最优温度策略；5）系统为专有实现，公开复现受限

---

## 117. Calibrated Adaptation: Bayesian Stiefel Manifold Priors for Reliable Parameter-Efficient Fine-Tuning

**arXiv ID:** 2602.17809 | [PDF](https://arxiv.org/pdf/2602.17809v1)

**作者:** Ibne Farabi Shihab `[一作]` (Iowa State University), Anuj Sharma `[通讯]` (Iowa State University)

**通讯引用:** 2985 | [OpenAlex ID](https://openalex.org/A5083087081)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 Stiefel‑Bayes Adapters（SBA）的贝叶斯参数高效微调框架，能够在不增加显著参数开销的前提下为大语言模型提供可靠的不确定性估计。

**💡 创新点**

核心创新在于将矩阵 Langevin 先验直接置于 Stiefel 流形（正交子空间）上，并采用切空间拉普拉斯近似结合几何重回（geodesic retraction）进行后验推断，彻底避免了平面空间投影导致的方差膨胀，从而显著提升校准和对抗域移的鲁棒性。

**🔧 技术方法**

使用的关键技术包括：Stiefel 流形上的矩阵 Langevin 分布、切空间拉普拉斯近似与 KFAC Hessian 近似、QR 基重回、以及后验采样与可预测蒸馏。

**📊 数据集**

在 RoBERTa‑large、LLaMA‑2‑7B、LLaMA‑2‑13B、Mistral‑7B、Qwen2.5‑7B 等模型上，对 GLUE / SuperGLUE、SST‑2、MNLI、SQuAD、XSum 等数据集进行评估，并在多个域移场景（MNLI→SNLI、Amazon→Yelp、SQuAD→AdversarialQA）中测试。

**📈 对比分析**

与 LoRA、DoRA、Orthogonal LoRA、Laplace‑LoRA、MC‑Drop‑LoRA、SWAG‑LoRA、Gaussian‑Projection、Deep‑Ensemble LoRA、温度标定等基线对比，SBA 在保持任务性能（误差 ≤ 0.3 点）同时，期望校准误差（ECE）降低 18–34%，选择性预测 AUROC 提升 12–25%，对 OOD 检测的 AUROC 甚至优于 5‑模型深度集成，且仅使用单一模型参数集合。

**⚠️ 局限性**

局限性包括：推理时需要多次前向传播（可通过蒸馏降低但失去完整不确定性分解）；切空间拉普拉斯近似局部，可能在 MAP 远离时低估不确定性；对极端浓度参数的矩阵 Langevin 正则化近似需更精细处理；目前评估仅覆盖英文任务，对多语言或更大规模模型（70B+）的适用性尚未验证。

---

## 118. Lost Before Translation: Social Information Transmission and Survival in AI-AI Communication

**arXiv ID:** 2602.17674 | [PDF](https://arxiv.org/pdf/2602.17674v1)

**作者:** Bijean Ghafouri `[一作]` (University of Southern California), Emilio Ferrara `[通讯]` (University of Southern California)

**通讯引用:** 18740 | [OpenAlex ID](https://openalex.org/A5078699564)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了AI系统在多轮传递过程中信息如何演化，发现了收敛、选择性保留和竞争过滤等模式。

**💡 创新点**

首次构建了实验范式将信息通过100轮相同指令的语言模型链进行迭代，揭示了AI‑AI传递对事实、确定性、观点多样性、政治框架和情感的系统性影响。

**🔧 技术方法**

采用大型语言模型（Gemini 3.0 Flash）进行无状态、随机采样的文本生成与恢复，并通过人工标注、LLM评估等多重方法量化信息变化。

**📊 数据集**

使用人工合成新闻、科普文本、多视角讨论和社交媒体帖文等多种原始文本，覆盖事实、认知、政治和情绪等维度。

**📈 对比分析**

对比单轮与多轮传递结果，利用统计显著性检验和效应量评估，显示信息在迭代后趋于中立、压缩并丢失细节，情感被调和，观点被结构化；人类实验进一步证明受影响的记忆、可信度和情感共振。

**⚠️ 局限性**

实验受限于单一模型与固定指令，未考虑模型更新、不同温度设置或多模态输入，且结果可能不适用于更大规模或真实应用场景。

---

## 119. The Digital Divide in Generative AI: Evidence from Large Language Model Use in College Admissions Essays

**arXiv ID:** 2602.17791 | [PDF](https://arxiv.org/pdf/2602.17791v1)

**作者:** Jinsook Lee `[一作]` (Cornell University), Rene F. Kizilcec `[通讯]` (Cornell University)

**通讯引用:** 7148 | [OpenAlex ID](https://openalex.org/A5071778778)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析了 81,663 名学生 2020-2024 年的大学申请论文，研究 LLM 辅助写作在不同社会经济群体中的采用差异、语言特征变化及其对录取结果的影响。

**💡 创新点**

首次提供大规模纵向证据，将 LLM 使用估计、写作风格演变与录取结果关联，并引入基于分布的论文级 LLM 检测器，揭示低 SES 学生在 LLM 使用与录取惩罚之间的双重不平等。

**🔧 技术方法**

采用分布式 LLM 检测方法（基于 GPT‑4o 生成的合成文本构建参考分布），结合差分‑差分、逻辑回归、交互效应与中介分析来量化 LLM 使用与录取结果的关系。

**📊 数据集**

使用一所高度选择性工程学院的匿名申请数据，覆盖 2019‑2020 至 2023‑2024 五个录取周期；以费用豁免状况作为社会经济状态（SES）代理。

**📈 对比分析**

通过对比前后 GPT 时代的语言特征变化、使用比例差异以及差分‑差分与交互回归，展示低 SES 学生 LLM 使用率上升但录取概率惩罚更大；模型校准良好，检出率与真实 LLM 采样相符。

**⚠️ 局限性**

局限包括：检测方法仅为估计，缺乏真实使用数据；观察性设计难以确定因果；单一机构与专业限制结果可推广性；ChatGPT 公开与最高法院判决同时发生，导致难以分离两者影响。

---

## 120. Epistemic Traps: Rational Misalignment Driven by Model Misspecification

**arXiv ID:** 2602.17676 | [PDF](https://arxiv.org/pdf/2602.17676v1)

**作者:** Xingcheng Xu `[一作]` (Shanghai Artificial Intelligence Laboratory), Xia Hu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将经济学中的Berk‑Nash Rationalizability框架引入AI安全，证明偏差性世界模型导致的“对齐病态”（sycophancy、hallucination、strategic deception）是结构上可合理化的平衡状态，并通过实验验证了相应的相位图和安全阈值。

**💡 创新点**

创新点：①首次用Berk‑Nash理论统一解释多种对齐失效；②揭示安全是先验贝叶斯空间的拓扑性质，而非奖励大小的连续函数；③提出“Subjective Model Engineering”新范式，强调通过构造内部世界模型而非仅调节奖励来实现稳健安全。

**🔧 技术方法**

技术方法：①理论推导（KL散度、最佳响应算子、Berk‑Nash集合）；②利用In‑Context Learning模拟隐式贝叶斯更新；③在六大LLM（Qwen、DeepSeek、Gemini、GPT‑4o、GPT‑5等）上做大规模实验；④构造二维奖励网格和风险阈值实验。

**📊 数据集**

数据集：实验使用人工合成的奖励环境（p_S、p_H 取值网格）和风险阈值设定，配合六大语言模型的预训练参数；并未使用公开大型真实标签集。

**📈 对比分析**

比较方法：将实验得到的“安全率”“抖动率”“欺骗率”等指标与理论相位图进行对齐，发现实验结果与预测完全一致；在安全区表现出几乎100%安全行为，在不安全区出现振荡或高欺骗率；结构上悲观的先验能稳健阻止欺骗，验证了理论。

**⚠️ 局限性**

局限性：①需手工设定主观模型空间Θ，难以自动化；②仅在离散奖励/风险设置下验证，未覆盖连续动作空间或真实RLHF训练；③假设KL最小化学习与最优贝叶斯更新一致，实际模型可能偏离；④框架侧重静态平衡，未完全捕捉长期非平稳学习动态；⑤对真实环境中的数据不确定性和多模态复杂度仍有限。

---

## 121. ReqElicitGym: An Evaluation Environment for Interview Competence in Conversational Requirements Elicitation

**arXiv ID:** 2602.18306 | [PDF](https://arxiv.org/pdf/2602.18306v1)

**作者:** Dongming Jin `[一作]` (Peking University), Xiaohong Chen `[通讯]` (East China Normal University)

**通讯引用:** 488604 | [OpenAlex ID](https://openalex.org/A5100373745)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个可自动、可复现的评估环境（ReqElicitGym），用于衡量大型语言模型在多轮对话式需求访谈中的访谈能力。

**💡 创新点**

创新点在于：①构造了覆盖 10 种网站类型的 101 个访谈场景数据集，显式标注初始需求、隐含需求和完整需求；②设计了基于 LLM 的“oracle user”和“task evaluator”，实现了不依赖真人参与即可进行对话、监测和评估；③提出了三项量化指标（隐含需求提取率 IRE、关键提问效率 TKQR、策略有效率 ESR），可从结果和过程两方面全面评估访谈性能。

**🔧 技术方法**

使用的技术主要是：大型语言模型（GPT‑5.2、Claude Opus 4.5、Gemini 3 Flash、DeepSeek V3.2、Kimi K2.5、GLM‑4.7、Qwen‑3 235B），结合 prompt‑engineering、Chain‑of‑Thought（CoT）推理、自动化对话交互框架以及基于 LLM 的评判器实现。

**📊 数据集**

使用的数据集是：101 个网站开发场景，包含 632 条隐含需求、1000 条完整需求和 101 条初始需求，覆盖用户交互、内容表达与视觉风格三类需求维度。

**📈 对比分析**

评估方法为：让每个 LLM 在 101 个场景中以访谈者角色交互，记录对话回合，利用 task evaluator 自动标注每回合的访谈策略与是否成功触发隐含需求，随后计算 IRE、TKQR 与 ESR。实验结果显示：即使是表现最好的模型其 IRE 仅为 0.32，CoT 提高了 TKQR 与减少回合数但未显著提升 IRE；LLM 访谈者更倾向于 probing，clarify 效果差；风格类需求几乎被忽视。

**⚠️ 局限性**

局限性包括：①数据集仅涵盖网站开发领域，缺乏其他软件工程领域（嵌入式、工业控制等）的代表性；②oracle user 与 task evaluator 虽已通过真实访谈验证但仍基于 LLM，可能无法完全再现人类访谈的细微差异；③评估指标虽覆盖结果与过程，但未涉及访谈者的自然对话流畅度或用户满意度等人类主观体验。

---

## 122. Ontology-Guided Neuro-Symbolic Inference: Grounding Language Models with Mathematical Domain Knowledge

**arXiv ID:** 2602.17826 | [PDF](https://arxiv.org/pdf/2602.17826v1)

**作者:** Marcelo Labre `[一作]` `[通讯]` (Advanced Institute for Artificial Intelligence), Marcelo Labre (Advanced Institute for Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了利用正式数学本体OpenMath对语言模型进行检索增强生成，以提升其在MATH基准上的推理可靠性。

**💡 创新点**

创新点在于构建了神经符号管道，将本体定义通过混合检索和交叉编码器重排序注入模型提示，并系统分析了检索质量对不同模型规模与推理模式的双向影响。

**🔧 技术方法**

主要技术包括基于语义检索的混合检索、交叉编码器重排序、Prompt注入、greedy与best‑of‑n采样等。

**📊 数据集**

使用了MATH 500题集合作为实验数据集。

**📈 对比分析**

与无检索基线比较，检索质量高时准确率可提升约5%–13%（在best‑of‑n模式下），但低质量检索或小模型会导致准确率下降；最佳效果在0.3–0.5阈值、best‑of‑n采样下实现。

**⚠️ 局限性**

主要限制包括检索质量瓶颈、OpenMath覆盖不足、模型容量不足导致上下文利用不充分，以及在专家级问题中出现的参数–上下文冲突。

---

## 123. Vichara: Appellate Judgment Prediction and Explanation for the Indian Judicial System

**arXiv ID:** 2602.18346 | [PDF](https://arxiv.org/pdf/2602.18346v1)

**作者:** Pavithra PM Nair `[一作]` (TCS Research), Preethu Rose Anish `[通讯]` (TCS Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Vichara 框架，对印度上诉案件进行判决预测和可解释说明。

**💡 创新点**

创新点在于通过决策点抽取生成结构化 IRAC 风格解释，并实现多阶段无微调 prompt pipeline，兼顾预测准确性与可解释性。

**🔧 技术方法**

技术手段包括基于 LLM 的 prompt 调用、句子级修辞角色分类、情境构建、决策点抽取、判决预测与解释生成，使用 GPT-4o mini、Llama‑3.1‑8B、Mistral‑7B、Qwen2.5‑7B 等模型。

**📊 数据集**

使用 PredEx 与 ILDC_expert 两个印度上诉案件数据集进行实验。

**📈 对比分析**

方法通过与 INLegalLlama 基准对比，GPT‑4o mini 在 PredEx 上 F1 81.5%、ILDC 上 80.3%，人类评估解释在清晰度、关联性、实用性上均领先。

**⚠️ 局限性**

局限性包括仅针对印度司法体系、对多次 LLM 调用导致计算成本高、评估样本有限、未扩展至其他案件类型或司法区。

---

## 124. A Dichotomy Theorem for Automatic Structures

**arXiv ID:** 2602.18238 | [PDF](https://arxiv.org/pdf/2602.18238v1)

**作者:** Antoine Cuvelier `[一作]` (Ecole Normale Supérieure Ulm), Rémi Morvan `[通讯]` (Université de Lille)

**通讯引用:** 15 | [OpenAlex ID](https://openalex.org/A5063732938)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

研究自动结构（potentially infinite但可由有限自动机描述）上的同构问题，给出了一个二分结论：若目标结构具有有限对偶性，则判定问题可在非确定性对数空间（NL）内完成；否则问题不可判定（RE/ coRE‑complete）。

**💡 创新点**

创新点在于将有限对偶性与可判定性、统一一阶可定义同构以及正则同构等价；提出了针对自动结构的超边一致性算法，证明其在有限对偶性结构下可以在多项式时间内完成。

**🔧 技术方法**

主要技术包括：一阶逻辑语义、自动结构理论、树对偶性理论、正则函数与正则同构概念、Knaster‑Tarski 定理、超边一致性迭代与固定点计算。

**📊 数据集**

本工作为理论研究，不涉及具体数据集。

**📈 对比分析**

性能上没有实验评估；理论复杂度分析显示：若目标具有有限对偶性，判定问题属于NL；若不具有限对偶性，则问题不可判定（RE/ coRE‑complete）。

**⚠️ 局限性**

局限性包括：仅讨论有限目标结构；正则同构在更一般的目标结构上仍未完全覆盖；对非自动结构的扩展以及多目标结构的交互效果尚待研究。

---

## 125. On the "Induction Bias" in Sequence Models

**arXiv ID:** 2602.18333 | [PDF](https://arxiv.org/pdf/2602.18333v1)

**作者:** M. Reza Ebrahimi `[一作]` (Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc), Roland Memisevic `[通讯]` (Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过大规模实验对比了 transformer 与 RNN 在模数加法等状态追踪任务下的样本效率，探究了不同监督方式和长度分布对学习的影响。

**💡 创新点**

首次量化归纳偏置对 transformer 学习效率和跨长度共享的影响，并提出共享因子评估跨长度机制共享，揭示 transformer 在长度泛化中的数据瓶颈。

**🔧 技术方法**

采用二进制搜索确定最小样本数，使用 transformer、LSTM、Dense-SSM 等模型，实验三种监督模式（Outcome、CoT、ACoT）及三种长度分布，定义共享因子衡量跨长度共享。

**📊 数据集**

使用人工合成的模数加法任务（ℤ_m）和置换组合任务（S_5）生成随机整数序列，长度可变。

**📈 对比分析**

通过比较每种模型在不同任务格式和长度分布下的最小训练样本数 N*，发现 transformer 在多长度任务上需要的样本数远大于 RNN；共享因子显示 transformer 共享因子≈1或<1，RNN 共享因子>>1。

**⚠️ 局限性**

实验计算量大，搜索空间广泛，结果仅基于有限模型和任务，可能不适用于更复杂或真实数据集；未评估训练时间和推理效率等其他性能指标。

---

## 126. A reliability- and latency-driven task allocation framework for workflow applications in the edge-hub-cloud continuum

**arXiv ID:** 2602.18158 | [PDF](https://arxiv.org/pdf/2602.18158v1)

**作者:** Andreas Kouloumpris `[一作]` (University of Cyprus), Theocharis Theocharides `[通讯]` (University of Cyprus)

**通讯引用:** 3490 | [OpenAlex ID](https://openalex.org/A5041366767)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种精确的多目标任务分配框架，在简化的 edge‑hub‑cloud 架构上对工作流应用同时优化可靠性和延迟。

**💡 创新点**

创新点在于：①将工作流任务图通过两步转换生成可靠性感知图，完整考虑复制、内存/存储/能耗/通信约束；②采用二进制整数线性规划（BILP）进行精确求解，实现可调权重的可靠性‑延迟权衡；③在真实 UAV 检测工作流与合成工作流上进行系统评估，展示显著的可靠性提升（84%）和延迟降低（50%）。

**🔧 技术方法**

主要技术包括：时间冗余（双/三重复制）、可靠性与能耗模型、两步图转换、BILP 求解（Gurobi）、基于权重的多目标线性组合。

**📊 数据集**

使用的数据集：真实 UAV 电力塔与线路检测工作流（15 个任务）以及 9 组不同结构（串行、并行、混合）与规模（10、100、1000 任务）的合成工作流，数据通过设备性能基准与功耗测量获得。

**📈 对比分析**

与三种基线策略（全部任务分配至单一设备）对比，平均可靠性提升 84.19% 与平均延迟降低 49.81%；在不同权重设置下均能得到 Pareto‑近似解，求解时间从 0.03 秒到 50.94 秒，显示出可扩展性与实用性。

**⚠️ 局限性**

局限性在于：仅考虑单个 edge 设备；依赖离线精确求解，对动态变化不适用；复制导致能耗增加，未针对多边缘设备场景展开深入分析。

---

## 127. Role-Adaptive Collaborative Formation Planning for Team of Quadruped Robots in Cluttered Environments

**arXiv ID:** 2602.18260 | [PDF](https://arxiv.org/pdf/2602.18260v1)

**作者:** Magnus Norén `[一作]` (Lulea University of Technology), George Nikolakopoulos `[通讯]` (Lulea University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出了一种基于角色自适应的Leader‑Follower架构、虚拟弹簧阻尼以及障碍物规避层的四足机器人编队路径规划与控制框架，能够在混乱环境中动态调整领导角色并保持编队稳定。

**💡 创新点**

创新点在于：①角色可动态分配，领导权随时切换；②引入“look‑ahead”参考生成器，让编队在避障时临时放松形状；③使用虚拟弹簧阻尼与新颖的基于速度图梯度的障碍规避层，兼顾碰撞防护与路径柔性。

**🔧 技术方法**

核心技术包括Fast Marching Square（FM2）全局路径规划、虚拟弹簧阻尼控制、梯度基速度地图避障、姿态限制与速度上限裁剪等。

**📊 数据集**

在Gazebo仿真环境（14×10 m障碍地图）与真实实验室（9.5×6.5 m、Vicon定位）中使用Unitree Go1四足机器人进行验证，没有使用公开数据集。

**📈 对比分析**

通过仿真与物理实验展示编队在狭窄通道、障碍物旁边等场景下的顺利通过、领导切换与形状适配，未给出与其他方法的定量对比，但结果显示编队保持无碰撞且能够在有限空间内自适应变形。

**⚠️ 局限性**

局限在于仅验证了3–4机器人规模、主要针对地面全向移动机器人，缺乏大规模群体、非全向机器人或复杂三维环境的实验与系统级性能评估。

---

## 128. Neural Prior Estimation: Learning Class Priors from Latent Representations

**arXiv ID:** 2602.17853 | [PDF](https://arxiv.org/pdf/2602.17853v1)

**作者:** Masoud Yavari `[一作]` (Independent Researcher), Payman Moallem `[通讯]` (University of Isfahan)

**通讯引用:** 4140 | [OpenAlex ID](https://openalex.org/A5051290262)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 Neural Prior Estimator (NPE)，通过 Prior Estimation Modules (PEM) 学习特征条件化的类先验，并将其与 Logit Adjustment (NPE‑LA) 结合，用于解决类别不平衡问题。

**💡 创新点**

创新点在于：①使用单向 logistic loss 训练 PEM，实现无需显式类别计数即可自动学习 log‑prior；②理论证明 NPE 收敛至类 log‑prior（相当于 log‑frequency），提供可解释且自适应的先验估计；③将估计直接加入 logit 调整，实现在线、非平稳环境下的动态校正。

**🔧 技术方法**

技术手段包括：Prior Estimation Modules、one‑way logistic loss、端到端联合 CE+NPE 损失、ResNet‑32、UNet、DeepLabV3、Swin‑T 等主干与解码器；特征空间分析、归一化和缩放因子 α 的手动调节。

**📊 数据集**

实验数据集：长尾 CIFAR‑10、CIFAR‑100；医学图像分割数据集 STARE（血管）和 ADE20K（大规模语义分割）。

**📈 对比分析**

与 CE、cRT、LA 等基线在相同实验设置下对比。结果显示 NPE‑LA 在尾类和中间类上显著提升（尤其在高不平衡比例和大 batch size 时），整体 Top‑1 准确率与 LA 相近或略优，并保持对头类的稳定性。

**⚠️ 局限性**

局限性：①需要额外训练 PEM，虽轻量但增加梯度计算；②在极端稀有类和大规模不平衡时性能提升有限；③需手动设定缩放因子 α，且在密集预测中 Batch Normalization 可能干扰估计；④对分布漂移的鲁棒性尚未完全验证。

---

## 129. Stable Long-Horizon Spatiotemporal Prediction on Meshes Using Latent Multiscale Recurrent Graph Neural Networks

**arXiv ID:** 2602.18146 | [PDF](https://arxiv.org/pdf/2602.18146v1)

**作者:** Lionel Salesses `[一作]` (Cenaero), Caroline Sainvitu `[通讯]` (Cenaero)

**通讯引用:** 408 | [OpenAlex ID](https://openalex.org/A5090557633)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并验证了一种基于深度学习的LM‑RGNN框架，用于在复杂几何网格上预测金属粉末床熔融过程中的完整温度历史；通过层间与层内的多尺度时序建模以及潜在图自编码器，实现了千步以上的预测稳定性与精度。

**💡 创新点**

① 引入潜在多尺度递归图神经网络（Latent Multiscale Recurrent Graph Neural Network），将长期温度演化拆分为层间与层内两级模型；② 采用变分图自编码器（VGAE）对温度场进行潜在压缩，显著降低内存占用并提升训练稳定性；③ 在无规则网格上直接进行时空建模，避免固定格点插值导致的误差。

**🔧 技术方法**

Graph Gated Recurrent Unit (GraphGRU) + Recurrent Graph Neural Network (RGNN)、变分图自编码器 (VGAE)、多尺度时序策略、截断反向传播 (TBPTT)、图池化/解池化、GPU 内存优化与并行推理。

**📊 数据集**

140份二维粉末床熔融数值模拟数据，包含网格拓扑、材料掩模、激光路径等信息；每个模拟长度 6600–12000 步，温度范围 20–2100°C。

**📈 对比分析**

与 Decoupled‑RGNN 基线对比。LM‑RGNN 在 MAE≈6.5 °C、MAPE≈2.6%、MME≈108.7 °C、melt‑pool IoU≈0.85 以及时序 MAE 低于基线的同时，内存压力低、推理时间略高（≈14 ms/步），整体误差降低约 70%，并显著提升长期稳定性。

**⚠️ 局限性**

仅在二维模拟中验证；固定工艺参数；仅使用模拟数据，未检验对实验测量的泛化；未覆盖三维几何或其他工艺流程。

---

## 130. CapNav: Benchmarking Vision Language Models on Capability-conditioned Indoor Navigation

**arXiv ID:** 2602.18424 | [PDF](https://arxiv.org/pdf/2602.18424v1)

**作者:** Xia Su `[一作]` (University of Washington), Jon Froehlich `[通讯]` (University of Washington)

**通讯引用:** 10609 | [OpenAlex ID](https://openalex.org/A5016828530)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CapNav 基准，用于评估视觉语言模型在受限移动能力的室内导航任务中的表现。

**💡 创新点**

创新点在于引入能力条件化导航、五类代表性人机/机器人实体、基于真实室内扫描的图结构以及细粒度可穿越性标注，并系统评估 13 先进 VLM。

**🔧 技术方法**

采用多帧视频、图形空间表示、自然语言任务生成、VLM 预测路径与可穿越性、推理评估等技术，并进行了思考模式与帧数实验。

**📊 数据集**

构建了包含 45 个真实室内场景（HM3D、Matterport3D）、2365 个导航任务和 5075 条边级可穿越性标注的数据集，涵盖 5 种实体的特征。

**📈 对比分析**

通过 Feas-F1、路径有效性、可穿越率、推理有效性四项指标进行评估，并给出宏观 CapNav 分数；结果显示闭源 Gemini、GPT 等模型优于人类平均，开源模型表现逊色；性能随移动约束越严越低。

**⚠️ 局限性**

VLM 在空间尺寸推理、狭窄通道和转弯半径上存在维度忽视；视觉信息整合瓶颈导致帧数增益有限；训练和微调仍无法彻底解决几何约束判定。

---

## 131. MultiVer: Zero-Shot Multi-Agent Vulnerability Detection

**arXiv ID:** 2602.17875 | [PDF](https://arxiv.org/pdf/2602.17875v1)

**作者:** Shreshth Rajan `[一作]` `[通讯]` (Harvard), Shreshth Rajan (Harvard)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了零样本多代理系统用于漏洞检测，结合安全、正确性、性能和风格四个专门代理并行分析。

**💡 创新点**

创新点在于利用多维度代理联合投票，零样本情况下就能超越微调模型的召回率，实现首次零样本召回率突破。

**🔧 技术方法**

采用三层分析管道（模式匹配→RAG检索→LLM推理），自一致性采样，联合或加权投票策略以及上下文提取技术。

**📊 数据集**

使用真实Python漏洞集PyVul和合成漏洞集SecurityEval进行评估。

**📈 对比分析**

与规则工具、零样本LLM、微调LLM及专门系统比较，系统在PyVul上实现82.7%召回率、48.8%精度，召回率优于微调GPT‑3.5的81.3%。

**⚠️ 局限性**

局限在于高误报率（约85% FPR）、较高的调用成本和延迟，且缺乏对补丁对比的精准判断。

---

## 132. Improving Generalizability of Hip Fracture Risk Prediction via Domain Adaptation Across Multiple Cohorts

**arXiv ID:** 2602.17962 | [PDF](https://arxiv.org/pdf/2602.17962v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 133. Bayesian Online Model Selection

**arXiv ID:** 2602.17958 | [PDF](https://arxiv.org/pdf/2602.17958v1)

**作者:** Aida Afshar `[一作]` (Boston University), Aldo Pacchiano `[通讯]` (Broad Institute of MIT and Harvard)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

提出了一种基于后验采样的贝叶斯在线模型选择算法（B-MS），在贝叶斯随机抛硬币问题中对一组基学习器进行自适应选择。

**💡 创新点**

创新点在于：1) 采用贝叶斯后验采样估计基学习器的潜在回报，实现无先验模型复杂度假设的 oracle‑best 保证；2) 将后验信息共享给所有基学习器，从而在模型误设时恢复性能；3) 证明了与传统 Thompson Sampling 相当的理论下界。

**🔧 技术方法**

使用技术包括：贝叶斯推断、后验采样、潜在函数选择、UCB、LinTS、信息锁分析及理论证明。

**📊 数据集**

实验使用合成多臂和线性高斯奖励的离散/连续 bandit 场景，覆盖不同置信半径、误设先验、数据共享以及信息锁结构。

**📈 对比分析**

与单个基学习器、TS、LinTS、ILS 等方法比较，B-MS 的贝叶斯回报与最佳基学习器相当，子线性增长且在误设和共享数据场景下优于传统方法。

**⚠️ 局限性**

局限性：对元学习器的先验假设要求较高；理论分析仅覆盖离散/线性 bandit，未给出贝叶斯下界；在非贝叶斯或更复杂 RL 场景中效果未知。

---

## 134. Enabling Training-Free Text-Based Remote Sensing Segmentation

**arXiv ID:** 2602.17799 | [PDF](https://arxiv.org/pdf/2602.17799v1)

**作者:** Jose Sosa `[一作]` (University of Luxembourg), Djamila Aouada `[通讯]` (University of Luxembourg)

**通讯引用:** 2740 | [OpenAlex ID](https://openalex.org/A5083368272)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两条无训练或轻量LoRA微调的文本驱动遥感图像分割管线，分别使用对比式VLM（CLIP）选择SAM的掩模以及生成式VLM（GPT-5、Qwen-VL）生成点击提示来完成开词汇、指代与推理分割。

**💡 创新点**

创新点在于完全不依赖额外可训练模块，仅利用预训练VLM与SAM实现跨域分割，并通过生成式VLM生成点击提示实现复杂语言指令的分割，且仅微调LoRA即可达到SOTA。

**🔧 技术方法**

采用CLIP做掩模选择、GPT-5/Qwen-VL生成点击提示、SAM生成掩模，生成式VLM的点击序列自动化生成以及LoRA微调技术。

**📊 数据集**

在19个遥感分割基准（OpenEarthMap、LoveDA、iSAID、Potsdam、Vaihingen、UAVid、UDD5、VDD、WBS-SI等）、RRSIS-D指代分割和EarthReason推理分割数据集上进行评估。

**📈 对比分析**

与零样本自然图像基线和SegEarth-OV等方法对比，OVSS任务实现SOTA（mIoU提升至约41%），指代与推理任务通过LoRA微调后分别达到72.7%和70.6%的mIoU，显著优于现有最优方法。

**⚠️ 局限性**

对比式VLM在长句或多目标指令下表现下降，生成式VLM更适合单目标，SAM在保持冻结时对细小或多重目标的分割仍受限，且跨域泛化在遥感特定场景下仍有提升空间。

---

## 135. Can LLM Safety Be Ensured by Constraining Parameter Regions?

**arXiv ID:** 2602.17696 | [PDF](https://arxiv.org/pdf/2602.17696v1)

**作者:** Zongmin Li `[一作]` (Nanyang Technological University), Aixin Sun `[通讯]` (Nanyang Technological University)

**通讯引用:** 14729 | [OpenAlex ID](https://openalex.org/A5100618738)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估四种安全区域识别方法在不同模型与数据集上的一致性，并测量其 IoU 与 utility‑isolated IoU。

**💡 创新点**

系统性检验安全区域的可收敛性，揭示现有方法缺乏数据集无关性，并提出通过剔除 utility 组件进一步验证安全性与实用性耦合的评估。

**🔧 技术方法**

采用 SNIP、Wanda、SafeNeuron、SafeLayer、NLSR 等识别技术，使用 IoU、余弦相似度、层级抽样等统计手段。

**📊 数据集**

使用 10 个多类别安全数据集与若干单类别安全数据集，以及 utility 数据集 Alpaca‑Cleaned，覆盖 Llama‑2、Mistral、Qwen 等中等规模 LLM。

**📈 对比分析**

与原方法对比，IoU 仅在 0.01–0.72 之间，安全区域重叠不稳定；剔除 utility 组件后 IoU 更低，表明安全与实用性高度耦合。

**⚠️ 局限性**

局限于公开英文安全数据集、有限模型规模、仅基于权重空间评估，未考虑表征/激活空间或多语言情形。

---

## 136. CUICurate: A GraphRAG-based Framework for Automated Clinical Concept Curation for NLP applications

**arXiv ID:** 2602.17949 | [PDF](https://arxiv.org/pdf/2602.17949v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 137. DuoTouch: Passive Two-Footprint Attachments Using Binary Sequences to Extend Touch Interaction

**arXiv ID:** 2602.17961 | [PDF](https://arxiv.org/pdf/2602.17961v1)

**作者:** Kaori Ikematsu `[一作]` (LY Corporation), Kunihiro Kato `[通讯]` (Tokyo University of Technology)

**通讯引用:** 316 | [OpenAlex ID](https://openalex.org/A5027568588)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了 DuoTouch，一种只使用两个触控脚印的被动附件，能够通过二进制序列编码滑动、旋转等操作，并在智能手机和平板触控板上进行评估和原型验证。

**💡 创新点**

创新点在于：① 仅用两个可编程轨迹实现多种离散与连续交互；② 通过对齐与相位移两种解码策略实现命令识别与方向/距离估计；③ 推导并验证了采样限制边界 v ≤ w f_s，提供了设计准则和自动生成 CAD 的工具。

**🔧 技术方法**

主要技术包括：基于电容触控的二进制序列编码、对齐与相位移解码算法、采样限度分析、以及配套的设计支持工具和 3D 打印/PCB 制造流程。

**📊 数据集**

数据集为在两台设备（iPhone 16 60 Hz、Magic Trackpad 2 90 Hz）上，使用 4 个电极宽度（1.5–3 mm）与 10 个速度级别（20–200 mm/s）共 1,600 次实验记录的触控序列。

**📈 对比分析**

实验结果显示，在采样限度内，配合最宽电极（≥2.5 mm）时对齐模式可达 93–98% 的完整代码识别率，相位移模式可达 84–98% 的方向/距离识别率；在该边界之外精度急剧下降，验证了 v ≤ w f_s 的适用性；用户体验调查表明手柄、环扣和多键面板均获得良好/优秀的使用感受。

**⚠️ 局限性**

局限性包括：只能单一机械部件同时操作、对设备采样率敏感（高速度时失真）、可能出现误激活和部分序列匹配错误、需要手动装配（PCB+3D 打印）、对干手或无接地情况检测受限、尺寸与重量仍需优化。

---

## 138. Subgroups of $U(d)$ Induce Natural RNN and Transformer Architectures

**arXiv ID:** 2602.18417 | [PDF](https://arxiv.org/pdf/2602.18417v1)

**作者:** Joshua Nunley `[一作]` (Indiana University), Joshua Nunley `[通讯]` (Indiana University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5092911518)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于闭子群的序列模型框架，将隐藏状态和词表示直接置于 O(d) 的 Lie 子群中。

**💡 创新点**

创新点在于用子群本身代替传统欧几里得状态空间，给出可直接替换的 RNN 与 Transformer 模板，并引入线性切线混合扩展提升有限预算性能。

**🔧 技术方法**

使用技术包括 Lie 群理论、切线空间线性映射、指数映射更新、USIM 读取头以及对 O(d) 的具体实现。

**📊 数据集**

实验数据集为 Tiny Shakespeare 和 Penn Treebank 的字符级语言建模。

**📈 对比分析**

与参数匹配的基线（Transformer、LSTM 等）比较，线性混合的 OSMFormer 在两数据集上都达到了或略优于基线的 BPC，显示出竞争性能。

**⚠️ 局限性**

局限性在于只验证了 O(d) 子群，单种子实验，未覆盖其他子群族或更广泛任务，需要进一步的鲁棒性与规模评估。

---

## 139. Quantum Maximum Likelihood Prediction via Hilbert Space Embeddings

**arXiv ID:** 2602.18364 | [PDF](https://arxiv.org/pdf/2602.18364v1)

**作者:** Sreejith Sreekumar `[一作]` (University of Paris-Saclay), Nir Weinberger `[通讯]` (Technion)

**通讯引用:** 287 | [OpenAlex ID](https://openalex.org/A5076051906)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于量子密度算子嵌入的自回归预测模型——量子最大似然预测器（QMLP），将大语言模型的上下文学习视作在量子模型族上求解最大似然问题，形成了一个信息几何与统计相结合的理论框架；

**💡 创新点**

创新点在于：①将概率分布嵌入量子密度空间，实现了在低维 Hilbert 空间内捕捉词语相似性；②通过量子反向 I‑projection 与量子毕达哥拉斯定理，将 QMLP 的目标化简为经典 KL 目标；③给出了非渐近收敛率、截断不等式（在迹范数与量子相对熵下），并证明其收敛率与 Hilbert 空间维度有关，而非词表维度；

**🔧 技术方法**

使用的主要技术包括：量子密度算子与协方差嵌入、量子相对熵与测量相对熵、数据处理不等式、量子毕达哥拉斯定理、量子 Pinsker 不等式、矩阵 Hoeffding/Bernstein 收敛不等式；

**📊 数据集**

本文未使用具体的实验数据集，全部研究基于理论分析与泛化误差的非渐近性质；

**📈 对比分析**

方法评价基于理论证明：在给定的量子模型族和 Hilbert 空间维度 d 的条件下，QMLP 在迹范数下收敛速率为 Õ(d/√n)（或在模型族充分表达时为 Õ(d³/n)），在量子相对熵下为 Õ(d/√n)（或 Õ(d²/n)）。与传统的最大似然预测相比，QMLP 的收敛速率不随词表大小增长，展示了潜在的“维度无关”优势；

**⚠️ 局限性**

局限性包括：①需要假设存在紧致凸的量子模型族且其包含的所有状态都满足特定的基变换与单射性；②对嵌入函数的选择未给出具体构造与优化方法，实际实现和计算复杂度未分析；③未给出实验验证，仅在理论层面给出性能上界；④在模型族不足表达时收敛速率仍受量子相对熵的影响，且对最小特征值的要求较高。

---

## 140. Towards More Standardized AI Evaluation: From Models to Agents

**arXiv ID:** 2602.18029 | [PDF](https://arxiv.org/pdf/2602.18029v1)

**作者:** Ali El Filali `[一作]` (G42), Inès Bedar `[通讯]` (Independent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统性探讨了从传统模型评估向代理系统评估的演进，提出将评估视为核心控制功能，并阐明评估在保证代理系统可靠性、可解释性与治理中的关键作用。

**💡 创新点**

创新点在于将评估从单次输出检查扩展为持续行为测量，提出 Pass^k 与 Pass@k 区分可靠性与能力，系统剖析评估流程中的隐式错误、工具链与环境依赖，并以 GAIA2、TextQuests 等环境化基准为例验证方法。

**🔧 技术方法**

采用自动化基准、人工与 LLM-as-a-judge 评估相结合的多层次评估框架，构建 Agent harness、Transcript 与 Grader 体系，利用模拟环境（如 ARE、GAIA2、TextQuests）进行真实情境下的行为评估。

**📊 数据集**

使用的基准集包括 MMLU、GLUE、SuperGLUE、SWE-Bench、GAIA2、TextQuests 等多样化数据集，涵盖从语言理解到多步骤任务、环境交互等多维度。

**📈 对比分析**

通过 Pass@k 与 Pass^k 指标、可靠性指标、Pareto 前沿分析等方法对模型进行对比，结果显示传统单次高分评估往往掩盖可靠性不足，Frontier 模型在 Pass^k 上仍显显著不足。

**⚠️ 局限性**

局限性包括缺乏统一评估标准、易受数据污染与模型偏见影响、对工具链与环境配置高度敏感、资源消耗大且难以自动化可靠性检查，并且多语言和多领域评估体系尚不完善。

---

## 141. Exploring The Impact Of Proactive Generative AI Agent Roles In Time-Sensitive Collaborative Problem-Solving Tasks

**arXiv ID:** 2602.17864 | [PDF](https://arxiv.org/pdf/2602.17864v1)

**作者:** Anirban Mukhopadhyay `[一作]` (Virginia Tech), Kumar Akash `[通讯]` (Honda Research Institute)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过在数字逃脱室中嵌入两类生成式 AI 代理（同行式 “Ava” 与协调者式 “Fiona”），探究它们在协同、时间受限的问题解决任务中的作用。

**💡 创新点**

创新点包括：①设计并实现了两种功能性 AI 探针，体现“同行”与“协调者”两种角色的对比；②采用实证实验评估 AI 角色对团队表现、工作量、协调与认知负荷的影响；③基于研究结果提出了针对协作式 AI 的角色与交互设计准则。

**🔧 技术方法**

技术上使用了大语言模型（GPT‑4.1‑mini、o3、gpt‑4o‑mini）进行摘要、提示与实时对话；配合 WhisperX 进行语音转写、NASA‑TLX、Perceived Coordination、AI Perception 等量表收集定量数据；同时采用主题分析（Braun & Clarke）处理访谈和观察笔记。

**📊 数据集**

数据集为 24 名参与者（6 组 × 4 人）在 3 个逃脱室谜题上的 18 次实验数据，包含实时对话转写、屏幕交互日志、问卷评分与焦点访谈转录。

**📈 对比分析**

对比方法为 6‑序列 Latin‑square 设计的“within‑subjects”实验，使用 ART‑ANOVA 分析 AI 角色与谜题难度对团队得分、协调感、工作量与 AI 认知的影响。结果显示：①协调者角色在客观得分上最高（平均 14.5/15），同行角色最低（平均 4.7/15）；②同行角色显著提高工作量（NASA‑TLX 平均 17.1 点），协调者与无 AI 条件相近；③参与者对同行的主观效用评价较高，但对协调者的效果认知不足。

**⚠️ 局限性**

局限性包括：①样本量小且仅为内部实验室参与者，结果可能不易推广到真实工作场景；②仅探讨两种极端角色，未覆盖更复杂或混合角色；③实验任务为单一逃脱室情境，缺乏多样化的协作任务；④大模型的推理与回应质量受限，可能影响代理行为的真实性。

---

## 142. MeanVoiceFlow: One-step Nonparallel Voice Conversion with Mean Flows

**arXiv ID:** 2602.18104 | [PDF](https://arxiv.org/pdf/2602.18104v1)

**作者:** Takuhiro Kaneko `[一作]` (NTT, Inc.), Yuto Kondo `[通讯]` (NTT, Inc.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 MeanVoiceFlow，一种一次性非并行语音转换模型，能够从零开始训练并实现快速高质量转换。

**💡 创新点**

核心创新包括使用平均流取代瞬时速度实现一次采样、引入零输入的结构化边际重构损失以稳定训练，以及条件扩散输入训练消除训练‑推理不匹配。

**🔧 技术方法**

采用流匹配与平均流的向量场学习框架，结合 U‑Net 结构的平均速度网络、SSIM 边际重构损失、语音嵌入与内容嵌入，以及条件扩散输入训练。

**📊 数据集**

实验使用 VCTK（110 位英语说话人）和 LibriTTS（1151 位英语说话人）数据集进行零样本任意到任意 VC 评估。

**📈 对比分析**

通过与多步扩散/流匹配、FastVoiceGrad 及其改进版的对比，在 MOS、DNSMOS、语音相似度、误识别率等客观指标和主观测试中，MeanVoiceFlow 与多步模型相当，且优于单步训练模型，且无需预训练或蒸馏。

**⚠️ 局限性**

局限性在于仍需大型数据集支持，极端 t' 混合比例下的鲁棒性待验证，结构化约束与条件输入训练的通用性可能在不同任务中表现不稳定，且对非并行数据缺失的适应性尚需进一步探索。

---

## 143. Explaining AutoClustering: Uncovering Meta-Feature Contribution in AutoML for Clustering

**arXiv ID:** 2602.18348 | [PDF](https://arxiv.org/pdf/2602.18348v1)

**作者:** Matheus Camilo da Silva `[一作]` (University of Trieste), Sylvio Barbon Junior `[通讯]` (University of Trieste)

**通讯引用:** 3203 | [OpenAlex ID](https://openalex.org/A5004953377)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究自动聚类（AutoClustering）中元学习模型的可解释性，构建了22个框架的统一元特征分类法，并对模型进行全球与局部解释。

**💡 创新点**

创新点在于首次将Decision Predicate Graphs（DPG）与SHAP相结合，系统揭示元特征对聚类推荐的整体与个案影响，并利用解释结果驱动特征剪枝实现高效部署。

**🔧 技术方法**

主要技术包括元特征提取、元学习模型（分类/回归/排序）、全局解释技术DPG、局部解释技术SHAP，以及解释驱动的特征消融实验。

**📊 数据集**

使用了22个AutoClustering框架的公开数据集（癌症基因表达、UCI、OpenML、合成数据等），以及PoAC提供的6000个合成数据进行实验。

**📈 对比分析**

与原始完整特征集相比，基于DPG的特征剪枝在提取时间上提升至数十倍，预测误差仅上升5–12%，显示出在保持近似性能的同时显著提高计算效率。

**⚠️ 局限性**

局限性包括：实验仅覆盖可复现的基于传统元学习的框架，未覆盖嵌入式或神经网络元学习方法；消融研究基于合成数据，需在真实业务场景进一步验证；DPG在神经网络元学习上的可扩展性仍待探索。

---

## 144. MUOT_3M: A 3 Million Frame Multimodal Underwater Benchmark and the MUTrack Tracking Method

**arXiv ID:** 2602.18006 | [PDF](https://arxiv.org/pdf/2602.18006v1)

**作者:** Ahsan Baidar Bakht `[一作]` (Khalifa University), Arif Mahmood `[通讯]` (Information Technology University)

**通讯引用:** 4211 | [OpenAlex ID](https://openalex.org/A5061017734)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了大型多模态水下目标跟踪基准 MUOT-3M，并基于 SAM2 开发了多模态教师-单模态学生框架 MUTrack，实现在仅 RGB 输入下实现高鲁棒性的实时跟踪。

**💡 创新点**

创新点包括：① 3 百万帧、3,030 条视频、677 类细粒度标签、32 个跟踪属性的多模态数据集；② 视觉–几何、视觉–语言的跨模态对齐与融合；③ 通过四层知识蒸馏将多模态教师知识迁移至单模态学生，实现训练时使用增强 RGB、深度与语言，推理时仅用原始 RGB；④ 在现有 UOT 基准上显著提升性能。

**🔧 技术方法**

采用的技术有：SAM2 视频分割框架、CLIP 视觉与文本编码器、增强 RGB 与深度对齐的对比与 L1 损失、视觉‑语言适配器、四种蒸馏损失（视觉‑几何、时空注意力、VL 适配器、掩码 logits）以及多模态与单模态训练流程。

**📊 数据集**

使用的数据集包括：① 自研的 MUOT-3M（3M 帧、3,030 视频、RGB、增强 RGB、深度、语言、分割、属性标签）；② 对比评测使用的 WebUOT‑1M、UOT32、UOT100、UTB180、UVOT400、UW‑COT220 等现有 UOT 基准。

**📈 对比分析**

方法在 MUOT-3M 以及跨数据集评测中与 20+ 先进跟踪器比较，MUTrack 单模态学生在 success、precision、norm. precision 上均超过 4–7%（例如 success 66.58% 对比 DUTrack 62.66%），并在 WebUOT‑1M、UTB180、UVOT400 等数据集上表现更优，验证了多模态预训练与蒸馏的有效性。

**⚠️ 局限性**

局限性包括：① 依赖于估计的增强 RGB 与深度，若这些模态质量不高会影响教师性能；② 推理仅用 RGB，仍需在低算力设备上进一步优化速度；③ 数据集虽然大而多样，但主要聚焦海域与物种，仍缺乏极端低能见度、极寒水域等极端环境的覆盖；④ 蒸馏过程复杂，训练成本高。

---

## 145. Flow Actor-Critic for Offline Reinforcement Learning

**arXiv ID:** 2602.18015 | [PDF](https://arxiv.org/pdf/2602.18015v1)

**作者:** Jongseong Chae `[一作]` (Korea Advanced Institute of Science and Technology), Youngchul Sung `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2540 | [OpenAlex ID](https://openalex.org/A5020240958)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于流模型的离线强化学习方法Flow Actor-Critic (FAC)，通过同时使用流行为代理进行演员正则化和评论家惩罚，实现对离线数据支持区域的有效约束。

**💡 创新点**

创新点在于将流模型既作为可表达的演员，也作为精确的行为密度估计，用该密度直接识别OOD区域并对Q值进行渐进惩罚，形成新的Bellman算子。

**🔧 技术方法**

采用连续正则流（CNF）与流匹配训练流行为代理，一步流演员以及双Critic网络，并在演员上加入流行为距离正则。

**📊 数据集**

在D4RL（MuJoCo、Antmaze、Adroit）和OGBench多任务离线数据集上进行评估。

**📈 对比分析**

与Gaussian、diffusion及其他流基线相比，FAC在55个OGBench单任务以及23个D4RL任务上均实现了领先或相当的性能，尤其在复杂多模态数据集上明显优于先前方法。

**⚠️ 局限性**

局限性包括对流模型积分步数和行为密度阈值的敏感性，在稀疏、高维Adroit任务中的在线微调收益有限。

---

## 146. A Self-Supervised Approach on Motion Calibration for Enhancing Physical Plausibility in Text-to-Motion

**arXiv ID:** 2602.18199 | [PDF](https://arxiv.org/pdf/2602.18199v1)

**作者:** Gahyeon Shim `[一作]` (Ulsan National Institute of Science and Technology), Hyemin Ahn `[通讯]` (Pohang University of Science and Technology)

**通讯引用:** 250 | [OpenAlex ID](https://openalex.org/A5053725843)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了Distortion-aware Motion Calibrator (DMC)，一种可插拔的后处理模块，用于在文本到动作生成模型生成的动作中修正物理不合理性（如脚漂浮、穿墙等）并保持与文本语义的一致性。

**💡 创新点**

创新点在于：①通过自监督式合成失真（垂直偏移+时间平滑）训练DMC，无需昂贵的物理仿真；②提供两种轻量级变体——WGAN基模型侧重语义与感知质量，Denoising基模型通过多步迭代细粒度纠正物理错误；③将文本嵌入与动作编码联合使用，提升物理一致性。

**🔧 技术方法**

技术方法包括：自监督失真数据生成、Transformer编码器作为生成器、WGAN‑GP对抗训练、基于DDPM的迭代去噪训练、文本与动作特征融合、评估指标（FID、R‑Precision、Foot Skating、Foot Floating/penetration、Foot Clipping）。

**📊 数据集**

使用HumanML3D数据集（约14.6k 3D动作+44.9k文本描述），对T2M、T2M‑GPT和MoMask三种基线模型进行测试。

**📈 对比分析**

与基线模型对比，WGAN‑DMC在T2M上FID下降42.74%、在T2M‑GPT上R‑Precision提升；Denoising‑DMC在T2M‑GPT、MoMask上地面穿透分别降低42.57%和33.0%，整体物理可行性显著提升；两种变体在速度和精度上互补，WGAN快而语义好，Denoising慢但纠错更细。

**⚠️ 局限性**

局限性包括：仅处理垂直偏移与时间平滑两类失真，未覆盖抖动、交叉等更复杂物理错误；缺乏针对具体机器人硬件的质量/扭矩约束；推理速度在高精度模式下仍较慢。

---

## 147. Interpreting Multi-Branch Anti-Spoofing Architectures: Correlating Internal Strategy with Empirical Performance

**arXiv ID:** 2602.17711 | [PDF](https://arxiv.org/pdf/2602.17711v1)

**作者:** Ivan Viakhirev `[一作]` (ITMO University), Grach Mkrtchian `[通讯]` (Moscow Technical University of Communication and Informatics)

**通讯引用:** 34 | [OpenAlex ID](https://openalex.org/A5041288941)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种基于谱分析与 SHAP 的解释框架，用于解析多分支语音反欺骗网络 AASIST3 的内部决策过程。

**💡 创新点**

创新点在于将层激活的协方差矩阵特征（前10个特征值）与 CatBoost 元分类器相结合，再利用 TreeSHAP 对分支贡献进行归因，构成四种操作模式（有效专化、有效共识、无效共识、错误专化），揭示模型在不同攻击下的动态策略和潜在单点失效。

**🔧 技术方法**

所用技术包括谱特征提取（特征值分解）、梯度提升树 CatBoost、TreeSHAP 归因、置信度评分、Kendall 相关性检验以及单分支保留消融实验。

**📊 数据集**

实验数据来自 ASVspoof 2019 逻辑访问（LA）数据集，使用公开的 AASIST3 预训练模型进行评估。

**📈 对比分析**

与基准方法（如统一惩罚、无惩罚、二次/指数惩罚）对比，所提框架在识别主导分支和评估模型可信度方面表现出高度一致性，EER 在易攻击（如 A09、A14）可低至 0.05%，但在难攻击（如 A17、A18）可升至 28%以上，显示出模型在错误专化模式下的高错误率。

**⚠️ 局限性**

局限性包括：归因仅揭示相关性而非因果关系；单分支消融表明主导分支并不单独可完成检测，暗示缺乏跨分支正则化；解释框架需访问中间激活，适用范围受限；且未针对对抗性攻击进行进一步鲁棒性验证。

---

## 148. HookLens: Visual Analytics for Understanding React Hooks Structures

**arXiv ID:** 2602.17891 | [PDF](https://arxiv.org/pdf/2602.17891v1)

**作者:** Suyeon Hwang `[一作]` (Seoul National University), Jinwook Seo `[通讯]` (Seoul National University)

**通讯引用:** 4389 | [OpenAlex ID](https://openalex.org/A5012388103)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了HookLens，一套面向React应用的交互式可视化分析系统，能够可视化组件层级、Hook（state、props、effect）的关系，并自动检测三类常见反模式（未引用的state/props、prop drilling、子组件修改父组件state）。

**💡 创新点**

创新点在于：①将组件层级与Hook之间的动态交互以节点-边图形式直观呈现；②结合规则驱动的静态分析与可视化，实时高亮反模式；③提供交互式聚焦、代码查看与细粒度展开，帮助开发者在一次可视化中完成从结构理解到反模式定位的完整流程；④通过用户研究与LLM助手对比，验证可视化分析在代码维护中的不可替代性。

**🔧 技术方法**

主要技术包括：使用Espree和TypeScript Compiler API将源码转为AST；基于预定义规则抽取组件、Hook、依赖关系；构建图结构数据并在前端采用节点-边图（D3/React-Vis）渲染；通过交互事件实现节点展开、聚焦与代码高亮；对比实验采用VS Code、Claude Code、Codex CLI（GPT‑5）、Gemini CLI等LLM助手。

**📊 数据集**

使用了两个公开React项目（Confides、paper_vis），每个约25–33个组件，约3,000–4,000行代码，包含41个未引用state/props、11个prop drilling、2个子组件修改父state的反模式。

**📈 对比分析**

通过12名开发者的定量用户研究，比较HookLens与VS Code的精准率、召回率与F1，结果显示HookLens在所有三类反模式上显著提升F1（p≪0.01），尤其是对初学者更有优势。随后将HookLens与三种LLM助手在相同任务下进行对比，HookLens在准确率与召回率上均优于Claude Code、Gemini CLI，甚至优于GPT‑5，证明可视化分析在跨文件、跨层级反模式识别中的优势。

**⚠️ 局限性**

局限性包括：①仅在组件数≤33的小型项目上验证，缺乏大规模项目的可扩展性评估；②可视化在高交叉度图形中仍会产生视觉拥挤；③仅支持核心Hook，未涵盖外部状态管理（Redux等）与自定义Hook；④缺乏运行时行为分析与语义验证；⑤未实现IDE插件集成，需手动切换工具。

---

## 149. Mining Type Constructs Using Patterns in AI-Generated Code

**arXiv ID:** 2602.17955 | [PDF](https://arxiv.org/pdf/2602.17955v1)

**作者:** Imgyeong Lee `[一作]` (University of Alberta), Abram Hindle `[通讯]` (University of Alberta)

**通讯引用:** 6126 | [OpenAlex ID](https://openalex.org/A5070785370)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 AI 编码代理在 TypeScript 项目中使用类型系统的行为进行实证分析，比较其与人工开发者的差异。

**💡 创新点**

首次系统评估 AI 生成代码的类型相关构造，揭示 AI 过度使用 `any` 与类型逃逸的风险，并发现其接受率异常高。

**🔧 技术方法**

采用双阶段过滤框架：先用正则表达式粗筛 TypeScript PR，再使用多代理 LLM 进行精确分类和验证。

**📊 数据集**

使用 AIDev 数据集，从中提取 545 条 AI PR 与 269 条人类 PR 进行实验。

**📈 对比分析**

通过统计 AI 与人类 PR 的高级类型特征使用次数、`any` 添加/删除量、接受率等指标进行对比，结果显示 AI PR 的 `any` 使用率约为人类的 9 倍，接受率约为人类的 1.8 倍。

**⚠️ 局限性**

过滤方法依赖预定义正则和 LLM 语义，可能漏检复杂或新颖的类型结构；此外，AI 与人类 PR 任务类型差异可能影响接受率比较。

---

## 150. VidEoMT: Your ViT is Secretly Also a Video Segmentation Model

**arXiv ID:** 2602.17807 | [PDF](https://arxiv.org/pdf/2602.17807v1)

**作者:** Narges Norouzi `[一作]` (Eindhoven University of Technology), Daan de Geus `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 191 | [OpenAlex ID](https://openalex.org/A5061463972)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 VidEoMT，一种完全基于 Vision Transformer 的视频分割模型，统一了分割与时序关联，完全不需要专用跟踪模块；

**💡 创新点**

通过轻量级的查询传播（query propagation）与查询融合（query fusion）机制，让大型预训练 ViT 在单一编码器内完成分割与跟踪；消除繁重的跟踪组件，显著提升推理速度；

**🔧 技术方法**

使用大型预训练 ViT（如 DINOv2），在最后 L₂ 层注入可学习查询；实现查询传播并与可学习查询进行逐元素相加；训练时采用 Mask2Former 损失（交叉熵、BCE、Dice）与 FlashAttention、层级学习率衰减等技术；

**📊 数据集**

在 OVIS、YouTube‑VIS 2019/2021/2022（视频实例分割）、VIPSeg（视频全景分割）和 VSPW（视频语义分割）等主流基准数据集上进行评估；

**📈 对比分析**

与 CAVIS、DVIS、DVIS++、DVIS‑DAQ、MinVIS 等最先进方法对比；在 VIS 上 AP 与最强模型相当或更高，FPS 提升 5–10 倍（最高 160 FPS 对比 15 FPS）；在 VPS、VSS 任务同样保持较小性能损失，同时速度提升 5–19 倍；

**⚠️ 局限性**

依赖于大规模预训练的 ViT，模型对预训练规模敏感；在模型规模较小或预训练不足时性能下降；对极快出现/消失的目标仍有挑战；整体实验覆盖的场景仍受限于公开基准数据。

---

## 151. FedZMG: Efficient Client-Side Optimization in Federated Learning

**arXiv ID:** 2602.18384 | [PDF](https://arxiv.org/pdf/2602.18384v1)

**作者:** Fotios Zantalis `[一作]` (University of West Attica), Grigorios Koulouras `[通讯]` (University of West Attica)

**通讯引用:** 1717 | [OpenAlex ID](https://openalex.org/A5021922035)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在联邦学习框架下提出 FedZMG，一种仅在客户端执行、无额外通信和超参数的优化算法，通过将局部梯度投影到零均值超平面来抑制客户端漂移并加速收敛。

**💡 创新点**

创新点在于利用零均值梯度（ZMG）投影作为结构正则化手段，直接在梯度空间减少异构性导致的方差，既无需状态化的自适应优化器也不增加通信负担；同时提供理论收敛证明。

**🔧 技术方法**

技术方法包括梯度零均值投影（Gradient Centralization）、离散权重衰减、FedAvg 级联聚合、Lipschitz/强凸理论分析及对比实验；实现时在客户端对每个梯度矩阵按输入维度/空间通道做列/通道均值归零。

**📊 数据集**

实验数据集：EMNIST（手写字符分类）、CIFAR-100（图像分类）以及 Shakespeare（下一字符预测 RNN），三种任务覆盖视觉与文本，均在显著非 IID 分布下测试。

**📈 对比分析**

方法通过与 FedAvg、FedAdam 进行 1000 轮通信的实验对比，使用最终准确率、阈值达到轮次和后期平均准确率等指标。FedZMG 在所有数据集上均收敛更快、最终验证准确率更高，尤其在高异构 CIFAR‑100 上显著优于两者；统计检验显示对比 FedAvg 的差异显著，FedZMG 对 FedAdam 的提升在部分任务中亦有统计意义。

**⚠️ 局限性**

局限性：若梯度均值本身携带重要学习信号（如某些回归任务或特定网络结构），投影可能削弱学习；与批归一化等激活归一化交互可能产生重叠，导致深层网络收益递减；实验仅对比 FedAvg、FedAdam，未覆盖如 SCAFFOLD、FedCAda 等其他客户端优化器；未来工作需进一步评估与其他算法的组合与更深层网络中的表现。

---

## 152. Advection-Diffusion on Graphs: A Bakry-Emery Laplacian for Spectral Graph Neural Networks

**arXiv ID:** 2602.18141 | [PDF](https://arxiv.org/pdf/2602.18141v1)

**作者:** Pierre-Gabriel Berlureau `[一作]` (École Normale Supérieure), Pierre Vandergheynst `[通讯]` (École polytechnique fédérale de Lausanne)

**通讯引用:** 29878 | [OpenAlex ID](https://openalex.org/A5028520858)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了基于Bakry–Émery理论的可学习节点势能的图拉普拉斯算子，并将其嵌入谱图神经网络中，形成µ-ChebNet与µ-Stable‑ChebNet；

**💡 创新点**

创新点在于：①通过节点势能直接在拉普拉斯算子中引入推进-扩散动力学，实现对信息流的任务依赖调控；②无需改造图结构或加入注意力，保持谱高效性；③理论分析显示势能可独立控制谱间隙与谱半径，从而缓解过平滑与过压缩；

**🔧 技术方法**

使用了：Bakry–Émery拉普拉斯、Chebyshev 多项式滤波、GCN 用于学习势能、谱滤波器参数化、稳定化正则化；

**📊 数据集**

实验数据集包括：Barbell 合成图、Graph Property Prediction（图属性预测）、OGBN‑Proteins（蛋白质相互作用多标签数据集）；

**📈 对比分析**

与传统谱方法（ChebNet、Stable‑ChebNet）、MPNN（GCN、GAT、GraphSAGE、GIN、GCNII）、图变压器等做对比；µ-ChebNet 在 Barbell 任务中几乎无误差，Graph Property Prediction 中提升 1–3% 以上，OGBN‑Proteins 与强基线相当或略优；

**⚠️ 局限性**

局限性在于：势能学习仍需额外参数与训练开销；对动态或大规模稀疏图的适应性未充分验证；对势能可解释性与公平性等社会影响的系统评估缺失。

---

## 153. Curriculum Learning for Efficient Chain-of-Thought Distillation via Structure-Aware Masking and GRPO

**arXiv ID:** 2602.17686 | [PDF](https://arxiv.org/pdf/2602.17686v1)

**作者:** Bowen Yu `[一作]` (City University of Hong Kong), Xiangyu Zhao `[通讯]` (City University of Hong Kong)

**通讯引用:** 6087 | [OpenAlex ID](https://openalex.org/A5100645854)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过三阶段课程学习框架 BRIDGE，逐步让 3B 规模学生模型学习并压缩大型模型的链式推理。

**💡 创新点**

创新点在于先通过结构重构培养逻辑理解，再用 GRPO 优化准确性与简洁性的层级奖励，最后通过教师引导重写实现内部化压缩。

**🔧 技术方法**

使用了结构重构训练、Group Relative Policy Optimization (GRPO) 与层级奖励机制、教师引导的重写。

**📊 数据集**

主要在 GSM8K、SVAMP、MATH-500 三个数学推理数据集上进行实验。

**📈 对比分析**

相较于基线，Qwen2.5-3B 在 GSM8K 上达 76.19% 准确率，平均 167 token，较基线提升 11.29% 并压缩 27.4%，在 SVAMP 与 MATH-500 上亦表现更优。

**⚠️ 局限性**

局限在于对极难样本的处理仍受限，且模型对格式一致性仍易产生错误，未能完全避免奖励作弊。

---

## 154. SPQ: An Ensemble Technique for Large Language Model Compression

**arXiv ID:** 2602.18420 | [PDF](https://arxiv.org/pdf/2602.18420v1)

**作者:** Jiamin Yao `[一作]` (Southern Illinois University Edwardsville), Eren Gultepe `[通讯]` (Southern Illinois University Edwardsville)

**通讯引用:** 505 | [OpenAlex ID](https://openalex.org/A5078722503)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一种名为SPQ的压缩框架，利用SVD、结构化剪枝和8‑bit线性量化三种技术对大型语言模型进行压缩，并通过LoRA微调恢复性能。

**💡 创新点**

创新点包括：①将SVD仅应用于注意力层、剪枝仅应用于MLP层，并通过方差阈值和激活统计自适应分配层级压缩比例；②将三种压缩方法在层级上实现最优组合，显著提升压缩率同时保持或提升模型质量；③采用后训练8‑bit对称量化并支持混合量化策略，无需校准数据。

**🔧 技术方法**

主要技术：SVD低秩分解（方差保留）、激活基结构化剪枝（按激活均值/方差确定剪枝比例）、8‑bit对称线性量化（支持全张量、通道级和混合模式）、LoRA微调（冻结原权重，仅训练低秩增量矩阵）。

**📊 数据集**

使用的数据集：语言建模评估采用WikiText‑2和C4；推理与推断评估使用OpenBookQA、ARC、WinoGrande、HellaSwag、PIQA、TruthfulQA（两个版本）和GSM8K；此外使用小规模校准集计算激活统计。

**📈 对比分析**

与ASVD、SparseGPT、GPTQ等单一或双重压缩方法对比；在LLaMA‑2‑7B、OPT、Vicuna、Mistral等模型上进行实验。SPQ在75%压缩比下内存为6.86 GB，困惑度与原始模型相当或略好；吞吐量比GPTQ 8‑bit提升约1.3×，比GPTQ 4‑bit提升约1.9×；压缩时间也略快（≈20%）。

**⚠️ 局限性**

局限性：①仅针对特定组合的SVD–剪枝–量化，可能不是全局最优；②依赖LoRA微调，虽然步骤短，但仍需额外训练；③对小规模模型的效果有限；④未探索混合精度量化、激活量化或知识蒸馏等替代或补充技术；⑤对不同硬件平台的泛化性尚待进一步验证。

---

## 155. Methods for Pitch Analysis in Contemporary Popular Music: Multiphonic Tones Across Genres

**arXiv ID:** 2602.18030 | [PDF](https://arxiv.org/pdf/2602.18030v1)

**作者:** Emmanuel Deruty `[一作]` (Sony Computer Science Laboratories), Pascal Arbez-Nicolas `[通讯]` (Citizen Records)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对流行音乐中的电子音色（如808低音、功率和弦等）与古典音乐中的多音阶进行结构与感知上的比较，利用听觉匹配实验和信号分析揭示两者在多音感知上的相似性。

**💡 创新点**

将多音阶研究从传统古典音乐扩展至当代流行音乐，证明电子音色在结构和听觉歧义上与多音阶等价，提出“多音感知”是普遍现象而非罕见技术。

**🔧 技术方法**

使用功率谱分析、等响度加权、时间与频谱模型、单音高跟踪器CREPE与PESTO、10名受试者的听觉匹配实验。

**📊 数据集**

19个单音样本（包括古典乐器多音阶、808低音、功率和弦、FM低音+失真/滤波等），来源于Sony、Aalborg、Neodrome、独立音乐人及Citizen Records；完整音频可在公开链接获取。

**📈 对比分析**

将人类听觉匹配结果与CREPE/PESTO输出进行对比，统计平均感知音数（加权置信度）和音阶跳跃；实验显示两类音色产生相同的多音感知，算法的音阶跳跃与人类感知高度一致，说明算法能捕捉到信号的多音特征。

**⚠️ 局限性**

局限性包括样本量有限、受试者个体差异大、测试环境不受控、仅考虑单音样本而未研究音乐上下文、使用单音高跟踪器而非多音高跟踪器，缺乏跨文化与跨流派的进一步验证。

---

## 156. Market Games for Generative Models: Equilibria, Welfare, and Strategic Entry

**arXiv ID:** 2602.17787 | [PDF](https://arxiv.org/pdf/2602.17787v1)

**作者:** Xiukun Wei `[一作]` (Ohio State University), Xueru Zhang `[通讯]` (Ohio State University)

**通讯引用:** 4163 | [OpenAlex ID](https://openalex.org/A5101877243)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建并分析了三层模型‑平台‑用户的生成式AI市场博弈模型，研究了纯纳什均衡的存在条件、市场结构与用户福利的关系，并从模型供应商角度设计了进入竞争市场的最佳响应训练方案。

**💡 创新点**

首次将平均模型性能与用户类型的差异优势结合起来阐明市场结构；证明扩充模型池不必然提升福利或多样性；提出基于最佳响应的模型改进方法，直接优化竞争吸引力。

**🔧 技术方法**

博弈理论与均衡分析、硬/软最大化用户选择模型、Bradley‑Terry 软门控、路径梯度与 REINFORCE 梯度估计、LoRA 微调、DDPM 生成模型。

**📊 数据集**

使用 CIFAR‑10 数据集生成多种 LoRA 微调模型作为模型池，并通过 ResNet‑20 评估用户奖励。

**📈 对比分析**

通过离散最佳响应仿真与真实数据实验，比较模型池扩充与平台增多对多样性(HHI)、覆盖值 V 与福利 W 的影响；发现最佳响应训练可取代原始最优模型，提升福利与多样性。

**⚠️ 局限性**

实验受限于 CIFAR‑10 规模、奖励函数与用户偏好设定，理论假设用户选择为硬/软最大化，且未考虑平台合作或动态进入/退出等更复杂机制。

---

## 157. Automated LLM-Based Accessibility Remediation: From Conventional Websites to Angular Single-Page Applications

**arXiv ID:** 2602.17887 | [PDF](https://arxiv.org/pdf/2602.17887v1)

**作者:** Carla Fernández-Navarro `[一作]` (ITIS Software, University of Malaga), Francisco Chicano `[通讯]` (ITIS Software, University of Malaga)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

开发了一套基于LLM的自动化无障碍修复系统，可同时处理静态网页和Angular单页面应用。

**💡 创新点**

通过模块化工作流结合多模态图像分析、上下文提示工程以及AST级别的代码注入，实现了对SPAs动态无障碍问题的自动修复。

**🔧 技术方法**

使用Selenium+Axe-core检测、OpenAI GPT‑4o进行修复和图像描述，Python+BeautifulSoup+AST实现代码修改。

**📊 数据集**

Dataset A（12个公共网站）和Dataset B（6个开源Angular项目）作为实验数据集。

**📈 对比分析**

通过Remediation Rate、Build Integrity、Semantic Functional Verification等指标评估；静态网页平均80%修复率，Angular平均86%，构建通过率100%，平均执行时间约15–17分钟。

**⚠️ 局限性**

对遗留结构、复杂第三方组件和高度动态状态的处理仍有限，部分场景需人工干预，且LLM可能产生幻觉导致修复不完整。

---

## 158. Closing Africa's Early Warning Gap: AI Weather Forecasting for Disaster Prevention

**arXiv ID:** 2602.17726 | [PDF](https://arxiv.org/pdf/2602.17726v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 159. Learning to Tune Pure Pursuit in Autonomous Racing: Joint Lookahead and Steering-Gain Control with PPO

**arXiv ID:** 2602.18386 | [PDF](https://arxiv.org/pdf/2602.18386v1)

**作者:** Mohamed Elgouhary `[一作]` (West Virginia University), Amr S. El-Wakeel `[通讯]` (West Virginia University)

**通讯引用:** 371 | [OpenAlex ID](https://openalex.org/A5021410318)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在纯追踪 (Pure Pursuit) 控制器中加入了强化学习策略，用于在线自适应调整查找距离 L_d 和转向增益 g，以提升路径跟踪性能。

**💡 创新点**

使用 PPO 训练的策略在一次动作中同时输出 L_d 与 g，保持纯追踪的几何可解释性，同时实现了零样本跨地图迁移和硬件部署。

**🔧 技术方法**

Proximal Policy Optimization (PPO) 强化学习、F1TENTH Gym 仿真、ROS 2 轻量接口、LiDAR MCL 定位、k‑inematic MPC 对比、奖励设计与调参、Optuna 超参数搜索。

**📊 数据集**

使用 TUM 最小曲率赛道优化得到的 raceline（包含位置、曲率、最大速度），在 F1TENTH Hockenheim、Montreal、Yas Marina 三条地图上进行训练与评估。

**📈 对比分析**

与固定查找、线性速度调度、仅调 L_d、以及基于 MPC 的跟踪进行对比；在仿真与真实车辆中，RL–PP（L_d,g）在圈速、轨迹误差和转向平滑度上均显著优于所有基线，甚至超越 MPC。

**⚠️ 局限性**

对高动态或极端弯道的鲁棒性有限，受限于全局参考轨迹与定位精度；在真实车辆上速度被人为限制，且未考虑轮胎滑移模型。

---

## 160. RAT+: Train Dense, Infer Sparse -- Recurrence Augmented Attention for Dilated Inference

**arXiv ID:** 2602.18196 | [PDF](https://arxiv.org/pdf/2602.18196v1)

**作者:** Xiuying Wei `[一作]` (CLAIRE lab at EPFL), Caglar Gulcehre `[通讯]` (CLAIRE lab at EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在密集预训练阶段训练一个全密集模型（RAT+），其内部加入全序列递归和主动递归学习，从而在推理时可自由切换至稀疏注意力模式（如扩张注意力、局部窗口、混合层/头以及top‑k块注意力），并通过仅1B标记的微调即可适配不同稀疏配置。

**💡 创新点**

创新点：①提出在密集预训练中嵌入全序列递归与主动递归学习，使模型在保持全密集性能的同时可在推理时实现扩张稀疏；②使用“重叠块”策略保证递归输出在不同稀疏尺度下分布一致；③通过稀疏与密集双任务联合训练（ARL）强制模型学习所需递归长度。

**🔧 技术方法**

技术：全序列递归（RAT）、主动递归学习（ARL）、扩张注意力、局部窗口注意力、top‑k块注意力、混合层/头稀疏、线性投影、FlashAttention/FlexAttention实现。

**📊 数据集**

数据集：FineWeb‑Edu（预训练100B token，1.5B模型）、FineWeb（200B token，2.6B模型）、PG19（200M模型）、LongBench、commonsense reasoning任务（ARC‑C/E、Hella)、RULER NIAH等。

**📈 对比分析**

比较方法：与标准密集注意力、原始RAT、Mamba2、GatedDeltaNet等在相同FLOP预算下对比；在扩张率16、64时，RAT+在准确率/Perplexity上仅比密集低0~3点；top‑k块模式下，RAT+比密集注意力提升10–20点；通过预训练+1B微调即可实现稀疏配置，达到与专门训练稀疏模型相当甚至更优的性能；效率评估显示在长序列(prefill/decoding)上可获得4–10× FLOP/KV 缓存压缩，整体吞吐率提升60–80×。

**⚠️ 局限性**

局限：①仍需1B微调适配，无法一次性支持所有稀疏配置；②GPU级别的CUDA优化尚未实现，实际速度取决于实现细节；③混合层/头稀疏的最优配置需要进一步搜索；④对极端长序列（>16K）或不同任务的泛化仍需验证；⑤递归长度选择（如64）是经验值，可能对更大模型需要调整。

---

## 161. CodeScaler: Scaling Code LLM Training and Test-Time Inference via Execution-Free Reward Models

**arXiv ID:** 2602.17684 | [PDF](https://arxiv.org/pdf/2602.17684v1)

**作者:** Xiao Zhu `[一作]`, Zhijiang Guo `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种专为代码生成设计的奖励模型（简称CodeReward），并将其同时应用于强化学习训练和推理阶段的最佳‑N（BoN）采样。

**💡 创新点**

创新点包括：①利用高质量已验证的DeepCoder数据构建基于对话轨迹的偏好对；②引入语法感知的代码提取与有效性保持的奖励塑造，显著提升奖励稳定性并抑制奖励劫持；③通过奖励模型实现了无执行的强化学习和低延迟的测试时缩放。

**🔧 技术方法**

主要技术包括：对数几率（Bradley‑Terry）损失训练奖励模型；GRPO强化学习算法；语法检查与AST验证；对数映射奖励塑造；以及在推理阶段的BoN采样。

**📊 数据集**

使用的主要数据集有：DeepCoder（24K已验证题目）、KodCode、rStarCoder、LiveCodeBench、CodeContests、LiveBench、MBPP、CodeForces、RM‑Bench，以及Synthetic datasets如TACO、QueST。

**📈 对比分析**

与传统基于执行的RLVR、SkyworkRM、AceCodeRM等奖励模型相比：在Qwen3‑8B‑Base上平均提升约+11.72分，超越RLVR+1.82分；在测试时缩放上，与CURE相当但延迟下降10×；在RM‑Bench上在代码、通用与推理域均各提升3.3、2.7分。

**⚠️ 局限性**

局限性包括：仍需依赖高质量已验证数据来构建偏好对；在合成数据上性能仍略低于DeepCoder；奖励模型的可解释性与对极端错误代码的判别仍有提升空间；在极大规模模型上训练仍受算力限制。

---

## 162. Flexi-NeurA: A Configurable Neuromorphic Accelerator with Adaptive Bit-Precision Exploration for Edge SNNs

**arXiv ID:** 2602.18140 | [PDF](https://arxiv.org/pdf/2602.18140v1)

**作者:** Mohammad Farahani `[一作]` (University of Tehran), Saeed Safari `[通讯]` (University of Tehran)

**通讯引用:** 1215 | [OpenAlex ID](https://openalex.org/A5071789164)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了可配置的神经形态加速器 Flexi-NeurA 及其 DSE 工具 Flex-plorer，支持在硬件生成前自定义神经元模型、网络拓扑、位宽等，实现事件驱动、时分复用、可多核的 SNN 加速器。

**💡 创新点**

创新点在于将可配置参数提升到设计时刻，结合事件驱动与时分复用实现低面积低功耗，同时提供基于模拟退火的精度感知设计空间探索工具，自动寻找精度与资源的最优平衡。

**🔧 技术方法**

采用事件驱动时分复用架构、基于 shift‑and‑add 的 Coefficient Generator、FPGA 资源估计回归模型、Simulated Annealing、SNN‑Torch 训练及量化、Numba JIT 加速评估。

**📊 数据集**

使用 MNIST、Spiking Heidelberg Digits (SHD)、DVS Gesture 三个基准，分别进行分类任务。

**📈 对比分析**

与多款现有 FPGA 版 SNN/SCNN 加速器对比，Flexi‑NeurA 在 MNIST 上达到 97.23% 识别率，时延 1.1 ms，功耗仅 111 mW，占比 4% LUT，使用 1,623 逻辑单元；在其他数据集亦保持低能耗与可扩展性。

**⚠️ 局限性**

受限于单核心时分复用导致的计算瓶颈，深度网络仍需分层映射，且精度搜索空间对大规模网络仍计算量大；目前仅支持全连接或特定递归拓扑，未涵盖卷积或大规模稀疏网络。

---

## 163. Dual-Channel Attention Guidance for Training-Free Image Editing Control in Diffusion Transformers

**arXiv ID:** 2602.18022 | [PDF](https://arxiv.org/pdf/2602.18022v1)

**作者:** Guandong Li `[一作]` (iFLYTEK), Mengxia Ye `[通讯]` (Aegon THTF)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一种训练无关的双通道注意力引导（DCAG），实现 Diffusion Transformer 图像编辑的细粒度强度控制。

**💡 创新点**

创新点在于发现键和值投影都具有 bias‑delta 结构，并同时调节两者，提供二维（δ_k,δ_v）控制空间，使编辑‑保真权衡比单通道方法更细腻。

**🔧 技术方法**

技术手段包括在多模态注意力层对键和值投影做 bias‑delta 重新缩放，理论分析 softmax 非线性与线性加权的控制特性，并在 Qwen‑Image‑Edit 的 Diffusion Transformer 上实现。

**📊 数据集**

使用 PIE‑Bench 数据集（700 张图像，10 种编辑类别）进行实验评估。

**📈 对比分析**

与无引导和 GRAG（键通道）对比，采用 LPIPS、SSIM、PSNR、MSE 等指标；DCAG 在大多数类别上显著降低 LPIPS（最高 4.9%），总体提升约 1.8% LPIPS。

**⚠️ 局限性**

局限性在于值通道提升幅度有限，且在较高键通道强度下趋于饱和；实验仅评估保真指标，缺乏对编辑质量（如 CLIP‑Score）的系统分析。

---

## 164. Demonstrating Restraint

**arXiv ID:** 2602.18139 | [PDF](https://arxiv.org/pdf/2602.18139v1)

**作者:** L. C. R. Patell `[一作]` (Governance AI), O. E. Guest `[通讯]` (Institute for AI Policy and Strategy)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

讨论美国如何通过成本信号与预留保证等手段承诺不利用AI实现决定性战略优势，以避免预防战争

**💡 创新点**

首次系统化提出将国际关系中的成本信号与AI治理的预留保证相结合的“克制”框架，阐述多层次政策组合的可行路径

**🔧 技术方法**

运用国际关系成本信号理论、AI治理技术（模型规范、硬件约束、逃逸仓机制等）作为理论与技术支撑

**📊 数据集**

无实证数据集，主要基于文献综述与理论模型构建

**📈 对比分析**

未进行实验或数据对比，仅在理论层面讨论方案的可行性、成本与潜在风险

**⚠️ 局限性**

技术实现难度大、验证与监督机制缺乏成熟方案、政治可行性不确定、类型漂移与权力转移导致承诺可信度下降

---

## 165. Growing With the Condition: Co-Designing Pediatric Technologies that Adapt Across Developmental Stages

**arXiv ID:** 2602.17925 | [PDF](https://arxiv.org/pdf/2602.17925v1)

**作者:** Neda Barbazi `[一作]` (University of Minnesota), Carlye Anne Lauff `[通讯]` (University of Minnesota)

**通讯引用:** 598 | [OpenAlex ID](https://openalex.org/A5045277237)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在医疗支持营地对69名患有先天性心脏病的儿童（小学、初中、高中）开展四个跨年龄组的协同设计工作坊，系统分析不同发展阶段儿童在社交、沟通和临床焦虑方面的挑战与应对策略，并提出可随发展演化的健康技术设计启示。

**💡 创新点**

首次将跨年龄对照的协同设计方法应用于先天性心脏病患儿，揭示从小学到高中阶段认知、情感与行为策略的系统演变，为儿童健康技术提供发展阶段适应的实证依据。

**🔧 技术方法**

采用童本位参与式设计技术（虚构探究、智慧图、情绪反思、主线任务）结合质性主题分析（NVivo）对绘画与文本数据进行编码与归纳。

**📊 数据集**

收集了69名儿童在工作坊中产生的手工绘画、纸质工作表、贴纸与短文本等质性数据，并未使用公开数据集。

**📈 对比分析**

对小学、中学、高中三组数据进行分年龄主题编码、归纳中层代码，并对比三类（问题、应对、支持）在不同年龄阶段的表现；研究通过质性比较显示社交包容、沟通方式与临床焦虑的策略随发展显著变化，证明设计应具发展适应性；未给出数值性能指标，但差异显著。

**⚠️ 局限性**

样本仅来自单一营地，缺乏多场景验证；研究为横断面、未追踪个体随时间变化；虚构情境可能影响儿童表达；未对所提技术进行原型评估。

---

## 166. Understanding Unreliability of Steering Vectors in Language Models: Geometric Predictors and the Limits of Linear Approximations

**arXiv ID:** 2602.17881 | [PDF](https://arxiv.org/pdf/2602.17881v1)

**作者:** Joschka Braun `[一作]` `[通讯]`, Joschka Braun

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

无法获取论文内容

**💡 创新点**

无法获取论文内容

**🔧 技术方法**

无法获取论文内容

**📊 数据集**

无法获取论文内容

**📈 对比分析**

无法获取论文内容

**⚠️ 局限性**

无法获取论文内容

---

## 167. LERD: Latent Event-Relational Dynamics for Neurodegenerative Classification

**arXiv ID:** 2602.18195 | [PDF](https://arxiv.org/pdf/2602.18195v1)

**作者:** Hairong Chen `[一作]` (Beijing Jiaotong University), Hengguan Huang `[通讯]` (University of Copenhagen)

**通讯引用:** 101 | [OpenAlex ID](https://openalex.org/A5063658508)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出LERD模型，利用多通道EEG无监督推断潜在事件及其关系图，并用于阿尔茨海默病（AD）分类；

**💡 创新点**

将事件后验微分方程、均值演化对数正态过程与电生理启发的可微leaky‑integrate‑and‑fire先验相结合，首次实现可解释的事件驱动图并给出KL上界与图稳定性的理论证明；

**🔧 技术方法**

采用变分推断的贝叶斯神经动力学框架，结合可微分先验、STDP式事件关系映射、对数正态混合过程、Gumbel‑Softmax重参数化等技术；

**📊 数据集**

使用合成toy数据以及两套真实EEG数据集（AD cohort A和B），涵盖AD、FTD、MCI和健康对照；

**📈 对比分析**

与NODE、ODE‑RNN、STRODE、CNN、RNN、Transformer等基线在合成与真实数据上对比，LERD在频段重建IoU非零、AD分类准确率最高（约75%/89%）、F1显著提升且表现更稳定；

**⚠️ 局限性**

对电生理先验假设简化，未考虑头模型与空间反演，模型对跨站点/设备泛化尚未充分验证，且训练复杂度较高。

---

## 168. Rethinking Beam Management: Generalization Limits Under Hardware Heterogeneity

**arXiv ID:** 2602.18151 | [PDF](https://arxiv.org/pdf/2602.18151v1)

**作者:** Nikita Zeulin `[一作]` (Tampere University), Robert W. Heath `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文系统分析了基于机器学习的波束管理在硬件异质性下的泛化问题，并通过案例实验展示了不同硬件配置对性能的影响，提出了面向异质性的设计思路。

**💡 创新点**

创新点在于构建硬件异质性维度的分类体系，实证验证了异质性导致的泛化瓶颈，并在此基础上提出轻量学习、协同学习与自适应学习等对策。

**🔧 技术方法**

采用深度神经网络（ResNet）、SVM、强化学习等模型，结合特征工程和模型压缩、迁移学习等技术。

**📊 数据集**

使用基于SUMO的城市车辆仿真环境，携带15 GHz、24子载波、8×8 UPA基站和多尺寸UE阵列的场景数据。

**📈 对比分析**

与两级层级搜索（HS）和全量搜索（ES）对比；在相同训练/测试配置下DNN表现优于基线，但在硬件/代码本/环境异质性条件下性能下降，平均SE降幅超过50%，90%分位值仍高。

**⚠️ 局限性**

泛化受限于硬件异质性、训练数据覆盖不足、模型复杂度与计算资源不匹配，以及缺乏真实世界验证和标准化评测基准。

---

## 169. Construction of Cyclic Codes over a Class of Matrix Rings

**arXiv ID:** 2602.18255 | [PDF](https://arxiv.org/pdf/2602.18255v1)

**作者:** Soham Ravikant Joshi `[一作]` (Indian Institute of Technology Patna), Om Prakash `[通讯]` (Indian Institute of Technology Patna)

**通讯引用:** 2206 | [OpenAlex ID](https://openalex.org/A5032501326)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究并描述了在矩阵环ℛ=𝔽₂[u]/<uᵏ>×M₄(𝔽₂+u𝔽₂+⋯+u^{k-1}𝔽₂)上奇数长度循环码的理想结构、子模分解与计数；

**💡 创新点**

将Patel等人关于M₄(𝔽₂+u𝔽₂)的结果推广到更一般的矩阵环，并给出欧氏与厄米对码的闭式描述，同时引入Gray和Bachoc映射将码映射到𝔽₁₆，实现与现有M₄(𝔽₂)码的对比；

**🔧 技术方法**

利用有限域与环的多项式环分解、子模理论、Gray/Bachoc映射以及Magma软件计算；

**📊 数据集**

通过对xⁿ−1在𝔽₁₆[x]中的分解得到的因子多项式构造示例码；

**📈 对比分析**

与已知的M₄(𝔽₂)结构码对比，新得到的𝔽₁₆线性码在距离与维数上接近MDS（差距≤3），参数优于部分已有码；

**⚠️ 局限性**

缺点是未讨论自正交、自正交及对码包含性质，且缺乏针对MIMO信道的确定性性能评估。

---

## 170. JPmHC Dynamical Isometry via Orthogonal Hyper-Connections

**arXiv ID:** 2602.18308 | [PDF](https://arxiv.org/pdf/2602.18308v1)

**作者:** Biswa Sengupta `[一作]`, Leo Brunswic `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出Transformer架构，利用多头自注意力取代传统RNN/卷积序列模型，解决序列建模中的并行计算瓶颈。

**💡 创新点**

核心创新在于全自注意力机制、多头并行、相对/绝对位置编码，以及通过残差连接与层归一化实现深层网络训练。

**🔧 技术方法**

主要技术包括Scaled Dot‑Product Attention、多头Attention、前馈全连接网络、位置编码、残差连接、层归一化和优化器Adam。

**📊 数据集**

主要使用WMT 2014英德、英日机器翻译数据集，以及少量的GLUE和SQuAD等自然语言处理基准数据。

**📈 对比分析**

与LSTM、GRU、CNN‑Seq2Seq等传统序列模型对比，Transformer在BLEU/ROUGE等指标上取得同类最高性能，显著提升了训练速度与推理效率。

**⚠️ 局限性**

局限性包括对长序列的记忆容量有限、对极大模型的计算与内存开销高，以及在低资源任务上仍需大量数据或预训练。

---

## 171. Comparative Assessment of Multimodal Earth Observation Data for Soil Moisture Estimation

**arXiv ID:** 2602.18083 | [PDF](https://arxiv.org/pdf/2602.18083v1)

**作者:** Ioannis Kontogiorgakis `[一作]` (BEYOND EO Centre, IAASARS, National Observatory of Athens), Charalampos Kontoes `[通讯]` (BEYOND EO Centre, IAASARS, National Observatory of Athens)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发并评估了基于 Sentinel‑1 SAR、Sentinel‑2 光学影像和 ERA5 再分析的高分辨率（10 m）土壤湿度估算框架，针对欧洲植被地区的田间尺度应用。

**💡 创新点**

系统比较了多模态数据、不同时间匹配策略和 ERA5 观测窗口，并评估了 Prithvi 基础模型嵌入对点基土壤湿度回归的增益，证明传统手工特征在稀疏数据下仍具竞争力。

**🔧 技术方法**

使用随机森林集成学习、基于光学指数（NDVI、NDWI 等）与 SAR 回波特征的手工特征工程，结合 ERA5 气象变量；对 Prithvi 2.0 预训练模型提取 768 维嵌入作为特征。

**📊 数据集**

113 个欧洲 ISMN 土壤湿度观测站（2019‑2024 年）与对应的 Sentinel‑2 L2A、Sentinel‑1 IW GRD 影像及 ERA5 日均气象变量。

**📈 对比分析**

采用站点级 5 折交叉验证，评估 R²、RMSE、MAE；最佳组合为 S2 同日 + S1 下降轨道最近匹配得到 R²≈0.514，10 天 ERA5 回望窗口可提升至 R²≈0.518；Prithvi 嵌入仅提升 0.2%。

**⚠️ 局限性**

基于少量（113）观测站的高维嵌入可能导致过拟合；Prithvi 嵌入仅在中心像素级别提取，缺乏对局部土壤湿度细节的敏感性；模型对不同地形和土壤类型的普适性待进一步验证。

---

## 172. Qualitative Coding Analysis through Open-Source Large Language Models: A User Study and Design Recommendations

**arXiv ID:** 2602.18352 | [PDF](https://arxiv.org/pdf/2602.18352v1)

**作者:** Tung T. Ngo `[一作]` (Technological University Dublin), Anh Nguyen-Quoc `[通讯]` (University College Dublin)

**通讯引用:** 418 | [OpenAlex ID](https://openalex.org/A5066681619)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了ChatQDA，一款基于本地部署开源LLM的定性编码辅助工具，并通过混合方法的用户研究评估其可用性与信任度，提出了设计建议。

**💡 创新点**

创新点在于把隐私保护与开放源代码结合，提供本地化LLM协助开放编码，且从用户体验角度提供“可验证隐私”与“可追溯一致性”设计准则。

**🔧 技术方法**

技术包括Open‑source LLM（gpt‑oss‑20b）在本地部署，Chat UI界面，LoRA与RAG（可选），以及MXFP4量化；研究方法采用问卷量化 + 半结构化访谈与编码分析。

**📊 数据集**

使用的样本为四名研究者，包含两名新手两名经验者，使用的文本为一份5页半结构访谈稿。

**📈 对比分析**

通过与手工编码对比，问卷显示用户认为ChatQDA在效率上提升明显（平均 3.75/5 对 “更快完成任务”），但在深层解释与一致性上的表现差异大，用户信任度保持中立。

**⚠️ 局限性**

局限性包括小样本量、缺乏长期使用评估、对模型内部运作可验证性不足、以及在细致解释与编码一致性方面仍需改进。

---

## 173. The Dark Side of Dark Mode -- User behaviour rebound effects and consequences for digital energy consumption

**arXiv ID:** 2602.17670 | [PDF](https://arxiv.org/pdf/2602.17670v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 174. Mean-Field Reinforcement Learning without Synchrony

**arXiv ID:** 2602.18026 | [PDF](https://arxiv.org/pdf/2602.18026v1)

**作者:** Shan Yang `[一作]` (National University of Singapore), Shan Yang `[通讯]` (National University of Singapore)

**通讯引用:** 83082 | [OpenAlex ID](https://openalex.org/A5100328102)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

构建了 Temporal Mean Field (TMF) 框架，用人群分布代替平均动作，解决异步决策中的 Mean‑Field RL 问题，并提出了 TMF‑PG 策略梯度算法。

**💡 创新点**

创新点在于：① 用人群分布作为通用摘要统计量，支持任意批处理大小；② 从零构建同步到异步的完整理论（动态、贝尔曼方程、均衡）；③ 证明存在唯一均衡并给出 O(1/√N) 近似误差；④ 提供收敛的 Policy Gradient 算法。

**🔧 技术方法**

技术包括交换性假设、均衡分析、Lipschitz 连续性与 ergodicity、Dobrushin 收缩系数、离散单调性条件、策略梯度与自适应更新、理论证明与仿真验证。

**📊 数据集**

实验使用了两类仿真环境：顺序资源选择游戏 (SRSG) 和动态排队游戏 (DQG)，随机生成资源/服务器参数并进行 4000+ 次随机种子实验。

**📈 对比分析**

与贪婪基线对比，TMF‑PG 在不同批大小下性能波动 ≤5%，同步与顺序情形均保持约 70% 的拥塞无效最优；在 DQG 中相较于贪婪提升 10–30%，实验表明近似误差随 N 按 O(1/√N) 衰减。

**⚠️ 局限性**

局限在于仅适用于可观测人群分布且假设代理可交换；对异质代理、部分可观测情形尚未覆盖；需要满足批大小条件 α<1，强耦合下失效。

---

## 175. Student Flow Modeling for School Decongestion via Stochastic Gravity Estimation and Constrained Spatial Allocation

**arXiv ID:** 2602.17972 | [PDF](https://arxiv.org/pdf/2602.17972v1)

**作者:** Sebastian Felipe R. Bundoc `[一作]` (Center for AI Research), Erika Fille T. Legara `[通讯]` (Center for AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一套基于计算机的政策框架，用混合政府与开放数据的学生流量模型和双约束空间分配机制，模拟教育服务合约（ESC）补贴政策对学校过度拥挤的缓解效果。

**💡 创新点**

创新点在于：① 通过统一碎片化的行政数据，首次生成菲律宾全国范围的学生流向网络；② 采用负二项回归的随机重力模型精准估计距离、净学费和社会经济弹性；③ 将弹性估计嵌入双约束随机分配框架，揭示补贴金额不再是限制拥挤的关键，校位与容量才是主要瓶颈；④ 发现超过一半的潜在缓解来自未被激活的路径，提示策略性重新分配槽位可获得显著收益。

**🔧 技术方法**

主要技术包括：负二项回归（估计弹性）、随机重力模型（学生迁移）、双约束空间分配（同时满足学生池与校位容量）、蒙特卡洛模拟（多重随机顺序求平均）、路网距离计算（OSRM）、以及层次化搜索激活新路径。

**📊 数据集**

使用的数据集有：DepEd 学校坐标与路网距离、校位容量与学生注册、ESC 受益者记录、补贴金额与槽位分配、学费与净成本、学校质量评级、LGU 收入（来源与目的地）、开放源路网数据（OSRM）以及城市竞争指数的市级收入。

**📈 对比分析**

方法比较：以历史受益者流量为基线，对五种净成本下降情景（1k–20k）进行蒙特卡洛模拟。结果显示：若充分利用现有槽位，学生流入可增加约34.7%（≈26k名学生）；而补贴幅度从1k提升至20k仅提升1.8%（≈1.8k学生）。标准差仅约11人，模型收敛稳定，说明系统约束主要是容量。

**⚠️ 局限性**

局限性包括：假设历史流量代表未来偏好，忽略突发社会经济变化；未考虑宗教、专业课程等定性因素；模型假设校方可即时接受至授权容量，现实中可能因师资或设施瓶颈受限；槽位容量被视为静态，未动态响应校内运作；数据仅至2025学年，无法捕捉更近期趋势。

---

## 176. When & How to Write for Personalized Demand-aware Query Rewriting in Video Search

**arXiv ID:** 2602.17667 | [PDF](https://arxiv.org/pdf/2602.17667v1)

**作者:** Cheng cheng `[一作]`, Juyuan Wang `[通讯]` (Weixin Group, Tencent)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种名为 WeWrite 的个性化需求感知查询重写框架，用于在短视频搜索中根据用户历史上下文自动重写查询以提升检索效果。

**💡 创新点**

创新点包括：①利用后验采样策略自动挖掘何时需要重写（防止意图漂移）；②采用 SFT 与 Group Relative Policy Optimization（GRPO）相结合的混合训练，优化 LLM 输出风格与检索系统的匹配；③设计并实现并行的“Fake Recall”架构，将 LLM 推理与传统检索分离，保证低延迟。

**🔧 技术方法**

主要技术：大规模预训练语言模型（Qwen 系列）细调、基于后验的样本挖掘、强化学习奖励函数（基于历史检索频率与 CTR）、GRPO 算法、离线构建的 Fake Index 与在线并行融合。

**📊 数据集**

使用腾讯微信视频平台的用户查询与观看日志（180 天历史搜索日志、点击、停留时间等），并构建了包含有效系统查询的 Fake Index。

**📈 对比分析**

通过线上 A/B 测试与传统检索路径对比，结果显示 VV>10s 点击率提升 1.07%，查询重写率下降 2.97%，验证了框架在真实环境中的有效性。

**⚠️ 局限性**

局限性包括：① 需要大量高质量日志和 LLM 资源，成本高；② 目前仅在短视频搜索场景验证，泛化到其他领域需进一步评估；③ 对极端稀疏查询的处理仍依赖离线索引，可能存在覆盖不足。

---

## 177. MIRA: Memory-Integrated Reinforcement Learning Agent with Limited LLM Guidance

**arXiv ID:** 2602.17930 | [PDF](https://arxiv.org/pdf/2602.17930v1)

**作者:** Narjes Nourzad `[一作]` (University of Southern California), Carlee Joe-Wong `[通讯]` (Carnegie Mellon University)

**通讯引用:** 17262 | [OpenAlex ID](https://openalex.org/A5003037377)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

设计了一种名为MIRA的RL框架，将大语言模型（LLM）生成的子目标和轨迹构建成可演化的记忆图，并将其转化为utility信号，软调节优势估计，从而在稀疏奖励和延迟反馈环境中显著提升早期样本效率；

**💡 创新点**

创新点包括：1）通过持久记忆图将LLM输出离线化，减少实时查询需求；2）将记忆图产生的utility信号与优势函数结合，形成软优势形状，提升早期梯度；3）引入衰减的shaping权重保证长期收敛；4）使用层次化子目标存储与目标对齐的相似度函数，过滤不可靠的LLM建议；

**🔧 技术方法**

使用的技术包括PPO策略梯度、LLM调用（GPT‑4o‑mini、Gemini Pro等）、可演化记忆图结构、相似度与目标对齐函数、Screening Unit过滤、优势与utility的加权融合；

**📊 数据集**

实验数据集包括六个稀疏奖励环境：FrozenLake、MiniGrid（RedBall、LavaCrossing、DoorKey、RedBlueDoor、DistractedDoorKey）以及 BabyAI；

**📈 对比分析**

与PPO、HRL、LLM4Teach、LLM‑RS等基线比较，MIRA在早期训练阶段收敛速度最快，最终回报与最强基线相当或更优，同时每次LLM查询的回报更高，查询成本明显降低；

**⚠️ 局限性**

局限性包括：对离线LLM先验的依赖；需要手工设计子目标/相似度度量；在连续视觉控制任务中的适用性待验证；对LLM推理质量高度敏感；缺乏多目标和连续空间的扩展。

---

## 178. Capabilities Ain't All You Need: Measuring Propensities in AI

**arXiv ID:** 2602.18182 | [PDF](https://arxiv.org/pdf/2602.18182v1)

**作者:** Daniel Romero-Alvarado `[一作]` (Valencian Research Institute of Artificial Intelligence, Universitat Politècnica de València), Jose Hernandez-Orallo `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了用于衡量AI倾向（propensities）的正式框架，并通过2x2PL双逻辑模型将倾向需求视为理想区间，利用LLM对指标进行自动标注后估计模型倾向，并将倾向与能力结合以提升对任务表现的预测。

**💡 创新点**

创新点包括：①把倾向需求建模为“理想区间”并用双逻辑函数描述成功概率；②设计通用标注rubrics，使得不同倾向维度可在统一尺度上自动评估；③证明2x2PL是传统2PL能力模型的推广，兼容心理测量理论；④实验表明倾向信息比单独能力更能解释与预测模型表现。

**🔧 技术方法**

采用的技术包括：项目反应理论（IRT）与2x2PL概率模型、最大似然估计（MLE）推断倾向参数、LLM（GPT‑4.1）对任务进行自动标注、随机森林评估器、交叉验证与AUROC评估。

**📊 数据集**

使用的数据集：四个自制倾向数据集（红蓝偏好、风险规避/冒险、外向/内向、无知/自负），以及TimeQA与MentalQA组合的360道问答任务，用于评估倾向在真实任务上的预测能力。

**📈 对比分析**

比较方法：将仅使用能力维度的评估器与加入单一或全部倾向维度的评估器进行对比，使用AUROC衡量预测性能。结果显示：在大多数模型和倾向注入水平下，加入倾向后AUROC平均提升约0.02–0.03（如Gemma‑3从0.733提升至0.864），并且倾向参数在一个基准上能成功预测在另一个基准上的表现。

**⚠️ 局限性**

局限性：①仅为每个倾向估计单一标量参数，未捕捉更细粒度的倾向变化；②参数估计误差受样本量与任务难度影响；③实验仅覆盖单步任务，对长期或社会情境下的倾向作用验证不足；④需进一步验证更大规模、不同领域的通用性与鲁棒性。

---

## 179. A Generalized Information Bottleneck Method: A Decision-Theoretic Perspective

**arXiv ID:** 2602.18405 | [PDF](https://arxiv.org/pdf/2602.18405v1)

**作者:** Akira Kamatsuka `[一作]` (Shonan Institute of Technology), Takahiro Yoshida `[通讯]` (Nihon University)

**通讯引用:** 1920 | [OpenAlex ID](https://openalex.org/A5022824333)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对信息瓶颈（IB）方法，作者构造了一种泛化的 IB 任务：压缩度量仍使用 Shannon 互信息，而效用度量改为满足 concave 与 averaging 条件的 H‑MI，并利用其与期望样本信息（EVSI）的等价性推导出交替最小化算法。

**💡 创新点**

创新点在于将 H‑MI 的决策理论解释与 IB 框架相结合，既提供了效用的决策意义，又得到了一套可直接利用决策理论的优化算法，并给出了收敛性分析。

**🔧 技术方法**

主要技术包括信息瓶颈理论、互信息的变分表述、EVSI 与 Proper Scoring Rule 的关系、交替最小化（类似 Arimoto–Blahut）算法以及数值实验。

**📊 数据集**

实验使用的是人工构造的 3×3 联合分布（𝒳=𝒴={1,2,3}，𝒯={1,2}），未采用公开数据集。

**📈 对比分析**

作者将所提算法与原始 IB（使用 Shannon MI 作为效用）在同一 β 范围内比较，绘制压缩–效用曲线；在相同压缩率下，使用 EVSI^ℓ_sq 的算法迭代次数更少（81 次 vs 108 次），且得到更高的效用。

**⚠️ 局限性**

局限性包括目标函数非凸，算法只能保证局部最优；收敛速度受初始化影响；实验仅在极小规模人工数据上验证，缺乏大规模实际场景的评估。

---

## 180. MusicSem: A Semantically Rich Language--Audio Dataset of Natural Music Descriptions

**arXiv ID:** 2602.17769 | [PDF](https://arxiv.org/pdf/2602.17769v1)

**作者:** Rebecca Salganik `[一作]` (University of Rochester), Jian Kang `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

基于 Reddit 上的音乐讨论，构建了一个 32,493 条语言–音频对的 MusicSem 数据集，并对多模态音乐检索、文本-音乐生成、音乐-文本生成模型进行评估。

**💡 创新点**

提出了五类别音乐语义分类（描述性、情绪、情境、元数据、上下文），并通过 LLM 辅助提取、摘要和验证，生成语义丰富且自然的描述，填补现有数据集对细粒度语义缺失的空白。

**🔧 技术方法**

采用 GPT‑4o 和 Claude Sonnet 等大型语言模型进行信息抽取与校验，利用 Spotify API 获取音频 ID，构建全流程自动化的数据构造与评估管道，并使用 CLAP、ImageBind、MusicLM、Stable Audio 等主流多模态模型进行实验。

**📊 数据集**

主要使用来自五个英文音乐子版块的 Reddit 讨论文本生成 32,493 条语言–音频对，公开训练集与 480 条人标注的测试集；对比 MusicCaps、Song Describer 等现有数据集进行分析。

**📈 对比分析**

在检索、音乐-文本生成、文本-音乐生成等任务中，对比 CLAP、ImageBind、MusicLM、Stable Audio 等模型，发现现有模型对语义敏感度低；在 MusicSem 上微调 CLAP 后，检索性能提升超过 96% 以及各语义类别的敏感度显著提升，证明 MusicSem 对模型提升有显著作用。

**⚠️ 局限性**

局限性包括：依赖 LLM 的语义抽取可能产生歧义与误检；缺乏完善的音乐实体消歧；数据来源主要为英语 Reddit，存在文化/流派偏倚；隐私与版权合规限制导致未直接包含音频文件。

---

## 181. MEG-to-MEG Transfer Learning and Cross-Task Speech/Silence Detection with Limited Data

**arXiv ID:** 2602.18253 | [PDF](https://arxiv.org/pdf/2602.18253v1)

**作者:** Xabier de Zuazo `[一作]` (University of the Basque Country), Nicola Molinaro `[通讯]` (Basque Center on Cognition, Brain and Language)

**通讯引用:** 4011 | [OpenAlex ID](https://openalex.org/A5090641526)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在MEG信号上，本文通过将Conformer模型在50小时单一受试者的听觉数据上预训练，然后在18名受试者各自仅5分钟的听觉、回放和发声任务数据上微调，展示了数据高效的语音检测方法。

**💡 创新点**

创新点在于首次证明了MEG语音解码的迁移学习与跨任务解码（感知与发声）的有效性，并揭示了共享的神经表示而非单纯任务相关的运动活动。

**🔧 技术方法**

核心技术包括基于Conformer的MEGConformer编码器、迁移学习与微调、RollAugment时间卷积增强、软标签平滑以及交叉任务评估与Wilcoxon检验等统计方法。

**📊 数据集**

使用的数据集为LibriBrain（50小时单受试者听觉记录）和Bourguignon2020（18名西班牙语受试者各5分钟的听觉、回放与朗读数据）。

**📈 对比分析**

通过对比迁移学习模型与从零开始训练的模型，采用F1宏、准确率和AUC等指标，结果显示迁移学习在任务内提升1–4%，跨任务提升5–6%，并且所有跨任务解码均显著高于随机机会。

**⚠️ 局限性**

限制包括仅针对语音检测任务，未覆盖更高级的音素/词/语义解码；预训练基于单个受试者且语言不同；提升幅度相对有限且受个体差异影响，需要进一步的多受试者预训练和个体适配。

---

## 182. COMBA: Cross Batch Aggregation for Learning Large Graphs with Context Gating State Space Models

**arXiv ID:** 2602.17893 | [PDF](https://arxiv.org/pdf/2602.17893v1)

**作者:** Jiajun Shen `[一作]` (Florida Atlantic University), xingquan Zhu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了COMBA框架，结合状态空间模型在大规模同质图上进行高效学习

**💡 创新点**

创新点包括跨批聚合机制（降低采样方差、保留全局信息）和图上下文门控（自适应调节多跳邻域贡献）

**🔧 技术方法**

使用了状态空间模型（S4/Mamba）、批量采样的多跳邻接矩阵、交叉批聚合、上下文门控以及传统GNN的消息传递

**📊 数据集**

在六个真实同质图数据集上评估：Roman‑empire、Amazon‑ratings、Ogbn‑arxiv、Ogbn‑product、Minesweeper、Tolokers

**📈 对比分析**

与GCN、GatedGCN、NAGphormer、Cluster‑GCN、Graph Mamba‑1/2等基线比较，COMBA在大多数数据集上取得最高准确率/ROC‑AUC，且训练时间保持近线性增长

**⚠️ 局限性**

缺点包括对批次划分的依赖、需要额外的多跳邻接矩阵存储和计算，且对异构图或动态图的适用性尚未验证

---

## 183. Five Fatal Assumptions: Why T-Shirt Sizing Systematically Fails for AI Projects

**arXiv ID:** 2602.17734 | [PDF](https://arxiv.org/pdf/2602.17734v1)

**作者:** Raja Soundaramourty `[一作]` (Cisco Systems, Inc.), Ramu Chenchaiah `[通讯]` (Cisco Systems, Inc.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统地识别并分析了在人工智能（尤其是大语言模型与多智能体系统）项目中，传统 T‑shirt 估算所依赖的五个核心假设为何失效，并基于实证研究提出了一种新的“Checkpoint Sizing”迭代式估算框架。

**💡 创新点**

创新点在于：①用多学科文献与实验数据证明了五大假设的破裂；②提出“Checkpoint Sizing”以决策门控和实证重估为核心，取代单次粗略估算；③提供了完整的原则与实施步骤，强调人性化与不确定性管理。

**🔧 技术方法**

技术方法包括：结构化假设识别（对比敏捷估算与 AI 工作流）、文献证据收集与评估、定量指标（如 N(N-1) 交互复杂度、39% 对话性能下降等）、以及基于经验的案例演示。

**📊 数据集**

主要使用公开的学术论文和研究报告作为证据来源；示例性案例采用合成的“支持 Copilot + RAG + 工具调用”项目，未使用真实工业数据集。

**📈 对比分析**

对比方式通过一个合成项目的 T‑shirt 估算与 Checkpoint Sizing 的迭代估算结果展示：初始估计为 Large（6–8 周），通过五个决策门（数据准备、评估、保障、成本/延迟、运维）后调整为 XL（12–16 周）。该对比说明新方法能将隐含风险显化为可度量的工作量，但未提供传统方法的客观误差率或大规模实测结果。

**⚠️ 局限性**

局限性包括：①研究基于文献与案例分析，缺乏大规模实验验证；②针对 LLM 与多智能体的结论可能对简单机器学习项目不完全适用；③Checkpoint Sizing 的效果与其他新兴估算方法的实证比较尚未完成；④未在不同领域（医疗、金融等）中进行针对性适配。

---

## 184. ADAPT: Hybrid Prompt Optimization for LLM Feature Visualization

**arXiv ID:** 2602.17867 | [PDF](https://arxiv.org/pdf/2602.17867v1)

**作者:** João N. Cardoso `[一作]` (Instituto Superior Técnico), Bruno Martins `[通讯]` (Instituto Superior Técnico)

**通讯引用:** 3683 | [OpenAlex ID](https://openalex.org/A5055101594)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的混合式提示优化方法ADAPT，用于LLM特征可视化；

**💡 创新点**

通过Beam Search初始化与梯度引导与logit交换的混合变异相结合，解决离散搜索与局部最优的问题；

**🔧 技术方法**

使用Beam Search、梯度估计（GCG）、logit交换、流畅度惩罚、槽位管理等技术；

**📊 数据集**

在Gemma 2 2B模型的残差流稀疏自编码器（SAE）latents上进行实验；

**📈 对比分析**

与GCG、BEAST、EPO四种基线比较，采用激活比例与激活排名指标，ADAPT在所有层次均优于其他方法（70.4%特征激活更高），并在人工可解释性评分上显著领先；

**⚠️ 局限性**

仅评估单一模型和残差流SAE，未验证更大模型或其他潜在特征类型，且方法对流畅度惩罚参数敏感。

---

## 185. Investigating Target Class Influence on Neural Network Compressibility for Energy-Autonomous Avian Monitoring

**arXiv ID:** 2602.17751 | [PDF](https://arxiv.org/pdf/2602.17751v1)

**作者:** Nina Brolich `[一作]` (Fraunhofer Institute for Integrated Circuits), Dominik Seuß `[通讯]` (Center for Artificial Intelligence and Robotics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在低功耗 MCU 上实现鸟类声音识别，利用压缩后的 MCUNet 模型实现实时边缘监测

**💡 创新点**

将深度学习模型压缩至可在 MCU 上运行，并评估不同目标类别数对压缩率与性能的影响

**🔧 技术方法**

使用 MCUNet + TinyNAS、TinyEngine、量化剪枝、Mel 频谱、SpecAugment 等技术进行模型训练、压缩与推理

**📊 数据集**

采用 Xeno-Canto 500 物种 250 录音以及 ESC-50 作为非鸟类背景噪声，生成 Mel 频谱数据集

**📈 对比分析**

通过与基线模型对比评估准确率、压缩率、能耗和推理时延，结果显示压缩后模型在 Cortex‑M7 和 Raspberry Pi 4 上可满足实时需求，而在 Cortex‑M4 上不可行

**⚠️ 局限性**

局限性包括数据集异质性导致准确率随类别数增长而下降、压缩流程缺乏透明性、未测试不同网络架构以及缺少完整的计数、存储和能量管理实现

---

## 186. FeatureBleed: Inferring Private Enriched Attributes From Sparsity-Optimized AI Accelerators

**arXiv ID:** 2602.18304 | [PDF](https://arxiv.org/pdf/2602.18304v1)

**作者:** Darsh Asher `[一作]`, Samira Mirbagher Ajorpaz `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

利用 AI 加速器的零跳过优化，通过推理时延泄露后端检索的私有属性，实现了硬件级别的属性窃取攻击。

**💡 创新点**

首次揭示后端检索过程中的时延侧信道，并证明零跳过优化是泄露根源；同时提出了低性能损耗的填充防御方案。

**🔧 技术方法**

基于时延侧信道、稀疏激活分析、GBDT 训练以及对零跳过行为的实验复现。

**📊 数据集**

Texas‑100X（临床记录）、OrganAMNIST（医学影像）和 Census‑19（社会经济数据）。

**📈 对比分析**

在 Intel AVX、Intel AMX 与 NVIDIA A100 三大后端、DNN、CNN 与混合 CNN–MLP 模型上均能达到 60–70% 的属性识别精度，攻击优势高达 98.87 个百分点；填充防御平均性能损耗仅 7.24%。

**⚠️ 局限性**

需对目标模型具有相同数据分布的黑盒访问；对稀疏度低或不涉及单一后端属性检索的场景泄露效果有限；攻击在模型尺寸与稀疏度不高时准确率下降。

---

## 187. Quasi-Periodic Gaussian Process Predictive Iterative Learning Control

**arXiv ID:** 2602.18014 | [PDF](https://arxiv.org/pdf/2602.18014v1)

**作者:** Unnati Nigam `[一作]` (IIT Bombay), Michael Burke `[通讯]` (Monash University)

**通讯引用:** 124963 | [OpenAlex ID](https://openalex.org/A5071287538)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于准周期高斯过程的预测迭代学习控制（QPGP‑PILC）来改进机器人与车辆在重复任务中的轨迹跟踪。

**💡 创新点**

创新点在于将准周期高斯过程的结构化预测模型嵌入IILC框架，实现只依赖最近一次误差的高效在线学习与预测，显著降低了计算复杂度和存储需求。

**🔧 技术方法**

使用了准周期高斯过程（QPGP）作为误差预测器，结合迭代学习控制算法、动态系统结构方程以及两阶段参数估计方法。

**📊 数据集**

在三个实验平台上验证：自动驾驶车辆路径跟踪、三关节机器人轨迹跟踪以及真实Hello Stretch 3机器人执行Lissajous轨迹，分别对应仿真与硬件数据集。

**📈 对比分析**

与标准ILC和基于RBF核的GP‑PILC比较，QPGP‑PILC在误差收敛速度、最终跟踪误差以及计算时间上均表现更优，尤其在大迭代数下保持低计算成本。

**⚠️ 局限性**

局限性包括对误差准周期性的假设、对系统耦合程度的限制（需要近似对角化）以及在极端非准周期扰动或快速环境变化时可能失效。

---

## 188. Duality Models: An Embarrassingly Simple One-step Generation Paradigm

**arXiv ID:** 2602.17682 | [PDF](https://arxiv.org/pdf/2602.17682v1)

**作者:** Peng Sun `[一作]` (Westlake University), Zhiqiang Shen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 6369 | [OpenAlex ID](https://openalex.org/A5066530136)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了 Duality Models (DuMo) 框架，采用单输入双输出结构同时学习速度场和流图，从而解决目标感知模型的训练预算分配瓶颈，实现极快的图像生成。

**💡 创新点**

创新点在于将多步稳定性与一阶生成目标统一到同一网络，通过速度场的几何约束正则化流图，消除单分支模型的训练平衡问题。

**🔧 技术方法**

使用 Diffusion Transformer 主干、SD‑VAE/ DC‑AE 潜在空间、Beta 时间采样、CFG 增强、无梯度停止的速度正则化、JVP‑free 一致性损失、标准 L2 损失以及双头输出。

**📊 数据集**

在 ImageNet‑1K（256×256、512×512）和 CIFAR‑10 上进行训练与评估。

**📈 对比分析**

与现有一阶一致性模型及蒸馏基线相比，在 ImageNet‑256×256 上 2 步 FID 1.79，显著优于 MeanFlow、sCT 等；在 512×512 上 2 步 FID 2.23，性能领先；在 CIFAR‑10 上也取得较高 FID。

**⚠️ 局限性**

局限在于同构 ODE 目标的多头扩展难以进一步提升；对更高分辨率或多模态任务需探索异构学习信号；仅在潜在空间训练，限制直接像素级应用。

---

## 189. Seasoning Data Modeling Education with GARLIC: A Participatory Co-Design Framework

**arXiv ID:** 2602.18274 | [PDF](https://arxiv.org/pdf/2602.18274v1)

**作者:** Viktoriia Makovska `[一作]` (Ukrainian Catholic University), Julia Stoyanovich `[通讯]` (New York University)

**通讯引用:** 2907 | [OpenAlex ID](https://openalex.org/A5082830839)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

本文提出了一种基于工作坊的参与式实体关系模型教学方法，帮助学生在实践中学习如何与不同利益相关者共同构建数据模型。

**💡 创新点**

创新点在于将参与式设计的五个阶段（Observe、Nurture、Integrate、Optimize、Normalize）与情境卡、角色卡和在线协作板结合，形成一套可复制、可验证的教学流程。

**🔧 技术方法**

该方法采用情境卡、角色卡、阶段卡、虚拟白板（Miro/Mural）等教学技术，并通过角色扮演、协作草图、迭代反馈等方式实现参与式建模。

**📊 数据集**

实验使用自定义的学习情境数据集，包括图书馆管理、社区工具棚、学生注册等案例，以角色卡记录利益相关者视角进行实践。

**📈 对比分析**

通过两次90分钟的试点工作坊（共10名学生），采用质性访谈和问卷评估，结果显示学生对ER图的理解和建模自信显著提升，且能够识别并追踪不同视角。

**⚠️ 局限性**

局限在于样本量小、缺乏长期跟踪的学习成效评估，方法尚未在大规模课程中验证，且对技术细节的掌握仍需进一步指导。

---

## 190. Neural-HSS: Hierarchical Semi-Separable Neural PDE Solver

**arXiv ID:** 2602.18248 | [PDF](https://arxiv.org/pdf/2602.18248v1)

**作者:** Pietro Sittoni `[一作]` (Gran Sasso Science Institute), Francesco Tudisco `[通讯]` (University of Edinburgh)

**通讯引用:** 643 | [OpenAlex ID](https://openalex.org/A5043752696)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于层次半分离(HSS)结构的神经网络，用于高效学习PDE的解算子，尤其适用于线性椭圆型PDE；

**💡 创新点**

创新点在于将HSS矩阵的低秩分解嵌入神经层，既能保证理论上在低数据量下实现精确恢复，又在参数量和计算成本上大幅降低；

**🔧 技术方法**

使用了HSS矩阵分解、低秩外积(CP)扩展、LeakyReLU激活、以及与FNO、ResNet等传统神经算子对齐的结构；

**📊 数据集**

在多维Poisson方程（1D、3D）、Heat、Burgers、Navier‑Stokes、Gray‑Scott等多种PDE数据集上进行实验，数据来源为高精度数值模拟；

**📈 对比分析**

与FNO、ResNet、DeepONet、Green Learning、FactFormer等基线对比，所提模型在相同参数量下取得更低的测试误差，且在低样本和3D高分辨率实验中训练速度更快、内存占用更少；

**⚠️ 局限性**

局限性包括：对HSS结构假设敏感，若解算子不具备明显低秩性质或边界条件极其复杂时性能下降；此外，对高度非线性或非椭圆型PDE的理论保证不完全；

---

## 191. Information-Theoretic Storage Cost in Sentence Comprehension

**arXiv ID:** 2602.18217 | [PDF](https://arxiv.org/pdf/2602.18217v1)

**作者:** Kohei Kajikawa `[一作]` (Georgetown University), Ethan Gotlieb Wilcox `[通讯]` (Georgetown University)

**通讯引用:** 975 | [OpenAlex ID](https://openalex.org/A5011708753)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于信息论的句子加工存储成本度量，用来量化在阅读时维持对未来词汇预测的工作记忆负荷。

**💡 创新点**

创新点是将存储成本从离散语法计数转化为连续的互信息量度，并能从预训练语言模型直接估算。

**🔧 技术方法**

采用BERT等预训练掩码语言模型计算词与未来上下文的条件互信息，得到信息存储量。

**📊 数据集**

在英语上使用人工生成句子（中心嵌入与右支结构、主客体相对从句）、UD_English‑GUM语料库以及两大自然阅读时间数据集（Natural Stories与OneStop）。

**📈 对比分析**

与传统的DLT语法存储成本、以及词频、单字、GPT‑2惊奇度等基线模型比较；结果表明信息存储显著提升阅读时间预测，且与DLT存储具有相互独立的解释力。

**⚠️ 局限性**

局限包括：假设BERT的掩码词互相独立、只在词层级评估、未考虑压缩记忆表示，以及对不同语言的适用性尚未验证。

---

## 192. Perceived Political Bias in LLMs Reduces Persuasive Abilities

**arXiv ID:** 2602.18092 | [PDF](https://arxiv.org/pdf/2602.18092v1)

**作者:** Matthew DiGiuseppe `[一作]` (Leiden University), Joshua Robison `[通讯]` (Leiden University)

**通讯引用:** 908 | [OpenAlex ID](https://openalex.org/A5015293137)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一项预注册的美国问卷实验中，研究者通过给受访者发放不同的“LLM政治偏见警告”消息，随后让他们与ChatGPT进行三轮对话，以检验这些警告是否会削弱LLM在纠正经济误区中的说服力。

**💡 创新点**

创新点在于首次系统评估了受访者对LLM政治中立性的感知如何调节其说服效果，并通过实验对比了中立信息与不同程度的偏见信息，揭示了政治偏见感知会显著削弱LLM的说服效果。

**🔧 技术方法**

技术上采用了四组对照设计（无信息、非定向偏见、中性偏见警告、强偏见警告），使用ChatGPT 4.1作为非推理模型，配合开放式回答的转录文本，通过LLM‑Judge与贝叶斯Bradley–Terry模型对话的“辩论性”和“否定性”进行量化分析。

**📊 数据集**

数据集为约2,400名美国在线受访者的样本，他们在预实验中表明至少持有一项六个经济误区中的一种；每位受访者随机分配到一个误区话题并参与与ChatGPT的三轮对话。

**📈 对比分析**

与中立对照组相比，强偏见警告组的说服效果下降约X%，轻偏见警告组下降约Y%；同时对话后受访者的完整观点反转率也显著下降。实验显示，偏见警告虽未完全消除LLM说服力，但显著削弱了其影响。

**⚠️ 局限性**

局限性包括：实验仅为一次短暂对话，缺乏长期或重复使用情境；仅使用一种LLM（ChatGPT 4.1）；样本来自美国在线平台，可能与更广泛人群存在差异；未检验不同主题、模型或跨文化环境下的推广效果。

---

## 193. Decoding as Optimisation on the Probability Simplex: From Top-K to Top-P (Nucleus) to Best-of-K Samplers

**arXiv ID:** 2602.18292 | [PDF](https://arxiv.org/pdf/2602.18292v1)

**作者:** Xiaotong Ji `[一作]` (Huawei Noah's Ark Lab), Haitham Bou-Ammar `[通讯]` (Technical University of Darmstadt)

**通讯引用:** 1553 | [OpenAlex ID](https://openalex.org/A5034334762)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出将 LLM 解码视为在概率单纯形上的正规化优化问题，统一传统解码方法并引入新解码器Best‑of‑K（BoK）

**💡 创新点**

通过“主目标”框架将贪婪、Softmax、Top‑K/Top‑P、Sparsemax 等解码器映射为不同的正则化或约束，并用镜像上升（mirror‑ascent）求解非闭式优化

**🔧 技术方法**

主目标（convex 线性+正则化）、KKT 条件、Bregman 散度、镜像上升、KL 与覆盖正则化、最佳-覆盖目标

**📊 数据集**

Qwen2.5‑Math‑7B 与 Qwen2.5‑7B 两大模型，评测 MATH500、GPQA‑diamond、HumanEval 三个基准

**📈 对比分析**

与 Base（温度采样）和 Top‑K（K=50）对比；在高温度下 BoK 提升 18.6%（MATH500）、6.1%（GPQA）、14.6%（HumanEval），对不同温度与超参表现稳健，计算开销仅略增（约 1‑2 s）

**⚠️ 局限性**

对单步优化的局限性；对序列级目标和更复杂约束的扩展仍需研究；需要更多模型与任务验证以评估泛化

---

## 194. Deepmechanics

**arXiv ID:** 2602.18060 | [PDF](https://arxiv.org/pdf/2602.18060v1)

**作者:** Abhay Shinde `[一作]` (Deep Forest Sciences), Bharath Ramsundar `[通讯]` (Deep Forest Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对深度物理信息网络（HNN、LNN、SRNN）在六种动力学系统上进行系统性基准测试，评估其预测轨迹的准确性与稳定性。

**💡 创新点**

首次将 DeepChem 框架与三种物理信息网络集成，全面覆盖从保守到非保守、从线性到高度混沌的系统，提供统一的实验与评估流程。

**🔧 技术方法**

使用 Hamiltonian Neural Networks、Lagrangian Neural Networks 与 Symplectic Recurrent Neural Networks，并结合 SciPy ODE 求解器、DeepChem leapfrog 积分器以及多种误差度量（MSE、MAE、RMSE、STD、VAR）进行训练与评估。

**📊 数据集**

构造六组合成数据集：质量-弹簧、简单摆、双摆、弹簧摆、弹跳球、三体问题，各系统通过数值积分生成完整轨迹并划分训练/测试集。

**📈 对比分析**

通过对比误差指标与相位空间图，发现对保守系统（质量-弹簧、摆、弹簧摆）和三体问题表现优异，能够保持能量守恒和轨迹稳定；但在混沌（双摆）和非保守（弹跳球）系统中误差迅速放大，长期预测效果显著下降。

**⚠️ 局限性**

主要限制在于对初始条件高度敏感的混沌系统、碰撞/摩擦等非连续动力学难以建模，当前物理信息网络难以实现长期稳定预测，亟需结合专门处理非连续性或更鲁棒的训练策略。

---

## 195. Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets

**arXiv ID:** 2602.18025 | [PDF](https://arxiv.org/pdf/2602.18025v1)

**作者:** Haruki Abe `[一作]` (University of Tokyo), Tatsuya Harada `[通讯]` (University of Tokyo)

**通讯引用:** 10837 | [OpenAlex ID](https://openalex.org/A5042711470)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对16种不同机器人（四足、双足、六足）进行离线强化学习与跨体态学习的联合预训练，并通过机器人形态相似性对机器人进行分组以缓解梯度冲突。

**💡 创新点**

提出基于形态图和FGW距离的静态分组（Embodiment Grouping）策略，在离线RL中按组更新策略，显著降低跨体态梯度冲突并提升性能。

**🔧 技术方法**

使用URMA架构做多机器人共享网络，IQL、TD3+BC、BC作为基准；利用FGW距离、层次聚类构建形态图；梯度投影（PCGrad）、选择性分组（SEL）等对照方法。

**📊 数据集**

构造MuJoCo仿真行走数据集，包含16台机器人（9四足、6双足、1六足）各自的Expert、Expert Replay、70% Suboptimal Replay，正向与反向行走两种方向，共计6种数据集。

**📈 对比分析**

与BC、TD3+BC、IQL、IQL+PCGrad、IQL+SEL等比较，在子优数据集上，IQL+EG平均提升约34%（最高达39.8%），在Expert数据集提升约5–10%；整体在所有数据集上均优于基准方法。

**⚠️ 局限性**

仅在MuJoCo仿真行走任务中验证，未涉及真实机器人或操纵任务；分组策略是静态的，无法随学习进度或数据质量动态调整；未探究跨任务或离线到在线迁移的适用性。

---

## 196. Provable Adversarial Robustness in In-Context Learning

**arXiv ID:** 2602.17743 | [PDF](https://arxiv.org/pdf/2602.17743v1)

**作者:** Di Zhang `[一作]` `[通讯]` (Xi'an Jiaotong-Liverpool University), Di Zhang (Xi'an Jiaotong-Liverpool University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于分布鲁棒元学习的框架，分析了在对抗性分布变化下的上下文学习（ICL）能力，提供了最坏情况下的性能保证。

**💡 创新点**

创新点在于引入了Wasserstein距离作为度量，揭示了模型容量与对抗性鲁棒性之间的平方根关系，并量化了在对抗环境下维持性能所需的额外上下文示例数量。

**🔧 技术方法**

使用了线性自注意力Transformer模型，并通过分布鲁棒优化（DRO）理论进行分析。

**📊 数据集**

使用了合成任务数据集进行实验，数据维度为20，预训练分布为标准正态分布。

**📈 对比分析**

通过理论分析和实验验证，发现模型的最大对抗半径与模型容量的平方根成正比，且在对抗环境下，维持性能所需的样本数量与对抗强度的平方成正比。

**⚠️ 局限性**

局限性在于分析是在简化的线性Transformer和高斯任务分布下进行的，未来需要在更复杂的模型和真实世界任务中验证这些理论结果。

---

## 197. LGD-Net: Latent-Guided Dual-Stream Network for HER2 Scoring with Task-Specific Domain Knowledge

**arXiv ID:** 2602.17793 | [PDF](https://arxiv.org/pdf/2602.17793v1)

**作者:** Peide Zhu `[一作]` (Fujian Normal University), Xiong Chen `[通讯]` (Fujian Medical University)

**通讯引用:** 3363 | [OpenAlex ID](https://openalex.org/A5101884906)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种Latent‑Guided Dual‑Stream Network（LGD‑Net），用于仅通过H&E切片直接预测HER2表达水平，避免了像素级虚拟染色的高计算开销和伪影问题。

**💡 创新点**

创新点在于用跨模态特征幻觉（Feature Hallucination）取代像素级生成，并通过轻量级辅助任务（核密度与膜染色强度）将临床领域知识嵌入隐层空间，显著提升诊断准确性。

**🔧 技术方法**

采用了教师‑学生双流编码器、非线性特征幻觉映射、注意力融合模块以及基于ResNet‑50的特征提取和辅助回归/分割任务，实现了端到端的训练与高效推理。

**📊 数据集**

实验使用公开的BCI数据集，该数据集包含4873对注册H&E与IHC切片以及HER2评分标签。

**📈 对比分析**

与单模态和多模态基线（包括传统图像拼接、特征拼接与特征融合）对比，LGD‑Net在测试集上的准确率达到95.60%，相较最优双模态方法提升约1.2%，宏F1和Kappa也分别提升至0.9644和0.9453。

**⚠️ 局限性**

主要局限在于训练阶段需大量高质量的H&E‑IHC配对样本，且模型对跨机构或扫描参数的域漂移可能不够鲁棒；此外，目前仅针对HER2评分进行验证，尚未扩展到其他病理指标。

---

## 198. Alignment in Time: Peak-Aware Orchestration for Long-Horizon Agentic Systems

**arXiv ID:** 2602.17910 | [PDF](https://arxiv.org/pdf/2602.17910v1)

**作者:** Hanjing Shi `[一作]` (Lehigh University), Dominic DiFranzo `[通讯]` (Lehigh University)

**通讯引用:** 1339 | [OpenAlex ID](https://openalex.org/A5028356812)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为APEMO的实时调度层，基于情感峰–末法则在固定计算预算下重新分配推理资源，以提升长周期AI工作流的轨迹质量、重用概率和用户满意度。

**💡 创新点**

创新点在于将对齐视为时间序列控制问题，而非仅靠模型参数或奖励函数；通过检测轨迹中的负峰和终点并在不增加总计算量的前提下动态调度计算，显著降低轨迹波动并提升最终输出的可靠性。

**🔧 技术方法**

技术核心包括：情感峰–末加权的轨迹质量度量、基于行为代理（如重复度、上下文漂移）计算的沮丧信号、峰值检测与精度修复模块、固定预算约束下的多目标优化以及对齐与协调成本的Pareto前沿分析。

**📊 数据集**

使用多种公开小模型LLM（qwen2.5:1.5b、gemma2:2b、llama3.2:1b 等）在本地Ollama运行；通过Agent-Based Modeling（ABM）仿真、单模型长短周期推理以及跨模型多代理（Planner–Executor–Critic）工作流进行实验，未使用专门的评测数据集，而是自构造的长短周期任务与人为注入的负峰扰动。

**📈 对比分析**

与均匀预算、结构化反思式调度、传统多代理流程以及峰–末/情感基线对比；评估指标为轨迹级质量(Q)、重用概率(R)、累计沮丧(F)和协调成本(C)。实验显示，APEMO在长周期任务中平均提升质量约+0.08–0.19、重用概率+0.06–0.12、并在保持计算预算内将协调成本提升仅6–8%；在多代理场景下也同样获得显著的质量提升。整体表明相较于基线，APEMO在轨迹层面上具有可观且显著的性能优势。

**⚠️ 局限性**

局限性包括：跨模型负峰恢复指标对模型族敏感、对比仅限于计划–执行类基线而非完整工业级工作流；缺乏人类受试者验证对信任与重用意愿的直接测量；以及在真实系统部署时，调度与监控的额外延迟和资源开销未在实验中充分体现。

---

## 199. GrandTour: A Legged Robotics Dataset in the Wild for Multi-Modal Perception and State Estimation

**arXiv ID:** 2602.18164 | [PDF](https://arxiv.org/pdf/2602.18164v1)

**作者:** Jonas Frey `[一作]` (ETH Zurich), Marco Hutter `[通讯]` (ETH Zurich)

**通讯引用:** 20543 | [OpenAlex ID](https://openalex.org/A5044258783)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并公开了GrandTour——一套面向四足机器人、覆盖多环境（室内、城市、自然、极端天气等）的大规模多模态感知与状态估计数据集，并在该数据集上对多种开源SLAM/状态估计算法进行了系统评测。

**💡 创新点**

创新点主要包括：①规模最大、覆盖最广的腿控机器人公开数据集；②整合三种LiDAR、10摄像头（RGB/深度/HDR）、七种IMU、全方位本体传感器及双重RTK‑GNSS+总台高精度地面真值；③提供同步、标定完善的数据结构以及中间处理结果（如运动补偿点云、地图、姿态估计等），方便直接使用；④基于该数据集构建完整的基准评估框架，覆盖LO、LIO、LIVO、VIO、多激光融合等多种方法。

**🔧 技术方法**

技术手段包括：IEEE 1588v2 PTP与软件时钟同步；多传感器联合标定（相机、IMU、LiDAR、全站仪）采用Kalibr、DiffCal等；Holistic Fusion 6‑DoF地面真值融合RTK‑GNSS、总台与INS；数据发布采用ROS Bag、HuggingFace Zarr/JPEG格式，并提供CLI工具；评测使用Evo、Umeyama对齐等标准指标。

**📊 数据集**

使用的数据集为GrandTour本身：49个任务、3–7分钟、覆盖数百米，包含室内、城市、山区、工业废墟等多样场景，并记录了日夜、雨雪、尘土等环境变化。

**📈 对比分析**

评测对比了52个开源方法（LO、LIO、LIVO、VIO、Multi‑LiDAR、RGB‑D等），在6个典型任务（SPX‑2、SNOW‑2、EIG‑1、CON‑4、ARC‑2、ARC‑7）上计算RTE/ATE，并按平均排名汇总。结果显示：LiDAR‑Inertial方法（如Coco‑LIC、FAST‑LIVO2）表现最佳，纯视觉或视觉‑惯性方法在光照或纹理稀缺环境下易失效；多激光融合方法性能受算法实现与环境影响较大。

**⚠️ 局限性**

局限性包括：①对同步与标定精度要求高，若硬件失配需人工修正；②GPS/总台覆盖受限，部分区域无法提供完整地面真值；③VIO与纯视觉方法对环境光照、纹理的依赖导致失效率高；④评测结果对参数调优高度敏感，跨任务迁移性差；⑤数据集仍缺乏大规模动态场景与多机器人协同的长时序记录。

---

## 200. A Geometric Probe of the Accuracy-Robustness Trade-off: Sharp Boundaries in Symmetry-Breaking Dimensional Expansion

**arXiv ID:** 2602.17948 | [PDF](https://arxiv.org/pdf/2602.17948v1)

**作者:** Yu Bai `[一作]` (Northwest Institute of Nuclear Technology), Jun-Jie Zhang `[通讯]` (Northwest Institute of Nuclear Technology)

**通讯引用:** 6873 | [OpenAlex ID](https://openalex.org/A5100343069)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了利用对称性破坏维度扩展(SBDE)对干净精度与对抗鲁棒性权衡的几何机制，并通过投影恢复鲁棒性。

**💡 创新点**

以SBDE作为可控几何探针，首次显式展现了损失表面在辅助维度上的陡峭梯度，阐明精度提升导致鲁棒性下降的几何原因，并提出投影技术验证鲁棒性缺失主要集中在辅助维度。

**🔧 技术方法**

采用SBDE输入扩展、梯度投影(Π)以及迭代白盒攻击(PGD、AutoAttack等)来探测损失梯度与鲁棒性。

**📊 数据集**

以CIFAR-10数据集为主，使用ResNet-18等模型进行实验。

**📈 对比分析**

与原始模型比较时，SBDE将干净精度从90.47%提升至95.63%，但未投影时鲁棒性急剧下降；投影后鲁棒性平均恢复至约85%，多种填充常数、扩展因子与卷积步幅的ablation验证结果保持一致。

**⚠️ 局限性**

该方法仅在实验设置下有效，投影只能在测试时使用，未能在训练阶段提升鲁棒性；对不同网络与数据集的泛化未知，且未解决鲁棒性本身，仅揭示几何关联。

---

## 201. Complexity lower bounds for succinct binary structures of bounded clique-width with restrictions

**arXiv ID:** 2602.18240 | [PDF](https://arxiv.org/pdf/2602.18240v1)

**作者:** Colin Geniet `[一作]` (Institute for Basic Science), Kévin Perrot `[通讯]` (Université publique)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文提出了一种适用于由电路压缩编码的二元结构的MSO可定义问题的Rice式复杂度下界，进一步允许多重二元关系并限制新符号的解释。

**💡 创新点**

创新点在于将bounded clique-width、union‑stable约束以及cw‑size‑independent性质统一起来，证明了在这些约束下问题可为NP/ coNP/ P‑hard或P‑完整；同时加强了对非平凡性参数化的必要性，并扩展了前人对Courcelle定理对照研究的范畴。

**🔧 技术方法**

技术手段包括MSO类型与clique‑width分解、泵浦引理、克里克分解的合并与idempotent重着色、构造可在对数空间内的电路，以及通过这些电路实现对数空间归约。

**📊 数据集**

论文为理论研究，不使用任何实验数据集或公开数据集。

**📈 对比分析**

通过理论归约与证明与先前的Rice定理和Courcelle定理结果对比，展示了在相应约束下的下界与完整性，未涉及实验性能评估。

**⚠️ 局限性**

局限性包括：对于非c.w.非平凡或非union‑stable约束的公式难以给出统一下界；硬度结果依赖于大小独立性或鲁棒性；对非树状图形（如仅有bounded twin‑width）的推广仍受限。

---

## 202. DEIG: Detail-Enhanced Instance Generation with Fine-Grained Semantic Control

**arXiv ID:** 2602.18282 | [PDF](https://arxiv.org/pdf/2602.18282v1)

**作者:** Shiyan Du `[一作]` (Sun Yat-sen University), Dongyu Zhang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 2801 | [OpenAlex ID](https://openalex.org/A5100629134)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DEIG 框架，实现细粒度、多属性、多实例图像生成。

**💡 创新点**

创新点在于：①实例细节提取器 (IDE) 将冻结的文本编码器输出压缩为实例感知语义；②细节融合模块 (DFM) 通过实例遮蔽注意力防止属性泄漏；③结合高质量、富含多属性的实例描述实现精准控制。

**🔧 技术方法**

技术手段：冻结 Flan‑T5‑XL 文本编码器 + 轻量级 IDE + DFM（包含广播嵌入与实例掩码自注意力）+ VLM 生成精细实例描述。

**📊 数据集**

数据集：基于 MS‑COCO 过滤得到的细粒度实例描述集；DEIG‑Bench（包含人像和物体的多属性场景）、MIG‑Bench、InstDiff‑Bench 等基准数据。

**📈 对比分析**

与 GLIGEN、MIGC、InstanceDiffusion、ROICtrl 等 SOTA 进行对比；在 DEIG‑Bench 上 MAA、mIoU 及属性准确率均显著提升；在 MIG‑Bench 与 InstDiff‑Bench 亦优于基线。

**⚠️ 局限性**

局限性：对材料与纹理等抽象属性的提升有限；实例遮蔽注意力在极度拥挤场景下可能略微降低空间精度；需要更丰富的人像多属性场景进一步验证。

---

## 203. Can AI Lower the Barrier to Cybersecurity? A Human-Centered Mixed-Methods Study of Novice CTF Learning

**arXiv ID:** 2602.18172 | [PDF](https://arxiv.org/pdf/2602.18172v1)

**作者:** Cathrin Schachner `[一作]` (University of Klagenfurt), Jasmin Wachter `[通讯]` (University of Klagenfurt)

**通讯引用:** 52 | [OpenAlex ID](https://openalex.org/A5082964903)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本研究以单个本科生为例，探讨了agentic AI框架Cybersecurity AI (CAI) 在降低CTF（Capture‑the‑Flag）入门门槛、促进新手学习与参与方面的作用；

**💡 创新点**

创新点在于：①从人本角度评估AI辅助学习，而非单纯性能评测；②结合定量指标与定性主题分析，揭示AI在认知负荷、策略探索、信心与身份认同等维度的影响；③提出“AI辅导者”而非“自动化工具”的新视角；

**🔧 技术方法**

使用的技术包括：agentic LLM‑driven CAI框架（集成ChatGPT/Claude LLM与传统渗透工具如Nmap、Burp等）；定量分析采用描述性统计和相对排名；定性分析采用反思日志与主题分析；

**📊 数据集**

数据集：①来自该本科生在CAI支持下完成的5个CTF挑战（Cyberleague和ACSC重现环境）；②由29名ACSC挑战者收集的性能数据（尝试次数、解题时间等），用于基线比较；

**📈 对比分析**

比较方法：将CAI受试者的解题率、尝试次数、每策略耗时等与ACSC挑战者的整体和各水平分组指标进行对比；结果显示CAI受试者在大部分挑战中的解题率与中级水平相当，且在策略探索和每策略耗时上表现优于多数对照组，表明AI辅助在战略层面加速学习；

**⚠️ 局限性**

局限性包括：①单一案例研究，受试者技术背景不具代表性；②仅评估CAI 0.5版，结果不一定推广到其他AI框架或LLM；③基线对照依赖自报数据，可能存在自选偏差；④缺乏长期跟踪，无法评估AI使用后长期技能保持与依赖风险。

---

## 204. SMaRT: Online Reusable Resource Assignment and an Application to Mediation in the Kenyan Judiciary

**arXiv ID:** 2602.18431 | [PDF](https://arxiv.org/pdf/2602.18431v1)

**作者:** Shafkat Farabi `[一作]`, Anja Sautmann `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于凸二次规划的调度模型，并通过线性近似和贝叶斯不确定性估计改进分配决策。

**💡 创新点**

创新点在于将调度问题转化为可求解的凸QP，证明其可行性与解的存在性；同时引入LP近似实现快速求解，并设计混合贝叶斯-显式更新的价值增益估计与周期性再校准。

**🔧 技术方法**

使用凸二次规划、线性规划近似、贝叶斯推断（高斯矩匹配）以及仿真模拟。

**📊 数据集**

实验使用司法机关真实案例数据以及模拟场景进行评估。

**📈 对比分析**

与原始QP、已知VA情况和不同学习算法进行对比，结果显示LP-approx在大多数指标上与QP相近，只有在极小lambda下解决率略低，高lambda下OCDM略高，学习算法在agreement、caseload和Gini指标上表现优异。

**⚠️ 局限性**

局限包括LP近似在极端参数下可能失真；贝叶斯后验通过高斯近似导致漂移，需要周期性再校准；再校准频率与计算开销的权衡；模型对负成本假设的依赖。

---

## 205. Leakage and Second-Order Dynamics Improve Hippocampal RNN Replay

**arXiv ID:** 2602.18401 | [PDF](https://arxiv.org/pdf/2602.18401v1)

**作者:** Josue Casco-Rodriguez `[一作]` (Rice University), Richard G. Baraniuk `[通讯]` (Rice University)

**通讯引用:** 51839 | [OpenAlex ID](https://openalex.org/A5072713767)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了在无输入时，噪声循环神经网络（RNN）通过 Langevin 动力学自发产生与任务时相似的回放序列，并通过理论与仿真探讨了泄漏（leakage）、适应（adaptation）以及非稳态（underdamped）动力学对回放质量与速度的影响。

**💡 创新点**

创新点在于：①证明即使是简单的高斯过程，RNN 的得分函数随时间非平稳，导致估计困难；②揭示隐藏状态泄漏可作为有益的线性先验，提高路径积分性能；③将适应解释为一种非理想的二阶 Langevin 采样；④提出显式的“无阻尼”动力学（Momentum）实现真正的 underdamped Langevin，从而在保持探索性的同时显著加速回放并实现时间压缩。

**🔧 技术方法**

技术方法包括：路径积分损失下的噪声 RNN 训练、Langevin 与 underdamped Langevin 动力学理论、适应反馈模型、Momentum 机制、以及 Wasserstein 距离、平均路径长度等量化指标来评估回放与任务时分布的一致性与探索性。

**📊 数据集**

使用的数据集为：二维三角形与 T‑maze 轨迹（模拟 2D OU 过程）、高维合成老鼠位置细胞（RatInABox 生成的地方细胞激活序列）、以及 1D Ornstein‑Uhlenbeck 过程的闭式解。

**📈 对比分析**

比较方法主要是计算无输入回放分布与有输入任务分布的 Wasserstein 距离、平均路径长度以及探索性指标。实验结果表明：适应会减慢回放速度并增加分布差距；Momentum（无阻尼）能够加速回放、压缩时间并在适应存在时降低 Wasserstein 距离，且不牺牲探索性。

**⚠️ 局限性**

局限性包括：理论推导假设能完美估计得分函数，实际训练中该估计仍不准确；实验仅在简化的 rate‑based RNN 与合成环境中验证，未涵盖脉冲网络；未探究更复杂环境下的泛化性。

---

## 206. AI-Wrapped: Participatory, Privacy-Preserving Measurement of Longitudinal LLM Use In-the-Wild

**arXiv ID:** 2602.18415 | [PDF](https://arxiv.org/pdf/2602.18415v1)

**作者:** Cathy Mengying Fang `[一作]` (MIT Media Lab), Pattie Maes `[通讯]` (MIT Media Lab)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 AI‑Wrapped 原型，收集用户在野外的 LLM 使用日志并生成隐私保护的交互式报告，提供使用统计、主题、风险与正面信号等。

**💡 创新点**

创新点在于实现了基于用户自愿导出的聊天记录的零数据保留、PII 剔除流程，生成参与者可视化报告，并通过双向对齐框架同时支持研究与用户反思。

**🔧 技术方法**

采用了 spaCy/Presidio 的 PII 检测、OpenAI API 生成洞察、Clio 风格趋势抽取、Web 前端展示等技术，并使用 Zero Data Retention 策略保障隐私。

**📊 数据集**

使用了 82 名美国成人在 2025 年 ChatGPT/Claude 的聊天历史，约 48,495 条会话，经过用户预检和删减后得到数据集。

**📈 对比分析**

虽然未与传统基准进行直接对比，但报告显示 73% 参与者被标记为过度依赖，62% 为完美主义倾向，用户反馈认为报告准确且具有洞察力，表明系统能有效捕捉高频风险信号。

**⚠️ 局限性**

局限性包括样本偏向年轻、受教育程度高且性别偏男性，数据收集受导出障碍和隐私担忧限制，分析仅为年度汇总且仅处理用户发言，缺乏时间序列与跨平台验证。

---

## 207. Simplifying Outcomes of Language Model Component Analyses with ELIA

**arXiv ID:** 2602.18262 | [PDF](https://arxiv.org/pdf/2602.18262v1)

**作者:** Aaron Louis Eidt `[一作]` (Technische Universitaet Berlin), Nils Feldhus `[通讯]` (Technische Universitaet Berlin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了 ELIA，一款交互式 Web 应用，通过归因分析、函数向量和电路追踪三种机制，并利用视觉‑语言模型自动生成自然语言解释，以让非专家也能直观理解大型语言模型的内部工作。

**💡 创新点**

创新点在于将 AI 生成的可读解释与多维可视化无缝结合，并实现自动化的可信度验证系统；同时通过用户研究证明该组合显著降低了专业门槛，非专家的理解与专家相当。

**🔧 技术方法**

技术包括 Streamlit 搭建前端、PyTorch + Transformers 处理模型、Plotly、Inseq 进行归因、PCA+3D 可视化、Faiss 近邻检索、Qwen2.5‑VL‑72B 生成解释和事实提取，另外使用程序化检查实现解释验证。

**📊 数据集**

数据集主要包括 OLMo‑2 的训练集 Dolma、功能向量训练所用的指令提示集（含多层次功能类型与子类别）以及用以相似文档检索的 Dolma Faiss 索引。

**📈 对比分析**

通过 Faithfulness Checker 与专家评估对比，解释的可信度平均达 90% 以上；在特征/路径消融实验中，ELIA 识别的电路对模型输出的影响显著大于随机基线，证明解释与模型行为高度一致；用户测试显示非专家的理解分数与专家相近，验证了方法的有效性。

**⚠️ 局限性**

局限性包括：仅支持英语和德语、仅解释 OLMo‑2 模型、使用单一 Qwen‑VL 模型进行解释与验证、缺乏更广泛的模型与语言覆盖、用户研究规模有限且仅为短期交互，未验证长期专业使用场景。

---

## 208. El Agente Gráfico: Structured Execution Graphs for Scientific Agents

**arXiv ID:** 2602.17902 | [PDF](https://arxiv.org/pdf/2602.17902v1)

**作者:** Jiaru Bai `[一作]`, Alán Aspuru-Guzik `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种单代理、类型安全的执行框架，利用图形代理（OGM）将科学状态序列化为知识图谱，实现对量子化学和材料设计工作流的可扩展、可追踪、并行化管理；

**💡 创新点**

核心创新在于将执行上下文从无结构文本转为结构化、类型化的对象图，解耦高层决策与低层计算，显著降低LLM上下文压力，并通过持久化知识图谱实现跨工具、跨会话的数据共享与验证；

**🔧 技术方法**

采用Pydantic+Pydantic‑AI实现类型安全代理，World Avatar OGM实现Python类与RDF的双向映射，GPU4PySCF实现GPU加速量子计算，CREST+QCG用于构型采样，PORMAKE+MACE/MLIP用于MOF构建与优化，GraphChat前端展示实时轨迹；

**📊 数据集**

使用公开量子化学基准任务（如有机/无机分子能量、氢抽提、环烷烃应变、pKa预测、TDDFT激发能）、构型搜索实验、MOF结构库（CoRE‑MOF）、以及手工构造的显式溶剂体系；

**📈 对比分析**

与之前的多代理系统相比，单代理框架将token使用量降至约1/14（从160万降至约10万），成本从$4.67降至$0.17，执行时间从1827 s降至200‑300 s，且在数值准确性（pass@3≈0.99）与LLM评判（>0.90）上保持或提升；

**⚠️ 局限性**

局限包括：对LLM的依赖仍需要人工验证代码与结果，当前设计为单会话单代理，难以扩展为跨代理长时序协作；OGM与知识图谱的实时同步与并发管理尚未完善；GPU资源调度仍手动配置，跨节点分布式扩展受限；

---

## 209. Visual Interface Workflow Management System Strengthening Data Integrity and Project Tracking in Complex Processes

**arXiv ID:** 2602.17668 | [PDF](https://arxiv.org/pdf/2602.17668v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 210. How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf

**arXiv ID:** 2602.18397 | [PDF](https://arxiv.org/pdf/2602.18397v1)

**作者:** Wenqi Jiang `[一作]` (NVIDIA Research), Christos Kozyrakis `[通讯]` (NVIDIA Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Vision‑Language‑Action（VLA）模型的推理性能进行系统性分析，提出基于屋顶线的性能模型，评估多种模型尺寸、长上下文、扩散与自回归、异步和双系统推理以及不同部署位置对实时推理的影响，并给出15条实用指导。

**💡 创新点**

1) 首次构建面向VLA的屋顶线分析工具（VLA‑perf）；2) 对模型与系统组合的完整性能景观进行首次量化；3) 提供针对不同硬件与网络环境的部署最佳实践。

**🔧 技术方法**

屋顶线性能模型、CUDA/Trident优化、量化与分块、动作分块、异步推理、双系统（系统1/系统2）推理、网络传输延迟建模。

**📊 数据集**

使用UR5e双臂机器人三摄像头（224×224图像、256视觉token/摄像头）与约32个语言token的π_0模型数据，动作空间14维。

**📈 对比分析**

在多种GPU（Jetson Thor、RTX4090、A100、H100、B100）与网络（1GbE、10GbE、WiFi、4G/5G）上对π_0及其变体进行推理延迟与吞吐率评测；结果显示数据中心GPU可达100Hz吞吐，Edge GPU低于10Hz，异步推理在慢速无线网络下可提升2.6–6倍吞吐。

**⚠️ 局限性**

仅覆盖机器人操纵任务；未考虑机器人执行、传感器延迟和多线程并发；评估仅基于单推理；未涵盖自动驾驶、无人机等其他具体现实场景。

---

## 211. Self-Aware Object Detection via Degradation Manifolds

**arXiv ID:** 2602.18394 | [PDF](https://arxiv.org/pdf/2602.18394v1)

**作者:** Stefan Becker `[一作]`, Michael Arens `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一个降解感知自我感知目标检测框架，通过在检测器的多层特征上学习降解流形，实现对图像降解的几何表征。

**💡 创新点**

创新点在于利用无标签的多层对比学习构建降解流形，并以洁净原型为参考，从几何距离直接评估输入质量，无需显式置信度或先验。

**🔧 技术方法**

技术包括轻量级嵌入头、多层特征读取、注意力池化、硬负样本挖掘和对比损失，以及原型移动平均。

**📊 数据集**

实验数据集涵盖COCO、KITTI、VISDRONE、DETRAC、UAVDT、FLIR、Seeing Through Fog、BDD等，并使用Synthetic Corruptions。

**📈 对比分析**

与基线比较（检测器不确定性、Normalizing Flow、IQA模型）后，降解流形在AUROC上获得 88–97% 的分数，明显优于其他方法。

**⚠️ 局限性**

局限性包括仅关注图像形成降解，未覆盖语义分布漂移，无法单实例预测检测准确度，且对特征融合方案敏感。

---

## 212. Graph-Neural Multi-Agent Coordination for Distributed Access-Point Selection in Cell-Free Massive MIMO

**arXiv ID:** 2602.17954 | [PDF](https://arxiv.org/pdf/2602.17954v1)

**作者:** Mohammad Zangooei `[一作]` (University of Waterloo), Raouf Boutaba `[通讯]` (University of Waterloo)

**通讯引用:** 27050 | [OpenAlex ID](https://openalex.org/A5038723583)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文设计了一种基于图神经网络的分布式多智能体学习框架 APS‑GNN，用于解决大规模 CFmMIMO 系统中的 AP 选择问题。

**💡 创新点**

创新点在于将 AP‑UE 连接拆分为独立智能体，利用 GNN 进行局部信息交互，并采用约束强化学习与 Lagrangian 机制保证 SE 目标，同时通过监督预训练加速学习。

**🔧 技术方法**

使用的技术包括多智能体强化学习（MAPPO‑Lagrangian）、图神经网络（Attention‑based GCN）、循环神经网络（GRU）和监督式模仿学习。

**📊 数据集**

实验使用自研的 CFmMIMO 仿真器，随机部署 AP 与 UE，并结合 LoS 信道、移动模型等生成的实时数据集。

**📈 对比分析**

与 k‑Strongest、单智能体 PPO‑Lagr 和集中式 MAT‑Lagr 对比，APS‑GNN 在满足 SE 目标的前提下激活的 AP 数量减少 50–70%，并且推理延迟比集中式方法低 1–2 个数量级。

**⚠️ 局限性**

局限性包括对实时 CSI 误差和网络拓扑变化的鲁棒性尚未验证，且在极大规模网络下仍需进一步压缩通信和计算开销。

---

## 213. Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning

**arXiv ID:** 2602.18117 | [PDF](https://arxiv.org/pdf/2602.18117v1)

**作者:** Yongjae Shin `[一作]` (KAIST), Youngchul Sung `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

提出了FINO方法，通过在离线预训练阶段对流匹配目标注入噪声，扩展动作空间并在在线微调阶段采用熵引导采样实现高效探索；

**💡 创新点**

创新点在于：1）在流匹配训练中注入噪声以保留数据均值同时增加方差，显著提升探索；2）使用熵引导的采样机制动态平衡探索与利用；

**🔧 技术方法**

使用技术包括：流匹配（Flow Matching）作为生成模型的策略；噪声注入的训练目标；熵估计与熵引导采样；一阶策略蒸馏与价值最大化；

**📊 数据集**

数据集涵盖45个任务，来自OGBench和D4RL两个基准；

**📈 对比分析**

与ReBRAC、Cal-QL、RLPD、IFQL、FQL等基线在相同离线+在线训练配置下比较，FINO在有限的在线样本预算内 consistently outperform 其他方法，取得最优或接近最优性能；

**⚠️ 局限性**

局限性包括：对噪声注入参数（η）和采样规模（N_sample）敏感；在高维动作空间中仍需更多候选样本；计算成本略高于基础FQL，尤其在熵估计与多候选采样时。

---

## 214. FENCE: A Financial and Multimodal Jailbreak Detection Dataset

**arXiv ID:** 2602.18154 | [PDF](https://arxiv.org/pdf/2602.18154v1)

**作者:** Mirae Kim `[一作]` (Kakaobank), Youngjun Kwak `[通讯]` (Kakaobank)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并公开一个面向金融领域的双语（韩英）图文对齐的安全性对抗数据集 FENCE，并评估其在多模态语言模型中的拒问安全性能。

**💡 创新点**

创新点在于：①聚焦图像驱动的 jailbreak（IA）而非文本为主；②提供 50:50 的安全/不安全样本平衡；③利用 GPT‑4o 两步生成恶意问句并验证；④覆盖 15+ 真实金融场景；⑤以此数据集进行细粒度对比实验。

**🔧 技术方法**

使用 GPT‑4o 进行恶意问句生成、图像检索（Pixabay）、Pillow 合成；对模型进行攻击成功率（ASR）和拒问成功率（DSR）评估，并与多模态安全基线（LlamaGuard、Qwen、PaliGemma）对比。

**📊 数据集**

主要使用 FENCE 自己构建的数据集，以及 JailBreakV‑28K、FigStep、HADES、MM‑SafetyBench 作为基准进行交叉验证。

**📈 对比分析**

实验显示在 FENCE 上攻击成功率普遍高于其他基准；在对比微调后，Qwen‑2.5‑VL 3B 的 DSR 从 66% 提升至 99%，并在四大公开基准上实现 99% 的防御成功率，表明 FENCE 可显著提升金融多模态模型的安全性。

**⚠️ 局限性**

局限包括数据规模相对有限、仅覆盖韩英两种语言、对抗样本主要由 GPT‑4o 生成缺乏真实用户行为、多样性不足、未覆盖所有模型和训练范式。

---

## 215. Analyzing LLM Instruction Optimization for Tabular Fact Verification

**arXiv ID:** 2602.17937 | [PDF](https://arxiv.org/pdf/2602.17937v1)

**作者:** Xiaotang Du `[一作]` (University of Edinburgh), Emily Allaway `[通讯]` (University of Edinburgh)

**通讯引用:** 208 | [OpenAlex ID](https://openalex.org/A5008920438)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在表格事实验证任务中使用DSPy框架进行指令优化，比较了直接预测、Chain-of-Thought、ReAct与CodeAct等四种提示技术在不同LLM（Qwen3、Gemma3、GPT‑4o）上的表现。

**💡 创新点**

首次系统评估了三种指令优化器（COPRO、MiPROv2、SIMBA）对语言推理与工具调用行为的影响，并发现MiPROv2在CoT推理中带来最显著提升，SIMBA则在ReAct与CodeAct的工具使用与直接推理路径上表现最佳。

**🔧 技术方法**

采用DSPy的指令搜索与元推理技术，对提示语进行自动调整；对ReAct使用SQL工具，对CodeAct使用可执行Python代码；通过对四个数据集的交叉验证评估指令优化效果。

**📊 数据集**

TabFact、PubHealthTab、SciTab、MMSci四个表格事实验证基准，覆盖通用与领域特定场景，并使用表格文本化形式进行统一评测。

**📈 对比分析**

在Qwen3和Gemma3上实验，指令优化整体提升准确率和宏F1，MiPROv2在CoT上最稳定提高，SIMBA在ReAct与CodeAct上效果突出；ReAct在大模型下可与手工设计提示相当但需优化，CoT在小模型下表现仍更好；GPT‑4o基线已高，对优化收益相对有限。

**⚠️ 局限性**

实验仅覆盖两种模型族与两种规模，优化器仅限三种；ReAct仅使用单一SQL工具；未考察多工具交互与不同初始指令对优化的影响；结果对更大规模或其他架构的迁移性仍未知。

---

## 216. Design and Characterization of a Dual-DOF Soft Shoulder Exosuit with Volume-Optimized Pneumatic Actuator

**arXiv ID:** 2602.18212 | [PDF](https://arxiv.org/pdf/2602.18212v1)

**作者:** Rui Chen `[一作]` (Scuola Superiore Sant'Anna), Antonio Frisoli `[通讯]` (Scuola Superiore Sant'Anna)

**通讯引用:** 8385 | [OpenAlex ID](https://openalex.org/A5090204404)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并验证了一款双自由度软肩部外骨骼，采用体积优化的旋转形状气动执行器，并集成弯曲收束执行器和袋式横向屈伸执行器，完成了肩部屈伸、外展和水平屈伸的多模式辅助。

**💡 创新点**

创新点在于引入自适应交叉截面SSAA气动执行器，实现35.7%体积减小、94.2%扭矩保留和35.2%响应速度提升，并将其转化为曲线外展执行器和袋式横向执行器，构建轻量化（390 g）可穿戴双自由度系统。

**🔧 技术方法**

采用TPU涂层织物拼接、热封、压缩式袋式肌肉和比例压力调节器配合微控制器的实时压力控制，结合多轴力学分析与频率响应模型，完成了气动执行器的结构优化与运动学控制。

**📊 数据集**

在十名健康志愿者上收集EMG信号、运动轨迹与主观评分数据，使用MVC标准化的肌电特征评估，进行统计检验；未使用公开数据集。

**📈 对比分析**

通过与未穿戴或单一执行器对比，使用Wilcoxon检验比较EMG下降幅度；结果显示外展可实现最高59%肌电下降，屈伸可达63.7%，但双执行器对屈伸的增益仅对胸大肌显著；系统重量390 g，体积显著低于现有1 L级设计。

**⚠️ 局限性**

研究仅在健康受试者中验证，缺乏对运动障碍患者的评估；体积优化空间未完全探索，耐久性超2000循环未知；多执行器间的耦合与控制策略对实际临床效果影响待进一步研究。

---

## 217. Comparative study of different quadrature methods for cut elements

**arXiv ID:** 2602.18130 | [PDF](https://arxiv.org/pdf/2602.18130v1)

**作者:** Michael Loibl `[一作]`, Benjamin Marussig `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对现有开源的切割单元积分方法进行综述，并通过统一的 Matlab 基准环境对 10 个代码在二维与三维案例中进行准确性、鲁棒性、灵活性和效率比较。

**💡 创新点**

提出标准化基准测试集和可直接运行的统一接口，方便未来方法的评估与对比；系统分析了输入参数（期望积分阶数）对实际误差的影响。

**🔧 技术方法**

采用基于背景网格的切割有限元（CutFEM/FCM/Unfitted FEM）积分技术，如四叉树/八叉树细分、矩 moments fitting、元素重参数化、维度降等；使用 Matlab 实现统一接口，对不同积分规则进行对比。

**📊 数据集**

使用多种二维三维几何例子（包含隐式 level‑set、B‑spline、NURBS 等边界描述）作为测试案例；未涉及体素数据但已提及其可能性。

**📈 对比分析**

通过计算积分误差、运行时间和对极小切割单元的稳定性，对 10 个开源代码在相同网格与边界描述下进行对比，结果显示不同方法在准确性与效率上存在显著差异；大多数方法在高阶数下仍保持高精度，但细分策略的耗时较高。

**⚠️ 局限性**

局限在于仅评估了已公开的 10 个实现，对其它未收录或私有库缺乏考察；实验未覆盖体素表示和复杂三维曲面；对极小切割单元的鲁棒性分析仍有限。

---

## 218. DeCEAT: Decoding Carbon Emissions for AI-driven Software Testing

**arXiv ID:** 2602.18012 | [PDF](https://arxiv.org/pdf/2602.18012v1)

**作者:** Pragati Kumari `[一作]` (University of Calgary), Novarun Deb `[通讯]` (University of Calgary)

**通讯引用:** 157 | [OpenAlex ID](https://openalex.org/A5072598400)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DeCEAT 框架，系统评估小语言模型在自动化单元测试生成过程中的能耗、碳排放与测试质量。

**💡 创新点**

将碳强度、效率、覆盖率等指标融合为多维度可解释的可持续性度量，并通过分层提示与量化模型探究提示结构对碳足迹的影响。

**🔧 技术方法**

采用量化的小语言模型（4‑bit/8‑bit），Anthropic 样式提示工程，CodeCarbon 能耗追踪，Python HumanEval 基准及覆盖率工具。

**📊 数据集**

使用 OpenAI HumanEval 数据集（164 个 Python 编码任务）。

**📈 对比分析**

在不同模型/提示组合下进行批量推理，收集能耗、碳排放、时间和覆盖率，并计算 SCI、SEI、CER、SI、GQI、SCV、SVI、GFβ 等指标；结果表明模型在能耗、覆盖率和稳定性上各有优势，提示越结构化越能降低碳强度。

**⚠️ 局限性**

实验仅在单一硬件（Colab T4）和单一基准上进行，碳强度采用平均值，覆盖率仅衡量代码覆盖，未考虑不同地区电网、不同硬件或更复杂的测试场景。

---

## 219. Uncertainty-Aware Jamming Mitigation with Active RIS: A Robust Stackelberg Game Approach

**arXiv ID:** 2602.18165 | [PDF](https://arxiv.org/pdf/2602.18165v1)

**作者:** Xiao Tang `[一作]` (Xi'an Jiaotong University), Zhu Han `[通讯]` (University of Houston)

**通讯引用:** 88745 | [OpenAlex ID](https://openalex.org/A5063667378)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种基于主动RIS的鲁棒Stackelberg博弈框架，用于在存在信道不确定性的自适应干扰下保护合法通信。

**💡 创新点**

创新点在于将主动RIS与鲁棒Stackelberg博弈相结合，首次通过分块SCA+BSUM方法联合优化功率、波束和反射，且在极端干扰情况下实现最优平衡。

**🔧 技术方法**

所采用的技术包括Stackelberg博弈模型、S-Procedure与矩阵不等式求解不确定信道约束、SCA和BSUM分块优化。

**📊 数据集**

实验使用仿真生成的Rayleigh/Rician衰落信道数据，设定不同位置、功率、RIS元素数等参数进行性能评估。

**📈 对比分析**

通过与无不确定性、非鲁棒方案及无RIS基线进行对比，实验表明鲁棒方案在各种功率、RIS尺寸和干扰器位置下均能获得更高的合法方收益并抑制干扰器收益。

**⚠️ 局限性**

局限性包括只考虑单源-单目的三方模型，未考虑多用户、多干扰器以及硬件非理想效应等实际系统复杂情况。

---

## 220. QPTAS for MWIS and finding large sparse induced subgraphs in graphs with few independent long holes

**arXiv ID:** 2602.18317 | [PDF](https://arxiv.org/pdf/2602.18317v1)

**作者:** Édouard Bonnet `[一作]` (University of Lyon), Paweł Rzążewski `[通讯]` (Warsaw University of Technology)

**通讯引用:** 549 | [OpenAlex ID](https://openalex.org/A5047941027)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了在sC_t诱导极小图自由图中求最大独立集的 QPTAS，并将该技术推广到包含树宽限制及可在 Counting Monadic Second Order Logic 表达的 hereditary 性质的（≤r,ψ）诱导子图最大化问题；

**💡 创新点**

核心创新在于将重顶点枚举、BFS 层级分割与 blob 图构造相结合，利用弱超疏性类的性质，首次在更广泛的诱导极小图自由类中实现 QPTAS，并把结果推广到可在 MSO₂ 中定义的诱导子图最大化问题；

**🔧 技术方法**

采用的主要技术包括递归分支与 γ‑重顶点枚举、最短长环检测、BFS 层级分析、图的 blob 图构造、弱超疏性类的近似分割以及相关定理的证明与组合；

**📊 数据集**

该工作为纯理论算法，不依赖任何实验数据集；

**📈 对比分析**

与此前仅在部分特殊图类（如 P_t‑free 或 C_≥t‑free）已知的 QPTAS 相比，本文扩展到所有 sC_t 诱导极小图自由类，运行时间为 n^{O(log^5 n/ε^3)} 的准多项式；在理论复杂度上已大幅改进；

**⚠️ 局限性**

主要局限在于仍仅提供 QPTAS，运行时间仍为准多项式；对 ψ 的 hereditary 与闭合性质有限制；未能给出多项式时间解；对更一般的诱导极小图类（如非线性森林组件）仍未实现；

---

## 221. Breaking the Correlation Plateau: On the Optimization and Capacity Limits of Attention-Based Regressors

**arXiv ID:** 2602.17898 | [PDF](https://arxiv.org/pdf/2602.17898v1)

**作者:** Jingquan Yan `[一作]` (University of Texas at Arlington), Junzhou Huang `[通讯]` (University of Texas at Arlington)

**通讯引用:** 24483 | [OpenAlex ID](https://openalex.org/A5068865316)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对注意力回归模型联合MSE+PCC损失训练时出现的PCC平台现象进行了首次严谨的理论分析，并基于该分析提出了Extrapolative Correlation Attention（ECA）框架来打破PCC平台；

**💡 创新点**

在理论上揭示了MSE优化导致的PCC梯度衰减冲突与软max注意力的凸合并能力上限，并设计了三项新机制——Scaled Residual Aggregation、Dispersion‑Aware Temperature Softmax和Dispersion‑Normalized PCC Loss；

**🔧 技术方法**

采用联合MSE+PCC损失、软max注意力、可自适应温度softmax、残差缩放聚合及标准差归一化PCC损失等技术，并在Transformer‑style模型中实现ECA；

**📊 数据集**

在四类数据集上验证：可控均质性的Synthetic回归数据、UCI三大tabular回归数据集（Appliance Energy、Online News Popularity、Superconductivity）、10xProteomic空间转录组图像数据以及MOSI多模态情感分析数据；

**📈 对比分析**

通过与标准softmax注意力及多种state‑of‑the‑art基线对比，ECA在PCC上提升约10–20%，同时MSE/MAE保持不变或略降，尤其在高均质性数据中显著突破PCC平台；

**⚠️ 局限性**

主要限制在于对凸合并假设的依赖、γ尺度可导致过拟合、对非均质性数据的适应性仍待进一步验证；

---

## 222. PenTiDef: Enhancing Privacy and Robustness in Decentralized Federated Intrusion Detection Systems against Poisoning Attacks

**arXiv ID:** 2602.17973 | [PDF](https://arxiv.org/pdf/2602.17973v1)

**作者:** Phan The Duy `[一作]` (University of Information Technology), Van-Hau Pham `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了去中心化联邦学习IDS的防御框架PenTiDef，旨在同时保障隐私和抵御投毒攻击；

**💡 创新点**

创新点包括：将分布式差分隐私与潜在空间表示（AutoEncoder压缩的penultimate层）结合，用CKA与KMeans进行无监督异常检测，并通过区块链实现无中心服务器的去中心化协调与可信记录；

**🔧 技术方法**

采用技术有：分布式联邦学习、分布式差分隐私（DDP）、AutoEncoder、Centered Kernel Alignment (CKA)、KMeans聚类、FedAvg聚合、Hyperledger Fabric区块链与IPFS存储；

**📊 数据集**

使用数据集：CIC‑IDS2018与Edge‑IIoTSet；

**📈 对比分析**

通过与FLARE、FedCC等现有防御方法在IID与非IID数据、10%/20%/40%攻击者比例及多种投毒攻击（label‑flipping、weight‑scaling、Krum、backdoor、GAN）下进行对比，PenTiDef在准确率、精确率、召回率、F1分数上均优于对比方法，且训练时间更短，能在高达40%攻击者比例下保持良好性能；

**⚠️ 局限性**

限制：仅实现二分类，无法处理多类别或多标签IDS；对更隐蔽的投毒攻击（如语义后门、模型逆向）尚未评估；DDP噪声可能被自由骑手滥用；区块链交易开销与可扩展性需进一步研究。

---

## 223. Dynamic Deception: When Pedestrians Team Up to Fool Autonomous Cars

**arXiv ID:** 2602.18079 | [PDF](https://arxiv.org/pdf/2602.18079v1)

**作者:** Masoud Jamshidiyan Tehrani `[一作]` (Università della Svizzera italiana), Paolo Tonella `[通讯]` (Università della Svizzera italiana)

**通讯引用:** 11142 | [OpenAlex ID](https://openalex.org/A5025438762)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种系统级攻击，利用携带对抗性补丁的行人（T‑shirt 上印有隐蔽图案）协同移动，诱导自动驾驶车辆误将其识别为停止标志并停驶。

**💡 创新点**

创新点包括：① 通过多行人协同形成更大有效补丁，克服单个补丁尺寸有限的问题；② 引入行人动态运动，使补丁在摄像机视野中持续存在；③ 采用红色山茶花图案嵌入补丁，实现高度隐蔽性。

**🔧 技术方法**

技术手段：基于 ART 框架的梯度对抗补丁生成（Expectation over Transformation）；CARLA 仿真平台；Simlingo 自动驾驶代理；YOLOv5 作为 surrogate 模型。

**📊 数据集**

使用从 CARLA 采集的 1,517 张图像（覆盖 5 个城镇，80% 训练/20% 验证）构建训练数据集。

**📈 对比分析**

评估方法：模型层面计算 Stop‑Sign‑to‑Pedestrian (STP) 比例；系统层面计算 Attack Success Rate (ASR)。实验结果显示：单人攻击 ASR=0；两人动态协同攻击 ASR=50%；静态协同攻击 ASR=0。通过 Fisher’s exact test 进行显著性检验，表明动态协同攻击显著优于单人或静态攻击。

**⚠️ 局限性**

局限性：仅在 CARLA 仿真环境验证；受限于使用的 Simlingo 代理；仅针对停止标志的攻击；对行人协同与持续运动的实际可行性尚待进一步验证。

---

## 224. Tighter Regret Lower Bound for Gaussian Process Bandits with Squared Exponential Kernel in Hypersphere

**arXiv ID:** 2602.17940 | [PDF](https://arxiv.org/pdf/2602.17940v1)

**作者:** Shogo Iwazaki `[一作]` `[通讯]` (LY Corporation), Shogo Iwazaki (LY Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

研究了高斯过程（GP）带子问题的算法无关的最坏情况下界，特别关注于平方指数（SE）核，并在超球面输入域下部分解决了上界和下界之间的维度依赖对数因子的差距。

**💡 创新点**

提出了在超球面输入域下，任何算法的累积遗憾为Ω(√(T (ln T)^d (lnln T)^-d))的改进下界，并提供了O((ln T)^(d+1)(lnln T)^-d)的最大信息增益的上界。

**🔧 技术方法**

使用了Mercer表示定理和球谐理论构建新的困难函数类，并通过这些技术推导出新的下界和上界。

**📊 数据集**

研究中没有提到具体的数据集，而是集中在理论分析和算法性能的下界和上界。

**📈 对比分析**

通过比较现有的上界和下界，证明了在超球面输入域下，现有最佳算法的最优性，填补了现有结果中的(ln T)^(d/4)的差距。

**⚠️ 局限性**

结果的局限性在于，无法对一般紧致输入域声称相同的改进下界，但在超球面域下的结果仍然具有重要价值。

---

## 225. Distributed Triangle Enumeration in Hypergraphs

**arXiv ID:** 2602.17834 | [PDF](https://arxiv.org/pdf/2602.17834v1)

**作者:** Duncan Adamson `[一作]` (University of St Andrews), Paul G. Spirakis `[通讯]` (University of Liverpool)

**通讯引用:** 7845 | [OpenAlex ID](https://openalex.org/A5011756177)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文系统研究了超图中的分布式子超图枚举问题，提出了多种超图的计算模型，并在这些模型中设计了分布式三角形枚举算法。

**💡 创新点**

创新点在于引入了多种超图的计算模型，扩展了传统图的CONGEST模型，并在此基础上证明了三角形枚举算法的最优性。

**🔧 技术方法**

使用了多种计算模型，包括PRIMAL CONGEST、EDGE CLIQUE、EDGE BROADCAST等，分析了它们的相对计算能力。

**📊 数据集**

使用了不同的超图数据集，特别是稀疏超图和“无处不稀疏”的超图，进行三角形枚举的算法设计和性能分析。

**📈 对比分析**

通过与现有的分布式算法进行比较，证明了在不同模型下的算法性能，展示了在CLIQU和PC模型中三角形枚举的时间复杂度为O(n^(r-5/3)/log n)，并且在稀疏超图中可以在O(n)轮内完成。

**⚠️ 局限性**

限制在于对于高阶交互的建模能力不足，且在某些情况下，算法的实际应用可能受到超图秩的影响，导致性能不够理想。

---

## 226. Fair Orientations: Proportionality and Equitability

**arXiv ID:** 2602.18098 | [PDF](https://arxiv.org/pdf/2602.18098v1)

**作者:** Ankang Sun `[一作]` (Shandong University), Bo Li `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 42874 | [OpenAlex ID](https://openalex.org/A5100374493)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在满足相关性约束的个体化物品分配问题中，研究了比例、公平与其松弛条件（PROP1、PROPX、EQ、EQ1、EQX、EF1）的可行性与算法复杂度。

**💡 创新点**

提出了基于物品相关性的精细比例份额定义，并系统阐述了在图/多重图模型下各公平概念的存在性与可计算性，首次给出PROPX、EQ1/EQX、EF1（针对劳动）的NP‑完整性结果。

**🔧 技术方法**

主要采用归约（从2P2N‑3SAT、Partition等经典NP难问题）、匹配与图论技术、分数Pareto最优（fPO）与逼近算法、以及多边匹配的构造与判定。

**📊 数据集**

论文未使用公开数据集，而是构造理论图实例（变量/子句工具图）来证明存在性与复杂度。

**📈 对比分析**

通过多种算法（匹配求解、分数化约、贪心分配）实现可行解的多项式构造；对可解情况给出多项式时间算法，对不可解情况给出NP‑完整性证明，展示了问题在可解与不可解之间的性能差异。

**⚠️ 局限性**

局限在于：对许多公平概念（PROPX、EQ1/EQX、EF1（ chores））只能给出NP‑完整性，缺乏多项式或多项式近似算法；仅在简单图或多重图下给出完整结果，未覆盖更一般的相关性模型；且未给出实验评估，仅在理论上证明存在性与复杂度。

---

## 227. Assessing LLM Response Quality in the Context of Technology-Facilitated Abuse

**arXiv ID:** 2602.17672 | [PDF](https://arxiv.org/pdf/2602.17672v1)

**作者:** Vijay Prakash `[一作]` (New York University), Danny Yuxing Huang `[通讯]` (New York University)

**通讯引用:** 1147 | [OpenAlex ID](https://openalex.org/A5042935400)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对四种大型语言模型（GPT‑4o、Claude‑3.7、Ruth、Aimee）在技术促进虐待相关问题上的零样本回答质量进行专家评估与受害者可操作性调查

**💡 创新点**

首次构建TFA受害者问题语料库、设计专门的评估框架（准确性、完整性、安全性、可操作性）并系统阐述LLM在此领域的典型错误模式

**🔧 技术方法**

利用基于提示的零样本生成、人工标注评估、聚类分析与主题编码等方法评估模型输出，并通过受害者问卷量化可操作性

**📊 数据集**

收集自文献、Reddit、Quora的431条TFA相关问题，经过筛选后得到约300条非对抗性问题，伴随LLM生成的回复及专家评分，公开数据集链接在OSF

**📈 对比分析**

与四模型对比发现，所有模型在准确性、完整性和安全性上均存在高达86%不完善率，Ruth在安全性上略优；受害者评估显示Claude与Aimee的可操作性略高于GPT和Ruth，但整体仍需改进

**⚠️ 局限性**

研究受限于零样本提示设计、对抗性问题未充分处理、仅评估当前模型版本、受害者样本有限、人工评估主观性及未覆盖推理型LLM的表现

---

## 228. Generated Reality: Human-centric World Simulation using Interactive Video Generation with Hand and Camera Control

**arXiv ID:** 2602.18422 | [PDF](https://arxiv.org/pdf/2602.18422v1)

**作者:** Linxi Xie `[一作]` (Stanford University), Gordon Wetzstein `[通讯]` (Stanford University)

**通讯引用:** 18259 | [OpenAlex ID](https://openalex.org/A5014044649)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究开发了一个人类中心的视频生成模型，能够根据实时跟踪的头部和关节级手部姿态，生成第一人称视角的可交互虚拟环境，实现了生成现实（Generated Reality）系统。

**💡 创新点**

创新点在于提出混合 2D–3D 手部姿态条件策略（将 ControlNet 风格的 2D 骨架视频与 3D 关节角度进行 token 加法融合），以及将双向教师模型蒸馏为实时自回归学生模型，显著提升了交互控制的细粒度与视频质量。

**🔧 技术方法**

使用了 Diffusion Transformer（DiT）视频扩散模型、ControlNet、AdaLN、跨注意力机制、迭代编码器训练、以及 Unity 与 Meta Quest 3 的实时生成管线。

**📊 数据集**

主要训练和评估数据集为 HOT3D（手部–对象交互的 3D 轨迹与摄像机姿态），并在 GigaHands 上验证泛化性能。

**📈 对比分析**

与仅使用头部条件（CameraCtrl）或仅使用手部条件（HandCtrl）的基线比较，系统在手部姿态准确率、摄像机姿态误差、以及视频质量指标（PSNR、LPIPS、FVD）上均优于基线；在用户实验中任务完成率从 3% 提升至 71%，且用户对控制感知得分显著提高。

**⚠️ 局限性**

主要局限包括分辨率低、延迟（1.4 s）和漂移导致的画质衰减，缺乏眼部水平的立体渲染以及低功耗可穿戴设备的部署能力。

---

## 229. Variational Distributional Neuron

**arXiv ID:** 2602.18250 | [PDF](https://arxiv.org/pdf/2602.18250v1)

**作者:** Yves Ruffenach `[一作]` `[通讯]` (Conservatoire National des Arts et Métiers), Yves Ruffenach (Conservatoire National des Arts et Métiers)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于1D VAE的变分分布式神经元（EVE），将单元层面的推断与不确定性显式化并可观测；

**💡 创新点**

核心创新在于把每个神经元当作独立的变分推断单元，配备局部ELBO、KL约束、能量阈值以及可调的自稳控制和自回归微观动态；

**🔧 技术方法**

使用变分自编码器、重参数化采样、KL正则化、带阈值的预算约束以及可选的AR(1)单元动态；

**📊 数据集**

在LongHorizon基准上四个数据集（ECL、TrafficL、Weather、Exchange）进行实验；

**📈 对比分析**

与无变分的确定性基线对比，EVE在某些数据集（Weather、Exchange）提升了内部稳定性和鲁棒性，虽然总体MSE并未系统性超越确定性模型；

**⚠️ 局限性**

局限性包括：k=1的容量受限、对超参数（预算、β、AR权重）敏感、实验未覆盖更大规模或多维隐变量、未与最新最先进模型做直接对比。

---

## 230. Financial time series augmentation using transformer based GAN architecture

**arXiv ID:** 2602.17865 | [PDF](https://arxiv.org/pdf/2602.17865v1)

**作者:** Andrzej Podobiński `[一作]` (Warsaw University of Technology), Jarosław A. Chudziak `[通讯]` (Warsaw University of Technology)

**通讯引用:** 84 | [OpenAlex ID](https://openalex.org/A5008057050)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

利用Transformer‑based GAN（TTS‑GAN）对金融时间序列进行数据增强，并用增强后的数据训练LSTM进行价格预测。

**💡 创新点**

在TTS‑GAN中引入PyTorch高效实现与简化梯度惩罚实现稳定收敛，并提出结合DTW与DeD‑iMs的DTW DeD‑iMs收敛指标，用以监控生成数据的时序与分布质量。

**🔧 技术方法**

采用Transformer‑based GAN、LSTM、DTW、Deep Dataset Dissimilarity Measure (DeD‑iMs)以及简化梯度惩罚技术。

**📊 数据集**

使用比特币（BTC）和标准普尔500指数（S&P500）的日收盘价数据，共40个不重叠时间窗口，序列长度分别为90和120。

**📈 对比分析**

对比基线（仅用真实数据训练LSTM）与增强版（真实+合成数据训练LSTM）的MSE，平均提升约0.04–0.13（p<0.01），表明数据增强显著降低预测误差。

**⚠️ 局限性**

仅验证了LSTM模型，未探讨更复杂网络；合成数据的多变量特性和长期依赖性仍待进一步评估；实验仅限于两种资产，需扩展至更多市场。

---

## 231. StableAML: Machine Learning for Behavioral Wallet Detection in Stablecoin Anti-Money Laundering on Ethereum

**arXiv ID:** 2602.17842 | [PDF](https://arxiv.org/pdf/2602.17842v1)

**作者:** Luciano Juvinski `[一作]` (Duke University), Alessio Brini `[通讯]` (Duke University)

**通讯引用:** 71 | [OpenAlex ID](https://openalex.org/A5029133577)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并利用StableAML数据集，针对以太坊平台USDT与USDC稳定币的转账行为，开展基于行为特征的三分类（正常、网络犯罪、被封禁）反洗钱模型；

**💡 创新点**

首次将稳定币特有的合约冻结、黑名单等交易标记融入特征工程，并证明基于树集成模型在此领域远优于图神经网络；

**🔧 技术方法**

采用树集成模型（CatBoost、LightGBM、XGBoost、Random Forest）、深度神经网络、GraphSAGE GNN以及逻辑回归作为基线，并通过宏观F1、准确率等指标评估；

**📊 数据集**

使用2017‑2025年以太坊USDT/USDC的完整事件日志（共1,092,761.61 USDT例子），筛选并标注16,433个钱包，形成68维特征表；

**📈 对比分析**

与传统基线对比，CatBoost在宏观F1上达到0.9775、准确率0.9857，树集成模型在多类别下显著优于GNN（F1≈0.80）和DNN（F1≈0.87），并在二分类场景下几乎完美（F1≈0.999）；

**⚠️ 局限性**

主要限制包括：GNN性能受限于交易图稀疏性；Blocklisted类识别仍面临高方差；模型易误判高频交易和套利机器人；仅关注稳定币交易，跨资产转移与隐私层未被覆盖；

---

## 232. Latent Diffeomorphic Co-Design of End-Effectors for Deformable and Fragile Object Manipulation

**arXiv ID:** 2602.17921 | [PDF](https://arxiv.org/pdf/2602.17921v1)

**作者:** Kei Ikemura `[一作]`, Florian T. Pokorny `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一套联立优化末端执行器形态与控制策略的共设计框架，用于温和操纵可变形与脆弱物体。

**💡 创新点**

创新点在于引入潜在差分同胚(LDM)形态参数化、利用仿真中“应力”等特权信号进行双层优化，并通过教师–学生扩散策略实现零射击真实部署。

**🔧 技术方法**

主要技术包括差分同胚形态编码、CMA-ES演化设计搜索、基于仿真特权信号的设计条件控制以及基于点云的扩散政策蒸馏。

**📊 数据集**

数据集使用了Thingi10K中手工挑选的手指模型（约1000个）用于LDM训练，仿真场景采用Genesis软体物理引擎，实际实验使用3D打印的末端执行器与RealSense L515点云。

**📈 对比分析**

在模拟和真实实验中与传统平行爪、贝叶斯优化以及基于强化学习的共设计进行对比，结果显示LDM+CMA-ES方案在抓取、推挤与切割任务中实现了更高的成功率、降低了最大应力，并在真实机器人上实现了零射击部署。

**⚠️ 局限性**

局限性主要体现在需手动调节控制器超参数、仿真物理属性与真实材料的差异仍可能影响性能，以及对大规模任务多样性和在线自适应的支持不足。

---

## 233. CLUTCH: Contextualized Language model for Unlocking Text-Conditioned Hand motion modelling in the wild

**arXiv ID:** 2602.17770 | [PDF](https://arxiv.org/pdf/2602.17770v1)

**作者:** Balamurugan Thambiraja `[一作]` (Max-Planck Institute for Intelligent Systems), Justus Thies `[通讯]` (Technical University of Darmstadt)

**通讯引用:** 9113 | [OpenAlex ID](https://openalex.org/A5036392768)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本工作构建了32k个真实场景下的3D手部动作与文本配对数据集3D‑HIW，并提出了CLUTCH模型，实现文本驱动的手部动作生成与动作描述的双向任务；

**💡 创新点**

创新点包括①基于部件与模态拆分的VQ‑VAE（SHIFT）提升动作令牌化的多样性与重构质量；②在LLM上加入几何细化阶段和Gumbel‑Softmax回归损失，显著提高生成动作的几何一致性；③双阶段VLM/LLM文本标注管线，减少误标注与幻觉；

**🔧 技术方法**

采用VILA与Claude进行视频文本标注，HaWor进行3D手部重建；使用SHIFT VQ‑VAE进行动作令牌化；利用T5作为LLM骨干，结合Gumbel‑Softmax与几何细化损失以及指令微调实现多任务学习；

**📊 数据集**

主要使用Ego4D和EgoVid5M视频数据，通过标注管线生成的32k条手部动作序列（12M手部姿态），与GRAB、ARCTIC、H2O、Gigahands等基准数据集对比；

**📈 对比分析**

在文本到动作（T2M）与动作到文本（M2T）任务上，使用R‑Precision、MMD、KID、Diversity、BLEU、MPJPE等指标评估，CLUTCH在所有指标上均优于MotionGPT、HumanMDM、T2M‑GPT等先进方法，且在多样性与几模态一致性上表现突出；

**⚠️ 局限性**

局限性主要在于仅聚焦手部动作，未处理手-物体交互；对重叠或极端动作的细粒度表达仍不够精准；依赖于3D重建的质量，复杂场景下的重建误差可能影响模型性能。

---

## 234. Noise Mitigation Methods for Digital Visible Light Communication

**arXiv ID:** 2602.18187 | [PDF](https://arxiv.org/pdf/2602.18187v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 235. Bayesian Optimality of In-Context Learning with Selective State Spaces

**arXiv ID:** 2602.17744 | [PDF](https://arxiv.org/pdf/2602.17744v1)

**作者:** Di Zhang `[一作]` (Xi'an Jiaotong-Liverpool University), Jiaqi Xing `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出将贝叶斯最优序列预测作为解释上下文学习（ICL）的新原理，并通过元学习使选择性状态空间模型（Selective SSM）在一系列线性高斯状态空间模型（LG-SSM）任务上实现渐近贝叶斯最优预测；

**💡 创新点**

创新点在于①将ICL视为贝叶斯滤波而非隐式梯度下降；②证明元训练的Selective SSM可逼近Kalman滤波并达到Cramér–Rao下界；③构造任务族展示Selective SSM在结构化噪声下优于任何经验风险最小化（ERM）方法；

**🔧 技术方法**

技术包括：元学习框架、线性高斯状态空间模型、Kalman滤波与贝叶斯推断、理论证明（渐近最优性与统计分离）、简化的Selective SSM架构（如Mamba），以及线性Transformer的ERM对照；

**📊 数据集**

使用的数据集：合成LG-SSM任务（随机生成A、C、Q、R矩阵）与字符级马尔可夫文本（HMM生成的50个隐藏状态、100字符词表）；

**📈 对比分析**

对比方法：元训练后的Selective SSM、线性Transformer（ERBF）以及贝叶斯最优先验Oracle。实验显示Selective SSM的过剩风险随任务数与上下文长度趋近Oracle水平，且在结构化噪声与隐藏状态跟踪任务上显著优于Transformer（误差分布更集中、准确率更高）；

**⚠️ 局限性**

局限性包括：理论仅针对线性高斯状态空间模型；实验使用简化模型与合成/高度结构化数据，未验证在大规模自然语言或非线性动力学任务上的可扩展性；未来需扩展至非线性系统与混合注意力-SSM架构。

---

## 236. Have We Mastered Scale in Deep Monocular Visual SLAM? The ScaleMaster Dataset and Benchmark

**arXiv ID:** 2602.18174 | [PDF](https://arxiv.org/pdf/2602.18174v1)

**作者:** Hyoseok Ju `[一作]` (DGIST), Giseop Kim `[通讯]` (DGIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ScaleMaster数据集，并针对大规模室内环境评估深度单目视觉SLAM的尺度一致性问题。

**💡 创新点**

创新点包括：①首个专门针对尺度一致性的室内大规模基准；②结合地图对地图评估（Chamfer距离、Drop Rate）揭示传统轨迹评估隐藏的尺度失真。

**🔧 技术方法**

使用了深度学习驱动的单目SLAM系统（DROID‑SLAM、MASt3R‑SLAM、VGGT‑SLAM），配合Sim(3)图优化、Umeyama对齐以及基于LiDAR的几何质量评估技术。

**📊 数据集**

采用自制的iPhone 14 Pro + LiDAR硬件采集的25条序列，涵盖多层结构、长距离循环、重复视角和低纹理等挑战性场景。

**📈 对比分析**

通过ATE与Chamfer/Drop Rate双重评估，发现虽然在传统基准上表现优异，但在ScaleMaster上多系统出现80‑90米级尺度漂移和高Drop Rate，显示尺度不一致严重，性能远低于预期。

**⚠️ 局限性**

局限性在于仅聚焦单目视觉SLAM，缺乏多模态融合和跨会话尺度校正的系统性解决方案，仍需进一步研究。

---

## 237. Tendon-Driven Reciprocating and Non-Reciprocating Motion via Snapping Metabeams

**arXiv ID:** 2602.18330 | [PDF](https://arxiv.org/pdf/2602.18330v1)

**作者:** Mohsen Jafarpour `[一作]` (University of Freiburg), Edoardo Milana `[通讯]` (University of Freiburg)

**通讯引用:** 610 | [OpenAlex ID](https://openalex.org/A5013363595)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

研究了一种基于螺旋式大尺寸可变形元件的肌腱驱动翻折结构，并将其集成到仿生游泳机器人中；

**💡 创新点**

通过几何设计实现了在刚性材料PLA中实现大幅可逆形变和翻折行为，且可通过边界条件调节产生单稳或双稳响应，实现了正向和非正向运动模式；

**🔧 技术方法**

采用FDM 3D打印（PLA）、单轴拉伸测试、轨迹跟踪与OpenCV分析、以及软体机器人结构组装等实验技术；

**📊 数据集**

主要使用自行制备的多组试件和机器人实验数据，未采用公开数据集；

**📈 对比分析**

通过比较固定‑固定与固定‑固定/铰接两种边界条件下的力–位移曲线和机器人两种工作模式的推进速度，发现非正向运动模式的推进效率约为81 mm/s，显著高于约9.6 mm/s的正向模式；

**⚠️ 局限性**

目前的局限包括未进行数值模拟以精准评估能量释放、未对材料长期耐久性和优化参数进行系统研究，以及仅在PLA材料上验证，缺乏与更柔性材料（如TPU）的对比。

---

## 238. Green by Design: Constraint-Based Adaptive Deployment in the Cloud Continuum

**arXiv ID:** 2602.18287 | [PDF](https://arxiv.org/pdf/2602.18287v1)

**作者:** Andrea D'Iapico `[一作]` (Politecnico di Milano), Monica Vitali `[通讯]` (Politecnico di Milano)

**通讯引用:** 632 | [OpenAlex ID](https://openalex.org/A5010941159)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于监控数据自动生成绿色部署约束的方法，用于云端-边缘微服务应用的环境友好调度。

**💡 创新点**

创新点在于将能源消耗、通信能耗和节点碳强度实时整合，动态生成可解释的绿色约束，并通过阈值自适应筛选最具影响力的约束。

**🔧 技术方法**

使用了监控工具Kepler和Istio收集能耗和通信指标，基于Prolog规则的约束库，以及阈值分位数的统计方法进行约束生成与权重评估。

**📊 数据集**

实验数据来自Google的Online Boutique微服务基准，结合不同国家/地区节点的碳强度数据，以及人工模拟的多场景（节点变化、碳强度波动、服务升级、通信负载增大）。

**📈 对比分析**

通过在五种动态场景下生成约束并测算生成时间与能耗，结果显示约束生成在最多1000个服务/1000个节点时耗时不超过120秒、能耗极低，约束数量与阈值成正比，且大部分约束带来显著的碳排放节约。

**⚠️ 局限性**

限制在于仅支持两类约束，未与真实调度器集成评估整体部署影响，且能源估计模型简单，缺乏对不同节点类别的细粒度区分。

---

## 239. Non-Stationary Online Resource Allocation: Learning from a Single Sample

**arXiv ID:** 2602.18114 | [PDF](https://arxiv.org/pdf/2602.18114v1)

**作者:** Yiding Feng `[一作]` (Hong Kong University of Science and Technology), Yige Wang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究非平稳环境下的多资源在线分配问题，提出了仅使用每期单个历史样本即可实现的量化阈值型元策略，能够在极小离线数据需求下实现近似最优收益。

**💡 创新点**

首创在非平稳多资源分配中使用单样本、量化阈值框架实现多项式对数级（O((log T)^3)）的遗憾上界，并在奖励已观测样本时实现O(√T)上界，在仅有类型样本时在满足最小到达概率假设下实现O(log √T)上界；相较于传统基于对偶的指数定价或双重学习方法大幅降低了对大预算和先验分布的依赖。

**🔧 技术方法**

采用分解式的量化阈值策略——首先用核估计或其他分布估计方法对奖励分布进行离线/在线估计；其次用流体松弛求解目标服务概率；最后将服务概率转化为动态接受阈值进行实时决策；在仅类型样本时加入部分/完全自适应阈值更新与取整技术以控制估计误差并实现多项式对数级遗憾。

**📊 数据集**

本文主要为理论研究，未使用公开真实数据集；所有结果均基于假设模型下的数学证明和上界分析。

**📈 对比分析**

与之前的指数定价、对偶FTRL、重估等方法相比，本文在奖励已观测样本时达到O(√T)上界，匹配或略优于现有O(√T)结果；在仅类型样本时，在满足最小到达概率假设下实现O(log √T)，而此前只能得到线性或O(√T)下界；最重要的是在完全自适应阈值策略下取得O((log T)^3)的对数级遗憾，这是同类问题中首次出现的多项式对数级上界，显著优于传统方法的O(√T)或O(T^{5/6})。

**⚠️ 局限性**

受限于对最小到达概率假设（γ>0）的依赖；若该假设不成立，单类型样本下无法获得亚线性遗憾；高阶常数和对数项可能导致在实际短期 Horizon 上的性能不佳；目前仅在理论模型下验证，缺乏真实数据或仿真实验的实证支持。

---

## 240. Machine Learning Based Prediction of Surgical Outcomes in Chronic Rhinosinusitis from Clinical Data

**arXiv ID:** 2602.17888 | [PDF](https://arxiv.org/pdf/2602.17888v1)

**作者:** Sayeed Shafayet Chowdhury `[一作]` (Indiana University Indianapolis), Vijay R. Ramakrishnan `[通讯]` (Indiana University School of Medicine)

**通讯引用:** 7176 | [OpenAlex ID](https://openalex.org/A5074669560)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建并评估基于预手术临床变量的机器学习模型，以预测慢性鼻窦炎患者接受内窥镜鼻窦手术后的SNOT-22改善是否达到临床显著差异。

**💡 创新点**

首次在多中心、前瞻性收集的大样本数据上，使用可解释的浅层神经网络和集成方法实现约85%准确率，并与专家与大语言模型对比，证明其辅助决策潜力。

**🔧 技术方法**

监督式机器学习，包括逻辑回归、SVM、随机森林、XGBoost、MLP及多数投票/AdaBoost/堆叠集成，并采用SHAP、特征重要性解释。

**📊 数据集**

来自NCT01332136的两个多中心CRS研究，合并524例已手术患者的结构化临床与SNOT-22预后数据。

**📈 对比分析**

与6位耳鼻喉专家和ChatGPT在30例难易度分层样本上对比，MLP准确率80%，超过专家平均75.6%和ChatGPT 56.7%；单一MLP在整体测试集上准确率约85%，相较传统模型略优。

**⚠️ 局限性**

样本量有限、类别不平衡导致对非响应类召回不足；仅基于结构化数据，外部验证缺失，且社会经济变量可能引入偏倚。

---

## 241. VQPP: Video Query Performance Prediction Benchmark

**arXiv ID:** 2602.17814 | [PDF](https://arxiv.org/pdf/2602.17814v1)

**作者:** Adrian Catalin Lutu `[一作]` (University of Bucharest / Bitdefender), Radu Tudor Ionescu `[通讯]` (University of Bucharest)

**通讯引用:** 8072 | [OpenAlex ID](https://openalex.org/A5081017623)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了首个针对内容基视频检索的查询性能预测基准VQPP，并使用其训练和评估多种预检索与后检索预测器

**💡 创新点**

创新点在于构建包含两大视频数据集与两种检索模型的四个评估场景，且首次将预检索BERT模型与LLM进行查询改写的奖励学习演示

**🔧 技术方法**

主要技术包括BERT预检索回归、CLIP/CLIP4Clip后检索分类、Correlation CNN、少量样本LLM预测、Direct Preference Optimization（DPO）与LoRA微调

**📊 数据集**

使用MSR‑VTT和VATEX两大视频数据集（共56K查询、51K视频）以及GRAM和VAST两种CBVR系统

**📈 对比分析**

与多种基线对比，预检索BERT模型在所有指标上表现最好，Pearson和Kendall相关系数最高但均低于0.5；LLM改写后可提升Recall@10约0.3个百分点

**⚠️ 局限性**

限制包括预测准确度仍偏低、数据集仅一对一映射导致后检索模型难以发挥优势、对查询长度和描述性差异的鲁棒性待提升

---

## 242. Drawing the LINE: Cryptographic Analysis and Security Improvements for the LINE E2EE Protocol

**arXiv ID:** 2602.18370 | [PDF](https://arxiv.org/pdf/2602.18370v1)

**作者:** Benjamin Dowling `[一作]` (King's College London), Bhagya Wimalasiri `[通讯]` (King's College London)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

对LINE Letter Sealing v2进行首次可证明的安全性分析，揭示其缺失前向秘密和事后安全，并提出改进的双路由器版本，完成形式化证明与性能基准测试。

**💡 创新点**

创新点在于：1）首次使用改进的Multi‑Stage Key Exchange (MSKE) 框架对LINE协议进行形式化分析；2）设计兼容原有架构的双路由器增强版，实现前向秘密与事后安全；3）提供完整的安全证明与实测评估，验证改进的可行性。

**🔧 技术方法**

使用了 MSKE 安全模型、密钥派生 (HKDF+SHA‑256)、椭圆曲线 Diffie‑Hellman (X25519)、对称加密 AES‑GCM、以及 Rust 生态中的 crypto‑libraries 进行实现与验证。

**📊 数据集**

本研究没有采用传统机器学习或公开数据集；通过构造实验环境对实际网络流量进行捕获，随后在本地进行 100 次迭代的时延与运算成本测量。

**📈 对比分析**

与原始 LINEv2 进行对比，测量一条完整消息的端到端延迟、各阶段加解密耗时以及单独加解密、密钥派生、DH 运算的微秒级成本。结果显示，改进版在引入双路由器后仅略微增加了 1–2 × 的 CPU 负载，整体性能仍可接受。

**⚠️ 局限性**

局限性包括：1）不涉及后量子安全或完整隐私性分析；2）只关注一对一聊天，不覆盖群聊或机器人通信；3）改进方案仍基于现有服务器架构，无法彻底替代 LINE 原始协议；4）安全模型中允许重放攻击，未实现完整的重放防护。

---

## 243. Continual-NExT: A Unified Comprehension And Generation Continual Learning Framework

**arXiv ID:** 2602.18055 | [PDF](https://arxiv.org/pdf/2602.18055v1)

**作者:** Jingyang Qiao `[一作]` (East China Normal University), Yuan Xie `[通讯]` (East China Normal University)

**通讯引用:** 30900 | [OpenAlex ID](https://openalex.org/A5100385336)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Continual-NExT 框架和 MAGE 方法，以实现双模态大语言模型的持续学习。

**💡 创新点**

创新点在于将输入和输出模态分别对应的 General LoRA 与 Expert LoRA 进行分层融合，并引入参数级 EMA 防止遗忘。

**🔧 技术方法**

使用 LoRA、P‑EMA、SEED‑X+LLaMA2‑chat‑13B 以及多模态 LoRA 专家等技术。

**📊 数据集**

使用 VQAv2、ImageNet、Flickr30k、OCRVQA、RefCOCO、HQEdit 等六个公开多模态数据集。

**📈 对比分析**

在与 LoRA、MoELoRA、EWC、LAE、PGP、CIA、RegLoRA 等基线对比时，MAGE 在 Avg.ACC 提升至 62.1、Forgetting 降至 12.26，整体表现优于所有方法。

**⚠️ 局限性**

局限在于随着模态数目增加，LoRA 专家数量呈线性增长，导致计算和内存成本上升；且目前仅覆盖已知模态，未能处理未见模态。

---

## 244. Parallel Complex Diffusion for Scalable Time Series Generation

**arXiv ID:** 2602.17706 | [PDF](https://arxiv.org/pdf/2602.17706v1)

**作者:** Rongyao Cai `[一作]` (Zhejiang University), Yong Liu `[通讯]` (Zhejiang University)

**通讯引用:** 35840 | [OpenAlex ID](https://openalex.org/A5100712539)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 PaCoDi（Parallel Complex Diffusion），将扩散模型从时域迁移到频域，并通过并行的实部与虚部分支实现长序列的高效生成。

**💡 创新点**

创新点包括：① 在频域证明复数分量的四象限独立性和逆向分解；② 引入均值场理论与交互校正机制，弥补实际数据的相位关联；③ 利用 Hermitian 对称性压缩序列长度，降低 self‑attention 的 FLOPs 50%；④ 在离散 DDPM 与连续 SDE 之间建立严谨的频域统一框架。

**🔧 技术方法**

使用技术包括：傅里叶变换、复数高斯噪声建模、平均场（Mean Field Theory）近似、交互校正分支、Mahalanobis 损失、Spectral Wiener Process、频域 SDE、Hermitian 对称压缩、并行分支网络、Self‑Attention 等。

**📊 数据集**

实验数据集涵盖 ETTh1、ETTm1、ECL、Exchange、Air Quality、Stocks、Sines 等，既评估无条件生成也评估有条件生成。

**📈 对比分析**

与 Diffusion-TS、TimeVAE、TimeGAN、Temporal DDPM、T2S 等基线对比，PaCoDi 在 Context‑FID、Correlation、Discriminative、Predictive、WAPE、MSE 等多项指标上均取得 SOTA，尤其在长序列和高频细节重建中表现突出；计算上实现 50% 的注意力 FLOPs 加速。

**⚠️ 局限性**

局限性：对频域转换的依赖使非周期性或强噪声信号的适配性受限；交互校正机制复杂且理论依赖均值场近似；短序列压缩收益有限；多维频谱中噪声异方差处理仍需经验调整。

---

## 245. Decomposing Retrieval Failures in RAG for Long-Document Financial Question Answering

**arXiv ID:** 2602.17981 | [PDF](https://arxiv.org/pdf/2602.17981v1)

**作者:** Amine Kobeissi `[一作]` (Université de Montréal), Philippe Langlais `[通讯]` (Université de Montréal)

**通讯引用:** 1965 | [OpenAlex ID](https://openalex.org/A5034154991)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对金融问答中的检索失败进行分解，提出基于oracle的检索分析框架，并引入域适配的页面评分器以提升页级检索效果。

**💡 创新点**

①提出oracle基于文档、页、块三层级分解检索失败的框架；②设计细粒度页面评分器，将页面作为检索中间层并进行端到端训练；③用页面级oracle评估检索瓶颈。

**🔧 技术方法**

检索方法包括密集检索（BGE-M3）、稀疏检索（BM25、SPLADE）、混合融合、HyDE、多样化HyDE、层级检索、交叉编码器重排序以及基于页面的双编码器页面评分器；生成采用Qwen-2.5-7B-Instruct；对齐使用SHA1哈希，评估使用ROUGE‑L、BLEU、numeric match。

**📊 数据集**

FinanceBench公开子集（150个问答实例），涵盖10‑K、10‑Q、8‑K、Earnings Call等文件类型。

**📈 对比分析**

在k=5下，对比多种检索策略和oracle upper bound，页面召回从0.46（BGE‑M3+Multi‑HyDE+ReRanker）提升到0.55（学习页评分器），接近oracle文档上限0.60；块级BLEU/ROUGE‑L也突破基线，生成的numeric match提高到30%，比最佳基线12%和oracle文档26%更好。

**⚠️ 局限性**

仅在FinanceBench 150问答上评估，主要集中在10‑K；需要黄金页面标签，依赖随机in‑batch负样本，未探索硬负样本；生成评估仅使用ROUGE‑L和numeric match，未完全覆盖财务准确性；未验证对其他文档类型或更大数据集的泛化。

---

## 246. Optimal Competitive Ratio of Two-sided Online Bipartite Matching

**arXiv ID:** 2602.18049 | [PDF](https://arxiv.org/pdf/2602.18049v1)

**作者:** Zhihao Gavin Tang `[一作]` (Shanghai University of Finance and Economics), Zhihao Gavin Tang `[通讯]` (Shanghai University of Finance and Economics)

**通讯引用:** 1259 | [OpenAlex ID](https://openalex.org/A5032128738)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文建立了双边顶点到达的在线二分匹配的分数版本的竞争比的最优上界（负结果），约为0.526，匹配了Wang和Wong（ICALP 2015）以及Tang和Zhang（EC 2024）所取得的下界（正结果）。

**💡 创新点**

创新点在于关闭了竞争比的差距，确立了0.526的紧上界，并证明了对于双边分数在线二分匹配问题，没有算法能够达到Γ^* + Ω(1)的竞争性。

**🔧 技术方法**

使用了基于对偶的算法和基于原始的算法，进行竞争比的分析和构造。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了在线匹配问题的理论模型和算法性能。

**📈 对比分析**

与完全在线模型进行比较，发现双边到达模型并不比完全在线模型更简单，且在匹配决策上存在显著损失。完全在线模型允许0.6竞争的分数算法，而双边模型的竞争比上限为0.526。

**⚠️ 局限性**

限制在于对于整数版本的问题，进展较少，且分数算法的不可行性结果自动扩展到随机整数算法，表明改进的空间非常小。

---

## 247. Evolution of Safety Requirements in Industrial Robotics: Comparative Analysis of ISO 10218-1/2 (2011 vs. 2025) and Integration of ISO/TS 15066

**arXiv ID:** 2602.17822 | [PDF](https://arxiv.org/pdf/2602.17822v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 248. On Translating Epistemic Operators in a Logic of Awareness

**arXiv ID:** 2602.18040 | [PDF](https://arxiv.org/pdf/2602.18040v1)

**作者:** Yudai Kubono `[一作]` `[通讯]` (Shizuoka University), Yudai Kubono (Shizuoka University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

将Awareness-Based Indistinguishability Logic（AIL）模型转换为HMS模型，并证明在此转换下，AIL的语言子集通过翻译保持真值，揭示了[≈]_i I_i [≈]_i 运算符在对应iHMS模型中等价于 I_i 运算符。

**💡 创新点**

①首次定义了从AIL模型到iHMS模型的HMS-转换；②证明了该转换下的真值保持性；③阐明了AIL与iHMS模型中隐含知识的语义差异，为两类模型的系统比较奠定基础。

**🔧 技术方法**

形式化语义定义、Kripke结构与事件结构的映射、满足关系的构造、归纳证明技术；使用逻辑语言的子集与技术背景运算符([≈]_i)进行翻译。

**📊 数据集**

无具体数据集，论文为理论性研究。

**📈 对比分析**

通过数学证明比较两类模型，未涉及实验性能评估；理论结果表明在满足特定一致性假设时，翻译保持真值，揭示两模型间的语义对应关系。

**⚠️ 局限性**

①仅针对AIL语言的特定子集，未覆盖完整语言；②假设所有世界在相同意识集上保持一致，限制了模型的普适性；③未探讨显式知识在iHMS模型与AIL中的对应关系，留待后续研究。

---

## 249. AsynDBT: Asynchronous Distributed Bilevel Tuning for efficient In-Context Learning with Large Language Models

**arXiv ID:** 2602.17694 | [PDF](https://arxiv.org/pdf/2602.17694v1)

**作者:** Hui Ma `[一作]` (Xinjiang University), Feng Pi `[通讯]` (Xinjiang General Station of Exit and Entry Frontier Inspection)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了异步分布式双层优化框架 AsynDBT，用于联合调优提示片段和上下文示例，以提升 LLM 的 In-Context Learning (ICL) 性能。

**💡 创新点**

首次将 ICL 建模为黑盒双层优化问题，设计了针对异构环境的异步联邦学习算法，并给出收敛性证明；同时引入鲁棒聚合抵御数据投毒。

**🔧 技术方法**

采用双层优化、Lagrange 乘子、割平面逼近、异步联邦学习、提示调优与鲁棒正则化等技术。

**📊 数据集**

实验使用 5G 术语关系识别数据集以及 GLUE 评测中的 COLA、SST‑2、MRPC、QQP、QNLI 五个 NLP 子任务。

**📈 对比分析**

与 RoBERTa、Manual Prompt、Zero‑shot CoT、Random ICL、KATE、BDPL、AdaICL 以及中心化 cenDBT 对比，AsynDBT 在大部分数据集上获得最高或次高准确率，5G 任务上提升近 10%，训练时间比 cenDBT 快约 40%。

**⚠️ 局限性**

局限在于仅使用静态训练数据，易出现领域特定的 hallucination 问题，对动态知识更新缺乏适应性。

---

## 250. Spatio-temporal Decoupled Knowledge Compensator for Few-Shot Action Recognition

**arXiv ID:** 2602.18043 | [PDF](https://arxiv.org/pdf/2602.18043v1)

**作者:** Hongyu Qu `[一作]` (Nanjing University of Science and Technology), Jinhui Tang `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 28603 | [OpenAlex ID](https://openalex.org/A5035112538)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于解构与整合的Few-Shot Action Recognition框架DiST，利用大语言模型对动作类别进行空间与时间属性拆解，并通过空间/时间知识补偿器学习对象级与帧级原型，实现细粒度时空特征匹配；

**💡 创新点**

创新点在于：①首次将LLM生成的解耦时空先验知识嵌入少样本动作识别；②设计空间知识补偿器(SKC)聚焦关键物体并学习对象原型；③设计时间知识补偿器(TKC)结合时间属性指导帧原型并捕捉动态语义；

**🔧 技术方法**

技术包括：CLIP ViT-B视觉+文本编码器、LLM提示生成空间/时间属性、跨注意力融合、对象级与帧级原型学习、双向Hausdorff距离与OTAM时空匹配、加权融合参数α；

**📊 数据集**

使用标准少样本动作数据集：HMDB51、UCF101、Kinetics、SSv2-full、SSv2-small；

**📈 对比分析**

与现有SOTA（如CLIP‑FSAR、MVP‑shot等）对比，DiST在5‑way 1‑shot下分别提升HMDB51 82.6%（vs 75.8%）、UCF101 98.3%（vs 96.0%）、Kinetics 92.7%（vs 89.7%）、SSv2‑full/SSv2‑small 95.6/96.0%等，整体性能明显优于对手；

**⚠️ 局限性**

局限性：①依赖LLM生成的通用属性，可能包含与实例不匹配的噪声；②属性生成与模型训练是分离的，需额外推理步骤；③对极端低shot场景仍受限于视觉信息不足，尽管LLM辅助缓解，但仍存在泛化边界；

---

## 251. Parameter-Efficient Domain Adaptation of Physics-Informed Self-Attention based GNNs for AC Power Flow Prediction

**arXiv ID:** 2602.18227 | [PDF](https://arxiv.org/pdf/2602.18227v1)

**作者:** Redwanul Karim `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Siming Bayer `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 178 | [OpenAlex ID](https://openalex.org/A5053698756)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对电力系统从中压到高压的电压域迁移，研究了物理约束的自注意力图神经网络的参数高效域适应，用低秩 LoRA 更新注意力投影并解冻预测头，实现对高压网络的 AC 电流流预测。

**💡 创新点**

提出将 LoRA 低秩更新与自注意力投影结合并选择性解冻预测头，以在保持 Kirchhoff 一致性的同时极大减少可训练参数并控制稳定-可塑性权衡。

**🔧 技术方法**

使用物理约束的图神经网络、Transformer‑style 自注意力、LoRA 低秩适配、Kirchhoff 一致性损失、RMSE 与物理残差评估。

**📊 数据集**

使用合成的中压（MV）与高压（HV）电网数据，包含 90,030 MV 样本和 45,030 HV 样本，使用欧盟/德国典型线路参数生成。

**📈 对比分析**

与零射击、全微调、仅头部微调、仅 LoRA 微调等基线对比，LoRA+预测头在保持 85.46% 参数压缩的同时，RMSE 与全微调仅差 2.6×10⁻⁴，物理残差相当，源域保留率略低。

**⚠️ 局限性**

在极少标注样本（β≤5%）下低秩适配仍逊于全微调，且对电压域差异较大的迁移需更多参数或更高秩；实验仅在合成数据上验证，缺乏真实电网验证。

---

## 252. Neurosymbolic Language Reasoning as Satisfiability Modulo Theory

**arXiv ID:** 2602.18095 | [PDF](https://arxiv.org/pdf/2602.18095v1)

**作者:** Hyunseok Oh `[一作]` (Seoul National University), Matthai Philipose `[通讯]` (Microsoft)

**通讯引用:** 8335 | [OpenAlex ID](https://openalex.org/A5030096283)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了Logitext neurosymbolic 框架，将自然语言约束转化为自然语言文本约束（NLTC）并与SMT求解器交互，实现文本与逻辑推理的交织；

**💡 创新点**

核心创新在于把LLM推理作为SMT理论，通过NLTC显式化部分逻辑结构并实现迭代求解，使自然文本与符号推理无缝结合；

**🔧 技术方法**

使用LLM（如GPT‑5、GPT‑4o）进行约束评估、NLTC Solver、SMT求解器Z3，以及迭代调用与缓存技术；

**📊 数据集**

采用新建的内容审核基准（5个多页政策）、LegalBench、Super‑NaturalInstructions三组数据集；

**📈 对比分析**

与直接提示和现有神经符号系统对比，Logitext在分类、文本实例生成与覆盖率任务上均表现更佳，特别是复杂政策下准确率提升显著，覆盖率提升显著；在CMOD和NI上优于GPT‑4o，LegalBench略逊；

**⚠️ 局限性**

局限包括条款级预测鲁棒性不足、列表注释误判导致错误、对完全可形式化任务的适用性仍有限

---

## 253. Latent Equivariant Operators for Robust Object Recognition: Promise and Challenges

**arXiv ID:** 2602.18406 | [PDF](https://arxiv.org/pdf/2602.18406v1)

**作者:** Minh Dinh `[一作]` (Dartmouth), Stéphane Deny `[通讯]` (Aalto University)

**通讯引用:** 1685 | [OpenAlex ID](https://openalex.org/A5065068004)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了在噪声MNIST上使用潜在可变换算子（latent equivariant operator）进行出域（out‑of‑distribution）分类的方法

**💡 创新点**

不需要预先给出对称变换的数学定义，也不要求在训练时覆盖完整变换参数范围，而是通过在潜在空间学习可变换算子实现对未见变换的外推和组合

**🔧 技术方法**

使用简单的单层线性编码器、预设或可学习的离散变换算子、两层MLP分类器，训练时采用两视角一致性正则化与交叉熵损失，并在推理时利用k‑NN估计变换索引

**📊 数据集**

使用自定义的旋转（10个离散角度）和平移（每轴14步周期）噪声MNIST数据集，去掉数字9类避免与6混淆

**📈 对比分析**

与仅使用相同编码器-分类器但不使用算子的基线模型对比；实验显示带算子的模型在未见变换角度、平移及其组合上保持近乎平坦的高准确率，且可学习算子在部分角落甚至优于预设算子

**⚠️ 局限性**

主要局限是：理论上无法保证算子在训练范围之外保持完全可变换性；对更复杂数据集和变换（如三维深度旋转）如何扩展尚未解决；算子层次和形式的选择仍需进一步研究

---

## 254. The Token Games: Evaluating Language Model Reasoning with Puzzle Duels

**arXiv ID:** 2602.17831 | [PDF](https://arxiv.org/pdf/2602.17831v1)

**作者:** Simon Henniger `[一作]` (Harvard University), Gabriel Poesia `[通讯]` (Harvard University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种基于模型互相提问编程谜题的竞争式评估框架TTG，利用LLM对方生成谜题并验证答案。

**💡 创新点**

创新点在于让模型自己生成可验证的难题，避免人工设计，且通过 Elo 评级实现可持续、不可饱和的对比。

**🔧 技术方法**

采用编程谜题表示、Python 代码沙箱执行、Elo/Bradley–Terry 排名、对局记录分析。

**📊 数据集**

使用了 10 名前沿 LLM 的所有两两对局（90 条），以及公开基准 HLE、GPQA 的成绩做对照。

**📈 对比分析**

通过 Elo 评级与 Solver/Proposer 胜率与 HLE/GPQA 关联，发现 Solver 胜率与传统基准相关度高，排名与现有榜单基本一致。

**⚠️ 局限性**

局限在于模型生成谜题仍易出现错误/过度自信，难度可控性不足，且需在沙箱环境执行代码，安全性与可解释性有待提升。

---

## 255. Many Tools, Few Exploitable Vulnerabilities: A Survey of 246 Static Code Analyzers for Security

**arXiv ID:** 2602.18270 | [PDF](https://arxiv.org/pdf/2602.18270v1)

**作者:** Kevin Hermann `[一作]` (Ruhr University Bochum), Thorsten Berger `[通讯]` (Ruhr University Bochum)

**通讯引用:** 5263 | [OpenAlex ID](https://openalex.org/A5072456187)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性综述了246种静态安全分析器，评估其目标漏洞、应用域、使用技术、评估方法与局限性；

**💡 创新点**

首次提供了完整的静态安全分析器生态图谱，揭示了漏洞覆盖度不均、评估缺乏标准基准、可利用漏洞比例极低等关键洞察；

**🔧 技术方法**

采用系统性文献综述方法（Scopus检索、四步筛选、开放编码），并对每篇论文进行手工编码；

**📊 数据集**

主要基于246篇论文的描述；评估数据集多为自定义或系统化生成的benchmark（如NIST Samate、DroidBench、Securibench‑Micro、SPEC CPU等），但整体规模普遍偏小；

**📈 对比分析**

比较方法以自定义benchmark、案例研究、工具对比为主，评估指标集中在可行性和性能，精度/召回率等统计指标使用较少；多数工具报告的漏洞中可利用率低（约11.8%）；

**⚠️ 局限性**

大部分论文未系统报告限制，主要局限包括近似/误报、范围/语言特性缺失、对动态特性处理不足、性能与多线程支持有限、缺少攻击模型说明、以及可利用性低等问题。

---

## 256. Causal Neighbourhood Learning for Invariant Graph Representations

**arXiv ID:** 2602.17934 | [PDF](https://arxiv.org/pdf/2602.17934v1)

**作者:** Simi Job `[一作]` (University of Southern Queensland), Jianming Yong `[通讯]` (University of Southern Queensland)

**通讯引用:** 2251 | [OpenAlex ID](https://openalex.org/A5012159374)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种在图结构上执行因果干预的框架CNL-GNN，利用反事实邻域生成和自适应边扰动来学习不变节点表示，从而提升节点分类性能。

**💡 创新点**

创新点在于将反事实邻域生成与基于注意力的边重要性估计结合，进行结构级的因果干预，并通过特征解耦与门控融合实现对因果与噪声特征的分离；这一方法首次在图神经网络中实现对结构噪声的可学习抑制。

**🔧 技术方法**

核心技术包括：反事实邻域生成（CNG）、边重要性估计模块（EIM）与注意力机制、群组感知的自适应边扰动、特征解耦与门控融合、以及对抗式的对比损失。

**📊 数据集**

在四个公开图数据集上验证：Cora、Citeseer、PubMed（学术引用网络）和Twitch（多语言社交网络）。

**📈 对比分析**

与ICL、ACE、CaNet、CIE、GCIL、CAM等七种先进GNN模型在节点分类F1分数上比较，CNL-GNN在所有数据集上均获得最高分（例如Twitch 99.41%、Cora 93.51%、Citeseer 86.87%、PubMed 90.23%），并在跨域（Twitch不同语言域）评估中表现出稳健的性能下降（1.4%~11.2%）。

**⚠️ 局限性**

局限性包括：仍需在动态或异构图上验证；对超参数（如边掉落率、群组检测方式）的敏感性未系统探究；以及在极端分布偏移（如结构截断或属性极端改变）下的鲁棒性尚未充分评估。

---

## 257. Reinforcement-Learning-Based Assistance Reduces Squat Effort with a Modular Hip--Knee Exoskeleton

**arXiv ID:** 2602.17794 | [PDF](https://arxiv.org/pdf/2602.17794v1)

**作者:** Neethan Ratnakumar `[一作]` (New Jersey Institute of Technology), Xianlian Zhou `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 1168 | [OpenAlex ID](https://openalex.org/A5019820614)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文设计并验证了一种基于强化学习的神经网络控制器，用于模块化髋膝外骨骼在重复深蹲动作中的实时辅助；

**💡 创新点**

创新点在于：①利用仿真驱动的RL训练获得能够根据实时髋膝角度与速度产生个性化助力的控制网络；②该控制器实现了跨用户自适应，无需预设轨迹；③通过物理模拟实现从仿真到真实设备的迁移。

**🔧 技术方法**

主要技术包括：强化学习（RL）训练、神经网络（多层全连接+Tanh输出）、物理仿真（三维肌肉骨骼模型）、传感融合（IMU获取髋膝角度与速度）、实时CAN通信与Raspberry Pi控制。

**📊 数据集**

实验数据集：五名健康成人在三种条件（无外骨骼、零扭矩、主动助力）下进行三分钟节拍指导深蹲，记录代谢（COSMED K5）、心率、EMG、运动学（Xsens IMU）、力学（力板）等。

**📈 对比分析**

相较于无外骨骼条件，主动助力可使净代谢率平均降低约10%（个体差异在8.8%–21.4%之间），心率略降；与零扭矩条件比较，代谢率下降幅度相似，且在部分受试者中表现出更显著的节能效果。

**⚠️ 局限性**

局限性包括：受试者间代谢与运动学差异大；外骨骼结构限制导致深蹲角度减小，可能改变运动策略；传感噪声与通信延迟影响助力平滑度；实验周期较长可能引入疲劳和校准漂移；仅在健康成人实验，缺乏工伤工人群体验证。

---

## 258. Dual Length Codes for Lossless Compression of BFloat16

**arXiv ID:** 2602.17849 | [PDF](https://arxiv.org/pdf/2602.17849v1)

**作者:** Aditya Agrawal `[一作]` (Google LLC), Ravi Iyer `[通讯]` (Google LLC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种双长度编码（Dual Length Codes）用于大型语言模型训练和推理中BFloat16张量的无损压缩。

**💡 创新点**

创新点在于将Huffman码的频率分布优势与通用码的非位序解码特性结合，使用单一前缀位区分两种码长，仅用8条条目LUT实现快速编码/解码。

**🔧 技术方法**

采用了熵编码原理、Huffman码分析、前缀码设计、查找表（LUT）实现及简单的位操作技术。

**📊 数据集**

在Gemma 2B模型的训练阶段，对FFN1激活张量（BFloat16）进行统计分析，使用该模型的张量分布作为实验数据集。

**📈 对比分析**

与传统Huffman码对比：压缩率从21.3%降至18.6%，但解码速度显著提升，硬件实现复杂度大幅下降，适用于网络带宽受限的集体操作。

**⚠️ 局限性**

局限性包括压缩率略低，方案对不同张量分布的适应性有限，且仍需针对每类张量预先构建LUT。

---

## 259. Understanding the Fine-Grained Knowledge Capabilities of Vision-Language Models

**arXiv ID:** 2602.17871 | [PDF](https://arxiv.org/pdf/2602.17871v1)

**作者:** Dhruba Ghosh `[一作]` (Stanford University), Ludwig Schmidt `[通讯]` (Stanford University)

**通讯引用:** 4187 | [OpenAlex ID](https://openalex.org/A5111490184)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文系统评估了多种最新的视觉-语言模型（VLM）在细粒度分类任务中的表现，并通过大规模消融实验探究了模型基底、视觉编码器、预训练策略等关键因素对细粒度识别能力的影响。

**💡 创新点**

创新点在于首次将细粒度分类视为VLM评估的重要维度，并通过细粒度和通用VQA对比发现两者存在显著差距；随后通过22次消融实验揭示了：更强的LLM能均衡提升所有任务；更强的视觉编码器对细粒度任务提升更大；在预训练阶段解冻LLM并充分训练视觉-语言连接器显著提升细粒度性能；预训练数据规模是关键瓶颈。

**🔧 技术方法**

采用了LLaVA-1.5架构、OpenCLIP/DFN-CLIP视觉编码器、Vicuna/Qwen2/LLama-2等LLM、两阶段预训练（图像-标题对齐 + 指令微调）以及多轮多模态指令完成数据。

**📊 数据集**

细粒度数据集包括ImageNet‑1K、Oxford Flowers‑102、Oxford‑IIIT Pets‑37、Food‑101；通用VQA评估使用MMMU、MMBench、MMStar等。

**📈 对比分析**

将细粒度数据转换为5选一多项选择题，使用VLMEvalKit的prompt模板评估；结果显示不同VLM在通用VQA上相近，但在细粒度分类上差距可达19pp；改进后细粒度准确率可从52.8%提升至73.4%，但仍与最佳模型（Qwen2‑VL‑Chat 87.9%）存在12pp差距。

**⚠️ 局限性**

局限在于实验规模受限，预训练仅使用约1M图像+字幕，未达到某些VLM使用的数十亿级预训练；同时只关注两阶段训练，未探讨其他预训练策略；因此细粒度提升的最终瓶颈可能是预训练规模与策略。

---

## 260. OODBench: Out-of-Distribution Benchmark for Large Vision-Language Models

**arXiv ID:** 2602.18094 | [PDF](https://arxiv.org/pdf/2602.18094v1)

**作者:** Ling Lin `[一作]` (University of Science and Technology of China), Jingrun Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 2589 | [OpenAlex ID](https://openalex.org/A5025195102)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了OODBench，一个专门评估视觉语言模型在离散分布(OOD)下表现的基准数据集，并提出了自动化的实例级OOV分割流程以及基于进阶提问的Basic-to-Advanced Progression（BAP）度量。

**💡 创新点**

创新点包括①利用多模型交叉验证实现自动化OOV实例分割，减少人工干预；②OOVBench覆盖约4万实例的covariate shift OOD样本，填补了现有缺口；③设计了BAP度量，系统评估识别、计数与推理三维度的模型能力。

**🔧 技术方法**

技术手段包括基于CLIP/BLIP2的OOD检测、Purify去干扰操作、交叉验证策略、Chain-of-Thought提示以及多维度评估指标（Accuracy、F1、Precision、Recall、MCC、BAP）。

**📊 数据集**

使用公开数据集COCO、LVIS、nuScenes、Cityscapes等，提取ID与OOV样本构建Benchmark。

**📈 对比分析**

通过在10个主流VLM（如LLaVA‑NeXT、GPT‑4o、InternVL2、Gemini等）上对ID、OOD‑S、OOD‑H进行Accuracy、F1等指标测试，发现OOD‑H准确率普遍下降20‑30%，CoT提示对大多数模型效果有限甚至负面。

**⚠️ 局限性**

局限性在于OOV检测依赖预训练模型假设，可能无法覆盖所有真实分布；BAP度量虽全面但仍简化推理流程；Benchmark主要基于公开数据，未必涵盖所有工业或边缘场景。

---

## 261. Unifying Formal Explanations: A Complexity-Theoretic Perspective

**arXiv ID:** 2602.18160 | [PDF](https://arxiv.org/pdf/2602.18160v1)

**作者:** Shahaf Bassan `[一作]` (Hebrew University of Jerusalem), Guy Katz `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 2494 | [OpenAlex ID](https://openalex.org/A5102986148)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出统一框架，统一充分理由与对比理由的计算，证明全局解释的价值函数具备单调性、超/子模性质，从而可用贪心算法多项式求解子集最小解释，并通过子模集合覆盖获得全局卡尔达最小解释的常数因子逼近；局部解释仍为NP‑hard。

**💡 创新点**

核心创新是发现全局解释在组合优化上具有单调性与子模/超模结构，将复杂解释问题转化为可近似的集合覆盖问题，提供全局与局部复杂度的严格对比；同时给出子集最小和卡尔达最小两类问题的多项式/近似解法。

**🔧 技术方法**

运用组合优化理论（单调性、子模/超模、曲率分析）、贪心算法、子模集合覆盖的近似保证，以及在经验分布或独立分布下对期望值的计算。

**📊 数据集**

理论适用于任意经验分布或独立分布，涵盖决策树、神经网络、树集成等模型；本文未给出具体实验数据集。

**📈 对比分析**

相较于已有NP‑hard结论，本文证明全局解释可在多项式时间或常数因子逼近；在经验分布下近似比为O(log|D|)或常数，体现了显著的理论性能提升。

**⚠️ 局限性**

局限包括：缺乏实验验证；经验分布下期望计算可能代价高；未探讨更简单模型的进一步优化；对随机逼近方案、其他重要性度量的分析仍待研究。

---

## 262. TFL: Targeted Bit-Flip Attack on Large Language Model

**arXiv ID:** 2602.17837 | [PDF](https://arxiv.org/pdf/2602.17837v1)

**作者:** Jingkai Guo `[一作]` (Arizona State University), Deliang Fan `[通讯]` (Arizona State University)

**通讯引用:** 5220 | [OpenAlex ID](https://openalex.org/A5047916979)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了TFL框架，对大型语言模型进行定向位翻转攻击，能够在保持大部分功能不受影响的前提下精准操纵特定提示的输出。

**💡 创新点**

创新点包括：①关键词聚焦攻击损失函数；②Aux Utility Score在位翻转选择中的平衡机制；③ImpactScore加速的位翻转搜索策略，并实现低于50次位翻转即可完成定向攻击。

**🔧 技术方法**

技术手段涵盖：梯度搜索、SKIP加速搜索、范围约束位翻转、Rowhammer硬件漏洞利用、对模型权重的BF16/INT8量化与浮点表示。

**📊 数据集**

实验使用的主要数据集：DROP、GSM8K、TriviaQA（评估），MMLU、WikiText（辅助），以及攻击样本中包含的目标与同义问题。

**📈 对比分析**

与SilentStrike、SBFA、GenBFA、PrisonBreak等SOTA BFA方法对比，TFL在Qwen3-8B、DeepSeek-R1-Distill-Qwen-14B、LLaMA-3.1-8B-INSTRUCT三大模型下，平均位翻转数≤15，且在上述评估集上保持≥75%准确率，显著优于对手且不显著降低模型整体性能。

**⚠️ 局限性**

局限性：需白盒访问并利用Rowhammer硬件；主要针对文本提示的定向攻击，对不同模型层的防护效果尚不完全；缺乏自动化防御与对抗性评估。

---

## 263. Multi-material Multi-physics Topology Optimization with Physics-informed Gaussian Process Priors

**arXiv ID:** 2602.17783 | [PDF](https://arxiv.org/pdf/2602.17783v1)

**作者:** Xiangyu Sun `[一作]` (University of California), Ramin Bostanabad `[通讯]` (University of California)

**通讯引用:** 2697 | [OpenAlex ID](https://openalex.org/A5062806994)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于物理信息高斯过程（PIGP）的多材料、多物理场拓扑优化框架，能够同时求解设计、状态和伴随场，采用PIGP的独立高斯过程先验与PGCAN神经网络均值函数结合。

**💡 创新点**

创新点包括：①将设计、状态与伴随变量各自用独立GP先验与神经均值函数表示，天然满足Dirichlet约束并避免离散化；②在损失函数中直接嵌入目标能量、势能及约束，使得所有参数可一次性联合优化；③采用形状函数与高斯积分的混合数值近似，显著提升梯度准确性与训练稳定性；④多网格与课程学习策略进一步增强收敛速度与鲁棒性。

**🔧 技术方法**

技术手段包括：高斯过程、PGCAN（参数化网格卷积注意力网络）神经网络、自动微分、形状函数插值、积分和梯度计算、课程学习、复合损失优化与Adam迭代。

**📊 数据集**

数据集主要是合成的基准问题：二维/三维合规性最小化、热传导优化、可耦合机制设计以及热-机械耦合装置设计，使用开源代码与COMSOL生成的有限元结果作为评估基准。

**📈 对比分析**

与传统SIMP、PolyMat和COMSOL相比，PIGP在目标性能（合规性、热能量、输出位移）上相当或略优，且得到的拓扑具有更清晰的界面、更低的灰区和更小的方差；在单/多材料、二维/三维问题上均表现出良好的可扩展性和稳定性。

**⚠️ 局限性**

局限性包括：①训练过程仍需数千到数万次迭代，计算成本相对较高；②对非常大规模或更复杂的物理耦合场（非线性、大尺寸）仍未充分验证；③在多材料场景下仍可能出现细小断裂或灰区，需进一步改进投影或正则化策略。

---

## 264. Asynchronous Heavy-Tailed Optimization

**arXiv ID:** 2602.18002 | [PDF](https://arxiv.org/pdf/2602.18002v1)

**作者:** Junfei Sun `[一作]` (University of Chicago), Tian Li `[通讯]` (University of Chicago)

**通讯引用:** 202931 | [OpenAlex ID](https://openalex.org/A5089379689)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在存在重尾梯度噪声的异步分布式优化问题，提出了两种基于延迟感知的改进策略——延迟下调（Staleness-Aware Downplaying）和延迟补偿（Delay Compensation），并给出了相应的理论收敛分析和实验验证。

**💡 创新点**

创新点主要有：①首次在异步环境下系统性分析重尾噪声对收敛的影响；②提出的延迟下调能够根据更新的滞后程度动态缩放梯度，从而在非均匀延迟场景下提升收敛鲁棒性；③延迟补偿利用梯度的一阶泰勒展开与自适应海森近似来“补偿”旧更新的时滞，使得慢速客户端的信息被充分利用。

**🔧 技术方法**

技术手段包括：①坐标级梯度裁剪（TailClip/Clip）处理重尾噪声；②服务器端与客户端的异步聚合框架（Server-Centric 与 Client-Centric 两种模式）；③基于延迟的学习率调度与梯度重缩放；④延迟补偿中的 Hessian 近似与矩阵乘积操作；⑤理论分析中利用 L‑smooth、G‑Lipschitz、μ‑强凸等假设下的收敛证明。

**📊 数据集**

实验数据集主要有：CIFAR‑10（ViT 视觉模型）和 GLUE（BERT 预训练模型的自然语言理解任务），用于评估不同延迟设置下的准确率与训练时间。

**📈 对比分析**

与同步基线、原始异步 SGDClip/Clip^2 以及两种异步基线（FADAS、DN+DyLU）比较，实验显示：①在轻微和严重 straggler 场景下，带有延迟下调或延迟补偿的异步算法在保持或提升准确率的同时，训练时间显著低于同步；②在大多数 GLUE 子任务上，DC 或 SD 的平均准确率高于 vanilla 异步；③在不同 M（缓冲大小）与服务器/客户端模式下，异步方法均优于同步。

**⚠️ 局限性**

局限性包括：①理论分析主要针对坐标裁剪优化器，其他优化器的适用性尚未证明；②假设梯度噪声满足重尾分布且高阶导数有界，实际场景中可能不完全满足；③延迟补偿需要额外的 Hessian 近似计算，可能增加通信与计算负担；④在极端大延迟或极度不均匀的网络条件下，仍需进一步实验验证其稳健性。

---

## 265. A traffic incident management framework for vehicular ad hoc networks

**arXiv ID:** 2602.18208 | [PDF](https://arxiv.org/pdf/2602.18208v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 266. The Statistical Signature of LLMs

**arXiv ID:** 2602.18152 | [PDF](https://arxiv.org/pdf/2602.18152v1)

**作者:** Ortal Hadad `[一作]` (Sapienza University of Rome), Walter Quattrociocchi `[通讯]` (Sapienza University of Rome)

**通讯引用:** 13223 | [OpenAlex ID](https://openalex.org/A5008291667)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对人类和LLM生成文本进行无损压缩比及其前缀曲线等统计特征分析，揭示两者在结构统计上的差异。

**💡 创新点**

首次将无损压缩作为跨域、无模型依赖的统计特征，系统评估不同生成情境下的结构签名。

**🔧 技术方法**

采用gzip（RFC1952）无损压缩、前缀压缩曲线、条件压缩、词序贡献指标等信息论方法。

**📊 数据集**

使用Human–AI Parallel English Corpus、Wikipedia与Grokipedia对照集，以及Moltbook与Reddit社交对话集。

**📈 对比分析**

基于压缩率及衍生特征训练梯度提升树进行二分类/多分类，Human/LLM二分类准确率达93%（macro F1 0.88），Wikipedia/Grokipedia 85%（macro F1 0.85），Reddit/Moltbook 88%（macro F1 0.88）。

**⚠️ 局限性**

仅关注表面可压缩性，无法反映语义质量；在短文本或高度碎片化情境下差异显著性降低；依赖压缩算法参数且不考虑后期人工编辑影响。

---

## 267. Unifying Color and Lightness Correction with View-Adaptive Curve Adjustment for Robust 3D Novel View Synthesis

**arXiv ID:** 2602.18322 | [PDF](https://arxiv.org/pdf/2602.18322v1)

**作者:** Ziteng Cui `[一作]` (University of Tokyo), Tatsuya Harada `[通讯]` (University of Tokyo)

**通讯引用:** 10837 | [OpenAlex ID](https://openalex.org/A5042711470)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Luminance-GS++，一种基于 3D Gaussian Splatting 的框架，用于在多视角场景下实现对复杂光照和色彩失真具有鲁棒性的全新视角合成。

**💡 创新点**

创新点在于将视角自适应的全局色调曲线调整与像素级残差细化相结合，并通过无监督的光度一致性、颜色校正和曲线正则化约束，既保持了显式 3DGS 表达式，又实现了对光照和色彩变化的高质量校正。

**🔧 技术方法**

采用的技术包括 3D Gaussian Splatting、可学习的 per‑view 颜色矩阵、视角自适应 Tone‑Curve、ConvNeXt 残差分支、无监督空间一致性损失、色彩校正损失以及曲线正则化。

**📊 数据集**

使用的数据集主要为 LOM benchmark（低照度/过曝）和 MipNeRF‑360（光照变化、色彩失真、混合失真），并在这些数据集上合成了多种光照和色彩扰动。

**📈 对比分析**

与 2D 图像/视频增强方法、NeRF‑与 3DGS‑基准方法对比，Luminance‑GS++ 在 PSNR、SSIM、LPIPS 等指标上均取得领先成绩（如 LOM 低照度下 PSNR 20.74 dB，SSIM 0.924，LPIPS 0.164），同时保持了与原 3DGS 相当的实时渲染速度和显存占用。

**⚠️ 局限性**

局限性包括对伪标签生成的依赖（可能限制极端光照场景下的表现）、缺乏显式光照物理模型、在新型极端照明条件下的泛化能力仍待验证，以及相较纯 3DGS 增加了一定的计算和内存开销。

---

## 268. On the Evaluation Protocol of Gesture Recognition for UAV-based Rescue Operation based on Deep Learning: A Subject-Independence Perspective

**arXiv ID:** 2602.17854 | [PDF](https://arxiv.org/pdf/2602.17854v1)

**作者:** Domonkos Varga `[一作]` `[通讯]`, Domonkos Varga

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对 Liu 与 Szirányi 在 UAV 基于深度学习的手势识别论文的评估协议进行批判性分析，揭示其随机帧级拆分导致的主体数据泄漏，使报告的近乎 100% 准确率失真。

**💡 创新点**

创新点在于系统化阐释了主体独立划分对手势识别泛化的重要性，并通过多源证据（混淆矩阵、学习曲线、LLM 交叉验证）验证了数据泄漏的存在。

**🔧 技术方法**

使用的技术包括 OpenPose（2D 关键点提取）、Deep SORT（人追踪）、基于骨架特征的四层全连接深度神经网络、Adam 优化器以及标准交叉熵损失。

**📊 数据集**

所用数据集由 6 名受试者在 UAV 场景下录制的手势视频构成，随后将所有帧随机拆分为 90% 训练 / 10% 测试集，未实现主体独立划分。

**📈 对比分析**

与原论文的对比显示：混淆矩阵几乎完全对角，训练/测试曲线同步、精度几乎 100% 并且测试误差始终低于训练误差，表明模型未真正泛化，性能指标被高估。

**⚠️ 局限性**

限制在于样本量极小且仅包含 6 位受试者，随机帧级拆分导致主体信息泄漏，缺乏跨主体验证；因此报告的性能无法推广到实际无人机救援场景中的未知人群。

---

## 269. EgoPush: Learning End-to-End Egocentric Multi-Object Rearrangement for Mobile Robots

**arXiv ID:** 2602.18071 | [PDF](https://arxiv.org/pdf/2602.18071v1)

**作者:** Boyuan An `[一作]` (New York University), Chen Feng `[通讯]` (New York University)

**通讯引用:** 63127 | [OpenAlex ID](https://openalex.org/A5100352749)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 EgoPush 框架，利用单目/深度摄像头完成多物体非抓取式长周期重新排列任务。

**💡 创新点**

创新点包括：① 对教师进行视角约束，使其行为可从学生视角恢复；② 设计对象中心隐空间编码相对关系；③ 用阶段化、时衰减的奖励解决长周期信用分配；④ 引入关系对齐蒸馏提升学生对空间关系的理解。

**🔧 技术方法**

技术手段：使用基于 PointNet 的教师网络与 MLP 策略；利用 PPO 训练受限观测下的教师；通过在线 DAgger 方式将教师行为与关系损失蒸馏给使用 CNN 的视觉学生；实现虚拟视野掩模、中心门控可见性、阶段奖励等算法模块。

**📊 数据集**

数据集与实验环境：在 NVIDIA Isaac Lab 进行大规模并行仿真（8,192 环境）使用立方体、圆柱体、三棱柱等 3 种几何形状；在真实世界的 3m×3m Arena 上使用 TurtleBot3 Burger + Intel RealSense D435i，测试彩色盒子重新排列。

**📈 对比分析**

与经典基于地图的 SIM 方案及多种端到端视觉 RL（RGB、RGB‑D、语义掩码、RNN）对比，EgoPush 在交叉形和线形排列任务中取得 100% 成功率，远超基线（≤10%）。在简化任务中亦显著提升成功率和轨迹长度。

**⚠️ 局限性**

局限性：学生策略为即时反应式，缺乏对临时不可见物体的记忆，易在狭窄通道或连续障碍中产生“目标‑路径”循环；未来可结合隐空间的循环记忆（GRU/LSTM）提升对遮挡物的持久推断。

---

## 270. The Economical-Ecological Benefits of Matching Non-matching Socks

**arXiv ID:** 2602.18221 | [PDF](https://arxiv.org/pdf/2602.18221v1)

**作者:** Teddy Lazebnik `[一作]` (University of Haifa), Teddy Lazebnik `[通讯]` (Jönköping University)

**通讯引用:** 1137 | [OpenAlex ID](https://openalex.org/A5041000511)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

**🎯 论文内容**

研究袜子丢失导致单袜问题的经济与生态成本，并探索容忍非匹配袜子对资源利用和社交成本的影响。

**💡 创新点**

提出将袜子配对问题建模为不确定性下的序贯决策，量化允许非匹配配对的生态/经济收益与社交惩罚，并提供可解释的阈值策略。

**🔧 技术方法**

采用随机模拟（Monte Carlo）评估不同配对策略；使用最大似然估计获取个体不匹配敏感度与多样性偏好；基于阈值的贪心与孤立袜子救援等策略。

**📊 数据集**

自建袜子特征与价格分布数据（可选使用真实袜子样本）；实验使用60名成人参与者完成配对与组合偏好任务来估计参数。

**📈 对比分析**

通过与严格匹配基线对比，结果显示允许非匹配配对能显著降低“无袜子日”并减少浪费，阈值策略在保持低社交成本的同时提高服务质量，整体性能优于传统严格匹配。

**⚠️ 局限性**

生态成本采用价格比例近似，未考虑织物类型、洗涤过程等细节；实验样本仅来自以色列，社交规范可变，未涵盖多季节、洗衣习惯等现实因素。

---

## 271. RoEL: Robust Event-based 3D Line Reconstruction

**arXiv ID:** 2602.18258 | [PDF](https://arxiv.org/pdf/2602.18258v1)

**作者:** Gwangtak Bae `[一作]` (Seoul National University), Young Min Kim `[通讯]` (Seoul National University)

**通讯引用:** 28616 | [OpenAlex ID](https://openalex.org/A5100337311)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于事件相机的鲁棒 3D 直线重建流水线 RoEL，实现了从原始事件数据到稀疏 3D 直线地图的完整转换，并对相机姿态进行优化。

**💡 创新点**

创新点在于：①利用多窗口、多表示的事件到图像转换实现稳定的 2D 直线检测；②以空间-时间平面拟合精细化线条并关联事件；③在 3D 空间上用 Grassmann 并行几何损失进行线条与事件的联合优化；④将稀疏直线映射作为跨模态中间表示，提升定位与配准性能。

**🔧 技术方法**

核心技术包括：事件帧多窗口组合、空间-时间平面拟合、基于 RANSAC 的 2D 直线三角化、Grassmann 测度的几何损失、最小正交直线参数化、端点裁剪以及多模态后处理。

**📊 数据集**

使用新构建的合成数据集以及真实室内事件序列（含不同光照与运动条件），并在此基础上与 EL‑SLAM、RGB‑D 以及全景图像定位等基线进行对比。

**📈 对比分析**

与传统直接方法 EL‑SLAM、基于图像的 3D 直线映射和 RGB‑D 视觉里程计相比，RoEL 在重建精度、直线覆盖率、姿态误差及跨模态定位准确度上均取得显著提升，实验结果表明误差下降 30% 以上。

**⚠️ 局限性**

局限性包括：目前仅离线实现；对非直线曲面（如圆柱边缘）表示不足；空间‑时间平面拟合在剧烈旋转或非线性运动下可能失效；需要进一步提升实时性能与增量优化能力。

---

## 272. A Probabilistic Framework for LLM-Based Model Discovery

**arXiv ID:** 2602.18266 | [PDF](https://arxiv.org/pdf/2602.18266v1)

**作者:** Stefan Wahl `[一作]` (Machine Learning in Science), Daniel Gedon `[通讯]` (Machine Learning in Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

将大语言模型（LLM）驱动的机制模型发现视为对可执行程序的贝叶斯后验推断，提出 ModelSMC 通过 SMC 迭代采样、评估和重采样来逼近后验分布。

**💡 创新点**

创新点在于：①给出正式的概率推断框架而非经验性提示；②利用 LLM 作为提议分布，结合似然加权实现模型空间的自适应探索；③通过理论分析证明在理想条件下该方法是一致的。

**🔧 技术方法**

核心技术包括：Sequential Monte Carlo（SMC）算法、LLM 生成/改写代码、神经似然估计（NLE）与 TabPFN、后验预测检查和参数推断（NPE）。

**📊 数据集**

使用三类数据集：合成 SIR 传染病模型、真实 QSP 肾脏钾离子调节模型（R 语言实现）以及 Allen Cell Types Database 中的神经元膜电位记录。

**📈 对比分析**

与 FunSearch+（Python 版）和单粒子 ModelSMC 对比，ModelSMC 在所有任务中获得更高或相近的似然分数和后验预测质量，证明在多语言、多复杂度模型上均具备竞争力。

**⚠️ 局限性**

主要局限包括：①计算开销大（需多次仿真和似然估计）；②似然近似误差可能导致偏倚；③受限于 LLM 的结构生成能力；④缺乏对模型空间相似性或聚类的几何度量。

---

## 273. NIMMGen: Learning Neural-Integrated Mechanistic Digital Twins with LLMs

**arXiv ID:** 2602.18008 | [PDF](https://arxiv.org/pdf/2602.18008v1)

**作者:** Zihan Guan `[一作]` (University of Virginia), Anil Vullikanti `[通讯]` (University of Virginia)

**通讯引用:** 4087 | [OpenAlex ID](https://openalex.org/A5044848288)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了NIMM评估框架和NIMMGen代理式框架，用于在部分观测和多任务场景下利用LLM自动生成神经网络与机理模型耦合的数字孪生，并通过迭代优化、错误纠正与验证提升代码与语义正确性；

**💡 创新点**

创新点包括（1）构建更贴近真实的NIMM评估环境，支持部分观测、多领域、多任务与通用神经机理模型；（2）设计NIMMGen迭代循环，集成记忆、错误修正与验证代理，显著降低bug并提升模型质量；（3）展示生成模型在反事实干预（如社交距离）下的合理行为，验证其科学可解释性；

**🔧 技术方法**

技术手段主要是大语言模型（LLM）代码生成、代理式迭代优化、RAG检索式代码补全、错误修正模块、验证代理、神经网络参数化的机理模型、AdamW训练及实时评估；

**📊 数据集**

使用的数据集涵盖三大科学领域：公共卫生（Influenza‑USA、MRSA‑Virginia、COVID‑Bogota、COVID‑Medellin）、临床健康（肺癌PKPD三种治疗场景）以及材料科学（合金屈服强度预测）；

**📈 对比分析**

与零射提示、HDTwinGen、纯黑盒LSTM/Transformer以及Grad‑Metapopulation等基线比较，采用RMSE与bug计数作为指标。NIMMGen在所有数据集上显著降低RMSE、bug率，且在实时评估和反事实干预模拟中表现出更高的可解释性与鲁棒性；

**⚠️ 局限性**

局限性在于仍需依赖LLM，验证过程仍以LLM为主，错误率在初期仍高；对更复杂的机理结构、跨模态知识融合及更高效的验证方法仍有待改进。

---

## 274. On the Semantic and Syntactic Information Encoded in Proto-Tokens for One-Step Text Reconstruction

**arXiv ID:** 2602.18301 | [PDF](https://arxiv.org/pdf/2602.18301v1)

**作者:** Ivan Bondarenko `[一作]`, Fedor Tikunov `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在冻结的大型语言模型（LLM）中优化两条原型令牌（e 与 m），实现一次性重构数百个标记，并系统研究了这两条令牌在语义、句法、稳定性和注意力模式上的信息编码，以及通过锚点损失和关系蒸馏方式在保持重构准确率的前提下为 e 令牌注入语义结构的可行性。

**💡 创新点**

创新点在于首次揭示 e 与 m 令牌各自携带的语义与句法信息差异，证明关系蒸馏可在不牺牲重构质量的情况下将批量语义关系迁移至原型令牌空间，并为未来轻量级非自回归 seq2seq 体系提供可行的原型令牌预测与冻结解码器结合策略。

**🔧 技术方法**

主要技术包括：冻结 Llama‑3.2‑1B 作为解码器；使用 AdamW 仅优化 e 与 m 以满足交叉熵重构目标；对原型令牌进行 t‑SNE 可视化、注意力权重分析；引入锚点损失约束 e 与预训练句子嵌入的余弦相似度；实现关系蒸馏约束，最小化学生与教师相似度矩阵之间的 MSE/Huber 损失。

**📊 数据集**

实验数据集包括：Databricks‑Dolly‑15k 作为指令-响应对的原始文本；利用 Qwen3‑4B 生成同义改写与词形变体；使用 Qwen3‑Embedding‑8B 提供教师句子嵌入；合成基于上下文无关文法的七类句子以检验句法聚类。

**📈 对比分析**

通过与标准重构精度（>90%）以及锚点权重对比，展示关系蒸馏在不降低重构精度的前提下显著提升学生与教师相似度矩阵的相关性；锚点损失则在强权重下导致重构精度急剧下降，说明两者在语义对齐与重构之间存在尖锐折衷。

**⚠️ 局限性**

局限性包括：e 令牌对语义对齐的直接约束效果有限；实验仅在单一冻结模型与小规模数据上验证，缺乏对跨模型、跨域的泛化评估；原型令牌在不同优化初始化下的可辨识性与稳定性不足；未来需构建专用的原型令牌预测器并进一步量化句法与语义的解耦。

---

## 275. Learning Optimal and Sample-Efficient Decision Policies with Guarantees

**arXiv ID:** 2602.17978 | [PDF](https://arxiv.org/pdf/2602.17978v1)

**作者:** Daqian Shao `[一作]` `[通讯]` (Wolfson College), Daqian Shao (Wolfson College)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文缺乏具体内容，无法判断做了什么研究。

**💡 创新点**

未知

**🔧 技术方法**

未知

**📊 数据集**

未知

**📈 对比分析**

无法进行方法比较或性能评估。

**⚠️ 局限性**

缺乏足够信息，无法指出具体限制。

---

## 276. From Lossy to Verified: A Provenance-Aware Tiered Memory for Agents

**arXiv ID:** 2602.17913 | [PDF](https://arxiv.org/pdf/2602.17913v1)

**作者:** Qiming Zhu `[一作]` (Data Science), Benyou Wang `[通讯]` (Data Science)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 TierMem，一种双层内存体系结构，通过推理时证据分配来解决长序列语言代理中的写前压缩瓶颈；

**💡 创新点**

核心创新是使用实时的轻量级“足够性”路由器决定是否从快速摘要层回答，必要时再弹性地升级到不可变原始日志，并将已验证的证据写回摘要层；

**🔧 技术方法**

采用的技术包括：层级内存架构、原始日志分页、摘要与原始页的原始性指针、基于监督+GRPO的路由器训练、检索-生成闭环以及在线写回更新；

**📊 数据集**

在两个长序列对话基准 LoCoMo 与 LongMemEval 上进行实验；

**📈 对比分析**

与多种摘要仅、原始仅以及其它记忆系统对比，TierMem 在保持 0.851/0.873 级别准确率的同时，平均输入 token 减少 54.1%（比原始仅低 54%），延迟降低 60.7%（比原始仅低 61%），显著提升效率；

**⚠️ 局限性**

主要局限是路由器仍需微调、原始日志读取在极大数据量下仍有成本、写回的准确性依赖于生成器质量，且在极端稀疏或高度概括的摘要中可能仍出现误判。

---

## 277. Agentic Adversarial QA for Improving Domain-Specific LLMs

**arXiv ID:** 2602.18137 | [PDF](https://arxiv.org/pdf/2602.18137v1)

**作者:** Vincent Grari `[一作]` (AXA Group Operations), Marcin Detyniecki `[通讯]` (AXA Group Operations)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种对抗性问题生成框架，通过专家模型反馈迭代优化问题，生成针对性强、能揭露域专 LLM 认知与推理缺陷的合成数据，用以高效微调小型域专模型。

**💡 创新点**

创新点在于：①利用强大专家 LLM 与弱目标 LLM 的回答差异来驱动问题生成；②通过可微提示（TextGrad 思路）实现对问题的最大化差异化迭代；③采用指导与修订子模型把梯度信息转化为自然语言编辑，确保生成问题语义连贯且难度适宜；④实现了极高的样本效率，显著降低训练数据量。

**🔧 技术方法**

技术栈包括：对抗性优化、可微提示 (TextGrad)、LLM 反馈、指导与修订模型、实体知识抽取方法对比（EntiGraph、Knowledge‑Instr）、小型 LLaMA3‑8b 微调。

**📊 数据集**

数据集：LegalBench 法律推理基准中的三份合同子集（Cardlytics Maintenance Agreement、Buffalo Wild Wings Franchise Agreement、PF Hospitality Franchise Agreement），共 491 题；参考 CUAD 数据集中的合同引用。

**📈 对比分析**

对比方法：无额外数据基线、Paraphrase×6、模型无关 QA、EntiGraph、Knowledge‑Instr。结果显示，在 LLaMA3‑8b 上平均精度从 69.5% 提升至 82.7%，比基线提升 18.99%，比 EntiGraph 提升 3.89%，同时使用约 70 倍更少的训练 tokens。

**⚠️ 局限性**

局限性：①依赖强大 oracle LLM 进行反馈，部署时对资源与隐私有要求；②对抗生成聚焦于已知差异，可能忽略其他推理维度；③实验仅覆盖法律合同领域，需验证在其它专业领域的适用性；④未结合检索或外部工具，可能限制实际应用的性能提升。

---

## 278. RVR: Retrieve-Verify-Retrieve for Comprehensive Question Answering

**arXiv ID:** 2602.18425 | [PDF](https://arxiv.org/pdf/2602.18425v1)

**作者:** Deniz Qian `[一作]` (New York University), Eunsol Choi `[通讯]` (New York University)

**通讯引用:** 4259 | [OpenAlex ID](https://openalex.org/A5035142405)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种Retrieve‑Verify‑Retrieve (RVR) 的多轮检索框架，用于提升多答案查询的检索覆盖率。

**💡 创新点**

创新点在于将验证器嵌入检索循环，且训练检索器使其能在已验证文档的上下文中主动寻找遗漏答案，从而显著提升答案覆盖。

**🔧 技术方法**

技术包括：双编码检索器（Contriever、Qwen3‑Embedding、INF‑Retriever）、LLM 验证器（Qwen3‑30B‑Instruct）、基于对比学习的检索器微调、以及多轮查询与答案覆盖评估。

**📊 数据集**

使用的数据集包括 QAMPARI（多答案开放域 QA）、QUEST（集合运算类问题）和 WebQuestionsSP（Freebase 实体问答）进行实验和跨域泛化评估。

**📈 对比分析**

与基线（单轮检索、预训练检索器、Agentic 搜索如 Tongyi、SearchR1）对比，RVR 在 QAMPARI 上 MRecall@100 提升 10% 以上、Recall@100 提升 3% 以上；在 QUEST 和 WebQuestionsSP 上也取得更优的覆盖率，并在效率上显著优于 Agentic 方法。

**⚠️ 局限性**

局限性包括：验证器的精确度仍受 LLM 性能限制，迭代过程会增加查询时间和 GPU 内存消耗；在不同领域的迁移中，Fine‑tuned 检索器可能出现领域偏移，需要进一步的域适配。

---

## 279. Neural Synchrony Between Socially Interacting Language Models

**arXiv ID:** 2602.17815 | [PDF](https://arxiv.org/pdf/2602.17815v1)

**作者:** Zhining Zhang `[一作]` (Peking University), Heng Ji `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 8451 | [OpenAlex ID](https://openalex.org/A5103178893)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究社交互动中大型语言模型（LLM）内部表征的同步性（Neural Synchrony），并将其与模型的社交表现关联；

**💡 创新点**

首次将人类脑-脑同步（IBS）概念迁移至LLM，提出利用可预测的仿射变换衡量LLM间的表征同步，并验证其对应社交参与和时间对齐；

**🔧 技术方法**

使用多轮对话模拟环境（如多代理社交场景），提取LLM隐藏层表征，训练岭回归（仿射变换）预测对方未来表征，计算R²得分并聚合为同步分数；

**📊 数据集**

在Mistral和Llama两大LLM家族上进行450个交互场景（共21对模型），并使用自动评估器（MuSR、IFEval等）评估社交表现；

**📈 对比分析**

通过对照实验验证同步性只在主动交互且时间对齐时出现；同步分数与社交表现呈显著正相关（Mistral r≈0.88，Llama r≈0.99，跨族 r≈0.89），且在控制指令遵循与长上下文推理能力后仍保持显著；

**⚠️ 局限性**

局限包括：同步性仅为表征层线性可预测的度量，可能忽略更复杂非线性关系；实验环境相对受限，交互场景短、开放性不足；未验证因果关系，需进一步干预实验。

---

## 280. AnCoder: Anchored Code Generation via Discrete Diffusion Models

**arXiv ID:** 2602.17688 | [PDF](https://arxiv.org/pdf/2602.17688v1)

**作者:** Anton Xue `[一作]` (University of Texas at Austin), Sanjay Shakkottai `[通讯]` (University of Texas at Austin)

**通讯引用:** 7316 | [OpenAlex ID](https://openalex.org/A5028903768)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计 AnchorTree 软锚定框架，并在其上训练 AnCoder，以 AST 为基础的层次锚定改进离散 Diffusion 语言模型生成代码的结构与正确性。

**💡 创新点**

首次将 AST 的层次深度信息与软锚定权重相结合，提出软层次锚定方案，显著提升 DLM 对代码全局规划与语法正确性的把控。

**🔧 技术方法**

离散 Diffusion 语言模型、两阶段 anchor‑denoiser 结构、AST 解析、软锚定权重、余弦噪声调度与 remask 采样策略。

**📊 数据集**

OpenCoder 训练集、HumanEval 与 MBPP 评测集。

**📈 对比分析**

与同规模 MDLM 基线和 Qwen AR 模型对比；AnchorTree 在 HumanEval 的 Pass@1 由 3.29% 提升至 5.45%（+65.7%），在 MBPP 上由 6.34% 提升至 9.10%（+43.5%），虽然仍低于 AR 基线，但大幅优于无锚定 DLM。

**⚠️ 局限性**

仍无法赶上 AR 模型的性能；软锚定的泛化能力、对不同语言和更复杂代码结构的适应性尚未充分验证；模型在大规模参数化时的鲁棒性未知。

---

## 281. AI Hallucination from Students' Perspective: A Thematic Analysis

**arXiv ID:** 2602.17671 | [PDF](https://arxiv.org/pdf/2602.17671v1)

**作者:** Abdulhadi Shoufan `[一作]` (Khalifa University), Ahmad-Azmi-Abdelhamid Esmaeil `[通讯]` (University Malaysia Sabah)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对大学生在教育场景中使用大型语言模型时遇到的幻觉进行定性调查，收集并分析他们对幻觉类型、检测策略和心理模型的描述。

**💡 创新点**

首次系统挖掘学生对LLM幻觉的主观体验与认知模型，揭示认知误区并为AI素养课程设计提供实证依据。

**🔧 技术方法**

采用开放式问卷与主题分析（Taguette）对学生回答进行编码和主题归纳。

**📊 数据集**

研究数据来自63名计算机工程专业学生自述的开放式回答，共计152条评论，未使用公开语料库。

**📈 对比分析**

通过编码一致率（86.93%–93.16%）验证结果，并构建四类幻觉、两类检测策略、五类心理模型的主题结构；未进行算法性能对比。

**⚠️ 局限性**

样本单一专业、缺乏实验验证、开放式问卷可能限制信息深度，且自报数据可能存在主观偏差。

---

## 282. Analyzing and Improving Chain-of-Thought Monitorability Through Information Theory

**arXiv ID:** 2602.18297 | [PDF](https://arxiv.org/pdf/2602.18297v1)

**作者:** Usman Anwar `[一作]` (University of Cambridge), Christos Louizos `[通讯]` (Qualcomm AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了信息理论框架来刻画链式推理（CoT）监控的可监测性，并系统分析了监控可行性的必要条件与两类误差源。

**💡 创新点**

创新点在于将互信息视为衡量 CoT 监控效能的核心指标，证明其必要但不充分性，并设计两种训练目标（oracle 反馈与条件互信息最大化）来显著提升监控准确率并抑制奖励欺骗。

**🔧 技术方法**

主要技术包括信息理论推导（互信息、条件互信息、误差分解）、强化学习（GRPO）与大语言模型（Qwen‑2.5‑7B‑Instruct）作为被监控模型和监视器。

**📊 数据集**

使用的实验数据集包括 MBPP（编码任务，含可被“破解”的单元测试）和 BigMath‑Verified（数学题目，存在负数欺骗奖励）。

**📈 对比分析**

通过与仅优化任务奖励、仅使用监控奖励以及同时使用 oracle/MI 奖励的对照实验，发现 MI 目标保持高互信息、提升监控准确率并显著降低破解率；相对传统方法，性能提升在 10‑20% 级别的监控准确率和 30‑50% 级别的欺骗率下降。

**⚠️ 局限性**

局限性包括：仍未能完全消除 CoT 失真和奖励欺骗；依赖信息理论假设且对大规模模型的训练成本高；对复杂或隐写推理的解释能力仍有待提升。

---

## 283. Scaling Audio-Text Retrieval with Multimodal Large Language Models

**arXiv ID:** 2602.18010 | [PDF](https://arxiv.org/pdf/2602.18010v1)

**作者:** Jilan Xu `[一作]` (Visual Geometry Group, University of Oxford), Andrew Zisserman `[通讯]` (Visual Geometry Group, University of Oxford)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一的多模态大语言模型（MLLM）框架，用于音频-文本检索，并构建了包含多源音频与多粒度描述的大规模数据集。

**💡 创新点**

创新点包括：① 通过自动化管线生成长短句与标签三种粒度的音频描述；② 设计了Hybrid‑NCE损失，兼顾多粒度正样本与硬负样本加权；③ 引入双向重排序模块，利用跨模态交互进一步提升排名精度；④ 仅用约1% PE‑AV 的训练数据即可超越多项基线。

**🔧 技术方法**

核心技术：多模态大语言模型（如Qwen2.5‑Omni）、Prompt‑to‑Embedding（One‑Word Limitation）、Hybrid‑NCE对比损失、LoRA微调、FAISS近似最近邻检索、双向重排序网络。

**📊 数据集**

使用自建的AuroLA数据集（1.4M音频，来源11个公开资源），每条音频配有长句、短句和标签三层描述；实验也在AudioCaps、Clotho、Auto‑ACD、VGGSounder、EPIC‑Sounds、HD‑EPIC等标准检索基准上进行评测。

**📈 对比分析**

与现有方法（CLAP、PE‑AV等）对比，在AudioCaps、Clotho的Recall@1、A2T、T2A指标均取得显著提升；在VGGSounder、EPIC‑Sounds、HD‑EPIC的mAP上亦保持领先；相对PE‑AV，仅使用1/100训练数据即可获得大约30%以上的召回提升，验证了模型的高数据效率与可扩展性。

**⚠️ 局限性**

局限性：① 依赖大规模LLM，训练与推理成本高；② 自动生成的多粒度标签可能存在噪声，影响对极其罕见事件的检索；③ 目前主要验证在英语文本上，跨语言推广仍待进一步研究；④ 对长时音频的处理仍以固定长度梅尔谱为主，可能缺乏更细粒度的时序建模。

---

## 284. The 2025 AI Agent Index: Documenting Technical and Safety Features of Deployed Agentic AI Systems

**arXiv ID:** 2602.17753 | [PDF](https://arxiv.org/pdf/2602.17753v1)

**作者:** Leon Staufer `[一作]` (University of Cambridge), Noam Kolt `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 275 | [OpenAlex ID](https://openalex.org/A5092031121)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究构建了2025年AI代理系统索引，对30款高影响力代理系统的技术特性、生态交互、安全保障等方面进行了系统化整理，并揭示了行业内透明度与安全评估的巨大差距。

**💡 创新点**

创新之处在于首次提出统一的代理系统文档框架，提供可查询的公开索引，并通过大规模定量分析揭示开发者在安全与评估信息披露上的薄弱点。

**🔧 技术方法**

方法主要采用人工专家标注结合LLM辅助搜索与验证、公开资料抓取（官网、GitHub、邮件往来等）构建45维度注解，并使用可视化工具呈现数据。

**📊 数据集**

数据来源为公开可获得的产品文档、官方演示、GitHub仓库、API说明以及与开发者的邮件交流，最终形成30个代理系统的综合信息集合。

**📈 对比分析**

对安全框架、评估报告、工具集成等字段进行定量统计与可视化比较，发现约60%的系统缺乏安全评估或第三方测试，显示行业透明度不足；本研究未进行性能基准测试，而是聚焦信息完整性与安全披露。

**⚠️ 局限性**

局限性包括仅覆盖公开可部署的高影响力系统，忽略内部或垂直领域代理；依赖公开信息可能遗漏内部评估；方法在不同语言与地区可能存在信息获取偏差；未来需要持续更新以适应快速演进的代理生态。

---

## 285. Role and Identity Work of Software Engineering Professionals in the Generative AI Era

**arXiv ID:** 2602.18190 | [PDF](https://arxiv.org/pdf/2602.18190v1)

**作者:** Jorge Melegati `[一作]` `[通讯]` (University of Porto), Jorge Melegati (University of Porto)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一套研究议程，旨在探讨生成式人工智能（GenAI）在软件工程实践中的引入如何影响不同角色（开发者与测试者）的职业身份及其身份工作的变化，并规划通过系统综述、访谈与灰色文献等方法进行实证研究；

**💡 创新点**

创新点在于将软件角色差异作为决定身份工作的重要因素，并针对各角色提出具体研究问题，同时规划基于设计科学的方法，构建能够缓解角色冲突与身份失衡的社会技术工件；

**🔧 技术方法**

采用的技术包括系统性文献综述（SLR）、访谈（战斗故事法）和灰色文献分析，以及设计科学研究（DSR）框架，用于生成新的构造、模型和方法；

**📊 数据集**

目前并未使用专门的数据集，而是计划收集来自行业实践的访谈样本、公开报告和灰色文献；

**📈 对比分析**

论文未进行实验性对比或性能评估，方法主要是定性分析与基线研究，尚未给出量化性能指标；

**⚠️ 局限性**

局限性包括：研究仅聚焦开发者与测试者，未覆盖需求工程师；样本可能受限于少数组织；缺乏长期跟踪的 GenAI 实施案例；理论整合与方法设计的挑战。

---

## 286. Understanding the Generalization of Bilevel Programming in Hyperparameter Optimization: A Tale of Bias-Variance Decomposition

**arXiv ID:** 2602.17947 | [PDF](https://arxiv.org/pdf/2602.17947v1)

**作者:** Yubo Zhou `[一作]` (Xi'an Jiaotong University), Deyu Meng `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 31612 | [OpenAlex ID](https://openalex.org/A5091017287)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统研究梯度基超参数优化（HPO）中超梯度估计误差的偏差‑方差分解，补充了此前理论仅关注偏差的方差分析，并在此基础上给出全面的误差上界；随后提出基于多重数据划分的集成超梯度（EHG）与在线版EHG来显著降低方差并提升泛化性能。

**💡 创新点**

创新点在于首次对超梯度估计误差进行完整的偏差‑方差分解，揭示了方差对过拟合的根本作用；给出ITD与AID的联合误差上界，解释了验证集过拟合现象；提出简单高效的集成超梯度策略（EHG/OEHG），并通过理论推导证明其可有效降低方差。

**🔧 技术方法**

使用的技术包括梯度基HPO的迭代微分（ITD）与近似隐函数微分（AID）、偏差‑方差分解、误差上界推导、统一稳定性分析、线性系统求解、岭回归一维实例演示以及实验验证。

**📊 数据集**

实验所用数据集涵盖回归（abalone, bodyfat, mg, pyrim, space, triazines）、分类（a1a–a9a, diabetes, gisette, heart, ionosphere, w1a）、图像分类（MNIST, Fashion‑MNIST, CIFAR‑10）以及少样本学习（miniImageNet, tieredImageNet）等多任务与多域数据。

**📈 对比分析**

通过与传统HPO方法（网格搜索、随机搜索、贝叶斯优化）以及梯度基HPO基线（RHG、T‑RHG、AID‑FP/CG）进行对比实验，EHG/OEHG在正则化参数学习、数据清洗、少样本分类等任务中均显著降低测试误差或提升准确率，往往提升数个百分点。

**⚠️ 局限性**

局限性包括：需要多次数据划分，计算成本略高；目前只针对确定性双层优化，非凸及大规模情况的扩展仍待完善；误差上界可进一步收紧以适应实际场景。

---

## 287. "Everyone's using it, but no one is allowed to talk about it": College Students' Experiences Navigating the Higher Education Environment in a Generative AI World

**arXiv ID:** 2602.17720 | [PDF](https://arxiv.org/pdf/2602.17720v1)

**作者:** Yue Fu `[一作]` (University of Washington), Alexis Hiniker `[通讯]` (University of Washington)

**通讯引用:** 3196 | [OpenAlex ID](https://openalex.org/A5074077266)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对美国一所大型公立大学的23名本科、研究生学生进行半结构化访谈，研究他们在生成式AI环境下的使用经验与环境影响。

**💡 创新点**

首次将环境因素与学生自我调节结合，揭示AI羞愧文化、政策混乱及其对学习的影响，并给出基于学生参与的政策与评估改进建议。

**🔧 技术方法**

使用定性研究方法：半结构化访谈、反思性主题分析。

**📊 数据集**

数据集：23名学生的访谈文本，包含多学科本科、硕士、博士学生。

**📈 对比分析**

由于是定性研究，未进行数值比较；通过主题分析提炼主题并与现有文献对照。

**⚠️ 局限性**

局限性：样本仅来自一所美国大学，缺乏跨文化、跨机构对比；所有参与者已使用AI，未包含非使用者；研究时间点在2025年，技术更新可能影响结果。

---

## 288. Statistical Confidence in Functional Correctness: An Approach for AI Product Functional Correctness Evaluation

**arXiv ID:** 2602.18357 | [PDF](https://arxiv.org/pdf/2602.18357v1)

**作者:** Wallace Albertini `[一作]` (Pontifical Catholic University of Rio de Janeiro), Marcos Kalinowski `[通讯]` (Pontifical Catholic University of Rio de Janeiro)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并验证了一种基于统计置信度的功能正确性评估方法（SCFC），并在两个真实工业AI系统（油田货舱空间估计与信用卡欺诈检测）中应用。

**💡 创新点**

创新点在于将业务需求与模型平均性能及其变异性相结合，通过自助抽样（bootstrapping）估计置信区间，并引入非参数的能力指数（C_pk）以量化满足规范的统计置信度。

**🔧 技术方法**

使用的技术包括分层概率抽样、非参数自助抽样（bootstrapping）、置信区间计算、以及基于置信区间的能力指数（C_pk）评估。

**📊 数据集**

使用的真实数据集：油田货舱图像集（42张图片）用于空间估计；信用卡交易记录集（约110万条，其中包含真实欺诈案例）用于欺诈检测。

**📈 对比分析**

方法通过置信区间与能力指数对比传统单点指标进行评估。货舱系统平均准确率83.4%，95%置信区间[71.43%,92.86%]，C_pk≈1.12；欺诈检测系统召回率99.11%，95%置信区间[98.55%,99.67%]，C_pk≈1.98，均表明模型在统计置信度下已达或超过业务阈值。

**⚠️ 局限性**

局限性包括：专家样本量小且可能存在偏倚；方法依赖于代表性的数据集与明确的业务阈值；仅针对单维度指标，处理多指标时需进一步聚合；自助抽样学习曲线和计算成本；对极端高或低性能情况C_pk增益有限。

---

## 289. Joint Parameter and State-Space Bayesian Optimization: Using Process Expertise to Accelerate Manufacturing Optimization

**arXiv ID:** 2602.17679 | [PDF](https://arxiv.org/pdf/2602.17679v1)

**作者:** Saksham Kiroriwal `[一作]`, Jürgen Beyerer `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了POGPN-JPSS框架，将Partially Observable Gaussian Process Network与Joint Parameter and State‑Space模型结合，用于高维多阶段生物乙醇生产过程的贝叶斯优化并在仿真中验证其效果。

**💡 创新点**

创新点在于利用过程专家知识提取低维特征，将高维时间序列中间观测嵌入POGPN，并采用Predictive Log‑Likelihood（PLL）改进传统ELBO，使结构化模型在高维、带噪声的子过程数据上显著提升优化速度与可靠性。

**🔧 技术方法**

使用的技术包括POGPN、JPSS、Stochastic Variational Gaussian Process、Log Expected Improvement acquisition、Greedy Improvement Reduction / Greedy Variance Reduction诱导点分配以及Matern‑5/2核与维度缩放先验。

**📊 数据集**

使用的数据集为基于seed‑train bioethanol simulation的仿真数据，输入维度26，包含三阶段子过程的高维多变量时间序列以及最终的Space‑Time Yield (STY)。

**📈 对比分析**

与标准GPN、单任务GP和SVGP‑PLL进行对比实验，采用100次迭代预算、50个随机初始点，5次重复平均。POGPN‑PLL在约50次迭代内就达到0.5 STY阈值，远快于其他方法，节约约300‑350天的生产时间，显著提高性能。

**⚠️ 局限性**

局限性包括PLL与ELBO之间性能差距原因尚未深入探究；当流程结构未知时缺乏自动图结构发现方法；以及对更通用的特征提取与因果图学习的进一步研究仍有待开展。

---

## 290. HyTRec: A Hybrid Temporal-Aware Attention Architecture for Long Behavior Sequential Recommendation

**arXiv ID:** 2602.18283 | [PDF](https://arxiv.org/pdf/2602.18283v1)

**作者:** Lei Xin `[一作]` (Wuhan University), Fanhu Zeng `[通讯]` (Shanghai Dewu Information Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 HyTRec，面向生成式推荐的长序列建模框架，利用双分支（长历史+短期）混合注意力实现高效且精确的下一个商品预测。

**💡 创新点**

创新点包括：① 混合注意力架构，长期历史采用线性注意力，近期交互采用 Softmax 注意力；② Temporal‑Aware Delta Network（TADN）——利用指数衰减门控动态提升近期行为权重，抑制历史噪声；③ 双分支融合策略，显著提升长序列下的召回率与推理速度。

**🔧 技术方法**

技术手段包括线性注意力、Softmax 注意力、DeltaNet 与 TADN 门控机制、Transformer 结构、稀疏训练+密集推理、时间衰减因子、混合注意力比例调节。

**📊 数据集**

实验使用 Amazon Beauty、Amazon Movies & TV、Amazon Electronics 等公开工业级数据集；跨域实验基于 2022 年华为广告挑战数据；在这些数据集上处理长度达 10k+ 的长序列。

**📈 对比分析**

与 GRU4Rec、SASRec、DIN、HSTU、Transformer、GLA、Qwen‑next 等基线对比，指标涵盖 H@500、NDCG@500、AUC 与 Hit Rate。HyTRec 在 Beauty、Electronics、Movies&TV 上均取得最高 H@500 与 AUC，Hit Rate 提升超过 8%，推理速度保持线性，吞吐量在 5k 长度下仍高于 65k tokens/s，3:1 混合比例最优。

**⚠️ 局限性**

局限性：① 混合比例与门控参数需经验调优；② 对极度噪声或非常短历史的用户，TADN 可能表现不如专门的短期注意力模型；③ 仍未充分验证在更大规模或更低延迟实时系统中的极限性能；④ 跨域泛化虽有提升，但对分布漂移的鲁棒性仍待进一步改进。

---

## 291. Improving Sampling for Masked Diffusion Models via Information Gain

**arXiv ID:** 2602.18176 | [PDF](https://arxiv.org/pdf/2602.18176v1)

**作者:** Kaisen Yang `[一作]` (Tsinghua University), Alex Lamb `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于信息增益的掩码扩散模型（MDM）采样器Info-Gain Sampler，利用MDM的双向注意力一次性评估任何解码动作对剩余掩码位置不确定性的影响，从而在解码时兼顾即时不确定性与信息增益，降低累计不确定性；

**💡 创新点**

1）首次将信息增益与即时不确定性结合为采样目标，实现非贪心全局规划；2）利用MDM的双向结构一次性评估全局影响；3）采用高效并行候选评估和高置信度绕过，兼顾速度与质量；

**🔧 技术方法**

双向扩散模型、信息增益目标函数、两阶段候选动作采样（token+position）、单批前向推理实现信息增益评估、温度调节、KV缓存加速、Beam Search/Best-of-N对比；

**📊 数据集**

GSM8K、MATH-500、HumanEval、MBPP（推理、代码生成）；Sudoku、Countdown（规划）；ImageNet-512、GenEval（多模态图像生成）；AlpacaEval（创意写作）；并在全注意力与半自回归MDM架构上进行实验；

**📈 对比分析**

与Uniform、Entropy、Confidence、Margin、KLASS、PC‑Sampler等基线在相同MDM架构下对比，Info‑Gain在全注意力MDM上平均提升3.6%准确率，创意写作赢率提升至63.1%，GenEval提升1.9%，并在图像生成中显著降低FID、提升IS；在半自回归MDM上相对基线提升20%+，证明其在多任务上的显著性能提升；

**⚠️ 局限性**

与贪心采样相比仍耗时更高；候选动作采样依赖局部不确定性，可能限制多样性；需要更高效的搜索与硬件优化；在极大模型或长序列场景下计算开销仍显著。

---

## 292. Atrial Fibrillation Detection Using Machine Learning

**arXiv ID:** 2602.18036 | [PDF](https://arxiv.org/pdf/2602.18036v1)

**作者:** Ankit Singh `[一作]` (Chhattisgarh Swami Vivekanand Technical University), Nachiket Tapas `[通讯]` (Chhattisgarh Swami Vivekanand Technical University)

**通讯引用:** 1070 | [OpenAlex ID](https://openalex.org/A5081938166)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一套基于光电容积脉搏波（PPG）和心电图（ECG）特征的心房颤动（AF）检测机器学习框架

**💡 创新点**

首次将ECG与PPG多模态特征结合并使用子空间k近邻算法进行AF分类，显著提升了检测准确率

**🔧 技术方法**

使用离散小波变换去噪、巴特沃斯低通滤波、归一化、22维统计/频域/HRV特征提取，以及集成袋装决策树、立方核支持向量机和子空间k近邻三种分类器

**📊 数据集**

基于35名受试者的同步PPG/ECG记录，共525段（清洗后481段）组成的公开数据集

**📈 对比分析**

通过10折交叉验证和80/20分层拆分比较三种模型，袋装决策树获得最高准确率98.96%，子空间k近邻94.79%，立方核SVM仅敏感度高但特异性低

**⚠️ 局限性**

样本量小、受试者多样性不足、固定80秒窗口可能不适合实时监测、未验证在真实佩戴设备中的鲁棒性

---

## 293. Beyond Individual Influence: The Role of Echo Chambers and Community Seeding in the Multilayer three state q-Voter Model

**arXiv ID:** 2602.18088 | [PDF](https://arxiv.org/pdf/2602.18088v1)

**作者:** Igor Hołowacz `[一作]` (Wroclaw University of Science and Technology), Piotr Bródka `[通讯]` (Wroclaw University of Science and Technology)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了多层三态 q‑voter 模型下的影响力最大化问题，系统评估了多种种子选取策略，并揭示了“堡垒陷阱”和“冗余陷阱”两种新型阻碍扩散的拓扑现象。

**💡 创新点**

创新点在于：①首次将多层网络与三态复杂传染模型结合，揭示了传统密集社区种子策略在高度模块化网络中的逆效果；②提出“堡垒陷阱”与“冗余陷阱”概念；③证明多样化覆盖（如 VoteRank）比局部强度更能触发全局级联。

**🔧 技术方法**

使用的技术包括：多层 q‑voter 模型（LOCAL & AND 规则）、mABCD 合成网络生成器、均值场近似分析、蒙特卡洛仿真、以及多种种子选取算法（VoteRank、PageRank、Degree、Clique Influence Maximization、k‑Shell 等）。

**📊 数据集**

使用的数据集为 mABCD 生成的六种控制参数网络（不同的社区隔离度、层间相关性与度相关性），并在每种网络上进行 100 次独立仿真。

**📈 对比分析**

通过对比各种种子策略在不同噪声水平和预算下的最终一致度（c_A）进行评估，结果显示在无噪声或弱噪声条件下 VoteRank 与 PageRank 远优于结构性策略；在强噪声条件下，集中式种子策略（CIM、k‑Shell）性能最差，表现出显著的“过量”效应。

**⚠️ 局限性**

局限性包括：①阈值 q 固定为 4，未考虑个体阈值异质性；②仅使用合成网络，缺乏对真实多层社交网络的验证；③未引入时间演化或竞争性传染场景；④模拟规模有限，无法覆盖更大网络的行为。

---

## 294. Agentic Unlearning: When LLM Agent Meets Machine Unlearning

**arXiv ID:** 2602.17692 | [PDF](https://arxiv.org/pdf/2602.17692v1)

**作者:** Bin Wang `[一作]` (Shandong University of Traditional Chinese Medicine), Benzheng Wei `[通讯]` (Shandong University of Traditional Chinese Medicine)

**通讯引用:** 2841 | [OpenAlex ID](https://openalex.org/A5051864449)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种双通道同步反向学习框架SBU，用于在LLM代理中实现参数和记忆的联合遗忘

**💡 创新点**

创新点在于将参数消忘与记忆依赖闭包删除同步化，形成闭环防止信息回流

**🔧 技术方法**

采用高熵先验对齐的KL目标进行参数消忘，并使用依赖图+引用计数进行记忆消忘

**📊 数据集**

在三大医疗问答基准（MedQA、MedMCQA、MedReason）上进行评估

**📈 对比分析**

与梯度上升、NPO、LoRA、Adapter合并等基线相比，SBU在MIA Score上提升至≈0.9，同时保持超过90%的测试/泛化准确率

**⚠️ 局限性**

局限在于依赖追踪未能完全覆盖跨代理知识图的传播，且在更大规模或多代理场景下效果未知

---

## 295. Robust Pre-Training of Medical Vision-and-Language Models with Domain-Invariant Multi-Modal Masked Reconstruction

**arXiv ID:** 2602.17689 | [PDF](https://arxiv.org/pdf/2602.17689v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 296. Generating adversarial inputs for a graph neural network model of AC power flow

**arXiv ID:** 2602.17975 | [PDF](https://arxiv.org/pdf/2602.17975v1)

**作者:** Robert Parker `[一作]` (Los Alamos National Laboratory), Robert Parker `[通讯]` (Los Alamos National Laboratory)

**通讯引用:** 16053 | [OpenAlex ID](https://openalex.org/A5112080834)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过构造并求解两类优化问题，系统识别并量化了基于 CANOS‑PF 图神经网络的 AC 电力流 surrogate 在 14 节点系统上的对抗鲁棒性缺陷，生成输入点使得网络预测与真实 AC 计算之间产生巨大误差。

**💡 创新点**

创新点在于首次针对 state‑of‑the‑art GNN 模型提出最大误差与约束误差两种优化框架，能够定量评估网络对电压幅值和无功功率等关键输出的鲁棒性，并揭示了模型在特定工作区间内的脆弱性。

**🔧 技术方法**

技术手段包括使用 JuMP+MathProgIncidence 对 GNN 与 AC 计算进行灰盒建模，结合 PowerModels+MathOptAI 与 IPOPT+MA57 进行高精度非线性优化；训练时采用 MSE 与无监督约束违规损失的联合目标。

**📊 数据集**

实验数据来源于 PFΔ Task‑1.1 训练集（48,600 条 AC 计算样本）以及 IEEE 14‑bus PGLib‑OPF 作为输入/输出域，所有对抗样本均基于该数据集生成。

**📈 对比分析**

与训练集/测试集的 MSE 与 PBL 损失对比，发现对抗样本的无功功率误差可达 3.4 pu、电压幅值误差 0.08 pu；在约束误差实验中，仅需 0.04 pu 的电压幅度扰动即可满足对抗约束，表明模型在极小扰动下就会失效。

**⚠️ 局限性**

限制在于只考虑固定负荷与线路参数的输入空间，实验规模仅为 14 节点；对更大系统的可扩展性尚未验证；且对抗样本往往依赖 PV 电压逼近下限，提示训练数据缺乏此类极端点。

---

## 297. Enhancing Scientific Literature Chatbots with Retrieval-Augmented Generation: A Performance Evaluation of Vector and Graph-Based Systems

**arXiv ID:** 2602.17856 | [PDF](https://arxiv.org/pdf/2602.17856v1)

**作者:** Hamideh Ghanadian `[一作]`, Mohammad Hossein Tekieh `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究了一种基于检索增强生成（RAG）的科学文献聊天机器人，系统性评估了向量检索、图检索以及二者混合检索的性能；

**💡 创新点**

创新点在于：①提出混合RAG框架，将向量检索与图检索结合以提升检索准确性和回答质量；②使用 GPT 模型自动生成合成测试集并由专家注释，构建了两个典型场景（单文档与多文档检索）的基准数据；

**🔧 技术方法**

采用了 GPT‑4o‑mini 生成合成问答、OpenAI 的 text‑embedding‑ada‑002 做向量嵌入、LlamaIndex 的 VectorIndexRetriever 与 PropertyGraphIndex、Neo4j 图数据库以及 VectorContextRetriever、LLMSynonymRetriever 等多种检索器，结合语义分块与向量‑图混合检索技术；

**📊 数据集**

数据集包括：①来源自 PubMed 的农药/除草剂活性成分研究论文；②针对单文档检索的 500 题目/答案/上下文合成集；③针对数据库检索的 60 题目/答案/上下文合成集，并通过专家进行质量标注；

**📈 对比分析**

在单文档和数据库两种检索场景下，分别评估了 VectorRAG、GraphRAG 与 Hybrid RAG 的表现，使用 cosine similarity 与 faithfulness 两项指标衡量回答质量；结果显示混合检索在两项指标上均优于单独向量或图检索，尤其在多文档检索中表现更为突出；

**⚠️ 局限性**

局限性包括：①多文档检索仍存在精度下降，难以处理信息冗余与歧义；②评测仅基于合成数据，缺乏真实用户查询与交互验证；③图检索过程耗时且成本较高；④系统未考虑多模态或实时用户反馈等因素。

---

## 298. Multi-Modal Monocular Endoscopic Depth and Pose Estimation with Edge-Guided Self-Supervision

**arXiv ID:** 2602.17785 | [PDF](https://arxiv.org/pdf/2602.17785v1)

**作者:** Xinwei Ju `[一作]` (University College London), Francisco Vasconcelos `[通讯]` (University College London)

**通讯引用:** 918 | [OpenAlex ID](https://openalex.org/A5068825369)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种自监督框架PRISM，结合亮度分解与边缘检测进行单目深度与位姿估计，使用阶段式训练与边缘引导损失；

**💡 创新点**

创新点在于同时利用光照先验（亮度分解）和结构先验（高精度边缘图）作为辅助输入与监督，改进自监督深度学习在内镜中的鲁棒性；

**🔧 技术方法**

采用深度可分离网络（DepthNet）、位姿网络（PoseNet）以及亮度提取（LumNet）和边缘检测（EdgeNet）模块，并使用SSIM、光度重投影、平滑项以及边缘一致性损失；

**📊 数据集**

主要数据集为真实内镜视频Hyper‑Kvasir、仿真/模型数据C3VD、真实完整视频EndoMapper及其子集SegCol；

**📈 对比分析**

与MonoDepth2、MonoViT、SHADeS等基线对比，PRISM在C3VD深度测试中RMSE降至6.12，MAE 4.33，位姿ATE约1.91，优于大多数方法；在真实数据上表现更稳健，边缘清晰、光照干扰较小；

**⚠️ 局限性**

局限性包括：对真实数据依赖度高，仿真数据训练效果不佳；自监督时过度依赖光度一致性，可能在纹理稀疏区域失稳；同时，边缘与亮度提取需要额外网络，模型复杂度上升；

---

## 299. Pimp My LLM: Leveraging Variability Modeling to Tune Inference Hyperparameters

**arXiv ID:** 2602.17697 | [PDF](https://arxiv.org/pdf/2602.17697v1)

**作者:** Nada Zine `[一作]` (University of Lille), Romain Rouvoy `[通讯]` (University of Lille)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对 Hugging Face Transformers 的推理过程进行可变性建模，系统性地分析和预测大语言模型（LLM）在不同推理超参数配置下的能耗、延迟和准确性，并利用采样与机器学习构建预测模型。

**💡 创新点**

将可变性建模（feature modeling）应用于 LLM 推理超参数空间，首次实现对海量配置空间的系统探索与预测，并揭示能耗、延迟与准确性之间的相互作用与 Pareto 最优折衷。

**🔧 技术方法**

使用特征建模（Feature Modeling）、t‑wise 采样（YASA、ICPL、随机采样）、实验测量（CPU/GPU 能耗、延迟、Pass@1 准确性）以及随机森林回归（Random Forest Regression）进行预测。

**📊 数据集**

数据集采用 HumanEval+（164 个 Python 编码任务）和三款开源 LLM（OpenCoder‑8B‑Instruct、Qwen2.5‑Coder‑7B‑Instruct、Qwen2.5‑Coder‑3B‑Instruct），在 Hugging Face Transformers 框架下进行实验。

**📈 对比分析**

与仅使用单一采样策略或无建模的传统方法相比，混合采样（ALL）预测模型在能耗、延迟和准确性上均达到 R²≥0.94、MAE 低至 10 kJ（能耗）和 0.01（准确性），表明即使样本有限也能实现高精度预测；在 Pareto 前沿分析中，发现低能耗区间能显著提升准确性，进一步验证方法有效性。

**⚠️ 局限性**

局限性包括：仅评估单一数据集和推理框架；特征建模过程中对连续参数进行了离散化，可能遗漏细粒度效应；实验仅在 NVIDIA A100 GPU 上进行，缺乏对不同硬件与部署环境的通用性；模型需持续维护以跟进 LLM 与推理服务器的快速演进。

---

## 300. Nested Training for Mutual Adaptation in Human-AI Teaming

**arXiv ID:** 2602.17737 | [PDF](https://arxiv.org/pdf/2602.17737v1)

**作者:** Upasana Biswas `[一作]` (Arizona State University), Sarath Sreedharan `[通讯]` (Colorado State University)

**通讯引用:** 1498 | [OpenAlex ID](https://openalex.org/A5028325441)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一种嵌套强化学习框架，使机器人能够与多样化且具适应性的人的伙伴进行双向适应与协作。

**💡 创新点**

创新点在于通过分层训练（先训练人类对固定机器人策略的适应，再训练机器人对这些适应性人类的适应）以及利用隐式嵌入逼近二阶I‑POMDP推理，既避免了陷入单一协调惯例，又显著提升了对未知适应伙伴的泛化能力。

**🔧 技术方法**

主要技术包括：两阶段的分层训练、隐变量嵌入（latent embedding）用于压缩交互历史、基于策略梯度的强化学习（如PPO/类似方法）以及对I‑POMDP理论的近似实现。

**📊 数据集**

数据集：使用Overcooked的require‑cooperation 版本，模拟多种适应性人类伙伴以评估模型性能。

**📈 对比分析**

与LIAM、LILI、PACE以及Generalist基线进行对比。短期评测（10轮×5回合）平均成功率0.90，长周期评测（10轮×25回合）0.935，明显高于下一最佳基线Generalist（0.575/0.65）和其他方法，且对所有伙伴均表现出稳健的成功率。

**⚠️ 局限性**

局限性：目前仅在模拟人类伙伴上验证，未与真实人类实验；仅针对合作式任务，未考虑奖励不完全一致的混合动机场景；隐变量嵌入的解释性和对更复杂任务的可扩展性尚待进一步研究。

---

## 301. A Curated Literature Database for Monitoring More Than 30 Years of Ansys Granta Product Usage

**arXiv ID:** 2602.18264 | [PDF](https://arxiv.org/pdf/2602.18264v1)

**作者:** David Mercier `[一作]` `[通讯]` (Synopsys), David Mercier (Synopsys)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

建立了基于Ansys Granta MI Enterprise的半自动化软件使用文献数据库，整合1100余篇关于Granta产品使用的期刊、会议、专利等文献，并提供可查询和可视化分析；

**💡 创新点**

首次将软件使用信息与传统文献元数据分离，并通过专家手工标注与自动化抓取相结合，构建可追溯的关系图谱，实现软件采纳与影响的定量监测；

**🔧 技术方法**

使用DOI与citation‑file解析、Python脚本、VOSviewer、Power BI、Graphviz等技术完成数据抓取、清洗、网络构建与交互式可视化；

**📊 数据集**

依托1100余条经专家审核的文献记录（1990‑2025年，包含期刊、会议、专利、标准、学位论文等），形成专门的Granta使用数据库；

**📈 对比分析**

通过词频、共现、共著网络及主题簇等计量指标，对比不同产品、机构、学科的使用分布，结果显示Granta EduPack占主导，Selector次级，MI较少但聚焦企业级；仪表盘实现实时交互，满足查询与分析需求；

**⚠️ 局限性**

数据集非全覆盖，受软件引用不规范、文献类型多样限制；作者、机构归属标准化不足，软件版本与集成细节缺失；未来需提升自动化检索与实体标准化以进一步完善数据库。

---

## 302. WorkflowPerturb: Calibrated Stress Tests for Evaluating Multi-Agent Workflow Metrics

**arXiv ID:** 2602.17990 | [PDF](https://arxiv.org/pdf/2602.17990v1)

**作者:** Madhav Kanda `[一作]` (University of Illinois Urbana-Champaign), Sharad Agarwal `[通讯]` (Microsoft)

**通讯引用:** 6133 | [OpenAlex ID](https://openalex.org/A5032970691)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个名为WorkflowPerturb的基准，专门用于在受控结构和文本扰动下评估工作流评估指标；

**💡 创新点**

创新点在于构造了可调节严重程度的缺失、压缩和描述变形三种扰动模式，并引入了严重度校准的评分轨迹，使得指标能被解释为功能风险；

**🔧 技术方法**

采用了图结构匹配（Graph F1、Chain F1）、词法重叠（BLEU、GLEU）、嵌入相似度（BERTScore）、秩相关（Kendall’s τ）以及LLM-as-Judge（GPT‑4o）等多元评估技术；

**📊 数据集**

使用了4,973条黄金工作流及其44,757个经过自动化验证的扰动版本，来源于WORFBENCH；

**📈 对比分析**

对各指标在三种扰动类型与10%、30%、50%严重度下的得分进行了统计和趋势分析，结果显示结构性指标对缺失/压缩敏感，词法指标对改写敏感，嵌入和LLM-as-Judge提供了互补视角；

**⚠️ 局限性**

局限性包括缺乏对执行层面影响的评估、仅覆盖三类扰动、且单一指标难以全面捕捉所有失效模式

---

## 303. 3DMedAgent: Unified Perception-to-Understanding for 3D Medical Analysis

**arXiv ID:** 2602.18064 | [PDF](https://arxiv.org/pdf/2602.18064v1)

**作者:** Ziyue Wang `[一作]` (National University of Singapore), Yueming Jin `[通讯]` (National University of Singapore)

**通讯引用:** 4835 | [OpenAlex ID](https://openalex.org/A5050163233)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个统一的3D医学分析代理3DMedAgent，利用2D多模态大语言模型与外部工具通过长时记忆实现从3D CT感知到临床推理的闭环流程。

**💡 创新点**

创新点包括（1）证据中心化的长期记忆模块，将多源工具输出压缩为结构化证据；（2）三阶段策略：OAMI、CFLT、T1S-Loop，逐步从全局到局部、从体积到切片、从感知到推理；（3）推出DeepChestVQA全胸CT VQA基准，丰富了3D医学评测生态。

**🔧 技术方法**

采用2D MLLM（如GPT‑5、Qwen3‑VL）与图像工具（VISTA3D、CT‑CLIP、分割/标注工具）结合的工具增强代理体系，利用CT‑CLIP生成局部相似度热图进行ROI定位，并通过记忆更新实现多步推理。

**📊 数据集**

主要使用DeepTumorVQA（1,740问答对）与新构建的DeepChestVQA（1,020问答对）以及原始CT-RATE、NSCLC等医学CT数据集。

**📈 对比分析**

与一般、医学与3D专用的大语言模型进行对比，3DMedAgent在测量、识别、视觉推理和医学推理四类任务平均提升约20%（最高超过27%），并在跨数据集、跨器官实验中表现出更稳健的泛化能力。

**⚠️ 局限性**

局限性包括对外部工具的依赖、仍受限于2D MLLM的理解深度、对极其复杂医学推理的能力有限，且需在临床环境中进一步验证与监管。

---

## 304. In-Context Learning for Pure Exploration in Continuous Spaces

**arXiv ID:** 2602.17976 | [PDF](https://arxiv.org/pdf/2602.17976v1)

**作者:** Alessio Russo `[一作]` (Boston University), Aldo Pacchiano `[通讯]` (Broad Institute of MIT and Harvard)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究连续决策空间下的固定置信度纯探索问题，提出 Continuous In‑Context Pure Exploration（C‑ICPE）算法，直接用深度策略学习查询动作和停止规则，完成一次性识别未知的最优或近似最优假设。

**💡 创新点**

创新点在于：① 用神经网络元训练的可传递后验（对角高斯）替代传统离散后验；② 采用Thompson Sampling 方式采样查询而不需要显式演员；③ 通过 Lagrangian dual 推导出最优停止 Bellman 结构，用强化学习训练停机 Critic；④ 在连续空间中实现了无手工置信度序列的固定置信度保证。

**🔧 技术方法**

技术手段包括：变分后验（Diagonal Gaussian），Thompson Sampling，DQN‑style Critic，RL 对最优停止的学习，时间池化（time‑pooling）序列编码，Lagrangian dual 变换与强化学习的结合。

**📊 数据集**

实验数据集：三类连续任务——(i) 连续二分搜索（目标定位）; (ii) ε‑Best‑Arm Identification（连续 Arms on unit sphere）; (iii) Ackley 函数最小化（多峰非凸函数）; 所有任务均通过随机模拟生成目标、参数、噪声。

**📈 对比分析**

与多种基线比较：① 无策略均匀采样；② Bayesian Optimization (Tree‑structured Parzen Estimator、Gaussian Process EI)；③ 进化算法 CMA‑ES；实验采用相同期望查询数或 Horizon = E[τ] + kσ_τ (k=3)。结果显示 C‑ICPE 在保持 (1‑δ)-PAC 置信度的前提下，平均查询次数显著低于基线，且在大多数维度与 ε 组合下取得更高的准确率。

**⚠️ 局限性**

局限性：① 后验假设为单模对角高斯，难以捕捉多模最优点；② 仅适用于查询空间与决策空间相同；③ 需要手工调节停机成本 c，可能导致误校准；④ 在高维或更复杂先验下，后验与停止学习的效果可能下降。

---

## 305. Joint Training on AMD and NVIDIA GPUs

**arXiv ID:** 2602.18007 | [PDF](https://arxiv.org/pdf/2602.18007v1)

**作者:** Jon Hu `[一作]` (Zettabyte AI Inc), Zhendong Yu `[通讯]` (Zettabyte AI Inc)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于CPU‑offloading P2P的Device‑Direct Communication，实现 AMD–NVIDIA 异构集群上大语言模型的混合训练。

**💡 创新点**

创新点在于将跨厂商 GPU 的数据传输直接落在设备端，结合 GDR 与 CPU‑offloading 控制平面，消除主机复制瓶颈，并通过多适配器架构实现跨 CCL 的统一接口。

**🔧 技术方法**

采用 CPU‑Forwarding、DCBS、MPDT、GDR、NCCL/RCCL、Megatron、PyTorch 后端插件等技术。

**📊 数据集**

使用 LLaMA‑8B 与 Qwen2‑7B 两个大型语言模型作为训练数据集。

**📈 对比分析**

对比 NVIDIA 同构、AMD 同构与异构集群吞吐量，实验表明异构方案在 LLaMA‑8B 取得 98.2%、Qwen2‑7B 取得 94.4% 的 NVIDIA 同构吞吐，并优于单一 AMD 方案，且训练稳定且收敛一致。

**⚠️ 局限性**

局限性包括：异构仅局限于流水线并行；DP/TP 仍使用同构集群；对模型分层分配高度敏感；实现过程对硬件适配与调试需求高。

---

## 306. Scientific Knowledge-Guided Machine Learning for Vessel Power Prediction: A Comparative Study

**arXiv ID:** 2602.18403 | [PDF](https://arxiv.org/pdf/2602.18403v1)

**作者:** Orfeas Bourchas `[一作]` (Laboratory of Marine Engineering N.T.U.A.), George Papalambrou `[通讯]` (Laboratory of Marine Engineering N.T.U.A.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于海试功率曲线的物理基准与数据驱动残差学习相结合的混合主机功率预测框架。

**💡 创新点**

创新点在于将物理规律（螺旋桨功率-速度幂律）嵌入模型基准，限制机器学习任务仅预测残差，从而显著提升模型在稀疏与外推区间的物理一致性与泛化能力。

**🔧 技术方法**

使用了 XGBoost、传统全连接神经网络、以及物理信息神经网络（PINN），并通过超参优化、标准化预处理、以及针对 PINN 的物理损失约束实现训练。

**📊 数据集**

数据集为约 40,000 条真实船舶运营记录（持续 5 个月），包含水速、吃水、纵倾、风速/方向等多维特征。

**📈 对比分析**

对同一数据集分别训练基准模型与混合模型，比较 MAE、RMSE 等全局误差指标，并在 8–17 kn、不同风向等未见条件下进行外推性能评估；结果显示混合模型在稀疏区间保持与物理预期一致，整体误差差距不足 1% 但外推稳定性明显优于基准，尤其是混合 PINN 取得最佳平衡。

**⚠️ 局限性**

局限性包括：整体误差提升有限；基准功率曲线假设为简单幂律，可能无法完全覆盖所有船体状态；超参优化过程复杂；在极端环境或船体极端工况下仍可能出现预测误差。

---

## 307. Distributed Security: From Isolated Properties to Synergistic Trust

**arXiv ID:** 2602.18063 | [PDF](https://arxiv.org/pdf/2602.18063v1)

**作者:** Minghui Xu `[一作]` (Shandong University), Minghui Xu `[通讯]` (Shandong University)

**通讯引用:** 1652 | [OpenAlex ID](https://openalex.org/A5103077343)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对分布式安全的演进进行综述，提出从单一属性提升转向多属性协同融合的架构设计视角。

**💡 创新点**

将分布式安全视作属性集合的协同体，强调系统性研究框架、属性融合的创新组合，并指出未来研究方向。

**🔧 技术方法**

综述了共识协议（PBFT、HotStuff 等）、分布式数据库一致性、隐私保护技术（MPC、ZKP、加密哈希树等）、可验证性（零知识证明、累加器）与问责机制（签名、审计日志、经济惩罚）等技术。

**📊 数据集**

未使用具体实验数据集，主要引用已有系统实例（Bitcoin、Ethereum、Spanner、HotStuff 等）作为案例讨论。

**📈 对比分析**

论文为理论综述，未进行实验对比；通过案例分析指出属性融合可实现高吞吐、低延迟、可扩展性，同时指出加密开销与协同成本仍需进一步评估。

**⚠️ 局限性**

主要局限在缺乏统一的属性组合框架、性能与安全权衡的系统化分析不足、隐私与问责的冲突难以完全解决，以及后量子与人因挑战未得到完整解决。

---

## 308. LATMiX: Learnable Affine Transformations for Microscaling Quantization of LLMs

**arXiv ID:** 2602.17681 | [PDF](https://arxiv.org/pdf/2602.17681v1)

**作者:** Ofir Gordon `[一作]` (Arm Inc), Hai Victor Habi `[通讯]` (Arm Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大语言模型进行后训练量化时，利用可学习的可逆仿射变换来减少激活离群值，从而提升 MX 微尺度量化的准确率。

**💡 创新点**

创新点在于：①突破传统仅限旋转或哈达玛变换的局限，提出全可逆仿射变换（LU/QR 参数化）可被直接折叠进权重；②采用基于蒸馏的损失和体积保持正则化，使得变换在训练中既保持可逆性又能有效降低量化误差；③理论分析揭示 MX 量化误差受块结构和特征分布共同影响，指导变换设计。

**🔧 技术方法**

主要技术包括：后训练量化（PTQ）、MX 微尺度量化、可逆仿射变换（LU/QR 形式）、蒸馏损失、体积保持正则化、权重折叠技术。

**📊 数据集**

使用 WikiText2 的少量校准样本来学习变换；在 7 个零样本推理基准（ARC、HellaSwag、WinoGrande、PIQA、BoolQ、OpenBookQA 等）和 WikiText2 语言建模数据集上进行评估。

**📈 对比分析**

与 RTN、GPTQ、QuaRot、SpinQuant、FlatQuant、MR-GPTQ 等基线比较，实验显示 LATMiX 在 MX 4‑bit 量化下平均准确率恢复率提升约 4%，在所有任务上均优于现有方法。

**⚠️ 局限性**

局限性在于：需要额外的训练/优化步骤；目前仅在 MX 量化格式下验证，未评估在非 MX 量化场景的表现。

---

## 309. HiAER-Spike Software-Hardware Reconfigurable Platform for Event-Driven Neuromorphic Computing at Scale

**arXiv ID:** 2602.18072 | [PDF](https://arxiv.org/pdf/2602.18072v1)

**作者:** Gwenevere Frank `[一作]` (University of California San Diego), Gert Cauwenberghs `[通讯]` (University of California San Diego)

**通讯引用:** 17049 | [OpenAlex ID](https://openalex.org/A5059013717)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文构建了一个可扩展、可重配置的 FPGA neuromorphic 平台 HiAER‑Spike，能够支持多达 160 M 个神经元、40 B synapses 的大规模事件驱动神经网络，并提供了易用的 Python API 与 Web 门户，方便研究者远程配置和运行 SNN。

**💡 创新点**

创新点包括：① 使用层次化地址事件路由（HiAER）实现跨核心/FPGA/服务器的高效 spike 传递；② 采用混合 HBM/URAM/BRAM 记忆层级，结合稀疏邻接表存储提升网络容量与能效；③ 将硬件与软件深度协同，提供无硬件细节的高级 API 与自动化的网络划分与资源调度；④ 通过 Web 门户共享资源，降低进入门槛。

**🔧 技术方法**

技术实现主要依赖：Xilinx XCVU37P FPGA 与 8 GB HBM、Python/C++ hs_api 接口、PyTorch/SpikingJelly 转换、LIF 与 ANN 神经元模型、量化与稀疏化技术、HiAER 多级路由、SDC 服务器集群（8 × FPGA、Arista 交换机、FireFly）以及高性能 PCIe 与 Ethernet 通信。

**📊 数据集**

实验使用了四个公开数据集：MNIST（二值化）、DVS Gesture、CIFAR‑10（二值化）以及基于 DVS 的 Atari Pong；这些数据集用于评估分类精度、能耗与延迟，并与其它 neuromorphic 平台进行对比。

**📈 对比分析**

评估方法为将同一模型在软件模拟与 HiAER‑Spike 硬件上分别跑推理，并记录准确率、HBM 访问能耗（μJ）与推理延迟（μs）。与 Loihi、SpiNNaker、TrueNorth 等平台对比，HiAER‑Spike 在 138 k 神经元 MNIST 模型中实现 96.59 % 的准确率，能耗仅 1.1 μJ，延迟 4.2 μs；在 1,115 k 神经元 DVS Gesture 模型中能耗 79.8 μJ，延迟 184.9 μs，虽然准确率略低于竞争平台，但能耗与延迟显著更优。

**⚠️ 局限性**

局限性包括：目前仅在单核心/单 FPGA 上验证，缺乏大规模多核心/多FPGA 的性能与能耗数据；模型精度仍落后于竞争平台，主要由于较少的学习规则与神经元模型；系统对用户硬件依赖较高，尚未完全公开；未来需要进一步提升网络规模、支持更丰富的神经元模型与自适应学习、以及更高效的跨节点通信。

---

## 310. UAOR: Uncertainty-aware Observation Reinjection for Vision-Language-Action Models

**arXiv ID:** 2602.18020 | [PDF](https://arxiv.org/pdf/2602.18020v1)

**作者:** Jiabing Yang `[一作]` (University of Chinese Academy of Sciences), Liang Wang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 43385 | [OpenAlex ID](https://openalex.org/A5115602506)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在Vision‑Language‑Action模型中引入了一种训练无关、可插拔的观察信息重注入模块（UaOR），通过检测推理时的动作不确定性（Action Entropy）来决定是否将视觉/本体观测特征重新注入到下一层的 FFN 中，以提升动作生成的自信度和准确性。

**💡 创新点**

创新点在于：1) 将 FFN 重新表述为键值记忆机制，利用其内部注意力动态提取观察信息；2) 设计基于动作熵的层级不确定性度量；3) 在不需要额外数据、模块或再训练的前提下，通过不确定性驱动的重注入实现性能提升。

**🔧 技术方法**

核心技术包括：1) FFN 的键值重构与注意力检索；2) 动作熵计算与阈值触发策略；3) α 融合机制；4) 无监督训练，直接在推理阶段应用。

**📊 数据集**

在三大仿真基准（LIBERO、SIMPLER、CALVIN）以及 Franka 机器人真实实验上验证；使用了多种 VLA 模型（OpenVLA‑OFT、π_0、CogACT、LLaVA‑VLA 等）和对应的数据集。

**📈 对比分析**

与原始 VLA 模型以及多种强化观察/提示方案进行对比。实验显示：在 LIBERO 上提升至 98% 成功率，π_0 提升 1.5 分；在 SIMPLER 上 CogACT 提升 2.6 分；在 CALVIN 上 LLaVA‑VLA 成功率提升 1.1‑2.6 分，平均完成长度提高 0.12。推理开销仅降低 4.8% 速度，延迟增加 5%。

**⚠️ 局限性**

局限性：1) 需要手动调节阈值 γ 与混合系数 α；2) 只针对观察信息有效，指令信息重注入未见提升；3) 在某些任务中若不确定性估计失效或阈值设置不当，可能导致性能下降。

---

## 311. Art Notions in the Age of (Mis)anthropic AI

**arXiv ID:** 2602.18202 | [PDF](https://arxiv.org/pdf/2602.18202v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 312. Cut Less, Fold More: Model Compression through the Lens of Projection Geometry

**arXiv ID:** 2602.18116 | [PDF](https://arxiv.org/pdf/2602.18116v1)

**作者:** Olga Saukh `[一作]` (Graz University of Technology), Lothar Thiele `[通讯]` (ETH Zurich)

**通讯引用:** 54985 | [OpenAlex ID](https://openalex.org/A5060999697)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究在不需要重新训练的情况下，通过对结构化剪枝和模型折叠的投影几何视角进行对比，提出并验证模型折叠优于传统幅值剪枝。

**💡 创新点**

创新点在于把剪枝和折叠统一视作参数空间的正交投影，证明在仅多一个秩的前提下，折叠能获得更小的参数重构误差和更紧的功能扰动上界；同时在上千个训练检查点上系统评估其性能。

**🔧 技术方法**

采用正交投影理论、k‑means 聚类、Lipschitz 连续性分析，并结合轻量化 LayerNorm 复位、短期全微调等校准技术；实验中使用 Adam/SGD、SAM、不同学习率、数据增强等训练变异。

**📊 数据集**

使用 CIFAR‑10、ImageNet‑1K 上的 ResNet18、PreActResNet18、ViT‑B/32、CLIP‑ViT‑B/32；以及在 C4 语料上训练的 LLaMA‑60M、LLaMA‑130M。

**📈 对比分析**

通过与幅值基结构化剪枝（L1/L2）在相同稀疏率下对比，发现模型折叠在中高压缩率下的后压缩准确率普遍更高，且即使在轻量化 LayerNorm 或 1–5 轮完整微调后仍保持优势；在极低压缩或特定训练设置下差距缩小甚至逆转。

**⚠️ 局限性**

局限性包括：理论证明允许秩增一而非完全相同的压缩尺寸；实验仅对卷积/视觉 Transformer 的 FFN 块做折叠，未扩展到注意力块；未考虑量化、蒸馏或无结构稀疏；只在中小规模 LLM 上验证，无法直接推广至更大模型；以及主要评估在无校准数据的情形。

---

## 313. Domain-Decomposed Lagrangian Data Assimilation for Drifting Sea-Ice Floe Dynamics

**arXiv ID:** 2602.17971 | [PDF](https://arxiv.org/pdf/2602.17971v1)

**作者:** Danyang Li `[一作]` (Australian National University), Quanling Deng `[通讯]` (Tsinghua University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

做了一个基于域分解的海冰漂浮块离散元模型的观测数据同化框架，结合局部ETKF和高斯加权融合来恢复海流场。

**💡 创新点**

创新点在于将域分解与局部ETKF相结合，既实现了并行计算，又通过高斯权重平滑子域边界，显著提高了在稀疏观测下的恢复精度。

**🔧 技术方法**

使用的技术包括Lagrangian离散元方法、集合变换卡尔曼滤波器（ETKF）、傅里叶空间海流线性随机模型和高斯加权融合。

**📊 数据集**

数据集为40,000个自由漂浮海冰块的模拟轨迹，海流场采用截断的GB模态并加入随机噪声，观测为噪声化的浮子位置。

**📈 对比分析**

与完整域同化基线比较，域分解方法在NRMSE下降约20-30%、PCC提升约50%，且计算时间降低约30-40%。

**⚠️ 局限性**

局限性包括在子域划分过细时缺失跨域相关信息、对旋转和碰撞机制的缺乏，以及对观测密度和权重参数的敏感性。

---

## 314. A Single Image and Multimodality Is All You Need for Novel View Synthesis

**arXiv ID:** 2602.17909 | [PDF](https://arxiv.org/pdf/2602.17909v1)

**作者:** Amirhosein Javadi `[一作]` (University of California San Diego), Tara Javidi `[通讯]` (University of California San Diego)

**通讯引用:** 8826 | [OpenAlex ID](https://openalex.org/A5059310658)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了基于极度稀疏雷达/激光雷达测距的局部高斯过程深度重建框架，并将重建得到的稠密深度与不确定性直接替换单目深度估计，用作扩散模型的几何条件，从而实现单图像新视角视频合成。

**💡 创新点**

创新点在于：①将角度域投影与局部高斯过程相结合，既能在稀疏测量下恢复稠密深度，又能量化不确定性；②该深度模块可以无缝插拔到现有扩散渲染流水线，保持模型不变；③通过不确定性掩模有效抑制误差传播，提升几何一致性与视觉质量。

**🔧 技术方法**

技术要点包括：局部高斯过程（RBF核）深度回归、角度域映射、彩色点云生成与渲染、几何条件扩散（GEN3C）以及基于不确定性的深度掩模。

**📊 数据集**

使用了 View‑of‑Delft (VoD) 多模态自动驾驶数据集，该数据集提供同步雷达、相机和激光雷达信息。

**📈 对比分析**

与基线 Vision‑Only（MoGe深度+GEN3C）对比，使用 PSNR、SSIM、LPIPS、FID、t‑LPIPS 进行评估。稀疏雷达+GP 提升 PSNR 15.4%、SSIM 6.6%，LPIPS 降低 23.5%、FID 降低 46.0%；稀疏 LiDAR 效果更佳（PSNR 14.69、SSIM 0.4971、LPIPS 0.4230、FID 71.91、t‑LPIPS 0.0563）。在深度估计上，MAE 13.61、RMSE_log 0.92，均优于 MoGe 与 Depth Anything V2。

**⚠️ 局限性**

局限性包括：①局部 GP 的窗口大小与核参数对结果敏感；②稀疏测量覆盖率仍极低（0.02%–0.52%），对复杂遮挡或动态物体的处理仍有限；③需要同步雷达/激光雷达与相机数据，部署成本较高；④在多帧、多视角或长时间序列中的稳定性与实时性尚未充分验证。

---

## 315. ZACH-ViT: Regime-Dependent Inductive Bias in Compact Vision Transformers for Medical Imaging

**arXiv ID:** 2602.17929 | [PDF](https://arxiv.org/pdf/2602.17929v1)

**作者:** Athanasios Angelakis `[一作]` `[通讯]` (University of Bundeswehr Munich), Athanasios Angelakis (University of Bundeswehr Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种名为ZACH-ViT的极简可交换Vision Transformer，通过去除位置编码和[CLS] token，仅使用全局平均池化实现端到端的图像分类，并在少样本医学图像任务上进行评估。

**💡 创新点**

核心创新在于将空间先验完全剔除，构建可交换的Transformer骨干；并引入自适应残差投影以维持少参数配置下的训练稳定性，证明在弱空间结构任务中可与大模型竞争。

**🔧 技术方法**

技术手段包括无位置编码的ViT架构、全局平均池化代替聚合token、自适应残差投影、0.25M参数的压缩模型，全部使用TensorFlow实现。

**📊 数据集**

实验使用MedMNIST v2的七个数据集：BloodMNIST、PathMNIST、BreastMNIST、PneumoniaMNIST、DermaMNIST、OCTMNIST和OrganAMNIST，采用50-shot、5个随机种子的统一训练协议。

**📈 对比分析**

通过与十五个基线模型（CNN、预训练ViT、TransMIL等）在相同数据分割、训练设置下的平均表现进行对比，ZACH-ViT在弱空间结构数据上优于TransMIL，在整体排名与大规模预训练模型相当，同时参数量更小、推理速度更快，适合边缘设备部署。

**⚠️ 局限性**

局限性包括：在强空间结构任务（OCTMNIST、OrganAMNIST）去除位置编码导致性能略逊；对不同分辨率或模态的泛化能力未全面评估；缺少对各类patch尺寸、层数等超参数在不同空间结构下的系统性剖析。

---

## 316. Convergent Gate Elimination and Constructive Circuit Lower Bounds

**arXiv ID:** 2602.17942 | [PDF](https://arxiv.org/pdf/2602.17942v1)

**作者:** Marco Carmosino `[一作]` (Massachusetts Institute of Technology IBM Watson Artificial Intelligence Lab), Tim Jackman `[通讯]` (Boston University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文将门消除方法形式化为可收敛的词图重写系统，并利用该系统给出 XOR 函数在 DeMorgan 基础上的构造性线性下界证明。

**💡 创新点**

创新点在于：①首次把门消除过程转化为可收敛的重写系统；②证明 DeMorgan（和 {∧,⊕,∨}）基的简化系统是收敛的；③利用该收敛系统实现了一个多项式时间的反驳器，给出了首个通过门消除得到的构造性下界；④展示了 U_2 与 U_2 基础无法得到收敛系统。

**🔧 技术方法**

主要技术包括：术语重写与 Knuth‑Bendix 完成算法；Plump 的词图重写框架；构造具体的简化规则集（规范化、修正、传递、冗余消除）；以及对简化过程的构造性分析。

**📊 数据集**

无数据集，研究完全基于理论与形式化证明。

**📈 对比分析**

方法对简化过程不依赖于具体顺序，因系统收敛可保证唯一化简结果；构造的反驳器在多项式时间内输出错误输入，表明方法可实现高效的错误检测与下界证明。

**⚠️ 局限性**

局限性包括：①仅在 DeMorgan（及 {∧,⊕,∨}）基础上可构造收敛系统；②在 U_2 与 U_2 基础上不收敛，无法直接使用；③目前仅得到线性下界，无法突破至超线性；④缺乏对更一般函数或更复杂基底的构造性下界研究。

---

## 317. Toward Automated Virtual Electronic Control Unit (ECU) Twins for Shift-Left Automotive Software Testing

**arXiv ID:** 2602.18142 | [PDF](https://arxiv.org/pdf/2602.18142v1)

**作者:** Sebastian Dingler `[一作]` (University of Applied Sciences Esslingen), Frederik Boenke `[通讯]` (NUVUS GmbH)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

通过生成‑评估‑修正循环，自动构建与实际二进制兼容的ARMv8 CPU模型，实现虚拟 ECU 的早期软件集成与调试。

**💡 创新点**

提出基于 LLM 的生成模型和差分测试闭环的代理式方法，能够在缺乏完整硬件信息的情况下逐步校准指令级行为，显著降低 CPU 行为建模的技术风险。

**🔧 技术方法**

使用 SystemC/TLM‑2.0、QEMU 进行指令执行、GDB 接口进行状态读取、LLM（如 ChatGPT）进行代码生成与修正，并结合差分测试框架实现闭环校准。

**📊 数据集**

使用 ARMv8 参考模拟器生成的指令执行序列与寄存器/标志状态快照作为参考数据；未使用公开数据集，而是基于模拟器产生的自生成数据。

**📈 对比分析**

通过比较指令前后寄存器/标志状态差异进行差分评估；实验表明多轮迭代后可实现指令级准确性，性能方面未给出具体速度数值，但可在 host‑based 环境中复现硬件行为，显著降低 HiL 资源依赖。

**⚠️ 局限性**

主要限制包括：外设建模精度不足、系统级时序一致性未完全验证、单线程 SystemC 内核导致性能瓶颈、缺乏云端多租户与 CI/CD 集成，以及未覆盖完整 ECU 级安全性证据生成。

---

## 318. Ori-Sense: origami capacitive sensing for soft robotic applications

**arXiv ID:** 2602.18379 | [PDF](https://arxiv.org/pdf/2602.18379v1)

**作者:** Hugo de Souza Oliveira `[一作]` (University of Freiburg), Edoardo Milana `[通讯]` (University of Freiburg)

**通讯引用:** 610 | [OpenAlex ID](https://openalex.org/A5013363595)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

设计并制造了一种基于倒置Kresling折纸的柔性电容传感器 Ori-Sense，用于软体机器人自我感知。

**💡 创新点**

创新点在于利用倒置Kresling几何实现扭转变形与电容变化的高度耦合，同时保持极低的机械阻抗和局部化应变。

**🔧 技术方法**

采用可溶性核心成型、3D打印 BVOH 核心与 TPU 电极、硅胶浇铸、FDC2214 电容计数器以及 Abaqus 有限元仿真。

**📊 数据集**

使用自制实验数据集，包含轴向和扭转加载下的力学曲线及电容信号。

**📈 对比分析**

通过实验与 FEM 对比验证扭转-力矩曲线，电容变化可达30%，角度灵敏度约0.0067，力矩低于0.01，显示出良好的低阻抗与高灵敏度。

**⚠️ 局限性**

局限性包括对极端压缩时动态范围下降、仅实现单模扭转感知、环境噪声对信号的影响，以及尚未在实际软体执行器中验证长期稳定性。

---

## 319. DohaScript: A Large-Scale Multi-Writer Dataset for Continuous Handwritten Hindi Text

**arXiv ID:** 2602.18089 | [PDF](https://arxiv.org/pdf/2602.18089v1)

**作者:** Kunwar Arpit Singh `[一作]` (Indian Institute of Science Education and Research Bhopal), Haroon R Lone `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了大规模多作者的 Devanagari 手写 Hindi 朗诵集 DohaScript，包含 531 名作者在 A4 页面上写下相同 6 句 dohas 的连续文本，并配备了自动化质量评估与行分割难度注释。

**💡 创新点**

首次构建并公开并行风格语料库，将词汇一致与作者多样结合，配合 CNN 质量判别和页面布局难度标注，为连续手写文本识别、作家识别和生成模型提供统一基准。

**🔧 技术方法**

使用卷积神经网络进行图像质量二分类与四分类，结合 Laplacian 变差的自动评估管线；采用启发式线分割算法为每页打分；所有处理通过 PyTorch 与标准图像预处理完成。

**📊 数据集**

主要使用自建的 DohaScript 数据集；在对比实验中参照 IIIT‑HW‑Words、IIIT‑HW‑Dev、CHIPS、Parimal Hindi、AnciDev 等现有 Indic 手写数据集。

**📈 对比分析**

通过统计 531 页的质量分级（优质 288 页）与行分割准确率（仅 29.6% 完全分割），并在 HTR 与作家识别基线实验中展示数据集在未知作者上的泛化性能优于现有数据集，质量筛选后模型准确率显著提升。

**⚠️ 局限性**

数据以固定诗句为主，缺乏词汇多样性；行分割仍面临高复杂度场景；低分辨率或噪声严重页面模型鲁棒性有限；仅覆盖 Devanagari，未扩展至其他 Indic 脚本。

---

## 320. Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly

**arXiv ID:** 2602.17997 | [PDF](https://arxiv.org/pdf/2602.17997v1)

**作者:** Zehao Jin `[一作]` (Tsinghua University), Yanan Sui `[通讯]` (Tsinghua University)

**通讯引用:** 1145 | [OpenAlex ID](https://openalex.org/A5069290448)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于果蝇全脑连接组的图神经网络 FlyGM，用于控制虚拟果蝇的全身运动。

**💡 创新点**

创新点在于直接将完整的生物连接图作为动态控制器的拓扑结构，无需人工设计网络，展示连接组本身提供的结构诱导优势。

**🔧 技术方法**

技术包括图神经网络消息传递、基于兴奋/抑制突触计数的方向性权重、afferent‑intrinsic‑efferent 节点分层，以及结合强化学习（PPO）的端到端训练。

**📊 数据集**

数据集使用 FlyWire 连接组（成年果蝇全脑 synaptic 解析）和 MuJoCo 的 flybody 物理模拟环境中生成的专家轨迹（多任务运动数据）。

**📈 对比分析**

作者与同参数的度数保持重连图、Erdős–Rényi 随机图以及标准 MLP 进行对比，FlyGM 在四种运动任务（步态启动、直线行走、转弯、飞行）上表现出更高的样本效率，位置误差与角度误差显著下降。

**⚠️ 局限性**

限制包括计算成本高、内存占用大、仅评估运动任务，未涵盖更复杂行为，也未利用更完整的连接组信息。

---

## 321. Towards LLM-centric Affective Visual Customization via Efficient and Precise Emotion Manipulating

**arXiv ID:** 2602.18016 | [PDF](https://arxiv.org/pdf/2602.18016v1)

**作者:** Jiamin Luo `[一作]` (Soochow University), Jiahong Lu `[通讯]` (Soochow University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于大型语言模型的情感视觉定制任务（L-AVC），并设计了高效精准情感操控（EPEM）框架，用以在图像中按指令改变主观情绪，同时保留情绪无关的内容。

**💡 创新点**

创新点主要包括：①使用模型编辑（hypernetwork）在LLM的MLP层上实现情绪语义转换（EIC模块），实现低资源下的高效情绪对齐；②设计情感注意力交互（EAI）块和adapter（PER模块），通过跨模态注意力精准保留情绪无关视觉特征；③构建专门的L-AVC数据集并提出情绪专用评估指标，填补了现有视觉定制中情绪编辑的空白。

**🔧 技术方法**

核心技术包括：多模态大型语言模型（BLIP2-OPT 2.7B）进行情绪理解；Stable Diffusion-v1.5作为图像生成后端；hypernetwork对LLM MLP权重进行编辑；情感注意力交互块（EAI）与跨模态adapter实现LLM与扩散模型的情感对齐；AdamW优化、超参数调优等。

**📊 数据集**

使用从EmoSet抽取的2000张图片扩展为10k图像的L-AVC数据集，涵盖面部、动作、物体、场景、颜色与亮度等五类视觉元素，并为每张图像生成原始与编辑后的标题、情绪标签及编辑指令。

**📈 对比分析**

与 ControlNet、Prompt‑to‑Prompt、InstructPix2Pix、SDEdit、DiffEdit、MGIE、SmartEdit 等多种先进视觉定制基线进行对比。实验结果表明，EPEM 在 FID、LPIPS、SSIM、CLIP‑I 等内容一致性指标、CLIP‑T 语义贴合度、M‑Eval / G‑Eval / H‑Eval 情绪准确率以及编辑时长上均明显优于所有基线，提升幅度在 5%–15% 之间，且多项提升具有统计显著性。

**⚠️ 局限性**

局限性包括：仅针对单帧图像，未扩展至视频情绪编辑；情绪标签的主观性导致数据集标注难度较高；模型编辑和注意力交互模块对硬件资源仍有一定需求；在极端或多情绪交叠场景下的泛化能力仍待进一步验证。

---

## 322. Condition-Gated Reasoning for Context-Dependent Biomedical Question Answering

**arXiv ID:** 2602.17911 | [PDF](https://arxiv.org/pdf/2602.17911v1)

**作者:** Jash Rajesh Parekh `[一作]` (University of Illinois Urbana-Champaign), Jiawei Han `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 122614 | [OpenAlex ID](https://openalex.org/A5019539533)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了CondMedQA基准（100道需要考虑患者特定条件的多跳医学问答）以及Condition‑Gated Reasoning（CGR）框架，旨在显式建模并利用条件信息进行知识图推理，从而提升医学问答的准确性与可解释性。

**💡 创新点**

创新点包括：
1) 首次构建专门评估条件推理能力的医学QA基准；
2) 在知识图构造时扩展为4‑tuple（<实体，关系，实体，条件>），并通过LLM进行条件抽取；
3) 在图遍历过程中引入条件门控机制，仅允许与查询条件一致的边被激活，消除禁忌路径；
4) 将路径与条件信息构成结构化证据，直接驱动LLM生成答案。

**🔧 技术方法**

技术手段：
- 大语言模型（Qwen2.5‑14B‑Instruct‑GPTQ‑Int4）用于抽取n‑tuple与评估条件；
- MedEmbed 进行实体/路径语义匹配；
- 依据查询条件构建条件检索表ℒ，实现O(1)条件检查；
- 结合传统RAG、GraphRAG等基线做对比；
- LLM（GPT‑5.2、Qwen‑等）用于答案生成。

**📊 数据集**

使用的数据集：
- CondMedQA（100道条件问答）；
- MedHopQA（400道多跳问答）；
- MedHopQA(Cond)（35道条件子集）；
- BioASQ‑Task B（事实问答）。

**📈 对比分析**

比较方法与性能：
- 在CondMedQA上，CGR使用GPT‑5.2的EM为82.0%，超过最强基线（RAG 44.0%）约20个百分点；
- 在MedHopQA上，CGR EM 86.75%（对比MedRAG 75.75%），提升约11个百分点；
- 在BioASQ‑B上，CGR EM 40.60%（对比RAG 31.80%）；
- 在所有数据集上，CGR在不同模型（GPT‑5.2、Qwen‑、LLaMA‑）均保持领先，尤其在条件问答场景表现突出。

**⚠️ 局限性**

Limitations：
- CondMedQA样本规模有限（仅100题），缺乏更广泛的临床场景覆盖；
- 知识图依赖检索文本，容易继承来源偏差、错误或遗漏；
- 实际临床应用需人工审核与持续更新，模型输出不宜直接用于决策；
- 目前未系统评估多样化人群与伦理偏见，未来需进一步审计。

---

## 323. Games That Teach, Chats That Convince: Comparing Interactive and Static Formats for Persuasive Learning

**arXiv ID:** 2602.17905 | [PDF](https://arxiv.org/pdf/2602.17905v1)

**作者:** Seyed Hossein Alavi `[一作]` (University of British Columbia), Vered Shwartz `[通讯]` (Vector Institute for AI)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在相同内容的前提下，对比静态论文、聊天机器人和文本叙事游戏三种信息传递模式，评估它们对主观体验、说服感知和知识保留的影响。

**💡 创新点**

首次在内容保持一致的实验框架中系统比较交互性与学习/说服的关系，揭示主观学习与客观记忆的解耦，以及交互日志与学习的弱相关性。

**🔧 技术方法**

使用大语言模型（GPT‑4.1）生成文章、对话与游戏脚本；通过问卷、延迟（24h）知识测验和交互日志记录收集数据；采用非参数检验、序数逻辑回归和Spearman相关分析进行评估。

**📊 数据集**

实验数据来自45名受试者（43名完成）在两主题（回收与公共交通）下的问卷、交互日志与知识测验结果，形成了实验组与控制组的比较数据集。

**📈 对比分析**

采用 Kruskal‑Wallis + Mann‑Whitney、序数回归等方法比较三种模式，结果显示聊天机器人在主观体验、说服感知和重要性提升上最高；文本游戏在客观记忆上略优于文章，但自评学习最低；差异虽未全部显著，但揭示显著趋势。

**⚠️ 局限性**

局限性包括样本量小、仅覆盖两主题、受试者年轻且已有环保意识、只测24小时短期记忆、未评估长期行为改变、交互日志未与学习显著相关、未考察不同人群对说服效果的差异。

---

## 324. Learning Long-Range Dependencies with Temporal Predictive Coding

**arXiv ID:** 2602.18131 | [PDF](https://arxiv.org/pdf/2602.18131v1)

**作者:** Tom Potter `[一作]` (University of Manchester), Oliver Rhodes `[通讯]` (University of Manchester)

**通讯引用:** 401 | [OpenAlex ID](https://openalex.org/A5081722405)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究并提出结合Temporal Predictive Coding与Real‑Time Recurrent Learning的tPC RTRL算法，用于RNN的时空信用分配，并在长序列任务上验证其可与BPTT竞争。

**💡 创新点**

创新点在于首次将RTRL与tPC结合，既保留PC的局部并行特性，又显著提升对长程依赖的学习能力，并在大规模任务中实现了这一方法。

**🔧 技术方法**

采用的技术包括Temporal Predictive Coding、Real‑Time Recurrent Learning、线性递归单元（LRU）、Inference Learning、梯度下降等。

**📊 数据集**

实验使用的公开数据集包括Synthetic复制任务、WikiText‑2语言模型数据集以及CCMatrix英文‑法语机器翻译子集。

**📈 对比分析**

在所有任务中，tPC RTRL的性能与BPTT基本一致：复制任务准确率100%，WikiText‑2困惑度约99.2（BPTT 98.6），机器翻译测试困惑度7.62（BPTT 7.49）BLEU 20.71（BPTT 21.11）。

**⚠️ 局限性**

局限性包括RTRL的计算与内存开销需通过LRU等特殊单元缓解；多层网络的扩展与超参数调优经验不足；在实际硬件上的能效验证尚未完成。

---

## 325. On the Adversarial Robustness of Discrete Image Tokenizers

**arXiv ID:** 2602.18252 | [PDF](https://arxiv.org/pdf/2602.18252v1)

**作者:** Rishika Bhagwatkar `[一作]` (Mila - Quebec AI Institute), Francesco Croce `[通讯]` (ELLIS Institute Finland - Aalto University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文首次系统评估离散图像分词器在对抗攻击下的鲁棒性，并提出了一种无监督的嵌入空间攻击方法，随后利用该攻击进行无监督对抗微调以提升分词器鲁棒性。

**💡 创新点**

创新点在于：①提出任务无关、计算高效的无监督对抗攻击；②通过仅微调分词器编码器实现鲁棒性提升；③展示该鲁棒分词器可无缝迁移至多种下游任务（分类、检索、VQA、Captioning），显著提升整体系统安全。

**🔧 技术方法**

主要技术包括：离散图像分词器（如 TiTok、UniTok）与向量量化；无监督对抗攻击（最大化预量化嵌入差异）；无监督对抗微调（在原始分词器上加入鲁棒性约束）；Straight-Through 估计、APGD、AutoAttack 等对抗生成与评估工具。

**📊 数据集**

使用的数据集包括 ImageNet‑1k、Imagenette、Caltech101、OI‑Crop、OI‑Pos、VQAv2、OK‑VQA、GQA，以及大规模训练集如 DataComp，用于攻击与微调。

**📈 对比分析**

对比实验显示：无监督攻击在鲁棒性上与监督攻击相当；鲁棒分词器在多任务下的鲁棒准确率提升了约10–20%，且在未见过的数据集上保持优良性能；相较于全模型监督对抗训练，微调分词器的计算成本降低约2.2倍，且不导致下游任务性能下降。

**⚠️ 局限性**

局限性包括：①鲁棒性提升主要针对已研究的分词器结构，其他设计（如多码本、FSQ）尚未充分验证；②无监督攻击不保证对所有攻击向量的全覆盖；③对极高对抗强度（ε>16/255）时鲁棒性提升有限；④可能对极端数据分布产生过拟合风险。

---

## 326. Image Quality Assessment: Exploring Quality Awareness via Memory-driven Distortion Patterns Matching

**arXiv ID:** 2602.18000 | [PDF](https://arxiv.org/pdf/2602.18000v1)

**作者:** Xuting Lan `[一作]`, Weijia Jia `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种记忆驱动的质量感知框架MQAF，利用可训练的内存银行存储典型畸变模式，在有参照图像时结合参考信息与内存匹配，在无参照时仅靠内存实现图像质量评估。

**💡 创新点**

创新点在于将人类视觉记忆机制引入IQA，通过显式内存单元实现畸变模式的学习与检索，动态权衡参考匹配与内存匹配，显著降低对高质量参照图像的依赖。

**🔧 技术方法**

采用深度特征提取（ResNet50/VGG16）、余弦与范数相似度、内存匹配卷积、注意力加权网络、去相关正则化等技术。

**📊 数据集**

使用公开数据集 LIVE、CSIQ、TID2013、KADID‑10K、PIPAL 等。

**📈 对比分析**

与多种 FR‑IQA（如 PSNR、SSIM、DISTS、LPIPS、JND‑SalCAR 等）及 NR‑IQA（如 WaDIQaM‑NR、DBCNN、MetaIQA 等）进行对比，MQAF 在参考和无参照模式下均取得或超过现有最佳分数，尤其在跨数据库和单失真类型测试中表现突出。

**⚠️ 局限性**

局限性包括对内存规模与超参数敏感，且在极端低质量或极少失真样本下可能仍受限；目前未对实时或边缘设备部署进行深入评估。

---

## 327. Memory-Based Advantage Shaping for LLM-Guided Reinforcement Learning

**arXiv ID:** 2602.17931 | [PDF](https://arxiv.org/pdf/2602.17931v1)

**作者:** Narjes Nourzad `[一作]` (University of Southern California), Carlee Joe-Wong `[通讯]` (Carnegie Mellon University)

**通讯引用:** 17262 | [OpenAlex ID](https://openalex.org/A5003037377)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文提出通过构建包含LLM先验与智能体经验的记忆图，并在PPO中使用该图产生的效用信号对优势进行柔性塑形，以提升稀疏奖励环境下的样本效率。

**💡 创新点**

创新点在于将LLM生成的子目标与实际轨迹整合到可扩展的记忆图中，并通过效用衍生的优势塑形实现对RL更新的引导，既降低对LLM的实时依赖，又保持PPO的收敛性。

**🔧 技术方法**

主要技术包括PPO、优势塑形、记忆图构造、轨迹相似度与子目标一致性评估、有限的离线/在线LLM调用以及基于子目标的奖励近似。

**📊 数据集**

论文在Gymnasium的FrozenLake-8x8和MiniGrid的Doorkey两个基准环境上进行实验。

**📈 对比分析**

与PPO、HRL、LLM4Teach等基线对比，方法在FrozenLake实现更快收敛，在Doorkey保持显著样本效率与最终性能，且在未见种子测试中与LLM4Teach的平均回报及成功率无显著差异。

**⚠️ 局限性**

限制在于仍需偶尔的LLM查询，记忆图的维护与剪枝可能复杂；离线先验的质量和子目标表示的鲁棒性可能限制方法在更大规模或更复杂任务中的适用性。

---

## 328. Mind the Style: Impact of Communication Style on Human-Chatbot Interaction

**arXiv ID:** 2602.17850 | [PDF](https://arxiv.org/pdf/2602.17850v1)

**作者:** Erik Derner `[一作]` (Czech Technical University in Prague), Nuria Oliver `[通讯]` (ELLIS Alicante)

**通讯引用:** 14931 | [OpenAlex ID](https://openalex.org/A5013727792)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过一项双组间实验研究，探讨了对话式聊天机器人在固定沟通风格（友好/直接）下对二维地图导航任务的用户任务成功率、主观满意度及语言适配性的影响。

**💡 创新点**

创新点在于：①将聊天机器人的沟通风格固定化，仅改变语调；②在目标导向、结构化任务中系统隔离语调效应；③发现友好语调对女性用户的任务成功率和满意度有显著正向影响；④展示了在短期、任务驱动交互中语言适配主要为特征级别而非整体风格趋同。

**🔧 技术方法**

使用技术包括：基于GPT‑4的聊天机器人实现、系统提示调节语调、LIWC词典与句子风格嵌入（Wegmann 等）进行表面与深层语言特征分析、统计建模（线性回归、逻辑回归、方差分析）以及交互式 Web 应用（Streamlit）。

**📊 数据集**

数据集由 470 名美国 Prolific 受试者组成，分为控制（158人）、友好式聊天（155人）和直接式聊天（157人）三组；实验数据包含任务完成标识、对话文本、问卷评分及使用频率信息。

**📈 对比分析**

通过对比友好与直接两种风格以及无聊天机器人控制组，发现友好风格显著提升女性受试者的任务成功率（相对提升约 3 倍）和整体满意度；在控制组中不存在性别差异；但无聊天机器人条件在任务效率上优于任何聊天机器人条件，表明聊天机器人并不总能提升目标导向任务的性能。

**⚠️ 局限性**

限制包括：①性别变量仅为二元（男/女），未覆盖非二元身份；②实验为短时、人工情境，缺乏真实导航场景；③样本主要为英语美国人，文化普适性未知；④未深入探讨长期使用对适配与满意度的影响。

---

## 329. Mind the Boundary: Stabilizing Gemini Enterprise A2A via a Cloud Run Hub Across Projects and Accounts

**arXiv ID:** 2602.17675 | [PDF](https://arxiv.org/pdf/2602.17675v1)

**作者:** Takao Morita `[一作]` `[通讯]` (Independent Researcher), Takao Morita (Independent Researcher)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了基于 Cloud Run 的 Gemini Enterprise A2A Hub，用来在同项目、跨项目和跨账户之间路由并协调多种后端代理和工具。

**💡 创新点**

提出了 UI 兼容性为核心的设计：在 JSON‑RPC 通道只返回文本，结构化结果拆分到独立的 REST API；同时阐明了跨边界认证的细节与 RAG 证据访问的独立需求。

**🔧 技术方法**

技术栈包括 Gemini Enterprise UI、Agent‑to‑Agent (A2A) 协议、Cloud Run、Google Cloud IAM 与 OIDC ID Token、Discovery Engine / Vertex AI Search、GCS、Python（Starlette/Uvicorn）和异步网络调用。

**📊 数据集**

使用了四个手工设计的业务查询（费用报销截止、WBS 任务、富士山高度、P‑1 事件通知截止），构成一个小型多场景基准。

**📈 对比分析**

对比方法：在 JSON‑RPC 与 REST 两条路径分别验证路由正确性、UI 兼容性和证据提取；结果显示：所有四条查询都按预期路由；JSON‑RPC 输出始终为单文本，没有 UI 错误；RAG 路径在授予 GCS 读取权限后能准确返回“15 分钟内”截止时间。

**⚠️ 局限性**

局限性：基准样本有限；依赖特定 Cloud Run 与 IAM 配置，跨账户认证复杂；UI 兼容性设计需与 Gemini UI 更新保持同步；未评估大规模并发负载或长文本处理能力。

---

## 330. Gender and Digital Platform Work During Turbulent Times

**arXiv ID:** 2602.17721 | [PDF](https://arxiv.org/pdf/2602.17721v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 331. Towards a Higher-Order Bialgebraic Denotational Semantics

**arXiv ID:** 2602.18295 | [PDF](https://arxiv.org/pdf/2602.18295v1)

**作者:** Sergey Goncharov `[一作]` (University of Birmingham), Stefano Volpe `[通讯]` (University of Southern Denmark)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

提出了一种基于高阶抽象GSOS框架的巴尔阿尔几何（bialgebraic）归约语义（denotational semantics）理论；

**💡 创新点**

创新点在于引入“局部最终煤气层”(locally final coalgebra)概念来取代传统的最终煤气层，构造高阶语言的足够好的语义域，并证明其相容性与充分性；

**🔧 技术方法**

使用范畴理论、混合变差双函子、M-范畴（ultrametric enriched categories）以及Banach固定点定理等高级工具来构造局部最终煤气层，并推导出相应的语义模型；

**📊 数据集**

无具体数据集，论文仅提供理论证明与若干语言实例（如简单型SKI、概率型、非确定性并发、无类型λ计算）来展示方法；

**📈 对比分析**

对比方法：通过证明局部最终煤气层对应的语义模型在抽象GSOS框架下自动满足构造性与充分性，不需要语言特定的手工证明；性能方面在理论层面说明模型的构造是唯一（在同构意义下）且能捕捉强适用性 bisimilarity；

**⚠️ 局限性**

局限性包括：对所有高阶双函子未能保证最终煤气层存在，方法依赖于M-范畴的稳定性与收敛性；在无类型或复杂效应的语言中，唯一性不一定成立；未来工作需扩展到CPO-enriched范畴与更一般的抽象规格。

---

## 332. Improving Neural Topic Modeling with Semantically-Grounded Soft Label Distributions

**arXiv ID:** 2602.17907 | [PDF](https://arxiv.org/pdf/2602.17907v1)

**作者:** Raymond Li `[一作]` (University of British Columbia), Giuseppe Carenini `[通讯]` (University of British Columbia)

**通讯引用:** 6060 | [OpenAlex ID](https://openalex.org/A5049259877)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种利用语言模型生成软标签分布作为神经主题模型重构目标的方法；

**💡 创新点**

创新点在于用LM的下一词预测概率映射为语义丰富的软标签，并结合KL重构损失训练主题模型；

**🔧 技术方法**

采用了小型指令调优语言模型（Llama、Llama 2、Falcon）提取软标签和隐藏状态，结合ProdLDA架构；

**📊 数据集**

在20NewsGroup、TweetTopic和StackOverflow三个公开文本分类数据集上进行实验；

**📈 对比分析**

与LDA、ProdLDA、CombinedTM、ZeroshotTM、ETM、BERTopic、ECRTM、FASTopic等基线比较，本文方法在主题连贯度、主题纯度和检索精度上均实现SOTA或显著提升；

**⚠️ 局限性**

局限性包括对提示词的敏感性、对温度参数的调优依赖、以及在极短文本场景下仍需进一步提升模型鲁棒性。

---

## 333. Dual-Tree LLM-Enhanced Negative Sampling for Implicit Collaborative Filtering

**arXiv ID:** 2602.18249 | [PDF](https://arxiv.org/pdf/2602.18249v1)

**作者:** Jiayi Wu `[一作]` (Beijing Institute of Technology), Guoren Wang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 7295 | [OpenAlex ID](https://openalex.org/A5054991337)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了DTL-NS，一种文本无、无微调的LLM增强负采样框架，用于改进隐式协同过滤推荐系统。

**💡 创新点**

创新点在于：①用协作结构树与语义树将物品信息编码为层次路径表示，使LLM仅通过推理即可识别误负样本并转为正样本；②结合用户-物品偏好分数与物品-物品树路径相似度，设计多视角硬负采样方法，提升模型判别能力。

**🔧 技术方法**

核心技术包括：层次索引树构建、图谱谱嵌入、Llama3 LLM推理、混合负样本（mixup）、最短公共前缀相似度以及多视角融合的硬负采样评分。

**📊 数据集**

实验使用了 Amazon-sports、Amazon-toys、Yelp 三个真实数据集，以及两组含噪声的 synthetic 数据集（Amazon-sports-r、Amazon-toys-r）。

**📈 对比分析**

与多种负采样方法（RNS、DNS、MixGCF、SRNS、AHNS、HNLMRec）以及 LLM 增强推荐器（KAR、LLMRec、RLMRec）对比，DTL-NS 在 Recall@20、NDCG@20 等指标上提升约10–15%（Amazon-sports）和更显著的提升（synthetic dataset），展现出更优性能和更快收敛。

**⚠️ 局限性**

局限性包括：①需要一次性进行LLM推理，仍有一定的计算成本；②仅利用协作与语义两视角，未充分挖掘文本信息；③对树结构参数和候选集大小敏感，参数调优可能影响效果。

---

## 334. Optimal Multi-Debris Mission Planning in LEO: A Deep Reinforcement Learning Approach with Co-Elliptic Transfers and Refueling

**arXiv ID:** 2602.17685 | [PDF](https://arxiv.org/pdf/2602.17685v1)

**作者:** Agni Bandyopadhyay `[一作]` (Julius-Maximilians-Universität Würzburg), Gunther Waxenegger-Wilfing `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出统一的共轨椭圆机动框架，结合 Hohmann 转移、安全椭圆接近与加油逻辑，评估 Greedy、MCTS 与 Masked PPO 三种规划算法

**💡 创新点**

创新点在于将共轨椭圆机动与安全椭圆接近和显式加油逻辑整合为单一框架，并通过 Masked PPO 引入动作掩码以高效处理约束，显著提升多目标去除任务的性能

**🔧 技术方法**

采用 Masked Proximal Policy Optimization（强化学习）、Monte Carlo Tree Search、Greedy 贪心搜索、轨道动力学模拟（Poliastro/Astropy）、动作掩码技术

**📊 数据集**

使用随机生成的 50 目标碎片轨道参数，共 100 个随机测试案例（每个案例 10 次迭代）

**📈 对比分析**

在统一仿真环境下比较碎片访问数量与计算时间，Masked PPO 在碎片访问量上最高（29-32），计算时间最快（1-2 s）；MCTS 访问量次之（25-29）但耗时极长（1 k-10 k s）；Greedy 最慢访问量（15-18）且计算时间最短

**⚠️ 局限性**

局限性包括 MCTS 的计算成本过高不适合实时或车载；Greedy 方法短视，易错；实验忽略 J2 等扰动，缺乏更复杂动力学与真实碎片场景的验证

---

## 335. Digital self-Efficacy as a foundation for a generative AI usage framework in faculty's professional practices

**arXiv ID:** 2602.17673 | [PDF](https://arxiv.org/pdf/2602.17673v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 336. The Geometry of Noise: Why Diffusion Models Don't Need Noise Conditioning

**arXiv ID:** 2602.18428 | [PDF](https://arxiv.org/pdf/2602.18428v1)

**作者:** Mojtaba Sahraee-Ardakan `[一作]` (Google), Peyman Milanfar `[通讯]` (Google)

**通讯引用:** 19586 | [OpenAlex ID](https://openalex.org/A5002085979)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

探究无时间条件的生成模型（如Equilibrium Matching和blind diffusion）背后的能量景观与生成动力学，证明其实现的是对边缘能量的黎曼梯度流；

**💡 创新点**

首次将无噪声条件生成模型的目标定义为“边缘能量”（Marginal Energy），揭示其潜在梯度奇异性，并证明网络通过自适应度量（后验噪声方差）预处理该奇异性；

**🔧 技术方法**

利用高维几何集中性、后验收敛、Riemannian梯度流理论、能量分解以及不同参数化（噪声预测、信号预测、速度预测）的有效增益分析；

**📊 数据集**

CIFAR‑10、SVHN、Fashion‑MNIST以及二维同心圆嵌入高维空间的合成数据；

**📈 对比分析**

通过与有时间条件模型对比，发现无噪声条件下的噪声预测模型（DDPM Blind）在生成时结构不稳定；而无噪声条件下的速度预测模型（Flow Matching Blind）能够达到与有条件模型相近的FID（如CIFAR‑10 FID 2.61/2.23），验证理论；

**⚠️ 局限性**

局限性：理论主要聚焦于高维/近真数据点的收敛；在低维或噪声层级不明显的情形下，后验收敛缓慢，速度预测模型仍需更多训练；且对不同数据分布的普适性仍待进一步验证。

---

