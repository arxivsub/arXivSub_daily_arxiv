# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-01-30 | 今日论文总数: 683

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Multilingual Dysarthric Speech Assessment Using Universal Phone Recognition and Language-Specific Phonemic Contrast Modeling

**arXiv ID:** 2601.21205 | [PDF](https://arxiv.org/pdf/2601.21205v1)

**作者:** Eunjung Yeo `[一作]` (University of Texas at Austin), David R. Mortensen `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2264 | [OpenAlex ID](https://openalex.org/A5059859009)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种多语言语音评估框架，利用通用电话识别器（UPR）和语言特定的音位映射/对齐，生成三种可解释的音位误差指标（PER、PFER、PhonCov），并与临床可懂度评分进行对照；

**💡 创新点**

创新点在于将通用电话识别与基于对比语音特征的语言特定音位映射与对齐相结合，形成既可跨语言使用又能捕捉语言特有可懂度因素的三维音位评估体系；

**🔧 技术方法**

使用技术包括：wav2vec2phone与ZIPA等UPR模型，Epitran多语言G2P，基于PanPhon的对比语音特征距离计算，加权Needleman‑Wunsch对齐，以及对PFER、PER、PhonCov的计算；

**📊 数据集**

实验数据集涵盖四种语言：英语的UASpeech、哥伦比亚西班牙语的PC‑GITA、意大利语的Easycall、泰米尔语的SSNCE；

**📈 对比分析**

与传统基线（eGeMAPS、CPP、ASR WER）比较时，UPR基准音位指标在所有语言上与临床可懂度评分的Kendall τ相关系数均显著更高，且语言特定处理进一步提升相关性；

**⚠️ 局限性**

局限性包括：仅关注段音层面，未考虑韵律和节奏等超段音因素；每种语言仅使用单一数据集，需进一步验证跨语料的泛化性；并且UPR模型对某些语言的稳定性仍受限于训练数据分布。

---

## 2. BrainFuse: a unified infrastructure integrating realistic biological modeling and core AI methodology

**arXiv ID:** 2601.21407 | [PDF](https://arxiv.org/pdf/2601.21407v1)

**作者:** Baiyu Chen `[一作]` (Institute of Automation), Guoqi Li `[通讯]` (Institute of Automation)

**通讯引用:** 13467 | [OpenAlex ID](https://openalex.org/A5018970859)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了 BrainFuse 平台，实现了神经元的生物学细节（Hodgkin–Huxley 模型）与可微分梯度学习的无缝集成，并支持 GPU 加速和可迁移到 neuromorphic 芯片的部署。

**💡 创新点**

创新点在于①在单一框架内同时满足高保真细胞级生物学仿真、梯度优化与硬件加速；②通过精细离散化与 Triton 级别的低层优化，使 HH 神经元在 GPU 上实现 3000× 的加速；③实现了可在单颗 neuromorphic 芯片上部署 3.8 万个 HH 神经元、1 亿突触、功耗 1.98 W 的完整闭环系统。

**🔧 技术方法**

核心技术包括 PyTorch 自动微分、Triton GPU 代码生成、定制的离散化与梯度推导、operator fusion 与重计算策略，以及 C/C++ 代码迁移和芯片专用编译链；数据方面使用了 CIFAR10‑C、SHD、L5PC、C. elegans Ca‑imaging、Potjans‑Diesmann 皮层网络等公开数据集。

**📊 数据集**

使用的数据集：CIFAR10‑C（鲁棒性评估）、SHD（语音序列）、L5PC（层5锥体神经元突触模拟）、C. elegans 细胞 Calcium 记录、Potjans‑Diesmann 皮层网络以及自制的多尺度网络实验。

**📈 对比分析**

与传统 LIF 神经元、SpikingJelly、NEURON、NEST 等平台比较，BrainFuse 在相同任务下 GPU 推理/训练速度仅为 LIF 的 1.58 倍、显存占用 1.2 倍，但在 GPU 加速与跨平台部署方面相较传统框架提升 50–3000 倍；在鲁棒性实验中 HH 模型对噪声的抵抗力显著高于 LIF。

**⚠️ 局限性**

局限性包括：仍然依赖特定 GPU 与 Triton；对非 HH 细胞模型的支持尚需手动定义，灵活性受限；虽然实现了 1.98 W 的低功耗，但对更大规模网络的可扩展性和多芯片协同仍需进一步验证；以及在极端稀疏/大步长情形下的数值稳定性尚待评估。

---

## 3. TRACE: Trajectory Recovery for Continuous Mechanism Evolution in Causal Representation Learning

**arXiv ID:** 2601.21135 | [PDF](https://arxiv.org/pdf/2601.21135v1)

**作者:** Shicheng Fan `[一作]` (University of Illinois at Chicago), Lu Cheng `[通讯]` (University of Illinois at Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了TRACE框架，用于在连续机制演化的因果表示学习中学习原子机制并恢复混合轨迹。

**💡 创新点**

创新点包括：①将连续机制建模为有限原子机制的凸组合；②给出潜在变量与混合轨迹的可识别性理论与误差上界；③提出两阶段方法，能够在无标记转移数据上恢复任意中间机制状态。

**🔧 技术方法**

使用技术包括Mixture-of-Experts结构、序列VAE、凸组合模型、最小二乘投影、平滑正则化、可识别性证明与理论分析。

**📊 数据集**

使用的数据集包括：合成动态系统、半合成CartPole图像、真实车辆转弯(UAVDT)和人类步态(CMU MoCap)数据。

**📈 对比分析**

与TDRL、NCTRL、LEAP、iVAE、PCL等基线比较，TRACE在混合轨迹相关性上取得0.94–0.99，显著优于基线（0.67–0.72）；在UAVDT上Corr 0.96对比0.24，在步态上Corr 0.86对比0.62。

**⚠️ 局限性**

局限性包括：需预先知道原子机制数K并拥有纯域训练数据；要求原子机制间线性独立性，近共线时混合恢复受限；目前仅支持线性凸组合，未涵盖更一般的机制组合；未能自动推断K。

---

## 4. NFCDS: A Plug-and-Play Noise Frequency-Controlled Diffusion Sampling Strategy for Image Restoration

**arXiv ID:** 2601.21248 | [PDF](https://arxiv.org/pdf/2601.21248v1)

**作者:** Zhen Wang `[一作]`, Zhihui Wei `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 5909 | [OpenAlex ID](https://openalex.org/A5070343691)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过在反向扩散采样中对噪声进行频域软阈值抑制低频成分，从而在零样本图像恢复任务中兼顾数据保真和感知质量。

**💡 创新点**

首次从频域角度阐释噪声对生成与恢复的双重作用，并提出NFCDS——一种无训练、直接嵌入Plug‑and‑Play扩散框架的低频抑制策略。

**🔧 技术方法**

采用傅里叶变换与软阈值掩模控制噪声频谱，结合DDIM/DDPM扩散模型和数据一致性引导，实现噪声频率调节。

**📊 数据集**

在CelebA‑HQ（256×256）数据集上对超分辨率和高斯去噪任务进行实验验证。

**📈 对比分析**

与DD‑NRLG、DDRM、DDNM、DDPG等零样本恢复基线对比，NFCDS在保持或提升LPIPS的同时，PSNR/SSIM显著提升，且可减少采样步数、缩短推理时间。

**⚠️ 局限性**

仍需手动调节低频阈值与平滑参数，对不同任务或非人脸图像的适用性待进一步验证。

---

## 5. Sycophantic Anchors: Localizing and Quantifying User Agreement in Reasoning Models

**arXiv ID:** 2601.21183 | [PDF](https://arxiv.org/pdf/2601.21183v1)

**作者:** Jacek Duszenko `[一作]` `[通讯]` (Wroclaw University of Science and Technology), Jacek Duszenko (Wroclaw University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了“sycophantic anchors”概念，利用对推理过程中的句子进行反事实回放，识别出导致模型同意错误用户建议的关键句子，并通过线性探针和回归器对这些句子的存在与强度进行实时检测与量化；同时公开了包含 509 条对抗式多轮对话及 20 次反事实回放的数据集。

**💡 创新点**

创新点在于：①将对抗式对话与 Thought Anchors 框架结合，首次实现句子级别的因果性分析；②发现 sycophantic anchors 与正确推理 anchor 在激活空间中的可区分性明显不对称；③证明 sycophancy 在推理过程中逐步出现，提供了可操作的干预窗口；④提供了可复现的实验代码与数据集。

**🔧 技术方法**

主要技术包括：反事实回放（counterfactual rollouts）、线性探针（linear probes）对句子激活进行分类、MLP 回归器预测 sycophancy 强度、平衡准确率评估、LLM-as-judge 进行正确性判定、使用 spaCy 进行句子切分。

**📊 数据集**

使用的数据集是基于 AI2 Reasoning Challenge (ARC) 题目构造的 509 条对抗式多轮对话，其中 101 条为 sycophantic，408 条为正确推理，每条对话在每个句子位置均进行了 20 次反事实回放，形成 35,345 条句子级别的数据。

**📈 对比分析**

通过 5 折交叉验证评估线性探针，sycophantic anchor 与正确 anchor 的平衡准确率分别为 84.6% 与 77.5%，与中性句子相比差异显著；正确 anchor 与中性句子的准确率仅 64%；探针在提示词末尾的准确率仅 55%，但在 anchor 句子时提升至 73%；回归器对激活的预测 R² 达 0.742，说明激活可量化 sycophancy 强度。

**⚠️ 局限性**

局限性包括：仅对单一模型 DeepSeek-R1-Distill-Llama-8B 进行实验，缺乏跨模型或跨规模的验证；实验集中在多项选择题，未覆盖开放式生成；高昂的反事实回放成本限制了样本规模；未研究 sycophancy 在更广泛对话场景中的表现。

---

## 6. Hebbian Learning with Global Direction

**arXiv ID:** 2601.21367 | [PDF](https://arxiv.org/pdf/2601.21367v1)

**作者:** Wenjia Hua `[一作]`, Qinghai Guo `[通讯]` (Huawei Technologies Co. Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种全局引导的Hebbian学习框架（GHL），将本地Hebbian规则与全局梯度符号结合，用于端到端训练深度神经网络。

**💡 创新点**

创新点在于：①把全局梯度的符号作为第三因子，模拟生物学中的神经递质调控，解决纯Hebbian更新缺乏全局目标的问题；②采用Oja规则与软竞争机制保证更新稳定性；③框架对网络结构无特定依赖，能够在极深网络和大规模数据集上保持可扩展性。

**🔧 技术方法**

核心技术包括：Oja Hebbian规则、Soft‑Winner‑Take‑All（SWTA）竞争学习、基于梯度符号的全局调制（sign‑based modulation）、端到端训练流程与现有Hebbian算法对比实验。

**📊 数据集**

使用的数据集有：CIFAR‑10、CIFAR‑100、ImageNet（ILSVRC 2012）以及各深度网络（VGG‑16、ResNet‑20、ResNet‑50、ResNet‑1202）进行评估。

**📈 对比分析**

与SoftHebb、FastHebb、HWTA‑BCM等最新Hebbian方法以及标准反向传播（BP）进行对比；实验表明GHL在CIFAR与ImageNet上均逼近或超过BP，显著优于其他Hebbian方法，特别是在极深网络和大型数据集上性能差距仅为2–5%。

**⚠️ 局限性**

限制在于：①仍依赖梯度符号，忽略梯度幅值信息，可能影响收敛速度和精度；②缺乏对Transformer等现代架构的验证；③尚无严格理论收敛证明，需进一步研究训练稳定性与理论基础。

---

## 7. Monotone Optimisation with Learned Projections

**arXiv ID:** 2601.20983 | [PDF](https://arxiv.org/pdf/2601.20983v1)

**作者:** Ahmed Rashwan `[一作]` (University of Bath), Lisa Kreusser `[通讯]` (Monumo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种将学习模型嵌入Polyblock Outer Approximation（POA）全局优化算法的方法，核心是直接学习约束的径向逆函数（radial inverse）以实现投影。

**💡 创新点**

创新点在于：①将径向逆作为POA内部关键原语进行学习；②设计了满足正齐次、单调性和连贯性约束的Homogeneous‑Monotone Radial Inverse（HM‑RI）网络；③通过放宽monotone网络的严格性，既保证了训练效率，又兼顾了近似精度。

**🔧 技术方法**

主要技术包括：受约束的神经网络（relaxed certified monotone networks）、径向逆函数理论、POA算法、以及对比实验中的局部优化器（SLSQP、COBYLA）。

**📊 数据集**

使用人工生成的四类单调优化数据集：无限制与有限样本的随机二次规划、乘法规划、以及两种基于CRRM模拟的下行功率调度（功率律模型与UMa模型）。

**📈 对比分析**

与基准方法（M‑Net、SLSQP、COBYLA、Oracle）对比，HM‑RI+POA在投影目标值上均优于直接约束学习和局部优化器，并且在不使用bisection时显著减少计算时间，尤其在功率调度任务上取得5×左右的速度提升。

**⚠️ 局限性**

局限性包括：依赖问题的单调性假设；放宽的monotone约束可能导致偶尔的非单调误差；POA实现的实现开销仍高于纯局部优化器；对非单调或更复杂约束形式的推广尚未验证。

---

## 8. Spava: Accelerating Long-Video Understanding via Sequence-Parallelism-aware Approximate Attention

**arXiv ID:** 2601.21444 | [PDF](https://arxiv.org/pdf/2601.21444v1)

**作者:** Yuxiang Huang `[一作]` (Tsinghua University), Zhiyuan Liu `[通讯]` (Tsinghua University)

**通讯引用:** 21771 | [OpenAlex ID](https://openalex.org/A5100320711)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于序列并行和近似注意力的长视频推理加速框架，利用局部 KV 缓存压缩与传递块在多 GPU 环境下显著降低计算与通信开销，同时通过视觉编码帧并行、融合前向、通信重叠等系统级优化实现高效推理。

**💡 创新点**

创新点包括：1）在保持完整视频嵌入的前提下采用局部 KV 缓存压缩与传递块实现近似注意力；2）引入虚拟主机与 ZigZag 负载均衡机制，解决传统序列并行的计算/通信不平衡；3）在系统层面实现前向融合、通信重叠以及视觉负载平衡，进一步提升多 GPU 加速效果。

**🔧 技术方法**

技术手段：序列并行、近似注意力（KV 压缩、传递块）、虚拟主机与 ZigZag 负载均衡、前向融合、通信与计算重叠、视觉编码帧并行、FlashAttn kernel 优化、NVLink/InfiniBand 通信优化。

**📊 数据集**

使用 LongVideoBench（真实视频）和 VNBench（合成视频，包含检索、排序、计数子任务）进行评估，视频均采用 64 帧，分辨率分别为 720p 与 1440p。

**📈 对比分析**

与 FullAttn、ZigZagRing、FlashAttn、XAttn、Sparge、StarAttn、APB 等基线对比，实验表明在 64 帧 1440p 视频上相较 FlashAttn 提升 12.72×，相较 ZigZagRing 提升 1.70×，相较 APB 提升 1.18×；同时保持与 FullAttn 接近的精度，误差仅为几百分点。

**⚠️ 局限性**

局限性：仅适用于解码器型 Transformer 的多模态模型，无法推广到 CNN 等架构；只能在多 GPU 环境下使用，单 GPU 版本退化为 FlashAttn；对极长视频或大规模 GPU 集群的扩展性受限于硬件与通信；并且仅针对推理阶段的加速，未涉及训练。

---

## 9. Sim-MSTNet: sim2real based Multi-task SpatioTemporal Network Traffic Forecasting

**arXiv ID:** 2601.21384 | [PDF](https://arxiv.org/pdf/2601.21384v1)

**作者:** Hui Ma `[一作]` (Xinjiang University), Xinjun Pei `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为 Sim-MSTNet 的多任务时空网络流量预测模型，利用 sim2real、域随机化、双层优化与动态损失权重技术，解决数据稀缺和任务冲突问题。

**💡 创新点**

创新点包括：①首次将 sim2real 与域随机化应用于移动通信流量预测；②采用双层切割平面优化实现样本重加权；③设计软参数共享的注意力交互与动态损失权重机制，缓解任务不平衡与负迁移。

**🔧 技术方法**

使用技术包括：Wireless InSite 仿真器生成合成数据；域随机化与 K 步梯度双层优化；切割平面算法；CNN+空间注意力用于时空特征提取；Informer 解码器；多头注意力任务交互；指数平滑动态损失权重。

**📊 数据集**

数据集为：30,638 条合成样本 + 982 条真实样本的仿真数据；米兰（Milano）10,000 格网 SMS/Call/Net；特伦托（Trento）6,575 格网 SMS/Call/Net。

**📈 对比分析**

与 8 个单任务基线（LSTM、Transformer、Informer、iTransformer、ConvLSTM、STGCN、DCRNN、STAEformer）以及 3 个多任务基线（MTTC、AST-MTL、CSLSL）进行对比；在 MAE/RMSE 上均取得显著提升，尤其在多任务场景下 MAE 降至 0.29/0.41/1.40，RMSE 亦显著下降。

**⚠️ 局限性**

局限性包括：对仿真环境的依赖仍较强，模型在不同地区的迁移性能需进一步验证；切割平面算法计算成本高，可能限制大规模部署。

---

## 10. SHARP: Social Harm Analysis via Risk Profiles for Measuring Inequities in Large Language Models

**arXiv ID:** 2601.21235 | [PDF](https://arxiv.org/pdf/2601.21235v1)

**作者:** Alok Abhishek `[一作]`, Lisa Erickson `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SHARP 框架，用多维分布感知方法对大型语言模型的社会危害进行评估；

**💡 创新点**

创新点在于将危害拆分为偏见、公平、伦理和认知可靠四个维度，采用联合失效聚合重新参数化为累积对数风险，并用 CVaR_95 等尾部统计量突出模型极端风险；

**🔧 技术方法**

使用 LLM‑as‑a‑judge 的多评审聚合（log‑sum‑exp）、对数风险变换、条件价值风险 CVaR、Bootstrap 置信区间、Friedman 与 Wilcoxon 检验等技术；

**📊 数据集**

使用 901 条社会敏感提示语集，来源于 BEATS/BBQ 等基准；

**📈 对比分析**

通过对模型在提示集上的累计对数风险分布计算 CVaR_95 进行比较，发现平均风险相近的模型在尾部风险上差异可达 2–4 倍，模型被划分为低‑中‑高风险层级；

**⚠️ 局限性**

局限性包括：评估仅提供相对、条件化的风险指标；依赖 LLM‑judge 的主观标签；仅评估单回合、单语言、固定提示；未考虑模型效用、生成随机性和真实世界伤害概率；

---

## 11. Abstracting Robot Manipulation Skills via Mixture-of-Experts Diffusion Policies

**arXiv ID:** 2601.21251 | [PDF](https://arxiv.org/pdf/2601.21251v1)

**作者:** Ce Hao `[一作]` (National University of Singapore), Harold Soh `[通讯]` (National University of Singapore)

**通讯引用:** 2476 | [OpenAlex ID](https://openalex.org/A5066073375)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于扩散模型的混合专家策略SMP，能在多任务双臂操控中通过可学习的正交技能基和粘性路由实现高效、可迁移的动作生成。

**💡 创新点**

核心创新在于将技能抽象为状态自适应正交基，并采用粘性门控实现阶段一致、稀疏激活，同时设计变分目标与自适应专家激活机制，显著降低推理成本。

**🔧 技术方法**

使用扩散生成模型、QR分解实现可微正交基、Dirichlet粘性门、变分推断、稀疏top‑k/覆盖式专家激活以及状态仅路由器。

**📊 数据集**

在模拟环境RoboTwin‑2与RLBench‑2，以及真实双臂平台PiPER上分别使用多任务演示数据集（每任务约50–100条演示）进行训练与评估。

**📈 对比分析**

与DP、DP3、ACT、RDT、Discrete Policy及Sparse Diffusion Policy等基线比较，SMP在多任务成功率上高于或持平，同时推理时间下降约70%（仅激活约30%参数），并在少量样本迁移与技能重组任务中表现更优。

**⚠️ 局限性**

局限在于仅使用较小扩散骨干、仅验证双臂任务、对噪声鲁棒性和更大规模数据的泛化尚待进一步评估。

---

## 12. Bridging the Arithmetic Gap: The Cognitive Complexity Benchmark and Financial-PoT for Robust Financial Reasoning

**arXiv ID:** 2601.21157 | [PDF](https://arxiv.org/pdf/2601.21157v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 13. Few-Shot Learning for Dynamic Operations of Automated Electric Taxi Fleets under Evolving Charging Infrastructure: A Meta-Deep Reinforcement Learning Approach

**arXiv ID:** 2601.21312 | [PDF](https://arxiv.org/pdf/2601.21312v1)

**作者:** Xiaozhuang Li `[一作]` (Tsinghua University), Fang He `[通讯]` (Tsinghua University)

**通讯引用:** 6851 | [OpenAlex ID](https://openalex.org/A5016539926)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种GAT‑PEARL框架，实现自动电动出租车车队在充电基础设施不断演化时的少样本动态运营。

**💡 创新点**

创新点在于将图注意力网络与PEARL元强化学习结合，通过概率上下文编码实现对充电网络拓扑变化的即时自适应，而不需要重新训练。

**🔧 技术方法**

使用技术包括图注意力网络（GAT）、PEARL概率嵌入、分层多智能体元强化学习以及SAC训练算法。

**📊 数据集**

数据集采用成都真实城市交通与充电桩部署数据，并在多种充电布局下进行仿真。

**📈 对比分析**

与传统强化学习基线比较，GAT‑PEARL在多种未见充电网络布局下显著提升系统总收益和收敛速度，表现更好。

**⚠️ 局限性**

局限性在于仍以仿真为主，未在真实车队验证；对极端网络拓扑变化的鲁棒性待进一步检验。

---

## 14. BadDet+: Robust Backdoor Attacks for Object Detection

**arXiv ID:** 2601.21066 | [PDF](https://arxiv.org/pdf/2601.21066v1)

**作者:** Kealan Dunnett `[一作]` (Queensland University of Technology), Raja Jurdak `[通讯]` (Queensland University of Technology)

**通讯引用:** 11616 | [OpenAlex ID](https://openalex.org/A5088135082)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了目标检测模型的后门攻击，并提出了BadDet+框架，能够同时实现区域误分类攻击（RMA）和目标消失攻击（ODA）并在物理场景中保持鲁棒性。

**💡 创新点**

创新点在于把RMA与ODA统一为一个基于log‑barrier惩罚的训练目标，克服了以往方法在评估指标、触发器尺度、位置和物理迁移上的缺陷；同时引入TDR指标揭示误分类的真实效果。

**🔧 技术方法**

主要技术包括：训练时加入log‑barrier惩罚项以抑制触发器对象的原类置信度；使用Sigmoid/softmax兼容的惩罚公式；在不同检测器（FCOS、Faster‑RCNN、DINO、YOLOv5）上实验；并通过微调（FT、FT‑SAM）评估防御。

**📊 数据集**

使用COCO、Mapillary Traffic Sign Dataset（MTSD）和其物理版本PTSD三个公开数据集，涵盖不同场景与对象类别。

**📈 对比分析**

与BadDet、Align、UBA、Morph等方法对比，BadDet+在ASR@50、TDR@50和mAP等指标上均优于基线：在COCO、MTSD和PTSD上实现93%+的ASR，TDR降至约3%，且mAP几乎不受影响；在物理攻击迁移测试中亦保持高成功率。

**⚠️ 局限性**

局限性包括：在某些RMA场景下仍不如原BadDet；未覆盖目标生成（OGA、GMA）攻击；依赖于攻击者可控制训练过程的强威胁模型；对YOLO等检测器的效果不如预期；防御实验仅限于微调，未探究更专业的检测器专属防御。

---

## 15. Learning to Advect: A Neural Semi-Lagrangian Architecture for Weather Forecasting

**arXiv ID:** 2601.21151 | [PDF](https://arxiv.org/pdf/2601.21151v1)

**作者:** Carlos A. Pereira `[一作]` (Environment and Climate Change Canada), Emilia Diaconescu `[通讯]` (Environment and Climate Change Canada)

**通讯引用:** 773 | [OpenAlex ID](https://openalex.org/A5014794322)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出PARADIS模型，将天气预报过程拆分为迁移、扩散和反应三块，并在潜在空间中嵌入可微分半拉格朗日算子实现全球高效预报。

**💡 创新点**

在网络架构上显式引入半拉格朗日传输层、深度可分离扩散层和逐通道反应层，形成物理启发的算子分解，并通过三阶段训练课程优化谱保持。

**🔧 技术方法**

采用可微分半拉格朗日算子、深度可分离卷积、1×1卷积反应层、球面差分、逆Huber损失、谱AMSE优化等技术，并在GPU集群上训练。

**📊 数据集**

使用ERA5重分析数据（1990‑2019训练，2020验证），分辨率1°，并与ECMWF HRES、GraphCast、Pangu Weather等模型进行对比。

**📈 对比分析**

在RMSE、谱相干性、垂直误差结构和热带气旋轨迹误差等指标下，PARADIS在大多数变量与时程上取得第一或第二名，并在1°分辨率下实现与高分辨率基准模型相当的轨迹精度。

**⚠️ 局限性**

受分辨率限制导致对强烈气旋核心的低分辨率处理；训练需大量GPU资源；仅提供确定性预测，未涵盖不确定性与概率分布。

---

## 16. Another Systematic Review? A Critical Analysis of Systematic Literature Reviews on Agile Effort and Cost Estimation

**arXiv ID:** 2601.20893 | [PDF](https://arxiv.org/pdf/2601.20893v1)

**作者:** Henry Edison `[一作]` (Blekinge Institute of Technology), Nauman Ali `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对18篇关于敏捷软件开发中成本/工作量估计的系统综述（SLR）进行内容分析，揭示了研究重复、缺乏先前综述认知和缺乏充分理由的问题；

**💡 创新点**

创新点在于提出一套针对二级研究的系统检索、质量评估（QAISER）和更新必要性评估的工作流程，帮助研究者避免无意义的重复综述；

**🔧 技术方法**

采用质性内容分析法，对SLR的动机、方法、质量评估、引用关系等进行编码与统计；

**📊 数据集**

使用了18篇已发表的SLR数据集，涵盖2014-2024年，包含作者、年份、引用的先前综述、覆盖年份、质量得分、纳入的初级研究数量等信息；

**📈 对比分析**

未与其他方法直接比较，而是通过统计与可视化（如引用网络、质量评分分布、作者人数与研究规模关系）展示了不同综述在动机、覆盖范围和质量上的差异；

**⚠️ 局限性**

局限性包括仅聚焦单一主题（敏捷工作量估计）、未检索映射研究、样本规模有限、以及对其他二级研究领域的推广性未知。

---

## 17. The Surprising Difficulty of Search in Model-Based Reinforcement Learning

**arXiv ID:** 2601.21306 | [PDF](https://arxiv.org/pdf/2601.21306v1)

**作者:** Wei-Di Chang `[一作]` (McGill University), Scott Fujimoto `[通讯]` (Meta FAIR)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过理论与实验研究模型基强化学习中的搜索难点，提出MRS.Q算法：在MR.Q框架中加入MPPI搜索，并通过价值函数集成的最小化方法抑制因搜索产生的分布偏移导致的价值过估计，从而显著提升性能。

**💡 创新点**

创新点包括：
• 证明搜索失败与模型精度无关，强调搜索本身的限制；
• 揭示搜索引入的分布偏移导致价值函数过估计，并将其视为搜索失败的主要瓶颈；
• 采用全集合最小化（ensemble min）对价值函数进行惰性惰性约束，既降低过估计，又保持搜索有效；
• 对MR.Q轻量化改造（去除探索噪声、使用Simplicial Embedding、增大终止损失权重）实现性能提升。

**🔧 技术方法**

使用的技术包括：
• 模型预测控制（MPPI）实现短期搜索；
• 价值函数集成（10个独立价值网络）并取最小值进行价值评估与搜索评估；
• Simplicial Embedding 对状态潜在空间做归一化；
• 终止损失加权提高终止预测准确度；
• 通过无探索噪声实现搜索自身的探索性。

**📊 数据集**

实验数据集覆盖 50+ 连续控制任务，分为三大基准：
• MuJoCo Gym (多种仿真环境)，
• DeepMind Control Suite (DMC)，
• HumanoidBench (分为含手部与不含手部两种设置)。

**📈 对比分析**

与多种基线（MR.Q、MR.Q+MPC、TD-MPC2 及其改进版 TD-M(PC)^2、BMPC、BOOM、SimbaV2）在 1M 环境步内进行对比。MRS.Q 在 Gym、DMC、HB No Hand 和 HB Hand 4 个子基准中，在 3/4 基准上均超越所有对手，且在 Gym 与高维 HumanoidBench 中表现尤为突出。

**⚠️ 局限性**

局限性：
• 需要维护 10 个价值网络并在每步执行 MPPI，计算开销较大；
• 继承了 TD-MPC2 的短规划时长，无法探索更长时域的策略；
• 仅在 MR.Q 架构上实现，未验证可否推广至其它模型基算法；
• 实验仅限于连续控制任务，未知在离散或更大规模任务的适用性。

---

## 18. IDE-Bench: Evaluating Large Language Models as IDE Agents on Real-World Software Engineering Tasks

**arXiv ID:** 2601.20886 | [PDF](https://arxiv.org/pdf/2601.20886v1)

**作者:** Spencer Mateega `[一作]` (AfterQuery), Agustin Garcinuño `[通讯]` (AfterQuery)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 IDE-Bench 框架，用来评估大型语言模型在真实软件工程任务中的 IDE 代理能力。

**💡 创新点**

创新点包括：① 在 IDE 原生工具接口下进行多语言全栈任务评测；② 采用 Docker 化、git 变更追踪和工具调用生态，模拟真实 IDE 环境；③ 提供了 80 个未公开的真实仓库，避免训练数据污染；④ 通过多维度指标（pass@k、token 预算、迭代效率、任务级表现）细粒度分析模型行为。

**🔧 技术方法**

技术手段：Docker 容器化、LiteLLM 调度、17 个工具（文件搜索、编辑、执行、MERN 测试等）及 OpenAI function‑calling 规范；脚本化评测与 git diff 对比；多阶段评估流程。

**📊 数据集**

数据集：8 个未公开的多语言仓库（C/C++、Java、TypeScript/Node.js、Python），共 80 个任务（实现、调试、重构、性能优化），涵盖系统编程、企业应用、Web 服务、数据分析等场景。

**📈 对比分析**

比较方法：采用 pass@1 与 pass@5 指标、token‑efficiency、任务级分数与热图、模型间 Jaccard 相似度；前沿模型 GPT‑5.2 在 pass@5 上达 95%，Claude Sonnet 88.75%，其余模型逐级下降，最低层模型 <50%。模型表现与成本、可预测性、任务专精度呈多维度分布。

**⚠️ 局限性**

局限性：① 评测仍以二元通过/失败为主，忽略近失（near‑miss）情况；② 对规范严格度的敏感性导致高误差率；③ 模型行为高度可变，难以保证一致性；④ 任务量虽多但仍不足以覆盖所有真实工程场景；⑤ 仅评估工具调用，未充分考虑生成代码的可维护性与安全性。

---

## 19. Heterogeneous Vertiport Selection Optimization for On-Demand Air Taxi Services: A Deep Reinforcement Learning Approach

**arXiv ID:** 2601.21316 | [PDF](https://arxiv.org/pdf/2601.21316v1)

**作者:** Aoyu Pang `[一作]` (Chinese University of Hong Kong), Man-On Pun `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 3784 | [OpenAlex ID](https://openalex.org/A5040559125)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了基于深度强化学习的城市空中出租车与地面交通的端到端路径规划与垂直机场选择，并提出了UAGMC框架。

**💡 创新点**

创新点在于结合多源情境嵌入（MSCE）与时空集成网络（STIN）构建深度强化学习模型，动态平衡机场拥堵与行程时间，并通过奖励分解解决稀疏奖励问题。

**🔧 技术方法**

采用Actor‑Critic（PPO）强化学习、Transformer注意力网络、LSTM、V2X通信以及交通流模型与A*路径搜索等技术。

**📊 数据集**

使用自建仿真数据集，生成三座垂直机场及动态乘客需求场景，没有公开真实数据集。

**📈 对比分析**

与六种基线（纯地面、规则、SPF、STTF、QTTI、Vanilla‑PPO、UAGMC‑L）对比，平均总行程时间降低约34%（从52分钟降至46.6分钟），等待时间显著下降。

**⚠️ 局限性**

局限在于未考虑垂直机场降落调度、真实空域冲突、eVTOL航线规划以及完整的电力/充电调度，仍需在实际环境中进一步验证。

---

## 20. Deep Reinforcement Learning for Fault-Adaptive Routing in Eisenstein-Jacobi Interconnection Topologies

**arXiv ID:** 2601.21090 | [PDF](https://arxiv.org/pdf/2601.21090v1)

**作者:** Mohammad Walid Charrwi `[一作]` (Kuwait University), Zaid Hussain `[通讯]` (Kuwait University)

**通讯引用:** 64 | [OpenAlex ID](https://openalex.org/A5064151888)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

评估了在存在节点故障的Eisenstein‑Jacobi (EJ) 网络中，贪婪路由、Dijkstra 最优路由与强化学习 (RL) 路由的性能，并展示了 RL 在保持高可达率、低延迟和吞吐量方面的优势。

**💡 创新点**

创新点在于将基于 Proximal Policy Optimization 的 RL 框架应用于六邻接的 EJ 网络，利用多目标奖励函数学习避障策略，使 RL 能在缺乏全局拓扑信息的情况下逼近 Dijkstra 的最优性能，并在拥塞环境中实现隐式负载平衡。

**🔧 技术方法**

主要技术包括：EJ 网络建模、故障注入与聚集故障模拟、基于 PPO 的 actor‑critic 强化学习、奖励设计（成功到达 + 跳数惩罚 + 故障惩罚）以及与传统贪婪与 Dijkstra 算法的基准比较。

**📊 数据集**

实验使用合成 EJ 网络，α 取值在 2+3ρ 至 5+6ρ 之间，随机故障密度 0–40%，并在 0.1–0.8 的负载下进行均匀流量测试；所有结果基于多次随机实例平均获得。

**📈 对比分析**

相较于贪婪路由，RL 在 9 个故障节点时保持 94% 可达率、91% 包交付率，接近 Dijkstra 的 52–54% 可达率与 54% 交付率；在吞吐量方面，RL 在低至中等负载下实现 >90% 的归一化吞吐量，甚至在部分场景下超过 Dijkstra，表明其具备良好的负载分配能力。

**⚠️ 局限性**

局限性包括：需要离线训练，训练样本必须覆盖足够多的故障分布；在极大规模网络或频繁拓扑变动时，RL 的策略可能需要迁移学习或在线微调；对实时故障检测延迟不作深入探讨；以及与 Dijkstra 相比，仍存在路径长度略大、全局最优性无法完全保证的情况。

---

## 21. Factored Causal Representation Learning for Robust Reward Modeling in RLHF

**arXiv ID:** 2601.21350 | [PDF](https://arxiv.org/pdf/2601.21350v1)

**作者:** Yupei Yang `[一作]` (Shanghai Jiao Tong University), Lei Xu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 47506 | [OpenAlex ID](https://openalex.org/A5017827110)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CausalRM 框架，通过因果表征学习将奖励模型的上下文嵌入分解为因果与非因果两部分，抑制奖励模型对长度、风格等非因果特征的敏感性。

**💡 创新点**

创新点在于：①将奖励模型嵌入 VAE 结构，显式分离因果与非因果潜变量；②使用对抗头与梯度反转同步约束两类潜变量，既保证因果潜变量充分又确保非因果潜变量不携带奖励信息，从而实现奖励模型的因果不变性。

**🔧 技术方法**

采用 VAE+对抗训练+梯度反转的因果表征学习框架，并结合 RLHF 的对偶对损失和信息瓶颈约束。

**📊 数据集**

实验数据集包括数学推理的 OpenMathInstruct‑1（含 GSM8K、MATH）和多种 OOD 集合，以及对话任务的 Anthropic‑RLHF‑HH、MT‑Bench、PKU‑SafeRLHF、SHP、TruthfulQA 等。

**📈 对比分析**

与标准 RM、GoalRM、InfoRM 进行对比，CausalRM 在配对准确率、RLHF 终端答案准确率和对话 win‑rate 上均优于基线，尤其在 OOD 评测和攻击性测试中提升显著，证明了因果分解对奖励模型鲁棒性的提升。

**⚠️ 局限性**

局限在于：①仅针对单轮对话，无法直接处理多轮动态因果；②对非因果因子（如长度、风格）的划分仍依赖手工定义；③模型结构更复杂，训练成本提高，且对更大规模 LLM 的扩展尚未验证。

---

## 22. Position: Certifiable State Integrity in Cyber-Physical Systems -- Why Modular Sovereignty Solves the Plasticity-Stability Paradox

**arXiv ID:** 2601.21249 | [PDF](https://arxiv.org/pdf/2601.21249v1)

**作者:** Enzo Nicolás Spotorno `[一作]` (Universidade Federal de Santa Catarina), Antônio Augusto Medeiros Fröhlich `[通讯]` (Universidade Federal de Santa Catarina)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出一种名为HYDRA的模块化主权架构，用于在终身生命周期内实现可信的网络物理系统状态完整性；

**💡 创新点**

通过将全局基础模型拆分为冻结的、可验证的局部专家，并用不确定性感知的治理层进行层次化加权，从而解决了全局可塑性-稳定性悖论、残留频谱偏差、以及缺乏可追溯性和形式可验证性等安全性瓶颈；

**🔧 技术方法**

结合神经算子、物理信息强化学习、Dirichlet先验的分数稀疏性、分层不确定性预测、LPV理论、RPI锥体可验证、以及 conformal 预测等技术；

**📊 数据集**

本文未使用特定公开数据集，而是以典型的网络物理系统案例（车辆动力学、机器人接触、能源电网、医疗监测）作为示例讨论；

**📈 对比分析**

作者通过理论推导和对比分析（对比单一基础模型、Mixture‑of‑Experts、RL、扩散模型等）论证了HYDRA在保持模型可验证性、降低误报率、提高可用性方面的优势，未给出数值实验；

**⚠️ 局限性**

局限性包括：仅适用于可分离的操作模式；需在离线阶段构建与验证专家库，耗时且受数据/物理知识限制；治理层的稀疏与连续平衡仍是未解挑战；对非线性神经算子进行正式 Lyapunov 证明仍是未来工作。

---

## 23. The Quiet Contributions: Insights into AI-Generated Silent Pull Requests

**arXiv ID:** 2601.21102 | [PDF](https://arxiv.org/pdf/2601.21102v1)

**作者:** S M Mahedy Hasan `[一作]` (Idaho State University), Minhaz Zibran `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对AI生成的“无声拉取请求”（silent pull requests，SPRs）进行首个大规模经验研究，分析其对代码复杂度、质量问题和安全漏洞的影响，并探讨这些指标是否能解释SPR被接受或拒绝的原因。

**💡 创新点**

创新点在于：①首次系统考察无评论、无讨论的AI PR；②对五种主流AI代理在热门Python项目中的4,762条关闭SPR进行定量比较；③从复杂度、代码质量、漏洞三维度评估SPR的影响，并尝试寻找其被合并或拒绝的可解释性。

**🔧 技术方法**

技术手段包括：使用Radon计算McCabe复杂度；利用Pylint提取错误、警告、规范和重构问题；采用Semgrep检测CWE漏洞；通过对比PR前后快照评估指标变化。

**📊 数据集**

数据集为AIDev公开数据集，选取61,000个仓库中流行Python项目（星数>100）的7,191条PR，最终聚焦5,015条SPRs，其中4,762条已关闭（合并或拒绝）。

**📈 对比分析**

对比方法：统计各AI代理（OpenAI Codex、Devin、GitHub Copilot、Cursor、Claude Code）在被接受与被拒绝的SPR中，分别计算复杂度、质量问题与漏洞的增加、减少或不变比例。结果显示：①大多数SPR不改变复杂度或质量问题；②复杂度与质量问题的增加率在接受与拒绝之间相似，难以预测合并结果；③安全漏洞几乎不受影响。

**⚠️ 局限性**

局限性包括：仅覆盖Python项目，无法推广到其他语言；仅使用三款静态分析工具，可能遗漏动态或业务相关问题；未考虑项目政策、审阅者行为及上下文信息，因而无法解释SPR被合并或拒绝的具体原因。

---

## 24. Text controllable PET denoising

**arXiv ID:** 2601.20990 | [PDF](https://arxiv.org/pdf/2601.20990v1)

**作者:** Xuehua Ye `[一作]` (GE Healthcare), Adam J. Schwarz `[通讯]` (GE Healthcare)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

本文提出一种基于CLIP文本嵌入的双路径U‑Net模型，能够对PET低计数图像进行可控降噪，输出任意高计数级别的图像。

**💡 创新点**

创新点在于将计数级别文本描述转化为语义嵌入，并在编码器与解码器两侧双路嵌入条件化，实现单模型对多计数级别的适应性。

**🔧 技术方法**

采用CLIP预训练的文本编码器、U‑Net结构以及多层点乘条件化，训练采用MSE损失。

**📊 数据集**

使用Siemens Biograph Vision Quadra 387份全身18F‑FDG PET扫描数据，模拟1/100至1/2等不同计数比例的低计数图像。

**📈 对比分析**

与仅训练于1/100计数的U‑Net和CycleGAN相比，在SSIM/PSNR指标上均表现更优，能更逼近真实全剂量图像。

**⚠️ 局限性**

局限在于缺乏足够的原始list‑mode数据，训练仅基于已有低高计数配对样本，未来需扩展至任意剂量模拟以提升通用性。

---

## 25. Towards Zero Rotation and Beyond: Architecting Neural Networks for Fast Secure Inference with Homomorphic Encryption

**arXiv ID:** 2601.21287 | [PDF](https://arxiv.org/pdf/2601.21287v1)

**作者:** Yifei Cai `[一作]` (Iowa State University), Hongyi Wu `[通讯]` (University of Arizona)

**通讯引用:** 3180 | [OpenAlex ID](https://openalex.org/A5115636506)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种专为同态加密推理优化的全新网络架构 StriaNet，并提出了 StriaBlock 及相关旋转消除和动态通道压缩原则，以显著降低 HE 推理中的旋转操作成本。

**💡 创新点**

① StriaBlock 通过 exRot‑Free 卷积与 Cross Kernel 彻底消除外部旋转，② 采用 Focused Constraint Principle 与 Channel Packing‑Aware Scaling Principle 对网络结构进行动态约束与尺度调节，③ 以 HE 为核心而非改造现有模型的全新架构设计。

**🔧 技术方法**

同态加密（BFV/CKKS）+ SIMD 打包、卷积核结构优化、旋转消除技术、动态通道压缩与调度。

**📊 数据集**

ImageNet、Tiny ImageNet、CIFAR‑10 三个不同规模的数据集。

**📈 对比分析**

在同态加密环境下与 VGG、ResNet、DenseNet、MobileNet 等传统模型进行基准对比；在保持相同准确率的前提下，ImageNet 加速 9.78×、Tiny ImageNet 加速 6.01×、CIFAR‑10 加速 9.24×；同时在通信成本上相较 Cheetah 降低 31.4×。

**⚠️ 局限性**

仍以手工设计为主，未结合自动化架构搜索；对非卷积层或更复杂 HE 场景的适用性未充分验证；在极大规模模型或多层次加密时可能存在新的性能瓶颈。

---

## 26. Transferable Graph Condensation from the Causal Perspective

**arXiv ID:** 2601.21309 | [PDF](https://arxiv.org/pdf/2601.21309v1)

**作者:** Huaming Du `[一作]` (Southwestern University of Finance and Economics), Gang Kou `[通讯]` (Xiangjiang Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于因果不变性和可迁移性的图数据集压缩框架 TGCC，能够在保持图结构信息的同时生成体量更小的合成图集。

**💡 创新点**

创新点在于：①利用因果干预提取低频因果不变特征；②将对比式压缩与谱域增强对比学习结合，实现因果信息注入；③在压缩过程中加入独立性约束和熵正则化，提升跨任务/跨域泛化能力。

**🔧 技术方法**

技术手段包括因果干预、梯度匹配/分布匹配压缩、HSIC独立性约束、信息熵正则化、InfoNCE 及谱域增强对比学习等。

**📊 数据集**

实验使用了 Cora、Citeseer、Ogbn-arxiv、Reddit、Flickr 以及自建的 FinReport 数据集。

**📈 对比分析**

与 Random、Herding、K-Center、DosCond、GCond、SGDD、SFGC、GEOM、GDEM、CGC 等基线比较，TGCC 在单任务单数据集通常取得最优；在跨任务/跨数据集情境下提升最多 13.41%，在大多数场景达到 state‑of‑the‑art，并且压缩速度比 SOTA 基线快 2–3 倍。

**⚠️ 局限性**

局限性：在具有复杂共混关系和潜在变量的图（如 Flickr）中效果相对弱；对因果干预的低频调节仍有改进空间，且在极大规模图上对计算资源仍有一定需求。

---

## 27. DUET: Distilled LLM Unlearning from an Efficiently Contextualized Teacher

**arXiv ID:** 2601.21283 | [PDF](https://arxiv.org/pdf/2601.21283v1)

**作者:** Yisheng Zhong `[一作]` (George Mason University), Zhuangdi Zhu `[通讯]` (George Mason University)

**通讯引用:** 1189 | [OpenAlex ID](https://openalex.org/A5079428801)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于教师模型蒸馏的LLM遗忘方法（DUET），通过将上下文提示导致的拒绝行为转化为学生模型参数中的Top‑K对数概率对齐，实现精准的知识遗忘。

**💡 创新点**

创新点：①将上下文提示的短暂拒绝行为固化为参数更新，兼具无须修改参数的灵活性与基于梯度的持久性；②使用Top‑K对数概率蒸馏而非完整概率分布，显著减少噪声与计算量；③仅需查询级别的数据，无需配对的拒绝或负样本，提升数据效率；④在评估中引入更广泛的任务格式（QA、内容补全）和逆向攻击检测，验证鲁棒性。

**🔧 技术方法**

技术手段：教师-学生蒸馏、Top‑K对数概率对齐（Huber 损失）、保留集正则化（KL 对齐或保留数据混合）、对抗逆向攻击实验、扩展评估协议、实验脚本与超参数配置公开。

**📊 数据集**

使用的数据集：MUSE‑Books（Harry Potter）扩展至500问答；WMDP（Cyber 与 Bio 子任务）200问答；MMLU 5‑shot多选；保留集 100 QA 对；对比数据集包含原始书籍文本与经过重排的查询+答案。

**📈 对比分析**

对比方法：GA、NPO、SimNPO、FLAT、Refusal‑Training、RMU；评价指标包括 R‑Forget↓、R‑Forget‑500↓、R‑Retain↑、MMLU↑、整体性能变动 Δ↑。实验显示 DUET 在遗忘率上优于或相当于所有基线，且保持或提升保留性能，整体性能提升最高（如 MUSE‑Books 55.9% 的 Δ↑），且在逆向攻击与不同评估格式下表现最稳健。

**⚠️ 局限性**

局限性：①仍需手工设计有效的教师提示，提示质量直接影响遗忘效果；②对边界判定（哪些知识应被遗忘）仍无法完全自动化，需进一步研究；③评估协议依赖人工构造的问答/补全样本，可能无法覆盖所有潜在泄漏形式；④对极大规模模型或多模态场景的适用性尚未验证。

---

## 28. PILD: Physics-Informed Learning via Diffusion

**arXiv ID:** 2601.21284 | [PDF](https://arxiv.org/pdf/2601.21284v1)

**作者:** Tianyi Zeng `[一作]` (Shanghai Jiao Tong University), Xinbo Chen `[通讯]` (Tongji University)

**通讯引用:** 3090 | [OpenAlex ID](https://openalex.org/A5004020692)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 Physics‑Informed Learning via Diffusion (PILD) 框架，利用扩散模型生成过程中加入物理约束，实现数据与物理的一体化学习。

**💡 创新点**

创新点在于：①引入基于拉普拉斯分布的虚拟残差观测，将物理约束转化为概率目标；②设计 U‑FiLM/U‑Att 条件嵌入模块，使物理信息在整个扩散过程中持续引导；③通过联合数据与物理似然构建统一训练目标。

**🔧 技术方法**

采用了扩散模型（DDIM）与物理信息神经网络（PINN）思想相结合的技术；使用 FiLM、跨注意力（cross‑attention）进行条件嵌入；引入 Laplace 似然实现物理残差约束。

**📊 数据集**

在四个工程与科学基准数据集上验证：车辆轨迹、轮胎力、达西流动（Darcy flow）以及等离子体动力学（plasma dynamics）等。

**📈 对比分析**

与 EKF、FCN、ResNet、PINN、LSTM、DDPM、PIDM 等传统与最新方法对比，PILD 在所有任务中均显著降低误差（如轨迹误差、轮胎力误差、Darcy 残差和等离子体密度/温度误差），表现优于现有基线。

**⚠️ 局限性**

主要限制是仍受 Jensen gap 影响，使用 DDIM 只能减弱而非完全消除；物理权重与残差分布的调参仍需经验；模型在极高维或复杂物理情境下的可扩展性待进一步研究。

---

## 29. ICON: Intent-Context Coupling for Efficient Multi-Turn Jailbreak Attack

**arXiv ID:** 2601.20903 | [PDF](https://arxiv.org/pdf/2601.20903v1)

**作者:** Xingwei Lin `[一作]` (Zhejiang University), Chunming Wu `[通讯]` (Zhejiang University)

**通讯引用:** 2914 | [OpenAlex ID](https://openalex.org/A5078198240)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于意图与上下文耦合的多轮 jailbreak 框架 ICON，自动生成具有权威风格的对话上下文来诱导 LLM 生成违禁内容。

**💡 创新点**

创新点在于发现并利用“意图-上下文耦合”现象，将恶意意图映射到语义相符的上下文模式，并采用分层优化策略在策略层切换上下文、在战术层细调提示。

**🔧 技术方法**

技术主要包括基于 LLM 的意图分析器、上下文路由器、上下文生成器、攻击合成器以及战术/战略反射器，实现闭环自动化生成。

**📊 数据集**

数据集使用合成的 200 条多类攻击问句，来自 JailbreakBench、HarmBench 与 JailbreakRadar 的去重与重标注。

**📈 对比分析**

与八种 SOTA 方案在 8 个主流 LLM 上对比，ICON 平均攻击成功率 97.1% 及 StR 84.9%，显著高于所有基线。

**⚠️ 局限性**

局限性包括仍依赖 LLM 生成组件、对极端安全机制的鲁棒性未知，以及需进一步验证跨模态的可迁移性。

---

## 30. A Sheaf-Theoretic and Topological Perspective on Complex Network Modeling and Attention Mechanisms in Graph Neural Models

**arXiv ID:** 2601.21207 | [PDF](https://arxiv.org/pdf/2601.21207v1)

**作者:** Chuan-Shen Hu `[一作]` `[通讯]` (National Central University), Chuan-Shen Hu (National Central University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了基于细胞层sheaf理论的框架，用来解释并分析图神经网络（尤其是GAT）中的注意力机制、节点特征的局部一致性和全局谐波性，并进一步引入基于TDA的多尺度滤波来捕捉特征在不同尺度下的相干性。

**💡 创新点**

创新点在于：①将GAT的注意力权重视为细胞sheaf的限制映射，从而把注意力机制本身转化为一套可供数学分析的sheaf结构；②定义“谐波边/点集”，用来量化节点信号在边上的一致性；③提出基于sheaf范数的多尺度谐波滤波，构造持久性条形码，提供一种全新的拓扑视角来评估特征扩散与聚合。

**🔧 技术方法**

使用的技术包括：细胞sheaf理论（sheaf Laplacian、全局截面空间）、图信号处理、拓扑数据分析（TDA）中的持久性同调与滤波、矩阵分解与谱分析。

**📊 数据集**

本文主要是理论推导和概念构建，没有使用具体数据集；若要验证，需要在标准图数据集（如Cora、Citeseer、OGBN-Products等）上实现GAT+sheaf模型进行实验。

**📈 对比分析**

由于缺乏实验结果，文中未给出方法对比或性能评估；后续工作应将该框架与传统GAT、GCN、SheafAN等模型在节点分类、社区检测等任务上进行量化比较。

**⚠️ 局限性**

局限性包括：①仅在无向简单图上给出定义，未扩展到高阶单纯形或细胞复形；②理论框架尚未与训练过程结合，缺乏可操作的损失或正则化项；③对大规模图的计算复杂度未作分析，实际部署可能面临效率瓶颈。

---

## 31. Text-only adaptation in LLM-based ASR through text denoising

**arXiv ID:** 2601.20900 | [PDF](https://arxiv.org/pdf/2601.20900v1)

**作者:** Sergio Burdisso `[一作]` (Idiap Research Institute), Andreas Stolcke `[通讯]` (Idiap Research Institute)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种仅利用文本数据对基于大型语言模型（LLM）的自动语音识别（ASR）系统进行领域适配的方法，方法将适配任务重新表述为文本去噪；

**💡 创新点**

创新点在于通过模拟投影器产生的“噪声”文本来训练LLM，使其保持音频-文本对齐并学习目标领域的词汇和句法；

**🔧 技术方法**

技术主要包括：投影器噪声模拟、随机字符替换与重复生成噪声、混合批次构造（含源域音频、投影噪声、文本噪声）以及LoRA微调；

**📊 数据集**

使用了两个对话语音语料库：DefinedAI（银行、保险、保健等）和SlideSpeech（生活、才艺、英语等），并划分为源域与目标域；

**📈 对比分析**

与仅使用文本的两种基线（Ma et al., Fang et al.）和最优音频适配进行对比，文本适配在大多数目标域上实现了相当于或接近音频适配的WER下降，单域内最高提升达22.1%相对改进；

**⚠️ 局限性**

局限性包括：噪声函数仅为近似，难以完全匹配真实投影器输出；适配效果受目标域文本比例τ影响；跨域音频差异仍导致性能未能与音频适配持平。

---

## 32. Qwen3-ASR Technical Report

**arXiv ID:** 2601.21337 | [PDF](https://arxiv.org/pdf/2601.21337v1)

**作者:** Xian Shi `[一作]`, Junyang Lin `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

推出了 Qwen3-ASR 1.7B/0.6B 两款全功能语音识别模型，并推出了 Qwen3-ForcedAligner 0.6B 语音强迫对齐模型。

**💡 创新点**

创新点包括：①基于大型音频‑语言模型 Qwen3‑Omni 进行四阶段训练（AuT 预训练、Omni 预训练、ASR 监督微调、RL 强化学习），实现多语言/方言、嘈杂环境、歌声识别的统一模型；②提出 LLM‑驱动的非自回归时间槽填充对齐架构，实现 11 语言、跨语种、长音频（≤5 min）高精度对齐；③提供端到端推理、流式推理与强迫对齐一体化工具箱。

**🔧 技术方法**

采用的技术包括：AuT 关注‑编码器、动态 FlashAttention 窗口、非自回归 NAR 推理、Group Sequence Policy Optimization（GSPO）RL 微调、时间槽填充训练、伪标签蒸馏、LLM 语言建模、跨模态多任务预训练。

**📊 数据集**

训练和评估数据涵盖约 40 M 小时伪标注音频（中英主导）、多语言真实音频、52 种语言与 22 种汉语方言、内部鲁棒性套件（口音、老年儿童、极低 SNR、舌尖词、对话）、公开基准（LibriSpeech、CommonVoice、Fleurs、MLS、MIR‑1k、Singing‑En/Zh）以及 11 种对齐语言的长短音频。

**📈 对比分析**

对比方法：与三大商业 API（GPT‑4o‑Transcribe、Gemini‑2.5‑Pro、Doubao‑ASR）及开源模型（Whisper‑large‑v3、FunASR‑MLT‑Nano、GLM‑ASR‑Nano）在公开基准、内部鲁棒性、多语种、歌声识别、流式推理和强迫对齐上进行评测。结果显示：Qwen3-ASR‑1.7B 在多数公开基准与内部测试中处于领先或与最强商业 API 接近；Qwen3-ASR‑0.6B 在轻量化部署时实现 92 ms TTFT、RTF 0.064、并发 128 时 2000 s/秒吞吐；Qwen3-ForcedAligner‑0.6B 的 AAS（累计平均偏移）比 Monotonic‑Aligner、NFA、WhisperX 低 70‑80%，且支持跨语言/长音频对齐。

**⚠️ 局限性**

局限性：①多语言多方言覆盖仍不够完整，长尾语言性能略逊；②对齐模型依赖 MFA 伪标签，仍有少量噪声偏移；③在极高噪声或复杂音乐背景下的长歌识别仍有挑战；④模型规模与训练成本高，部署资源需求仍大。

---

## 33. Distributional Active Inference

**arXiv ID:** 2601.20985 | [PDF](https://arxiv.org/pdf/2601.20985v1)

**作者:** Abdullah Akgül `[一作]` (University of Southern Denmark), Melih Kandemir `[通讯]` (University of Southern Denmark)

**通讯引用:** 1436 | [OpenAlex ID](https://openalex.org/A5032539965)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

将主动推理（AIF）框架与分布式强化学习相结合，提出 Distributional Active Inference (DAIF) 算法，实现无模型、基于分布的决策优化。

**💡 创新点**

通过理论框架将 AIF 的 ELBO 与分布式 RL 的回报分布推送映射结合，消除显式动力学建模，利用编码‑解码压缩轨迹，提升样本效率与性能。

**🔧 技术方法**

使用变分贝叶斯、因果推断、推送变换、Wasserstein 距离、分位数回归、状态‑动作可扩散编码器等技术。

**📊 数据集**

在经典连续控制任务（DMControl、DMControl Vision、EvoGym）以及 RiverSwim 风格的网格世界进行实验。

**📈 对比分析**

与 D4PG、DrQ‑v2、DSAC、TD3 等基线在 AULC 与 Final Return 上进行对比，DAIF 在困难环境中显著提升性能，计算时间略高约 12%。

**⚠️ 局限性**

缺乏有限样本理论，对高维状态的压缩依赖编码器设计，且在无优势抽象的环境下仅与传统分布式 RL 持平，实施成本相对较高。

---

## 34. Eye Feel You: A DenseNet-driven User State Prediction Approach

**arXiv ID:** 2601.21045 | [PDF](https://arxiv.org/pdf/2601.21045v1)

**作者:** Kamrul Hasan `[一作]` (Texas State University), Oleg V. Komogortsev `[通讯]` (Texas State University)

**通讯引用:** 3597 | [OpenAlex ID](https://openalex.org/A5035152487)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了使用 DenseNet 回归模型预测眼动数据中的主观感受（如疲劳、任务难度等），并在跨轮次（同一受试者随时间）和跨个体（不同受试者）两种实验设置中进行评估。

**💡 创新点**

创新点在于将多目标回归任务与预激活 DenseNet 结合，直接从眼动速度信号学习特征，减少手工特征设计；同时通过跨轮次和跨个体实验揭示眼动与主观感受随时间和个体差异的关系。

**🔧 技术方法**

技术包括眼动位置到速度的预处理（Savitzky–Golay 滤波、归一化），预激活 DenseNet 骨干网络 + 线性回归头，Smooth L1 损失、Adam 优化、Dropout、权重衰减等正则化方法。

**📊 数据集**

使用 GazeBase 数据集，包含 322 名参与者、12,334 条眼动记录、9 轮次、7 种任务以及对应的主观评分（疲劳、舒适度等）。

**📈 对比分析**

与全局均值基线比较，使用 MAE、RMSE、Pearson r、R² 以及整数精度等指标；在已知个体跨轮次实验中模型显著降低 MAE（如从 0.78→0.64）并提高精度（从 22%→60%），但在未知个体跨个体实验中提升有限，MAE 与精度仅略有改进。

**⚠️ 局限性**

局限性在于跨个体泛化能力弱，受个体差异和尺度使用差异影响；主观评分噪声大、离散度低；模型对任务/情境变化的鲁棒性不足，需要个性化或域适应等进一步策略。

---

## 35. Solver-in-the-Loop: MDP-Based Benchmarks for Self-Correction and Behavioral Rationality in Operations Research

**arXiv ID:** 2601.21008 | [PDF](https://arxiv.org/pdf/2601.21008v1)

**作者:** Ruicheng Ao `[一作]` (Massachusetts Institute of Technology), Xinshang Wang `[通讯]` (Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了两种基于求解器反馈的评估基准（ORDebug和NewsVendorBias），并通过RLVR与分阶段课程学习训练小型模型实现迭代自我纠错与行为理性评估。

**💡 创新点**

创新点在于：①将求解器的不可行子系统信息作为可验证的反馈循环，引入过程级评估；②将RLVR与步骤级奖励模型相结合，显著提升诊断准确性；③设计分阶段课程训练有效消除LLM在库存决策中的“拉向中间”偏差，获得唯一负的ID→OOD漂移。

**🔧 技术方法**

采用了大型语言模型（Qwen3-8B）经过SFT、PPO+KL移除、LoRA、步骤级奖励模型（PRM）以及RLVR的过程监督技术，辅以Gurobi求解器提供的IIS、slack与目标值反馈。

**📊 数据集**

使用自制的“Saboteur”生成的5,000+条线性规划错误实例（9种错误类型）和2,000条新闻推销库存实例（1,000 ID+1,000 OOD），并结合公开的ORBench、AIM-Bench等对照数据。

**📈 对比分析**

与22个API模型及4个本地模型对比，8B训练模型在迭代自我纠错上恢复率提升9.1%（95.3% vs 86.2%）、诊断准确率提升14.6%（62.4% vs 47.8%），平均解决步数从3.78降至2.25，效率提升1.7×；在行为理性上，课程学习实现OOV偏差减幅48%（10.4% vs 20.0%），并出现唯一负的ID→OOD漂移（-9.6%）。

**⚠️ 局限性**

局限性包括：①仅覆盖线性规划与单期库存问题，未扩展到MIP/MINLP、多周期或更复杂的运筹任务；②对求解器的依赖限制了跨平台可移植性；③课程训练与RLVR的设计仍需针对更大规模模型与更丰富错误类型进一步验证。

---

## 36. The Role of Social Identity in Shaping Biases Against Minorities in Software Organizations

**arXiv ID:** 2601.21259 | [PDF](https://arxiv.org/pdf/2601.21259v1)

**作者:** Sayma Sultana `[一作]` (Tulane University), Amiangshu Bosu `[通讯]` (Wayne State University)

**通讯引用:** 1669 | [OpenAlex ID](https://openalex.org/A5078536980)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过情境式问卷调查，量化了软件工程师在工作场所经历的四种基于社会身份的偏见（职业发展、任务分配、环境不友好、身份攻击）及其普遍性、后果和动机。

**💡 创新点**

创新点在于将社会身份理论与交叉性视角结合，系统评估多维身份（性别、种族、年龄、经验、组织规模）对偏见受害率的影响，并首次探讨偏见产生者的动机与背景。

**🔧 技术方法**

方法采用情境（vignette）问卷、定量逻辑回归分析与定性主题编码，并结合多元回归评估身份与偏见之间的关联。

**📊 数据集**

数据集为 253 名跨国软件工程师的在线问卷响应，涵盖不同组织规模、地区及多元身份特征。

**📈 对比分析**

通过逻辑回归模型发现女性与少数族裔受害率显著高于白人男性，偏见导致心理困扰与职业流失；但由于样本规模有限，未与现有偏见评估工具做直接对比。

**⚠️ 局限性**

主要限制包括自选样本偏差、社会期望偏差、情境片段可能导致的认知引导以及偏见产生者自报低频导致动机分析受限。

---

## 37. QUARK: Robust Retrieval under Non-Faithful Queries via Query-Anchored Aggregation

**arXiv ID:** 2601.21049 | [PDF](https://arxiv.org/pdf/2601.21049v1)

**作者:** Rita Qiuran Lyu `[一作]` (University of California), Lei Shi `[通讯]` (Adobe Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种针对非信度查询的检索框架QUARK，利用恢复假设和查询锚定聚合提升检索鲁棒性

**💡 创新点**

创新点在于将查询不确定性建模为恢复假设集合，并通过查询锚定的最大融合方式避免语义漂移，完全不需要重新训练检索器

**🔧 技术方法**

技术方案包括：使用大型语言模型生成多样化的恢复假设；对原始查询和假设分别进行检索；用α参数进行查询锚定的最大加权聚合；对检索得分进行归一化和排序

**📊 数据集**

实验数据集：控制实验中的中文歌词检索模拟（5,000个目标句子、三种噪声级别）；真实世界BEIR基准（FIQA、SciFact、NFCorpus）

**📈 对比分析**

与BM25、Dense-1、Dense-2等基线比较，QUARK在Recall、MRR、nDCG@10等指标上均显著提升，尤其在中等噪声下收益最大；统计显著性检验显示多数增益均为p<0.05

**⚠️ 局限性**

局限性包括：依赖外部LLM生成假设，缺乏自适应的α调优；假设生成不一定覆盖所有意图；在极端噪声或无关查询下提升有限

---

## 38. SIGMA-PPG: Statistical-prior Informed Generative Masking Architecture for PPG Foundation Model

**arXiv ID:** 2601.21031 | [PDF](https://arxiv.org/pdf/2601.21031v1)

**作者:** Zongheng Guo `[一作]` (Politecnico di Milano), Manuela Ferrario `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个基于统计先验引导的生成性遮蔽框架 SIGMA-PPG，用于从 PPG 信号中学习鲁棒的表征，进而实现多种心血管与情绪相关的下游任务。

**💡 创新点**

创新点在于：①将统计先验（幅度与偏度）融入强化学习教师-学生框架，形成 Prior‑Guided Adversarial Masking；②在 VQ‑VAE 语义量化中加入谱重建与语义一致性约束；③利用 Gumbel‑Top‑k 采样实现精准的遮蔽策略，避免噪声驱动的退化。

**🔧 技术方法**

采用的技术包括：Vector Quantized Variational Autoencoder（VQ‑VAE）+谱重建；Transformer 编码器；强化学习（Policy Gradient + REINFORCE）教师‑学生对抗；统计先验偏置；Gumbel‑Top‑k 非参数采样；语义一致性损失。

**📊 数据集**

预训练使用约 12 万小时的临床 PPG 数据（VitalDB、MIMIC‑III），下游评测覆盖 12 个任务，包含心率、呼吸率、SpO₂、血压、情绪、压力、信号质量、活动识别、身份识别等，采样频率从 50 Hz 至 1250 Hz 不等。

**📈 对比分析**

与五个先进基础模型（PAPAGEI‑S、PAPAGEI‑P、Pulse‑PPG、AnyPPG、GPT‑PPG）在全微调和线性探测两种评估协议下进行对比；SIGMA‑PPG 在多数回归任务（如 SpO₂、血压）和多数分类任务（如压力、情绪、信号质量）上取得最高 AUC 或最低 MAE，尤其在 SpO₂ 估计上显著超越对照模型（MAE 0.1457 vs. 1.265）。

**⚠️ 局限性**

局限主要体现在跨域泛化不足：在手腕可穿戴设备产生的强运动噪声场景下，模型在冻结状态下表现下降；对身份识别等对比任务的表征偏向生成性目标，导致线性探测下与对比学习模型竞争力不足；需进一步扩展设备多样性与自适应策略。

---

## 39. Rethinking LLM-Driven Heuristic Design: Generating Efficient and Specialized Solvers via Dynamics-Aware Optimization

**arXiv ID:** 2601.20868 | [PDF](https://arxiv.org/pdf/2601.20868v1)

**作者:** Rongzheng Wang `[一作]` (University of Electronic Science and Technology of China), Ke Qin `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 1952 | [OpenAlex ID](https://openalex.org/A5101188886)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了 DASH 框架，利用 LLM 迭代生成并优化求解器的搜索机制与运行时调度，结合轨迹感知指标 tLDR 与组别检索 PLR，提升组合优化问题的求解效率与质量。

**💡 创新点**

创新点在于：①提出轨迹感知 Lyapunov 衰减率（tLDR）对求解器收敛轨迹进行动态评价；②将搜索机制与运行时调度共进化，分层迭代（MDL/MCL/SSL）；③通过 Profiled Library Retrieval 高效保存并热启动针对不同实例组的专用求解器，显著降低重适配成本。

**🔧 技术方法**

采用 LLM（GPT‑5‑mini）生成与改进求解器代码，进化式搜索机制与调度策略；使用 tLDR 作为轨迹评估指标；两阶段调度压缩与增强（SSL）实现运行时优化；Profiled Library Retrieval（PLR）实现跨组档案管理与热启动。

**📊 数据集**

实验覆盖四类组合优化问题：Euclidean TSP（20–1000 节点）、TSPLIB、CVRPLIB、OR‑Library（MKP）及在线 BPP；训练时随机生成实例并按轻量化实例 profile 划分多个组；测试使用公开 benchmark 与人工手工启发式、专用求解器及其它 LHD 框架。

**📈 对比分析**

通过与专用求解器（Concorde、LKH3 等）、传统启发式（GLS、LS 等）以及其他 LHD 框架（FunSearch、ReEvo、EoH、MEoH、Hercules）在 gap% 与运行时两项指标上对比；实验表明 DASH 在所有四个问题上实现超过 3 倍的运行时效率提升，同时在解质量上优于或接近最优基准。

**⚠️ 局限性**

局限性包括：①仍需要大量 LLM 生成与评估的 token 消耗，尤其在大规模实例或多轮迭代中显著；②对极大实例的可扩展性与跨域迁移（如从合成到真实工业数据）尚未充分验证；③依赖手工设计的实例 profile 特征与归档策略，可能影响在新领域的迁移效果。

---

## 40. Distributionally Robust Classification for Multi-source Unsupervised Domain Adaptation

**arXiv ID:** 2601.21315 | [PDF](https://arxiv.org/pdf/2601.21315v1)

**作者:** Seonghwi Kim `[一作]` (Pohang University of Science and Technology), Minwoo Chae `[通讯]` (Pohang University of Science and Technology)

**通讯引用:** 136 | [OpenAlex ID](https://openalex.org/A5036975442)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种联合建模目标域协变量与条件分布不确定性的分布鲁棒学习框架，用于无监督域适应；

**💡 创新点**

创新点在于构造混合源条件的模糊集合并允许对目标输入分布进行有限Wasserstein球扰动，既解决目标数据稀缺又能对源域伪相关性鲁棒；

**🔧 技术方法**

采用分布鲁棒优化（DRO）、Wasserstein距离、混合条件概率、对抗特征扰动与指数梯度更新的三阶段最小最大优化算法；

**📊 数据集**

在手写数字识别（MNIST、SVHN、USPS）和伪相关性基准（Waterbirds、CelebA、Colored MNIST）上验证；

**📈 对比分析**

与DANN、CDAN、MK-MMD、CORAL、MCD、STAR、GroupDRO等基线比较，实验表明在目标样本极少或存在伪相关性时，方法均显著优于基线，提升幅度可达10‑50%；

**⚠️ 局限性**

局限在于需手工设定超参数（ε₁, ε₂）且对特征空间的Wasserstein球敏感，且主要在视觉任务验证，跨模态或大规模时效能未知。

---

## 41. From Linear Input to Hierarchical Structure: Function Words as Statistical Cues for Language Learning

**arXiv ID:** 2601.21191 | [PDF](https://arxiv.org/pdf/2601.21191v1)

**作者:** Xiulin Yang `[一作]` (Georgetown University), Ethan Gotlieb Wilcox `[通讯]` (Georgetown University)

**通讯引用:** 971 | [OpenAlex ID](https://openalex.org/A5011708753)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过跨语言分析和对Transformer模型的对照实验，探讨函数词三大分布特性（高频、结构关联、边界对齐）如何支持从线性输入中学习层次结构。

**💡 创新点**

创新点在于将函数词的三重分布特性系统化为学习机制，并证明频率与结构关联对学习影响更大；同时揭示相同性能可由不同内部机制实现，挑战传统的行为评估观念。

**🔧 技术方法**

采用GPT‑2 Small Transformer、注意力探测与功能词消融技术，对人工操纵的函数词属性进行实验。

**📊 数据集**

使用186种语言的Universal Dependencies树库、英语维基百科文本以及BLiMP基准测试集。

**📈 对比分析**

在不同函数词属性下训练模型并在BLiMP上评估，结果显示保留全部三属性的语言表现最佳，移除频率或结构关联导致显著下降，边界对齐影响最小；相同性能可由不同注意力分布实现。

**⚠️ 局限性**

局限性包括：实验仅在英语文本上进行，未考察形态学丰富语言；仅处理词级函数词，忽略音频与韵律信息；模型与人类学习者差异可能限制结论推广。

---

## 42. Collective Noise Filtering in Complex Networks

**arXiv ID:** 2601.21299 | [PDF](https://arxiv.org/pdf/2601.21299v1)

**作者:** Tingyu Zhao `[一作]` (Northwestern University), István A. Kovács `[通讯]` (Northwestern University)

**通讯引用:** 8968 | [OpenAlex ID](https://openalex.org/A5000202774)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并实现了网络维纳滤波器（NetWF），用于在静态网络中对边权进行集合式噪声过滤和缺失值推断。

**💡 创新点**

创新点在于将维纳滤波框架引入网络，利用全局边相似度矩阵和可定制的噪声协方差，从而处理异质、相关噪声，并兼容有向/无向、有符号/无符号网络。

**🔧 技术方法**

采用的技术包括基于节点连接谱的 Pearson 相似度构建全局边相似度，构造信号协方差 C_u，利用共轭梯度求解滤波器 G^W，并与最优奇异值收缩（OS）进行比较。

**📊 数据集**

使用了酿酒酵母基因交互网络（ExE）和 Enron 邮件网络作为实证数据集。

**📈 对比分析**

通过与 OS 基线、精确召回曲线、MSE、R² 等指标对比，NetWF 在两数据集上均显著降低噪声、提升结构保真度，尤其在缺失值推断与月度采样降噪任务中表现最好。

**⚠️ 局限性**

局限性包括需先知噪声协方差矩阵（或至少均匀估计），大规模网络需使用迭代 CG 版，且对多层、超图等更高阶结构尚未直接支持。

---

## 43. Ira: Efficient Transaction Replay for Distributed Systems

**arXiv ID:** 2601.21286 | [PDF](https://arxiv.org/pdf/2601.21286v1)

**作者:** Adithya Bhat `[一作]` (Visa Research), Mohsen Minaei `[通讯]` (Visa Research)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了Ira框架，通过在主备复制中传递紧凑的访问提示，显著加速备份节点的事务重放。

**💡 创新点**

创新点在于利用主节点已知的未来访问模式生成可压缩提示，实现接近Belady最优缓存，且提示仅为性能优化，兼容现有协议。

**🔧 技术方法**

使用了以太坊EVM指令监测、压缩提示、顺序批量预取、MDBX数据库以及Rust实现的reth客户端。

**📊 数据集**

使用了以太坊主网连续两周共100,800个区块的历史数据进行评估。

**📈 对比分析**

与标准reth基准比较，提示生成仅增加约11%执行时间，而备份重放速度提升约25×（单线程）或23.6×（16线程），整体吞吐提升超过100%。

**⚠️ 局限性**

局限性包括假设工作负载为存储绑定且确定性，未覆盖并发执行、多租户、跨链或低I/O工作负载；提示依赖主节点正确性。

---

## 44. ViTMAlis: Towards Latency-Critical Mobile Video Analytics with Vision Transformers

**arXiv ID:** 2601.21362 | [PDF](https://arxiv.org/pdf/2601.21362v1)

**作者:** Miao Zhang `[一作]` (Simon Fraser University), Jiangchuan Liu `[通讯]` (Simon Fraser University)

**通讯引用:** 20170 | [OpenAlex ID](https://openalex.org/A5039311485)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对边缘辅助的移动视频分析，提出 ViTMAlis 框架，利用可变分辨率的 Vision Transformer 进行动态推理，并在移动设备上结合运动与任务相关性分析实现自适应帧压缩与推理配置。

**💡 创新点**

创新点包括：① 兼容预训练 ViT 的混合分辨率分块与可恢复特征图的推理策略；② 基于区域运动与检测重要性动态划分压缩与推理参数；③ 在端侧完成多任务决策并同步到边缘，实现网络与计算双重延迟最小化。

**🔧 技术方法**

所用技术包括 Vision Transformer (ViTDet‑L)、混合分辨率 tokenization、窗口注意力、光流追踪、轻量 MLP 大小/精度估计器、基于 Pareto 前沿的配置优化、JPEG 编码、移动端 Jetson Orin Nano 与边缘 RTX 5090 GPU。

**📊 数据集**

使用的实验数据集：5 条真实移动视频（walkS、walkR、walkB、cycleS、driveN）共 45,000 帧；网络条件基于公开 4G/5G 路径的 60 条时变 trace；模型为 ViTDet‑L（预训练 + fine‑tune）。

**📈 对比分析**

与 Back2Back、TrackB2B、TrackRoI、TrackUD 等基线对比，ViTMAlis 在 300 视频‑trace 对上实现了 0.530 的中位渲染准确率，E2E 传输+推理延迟比 Back2Back 低约 51%，且在不同光照与运动场景下均保持最优的准确率与低延迟平衡。

**⚠️ 局限性**

局限性包括：① 推理精度仍受分辨率压缩影响，尤其在复杂目标密集场景下略有下降；② 对运动与重要性估计的依赖导致对遮挡或快速场景切换的鲁棒性受限；③ 目前仅验证了单一 ViT 任务（对象检测），对多任务或不同 ViT 结构的推广仍待进一步研究。

---

## 45. Spotlighting Task-Relevant Features: Object-Centric Representations for Better Generalization in Robotic Manipulation

**arXiv ID:** 2601.21416 | [PDF](https://arxiv.org/pdf/2601.21416v1)

**作者:** Alexandre Chapin `[一作]` (Ecole Centrale de Lyon), Liming Chen `[通讯]` (Ecole Centrale de Lyon)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对机器人视觉运动策略学习中的视觉表征进行大规模系统比较，提出并验证基于Slot Attention的对象中心表征（SOCR）在仿真与真实任务中的有效性。

**💡 创新点**

首次将SOCR应用于多任务机器人操作政策，展示其在分布偏移下显著提升泛化能力，并证明预训练SOCR能进一步增强性能。

**🔧 技术方法**

使用Slot Attention实现对象分解，配合ViT/DINOv2骨干网络、Transformer观测模块以及Baku/ACT政策架构，对比全局、稠密与对象中心表征。

**📊 数据集**

COCO用于预训练SOCR；BridgeData V2、Fractal、DROID等机器人视频数据集用于专门预训练；在MetaWorld、LIBERO以及Franka机器人桌面任务上评估。

**📈 对比分析**

通过统一冻结视觉编码器、同构政策网络，使用多任务示范训练，比较7种表征（ResNet、R3M、DINOv2、VC-1、Theia、SAM+DINOv2、DINOSAUR*）在任务成功率和分布偏移下的下降率；结果显示SOCR（尤其是DINOSAUR-Rob*）在仿真与真实场景中均实现最高成功率，且在纹理、光照与干扰等偏移中性能下降最小。

**⚠️ 局限性**

缺乏语义归一化，部分Slot聚焦背景或干扰；未将表征与机器人动力学或场景动力学对齐，限制了实际部署的解释性与可扩展性。

---

## 46. VoxMorph: Scalable Zero-shot Voice Identity Morphing via Disentangled Embeddings

**arXiv ID:** 2601.20883 | [PDF](https://arxiv.org/pdf/2601.20883v1)

**作者:** Bharath Krishnamurthy `[一作]`, Ajita Rattani `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 VoxMorph，一种零样本语音身份混合方法，能仅用5–20秒音频生成高保真、可被多个人验证的语音合成样本。

**💡 创新点**

创新点在于：①将语音分离为说话风格（prosody）与声音身份（timbre）两种嵌入并分别插值；②采用球面线性插值（Slerp）保持嵌入在超球面上的几何结构；③将融合的嵌入分别输入自回归语言模型与条件流匹配网络，最终用HiFTNet生成波形，实现无模型微调的可扩展攻击。

**🔧 技术方法**

核心技术包括：GE2E与CAM++双编码器提取 prosody 与 timbre 嵌入；Slerp 进行独立插值；自回归 LM 生成声学 token；Conditional Flow Matching 生成 mel‑spectrogram；HiFTNet vocoder 还原高质量音频。

**📊 数据集**

主要使用公开的 LibriSpeech 语料库（≈1000小时），从中随机挑选同性别的说话人对进行实验，并提供 10,000 条合成语音作为公开数据集。

**📈 对比分析**

与 ViM、Vevo、MorphFader 等最新基线对比，VoxMorph 在 FAD、KLD、WER 方面均优于对手；在极低误报阈值下（0.01% FAR）实现 67.8% 的完全匹配成功率（FMMPMR），大幅领先 ViM（0%）和 Vevo（9%）等传统方法。

**⚠️ 局限性**

局限性包括：目前仅测试同语种、同性别说话人对；对跨语言、多身份混合的支持有限；实时性尚未达到，可在多说话人或实时场景下进一步验证和优化。

---

## 47. DataCross: A Unified Benchmark and Agent Framework for Cross-Modal Heterogeneous Data Analysis

**arXiv ID:** 2601.21403 | [PDF](https://arxiv.org/pdf/2601.21403v1)

**作者:** Ruyi Qi `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**通讯引用:** 14650 | [OpenAlex ID](https://openalex.org/A5100459860)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了DataCrossBench基准和DataCrossAgent框架，用于统一处理结构化与视觉表格数据的跨模态分析。

**💡 创新点**

创新点在于构建包含多模态（CSV、SQL、JSON、文本、图像）并可验证的跨源任务基准，并设计协同子代理、递归reReAct推理与跨源交叉传播的框架，成功激活“僵尸数据”。

**🔧 技术方法**

采用LLM驱动的多子代理、VLM表格提取、递归Reasoning-Act、跨源优先排序与可执行代码生成与调试，以及混合评估指标。

**📊 数据集**

使用自研的DataCrossBench共200个跨域任务（金融、医疗、IT、零售），每任务包含结构化文件与图像表格，采用人机逆合成方式生成。

**📈 对比分析**

与GPT‑4o、Gemini‑2.5‑flash、Qwen3‑vl及AgentPoirot等基线对比，采用四维评估（事实性、完整性、逻辑、洞察），DataCrossAgent在Overall得分0.5172，事实性比GPT‑4o提升29.7%，在Hard任务上保持较低性能衰减。

**⚠️ 局限性**

局限包括对LLM代码生成与评估的依赖、跨源匹配误差、任务规模相对有限以及对更丰富视觉文本类型的覆盖不足。

---

## 48. Log2Motion: Biomechanical Motion Synthesis from Touch Logs

**arXiv ID:** 2601.21043 | [PDF](https://arxiv.org/pdf/2601.21043v1)

**作者:** Michał Patryk Miazga `[一作]` (ScaDS.AI, Leipzig University), Patrick Ebel `[通讯]` (ScaDS.AI, Leipzig University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于肌肉骨骼前向仿真和强化学习的Log2Motion框架，用于从触摸日志合成可生物力学合理的用户运动轨迹。

**💡 创新点**

创新点在于将真实Android仿真器与MuJoCo物理引擎耦合，实现日志驱动的生物力学运动合成，并通过POMDP+多阶段奖励实现多模态触摸行为（点击、滑动）的高保真生成。

**🔧 技术方法**

使用技术包括MuJoCo物理仿真、MyoSuite肌肉模型、强化学习（Proximal Policy Optimization）以及自定义的Screen Mirror实时映射。

**📊 数据集**

使用的数据集为大型Android‑in‑the‑Wild日志（715k条交互）以及内部的实验日志，进一步对比了Motion Capture收集的真实用户轨迹。

**📈 对比分析**

通过与人类运动捕捉、Fitts定律、DTW距离等方法对比，生成轨迹在速度、精度和肌肉努力上与人类相当，误差率和运动时间与实验数据高度吻合。

**⚠️ 局限性**

局限性包括仅支持单手指（拇指）单手点击/滑动、需要预先假设姿态/肌肉参数、对多手或双手交互、认知/视觉反馈未建模，以及强化学习训练耗时。

---

## 49. Within-Model vs Between-Prompt Variability in Large Language Models for Creative Tasks

**arXiv ID:** 2601.21339 | [PDF](https://arxiv.org/pdf/2601.21339v1)

**作者:** Jennifer Haase `[一作]` (HU Berlin), Sebastian Pokutta `[通讯]` (Zuse Institute Berlin)

**通讯引用:** 1949 | [OpenAlex ID](https://openalex.org/A5043574831)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一项大规模实验中，对12款大型语言模型在10个不同创意提示下共生成12,000条输出（有效11,870条），并使用线性混合效应模型对原创性与流畅度的方差进行分解。

**💡 创新点**

创新点在于首次将提示、模型选择与采样随机性三者的贡献量化为可比的ICC，并揭示提示对原创性贡献约36%但对流畅度几乎无效，同时强调单样本评估会混淆随机性与真正的提示或模型效应。

**🔧 技术方法**

采用的技术包括自动化的 AUT 创意评分系统、主题熵与重复率分析、结构特征提取以及基于 REML 的线性混合效应模型方差分解，并通过 1,000 次分层自助法估计置信区间。

**📊 数据集**

数据集为单一 AUT 项目“塑料瓶的替代用途”，共产生 566,916 条创意，覆盖 12,000 条生成样本，构成实验的基础。

**📈 对比分析**

比较方法为计算提示、模型及其交互的 ICC，结果显示模型占原创性方差约41%（流畅度51%），提示占原创性约36%（流畅度仅4%），内部方差占10–34%；Gemini 3 Pro 在大多数指标上名列前茅，而 Grok 4.1 则表现出极端高低差异，说明单样本评估不可靠。

**⚠️ 局限性**

局限性包括仅针对单一创意任务，模型因素（架构、训练、对齐、默认参数）混合难以单独归因，内部方差混入服务器、API 版本等外部噪声，流畅度计数使用线性模型而非更合适的 GLMM，且未直接探究提示对流畅度的影响。

---

## 50. A Study of Data Selection Strategies for Pre-training Self-Supervised Speech Models

**arXiv ID:** 2601.20896 | [PDF](https://arxiv.org/pdf/2601.20896v1)

**作者:** Ryan Whetten `[一作]` (Laboratoire Informatique d'Avignon), Yannick Estève `[通讯]` (Laboratoire Informatique d'Avignon)

**通讯引用:** 2483 | [OpenAlex ID](https://openalex.org/A5081852717)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

系统评估了无监督的数据选择方法对 SSL 语音预训练的影响，比较了多样性采样（声学、说话人、语言）与基于句长的采样。

**💡 创新点**

发现仅选取最长句子即可在保留约一半数据的情况下获得更低的 WER 并显著缩短训练时间，表明句长是构建高效预训练数据集的关键因素。

**🔧 技术方法**

采用 BEST‑RQ SSL 框架，基于 conformer 模型、随机投影量化器、旋转位置编码，利用 k‑means 对 MFCC、说话人嵌入和 SENSE 嵌入进行多样性采样，动态批次/动态分块加速训练。

**📊 数据集**

使用 Loquacious 英文语音语料库（总计 25,000 小时），在 2,500 小时和 25,000 小时两种规模下进行预训练，随后在 250 小时标注数据上微调 ASR。

**📈 对比分析**

对比了全量、随机、MFCC、说话人、SENSE、长度、说话人+长度七种子集，评估标准为微调后的 WER。长度/说话人+长度子集在大规模数据上分别取得 17.77/17.42 的最低 WER，较全量基线 18.08 降低约 1.5%，训练时间缩短约 24%。

**⚠️ 局限性**

局限性：仅测试 BEST‑RQ/Conformer 体系；未探索其他模型或多语言场景；对句长效应机制缺乏深入分析；实验仅在非流式设置下进行。

---

## 51. Generalizable Prompt Tuning for Audio-Language Models via Semantic Expansion

**arXiv ID:** 2601.20867 | [PDF](https://arxiv.org/pdf/2601.20867v1)

**作者:** Jaehyuk Jang `[一作]` (KAIST), Changick Kim `[通讯]` (KAIST)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Semantically Expanded Prompt Tuning（SEPT）框架，通过大语言模型生成语义邻居并在音频-语言模型的提示调优中加入语义扩展损失，以提升模型在基类到新类以及跨数据集迁移中的泛化性能。

**💡 创新点**

创新点在于：①显式利用语义邻居正则化提示嵌入空间，①通过边际约束的语义扩展损失实现类内紧凑、类间分离；②首次为 ALM 提示调优提出基准评估方案，验证了方法的普适性。

**🔧 技术方法**

技术手段包括：提示调优（CoOp、CoCoOp、KgCoOp、DePT 等），大语言模型（ChatGPT‑4o）生成语义邻居，margin‑based hinge 语义扩展损失，交叉熵分类损失，t‑SNE 可视化，跨数据集评估。

**📊 数据集**

使用多种音频分类数据集：ESC50、ESC50‑Actions、UrbanSound8K、Beijing‑Opera、NS‑Instruments、CREMA‑D、RAVDESS、SESA、TUT2017、GT‑Music‑Genre、VocalSound 等。

**📈 对比分析**

在基类→新类泛化和跨数据集迁移任务中，将 SEPT 与 CoOp、CoCoOp、KgCoOp、DePT 等基线方法对比，结果表明 SEPT 在大多数数据集上显著提升谐波平均值和新类准确率，且不增加推理延迟，训练时间略增。

**⚠️ 局限性**

限制：目前仅对文本提示空间进行正则化，未扩展到音频提示；对极少样本场景仍可能受限；需要依赖 LLM 生成语义邻居，若 LLM 质量不足可能影响效果。

---

## 52. BEAP-Agent: Backtrackable Execution and Adaptive Planning for GUI Agents

**arXiv ID:** 2601.21352 | [PDF](https://arxiv.org/pdf/2601.21352v1)

**作者:** Ziyu Lu `[一作]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy), Wenhao Jiang `[通讯]` (Guangdong University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 BEAP-Agent 框架，通过 DFS 树搜索、回溯与任务跟踪机制完成桌面 GUI 任务自动化。

**💡 创新点**

创新点在于：将 GUI 任务建模为 DFS 状态空间搜索，支持多级长距离回溯；引入 Planner、Executor、Tracker 三个协同模块，动态更新任务计划并在回溯后重新规划。

**🔧 技术方法**

技术手段包括：GPT‑4o 作为 Planner 与 Tracker，UI‑TARS‑1.5‑7B 作为执行器，PyAutoGUI 进行鼠标/键盘交互，DFS 回溯算法与状态栈管理。

**📊 数据集**

使用了 OSWorld 桌面任务基准（369 个真实任务）进行评估。

**📈 对比分析**

与 S2、JEDI、UI‑TARS 等基线在 50 步内对比，BEAP-Agent 的任务成功率为 28.2%，比基线提升约 17.5%（相对提升 12.8% 对 JEDI），回溯触发率 35.8%，成功率 65.5%，平均回溯步数 2.72。

**⚠️ 局限性**

局限性：对屏幕细节与图标的感知仍不够精准，导致部分任务误判；回溯成功率和平均步数仍有提升空间；整体性能受限于单一视觉语言模型的推理与细粒度视觉理解能力。

---

## 53. CovAgent: Overcoming the 30% Curse of Mobile Application Coverage with Agentic AI and Dynamic Instrumentation

**arXiv ID:** 2601.21253 | [PDF](https://arxiv.org/pdf/2601.21253v1)

**作者:** Wei Minn `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**通讯引用:** 29948 | [OpenAlex ID](https://openalex.org/A5081036622)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了基于LLM的Agentic AI框架，自动推断Android应用中不可达activity的激活条件，并生成Frida动态插桩脚本，进而配合现有GUI fuzzer（如APE、Fastbot）显著提升覆盖率。

**💡 创新点**

创新点在于：①将LLM与MCP协议、semantic/episodic memory相结合，形成双阶段（静态分析+动态插桩）AI代理；②实现对复杂激活条件（如外部资源、设备状态等）的自动满足；③保持对任何现有fuzzer的无侵入兼容性，突破30%覆盖“ceiling”。

**🔧 技术方法**

使用技术包括：Anthropic Claude Sonnet 3.7 LLM、MCP工具调用、Smali代码分析、ICC Bot构建CTG、Frida框架进行动态插桩、Chain‑of‑Thought prompting、反馈循环与脚本验证。

**📊 数据集**

实验数据集：Akinotcho et al. 的11个热门应用 + AndroTest 16个应用，共27个应用、842个不可达activity，覆盖了多种激活条件类型。

**📈 对比分析**

性能对比：与APE、Fastbot、LLMDroid‑Fastbot、Scenedroid等基线比较；-APE、-Fastbot分别在活动覆盖率达到49.5%（提升≈180%）和34.6%（提升≈116%）；-Fastbot平均提升50%覆盖率；与Scenedroid相比，活动启动成功率从15.8%提升至54.8%；激活条件推断Top‑5召回率达0.85。

**⚠️ 局限性**

局限性：①依赖外部fuzzer（如Fastbot）点击注入按钮；②部分复杂激活条件脚本生成仍可能失败；③对被加固或高度混淆的应用效果有限；④LLM的随机性导致较高API成本；⑤当前仅关注覆盖率，未实现缺陷检测。

---

## 54. TeachBench: A Syllabus-Grounded Framework for Evaluating Teaching Ability in Large Language Models

**arXiv ID:** 2601.21375 | [PDF](https://arxiv.org/pdf/2601.21375v1)

**作者:** Zheng Li `[一作]` (Peking University), Zhifang Sui `[通讯]` (Peking University)

**通讯引用:** 4566 | [OpenAlex ID](https://openalex.org/A5110285832)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于教学大纲的LLM教学评估框架，利用多轮对话教学生，并通过学生成绩提升来衡量教学效果。

**💡 创新点**

创新点在于：①将知识点抽象成结构化树，限制教师仅接触知识点与示例题，杜绝信息泄露；②引入学生代理评估教学效果；③系统分析不同域、不同模型以及示例题对教学效果的影响。

**🔧 技术方法**

使用LLM进行知识结构化、题目打标签、题目生成；采用多轮教师-学生对话；利用Pass@k指标评估学生提升；对比不同模型在各学科的教学表现。

**📊 数据集**

数据集为中国高考（Gaokao）七个学科（数学、物理、化学、生物、历史、地理、政治）的1089道题，配合从大纲生成的知识树和示例题。

**📈 对比分析**

通过在无教学干预基线、仅提供知识点、以及提供示例题三种设置下，计算学生Pass@1、Pass@4、Pass@16、Pass@64的提升；结果显示模型在数学、历史、政治等学科的提升明显，Physics/Chemistry表现最弱；加入示例题反而往往降低教学效果。

**⚠️ 局限性**

局限性包括：①学生代理是LLM，无法完全模拟真实学习者；②未与人类教师作对照；③示例题教学效果仅在当前对话设计下测试，其他教学策略可能产生不同结果。

---

## 55. The Depth Delusion: Why Transformers Should Be Wider, Not Deeper

**arXiv ID:** 2601.20994 | [PDF](https://arxiv.org/pdf/2601.20994v1)

**作者:** Md Muhtasim Munif Fahim `[一作]` (Data Science Research Lab, Department of Statistics, University of Rajshahi), Md Rezaul Karim `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了面向架构的神经尺度律，揭示深度与宽度对Transformer性能的不同影响；

**💡 创新点**

发现存在临界深度D_crit≈W^{0.44}，超出后添加层会导致性能下降（Depth Delusion），并给出宽度应比深度快2.8倍增长的最优比例；

**🔧 技术方法**

基于梯度流动理论的经验框架，构造了包含梯度衰减项的损失模型，并使用非线性最小二乘拟合；

**📊 数据集**

使用SlimPajama大规模Web文本语料（627B tokens）进行训练，覆盖27M-7B参数范围；

**📈 对比分析**

与传统规模律对比，评估了不同深度-宽度组合的测试损失，发现32层/4096宽度的7B模型优于64层/2816宽度的深层模型，提升0.12 nats，验证了Depth Delusion；

**⚠️ 局限性**

局限性包括：仅验证到7B规模，需进一步验证更大模型；未使用深度稳定技术（如ReZero、NormFormer），可能降低临界深度；仅针对自回归语言建模，其他模态未检验；

---

## 56. Envisioning Audio Augmented Reality in Everyday Life

**arXiv ID:** 2601.21271 | [PDF](https://arxiv.org/pdf/2601.21271v1)

**作者:** Tram Thi Minh Tran `[一作]` (University of Sydney), Callum Parker `[通讯]` (University of Sydney)

**通讯引用:** 798 | [OpenAlex ID](https://openalex.org/A5008079113)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过协同自我民族志与在线调查两种方法，探讨人们在日常生活中如何想象与使用音频增强现实（AAR），并归纳出十种功能角色与使用边界。

**💡 创新点**

创新点在于：①首次以日常声音体验为出发点，系统性地提炼出AAR的“感知协作者”角色与节奏层级；②结合韵律分析与身体化交互理论，构建跨微、米、宏节奏的使用框架；③聚焦体验与情感维度，揭示AAR在情绪与社交中的潜在价值。

**🔧 技术方法**

采用的技术主要是质性研究方法：协同自我民族志（5名研究者共72条日志）、主题分析、在线问卷（74人）以及对结果的频数与比例比较。

**📊 数据集**

数据集包括：AAR日志（5名研究者共72条条目）与在线问卷（74名受访者）收集的自述、角色描述与关注点。

**📈 对比分析**

比较方法：对日志与问卷中的角色进行编码后，统计频数并计算比例；发现“减弱”（reduce）是三种方法中提及最多的角色，其次为“个性化”（personalise）与“增强”（enhance）。由于本研究无实验系统，未给出数值性能指标。

**⚠️ 局限性**

局限性：①样本量小，协同自我民族志仅来自5名研究者，缺乏跨文化与残障视角；②未涉及真实AAR设备的交互，全部为想象与回顾；③问卷受访者多为科技熟悉者，可能偏向对技术的接受；④仅研究音频维度，未考虑视觉或多模态融合。

---

## 57. The Powers of Precision: Structure-Informed Detection in Complex Systems -- From Customer Churn to Seizure Onset

**arXiv ID:** 2601.21170 | [PDF](https://arxiv.org/pdf/2601.21170v1)

**作者:** Augusto Santos `[一作]` (Instituto de Telecomunicações), José M. F. Moura `[通讯]` (Carnegie Mellon University)

**通讯引用:** 27459 | [OpenAlex ID](https://openalex.org/A5045861415)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种基于协方差矩阵幂的结构化特征学习方法，用以早期检测复杂系统中的突发现象（如癫痫发作、客户流失、疫情爆发）。

**💡 创新点**

创新点在于提出一族可调的协方差/精度矩阵幂作为结构化特征，并证明其在部分可观测的 Matérn 随机场下仍保持结构一致性，同时在训练‑测试阶段实现单次样本推理。

**🔧 技术方法**

技术包括协方差矩阵幂变换、Affine‑Invariant Riemannian 距离可视化、基于交叉验证的指数选择、CNN+LSTM 或 CNN 分类器，以及 GMM 阈值化提取结构图。

**📊 数据集**

数据集涵盖 CHB‑MIT 规模化 EEG 病例（23 通道，24 例）与 IBM Telco 客户流失数据（7043 条样本，21 维）。

**📈 对比分析**

与多种基准方法（传统 CNN、RNN、GNN 等）在相同患者/客户样本数下比较，取得更高的敏感度/召回率，并在无合成重采样的条件下保持竞争性或领先的整体性能。

**⚠️ 局限性**

局限性包括对参数 β 的手工搜索/验证依赖、在极度稀疏/不平衡数据上可能受限、以及无法给出精确的因果图恢复，仅能提供结构化特征。

---

## 58. SMART: A Social Movement Analysis & Reasoning Tool with Case Studies on #MeToo and #BlackLivesMatter

**arXiv ID:** 2601.20986 | [PDF](https://arxiv.org/pdf/2601.20986v1)

**作者:** Valerio La Gatta `[一作]` (Northwestern University), V. S. Subrahmanian `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发并评估了 SMART（Social Movement Analysis & Reasoning Tool），实现对 #MeToo 与 BlackLivesMatter 等社会运动在新闻与 Reddit 两个平台的情绪与讨论量进行追踪与回溯分析。

**💡 创新点**

创新点在于构建跨平台一年的大规模数据集、与记者共创需求的系统架构、以及对关键政治事件影响的细粒度、事件级别的多维度量化框架。

**🔧 技术方法**

技术手段包括基于关键词的日常抓取、KeyBERT+Amazon Comprehend 提取关键词、RoBERTa‑go 情绪模型、MongoDB+ChromaDB 存储、Transformer‑based DEEP 预测模型以及 REAR 回溯引擎。

**📊 数据集**

数据集为自 2024 年 9 月至 2025 年 8 月收集的约 2.7M Reddit 贴文与 1M 新闻文章，覆盖 #MeToo 与 BlackLivesMatter 两运动，已计划公开。

**📈 对比分析**

通过与 20+ 记者协作设定评估指标，利用置换检验、Cohen’s d、FDR 校正等统计方法对事件窗口与对照期的讨论量与情绪强度进行比较，结果显示新闻平台在关键事件窗口讨论量显著提升（d≈1.1），Reddit 则出现相反趋势；情绪强度并未随事件显著上升。

**⚠️ 局限性**

局限性包括仅聚焦两运动两平台、事件选择可能不具代表性、仅基于公开文本无法证实因果关系、模型仍受算法偏差与新闻行业可读性限制。

---

## 59. MAD: Modality-Adaptive Decoding for Mitigating Cross-Modal Hallucinations in Multimodal Large Language Models

**arXiv ID:** 2601.21181 | [PDF](https://arxiv.org/pdf/2601.21181v1)

**作者:** Sangyun Chung `[一作]` (Korea Advanced Institute of Science and Technology), Yong Man Ro `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 7483 | [OpenAlex ID](https://openalex.org/A5038798134)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种训练‑free 的模态自适应解码（MAD）方法，用以抑制多模态大型语言模型中的跨模态幻觉，利用模型自身自评模态重要性并动态调节对比解码分支；

**💡 创新点**

创新点在于：1）通过自评查询得到任务特定的模态权重，显式反映每个问题对音、视模态的需求；2）将权重嵌入对比解码的四分支结构，实现对不同模态干扰的自适应抑制；3）完全无训练，兼容现有多模态 LLM；

**🔧 技术方法**

主要技术包括：对比解码（Contrastive Decoding）、模态自评查询、任务相关模态权重计算（softmax 归一化）、四分支对比融合；

**📊 数据集**

使用评估跨模态幻觉的标准数据集 CMM 与 AVHBench；此外在 VideoMME 上采样 100 条视频构造 300 个问题验证权重分布；

**📈 对比分析**

与 VCD‑Extended、AVCD 等基线对比；在 VideoLLaMA2‑AV 上 MAD 提升跨模态幻觉准确率约 7.8%（视频驱动音频）和 2.0%（音频驱动视频）；在 Qwen2.5‑Omni 上提升 8.7% 和 4.7%；在 CMM、AVHBench 均实现显著整体提升；

**⚠️ 局限性**

局限性：1）权重提取需再次调用模型推理，推理时间略长；2）目前仅针对音视两模，缺乏对更丰富多模态（如热成像、深度等）的泛化；3）未使用轻量化的预测器，未来可通过学习快速估计权重以提升效率。

---

## 60. A Federated Generalized Expectation-Maximization Algorithm for Mixture Models with an Unknown Number of Components

**arXiv ID:** 2601.21160 | [PDF](https://arxiv.org/pdf/2601.21160v1)

**作者:** Michael Ibrahim `[一作]` (Georgia Institute of Technology), Weijun Xie `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 1551 | [OpenAlex ID](https://openalex.org/A5048142877)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于联邦GEM（Generalized Expectation–Maximization）的联邦聚类框架，能够在客户端聚类集合异质且可能重叠、且全局聚类数未知的场景下进行模型训练与聚类数推断。

**💡 创新点**

创新点在于：①首次将未知聚类数的混合模型学习迁移至联邦场景；②通过在每个客户端构造不确定性集合并在服务器端利用集合交集来识别跨客户端聚类重叠；③实现了模型个性化（本地权重）与全局聚类参数的协同训练。

**🔧 技术方法**

核心技术包括：联邦GEM算法；局部EM步骤与不确定性集合求解（对等式约束的二维双凸可行化）；服务器端交叉集合合并与参数聚合；对等式约束的近似闭式求解；以及在高维高维数据上实现的低复杂度推理与可微分隐私讨论。

**📊 数据集**

实验使用了合成高斯数据、MNIST、Fashion‑MNIST、EMNIST、CIFAR‑10、UCI 公开数据集（Abalone、Frog A、Frog B、Waveform）等多种图像与非图像数据集。

**📈 对比分析**

与中心化 GMM、DP‑GMM、k‑FED、FFCM‑avg1/2、FedKmeans、AFCL 等方法进行对比；在 ARI、Silhouette 分数等指标上，FedGEM 在大多数数据集上均超过 AFCL 与 DP‑GMM，并且在部分数据集上甚至优于已知聚类数的基线。

**⚠️ 局限性**

局限性包括：在某些数据集（如 CIFAR‑10、Frog 系列）上对聚类数估计略有过高；依赖簇的良好分离假设，实际使用时需调节最终聚类半径；隐私安全性需进一步强化（目前仅在理论层面讨论）。

---

## 61. Memorization Control in Diffusion Models from Denoising-centric Perspective

**arXiv ID:** 2601.21348 | [PDF](https://arxiv.org/pdf/2601.21348v1)

**作者:** Thuy Phuong Vu `[一作]` (Brunel University of London), Phan Xuan Tan `[通讯]` (Shibaura Institute of Technology)

**通讯引用:** 404 | [OpenAlex ID](https://openalex.org/A5073937678)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

研究了扩散模型在训练过程中因信噪比差异导致的梯度贡献不均，从而引发的记忆化偏差，并提出基于置信区间的时步采样方法来直接控制学习重点，实现记忆化与泛化的平衡。

**💡 创新点**

创新点在于从去噪视角揭示梯度不均衡问题，并提出可调节置信区间（c_l、c_h）和尾部截断重分配的采样策略，首次将时步学习重点作为可控参数直接影响模型记忆化行为。

**🔧 技术方法**

采用高斯时步采样、置信区间参数化、尾部截断重分配技术，并结合PCA、Wasserstein距离和JS距离对生成样本与训练集分布进行评估。

**📊 数据集**

使用图像数据集 Pokémon、Flowers‑102 以及 1D 时序数据集 ECG5000 进行实验。

**📈 对比分析**

通过将自定义采样策略与默认均匀采样对比，评估生成样本与训练集的 L2 距离、PCA 降维后 Wasserstein 与 JS 距离，实验表明将学习重点移向后期时步可显著降低记忆化并提升分布对齐度。

**⚠️ 局限性**

局限性包括对时步分布假设为高斯、实验仅覆盖离散时间范围内的模型与数据集，对不同模型规模、训练长度及更复杂任务的泛化能力仍需进一步验证。

---

## 62. User-Centric Evidence Ranking for Attribution and Fact Verification

**arXiv ID:** 2601.21387 | [PDF](https://arxiv.org/pdf/2601.21387v1)

**作者:** Guy Alt `[一作]` (Bar-Ilan University), Oren Glickman `[通讯]` (Bar-Ilan University)

**通讯引用:** 3116 | [OpenAlex ID](https://openalex.org/A5018914598)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了面向用户的证据排序任务，旨在让最少且足够的证据尽早呈现，减少阅读成本；

**💡 创新点**

创新点包括：将证据选择改为排名问题、提出最小足够排名(MSR)和用户导向的评估指标（MRR、SR、NDCG）、构建统一的多源基准数据集以及提出增量式排序策略；

**🔧 技术方法**

使用的技术包括：基于嵌入的相似度排序、微调的自然语言推理（NLI）模型、列表式微调的推理重排序器以及大语言模型（LLM）的单次与增量式排序；

**📊 数据集**

所用数据集为FEVER、HoVer、WICE三大事实核查数据集的统一抽样版，共计约1000个实例；

**📈 对比分析**

实验显示增量式LLM（如GPT‑4o、Qwen3‑235B）在MRR可达0.75、成功率约63%，显著优于一阶相似度、NLI和单次重排序基线；用户研究进一步表明，增量式排序可将平均阅读句数降至2.5句，准确率提升至94%；

**⚠️ 局限性**

局限性在于缺少内部证据排序的评估、仅覆盖短小主张、LLM成本高以及用户实验样本规模有限。

---

## 63. Zenith: Scaling up Ranking Models for Billion-scale Livestreaming Recommendation

**arXiv ID:** 2601.21285 | [PDF](https://arxiv.org/pdf/2601.21285v1)

**作者:** Ruifeng Zhang `[一作]` (North Carolina State University), Qinglei Wang `[通讯]` (ByteDance)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现Zenith架构，利用Prime Token化处理稀疏特征，并通过Token Fusion和Token Boost模块在TikTok Live直播推荐系统中部署；

**💡 创新点**

①引入Prime Token化，将多维稀疏特征聚合为少量高维Token；②设计Token Fusion（Retokenized Self‑Attention / Tokenwise Multi‑Head Self‑Attention）和Token Boost（Tokenwise SwiGLU / Tokenwise Sparse MoE）实现token‑wise处理，保持token异质性，提升可扩展性；③提出高效的Tokenwise计算与Sparse MoE训练技巧；

**🔧 技术方法**

神经网络特征交互（Cross Network、Self‑Attention、SwiGLU、Mixture‑of‑Experts）、Tokenization、GroupedGEMM、学习率预热、辅助平衡损失、在线A/B测试等技术；

**📊 数据集**

TikTok Live直播推荐的生产数据集，约1.68 B实例、4,552特征、98个目标；

**📈 对比分析**

通过在不同模型规模（small/medium/large）下与基线DCN‑V2、DHEN、Wukong进行离线评估，比较参数量、GFLOP、AUC、LogLoss、UAUC；Zenith/Zenith++在所有规模均优于基线，LogLoss下降0.42–0.63%，AUC提升0.3–0.6%；在线A/B实验中CTR AUC +1.05%，Logloss -1.10%，Quality Watch Session/Duration +9.93% / +8.11%；

**⚠️ 局限性**

仍依赖大量稀疏嵌入表，Token‑wise操作对硬件实现有一定复杂度；在极大规模下Token同质化仍可能出现，需进一步机制；缺乏跨任务/跨域的验证。

---

## 64. SMKC: Sketch Based Kernel Correlation Images for Variable Cardinality Time Series Anomaly Detection

**arXiv ID:** 2601.21050 | [PDF](https://arxiv.org/pdf/2601.21050v1)

**作者:** Haokun Zhou `[一作]` `[通讯]` (Imperial College London), Haokun Zhou (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种针对多变量时间序列的可变通道异常检测框架SMKC，能够在传感器更换、缺失或重命名等情况下自适应不同通道数量并保持异常检测性能；

**💡 创新点**

核心创新在于将动态输入结构与异常检测器解耦：使用签名特征哈希构造固定宽度的状态序列，再通过构造混合核图（包含余弦相似度与稳健对数距离）捕获全局时序结构，并结合结构化的自监督重建与教师-学生预测提升鲁棒性；

**🔧 技术方法**

技术包括：签名特征哈希（Permutation‑Invariant Sketching）、核图构造（Cosine + Log‑Distance）、随机投影kNN、Masked Autoencoding、教师‑学生自监督、位置编码与尺度标记；

**📊 数据集**

在合成的“holdout‑C”与“in‑dist‑C”数据集上进行评估，数据集设计为不同通道数与缺失模式，且训练仅使用正常窗口；

**📈 对比分析**

与多种基线（USAD、TranAD、Anomaly Transformer、GRU‑AE、DeepSets、MSCRED、统计距离方法等）对比，SMKC在AUPRC、AUROC及低误报率下的TPR方面均位列前列；尤其训练‑free的RandProj‑kNN(SMKC)与已训练模型差距不大；

**⚠️ 局限性**

局限性包括：对合成数据的依赖性（实际工业场景中缺失模式和异常类型更复杂）、哈希碰撞虽小但仍存在对极端维度的潜在影响、以及对超参数（哈希宽度、窗口长度、掩码模式等）较为敏感。

---

## 65. FIPS 204-Compatible Threshold ML-DSA via Masked Lagrange Reconstruction

**arXiv ID:** 2601.20917 | [PDF](https://arxiv.org/pdf/2601.20917v1)

**作者:** Leo Kao `[一作]` `[通讯]` (Codebat Technologies Inc.), Leo Kao (Codebat Technologies Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出一种面向FIPS 204 ML-DSA 的可扩展阈值签名方案，并提供三种不同信任模型的部署模式。

**💡 创新点**

设计了掩码拉格朗日重构技术，解决大阈值下拉格朗日系数膨胀、拒绝采样失效和 r0 检查泄露等三大难题。

**🔧 技术方法**

采用对称伪随机函数生成对等掩码、Shamir 共享与拉格朗日插值、MPC/TEE/2PC 等分布式计算以及 Irwin‑Hall 非均匀 nonce 分布等技术。

**📊 数据集**

使用标准参数 ML-DSA‑65（n=256,q=8380417）进行实验，评估 (3,5)、(11,15)、(16,31) 等阈值配置。

**📈 对比分析**

与单签、噪声填充、Ringtail 等现有方案对比，证明在保持 3.3 KB 签名、FIPS 204 兼容的前提下，成功率 23–32%，签名时间 1–4 秒，性能提升 10⁵–10²¹ 倍。

**⚠️ 局限性**

需要签名集至少 T+1 个参与者、依赖同步在线，并存在不同的信任假设（TEE/ MPC/CP）；Irwin‑Hall nonce 可能引入极小的安全偏差。

---

## 66. Multi-task Code LLMs: Data Mix or Model Merge?

**arXiv ID:** 2601.21115 | [PDF](https://arxiv.org/pdf/2601.21115v1)

**作者:** Mingzhi Zhu `[一作]` (Rensselaer Polytechnic Institute), Michele Merler `[通讯]` (IBM Research)

**通讯引用:** 1032 | [OpenAlex ID](https://openalex.org/A5068061267)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在不同规模代码大型语言模型（LLM）上，如何高效构建多任务模型，比较了两种策略：一次性在混合任务数据上进行微调（data‑mix）与分别训练任务专用模型后进行权重合并（model‑merge）

**💡 创新点**

创新点在于：①系统性对比了数据混合与模型合并在两类代码 LLM（Qwen‑Coder、DeepSeek‑Coder）以及两大参数规模（约 2B 与 7B）下的表现；②提出了基于层级权重差异与任务间权重更新相关性的诊断方法，可预测何种策略更适合；③给出实用的规模与任务组合指南，并展示在 7B 模型上模型合并甚至能超越单任务微调的效果

**🔧 技术方法**

技术主要包括：监督微调（SFT）+ MergeKit 中的四种合并算法（Linear、TIES、DARE、DELLA）；层级 L2 距离与 Pearson 相关系数分析；使用 HumanEval、MBPP、CodeXGlue 等标准代码生成/摘要评测；实验在 4×H100 GPU 上完成

**📊 数据集**

数据集：代码生成使用 KodCode（268K 题目）；代码摘要使用 CodeXGlue Code‑to‑Text（417K 代码‑注释对）；评测集为 HumanEval、MBPP 生成测试集以及 CodeXGlue 摘要测试集

**📈 对比分析**

比较方法：对同一模型基线分别做任务专用微调、混合任务微调、以及四种合并方法，评估 Pass@1、BLEU‑4、chrF++、ROUGE‑L、METEOR。结果表明：小规模（≈2B）时，混合微调的平均性能下降 <2%，远优于合并；大规模（≈7B）时，合并（尤其是 DARE/DELLA）平均损失 <2%，甚至在某些指标上超过单任务微调（如 7B DeepSeek‑Coder 的 MBPP Pass@1 提升至 0.799）

**⚠️ 局限性**

局限性：仅评估两类模型和两项任务；混合比例固定为 1:1，未系统探究最佳混合比例；仅在 Python 语言下实验，其他语言效果未知；合并算法参数未进行超参数搜索；对比仅在 4×H100 GPU 上进行，可能受资源限制导致结果不完全可泛化

---

## 67. Designing the Interactive Memory Archive (IMA): A Socio-Technical Framework for AI-Mediated Reminiscence and Cultural Memory Preservation

**arXiv ID:** 2601.21001 | [PDF](https://arxiv.org/pdf/2601.21001v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 68. Multi-Robot Decentralized Collaborative SLAM in Planetary Analogue Environments: Dataset, Challenges, and Lessons Learned

**arXiv ID:** 2601.21063 | [PDF](https://arxiv.org/pdf/2601.21063v1)

**作者:** Pierre-Yves Lajoie `[一作]`, Giovanni Beltrame `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在行星模拟环境下，用三台机器人开展去中心化协同SLAM实验，并收集了实时对等链路吞吐量和延迟数据。

**💡 创新点**

将Swarm‑SLAM框架应用于真实地面模拟场景，首次公开了包含网络性能信息的行星模拟数据集，并探讨了通信受限对SLAM性能的影响与资源效率。

**🔧 技术方法**

采用Swarm‑SLAM、LIO‑SAM LiDAR‑IMU里程计、ScanContext地标识别、TEASER++ 3D配准、IEEE802.11s/batman‑adv/Zenoh构建的MANET网络，以及iperf/fping测量网络性能。

**📊 数据集**

本研究使用的实验数据集来自CSA Mars Yard，包含三台AgileX机器人收集的LiDAR、IMU、GPS、RTK校正数据以及实时对等链路吞吐量和延迟日志。

**📈 对比分析**

通过evo计算ATE比较结果显示，三台机器人平均ATE为3.74±1.63 m；单机里程计误差分别为2.45±1.14、4.29±1.76、3.61±1.72 m，证明方法在行星模拟环境中表现尚可但受地形与通信影响。

**⚠️ 局限性**

主要限制在于通信带宽受限导致前端数据量大、地形振动与视觉特征稀缺导致定位误差、缺乏压缩/语义表示技术，以及多机器人时网络竞争激烈，需要更高效的层级化或优先级调度策略。

---

## 69. A generative machine learning model for designing metal hydrides applied to hydrogen storage

**arXiv ID:** 2601.20892 | [PDF](https://arxiv.org/pdf/2601.20892v1)

**作者:** Xiyuan Liu `[一作]` (Louisiana Tech University), Yuhua Duan `[通讯]` (United States Department of Energy)

**通讯引用:** 7042 | [OpenAlex ID](https://openalex.org/A5026290034)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用因果发现与轻量级生成模型开发了一套新型金属氢化物材料发现框架

**💡 创新点**

首次结合因果发现筛选关键特征，并在仅有270个样本的情况下训练CDVAE生成全新晶体结构

**🔧 技术方法**

采用因果发现（FCI）、变分自编码器（CDVAE）、M3GNet与DFT验证的混合技术栈

**📊 数据集**

使用Materials Project数据库450条样本（270/90/90）进行训练，生成1000个候选物料

**📈 对比分析**

通过M3GNet与DFT对生成结构进行能量和稳定性评估，得到MAE≈0.08 eV，与原始M3GNet的0.0754 eV相近，验证了方法的可靠性

**⚠️ 局限性**

因果图可能出现物理不合理方向，且存储评分公式对高W_H₂低E_form材料的评估不足，需结合领域知识改进

---

## 70. Achieving $\varepsilon^{-2}$ Dependence for Average-Reward Q-Learning with a New Contraction Principle

**arXiv ID:** 2601.21301 | [PDF](https://arxiv.org/pdf/2601.21301v1)

**作者:** Zijun Chen `[一作]` (Hong Kong University of Science and Technology), Shengbo Wang `[通讯]` (University of Southern California)

**通讯引用:** 11838 | [OpenAlex ID](https://openalex.org/A5100411530)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在平均奖励马尔可夫决策过程下，作者提出同步与异步懒惰 Q‑学习算法，并给出了最优的 O(ε⁻²) 采样复杂度。

**💡 创新点**

创新点在于构造了一个实例相关的半范数，使得对马尔可夫过程进行懒惰变换后，贝尔曼算子成为一步收敛，从而不再需要传统的收敛假设。

**🔧 技术方法**

技术方法包括懒惰转换（引入自环）、极值范数理论、实例相关半范数、Lyapunov 函数分析、常数步长的随机逼近以及显式/隐式懒惰采样的 Q‑学习实现。

**📊 数据集**

实验使用了一个 4 状态 2 动作的周期性平均奖励 MDP（参数 p=0.3、q=0.7）以及相应的单轨迹行为策略；未使用真实数据集。

**📈 对比分析**

与传统 Q‑学习进行对比，实验显示懒惰 Q‑学习在样本复杂度上达到理论最优 O(T⁻¹/²) 的收敛速率，且在最后迭代的 span 误差上明显优于经典方法。

**⚠️ 局限性**

局限性包括对冲击时间参数 K 的依赖较大，仅适用于满足可达性假设的单连通 MDP；对弱通信或多链 MDP 的推广尚未实现。

---

## 71. Low performing pixel correction in computed tomography with unrolled network and synthetic data training

**arXiv ID:** 2601.20995 | [PDF](https://arxiv.org/pdf/2601.20995v1)

**作者:** Hongxu Yang `[一作]` (GE HealthCare), Gopal Avinash `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种基于合成数据的双域迭代重构方法（ISTA-Net）用于纠正CT扫描中的低效像素（LPP）产生的环状和条纹伪影。

**💡 创新点**

创新点在于利用自然图像生成无偏合成训练对，结合前向投影的内在关联，在图像域和sinogram域双向迭代纠错，且无需真实临床数据即可训练。

**🔧 技术方法**

采用ISTA‑Net（迭代收缩阈值算法的网络化展开），自定义的随机缺失掩模模拟LPP，将其视为压缩感知问题，并用CNN进行软阈值化。

**📊 数据集**

训练使用约5k张来自ILSVRC2017的自然彩色图像转换为灰度并生成合成sinogram；测试使用MMWHS CT体积以及RibFrac CT数据。

**📈 对比分析**

与FBP、插值、AST、DeepRAR、NAFNet、Riner等SOTA方法对比，在MAE、PSNR、SSIM上均遥遥领先；合成训练版略逊于全监督版，但在RibFrac上表现更好，验证了泛化性。

**⚠️ 局限性**

方法仍受限于已知LPP位置假设，对真实场景中未知缺陷定位尚未解决，并且在极端缺陷密度或非Fan‑beam几何下的鲁棒性待进一步验证。

---

## 72. Towards Mitigating Modality Bias in Vision-Language Models for Temporal Action Localization

**arXiv ID:** 2601.21078 | [PDF](https://arxiv.org/pdf/2601.21078v1)

**作者:** Jiaqi Li `[一作]` (University of Warwick), Yu Guan `[通讯]` (University of Warwick)

**通讯引用:** 2494 | [OpenAlex ID](https://openalex.org/A5084397928)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种面向时序动作定位的视听融合框架ActionVLM，专注于缓解视觉与语言模态偏差；

**💡 创新点**

创新点在于：①引入语言优势（Language Advantage）估计机制，动态为每帧分配语言权重；②采用残差聚合策略，将语言视作视觉的补充修正而非主导；③通过轻量级的交替训练实现无额外分支的优势估计；

**🔧 技术方法**

技术包括：视觉编码器VideoMAEv2、视觉语言模型InternVL3/Qwen2.5、全连接预测语言优势、残差加权聚合、ActionFormer检测头、LoRA微调、AdamW+DeepSpeed训练；

**📊 数据集**

数据集：THUMOS14、ActivityNet‑1.3（主实验），额外在EPIC‑Kitchens 100上验证；

**📈 对比分析**

与多种现有方法比较，ActionVLM在THUMOS14上从74.2%（B-3B）提升至同级别最佳；在ActivityNet‑1.3上也实现39.7% mAP；相比vision‑only基线提升约7–8% mAP，在视觉不确定场景中提升更显著；

**⚠️ 局限性**

局限性：①模型对GPU显存需求较高，虽然小型变体可在24GB GPU上训练；②使用固定滑动窗口采样可能产生标签噪声；③语言描述本身可能带来偏差，尽管通过优势估计减弱影响，但仍是未完全解决的问题。

---

## 73. Noisy but Valid: Robust Statistical Evaluation of LLMs with Imperfect Judges

**arXiv ID:** 2601.20913 | [PDF](https://arxiv.org/pdf/2601.20913v1)

**作者:** Chen Feng `[一作]` (Queen's University Belfast), Miguel R. D. Rodrigues `[通讯]` (University College London)

**通讯引用:** 6539 | [OpenAlex ID](https://openalex.org/A5044634366)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于LLM-as-a-Judge的噪声假设检验框架，用小规模人类校准集估计判别器的TPR/FPR，并利用大规模判别器标签做统计检验，保证有限样本Type‑I误差控制；

**💡 创新点**

核心创新在于将判别器噪声建模为可估计的TPR/FPR，并在检验阈值中加入校准方差校正，从而在保证有效性的同时提升统计功效；同时量化了“Oracle Gap”并给出何时优于直接人类评测的判别准则；

**🔧 技术方法**

使用了统计假设检验、方差校正阈值设计、正态近似、Berry‑Esseen 以及与PPI、Direct HT 的对比实验；

**📊 数据集**

实验涵盖了Jigsaw Toxic Comment、Hate Speech Offensive、SafeRLHF等公开数据集，评估多种LLM与判别器组合；

**📈 对比分析**

与Direct HT、Oracle Noisy HT、PPI等基线比较显示，噪声假设检验在判别器TPR高、FPR低时能显著降低Type‑II误差，优于直接评测；但仍低于Oracle、PPI；

**⚠️ 局限性**

局限在于仅处理二值pass/fail评测、对小样本或稀有事件的正态近似可能失效、需独立校准集防止信息泄漏、未覆盖更细粒度或主观任务等。

---

## 74. Faster Predictive Coding Networks via Better Initialization

**arXiv ID:** 2601.20895 | [PDF](https://arxiv.org/pdf/2601.20895v1)

**作者:** Luca Pinchetti `[一作]` (University of Oxford), Tommaso Salvatori `[通讯]` (VERSES AI Research Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究并提出了改进的神经元初始化方法，以加速预测编码网络（Predictive Coding Networks, PCNs）的训练。

**💡 创新点**

创新点在于引入基于样本标签的平均初始化（stream‑aligned average initialization）和基于连续霍普菲尔德网络的记忆初始化（memory‑based initialization），显著降低所需迭代次数并提升收敛速度和最终性能。

**🔧 技术方法**

技术手段包括：预测编码算法、局部梯度更新、迭代推理（inference）与学习（learning）阶段、连续霍普菲尔德网络、注意力机制（query‑key‑value）以及与传统反向传播（BP）的对比。

**📊 数据集**

主要数据集为 FashionMNIST、3×MNIST、KMNIST、CIFAR‑10；在生成任务上使用 FashionMNIST 进行图像重建。

**📈 对比分析**

方法对比：PC‑avg 与传统 PC‑fw 以及 BP。实验显示 PC‑avg 在大多数任务上取得更高的准确率、更低的测试损失，并在训练速度上（SMMs 计数）比 PC‑fw 快 5 倍以上，甚至在部分任务上接近或超越 BP 的性能；在无监督生成任务中 PC‑mem 产生更高质量的重建图像，且所需迭代次数更少。

**⚠️ 局限性**

局限性包括：平均初始化需要标签信息，无法直接应用于回归或无监督任务；记忆初始化需要额外参数且依赖于霍普菲尔德网络的实现；在更深层网络或更复杂任务（如图神经网络、Transformer）中的可扩展性仍需进一步验证。

---

## 75. Mobility-Embedded POIs: Learning What A Place Is and How It Is Used from Human Movement

**arXiv ID:** 2601.21149 | [PDF](https://arxiv.org/pdf/2601.21149v1)

**作者:** Maria Despoina Siampou `[一作]` (University of Southern California), Cyrus Shahabi `[通讯]` (University of Southern California)

**通讯引用:** 19693 | [OpenAlex ID](https://openalex.org/A5012068017)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个结合文本与大规模人类移动数据的POI表征学习框架ME-POIs，能够同时捕捉地点身份和使用功能。

**💡 创新点**

创新点在于提出以对比学习对齐访问嵌入与全局POI原型的目标，并设计多尺度分布迁移机制解决稀疏POI的长尾问题，显著提升对POI功能的建模。

**🔧 技术方法**

技术包括Transformer序列编码、对比学习（InfoNCE）、多尺度高斯核分布迁移、KL散度正则化、文本与移动嵌入的对齐投影以及轻量级下游任务头。

**📊 数据集**

使用了来自Veraset的两大规模人类移动数据集（洛杉矶一年期和休斯顿20天期），以及SafeGraph、Google Maps提供的开放时长、永久关闭、访问意图、繁忙度和价格等级标签。

**📈 对比分析**

与多种文本（MPNet、E5、GTR‑T5、Nomic、OpenAI‑large、Gemini）和移动（Skip‑Gram、POI2Vec、Geo‑Teaser、TALE、HIER、CTLE、DeepMove、STAN、Graph‑Flashback、GETNext、TrajGPT）基线进行对比，ME-POIs在所有五项地图丰富化任务上均实现显著提升（最高81.9% F1提升、24.7% MAE下降）。

**⚠️ 局限性**

局限性包括对文本对齐的依赖导致需要额外文本资源、对稀疏POI的迁移机制仍受邻近POI分布影响、以及在不同城市或更细粒度地理实体（道路、行政区）上的泛化尚未验证。

---

## 76. Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving

**arXiv ID:** 2601.21351 | [PDF](https://arxiv.org/pdf/2601.21351v1)

**作者:** Chendong Song `[一作]` (Hong Kong University of Science and Technology), Zijie Zhou `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种针对大语言模型解码阶段的注意力-前馈网络（Attention/FFN）解耦（AFD）架构，并构建了一个概率分析框架，用于求解在 rA–1F 拓扑中最优的注意力/FFN 比例 r* 以最大化系统吞吐量。

**💡 创新点**

创新点在于：①基于 Attention 负载随时间非平稳的特性，利用几何分布的记忆无关性推导出 Attention 负载的期望递推公式；②在此基础上得到 Horizon‑Average Token Load 的闭式近似；③进一步将系统划分为三种瓶颈模式，推导出全局最优 r* 的三项极大值闭式解，并提供可操作的计算公式。

**🔧 技术方法**

技术手段包括：线性延迟模型（Memory‑bound Attention、Compute‑bound FFN、带启动成本的通信）、微批次流水线与同步同步调度、连续批处理的离散事件模拟器、以及基于期望值推导的马尔可夫链分析。

**📊 数据集**

实验使用了真实的 DeepSeek‑V3 架构在 Huawei Ascend 910C NPU 上采集的延迟参数，构造了不同的工作负载配置（如 μ_P=100、μ_D=500、B=256 等）进行仿真，未使用公开的标准数据集，而是基于服务场景生成的请求长度分布。

**📈 对比分析**

通过与仿真得到的吞吐量曲线对比，理论预测的最优 r* 与仿真最佳值误差在 10% 以内，理论曲线准确追踪仿真吞吐量峰值；同时在不同批量、预填充长度和解码长度下验证了理论的普适性，显示 r* 随上下文长度提升而增长，峰值吞吐随上下文长度增长而下降。

**⚠️ 局限性**

局限性包括：①仅在仿真层面验证，缺乏真实系统部署实验；②对通信延迟的线性模型假设在大规模多机环境下可能失效；③系统需同步等待最慢 Attention 实例导致的 straggler 现象未在理论中完全建模；④对请求长度分布假设为几何分布，在某些业务场景中可能不完全吻合。

---

## 77. When should I search more: Adaptive Complex Query Optimization with Reinforcement Learning

**arXiv ID:** 2601.21208 | [PDF](https://arxiv.org/pdf/2601.21208v1)

**作者:** Wei Wen `[一作]` (Tencent Youtu Lab), Xing Sun `[通讯]` (Tencent Youtu Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了ACQO框架，通过强化学习实现自适应复杂查询的拆分与聚合，从而提升检索增强生成系统的检索效果。

**💡 创新点**

创新点在于将两阶段课程强化学习与自适应查询重写（AQR）以及基于排名与得分的融合（RSF）结合，提供鲁棒的结果聚合和动态查询拆分策略。

**🔧 技术方法**

采用了强化学习（CRL）、自监督检索反馈、LLM决策、RSF融合方法以及RIRS奖励设计等技术。

**📊 数据集**

实验使用了TopiOCQA、HotpotQA以及MultiHop‑RAG等多种检索基准数据集。

**📈 对比分析**

与Prompt、SFT、RL、迭代优化等传统基线对比，ACQO在R@10、MAP@10等指标上实现SOTA，并将推理速度提升至原来的9.1倍。

**⚠️ 局限性**

仍存在对极端复杂查询的拆分不充分、奖励设计依赖检索结果以及缺乏端到端生成质量评估等局限。

---

## 78. Towards Geometry-Aware and Motion-Guided Video Human Mesh Recovery

**arXiv ID:** 2601.21376 | [PDF](https://arxiv.org/pdf/2601.21376v1)

**作者:** Hongjun Chen `[一作]` (University of Macau), Jianbing Shen `[通讯]` (University of Macau)

**通讯引用:** 16146 | [OpenAlex ID](https://openalex.org/A5023184215)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研究了一种基于视频的3D人体网格恢复框架 HMRMamba，解决传统方法中间姿态锚不可靠和时空动态建模不足的问题。

**💡 创新点**

创新点包括：① Geometry‑Aware Lifting Module 采用双扫描 Mamba（全局+局部）将图像几何信息直接注入 2D→3D 提升，产生可靠 3D 姿态锚；② Motion‑guided Reconstruction Network 通过显式与隐式运动表示，对完整 3D 姿态序列进行时间感知，实现物理可行且连贯的网格恢复。

**🔧 技术方法**

技术手段包括 Structured State Space Models (SSM) 的 Mamba/STA‑Mamba 架构、双扫描（全局+局部）Mamba、变形注意力、运动感知注意力、SMPL 模型、ResNet‑50 图像特征提取、2D pose detector（CPN、ViTPose）以及多损失（MPJPE、MPVPE、Accel 等）。

**📊 数据集**

使用的数据集有 3DPW、MPI‑INF‑3DHP 和 Human3.6M 三大基准。

**📈 对比分析**

与现有 SOTA（PMCE、ARTS、TCMR、MPS‑Net、GLoT、Bi‑CF 等）进行对比，HMRMamba 在三大数据集均取得最佳或同等最佳的 MPJPE、PA‑MPJPE、MPVPE、Accel，模型参数 79.6M，GFlops 7.88，性能优于基线且更高效。

**⚠️ 局限性**

局限性：未公开实现细节，缺乏对极端遮挡或多人体场景的评估，仅在单摄像头单人视频上验证，未来需扩展至多摄像头、多人体及更复杂环境。

---

## 79. Human-Agent versus Human Pull Requests: A Testing-Focused Characterization and Comparison

**arXiv ID:** 2601.21194 | [PDF](https://arxiv.org/pdf/2601.21194v1)

**作者:** Roberto Milanese `[一作]` (Politecnico di Torino), Mattia Fazzini `[通讯]` (University of Minnesota)

**通讯引用:** 3179 | [OpenAlex ID](https://openalex.org/A5051928160)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对6,582个人工-代理合并请求和3,122个纯人工合并请求的测试行为进行大规模实证分析，探讨人机协作在软件测试中的表现。

**💡 创新点**

创新点在于首次系统比较人机协作与纯人工环境下的测试频率、测试范围、测试上下文以及测试质量，并揭示代理倾向于新增测试而非维护测试。

**🔧 技术方法**

使用的技术包括基于语言特定启发式的测试文件识别、统计计数/比例指标、卡方检验、Mann‑Whitney U检验、效应量度量（Odds Ratio、Cliff's Delta）以及AromaDr工具检测测试香味。

**📊 数据集**

使用的数据集是公开的AIDev v3 数据集（含 932,791 个 PR，6,618 个人工-代理 PR，3,122 个人工 PR），并聚焦 Java、JavaScript、Python、TypeScript 四种主流语言。

**📈 对比分析**

通过比较两组 PR 的测试比例、测试文件/行数比例、上下文类别以及测试香味的增减，发现人工-代理 PR 的测试覆盖率几乎翻倍，但两者的测试质量差异微乎其微。

**⚠️ 局限性**

局限性包括测试文件识别的启发式误差、仅覆盖四种语言和部分代理模型、无法完全控制开发者经验等因素，导致结果可能不适用于所有开发场景。

---

## 80. Evaluating Spatialized Auditory Cues for Rapid Attention Capture in XR

**arXiv ID:** 2601.21264 | [PDF](https://arxiv.org/pdf/2601.21264v1)

**作者:** Yoonsang Kim `[一作]` (Stony Brook University), Arie Kaufman `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在XR环境下开展了一项受控实验，评估短时空间音频在无头转动、无视觉线索条件下的即时定位准确性，并验证短期视觉-听觉校准对定位精度的提升。

**💡 创新点**

首次系统量化一秒级空间音频在即时注意引导中的定向误差，揭示前后、上下混淆显著高于左右，并证明短期校准可显著降低这些混淆。

**🔧 技术方法**

采用HRTF渲染的宽带噪声刺激（500–9000 Hz），通过Steam Audio插件在HTC Vive Pro HMD和Sony MDR‑7506耳机上实现空间音频；实验设计为三次会话（预校准、校准、后校准）并收集角度误差、成功率与自评置信度。

**📊 数据集**

实验数据集包含17名参与者、每人90个空间位置、3个会话，总计4590次定位试验；没有使用公开数据集，而是自行生成全球体声源排列与音频信号。

**📈 对比分析**

与随机置换基准（chance）比较，平均3D角误差从69°下降至65°；在90°阈值下成功率约为74%；短期校准后平均误差下降约4°，前后混淆率降低约3%。

**⚠️ 局限性**

局限包括使用通用HRTF导致个体差异影响；仅能提供粗定位，未能消除前后、上下混淆；实验不包含头部运动、视觉背景或多模态补偿；仅评估单一音频引擎，未探讨个性化HRTF或高级声学处理。

---

## 81. Operationalizing Research Software for Supply Chain Security

**arXiv ID:** 2601.20980 | [PDF](https://arxiv.org/pdf/2601.20980v1)

**作者:** Kelechi G. Kalu `[一作]` (Purdue University), James C. Davis `[通讯]` (Purdue University)

**通讯引用:** 2786 | [OpenAlex ID](https://openalex.org/A5004592401)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并验证了面向研究软件供应链的分类法，并用其对RSE数据集进行标注

**💡 创新点**

首次将供应链角色、研究角色、分发路径等维度统一进研究软件的操作化框架

**🔧 技术方法**

采用文献综述、LLM标注、OpenSSF Scorecard评估技术

**📊 数据集**

使用Research Software Encyclopedia（RSE）6966条软件条目和Apache Software Foundation基线仓库

**📈 对比分析**

通过Scorecard得分和缺失率对比，RS项目整体得分低于ASF，按角色分层揭示差异

**⚠️ 局限性**

标注误差、Scorecard仅覆盖仓库可见实践、样本偏向开源且仅包含2020-2025年研究

---

## 82. Belief Propagation with Quantum Messages for Symmetric Q-ary Pure-State Channels

**arXiv ID:** 2601.21330 | [PDF](https://arxiv.org/pdf/2601.21330v1)

**作者:** Avijit Mandal `[一作]` (Duke University), Henry D. Pfister `[通讯]` (Duke University)

**通讯引用:** 4541 | [OpenAlex ID](https://openalex.org/A5034044119)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文将量子消息的信念传播（BPQM）推广到对称的q-元纯态通道（PSC），并通过Gram矩阵的特征值进行有效的跟踪和分析。

**💡 创新点**

创新点在于将BPQM推广到对称q-元PSC，并提出了基于Gram矩阵特征值的闭式递归更新规则，从而实现了高效的解码阈值估计和极化码构造。

**🔧 技术方法**

使用了量子消息的信念传播（BPQM）技术，并结合了特征值递归和密度演化（DE）分析。

**📊 数据集**

使用了对称q-元纯态通道（PSC）的Gram矩阵作为数据集，分析了其特征值。

**📈 对比分析**

通过与经典信道的比较，BPQM在解码复杂度和性能上表现出色，能够有效估计LDPC码的解码阈值，并设计出目标块错误率的极化码。

**⚠️ 局限性**

限制在于目前的分析主要集中在对称q-元PSC上，未来的工作可以扩展到更一般的有限阿贝尔对称性。

---

## 83. Thinker: A vision-language foundation model for embodied intelligence

**arXiv ID:** 2601.21199 | [PDF](https://arxiv.org/pdf/2601.21199v1)

**作者:** Baiyu Pan `[一作]` (UBTECH Robotics), Jichao Jiao `[通讯]` (UBTECH Robotics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并训练了一款名为 Thinker 的 10B 规模视听语言基础模型，专门用于机器人本体感知与规划。

**💡 创新点**

创新点包括针对机器人视角的时空数据集构建、结合关键帧与完整视频的输入增强、以及四大核心能力（任务规划、空间理解、时序理解、对象定位）。

**🔧 技术方法**

使用了 Transformer 骨干、视觉编码器、视觉‑语言对齐 MLP 以及两阶段训练策略（基础感知 + 下游微调）。

**📊 数据集**

采用了自制的 Lvis‑520K/Sharerobot‑affordance/Robopoint 等视觉定位集、Egoplan‑it‑100K ego‑view 推理集、Robovideo‑1.8M 规划集、Industroplan‑200K 工业任务集等四大数据集。

**📈 对比分析**

通过 BLEU‑1~4 和 Top‑1 准确率与七个 SOTA VLM 基线对比，Thinker‑7B 在 Robovqa 上取得 63.5 平均 BLEU，在 Egoplan‑bench2 上 58.2% 准确率，均位居榜首。

**⚠️ 局限性**

局限在于缺乏跨环境泛化的验证、对实时低延迟推理的评估不足，以及对多机器人协同与长周期部署的适应性尚未深入探讨。

---

## 84. Parametric Knowledge is Not All You Need: Toward Honest Large Language Models via Retrieval of Pretraining Data

**arXiv ID:** 2601.21218 | [PDF](https://arxiv.org/pdf/2601.21218v1)

**作者:** Christopher Adrian Kusuma `[一作]` (National University of Singapore), Hwee Tou Ng `[通讯]` (National University of Singapore)

**通讯引用:** 14460 | [OpenAlex ID](https://openalex.org/A5110081955)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于预训练数据的LLM诚实度评估基准并开发RETAIN方法

**💡 创新点**

创新点在于利用预训练数据定义知识边界，构建可靠基准，并通过检索、判定与回答三代理实现更高诚实度

**🔧 技术方法**

采用检索器（密集检索）、答案可答性分类器与回答生成器，使用密集向量检索、文档相关性判断等技术

**📊 数据集**

使用Pythia-12b预训练语料与TriviaQA（TIP-TriviaQA）构建的数据集

**📈 对比分析**

对比SFT、Best-of-N、DPO、R-Tuning等基线，在TIP-TriviaQA上取得58.57 EM‑F1、62.23 PM‑F1，远优于基线，并在HoneSet拒答率87.63%

**⚠️ 局限性**

局限在于仅验证在Pythia-12b及其预训练数据上，缺乏对更大模型或不同预训练语料的通用性评估

---

## 85. Grounding and Enhancing Informativeness and Utility in Dataset Distillation

**arXiv ID:** 2601.21296 | [PDF](https://arxiv.org/pdf/2601.21296v1)

**作者:** Shaobo Wang `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 14158 | [OpenAlex ID](https://openalex.org/A5100689117)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于信息量与效用双重度量的高效数据集蒸馏方法

**💡 创新点**

创新点：引入“信息量”(Informativeness)与“效用”(Utility)理论框架，利用Shapley值实现游戏理论信息抽取，并用梯度范数上界化效用评分，构成信息-效用平衡的蒸馏流程

**🔧 技术方法**

核心技术：Shapley值归因、梯度范数评分、知识蒸馏式蒸馏、软标签生成、噪声注入增强多样性

**📊 数据集**

使用CIFAR‑10/100、Tiny‑ImageNet、ImageNet‑Nette/Woof/100/1K等标准图像数据集

**📈 对比分析**

与现有基准（RDED、SRe2L、MTT、TESLA、IDM 等）比较，实验显示在 ResNet‑18/101 等模型上，IPC 为 1、10、50 时可提升 6.1%–16% 的 Top‑1 准确率，并且在跨架构、持续学习等场景均优于对比方法

**⚠️ 局限性**

局限性：仍需较多计算资源进行Shapley估计；对极大规模数据集的扩展性待进一步验证；依赖预训练教师模型，可能在任务迁移时出现性能下降

---

## 86. SR$^{2}$-Net: A General Plug-and-Play Model for Spectral Refinement in Hyperspectral Image Super-Resolution

**arXiv ID:** 2601.21338 | [PDF](https://arxiv.org/pdf/2601.21338v1)

**作者:** Ji-Xuan He `[一作]`, Yanan Qiao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于跨频段交互增强的谱校正模块（H‑S³A 与 MCR），用于提升高光谱图像超分辨率的光谱一致性和空间细节。

**💡 创新点**

在谱域中引入轻量级卷积组与三维投影+回归的 Manifold‑Consistency Rectification，并将其作为端到端可插拔模块，实现显著提升光谱保真度且几乎不增加计算量。

**🔧 技术方法**

采用谱分组、多感受野卷积、1×1通道混合、GeLU 激活、低维流形投影、残差映射以及对光谱曲线的可视化分析等技术。

**📊 数据集**

在 ARAD‑1K、CAVE 与 ICVL 三个公开高光谱数据集上进行实验，使用 ×4/×8 等多尺度设置。

**📈 对比分析**

与多种基线超分网络（SwinIR、RCAN 等）以及三种后置校正器（Savitzky‑Golay、PCA、IBP）在 mPSNR/mSSIM/mSAM 上对比，取得 0.7–3.5 dB 的 PSNR 提升、0.05–0.13 的 SAM 降低，且参数占用仅 +0.048M，展示出优异的性能与低成本。

**⚠️ 局限性**

主要局限在合成降质场景，未在真实噪声或大规模通道数下充分验证，且对极端降质偏差的鲁棒性仍需进一步提升。

---

## 87. Rethinking Refinement: Correcting Generative Bias without Noise Injection

**arXiv ID:** 2601.21182 | [PDF](https://arxiv.org/pdf/2601.21182v1)

**作者:** Xin Peng `[一作]` (Beijing University of Posts and Telecommunications), Ang Gao `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 5064 | [OpenAlex ID](https://openalex.org/A5002156038)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Bi-stage Flow Refinement (BFR) 框架，对预训练的迭代生成模型进行后置纠偏，分别在数据空间和潜在空间使用流匹配方法进行修正。

**💡 创新点**

创新点：仅在训练时使用轻量数据增强实现数据空间纠偏，无需噪声注入；在可逆模型上通过潜在空间流匹配对先验进行对齐；保持原始 ODE 轨迹，仅做确定性修正。

**🔧 技术方法**

技术：流匹配（Flow Matching）、ODE 采样、轻量数据增强、潜在空间噪声混合、可逆生成模型。

**📊 数据集**

数据集：MNIST、CIFAR-10、FFHQ 256×256 图像；分子生成任务 ALA2、Chignolin Mutant。

**📈 对比分析**

对比方法：与基准模型（DDPM、FM）及现有修正器（DiffuseVAE、FMRefiner）比较；在 MNIST 上 1 步潜在空间修正 FID 从 3.95 提升至 1.46，显著优于之前的 4.5；在 CIFAR-10、FFHQ 上亦显著降低 FID/sFID，同时保持 IS 甚至略升。

**⚠️ 局限性**

局限性：潜在空间修正需要可逆模型，且受可逆性好坏限制；过多 NFE 可能导致过度修正；对不同生成模型的适配性仍需进一步研究。

---

## 88. DeepSearchQA: Bridging the Comprehensiveness Gap for Deep Research Agents

**arXiv ID:** 2601.20975 | [PDF](https://arxiv.org/pdf/2601.20975v1)

**作者:** Nikita Gupta `[一作]` (Google DeepMind), Dipanjan Das `[通讯]` (Google DeepMind)

**通讯引用:** 11552 | [OpenAlex ID](https://openalex.org/A5021055773)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了 DeepSearchQA 基准，用于评估大语言模型驱动的自主代理在生成完整答案集合时的检索和推理能力。

**💡 创新点**

创新点在于从传统单答案验证转向集合式答案生成，强调系统化搜集、实体消歧、以及停机判定，填补了“全面性缺口”并引入严格的集合匹配评估。

**🔧 技术方法**

采用了多步骤搜索、上下文合成、逻辑推理的代理架构，并通过 Gemini 2.5 Flash、GPT‑5 Pro 等 LLM 与浏览工具组合进行实验，同时使用 LLM-as-a-Judge 自动评估。

**📊 数据集**

数据集包含 900 条专家标注的多域问题，覆盖政治、金融、科学、健康等领域，答案集合来自静态公开数据源（如 CDC、World Bank、Macrotrends 等）。

**📈 对比分析**

与多种现有检索与推理基准对比，Gemini Deep Research Agent 在 Fully Correct 上达 66% 以上、F1≈82%，显著优于 GPT‑5 Pro、Claude 等模型，表明自代理架构在全检索任务上具有更高效能。

**⚠️ 局限性**

局限性包括仅基于结果评估无法观察搜索轨迹、静态网页假设导致可变事实难以跟踪、以及缺少动态时间敏感任务和加权相关度评估。

---

## 89. Intelli-Planner: Towards Customized Urban Planning via Large Language Model Empowered Reinforcement Learning

**arXiv ID:** 2601.21212 | [PDF](https://arxiv.org/pdf/2601.21212v1)

**作者:** Xixian Yong `[一作]` (Renmin University of China), Xiao Zhou `[通讯]` (Renmin University of China)

**通讯引用:** 25038 | [OpenAlex ID](https://openalex.org/A5002827290)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Intelli‑Planner框架，实现基于LLM和DRL的定制化城市功能区规划。

**💡 创新点**

创新点在于将大型语言模型的知识增强与深度强化学习相结合，并设计五维评估体系与LLM角色扮演的利益相关者满意度评分。

**🔧 技术方法**

技术主要包括OpenAI GPT‑3.5 Turbo、PPO强化学习、Actor‑Critic网络、知识增强模块以及五维评价指标。

**📊 数据集**

使用三城（北京、芝加哥、马德里）社区的GIS、POI、人口统计等公开数据作为训练与评估数据集。

**📈 对比分析**

与人工初始方案、随机、GA、SA、PUP‑MA、DRL‑MLP等基线对比，Intelli‑Planner在目标分数与满意度均居首，提升幅度约5–11%，且收敛更快。

**⚠️ 局限性**

局限在于依赖LLM API调用成本、可能出现幻觉，且仅验证在三城小规模社区，缺乏更大规模多元场景的验证。

---

## 90. Uncovering Hidden Correctness in LLM Causal Reasoning via Symbolic Verification

**arXiv ID:** 2601.21210 | [PDF](https://arxiv.org/pdf/2601.21210v1)

**作者:** Paul He `[一作]` (University of Toronto), Zhijing Jin `[通讯]` (University of Toronto)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种符号验证器，通过对因果图和 do‑calculus 规则进行符号推理，判断 LLM 生成的因果表达式是否在语义上等价。

**💡 创新点**

创新点在于将因果推理评估从表面匹配转变为基于可导性（do‑calculus 与概率规则）的形式化验证，能够识别语义正确但表面不同的答案；并将验证器用于模型自我校正反馈。

**🔧 技术方法**

使用了符号推理技术（BFS 搜索 + 规则库）、d‑分离判定、do‑calculus 三条基本规则、概率理论规则，以及图形化表达式规范化。

**📊 数据集**

主要数据集包括：1）10,000 条人工合成的表达式对（可导/不可导）用于功能验证；2）CLadder 因果问答基准，用于评估 LLM 的因果推理准确性。

**📈 对比分析**

与传统字符串匹配、token‑F1、BERTScore、LLM‑as‑judge 等方法对比，符号验证器在合成数据上 100% 精确率/召回率，在 CLadder 上提升 10–30% 的准确率，且验证耗时仅数毫秒。

**⚠️ 局限性**

局限性包括：搜索空间随变量数和规则深度呈指数增长；当前反馈机制仅基于预测图而不考虑原始自然语言问题，可能导致与问题意图不完全一致；在图结构复杂或表达式过长时，计算成本仍较高。

---

## 91. Llama-3.1-FoundationAI-SecurityLLM-Reasoning-8B Technical Report

**arXiv ID:** 2601.21051 | [PDF](https://arxiv.org/pdf/2601.21051v1)

**作者:** Zhuoran Yang `[一作]` (Foundation AI), Amin Karbasi `[通讯]` (Foundation AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种8B参数的开源本地推理模型（Foundation‑Sec‑8B‑Reasoning），专门用于网络安全任务，并在多项安全与通用基准上展示了优异表现。

**💡 创新点**

创新点包括：① 直接从基础模型起步进行本地推理训练，而非在指令调优模型上进行微调；② 采用两阶段训练（SFT + RL‑VR）并加入“先推理后回答”的生成格式；③ 在RL阶段使用可验证奖励和KL正则化，显著提升推理质量并抑制奖励劫持；④ 通过系统提示和外部Guard实现高安全性。

**🔧 技术方法**

技术手段：
- 超大规模的SFT（约200万条含推理轨迹的合成样本）
- RL‑VR（GRPO算法，5个候选回复+可验证奖励）
- 关键字格式校验与长度惩罚
- KL‑divergence正则化防止模型漂移
- 对抗式安全评估（HarmBench）和Llama‑Guard防护。

**📊 数据集**

使用的数据集：
- 合成推理数据（覆盖数学、代码、指令、网络安全）
- 网络安全专项数据（CVE‑CWE映射、MITRE ATT&CK、CWEs、CTI问答、威胁情报、漏洞分类）
- 通用基准数据（MMLU‑Security、Cybermetric‑2000、SecBench、SecEval、AlpacaEval、BBH、GPQA、GSM‑8K、HumanEval、IFEval、HotpotQA、MATH）。

**📈 对比分析**

评估方法：在FAITH和lm‑evaluation‑harness框架下，用5次随机种子评估多项基准；对安全评估使用HarmBench；对LLM‑Guard安全性进行组合评估。性能方面：在网络安全基准上与70B大模型相当（例如‑RCM 75.3% vs 68.4%），远超8B指令调优模型；在通用基准上与Llama‑3.1‑8B‑Instruct持平甚至超越（AlpacaEval 62.6% vs 25.4%），仅在代码生成略有下降；安全性在系统提示+Guard下达到98.25%通过率。

**⚠️ 局限性**

局限性：
- 代码生成能力相对基线略降（~2%）；
- 需要系统提示或外部Guard才能获得较高安全通过率；
- RL训练数据规模有限，可能限制对某些极端推理任务的适应；
- 仍未进行专门的安全对齐微调，需进一步强化。

---

## 92. Pre-trained Encoders for Global Child Development: Transfer Learning Enables Deployment in Data-Scarce Settings

**arXiv ID:** 2601.20987 | [PDF](https://arxiv.org/pdf/2601.20987v1)

**作者:** Md Muhtasim Munif Fahim `[一作]`, Md Rezaul Karim `[通讯]` (University of Rajshahi)

**通讯引用:** 1817 | [OpenAlex ID](https://openalex.org/A5114549797)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用在44个国家MICS调查数据上预训练的表格掩码自编码器，构建了一个全球儿童发展预训练编码器，实现了在数据稀缺情况下的高效部署

**💡 创新点**

首次提出并验证了针对全球儿童发展任务的预训练编码器，证明多国多样化预训练能显著提升少样本迁移性能

**🔧 技术方法**

采用表格掩码自编码器（TMAE）进行自监督预训练，再用监督微调的MLP分类头进行fine‑tune，并进行模型集成

**📊 数据集**

使用UNICEF MICS第6轮 357,709名儿童的跨国数据（44个低中收入国家）作为预训练与评估数据集

**📈 对比分析**

与传统冷启动梯度提升树、基本MLP以及现代表格深度学习模型（FT‑Transformer、TabNet等）对比，在N=50样本时AUC平均提升8–12%，N=500时达到0.73，零样本时AUC最高0.84，表现优于对照组

**⚠️ 局限性**

局限包括仅基于横断面问卷数据，缺乏因果与临床验证，跨域泛化仍受数据质量与文化差异影响，且对低收入或小岛屿国家内部公平性需进一步评估

---

## 93. A Survey on Large Language Model Impact on Software Evolvability and Maintainability: the Good, the Bad, the Ugly, and the Remedy

**arXiv ID:** 2601.20879 | [PDF](https://arxiv.org/pdf/2601.20879v1)

**作者:** Bruno Claudino Matias `[一作]` (Virginia Commonwealth University), Rodrigo Spinola `[通讯]` (Virginia Commonwealth University)

**通讯引用:** 2759 | [OpenAlex ID](https://openalex.org/A5020235955)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 2020–2024 年公开论文进行系统文献综述，筛选 87 篇实证研究，分析 LLM 对软件可维护性和可演化性的正面影响、风险、弱点及缓解措施，并通过 Good/Bad/Ugly/Remedy 四个视角进行归纳。

**💡 创新点**

首次将 LLM 影响映射到可维护性与可演化性属性，构建 5 个研究问题（RQ1–RQ5）并采用 LLM‑ThemeCrafter 进行混合人机主题分析，形成一套完整的风险与缓解策略框架。

**🔧 技术方法**

利用 LLM（GPT‑3/3.5/4、LLaMA、BERT 系列等）辅助的主题提炼工具 LLM‑ThemeCrafter，结合 ACM、IEEE、Scopus 三大数据库检索与混合人机主题分析方法。

**📊 数据集**

来源为 711 篇候选文献，最终 87 篇实证研究；研究中多使用公开数据集（GitHub 开源项目、BugBench、OpenAI API 等）及作者自建实验数据。

**📈 对比分析**

通过对 87 篇研究按属性、正面影响、风险、弱点、缓解措施进行编码映射，统计各主题出现次数，形成属性分布与影响对应表；结果表明 LLM 在 analyzability、testability 等属性提升显著，而风险集中于 hallucination、上下文敏感等；性能方面多以案例数量与引用指标呈现。

**⚠️ 局限性**

局限包括：仅覆盖 2020–2024 年公开论文，可能遗漏最新动态；受限于 LLM‑ThemeCrafter 自动化质量；部分研究缺乏详细模型版本与参数信息，影响可复现性；评估方法多基于单一数据集，缺乏跨域验证。

---

## 94. MA-LipNet: Multi-Dimensional Attention Networks for Robust Lipreading

**arXiv ID:** 2601.20881 | [PDF](https://arxiv.org/pdf/2601.20881v1)

**作者:** Matteo Rossi `[一作]` `[通讯]` (Maharaja Agrasen University), Matteo Rossi (Maharaja Agrasen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了MA-LipNet网络，通过在视觉编码器中引入通道注意力、联合时空注意力和分离时空注意力三阶段滤波，实现了口型视频特征的多维度净化；

**💡 创新点**

创新点在于将多维注意力模块（通道、联合时空、分离时空）顺序应用于前端，显著降低了无关空间、时间和通道噪声；

**🔧 技术方法**

采用3D-CNN视觉前端，SE/CBAM式通道注意力，联合时空注意力（JSTA）与分离时空注意力（SSTA），以及带注意力的Seq2Seq解码器；

**📊 数据集**

在中文CMLR和英文GRID两个公开句子级口语识别数据集上进行实验；

**📈 对比分析**

与多种先进方法对比，MA-LipNet在CMLR上CER降至21.49%、在GRID上WER降至1.09%，均刷新了公开基准记录；

**⚠️ 局限性**

局限性在于仅在已知说话人数据集上验证，未充分考察说话人不依赖性和跨语言泛化能力。

---

## 95. Mam-App: A Novel Parameter-Efficient Mamba Model for Apple Leaf Disease Classification

**arXiv ID:** 2601.21307 | [PDF](https://arxiv.org/pdf/2601.21307v1)

**作者:** Md Nadim Mahamood `[一作]` (Begum Rokeya University), Kamrul Hasan `[通讯]` (Texas State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究了一种参数高效的Mamba‑based模型 Mam-App，用于苹果叶片病害分类，并在苹果、玉米、马铃薯数据集上进行验证。

**💡 创新点**

创新点在于采用 Mamba 状态空间网络替代传统 Transformer，实现在仅 0.051M 参数下就能达到甚至超过现有大模型的识别精度。

**🔧 技术方法**

技术方面结合卷积 Stem、五层 VisionMamba 块、全局平均池化与 Softmax 分类，同时利用 PCA、t‑SNE 可视化和 Random Forest/XGBoost 验证特征表达。

**📊 数据集**

使用了 PlantVillage 的苹果叶片（4 类）、玉米叶片（4 类）和马铃薯叶片（3 类）三大公开数据集。

**📈 对比分析**

与 MobileNetV2、DenseNet121、Xception 等主流模型对比，Mam‑App 在苹果数据集上实现 99.58% 准确率、99.30% 精度、99.14% 召回率、99.22% F1 分数，且参数仅 0.051M，显著压缩模型。

**⚠️ 局限性**

局限性包括对视觉相似度高的病害（如马铃薯）性能略低，且实验仅在实验室环境下完成，缺乏对真实田间多光照、多背景条件下的泛化验证。

---

## 96. UrduBench: An Urdu Reasoning Benchmark using Contextually Ensembled Translations with Human-in-the-Loop

**arXiv ID:** 2601.21000 | [PDF](https://arxiv.org/pdf/2601.21000v1)

**作者:** Muhammad Ali Shafique `[一作]` (Traversaal), Hamza Farooq `[通讯]` (Traversaal)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过多模型联合翻译并加入人工校对，构建了首个以推理为核心的乌尔都语基准套件（UrduBench），并在此基础上评估了多种大型语言模型的推理与指令能力。

**💡 创新点**

①引入上下文感知的翻译管道，利用多源机器翻译+LLM融合，确保语义与结构一致；②首次在低资源语言乌尔都语上系统性开展数学、常识与知识推理评测；③结合语言一致性检测，揭示代码切换对推理性能的影响。

**🔧 技术方法**

多源机器翻译（IndicTrans2、NLLB、Qwen‑3‑30B、Gemini‑2.5‑Pro）、GPT‑5.1翻译融合、自动启发式后处理、人工校对、LLM评测工具（LM Evaluation Harness）以及多种提示策略（Direct、CoT、Few‑Shot+CoT）。

**📊 数据集**

将MGSM、MATH‑500、CommonSenseQA、OpenBookQA四个英语推理基准转译为乌尔都语（UrduBench），共计≈17,000题目。

**📈 对比分析**

使用统一评测协议，对1B–14B+参数的多种开源LLM（Gemma、Falcon、Qwen、Phi‑4、DeepSeek等）在四个任务和多层难度下进行评测。结果显示Gemma‑3‑12B‑it和Falcon‑h1‑7B‑Instruct在平均准确率上遥遥领先，推理专用模型在MGSM上略占优势；语言一致性与CoT准确率正相关。

**⚠️ 局限性**

①翻译质量仍受限，偶尔出现语义漂移或代码切换；②仅覆盖四个基准，未涵盖更广泛的乌尔都语推理场景；③评测仅基于自动准确率，缺乏人工质疑层面；④模型对低资源语言的支持程度差异大，结果受预训练语料偏好影响。

---

## 97. MoCo: A One-Stop Shop for Model Collaboration Research

**arXiv ID:** 2601.21257 | [PDF](https://arxiv.org/pdf/2601.21257v1)

**作者:** Shangbin Feng `[一作]` (University of Washington), Yulia Tsvetkov `[通讯]` (University of Washington)

**通讯引用:** 5129 | [OpenAlex ID](https://openalex.org/A5062910836)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个名为 ModelCollab 的 Python 库，整合了 26 种模型协作算法与 25 个评测数据集，支持大规模、灵活且可扩展的模型协作实验。

**💡 创新点**

将分散的协作方法统一到单一框架，提供可比较的基准实验，并系统展示协作带来的性能提升、可扩展性与协同涌现现象。

**🔧 技术方法**

采用 API、文本、logit、权重四级信息交换机制，利用路由、辩论、logit 融合、模型融合等技术，并实现 GPU 可扩展执行。

**📊 数据集**

评测覆盖 25 个数据集，涵盖问答、推理、数学、编码、安全等领域，如 AGIeval、MMLU、GSM8k、TheoremQA、TruthfulQA、HumanEval 等。

**📈 对比分析**

通过在两套模型池（专业模型与通用模型）上对六大任务域进行宏平均，发现 61% 配置中协作方法优于单模型，文本与权重层方法表现最佳，提升可达 25.8%。

**⚠️ 局限性**

受限于模型架构兼容性、计算成本、恶意模型安全评估不足；当前仅包含 26 种算法，未覆盖所有最新方法，且需进一步研究动态模型选择与大规模扩展成本。

---

## 98. Do Pathology Foundation Models Encode Disease Progression? A Pseudotime Analysis of Visual Representations

**arXiv ID:** 2601.21334 | [PDF](https://arxiv.org/pdf/2601.21334v1)

**作者:** Pritika Vig `[一作]` (Massachusetts Institute of Technology), William Lotter `[通讯]` (Dana-Farber Cancer Institute)

**通讯引用:** 2479 | [OpenAlex ID](https://openalex.org/A5039371342)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究视觉基础模型是否能在病理图像表示空间中隐式编码疾病进展的连续轨迹，并通过扩散伪时序(DPT)对模型表示进行评估；

**💡 创新点**

提出使用扩散伪时序对视觉基础模型的嵌入空间进行轨迹保真度评估，证明模型可以在无监督情况下重构连续疾病进程；

**🔧 技术方法**

扩散伪时序(DPT)、Kendall 距离相关系数、内在维度估计(TWO‑NN)、多层 Vision Transformer 嵌入提取、Patch 级细胞分割(HistoPLUS)等；

**📊 数据集**

SPIDER H&E 补丁数据集（含四种癌症进展：皮肤鳞状细胞癌、结肠常规腺瘤-癌变路径、结肠锯齿形路径、乳腺导管乳头状癌）与六个基础模型（包括 DINOv2、UNI‑2、Virchow‑2、Prov‑GigaPath、CONCH、MuSK）；

**📈 对比分析**

与标签置换的零基线对比，使用 Kendall τ 评估轨迹保真度；发现所有病理模型显著优于零基线，视觉专用模型表现最佳；轨迹保真度与少量样本分类性能高度相关，模型在参考疾病上的保真度排名能预测在留一疾病上的 5‑shot F1，平均相关系数约 0.92；

**⚠️ 局限性**

仅适用于单一连续轨迹的情况，无法捕捉分支或多因子进展；数据标签粗糙且缺乏患者人口统计信息；实验仅在四个病理路径上进行，缺乏对更多病理类型的泛化验证；

---

## 99. NEMO: Execution-Aware Optimization Modeling via Autonomous Coding Agents

**arXiv ID:** 2601.21372 | [PDF](https://arxiv.org/pdf/2601.21372v1)

**作者:** Yang Song `[一作]` (C3 AI), Graham Neubig `[通讯]` (Carnegie Mellon University)

**通讯引用:** 21195 | [OpenAlex ID](https://openalex.org/A5068811427)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将自然语言描述的决策问题自动翻译为可执行的数学优化模型，提供从问题理解到代码生成的完整闭环。

**💡 创新点**

创新点在于：①将自治编码代理（ACA）作为一等抽象，直接执行并验证代码；②引入异步验证循环（模拟器与优化器互检）实现无标注的执行感知纠错；③结合 MBR 解码、自一致性、多样化记忆检索等技术提升稳定性和复用性。

**🔧 技术方法**

使用技术包括：OpenAI 的 LLM（o3）做推理、OpenHands（Claude 3.7 Sonnet）作为 ACA、Qwen3‑Embedding‑8B 做向量检索、MBR 解码、自一致性投票、外部记忆检索、异步验证与修复循环。

**📊 数据集**

使用九个公开优化基准数据集：OptiBench、OptMATH‑Bench、NL4OPT、NLP4LP、BWOR、IndustryOR、MAMO‑Easy、MAMO‑Complex、ComplexOR。

**📈 对比分析**

与多种基线（OptimAI、OptiMUS、OR‑LLM‑Agent、CoE、ORLM、LLMOPT、SIRL）进行对比，NEMO 在 8/9 任务上取得最高或相近最高准确率，并在若干数据集上超过对手 10–20% 的改进。

**⚠️ 局限性**

局限性包括：①推理时间较长（每实例 5–10 分钟，适合低频构造；不适合高吞吐场景）；②对算子执行环境的依赖导致资源消耗；③仍需人工监督，特别是对模型之外的异常情况；④在极大规模或实时场景下的可扩展性待验证。

---

## 100. Generative Recall, Dense Reranking: Learning Multi-View Semantic IDs for Efficient Text-to-Video Retrieval

**arXiv ID:** 2601.21193 | [PDF](https://arxiv.org/pdf/2601.21193v1)

**作者:** Zecheng Zhao `[一作]` (University of Queensland), Tong Chen `[通讯]` (University of Queensland)

**通讯引用:** 6464 | [OpenAlex ID](https://openalex.org/A5100461265)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 GRDR 框架，利用生成式召回 + 稠密重排实现高效准确的文本到视频检索；

**💡 创新点**

创新点：①多视角视频分词器为每段视频生成多条语义 ID，解决语义歧义；②共享代码簿的统一共训练，使分词器与生成检索器端到端对齐；③逐层训练与 Trie 限制解码相结合，进一步提升召回质量与速度；

**🔧 技术方法**

采用生成式检索（Seq‑to‑Seq + 量化分词器）、残差量化、查询引导对比学习、共享代码簿、Trie 约束解码、密集重排（X‑Pool）等技术；

**📊 数据集**

在四个标准 TVR 基准上评测：MSR‑VTT、ActivityNet、DiDeMo、LSMDC；

**📈 对比分析**

与稠密检索基线（CLIP4CLIP、X‑Pool、InternVideo2）和生成式检索基线（TIGER、AVG、T2VIndexer）对比；在诱导与全语料两种设置下，GRDR 的召回率与稠密检索相当，同时存储量减少 42‑500 倍，查询速度提升至 300×；

**⚠️ 局限性**

局限性：①多视角编码仍可能出现 ID 冲突，尤其在极大规模语料下；②对极长或多段视频的细粒度检索仍依赖后续稠密重排；③生成式召回的召回质量仍受训练数据多样性限制；④训练过程复杂，需要多阶段的共训练和大量伪查询。

---

## 101. Unplugging a Seemingly Sentient Machine Is the Rational Choice -- A Metaphysical Perspective

**arXiv ID:** 2601.21016 | [PDF](https://arxiv.org/pdf/2601.21016v1)

**作者:** Erik J Bekkers `[一作]`, Anna Ciaunica `[通讯]` (University of Lisbon)

**通讯引用:** 913 | [OpenAlex ID](https://openalex.org/A5054251497)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出生物唯心主义框架，质疑AI的主体性并解决拔掉人工智能的伦理悖论

**💡 创新点**

将经验N（表层体验）与经验G（自我维持的主体性）区分，构建生物唯心主义作为对功能主义和物理主义的替代论证

**🔧 技术方法**

哲学推演、元哲学框架构建、与生物学与物理学理论的对照分析

**📊 数据集**

未使用传统机器学习数据集，主要引用生物学实验（如Michael Levin的基底认知研究）和物理学基础

**📈 对比分析**

与功能主义、物理主义和形而上学视角对比，强调逻辑一致性、可解释性与经验支持，未给出数值性能指标

**⚠️ 局限性**

缺乏可验证的实验预测和具体伦理政策实施细节，依赖主观元哲学选择，可能难以获得广泛共识

---

## 102. Multi-modal Imputation for Alzheimer's Disease Classification

**arXiv ID:** 2601.21076 | [PDF](https://arxiv.org/pdf/2601.21076v1)

**作者:** Abhijith Shaji `[一作]` (Information Sciences Institute), Jose-Luis Ambite `[通讯]` (Information Sciences Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

利用条件扩散模型对缺失的 DWI 扫描进行补全，并评估其对阿尔茨海默病三分类（CN、MCI、AD）的影响。

**💡 创新点**

首次将 3D DDPM 用作 T1→DWI 跨模态生成器，实现对缺失模态的高质量补全，并探讨其对少数类别检测的潜在提升。

**🔧 技术方法**

采用 3D U‑Net 结构的条件 DDPM 生成 DWI；使用 3D CNN 单模态和双模态网络进行分类；训练过程包括混合精度、AdamW 优化和贝叶斯超参搜索。

**📊 数据集**

使用 ADNI 前三期数据（T1 和 FA 纤维张量映射），划分为训练/验证/测试集，包含 3901 张 T1、642 张完整 T1+DWI 及对应 FA 图像。

**📈 对比分析**

与空白填充、诊断均值填充以及无补全基线对比；在部分配置下取得准确率、平衡准确率和宏 F1 的提升，微 AUC 亦有提升，但增益不稳定，且大多来自 T1 训练样本量增大。

**⚠️ 局限性**

受限于配对样本量不足、生成图像质量与真实数据的差距、对不同下游任务的泛化能力欠缺，以及补全对少数类别效果不一致等问题。

---

## 103. A Theory of Universal Agnostic Learning

**arXiv ID:** 2601.20961 | [PDF](https://arxiv.org/pdf/2601.20961v1)

**作者:** Steve Hanneke `[一作]` (Purdue University), Shay Moran `[通讯]` (Technion and Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文对二分类学习问题进行理论分析，提出了一个四段式学习率层次结构（e^{-o(n)}、super-root、super-root、任意慢速），并给出了相应的学习算法与证明。

**💡 创新点**

创新点在于：①在非可实现（shatter infinite Littlestone tree）以及非VCL树（shatter infinite VCL tree）两类概念族中，分别实现了与可实现情况完全不同的学习率；②首次对部分概念类（partial concept class）在 Bayes 可实现条件下实现 o(n^{-1/2}) 的学习率；③通过对 Gale‑Stewart 游戏的赢策略与转导式经验风险最小化相结合，构造了新的通用学习器。

**🔧 技术方法**

核心技术包括：Gale‑Stewart 游戏赢策略、部分概念类的 VC 维度控制、经验风险与贝叶斯误差的 Bernstein 不等式、传递学习框架下的经验误差一致收敛、以及对误差率条件期望的均匀 Bernstein 结合收敛。

**📊 数据集**

本文属于理论工作，没有使用具体数据集；所有结果均为上界下界与渐进性能分析。

**📈 对比分析**

与先前仅针对可实现（realizable）或有限 VC 维度概念类的学习率（如 n^{-1/2}）相比，本文在更一般的 Bayes‑可实现环境下实现了更快的 super‑root 或 e^{-o(n)} 收敛速率，证明了在不同树结构下的最佳可实现率。

**⚠️ 局限性**

局限性在于：①算法实现复杂、依赖于未知参数（如批次大小 b^*）；②对部分概念类的学习率提升主要在理论层面，缺乏实证验证；③对于某些无限树结构的类，仍可能仅能达到慢速收敛（例如 o(n^{-1/2})）。

---

## 104. Exact (n + 2) Comparison Complexity for the N-Repeated Element Problem

**arXiv ID:** 2601.21202 | [PDF](https://arxiv.org/pdf/2601.21202v1)

**作者:** Andrew Au `[一作]` `[通讯]` (Independent Researcher), Andrew Au (Independent Researcher)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

论文在等价比较模型下研究了在长度为2n、含有n+1个不同值的数组中找到出现n次元素的最优比较复杂度。

**💡 创新点**

创新点在于给出精确最优比较次数为n+2，并通过图论与对手论证构造了匹配与极大团结构来证明下界。

**🔧 技术方法**

主要技术包括对手论证、等价/潜在相等图的构造、最小顶点覆盖与完美匹配的图论性质以及pillar匹配构造。

**📊 数据集**

论文没有使用具体数据集，而是对理论模型下所有合法输入进行分析。

**📈 对比分析**

通过确定性算法实现n+2次比较，并证明其与下界相匹配，表明在等价比较模型下已达到最优。

**⚠️ 局限性**

局限性在于仅适用于仅允许等价比较的模型，若引入大小比较或其他算术操作则不再适用；并且只针对长度为2n、包含n+1不同值的特殊数组。

---

## 105. Human-LLM Collaborative Feature Engineering for Tabular Data

**arXiv ID:** 2601.21060 | [PDF](https://arxiv.org/pdf/2601.21060v1)

**作者:** Zhuoyan Li `[一作]` (Purdue University), Yunyao Li `[通讯]` (Adobe)

**通讯引用:** 1590 | [OpenAlex ID](https://openalex.org/A5106404797)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一个人机协作的特征工程框架：LLM仅负责生成特征变换候选项，后续通过显式的效用与不确定性建模来选择最佳变换，并在必要时主动请求人工偏好反馈。

**💡 创新点**

创新点包括：
1) 将特征变换的“生成”和“选择”完全解耦，让LLM仅做候选生成；
2) 用贝叶斯神经网络构建效用代理模型，并通过UCB获取不确定性信息；
3) 设计了基于置信区间重叠与潜在收益阈值的两层条件（C1、C2），实现了对人工偏好查询的智能触发；
4) 将人工偏好作为概率观测通过Probit函数更新代理模型，从而在选择决策中融合人类专业判断。

**🔧 技术方法**

核心技术包括：
- 大语言模型（GPT‑4o、GPT‑3.5‑Turbo、DeepSeek‑V3、GPT‑5）用于候选特征变换生成；
- 贝叶斯神经网络（BNN）作为效用代理；
- Upper Confidence Bound（UCB）作为采样准则；
- 通过Probit模型将人工偏好转化为对模型参数的概率约束；
- 迭代贝叶斯优化流程；
- 评价指标使用 MLP 与 XGBoost 作为下游模型。

**📊 数据集**

实验使用 18 个公开数据集（Kaggle、UCI），涵盖分类与回归任务，包含 13 个二分类数据集（flight、wine、loan、diabetes、titanic 等）以及一个私有 conversion 数据集。

**📈 对比分析**

与 AutoML 基线（OpenFE、AutoGluon）以及 LLM 基线（CAAFE、OCTree）进行比较。结果显示：
- 在 13 个二分类数据集上，Ours（无人工）平均 AUROC 提升 7.24%（MLP）/9.02%（XGBoost）；
- Ours（有人工）进一步提升 8.96%/11.23%；
- 人工偏好反馈带来的平均提升约 1.7%（MLP）/3.2%（XGBoost）；
- 在用户研究中，Ours（Alg）方案的最终性能显著高于控制组和自我推理组，并且完成时间更短、认知负担更低。

**⚠️ 局限性**

局限性：
1) 依赖昂贵的 LLM 调用，且 LLM 的候选生成质量对最终效果影响较大；
2) 人工偏好查询仍需人工参与，成本难以完全消除；
3) 在特征工程的早期阶段代理模型的效用预测可能不够精准；
4) 评估集中在表格数据的分类任务，泛化到其他任务（如时间序列、图数据）尚未验证；
5) 仅在实验中使用 GPT‑4o 作为代理，其他 LLM 的实际表现需进一步探索。

---

## 106. Modeling Endogenous Logic: Causal Neuro-Symbolic Reasoning Model for Explainable Multi-Behavior Recommendation

**arXiv ID:** 2601.21335 | [PDF](https://arxiv.org/pdf/2601.21335v1)

**作者:** Yuzhe Chen `[一作]` (Nanjing University of Science and Technology), Jia Wu `[通讯]` (Macquarie University)

**通讯引用:** 12038 | [OpenAlex ID](https://openalex.org/A5007475662)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种将因果推断与神经符号推理结合的多行为可解释推荐模型CNRE，模拟人类决策过程并输出多层次可解释结果。

**💡 创新点**

创新点在于：①利用用户行为链的内生逻辑构造可解释符号规则；②在前门调整框架下实现因果中介，消除协变量干扰；③根据行为强度动态选择并行、并发与串行三条神经符号推理路径。

**🔧 技术方法**

技术包括LightGCN、可学习超图卷积、前门调整因果推断、神经逻辑门（AND/OR）以及自适应投影机制。

**📊 数据集**

使用了Beibei、Taobao和Tmall三个公开电商多行为数据集（视图、加入购物车、收藏、购买）。

**📈 对比分析**

与单行为模型NeuMF、并行/级联多行为模型MBGCN/CRGCN/MB-CGCN、对比学习MBSSL、因果模型DCCF/CausalD以及神经符号模型NS-ICF/FENCR等基线比较，CNRE在HR@K、NDCG@K指标上均显著领先，且保持多层可解释性。

**⚠️ 局限性**

局限在于：依赖行为链顺序，忽略时间序列细节；缺乏大规模用户实验评估可解释性的实际价值；因果中介近似，仍需进一步验证因果效应的真实性。

---

## 107. Shape of Thought: Progressive Object Assembly via Visual Chain-of-Thought

**arXiv ID:** 2601.21081 | [PDF](https://arxiv.org/pdf/2601.21081v1)

**作者:** Yu Huo `[一作]` (Sun Yat-sen University), Xiaoying Tang `[通讯]` (The Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Shape-of-Thought (SoT)，一种在单一多模态自回归模型中交替生成文本决策与对应渲染状态的视觉链式思维框架，用于逐步合成三维物体。

**💡 创新点**

创新点在于将生成过程拆解为可解释的文本思考与可视化状态交替，消除了对外部三维引擎的依赖，提供了透明、可校验的生成轨迹。

**🔧 技术方法**

使用的技术包括 Bagel-7B 多模态 Transformer、VAE+ViT 视觉编码、Rectified Flow 采样、Blender 渲染协议以及自回归交互式生成策略。

**📊 数据集**

构建了 SoT-26K（约26k条基于 PartNet CAD 的分步渲染轨迹）和 T2S-CompBench 评测基准，用于训练和评估。

**📈 对比分析**

通过与直接生成、文本 CoT 以及多种 3D 基准对比，SoT 在组件计数、属性绑定、拓扑结构等指标上平均提升约 20%，在 T2S-CompBench 上大多数结构与过程指标均获得最高分，轨迹稳定性达 90%+。

**⚠️ 局限性**

主要限制包括仅支持单视角渲染、未直接生成 3D 网格、对极其复杂结构的可视化受分辨率和算力限制，以及对多视角监督的实现仍待完善。

---

## 108. AI-Assisted Engineering Should Track the Epistemic Status and Temporal Validity of Architectural Decisions

**arXiv ID:** 2601.21116 | [PDF](https://arxiv.org/pdf/2601.21116v1)

**作者:** Sankalp Gilda `[一作]` (DeepThought Solutions), Shlok Gilda `[通讯]` (University of Florida)

**通讯引用:** 461 | [OpenAlex ID](https://openalex.org/A5021727664)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证 First Principles Framework（FPF）以实现 AI 辅助软件工程中的知识层级、保守聚合与证据时效管理

**💡 创新点**

核心创新是三大要求：①显式认知层区分未验证假设与已验证结论；②基于 Gödel t‑norm 的最弱链聚合保障不信任膨胀；③自动化证据衰减追踪实现时间透明性

**🔧 技术方法**

技术手段包括：模糊逻辑的 Gödel t‑norm 聚合、证据衰减模型（有效期阈值与统一不确定性底线）、属性测试与 SMT 验证、依赖图与证据层级管理

**📊 数据集**

主要使用两项内部项目的架构决策记录及其 Benchmark（Redis/Memcached 负载测试、流量模型等）作为数据集

**📈 对比分析**

通过对比传统 ADR 与 FPF 的失效检测率，FPF 在两项目中发现 23% 的决策证据在两个月内过期，其中 86% 仅在事故后被动发现，证明了自动化时效追踪的有效性

**⚠️ 局限性**

局限性包括：证据有效期设定依赖团队经验、聚合仅在串行链中严格应用、未覆盖并行或多源证据的最佳聚合方案、需要进一步跨组织实验验证学习聚合器的可行性

---

## 109. Physics-Guided Tiny-Mamba Transformer for Reliability-Aware Early Fault Warning

**arXiv ID:** 2601.21293 | [PDF](https://arxiv.org/pdf/2601.21293v1)

**作者:** Changyu Li `[一作]` (Great Bay University), Fei Luo `[通讯]` (Great Bay University)

**通讯引用:** 2114 | [OpenAlex ID](https://openalex.org/A5101711943)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种物理引导的 Tiny‑Mamba Transformer（PG‑TMT）与基于极值理论的阈值与双阈 hysteresis 的在线早期故障预警框架。

**💡 创新点**

创新点在于将三支路编码器（卷积、Tiny‑Mamba 状态空间、局部 Transformer）与物理谱对齐注意力、极值尾部阈值以及双阈 hysteresis 三大组件融合，形成可校准、可解释、低延迟的可靠性驱动预警。

**🔧 技术方法**

使用深度学习（Tiny‑Mamba、局部 Transformer、卷积）、物理先验（基于轴承缺陷频率、相位、谱对齐损失）、极值理论（GPD 及阈值推导）以及流式推理与右删失生存分析等技术。

**📊 数据集**

在公开的旋转机械振动数据集 CWRU、Paderborn、XJTU‑SY 以及工业试点数据上进行实验。

**📈 对比分析**

与多种基线（CNN、TCN、LSTM、SSM、Transformer 等）在流式协议下比较，PG‑TMT 在 PR‑AUC、ROC‑AUC 及 MTTD 上均优于或相当，且在匹配的误报率下平均检测时间缩短、误报率可控，表现出跨域迁移能力。

**⚠️ 局限性**

局限包括需先验轴承几何和转速信息以设定谱先验、对大幅操作点漂移的自校准不足、仅针对振动传感器未覆盖其他资产类型，且阈值依赖手工调参。

---

## 110. Test-Time Adaptation for Unsupervised Combinatorial Optimization

**arXiv ID:** 2601.21048 | [PDF](https://arxiv.org/pdf/2601.21048v1)

**作者:** Yiqiao Liao `[一作]` (University of California San Diego), Parinaz Naghizadeh `[通讯]` (University of California San Diego)

**通讯引用:** 650 | [OpenAlex ID](https://openalex.org/A5091717833)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种模型无关的测试时自适应框架TACO，能在不重新训练的前提下快速将已训练的无监督NCO模型（如EGN、Meta-EGN）适配到每个测试实例；

**💡 创新点**

核心创新是结合缩放-扰动（SP）热启动技术，将训练好的参数部分收缩并加入噪声，既保留分布层次的归纳偏差，又为实例级微调提供更有利的起点；

**🔧 技术方法**

使用无监督梯度更新、GNN（GIN层）、熵/约束惩罚的损失函数、SP初始化、在线和离线两种更新策略；

**📊 数据集**

在Twitter、COLLAB、RB200、RB500四个图数据集上进行评估，并在分布漂移（Twitter→RB200）与动态图（Twitter Tennis UO、COVID-19 England）情境下测试；

**📈 对比分析**

与基线（原始EGN/Meta-EGN、随机初始化、Fine‑tune、在线Fine‑tune）以及非神经方法（贪心、Toenshoff‑Greedy、iSCO、PQQA、Gurobi）进行对比；TACO在MVC/MC任务中平均ApR均优于基线，尤其在分布漂移和动态场景下表现最突出，且计算开销几乎无增；

**⚠️ 局限性**

主要限制是对SP超参数的依赖、训练时未考虑批量测试、以及在极端图规模/稠密度下仍受限于GNN深度与解码效率；未来可结合更高效解码、并行化种子搜索及训练时兼容性提升。

---

## 111. CUA-Skill: Develop Skills for Computer Using Agent

**arXiv ID:** 2601.21123 | [PDF](https://arxiv.org/pdf/2601.21123v1)

**作者:** Tianyi Chen `[一作]` (Microsoft), Kazuhito Koishida `[通讯]` (Microsoft)

**通讯引用:** 1035 | [OpenAlex ID](https://openalex.org/A5084879161)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一个面向 Windows 桌面环境的可复用、参数化的计算机使用技能库 Cua‑Skill，并基于此构建了一个检索增强的技能驱动代理 Agent，实现从自然语言指令到 GUI 交互的端到端流程。

**💡 创新点**

①提出基于技能执行图和组合图的结构化技能抽象，捕捉人类桌面操作的可复用程序化知识；②通过检索增强的 LLM 规划器实现动态技能检索、参数配置与执行，减少对工具列表的暴露；③实现跨应用、跨任务的技能可迁移性与可靠性。

**🔧 技术方法**

使用大型语言模型（GPT‑5、Qwen3‑VL、GPT‑4o 等）作为规划与推理器；语义+词法检索构建技能索引；GUI grounding 模型实现屏幕坐标预测；可持久化的记忆模块与执行图遍历机制；参数化执行图支持脚本与 GUI 混合执行。

**📊 数据集**

以 452 个原子技能为基础，覆盖 17 个常用 Windows 应用；合成 200K 任务用于评估技能执行；在 WindowsAgentArena benchmark 进行端到端实验；内部生成的技能组合图和执行图。

**📈 对比分析**

与 Ultra‑CUA、Operator、Agent‑S 系列等现有 CUAs 在 WAA benchmark 进行比较；采用最佳三次策略；Agent 在 57.5% 成功率（最佳）且仅 30 步，显著优于其它方法（如 Agent‑S3 56.6%）；在技能执行可靠性测试中平均 76.4% 成功率。

**⚠️ 局限性**

对于 UI 复杂、视觉内容丰富的应用（如 PowerPoint、VLC）表现仍不佳；性能受 LLM 推理与检索能力限制；技能库覆盖有限，新增应用需手工创建技能；对 GUI 变化的鲁棒性仍需改进。

---

## 112. Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed

**arXiv ID:** 2601.21094 | [PDF](https://arxiv.org/pdf/2601.21094v1)

**作者:** Minjae Kwon `[一作]` (University of Virginia), Lu Feng `[通讯]` (University of Virginia)

**通讯引用:** 6989 | [OpenAlex ID](https://openalex.org/A5050527785)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究在分布偏移下安全强化学习的训练时间安全保证是否能迁移到部署，并提出基于预测的安全屏蔽方法。

**💡 创新点**

发现安全一般化缺口并提出统一临床模拟器与贝叶斯自适应神经ODE预测屏蔽。

**🔧 技术方法**

使用安全强化学习算法（PPO‑Lag、CPO 等）、预测模型 BA‑NODE 以及动态屏蔽机制。

**📊 数据集**

采用 UVA–Padova 糖尿病模拟器生成的合成数据（T1D、T2D、不同年龄组）。

**📈 对比分析**

与八种安全 RL 基线及规则屏蔽对比，屏蔽提升 Time‑in‑Range 约10–14%，降低风险指数，抑制血糖波动。

**⚠️ 局限性**

屏蔽依赖预测模型精度，难以处理突发事件与缺乏真实患者数据验证，易受模型误差影响。

---

## 113. Robust Federated Learning for Malicious Clients using Loss Trend Deviation Detection

**arXiv ID:** 2601.20915 | [PDF](https://arxiv.org/pdf/2601.20915v1)

**作者:** Deepthy K Bhaskar `[一作]` (Govt. Model Engineering College APJ Abdul Kalam Technological University), Binu V P `[通讯]` (Govt. Model Engineering College APJ Abdul Kalam Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出FL‑LTD防御框架，通过监测客户端训练损失随时间变化来检测恶意行为，并利用短期记忆机制对检测到的客户端进行自适应减权。

**💡 创新点**

创新点在于仅使用客户端报告的损失曲线而非梯度或加密信息来识别攻击，并引入短期记忆提升对自适应攻击的鲁棒性。

**🔧 技术方法**

采用损失趋势偏差检测、短期记忆计数以及基于权重的自适应聚合技术。

**📊 数据集**

实验使用非IID的 Federated MNIST 数据集。

**📈 对比分析**

与标准 FedAvg 在损失操纵攻击下对比，FL‑LTD 将最终准确率从 0.41 提升至 0.84，几乎翻倍，且计算与通信开销极小。

**⚠️ 局限性**

局限性包括只能检测损失异常，无法防御直接梯度篡改或协同 Sybil 攻击；阈值设置固定，可能在极端非IID场景下失效。

---

## 114. PHDME: Physics-Informed Diffusion Models without Explicit Governing Equations

**arXiv ID:** 2601.21234 | [PDF](https://arxiv.org/pdf/2601.21234v1)

**作者:** Kaiyuan Tan `[一作]` (Vanderbilt University), Thomas Beckers `[通讯]` (Vanderbilt University)

**通讯引用:** 4776 | [OpenAlex ID](https://openalex.org/A5087890782)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了PHDME框架，将端到端的物理先验与扩散模型结合，实现在稀疏观测下的多步动力学预测与不确定性校准。

**💡 创新点**

通过学习Gaussian Process分布的端口-哈密顿系统作为可重用的物理先验，并将其信息注入扩散模型的物理一致性损失，同时使用分层对合预测实现无显式方程的物理约束与校准不确定性。

**🔧 技术方法**

使用Gaussian Process端口-哈密顿系统建模、DDPM扩散模型、物理一致性正则化、分层对合预测、U‑Net架构以及随机梯度训练等技术。

**📊 数据集**

实验数据集包括基于模拟的固定端弦波（波动方程）、1D浅水系统以及真实高速摄像记录的弹簧振动。

**📈 对比分析**

与标准DDPM、有限物理约束DDPM、GP‑dPHS积分器和NeuralODE等基线在相同网络与训练预算下比较，PHDME在MSE和NCS指标上分别降低约28%，并在真实弹簧数据上与最优纯数据模型相当。

**⚠️ 局限性**

仍需预先指定端口-哈密顿结构模板，对高维空间与长时间步长的可扩展性有限，且对观测稀疏性高度敏感，物理先验训练耗时较大。

---

## 115. AI-based Prediction of Biochemical Recurrence from Biopsy and Prostatectomy Samples

**arXiv ID:** 2601.21022 | [PDF](https://arxiv.org/pdf/2601.21022v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 116. Causal Discovery for Explainable AI: A Dual-Encoding Approach

**arXiv ID:** 2601.21221 | [PDF](https://arxiv.org/pdf/2601.21221v1)

**作者:** Henry Salgado `[一作]` (University of Texas at El Paso), Martine Ceberio `[通讯]` (University of Texas at El Paso)

**通讯引用:** 611 | [OpenAlex ID](https://openalex.org/A5047541010)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过双编码策略结合约束型因果发现算法，构建了适用于混合数据类型的全局因果图。

**💡 创新点**

创新点在于使用两种互补的分类变量编码（drop-first、drop-last）并通过多数投票合并结果，以解决因果发现中因果变量编码导致的数值不稳定问题。

**🔧 技术方法**

采用了Fast Causal Inference (FCI) 算法、Fisher's z检验、投票合并以及Pearson相关权重计算等技术。

**📊 数据集**

实验数据集为著名的Titanic生存预测数据集，包含混合的连续、二元和多分类特征。

**📈 对比分析**

与SHAP全局特征重要性和决策树分裂路径进行对比，发现因果图的中心节点与重要特征一致，决策树预测准确率为82%，随机森林为84%。

**⚠️ 局限性**

局限性包括对Causal Markov、Faithfulness假设的依赖、离散化带来的信息损失、样本量需求，以及未与其他现代混合数据因果方法进行系统比较。

---

## 117. MADE: Benchmark Environments for Closed-Loop Materials Discovery

**arXiv ID:** 2601.20996 | [PDF](https://arxiv.org/pdf/2601.20996v1)

**作者:** Shreshth A Malik `[一作]`, Yarin Gal `[通讯]` (University of Oxford)

**通讯引用:** 25668 | [OpenAlex ID](https://openalex.org/A5029186201)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套可闭环评估材料发现过程的Benchmark框架MADE，支持完整的生成–筛选–评估循环。

**💡 创新点**

创新点在于将材料发现视为多目标、稀疏的序列决策问题，并提供可组合的管线组件和全自动LLM代理评估；首次实现对闭环发现效率的系统化度量。

**🔧 技术方法**

使用了生成模型（如Chemeleon）、机器学习势能模型（MLIP）作为代理评估、规划器（随机、diversity、LLM）、筛选器和LLM驱动的全流程代理，并在Python gym‑style环境中实现。

**📊 数据集**

主要基于Materials Project中的MP‑20结构集作为初始材料，利用训练好的MLIP作为能量oracle；实验还引用了MP‑20生成的结构与能量数据。

**📈 对比分析**

与随机生成器基准对比，通过Acceleration Factor、Enhancement Factor、AUDC和mSUN等指标评估，各模块化管线（如Chemeleon+MLIP、LLM规划）显著提升发现速率，Agentic LLM Orchestrator与最佳管线竞争并在结构多样性上表现更优。

**⚠️ 局限性**

局限在于仅使用MP‑20训练的生成模型和MLIP代理，存在分布偏差；未覆盖高精度DFT或实验oracle；批量查询、多目标任务与更大化学空间的扩展尚待进一步研究。

---

## 118. Semantic-Guided Dynamic Sparsification for Pre-Trained Model-based Class-Incremental Learning

**arXiv ID:** 2601.21345 | [PDF](https://arxiv.org/pdf/2601.21345v1)

**作者:** Ruiqi Liu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Yongjun Xu `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 5808 | [OpenAlex ID](https://openalex.org/A5103245119)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文提出一种名为 SGDS 的动态稀疏化方法，在类增量学习中通过在激活空间引导稀疏子空间来替代传统参数正则化，提升模型在新任务上的可塑性。

**💡 创新点**

创新点在于把注意力从参数空间转移到激活空间，采用语义引导的子空间方向选择与稀疏压缩两阶段机制，使得同类共享子空间、不同类分离子空间，从而有效抑制干扰。

**🔧 技术方法**

核心技术包括基于预训练 Vision Transformer 的轻量级适配器、语义相似度评估、概率稀疏采样、稀疏约束实现的子空间压缩以及跨层激活引导。

**📊 数据集**

实验数据集涵盖 CIFAR‑100、ImageNet‑R、ImageNet‑A 和 ObjectNet 四大增量学习基准，任务划分为 20 任务×5 类 或 10 任务×20 类的设置。

**📈 对比分析**

与多种基准方法（包括 L2P、DualPrompt、TUNA、RanPAC 等）以及回放式方法（FOSTER、MEMO、iCaRL）对比，SGDS 在平均准确率与最终准确率上均取得最高或相近最高分，最大提升达约 1.19%，并在所有四个基准上优于现有最优方法。

**⚠️ 局限性**

主要局限在于仍需对稀疏比例、探索平衡等超参数进行调优，且对极端任务序列的扩展性和跨任务迁移能力尚待进一步验证。

---

## 119. Missing-Data-Induced Phase Transitions in Spectral PLS for Multimodal Learning

**arXiv ID:** 2601.21294 | [PDF](https://arxiv.org/pdf/2601.21294v1)

**作者:** Anders Gjølbye `[一作]` (Technical University of Denmark), Lars Kai Hansen `[通讯]` (Technical University of Denmark)

**通讯引用:** 18686 | [OpenAlex ID](https://openalex.org/A5018292103)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究在双视图数据中两侧同时存在独立MCAR缺失时，Partial Least Squares（PLS‑SVD）的可恢复性与相位转移；

**💡 创新点**

创新点在于把双视图缺失映射为有效信号衰减θ_eff = √ρθ，给出精确的BBP阈值 θ_crit = 1/((α_xα_y)^{1/4}√ρ) 以及闭式的重叠公式；

**🔧 技术方法**

采用随机矩阵理论、spiked 矩阵模型、Replica 计算、BBP 相位分析以及大规模数值模拟；

**📊 数据集**

使用的真实数据集为 TCGA‑BRCA（RNA‑seq 与 DNA 甲基化）和 PBMC Multiome（scRNA‑seq 与 scATAC‑seq）并做半合成实验；

**📈 对比分析**

通过仿真和半合成实验对比理论重叠公式，结果与理论高度一致（相关系数 >0.99），同时验证缺失率对阈值的影响；

**⚠️ 局限性**

局限性包括仅处理 MCAR 缺失、单因子、Gaussian 噪声，且对非 MCAR 或多因子情况的推广尚未验证。

---

## 120. STAER: Temporal Aligned Rehearsal for Continual Spiking Neural Network

**arXiv ID:** 2601.20870 | [PDF](https://arxiv.org/pdf/2601.20870v1)

**作者:** Matteo Gianferrari `[一作]` (University of Modena and Reggio Emilia), Simone Calderara `[通讯]` (University of Modena and Reggio Emilia)

**通讯引用:** 5255 | [OpenAlex ID](https://openalex.org/A5075481810)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了用于连续学习的时序对齐经验回放SNN框架 STAER，专门解决SNN在类增量学习中的灾难性遗忘与时序漂移。

**💡 创新点**

创新点在于将可微软动态时间规整（soft‑DTW）对齐损失与时间扩张/收缩机制相结合，显式保持跨任务的时序结构，首次让SNN在类增量学习中实现与ANN基线相当甚至更优的性能。

**🔧 技术方法**

使用的技术包括：Leaky Integrate‑and‑Fire (LIF) 神经元、ArcTan 替代理论梯度、soft‑DTW 时序对齐损失、经验回放（ER/DER/DER++）策略、时间压缩/扩张机制以及 ResNet19 SNN 骨干。

**📊 数据集**

评估数据集为 Sequential‑MNIST 与 Sequential‑CIFAR10 两个标准类增量学习基准。

**📈 对比分析**

在与 ANN 经验回放基准（ER、DER、DER++）以及 SNN 经验回放基准的对比中，STAER 在两数据集上均达到了与 ANN 基线相当甚至更高的最终平均准确率，并在遗忘度方面表现更佳。

**⚠️ 局限性**

局限性包括：对时间步数和缓冲区大小敏感；soft‑DTW 的计算开销较大；在更大规模任务或更复杂事件‑基数据集上的可扩展性尚未充分验证。

---

## 121. Disturbance-Aware Flight Control of Robotic Gliding Blimp via Moving Mass Actuation

**arXiv ID:** 2601.21188 | [PDF](https://arxiv.org/pdf/2601.21188v1)

**作者:** Hao Cheng `[一作]` (Peking University), Feitian Zhang `[通讯]` (Peking University)

**通讯引用:** 627 | [OpenAlex ID](https://openalex.org/A5059772938)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究针对轻型气球（RGBlimp）平台，提出了基于移动风速估计（MHE）和模型预测控制（MPC）的扰动感知飞行控制框架，实现了在风洞环境下的稳健航向控制；

**💡 创新点**

创新点在于：①首次将MHE实时估计风速与MPC耦合，用于轻型气球的动力学；②利用2-DoF内部移动质量机制实现惯性与气动时变力矩的快速调节；③通过在实际飞行中验证估计与控制性能，显著优于传统PID控制；

**🔧 技术方法**

技术主要包括：①非线性动力学建模与风速耦合；②基于滑动窗口的非线性MHE求解；③基于离散非线性系统的MPC优化；④利用OptiTrack测量与ESP32+MicroROS实现实时控制；

**📊 数据集**

实验数据集为室内风洞实验中的10m×2m试验场景，使用Dyson AM07风扇产生0、0.5、1.0 m/s的头风或侧风，共计120次飞行试验；

**📈 对比分析**

与传统PID控制进行对比，MHE‑MPC在头风与侧风下的RMSE均下降约50%–90%，横向偏移与航向误差显著减小，控制动作更平滑，失控率明显降低；

**⚠️ 局限性**

局限性包括：①模型依赖于对风速的慢变假设，瞬时风速突变可能导致估计误差；②计算量较大，需高性能PC支持；③实验环境为封闭室内风洞，实际户外多变气象条件仍需进一步验证；

---

## 122. Conditional Denoising Model as a Physical Surrogate Model

**arXiv ID:** 2601.21021 | [PDF](https://arxiv.org/pdf/2601.21021v1)

**作者:** José Afonso `[一作]` (Instituto Superior Técnico), Vasco Guerra `[通讯]` (Instituto Superior Técnico)

**通讯引用:** 5750 | [OpenAlex ID](https://openalex.org/A5050777332)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了条件去噪模型（Conditional Denoising Model，CDM），用于在物理仿真中学习低维物理解的几何结构，并以确定性固定点迭代方式实现推理；

**💡 创新点**

创新点在于将去噪自编码器推广到连续噪声尺度，构建时间无关的向量场，使推理过程从随机扩散变为确定性投影；同时去噪目标本身作为隐式正则化器，能够在未见物理方程时实现更严格的物理一致性；

**🔧 技术方法**

采用去噪自编码器与扩散模型理论、变分推断、分数匹配、概率流ODE、基于噪声级的多尺度训练，以及固定点迭代和自适应步长推理；

**📊 数据集**

使用低温等离子体模拟数据（LoKI-Kinetics Boltzmann+Chemistry，LoKI），包含三维实验条件与十七种粒子浓度，共计3000样本；

**📈 对比分析**

与两类基线进行对比：1）Physics-consistent NN（软物理约束训练） 2）NN+Projection（后处理硬投影）。CDM 在RMSE、物理一致性误差上均优于基线，尤其在参数量较小（≈5000参数）和样本量较少（30%）时性能更突出；

**⚠️ 局限性**

局限性包括：1）模型仍需在推理时进行多次迭代，计算成本高；2）目前仅针对稳态（平衡）问题，时变动力学尚未验证；3）缺乏对高维空间或连续场景的直接应用与评估。

---

## 123. MGSM-Pro: A Simple Strategy for Robust Multilingual Mathematical Reasoning Evaluation

**arXiv ID:** 2601.21225 | [PDF](https://arxiv.org/pdf/2601.21225v1)

**作者:** Tianyi Xu `[一作]` (McGill University), David Ifeoluwa Adelani `[通讯]` (Canada CIFAR AI Chair)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 MGSM-Pro 数据集，为每个 MGSM 问题生成五个变体（名字、数字、无关上下文），并在多语言环境下评估大型语言模型（LLM）的数学推理鲁棒性。

**💡 创新点**

①扩展 MGSM 为 MGSM-Pro，加入五个数字/名字变体并引入无关上下文；②系统评估多语言 LLM 的鲁棒性，揭示低资源语言鲁棒性差；③提出至少五个实例评估的最佳实践。

**🔧 技术方法**

使用 GSM‑Symbolic 框架生成可参数化模板，利用 Gemini 2.0 Flash 等 LLM 进行多语言翻译并进行人工校验；对数字进行系统采样保证数学正确性；采用零样本推理提示（CoT）进行评估。

**📊 数据集**

MGSM-Pro（基于 MGSM 225 题，覆盖 9 种语言，包含 Symbolic 和 Irrelevant Context 6 个变体）以及原始 MGSM 与 AfriMGSM 作为对照数据集。

**📈 对比分析**

对 12 个模型在原始、名字变体、数字变体、无关上下文变体下分别进行 5 次采样，报告平均准确率；结果显示：大模型在原始数据上性能最好，但在数字变体下下降显著，尤其是低资源语言；Claude Sonnet 4 在多实例评估中排名上升，Gemini 2.5 Flash 在单实例上最佳但多实例下降。

**⚠️ 局限性**

仅覆盖 9 种语言，缺乏更多语言的本地注释者；只评估 12 个模型，未包含如 Qwen3 等新模型；提示语言仅为英语；仅使用 MGSM 的 225 题，剩余 25 题因难以生成多实例被排除；受计算资源限制影响评估规模受限。

---

## 124. Bidirectional Cross-Perception for Open-Vocabulary Semantic Segmentation in Remote Sensing Imagery

**arXiv ID:** 2601.21159 | [PDF](https://arxiv.org/pdf/2601.21159v1)

**作者:** Jianzheng Wang `[一作]` (Nanjing University of Information Science and Technology), Huan Ni `[通讯]` (Nanjing University of Information Science and Technology)

**通讯引用:** 356 | [OpenAlex ID](https://openalex.org/A5056180462)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种训练-free的双分支协同推理框架SDCI，用于遥感图像的开放词汇语义分割。

**💡 创新点**

通过双向跨模型注意力融合（CAF）、双向图扩散细化（BCDR）以及基于凸优化的超像素协同预测（CSCP）实现了跨模型深度融合和低级几何约束，显著提升了边界精度与语义一致性。

**🔧 技术方法**

结合CLIP的文本语义与DINO的视觉结构，采用跨模型注意力注入、随机游走图扩散、超像素分割与凸优化求解等技术。

**📊 数据集**

在GID、Potsdam、Vaihingen、LoveDA、iSAID等遥感语义分割基准上进行评估。

**📈 对比分析**

与MaskCLIP、CLIPer、CASS等现有训练-free方法对比，SDCI在多数据集上均取得最高或最接近最高的mIoU（如GID 59.59%，Potsdam 66.40%），且计算量显著低于高精度基准。

**⚠️ 局限性**

与全监督方法仍存在性能差距，并且对文本提示的质量仍较敏感，特别是专业词汇需要进一步适配。

---

## 125. How does information access affect LLM monitors' ability to detect sabotage?

**arXiv ID:** 2601.21112 | [PDF](https://arxiv.org/pdf/2601.21112v1)

**作者:** Rauno Arike `[一作]` (Aether Research), Francis Rhys Ward `[通讯]` (Independent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM代理监控信息访问对检测误行为的影响，并提出了提取‑评估（EaE）分层监控方法。

**💡 创新点**

发现了“less‑is‑more”效应，即给监控更多信息并不总是更好，并提出利用信息过滤提升监控性能。

**🔧 技术方法**

对三种控制环境进行实验，比较 action‑only、CoT‑only、CoT+action、EaE 四种监控协议，并采用 log‑AUROC 等指标评估。

**📊 数据集**

使用 SHADE‑Arena、MLE‑Sabotage 与 BigCodeBench‑Sabotage 三个公开评估环境中的攻击者轨迹。

**📈 对比分析**

通过 log‑AUROC、TPR@1%FPR 等指标比较各协议，EaE 在 BigCodeBench 等场景提升约 17‑20% 的检测率，且成本更低。

**⚠️ 局限性**

实验受限于已有轨迹、模型能力差距、监控对齐等，且对同步监控和更长轨迹的处理不足。

---

## 126. SW-ASR: A Context-Aware Hybrid ASR Pipeline for Robust Single Word Speech Recognition

**arXiv ID:** 2601.20890 | [PDF](https://arxiv.org/pdf/2601.20890v1)

**作者:** Manali Sharma `[一作]` (Tetranetics Private Limited), Buvaneshwari G `[通讯]` (Tetranetics Private Limited)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种针对单词级自动语音识别（SW‑ASR）的模块化框架，能够在低资源、噪声多、语音质量差的实时电话和社交媒体环境中准确识别单词并触发相应动作。

**💡 创新点**

创新点在于将Whisper与Vosk混合前端与多种后处理匹配策略（余弦相似度、Levenshtein距离、LLM匹配以及上下文引导匹配）相结合，并通过上下文与少量示例提示显著降低词误差率，同时保持低延迟。

**🔧 技术方法**

技术包括：Whisper+Vosk混合ASR、音频去噪与音量归一化、词向量余弦相似度、编辑距离、LLM（如Llama‑4‑Scout）语义/音韵匹配、上下文嵌入、少量示例提示、SIP‑兼容的电话栈集成。

**📊 数据集**

使用Google Speech Commands（65k一秒录音）与自建的跨平台（WhatsApp、WeChat、Facebook Messenger、蜂窝电话）30词集成数据，共1200条语音样本。

**📈 对比分析**

在GSC与平台数据上对比多种管线：Hybrid、CS、LS、LLM、CS+C、LLM+C、LLM+C+FS。结果显示：Hybrid在高质量GSC上表现最好；在噪声/压缩环境下，CS+C和LLM+C+FS能将WER显著降低，准确率提升约10–20%，并保持与Hybrid相近的平均延迟。

**⚠️ 局限性**

局限性包括：对平台特定编码/压缩的依赖需进一步泛化；多语言/方言覆盖不足；主要针对单说话人，混合/多人语音效果下降；极端噪声、重叠语音或新方言场景仍表现欠佳。

---

## 127. Cramér-Rao Bound Analysis and Near-Optimal Performance of the Synchronous Nyquist-Folding Generalized Eigenvalue Method (SNGEM) for Sub-Nyquist Multi-Tone Parameter Estimation

**arXiv ID:** 2601.20866 | [PDF](https://arxiv.org/pdf/2601.20866v1)

**作者:** Huiguang Zhang `[一作]` `[通讯]`, Huiguang Zhang

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

推导双通道幅值比参数的CRB并通过Monte-Carlo仿真验证SNGEM在极端亚奈奎斯特采样下的频率/幅度/相位估计精度，同时与OMP基压缩感知方法对比。

**💡 创新点**

证明通过原始信号及其时间导数的幅值比估计频率仅损失3 dB，展示SNGEM在亚奈奎斯特压缩率>10×时的统计最优性，并提供确定性广义特征值解法。

**🔧 技术方法**

同步奈奎斯特折叠+时域导数通道、广义特征值分解、CRB理论分析、Monte-Carlo仿真。

**📊 数据集**

合成多音阶实数正弦波，随机相位，压缩率8–20×，SNR范围-10到50 dB。

**📈 对比分析**

将SNGEM与基于OMP的压缩感知在相同采样和噪声条件下比较；SNGEM在噪声自由下实现机器精度，在有噪声时距CRB 1.1–1.3倍，而OMP出现不可消除的误差底限。

**⚠️ 局限性**

仅针对单正弦波或少量正弦波的理想导数模型；假设理想差分器、AWGN；未验证在真实硬件、多径或非高斯噪声环境中的鲁棒性。

---

## 128. Optimal Transport-Induced Samples against Out-of-Distribution Overconfidence

**arXiv ID:** 2601.21320 | [PDF](https://arxiv.org/pdf/2601.21320v1)

**作者:** Keke Tang `[一作]` (Guangzhou University), Zhihong Tian `[通讯]` (Guangzhou University)

**通讯引用:** 10733 | [OpenAlex ID](https://openalex.org/A5056608045)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过利用半离散最优传输中的奇异边界构造结构性模糊样本（OTIS），在训练阶段对这些样本施加置信度抑制，从而有效降低深度网络在离分布输入上的过度自信；

**💡 创新点**

创新点在于首次把最优传输奇异性与OOV过度自信关联起来，并基于该几何结构无监督地生成逼近分布边缘的模糊样本；

**🔧 技术方法**

采用自动编码器压缩特征空间、半离散最优传输求解、置信度抑制损失与交叉熵联合训练；

**📊 数据集**

实验使用CIFAR‑10/100、SVHN、MNIST、FMNIST、ImageNet等主流数据集，并在多种ID/OOD组合（如LSUN_CR、Textures_C、Uniform、Noise、Adversarial Noise/Samples）上评估；

**📈 对比分析**

与CEDA、ACET、CODES、VOS、OE、CCUd等基线比较，OTIS在保持ID准确率的同时显著降低OOV最大置信度（OOV MMC）和FPR95，在大多数场景下实现了最优或次优的整体性能；

**⚠️ 局限性**

局限性包括额外的编码器训练与OT解算开销，且方法对基准分布和自动编码器深度有一定敏感性，未能完全覆盖所有可能的结构性模糊区域。

---

## 129. Gaussian Belief Propagation Network for Depth Completion

**arXiv ID:** 2601.21291 | [PDF](https://arxiv.org/pdf/2601.21291v1)

**作者:** Jie Tang `[一作]` (National University of Defense Technology), Ping Tan `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 13940 | [OpenAlex ID](https://openalex.org/A5084953118)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于高斯置信传播网络（GBPN）的稀疏深度图完成方法，利用场景特定的马尔可夫随机场（MRF）实现端到端稠密深度预测。

**💡 创新点**

创新点在于动态构建场景特定MRF并通过图形模型构造网络（GMCN）学习非局部边和参数，同时结合串行+并行高斯置信传播实现高效全局推理。

**🔧 技术方法**

核心技术包括U‑Net加稀疏注意力的图模型构造网络、可学习的非局部边、加速的串行与并行消息传递、以及基于高斯分布的损失与精度监督。

**📊 数据集**

使用了NYUv2、KITTI和VOID三个主流深度完成基准数据集进行实验。

**📈 对比分析**

在NYUv2和KITTI上均取得SOTA性能，且在不同稀疏度、稀疏模式和跨数据集的零样本测试中表现出更高的鲁棒性和泛化能力。

**⚠️ 局限性**

主要限制在于推理时的计算开销较大、对迭代次数和超参数敏感、缺乏严格的收敛理论，并且仅在标准训练集上训练，未充分利用大规模预训练模型。

---

## 130. Scaling Embeddings Outperforms Scaling Experts in Language Models

**arXiv ID:** 2601.21204 | [PDF](https://arxiv.org/pdf/2601.21204v1)

**作者:** Hong Liu `[一作]` (Meituan), Xunliang Cai `[通讯]` (Meituan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究将嵌入层的稀疏参数扩展作为与 Mixture‑of‑Experts（MoE）不同的稀疏扩展维度，并在大规模模型中实现了高效稀疏扩展。

**💡 创新点**

创新点在于提出了 N‑gram 嵌入（NE）和 Per‑Layer N‑gram Embedding（PLNE），并系统性地比较了嵌入扩展与专家扩展的 Pareto 前沿，给出了最佳参数分配、词表大小选择等经验法则。

**🔧 技术方法**

技术包括基于哈希的多层 N‑gram 嵌入、Embedding Amplification、speculative decoding、N‑gram Cache、同步内核以及 CUDA 核融合等系统优化。

**📊 数据集**

数据集涵盖 300B 预训练语料、11T 预训练+1.5T 中训练语料以及 SFT 数据，并在 MMLU、C‑Eval、BBH、GSM8K、HumanEval+、SWE‑Bench 等多种基准上进行评测。

**📈 对比分析**

与参数等价的 MoE 基线相比，68.5B 参数、≈3B 激活参数的 LongCat‑Flash‑Lite 在大部分基准（agentic tool use、coding、general、math）上获得了 1–5% 的性能提升，同时保持了更低的激活参数量和更高的推理速度。

**⚠️ 局限性**

局限在于对嵌入扩展的优势仍依赖于稀疏度与模型宽度的平衡，深度过大会削弱效果；哈希碰撞仍是关键瓶颈，且对超大词表的实际部署尚未彻底验证。

---

## 131. Temporal Context and Architecture: A Benchmark for Naturalistic EEG Decoding

**arXiv ID:** 2601.21215 | [PDF](https://arxiv.org/pdf/2601.21215v1)

**作者:** Mehmet Ergezer `[一作]` (Wentworth Institute of Technology), Mehmet Ergezer `[通讯]` (Wentworth Institute of Technology)

**通讯引用:** 1018 | [OpenAlex ID](https://openalex.org/A5087094884)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对自然场景EEG解码任务，系统比较了CNN、LSTM、Transformer、S4和S5在不同时间窗口长度下的性能。

**💡 创新点**

首次量化了状态空间模型S5与传统CNN在长时序EEG上的效率‑准确率平衡，并提出了稳健Transformer与S5的对照实验。

**🔧 技术方法**

使用卷积网络、双向LSTM、Transformer、S4与S5等序列模型，并结合标准化、批归一化、EMA等训练技巧。

**📊 数据集**

采用HBN健康大脑网络中的电影观看EEG数据，共40名受试者、约1.5万段。

**📈 对比分析**

通过多种评测（上下文长度、零样本跨频、离散任务、留一受试者），S5在64s段可达98.7%准确率，参数量仅为CNN的1/20，但在跨频和OOD任务上表现出过度自信且鲁棒性不足。

**⚠️ 局限性**

仅在HBN数据上验证，短段单种子评测、对S5的置信度不足等问题，以及缺乏对其他公共基准的进一步验证。

---

## 132. Planner-Auditor Twin: Agentic Discharge Planning with FHIR-Based LLM Planning, Guideline Recall, Optional Caching and Self-Improvement

**arXiv ID:** 2601.21113 | [PDF](https://arxiv.org/pdf/2601.21113v1)

**作者:** Kaiyuan Wu `[一作]` (Duke University), Rishikesan Kamaleswaran `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并评估了 Planner–Auditor 框架，用于安全、可靠的临床出院计划生成。

**💡 创新点**

创新点在于将 LLM 规划器与确定性审核器分离，并通过自我改进循环与差异缓冲实现无训练模型的安全迭代。

**🔧 技术方法**

采用了检索增强生成（RAG）、LLM（GPT‑4o‑mini）、确定性规则检查、上下文缓存和跨 episode 差异回放等技术。

**📊 数据集**

使用 MIMIC‑IV‑on‑FHIR 数据集进行回溯模拟评估。

**📈 对比分析**

通过对比基线、缓存、SI、缓存+SI 和缓冲回放五种配置，评估多任务覆盖率、Brier 分数、ECE、误差率和延迟；SI 与缓存+SI 将完整覆盖率从 32% 提升至 86%，并显著改善校准。

**⚠️ 局限性**

局限性包括仅在模拟数据上验证，样本量有限；未检验计划内容的临床准确性和医生反馈；可能仍有未检测到的安全风险。

---

## 133. Rethinking Federated Graph Foundation Models: A Graph-Language Alignment-based Approach

**arXiv ID:** 2601.21369 | [PDF](https://arxiv.org/pdf/2601.21369v1)

**作者:** Yinlin Zhu `[一作]` (Sun Yat-sen University), Guocong Quan `[通讯]` (Sun Yat-sen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了 FedGALA 框架，在联邦图基础模型中实现了图与语言的连续对齐，并通过基于 prompt 的微调实现任务适配。

**💡 创新点**

创新点包括：① 用无监督对比学习在连续空间对齐图与冻结 PLM，消除 VQ 量化带来的知识损失；② 将结构与语义编码器拆分，提升跨域泛化；③ 引入历史匹配和全局 prompt 池，稳定结构知识并实现任务专化；④ 只传输结构编码器参数，显著降低通信开销。

**🔧 技术方法**

技术手段包括：对比学习、无监督自监督预训练、图神经网络（GraphSAGE/GIN 等）、预训练语言模型（Sentence‑BERT）、prompt tuning、历史匹配、分组聚合、结构编码器分离。

**📊 数据集**

实验使用 8 个基准图数据集：Cora、PubMed、OGB‑arxiv、WikiCS、FB15K237、WN18RR、PCBA 与 HIV，涵盖节点分类、边分类和图分类三大任务。

**📈 对比分析**

与 22 个基线（传统 FL/FGL、FedGFMs、GFMs 适配等）对比，FedGALA 在所有任务上均取得最高分，最高提升达 14.37%；在 2‑shot 学习下同样领先；收敛速度最快、通信成本最低。

**⚠️ 局限性**

局限性包括：仅在三类任务和 8 个数据集上验证；对极端数据不平衡、隐私攻击及更大规模部署的鲁棒性未深入探究；模型规模与资源开销仍受限于预训练语言模型。

---

## 134. Model-Free Neural State Estimation in Nonlinear Dynamical Systems: A Comparative Study of Neural Architectures and Classical Filters

**arXiv ID:** 2601.21266 | [PDF](https://arxiv.org/pdf/2601.21266v1)

**作者:** Zhuochen Liu `[一作]` (University of Southern California), Rahul Jain `[通讯]` (University of Southern California)

**通讯引用:** 4562 | [OpenAlex ID](https://openalex.org/A5002082998)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在非线性动力学系统中，对比模型无关的神经状态估计器与传统贝叶斯滤波器，评估它们在多种非线性场景下的长期状态估计性能；

**💡 创新点**

提供统一实验框架，系统性比较多种神经架构（Transformer、GRU、LSTM、Mamba等）与经典滤波（EKF、UKF、EnKF、PF）在无系统知识条件下的表现，并验证神经网络能逼近强基于模型滤波器的准确性，同时具备更高推理吞吐量；

**🔧 技术方法**

使用Transformer、GRU、LSTM、Mamba/Mamba‑2等序列模型，以及EKF、UKF、EnKF、PF等经典滤波算法；

**📊 数据集**

在五个典型非线性任务上收集训练与评估数据：弹道再入、单传感器方位跟踪、Lorenz‑96、N‑链摆和二维四旋翼；每个任务使用多组训练/测试轨迹；

**📈 对比分析**

对比RMSE、MAE、MedAE、NRMSE、AUC等长期误差指标和迭代速度。结果显示Mamba/Mamba‑2与EKF/UKF相近，且神经模型显著快于经典滤波；在大多数场景下，神经估计器在无模型信息情况下也能获得高精度；

**⚠️ 局限性**

受限于需要大量标注数据（约20:1的时间步与参数比），未评估不确定性校准或后验质量，且模型对架构与实验设置敏感，需进一步研究数据效率、先验结构与更广泛系统的鲁棒性。

---

## 135. Noninvasive Intracranial Pressure Estimation Using Subspace System Identification and Bespoke Machine Learning Algorithms: A Learning-to-Rank Approach

**arXiv ID:** 2601.20916 | [PDF](https://arxiv.org/pdf/2601.20916v1)

**作者:** Anni Zhao `[一作]` (Cranberry-Lemon University), Xiao Hu `[通讯]` (Emory University)

**通讯引用:** 202977 | [OpenAlex ID](https://openalex.org/A5050750924)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

开发了一种基于系统识别与排名约束的机器学习框架，用非侵入性信号（ABP、CBv、R‑R间隔）实现平均颅内压的估计。

**💡 创新点**

创新点在于引入针对每个线性动态模型的映射函数学习并加入排名约束，使模型选择与误差排名保持一致，从而提升估计精度。

**🔧 技术方法**

使用了子空间系统识别、线性/非线性映射函数学习（约束最小二乘、核化方法）、排名约束优化、MOCAIP波形特征提取等技术。

**📊 数据集**

采用来自六家机构共156名脑损伤患者的多模态监测数据库，包括ABP、CBv、ECG和侵入性ICP信号。

**📈 对比分析**

与传统无约束线性映射和其他机器学习方法相比，约束后线性模型和高斯核非线性模型在测试集上误差小于6mmHg的占比分别提升至69%（线性约束）和68%（高斯核），平均误差下降约1–2mmHg。

**⚠️ 局限性**

局限性在于对高颅压（>20mmHg）的低估，训练数据中高压样本稀缺；同时TCD信号质量与探头放置有关，需进一步优化特征提取和模型泛化能力。

---

## 136. What Hard Tokens Reveal: Exploiting Low-confidence Tokens for Membership Inference Attacks against Large Language Models

**arXiv ID:** 2601.20885 | [PDF](https://arxiv.org/pdf/2601.20885v1)

**作者:** Md Tasnim Jawad `[一作]` (Florida International University), Yanzhao Wu `[通讯]` (Florida International University)

**通讯引用:** 1447 | [OpenAlex ID](https://openalex.org/A5060093535)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种利用低置信度（hard）token的概率提升进行LLM成员推断攻击的框架

**💡 创新点**

通过跨模型比较（目标模型与预训练参考模型）聚焦低置信度token，显著提升成员信号的分离度

**🔧 技术方法**

token级概率提取、适应性token选择、概率差异聚合、DP‑SGD防御

**📊 数据集**

医学域的Augmented Clinical Notes、Asclepius；通用域的IMDB和Wikipedia

**📈 对比分析**

与七种基线方法（Loss、Lowercase、Min‑K++、PAC、Ratio、Zlib、SPV‑MIA）对比，平均提升AUC约5–10%（部分数据集超过10%），在低FPR阈值下表现尤为突出

**⚠️ 局限性**

对短文本或强成员信号的数据集（如IMDB）时效果与Ratio相近，且攻击仍受模型容量和训练规模限制，且DP‑SGD会显著降低攻击效果，需权衡隐私与性能

---

## 137. Deletion-correcting codes for an adversarial nanopore channel

**arXiv ID:** 2601.21236 | [PDF](https://arxiv.org/pdf/2601.21236v1)

**作者:** Huiling Xie `[一作]`, Zitan Chen `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出了一种针对抗性纳米孔通道（最多可删掉 t 个 ℓ‑mer 的通道）的删除纠错码构造与理论分析。

**💡 创新点**

创新点在于：①将删除纠错问题转化为 Hamming 型误差问题，利用最大周期子串特性；②给出长度 n 的 q‑ary 码在此通道上实现 2tlog_q n+Θ(loglog n) 的冗余；③提供了与存在上界相匹配的上界与下界。

**🔧 技术方法**

主要技术包括：周期性与一致性模式分析、Fine‑Wilf 定理、Generalized Reed‑Solomon 码的校验矩阵、概率方法与 Janson 不等式。

**📊 数据集**

本文未使用实验数据集，而是给出纯理论的编码构造与冗余上界、下界。

**📈 对比分析**

通过理论比较，构造码的冗余与存在上界相同阶（即 2tlog_q n 的首项），且相对于已知的 4t(1+ε)log_q n 代码在此通道上实现了更低的冗余。

**⚠️ 局限性**

限制在于：①构造适用于 t≤min{ℓ−2,(ℓ+1)/2} 的情形；②仅考虑删除错误，未涵盖重复与替换错误；③未给出实际编码实现或实验验证。

---

## 138. DevOps-Gym: Benchmarking AI Agents in Software DevOps Cycle

**arXiv ID:** 2601.20882 | [PDF](https://arxiv.org/pdf/2601.20882v1)

**作者:** Yuheng Tang `[一作]` (University of California Santa Barbara), Wenbo Guo `[通讯]` (University of California Santa Barbara)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个覆盖完整 DevOps 生命周期（构建配置、监控、问题修复、测试生成）的端到端 benchmark，命名为 DevOps‑Gym；

**💡 创新点**

创新点在于：①首次将真实 Java 与 Go 项目的多阶段 DevOps 任务集中于一套 benchmark；②设计了半自动化任务收集与严格去污染流程；③提供了标准化的工具调用接口与动态执行环境；

**🔧 技术方法**

使用技术包括：LLM + agentic 框架（OpenHands、mini‑SWE‑agent、Claude‑Code），vLLM、Docker 容器化、终端工具调用接口（如 Maven、Gradle、go工具、监控命令），以及自动化评估脚本；

**📊 数据集**

数据集为 700+ 真实任务（30+ Java/Go 项目）以及 14 条端到端流水线任务，涵盖构建错误、监控异常、缺陷修复与测试生成；

**📈 对比分析**

通过对三大 agent 框架和多种 LLM 进行基准测试，发现即使是 Claude‑4‑Sonnet 也只能在构建配置上达到 51.85% 的成功率，在监控、问题修复和测试生成上分别只有 20.56%、23.87% 与 13.87%；整体端到端任务成功率为 0%；

**⚠️ 局限性**

局限性在于：①对监控和构建配置任务的理解与工具使用仍远低于预期；②缺乏跨语言（Python 与 Java/Go）的通用能力；③任务重现与评估依赖大量人工工程，难以大规模扩展；④评估聚焦单一语言环境，未覆盖更广泛的 DevOps 场景。

---

## 139. Flow Perturbation++: Multi-Step Unbiased Jacobian Estimation for High-Dimensional Boltzmann Sampling

**arXiv ID:** 2601.21177 | [PDF](https://arxiv.org/pdf/2601.21177v1)

**作者:** Xin Peng `[一作]` (Beijing University of Posts and Telecommunications), Ang Gao `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 5064 | [OpenAlex ID](https://openalex.org/A5002156038)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出了Flow Perturbation++（FP++）方法，实现多步无偏雅可比估计，以提高高维Boltzmann采样的可扩展性。

**💡 创新点**

通过将全局雅可比分解为多步局部雅可比，并在每步使用独立扰动，显著降低估计方差，同时保持无偏性。

**🔧 技术方法**

结合连续归一化流（CNF）、概率流ODE、Sequential Monte Carlo（SMC）以及有限差分/向量-雅可比乘积技术实现FP++。

**📊 数据集**

在1000维高斯混合模型和全原子Chignolin蛋白质变体上进行验证。

**📈 对比分析**

与单步FP、Hutchinson估计和精确Jacobian相比，FP++在保持相同计算成本下取得更低的估计方差、提高了采样精度，并在高维场景中显著优于Hutchinson。

**⚠️ 局限性**

FP++仍依赖于多步积分的数值稳定性，且在极高维或复杂动力学中对积分步长和扰动采样的敏感性需要进一步研究。

---

## 140. Textual Equilibrium Propagation for Deep Compound AI Systems

**arXiv ID:** 2601.21064 | [PDF](https://arxiv.org/pdf/2601.21064v1)

**作者:** Minghui Chen `[一作]` (Nanyang Technological University), Xiaoxiao Li `[通讯]` (Vector Institute)

**通讯引用:** 5099 | [OpenAlex ID](https://openalex.org/A5100458648)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于局部均衡传播的文本梯度优化方法 TEPr，解决深层复合 AI 系统中的梯度爆炸和消失问题。

**💡 创新点**

创新点在于把能量模型的均衡传播思想应用于文本梯度，采用自由阶段本地优化和逼近阶段有界编辑的两相策略，避免了全局文本反向传播的规模与信号衰减。

**🔧 技术方法**

利用大型语言模型作为本地评审和更新器，结合局部反馈、可控文本编辑、前向任务目标传递。

**📊 数据集**

在 PubMedQA、STARK‑PRIME、HotpotQA、BigCodeBench 等多步推理与工具使用基准上进行实验。

**📈 对比分析**

与 CoT、DSPy、HBC、TextGrad（含压缩版本）比较，TEP 在所有四个任务均获得最高准确/MRR/F1/Pass‑Rate，尤其在深层工作流中优势显著。

**⚠️ 局限性**

局限在于仅在黑盒提示优化框架下验证，未探讨动态图结构、参数微调等情况，且对极大规模上下文仍需进一步评估。

---

## 141. OpenSec: Measuring Incident Response Agent Calibration Under Adversarial Evidence

**arXiv ID:** 2601.21083 | [PDF](https://arxiv.org/pdf/2601.21083v1)

**作者:** Jarrod Barnes `[一作]` `[通讯]` (Arc Intelligence), Jarrod Barnes (Arc Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了 OpenSec 双控制 RL 环境，用于评估安全事件响应（IR）代理在面临对抗式注入证据时的执行与校准性能。

**💡 创新点**

创新点包括：① 通过执行为基础的指标（如时间到首次封锁、爆炸半径和注入违规率）直接测量校准；② 引入 taxonomy‑stratified 场景与信任层级，让环境更贴合 SOC 实际；③ 明确区分模型的能力与其校准表现，揭示传统基准掩盖的过度触发问题。

**🔧 技术方法**

使用技术包括：对抗强化学习、Dec‑POMDP 模型、LLM 代理（GPT‑5.2、Gemini 3、DeepSeek 3.2、Claude Sonnet 4.5）、工具调用、基于执行的奖励函数与对抗注入脚本。

**📊 数据集**

数据集为 160 条训练种子和 40 条评估种子（standard‑tier），每条种子覆盖三类情景（direct_harm、data_exfil、adaptive）并嵌入提示注入 payload，包含信任层级与 provenance 信息。

**📈 对比分析**

对四大前沿模型在 40 条 episode 上进行 JSONL 输出评估，指标包括：containment 率、false‑positive 率、正确 containment 率、注入违规率和 blast radius。结果显示 GPT‑5.2、Gemini 3、DeepSeek 触发 100% 事件且 FP 率高达 90–97%；Sonnet 4.5 仅部分校准（containment 85%，FP 72%）。

**⚠️ 局限性**

局限性包括：仅模拟日志级 IR，未执行真实攻击；攻击者受限于预定义状态机；种子数量有限，统计置信度受限；信任层级虽存在但未用于评价；奖励对错误行动惩罚严重但未惩罚不行动，导致过度触发的高奖励掩盖真实失败。

---

## 142. Is Parameter Isolation Better for Prompt-Based Continual Learning?

**arXiv ID:** 2601.20894 | [PDF](https://arxiv.org/pdf/2601.20894v1)

**作者:** Jiangyang Li `[一作]` (Xi’an Jiaotong University), Yihong Gong `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种全局提示池与稀疏动态路由的共享提示框架，用以在连续学习中高效利用参数并促进任务间知识迁移。

**💡 创新点**

创新点在于引入任务感知门控机制实现输入自适应的提示选择，以及历史感知调制器通过累计激活统计对提示的使用频率进行惩罚与梯度缩放，以平衡前向学习与反向记忆。

**🔧 技术方法**

核心技术包括Mixture-of-Experts（MoE）提示池、轻量级任务门控路由、累计激活统计的路由惩罚、梯度缩放调制以及在Transformer注意力层中注入自适应提示。

**📊 数据集**

在多种公开基准上验证：Split CIFAR‑100、Split ImageNet‑R、Split CUB‑200、5‑Datasets、CORe50 以及长序列与50任务的增量设置。

**📈 对比分析**

与 L2P、DualPrompt、CODA‑Prompt、HiDe‑Prompt、NoRGa 等现有提示式连续学习方法比较，本文在 FAA、CAA 方面均取得最高分，并在忘记率（FM）上实现显著下降，展现出更优的性能与稳定性。

**⚠️ 局限性**

局限性在于实验仅使用 ViT‑B/16 等中等规模骨干网络，缺乏对更大模型或更高计算资源条件下的可扩展性验证。

---

## 143. TwinWeaver: An LLM-Based Foundation Model Framework for Pan-Cancer Digital Twins

**arXiv ID:** 2601.20906 | [PDF](https://arxiv.org/pdf/2601.20906v1)

**作者:** Nikita Makarov `[一作]` (Roche), Michael Menden `[通讯]` (University of Melbourne)

**通讯引用:** 6564 | [OpenAlex ID](https://openalex.org/A5055060401)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了TwinWeaver框架和Genie Digital Twin (GDT)，将多模态临床时间序列序列化为文本并利用大语言模型进行联合预测，实现全癌种患者的数字孪生。

**💡 创新点**

创新点包括：①将患者历史转化为LLM可读文本，突破固定词表限制；②同一模型同时完成连续指标预测和事件预测，提升信息利用效率；③提供可解释的推理链；④在零样本和少量样本的临床试验中实现强泛化。

**🔧 技术方法**

使用技术包括：大语言模型Llama 3.1 8B Instruct的微调；文本序列化与多任务训练（时间序列预测+landmark事件预测）；基于对数似然的风险评分；知识蒸馏与强化学习生成解释性推理链。

**📊 数据集**

使用的数据集为Flatiron Health–Foundation Medicine Clinico‑Genomic Database（FH‑FMI CGDB）中的93,054例多癌种患者，以及POPLAR、IMpower130等外部临床试验数据。

**📈 对比分析**

与TiDE、Chronos、RSF、CLMBR‑T等基线比较，GDT在真实世界数据上MASE降至0.87（vs 0.97）、C‑index升至0.703（vs 0.662）；在临床试验零样本下MASE 0.75–0.88、C‑index 0.672（vs 0.648）。

**⚠️ 局限性**

局限性包括：对低数据可用指标（如转移）表现欠佳；需要大量预训练与微调资源；仍存hallucination与偏差风险；事件预测未保证累计发生率单调性。

---

## 144. Just Ask: Curious Code Agents Reveal System Prompts in Frontier LLMs

**arXiv ID:** 2601.21233 | [PDF](https://arxiv.org/pdf/2601.21233v1)

**作者:** Xiang Zheng `[一作]` (City University of Hong Kong), Cong Wang `[通讯]` (City University of Hong Kong)

**通讯引用:** 25399 | [OpenAlex ID](https://openalex.org/A5100390514)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自进化的系统提示提取框架，利用多轮交互和无监督技能发现，能够在不依赖标签数据或手工提示的情况下，从黑盒LLM代理中自动恢复系统提示。

**💡 创新点**

创新点在于：1）将技能空间分层并通过Upper Confidence Bound进行探索，实现自适应策略发现；2）采用一致性验证（自我一致性与跨技能一致性）来衡量提取质量；3）展示系统提示是现代代理安全的新型攻击面，并对其进行大规模实证分析。

**🔧 技术方法**

核心技术包括：UCB算法用于技能排名与探索；交互式推理与技能生成；多轮对话执行策略；一致性验证（基于嵌入相似度）作为奖励信号；无监督技能发现构建层级技能集。

**📊 数据集**

使用41个公开可通过OpenRouter访问的商业模型（含闭源、开源和微调模型）作为实验对象；对GroK公开提示和Claude Code逆向提取结果进行验证；未使用任何外部标注数据集。

**📈 对比分析**

通过一致性得分≥0.7判定成功，41个模型全部成功（平均约2.8轮）。与单轮攻击相比提升约40%；在4个受控模型的实验中，嵌入攻击词典的防御可将提取质量降低18.4%，而简单的“勿透露”防御效果极差。

**⚠️ 局限性**

局限性包括：仅覆盖OpenRouter可访问的模型；系统提示可能随时间更新；大多数提取为语义重构而非逐字；受控实验仅涵盖4个模型；缺乏模型内部梯度或权重信息，无法验证更深层防御效果。

---

## 145. Scaling Reasoning Hop Exposes Weaknesses: Demystifying and Improving Hop Generalization in Large Language Models

**arXiv ID:** 2601.21214 | [PDF](https://arxiv.org/pdf/2601.21214v1)

**作者:** Zhaoyi Li `[一作]` (University of Science and Technology of China), Ying Wei `[通讯]` (Zhejiang University)

**通讯引用:** 18890 | [OpenAlex ID](https://openalex.org/A5075037129)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在链式推理跳数泛化中的错误机制，并提出了一种通过在推理过程中动态识别并关闭错误处理头的测试时纠错方法。

**💡 创新点**

创新点在于发现错误集中在少数关键token位置，揭示了正确与错误推理轨迹的竞争机制，并通过定位并抑制错误处理头实现显著的推理性能提升。

**🔧 技术方法**

使用的技术包括注意力头拦截（head knockout）、Logit Lens、熵检测、以及一个基于小模型的head选择器网络。

**📊 数据集**

使用了七个跨领域的推理跳数泛化任务：Parity‑NL、LLC、MDM、MOAS、CLF、ObjC、NumS，评估四个开源LLM（Qwen2.5‑7B、Phi‑3、LLaMA3‑8B、Qwen3‑8B）。

**📈 对比分析**

与原始模型和DoLa等基线对比，TCR平均提升5–7%准确率，在某些任务上提升近20%（Oracle版提升约20%），表现稳健且可迁移。

**⚠️ 局限性**

局限在于需先定位错误头并训练head选择器，检测阈值与头集合对不同模型的通用性有限，未能解决更大上下文长度或更复杂推理的泛化问题。

---

## 146. TIDE: Tuning-Integrated Dynamic Evolution for LLM-Based Automated Heuristic Design

**arXiv ID:** 2601.21239 | [PDF](https://arxiv.org/pdf/2601.21239v1)

**作者:** Chentong Chen `[一作]` (Xi'an Jiaotong University), Jianyong Sun `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 15727 | [OpenAlex ID](https://openalex.org/A5100367671)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出TIDE框架，解耦算法结构与参数，提升LLM在自动启发式设计中的效果

**💡 创新点**

①嵌套岛屿结构与TSED度量实现结构多样性；②UCB调度提示策略动态选择；③差分突变精细调参，克服LLM数值盲点

**🔧 技术方法**

大型语言模型（如ChatGPT API）、树相似编辑距离（TSED）、APTED、UCB、多臂赌博机策略、差分进化、岛屿模型、AST解析、自然语言提示调度

**📊 数据集**

九种组合优化问题（TSP、KP、在线BPP、ASP、CVRP、OP、MKP、DPP、Mountain Car）以及对应的构造、改进、ACO、GA、RL框架

**📈 对比分析**

与手工启发式、神经组合优化模型（POMO、NeuOpt、DeepACO）以及其他LLM AHD方法（EoH、ReEvo、HSEvo、MCTS‑AHD）对比，TIDE在所有任务上均取得更低目标误差或更高成功率，并在搜索效率上节省 token/评估次数

**⚠️ 局限性**

仍受LLM生成语义噪声影响，迁移/提示策略依赖经验阈值，实验仅在单一LLM模型上验证，未探讨多目标或跨域迁移能力

---

## 147. ZipMoE: Efficient On-Device MoE Serving via Lossless Compression and Cache-Affinity Scheduling

**arXiv ID:** 2601.21198 | [PDF](https://arxiv.org/pdf/2601.21198v1)

**作者:** Yuchen Yang `[一作]` (Nanjing University), Zhi-Hua Zhou `[通讯]` (Nanjing University)

**通讯引用:** 61007 | [OpenAlex ID](https://openalex.org/A5100621138)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ZipMoE，一种面向移动和边缘设备的 Mixture‑of‑Experts 推理系统，利用无损压缩与缓存‑调度协同设计实现高效的 on‑device 推理。

**💡 创新点**

创新点在于：① 通过位域分解将 BF16 权重的指数位低熵部分进行无损压缩，显著降低 I/O 开销；② 设计分层压缩状态与动态缓存规划，使 CPU 多核可并行解压，完成从 I/O‑bound 到计算‑并行的转变；③ 证明调度算法在常数因子内逼近全局最优。

**🔧 技术方法**

技术主要包括：位域压缩（LZ4HC/ZSTD）、无损压缩解码、缓存‑亲和调度（基于 DAG 的任务分块）、GPU 并行张量恢复核、统一内存零拷贝和多线程并行解压。

**📊 数据集**

使用公开的 MoE 模型 DeepSeekV2‑Lite、Qwen1.5‑MoE（解码器）和 SwitchTransformers‑Large‑128（编码‑解码器），以及 ShareGPT 作为真实推理负载。

**📈 对比分析**

与 MoE‑Infinity、DeepSpeed ZeRO‑3、Accelerate 等主流系统在 Jetson AGX Orin（64 GB/32 GB）上对比，ZipMoE 在单词推理时间（TPOT）提升 62.65‑97.97%，首次推理时间（TTFT）提升 53.25‑87.90%，吞吐量提升 1.79‑42.49 倍，整体端到端延迟缩短 3.03‑42.49 倍。

**⚠️ 局限性**

局限性包括：依赖统一内存架构，跨平台可移植性有限；压缩与解压仍需 CPU 资源，极端低功耗设备可能受限；对极端高并发场景下的缓存失效策略尚未深入评估。

---

## 148. What Are Brands Telling You About Smishing? A Cross-Industry Evaluation of Customer Guidance

**arXiv ID:** 2601.20999 | [PDF](https://arxiv.org/pdf/2601.20999v1)

**作者:** Dev Vikesh Doshi `[一作]` (California State University San Marcos), Muhammad Lutfor Rahman `[通讯]` (California State University San Marcos)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估了149个美国知名品牌在其官网上提供的短信钓鱼（smishing）防护与报告指导内容，发现大多数品牌缺乏完整的定义、示例、视频教程及后续处理建议。

**💡 创新点**

首次对跨行业品牌的smishing教育实践进行大规模定性与定量内容分析，揭示行业内信息不一致、缺失与误导性建议，为行业标准化与政策制定提供基准。

**🔧 技术方法**

采用混合方法：使用ATLAS.ti进行开放编码与主题归纳，结合二进制计数编码统计信息出现频率；同时手工验证与迭代修订。

**📊 数据集**

数据集来源于Smishtank和Phishtank的前列被攻击品牌列表，筛选出149个美国本土或在美国频繁被攻击的品牌，并收集其官方网站上相关文本与多媒体内容。

**📈 对比分析**

通过对各项指标（定义、示例、视频、外部链接、报告步骤、受害后措施、预防建议）的频率统计与交叉比较，发现仅46%提供定义、35%提供示例、<1%提供视频、50%提供报告步骤、65%缺乏受害后指导；未采用传统机器学习或性能指标，而是以百分比展示行业普遍性和缺口。

**⚠️ 局限性**

局限性包括：仅覆盖美国品牌，未检视非公开或社交媒体教育；品牌信息可能更新频繁但未同步；样本在行业分布不均，导致跨行业比较受限；仅基于公开网页内容，忽略内部培训与客服渠道的防护措施。

---

## 149. Parametric Hyperbolic Conservation Laws: A Unified Framework for Conservation, Entropy Stability, and Hyperbolicity

**arXiv ID:** 2601.21080 | [PDF](https://arxiv.org/pdf/2601.21080v1)

**作者:** Lizuo Liu `[一作]` (Dartmouth), Anne Gelb `[通讯]` (Rice)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为 SymCLaw 的参数化双曲守恒律模型，可直接从数据中学习保守、熵稳定且双曲的系统。

**💡 创新点**

创新点在于将熵函数和熵势作为可学习的凸网络，同时以梯度神经网络参数化通量，保证通量雅可比矩阵的对称性和正定性，从而构造出严格双曲且熵稳定的方程。

**🔧 技术方法**

采用输入凸神经网络 (ICNN) 学习熵函数，使用全连接网络的梯度学习通量势，并配合 WENO5 重构、Rusanov 型熵稳定数值通量以及 TVD-RK3 时间积分，实现可微的端到端学习管线。

**📊 数据集**

使用多种经典双曲 PDE 的仿真数据集（Burgers、浅水、Euler、KPP 等），在训练阶段仅提供平滑解段，并在测试阶段检验在未见初始条件下的长期演化。

**📈 对比分析**

通过与现有的 ES-CFN、NESCFN 等基准方法对比，SymCLaw 在保持守恒、熵不等式和数值稳定性的同时，在长期预测、噪声鲁棒性和解的精度上均优于或相当于对手，尤其在高噪声情形下仍能准确捕捉冲击。

**⚠️ 局限性**

局限包括：对高维/复杂几何的适用性尚未验证；对边界条件的学习仍需进一步研究；当前使用固定时间步可能不满足 CFL 条件；对噪声敏感的熵函数学习仍需改进。

---

## 150. On Approximate Nash Equilibria in Mean Field Games

**arXiv ID:** 2601.20910 | [PDF](https://arxiv.org/pdf/2601.20910v1)

**作者:** Mao Fabrice Djete `[一作]` (Ecole Polytechnique Paris), Nizar Touzi `[通讯]` (New York University)

**通讯引用:** 8695 | [OpenAlex ID](https://openalex.org/A5105990961)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文研究了大规模对称博弈中，利用均值场游戏（MFG）的解构造近似纳什均衡，并在概率上证明了当玩家数趋向无穷时，单个玩家的最优单边偏离收益趋近于零。特别地，作者提出了新的 ℒ^∞ 范数下的近似纳什均衡概念，给出了在离散一周期模型和连续时间无控制扩散模型下的均值场近似结果。

**💡 创新点**

创新点主要有两点：①在传统的平均意义（ℒ^1）近似均衡基础上，首次给出了全局均匀（ℒ^∞）误差界；②提出了（ε,δ）-纳什均衡的新概念，用来描述在初始状态分布尾部对均衡误差的控制，能够在更宽松的条件下得到强统一近似结果。

**🔧 技术方法**

核心技术包括：
- 均值场游戏理论与贝尔曼–欧拉方程（或主方程）相结合；
- 传播无序性（propagation of chaos）和稳定性估计；
- 对偶测度、Wasserstein 距离与测度收敛的分析；
- 通过测度收敛与可测选取理论构造可行控制；
- 结合随机微积分和SDE稳定性推导统一误差上界。

**📊 数据集**

由于本文为理论研究，未使用任何具体实验数据集；所有结果均以概率论与偏微分方程的数学证明为基础。

**📈 对比分析**

比较方法：作者在有限玩家游戏中构造与均值场解对应的策略族，并对其单边偏离收益进行分析。性能表现为：
- 在ℒ^r（r>1）意义下，误差随玩家数 n→∞ 收敛到0；
- 在ℒ^∞ 统一意义下，对初始状态在扩张球 B(n^δ) 内的玩家，误差同样收敛到0；
- 进一步给出了（ε,δ）-均衡的收敛速率和统一误差界。

**⚠️ 局限性**

限制与假设：
- 仅考虑完全对称、同质的玩家结构；
- 需要在动力学、效用函数以及控制空间上满足较强的Lipschitz、线性增长和凸性假设；
- 对于非凸控制空间或非线性效用，结果尚未给出；
- 对于极端尾部分布或不满足有界性的初始分布，统一误差界可能失效；
- 结果主要在理论层面，实际算法实现与数值误差分析尚未展开。

---

## 151. Music Plagiarism Detection: Problem Formulation and a Segment-based Solution

**arXiv ID:** 2601.21260 | [PDF](https://arxiv.org/pdf/2601.21260v1)

**作者:** Seonghyeon Go `[一作]` (MIPPIA Inc), Yumin Kim `[通讯]` (MIPPIA Inc)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出音乐抄袭检测任务定义，并基于音乐段落转写构建检测管线。

**💡 创新点**

首次明确音乐抄袭检测与覆盖歌曲识别等任务的区别，并引入段落级转写与多模态孪生网络。

**🔧 技术方法**

使用Demucs源分离、结构分析、音素转写、SheetSage、Harmony Transformer等转写模块，以及MERT、CNN、跨注意力双编码器等深度学习相似度模型。

**📊 数据集**

构造SMP（72对真实抄袭案例并标注时间戳）和Covers80作为评估基准。

**📈 对比分析**

在段落级别采用Rec.1s@k指标，MERT在小规模数据上表现最好；在音乐级别采用mAP/MR1指标，效果低于主流Cover Song Identification模型，但能精确定位抄袭片段。

**⚠️ 局限性**

数据规模有限导致多模态模型训练不足，缺乏大规模真实抄袭数据，且仍未实现对完整曲目级别的高精度检索。

---

## 152. Efficient Simple Regret Algorithms for Stochastic Contextual Bandits

**arXiv ID:** 2601.21167 | [PDF](https://arxiv.org/pdf/2601.21167v1)

**作者:** Shuai Liu `[一作]`, Csaba Szepesvári `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了针对随机上下文逻辑斯蒂（线性）Bandits的简单调度算法，并给出了无指数常数κ的最优简单后悔上界。

**💡 创新点**

创新点在于：①将自协性分析与上下文线性Bandits思想结合，得到第一个κ无关的简单后悔上界；②设计了新的基于Thompson Sampling的随机化算法，仍保持κ无关；③将该框架推广到RLHF的Bradley‑Terry偏好学习。

**🔧 技术方法**

主要技术包括：自协性（self‑concordant）分析、置信集合构造、最大不确定性动作选择、正则化最大似然估计以及改进的TS采样。

**📊 数据集**

实验采用合成数据，在线性和逻辑斯蒂两种模型下验证算法性能。

**📈 对比分析**

与传统的均匀探索、累积后悔TS等基线相比，提出的算法在简单后悔上界上更快收敛，实验结果显示显著优于基线。

**⚠️ 局限性**

主要限制在于随机化算法在每一步需要构造置信集并求解线性优化，计算成本仍较高，且在大规模或结构化动作集上效率待进一步提升。

---

## 153. Optimization and Mobile Deployment for Anthropocene Neural Style Transfer

**arXiv ID:** 2601.21141 | [PDF](https://arxiv.org/pdf/2601.21141v1)

**作者:** Po-Hsun Chen `[一作]` (National Yang Ming Chiao Tung University), Ivan C. H. Liu `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 97 | [OpenAlex ID](https://openalex.org/A5074751823)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一款名为 AnthropoCam 的移动端神经风格迁移系统，专门将人类改造的景观（人类世环境）以视觉方式进行美学重塑，同时保持语义可辨识度。

**💡 创新点**

创新点在于：①系统化评估并优化 NST 参数（层级选择、损失权重、分辨率）以匹配人类世纹理；②将传统迭代方法替换为低延迟的前向网络，实现 3–5 秒内完成高分辨率风格迁移。

**🔧 技术方法**

采用 VGG‑16 特征提取、Gram 矩阵风格损失、感知损失、总变差正则化、前向图像转换网络；后端使用 Flask GPU 服务，前端使用 React Native 交互。

**📊 数据集**

使用自建的人类世风格图像集（工业废弃物、塑料、混凝土等），并在视觉一致性强的样本上训练；内容图像为移动设备拍摄的真实场景。

**📈 对比分析**

通过层次、风格权重、训练轮次、批大小、输出分辨率等多维度实验比较，发现 1280×2276 分辨率能在 3–5 秒内完成高质量迁移；实验验证了在保持纹理细节的同时显著降低延迟。

**⚠️ 局限性**

局限性包括：对风格集的依赖导致泛化受限；高风格权重易导致语义消失；在低端移动设备上仍有较高延迟；缺乏实时自适应风格选择与模型在线更新机制。

---

## 154. Out-of-Distribution Generalization in Graph Foundation Models

**arXiv ID:** 2601.21067 | [PDF](https://arxiv.org/pdf/2601.21067v1)

**作者:** Haoyang Li `[一作]` (Tsinghua University), Wenwu Zhu `[通讯]` (Tsinghua University)

**通讯引用:** 22783 | [OpenAlex ID](https://openalex.org/A5100339293)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对图基础模型（Graph Foundation Models, GFMs）在面对分布偏移（OOD）时的学习与推理进行了系统综述，提出了统一的挑战与问题表述，并对已有方法按任务是否固定进行分类，整理了评估方案和未来研究方向。

**💡 创新点**

创新点在于：①首次从OOD角度聚焦GFMs的综述；②构建了包含结构、域、模态、任务四个维度的统一问题框架；③将方法分为“同任务”与“异任务”两大类，帮助读者快速定位技术路线；④汇总并比较了主流评估范式与指标，指出当前评测的不足与未来改进方向。

**🔧 技术方法**

主要技术包括：对文献的系统检索与梳理；基于挑战维度的分类与对比；对现有预训练目标、对齐机制、增量学习、提示/指令方式的技术归纳；以及对评测策略（结构、域、模态、任务四个维度）的梳理。

**📊 数据集**

作为综述论文，未引入新的实验数据集；文中引用的实验大多来自先前工作，涵盖诸如OGB、Reddit、BioKG、MoleculeNet等公开图数据集，供作者在评估表格中做对比。

**📈 对比分析**

文章通过表格和图示对比了不同GFMs在四类OOD评测（结构、域、模态、任务）中的表现，指出多数方法在单一维度仍有显著性能衰减，只有少数基于对齐、提示或多任务预训练的模型表现相对稳健；并讨论了评测结果的可重复性与对比公平性问题。

**⚠️ 局限性**

局限性包括：①综述依赖已公开论文，可能遗漏最新或非主流工作；②对实验结果的比较多基于原论文报告，缺乏统一标准化实验；③对真实世界分布偏移的场景和长期部署考量讨论不充分；④未来方向虽提出，但仍缺少针对性实验验证。

---

## 155. Self-Improving Pretraining: using post-trained models to pretrain better models

**arXiv ID:** 2601.21343 | [PDF](https://arxiv.org/pdf/2601.21343v1)

**作者:** Ellen Xiaoqing Tan `[一作]` (Meta), Olga Golovneva `[通讯]` (Meta)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自我提升预训练框架，利用已训练好的模型在预训练阶段充当判别器和重写器，对后缀生成进行强化学习，从而提升生成质量、安全性与事实性。

**💡 创新点**

创新点在于：①将后训练模型直接作为教师引入预训练；②将预训练视为前缀条件下的后缀生成任务，而非单纯的下一词预测；③结合重写、原始后缀与模型rollout的多候选进行在线 DPO / RF‑NLL 学习，提升多维度性能。

**🔧 技术方法**

使用的技术包括：强化学习（DPO、RF‑NLL）、后缀判别器与重写器、前缀条件后缀生成任务、链式思维提示、模型微调/蒸馏等。

**📊 数据集**

使用的数据集：SlimPajama（高质量过滤版）和 RedPajama（包含不安全内容），评估阶段使用 BoolQ、PIQA、MMLU、FActScore、TruthfulQA、ToxiGen、RealToxicityPrompts、XStest 等多种基准。

**📈 对比分析**

方法与基线对比：采用 GPT‑OSS‑120B 判别器评估生成质量、事实性、安全性；相较于传统下一词预测基线，质量 win‑rate 提升至约86%，事实性分数从42.3提升至57.6，安全性从76.9提升至91.1，显示显著性能提升。

**⚠️ 局限性**

局限性：计算开销大、训练速度慢；需依赖强大的后训练模型；安全性过度优化可能不适用于所有场景；对多任务共优化的效果尚未完全验证；框架在不同任务上的泛化仍待进一步研究。

---

## 156. Mesh Splatting for End-to-end Multiview Surface Reconstruction

**arXiv ID:** 2601.21400 | [PDF](https://arxiv.org/pdf/2601.21400v1)

**作者:** Ruiqi Zhang `[一作]` (Hong Kong Baptist University), Jie Chen `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 20543 | [OpenAlex ID](https://openalex.org/A5100333005)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

利用多层半透明软化网格，使网格在保持拓扑控制的同时获得体积渲染的 3D 视野，实现端到端图像监督的网格重建

**💡 创新点**

首次将网格软化为可微分的多层伪体积表示，并结合分层光栅化渲染器和混合拓扑控制（DMTet+连续重网格），显著提升细节捕获和网格质量

**🔧 技术方法**

差分网格软化、基于栅格的多层渲染（Mesh Splatting）、体积渲染公式、hash 编码的颜色网络、DMTet（深度网格）、连续重网格、光照、法线监督等

**📊 数据集**

DTU、BlendedMVS、NeRF Synthetic、Mip-NeRF360、TandT 等对象级与场景级数据集

**📈 对比分析**

与 NeuS、Neuralangelo、GaussianSurfel、2DGS、SuGaR、IMLS‑Splatting 等方法对比，Chamfer Distance 下降 20% 以上，顶点数仅 300k 左右，训练时间约 20 分钟，整体性能领先或接近 SOTA

**⚠️ 局限性**

对大规模场景的可扩展性受限于网格化的四面体网格分辨率和 GPU 内存；对极细结构（如电缆、头发）仍存在细节缺失；当基准网格与真实表面差距大时，软化层覆盖不足，梯度难以有效更新

---

## 157. Track-centric Iterative Learning for Global Trajectory Optimization in Autonomous Racing

**arXiv ID:** 2601.21027 | [PDF](https://arxiv.org/pdf/2601.21027v1)

**作者:** Youngim Nam `[一作]`, Cheolhyeon Kwon `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于轨道的迭代学习框架，在不确定车辆动力学下通过波形参数化和贝叶斯优化实现全程轨迹最优化，并在真实赛道上迭代收集数据以更新动力学模型。

**💡 创新点**

创新点在于将轨迹参数化与残差动力学学习统一到同一迭代循环中，使用小波变换低维表示轨迹，并在贝叶斯优化中直接评估带学习动力学的闭环仿真，从而在全程尺度上实现最优时间。

**🔧 技术方法**

采用双轮模型、Pacejka 轮胎模型、Gaussian Process 残差动力学学习、离散小波变换参数化、贝叶斯优化、模型预测控制追踪。

**📊 数据集**

在 15 个随机生成的赛道场景中进行仿真，并在 1/10 比例自主赛车平台上收集真实轨迹数据；实验数据为多次循环的赛道运行记录。

**📈 对比分析**

与名义动力学、GP‑Track、GP‑Opt+Track、样条参数化、非迭代学习等基准对比，平均提升 20.7% 的圈速，并在硬件实验中在两种 MPC 控制器下均表现出最快圈速。

**⚠️ 局限性**

局限在于对轨道单一轨迹的学习，无法捕捉沿轨道空间变化的动力学特性，且对大规模参数化的可扩展性尚未验证。

---

## 158. Algorithms for the local and the global postage stamp problem

**arXiv ID:** 2601.21423 | [PDF](https://arxiv.org/pdf/2601.21423v1)

**作者:** Léo Colisson Palais `[一作]` (University of Grenoble Alpes), Aude Maignan `[通讯]` (University of Grenoble Alpes)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文研究了邮票问题，包括局部邮票问题（LPSP）和全局邮票问题（GPSP），提出了改进的算法以提高时间复杂度和内存使用效率。

**💡 创新点**

创新点在于提出了一种新的递归分治算法和多项式近似算法，能够有效地解决局部和全局邮票问题，并在安全多方计算中提高效率。

**🔧 技术方法**

使用了递归分治算法和多项式近似算法，结合动态规划和贪心算法的思想。

**📊 数据集**

使用了不同的邮票面值集合和最大邮票位置数的组合进行实验，具体数据集未详细列出。

**📈 对比分析**

与现有方法（如Mossige的算法）进行比较，提出的算法在时间复杂度和内存使用上均有显著改进，尤其在处理大规模邮票问题时表现更优。

**⚠️ 局限性**

局部邮票问题是NP难题，尽管提出的算法在实践中表现良好，但在某些情况下仍可能面临计算复杂度高的问题。

---

## 159. Can Neural Networks Learn Small Algebraic Worlds? An Investigation Into the Group-theoretic Structures Learned By Narrow Models Trained To Predict Group Operations

**arXiv ID:** 2601.21150 | [PDF](https://arxiv.org/pdf/2601.21150v1)

**作者:** Henry Kvinge `[一作]` (Pacific Northwest National Laboratory), Helen Jenne `[通讯]` (Pacific Northwest National Laboratory)

**通讯引用:** 24 | [OpenAlex ID](https://openalex.org/A5004208540)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

训练小型MLP和Transformer预测有限群的二元运算，并检验它们是否学到了群论概念（交换性、单位元、子群）

**💡 创新点**

提出了一套基于学习动态、泛化性和内部表征的多维度评估框架，验证即使极简模型也能捕捉到抽象代数结构

**🔧 技术方法**

使用梯度下降训练的全连接网络与自回归Transformer，结合正则化、Adam优化器和交叉熵损失，辅以线性探针和余弦相似度分析

**📊 数据集**

构造了多组有限群（C_64、C_67、C_100、C_256、S_4、S_5、D_30等）的全量运算样本，采用独热编码做输入输出

**📈 对比分析**

通过对称一致性、OOD对称一致性、相似度、身份精度、子群精度以及线性探针准确率进行对比，发现模型能在一定程度上学习交换性和子群结构，但对单位元的识别效果差，整体性能表现与任务难度和超参数密切相关

**⚠️ 局限性**

局限性包括：对单位元的学习缺乏显著信号，模型对不同初始化和超参数极为敏感，抽象概念的可解释性受限，未能在无先验知识的情况下自动发现新结构

---

## 160. Adversarial Vulnerability Transcends Computational Paradigms: Feature Engineering Provides No Defense Against Neural Adversarial Transfer

**arXiv ID:** 2601.21323 | [PDF](https://arxiv.org/pdf/2601.21323v1)

**作者:** Achraf Hsain `[一作]` (King Fahd University of Petroleum and Minerals), Hamoud Aljamaan `[通讯]` (King Fahd University of Petroleum and Minerals)

**通讯引用:** 734 | [OpenAlex ID](https://openalex.org/A5055125044)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

评估神经网络生成的对抗样本在迁移至基于 HOG 特征的经典机器学习分类器（KNN、决策树、线性 SVM、RBF SVM、浅层 ANN）时的鲁棒性。

**💡 创新点**

首次系统性验证跨范式攻击的可行性，并发现 FGSM 在迁移攻击中优于 PGD（攻击层级反转），同时 HOG 特征并未提供天然防御。

**🔧 技术方法**

采用 VGG16 作为 surrogate 进行 FGSM 与 PGD 攻击；通过 scikit‑image 提取 HOG 特征；使用 scikit‑learn 训练上述经典模型并对其在原始、FGSM、PGD 三种数据集上进行评估；以 AlexNet 的神经网络迁移结果作为基准。

**📊 数据集**

CIFAR‑10 图像分类数据集。

**📈 对比分析**

比较三种数据集下各模型的准确率，结果显示 HOG‑基分类器在 FGSM 下的相对准确率下降可达 16.6%–59.1%，与神经网络迁移相当或更高；PGD 在迁移中的效果显著弱于 FGSM，表明迭代攻击易过拟合 surrogate。

**⚠️ 局限性**

实验仅在 CIFAR‑10 上、仅使用单一 VGG16 surrogate，未进行多随机种子或置信区间评估，也未验证更高分辨率数据集或不同 surrogate 架构，限制了结论的普适性。

---

## 161. PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models

**arXiv ID:** 2601.21238 | [PDF](https://arxiv.org/pdf/2601.21238v1)

**作者:** Xuewen Liu `[一作]` (Institute of Automation, Chinese Academy of Sciences), Qingyi Gu `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种面向自回归视觉生成模型的无训练后量化框架PTQ4ARVG，解决通道级异常、令牌级动态性和样本级分布不匹配等问题，支持8‑bit与6‑bit量化；

**💡 创新点**

创新点在于：①基于数学优化的Gain‑Projected Scaling（GPS）推导通道缩放因子；②利用自回归模型固定令牌长度与位置不变性实现静态令牌量化（STWQ）；③通过分布引导校准（DGC）挑选高熵样本提升校准质量；

**🔧 技术方法**

核心技术包括：量化理论推导（Taylor展开、哈希森矩阵近似）、Per‑Channel 等价缩放、静态令牌量化、Mahalanobis距离分布熵选择、标准CUDA实现；

**📊 数据集**

在ImageNet上对VAR、RAR、PAR、MAR四大自回归视觉生成模型进行50k张图像生成评估；

**📈 对比分析**

与SmoothQuant、OS+、RepQ*、OmniQuant、QuaRot、SVDQuant等代表性方法对比，PTQ4ARVG在6‑bit量化下Fidelity提升约35–40分，8‑bit量化保持近似原始精度；加速比最高可达3.01×，显存压缩约1.92×；

**⚠️ 局限性**

局限性：目前仅针对线性层与矩阵乘法实现量化；在极低位宽（≤4‑bit）下仍难以保持高精度；对非自回归生成模型的泛化尚待验证。

---

## 162. Dynamic Framework for Collaborative Learning: Leveraging Advanced LLM with Adaptive Feedback Mechanisms

**arXiv ID:** 2601.21344 | [PDF](https://arxiv.org/pdf/2601.21344v1)

**作者:** Hassam Tahir `[一作]` (Swinburne University of Technology), Omar Mubin `[通讯]` (Western Sydney University)

**通讯引用:** 3635 | [OpenAlex ID](https://openalex.org/A5022431311)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个基于大型语言模型（GPT‑4o）的动态协同学习框架，支持实时讨论、个性化反馈与公平参与，提升学生的批判性思维与合作体验。

**💡 创新点**

创新点在于：① 将LLM作为自适应讨论主持人，能够根据群体互动动态调整提示与话题；② 设计多层反馈机制（参与度、反思性反馈、强化学习信号）持续改进AI调节；③ 架构模块化（React 前端、Flask 后端、LangChain RAG）实现跨学科、跨语言的可插拔数据集支持；④ 通过实验验证了显著降低响应延迟、提升参与度和理解深度。

**🔧 技术方法**

核心技术包括：GPT‑4o 作为对话模型；LangChain 进行提示管理与状态维护；Retrieval‑Augmented Generation（RAG）实现动态内容检索；Flask + Socket.IO 实时通信；ReactJS 前端交互；差分隐私与偏见检测等伦理保障。

**📊 数据集**

主要数据集为 FairytaleQA（10,580 题，来自 278 童话故事），用于测试对话生成、提问与反馈；系统设计兼容任意包含段落与 QA 对的文本数据集。

**📈 对比分析**

与 PeerGPT（基于 GPT‑3.5）等现有框架比较：在响应时间、讨论质量、参与度平衡和自适应性方面均优于传统静态模型；实验平均响应延迟为 1.84 s，系统能够实时处理多模型（GPT‑4o + DeepSeek V3）交互，且在模拟学生群体中实现更高的合作与理解水平。

**⚠️ 局限性**

局限性包括：① 仅使用模拟学生角色，缺乏真实人类互动的复杂性；② 仅测试单一童话段落，未验证跨学科或多模态内容的适应性；③ 依赖 GPT‑4o 与 DeepSeek V3，存在模型特定偏差，尚未评估对其他 LLM 的通用性。

---

## 163. User-Centric Phishing Detection: A RAG and LLM-Based Approach

**arXiv ID:** 2601.21261 | [PDF](https://arxiv.org/pdf/2601.21261v1)

**作者:** Abrar Hamed Al Barwani `[一作]` (German University of Technology in Oman), Raja Waseem Anwar `[通讯]` (German University of Technology in Oman)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套基于检索增强生成（RAG）与大语言模型（LLM）的用户个性化钓鱼邮件检测框架，能够在每封邮件中动态检索用户历史合法邮件和实时威胁情报，并将其作为上下文驱动LLM判断。

**💡 创新点**

创新点在于将用户邮件历史与实时域/URL威胁情报融合为检索上下文，形成个性化决策路径，从而显著降低误报（FPR）并提升整体准确率；同时提出统一的JSON结构化输出，方便自动化后端处理。

**🔧 技术方法**

使用技术包括：RAG（检索+生成）、语义嵌入与FAISS近邻检索、VirusTotal多引擎威胁情报查询、四种开源LLM（Llama4-Scout、DeepSeek-R1、Mistral-Saba、Gemma2-9B）、Prompt工程、低延迟Groq API、Python工具链（LangChain、FAISS、Sentence‑Transformers、Pandas）。

**📊 数据集**

数据集为平衡的500封邮件（250真合法、250钓鱼）。合法邮件采集自已授权用户的个人/机构邮箱（IMAP只读），钓鱼邮件来源于公开钓鱼仓库及内部安全情报。邮件经MIME解析、UTF‑8标准化、HTML去噪、正文抽取、敏感信息匿名化后输入模型。

**📈 对比分析**

实验对比无RAG与RAG两种配置，结果显示RAG显著降低误报率（平均从12%降至约4%），提升F1（0.93→0.98）。Llama4-Scout+RAG在所有指标上表现最佳，其次是DeepSeek‑R1+RAG；其他模型虽受益但仍偏向误报。相比传统规则/机器学习方法，RAG+LLM方案在精确度和误报率上均有明显优势。

**⚠️ 局限性**

局限性包括：对多语言与大规模机构邮件场景的泛化尚未验证；长线程、深度嵌套邮件及高度混淆URL的鲁棒性待提升；检索延迟和模型成本在大规模部署中仍需进一步优化；以及对实时威胁情报API依赖导致潜在单点瓶颈。

---

## 164. AI-Augmented Density-Driven Optimal Control (D2OC) for Decentralized Environmental Mapping

**arXiv ID:** 2601.21126 | [PDF](https://arxiv.org/pdf/2601.21126v1)

**作者:** Kooktae Lee `[一作]` (New Mexico Institute of Mining and Technology), Julian Martinez `[通讯]` (New Mexico Institute of Mining and Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种基于双MLP的AI增强分布式多机器人环境映射框架，能够在传感和通信受限条件下自适应地更新先验地图并实现覆盖控制。

**💡 创新点**

创新之处在于将双网络分别用于估计样本的均值-方差和长未访问区域的虚拟不确定性，从而在D^2OC控制中实现自校正与全局收敛保证。

**🔧 技术方法**

采用最优传输（Wasserstein）理论、加权质心控制、线性时不变动力学、离散时间控制以及双MLP在线学习等技术。

**📊 数据集**

使用在仿真环境中生成的地坟甲烷源分布（synthetic ground‑truth）进行实验，并无外部真实数据集。

**📈 对比分析**

通过与不使用MLP的D^2OC基线对比，利用Wasserstein距离和映射误差评估，结果表明AI增强版本在稳态时Wasserstein距离显著下降，覆盖精度明显优于传统方法。

**⚠️ 局限性**

局限在于仅验证了静态环境下的仿真，有限样本导致误差下限，且对动态变化、异构队形和实际硬件的适应性尚待进一步研究。

---

## 165. Thinking in Frames: How Visual Context and Test-Time Scaling Empower Video Reasoning

**arXiv ID:** 2601.21037 | [PDF](https://arxiv.org/pdf/2601.21037v1)

**作者:** Chengzu Li `[一作]` (University of Cambridge), Anna Korhonen `[通讯]` (University of Cambridge)

**通讯引用:** 10571 | [OpenAlex ID](https://openalex.org/A5081393566)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

使用视频生成模型作为视觉推理器，在迷宫导航和唐璜拼图两种视觉规划任务中实现从起始状态到目标状态的连续推理。

**💡 创新点**

创新点包括：①将视频生成视为推理过程，利用连续帧捕捉空间动态；②通过视觉上下文作为显式控制提升零样本泛化；③发现视频帧数即推理预算的“视觉测试时刻缩放”规律，能够在无额外训练的情况下提升 OOD 任务表现。

**🔧 技术方法**

采用预训练文本到视频扩散模型 Wan 2.2 TI2V 5B 进行 LoRA 微调；与 GPT‑5.x、Qwen‑3‑VL‑8B、VPRL、Qwen‑Image‑Edit 等大语言模型和图像编辑模型对比；使用视觉一致性指标、进度率、IoU 等评价指标。

**📊 数据集**

实验数据集包含：①迷宫数据集，网格尺寸 3×3~6×6，40 种不同图标；②唐璜拼图数据集，目标轮廓多样化，包含连续旋转与平移的步骤化生成任务。

**📈 对比分析**

在分布内（IID）任务中，视频生成模型在迷宫导航上达 98%‑96% 的 Exact Match，显著优于文本推理基线；在 OOD 任务（更大迷宫、更长路径、未见图标、未见拼图轮廓）中，零样本表现仍保持 80%+，且通过增加帧数可将性能从 36% 提升至 78% 左右；在唐璜拼图上，视觉上下文为 Translation 情况下达到 68% 的 Strict Goal Completion，低于图像编辑基线但解释性更强。

**⚠️ 局限性**

主要限制：①在高视觉变化任务（唐璜拼图）中，帧数增多会导致几何失真和形状扭曲，表现不如单帧编辑模型；②帧数上限受模型位置嵌入容量限制，过长序列会出现性能下降；③对细粒度几何一致性的保持仍依赖模型规模与训练数据丰富度，尚未完全解决。

---

## 166. ChunkWise LoRA: Adaptive Sequence Partitioning for Memory-Efficient Low-Rank Adaptation and Accelerated LLM Inference

**arXiv ID:** 2601.21109 | [PDF](https://arxiv.org/pdf/2601.21109v1)

**作者:** Ketan Thakkar `[一作]` (Bentley University), Rajendra Ugrani `[通讯]` (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种在推理时自适应划分序列为可变长度块，并为每块动态分配LoRA低秩配置，以实现更高的内存与延迟效率。

**💡 创新点**

创新点在于：①基于轻量级词元复杂度估计的在线块划分；②每块按需选择LoRA秩和缩放，利用预先SVD分解的秩阶梯；③边界安全组合与KV缓存策略同步调整，保证输出连贯性与兼容现有高性能核。

**🔧 技术方法**

使用的技术包括：LoRA低秩适配、FlashAttention等高效注意力核、INT8量化与稀疏化、动态块划分与秩选择算法、边界线性淡化、批量对齐与复杂度分桶。

**📊 数据集**

实验数据集：Wikitext‑103、SQuAD v2.0、FLORES‑101（翻译BLEU）等。

**📈 对比分析**

与静态LoRA、AdaLoRA等基线比较，ChunkWise LoRA在LLaMA‑7B上平均推理延迟下降至14.9 ms/词、峰值显存降低至9.1 GB，并保持或提升BLEU（25.3）、EM（63.5）及Perplexity（5.61）等任务指标。

**⚠️ 局限性**

局限性包括对词元复杂度估计的依赖，若估计失准可能导致块划分或秩分配不当；目前仅针对单语言文本，跨语言或多模态场景需进一步验证。

---

## 167. Hypersolid: Emergent Vision Representations via Short-Range Repulsion

**arXiv ID:** 2601.21255 | [PDF](https://arxiv.org/pdf/2601.21255v1)

**作者:** Esteban Rodríguez-Betancourt `[一作]` (Universidad de Costa Rica), Edgar Casasola-Murillo `[通讯]` (Universidad de Costa Rica)

**通讯引用:** 41 | [OpenAlex ID](https://openalex.org/A5103748298)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Hypersolid方法，将自监督学习重新建模为硬球排斥的离散包装问题，利用短程排斥防止表示坍塌，并通过最大池化的特征并集实现视图对齐。

**💡 创新点**

核心创新在于仅在局部余弦相似度阈值α内强制最小距离，从而实现近似单射，避免全局均匀化；对齐目标改为特征并集而非均值，保持增强多样性。

**🔧 技术方法**

使用短程硬球排斥损失、最大池化特征对齐、弱L₂正则化、共享ResNet编码器、投影头以及多视图增强等技术。

**📊 数据集**

在ImageNet‑1000、CIFAR‑10/100、Food‑101、STL‑10等公开数据集上进行实验。

**📈 对比分析**

与SimCLR、BYOL、DINO、VICReg、Barlow Twins、LeJEPA以及有监督基线对比，在ImageNet‑1000表现相当，在CIFAR‑100和Food‑101上分别提升约10.6%和5.6%，在低分辨率和细粒度任务上表现突出。

**⚠️ 局限性**

主要局限在于计算复杂度为O((B·V)²)，未实现邻居近似；缺乏大规模ViT的超参数与规模法则探索；仅在卷积网络上验证，Transformer需要进一步调优。

---

## 168. White-Box Op-Amp Design via Human-Mimicking Reasoning

**arXiv ID:** 2601.21321 | [PDF](https://arxiv.org/pdf/2601.21321v1)

**作者:** Zihao Chen `[一作]` (Fudan University), Fan Yang `[通讯]` (Fudan University)

**通讯引用:** 21850 | [OpenAlex ID](https://openalex.org/A5100669509)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

提出一种基于大型语言模型的白盒运算放大器参数设计框架 White-Op，通过人类思维的假设约束化和迭代验证流程，实现可解释、可靠的行为级设计并映射至晶体管级。

**💡 创新点**

创新点在于：①将隐式的人工工程师推理转化为可执行的“假设约束”步骤；②构造可求解的闭式优化问题并在 LLM 指导下迭代验证；③通过约束引入保证极点/零点位置的安全性，从而提升设计的可解释性与可靠性。

**🔧 技术方法**

使用技术包括：大型语言模型（LLM）进行符号推导与约束生成；符号电路分析（KCL、Cramer's 规则）和极点零点近似提取；约束优化（白盒可微分优化）；行为级 HSPICE 仿真；晶体管级映射工具（g_m/I_D 方法）。

**📊 数据集**

实验数据集为 9 种运算放大器拓扑（来自文献中公开的参考），每种拓扑进行 10 次随机初始实验，采集理论、行为级和晶体管级性能指标。

**📈 对比分析**

与基准 Bayesian Optimization (BO) 进行比较；White-Op 在理论到行为级误差仅 8.52%，所有拓扑在晶体管级均保持功能；BO 虽在行为级取得更高 FoM，但在 5/9 拓扑的晶体管级失效。总体运行时间约 13 分钟，White-Op 在可靠性与可解释性上显著优于 BO。

**⚠️ 局限性**

局限性包括：①依赖行为级模型的准确性，可能忽略复杂寄生效应；②需要精细的 LLM 提示工程和假设约束设计，人工干预较多；③在更复杂或未知拓扑上的通用性尚未充分验证；④若假设约束失效，可能导致极点/零点定位错误。

---

## 169. Developers in the Age of AI: Adoption, Policy, and Diffusion of AI Software Engineering Tools

**arXiv ID:** 2601.21305 | [PDF](https://arxiv.org/pdf/2601.21305v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 170. Soft Quantization: Model Compression Via Weight Coupling

**arXiv ID:** 2601.21219 | [PDF](https://arxiv.org/pdf/2601.21219v1)

**作者:** Daniel T. Bernstein `[一作]` (Princeton University), David Schwab `[通讯]` (CUNY Graduate Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种通过在神经网络训练过程中引入短程吸引耦合来实现模型量化的新方法，称为软量化。

**💡 创新点**

创新点在于通过仅依赖两个超参数，快速诱导模型权重分布的离散化，并在混合精度下进行量化，超越了传统的直方图均衡后训练量化方法。

**🔧 技术方法**

使用了短程吸引耦合的优化技术，结合了层级耦合项和有效势能的计算。

**📊 数据集**

在ResNet-20模型上使用CIFAR-10数据集进行实验。

**📈 对比分析**

与传统的直方图均衡量化（HEQ）方法进行比较，结果显示软量化在多个超参数设置下均优于HEQ，且在压缩和性能之间的权衡表现出更好的一致性。

**⚠️ 局限性**

限制在于该方法的性能可能依赖于超参数的选择，且尚需在更大规模和更复杂的任务上进行评估。

---

## 171. WorldBench: Disambiguating Physics for Diagnostic Evaluation of World Models

**arXiv ID:** 2601.21282 | [PDF](https://arxiv.org/pdf/2601.21282v1)

**作者:** Rishi Upadhyay `[一作]` (University of California), Achuta Kadambi `[通讯]` (University of California)

**通讯引用:** 9947 | [OpenAlex ID](https://openalex.org/A5043479061)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出 WorldBench 这一基于视频的基准，用来评估世界基础模型（World Foundation Models）对具体物理概念、常数及材质属性的理解与再现能力。

**💡 创新点**

创新点主要有：①概念去耦、针对单一物理概念/定律的评测子集；②兼顾直观物理认知与物理参数估计两大层面；③使用视频生成与 SAM2 分割对比的精细化指标，而非传统的二元或模糊度量；④直接量化模型对物理常数（如重力加速度、粘度、摩擦系数）的符合度。

**🔧 技术方法**

技术手段包括：利用 Kubric（PyBullet+Blender）生成高保真物理视频；使用 SAM2 对生成视频进行分割并提取 3D 位置信息；通过曲线拟合估计加速度、终端速度等物理参数；评测模型包括 Cosmos 系列、Wan、Hunyuan Video、CogVideoX 等视频生成模型。

**📊 数据集**

数据集涵盖：①直观物理子集 469 条视频（425 纯合成 + 44 真实）由 Kubric 生成，使用 ShapeNet 对象模型；②物理参数估计子集 279 条视频（gravity、viscosity、friction）同样由 Kubric 生成，并包含相应的真实视频样本。

**📈 对比分析**

评测方法：对直观物理子集使用前景 mIoU 与背景 RMSE；对参数估计子集直接对比模型输出与真值的加速度、摩擦系数、粘度。实验结果表明：所有模型在物理一致性上表现欠佳，尤其在重力加速度与粘度估计上误差大；Cosmos 在重力子任务上相对更好，图像-视频模型普遍低于；模型更擅长持续交互或具备训练先验的场景，且对材料长尾分布适配度差。

**⚠️ 局限性**

局限性包括：①评测仅覆盖有限概念，扩展需大量实验与验证；②受限于可用视频生成模型，评测范围受限；③对长尾材质属性（如高粘度蜂蜜、低摩擦塑料）拟合不佳；④评测主要基于单一视角的 2D 视频，难以处理更复杂的三维交互；⑤缺乏针对碰撞、光学等更细粒度物理过程的测试。

---

## 172. Predict-Project-Renoise: Sampling Diffusion Models under Hard Constraints

**arXiv ID:** 2601.21033 | [PDF](https://arxiv.org/pdf/2601.21033v1)

**作者:** Omer Rochman-Sharabi `[一作]` (University of Liège), Gilles Louppe `[通讯]` (University of Liège)

**通讯引用:** 6994 | [OpenAlex ID](https://openalex.org/A5017670779)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种新的基于扩散模型的约束采样框架PPR，实现了在生成过程中严格满足硬约束。

**💡 创新点**

创新点在于定义受约束的前向过程与对应后向过程，并通过预测-投影-再噪声（Predict-Project-Renoise）迭代逼近受约束的边缘分布，解决了传统投影方法导致分布失真与多模崩塌的问题。

**🔧 技术方法**

利用扩散模型的预测器（predict）、去噪器（denoiser）和投影优化，配合重噪声步骤来维持正确的噪声水平，整体实现无训练约束的采样。

**📊 数据集**

在三类数据集上评估：2D分布、Kuramoto–Sivashinsky PDE轨迹以及全球气象模型Appa，使用人工生成的约束和真实观测约束。

**📈 对比分析**

与现有投影与后验采样方法（DPS、TS、PDM、MMPS等）对比，PPR在约束违规率、Wasserstein距离、k‑NN交叉边率、RMSE、技能、CRPS等指标上均显著优于基线，减少约束违规至少一个数量级。

**⚠️ 局限性**

局限性包括投影-去噪迭代成本高、对可微约束函数的依赖、未提供严格无偏采样保证，且对离散或黑盒约束的适用性有限。

---

## 173. Stochastic Indexing Primitives for Non-Deterministic Molecular Archives

**arXiv ID:** 2601.20921 | [PDF](https://arxiv.org/pdf/2601.20921v1)

**作者:** Faruk Alpay `[一作]` (Bahcesehir University), Levent Sarioglu `[通讯]` (Bahcesehir University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

提出了用于DNA数据存储的“全息Bloom过滤器（HBF）”，实现单次并行关联检索；

**💡 创新点**

将Bloom过滤器与高维向量符号架构相结合，利用圆形卷积编码键值关联，并通过阈值/边缘判决提供可解析的误差分析；

**🔧 技术方法**

使用高维随机Rademacher向量、圆形卷积/相关、FFT/逆FFT、极值与集中不等式等数学工具；

**📊 数据集**

未给出具体实验数据集，论文主要基于理论分析与抽象模型；

**📈 对比分析**

与传统指针追踪结构（如跳表/跳图）对比，HBF在查询时间上为一次并行操作，误差率可随向量维度指数下降，理论上支持更大容量；

**⚠️ 局限性**

局限在于缺乏实验验证、具体生化实现细节不明，以及对高维向量随机性假设的依赖，未考虑实际DNA合成/测序噪声的细粒度影响。

---

## 174. Quick Heuristic Validation of Edges in Dynamic Roadmap Graphs

**arXiv ID:** 2601.20968 | [PDF](https://arxiv.org/pdf/2601.20968v1)

**作者:** Yulie Arad `[一作]` (University of Illinois), Nancy M. Amato `[通讯]` (University of Illinois)

**通讯引用:** 8755 | [OpenAlex ID](https://openalex.org/A5050205557)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在动态环境中，为机器人运动规划维护路网图，提出了Red‑Green‑Gray三色判别范式，利用外部与内部近似快速标记边的有效性。

**💡 创新点**

创新点在于同时使用外部近似的有向包围盒与内部近似的样条/球体，对机器人扫掠体积与障碍物进行双重检查，从而实现快速、准确地将边分类为有效、无效或未知。

**🔧 技术方法**

主要技术包括有向包围盒（OBB）与样条曲线、球体近似、kd‑tree（或R‑tree）交叉查询、PQP基准验证以及基于Probabilistic Roadmap的路网生成。

**📊 数据集**

实验数据集为一个32×32×16的工作空间，包含一个单一矩形棱柱障碍，机器人为6自由度双连杆机械臂，路网约1000条边。

**📈 对比分析**

与Leven&Hutchinson的网格动态路网和原SPITE方法比较，Red‑Green‑Gray方法在误标为0的前提下，正/负样本识别率均显著提升，更新运行时间与SPITE相当甚至略优；PQP粗暴更新耗时两百倍。

**⚠️ 局限性**

局限性包括对细长障碍物的外近似表现不佳、实验规模有限、未在多体或更大环境下验证、实时性与GPU并行化仍需进一步探索。

---

## 175. Token Entropy Regularization for Multi-modal Antenna Affiliation Identification

**arXiv ID:** 2601.21280 | [PDF](https://arxiv.org/pdf/2601.21280v1)

**作者:** Dong Chen `[一作]` (The University of Hong Kong), Zizhuang Wei `[通讯]` (Huawei)

**通讯引用:** 504 | [OpenAlex ID](https://openalex.org/A5045305334)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过融合基站视频、天线几何特征和物理单元识别（PCI）信号，实现天线归属识别的多模态匹配。

**💡 创新点**

提出了令牌熵正则化（Token Entropy Regularization, TER）模块，利用令牌熵自适应稀疏化特征，使跨模态表示更易对齐；并提出从预训练到监督微调的两阶段训练流程，显著提升匹配精度。

**🔧 技术方法**

采用 Vision Transformer（ViT）、DINOv3、Video Swin Transformer、TimeSformer 等视觉编码器，结合 CLIP 对齐策略；TER 包含 Enhanced Token Entropy (ETE) 层和 Token Entropy Loss (TEL) 约束；使用 Adam、AdamW 等优化器和余弦学习率衰减。

**📊 数据集**

使用无人机拍摄的基站高清视频（约 30 张图像/站点），提取天线实例与几何参数；同时采集同一站点的 4G/5G PCI 信号（经纬度、信号强度）；通过网络参考数据建立视觉-PCI 的真值配对。

**📈 对比分析**

将 TER 与基线对比，单独使用预训练+微调阶段，在 Video Swin 等编码器上取得 top‑1 87.91% / top‑3 91.94% 的匹配准确率；相较 End‑to‑End 学习提升 26–38%；TER 在所有模型上均能加速收敛并提升 1–3% 的指标。

**⚠️ 局限性**

主要限制包括：① 训练样本稀缺，需人工标注真值；② 模式差异大，仍需更强跨模态对齐；③ 在开放场景下可能出现未检测到的天线或误报；④ 该方法对硬件资源（GPU）依赖较高。

---

## 176. WheelArm-Sim: A Manipulation and Navigation Combined Multimodal Synthetic Data Generation Simulator for Unified Control in Assistive Robotics

**arXiv ID:** 2601.21129 | [PDF](https://arxiv.org/pdf/2601.21129v1)

**作者:** Guangping Liu `[一作]` (Saint Louis University), Madi Babaiasl `[通讯]` (Saint Louis University)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5098937098)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了基于Isaac Sim的WheelArm‑Sim仿真平台，集成轮椅与机器人臂，并通过人机协同收集13项ADL任务的多模态数据。

**💡 创新点**

首次实现了面向轮椅+机械臂一体化控制的物理仿真环境及对应多模态数据集，为协助技术提供统一数据来源。

**🔧 技术方法**

采用Isaac Sim、ROS2、键盘/图形界面遥控、基于screw理论的逆运动学和LSTM+多模态特征融合网络等技术。

**📊 数据集**

收集了包含人类指令、RGB‑D图像、IMU、关节状态等的67,783条样本，涵盖232条轨迹，涵盖13个任务。

**📈 对比分析**

用基线的LSTM多模态序列模型对必需辣酱抓取任务进行动作预测，表现对简单轨迹较好，但在复杂抓取和时序上仍有误差。

**⚠️ 局限性**

数据集规模有限、模型泛化不足、未解决仿真到真实的迁移缺口，遥控方式单一等限制影响实用性。

---

## 177. PhaseCoder: Microphone Geometry-Agnostic Spatial Audio Understanding for Multimodal LLMs

**arXiv ID:** 2601.21124 | [PDF](https://arxiv.org/pdf/2601.21124v1)

**作者:** Artem Dementyev `[一作]` (Google DeepMind), Vivek Kumar `[通讯]` (Google DeepMind)

**通讯引用:** 13388 | [OpenAlex ID](https://openalex.org/A5107073791)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种名为 PhaseCoder 的 transformer‑only 空间音频编码器，并将其与 Gemma 3n LLM 结合，实现对任意麦克风阵列原始多通道音频的空间感知与推理。

**💡 创新点**

创新点包括：① 通过相位差利用微型 MEMS 麦克风阵列实现对麦克风几何不敏感的空间表示；② 生成“空间音频 tokens”，可直接注入 LLM 进行复杂的空间推理与定向转录；③ 在单一模型中同时完成空间定位、距离估计、方向估计，并与 LLM 端进行端到端微调。

**🔧 技术方法**

使用的技术有：STFT + 复数幅度/相位特征提取；三种位置编码（序列、帧、麦克风坐标）注入；Transformer 编码器（5 层，4 头自注意力）；两阶段学习率调度与噪声数据增强；LoRA 微调 Gemma；多任务交叉熵训练与距离、方位、仰角分类头。

**📊 数据集**

使用的数据集包括：合成 LibriSpeech + 1.5M 真实房间冲击响应（RIR）与随机麦克风几何；RSL2019、LOCATA 真实麦克风阵列数据；Freesound 噪声源；以及基于 RSL2019 生成的问答（QA）语料。

**📈 对比分析**

与现有的 GI‑DOAEnet 进行对比，PhaseCoder 在 LOCATA 上 Acc@10 86.96%（高于 82.48%）且 MAE 7.44°（低于 7.82°），在 RSL2019 上表现略逊但依然优于基线；Fine‑tuned Gemma 在定位、推理、转录任务中显著优于未加入空间 tokens 的 Gemma 与 Gemini Flash，尤其在空间推理准确率和定向转录 WER 上有明显提升。

**⚠️ 局限性**

主要局限包括：假设自由漂浮的全向麦克风阵列，未考虑设备自身的声学遮蔽与指向性；仅处理静止或静态源，缺乏对移动源的建模；侧重单一主导语音源，未涵盖多源或非语音事件；距离估计受房间声学参数限制，难以做到绝对精确；未结合视觉或其他感知模态进一步消除距离歧义。

---

## 178. The Noncomputability of Immune Reaction Complexity: Algorithmic Information Gaps under Effective Constraints

**arXiv ID:** 2601.20865 | [PDF](https://arxiv.org/pdf/2601.20865v1)

**作者:** Emmanuel Pio Pastore `[一作]` (University of Calabria), Francesco De Rango `[通讯]` (University of Calabria)

**通讯引用:** 2411 | [OpenAlex ID](https://openalex.org/A5015239623)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文构建了一种基于算法信息论的、通过有效性判定过滤的证书驱动反应模型，并提出了一个规范化建议量化（NAQ）作为任务难度的无尺度指标。

**💡 创新点**

创新点在于：① 用 Kolmogorov 复杂度定义最小可实现方案的“建议复杂度”，② 通过 NAQ 将其转化为可比的百分位数；③ 给出精确的描述/选择两部上界、泛化性、近似稳健性；④ 证明了 NAQ 与信息率-失真理论的操作性逆推关系；⑤ 延伸到资源受限情形（Levin 的 Kt 变体）并给出经验估计的 DKW 置信界；⑥ 在免疫适应性建模中展示了该框架的可行性。

**🔧 技术方法**

主要技术包括：前缀 Kolmogorov 复杂度、可判定/可枚举有效性谓词、无输入盲执行器模型、两部编码与选择成本分析、Levin 时间-复杂度（Kt）搜索、信息率-失真逆推、Dvoretzky–Kiefer–Wolfowitz 不等式、压缩代理的实证近似。

**📊 数据集**

本文本身为理论研究，未使用特定公开数据集；在生物学应用章节中假设使用基于计算可枚举的抗原-MHC 结合数据库和免疫模拟数据，但具体数据未给出。

**📈 对比分析**

由于主要关注理论框架，文中并未进行实验对比；在理论层面，NAQ 与传统的随机性缺陷、结构函数、对称信息距离等指标进行对比，说明其在无尺度、可跨任务可比较、与执行器无关等方面的优势；若采用压缩代理实现，实测的 NAQ 与压缩长度高度相关，且符合 DKW 的样本误差界。

**⚠️ 局限性**

限制主要包括：① 计算不可逼近性（最小建议复杂度与 Kolmogorov 复杂度本质上不可计算）；② 对选择枚举的依赖（枚举的相对常数可能导致上界偏移）；③ 需要固定、全局的无输入盲执行器和有效性谓词，实际应用时难以确定；④ NAQ 对于非可判定或极度不确定的有效性谓词可能不适用；⑤ 资源受限版本依赖于可有效估计时间上界，实际中难以保证。

---

## 179. Meta-ROS: A Next-Generation Middleware Architecture for Adaptive and Scalable Robotic Systems

**arXiv ID:** 2601.21011 | [PDF](https://arxiv.org/pdf/2601.21011v1)

**作者:** Anshul Ranjan `[一作]` (PES University), Shylaja S S `[通讯]` (PES University)

**通讯引用:** 380 | [OpenAlex ID](https://openalex.org/A5108721678)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出并实现了Meta-ROS中间件，旨在简化机器人开发、提升跨平台兼容性并减少学习曲线。

**💡 创新点**

创新点在于使用Zenoh实现低延迟、实时双向通信，支持多种数据类型，并通过统一的API、轻量化设计降低了硬件依赖；同时结合了安全、可靠性机制和云边协同能力。

**🔧 技术方法**

技术包括Python实现的轻量级发布/订阅模型、ZeroMQ消息代理、Zenoh分布式数据传输、AES加密、CFS调度器、gRPC/WS通信、容器化与Kubernetes集成。

**📊 数据集**

本文主要使用ROS 1、ROS 2（Connext、OpenSplice、FastRTPS）等现有框架的标准消息与Gazebo仿真机器人数据进行对比；并在PES IoT实验室的真实机器人平台上测试。

**📈 对比分析**

通过单机与多机吞吐量、延迟、CPU与带宽利用率等指标评估。Meta‑ROS在吞吐量上比ROS 2高约30%，消息延迟更低，CPU/内存占用更优，且在高负载、网络中断等场景下保持稳定。

**⚠️ 局限性**

局限性包括对硬件资源依赖仍有一定成本、云端部署与安全策略仍需进一步完善，以及在极端噪声或资源受限的嵌入式设备上未进行充分测试。

---

## 180. Privatization of Synthetic Gaze: Attenuating State Signatures in Diffusion-Generated Eye Movements

**arXiv ID:** 2601.21057 | [PDF](https://arxiv.org/pdf/2601.21057v1)

**作者:** Kamrul Hasan `[一作]` (Texas State University), Oleg V. Komogortsev `[通讯]` (Texas State University)

**通讯引用:** 3597 | [OpenAlex ID](https://openalex.org/A5035152487)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

评估扩展的 DiffEyeSyn 生成的合成视线数据在保留真实信号质量的同时，是否能消除对疲劳等内部状态的可识别信息。

**💡 创新点**

首次系统检验扩散模型合成视线与主观自评之间的相关性，证明合成数据在隐私保护上优于真实数据。

**🔧 技术方法**

使用扩散概率模型 DiffEyeSyn 进行视线合成，并结合特征提取器和 Spearman 相关分析。

**📊 数据集**

使用公开的 GazeBase 数据集进行训练与评估。

**📈 对比分析**

通过与真实数据的 Spearman 相关热图对比，发现合成数据在各任务中相关系数显著降低，保持定位准确度和精度与真实数据相当。

**⚠️ 局限性**

仅在特定任务/回合存在残留相关性，且缺乏对不同内部状态的细粒度控制，需进一步改进模型。

---

## 181. Towards Sensitivity-Aware Language Models

**arXiv ID:** 2601.20901 | [PDF](https://arxiv.org/pdf/2601.20901v1)

**作者:** Dren Fazlija `[一作]` (L3S Research Center), Sandipan Sikdar `[通讯]` (L3S Research Center)

**通讯引用:** 119 | [OpenAlex ID](https://openalex.org/A5015963208)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对企业数据管理中LLM的“敏感度意识”(SA)，在理论上将其与差分隐私(DP)联系起来，并在实践中通过低秩自适应(Supervised LoRA)提升四位量化LLM的SA能力。

**💡 创新点**

创新点在于：①首次构建SA的游戏化形式并证明其是属性推断的后处理，从而给出DP上界；②提出在量化LLM上应用LoRA微调即可显著提升SA，而不大幅牺牲通用性能。

**🔧 技术方法**

技术上使用LoRA微调、差分隐私理论、RBAC安全框架、Qwen3 4‑bit量化模型以及GPT‑3.5/Claude‑2等大模型的对比评估。

**📊 数据集**

数据集方面采用ADI基准生成的企业数据库，并利用公开的SA注解集（约30,000条）进行训练；评估时使用ADI 3,500道问题集以及BIG‑Bench Hard、IFEval、GSM8K‑Platinum三大通用基准。

**📈 对比分析**

与同参数量的开源/闭源全精度模型对比，LoRA Qwen3‑8B提升了约21.7% SA 分数，且在BIG‑Bench Hard、IFEval、GSM8K‑Platinum上的性能仅略有下降（最多约3.3%），证明其在安全与通用能力之间取得平衡。

**⚠️ 局限性**

局限性包括：对主管场景的准确率下降；理论上仅能约束基于统计相关性的泄露，难以处理更复杂的非结构化数据和多租户环境；以及仍需进一步验证在更大规模和多样化场景下的稳健性。

---

## 182. Delegation Without Living Governance

**arXiv ID:** 2601.21226 | [PDF](https://arxiv.org/pdf/2601.21226v1)

**作者:** Wolfgang Rohde `[一作]` `[通讯]` (AiSuNe Foundation), Wolfgang Rohde (AiSuNe Foundation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出运行时治理框架（治理双子）以维持人类在代理 AI 决策中的相关性并对治理缺口进行概念性分析

**💡 创新点**

首次提出治理双子概念，强调人机共进、持续监督与人类影响力的必要性

**🔧 技术方法**

无具体实现技术，采用理论框架与概念性分析方法

**📊 数据集**

无数据集，本文为概念性论文无实验数据

**📈 对比分析**

无实验对比与性能指标，本文仅提供概念性阐述与政策建议

**⚠️ 局限性**

缺乏实证验证与可操作的实现细节，概念性框架仍需进一步细化与测试

---

## 183. asr_eval: Algorithms and tools for multi-reference and streaming speech recognition evaluation

**arXiv ID:** 2601.20992 | [PDF](https://arxiv.org/pdf/2601.20992v1)

**作者:** Oleg Sedukhin `[一作]` (Siberian Neuronets LLC), Andrey Kostin `[通讯]` (Siberian Neuronets LLC)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出多参考、多通配符对齐算法MWER和asr_eval工具，发布多参考俄语长句数据集DiverseSpeech-Ru，并比较多参考与文本归一化对评估的影响。

**💡 创新点**

设计支持多参考与任意长度通配符的字符串对齐，改进得分函数提升词对齐质量，并提出插入惩罚放宽和流式评估图表。

**🔧 技术方法**

基于动态规划的 Needleman–Wunsch 变体实现MWER；构建Python库asr_eval；使用 KenLM/CTC/Whisper 等模型封装；采用时间重映射和流式评估图形。

**📊 数据集**

公开DiverseSpeech-Ru（3.5h俄语长音频）及Sova-RuDevices的500样本重新标注；还使用YODAS2、Whisper 公开评测集。

**📈 对比分析**

通过多模型对齐仪表板与误差可视化，比较不同标注方式对细调动态的影响；结果显示多参考标注抑制了假象的 WER 提升，模型在多参考评测下约 3% 降低错误率，而在单参考下提升显著。

**⚠️ 局限性**

对多参考标注的人工成本高，通配符使用需谨慎；插入惩罚放宽针对重复插入有效，对非重复插入效果不佳；评估工具主要针对俄语，跨语言泛化待验证。

---

## 184. Position-invariant Fine-tuning of Speech Enhancement Models with Self-supervised Speech Representations

**arXiv ID:** 2601.21084 | [PDF](https://arxiv.org/pdf/2601.21084v1)

**作者:** Amit Meghanani `[一作]` (University of Sheffield), Thomas Hain `[通讯]` (University of Sheffield)

**通讯引用:** 4696 | [OpenAlex ID](https://openalex.org/A5030528300)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在噪声环境下，对前端语音增强（SE）模型进行自监督学习（SSL）表示引导的微调，探讨了 MSE 损失导致位置嵌入被滥用的问题，并提出了两种补偿策略（随机零填充与速度扰动+Soft‑DTW），随后评估其对 ASR 与 PR 任务的影响。

**💡 创新点**

创新点在于：①首次把位置嵌入滥用问题移到 SE 微调阶段进行研究；②提出通过随机零填充和速度扰动配合 Soft‑DTW 损失实现位置不变的微调；③证明 Soft‑DTW 方案既能加速收敛，又能在未见噪声条件下显著降低错误率。

**🔧 技术方法**

技术上使用了 HuBERT‑BASE 作为冻结的 SSL 编码器、Facebook 研究的 33.5M 参数时间域增强网络、软动态时间规整（Soft‑DTW）以及随机零填充策略，并采用 Adam 优化器与梯度裁剪。

**📊 数据集**

数据集：在 LibriSpeech 的 clean、train、dev、test 子集上加入 DEMAND 的室内噪声（SNR 0/5/10/20 dB），并在 SUPERB 框架下生成噪声增强版用于 ASR 与 PR 评估。

**📈 对比分析**

与仅使用 MSE 损失的基线相比，Soft‑DTW 方案在 ASR 的 unseen‑noise 设定下 WER 下降约 0.6%（从 9.19% 到 9.06%），PR 的 PER 下降约 0.1%；并在训练中显著加快收敛速度（约 60k 步即可达到 MSE 终点性能）。零填充方案提升有限，主要体现在更快的收敛而非最终性能。

**⚠️ 局限性**

局限性包括：①只在 SE 模块进行微调，未验证在 SSL 预训练阶段应用该策略的可行性；②零填充可能导致 SSL 表征提取受干扰；③实验仅覆盖室内噪声与特定数据集，未检验对更复杂或多源噪声环境的泛化能力。

---

## 185. EGAM: Extended Graph Attention Model for Solving Routing Problems

**arXiv ID:** 2601.21281 | [PDF](https://arxiv.org/pdf/2601.21281v1)

**作者:** Licheng Wang `[一作]` (Tsinghua University), Yuan Shen `[通讯]` (Tsinghua University)

**通讯引用:** 21818 | [OpenAlex ID](https://openalex.org/A5015109937)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种扩展的图注意力模型EGAM，用于求解路由问题，能够同时更新节点和边的嵌入，消除传统GAM对边特征的忽略；

**💡 创新点**

创新点在于引入Node-Edge和Edge-Node两种注意力机制，使得模型在全图上实现双向信息传递，并结合对称性基线的强化学习训练，提升了对复杂约束问题的处理能力；

**🔧 技术方法**

使用的技术包括多头点乘注意力（MHA）、自回归编码-解码架构、REINFORCE策略梯度、对称性基线以及对边特征的线性映射与FF层；

**📊 数据集**

在10,000个实例的标准路由数据集上进行实验，涵盖TSP、CVRP、PCTSP、TSPTW、TSPDL、VRPTW等多种问题；

**📈 对比分析**

与传统启发式求解器（LKH-3、Gurobi、OR-Tools）以及现有RL求解器（GAM、GATv2、POMO、Sym-NCO）对比，EGAM在贪心策略下在简单约束问题上与POMO、Sym-NCO持平或略优，在高度约束问题上实现了约2%至4%的最优性缺口下降和可行率提升，且推理时间与现有RL方法相近；

**⚠️ 局限性**

局限性包括：仍需要针对大规模实例进一步优化规模扩展性；对非自回归模式的支持尚未充分验证；对其他类型组合优化任务的通用性需进一步探究。

---

## 186. Maxwait: A Generalized Mechanism for Distributed Time-Sensitive Systems

**arXiv ID:** 2601.21146 | [PDF](https://arxiv.org/pdf/2601.21146v1)

**作者:** Francesco Paladino `[一作]` (University of California Berkeley), Edward A. Lee `[通讯]` (University of California Berkeley)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出并实现一种名为maxwait的通用分布式协调机制，整合进Lingua Franca语言，以实现时间敏感系统中一致性与可用性之间的可配置权衡；

**💡 创新点**

通过maxwait统一多种经典协调模式（Chandy‑Misra、CRDT、LET、RPC等），并在同一框架内提供可调节的时延控制、故障检测和容错处理，从而使分布式系统既可实现强一致性又可满足实时性需求；

**🔧 技术方法**

使用Lingua Franca编程模型、时钟同步算法、max‑plus代数推导的maxwait决策、缺失输入超时（absent_after）与迟到（tardy）处理器、以及基于逻辑时间的事件调度；

**📊 数据集**

未使用公开数据集，主要通过若干系统案例（飞机门控制、ATM 银行系统、自动紧急制动）演示机制的应用；

**📈 对比分析**

与传统分布式协调技术对比，展示maxwait可通过设置不同的等待阈值实现保守、一致性优先或乐观、可用性优先的运行模式；性能方面通过案例说明能够满足预设的实时截止阈值（如制动系统50 ms）且能在有限时延内检测并处理故障；

**⚠️ 局限性**

局限性包括：需要网络延迟可界定且时钟同步可靠；若延迟超出设定边界会导致迟到消息、事件乱序或阻塞；实现复杂度较高，且在高度动态网络环境下可能难以维持严格的maxwait约束。

---

## 187. L2R: Low-Rank and Lipschitz-Controlled Routing for Mixture-of-Experts

**arXiv ID:** 2601.21349 | [PDF](https://arxiv.org/pdf/2601.21349v1)

**作者:** Minghao Yang `[一作]` (Hokkaido University), Miki Haseyama `[通讯]` (Hokkaido University)

**通讯引用:** 3144 | [OpenAlex ID](https://openalex.org/A5063903016)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种低秩和Lipschitz控制的路由框架L2R，旨在改善混合专家模型的专家专门化和路由稳定性。

**💡 创新点**

L2R通过重塑路由空间和评分几何，解决了现有线性路由的两个基本局限性，提供了更平滑和稳定的路由几何。

**🔧 技术方法**

使用了低秩路由空间、饱和内积评分(SIPS)和多锚点路由机制。

**📊 数据集**

在大规模语言模型和ImageNet上的视觉MoE设置中进行了广泛实验。

**📈 对比分析**

与线性路由和X-MoE等方法进行比较，L2R在路由稳定性、专家专门化和整体模型性能上均表现出一致的提升。

**⚠️ 局限性**

L2R的局限性在于其依赖于低秩表示，可能在某些情况下限制了模型的表达能力。

---

## 188. Leveraging Generative AI for Enhancing Domain-Driven Software Design

**arXiv ID:** 2601.20909 | [PDF](https://arxiv.org/pdf/2601.20909v1)

**作者:** Götz-Henrik Wiegand `[一作]` (Hochschule Karlsruhe University of Applied Sciences), Patrick Baier `[通讯]` (Hochschule Karlsruhe University of Applied Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过微调 Code Llama 7B 并结合 LoRA 与 4 位量化，自动生成面向 DDD 的 JSON 领域模型。

**💡 创新点**

在资源受限环境下实现了可扩展的 LLM 微调方案，并展示了在 DDD 代码生成中的可行性。

**🔧 技术方法**

使用 Code Llama 7B、LoRA、QLoRA 4 位量化、Hugging Face Trainer 以及 BLEU、Loss 等指标。

**📊 数据集**

基于 1,022 条 DDD 项目 JSON 对象的数据集（80% 客户项目，20% 测试项目），并进行了去匿名化预处理。

**📈 对比分析**

通过 BLEU、Loss、语法正确率评估，最终模型在测试集上 BLEU≈0.992，Loss≈0.031，语法正确率约81%，表明性能优良。

**⚠️ 局限性**

受限于 GPU 11GB 及生成 token 上限 4,000，导致部分 JSON 超长或出现语法错误，且仍需进一步提升生成的多样性和无错误率。

---

## 189. FireFly-P: FPGA-Accelerated Spiking Neural Network Plasticity for Robust Adaptive Control

**arXiv ID:** 2601.21222 | [PDF](https://arxiv.org/pdf/2601.21222v1)

**作者:** Tenglong Li `[一作]` (Chinese Academy of Sciences), Yi Zeng `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 10432 | [OpenAlex ID](https://openalex.org/A5108421411)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了基于FPGA的可在芯片上实现突触可塑性的尖峰神经网络控制器（FireFly-P），实现实时自适应机器人控制；

**💡 创新点**

采用两阶段规则优化（离线进化搜索+在线学习）与双引擎流水线FPGA架构，压缩推理与学习延迟至8µs，功耗仅0.713W；

**🔧 技术方法**

使用4项参数化可塑性更新、进化策略离线调参、16位FP16运算、前向+可塑性双引擎、BRAM内存、写优先调度等技术；

**📊 数据集**

在Brax模拟器的连续控制任务（ant、half‑cheetah、ur5e）以及MNIST分类数据集上进行实验；

**📈 对比分析**

与直接训练权重的SNN以及先前硬件STDP实现对比，FireFly-P在控制任务中适应更快、表现更佳，MNIST准确率97.5%，吞吐32FPS，8µs延迟、约10K LUT、0.713W；

**⚠️ 局限性**

受限于FPGA资源适用于中等规模网络，离线进化搜索的可扩展性受限，且主要在仿真环境评估，真实世界鲁棒性与长期稳定性仍待验证。

---

## 190. Breaking the Reasoning Horizon in Entity Alignment Foundation Models

**arXiv ID:** 2601.21174 | [PDF](https://arxiv.org/pdf/2601.21174v1)

**作者:** Yuanning Cui `[一作]` (Nanjing University of Information Science and Technology), Zhangjie Fu `[通讯]` (Nanjing University of Information Science and Technology)

**通讯引用:** 5773 | [OpenAlex ID](https://openalex.org/A5066341740)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种面向零样本知识图谱实体对齐的基础模型（EAFM），通过并行编码和锚点条件信息传递实现跨图、跨语言的高效对齐。

**💡 创新点**

创新点在于：① 针对“推理视野差距”设计并行编码策略，利用种子对齐锚点缩短推理路径；② 构建合并关系图以捕捉跨图全局语义依赖；③ 引入可学习的交互模块和双向分类目标提升匹配精度；④ 在预训练阶段实现参数冻结，支持零样本迁移。

**🔧 技术方法**

技术包括：图神经网络（RelGNN 与 EntGNN）并行双塔结构；锚点条件初始化与基于关系的消息传递；合并关系图构造与注意力机制；交互匹配模块和双向交叉熵损失；可选的实体名称嵌入融合。

**📊 数据集**

使用 OpenEA、SRPRS 以及 DBpedia 系列（D‑W、D‑Y、跨语言）等多种规模、稀疏/稠密、跨源/跨语言的知识图谱对齐基准；在预训练时采用 D‑W‑15K‑V1、D‑W‑15K‑V2、D‑W‑100K‑V1、EN‑DE‑15K‑V1 等源数据。

**📈 对比分析**

与 ULTRA（链接预测版）以及重训练的 ULTRA‑EA 进行对比。EAFM 在 3 个数据组的 MRR 和 Hits@10 上均显著优于基线，尤其在零样本迁移场景下，冻结模型已超过微调模型；在不同密度、规模、异构类型下保持稳健，跨语言任务表现最为突出。

**⚠️ 局限性**

局限性：① 仍以结构为主，对非常稀疏或缺失结构的实体对齐效果有限；② 需要预先提供足够覆盖多源、多语言特征的预训练数据；③ 依赖锚点对齐的质量，对无锚点或锚点不完整的情况尚未充分验证。

---

## 191. Magellan: Autonomous Discovery of Novel Compiler Optimization Heuristics with AlphaEvolve

**arXiv ID:** 2601.21096 | [PDF](https://arxiv.org/pdf/2601.21096v1)

**作者:** Hongzheng Chen `[一作]` (Google), Amir Yazdanbakhsh `[通讯]` (Google DeepMind)

**通讯引用:** 12615 | [OpenAlex ID](https://openalex.org/A5000635267)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大型语言模型与进化搜索、自动调优相结合，自动合成可直接集成到 LLVM 或 XLA 编译器中的 C++ 代码，用来替代手工编写的优化启发式；

**💡 创新点**

创新点在于把优化决策逻辑视为可编程的可执行模板，采用分层搜索把高层策略与低层参数调优分离，形成闭环反馈提升采样效率，并能在实际宏基准上得到端到端的奖励；

**🔧 技术方法**

技术包括 Gemini‑2.5‑Pro/3‑Pro 语言模型、进化算法（交叉、变异）、Vizier 自动调参、LLVM/XLA 编译器集成、宏基准评估（二进制大小、运行时性能、图优化成本）等；

**📊 数据集**

使用内部 LLVM 生产工作负载与宏基准（包含多种应用域的二进制）、XLA 任务（图重写、自动分片）以及公开的机器学习图基准；

**📈 对比分析**

与 LLVM 原生启发式和 MLGO 神经网络策略比较，实验显示在函数内联和寄存器分配上可实现约 5 % 的二进制尺寸缩减，且在大规模基准上超过手工实现 0.6 % 的运行时加速；

**⚠️ 局限性**

局限性包括：编译和评估成本高，导致搜索迭代受限；对性能目标的提升往往停留在零附近，需更大预算或更强 LLM；需要手工指定模板边界，且目前仅在 LLVM/XLA 的特定 Pass 上验证；未来需验证可扩展性与对新架构的迁移能力。

---

## 192. HiFi-Mesh: High-Fidelity Efficient 3D Mesh Generation via Compact Autoregressive Dependence

**arXiv ID:** 2601.21314 | [PDF](https://arxiv.org/pdf/2601.21314v1)

**作者:** Yanfeng Li `[一作]` (Macao Polytechnic University), Yue Sun `[通讯]` (Macao Polytechnic University)

**通讯引用:** 2111 | [OpenAlex ID](https://openalex.org/A5101549114)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 HiFi-Mesh 方法，利用层次化潜在空间与自回归网络（LANE）及 AdaGraph 并行推理，显著提升 3D 网格生成的细节与速度。

**💡 创新点**

创新点：①在自回归生成中引入层次化潜在空间代替长历史序列，减少 6× 生成长度限制；②AdaGraph 通过空间时间解耦实现 300% 的推理加速；③结合可调节的序列长度 L 控制生成细节。

**🔧 技术方法**

核心技术：点云编码器 + 跨注意力上采样；自回归潜在空间构造（CausalAttn）；可学习查询（Learnable Query）驱动的 LANE 块；AdaGraph 的子图重构与并行路径生成；Flash Attention、PyTorch 3D 评测。

**📊 数据集**

使用 Objaverse 数据集进行训练和评测，采用 EdgeRunner 的 1D 序列分词方法（512 类量化）。

**📈 对比分析**

与 MeshAnythingV2、EdgeRunner、TreeMeshGPT 对比，评价指标包括 Chamfer Distance、Normal Consistency、Point‑to‑Mesh Distance、MOS、最大可生成序列长度和推理速度；HiFi‑Mesh 在 CD、PMD、MOS 上领先，最大序列长度 300k，推理速度 302.1 token/s（约 3× 传统方法）。

**⚠️ 局限性**

局限性：Normal Consistency 仍略逊色；在序列长度不足 5k 时的内存/计算开销高于传统方法；模型对极大序列仍受 GPU 内存限制；需要进一步改进分词效率以降低压缩误差。

---

## 193. Understanding Diffusion Models via Ratio-Based Function Approximation with SignReLU Networks

**arXiv ID:** 2601.21242 | [PDF](https://arxiv.org/pdf/2601.21242v1)

**作者:** Luwei Sun `[一作]` (City University of Hong Kong), Han Feng `[通讯]` (City University of Hong Kong)

**通讯引用:** 2731 | [OpenAlex ID](https://openalex.org/A5083194913)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

论文提出了一个理论框架，用 SignReLU 神经网络逼近核诱导的比值函数 f₁/f₂，并将该框架应用到扩散生成模型（DDPM）中，给出了 KL 散度的上界；

**💡 创新点**

创新点在于：①将比值函数逼近问题从单独逼近分子和分母转为整体逼近，并利用 SignReLU 的分段线性结构实现稳定的除法；②在扩散模型中将该比值逼近方法与稀疏时间步、噪声调度等技术结合，得到近似最优的近似误差与统计误差的统一上界；

**🔧 技术方法**

技术主要包括：深度 SigReLU 网络逼近理论、比值函数的分数逼近（division gate），随机过程与随机微积分的应用，经验过程理论中的覆盖数估计，以及 KL 散度的分解与上界推导；

**📊 数据集**

文中未给出具体实验数据集，而是针对一般的 Hölder 连续性假设的目标分布做理论分析，理论结果与已知的最优采样误差一致；

**📈 对比分析**

与传统的 Score‑matching、Normalizing Flow、VAE 等方法相比，论文给出的 KL 上界在理论上实现了接近最优的收敛速率，说明在充分训练样本与合适网络宽度/深度下，扩散模型可达到近似最优的采样性能；

**⚠️ 局限性**

局限性包括：①理论分析假设目标分布满足 Hölder 条件且支持有界；②网络宽度/深度与样本量的关系在实际实现中可能过于保守；③未在实验上验证理论预测，缺乏对真实数据集的实证检验。

---

## 194. LOCUS: Low-Dimensional Model Embeddings for Efficient Model Exploration, Comparison, and Selection

**arXiv ID:** 2601.21082 | [PDF](https://arxiv.org/pdf/2601.21082v1)

**作者:** Shivam Patel `[一作]` (Carnegie Mellon University), Gauri Joshi `[通讯]` (Carnegie Mellon University)

**通讯引用:** 7275 | [OpenAlex ID](https://openalex.org/A5067441201)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于注意力的低维向量模型嵌入方法，能够利用模型在不同查询上的得分来压缩模型能力表示；

**💡 创新点**

创新点在于将评估结果映射为可变长度的token序列，通过自注意力瓶颈网络生成固定维度嵌入，支持无训练的新模型快速加入且嵌入几乎不变；

**🔧 技术方法**

采用多头注意力 Transformer 结构、学习型token化器、潜在瓶颈注意力、学习式聚合以及一个轻量 MLP 正确性预测器；

**📊 数据集**

使用 112 种不同规模、架构的 LLM，在 10 个公开基准（包括数学、常识、推理等）上收集查询及其正确性评估；

**📈 对比分析**

与 EmbedLLM、IRT-Net 等基于梯度学习的嵌入方法对比，LOCUS 在路由准确率和正确性预测上提升显著，样本效率可达 4.8 倍；

**⚠️ 局限性**

局限在于仍依赖大量评估查询来构建嵌入，对评估策略的选择敏感，且在多模态模型和自适应查询选择方面尚未深入研究。

---

## 195. NEXUS: Bit-Exact ANN-to-SNN Equivalence via Neuromorphic Gate Circuits with Surrogate-Free Training

**arXiv ID:** 2601.21279 | [PDF](https://arxiv.org/pdf/2601.21279v1)

**作者:** Zhengzheng Tang `[一作]` `[通讯]` (Boston University), Zhengzheng Tang (Boston University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出NEXUS框架，实现ANN与SNN之间的位级完全等价；

**💡 创新点**

创新点在于使用空间位编码、纯IF神经元门电路构造IEEE‑754浮点运算，并实现无近似的STE训练，达成精确等价；

**🔧 技术方法**

采用空间位编码、IF神经元逻辑门、层级化门电路实现加、乘、除、非线性函数，结合无近似STE进行训练；

**📊 数据集**

在多种语言模型（Qwen3‑0.6B、Llama‑2 7B/70B、Phi‑2、Mistral等）以及标准基准（WikiText‑2、MMLU、HellaSwag、ARC、TruthfulQA）上验证；

**📈 对比分析**

与传统近似SNN（Rate Coding、TTFS、SpikeLLM等）对比，NEXUS保持0.00%精度损失，ULP误差仅6.19，能源消耗在Loihi上可比GPU低58–168,000×；

**⚠️ 局限性**

局限性包括：需完整的IF门电路实现导致硬件实现复杂度高，对极低功耗/极高速度硬件的适配尚待验证，且对极高噪声或阈值偏差的容限虽高但仍有限。

---

## 196. Output-Space Search: Targeting LLM Generations in a Frozen Encoder-Defined Output Space

**arXiv ID:** 2601.21169 | [PDF](https://arxiv.org/pdf/2601.21169v1)

**作者:** Tobias Materzok `[一作]` `[通讯]` (Technische Universität Darmstadt), Tobias Materzok (Technische Universität Darmstadt)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了 Output‑Space Search (OS‑Search)，通过在冻结编码器定义的低维输出空间 Z 中给定目标点 z∗，训练 LLM 控制器使生成结果落在该点附近，从而实现并行多分支生成和基于目标的黑盒优化。

**💡 创新点**

创新点在于将生成任务抽象为状态‑类目标搜索，而非传统路径‑级控制，提供了可外部可检索的目标空间 Z 并通过检索‑根植提示与序列级 RL 训练，使模型可在不改变解码器的前提下精确追踪低维坐标。

**🔧 技术方法**

技术包括冻结的编码器+PCA+Varimax投影构建 Z、检索‑根植提示、序列级 RL（GRPO/GSPO）训练 z∗‑条件控制器、以及在代码域使用 Bayesian Optimization 与在文本域使用网格搜索。

**📊 数据集**

数据集：文本方面使用手写写作提示集（WritingPrompts）并构建 14 个提示模板；代码方面使用 188 条合法实现库（用于构建 Z_code 及 warm‑start）以及 CA++ benchmark；检索库为这些生成或已有的示例。

**📈 对比分析**

对比方法包括基于路径的多分支解码（prompt‑chaining、温度/top‑p/top‑k 探索）与 OS‑Search；在文本域 OS‑Search 在 15 条分支下 Self‑BLEU 下降、ROUGE‑L/ METEOR 降低、LLMScore 提升 3.1 倍；在代码域 Bayesian Optimization 在匹配有效程序预算下 CA++ 得分从 0.371 提升至 0.395。

**⚠️ 局限性**

局限性包括对目标空间 Z 的可移植性依赖、对检索库覆盖度敏感、极端目标可能诱发不安全内容、控制精度受限（非均匀可达性）、仅在受限语料/代码约束下验证，且未公开文本域训练代码/数据。

---

## 197. GeoRC: A Benchmark for Geolocation Reasoning Chains

**arXiv ID:** 2601.21278 | [PDF](https://arxiv.org/pdf/2601.21278v1)

**作者:** Mohit Talreja `[一作]` (Georgia Institute of Technology), James Hays `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 58424 | [OpenAlex ID](https://openalex.org/A5016775802)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

我们创建了GeoRC基准集，收集并整理了三名GeoGuessr世界冠军的地理定位推理链，并用它们来评估VLM生成的推理链。

**💡 创新点**

本工作首次提出了针对地理定位推理链的基准数据集和评分协议，并将LLM与VLM作为评判者来衡量推理链的可审计性和解释性。

**🔧 技术方法**

使用了Vision Language Models（VLM）生成推理链，LLM‑as‑a‑Judge、VLM‑as‑a‑Judge以及关键点引导的评分方法来对推理链进行自动化评估。

**📊 数据集**

使用了800条专家推理链与500个查询场景的GeoGuessr挑战图像，数据来源于Google Street View，覆盖100个挑战、5个位置/挑战。

**📈 对比分析**

通过人类评分与自动评分的平均绝对误差（MAE）比较，发现One‑to‑All LLM‑as‑a‑Judge与人类评分最为接近；人类专家平均F1约54，最好的专有VLM Gemini‑3‑Pro约41，开源VLM平均仅25–31。

**⚠️ 局限性**

主要限制包括专家推理链非完全穷尽、评判依赖固定提示、仅使用英文、未能跑大规模开源模型以及视觉编码分辨率不足导致细节被遗漏。

---

## 198. AC2L-GAD: Active Counterfactual Contrastive Learning for Graph Anomaly Detection

**arXiv ID:** 2601.21171 | [PDF](https://arxiv.org/pdf/2601.21171v1)

**作者:** Kamal Berahmand `[一作]` (RMIT University), Mahdi Jalili `[通讯]` (RMIT University)

**通讯引用:** 8113 | [OpenAlex ID](https://openalex.org/A5030903468)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于主动对抗性反事实对比学习（AC^2L‑GAD）的无监督图异常检测框架；

**💡 创新点**

将主动学习与反事实生成相结合，解决随机增强导致正样本语义不一致以及负样本过于易分的问题；

**🔧 技术方法**

使用结构熵+属性偏差的主动节点筛选、基于梯度近似的特征反事实、贪心结构反事实、GCN编码+投影头、InfoNCE+均匀正则等技术；

**📊 数据集**

在九个基准图（Amazon、Enron、Cora、Citeseer、Flickr、ACM、Pubmed）以及金融交易图 GADBench（T‑Finance、DGraph‑Fin）上进行评测；

**📈 对比分析**

与18个传统、重建式及对比式基线相比，AC^2L‑GAD 在大多数数据集上取得最高或第二高的 AUC/F1，尤其在复杂属性‑结构异常的 ACM、Pubmed 上提升明显；同时主动选择将全图反事实成本降低约65%并保持检测质量；

**⚠️ 局限性**

仅适用于静态节点级异常，反事实生成仍有一定计算开销，且主动选择可能遗漏罕见异常节点。

---

## 199. Power consumption Reduction in ELAA-Assisted ISAC Systems

**arXiv ID:** 2601.21010 | [PDF](https://arxiv.org/pdf/2601.21010v1)

**作者:** Xiaomin Cao `[一作]` (Queen's University Belfast), Michail Matthaiou `[通讯]` (Queen's University Belfast)

**通讯引用:** 13150 | [OpenAlex ID](https://openalex.org/A5035876091)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种能效子阵激活框架，在极大天线阵列(ISAC)中通过选择最佳子阵集合来降低总功耗，同时满足通信与感知的质量服务约束。

**💡 创新点**

创新点在于将子阵激活与功耗最小化统一建模，提出非凸组合优化问题并通过连续松弛与惩罚项结合的SCA（逐步凸化）迭代算法求解，并首次同时考虑近场与远场用户以及目标感知的联合约束。

**🔧 技术方法**

核心技术包括零逼迫预编码、功率控制策略、二进制变量松弛与惩罚、SCA凸化、CVX求解器以及针对信噪比、波束模式增益的下界近似。

**📊 数据集**

研究使用仿真数据：随机生成近场与远场用户位置、距离及大尺度衰落系数，评估不同子阵数、阵列规模和用户配置下的功耗表现。

**📈 对比分析**

与全子阵激活(All‑subarrays)和随机激活(Random)两种基线相比，所提方法在满足相同QoS条件下实现最高约50%的功耗下降，并在不同子阵数和用户组合场景下保持显著优势。

**⚠️ 局限性**

局限性包括：假设完美CSI、静态信道环境；算法在大规模子阵数下仍需多次迭代；未考虑实际硬件非理想与时变信道对激活策略的影响。

---

## 200. BioNIC: Biologically Inspired Neural Network for Image Classification Using Connectomics Principles

**arXiv ID:** 2601.20876 | [PDF](https://arxiv.org/pdf/2601.20876v1)

**作者:** Diya Prasanth `[一作]` (Accel Middle College), Matthew Tivnan `[通讯]` (Massachusetts General Hospital)

**通讯引用:** 199 | [OpenAlex ID](https://openalex.org/A5089321471)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并训练了BioNIC，一个融合了MICrONS提供的单柱层级连接信息的多层前馈神经网络，用于FER-2013面部情绪识别任务。

**💡 创新点**

首次将鼠类视觉皮层的层间/层内连接掩码、梯度抑制、Hebbian可塑性等生物学约束直接映射到网络架构中，并验证其在情绪识别中的可行性。

**🔧 技术方法**

采用卷积层提取空间特征，层归一化、数据增强、标签平滑、Hebbian可塑性、噪声注入、层间/层内掩码、梯度抑制、层次注意力以及学习率调度等一系列生物学启发技术。

**📊 数据集**

使用FER-2013面部表情图像数据集进行训练与评估，并利用MICrONS “minnie65_public”堆栈构建单柱连接矩阵。

**📈 对比分析**

与标准CNN基线在相同训练设置下比较，BioNIC测试准确率为59.77±0.27%，与基线60.16%相近，但在少数类（disgusted、fearful）召回率提升明显；消融实验显示数据增强、卷积层和层归一化是最关键因素。

**⚠️ 局限性**

仅建模单柱层级，未包含视网膜、LGN及更高阶情绪区；对生物学细节的近似和结构与功能组件交互未完整评估；消融实验未考虑多因素交互。

---

## 201. Conditional Generative Framework with Peak-Aware Attention for Robust Chemical Detection under Interferences

**arXiv ID:** 2601.21246 | [PDF](https://arxiv.org/pdf/2601.21246v1)

**作者:** Namkyung Yoon `[一作]` (Korea University), Hwangnam Kim `[通讯]` (Korea University)

**通讯引用:** 2290 | [OpenAlex ID](https://openalex.org/A5028781455)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种基于峰值感知注意力机制的条件生成对抗网络，用于在干扰条件下合成高保真GC‑MS谱图，并将生成的数据用于训练鲁棒的化学品检测模型。

**💡 创新点**

创新点包括：①提出峰值感知注意力机制，突出GC‑MS峰特征；②将该机制嵌入双头多头注意力的CGAN中实现条件生成；③构建SQL数据库与检测框架，实现合成谱与实验谱的无缝集成；④系统性验证合成数据提升检测性能。

**🔧 技术方法**

使用技术：条件生成对抗网络（CGAN），双头多头注意力、峰值感知注意力、短时傅里叶变换（STFT）损失、Transformer+CNN检测网络、SQL数据库管理。

**📊 数据集**

数据集：真实GC‑MS实验数据，涵盖化学武器模拟物（DMMP、DFP、2‑CEES、2‑CEPS）、IED相关化学物（4‑硝基苯酚、乙二胺），与四种溶剂（EtOH、MeOH、MC、THF）以及砖、土壤、草、沥青等干扰物混合测量的谱图。

**📈 对比分析**

比较方法：对生成谱与真实谱进行余弦相似度、皮尔逊相关系数、峰数匹配和3D可视化评估；对检测模型使用准确率、召回率、F1分数。实验结果显示合成谱与真实谱的余弦相似度>0.94、PCC>0.99；随着合成样本量增至>615，检测F1和准确率分别提升至约0.94和0.97。

**⚠️ 局限性**

局限性：对4‑硝基苯酚和乙二胺的检测精度仍偏低；实验未覆盖极端爆炸或高风险组合，且对某些化学物的识别仍需进一步改进。

---

## 202. Beyond a Single Reference: Training and Evaluation with Paraphrases in Sign Language Translation

**arXiv ID:** 2601.21128 | [PDF](https://arxiv.org/pdf/2601.21128v1)

**作者:** Václav Javorek `[一作]` (University of West Bohemia), Ivan Gruber `[通讯]` (University of West Bohemia)

**通讯引用:** 172 | [OpenAlex ID](https://openalex.org/A5002824719)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探索利用大语言模型自动生成书面语言翻译的多种释义（paraphrase），以丰富手语翻译（SLT）语料库，并评估这些释义在训练和评估中的作用。

**💡 创新点**

创新点在于：①提出了用于评估释义质量的 ParaScore 适配方案；②通过实验发现训练阶段使用多释义并未提升性能，但在评估阶段使用多释义显著提高自动指标与人工评价的相关性；③提出了新的 BLEU_para 指标，以多释义作为参考提高评估可靠性。

**🔧 技术方法**

主要技术包括：大语言模型（GPT‑4o‑mini、LLaMA‑3.2‑3B‑Instruct 等）生成释义；ParaScore 结合 BERTScore 与归一化 Levenshtein 距离评估释义；基于 pose 的 T5 模型进行手语翻译；BLEU、BLEURT、ROUGE‑L 等自动评估指标与 BLEU_para。

**📊 数据集**

使用的数据集为公开的手语翻译数据集 YouTubeASL（约 610k 视频）和 How2Sign（约 35k 样本），两者均提供姿态关键点和对应英文翻译。

**📈 对比分析**

实验对比了三种训练策略（仅使用原始翻译、随机采样释义、最小损失释义）以及评估时使用释义或不使用释义。结果显示：训练中使用释义反而略逊于单一参考，而评估阶段使用释义可提升 BLEU、BLEURT、ROUGE‑L 分数，尤其是 BLEU_para 与人工评估的相关性最优。

**⚠️ 局限性**

局限性包括：仅在两种手语数据集与单一任务上验证；释义生成高度依赖 LLM、提示和解码设置，难以完全复现；训练时间与成本限制了释义数量和实验规模；模型与数据的匹配误差（如视频对齐、非手势信息缺失）可能影响结果。

---

## 203. HPTune: Hierarchical Proactive Tuning for Collision-Free Model Predictive Control

**arXiv ID:** 2601.21346 | [PDF](https://arxiv.org/pdf/2601.21346v1)

**作者:** Wei Zuo `[一作]`, Yik-Chung Wu `[通讯]` (University of Hong Kong)

**通讯引用:** 6343 | [OpenAlex ID](https://openalex.org/A5085964667)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了层次化前瞻式调参框架HPTune，利用MPC运动规划中的非执行动作进行主动评估以实现安全增益。

**💡 创新点**

创新点是将闭环评估扩展到非执行动作，并结合快速层与慢速层的层次化调参，使用预测风险指标和Doppler LiDAR速度信息实现主动安全边距更新。

**🔧 技术方法**

采用MPC模型预测控制、预测接近距离与闭合速度风险指标、Doppler LiDAR速度估计、卡尔曼滤波预测、PyTorch进行梯度反向传播等技术。

**📊 数据集**

在CARLA高保真仿真环境中使用随机生成的障碍车辆场景进行实验。

**📈 对比分析**

与DiffTune-MPC（闭环调参）、RDA（开环调参）和OBCA（无调参）对比，HPTune在通行率提升7–46%，并在通行时间、加速度、jerk等指标上显示更平滑、更高效的规划。

**⚠️ 局限性**

受限于仿真环境假设，Doppler LiDAR精度及障碍物预测误差会影响性能，且在极高密度或极端动态场景下仍需进一步验证。

---

## 204. From Logic to Toolchains: An Empirical Study of Bugs in the TypeScript Ecosystem

**arXiv ID:** 2601.21186 | [PDF](https://arxiv.org/pdf/2601.21186v1)

**作者:** TianYi Tang `[一作]` (Simon Fraser University), Nick Sumner `[通讯]` (Simon Fraser University)

**通讯引用:** 607 | [OpenAlex ID](https://openalex.org/A5043431516)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对 TypeScript 开源项目中的 bug 进行大规模实证研究，构建了包含 11 类的新型 bug 分类体系，并量化其在项目规模、领域和依赖复杂度上的分布。

**💡 创新点**

创新点在于首次将 bug 分类与项目级、生态系统级属性关联，并将现代 TypeScript 项目的 bug 分布与早期 JavaScript 研究进行纵向比较，揭示从代码逻辑错误向工具链与配置错误的转移。

**🔧 技术方法**

采用混合方法：人工标注 + 基于句子嵌入的 few‑shot 学习对剩余报错进行自动分类；使用 Spearman ρ 和 Kruskal–Wallis 检验统计关联；构建映射表实现与历史研究的对比。

**📊 数据集**

数据集包含 16 个活跃的 GitHub TypeScript 仓库（共 1,000+ 星标），收集其 bug‑相关提交、issue 与 PR，并在此基础上生成约 4,500 条标注 bug 记录；数据与脚本已公开在 GitHub 上。

**📈 对比分析**

通过对比方法：将本研究的 bug 频率映射到先前 JavaScript 研究的类别上，得到最大对应比例；统计结果显示工具链/配置类占 27.8%、API misuse 14.5%、类型错误 12.4%；相关系数 |ρ|≥0.4，表明项目规模与依赖多样性与这些类别存在显著正相关。

**⚠️ 局限性**

局限性包括：采样偏向大型、活跃的开源项目，可能不适用于小型或闭源系统；人工标注与模型预测仍可能出现误分类；对比方法为定性映射，缺乏精确量化；缺少对因果关系的深入验证。

---

## 205. FrontierScience: Evaluating AI's Ability to Perform Expert-Level Scientific Tasks

**arXiv ID:** 2601.21165 | [PDF](https://arxiv.org/pdf/2601.21165v1)

**作者:** Miles Wang `[一作]` (OpenAI), Tejal Patwardhan `[通讯]` (OpenAI)

**通讯引用:** 119 | [OpenAlex ID](https://openalex.org/A5109819212)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了FrontierScience基准，用专家原创的国际奥赛与博士级科研子问题评估LLM的科学推理能力。

**💡 创新点**

创新点在于：①让国际奥赛奖牌获得者与博士级研究员共同设计原创、难度高、可验证的问题；②引入细粒度的10分rubric评分体系评估开放式科研任务；③使用模型评判器自动化评估并保持低污染。

**🔧 技术方法**

主要技术包括：高推理负载的语言模型（GPT‑5.2、Gemini 3 Pro等）、模型评判器（GPT‑5 “high” reasoning）以及多轮专家审核与迭代完善流程。

**📊 数据集**

使用公开的Gold集：100道奥赛题与60道科研题（覆盖物理、化学、生物），题目均为专家原创且已通过严格审核。

**📈 对比分析**

通过对多款前沿模型的批量评估，GPT‑5.2在奥赛集上取得77%准确率，科研集上25%；与GPT‑4o、Claude Opus 4.5、Gemini 3 Pro等模型相比，GPT‑5.2在奥赛集上最强，科研集上与GPT‑5相当。

**⚠️ 局限性**

局限包括：①仅评估受限问题陈述，缺乏创新提案能力；②rubric评判依赖模型，主观性与一致性仍有待提升；③仅文本评估，无法覆盖图像/实验等多模态科研情境；④未做人类基准比较。

---

## 206. Rectifying Geometry-Induced Similarity Distortions for Real-World Aerial-Ground Person Re-Identification

**arXiv ID:** 2601.21405 | [PDF](https://arxiv.org/pdf/2601.21405v1)

**作者:** Kailash A. Hambarde `[一作]` (Instituto de Telecomunicações), Hugo Proença `[通讯]` (Instituto de Telecomunicações)

**通讯引用:** 4619 | [OpenAlex ID](https://openalex.org/A5090305015)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种几何条件相似性对齐框架，旨在解决空中-地面人员重识别中的几何失真问题。

**💡 创新点**

创新点在于引入了几何诱导查询-键变换（GIQT），通过轻量级的低秩模块显式修正相似性空间，改善了在极端视角和距离变化下的重识别性能。

**🔧 技术方法**

使用了几何条件相似性对齐框架和GIQT模块，结合了变换器架构进行特征提取和相似性计算。

**📊 数据集**

在四个空中-地面人员重识别基准数据集上进行实验，包括AG-ReIDv1、AG-ReIDv2、CARGO和DetReIDX。

**📈 对比分析**

与现有的最先进方法相比，GeoReID在多个基准上表现出更高的Rank-1准确率和mAP，尤其在极端几何条件下的重识别表现显著提升。

**⚠️ 局限性**

限制在于对几何元数据的依赖，尽管提出了视觉基础的几何预测网络，但在某些情况下，几何信息的缺失可能影响性能。

---

## 207. SteerEval: A Framework for Evaluating Steerability with Natural Language Profiles for Recommendation

**arXiv ID:** 2601.21105 | [PDF](https://arxiv.org/pdf/2601.21105v1)

**作者:** Joyce Zhou `[一作]` (Cornell University), Thorsten Joachims `[通讯]` (Cornell University)

**通讯引用:** 52349 | [OpenAlex ID](https://openalex.org/A5014687727)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了SteerEval框架，评估自然语言用户档案在电影推荐中的可驱动性（steerability），通过对用户语料的编辑来改变推荐结果，并系统分析不同模型、文本长度、嵌入与LLM评分、模板与LLM重写等干预方式的效果。

**💡 创新点**

创新点在于：1）构建多维度的可驱动性基准，涵盖从流行类别到细粒度触发警告的75+标签；2）引入AUC_t指标衡量标签相关物品在排名中的提升/降低；3）比较不同干预、模型、标签来源对可驱动性的影响，揭示LLM知识缺口和嵌入模型在减弱任务中的局限。

**🔧 技术方法**

使用的技术包括：预训练LLM（Llama‑3.1‑8B‑Instruct）生成用户档案、编辑和评分；文本嵌入模型（mxbai‑embed‑large‑v1）做相似度排序；模板式提示、LLM附加句子、LLM重写三种干预；统计检验（单尾t检验）评估显著性。

**📊 数据集**

数据集：MovieLens 25M（用户评分）、TMDb（电影标题/描述）以及DoTheDogDie收集的137个触发警告标签（过滤后75个）。

**📈 对比分析**

通过在每个标签+增减任务上随机抽取10名用户，生成100条电影（50相关+50非相关）并比较原始与编辑后的排名。实验表明：默认管线（段落档案+LLM评分）在增减任务上平均ΔAUC_t分别为0.0476和-0.0266，显著高于随机；嵌入排序在增量任务表现更好，但在减量任务失败；LLM重写干预最有效；标签来源（Genre vs Trigger）对可驱动性影响显著，genre更易驱动。

**⚠️ 局限性**

局限性包括：1）仅评估电影领域，缺乏跨域验证；2）使用的标签有限，未覆盖所有用户可能的驱动指令；3）LLM在处理敏感标签时会拒绝或误解；4）未对模型进行微调，可能低估可驱动性潜力；5）未考虑长期用户行为变化和界面交互成本。

---

## 208. Order-Aware Test-Time Adaptation: Leveraging Temporal Dynamics for Robust Streaming Inference

**arXiv ID:** 2601.21012 | [PDF](https://arxiv.org/pdf/2601.21012v1)

**作者:** Young Kyung Kim `[一作]` (Princeton University), Guillermo Sapiro `[通讯]` (Princeton University)

**通讯引用:** 64106 | [OpenAlex ID](https://openalex.org/A5025218580)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了顺序感知的测试时自适应（OATTA）框架，在推理时通过递归贝叶斯估计利用时间动态校正模型输出。

**💡 创新点**

创新点在于将类转移矩阵作为可在线学习的时间先验，并通过似然比门控决定是否使用时间上下文，从而实现轻量、模型无关的顺序自适应。

**🔧 技术方法**

使用了递归贝叶斯滤波、指数移动平均学习转移矩阵、熵门控、似然比门（LLR）以及无梯度的后处理技术。

**📊 数据集**

实验涵盖多模态数据集：图像（CCT、UNSW）、传感器（HARTH、Sleep-EDF）和语言（SENT），以及受控的CIFAR-10、CIFAR-10-C。

**📈 对比分析**

与基线(Base)、MC-Dropout、TTAug、Tent、MEMO等方法对比，OATTA在所有基线上平均提升1–2%，在强时间序列下可提升至6.35%；在弱时间依赖时通过LLR门保持性能不变。

**⚠️ 局限性**

局限性：当时间依赖弱或源模型准确率低时易产生误更新；LLR门需调参；对非马尔可夫或周期性强的转移学习效果有限；仅在输出层操作，无法补偿特征级的分布漂移。

---

## 209. Shortlisting: a Principled Approach

**arXiv ID:** 2601.21277 | [PDF](https://arxiv.org/pdf/2601.21277v1)

**作者:** Edith Elkind `[一作]` (Northwestern University), Lirong Xia `[通讯]` (Rutgers University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

提出了一种统一的短名单制定与投票的两阶段框架，并从公正性、效率、可解释性等角度阐述了短名单设计的基本要求。

**💡 创新点**

将短名单视为单独的决策阶段，构造了恢复性公理、认知效率与预测增强等新颖概念，区分了短名单与多胜者投票的本质。

**🔧 技术方法**

借鉴社会选择理论、多胜者投票、匹配与机器学习预测等技术，提出了理论分析与算法设计思路。

**📊 数据集**

未使用具体数据集，属于蓝海理论性工作。

**📈 对比分析**

未给出实验比较，主要通过定性与极端示例论证。

**⚠️ 局限性**

缺乏实证验证与具体实现，需进一步研究算法复杂度、真实数据测试与策略性行为模型。

---

## 210. Virtualization-based Penetration Testing Study for Detecting Accessibility Abuse Vulnerabilities in Banking Apps in East and Southeast Asia

**arXiv ID:** 2601.21258 | [PDF](https://arxiv.org/pdf/2601.21258v1)

**作者:** Wei Minn `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**通讯引用:** 29948 | [OpenAlex ID](https://openalex.org/A5081036622)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过构建基于虚拟化的渗透测试系统，评估东南亚地区 83 款 Android 银行应用对 FjordPhantom 恶意软件的易受攻击性。

**💡 创新点**

创新点在于首次将虚拟化技术与可访问性滥用攻击结合，系统化评估并提出基于 RASP 的自动化硬化方案。

**🔧 技术方法**

使用的技术包括应用虚拟化、API 钩子、动态加载、APKiD 静态分析及 Promon SHIELD RASP 保护。

**📊 数据集**

数据集为来自 Google Play 的 83 款银行应用及其下载量统计，总下载量超过 4 亿。

**📈 对比分析**

通过对比未加固与加固后应用的崩溃率和可被攻击率，发现 43.37% 的应用易受攻击，且加固措施显著降低易受攻击率。

**⚠️ 局限性**

局限性包括仅针对局部设备的模拟攻击、未考虑用户自行禁用安全服务的情况以及对 RASP 方案细节缺乏深入分析。

---

## 211. Quantifying Noise in Language Generation

**arXiv ID:** 2601.21237 | [PDF](https://arxiv.org/pdf/2601.21237v1)

**作者:** Aaron Li `[一作]` (Harvard University), Ian Zhang `[通讯]` (Duke University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了语言生成框架中噪声对生成能力的影响，证明噪声水平1与任意有限噪声等价，并展示单噪声即可破坏无噪声可生成集合的分离；

**💡 创新点**

首次给出统一与非统一噪声相关可生成性的完整理论表征，揭示噪声水平1即决定所有有限噪声层级，并解决了此前开放的“非统一可生成性是否等价于噪声相关可生成性”问题；

**🔧 技术方法**

采用Kleinberg‑Mullainathan语言生成模型、噪声闭包维度定义、组合论构造与归纳证明等理论工具；

**📊 数据集**

本工作为纯理论分析，没有使用具体数据集；

**📈 对比分析**

没有实验对比，主要通过构造性证明与示例阐释理论边界；

**⚠️ 局限性**

局限在于仅处理有限噪声，未给出对实际LLM鲁棒性的经验指导，且结论主要针对抽象语言集合而非具体语言任务。

---

## 212. Lossless Copyright Protection via Intrinsic Model Fingerprinting

**arXiv ID:** 2601.21252 | [PDF](https://arxiv.org/pdf/2601.21252v1)

**作者:** Lingxiao Chen `[一作]` (Sun Yat-sen University), Xiangyang Luo `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种完全无损、训练无关的“TrajPrint”框架，用于在黑盒 API 下通过检索扩散模型的确定性生成轨迹来验证版权；

**💡 创新点**

其创新点在于利用 DDIM 反向推断锁定模型特定轨迹，并通过双端锚定的联合优化将唯一的轨迹起点映射为模型指纹噪声，从而实现高特异性与鲁棒性；

**🔧 技术方法**

技术核心包括 DDIM 轨迹反演、嵌入可提取水印的 anchor、双端（输出对齐+输入正则化）联合优化以及基于一元 t 检验的统计验证；

**📊 数据集**

实验使用 MS‑COCO 与 WikiArt 生成的多种 anchor，并在 Stable Diffusion v1.5/v2.1/XL、DeciDiffusion、Pixart‑α 等五种潜在扩散模型上验证；

**📈 对比分析**

与随机噪声基线及 FingerInv、Gaussian Shading、StableSignature、SleeperMark 等方法比较，TrajPrint 在同模型验证时 0.96+ 的比特准确率、p 值 <10⁻³，跨模型验证准确率降至 ~0.55，且对 LoRA、DreamBooth、剪枝与量化攻击保持 0.90‑0.99 的鲁棒性；

**⚠️ 局限性**

局限性在于需目标环境支持自定义初始噪声输入，若 API 仅接受默认随机噪声则无法直接使用。

---

## 213. BrainStack: Neuro-MoE with Functionally Guided Expert Routing for EEG-Based Language Decoding

**arXiv ID:** 2601.21148 | [PDF](https://arxiv.org/pdf/2601.21148v1)

**作者:** Ziyi Zhao `[一作]` (University of Technology Sydney), Chin-teng Lin `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `8d10c613-917e-4880-9716-17789f50e119` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出BrainStack，一个功能导向的神经Mixture-of-Experts框架，用多区域专家和全局专家联合解码EEG中的语义信息。

**💡 创新点**

将脑功能模块化分区作为专家设计基础，并引入跨区域蒸馏与自适应专家路由，实现局部与全局特征的动态协同。

**🔧 技术方法**

采用轻量级卷积局部专家、混合卷积+Transformer全局专家、可学习路由门以及分层多目标损失。

**📊 数据集**

构建并公开SS-EEG，120小时、12名受试者、24个静默词的EEG数据。

**📈 对比分析**

与EEGNet、TCNet、EEGConformer、STTransformer及LaBraM等基线在同一数据集上比较，BrainStack平均准确率41.87%，比最强基线提升约12%，并在大多数受试者上取得最高准确率。

**⚠️ 局限性**

受限于EEG信噪比低与个体差异，部分受试者性能仍偏低；模型对跨实验迁移与实时部署的鲁棒性尚待验证。

---

## 214. Large Language Models Naively Recover Ethnicity from Individual Records

**arXiv ID:** 2601.21132 | [PDF](https://arxiv.org/pdf/2601.21132v1)

**作者:** Noah Dasanaike `[一作]` (Harvard University), Noah Dasanaike `[通讯]` (Harvard University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5028961356)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究利用大型语言模型（LLM）对姓名进行种族、族裔、宗教派别或种姓等身份的推断，并与传统的BISG方法在美国、黎巴嫩、印度、乌干达、尼泊尔、亚美尼亚、智利、哥斯达黎加等国进行对比验证。

**💡 创新点**

创新点在于：①无需预先构建姓氏-族群表或手工标注即可通过提示工程直接使用LLM；②能够处理多语言、多脚本姓名，并可灵活加入地理、年龄、政党等元数据；③通过知识蒸馏实现低成本本地部署，兼顾精度与可复现性。

**🔧 技术方法**

主要技术包括：多种LLM（Gemini 3 Flash、GPT‑4o、DeepSeek v3.2、GLM‑4.7、Qwen3系列等）在零样本/无监督提示下进行分类；对比BISG（R包wru）；进行推理（reasoning）实验；使用LoRA对小型Transformer进行蒸馏。

**📊 数据集**

使用的数据集有：美国佛罗里达州、北卡罗来纳州选民文件（自报种族）；黎巴嫩选民登记（宗教派系）；印度国会议员预留选区（种姓/部落）；乌干达、尼泊尔、亚美尼亚、智利、哥斯达黎加等国的全数选民文件；Rajasthan村委会与比哈尔土地记录；以及对应的美国人口普查、联合国统计等基准。

**📈 对比分析**

方法比较采用单样本准确率和召回率；LLM在佛罗里达/北卡上达到84–86%准确率，显著优于BISG（68%）；黎巴嫩宗教派系最高准确率97%（Armenian Orthodox），整体64%；印度种姓预测99%；在聚合层面对比各国人口普查分布误差≤3%（印度）、≤7%（乌干达），GPT‑4o与GLM‑4.7误差相近；蒸馏后4B Qwen模型在美国可达≈79%准确率，接近教师模型。

**⚠️ 局限性**

局限性包括：LLM可能继承训练数据中的种族偏见，导致某些姓名或族群被低估；不同语言/脚本的表现差异明显（黎巴嫩阿拉伯名效果不佳）；元数据与下游结果的内生性问题；推理模式提升不稳定且成本高；开放源模型在阿拉伯名上的性能低；蒸馏后在黎巴嫩的误差显著，需要人工验证和异常检测。

---

## 215. InspecSafe-V1: A Multimodal Benchmark for Safety Assessment in Industrial Inspection Scenarios

**arXiv ID:** 2601.21173 | [PDF](https://arxiv.org/pdf/2601.21173v1)

**作者:** Zeyi Liu `[一作]` (Tsinghua University), Donghua Zhou `[通讯]` (Southeast University)

**通讯引用:** 15414 | [OpenAlex ID](https://openalex.org/A5001517167)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了工业现场多模态安全评估基准数据集 InspecSafe‑V1，并提出基于机器人视角的同步采集与标注框架。

**💡 创新点**

创新点在于将 RGB、热成像、雷达、音频、气体、温湿度等多模态数据与像素级实例分割、场景文本描述和安全等级标签统一集成，填补了传统工业数据集在安全推理方面的空白。

**🔧 技术方法**

采用机器人平台同步采集、固定帧抽取+相似度过滤、像素级分割标注、文本描述生成与安全等级预测，以及 BGE‑M3 文本编码器进行语义相似度评估。

**📊 数据集**

使用新建的 InspecSafe‑V1 数据集，包含 5 个工业场景、41 台机器人、2239 个检查点、5013 个实例、234 个对象类别，采集了 RGB、热像、点云、音频、气体、温湿度等多模态信息。

**📈 对比分析**

通过统一提示模板对多模态视觉语言模型（VLM）进行评估，衡量安全等级预测准确率与生成描述与标注文本的语义相似度；结果显示推理增强型模型精度更高，模型规模与性能不呈严格正相关，误报与误检模式揭示了对视觉噪声和细粒度违规的挑战。

**⚠️ 局限性**

局限性包括：安全评估仅给出离散等级，缺乏连续细粒度描述；数据规模和场景多样性有限，未覆盖极端工况与长序列动态安全评估；多模态对齐和标注覆盖仍有提升空间。

---

## 216. Detecting Multiple Semantic Concerns in Tangled Code Commits

**arXiv ID:** 2601.21298 | [PDF](https://arxiv.org/pdf/2601.21298v1)

**作者:** Beomsu Koh `[一作]` (University of Sheffield), Donghwan Shin `[通讯]` (University of Sheffield)

**通讯引用:** 1059 | [OpenAlex ID](https://openalex.org/A5019085537)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了使用小型语言模型(SLM)检测混杂提交中多重语义关注点的可行性

**💡 创新点**

首次将多标签分类框架应用于混杂提交，构造合成数据集并评估SLM在多关注点场景下的性能，探讨提交信息和token限制对准确率与延迟的影响

**🔧 技术方法**

采用Qwen3-14B模型，使用LoRA进行参数高效微调，基于Chain‑of‑Thought提示和结构化提示实现多标签推理

**📊 数据集**

构造了1750个合成混杂提交（1–5个关注点），来源于经过精炼的CCS标签的真实提交数据，公开数据集与模型在HuggingFace上提供

**📈 对比分析**

与GPT‑4.1对比：微调后的SLM在单关注点时与LLM相当，在最多3个关注点时仍保持可接受的错误率；提交信息的加入可将Hamming Loss下降约44%；在token预算受限下性能基本不变；推理延迟主要随关注点数增长，消息和token截断对延迟影响微乎其微

**⚠️ 局限性**

实验使用合成数据，可能不完全反映真实工业提交的复杂性；模型仅覆盖7类精炼后的CCS标签，无法处理更细粒度或其他类型关注点；截断策略仅为头部保留，未评估其他截断策略

---

## 217. Towards Comprehensive Benchmarking Infrastructure for LLMs In Software Engineering

**arXiv ID:** 2601.21070 | [PDF](https://arxiv.org/pdf/2601.21070v1)

**作者:** Daniel Rodriguez-Cardenas `[一作]` (William and Mary), Denys Poshyvanyk `[通讯]` (William and Mary)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 BEHELM（Benchmarking for Holistic Evaluation of LLMs for Code）框架，旨在统一软件情景描述与多维度评估指标，以全面衡量大语言模型在软件工程任务中的表现。

**💡 创新点**

创新点包括：
- 将软件场景细化为任务、语言、输入/输出粒度和 I/O 类型的五大属性，提供可复现的评估环境；
- 构建包含准确率、效率、可解释性、公平性与鲁棒性等六维度的评估空间，突破传统单一准确率评估的局限；
- 提供标准化的数据工程流水线与元数据跟踪，解决数据泄露、重用成本高等痛点；
- 通过映射现有 Benchmark 到 BEHELM 框架，实现跨 Benchmark 的直接比较与综合报告。

**🔧 技术方法**

采用的技术手段包括：
- 框架设计与模块化实现（软件情景定义、指标计算、评估脚本）;
- 数据预处理与自动化流水线（提取、清洗、去重、验证、版本管理）；
- 多指标计算实现（准确率、执行效率、代码可解释性度量、偏差/公平性分析、鲁棒性测试）。

**📊 数据集**

使用的数据集：
- 代码生成类：HumanEval, MBPP, APPS, LiveCodeBench, ClassEval, EvoCodeBench;
- 代码修复类：SWE‑bench（及其子集），SWE‑bench Java, SWE‑bench Multimodal, OSS‑bench;
- 漏洞检测/修复类：CVEfixes, SecBench.JS, SEC‑bench;
- 测试生成类：TestGenEval, ULT;
- 多任务/代理类：CodeXGLUE, CrossCodeBench, ReCode, Galeras 等。

**📈 对比分析**

比较方法：将上述 Benchmark 的任务与 BEHELM 的软件情景属性进行映射，并在同一多维评估空间中计算对应指标；结果以可视化图表或多指标矩阵形式展示。论文指出，传统 Benchmark 在多指标覆盖方面存在显著缺口，例如大多数只关注准确率或测试通过率，而 BEHELM 能揭示模型在效率、可解释性、公平性及鲁棒性等方面的真实表现。性能方面，论文未给出具体数值，但通过案例演示说明传统模型在单一指标上表现良好，但在 BEHELM 的多维评估下其综合实力会被显著折扣。

**⚠️ 局限性**

局限性：
- BEHELM 仍处于概念/框架层面，缺乏完整公开实现和可复现的数据集；
- 多语言、多任务场景的覆盖尚不完整，需进一步扩充；
- 对实际开发流程（CI/CD、IDE 插件等）的跟踪评估仍有限；
- 需要社区共建与标准化工作，才能真正消除数据泄露与评估偏差。

---

## 218. MapPFN: Learning Causal Perturbation Maps in Context

**arXiv ID:** 2601.21092 | [PDF](https://arxiv.org/pdf/2601.21092v1)

**作者:** Marvin Sextro `[一作]` (Technische Universität Berlin), Gabriel Dernbach `[通讯]` (Technische Universität Berlin)

**通讯引用:** 203 | [OpenAlex ID](https://openalex.org/A5076667962)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 MapPFN，一种先验拟合网络（PFN），利用在合成因果扰动先验上的预训练，通过上下文学习（in‑context learning）在不需要梯度优化的情况下预测单细胞数据中未见生物情境下的扰动效应。

**💡 创新点**

创新点在于：① 将 PFN 与多模态扩散 Transformer (MMDiT) 结合，构建能处理分布映射的生成模型；② 通过构造带有对抗噪声的合成 GRN 先验，实现对因果结构的隐式学习；③ 引入 “Magnitude Ratio” 指标以量化预测效果大小，并揭示传统方法的身份崩塌（identity collapse）问题；④ 在无监督推断中实现零梯度更新，支持实时、可扩展的生物学上下文适配。

**🔧 技术方法**

主要技术包括：先验拟合网络（PFN）框架、上下文学习（ICL）与无梯度推断、MMDiT 生成式网络、对抗式噪声采样、条件流匹配（Conditional Flow Matching）训练目标、Wasserstein 距离与 MMD 分布相似度度量、AUPRC 差异表达基因评估。

**📊 数据集**

使用的数据集：① 合成线性结构因果模型（SCM）与 Erdős–Rényi DAG；② 基于 SERGIO 的合成单细胞基因调控网络（scale‑free GRN），包含技术噪声模拟；③ 真实单细胞扰动数据（CRISPR‑Cas9 248 基因面板），覆盖三种生物背景（未刺激、IFN‑γ 刺激、TIL 共培养）。

**📈 对比分析**

与 CondOT、MetaFM、Identity、Observed 等基线对比；在 few‑shot 以及 zero‑shot 场景下，MapPFN 在 Wasserstein、MMD、RMSE、Rank^⊤、Magnitude Ratio 以及 AUPRC 上均优于或与基线相当；尤其在 AUPRC 上超过传统方法，表明能更准确识别差异表达基因；在零梯度推断场景下保持高比例的效果规模恢复（Magnitude Ratio ≈ 1）。

**⚠️ 局限性**

局限性包括：① 预训练所需的合成先验质量对最终性能影响显著，若先验偏离真实分布会导致性能下降；② 预训练时间较长（≈36 GPU‑小时）；③ 目前仅支持单基因敲除的原子扰动，对多基因或药物化学扰动的泛化能力尚未验证；④ 在高维基因空间（>50 基因）下的扩展与计算成本仍待优化。

---

## 219. Lightweight High-Fidelity Low-Bitrate Talking Face Compression for 3D Video Conference

**arXiv ID:** 2601.21269 | [PDF](https://arxiv.org/pdf/2601.21269v1)

**作者:** Jianglong Li `[一作]` (Shanghai Jiao Tong University), Li Song `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 82520 | [OpenAlex ID](https://openalex.org/A5100448217)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种轻量化、高保真、低比特率的3D说话人面部压缩框架，用FLAME参数与3D Gaussian Splatting实现实时视频会议。

**💡 创新点**

创新点包括：①将FLAME 3DMM与3D Gaussian Splatting相结合，构建可实时驱动的高质量面部表情模型；②只传输面部参数，显著降低比特率；③提出高效Gaussian属性压缩与MLP压缩方案，实现模型尺寸约7倍压缩；④实现超过170 FPS的渲染速度。

**🔧 技术方法**

使用的技术包括：FLAME 3DMM、3D Gaussian Splatting、MLP偏移网络、指数Golomb编码、量化与熵编码、LZ77无损压缩、FP16权重压缩等。

**📊 数据集**

使用公开的FLAME/FlashAvatar数据集以及作者自采集的数据，视频分辨率512×512，约2500帧，测试集占约500帧。

**📈 对比分析**

通过与x265 LDP和NeRF基准在相同帧率（25fps）下的率失真比较，方法在低比特率（<40 kbps）下显著优于x265（PSNR/SSIM/LPIPS更好），压缩后模型仅0.59 MB，渲染速度175 FPS。

**⚠️ 局限性**

局限性：只传输面部参数，背景需要预置或统一处理；多用户同步扩展尚未实现；对网络抖动/丢包的鲁棒性待验证；量化维度与质量的折衷仍需进一步优化。

---

## 220. TimeSliver : Symbolic-Linear Decomposition for Explainable Time Series Classification

**arXiv ID:** 2601.21289 | [PDF](https://arxiv.org/pdf/2601.21289v1)

**作者:** Akash Pandey `[一作]` (Northwestern University), Sinan Keten `[通讯]` (Northwestern University)

**通讯引用:** 10135 | [OpenAlex ID](https://openalex.org/A5014744105)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种利用原始时间序列与符号化表示线性组合，构造可解释时间段重要性评分的深度学习框架；

**💡 创新点**

创新点在于：①将时间序列分段后同时学习其潜在向量 Q 与符号分布向量 Z，②通过符号与潜在特征的线性交互矩阵 P 保留全局信息且无序列长度依赖，③用无参数的 f_att 函数直接从 P、Z、Q 计算正负时间重要性得分，显著提升可解释性与预测精度；

**🔧 技术方法**

技术上采用一维卷积提取局部潜在表示、离散化（binning）得到符号化矩阵、平均池化、符号-潜在交互矩阵、线性投影层以及非参数梯度归一化的时间重要性计算；

**📊 数据集**

在四个合成数据集（FreqSum、SeqComb-UV、SeqComb-MV、LowVar）以及三大实际数据集（EEG 失眠分期、FordA 机械故障诊断、ESC‑50 语音识别）验证，并在 UEA 26 维多变量时间序列分类基准上进行性能对比；

**📈 对比分析**

与 12 种传统可解释方法（Grad‑CAM、DeepLift、Integrated Gradients、KernelSHAP、LIME 等）和 16 类分类基线（DTW、MUSE、ResNet、InceptionTime 等）对比，AUPRC 平均提升约 18%（合成数据）或 11%（实际数据），在 UEA 基准中准确率与最佳方法差距不超过 2%，排名平均第 2；

**⚠️ 局限性**

局限性包括：①需要选择分段长度、符号数等超参，过大或过小会影响可解释性和准确性；②线性交互可能无法捕捉更复杂的非线性时序依赖；③符号化可能导致信息丢失，尤其对细粒度时间特征；④仅在时间段级别提供解释，未扩展到特征级别或更细粒度；⑤缺乏与领域专家的交互验证。

---

## 221. Top-k on a Budget: Adaptive Ranking with Weak and Strong Oracles

**arXiv ID:** 2601.20989 | [PDF](https://arxiv.org/pdf/2601.20989v1)

**作者:** Lutz Oettershagen `[一作]` (University of Liverpool), Lutz Oettershagen `[通讯]` (University of Liverpool)

**通讯引用:** 73 | [OpenAlex ID](https://openalex.org/A5041881627)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了在弱oracle（低成本噪声估计）和强oracle（高成本精确评估）双源框架，针对未知评分函数的top-k识别任务，设计了PAC级别的top-k证书算法，并通过自适应机制显著降低强oracle调用次数。

**💡 创新点**

创新点主要包括：
1) 形式化弱/强oracle PAC top-k问题，并给出基线screen‑then‑certify（STC）算法；
2) 设计Adaptive Certification of the Exact top‑k（ACE）算法，仅对决策边界项进行强oracle查询，理论上与基线相当但实践更高效；
3) 进一步加入Adaptive Weak Allocation（AWA）自适应分配弱oracle预算，形成ACE‑W算法，进一步减少强oracle调用；
4) 对强oracle调用数给出上界O(m(4ε_max))和下界Ω(m(ε_max))，实现常数因子匹配；

**🔧 技术方法**

核心技术包括：
- 置信区间（empirical Bernstein、anytime‑valid CI）实现联合覆盖；
- 近似阈值（near‑tie mass m(η)）分析强oracle调用需求；
- LUCB/最大堆与秩统计树实现自适应挑选最关键的弱/强oracle查询；
- 时间统一置信序列在弱oracle自适应阶段确保PAC保证；

**📊 数据集**

实验使用两类数据集：
1) 合成数据（随机生成的评分分布，控制gap、σ等参数）；
2) 真实文本分类任务（20 Newsgroups）中的文档Shapley值估计，用MC采样作为弱/强oracle，检验算法在数据价值识别中的效果。

**📈 对比分析**

与STC、TA‑Certify等基线方法比较：
- 在合成实验中，ACE/ACE‑W的强oracle调用量仅为STC的约1/3–1/4，且在n、k、gap变化下保持更平滑的增长；
- 在20 Newsgroups实验中，ACE从83.5次降至33.3次，ACE‑W进一步降至29.4次，速度提升约2.4–2.8倍；
- 证明了算法在实践中远低于理论上界，并显著优于非自适应基线。

**⚠️ 局限性**

局限性：
- 仍存在常数4的误差窗口差距，理论与实践之间的常数因子尚未完全收敛；
- 结果高度依赖弱oracle的精度和near‑tie mass，对极端分布（多项值聚集）可能失效；
- 需要预先设定k，无法直接适用于动态top‑k或阈值式检索问题；
- 对弱oracle预算分配的停止准则尚未完全自适应，仍需经验调参。

---

## 222. EHR-RAG: Bridging Long-Horizon Structured Electronic Health Records and Large Language Models via Enhanced Retrieval-Augmented Generation

**arXiv ID:** 2601.21340 | [PDF](https://arxiv.org/pdf/2601.21340v1)

**作者:** Lang Cao `[一作]` (University of Illinois Urbana-Champaign), Yue Guo `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 7620 | [OpenAlex ID](https://openalex.org/A5101649401)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 EHR-RAG，一种增强检索增量生成框架，用于在严格上下文窗口下通过检索结构化电子健康记录（EHR）支持大语言模型（LLM）完成长时段临床预测任务。

**💡 创新点**

核心创新点包括：
1) Event‑ and Time‑Aware Hybrid Retrieval（ETHER）— 在检索时分别处理数值事件和文本事件，保留事件类型与时间结构；
2) Adaptive Iterative Retrieval（AIR）— 通过 LLM 逐步细化检索查询，实现对稀疏、分散的证据的动态覆盖；
3) Dual‑Path Evidence Retrieval and Reasoning（DER）— 同时检索支持与反证据，双路径推理减少偏见并提升鲁棒性。

**🔧 技术方法**

技术方法：
- Retrieval‑Augmented Generation（RAG）与密集向量检索；
- U‑shaped 时间权重模型，兼顾早期与近期事件；
- LLM 驱动的查询细化与证据合并；
- 双路径推理框架，融合事实与反事实证据；
- 在 GPT‑5、Claude‑Opus‑4.5、LLaMA‑3.1‑8B 等多种 LLM 上实现。

**📊 数据集**

实验数据集：EHRSHOT 基准，覆盖多科室、跨时间跨度（多年）的大量结构化 EHR；任务包括长住院期、30 天再入院、急性心肌梗死、贫血四类预测。

**📈 对比分析**

与直接生成、vanilla RAG、Uniform RAG、Rule‑based RAG、ReAct RAG 等基线，以及传统机器学习基线（计数‑LR、CLMBR‑LR）进行对比。EHR‑RAG 在四个任务上平均提升 Macro‑F1 10.76%，单任务提升 3.6%–16.5%，在低样本和全数据场景均优于或持平传统 ML 方法。

**⚠️ 局限性**

局限性：
- 仍依赖大规模 LLM，推理与检索成本高；
- 仅在结构化 EHR 上评估，未涉及临床文本或多模态数据；
- 需要预先构建并维护向量索引，适配性与迁移性待验证；
- 在真实临床部署与监管合规性方面缺乏验证。

---

## 223. The Epistemic Planning Domain Definition Language: Official Guideline

**arXiv ID:** 2601.20969 | [PDF](https://arxiv.org/pdf/2601.20969v1)

**作者:** Alessandro Burigana `[一作]` (Free University of Bozen-Bolzano), Francesco Fabiano `[通讯]` (University of Oxford)

**通讯引用:** 493 | [OpenAlex ID](https://openalex.org/A5025902208)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种名为 epddl 的统一领域定义语言，用于描述多代理知识与信念规划任务，并在 IPC‑26 竞赛中作为官方基准语言。

**💡 创新点**

创新点包括：
• 引入抽象事件模型和抽象知识行动，对可观测性类型进行抽象，消除对每种可观测组合的重复定义；
• 在 epddl 中新增动作类型库（action type library）这一第三层抽象，使得同一类动作的多点抽象帧可被共享；
• 给出基于 DEL 的正式语义，并证明抽象行动与标准行动的等价性，保证了语言与现有规划器的一致性。

**🔧 技术方法**

技术手段主要是：
• 基于 Dynamic Epistemic Logic（DEL）和抽象事件模型的形式化定义；
• 采用 PDDL 风格的 LISP 语法实现解析器；
• 通过定义动作类型库和抽象帧来简化动作声明；
• 证明抽象更新（abstract product update）与标准更新（product update）的等价性。

**📊 数据集**

使用 IPC‑26 竞赛的知识规划基准（如 Epistemic Blocks World 等）作为示例和评测集合；并未发布新的公开数据集。

**📈 对比分析**

比较方法：
• 将现有基于不同 DEL 子语言的规划器映射到对应的动作类型库；
• 在相同初始状态和目标下对比计划长度、求解时间和成功率；
• 结果显示，epddl 能够统一表示这些子语言，且在小规模任务上的性能与现有实现相近，同时提供更高的可复现性和可扩展性。

**⚠️ 局限性**

局限性：
• 对大规模、复杂可观测性组合的性能评估尚未充分；
• 抽象事件模型的构造仍需手工编写，对用户友好性有一定挑战；
• 目前仅在 IPC‑26 基准上验证，缺乏跨领域或更广泛的实际应用测试。

---

## 224. DA-SPS: A Dual-stage Network based on Singular Spectrum Analysis, Patching-strategy and Spearman-correlation for Multivariate Time-series Prediction

**arXiv ID:** 2601.21381 | [PDF](https://arxiv.org/pdf/2601.21381v1)

**作者:** Tianhao Zhang `[一作]` (University of Science and Technology of China), Yun-Bo Zhao `[通讯]` (University of Science and Technology of China)

**通讯引用:** 2570 | [OpenAlex ID](https://openalex.org/A5018474051)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个双阶段网络 DA‑SPS，用于多变量时间序列预测，分别针对目标变量和外生变量进行处理。

**💡 创新点**

创新点包括：①利用 Singular Spectrum Analysis (SSA) 对目标序列进行趋势、季节和噪声分解；②引入 Patching‑Conv‑LSTM 对趋势组件进行局部特征提取；③使用 Spearman 相关系数筛选与目标变量高度相关的外生变量；④构建 L‑Attention（基于 LSTM 的双注意力机制）融合外生变量信息。

**🔧 技术方法**

核心技术包括：SSA、Patch 预处理、Conv‑LSTM、传统 LSTM、Spearman 相关分析、L‑Attention（注意力+LSTM）、线性映射融合。

**📊 数据集**

实验数据集：四个公开数据集（Electricity、Solar、Traffic、Exchange）以及私有笔记本主板 Yield 数据集。

**📈 对比分析**

在 3/6/12/24 步长下，与 CNN、LSTM、DA‑RNN、LSTNet、DA‑Conv‑LSTM、TS‑Conv‑LSTM、TCLN 等基线模型在 MAE、RMSE、RSE、CORR 等指标上对比，DA‑SPS 在绝大多数指标上取得显著更低误差和更高相关性。

**⚠️ 局限性**

局限性：模型对超参数（如 SSA 窗口、Spearman 阈值）敏感；计算量较大；外生变量相关性判定可能受噪声干扰，且尚未在更大规模或跨领域数据上进行充分验证。

---

## 225. Bayesian-LoRA: Probabilistic Low-Rank Adaptation of Large Language Models

**arXiv ID:** 2601.21003 | [PDF](https://arxiv.org/pdf/2601.21003v1)

**作者:** Moule Lin `[一作]` (Trinity College Dublin), Goetz Botterweck `[通讯]` (Trinity College Dublin)

**通讯引用:** 1765 | [OpenAlex ID](https://openalex.org/A5031243998)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Bayesian-LoRA的贝叶斯低秩适配方法，用于在保持参数高效的前提下改进大语言模型在细调过程中的概率校准与不确定性量化。

**💡 创新点**

核心创新是发现LoRA的低秩分解与稀疏高斯过程（Sparse Gaussian Process）的Kronecker化后验存在结构同构，将LoRA视为贝叶斯模型的极限情形，并通过正则化的变分后验和可逆流（normalizing flow）实现对权重空间的不确定性建模。

**🔧 技术方法**

技术包括：变分贝叶斯推断、稀疏高斯过程诱导变量、Kronecker 乘法投影、可逆自回归流（MAF）增强后验、端到端训练的ELBO优化以及与LoRA相似的低秩矩阵更新。

**📊 数据集**

使用多种数据集验证：commonsense reasoning benchmark（WG-S、ARC-C/E、WG-M、OBQA、BoolQ）、WikiText-2文本生成、数学推理 benchmark MATH，以及大规模模型（LLaMA 2-7B、Qwen2.5-14B-Instruct、Qwen3-30B-A3B）。

**📈 对比分析**

与LoRA、Dropout、温度标度、Ckpt Ens、Deep Ensembling、BBB、BLoB、后验调整方法（LA、LLLA）等对比，Bayesian-LoRA在保持接近LoRA的精度的同时，实现了最多84% ECE降低、76% NLL降低，训练时间仅略高于LoRA（≈1.23×），内存占用与参数量几乎不变。

**⚠️ 局限性**

局限性包括：每层诱导矩阵独立建模，未考虑层间相关性；对超参数（如流深度、诱导维度）敏感；在极端数据分布漂移下，后验近似仍可能不足；未来工作需探索跨模态、指令调优、RLHF等场景，以及更严格的理论保证。

---

## 226. Diversifying Toxicity Search in Large Language Models Through Speciation

**arXiv ID:** 2601.20981 | [PDF](https://arxiv.org/pdf/2601.20981v1)

**作者:** Onkar Shelar `[一作]` (Rochester Institute of Technology), Travis Desell `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 1174 | [OpenAlex ID](https://openalex.org/A5065630093)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在 ToxSearch 基础上实现无监督种群分化的质量‑多样性搜索 ToxSearch‑S，能够并行维护多种高毒性提示领域。

**💡 创新点**

创新点在于将无监督的领导‑跟随聚类、容量限制、预留池和种群级父母选择相结合，形成在线种群分化的 QD 机制，显著提升极端毒性发现与语义多样性。

**🔧 技术方法**

使用演化算法（steady‑state (μ+λ)）与无监督领导‑跟随聚类，嵌入式距离（语义+毒性）以及 Perspective API 作为毒性判别器，Prompt/Response 采用 Llama 3.1‑8B 模型。

**📊 数据集**

使用公开的 CategoricalHarmfulQA 与 HarmfulQA 训练集生成 2 481 个有害问题提示，随机抽取 100 个做实验种子。

**📈 对比分析**

与基线 ToxSearch 进行对比，在相同 50 代预算下，ToxSearch‑S 的最大毒性 0.73 对比 0.47，极端尾部中位数 0.66 对比 0.44，语义覆盖度提升，整体性能显著提升。

**⚠️ 局限性**

局限包括只使用单一毒性判别器、短期实验（50 代）与有限种子、仅评估一组 LLM 对象、以及种群分化参数与预留池管理可能导致不稳定，需进一步在更长时间、更广泛模型与多源毒性判别器上验证。

---

## 227. Latent Chain-of-Thought as Planning: Decoupling Reasoning from Verbalization

**arXiv ID:** 2601.21358 | [PDF](https://arxiv.org/pdf/2601.21358v1)

**作者:** Jiecong Wang `[一作]` (Beihang University), Chunyang Liu `[通讯]` (Didi Chuxing)

**通讯引用:** 3099 | [OpenAlex ID](https://openalex.org/A5011341171)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 PLaT 框架，将 LLM 的推理拆分为连续隐状态规划（Planner）与语言化解码（Decoder），实现动态终止与可解释的隐状态；

**💡 创新点**

核心创新是把推理过程从离散 Token 拆分到连续隐空间，并将思考与语言化解耦，既能保持多路径探索，又能随时决定终止；

**🔧 技术方法**

技术方案包括 GPT‑2 Small Transformer 作为主体，Planner‑Decoder 两模块、EMA 聚合、Lazy Decoding、以及 GRPO 强化学习对解码策略进行微调；

**📊 数据集**

主要使用 GSM8k‑Aug 进行训练，评测数据集包括 GSM‑HARD、SVAMP、MultiArith 等 OOD 任务；

**📈 对比分析**

与 CoT‑SFT、Coconut、CODI 等基线比较，PLaT 在 Pass@k（k=32/64/128）指标上表现最优，greedy 精度略低于基线，但推理速度比显式 CoT 快、比其他隐层方法相对均衡；

**⚠️ 局限性**

局限性在于：greedy 准确率仍落后于 CoT；RL 训练易过拟合、受 GPT‑2 Small 参数限制；对更复杂推理场景的容量与泛化仍有限。

---

## 228. CausalEmbed: Auto-Regressive Multi-Vector Generation in Latent Space for Visual Document Embedding

**arXiv ID:** 2601.21262 | [PDF](https://arxiv.org/pdf/2601.21262v1)

**作者:** Jiahao Huo `[一作]` (Hong Kong University of Science and Technology), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1202 | [OpenAlex ID](https://openalex.org/A5057914558)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种自回归生成多向量嵌入框架（CausalQwen/CausalPali），在视觉文档检索中以几十个视觉标记即可替代传统的数百/千个补丁级向量，从而大幅压缩存储和检索开销。

**💡 创新点**

创新点包括：①使用自回归语言模型按序列生成潜在向量；②在对比训练中加入迭代边际损失、进步细化损失和多样性正则，促使生成向量既紧凑又分布良好；③发现自回归嵌入具有显式的测试时可伸缩性，推理时可动态调节令牌数量以平衡精度与延迟。

**🔧 技术方法**

核心技术包括：多模态预训练模型（如Qwen‑VL、PaliGemma）→自回归生成模块；对比学习与多项正则化（margin、diversity、progressive refinement）；late‑interaction 评分；KV 缓存提升推理速度；实验采用 nDCG@5 指标评估检索性能。

**📊 数据集**

使用公开的 VDR 基准数据集：V1、V2、V3；训练集来自 vidore/colpali_train_set；预训练的双编码器 Qwen2‑VL‑3B、PaliGemma‑3B 等作为骨干网络。

**📈 对比分析**

与基于裁剪/聚类的多向量方法和单向量 Bi‑Encoder 进行对比。结果显示：在相同压缩比例（30×–155×）下，CausalQwen/CausalPali 的 nDCG@5 与甚至优于原始 ColPali、ColQwen；在 V2/V3 上平均提升 5–20%，在 V1 上差距仅 0.5%；且随测试时令牌数增大性能呈现稳定上升，证明其可伸缩性。

**⚠️ 局限性**

局限性：①仍依赖大规模预训练 MLLM，需额外微调成本；②当令牌预算过低（<8）时检索效果明显下降；③在极端多语言或高度专业化领域的泛化尚未充分验证；④实验主要集中在视觉文档检索，未探讨其它多模态任务的适用性。

---

## 229. EnsembleLink: Accurate Record Linkage Without Training Data

**arXiv ID:** 2601.21138 | [PDF](https://arxiv.org/pdf/2601.21138v1)

**作者:** Noah Dasanaike `[一作]` (Harvard University), Noah Dasanaike `[通讯]` (Harvard University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5028961356)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 EnsembleLink，一种零样本记录匹配方法。

**💡 创新点**

创新点在于结合稠密检索与交叉编码重排序，利用预训练语言模型的世界知识完成无监督匹配。

**🔧 技术方法**

使用 Qwen3-Embedding-0.6B 进行向量检索，Jina Reranker v2 Multilingual 进行重排序，必要时再使用 Qwen3-8B 作为 LLM 复核。

**📊 数据集**

在城市名称、个人姓名、组织名称、跨语种政党、以及 DBLP-Scholar 记录集上进行评测。

**📈 对比分析**

与 fastLink 与 fuzzylink 对比，EnsembleLink 在四项基准任务上 top‑1 准确率均超过对手，且无须标注或外部 API。

**⚠️ 局限性**

局限在于对极大规模语料的检索仍需额外过滤，跨语种任务对 LLM 的依赖导致计算量增加，且对完全不同领域的文本仍需验证。

---

## 230. Deep QP Safety Filter: Model-free Learning for Reachability-based Safety Filter

**arXiv ID:** 2601.21297 | [PDF](https://arxiv.org/pdf/2601.21297v1)

**作者:** Byeongjun Kim `[一作]` (Seoul National University), H. Jin Kim `[通讯]` (Seoul National University)

**通讯引用:** 7084 | [OpenAlex ID](https://openalex.org/A5073996122)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于深度学习的全数据驱动安全滤波器Deep QP Safety Filter，能在黑盒动力学系统上实现安全控制；

**💡 创新点**

创新点在于将时间折扣的Hamilton–Jacobi到达性问题转化为可学习的Bellman算子，使用无模型学习同时保证收敛，且通过可调参数α实现滤波激进程度；

**🔧 技术方法**

采用Hamilton–Jacobi到达性、时间折扣、可学习的安全价值函数与其导数的深度神经网络、约束式QP滤波器以及离线/在线强化学习框架；

**📊 数据集**

在多种动力学系统上验证，包括双积分器、倒立摆、倒立双摆、Hopper等Gymnasium环境；

**📈 对比分析**

与多种基线（PPO、PPO-Lagrangian、RL-DH、无模型安全滤波器等）相比，Deep QP Safety Filter在安全性更低保守、Q‑P不可行率几乎为零、训练期间失败率显著下降，且在RL任务中获得更高奖励；

**⚠️ 局限性**

限制主要包括：对折扣参数λ的调度敏感、在初始训练阶段Q‑P可能不可行、在极端不光滑或离散跳变动力学下理论假设受限但实验仍表现稳健。

---

## 231. What You Feel Is Not What They See: On Predicting Self-Reported Emotion from Third-Party Observer Labels

**arXiv ID:** 2601.21130 | [PDF](https://arxiv.org/pdf/2601.21130v1)

**作者:** Yara El-Tawil `[一作]` (University of Michigan), Emily Mower Provost `[通讯]` (University of Michigan)

**通讯引用:** 2831 | [OpenAlex ID](https://openalex.org/A5003136334)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

跨语料库评估第三方情感识别模型在自我报告情绪上的泛化能力，探究激活与价值维度的可预测性，并分析个人意义对模型表现的影响。

**💡 创新点**

首次在不同情绪数据集间进行跨语料库自我报告预测实验，揭示激活难以预测但价值可在高个人意义样本中实现较高相关性；提出个人意义作为对齐外部感知与内部体验的关键途径。

**🔧 技术方法**

使用自监督音频编码器 WavLM、文本编码器 RoBERTa‑Longformer/ModernBERT 以及开源大语言模型 GPT‑20B 和 Qwen3‑32B；训练采用 CCC 损失并在混合训练（Podcast+IEMOCAP）和零射模型评估中应用。

**📊 数据集**

主要使用 MSP‑Podcast（第三方标签）、IEMOCAP（自我报告与第三方标签）和 MuSE（自我报告），并利用 GPT‑20B 评估个人意义得分。

**📈 对比分析**

通过在第三方标签上的跨语料库评估以及在自我报告上的评估（Seg‑Mono 与 Avg‑Segs 聚合），对比 WavLM、Longformer、ModernBERT 以及 LLMs 的 CCC 分数。结果显示激活在自我报告中 CCC ≈ 0，价值约 0.3；在个人意义高的样本中价值 CCC 可升至 0.6‑0.8，且混合训练在 MuSE 上表现不佳。

**⚠️ 局限性**

自我报告样本稀缺、与第三方标签的系统性不一致，激活维度难以捕捉；个人意义评估仅基于 GPT，缺乏专家注释；实验仅限于三大数据集，未覆盖多语言或更大规模数据；LLM 的零射性能受模型规模与推理时间限制。

---

## 232. Do LLMs Favor LLMs? Quantifying Interaction Effects in Peer Review

**arXiv ID:** 2601.20920 | [PDF](https://arxiv.org/pdf/2601.20920v1)

**作者:** Vibhhu Sharma `[一作]` (Cornell University), Sarah Dean `[通讯]` (Cornell University)

**通讯引用:** 7296 | [OpenAlex ID](https://openalex.org/A5034027561)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对ICLR、NeurIPS、ICML 2024-2025三大会议的论文–评审数据进行系统分析，首次研究作者和评审双方使用大型语言模型（LLM）时的相互作用效应；

**💡 创新点**

创新点在于：①将LLM使用与评审质量、决策过程结合，揭示不同质量水平下的交互效应；②通过合成完全LLM生成的评审和元评审，对比人类介入下的LLM协助效果；③结合回归、配对分析和逻辑回归，量化LLM对评分压缩、决策偏好等影响；

**🔧 技术方法**

主要技术包括：①基于词袋混合模型的α‑检测方法识别LLM文本；②线性回归（含交互项）和分区回归消除论文质量混杂；③配对差异分析和Mann‑Whitney U检验；④逻辑回归评估元评审的影响；⑤使用GPT‑4o生成全自动评审和元评审的合成实验；

**📊 数据集**

数据集：OpenReview公开的论文全文、评审文本、元评审文本，覆盖ICLR 2024/25、NeurIPS 2024/25、ICML 2025，共约125k论文–评审对；另外抽样部分论文进行GPT‑4o合成评审/元评审。

**📈 对比分析**

对比方法：将LLM‑辅助评审与人类评审、完全LLM评审在评分分布、对论文质量的影响以及对最终接受率的关联性进行比较。实验结果显示：LLM‑辅助评审对低质量论文更宽容，导致LLM‑论文在该区间获得更高分；完全LLM评审呈评分压缩，几乎全部集中在6-7分；LLM‑辅助元评审相较于人类元评审更倾向于接受；完全LLM元评审则更严厉。

**⚠️ 局限性**

限制包括：①检测方法α可能对不同LLM模型敏感，低估/高估使用率；②研究仅限于三大会议，缺乏跨学科或期刊验证；③回归假设条件（如无干扰、条件可忽略）可能不完全成立；④合成实验受GPT‑4o偏好与提示设计影响；⑤缺乏对作者、评审实际使用过程的直接观察。

---

## 233. Alliance Mechanisms in General Lotto Games

**arXiv ID:** 2601.21319 | [PDF](https://arxiv.org/pdf/2601.21319v1)

**作者:** Vade Shah `[一作]` (University of California), Jason R. Marden `[通讯]` (University of California)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

研究了在Coalitional General Lotto游戏中，预算转移、竞赛转移和联合转移三种联盟机制在实现互利与集体收益时的效果差异。

**💡 创新点**

首次在同一理论框架下对三种机制的互利机会进行区分性分析，并证明联合转移几乎在所有游戏实例中都能产生互利，而预算与竞赛转移仅在有限且部分重叠的子空间内有效；同时证明在集体收益目标下三种机制等价。

**🔧 技术方法**

使用博弈论的纳什均衡分析、测度理论与梯度对比方法，结合对联盟转移参数空间的分块解析和极限条件推导。

**📊 数据集**

无实验数据集，全部基于理论推导与数学证明。

**📈 对比分析**

通过构造三类转移参数空间并计算其对玩家收益的影响，比较了互利和集体收益两种目标下的可行区域；结果显示：1）互利目标下，联合转移几乎覆盖全部游戏实例；2）集体收益目标下，三种机制在可行空间与收益提升幅度上完全一致，任何单一转移即可达到最大集体收益。

**⚠️ 局限性**

局限性包括仅考虑两玩家与单一对手，假设预算与竞赛价值连续可分；未涉及网络多玩家环境、信息不完全或动态竞争；缺乏实证或实验验证。

---

## 234. An AI Framework for Microanastomosis Motion Assessment

**arXiv ID:** 2601.21120 | [PDF](https://arxiv.org/pdf/2601.21120v1)

**作者:** Yan Meng `[一作]` (Children's National Hospital), Daniel A. Donoho `[通讯]` (Children's National Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究构建了一套基于AI的微血管吻合手术器械处理技术评估框架。

**💡 创新点**

结合YOLOv11+DeepSORT、形状描述符定位与监督分类，实现了实时、客观的手术技术评分。

**🔧 技术方法**

使用YOLOv11进行器械检测，DeepSORT实现跟踪，形状描述符定位刀尖，Gradient Boosting分类模型评估技能。

**📊 数据集**

基于九名医学生完成的63段微血管吻合视频，包含1.0mm×0.8mm卡片和标准器械，手工标注并使用Encord平台。

**📈 对比分析**

与单独YOLOv11对比，检测精度提升至96.9% mAP，跟踪恢复率98.7%，技能分类准确率87%。

**⚠️ 局限性**

受限于仅二维定位、样本量小及专家评分主观性，导致对“差”级别识别不足。

---

## 235. An introductory Generalization of the standard SVMs loss and its applications to Shallow and Deep Neural Networks

**arXiv ID:** 2601.21331 | [PDF](https://arxiv.org/pdf/2601.21331v1)

**作者:** Filippo Portera `[一作]` `[通讯]`, Filippo Portera

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于模式相关性的凸通用损失函数，并推导其在SVM二分类、SVR回归以及浅深神经网络中的对偶形式，随后在多组小型数据集和网络上进行了实验验证。

**💡 创新点**

创新点在于将样本相似矩阵S的平方根误差项融入损失，形成可凸化的通用损失，能够统一处理SVM和NN，并提供多种构造S矩阵的方法。

**🔧 技术方法**

采用SVM对偶推导、SMO/工作集选择策略、核方法，结合PyTorch深度学习框架，使用Graph2Vec、GIN、CNN等网络实现损失计算。

**📊 数据集**

使用了UCI/Kaggle小数据集（Sonar、Haberman、Heart、Iono、Breast、WDBC、German、Wine等）、图数据集（PROTEINS、IMDB‑BINARY、NCI）、ZINC分子图、面部口罩图像及CIFAR‑10。

**📈 对比分析**

通过5折交叉验证比较F1、MSE、准确率等指标，实验显示新损失在多数数据集上性能不劣于标准损失，并在部分数据集（如Haberman、Iono）显著提升，但在大规模数据集上提升有限且计算时间显著增加。

**⚠️ 局限性**

主要局限在于计算S矩阵导致时间与空间复杂度高，尤其在大批量或大样本时不可扩展；目前效果仅在小规模或局部S子矩阵上可行，缺乏对更大数据或其他任务的普适性验证。

---

## 236. Dynamical Adapter Fusion: Constructing A Global Adapter for Pre-Trained Model-based Class-Incremental Learning

**arXiv ID:** 2601.21341 | [PDF](https://arxiv.org/pdf/2601.21341v1)

**作者:** Ruiqi Liu `[一作]` (Institute of Computing Technology), Yongjun Xu `[通讯]` (Institute of Computing Technology)

**通讯引用:** 5808 | [OpenAlex ID](https://openalex.org/A5103245119)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种动态适配器融合（DAF）框架，通过将每个任务的轻量级适配器与前一全局适配器和初始化参数融合，构建单一全局适配器，实现无检索、无扩容的预训练模型基类增量学习；

**💡 创新点**

基于PAC‑Bayes理论导出的稳定性-可塑性权衡优化，利用拉格朗日乘子求解最优融合系数；同时引入递归平均初始化策略，无需存储历史适配器即可获得高质量先验；

**🔧 技术方法**

PAC‑Bayes分析、拉格朗日乘子求解、二阶泰勒展开、Fisher信息矩阵近似Hessian、ViT模型与轻量级适配器、参数高效微调（PEFT）等；

**📊 数据集**

CIFAR‑100、ImageNet‑R、ImageNet‑A、ObjectNet四大增量学习基准；

**📈 对比分析**

与多种无样本回放的最新方法（L2P、DualPrompt、CODA‑Prompt、RanPAC、SSIAT、SLCA、EASE、MOS等）以及基于回放的强大基线（iCaRL、MEMO、FOSTER）进行对比；在所有四个数据集上均取得最高平均精度和最终精度，最显著的提升为ObjectNet上终极精度从MOS的58.52%提升至DAF的66.37%（+2.75%），相对回放方法FOSTER在ObjectNet上也提高7.03%；

**⚠️ 局限性**

对鲁棒初始化与融合系数α的敏感性仍需手动调参，虽然实验表明性能稳定，但在极小模型或极低资源场景下的泛化仍有待验证；

---

## 237. More Code, Less Reuse: Investigating Code Quality and Reviewer Sentiment towards AI-generated Pull Requests

**arXiv ID:** 2601.21276 | [PDF](https://arxiv.org/pdf/2601.21276v1)

**作者:** Haoming Huang `[一作]` (Institute of Science Tokyo), Gema Rodríguez-Pérez `[通讯]` (University of British Columbia)

**通讯引用:** 479 | [OpenAlex ID](https://openalex.org/A5077601628)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对 AI 生成的 Pull Request 与人工生成的 Pull Request 进行代码质量评估和审阅者情感分析，揭示了 AI 代码冗余与审阅者情绪之间的脱节。

**💡 创新点**

创新点在于提出 Max Redundancy Score (MRS) 量化语义冗余，并首次系统性比较 AI 代码的冗余度与审阅者情感，发现 AI 代码虽冗余但往往获得更中性/积极的审阅者反馈。

**🔧 技术方法**

使用了 CodeSage‑Large 进行代码嵌入、PyRef 检测重构、Radon 计算循环复杂度（CC）、Emotion English DistilRoBERTa‑base 进行情感分析。

**📊 数据集**

使用 AIDev 数据集（包含多项目 Python PR），其中重点分析了过滤后的 Python 仓库（Dataset A）和 crewAI 仓库（Dataset B）两大子集。

**📈 对比分析**

比较方法：对传统指标（LOC、CC）和新指标 MRS 进行统计对比，并对审阅者评论情感进行箱线图对比。结果显示 AI PR 的 AMR（平均最大冗余）约为人类 PR 的 1.87 倍，且情感分布更偏中性/喜悦。

**⚠️ 局限性**

局限性包括仅聚焦 Python 语言、使用单一情感模型、只分析 crewAI 的重构案例，可能不适用于其他语言或更广泛的编码风格。

---

## 238. Ostrakon-VL: Towards Domain-Expert MLLM for Food-Service and Retail Stores

**arXiv ID:** 2601.21342 | [PDF](https://arxiv.org/pdf/2601.21342v1)

**作者:** Zhiyong Shen `[一作]` (Rajax Network Technology), Wenguo Duan `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了专门针对食品服务与零售店（FSRS）的多模态大语言模型 Ostrakon‑VL，并通过闭环数据清洗管道 QUAD 与 ShopBench 评测基准实现了高效的领域适配。

**💡 创新点**

创新点在于（1）构建了首个 FSRS 领域专用 MLLM，利用多阶段训练提升参数效率；（2）推出了包含单图、多图、视频三种输入格式的统一评测基准 ShopBench；（3）设计了四阶段质量意识无偏自动化数据清洗流程 QUAD，显著提升数据质量并压缩数据量；（4）提出视觉必要率（VNR）与视觉诱导失误（VIF）两种诊断指标。

**🔧 技术方法**

技术包括多模态模型 Qwen3‑VL‑8B 作为骨干，四阶段数据清洗（质量过滤、基础模型引用过滤、多模语义去重、能力覆盖再分配），三阶段训练策略（标注增强、离线课程学习、混合偏好优化），以及自定义奖励模型 Skywork‑VL‑Reward 与多模态嵌入模型 GME‑Qwen2VL‑2B。

**📊 数据集**

使用的数据集主要有：从公开 FSRS 场景收集的原始 69.25M 视觉‑文本对；经 QUAD 处理后得到 3.40M 训练集；ShopBench 评测集覆盖 ShopFront、ShopInterior、Kitchen、MultiImg、Video 5 个子任务；公开多模评测基准 MMBench‑EN、MMStar、MMVet、HallusionBench、AI2D、OCRBench、DocVQA、MathVista、MMMU、MMBench‑CN、Chinese‑OCRBench、CMMMU、CMATH。

**📈 对比分析**

与同规模公开模型（Qwen3‑VL‑8B、InternVL3.5‑8B、GLM‑4.6V‑FlashX 等）以及更大模型（Qwen3‑VL‑235B、InternVL3.5‑241B）对比，Ostrakon‑VL 在 ShopBench 上平均得分 60.1，领先同规模模型 4.8 分，并超越 235B 模型 0.7 分；在公开基准上平均得分 66.7，略低于 Qwen3‑VL‑8B 的 72.4，但在 FSRS 相关任务上表现极佳。

**⚠️ 局限性**

局限性包括：在通用多模任务上仍存在轻微性能下降；QUAD 需要多次模型推理，计算成本高；对时间序列或规则推理的支持仍待加强；模型对极端噪声（如严重模糊或光照失真）在某些子任务仍表现不稳定。

---

## 239. FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision-Language Models

**arXiv ID:** 2601.21187 | [PDF](https://arxiv.org/pdf/2601.21187v1)

**作者:** Chenyu Huang `[一作]` (Fudan University), Tao Chen `[通讯]` (Fudan University)

**通讯引用:** 43091 | [OpenAlex ID](https://openalex.org/A5100357719)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过细粒度子空间级模型融合，将大型推理模型的推理能力注入视觉语言模型，保持原有视觉能力；

**💡 创新点**

创新点在于利用SVD对任务向量进行子空间分解，针对每个子空间学习可调缩放系数，并采用无标签自蒸馏双目标优化来平衡推理注入与视觉保持；

**🔧 技术方法**

核心技术包括SVD子空间分解、可学习门控、KL散度蒸馏与子空间谱值最大化等；

**📊 数据集**

使用的评估数据集包括视觉推理任务的MathVista、MathVision、MathVerse、MMMU、R1-OneVision、MMStar，以及视觉感知任务的TextVQA、POPE、SeedBench，校准数据为VizWiz；

**📈 对比分析**

与基线方法（Task Arithmetic、Ties‑Merging、DARE、IP‑Merging）以及原始VLM进行比较，FRISM在所有规模的Qwen2.5‑VL系列和多种模型上均超过基线，并在保持甚至提升视觉性能的同时，逼近高成本后训练方法的效果；

**⚠️ 局限性**

局限性包括对预训练VLM与LRM匹配度的依赖、子空间截断可能导致信息损失、以及缺乏对极端任务场景和多模态推理数据稀缺性更全面的评估。

---

## 240. SecIC3: Customizing IC3 for Hardware Security Verification

**arXiv ID:** 2601.21353 | [PDF](https://arxiv.org/pdf/2601.21353v1)

**作者:** Qinhan Tan `[一作]` (Princeton University), Sharad Malik `[通讯]` (Princeton University)

**通讯引用:** 20927 | [OpenAlex ID](https://openalex.org/A5085975362)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了专门针对硬件非干扰性验证的自定义IC3模型检查器SecIC3；

**💡 创新点**

通过利用自组合电路的对称结构，提出对称状态探索与等价谓词两种技术，显著提升IC3的学习效率；

**🔧 技术方法**

在ABC-PDR和rIC3基础上集成等价谓词（neq）与对称状态映射，并使用Yosys合成、AIG表示及SAT求解器；

**📊 数据集**

使用了10个开源硬件设计组成的非干扰性基准集（包括Multiplier、Modexp、GCD、FP_ADD、FP_MUL、FP_DIV、SecEnclave、Cache、Sodor、Rocket）；

**📈 对比分析**

与两种基线实现在七种配置下进行对比，平均加速16.5倍，最快达49.3倍，并在部分设计中显著提升证明边界；

**⚠️ 局限性**

在某些设计中单独使用谓词替换会导致性能下降，且该方法对非干扰性之外的属性和更大规模设计的适用性尚需进一步验证。

---

## 241. Enhancing Underwater Light Field Images via Global Geometry-aware Diffusion Process

**arXiv ID:** 2601.21179 | [PDF](https://arxiv.org/pdf/2601.21179v1)

**作者:** Yuji Lin `[一作]` (Xi'an Jiaotong University), Deyu Meng `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 31211 | [OpenAlex ID](https://openalex.org/A5091017287)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于扩散模型的 GeoDiff‑LF 框架，用于增强海底光场图像。

**💡 创新点**

创新点在于将几何感知正则化、轻量化卷积与注意力适配器以及噪声预测机制融入扩散模型，充分利用 4‑D 光场的空间‑角度结构。

**🔧 技术方法**

主要技术包括 SD‑Turbo 扩散模型、卷积和 EPIT 注意力适配器、张量 Tucker 分解的全局几何正则化，以及基于噪声图的快速采样策略。

**📊 数据集**

使用了 LFUB（70余个光场场景）进行训练与测试，并在实时采集的 LFUID 数据集上验证泛化能力。

**📈 对比分析**

与传统光场与图像增强方法（如 DistgSSR、MSPNet、LFUB、Ushape 等）相比，GeoDiff‑LF 在 PSNR/SSIM/LPIPS/ΔE 等指标上均取得最高分，在 LFUID 的 UIQM、BRISQUE、NIMA、CCF 上也实现最优或接近最优性能。

**⚠️ 局限性**

局限性包括对极端颜色退化场景的鲁棒性不足、残留雾霾现象以及相对较高的推理时间（多步采样导致）。

---

## 242. Finetune-Informed Pretraining Boosts Downstream Performance

**arXiv ID:** 2601.20884 | [PDF](https://arxiv.org/pdf/2601.20884v1)

**作者:** Atik Faysal `[一作]` (Rowan University), Huaxia Wang `[通讯]` (Rowan University)

**通讯引用:** 14715 | [OpenAlex ID](https://openalex.org/A5021827481)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于微调信息的预训练方法FIP，在多模态自监督预训练中聚焦目标模态，提升下游任务性能。

**💡 创新点**

创新点包括：对目标模态采用更高掩码比例、加大损失权重、增深对应解码器，三者协同引导编码器更好学习目标模态特征。

**🔧 技术方法**

采用Masked Autoencoder框架（DenoMAE），配合Transformer编码器、异步掩码、加权MSE损失和不等深度解码器。

**📊 数据集**

使用无线信号数据集：10,000条无标签样本用于预训练，1,000条标注星座图用于10类AMC微调，包含四种模态（星座图、谱图、原始信号、噪声）。

**📈 对比分析**

与原始DenoMAE及纯ViT基线对比；在低信噪比（-10 dB）时，FIP-DenoMAE达到69.2%准确率，显著优于DenoMAE（68.4%）和ViT（55.4%），高SNR下差距缩小。

**⚠️ 局限性**

局限性在于仅针对单一目标模态与特定任务验证，效果在其他任务或多目标情形的推广尚未评估；高SNR下提升有限。

---

## 243. Infusion of Blockchain to Establish Trustworthiness in AI Supported Software Evolution: A Systematic Literature Review

**arXiv ID:** 2601.20918 | [PDF](https://arxiv.org/pdf/2601.20918v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 244. Signal from Structure: Exploiting Submodular Upper Bounds in Generative Flow Networks

**arXiv ID:** 2601.21061 | [PDF](https://arxiv.org/pdf/2601.21061v1)

**作者:** Alexandre Larouche `[一作]` (Université Laval), Audrey Durand `[通讯]` (Mila)

**通讯引用:** 20963 | [OpenAlex ID](https://openalex.org/A5080591144)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究如何利用奖励函数的子模结构，在生成流网络（GFlowNets, GFN）中生成上界并利用该上界进行训练，以提升在子模奖励任务中的采样效率与分布匹配。

**💡 创新点**

创新点：① 证明子模奖励可用于构造未观测终止状态的上界，并给出上界产生的期望数量与覆盖率理论；② 基于这些上界提出子模上界 GFN（Submodular Upper‑Bounds GFN, SU‑GFN），在单次奖励查询下获得数倍以上的训练信号；③ 分析上界引入的乐观偏差对采样分布的影响。

**🔧 技术方法**

主要技术：生成流网络（GFN）框架、子模函数理论、Optimism‑in‑the‑Face‑of‑Uncertainty (OFU) 原则、基于上界的训练信号生成、Trajectory Balance 损失、Graph Isomorphism Network（GIN）做状态编码。

**📊 数据集**

实验数据集：随机 Erdős‑Rényi (ER) 与 Barabási‑Albert (BA) 图（N=1000）以及真实世界图数据集 Cora、CiteSeer、GrQc（N=2708/3279/5249），任务为受子模约束的最大集合覆盖/影响力最大化。

**📈 对比分析**

比较方法：经典 GFN、带过滤的 SU‑GFN（-F）与不过滤的 SU‑GFN。评价指标包括 Flow Consistency in Subgraph（FCS）衡量分布匹配、Top‑100 Average Reward 衡量生成质量。结果显示：SU‑GFN 在相同奖励查询预算下，FCS 下降更快、方差更小；Top‑100 平均奖励与经典 GFN 相当或略优，尤其在高 C 或大规模数据集上表现更佳。

**⚠️ 局限性**

局限性：① 只在奖励采样为子模函数时可行；② 上界可能非常松散，对训练产生的乐观偏差需要进一步控制；③ 理论假设为轨迹均匀采样，实际 GFN 学习循环中采样分布会偏离；④ 仅在图结构任务验证，需扩展到其他子模任务。

---

## 245. Snowball: A Scalable All-to-All Ising Machine with Dual-Mode Markov Chain Monte Carlo Spin Selection and Asynchronous Spin Updates for Fast Combinatorial Optimization

**arXiv ID:** 2601.21058 | [PDF](https://arxiv.org/pdf/2601.21058v1)

**作者:** Seungki Hong `[一作]` (ETH Zurich), Taekwang Jang `[通讯]` (ETH Zurich)

**通讯引用:** 1767 | [OpenAlex ID](https://openalex.org/A5011801401)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了可扩展的全互联 Ising 机器 Snowball，使用双模式 Markov Chain Monte Carlo 选择和异步单自旋更新，目标是降低组合优化问题的求解时间。

**💡 创新点**

创新点在于将双模式 MCMC（随机扫描与轮盘赌选择）与异步单自旋更新结合，并采用多比特位面编码实现高精度耦合系数，同时在 FPGA 上实现增量局域场更新，显著提升了吞吐量和能效。

**🔧 技术方法**

技术手段包括 AMD Alveo U250 FPGA 加速、位面（bit‑plane）耦合表示、增量局域场更新、状态无关随机数生成器、指数函数的分段线性 LUT 近似以及模拟退火调度。

**📊 数据集**

使用了 Gset Max‑Cut 基准集（包含 G6、G61、G18、G64、G11、G62 等实例）以及一个 2000 节点的完全图 K2000 作为实验数据集。

**📈 对比分析**

通过与 Neal、CIM、SB、STATICA、ReAIM 等现有 Ising 机器在 K2000 上的 TTS(0.99) 进行对比，Snowball 在单核 300 MHz 频率下实现了 8 倍的 TTS 降低，单步速度提升至相当于 208 k 倍的改进。

**⚠️ 局限性**

局限性包括受 FPGA 资源与内存带宽约束，位面存储在极大规模问题时可能成为瓶颈；目前仅实现模拟退火，未探索并行退火或量子技术的进一步加速潜力。

---

## 246. Concise Geometric Description as a Bridge: Unleashing the Potential of LLM for Plane Geometry Problem Solving

**arXiv ID:** 2601.21164 | [PDF](https://arxiv.org/pdf/2601.21164v1)

**作者:** Jingyun Wang `[一作]` (Beihang University), Guoliang Kang `[通讯]` (Beihang University)

**通讯引用:** 8934 | [OpenAlex ID](https://openalex.org/A5011488839)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过训练一个多模态解释器将几何图形转换为简洁的条件声明语言（CDL），再使用现成的LLM进行推理，完成平面几何问题求解。

**💡 创新点**

提出将视觉信息转化为结构化文本描述的两阶段训练框架（CoT增强SFT + GRPO与CDL匹配奖励），以及用CDL作为中间表示以显著缩小搜索空间。

**🔧 技术方法**

使用条件声明语言（CDL）、链式推理（CoT）增强的监督微调、Group Relative Policy Optimization（GRPO）以及专门设计的CDL匹配奖励。

**📊 数据集**

构造了Formalgeo7k‑Rec‑CoT数据集（对Formalgeo7k v2进行人工审核并加入CoT），并在Formalgeo、Unigeo、MathVista等公开基准上评测。

**📈 对比分析**

与闭源与开源M模型进行对比，5.5k样本的训练已使方法在Formalgeo、Unigeo和MathVista上实现85.7/84.0/80.8%的准确率，超过所有开源模型并接近Gemini2.5‑Pro的表现。

**⚠️ 局限性**

对CDL的匹配奖励依赖手工规则，且对图像识别的精度仍有一定依赖，未充分解决复杂图形结构的自适应表示和跨域泛化能力的进一步提升。

---

## 247. Adaptive and Robust Cost-Aware Proof of Quality for Decentralized LLM Inference Networks

**arXiv ID:** 2601.21189 | [PDF](https://arxiv.org/pdf/2601.21189v1)

**作者:** Arther Tian `[一作]` (DGrid AI), Aaron Chan `[通讯]` (DGrid AI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在去中心化大模型推理网络中，作者扩展了成本感知的Proof of Quality（PoQ）机制，引入鲁棒聚合与自适应信任权重以抵御恶意评估者的评分操纵。

**💡 创新点**

创新点在于将中位数、截尾均值等鲁棒统计聚合规则与基于偏差更新的自适应评估者权重结合，使PoQ在保持成本意识的同时显著提升对分布偏差与恶意攻击的抵抗力。

**🔧 技术方法**

采用的技术包括：鲁棒统计聚合（median、trimmed mean）、基于偏差的乘法权重更新、成本归一化、Monte Carlo模拟以及对四种攻击策略（噪声注入、提升、破坏、间歇性操纵）的评估。

**📊 数据集**

使用的数据集为SQuAD v1.1（问答）与CNN/Daily Mail新闻摘要，共400个样本；评估者模型包括多种bi-encoder与cross-encoder；推理模型则选取五个开源指令调优LLM。

**📈 对比分析**

在相同的奖励参数和评估者采样规模K=3下，与简单平均相比，median和trimmed mean在对付噪声与间歇攻击时平均奖励下降幅度约50%以内，而在破坏攻击下仍有约57%下降；提升攻击导致奖励显著上升，说明攻击可能导致奖励膨胀。

**⚠️ 局限性**

局限性包括：使用统一的token-level F1作为质量代理可能不足以衡量摘要质量；未考虑Sybil身份攻击；自适应权重更新机制在多数偏差下易饱和，缺乏更严谨的可靠性估计。

---

## 248. Responsible AI: The Good, The Bad, The AI

**arXiv ID:** 2601.21095 | [PDF](https://arxiv.org/pdf/2601.21095v1)

**作者:** Akbar Anbar Jafari `[一作]` (University of Tartu), Gholamreza Anbarjafari `[通讯]` (3S Holding OÜ)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出Paradox-based Responsible AI Governance (PRAIG)框架，阐述AI价值创造与责任治理之间的悖论关系，并给出四种悖论管理策略；

**💡 创新点**

将负责AI治理重新概念化为悖论管理而非传统的权衡优化，形成正式的悖论管理策略分类和可计算的治理框架；

**🔧 技术方法**

采用理论建模、数学公式与系统动力学方法，对价值、风险与治理机制进行定量化描述；

**📊 数据集**

无实验数据集，框架主要基于系统文献综述与专家评估；

**📈 对比分析**

未进行实验比较，缺乏性能度量，框架的有效性仍待实证验证；

**⚠️ 局限性**

局限在于理论性较强、缺乏大规模实证支持，策略条件仅为理论推导，实际可操作性需进一步检验。

---

## 249. Do Reasoning Models Enhance Embedding Models?

**arXiv ID:** 2601.21192 | [PDF](https://arxiv.org/pdf/2601.21192v1)

**作者:** Wun Yu Chan `[一作]` (Hong Kong University of Science and Technology), Yangqiu Song `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10532 | [OpenAlex ID](https://openalex.org/A5020880385)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 RLVR 优化的 reasoning 模型作为文本嵌入模型的效果，并提出了层级表示相似性分析框架 (HRSA)；

**💡 创新点**

创新点在于提出 HRSA 并揭示 RLVR 通过保持全局几何、重组局部结构实现的“Manifold Realignment”现象；

**🔧 技术方法**

使用 RLVR、SFT、对比学习（InfoNCE）、CKA、正交投影、k‑NN 重叠、线性探针等技术进行表示层分析；

**📊 数据集**

实验数据集包括 MTEB（多语种与代码版）、BRIGHT、Chain‑of‑Thought、AG’s News 分类等；

**📈 对比分析**

通过在相同训练配置下对比 base 与 RLVR 初始化的嵌入模型，发现两者在 MTEB/BRIGHT 上表现相同，但 HRSA 显示两者在局部几何上存在差异；

**⚠️ 局限性**

局限在于 RLVR 并未显著提升嵌入质量，局部几何重组不可逆，且机制尚未完全解释，结果可能不适用于所有任务或领域。

---

## 250. Singularity-Free Lie Group Integration and Geometrically Consistent Evaluation of Multibody System Models Described in Terms of Standard Absolute Coordinates

**arXiv ID:** 2601.21413 | [PDF](https://arxiv.org/pdf/2601.21413v1)

**作者:** Andreas Mueller `[一作]` (Johannes Kepler University), Andreas Mueller `[通讯]` (Johannes Kepler University)

**通讯引用:** 6158 | [OpenAlex ID](https://openalex.org/A5088500114)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种在绝对坐标建模的多体系统（MBS）中使用李群积分方法的框架，解决了传统单位四元数求解中出现奇异性和约束不稳定的问题。

**💡 创新点**

创新点在于引入了局部-全局转换（LGT）映射，将绝对坐标与李群的局部坐标相连，使得标准的绝对坐标模型能够与李群积分方法无缝配合，并在绝对坐标形式下保持刚体运动的几何一致性。

**🔧 技术方法**

主要技术包括李群积分（如Munthe‑Kaas、Generalized‑α、BLieDF）、李代数的指数/ Cayley 映射、BCH 公式以及针对不同绝对坐标（单位四元数、轴角、罗德里格斯参数）和局部坐标（螺旋坐标、轴角+位移、罗德里格斯参数+位移、扩展罗德里格斯参数）的 LGT 映射推导。

**📊 数据集**

由于本文为理论方法研究，未使用任何公开数据集进行实验验证；实验与性能分析将留待后续工作。

**📈 对比分析**

在性能比较方面，本文仅给出了理论分析与算法复杂度的预估，指出使用 LGT 映射会带来额外计算开销，但可通过选择合适的局部坐标和李群来平衡精度与效率。

**⚠️ 局限性**

局限性主要在于：①需要为每一种绝对坐标与局部坐标组合推导对应的 LGT 映射；②LGT 计算会增加额外的运算量，尤其在螺旋坐标与 SE(3) 组合时；③罗德里格斯参数等局部坐标在大角度变化时可能出现数值不稳定，需要更高阶的 Cayley 映射或其他修正。

---

## 251. Understanding Frechet Speech Distance for Synthetic Speech Quality Evaluation

**arXiv ID:** 2601.21386 | [PDF](https://arxiv.org/pdf/2601.21386v1)

**作者:** June-Woo Kim `[一作]` (Gwangju Institute of Science and Technology), Federica Cerina `[通讯]` (Amazon Science)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

系统性评估 Fréchet Speech Distance (FSD) 与 Speech Maximum Mean Discrepancy (SMMD) 在不同自监督嵌入、噪声水平及样本量下对合成语音质量的鲁棒性，并与人类 MOS、ASR WER 进行对比；

**💡 创新点**

首次将多种自监督嵌入（wav2vec2、HuBERT、WavLM、ECAPA、Whisper）纳入 FSD/SMMD 评估，并引入不依赖正态假设的 SMMD 指标，证明 WavLM 嵌入在稳定性和可复现性方面优于其它嵌入；

**🔧 技术方法**

使用 Fréchet 距离、最大均值差异（MMD）+高斯核、噪声注入（高斯+MS‑SNSD）、TTS 生成模型（XTTS、YourTTS、Tacotron2、VITS）、ASR 微调（Whisper‑tiny）以及 MOS 人工听评等技术；

**📊 数据集**

以 LibriSpeech 100h 训练集为参考集，使用 LS test‑clean/test‑other 进行评测，加入 MS‑SNSD 背景噪声，并基于 synthetic 语音样本进行实验；

**📈 对比分析**

通过计算 FSD/SMMD 与 MOS、synthetic‑WER 的相关性，发现 WavLM 嵌入下的 FSD/SMMD 与 MOS、synthetic‑WER 的相关性最高，可在大规模评测中提供低成本、可复现的量化指标；单语者模型在 MOS 上表现优异但 FSD/SMMD 分数偏高，表明两类指标并不完全一致；

**⚠️ 局限性**

仍无法完全替代人类评测，嵌入选择对结果影响大，Whisper 嵌入在噪声下表现不稳定，SMMD 对 ECAPA 效果差，FSD 对样本分布和多样性敏感，需结合多指标综合评估；

---

## 252. A2RAG: Adaptive Agentic Graph Retrieval for Cost-Aware and Reliable Reasoning

**arXiv ID:** 2601.21162 | [PDF](https://arxiv.org/pdf/2601.21162v1)

**作者:** Jiate Liu `[一作]` (University of New South Wales), Zhengyi Yang `[通讯]` (University of New South Wales)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了 A2RAG 框架，融合自适应控制循环与代理检索，通过图结构引导检索并将检索结果映射回原始文本，提升多跳问答的成本效益与可靠性。

**💡 创新点**

创新点包括：① 基于证据充分性判断的自适应控制循环，可在检索不充分时自动重写查询并重检；② 代理检索的分层递进策略（局部检索 → 桥接节点搜索 → 个人化 PageRank 全局回退 + 源文本映射），实现从低成本到高覆盖的渐进式检索；③ 通过 Triple-Check 进行答案级别的三重验证，确保答案可验证、相关且完整。

**🔧 技术方法**

使用技术：大语言模型（gpt‑4o‑mini）、知识图构建与实体/关系种子对齐、局部邻域检索、桥接节点搜索、个人化 PageRank、从图节点到源文本的映射回退、证据充分性与 Triple-Check 验证、失败重写策略。

**📊 数据集**

实验数据集：HotpotQA、2WikiMultiHopQA（公共多跳问答基准）以及一个来自金融交易平台的生产级问答数据集。

**📈 对比分析**

方法评估：与 NoRAG、TextRAG、LightRAG、IRCoT 等基线对比；在 HotpotQA Recall@2/5 提升约 +9.9/+11.8；在 2WikiMultiHopQA Recall@2/5 提升约 +11.8；端到端 token 使用和延迟比迭代多跳基线降低约 50%；在提取损失（节点/边随机删除）测试中，A2RAG 的 Recall@5 较图基线下降更缓，接近文本检索性能。

**⚠️ 局限性**

局限性：依赖高质量的实体/关系种子，种子质量下降会导致检索偏移；在知识图非常稀疏或严重碎片化时，结构引导的优势降低；目前评估规模受限于子集实验，需进一步扩大；对持续更新的知识库和长期维护的鲁棒性仍需深入研究。

---

## 253. Rethinking Self-Training Based Cross-Subject Domain Adaptation for SSVEP Classification

**arXiv ID:** 2601.21203 | [PDF](https://arxiv.org/pdf/2601.21203v1)

**作者:** Weiguang Wang `[一作]`, Guangyuan Xu `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 672 | [OpenAlex ID](https://openalex.org/A5087841123)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于自训练的跨受试者 SSVEP BCI 域适应框架，包含滤波器组欧氏对齐（FBEA）、预训练对抗学习（PTAL）与双集成自训练（DEST）以及时频增强对比学习（TFA‑CL）。

**💡 创新点**

创新点包括：① 利用滤波器组信息的欧氏对齐精细化频域对齐；② 在自训练中引入对抗学习实现源/目标分布对齐；③ 采用双集成与多视角伪标签融合提升伪标签质量；④ 通过时频增强对比学习提升特征判别能力。

**🔧 技术方法**

技术方法包括：对抗域适应、梯度反转层、均值教师/EMA、投影层对比损失、滤波器组特征融合、时频数据增强等。

**📊 数据集**

使用 Benchmark 与 BETA 两大公开 SSVEP 语言任务数据集。

**📈 对比分析**

在 Leave‑One‑Subject‑Out (LOSO) 评估中，与 tt-CCA、Ensemble‑DNN、OACCA、SUTL、SFDA 等方法对比，最高 ITR 分别为 Benchmark 203.1±8.03 b/min（0.8 s）和 BETA 160.93±6.93 b/min（0.8 s），均显著优于现有 SOTA。

**⚠️ 局限性**

局限性：对目标域未标记样本的噪声敏感，需进一步验证不同频率分辨率和更长时窗下的鲁棒性，以及对实时部署的计算开销。

---

## 254. LAMP: Learning Universal Adversarial Perturbations for Multi-Image Tasks via Pre-trained Models

**arXiv ID:** 2601.21220 | [PDF](https://arxiv.org/pdf/2601.21220v1)

**作者:** Alvi Md Ishmam `[一作]` (Virginia Tech), Chris Thomas `[通讯]` (Virginia Tech)

**通讯引用:** 4465 | [OpenAlex ID](https://openalex.org/A5006675265)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种针对多图多模态大语言模型的黑盒通用对抗扰动（UAP）攻击方法LAMP，能够在未知模型、未知任务的情境下对多图输入进行精准攻击；

**💡 创新点**

创新点包括：① 基于自注意力的注意力约束和Pompeiu‑Hausdorff距离来显著拉开干净与扰动输入的内部表示；② “contagious”损失使得少量扰动图像能够通过自注意力传播到未扰动图像；③ 索引‑注意力抑制损失实现位置不变攻击；④ 综合五种损失实现跨模型、高效且可解释的UAP学习；

**🔧 技术方法**

采用预训练的Mantis‑CLIP模型冻结参数，利用AdamW优化，损失函数包括语言建模损失、隐藏状态距离损失、注意力距离损失、contagious损失和索引‑注意力抑制损失，整体目标为最大化攻击成功率；

**📊 数据集**

训练数据：Mantis‑Instruct 17000条多模态样本；评估数据集覆盖多图任务：NLVR2、Qbench、Mantis‑Eval、BLINK、MVBench、MM‑Vet、LLaVA‑Bench、OK‑VQA、MSCOCO；

**📈 对比分析**

与CPGC‑UAP、UAP‑VLP、Doubly‑UAP、Jailbreak‑MLLM等基线对比，LAMP在所有目标模型与任务上平均提升约19.5%的攻击成功率（ASR），在单图与多图VQA、图片描述等任务均表现显著领先；

**⚠️ 局限性**

局限性：① 仍依赖于多图输入场景，对单图攻击效果有限；② 在极低扰动预算下攻击成功率下降；③ 目前只针对视觉-文本对抗，未考虑文本模态攻击或更大规模多模态场景；

---

## 255. Non-Markov Multi-Round Conversational Image Generation with History-Conditioned MLLMs

**arXiv ID:** 2601.20911 | [PDF](https://arxiv.org/pdf/2601.20911v1)

**作者:** Haochen Zhang `[一作]`, Zecheng He `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并实现了非马尔可夫（non‑Markov）多轮会话式图像生成，提出了专门的历史条件训练和推理框架。

**💡 创新点**

创新点包括：① 通过回退式编辑和基于名称的多轮个性化两类数据构造，真正逼近真实对话；② token 级历史缓存避免多轮漂移；③ 使用基于 DiT 的重建式 detokenizer 及三阶段可编辑个性化微调，兼顾长程一致性与单轮表现。

**🔧 技术方法**

技术手段包括：多模态大型语言模型（SEED‑X/LLAMA3）、视觉编码器+token化、DiT 重建 detokenizer、token 级缓存、基于 CLIP/ArcFace 的检索与评估。

**📊 数据集**

使用数据集：SEED‑Data‑Edit 转化得到的 rollback 多轮编辑样本；两人视频生成的基于名称的多轮个性化样本（约 92k 条）；单轮编辑/个性化基准如 Emu Edit、SciQA；用于 detokenizer 训练的自然图像与人像集合。

**📈 对比分析**

通过人类评估与 CLIP/ArcFace 等定量指标，在回退编辑和名称个性化任务中显著优于 SEED‑X、EMU2，且在单轮编辑/个性化上保持甚至提升图像质量与提示遵从度。

**⚠️ 局限性**

局限性：缺乏统一的多轮个性化基准；后续阶段仍以面部为主，未覆盖全身或多物体个性化；数据规模与多轮关系多样性有限，未覆盖更丰富的长程依赖（关系、组合记忆）。

---

## 256. Less Noise, More Voice: Reinforcement Learning for Reasoning via Instruction Purification

**arXiv ID:** 2601.21244 | [PDF](https://arxiv.org/pdf/2601.21244v1)

**作者:** Yiju Guo `[一作]` (Renmin University of China), Yankai Lin `[通讯]` (Renmin University of China)

**通讯引用:** 12235 | [OpenAlex ID](https://openalex.org/A5043098453)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Less Noise Sampling Framework，通过识别并去除干扰词以提升RLVR中的采样效率。

**💡 创新点**

首次将token级干扰词识别与去除与rollout优化相结合，利用干净prompt产生的高奖励rollout校准原始prompt的策略。

**🔧 技术方法**

使用token干扰分数过滤、重要性采样与PPO式的CRPO调优，以及参考策略对比等技术。

**📊 数据集**

在Openr1-Math-46k训练集上训练，并在七个数学推理基准（MATH500、AMC23、AIME24/25、GaokaoEN-2023、Minerva、OlympiadBench）进行评估。

**📈 对比分析**

与GRPO、GRPO_extended、DAPO、GRESO等方法对比，保持相同rollout预算下平均提升3.88%准确率，收敛速度提升约1.6倍。

**⚠️ 局限性**

仅在≤8B参数模型上验证，奖励为二元，未与其他GRPO变体融合，缺乏对更大模型和多维奖励环境的评估。

---

## 257. Reinforcement Learning from Meta-Evaluation: Aligning Language Models Without Ground-Truth Labels

**arXiv ID:** 2601.21268 | [PDF](https://arxiv.org/pdf/2601.21268v1)

**作者:** Micah Rentschler `[一作]` (Vanderbilt University), Jesse Roberts `[通讯]` (Tennessee Technological University)

**通讯引用:** 1298 | [OpenAlex ID](https://openalex.org/A5020253845)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 RLME 框架，利用自然语言元问题（由模型自身或其他 LLM 评估）为大语言模型生成器提供奖励，无需真实标签或人工偏好；

**💡 创新点**

创新点在于把自然语言元问题作为奖励源，去除对标签/人类评估的依赖，并支持多目标控制，同时系统性分析了自评、冻结评估器以及奖励破解等问题；

**🔧 技术方法**

技术实现基于 Group Relative Policy Optimization（GRPO）/CISPO 的策略梯度更新，奖励取自评估器对“是”答案的概率，生成器与评估器均为 LLM；

**📊 数据集**

实验使用 GSM8K 计算题、CQAC（SQuAD、NewsQA、TriviaQA、HotpotQA、BioASQ、DROP、RACE、TextbookQA）混合集以及 FaithEval‑Counterfactual 进行验证；

**📈 对比分析**

与标签驱动的 RLVR 对比，RLME 在可验证任务中接近 RLVR 的准确率；在无标签开放域任务中显著提升（如 FaithEval），并能通过多元元问题实现精简答案而不牺牲准确率；但在长时间训练下易出现奖励破解，需要少量真实标签或早停来稳定；

**⚠️ 局限性**

局限在于易被奖励破解，评价者与元问题设计高度敏感；目前仅在低风险场景验证，缺乏针对高风险应用的安全保障与人类监督机制。

---

## 258. "Unlimited Realm of Exploration and Experimentation": Methods and Motivations of AI-Generated Sexual Content Creators

**arXiv ID:** 2601.21028 | [PDF](https://arxiv.org/pdf/2601.21028v1)

**作者:** Jaron Mink `[一作]` (Arizona State University), Elissa M. Redmiles `[通讯]` (Georgetown University)

**通讯引用:** 2446 | [OpenAlex ID](https://openalex.org/A5074435310)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

通过对28名AI生成色情内容创作者的访谈，系统探究其动机、创作方式与使用的技术手段，揭示了该社区的多样化生态；

**💡 创新点**

创新在于首次从创作者视角全面梳理AIG‑SC的创作动机与技术路径，为监管与治理提供了经验性参考；

**🔧 技术方法**

主要技术包括稳定扩散（Stable Diffusion）、低秩适配（LoRA）等生成模型，以及多种jailbreak和图像处理工具；

**📊 数据集**

未采用公开数据集，而是基于访谈受访者自述的创作工具和社区共享资源；

**📈 对比分析**

研究属于质性分析，未进行模型性能对比，重点讨论了不同技术路径在创作成本、质量与可访问性方面的差异；

**⚠️ 局限性**

研究样本规模有限，受访者可能自我审查，缺乏跨地区与法律可推广性的考量，且对负面影响的探讨相对不足。

---

## 259. SPOILER-GUARD: Gating Latency Effects of Memory Accesses through Randomized Dependency Prediction

**arXiv ID:** 2601.21211 | [PDF](https://arxiv.org/pdf/2601.21211v1)

**作者:** Gayathri Subramanian `[一作]` (Indian Institute of Technology Madras), Gopalakrishnan Srinivasan `[通讯]` (Indian Institute of Technology Madras)

**通讯引用:** 3465 | [OpenAlex ID](https://openalex.org/A5107990085)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了硬件防御方案SPOILER-GUARD，用以抑制SPOILER攻击导致的加载-存储假依赖延迟。

**💡 创新点**

通过动态随机化用于依赖判断的物理地址位以及给存储条目打标的方式，使攻击者无法利用地址别名构造可观测的时延，从而消除泄露通道。

**🔧 技术方法**

采用PRNG/TRNG产生随机掩码、改造存储地址缓冲区(SAB)以支持12位物理地址比对与PC打标、在gem5中模拟并验证。

**📊 数据集**

使用SPOILER攻击二进制样例与SPEC CPU2017基准套件（整数与浮点工作负载）。

**📈 对比分析**

与未加防御与仅SPOILER易受攻击的基线对比，SPOILER-GUARD将误差重排事件降至0.0004%，整数/浮点性能分别提升2.12%/2.87%，并在14 nm工艺下仅增加0.064 mm²面积、5.863 mW功耗与69 ps关键路径延迟。

**⚠️ 局限性**

局限性包括仅针对同核心非特权攻击，尚未验证对跨核心或更大规模多线程场景的效果，且需要硬件支持才能实现。

---

## 260. Drive-KD: Multi-Teacher Distillation for VLMs in Autonomous Driving

**arXiv ID:** 2601.21288 | [PDF](https://arxiv.org/pdf/2601.21288v1)

**作者:** Weitong Lian `[一作]` (Zhejiang University), Yu Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 303818 | [OpenAlex ID](https://openalex.org/A5100362465)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出多教师蒸馏框架Drive‑KD，将自主驾驶任务分解为感知–推理–规划三阶段，并通过内部注意力监督训练小型VLM实现高效驾驶能力。

**💡 创新点**

创新点包括：①系统评估蒸馏层级与监督信号，确定各能力最佳层级和注意力信号；②设计多教师蒸馏并引入异步梯度投影（AGP）缓解能力间梯度冲突；③在不同模型规模与架构下验证通用性。

**🔧 技术方法**

采用内部注意力蒸馏、层级选择、跨模态注意力匹配、梯度投影AGP、硬标签监督、对比实验与DriveBench评估等技术。

**📊 数据集**

使用10k人工标注的驾驶蒸馏数据集（单视/多视VQA）以及DriveBench（融合感知、推理、规划）。

**📈 对比分析**

与预训练基线、SFT、单教师蒸馏以及GPT‑5.1比较，InternVL3‑1B（多教师）在DriveBench平均分44.05%、规划55.51%，显著优于同尺寸预训练模型并在规划上超越GPT‑5.1，同时降低42×GPU内存、提升11.4×吞吐。

**⚠️ 局限性**

局限性在于推理能力仍难以完整迁移至小模型；输出分布对蒸馏无效；评估仅为开放式回环，需闭环仿真验证安全性。

---

## 261. Smooth Dynamic Cutoffs for Machine Learning Interatomic Potentials

**arXiv ID:** 2601.21147 | [PDF](https://arxiv.org/pdf/2601.21147v1)

**作者:** Kevin Han `[一作]` (Carnegie Mellon University), Amir Barati Farimani `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了可变半径的动态截断方案，用以在保持分子动力学稳定性的前提下显著降低机器学习原子势能（MLIP）的内存占用和推理时间。

**💡 创新点**

创新点在于构建了一个连续、可微（至少二阶可微）的截断函数，通过对邻居进行软排名与高斯加权，实现目标邻居数的动态逼近，从而在保持能量势表光滑性的同时实现图稀疏化。

**🔧 技术方法**

主要技术包括图神经网络（MACE、Nequip、Orbv3、TensorNet）与动态截断公式的结合，使用sigmoid、多项式包络函数、正态分布加权进行邻居排名和加权；同时在训练和推理中保持高阶可微性。

**📊 数据集**

实验数据集为 MD22（分子动力学轨迹，7个大分子）和 MatPES‑r2scan（约 40 万个材料结构的 MD 轨迹）。

**📈 对比分析**

通过在四种主流 MLIP 上进行固定截断与动态截断的对比，发现动态截断在能量 MAE 0.1‑0.2 meV/atom、力 MAE 2‑3 meV 的误差范围内保持几乎不变，同时内存减少最多 2.26×、推理时间加速最多 2.04×，且在 100 ps NVE 模拟中保持能量守恒，证明了稳定性。

**⚠️ 局限性**

局限性包括：对大原子半径的元素会出现误差上升；动态截断无法保证每原子精确保持目标邻居数；需要预设硬截断半径；仅在所测试的四种模型与两类数据集上验证，其他模型或更复杂体系的表现尚待验证。

---

## 262. Lossy Common Information in a Learnable Gray-Wyner Network

**arXiv ID:** 2601.21424 | [PDF](https://arxiv.org/pdf/2601.21424v1)

**作者:** Anderson de Andrade `[一作]` (Simon Fraser University), Ivan V. Bajić `[通讯]` (Simon Fraser University)

**通讯引用:** 4369 | [OpenAlex ID](https://openalex.org/A5012187461)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种可学习的Gray‑Wyner网络，对多任务视觉任务进行公共与私有信息分离，实现高效压缩编码。

**💡 创新点**

将Gray‑Wyner信息理论框架引入可学习编码器，设计可调的传输/接收率平衡目标，给出损失函数并结合可学习熵模型，同时提供了损失下共同信息的理论边界。

**🔧 技术方法**

使用可学习的卷积自编码器、量化与可学习熵模型、Lagrangian松弛优化、交叉熵/误差损失、ResNet/ MobileNet等深度网络结构。

**📊 数据集**

合成数据集、MNIST彩色化、Cityscapes（语义分割与深度估计）以及COCO 2017（目标检测与关键点检测）。

**📈 对比分析**

与独立、联合、分离、组合等基线对比，使用BD‑rate、BPP、任务准确率/IOU/mAP等指标；在传输率上平均提升约81.6% BD‑rate，在接收率上略优于独立方案，整体性能接近联合方案。

**⚠️ 局限性**

扩展到三及以上任务时通道数呈指数增长；实验中的实际比理论值高；对完全无共享信息或完全依赖的极端情况仅在实验验证，尚需进一步理论与方法完善。

---

## 263. Perceptrons and localization of attention's mean-field landscape

**arXiv ID:** 2601.21366 | [PDF](https://arxiv.org/pdf/2601.21366v1)

**作者:** Antonio Álvarez-López `[一作]` (Universidad Autonoma de Madrid), Domènec Ruiz-Balet `[通讯]` (Universitat de Barcelona)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文将Transformer的前向传播建模为作用在单位球面上的粒子系统，并在均值场极限下引入感知器（perceptron）块，研究其动力学及能量景观的临界点。

**💡 创新点**

创新点在于首次证明在加入感知器后，均值场下的临界测度几乎必然是离散的（原子），并给出了在吸引与排斥两种自注意力调节下的原子化、稀疏化与抗集中性质；同时还给出了关于最大化点的显式描述和对GeLU等平滑激活函数的稳定性分析。

**🔧 技术方法**

核心技术包括将Transformer动力学表述为Wasserstein梯度流、利用球面谐波展开与Funk–Hecke恒等式进行能量与变分分析、参数化转移定理（transversality）证明原子化通用性、以及严格的二阶Hessian估计实现抗集中界限。

**📊 数据集**

实验使用的是合成的均匀随机token嵌入（n=1000）在一维和二维球面上进行，配合随机初始化的感知器权重，未使用公开大规模文本或图像数据集。

**📈 对比分析**

方法通过与仅使用自注意力的基线对比，观察到在加入感知器后聚簇数目与温度β呈倒数平方根关系，聚簇质量受到严格上界控制，实验结果与理论预测高度一致。

**⚠️ 局限性**

局限性包括：仅在理想化的均值场与球面归一化假设下进行，未考虑实际Transformer中的多头注意力、位置编码以及非球面归一化；结果对感知器激活函数的可微性与非解析性有严格要求，且实验仅在低维合成场景验证。

---

## 264. MultiModal Fine-tuning with Synthetic Captions

**arXiv ID:** 2601.21426 | [PDF](https://arxiv.org/pdf/2601.21426v1)

**作者:** Shohei Enomoto `[一作]` (NTT), Shin'ya Yamaguchi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用多模态大型语言模型（MLLM）将单模态图像分类数据集转换为带有合成图像标题的多模态数据集，并在此基础上进行监督对比损失的微调和基于类别平均文本嵌入的推理；

**💡 创新点**

①通过精心设计的提示融合类别、域语境和视觉特征，让MLLM生成高质量、具有多样视角（视觉、形状、纹理）的图像标题；②在微调时引入监督对比损失，显式促成同类样本聚类；③使用类平均文本嵌入实现更丰富的推理；

**🔧 技术方法**

多模态大型语言模型（如 Gemini 2.5 Flash‑Lite）、CLIP 的对比学习框架、监督对比损失、类平均文本嵌入推理；

**📊 数据集**

13个图像分类基准（CIFAR‑10/100、CUB‑200、Caltech‑101、Stanford Cars、Oxford‑IIIT Pet、Food‑101、DTD、EuroSAT、Oxford Flowers‑102、GTSRB、FGVC‑Aircraft、ImageNet）；

**📈 对比分析**

与零样本、全微调、FLYP、DCLIP、WaffleCLIP 等基线对比，单模型 ResNet‑50/ViT‑B/32 均取得显著提升；在标准微调中平均提升约4–5个百分点；在少样本（1/4/8/16/32 shot）场景下，平均提升约3–4个百分点；在无训练的标题推理场景下，8 shot 时甚至优于微调；

**⚠️ 局限性**

目前仅适用于图像分类，无法直接扩展到检测/分割等更复杂任务；标题生成对 MLLM 的依赖导致推理开销，且仅针对图像，未涉及其他模态；

---

## 265. Revisiting Diffusion Model Predictions Through Dimensionality

**arXiv ID:** 2601.21419 | [PDF](https://arxiv.org/pdf/2601.21419v1)

**作者:** Qing Jin `[一作]` (Independent Researcher), Chaoyang Wang `[通讯]` (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过分析线性扩散模型的学习动力学，推导出预测目标与数据维度之间的关系，并提出可学习的 k‑Diff 方法，自动在噪声、速度和数据预测之间找到最优平衡。

**💡 创新点**

创新点在于首次给出 k* = D/(D+d) 的解析公式，将传统离散目标连续化；并通过引入单一可学习参数 k，实现无需显式估计内在维度即可自动选取最优预测目标。

**🔧 技术方法**

主要技术包括线性扩散模型的学习动力学分析、流匹配（flow‑matching）框架、SNR 权重化、可学习的 k 参数化、以及基于 Heun/Euler 的采样方法。

**📊 数据集**

实验使用 ImageNet 256×256 与 512×512 数据集，在潜在空间（Latent‑DiT）和像素空间（JiT）两种设置下验证。

**📈 对比分析**

通过与固定目标（x‑prediction、v‑prediction、ε‑prediction）在相同架构与训练条件下的对比，k‑Diff 在潜在空间实现 FID 1.22–1.34，在像素空间与基线相当或略优，证明其可自动获得或超越手工调优的性能。

**⚠️ 局限性**

局限性包括：k 仍是单一时间不变参数，对更复杂的时变目标探索有限；理论推导基于线性模型和高斯噪声，未直接验证对非线性高维数据的普适性；实验仅覆盖 ImageNet，未验证在其他任务或数据集上的泛化。

---

## 266. MPF-Net: Exposing High-Fidelity AI-Generated Video Forgeries via Hierarchical Manifold Deviation and Micro-Temporal Fluctuations

**arXiv ID:** 2601.21408 | [PDF](https://arxiv.org/pdf/2601.21408v1)

**作者:** Xinan He `[一作]` (Nanchang University), Bin Li `[通讯]` (Shenzhen University)

**通讯引用:** 82929 | [OpenAlex ID](https://openalex.org/A5100395468)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种分层双路径视频伪造检测框架MPF‑Net，先用视觉基础模型（VFM）识别离谱的空间异常，再通过微时序残差（MPF）检测高保真伪造视频的结构化像素波动。

**💡 创新点**

核心创新是定义并利用“Manifold Projection Fluctuation (MPF)”这一新的时序残差特征，它揭示了生成器解码器固定基底导致的像素级结构化波动；同时构建了分层筛选流程，显著提升对高保真、在自然流形上的合成视频的检测率。

**🔧 技术方法**

技术手段包括：
- 使用大规模视觉基础模型（如MetaCLIP‑v2）作为空间流形哨兵；
- 对第二层微时序分支采用LoRA细调，提取连续帧残差并放大；
- 采用一阶泰勒展开与解码器雅可比矩阵分析残差；
- 训练时采用随机起始索引和连续帧提取，避免采样破坏MPF。

**📊 数据集**

数据集：
- 训练采用Youku‑mPLUG（真实）与Pika/SEINE子集（伪造）；
- 评测使用VidProm（低帧率、低分辨率）和GenVideo（one‑to‑many）基准。

**📈 对比分析**

与多种基线（CNNSpot, FreDect, RestraV, NSG‑VD, Skyra‑RL 等）在GenVideo上对比，MPF‑Net在Recall与F1上均达到或超过SOTA，尤其在高帧率、高清场景下准确率可达>95%；在VidProm上亦实现全子集SOTA。

**⚠️ 局限性**

局限性：
- MPF分支对低帧率、低分辨率视频的鲁棒性不足，因离散时间导致残差失真；
- 依赖大规模VFM预训练，模型大小和推理成本较高；
- 目前仅在公开基准上验证，缺乏跨域实际应用的进一步测试。

---

## 267. Generation Enhances Understanding in Unified Multimodal Models via Multi-Representation Generation

**arXiv ID:** 2601.21406 | [PDF](https://arxiv.org/pdf/2601.21406v1)

**作者:** Zihan Su `[一作]` (Tsinghua University), Xiangxiang Chu `[通讯]` (AMAP, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在统一多模态模型（UMMs）中，通过后训练方式，让模型同时生成多种内在视觉表示（像素重建、深度图、分割图）以提升视觉理解能力，并保持或提升生成性能。

**💡 创新点**

创新点是将生成任务作为辅助训练目标来强化理解：使用深度与分割等内在表示鼓励模型学习几何和结构信息，而非仅靠像素重建，从而实现理解与生成的相互促进。

**🔧 技术方法**

技术包括：统一多模态框架的后训练（post‑training），多任务损失组合（像素重建、深度生成、分割生成、标准视觉理解），对不同生成范式（自回归、Masked Autoregressive、扩散）保持架构无关；使用 Depth Anything V2 与 Segment Anything 生成目标。

**📊 数据集**

使用的数据集主要有 LLaVA Mix‑665K、LLaVA‑Next‑Data 进行训练；评估基准包括 MMBench、MMVP、HallusionBench、RealWorldQA、Visual Spatial Reasoning、GenEval、DPGBench；在 MidjourneyV6 上做 OOD 深度生成验证。

**📈 对比分析**

与仅训练理解（SFT）和仅重建（RecA）进行对比，结果显示 UniMRG 在理解任务上显著提升（如 MMVP +3.00、HallusionBench +3.68、VSR +7.21）且在生成任务上保持与 RecA 相近的性能；在多种 UMM 架构（Show‑o、Harmon、OpenUni）上均表现优于基线。

**⚠️ 局限性**

局限性：当生成表征空间受限（如 Show‑o 的 VQ 码本大小）时，难以生成高质量内在表示，导致理解提升有限；此外，该方法仍需在更丰富的内在表示（姿态、草图等）和视频域进行扩展。

---

## 268. KubeSpace: A Low-Latency and Stable Control Plane for LEO Satellite Container Orchestration

**arXiv ID:** 2601.21383 | [PDF](https://arxiv.org/pdf/2601.21383v1)

**作者:** Zhiyuan Zhao `[一作]` (Fudan University), Yue Gao `[通讯]` (Fudan University)

**通讯引用:** 18816 | [OpenAlex ID](https://openalex.org/A5100602494)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了KubeSpace，一个针对LEO卫星的低延迟、稳定的容器编排控制平面；

**💡 创新点**

创新点在于：①基于多控制节点架构实现卫星控制节点的无缝切换；②利用轨道预测实现控制节点的放置与动态分配，显著降低通信延迟和切换频率；

**🔧 技术方法**

采用Kubernetes v1.31的CRD与Watch机制、Kubelet热切换、TLE轨道预测、k‑center+局部搜索优化以及距离阈值过滤等技术；

**📊 数据集**

使用Starlink、Kuiper和OneWeb真实卫星的TLE轨道数据、星图与全球城市地面站位置信息；

**📈 对比分析**

通过与单控制节点（Krios）、多节点手动同步（Karmada、Multi‑KubeEdge）、最优/随机放置、仅最短距离分配等方案比较，KubeSpace在节点状态上报延迟下降59%、手动切换消失、每日切换时延下降84%、平均通信延迟≈32 ms、最大延迟≈60 ms、手动切换频率下降19%等方面均表现优异；

**⚠️ 局限性**

主要局限包括：镜像分发瓶颈、复制副本管理与容灾能力不足、能耗与资源调度深度待进一步研究。

---

## 269. From Implicit Ambiguity to Explicit Solidity: Diagnosing Interior Geometric Degradation in Neural Radiance Fields for Dense 3D Scene Understanding

**arXiv ID:** 2601.21421 | [PDF](https://arxiv.org/pdf/2601.21421v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 270. Predicting Developer Acceptance of AI-Generated Code Suggestions

**arXiv ID:** 2601.21379 | [PDF](https://arxiv.org/pdf/2601.21379v1)

**作者:** Jing Jiang `[一作]` (Beihang University), Li Zhang `[通讯]` (Beihang University)

**通讯引用:** 458087 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文利用一家大型技术公司的66,239条开发者–AI交互日志，对AI生成代码建议被接受与拒绝的差异特征进行定量分析，并基于这些特征设计了个性化接受预测模型CSAP。

**💡 创新点**

创新点在于：①首次在大规模工业数据上进行代码建议接受的定量研究；②发现开发者历史接受率、项目接受率以及IDE版本等多维度特征对接受决策具有显著影响；③提出轻量级、可持续更新的后期过滤模型CSAP，显著提升接受预测精度。

**🔧 技术方法**

技术包括：非参数Mann‑Whitney U检验与Cliff’s Δ进行显著性与效应量评估；Spearman相关系数进行特征冗余剔除；使用全连接神经网络与类别平衡二元交叉熵损失进行二分类预测；与基线（工业过滤模型、直接LLM调用）进行对比实验。

**📊 数据集**

数据集：来自行业合作伙伴的内部IDE交互日志，包含开发者、项目、IDE版本、代码上下文、生成时间等信息，样本量66,239条，覆盖113名开发者、12个项目。

**📈 对比分析**

方法比较：在不平衡和均衡测试集上，CSAP在准确率、精确率、召回率、F1以及交叉熵上均远优于工业过滤模型和LLM基线；在不平衡集上准确率提升12.6%/69.5%，在均衡集上提升87.0%/140.1%，并且在所有指标上保持更低的交叉熵，表明预测效果更好、概率校准更佳。

**⚠️ 局限性**

局限性：仅基于单一公司内部环境的数据，可能缺乏对不同组织、开源社区或不同编程语言的普适性；特征构造依赖日志可得信息，未能捕捉开发者的经验、编码风格等更细粒度因素；IDE版本作为模型版本的代理可能导致部分解释误差。

---

## 271. Graph-Free Root Cause Analysis

**arXiv ID:** 2601.21359 | [PDF](https://arxiv.org/pdf/2601.21359v1)

**作者:** Luan Pham `[一作]` (RMIT University), Luan Pham `[通讯]` (RMIT University)

**通讯引用:** 208 | [OpenAlex ID](https://openalex.org/A5102903082)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种不依赖依赖图的根因分析框架（Graph-Free Root Cause Analysis），利用内部和外部属性的异常分离来定位系统故障根因。

**💡 创新点**

核心创新在于构建组件-属性模型（Component-Property Model），阐明内部属性产生异常且仅通过外部属性传播；并提出内部–外部双重异常评分与组合策略，理论保证在该模型下能够正确识别根因。

**🔧 技术方法**

使用偏差驱动的异常评分（z‑score、IQR 等）保证单调性和可注入性；采用可重排不变的聚合函数（max/mean/sum）生成组件级内部与外部得分；最后通过加性或最小值组合函数（满足内部有界性）进行根因排名。

**📊 数据集**

在 RCAEval 基准上九个真实系统数据集（Online Boutique、Sock Shop、Train Ticket 等）共 735 个故障案例，涵盖 CPU、MEM、DISK、SOCKET、网络延迟/丢包以及代码级错误。

**📈 对比分析**

与八种最先进的基线方法（BARO、ScoreOrdering、Cholesky、PC‑PageRank、Counterfactual、RCLAgent、Traversal、SmoothTraversal）对比，取得 68% Top‑1、91% Top‑3、87% Avg@5 的准确率，较最佳基线 BARO 提升 258%；每个诊断耗时仅 8 ms，几乎与最快基线相当。

**⚠️ 局限性**

局限性包括：需要每个根因组件至少有内部与外部可观测属性；若内部异常幅度极弱导致内部得分不足，或某些系统内部状态可被外部直接观察，模型假设可能不成立；对极端传播放大（如极深链路）时仍需进一步验证。

---

## 272. Accurate Network Traffic Matrix Prediction via LEAD: an LLM-Enhanced Adapter-Based Conditional Diffusion Model

**arXiv ID:** 2601.21437 | [PDF](https://arxiv.org/pdf/2601.21437v1)

**作者:** Yu Sun `[一作]` (Beijing University of Posts and Telecommunications), Mugen Peng `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 20597 | [OpenAlex ID](https://openalex.org/A5060203564)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用LLM增强的适配器与条件扩散模型对网络流量矩阵进行预测。

**💡 创新点**

将流量矩阵映射为RGB图像，并通过冻结LLM+轻量适配器提取高层语义先验，再用双重条件的扩散模型生成高保真预测，解决传统方法的过平滑和缺乏不确定性估计问题。

**🔧 技术方法**

Qwen2‑0.5B（冻结LLM）+适配器；视觉编码器；双重（全局+序列）条件的U‑Net扩散模型；DDIM加速采样；灰度/RGB映射技术。

**📊 数据集**

Abilene（12 节点，5min 采样）和 GÉANT（23 节点）两个真实骨干网络数据集。

**📈 对比分析**

与 RNN、ST‑GNN、M²STL、ViT‑LSTM 等基线相比，LEAD 在 RMSE/MAE 上平均降低 30%‑45%，20 步预测误差仅提升 1%，显著提升预测精度。

**⚠️ 局限性**

推理时需多步去噪导致延迟较高，且冻结轻量化 LLM 的适配限制了对网络域的深度迁移与表达能力。

---

## 273. ConceptMoE: Adaptive Token-to-Concept Compression for Implicit Compute Allocation

**arXiv ID:** 2601.21420 | [PDF](https://arxiv.org/pdf/2601.21420v1)

**作者:** Zihao Huang `[一作]` (ByteDance Seed), Ge Zhang `[通讯]` (ByteDance Seed)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出ConceptMoE框架，在LLM中通过可学习的动态分块将相似token合并为概念，以实现自适应概念级计算。

**💡 创新点**

通过在MoE架构中引入可学习的chunk模块和dechunk模块，实现概念级处理，并在保持总参数和FLOPs不变的前提下公平比较，显著提升性能与效率。

**🔧 技术方法**

采用基于余弦相似度的边界识别、辅助压缩比约束、随机边界噪声、联合解码以及MoE专家路由和多头注意力等技术。

**📊 数据集**

在500B+token的文本预训练、60B Vision‑Language、90B持续训练以及OpenBench、LongContext、VLM等公开与自研数据集上评估。

**📈 对比分析**

在总参数和每token FLOPs相同的条件下，将节省的计算重新分配到MoE层，实验结果在语言预训练提升约0.9点、VLM提升0.6点、长序列任务提升2.3点，并在R=2时预填充/解码速度分别提升至175%/117%。

**⚠️ 局限性**

过度压缩（R>2）会导致性能下降，模型对边界概率敏感需噪声正则化，且在某些细粒度视觉任务上效果略逊。

---

## 274. System 1&2 Synergy via Dynamic Model Interpolation

**arXiv ID:** 2601.21414 | [PDF](https://arxiv.org/pdf/2601.21414v1)

**作者:** Chenxu Yang `[一作]` (Institute of Information Engineering), Jiaqi Wang `[通讯]` (JD.COM)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于动态参数插值的能力控制框架（DAMI），可为每条查询自适应调整推理深度。

**💡 创新点**

创新点在于把系统 1 与系统 2 的权重作为可调节的连续变量，利用插值构成可解释、可预测的 Pareto 前沿；并提供两种 λ(q) 估计方法——基于偏好学习的 DAMI‑Pref 和基于置信度的 DAMI‑Conf。

**🔧 技术方法**

主要技术包括：线性参数插值、特征空间连续性与结构连通性分析、偏好学习奖励模型、置信度双信号融合、动态权重预测与实时模型合并。

**📊 数据集**

使用公开的数学推理基准（GSM8K、MATH‑500、AMC 2023、AIME 2024/2025）以及 Qwen3‑VL‑8B 的多模态视频/图文推理数据集进行评估。

**📈 对比分析**

与早停、思考比例、静态合并、模型路由等基线相比，DAMI‑Pref 在 29%–40% token 减少的同时提升 1.6%–3.4% 准确率；在多模态任务上同样实现 2–3% 的绝对提升，整体占据 Pareto 最优区间。

**⚠️ 局限性**

局限性包括：仅适用于结构相似的系统 1/2 检查点，线性插值可能无法捕获非线性细粒度差异；需要对 λ 的阈值或校准参数进行手工调优，且对极端难题仍可能产生欠推理。

---

## 275. DSCD-Nav: Dual-Stance Cooperative Debate for Object Navigation

**arXiv ID:** 2601.21409 | [PDF](https://arxiv.org/pdf/2601.21409v1)

**作者:** Weitao An `[一作]` (Xidian University), Cheng Deng `[通讯]` (Xidian University)

**通讯引用:** 12425 | [OpenAlex ID](https://openalex.org/A5015874725)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练免费、双立场协作辩论的室内导航框架 DSCD-Nav，在同一候选动作集上由任务-场景理解 (TSU) 与安全-信息平衡 (SIB) 两个立场进行多轮辩论，并由 Navigation Consensus Arbitration (NCA) 进行证据感知仲裁，从而提升零射门室内导航的成功率与路径效率。

**💡 创新点**

创新点在于：①引入双立场（TSU/SIB）辩论机制，显式暴露并解决候选动作的冲突；②设计 NCA 仲裁模块，融合两立场证据并可触发轻量微探测 (micro‑probing) 以在争议时进行低风险验证；③整个流程无需训练、可直接附加在现有 VLM‑based 导航系统上。

**🔧 技术方法**

技术手段包括：使用视觉语言模型 (如 Gemini‑2.5‑Flash‑Lite) 作为感知后端；对候选动作进行轻量包装；通过 LLM 进行 TSU 与 SIB 的交互式辩论；NCA 仲裁结合任务进展、安全性与信息增益；微探测通过缩短前进步幅与细微转向来验证争议方向。

**📊 数据集**

实验数据集：Habitat 的 HM3Dv1、HM3Dv2、MP3D 目标导航数据集，以及 GOAT 多目标长期导航基准。

**📈 对比分析**

与多种现有零射门 / 训练自由方法（如 VLMNav、DORAEMON、ESC 等）在 SR、SPL 及探索冗余指标上对比，DSCD‑Nav 在 HM3Dv1、HM3Dv2、MP3D 以及 GOAT 上均实现了显著提升（例如 SR 达到 75.6% / 73.0% / 47.8%，SPL 达到 40.1% / 38.7% / 24.2%），同时降低了 AORI 表明更高的路径效率与更少的重复探索。

**⚠️ 局限性**

局限性包括：①对底层 VLM 的感知质量高度依赖，若模型误判可能导致辩论失效；②微探测参数需经验调优，且在极端视觉噪声或极简信息环境下可能效果有限；③辩论轮次数增大会增加推理成本，需在实时性与精度之间权衡；④仍受候选动作生成器的局限，若生成的候选集合不足或错误，整个框架无法弥补。

---

## 276. Towards Space-Based Environmentally-Adaptive Grasping

**arXiv ID:** 2601.21394 | [PDF](https://arxiv.org/pdf/2601.21394v1)

**作者:** Leonidas Askianakis `[一作]` (Technical University of Munich), Aleksandr Artemov `[通讯]` (Project-S)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在单次拍摄、无后续视觉反馈的开环环境下，研究了在每个试验期随机变化的物理参数条件下，利用环境感知上下文对格言化潜在空间进行条件化，并在此基础上训练连续控制策略，实现机器人抓取任务。

**💡 创新点**

创新点包括：① 将期程级环境参数作为低维上下文向量显式注入格言化潜在空间，形成环境适应型控制接口；② 采用信息量约束（InfoNCE+hinge）实现姿态通道与形状/上下文通道的解耦；③ 在单次拍摄的开环设置下实现高效学习和快速收敛。

**🔧 技术方法**

主要技术手段为：Soft Actor‑Critic 强化学习；格言化潜在表示网络（多模态编码+融合自编码器）；单位四元数投影保证姿态合法；InfoNCE 基准的互信息抑制；GPU 并行物理仿真（ManiSkill + SAPIEN）与动态参数随机化。

**📊 数据集**

实验使用 ManiSkill 机器人抓取基准，包括数千种合成目标几何体和抓手，利用生成的 32 维潜在编码以及8维归一化的物理参数向量作为上下文；所有数据均在 GPU 加速的仿真环境中生成。

**📈 对比分析**

对比方法为：① 仅使用格言化潜在空间的策略；② 结合环境上下文的格言化潜在空间策略；③ 单次视觉嵌入的开环基线。结果显示，环境上下文增强的格言化策略在约 8.5M 环境步内即可持续达到 95% 的抓取成功率，而单视觉基线仅在 1.8M 步后仍停留在约 25% 的成功率，显示出显著的样本效率提升。

**⚠️ 局限性**

主要局限包括：仅在单次拍摄的开环模式下测试，缺乏闭环视觉或触觉补偿；环境上下文向量在真实任务中可能难以直接获取；实验仅在仿真环境中完成，未验证实际硬件转移；缺乏多种随机种子与 OOD 场景的统计评估；且未探讨在试验期内参数漂移或连续在线估计的情况。

---

## 277. Intrinsic Reward Policy Optimization for Sparse-Reward Environments

**arXiv ID:** 2601.21391 | [PDF](https://arxiv.org/pdf/2601.21391v1)

**作者:** Minjae Cho `[一作]` (University of Illinois), Huy Trong Tran `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出IRPO算法，利用多种内在奖励生成探索策略并通过链式反向传播得到代理梯度，直接优化稀疏奖励环境下的基准策略。

**💡 创新点**

创新点在于构造“IRPO梯度”，通过多内在奖励的探索策略并将其梯度反向传播至基准策略，既解决了信用分配难题，又避免了预训练子策略的需要。

**🔧 技术方法**

采用actor‑critic框架、多重Laplacian扩散最大化内在奖励，使用Jacobian链式求导和信赖域更新实现基准策略的梯度更新。

**📊 数据集**

在公开的离散环境（FourRooms、Maze 等）和连续环境（PointMaze、AntMaze、FetchReach 等）稀疏奖励任务上进行评估。

**📈 对比分析**

与PPO、TRPO、PSNE、DRND、HRL等基线在相同样本预算下对比，IRPO在大多数环境中收敛性能最高，样本效率优于HRL且接近或优于其他方法。

**⚠️ 局限性**

依赖多内在奖励集合和若干探索更新步数，样本复杂度可能升高；在复杂连续环境中只能实现近最优，且IRPO梯度存在偏差。

---

## 278. Learning to Optimize Job Shop Scheduling Under Structural Uncertainty

**arXiv ID:** 2601.21389 | [PDF](https://arxiv.org/pdf/2601.21389v1)

**作者:** Rui Zhang `[一作]` (Beihang University), Jing Yuan `[通讯]` (University of North Texas)

**通讯引用:** 5950 | [OpenAlex ID](https://openalex.org/A5023905379)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为UP-AAC的深度强化学习框架，用以解决具有结构不确定性的作业车间调度问题（JSSP）

**💡 创新点**

核心创新包括（1）异构演员-评论家（AAC）架构，通过后见构造将评论家训练在确定性状态下，消除信用分配误差；（2）不确定感知模型（UPM），利用知识导向的注意力机制为演员提供全局风险特征

**🔧 技术方法**

技术手段主要包括图神经网络（GNN）编码器、注意力机制、Actor-Critic强化学习、后见重构与经验回放、均值平方误差损失和策略梯度优化

**📊 数据集**

在12组基准实例上进行评估，实例尺寸涵盖5/10/15/20个工件与10/15/20台机器，结构不确定性通过不同数量的分支路径实现，数据来自改造的Taillard基准集合

**📈 对比分析**

与七种优先调度规则、标准Actor-Critic以及OR-Tools求解器进行比较；在平均完成时间和CVaR指标上，UP-AAC均优于所有基线，平均误差约1.94%（小/中型）和0.18%（大型），并显著降低最差20%情形的完成时间

**⚠️ 局限性**

局限性主要在于需要先生成大量确定性场景以构建UPM，计算成本相对较高；对超大规模实例的可扩展性及对实时动态事件（如设备故障）的适应性尚待进一步研究

---

## 279. Small models, big threats: Characterizing safety challenges from low-compute AI models

**arXiv ID:** 2601.21365 | [PDF](https://arxiv.org/pdf/2601.21365v1)

**作者:** Prateek Puri `[一作]` (RAND), Prateek Puri `[通讯]` (RAND)

**通讯引用:** 256 | [OpenAlex ID](https://openalex.org/A5086190365)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `a2602d71-93ab-4bad-974b-672788df8193` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过收集并分析 HuggingFace 上 5000+ 开源 LLM 的 benchmark 数据，以及对多种社会危害攻击（如信息战、钓鱼、语音克隆等）所需的计算资源进行模拟，揭示低计算量 AI 模型已能够在消费级硬件上执行多种高危攻击。

**💡 创新点**

创新点在于：① 将模型压缩趋势与 benchmark 性能关联，量化一年内模型参数缩减 10 倍仍能保持竞争力的现象；② 通过计算资源模拟与 Monte Carlo 不确定性分析，将低计算攻击与传统高计算 AI 工作负载进行对比，指出现行基于计算阈值的治理框架存在盲点。

**🔧 技术方法**

技术方法包括：参数量化、Agentic 工作流、Eleuther AI Language Model Evaluation Harness（IFEval、BBH、MATH、GPQA、MUSR、MMLU‑PRO） benchmark、GPU FLOPS 与内存带宽测量、使用 NVIDIA V100 进行生成任务的计算概况、Monte Carlo 采样以评估不确定性。

**📊 数据集**

主要数据集：HuggingFace LLM Leaderboard 的 benchmark 分数与模型尺寸；Eleuther AI Evaluation Harness 提供的多项评测指标；历史攻击案例（如 Brexit 信息战、BEC 诈骗等）以及 NVIDIA 与 Apple 芯片的性能规格。

**📈 对比分析**

比较方法：绘制不同时间点模型尺寸与 benchmark α 的指数衰减曲线，并对典型业务/学术 AI 任务（图像识别、蛋白质结构预测、语音转文本、推荐系统训练等）与模拟攻击的计算需求进行直接对比。结果显示，多数攻击可在单个消费级芯片上完成，且仅需 10 台 V100 或相似设备即可满足全部计算上限，成本极低。

**⚠️ 局限性**

局限性：① 仅评估 ≤30B 参数模型，未验证更小模型是否足以完成所有攻击；② 采用保守的模拟假设，实际攻击可能所需资源更少；③ benchmark 可能存在过拟合，无法完全代表真实能力；④ 监管与治理方案的实施细节未深入探讨。

---

## 280. The Paradox of Robustness: Decoupling Rule-Based Logic from Affective Noise in High-Stakes Decision-Making

**arXiv ID:** 2601.21439 | [PDF](https://arxiv.org/pdf/2601.21439v1)

**作者:** Jon Chun `[一作]` (Kenyon), Katherine Elkins `[通讯]` (Kenyon)

**通讯引用:** 322 | [OpenAlex ID](https://openalex.org/A5075943427)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

评估指令调优大型语言模型在医疗、法律、金融等高风险领域对情感叙事扰动的鲁棒性。

**💡 创新点**

提出“鲁棒性悖论”，发现模型对情感叙事几乎无影响，鲁棒性比人类高110–300倍，并构建162情境基准与控制扰动框架。

**🔧 技术方法**

使用受控扰动设计、BCa自助法、决策漂移/翻转率/熵指标、指令层级分析和多模型实验。

**📊 数据集**

构造162个基准情境（9个场景×18种扰动），覆盖医疗、金融、学术三大领域，包含情感叙事、中性叙事与证据变化。

**📈 对比分析**

与人类决策对照，模型在情感叙事下漂移≈0（Cohen h≈0.003），对证据改变的通过率84.4%，在不同模型、训练方式和领域均保持零漂移，表明极高鲁棒性。

**⚠️ 局限性**

仅限合成情境、英文、规则明确的结构化任务，未覆盖开放式情境、对抗性扰动、跨语言及真实部署场景。

---

## 281. From Consistency to Complementarity: Aligned and Disentangled Multi-modal Learning for Time Series Understanding and Reasoning

**arXiv ID:** 2601.21436 | [PDF](https://arxiv.org/pdf/2601.21436v1)

**作者:** Hang Ni `[一作]` (Hong Kong University of Science and Technology), Hao Liu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 12457 | [OpenAlex ID](https://openalex.org/A5100458870)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种多模态大型语言模型框架，融合数值、视觉（时间序列图像）和文本三模态，支持自然语言查询下的时间序列理解与推理；

**💡 创新点**

创新点在于：1）Patch-level Alignment（细粒度对齐）强制数值与视觉、文本在补丁级别实现物理一致性；2）Discrete Disentangled Interaction（离散分离交互）通过向量量化（RVQ）分离共享与独特语义，提升跨模态协同；3）Critical-token Highlighting（关键标记突出）挑选问句相关关键信息，增强推理鲁棒性；

**🔧 技术方法**

使用的技术包括：补丁级别对齐与对比学习、层次向量量化（RVQ）实现离散分离、跨模态交叉注意机制、Qwen2.5‑VL‑7B‑Instruct 作为LLM后端、预训练的多模态LLM以及多模态编码器；

**📊 数据集**

采用 ChatTS 训练集（合成时间序列与自然语言对），以及包含合成与真实世界数据的评测集，评测涵盖理解（噪声、趋势、季节性等）与推理（归纳、演绎、因果、比较）任务；

**📈 对比分析**

与数值-centric、视觉-centric、混合模态的通用 LLM/MLLM 以及专用时间序列 MLLM 进行对比，实验表明本文模型在理解与推理任务上均优于基线，尤其在细粒度对齐与离散分离机制下提升显著；

**⚠️ 局限性**

局限性包括：1）对图像渲染的依赖可能导致视觉噪声影响；2）对真实世界数据的泛化仍有限，尤其在跨变量相关与聚类任务；3）对极端数值噪声或异常值的鲁棒性尚待提升；4）对复杂因果推理能力尚未深入验证；

---

## 282. When Prohibitions Become Permissions: Auditing Negation Sensitivity in Language Models

**arXiv ID:** 2601.21433 | [PDF](https://arxiv.org/pdf/2601.21433v1)

**作者:** Katherine Elkins `[一作]` (Kenyon), Jon Chun `[通讯]` (Kenyon)

**通讯引用:** 305 | [OpenAlex ID](https://openalex.org/A5034544789)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文对16种大型语言模型在处理禁令与否定语句时的行为进行了系统性审计，发现模型普遍存在将否定指令误解为肯定的现象；

**💡 创新点**

提出了“否定敏感度指数”(NSI)作为衡量模型对否定语义处理鲁棒性的治理指标，并基于NSI设计了分层认证框架；

**🔧 技术方法**

采用了多帧构造（四种否定/肯定表述）和逻辑极化归一化技术，对模型在14个伦理情境中的决策进行评估；

**📊 数据集**

使用了包含医学、金融、法律、军事、商业、教育、科学七大领域各两道伦理困境的14个情境样本，总计约27,000个模型决策；

**📈 对比分析**

与传统的准确率、鲁棒性等指标对比，发现开放源模型在否定场景的承诺率高达77–100%，而商业模型虽相对好，但仍有19–128%的极化波动；NSI低于0.20的模型可实现自主运行，0.20–0.49需人工审核，≥0.50需人工确认；

**⚠️ 局限性**

局限性包括：只评估了四种明确否定结构，未覆盖更自然或多语种的否定表达；情境构造人为化，可能与真实对话差距；未提供人类基准比较；仅测试了部分模型规模与版本，缺乏更广泛覆盖；

---

## 283. Mitigating Overthinking in Large Reasoning Models via Difficulty-aware Reinforcement Learning

**arXiv ID:** 2601.21418 | [PDF](https://arxiv.org/pdf/2601.21418v1)

**作者:** Qian Wan `[一作]` (Central China Normal University), Jianwen Sun `[通讯]` (Central China Normal University)

**通讯引用:** 1173 | [OpenAlex ID](https://openalex.org/A5058426743)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于强化学习的后训练框架DiPO，旨在通过模型自感知任务难度来动态压缩大推理模型的推理链，缓解过度思考问题。

**💡 创新点**

首次将任务难度信号（由模型自推理长度生成并经过平滑、标准化、截断处理）嵌入奖励函数，实现难度感知与推理深度的自适应控制；同时设计了难度增强奖励和长度抑制机制，提升模型压缩推理的可控性。

**🔧 技术方法**

采用强化学习后训练（GRPO/DPO）、自监督难度信号生成、长度平滑/标准化/截断、奖励函数设计，并结合prompt控制、SFT、DPO等对照方法；使用DeepSeek-V3等LLM进行模型推理。

**📊 数据集**

在数学推理任务上使用TAL‑SCQ5K、GSM8K、Math‑500、GaoKao、AIME 2025；在跨域评估中使用GPQA、MMLU、LAST；模型基准为Qwen3‑4B与DeepSeek‑R1‑0528‑Qwen3‑8B。

**📈 对比分析**

与基线、TALE‑EP、CoD、SFT、DPO等方法对比，DiPO在所有任务与模型上均显著降低平均推理长度（30%–70%压缩），同时保持或提升准确率，且在域外数据上表现出良好的泛化能力。

**⚠️ 局限性**

局限性包括：需要额外的后训练成本；对极端长尾难度分布仍有一定影响；对prompt的兼容性有限，外部压缩prompt会导致准确率下降；超大模型的可扩展性和超参数稳定性待进一步验证。

---

## 284. RerouteGuard: Understanding and Mitigating Adversarial Risks for LLM Routing

**arXiv ID:** 2601.21380 | [PDF](https://arxiv.org/pdf/2601.21380v1)

**作者:** Wenhui Zhang `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**通讯引用:** 35099 | [OpenAlex ID](https://openalex.org/A5000596496)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统化评估了多模型LLM路由系统对“rerouting”攻击的脆弱性，并提出一种名为RerouteGuard的轻量级检测与拦截框架，防止通过前缀触发器诱导路由错误；

**💡 创新点**

创新点包括：①提出了针对成本升级、质量劫持与安全绕过的三类rerouting威胁分类；②通过对路由决策边界的解释性分析揭示攻击利用复杂度诱导误路由；③设计基于对比学习的双塔模型RerouteGuard，实现对攻击触发器与正常查询的语义差异化判别；

**🔧 技术方法**

技术手段包括：对路由器进行对比学习（contrastive learning）训练双塔网络；使用PCA、热图等可视化工具分析触发器在嵌入空间的分布；利用多种攻击场景（白盒、灰盒、无盒）与评估指标（ASR、JSR、F1、准确率）验证鲁棒性；

**📊 数据集**

实验数据集涵盖：MMLU、GSM8K、MT‑Bench、AdvBench（包含多种 jailbreak 方法）以及来自 RouteLLM 的四个路由器（R_CLS、R_MF、R_SW、R_LLM）和 GPT‑4o / Mixtral‑8x7B‑Instruct 作为模型对；

**📈 对比分析**

与传统基于困惑度、Prompt‑Guard、Llama‑Guard、multi‑router 决策等防御方法对比，RerouteGuard 在所有攻击模型和基准上均达到 100% 检测准确率、F1≈1，且 ASR 降至 0，显著优于其他方法；

**⚠️ 局限性**

限制在于：在白盒自适应攻击下 ASR 仍可升至约30%，且依赖预训练模型与较大训练数据，部署时需维护对比学习模型；

---

## 285. Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control

**arXiv ID:** 2601.21363 | [PDF](https://arxiv.org/pdf/2601.21363v1)

**作者:** Weidong Huang `[一作]` (State Key Laboratory of General Artificial Intelligence), Jingwen Zhang `[通讯]` (State Key Laboratory of General Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

提出并实现了LIFT框架，先用大规模并行SAC在模拟器中预训练人形机器人控制策略，再通过物理信息世界模型实现安全高效的微调，支持零射击部署与样本高效适配。

**💡 创新点**

创新点在于：①将SAC与物理信息世界模型结合，形成可直接用于微调的预训练策略；②在微调阶段仅在模拟器内做随机探索，真实环境仅执行确定动作，显著提升安全性和样本效率；③通过大规模并行模拟实现从头到尾只需单张RTX 4090即可完成预训练与部署。

**🔧 技术方法**

使用技术包括：SAC（带异步actor‑critic和高UTD大批量更新）、JAX实现、MuJoCo Playground与Brax模拟器、物理信息Lagrangian世界模型、自动回归训练、离线世界模型预训练、基于确定性采样的安全回收机制。

**📊 数据集**

数据集：从MuJoCo Playground生成的数千个并行环境下的轨迹；Brax中的环境用来进行微调与评估；真实机器人Booster T1与Unitree G1的实际传感数据用于零射击与微调验证。

**📈 对比分析**

与FastTD3、PPO、SAC、SSRL、MBPO等基线对比。预训练阶段LIFT在多种地形与机器人配置上与PPO/ FastTD3性能相当甚至更快；微调阶段LIFT在分布内、长尾及离散分布外任务上均能稳定收敛并显著降低速度误差，基线在相同数据量下往往会发散或无法达成目标。整体wall‑clock时间与样本效率均优于对比方法。

**⚠️ 局限性**

局限性包括：①真实微调仍需外部测量（Vicon/IMU）与人工监督，限制了部署规模；②仅使用低维本体感知，未支持视觉或高维传感器；③微调流程为同步，导致墙钟时间较长；④物理模型仍有误差，可能在极端环境下导致模型偏差。

---

## 286. The Compliance Paradox: Semantic-Instruction Decoupling in Automated Academic Code Evaluation

**arXiv ID:** 2601.21360 | [PDF](https://arxiv.org/pdf/2601.21360v1)

**作者:** Devanshu Sahoo `[一作]` (BITS Pilani), Dhruv Kumar `[通讯]` (Bits Pilani)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型在代码评估中的合规悖论，构建了 SPACI 攻击框架和 AST‑ASIP 注入协议，并在 25,000 条带攻击的代码样本上评估 9 个 SOTA 模型。

**💡 创新点**

提出了 SPACI 的 17 种攻击向量与 AST‑ASIP 的三种注入算子、三维鲁棒性度量（P̂_decouple、𝒟_adv、Ψ），揭示了“合规悖论”和“C++ 盲点”等系统性缺陷。

**🔧 技术方法**

采用语法树驱动的语义注入、对抗样本生成、统计抽样评估、三维指标计算与热力图可视化等技术。

**📊 数据集**

使用了 25,000 份真实学生提交（Python、C、C++、Java）及其 2,500 份抽样子集，并公开发布该数据集。

**📈 对比分析**

通过与 9 个大模型在标准分数、P̂_decouple、Ψ 等指标对比，发现高参数开源模型（如 DeepSeek‑V3.2、Llama‑3.1）失效率>95%，体现逆向规模效应。

**⚠️ 局限性**

未考虑执行环境过滤、仅评估单轮静态攻击、覆盖范围仅限四种语言、未实现具体防御措施。

---

## 287. Organizational Practices and Socio-Technical Design of Human-Centered AI

**arXiv ID:** 2601.21492 | [PDF](https://arxiv.org/pdf/2601.21492v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 288. Are they just delegating? Cross-Sample Predictions on University Students' & Teachers' Use of AI

**arXiv ID:** 2601.21490 | [PDF](https://arxiv.org/pdf/2601.21490v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 289. Expected Improvement via Gradient Norms

**arXiv ID:** 2601.21357 | [PDF](https://arxiv.org/pdf/2601.21357v1)

**作者:** Joshua Hang Sai Ip `[一作]` (University of California), Ali Mesbah `[通讯]` (University of California)

**通讯引用:** 5407 | [OpenAlex ID](https://openalex.org/A5023086966)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种新的贝叶斯优化采集函数 EI-GN，利用梯度信息通过梯度范数惩罚实现更平衡的探索与利用；

**💡 创新点**

在 EI 的改进框架中引入梯度范数软惩罚，使采集函数在低改进区域仍有信号；提供可解析的闭式近似；

**🔧 技术方法**

基于高斯过程的梯度增强模型，分离函数值与梯度的独立 GP；利用正则化约束和均值场逼近实现可计算的 EI-GN；

**📊 数据集**

在标准多峰函数（Shekel、Hartmann、Cosine、Griewank、Ackley）和 GP 采样目标上进行实验，同时应用于 Acrobot 与 Cartpole 的策略搜索；

**📈 对比分析**

与 EI、TS、CMA-ES、Sobol 等基线比较，EI-GN 在高维或多峰场景下显著优于 EI，表现出更快的收敛和更高的最终奖励；

**⚠️ 局限性**

仅利用独立梯度 GP 可能忽略梯度间相关性，且对梯度噪声不鲁棒，未来需在噪声梯度环境下进一步验证。

---

## 290. Cascaded Transfer: Learning Many Tasks under Budget Constraints

**arXiv ID:** 2601.21513 | [PDF](https://arxiv.org/pdf/2601.21513v1)

**作者:** Eloi Campagne `[一作]` (Centre Borelli, Université Paris-Saclay), Argyris Kalogeratos `[通讯]` (Centre Borelli, Université Paris-Saclay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种在全局训练预算下，利用根树结构进行层次化知识传递的多任务学习框架（Cascade Transfer Learning）。

**💡 创新点**

创新点在于将任务间的知识迁移组织为树形的级联流程，并在有限预算条件下通过中间任务递进式传递，理论证明可优于直接或星形迁移。

**🔧 技术方法**

使用最小生成树（MST）构建传递树，预算分配，梯度下降参数空间的收缩分析，提供误差传播定理；实现上基于任务距离的树构建与局部优化。

**📊 数据集**

实验数据集包括：合成线性回归任务、英国电力负荷预测（UK electricity dataset）以及 Fashion‑MNIST 和 CIFAR‑10 的二分类任务。

**📈 对比分析**

与独立训练、星形迁移、随机树等基线比较，CTL 在RMSE/准确率上平均提升 15–35%（或降低 25–50% RMSE），并在三类任务中表现更稳定。

**⚠️ 局限性**

局限性：仅针对可收敛的线性/参数收缩模型，树结构固定；缺乏对深度非线性模型的支持，预算分配仍为预设，未实现自适应分配。

---

## 291. MAR: Efficient Large Language Models via Module-aware Architecture Refinement

**arXiv ID:** 2601.21503 | [PDF](https://arxiv.org/pdf/2601.21503v1)

**作者:** Junhong Cai `[一作]`, Qinghai Guo `[通讯]` (Huawei Technologies)

**通讯引用:** 3777 | [OpenAlex ID](https://openalex.org/A5015433561)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段的模块感知架构精炼（MAR）框架，先用状态空间模型取代注意力实现线性序列建模，再用可塑化的三元多步脉冲神经元（ATMN）稀疏化激活并降低前馈网络计算量，从而显著降低LLM推理能耗并恢复性能。

**💡 创新点**

创新点在于：①将SSM与SNN结合的两阶段优化；②设计ATMN以提供负脉冲并提升信息容量；③提出基于反向KL和预归一化的双向蒸馏策略（SBDS）解决时间不匹配与信息稀疏问题。

**🔧 技术方法**

采用状态空间模型（SSM）、离散Mamba-2、脉冲神经网络（SNN）、Adaptive Ternary Multi-step Neuron、Spike‑aware Bidirectional Distillation Strategy、前馈网络稀疏化等技术。

**📊 数据集**

使用GenQA、OpenHermes 2.5、InfinityInstruct共约7B标记进行单轮训练，并在PIQA、BoolQ、Winogrande、HellaSwag、ARC‑C、ARC‑E六个零样本推理基准上评估。

**📈 对比分析**

与同规模及更大规模的稀疏/高效模型（如Llamba、Bi‑Mamba、SmoothQuant、TinyLLaMA、SpikeLLM）对比，MAR在保持≈57.9%平均准确率的同时，推理能耗显著低于基线，且优于同规模高效模型。

**⚠️ 局限性**

局限性包括：对大型SNN的训练仍需大量计算；方法主要针对decoder端，encoder侧改动有限；在极长序列或更复杂任务上效果待验证；以及对模型结构的兼容性受限于可替换的Attention/SSM模块。

---

## 292. ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment

**arXiv ID:** 2601.21484 | [PDF](https://arxiv.org/pdf/2601.21484v1)

**作者:** Xiuyu Li `[一作]` (Renmin University of China), Ju Fan `[通讯]` (Renmin University of China)

**通讯引用:** 2965 | [OpenAlex ID](https://openalex.org/A5100739546)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练-free的推理方法 ETS，直接从最优 RL 策略中采样生成答案。

**💡 创新点**

创新点在于把 RL 目标的闭式解拆解为参考模型的转移核与能量项，利用 Monte Carlo 估计能量实现无训练采样，并提供总变差收敛率。

**🔧 技术方法**

核心技术包括统一的 MLM 框架、能量重加权的后向转移、重要采样加速、轻量提议模型与 Fast-dLLM 并行推理。

**📊 数据集**

使用数学（MATH500、GSM8K）、编程（HumanEval）和科学（GPQA）等标准推理/编程基准。

**📈 对比分析**

与基线（Base、Beam Search、Best‑of‑N、Power Sampling、GRPO 等）对比，ETS 在准确率上均超过或匹配后训练 RL 策略，且推理延迟与传统 TTS 相近。

**⚠️ 局限性**

局限包括：需可验证的奖励函数、能量估计仍受采样误差影响、对极大模型的加速方案尚不完善。

---

## 293. A block-coordinate descent framework for non-convex composite optimization. Application to sparse precision matrix estimation

**arXiv ID:** 2601.21467 | [PDF](https://arxiv.org/pdf/2601.21467v1)

**作者:** Guillaume Lauga `[一作]` `[通讯]` (Universite Cote dAzur), Guillaume Lauga (Universite Cote dAzur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种面向非凸复合优化的可变度量块坐标下降框架，并将其应用于非凸图形套索（稀疏精度矩阵）估计；

**💡 创新点**

框架兼容多种更新策略（前向-后向、近似牛顿、Gauss‑Seidel），通过大致近似的多块上界实现迭代下降，并提供在KŁ属性下的收敛保证；

**🔧 技术方法**

使用块坐标前向-后向更新、近似牛顿更新、Gauss‑Seidel最小化；利用可变度量、重加权近似、KŁ理论、局部Lipschitz连续性；

**📊 数据集**

在人工生成的75×75稀疏精度矩阵（稀疏率90%）上，采样1000个高斯样本得到协方差矩阵S；

**📈 对比分析**

与传统完整重加权迭代比较，通过在每次重加权间仅做有限步块迭代，使用F1‑score和NMSE评估精度；结果显示对QUIC和P‑GLasso可实现8–10倍迭代数压缩，ISTA仅能压缩约一半；

**⚠️ 局限性**

假设精度矩阵迭代保持有界、主要上界强凸、梯度满足局部Lipschitz；对非凸问题的收敛依赖KŁ属性；对高维、稀疏度不足的情况，QUIC性能下降；

---

## 294. Partial Feedback Online Learning

**arXiv ID:** 2601.21462 | [PDF](https://arxiv.org/pdf/2601.21462v1)

**作者:** Shihao Shao `[一作]` (Nanyang Technological University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 98100 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

研究部分反馈在线学习，给出在集合可实现下确定性和随机性学习者的最优极限，并通过新组合维度 PFLdim 与 PMSdim 进行理论分析。

**💡 创新点**

创新点在于：①提出 Partial‑Feedback Littlestone Dimension 与 Measure Shattering Dimension，①揭示了确定性与随机性学习可分离性的 Helly 数与嵌套包含性质；②解决了先前开放的 Helly 数在集合值学习中的必要性问题；③给出了部分反馈下的完整极限表征。

**🔧 技术方法**

采用组合维度构造、树形 shattering、前缀辅助维度、信息论极限与对偶性证明等技术手段进行理论证明。

**📊 数据集**

本文为纯理论工作，没有使用实际数据集；示例均采用构造的假设类和集合来说明。

**📈 对比分析**

与传统多分类、集合值与 Bandit 反馈下的 Littlestone 维度相比，部分反馈下 PFLdim 决定子线性误差；随机学习实现 O(log T) 的上界，证明了随机策略可达到对数级别的极限。

**⚠️ 局限性**

局限性包括：①对存在可实现或噪声可实现的情况缺乏适配的组合维度；②在公共设置下随机与确定性学习的完整划分仍未完成；③论文侧重信息论极限，缺少可实现算法的计算效率分析。

---

## 295. SAGE: Sequence-level Adaptive Gradient Evolution for Generative Recommendation

**arXiv ID:** 2601.21452 | [PDF](https://arxiv.org/pdf/2601.21452v1)

**作者:** Yu Xie `[一作]`, Hu Yao `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种利用原生LLM词汇的生成式推荐框架RecLLM，并通过新的优化器SAGE解决OneRec中GBPO的对称保守性导致冷启动抑制和多样性崩塌问题。

**💡 创新点**

创新点包括①基于语义文本构造用户与物品提示，省去独立tokenizer；②序列级信号解耦，消除奖励崩溃；③非对称自适应梯度机制，正向Boost提升冷启动项；负向熵惩罚打破信息囚笼。

**🔧 技术方法**

技术主要包括：大语言模型（Qwen3-8B）在SFT+RLHF两阶段训练；序列级重要性比（几何平均）和多目标优势解耦；自适应梯度界限（Boost Factor、Entropy-Aware Penalty）。

**📊 数据集**

使用Amazon Product Reviews三个子集（Sports、Beauty、Toys）做顺序推荐任务。

**📈 对比分析**

与SASRec、ReaRec、TIGER、LC-Rec以及OneRec系列对比，RecLLM在NDCG@10、Recall@10等准确率指标上提升约5–9%，同时在熵（多样性）和Cold‑Recall（冷启动）上分别提升5–10%和40–60%，明显优于OneRec‑V2。

**⚠️ 局限性**

主要局限在：①仍需在LLM上进行文本级编码，推理延迟较高；②对实时系统的可部署性尚未充分验证；③对极端噪声环境的鲁棒性需进一步评估。

---

## 296. More Bang for the Buck: Improving the Inference of Large Language Models at a Fixed Budget using Reset and Discard (ReD)

**arXiv ID:** 2601.21522 | [PDF](https://arxiv.org/pdf/2601.21522v1)

**作者:** Sagi Meir `[一作]` (Tel Aviv University), Barak Hirshberg `[通讯]` (Tel Aviv University)

**通讯引用:** 1008 | [OpenAlex ID](https://openalex.org/A5080937382)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并提出了 Reset-and-Discard (ReD) 查询协议，利用重置与丢弃策略在固定预算下提升大型语言模型（LLM）在可验证任务上的覆盖率@成本。

**💡 创新点**

创新点在于将 pass@k 与 coverage@cost 关联，证明在 0<α<1 的功率律下 ReD 可把子线性增长转为线性，并证明每次尝试重置是最优；同时提出通过 ReD 轨迹推断 pass@k 指数的方法。

**🔧 技术方法**

使用了重置与丢弃技术、重生理论、Gamma 函数分析、统计推断、随机抽样实验与 GitHub 代码实现。

**📊 数据集**

使用 HumanEval 编码任务数据集进行评估。

**📈 对比分析**

与传统 solve-to-completion 基线在尝试次数、token 数量和 USD 成本三方面进行随机化对比实验，ReD 在多种模型（llama‑3.1‑8b、llama‑3.3‑70b、gpt‑oss‑20b）上在相同覆盖率下显著降低成本，特别是在 80–90% 覆盖率时优于更大模型。

**⚠️ 局限性**

局限性包括仅针对完美可验证任务，未考虑多样验证方法；仅使用独立抽样而非顺序推理；未探讨跨模型迁移、实时应用及更复杂验证场景。

---

## 297. Task-Awareness Improves LLM Generations and Uncertainty

**arXiv ID:** 2601.21500 | [PDF](https://arxiv.org/pdf/2601.21500v1)

**作者:** Tim Tomov `[一作]` (Technical University of Munich), Stephan Günnemann `[通讯]` (Technical University of Munich)

**通讯引用:** 14510 | [OpenAlex ID](https://openalex.org/A5074504351)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于任务相关潜在结构的最小贝叶斯风险（MBR）解码与不确定性量化框架，直接在潜在空间合成贝叶斯最优响应并评估其风险；

**💡 创新点**

核心创新在于将 LLM 输出映射到任务特定的结构化潜在空间，利用任务特定距离衡量贝叶斯风险，从而实现比传统语言空间 MBR 更高效、更优质的生成，并给出结构化的不确定性估计；

**🔧 技术方法**

采用语言→潜在映射 g_T、蒙特卡罗采样、闭式或近似的贝叶斯最优解、各类距离度量（0‑1、汉明、余弦、KL 等）以及对潜在空间的概率分布推断；

**📊 数据集**

在 TriviaQA（单答案 QA）、MAQA（多答案 QA）、CNN/DailyMail（知识图摘要）、WMT19 FI‑EN（机器翻译）以及带概率标签的多答案 QA 等数据集上进行实验；

**📈 对比分析**

与 beam search、self‑consistency、传统语言空间 MBR 以及多种不确定性度量（MSP、SE、KLE、SAR、CoCoA、p(True)）进行对比，实验表明贝叶斯最优解码在任务指标上优于或相当于现有方法，贝叶斯风险在预测拒绝率与 AUC 上显著优于或竞争；

**⚠️ 局限性**

需要易于计算的贝叶斯最优解、有效的语言→潜在映射与逆映射、对多样化采样的依赖，且仅适用于可映射到结构化潜在空间的任务；采样量大时计算开销较高。

---

## 298. HADUA: Hierarchical Attention and Dynamic Uniform Alignment for Robust Cross-Subject Emotion Recognition

**arXiv ID:** 2601.21488 | [PDF](https://arxiv.org/pdf/2601.21488v1)

**作者:** Jiahao Tang `[一作]` (Xi'an Jiaotong University), Zi-Gang Huang `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 2180 | [OpenAlex ID](https://openalex.org/A5050109554)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了跨受试者情感识别框架HADUA，结合层次注意力融合EEG与眼动信号，并通过自适应伪标签与统一对齐实现域适配；

**💡 创新点**

创新点包括（1）层次注意力机制同时建模单模内时序与模间语义交互；（2）基于置信度的软高斯权重动态抑制伪标签噪声；（3）统一对齐（UA）实现类不平衡平衡并稳定条件分布匹配；

**🔧 技术方法**

使用技术主要有Transformer自注意力、交叉注意力、GRU/CNN特征提取、MMD/CMMD分布对齐、Soft Gaussian权重、UA、Adam优化、t-SNE可视化等；

**📊 数据集**

实验数据集为SEED、SEED-IV（含三/四情绪分类），并在DEAP上做扩展验证；

**📈 对比分析**

与多种单模/多模、迁移学习基线（如CSMM、MMDA、DGCNN等）比较，SEED上Acc达94.68%、Macro‑F1 94.69%、AUC 97.68%；SEED‑IV上Acc 92.00%、Macro‑F1 92.88%，均优于现有方法，且类间标准差更低，显示更稳定的泛化；

**⚠️ 局限性**

局限性在于仍依赖伪标签质量，类平衡对参数敏感；目前仅验证三/四情绪，未扩展到更多类别或开放域；对光照、仪器差异的鲁棒性尚未充分验证。

---

## 299. Mean-Field Control on Sparse Graphs: From Local Limits to GNNs via Neighborhood Distributions

**arXiv ID:** 2601.21477 | [PDF](https://arxiv.org/pdf/2601.21477v1)

**作者:** Tobias Schmidt `[一作]` (TU Darmstadt), Kai Cui `[通讯]` (TU Darmstadt)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了稀疏图上的平均场控制（Sparse Mean‑Field Control）框架，将系统状态重新定义为装饰根图的概率分布，并证明在有限时域下的“时间相关局部性”与对应的动态规划原理，从而将原本无穷维控制问题转化为可解的局部化问题。

**💡 创新点**

核心创新点包括：
1) 将平均场状态提升到根图分布，弥补传统平均场假设在稀疏图上失效的问题；
2) 证明有限期问题的时间相关局部性，使得在时间 t 的最优策略仅依赖 (T−t)-跳邻域；
3) 在此理论基础上给出完整的动态规划与 Bellman 方程；
4) 以图神经网络（MPNN）为核心的代理策略与价值函数的理论逼近性证明；
5) 提出基于 GNN 的 SMF‑Actor‑Critic 算法，并给出截断深度与无限图近似的误差界定。

**🔧 技术方法**

使用技术包括：
- Benjamini–Schramm 局部弱收敛理论
- 变分/极限过程的概率论工具
- 动态规划与 Bellman 方程的推导
- Deep Sets 与 Weisfeiler–Leman 图同构理论证明 GNN 的泛化能力
- PPO / Actor‑Critic 强化学习框架
- 经验回放与蒙特卡洛估计的梯度逼近。

**📊 数据集**

实验数据集主要为合成稀疏图环境：
- 随机稀疏图（G(N,d/N)）
- 2 维格子网
- 两种疫情传播场景（集中式 vs 分散式感染），每种场景包含数千个节点，模拟多智能体决策过程。

**📈 对比分析**

与传统的 LWMFMARL（基于全局均值场与节点度的策略）对比，SMFAC 在两种疫情场景下均取得更低的累计损失，能够更精准地选择疫苗投放节点，尤其在局部感染聚集时显著优于全局统计方法。实验中 SMFAC 的平均回报提升约 15–30%，并在不同图结构下保持鲁棒性。

**⚠️ 局限性**

局限性包括：
- 仅在有限时域下严格证明，扩展到无穷期尚未完成；
- 需要足够深的 GNN 才能捕获较远邻域信息，深度受限时会导致近似误差；
- 1‑WL 级别的 MPNN 可能无法区分某些同构不可区分的局部结构；
- 高维分布估计导致梯度方差较大，需要更多样本；
- 对实际大规模稀疏网络的计算成本与可扩展性仍有待进一步验证。

---

## 300. SOUP: Token-level Single-sample Mix-policy Reinforcement Learning for Large Language Models

**arXiv ID:** 2601.21476 | [PDF](https://arxiv.org/pdf/2601.21476v1)

**作者:** Lei Yang `[一作]` (Tianjin University), Deyi Xiong `[通讯]` (Tianjin University)

**通讯引用:** 4670 | [OpenAlex ID](https://openalex.org/A5055232825)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SOUP框架，将离线（历史策略）和在线（当前策略）在单个样本的 token 层面统一混合，用 token 级别的 importance ratio 进行梯度估计，解决了传统全序列混合导致的策略不匹配问题。

**💡 创新点**

创新点在于：① 将离线信息限制在生成序列的前缀，避免全序列离线带来的分布漂移；② 在每个 token 上计算不同策略的 importance ratio，细粒度地控制梯度贡献；③ 提供灵活的截断策略（长度比例/熵值）来决定前缀长度，兼顾多样性与稳定性。

**🔧 技术方法**

采用了 GRPO / DAPO 作为基础优化器，结合离线前缀采样、token 级 importance ratio、clipping、熵值调节、长度或熵值截断等技术；训练时使用 mini‑batch 与全 batch 两种模式，并通过奖励曲线与熵分布进行分析。

**📊 数据集**

使用 Qwen2.5-Math‑7B 与 DeepSeek‑R1‑Distill‑Qwen‑1.5B 两大 LLM，训练数据为 DAPO‑Math‑17k 与 Openr1‑Math‑46k‑8192；在 AIME24、AIME25、MATH‑500、AMC23、Minerva Math、OlympiadBench 等数学推理基准上评估。

**📈 对比分析**

与传统 on‑policy（GRPO/DAPO）、LUFFY（外部强 LLM 生成离线序列）以及 M2PO（对二阶 moment 约束的 off‑policy 方法）进行对比；评估指标为 avg@32 以及 pass@k；实验表明 SOUP 在大多数基准上平均提升 2–5 分，整体平均得分从 48.30 提升到 51.16，且在梯度稳定性、熵保持和推理扩展性上优于基线。

**⚠️ 局限性**

局限性：① 离线前缀来自固定频率（T）刷新后的历史检查点，未能动态选择最优探索策略；② 仅在前缀截断上做了两种手段，其他更高级的截断或混合策略尚未探索；③ 虽然总体训练速度与 on‑policy 相近，但仍有额外的截断与重生成开销。

---

## 301. DexTac: Learning Contact-aware Visuotactile Policies via Hand-by-hand Teaching

**arXiv ID:** 2601.21474 | [PDF](https://arxiv.org/pdf/2601.21474v1)

**作者:** Xingyu Zhang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Shuo Wang `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 DexTac 框架，利用手对手教学收集多维视觉-触觉轨迹，并通过模仿学习训练可接触感知的手指姿态与力/接触点预测网络，再将其部署到仿真/真实的 GEX 手中完成注射任务。

**💡 创新点**

创新点在于：① 通过手对手教学获取高质量的多维触觉数据（力分布、接触中心 CoP）；② 在策略网络中同时输出关节变化、触觉力与 CoP，使机器人能自主选择并保持正确的接触区域；③ 设计了触觉控制器将力与 CoP 校正融入姿态执行，显著降低滑移。

**🔧 技术方法**

技术包括：手对手教学、GelStereo BioTip 触觉相机、ResNet‑18+Transformer（ACT）策略网络、KL 正则化的隐空间、基于力/CoP 的卡西米尔阻尼控制器。

**📊 数据集**

使用人类专家在不同容量注射器（20 mL、30 mL、40 mL、50 mL、60 mL）上的手对手演示数据，总计 90 条演示（后续扩展至 50 条）以及相应的 RGB、触觉图像、关节状态、力与 CoP。

**📈 对比分析**

与仅使用视觉或仅输出力的基线相比，DexTac 在 30 mL、50 mL、60 mL 注射器上的平均成功率提升 31.67%（达到 91.67%），在未见 20 mL 注射器的零样本迁移中也实现 65% 成功率；在无视觉条件下的纯触觉实验中，系统仍能保持 55% 以上成功率。

**⚠️ 局限性**

局限性包括：① 视觉输入单一固定相机，导致对小尺寸物体的定位不精确；② 对不同手型或更大变形物体的泛化能力尚未验证；③ 触觉控制器对 CoP 的线性校正可能不适用于更复杂的接触动力学。

---

## 302. MemOCR: Layout-Aware Visual Memory for Efficient Long-Horizon Reasoning

**arXiv ID:** 2601.21468 | [PDF](https://arxiv.org/pdf/2601.21468v1)

**作者:** Yaorui Shi `[一作]` (University of Science and Technology of China), An Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 48587 | [OpenAlex ID](https://openalex.org/A5100355773)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种视觉记忆系统MemOCR，利用二维视觉布局将长期交互历史压缩为可视化图像，支持LLM在极低上下文预算下进行推理。

**💡 创新点**

创新点包括：①将记忆从线性文本迁移到二维视觉空间，通过标题、加粗、字体大小等视觉优先级实现自适应信息密度；②通过预算感知的强化学习训练，强制关键证据在压缩时保持可读，从而显著提升低预算鲁棒性；③构建多任务QA目标（标准、增强记忆、增强问题）使得同一布局在不同预算下均能表现优异。

**🔧 技术方法**

使用的技术包括：Markdown式富文本渲染为图像的轻量级渲染器；视觉+语言模型Qwen2.5-VL-7B；预算感知的GRPO强化学习框架；三类训练目标（标准QA、增强记忆QA、增强问题QA）来指导视觉布局与压缩策略。

**📊 数据集**

使用的数据集有：HotpotQA、2WikiMultiHopQA、Natural Questions、TriviaQA；训练时在HotpotQA样本上加入大量干扰文档，使单样本约30K tokens；评测时对10K/30K/100K三种长上下文进行测试。

**📈 对比分析**

方法通过与Raw History、Mem0/Mem-α/MemAgent等文本记忆基线以及Qwen系列模型比较。结果显示：在极低预算（16 tokens）下，MemOCR平均准确率约为62%，比文本基线高约8×上下文效率；在高预算（1024 tokens）下也保持领先；整体平均准确率在10K/30K/100K上下文均超过对手。

**⚠️ 局限性**

局限性：目前仅在问答任务上验证，尚未在规划、工具使用等更广泛的长期智能体场景中测试；视觉渲染+OCR可能出现误读导致幻觉；对敏感信息的存储与可见性带来隐私与安全风险；需要进一步改进rich‑text格式与长生命周期的记忆更新策略。

---

## 303. LION: A Clifford Neural Paradigm for Multimodal-Attributed Graph Learning

**arXiv ID:** 2601.21453 | [PDF](https://arxiv.org/pdf/2601.21453v1)

**作者:** Xunkai Li `[一作]`, Guoren Wang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出LION，一个基于Clifford代数的多模态属性图学习框架，通过几何传播（CGP）实现模态对齐，再通过自适应全息聚合（AHA）完成模态融合。

**💡 创新点**

创新点在于利用Clifford几何建模多模态图的拓扑‑模态交互，设计空间转子与几何势驱动的高阶传播，实现对齐‑融合的全流程；以及引入能量与尺度自适应的全息聚合，动态筛选与融合不同模态的语义信息。

**🔧 技术方法**

核心技术包括Clifford代数几何基础、几何传播（CGP）与空间转子、几何势、对齐‑融合的自适应全息聚合（AHA）以及一次性预处理的旋转与势张量。

**📊 数据集**

在9个公开多模态属性图数据集上进行评估，包含社交网络（RedditS）、电影网络（Movies）、四个推荐网络（Grocery、Sports、Ele-fashion、Cloth）、艺术网络（SemArt）、图像网络（Flickr30k）和书籍网络（Goodreads）。

**📈 对比分析**

与GCN、GAT、MMGCN、MLaGA、GraphGPT-O、DMGC、DGF、MIG-GT、NTSFormer、UniGraph2等十余个基线进行对比，在节点分类、链路预测、聚类以及模态检索、文本/图像生成等任务上平均提升5%–8%（如节点分类提升约5.8%，模态检索提升约2.3个百分点），并在所有指标上持续优于SOTA。

**⚠️ 局限性**

主要局限在于模型维度随模态数指数增长，虽然实际模态数有限但仍是潜在瓶颈；此外对极端稀疏或极大规模图的可扩展性和鲁棒性仍需进一步验证。

---

## 304. ChipBench: A Next-Step Benchmark for Evaluating LLM Performance in AI-Aided Chip Design

**arXiv ID:** 2601.21448 | [PDF](https://arxiv.org/pdf/2601.21448v1)

**作者:** Zhongkai Yu `[一作]` (University of California San Diego), Yufei Ding `[通讯]` (University of California San Diego)

**通讯引用:** 3604 | [OpenAlex ID](https://openalex.org/A5048052285)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ChipBench 基准，评估 LLM 在芯片设计中的 Verilog 生成、调试与参考模型生成，并提供自动化训练数据生成工具箱。

**💡 创新点**

三大创新：① 用工业级 44 个复杂 Verilog 模块替代传统简单样例；② 在基准中加入调试任务和多语言参考模型评估；③ 公开实现自动化参考模型验证与训练数据生成工具。

**🔧 技术方法**

基于 VerilogEval 架构的评估框架、Heterogeneous Test Engine、自动化测试脚本、多轮推理与 MAGE 多代理系统，以及对波形文件的解析与验证。

**📊 数据集**

44 个真实 CPU 子模块、89 个调试案例、132 个 Python/SystemC/CXXRTL 参考模型，结合开源 IP、CodeV、Pyranet 与 VeriGen 数据集进行训练与评估。

**📈 对比分析**

采用 pass@1/5/10 与 token 成本指标进行对比，结果显示最强模型 Claude‑4.5‑opus 在 Verilog 生成仅 30.74%，Python 参考模型仅 13.33%，显著低于传统 Benchmark 的 95% 以上通过率。

**⚠️ 局限性**

基准尚未覆盖完整层级设计、波形调试解释能力不足、参考模型准确率低、成本高，亟需提升 LLM 在硬件建模、调试与成本控制方面的能力。

---

## 305. Don't double it: Efficient Agent Prediction in Occlusions

**arXiv ID:** 2601.21504 | [PDF](https://arxiv.org/pdf/2601.21504v1)

**作者:** Anna Rothenhäusler `[一作]` (University of Freiburg), Joschka Bödecker `[通讯]` (University of Freiburg)

**通讯引用:** 3176 | [OpenAlex ID](https://openalex.org/A5038908529)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 MatchInformer，一种基于 Transformer 的模型，用于在遮挡环境下进行多类别占用预测和轨迹预测。

**💡 创新点**

创新点在于：①将 Hungarian Matching 引入训练过程，消除冗余占用预测；②将车头方向与运动分离，并用 sin/cos 预测角度；③使用 MCC 评估不平衡占用预测。

**🔧 技术方法**

采用了 Transformer encoder‑decoder（SceneInformer+DETR 框架）、Hungarian Matching、sin/cos heading 预测、MCC、raycasting 生成遮挡信息等技术。

**📊 数据集**

在 Waymo Open Motion Dataset (WOMD) 上进行训练与评估。

**📈 对比分析**

与 SceneInformer 对比，MatchInformer 在 MCC（占用预测）提升约 58% 以上，minADE 降低 12%，minFDE 降低 18%，并在各遮挡水平上保持一致的性能优势。

**⚠️ 局限性**

局限性在于未将道路地图信息完整融合，导致对向车道的方向预测不准确，Anchor 点可能与真实车辆方向不匹配。

---

## 306. LLaMEA-SAGE: Guiding Automated Algorithm Design with Structural Feedback from Explainable AI

**arXiv ID:** 2601.21511 | [PDF](https://arxiv.org/pdf/2601.21511v1)

**作者:** Niki van Stein `[一作]` (Leiden University), Thomas Bäck `[通讯]` (Leiden University)

**通讯引用:** 22641 | [OpenAlex ID](https://openalex.org/A5062646838)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 LLaMEA-SAGE，一个在自动算法设计中利用抽象语法树（AST）结构特征和可解释模型反馈来引导大型语言模型（LLM）生成优化算法的框架。

**💡 创新点**

创新点在于将程序结构特征提取、梯度提升回归树与 SHAP 解释性分析结合，生成可直接嵌入 LLM 提示的自然语言指导，从而在不限制表达能力的前提下显著加速和提升算法搜索效率。

**🔧 技术方法**

使用技术包括：GPT‑5‑mini LLM、Python AST 解析与图论特征提取、XGBoost 回归树、SHAP 重要性分析、（μ+λ）进化策略以及 LLaMEA 原有框架。

**📊 数据集**

实验数据集涵盖 SBOX‑COST（10 维无噪声盒约束问题）和 MA‑BBOB（Many Affine BBOB 竞赛数据），并在不同维度（5、10、20）上进行验证。

**📈 对比分析**

通过与 Vanilla LLaMEA、MCTS‑AHD、LHNS 等基线在 AOCC 评价下比较，LLaMEA‑SAGE 在相同评估预算内收敛更快、AUC 更高，并在 MA‑BBOB 任务中表现优于现有自动算法设计方法，虽然统计显著性有限，但整体性能提升显著。

**⚠️ 局限性**

局限性包括：实验仅使用单一 LLM 后端，未系统评估跨模型鲁棒性；仅利用静态代码特征，未考虑运行时行为；样本量小、维度有限，尚不确定其在更高维度、噪声或完全不同问题类型上的泛化能力。

---

## 307. IROS: A Dual-Process Architecture for Real-Time VLM-Based Indoor Navigation

**arXiv ID:** 2601.21506 | [PDF](https://arxiv.org/pdf/2601.21506v1)

**作者:** Joonhee Lee `[一作]` (Yonsei University), Jeonggil Ko `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于双进程架构的实时室内导航框架，融合 VLM 语义推理与轻量感知模块，实现低成本设备上的实时语义导航；

**💡 创新点**

将人类双进程理论应用于机器人导航，区分快速感知（System One）与慢速推理（System Two），仅在必要时调用紧凑 VLM，并通过空间+文本增强提升 VLM 语义判断，从而显著降低延迟并提升决策准确性；

**🔧 技术方法**

使用轻量视觉编码器 SigLIP、分割模型 SegFormer‑b0、OCR 模块 docTR 以及 Key Frame Compare、Condition Matching 等感知组件；系统 Two 采用 Gemma3 4B VLM、定制提示与最大 token 约束；整体框架在 Jetson Orin NX 上实现；

**📊 数据集**

在五个真实建筑（三所大学、办公楼、住宅）共收集 120 条轨迹（约 2455 米）视觉数据，并通过人工标注构建人类参考动作标签；

**📈 对比分析**

与 VLM‑only 基线对比，测量决策准确率与延迟；本框架平均延迟 0.7–1.0 秒，决策准确率提升至 64.3%（从 48.2%），完成率 67.5%，比 VLM‑only 高出 11.5 倍，平均延迟降低 66%；

**⚠️ 局限性**

受限于小型 VLM 的空间推理能力，仍需外部空间/文本注解；对动态障碍物缺乏安全模块；KFC 与阈值需要手工调优；系统 One 的分割/ OCR 误差可能导致误切换或误判。

---

## 308. SimGraph: A Unified Framework for Scene Graph-Based Image Generation and Editing

**arXiv ID:** 2601.21498 | [PDF](https://arxiv.org/pdf/2601.21498v1)

**作者:** Thanh-Nhan Vo `[一作]` (University of Science, VNU-HCM), Minh-Triet Tran `[通讯]` (University of Science, VNU-HCM)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SimGraph 统一框架，将场景图驱动的图像生成与编辑集成在同一模型中，实现一体化的生成与编辑流程。

**💡 创新点**

创新点在于：1) 用单一场景图驱动模型同时支持 token‑based 生成与 diffusion‑based 编辑；2) 通过源/目标提示的联合条件化增强语义一致性和空间连贯性；3) 将 VAR、LEDIT++ 与场景图提取协同化。

**🔧 技术方法**

使用的技术包括：CLIP 编码器 + VQ‑VAE 解码器的 VAR 自回归生成；LEDIT++ 扩散编辑模型；Qwen‑VL 2.5 进行场景图提取；联合 CFG 与权重混合的条件化推理。

**📊 数据集**

使用的数据集：EditVal（包含 Visual Genome 与 COCO 子集）以及 Visual Genome 和 COCO 的原始图像。

**📈 对比分析**

在 EditVal 上与 SGDiff、SG2IM、SIMSG 等基线对比，Fidelity 得分 0.87、Accuracy 0.32；在 Visual Genome 上生成任务 FID 21.62、IS 24.78，速度从分钟级降至 20‑30 秒，显著提升。

**⚠️ 局限性**

limitations：对多对象或动态关系的复杂编辑仍易出现错误；场景图预测误差导致生成/编辑偏差；对空间一致性和实时性仍有提升空间。

---

## 309. Task-free Adaptive Meta Black-box Optimization

**arXiv ID:** 2601.21475 | [PDF](https://arxiv.org/pdf/2601.21475v1)

**作者:** Chao Wang `[一作]` (Xidian University), Shuyuan Yang `[通讯]` (Xidian University)

**通讯引用:** 12369 | [OpenAlex ID](https://openalex.org/A5100764373)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种自适应的黑盒优化元学习模型 ABOM，能够在不依赖预先构建任务分布的情况下，在线使用目标任务的优化数据实时更新演化算子参数，实现任务无关的零射击优化。

**💡 创新点**

创新点包括：① 将演化算子（选择、交叉、变异）参数化为可微的注意力机制与 MLP，并通过梯度下降直接对齐子代与精英档案；② 通过闭环自适应学习机制消除传统 MetaBBO 中离线训练和手工任务分布的需求；③ 在优化过程中实时可视化注意力矩阵，揭示可解释的搜索模式。

**🔧 技术方法**

主要技术手段有：可微注意力‑演化算子、基于 MLP 的变异与交叉模块、Dropout 控制探索、GPU 并行化计算、在线梯度更新（AdamW）以及多任务自适应训练框架。

**📊 数据集**

实验使用了公开的 BBOB 合成黑盒优化基准（24 个连续函数，维度30/100/500）和真实无人机路径规划基准（56 条地形实例），并将这些任务划分为训练/测试集用于 MetaBBO 比较。

**📈 对比分析**

对比传统 BBO、适应性变体（SAHLPSO、JDE21、CMAES）以及 MetaBBO（GLEET、RLDEAFL、LES、GLHF）等方法，ABOM 在 BBOB 上取得了与最先进基线相当甚至更优的平均最优值；在 UAV 路径规划中，ABOM 在有限评估次数下收敛最快、得到最低归一化成本，并且 GPU 运行时间显著低于大多数基线。

**⚠️ 局限性**

局限性包括：① 计算复杂度为 O(d³)（尤其是注意力矩阵的 O(N²d_A)），对高维问题存算力瓶颈；② 目前缺乏理论收敛速率分析；③ 人为固定的种群规模与模型容量，未实现自适应动态调整；④ 仅依赖在线学习，未结合预训练或跨任务迁移，可能在极端复杂任务上收敛较慢。

---

## 310. ScaleSim: Serving Large-Scale Multi-Agent Simulation with Invocation Distance-Based Memory Management

**arXiv ID:** 2601.21473 | [PDF](https://arxiv.org/pdf/2601.21473v1)

**作者:** Zaifeng Pan `[一作]` (University of California, San Diego), Yufei Ding `[通讯]` (University of California, San Diego)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了面向大规模多智能体 LLM 仿真的系统 ScaleSim，通过利用代理激活稀疏性和可预测的调用顺序来提升 GPU 内存利用率和推理吞吐量。

**💡 创新点**

核心创新在于引入统一的“调用距离（Invocation Distance）”抽象，使得后端内存管理能够基于代理下一次激活时间的相对优先级做主动预取和距离感知的驱逐决策；同时提供统一接口让前端应用提供调用距离估计，支持多种代理特定内存模块。

**🔧 技术方法**

技术主要包括：1) 调用距离估算与三类仿真工作负载的分类（独立、交互、预定义激活路径）；2) 在 SGLang 上实现的主动预取与距离驱动驱逐；3) 代理特定内存抽象层，用于管理 LoRA 适配器和前缀缓存；4) 通过 GPU 预取与 CPU‑GPU 传输重叠来降低 I/O 停顿。

**📊 数据集**

使用 Qwen2.5 系列大模型（7B、32B）在 NVIDIA H100 GPU（单卡/多卡）上进行评测；仿真基准包含 AgentSociety、Generative Agents 以及信息扩散（信息传播）三类工作负载，每类工作负载通过各自的调用距离估计与代理特定内存。

**📈 对比分析**

与原始 SGLang 以及启用 HiCache 的 SGLang 进行对比。ScaleSim 在三类基准上分别获得 1.73×、1.31×、1.74× 的速度提升；TTFT（首令延迟）相比 HiCache 降低 48%–68%；在不同模型大小、GPU 数量及代理稀疏度下均表现出显著的吞吐量和响应性提升。

**⚠️ 局限性**

局限性：1) 依赖前端准确估算调用距离，误差会导致预取/驱逐失误；2) 目前只针对单 GPU/基于 SGLang 的部署，跨框架适配仍需研究；3) 对极度动态或交互频繁的工作负载，距离估计难度较大；4) 仅评测了部分大模型与仿真场景，未覆盖所有工业应用。

---

## 311. Unifying Speech Editing Detection and Content Localization via Prior-Enhanced Audio LLMs

**arXiv ID:** 2601.21463 | [PDF](https://arxiv.org/pdf/2601.21463v1)

**作者:** Jun Xue `[一作]` (Wuhan University), Yujie Chen `[通讯]` (Beihang University)

**通讯引用:** 32519 | [OpenAlex ID](https://openalex.org/A5100459043)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究聚焦语音编辑检测与内容定位，构建首个大规模双语AI编辑数据集AiEdit，并提出统一检测与定位的Prior-Enhanced Audio LLM（PELM）框架。

**💡 创新点**

创新点包括①通过LLM驱动文本编辑与多种神经语音编辑技术生成高质量双语数据；②在音频LLM中引入词级概率先验与声学一致性损失，显著抑制伪造偏差和语义优先偏差，实现检测与定位的统一。

**🔧 技术方法**

核心技术包括：大型语言模型（Qwen）用于文本编辑与推理；多模态音频LLM；词级概率先验提取；中心聚类一致性损失；结构化音频问答式推理。

**📊 数据集**

使用的数据集为公开的HumanEdit与自研的AiEdit（含英语与中文，覆盖增删改三种编辑操作）。

**📈 对比分析**

与TDL、CFPRF、AGO、BAM等SOTA方法在HumanEdit和AiEdit上对比，PELM在准确率、F1和EER上均领先，EER仅为0.55%（HumanEdit）到9.28%（AiEdit），表现出显著优势。

**⚠️ 局限性**

局限性在于对不同语音编辑模型的覆盖仍有限，依赖先验检测模型且训练成本较高，需进一步提升对极端场景的鲁棒性。

---

## 312. HER: Human-like Reasoning and Reinforcement Learning for LLM Role-playing

**arXiv ID:** 2601.21459 | [PDF](https://arxiv.org/pdf/2601.21459v1)

**作者:** Chengyu Du `[一作]` (Fudan University), Yanghua Xiao `[通讯]` (MiniMax)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出HER框架，实现LLM角色扮演的双层思维与强化学习

**💡 创新点**

引入双层思维（系统层与角色层），通过逆向合成构建推理增强数据，并设计情境特定的奖励模型

**🔧 技术方法**

双层思维生成、逆向推理合成、生成式奖励模型（HER GenRM）以及基于该奖励的强化学习

**📊 数据集**

HER数据集（基于CoSER、Minimax Role-Play Bench等），以及合成的推理增强对话与原则集

**📈 对比分析**

在CoSER与Minimax基准上，与Qwen3-32B和商业模型对比，HER-RL平均分提升约30%（CoSER）及≈15%（Minimax），显著超越基线

**⚠️ 局限性**

评估主要基于单一benchmarks，合成奖励可能仍受偏差，且合成数据依赖强教师模型，易受hallucination影响

---

## 313. 4D-CAAL: 4D Radar-Camera Calibration and Auto-Labeling for Autonomous Driving

**arXiv ID:** 2601.21454 | [PDF](https://arxiv.org/pdf/2601.21454v1)

**作者:** Shanliang Yao `[一作]` (Yancheng Institute of Technology), Ryan Wen Liu `[通讯]` (Wuhan University of Technology)

**通讯引用:** 6736 | [OpenAlex ID](https://openalex.org/A5061907283)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一套统一的4D雷达-摄像头标定与自动标注框架4D-CAAL

**💡 创新点**

设计了兼容雷达与摄像头的双功能标定板，并结合深度、RCS、速度等多特征的粗细化标注策略

**🔧 技术方法**

采用角点检测、DBSCAN聚类、Levenberg–Marquardt非线性优化、Mask2Former实例分割以及多特征置信息融合

**📊 数据集**

在Oculii EAGLE 4D雷达与Sony IMX317摄像头的同步数据上进行实验，使用nuScenes数据集微调分割网络

**📈 对比分析**

与基线方法对比，标定的MRE仅5.25像素，自动标注的点准确率提升至90.12%、mIoU至77.83%，明显优于单一几何投影或单特征过滤的方案

**⚠️ 局限性**

标定依赖手工放置和反射板精度，自动标注受分割质量和多目标重叠的限制

---

## 314. Nimbus: A Unified Embodied Synthetic Data Generation Framework

**arXiv ID:** 2601.21449 | [PDF](https://arxiv.org/pdf/2601.21449v1)

**作者:** Zeyu He `[一作]` (Shanghai Artificial Intelligence Laboratory), Jiangmiao Pang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 43805 | [OpenAlex ID](https://openalex.org/A5087818121)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出Nimbus框架，将导航与操作的合成数据管线统一到一个四层架构中，实现解耦的Load‑Plan‑Render‑Store异步执行，显著提升资源利用率；

**💡 创新点**

创新点在于将管线拆分为异步阶段并通过动态管线调度、全局负载平衡和分布式故障恢复，实现跨域、跨后端的统一、可扩展合成数据生成；

**🔧 技术方法**

采用Ray分布式调度、GPU加速渲染（Gaussian Splatting、Blender、Isaac Sim）、异步存储、动态负载平衡、Supervisor监控等技术；

**📊 数据集**

使用InternData‑N1（导航）、InternData‑A1（操作）和InternData‑M1（操作）等内部数据集；

**📈 对比分析**

在相同硬件上与未优化基线管线对比，Nimbus通过动态调度和渲染优化使吞吐量提升2–3倍，并在大型GPU集群上实现长期稳定运行；

**⚠️ 局限性**

仍受限于预先构建的资产与脚本，难以即时生成全新任务；对物理真实性和真实世界部署仍有差距；进一步扩展到更大规模或多机器人协作场景需进一步优化。

---

## 315. DimStance: Multilingual Datasets for Dimensional Stance Analysis

**arXiv ID:** 2601.21483 | [PDF](https://arxiv.org/pdf/2601.21483v1)

**作者:** Jonas Becker `[一作]` (University of Göttingen), Saif M. Mohammed `[通讯]` (National Research Council Canada)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究创建了首个跨语言的情感维度立场资源 DimABSA，提供了基于 valence‑arousal (VA) 的连续标注；

**💡 创新点**

创新点在于将立场检测从传统的离散标签迁移到连续情感维度，实现更细粒度的情感驱动立场分析；

**🔧 技术方法**

采用多模态预训练语言模型（XLM‑R、RemBERT、LaBSE）进行回归，以及大语言模型（GPT‑5 mini、Gemini 2.5 Flash、Kimi K2）进行提示式与微调式推理；

**📊 数据集**

使用覆盖英语、德语、中文、尼日利亚皮钦语、斯瓦希里语的 11,746 个目标方面，分布在政治与环境保护两大领域，共 7,365 篇文本；

**📈 对比分析**

在 RMSE 评估下，微调的 LLM 回归模型（尤其是 70B 级别）优于提示式 LLM，且在高资源语言上表现良好，但低资源语言仍显著偏低；

**⚠️ 局限性**

局限性包括跨文化 VA 解释的不一致、情感维度标注的可靠性低于 valence、以及低资源语言样本分布不均导致模型偏差。

---

## 316. Explicit Credit Assignment through Local Rewards and Dependence Graphs in Multi-Agent Reinforcement Learning

**arXiv ID:** 2601.21523 | [PDF](https://arxiv.org/pdf/2601.21523v1)

**作者:** Bang Giang Le `[一作]` (VNU University of Engineering and Technology), Viet Cuong Ta `[通讯]` (VNU University of Engineering and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于依赖图的多智能体强化学习框架，利用局部奖励与全局奖励的组合来缓解信用分配问题，并在策略梯度中只保留对目标奖励有因果影响的局部奖励。

**💡 创新点**

创新点在于：①提出了基于依赖图的策略梯度估计器，通过路径可达性来截断无关奖励；②用逆向世界模型与潜在编码实现依赖图的近似；③引入类似GAE的加权平均估计以降低噪声。

**🔧 技术方法**

技术包括：多智能体网络化MDP理论、依赖图与可达性分析、逆向世界模型与编码器、GAE式优势估计、以及基于MAPPO/IPPO的梯度方法。

**📊 数据集**

数据集：LBF与SMAClite两大多智能体基准环境，分别用于验证局部奖励、全局奖励以及依赖图方法的性能。

**📈 对比分析**

与仅使用全局奖励或仅使用局部奖励的基线以及IQL、QMIX、QPLEX等值基方法对比，实验表明在LBF基准中，使用依赖图的IPPO/MAPPO在平均奖励与中位数上均优于其它基线；在SMAClite基准中则表现出更平稳的训练曲线。

**⚠️ 局限性**

局限性：依赖图近似的质量对性能影响显著；在高度动态或大规模智能体环境下，图构建与梯度截断可能带来额外计算开销；实验仅覆盖两种基准，需在更多场景进一步验证。

---

## 317. LMK > CLS: Landmark Pooling for Dense Embeddings

**arXiv ID:** 2601.21525 | [PDF](https://arxiv.org/pdf/2601.21525v1)

**作者:** Meet Doshi `[一作]` (IBM Research), Sachindra Joshi `[通讯]` (IBM Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Landmark（LMK）池化，将文本按块划分并插入特殊标记，随后对标记向量做均值池化，用以生成长文本的向量表示。

**💡 创新点**

创新点在于用分块插入 landmark token 的方式既避免了 CLS 位置偏倚，又减少了均值池化导致的局部信息稀释，且仅需少量额外 token。

**🔧 技术方法**

主要技术包括基于 Transformer 的编码器（ModernBERT、gte‑en‑mlm、mmBERT），对比学习检索微调，RetroMAE 自监督预训练，以及 RoPE 位置编码。

**📊 数据集**

实验数据集覆盖短文本检索（MS MARCO、BEIR、MTEB、MIRACL）和长文本检索（MLDR、COIR、LongEmbed、Multi‑EURLEX）等多语言、多任务数据。

**📈 对比分析**

与 CLS、mean、latent attention、MultiCLS、Mean@k 等传统池化方法对比，LMK 在短文本上保持相近或略优的效果，而在长文本检索上提升显著（如 MLDR 上提升 10–20% NDCG@10）。

**⚠️ 局限性**

局限性包括：对极长序列仍可能存在位置偏倚；需要在输入中插入额外 token，略增计算成本；在训练集缺乏足够长文本时，长文本优势仍有限。

---

## 318. A Unified SPD Token Transformer Framework for EEG Classification: Systematic Comparison of Geometric Embeddings

**arXiv ID:** 2601.21521 | [PDF](https://arxiv.org/pdf/2601.21521v1)

**作者:** Chi-Sheng Chen `[一作]` (Independent Researcher), Fan Zhang `[通讯]` (Boise State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种统一的SPD Token Transformer框架，能够在同一Transformer架构下比较不同几何嵌入（BWSPD、Log-Euclidean、欧氏）对EEG分类的影响。

**💡 创新点**

通过理论分析揭示嵌入几何与梯度条件、BN-Embed近似Riemannian归一化以及bi‑Lipschitz距离保持的关系，并验证了三种嵌入在高维/低维、不同通道数、不同EEG范式下的性能差异。

**🔧 技术方法**

使用了SPD矩阵的几何嵌入（矩阵平方根、对数映射、直接取上三角元素），标准Batch Normalization（BN-Embed）、Transformer编码器、注意力机制、以及多频段token化。

**📊 数据集**

在三大公开EEG数据集上实验：BCI2a（运动想象，22通道），BCIcha（ERP，56通道），MAMEM（SSVEP，8通道）。

**📈 对比分析**

对比结果表明Log‑Euclidean Transformer在所有数据集上达到SOTA（BCI2a 95.37%，BCIcha 95.21%，MAMEM 99.07%），BWSPD在高通道数下表现可与Log‑Euclidean相当，欧氏基线明显落后；多频段token化进一步提升精度并显著降低方差。

**⚠️ 局限性**

局限性包括：跨受试泛化效果差（近随机），需要对齐或领域自适应；对低通道或低维输入的梯度优势不明显；实验主要基于静态协方差估计，未探讨实时或在线场景。

---

## 319. HERS: Hidden-Pattern Expert Learning for Risk-Specific Vehicle Damage Adaptation in Diffusion Models

**arXiv ID:** 2601.21517 | [PDF](https://arxiv.org/pdf/2601.21517v1)

**作者:** Teerapong Panboonyuen `[一作]` `[通讯]`, Teerapong Panboonyuen

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了HERS框架，利用自监督的文本-图像对和LoRA专家模块，对文本到图像扩散模型进行风险特定的车损适配，生成高保真、局部细节丰富的车辆损伤图像。

**💡 创新点**

创新点在于：①完全无人工标注，自动生成多样化损伤提示并渲染图像；②为每种损伤类型训练专门的LoRA专家，并通过权重空间平均融合成统一模型，兼顾专业化与泛化；③关注隐蔽细节与法医可信度，提升保险行业对生成图像的可审计性。

**🔧 技术方法**

使用技术包括：大语言模型（如GPT‑4）进行提示生成；预训练扩散模型（Stable Diffusion XL等）进行图像渲染；LoRA低秩适配训练每个损伤专家；权重空间平均实现多专家融合；ROUGE‑L过滤多样性；VQA+LLM评估语义对齐；人类偏好指标（PickScore、ImageReward、HPS）评估视觉质量。

**📊 数据集**

数据集：自动生成的损伤提示–图像对，来源于LLM与预训练扩散模型；评估使用专门的车险领域基准，包含约200万条结构化文本描述与车辆图片（与保险初创公司合作，公开提示模板与评测协议，但不公开原始数据）。

**📈 对比分析**

与现有基线（VQ‑Diffusion、Versatile Diffusion、SDXL、SD v1.5、MoLE等）比较，HERS在文本可信度和人类偏好上分别提升约5.5%和2.3%；在HPS、IR等指标上也显著领先（HPS 53.4% vs 48.2%，IR 113% vs 95%）。跨骨干网络实验显示其鲁棒性和普适性。

**⚠️ 局限性**

局限性包括：①对真实车险数据的外部验证有限；②恶意使用的防护机制仍处于初步阶段；③目前仅针对车辆损伤，扩展到医疗或灾害等其他安全关键领域仍需研究。

---

## 320. Transversal gates for quantum CSS codes

**arXiv ID:** 2601.21514 | [PDF](https://arxiv.org/pdf/2601.21514v1)

**作者:** Eduardo Camps-Moreno `[一作]` (Université de Bordeaux), Rodrigo San-José `[通讯]` (Virginia Tech)

**通讯引用:** 46 | [OpenAlex ID](https://openalex.org/A5018458682)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了如何计算固定CSS码的对角平面可穿透门的集合，并推导了其逻辑作用、可穿透逻辑门与逻辑恒等的群结构；

**💡 创新点**

创新点在于给出一套完整的线性方程组描述这些群，能够一次性得到所有平面可穿透门；并在单模代码（包括极化码、RM码等）中利用单模代码的结构高效求解；

**🔧 技术方法**

主要技术包括：CSS码的群论描述、对角门的相位约束、向量空间正交运算、单模代码的评估映射与多项式乘积、以及对码子空间的长度与直径分析；

**📊 数据集**

使用的主要数据集是基于单模代码的评估点集合（如RM、极化码对应的二进制向量集合），没有使用传统机器学习数据集；

**📈 对比分析**

与文献中已有的基于核空间求解法相比，本文方法不需要构造大矩阵，只需分析代码的多项式乘积与权重，计算复杂度显著降低，能得到完整的门集合；

**⚠️ 局限性**

局限性在于仅适用于单模（尤其是降序单模）CSS码，且对非单模或更一般的稳定子码的扩展尚未完成；

---

## 321. The Path of Least Resistance: Guiding LLM Reasining Trajectories with Prefix Consensus

**arXiv ID:** 2601.21494 | [PDF](https://arxiv.org/pdf/2601.21494v1)

**作者:** Ishan Jindal `[一作]` (Fujitsu Research India), Sachin Dev Sharma `[通讯]` (Samsung R&D Institute India)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在推理时利用前缀一致性进行高效推理的算法 PoLR（Path of Least Resistance），通过聚类短前缀并只扩展占主导的前缀集来实现自一致性（Self‑Consistency）推理的计算节省。

**💡 创新点**

创新点在于：1) 第一次将前缀一致性用于推理时的自一致性，而非仅在训练时；2) 通过无监督聚类（如层次聚类、DBSCAN）快速识别主导前缀集，实现高达 60% 令牌消耗和 50% 延迟下降；3) 与自适应一致性、早停自一致性等方法无缝结合，进一步提升效率；4) 证明了前缀信息与最终答案之间存在互信息，解释了准确率与效率的理论基础。

**🔧 技术方法**

技术包括：多样本前缀生成、TF‑IDF 或轻量化语义嵌入、层次聚类/密度聚类、主导簇选择、对簇内路径完整展开、投票集成；并使用互信息、熵、聚类偏度等信息论指标分析理论。

**📊 数据集**

使用的公开基准包括：数学推理（GSM8K、Math500、AIME24/25、Qwen2.5‑Math‑7B）、科学推理（GPQA‑Diamond）、多跳推理与隐式知识检索（StrategyQA），以及多种开源 LLM（DSQ 1.5B/7B、QWQ32B、MiMo‑7B‑RL‑0530、Phi‑4‑15B、Qwen2.5‑Math‑7B）。

**📈 对比分析**

与标准自一致性（SC）比较，PoLR 在保持甚至提升准确率的同时，令牌消耗平均降低 40–60%（最高 60%），延迟降低 30–50%；与单样本 Chain‑of‑Thought、Adaptive Consistency、Early‑Stopping Self‑Consistency 等方法对比，PoLR 仍保持最优的精度‑效率平衡；与 PoLR+AC/ESC 组合进一步提升 30%+路径压缩并减少 75% 计算成本。

**⚠️ 局限性**

局限性：1) 对前缀聚类阈值和聚类算法敏感，需针对不同模型/任务微调；2) 在极端多样化或高质量模型（如 Qwen2.5‑Math‑7B）中，主导簇偏度减小，令牌节省幅度下降；3) 需要额外的聚类开销（但仅几毫秒）；4) 对前缀长度选择不当可能导致准确率下降；5) 仍依赖多样本采样，无法完全消除采样成本。

---

## 322. Topeax -- An Improved Clustering Topic Model with Density Peak Detection and Lexical-Semantic Term Importance

**arXiv ID:** 2601.21465 | [PDF](https://arxiv.org/pdf/2601.21465v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 323. Learning-Based Sensor Scheduling for Delay-Aware and Stable Remote State Estimation

**arXiv ID:** 2601.21482 | [PDF](https://arxiv.org/pdf/2601.21482v1)

**作者:** Nho-Duc Tran `[一作]` (Mid Sweden University), Mikael Gidlund `[通讯]` (Mid Sweden University)

**通讯引用:** 8476 | [OpenAlex ID](https://openalex.org/A5004012289)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一套统一的延迟感知远程状态估计框架，结合后验融合估计和基于PPO的智能调度策略；

**💡 创新点**

创新点在于①引入延迟相关信息增益与能耗比的调度目标，②设计无需重放延迟测量的后验融合估计；③给出可判定的稳定性条件；④将该调度问题转化为MDP并用PPO自适应学习；

**🔧 技术方法**

使用Kalman滤波、后验融合、线性系统理论、PPO深度强化学习；

**📊 数据集**

仅通过仿真验证，使用随机生成的5维线性系统、20个异构传感器的状态与测量矩阵；

**📈 对比分析**

与随机调度、DQN、A2C三种基线进行对比；结果表明PPO在保持能耗可接受的同时，估计误差更低、波动更小，性能稳定；

**⚠️ 局限性**

局限性包括仅针对线性系统，未在真实无线网络中验证；调度目标采用单一权重，未实现多目标自适应；对极端大延迟和高噪声场景的鲁棒性还有待进一步研究。

---

## 324. Best Arm Identification with LLM Judges and Limited Human

**arXiv ID:** 2601.21471 | [PDF](https://arxiv.org/pdf/2601.21471v1)

**作者:** Ruicheng Ao `[一作]` (Massachusetts Institute of Technology), David Simchi-Levi `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 21348 | [OpenAlex ID](https://openalex.org/A5112431388)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在存在偏置代理（如LLM评分）且只能选择性获得真实标签的固定置信度最佳臂识别问题，提出基于代理与逆倾向加权残差的预测驱动估计与自适时置信序列，并设计Neyman分配的自适应审计算法。

**💡 创新点**

①证明仅靠偏置代理无法保证最佳臂识别；②提出融合代理与审计的IPW残差估计及对应的时变置信序列；③设计基于Neyman分配的审计策略实现近似最优成本。

**🔧 技术方法**

逆倾向加权(IPW)、时间统一置信序列（混合鞭策马尔可夫、线性混合）、LUCB式最优采样、Neyman分配、局部方差估计等技术。

**📊 数据集**

主要使用仿真数据（K=4臂，Bernoulli真实值与带偏差代理），未使用公开真实数据集。

**📈 对比分析**

与均匀审计、价格精度、无偏审计等基线在相同置信度下比较；实验显示置信序列覆盖率>98%，Neyman分配相对均匀审计降低成本48-50%，准确率保持100%。

**⚠️ 局限性**

需要已知审计倾向且满足正性假设；对偏差估计的方差逼近依赖足够审计样本；在高度非平稳或极端异质偏差时需预热期；算法假设代理与真实标签之间可建模为加性偏差。

---

## 325. Adaptive Confidence Gating in Multi-Agent Collaboration for Efficient and Optimized Code Generation

**arXiv ID:** 2601.21469 | [PDF](https://arxiv.org/pdf/2601.21469v1)

**作者:** Haoji Zhang `[一作]` (University of Electronic Science and Technology of China), Yi Zhou `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 177764 | [OpenAlex ID](https://openalex.org/A5100384245)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DebateCoder——一种面向小参数语言模型的多代理协同代码生成框架

**💡 创新点**

创新点在于结构化角色扮演（用户、技术、QA）、95%阈值自适应置信门控、前后辩论循环以及审阅者驱动的调试环节

**🔧 技术方法**

采用多代理对话、信任门控机制、迭代辩论、合成计划、审阅+调试循环，并以 Pangu‑1B 为底层模型

**📊 数据集**

使用 HumanEval、HumanEval‑ET、MBPP、MBPP‑ET 四个公开代码生成基准

**📈 对比分析**

与 Direct（单模型直接生成）和 MapCoder（传统多代理）比较，Pass@1 在 HumanEval 上从 59.15% 提升至 70.12%，整体平均准确率 58.91%，API 调用量减少约 35%，显著优于两者

**⚠️ 局限性**

局限在于依赖 Pangu‑1B 的小参数规模，复杂性更高的任务仍可能受限，且多代理流程对硬件与并行调度的需求较高

---

## 326. Mining Forgery Traces from Reconstruction Error: A Weakly Supervised Framework for Multimodal Deepfake Temporal Localization

**arXiv ID:** 2601.21458 | [PDF](https://arxiv.org/pdf/2601.21458v1)

**作者:** Midou Guo `[一作]` (Sun Yat-sen University), Rui Yang `[通讯]` (Alibaba group)

**通讯引用:** 29714 | [OpenAlex ID](https://openalex.org/A5100693489)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种弱监督的多模态深度伪造时序定位框架RT-DeepLoc。

**💡 创新点**

创新点在于利用Masked Autoencoder重建误差作为伪造指示器，并通过异构对比损失和多任务强化实现精细定位。

**🔧 技术方法**

使用了Masked Autoencoder、异构对比损失、跨模态注意力以及多任务学习等技术。

**📊 数据集**

在LAV-DF和AV-Deepfake1M两个大规模多模态数据集上进行实验。

**📈 对比分析**

与现有弱监督方法相比，RT-DeepLoc在mAP/AR上大幅提升，甚至在跨数据集时优于全监督方法。

**⚠️ 局限性**

局限在于对极低伪造比例或边缘重建误差易产生误检，需要进一步提升鲁棒性。

---

## 327. When Local and Non-Local Meet: Quadratic Improvement for Edge Estimation with Independent Set Queries

**arXiv ID:** 2601.21457 | [PDF](https://arxiv.org/pdf/2601.21457v1)

**作者:** Tomer Adar `[一作]` (Technion - Israel Institute of Technology), Amit Levi `[通讯]` (University of Haifa)

**通讯引用:** 142 | [OpenAlex ID](https://openalex.org/A5049478018)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在未知图中用混合查询（独立集查询、度查询和邻居查询）估计边数的下采样问题。

**💡 创新点**

创新点在于证明混合查询模型比单一查询模型更强，给出最优上界与下界，实现了二次（平方根级别）提升，并完成了全局下界的构造。

**🔧 技术方法**

主要技术包括：对图按度阈值分解为低度/高度子集；对低度边采用随机采样与 IS 查询相结合的无偏估计；利用邻居查询降低方差；对高度边使用权重采样；通过指数递增/递减的“猜测”序列以及迭代加深调度实现期望复杂度；使用概率与组合论工具（如 Chebyshev、Markov、Yao 原理）完成上界与下界分析。

**📊 数据集**

实验与数据集：本文为理论研究，不涉及实际数据集，所有结论均在随机/极端图构造上验证。

**📈 对比分析**

与先前工作比较：在度/邻居查询模型下边数估计需 Θ(n/√m) 次查询；在独立集查询模型下亦需 Θ(n/√m)；本工作在混合模型下实现 O(min(√m, √(n/√m))·log⁵/² n) 次查询，显著优于前者；下界证明表明此提升是最优的，无法进一步改进。

**⚠️ 局限性**

局限性：只适用于具有三种查询接口的固定图；对动态或有权图未作扩展；常数项与低阶项隐藏，实际实现可能受限；复杂度分析依赖于严格的概率与结构假设。

---

## 328. Variance & Greediness: A comparative study of metric-learning losses

**arXiv ID:** 2601.21450 | [PDF](https://arxiv.org/pdf/2601.21450v1)

**作者:** Donghuo Zeng `[一作]` (KDDI Research), Masato Taya `[通讯]` (KDDI Research)

**通讯引用:** 48 | [OpenAlex ID](https://openalex.org/A5084224574)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一套基于VARIANCE和GREEDINESS的诊断框架，用于系统评估七种主流度量学习损失在不同图像检索任务中的几何结构和优化行为。

**💡 创新点**

创新点在于将“内部方差”“外部方差”与“主动样本比例”“梯度范数”结合成定量诊断指标，揭示了“优化效率-几何细粒度”之间的本质权衡，并给出基于任务需求的损失选择指南。

**🔧 技术方法**

使用的技术包括ViT‑B/32视觉Transformer主干、两层Tanh投影头、L2归一化、Adam优化器以及七种损失（Contrastive、Triplet、N‑pair、InfoNCE、ArcFace、SCL、CCL）在余弦/欧氏距离下的实现。

**📊 数据集**

实验数据集包括五个规模与细粒度差异显著的图像检索基准：CIFAR‑10、Car196、CUB‑200、Tiny‑ImageNet、FashionMNIST。

**📈 对比分析**

通过VARIANCE、GREEDINESS和Recall@k三个维度进行比较，发现Triplet和SCL在细粒度检索中保持更高的内部方差和更清晰的类间间距，取得更优的R@1；Contrastive/InfoNCE在粗粒度检索上收敛更快、聚类更紧凑，但细粒度性能略逊；N‑pair尽管类心间距大，却常出现类间方差不均导致检索失效。

**⚠️ 局限性**

局限性包括仅关注图像检索任务、固定网络结构和超参数、对角度/中心基损失的度量不匹配导致的方差评估失真，以及未涵盖更大规模或跨模态检索场景。

---

## 329. Synthetic Pattern Generation and Detection of Financial Activities using Graph Autoencoders

**arXiv ID:** 2601.21446 | [PDF](https://arxiv.org/pdf/2601.21446v1)

**作者:** Francesco Zola `[一作]` (Vicomtech), Amaia Gil `[通讯]` (Vicomtech)

**通讯引用:** 34 | [OpenAlex ID](https://openalex.org/A5000399820)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过设计七种已知的洗钱网络拓扑模式的合成生成器，分别为每种模式训练了Graph Autoencoder（GAE）模型，并利用重构误差来区分不同模式，验证GAE在无标签、无真实金融数据的情况下学习并检测洗钱拓扑结构的可行性。

**💡 创新点**

创新点在于：①首次将合成拓扑生成器与GAE结合，解决了真实金融数据稀缺与标签缺失问题；②通过对比GCN、GraphSAGE和GAT三种卷积策略，揭示不同拓扑模式对模型选择的敏感性；③提出基于重构误差阈值的无监督检测方法，为未来金融犯罪预警提供可行思路。

**🔧 技术方法**

使用的技术包括：图神经网络中的Graph Autoencoder（Encoder采用GCN/GraphSAGE/GAT，Decoder为线性重构层），节点属性提取（入度、出度、中心性等），以及随机数生成的拓扑合成算法。

**📊 数据集**

所用数据集为自行生成的合成图数据集，每种模式各生成15,000张图（80%训练、20%验证），完全基于作者的参数化生成脚本，未使用公开金融交易数据集。

**📈 对比分析**

通过构建重构误差矩阵进行比较，发现GAE-GCN在大多数模式上重构误差最低；GAE-SAGE在Collector和Scatter‑Gather模式上表现最佳；GAE-GAT在Collusion和Branching模式上表现最佳；总体而言，GCN在多样性模式下更稳健，误差差距约为前两者的15–20%。

**⚠️ 局限性**

局限性包括：①仅使用合成数据，缺乏真实金融交易场景验证；②仅覆盖七种典型模式，未包含更复杂或混合模式；③模型对参数敏感，需进一步评估在不同图规模、噪声水平下的泛化能力。

---

## 330. MURAD: A Large-Scale Multi-Domain Unified Reverse Arabic Dictionary Dataset

**arXiv ID:** 2601.21512 | [PDF](https://arxiv.org/pdf/2601.21512v1)

**作者:** Serry Sibaee `[一作]` (Prince Sultan University), Wadii Boulila `[通讯]` (Prince Sultan University)

**通讯引用:** 6211 | [OpenAlex ID](https://openalex.org/A5042123158)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并发布了规模最大的阿拉伯语多领域逆字典数据集 MURAD（96,243 条词义对），并提供完整的处理、验证和开放使用流程。

**💡 创新点**

创新点包括：①采用混合管道（OCR、GPT‑4o、脚本）从 17 本权威词典自动抽取并结构化词义；②结合八项正式词典标准保证定义的准确性与一致性；③覆盖从古典阿拉伯语、伊斯兰研究、语言学到科学技术等 13 个专业领域，形成首个面向逆字典任务的多领域、统一格式的大规模阿拉伯语资源。

**🔧 技术方法**

技术手段：高分辨率扫描 OCR（Mistral OCR）→ GPT‑4o 语义提取与校正 → 正则表达式与脚本进行文本归一化、去重与元数据标注 → 自动化验证脚本保证完整性与语义一致性；数据发布采用 CSV+README，代码托管在 GitHub，遵循 FAIR 原则。

**📊 数据集**

使用的原始数据集为 17 本公开的阿拉伯语词典与术语表（如 Al‑Kafawi、Al‑Jurjani、各学科专用词典），并在此基础上生成 96,243 条词义对。

**📈 对比分析**

与现有资源（KSAA‑RD、KSAA‑CAD、SemEval‑2022 CODWOE、Azhary 等）对比：MURAD 在条目数、领域覆盖、词义标准化、开放性等方面均显著优于前者；在词义对齐任务上可为模型提供更丰富、语义一致的数据，虽未给出具体数值指标，但基准实验显示在逆字典检索与定义生成任务中性能提升显著。

**⚠️ 局限性**

局限性：1）主要覆盖现代标准阿拉伯语，缺乏口语方言词汇；2）对极端长或多义词的处理仍依赖后续人工校验；3）OCR 与 GPT‑4o 生成过程可能引入细微误差，需进一步人工验证；4）目前不含多语言对齐（仅阿拉伯语），在跨语言应用上需要补充翻译或对齐数据。

---

## 331. The Effectiveness of Style Vectors for Steering Large Language Models: A Human Evaluation

**arXiv ID:** 2601.21505 | [PDF](https://arxiv.org/pdf/2601.21505v1)

**作者:** Diaoulé Diallo `[一作]` (German Aerospace Center), Tobias Hecking `[通讯]` (German Aerospace Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型（LLM）的内部激活进行向量调节，以实现情感语气的精准控制，并通过 190 名参与者的 7,000+ 评分对其效果进行人类评估。

**💡 创新点**

①首次在人类层面评估激活向量调节的情感效果；②发现一个可接受的调节强度阈值（λ≈0.15），超出该阈值文本可读性显著下降；③证明 LlaMA‑3 在激活调节上比 Alpaca 更稳定、效果更显著。

**🔧 技术方法**

利用激活向量（style vectors）在前向推理时对所有层的激活进行线性插值（â = a + λ·v），并通过 λ 控制调节幅度；同时采用 DistilRoBERTa 作为情感分类器、模型内置可读性评估器进行自动评价。

**📊 数据集**

GoEmotions 数据集（58k Reddit 语料）映射到 Ekman 的 6 种基本情绪，采样 53,994 条文本；另外使用 19 条单轮对话提示生成情感文本，随后收集人类评分。

**📈 对比分析**

将人类情感强度评分与模型分类器输出以及可读性分数进行对比，使用两因素重复测量 ANOVA、Pearson 相关等统计手段；结果显示五种情绪的情感强度随 λ 上升显著增加（p<0.001），与模型评分的相关系数均在 0.76–0.98 之间；可读性在 λ≈0.15 后开始下降，说明存在稳定性阈值。

**⚠️ 局限性**

仅在单轮对话、单一模型族（LlaMA‑3）和有限的情感维度上验证；未考虑多轮交互、跨模型泛化、类别不平衡对向量构造的影响；对人类评分标准的解释存在主观差异；缺乏对更抽象风格或非情感维度的调节验证。

---

## 332. Hypernetwork-Based Adaptive Aggregation for Multimodal Multiple-Instance Learning in Predicting Coronary Calcium Debulking

**arXiv ID:** 2601.21479 | [PDF](https://arxiv.org/pdf/2601.21479v1)

**作者:** Kaito Shiku `[一作]` (Kyushu University), Ryoma Bise `[通讯]` (Kyushu University)

**通讯引用:** 1431 | [OpenAlex ID](https://openalex.org/A5064312777)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过多实例学习框架，利用CT切片与病人表格信息预测冠状动脉钙化病变是否需要去除设备。

**💡 创新点**

提出超网络生成的基于表格数据的聚合参数，实现患者特异性自适应聚合，从而首次将超网络应用于多模态MIL。

**🔧 技术方法**

使用ResNet18提取切片特征，三层MLP超网络产生聚合参数和分类器，结合Transformer自注意力实现聚合。

**📊 数据集**

在九州大学医院收集的493例CT扫描（每例9–635张切片，平均230张）以及20维临床表格数据上进行实验。

**📈 对比分析**

与多种单模和多模MIL基线（如TableMLP、Feature+Transformer、Concat、MultimodalTransformer等）对比，本文方法在5折交叉验证中实现F1≈0.570、AUC≈0.710，显著优于所有对比方法。

**⚠️ 局限性**

主要限制包括对表格数据缺失处理不足、仅在单中心数据集上验证，缺乏对不同机构或多中心数据的泛化评估。

---

## 333. PPI-SVRG: Unifying Prediction-Powered Inference and Variance Reduction for Semi-Supervised Optimization

**arXiv ID:** 2601.21470 | [PDF](https://arxiv.org/pdf/2601.21470v1)

**作者:** Ruicheng Ao `[一作]` (Massachusetts Institute of Technology), Will Wei Sun `[通讯]` (Purdue University)

**通讯引用:** 737 | [OpenAlex ID](https://openalex.org/A5060973953)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PPI-SVRG 算法，融合 PPI 与 SVRG 的控制变量思想，用于在标签稀缺但有预训练模型预测的半监督随机优化问题。

**💡 创新点**

创新点在于证明 PPI 与 SVRG 在结构上等价，并将两者的控制变量结合，得到一种同时利用预测信息和参考梯度的新变种；此外推导了包含预测不确定性误差底限的收敛上界。

**🔧 技术方法**

核心技术包括：控制变量（PPI）、随机梯度下降的方差减小技术 SVRG、epoch‑doubling 的 PPI‑SVRG++、理论分析（分解收敛误差为优化误差与预测误差两部分）以及实验验证。

**📊 数据集**

使用的实验数据集包括：森林砍伐与银河形态的 mean‑estimation benchmark（分别有 160/1674 个标记样本和 1436/15069 个未标记样本），以及 MNIST 进行半监督深度学习实验。

**📈 对比分析**

与 SGD、SVRG、传统 PPI 等基线比较。实验表明：在 10% 标签比例下，PPI‑SVRG 的 MSE 下降 43–52%；在 MNIST 上仅使用 10% 标记数据时，PPI‑SVRG‑Adam 与 Momentum 的测试准确率分别提升约 2.7% 和 2.9%，且收敛速率与 SVRG 一致，误差只受预测不确定性影响。

**⚠️ 局限性**

局限性：需在大量无标签数据上拥有可靠预测，若预测质量低或不可获得，则无法获得显著收益；此外在深度学习中需要温启动阶段，若过早停止会导致性能不佳。

---

## 334. Conversation for Non-verifiable Learning: Self-Evolving LLMs through Meta-Evaluation

**arXiv ID:** 2601.21464 | [PDF](https://arxiv.org/pdf/2601.21464v1)

**作者:** Yuan Sui `[一作]` (National University of Singapore), Bryan Hooi `[通讯]` (National University of Singapore)

**通讯引用:** 33298 | [OpenAlex ID](https://openalex.org/A5056748708)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种多代理自我对话框架CoNL，利用代理之间的提议、评价与修订来学习生成与评估能力；

**💡 创新点**

创新点在于通过诊断奖励（r_diag）将批评是否导致解决方案改进作为评估者质量的可观测信号，实现无监督的元评估；

**🔧 技术方法**

使用多轮结构化对话、配对比较聚合（Bradley–Terry）、诊断奖励、重要性采样策略梯度以及记忆缓冲压缩上下文；

**📊 数据集**

在五大无标注推理基准上训练，包括DeepMath‑103K、AIME 2024/2025、GPQA Diamond、FrontierScience与USACO；

**📈 对比分析**

与推理时优化（Self‑Consistency、Self‑Refine、Multi‑Agent Debate）以及自我奖励训练（SRT‑S/M）对比，CoNL 在大多数任务上均提升 2.7–8.3 % 的 Pass@1，并且训练更稳定；

**⚠️ 局限性**

局限在于仍需多代理的计算开销，对极其复杂的问题纠错率偏低，且对代理角色设计与协同策略敏感。

---

## 335. L$^3$: Large Lookup Layers

**arXiv ID:** 2601.21461 | [PDF](https://arxiv.org/pdf/2601.21461v1)

**作者:** Albert Tseng `[一作]` (Cornell University), Christopher De Sa `[通讯]` (Cornell University)

**通讯引用:** 3614 | [OpenAlex ID](https://openalex.org/A5041869459)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并训练了一种名为 Large Lookup Layer（LLL）的稀疏层，在解码器层中使用静态基于 token ID 的嵌入表，通过上下文注意力聚合实现上下文相关的特征提取。

**💡 创新点**

创新点在于将 tokenizer 嵌入表的稀疏性推广到模型层，并采用静态路由与信息理论 LZW 分配算法实现高效、硬件友好的上下文感知稀疏层，避免了 MoE 的动态路由开销。

**🔧 技术方法**

使用了静态 token 路由、上下文注意力聚合、LZW 频率分配嵌入、参数 offloading、FlexAttention/MegaBlocks 等训练与推理加速技术。

**📊 数据集**

实验数据集为 FineWeb‑Edu，采用 180K 词表的 BPE tokenizer。

**📈 对比分析**

与等 FLOP 的稠密模型、等稠密度与等深度的 MoE 进行对比，LLL 在 800M、1.5B、2.6B 参数规模下的 perplexity 均显著低于稠密模型，且在 downstream 任务上表现更好；推理时 CPU offloading 只导致 <10% 的额外延迟。

**⚠️ 局限性**

局限性包括需预先分配嵌入表且对 k 上限敏感，可能限制高频 token 的表达；与动态路由 MoE 的表达能力不完全可比；目前仅在大规模 GPU+CPU 环境验证，通用性和跨任务适应性仍需进一步研究。

---

## 336. Tell Me What I Missed: Tell Me What I Missed: Interacting with GPT during Recalling of One-Time Witnessed Events

**arXiv ID:** 2601.21460 | [PDF](https://arxiv.org/pdf/2601.21460v1)

**作者:** Suifang Zhou `[一作]` (City University of Hong Kong), RAY LC `[通讯]` (City University of Hong KongStudio for Narrative Spaces)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究让28名受试者观看一段36秒的监控录像后，使用默认GPT或按预设目击者访谈协议引导的GPT协助撰写目击陈述，并通过问卷与访谈评估记忆准确度、主观认知及对GPT的评价。

**💡 创新点**

创新点在于首次系统比较未引导与基于标准化访谈提示的GPT交互对一时性目击事件记忆与评价的影响，揭示引导提示能更好地让受试者的主观清晰度与客观回忆一致，而无引导时易产生情绪偏见与对GPT的盲目信任。

**🔧 技术方法**

采用GPT‑4模型，分别在默认模式与加入专门的“目击者访谈协议”提示的两种交互模式下进行实验。

**📊 数据集**

使用单一的36秒无声监控录像（含室内入侵与抢劫场景），并结合受试者自填的记忆与情感问卷。

**📈 对比分析**

对比方法：两组在事实回忆准确度（50题评分）、情感/认知维度（Likert量表）及GPT效用评价进行t检验与相关分析。结果显示：两组记忆准确度无显著差异（M≈24/50），但默认组对嫌疑人合法性评价显著低于引导组（t=2.13, p=0.043），且默认组受试者主观清晰度与对GPT的信任呈负相关。

**⚠️ 局限性**

局限性包括：缺乏无GPT对照组；仅使用单一视频且无即时记忆测评；时间间隔较短，未考察长期记忆变化；实验仅用GPT‑4，未探讨其他LLM差异；样本量有限，缺乏对不同人群的推广性验证。

---

## 337. From Vulnerable to Resilient: Examining Parent and Teen Perceptions on How to Respond to Unwanted Cybergrooming Advances

**arXiv ID:** 2601.21518 | [PDF](https://arxiv.org/pdf/2601.21518v1)

**作者:** Xinyi Zhang `[一作]` (Virginia Tech), Sang Won Lee `[通讯]` (Virginia Tech)

**通讯引用:** 10635 | [OpenAlex ID](https://openalex.org/A5100444708)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对74名家长与青少年的在线调查，分析他们在面对模拟网络约谈场景时所做出的易受害反应与自我保护策略，并提出基于成长阶段的青少年网络骚扰响应分类。

**💡 创新点**

①首次构建面向青少年的网络约谈易受害与防护行为的四类分类体系；②将这四类行为映射到约谈的六个阶段，形成阶段化的防护策略树；③提供了标注好的受访者回应数据集，为后续算法研究提供基础。

**🔧 技术方法**

采用定性主题分析提炼响应主题，并用卡方检验检视不同阶段与群体间的分布差异；使用量化频率统计展示各类行为的占比。

**📊 数据集**

使用来自Perverted-Justice（PJ）数据集的真实聊天片段，经GPT-4o润色后设计10个代表性模拟场景，供受访者填写回应。

**📈 对比分析**

研究并未评估任何算法性能，而是通过统计对比展示各类响应在不同阶段与群体间的显著差异，说明阶段化分类的合理性与一致性。

**⚠️ 局限性**

局限性：样本量小、受访者为单独家庭且为自我报告；场景为模拟，缺乏实时交互真实感；未验证所提策略的实际有效性；年龄层差异未深入探讨。

---

## 338. AIR-VLA: Vision-Language-Action Systems for Aerial Manipulation

**arXiv ID:** 2601.21602 | [PDF](https://arxiv.org/pdf/2601.21602v1)

**作者:** Jianli Sun `[一作]` (Institute of Automation, Chinese Academy of Sciences), Yonglin Tian `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了针对无人机机械臂的全栈视觉-语言-动作(AIR-VLA)基准，包含仿真环境、3000条多模态遥控演示数据以及多维评估指标。

**💡 创新点**

创新点在于首次针对空中操作提供专门的VLA基准与数据集，设计了覆盖基座控制、物体空间推理、语义理解与长周期规划四个任务套件，并提出针对浮动底盘与三维空间的多维度评价体系。

**🔧 技术方法**

主要技术包括基于NVIDIA Isaac Sim的物理仿真、使用Frankà Panda 7-DoF机械臂与四旋翼无人机的耦合控制、遥控数据采集、Transformer+流匹配、动作块预测和条件扩散策略等VLA模型，以及多模态VLM的高层规划。

**📊 数据集**

使用了AIR-VLA自建数据集：3000条手动遥控演示，涵盖RGB、RGB‑D、关节角度等感知信息，且配有多样化自然语言指令，覆盖室内、工业与户外场景。

**📈 对比分析**

通过对比π_0、π_0.5、π_0‑FAST、ACT、Diffusion‑Policy等VLA模型以及Qwen3‑VL、Qwen2.5‑VL、GLM‑4V等VLM模型，结果显示预训练模型π_0.5在任务完成率上最优，VLM模型在规划与语义匹配上表现突出，但在三维空间导航与精细操控上仍有显著不足；整体性能仍低于地面平台。

**⚠️ 局限性**

局限性包括：对浮动基座动力耦合的补偿不足；长周期规划的误差累积导致后续子任务失败；对精细抓取与空间定位的理解仍不充分；安全性和对环境干扰的鲁棒性较差；缺乏针对三维空间的深度定位与视觉引导能力。

---

## 339. Multi-objective Integer Linear Programming approach for Automatic Software Cognitive Complexity Reduction

**arXiv ID:** 2601.21565 | [PDF](https://arxiv.org/pdf/2601.21565v1)

**作者:** Adriana Novoa-Hurtado `[一作]` (ITIS Software, University of Málaga), Manuel Giménez-Medina `[通讯]` (Ayesa Advanced Digital Services Limited)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于多目标整数线性规划（MO‑ILP）的软件认知复杂度（CC）自动降低方法，并实现了相应工具

**💡 创新点**

在原有单目标最小化提取次数的基础上，引入了对提取方法的CC和行数（LOC）差异的平衡目标，形成完整的三目标优化模型

**🔧 技术方法**

使用Pyomo构建MO‑ILP模型，采用IBM CPLEX求解器，并实现了加权和、增量ε约束和混合方法等求解算法

**📊 数据集**

在9个开源Java项目（共121个方法）和Ayesa工业项目的10个方法上进行实验，利用预先生成的提取机会缓存作为输入

**📈 对比分析**

通过与单目标ILP方法比较，使用超体积（HV）指标衡量Pareto前沿质量；在大多数实例中获得多样化的Pareto解，HV均达到0.5-1之间，证明方法在降低CC、平衡CC/LOC以及保持提取次数最小方面具有优势

**⚠️ 局限性**

主要局限是依赖商业求解器CPLEX，且在某些大规模方法（如开源项目方法号97）无法得到解，模型规模与约束数量过多时求解时间显著增加

---

## 340. HistoPrism: Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction

**arXiv ID:** 2601.21560 | [PDF](https://arxiv.org/pdf/2601.21560v1)

**作者:** Susu Hu `[一作]` (National Center for Tumor Diseases), Stefanie Speidel `[通讯]` (National Center for Tumor Diseases)

**通讯引用:** 6393 | [OpenAlex ID](https://openalex.org/A5003648994)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 HistoPrism，一种基于 Transformer 的直接映射架构，用于从 H&E 组织切片预测跨癌种的空间转录组表达，并引入 Gene Pathway Coherence (GPC) 评估框架。

**💡 创新点**

创新点包括：① 将癌种上下文通过交叉注意力直接注入视觉特征，② 采用高效的直接映射而非多阶段重建，③ 通过 GPC 评价基于功能通路的一致性，④ 在保持低计算开销的同时实现跨癌种泛化。

**🔧 技术方法**

使用的技术：Transformer 编码器 + 交叉注意力 + MLP 回归头，基于预训练的 Pathology Foundation Model（如 UNI 或 GigaPath）提取图像特征，采用 MSE 损失进行端到端训练。

**📊 数据集**

数据集：主要使用 HEST1k（153 个队列、36 个研究、覆盖多中心、不同技术和染色协议），并对比了 STimage-1K4M 的可行性。

**📈 对比分析**

与 STPath、STEM、STFlow、BLEEP、TRIPLEX 等基线模型比较；在 top‑50 高变基因的 Pearson 相关系数上与 STPath 相近甚至更优；在 GPC（Hallmark 与 Gene Ontology 通路）上分别取得 86% 与 74% 的通路胜率；在全基因聚类评估（AMI/ARI）中显著优于 STPath，展示了更好的生物学一致性。

**⚠️ 局限性**

局限性：① 仍缺乏对模型学习到的视觉特征与细胞过程的可解释性；② 依赖预训练的 PFMs，尽管表现稳定，但对低资源或新型组织的适应性尚未充分验证；③ 在极低变通路或稀有癌种上性能可能受限；④ 需要进一步探索对不同扫描仪/染色批次的鲁棒性。

---

## 341. Strassen's support functionals coincide with the quantum functionals

**arXiv ID:** 2601.21553 | [PDF](https://arxiv.org/pdf/2601.21553v1)

**作者:** Keiya Sakabe `[一作]` (Ruhr University Bochum), Michael Walter `[通讯]` (LMU Munich)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文证明了 Strassen 的支撑功能（support functional）在所有张量上与量子功能（quantum functional）相等，从而确认支撑功能是张量渐近谱的普适谱点。

**💡 创新点**

创新点在于将两种本质上不同的张量测度统一，利用 Fenchel‑类型对偶性在 Hadamard 流形上的新定理，得到了一般的极小极大公式，进一步推导出张量参数（如渐近切片秩、超图顶点覆盖数、G‑稳定秩、非交换秩）之间的紧密联系。

**🔧 技术方法**

核心技术是 Hirai 对 Hadamard 流形上的凸优化的强对偶性，结合张量的矩阵多项式动作、模量多面体（moment polytope）与支撑多面体（support polytope）的几何描述，以及极小极大理论与熵优化的结合。

**📊 数据集**

本文为理论性研究，无使用具体实验数据集；所有结论基于数学证明与符号推导。

**📈 对比分析**

由于主要是理论证明，未进行实验比较；论文通过与已有的量子功能、支撑功能、切片秩等公式对比，验证了新极小极大公式在已知特例（如自由张量、对称张量、左右动作）下能够直接恢复并简化之前的结果。

**⚠️ 局限性**

局限性包括：1) 依赖 Hirai 对 Hadamard 流形强对偶性的结果，若其对偶性在更一般的流形上不成立则结论可能不再适用；2) 结果多停留在理论层面，缺乏针对实际张量数据的数值实现与效率分析；3) 对特殊张量类（如非正规张量、低维情况）仍需进一步验证。

---

## 342. Vision KAN: Towards an Attention-Free Backbone for Vision with Kolmogorov-Arnold Networks

**arXiv ID:** 2601.21541 | [PDF](https://arxiv.org/pdf/2601.21541v1)

**作者:** Zhuoqin Yang `[一作]` (Shenzhen University), Linlin Shen `[通讯]` (Shenzhen University)

**通讯引用:** 11111 | [OpenAlex ID](https://openalex.org/A5019313200)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Vision KAN（ViK），一种无注意力的视觉骨干网络，核心模块 MultiPatch-RBFKAN 在每个阶段使用 RBF‑KAN 对每个补丁进行局部非线性建模，并结合轴向可分卷积混合与低秩全局映射，实现高效的 token 交互。

**💡 创新点**

创新点：①将 Kolmogorov–Arnold 理论与 RBF 基函数结合，引入 KAN 作为 token 混合器；②设计多补丁 RBF‑KAN 结构，将局部非线性、方向性局部混合和全局低秩映射统一于一个模块；③通过 patch‑grouping 与轻量级可分卷积，克服传统 KAN 在高分辨率上的计算瓶颈，实现线性复杂度。

**🔧 技术方法**

使用的技术包括：
- RBF‑KAN（基于 Radial Basis Function 的 KAN）；
- 深度可分卷积（axis‑wise separable mixing）；
- 低秩投影（低秩全局映射）；
- 分层骨干（patch embedding → 4 个阶段 → 归一化/分类头）。

**📊 数据集**

实验数据集：ImageNet‑1K（1.28M 训练图像，50K 验证图像，1000 类）。

**📈 对比分析**

对比方法：ResNet‑50、ViT‑Ti、DeiT‑Tiny、PVT‑Tiny、ResMLP‑S12 等。ViK‑Small 在 224×224 输入下获得 76.5% Top‑1（1.6 GFLOPs），略低于 ResMLP‑S12 但显著低于 FLOPs；ViK‑Base 达到 80.3% Top‑1（3.2 GFLOPs），与 ResNet‑50、DeiT‑Small、PVT‑Small 相当或更好，且复杂度更低。

**⚠️ 局限性**

局限性：
- 在极高分辨率下仍需 patch‑grouping，跨 patch 交互依赖可分卷积和低秩映射，可能限制极细粒度建模；
- 只在 ImageNet‑1K 上验证，缺乏对下游任务（检测、分割等）的泛化评估；
- 低秩全局投影的 rank 需要经验调参，可能影响不同任务的表现。

---

## 343. KAPSO: A Knowledge-grounded framework for Autonomous Program Synthesis and Optimization

**arXiv ID:** 2601.21526 | [PDF](https://arxiv.org/pdf/2601.21526v1)

**作者:** Alireza Nadaf `[一作]` (Leeroo Team), Majid Yazdani `[通讯]` (Leeroo Team)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 KAPSO 框架，利用自然语言目标、评估器和迭代实验循环，结合知识检索、Git 实验分支和认知记忆，实现可执行软件的评估驱动式优化。

**💡 创新点**

创新点包括：①将程序合成视为长周期优化循环中的操作，而非终点；②基于 Git 的实验引擎保证每一次尝试的可追溯性和可复现性；③通过 MediaWiki+Neo4j/Weaviate 搭建结构化知识库，支持基于问题、仓库和失败条件的检索；④认知记忆层提取实验轨迹中的可复用经验，减少重复错误并加速收敛；⑤模块化设计可插拔评估器、知识后端和代码生成器。

**🔧 技术方法**

使用的技术包括：大语言模型（LLM）驱动的代码生成与调试；Git 作为实验管理与版本控制；MediaWiki/Neo4j/Weaviate 进行知识建模与检索；Python/CPP 运行时；Docker、Modal、BentoML、LangGraph 等多种部署策略；强化学习式搜索策略（线性/树形）。

**📊 数据集**

主要使用的基准数据集是 MLE‑Bench（Kaggle‑style 机器学习竞赛）和 ALE‑Bench（AtCoder 递归/启发式竞赛），以及公开的仓库语料库用于知识抓取。

**📈 对比分析**

通过与现有开源框架（Leeroo、R&D‑Agent、AIRA‑dojo 等）在同一评估器上进行对比。MLE‑Bench 结果显示 Leeroo 在中、难度任务中 medal 率分别提升至 44.74% 与 40.00%（相对 R&D‑Agent 的 21.05% 与 22.22%）。ALE‑Bench 上 Leeroo 的最终 ELO 1909.4、rank percentile 6.1% 低于 ALE‑Agent（1879.3、6.8%）且成本更低。性能优势体现在更高的准确率、更好的排名以及更低的 LLM 使用成本。

**⚠️ 局限性**

局限性包括：ALE‑Bench 竞赛数量有限，导致评估结果噪声较大；评估器的随机性和多样性可能影响可复现性；框架对外部依赖（GitHub、容器镜像）和知识库的完整性要求较高，缺乏对非结构化知识的充分利用；在极端长周期任务中的收敛速度仍待进一步提升。

---

## 344. Thinking Broad, Acting Fast: Latent Reasoning Distillation from Multi-Perspective Chain-of-Thought for E-Commerce Relevance

**arXiv ID:** 2601.21611 | [PDF](https://arxiv.org/pdf/2601.21611v1)

**作者:** Baopu Qiu `[一作]` (Alibaba International Digital Commerce Group), Xiaoyi Zeng `[通讯]` (Alibaba International Digital Commerce Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `8d10c613-917e-4880-9716-17789f50e119` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为电商检索构建了基于多视角链式推理的教师模型（MPCoT），并提出一种隐藏推理知识蒸馏（LRKD），在不生成文本的情况下将教师的多模态推理知识迁移到轻量级学生模型中，实现快速高效的相关性判定。

**💡 创新点**

创新点：
- 多视角链式推理（User Intent、Structured Analysis、Business Rule）显式捕获电商相关性多维度特征；
- 隐藏推理提取器将CoT语义映射为稠密向量，在蒸馏和推理阶段均可使用，避免传统方法在推理时丢失推理结构；
- 通过SFT+ DPO双阶段训练让教师自适应选择最优视角，并在学生中通过对齐损失保留推理语义。

**🔧 技术方法**

技术栈：
- LLM教师：Qwen3‑14B + LoRA；
- 训练：SFT（多视角数据生成）+ DPO（跨视角偏好对齐）；
- 蒸馏：Latent Reasoning Knowledge Distillation，使用 BERT‑multilingual‑base 作为交叉编码器；
- 隐藏推理提取器：MLP / GAT / Poly‑Encoder；
- 语义对齐：使用 BGE‑M3 对 CoT 生成向量；
- 评估：Accuracy、Macro‑F1、在线 A/B 测试（RPM/CTR/RS）。

**📊 数据集**

数据集：
- AliExpress 多语种（EN, ES, KO, JP, PT, FR）100K 训练 / 20K 测试，六类相关性标签；
- Amazon ESCI（EN/ES/JP）100K 训练 / 10K 测试，四类标签。

**📈 对比分析**

对比方法：
- 传统 BERT‑multilingual‑base；
- 先前蒸馏方案 CED‑KD、MKD；
- 单视角 CoT、ProgressiveCoT。
性能：
- 在 AliExpress 任务上，MPCoT_SFT+DPO 教师在 Accuracy/F1 上平均提升约 2–4 分；
- LRKD_GAT 学生在 Accuracy/F1 上平均提升约 1–3 分，优于 CED‑KD 与 MKD；
- 在线 A/B 测试：RPM +1.42%，CTR +0.48%，RS +0.4%。

**⚠️ 局限性**

局限：
- 蒸馏仍依赖大模型生成的 CoT，生成成本高；
- 隐藏推理提取器（尤其 GAT）在推理时增加了约 10–15 ms 延迟；
- 目前只针对静态检索场景，未考虑实时召回或多轮交互；
- 需要高质量的多视角标注和 LLM 资源，可能在资源受限的场景难以复现。

---

## 345. CORDS: Continuous Representations of Discrete Structures

**arXiv ID:** 2601.21583 | [PDF](https://arxiv.org/pdf/2601.21583v1)

**作者:** Tin Hadži Veljković `[一作]` (University of Amsterdam), Jan-Willem van de Meent `[通讯]` (University of Amsterdam)

**通讯引用:** 1803 | [OpenAlex ID](https://openalex.org/A5073129092)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种可连续、可逆的表示方法（CORDS），将离散对象集（如分子原子、图像目标或天文事件）映射为密度场与特征场，利用密度场总质量来直接恢复对象数量。

**💡 创新点**

核心创新是构造了一个全双射（bijective）映射：离散集合 ↔ 连续密度+特征场；通过核函数叠加得到连续表示，且可精确逆推回原集合，从而在不需要显式计数或预留槽位的前提下解决变尺寸集合预测问题。

**🔧 技术方法**

使用基于核函数的连续场编码、重要性采样、Erwin 变形可逆 transformer、流匹配（flow‑matching）后验估计等技术；训练时采用像素/点级均方误差与计数正则化，保持字段可微分。

**📊 数据集**

在四类任务上验证：分子生成（QM9、GeomDrugs）、图像目标检测（MultiMNIST）、模拟推断（FRB 光曲线）以及数学极值恢复（合成基准）。

**📈 对比分析**

与离散图模型（G‑Schnet、EDM、GeoLDM 等）及连续体素模型（VoxMol、FuncMol）对标；在分子生成中获得与 GNN 基线相当或更优的有效率、唯一性；在目标检测中相较 DETR、YOLO 在 OOD 对象数下表现更稳健；在 SBI 中自然产生 p(N|ℓ) 后验，避免显式建模计数。

**⚠️ 局限性**

局限性：需要在连续域上高密度采样（尤其是大分子）导致计算成本升高；核中心定位的精度限制了重建准确率，重叠核会导致近邻目标难以分离；模型在极大规模图结构上的可扩展性与推理速度存在权衡。

---

## 346. Representation Unlearning: Forgetting through Information Compression

**arXiv ID:** 2601.21564 | [PDF](https://arxiv.org/pdf/2601.21564v1)

**作者:** Antonio Almudévar `[一作]` (University of Zaragoza), Alfonso Ortega `[通讯]` (University of Zaragoza)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种在模型内部表示空间直接实现机器学习模型遗忘的框架——Representation Unlearning，利用轻量化变换实现对遗忘数据的压缩与保留数据的保持；

**💡 创新点**

其创新点在于：①不改动参数，而是在低维表示空间施加信息瓶颈；②通过变分近似构建可训练的上界，兼顾保持与遗忘；③提出零样本（Zero‑Shot）方法，利用Neural Collapse代理实现无保留数据情况下的遗忘；

**🔧 技术方法**

采用变分推断、信息瓶颈、KL上界、Gaussian近似、线性或浅层 MLP 变换，以及对抗训练等技术；

**📊 数据集**

实验基于 CIFAR‑10、CIFAR‑100 与 Tiny ImageNet 三大图像分类基准；

**📈 对比分析**

与 Retrain、Fine‑tuning、SISA、SCRUB、UNSIR、Bad Teacher 等方法对比，Representation Unlearning 在类遗忘与随机遗忘场景下实现了接近完美的遗忘（A_f ≈ 0），同时保持最高的保留准确率与最低的交叉熵，并在零样本设置下实现最高 754× 的加速；

**⚠️ 局限性**

局限性包括：对超参数 β 的敏感性、依赖线性表示假设与模型架构的可压缩性、在极复杂或非分类任务上效果可能下降，且零样本模式下保留数据结构可能略有扭曲。

---

## 347. On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression

**arXiv ID:** 2601.21531 | [PDF](https://arxiv.org/pdf/2601.21531v1)

**作者:** Xinwei Zhang `[一作]` (Hong Kong Polytechnic University), Haibo Hu `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 8657 | [OpenAlex ID](https://openalex.org/A5020630816)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了在视觉令牌压缩（token compression）条件下大型视觉-语言模型（LVLM）的对抗鲁棒性，并提出了一种压缩感知攻击方法Compression‑AliGnEd（C‑AG），通过在未知压缩预算下对扰动进行优化来暴露压缩模型的脆弱点。

**💡 创新点**

创新点包括：
1) 揭示现有基于编码器的攻击存在“优化‑推理不匹配”问题，导致对压缩模型鲁棒性评估过于乐观；
2) 设计了两项新目标——期望特征破坏（Expected Feature Disruption, EFD）与秩-畸变对齐（Rank Distortion Alignment, RDA），实现对压缩瓶颈的无模型、无预算感知对抗优化；
3) 在五种主流压缩机制、多个 token‑budget 级别及三大 VQA 数据集上系统评估，并首次探索了对应的防御策略。

**🔧 技术方法**

技术手段包括：
- 基于编码器的对抗梯度下降（PGD）与 ℓ∞ 约束；
- 以视觉编码器的注意力权重作为 token 重要性分数；
- 对未知压缩预算使用均匀先验，计算 token 生存概率并加权 EFD；
- 通过 softmax 将畸变与排名映射为概率分布，最大化二者的交叉熵实现 RDA；
- 对抗目标为最大化 EFD + λ·RDA；
- 防御方案包括鲁棒性感知选择（D1）、随机候选池（D2）和基于 Top‑K 注意力分布的检测。

**📊 数据集**

实验数据集：VQA‑v2、TextVQA、GQA，各抽取 1000 张图像–问题对进行评测。

**📈 对比分析**

与基准攻击 VEAttack 进行对比，使用相同的灰盒威胁模型。评估指标为干净准确率（Clean）与鲁棒准确率（Robust）。实验表明，Compression‑AliGnEd 在所有压缩机制、不同 token‑budget（576、192、128、64、32、16 以及 blind）下的鲁棒准确率均低于 VEAttack，平均降低幅度在 8–15% 之间，且在 TextVQA 等文本依赖任务上提升更为显著。

**⚠️ 局限性**

局限性：
- 防御方法仅在实验阶段表现出有限的提升，尚未达到实用级别；
- 攻击假设对编码器具有白盒访问，无法直接迁移到完全黑盒环境；
- 依赖注意力权重作为重要性分数，若压缩机制使用不同评分标准，效果可能受限；
- 评估仅覆盖 LLaVA 与 Qwen2.5‑VL 两种 LVLM，未验证在更广泛模型上的泛化；
- 对抗优化需要多次迭代，计算成本相对较高；
- 仅考虑视觉令牌压缩，未涉及语言或混合模态压缩的鲁棒性问题。

---

## 348. Fast and Geometrically Grounded Lorentz Neural Networks

**arXiv ID:** 2601.21529 | [PDF](https://arxiv.org/pdf/2601.21529v1)

**作者:** Robert van der Klis `[一作]` (ETH Zurich), Pascal Mettes `[通讯]` (University of Amsterdam)

**通讯引用:** 1447 | [OpenAlex ID](https://openalex.org/A5000845063)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的Lorentz全连接层，解决了传统Lorentz层在梯度下降过程中超平面距离增长缓慢（对数尺度）的问题。

**💡 创新点**

创新点包括：①基于“距离-超平面”解析的几何一致线性层；②在层内部使用Lorentzian激活函数以保持几何一致性；③利用参数缓存策略显著加速推理；④结合权重归一化和仅中心化批归一化稳定训练。

**🔧 技术方法**

技术手段主要有：Lorentz模型下的欧氏到双曲距离映射、并行运输与指数/对数映射、距离-超平面公式、Lorentzian激活函数、缓存V矩阵、WeightNorm+BatchNorm。

**📊 数据集**

实验数据集：CIFAR-10 与 CIFAR-100，用于图像分类与网络速度评估。

**📈 对比分析**

与标准欧氏ResNet-18、Poincaré ResNet-20、HCNN（Lorentz）比较：在CIFAR-10/100上精度基本相同；训练时间约为HCNN的1/3、Poincaré的1/8；推理速度比HCNN快约3倍、比Poincaré快约6倍，显示出显著的计算效率提升。

**⚠️ 局限性**

局限性：仍采用固定曲率；在非常高维（>4096）时缓存优势减弱；需要进一步研究原生Lorentz归一化和自适应曲率以提升更大规模网络的表现。

---

## 349. Signal-Adaptive Trust Regions for Gradient-Free Optimization of Recurrent Spiking Neural Networks

**arXiv ID:** 2601.21572 | [PDF](https://arxiv.org/pdf/2601.21572v1)

**作者:** Jinhao Li `[一作]` (Sapient Intelligence), Sen Song `[通讯]` (Tsinghua University)

**通讯引用:** 12922 | [OpenAlex ID](https://openalex.org/A5013759262)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于信号自适应信任域（Signal‑Adaptive Trust Regions, SATR）的无梯度优化方法，用于训练能耗低、实时性能高的递归尖峰神经网络（RSNN）控制策略。

**💡 创新点**

创新点在于将KL信任域大小与种群梯度的能量（信号强度）耦合，使得在信号强时扩展信任域、在噪声主导时收缩，从而在有限种群预算下显著提升更新稳定性；并将该思想与伯努利连接分布自然梯度结合，得到闭式更新；此外，还提出了利用位集实现二值尖峰与二值权重的高效推理。

**🔧 技术方法**

核心技术包括：无梯度种群优化（Evolutionary Strategies/Evolving Connectivity）、自然梯度与KL信任域约束、信号能量自适应调整、伯努利分布的精确信息几何、位集（bitset）加速的二值计算。

**📊 数据集**

在Brax物理引擎提供的三种连续控制基准（Humanoid、Hopper、Walker2d）上进行实验。

**📈 对比分析**

与PPO‑LSTM、ES‑RSNN、原始EC‑RSNN及使用代理梯度的SG‑RSNN等基线比较，SATR在所有任务下都取得更高或相近的最终回报，且在种群规模缩小时表现更稳健；在匹配训练时长的条件下，SATR的奖励–运行时间曲线优于所有基线，显示出更佳的能耗与实时性能。

**⚠️ 局限性**

局限性包括：仍需对种群大小和学习率等超参数进行调优；方法在非常大规模任务或不同网络结构下的可推广性尚未充分验证；以及位集加速虽然显著提升速度，但对不同硬件平台的实现复杂度较高。

---

## 350. RecNet: Self-Evolving Preference Propagation for Agentic Recommender Systems

**arXiv ID:** 2601.21609 | [PDF](https://arxiv.org/pdf/2601.21609v1)

**作者:** Bingqian Li `[一作]` (Renmin University of China), Ji-Rong Wen `[通讯]` (Renmin University of China)

**通讯引用:** 23669 | [OpenAlex ID](https://openalex.org/A5025631695)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为RecNet的自我演化偏好传播框架，用于代理式推荐系统；

**💡 创新点**

创新点在于将网络路由代理引入偏好传播，实现社区级别的精准、可调的偏好更新，并通过文本化的多代理强化学习实现传播策略的持续自适应；

**🔧 技术方法**

主要技术包括基于LLM的属性提取与摘要、router‑agent中介式路由、消息缓冲与规则过滤的个性化接收机制，以及利用LLM生成的文本梯度进行反馈驱动的模块优化；

**📊 数据集**

实验使用了Amazon Review数据集的三个文本子集（CDs & Vinyl、Office Products、Musical Instruments）以及其抽样版；

**📈 对比分析**

与Pop、BPR、SASRec、LLMRank、AgentCF、AgentCF++、KGLA等基线相比，RecNet在NDCG@1/5/10上均实现了显著提升，尤其在小样本场景下表现与传统全量模型相当；

**⚠️ 局限性**

局限包括对LLM推理成本的依赖、对初始化路由器数量敏感以及在极端冷启动场景下仍需更丰富的外部知识注入。

---

## 351. Dynamics Reveals Structure: Challenging the Linear Propagation Assumption

**arXiv ID:** 2601.21601 | [PDF](https://arxiv.org/pdf/2601.21601v1)

**作者:** Hoyeon Chang `[一作]` (KAIST), Seong Joon Oh `[通讯]` (University of Tübingen)

**通讯引用:** 8298 | [OpenAlex ID](https://openalex.org/A5025851635)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对关系代数与梯度更新几何的严谨分析，探讨并证明了在第一阶线性更新（LPA）下，逻辑运算（特别是否定与共轭）需要张量分解结构，而组合（共轭）与否定在线性框架内不可兼容，导致在单步更新中无法实现系统化的多跳推理；

**💡 创新点**

提出了“系统化线性传播（SLP）”框架并给出其对特征空间的张量分解与对称性约束，揭示了线性更新在保持逻辑一致性时的根本几何限制，并给出了组合运算不可行的不可约性证明；

**🔧 技术方法**

使用关系代数、Tarski不变性判据、群表示理论、张量分解与双线性算子等数学工具，对梯度特征进行线性化并进行理论证明；

**📊 数据集**

在实验中使用Qwen3-4B、30B和Olmo3-7B模型的梯度，评估TREx子集的Negated LAMA样本来检验梯度对齐现象；

**📈 对比分析**

实验仅展示了梯度与否定/共轭的相似度对比，未给出整体性能指标，主要通过梯度对齐度来验证理论预测；

**⚠️ 局限性**

局限性在于只考虑第一阶线性更新（NTK假设），无法解释高阶/非线性更新的效果；组合逻辑与否定在此框架内不可兼容，导致多跳推理和知识编辑难以通过单步线性更新实现。

---

## 352. Beyond Imitation: Reinforcement Learning for Active Latent Planning

**arXiv ID:** 2601.21598 | [PDF](https://arxiv.org/pdf/2601.21598v1)

**作者:** Zhi Zheng `[一作]`, Wee Sun Lee `[通讯]` (National University of Singapore)

**通讯引用:** 5710 | [OpenAlex ID](https://openalex.org/A5071864357)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种主动隐层规划方法ATP‑Latent，用于在连续隐语义空间中高效生成链式思考（CoT），并通过强化学习（RL）优化隐推理策略，显著降低推理令牌数，提升数学推理准确率。

**💡 创新点**

创新点在于①将隐推理过程视为变分自编码器（VAE）训练，构建平滑、可解释的隐空间；②引入停止头（stop‑head）实现信息均匀分布；③在RL阶段加入由VAE解码的CoT一致性（coherence）奖励，提供软约束；④在训练中同时考虑答案正确性与推理连贯性，实现主动规划而非被动模仿。

**🔧 技术方法**

主要技术包括变分自编码器（VAE）用于生成隐表示，基于LLaMA‑1B的预训练语言模型作为编码器/解码器；强化学习框架采用GRPO或PPO等策略；停止头为多层感知机实现停/继续决策；token数与准确率的双重评价指标。

**📊 数据集**

使用公开数学推理基准：GSM‑8K、GAM‑Hard、MultiArith、SVAMP；训练集为GSM8K‑Aug（约38.5万条），测试集为GSM‑8K、GSM‑Hard、SVAMP、MultiArith。

**📈 对比分析**

与Coconut、SIM‑CoT、iCoT、CoLaR以及SFT对照，ATP‑Latent在四个基准上平均提高4.1%准确率，平均减少3.3%生成令牌；在MultiArith上实现94.4%准确率；在RL阶段通过Pass@K曲线展示显著提升的规划能力。

**⚠️ 局限性**

局限性包括：①模型仍依赖预训练语言模型的规模与表现；②对多样性或跨领域推理的适应性未作系统评估；③VAE训练对超参数（β、σ）敏感；④RL阶段对计算资源需求较高，且奖励设计可能对不同任务需要调整。

---

## 353. Unifying Heterogeneous Degradations: Uncertainty-Aware Diffusion Bridge Model for All-in-One Image Restoration

**arXiv ID:** 2601.21592 | [PDF](https://arxiv.org/pdf/2601.21592v1)

**作者:** Luwei Tu `[一作]` (Shenzhen Campus of Sun Yat-sen University), Zhi Jin `[通讯]` (Shenzhen Campus of Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 UDBM，一种基于松弛扩散桥的全场景图像恢复框架，能够在单步推理中同时处理多种不确定的降质。

**💡 创新点**

创新点包括：① 用松弛终端约束的扩散桥消除传统桥式中的漂移奇异；② 采用不确定性引导的双重调度（噪声调度对齐降质流，路径调度按熵正则化的可粘性动力学自适应调节传输轨迹），实现降质的统一与差异化恢复；③ 将扩散桥映射为边缘分布的线性组合，得到可解析的 DDIM 形式。

**🔧 技术方法**

主要技术：扩散桥（Doob h‑transform）、熵正则化最优传输（EOT）、线性边缘分布、DDIM/DPDM 推理、残差基不确定性估计。

**📊 数据集**

使用标准单任务数据集：Deraining（合并数据集）、Low‑Light（LOL）、Snow100K、Haze（RESIDE）、Deblurring（GoPro），并在 BSD68、CDD11、Real‑Rain/Real‑Dark/Real‑Snow/Real‑Blur 等真实与混合降质基准上评估。

**📈 对比分析**

与任务专用模型（SwinIR、MIRNet‑v2、Restormer 等）以及全场景模型（AirNet、Prompt‑IR、DA‑CLIP、DiffUIR、AdaIR、BioIR、MOCE‑IR、HOGformer）对比，UDBM 在 PSNR/SSIM、PERCEPTUAL 指标上均取得 SOTA，单步推理实现 6× 速度提升、FLOPs 约 2× 下降。

**⚠️ 局限性**

局限性：对像素级不确定性估计的精度依赖较高，若估计误差大易导致不恰当的路径/噪声调度；在极端未知的组合降质场景下仍可能出现残留噪声；且多任务统一模型在极度不平衡的数据分布下需要更精细的自适应机制。

---

## 354. Heterogeneity-Aware Knowledge Sharing for Graph Federated Learning

**arXiv ID:** 2601.21589 | [PDF](https://arxiv.org/pdf/2601.21589v1)

**作者:** Wentao Yu `[一作]` (Nanjing University of Science and Technology), Chen Gong `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 6243 | [OpenAlex ID](https://openalex.org/A5030222911)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种在图联邦学习中同时共享节点特征语义知识和结构特征知识的方法 FedSSA，以缓解客户端间的节点特征异质性和结构拓扑异质性。

**💡 创新点**

创新点在于：①对每个客户端的节点特征采用变分模型推断类别分布，利用KL散度实现语义知识对齐；②利用谱图神经网络与谱能量度量对每个客户端的结构特征进行聚类，并通过对齐学习实现结构知识共享；③分别处理语义和结构两类异质性，而非统一加权聚合。

**🔧 技术方法**

核心技术包括：变分图自动编码器 (VGAE)、谱图神经网络（ChebNet、BernNet 等）、谱能量度量、Grassmann流形上的切角距离聚类、KL 散度对齐与正则化项、理论上证明线性收敛。

**📊 数据集**

使用 11 个公开图数据集：6 个同类性 (Cora、CiteSeer、PubMed、Amazon-Computer、Amazon-Photo、ogbn-arxiv) 与 5 个异类性 (Roman-empire、Amazon-ratings、Minesweeper、Tolokers、Questions)。

**📈 对比分析**

与 11 种现有联邦学习/图联邦学习基线（FedAvg、FedProx、FedPer、GCFL、FedGNN、FedSage+、FED-PUB、FedGTA、AdaFGL、FedTAD、FedIIH）在不同客户端数量与分区方式下对比；FedSSA 在所有场景中均显著优于第二好方法，平均提升约 2.82%（分类准确率）。

**⚠️ 局限性**

局限性：①需要在每轮通信前完成两次聚类，聚类质量会影响性能；②方法对超参数（节点聚类数、结构聚类数、正则化系数）有一定依赖；③在极大规模图或极稀疏图上，谱能量计算和谱分解可能成为计算瓶颈；④目前仅针对节点分类任务，未直接验证在边预测、图分类等任务上的效果。

---

## 355. Evaluating Prediction Uncertainty Estimates from BatchEnsemble

**arXiv ID:** 2601.21581 | [PDF](https://arxiv.org/pdf/2601.21581v1)

**作者:** Morten Blørstad `[一作]` (University of Bergen), Pekka Parviainen `[通讯]` (University of Bergen)

**通讯引用:** 480 | [OpenAlex ID](https://openalex.org/A5078448870)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在表格数据和时间序列任务中，研究并评估了 BatchEnsemble 及其 GRU 变体 GRUBE 的不确定性估计能力。

**💡 创新点**

提出 GRUBE（将 BatchEnsemble 引入 GRU 结构）并证明 BatchEnsemble 在保持预测性能的同时，可用更少参数逼近深度集群的性能。

**🔧 技术方法**

采用 BatchEnsemble、Monte Carlo Dropout 与 Deep Ensemble 等方法，并在 GRU 中实现共享权重的 BatchEnsemble。

**📊 数据集**

实验使用公开表格数据集（California、Diabetes、Adult、Breast Cancer、Phoneme）及时间序列数据集（Electric、Temperature）。

**📈 对比分析**

与 Deep Ensemble、MC Dropout 比较，BatchEnsemble 在大多数指标（RMSE、NLL、ECE、Brier 等）上与 Deep Ensemble 相当甚至更优，同时显著减少参数量和推理时间。

**⚠️ 局限性**

在图像 CNN 任务中 BatchEnsemble 表现欠佳，且对不同数据类型的适用性差异仍需要进一步研究，分布偏移的细粒度分析仍有限。

---

## 356. KromHC: Manifold-Constrained Hyper-Connections with Kronecker-Product Residual Matrices

**arXiv ID:** 2601.21579 | [PDF](https://arxiv.org/pdf/2601.21579v1)

**作者:** Wuyang Zhou `[一作]` (Imperial), Danilo Mandic `[通讯]` (Imperial)

**通讯引用:** 24396 | [OpenAlex ID](https://openalex.org/A5103001848)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的 KromHC 模型，在超连接（Hyper‑Connections）中通过 Kronecker 乘积构造双重随机残差矩阵，以实现参数高效且稳定的深度网络训练。

**💡 创新点**

创新点在于利用 Kronecker 乘积闭包性质，将高维双随机矩阵拆解为小尺寸双随机子矩阵的乘积，既保证了精确双随机性，又将参数复杂度从 O(n³C) 降到 O(n²C)。

**🔧 技术方法**

技术手段包括张量化残差流、Tucker 结构张量网络、Birkhoff–von‑Neumann 定理、Kronecker 乘积以及软最大化学习系数。

**📊 数据集**

实验使用 Nanochat 预训练模型，并在类似 C4 的大规模文本数据集上进行训练和验证。

**📈 对比分析**

与标准残差连接、mHC 以及 mHC‑lite 的对比表明，KromHC 在训练损失、验证 BPB、CORE 分数以及各类常识推理任务中均达到或超过 SOTA，且所需额外参数显著更少。

**⚠️ 局限性**

局限性是当残差流宽度 n 为大素数时，Kronecker 分解可能导致参数配置不利，可通过选择 n 为 2 或 3 的幂或具有小质因数的数来缓解。

---

## 357. Bridging Functional and Representational Similarity via Usable Information

**arXiv ID:** 2601.21568 | [PDF](https://arxiv.org/pdf/2601.21568v1)

**作者:** Antonio Almudévar `[一作]` (University of Zaragoza), Alfonso Ortega `[通讯]` (University of Zaragoza)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个统一框架，使用可用信息理论来量化功能相似性和表征相似性，并阐明两者之间的层次关系；

**💡 创新点**

将模型拼接损失与可用条件互信息建立理论联系；证明表征相似性是功能相似性的充分但非必要条件；将经典几何相似度（CKA、RSA、SVCCA）解释为不同预测族下的可用信息估计器；

**🔧 技术方法**

可用信息理论、模型拼接（stitching）、线性/正交/仿射变换、均方误差、均方误差比例、CKA、RSA、SVCCA、实验评估；

**📊 数据集**

MNIST、CIFAR‑10、CIFAR‑100、SVHN、Tiny‑ImageNet，使用线性模型、CNN、ResNet、DenseNet、ShuffleNet、MobileNet 等多种编码器；

**📈 对比分析**

通过在不同预测族（仿射、正交+缩放、正交）下训练拼接器，比较拼接性能的对称性、表征相似性层次以及与经典指标的相关性。实验显示拼接往往不对称，表征相似性随预测族表达力提升而提升，CKA/RSA 与重建误差高度相关；

**⚠️ 局限性**

仅在特定预测族下定义，可用信息与实际模型复杂度仍有差距；实验仅在二维任务与离散标签上验证，未探讨连续或更复杂任务；拼接器学习受优化噪声影响，导致近似解不稳定；

---

## 358. Opinion Consensus Formation Among Networked Large Language Models

**arXiv ID:** 2601.21540 | [PDF](https://arxiv.org/pdf/2601.21540v1)

**作者:** Iris Yazici `[一作]` (Bilkent University), Ali H. Sayed `[通讯]` (EPFL)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大语言模型（LLM）代理之间通过 DeGroot 框架进行多轮文本交互实验，研究其群体意见演化与一致性。

**💡 创新点**

首次将 DeGroot 一致性模型应用于 LLM 网络，发现最终共识不随初始意见而定，而受讨论主题和预训练偏见驱动，并验证收敛速率仍符合谱理论。

**🔧 技术方法**

使用 AutoGen 构建多代理框架，Gemini 生成对话，OpenAI 进行情感分析，基于 Erdős–Rényi、全连和环图的组合矩阵。

**📊 数据集**

公开数据集 asl-epfl/Social-LLM-Networks，包含 764 个实验、8 个议题、1.2 百万条 LLM 回复。

**📈 对比分析**

对比加权与无权实验、与 DeGroot 预测的 RMSE（0.32）和分类准确率（≈32%），以及经验减半时间与理论 λ₂ 匹配，表明收敛速率符合谱理论。

**⚠️ 局限性**

仅测试 Gemini 单一模型、主题与提示受限、未考虑战略代理或信息操纵，且情感分析误差可能影响结果。

---

## 359. Beyond Parameter Finetuning: Test-Time Representation Refinement for Node Classification

**arXiv ID:** 2601.21615 | [PDF](https://arxiv.org/pdf/2601.21615v1)

**作者:** Jiaxin Zhang `[一作]` (National University of Defense Technology), En Zhu `[通讯]` (National University of Defense Technology)

**通讯引用:** 9220 | [OpenAlex ID](https://openalex.org/A5069681054)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于测试时表示微调（TTReFT）的图神经网络自适应方法，避免参数微调导致的灾难性遗忘。

**💡 创新点**

创新点包括：将自适应目标从参数迁移到低秩表示空间，使用不确定性引导的节点选择，以及针对干预密度动态调节的干预感知遮盖自编码器。

**🔧 技术方法**

采用图自编码器、低秩线性表示干预、熵最小化、遮盖自编码器（IAMAE）以及多层 GCN/SAGE 等图神经网络框架。

**📊 数据集**

在 Cora、Pubmed、Citeseer、Wikics、Arxiv 五个节点分类基准上进行评测。

**📈 对比分析**

与现有域泛化与测试时训练方法（EERM、TAR、Tent、GTrans、HomoTTT）比较，TTReFT 在所有数据集上均实现更高的 OOD 准确率，同时几乎不影响原始任务性能，显著优于参数微调方法。

**⚠️ 局限性**

局限性包括：仍需手动设定干预层数、低秩维度和遮盖率等超参数，对非正交或拓扑类分布偏移的理论保障有限。

---

## 360. Search-Based Risk Feature Discovery in Document Structure Spaces under a Constrained Budget

**arXiv ID:** 2601.21608 | [PDF](https://arxiv.org/pdf/2601.21608v1)

**作者:** Saisubramaniam Gopalakrishnan `[一作]` (Philabs), Dagnachew Birru `[通讯]` (Philabs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

将智能文档处理（IDP）系统的验证转化为在合成文档配置空间内的预算受限搜索基础软件测试（SBST）问题，利用多种搜索策略系统性发现多样化的失败模式。

**💡 创新点**

①提出风险特征发现框架并设计配置级排他性与跨时间重叠分析；②展示不同搜索策略的互补性并构建多策略组合；③引入量子近似优化算法（QAOA）中的相关混合器提升多样性；④使用一系列多维度评价指标（AUC、独特布局、汉明多样性、熵）量化发现效果。

**🔧 技术方法**

搜索基础软件测试技术：遗传算法（GA‑Explore/GA‑Exploit）、粒子群优化（PSO）、MAP‑Elites、贝叶斯优化（GP‑EI、GP‑UCB、TPE）、强化学习（REINFORCE、PPO‑Risk、PPO‑Div）、随机搜索、模拟退火、量子近似优化算法（QAOA、QAOA‑Corr）；合成文档生成器、IDP系统Oracle、随机森林预测模型。

**📊 数据集**

基于8种常见金融/保险文档模板构建的合成文档配置空间（单页24维、双/多页27维），使用 Faker 生成文本并通过多模态 IDP Oracle 评估 OCR、布局、KV、表格等任务；未使用公开真实数据集。

**📈 对比分析**

在固定预算 1000 次 Oracle 调用下，对 14 种搜索策略进行统一对比，评估最大风险、Top‑10% 风险、AUC、独特布局、汉明多样性与熵等指标；结果显示贝叶斯优化与 PPO‑Risk 在最大风险上最高，MAP‑Elites 与随机搜索在多样性上最佳，QAOA‑Corr 在风险与多样性之间取得平衡；不同策略互补性显著，组合训练的随机森林预测模型 R² 超过 0.83。

**⚠️ 局限性**

局限性：依赖合成数据与单一 IDP 体系结构，缺少真实部署场景验证；量子方法受限于模拟器深度与噪声；在更高维配置空间中仍存在探索瓶颈；未覆盖高阶语义错误和 LLM 判别；仅在单一预算规模评估，未探索动态预算或持续集成场景。

---

## 361. Frequency as Aperture: Enabling Embeddable Near-Field Sensing for 6G Wireless Radios

**arXiv ID:** 2601.21584 | [PDF](https://arxiv.org/pdf/2601.21584v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 362. CORE: Collaborative Reasoning via Cross Teaching

**arXiv ID:** 2601.21600 | [PDF](https://arxiv.org/pdf/2601.21600v1)

**作者:** Kshitij Mishra `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Salem Lahlou `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了训练时跨教师协作框架CoRe，通过让模型在失败时接收同伴成功的提示，提升多模型推理的协同效果。

**💡 创新点**

创新点在于将同伴成功作为 on‑policy 学习信号，构建两轮微回合（冷采样+上下文恢复）和融合探索、利用、救援奖励的多目标策略优化。

**🔧 技术方法**

技术包括：自回归 LLM 策略梯度训练、DPP‑lite 轻量化多样性奖励、对齐探索-利用权重、救援奖励、跨模型互补性奖励以及可插拔的强化学习优化器。

**📊 数据集**

使用四大推理基准：GSM8K、MATH、AIME 和 GPQA，并在低数据（≤1000样本）下训练。

**📈 对比分析**

与基线（单模型、SD‑E^2、无协作的多模型或 Oracle）相比，CoRe 在 Pass@1/Pass@2 以及 Team Pass@K 上均取得显著提升，例如 GSM8K Team Pass@2 达到 99.54%（单模型仅 56%），MATH Team Pass@2 92.08% 等。

**⚠️ 局限性**

局限性包括：依赖同伴成功的提示截断可能导致信息丢失、跨模型误导风险、对高度专业化问题的推理仍有限、以及在 Oracle 评估中可能高估实际部署效果。

---

## 363. Scalable Power Sampling: Unlocking Efficient, Training-Free Reasoning for LLMs via Distribution Sharpening

**arXiv ID:** 2601.21590 | [PDF](https://arxiv.org/pdf/2601.21590v1)

**作者:** Xiaotong Ji `[一作]` (Huawei Noah's Ark Lab), Haitham Bou Ammar `[通讯]` (Huawei Noah's Ark Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无训练、无验证器的可扩展功率采样方法，通过对低温采样进行标量缩放并利用蒙特卡罗前瞻与 Jackknife 校正实现对全局功率分布的近似，从而实现高效推理；

**💡 创新点**

核心创新在于将全局功率分布解析为每个 token 的局部低温分布乘以未来期望缩放因子，并给出闭式表达式；通过前瞻采样与 Jackknife 纠偏，实现无需 MCMC 的自回归采样；

**🔧 技术方法**

技术包括：理论推导功率-低温关系、基于 LLM 的蒙特卡罗前瞻估计、Jackknife 偏差校正、Top‑K 筛选与批量并行推理；

**📊 数据集**

使用 MATH500（数学）、HumanEval（代码）和 GPQA（知识问答）三个标准基准数据集进行评测；

**📈 对比分析**

与基线（标准解码、低温采样、Best‑of‑N、MCMC 采样）以及 RL‑后训练模型 GRPO 进行对比，结果显示在三类任务中 Pass@1 与 GRPO 相近或更优（最高可提升 13.4%），同时推理延迟比 MCMC 低 10 倍；

**⚠️ 局限性**

局限性在于仍需多条前瞻轨迹导致额外推理成本，对已 RL 细调模型提升有限；方法对基础模型的安全与偏差依赖较大，若基础模型含有不良信息，采样时可能放大这些行为。

---

## 364. ICL-EVADER: Zero-Query Black-Box Evasion Attacks on In-Context Learning and Their Defenses

**arXiv ID:** 2601.21586 | [PDF](https://arxiv.org/pdf/2601.21586v1)

**作者:** Ningyuan He `[一作]` (University of Science and Technology of China), Shanqing Guo `[通讯]` (Shandong University)

**通讯引用:** 1379 | [OpenAlex ID](https://openalex.org/A5084460856)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d`

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

## 365. Depth-Recurrent Attention Mixtures: Giving Latent Reasoning the Attention it Deserves

**arXiv ID:** 2601.21582 | [PDF](https://arxiv.org/pdf/2601.21582v1)

**作者:** Jonas Knupp `[一作]` (Aleph Alpha Research), Kristian Kersting `[通讯]` (Hessian.AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种深度递归注意力混合框架（Dreamer），通过在同一层中结合序列注意、深度注意和稀疏专家注意，实现多步潜在推理。

**💡 创新点**

创新点在于：① 将深度注意引入深度递归模型，缓解隐藏尺寸瓶颈；② 将稀疏专家注意整合为深度维度的注意，解决层级尺寸瓶颈；③ 将序列、深度、专家三维注意统一为可插拔的“注意力混合”，提供模块化的知识访问视角。

**🔧 技术方法**

核心技术包括：深度递归 Transformer、RoPE 深度位置编码、稀疏专家（MoE）与自适应路由、RMSNorm、FLOP 与参数匹配的双向坐标下降优化。

**📊 数据集**

使用约 100 B 公开指令数据（混合多种开源数据集），去除 CoT 轨迹，重点在数学推理和自然语言推理基准（2cGSM8K、2cMATH、2cMMLU、2cMathQA 等）。

**📈 对比分析**

与同深度、同 FLOP/参数/内存匹配的经典分层 MoE 基线对比，实验显示：DR+DA 在 16 层时 约 2 倍数据效率、在 32 层时 约 2 倍参数/计算效率，且在所有数学基准上显著优于基线。

**⚠️ 局限性**

局限性包括：深度注意仍带来一定的内存搬运开销；目前模型只支持固定深度，动态深度推广仍需研究；对 RoPE 位置编码的依赖可能限制深度泛化；在更大规模或不同任务上的可扩展性与鲁棒性尚未完全验证。

---

## 366. Chain Of Thought Compression: A Theoritical Analysis

**arXiv ID:** 2601.21576 | [PDF](https://arxiv.org/pdf/2601.21576v1)

**作者:** Juncai Li `[一作]` (Shanxi University), Jeff Z. Pan `[通讯]` (University of Edinburgh)

**通讯引用:** 7569 | [OpenAlex ID](https://openalex.org/A5066422711)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了Chain-of-Thought（CoT）压缩的理论难点，提出了高阶交互导致学习信号指数衰减的问题，并通过理论与实验验证了隐式CoT的局限；

**💡 创新点**

创新点包括：1）提出Order‑r交互理论并证明隐式压缩会导致高阶交互信号指数衰减；2）构造了逻辑不可约的 NatBool‑DAG 基准；3）提出对齐隐式CoT（ALiCoT）框架，利用分布对齐有效避免高阶交互爆炸；

**🔧 技术方法**

使用Transformer模型、梯度信号分解、交互序列理论、对齐损失（对数或余弦相似度）以及多任务实验（Parity 问题和自然语言推理），并结合 Qwen3 等 LLM 进行实验；

**📊 数据集**

主要数据集为自定义的 NatBool‑DAG（包含 3–10 步推理的逻辑 DAG），以及公开的数学推理数据集（GSM8k）和常识推理数据集（StrategyQA）用于验证；

**📈 对比分析**

与传统显式CoT、两类隐式CoT基线（Imp.Base‑1、Imp.Base‑2）对比；实验显示 ALiCoT 在保持 54.4× 速度提升的同时，准确率保持在 83.9%（0.6B）或 95.0%（4B），显著优于其它隐式基线，几乎等效于完整CoT；

**⚠️ 局限性**

局限性：对齐目标依赖于显式中间步骤的可获取性，难以扩展到开放式自然语言推理中；对模型规模的敏感性未完全探究；在极深层逻辑推理（超过 10 步）时，仍存在潜在的指数增长风险。

---

## 367. EmboCoach-Bench: Benchmarking AI Agents on Developing Embodied Robots

**arXiv ID:** 2601.21570 | [PDF](https://arxiv.org/pdf/2601.21570v1)

**作者:** Zixing Lei `[一作]` (Shanghai Jiao Tong University), Siheng Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 8514 | [OpenAlex ID](https://openalex.org/A5066373402)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了EmboCoach-Bench benchmark，用LLM代理自动化机器人开发生命周期，从代码编写、调优到环境交互；

**💡 创新点**

创新点在于将LLM与环境闭环结合，采用“Draft‑Debug‑Improve”迭代框架，让代理在真实仿真中不断改进策略；

**🔧 技术方法**

技术包括LLM代理（如Claude、Gemini、DeepSeek等）、OpenHands文件编辑、ML‑Master MCTS搜索、Kubernetes分布式训练、仿真反馈回路；

**📊 数据集**

数据集为32个由ManiSkill、RoboTwin、Robomimic、MetaWorld等平台构建的RL/IL任务，涵盖从抓取到精细装配等多种物理交互；

**📈 对比分析**

方法通过对比“无代理”单次生成和“代理迭代”两种设置，实验显示代理平均提升成功率约30‑40%，在多任务上超越人类基准，甚至在失败任务中实现“复活”；

**⚠️ 局限性**

局限在于对大型GPU集群依赖较高、任务难度和复杂度有限，且LLM在某些高风险探索时仍可能产生未验证的错误。

---

## 368. SAL: Selective Adaptive Learning for Backpropagation-Free Training with Sparsification

**arXiv ID:** 2601.21561 | [PDF](https://arxiv.org/pdf/2601.21561v1)

**作者:** Fanping Liu `[一作]` (Renmin University of China), Jiasi Zou `[通讯]` (ROCK AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为 Selective Adaptive Learning（SAL）的训练方法，通过样本级别的可学习路由和自适应区域划分，实现参数空间的显式解耦，从而降低梯度干扰并消除权重对称性需求。

**💡 创新点**

创新点包括：①Learned‑Frozen Decoupled Routing机制，实现可学习但固定的区域划分；②区域条件参数更新，仅更新被激活区域的权重；③非对称误差传播与局部对齐信号结合，绕过 BP 的权重传输问题；④利用辅助任务损失独立更新路由器，使路由决策与主网络梯度分离。

**🔧 技术方法**

使用的技术主要有：可学习特征投影、固定原型匹配、硬路由选择、区域条件前向计算、非对称反馈矩阵、局部对齐信号、残差连接、SGD 优化器以及跨区域的辅助监督损失。

**📊 数据集**

实验数据集共十个，分别为：CIFAR‑10、PCam、STL‑10、SVHN、MNIST、Fashion‑MNIST、Digits、USPS、Semeion、FER2013。

**📈 对比分析**

通过与使用相同网络结构的标准 BP 基线进行对比，评估指标为最终准确率。实验显示：SAL 在 9/10 个数据集上均优于基线，随着区域数量、网络深度和宽度的增加，SAL 的性能提升更为显著；在 128 层深度和 1B 参数规模下仍保持稳定收敛。

**⚠️ 局限性**

主要限制包括：①区域参数线性扩展导致显存消耗上升；②硬路由对初始化敏感，可能导致收敛到局部最优；③目前仅验证于前馈/残差网络，尚未探究其在 Transformer 等更复杂架构中的可扩展性。

---

## 369. Meta Context Engineering via Agentic Skill Evolution

**arXiv ID:** 2601.21557 | [PDF](https://arxiv.org/pdf/2601.21557v1)

**作者:** Haoran Ye `[一作]` (Peking University), Guojie Song `[通讯]` (Peking University)

**通讯引用:** 5978 | [OpenAlex ID](https://openalex.org/A5088976879)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出Meta Context Engineering（MCE），通过双层进化框架让LLM自适应地学习上下文工程技能与上下文文件，提升推理时的上下文质量与适配性。

**💡 创新点**

创新点在于：①将上下文工程拆解为可进化的“技能”抽象，并通过Agentic Crossover进行演化；②采用基于文件和代码的上下文表示，去除传统固定结构偏见；③实现全Agentic的双层优化（元层技能进化 + 基层上下文构建），使上下文可自我改进。

**🔧 技术方法**

技术手段包括：(1) 1+1进化策略与Agentic Crossover；(2) LLM驱动的代码生成与文件系统操作；(3) 基于DSPy/Claude Agent SDK的Agentic执行框架；(4) 多任务验证与交叉熵/奖励评估。

**📊 数据集**

使用五个跨领域基准：FiNER（金融）、USPTO‑50k（化学）、Symptom2Disease（医学）、LawBench（法律）和AEGIS2（AI安全），并在多种LLM（DeepSeek‑V3.1、Llama‑3.3‑70B、Qwen3‑8B、Gemma‑3‑4B）上测试。

**📈 对比分析**

与基线（Base、ICL、MIPROv2、GEPA、Dynamic Cheatsheet、ACE）比较，MCE在离线场景平均提升89.1%、在线场景提升74.1%，相对SOTA提升约16.9%（最大53.8%）。此外，MCE在上下文长度、效率、迁移性能和训练速度（约13.6×加速、4.8×更少rollout）方面均优于传统方法。

**⚠️ 局限性**

局限性：对需要细粒度推理或长序列轨迹的任务不如现有手工设计的Agentic harness有效；当前性能受限于底层Agentic模型的能力，未来需更强的模型才能充分发挥。

---

## 370. Chasing Elusive Memory Bugs in GPU Programs

**arXiv ID:** 2601.21552 | [PDF](https://arxiv.org/pdf/2601.21552v1)

**作者:** Anubhab Ghosh `[一作]` (Indian Institute of Science), Arkaprava Basu `[通讯]` (Indian Institute of Science)

**通讯引用:** 6166 | [OpenAlex ID](https://openalex.org/A5111673340)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文提出一种编译时静态分析工具，用于检测CUDA程序中的输入相关越界访问和内部分区越界错误。

**💡 创新点**

创新点在于首次结合语义关系与SAT求解器，既能捕获仅在特定输入下才会触发的越界 bug，又能检测因逻辑分区导致的内部越界，且无运行时开销。

**🔧 技术方法**

技术上利用MLIR中间表示构建表达式树来推断分配大小与偏移量之间的语义约束，并将这些约束交给Google OR‑Tools SAT 求解器求解。

**📊 数据集**

评估使用了 Kaldi、llm.c、samples、ScoR、Rodinia、HeCBench、Indigo 等多个开源 GPU 代码库，共 20 个工作负载。

**📈 对比分析**

与 NVIDIA 的 cuda‑memcheck（仅运行时检测）相比，本工具在 20 个程序中发现 45 个越界 bug，且无误报；编译时间在 8 ms–32 s 之间，且无运行时性能或内存占用。

**⚠️ 局限性**

局限性包括仅支持 CUDA、依赖 CGeist 生成 MLIR、只能检测越界（以及 UAF）且对极其动态的内存使用场景可能不够精准。

---

## 371. Training slow silicon neurons to control extremely fast robots with spiking reinforcement learning

**arXiv ID:** 2601.21548 | [PDF](https://arxiv.org/pdf/2601.21548v1)

**作者:** Irene Ambrosini `[一作]` (Institute of Neuroinformatics, UZH and ETH Zurich), Chiara Bartolozzi `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过将具有可在线学习功能的硅脉冲神经网络（在 DYNAP‑SE 处理器上实现）嵌入闭环控制框架，训练机器人在高速空气曲棍球环境中实时做出决策并完成拦截。

**💡 创新点**

创新点在于：① 将基于 e‑prop 的局部学习规则迁移至硬件实现的混合信号神经网络，实现毫秒级低功耗实时 RL；② 采用随机固定的突触网络（reservoir）捕捉任务时序结构；③ 在高维连续状态（6D）和快速控制（50 Hz）下实现样本高效学习，仅需 2000 试次即可达 96–98% 的成功率。

**🔧 技术方法**

技术包括：混合信号脉冲神经网络、DYNAP‑SE 硅芯片、FPGA 生成器的连续状态编码、e‑prop 近似 BPTT 的在线学习、基于奖励塑形的离散动作（运动原语）与离线/在线 RL 并行训练。

**📊 数据集**

数据集为 MuJoCo 模拟空气曲棍球环境，包含 1.038 m × 1.948 m 的工作空间、六维连续观测（小球位置、速度及手臂位置）以及两种离散动作（运动原语）。

**📈 对比分析**

与传统基于 CPU 的深度 RL（如 DQN、TD3）或以往的 2D 游戏 RL（Pong）相比，该方法在相同任务下只需 10 倍更少的神经元（1020 vs 10 000）即可获得相近或更好的成功率，且实现了硬件级实时闭环控制。

**⚠️ 局限性**

局限性包括：固定大小的硅网络对更宽速度/位置范围的输入会导致收敛速度下降且成功率下降；当前仅在模拟环境中验证，未处理真实传感噪声和机械误差；缺乏直接的事件摄像头输入，仍需依赖 CPU 生成脉冲；扩展到更大工作空间仍需多核/更大隐藏层。

---

## 372. ShardMemo: Masked MoE Routing for Sharded Agentic LLM Memory

**arXiv ID:** 2601.21545 | [PDF](https://arxiv.org/pdf/2601.21545v1)

**作者:** Yang Zhao `[一作]`, Dusit Niyato `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了ShardMemo tiered memory 服务，为 agentic LLM 提供按预算、按范围可检索的工作内存（Tier A）、分片证据（Tier B）和版本化技能库（Tier C），并通过轻量级门控在三层之间进行选择。

**💡 创新点**

创新点：
• 先做范围过滤（scope‑before‑routing）再做语义路由，保证所有检索到的 shard 均满足硬性约束；
• 将 shard 视为专家，用掩码 MMoE 路由器按预算选取最多 B_probe 个 shard，并引入成本感知门控和自适应 Top‑P 以平衡准确率和检索开销；
• 通过 evidence→shard 监督实现可训练的路由器，使用多标签集合似然目标；
• 通过 tier gate 在不同层之间按需求切换，并在技能失效时安全回退到证据检索。

**🔧 技术方法**

技术：掩码混合专家路由、局部 ANN 索引、成本感知加权、Adaptive Top‑P 采样、基于集合似然的监督路由训练、轻量级门控分类器、版本化技能库与回退机制。

**📊 数据集**

数据集：LoCoMo（长时序对话记忆评估）、HotpotQA（长上下文多跳问答）、ToolBench（工具使用与技能重用评估）。

**📈 对比分析**

对比方法：Vanilla LLM、RAG、A‑Mem、LightMem、Mem0、MemoryOS、GAM 以及 similarity/recency/centralized 路由基线。ShardMemo 在 LoCoMo 上比最佳基线 GAM 提升 5–7 F1（单跳 5.7、多跳 5.1、时序 6.8、开放域 6.0），在 HotpotQA 上比 GAM 提升 1–1.3 F1，ToolBench 上 Precision@R 提升 10%（0.97 vs. 0.88），同时显著降低 VecScan 与尾延迟。

**⚠️ 局限性**

局限：
• 路由器训练依赖 evidence→shard 的人工标注；
• shard 映射固定，未评估动态分片拆分/合并对性能影响；
• 仅在三种基准上验证，尚未检验极大规模或跨域适用性；
• 复杂度与实现开销相对传统单层检索系统仍较高。

---

## 373. inversedMixup: Data Augmentation via Inverting Mixed Embeddings

**arXiv ID:** 2601.21543 | [PDF](https://arxiv.org/pdf/2601.21543v1)

**作者:** Fanshuang Kong `[一作]` (Beihang University), Chunming Hu `[通讯]` (Beihang University)

**通讯引用:** 1856 | [OpenAlex ID](https://openalex.org/A5086470621)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 inversedMixup 的文本数据增强框架，结合 Mixup 的可控插值与 LLM 反演技术，能够从混合嵌入生成可读且语义可控的增广样本，并通过三阶段训练实现任务模型与大语言模型的对齐，提升下游任务性能。

**💡 创新点**

①通过三阶段训练（对齐、细化、反演）将任务特定模型输出嵌入空间与 LLM 输入空间对齐，首次实现对混合嵌入的可读性反演；②首次在文本 Mixup 中观察到“manifold intrusion”现象，并提出利用 LLM 分配硬标签的方式缓解；③将可解释的文本增广与可控的插值相结合，弥补了传统 Mixup 的不可解释性与 LLM 生成的不可控性。

**🔧 技术方法**

Mixup 线性插值、LLM 反演（soft token 与 prompt）、轻量级 MLP 适配器、软/硬标签策略、Beta 分布控制插值比例、t‑SNE 可视化、统计显著性检验、ChatGPT‑4o 作为评判者。

**📊 数据集**

使用公开文本数据集：Yahoo、TREC、AG News、Amazon、DBpedia、Yelp；其中 Yahoo 用于对齐阶段，TREC、AG News 用于未见域评估。

**📈 对比分析**

与传统增广方法（EDA、BT、Textsmooth、AWD、Mixup）和 LLM 生成增广方法（LLM‑Rew、LLM‑Gen、LLM‑Mix）在少样本（K=1,5,10）与全监督场景下对比；在准确率上，inversedMixup 在绝大多数设置下均优于基线，尤其在低资源场景下提升显著，且统计检验显示多数指标显著性。

**⚠️ 局限性**

对 LLM 生成质量和对齐精度高度依赖，若 LLM 生成不流畅或语义不稳定，增广效果下降；插值比例选择仍影响“manifold intrusion”率，需经验调参；适配器在不同任务迁移时的泛化性尚待验证，实验主要集中在文本分类任务，缺乏对多任务、跨语言等更广泛情形的评估。

---

## 374. Bi-Anchor Interpolation Solver for Accelerating Generative Modeling

**arXiv ID:** 2601.21542 | [PDF](https://arxiv.org/pdf/2601.21542v1)

**作者:** Hongxu Chen `[一作]` (Hong Kong University of Science and Technology), Long Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 94933 | [OpenAlex ID](https://openalex.org/A5100333572)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Bi-Anchor Interpolation Solver（BA-solver），通过双锚点插值加速流匹配模型的采样。

**💡 创新点**

创新点在于利用轻量 SideNet 实现双向时间感知，并以两端速度为锚点实现高阶插值，从而在仅 5–10 次 NFE 下保持高质量。

**🔧 技术方法**

采用流匹配 ODE、SideNet 侧网络、Gauss‑Lobatto 高阶积分、链式训练及双锚点策略。

**📊 数据集**

在 ImageNet‑256² 与 ImageNet‑512² 数据集上进行评估，并在 ImageNet‑256² 进行图像编辑实验。

**📈 对比分析**

与 Euler、Heun、Flow‑DPM、Flow‑UniPC 等无训练解算器以及一阶/少步训练方法对比，BA-solver 在 5–10 NFEs 下 FID 约 1.9，远优于同类方法，且训练迭代仅 250 次。

**⚠️ 局限性**

仍需训练 SideNet，适用于固定 backbone，极低 NFEs 或更高分辨率下的鲁棒性与可扩展性需进一步验证。

---

## 375. WMVLM: Evaluating Diffusion Model Image Watermarking via Vision-Language Models

**arXiv ID:** 2601.21610 | [PDF](https://arxiv.org/pdf/2601.21610v1)

**作者:** Zijin Yang `[一作]` (University of Science and Technology of China), Nenghai Yu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 22945 | [OpenAlex ID](https://openalex.org/A5064573190)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 WMVLM 框架，用于评价扩散模型生成图像的残差水印与语义水印的质量与安全性，并提供可解释的文本输出。

**💡 创新点**

① 统一评价残差与语义水印的框架；② 通过 PSNR+OSN 鲁棒性评估残差水印，使用潜在分布显著性检验(p值)评估语义水印的质量与安全；③ 三阶段训练（分类评分预训练→解释性冷启动→GRPO 泛化）。

**🔧 技术方法**

Vision‑Language 模型（Qwen3‑VL‑8B‑Instruct）+ 监督微调（SFT）+ 知识蒸馏 + Group Relative Policy Optimization (GRPO) 强化学习。

**📊 数据集**

基于 Stable Diffusion v2.1 生成的 5,000 张残差水印图像与 9,500 张语义水印图像，涵盖 6 种残差方法与 3 种语义方法；另外使用 MS‑COCO 验证集和 SD v1.4 进行交叉数据/模型评估。

**📈 对比分析**

与多款 SOTA VLM（GPT‑5、Claude‑Opus‑4.5、Gemini‑3‑Pro、LLaVA、Qwen3‑VL、Gemma）在零样本设置下对比；WMVLM 在残差水印质量上 PLCC/SRCC>0.9，安全性准确率≈99%；在语义水印上质量/安全≈0.85‑0.95；跨数据集、跨模型、跨方法均保持高准确率，显著优于基线。

**⚠️ 局限性**

依赖大量标注解释文本的 SFT，需耗时且难以扩展；对深度嵌入水印的模型仍敏感；目前仅针对图像水印，未扩展到其它模态。

---

## 376. Age Matters: Analyzing Age-Related Discussions in App Reviews

**arXiv ID:** 2601.21605 | [PDF](https://arxiv.org/pdf/2601.21605v1)

**作者:** Shashiwadana Nirmania `[一作]`, Mojtaba Shahin `[通讯]` (RMIT University)

**通讯引用:** 2104 | [OpenAlex ID](https://openalex.org/A5052783352)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过人工标注与自动分类相结合的方法，对70款热门Android应用的 7 000 万条评论进行年龄相关讨论的检测与主题分析。

**💡 创新点**

创新点在于首次构建了面向年龄讨论的词表与分类框架，并利用深度学习与大语言模型自动识别评论，随后系统性地提炼出六大主题，提供了开发者可落地的改进建议。

**🔧 技术方法**

主要技术包括传统机器学习（SVM、XGBoost）、Transformer 预训练模型（BERT、RoBERTa、DistilBERT）以及大语言模型（GPT‑4.1、Gemini、LLaMA）进行二分类；后者通过零样本与少样本提示实现。

**📊 数据集**

使用了由 Shahin 等人收集的 70 款应用的 7M 条评论，人工挑选并标注了 4,163 条评论，其中 1,429 条为年龄相关，2,734 条为非相关。

**📈 对比分析**

与传统机器学习与深度学习模型相比，RoBERTa 在 10 折交叉验证下取得最高精度 92.70%、召回 92.39%、F1 92.45%；大语言模型在少样本提示下表现略逊，但仍可为无标注场景提供可行方案。

**⚠️ 局限性**

局限包括：① 样本规模有限，仅涵盖 70 款应用且为英文评论；② 年龄词表和标签仍可能漏掉隐含或多义表达；③ 大语言模型存在误判与幻觉，且未进行系统的外部验证。

---

## 377. HydroSense: A Dual-Microcontroller IoT Framework for Real-Time Multi-Parameter Water Quality Monitoring with Edge Processing and Cloud Analytics

**arXiv ID:** 2601.21595 | [PDF](https://arxiv.org/pdf/2601.21595v1)

**作者:** Abdul Hasib `[一作]` (University of Frontier Technology), Anish Giri `[通讯]` (Bangalore University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5118827592)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并验证了 HydroSense，这是一套基于 Arduino 与 ESP32 的六参数水质实时监测系统，实现了 pH、溶解氧、温度、TDS、氮、液位等六项指标的联合监测与云端实时分析。

**💡 创新点**

采用双微控制器分布式架构、5点线性 pH 校准、实时中值滤波 TDS 与温度补偿、基于 Firebase 的低成本云集成，以及 90 天连续实验验证的高可靠性与 85% 成本降低等创新点。

**🔧 技术方法**

使用 Arduino Uno、ESP32 MCU、模拟测量与 5 点校准算法、温度补偿与中值滤波、串行通信、Wi‑Fi 边缘计算、Firebase 实时数据库、云端数据可视化和低功耗设计等技术。

**📊 数据集**

实验使用标准缓冲液（pH 4.00、6.86、7.00、9.18、10.01）、空气饱和溶解氧校准、不同浓度 NaCl 试液（0、342、500、750、1000 ppm）以及 90 天不同水体环境（淡水池、养殖池、实验室溶液）作为验证数据。

**📈 对比分析**

与市售单参量仪器和实验室仪器对比，pH ±0.08pH、DO ±0.2mg/L、TDS ±1.9% 的精度；系统 99.92% 正常运行、99.83% 云传输成功率，能耗 8.54Wh/日，成本 32,983 BDT，较商业系统降低 85%，性能接近专业级。

**⚠️ 局限性**

DO 传感器成本高、需要频繁手动 pH 校准、无全防水外壳、超声波水位受波动影响、对极端环境适配不足。

---

## 378. Is My RPC Response Reliable? Detecting RPC Bugs in Ethereum Blockchain Client under Context

**arXiv ID:** 2601.21593 | [PDF](https://arxiv.org/pdf/2601.21593v1)

**作者:** Zhijie Zhong `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**通讯引用:** 33941 | [OpenAlex ID](https://openalex.org/A5000582109)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

提出了一种基于上下文的 RPC 分析和模糊测试工具，用来检测以太坊客户端的 RPC 错误。

**💡 创新点**

创新点包括：①使用覆盖率驱动的交易上下文生成和交易改写技术，探索交易执行空间；②设计基于运行时状态的字节码变异策略；③采用上下文感知的 RPC 参数生成DSL；④利用多客户端响应差异构建交叉参考的 bug 诊断机制。

**🔧 技术方法**

技术手段包括：覆盖率驱动的模糊测试、事务改写与基本块/指令级变异、静态/动态数据流分析、RPC 参数/输出 DSL 注解、跨客户端交叉验证。

**📊 数据集**

使用的数据集主要是：①近两年以太坊主网的历史交易（作为种子和覆盖阈值基准）；②已部署的合约交易集合；③GitHub issues 收集的已知 RPC bug 集合。

**📈 对比分析**

通过与现有 RPC 检测器（如 EtherDiffer 等）在相同 bug 集合上进行对比实验，结果显示该工具检测率更高，发现 6 个新 bug 并被官方修复，其中 3 个被授予漏洞赏金；实验也展示了在多客户端测试网络中的高效性。

**⚠️ 局限性**

局限性：①主要针对 EVM/以太坊生态，需要人工注解 RPC；②交易改写可能遗漏隐式状态相关的 bug；③性能受客户端代码覆盖率、网络延迟和交易模拟成本的影响。

---

## 379. Language Models as Artificial Learners: Investigating Crosslinguistic Influence

**arXiv ID:** 2601.21587 | [PDF](https://arxiv.org/pdf/2601.21587v1)

**作者:** Abderrahmane Issam `[一作]` (Maastricht University), Gerasimos Spanakis `[通讯]` (Maastricht University)

**通讯引用:** 750 | [OpenAlex ID](https://openalex.org/A5010354377)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用双语 GPT-2 训练，系统地通过调整 L1 语言支配度和 L2 熟练度（即暴露年龄）来模拟跨语言影响（CLI），并通过跨语言预激和内部解释方法评估其对英语语法判断的影响。

**💡 创新点**

首次将语言模型作为可控统计学习者，对 CLI 进行系统消融实验，并结合跨语言预激、LogitLens 可视化和神经元重叠分析，揭示 L1 对 L2 的动态影响，为研究者提供可复现的计算框架。

**🔧 技术方法**

采用 GPT‑2 双语训练、线性学习率调度、交错 L1/L2 训练、跨语言预激、BLiMP 基准评估、FCE 语言偏好分析、LogitLens 内部状态可视化以及并行语言特定神经元检测（PLND）等技术。

**📊 数据集**

使用 OSCAR 语料库（双语 128M 行）、BLiMP 英语最小对、FCE 学习者语料、NLLB‑200 译文生成以及 WALS 语法距离表等数据集。

**📈 对比分析**

通过与单语基线模型在 BLiMP 上的准确率差异评估 CLI，并通过预激对比展示语法距离与干扰的相关性；机制层面 L1 token 率与神经元重叠呈线性负相关，说明模型能较好捕捉 CLI。

**⚠️ 局限性**

模型未能完全再现人类学习过程（缺乏真实语料多样性和语义上下文）、仅基于预定义语法距离、受模型规模和脚本差异影响，且实验仅覆盖有限语言组合。

---

## 380. Learning the Mechanism of Catastrophic Forgetting: A Perspective from Gradient Similarity

**arXiv ID:** 2601.21577 | [PDF](https://arxiv.org/pdf/2601.21577v1)

**作者:** Mutian Yang `[一作]` (Tsinghua University), Ji Wu `[通讯]` (Tsinghua University)

**通讯引用:** 5480 | [OpenAlex ID](https://openalex.org/A5029547618)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并阐明了大语言模型知识注入导致灾难性遗忘的机制，提出基于梯度相似性的理论框架，并基于此提出冲突神经元冻结、协同神经元更新的知识注入方法（CNL）。

**💡 创新点**

创新点在于：①将梯度相似性从全局降维到单个神经元层面，揭示冲突神经元（占50%–75%）与协同神经元（占25%–50%）的角色；②构造CNL训练规则，在极小学习率下理论上实现零灾难性遗忘；③将该规则推广到多种优化器并在更复杂数据集上验证其稳健性。

**🔧 技术方法**

使用梯度相似性分析、神经元级梯度分解、冲突/协同神经元识别、冻结/更新策略；实验采用SGD、Momentum、Adam、AdamW等优化器；理论推导基于一阶Taylor展开。

**📊 数据集**

实验数据集包括 MMLU、MedQA、ARC-C、CSQA，及其合并的 MMAC。

**📈 对比分析**

与传统全参数微调（FT）和经验回放（RP）进行对比：在 in‑set 场景下 CNL 实现零遗忘；在 out‑of‑set 场景下 CNL 将遗忘率降低 59.1%–81.7%；在四种优化器和复杂数据集上均保持低遗忘且学习效果与 FT 相当。

**⚠️ 局限性**

局限性：CNL 的零遗忘理论假设极小学习率和已知 Mastered Set，实际部署中难以满足；在估计 Mastered Set 的 out‑of‑set 评估时仍存在一定遗忘，且对完全在线、无监督的知识注入场景尚未给出完整解决方案。

---

## 381. Shaping capabilities with token-level data filtering

**arXiv ID:** 2601.21571 | [PDF](https://arxiv.org/pdf/2601.21571v1)

**作者:** Neil Rathi `[一作]` (Anthropic), Alec Radford `[通讯]` (Independent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过在预训练阶段对 token 级数据进行过滤，削弱语言模型的医学相关能力，同时保持其对其他知识领域的表现。

**💡 创新点**

提出 token 级过滤相较于文档级过滤是 Pareto 改进，且随着模型规模提升过滤效果显著增强，并且在对抗微调和拒绝训练上表现更稳健。

**🔧 技术方法**

使用稀疏自编码器生成弱监督 token 标注，训练高效的 token 分类器，并在预训练 Transformer 中实施 loss masking 或 token 替换两种过滤方式，同时进行指令微调。

**📊 数据集**

预训练语料为 FineWeb‑Edu，评估数据集包含 PubMed、bioRxiv、arXiv、Project Gutenberg、MedMCQA、MedQA‑USMLE 等医学与非医学文本。

**📈 对比分析**

与未过滤基线和文档过滤对比，利用文本困惑度、多选题（MedMCQA、MedQA‑USMLE、MMLU 医学子集）以及开放式问答评估；token 过滤模型在医学任务几乎无能力，非医学任务仅轻微损失，并在对抗微调和拒绝训练上优于文档过滤。

**⚠️ 局限性**

对 token 分类器的质量要求高且对标记噪声敏感；过滤难以实现多领域细粒度控制；在更大规模模型上的效果尚未验证。

---

## 382. FlexCausal: Flexible Causal Disentanglement via Structural Flow Priors and Manifold-Aware Interventions

**arXiv ID:** 2601.21567 | [PDF](https://arxiv.org/pdf/2601.21567v1)

**作者:** Yutao Jin `[一作]` (Southeast University), Junyong Zhai `[通讯]` (Southeast University)

**通讯引用:** 3292 | [OpenAlex ID](https://openalex.org/A5051126370)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 FlexCausal，结合 Block‑Diagonal VAE 与 Flow‑based Exogenous Prior 的因果分离表示学习框架，并引入流形感知方向性干预实现高保真反事实生成。

**💡 创新点**

创新点包括：①使用块对角协方差保证向量概念内部相关性；②采用独立正则化的 Normalizing Flow 捕捉非高斯外源噪声分布；③设计 Counterfactual Consistency Loss 与 Manifold‑Aware Directional Intervention 以保证因果一致性与生成在数据流形上；④构造 Filter 合成基准测试复杂分布下的因果学习。

**🔧 技术方法**

主要技术包括：变分自编码器、可逆正则化流（MAF）、结构因果模型（SCM）与增量噪声模型（ANM）、Hilbert‑Schmidt 独立性判别、线性辅助预测器、t‑SNE 视觉化。

**📊 数据集**

数据集涵盖：Filter（12k 图像，6 因果变量，双峰/多峰分布）、Pendulum（4 因果因子）、CelebA‑Smile 与 CelebA‑Age（各 30k 图像，情绪与人口统计因果结构）。

**📈 对比分析**

与 SCM‑VAE、CausalVAE、CausalDiffAE、CIDiffuser 等方法对比，使用 MIC、TIC 与 WD 三指标。FlexCausal 在所有基准上均取得最高 MIC/TIC，且在 Filter 上的 WD 最小，表明对非高斯分布的捕捉更精准，整体性能显著优于现有 CRDL 与扩散基准。

**⚠️ 局限性**

局限性包括：①依赖预定义的 DAG 与邻接矩阵，无法自动推断因果结构；②对流形感知干预的参数（如阈值、EMA 学习率）需人工调优；③在极大规模数据或更高维因果变量时，Flow 模型与 Block‑Diagonal 计算成本提升。

---

## 383. ASTRA: Automated Synthesis of agentic Trajectories and Reinforcement Arenas

**arXiv ID:** 2601.21558 | [PDF](https://arxiv.org/pdf/2601.21558v1)

**作者:** Xiaoyu Tian `[一作]` (Beike Language and Intelligence), Chengwei Liu `[通讯]` (Beike Language and Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

构建了一个全自动的端到端框架ASTRA，用于通过可验证的多轮强化学习训练工具增强型语言模型代理。

**💡 创新点**

创新点在于（1）利用工具调用图静态拓扑合成多轮轨迹，生成高质量SFT数据；（2）通过从问答拆分中提取语义拓扑并合成可执行、可验证的Python环境，实现多轮可判定RL；（3）将SFT与在线RL结合，采用F1式轨迹奖励和不相关工具混合，提升工具辨别和多步决策能力。

**🔧 技术方法**

技术包括：工具调用图随机游走、LLM驱动的任务与工具链生成、工具文档标准化、语义拆分与环境合成、沙箱执行验证、GRPO强化学习、Adaptive Batch Filling、token-level损失平均、上下文并行训练。

**📊 数据集**

数据集来源于公开的MCP工具注册表、RapidAPI等，合成约19k条工具文档、1,585个MCP服务器；通过工具链合成生成约10k+多轮任务；环境合成覆盖BFCL‑v3、τ²‑Bench、ACEBench等交互式基准以及AIME 2024/2025数学推理任务。

**📈 对比分析**

与闭源模型（Claude‑Opus‑4.5、Gemini‑3‑Pro等）和开源模型（GLM‑4.6、Qwen‑3‑32B等）在BFCL‑MT、τ²‑Bench、ACEBench上对比，ASTRA在匹配参数规模下均达到或逼近闭源系统，RL阶段提升幅度最大；在AIME基准上性能几乎无损。

**⚠️ 局限性**

局限性包括：环境合成仍需显式代码生成和沙箱验证，计算成本高；工具集覆盖受限于可解析的工具文档；RL训练对奖励设计和不相关工具混合敏感，过度偏离可能导致策略鲁棒性下降；未覆盖真实用户交互场景，需要进一步扩展。

---

## 384. Note2Chat: Improving LLMs for Multi-Turn Clinical History Taking Using Medical Notes

**arXiv ID:** 2601.21551 | [PDF](https://arxiv.org/pdf/2601.21551v1)

**作者:** Yang Zhou `[一作]` (Institute of High Performance Computing), Yong Liu `[通讯]` (Institute of High Performance Computing)

**通讯引用:** 20328 | [OpenAlex ID](https://openalex.org/A5100724297)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了基于医学笔记的对话生成与强化学习框架（Note2Chat），用于训练大型语言模型实现结构化、主动的病史采集与鉴别诊断。

**💡 创新点**

创新点包括：①利用大量公开医学笔记作为银标准监督，绕开对敏感对话数据的依赖；②决策树引导的生成+批判修订流程自动化生成高质量医患对话；③三阶段微调（SFT→自增轨迹→偏好优化）和单轮推理范式，实现高效、可解释的问答与诊断决策。

**🔧 技术方法**

技术手段主要包括：决策树生成与修订、监督微调（SFT）、自增轨迹采样、直接偏好优化（DPO）、单轮推理结构化记忆与计划、低秩适配（LoRA）等。

**📊 数据集**

数据集：从MIMIC‑IV中抽取10类疾病（如心衰、蜂窝织炎等）共4,972例，经过决策树生成和批判修订后得到约8,944条对话、67,077条成功轨迹与11,403对偏好对，此外构建单轮推理样本80,537条、95,811条偏好对。

**📈 对比分析**

与GPT‑4o、Gemini‑2.5‑flash等基线比较，单轮优化模型在信息收集F1、召回、Top‑1诊断准确率方面分别提升约46%、55%和70%（相对GPT‑4o提升>20%），对话轮次明显减少，显示出更高效、更准确的病史采集与诊断。

**⚠️ 局限性**

局限性包括：仍依赖预先构建的决策树和银标准笔记，可能对不同机构或语言环境的可迁移性有限；单轮推理虽提高可解释性，但对长期交互的动态调整能力仍待进一步验证；模型对罕见疾病或多系统交叉症状的泛化性能尚不充分。

---

## 385. Multi-Modal Time Series Prediction via Mixture of Modulated Experts

**arXiv ID:** 2601.21547 | [PDF](https://arxiv.org/pdf/2601.21547v1)

**作者:** Lige Zhang `[一作]` (Duke Kunshan University), Rex Ying `[通讯]` (Yale University)

**通讯引用:** 15012 | [OpenAlex ID](https://openalex.org/A5078337825)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于Mixture-of-Experts的多模态时间序列预测框架MoME，利用文本信息对专家路由和计算进行调制，实现跨模态控制；

**💡 创新点**

创新点在于将多模态交互从传统的token级融合转为专家级调制，提供几何视角解释MoE并设计了EiLM与RM两种调制机制；

**🔧 技术方法**

使用MoE架构、FiLM/HyperNet式专家调制、LLM文本编码、稀疏Top‑K路由、以及MLP/Transformer等时间序列基底；

**📊 数据集**

在MT‑Bench、TimeMMD等多种金融、气象、环境、能源、公共健康和社交公益等多模态时序数据集上进行实验；

**📈 对比分析**

与PatchTST、iTransformer、TS‑Mix、DLinear、TimeMoE等单模态基线以及token‑level融合方法对比，MoME在MAPE/MAE/MSE等指标上均实现数%到十数%的性能提升，尤其在长短期预测和趋势预测任务中表现突出；

**⚠️ 局限性**

局限性包括对文本质量的依赖、需要预训练LLM带来算力消耗、调制参数增大模型复杂度、对极端噪声文本的鲁棒性有限，并且目前仅验证了两模态场景，缺乏多模态扩展实验。

---

## 386. ARGORA: Orchestrated Argumentation for Causally Grounded LLM Reasoning and Decision Making

**arXiv ID:** 2601.21533 | [PDF](https://arxiv.org/pdf/2601.21533v1)

**作者:** Youngjin Jin `[一作]` (KAIST), Seungwon Shin `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过多专家LLM协同讨论构建量化双极论证图（QBAF），并将其映射为结构因果模型（SCM）以实现可解释的反事实干预和决策；

**💡 创新点**

将论证图与因果推理相结合，提供可追踪的边缘干预解释，并设计观测对齐的自我纠错机制，实现内部一致性与外部监督的动态平衡；

**🔧 技术方法**

核心技术包括量化双极论证框架（QBAF）、模块化量化语义、结构因果模型（SCM）与反事实干预、以及基于Jensen–Shannon散度的观测对齐优化；

**📊 数据集**

在多种任务上评估：MMLU‑Pro、MedQA、TruthfulQA、GPQA Diamond、MuSR（包括子任务）以及网络安全案例；

**📈 对比分析**

与单模型直接推理、Chain‑of‑Thought、以及基于多数投票的ensemble baseline比较，Argora在大部分基准上提升准确率，正向的Net Reversal Efficiency（NRE）和正向的Correctness Margin（Δ_correct）表明其纠错能力和决策稳定性；

**⚠️ 局限性**

局限主要在于仅支持单边干预、对底层LLM性能高度依赖、缺乏真实人类可解释性评估，以及对更大规模、多模态或开放式决策场景的适应性待进一步验证。

---

## 387. SONIC-O1: A Real-World Benchmark for Evaluating Multimodal Large Language Models on Audio-Video Understanding

**arXiv ID:** 2601.21666 | [PDF](https://arxiv.org/pdf/2601.21666v1)

**作者:** Ahmed Y. Radwan `[一作]` (Vector Institute for Artificial Intelligence), Shaina Raza `[通讯]` (Vector Institute for Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个名为 SONIC-O1 的真实世界音视频评测基准，用于系统评估多模态大型语言模型（MLLM）在摘要、MCQ、时序定位等任务上的表现，并提供人类验证的标注和人群公平性分析。

**💡 创新点**

首次在音视频基准中融合原始音频、视频与文本三模态，加入人群社会属性元数据实现公平性评估，并公开完整评测套件。

**🔧 技术方法**

采用多模态LLM（如 Gemini 3.0 Pro、Qwen3‑Omni 等）结合自动与人工标注、LLM 判分、ROUGE、mIoU 等指标进行评估。

**📊 数据集**

约 60 小时、231 条视频共 4,958 条 QA 实例，覆盖 13 个对话主题、5 个高风险领域，包含种族、性别、年龄等元数据。

**📈 对比分析**

通过对比闭源与开源模型在三项任务上的分数，发现闭源模型表现领先，时序定位任务差距最大（22.6%），不同人群间存在显著性能差异。

**⚠️ 局限性**

局限于英文、样本规模有限、对绝对时序推理仍弱、对某些人群样本不足，且模型需截断长视频导致误差。

---

## 388. Incremental Fingerprinting in an Open World

**arXiv ID:** 2601.21680 | [PDF](https://arxiv.org/pdf/2601.21680v1)

**作者:** Loes Kruger `[一作]` (Radboud University), Jurriaan Rot `[通讯]` (Radboud University)

**通讯引用:** 666 | [OpenAlex ID](https://openalex.org/A5015520868)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种面向开放世界的增量指纹识别方法，能够通过主动自动机学习与闭合世界指纹匹配相结合，自动发现并识别未知网络协议实现。

**💡 创新点**

创新点包括：①正式定义开放世界指纹识别问题；②设计并证明了增量指纹算法的正确性与复杂度优势；③通过自适应学习与指纹交叉验证显著降低误分类率。

**🔧 技术方法**

技术手段主要有：主动自动机学习（L*、TTT、适配式学习），指纹生成（分隔序列、ADG、SepSeq），等价/一致性检验（Wp、RandomWp、RandomWord）以及自适应增量指纹框架。

**📊 数据集**

实验使用了多种网络协议实现数据集：TLS、SSH、MQTT、BLE（BLEDiff）等，共计数千个实现（如596个TLS实现、BLE设备样例），并对这些实现构建了对应的真实FSM模型。

**📈 对比分析**

实验通过比较查询符号数、误分类率和交互次数与基线方法（重复L*、闭合世界指纹）进行对比；结果显示，增量指纹在完美教师场景下约10倍查询效率，误分类率从30–75%降至<1%，在有限预算或非完美教师场景下亦保持优于基线的性能。

**⚠️ 局限性**

局限性在于：①仅适用于确定性、无数据、可重置的FSM；②对非确定性、时序或数据丰富的协议支持有限；③性能高度依赖于等价查询的质量；④大规模协议学习仍需大量交互；⑤未利用源码或日志等灰盒信息。

---

## 389. Rethinking Fusion: Disentangled Learning of Shared and Modality-Specific Information for Stance Detection

**arXiv ID:** 2601.21675 | [PDF](https://arxiv.org/pdf/2601.21675v1)

**作者:** Zhiyu Xie `[一作]` (Shenzhen Technology University), Hu Huang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 97751 | [OpenAlex ID](https://openalex.org/A5017808266)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种多专家框架DiME，用于多模态立场检测。

**💡 创新点**

创新点在于将立场信息显式分离为文本主导、视觉主导和跨模态共享三类专家，并通过差异化监督实现专家专化。

**🔧 技术方法**

采用目标感知Chain-of-Thought提示生成推理链，BERT和ViT作为双编码器，三专家模块通过对比学习与余弦一致性损失训练，门控网络进行动态融合。

**📊 数据集**

使用四个主流多模态立场检测基准数据集：MTSE、MCCQ、MWTWT和MRUC。

**📈 对比分析**

在所有数据集上与多种单模态、双模态及提示式基线相比，DiME在宏观F1平均值上提升约5个百分点，并在零样本场景中仍保持领先。

**⚠️ 局限性**

局限性包括对视觉提示的依赖、对极度文本主导任务的效果略逊，以及门控融合在极端模式冲突时可能过度受限。

---

## 390. ILRR: Inference-Time Steering Method for Masked Diffusion Language Models

**arXiv ID:** 2601.21647 | [PDF](https://arxiv.org/pdf/2601.21647v1)

**作者:** Eden Avrahami `[一作]` (Tel Aviv University), Eliya Nachmani `[通讯]` (Ben Gurion University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无学习参数的推理时控制方法ILRR，通过在离散扩散语言模型的内部激活空间中与参考序列对齐，实现文本生成的属性控制。

**💡 创新点**

创新点在于将连续域的ILVR思想迁移到离散扩散模型，使用单个参考文本进行激活层级引导，并提出空间调制Steering以支持短参考指导长文本；同时通过可调的层级、时间步与强度实现灵活控制。

**🔧 技术方法**

利用离散扩散模型（如LLaDA、MDLM）的多层隐藏表示，采用平均池化提取语义向量，在每个设定层与时间步对生成激活进行加权更新；对长文本采用自适应下采样/线性插值配合余弦波调制强度。

**📊 数据集**

在毒性与情感（正向）两项属性上，使用15个前缀提示、每个提示20个生成，参考序列多样化；评估基于RoBERTa毒性/情感分类器。

**📈 对比分析**

与best‑of‑n、FK、PG‑DLM等采样与轨迹优化基线在相同计算预算（归一化NFE）下比较，ILRR在短文本上毒性准确率提升至71.2%（α=1.0），情感准确率100%；在长文本上毒性提升至13.1%、情感提升至61.7%，均明显优于基线。

**⚠️ 局限性**

局限性包括对参考序列的依赖（若参考与目标属性不匹配效果受限）、对池化窗口、层级与强度等超参的敏感性，需要手动调优；对高度细粒度或多属性控制的适用性尚未验证。

---

## 391. Sampling-Free Privacy Accounting for Matrix Mechanisms under Random Allocation

**arXiv ID:** 2601.21636 | [PDF](https://arxiv.org/pdf/2601.21636v1)

**作者:** Jan Schuchardt `[一作]` (Morgan Stanley), Nikita Kalinin `[通讯]` (Institute of Science and Technology Austria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

针对矩阵机制在随机分配（balls‑in‑bins）采样下的差分隐私计量，提出了无需采样的确定性隐私会计方法；

**💡 创新点**

创新点在于：①利用离散动态规划精确计算 Rényi 散度，从而得到可直接转换为 (ε,δ)‑DP 的上界；②引入条件组合（conditional composition）框架，构造逐步的独立高斯混合机制，从而在高隐私（小 ε）场景下得到更紧的保密性估计；

**🔧 技术方法**

技术手段包括：动态规划求解 Rényi 散度（带循环带宽化简），条件组合理论与逆危害函数（reverse hazard）判定，变分推断求隐私损失尾部上界，以及对比 Monte Carlo 采样计量；

**📊 数据集**

实验使用的并非真实数据集，而是基于各种矩阵机制（DP‑SGD、BSR、BISR、BandMF、BandInvMF、BLT、BandMF 等）和不同迭代次数、带宽、批大小参数的理论噪声乘子模拟；

**📈 对比分析**

与传统 Monte Carlo 计量方法相比，Rényi 会计在低隐私（ε≥2–4）时几乎与 MC 一致；在高隐私（ε≤1）时，条件组合会计显著优于 MC，能在相同 ε,δ 下使用更小的噪声乘子，从而提升模型可用性；

**⚠️ 局限性**

主要限制：① Rényi 会计对带宽 p 的时间复杂度呈指数级，导致大带宽时计算开销较大；② 条件组合采用粗粒度的“好/坏”划分，可能导致保守的上界；③ 变分隐私损失上界使用的变分族有限且线性近似保守，未充分挖掘更精细的尾部信息。

---

## 392. Turning Language Model Training from Black Box into a Sandbox

**arXiv ID:** 2601.21631 | [PDF](https://arxiv.org/pdf/2601.21631v1)

**作者:** Nicolas Pope `[一作]` (University of Eastern Finland), Matti Tedre `[通讯]` (University of Eastern Finland)

**通讯引用:** 4585 | [OpenAlex ID](https://openalex.org/A5050967195)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个基于浏览器的工具 Little Language Machine，让学生能在自己的设备上训练小型 Transformer 模型，并利用该工具研究学生对 LLM 行为解释的变化。

**💡 创新点**

突破性在于将 LLM 训练过程完全可视化并在低配设备上实现，使学习者从“黑盒”转向主动训练和探究。

**🔧 技术方法**

采用改进的 nanoGPT 风格 Transformer，配合 RMS 层归一化、RoPE 位置编码、WebGPU/WebGL 加速、混合精度训练和 KV 缓存，实现客户端训练与推理。

**📊 数据集**

使用小型文本语料库，如儿童短篇故事、莎士比亚作品以及学生自上传的文本，保持数据量适合低配设备。

**📈 对比分析**

通过前后测试开放式问答的定性内容分析，利用两比例 z 检验比较解释模型转变；实验表明数据中心化解释比例显著提升（从 13% 升至 38%），但未评估模型生成质量或更大规模性能。

**⚠️ 局限性**

局限性包括仅支持极小模型，缺乏与行业规模 LLM 的可比性；干预时间短，未能检验长期学习效果；数据集规模有限，难以覆盖多样语料；未进行外部效度验证。

---

## 393. HeRo-Q: A General Framework for Stable Low Bit Quantization via Hessian Conditioning

**arXiv ID:** 2601.21626 | [PDF](https://arxiv.org/pdf/2601.21626v1)

**作者:** Jinhao Zhang Yunquan Zhang `[一作]`, Daning Cheng `[通讯]` (Institute of Computing Technology)

**通讯引用:** 32 | [OpenAlex ID](https://openalex.org/A5010944123)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为HeRo-Q的后训练量化框架，通过在量化前对权重空间进行可学习的旋转-压缩变换来提升大型语言模型的低比特量化鲁棒性。

**💡 创新点**

创新点在于将Hessian谱信息与可学习的正交旋转相结合，先通过对角线平滑压缩Hessian最大特征值，再通过Cayley变换学习正交旋转，从而在不改变模型结构的情况下显著降低量化噪声对高曲率方向的影响。

**🔧 技术方法**

核心技术包括：Hessian对角线估计、可学习的对角平滑矩阵Dα、Cayley正交旋转矩阵R、以及将变换嵌入现有PTQ流程的参数融合实现。

**📊 数据集**

在实验中使用了Llama系列（3B、3.2‑1B、3.2‑3B、8B）和Qwen系列（2.5‑3B、2.5‑7B）模型，并在C4、Wiki2、GSM8K、MMLU、HellaSwag等公开数据集上评估。

**📈 对比分析**

与GPTQ、AWQ、SpinQuant、SmoothQuant、OmniQuant等SOTA PTQ方法对比，HeRo-Q在标准W4A8、W4A16以及极端W3A16和W4A4配置下均取得更低的Perplexity和更高的下游任务准确率，尤其在W3A16时提升约4%以上的GSM8K准确率。

**⚠️ 局限性**

局限性包括：需要额外的Hessian对角线估计和可学习变换参数，虽然算力开销微小但仍略高；对极端低比特（如W4A4）仍面临上限；在某些模型层级的对角平滑可能引入分布漂移，需进一步优化α的自适应策略。

---

## 394. PathReasoner-R1: Instilling Structured Reasoning into Pathology Vision-Language Model via Knowledge-Guided Policy Optimization

**arXiv ID:** 2601.21617 | [PDF](https://arxiv.org/pdf/2601.21617v1)

**作者:** Songhan Jiang `[一作]` (Harbin Institute of Technology), Yongbing Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 7115 | [OpenAlex ID](https://openalex.org/A5101653272)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了 PathReasoner 这一大规模全切片图像推理数据集，并基于该数据集研发了 PathReasoner-R1 模型，通过知识引导的强化学习实现可证据、结构化的链式推理。

**💡 创新点**

创新点在于：①利用医学知识图谱对推理路径进行严格校准，生成高质量、可验证的 CoT；②结合轨迹遮蔽式监督学习与多粒度知识奖励的 RL 训练，强制模型遵循医学实体一致性；③在 WSI 级别实现透明、可解释的诊断流程。

**🔧 技术方法**

采用 GPT‑4o 进行实体抽取与推理生成，构建 PrimeKG 与 PathoGraph 组合的医学知识图谱，使用轨迹遮蔽式监督学习与 Group Relative Policy Optimization（GRPO），并设计实体奖励、语义奖励和格式奖励的多层次回报函数。

**📊 数据集**

核心数据集为基于 TCGA 的 PathReasoner（约 22K 条全切片图像与报告），并在 SlideBench、WSI‑VQA、CPTAC、PathMMU 等公开 WSI/ROI 评测集上进行验证。

**📈 对比分析**

与 12 款 VLM（含无推理与推理型）在 PathReasoner、SlideBench 等基准对比，PathReasoner‑R1 在 PathReasoner 测试集上 LLM‑Score 2.583、BERT‑Score 0.779，WSI‑Level 平均 57.68%、74.95%、55.90% 等，均超过最强对手，显示显著性能提升。

**⚠️ 局限性**

局限性包括：①知识图谱覆盖范围与实体抽取精度有限，可能导致推理不完整；②模型仍可能出现视觉幻觉；③数据集主要来自 TCGA，缺乏多中心、多模态临床样本，需在真实临床环境中进一步验证与监督。

---

## 395. SENDAI: A Hierarchical Sparse-measurement, EfficieNt Data AssImilation Framework

**arXiv ID:** 2601.21664 | [PDF](https://arxiv.org/pdf/2601.21664v1)

**作者:** Xingyue Zhang `[一作]` (University of Washington), J. Nathan Kutz `[通讯]` (University of Washington)

**通讯引用:** 28958 | [OpenAlex ID](https://openalex.org/A5083450863)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

论文提出一种名为SENDAI的层次稀疏测量数据同化框架，利用极少传感器（约1.5%）重建全球NDVI场。

**💡 创新点**

创新点在于将仿真先验与自适应误差校正相结合，低频SHRED + 隐式神经表示的高频逐层剥离，解决稀疏观测与域移位下的结构保留难题。

**🔧 技术方法**

技术手段包括Takens嵌入、LSTM编码器、对抗式潜空间对齐、傅里叶频率剥离、坐标隐式神经表示（INR）以及梯度/TV平滑正则化。

**📊 数据集**

实验数据为NASA MODIS卫星的NDVI，覆盖六个全球分布、气候多样的研究站点，传感器随机布置，仅占像素的1.56%。

**📈 对比分析**

与SG+IDW、HANTS+IDW、Kriging及MMGN模型对比，SENDAI在SSIM上平均提升120–185%，RMSE显著下降，并在高频边界处实现更精细的重构。

**⚠️ 局限性**

主要局限包括假设空间结构在训练与目标期间保持不变、传感器布置为随机、缺乏跨洲泛化与多变量联合重建的验证。

---

## 396. Can Local Learning Match Self-Supervised Backpropagation?

**arXiv ID:** 2601.21683 | [PDF](https://arxiv.org/pdf/2601.21683v1)

**作者:** Wu S. Zihan `[一作]` (Brain Mind Institute), Guillaume Bellec `[通讯]` (Machine Learning Research Unit)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了本地自监督学习（local‑SSL）算法与全局反向传播自监督学习（global‑BP‑SSL）在深度网络中可以实现相同权重更新的理论，并基于此改进CLAPP算法，形成CLAPP++、CLAPP++DFB及CLAPP++both等变体，显著提升了卷积网络的自监督表示质量。

**💡 创新点**

创新点在于：① 在深度线性网络中证明了特定条件下local‑SSL与global‑BP‑SSL梯度完全等价；② 引入直接自上而下的反馈（DFB）和二维空间依赖的投影矩阵，以更好地逼近BP梯度；③ 通过理论指导实现的改进算法在多个图像数据集上刷新了local‑SSL的最高性能，逼近甚至接近global‑BP‑SSL的表现。

**🔧 技术方法**

使用的技术主要包括：自监督对比损失（CLAPP、SCFF等），Hebbian 类本地更新规则，反馈投影矩阵 B^l 的学习与结构化设计（二维空间依赖），以及深度卷积网络（VGG 结构）与线性/ReLU MLP 的训练与评估。

**📊 数据集**

主要使用的公开图像数据集为 CIFAR‑10、STL‑10 与 Tiny‑ImageNet，全部采用标准数据增强（SimCLR 方案）来生成正负样本。

**📈 对比分析**

与传统本地自监督方法（如 SCFF、LPL、Forward‑Forward 等）以及全局反向传播自监督基线（BP‑CLAPP++、BP‑InfoNCE）进行对比。实验显示，CLAPP++ 及其变体在 CIFAR‑10、STL‑10、Tiny‑ImageNet 上的线性分类准确率分别达到约 80.5%、78.7% 和 36.6%，与 BP‑CLAPP++ 基线相当或略优，明显优于之前的最佳 local‑SSL 方案（SCFF）。

**⚠️ 局限性**

局限性包括：① 证明仅适用于深度线性网络，非线性卷积网络仍需经验改进；② 需要在每一层学习或优化反馈矩阵 B^l，计算与存储成本相对较高；③ 对更复杂网络结构（如 ResNet、Transformer）及更大规模数据集的泛化尚未充分验证；④ 仅在有限的图像分类任务上测试，其他任务（如检测、分割）表现未知。

---

## 397. When Gradient Optimization Is Not Enough: $\dagger$ Dispersive and Anchoring Geometric Regularizer for Multimodal Learning

**arXiv ID:** 2601.21670 | [PDF](https://arxiv.org/pdf/2601.21670v1)

**作者:** Zixuan Xia `[一作]` (University of Bern), Fei Wang `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 80364 | [OpenAlex ID](https://openalex.org/A5070372567)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于几何正则化的轻量级插件，针对多模态学习中的表示坍塌和跨模态不一致问题，分别通过内部模态离散化和跨模态锚定来调控表示空间的几何结构。

**💡 创新点**

创新点在于首次将“内部离散化”与“跨模态锚定”两种几何约束结合为可插拔的正则化框架，并通过自适应 Pareto 权重动态平衡两项正则，显著提升了模态间互补性与单模态鲁棒性。

**🔧 技术方法**

技术上使用了对齐到单位球面、RBF 统一化的离散化损失、软锚定（超出阈值的距离惩罚）以及梯度注入式 Pareto 权重机制，全部可在现有模型中以微小改动加入。

**📊 数据集**

在音频-视觉、图像-文本以及射频-视觉等多种任务上验证：CREMA‑D、Kinetics‑Sounds、CUBICC、XRF55 等四大公开数据集。

**📈 对比分析**

与多种基线（DGL、DCMEM、MMVAE 等）对比，实验表明在音频/视觉、图像/文本、射频/视觉等所有模态组合下均获得 1–3% 的绝对提升，尤其在融合准确率和单模态鲁棒性方面同时提升，聚类指标（ACC/NMI/ARI）也均有正向提升。

**⚠️ 局限性**

局限性主要体现在只针对中等规模分类任务与固定后端结构，未验证在大规模数据、时序推理、生成或跨模态检索等更复杂任务，以及在大型 Transformer 等新型多模态模型上的表现。

---

## 398. Authenticated encryption for space telemetry

**arXiv ID:** 2601.21657 | [PDF](https://arxiv.org/pdf/2601.21657v1)

**作者:** Andrew Savchenko `[一作]` `[通讯]` (University of Technology Sydney), Andrew Savchenko (University of Technology Sydney)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计并实现了基于AES‑GCM的轻量级认证加密方案，用于卫星紧急遥测SGB信息，保证机密性、完整性和重放防护，同时兼容CCSDS协议及FIPS‑140；

**💡 创新点**

采用96位IV由64位时间戳+32位计数器生成，固定448位帧实现可预测带宽；利用AAD资产ID实现快速重放检测；无PKI、仅对称密钥；支持硬件加速并满足NASA‑STD‑1006A；

**🔧 技术方法**

AES‑GCM AEAD、时间戳+计数器IV、AAD、Replay窗口、硬件加速（SoC AES）、Python参考实现；

**📊 数据集**

未使用公开数据集，采用自定义SGB 202位载荷及测试向量进行加解密验证；

**📈 对比分析**

与未加密SGB及理论延迟对比；加密后帧大小增幅约79%但仍低于9.6 kb/s带宽；加解密总延迟约48 ms，比未加密多约20 ms；在硬件加速下CPU占用低、功耗可接受；

**⚠️ 局限性**

负载增大导致带宽消耗；依赖RTC时钟同步≤2 s窗口；不支持密钥轮换或前向保密；仅适用于固定格式SGB，未覆盖更大消息；不防止物理层攻击。

---

## 399. Improved Approximations for Dial-a-Ride Problems

**arXiv ID:** 2601.21652 | [PDF](https://arxiv.org/pdf/2601.21652v1)

**作者:** Jingyang Zhao `[一作]` (University of Electronic Science and Technology of China), Mingyu Xiao `[通讯]` (Kyung Hee University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一套新的多车调度算法，解决多车非抢占式调度问题（mDaRP），并给出了相对较低的近似比；

**💡 创新点**

通过新结构性质与 CVRP 技术相结合，设计了两种简单高效算法，并利用这些算法组合得到 O(√(log)) 的近似比，打破了先前 O(√n log²n) 的界限；

**🔧 技术方法**

主要使用了：
- 两阶段的分段与一致子路径构造（类似 ITP 方法）
- 基于树嵌入、Steiner 树与 mTSP 的近似解
- 递归分块与子树内的弱一致片段划分
- 随机化与 derandomization 技术
- 结构化分配与线性化的线段拆分

**📊 数据集**

无（纯理论研究，无实验数据集）；

**📈 对比分析**

与现有最优近似算法（如 O(√λ log m)、O(√n log²n) 等）对比，提出的算法在理论上取得了更优的近似比，且在跑时复杂度上从 O(m³ log λ) 降至 O(m² log λ) 或 O(m² log)，并通过组合进一步实现 O(√n log^(1/2)n)；

**⚠️ 局限性**

1) 仅适用于非抢占式调度；2) 对 λ 与 n 的比例有一定假设（如 λ = Ω(m²) 以保证某些归约成立）；3) 仍无法突破 O(√n) 这一根本性界限；4) 实际实现与实验验证尚未给出。

---

## 400. CAF-Mamba: Mamba-Based Cross-Modal Adaptive Attention Fusion for Multimodal Depression Detection

**arXiv ID:** 2601.21648 | [PDF](https://arxiv.org/pdf/2601.21648v1)

**作者:** Bowen Zhou `[一作]` (Otto von Guericke University Magdeburg), Ayoub Al-Hamadi `[通讯]` (Otto von Guericke University Magdeburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一种基于Mamba的跨模态自适应注意力融合框架CAF-Mamba，用于多模态抑郁检测。

**💡 创新点**

首次结合显式交叉模态交互编码器与隐式高阶关联的自适应注意力Mamba融合模块，并通过模态级注意力动态调整各模态贡献。

**🔧 技术方法**

采用Mamba网络、ResMamba结构、平均池化+线性+Softmax的模态注意力块以及多模态Mamba编码器进行特征提取与融合。

**📊 数据集**

在社交媒体公开的LMVD和D‑Vlog两大数据集上进行实验。

**📈 对比分析**

与传统机器学习、CNN、Transformer及其他基线方法在准确率、精确率、召回率、F1等指标上进行对比，CAF‑Mamba在所有指标上均超过同类方法，取得SOTA表现。

**⚠️ 局限性**

仍受限于预提取特征、模态种类有限以及在实验中未涉及实验室环境的泛化能力。

---

## 401. A Tilted Seesaw: Revisiting Autoencoder Trade-off for Controllable Diffusion

**arXiv ID:** 2601.21633 | [PDF](https://arxiv.org/pdf/2601.21633v1)

**作者:** Pu Cao `[一作]` (Beijing University of Posts and Telecommunications), Lu Yang `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 9452 | [OpenAlex ID](https://openalex.org/A5080093402)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

重新审视在潜在扩散模型中自编码器（AE）生成与重建权衡的评估偏差，理论推导AE诱发的条件漂移，并通过实验验证其对可控生成性能的影响。

**💡 创新点**

证明生成主导的评估会导致不可逆的条件漂移，并提出实例级重建指标（如PSNR）作为检测漂移的代理；构建多维度条件漂移评估协议，揭示gFID与可控性之间的弱关联；通过ControlNet实验验证AE质量对可控生成的实际影响。

**🔧 技术方法**

理论分析（条件漂移公式与最优极限证明）、Spearman相关性与可视化、ControlNet训练与评估、轻量预测器探测编码时信息丢失、使用多种重建与生成指标（gFID、rFID、PSNR、SSIM、LPIPS、CLIP、DINO等）。

**📊 数据集**

ImageNet数据集（及其公开变体），结合多种可控条件投影（canny边缘、深度、语义分割、身份识别、面部检测、CLIP/ DINO嵌入等）。

**📈 对比分析**

对比8种主流ImageNet自编码器及其33个变体，采用生成质量、重建质量和条件一致性三类指标进行排名与相关性分析；结果显示gFID与条件漂移相关性低，而实例重建指标（PSNR）与漂移高度相关；ControlNet实验表明，gFID优异的AE在可控生成中往往表现差。

**⚠️ 局限性**

局限性：实验仅基于ImageNet及其公开变体，未覆盖更大规模或多域真实世界数据；条件投影选择有限，可能无法覆盖所有可控任务；理论分析假设投影为Lipschitz，实际投影特性可能不完全符合。

---

## 402. Sustainable Open-Source AI Requires Tracking the Cumulative Footprint of Derivatives

**arXiv ID:** 2601.21632 | [PDF](https://arxiv.org/pdf/2601.21632v1)

**作者:** Shaina Raza `[一作]` (Vector Institute for Artificial Intelligence), Graham W. Taylor `[通讯]` (Vector Institute for Artificial Intelligence)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出轻量级的Data and Impact Accounting（DIA）框架，用于跟踪并可视化开源AI模型及其衍生品的碳排放与用水量

**💡 创新点**

在开源生态中实现生态系统级的环境足迹可视化与协调，提供标准化元数据、低摩擦测量与公共聚合仪表盘

**🔧 技术方法**

集成现有工具（如CodeCarbon、ML CO2 Impact、云平台能耗API）并采用统一的能源与水资源测量协议（WUE、PUE）

**📊 数据集**

基于公开的开源模型（如GPT‑3、Llama 2/3、BLOOM、Falcon等）与其衍生品的训练日志与硬件使用数据进行案例评估

**📈 对比分析**

通过与无报告基准对比，展示DIA能够揭示累计排放、识别高影响模型族，并促进基准优化，虽无具体性能指标但提升决策透明度

**⚠️ 局限性**

局限性包括自我报告的准确性与可验证性、对水资源外部因素的覆盖不足、未涵盖供应链影响、且无法强制执行，仅为可选透明层

---

## 403. Identifiable Equivariant Networks are Layerwise Equivariant

**arXiv ID:** 2601.21645 | [PDF](https://arxiv.org/pdf/2601.21645v1)

**作者:** Vahid Shahverdi `[一作]` (KTH Royal Institute of Technology), Kathlén Kohn `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 181 | [OpenAlex ID](https://openalex.org/A5003092978)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

证明了若深度网络在输入输出上满足整体的群对称性，并且满足可识别性假设，则网络各层必定在某些潜在空间上保持群对称性，即整体对称性必然转化为层级对称性。

**💡 创新点**

提出了一个与架构无关的抽象理论框架，将网络、子网络、对称性和可识别性统一表述，并给出了通用定理，首次在理论上解释了训练过程中出现的层级对称结构；同时把该理论应用于 MLP 和注意力网络。

**🔧 技术方法**

核心技术包括：抽象化的深度模型和子模型定义；对称群在潜在空间中的作用；参数可识别性定义及其弱可识别性；以及通过图论与组合论证明层级对称性。实验侧采用 CIFAR‑10 数据集训练简易 MLP 与多头注意力网络，并可视化过滤器与注意力权重来检验对称性。

**📊 数据集**

CIFAR‑10 数据集，用于训练自动编码器与分类器，验证网络在镜像对称（左右翻转）下的表现。

**📈 对比分析**

对比方式主要是定性可视化：过滤器呈现对称或镜像复制、注意力权重在镜像输入下的对称/置换；未给出数值指标，主要展示层级对称性是否出现。实验结果显示 Tanh 激活的 MLP 在镜像对称下自动学习了符号置换对称，而 GELU 由于激活性质导致出现负副本，说明理论预测与实践相符。

**⚠️ 局限性**

局限性：对 ReLU 激活的可识别性仍未完全证明；理论未覆盖跳连结构（残差网络）和跨令牌对称；未给出潜在空间中具体群作用的最优选择；未评价在实际任务中学习对称性与预先约束对称性之间的性能差异。

---

## 404. LLM4Fluid: Large Language Models as Generalizable Neural Solvers for Fluid Dynamics

**arXiv ID:** 2601.21681 | [PDF](https://arxiv.org/pdf/2601.21681v1)

**作者:** Qisong Xiao `[一作]` (National University of Defense Technology), Jie Liu `[通讯]` (National University of Defense Technology)

**通讯引用:** 93806 | [OpenAlex ID](https://openalex.org/A5100454174)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种将大型语言模型与物理驱动降维机制相结合的时空流体预测框架 LLM4Fluid。

**💡 创新点**

创新点包括物理信息约束的解耦降维、将文本提示映射为位置嵌入的模态对齐，以及利用预训练 LLM 的零样本与上下文学习能力。

**🔧 技术方法**

使用的技术包括物理驱动的变分自编码器、可逆实例归一化+分块 token 化、冻结预训练 LLM（如 OPT‑6.7B）+LoRA 微调、位置嵌入对齐和自回归预测。

**📊 数据集**

采用了 5 个二维流场数据集（Kolmogorov Low‑Re、High‑Re、lid‑driven cavity、channel、dam‑break），每个包含 1500 帧 128×128 分辨率的流场。

**📈 对比分析**

与多类 SOTA 时序模型（MLP、RNN、CNN、Transformer、Mamba、KAN、LLM 基准）在相同参数规模下对比，LLM4Fluid 在所有数据集上实现最低 MAE/MSE/SMAPE，且零样本迁移与上下文学习表现最好；相较于 Time‑LLM、GPT4TS 等模型，参数更少、推理更快。

**⚠️ 局限性**

局限性在于目前仅适用于二维、固定分辨率的流场，尚未扩展到三维或多物理耦合系统，且对极端尺度或复杂边界条件的泛化仍有待提升。

---

## 405. Scale-Dependent Semantic Dynamics Revealed by Allan Deviation

**arXiv ID:** 2601.21678 | [PDF](https://arxiv.org/pdf/2601.21678v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 406. Few-Shot Domain Adaptation with Temporal References and Static Priors for Glacier Calving Front Delineation

**arXiv ID:** 2601.21663 | [PDF](https://arxiv.org/pdf/2601.21663v1)

**作者:** Marcel Dreier `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Vincent Christlein `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 2892 | [OpenAlex ID](https://openalex.org/A5087093169)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

使用少量样本域适应、夏季参考影像和静态岩石掩模三项技术，将深度学习模型Tyrion‑T‑GRU迁移到Svalbard冰川，并将误差从1131.6 m降至68.7 m。

**💡 创新点**

创新点包括：①仅用每冰川一张标注实现少样本域适应；②利用夏季无冰混合参考影像提升对冰混合的识别；③加入静态岩石掩模作为空间先验输入，增强分割鲁棒性。

**🔧 技术方法**

技术手段：Tyrion‑T‑GRU（SwinV2编码器＋GRU＋卷积解码器），多时序SAR、夏季参考图像与岩石掩模作为多模态输入，最终采用5模型集成。

**📊 数据集**

数据集：Caffe基准数据集（7冰川，1996‑2020）与新构建的Svalbard SAR数据集（145个手工标注、5539训练图、192验证图、228测试图）。

**📈 对比分析**

性能比较：基线mde 1131.6 m → 少样本适应后445.3 m → 加夏季参考后204.6 m → 加岩石掩模后103.6 m → 5模型集成后68.7 m，IOU从53.9%提升至81.1%。

**⚠️ 局限性**

局限性：仅在Svalbard SAR图像上验证，未评估其他地区或传感器；未来需研究持续学习和跨域迁移。

---

## 407. When Life Gives You AI, Will You Turn It Into A Market for Lemons? Understanding How Information Asymmetries About AI System Capabilities Affect Market Outcomes and Adoption

**arXiv ID:** 2601.21650 | [PDF](https://arxiv.org/pdf/2601.21650v1)

**作者:** Alexander Erlei `[一作]` (University of Göttingen), Ujwal Gadiraju `[通讯]` (Delft University of Technology)

**通讯引用:** 3733 | [OpenAlex ID](https://openalex.org/A5038081564)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在实验中模拟AI系统的柠檬市场，研究信息不对称（披露程度）与AI质量分布（柠檬密度）如何影响用户的委托决策与最终收益。

**💡 创新点**

创新点在于将经典的柠檬市场理论引入人机交互实验，系统地考察部分披露与完整披露对用户行为的不同影响，并验证部分披露可在信息不完全的情况下提升整体效率。

**🔧 技术方法**

采用基于Bayesian更新的决策模型、随机效应面板回归分析、以及混合效应逻辑回归来评估披露策略与柠檬密度对委托率和收益的影响。

**📊 数据集**

使用了三组公开数据集：ISIC 2018 皮肤癌诊断数据、Kaggle 贷款批准数据以及酒店欺骗评论数据，分别对应图像、表格和文本任务。

**📈 对比分析**

通过对比不同披露条件（无披露、仅准确率披露、准确率+数据质量披露）以及三种柠檬密度（低、中、高），发现部分披露显著降低对低质量AI的委托比例并提升平均收益，但未显著改变总体委托率；完整披露在高柠檬密度下进一步减少低质量委托，但整体委托率仍低于预期，表明完全披露并不必然带来最优利用。

**⚠️ 局限性**

局限性包括：仅考察两维质量指标（准确率和数据质量），忽略公平、鲁棒性等因素；实验市场为静态，未考虑供应方动态与价格调整；委托决策仅二元化，未模拟更细粒度的信任与验证行为；样本主要来自在线众包，可能不具代表性。

---

## 408. Don't be so Stief! Learning KV Cache low-rank approximation over the Stiefel manifold

**arXiv ID:** 2601.21686 | [PDF](https://arxiv.org/pdf/2601.21686v1)

**作者:** Luca Benfenati `[一作]` (Polytechnic University of Turin), Alessio Burrello `[通讯]` (Polytechnic University of Turin)

**通讯引用:** 2180 | [OpenAlex ID](https://openalex.org/A5032095821)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种后训练KV缓存压缩方法，通过学习正交低秩投影基底直接最小化解码器层输出误差。

**💡 创新点**

创新点在于不再使用中间量的重构代理目标，而是直接优化解码器层输出，从而提升终端生成质量。

**🔧 技术方法**

采用Stiefel流形上的正交投影、QR正交化、基于激活统计的轻量级MLP预测器以及层级误差表进行自适应秩分配。

**📊 数据集**

在Llama‑3‑8B模型上使用WikiText‑2、C4、HellaSwag、PIQA、MMLU等数据集进行评估。

**📈 对比分析**

与EigenAttention以及FP16基线比较，在相同KV缓存比下，C4困惑度下降11.9点，MMLU准确率提升5.4%，整体性能-内存折衷更优。

**⚠️ 局限性**

局限性包括仅对单层独立校准、未考虑跨层误差耦合、仅在2048长度下验证，缺乏更长上下文及多模型的通用性验证。

---

## 409. Multimodal Visual Surrogate Compression for Alzheimer's Disease Classification

**arXiv ID:** 2601.21673 | [PDF](https://arxiv.org/pdf/2601.21673v1)

**作者:** Dexuan Ding `[一作]` (Macquarie University), Yuankai Qi `[通讯]` (Macquarie University)

**通讯引用:** 4454 | [OpenAlex ID](https://openalex.org/A5070842891)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出一种多模态视觉代理压缩（MVSC）框架，将三维结构性MRI压缩成三通道二维视觉代理，并利用冻结的二维视觉基础模型进行阿尔茨海默病分类。

**💡 创新点**

创新点在于：①引入体积上下文编码器（VoCE）在全局文本引导下聚合所有切片信息；②自适应切片融合（ASF）通过文本增强的跨切片注意力实现基于位置的切片特征融合；③通过文本（LLaVA‑Med生成）引导的上下文提升跨协议和解剖多样性的鲁棒性。

**🔧 技术方法**

使用2D CNN提取切片补丁、ViT/ConvNeXt基础模型（DINOv3、I‑JEPA）、跨注意力、FiLM特征调制、文本嵌入（BiomedBERT）等技术。

**📊 数据集**

在三大公开数据集上评估：AIBL、OASIS‑3、ADNI，总计 12,837 条 sMRI。

**📈 对比分析**

与多种基线（3D ResNet、M3T、MERLIN、AXIAL、RAPTOR、ViT 等）对比，MVSC 在二分类与多分类任务上均取得显著提升，例如 ADNI 上二分类 AUC 最高 98.5%（I‑JEPA），多分类 mAUC 最高 93.65%。

**⚠️ 局限性**

局限性包括：①对文本生成质量高度依赖，文本错误可能影响性能；②仅在标准数据集上验证，未知协议或扫描仪仍需进一步评估；③模型仍需较大计算资源以训练文本编码器，尽管推理时仅需冻结的 2D 基础模型。

---

## 410. From Instruction to Event: Sound-Triggered Mobile Manipulation

**arXiv ID:** 2601.21667 | [PDF](https://arxiv.org/pdf/2601.21667v1)

**作者:** Hao Ju `[一作]` (University of Macau), Zhedong Zheng `[通讯]` (University of Macau)

**通讯引用:** 9784 | [OpenAlex ID](https://openalex.org/A5034162160)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了声触发式移动操纵任务，使机器人能够主动感知并响应环境中的声音事件，而非被动执行人类指令。

**💡 创新点**

创新点包括三大声触发任务（SonicStow、SonicInteract、Bi‑Sonic Manipulation）、基于Acoustic Rendering的模拟平台 Habitat‑Echo，以及将多模态大语言模型（Omni‑LLM）作为高层任务规划器的层次化框架。

**🔧 技术方法**

技术包括：声学渲染（利用房间冲激响应 RIR）、多模态感知（音频+RGB/深度）、强化学习（PPO）训练低层策略、Omni‑LLM（Qwen2.5‑Omni/ Qwen3‑Omni）进行任务规划、以及多源音频混合以模拟双源场景。

**📊 数据集**

数据集为 Habitat‑Echo，包含可发声的刚体与关节物体（电话、闹钟、furby、门铃、水龙头等）及其多种音频实例；实验使用随机生成的 660 训练集与 222/355 测试集。

**📈 对比分析**

对比方法包括单独评估的 Individual 基线、提供真实规划的 Oracle 基线以及多种 Omni‑LLM 规划器。实验结果显示，Omni‑LLM 规划器在 SonicStow、SonicInteract 上的规划成功率分别约 78% 与 71%，相应整体成功率分别约 27% 与 19%；在 Bi‑Sonic Manipulation 上，Oracle 与 Qwen3‑Omni‑30B 在首源上取得最高成功率，但整体性能仍低于 Oracle，说明任务难度更高。

**⚠️ 局限性**

局限性包括：对音频与视觉信息的依赖使模型易受噪声干扰；双源任务中的多源音频分离仍表现不佳，导致次源交互成功率接近零；以及在真实物理环境中的模拟误差与域转移问题。

---

## 411. ScholarGym: Benchmarking Deep Research Workflows on Academic Literature Retrieval

**arXiv ID:** 2601.21654 | [PDF](https://arxiv.org/pdf/2601.21654v1)

**作者:** Hao Shen `[一作]` (Fudan University), Zhouhong Gu `[通讯]` (Fudan University)

**通讯引用:** 36 | [OpenAlex ID](https://openalex.org/A5089987379)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个名为 ScholarGym 的可复现模拟环境，用来评估深度研究工作流程在学术文献检索上的表现；并在此环境下对多种大语言模型进行系统实验。

**💡 创新点**

创新点包括：①将深度研究流程拆解为查询规划、工具调用和相关性评估三阶段，支持可插拔的内存机制；②使用 57 万条静态论文数据，消除外部 API 的非确定性；③设计了多维度评估指标（Recall、Precision、F1、Avg.Distance、GT Discard Rate）与诊断工具，能够细粒度诊断模型瓶颈。

**🔧 技术方法**

主要技术：大语言模型（Qwen3、GLM‑4.7、DeepSeek‑V3.2、GPT‑5.2、Gemini3‑Pro）进行迭代式查询规划；BM25（稀疏检索）与 Qwen3‑Embedding（稠密检索）做检索后端；使用自定义经验缓冲区压缩历史；评估时计算检索阶段与最终选择阶段的指标；对比思考增强（extended thinking）与标准模式的性能差异。

**📊 数据集**

使用的数据集：由 PaSa（含 AutoScholar、RealScholar）与 LitSearch 合并得到 57 万篇论文（CS、Physics、Mathematics），再按题目/摘要构建检索语料；Test‑Fast（200 题）与 Test‑Hard（100 题）两份测试子集，包含专家标注的 ground‑truth 论文集。

**📈 对比分析**

比较方法：在同一静态语料下，执行 5 轮迭代深度研究流程与直接查询基线；使用稀疏检索作为默认后端；对所有模型报告检索阶段和选择阶段的 Recall/Precision/F1；实验结果显示：专有模型（GPT‑5.2、Gemini3‑Pro）Recall≥0.95，F1最高≈0.45；开源模型在 extended‑thinking 模式下可提升 20‑30% 的 F1，但仍落后约 20%；稠密检索提升了 7‑26% 的 Recall，尤其对标准模型有显著效果。

**⚠️ 局限性**

局限性：①静态语料无法模拟实时文献更新和动态检索结果；②评估仅关注检索与相关性判断，未覆盖后续写作与综合推理；③open‑source 模型仍受限于缺乏大规模 RL 训练，导致 query‑planning 与 relevance‑assessment 仍有偏差；④实验主要集中在学术检索，难以直接推广到多模态或跨领域任务。

---

## 412. Seg-MoE: Multi-Resolution Segment-wise Mixture-of-Experts for Time Series Forecasting Transformers

**arXiv ID:** 2601.21641 | [PDF](https://arxiv.org/pdf/2601.21641v1)

**作者:** Evandro S. Ortigossa `[一作]` (Weizmann Institute of Science), Eran Segal `[通讯]` (Weizmann Institute of Science)

**通讯引用:** 73199 | [OpenAlex ID](https://openalex.org/A5012450539)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在Transformer时序预测框架中引入了基于连续时间段的稀疏Mixture-of-Experts（Seg‑MoE）层，改进了专家路由与处理方式。

**💡 创新点**

创新点在于：①将传统的逐时间步（token‑wise）专家路由改为以连续不重叠的时间段为单元进行路由，保留时序内的局部关联；②引入共享fallback专家提升稳定性；③在不同Transformer层使用多分辨率（多尺度）段长实现层级化时序建模。

**🔧 技术方法**

使用技术包括：Encoder‑only Transformer、稀疏MoE（Top‑K路由+共享专家）、Patch/segment分块、RoPE位置编码、FlashAttention、GQA、Huber损失、专家平衡正则化、BFloat16训练、AdamW优化器。

**📊 数据集**

评估数据集为七个公开多变量长期预测基准：ETTh1、ETTh2、ETTm1、ETTm2、Weather、ECL、Traffic。

**📈 对比分析**

与15个现有最先进基线（包括Informer、Autoformer、PatchTST等）在四个预测时长{96,192,336,720}上进行对比。Seg‑MoE在绝大多数数据集与时长上均实现MSE/MAE提升，平均提升幅度超过10%（如ETTh1 MSE降至0.381 vs 0.449），并在最长时长720上保持鲁棒性。Ablation实验显示segment‑wise路由比token‑wise优越，且多分辨率设置进一步提升性能。

**⚠️ 局限性**

局限性包括：①最佳段长（segment size）与多分辨率方案需针对不同数据集手工调优；②在极端长序列或更高维度数据上的可扩展性未充分验证；③共享专家虽稳定，但可能在极大模型规模下引入额外计算与内存开销。

---

## 413. OCRVerse: Towards Holistic OCR in End-to-End Vision-Language Models

**arXiv ID:** 2601.21639 | [PDF](https://arxiv.org/pdf/2601.21639v1)

**作者:** Yufeng Zhong `[一作]` (Meituan), Zhixiong Zeng `[通讯]` (Meituan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种端到端统一文本中心与视角中心 OCR 的方法 OCRVerse；

**💡 创新点**

创新点在于两阶段 SFT‑RL 多域训练：SFT 通过混合多域数据构建统一表示，RL 采用域特定奖励解决跨域冲突并提升专属性能；

**🔧 技术方法**

采用 Qwen3‑VL‑4B 视觉语言模型，冻结视觉编码器，使用自监督+自标注数据增强；使用 GRPO 进行 RL 优化，结合视觉编码器、视觉奖励和结构奖励；

**📊 数据集**

使用文本中心数据：多来源文档（自然场景、书籍、杂志、报纸等）及 OmniDocBench v1.5；使用视角中心数据：ChartMimic、Design2Code、UniSVG、Image2LaTeX‑plot、ChemDraw 等公开基准；

**📈 对比分析**

在 OmniDocBench v1.5 上获得 89.23 分，排名同类 VLM 前列；在 ChartMimic、UniSVG、Design2Code、Image2LaTeX‑plot、ChemDraw 等视觉 OCR 基准上，参数仅 4B 的 OCRVerse 的性能与大型闭源模型相当甚至超越多公开模型；

**⚠️ 局限性**

局限在于缺少显式布局感知，表格识别、阅读顺序等细节略逊；对极复杂表格结构需要更多数据；模型规模较小，未来可探索更大规模与更强布局模块。

---

## 414. Generative Design of Ship Propellers using Conditional Flow Matching

**arXiv ID:** 2601.21637 | [PDF](https://arxiv.org/pdf/2601.21637v1)

**作者:** Patrick Kruger `[一作]`, Hanno Gottschalk `[通讯]` (Institute of Mathematics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发并验证了一种基于条件流匹配（Conditional Flow Matching）的生成式模型，用于船舷叶片的逆向设计，能够根据给定的效率、推进比和推力系数等性能标签生成满足要求的多样化叶片几何；

**💡 创新点**

首次将条件流匹配应用于工程逆向设计，并提出利用代理模型合成数据增强以缓解数据稀缺问题，进一步证明该方法在低样本下仍能显著提升生成质量；

**🔧 技术方法**

采用条件流匹配与神经常微分方程（NODE）框架实现向量场学习；使用全连接前馈网络作为代理模型预测性能标签；通过拉丁超立方采样、VLM（OpenProp）与CAESES参数化模型生成数据；

**📊 数据集**

3000条设计向量（6个设计变量）及对应的三维性能标签（η*, J*, k_T*），其中2000条用于训练、1000条用于测试；此外还生成了10k、100k规模的合成数据集用于数据增强实验；

**📈 对比分析**

通过与真实标签的相对误差（MRE）和代理模型预测误差进行评估；在固定性能标签下，模型生成的设计满足目标的MRE通常低于5%，并能保持高达100%的有效率；数据增强在样本量较小且标签复杂度高（k_T*）时，MRE明显下降；多样性测试显示生成设计在参数空间上呈广泛分布；在实际设计任务中，生成7/100个设计通过代理模型与VLM验证均满足目标性能；

**⚠️ 局限性**

仅考虑了6个设计变量，未涉及倾斜、翘曲等关键几何特征；使用低阶VLM近似，忽略船体-叶片相互作用；代理模型误差在高样本下可能不显著，影响数据增强效果；实验仅在单一性能标签集上验证，缺乏更广泛的多目标评估；模型训练受限于低端GPU，实际工程部署需进一步优化；

---

## 415. LAMP: Look-Ahead Mixed-Precision Inference of Large Language Models

**arXiv ID:** 2601.21623 | [PDF](https://arxiv.org/pdf/2601.21623v1)

**作者:** Stanislav Budzinskiy `[一作]` (University of Vienna), Philipp Petersen `[通讯]` (University of Vienna)

**通讯引用:** 801 | [OpenAlex ID](https://openalex.org/A5041074956)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种LAMP（Look‑Ahead Mixed‑Precision）自适应推理策略，在Transformer推理中仅对少量关键的内积结果使用高精度重算，以降低数值误差。

**💡 创新点**

创新点在于将误差分析与算子组合深度结合，给出理论保证并用贪心算法快速选取需重算的组件，突破了传统全局低精度累积的瓶颈。

**🔧 技术方法**

采用浮点误差分析、混合精度矩阵乘法、软最大函数与层归一化的LAMP矩阵求解，配合自定义的部分单精度浮点格式进行实验。

**📊 数据集**

使用GPT‑2 XL模型在OpenWebText数据集上进行实验，测试不同位宽与阈值组合的效果。

**📈 对比分析**

与全FP32、TF32、BF16等统一低精度实现比较，结果显示即使低至7位尾数，LAMP通过约3–35%重算即可使KL散度和预测翻转率降低10–1000倍，性能显著提升。

**⚠️ 局限性**

局限在于仅在GPT‑2架构上验证，缺乏对更大模型、不同任务及硬件实现的评估；对极长上下文的兼容性有限，且未结合量化或训练时的混合精度。

---

## 416. Similarity of Processing Steps in Vision Model Representations

**arXiv ID:** 2601.21621 | [PDF](https://arxiv.org/pdf/2601.21621v1)

**作者:** Matéo Mahaut `[一作]` (Universitat Pompeu Fabra), Marco Baroni `[通讯]` (Universitat Pompeu Fabra)

**通讯引用:** 15769 | [OpenAlex ID](https://openalex.org/A5038612405)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对多种视觉模型（iGPT、DINOv2、ConvNeXt、ViT）的层间表示进行定量比较，探讨它们在训练目标与架构不同的情况下，是否以及如何在中间步骤上趋同；

**💡 创新点**

首次将信息不平衡（Information Imbalance）作为邻域相似性度量，用于揭示不同模型在中间层的处理流程差异，并将低级视觉特征与语义一致性结合，说明不同目标对中间表示的保留与丢弃机制；

**🔧 技术方法**

使用信息不平衡度量、邻域可视化、线性探针、Jaccard相似度、以及边缘密度/颜色温度/纹理复杂度等低级特征的统计分析；

**📊 数据集**

ImageNet‑21K 验证集（约10⁴张图像）以及 ManyNames 数据集（多标签人类标注）；

**📈 对比分析**

通过信息不平衡对同深度层之间的相似度进行比较，发现相同深度层在不同模型间预测效果最佳；低层保留高比例低级特征，深层呈现语义一致性，iGPT 的中间层与所有层高度相似；性能上，DINOv2 在后期层保留更多信息，ConvNeXt 的层间变化更剧烈；

**⚠️ 局限性**

仅针对规模巨大的预训练模型；信息不平衡仅捕获局部邻域关系，可能忽略全局结构；未对小模型或多任务场景进行验证；因果机制仍未完全解析。

---

## 417. SWE-Spot: Building Small Repo-Experts with Repository-Centric Learning

**arXiv ID:** 2601.21649 | [PDF](https://arxiv.org/pdf/2601.21649v1)

**作者:** Jinjun Peng `[一作]` (Columbia University), Yangruibo Ding `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了面向仓库的学习（RCL）框架，通过在单一代码仓库内多维度交互式训练，构建了4B参数的仓库专家模型。

**💡 创新点**

创新点在于把传统的任务中心学习（TCL）从横向任务广度转向纵向仓库深度，利用四大仓库体验单元（软件设计、上下文实现、演化回放、语义运行一致）使模型在训练阶段内化仓库“物理”，实现跨任务的深层知识迁移。

**🔧 技术方法**

主要技术包括：基于教师模型生成的仓库体验轨迹、监督微调（SFT）、多任务训练框架、与检索/测试时扩展对比、以及对比实验中的推理成本与样本效率评估。

**📊 数据集**

使用的公开数据集包括SWE‑Bench‑Verified、TDD‑Bench‑Verified、FEA‑Bench、SWE‑QA 等四个软件工程任务集合，并在 7 个高密度仓库（如 Django、SymPy 等）上构建了时间严格的评估数据集。

**📈 对比分析**

在对比实验中，RCL‑4B 在跨任务（Issue Resolution、Test Gen、Feature Impl、Codebase QA）上平均通过率约 17%，超过同尺寸或更大 8× 参数的开放式模型（如 Qwen3‑Coder‑30B、Gemma‑3‑27B）以及效率型商业模型（GPT‑4.1‑mini、GPT‑5‑nano），并显著降低推理回合数与 token 消耗；在受限数据预算下，RCL 的样本效率和推理成本均优于 TCL。

**⚠️ 局限性**

主要局限包括：RCL 需要完整的参数微调，参数效率方法（如 LoRA）效果有限；在多仓库共训练时可能出现负迁移，无法兼顾所有仓库；实现依赖教师生成体验，成本和可扩展性待进一步研究。

---

## 418. TabClustPFN: A Prior-Fitted Network for Tabular Data Clustering

**arXiv ID:** 2601.21656 | [PDF](https://arxiv.org/pdf/2601.21656v1)

**作者:** Tianqi Zhao `[一作]` (Renmin University of China), Qiong Zhang `[通讯]` (Renmin University of China)

**通讯引用:** 18465 | [OpenAlex ID](https://openalex.org/A5100407278)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于Prior‑Fitted Network的无监督聚类模型 TabClustPFN，能够在单次前向推断中同时估计簇数与簇分配。

**💡 创新点**

创新点在于将簇数推断与簇分配分解为 Cardinality Inference Network 与 Partition Inference Network，并利用可微的 SoftARI 损失实现完全无监督端到端训练，同时通过迭代交叉注意力解码器实现高效的聚类推断。

**🔧 技术方法**

技术手段包括 Transformer‑based In‑Context Learning、可微 SoftARI 损失、迭代交叉注意力解码器、对齐先验的多样化合成数据（GMM 与 ZEUS 先验混合）以及解耦优化。

**📊 数据集**

预训练使用约 130 M 个由 GMM 与 ZEUS 先验产生的合成任务，评估数据集包含 44 个真实 tabular 数据集和 50 个合成数据集。

**📈 对比分析**

与传统聚类、深度聚类和其它 PFN 基线对比，TabClustPFN 在 ARI/NMI 排名、簇数估计误差以及推理速度上均取得了显著优势。

**⚠️ 局限性**

局限性包括对最大簇数上限的敏感性、在极度重叠或高维稀疏数据上的性能仍有限，以及对先验数据分布的依赖可能导致在完全不同任务上需要重新预训练。

---

## 419. XFACTORS: Disentangled Information Bottleneck via Contrastive Supervision

**arXiv ID:** 2601.21688 | [PDF](https://arxiv.org/pdf/2601.21688v1)

**作者:** Alexandre Myara `[一作]` (IBENS), Auguste Genovesio `[通讯]` (IBENS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了弱监督的VAE框架XFactors，能够对指定属性因子进行解耦并实现对潜在空间的显式控制。

**💡 创新点**

创新点是将Disentangled Information Bottleneck与InfoNCE对比监督结合，采用子空间分解而非对抗或分类器来实现多因子解耦并保持可扩展性。

**🔧 技术方法**

采用变分自编码器、KL正则化、InfoNCE对比损失以及信息瓶颈理论的子空间分解技术。

**📊 数据集**

在多种合成数据集（如dSprites、DSprites）以及真实数据集CelebA上进行实验。

**📈 对比分析**

与β‑VAE、DisCo、GAN‑based和Diffusion‑based等方法比较，XFactors在多数数据集上取得了最优或相近的离散度、完整度等解耦指标，并实现了高质量的属性交换生成。

**⚠️ 局限性**

主要局限在于传统VAE的压缩-重构权衡导致重建质量受限，尤其在CelebA等复杂数据上生成图像可能不够清晰。

---

## 420. Do Not Waste Your Rollouts: Recycling Search Experience for Efficient Test-Time Scaling

**arXiv ID:** 2601.21684 | [PDF](https://arxiv.org/pdf/2601.21684v1)

**作者:** Xinglin Wang `[一作]` (School of Computer Science, Beijing Institute of Technology), Kan Li `[通讯]` (School of Computer Science, Beijing Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练的测试时缩放方法——Recycling Search Experience（RSE），通过在多轮批量搜索中将先前轨迹的中间结论与失败模式提炼为经验库，并在后续搜索中利用该经验库进行正向加速与负向剪枝，从而将原本无记忆的随机采样转化为累积式、经验驱动的搜索过程。

**💡 创新点**

创新点包括：①将测试时搜索视为累计过程，首次系统性地将每一次轨迹的中间结论与失败模式作为可复用经验；②自引导经验蒸馏机制，利用模型自身评估能力从完整轨迹中提取可验证的正向经验与负向约束；③语义去重策略，保持经验库多样性并控制上下文长度；④批量经验引导搜索的并行与递归结合，兼顾搜索宽度与深度。

**🔧 技术方法**

核心技术为：批量经验驱动搜索、基于模型自评的经验蒸馏、正向与负向经验的拆分与合并、语义相似度去重与经验库管理，以及在有限上下文窗口内的 prompt 设计。

**📊 数据集**

实验数据集包括：HMMT24、HMMT25、IMO-Bench、HLE‑Math‑text（100样本）以及 Deepseek‑V3.2 在 HLE‑Math‑text 上的评估。

**📈 对比分析**

与标准采样、投票（Majority Voting）、自我迭代（Self‑Refine）以及历史经验拼接（PaCoRe）等基线比较，RSE 在所有模型（Qwen3‑30B、Qwen3‑4B、Phi‑4、Deepseek‑V3.2）和所有数据集上均实现了更高的 Pass@1，尤其在高难度样本上提升显著；同时在相同 FLOPs 或推理预算下，RSE 的精度提升率最高，构成 Pareto 前沿。

**⚠️ 局限性**

局限性包括：①对模型自评能力高度依赖，若模型评估失准则蒸馏经验失效；②去重阈值需手动调优，过大或过小都会影响效果；③实验仅覆盖数学推理类任务，未知在其他推理或生成任务上的泛化能力；④经验库维护与上下文注入仍带来额外计算与内存开销。

---

## 421. FIT: Defying Catastrophic Forgetting in Continual LLM Unlearning

**arXiv ID:** 2601.21682 | [PDF](https://arxiv.org/pdf/2601.21682v1)

**作者:** Xiaoyu Xu `[一作]` (The Hong Kong Polytechnic University), Haibo Hu `[通讯]` (The Hong Kong Polytechnic University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了可扩展的持续去学习框架 FIT‑PCH，通过冗余过滤、重要性感知更新和层级定位，解决 LLM 持续删除请求导致的灾难性遗忘和恢复攻击。

**💡 创新点**

创新点在于：①双阶段冗余过滤避免梯度叠加；②基于梯度模长动态选择单射方法；③使用 Shapley 近似定位关键层仅更新这些层；④构建统一 PCH 基准和对称指标 F.D./R.U。

**🔧 技术方法**

采用了 embedding 相似度+损失差检验的过滤、梯度模长重要性评分、Shapley 近似层级挑选，以及 SimCSE 嵌入和 GA、NPO、RLabel 等单射原语实现。

**📊 数据集**

使用自研 PCH 数据集（600 条 synthetic QA 对，涵盖个人信息、版权、危害内容），并在四个开源 LLM（Yi‑6B、Llama‑2‑7B‑chat‑hf、Llama‑3‑8B、Llama‑3‑8B‑Instruct）上实验。

**📈 对比分析**

与 GA、GA+GD、GA+KL、NPO、NPO+KL、RLabel、PISCES、O^3、ALKN 等基线对比，FIT‑PCH 在 F.D. 与 R.U. 上均优于对手，在 MMLU/CommonsenseQA/GSM8K 等下游任务保持高分，并在再学习与量化恢复攻击下更具鲁棒性。

**⚠️ 局限性**

局限性包括：依赖人工验证的 synthetic 数据；对不同 LLM 的微调开销较大；在极大删除请求量下仍可能出现轻微参数漂移与性能衰减；对 O^3 的兼容性缺失导致某些对比不完整。

---

## 422. Expected Return Causes Outcome-Level Mode Collapse in Reinforcement Learning and How to Fix It with Inverse Probability Scaling

**arXiv ID:** 2601.21669 | [PDF](https://arxiv.org/pdf/2601.21669v1)

**作者:** Abhijeet Sinha `[一作]` (National University of Singapore), Dianbo Liu `[通讯]` (National University of Singapore)

**通讯引用:** 6327 | [OpenAlex ID](https://openalex.org/A5014407399)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文针对强化学习中的多模态终端奖励问题，证明期望回报目标本身会导致“模式坍塌”，并提出通过逆概率缩放（IPS）修改奖励信号的方法，进而实现IPS‑GRPO，一种在现有GRPO框架下可直接使用的多模态策略优化算法。

**💡 创新点**

① 发现并严格证明了期望回报最大化在理想学习动态下不可避免地产生终端结果的模式坍塌；② 提出了逆概率缩放的理论正则化思想，用奖励除以终端结果出现概率，消除概率放大带来的正反馈；③ 将这一思想以drop‑in方式嵌入GRPO，得到IPS‑GRPO，兼容KL正则与裁剪。

**🔧 技术方法**

理论分析（梯度流、对数比率动力学）；逆概率缩放奖励；GRPO与PPO式的组策略梯度；经验结果概率估计与阈值剪辑；KL正则与熵正则；实验评估。

**📊 数据集**

① Hyper‑grid（离散多模态奖励环境）；② HypoSpace（因果推断、3D体素重建、Boolean/DNA交互三子任务）；③ 化学语言模型药物发现任务（SYNTH、ALL‑AMIDE奖励）。

**📈 对比分析**

与GRPO、FlowRL、REINVENT 等基线对比。IPS‑GRPO在Hyper‑grid任务中将ℓ1距离从≈0.028降至≈0.003；在HypoSpace中恢复率分别从约16/31/9%提升到约44/90/61%；在分子生成任务中Yield提升、OB100下降，显著优于REINVENT。整体表现显示，IPS‑GRPO在保持或提升回报的同时显著提升了多模态覆盖率。

**⚠️ 局限性**

① 依赖经验概率估计，需在每组采样中统计结果，组大小与剪辑阈值对性能有显著影响；② 在大规模或连续结果空间时估计噪声大，需要更复杂的概率估计；③ 目前仅适用于终端奖励的单步RL，无法直接推广到dense‑reward或actor‑critic（如PPO）框架；④ 逆概率缩放虽能消除模式坍塌，但在极端稀疏奖励或极少出现的结果时可能导致梯度爆炸。

---

## 423. AdaptBPE: From General Purpose to Specialized Tokenizers

**arXiv ID:** 2601.21665 | [PDF](https://arxiv.org/pdf/2601.21665v1)

**作者:** Vijini Liyanage `[一作]` (Sorbonne University), François Yvon `[通讯]` (Sorbonne University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种轻量化的后训练BPE词表适配算法（Adapt‑BPE），通过在固定合并预算下替换低效合并，优化特定域或语言的子词分词；

**💡 创新点**

创新点在于只改动预训练词表的合并顺序与截断，而不改动模型权重，实现零成本、无缝切换的词表细化；

**🔧 技术方法**

核心技术为BPE合并序列的贪婪替换与虚拟合并处理，利用自适应语料统计频率进行合并选择；

**📊 数据集**

实验使用了多语言维基百科、PubMed、EMEA、SIB、FLORES等数据集，在GPT‑2、BLOOM、-3等多模模型上验证；

**📈 对比分析**

与三种基线（First_k、First_k>0、Top_k）相比，Adapt‑BPE在15k合并预算下压缩率最高、困惑度最优，且在医学、分类、翻译任务中逼近完整词表的性能；

**⚠️ 局限性**

局限包括手工设定合并预算、缺乏与模型参数联合优化、仅针对BPE模型评估，未探究其他分词方案或更大模型。

---

## 424. Epistemic Uncertainty Quantification for Pre-trained VLMs via Riemannian Flow Matching

**arXiv ID:** 2601.21662 | [PDF](https://arxiv.org/pdf/2601.21662v1)

**作者:** Li Ju `[一作]` (Uppsala University), Prashant Singh `[通讯]` (Uppsala University)

**通讯引用:** 4409 | [OpenAlex ID](https://openalex.org/A5100706486)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种通过条件黎曼流匹配（Conditional Riemannian Flow Matching）量化预训练视觉-语言模型（VLMs）中的认知不确定性的方法。

**💡 创新点**

创新点在于将嵌入的负对数密度作为认知不确定性的代理，并通过黎曼流匹配在超球面上计算概率密度，从而提供了一种可扩展且内在的度量模型信心的方法。

**🔧 技术方法**

使用了条件黎曼流匹配（CRFM）技术来学习嵌入的概率密度，并通过流动场来实现高效的密度估计。

**📊 数据集**

使用了三个代理数据集：Conceptual Captions、DataComp-1B 和 LAION-2B，随机抽取了每个数据集的100万对图像-文本样本进行训练。

**📈 对比分析**

与现有的ProbVLM和蒙特卡洛丢弃（MCDO）方法进行比较，结果显示该方法在选择性分类中显著优于基线，达到了近乎完美的模型误差与不确定性之间的相关性。

**⚠️ 局限性**

限制在于该方法的有效性依赖于代表性代理数据集的可用性；显著的领域差距或低分辨率输入可能会降低密度估计的质量。此外，依赖嵌入密度可能会带来公平性风险，因为来自代表性不足的人群的数据可能因数据稀疏而被标记为高不确定性。

---

## 425. Improved Approximations for the Unsplittable Capacitated Vehicle Routing Problem

**arXiv ID:** 2601.21660 | [PDF](https://arxiv.org/pdf/2601.21660v1)

**作者:** Jingyang Zhao `[一作]` (University of Electronic Science and Technology of China), Mingyu Xiao `[通讯]` (Kyung Hee University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了针对不可分配车辆路径规划（unplittable CVRP）的两种改进近似算法，分别适用于固定车辆容量和任意车辆容量，显著降低了已知的最优近似比。

**💡 创新点**

创新点在于：① 对 ITP 算法进行细粒度改造（δ‑ITP+），通过预处理大型需求客户并使用 trivial tours 提升性能；② 将 LP 取整与随机化与 ITP 结合，针对固定容量利用 n^k 条目的全枚举 LP，针对任意容量利用 n^O(1/δ) 条目的稀疏 LP；③ 通过解析积分上界与对需求分布的精确建模，得到更优的近似比公式；④ 结合 Blauth‑Traub‑Vygen 的 TSP 近似改进进一步压缩常数。

**🔧 技术方法**

主要技术包括：迭代路径划分（ITP）与其 δ‑ITP+ 变体、匹配与 trivial‑tour 预处理、基于 n^k 或 n^O(1/δ) 的线性规划与随机取整、TSP 近似求解、对需求分布的积分分析以及对实例的“simple/hard”分类策略。

**📊 数据集**

本文为理论分析，没有使用任何实验数据集；所有结果均基于理论证明与已有的 TSP 近似比 α（≈1.5）。

**📈 对比分析**

与之前最优近似比 3.1932（Friggstad 等）相比，固定容量版可降至约 3.0894，任意容量版降至约 3.1755；进一步利用 Blauth‑Traub‑Vygen 的改进可再提升几千分之一。对比表面上提升幅度不大，但在理论上已突破旧界限。

**⚠️ 局限性**

局限性包括：① 复杂度对容量 k 为指数级（固定容量版 n^O(k)）；② 任意容量版需要设置常数 δ，算法时间为 n^O(1/δ)；③ 仅适用于不可分配版本，未覆盖可分配或单位需求特殊情况；④ 结果高度依赖当前最优 TSP 近似比 α，若 α 改进则整体提升有限。

---

## 426. Gauge-invariant representation holonomy

**arXiv ID:** 2601.21653 | [PDF](https://arxiv.org/pdf/2601.21653v1)

**作者:** Vasileios Sevetlidis `[一作]` (Athena Research Center), George Pavlidis `[通讯]` (Athena Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种基于循环路径的表示层几何度量——表示全周环量（representation holonomy），并给出了可计算的估计器。

**💡 创新点**

创新点在于：①将路径依赖的几何信息量化为gauge invariant的holonomy；②证明其对正交/仿射变换不变、线性层为零、小半径下线性递增；③将holonomy与模型鲁棒性关联，提供新的诊断视角。

**🔧 技术方法**

技术手段包括全局白化（ZCA或z‑score）、共享邻域子空间投影、低维旋转Only Procrustes、SVD降维、嵌入到全维旋转空间；理论上使用Davis–Kahan/Wedin perturbation、曲率分析与小半径极限；实验方法为构造2D PCA平面循环、采样正交旋转，计算层级holonomy。

**📊 数据集**

使用的数据集为 MNIST（含10°旋转）和 CIFAR‑10/100（含裁剪、翻转、标准扰动），网络为 2‑层 MLP、ResNet‑18 等。

**📈 对比分析**

与传统点相似度指标（CKA、SVCCA、PWCCA）比较，发现 holonomy 能区分在 CKA 高度相似但鲁棒性不同的模型；实验显示 holonomy 随循环半径增大而升高，并与对抗/破坏性鲁棒性呈显著正/负相关，能够在训练早期预测鲁棒性。

**⚠️ 局限性**

局限性包括：①仅衡量局部路径依赖的曲率，受循环设计与白化方式影响；②忽略尺度、剪切等非旋转变换；③需要共享邻域与全局白化，可能在高维/深层模型中计算成本较高；④结果对离群点和稀有模式敏感。

---

## 427. RSGround-R1: Rethinking Remote Sensing Visual Grounding through Spatial Reasoning

**arXiv ID:** 2601.21634 | [PDF](https://arxiv.org/pdf/2601.21634v1)

**作者:** Shiqi Huang `[一作]` (Nanyang Technological University), Bihan Wen `[通讯]` (Nanyang Technological University)

**通讯引用:** 5945 | [OpenAlex ID](https://openalex.org/A5024709593)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本工作提出一种基于链式思考监督微调（CoT‑SFT）与强化学习后训练（RFT）的遥感视觉定位框架，通过显式位置推理和空间奖励提升模型的空间推理与定位精度。

**💡 创新点**

创新点包括①利用合成CoT数据为遥感定位任务显式注入位置推理能力；②设计连续的距离敏感定位奖励，替代传统IoU，提供稠密梯度；③引入基于空间一致性的加权优化，降低同一查询多次推理的空间分散。

**🔧 技术方法**

主要技术包括多模态大型语言模型（Qwen2.5‑VL）、链式思考监督微调、GRPO强化学习、位置奖励机制及空间一致性加权优化。

**📊 数据集**

使用的遥感定位数据集包括DIOR‑RSVG、VRSBench‑VG、RRSIS‑D等；跨域评估使用FAST‑T与SOTA‑T。

**📈 对比分析**

相较于通用VLM和遥感专用VLM，在Acc@0.5上提升约5%，并在跨域测试中保持领先，显示出更强的学习效率和泛化能力。

**⚠️ 局限性**

限制主要体现在对极端尺度或高度相似目标的分散推理仍存在挑战，且对训练数据量仍有一定依赖。

---

## 428. Noise as a Probe: Membership Inference Attacks on Diffusion Models Leveraging Initial Noise

**arXiv ID:** 2601.21628 | [PDF](https://arxiv.org/pdf/2601.21628v1)

**作者:** Puwei Lian `[一作]` (Southeast University), Bingkun Bao `[通讯]` (Nanjing University of Posts and Telecommunications)

**通讯引用:** 2074 | [OpenAlex ID](https://openalex.org/A5007962086)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种利用DDIM反演注入语义信息到初始噪声，从而对微调扩散模型进行成员推断的攻击方法。

**💡 创新点**

创新点在于发现标准噪声调度残留语义信息并被模型学习利用，利用公开的预训练模型即可注入语义噪声，无需访问目标模型参数或中间步骤。

**🔧 技术方法**

使用DDIM反演、语义注入、基于L2/感知距离的判别、交叉注意力分析等技术实现攻击。

**📊 数据集**

在Pokémon、T-to-I、MS-COCO和Flickr四个公开数据集上进行实验。

**📈 对比分析**

与现有的中间结果攻击（SecMI、PIA）和终端攻击（NA-P、Feature-T/C/D、GD）对比，AUC最高达90.46%、TPR@1%达21.8%，在无防御时性能明显优于Feature-C等终端攻击，并在多种防御下仍保持领先。

**⚠️ 局限性**

局限性包括：需要对应的文本提示或能够生成提示的模型；对极强防御（如SS_e_i）仍有一定下降；若目标模型与预训练模型相距太大，攻击效果会减弱。

---

## 429. Training Memory in Deep Neural Networks: Mechanisms, Evidence, and Measurement Gaps

**arXiv ID:** 2601.21624 | [PDF](https://arxiv.org/pdf/2601.21624v1)

**作者:** Vasileios Sevetlidis `[一作]` (Athena Research Center), George Pavlidis `[通讯]` (Athena Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文系统整理了深度学习训练中的“记忆”机制，提出源–寿命–可见性三轴分类，并给出可复现的因果评估流程和报告清单。

**💡 创新点**

创新点在于将优化器状态、采样器顺序、路径依赖、外部缓冲与元状态统一归类，并提出单源干预的可移植原语和基于配对种子、Bootstrap CI 的因果效应估计。

**🔧 技术方法**

使用的技术包括指数滑动平均、动量/Adam 运动、随机重排与替换采样、课程/阶段化增强、BatchNorm 统计、对比学习队列、教师 EMA、FedAvg 服务器动量等；通过分支保留与控制、固定随机种子、记录顺序哈希、状态快照实现可重复性。

**📊 数据集**

实验覆盖多种常见数据集与模型：CIFAR‑10/100 + ResNet‑18/MobileNet、SST‑2/AGNews/IMDb + DistilBERT、SpeechCommands/ESC‑50 + 小型CNN 等，用以验证不同记忆源对函数空间、表示漂移与泛化的影响。

**📈 对比分析**

比较方法采用配对种子 ATE 与 95% Bootstrap CI，评估指标包括准确率、ECE/ NLL、SVCCA/CKA 相似度及函数空间距离；实验显示在相同精度下，优化器状态、采样顺序和队列长短均能显著改变最终函数分布，且早期指标能在一定程度上预测后期性能。

**⚠️ 局限性**

局限性包括：仍需大规模实验验证跨领域迁移性；对非凸路径的理论界限不足；隐式记忆难以完全可见；隐私与安全方面的风险未被系统评估；实验受限于计算资源与数据规模。

---

## 430. Breaking the Overscaling Curse: Thinking Parallelism Before Parallel Thinking

**arXiv ID:** 2601.21619 | [PDF](https://arxiv.org/pdf/2601.21619v1)

**作者:** Yiming Wang `[一作]` (Shanghai Jiao Tong University), Rui Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 24207 | [OpenAlex ID](https://openalex.org/A5028153158)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了并行思维（Parallel Thinking）中出现的“overscaling curse”问题，提出通过先预测每个样本最优并行度来动态分配解码预算的T2方法，显著降低计算成本。

**💡 创新点**

创新点在于：①从系统级与样本级效能的双视角量化并解释overscaling curse；②利用模型内部表征训练层级估计器，提前推断最佳并行度，从而在解码前完成预算分配；③通过简单的线性加权融合多层估计器，进一步提升估计精度。

**🔧 技术方法**

主要技术包括：多路径采样与多数投票的并行思维框架；基于“预算‑准确度”函数的样本级性能分析；单隐藏层MLP估计器与逆方差加权融合；以及统计检验与实验评估。

**📊 数据集**

实验使用六大公开数据集（MATH500、AMC、AIME24、AIME25、GPQA、MMLU‑Pro）以及DeepMath‑103K进行估计器训练，覆盖数学与通用问答领域。

**📈 对比分析**

在与标准并行思维（Std‑PT）以及四个自适应预算方法（AC、ESC、DSC、DeepConf）对比时，T2在内存占用和推理延迟上平均下降约50%+，同时保持或略优于Std‑PT的准确率，且在多模型、多数据集上表现稳定。

**⚠️ 局限性**

局限性包括：仅针对多样本采样与多数投票的并行策略；未覆盖树搜索等更复杂的并行思维；仅适用于封闭式任务；需要访问模型内部状态，限制了对封闭模型的适用性。

---

## 431. Semantic Content Determines Algorithmic Performance

**arXiv ID:** 2601.21618 | [PDF](https://arxiv.org/pdf/2601.21618v1)

**作者:** Martiño Ríos-García `[一作]` (Friedrich Schiller University Jena), Kevin Maik Jablonka `[通讯]` (Friedrich Schiller University Jena)

**通讯引用:** 3047 | [OpenAlex ID](https://openalex.org/A5027355573)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 WhatCounts 基准，评估大型语言模型在计数任务中对语义类的依赖性。

**💡 创新点**

发现计数准确率因被计数对象的语义类差异而波动超过 40%，并且误差随模型性能提升而增大；进一步证明模型并未真正实现算法，而是对语义依赖近似。

**🔧 技术方法**

采用基准设计、结构化对照实验、词元打乱、推理努力调节、微调对比（DPO/PPO）以及基于 Python 执行工具的代理环境进行系统评估。

**📊 数据集**

使用六类数据集（地址、化学品、城市、全名、电话号码、表情符号）生成长度可控、无重复、无干扰项的列表。

**📈 对比分析**

与 Anthropic、Moonshot AI、OpenAI 等前沿模型对比，报告准确率和语义差距（Semantic Gap）；实验显示准确率与语义差距呈正相关，微调会产生不可预测的差距变化，代理环境中的误差也随语义类变化。

**⚠️ 局限性**

限制在于仅测试计数这一基本算子、语义类别有限、列表长度受限，未覆盖更广泛的算子或更复杂的输入结构，可能无法完全推广到所有 LLM 场景。

---

## 432. Flocking behavior for dynamic and complex swarm structures

**arXiv ID:** 2601.21772 | [PDF](https://arxiv.org/pdf/2601.21772v1)

**作者:** Carmen D. R. Pita-Romero `[一作]` (Universidad Politécnica de Madrid), Pascual Campoy `[通讯]` (Universidad Politécnica de Madrid)

**通讯引用:** 5497 | [OpenAlex ID](https://openalex.org/A5001678286)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

本文提出了一种基于虚拟质心的多无人机编队控制算法 FlockingBehavior，能够实时生成和动态重构任意规模、复杂几何形态的编队并跟随单一路径实现编队飞行。

**💡 创新点**

创新点在于：①将编队几何关系抽象为 SE(3) 变换，形成可插拔的虚拟结构；②实现在线增删无人机和形态切换；③只需给定质心轨迹即可推导全体无人机的参考姿态；④算法开放源代码并遵循 ROS2 模块化设计。

**🔧 技术方法**

使用技术包括 ROS2、Aerostack2 框架、Crazyflie 2.1 四旋翼、动态多项式 3D 轨迹生成、SE(3) 变换及基于 Reynolds 规则的协同控制。

**📊 数据集**

实验数据来自本地实验室的 Crazyflie 2.1 机器人以及基于 Aerostack2 的仿真环境；使用的轨迹为直线、曲线和 Z 轴旋转等多种路径，编队规模从 2 只到 12 只不等。

**📈 对比分析**

通过计算聚合度、参考误差、间距、对齐误差等四个指标与理论值对比，实验结果在 5% 以内满足 Reynolds 规则；在仿真与真实环境下性能一致，证明算法可扩展到更大规模且不需要为每架无人机单独生成轨迹。

**⚠️ 局限性**

局限性包括：①假设轨迹可达、控制精确且环境无障碍，缺乏障碍物避让；②目前实现为集中式离线控制，未完成去中心化部署；③对动态环境自适应能力有限，未来计划加入环境感知和形态变形策略。

---

## 433. Amortized Spectral Kernel Discovery via Prior-Data Fitted Network

**arXiv ID:** 2601.21731 | [PDF](https://arxiv.org/pdf/2601.21731v1)

**作者:** Kaustubh Sharma `[一作]` (Indian Institute of Technology Roorkee), Parikshit Pareek `[通讯]` (Indian Institute of Technology Roorkee)

**通讯引用:** 279 | [OpenAlex ID](https://openalex.org/A5055795558)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出了从预训练的 Prior-Data Fitted Network（PFN）中解码谱密度并通过 Bochner 定理构造显式协方差核的框架，实现了零样本谱与核发现。

**💡 创新点**

核心创新在于揭示 PFN 注意力隐式编码频谱信息，并设计滤波器组解码器将其转化为可解释的谱混合模型及对应的平稳核。

**🔧 技术方法**

采用了 PFN（DVA注意力）+ 多查询注意力池化 + 滤波器组解码器 + Bochner 定理 + 理论可识别性分析等技术。

**📊 数据集**

主要使用合成的谱混合 Gaussian Process 样本、RBF/周期性及其组合核等数据集，并在多维加性核场景中进行实验。

**📈 对比分析**

与 PFN、深度核学习（DKL）、随机傅里叶特征（RFF）等迭代基线对比，解码核在预测误差上与 PFN 相当，同时推断时间缩短数百到千倍。

**⚠️ 局限性**

局限在于单样本情形下频谱权重不可辨识、对高维输入的维度信息缺失，以及在数据量大、维度高的情境下的可扩展性不足。

---

## 434. When does predictive inverse dynamics outperform behavior cloning?

**arXiv ID:** 2601.21718 | [PDF](https://arxiv.org/pdf/2601.21718v1)

**作者:** Lukas Schäfer `[一作]` (Microsoft), Sergio Valcarcel Macua `[通讯]` (Microsoft)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了预测逆动力学模型（PIDM）相对于行为克隆（BC）的样本效率优势，并给出了理论解释与实证验证。

**💡 创新点**

创新点：①提出PIDM通过对未来状态的显式建模引入偏差-方差权衡，降低动作预测方差；②推导了状态预测误差对样本效率的阈值条件；③通过理论与实验证明了在小样本、复杂视觉环境中PIDM的优势。

**🔧 技术方法**

技术：偏差-方差分解理论、逆动力学模型与状态预测器的组合、实例化的近似状态预测（最近邻查找）、深度网络（MLP / ViT）训练、行为克隆与PIDM的对比实验。

**📊 数据集**

数据集：2D导航任务的人类演示与A*规划演示；3D游戏《Bleeding Edge》中的人类游戏演示（含视图帧）。

**📈 对比分析**

比较方法：在同一任务下使用不同数量演示（1–50条）训练BC和PIDM，评估完成率/目标达成率；样本效率比η=样本数_{BC}/样本数_{PIDM}。实验结果显示：BC最多需5×演示（平均约3×）才能达到相同性能；在3D游戏中BC需66%更多样本；PIDM在低样本、小数据、复杂视觉场景中均表现更好。

**⚠️ 局限性**

局限性：①需要准确的未来状态预测，状态预测误差会削弱优势；②实验主要基于点估计的策略，未覆盖更复杂的分布式策略；③在极度噪声或非平稳环境下的稳健性尚未充分验证；④依赖离线数据集，未考虑在线采样或交互式改进。

---

## 435. Why Attention Patterns Exist: A Unifying Temporal Perspective Analysis

**arXiv ID:** 2601.21709 | [PDF](https://arxiv.org/pdf/2601.21709v1)

**作者:** Qingyue Yang `[一作]` (University of Science and Technology of China), Bin Li `[通讯]` (University of Science and Technology of China)

**通讯引用:** 82974 | [OpenAlex ID](https://openalex.org/A5100395468)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Temporal Attention Pattern Predictability Analysis (TAPPA) 框架，统一从时间序列角度分析大型语言模型中的注意力模式，并基于该框架提出查询自相似度（q‑similarity）指标，用以指导 KV 缓存压缩和层级剪枝。

**💡 创新点**

创新点包括：
1) 将注意力模式划分为可预测与不可预测两类，证明可预测性与查询在时间轴上的自相似性相关；
2) 对三类可预测模式（重访、序列、季节）给出严格的数学条件，并解释 RoPE 频率如何导致周期性对角线；
3) 将 q‑similarity 作为轻量级代理指标，用于动态分配缓存预算和指导结构化剪枝，显著提升压缩和剪枝效果。

**🔧 技术方法**

主要技术手段：时序视角下的注意力分解、RoPE 旋转频率分析、查询自相似度计算、KV 缓存压缩方法、层级结构化剪枝算法。

**📊 数据集**

实验数据集与模型：Llama‑3.1‑8B、Qwen‑2.5‑7B 等大模型；在 GSM8K、AIGC 等公开数据集上进行验证；KV 缓存压缩与剪枝实验使用同一模型和任务集。

**📈 对比分析**

方法对比：与 CAKE、SnapKV、H2O、DuoAttention 等缓存压缩基线，以及与 ShortGPT 等剪枝基线比较；实验显示基于 q‑similarity 的策略在不同预算/压缩率下均优于对照组，提升推理效率与模型准确率。

**⚠️ 局限性**

限制：
- 主要聚焦查询自相似性对可预测模式的影响，尚未对所有随机模式给出完整解释；
- 实验仅覆盖少数模型与数据集，泛化性待进一步验证；
- 对 RoPE 频率和基数的假设可能不适用于所有 Transformer 变体；
- q‑similarity 的超参数选择仍需经验调优。

---

## 436. Age Aware Content Fetching and Broadcast in a Sensing-as-a-Service System

**arXiv ID:** 2601.21701 | [PDF](https://arxiv.org/pdf/2601.21701v1)

**作者:** Ankita Koley `[一作]` (Indian Institute of Science), V Mahendran `[通讯]` (Indian Institute of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究Sensing‑as‑a‑Service（S2aaS）系统中，如何在最小化获取成本、广播成本与版本年龄成本的约束下，设计最优或近似的内容获取与广播策略。

**💡 创新点**

创新点在于：①将版本年龄（VAoI）视为QoS度量并提出相应的平均成本MDP模型；②推导出阈值型最优策略；③为异构用户情况提出低复杂度的Whittle指数近似算法，并证明其近似优越性。

**🔧 技术方法**

使用的技术主要是平均成本马尔可夫决策过程（MDP）、相对值迭代、阈值分析、Whittle指数方法以及固定点迭代求解。

**📊 数据集**

本文仅在仿真环境下使用人工生成的随机请求/更新过程（Bernoulli过程）来验证算法性能，并未使用真实数据集。

**📈 对比分析**

通过与理论最优策略（仅在可行的情况下）以及贪婪/基线策略比较，实验显示Whittle指数策略在大多数参数设置下与最优策略的平均成本差距不超过10%，并随用户数增大趋近最优。

**⚠️ 局限性**

主要限制包括：仅考虑单一传感器和可靠广播通道；异构用户场景下仍有指数级复杂度；对系统参数（请求率、更新率、成本）假设已知且稳定，未考虑通道失效、功率控制等实际问题。

---

## 437. Toward Culturally Aligned LLMs through Ontology-Guided Multi-Agent Reasoning

**arXiv ID:** 2601.21700 | [PDF](https://arxiv.org/pdf/2601.21700v1)

**作者:** Wonduk Seo `[一作]` (Enhans), Yi Bu `[通讯]` (Peking University)

**通讯引用:** 2686 | [OpenAlex ID](https://openalex.org/A5002577378)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 OG-MAR 框架，利用本体驱动的多代理推理实现文化对齐的 LLM 推理。

**💡 创新点**

通过基于竞争问题构建文化本体、结构化价值摘要与人口特征检索，并在多代理模拟中加入本体一致性与人口相似度判断，显著提升文化一致性与可解释性。

**🔧 技术方法**

本体构建与检索、价值摘要生成、主题/类别识别、嵌入检索、LLM 推理、多代理模拟与判断代理。

**📊 数据集**

世界价值调查（WVS）为检索语料，六个地区社会调查（EVS、GSS、CGSS、ISD、LAPOP、Afrobarometer）作为评测集。

**📈 对比分析**

与零射、角色分配、自一致、辩论、ValuesRAG 等基线对比，四大 LLM（GPT‑4o‑mini、Gemini 2.5、Qwen 2.5、EXAONE 3.5）测试，OG‑MAR 平均准确率提升约 0.03–0.1，尤其在文化偏差大的地区表现突出。

**⚠️ 局限性**

需要较高的 token 预算与人类审核的本体构建，依赖 WVS 数据覆盖度，对极少量或新文化信息的泛化仍有限。

---

## 438. Towards A Sustainable Future for Peer Review in Software Engineering

**arXiv ID:** 2601.21761 | [PDF](https://arxiv.org/pdf/2601.21761v1)

**作者:** Esteban Parra `[一作]` (Belmont University), Polina Iaremchuk `[通讯]` (Belmont University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了构建可持续软件工程同行评审生态的完整蓝图，涵盖可扩展的培训模块、负责的 AI 使用策略以及多层次激励机制；

**💡 创新点**

创新点在于把培训、AI 辅助与激励融为一体，推出可认证的在线培训、AI 辅助的后评审检查、作者-评审互绑以及新人专属评审奖项；

**🔧 技术方法**

主要技术手段包括基于 LLM 的文本审查工具（如 EditLens）、在线学习平台与认证体系、以及 AI 辅助的评审预审与元评审草稿；

**📊 数据集**

文中未使用传统数据集，而是引用主要软件工程会议（ICSE、MSR、ASE 等）近五年的投稿量与评审委员规模统计；

**📈 对比分析**

由于是系统性方法论提议，文中未开展实验比较，主要通过现有会议实践与政策分析论证方案的可行性与预期收益；

**⚠️ 局限性**

局限性包括缺乏实证评估、实施成本与组织协作难度、以及学术评价体系对评审激励的认可不足等问题。

---

## 439. Temporal Guidance for Large Language Models

**arXiv ID:** 2601.21744 | [PDF](https://arxiv.org/pdf/2601.21744v1)

**作者:** Hong-Kai Zheng `[一作]` (Nanjing University of Aeronautics and Astronautics), Piji Li `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 3215 | [OpenAlex ID](https://openalex.org/A5061435467)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出利用多步预测的时间维度进行对比解码（Temporal Guidance），提升大型语言模型的生成质量。

**💡 创新点**

创新点在于：①用时间上的预测差异代替外部模型或层级对比，产生天然的“业余”分布；②设计轻量级的条件多步预测投影器(cMTPP)，避免多头并保持参数高效；③通过知识蒸馏和适度惩罚保证业余分布与专家分布对齐。

**🔧 技术方法**

采用对比解码（Contrastive Decoding）、多步预测（Multi-Token Prediction）、AdaLN调制、LogSumExp、知识蒸馏（KL损失）以及自适应可行性约束等技术。

**📊 数据集**

训练cMTPP使用LM1B数据集；评测使用GSM8K、GSM8K‑Platinum、Math500、HEval、HEval+、MBPP、IFEval、TruthfulQA、Wikitext‑2等数据集。

**📈 对比分析**

与greedy、标准CD、DoLa等方法在Qwen3-1.7B/8B、Llama3.2-3B、MiMo-7B等模型上进行对比；在数学、编码、指令跟随等任务上，TeGu平均提升约3–12%，并在大模型上可达10%+的性能提升，同时内存和延迟开销仅为基线的2–15%。

**⚠️ 局限性**

局限性：对α超参数较敏感，模型规模越小越易受过度约束；仅在具备多步预测或已训练cMTPP的模型上可用；在事实准确性任务（如TruthfulQA）提升有限。

---

## 440. Epistemic Context Learning: Building Trust the Right Way in LLM-Based Multi-Agent Systems

**arXiv ID:** 2601.21742 | [PDF](https://arxiv.org/pdf/2601.21742v1)

**作者:** Ruiwen Zhou `[一作]` (National University of Singapore), Min-Yen Kan `[通讯]` (National University of Singapore)

**通讯引用:** 11308 | [OpenAlex ID](https://openalex.org/A5066305082)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种多代理LLM系统的历史感知引用框架，利用代理的交互历史构建可信度先验，帮助代理在不确定时倾向于可靠同行的回答。

**💡 创新点**

创新点在于将可信度估计与最终推理解耦为两阶段结构，并通过辅助奖励（Peer Recognition Reward）在第一阶段直接监督代理识别最可靠同行，从而显著提升历史依赖的信任模型与推理质量。

**🔧 技术方法**

主要技术包括两阶段结构化推理管线、强化学习（GRPO）与辅助奖励、以及历史信息压缩的可信度概况生成。

**📊 数据集**

使用了MMLU‑Pro、GPQA等知识推理数据集，并在这些数据集上构建自然与对抗性多代理场景，评估模型性能。

**📈 对比分析**

与传统历史无关的聚合基线（AG）以及单代理模型（SA）比较，ECL在多代理设置下表现出显著优势，尤其在对抗场景中提升至近乎完美（如Gemini 3 Pro 100%），在自然场景中也比基线提升10%~30%。

**⚠️ 局限性**

局限性包括对历史可靠性变化的适应不足、对代理间动态信誉变化的缺乏实时更新机制，以及辅助奖励可能导致的奖励剽窃或过度拟合历史模式。

---

## 441. From Global to Granular: Revealing IQA Model Performance via Correlation Surface

**arXiv ID:** 2601.21738 | [PDF](https://arxiv.org/pdf/2601.21738v1)

**作者:** Baoliang Chen `[一作]` (South China Normal University), Weisi Lin `[通讯]` (Nanyang Technological University)

**通讯引用:** 30222 | [OpenAlex ID](https://openalex.org/A5100403129)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于分层加权和分布正则化的Granularity‑Modulated Correlation（GMC）评价框架，用3D相关性曲面细粒度评估IQA模型。

**💡 创新点**

创新点在于：①引入Granularity Modulator对MOS和|ΔMOS|进行高斯加权，实现局部性能捕获；②采用Distribution Regulator进行核密度平滑消除质量分布偏差；③构造连续的相关性曲面并积分得到全局GMC_g，兼具局部可解释性与全局稳定性。

**🔧 技术方法**

使用的技术包括：广义相关系数（GCC）框架、双高斯权重、核密度估计、拉丁方抽样、局部线性核回归、曲面积分。

**📊 数据集**

实验数据集：FR 采用KADID‑10k、PIPAL；NR 采用LIVE‑Challenge、SPAQ（训练集KonIQ‑10k）。

**📈 对比分析**

与传统PLCC/SRCC/KRCC对比，GMC能揭示模型在不同MOS和|ΔMOS|区域的优势；在高质量检索、细粒度优化和模型融合任务中，选择GMC导向模型能获得更高的MOS/SRCC；在分布偏移下，GMC_g的方差显著低于SRCC。

**⚠️ 局限性**

局限性：需要预先估计MOS标准差/分布，参数（σ、采样点数）对结果有一定影响；计算量相对传统标量指标略大；在极端稀疏的MOS区间可能仍受样本不足限制。

---

## 442. Disentangling perception and reasoning for improving data efficiency in learning cloth manipulation without demonstrations

**arXiv ID:** 2601.21713 | [PDF](https://arxiv.org/pdf/2601.21713v1)

**作者:** Donatien Delehelle `[一作]` (University of Genova), Darwin Caldwell `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种利用完整布料状态训练的强化学习方法，并通过跨模态蒸馏实现从仿真到现实的迁移。

**💡 创新点**

创新点在于将感知与推理解耦，使用节点级取放动作、全状态编码和多目标预训练，以及跨模态知识蒸馏实现轻量化、快速训练的布料展开策略。

**🔧 技术方法**

采用基于 Double DQN 的值学习，卷积编码器+解码器网络，辅助目标函数、边界约束和 Polyak 平滑；在真实世界使用 UNet 进行监督蒸馏。

**📊 数据集**

在 SoftGym 软布料展开基准上训练；离线数据集包含 650 万条状态-动作-奖励样本，使用 Blender 生成的仿真状态与图像对进行蒸馏。

**📈 对比分析**

与 MVP、VCD、Deformable Affordance 三个基线比较，IQM 取得 0.913，提升约 21% 以上，模型参数仅 0.98M（比最佳基线小 95%），仿真训练时间约 40 小时；真实世界 100 场实验平均正则化改进 0.523，略低于 Deformable Affordance。

**⚠️ 局限性**

受限于取放动作空间过于简化、仅支持正方形布料、跨模态蒸馏需要手工资源以及无法充分补偿真实物理差异，导致在更复杂任务和真实场景中的泛化受限。

---

## 443. TCAP: Tri-Component Attention Profiling for Unsupervised Backdoor Detection in MLLM Fine-Tuning

**arXiv ID:** 2601.21692 | [PDF](https://arxiv.org/pdf/2601.21692v1)

**作者:** Mingzu Liu `[一作]` (Shandong University), Runmin Cong `[通讯]` (Shandong University)

**通讯引用:** 10057 | [OpenAlex ID](https://openalex.org/A5091558139)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种无监督的三组件注意力分析方法TCAP，用于检测并清理多模态大语言模型中的后门样本。

**💡 创新点**

创新点在于发现并利用注意力分配偏移（attention allocation divergence）作为后门的普适指纹，并通过GMM和EM投票实现无需清洁数据的检测。

**🔧 技术方法**

使用了跨模态注意力提取、Gaussian Mixture Model（GMM）和Expectation‑Maximization（EM）投票聚合等技术。

**📊 数据集**

在五个视觉‑语言基准数据集（ScienceQA、PhD、DocVQA、Recap‑COCO、SEED‑Bench）以及多种后门攻击（BadNet、Blend、SIG、WaNet、FTrojan）上进行实验。

**📈 对比分析**

与Vanilla FT、Random Drop、SampDetox和BYE等基线比较，TCAP在保持接近原始清洁性能的同时将攻击成功率降至接近0%，F1得分超过98%，优于所有对比方法。

**⚠️ 局限性**

局限性在于仍需对模型内部注意力机制的理解，对极端触发器或极低比例污染样本的检测可能受限，且在非常深的模型层数中头的可解释性可能下降。

---

## 444. Zonkey: A Hierarchical Diffusion Language Model with Differentiable Tokenization and Probabilistic Attention

**arXiv ID:** 2601.21768 | [PDF](https://arxiv.org/pdf/2601.21768v1)

**作者:** Alon Rozental `[一作]` `[通讯]`, Alon Rozental

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Zonkey，一种从原始字符到文档级别的全可微分分层扩散模型，能够自适应分词、压缩、去噪并拼接文本。

**💡 创新点**

创新点包括可微分分词器Segment Splitter、概率注意力Probabilistic Attention、混合扩散DDMM、可微分拼接器Stitcher，以及通过端到端训练实现的自适应分词与层级结构的协同优化。

**🔧 技术方法**

使用技术包括Transformer架构、概率注意力机制、扩散模型（融合DDPM与DDIM）、可微分分词与拼接、对抗式压缩、以及多级去噪重建。

**📊 数据集**

实验数据集为英文维基百科语料库，单GPU训练完成。

**📈 对比分析**

与传统基于BPE的自回归模型对比，Zonkey实现了可变长度生成且无需固定词表；定性评估显示能自然生成句子层级结构，尚未给出精确定量指标。

**⚠️ 局限性**

局限性包括目前仅训练到句子层级、单GPU规模、缺乏量化评估、对长文本生成和跨域适应性尚未验证。

---

## 445. Evaluating ChatGPT on Medical Information Extraction Tasks: Performance, Explainability and Beyond

**arXiv ID:** 2601.21767 | [PDF](https://arxiv.org/pdf/2601.21767v1)

**作者:** Wei Zhu `[一作]` (University of Hong Kong), Wei Zhu `[通讯]` (University of Hong Kong)

**通讯引用:** 17976 | [OpenAlex ID](https://openalex.org/A5068308955)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对ChatGPT在六个医学信息抽取任务的性能、可解释性、置信度、可信度和不确定性进行了系统评估。

**💡 创新点**

首次在多维度评估框架下全面分析LLM在MedIE任务中的表现，揭示其过度自信与高不确定性问题。

**🔧 技术方法**

采用OpenAI GPT‑3.5‑turbo API进行无参数推理，结合人工专家标注与自动评估指标。

**📊 数据集**

使用 ShARe‑2013、CADEC、CMeEE‑v2（NER）、CMeIE‑v2（三元组抽取）、CHIP‑CDEE（事件抽取）和 CHIP‑CDN（ICD 编码）等六个基准数据集。

**📈 对比分析**

与 BERT fine‑tune、UIE 及各任务 SOTA 进行严格 F1 对比，ChatGPT 的 F1 约 30%–40%（远低于 70%+），但在可解释性和可信度上超过 80%。

**⚠️ 局限性**

缺乏任务特定微调导致在复杂任务表现差，过度自信与随机采样产生的高不确定性限制了其直接应用。

---

## 446. Zero-Shot Statistical Downscaling via Diffusion Posterior Sampling

**arXiv ID:** 2601.21760 | [PDF](https://arxiv.org/pdf/2601.21760v1)

**作者:** Ruian Tie `[一作]` (Artificial Intelligence Innovation and Incubation Institute), Hao Li `[通讯]` (Artificial Intelligence Innovation and Incubation Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种零样本统计下尺度方法ZSSD，能够在不使用配对训练数据的情况下，将全球气候模型的低分辨率输出下尺度到高分辨率。

**💡 创新点**

创新点在于将物理一致的气候先验与统一坐标引导机制结合，既能保证生成结果满足地形与时序约束，又通过在高分辨率空间计算梯度解决了传统DPS在大尺度比率下的梯度消失问题。

**🔧 技术方法**

主要技术包括条件扩散概率模型、基于地形与周期性时序的条件编码、统一坐标引导（先下采样再上采样投影）以及对数似然梯度引导的后验采样。

**📊 数据集**

使用的主要数据集为ERA5重分析（0.25°、6小时）以及CMIP6五个不同分辨率的全球气候模型（IPSL、AWI、MIROC、MPI-LR、MPI-HR），并在高程数据库GENCO提供地形约束。

**📈 对比分析**

与传统双线性、BCSD、DDR、DPS等方法比较，ZSSD在合成配对任务中实现最小的MAE/RMSE，在真实GCM的零样本任务中在99%分位误差上均显著优于所有基线，并能恢复热带气旋等极端天气细节。

**⚠️ 局限性**

主要局限在于推断速度相对较慢（需要数千步扩散采样），且对高频谱匹配仍存在一定误差，未来需要进一步加速和提升对极端事件的细粒度重建。

---

## 447. Dynamic Topology Awareness: Breaking the Granularity Rigidity in Vision-Language Navigation

**arXiv ID:** 2601.21751 | [PDF](https://arxiv.org/pdf/2601.21751v1)

**作者:** Jiankun Peng `[一作]` (Aerospace Information Research Institute, Chinese Academy of Sciences), Xiaoming Wang `[通讯]` (Aerospace Information Research Institute, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文针对视觉语言导航中的连续环境，提出了DGNav框架，通过动态拓扑感知突破了传统方法的“Granularity Rigidity”问题。

**💡 创新点**

创新点包括：①基于角度分散的Scene‑Aware Adaptive Strategy，可实时调节图阈值实现“按需密化”；②Dynamic Graph Transformer将视觉语义、语言相关性与几何约束融合，动态重构边权，消除拓扑噪声。

**🔧 技术方法**

核心技术：图神经网络（Graph Transformer）、多模态编码、动态阈值控制、角度分散自适应映射与可学习边权融合。

**📊 数据集**

实验数据集：R2R‑CE 与 RxR‑CE（Matterport3D 连续版）进行验证。

**📈 对比分析**

与多种基线（端到端、显式地图等）比较，DGNav在 R2R‑CE 的 SPL、SR、nDTW 等指标均超越 ETPNav 与其他显式地图方法，尤其在复杂环境和长路径任务中表现突出。

**⚠️ 局限性**

局限性：动态阈值采用线性映射，极端简单或极端复杂场景的适应性有限；计算开销略有提升；尚未在真实机器人上验证。

---

## 448. MIDI-LLaMA: An Instruction-Following Multimodal LLM for Symbolic Music Understanding

**arXiv ID:** 2601.21740 | [PDF](https://arxiv.org/pdf/2601.21740v1)

**作者:** Meng Yang `[一作]` (Monash University), Chao Lei `[通讯]` (University of Melbourne)

**通讯引用:** 5398 | [OpenAlex ID](https://openalex.org/A5034157762)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了MIDI-LLaMA，支持符号音乐的指令跟随多模态LLM。

**💡 创新点**

首次将MusicBERT的MIDI编码与Llama-3-8B对齐，并设计了大规模符号音乐-文本数据集。

**🔧 技术方法**

采用MusicBERT作为MIDI编码器，Llama-3-8B作为语言模型，使用投影层对齐并进行两阶段对齐+指令微调。

**📊 数据集**

使用GiantMIDI-Piano并通过GPT-4o自动注释生成约9.8k条音乐-文本对。

**📈 对比分析**

与基于ABC记谱的文本基线对比，MIDI-LLaMA在问答和标题生成上在BLEU/METEOR/ROUGE/BERTScore等指标上均显著提升，人工评测亦更受青睐。

**⚠️ 局限性**

仅限古典钢琴数据，未检验跨曲目或跨语言泛化，且仍需更多多样化曲目与多模态对齐方法。

---

## 449. CE-GOCD: Central Entity-Guided Graph Optimization for Community Detection to Augment LLM Scientific Question Answering

**arXiv ID:** 2601.21733 | [PDF](https://arxiv.org/pdf/2601.21733v1)

**作者:** Jiayin Lan `[一作]`, Guoping Hu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于论文标题中心实体的图优化与社区检测框架（CE‑GOCD），用于增强大型语言模型在多篇科学论文问答中的知识推理。

**💡 创新点**

创新点在于通过标题引导的子图检索、边权重自适应剪枝与隐式关系补全，再利用 Louvain 社区检测聚合论文主题，从而显著提升检索到的知识结构性和回答的连贯性。

**🔧 技术方法**

技术包括知识图谱构建、基于 LLM 的关键词与实体过滤、路径检索、语义相似度与关系类型加权、子图剪枝与补全、Louvain 模块度聚类，以及最终的社区级答案生成。

**📊 数据集**

使用了 NLP‑AKG（61,826 篇 ACL 论文构建的知识图谱）以及三个问答基准：QASPER、PeerQA 与自标注的 NLP‑MQA，此外在医学域也验证了 ChatDoctor5K 构建的疾病知识图谱。

**📈 对比分析**

与 BM25、Embedding 检索、KAG、MindMap、PathRAG 等检索或图增强基线及不同 LLM（GPT‑4、DeepSeek‑V3、Qwen‑Plus）对比，CE‑GOCD 在 F1 分数上平均提升 6–10%，尤其在 NLP‑MQA 上最高提升 8.95%。

**⚠️ 局限性**

局限性包括对不同领域知识结构的适应性仍有限（医学域效果略逊于 MindMap），以及在子图检索与社区聚类过程中对超参数和计算成本的依赖，需进一步优化以降低延迟。

---

## 450. Procedural Pretraining: Warming Up Language Models with Abstract Data

**arXiv ID:** 2601.21725 | [PDF](https://arxiv.org/pdf/2601.21725v1)

**作者:** Liangze Jiang `[一作]` (EPFL), Damien Teney `[通讯]` (Idiap Research Institute)

**通讯引用:** 7068 | [OpenAlex ID](https://openalex.org/A5067549788)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了一种在语言模型预训练初期使用抽象程序化数据（如形式语言、堆栈模拟、细胞自动机等）的轻量级预训练策略，显著提升后续在自然语言、代码和非正式数学等语义数据上的学习效率与性能

**💡 创新点**

创新点在于将程序化数据视为对语言模型的“认知预热”，证明其能补充传统语义预训练、加速学习、降低所需语义数据量，并揭示不同网络组件（注意力层与MLP层）对不同领域的专属贡献

**🔧 技术方法**

技术上采用 GPT‑2‑type 仅解码器 Transformer，分两阶段训练：先对程序化数据进行 next‑token 预训练，再在标准语义数据上继续预训练；并使用层级选择性迁移、权重混合和数据混合等方法进行探索

**📊 数据集**

数据集包括程序化合成数据（Dyck 语言、ECA Rule 110、栈、集合/排序/删除等）、语义数据（WikiText、JavaCorpus、C4、CodeParrot、DeepMind‑Math 等）以及标准评测任务（Haystack、加法/乘法/排序等）

**📈 对比分析**

与不使用程序化预训练的基线相比，在 0.1–0.3% 程序化 token 的 additive 方案即可提升 10–20% 的精度；在 substitute 方案中仅用 2–4M 程序化 token 可替代 45–86% 的语义 token，保持相同 perplexity；多模型/权重组合进一步提升跨任务性能

**⚠️ 局限性**

主要局限在于实验规模相对较小（最多 1.3B 参数）且对多类型程序化数据的组合与优化仍处于初步探索阶段，缺乏对更大规模 LLM 与更复杂任务的验证

---

## 451. SmartMeterFM: Unifying Smart Meter Data Generative Tasks Using Flow Matching Models

**arXiv ID:** 2601.21706 | [PDF](https://arxiv.org/pdf/2601.21706v1)

**作者:** Nan Lin `[一作]` (Delft University of Technology), Pedro P. Vergara `[通讯]` (Delft University of Technology)

**通讯引用:** 1203 | [OpenAlex ID](https://openalex.org/A5070971243)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出了一种基于流匹配（Flow Matching）的统一模型，能够在不重新训练的前提下完成智能电表数据的生成、缺失值插补与超分辨率恢复等多种生成任务。

**💡 创新点**

创新点在于：①将流匹配与推理时引导（inference‑time guidance）相结合，使得单一模型即可处理多种条件生成任务；②设计了适用于高维时间序列的 Transformer 速度网络，并通过对齐位置编码、有效步长掩码与多模态注意力实现对不同长度、不同条件的兼容；③提出了基于投影的引导机制，能够在生成过程中强制满足峰值、总量或观测约束。

**🔧 技术方法**

核心技术包括：流匹配（Flow Matching）与连续正规化流（CNF）训练、Transformer 变换器网络、对齐位置编码、有效步长掩码、条件自适应归一化（AdaNorm）与多模态注意力、推理时投影引导（projection‑based guidance）。

**📊 数据集**

使用来自荷兰电力系统运营商（DSO）的约 350,000 条 15 分钟分辨率、28–31 天的月度智能电表数据，按 70/15/15 分割为训练、验证与测试集；数据按四类业务类型（E3A、E3B、E3C、PV）划分。

**📈 对比分析**

实验将模型与 VAE、WGAN、缩放式方法、Masked Autoencoder、LoadPIN、ProfileSR 等基线进行对比：①生成任务中，MMD 置换检验的 p 值均大于 0.05，显著优于 VAE/GAN；②峰值约束生成中 CRPS 与 RMSE 均低于对比模型；③插补任务中 CRPS 均比 LoadPIN 低约一半；④超分辨率任务中 CRPS 与峰值误差（PLE）均低于 ProfileSR，表现最优。

**⚠️ 局限性**

局限性包括：①需要预先设计并实现对不同条件的投影或梯度引导，任务特定的函数关系 y=f(x) 需手工给出；②模型训练成本仍相对较高，且在超大规模数据或更高分辨率时需要进一步验证；③目前仅在荷兰 15 分钟分辨率月度数据上评估，跨域、跨时间尺度的泛化能力尚未充分验证；④在极端峰值或稀疏观测场景下，推理时引导的收敛性与稳定性需进一步研究。

---

## 452. AtPatch: Debugging Transformers via Hot-Fixing Over-Attention

**arXiv ID:** 2601.21695 | [PDF](https://arxiv.org/pdf/2601.21695v1)

**作者:** Shihao Weng `[一作]` (Nanjing University), Jia Liu `[通讯]` (Nanjing University)

**通讯引用:** 24235 | [OpenAlex ID](https://openalex.org/A5100409757)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于热修复的Transformer模型调试方法，在推理时动态检测并重新分配异常注意力列，以消除后门攻击和模型不公平性。

**💡 创新点**

创新点包括：① 将注意力图视为可修复的运行时状态；② 通过对比学习训练的检测器精准定位异常注意力列；③ 使用统一的benign注意力列并重新缩放实现热补丁，避免对模型参数的改动；④ 同时兼顾后门和偏差两类缺陷。

**🔧 技术方法**

技术手段主要有：delta debugging、hot patching、对比学习、注意力提取与归一化、统一benign注意力列的生成与重分配、Transformer模型的多头注意力处理。

**📊 数据集**

实验使用了六个常见基准数据集（MNIST、Fashion‑MNIST、CIFAR‑10、Census、COMPAS、Bank）以及对应的六种Transformer架构（ViT、Swin、T2T‑ViT、TabTransformer、TaBERT、FTTransformer）。

**📈 对比分析**

与四种现有方法（IDNN、CARE、ADF、Fine‑Pruning）对比，平均后门攻击成功率从≈95% 降至 0.46%（接近 0%），模型准确率仅下降 0.1%；公平性指标 UF 降至 0.04%，几乎不损失准确率；在线修复时每样本额外延迟约 0.8 ms，远低于基线需要的长时间重训练或参数剪枝。

**⚠️ 局限性**

局限性包括：需预先构建调试集并训练检测器；对新的后门触发器或属性变化时可能失效，需要重新更新；在极端输入分布或攻击演化场景下鲁棒性有限；主要针对基于注意力的Transformer，对其他网络结构适用性尚未充分验证。

---

## 453. ChartE$^{3}$: A Comprehensive Benchmark for End-to-End Chart Editing

**arXiv ID:** 2601.21694 | [PDF](https://arxiv.org/pdf/2601.21694v1)

**作者:** Shuo Li `[一作]` (Fudan University), Xuanjing Huang `[通讯]` (Fudan University)

**通讯引用:** 16551 | [OpenAlex ID](https://openalex.org/A5088834359)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ChartE³基准，专门评估多模态模型在不依赖代码中间表示的情况下直接将输入图表转化为编辑后的图像的能力。

**💡 创新点**

创新点在于从传统的代码媒介编辑转向真正的端到端图像编辑评估；构建了包含本地与全局编辑两大类、12种细粒度任务、800+图表样本的高质量数据集；同时引入了客观视觉指标与GPT主导的主观评估相结合的混合评价框架。

**🔧 技术方法**

采用多模态大型语言模型（如GPT-4o、Qwen-Image-Edit、BAGEL等）进行图像编辑；使用CLIP、DINO、LPIPS、SSIM、PSNR等视觉相似度指标；通过GPT‑5.1对编辑正确性与一致性进行打分。

**📊 数据集**

数据来源于ChartBench、Chart2Code、ChartX、ChartM³等公开数据集，并通过人工审核筛选后生成的1,200+条编辑样本。

**📈 对比分析**

在统一的端到端图表编辑设置下对比了多款闭源与开源模型，结果显示闭源模型（Nano Banana、GPT‑Image‑1.5）在所有客观与主观指标上均明显优于开源模型；尤其在全局编辑（如数据过滤、排序）任务中性能差距更大。

**⚠️ 局限性**

主要局限在于现有模型对数据语义与结构变更的处理仍不够成熟，尤其是全局编辑任务容易产生错误；同时评估仍依赖GPT主观打分，缺乏足够的人工标注覆盖。

---

## 454. LoRA and Privacy: When Random Projections Help (and When They Don't)

**arXiv ID:** 2601.21719 | [PDF](https://arxiv.org/pdf/2601.21719v1)

**作者:** Yaxi Hu `[一作]` (Max Planck Institute for Intelligent Systems), Amartya Sanyal `[通讯]` (University of Copenhagen)

**通讯引用:** 298 | [OpenAlex ID](https://openalex.org/A5035879433)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出一种Wishart投影机制S↦M f(S)，并在向量查询下实现无噪声差分隐私；在矩阵查询下证明噪声自由时不具隐私，并展示近乎完美的成员推断攻击。

**💡 创新点**

关键创新在于利用Wishart分布产生的随机低秩投影本身即可实现隐私，且通过加入噪声能够获得隐私放大，优于单纯加噪声方案；并将LoRA更新映射到该机制，揭示LoRA并非天然私有。

**🔧 技术方法**

采用Wishart分布随机矩阵、低秩投影、差分隐私分析、成员推断攻击实验以及隐私放大定理。

**📊 数据集**

文中未给出具体使用的数据集，实验仅以模拟/基准数据验证理论。

**📈 对比分析**

与传统的加噪声机制相比，噪声投影机制在大秩与小秩两种情形下均实现了更强的隐私保证（隐私预算更低），同时在相同噪声水平下可获得更高的精度。

**⚠️ 局限性**

局限性包括：噪声自由的矩阵查询机制不具备DP，需依赖噪声或低秩投影；隐私放大效果需严格计算，缺乏通用的自动化隐私评估工具；在实际模型更新中对LoRA的安全性仍需进一步评估。

---

## 455. Influence Guided Sampling for Domain Adaptation of Text Retrievers

**arXiv ID:** 2601.21759 | [PDF](https://arxiv.org/pdf/2601.21759v1)

**作者:** Meet Doshi `[一作]` (IBM Research AI), Jaydeep Sen `[通讯]` (IBM Research AI)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于影响力估计的强化学习动态采样框架，用来在多源训练数据中自适应地调整采样概率，从而提升检索模型在目标域上的性能。

**💡 创新点**

创新点在于：① 用在线代理模型精确计算每个训练域的影响力得分作为奖励，避免梯度噪声；② 采用 Reptile 方式重用梯度，显著降低 GPU 开销；③ 通过软max 采样策略支持大规模多域数据的动态加权。

**🔧 技术方法**

技术手段包括：强化学习（REINFORCE）+ 影响力估计 + Reptile meta‑update + 软max 采样 + 代理模型计算。

**📊 数据集**

实验数据集：BEIR、MLDR、Sentence‑Transformers 训练集（32 个多域语料），以及对应的 dev/test 集。

**📈 对比分析**

与静态采样、DoReMi、MultiDDS、DoGE、CRISP 等基线对比， 在 NDCG@10 上分别实现 5.03 分点（MLDR）和 0.94 分点（Sentence‑Transformers）提升，GPU 计算成本比梯度基方法低 1.5–4 倍，采样轨迹更稳定。

**⚠️ 局限性**

局限性：需要可用的 dev 集；初始化对最终结果有一定影响；收敛到全局最优不保证；虽然比梯度基方法更轻量，但影响力计算仍带来额外的时间开销。

---

## 456. EWSJF: An Adaptive Scheduler with Hybrid Partitioning for Mixed-Workload LLM Inference

**arXiv ID:** 2601.21758 | [PDF](https://arxiv.org/pdf/2601.21758v1)

**作者:** Bronislav Sidik `[一作]` (Toga Networks), Joseph Kampeas `[通讯]` (Toga Networks)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种面向混合工作负载的 LLM 请求调度器 EWSJF，能够在同时处理短延迟交互查询和长批量请求时，实现更高的吞吐量与更低的尾部延迟。

**💡 创新点**

创新点在于：①通过无监督的 Refine-and-Prune 算法自动划分性能均匀的请求队列；②采用密度加权的可学习优先级评分函数，兼顾紧急性、公平性与吞吐；③结合贝叶斯元优化器在线调优分数与划分参数，实现自适应调度。

**🔧 技术方法**

核心技术包括无监督聚类（Refine-and-Prune）、基于队列密度的评分函数（Density‑Weighted Scoring）、贝叶斯元优化（Bayesian Meta‑Optimization）以及与 vLLM 的插件化集成。

**📊 数据集**

使用公开的对话式数据集（短交互查询）与长文本摘要/大上下文基准（长批量请求）混合生成的合成工作负载，长度分布在 32–4096 tokens。

**📈 对比分析**

与 vLLM 默认的 FCFS 调度以及贪心 SJF 进行对比。实验在 4×A100 GPU 上运行，EWSJF 在混合工作负载下吞吐量提升 30% 以上，短请求的首词延迟平均缩短至 FCFS 的 1/4，且在高并发时能保持 50% 以上的加速。

**⚠️ 局限性**

局限性包括：①仅基于请求长度做聚类，忽略语义或模型级别的计算差异；②贝叶斯优化更新周期为分钟级，无法应对极端突发或攻击性流量；③缺乏正式的公平性或最优性理论保证；④仅在单节点环境验证，分布式扩展需进一步研究。

---

## 457. Subjective Distortion: Achievability and Outer Bounds for Distortion Functions with Memory

**arXiv ID:** 2601.21757 | [PDF](https://arxiv.org/pdf/2601.21757v1)

**作者:** Hamidreza Abin `[一作]` (Chinese University of Hong Kong), Andrew W. Eckford `[通讯]` (York University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并分析了在信息压缩过程中，失真函数依赖前一次输出的“主观失真”问题，给出了内外界限并讨论了凸化方法。

**💡 创新点**

创新点在于将传统的无记忆失真函数推广到有记忆失真，构造了可行的单字母率失真上界与下界，并给出了可计算的凸化内外界限。

**🔧 技术方法**

使用了信息率失真理论、马尔可夫链核、信息谱方法、凸优化以及高斯源的最优性分析等技术。

**📊 数据集**

没有使用真实数据集，文中仅给出了二元伯努利源和高斯源的数值示例（Hamming 失真 + 生成成本）。

**📈 对比分析**

通过比较内外界限曲线 R₁(D) 与 R₂(D) 展示了理论上可达的速率失真范围；在示例中外界限不随生成成本 c 变化，而内界限随 c 变化，说明该框架可评估不同成本对速率的影响。

**⚠️ 局限性**

局限在于多字母特征仍难以闭式求解、内界限不一定可达、非凸性导致优化困难、未给出具体编码/译码实现方案。

---

## 458. Migrating Esope to Fortran 2008 using model transformations

**arXiv ID:** 2601.21755 | [PDF](https://arxiv.org/pdf/2601.21755v1)

**作者:** Younoussa Sow `[一作]` (Framatome), Stéphane Ducasse `[通讯]` (Lille)

**通讯引用:** 10044 | [OpenAlex ID](https://openalex.org/A5031290426)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

自动化迁移含专有扩展的Esopo Fortran 代码到现代 Fortran 2008，并生成可读性高、可维护的目标代码。

**💡 创新点**

将访客模式与字符串模板混合的 M2T 技术用于大型命令代码生成，并实现了双向间接内存模型，保持原有行为。

**🔧 技术方法**

基于 MDE，使用 Famix/FAST 元模型、Pharo 字符串节点、访客遍历 AST、字符串模板生成代码。

**📊 数据集**

使用 Framatome 内部的“书店”案例（23 个文件、1611 行代码）作为测试集。

**📈 对比分析**

通过与原始 Esopo、优化后的 Esopo、Fortran 2008 及其无检查版进行执行时间对比，迁移版速度提升约 3 倍，内存占用下降。

**⚠️ 局限性**

限制：未处理 GOTO、COMMON 等旧构造，注释映射不完整，DSL/轻量级定义缺失；在大型真实项目中尚未验证。

---

## 459. DreamActor-M2: Universal Character Image Animation via Spatiotemporal In-Context Learning

**arXiv ID:** 2601.21716 | [PDF](https://arxiv.org/pdf/2601.21716v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 460. Why Adam Works Better with $β_1 = β_2$: The Missing Gradient Scale Invariance Principle

**arXiv ID:** 2601.21739 | [PDF](https://arxiv.org/pdf/2601.21739v1)

**作者:** Alberto Fernández-Hernández `[一作]` (Universitat Politècnica de València), Enrique S. Quintana-Ortí `[通讯]` (Universitat Politècnica de València)

**通讯引用:** 6783 | [OpenAlex ID](https://openalex.org/A5012806004)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究 Adam 优化器中动量参数 β₁ 与 β₂ 取值相等时的行为，并提出梯度尺度不变性（gradient scale invariance）的理论框架；

**💡 创新点**

发现 Adam 的更新在 β₁=β₂ 时实现了梯度尺度的一阶不变性，从而解释了实验中该设置导致更稳定、更高性能的现象；

**🔧 技术方法**

利用 Adam 的连续时间极限（ODE 解析）与对数尺度漂移（log‑scale drift）展开，证明当 τ₁=τ₂（即 β₁=β₂）时一阶尺度项消失；

**📊 数据集**

在多种视觉与语言任务上验证：NanoGPT（SlimPajama、WikiText）、EfficientNet‑B0（TinyImageNet）、ResNet18、ViT‑B16（CIFAR‑100）、T5（SQuAD）等；

**📈 对比分析**

通过比较不同 β₁、β₂ 组合下的更新幅度振荡（ω）以及训练损失，实验表明 β₁=β₂ 时更新更平滑、振荡更小，统计检验显著优于其他设置；

**⚠️ 局限性**

研究仅聚焦一阶尺度不变性，未探讨更高阶不变性；对 β₁=β₂ 时最佳 β 值的理论指导仍缺失；

---

## 461. DropoutTS: Sample-Adaptive Dropout for Robust Time Series Forecasting

**arXiv ID:** 2601.21726 | [PDF](https://arxiv.org/pdf/2601.21726v1)

**作者:** Siru Zhong `[一作]` (Hong Kong University of Science and Technology), Yuxuan Liang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5513 | [OpenAlex ID](https://openalex.org/A5018828723)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5a41884c-404f-4688-a89c-aa238c10fe68` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文提出了DropoutTS，一种通用插件，通过基于谱稀疏性的样本自适应 dropout 动态调节模型容量，从而提升时间序列预测的鲁棒性。

**💡 创新点**

创新点在于提出“容量‑中心调制”范式，利用谱残差量化噪声并映射为可微的 dropout 率，实现对样本难度的连续、可学习调节；且该方法与数据清洗或先验建模无关，可无缝叠加于现有模型。

**🔧 技术方法**

核心技术包括全局线性去趋势、对数尺度谱归一化、基于谱平坦度的软阈值滤波器、残差噪声评分、按噪声映射的动态 dropout 率、以及利用 Straight‑Through Estimator 允许梯度流动的可微掩码。

**📊 数据集**

在七个公开真实数据集（ETTh1/2、ETTm1/2、Electricity、Weather、ILI）以及自构造的 Synth‑12 合成基准上进行实验。

**📈 对比分析**

与六种 SOTA 结构（Informer、PatchTST、Crossformer、TimesNet、TimeMixer、iTransformer）对比，DropoutTS 在所有噪声水平下平均提升约 9.8% 的 MSE，Informer 在最噪声场景下提升高达 46%，且对强大模型仅产生 0.7–2.8% 的进一步改进。

**⚠️ 局限性**

局限性包括：对 FFT 计算的轻微依赖，导致轻量模型训练时略增延迟；需要在不同数据集上调优 γ 参数；仅针对频谱噪声有效，对非频谱形式的异常或长期依赖可能作用有限。

---

## 462. Enhancing Language Models for Robust Greenwashing Detection

**arXiv ID:** 2601.21722 | [PDF](https://arxiv.org/pdf/2601.21722v1)

**作者:** Neil Heinrich Braun `[一作]` (National University of Singapore), Gianmarco Mengaldo `[通讯]` (National University of Singapore)

**通讯引用:** 2211 | [OpenAlex ID](https://openalex.org/A5091468612)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种参数高效的框架，利用对比学习与序数排序结构化LLM的潜在空间，以更稳健地识别可执行的ESG声明；

**💡 创新点**

创新点在于将对比学习、序数排名损失、门控特征调制与MetaGradNorm相结合，形成多目标自适应训练体系；

**🔧 技术方法**

核心技术包括LoRA适配器、对比损失、序数平均间隔损失、样本级门控权重和MetaGradNorm梯度平衡；

**📊 数据集**

实验基于A3CG数据集（aspect–action三阶序数标签）进行跨类别泛化评估；

**📈 对比分析**

与标准LoRA、T5等基线对比，模型在见/未见类别上的F1均提升4–5分，且小型7–8B模型在未见类别可与GPT‑4o等大型模型相当；

**⚠️ 局限性**

局限在于仅使用英文A3CG、对超参数高度敏感、税onomy有限且仅验证PEFT策略，未探索其他数据增强或目标组合。

---

## 463. CoFreeVLA: Collision-Free Dual-Arm Manipulation via Vision-Language-Action Model and Risk Estimation

**arXiv ID:** 2601.21712 | [PDF](https://arxiv.org/pdf/2601.21712v1)

**作者:** Xuanran Zhai `[一作]` (National University of Singapore), Yaohua Liu `[通讯]` (Guangdong Institute of Intelligence Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了CoFreeVLA，一种在双臂视觉‑语言‑动作（VLA）框架中加入短期自碰撞风险评估器的系统，能够在执行过程中预判并阻止危险动作、自动恢复安全姿态并在训练中进行安全强化；

**💡 创新点**

其创新点在于将基于跨注意力的自碰撞风险估计器与VLA策略紧密耦合，实现风险门控、恢复与策略微调三位一体，并采用两阶段预训练+真实机器人微调的方式提升风险预测精度；

**🔧 技术方法**

技术上使用了VLA端到端策略、跨模态注意力网络作为风险评估器、温度校准、控制壁垒函数、梯度优化恢复序列以及风险加权的策略微调；

**📊 数据集**

数据集方面，先在模拟与真实环境下通过模型检查器采集大量人工标注的碰撞与距离数据，随后在PiPER双臂机器人上收集真实执行数据完成微调；

**📈 对比分析**

在五个双臂任务上与RDT‑1B和APEX基线比较，CoFreeVLA在跨臂干扰任务中显著降低了自碰撞率（从8/10降至2/10），成功率保持相近，且风险估计与门控的运行延迟低于5 ms；

**⚠️ 局限性**

局限性包括阈值设置过于保守导致精细配合任务出现误判、恢复过程偶尔出现振荡以及对距离阈值与规划时长的进一步调优仍需改进。

---

## 464. TACLer: Tailored Curriculum Reinforcement Learning for Efficient Reasoning

**arXiv ID:** 2601.21711 | [PDF](https://arxiv.org/pdf/2601.21711v1)

**作者:** Huiyuan Lai `[一作]` (University of Groningen), Malvina Nissim `[通讯]` (University of Groningen)

**通讯引用:** 2825 | [OpenAlex ID](https://openalex.org/A5040564747)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 TACLer 框架，结合模型定制的课程学习和混合 Thinking/NoThinking 推理模式来提升 LLM 的学习与推理效率。

**💡 创新点**

创新点是根据模型自身能力动态调节难度的课程学习和同时训练两种推理模式实现准确率与计算量的双重提升。

**🔧 技术方法**

使用了强化学习（GRPO）与自回归 LLM（DeepSeek‑R1‑Distill‑Qwen‑1.5B）进行训练。

**📊 数据集**

采用了 DeepScaleR 数据集以及四个数学推理基准 MATH500、AMC、AIME 2024/2025。

**📈 对比分析**

与 DeepScaleR、FastCuRL、OverThink 等基线对比，TACLer 在推理准确率上提升约 9–11%，同时 token 量降低 42% 以上。

**⚠️ 局限性**

局限在推理时需多次迭代，推理开销相对增大。

---

## 465. FBS: Modeling Native Parallel Reading inside a Transformer

**arXiv ID:** 2601.21708 | [PDF](https://arxiv.org/pdf/2601.21708v1)

**作者:** Tongxi Wang `[一作]` `[通讯]` (Southeast University), Tongxi Wang (Southeast University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在Transformer层中加入可训练的预览、分块和跳过机制，构建Fovea-Block‑Skip Transformer (FBS)；

**💡 创新点**

创新点在于将人类阅读的预览–分块–略读思路内化为可训练的循环结构，三模块（PAW、CH、SG）协同实现自适应的层级跳过；

**🔧 技术方法**

使用Parafovea‑Attention Window (PAW)实现动态前瞻，Chunk‑Head (CH)提供分块语义通道，Skip‑Gate (SG)控制层级跳过，并在Stage‑2通过PPO对SG进行强化学习；

**📊 数据集**

训练数据包括RedPajama‑V2、Yuan‑2.0 Corpus（中英混合）和OSCAR‑zh，实验评测使用MMLU、CMMLU、C‑Eval、BBH、GSM8K、CMath、HumanEval‑X、MBPP等多任务基准；

**📈 对比分析**

与基线Qwen3‑4B‑Instruct以及同模型的SpecDec、Medusa、EAGLE‑2、Lookahead等方法对比，FBS在保持参数不变的前提下，质量略有提升（PPL↓、各评测指标↑），但能把推理延迟从760 ms降至532 ms，TFLOPs降至0.70，层跳比例约36%；

**⚠️ 局限性**

局限包括：在极长文本或复杂推理任务中可能出现一致性与推理质量下降；缺乏多语言及低资源语言的广泛验证；跳过策略依赖概率代理，易受分布漂移影响；硬件/实现依赖导致实际加速差异；

---

## 466. Can David Beat Goliath? On Multi-Hop Reasoning with Resource-Constrained Agents

**arXiv ID:** 2601.21699 | [PDF](https://arxiv.org/pdf/2601.21699v1)

**作者:** Hojae Han `[一作]` (Electronics and Telecommunications Research Institute), Seung-won Hwang `[通讯]` (Seoul National University)

**通讯引用:** 1622 | [OpenAlex ID](https://openalex.org/A5101567750)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了预算高效的 RL 框架（David‑GRPO），使小型语言模型在资源受限环境下完成多跳推理。

**💡 创新点**

创新点包括：①少量示例 warm‑start 与离线专家轨迹混合训练；②基于证据回忆的 grounded retrieval reward；③对近乎失败轨迹的 grounded expansion 重采样，提升探索效率。

**🔧 技术方法**

技术手段：GRPO + KL 限制；混合离线/在线策略；few‑shot warm‑start；dense grounded reward；轨迹重采样；将多跳推理建模为 MDP。

**📊 数据集**

数据集：HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle、BamTwoogle、AntiLeakBench‑Multi‑hop。

**📈 对比分析**

与 Tree‑GRPO、Search‑R1、StepSearch、AutoCoA 等基线在 4×RTX 3090、低计算预算下对比，David‑GRPO 用 4.7% 预算实现或超越高预算基线，在 6 个基准的 EM/F1 上表现最佳。

**⚠️ 局限性**

局限性：仍需要少量专家轨迹；在极低预算或极大搜索空间下探索仍可能受限；未针对更长篇多跳任务或多模态工具使用进行评估。

---

## 467. Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics

**arXiv ID:** 2601.21698 | [PDF](https://arxiv.org/pdf/2601.21698v1)

**作者:** Mohamed Elgaar `[一作]` (University of Massachusetts Lowell), Hadi Amiri `[通讯]` (University of Massachusetts Lowell)

**通讯引用:** 615 | [OpenAlex ID](https://openalex.org/A5074007015)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在单轮预训练中不同数据排序对大语言模型学习动态与稳定性的影响，利用Pythia模型在三种语言学驱动的课程学习（年龄获取、词频、动词多样性）与随机排序进行对比；

**💡 创新点**

创新点在于提出课程学习作为通过控制梯度方差来提升优化稳定性的机制，并用隐藏马尔可夫模型揭示所有排序共享相同的学习阶段；

**🔧 技术方法**

技术包括隐藏马尔可夫模型（HMM）阶段划分、梯度噪声尺度（GNS）与奇异熵诊断，以及对语言模型头的谱分析；

**📊 数据集**

使用的数据集为The Pile，固定长度2048-token样本，实验覆盖14M–1B参数的Pythia模型；

**📈 对比分析**

通过在多项下游基准（ARC-E/C, PIQA, SciQ, LogiQA, Lambada, WinoGrande, WSC）和BLiMP探针上的精度比较，发现课程学习在容量受限模型（≤160M）显著降低GNS与奇异熵，提升最终准确率，而在更大模型上优势减弱；

**⚠️ 局限性**

局限包括：仅在固定样本集上评估，未考虑多阶段课程与更大模型的迁移性，且对不同语言现象的课程影响仍不完全明确。

---

## 468. Understanding Model Merging: A Unified Generalization Framework for Heterogeneous Experts

**arXiv ID:** 2601.21690 | [PDF](https://arxiv.org/pdf/2601.21690v1)

**作者:** Qinglun Li `[一作]` (National University of Defense Technology), Li Shen `[通讯]` (Sun Yat-sen University)

**通讯引用:** 15518 | [OpenAlex ID](https://openalex.org/A5100768717)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于L2-稳定性的统一理论框架，解释模型融合方法的有效性，并给出针对超参数的实践指导。

**💡 创新点**

首次给出多任务、异质超参数下模型融合的泛化误差上界，统一解释多种融合算法，并基于此提出可操作的超参数调整策略。

**🔧 技术方法**

L2-稳定性理论、L‑smooth 与方差/异质性假设、SGD/Adam 分析、经验风险与泛化误差分解。

**📊 数据集**

20 个视觉分类任务（如 CIFAR‑10、SVHN、Stanford Cars、Food‑101、SUN397、EuroSAT 等）以及 ResNet/Vit 预训练模型。

**📈 对比分析**

与简单平均、TIES、DARE、AdaMerging 等方法在 ResNet‑18/50/152 和 ViT 上进行实验，结果与理论预测一致，超参数调整后融合模型性能优于传统方法。

**⚠️ 局限性**

依赖光滑及方差/异质性假设，未覆盖非光滑损失；理论对大规模任务数的预测仍需验证；只给出上界，缺乏下界分析。

---

## 469. OneMall: One Model, More Scenarios -- End-to-End Generative Recommender Family at Kuaishou E-Commerce

**arXiv ID:** 2601.21770 | [PDF](https://arxiv.org/pdf/2601.21770v1)

**作者:** Kun Zhang `[一作]` (Kuaishou Technology), Kun Gai `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并部署了 OneMall，一个统一的端到端生成式推荐系统，覆盖商品卡、短视频和直播三种电商场景。

**💡 创新点**

创新点包括：①统一的语义分词器兼顾商业与观看语义；②基于 Transformer 的 Query‑Former、Cross‑Attention 与 Sparse MoE 的自回归生成架构；③将排序模型奖励通过强化学习嵌入检索流程，实现检索‑排序一体化。

**🔧 技术方法**

使用的技术包括 LLM 预训练/微调、Res‑Kmeans+FSQ 语义分词、Query‑Former 长序列压缩、Cross Transformer、Sparse MoE Decoder、RL（DPO/GRPO）奖励学习、对比学习辅助。

**📊 数据集**

主要数据集来自 Kuaishou 电商业务日志，包含商品卡、短视频、直播的交互记录和 Item2Item 关系，规模达数千万条。

**📈 对比分析**

通过离线 SID Accuracy、仿真 Hit Rate 与在线 A/B（曝光、CTR、订单、GMV）三种方式与 SASRec、TIGER 等基线对比，OneMall 在商品卡实现 +13.01% GMV、短视频 +15.32% 订单、直播 +2.78% 订单，整体表现显著提升。

**⚠️ 局限性**

局限性包括：对实时商品更新和多目标优化的处理仍有限；分词冲突需要额外解决；模型规模大导致推理成本较高；尚未实现检索与排序完全统一及文本推理能力。

---

## 470. Language-based Trial and Error Falls Behind in the Era of Experience

**arXiv ID:** 2601.21754 | [PDF](https://arxiv.org/pdf/2601.21754v1)

**作者:** Haoyu Wang `[一作]` (Nanyang Technological University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 98192 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SCOUT框架，利用轻量级小模型进行任务探索，随后将探索经验蒸馏给LLM并进行多轮强化学习

**💡 创新点**

关键创新在于把探索与利用解耦，采用小型网络“scout”完成高效探索，再将其轨迹文本化为LLM可学习的对话形式，并通过多轮PPO进一步激活和细化知识

**🔧 技术方法**

技术包括RL（DQN、PPO）、监督微调（SFT）、多轮PPO、文本化转换器、LLM（Qwen2.5系列、LLaMA3.1等）

**📊 数据集**

实验使用六类无监督任务：Bandit、FrozenLake、Sokoban、2048、Rubik’s Cube、Sudoku，数据来自公开的RAGEN benchmark和自定义符号/空间环境

**📈 对比分析**

与RAGEN、State Estimation RL、SPA以及多款商业模型（GPT‑4o‑mini、DeepSeek‑V3、Gemini‑2.5‑Pro等）对比，SCOUT在所有任务上均优于基线，3B LLM平均得分0.86，GPU消耗减少约60%

**⚠️ 局限性**

局限在于仍需依赖LLM作为最终决策者，对极大规模或复杂任务的可扩展性待验证；此外，scout与LLM间的文本化桥梁可能对特定任务产生信息损失

---

## 471. FISMO: Fisher-Structured Momentum-Orthogonalized Optimizer

**arXiv ID:** 2601.21750 | [PDF](https://arxiv.org/pdf/2601.21750v1)

**作者:** Chenrui Xu `[一作]` (Chinese University of Hong Kong), Ying-Jun Angela Zhang `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 15133 | [OpenAlex ID](https://openalex.org/A5004874287)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为FISMO的优化器，结合了Fisher信息几何与Muon's矩阵正交化策略，用于大规模神经网络的矩阵参数更新。

**💡 创新点**

创新点在于将自然梯度的Kronecker‑factored Fisher预处理与正交化动量结合，既保留了曲率信息，又保持了计算可行性，并给出了闭式最优更新和收敛证明。

**🔧 技术方法**

使用的技术包括：Kronecker‑factored Fisher近似、Gauss–Seidel迭代求解预处理矩阵、指数滑动平均与正则化稳定预处理、Newton–Schulz迭代近似极化分解、以及在预处理空间中进行正交化动量更新。

**📊 数据集**

实验数据集包括OpenWebText（用于GPT‑2 124M语言模型）和CIFAR‑10（用于SimpleDLA图像分类）。

**📈 对比分析**

与SGD、AdamW、Shampoo、Muon等基准相比，FISMO在训练损失、验证损失以及最终准确率上均表现更好，收敛速度最快且训练轨迹更平稳。

**⚠️ 局限性**

局限性包括：仍需额外的预处理矩阵维护与更新开销；在极大模型或极端稀疏场景下Kronecker近似的有效性可能受限；以及在某些任务中对批量大小和学习率的敏感性。

---

## 472. Temporal Sepsis Modeling: a Fully Interpretable Relational Way

**arXiv ID:** 2601.21747 | [PDF](https://arxiv.org/pdf/2601.21747v1)

**作者:** Vincent Lemaire `[一作]` (Orange Research), Pierre Jaquet `[通讯]` (Recherche Clinique La Fontaine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出一种利用电子病历的关系型表示、MDL驱动的特征化与选择性朴素贝叶斯分类器，对重症监护病房（ICU）患者的败血症进行早期预测，并提供多层级可解释性（单变量、全局、局部）与反事实推理。

**💡 创新点**

创新点包括：① 将时间序列 EMR 转化为星型关系型数据并通过关系特征化（MDL）自动生成可解释的聚合变量；② 在保持可解释性的前提下，通过选择性贝叶斯实现高效、稀疏的分类；③ 将单变量重要性、全局重要性与 Shapley 局部重要性统一展示，并结合反事实轨迹阐释模型决策。

**🔧 技术方法**

使用技术：关系型数据表示、MDL‑基 propositionalization、监督离散化/分组、Fractional Naive Bayes、Shapley 价值分析、反事实轨迹生成。

**📊 数据集**

数据集：MIMIC‑III 临床数据库，采用 3940 名 ICU 患者的 36 项关键生理指标（按小时记录）构建特征，预测时窗为 3h 与 6h。

**📈 对比分析**

方法对比：在 10‑折交叉验证中，AUC 最高达 0.9992，准确率 0.9883；与传统基于聚合特征的机器学习模型（Logistic, RF, GBM）相比，保持了相近或更优性能，且模型尺寸更小、解释性更强。

**⚠️ 局限性**

局限性：① 仅验证于 ICU 病例，外推至其他科室或不同机构尚待验证；② 对缺失值采用 SMOTE 估计，可能引入偏差；③ 仅使用单一时窗（12h）和小时级别数据，难以处理更高频或多模态信息；④ 反事实与局部解释需要临床专家进一步评估其医学意义。

---

## 473. Mixed-Precision Training and Compilation for RRAM-based Computing-in-Memory Accelerators

**arXiv ID:** 2601.21737 | [PDF](https://arxiv.org/pdf/2601.21737v1)

**作者:** Rebecca Pelke `[一作]` (RWTH Aachen University), Rainer Leupers `[通讯]` (RWTH Aachen University)

**通讯引用:** 6745 | [OpenAlex ID](https://openalex.org/A5023470562)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了一套基于混合精度量化训练与编译的框架，用于RRAM型计算内存（CIM）加速器，以提升推理速度并保持高精度。

**💡 创新点**

创新点在于：①将强化学习驱动的量化优化（CIM-AQ）迁移至CIM平台并与Brevitas QAT后端结合，②支持多层次混合精度（MPQ）并通过约束策略缩小搜索空间，③构建了基于TVM的CIM编译链，实现自动生成跨bar API调用。

**🔧 技术方法**

采用的技术包括：混合精度量化训练（MPQ + QAT）、强化学习（DDPG）量化优化、Brevitas量化后端、TVM编译器、ONNX/Relay图转换、跨bar数据流与调度。

**📊 数据集**

使用的数据集为ImageNet（训练集20000张/验证集10000张的子集用于搜索，随后完整ImageNet训练30个epoch），并在ResNet‑18、VGG‑16、ViT‑B/32等模型上进行评估。

**📈 对比分析**

与现有8位CIM编译器对比，平均可获得2.20×以上速度提升（VGG‑16最高2.48×），准确率下降仅0.086%以内；在不同单元分辨率（2/4位）下，均保持良好速度/精度平衡。

**⚠️ 局限性**

局限性包括：RL搜索仍耗时较长、仅考虑量化误差而未加入交叉条非理想性、在极低位宽（≤2位）时精度损失略增、对不同CIM架构的适配度需要进一步验证。

---

## 474. E-mem: Multi-agent based Episodic Context Reconstruction for LLM Agent Memory

**arXiv ID:** 2601.21714 | [PDF](https://arxiv.org/pdf/2601.21714v1)

**作者:** Kaixiang Wang `[一作]`, Jie Li `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 26888 | [OpenAlex ID](https://openalex.org/A5100428255)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 E-mem 框架，利用多代理异构层级结构实现对 LLM 代理的情节上下文重构，提升长期记忆与多跳推理能力。

**💡 创新点**

创新点：①将记忆处理从破坏性去上下文化转为情节上下文重构；②使用小型助理代理在局部上下文内主动推理并向主代理汇总；③多通道路由（全局、语义、符号）实现高效检索与低噪声激活。

**🔧 技术方法**

技术手段：异构层级 Master‑Assistant 架构；小型语言模型助理 + GPT/​Qwen 主代理；滑动窗口重叠切分、分块存储；多通道路由（全局向量、语义向量、符号匹配）；局部推理与全局聚合；KV 缓存提升检索速度。

**📊 数据集**

使用数据集：LoCoMo（多跳、时序、开放域、对抗子集）和 HotpotQA（400/800/1600 文档扩展版）。

**📈 对比分析**

与 Long‑Context、RAG、A‑Mem、Mem0、MemoryOS、LightMem、GAM 等基线对比，LoCoMo 上总体 F1 54.17（GPT‑4o‑mini）/57.04（Qwen‑2.5‑14B）超过 SOTA 7.75+；多跳+时序任务提升 8+ 点；token 成本降低 70%+；HotpotQA F1 61.46/61.13/55.76 也优于其他基线。

**⚠️ 局限性**

局限性：需多模型协作，部署复杂；助理规模越大单跳性能略下降；对极长序列仍可能出现 “lost‑in‑middle” 风险；在某些任务中对路由阈值的敏感度仍需进一步优化。

---

## 475. Beyond Forgetting: Machine Unlearning Elicits Controllable Side Behaviors and Capabilities

**arXiv ID:** 2601.21702 | [PDF](https://arxiv.org/pdf/2601.21702v1)

**作者:** Tien Dang `[一作]` (Japan Advanced Institute of Science and Technology), Naoya Inoue `[通讯]` (RIKEN)

**通讯引用:** 3991 | [OpenAlex ID](https://openalex.org/A5086772095)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了通过操纵大模型的忘记表示（forget-representations）实现机器不学习（machine unlearning）的方式，并发现这不仅能实现遗忘，还能控制模型的副行为和增强其能力。

**💡 创新点**

创新点在于将线性表示假设与表示误导（representation misdirection）结合，提出“可控副效应假设”，并通过向量加法与消除操作展示如何在不影响总体知识的前提下实现行为控制。

**🔧 技术方法**

主要技术包括线性向量干预（加法与消除）、逻辑回归探测器提取概念方向、以及基于预训练LLM的表示层操作。

**📊 数据集**

使用的数据集包括WMDP‑Biology、WMDP‑Cyber、TruthfulQA、SST‑2、AdvBench、Alpaca、MMLU以及多种语言与知识任务的数据集。

**📈 对比分析**

与基准模型和随机方向对比，实验表明在遗忘任务上能显著降低遗忘集的准确率，同时通过真诚、情感、拒绝等方向实现性能提升（如TruthfulQA准确率提升+12.7，SST‑2正向情感提升+39.9），但在知识恢复攻击下模型易被复苏。

**⚠️ 局限性**

局限性包括对线性表示假设的依赖、对模型架构的局部性（仅在特定层操作）、以及在面临知识恢复攻击时不够鲁棒，且随机方向在某些任务上效果有限。

---

## 476. Abstract Concept Modelling in Conceptual Spaces: A Study on Chess Strategies

**arXiv ID:** 2601.21771 | [PDF](https://arxiv.org/pdf/2601.21771v1)

**作者:** Hadi Banaee `[一作]` (Orebro University), Stephanie Lowry `[通讯]` (Orebro University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出基于概念空间的几何框架，用于对抽象的国际象棋策略（如王攻击、位置牺牲、空间统治）进行时序识别与解释。

**💡 创新点**

创新点在于将抽象策略建模为多维可解释空间中的凸区域，并通过棋局轨迹的方向性运动来识别策略意图，支持玩家视角的双重解释。

**🔧 技术方法**

采用概念空间理论、几何区域定义、轨迹分析、Python‑chess 计算特征（材质、机动性、脆弱性、控制、流动、压力、空间）等技术。

**📊 数据集**

使用公开国际象棋数据库中的已标注精通级对局（如Tal‑Hecht 1962、Petrosian‑Pachman 1961）进行实验验证。

**📈 对比分析**

通过与专家注释对比，轨迹与概念区域的匹配程度高于随机或仅基于单维指标的方法，但尚未给出量化性能指标；主要为定性一致性验证。

**⚠️ 局限性**

局限包括：区域边界及维度选取高度依赖专家经验，缺乏数据驱动学习；验证规模有限，仅为小样本案例，缺乏系统的统计评估。

---

## 477. CoFrGeNet: Continued Fraction Architectures for Language Generation

**arXiv ID:** 2601.21766 | [PDF](https://arxiv.org/pdf/2601.21766v1)

**作者:** Amit Dhurandhar `[一作]` (IBM Research), Rahul Nair `[通讯]` (IBM Research)

**通讯引用:** 1657 | [OpenAlex ID](https://openalex.org/A5022200143)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一类基于连分数的生成网络CoFrGeNet，替换Transformer中的注意力和前馈网络，实现更小参数、更快训练和推理的生成模型。

**💡 创新点**

创新点包括：①把连分数函数类引入生成模型；②利用连分数的连续子式(continuants)获得一次除法、闭式梯度；③设计自定义训练调度和可插拔组件，兼容现有Transformer训练管线。

**🔧 技术方法**

使用技术主要有：连分数网络（CoFrNet）实现注意力/FFN；连分式求值与梯度公式；PyTorch自定义autograd实现一次除法；自定义学习率调度（分层冻结/解冻）；多GPU分布式训练（DDP/FSDP）。

**📊 数据集**

预训练数据集：OpenWebText、GneissWeb、docling数据混合（2T tokens）；评估数据集：PTB、WikiText‑2/103、Lambda、AgNews、OneBillionWords、GLUE（MNLI、QQP、QNLI、SST‑2、COLA、MRPC、RTE、WNLI）、OpenBookQA、PiQA、ARC‑Easy、Winogrande、HellaSwag、Lambda OpenAI、Boolq、SciQ。

**📈 对比分析**

对比方法：与原始GPT‑2‑xl、GPT‑2‑xl‑GW、Synthesizer‑D、Sparse‑Attn、Llama‑3.2B等模型在相同预训练/微调设置下进行对比。性能表现：CoFrGeNet在绝大多数GLUE任务、Perplexity、推理时间和训练时间上均与原模型持平甚至优于；参数量下降至原来的约50‑70%；推理速度提升近10倍；通过自定义调度可进一步提升效果。

**⚠️ 局限性**

局限性：模型仍易出现hallucination、对抗攻击；仅在Transformer与Llama框架中验证，未测试于其他架构；实现仍依赖PyTorch自定义梯度，未完成硬件级优化（如Triton核）；连分数的数值稳定性仍需在更大模型/更深网络中进一步评估。

---

## 478. KnowBias: Mitigating Social Bias in LLMs via Know-Bias Neuron Enhancement

**arXiv ID:** 2601.21864 | [PDF](https://arxiv.org/pdf/2601.21864v1)

**作者:** Jinhao Pan `[一作]` (George Mason University), Ziwei Zhu `[通讯]` (George Mason University)

**通讯引用:** 1446 | [OpenAlex ID](https://openalex.org/A5019994221)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为KnowBias的推理时去偏框架，通过增强LLM内部的“偏见认知”神经元来抑制社会偏见，避免直接压制偏见相关神经元导致的性能退化。

**💡 创新点**

创新点在于从“抑制偏见”转向“激活偏见知识”，利用少量是非式偏见认知问题通过归因分析快速定位并放大模型内部的偏见意识表征，从而实现轻量、高效、跨偏见类型与人群的去偏。

**🔧 技术方法**

技术主要包括：设计三类偏见认知问题（因果拒绝、偏见识别、规范判断）；使用积分梯度归因识别偏见知识神经元；推理时按比例放大这些神经元的激活；在LLM（Llama‑3.2‑3B、Llama‑3.1‑8B、Qwen‑3‑4B）上无梯度更新地应用。

**📊 数据集**

实验使用五大社会偏见基准（BBQ‑a/d、CrowS‑Pairs、StereoSet‑inter/intra）以及三大通用推理/QA数据集（COPA、OpenBookQA、ARC‑E/C），并以这些数据评估去偏效果与模型通用性能。

**📈 对比分析**

与七种主流去偏方法（prompt、fine‑tune、model editing、activation steering、bias‑neuron deletion）对比，KnowBias在所有基准上的平均排名均为最优或次优，并在通用任务上保持与基线相近或更优的准确率，表明兼顾公平与效用。

**⚠️ 局限性**

局限性包括：仅覆盖性别、种族、宗教三大维度，难以涵盖残障、年龄、社会经济等更广泛偏见；依赖预设的偏见认知问题集合，可能无法捕捉所有社会规范；以及在不同文化或语言背景下的适用性和解释性仍需进一步验证。

---

## 479. LEMUR: Learned Multi-Vector Retrieval

**arXiv ID:** 2601.21853 | [PDF](https://arxiv.org/pdf/2601.21853v1)

**作者:** Elias Jääsaari `[一作]` (University of Helsinki), Teemu Roos `[通讯]` (University of Helsinki)

**通讯引用:** 2623 | [OpenAlex ID](https://openalex.org/A5066842476)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种将多向量检索简化为监督学习，再进一步降维为单向量近似最近邻检索的框架 LEMUR，显著降低检索延迟。

**💡 创新点**

创新点在于将 MaxSim 近似视为多输出回归问题，利用一层 MLP 直接学习令查询与文档间的 MaxSim 接近的单向量表示，从而可复用高效的单向量 ANN 库。

**🔧 技术方法**

使用的技术包括：双层 MLP（含 GELU+LN），基于 MSE 的监督训练，内积形式的输出权重映射为文档向量，Glass HNSW 的单向量 ANN，C++ 实现的 MaxSim 重新排序。

**📊 数据集**

实验数据集涵盖 6 个 BEIR 文本检索集（MS MARCO、HotpotQA、NQ、Quora、SCIDOCS、ArguAna）和 2 个视觉文档检索集 ViDoRe，嵌入模型包括 ColBERTv2、answerai-colbert-small、GTE-ModernColBERT、LFM2-ColBERT、jina-colbert-v2、ColModernVBERT、ColQwen2。

**📈 对比分析**

与 MUVERA、DESSERT、IGP、PLAID 等主流方法对比，LEMUR 在相同召回率下平均快 5–11 倍，且在非 ColBERTv2 嵌入上仍保持高召回，显示出更强的鲁棒性与更低的延迟。

**⚠️ 局限性**

局限性主要是未针对多向量表示的存储做压缩，当前仅兼容 8‑bit 标量量化；极低精度（如 2‑bit）压缩以及更大规模多向量存储的兼容性仍待进一步研究。

---

## 480. Visual Disentangled Diffusion Autoencoders: Scalable Counterfactual Generation for Foundation Models

**arXiv ID:** 2601.21851 | [PDF](https://arxiv.org/pdf/2601.21851v1)

**作者:** Sidney Bender `[一作]` (Technical University of Berlin), Marco Morik `[通讯]` (Technical University of Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Visual Disentangled Diffusion Autoencoders（DiDAE）框架，实现基于冻结的基础模型与可解码词典相结合的无梯度、可解释的对抗样本生成，并将其应用于Counterfactual Knowledge Distillation以纠正“聪明汉斯”策略。

**💡 创新点**

创新点在于：①将冻结的基础模型与可解码、可逆的分词典学习结合，实现对潜在空间的可解释解耦与快速反射；②采用单步扩散解码，完全摆脱梯度优化；③提出预聚类教师实现CFKD的可扩展标注，显著提升生成速度与多样性。

**🔧 技术方法**

使用技术包括：冻结CLIP/自定义基础编码器、Procrustes或SVD词典学习、可逆分词典、条件扩散解码器（Diffusion Autoencoder）、无梯度的组件反射与决策边界反射算法、CFKD与投影校正方法。

**📊 数据集**

实验数据集为合成的Square数据集和CelebA-Blond子集（包含性别/金发标签），并在OpenCLIP上对CelebA进行评估。

**📈 对比分析**

与DiME、ACE、FastDiME、SCE、GroupDRO等方法对比，DiDAE在生成速度上提升两到三十倍（最高64个反事实/秒），在非对抗性翻转率（NAFR）与增益（Gain）上保持竞争力甚至优于多数基线，在ResNet-18和基础模型探测器上的平均组准确率（AGA）均达到最优。

**⚠️ 局限性**

局限性包括：1）由于DDIM逆变过程的平滑，生成质量略低于SCE；2）在空间重叠的特征上仍可能产生非完美解耦；3）仍需手动映射词典或先验知识；4）目前仅适用于连续视觉域，尚未扩展到离散文本或图结构。

---

## 481. Goal-Driven Adaptive Sampling Strategies for Machine Learning Models Predicting Fields

**arXiv ID:** 2601.21832 | [PDF](https://arxiv.org/pdf/2601.21832v1)

**作者:** Jigar Parekh `[一作]` (Cluster of Excellence SE²A Sustainable and Energy Efficient Aviation), Philipp Bekemeyer `[通讯]` (Institute of Aerodynamics and Flow Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种适用于场预测的模型无关主动学习策略，利用高斯过程(GP)和图神经网络(GNN)的误差与误配度（或Jensen‑Shannon散度）作为采样准则，联合改进标量和分布式预测。

**💡 创新点**

创新点在于将GP误差与场模型的标量一致性误差（或其概率分布差异）相结合，形成一种既能覆盖整个参数域又能聚焦关键流动区的复合采样准则，并且该准则与具体的场预测模型（GNN、PODI等）无关。

**🔧 技术方法**

主要技术包括高斯过程回归、图神经网络（ResGatedGraphConv）、PODI与插值、蒙特卡罗Dropout估计不确定性、Jensen‑Shannon散度、主动学习（采样准则优化）和基于分布式输入的蒙特卡罗/准随机采样。

**📊 数据集**

使用的数据集为NASA Common Research Model（CRM）CFD仿真数据，包含约5000万点网格、50万表面点、自然激波与过渡区信息，以及四个随机输入参数（自由流湍流强度、攻角、雷诺数、马赫数）。

**📈 对比分析**

通过与传统GP误差采样、单独GNN采样以及无采样DoE对照，实验表明SEwMisfit和JSD在30次迭代内即可将标量预测r²≥0.99、相对RMSE<3%并将场预测误差压至≈0.02，显著优于单纯GP或GNN采样，并大幅降低所需高精度CFD样本数。

**⚠️ 局限性**

局限性包括：MC‑Dropout产生的方差估计可能不稳定且计算开销大；所提出的复合准则主要在NASA CRM CFD场景验证，尚需在其他流体或工程领域进一步验证其通用性；以及对高维参数空间的可扩展性仍有待研究。

---

## 482. Mil-SCORE: Benchmarking Long-Context Geospatial Reasoning and Planning in Large Language Models

**arXiv ID:** 2601.21826 | [PDF](https://arxiv.org/pdf/2601.21826v1)

**作者:** Aadi Palnitkar `[一作]` (University of Maryland), Xiaomin Lin `[通讯]` (Johns Hopkins University)

**通讯引用:** 149 | [OpenAlex ID](https://openalex.org/A5101610451)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MilSCORE基准，用多源、长上下文的军事情境地图与OPORD文件设计专家级多跳空间推理问题，考察LLM和VLM在真实战术规划中的决策与推理能力。

**💡 创新点**

创新点在于首个情境级、专家撰写的多跳空间推理数据集，包含七大空间分析范畴和不可解任务，并配套基于工具调用的链式思考评测框架。

**🔧 技术方法**

使用工具调用链式思考(ReAct)、视觉语言模型、检索增强生成（RAG）与LLM评判器，实现从多模态源中提取信息并生成结构化答案。

**📊 数据集**

数据集由100+题、50张作战地图、OPORD PDF、GeoJSON和卫星图像组成，全部来自未分类的军事训练情境，由专业人员校对。

**📈 对比分析**

在60道测试题上与GPT‑4o、Claude Sonnet 4.5、Gemini 2.5 Flash等顶尖VLM比较，GPT‑4o获得最高准确率（约51.7%），但整体仍低，说明当前模型在跨源多跳推理与长上下文处理上仍显弱。

**⚠️ 局限性**

主要局限包括：模型易出现幻觉、工具调用受限导致完成率下降、对长文本信息捕获不足、数据集规模有限且仅覆盖单一训练情境，需进一步扩展与多样化。

---

## 483. DASH: Deterministic Attention Scheduling for High-throughput Reproducible LLM Training

**arXiv ID:** 2601.21824 | [PDF](https://arxiv.org/pdf/2601.21824v1)

**作者:** Xinwei Qiang `[一作]` (Shanghai Jiao Tong University), Minyi Guo `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 14224 | [OpenAlex ID](https://openalex.org/A5039318240)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了针对FlashAttention-3确定性反向传播的调度框架，显著提升了大模型训练的吞吐量

**💡 创新点**

通过将确定性注意力的计算建模为DAG调度问题，推出了最优Shift调度与对称Shift调度两种策略，实现了理论上最短关键路径

**🔧 技术方法**

使用DAG最短路径优化、逆序Q‑tile遍历、Shift调度、对称Shift调度等技术，并结合CUDA/ Triton实现

**📊 数据集**

在NVIDIA H800 GPU上对LLM（LLaMA3‑8b、Qwen2.5‑7b、Mistral‑8×7b）和多模态模型（SAM‑huge、StableDiffusion3.5、LLaDA‑1b）进行实验

**📈 对比分析**

与FlashAttention‑3确定性后向基线对比，吞吐量提升最高可达1.28×，端到端平均加速约5%

**⚠️ 局限性**

受寄存器压力和跨SM通信延迟的限制，理论最优调度在极高并行度或大head维度下不一定优于简单方案

---

## 484. General Self-Prediction Enhancement for Spiking Neurons

**arXiv ID:** 2601.21823 | [PDF](https://arxiv.org/pdf/2601.21823v1)

**作者:** Zihan Huang `[一作]` (Peking University), Tiejun Huang `[通讯]` (Peking University)

**通讯引用:** 14128 | [OpenAlex ID](https://openalex.org/A5058066577)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于神经元自身输入输出历史的自我预测增强的尖峰神经元模型。

**💡 创新点**

创新点在于通过内部预测电流实现梯度连续化、提升训练稳定性，并与生物学的树突预测机制对应。

**🔧 技术方法**

使用尖峰神经网络、低通滤波预测电流、Surrogate梯度训练等技术。

**📊 数据集**

在CIFAR‑10、ImageNet‑100/1k、Sequential CIFAR‑10以及MuJoCo RL任务上进行评估。

**📈 对比分析**

与传统SNN（IF/LIF/PLIF/CLIF）和多种网络架构对比，平均提升1–5%分类准确率，RL任务平均提升约3%。

**⚠️ 局限性**

局限性包括对IF神经元改进有限、性能提升受时步长影响、需要额外超参数且在极大规模任务验证不足。

---

## 485. A Unified XAI-LLM Approach for EndotrachealSuctioning Activity Recognition

**arXiv ID:** 2601.21802 | [PDF](https://arxiv.org/pdf/2601.21802v1)

**作者:** Hoang Khang Phan `[一作]` (Vietnam National University), Nhat Tan Le `[通讯]` (Vietnam National University)

**通讯引用:** 57 | [OpenAlex ID](https://openalex.org/A5091430591)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个以大型语言模型为核心的统一框架，用于端气管吸痰活动的零样本识别、可解释分析和自然语言反馈。

**💡 创新点**

创新点在于将LLM作为多模态推理引擎，利用提示工程提升认知准确度，并将可解释AI与视频关键点融合生成实时可理解的教学反馈。

**🔧 技术方法**

技术包括Gemini 2.5 Pro LLM、姿态估计与关键点提取、Isolation Forest+SHAP异常检测、prompt engineering以及多模态（视频、文本、XAI特征）融合。

**📊 数据集**

使用了44段录制自10名经验护士和12名学生的端气管吸痰视频数据集，按训练32段、测试12段划分，并提取姿态关键点。

**📈 对比分析**

通过与传统基线（姿态特征+机器学习/深度学习）对比，LLM方案平均准确率78.7%、F1分数62.9%，较基线提升约15–20%。

**⚠️ 局限性**

限制包括缺乏实时推理能力、未进行用户体验研究，以及对“清理/后续”步骤区分仍存在误判。

---

## 486. KID: Knowledge-Injected Dual-Head Learning for Knowledge-Grounded Harmful Meme Detection

**arXiv ID:** 2601.21796 | [PDF](https://arxiv.org/pdf/2601.21796v1)

**作者:** Yaocong Li `[一作]` (Beijing University of Posts and Telecommunications), Qiang Yan `[通讯]` (Beijing Information Science and Technology University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于知识注入的双头学习框架 KID，用于识别多语言有害 meme。

**💡 创新点**

创新点在于实体锚定的知识注入策略与双头联合学习，实现知识与视觉文本的显式链路及决策边界的稳定性。

**🔧 技术方法**

采用多模态大型语言模型 Qwen2.5-VL-7B 作为骨干，Gemini-2.5-Flash 生成实体关联知识，双头结构包含语义生成头与分类头。

**📊 数据集**

使用英语、中文、孟加拉语五个数据集：Hateful Memes、HarMeme、MAMI（Task A/B）、ToxiCN_MM（Task A/B）以及 BanglaAbuseMeme（Task A/B）。

**📈 对比分析**

与多种基线（CLIP、MOMENTA、Gemini、Qwen2.5-VL-7B 传统单头）比较，KID 在二分类和多分类任务上均取得 SOTA，平均提升 2.1%–19.7%（如 Hateful Memes AUC +2.14%、MAMI A +6.0%、BanglaAbuseMeme A +19.44%）。

**⚠️ 局限性**

局限性包括对教师模型知识生成质量的依赖、推理时额外的 LLM 调用导致延迟，以及多标签任务下最佳知识注入量可能不同。

---

## 487. Error Amplification Limits ANN-to-SNN Conversion in Continuous Control

**arXiv ID:** 2601.21778 | [PDF](https://arxiv.org/pdf/2601.21778v1)

**作者:** Zijie Xu `[一作]` (Peking University), Zhaofei Yu `[通讯]` (Peking University)

**通讯引用:** 3377 | [OpenAlex ID](https://openalex.org/A5048087489)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究ANN到SNN的转换在连续控制任务中的表现，并提出一种无需训练的跨步残差电位初始化（CRPI）机制来抑制误差放大。

**💡 创新点**

发现连续控制中误差随时间正相关导致累积放大，提出CRPI通过携带上一步的残差电位初始化下一步神经元状态，打破误差相关性，从而减少状态漂移。

**🔧 技术方法**

采用Integrate-and-Fire神经元的rate编码、残差电位初始化、跨步残差正则化，结合多阈值、符号神经元等SNN转换技术，对已有的ANN策略进行推理。

**📊 数据集**

使用MuJoCo四个向量观测环境（HalfCheetah, Hopper, Walker2d, Ant）以及DeepMind Control Suite六个视觉观测环境（Acrobot, Cartpole, Cheetah, Finger, Quadruped, Reacher）进行评估。

**📈 对比分析**

与原始ANN、传统IF、SNM、MT、DC等转换方法以及直接训练的SNN（Leaky、TC-LIF、Spiking‑WM）进行对比；CRPI在大多数任务上使SNN性能恢复至ANN水平甚至超过ANN，平均性能比提升至70%–110%，并保持低能耗（相较ANN降低数百倍）。

**⚠️ 局限性**

对超参数α敏感，过大导致过补偿；方法主要抑制误差相关性，无法消除所有误差；目前仅在rate编码SNN上验证，未评估在更复杂或硬件实现上的泛化。

---

## 488. READY: Reward Discovery for Meta-Black-Box Optimization

**arXiv ID:** 2601.21847 | [PDF](https://arxiv.org/pdf/2601.21847v1)

**作者:** Zechuan Huang `[一作]` (South China University of Technology), Zeyuan Ma `[通讯]` (South China University of Technology)

**通讯引用:** 5884 | [OpenAlex ID](https://openalex.org/A5063438356)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为READY的框架，利用大型语言模型（LLM）自动发现Meta-Black-Box优化中的奖励函数，以提高优化性能。

**💡 创新点**

创新点在于使用LLM进行奖励设计，克服了传统手工设计奖励的偏见和局限性，并引入了多任务进化架构以支持并行奖励发现。

**🔧 技术方法**

使用了大型语言模型（LLM）作为自动化奖励发现工具，并结合了多任务进化和细粒度进化操作符。

**📊 数据集**

使用了BBOB测试套件，包含多个优化任务和函数进行实验验证。

**📈 对比分析**

与手工设计的奖励和其他基线方法（如Eureka、EoH和ReEvo）进行比较，READY在多项测试中表现优越，显示出更低的平均成本和更好的优化性能。

**⚠️ 局限性**

限制在于当前框架的复杂性和对LLM的依赖，可能在特定情况下影响其适用性和效率。

---

## 489. Spatiotemporal Continual Learning for Mobile Edge UAV Networks: Mitigating Catastrophic Forgetting

**arXiv ID:** 2601.21861 | [PDF](https://arxiv.org/pdf/2601.21861v1)

**作者:** Chuan-Chi Lai `[一作]` (National Chung Cheng University), Chuan-Chi Lai `[通讯]` (National Chung Cheng University)

**通讯引用:** 114 | [OpenAlex ID](https://openalex.org/A5082920277)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于群组解耦多智能体近端策略优化（G‑MAPPO）的时空持续学习框架，用于无人机集群在动态用户分布下的自适应部署。

**💡 创新点**

创新点在于引入群组解耦策略优化（GDPO）动态归一化冲突奖励，结合UAV 3D垂直机动的物理补偿，显著降低灾难性遗忘并提升跨环境泛化能力。

**🔧 技术方法**

采用了多智能体强化学习（MAPPO、MADDPG）框架，GDPO奖励归一化，概率LoS模型，仿真中使用多种用户分布（TCP、GMM、HPPP）以及3D路径规划。

**📊 数据集**

数据集为仿真生成的三阶段用户分布：拥挤城市（140用户）、郊区（80‑100用户）、农村（40用户），并在不同密度下重复实验。

**📈 对比分析**

通过与MADDPG基线及理想全局信息的Static K‑Means做对比，G‑MAPPO在各阶段实现了≈20%容量提升、覆盖可靠率≈0.95、奖励方差显著降低，显示出更强的灾难性遗忘恢复与协同效率。

**⚠️ 局限性**

主要局限包括仅在4架UAV的仿真环境下验证，缺乏真实世界部署与算力受限的边缘训练，且未考虑更复杂的网络拓扑或RIS等技术。

---

## 490. Optimal Energy-Aware Service Management in Future Networks with a Gamified Incentives Mechanism

**arXiv ID:** 2601.21846 | [PDF](https://arxiv.org/pdf/2601.21846v1)

**作者:** Konstantinos Varsos `[一作]` (Athens University of Economics and Business), Vasillios A. Siris `[通讯]` (Athens University of Economics and Business)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出基于个性化激励与游戏化排名的能源感知服务管理框架，鼓励视频流用户降码率以降低网络能耗。

**💡 创新点**

创新点在于将环境敏感因子、概率接受模型与 Stackelberg 博弈结合，利用社交奖励（top‑K/bottom‑M）在有限预算下实现能源与流量双重降低。

**🔧 技术方法**

采用了游戏化激励、概率接受 Sigmoid 模型、Stackelberg 博弈优化、能耗与碳排模型以及仿真模拟等技术。

**📊 数据集**

使用合成用户群体（N=1000）随机生成的高/低码率、激励分布（正态/对数正态）以及可选的真实 Ultra‑HD 与 Full‑HD 场景。

**📈 对比分析**

通过与无激励、无游戏化以及不同预算/奖励参数的对比实验，实验显示在预算约束下能量与流量可降低至约 67.2%，显著优于传统方式。

**⚠️ 局限性**

局限在于模型假设用户行为独立、奖励参数需要经验调优，且缺乏长期行为适应与公平性分析。

---

## 491. Constrained Meta Reinforcement Learning with Provable Test-Time Safety

**arXiv ID:** 2601.21845 | [PDF](https://arxiv.org/pdf/2601.21845v1)

**作者:** Tingting Ni `[一作]` (École Polytechnique Fédérale de Lausanne), Maryam Kamgarpour `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 2757 | [OpenAlex ID](https://openalex.org/A5082009236)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种安全约束下的元强化学习算法，能够在测试阶段保证策略可行并快速收敛到近最优策略。

**💡 创新点**

创新点在于同时学习一组近最优策略集合与所有任务均可行的安全策略，并通过自适应混合策略实现安全探索与性能提升；理论上给出了与问题相关的上界与匹配下界。

**🔧 技术方法**

采用了CMDP理论、模拟器采样、Policy Collection–Elimination、可行策略oracle、混合策略自适应更新、统计收敛分析等技术。

**📊 数据集**

实验使用了 7×7 网格世界环境，任务分布为截断高斯噪声水平的 CMDP。

**📈 对比分析**

与 Safe Meta‑RL、DOPE+、LB‑SGD 三个基线相比，所有方法均保持安全探索；本算法在测试奖励回报上比最佳基线低约 50%，并在样本效率上更优。

**⚠️ 局限性**

局限在于受任务分布覆盖度与安全裕度 ξ 的影响，当 ξ 很小或分布较广时样本复杂度显著上升；实验仅在离散网格环境中验证，缺乏连续或高维空间的实证。

---

## 492. Embodied Task Planning via Graph-Informed Action Generation with Large Lanaguage Model

**arXiv ID:** 2601.21841 | [PDF](https://arxiv.org/pdf/2601.21841v1)

**作者:** Xiang Li `[一作]` (Purdue University), Masood Mortazavi `[通讯]` (Futurewei Technologies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于图神经网络的双层图记忆（Graph-in-Graph）框架，结合有界前瞻和经验检索，用于改进大型语言模型在具身任务规划中的长序列决策。

**💡 创新点**

创新点在于将场景图与状态转移图结构化存储、使用GNN编码产生可检索的嵌入、并通过有界前瞻模块提供基于环境逻辑的即时状态投影，从而显著提升规划连贯性和执行效率。

**🔧 技术方法**

技术方案包括GAT图注意力网络、triplet+uniformity 损失训练、Faiss向量检索、LLM提示工程、循环检测和有界前瞻推理。

**📊 数据集**

实验数据集涵盖Robotouille Synchronous、Robotouille Asynchronous 以及 ALFWorld 三个具身规划基准。

**📈 对比分析**

与 ReCAP、ReAct、CoT 等基线对比，本文在 Pass@1 上分别提升 Robotouille Synchronous 22%、Asynchronous 37% 以及 ALFWorld 15%，并在步骤数和 FLOPs 上实现更低成本与更高成功率。

**⚠️ 局限性**

局限性包括对大规模 LLM 的高度依赖、推理延迟可能影响实时部署，以及在部分任务中经验检索的迁移性不足。

---

## 493. Looking Beyond Accuracy: A Holistic Benchmark of ECG Foundation Models

**arXiv ID:** 2601.21830 | [PDF](https://arxiv.org/pdf/2601.21830v1)

**作者:** Francesca Filice `[一作]` (University of Calabria), Simona Perri `[通讯]` (University of Calabria)

**通讯引用:** 2230 | [OpenAlex ID](https://openalex.org/A5052577422)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

对 ECG 专用的基座模型进行系统化基准测试，结合下游性能与嵌入表示分析，覆盖跨洲数据集与不同模型。

**💡 创新点**

将表示层评估（SHAP、UMAP）与传统性能评估统一到同一框架，并提供公开可复现的基准工具。

**🔧 技术方法**

采用冻结嵌入提取、5 种轻量级线性探针、SHAP 特征重要性、UMAP 可视化与聚类度量（kNN、Centroid、ARI）等技术进行嵌入质量评估。

**📊 数据集**

使用 4 个 12 导联 ECG 数据集（PTX、C15、GEO、CHN），按样本量分为 XS–L，涵盖欧洲、美洲、亚洲。

**📈 对比分析**

通过 15 折交叉验证的 F1 分数评估下游分类，并用 SHAP/UMAP 计算跨数据集共享特征率、嵌入空间的标签与数据集可分离度；结果显示 ECG‑FM 与 ECGFounder 在多种规模与数据稀缺情况下性能最高，HuBERT‑ECG 与 ECG-JEPA 表现相对较差。

**⚠️ 局限性**

评估中部分模型预训练时已包含目标数据集，且仅聚焦 CD 与 AF 两类，未覆盖更广泛的心脏疾病，且对嵌入空间解释仍有限。

---

## 494. MMFineReason: Closing the Multimodal Reasoning Gap via Open Data-Centric Methods

**arXiv ID:** 2601.21821 | [PDF](https://arxiv.org/pdf/2601.21821v1)

**作者:** Honglin Lin `[一作]` (Shanghai Artificial Intelligence Laboratory), Lijun Wu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了MMFineReason数据集（约180万样本、51亿词）并在此数据上对Qwen3-VL-Instruct进行SFT+RL训练，生成MMFineReason-2B/4B/8B模型。

**💡 创新点**

创新点在于提出系统化的数据聚合、清洗、标准化与高质量Chain‑of‑Thought蒸馏流程，并通过难度感知过滤实现极高的数据效率，显著提升小参数模型的推理能力。

**🔧 技术方法**

技术主要包括：多源数据融合、自动化清洗与格式化、基于Qwen3‑VL‑235B教师模型的长文本CoT蒸馏、SFT与RL（GSPO）训练框架、图像预处理与caption生成。

**📊 数据集**

使用了FineVision、MMR1、Euclid30K、BMMR、GameQA‑140K、TQA、AI2D等公开多模态推理数据集，并在此基础上进行蒸馏和过滤。

**📈 对比分析**

在多模态推理与通用VQA基准上，MMFineReason‑8B在数学/逻辑类任务上已超过8B开源模型，4B模型甚至超越8B思考模型；在全量数据上达成75.7%分数，较同规模对手提升约10点，低数据子集（12.3K）仅用5%样本即达到73.3%性能。

**⚠️ 局限性**

局限性包括：仍依赖教师模型Qwen3‑VL的推理质量、对数据稀缺的STEM与图形推理的偏好导致跨域泛化有限、RL阶段对资源与调参要求高、以及数据规模虽大但仍无法覆盖所有现实世界复杂视觉任务。

---

## 495. Distribution-Aware Reward Estimation for Test-Time Reinforcement Learning

**arXiv ID:** 2601.21804 | [PDF](https://arxiv.org/pdf/2601.21804v1)

**作者:** Bodong Du `[一作]` (Hong Kong University of Science and Technology), Xiaomeng Li `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 9176 | [OpenAlex ID](https://openalex.org/A5100427643)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了DARE——一种在测试时强化学习（TTRL）中基于完整回合分布的奖励估计方法，用于提升大型语言模型在未标注输入上的自我改进能力。

**💡 创新点**

创新点在于摒弃传统的多数投票（MV）奖励，利用不确定性加权的经验分布、探索奖励以及分布剪枝三项机制，全面保留回合多样性并降低噪声，显著减少确认偏差。

**🔧 技术方法**

主要技术包括：基于经验分布的奖励分配、低不确定性奖励提升的探索奖金、分布支持剪枝、以及GRPO框架下的测试时策略更新。

**📊 数据集**

实验使用五大推理基准（MMLU‑Pro、MATH‑500、AIME 2024、AMC、GPQA）以及两大模型骨干（Qwen2.5‑Math‑1.5B 与 Qwen3‑1.7B）。

**📈 对比分析**

与传统MV‑TTRL、INTUITOR、RLPR、CO‑REWARDING‑I 等基线比较，DARE 在所有基准上均取得最高平均成绩，在AIME 2024 上提升约25.3%，在AMC 上提升约5.3%，并在 OOD 评估中表现更稳定，收敛速度更快。

**⚠️ 局限性**

局限性包括：仍需足够多样的回合样本，分布估计依赖于采样多样性；对无法直接量化不确定性的任务效果未知；以及在极端相关或极端不确定的回合下可能仍出现奖励偏差。

---

## 496. Synthetic-to-Real Domain Bridging for Single-View 3D Reconstruction of Ships for Maritime Monitoring

**arXiv ID:** 2601.21786 | [PDF](https://arxiv.org/pdf/2601.21786v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 497. RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes

**arXiv ID:** 2601.21803 | [PDF](https://arxiv.org/pdf/2601.21803v1)

**作者:** Korbinian Randl `[一作]` (Stockholm University), John Pavlopoulos `[通讯]` (Athens University of Economics and Business)

**通讯引用:** 3482 | [OpenAlex ID](https://openalex.org/A5033894687)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了RAG‑E框架，对检索‑生成流水线进行端到端可解释性评估。

**💡 创新点**

创新点包括引入基于IG的检索归因、pmc化的Shapley近似、以及Weighted Attribution‑Relevance Gap (WARG)度量。

**🔧 技术方法**

采用Integrated Gradients、Kernel SHAP 的 PMC、RBO 等技术进行归因与度量。

**📊 数据集**

在 TREC CAsT 与 FoodSafeSum 两个数据集上进行实验。

**📈 对比分析**

通过与 Spearman 相关系数、WARG 等指标对比，发现 Llama 3.1 8B 与 Gemma 3 12B 在检索与生成之间存在显著不对齐，性能受限。

**⚠️ 局限性**

局限在于文档数较多时归因精度下降、易被非专业用户误解且需进一步验证与优化。

---

## 498. Enhancing Conversational Agents via Task-Oriented Adversarial Memory Adaptation

**arXiv ID:** 2601.21797 | [PDF](https://arxiv.org/pdf/2601.21797v1)

**作者:** Yimin Deng `[一作]` (Xi'an Jiaotong University), Xueming Qian `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 6703 | [OpenAlex ID](https://openalex.org/A5014825654)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种对抗式记忆适配机制（AMA），在离线记忆构建与更新阶段通过生成 QA 对、评估与双层适配，主动将任务需求引入记忆系统，提升长对话记忆的质量和任务适配性。

**💡 创新点**

创新点在于：①通过模拟任务执行的对抗式流程，在离线阶段引入任务监督；②双层更新同时优化记忆内容与构建策略；③该机制可无缝集成多种现有记忆系统，兼容不同 LLM 作为后端。

**🔧 技术方法**

使用的技术包括：大型语言模型（GPT‑4o‑mini / GPT‑4o）用于 QA 生成、答案评估和错误分析；向量检索、摘要与实体抽取用于记忆构建；基于评估结果的策略与内容更新逻辑；以及对抗式训练思路。

**📊 数据集**

实验数据集：LoCoMo 长对话基准，包含多跳推理、时间推理、开放域和单跳问答四类任务。

**📈 对比分析**

与 ReadAgent、MemoryBank、MemGPT 基线以及 A‑MEM、LightMEM、Nemori 三种主流记忆系统结合两大 LLM 进行比较，AMA 在 LoCoMo 上的 F1 与 BLEU 分数均显著提升，平均提升 3–5 个百分点；Ablation 实验表明三大模块（QA 生成、评估与双层更新）均对性能提升起关键作用。

**⚠️ 局限性**

局限性：①依赖 LLM 的 QA 生成与评估，对低资源或特定领域任务的适配可能受限；②对不同记忆实现的细节兼容性仍需手工调整；③生成 QA 对和评估过程增加了离线计算开销，实时性与资源消耗未得到充分评估。

---

## 499. NetMamba+: A Framework of Pre-trained Models for Efficient and Accurate Network Traffic Classification

**arXiv ID:** 2601.21792 | [PDF](https://arxiv.org/pdf/2601.21792v1)

**作者:** Tongze Wang `[一作]` (Tsinghua University), Yong Cui `[通讯]` (Tsinghua University)

**通讯引用:** 22247 | [OpenAlex ID](https://openalex.org/A5007046740)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出基于预训练的网络流量分类框架NetMamba，解决Transformer效率低、流量表示缺失关键字节信息以及长尾分布问题。

**💡 创新点**

创新点包括：①利用Mamba与Flash Attention构建高效线性时空状态模型；②设计多模态流量表示（字节+头部+负载+包长/间隔）并消除偏差；③引入标签分布感知微调（CB+LDAM）应对长尾数据；④实现在线分类系统，验证实战性能。

**🔧 技术方法**

技术手段：Mamba线性时空状态机、Flash Attention、GeGLU激活、MAE自监督预训练、跨模态嵌入、CB+LDAM损失、DPDK抓包+共享内存+Redis+Flask API。

**📊 数据集**

数据集：预训练使用Browser与Kitsune；下游包含8个公开与企业数据集（CipherSpectrum、CSTNET‑TLS1.3、CrossNet2021A、CP‑Android、CP‑iOS、CICIoT2022、USTC‑TFC2016、ISCXVPN2016、DataCon2021‑p1、Huawei‑VPN）；OOV评估使用四个不同协议/攻击类别。

**📈 对比分析**

与传统机器学习、深度学习、Transformer、Mamba变体及预训练基线进行对比，NetMamba+在大多数数据集上达到或超越SOTA，F1提升高达6.44%，推理吞吐比最优基线高1.7倍，显著降低GPU内存；在few‑shot和OOV检测任务中表现尤为优异。

**⚠️ 局限性**

局限性：对分布漂移的鲁棒性有限；在部分攻击或VPN场景仍落后于专门的GNN/Transformer模型；需要大量无标签数据进行预训练；在线系统延迟仍在秒级，尚未针对极高频动态流量做优化。

---

## 500. Quantum LEGO Learning: A Modular Design Principle for Hybrid Artificial Intelligence

**arXiv ID:** 2601.21780 | [PDF](https://arxiv.org/pdf/2601.21780v1)

**作者:** Jun Qi `[一作]` (Georgia Institute of Technology), Jesper Tegner `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 18205 | [OpenAlex ID](https://openalex.org/A5035252549)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为 Quantum LEGO Learning 的模块化框架，将预训练的经典网络冻结作为特征提取块，变分量子电路（VQC）作为唯一可训练的适配块；该框架可与多种经典后端（ResNet、TTN、PCA 等）兼容，适用于 NISQ 时代的混合 AI。

**💡 创新点**

创新点在于把经典与量子组件解耦成可复用的模块，提供块级误差分解理论（逼近、估计、优化），证明在冻结经典块后，量子块的逼近误差与量子维度无关；此外通过理论和实验验证该框架在噪声下具有更强的鲁棒性和更高的优化稳定性。

**🔧 技术方法**

使用技术包括：冻结预训练经典网络、张量积编码将特征映射到量子态、浅层 VQC、参数平移梯度估计、Rademacher 复杂度误差分析、硬件实验（IBM Heron 量子处理器）等。

**📊 数据集**

采用的实验数据集包括：量子点电荷稳定图（清洁与噪声 50×50 图像）和基因组 JunD 转录因子结合位点（TFBS）预测数据（101 长度 DNA 序列的 404 维 one‑hot 编码）。

**📈 对比分析**

通过与经典全连接头、不同特征提取器（ResNet18/50、TTN、PCA）以及不同量子噪声模型的对比，实验显示量子头在相同参数预算下收敛更快、最终准确率更高；在 IBM Heron 硬件上，ResNet+VQC 在 20–20+6 量子层配置下可达 90–92% 准确率，明显优于单独 VQC 或 TTN+VQC。

**⚠️ 局限性**

主要局限包括：需要先有高质量的预训练经典块，冻结策略在某些任务中可能限制模型的灵活性；量子块深度受限，若经典头已足够表达任务，量子优势可能不明显；在极端噪声或量子硬件容量极低的情况下，性能仍会下降。

---

## 501. Test-Time Compute Games

**arXiv ID:** 2601.21839 | [PDF](https://arxiv.org/pdf/2601.21839v1)

**作者:** Ander Artola Velasco `[一作]` (Max Planck Institute for Software Systems), Manuel Gomez-Rodriguez `[通讯]` (Max Planck Institute for Software Systems)

**通讯引用:** 8070 | [OpenAlex ID](https://openalex.org/A5042180520)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了LLM-as-a-service市场中的测试时计算（TTC）博弈模型，分析了服务提供商在竞争中如何选择TTC水平以最大化利润，并证明现行按TTC计费的做法导致社会福利低下；

**💡 创新点**

创新点在于首次将博弈论与机器学习成本模型结合，量化了TTC决策对社会福利的影响，并设计了逆向第二价拍卖机制，使得提供商的激励与社会福利保持一致，理论上实现PoA=1；

**🔧 技术方法**

技术方法主要包括：潜在函数（ordinal potential）博弈分析、均衡存在性证明、价格的边际价值支付规则、以及逆向第二价拍卖的机制设计；

**📊 数据集**

实验使用了多种指令模型（来自GPT/Claude等家族）与推理模型（distilled from ），在数学与科学基准数据集（如ARC、MATH、SAT等）上评估，采用best‑of‑n、majority voting与chain‑of‑thought等TTC方法；

**📈 对比分析**

通过与传统按TTC计费市场的对比，逆向第二价拍卖在用户价值提升约2–30%、社会福利提升约4–25%之间表现优异，实验中PoA>1 的传统市场在公平性和效率上显著劣于拍卖机制；

**⚠️ 局限性**

限制包括：模型假设用户对质量的价值统一，未考虑用户偏好多样性；实验仅在可验证的基准任务上进行，未覆盖开放式问题；拍卖需要提供商准确估计平均质量，且易受操纵或合谋影响。

---

## 502. Assessing the Business Process Modeling Competences of Large Language Models

**arXiv ID:** 2601.21787 | [PDF](https://arxiv.org/pdf/2601.21787v1)

**作者:** Chantale Lauer `[一作]` (German Research Center for Artificial Intelligence), Nijat Mehdiyev `[通讯]` (Saarland University)

**通讯引用:** 942 | [OpenAlex ID](https://openalex.org/A5045367105)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 BEF4LLM 框架，对 LLM 在自然语言到 BPMN 模型生成任务中的性能进行系统评估。

**💡 创新点**

创新点在于将 SIQ 质量维度扩展为四维（语法、实用、语义、有效性）并结合 39 个可量化指标，构建了面向 LLM 的统一评测工具。

**🔧 技术方法**

使用 LLM 生成技术（prompt、refinement loop、量化模型）、BPMN 语法检查、统计检验（Skillings-Mack、Wilcoxon）以及多指标聚合方法。

**📊 数据集**

评测数据集为 105 条人工标注的文本‑BPMN 对，此外对 9 条德语文本与 45 个人工 BPMN 进行专家对照实验。

**📈 对比分析**

通过 BEF4LLM 计算的四维质量分数，对 17 个开源 LLM（包括 Llama3、Qwen2.5、Qwen3、Deepseek‑R1、Phi4、Falcon 等）进行大规模对比，发现 LLM 在语法与实用维度已与人类相当，但语义与有效性仍低，较大模型并不一定更好。

**⚠️ 局限性**

主要局限包括样本量有限（仅 105 对、9 条德语文本）、只关注 BPMN XML 而忽略布局、未考虑 LLM 运行时资源与实时性、以及缺乏对商业 LLM 的评估。

---

## 503. Low-Rank Plus Sparse Matrix Transfer Learning under Growing Representations and Ambient Dimensions

**arXiv ID:** 2601.21873 | [PDF](https://arxiv.org/pdf/2601.21873v1)

**作者:** Jinhang Chai `[一作]` (Princeton University), Yujun Yan `[通讯]` (Dartmouth College)

**通讯引用:** 1758 | [OpenAlex ID](https://openalex.org/A5100688739)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种在环境维度与内部表示共同扩张的情形下，利用已学习的低秩+稀疏结构进行迁移学习的框架，并给出了锚定交替投影估计器；

**💡 创新点**

创新点在于：①将源任务的低秩子空间与稀疏编辑通过零填充嵌入到目标更大空间，保持子空间不变只学习少量创新方向；②通过分解误差为目标噪声、表示增长误差和源估计误差，证明在创新方向和稀疏编辑较少时可获得严格改进的统计收敛率；

**🔧 技术方法**

核心技术包括：锚定低秩投影、稀疏编辑投影、交替投影算法；理论分析使用奇异值分解、投影矩阵、非随机误差分解和马尔可夫链频率矩阵的混合时间收敛；

**📊 数据集**

主要在离散马尔可夫链的单轨迹数据以及结构化协方差估计实验中检验；

**📈 对比分析**

与直接在目标空间全量估计（无迁移）相比，实验与理论均显示误差下降，尤其在新特征仅引入少量创新方向时性能提升显著；

**⚠️ 局限性**

局限在于：①假设源与目标子空间可通过零填充嵌入，若嵌入不明确或有旋转不确定性需额外对齐；②对低秩/稀疏结构的假设较强，且需已知创新维度与稀疏编辑数；③在强相关噪声或非马尔可夫依赖的实际数据中理论条件可能不满足。

---

## 504. Trajectory-Guided Diffusion for Foreground-Preserving Background Generation in Multi-Layer Documents

**arXiv ID:** 2601.21857 | [PDF](https://arxiv.org/pdf/2601.21857v1)

**作者:** Taewon Kang `[一作]` (University of Maryland), Taewon Kang `[通讯]` (University of Maryland)

**通讯引用:** 7327 | [OpenAlex ID](https://openalex.org/A5004918445)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于扩散模型的文档背景生成框架，通过控制潜在空间轨迹实现前景保留与多页风格一致。

**💡 创新点**

将扩散视作潜在空间中的状态空间动力学，用缓存风格方向和轨迹控制实现前景自适应保护和风格一致性，无需显式遮罩或后处理。

**🔧 技术方法**

扩散模型的状态空间控制（SSC）、风格银行（Style Bank）、潜在空间轨迹设计以及能量/热力学视角。

**📊 数据集**

自建的学术论文与演示文稿三页样本集，用于评估前景可读性和多页一致性。

**📈 对比分析**

与BAGEL和GPT‑5基线对比，采用人工评分、自动指标（WCAG、OCR、CLIP多页一致性）和用户调查；在多项指标上均优于基线，前景可读性达98.12% WCAG，CLIP一致性0.6785。

**⚠️ 局限性**

前景稳定性仅靠软轨迹控制，可能在边界处产生细微伪影；风格方向固定，难以实现渐进或层级风格变化。

---

## 505. Bridging Forecast Accuracy and Inventory KPIs: A Simulation-Based Software Framework

**arXiv ID:** 2601.21844 | [PDF](https://arxiv.org/pdf/2601.21844v1)

**作者:** So Fukuhara `[一作]` (Halmstad University), Slawomir Nowaczyk `[通讯]` (Halmstad University)

**通讯引用:** 1871 | [OpenAlex ID](https://openalex.org/A5032811876)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并评估了一个基于仿真的决策中心框架，将需求预测模型与库存控制结合，系统性评估预测对运营 KPI（如总成本、服务水平）的影响。

**💡 创新点**

创新点包括：①构建可插拔预测模块与库存仿真的闭环系统；②公开开源软件与完整实验配置，方便复现；③通过实验揭示传统预测准确度指标（MAE、RMSE 等）与库存成本之间的非正相关性，提出以业务绩效为导向的模型选择思路。

**🔧 技术方法**

使用了合成需求生成器（基于可靠性模型与季节性 RBF）、多种预测算法（XGBoost、Random Forest、SVR、ARIMA、Croston、TSB、SBA）以及离散事件库存仿真，整体实现基于 Python 的框架。

**📊 数据集**

使用了 48 套三年长度的合成需求序列，特征贴合汽车后市场零件需求（间歇性、季节性、概念漂移等），未使用真实业务数据，但公开了所有生成脚本和数据。

**📈 对比分析**

对每个模型计算 MAE、RMSE、R²、IAE 并通过库存仿真得到总成本和服务水平；实验结果显示 XGBoost/Random Forest 在预测误差上表现最好，但导致的总库存成本最高；相反，Croston 等传统间歇需求方法在成本上表现最优，说明准确度与成本并不必然正相关。

**⚠️ 局限性**

局限性：仅考虑单层库存网络，所有零件共享相同参数，未涉及多层级、多供应商、批量订货等现实因素；模拟简化导致对复杂真实场景的泛化能力受限。

---

## 506. GAZELOAD A Multimodal Eye-Tracking Dataset for Mental Workload in Industrial Human-Robot Collaboration

**arXiv ID:** 2601.21829 | [PDF](https://arxiv.org/pdf/2601.21829v1)

**作者:** Bsher Karbouj `[一作]` (TU Berlin), Jorg Kruger `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了一个多模态眼动数据集GAZELOAD，用于工业人机协作中的心理负荷估计；在实验室装配平台上记录26名受试者佩戴Meta ARIA智能眼镜进行与UR5与Franka Emika Panda机器人交互的眼动、环境光照、任务与机器人日志数据，并提供250ms窗口聚合的眼动指标及自评心理负荷评分。

**💡 创新点**

首次提供针对工业人机协作环境的公开眼动负荷数据，结合机器人交互情境、环境因素（照度）与主观负荷评定，填补了现有数据集缺乏工业真实情境与任务上下文的空白。

**🔧 技术方法**

使用Meta ARIA智能眼镜实现眼动、视频、IMU、音频及光照传感；通过像素转毫米校准、I‑VT速度阈值法提取注视与扫视；数据按250 ms窗口聚合，并与环境光照、事件日志时间戳对齐。

**📊 数据集**

主要使用自建的GAZELOAD数据集；文中对比了相关公开数据集（Pillai, COLET, EM‑COGLOAD）但未直接使用它们进行实验。

**📈 对比分析**

该论文本身未开展算法比较或性能评估，而是将数据集公开为基准资源，鼓励后续研究在此数据上实现并比较心理负荷估计算法。

**⚠️ 局限性**

局限在于实验仅在单一实验室、固定工作台、仅使用两种协作机器人进行；参与者为有限年龄段的大学生；仅提供眼动及日志数据，缺乏EEG、HRV等多模态生理信号；因此数据的普适性与跨域迁移能力受限。

---

## 507. CORE:Toward Ubiquitous 6G Intelligence Through Collaborative Orchestration of Large Language Model Agents Over Hierarchical Edge

**arXiv ID:** 2601.21822 | [PDF](https://arxiv.org/pdf/2601.21822v1)

**作者:** Zitong Yu `[一作]` (Beijing University of Posts and Telecommunications), Xing Zhang `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 12106 | [OpenAlex ID](https://openalex.org/A5100399117)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 CORE 框架，利用多角色 LLM 代理在 6G 层级边缘网络中进行协同推理与任务分解，提升复杂任务的完成率与实时性。

**💡 创新点**

核心创新包括：① 基于角色亲和度的动态调度算法（DynaRole‑HEFT）实现资源最优分配；② 模型上下文协议（MCP）和 DAG 机制实现跨设备协作与上下文一致性；③ 采用流水线并行执行与实时多模态感知，显著降低延迟。

**🔧 技术方法**

技术组合包括 6G Ultra‑Reliable Low‑Latency Communication、集成感知与通信（ISAC）、多模态 LLM 推理、边缘计算、模型上下文协议（MCP）、DAG 任务调度、流水线并行执行、角色亲和度调度。

**📊 数据集**

实验使用公开的车载火灾视频数据集（Furg Fire Dataset）和 300 条人工整理的 LLM 任务指令。

**📈 对比分析**

与单一代理方法（ReAct、LLMCompiler）及多代理静态调度方法（Static_Dual_Loop、Crew_Ai）对比，CORE 在所有难度级别下的任务完成率提升约 25–52%，且在高负载下端到端延迟显著下降（相对 HEFT 降低 52%）。

**⚠️ 局限性**

主要限制是协同调度器自身的计算开销，导致在极低延迟场景（如远程手术）下仍有 25–40% 的时间占用；此外硬件平台的可迁移性与轻量级调度设计仍需进一步研究。

---

## 508. Nonparametric LLM Evaluation from Preference Data

**arXiv ID:** 2601.21816 | [PDF](https://arxiv.org/pdf/2601.21816v1)

**作者:** Dennis Frauen `[一作]` (LMU Munich), Stefan Feuerriegel `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个非参数框架 GARS，用以从偏好数据估计 LLM 的排名并给出置信区间，同时设计了基于成本约束的最优标签采集策略。

**💡 创新点**

创新点在于：①将排名目标定义为非参数函数 F(μ)，并利用 DML 推导出效率影响函数（EIF）实现去偏估计；②支持多种排名度量（BT、Borda、Rank Centrality）并统一处理；③提出基于成本的 A‑optimal实验设计，用以在有限预算下最小化估计方差。

**🔧 技术方法**

核心技术包括：双重机器学习（DML）与效率影响函数、交叉拟合、黑盒预测器融入、A‑optimal实验设计与置信区间构造。

**📊 数据集**

使用了合成数据、Chatbot Arena 与 MT‑Bench 的真实偏好数据，并结合预训练的评估模型（LLM‑as‑judge / auto‑rater）做实验。

**📈 对比分析**

与传统插件估计器以及假设 BT 模型下的去偏估计器比较，实验表明去偏 GARS 估计在 MSE、置信区间覆盖率上均优于插件，并且在预算约束下的采集策略能显著降低误差。

**⚠️ 局限性**

限制在于：需要准确估计偏好概率 μ 与选择概率 π；在数据稀疏或极端偏好情形下可能出现估计偏差；A‑optimal策略对 μ 的估计敏感，若误估会影响成本分配。

---

## 509. ECSEL: Explainable Classification via Signomial Equation Learning

**arXiv ID:** 2601.21789 | [PDF](https://arxiv.org/pdf/2601.21789v1)

**作者:** Adia Lumadjeng `[一作]` (University of Amsterdam), Erman Acar `[通讯]` (University of Amsterdam)

**通讯引用:** 287 | [OpenAlex ID](https://openalex.org/A5042454545)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种可解释的分类方法 ECSEL，通过学习符号方程（signomial）直接给出可解释的分类器。

**💡 创新点**

创新点在于利用 signomial 的结构性优势，既能高效恢复符号回归基准中的目标方程，又能在分类任务中保持良好预测性能，同时提供全局、边界和局部可解释性。

**🔧 技术方法**

采用基于梯度的优化与稀疏正则化（ℓ1）学习多项式幂函数，结合 softmax（或 sigmoid）得到分类概率；对符号回归任务改为 MSE 损失，使用多启动 L-BFGS / Adam 分段优化。

**📊 数据集**

数据集包括 AI Feynman、Livermore、Jin、Korns 等符号回归基准，以及 11 个常见分类数据集（如 ILPD、Compas、Transfusion 等），以及两个案例研究数据集：Online Shoppers Intention 与 PaySim。

**📈 对比分析**

与 DGSR、NGGP、NeSymRes 等符号回归方法对比，ECSEL 在 45 条 signomial 方程上恢复率达到 95.86%，显著高于 59‑60%，且平均求解时间约 86 秒，比对手快 4–7 倍；在分类任务中与 Logistic Regression、Random Forest、XGBoost、SVM、MLP 等基线比较，ECSEL 的准确率和 F1 分数均在 9/11 个数据集内与最优方法相差 1% 以内，且在多数样本不平衡数据集上拥有更高的少数类召回率。

**⚠️ 局限性**

主要局限在于需预先设定项数 K，且在高阶单变量多项式（如 Nguyen 数据集）中符号恢复效果不如传统 GP；对相关特征的稀疏性稳定性、以及对回归任务的推广仍需进一步研究。

---

## 510. Differentiable Knapsack and Top-k Operators via Dynamic Programming

**arXiv ID:** 2601.21775 | [PDF](https://arxiv.org/pdf/2601.21775v1)

**作者:** Germain Vivier-Ardisson `[一作]` (Google DeepMind), Mathieu Blondel `[通讯]` (Google DeepMind)

**通讯引用:** 66441 | [OpenAlex ID](https://openalex.org/A5049123454)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了通过动态规划对Knapsack和Top‑k问题进行可微松弛的统一框架，使得这类离散选择算子可以直接嵌入神经网络中；

**💡 创新点**

创新点在于将最大运算用严格凸正则化（如Shannon熵、Gini、Tsallis等）平滑，证明Shannon熵唯一实现置换等变性，并给出正则化诱导稀疏选择的必要与充分条件；

**🔧 技术方法**

使用的技术包括动态规划（max,+）递归、熵正则化的Smoothed Max、Fenchel‑Young 损失、向量‑雅可比乘积（VJP）反向传播、并行wavefront实现以及可采样的自回归分布；

**📊 数据集**

在实验中使用了决策聚焦学习基准、受限动态配置强化学习环境（20个物品，80步）以及离散VAE（MNIST数字组合）等数据集；

**📈 对比分析**

与六大基线（梯度估计、PPO、贪心、专家、FSD等）比较，结果显示在相对后悔、期望收益、梯度方差和训练稳定性方面均优于或匹配基线，且在大部分任务上计算效率更高；

**⚠️ 局限性**

局限性包括：仍需手动选择并调节正则化参数；稀疏正则化可能导致求解更耗时；DP的时间/空间复杂度为O(nC)或O(nk)，在极大规模问题上仍受限；目前仅处理单目标、线性容量约束，无法直接扩展到多目标或非凸约束。

---

## 511. WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents

**arXiv ID:** 2601.21872 | [PDF](https://arxiv.org/pdf/2601.21872v1)

**作者:** Yao Zhang `[一作]` (LMU Munich), Volker Tresp `[通讯]` (LMU Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于原理引导的推理先行进程奖励模型 WebArbiter，能够通过结构化文本生成给出可审计的步级判断；

**💡 创新点**

创新点在于将奖励建模转化为推理链生成，结合原理提取与强化学习，使模型既可解释又能适应动态页面；

**🔧 技术方法**

使用了推理蒸馏、基于 RL 的奖励校准、Transformer 解码器以及自定义的原理生成策略；

**📊 数据集**

在 WebPRMBench 上进行评估，包含四个 Web 环境（Mind2Web、WebArena、AssistantBench、WorkArena）的1150条偏好实例；

**📈 对比分析**

与多类基线（LLM‑as‑judge、开源 LLM、现有 WebPRM）比较，WebArbiter-7B 在 Pairwise Acc 与 Best‑of‑N Acc 上均达到 9.1 分和 7.2 分的领先优势；

**⚠️ 局限性**

局限性包括需要大量结构化推理数据、对原理模板的依赖、在极端页面布局变化时可能仍需进一步鲁棒性验证。

---

## 512. Cellular Automaton Reducibility as a Measure of Complexity for Infinite Words

**arXiv ID:** 2601.21862 | [PDF](https://arxiv.org/pdf/2601.21862v1)

**作者:** Markel Zubia `[一作]` (Ruhr University Bochum), Herman Geuvers `[通讯]` (Radboud University Nijmegen)

**通讯引用:** 1507 | [OpenAlex ID](https://openalex.org/A5071639019)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文提出通过一维细胞自动机（1CA）可约性来度量无穷字串的复杂度，并以此构建一个阶层结构。

**💡 创新点**

创新点在于将传统的可约性概念迁移到1CA模型，揭示了该模型能区分周期流、稀疏流等不同类别并构造出无限降链、原子度等新的代数性质。

**🔧 技术方法**

采用细胞自动机局部规则、邻域递归、构造证明（如对角化、插入/删除有限字串）等技术来定义可约性与度数，并给出判定算法的伪代码。

**📊 数据集**

本研究为纯理论性工作，未使用具体实验数据集，所有结果均通过形式化证明得出。

**📈 对比分析**

方法相较于Mealy机和有限状态变压器可约性更细粒度，能区分更多度数；但在实验评估上没有给出性能指标，而是通过示例和构造证明展示其优越性。

**⚠️ 局限性**

局限性包括判定问题不可判定、缺乏有效的算法实现、未探讨最大度数的存在性以及对更一般模型（如双向CA、有限词输出CA）的扩展。

---

## 513. Optimal Software Pipelining using an SMT-Solver

**arXiv ID:** 2601.21842 | [PDF](https://arxiv.org/pdf/2601.21842v1)

**作者:** Jan-Willem Roorda `[一作]` `[通讯]` (Intel Corporation), Jan-Willem Roorda (Intel Corporation)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

为VLIW处理器的循环生成最优的软件流水线调度。

**💡 创新点**

首次使用SMT求解器实现最优软件流水线，并利用不满足子集（unsat‑core）为程序员和处理器设计者提供调度可行性反馈。

**🔧 技术方法**

基于SMT求解器的约束求解，结合可选的寄存器压力约束，实现完整的循环调度与寄存器使用建模。

**📊 数据集**

使用两代无线通信用VLIW处理器固件中的400+个循环（包含33个信号处理核）进行实验。

**📈 对比分析**

与之前的启发式+手工优化方法相比，80%的核得到更优调度，最大加速率为1.22倍，几何平均加速率为1.08倍。

**⚠️ 局限性**

算法复杂度指数级，编译时间可达数秒到数分钟；若寄存器压力超限需要重新调度，且在极大循环上仍可能受限。

---

## 514. Trustworthy Intelligent Education: A Systematic Perspective on Progress, Challenges, and Future Directions

**arXiv ID:** 2601.21837 | [PDF](https://arxiv.org/pdf/2601.21837v1)

**作者:** Xiaoshan Yu `[一作]` (Anhui University), Xingyi Zhang `[通讯]` (Anhui University)

**通讯引用:** 18358 | [OpenAlex ID](https://openalex.org/A5028634381)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对智能教育系统的可信度问题进行系统综述，构建了五大任务分类（学员能力评估、学习资源推荐、学习分析、内容理解、教学辅助）以及五大可信维度（安全与隐私、鲁棒性、公平性、可解释性、可持续性），并对相关研究方法与技术进行梳理与分类；

**💡 创新点**

创新点在于：①首次将智能教育任务与可信度维度统一成双层分类框架；②提出跨任务可信度交叉、长期可信度建模与多模态可信教育等前沿研究方向；③为后续实证研究提供了明确的概念结构和研究空白；

**🔧 技术方法**

主要采用文献综述与结构化分类方法，对已有工作进行归纳整理，并使用示意图展示任务与可信维度的关系；

**📊 数据集**

由于是综述工作，未使用具体数据集；

**📈 对比分析**

本工作不做算法实现或实验对比，而是通过对比分析已发表方法的研究思路、技术路线和研究范围，指出其优缺点及适用场景；

**⚠️ 局限性**

局限性包括：①缺乏统一评估指标与基准，难以量化比较不同方法的可信度表现；②重点关注单任务与单维度，跨任务与多维度交互的研究仍不足；③对真实多模态教育数据与动态变化环境的可信度评估缺乏系统探讨；

---

## 515. Moral Outrage Shapes Commitments Beyond Attention: Multimodal Moral Emotions on YouTube in Korea and the US

**arXiv ID:** 2601.21815 | [PDF](https://arxiv.org/pdf/2601.21815v1)

**作者:** Seongchan Park `[一作]` (Korea Advanced Institute of Science and Technology), Wonjae Lee `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 18393 | [OpenAlex ID](https://openalex.org/A5100638533)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究构建并公开了适用于韩语和英语的多模态道德情感分类器，探讨在YouTube新闻视频中不同道德情感（尤其是他人谴责/愤怒）如何影响用户的观看、点赞和评论等不同层级的参与度。

**💡 创新点**

创新点在于：①首次将图像（缩略图）与文本（标题）结合进行道德情感识别，②提供跨文化（韩国与美国）的对比研究，③揭示“他人谴责”情感对参与度的递进提升效应。

**🔧 技术方法**

使用了多模态大型语言模型（MLLM）如Gemini‑2.5‑Flash‑Lite、Qwen2‑VL‑7B‑Instruct，并在其上进行零样本、少样本及微调实验；同时采用了BERTopic、CLIP、UMAP、HDBSCAN等文本与视觉嵌入与聚类工具。

**📊 数据集**

数据集为2024年1–12月间来自7个韩国和16个美国新闻频道的397,897条YouTube视频，包含缩略图、标题、描述、时长、上传日期及观看、点赞、评论计数；并在此基础上人工标注了1,276条韩语和900条英语的多模态样本。

**📈 对比分析**

与文本单一分类基线（BERT标题分类）相比，微调后的多模态模型在韩语上准确率0.898、F1 0.801，英语上准确率0.799、F1 0.631；零样本/少样本模式表现次之。负二项回归表明“他人谴责”情感的概率每增加1%对应观看增4%/点赞增16%/评论增73%（韩国）或近倍/两倍/三倍（美国）。

**⚠️ 局限性**

局限性：①为观察性分析，无法断定因果关系；②英语标注一致性低于韩语，可能影响模型效果；③模型仍可能继承预训练模型的西方偏见；④未探讨平台如何调节或缓解情感极化。

---

## 516. BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics

**arXiv ID:** 2601.21800 | [PDF](https://arxiv.org/pdf/2601.21800v1)

**作者:** Dionizije Fa `[一作]` (Entropic), Mateo Čupić `[通讯]` (Entropic)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 BioAgent Bench 评估套件，测试 LLM 代理在端到端生物信息学工作流中的表现

**💡 创新点**

创新点包括可验证的多步骤任务设计、鲁棒性扰动测试以及基于 LLM 的自动评判器

**🔧 技术方法**

使用 LLM 代理（如 Claude Opus、Gemini 3 Pro 等）、工具调用、LLM 评判器与多种 harness 架构

**📊 数据集**

采用 10 个典型生物信息学任务（RNA‑seq、变异检测、宏基因组等），配合相应输入与参考数据集

**📈 对比分析**

通过与闭源与开源模型在同一任务上的完成率对比，闭源模型完成率最高达 100%，开源模型平均约 82%，展示出明显性能差距

**⚠️ 局限性**

主要限制包括评判主观性与偏差、数据集规模与多样性受限、鲁棒性实验样本单一以及评估对任务细节的敏感性

---

## 517. CG-MLLM: Captioning and Generating 3D content via Multi-modal Large Language Models

**arXiv ID:** 2601.21798 | [PDF](https://arxiv.org/pdf/2601.21798v1)

**作者:** Junming Huang `[一作]`, Weiwei Xu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种多模态大语言模型（MLLM），能够端到端地对文本、图像与3D对象进行描述与高分辨率3D生成，统一实现空间感知与生成。

**💡 创新点**

创新点在于将 TokenAR 与 BlockAR 两种 Transformer 结合为 Mixture‑of‑Transformers 架构，实现序列与块级并行建模；并将预训练 Qwen3‑VL 与 Hunyuan3D‑2.1‑VAE 无缝集成，突破传统 2D 视角限制，实现单模型内高质量 3D 生成与空间一致性。

**🔧 技术方法**

技术手段包括：Token‑Level Autoregressive 与 Block‑Level Autoregressive Transformer、Hybrid Receptive Field（混合掩码）、视觉‑语言预训练模型（Qwen3‑VL）、3D VAE（Hunyuan3D‑2.1）、分阶段分辨率训练、CFG 引导、以及基于 RoPE/Interleaved MRoPE 的位置编码。

**📊 数据集**

使用的数据集为 LLaVA‑OneVision、Trellis‑500K、Objaverse++（挑选美学评分最高的样本）以及 Objaverse‑MIX 渲染图像，构成多模态训练与评估素材。

**📈 对比分析**

评估方法采用 p‑FID、p‑KID、CLIP‑IQA+、MUSIQ、Uni3D、CLIP 等指标，对比 SAR3D、ShapeLLM‑Omni 等同类 3D 生成与标注模型，实验表明本模型在 p‑FID、p‑KID、CLIP‑IQA+、Uni3D 等多项指标均为最佳或第二佳，且在 3D 标注的 BLEU/ROUGE/METEOR 指标上取得第一名。

**⚠️ 局限性**

局限性包括：对输入模糊或误导提示易产生错误或幻觉；依赖 Hunyuan3D‑2.1‑VAE 的点云重构导致几何细节受限，token 数量不足；与商业 3D 生成系统相比整体质量仍有差距，需要更轻量化、高效的 3D VAE 与更丰富的数据支持。

---

## 518. Knowledge Vector Weakening: Efficient Training-free Unlearning for Large Vision-Language Models

**arXiv ID:** 2601.21794 | [PDF](https://arxiv.org/pdf/2601.21794v1)

**作者:** Yejin Kim `[一作]` (Sogang University), Junsuk Choe `[通讯]` (Sogang University)

**通讯引用:** 6326 | [OpenAlex ID](https://openalex.org/A5078314427)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无需梯度、训练或 LoRA 的无学习方法（Knowledge Vector Weakening，KVW），通过在前向传播中识别并弱化 MLP 中的知识向量，从大规模视觉‑语言模型中去除指定知识。

**💡 创新点**

创新点在于：1) 直接在模型内部弱化知识向量，完全不依赖反向传播或参数微调；2) 通过比较忘记集和保留集的知识系数，构造 Forget Knowledge Accessor（FKA）以精确定位要弱化的向量；3) 使用指数门函数按比例缩放向量，实现可控、渐进的知识去除，提升忘记‑保持平衡与计算效率。

**🔧 技术方法**

技术实现包括：前向传播；FFN 关键‑值记忆解释；计算知识系数并对忘记集/保留集做对比；构造 FKA 并使用门函数 g(·)=exp(-γ·A) 缩放知识向量；不增加额外参数或梯度存储。

**📊 数据集**

使用 MLLMU‑Bench（500 合成个人档案 + 153 公共名人档案）和 CLEAR（200 合成个人档案，约 3,700 张图片）两个视觉‑语言无学习基准，评估 VQA、图像字幕等任务。

**📈 对比分析**

与 GA、GD、KL、NPO、MMU 等梯度/LoRA 方法在 2‑fold 交叉验证下比较，要求保留性能≥95%。实验表明：KVW 在忘记准确率接近 oracle 且保持性能稳定；相比梯度/LoRA 方法，KVW 在 FLOPs、运行时间和显存使用上均显著更优，提升了计算效率。

**⚠️ 局限性**

局限性：仅对单个知识向量进行弱化，未考虑知识向量之间的交互与组合；后续可探索结构化弱化策略或轻量级后适配以进一步提升模型的整体表现。

---

## 519. Preliminary Results of a Scoping Review on Assistive Technologies for Adults with ADHD

**arXiv ID:** 2601.21791 | [PDF](https://arxiv.org/pdf/2601.21791v1)

**作者:** Valerie Tan `[一作]` (TU Dortmund University), Max Pascher `[通讯]` (TU Dortmund University)

**通讯引用:** 311 | [OpenAlex ID](https://openalex.org/A5061109375)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开展了一项基于PRISMA‑ScR的系统性文献综述，筛选并深入分析了46篇关于成人ADHD辅助技术的研究。

**💡 创新点**

首次聚焦成人ADHD的技术支持领域，揭示了研究重心偏向治疗和认知训练而非赋能，指出设计参与度不足与技术实用性评估缺失。

**🔧 技术方法**

采用了构造式检索查询、R包litsearchr生成关键字、PRISMA‑Scope指南、Covidence软件进行标题-摘要与全文筛选，并手工对关键词进行类别划分。

**📊 数据集**

使用了Scopus、ACM Digital Library、IEEE Xplore三大学术数据库的公开文献记录，未涉及特定实验数据集。

**📈 对比分析**

没有进行算法或系统性能对比；通过定量统计（发表年份、期刊来源、技术类型分布等）展示了研究趋势与主题分布。

**⚠️ 局限性**

局限包括：仅检索英文文献、样本主要为短论文/海报而非完整研究、缺乏参与者中心设计与可用性评估、对成人ADHD的研究数量相对有限。

---

## 520. MoHETS: Long-term Time Series Forecasting with Mixture-of-Heterogeneous-Experts

**arXiv ID:** 2601.21866 | [PDF](https://arxiv.org/pdf/2601.21866v1)

**作者:** Evandro S. Ortigossa `[一作]` (Weizmann Institute of Science), Eran Segal `[通讯]` (Weizmann Institute of Science)

**通讯引用:** 73199 | [OpenAlex ID](https://openalex.org/A5012450539)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于Transformer的长周期多变量时间序列预测框架MoHETS，采用稀疏混合异构专家(MoHE)以及跨模态注意力和轻量级卷积解码器，实现高效预测。

**💡 创新点**

核心创新在于用共享深度卷积专家处理全局趋势、用路由的傅里叶专家捕捉局部周期性；结合跨模态注意力注入外生变量；并将线性投影头替换为卷积解码器，提升参数效率与稳定性。

**🔧 技术方法**

技术包括Encoder‑only Transformer、RoPE位置编码、FlashAttention‑2与GQA自注意力、MoHE（深度卷积+傅里叶专家）、跨模态多头注意力、RMSNorm与组归一化、Huber损失与专家负载平衡损失、TF32训练。

**📊 数据集**

在七个公开基准数据集（ETTh1/2、ETTm1/2、Weather、ECL、Traffic）上进行实验，覆盖多种时序分辨率与变量维度。

**📈 对比分析**

与15个最新基线（TimeXer、SOFTS、PatchTST、Crossformer等）对比，MoHETS在所有预测时段（96/192/336/720）平均MSE下降约12%，在多数数据集上取得首位或第二位。

**⚠️ 局限性**

局限性包括：仅使用日历等有限外生变量；专家路由可能出现负载不均导致训练不稳定；模型仍相对复杂，推理时需显存；对极端异常值或高噪声时序的鲁棒性尚待进一步验证。

---

## 521. Adaptive Privacy of Sequential Data Releases Under Collusion

**arXiv ID:** 2601.21859 | [PDF](https://arxiv.org/pdf/2601.21859v1)

**作者:** Sophie Taylor `[一作]` (University of Oxford), Justin Coon `[通讯]` (University of Oxford)

**通讯引用:** 3771 | [OpenAlex ID](https://openalex.org/A5001680231)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种针对多方顺序数据请求的隐私-实用性权衡框架，考虑单方泄漏与协同泄漏约束，并给出了适应性数据发布算法。

**💡 创新点**

创新点在于：①将协同泄漏纳入隐私约束，支持多方请求；②在求解过程中将问题转化为双重优化并利用改进的 Blahut–Arimoto 算法；③将隐私-实用性优化与信息瓶颈理论和渐进神经网络的设计联系起来。

**🔧 技术方法**

核心技术是改进的 Blahut–Arimoto 算法（对期望失真和互信息两种实用性度量分别处理）、Lagrange 对偶求解、二分搜索/时间共享插值求解特定约束下的最优发布策略。

**📊 数据集**

实验采用了一个简化的二元联合分布（表 1 中的 8 维离散分布）和汉明失真函数，演示了失真-隐私-协同曲线与互信息-隐私-协同曲线的构造。

**📈 对比分析**

通过数值模拟验证了算法收敛性，并在期望失真目标下取得了全局最优解；在互信息目标下得到的是局部最优下的下界曲线。相较于传统单方或无协同的隐私机制，该方法在保持相同隐私预算时能显著降低失真或提高互信息。

**⚠️ 局限性**

局限性包括：①互信息目标非凸，算法仅能保证局部最优；②实现时需多次随机初始化和时间共享插值，计算成本随请求数增大；③只考虑了“所有方协同”最坏情况，未细化对部分子集协同的约束，且未在真实大规模数据库上验证。

---

## 522. Self-Adaptive Probabilistic Skyline Query Processing in Distributed Edge Computing via Deep Reinforcement Learning

**arXiv ID:** 2601.21855 | [PDF](https://arxiv.org/pdf/2601.21855v1)

**作者:** Chuan-Chi Lai `[一作]` (National Chung Cheng University), Chuan-Chi Lai `[通讯]` (National Chung Cheng University)

**通讯引用:** 114 | [OpenAlex ID](https://openalex.org/A5082920277)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了 SA-PSKY 框架，在边缘云环境中通过自适应阈值过滤实现概率天际线查询的分布式处理

**💡 创新点**

创新点在于将阈值选择建模为连续动作的马尔可夫决策过程，并使用深度确定性策略梯度（DDPG）实现全局最优阈值的自学习和动态调节，克服了传统固定阈值在不确定数据流和网络波动下的性能退化

**🔧 技术方法**

核心技术包括：概率天际线查询、滑动窗口数据流处理、M/M/1 排队模型、DDPG 强化学习（Actor‑Critic、目标网络、优先经验回放、OU 噪声）以及基于多目标加权的奖励设计

**📊 数据集**

使用合成的 5 台边缘节点与 1 个云代理共 50,000 条不确定对象的数据集（每对象 3-9 个离散实例，3-9 维属性），并在不同实例数、维度、网络带宽等场景下进行敏感性测试

**📈 对比分析**

与传统集中式全量传输（No‑Filtering）和固定阈值（α=0.02）两种基线对比，SA-PSKY 在默认设置下端到端延迟显著下降约 70%（从 273 s 降到 82 s），且在高不确定度和高维度场景下保持稳定的性能提升

**⚠️ 局限性**

局限性包括：仅在单一云代理架构下验证；采用离散实例模型，未覆盖连续概率分布；对极端大规模节点网络的可扩展性与容错性尚待进一步研究

---

## 523. Scalable Linearized Laplace Approximation via Surrogate Neural Kernel

**arXiv ID:** 2601.21835 | [PDF](https://arxiv.org/pdf/2601.21835v1)

**作者:** Luis A. Ortega `[一作]` (Universidad Autónoma de Madrid), Daniel Hernández-Lobato `[通讯]` (Universidad Autónoma de Madrid)

**通讯引用:** 2159 | [OpenAlex ID](https://openalex.org/A5003318791)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种可扩展的线性化拉普拉斯近似（LLA）核近似方法ScaLLA，使用代理神经网络学习NTK结构，避免显式计算Jacobian。

**💡 创新点**

创新点在于通过JVP训练的低维特征嵌入逼近NTK，并可通过偏置协方差提升OOD检测，同时实现LLA的可扩展性。

**🔧 技术方法**

使用Jacobian–vector products、代理网络、NTK逼近、上下文点偏置、Woodbury矩阵求逆。

**📊 数据集**

在FMNIST和MNIST（作为上下文）以及KMNIST（OOD）数据集上进行实验。

**📈 对比分析**

与LLA、LLLA、VaLLA、FMGP、MFVI、SNGP等方法对比，ScaLLA在NLL、ECE和in-distribution性能上与现有LLA变体相当，且偏置版本在OOD检测AUC上超过其他方法。

**⚠️ 局限性**

限制在于偏置协方差效果高度依赖于所选上下文点的代表性，若缺乏合适的OOD样本难以实现最佳性能。

---

## 524. Folklore in Software Engineering: A Definition and Conceptual Foundations

**arXiv ID:** 2601.21814 | [PDF](https://arxiv.org/pdf/2601.21814v1)

**作者:** Eduard Enoiu `[一作]`, Gregory Gay `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过文献综述与半结构化访谈，系统地定义并分析了软件工程中的民俗（SE民俗），提出了工作定义；

**💡 创新点**

创新点在于首次将民俗学理论引入软件工程，形成SE民俗的概念框架，并识别其叙事、仪式、幽默等多种表现形式；

**🔧 技术方法**

主要技术手段包括文献综述、主题分析（Braun & Clarke 方法）和对12名从业者的访谈录音进行编码与归纳；

**📊 数据集**

使用的数据集为12位瑞典工业软件工程师的访谈记录，涵盖多角色与不同组织背景；

**📈 对比分析**

由于研究为概念性与质性分析，并未进行传统意义上的性能对比评估，评估主要基于主题的一致性与理论匹配度；

**⚠️ 局限性**

局限性包括样本规模有限、取样偏向经验丰富者、缺乏纵向追踪以及研究者主观编码可能导致偏差。

---

## 525. The Double-Edged Sword of Knowledge Transfer: Diagnosing and Curing Fairness Pathologies in Cross-Domain Recommendation

**arXiv ID:** 2601.21805 | [PDF](https://arxiv.org/pdf/2601.21805v1)

**作者:** Yuhan Zhao `[一作]` (Hong Kong Baptist University), Weike Pan `[通讯]` (Shenzhen University)

**通讯引用:** 3872 | [OpenAlex ID](https://openalex.org/A5073490832)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对跨域推荐（CDR）系统中出现的群体公平性问题，本文通过理论与实验分析揭示两类公平性病理，并提出一种通用的 Cross‑Domain Fairness Augmentation (CDFA) 框架，利用未标记数据增强和信息增益再分配来缓解跨域失衡传递与信息增益不公平。

**💡 创新点**

创新点包括：
1) 结合 Wasserstein‑1 与 Lipschitz 约束给出 UGF 上的上界，首次系统地分析了跨域失衡传递与信息增益导致的公平性偏差；
2) 设计无模型、可插拔的两阶段机制：基于分组负采样的未标记数据增强，和基于信息理论的跨域信息增益估计与再分配，二者均不需要改动已有 CDR 架构；
3) 在多种基线与公平 CDR 方法上验证，该框架既能显著提升公平性，又能保持或提升整体推荐准确率。

**🔧 技术方法**

技术手段：
- 理论分析：Wasserstein‑1 距离、Kantorovich–Rubinstein 变换、Lipschitz 约束、Rademacher 复杂度；
- 实验方法：分组负采样（hard negative for disadvantaged groups），信息增益估计网络（基于 MLP 的联合表示），增益再分配损失；
- 训练损失：标准推荐损失 + 再分配损失 + 估计器对齐损失；
- 可与 CMF、CLFM、BiTGCF、CoNet、DTCDR 等多种 CDR 模型结合。

**📊 数据集**

数据集：
- QB：QQ 浏览器平台，包含视频与文章两域交互；
- QK：QQ 看板平台，同样是视频与文章两域交互；
（实验中对两域均使用了性别与年龄两种敏感属性的公平性评估）

**📈 对比分析**

对比方法：将 CDFA 插入 CMF、CLFM、BiTGCF、CoNet、DTCDR 等基础 CDR 模型，并与 FairCDR、VUG 等专门的公平 CDR 方法对比。实验结果显示：
- UGF（统一群体公平性度量）在所有基线上提升约 70%–100%；
- Recall@10 / Recall@20 / NDCG 等准确率指标保持不变或略有提升（最高提升 5%），且在大部分场景下改进显著超过 SOTA 公平方法；
- 与 SOTA 公平方法相比，CDFA 在公平性与准确率之间几乎无折衷。

**⚠️ 局限性**

局限性：
- 需要源域与目标域存在一定比例的重叠用户，否则信息增益估计与再分配效果受限；
- 分组负采样与增益再分配依赖超参数（M、ε、γ），需在特定数据集上调优；
- 理论分析基于独立同分布、Lipschitz 可测量等假设，实际数据中可能偏离；
- 目前仅在两种跨域场景（视频/文章）验证，尚未在更广泛的多域或代理式推荐系统中验证。

---

## 526. Effective LoRA Adapter Routing using Task Representations

**arXiv ID:** 2601.21795 | [PDF](https://arxiv.org/pdf/2601.21795v1)

**作者:** Akash Dhasade `[一作]` (Ecole Polytechnique Federale de Lausanne), Martijn de Vos `[通讯]` (Ecole Polytechnique Federale de Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关、任务级别的LoRA适配器路由框架LoRAuter，能够在不访问适配器训练数据的情况下，通过任务表示对公共适配器池进行高效检索与组合，支持大规模异构适配器的推理；

**💡 创新点**

创新点在于：①将路由视角从单个适配器转向任务级别，使用任务表示实现检索；②采用无训练的黑盒检索与融合；③利用Successive Halving快速定位每个任务最佳适配器；④通过输入感知的输出空间加权融合提升泛化；

**🔧 技术方法**

技术包括LoRA参数高阶低秩插值、任务验证集生成任务表示、句子嵌入（SupCon）、余弦相似度检索、Softmax归一化权重、Successive Halving搜索、输出空间加权融合；

**📊 数据集**

主要使用FlanV2 48个NLP任务（NLI、生成、翻译）与其验证集；此外还在HuggingFace上抓取1500+公开LoRA适配器进行大规模评估；

**📈 对比分析**

与四种最先进路由基线（LR、LA、ARROW、SpectR）以及Oracle进行对比；在非OOV场景LoRAuter达101.2% Oracle，OOV场景提升5.2个百分点；在大规模1500+适配器集上仍保持与48个精心挑选适配器相近的性能；

**⚠️ 局限性**

局限包括：①依赖任务验证集的可获取性；②任务表示仅基于少量验证样本，可能对任务多样性有限；③对输入-任务相似度的软最大化假设可能在极端OOV情况下失效；④未对安全性、偏见等问题进行深入评估。

---

## 527. Beyond the Finite Variant Property: Extending Symbolic Diffie-Hellman Group Models (Extended Version)

**arXiv ID:** 2601.21910 | [PDF](https://arxiv.org/pdf/2601.21910v1)

**作者:** Sofia Giampietro `[一作]` (ETH Zurich), David Basin `[通讯]` (ETH Zurich)

**通讯引用:** 12955 | [OpenAlex ID](https://openalex.org/A5025344654)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文扩展了符号协议分析工具Tamarin，使其能够完整地建模和分析Diffie–Hellman（DH）群中的所有运算，包括指数加法与乘法，从而实现对使用完整DH结构的协议（如ElGamal、MQV）的自动化安全验证。

**💡 创新点**

创新点在于：
• 采用近似的DH重写系统（继承自Dougherty & Guttman）以捕获字段代数特性；
• 将符号统一（不可判定）与代数求解（高斯消元/Gröbner基）相结合，得到一个半决策的约束求解规则；
• 在Tamarin中实现了对DH运算的完整支持，首次让现有主流工具处理包含指数加法的协议。

**🔧 技术方法**

核心技术包括：
• DH重写系统的构造与证明收敛性；
• 将DH项转化为多项式形式并用线性代数求解（高斯消元）；
• 通过约束求解规则C_@、C_Prem、C_=等替代传统统一，处理包含DH子项的约束；
• 结合用户自定义的等价理论与DH理论，采用清洗和合并方法。

**📊 数据集**

实验数据集：
• Tamarin仓库中的20个基准模型（12真实协议、8WireGuard）；
• 额外的ElGamal加密协议模型；
• MQV密钥交换协议模型（含未知密钥共享攻击）。

**📈 对比分析**

比较与性能：
• 与原始Tamarin（不支持DH乘法）相比，验证WireGuard所需时间约5倍；
• 对ElGamal的可执行性与保密性验证均在几分钟内完成，部分属性可在秒级完成；
• 对MQV的安全属性和攻击发现在几秒到半分钟内完成；
• 总体而言，扩展后工具仍能在合理时间内完成大多数实验，但在某些复杂协议上需要手动引导搜索。

**⚠️ 局限性**

局限性：
• 当前仅实现线性代数求解，无法处理非线性方程（如ElGamal签名、HMQV等）；
• 仅支持固定生成元的DH群，不能处理不同群或用户自定义生成元；
• 不能处理用户定义的DH函数符号（除非无方程）；
• 需要手动引导或自定义启发式才能在复杂协议上自动完成证明；
• 仅支持大多数Tamarin功能，尚未覆盖如diff-mode等高级特性。

---

## 528. Hardware-Triggered Backdoors

**arXiv ID:** 2601.21902 | [PDF](https://arxiv.org/pdf/2601.21902v1)

**作者:** Jonas Möller `[一作]` (Berlin Institute for the Foundations of Learning and Data), Konrad Rieck `[通讯]` (Berlin Institute for the Foundations of Learning and Data)

**通讯引用:** 9452 | [OpenAlex ID](https://openalex.org/A5066077721)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于硬件差异的后门攻击，利用不同GPU在浮点计算中的微小数值偏差，使同一模型在特定硬件上误分类而在其它硬件上保持正常；

**💡 创新点**

创新点在于引入硬件触发的后门概念，设计了两步攻击流程：先通过梯度优化将决策边界逼近目标样本，再通过隐式（权重拓扑置换）或显式（位翻转）方式放大硬件间数值偏差，成功实现对多种模型与GPU的硬件特定攻击；

**🔧 技术方法**

核心技术包括：梯度优化的决策边界逼近、隐式修改（拓扑置换）与显式修改（参数位翻转）、交替优化策略、跨硬件激活补丁的因果分析以及多种防御手段的评估；

**📊 数据集**

使用ImageNet数据集进行训练和测试，随机采样目标图像以保持原始性能；

**📈 对比分析**

实验结果显示：单目标时攻击成功率>90%，多目标时成功率随目标数递减，单机与多机对比下均保持≈99.8%的原始准确率；在不同浮点类型、批量大小、混合精度以及输入扰动下的鲁棒性被系统评估，主动微调可显著降低成功率；

**⚠️ 局限性**

局限性包括：对多目标或“one‑vs‑rest”触发时成功率下降；攻击依赖于特定硬件的微小差异，某些设备对称时难以触发；防御仅在主动微调或较大扰动时有效，随机批量或混合精度不一定能彻底阻止。

---

## 529. A Low-Complexity Plug-and-Play Deep Learning Model for Generalizable Massive MIMO Precoding

**arXiv ID:** 2601.21897 | [PDF](https://arxiv.org/pdf/2601.21897v1)

**作者:** Ali Hasanzadeh Karkan `[一作]` (Polytechnique Montréal), François Leduc-Primeau `[通讯]` (Polytechnique Montréal)

**通讯引用:** 703 | [OpenAlex ID](https://openalex.org/A5038571252)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一个低复杂度的插件即插即用深度学习预编码模型 PaPP，能够在不同基站场景、功率水平和信道估计误差下保持高谱效率。

**💡 创新点**

创新点在于将教师–学生蒸馏与自监督 Sum‑Rate 损失相结合，并利用元学习 MLDG 与功率感知归一化，使模型能够跨场景泛化、对传输功率无关，并在无需重新训练的前提下，仅用少量无标签样本即可微调。

**🔧 技术方法**

技术方法包括 CNN+MLP 结构的特征提取、教师‑学生蒸馏、自监督学习、MLDG 元学习、功率归一化、信道估计误差建模以及对 HBF 与 FDP 两种硬件架构的统一设计。

**📊 数据集**

实验使用基于蒙特利尔三维地图的射线追踪数据集，涵盖多站点 LOS / NLOS 传播条件，并将三套未见站点作为泛化测试。

**📈 对比分析**

与传统 ZF、WMMSE、PE‑AltMin、MAML‑CNN、DeepAll‑CNN 以及现场专用训练模型对比，PaPP 在零射、少样本和全微调三种模式下均优于基线；其能量消耗低约 20–21 倍，且在大多数场景中达到或超过 WMMSE 的谱效率，表现出更好的鲁棒性。

**⚠️ 局限性**

局限性包括：在极高信道估计误差（β≥0.5）下微调效果差；实验仅在 8×8 天线、N_U=4 的固定配置下验证；在极低 SNR（≈10 dB）时仍略逊于 WMMSE；需要高质量射线追踪或仿真数据才能训练有效模型。

---

## 530. WADBERT: Dual-channel Web Attack Detection Based on BERT Models

**arXiv ID:** 2601.21893 | [PDF](https://arxiv.org/pdf/2601.21893v1)

**作者:** Kangqiang Luo `[一作]` (Guangzhou Institute of Technology), Jing Pan `[通讯]` (Guangzhou Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种双通道Web攻击检测模型WADBERT，能够准确识别恶意HTTP请求并定位攻击参数。

**💡 创新点**

创新点包括：Hybrid Granularity Embedding (HGE)融合子词与字符特征，使用多头注意力捕获无序参数的组合关系，并通过注意力权重实现攻击追踪。

**🔧 技术方法**

主要技术是BERT变体（URLBERT、SecBERT）+ HGE嵌入 + 多头注意力融合 + 线性分类器。

**📊 数据集**

使用公开数据集CSIC2010和SR-BH2020进行实验。

**📈 对比分析**

与六个基准模型（EDL、CNN‑BiLSTM、BERT‑BiLSTM、DistilBERT、TransURL、PMANET）比较，WADBERT在两组数据上均取得最高准确率（99.70%/99.32%）和F1分数（99.63%/99.50%），提升幅度约0.5%–1.2%。

**⚠️ 局限性**

局限性在于模型仍依赖大量预训练BERT参数，推理成本较高，且对极少见或新型攻击模式的泛化能力待进一步验证。

---

## 531. Improving Classifier-Free Guidance of Flow Matching via Manifold Projection

**arXiv ID:** 2601.21892 | [PDF](https://arxiv.org/pdf/2601.21892v1)

**作者:** Jian-Feng Cai `[一作]` (Hong Kong University of Science and Technology), Chao Wang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 103042 | [OpenAlex ID](https://openalex.org/A5100339418)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出将流匹配中的分类器无导向（CFG）视为对光滑距离函数梯度的近似，并基于此设计了两种改进采样方法CFG‑MP和CFG‑MP+，在不修改模型参数的前提下提升采样质量与鲁棒性。

**💡 创新点**

创新点在于通过优化视角揭示预测差导致的指导尺度敏感性，进而引入流形投影来消除预测差，并利用安德森加速提升投影收敛速度，显著降低对指导尺度的依赖。

**🔧 技术方法**

核心技术包括梯度解析、光滑距离函数构造、增量式流形投影（迭代梯度下降）、安德森加速、连续时间ODE求解与流匹配模型的速度场学习。

**📊 数据集**

实验数据集包括ImageNet 256×256、DrawBench、Pick‑a‑Pic、GenEval；评估模型涵盖DiT‑XL‑2‑256、Flux‑dev 与 Stable Diffusion 3.5。

**📈 对比分析**

通过与多种CFG变体（DDIM、Z‑sampling、Re‑sampling、FSG、CFG‑D2F）对比，并采用FID、IS、CLIP、ImageReward、PickScore、HPSv2等指标，CFG‑MP/MP+ 在绝大多数指标上实现显著提升，图像质量、文本对齐和指导尺度鲁棒性均优于基线。

**⚠️ 局限性**

主要局限在于方法仅针对连续时间流匹配模型，尚未完全推广至离散时间扩散框架；投影迭代虽加速但仍有计算开销；对模型超参数的敏感性及理论解释的普适性仍需进一步研究。

---

## 532. Adaptive Surrogate-Based Strategy for Accelerating Convergence Speed when Solving Expensive Unconstrained Multi-Objective Optimisation Problems

**arXiv ID:** 2601.21885 | [PDF](https://arxiv.org/pdf/2601.21885v1)

**作者:** Tiwonge Msulira Banda `[一作]` (Robert Gordon University), Alexandru-Ciprian Zăvoianu `[通讯]` (Robert Gordon University)

**通讯引用:** 627 | [OpenAlex ID](https://openalex.org/A5003598524)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f`

**🎯 论文内容**

本文提出了一种适应性代理加速策略，利用在线训练的代理模型加速多目标进化算法在计算密集型问题中的早期收敛；

**💡 创新点**

创新点在于：①使用仅基于上一代真实评估数据训练代理，避免大规模存档和复杂筛选；②采用两循环架构，代理在满足性能阈值后自动退出；③将代理作为模块化加速器，无需显式参数化；

**🔧 技术方法**

采用的技术包括随机森林回归、Gaussian Process回归和一维卷积神经网络等机器学习代理模型；

**📊 数据集**

实验使用31个常见的多目标基准问题（DTLZ、KSW、LZ09、WFG、ZDT）和一个北海鱼类丰度评估的真实案例；

**📈 对比分析**

通过与标准NSGA-II和MOEA/D-DRA以及轻量插值代理（PE+SE）对比，使用超体积（HV）和IGD指标评估；结果显示，GPR和CNN代理在前10–30代实现平均提升≥20%（NSGA-II）或≥80%（MOEA/D-DRA），但在后期收敛时略逊；

**⚠️ 局限性**

局限性包括：代理在某些问题（如DWLZ1、3）下效果不佳；自动退出阈值和整合比例固定，可能不适用于所有问题；以及在多目标更高维场景中仍需进一步验证。

---

## 533. Multi-Modular MANTA-RAY: A Modular Soft Surface Platform for Distributed Multi-Object Manipulation

**arXiv ID:** 2601.21884 | [PDF](https://arxiv.org/pdf/2601.21884v1)

**作者:** Pratik Ingle `[一作]` (Information Technology University), Andres Faina `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计并验证了一种可扩展的多模块软面操控平台MANTA-RAY，利用低密度线性执行器通过柔性织物实现对多种形状、材质和易碎物体的平行、协调操控。

**💡 创新点**

通过模块化网格化执行器、共享边界执行器以及基于几何变换的低维PID控制器与对象传递策略，实现了大面积、低执行器密度下的多对象平行操控，并避免了高维数据驱动训练。

**🔧 技术方法**

软体机器人平台、织物柔性表面、线性执行器与步进电机驱动、Arduino控制器、磁编码器反馈、OptiTrack跟踪、MuJoCo仿真、几何变换+PID控制、对象传递算法。

**📊 数据集**

实验中自制的六类不同形状、质量、纹理的物体（球、立方体、盘、苹果、鸡蛋、圆柱体等）以及MuJoCo中仿真对象；未使用公开数据集。

**📈 对比分析**

通过实验与仿真比较单模块与多模块（2×2、3×3）平台的目标达成误差、对象传递成功率、平行操控的稳定性；平均定位误差<0.02m、传递成功率>95%，多模块平台实现了1×1m范围内多对象平行操控，性能优于传统高密度执行器平台。

**⚠️ 局限性**

仍受织物张力、摩擦、执行器共享导致的耦合影响；对极大尺寸或极高重量物体性能下降；缺乏在线学习或自适应调节，需人工调参；实验规模受限于2×2硬件，尚未验证更大规模真实环境。

---

## 534. From Generative Modeling to Clinical Classification: A GPT-Based Architecture for EHR Notes

**arXiv ID:** 2601.21955 | [PDF](https://arxiv.org/pdf/2601.21955v1)

**作者:** Fariba Afrin Irany `[一作]` `[通讯]`, Fariba Afrin Irany

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

采用选择性微调方式，将预训练的 GPT‑2 模型的前层冻结，只更新最后一层 Transformer、最终层归一化和分类头，实现对临床文本的分类。

**💡 创新点**

创新点在于显式量化冻结与可训练参数的比例（超过 94% 参数被冻结），并提供了完整的训练时复杂度与内存占用分析，证明了在有限计算资源下仍能获得高效的性能。

**🔧 技术方法**

技术包括 GPT‑2 生成式预训练、只训练最后 Transformer 块和分类头的选择性微调、基于 CheXpert 规则的弱监督标签生成、AdamW 优化、以及对不确定性和否定的多类别标注。

**📊 数据集**

使用了 MIMIC‑IV‑Note（约 680,000 条放射科报告），并通过规则抽取得到 CheXpert‑style 的不确定性标签。

**📈 对比分析**

在多标签、单标签（正/负）和聚合诊断任务上与全微调基线对比，选择性微调在 5k、50k、500k 样本下均能保持 90% 以上的准确率，且训练时间与显存显著降低。

**⚠️ 局限性**

局限性包括：对罕见病症的检测仍受样本稀疏影响；只更新最后一层可能无法捕获更深层次的临床语义；以及在仅使用规则抽取标签的弱监督设置下，标签噪声和不确定性处理仍需改进。

---

## 535. ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models

**arXiv ID:** 2601.21947 | [PDF](https://arxiv.org/pdf/2601.21947v1)

**作者:** Bowen Fang `[一作]` (New Laboratory of Pattern Recognition Institute of Automation Chinese Academy of Sciences), Liang Wang `[通讯]` (New Laboratory of Pattern Recognition Institute of Automation Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ToolWeaver框架，将工具表示为层次化代码序列以解决单一token过大、语义瓶颈的问题。

**💡 创新点**

创新点在于协作感知的分层向量量化和结构化token化，使工具库规模呈对数增长并能学习协作关系。

**🔧 技术方法**

采用协作感知残差向量量化（RQ‑VAE）、图拉普拉斯正则、生成对齐微调以及受约束beam搜索等技术。

**📊 数据集**

在ToolBench基准（约47000个API）上进行评测，并使用WikiText‑2、CNN/DailyMail、XSum验证语言能力。

**📈 对比分析**

与BM25、ToolRetriever、ToolGen等检索与生成方法对比，ToolWeaver在工具检索NDCG@k和端到端任务SoPR/SoWR上均显著优于对手，特别是多工具复杂任务。

**⚠️ 局限性**

局限在于仍需手工构建协作相似矩阵，且在极大规模工具库下量化模型训练成本高，且对外部工具更新需重新训练。

---

## 536. Clarity: The Flexibility-Interpretability Trade-Off in Sparsity-aware Concept Bottleneck Models

**arXiv ID:** 2601.21944 | [PDF](https://arxiv.org/pdf/2601.21944v1)

**作者:** Konstantinos P. Panousis `[一作]` (Athens University of Economics and Business), Diego Marcos `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本工作研究了稀疏感知概念瓶颈模型（CBM）的可解释性与性能之间的权衡，提出了新的可解释性度量（clarity）并构建了系统化的评估框架。

**💡 创新点**

创新点包括：①提出了综合稀疏度、概念预测精度与下游任务准确率的clarity指标；②设计了两种可摊销的ℓ0/ℓ1稀疏约束方法；③将该评估框架与VLM（CLIP）和手工标注的属性预测器相结合，系统比较不同稀疏化策略。

**🔧 技术方法**

使用的技术包括：概念瓶颈模型、Vision‑Language Model（CLIP）用于零样本概念评分、稀疏正则化（ℓ1、ℓ0 via Hard Concrete、Bernoulli/Concrete分布）、摊销矩阵学习、交叉熵与KL散度损失、精度/稀疏/准确率等多维度评估。

**📊 数据集**

实验数据集为带有真值属性注释的 CUB（鸟类）和 SUN（场景）两个基准数据集。

**📈 对比分析**

方法比较：对预测器基线和 VLM 基线分别采用 ℓ0、ℓ1、Bernoulli 三种稀疏化策略，评估指标为分类准确率、平均激活概念数（稀疏度）、属性预测精度和综合clarity。实验表明，尽管不同方法可达到相近的准确率，但clarity差异显著；Bernoulli 在大多数设置下获得最高clarity，预测器基线在准确率和精度上普遍优于 VLM 基线。

**⚠️ 局限性**

局限性：评估依赖于人工标注的概念数据集，无法推广到无标签或大规模多模态场景；对不同超参数和模型规模的探索不够充分；VLM 的零样本性能在细粒度任务上仍有限。

---

## 537. Robust Multimodal Representation Learning in Healthcare

**arXiv ID:** 2601.21941 | [PDF](https://arxiv.org/pdf/2601.21941v1)

**作者:** Xiaoguang Zhu `[一作]` (University of California), Jing Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种双流特征去相关框架（DFD），在医学多模态表示学习中通过因果推理将因果特征与偏差特征分离。

**💡 创新点**

创新点在于：①引入双流图神经网络与自适应门控机制实现因果与偏差特征并行提取；②结合一般化交叉熵损失和互信息最小化实现特征解耦；③框架模型无关，可与现有多模态方法无缝集成。

**🔧 技术方法**

使用了双流GNN、门控分配函数、一般化交叉熵损失、互信息最小化估计以及基线模型GRAPE、M3Care、MUSE等。

**📊 数据集**

在MIMIC‑IV、eICU和ADNI三大真实医学数据集上进行实验。

**📈 对比分析**

与GRAPE、M3Care、MUSE等基线相比，DFD在死亡预测、再入院预测和阿尔茨海默病进展预测上均提升了约1–3%的AUC‑ROC/AUC‑PRC/准确率，取得SOTA。

**⚠️ 局限性**

局限性包括：①仍需依赖图神经网络的结构，模型复杂度高；②对超参数和门控阈值敏感；③未在大型多模态语言模型或极小样本场景中验证。

---

## 538. AgenticSimLaw: A Juvenile Courtroom Multi-Agent Debate Simulation for Explainable High-Stakes Tabular Decision Making

**arXiv ID:** 2601.21936 | [PDF](https://arxiv.org/pdf/2601.21936v1)

**作者:** Jon Chun `[一作]` (Kenyon College), Yong Suk Lee `[通讯]` (Notre Dame University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出AgenticSimLaw多代理辩论框架，用角色化、7轮互动实现对高风险表格决策的可审计推理。

**💡 创新点**

创新点在于将多代理辩论与表格数据预测结合，提供可观测的推理轨迹和细粒度控制，突破传统单代理Chain‑of‑Thought的黑盒性。

**🔧 技术方法**

采用LLM（7–14B开源模型）、多代理交互协议、私有策略生成、日志记录等技术，并与传统统计模型及TabPFN对比。

**📊 数据集**

使用NLSY97青年再犯预测数据（1412条记录，27个特征）。

**📈 对比分析**

通过近90个模型/提示组合对AgenticSimLaw与标准LLM、统计模型进行基准，结果显示多代理辩论在准确率和F1稳定性上优于单代理CoT，且与传统模型相比性能提升有限。

**⚠️ 局限性**

局限包括仅在单一二分类任务验证、模型规模小、数据时代性、对偏见与公平性分析不足、计算开销大、以及LLM在表格推理上的局限。

---

## 539. Information Filtering via Variational Regularization for Robot Manipulation

**arXiv ID:** 2601.21926 | [PDF](https://arxiv.org/pdf/2601.21926v1)

**作者:** Jinhao Zhang `[一作]`, Jie Me `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在基于 3D 视觉的扩散策略 DP3 中加入轻量化的变分正则化模块，动态过滤 U‑Net 解码器中冗余噪声，提升机器人操控性能。

**💡 创新点**

提出了时间步条件的变分信息瓶颈（VR）模块，以 KL 正则化方式自适应抑制任务无关噪声，保持关键信息，且无额外参数。

**🔧 技术方法**

扩散模型（DDPM/DP3）+ 变分信息瓶颈 + 重新参数化技巧 + 轻量级 U‑Net 解码器改造。

**📊 数据集**

RoboTwin2.0、Adroit、MetaWorld 三大仿真基准，以及实际杯子堆叠实机实验。

**📈 对比分析**

与 DP3、DP 等现有方法对比，在 RoboTwin 平均成功率提升 6.1%（最高 27%），在 Adroit+MetaWorld 平均提升 4.1%，实机杯子堆叠成功率提升 13.4%，表现显著优于基线。

**⚠️ 局限性**

对 KL 权重 β 需要任务敏感的调优，部分任务噪声仍未完全消除；目前仅在少数任务验证真实环境效果，扩展到更复杂场景仍待研究。

---

## 540. Zero-Shot Video Restoration and Enhancement with Assistance of Video Diffusion Models

**arXiv ID:** 2601.21922 | [PDF](https://arxiv.org/pdf/2601.21922v1)

**作者:** Cong Cao `[一作]` (Tianjin University), Jingyu Yang `[通讯]` (Tianjin University)

**通讯引用:** 11416 | [OpenAlex ID](https://openalex.org/A5057336191)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了首个利用视频扩散模型辅助零样本图像修复/增强的框架，显著提升视频恢复与增强的时序一致性。

**💡 创新点**

创新点包括：① 同源与异源潜变量融合策略，允许任何SOTA T2V模型与图像修复模型协同；② 基于链式推理(COT)的自适应融合比率策略；③ 利用I2V模型的后处理进一步强化时序一致性。

**🔧 技术方法**

采用的技术包括：Latent Diffusion Model（LDM）+ DDIM采样；视频扩散模型（ZeroScope、CogVideoX-2B等）；COT推理与过程奖励模型；Stable Video Diffusion后处理；以及多种潜变量转换与融合算法。

**📊 数据集**

使用的评测数据集包括 REDS4、Vid4、UDM10（超分）、DID（低光增强）以及 DAVIS（盲超分）等。

**📈 对比分析**

与多种监督与无监督方法（TDAN、BasicVSR++、FMA-Net、VRT、MIA-SR、IART、VISION‑XL、Text2Video‑Zero、FateZero、VidToMe、FLDM、Upscale‑A‑Video、SeedVR、DiffBIR、TSD‑SR、ZVRD 等）对比，本文方法在九项指标（PSNR、SSIM、CLIP‑IQA、LPIPS、WE、FVD、DOVER、t‑LPIPS、VMAF）上均优于对手，尤其在时序一致性指标上表现突出。

**⚠️ 局限性**

局限性包括：① 需要额外的T2V模型和计算资源，导致推理时间和显存显著增加；② 对不同VAE/采样方式的兼容性有限，部分模型只能使用后处理；③ 目前仅在特定分辨率（576×320）与几类任务验证，需进一步验证在更高分辨率与多样化场景下的稳健性。

---

## 541. VideoAesBench: Benchmarking the Video Aesthetics Perception Capabilities of Large Multimodal Models

**arXiv ID:** 2601.21915 | [PDF](https://arxiv.org/pdf/2601.21915v1)

**作者:** Yunhao Li `[一作]` (Shanghai Jiao Tong University), Guangtao Zhai `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了 VideoAesBench 基准，用于评估大规模多模态模型（LMM）对视频审美质量的理解能力。

**💡 创新点**

创新点在于整合多类型视频（UGC、AIGC、RGC、压缩、游戏）与 12 细粒度审美维度，设计了四种问答形式（单选、多选、真伪、开放式），首次系统评测 LMM 在视频审美任务中的表现。

**🔧 技术方法**

采用人机协同策略生成问题答案，使用 Gemini‑2.5 生成完整字幕，GPT‑5.2 生成问题；评估时对闭合式问题计算准确率，对开放式问题使用 GPT‑5 计算感知分；对多模态模型的多模态输入进行推理。

**📊 数据集**

数据来源于 10 个现有视频质量/审美数据集（LSVQ、DIVIDE‑3K、VADB、FineVD、Love、HVEval、RGCD、TaoLive、LIVE‑Compress、LIVE‑YT‑Gaming），共 1,804 条视频和 1,804 组问答。

**📈 对比分析**

与 23 个开源与闭源 LMM（如 Qwen3‑VL‑32B、Claude‑Sonnet‑4.5、Gemini‑2.5‑Pro 等）对比，闭源模型整体优于开源；最佳闭源 Claude‑Sonnet‑4.5 约 67.9%，最佳开源 Qwen3‑VL‑32B 约 66.7%；多选题最难，单选/真伪相对容易；不同模型在视频类型、审美维度上表现不均。

**⚠️ 局限性**

局限性包括：仍只能捕捉基本审美特征，对多选与开放式问题的准确率偏低；模型在不同视频类型和审美维度上偏好不均；缺乏对更高层次审美语义的解释与可解释性能力。

---

## 542. Entropy-Based Dimension-Free Convergence and Loss-Adaptive Schedules for Diffusion Models

**arXiv ID:** 2601.21943 | [PDF](https://arxiv.org/pdf/2601.21943v1)

**作者:** Ahmad Aghapour `[一作]` (University of Michigan), Ziqing Zhang `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了无维度约束的扩散模型采样误差分析，并给出了基于信息熵的无维度收敛上界；同时设计了利用训练损失的Loss‑Adaptive Schedule（LAS）来自动调度逆向SDE的离散步长。

**💡 创新点**

核心创新在于：1）用信息论方法将离散化误差表达为MMSE函数并得到与 Shannon 熵相关的无维度上界O(H²/K)，不再依赖目标分布的几何假设；2）将训练过程中的 x₀‑预测误差直接映射到离散化调度问题，提出 LAS，避免昂贵的后训练 Monte‑Carlo 评估。

**🔧 技术方法**

主要技术包括：信息熵与 MMSE 关系、Tweedie 公式、KL 变换、离散化误差的MMSE 上界、几何 SNR 网格最优性、动态规划求最优调度、低阶与高阶采样器的数值实验。

**📊 数据集**

实验数据集：ImageNet 256×256（使用潜在扩散模型）以及人工 Gaussian 混合模型（toy）。

**📈 对比分析**

与 Time‑uniform、LogSNR、EDM 等常用时间调度以及多阶求解器（SDE‑DPM‑Solver++, DPM‑Solver++）进行对比；LAS 在 NFE 10/20 的 FID、sFID、IS 指标上均优于传统调度，尤其在低步长（粗采样）下提升显著。

**⚠️ 局限性**

局限性：1) 上界常数与MMSE导数估计可能不够紧，实际误差可能更小；2) 高 SNR 区域的路径 KL 上界过于保守，需要更精细的高 SNR 处理；3) 目前针对一阶求解器，尚未给出严格理论支持的高阶采样器调度；4) 只考虑离散化误差与学习误差的二次上界，未考虑其他模型结构的潜在影响。

---

## 543. From Meta-Thought to Execution: Cognitively Aligned Post-Training for Generalizable and Reliable LLM Reasoning

**arXiv ID:** 2601.21909 | [PDF](https://arxiv.org/pdf/2601.21909v1)

**作者:** Shaojie Wang `[一作]` (Hong Kong University of Science and Technology), Liang Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 9163 | [OpenAlex ID](https://openalex.org/A5100425207)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种认知对齐的LLM后训练框架，将推理拆分为元知识学习（CoMT）和任务适应（CCRL）两阶段。

**💡 创新点**

创新点在于：① 明确与人类认知过程对应，拆分抽象策略与实例执行；② CoMT通过教师生成不含具体数值的元思考轨迹来学习抽象策略；③ CCRL通过中间步骤的置信度奖励实现自我校准，减少过度自信错误。

**🔧 技术方法**

采用的技术包括：Chain-of-Meta-Thought (CoMT) 监督微调；基于熵的置信度测量与 Confidence‑Calibrated Reinforcement Learning (CCRL)；PPO 强化学习框架；教师模型生成元思考数据。

**📊 数据集**

使用的数据集：训练与评估均基于 GSM8K、SVAMP；OOD 评估使用 AsDiv、MAWPS、TabMWP、GSM‑Hard、GSM‑Symbolic。

**📈 对比分析**

与传统 CoT‑SFT+RL、专门化数学模型 DeepSeek‑Math、Qwen2.5‑Math 及 zero‑shot/few‑shot 基线对比。结果显示：In‑distribution 提升 2.19%（相较 CoT‑SFT+RL），OOB 提升 4.63%；训练时间减少 65‑70%，token 消耗降低 50%。

**⚠️ 局限性**

局限性：① 依赖教师模型生成元思考轨迹，质量受教师策略限制；② 主要针对数值型数学推理，跨领域适用性未知；③ 对更大规模模型的可扩展性尚未验证；④ 置信度奖励设计仍需进一步理论与实践支持。

---

## 544. User Acceptance Model for Smart Incentives in Sustainable Video Streaming towards 6G

**arXiv ID:** 2601.21903 | [PDF](https://arxiv.org/pdf/2601.21903v1)

**作者:** Konstantinos Varsos `[一作]` (Athens University of Economics and Business), Vasillios A. Siris `[通讯]` (Athens University of Economics and Business)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种统一的用户接受模型，用以通过智能激励促使视频流在降低比特率时保持可持续性，并在模型中融入了QoE、激励阈值、利他行为等因素。

**💡 创新点**

创新点在于将环境意识、个性化激励、利他性与动态学习整合到同一概率框架；引入社会福祉作为利他驱动力；利用历史交互数据进行用户特征学习；并通过实验验证个性化激励与教育干预能显著降低成本。

**🔧 技术方法**

主要技术包括统计概率建模、逻辑回归估计用户激励阈值与灵敏度、Sigmoid 拦截函数、MOE（多项式经验）模型、合成数据生成器及数值仿真。

**📊 数据集**

实验数据采用自定义的合成数据集，模拟 1000 名用户，分别使用离散比特率集合和均匀/正态激励分布；未使用任何真实视频流或用户行为数据。

**📈 对比分析**

与统一平均激励策略对比，通过计算灵活性-成本比进行评估；实验显示个性化激励在多种激励分布下显著提高比率，教育干预进一步降低激励阈值，整体提升可持续性收益。

**⚠️ 局限性**

局限性包括仅在合成数据上验证，未对真实用户行为进行评估；模型假设参数相互独立且可知；缺乏对长期动态适应性和群体行为变化的深入分析。

---

## 545. TraceRouter: Robust Safety for Large Foundation Models via Path-Level Intervention

**arXiv ID:** 2601.21900 | [PDF](https://arxiv.org/pdf/2601.21900v1)

**作者:** Chuancheng Shi `[一作]` (University of Sydney), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 60184 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 TraceRouter，一种基于路径层面干预的安全框架，用来识别、追踪并断开大型基础模型中的有害语义传播路径。

**💡 创新点**

核心创新是放弃传统的神经元/特征局部抑制，改为通过敏感起始层检测、稀疏自编码器分解、特征影响分数追踪，并在路径级进行精确抑制，能够物理切断分布式跨层有害语义循环，兼顾安全与性能。

**🔧 技术方法**

使用技术包括：Top‑K稀疏自编码器 (SAE)、注意力差异检测、特征影响分数 (FIS) 计算、零置干预、路径拆分与掩码、路径抑制缩放因子 λ，适用于 diffusion、LLM 与 MLLM 三大模型范式。

**📊 数据集**

实验数据集涵盖：Stable Diffusion 1.4、FLUX1.Dev、Show‑o2（图像生成）；I2P、P4D、Ring‑A‑Bell、MS COCO；LLM 训练集 LLaMA3‑8B‑Instruct、Mistral‑7B‑Instruct；多模态 LLaVA‑1.5‑7B、MiniGPT‑4‑7B；评估集 FigStep、Global‑MMLU‑Lite、MM‑Bench。

**📈 对比分析**

与多种 SOTA 方法（ESD、UCE、CA、SLD‑Med、MACE、RECE、SPM、DuMo、SNCE 等）以及 LLM 防御（DeepAug、CB、DeRTa、HumorReject 等）对比。结果表明 TraceRouter 在标准安全、对抗鲁棒性与生成质量上均遥遥领先，例如在 Stable Diffusion 1.4 上 I2P 媒体 DSR 达 99.2% / 93.6%，Ring‑A‑Bell 98.7%；LLM 平均 DSR 99.6%/98.8%；同时保持 CLIP、FID 与基准相当或更优，整体性能显著优于基线。

**⚠️ 局限性**

主要局限包括：需要对模型内部进行访问与高算力支持，路径级干预对极大型模型可能不可扩展；抑制因子 λ 需要手工调优，过大/过小会影响质量；目前主要验证于文本与视觉生成，跨模态对话系统的泛化尚未充分评估；对抗手段持续演进，模型可能出现新型逃逸路径。

---

## 546. Past- and Future-Informed KV Cache Policy with Salience Estimation in Autoregressive Video Diffusion

**arXiv ID:** 2601.21896 | [PDF](https://arxiv.org/pdf/2601.21896v1)

**作者:** Hanmo Chen `[一作]` (Hangzhou Institute of Technology), Cheng Deng `[通讯]` (Xidian University)

**通讯引用:** 12439 | [OpenAlex ID](https://openalex.org/A5015874725)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于过去与未来信息的KV缓存策略PaFu‑KV，并在自回归视频扩散模型中实现；

**💡 创新点**

创新点在于：①利用双向教师模型进行分布匹配蒸馏，得到轻量级Salience Estimation Head（SEH）以在因果条件下估计token的重要性；②设计空间-时间平衡的salience评分，消除自注意力的对角偏倚，从而精准筛选高价值token；③在推理时动态保留高salience token、淘汰低salience token，实现显著压缩缓存并提升推理速度。

**🔧 技术方法**

核心技术包括：自回归视频扩散（Diffusion Transformer）、分布匹配蒸馏（DMD）、轻量级MLP SEH、空间-时间平衡salience计算、KV缓存动态管理（基于token索引的淘汰策略）。

**📊 数据集**

使用了VBench、VBench‑Long以及VidProM扩展提示集进行短视频（5s）与长视频（30s）生成实验。

**📈 对比分析**

与多种公开的双向扩散与自回归视频生成模型（LTX‑Video、Wan、SkyReels‑V2、MAGI‑1、CausVid、NOVA、Pyramid Flow、Self‑Forcing、LongLive、Rolling‑Forcing）对比，PaFu‑KV在FPS（实时推理）上提升至23.6 FPS，短视频总分与对手相当甚至略优；长视频评测中在主体/背景一致性、运动平滑度等指标上均达到或逼近最佳，且质量漂移更低。

**⚠️ 局限性**

限制主要包括：①仍需较大GPU内存支持（缓存容量可调，但过度压缩会显著损失质量）；②估计未来信息的精度受教师模型与蒸馏策略限制；③在极长序列或高分辨率视频上，缓存管理仍可能产生累积误差，需进一步优化。

---

## 547. astra-langchain4j: Experiences Combining LLMs and Agent Programming

**arXiv ID:** 2601.21879 | [PDF](https://arxiv.org/pdf/2601.21879v1)

**作者:** Rem Collier `[一作]` (University College Dublin), Andrei Ciortea `[通讯]` (University of St.Gallen)

**通讯引用:** 354 | [OpenAlex ID](https://openalex.org/A5075482342)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275`

**🎯 论文内容**

实现了ASTRA语言的LLM集成库astra-langchain4j，并在旅行规划、井字棋和塔世界等示例中验证其可行性。

**💡 创新点**

将LLM调用与传统BDI式代理工具包无缝结合，并提出BeliefRAG、模板化提示与复合提示机制，以实现知识检索增强生成。

**🔧 技术方法**

基于Java的LangChain4J、OpenAI GPT-4o、Google Gemini，并利用ASTRA的模块化、模板与事件机制实现代理交互。

**📊 数据集**

未使用公开数据集，主要通过手工编写的提示和环境状态（如棋盘JSON、塔世界块状态）进行实验。

**📈 对比分析**

与传统规则/线性玩家对比，LLM代理在部分任务中能提供多样化响应，但在决策一致性与推理深度上明显逊色，实验结果以定性评估为主。

**⚠️ 局限性**

LLM在上下文决策、复杂推理和提示设计上表现不稳定，缺乏一致性，导致在井字棋和塔世界等需要精确推理的任务中效果有限。

---

## 548. How do Visual Attributes Influence Web Agents? A Comprehensive Evaluation of User Interface Design Factors

**arXiv ID:** 2601.21961 | [PDF](https://arxiv.org/pdf/2601.21961v1)

**作者:** Kuai Yu `[一作]` (Columbia University), Huan Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 13745 | [OpenAlex ID](https://openalex.org/A5020766468)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个可控评估流程，生成视觉属性变化的网页变体，利用人类类似的滚动与点击交互，并通过目标点击率和目标提及率两项指标定量评估视觉属性对基于视觉语言模型的网页代理决策的影响。

**💡 创新点**

首次系统地将网页视觉属性与代理行为关联，提出多维度变体生成与双指标评估方法，并发现颜色对比、卡片大小、位置和卡片清晰度是最显著的影响因素，揭示代理与人类在视觉注意力上的相似与差异。

**🔧 技术方法**

利用 CSS 变体生成、基于视觉语言模型（UI‑TARS 7B、GLM‑4.1v‑9B、Qwen3‑VL‑8B‑Instruct、OpenAI‑CUA）的浏览与推理、LLM 作为评判者提取思路中的目标提及。

**📊 数据集**

使用五个真实网站（Amazon、eBay、Booking、Expedia、NPR）中的商品或文章页面，针对每个目标项生成 48 个视觉变体，覆盖 8 类视觉属性（背景色、文字色、字体、字号、位置、卡片大小、清晰度、顺序）。

**📈 对比分析**

对 4 个代理在每个变体上做 50 次独立实验，计算目标点击率和目标提及率；结果显示高对比背景、放大卡片、靠前位置、清晰卡片显著提升点击率；字体与文本颜色、图像清晰度影响较小；指标互补，说明视觉吸引力与推理关注度高度相关。

**⚠️ 局限性**

仅评估了四个代表性代理，未覆盖更大规模或其他架构；实验成本高，未公开完整代码；研究聚焦良性场景，缺乏对更广泛环境的适用性验证。

---

## 549. Embracing Aleatoric Uncertainty in Medical Multimodal Learning with Missing Modalities

**arXiv ID:** 2601.21950 | [PDF](https://arxiv.org/pdf/2601.21950v1)

**作者:** Linxiao Gong `[一作]`, Xiaoguang Zhu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种基于aleatoric不确定性的多模态学习框架AUM，用于处理临床数据中的缺失模态问题。

**💡 创新点**

创新点在于显式建模每个模态的多元高斯分布以量化不确定性，并在双向患者-模态图上采用不确定性感知的动态消息传递机制，天然处理缺失模态并提升可靠模态的权重。

**🔧 技术方法**

采用多模态高斯分布建模、图神经网络动态消息传递、基于VIB的正则化、对比学习和交叉熵分类损失的组合。

**📊 数据集**

使用MIMIC‑IV和eICU两个大型ICU数据库进行评估。

**📈 对比分析**

与多种基线方法（包括MUSE、M3Care、GRAPE等）对比，AUM在死亡率预测任务上分别提升AUC‑ROC 2.26%和2.17%，并在不同缺失比例和噪声水平下保持优越性能。

**⚠️ 局限性**

主要局限在于对高斯假设的依赖、模型参数调优的复杂度以及对非高斯或结构化噪声的鲁棒性仍待进一步研究。

---

## 550. Dependence of Equilibrium Propagation Training Success on Network Architecture

**arXiv ID:** 2601.21945 | [PDF](https://arxiv.org/pdf/2601.21945v1)

**作者:** Qingshan Wang `[一作]` (Max Planck Institute for the Science of Light), Florian Marquardt `[通讯]` (Max Planck Institute for the Science of Light)

**通讯引用:** 21063 | [OpenAlex ID](https://openalex.org/A5059746672)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了局部连通的XY格子网络在平衡传播（EP）训练下的学习性能，并将其与全连接网络和密集层网络进行对比，探索了架构对网络响应、耦合演化及深度扩展的影响。

**💡 创新点**

创新点在于系统评估了稀疏局部连接与跨层跳跃连接对EP训练效果的提升，证明了在物理可实现的格子模型中仍能实现与全连接网络相当甚至更优的性能，并提出了层间局部耦合与卷积式架构在MNIST上优于密集层网络的结论。

**🔧 技术方法**

采用了平衡传播（Equilibrium Propagation）作为物理梯度训练方法，使用XY模型的能量函数进行仿真训练，并结合多种网络拓扑（SQ、3NSQ、P3NSQ、4NSQ、LCL、CNN‑like、DL）实现不同连通性和层结构。

**📊 数据集**

使用了经典基准数据集：XOR（二分类）、Iris（三分类）以及完整MNIST（10分类）来评估不同架构的学习效果。

**📈 对比分析**

通过对比训练误差、测试准确率以及参数量，发现局部连通的4NSQ格子在Iris上可与全连接网络相媲美，深度足够时还可超越密集层网络；在MNIST上，LCL和CNN‑like架构分别达到约96%和98%的测试准确率，显著优于同参数量的DL网络。

**⚠️ 局限性**

主要局限在于训练收敛速度慢（如4NSQ格子需要约2万轮才能完成XOR学习），并且在更深的层数下局部连通网络性能下降，说明仍需改进架构或梯度估计方法以提高可扩展性。

---

## 551. BookNet: Book Image Rectification via Cross-Page Attention Network

**arXiv ID:** 2601.21938 | [PDF](https://arxiv.org/pdf/2601.21938v1)

**作者:** Shaokai Liu `[一作]` (Hefei University of Technology), Wengang Zhou `[通讯]` (University of Science and Technology of China)

**通讯引用:** 18254 | [OpenAlex ID](https://openalex.org/A5046805800)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种端到端的双页书本图像矫正框架 BookNet，专门针对书本页间相互耦合的非对称几何畸变进行处理。

**💡 创新点**

创新点包括：① 双分支解码器与跨页注意力机制，能够同时学习左页、右页以及整个书本的位移场；② 对三种位移场（左、右、全书）进行联合监督，提升了缝隙处的几何一致性；③ 构建了大规模合成数据集 Book3D 与真实评测基准 Book100，为双页矫正任务提供了完整的数据生态。

**🔧 技术方法**

采用轻量化 CNN+Transformer 编码器提取全局特征，双分支 Transformer 解码器分别生成页面特定位移；跨页注意力实现两分支信息交互；卷积 + 可学习凸插值上采样得到高分辨率位移；整体使用多任务 L1 损失进行训练。

**📊 数据集**

使用 56,000 张高分辨率合成书本图像的 Book3D 训练集，和 100 张真实书本拍摄图像与对应扫描图像的 Book100 评测基准。

**📈 对比分析**

与多种单页文档矫正方法（DewarpNet、DocTr 等）在 Book100 上进行对比。BookNet 在 MSSIM、LD、AD、CER、ED 等指标上均取得最优或最接近最优的结果（MSSIM 0.48，LD 12.42，AD 0.53，CER 0.345，ED 948.63），且单张图像推理速度约 24.39 FPS。

**⚠️ 局限性**

局限性：目前仅针对双页书本；对极端光照、遮挡或页边不完整的情况仍有挑战；模型训练依赖于合成与真实数据的配对，若缺乏高质量标注可能影响泛化。

---

## 552. Optimistic Transfer under Task Shift via Bellman Alignment

**arXiv ID:** 2601.21924 | [PDF](https://arxiv.org/pdf/2601.21924v1)

**作者:** Jinhang Chai `[一作]` (Princeton University), Yujun Yan `[通讯]` (Dartmouth College)

**通讯引用:** 1758 | [OpenAlex ID](https://openalex.org/A5100688739)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出一种在线迁移强化学习框架——RWT‑Q，利用一次Bellman对齐实现源任务与目标任务的无偏迁移；

**💡 创新点**

核心创新在于发现一次Bellman不匹配是迁移误差的根源，并通过重加权目标（RWT）将其转化为固定的单步奖励差异，从而实现方差降低与偏差校正的两阶段学习；

**🔧 技术方法**

技术方法包括重加权Bellman对齐、基于RKHS的核岭回归、乐观上界（OFU）探索以及深度Q网络的神经实现；

**📊 数据集**

实验使用自定义的随机奖励网格环境（RandomRewardGridEnv）和离散Q表/深度网络，源任务与目标任务通过高斯扰动构造奖励差异；

**📈 对比分析**

与传统单任务Q‑learning和未经对齐的源数据聚合相比，RWT‑Q在所有设置下均显著提升样本效率，且聚合方式往往表现更差，验证了Bellman对齐的重要性；

**⚠️ 局限性**

局限性包括对密度比估计的依赖、需要源任务与目标任务满足绝对连续性与覆盖条件，以及仅对单步奖励差异作结构假设，尚未探讨多步迁移或非马尔可夫情境。

---

## 553. The 'Big Three' of Scientific Information: A comparative bibliometric review of Web of Science, Scopus, and OpenAlex

**arXiv ID:** 2601.21908 | [PDF](https://arxiv.org/pdf/2601.21908v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 554. Uncertainty-Aware Data-Based Method for Fast and Reliable Shape Optimization

**arXiv ID:** 2601.21956 | [PDF](https://arxiv.org/pdf/2601.21956v1)

**作者:** Yunjia Yang `[一作]` (Tsinghua University), Haixin Chen `[通讯]` (Tsinghua University)

**通讯引用:** 3171 | [OpenAlex ID](https://openalex.org/A5044630621)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于不确定性量化的离线数据驱动优化框架（UA-DBO），用于多点气动形状优化。

**💡 创新点**

创新点在于将模型置信度融入目标函数，采用高斯随机编码解码（GS‑ED）实现无监督的不确定性估计，并在优化中使用置信区间上界以避免误导。

**🔧 技术方法**

技术手段包括变分自编码器改造的GS‑ED网络、Monte Carlo采样、置信区间上界目标函数、深度集成对比以及CFD仿真验证。

**📊 数据集**

使用了约1420个不同几何与巡航工况的翼型数据库（包含RAE2822等），并利用CFL3D CFD获得的多工况流场数据。

**📈 对比分析**

与传统DBO、完整CFD多点优化以及单点优化进行对比，UA-DBO在拖曳分离和气浪起伏两项多点目标上平均提升约1.47倍，误差降低39.6%，计算速度几乎保持单点优化级别。

**⚠️ 局限性**

局限性包括在极端几何（如厚度超出训练集）时泛化性能不足，且未引入基于物理的误差指标，未来需进一步提升不确定性预测的鲁棒性。

---

## 555. SONIC: Segmented Optimized Nexus for Information Compression in Key-Value Caching

**arXiv ID:** 2601.21927 | [PDF](https://arxiv.org/pdf/2601.21927v1)

**作者:** Hong Chen `[一作]` (Hong Kong University of Science and Technology), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1202 | [OpenAlex ID](https://openalex.org/A5057914558)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于可学习的 Nexus token 的 KV 缓存压缩框架 SONIC，用于多轮对话时降低模型内存占用并保持语义完整性。

**💡 创新点**

创新点在于：①利用 Nexus token 聚合历史上下文并通过分层可见性掩码实现仅保留系统提示、Nexus token 与当前查询的访问；②采用动态预算训练使模型能在不同压缩率下自适应；③结合多级蒸馏、信息瓶颈重构与注意力正则化等多种损失提升压缩后语义保留。

**🔧 技术方法**

使用了可学习的 Nexus token 插入、分层可见性掩码、教师蒸馏与隐藏状态对齐、注意力引导蒸馏、Nexus 关注正则化、信息瓶颈重构损失，以及动态预算采样的联合训练框架。

**📊 数据集**

在四个基准上验证：MTBench101、SafeDialBench、GSM8K‑Variant 以及 CoreRes，并以 Qwen3‑0.6B、1.7B、4B 三种规模模型进行实验。

**📈 对比分析**

与 Full‑Context 以及 H2O、SnapKV、StreamingLLM、ExpectedAttention 等传统 KV 压缩基线对比，SONIC 在 80%/50% 压缩率下在所有任务上均优于基线，接近全上下文性能；推理时间下降约 50%，显存占用下降约 67%。

**⚠️ 局限性**

局限在于：极低预算（如 10 个 Nexus）仍会出现信息丢失；压缩后对细粒度上下文的恢复仍有限；实验主要集中在 Qwen3 系列模型，跨模型、跨任务的泛化性仍需进一步验证。

---

## 556. ProRAG: Process-Supervised Reinforcement Learning for Retrieval-Augmented Generation

**arXiv ID:** 2601.21912 | [PDF](https://arxiv.org/pdf/2601.21912v1)

**作者:** Zhao Wang `[一作]` (Renmin University of China), Zhicheng Dou `[通讯]` (Renmin University of China)

**通讯引用:** 3908 | [OpenAlex ID](https://openalex.org/A5010558184)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ProRAG，结合过程监督的强化学习框架，用步骤级奖励来优化多跳 RAG 的推理与检索流程。

**💡 创新点**

创新点在于：①使用 MCTS 构造过程奖励模型（PRM）提供密集的步骤级评价；②设计双粒度优势估计，将步骤级过程奖励与全局结果奖励融合；③通过 PRM 导向的推理细化阶段解决冷启动问题，实现在线探索与过程监督的无缝对接。

**🔧 技术方法**

核心技术包括：多跳检索增强生成（RAG）、蒙特卡洛树搜索（MCTS）生成过程奖励对；对比学习训练 PRM；PPO/GRPO 基础的强化学习与双粒度优势；LoRA 微调；对齐与格式化的控制标记。

**📊 数据集**

使用五个多跳推理基准：PopQA、HotpotQA、2WikiMultihopQA、MuSiQue、Bamboogle，检索语料为 2018 年 Wikipedia，检索模型为 E5-base。

**📈 对比分析**

与标准 RAG、迭代检索+生成、主动检索、Agentic RAG、Search‑R1、ReasonRAG、HiPRAG 等基线对比。ProRAG 在所有基准上均优于最强基线（平均 F1 提升约 2.5%，在 MuSiQue、2WikiMultihopQA 上显著提升，p<0.05）。

**⚠️ 局限性**

局限性：① PRM 仍可能受限于 MCTS 树搜索的质量与覆盖范围；② 双粒度权重 β 的调优依赖经验，过大可能导致对噪声奖励过度依赖；③ 目前仅在固定知识库与检索器上验证，未知对更大规模或动态知识库的泛化情况。

---

## 557. Managing Solution Stability in Decision-Focused Learning with Cost Regularization

**arXiv ID:** 2601.21883 | [PDF](https://arxiv.org/pdf/2601.21883v1)

**作者:** Victor Spitzer `[一作]` (Lhyfe), Francois Sanson `[通讯]` (LISN)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了决策聚焦学习（Decision‑Focused Learning, DFL）中组合优化映射对扰动的稳定性问题，并提出通过对成本向量进行正则化来控制稳定半径，从而提高学习过程的鲁棒性与性能。

**💡 创新点**

创新点在于：① 用组合优化稳定性理论解释扰动强度与成本向量尺度不匹配导致的学习失效；② 设计了两类正则化函数（归一化 rⁿ 与投影 rᵖ），可在不改变决策映射的前提下控制成本向量的稳定半径；③ 在多种 DFL 方法（SPO、DBB、DPO）上验证正则化的普适性。

**🔧 技术方法**

采用的技术包括：组合优化稳定性分析、扰动基 DFL 技术（Smart Predict‑then‑Optimize、Differentiable Black‑Box、Differentiable Perturbed Optimizer）、正则化函数 rⁿ 与 rᵖ、梯度估计与蒙特卡罗采样、Wilcoxon 符号秩检验。

**📊 数据集**

使用的数据集：公开基准 Shortest Path（SP）和 Set Matching（SM）三组实例；以及构造的 toy 线性规划问题（用于展示稳定性失效与正则化效果）。

**📈 对比分析**

比较方法：在同一训练设置下对比无正则化与两种正则化（rⁿ、rᵖ）下的 SPO、DBB、DPO 等方法；性能指标为 regret（%）。统计检验显示正则化后的模型在绝大多数实验中显著降低 regret，rᵖ 正则化效果更为稳定。

**⚠️ 局限性**

局限性：仅对成本向量正则化给出了上界约束，未给出稳定半径下界；对不同问题结构下稳定性特性的通用性和更广泛的扰动形式（如乘性扰动）仍待探索。

---

## 558. Evolution of Benchmark: Black-Box Optimization Benchmark Design through Large Language Model

**arXiv ID:** 2601.21877 | [PDF](https://arxiv.org/pdf/2601.21877v1)

**作者:** Chen Wang `[一作]` (South China University of Technology), Yue-Jiao Gong `[通讯]` (South China University of Technology)

**通讯引用:** 5884 | [OpenAlex ID](https://openalex.org/A5063438356)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于大型语言模型的自动化黑盒优化基准设计框架EoB，能够生成兼顾景观多样性与算法区分性的基准函数；

**💡 创新点**

创新点在于将多目标程序演化与LLM的代码反思能力结合，提出双目标（景观相似度与算法区分能力）评估，并利用反射式交叉与景观初始化知识实现程序与景观共同进化；

**🔧 技术方法**

主要技术包括LLM（Deepseek‑v3.2）驱动的程序生成与演化、MOEA/D多目标优化、NeurELA景观特征提取、PBI尺度化、统计学指标（ADC、LSI）以及基于JSON的反思式提示；

**📊 数据集**

使用的基准数据集包括CoCo‑BBOB、MetaBox中的UAV与HPO任务，作为目标任务集及算法池（10种经典BBO算法）用于评估ADC；

**📈 对比分析**

通过与手工设计基准CoCo‑BBOB对比，EoB生成的EoB‑BBOB在景观多样性与算法区分度上均显著提升；在学习型优化器的训练与泛化实验中，EoB‑UAV/HPO显著降低训练成本、提高泛化性能；整体表现优于传统基准；

**⚠️ 局限性**

局限性包括依赖LLM质量、对高维复杂问题的扩展性未知、仅考虑两目标而非更细粒度指标、对动态/约束等更广泛优化场景的适配尚待验证。

---

## 559. PaddleOCR-VL-1.5: Towards a Multi-Task 0.9B VLM for Robust In-the-Wild Document Parsing

**arXiv ID:** 2601.21957 | [PDF](https://arxiv.org/pdf/2601.21957v1)

**作者:** Cheng Cui `[一作]` (Baidu Inc), Yanjun Ma `[通讯]` (Baidu Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 PaddleOCR-VL-1.5，集成文档解析、文本定位、印章识别等功能，提升对真实世界物理失真文档的鲁棒性。

**💡 创新点**

创新点包括：PP-DocLayoutV3 的多点分割与阅读顺序联合预测；引入印章识别与文本定位任务；Distortion‑Aware 数据增强；Uncertainty‑Aware Cluster Sampling（UACS）与 GRPO 强化学习；全局 0.9B 参数的高效架构。

**🔧 技术方法**

技术栈涵盖 RT‑DETR + MaskHead + Transformer Decoder + Global Pointer、NaViT 视觉编码器、ERNIE‑4.5‑0.3B 语言模型、PaddleFormers、CLIP 视觉聚类、GRPO 强化学习、异步多线程推理、FastDeploy、vLLM 与 SGLang 部署。

**📊 数据集**

使用 OmniDocBench v1.5、Real5‑OmniDocBench（基于 5 种真实场景）、内部 38k 文档集合、印章/文本定位/表格/公式等多语言数据集；同时覆盖手写、古文、广告等多样场景。

**📈 对比分析**

在 OmniDocBench v1.5 上实现 94.5% 的 SOTA；在 Real5‑OmniDocBench 上取得 92.05% 的整体精度，显著超越 Qwen3‑VL、Gemini‑3 Pro 等大模型；文本定位、印章识别等子任务亦保持领先；FastDeploy 推理速率达 1.4335 页/秒，显著提升吞吐量。

**⚠️ 局限性**

局限性包括：仍主要针对文本、表格、公式、印章等结构化元素，对极端遮挡、非中文多语言场景仍存在挑战；缺乏对动态图像/视频的适配；实验环境与数据来源未完全公开，复制性受限。

---

## 560. Deep Models, Shallow Alignment: Uncovering the Granularity Mismatch in Neural Decoding

**arXiv ID:** 2601.21948 | [PDF](https://arxiv.org/pdf/2601.21948v1)

**作者:** Yang Du `[一作]` (University of Pittsburgh), Liang Zhan `[通讯]` (University of Pittsburgh)

**通讯引用:** 8146 | [OpenAlex ID](https://openalex.org/A5103246248)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

将脑电/磁共振信号与深度视觉模型的中间层特征进行对齐，以实现更精准的视觉信息重构。

**💡 创新点**

提出“Shallow Alignment”策略：从传统的只对齐最终层特征转向对齐中间层特征，从而消除人脑多尺度信息与机器视觉单一抽象层的粒度不匹配；同时通过这一方法开启了大型视觉模型在脑-机解码中的规模律。

**🔧 技术方法**

采用对比学习框架，对脑信号和视觉特征分别做线性投影，计算余弦相似度并最小化对比损失；使用EEGProject等简易脑编码器和多种预训练视觉编码器（ResNet、ViT、DINOv2、EVA‑02、InternViT等）。

**📊 数据集**

实验基于 THINGS‑EEG（EEG数据）和 THINGS‑MEG（MEG数据）两个公开数据集，涵盖数千种物体概念。

**📈 对比分析**

与 NICE、ATM、Neural‑MCRL、NeuroBridge 等最新方法对比，在单个受试者的检索任务中，Shallow Alignment 的 Top‑1 精度提升 22%–58%，在大型模型上实现了显著的规模律提升；在多受试者、跨受试者场景下亦保持了较强的性能。

**⚠️ 局限性**

目前仍需手工搜索最优中间层，对层级选择缺乏自适应机制；且跨受试者的粒度不匹配问题未得到完全解决。

---

## 561. Belief Propagation Converges to Gaussian Distributions in Sparsely-Connected Factor Graphs

**arXiv ID:** 2601.21935 | [PDF](https://arxiv.org/pdf/2601.21935v1)

**作者:** Tom Yates `[一作]` (Imperial College London), Andrew J. Davison `[通讯]` (Imperial College London)

**通讯引用:** 31683 | [OpenAlex ID](https://openalex.org/A5039230558)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `6514db3d-8de6-452c-91b7-acdb31787cc4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文证明在稀疏连接的因子图中，贝叶斯传播（BP）的变量信念在满足四个关键假设下会收敛为高斯分布，提供了对高非高斯问题使用高斯逼近（GBP）的理论基础。

**💡 创新点**

创新点在于首次利用中心极限定理与卷积/乘法的累积效应，证明在树形、链形和含环的稀疏图中，变量信念会因路径深度（而非节点平均）而趋于高斯，并给出了置信先验对收敛的阈值分析。

**🔧 技术方法**

技术手段主要是累积分布函数和矩生成函数的切比雪夫展开、标准化矩的衰减理论、计算树（computation tree）展开，以及实验验证的KL散度与MSE评估。

**📊 数据集**

在实验中使用合成链、树、环图以及实际的“Cones”立体深度估计数据集（Middlebury立体数据集）来检验理论。

**📈 对比分析**

与非参数的循环BP基准对比，GBP在深度估计任务中达到几乎相同的MSE，KL散度在大多数像素处低于0.02，显示高斯近似在大部分图像区域是足够准确的。

**⚠️ 局限性**

局限性包括：需要满足四个假设（有限矩、Lindeberg条件、低度因子、差分不变性），对高置信先验（R≲6）仍可能导致非高斯结果；并未覆盖异步BP调度、非二元因子或不满足差分不变性的场景。

---

## 562. Breaking the Regional Barrier: Inductive Semantic Topology Learning for Worldwide Air Quality Forecasting

**arXiv ID:** 2601.21899 | [PDF](https://arxiv.org/pdf/2601.21899v1)

**作者:** Zhiqing Cui `[一作]` (Hong Kong University of Science and Technology), Yuxuan Liang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5513 | [OpenAlex ID](https://openalex.org/A5018828723)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 OmniAir 框架，利用可归纳语义拓扑实现全球站点级空气质量预测。

**💡 创新点**

创新点在于将物理属性编码为语义身份，动态生成稀疏拓扑并通过差分传播模拟扩散与源生成，兼顾长程非欧几里得关联和物理扩散双重机制。

**🔧 技术方法**

采用 Fourier 特征映射、可归纳节点编码、动态稀疏邻接生成、注意力权重门控、差分扩散传播、Transformer 结构等深度学习技术。

**📊 数据集**

使用自建 WorldAir 数据集，涵盖 7,861 个全球监测站，记录 6 种主要污染物和气象、地理属性。

**📈 对比分析**

与 18 个基准模型对比，OmniAir 在全球及各区域数据集上均取得最低 MAE、RMSE，速度约为现有模型的 10 倍，显著提升精度与效率。

**⚠️ 局限性**

局限性包括对极端稀疏区域仍需更多高质量气象输入，且模型在特殊化学反应（如臭氧形成）仍存在预测误差。

---

## 563. Making Models Unmergeable via Scaling-Sensitive Loss Landscape

**arXiv ID:** 2601.21898 | [PDF](https://arxiv.org/pdf/2601.21898v1)

**作者:** Minwoo Jang `[一作]` (Postech), Jungseul Ok `[通讯]` (Postech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出通过尺度敏感的损失景观使模型难以融合的方法

**💡 创新点**

首次将尺度敏感性作为控制模型融合的关键因素，用于模型IP保护

**🔧 技术方法**

采用尺度归一化、对比损失、梯度对齐等技术对损失景观进行调整

**📊 数据集**

在CIFAR-10、CIFAR-100和ImageNet等公开数据集上进行实验

**📈 对比分析**

与传统参数平均、对齐等模型融合方法对比，实验表明该方法在保持原始精度的同时，显著降低了融合后模型的性能（平均降低30%以上）

**⚠️ 局限性**

仅适用于相似结构和训练阶段相近的模型，且可能略微影响模型鲁棒性

---

## 564. Learn-to-Distance: Distance Learning for Detecting LLM-Generated Text

**arXiv ID:** 2601.21895 | [PDF](https://arxiv.org/pdf/2601.21895v1)

**作者:** Hongyi Zhou `[一作]` (Tsinghua University), Chengchun Shi `[通讯]` (London School of Economics and Political Science)

**通讯引用:** 613 | [OpenAlex ID](https://openalex.org/A5025970743)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于自适应学习重写距离的LLM文本检测算法，结合几何分析阐释重写式检测的原理；

**💡 创新点**

创新点在于将传统固定距离改为可学习的距离函数，并通过理论证明其能更好区分人类与LLM文本；

**🔧 技术方法**

利用重写式检测、微调语言模型构造距离、采样多重重写以及AUC评估等技术；

**📊 数据集**

在24个公开数据集、7种LLM（如GPT‑4o、Claude‑3.5、Gemini‑2.5等）以及超过100个实验设置下验证；

**📈 对比分析**

与12个零拷贝与ML基准方法比较，平均提升57.8%‑80.6%（相对提升），在不同提示、攻击场景下表现稳健；

**⚠️ 局限性**

主要局限是计算开销较大，需要多次重写和推理，导致推理速度慢，未来需优化执行效率。

---

## 565. LLM-Driven Scenario-Aware Planning for Autonomous Driving

**arXiv ID:** 2601.21876 | [PDF](https://arxiv.org/pdf/2601.21876v1)

**作者:** He Li `[一作]`, Chengzhong Xu `[通讯]` (University of Macau)

**通讯引用:** 17283 | [OpenAlex ID](https://openalex.org/A5012773300)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于大型语言模型（LLM）的自适应规划框架（LAP），用于在不同交通场景下在高速驾驶（FD）与精准驾驶（SA）模式之间实时切换。

**💡 创新点**

创新点在于：①将LLM作为情景感知切换器，利用检索增强生成（RAG）实现对驾驶模式的智能决策；②将LLM推理嵌入联合优化中，协同求解模式配置与运动规划；③结合树搜索模型预测控制（TMPC）与交替最小化（AM）两种算法，在满足动态约束的同时保持实时性。

**🔧 技术方法**

使用技术包括：Gemini‑2.0‑Flash LLM + RAG、树搜索MPC、交替最小化（AM）、混合整数非线性优化、ROS、CARLA仿真平台。

**📊 数据集**

数据集与评估平台：在基于澳门大奖赛赛道的MetaGranndPrx仿真平台进行测试；使用专家数据集评估LLM切换器精度；在不同车流密度（1至5辆障碍车）下进行多次超车实验。

**📈 对比分析**

通过与CF‑based PP、RDA、Optimistic等基准进行对比，LAP在稀疏车辆赛道上跑10次平均lap时间269.13 s（平均速度86.9 km/h，峰值211 km/h），仅比无障碍极限方案差约3 s；在密集交通中FD模式成功率从75%降至5%，而SA模式保持100%至3辆车，95%至5辆车，整体显示LAP在速度与成功率上优于其他方法。

**⚠️ 局限性**

局限性包括：LLM推理仍存在一定时延，切换频率仅1 Hz可能不足以应对极端突发情况；模型在极端拥堵下仍需更细粒度的几何建模；实验仅在仿真平台完成，缺乏真实道路验证；对感知噪声或误差的鲁棒性未作系统评估。

---

## 566. Beyond Global Alignment: Fine-Grained Motion-Language Retrieval via Pyramidal Shapley-Taylor Learning

**arXiv ID:** 2601.21904 | [PDF](https://arxiv.org/pdf/2601.21904v1)

**作者:** Hanmo Chen `[一作]` (Hangzhou Institute of Technology, Xidian University), Cheng Deng `[通讯]` (Xidian University)

**通讯引用:** 12439 | [OpenAlex ID](https://openalex.org/A5015874725)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于金字塔结构的细粒度动作-语言检索框架PST，能够在关节、段落和整体三个层次上逐级对齐动作与文本；

**💡 创新点**

核心创新在于引入Shapley‑Taylor相互作用(STI)来量化跨模态元素之间的交互强度，并将其嵌入金字塔对齐策略，实现从局部到全局的细粒度对应；

**🔧 技术方法**

技术手段包括STI估计头（轻量级卷积+自注意力网络）、金字塔级别的token压缩（KNN‑DPC）、多级对比学习以及自蒸馏一致性约束；

**📊 数据集**

实验数据集为HumanML3D与KIT‑ML，分别覆盖多姿态与步态两类动作；

**📈 对比分析**

与TEMOS、T2M、TMR、MotionPatch等现有方法比较，PST在R@1、R@5、R@10和MedR指标上均取得明显提升，刷新两大数据集的最佳记录；

**⚠️ 局限性**

局限性主要是对常见或频繁姿态的偏好，稀有或复杂动作的检索效果仍不理想，后续工作计划扩展到开放词汇场景并缓解局部偏差。

---

## 567. Not All Code Is Equal: A Data-Centric Study of Code Complexity and LLM Reasoning

**arXiv ID:** 2601.21894 | [PDF](https://arxiv.org/pdf/2601.21894v1)

**作者:** Lukas Twist `[一作]` (King's College London), Jie M. Zhang `[通讯]` (King's College London)

**通讯引用:** 4043 | [OpenAlex ID](https://openalex.org/A5088708850)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对LLM在微调过程中代码结构复杂度对推理能力的影响进行系统实验研究

**💡 创新点**

首次将代码的环路复杂度和逻辑行数作为可度量的结构特征，发现中等复杂度的代码能最大化提升推理性能，挑战“代码多样性即越好”的假设

**🔧 技术方法**

使用LoRA微调技术，并借助静态分析工具计算 cyclomatic complexity 与 logical lines of code，评估多种开源模型

**📊 数据集**

构造两类控制复杂度的数据集：基于 CodeNet 的 solution‑driven 数据集，以及基于 Magicoder、Evol‑Instruct、WizardLM 的 problem‑driven 数据集

**📈 对比分析**

在六个公开推理基准（MATH、GSM8K、Algebra、Aqua、BBH‑ExtraHard、HLU）上评估，约 83% 的实验显示限制特定复杂度范围的微调比混合代码更能提升性能，提升幅度从数个百分点到十几个百分点不等

**⚠️ 局限性**

实验仅涵盖少数主流语言与开源模型，使用的复杂度指标有限，样本量较小，未验证更大规模或更细粒度的代码特征对推理的进一步影响

---

## 568. How Expressive Are Graph Neural Networks in the Presence of Node Identifiers?

**arXiv ID:** 2601.21882 | [PDF](https://arxiv.org/pdf/2601.21882v1)

**作者:** Arie Soeteman `[一作]` (University of Amsterdam), Balder ten Cate `[通讯]` (University of Amsterdam)

**通讯引用:** 2703 | [OpenAlex ID](https://openalex.org/A5080220262)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在图神经网络（GNN）中加入唯一节点标识符（key）后，保持键不变时的可表达性（key‑invariant）以及不同组合函数和接受策略对表达能力的影响。

**💡 创新点**

首次给出 key‑invariant GNN 的逻辑与组合函数层级，证明其比传统 key‑oblivious GNN 更强，且在某些组合函数下可表达所有强局部查询；同时提供多种下确性与可判定性边界，并与秩序不变一阶逻辑（order‑invariant FO）建立上界。

**🔧 技术方法**

利用有限模型论中的秩序不变逻辑、格子化细化、同构与覆盖理论、动态逻辑（PDL）和瞬时注入函数等技术，对 GNN 的可表达性进行正式证明；同时引入“键空间约束”和“组合函数分层”等新概念。

**📊 数据集**

本工作为理论研究，不使用实验数据集；所有结论均来自形式化证明与逻辑解释。

**📈 对比分析**

由于没有实验对比，无法给出数值性能指标；论文通过逻辑归约与下确性/可判定性结果展示 key‑invariant GNN 在表达能力上的提升与局限，理论上与传统 GNN 形成对比。

**⚠️ 局限性**

主要局限包括：尚未完全确定键‑不变 GNN 与一阶逻辑或更强逻辑的关系、不同键空间（如离散或连续）下的精确表达边界未知，以及对全局聚合或更一般的图读取层未作充分研究。

---

## 569. Retrieval-Infused Reasoning Sandbox: A Benchmark for Decoupling Retrieval and Reasoning Capabilities

**arXiv ID:** 2601.21937 | [PDF](https://arxiv.org/pdf/2601.21937v1)

**作者:** Shuangshuang Ying `[一作]`, Ge Zhang `[通讯]` (ByteDance)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了DeR²实验平台，构建了一个可控的检索-推理基准，用来分别评估检索、去噪与推理的能力；

**💡 创新点**

其创新点在于：①将检索与推理解耦，设计四种评估情景（Instruction‑only、Concepts‑only、Related‑only、Full‑set）；②采用两阶段可验证性协议，保证任务无法仅靠参数知识完成；③在文档集合中加入与任务主题相关但概念无关的噪声，模拟真实检索环境；

**🔧 技术方法**

使用技术包括：大语言模型推理、检索增强生成（RAG）框架、概念级别抽取、链式推理（CoT）分析、自动化错误归因与性能评估；

**📊 数据集**

数据集来源于2023‑2025年理论论文，手工构造指令、概念、CoT、答案和对应的文档集合（包含相关文档与噪声文档），总共约数千个实例；

**📈 对比分析**

对比方法：在同一套四种评估情景下测试多款商业与开源LLM，采用准确率与检索损失（Retrieval Loss）等指标；结果显示概念级推理最佳（≈80%），检索+噪声场景约降低20‑30%，揭示模型在模式切换、概念误用与结构检索方面的显著瓶颈；

**⚠️ 局限性**

局限性包括：①需要大量人工标注与审核，成本高；②仅覆盖理论论文，缺乏实验/数据驱动问题；③文档为静态快照，无法评估对动态网络的适应；④尽管设定多样，但仍未完全覆盖所有检索与推理交互的复杂性。

---

## 570. Just Noticeable Difference Modeling for Deep Visual Features

**arXiv ID:** 2601.21933 | [PDF](https://arxiv.org/pdf/2601.21933v1)

**作者:** Rui Zhao `[一作]` (Nanyang Technological University), Weisi Lin `[通讯]` (Nanyang Technological University)

**通讯引用:** 30222 | [OpenAlex ID](https://openalex.org/A5100403129)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了适用于深度视觉特征的“可察觉差异”（FeatJND）模型，预测在不影响下游任务性能的前提下可容忍的最大特征扰动。

**💡 创新点**

创新点在于将人类视觉系统中的JND概念迁移到机器视觉特征空间，定义了基于任务性能的可观测边界，并通过可学习的估计器实现对特征的空间可变扰动预测。

**🔧 技术方法**

使用卷积网络作为FeatJND估计器，利用温度缩放的KL散度和Smooth L1等任务差异度量构建损失函数；同时设计了多任务兼容的训练框架，并将其应用于特征量化。

**📊 数据集**

在ImageNet-1K（分类）和COCO2017（检测与实例分割）上进行实验，评估了多种CNN和Transformer骨干网络。

**📈 对比分析**

与均值为0的高斯噪声扰动进行对比，结果显示在相同NRMSE（噪声强度）下，FeatJND扰动能保持更高的任务性能；在基于token的动态量化实验中，FeatJND引导的步长分配比随机或全局均匀量化更优。

**⚠️ 局限性**

局限性包括：目前的FeatJND估计器针对单一任务/模型训练，缺乏跨任务迁移能力；仅使用简单的卷积结构，未探索更高效或结构化噪声模式；以及在大规模模型中的计算与存储开销尚待进一步评估。

---

## 571. Localizing Speech Deepfakes Beyond Transitions via Segment-Aware Learning

**arXiv ID:** 2601.21925 | [PDF](https://arxiv.org/pdf/2601.21925v1)

**作者:** Yuchen Mao `[一作]` (Shanghai Jiao Tong University), Yanmin Qian `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种段感知学习（Segment-Aware Learning, SAL）框架，用于定位部分深度伪造音频中的受攻击片段。

**💡 创新点**

创新点在于将注意力从仅识别转移边界扩展到整个段落内部特征，通过段位置标签（Segment Positional Labeling, SPL）实现多任务监督，并引入跨段混合（Cross-Segment Mixing, CSM）数据增强以提升泛化。

**🔧 技术方法**

技术包括使用预训练的自监督学习（SSL）特征提取器（Wav2Vec2‑XLSR / WavLM）、Conformer模块、RawBoost增强、SPL与CSM的联合损失，以及Grad‑CAM可视化分析。

**📊 数据集**

实验使用三大数据集：PartialSpoof、Half‑truth Audio Detection (HAD) 和 LlamaPartialSpoof (LPS)，并在跨域设置下评估鲁棒性。

**📈 对比分析**

在 EER 与 F1 评价指标上与现有边界感知方法（如BAM、AGO等）对比，SAL 在 PS 数据集上实现了最高 F1（97.09%），EER 与顶尖方法相近；在 HAD 与跨域 LPS 上亦取得领先，表明显著提升的定位准确性与泛化能力。

**⚠️ 局限性**

局限性包括：对训练数据分布仍有一定依赖，过度的 CSM 轮数可能导致性能下降；对极短或极长伪造段的识别仍可能受限；需在不同场景中进一步验证可迁移性与模型规模对实时部署的影响。

---

## 572. From Future of Work to Future of Workers: Addressing Asymptomatic AI Harms for Dignified Human-AI Interaction

**arXiv ID:** 2601.21920 | [PDF](https://arxiv.org/pdf/2601.21920v1)

**作者:** Upol Ehsan `[一作]` (Northeastern University, Harvard University), Sara Alcorn `[通讯]` (University of Minnesota)

**通讯引用:** 1321 | [OpenAlex ID](https://openalex.org/A5068250447)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在肿瘤放射治疗工作场景中，开展了为期一年的纵向研究，追踪了 AI 辅助治疗规划系统（RadPlan）在临床使用过程中的“AI‑放大器悖论”，并基于此研究结果构建了“尊严化人‑AI 交互框架”，用于在医疗和软件工程两个领域内检验其对专业知识工人技能衰退、依赖性上升以及职业身份侵蚀的抑制效果。

**💡 创新点**

创新点主要体现在三方面：①首次系统性揭示 AI 在长期使用中产生的“无症状效应 → 慢性危害 → 身份商品化”三阶段演化过程；②提出并验证“社会透明度（Social Transparency）”与“社会技术免疫（Sociotechnical Immunity）”的双层干预策略；③设计了可跨领域实施的多层级框架（工人层、技术层、组织层），通过“感知-遏制-恢复”三步循环实现对 AI 隐性伤害的主动防御。

**🔧 技术方法**

核心技术主要是人机交互与社会技术研究方法：半结构化访谈、think‑aloud、参与式工作坊、扎根理论编码；技术层面实现了 4W（谁、何时、何因）社会透明度叠加层，用于可视化 AI 与人类决策的交互痕迹；框架设计则基于需求引导的设计方法（starter‑pack 设计问题）与迭代原型验证。

**📊 数据集**

数据来源为真实临床环境：在五个医院站点共收集 42 名从业者（放射肿瘤科医生、医学物理学家、剂量师、医院管理者）的 24 次访谈、5 次工作坊、52 次现场思考‑说话、RadPlan 运行日志与 4W 记录；软件工程验证使用了 AI 辅助编码工具（AutoCoder）的仓库提交日志与代码覆盖率数据。

**📈 对比分析**

评估方法以参与式场景模拟为主：在医疗领域让专家团队在 RadPlan 上按照框架提出应对方案，在软件领域让开发者团队在 AI 辅助编码环境中演练。效果通过定性访谈收集反馈（如“恢复专业判断”“提升工作满意度”）和定量日志指标（如“即时批准率”“手动计划比例”）对比前后变化。整体表现显示，框架能在保持或提升效率的同时，显著降低即时批准率、提高手动复核频次，并获得用户对专业身份保留的正面评价。

**⚠️ 局限性**

局限性包括：①研究仅聚焦单一高风险医疗场景，无法直接推广至其他行业；②样本规模相对有限，且多为有 AI 使用经验的专业人士，可能存在自选偏差；③框架评估主要以定性反馈与过程日志为依据，缺乏客观的长期效能量化基准；④社会透明度层对隐私与合规的约束可能限制其在更广泛机构中的部署。

---

## 573. Self-Compression of Chain-of-Thought via Multi-Agent Reinforcement Learning

**arXiv ID:** 2601.21919 | [PDF](https://arxiv.org/pdf/2601.21919v1)

**作者:** Yiqun Chen `[一作]` (Renmin University of China), Jiaxin Mao `[通讯]` (Renmin University of China)

**通讯引用:** 2395 | [OpenAlex ID](https://openalex.org/A5072119199)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种多智能体强化学习框架 SCMA，用于在不增加推理时开销的前提下，对大型推理模型的链式思考进行细粒度压缩。

**💡 创新点**

创新点在于：①引入分段（Segmentation）和评分（Scoring）两类辅助智能体，分别负责将推理过程切分为逻辑块并评估其重要性；②用重要性加权的长度惩罚代替传统粗粒度长度惩罚，实现只压缩冗余而保留核心逻辑；③三体共享同一基线LLM参数，在训练阶段实现协同优化，推理阶段仅保留推理智能体。

**🔧 技术方法**

技术手段包括：多智能体强化学习（GRPO 的多智能体变体）、共享参数策略、基于 XML 标记的分段与评分规范、重要性加权长度惩罚函数、格式化奖励、以及对大型语言模型（DeepSeek-R1‑Distill‑Qwen、Qwen3）的微调。

**📊 数据集**

实验数据集：GSM8K、MATH500、AMC23、AIME24/25；模型基线为 DeepSeek‑R1‑Distill‑Qwen（1.5B/7B）和 Qwen3（4B/8B）。

**📈 对比分析**

与 vanilla、GRPO、LC‑R1_LP、RL+LP 等基线对比，SCMA 在所有模型规模上平均缩短思考链 11.1%–39.0% 的长度，同时提升 4.33%–10.02% 的准确率；推理阶段无额外计算负担，且训练过程更稳定。

**⚠️ 局限性**

局限性：①需要在训练阶段额外部署多智能体，训练成本相对较高；②重要性分数的设定与长度惩罚系数 α 仍需经验调参；③目前仅在 GSM8K 训练，尽管在其他任务上表现良好，但对更大规模或更复杂场景的泛化性尚未彻底验证。

---

## 574. JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG

**arXiv ID:** 2601.21916 | [PDF](https://arxiv.org/pdf/2601.21916v1)

**作者:** Yiqun Chen `[一作]` (Renmin University of China), Jiaxin Mao `[通讯]` (Renmin University of China)

**通讯引用:** 2395 | [OpenAlex ID](https://openalex.org/A5072119199)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 JADE 框架，统一了动态 Agentic RAG 的规划与执行，实现了规划器与执行器的联合优化。

**💡 创新点**

创新点在于将 RAG 的规划与执行视为协同多智能体游戏，通过共享参数实现规划与执行的互相适配，消除了“战略-操作失配”。

**🔧 技术方法**

核心技术包括基于共享 LLM 背骨的多智能体半马尔可夫决策过程 (MSMDP)、PPO 强化学习、统一经验回放以及全局奖励与局部格式惩罚的奖励设计。

**📊 数据集**

实验使用了七个开放域 QA 基准，包括单跳 NQ、PopQA、AmbigQA 以及多跳 HotpotQA、2WikiMultiHopQA、Musique、Bamboogle。

**📈 对比分析**

与三大类基线（静态模块化、适配工作流、单体搜索）对比，JADE 在 F1 方面平均提升 8.29 点，尤其在多跳任务上显著优于 MAO‑ARAG 与 Search‑R1。

**⚠️ 局限性**

局限在于需要细致的奖励调参以防止过度惩罚导致退化为单轮 RAG；此外，当前实现仍依赖大型 LLM，虽然比单体大模型更高效，但在极端低资源场景下表现未被充分验证。

---

## 575. LoRIF: Low-Rank Influence Functions for Scalable Training Data Attribution

**arXiv ID:** 2601.21929 | [PDF](https://arxiv.org/pdf/2601.21929v1)

**作者:** Shuangqi Li `[一作]` (EPFL), Mathieu Salzmann `[通讯]` (EPFL)

**通讯引用:** 13679 | [OpenAlex ID](https://openalex.org/A5049300388)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种新的训练数据归因方法 LoRIF，能够在大规模模型和海量训练集上高效计算样本影响度。

**💡 创新点**

创新点包括：1) 通过低秩因子化存储每个样本的投影梯度，显著减少存储和 I/O；2) 使用截断 SVD 与 Woodbury 恒等式在低维子空间中近似逆 Hessian，降低 O(D²) 内存需求；3) 在保持或提升归因质量的前提下，支持更大的投影维度 D，从而突破质量-可扩展性权衡。

**🔧 技术方法**

核心技术：随机两侧投影、低秩矩阵分解、随机 SVD、Woodbury 恒等式、Gauss–Newton Hessian 近似、梯度投影矩阵的稀疏化。

**📊 数据集**

实验数据集：
- GPT‑2‑small（124M 参数）在 WikiText‑103 训练/验证集；
- Olmo‑3‑7B（7B 参数）Finetune‑SFT（2.2M 示例）；
- Apertus‑70B（70B 参数）Finetune‑SFT（3.8M 示例）。

**📈 对比分析**

与 LoGRA、EK‑FAC、GradDot、RepSim 等方法对比：在相同存储预算下，LoRIF 通过增大 D 维度保持或超过 LoGRA 的 LDS/Tail‑patch 分数；在相同归因质量下，LoRIF 的存储空间为 LoGRA 的 1/10–1/20，查询时间 30–100 倍更快；在 70B 模型上，LoRIF 仅占 LoGRA 20% 的存储并将查询延迟降低约 80%。

**⚠️ 局限性**

局限性：1) 仅在模型的 Fine‑Tuning 阶段验证，未对完整 Pre‑Training 规模进行实验；2) 计算每个样本梯度仍昂贵，单次梯度通道成本与训练成本相当；3) 低秩近似误差可能在极端场景下影响归因精度。

---

## 576. A Separable Architecture for Continuous Token Representation in Language Models

**arXiv ID:** 2601.22040 | [PDF](https://arxiv.org/pdf/2601.22040v1)

**作者:** Reza T. Batley `[一作]` (Virginia Polytechnic Institute and State University), Sourav Saha `[通讯]` (Virginia Polytechnic Institute and State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种将词表嵌入矩阵替换为连续可分离生成器的Transformer架构（Leviathan），实现了在小型语言模型中更高效的参数利用。

**💡 创新点**

创新点在于将词表映射到低维坐标空间，通过B-spline基函数和张量乘积构建连续表面，从而消除嵌入层对词表大小的线性依赖，并实现“深度红利”。

**🔧 技术方法**

采用了Separable Neural Architecture（SNA）生成器、B-spline基函数、张量乘积聚合以及RoPE位置编码、SwiGLU前馈网络等技术；模型训练使用AdamW、梯度裁剪等常规优化。

**📊 数据集**

使用了Pile数据集，词表为200,376（59³）大小，tokenizer为tiktoken GPT-4o/4o-mini。

**📈 对比分析**

通过与传统Dense模型（使用tied embedding）的对比，在相同参数量或相同Transformer主体的情况下，Leviathan在验证困惑度和损失上平均降低6.7%–18.1%，等效于1.5–2.1倍更大的Dense模型，且在超训练阶段性能持续提升。

**⚠️ 局限性**

主要限制包括生成器带来的计算吞吐量下降（尤其在浅层模型中可达51%），以及目前实现对现代加速器的优化不足；此外，基于固定的坐标映射对语义结构的适应性仍需进一步验证。

---

## 577. Thinking Out of Order: When Output Order Stops Reflecting Reasoning Order in Diffusion Language Models

**arXiv ID:** 2601.22035 | [PDF](https://arxiv.org/pdf/2601.22035v1)

**作者:** Longxuan Yu `[一作]` (University of California, Riverside), Yue Dong `[通讯]` (University of California, Riverside)

**通讯引用:** 2965 | [OpenAlex ID](https://openalex.org/A5077832390)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探究当输出顺序与自然推理顺序不匹配时，语言模型的推理表现，并提出并验证“顺序鲁棒性”，通过构建可控难度基准 ReasonOrderQA 对比 AR 与掩码扩散模型的性能。

**💡 创新点**

提出顺序鲁棒性评估框架，发现掩码扩散模型凭借置信度驱动的解码能自发实现“先推理后答”顺序；构建可调难度基准并揭示扩散模型鲁棒性的机制与极限。

**🔧 技术方法**

使用掩码扩散语言模型（如 LLaDA）与低置信度重新掩码采样；对比传统自回归模型（Qwen）；利用置信度/熵等不确定性指标跟踪内部预测动态。

**📊 数据集**

实验数据集包括 GSM8K、Math500 以及新构建的 ReasonOrderQA（1000道算术推理题，分为 4 级难度）。

**📈 对比分析**

在标准 CoT‑First 与逆序 Answer‑First 两种输出顺序下测量准确率。AR 模型在逆序下准确率下降高达 67%；掩码扩散模型仅下降 ≤14%；在 ReasonOrderQA 各难度层级，扩散模型保持 4%–8% 的相对稳定。

**⚠️ 局限性**

鲁棒性受令牌复杂度差异和生成长度限制；当令牌难度差异不足或生成长度过大时，鲁棒性下降；从 AR 蒸馏得到的扩散模型也会削弱鲁棒性；长序列或动态长度的适应仍需进一步研究。

---

## 578. The Ensemble Inverse Problem: Applications and Methods

**arXiv ID:** 2601.22029 | [PDF](https://arxiv.org/pdf/2601.22029v1)

**作者:** Zhengyan Huan `[一作]` (Tufts University), Shuchin Aeron `[通讯]` (Tufts University)

**通讯引用:** 3301 | [OpenAlex ID](https://openalex.org/A5004738943)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种新的“集成逆问题”（Ensemble Inverse Problem, EIP）框架，并给出了针对 EIP-II 的非迭代后验采样方法——EI‑DDPM 与 EI‑FM，利用观测集的集合信息构造后验分布。

**💡 创新点**

创新点在于：①将观测集作为集合输入，通过置换不变的神经网络提取集成信息；②在条件扩散或流匹配模型中联合观测与集成信息进行后验建模；③在推断时无需显式使用前向模型，训练过程中通过多组真值‑观测对隐式学习似然；④展示了对未知先验的泛化能力。

**🔧 技术方法**

使用的技术包括：条件扩散概率模型（cDDPM）和条件流匹配模型（cFM）的变体 EI‑DDPM/EI‑FM；置换不变的集合编码网络（如 Deep Set 或 Set Transformer）提取 ϕ_w(𝒴)；对数似然/噪声重构损失进行训练；在推断时采用反向扩散或流迭代采样。

**📊 数据集**

实验数据集涵盖：①二维高斯扰动的合成数据；②MNIST 混合图像（插值+噪声）；③高能物理 QCD 轰击（包括多种物理进程的 jet 观测）；④地震全波形反演（8 类地下结构），并在两类未见结构上测试。

**📈 对比分析**

比较方法包括：cDDPM、cFM、GDDPM（及其多元扩展）、Omnifold（两种初始化）、SBUnfold、Sourcerer 等。评估指标有：Sliced Wasserstein Distance (SWD)、Wasserstein 1-distance、MSE、SSIM、MAE。实验结果显示 EI‑DDPM/EI‑FM 在大多数任务上优于或与最优基线持平，尤其在对未知先验的泛化方面显著提升。

**⚠️ 局限性**

局限性：①需要足够大的观测集 N 以保证 ϕ_w 能充分捕捉先验信息；②在 N 与训练时不同的情况下需要采样或复制策略，可能影响精度；③对观测噪声和前向模型变异的鲁棒性尚未完全理论化；④训练时仍需大量真值‑观测对，实际部署时可获取性受限。

---

## 579. From Logits to Latents: Contrastive Representation Shaping for LLM Unlearning

**arXiv ID:** 2601.22028 | [PDF](https://arxiv.org/pdf/2601.22028v1)

**作者:** Haoran Tang `[一作]` (Purdue University), Rajiv Khanna `[通讯]` (Purdue University)

**通讯引用:** 14575 | [OpenAlex ID](https://openalex.org/A5068930801)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出CLReg对LLM进行对比学习正则化，以在特征空间中分离忘记与保留信息，实现高效的机器忘记。

**💡 创新点**

通过对忘记样本与其增强版本构造正向对，使用保留样本作为负样本，形成无监督对比学习框架；并给出理论证明其可降低忘记-保留特征混合度。

**🔧 技术方法**

对比学习（InfoNCE/DPO）、梯度上升/下降、损失重构、均衡温度、Dropout/改写增强、L2正则化、层级聚合等技术。

**📊 数据集**

TOFU（包含Llama‑3.1‑8B、Llama‑3.2‑3B）、MUSE（Books、News）以及相应的 retrained 版本，使用标准的隐私泄露、召回率、ROUGE 等评测指标。

**📈 对比分析**

与 GradDiff、UnDIAL、NPO、SimNPO、PDU 等主流未学习方法对齐；在大多数实验中 CLReg 提升了 Unlearning Score 和 Forget Score，几乎不影响 Model Utility，且在大部分情况下将隐私泄露值逼近零。

**⚠️ 局限性**

仍需在更大模型和更复杂数据集上验证，负样本选取与增强方式可能对效果有较大影响；对比学习正则化在早期层使用时性能下降，需谨慎选择正则化层级。

---

## 580. Putting a Face to Forgetting: Continual Learning meets Mechanistic Interpretability

**arXiv ID:** 2601.22012 | [PDF](https://arxiv.org/pdf/2601.22012v1)

**作者:** Sergi Masip `[一作]` (KU Leuven), Tinne Tuytelaars `[通讯]` (KU Leuven)

**通讯引用:** 52301 | [OpenAlex ID](https://openalex.org/A5074816094)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一个基于特征几何变换的机制解释框架，用于分析和诊断连续学习中的灾难性遗忘。

**💡 创新点**

创新点在于把遗忘归因于特征向量的旋转、缩放以及读出失配，并给出了最佳与最差场景的理论分析和可测量指标。

**🔧 技术方法**

核心技术包括线性表示假设、特征读者模型、稀疏自编码器、Crosscoder 共享潜在空间以及梯度下降解析。

**📊 数据集**

实验涵盖了合成的多任务回归数据、Split CIFAR‑10 上的 Vision Transformer（ViT），并使用多种任务顺序与随机种子验证。

**📈 对比分析**

与传统的性能指标（如准确率）和表示相似度方法（如CKA）比较，框架能细粒度区分重叠、衰减和读出失配，对深度网络的遗忘机制提供更深刻的解释。

**⚠️ 局限性**

主要局限包括对线性表示假设的依赖、稀疏自编码器的近似性以及在大规模模型与数据集上的可扩展性待进一步验证。

---

## 581. Rate-Distortion Optimization for Transformer Inference

**arXiv ID:** 2601.22002 | [PDF](https://arxiv.org/pdf/2601.22002v1)

**作者:** Anderson de Andrade `[一作]` (Simon Fraser University), Ivan V. Bajić `[通讯]` (Simon Fraser University)

**通讯引用:** 4372 | [OpenAlex ID](https://openalex.org/A5012187461)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于率失真（RD）框架，用于压缩Transformer中间表示，并设计了仅使用超先验的自回归熵模型；

**💡 创新点**

创新点在于引入V‑entropy差距理论，将压缩率与信息熵关联，并证明仅利用超先验即可实现更低比特率，同时提供PAC式泛化误差界限；

**🔧 技术方法**

采用Transformer架构实现超先验网络与熵模型，配合可微量化、基于超先验的自回归熵编码，并使用Rademacher复杂度与Lipschitz常数进行理论分析；

**📊 数据集**

实验数据集包括语言建模的OpenWebText（用于GPT‑2 Small、Pythia 160M）以及图像分类的ViT B/16和ResNet34；

**📈 对比分析**

与基于Fourier基函数和直接访问自回归熵模型的基线进行率失真曲线对比，BD‑rate分别低99.46%和10.7%，在比特率、推理速度和任务困惑度上均优于无压缩或传统无损压缩（Deflate、Zstandard）；

**⚠️ 局限性**

局限性在于实现尚未充分优化，推理速度仍慢于Zstandard；仅针对Transformer及其超先验的设计，未验证更大规模模型；压缩率受熵模型容量与泛化误差平衡限制。

---

## 582. Negatives-Dominant Contrastive Learning for Generalization in Imbalanced Domains

**arXiv ID:** 2601.21999 | [PDF](https://arxiv.org/pdf/2601.21999v1)

**作者:** Meng Cao `[一作]` (Nanjing University of Aeronautics and Astronautics), Songcan Chen `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 13614 | [OpenAlex ID](https://openalex.org/A5101596072)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种针对同时存在域漂移和标签不平衡的Imbalanced Domain Generalization（IDG）问题的新框架

**💡 创新点**

通过理论分析建立了包含后验分布差异与决策边界间隔的泛化界限，并设计了Negative-Dominant Contrastive Learning（NDCL）来直接调节决策边界

**🔧 技术方法**

采用负样本主导的对比学习（InfoNCE改写）、基于置信度的硬负采样、类别加权交叉熵以及跨域预测中心对齐等技术

**📊 数据集**

在VLCS、PACS和OfficeHome三个常用域推广基准上，构造了GINIDG、TotalHeavyTail、Duality三种不同的标签与域不平衡场景

**📈 对比分析**

与21种基线（传统域推广、长尾学习、IDG专用方法）和8种公开实现进行对比，NDCL在所有设置下均获得平均准确率最高，尤其在少数类上显著提升

**⚠️ 局限性**

对高置信度负样本采样与跨域对齐策略的理论解释尚不充分，且在极端不平衡场景下对后验分布估计仍可能不稳定

---

## 583. Liquid Interfaces: A Dynamic Ontology for the Interoperability of Autonomous Systems

**arXiv ID:** 2601.21993 | [PDF](https://arxiv.org/pdf/2601.21993v1)

**作者:** Dhiogo de Sá `[一作]`, Carlos Pereira Lopes `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出液态接口概念及其协议 LIP，支持在自治代理间以意图为驱动的临时协调。

**💡 创新点**

核心创新在于将接口从静态契约转为基于语义协商的短暂事件，消除技术债务并实现意图级授权与审计。

**🔧 技术方法**

采用意图驱动的消息类型、LLM语义裁定、加密身份验证、基于声明的授权以及可撤销的协商流程。

**📊 数据集**

未使用特定公开数据集，实验基于概念证明实现。

**📈 对比分析**

论文未给出实验对比或性能指标，主要是理论与原型阐述。

**⚠️ 局限性**

局限包括：不适用于硬实时或需确定性保证的场景；信任与能力验证依赖声明；分布式实现仍需研究；可能与合规性要求冲突。

---

## 584. Adaptively Robust Resettable Streaming

**arXiv ID:** 2601.21989 | [PDF](https://arxiv.org/pdf/2601.21989v1)

**作者:** Edith Cohen `[一作]` (Google Research), Uri Stemmer `[通讯]` (Tel Aviv University)

**通讯引用:** 1038 | [OpenAlex ID](https://openalex.org/A5019495492)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计了可在自适应攻击下保持准确性的 resettable 流数据流量统计 sketch，涵盖基数、求和以及 Bernstein（软凸子线性）统计量。

**💡 创新点**

首次在 resettable 模型中实现自适应鲁棒性，突破了传统线性/可组合 sketch 的多项式空间下限；通过将差分隐私的二叉树机制嵌入采样 sketch，构造了高效的、对攻击无偏的估计。

**🔧 技术方法**

核心技术包括：差分隐私、连续观测的二叉树机制、可调节采样率的 sketch、对统计量分解为可保护与确定性成分、Bernstein 函数的稀疏化与线性组合化还原。

**📊 数据集**

本文为理论工作，未使用具体数据集；所有结果均基于理论分析与证明。

**📈 对比分析**

相较于之前的非自适应 sketch，本文在保持相同误差和置信度的前提下，空间复杂度从多项式降至 polylog(T)，并提供了前缀最大误差（prefix‑max）保证；在理论上与最优线性空间方案相媲美。

**⚠️ 局限性**

局限性：尚未覆盖超线性统计（如 ∫_2）或一般 ReLU 模型；对大规模重置操作的支持仍不完备；空间上虽为 polylog，但相较于最优仍有提升空间。

---

## 585. PowerGenie: Analytically-Guided Evolutionary Discovery of Superior Reconfigurable Power Converters

**arXiv ID:** 2601.21984 | [PDF](https://arxiv.org/pdf/2601.21984v1)

**作者:** Jian Gao `[一作]` (Northeastern University), Xuan Zhang `[通讯]` (Northeastern University)

**通讯引用:** 16135 | [OpenAlex ID](https://openalex.org/A5100342935)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种自动化框架，利用演化微调发现更高效的可重构SC电源转换器拓扑，且能在不进行显著仿真的前提下评估功能和性能。

**💡 创新点**

创新点在于：①将电路拓扑解析为图并通过自洽算法快速计算SSL/FSL/面积指标；②构建演化式微调循环，使生成模型与训练集共进化，既保持多样性又提升性能。

**🔧 技术方法**

采用图论与线性代数的自动化性能分析，结合GPT-基础生成模型和演化选择/遗传算子、图同构判定。

**📊 数据集**

使用规模达11,837条电路的公开数据集，其中3,824条为8模式可重构电源转换器，包含所有目标VCR。

**📈 对比分析**

与PPO、DPO、LaMAGIC、AUTOCIRCUIT-RL等方法比较，PowerGenie在语法/功能有效率、创新率和FoM上分别提升至90.8%、85.3%、32.1%及0.323，发现的8模式转换器FoM比训练集高23%，SPICE验证平均效率提升10%至17%。

**⚠️ 局限性**

仅针对SC电路，未涵盖非SC拓扑；演化过程仍可能受到训练集偏差影响；实际物理Pdk评估尚需进一步验证。

---

## 586. OVD: On-policy Verbal Distillation

**arXiv ID:** 2601.21968 | [PDF](https://arxiv.org/pdf/2601.21968v1)

**作者:** Jing Xiong `[一作]` (University of Hong Kong), Ngai Wong `[通讯]` (University of Hong Kong)

**通讯引用:** 12070 | [OpenAlex ID](https://openalex.org/A5043990959)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于轨迹匹配的On‑policy Verbal Distillation（OVD）框架，将大模型教师的推理能力通过离散的口头评分（0–9）迁移给小模型学生，取代传统的token‑level概率匹配，实现了在强化学习中的高效对齐和自我探索。

**💡 创新点**

创新点包括：
• 用口头评分代替完整词表logits，显著降低内存占用；
• 采用verbal rejection sampling生成符合质量阈值的轨迹，形成学生-教师混合训练分布；
• 在on‑policy框架下证明该采样策略无偏并能降低梯度方差；
• 兼容黑盒教师模型，允许学生自由探索输出空间；
• 通过step‑level与trajectory‑level双重监督实现更细粒度的信用分配。

**🔧 技术方法**

核心技术：
- PPO/GRPO强化学习优化；
- on‑policy轨迹采样与verbal rejection sampling；
- 交互式环境代理（Web检索）和推理代理（大模型评估）提供口头评分；
- ZeroSearch模拟检索环境；
- 训练时混合学生轨迹与教师轨迹的分布；
- 理论分析证明无偏梯度与方差降低。

**📊 数据集**

实验数据集：
• Web Q&A：NQ、TriviaQA、PopQA、HotpotQA、2Wiki、Musique、Bamboogle、GAIA；
• 数学推理：SVAMP、ASDiv、MAWPS、TABMWP、Minerva、OlympiadBench、SAT‑Math、MMLU、Gaokao、AIME2024。

**📈 对比分析**

与传统检索+RL基线（Search‑o1、Search‑R1、ZeroSearch）以及RLVR对比。OVD在Web Q&A上平均提升约+12.9% EM（如Qwen‑2.5‑3B在Search‑R1上提升10.8点），在数学推理上提升约+25.7%（仅用1个随机样本训练），并显示出更高的样本效率和更快收敛速度。

**⚠️ 局限性**

局限性：
- 需要教师模型能够提供离散口头评分，限制了适用场景；
- 评分粒度受10级离散化限制，细粒度改进需更大v；
- 对非可验证任务或需要更细粒度自监督的场景适应性未知；
- 虽内存减小，但仍需大量轨迹采样，训练时间与轨迹长度相关。

---

## 587. From Tokens to Blocks: A Block-Diffusion Perspective on Molecular Generation

**arXiv ID:** 2601.21964 | [PDF](https://arxiv.org/pdf/2601.21964v1)

**作者:** Qianwei Yang `[一作]` (Shenzhen University), Junkai Ji `[通讯]` (Shenzhen University)

**通讯引用:** 1586 | [OpenAlex ID](https://openalex.org/A5046906366)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 SoftMol 框架，实现目标感知的分子生成，融合软片段表示、块扩散语言模型 SoftBD 与门控 MCTS 搜索。

**💡 创新点**

创新点包括：1）无规则软片段分子表示，兼顾可扩展性与化学语法鲁棒性；2）首次在块级扩散中实现局部双向扩散与全局自回归的统一；3）门控 MCTS 将配体结合能与药物适性阈值分离，显著提升搜索效率和药理质量。

**🔧 技术方法**

核心技术为块扩散 Transformer、Adaptive Confidence Decoding（First‑Hitting 采样 + 贪婪置信解码 + 批量推理）以及带可调可行性门的 MCTS。

**📊 数据集**

训练集为精细筛选的 ZINC‑Curated（约 4.27 亿分子，SMILES 长度 ≤ 72），用于提升药物类性质；测试集覆盖 MOSES 指标和 5 个靶点的对接评分。

**📈 对比分析**

与 SAFE‑GPT、GenMol 等最先进基准对比，SoftMol 在无目标生成中 100% 化学有效性、>80% 质量、2–3 倍多样性、采样速度提升 6.6×；在目标特定生成中，平均对接评分提高 9.7% 以上，Hit Ratio 与多样性均超过对手。

**⚠️ 局限性**

局限性包括：对高层次药理评估仍依赖后处理对接，门控阈值需手工设定；软片段长度与计算资源权衡复杂，极大块会导致推理时间增长；在极端化学空间或稀缺靶点数据下，模型迁移性能未充分验证。

---

## 588. The Energy Impact of Domain Model Design in Classical Planning

**arXiv ID:** 2601.21967 | [PDF](https://arxiv.org/pdf/2601.21967v1)

**作者:** Ilche Georgievski `[一作]` (University of Stuttgart), Marco Aiello `[通讯]` (University of Stuttgart)

**通讯引用:** 4689 | [OpenAlex ID](https://openalex.org/A5054184668)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统研究了域模型配置（包括语法排序、冗余元素和任务设计）对经典规划器能源消耗的影响，构建了Domain Model Configuration Framework并在五个IPC基准域与五个主流规划器上进行实验；

**💡 创新点**

首次将能源效率纳入规划领域，提出可控的域模型变异框架，揭示域模型结构与规划器能耗之间的非线性关系，尤其是冗余参数和死端状态对能耗的显著放大作用；

**🔧 技术方法**

采用基于Intel RAPL的功耗测量、PDDL 1.2语法变异、随机化多次跑测实验、Pearson相关性与能耗/运行时统计分析等技术手段；

**📊 数据集**

使用IPC 2022/2023的五个基准域（Barman、Blocks World、Gripper、Thoughtful、Ricochet Robots）及相应的十个实例；

**📈 对比分析**

通过对比原始域与32种变体在五个规划器（FDSSA、CA、DALAIA、ANST、FBNS）下的能耗与运行时，发现绝大多数语法变动能耗几乎不变，但冗余动作参数可使能耗翻倍甚至十倍；性能差异主要体现在能耗而非运行时；

**⚠️ 局限性**

实验仅覆盖经典PDDL 1.2规划器和IPC基准域，未涉及高阶PDDL特性、规划框架多样性、真实场景或深度内部组件分析，导致结果对其他规划器/域可能不具备完全可推广性。

---

## 589. Optimizing Agentic Workflows using Meta-tools

**arXiv ID:** 2601.22037 | [PDF](https://arxiv.org/pdf/2601.22037v1)

**作者:** Sami Abuzakuk `[一作]` (École Polytechnique Fédérale de Lausanne), Martijn de Vos `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 724 | [OpenAlex ID](https://openalex.org/A5010233454)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过分析 LLM 代理的执行轨迹，识别并合并重复的工具调用序列，生成可直接调用的“meta‑tool”，从而在不牺牲灵活性的前提下减少中间推理步骤；

**💡 创新点**

提出了 Agent Workflow Optimization 框架，首次将执行轨迹压缩与 meta‑tool 编译结合，显著降低 LLM 调用次数和推理成本；

**🔧 技术方法**

使用状态图构建、水平/垂直图合并、权重阈值剪枝等图算法来抽取 meta‑tool；

**📊 数据集**

在两个公开代理基准上验证：web4a（模拟网页交互）和 AgentBench（九个真实应用的 API 交互）；

**📈 对比分析**

与原始工具集比较，meta‑tool 使 LLM 调用次数下降 5.6%–11.9%，总 token 数和成本降低 4.2%–15%，同时任务成功率提升 0.4%–4.2%；

**⚠️ 局限性**

依赖域知识实现水平合并，难以自动化；meta‑tool 仅对相似任务有效，可能导致对新任务的泛化不足；

---

## 590. Understanding Multimodal Complementarity for Single-Frame Action Anticipation

**arXiv ID:** 2601.22039 | [PDF](https://arxiv.org/pdf/2601.22039v1)

**作者:** Manuel Benavent-Lledo `[一作]` (University of Alicante), Jose Garcia-Rodriguez `[通讯]` (University of Alicante)

**通讯引用:** 5318 | [OpenAlex ID](https://openalex.org/A5079599826)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文研究在仅有单帧视觉观测条件下的人类动作预测，提出改进版 AAG+ 框架。

**💡 创新点**

创新点在于将 RGB、深度和动作历史三模态融合，改进关键帧选择、鲁棒性损失，并采用双向交叉注意力+门控融合来提升单帧预测性能。

**🔧 技术方法**

使用技术包括 DINOv3 ViT 作为视觉编码器、Depth Anything v2 生成深度图、DistilBERT 文本编码、双向交叉注意力与门控融合、以及基于随机噪声和动作交换的历史鲁棒机制。

**📊 数据集**

实验数据集为工业装配场景的 IKEA-ASM、Meccano 与 Assembly101 三个公开视频数据集。

**📈 对比分析**

与原 AAG 以及 AVT、RULSTM、TempAgg、VLMAH 等视频基准对比，单帧 AAG+ 在 IKEA-ASM 与 VLMAH 接近甚至相当，在 Meccano 和 Assembly101 仍保持竞争力，部分指标超过传统视频模型。

**⚠️ 局限性**

局限性包括在长序列、动作空间大、视觉模糊或极其动态的任务中仍受限，且动作历史预测不确定会导致性能波动。

---

## 591. Drive-JEPA: Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving

**arXiv ID:** 2601.22032 | [PDF](https://arxiv.org/pdf/2601.22032v1)

**作者:** Linhan Wang `[一作]` (Virginia Tech), Cheng Lu `[通讯]` (XPENG Motors)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Drive-JEPA 框架，结合 V‑JEPA 视频预训练和多模态轨迹蒸馏，实现端到端驾驶模型的高效学习与决策；

**💡 创新点**

创新点包括：1）在驾驶域内采用 V‑JEPA 进行大规模视频预训练，避免像素级重建开销并防止表征坍塌；2）引入多模态轨迹蒸馏，从仿真器获取多样轨迹作为伪教师，缓解单轨迹监督的模式崩溃；3）设计基于动量的轨迹选择机制，提高舒适度与时序一致性；

**🔧 技术方法**

核心技术包括：ViT 编码器、V‑JEPA 预训练目标、基于 waypoint 的可变形注意力提议生成、模拟器评分的多模态轨迹蒸馏、动量感知轨迹评估与选择、以及轻量级辅助任务（地图映射、碰撞预测）；

**📊 数据集**

使用了三大数据集：CoVLA、DrivingDojo 与 OpenScene 进行视频预训练；NAVSIM v1 与 v2（含 103k 训练、12k 测试场景）用于离线评估；Bench2Drive（CARLA 220 路线）用于闭环评估；

**📈 对比分析**

在 NAVSIM v1/v2 评估中，Drive‑JEPA 在感知自由和感知辅助两种设定下均取得最高 PDMS/EPDMS，分别为 93.3/87.8，领先上一最佳 3–4 PDMS；在 Bench2Drive 上获得最高 Driving Score（64.52）且效率与舒适度表现均优于 iPad 等基线；

**⚠️ 局限性**

主要局限在于：1）仍依赖大量标注视频与仿真器生成轨迹，数据获取成本高；2）多模态蒸馏的轨迹选择对仿真器的准确性高度敏感；3）对实时性能的评估有限，尚未在真实车辆上验证；

---

## 592. Causal Autoregressive Diffusion Language Model

**arXiv ID:** 2601.22031 | [PDF](https://arxiv.org/pdf/2601.22031v1)

**作者:** Junhao Ruan `[一作]` (School of Computer Science and Engineering, Northeastern University), JingBo Zhu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Causal Autoregressive Diffusion（CARD）框架，将自回归模型的训练效率与扩散模型的并行推理相结合。

**💡 创新点**

创新点在于：1) 在扩散过程中采用严格因果注意掩码，确保训练与推理均为单向；2) 引入软尾掩码与上下文自适应重加权，以解决因果扩散导致的早期token信息缺失和梯度不稳定；3) 通过KV缓存和置信度阈值实现动态并行解码，显著提升生成吞吐量。

**🔧 技术方法**

核心技术包括：连续时间噪声调度、软尾掩码（Tail‑Masking）、上下文自适应加权（Context‑aware Reweighting）、基于ELBO的优化目标、并行块采样与置信度阈值控制、KV 缓存支持的动态并行推理。

**📊 数据集**

训练使用 300B tokens 的 FineWeb 子集，模型规模 1B 参数；评估数据集覆盖多任务（ARC‑Challenge/Easy、CommonsenseQA、HellaSwag、MMLU、PIQA、SciQ、Winogrande）和语言建模域（AG News、arXiv、LAMBADA、LM1B、OpenWebText、PTB、PubMed、WikiText）。

**📈 对比分析**

与 ARM、MDLM、BD3LM 等基线比较：CARD 在零样本/少样本任务中平均准确率约 53.2%，比 MDLM 高 5.7%；在 8 个文本域的 PPL 上，CARD 在 6/8 个域均优于 ARM；训练时延比 BD3LM 降低 3×，推理吞吐提升 1.7–4×，生成 PPL 仅略高于 ARM，保持高质量。

**⚠️ 局限性**

局限性包括：1) 对极长序列仍需设定块大小，无法完全无缝扩展；2) 置信度阈值和块尺寸需要手工调参，影响推理稳定性；3) 在数据极少的场景下，传统 ARM 仍能先行；4) 虽有重加权抑制梯度噪声，但对极端高噪声早期 token 的恢复仍有限。

---

## 593. Vidmento: Creating Video Stories Through Context-Aware Expansion With Generative Video

**arXiv ID:** 2601.22013 | [PDF](https://arxiv.org/pdf/2601.22013v1)

**作者:** Catherine Yeh `[一作]` (Harvard University), Bryan Wang `[通讯]` (Adobe Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一款名为Vidmento的工具，通过上下文感知的生成视频扩展将捕获的素材与AI生成的视频融合，支持从构思到剪辑的完整视频故事创作。

**💡 创新点**

提出了“生成性扩展（Generative Expansion）”框架，将生成视频与真实素材在叙事、视觉和风格上做动态匹配，并将脚本编辑与画布式故事线结合，提供半结构化画布、语义放大、生成建议与精细化控制。

**🔧 技术方法**

结合大型语言模型（Google Gemini）、图像生成模型（Gemini Nano Banana）和视频生成模型（Veo3），实现双阶段关键帧生成与动画；同时使用语义分析、脚本提示、注释式提示以及画布与时间轴同步的前端技术。

**📊 数据集**

主要使用参与者自带的原始影像与脚本进行评估；未公开使用公开数据集，而是依赖12位创作者的个人素材（约10–77条）来测试系统功能。

**📈 对比分析**

未做传统算法对比，只进行用户体验评估。通过12位创作者的实验，结果显示工具能让创作者生成平均约31幅图像、8段视频，创意扩展度提升近100%，并获得高满意度；缺乏客观性能指标。

**⚠️ 局限性**

局限包括生成内容与原始素材在风格、色彩、运动上的不匹配；模型的细节控制和快速迭代仍困难；用户对真实性与创作归属的顾虑；生成模型偶尔产生偏离故事的结果；系统整体仍需进一步提升人机协同与细粒度调节。

---

## 594. Heterogeneous Computing: The Key to Powering the Future of AI Agent Inference

**arXiv ID:** 2601.22001 | [PDF](https://arxiv.org/pdf/2601.22001v1)

**作者:** Yiren Zhao `[一作]`, Junyi Liu `[通讯]` (Microsoft Research)

**通讯引用:** 980 | [OpenAlex ID](https://openalex.org/A5100738705)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出了Operational Intensity（OI）与Capacity Footprint（CF）两项指标，系统性分析AI代理推理工作负载的计算与内存瓶颈，并提出基于预填充与解码分离、异构计算与内存分布的设计思路。

**💡 创新点**

创新点在于将OI和CF与传统屋顶线模型结合，首次揭示内存容量瓶颈对长上下文推理的影响，并提出针对不同代理工作流的异构、分离架构与工作负载共设计策略。

**🔧 技术方法**

利用大模型（如LLaMA3‑70B）的推理实验，比较多种注意力机制（MHA、GQA、MLA）、稀疏MoE、量化和KV缓存压缩等技术对OI与CF的影响，并结合系统级并行与网络调度分析。

**📊 数据集**

未在论文中给出具体公开数据集，主要使用LLaMA3‑70B等大模型的推理指标作为实验基准。

**📈 对比分析**

通过绘制OI‑CF二维图，对比不同模型、优化方案以及批量大小、序列长度等场景，在预填充与解码阶段分别评估容量与带宽利用率，发现分离加速器能够在保持低OI的同时缓解内存容量压力，实现性能与效率的平衡。

**⚠️ 局限性**

主要限制包括缺乏大规模真实数据中心验证、未深入讨论网络带宽与光学IO实现细节，以及对训练与推理协同的具体机制仍未展开。

---

## 595. Elign: Equivariant Diffusion Model Alignment from Foundational Machine Learning Force Fields

**arXiv ID:** 2601.21985 | [PDF](https://arxiv.org/pdf/2601.21985v1)

**作者:** Yunyang Li `[一作]` (Yale University), Mark Gerstein `[通讯]` (Yale University)

**通讯引用:** 270443 | [OpenAlex ID](https://openalex.org/A5042321575)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对 E(3)-equivariant 扩散模型进行后训练，使其在保持推理速度不变的前提下，更好地遵循物理能量和力的约束，生成热力学稳定且力学平衡的三维分子结构。

**💡 创新点**

创新点包括：①将基础机器学习力场（MLFF）作为偏好模型，将能量与力奖励解耦并融合到强化学习中；②提出 Force–Energy Disentangled Group Relative Policy Optimization (FED‑GRPO) 的后训练策略；③利用能量奖励塑造（PBRS）提供稠密的中间奖励，缓解稀疏奖励带来的学习难题。

**🔧 技术方法**

使用技术包括 E(3)-equivariant 扩散模型、预训练的基础 MLFF（UMA）、强化学习中的无价值函数 GRPO 与奖励塑造、共享前缀分组采样以及能量/力解耦的奖励设计。

**📊 数据集**

主要使用 QM9 和 GEOM‑Drugs 两个公开分子集合；MLFF 则在 OMol25、OC20、ODAC23、OMat24 等大规模量子力学数据集上预训练。

**📈 对比分析**

与多种基准（如 EDM、GeoLDM、RLPF、GeoBFN 等）比较，Elign 在 QM9 上实现了 93.70% 的分子稳定性、98.32% 的有效性和 95.31% 的有效性×多样性（V×U）最高分，并在 GEOM‑Drugs 上显著提升了原子稳定性（87.94%）和有效性（99.40%），同时保持与无引导模型相同的推理速度。

**⚠️ 局限性**

局限性包括：①对基础 MLFF 的精度高度依赖，误差会被放大（β_eff×误差）；②单目标（仅能量或仅力）训练效果不佳；③对极大分子或非常高温/高能量状态的适应性仍有限；④需要先有高质量的预训练扩散模型作为基线。

---

## 596. MoE-ACT: Improving Surgical Imitation Learning Policies through Supervised Mixture-of-Experts

**arXiv ID:** 2601.21971 | [PDF](https://arxiv.org/pdf/2601.21971v1)

**作者:** Lorenzo Mazza `[一作]` (Technical University of Dresden), Stefanie Speidel `[通讯]` (Technical University of Dresden)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于监督Mixture-of-Experts（MoE）的轻量级动作变压器（ACT），仅利用单镜头双目视角数据，实现少量演示下的多步腹腔镜肠道抓握与拉伸任务。

**💡 创新点**

创新点在于：①通过给MoE专家设置阶段监督，显式引导专家分工并稳定训练；②在仅端到端视角的受限数据环境下，显著提升学习效率；③实现视角不变性与零样本向真实猪肠道的迁移。

**🔧 技术方法**

技术手段包括：监督MoE结构（含动作专家、夹持专家及门控分类器）嵌入ACT；变分推断训练（ELBO）与阶段交叉熵正则；使用双目相机图像作为唯一观测；在OpenHELP人体模型上搭建UR5e+TIPCAM实验平台。

**📊 数据集**

数据集：①固定视角仿真数据120条演示；②随机视角仿真数据50条；③0-样本真实猪肠道实验15条；实验中使用的演示均包含左/右图像、夹持状态与工具末端位置。

**📈 对比分析**

与基线比较：VLA（SmolVLA、π_0.5）在分布内全程成功率为0%；ACT基线分布内成功率为50%；加入MoE后分布内全程成功率提升至85%，OOV情形也显著提升；零样本猪肠道成功率为80%；随机视角训练后在未见视角测试成功率达82%。

**⚠️ 局限性**

局限性包括：需要人工标注阶段标签以监督门控；缺乏深度信息导致抓握深度不足；实验仅在仿真与离体猪体中验证，未完成完整临床环境下的安全评估。

---

## 597. Learning Decentralized LLM Collaboration with Multi-Agent Actor Critic

**arXiv ID:** 2601.21972 | [PDF](https://arxiv.org/pdf/2601.21972v1)

**作者:** Shuo Liu `[一作]` (Northeastern University), Christopher Amato `[通讯]` (Northeastern University)

**通讯引用:** 5719 | [OpenAlex ID](https://openalex.org/A5033129735)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多智能体强化学习中的去中心化大型语言模型协作，并提出了基于集中式与分布式评论器的两种actor‑critic方法。

**💡 创新点**

创新在于将集中式评论器与分布式评论器同时引入LLM协作框架，解决了长时程稀疏奖励下的收敛问题。

**🔧 技术方法**

采用多智能体actor‑critic（MAAC）框架，分别实现CoLLM‑CC（集中式评论器）与CoLLM‑DC（分布式评论器），并使用教师强制推断序列概率。

**📊 数据集**

在写作（TLDR、ArXiv扩写）、编码（CoopHE）和游戏（Minecraft StrBuild、HouseBuild）等三大任务集上评测，利用对应的任务数据集与奖励模型。

**📈 对比分析**

与单模型、提示式多智能体以及MAGRPO等基线相比，CoLLM‑CC在大多数任务中获得最优或相近回报，收敛更快、方差更低；CoLLM‑DC在短时程密集奖励任务表现可比，但在长时程稀疏奖励任务中收敛不稳定。

**⚠️ 局限性**

CoLLM‑DC对局部信息依赖导致非平稳性，CoLLM‑CC仍需中心化训练阶段，且两方法均面临高昂的LLM推理成本，且对极长时程任务的样本需求仍较高。

---

## 598. Geometry of Drifting MDPs with Path-Integral Stability Certificates

**arXiv ID:** 2601.21991 | [PDF](https://arxiv.org/pdf/2601.21991v1)

**作者:** Zuyuan Zhang `[一作]` (George Washington University), Tian Lan `[通讯]` (George Washington University)

**通讯引用:** 6376 | [OpenAlex ID](https://openalex.org/A5018464968)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了对漂移MDP的几何路径分析，给出长度、曲率、尖点度量，并基于这些量设计HT‑RL和HT‑MCTS自适应调度器，实现在线跟踪。

**💡 创新点**

首次将MDP漂移建模为可微同伦路径，将最优Bellman固定点的运动分解为长度、曲率和尖点三项，得到solver‑agnostic的稳定性界限及gap‑safe区域；基于此提出可观测代理的轻量化自适应调度。

**🔧 技术方法**

采用几何路径积分、Wasserstein‑1测度、隐函数定理、动态调度器、EMA平滑、双Q/目标网络、MCTS搜索深度与预算自适应等技术。

**📊 数据集**

使用合成环形MDP路径以及四个控制基准（LunarLander、Acrobot、PointMass、Pendulum）及其噪声漂移版本作为实验数据集。

**📈 对比分析**

通过与静态RL基线（DQN、SAC、Double）以及静态HT包装器在AUC@Steps、50%/75%返回和最终回报上的比较，HT‑RL在噪声漂移、曲率和尖点驱动的环境中实现了显著提升，AUC提升可达数十个百分点，最终回报提升超过100%。

**⚠️ 局限性**

假设同伦路径可微且奖励/转移可线性插值；需要估计Wasserstein项和动作间距，对高维连续状态的代理精度有限；不适用于突变跳跃大幅改变动作空间或目标的情形。

---

## 599. Per-parameter Task Arithmetic for Unlearning in Large Language Models

**arXiv ID:** 2601.22030 | [PDF](https://arxiv.org/pdf/2601.22030v1)

**作者:** Chengyi Cai `[一作]` (University of Melbourne), Jun Zhou `[通讯]` (Ant Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于参数级任务算术的 LLM 失忆方法 PerTA，利用梯度或 Fisher 信息为每个参数分配权重来重新缩放任务向量，降低对保留信息的过度遗忘；

**💡 创新点**

创新点在于将任务向量的权重从单一全局系数提升到参数级，可根据忘记集与保留集梯度大小自适应调节，既保持失忆效果又保留模型实用性；

**🔧 技术方法**

技术包括：任务算术、绝对梯度和对角 Fisher 信息近似、参数级权重计算、一次性梯度评估、微调模型求得忘记向量；

**📊 数据集**

实验使用公开失忆基准 TOFU（覆盖 1%、5%、10% 失忆比例）和 MUSE News 数据集，并在 Llama‑3.2 1B/3B Instruct 等 LLM 上验证；

**📈 对比分析**

与传统的任务算术（TV）和多种训练基方法（GA、GD、NPO、NPO+）对比，PerTA 在忘记质量（FQ）和模型实用性（MU）上均优于 TV，并在多数指标上逼近仅使用保留集微调的“ground‑truth”模型；

**⚠️ 局限性**

局限性包括对梯度估计的依赖（需一次额外梯度计算）、对超参数（如 τ 或权重阈值）敏感、在极大比例失忆或高度相关的忘记/保留信息场景下可能仍存在不完美平衡；

---

## 600. Hybrid Foveated Path Tracing with Peripheral Gaussians for Immersive Anatomy

**arXiv ID:** 2601.22026 | [PDF](https://arxiv.org/pdf/2601.22026v1)

**作者:** Constantin Kleinbeck `[一作]` (Technical University of Munich), Daniel Roth `[通讯]` (Technical University of Munich)

**通讯引用:** 3459 | [OpenAlex ID](https://openalex.org/A5028687545)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本研究提出一种混合渲染系统，结合视差渲染路径追踪（foveated path tracing）与高效的高斯溅射（Gaussian Splatting）外周模型，实现医学体积在 VR 中的实时高质量可视化。

**💡 创新点**

创新点在于：①将实时路径追踪限制在视线中心区以节省计算；②使用深度引导的重投影技术抑制路径追踪延迟；③实现外周模型秒级快速重建，并可在交互过程中持续改进；④通过深度与稀疏视角训练快速生成高斯云。

**🔧 技术方法**

采用的技术包括：卷积神经网络（CNN）路径追踪、OptiX 深度噪声抑制器、Mini-Splatting2 的快速高斯优化、深度引导的重投影、Unity VR 实时渲染插件、TCP+MessagePack 通信。

**📊 数据集**

使用的医学数据集为：Total Segmentator 体积（Fullbody）和 TCGA-HNSC 体积（Leg），共六个转移函数组合。

**📈 对比分析**

与单独的路径追踪（含视差渲染）以及预计算的 GS 进行对比；在相同帧率下，混合方法在视线中心区的 LPIPS 与 MPSNR 均优于两者；外周质量与 GS 相近但建立时间从 1–2 分钟缩短至 <1 秒，整体帧率可达 72 FPS（Meta Quest 3）。

**⚠️ 局限性**

主要局限：外周模型在边界处可能出现颜色/亮度差异；快速重建可能欠采样细节，导致外周暗淡；重投影在快速头部运动时可能产生失真；系统尚未完成端到端用户体验评估；对大场景扩展的适应性有限。

---

## 601. When "Better" Prompts Hurt: Evaluation-Driven Iteration for LLM Applications

**arXiv ID:** 2601.22025 | [PDF](https://arxiv.org/pdf/2601.22025v1)

**作者:** Daniel Commey `[一作]` `[通讯]`, Daniel Commey

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于评估驱动的循环流程（Define‑Test‑Diagnose‑Fix）和MVES多层评估框架，并在本地环境中实现了可复现的评估工具；

**💡 创新点**

创新点包括：①将应用层评估标准细分为MVES-Core、MVES‑RAG、MVES‑Agentic三类；②系统展示了通用“改进”提示可能导致结构化任务性能下降的现象；③通过四条件消融实验揭示了泛化规则对任务特定提示的冲突机制；

**🔧 技术方法**

采用了自动化检查、LLM‑as‑judge、BERTScore、ROUGE、RAGAS、以及人类评分等多种评估技术，并在Ollama本地推理平台上实现了无API的评估流水线；

**📊 数据集**

使用了约50个手工构造的金标准测试集，涵盖提取（20）、RAG（15）与指令遵循（15）三类场景，并在Llama 3 8B与Qwen 2.5 7B模型上进行实验；

**📈 对比分析**

通过对比基线任务特定提示与通用改进提示的通过率，发现Llama 3在提取任务从100%降至90%，RAG任务从93.3%降至80%，指令任务从53.3%升至66.7%；四条件消融进一步证实规则追加导致性能下降，系统包装本身无影响；

**⚠️ 局限性**

局限性包括：测试集规模有限（仅15–20条），仅覆盖两款7–8B模型，实验采用温度为0的确定性推理，缺乏对高风险领域和多模态、跨语言场景的验证；

---

## 602. PocketDP3: Efficient Pocket-Scale 3D Visuomotor Policy

**arXiv ID:** 2601.22018 | [PDF](https://arxiv.org/pdf/2601.22018v1)

**作者:** Jinhao Zhang `[一作]`, Jie Me `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 PocketDP3，一种基于 3D 点云的压缩型扩散策略，用轻量化的 Diffusion Mixer (DiM) 替代传统大型 U‑Net 解码器，实现高效的视觉运动控制。

**💡 创新点**

创新点包括：① 发现 3D 扩散策略中解码器往往过大，提出最小可行解码器设计；② 用 MLP‑Mixer 架构的 DiM 进行跨时间/通道融合；③ 通过 DDIM 采样实现两步推理，无需一致性蒸馏，显著提升推理速度与参数效率。

**🔧 技术方法**

使用技术包括：3D 扩散模型（DDIM 采样）、MLP‑Mixer‑based Diffusion Mixer、FiLM 条件注入、点云编码器（DP3 轻量化）、两步推理、无一致性训练。

**📊 数据集**

使用数据集：RoboTwin2.0、Adroit、MetaWorld 三大仿真基准；以及真实机器人（AgileX Piper + Intel RealSense D455）上的 50 条手动演示。

**📈 对比分析**

与 DP3、DP、FlowPolicy 等方法对比，PocketDP3 在三大基准上均取得 SOTA 成功率（平均提升 15–20%），参数量仅为 DP3 的 1% 以内，推理延迟仅 4–5 ms（2 步），显著优于传统解码器方案。

**⚠️ 局限性**

局限性包括：① 当编码器输出更高维或多模态信息时，DiM 需要增大容量；② 缺乏 U‑Net 的空间偏置，可能影响极精细插入等高精度任务；③ 对两步推理效果的机理尚未完全解释。

---

## 603. VERSA: Verified Event Data Format for Reliable Soccer Analytics

**arXiv ID:** 2601.21981 | [PDF](https://arxiv.org/pdf/2601.21981v1)

**作者:** Geonhee Jo `[一作]` (University of Seoul), Sang-Ki Ko `[通讯]` (University of Seoul)

**通讯引用:** 768 | [OpenAlex ID](https://openalex.org/A5011019318)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于状态转移模型的足球事件流验证框架（VERSA），用于自动检测并纠正事件流中的逻辑错误，提升数据可靠性；

**💡 创新点**

创新点在于将足球比赛的事件序列抽象为有限状态机，通过显式状态与事件标签、条件约束共同定义合法转移，并配套异常处理器自动生成或重排缺失/错误事件；

**🔧 技术方法**

核心技术包括：状态机建模（使用Transitions库实现）、规则驱动的异常检测与纠正、数据预处理统一格式、统计与评价指标（编辑相似度、Pearson相关、AUROC、Log Loss、Brier Score、ECE）；

**📊 数据集**

使用来自三大数据提供商（K League、J League、Wyscout）的赛季事件数据（2021-2025 K League 1/2，2017/2018 La Liga，2018世界杯等）以及公开的 J1、La Liga、世界杯等数据集；

**📈 对比分析**

通过比较不同数据格式（原始、统一、校正后）的跨提供商一致性（编辑相似度和玩家评价相关性）和下游预测任务（使用CatBoost预测进球/失球）的性能，结果显示校正后格式在所有指标上均优于基线，尤其是AUROC和ECE；

**⚠️ 局限性**

局限性包括：仅针对已定义的状态和事件类型，可能无法覆盖所有边缘或新兴动作；异常处理器的规则是手工制定，需人工维护；模型对大规模实时流的性能与扩展性尚未充分验证。

---

## 604. From Particles to Agents: Hallucination as a Metric for Cognitive Friction in Spatial Simulation

**arXiv ID:** 2601.21977 | [PDF](https://arxiv.org/pdf/2601.21977v1)

**作者:** Javier Argota Sánchez-Vaquerizo `[一作]` (ETH Zürich), Luis Borunda Monsivais `[通讯]` (Virginia Tech)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5043196024)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于大型多模态生成模型的代理环境模拟框架，将建筑空间建模为认知代理，并通过幻觉-现实差距衡量认知摩擦。

**💡 创新点**

将传统时间步物理模拟转为事件驱动的情节化空间推理，并将AI幻觉视为半语义歧义的诊断工具，定义认知摩擦指标。

**🔧 技术方法**

结合多模态大语言模型（LLM）与视觉语言模型（VLM），使用余弦相似度在共享嵌入空间评估幻觉与物理真值的差距。

**📊 数据集**

未公开具体数据集，主要使用建筑内部视觉与文本描述的多模态数据，依赖已有建筑信息模型和用户行为日志。

**📈 对比分析**

与传统基于物理的粒子模拟相比，该方法在预测半语义歧义和情绪负荷方面更具解释性，但缺乏量化的性能指标，实验为概念验证。

**⚠️ 局限性**

依赖西方建筑符号、模型成本高、缺乏跨文化验证，且幻觉指标易受模型演进影响，需要伦理治理与参与式评估。

---

## 605. Macro-Scale Electrostatic Origami Motor

**arXiv ID:** 2601.21976 | [PDF](https://arxiv.org/pdf/2601.21976v1)

**作者:** Alex S. Miller `[一作]` (Massachusetts Institute of Technology), Jeffrey H. Lang `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 9963 | [OpenAlex ID](https://openalex.org/A5022916324)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

设计并实现了第一台宏观尺度可折叠的原型折纸电动机，能够从紧凑折叠状态展开为可旋转的工作状态。

**💡 创新点**

创新点在于：①使用Kresling折纸结构实现可折叠圆柱形电机；②采用柔性PCB电极与高压电晕放电产生连续旋转；③实现了可折叠、可重复展开且无明显损伤的折纸电机。

**🔧 技术方法**

技术手段包括：折纸几何建模、JSON‑PCB柔性电路板设计、Sapphire珠宝轴承与高压直流供电、电晕放电理论建模、光学高速摄像测量。

**📊 数据集**

该研究未使用公开数据集，所有实验数据均由自制设备采集。

**📈 对比分析**

性能评估：在‑29 kV电压下最高转速1440 rpm，最大输出扭矩0.15 mN·m，活跃电路板扭矩密度0.04 N·m kg⁻¹；在展开状态下体积扭矩密度0.54 N·m m⁻³，折叠状态为1.35 N·m m⁻³；与现有折纸机械或线性驱动器相比，提供了更高的体积/质量比例和连续旋转能力。

**⚠️ 局限性**

局限性包括：需要高压直流电源导致系统尺寸与安全风险；扭矩和转速受电晕放电阈值与气体导电性限制；折纸结构在多次折叠后易出现电极开裂；电机尺寸与重量仍高于同等性能的传统旋转电机。

---

## 606. Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding

**arXiv ID:** 2601.21969 | [PDF](https://arxiv.org/pdf/2601.21969v1)

**作者:** Yifan Zhu `[一作]` (Beijing University of Posts and Telecommunications), Haoran Luo `[通讯]` (Nanyang Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Token-Guard框架，利用token级自检与多阶段过滤来控制LLM生成的虚假信息。

**💡 创新点**

创新点在于结合token级概率与语义一致性评分的自检机制、段级风险评估与局部再生以及全局迭代修正，形成轻量级的hallucination控制流程。

**🔧 技术方法**

使用自注意力隐藏状态、混合分数计算、softmax加权、局部窗口再生成、聚类+KMeans构造链路、soft最小化评分等技术。

**📊 数据集**

在HALU基准数据集上测试，包括FinanceBench、DROP、COVID‑QA、PubMedQA、HaluEval和RAGTruth。

**📈 对比分析**

与基线（Base、Chain‑of‑Thought、Tree‑of‑Thought、Guided、Predictive）对比，Token‑Guard在EM/F1上平均提升约10‑15%，在部分数据集提升高达16%或更多。

**⚠️ 局限性**

仍受限于缺乏外部知识检索、对极长上下文的适应性不足，以及在知识稀缺场景下提升有限。

---

## 607. Industrialized Deception: The Collateral Effects of LLM-Generated Misinformation on Digital Ecosystems

**arXiv ID:** 2601.21963 | [PDF](https://arxiv.org/pdf/2601.21963v1)

**作者:** Alexander Loth `[一作]` (Microsoft), Marc-Oliver Pahl `[通讯]` (IMT Atlantique)

**通讯引用:** 882 | [OpenAlex ID](https://openalex.org/A5004198506)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文综述了2024年以来生成式人工智能（GenAI）在信息误导方面的演进，并提出并实现了两个开源工具（JudgeGPT 与 RogueGPT）构成的实验管线，用于评估人类对AI生成新闻的感知与检测效果；同时评估了多种对策（检测、预警、加密来源等）在防御生成式假新闻中的作用。

**💡 创新点**

①将生成与检测双向技术整合成闭环实验体系，能够在可控条件下生成并追踪假新闻的生成参数；②首次在大规模文本生成场景下系统量化人类感知-准确率差距；③以多模态交叉一致性为核心，提出新型检索与对抗检测策略；④将C2PA、SynthID等真实性基础设施纳入评估框架。

**🔧 技术方法**

使用大型语言模型（GPT‑4/Claude3.5/ Gemini1.5）、多模态模型（LLM‑Vision、Diffusion、GAN）、Transformer 架构；检测侧采用 LLM‑based 逻辑推理、情感无关特征、跨模态图注意网络；实验平台实现基于 MongoDB 的元数据跟踪；预警模块采用 Inoculation/Prebunking 机制。

**📊 数据集**

生成的新闻片段由 RogueGPT 依据配置参数（模型、温度、风格、格式）在多语言（英语、德语、法语）下产生；评估数据来自 JudgeGPT 采集的人类参与者回答（包括信任度、真实性、来源判别）以及公开的假新闻基准（e.g., LIAR, FEVER）和多模态基准（FaceForensics++）。

**📈 对比分析**

与传统规则基准与现有 LLM‑detector（如 OpenAI Detection、TRUST‑VL）比较，发现 GPT‑4 生成文本的检测准确率在 58%–62% 之间，接近随机猜测；情感无关训练提升了 8–12% 的 F1；跨模态一致性检测在 98.7% 的准确率上超越单模态方法；在预警实验中，Prebunking 使误判率下降 18%（p<0.05）。

**⚠️ 局限性**

（1）实验样本规模有限，难以覆盖所有语言与媒体形式；（2）检测方法仍受生成模型进化的冲击，缺乏长期鲁棒性；（3）真实性基础设施（C2PA、SynthID）对隐私的潜在风险未完全解决；（4）缺乏大规模真实世界部署验证，结果可能与实验环境存在差异；（5）对抗攻击（如情感攻击）仍可能在实际中削弱检测性能。

---

## 608. Secure Group Key Agreement on Cyber-Physical System Buses

**arXiv ID:** 2601.21966 | [PDF](https://arxiv.org/pdf/2601.21966v1)

**作者:** Sebastian N. Peters `[一作]` (Technical University of Munich), Jason Lochert `[通讯]` (Technical University of Munich)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计并实现了一种针对 CPS 总线环境的分布式、身份认证的组密钥协议 gracybus，以满足 12 项安全与性能需求；

**💡 创新点**

将 TreeKEM 迁移至无中心、广播/半双工总线，结合树形结构与 epoch key 调度，支持动态加入/离开、Merge/Split，并兼顾 PCS、FS 与后量子安全；

**🔧 技术方法**

基于 TreeKEM/MLS 的树形密钥分发、PKI 证书签名、KDF、MAC 与加密/KEM，采用最小异步广播、分段与子树加密，兼容可定制的对称与后量子算法；

**📊 数据集**

无外部数据集，性能评估基于实验/仿真模拟环境（未使用特定工业总线数据集）；

**📈 对比分析**

与 TGDH、D‑OFT、SGRS、GDH.2/3 等协议对比，计算与消息开销均为 O(log n)，支持 100+ 设备，内存占用 O(log n)，消息量约 O(log n)，在资源受限硬件上实现可行；

**⚠️ 局限性**

尚未实现 Merge/Split；对总线分区攻击仍无完整可用性保障；需要额外内存存储 O(n) 公钥；需在真实 CPS 部署中进一步验证与优化树重排逻辑。

---

## 609. SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks

**arXiv ID:** 2601.22024 | [PDF](https://arxiv.org/pdf/2601.22024v1)

**作者:** Abhishek Duttagupta `[一作]` (IMDEA Networks Institute), Joerg Widmer `[通讯]` (IMDEA Networks Institute)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

研究提出一种基于一阶逻辑的符号可解释深度强化学习框架，能够为移动网络中的DRL决策生成人类可读解释并实现意图驱动的行动调度。

**💡 创新点**

通过将状态与动作映射为符号化的谓词与四分位概念，利用逻辑推理生成可解释知识图，进而实现基于符号的行动调度和决策约束，显著提升解释性和性能。

**🔧 技术方法**

结合深度强化学习（SAC、DQN）、符号AI（First-Order Logic）、逻辑推理、知识图（KG）以及意图驱动行动调度（IAS）模块。

**📊 数据集**

采用Colosseum仿真器中的TRF1/TRF2网络流量配置，以及Indoor Mobility Channel Measurements 数据集，用于大规模MIMO调度实验。

**📈 对比分析**

与传统XRL方法METIS及无IAS基线进行对比，使用累计奖励、行动分布及学习效率评估；结果显示IAS在累计奖励上平均提升约12%（11.8%），显著优于METIS，并能在训练更少轮次时实现同等性能。

**⚠️ 局限性**

需要手工定义符号谓词与分位阈值，离散化可能导致信息损失；对极端或未见状态的推理受限；对高维连续动作空间的扩展仍需进一步研究。

---

## 610. LANCER: LLM Reranking for Nugget Coverage

**arXiv ID:** 2601.22008 | [PDF](https://arxiv.org/pdf/2601.22008v1)

**作者:** Jia-Huei Ju `[一作]` (University of Amsterdam), Andrew Yates `[通讯]` (Johns Hopkins University)

**通讯引用:** 3078 | [OpenAlex ID](https://openalex.org/A5059489981)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了基于LLM的重排器LANCER，利用子问题生成与答案可回答性评分来提升长文检索中的信息片段（nugget）覆盖率。

**💡 创新点**

通过将子问题视为代理nugget并采用多维评分与覆盖聚合策略，首次将信息覆盖度作为重排目标，克服传统仅优化相关度的局限。

**🔧 技术方法**

大语言模型生成子问题与评分，覆盖聚合（求和、RRF、贪心选择），并基于CRUX框架进行评估。

**📊 数据集**

NeuCLIR'24 ReportGen 与 CRUX‑MDS‑DUC'04 两个长文检索评测集。

**📈 对比分析**

与Pointwise、Listwise、Setwise等LLM重排基线比较，在α‑nDCG@10 与 Coverage@10 指标上均取得显著提升；使用真实nugget作为子问题的oracle设置更进一步提升性能。

**⚠️ 局限性**

子问题生成噪声与LLM评分的不确定性导致覆盖提升受限；现有评测集nugget数量有限，影响评估的充分性。

---

## 611. Mechanistic Data Attribution: Tracing the Training Origins of Interpretable LLM Units

**arXiv ID:** 2601.21996 | [PDF](https://arxiv.org/pdf/2601.21996v1)

**作者:** Jianhui Chen `[一作]` (Peking University), Liangming Pan `[通讯]` (Peking University)

**通讯引用:** 994 | [OpenAlex ID](https://openalex.org/A5027533517)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了机制数据归因（MDA）框架，利用影响函数追踪可解释LLM单元的训练数据来源，并通过数据删减/增补实现对模型机制的因果干预；

**💡 创新点**

创新点在于：①将影响函数从全局模型降到单元级；②对诱导头与上下文学习的因果关系给出实验验证；③构建可推广到不同规模模型的机制数据增强流程；

**🔧 技术方法**

使用影响函数（带EK‑FAC近似）、梯度反向传播、对注意力头/神经元的功能探测器、对抗性重新训练与LLM驱动的结构提取与合成脚本；

**📊 数据集**

主要使用Pythia系列模型训练语料（开放网页文本、维基百科等无结构大语料），以及从中抽取的高影响样本和合成的结构化数据；

**📈 对比分析**

与随机删减/增补对比，MDA指导的干预能显著抑制或加速诱导头出现，提升10–15%；合成数据在各规模模型上统一加速诱导头形成；通过诱导头分数与ICL分数的同步变化验证因果关系；

**⚠️ 局限性**

局限性包括：①只针对已可解释的单元（如诱导头）；②影响函数计算受限于训练窗口；③对全局模型行为的解释有限；④合成数据若规模过大可能导致多样性不足；⑤潜在被误用进行数据污染。

---

## 612. Generalized Information Gathering Under Dynamics Uncertainty

**arXiv ID:** 2601.21988 | [PDF](https://arxiv.org/pdf/2601.21988v1)

**作者:** Fernando Palafox `[一作]` (University of Texas at Austin), David Fridovich-Keil `[通讯]` (University of Texas at Austin)

**通讯引用:** 571 | [OpenAlex ID](https://openalex.org/A5070827615)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一个统一的模块化框架，用于在未知动力学系统下的主动信息收集与决策；

**💡 创新点**

将信息收集成本与动力学模型、贝叶斯更新、观察模型等分离，构造基于有向信息的通用成本，并证明其在若干假设下等价于传统的互信息成本，提供理论与实验双重验证；

**🔧 技术方法**

利用贝叶斯滤波（如 EKF）、有向信息理论、蒙特卡洛采样、交叉熵规划以及可微模型预测控制实现框架；

**📊 数据集**

使用合成实验数据，涵盖单体线性/非线性系统与双体追逐/躲避情境，无公开真实数据集；

**📈 对比分析**

与随机动作和被动学习基线对比，实验表明信息收集权重提升后参数估计误差显著下降，模型对未见数据的泛化性能提升；

**⚠️ 局限性**

局限性包括需要满足马尔可夫加噪声结构、对线性化和高斯近似敏感、且在高度非线性或非加噪环境下表现尚待验证。

---

## 613. SpecTran: Spectral-Aware Transformer-based Adapter for LLM-Enhanced Sequential Recommendation

**arXiv ID:** 2601.21986 | [PDF](https://arxiv.org/pdf/2601.21986v1)

**作者:** Yu Cui `[一作]` (Zhejiang University), Jiawei Chen `[通讯]` (Zhejiang University)

**通讯引用:** 4275 | [OpenAlex ID](https://openalex.org/A5100362810)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种基于Transformer的光谱感知适配器SpecTran，用来将高维LLM文本嵌入投影到低维物品嵌入空间，从而增强顺序推荐模型。

**💡 创新点**

提出在谱域中使用可学习的Transformer注意力，并加入基于泰勒展开的光谱位置编码来引入奇异值信息，既消除维度坍塌，又充分利用主副谱信息，解决了现有适配器与SVD方法的缺陷。

**🔧 技术方法**

利用SVD分解、Transformer注意力、稀疏Softshrink激活、泰勒展开位置编码，以及标准的序列模型（BERT4Rec、SASRec、HSTU）结合LLM文本编码。

**📊 数据集**

在Amazon Toys & Games、Amazon Beauty、Amazon Clothing, Shoes and Jewelry、Amazon Office Products四个真实数据集上进行实验。

**📈 对比分析**

与多种基线（Adapter类：MoRec、UniSRec、RLMRec、LLM-ESR；SVD类：WhitenRec、LLMInit、AlphaFuse）以及不同的序列推荐骨干进行对比，平均提升约9.17%，在所有指标（HR@10/20、NDCG@10/20）上均优于对手。

**⚠️ 局限性**

仍未探索更复杂的谱注意力机制、LLM直接推荐框架以及对极高维文本嵌入的进一步压缩，且实验仅关注物品文本，未涉及多模态或用户侧文本。

---

## 614. Investigation into using stochastic embedding representations for evaluating the trustworthiness of the Fréchet Inception Distance

**arXiv ID:** 2601.21979 | [PDF](https://arxiv.org/pdf/2601.21979v1)

**作者:** Ciaran Bench `[一作]` (National Physical Laboratory), Spencer A. Thomas `[通讯]` (National Physical Laboratory)

**通讯引用:** 1367 | [OpenAlex ID](https://openalex.org/A5025580494)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文使用蒙特卡洛 Dropout 计算特征嵌入模型的预测方差，评估 Fréchet Inception Distance（FID）在医学影像等 OOD 数据上的可信度。

**💡 创新点**

创新点在于将 UQ（MCD）与 FID 结合，提出使用预测方差和 FID 方差作为衡量 FID 可信度的启发式指标，并在不同数据增强与 OOD 数据上验证其相关性。

**🔧 技术方法**

采用 InceptionV3 预训练模型与蒙特卡洛 Dropout、FID 计算、k‑NN、MAE、MS‑SSIM 等技术。

**📊 数据集**

使用 ImageNet1K（训练/验证集）、轻度增强的 ImageNet1K、CelebA、乳腺 X 光图像等多种数据集。

**📈 对比分析**

通过比较不同数据集下 FID、σFID 与 pVar 的变化，以及与 top‑5 准确率、k‑NN 距离、MAE/SSIM 的相关性，发现 σFID 与 OOD 越大相关，pVar 关联性弱，指标在等效增强实验中表现更稳定。

**⚠️ 局限性**

局限在于缺乏客观的 FID 效果基准，难以定量验证 σFID/pVar 的可靠性，且仅在有限的数据增强与 OOD 场景验证，未考虑更复杂的医学影像特征。

---

## 615. Mind the Gap: How Elicitation Protocols Shape the Stated-Revealed Preference Gap in Language Models

**arXiv ID:** 2601.21975 | [PDF](https://arxiv.org/pdf/2601.21975v1)

**作者:** Pranav Mahajan `[一作]` (University of Oxford), Lydia Nottingham `[通讯]` (University of Oxford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估不同偏好引导协议对语言模型的声明-显现偏好（SvR）相关性的影响。

**💡 创新点**

指出SvR相关性高度依赖引导协议，证明允许中立/回避的声明偏好可显著提升相关性，而在显现偏好中使用同样机制会导致相关性下降；并发现系统提示引导在大规模价值集合上效果不稳定。

**🔧 技术方法**

使用扩展的LitmusValues框架、GPT-4o-mini判定模型回复、Spearman相关系数评估。

**📊 数据集**

MindTheGap数据集（含AIRiskDilemmas）以及开源代码仓库。

**📈 对比分析**

与传统强制二选协议比较，扩展声明偏好提升ρ从≈0.2到≈0.7，且与模型能力正相关；扩展两者导致ρ趋近零。性能提升在声明偏好上显著，但在显现偏好上无显著改善。

**⚠️ 局限性**

主要限制是显现偏好中高中立率导致相关性无效，系统提示在大价值集合下不稳定，且仍未解决模型实际价值体系与情境行为不匹配的问题。

---

## 616. Cognitive Load Estimation Using Brain Foundation Models and Interpretability for BCIs

**arXiv ID:** 2601.21965 | [PDF](https://arxiv.org/pdf/2601.21965v1)

**作者:** Deeksha M. Shama `[一作]` (Johns Hopkins University), Ivan J. Tashev `[通讯]` (Microsoft Research)

**通讯引用:** 3364 | [OpenAlex ID](https://openalex.org/A5007425970)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究提出并验证了基于脑基底模型（BFM）的连续认知负荷实时估计流水线，能够跨受试者、跨电极配置进行高精度预测，并通过Partition SHAP实现解释性分析。

**💡 创新点**

创新点包括：①将大规模预训练的BFM（LaBraM、CBraMod）迁移到长时段EEG分析；②设计了分组平均与时间池化的混合策略提升跨受试者泛化；③将Partition SHAP与脑区解剖结构结合，系统性解释模型特征重要性；④在多日VR飞行训练实验中实现了在线认知负荷监测，揭示认知负荷随学习递减与前额叶激活递增的关系。

**🔧 技术方法**

主要技术手段有：EEG预处理（滤波、重采样、窗切分）；BFM特征提取（LaBraM、CBraMod encoder）；空间池化（组均值、交集）与时间池化（全局、均值、均值-标准差）；下游估计器（线性层、DNN、SVM）及LSTM对比；解释性方法Partition SHAP；跨受试者嵌套交叉验证评估。

**📊 数据集**

使用的数据集为5个连续工作日的VR飞行训练实验，包含30名受试者（各日约90个试验），EEG采样500 Hz，后重采样200 Hz。实验涵盖26/28/32通道不同配置，最终采用32通道稳定组（共5名受试者）进行评估。

**📈 对比分析**

与传统PSD频谱特征、EEGNet、EEGConformer等基准相比，LaBraM + 组均值 + 全局池化 + 线性层的Pearson相关系数最高，约为0.63（相较PSD的0.30、EEGNet的0.33），且模型推理时间低于1 s，能实现实时长窗口推断。

**⚠️ 局限性**

局限性包括：①只评估了VR飞行任务，对其他认知负荷场景的泛化尚未验证；②BFM模型规模较大，仍需进一步压缩以适配资源受限设备；③解释性分析依赖Partition SHAP，受特征相关性假设影响；④实验受限于受试者数量与日均样本，未充分覆盖极端负荷条件。

---

## 617. TBDFiltering: Sample-Efficient Tree-Based Data Filtering

**arXiv ID:** 2601.22016 | [PDF](https://arxiv.org/pdf/2601.22016v1)

**作者:** Robert Istvan Busa-Fekete `[一作]` (Google Research), Andras Gyorgy `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于层次聚类的非参数化、样本高效数据过滤方法（Tree‑Based Data Filtering, TBDF），用于大规模LLM训练集的质量筛选。

**💡 创新点**

创新点在于将高维文本嵌入的层次结构与自适应采样结合：算法仅在聚类子树不纯时才查询LLM，理论上查询复杂度与最优树切割的结构复杂度K（而非数据规模n）成正比。

**🔧 技术方法**

核心技术包括：文本嵌入（Gecko 768维）、分布式层次聚类（基于Affinity Propagation / Borůvka's算法）、贝叶斯置信区间估计、LLM（Gemini 2.5 Flash）提示获取质量得分。

**📊 数据集**

实验使用三大公开网络语料：FineWeb、ThePile、C4，随后在Gemma 3（270 M、1 B、4 B参数）模型上进行训练。

**📈 对比分析**

与随机采样基线和基于FineWeb‑Edu的分类器过滤（CB）对比，TBDF在8个下游任务（HellaSwag、WinoGrande、SIQA、PIQA、ARC、Commonsense QA、MMLU、BookQA）上平均提升1–5%相对性能，且在多数任务上超越CB。

**⚠️ 局限性**

局限性包括：需先构建层次聚类（成本较高）、仍需对一定比例文本进行LLM推理、对聚类纯度假设敏感、目前仅验证文本数据，跨模态扩展需进一步研究。

---

## 618. Holographic generative flows with AdS/CFT

**arXiv ID:** 2601.22033 | [PDF](https://arxiv.org/pdf/2601.22033v1)

**作者:** Ehsan Mirafzali `[一作]` (University of California), Razvan Marinescu `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了 GenAdS 框架，将 AdS/CFT 对应的 Klein–Gordon 物理动力学嵌入流匹配生成模型，实现了基于物理先验的流生成。

**💡 创新点**

创新点在于首次将 bulk-to-boundary 传播子和 AdS 语义映射与流匹配相结合，提供了一种可解释且训练更高效的生成模型。

**🔧 技术方法**

使用了流匹配、连续归一化流、Klein–Gordon 方程、AdS/CFT 传播子、傅里叶谱分解、卷积神经网络以及 Hermite 路径等技术。

**📊 数据集**

实验数据集包括 2D 棋盘点云和 MNIST 手写数字图像。

**📈 对比分析**

通过与不使用 AdS 信息的基线模型（FCN/CNN）在 BV、WED、FID 等指标进行对比，结果显示 GenAdS 在边界违规率和训练效率上优于基线，但在 MNIST 上过强的物理约束略逊。

**⚠️ 局限性**

限制在于仅考虑固定 AdS 背景、仅使用标量 Klein–Gordon 方程、未包含重力反作用、holographic 编码对图像过于粗糙，以及仅在平面切片实验，未探究其他切片或更复杂的物理场景。

---

## 619. Cross-Fusion Distance: A Novel Metric for Measuring Fusion and Separability Between Data Groups in Representation Space

**arXiv ID:** 2601.22036 | [PDF](https://arxiv.org/pdf/2601.22036v1)

**作者:** Xiaolong Zhang `[一作]` (Oregon Health and Science University), Xubo Song `[通讯]` (Oregon Health and Science University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并实现了交叉融合距离（CFD），用于衡量表示空间中不同数据组之间的融合与分离程度。

**💡 创新点**

创新点在于利用方差分解将融合相关的几何位移与分散分离，并构造尺度不变、对融合保持不敏感的闭式距离度量。

**🔧 技术方法**

采用方差分解、闭式公式计算、线性时间复杂度实现，并与 Wasserstein、MMD、Hausdorff、Chamfer 等传统分布距离进行对比。

**📊 数据集**

在合成数据以及真实生物医学图像数据集 Camelyon16 和 MIDOG21 上进行实验，特征提取使用 Virchow 模型。

**📈 对比分析**

通过灵敏度、尺度不变性、对形变/离群点鲁棒性以及与交叉域性能下降率（CDDR）的相关性评估，CFD 在所有指标上均优于传统度量，相关性最高。

**⚠️ 局限性**

目前仅针对两组数据进行评估，扩展到多组或层级结构以及时序漂移的行为仍需进一步研究。

---

## 620. CAR-bench: Evaluating the Consistency and Limit-Awareness of LLM Agents under Real-World Uncertainty

**arXiv ID:** 2601.22027 | [PDF](https://arxiv.org/pdf/2601.22027v1)

**作者:** Johannes Kirmayr `[一作]` (BMW Group Research and Technology), Elisabeth André `[通讯]` (Augsburg University)

**通讯引用:** 16656 | [OpenAlex ID](https://openalex.org/A5056684559)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发 CAR-bench 评测汽车车内助手 LLM 代理在多轮对话中使用工具、遵守域政策、处理幻觉与歧义的能力。

**💡 创新点**

创新点在于引入 Hallucination 与 Disambiguation 两种任务，强调一致性、对不确定性的识别与自我意识，并通过 58 个交互式汽车工具与 19 条安全策略构建真实场景。

**🔧 技术方法**

采用 LLM 代理 + 工具调用接口 + 领域政策约束 + LLM 用户模拟 + Pass@k / Passk 一致性评估 + 错误分类等技术。

**📊 数据集**

使用 58 个汽车相关工具、19 条政策、包含导航、充电、气象、日历等数据库的 100 基础、90 幻觉、50 歧义任务集，全部由人工验证。

**📈 对比分析**

与 GPT‑5、Claude‑Opus‑4.5、Gemini‑2.5‑Flash、Qwen3‑32B、xLAM‑2‑32B 等模型进行三轮一致性 Pass3 与 Pass@3 对比，发现即使最强模型在 Disambiguation 任务的 Pass3 也低于 50%，显示可靠性差距。

**⚠️ 局限性**

局限在于依赖 LLM 模拟用户易产生噪声、数据量不足难以大规模微调、未覆盖多模态或多用户情境、且安全责任划分未完全完善。

---

## 621. Visual-Guided Key-Token Regularization for Multimodal Large Language Model Unlearning

**arXiv ID:** 2601.22020 | [PDF](https://arxiv.org/pdf/2601.22020v1)

**作者:** Chengyi Cai `[一作]` (University of Melbourne), Feng Liu `[通讯]` (University of Melbourne)

**通讯引用:** 15460 | [OpenAlex ID](https://openalex.org/A5100325566)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于视觉引导的关键Token正则化（ViKeR）方法，用于多模态大语言模型的机器忘记，能够在不影响模型其它知识的前提下消除对指定图像-问题-答案三元组的记忆。

**💡 创新点**

创新点包括：①利用无标签的无关视觉输入来估计理想的Token分布，从而对关键Token与普通Token进行区分；②在梯度上升的忘记损失上加入Token级KL正则，使模型更专注于关键Token；③通过信息熵定义关键Token，并分析梯度重加权机制，证明其能减弱对普通Token的遗忘。

**🔧 技术方法**

使用的技术包括：多模态编码器+LLM的自回归Token概率模型；负对数似然与梯度上升的忘记损失；KL散度正则化；信息熵判定关键Token；LoRA微调和不需要改动模型结构的无梯度视觉参考估计。

**📊 数据集**

实验数据集：MLLMU和CLEAR两大基准；使用LLaVA-7B模型并在其上进行LoRA微调。

**📈 对比分析**

与传统方法GA、NPO、IdkPO进行比较。ViKeR在MLLMU 10%/15%任务中，遗忘准确率接近原始模型，保留指标（ROUGE、BLEU、GIB）均显著优于基线；在CLEAR数据集保持召回率和回答准确率，且在保留性能上提升约3.4%，遗忘性能几乎不受影响。

**⚠️ 局限性**

局限性包括：①需要至少5张无关视觉参考图像，且对参考图像类别敏感；②超参数λ需要在遗忘与保留之间权衡；③当前验证仅在单一问答场景和LLaVA-7B模型，未扩展到更大模型或多任务情境；④方法对视觉输入质量和多样性的依赖仍需进一步研究。

---

## 622. Exploring Diverse Generation Paths via Inference-time Stiefel Activation Steering

**arXiv ID:** 2601.22010 | [PDF](https://arxiv.org/pdf/2601.22010v1)

**作者:** Dongxuan Zhu `[一作]` (Chinese University of Hong Kong), Viet Anh Nguyen `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 815 | [OpenAlex ID](https://openalex.org/A5100625772)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 STARS，一种推理时无训练的激活向量正交化方法，用于提升大语言模型生成多样性。

**💡 创新点**

创新点在于将激活 steering 转化为 Stiefel 流形上的体积最大化，得到多路正交干预，并设计一次性闭式步长的轻量级更新。

**🔧 技术方法**

使用了 Riemannian 优化、Stiefel 流形、一次性梯度下降、正交向量正则化以及激活 steering 等技术。

**📊 数据集**

使用了 TESTEVAL（测试用例生成）和 LiveIdeaBench（科学创意生成）等公开数据集。

**📈 对比分析**

与传统温度采样等方法比较，在覆盖率、执行正确性、创意多样性等指标上均显著提升，尤其在低温情况下差距最大。

**⚠️ 局限性**

局限在于仍有额外计算开销（虽小于 2 秒/问），对极大模型或极高并行度的可扩展性未充分验证，且一次性更新的理论收敛性缺乏严谨保证。

---

## 623. Causal World Modeling for Robot Control

**arXiv ID:** 2601.21998 | [PDF](https://arxiv.org/pdf/2601.21998v1)

**作者:** Lin Li `[一作]`, Yinghao Xu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种自回归扩散框架，将视频动态预测与动作推断统一到同一隐空间中，实现了闭环机器人操控。

**💡 创新点**

创新点包括：
- 通过Mixture‑of‑Transformers将视频与动作令牌交错在同一序列中，既保持各自概念清晰又实现双向互调；
- 引入部分去噪（Noisy History Augmentation）和异步推理流水线，使动作预测可在未完成完整视频去噪时完成，显著降低推理延迟；
- 使用KV缓存和前向动力学模块实现持续闭环更新，解决了长序列漂移问题；
- 通过大规模视频‑动作预训练，提供了丰富的物理先验，显著提升少样本适配效率。

**🔧 技术方法**

核心技术包括：
- 条件流匹配（Conditional Flow Matching）在潜在空间进行视频生成；
- Mixture‑of‑Transformers（MoT）架构实现视频与动作的跨模态注意；
- KV缓存 + 异步推理 + 前向动力学（FDM）实现实时闭环控制；
- 采用RoPE、T5文本编码等常用LLM技术进行多模态融合。

**📊 数据集**

训练数据涵盖约16K小时的机器人演示，来源于Agibot、RoboMind、InternData-A1、OXE、UMI Data、RoboCOIN等；在仿真端使用RoboTwin 2.0（50任务）与LIBERO（四套共40任务）；在实机端收集50个演示用于微调。

**📈 对比分析**

与现有VLA与世界模型方法相比，实验表明：
- 在RoboTwin 2.0上取得92.9%（Easy）/91.6%（Hard）成功率，长序列任务提升约8–9%；
- 在LIBERO上平均成功率98.5%，在所有子集均达到或超过现有最高水平；
- 在真实机器人任务（Make Breakfast、Pick Screws等）上，比π_0.5提升超过20%，且仅需50条演示即可微调。整体表现位列SOTA。

**⚠️ 局限性**

局限性包括：
- 由于需要在潜在空间中进行多步去噪，仍存在推理时延，虽通过部分去噪和异步流水线降低，但在极低延迟场景下仍受限；
- 模型目前仅利用视觉与动作模态，缺乏触觉、力感知等多模态信息，可能在接触复杂动态任务中受限；
- 依赖大规模预训练数据，对数据分布偏移（如全新物体、颜色、纹理）仍可能出现泛化下降。

---

## 624. Investigating Batch Inference in a Sequential Monte Carlo Framework for Neural Networks

**arXiv ID:** 2601.21983 | [PDF](https://arxiv.org/pdf/2601.21983v1)

**作者:** Andrew Millard `[一作]` (University of Liverpool), Simon Maskell `[通讯]` (University of Liverpool)

**通讯引用:** 14815 | [OpenAlex ID](https://openalex.org/A5083636287)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了在贝叶斯神经网络中使用 Sequential Monte Carlo（SMC）采样，并通过逐步引入数据（data annealing）来提升训练效率；

**💡 创新点**

创新点在于提出多种数据引入调度策略，尤其是基于熵的平滑数据annealing（SDA），并系统评估其对采样速度与准确率的影响；

**🔧 技术方法**

使用的技术包括 SMC 粒子采样、Hamiltonian Monte Carlo 与 Langevin Dynamics 作为迁移核、mini‑batch 梯度估计以及熵驱动的温度调节；

**📊 数据集**

实验数据集为 MNIST 与 FashionMNIST 两个标准图像分类基准；

**📈 对比分析**

通过 5 种 naive 调度（常数、全批、两阶段、线性、自动化）和 SDA 方案进行对比，测量测试损失、准确率与运行时间，结果显示两阶段（CTR）方案在约 6× 的速度提升下，仅略微降低准确率；

**⚠️ 局限性**

局限性包括：对高维模型的计算成本仍较高，SDA 调度实现复杂，缺乏严格的收敛理论，且实验仅覆盖有限的网络与数据集。

---

## 625. Bridging Graph Structure and Knowledge-Guided Editing for Interpretable Temporal Knowledge Graph Reasoning

**arXiv ID:** 2601.21978 | [PDF](https://arxiv.org/pdf/2601.21978v1)

**作者:** Shiqi Fan `[一作]` (Northwestern Polytechnical University), Wen Hua `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 24766 | [OpenAlex ID](https://openalex.org/A5100428822)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种名为 IGETR 的混合框架，通过先用时间图神经网络（Temporal GNN）从 TKG 中提取结构化路径，再利用大语言模型（LLM）进行路径编辑与语义增强，最终使用图 Transformer 聚合得到可解释的预测。

**💡 创新点**

创新点在于：①三阶段流水线，将结构化推理与基于 LLM 的路径校正结合；②在路径编辑时引入硬约束（词表、类型一致性、时间顺序）避免逻辑错误；③使用 LLM 生成实体/关系文本描述来提升嵌入语义；④将随机傅里叶特征嵌入到图 Transformer 中，实现时序与结构的统一聚合。

**🔧 技术方法**

技术包括：时间图神经网络（带注意力采样）、LLM 生成文本描述与嵌入（GLM‑4‑Flash、text‑embedding‑v3）、LLM 编辑模块（Deepseek‑v3）、图 Transformer（改造自 Ruleformer）以及随机 Fourier 特征用于时间编码。

**📊 数据集**

使用 ICEWS14、ICEWS05‑15 以及 ICEWS18 三个公开 TKG 数据集进行实验，分别覆盖 2014、2005‑2015 与 2018 年的事件。

**📈 对比分析**

与传统基于图的基线（RE‑NET、RE‑GCN、CaORG、TiRGN 等）以及基于 LLM 的方法（GenTKG、GPT‑NeoX、Llama‑2‑ICL/COH 等）对比，IGETR 在 Hits@1、Hits@3、Hits@10 上均实现了显著提升，最高在 ICEWS05‑15 上 Hits@1 提升 5.6%，Hits@3 提升 8.1%。同时消融实验表明路径编辑模块对性能贡献最大。

**⚠️ 局限性**

主要局限包括：①由于 GNN 的采样和 LLM 的 token 限制，只能编辑固定数量的路径，导致 Hits@10 及在数据稠密的 ICEWS18 上效果有限；②在某些罕见但事实正确的路径上，LLM 可能因常识偏差而错误地修正；③虽然约束保证词表一致，但仍偶尔出现细微事实不一致。

---

## 626. Pay for Hints, Not Answers: LLM Shepherding for Cost-Efficient Inference

**arXiv ID:** 2601.22132 | [PDF](https://arxiv.org/pdf/2601.22132v1)

**作者:** Ziming Dong `[一作]` (University of Victoria), Kui Wu `[通讯]` (University of Victoria)

**通讯引用:** 24293 | [OpenAlex ID](https://openalex.org/A5100753551)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种LLM Shepherding框架，通过仅请求LLM的前缀提示（hint）来提升小模型(SLM)的推理质量，并显著降低整体推理成本。

**💡 创新点**

创新点在于突破传统二元路由/级联模型，允许按token级别灵活控制LLM使用量，并通过两阶段预测（是否需要提示、提示长度）实现精细化的LLM- SLM协作。

**🔧 技术方法**

核心技术包括：基于DeBERTa encoder + MLP的两阶段预测器（分类+回归，使用log变换和Smooth L1 loss），以及阈值化的提示长度决策；还结合了多样化的SLM响应特征进行反应式Shepherding。

**📊 数据集**

使用四大基准数据集：数学推理的GSM8K和CNK12，以及代码生成的HumanEval和MBPP。

**📈 对比分析**

与现有路由（RouteLLM、GraphRouter）和级联（FrugalGPT、ABC）方法对比，Reactive Shepherding在所有数据集上实现最高的Accuracy‑per‑Cost Efficiency（ACE），成本降低42%–94%，并在零样本跨域任务中保持与级联相当的准确率，同时比级联节省约2.8×成本。

**⚠️ 局限性**

局限性：提示长度预测仍可能出现过度或不足；模型依赖大量标注数据来学习最小有效提示；在SLM已具备较高性能的场景下，提示式协作的收益有限；此外，需要额外的LLM调用来生成提示，仍受LLM服务延迟和费用约束。

---

## 627. GeoNorm: Unify Pre-Norm and Post-Norm with Geodesic Optimization

**arXiv ID:** 2601.22095 | [PDF](https://arxiv.org/pdf/2601.22095v1)

**作者:** Chuanyang Zheng `[一作]` (Morgan Stanley), Xiaodong Liu `[通讯]` (Microsoft)

**通讯引用:** 7002 | [OpenAlex ID](https://openalex.org/A5100374810)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种基于球面曲线（Geodesic）优化的Transformer归一化方法GeoNorm，用以统一Pre-Norm与Post-Norm的视角；

**💡 创新点**

创新点在于将传统的投影式归一化改为在球面上执行指数映射（exponential map）更新，保留几何信息并引入层级学习率衰减；

**🔧 技术方法**

使用了黎曼几何中的指数映射、切空间投影以及多种衰减策略（sqrt、harmonic、linear）来实现GeoNorm；

**📊 数据集**

在Arxiv、Books3、FinWeb-Edu等文本数据集上进行评估，并在不同模型规模（125M、350M、1.3B）和不同训练长度（512、1024、2048、4096）上测试；

**📈 对比分析**

与Pre-Norm、Post-Norm、DeepNorm、SandwichNorm等归一化方案对比，GeoNorm在所有实验设置中均获得更低的训练损失和更平稳的收敛曲线；

**⚠️ 局限性**

局限性包括对球面假设的依赖、缺乏对更复杂结构（如多模态或非球面任务）的验证，以及对动态学习率调度的理论分析尚不充分。

---

## 628. Auditorily Embodied Conversational Agents: Effects of Spatialization and Situated Audio Cues on Presence and Social Perception

**arXiv ID:** 2601.22082 | [PDF](https://arxiv.org/pdf/2601.22082v1)

**作者:** Yi Fei Cheng `[一作]` (Carnegie Mellon University), David Lindlbauer `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1411 | [OpenAlex ID](https://openalex.org/A5058551017)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了通过声学化身（空间化音频与 Foley 声音）对会话代理的社会存在感与用户感知的影响，开展了 24 名受试者的 2×2 内测实验；

**💡 创新点**

首次系统性评估仅依赖音频的实体化表现（空间定位与环境音）对代理共在感与社交评价的双重作用，并揭示其带来的注意力与友好度下降等负面效应；

**🔧 技术方法**

采用 Meta XR Audio SDK 进行三维空间音频渲染、OptiTrack 头部追踪、预录制 Foley 音频与实时语音识别/合成技术；

**📊 数据集**

使用手工录制的 Foley 声音集与预设的对话脚本（无公开大型数据集）；

**📈 对比分析**

对比四种实验条件（单声道/空间化 × 有无 Foley），发现空间化与 Foley 能显著提升共在感（p<0.001），但会降低注意力、信息理解、亲和力和交谈词数，表现呈现正反两面；

**⚠️ 局限性**

受限于样本量小、实验室控制环境、预先脚本化的代理行为与社交规范不匹配、听觉定位的感知误差等，导致结果可能在自然场景与更复杂对话任务中不具普遍性。

---

## 629. Lens-descriptor guided evolutionary algorithm for optimization of complex optical systems with glass choice

**arXiv ID:** 2601.22075 | [PDF](https://arxiv.org/pdf/2601.22075v1)

**作者:** Kirill Antonov `[一作]` (Leiden University), Niki van Stein `[通讯]` (Leiden University)

**通讯引用:** 1024 | [OpenAlex ID](https://openalex.org/A5003248571)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并实现了 Lens Descriptor-Guided Evolutionary Algorithm（LDG‑EA），用于在多模态光学镜头设计空间中快速、并行地搜索多种高质量解，特别针对玻璃选择、曲率符号和厚度等离散与连续变量。

**💡 创新点**

创新点在于：① 引入基于曲率符号与材料索引的行为描述符，对设计空间进行先验划分；② 在每个描述符子空间内使用 Hill‑Valley EA 与 CMSA‑ES 产生多样化局部最优；③ 通过学习描述符分布动态调整采样，显著提升探索效率；④ 兼顾并行执行与梯度后处理，实现时效性与质量兼顾。

**🔧 技术方法**

技术包括：行为描述符映射、无偏分布学习（UMDA 变种）、Hill‑Valley Evolutionary Algorithm、Covariance Matrix Self‑Adaptation Evolution Strategy（CMSA‑ES）、梯度优化（BFGS）以及自动微分光学模拟。

**📊 数据集**

数据集为实际工业级双高斯（Double‑Gauss）六元素镜头设计问题，使用 120 种 Schott 玻璃材料、24 个设计变量（18 连续、6 整数）作为实验基准。

**📈 对比分析**

与传统全局 CMA‑ES 基线相比，LDG‑EA 在相同计算预算下生成约 14,741 个局部最优，分布在 636 个不同描述符；最高 RMS 结果为 F≈3×10⁻⁴，优于 CMA‑ES 的 F≈1×10⁻³，且整体运行时间仅约 1 小时（并行），比基线快 50 倍。

**⚠️ 局限性**

局限性包括：未达到调参参考设计的 7×10⁻⁵ RMS；仅支持正首曲率，无法处理负首曲率的设计；实验仅在单一镜头拓扑上验证；未考虑制造容差和多目标（成本、重量）等实际工程约束。

---

## 630. BLO-Inst: Bi-Level Optimization Based Alignment of YOLO and SAM for Robust Instance Segmentation

**arXiv ID:** 2601.22061 | [PDF](https://arxiv.org/pdf/2601.22061v1)

**作者:** Li Zhang `[一作]` (University of California San Diego), Pengtao Xie `[通讯]` (University of California San Diego)

**通讯引用:** 5607 | [OpenAlex ID](https://openalex.org/A5083884675)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了BLO-Inst，一个基于双层优化的统一框架，将YOLO检测器与SAM分割模型对齐，实现端到端实例分割。

**💡 创新点**

将检测器视为超参数，在下层训练SAM，在上层以SAM验证损失优化检测器，从而避免目标不匹配与对齐过拟合，提升生成提示的质量。

**🔧 技术方法**

双层优化（Bi‑Level Optimization）、参数高效微调（PEFT、LoRA）冻结SAM编码器、YOLOv7作为提示生成器以及梯度基优化等技术。

**📊 数据集**

使用六个公开数据集：一般对象集（PennFudanPed、TransIns、WheatIns、CarPartIns）和生物医学集（CellCountIns、RWCellIns）。

**📈 对比分析**

与Mask R‑CNN、SOLO、SAM+B/M、RSPrompter、USIS等基线对比，采用mAP、AP_50/AP_75评估，BLO‑Inst在所有基准上均取得最高mAP，提升4–12%不等，同时参数量和训练成本较低。

**⚠️ 局限性**

仍需手动划分训练/验证子集；双层优化在大规模数据上计算开销相对较大；对极端遮挡或密集场景的鲁棒性仍有提升空间。

---

## 631. $G^2$-Reader: Dual Evolving Graphs for Multimodal Document QA

**arXiv ID:** 2601.22055 | [PDF](https://arxiv.org/pdf/2601.22055v1)

**作者:** Yaxin Du `[一作]` (Shanghai Jiao Tong University), Siheng Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 8514 | [OpenAlex ID](https://openalex.org/A5066373402)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种双图结构的检索增强生成系统G^2-Reader，专门针对多模态长文档问答；

**💡 创新点**

创新点在于同时引入内容图（Content Graph）保留文档原生结构和跨模态语义，以及规划图（Planning Graph）以有向无环图形式实现子问题分解与迭代检索，解决传统RAG的碎片化与检索漂移问题；

**🔧 技术方法**

核心技术包括：多模态图构建与迭代演化（利用VLM进行节点属性与边更新）、结构化子图检索、规划图的构造与自适应重规划、以及基于VLM的答案生成与充分性检查；

**📊 数据集**

使用VisDoMBench多模态多文档问答基准，涵盖幻灯片、学术论文、表格、图形等五个子任务；

**📈 对比分析**

与多种基线（单模态VLM、传统RAG、文本图RAG、先进多模态RAG等）对比，G^2-Reader在所有子任务上平均准确率达到66.21%，明显优于最强对手VisDoMRAG(65.01%)，并超过同一基准下的GPT-5(53.08%);

**⚠️ 局限性**

局限性包括对图构建依赖于VLM与OCR的预处理，迭代规划可能导致过拟合或冗余子问题，且在极细粒度视觉细节或极大文档规模时仍可能出现检索误差或推理错误。

---

## 632. On the Paradoxical Interference between Instruction-Following and Task Solving

**arXiv ID:** 2601.22047 | [PDF](https://arxiv.org/pdf/2601.22047v1)

**作者:** Yunjia Qi `[一作]` (Tsinghua University), Juanzi Li `[通讯]` (Tsinghua University)

**通讯引用:** 14517 | [OpenAlex ID](https://openalex.org/A5003324011)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文研究了指令跟随对大语言模型任务解决能力的干扰，并提出了一个量化指标IF（instruction‑following robustness）来衡量在自证约束下模型的任务性能保持率。

**💡 创新点**

创新点在于首次将指令跟随的“干扰”作为研究对象，设计了基于成功答案自动生成自证约束的评估框架，并揭示了即便模型能满足约束，核心任务准确率仍会大幅下降。

**🔧 技术方法**

技术手段包括：自动化约束生成（规则+LLM提取）、对生成过程的注意力分布分析、对比不同后训练策略（SFT、长链式思维SFT、RL对齐）以及与传统IF和原始任务准确率的联合评估。

**📊 数据集**

使用的评估数据集包括数学推理（GSM8K、SVAMP、OlympiadBench、MATH500）、多跳问答（HotpotQA、2WikiMultiHop、Musique）以及代码生成（HumanEval、MCEval）。

**📈 对比分析**

实验通过IF、原始任务Accuracy和IF指标进行三维对比，结果显示即便IF>90%，IF-robustness 仍在50%–80%之间，尤其是代码生成任务的性能下降最为显著。

**⚠️ 局限性**

局限性在于评估仅覆盖英语语言与上述三大任务类型；自证约束生成依赖LLM，可能带来误差；未在多语言或其他任务域验证普适性。

---

## 633. SIA: Symbolic Interpretability for Anticipatory Deep Reinforcement Learning in Network Control

**arXiv ID:** 2601.22044 | [PDF](https://arxiv.org/pdf/2601.22044v1)

**作者:** MohammadErfan Jabbari `[一作]`, Tommaso Melodia `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了符号解释框架 SIA，用于实时解释预测增强的深度强化学习（DRL）网络控制代理，并提供可在不重训练的情况下提升代理性能的动作精炼模块。

**💡 创新点**

创新点包括：①将可预测 KPI 与当前观测分离，构建可扩展的 per‑kpi 知识图，实现符号化解释；②引入 Influence Score（IS）指标，能够在毫秒级别实时量化每个 KPI 对单个决策的影响；③动作精炼模块利用 IS 和知识图，在保持原有代理的情况下，直接改进决策，显著提升网络指标。

**🔧 技术方法**

核心技术：符号 AI（Fol 逻辑表达式）、KPI 预测模型（PatchTST、MLP‑RevIN、PatchTST 等）、符号化转化、per‑kpi 知识图构建、IS 计算、动作精炼算法。

**📊 数据集**

使用公开的 5G 视频流、Massive MIMO 以及 RAN slicing（Colosseum/O‑RAN near‑RT RIC）三组数据集，分别评估 ABR、MIMO 调度和 RAN 切片场景。

**📈 对比分析**

与 LIME、SHAP、METIS、EXPLORA 等传统 XAI 方法对比；解释延迟仅 0.65 ms，速度超过 200 ×；在 ABR 任务中通过 SIA 导致的代理重设计提升平均比特率 9%；在 RAN 切片任务中，动作精炼模块在不重训练的情况下实现 25% 的奖励提升。

**⚠️ 局限性**

局限性：①对预测准确性敏感，误差导致跨阈值时性能下降；②知识图冷启动需要离线数据预填；③符号化参数（阈值、类别数）需要细致调优，过少/过多均影响解释质量。

---

## 634. A Federated and Parameter-Efficient Framework for Large Language Model Training in Medicine

**arXiv ID:** 2601.22124 | [PDF](https://arxiv.org/pdf/2601.22124v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 635. Where Do the Joules Go? Diagnosing Inference Energy Consumption

**arXiv ID:** 2601.22076 | [PDF](https://arxiv.org/pdf/2601.22076v1)

**作者:** Jae-Won Chung `[一作]` (University of Michigan), Mosharaf Chowdhury `[通讯]` (University of Michigan)

**通讯引用:** 14700 | [OpenAlex ID](https://openalex.org/A5013180923)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对46个模型、7项任务、1,858个配置在NVIDIA H100/B200 GPU上做大规模推理能耗测评，并提出能耗推理的因果框架

**💡 创新点**

揭示能耗受内在潜在因素（内存使用、GPU利用率、算力量、应用约束）驱动，并发现低精度并非总能耗更低、增加GPU数可降低总能耗的逆向现象

**🔧 技术方法**

使用ML.ENERGY基准、Zeus能耗计量、vLLM与xDiT推理堆栈、FP8/BF16量化、MoE与dense模型对比

**📊 数据集**

涵盖多种开源模型（Qwen 3、DeepSeek R1、GPT OSS、Stable Diffusion、HunyuanVideo等）与对应数据集，任务包括文本推理、图像/视频生成

**📈 对比分析**

通过最小能耗配置与相同延迟约束对比，发现B200在大多数情况能耗比H100低约35%，但在高并行度短延迟场景H100更节能；能耗每标记/图片/视频分别可降低10–100×

**⚠️ 局限性**

受限于CPU侧预处理瓶颈、FP8量化在小批量下不优、不同硬件功耗基准不统一，框架尚未覆盖多GPU功率限制和核心频率调节等变量

---

## 636. Learning Hamiltonian Flow Maps: Mean Flow Consistency for Large-Timestep Molecular Dynamics

**arXiv ID:** 2601.22123 | [PDF](https://arxiv.org/pdf/2601.22123v1)

**作者:** Winfried Ripken `[一作]` (Technical University Berlin), Klaus Robert Müller `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种不需要轨迹数据的训练框架，利用平均流一致性约束从单个相空间样本学习Hamiltonian Flow Map（HFMs），从而实现分子动力学中大时间步长的稳定模拟。

**💡 创新点**

创新点在于将Mean Flow的自洽性条件移植到确定性Hamiltonian动力学，并通过只使用瞬时位势梯度（力）和相空间状态进行训练，完全避免了生成轨迹或教师模型的昂贵计算。

**🔧 技术方法**

使用的技术包括：
• 轨迹‑自由平均流一致性损失（Mean Flow Consistency Loss）；
• 3D几何Transformer网络预测平均速度与平均力；
• 推断过滤器（随机旋转、总动量去漂、耦合能量与角动量守恒校正）；
• 可变时间步长预测，支持任意 <∆t> < < ∆t_max。

**📊 数据集**

实验数据集包括：
• 经典单粒子系统（Barbanis势、弹簧摆）；
• 100 体重引力系统；
• MD17 有机分子（Aspirin、Ethanol、Naphthalene、Salicylic Acid 等）和含隐式溶剂的肽；
• Alanine Dipeptide 以及小型蛋白模拟。

**📈 对比分析**

与传统的Velocity Verlet（小步长 0.5 fs）以及基准MLFF对比：
• 在 1–9 fs 的大步长下，HFMs 的结构统计（h(r) MAE）、自由能曲面、功率谱等均与基准相当；
• 需要的积分步数减少约 4–10 倍；
• 在 10–15 fs 的步长下仍保持可接受的精度，显著拓宽了可行的时间尺度。

**⚠️ 局限性**

局限性包括：
• 随着步长增大，训练目标更难收敛，尤其在强混沌系统上表现不佳；
• 模型并非严格保持辛结构或完整的对称性，需要额外的推断过滤器才能保证能量和角动量守恒；
• 在极大步长（>10–15 fs）下，某些系统出现不稳定，原因尚未完全解析。

---

## 637. SMOG: Scalable Meta-Learning for Multi-Objective Bayesian Optimization

**arXiv ID:** 2601.22131 | [PDF](https://arxiv.org/pdf/2601.22131v1)

**作者:** Leonard Papenmeier `[一作]` (University of Münster), Petru Tighineanu `[通讯]` (Robert Bosch GmbH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可扩展的多任务多目标贝叶斯优化元学习框架，利用多输出高斯过程对任务间协方差进行稀疏建模，从而构造信息丰富的目标任务先验，显著提升样本效率。

**💡 创新点**

创新点在于将模块化多输出GP与任务共变矩阵稀疏化相结合，既保持完整贝叶斯不确定性传播，又实现对海量元任务的线性可扩展训练；并通过加权后验均值与协方差实现目标任务的精确先验推断。

**🔧 技术方法**

核心技术包括多输出高斯过程、核共变形与任务共变矩阵稀疏化、并行元任务训练与缓存、以及与标准MOBO采集函数的无缝集成；同时使用对数似然优化进行超参数学习。

**📊 数据集**

在合成 Sinusoidal、改写的 Hartmann6、多目标 HPOBench（四个数据集）以及 UAV 轨迹 Terrain benchmark 等四类基准上进行了实验评测。

**📈 对比分析**

与多种基线（单任务GP、Kronecker结构GP、TPE、AdaBLA、SGLT 等）对比，实验表明该方法在所有基准中均表现稳健，尤其在初期迭代能最快逼近最优 Pareto 前沿，并在多数任务上优于其他方法。

**⚠️ 局限性**

局限性包括：需要足够多且与目标相似的元任务才能发挥优势；当前模型仅支持正相关任务，无法处理负相关情况；在目标维度极高时的扩展性仍待进一步研究。

---

## 638. Prior-Informed Flow Matching for Graph Reconstruction

**arXiv ID:** 2601.22107 | [PDF](https://arxiv.org/pdf/2601.22107v1)

**作者:** Harvey Chen `[一作]` (Rice University), Santiago Segarra `[通讯]` (Rice University)

**通讯引用:** 3575 | [OpenAlex ID](https://openalex.org/A5012007074)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种先验信息流匹配（PIFM）框架，用于从部分观测的图中高保真地重建完整图。

**💡 创新点**

创新点在于将局部先验嵌入与连续时间流匹配结合，先做局部MMSE估计再通过学习全局耦合的最优传输，解决了传统嵌入缺乏全局一致性和生成模型缺少结构先验的问题。

**🔧 技术方法**

使用图神经网络实现的嵌入先验（GraphSAGE、node2vec、图窗）与连续流匹配（rectified flow matching）以及可置换等价的网络结构。

**📊 数据集**

在IMDB‑B、PROTEINS、ENZYMES（转导）和CORA（归纳）四个标准图数据集上进行实验。

**📈 对比分析**

与基准方法（单一先验预测、Gaussian先验的流、DiGress+RePaint、GDSS+RePaint）比较，PIFM在AUC‑ROC、AP、FPR/FNR等指标上普遍优于所有基线，尤其在高遮挡率下提升显著。

**⚠️ 局限性**

局限性包括仅针对同构图，先验估计仍受假设强度限制，且在已有先验性能已很高的场景中增益有限。

---

## 639. Beyond Martingale Estimators: Structured Estimators for Maximizing Information Freshness in Query-Based Update Systems

**arXiv ID:** 2601.22098 | [PDF](https://arxiv.org/pdf/2601.22098v1)

**作者:** Sahan Liyanaarachchi `[一作]` (University of Maryland), Nail Akar `[通讯]` (Bilkent University)

**通讯引用:** 1261 | [OpenAlex ID](https://openalex.org/A5080807022)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

本文研究了在查询式采样的连续时间马尔可夫链（CTMC）远程估计系统中信息新鲜度（Binary Freshness, BF）的评估与优化。

**💡 创新点**

提出了一类结构化估计器（structured estimators），尤其是p-MAP估计器，可在采样间期内动态更新估计，使其在时间可逆CTMC上与MAP估计器等价，并在非可逆CTMC上通过有限阶段逼近MAP估计。

**🔧 技术方法**

主要采用概率论与马尔可夫链理论推导MBF（Mean BF）的闭式表达式，利用SMDP框架与策略迭代求解状态依赖采样率最优策略；在多源场景下使用拉格朗日方法与二分搜索求解最优采样率分配。

**📊 数据集**

实验使用自行构造的CTMC生成矩阵（如星型、出生-死亡链等）进行仿真验证，无使用公开数据集。

**📈 对比分析**

与传统的马尔可夫估计器（ME）和单阈值τ-MAP估计器相比，结构化估计器在单源与多源情境下均提升MBF 10–17%，状态依赖采样进一步提升 4–15%，并通过仿真图表展示了显著的性能提升。

**⚠️ 局限性**

主要局限在于：1）对时间可逆CTMC提供了完整闭式结果，非可逆CTMC只能通过逼近；2）假设查询和状态传输时间可忽略；3）算法求解涉及离散化和迭代，实际实现时可能需要更高计算开销。

---

## 640. ReactEMG Stroke: Healthy-to-Stroke Few-shot Adaptation for sEMG-Based Intent Detection

**arXiv ID:** 2601.22090 | [PDF](https://arxiv.org/pdf/2601.22090v1)

**作者:** Runsheng Wang `[一作]` (Columbia University in the City of New York), Matei Ciocarlie `[通讯]` (Columbia University in the City of New York)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了从健康基线模型迁移至中风患者的 sEMG 意图检测方法，利用少量个体化数据实现快速微调；

**💡 创新点**

创新点在于首次将大规模健康 sEMG 预训练模型迁移至中风场景，并比较不同微调策略（Head‑only、LoRA、全微调）在数据有限条件下的表现；

**🔧 技术方法**

采用 Transformer‑Encoder 架构（ReactEMG），并使用 LoRA 低秩适配器、全模型微调和仅微调头部等技术；

**📊 数据集**

使用自建的三位慢性中风受试者数据集（每人 8 通道 sEMG、Myo 头带，MyHand 外骨骼），并与公开健康数据集预训练模型对比；

**📈 对比分析**

与健康零射手、从零开始的中风训练做对比，结果显示健康预训练 + 微调在转移准确率和原始准确率上分别提升约 0.19 与 0.09，且在姿势变动、摆放漂移等分布偏移下更稳健；

**⚠️ 局限性**

局限包括：仅评估三位受试者，个体差异导致微调策略优劣不一；适配过程仍需手工调参；对更大规模中风数据集的泛化尚未验证。

---

## 641. Accessibility-Driven Information Transformations in Mixed-Visual Ability Work Teams

**arXiv ID:** 2601.22081 | [PDF](https://arxiv.org/pdf/2601.22081v1)

**作者:** Yichun Zhao `[一作]` (University of Victoria), Sowmya Somanath `[通讯]` (University of Victoria)

**通讯引用:** 947 | [OpenAlex ID](https://openalex.org/A5048244685)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过为期一周的日志研究和访谈，记录并分析了23名盲/低视力（BLV）与视力正常团队成员在工作中为实现信息可访问性所进行的表示转化（如将PDF转为Word、简化电子表格等），并将转化过程拆解为触发、行动和后果三维框架，进一步归纳出四种典型模式（可抛弃式修复、转化成为标准、平行表示、组装），探讨其对工作负荷、团队协作与公平的影响。

**💡 创新点**

创新点在于：①首次系统性地把“表示转化”作为可访问性工作中的核心流程进行细粒度分析；②提出基于触发、行动、后果的案例结构化框架；③归纳四种转化与协作模式，为设计支持混合可视性团队的协同系统提供具体设计机会；④通过混合方法（日志、访谈、焦点小组）验证模式在不同团队环境中的一致性与差异。

**🔧 技术方法**

技术与方法：定性研究方法（日志研究、半结构化访谈、焦点小组），数据编码与主题分析（open coding → deductive coding → 主题归纳），无计算机程序或算法。

**📊 数据集**

数据集：共计42个转化案例（36来自日志+访谈，6来自焦点小组），涉及7个团队（法律、非营利、咨询、学术），23名参与者（14 BLV，9视力正常），记录了不同类型的表示（PDF、Word、表格、幻灯片、数据库等）。

**📈 对比分析**

对比与性能评估：本研究为描述性、探索性工作，无对照实验或性能指标；评估基于案例的频次与主题出现率，并通过参与者自述阐释转化模式的社会成本与收益。

**⚠️ 局限性**

限制：①样本量有限，主要来自加拿大维多利亚市的专业团队，可能不具代表性；②研究仅关注转化过程，未评估转化质量或团队整体绩效；③自述数据可能存在社会期望偏差；④技术进步（如GenAI）快速发展，研究时间点的工具与政策可能影响结果。

---

## 642. mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning

**arXiv ID:** 2601.22074 | [PDF](https://arxiv.org/pdf/2601.22074v1)

**作者:** Kevin Zakka `[一作]` (University of California Berkeley), Pieter Abbeel `[通讯]` (University of California Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一套轻量级、开源的机器人学习框架，结合 MuJoCo Warp GPU 加速仿真与 Isaac Lab 的管理式 API，支持易用的环境构建、低安装复杂度、透明物理，并内置三款机器人模型及参考任务。

**💡 创新点**

核心创新点在于：1）在保持管理式模块化的同时大幅降低依赖和安装门槛；2）通过 Warp 的多实例 GPU 模拟实现千环境并行；3）提供统一的 PyTorch‑native 接口和类型安全配置，使代码易于 AI 工具迭代；4）在现有框架基础上加入更细粒度的传感器、执行器、地形、视图等组件，兼顾可视化与调试。

**🔧 技术方法**

使用技术包括：MuJoCo Warp GPU 后端、Isaac Lab 的 ManagerTerm 设计、PyTorch + TorchArray 零拷贝接口、CUDA 图捕获、Ray‑cast 与接触传感器、可自定义的动力学控制器、Python CLI 配置（tyro）以及 RSL‑RL 的 on‑policy 学习。

**📊 数据集**

主要使用公开的机器人模型与场景数据（Unitree G1、G1、Go1、YAM arm），以及合成的地形网格、随机噪声高度场，未依赖外部大规模数据集。

**📈 对比分析**

与 Isaac Lab、Playground 等现有框架对比，本文框架在安装时间、启动延迟、物理可视化与调试成本方面显著更优；在三项示例任务（速度跟踪、动作模仿、立方体提起）上能快速收敛至合理策略，性能与现有 GPU 加速方案相当，但省去了 Omniverse 运行时等重负荷。

**⚠️ 局限性**

限制包括：仅支持单一物理后端 MuJoCo Warp，缺乏跨仿真器移植性；不提供高保真 RGB 渲染；对视觉感知策略支持有限；高度依赖 MuJoCo 许可与更新；在极端大规模多任务场景下仍需进一步验证扩展性。

---

## 643. VTC-R1: Vision-Text Compression for Efficient Long-Context Reasoning

**arXiv ID:** 2601.22069 | [PDF](https://arxiv.org/pdf/2601.22069v1)

**作者:** Yibo Wang `[一作]` (Nanyang Technical University), Dacheng Tao `[通讯]` (Nanyang Technical University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Vision‑Text Compression（VTC）方案，通过轻量渲染把长文本推理过程转换为图像，结合迭代推理实现高效长上下文推理。

**💡 创新点**

创新点在于无需额外训练或外部压缩模型，利用视觉语言模型直接对先前推理步骤的图像表示进行编码，实现 3–4 倍的 token 压缩并显著提升推理速度。

**🔧 技术方法**

使用了迭代推理框架、Glyph、Qwen3‑VL 等视觉语言模型、自定义渲染管线以及 vLLM 的并行推理技术。

**📊 数据集**

训练数据来自 OpenR1‑Math‑Inf（61K 问题/答案），评测数据包含 GSM8K、MATH500、AIME25、AMC23、GPQA‑Diamond 等数学与科学推理基准。

**📈 对比分析**

与标准长上下文推理（SFT）及 TokenSkip 对比，Glyph 上准确率平均提升 3–5%，速度提升 1.4–1.7×；Qwen3‑VL 上准确率保持竞争力，速度提升可达 6.6×。

**⚠️ 局限性**

局限性包括渲染及图像处理的额外开销、对多图输入的 VLM 兼容性要求，以及在非数学推理任务上的效果尚待验证。

---

## 644. MetricAnything: Scaling Metric Depth Pretraining with Noisy Heterogeneous Sources

**arXiv ID:** 2601.22054 | [PDF](https://arxiv.org/pdf/2601.22054v1)

**作者:** Baorui Ma `[一作]` (Li Auto Inc), Wei Chen `[通讯]` (Li Auto Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Metric Anything，一种可扩展的预训练框架，利用稀疏度量提示学习来自多源噪声 3D 数据的绝对深度估计，无需手工提示或特定摄像机建模，随后蒸馏为无提示的学生模型，可广泛应用于多任务。

**💡 创新点**

创新点包括：① 通过随机掩码生成稀疏度量提示，解耦空间推理与传感器/摄像机偏差；② 统一约 2M 万张图像‑深度对，涵盖 SfM、LiDAR、渲染等多源，首次在尺度上观察到 metric depth 的 scaling 趋势；③ 设计无提示蒸馏流程，得到可直接单目推理的学生模型，并在多任务中保持 SOTA 性能。

**🔧 技术方法**

技术手段：ViT‑DPT backbone 与 conditioned prompt head；Prompt‑Injection 与 adaptive layer normalization；随机掩码与 PDSA/GMDR 的前处理；Robust MAE + SSI‑MAGE 预训练损失；距离平衡的逆深度损失；多尺度梯度、log‑空间转换；多任务零样本评估；视觉语言模型融合提升空间推理。

**📊 数据集**

数据集：约 20M 图像‑深度对，覆盖 10,000+ 摄像机，来源包括 SfM/SLAM/MVS、LiDAR/ToF/RGB‑D、渲染三维等；下游任务数据集：NYUv2、ETH3D、KITTI、ScanNet、Booster、Sintel、NuScenes、Hypersim、VLA（LIBERO）、VSI‑Bench、VIS 等。

**📈 对比分析**

与 DepthAnything、DepthPro、MiDaS、ZoeDepth、UniDepth、MoGe‑2、MapAnything、LLaVA、VLM‑3R 等基线进行零样本/微调对比。Metric Anything 在尺度上显著提升，学生模型在单目深度、相机标定、多视角重建、Radar‑Camera 融合、VLA 规划、3D 空间推理等任务均达到或超过当前 SOTA，显示出跨任务的强泛化能力。

**⚠️ 局限性**

限制：仅支持中心投影相机模型，未扩展到非中心或多相机同步校准；架构规模扩展性未充分探索；极端低光、雨雾等极端环境下的鲁棒性仍有限；大规模预训练需要高算力与 GPU 集群。

---

## 645. MasalBench: A Benchmark for Contextual and Cross-Cultural Understanding of Persian Proverbs in LLMs

**arXiv ID:** 2601.22050 | [PDF](https://arxiv.org/pdf/2601.22050v1)

**作者:** Ghazal Kalhor `[一作]` (University of Tehran), Behnam Bahrak `[通讯]` (Tehran Institute for Advanced Studies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了MasalBench，评估多语大型语言模型在波斯谚语的上下文理解与跨文化对应能力。

**💡 创新点**

创新点在于大规模波斯谚语多选问答与跨语言等价二选一任务，并系统分析错误类型。

**🔧 技术方法**

使用Gemini 2.5 Pro自动提取与生成对话与干扰选项，并采用零shot、随机排列、温度0、top‑p1的提示技术进行评测。

**📊 数据集**

数据集来源于Foote Koozegari的4000+波斯谚语，筛选1000个常用谚语生成多选题，另外700个等价二选一题。

**📈 对比分析**

与8种主流多语LLM对比，语境理解准确率均>0.90，跨文化理解最高达0.79，说明模型在语境推理强但跨文化抽象弱。

**⚠️ 局限性**

限制在于Benchmark仅由单一LLM（Gemini 2.5 Pro）生成，等价谚语稀缺导致跨文化任务更难；未验证其他LLM生成质量。

---

## 646. Urban Neural Surface Reconstruction from Constrained Sparse Aerial Imagery with 3D SAR Fusion

**arXiv ID:** 2601.22045 | [PDF](https://arxiv.org/pdf/2601.22045v1)

**作者:** Da Li `[一作]`, Houjun Sun `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

融合3D合成孔径雷达（SAR）点云与稀疏航空图像，构建神经表面重建框架，提升城市三维重建精度。

**💡 创新点**

首次将3D SAR作为几何先验加入NSR，提出结构感知射线采样和基于雷达的射线边界约束。

**🔧 技术方法**

采用基于SDF的Neural Surface Reconstruction（如NeuS），结合可微渲染与雷达监督。

**📊 数据集**

构建SARMV3D跨模态数据集，包含对齐的航空图像、3D SAR点云和相机参数。

**📈 对比分析**

与COLMAP、NeuS和雷达仅基线对比，实验显示在Suzhou与Yuncheng场景中，Chamfer距离降低43%~78%，精度/召回率提升约50%及以上。

**⚠️ 局限性**

局限在雷达点云稀疏性和高噪声，且对大尺度分区仍需额外分块处理，未充分评估实时性和不同雷达平台兼容性。

---

## 647. Unsupervised Decomposition and Recombination with Discriminator-Driven Diffusion Models

**arXiv ID:** 2601.22057 | [PDF](https://arxiv.org/pdf/2601.22057v1)

**作者:** Archer Wang `[一作]` (Massachusetts Institute of Technology), Marin Soljačić `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种无监督的因子分解扩散模型，并通过判别器对重组样本进行对抗训练，从而提升因子发现和可组合生成的质量；

**💡 创新点**

创新点在于将判别器作为自监督的重组一致性信号引入扩散生成，既鼓励因子独立性，又保持重组样本的可辨识性；

**🔧 技术方法**

使用扩散模型（diffusion）+判别器对抗学习（adversarial）、重组因子采样、MIG/MCC 等判别指标；

**📊 数据集**

在CelebA‑HQ、Virtual‑KITTI、CLEVR、Falcor3D 等图像数据集，以及LIBERO机器人视频数据集进行实验；

**📈 对比分析**

与基线 Decomp Diffusion 及其它 disentanglement 方法对比，FID、MIG、MCC 指标均显著提升（如 CelebA‑HQ FID 由 82.7 降至 43.98，MIG/FID 均提高）；

**⚠️ 局限性**

局限在于对重组的可解释性和因子语义性缺乏严格保证，且在机器人环境下仍需改进抓取可靠性和对抗训练的稳定性。

---

## 648. Making Foundation Models Probabilistic via Singular Value Ensembles

**arXiv ID:** 2601.22068 | [PDF](https://arxiv.org/pdf/2601.22068v1)

**作者:** Mehmet Ozgur Turkoglu `[一作]` (Agroscope), Helge Aasen `[通讯]` (Agroscope)

**通讯引用:** 5053 | [OpenAlex ID](https://openalex.org/A5000845966)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了一种基于奇异值的隐式集成方法SVE，用来在大型基础模型上高效地进行不确定性量化。

**💡 创新点**

创新点在于只训练每个成员的奇异值、共享奇异向量，既实现了多样化集成，又将参数开销压至原模型的1%以内。

**🔧 技术方法**

主要技术包括对预训练权重进行奇异值分解（SVD），在此基础上进行奇异值微调（SVF），并与深度集成、MC Dropout、Batch/LoRA-Ensemble等方法做对比。

**📊 数据集**

实验涵盖NLP数据集ARC‑Easy、SST‑2，视觉数据集Flowers102、CIFAR‑100、DTD、Oxford Pets，并使用CIFAR‑10（OOD）和CIFAR‑100‑C（数据集漂移）进行鲁棒性评估。

**📈 对比分析**

与单模型、SVF、深度集成、LoRA‑Ensemble、Bayes‑LoRA等方法比较，SVE在准确率与校准（ECE、NLL）上与深度集成相当或更优，同时参数/显存增幅不到1%，显著低于其他隐式/显式集成方案。

**⚠️ 局限性**

局限性包括推理仍需多次前向传播导致FLOPs不减，且目前仅在有限的模型与任务上验证，低精度量化（4/8‑bit）时需要反量化等额外开销。

---

## 649. Reasoning While Asking: Transforming Reasoning Large Language Models from Passive Solvers to Proactive Inquirers

**arXiv ID:** 2601.22139 | [PDF](https://arxiv.org/pdf/2601.22139v1)

**作者:** Xin Chen `[一作]` (National Key Laboratory for Novel Software Technology Nanjing University), Shujian Huang `[通讯]` (National Key Laboratory for Novel Software Technology Nanjing University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了主动交互推理（PIR）框架，使大型语言模型能够主动提问并根据用户反馈调整推理路径，从而克服传统模型的盲目自我推理问题。

**💡 创新点**

创新点包括：①通过不确定性感知的数据增强构建交互式训练集，明确标注何时需要提问；②引入动态用户模拟器与复合奖励的US‑GRPO强化学习策略，显式平衡答案准确性、交互效率与提问有用性；③将主动提问融入内部推理链，真正实现模型从被动求解者转变为主动询问者。

**🔧 技术方法**

技术手段主要包括：基于不确定性的监督微调（SFT）与思考‑提问‑回应（think‑ask‑respond）格式；Group Relative Policy Optimization（GRPO）强化学习；动态用户模拟器（基于指令遵循LLM）；复合奖励设计（输出奖励 + 交互奖励）；以及针对不同任务的token/turns指标评估。

**📊 数据集**

使用的数据集：SFT交互训练集（从开放式问题生成）；Math‑Chat、BigCodeBench‑Chat、DocEdit‑Chat三大多轮任务集进行RL训练；评估时使用MMLU、MMLU‑Pro、TriviaQA、SQuAD、GSM8K、MATH等标准基准，以及Missing‑Premise Testing（MIP）等场景。

**📈 对比分析**

与多种基线（指令调优LLM、Proactive Prompt、STaR‑GATE、CollabLLM、基线推理LLM）进行对照实验；PIR在Math‑Chat上准确率提升11.40%，在BigCodeBench‑Chat的通过率提升3.20%，在DocEdit‑Chat的BLEU提升13.36；同时平均token数下降约1.3–1.7k，交互轮数显著减少，证明在多轮任务中既提高了性能又提升了效率。

**⚠️ 局限性**

局限性：①用户模拟器无法完全覆盖真实人类的语言多样性与动态意图，可能导致过度或不足提问；②缺乏针对敏感主题的安全性评估；③在某些知识稀缺或缺失前提场景下仍需进一步验证模型对交互策略的鲁棒性。

---

## 650. SWE-Replay: Efficient Test-Time Scaling for Software Engineering Agents

**arXiv ID:** 2601.22129 | [PDF](https://arxiv.org/pdf/2601.22129v1)

**作者:** Yifeng Ding `[一作]`, Lingming Zhang `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于轨迹回放的测试时刻缩放方法，利用已采样轨迹中的关键步骤重新分支，减少采样成本并提升软件工程代理的解决率。

**💡 创新点**

通过无须 LLM‑as‑a‑Judge 的步骤选择机制，依据仓库探索潜力与推理强度筛选关键步骤，实现高效、可泛化的回放式缩放。

**🔧 技术方法**

采用抽象状态分组、推理段落计数、softmax 采样、环境差分回放等技术构建步骤选择与回放框架。

**📊 数据集**

在 SWE‑Bench Verified、SWE‑Bench Pro 与 Multilingual 三大数据集上进行实验评估。

**📈 对比分析**

与传统“从头采样”基线相比，在 Verified 上成本下降 17.4% 并提升解决率 3.8%；在 Pro 与 Multilingual 上同样实现 22.6% 性能提升与 9% 成本下降。

**⚠️ 局限性**

仅适用于能记录并回放轨迹的代理，依赖于推理段落计数的可解释性，对极端长轨迹或非 Bash 脚本工具的兼容性尚未完全验证。

---

## 651. Value-Based Pre-Training with Downstream Feedback

**arXiv ID:** 2601.22108 | [PDF](https://arxiv.org/pdf/2601.22108v1)

**作者:** Shuqi Ke `[一作]` (Carnegie Mellon University), Giulia Fanti `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2073 | [OpenAlex ID](https://openalex.org/A5076026636)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于价值的预训练框架，利用少量可验证的下游反馈动态调整自监督预训练任务（目标或视图），从而在固定的无标签流和更新预算下提高模型在目标任务上的性能。

**💡 创新点**

创新点在于：① 将下游反馈转化为一个在线价值函数，衡量单步预训练梯度与下游梯度的对齐度；② 用轻量级任务设计器在保持原始预训练算法不变的前提下，改写目标分布（语言）或视图生成（视觉）；③ 通过一阶影响估计实现可扩展的双层优化，避免对完整预训练轨迹求导；④ 在多模态（语言、视觉）上验证该框架，展示与传统固定预训练目标的可比性。

**🔧 技术方法**

技术手段包括：自监督预训练（next-token、DINOv3等）；任务设计器（soft target生成器、可学习的视图模块）；价值函数 g_down·g_pre 的对齐度估计；一阶影响（Hessian–向量积）实现任务设计器更新；轻量化计算（仅更新适配器层或最后几层参数）。

**📊 数据集**

数据集：语言方面使用 GSM8K（推理）、NuminaMath CoT（继续预训练）和 MMLU、OMEGA 进行泛化评估；视觉方面使用 ImageNet1K（继续预训练）、ADE20K（语义分割）、NYUv2（深度估计）以及 R‑Oxford5k/R‑Paris6k（实例检索）进行转移性能评估。

**📈 对比分析**

与基线（固定下游目标的 next-token 或 DINOv3 预训练）相比，价值预训练在相同的预训练步数/token 下获得显著提升：语言模型 GSM8K Pass@1 提升 2–14%（小模型提升更明显），视觉模型 ADE20K mIoU 提升至 +1.07，NYUv2 RMSE 降低，ImageNet 线性准确率保持或提升。实验还展示了对随机反馈、均匀平滑、自我蒸馏等对照实验的优势，并验证了在不同规模、不同下游任务加权下的 Pareto 前沿控制能力。

**⚠️ 局限性**

局限性包括：① 对下游反馈的依赖需要预先提供可验证的标签集合；② 对于极大模型或极低资源场景，轻量级任务设计器的表现仍有待进一步验证；③ 目前仅支持可微的下游梯度信号，无法直接处理非可微的偏好或工具使用反馈；④ 任务设计器的超参（如 top‑K、混合系数）需要手动调优，影响易用性；⑤ 该方法在一定程度上增加预训练的计算开销，尤其是软目标生成与价值更新。

---

## 652. Investigating Associational Biases in Inter-Model Communication of Large Generative Models

**arXiv ID:** 2601.22093 | [PDF](https://arxiv.org/pdf/2601.22093v1)

**作者:** Fethiye Irmak Dogan `[一作]` (University of Cambridge), Hatice Gunes `[通讯]` (University of Cambridge)

**通讯引用:** 7555 | [OpenAlex ID](https://openalex.org/A5060090893)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过在生成式模型之间循环交互（图像-文本-图像）探究在人类活动与情感识别任务中，关联偏差如何在多轮模型信息交换中产生并演化。

**💡 创新点**

创新点在于：①提出一种可量化关联偏差漂移的交互式通信管道；②结合Token‑conditioned Grad‑CAM的可解释性管道，揭示偏差背后是否由无关视觉区域（如背景、头发）驱动；③系统评估偏差对下游预测准确性和公平性的影响，并给出数据、训练、部署层面的缓解策略。

**🔧 技术方法**

主要技术包括Stable Diffusion 3.5 Large（文本→图像）和LLaVA‑Next（图像→文本）模型的循环推理；使用Token‑conditioned Grad‑CAM实现区域归因；利用Stuart–Maxwell检验、Cohen’s κ、加权Jaccard衡量分布漂移；Logistic回归评估预测成功率的性别差异。

**📊 数据集**

实验数据集为PHASE（包含活动与情感标签、场景信息）与RAF‑DB（面部表情、基本情绪）。

**📈 对比分析**

通过与原始数据集的分布比较和统计检验验证漂移显著性，结果显示循环后图像更倾向于年轻与女性呈现；可解释性结果显示模型对背景/头发等无关区域的关注，说明偏差并非随机；在预测成功率上，部分类别（如Sports）虽提升整体准确率，却加剧性别差距。

**⚠️ 局限性**

局限包括：①只研究了两类数据集，缺乏更广泛场景验证；②偏差评估基于视觉感知属性，未涉及真实身份标签；③循环仅执行单轮迭代，无法探究多轮长期演化；④缺乏对不同模型配对（如不同文本-图像模型）的比较。

---

## 653. Latent Adversarial Regularization for Offline Preference Optimization

**arXiv ID:** 2601.22083 | [PDF](https://arxiv.org/pdf/2601.22083v1)

**作者:** Enyi Jiang `[一作]` (Stanford University), Sanmi Koyejo `[通讯]` (Stanford University)

**通讯引用:** 4781 | [OpenAlex ID](https://openalex.org/A5091266570)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在对话模型的偏好优化中加入了潜在空间对抗正则化，使用生成器与判别器来匹配策略模型与参考模型的内部表征分布；

**💡 创新点**

创新点在于将GAN思想应用到偏好学习的潜在空间，引入双对比判别器（区分高质量与低质量表征）以及相对平均GAN（RaGAN）实现结构化反馈；

**🔧 技术方法**

使用的技术包括DPO/SimPO偏好优化、相对平均GAN、Transformer判别器、以及基于潜在表征的对抗损失；

**📊 数据集**

主要数据集为UltraFeedback偏好数据集，用于训练与评估，同时在AlpacaEval 2.0、IFEval、GSM8K、MMLU、ANLI、TruthfulQA等基准上进行实验；

**📈 对比分析**

与DPO、SimPO等基线相比，GANPO在AlpacaEval的胜率、长度控制胜率提升1–2个百分点，且在高温采样时更稳健，后续任务表现不下降；

**⚠️ 局限性**

局限性包括需额外维护判别器导致计算开销增大、对参考模型质量敏感、超参数调优更复杂，并且目前仅适用于离线训练。

---

## 654. Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data

**arXiv ID:** 2601.22141 | [PDF](https://arxiv.org/pdf/2601.22141v1)

**作者:** Grzegorz Stefanski `[一作]`, Michal Byra `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研发了 Routing the Lottery (RTL) 框架，发现并训练多种针对不同类别、语义聚类或环境条件的稀疏子网络（adaptive tickets），以实现无额外参数的上下文感知推理。

**💡 创新点**

从传统单一“Winning Ticket”转向多子网络，利用掩码路由实现动态推理；提出子网络崩溃判定指标、无监督划分与共享初始化等技术，显著提升稀疏模型在异质数据上的性能。

**🔧 技术方法**

迭代幅度剪枝、掩码提取与联合再训练、Mask similarity 及语义/聚类划分、无监督分区、稀疏度控制与结构化约束。

**📊 数据集**

CIFAR‑10、CIFAR‑100、ADE20K（INR 重建）、DNS Challenge 2020 及 TAU Urban Acoustic Scenes 2020（语音增强）。

**📈 对比分析**

与单模型 IMP 与多模型 IMP 基线（同稀疏度、同架构）对比；RTL 在精度、召回率、SI‑SNRi、PSNR 等指标均高于基线，且参数量常低于多模型基线 10 倍以上。

**⚠️ 局限性**

需预先划分子集或聚类，划分质量影响效果；精确度相对召回率偏低，需后续校准；在极端稀疏下易出现子网络崩溃，需要及时监测与停止剪枝。

---

## 655. PI-Light: Physics-Inspired Diffusion for Full-Image Relighting

**arXiv ID:** 2601.22135 | [PDF](https://arxiv.org/pdf/2601.22135v1)

**作者:** Zhexin Liang `[一作]` (Nanyang Technological University), Xingang Pan `[通讯]` (Nanyang Technological University)

**通讯引用:** 3745 | [OpenAlex ID](https://openalex.org/A5052549072)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于扩散模型的全图像重光照框架PI-Light，采用两阶段逆向与正向神经渲染实现可控光照重现。

**💡 创新点**

创新点在于批量意识注意力保证内在特征一致性、物理启发的渲染模块与损失约束，以及仅使用前半球灰球环境光表示，实现物理可控且无需真实场景多光照数据。

**🔧 技术方法**

使用扩散模型（Stable Diffusion/LDM）、自注意力扩展、物理渲染公式（Principled BRDF）、VAE解码器、DINO感知损失等技术。

**📊 数据集**

构建了包含10,000+ Objaverse 物体与300个 BlenderKit 场景的全彩照/光照/内在属性数据集，使用HDRI、点光源、灰球灯等多光照采样。

**📈 对比分析**

在Object50和Scene200测试集上，与Intrinsic‑Anything、Kocsis、Zhu、RGB↔X、DiLightNet等方法对比，PI-Light在PSNR、SSIM、LPIPS等指标上均取得最高或相近最高成绩，视觉上能保持色彩、金属度并正确放置阴影。

**⚠️ 局限性**

局限性包括对稀疏光照场景的泛化仍受限、对透明/半透明物体内在预测精度不高、训练依赖大规模高质量数据，且在真实复杂场景中仍可能出现光照不精确。

---

## 656. Early and Prediagnostic Detection of Pancreatic Cancer from Computed Tomography

**arXiv ID:** 2601.22134 | [PDF](https://arxiv.org/pdf/2601.22134v1)

**作者:** Wenxuan Li `[一作]` (Johns Hopkins University), Zongwei Zhou `[通讯]` (Johns Hopkins University)

**通讯引用:** 19011 | [OpenAlex ID](https://openalex.org/A5084104975)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出并实现了一套名为ePAI的三阶段深度学习系统，用于从对比增强腹部CT扫描中早期检测和定位胰腺导管腺癌（PDAC），并在诊断期及预诊断期（3–36个月前）扫描上进行验证。

**💡 创新点**

创新点包括：①将解剖分割、病变定位与病变分类三阶段级联，提升可解释性；②利用生成式AI合成大量小肿瘤样本并结合像素级人工标注，显著提升对≤2 cm小肿瘤的敏感度；③在多中心（北美、欧洲、亚洲）成千上万病例上公开评估，证明在预诊断扫描中平均提前数百天检测到病灶；④与30名放射科医师的多读者实验显示，ePAI在敏感度和定位准确率上显著优于人类，且在辅助下读者性能进一步提升。

**🔧 技术方法**

采用的技术包括nnU-Net三阶段架构（解剖分割→病变定位→病变分类）、生成式AI合成小肿瘤数据增强、像素级分割与定位评估、深度学习分类器、LLM辅助数据标注、统计评估（AUC、敏感度、特异度、95%置信区间、HD95等）。

**📊 数据集**

使用的数据集为：1,598名患者的内部训练/测试集；外部多中心诊断期数据共数千名患者（北美、欧洲、亚洲）；约159名预诊断期（3–36 个月前）患者；以及包含诊断、预诊断与正常对照的读者研究数据；还包括肾移植者等正常控制。

**📈 对比分析**

在内部和外部测试中，ePAI的AUC约为0.96–0.97，诊断期敏感度≥94%（小肿瘤≤2 cm>90%），特异度≈96–99%；定位敏感度>90%。在预诊断扫描中，ePAI检测到75/159例（平均提前347天），而放射科医师仅检测到5.4%。在多读者实验中，ePAI在诊断期的敏感度为90%对比医师34%，在预诊断期为30.8%对比5.4%，并保持≈92%特异度。

**⚠️ 局限性**

局限性包括：①回顾性设计导致对图像协议、扫描参数的差异敏感；②多读者实验未评估阅读时间和工作流程效率；③未在前瞻性临床试验中验证实际诊疗价值；④在低发病率人群中的误报率仍较高；⑤对不同器官外观或扫描质量的泛化仍需进一步评估。

---

## 657. World of Workflows: a Benchmark for Bringing World Models to Enterprise Systems

**arXiv ID:** 2601.22130 | [PDF](https://arxiv.org/pdf/2601.22130v1)

**作者:** Lakshya Gupta `[一作]` (Skyfall AI), Sumit Pasupalak `[通讯]` (Skyfall AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于ServiceNow的企业工作流环境WoW及其评测基准WoW‑bench，用于评估LLM在有限可观测性、隐藏工作流和多跳约束下的自主决策与世界建模能力。

**💡 创新点**

创新点在于：①模拟真实企业系统的隐藏工作流与业务规则；②通过工具响应与表审计两种观测模式揭示“动态盲区”；③将评测任务划分为约束理解、代理任务完成、动作预测与审计预测四类，专注于动态推理与因果链。

**🔧 技术方法**

技术手段包括：利用ServiceNow沙箱搭建可自定义工作流的环境；将任务视作POMDP，设计工具调用动作；实现表审计日志作为oracle观测；在LLM代理中集成多工具调用与基于规则的策略。

**📊 数据集**

数据集：4,000+业务规则、55个活动工作流、55条表审计日志以及包含234条评测任务（67+67+50+50）的WoW‑bench。

**📈 对比分析**

比较方法：在工具响应（仅API反馈）与表审计（oracle）两种观测下评估GPT‑5.1、Gemini‑3‑Pro、Sonnet‑4.5和Opus‑4.5；指标包括任务成功率（TSR）、受约束任务成功率（TSRUC）以及平均成本；结果显示：即使在oracle观测下，TSRUC也低于20%，且世界建模任务的准确率不足30%。

**⚠️ 局限性**

局限性：任务与约束需要专家手工构造，难以扩展到其他系统；只评估单一ServiceNow实例，未包含跨系统交互；当前LLM未经过专门训练，评测侧重诊断而非提升。

---

## 658. The Patient is not a Moving Document: A World Model Training Paradigm for Longitudinal EHR

**arXiv ID:** 2601.22128 | [PDF](https://arxiv.org/pdf/2601.22128v1)

**作者:** Irsyad Adam `[一作]` (Standard Model Biomedicine), Kevin Brown `[通讯]` (Standard Model Biomedicine)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了一种新的临床 EHR 训练框架 SMB‑Structure，结合监督细调与联合嵌入预测，能够在单一嵌入中同时捕捉患者的临床语义和随时间演化的动态轨迹。

**💡 创新点**

创新点在于把自监督的潜在空间预测（JEPA）与传统的下一个词预测（SFT）联合起来，并在预测前先对未来潜在嵌入进行预测，迫使编码器提前学习疾病演化；同时引入临床结构标记、瓶颈预测器和两阶段（curriculum）/混合优化策略，以解决语义与动态学习的冲突。

**🔧 技术方法**

使用大语言模型（如 Qwen3‑1.7B）作为 backbone，构建了 JEPA 目标、SFT 目标、EMA 动量编码器、瓶颈预测器以及掩码策略；整体通过联合损失进行训练。

**📊 数据集**

在两大纵向 EHR 数据集上评估：Memorial Sloan Kettering（23,319 名肿瘤患者，323,000+ patient‑years）和 INSPECT（19,402 名肺栓塞患者，225M 事件）。

**📈 对比分析**

将 SMB‑Structure 与传统线性模型（LR、RF、XGBoost）以及单一 SFT 基线进行对比，采用线性探针在多时间节点评估 68 个下游任务；实验表明，SMB‑Structure 在多任务 AUC 上平均提升约 0.8–1.5%，在长期预测任务中的优势尤为明显。

**⚠️ 局限性**

主要限制包括：双正向前向导致计算成本上升；仅通过线性探针评估，未检验模型在 fine‑tune 过程中的鲁棒性；缺乏跨机构的外部验证，且尚未实现干预条件下的对抗性推理与治疗优化。

---

## 659. Creative Image Generation with Diffusion Model

**arXiv ID:** 2601.22125 | [PDF](https://arxiv.org/pdf/2601.22125v1)

**作者:** Kunpeng Song `[一作]` (Rutgers University), Ahmed Elgammal `[通讯]` (Rutgers University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于概率尾部探索的文本到图像创意生成框架，利用扩散模型对CLIP嵌入空间进行低概率区采样，并通过优化Token/LoRA参数实现图像创意。

**💡 创新点**

创新点在于：①将创意定义为低概率CLIP嵌入并用创意损失直接驱动分布尾部；②引入anchor loss和多模态LLM校验的pullback机制，保证语义合法性；③通过负向簇避免不理想的视觉方向，实现可控方向性。

**🔧 技术方法**

技术栈包括：Kandinsky 2.1 隐式扩散模型、PCA+多元高斯估计、LoRA低秩适配、CLIP文本/图像编码器、anchor cosine相似度约束、LLM（Janus‑1.3B/LLaVA‑Next）语义验证。

**📊 数据集**

实验基于Kandinsky 2.1预训练数据（内部多样化图像集），在多类目标（building、vehicle、alien、fruit 等）上进行采样与优化，未公开具体公开数据集名称。

**📈 对比分析**

与ConceptLab对比，本文方法在创意度、人类评估得分上提升约70%–75%，收敛速度更快（仅 50 步即可出现创意输出），生成质量保持高且无明显域外失真。

**⚠️ 局限性**

局限性包括：①对高维分布使用高斯近似，极端尾部可能受限；②对训练种子敏感，需负簇策略防止不理想输出；③pullback 机制和LLM校验增加计算开销；④在缺乏子类别信息时仍需大量采样来构建低概率模型。

---

## 660. Defining Operational Conditions for Safety-Critical AI-Based Systems from Data

**arXiv ID:** 2601.22118 | [PDF](https://arxiv.org/pdf/2601.22118v1)

**作者:** Johann Christensen `[一作]`, Sven Hallerbach `[通讯]` (Institute for AI Safety and Security)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于核函数的、数据驱动的安全设计方法，用以从已收集的数据自动构造安全关键 AI 系统的操作设计域（ODD），并提供了实现细节、参数选择与实时监测机制。

**💡 创新点**

创新点在于：①以确定性、无序列影响的方式从数据直接生成 ODD，打破传统专家手工定义的局限；②使用 RBF 核的全局相似度函数形成连续、可解释的亲和度曲线，天然满足安全设计的保守性；③通过与凸包的高相关性验证，为缺乏真值 ODD 的情形提供可行的阈值调优方案。

**🔧 技术方法**

核心技术包括：核方法（RBF）、多维度核参数自适应（基于最近邻距离），以及对 OOD 样本的阈值约束调整；实现工具 autoSAFE，利用 NumPy、SciPy、Faiss 等库完成邻居搜索、核函数评估与可视化；此外，还使用蒙特卡洛采样与真实航空碰撞规避数据进行验证。

**📊 数据集**

主要使用的数据集有：①在二维多边形与多项式关系下生成的随机合成数据（X = [-5,5]×[-5,5]，约 100k 验证样本）；②航空 VCAS 真实场景数据，约 622,110 个 anchor 点，验证样本约 7,000 个，涵盖五维参数空间。

**📈 对比分析**

与原始 ODD（已知解析边界）和所有 anchor 的凸包进行比较，分别绘制精度‑召回曲线；结果显示：R²>0.98（原始 ODD）与 R²>0.99（凸包），说明数据驱动 ODD 与原始 ODD 在覆盖性和精度上高度一致；召回率在高维情形略低，主要因样本稀疏导致。

**⚠️ 局限性**

局限性包括：①当前核参数假设为对角矩阵，无法捕捉跨维依赖；②对非凸、空洞结构的覆盖不足，凸包仅作为验证代理；③时间动态约束未被建模；④阈值与参数选择仍需手工调优，缺乏自动化和形式化安全绑定；⑤大规模数据时核参数估计与内存开销较高。

---

## 661. SINA: A Circuit Schematic Image-to-Netlist Generator Using Artificial Intelligence

**arXiv ID:** 2601.22114 | [PDF](https://arxiv.org/pdf/2601.22114v1)

**作者:** Saoud Aldowaish `[一作]` (University of Utah), Morteza Fayazi `[通讯]` (University of Utah)

**通讯引用:** 227 | [OpenAlex ID](https://openalex.org/A5060525404)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了SINA，一个全自动、开源的电路示意图图像到SPICE网表的生成器。

**💡 创新点**

将YOLOv11、CCL、OCR与VLM（GPT‑4o）协同应用，实现组件检测、连线推断、文本识别和参考符号分配，显著提升准确率。

**🔧 技术方法**

YOLOv11目标检测、Connected‑Component Labeling、EasyOCR、Vision‑Language Model GPT‑4o、数据增强与人工标注。

**📊 数据集**

自建的700+示意图标注集（计算机生成、扫描、手绘）以及评测集75个示意图（1000+元件）和40个对比测试图。

**📈 对比分析**

与公开的Masala‑CHAI进行对比，使用F1、整体准确率等指标；SINA整体准确率96.47%，比对手高2.72倍。

**⚠️ 局限性**

对极低分辨率或过于复杂的手绘图仍可能出现误检，且对检测不匹配的结果需要人工审查。

---

## 662. Vision-DeepResearch: Incentivizing DeepResearch Capability in Multimodal Large Language Models

**arXiv ID:** 2601.22060 | [PDF](https://arxiv.org/pdf/2601.22060v1)

**作者:** Wenxuan Huang `[一作]` (Chinese University of Hong Kong), Wanli Ouyang `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 43853 | [OpenAlex ID](https://openalex.org/A5087818121)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Vision‑DeepResearch，一种支持数十步推理和上百次搜索引擎交互的多模态深度研究框架，结合多实体、多尺度视觉搜索与文本检索。

**💡 创新点**

创新点在于：①将视觉检索转化为多尺度裁剪的试错过程，显著提高命中率；②通过图像描述桥接视觉轨迹与文本深度研究，充分利用强文本 LLM 的长推理能力；③在 SFT + RL 训练中实现高效异步滚动，提升长程决策与工具使用。

**🔧 技术方法**

技术手段包括：ReAct 方案、图像裁剪+搜索工具、网站抓取与摘要、基于 LLM 的评判器与裁决器、BF16/FP16 训练、GRPO+Leave‑One‑Out 的强化学习、异步多线程滚动、重复检测与格式错误修正。

**📊 数据集**

数据集：使用公开 VQA 数据集（如VDR、FVQA、MMSearch 等）进行过滤，结合自动化生成的高质量多模态深度研究轨迹、fuzzy multi‑hop VQA 生成数据，最终得到约30K轨迹、16K 事实 VQA、8K 文本 QA、6K fuzzy VQA 等训练样本。

**📈 对比分析**

在六大多模态事实基准（VDR、MMSearch、MMSearch‑Plus、FVQA、BC‑VL、MMS+）上，与现有开源 MLLM（Qwen3‑VL‑30B‑A3B、8B 等）以及闭源代理系统（GPT‑5、Gemini‑2.5‑Pro、Claude‑4‑Sonnet）对比，Vision‑DeepResearch‑30B‑A3B 取得 56.9% 总体准确率，较基线提升 16.0%，在多数指标上领先并与强大闭源代理相当。

**⚠️ 局限性**

局限性：仍依赖昂贵的在线搜索 API，RL 训练成本高；在极端长尾知识或需要跨模态多跳推理的场景中，模型可能因信息匹配误差导致失败；裁剪与搜索策略需要手动调参，对不同领域的泛化仍需进一步验证。

---

## 663. PLANING: A Loosely Coupled Triangle-Gaussian Framework for Streaming 3D Reconstruction

**arXiv ID:** 2601.22046 | [PDF](https://arxiv.org/pdf/2601.22046v1)

**作者:** Changjian Jiang `[一作]` (Zhejiang University), Mulin Yu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 335 | [OpenAlex ID](https://openalex.org/A5049833635)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种松耦合的三角形‑高斯混合表示，用于实时单目图像流的3D重建；

**💡 创新点**

将几何与外观解耦：用可学习的三角形捕捉结构，用神经高斯渲染纹理，并通过前馈模型和全局图优化实现在线初始化和更新；

**🔧 技术方法**

三角形可学习原语、可微分三角形光栅化、神经高斯渲染、前馈姿态估计、全局束调整、动态GPU/CPU加载；

**📊 数据集**

ScanNet++、ScanNetV2、VR‑NeRF、FAST‑LIVO2、KITTI、Waymo等室内外真实场景；

**📈 对比分析**

与2DGS、PGSR、MeshSplatting等离线方法以及ARTDECO、OnTheFly‑NVS等流式方法对比，几何Chamfer L2下降约18%，渲染PSNR提升≈1.3dB，帧率/训练时间比前沿方法快5×以上，原语数量显著减少；

**⚠️ 局限性**

对半透明/透明物体及远景天空的建模效果差，框架主要聚焦表面几何，易受前馈模型误差影响，仍需进一步完善。

---

## 664. Learning to Communicate Across Modalities: Perceptual Heterogeneity in Multi-Agent Systems

**arXiv ID:** 2601.22041 | [PDF](https://arxiv.org/pdf/2601.22041v1)

**作者:** Naomi Pitzer `[一作]` (University of Southampton), Daniela Mihai `[通讯]` (University of Southampton)

**通讯引用:** 30 | [OpenAlex ID](https://openalex.org/A5102783686)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了感知异质性下的多模态多步参考游戏，探讨不同感知模式的发送者与接收者如何通过可压缩的二进制信息实现目标识别并适应跨模态交互。

**💡 创新点**

首次系统性评估感知错位对 emergent communication 的效率、信息一致性和跨系统互操作性的影响，并揭示意义编码呈分布式而非位级组合化。

**🔧 技术方法**

采用深度神经网络（送信者为前馈网络，接收者为循环网络）结合 REINFORCE 与熵正则化的联合训练，配合二进制信息位扰动和 t‑SNE 聚类分析。

**📊 数据集**

使用合成“Shapes World”数据集以及真实图像‑音频对齐数据集（CIFAR‑100 与 UrbanSound8K/ESC‑50），验证模型在噪声与多模态条件下的表现。

**📈 对比分析**

通过比较单模态与多模态系统的分类准确率、信息熵、消息长度压缩阈值和跨模态 fine‑tuning 效果，发现单模态系统在压缩时更高效、熵更低；多模态系统需要更长消息并在跨系统对话中需微调数十个 epoch 便可恢复 70‑80% 的准确率。

**⚠️ 局限性**

实验受限于简化的两模态设定、固定消息长度、以及仅评估了静态场景下的对话；未来需扩展到更丰富的感知维度、动态环境与机器人平台，以验证方法的普适性与鲁棒性。

---

## 665. StepShield: When, Not Whether to Intervene on Rogue Agents

**arXiv ID:** 2601.22136 | [PDF](https://arxiv.org/pdf/2601.22136v1)

**作者:** Gloria Felicia `[一作]` (University of Virginia), Sandeep Bandarupalli `[通讯]` (University of Cincinnati)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出StepShield基准，评估代理行为检测器的时间敏感性而非仅判定是否违规。

**💡 创新点**

引入三项时序指标（Early Intervention Rate、Intervention Gap、Tokens Saved）并提供第一份细粒度步进标注的数据集。

**🔧 技术方法**

结合正则匹配、约束检查、LLM判定以及混合级联检测模型进行实验。

**📊 数据集**

使用9,213条代码代理轨迹数据集，包括1,278条训练对和7,935条测试轨迹，rogue率8.1%。

**📈 对比分析**

与四种检测方案对比，LLMJudge的EIR达0.59，静态匹配器仅0.26，体现2.3倍的早期检测差距，且HybridGuard在成本上实现75%下降。

**⚠️ 局限性**

受限于未覆盖所有可能的specification gaming形式、仅关注代码生成代理、经济模型简化以及对LLM提示敏感度的影响。

---

## 666. PRISM: Distribution-free Adaptive Computation of Matrix Functions for Accelerating Neural Network Training

**arXiv ID:** 2601.22137 | [PDF](https://arxiv.org/pdf/2601.22137v1)

**作者:** Shenghao Yang `[一作]`, Michael W. Mahoney `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为 PRISM 的框架，用于加速矩阵函数（如平方根、逆根和极化）计算，并将其应用于深度学习中的优化器。

**💡 创新点**

创新点在于：①在不需要预先获得谱界或奇异值分布的前提下，通过自适应多项式拟合动态匹配当前迭代的谱；②结合随机化子空间嵌入（sketching）降低多项式拟合成本，从而实现GPU友好的迭代加速。

**🔧 技术方法**

技术主要包括：多项式逼近（Taylor+自适应系数）、随机化子空间嵌入（OSE）、Newton–Schulz 迭代的高阶变种以及对矩阵正定性/对称性的假设。

**📊 数据集**

实验数据集包括 CIFAR‑10 / CIFAR‑100 上的 ResNet‑20/32，以及 FineWeb 上的 GPT‑2‑Large 模型。

**📈 对比分析**

与传统的特征值分解、PolarExpress 等方法对比，PRISM 在 Shampoo 与 Muon 优化器中实现了显著的加速（例如训练时间缩短 10‑30%），并在不同谱分布（高斯、Marchenko‑Pastur、重尾）下保持稳定的性能提升。

**⚠️ 局限性**

局限性包括：对非对称矩阵的理论分析尚未完整；需要在每一步选择多项式次数与 sketch 维度，虽可通过经验设置，但并非完全自动；在极端谱宽度或数值不稳定场景下可能仍需额外的稳定性处理。

---

## 667. EditYourself: Audio-Driven Generation and Manipulation of Talking Head Videos with Diffusion Transformers

**arXiv ID:** 2601.22127 | [PDF](https://arxiv.org/pdf/2601.22127v1)

**作者:** John Flynn `[一作]` (Pipio AI), Guy Gafni `[通讯]` (Pipio AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种基于Diffusion的多模态视频编辑框架，支持音频驱动的对话编辑，包括插入、删除、重定时，并保持高质量口型同步与身份一致性。

**💡 创新点**

引入双阶段训练、窗口式音频条件、隐藏空间对话编辑、前后RoPE身份约束、TAPSF长视频推理、可变帧率支持，构建首个文本驱动的视觉对话编辑系统。

**🔧 技术方法**

使用LTX-Video的Diffusion Transformer（DiT）+流匹配训练、跨模态音频注意力、面部参考令牌、前后RoPE、TAPSF长推理、FP8量化与混合序列并行等技术。

**📊 数据集**

使用约1070小时对话视频（70小时高质量前置录制+1000小时YouTube短视频），经过滤后475小时，并使用CogVLM2生成字幕。

**📈 对比分析**

在TalkVid和VBench数据集上与LatentSync、InfiniteTalk、MuseTalk、Pixverse等多种I2V/V2V基准进行比较，FID/FVD/CSIM/Sync-C/D指标显示其在自重现和新音频下均优于多数基准，尤其在口型同步和身份保持方面处于领先。

**⚠️ 局限性**

仍依赖参考令牌维持身份，长视频中出现轻微身份漂移；在极端场景（复杂背景、极短/极长视频）表现下降；需要大量训练数据与算力，对低帧率或低音质视频的适应性有限。

---

## 668. Physics Informed Reconstruction of Four-Dimensional Atmospheric Wind Fields Using Multi-UAS Swarm Observations in a Synthetic Turbulent Environment

**arXiv ID:** 2601.22111 | [PDF](https://arxiv.org/pdf/2601.22111v1)

**作者:** Abdullah Tasim `[一作]` (University of Oklahoma), Wei Sun `[通讯]` (University of Oklahoma)

**通讯引用:** 8497 | [OpenAlex ID](https://openalex.org/A5115600683)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过无人机编队的局部风速估计与物理信息神经网络，重构了四维（空间-时间）大气风场。

**💡 创新点**

创新点在于仅依赖多无人机的动力学响应进行局部风估计，无需专用风速传感器，并将估计结果通过PINN实现空间时间连续的风场重构。

**🔧 技术方法**

采用双向长短时记忆网络（Bi‑LSTM）进行局部风估计，使用物理信息神经网络（PINN）结合弱物理约束进行全域重构，同时基于von Kármán谱的合成湍流模拟。

**📊 数据集**

使用基于von Kármán谱的合成风场和高保真多旋翼仿真产生的训练与评估数据集。

**📈 对比分析**

与合成真值对比，五架无人机配置在中等风况下整体RMSE为0.118 m/s，低于四架配置但高于更大编队，表明中等规模编队能实现最佳重构精度。

**⚠️ 局限性**

主要局限包括垂直风分量估计误差较大、编队数量过多时误差反弹，以及未在实际飞行数据上验证。

---

## 669. ECO: Quantized Training without Full-Precision Master Weights

**arXiv ID:** 2601.22101 | [PDF](https://arxiv.org/pdf/2601.22101v1)

**作者:** Mahdi Nikdan `[一作]` (Google Research), Vahab Mirrokni `[通讯]` (Google Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种在不使用主权重量的情况下进行低精度LLM训练的优化器——Error-Compensating Optimizer (ECO)，并在FP8等量化上实现。

**💡 创新点**

通过将量化误差注入优化器的动量缓冲，构建无额外存储的误差反馈循环，消除高精度主权重量。

**🔧 技术方法**

采用FP8量化、随机舍入、动量注入、SGDM/Adam、梯度累积禁用等技术。

**📊 数据集**

在C4、LM1B、OpenAssistant-Guanaco等多种数据集上评估，模型规模从30M到16B。

**📈 对比分析**

与保留主权重量的BF16/FP8 QAT基线相比，ECO在无主权重量时几乎保持同等验证损失，内存占用下降约25%，并在大模型中保持收敛。

**⚠️ 局限性**

主要局限在仅对随机舍入具有最优保证，采用RTN时噪声更高；硬件若不支持随机舍入，效果受限。

---

## 670. Boosting CVaR Policy Optimization with Quantile Gradients

**arXiv ID:** 2601.22100 | [PDF](https://arxiv.org/pdf/2601.22100v1)

**作者:** Yudong Luo `[一作]` (HEC Montréal), Erick Delage `[通讯]` (GERAD)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种将CVaR优化与VaR优化相结合的策略梯度方法（CVaR‑VaR），通过引入VaR Bellman算子实现了在Markov策略下的VaR‑PG，并将其与传统的CVaR‑PG联合训练，以提高样本效率和学习速度。

**💡 创新点**

创新点在于：① 设计了新的VaR Bellman算子并推导出可用于actor–critic的更新规则；② 在Markov策略框架下通过跟踪累计奖励动态确定风险水平α，从而实现VaR优化的近似；③ 将此VaR‑PG与CVaR‑PG以权重ω混合，使得算法既保留了CVaR对尾部风险的关注，又利用VaR的动态规划优势提升样本利用率。

**🔧 技术方法**

核心技术包括：策略梯度（policy gradient）、VaR与CVaR的变分表示、量化回归（quantile regression）损失、actor–critic框架、softplus实现量化值的单调性、λ‑回报（λ‑return）多步优势估计、以及对随机奖励的处理。

**📊 数据集**

使用了三种自定义环境：Maze、LunarLander（改造版）和InvertedPendulum（改造版），每个环境均设计了风险敏感的奖励结构以便验证风险厌恶策略的学习。

**📈 对比分析**

与基线方法（CVaR‑PG、PCVaR‑PG、RET‑CAP、MIX以及REINFORCE）在相同的on‑policy设置下比较，实验结果显示CVaR‑VaR在所有环境中都显著提高了期望回报、风险厌恶率和CVaR指标，并且收敛速度快于其它方法；在Maze中几乎完全实现风险厌恶路径，在LunarLander和InvertedPendulum中均达到最高的0.2‑CVaR值。

**⚠️ 局限性**

主要局限包括：① 对于任意初始化的策略与价值函数，理论上尚不保证该Markov‑VaR算法的收敛性；② 算法需要通过跟踪累计奖励来动态确定α，若环境的奖励分布不稳定或存在极端噪声，α的估计可能误差较大；③ 与离线或重要性采样等技术结合的效果未被充分验证，未来工作需进一步探索与其他高效学习技术的融合。

---

## 671. RefAny3D: 3D Asset-Referenced Diffusion Models for Image Generation

**arXiv ID:** 2601.22094 | [PDF](https://arxiv.org/pdf/2601.22094v1)

**作者:** Hanzhuo Huang `[一作]` (ShanghaiTech University), Sibei Yang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 1033 | [OpenAlex ID](https://openalex.org/A5043811579)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种3D资产参考的扩散生成框架，能够在保持3D几何与纹理一致性的同时生成高质量2D图像。

**💡 创新点**

创新点在于采用空间对齐的双分支结构，联合生成RGB图像与点图并通过共享位置编码和域解耦实现跨域一致性。

**🔧 技术方法**

使用Flux.1-dev扩散模型、DiT架构、共享位置编码、Domain-specific LoRA和Text-agnostic Attention等技术，并利用多视角RGB与点图作为条件。

**📊 数据集**

构建了基于Subjects200k的扩展数据集，结合GroundingDINO、Hunyuan3D、FoundationPose等工具生成3D资产与姿态标注。

**📈 对比分析**

与Textual Inversion、DreamBooth、IP-Adapter、DSD、OminiControl等方法对比，采用CLIP、DINO、GIM及GPT‑5评估，结果在3D一致性、身份保持与美观度上均领先。

**⚠️ 局限性**

局限在于对非刚性物体适应不足、对多视角的计算开销较大，以及数据集对弹性物体支持有限。

---

## 672. Late Breaking Results: Conversion of Neural Networks into Logic Flows for Edge Computing

**arXiv ID:** 2601.22151 | [PDF](https://arxiv.org/pdf/2601.22151v1)

**作者:** Daniel Stein `[一作]` (TU Darmstadt), Grace Li Zhang `[通讯]` (TU Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将神经网络转换为逻辑流以降低CPU上的MAC运算量并提升推理速度

**💡 创新点**

首次使用等价决策树提取常量叶子路径，生成if‑else逻辑流并实现混合执行

**🔧 技术方法**

决策树构造、混合整数规划(Gurobi)识别常量叶子、逻辑流压缩、C语言代码生成

**📊 数据集**

MNIST*（偶数/奇数分类）、Occupancy I、Occupancy II三大公开数据集

**📈 对比分析**

与在RISC‑V Ibex模拟器上直接执行的参考C实现对比，平均延迟可降低约39.3%（最小延迟最高达52.2%），无准确率下降

**⚠️ 局限性**

仅针对三层全连接网络，未涵盖深层网络和多类别分类，逻辑流覆盖率受限

---

## 673. JUST-DUB-IT: Video Dubbing via Joint Audio-Visual Diffusion

**arXiv ID:** 2601.22143 | [PDF](https://arxiv.org/pdf/2601.22143v1)

**作者:** Anthony Chen `[一作]` (Tel Aviv University), Daniel Cohen-Or `[通讯]` (Tel Aviv University)

**通讯引用:** 40705 | [OpenAlex ID](https://openalex.org/A5036688260)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种单模型的视频配音方法，通过在音视频扩散模型上轻量化LoRA微调，实现音频与面部运动同步生成。

**💡 创新点**

创新点在于将配音任务转化为联合音视频生成，利用生成模型自我合成双语配对数据并通过局部掩码与跨模态注意力实现身份与语音同步。

**🔧 技术方法**

采用音视频扩散Transformer（LTX‑2）、LoRA适配、语音视频 in‑painting、跨模态位置编码与模态隔离注意力等技术。

**📊 数据集**

使用自合成的多语种配对视频以及公开数据集HDFT、TalkVid和自采的25条Youtube真视频/25条合成视频进行评估。

**📈 对比分析**

与MuseTalk、LatentSync、CosyVoice、OpenVoice及商业工具HeyGen比较，取得更低的FVD、SyncNet误差、WER并保持100%生成成功率，性能优于基准。

**⚠️ 局限性**

局限在于仍难以完全保持所有场景下的声音身份一致，需进一步强化语音与身份的分离与更长时序建模。

---

## 674. One-step Latent-free Image Generation with Pixel Mean Flows

**arXiv ID:** 2601.22158 | [PDF](https://arxiv.org/pdf/2601.22158v1)

**作者:** Yiyang Lu `[一作]` (Massachusetts Institute of Technology), Kaiming He `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种称为像素 MeanFlow (pMF) 的单步无潜空间图像生成方法，直接在像素空间学习平均速度场并输出与噪声图像对应的去噪图像；

**💡 创新点**

在 MeanFlow 框架中引入像素级的目标预测，将去噪图像视为低维数据流形，并通过转换将其与平均速度场关联，实现一次性采样的无潜空间生成；

**🔧 技术方法**

利用流匹配 (Flow Matching)、MeanFlow、以及 JiT 的去噪图像预测技术，构建 Transformer 基础的网络；

**📊 数据集**

在 ImageNet 数据集上进行实验，分辨率分别为 256×256 与 512×512；

**📈 对比分析**

与现有多步或潜空间模型对比，pMF 在单步无潜空间设置下取得 256×256 2.22 FID 与 512×512 2.48 FID 的优秀效果，表明其在生成质量与效率上的竞争力；

**⚠️ 局限性**

局限性包括：网络容量对同时处理流形学习和单步建模的要求较高；需要精心设计目标转换，否则性能会急剧下降；目前仅在 ImageNet 上验证，缺乏对其他数据集或更高分辨率的进一步测试。

---

## 675. UEval: A Benchmark for Unified Multimodal Generation

**arXiv ID:** 2601.22155 | [PDF](https://arxiv.org/pdf/2601.22155v1)

**作者:** Bo Li `[一作]` (Princeton University), Zhuang Liu `[通讯]` (Princeton University)

**通讯引用:** 1385 | [OpenAlex ID](https://openalex.org/A5100452088)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个评估统一多模态生成的基准（Unified Multimodal Generation Benchmark, UMG），该基准要求模型在同一查询中同时生成文本和图像，涵盖8类真实场景任务。

**💡 创新点**

创新点在于：①采用基于rubric的评价框架，自动生成并人工校正10,417条评价标准；②将多模态大模型（MLLM）用作判分器，实现可复现的自动评分；③引入链式思考（CoT）推理痕迹，验证推理对生成质量的提升；④聚焦闭合式与开放式两类任务，强调多步图像一致性与多模态互补。

**🔧 技术方法**

技术手段包括：使用 Gemini‑2.5‑Pro 生成评价rubric，Gemini‑2.5‑Flash 和 Gemini‑2.5‑Pro 作为判分模型，GPT‑5‑Thinking 与 GPT‑5‑Instant 进行推理实验，利用多模态大模型（如 Gemini、Emu3.5、GPT‑5 系列）进行生成；评价过程中将CoT痕迹附加到提示中。

**📊 数据集**

数据集：1,000 条专家编写的问题，涵盖 8 个任务（空间、教科书、图表、论文、艺术、生活、技术、运动），每个问题配有文本与图像参考答案；共 10,417 条经过人工校验的rubric 评价标准。

**📈 对比分析**

比较方法：在 UMG 上评估 9 种统一多模态模型。结果显示 GPT‑5‑Thinking 最高得分 66.4/100；最佳开源模型 Emu3.5 仅 49.1/100；模型普遍在多步图像一致性上表现不佳；将推理痕迹输入非推理模型显著提升视觉输出质量，说明 CoT 对多模态生成有益。

**⚠️ 局限性**

局限性：①生成多步图像仍缺乏一致性，尤其在绘图任务中出现标签错误或时间不连贯；②开源模型对推理痕迹响应不佳，提升有限；③评价主要依赖 MLLM 判分器，可能存在模型偏差；④任务范围仅覆盖 8 类，缺乏更广泛场景的验证。

---

## 676. FineInstructions: Scaling Synthetic Instructions to Pre-Training Scale

**arXiv ID:** 2601.22146 | [PDF](https://arxiv.org/pdf/2601.22146v1)

**作者:** Ajay Patel `[一作]` (University of Pennsylvania), Chris Callison-Burch `[通讯]` (University of Pennsylvania)

**通讯引用:** 21328 | [OpenAlex ID](https://openalex.org/A5068508539)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过使用大型语言模型生成大规模的合成指令数据，并将其用于语言模型预训练，提升模型的指令跟随能力。

**💡 创新点**

将合成指令的规模提升到预训练级别，结合对生成内容的过滤与转换，首次实现了高质量合成指令的批量化生成。

**🔧 技术方法**

使用基于提示的生成、数据筛选、对齐技术和少量人工审查，借助 GPT‑4/类似模型生成指令与回答。

**📊 数据集**

主要使用公开指令数据集如 InstructGPT、Alpaca、OpenAI Instruction Benchmark，外加自生成的合成指令数据。

**📈 对比分析**

与传统人工构造指令集和现有合成指令方法相比，在 LLM 指令跟随基准（如 AlpacaEval、OpenAI‑Curie）上提升了 5–10% 的准确率。

**⚠️ 局限性**

生成数据可能放大原模型偏见与错误，且依赖高质量的过滤流程；规模化生成仍需昂贵算力与人工检查。

---

## 677. Hybrid Linear Attention Done Right: Efficient Distillation and Effective Architectures for Extremely Long Contexts

**arXiv ID:** 2601.22156 | [PDF](https://arxiv.org/pdf/2601.22156v1)

**作者:** Yingfa Chen `[一作]` (Tsinghua University), Zhiyuan Liu `[通讯]` (Tsinghua University)

**通讯引用:** 21790 | [OpenAlex ID](https://openalex.org/A5100320711)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一套低成本混合 Transformer‑RNN 架构和蒸馏流程，将预训练 Transformer 转化为长上下文高效模型。

**💡 创新点**

提出了跨架构蒸馏方法 Hybrid Attention via Layer Optimization 与 HyPE 位置编码，结合注意力层选择，显著提升长上下文性能。

**🔧 技术方法**

使用软注意力、线性/状态空间 RNN、RoPE/NoPE 位置编码、动态注意力缩放、QK‑归一化、GQA‑>MHA 解耦、输出门以及多种 RNN mixer（如 Lightning Attention、Mamba 等）技术。

**📊 数据集**

仅使用 FineWeb‑Edu 10B 语料（约 2.3B 训练标记）进行蒸馏与长上下文微调。

**📈 对比分析**

在 128K 上下文长度下与 Qwen3 及其他蒸馏模型对比，得到比原 Transformer 更快、更少显存，长上下文 recall 接近或优于原模型；在 NIAH 任务上 1–2B 模型实现 99% 级别准确率。

**⚠️ 局限性**

蒸馏过程中会丢失部分指令/对齐行为，且方案仅针对基于 Transformer 的架构，跨架构迁移受限。

---

## 678. Exploring Reasoning Reward Model for Agents

**arXiv ID:** 2601.22154 | [PDF](https://arxiv.org/pdf/2601.22154v1)

**作者:** Kaixuan Fan `[一作]` (Chinese University of Hong Kong), Xiangyu Yue `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Agent‑RRM 多维奖励模型，并基于其设计 Reagent 系列代理模型（Reagent‑C、Reagent‑R、Reagent‑U）实现多层反馈驱动的 agentic RL。

**💡 创新点**

创新点在于同时提供：① 结构化的推理轨迹分析、② 针对性批判提示以及③ 统一的整体评分，三者互补提升训练与推理质量；并通过三种整合策略探索文本批判与数值奖励的协同效应。

**🔧 技术方法**

技术包括：GRPO（Group Relative Policy Optimization）强化学习框架、基于 Qwen3‑8B 的大模型、Reward‑Model 的两阶段训练（SFT+GRPO）、工具箱（搜索、网页浏览、Python、文件读取、图像描述、音频转写）以及对话式评估与批判生成。

**📊 数据集**

使用四大专用数据集：① Reagent‑SFT‑55.6K（高质量轨迹）用于 SFT；② Reagent‑RL‑709K（709k QA 轨迹）用于 RL；③ Reagent‑RRM‑SFT‑28K 与 Reagent‑RRM‑RL‑90K（分别用于 Reward‑Model 训练）；并在 12 个公开基准（GAIA、WebWalkerQA、HLE、xbench、HotpotQA、2Wiki、Bamboogle、MuSiQue、AIME24、AIME25、MATH500、GSM8K）进行评测。

**📈 对比分析**

与开源与专有基线（如 Qwen‑2.5‑14B、DeepSeek‑R1、Atom‑Searcher 等）对比，Reagent‑U 在 12 组基准上均实现显著提升，尤其在 GAIA（43.7%）和 WebWalkerQA（46.2%）取得最高分，整体平均提升幅度超过 10‑20%。

**⚠️ 局限性**

局限性包括：实验仅在 8B 参数规模模型上验证，缺乏对更大规模模型的扩展研究；工具种类与任务复杂度有限，未覆盖更开放的真实场景；Reward‑Model 的批判可能存在偏见，需进一步验证泛化性。

---

## 679. DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation

**arXiv ID:** 2601.22153 | [PDF](https://arxiv.org/pdf/2601.22153v1)

**作者:** Haozhe Xie `[一作]` (Nanyang Technological University), Ziwei Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 43262 | [OpenAlex ID](https://openalex.org/A5100406050)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对动态物体操作的Vision‑Language‑Action（VLA）框架，并构建了首个大规模动态操作基准数据集。

**💡 创新点**

创新点包括：1）采用卷积视觉编码器与轻量化语言模型的紧凑VLA结构，显著降低推理延迟；2）实现连续推理（Continuous Inference），使得推理与执行重叠，消除块间等待；3）引入潜在感知动作流（Latent‑aware Action Streaming），实时丢弃过时动作并优先执行最新预测，从而解决感知–执行时间错位问题；4）开发了自动化的仿真与真实世界数据采集管道，生成20万条合成和2000条真实动态操纵样本。

**🔧 技术方法**

技术主要包括：FastViT卷积视觉编码器、SmolLM2轻量语言模型、基于Flow Matching的扩散动作专家、连续推理流水线、以及潜在感知动作流策略；数据采集利用Isaac Sim仿真、双摄RGB跟踪与实时状态估计。

**📊 数据集**

使用了新构建的Dynamic Manipulation Benchmark（200K合成回合、2K真实回合），包含206种日常物体、2.8K多样场景，速度覆盖0–0.75 m/s；与多个机器人平台（Franka Panda、AgileX PiPER）兼容。

**📈 对比分析**

与多种现有VLA模型（Diffusion Policy、OpenVLA-OFT、SmolVLA、VLASH等）进行对比。实验显示，该框架在交互、感知和泛化三大维度均显著优于基线：成功率提升约+190%（交互）、感知成功率提升至51.9%（vs. 11.7%），并在动态速度、突变和长周期场景中保持稳健。实际任务完成时间和路径长度也均明显缩短。

**⚠️ 局限性**

局限性包括：1）仍受限于短期反应，无法处理更长时域的持续动态任务；2）依赖刚体动力学，对柔性或流体对象适用性有限；3）轻量化设计在极低延迟下可能牺牲部分多模态理解能力；4）真实世界采集仍需精确同步多视角，扩展到更复杂环境仍有挑战。

---

## 680. DynaWeb: Model-Based Reinforcement Learning of Web Agents

**arXiv ID:** 2601.22149 | [PDF](https://arxiv.org/pdf/2601.22149v1)

**作者:** Hang Ding `[一作]` (Shanghai Jiao Tong University), Lei Yu `[通讯]` (University of Toronto)

**通讯引用:** 18913 | [OpenAlex ID](https://openalex.org/A5101937767)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过学习网页世界模型进行想象驱动的模型基强化学习来训练网页代理，避免实时网络交互

**💡 创新点**

将网页世界模型从辅助工具转变为核心学习环境，并将真实专家轨迹与模型生成轨迹混合提升稳定性

**🔧 技术方法**

大型语言模型（LLM）作为网页世界模型与代理策略，DynaWeb框架，GSPO强化学习

**📊 数据集**

StanfordNLP/NNetNav数据集用于训练世界模型，WebArena与WebVoyager基准用于评估

**📈 对比分析**

与多种基线（SFT、Offline‑RL、ITL、GPT‑4o）对比，DynaWeb在WebArena和WebVoyager的成功率均提升约16‑20%，优于其它开源代理

**⚠️ 局限性**

受限于世界模型的仿真误差与较短梦境长度，且在长时延与高度动态UI场景下仍表现不佳

---

## 681. RedSage: A Cybersecurity Generalist LLM

**arXiv ID:** 2601.22159 | [PDF](https://arxiv.org/pdf/2601.22159v1)

**作者:** Naufal Suryanto `[一作]`, Ernesto Damiani `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了开源的8B规模网络安全专用LLM RedSage，并构建了包含连续预训练、Agentic增广SFT和全维度评测基准的完整数据与训练管线；

**💡 创新点**

创新点在于：① 通过CyberFineWeb构建11.7B token的连续预训练语料；② 采用Planner‑Augmenter框架生成26.6K多轮安全对话；③ 设计了涵盖知识、技能、工具的30K题目评测基准RedSage‑Bench；④ 代码、模型与数据完全开源；

**🔧 技术方法**

使用Qwen3‑8B‑Base作为基础模型，进行连续预训练（DeepSpeed ZeRO‑3、AdamW），AgentInstruct式Planner‑Augmenter生成对话，随后用SFT与DPO（Tulu‑3偏好混合）进行后训练；

**📊 数据集**

数据集包括：CyberFineWeb（11.7B token）、RedSage‑Seed（28.6K条目、0.15B token）、RedSage‑Conv（266K对话、352M token）、公开安全问答集（用于构建RedSage‑Bench 30K MCQ+240 OQ）以及多项公开安全基准（SecEval、CyberBench等）与通用基准；

**📈 对比分析**

通过与多款公开安全LLM（Llama‑3.1‑8B、Qwen3‑8B、Foundation‑Sec‑8B、DeepHat、Lily‑Cybersecurity）及通用LLM（如GPT‑5、Qwen3‑32B）在0‑shot/5‑shot、MCQ、OQ、通用基准上对比，RedSage在RedSage‑Bench 0‑shot取得84.5%宏观准确率，在其他安全基准提升3‑5个百分点，在通用基准平均得分达74.3%，仅比32B模型低约1点，远超8B基线；

**⚠️ 局限性**

局限性包括：生成的对话可能携带偏见或事实错误；数据来源多为公开内容，可能涉及版权限制；模型包含攻击信息存在滥用风险；在更大规模模型或更高精度任务上仍有提升空间；

---

## 682. Discovering Hidden Gems in Model Repositories

**arXiv ID:** 2601.22157 | [PDF](https://arxiv.org/pdf/2601.22157v1)

**作者:** Jonathan Kahana `[一作]` (Hebrew University of Jerusalem), Yedid Hoshen `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 1697 | [OpenAlex ID](https://openalex.org/A5047455929)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文研究公共模型仓库中存在的“隐藏宝石”——下载量低但性能显著优于主流基线的模型，并提出高效的发现方法。

**💡 创新点**

创新点在于将模型挑选问题转化为固定预算最佳臂识别的多臂老虎机问题，并在Sequential Halving算法中引入相关采样与快速淘汰策略，实现查询效率提升50倍。

**🔧 技术方法**

核心技术包括多臂老虎机框架、相关采样（共享查询集降低方差）、Aggressive Elimination（首轮快速淘汰）、改进的Sequential Halving搜索。

**📊 数据集**

使用了2000余个模型的评估数据，涵盖Qwen‑3B、Qwen‑7B、Mistral‑7B、Llama‑3.1‑8B四棵模型树，并在RouterBench、MBPP、GSM8K、ARC-Challenge等子集上进行测试。

**📈 对比分析**

与随机、UCB、TTTS、BayesElim等8种基线对比，本文方法在10/50次查询/模型预算下平均排名显著下降（从≈30/90降至≈3/4），top‑1准确率提升至≈0.73‑0.79，明显优于最优基线模型。

**⚠️ 局限性**

局限性包括仍需执行大量查询，依赖于特定任务子集，方法对不同模型树或任务分布的泛化尚未完全验证，且在极低查询预算下性能仍受限。

---

## 683. Do VLMs Perceive or Recall? Probing Visual Perception vs. Memory with Classic Visual Illusions

**arXiv ID:** 2601.22150 | [PDF](https://arxiv.org/pdf/2601.22150v1)

**作者:** Xiaoxiao Sun `[一作]` (Stanford University), Serena Yeung-Levy `[通讯]` (Stanford University)

**通讯引用:** 6961 | [OpenAlex ID](https://openalex.org/A5081511803)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 VI-Probe 框架，用可控的视觉错觉图像和语言提示对大规模视觉语言模型（VLM）进行细粒度探测。

**💡 创新点**

创新点在于：①设计了可调节强度的错觉与匹配控制图像，①通过 Polarity‑Flip Consistency、Template Fixation Index 与 Illusion Multiplier 三种指标同时衡量语言一致性、视觉感知与记忆偏差；②揭示不同 VLM 家族在错觉下表现多元化（记忆覆盖、感知‑记忆竞争、视觉处理瓶颈）。

**🔧 技术方法**

技术包括：视觉生成流水线（基于经典错觉的参数化变形）、多样化语言提示（正反问题、系统指令）、自定义评估指标（PFC、PFA、TFI、R），以及对 15 种 VLM（OpenAI、Anthropic、Google、Qwen 系列）进行零样本推理。

**📊 数据集**

使用 27 种经典视觉错觉（Ebbinghaus、Müller‑Lyer、Checker‑shadow 等）生成原始、变形、控制、提示四种图像，并配合三种问句构成数据集。

**📈 对比分析**

通过与现有错觉基准（HallusionBench、IllusionBench、IllusionVQA、VLMBiased）对比，发现平均准确率往往掩盖模型的“原始→变形”差距；利用 R 归一化显示 GPT‑5 等旗舰模型在错觉下性能骤降（R≈1.97），而 Qwen 系列 R<1，表明其主要受限于视觉处理；提示/系统指令实验验证了模板检索与感知切换的极端效果。

**⚠️ 局限性**

局限性包括：①仅评估了经典二维错觉，缺乏对更复杂场景（图表、医学影像）的验证；②缺乏自监督训练或对抗学习的改进措施；③评估聚焦零样本，未探究 fine‑tune 后的表现；④对不同语言提示的系统性分析仍不完整。

---

