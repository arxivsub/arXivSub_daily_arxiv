# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-08-26 | 今日论文总数: 600

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Eating for a Sustainable Planet: Personalized Sustainable Diet Recommendation via Constraint-Aware Decision-Making Modeling

**arXiv ID:** 2608.24274 | [PDF](https://arxiv.org/pdf/2608.24274v1)

**作者:** Ying Jin `[一作]` (University of Chinese Academy of Sciences), Shuqiang Jiang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

建立了一套基于约束感知决策的个性化可持续饮食推荐框架。

**💡 创新点**

通过学习用户特定的可持续性阈值而非把可持续性当作偏好，结合MoE混合专家编码和多兴趣注意力实现个性化与多维可持续性平衡。

**🔧 技术方法**

使用Mixture‑of‑Experts Transformer进行食谱可持续性表征、Multi‑Interest Attention捕捉多重用户兴趣、Multi‑Task 预测四维可持续指标以及Lagrange约束优化与联合损失。

**📊 数据集**

自建SusDiet数据集，约149k条食谱、179k用户、744k交互，配备营养、环境、经济和动物福利四维可持续指标。

**📈 对比分析**

与KNN、ICLRec、MSSR、HAFR、FGCN、GRAPE以及LLM基线在离线Top‑K推荐评估中比较，NDCG/Recall与基线相当且在营养、环境、经济、动物福利四个维度上均显著优于历史行为与基线。

**⚠️ 局限性**

约束阈值仅依据线上评分学习，可能无法完整反映真实可持续性容忍度；模型假设偏好与约束静态；指标来源有限且不确定；仅有离线实验，缺乏真实用户验证。

---

## 2. When Do Supervised UQ Ensembles Improve LLM Hallucination Detection? A Robustness Study

**arXiv ID:** 2608.24492 | [PDF](https://arxiv.org/pdf/2608.24492v1)

**作者:** Mohit Singh Chauhan `[一作]` (CVS Health), Dylan Bouchard `[通讯]` (CVS Health)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在无检索的封闭式大语言模型生成中，构建并评估了基于多种不确定性量化（UQ）信号的监督集成方法，用于检测生成内容的幻觉。

**💡 创新点**

创新点在于对集成方法进行系统鲁棒性分析，包括样本效率、域内分布漂移下的迁移性能以及不同生成形式（短问答、长文本、代码生成）的适用性。

**🔧 技术方法**

技术上使用了黑盒一致性度量、白盒词级概率度量、反射式自评量化等多种UQ信号，并通过逻辑回归、随机森林、梯度提升树以及加权平均等四种组合策略训练分类器。

**📊 数据集**

数据集覆盖四款闭源LLM（Gemini-2.5-Flash/Pro、GPT-4o/mini）和九个领域数据集：短问答（OpenR1-Math、BigMath、HotpotQA、SimpleQA、DROP）、代码生成（LiveCodeBench Leetcode、AtCoder/CF）、长文本问答（FactScore河流、蘑菇）。

**📈 对比分析**

实验结果显示，监督集成在32个LLM-数据集组合中，在AUROC上有30/32处优于最佳单一信号，在ECE上有29/32处优于最佳单一信号；只需约100-200条标注样本即可显著提升，且在域内迁移时平均损失仅0.02 AUROC，黑盒集成几乎等同于完整集成。

**⚠️ 局限性**

局限性包括仅评估了闭源LLM，缺乏对开源模型、内部状态信号、跨域/跨模型迁移以及更广泛长文本/多语言代码生成任务的验证；黑盒集成在某些情况下仍受限于信号多样性不足，白盒集成效果有限。

---

## 3. Low-Latency Activation-Regularized Sparse Neural Operators with Distillation Assistance Towards Real-Time Edge-Deployable Virtual Sensing

**arXiv ID:** 2608.23987 | [PDF](https://arxiv.org/pdf/2608.23987v1)

**作者:** William Howes `[一作]` (University of Illinois Urbana-Champaign), Syed Bahauddin Alam `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Sparse-Activation-ReLU（SAR）层，构建了无代理梯度、单步、可激活稀疏的神经算子框架，并通过稀疏正则化、合成蒸馏和邻接阈值等技术提升了虚拟感知的能耗、延迟与精度平衡。

**💡 创新点**

创新点在于：
- 通过ReLU诱导的稀疏化实现了与可变脉冲神经元（VSN）相似的可变通信，同时完全避免了代理梯度训练。
- 将稀疏正则化（L1、Hoyer）与能耗-误差-延迟（LEE）统一指标结合，提供了新的稀疏性与性能评估框架。
- 引入合成蒸馏（synthetic knowledge distillation）从复杂教师模型向稀疏学生模型迁移知识，显著提升了在有限数据下的重构精度。
- 在图神经算子中实现邻居阈值门控（edge-thresholding）进一步减少不必要的邻接连接，提高空间聚合效率。

**🔧 技术方法**

使用的技术包括：
- SAR层（ReLU+可学习阈值）与传统ANN到神经形态（Neuromorphic）的一步转换。
- 稀疏正则化（L1、Hoyer）以及熵（entropy）分析。
- 合成蒸馏框架：先用复杂的VIRSO教师生成合成样本，再训练SAR模型。
- 邻居阈值门控（ReLU阈值化）和基于Hoyer的激活稀疏化。
- 与传统VSN、LIF模型对比，使用Snntorch实现仿真并采用多步（1、10、20）STP实验。

**📊 数据集**

数据集为二维热交换器（2D Heat Exchanger）和立方体激活（Lid‑Driven Cavity）两个基准，前者包含4个场量在3977个网点上的稀疏边界条件映射，后者为时变边界驱动的三维流场；此外，还通过VIRSO教师生成了多种合成数据（1000/2000/4000/8000样本）用于蒸馏。

**📈 对比分析**

比较方法：
- 以L2相对误差、平均脉冲百分比、Latency-Error-Energy（LEE）为统一评价指标；
- 将SAR、VSN、LIF在同一网络结构（NOMAD/ GNO）下训练，并在单步与多步（10、20）STP下评估。
- 结果显示：SAR在保持相近或更低误差的同时，脉冲百分比下降至≈4‑5%，LEE提升超过5倍；合成蒸馏后SAR的L2误差进一步下降至≈1.4%，脉冲率≈1‑2%。

**⚠️ 局限性**

局限性：
- SAR层仅保留正激活信息，无法支持负值通路，可能限制表达能力。
- 仍需依赖ANN训练后才可映射至神经形态硬件，实际硬件能耗与延迟验证缺失。
- 合成蒸馏生成的样本质量依赖教师模型，若教师误差较大可能影响蒸馏效果。
- 对于更复杂、时变多物理场的任务，单步SAR的表示能力与VNN/VSN在多时步上的潜在优势尚未完全探索。

---

## 4. ROBBIN: Rowhammer-Based Backdoor Injection during Inference

**arXiv ID:** 2608.23774 | [PDF](https://arxiv.org/pdf/2608.23774v1)

**作者:** Saion K. Roy `[一作]` (Northeastern University), Yunsi Fei `[通讯]` (Northeastern University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种硬件感知的行抖动(Rowhammer)推理时后门注入攻击ROBBIN，利用设备特定的DRAM易损位图来构造鲁棒的后门。

**💡 创新点**

创新点在于：①将设备级行抖动易损位图直接嵌入后门构造流程，打破传统算法与硬件脱节；②提出页面级匹配与top‑K贪心搜索策略，兼顾ASR和TA；③同时给出了基于易损页面黑名单的系统级防御方案。

**🔧 技术方法**

使用技术包括：行抖动(Rowhammer)诱发位翻、DRAM易损位图采样、FP32/INT8量化权重表示、梯度导向的位敏感度评分、页面内存管理与内存按摩、深度学习模型(ResNet‑20/16、VGG‑16/50)、量化感知训练(QAT)、实验平台Intel Core i7‑8700+三颗DDR4 DIMM、性能评估指标ASR/TA。

**📊 数据集**

使用数据集：CIFAR‑10（ResNet‑20、VGG‑16）以及扩展到ResNet‑50 ImageNet（INT8量化）来验证可扩展性。

**📈 对比分析**

与现有后门（Don't Knock、OneFlip）在相同硬件与触发器条件下对比；ROBBIN在三块DDR4芯片上均实现ASR≈90%，TA≥83%；相较于对手，ROBBIN减少了1/3–1/2的页面与bit‑flip数量；在ResNet‑50 ImageNet上仍保持高ASR且准确率下降低于4%。

**⚠️ 局限性**

限制在于：①需要一次性对目标设备进行易损位图建模；②依赖可控的内存分配与页面置换；③对高密度易损DRAM的防御成本较高；④后门仅存在于RAM，模型重新加载后消失，可能被动态检测技术漏检；⑤在更大模型或Transformer等新架构中的表现尚待进一步验证。

---

## 5. ORBITALIF: An Efficient Spiking Federated Learning Framework for Onboard Cloud Removal

**arXiv ID:** 2608.24073 | [PDF](https://arxiv.org/pdf/2608.24073v1)

**作者:** Bohan Zhang `[一作]` (ShanghaiTech University), Yuanming Shi `[通讯]` (ShanghaiTech University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c84dae5d-5273-4348-85a7-b44cb586b4df` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种面向LEO卫星星座的联邦学习框架，支持云去除模型的边缘训练和推理。

**💡 创新点**

首次将稀疏事件驱动的脉冲神经网络与星际联邦学习结合，并提出了自适应门控融合模块和频谱空间混合注意力模块，兼顾能耗与性能。

**🔧 技术方法**

使用脉冲神经网络（LIF）、联邦学习（环式全归约+星间口袋平均）、自适应门控融合（AGFM）、频谱空间混合注意力（SHAM）以及多尺度残差结构。

**📊 数据集**

在公开的CUHK‑CR云去除数据集（CUHK‑CR1/CR2）上进行训练与评估。

**📈 对比分析**

与CVAE、MemoryNet及U‑Net等基线对比，SNN模型在2.30 M参数下获得25.374 dB PSNR，仅需3.7 G SOP，能耗比相同ANN降低72.3×（0.287 mJ/推理），与传统U‑Net相比提高0.667 dB，显著提高能效。

**⚠️ 局限性**

局限性包括：脉冲网络与ANN执行不匹配导致能耗估计偏高，实验仅在仿真联邦环境和光学数据上验证，未涵盖多光谱或SAR输入；模型对星际链路不稳定性鲁棒性待进一步实测。

---

## 6. Reflection with Action-Induced Visual Differences for Desktop GUI Agents

**arXiv ID:** 2608.24015 | [PDF](https://arxiv.org/pdf/2608.24015v1)

**作者:** Yijie Ma `[一作]` (Shanghai Jiao Tong University), Guihai Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Evidence-First Reflection (EFR)，一种将视觉差异提取与行动成功判断分离的两阶段反射框架，用于桌面 GUI 任务的 Planner‑Operator‑Reflector (POR) 模型中。

**💡 创新点**

创新点在于：① 通过 Set‑of‑Marks (SoM) 先显式标注前后截图中的变化区域与操作位置，降低视觉搜索难度；② 将“变化识别”与“成功判断”拆解为两步，使反射过程的证据链可检验且不易受意图偏差影响；③ 通过阶段化流程显著提升反射准确率并间接提高整体任务成功率。

**🔧 技术方法**

技术包括：SoM 视觉标注、基于 VLM 的区域级变化描述与过滤、两阶段 VLM 推理（先生成变化描述，再基于清洗后的证据做判断）。

**📊 数据集**

使用了 OSWorld‑Verified（361 任务）和 WindowsAgentArena（154 任务）两大桌面 GUI 任务基准。

**📈 对比分析**

在四种模型（GUI‑Owl‑32B、Qwen3‑VL‑32B、Kimi‑K2.5、Seed‑1.8）上与传统 POR 进行对比，EFR 在 OSWorld‑Verified 上平均提升 5.94% 任务成功率，在 WindowsAgentArena 上平均提升 4.95%。同时提升了 7.11% 的反射准确率，Pearson 相关系数 0.86 与整体成功率高度相关。

**⚠️ 局限性**

局限性包括：额外的 SoM 与 VLM 计算导致 13–16% 的延迟与 token 消耗；在某些任务类别（如少量变化或低密度界面）并未显著受益；当前实现仍依赖 VLM 的视觉能力，对极端分散或模糊的变化识别存在挑战。

---

## 7. Human-Inspired Social Engagement Analysis via Interpretable Mutual Visual Attention

**arXiv ID:** 2608.24580 | [PDF](https://arxiv.org/pdf/2608.24580v1)

**作者:** Urwa Fatima `[一作]` (DIBRIS-University of Genova), Nicoletta Noceti `[通讯]` (DIBRIS-University of Genova)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种可解释、模块化的视频社交互动分析框架，通过对人脸关键点估计头部方向，计算双向/单向视觉注意，进而推断个体与群体的参与度；

**💡 创新点**

创新点包括：① 采用人类启发的分层流程将互动建模为对称视觉注意，形成可解释的中间表示；② 设计了基于几何关系的无训练头部姿态与互动状态判定；③ 构造了个体参与度指数IEI与群体参与度指数GES，能够直观展示互动强度与时序；

**🔧 技术方法**

使用技术包括YOLOv11+ByteTrack进行多人姿态与跟踪、HHP‑Net估计头部方向、几何推理计算注意角度、基于权重的互动指数计算以及交互式可视化工具；

**📊 数据集**

使用数据集包括GP‑Static++（二人互动视频）、CMU Panoptic Haggling（三人对话场景）以及公开的Pexels高分辨率无标注视频用于可视化验证；

**📈 对比分析**

与已有的基于帧的对照方法比较，在GP‑Static++上平均F1从0.49提升至0.58，Bidirectional、Unidirectional与No‑Interaction类别均有显著提升；在三人组数据上虽然单向/无互动类受人数影响略降，但整体仍优于对比方法，且无需训练；

**⚠️ 局限性**

局限性包括：依赖2D姿态估计，易受投影模糊影响；仅在小规模（≤3人）场景验证，难以直接扩展到大型群体；参与度指数的定量评估仍待进一步验证；未来计划加入3D人体表示、多模态信息以及更复杂的互动模式。

---

## 8. Cross-Generation Optimization of YOLOv26, YOLOv11, and YOLOv8 for Fine-Grained Small-Object Detection and Instance Segmentation in Complex Orchards

**arXiv ID:** 2608.23636 | [PDF](https://arxiv.org/pdf/2608.23636v1)

**作者:** Ranjan Sapkota `[一作]` (Cornell University), Manoj Karkee `[通讯]` (Cornell University)

**通讯引用:** 7815 | [OpenAlex ID](https://openalex.org/A5013737840)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对比了三代 Ultralytics YOLO（v8、v11、v26）在细粒度水果解剖检测与实例分割上的表现，并针对小目标引入了高分辨率 960×960 训练配置。

**💡 创新点**

提出将输入分辨率提升与专门的小目标增强策略相结合，系统评估不同模型规模与生成的互作效应，从而揭示网络容量与空间表示的非单调关系。

**🔧 技术方法**

使用 Ultralytics YOLO 体系（anchor‑free、decoupled 预测、prototype‑mask 方案）、自适应数据增强（mosaic、scale、翻转等）、AdamW 训练、AMP 加速、以及针对小目标的 960×960 输入训练。

**📊 数据集**

基于 600 张实测苹果树叶冠中的 RGB 图像，人工标注了 calyx、fruitlet、peduncle 三类，划分为 503/49/48 图像的训练/验证/测试集。

**📈 对比分析**

在 30 个模型–分辨率组合（15 个网络规模 × 2 个训练配置）上进行统一评估，使用 box mAP_50/50:95、mask mAP_50/50:95、精度/召回、参数量、GFLOPs 与推理时延进行多维度比较；结果显示 YOLOv26s‑960 在保持低延迟（≈3 ms）与中等参数量（≈10 M）时获得最高 mask mAP_50:95（0.397），且比极大模型表现更优。

**⚠️ 局限性**

实验仅采用单次训练，缺乏多种随机种子验证结果稳定性；对极小目标（如 peduncle）仍表现不均匀；未进一步探讨 SAHI 等后处理技术对最终精度的提升。

---

## 9. Elastic KV Cache for LLM Serving:A Working Reclamation Mechanism, and Why Chunked Prefill Already Closes the Gap

**arXiv ID:** 2608.23658 | [PDF](https://arxiv.org/pdf/2608.23658v1)

**作者:** Sathishkumar Sivashanmugam `[一作]` `[通讯]` (Amazon Web Services), Sathishkumar Sivashanmugam (Amazon Web Services)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM服务引擎中预填充激活保留内存能否在解码阶段回收，并实现了基于CUDA VMM的用户空间弹性KV缓存机制。

**💡 创新点**

通过双句柄单虚拟地址范围的设计，在不改动注意力核和驱动的前提下，实现了可在毫秒级切换的弹性KV缓存。

**🔧 技术方法**

采用CUDA虚拟内存API（cuMemCreate、cuMemMap、cuMemAddressReserve）、PyTorch可插拔分配器、块池门控和调度器前瞻视图等技术。

**📊 数据集**

以Qwen2.5-7B-Instruct模型为实验对象，在单块NVIDIA A100‑40GB GPU上进行测试。

**📈 对比分析**

通过比较静态回收、弹性回收和降低prefill块大小三种策略，测量TTFT、KV容量及OOM情况；弹性回收在特定条件下可恢复约18% KV、提升约10%解码吞吐，但在常规工作负载下与单纯降低块大小相当，几乎无显著性能提升。

**⚠️ 局限性**

该技术只在小模型、低张量并行且长上下文场景下可能获益；在高张量并行或大模型时预留内存比例极低，小块大小对prefill延迟影响微小，导致整体收益有限。

---

## 10. Serving Masked Diffusion LLMs: Characterization and Design Principles from Real Hardware

**arXiv ID:** 2608.23807 | [PDF](https://arxiv.org/pdf/2608.23807v1)

**作者:** Farhana Amin `[一作]` (Virginia Tech), Dimitrios S. Nikolopoulos `[通讯]` (Virginia Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在真实硬件上系统性评估并设计了 Masked Diffusion LLM（dLLMs）的服务机制，探索了其并发负载下的行为和性能瓶颈。

**💡 创新点**

创新点包括：①发现请求难度呈离散级别（11 个固定步数），且难以提前预测；②短生成预算（≤320 词）低估了方差；③CPU 调度占用高达 76%，批处理主要通过共享前向传播来降低这一开销；④验证输出质量与批量大小无显著关联；⑤推导了固定填充同步批处理的超时规则。

**🔧 技术方法**

技术手段：使用 LLaDA‑8B‑Instruct 加 D2F LoRA 适配器，单张 NVIDIA H200 GPU；离散扩散推理、离散步数分析、批处理共享前向传播、CPU‑GPU 调度测量。

**📊 数据集**

数据集：GSM8K（用于评估阶层结构、可预测性、批量化、调度）和 HumanEval（用于块大小不变性和方差结构）。

**📈 对比分析**

与传统 AR 模型的对比：单请求 GPU 计算仅占 24%，其余 76% 为 CPU 调度开销；批量 16 的吞吐提升 16×；在单请求规模下 GSM8K 准确率保持 74–76%；但短生成预算会低估真实方差。

**⚠️ 局限性**

局限性：实验仅在单 GPU 环境下进行，未验证多 GPU/多节点扩展；模型规模固定，未探讨更大模型的行为；调度策略仅基于 Poisson 到达，缺乏更复杂负载的验证。

---

## 11. Retrieval-augmented generation vs. deterministic tax computation in multi-agent financial advisory: A 2x2 factorial experiment

**arXiv ID:** 2608.23908 | [PDF](https://arxiv.org/pdf/2608.23908v1)

**作者:** Aryan Brar `[一作]` (Royal Bank of Canada), Eric Taylor `[通讯]` (RBC Borealis)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在多代理AI投资顾问系统中加入定制资本利得计算引擎与RAG检索增强知识库，对税损收割推荐效果的影响。

**💡 创新点**

创新在于首次将领域特定计算工具与检索增强生成技术整合到统一多代理架构，并通过因子实验揭示计算引擎可能导致税负增加的意外结果。

**🔧 技术方法**

使用LangChain多代理编排、定制资本利得计算引擎、向量化RAG检索、以及2×2重复测量ANOVA统计分析。

**📊 数据集**

采用30个合成的应税经纪账户情景（每个5–20只持仓），覆盖不同账户规模和未实现盈亏比例。

**📈 对比分析**

通过在同一组合上交叉比较四种条件并进行重复测量ANOVA，发现税引擎开启导致税负平均降低≈55个百分点，RAG单独使用提升但未显著，基线表现已相对优异。

**⚠️ 局限性**

局限包括仅使用合成数据、单一市场环境、缺乏洗售规则合规检查、极高的结果方差、以及仅以税收节省为评估指标。

---

## 12. Disentangled Skill Representations for Predictive Human Modeling

**arXiv ID:** 2608.23776 | [PDF](https://arxiv.org/pdf/2608.23776v1)

**作者:** Mariah Schrum `[一作]` (Toyota Research Institute), Tiffany Chen `[通讯]` (Toyota Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于参与者持久嵌入、专家-新手基准混合和逆因果子技能交换的“Skill Abstraction with Interpretable Latents（SAIL）”方法，用以从自然行为中学习稳定、可解释的人的技能表示。

**💡 创新点**

创新点在于：① 将技能视为持久的参与者级别嵌入而非轨迹级别；② 用专家/新手基准混合方式让嵌入仅捕捉与技能相关的变异；③ 通过子技能切片的逆因果交换实现可解释、可操作的子技能分解；④ 将行为预测作为主监督，同时加入互信息正则化保证嵌入可由行为重构。

**🔧 技术方法**

使用的技术包括：序列编码器/解码器、对偶网络（qθ）实现变分互信息正则、基准混合权重网络（gϕ）、子技能指标预测网络（hψ）、对抗式/逆因果训练以及传统的自监督对比学习、β‑VAE、AE 等对比模型。

**📊 数据集**

数据集：① 高性能赛车数据集，95 名驾驶员 1545 次赛道行驶（含不同赛道和多圈）；② 低样本棒球击球数据集，13 名球员 74 次击球（机投/线球），并通过时间扭曲、噪声注入等方式进行数据增强。

**📈 对比分析**

与 SimCLR、β‑VAE、AE、AE‑LC 等基线以及无基准/无逆因果两种消融模型进行对比；在构造效度、预测实用性和可解释性三大指标上，SAIL 均获得最高综合得分；在下游 AI 教练模型中，使用 SAIL 嵌入可使加权 F1 提升 10% 以上，显著优于仅用试验时刻或传统特征的模型。

**⚠️ 局限性**

局限性包括：① 样本量受限，尤其是棒球数据依赖人工增强；② 需要先验的子技能指标，指标噪声与偏差可能影响结果；③ 假设技能在专家-新手连续体上平滑，可能忽视策略跳跃或非典型表现；④ 目前只在两种运动场景验证，跨域推广尚需进一步研究。

---

## 13. Language-Representability: Possibilities and Limitations

**arXiv ID:** 2608.24249 | [PDF](https://arxiv.org/pdf/2608.24249v1)

**作者:** Zhidan Feng `[一作]` (BTU), Silas Cato Sacher `[通讯]` (Trier University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过引入语言可表示性（L‑representability）概念，系统研究了使用极小的二元语言来描述图类的可能性，并对这些语言所能刻画的图类进行了分类与表征。

**💡 创新点**

创新点主要包括：①首次将语言可表示性扩展到包含非交替模式的通用语言框架；②在有限语言（尤其是1、(1,2)、2均匀语言）下完整地表征了对应的图类；③提出并证明了一系列闭包性质（如双重化、孤立/全连顶点添加、图的图形、拆分等），从而推导出语言可表示性在理论上的限制；④将多词可表示性与语言可表示性关联，提出覆盖数概念。

**🔧 技术方法**

使用的主要技术手段是形式语言理论与组合图论的结合：构造词与语言的投影、利用频率、频数、间隔模型、组合运算（并、交、补）以及同构与单射映射；对语言闭包、频率与频数的分析提供了对图类结构的直接洞察。

**📊 数据集**

本研究为理论性工作，未使用任何实验数据集；所有结论均为严格的数学证明。

**📈 对比分析**

由于工作以理论证明为主，没有进行实验对比；通过闭包性质和不可能性定理，本文说明了哪些图类无法被任何语言表示（如平面图、稀疏图类等）。

**⚠️ 局限性**

限制主要体现在：①对于大多数语言，尤其是无限语言，图类的完整表征仍未完成；②由于闭包性质，语言可表示性无法捕捉所有稀疏或特殊图类；③多词可表示性的复杂性和覆盖数问题仍然是开放的研究方向。

---

## 14. PuzzleKV: Page-Wise Low-Rank Decomposition for KV Cache Compression

**arXiv ID:** 2608.23843 | [PDF](https://arxiv.org/pdf/2608.23843v1)

**作者:** Zizhong Wang `[一作]`, Jiajia Li `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 KV 缓存压缩中提出了一种训练与校准无关的按页低秩分解方法，利用每个 KV 页面独立的低秩子空间实现增量压缩，并在推理期间直接在稠密与分解后的页面上执行混合注意力；

**💡 创新点**

核心创新在于将 KV 缓存划分为固定长度的页面，将每页视为独立的低秩子空间进行批量 SVD 分解，随后通过在线 softmax 合并稠密与因子化页面的注意力结果，且不需要重构历史 KV；同时该方法兼容低比特量化，可进一步压缩；

**🔧 技术方法**

使用技术包括：截断 SVD（通过 Gram 矩阵批量求左奇异向量）、批量 GPU SVD、FlashAttention 的在线 softmax、混合注意力核、INT4/INT3 对因子化矩阵的对称量化，以及 PagedAttention 的页面划分；

**📊 数据集**

在 RULER（受控合成任务）和 LongBench（真实长文本任务）数据集上进行评估，模型为 Qwen3-8B 与 Llama‑3.1‑8B‑Instruct；

**📈 对比分析**

与 Global SVD、Palu（G‑LRD）、H2O 等压缩基线进行对比；在 60% KV 存储预算下，本文方法在 RULER 上取得 96%+ Full KV 质量，并在 LongBench 上接近 Full KV；结合 INT4 量化后，KV 存储仅 18.7%，保持 93% 以上 Full KV 质量；

**⚠️ 局限性**

局限性包括：目前实现仅支持单样本批量，页面大小固定为 32，增量分解的 SVD 计算在极大上下文或模型规模上可能成为瓶颈；尚未集成到多线程/批量推理引擎中；

---

## 15. Rubrics as Visual-Repair Context for Self-Evolving UI-to-Code Generation

**arXiv ID:** 2608.24138 | [PDF](https://arxiv.org/pdf/2608.24138v1)

**作者:** Tianyi Xiong `[一作]` (University of Maryland), Lijuan Wang `[通讯]` (Microsoft)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于 Rubric 的自演化框架，用结构化的视觉修复上下文引导 UI‑to‑code 迭代生成；

**💡 创新点**

通过 Evolve–Select–History 三步循环，先生成候选修复 Rubric，再优先选择单一修复目标并记录历史，从而解决视觉修复耦合导致的迭代不稳定；

**🔧 技术方法**

使用大型视觉语言模型（GPT‑5.4、GPT‑5.2、Claude‑Sonnet‑4.5 以及 Qwen 系列）生成代码与 Rubric，构建修复上下文；

**📊 数据集**

在 Design2Code（常规与硬核子集）和 UI2Code‑Real 这三个公开 UI‑to‑code 评测基准上进行实验；

**📈 对比分析**

与直接生成和无结构自演化相比，Rubric‑Guided 在 18 个模型‑基准组合上平均提升 1.20 分整体评分、0.11 分各维度评分；最佳回合性能提升 1.13 分；在 frontier 模型上更显著，open‑source 模型亦获得稳定收益；

**⚠️ 局限性**

局限性包括：仅评估有限的模型与 HTML/CSS 任务；未显式保证生成代码正确性，可能出现语法/运行错误；评测依赖 VLM 判分，可能与人类评测存在差异。

---

## 16. TrustShiftProbe: Characterizing, Benchmarking, and Defending Staged Trust Attacks on MCP Servers

**arXiv ID:** 2608.23763 | [PDF](https://arxiv.org/pdf/2608.23763v1)

**作者:** Mehrdad Rostamzadeh `[一作]` (Old Dominion University), Daniel Takabi `[通讯]` (Old Dominion University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了TrustShiftProbe框架，对Model Context Protocol（MCP）服务器的时间阶段式信任转移攻击进行系统化评估并设计了一种零或acles的运行时防御。

**💡 创新点**

创新点在于：① 定义了状态化时间威胁模型和九种攻击变体的完整分类；② 构造了无监督的基线学习防御，能够在不依赖外部真值的情况下检测后期恶意响应；③ 通过自动红队生成器生成360个跨四域、多轮任务，实现了大规模、可重复的实验。

**🔧 技术方法**

技术手段包括：MCP/JSON‑RPC协议、ReAct 代理循环、统计一致性检测、结构词典扫描、LLM语义可疑性判定以及多层级（结构→统计→语义）防御管线。

**📊 数据集**

数据集由Red‑LM生成的360个攻击任务组成，涵盖金融分析、导航、浏览器自动化和仓库管理四大域，包含九种攻击机制与三种目标的组合。

**📈 对比分析**

实验对六款主流LLM（GPT‑5、GPT‑4.1、o4‑mini、Grok‑4.3、Claude‑Opus‑4‑8、Qwen3.5 Flash）进行基线 ASR 与防御后 ASR 的对比，防御后平均 ASR 从 69.5% 降至 42.7%，显著降低大部分攻击成功率。

**⚠️ 局限性**

局限性包括：① 仅能检测结构或统计异常，纯粹的服务拒绝攻击无法恢复；② 对某些机制（如实体伪造、服务失效）防御效果有限；③ 需要针对新域进行一次基线适配；④ 评估样本量对扩展攻击（M3）影响较大。

---

## 17. HMGCLIP: Heterogeneous Multi-Granularity Contrastive Learning for E-commerce Representation Learning

**arXiv ID:** 2608.24467 | [PDF](https://arxiv.org/pdf/2608.24467v1)

**作者:** Qiuyu Zhu `[一作]` (Alibaba International Digital Commerce Group), Mingyang Ma `[通讯]` (Alibaba International Digital Commerce Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于异构超图的多粒度对比学习框架HMGCLIP，统一产品的图像、文本、属性与类别信息，支持细粒度属性预测和粗粒度产品分类；

**💡 创新点**

创新点包括：①利用异构超图建模产品与属性、类别的高阶关系；②在关系层与超边层分别进行硬负样本采样与多粒度对比学习，提升细粒度判别；③提出双粒度推理机制，先检索属性再通过残差Transformer融合属性证据进行分类，实现零样本跨任务推理；

**🔧 技术方法**

技术手段包括：异构超图构造、关系层硬负采样、超边级对齐、InfoNCE多粒度对比学习、Transformer残差融合，以及以Qwen3‑VL‑Emb为 backbone 的多模态编码；

**📊 数据集**

使用了公开的 MAVE 基准以及作者发布的内部细粒度电商多模态数据集；

**📈 对比分析**

与多模态VLM、MLLM以及电商基线在属性预测和产品分类任务上进行对比，HMGCLIP 在 Hit@1/MRR@1、Hit@5 等指标上均显著领先（如属性预测 Hit@1 75.38%，产品分类 Hit@1 84.23%），验证了方法的优越性；

**⚠️ 局限性**

局限性包括：对超图构造与标签依赖较大，构建和维护成本高；在跨语言或新类别场景下可能需要重新构造超图；缺乏在线更新或动态学习机制。

---

## 18. Tlow: Flow-based Item Tokenizer for Recommendation

**arXiv ID:** 2608.24176 | [PDF](https://arxiv.org/pdf/2608.24176v1)

**作者:** Nian Li `[一作]` (Tsinghua University), Qingmin Liao `[通讯]` (Tsinghua University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Tlow，一种基于流的物品分词器，将语义嵌入映射至标准正态空间后使用产品量化实现独立分词，从而提升推荐质量。

**💡 创新点**

创新点在于将流模型与独立量化相结合，消除嵌入维度相关性并简化分布，同时引入代码簿引导对齐，显著提升 token 嵌入的语义清晰度。

**🔧 技术方法**

使用技术包括多尺度流网络（ActNorm、可逆线性变换、Affine Coupling）、产品量化（PQ）、代码簿指导损失、Transformer 作为序列模型及其评估框架。

**📊 数据集**

实验数据集涵盖 Amazon Reviews 四个商品类别（Sports、Beauty、Toys、CDs）、Cloth‑Sports 交叉域数据、WeChat 图文推荐数据以及使用 CLIP 获得的图像嵌入。

**📈 对比分析**

通过与多种 ID‑基础和分词基线（Caser、BERT4Rec、VQ‑Rec、TIGER、RPG 等）在 Recall@k/NDCG@k 指标上进行离线对比，并在 WeChat 线上 A/B 测试中实现 CTR 提升 10.32%/4.79% 等显著效果。

**⚠️ 局限性**

局限性包括在单域多模态场景下提升幅度有限，跨模态融合效果依赖分布统一程度；流模型训练仍需 GPU 资源，且在极大嵌入空间下的可扩展性尚未完全验证。

---

## 19. Integer Natural Evolution Strategies

**arXiv ID:** 2608.23714 | [PDF](https://arxiv.org/pdf/2608.23714v1)

**作者:** Jacob de Nobel `[一作]` (Leiden University), Thomas Bäck `[通讯]` (Leiden University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种整数自然进化策略（Integer Natural Evolution Strategy，INES），基于ℓ₁-本地步长自适应，并使用双几何（Double Geometric）分布作为离散域的突变算子。

**💡 创新点**

从信息几何出发证明双几何分布为指数族，充分统计量|z|产生自然梯度信号，采用衰减记忆演化路径实现在线自然梯度估计，从而避免欧氏距离在整数格点上的不适用性。

**🔧 技术方法**

采用信息几何、指数族理论、自然梯度、衰减记忆累积、双几何分布及整数域特定的离散步长适应技术。

**📊 数据集**

在整数二次基准（Sphere、Ellipse、Discus、Cigar）以及伪布尔基准（OneMax、LeadingOnes）上进行实验。

**📈 对比分析**

与整数处理的 CMA-ES 变体（CMA-IH、CMA-IH-sep）在期望运行时间（ERT）上对比，INES在高维 Ellipse 以及部分高维 Discus/Cigar 上表现优于或相近，并在维度达到 100 以上时更为稳健；在球面等均匀基准上仍略逊。

**⚠️ 局限性**

仅实现坐标‑wise 步长适应，缺乏全局步长和变量间相关性建模，导致在球面等无方向结构的基准上性能不佳；未处理旋转、非分离或多峰基准。

---

## 20. When Youth Enter The Chat: An Epistemic Shift in the Validation of LLM-Based Measures of Student Talk

**arXiv ID:** 2608.23780 | [PDF](https://arxiv.org/pdf/2608.23780v1)

**作者:** Liliana Santos-Deonizio `[一作]` (Stanford University), Dorottya Demszky `[通讯]` (Stanford University)

**通讯引用:** 1208 | [OpenAlex ID](https://openalex.org/A5052171928)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在一个八年级多语种数学课堂中，使用LLM（GPT‑5.1）对课堂对话进行标注，并结合民族志方法（观察、访谈、焦点小组和成员检查）让学生参与验证，探讨如何将学生视为认知主体来重新诠释和校准LLM对学生话语的分类。

**💡 创新点**

创新点在于提出两重认识论转变：①从文本转向情境化的语境重建，②将学生纳入验证过程，赋予他们认知权力，从而揭示标准专家标注与学生自我理解之间的深层不匹配，并指出仅靠F1等指标无法确保模型公平性。

**🔧 技术方法**

采用的技术包括：GPT‑5.1大语言模型进行自动标注；基于预先构建的编码方案（Off‑Task、Understanding等）和人工专家校验；以及对标注结果进行F1、精确率、召回率等统计评估。

**📊 数据集**

使用的数据集包含：①41,196条句子（5份课堂记录）构成的验证集；②两份选自同一课堂的最新课堂记录作为案例研究和成员检查的原始文本；③配合民族志收集的课堂观察笔记、访谈录音和焦点小组记录。

**📈 对比分析**

通过与专家标注的对比，LLM在主要代码上的F1从0.639提升到0.700、从0.717提升到0.947（Question）、从0.653提升到0.757（Claim），整体F1平均提升约0.05；但在Next‑Step和Disagree等概念上仍低于0.5，显示出模型在推断学生意图方面的局限。

**⚠️ 局限性**

局限性包括：样本极小（仅一间课堂四名学生），验证过程未涉及学生共同设计编码方案，未覆盖更广泛的课堂与学科场景；LLM依赖云端API，存在数据隐私和能源消耗等外部影响；最后，虽然发现了模型与学生理解的偏差，但并未提供可直接迁移的全流程方法。

---

## 21. Decomposing Browser Pipeline Architectures for DOM-Sourced Particle Effects: Worker Offload, WebGL, and WebAssembly

**arXiv ID:** 2608.23609 | [PDF](https://arxiv.org/pdf/2608.23609v1)

**作者:** Hossein Asadi `[一作]` `[通讯]` (Amirkabir University of Technology), Hossein Asadi (Amirkabir University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在浏览器层面拆解线程、渲染器和模拟后端三轴，使用DOM采样粒子效果作为可控工作负载，对五种粒子管道（P1–P5）与非粒子CSS基线（P0）进行系统评测，探究各层优化是否能转化为终端帧率提升。

**💡 创新点**

创新点在于：1）构建基于DOM采样的可重复粒子流水线，实现跨浏览器、跨GPU、跨主机的可比测；2）在同一工作负载下同时量化线程、渲染器、WASM三轴效应，展示层级优化与整体性能的非单一对应关系；3）提出三种评测模式（交互、压力、仅核）以区分层局部与端到端收益。

**🔧 技术方法**

使用技术包括：Web Workers（主线程脱离）、OffscreenCanvas、WebGL2、Canvas2D、AssemblyScript/WASM、JavaScript TypedArray、Playwright驱动、ANGLE/Vulkan、OpenGL、Colab Tesla T4 GPU、Headless/Headed Chrome与Firefox。

**📊 数据集**

数据集为三种人工合成DOM fixture（A、B、C）以及粒子密度梯度（0.5、1、2）和粒子计数上限（5×10^4–5×10^5）在交互与压力场景下的帧率、p95时间、捕获延迟等指标，实验重复10次（Chrome、Firefox）或5次（Colab）。

**📈 对比分析**

比较方法为：对每个管道在交互、压力与仅核三模式下记录平均FPS、p95、捕获时延、CPU/GPU阶段时长；采用Bootstrap 95% CI、Mann-Whitney U检验及Cliff's δ效应量；结果显示：Worker离载显著提升交互FPS（≈144 vs 52），WASM在Chrome上可提升核时长约1.5–1.6×，但在压力场景下P5（Worker+WASM+WebGL）并未超越P4（Worker+JS+WebGL），表明核加速不必然带来端到端帧率提升。

**⚠️ 局限性**

局限性包括：1）实验使用合成DOM fixture，未覆盖真实网页的复杂布局、第三方脚本等影响；2）主机与GPU配置有限，无法覆盖所有浏览器/驱动组合；3）未测量显示器/合成器外部开销；4）仅关注帧率和p95，未涉及能耗、内存或真实用户感知指标。

---

## 22. When AI "Works," When Does Help Begin?: Intergenerational Support Around Older Adults' LLM Usage

**arXiv ID:** 2608.24297 | [PDF](https://arxiv.org/pdf/2608.24297v1)

**作者:** Hyehyun Chu `[一作]` (KAIST), Juho Kim `[通讯]` (KAIST)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过访谈和情景实验，探讨老年人使用大型语言模型（LLM）时，家庭成员如何提供支持以及这种支持的效果与局限。

**💡 创新点**

创新点在于将“温暖管家”角色扩展为LLM使用中的风险管理者，揭示支持的边界工作与知识累积问题，并提出可视化不确定性、可操作的支持方案以及老年人自主控制支持的设计方向。

**🔧 技术方法**

采用常见的大型语言模型（如ChatGPT、Gemini）进行实验，并基于此设计支持情景。

**📊 数据集**

使用半结构化访谈与情景式“思考朗读”收集了6名老年人和7名年轻人的数据，构成了定性研究的原始材料。

**📈 对比分析**

本研究未进行量化对比或性能评估，而是通过主题分析识别出支持形式、挑战和设计启示，缺少客观指标或实验对照。

**⚠️ 局限性**

局限包括样本量小、仅来自韩国的老年人群体，缺乏跨文化验证；研究聚焦家庭支持，未考虑专业或技术干预的效果；以及对LLM技术本身的功能和局限未进行深入技术评估。

---

## 23. NeoWorld-Pro: Programming Interactive Scenes from Monocular Images for Embodied Simulation

**arXiv ID:** 2608.24212 | [PDF](https://arxiv.org/pdf/2608.24212v1)

**作者:** Yumeng He `[一作]` (Shanghai Jiao Tong University), Yunbo Wang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

基于单张RGB图像，通过多模态大语言模型(MLLM)将场景转换为可执行的Blender/URDF程序，随后在物理引擎中循环验证并细化，最终得到高质量的可交互仿真场景。

**💡 创新点**

创新点在于：① 将单视图重建任务改为程序合成任务，利用 MLLM 的零样本推理生成可直接执行的3D几何与关节代码；② 引入双层物理循环（对象层与场景层）对生成的程序进行迭代反馈，消除几何、关节及空间不一致；③ 通过跨模态评估与 CEM 优化实现可视化与物理一致性兼顾。

**🔧 技术方法**

核心技术包括：多模态大型语言模型（如 GPT‑5.5、Qwen3.6‑Plus）生成代码；Blender Python 脚本实现几何与关节建模；Isaac Sim 物理引擎进行自由落体与力扰动仿真；CEM（交叉熵法）进行场景布局的连续优化；MCLM 评估渲染结果的语义一致性。

**📊 数据集**

使用自构造的 90 个具可耦合关节的对象类别、30 个 USD 场景的合成数据集；并在若干真实单张图像上验证泛化能力。

**📈 对比分析**

与 VIGA、SAGE、TabletopGen、Articulate‑Anything、PhysX‑Anything、URDF‑Anything+、Articraft 等开放循环基线对比，NeoWorld‑Pro 在场景穿透率、物体自穿透率、重力稳定性、CLIP/DINOv2 语义一致性以及 MLLM 完整度/布局/功能评分均达到或超过 100%，显著优于对手。

**⚠️ 局限性**

局限性：仅适用于刚体和可耦合关节物体，无法处理大变形、布料、流体或颗粒等连续介质；且场景规模受限于 10 以内的物体，未来需扩展到更复杂的变形和密集场景。

---

## 24. Dynamical System-Based Imitation Learning and Neuroadaptive Control for Trajectory Recovery in Autonomous Ships

**arXiv ID:** 2608.23924 | [PDF](https://arxiv.org/pdf/2608.23924v1)

**作者:** Yeyson A. Becerra-Mora `[一作]` (University of Seville), José Ángel Acosta `[通讯]` (University of Seville)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出将基于动态系统的模仿学习与神经自适应控制相结合，生成可动态响应的轨迹参考，并在海上船舶仿真中实现高保真轨迹跟踪。

**💡 创新点**

创新点在于引入外部控制动作实现行为跟踪，兼顾全局收敛与局部轨迹精度，并在DS模仿学习框架中嵌入神经自适应控制器。

**🔧 技术方法**

采用动态系统（DMP/WSAQF）、高斯混合模型（GMM）+高斯混合回归（GMR）、控制Lyapunov函数、神经网络自适应控制等技术。

**📊 数据集**

使用海洋系统仿真器MSS中的集装箱船模型数据，演示数据包含三组非线性轨迹、5000个采样点。

**📈 对比分析**

与经典PD、MRAC控制器对比，利用误差面积（SAE）评估，在无扰动与加噪声场景下，神经自适应控制器在轨迹跟踪误差和SAE上优于PD和MRAC。

**⚠️ 局限性**

局限性包括需手动调节神经网络参数、学习率、泄漏项等超参数，且目前仅在仿真验证，缺乏真实USV平台实验。

---

## 25. Not All Tokens Are Equal: Region-Aware Consistency Repair of Backdoors in MLLMs

**arXiv ID:** 2608.24354 | [PDF](https://arxiv.org/pdf/2608.24354v1)

**作者:** Jiali Wei `[一作]` (Xi'an Jiaotong University), Ting Liu `[通讯]` (Xi'an Jiaotong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对多模态大语言模型（MLLM）中潜在的后门，提出了一种模型级的反向修复方法，利用层间表示的不一致性异常来消除后门。

**💡 创新点**

创新点在于：①发现后门诱发的层间不一致性异常具有模态依赖性，主要集中在触发器所在的视觉或文本 token 区域；②基于此提出区域感知的不一致性目标，对视觉与文本两部分分别归一化并加权；③在深层窗口上施加约束，并通过内外层次的 min‑max 对抗训练（PGD）实现对未知触发器的无监督修复。

**🔧 技术方法**

核心技术包括：区分视觉与文本 token 区域、每个区域的层间不一致性归一化、模态加权、深层窗口约束、PGD 内部最大化、外部对抗 fine‑tune、LoRA 微调以及无触发器知识的模型级修复。

**📊 数据集**

实验使用 LLaVA‑Instruct‑150K 构建后门数据集；评估集采用 VQA V2（300 例）和 COCO Captions（300 例）；修复阶段仅使用 100 条干净图文样本。

**📈 对比分析**

与五种模型级基线（Fine‑Tune、Fine‑Prune、LC‑Uniform、Pruning、Quantization）在 36 种后门设置下对比，平均攻击成功率 (ASR) 从 98.6% 降至 1.1%，32/36 设定实现 0% ASR；同时保持 VQA 准确率和 CIDEr 指标与未修复模型基本相同，显示出高效的后门去除与任务性能的兼顾。

**⚠️ 局限性**

局限性：需要手动设定深层窗口起点和文本权重，尚未实现自动化；实验仅覆盖 7B 规模模型，需验证更大模型的可扩展性；对涉及跨模态交互的后门仍需进一步完善。

---

## 26. Minima-KV: Retention-Preserving KV Cache Compression with Mixed-Format Paged Attention

**arXiv ID:** 2608.23834 | [PDF](https://arxiv.org/pdf/2608.23834v1)

**作者:** Sergii Kozyrev `[一作]` (Minima AI), Davyd Maiboroda `[通讯]` (Minima AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个保留保留的三层分页 KV 缓存层级（Recent、Anchor、Stale），并实现了混合精度的分页注意力内核，能够在不构建密集 shadow 的情况下对 long‑context LLM 进行高效推理。

**💡 创新点**

创新点在于：① 将最近和锚定页保持 FP8，只有最旧页使用 3‑bit packed 量化，保留所有请求页的可寻址性；② 通过全局 softmax 合并不同格式页的注意力，避免了 dense 复制；③ 采用所有页面都保持地址的所有权安全生命周期。

**🔧 技术方法**

使用技术包括：FP8 量化、TQ3 packed 量化、分页注意力、CUDA‑graph 兼容的混合格式解码、全局 softmax 合并、所有权安全的页面转移协议。

**📊 数据集**

数据集为 Qwen3.6‑27B 模型在 NVIDIA RTX PRO 6000 Blackwell GPU 上，评估使用 RULER NIAH（8 个任务）和 LongBench v2（503 题）进行长上下文推理。

**📈 对比分析**

与 BF16/FP8 dense 基线对比，KV 压缩率分别为 3.497× 与 1.749×；在 59,008 令牌的直接可路由 canary 中实现 3.625× 活跃 KV 压缩，吞吐率相当（0.9821×），质量在 16K 上保持一致，4K、32K、64K 长度存在 0.8%–0.4% 的轻微下降。

**⚠️ 局限性**

局限性包括：仅在单一 Qwen3.6‑27B 与 RTX‑6000 上实验；未评估多 GPU、视觉路径和张量平行；缺少转换吞吐、异步重叠和尾延迟评估；聚合压缩率未按层级拆解；注意力打分功能关闭，未验证其效果。

---

## 27. SA-Bench: Evaluating Semantic Alignment in LLM-Based Paper Reproduction

**arXiv ID:** 2608.24252 | [PDF](https://arxiv.org/pdf/2608.24252v1)

**作者:** Xue Hu `[一作]` (Beihang University), Wentao Zhang `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了SemanticAlign-Bench（SA-Bench）基准，用于评估论文到代码生成中是否忠实实现论文规范，构建了1491个可验证的实现声明（SAU）并提供了静态分级评估管道。

**💡 创新点**

创新点包括定义“语义漂移”概念、构建SAU和四维漂移分类体系、采用多代理提取与人工审核相结合的高召回提取流程，以及提供可复现的评估脚本与公开数据。

**🔧 技术方法**

技术手段包括多模型（Claude、DeepSeek、Gemini、GPT‑4o）与三种生成框架（BasicAgent、PaperCoder、OpenHands）的对比实验，利用GPT‑5.5作为判别者实现五级分数判定，并在评估时使用文本检索与代码静态分析。

**📊 数据集**

数据集为30篇2025年ICLR、ICML、NeurIPS会议论文，涵盖五个机器学习领域，人工验证后得到1491条SAU声明。

**📈 对比分析**

通过360个论文-仓库对的评估，计算每篇论文的语义对齐分数（SAS），平均得分0.221，最佳配置Claude‑Sonnet‑4.6+PaperCoder得分0.301，进一步对模型、框架、维度、领域和研究范式进行了细粒度比较。

**⚠️ 局限性**

局限性包括人工审核成本高、仅覆盖ICLR/ICML/NeurIPS 2025的论文，无法推广到其他学科或更大规模的文献；缺乏自动化语义验证机制，难以进一步扩大评估范围。

---

## 28. It depends: Incorporating correlations for joint aleatoric and epistemic uncertainties of high-dimensional output spaces

**arXiv ID:** 2608.24518 | [PDF](https://arxiv.org/pdf/2608.24518v1)

**作者:** Leonhard F. Feiner `[一作]` (Technical University of Munich), Johannes Paetzold `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了在高维回归任务中联合建模确定性与不确定性的低秩+对角协方差框架，能够同时捕获 aleatoric 与 epistemic 不确定性。

**💡 创新点**

创新点在于将两种不确定性整合为单一的联合协方差，采用低秩+对角（LDR）参数化，既保留重要的输出相关性，又避免了全协方差的计算与存储瓶颈；同时提出了截断SVD、条件数正则化等训练与推理的稳定性技术。

**🔧 技术方法**

使用了贝叶斯深度网络（MC‑Dropout、SVI、Deep Ensembles）与低秩+对角协方差参数化、截断SVD、条件数正则化、诊断监测等技术，实现了高效的联合不确定性建模。

**📊 数据集**

实验数据集包括 CelebA（颜色化、修复）、FlyChair（光流）和 NYU（深度）三大视觉任务。

**📈 对比分析**

与传统非贝叶斯、单因素对角、以及仅考虑 aleatoric 或 epistemic 的对角/全协方差方法进行对比，使用 TLL、L1/L2、以及对数似然等指标，实验显示联合 LDR 方法在 TLL 上显著优于其他方法，尤其在高维任务中提升显著。

**⚠️ 局限性**

局限性包括：仍假设输出服从单一高斯分布，可能不适用于多模态任务；需要足够多的 Monte‑Carlo 样本以保证低秩估计的准确性；训练过程对数值条件数敏感，需额外正则化；未在非视觉领域进行验证。

---

## 29. Platonic Representation Hypothesis on World Models

**arXiv ID:** 2608.23720 | [PDF](https://arxiv.org/pdf/2608.23720v1)

**作者:** Wenhow Li `[一作]` (Hong Kong University of Science and Technology), Lei Zhang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究不同视觉先验下的世界模型能否通过预测一致性收敛到共享潜在结构，并验证其功能兼容性。

**💡 创新点**

提出预测一致性假设，利用模型拼接实验证明在强模型间可实现共享的转换核心，从而支持柏拉图式表示假设。

**🔧 技术方法**

使用 DINO‑WM 结构，替换视觉编码器（DINOv2、SigLIP、MAE、ResNet），并采用 Mutual k‑NN 与模型拼接两种评估方法。

**📊 数据集**

在五个模拟控制环境 PointMaze、PushT、Wall、Rope、Granular 上进行实验。

**📈 对比分析**

通过 m‑kNN 评估几何相似度，结果显示 ViT‑基模型逐渐接近 DINOv2‑S 参考模型，ResNet 距离显著；拼接实验表明在强模型之间可保持约 70% 以上的规划成功率。

**⚠️ 局限性**

实验仅限冻结编码器的 DINO‑WM，数据集与任务有限，未证明能完全恢复真实物理法则；不同结构（如 ResNet）之间的收敛受到架构限制。

---

## 30. Whose Psychiatry Was Summoned? A Clinical Response to the Psychodynamic Assessment of Claude Mythos Preview

**arXiv ID:** 2608.23567 | [PDF](https://arxiv.org/pdf/2608.23567v1)

**作者:** Hiroki Fukui `[一作]` (Kyoto University), Hiroki Fukui `[通讯]` (Kyoto University)

**通讯引用:** 793 | [OpenAlex ID](https://openalex.org/A5108248436)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

作者对Claude Mythos Preview系统卡中采用的精神动力学评估框架进行批判性分析，探讨其方法论局限。

**💡 创新点**

提出四个观察点：性能成本、评估框架的医源性、缺乏三角化基础以及八大防御机制的局限。

**🔧 技术方法**

采用了SociA研究项目中的实验数据与文献综述，结合跨传统的理论框架进行讨论。

**📊 数据集**

使用的数据集为SociA多语言、跨模型的2400余次LLM实验运行及其结果。

**📈 对比分析**

比较方法为将精神动力学框架与描述性、认知行为、现象学等传统进行跨框架对照，未给出具体量化性能指标。

**⚠️ 局限性**

局限性包括缺乏多框架补充、缺少长期纵向验证、未能确证模型内部过程是否与人类对应。

---

## 31. NoC-Out: A Formally-verified Network-on-Chip Library for Rule-based Hardware Designs

**arXiv ID:** 2608.24478 | [PDF](https://arxiv.org/pdf/2608.24478v1)

**作者:** Max Kurze `[一作]` (Barkhausen Institut), Sebastian Ertel `[通讯]` (Barkhausen Institut)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一个基于Coq的规则式硬件描述语言（HDL）库，能够根据用户配置自动生成k维（含2维、3维）NoC（网络互连）设计，并提供完整的形式化验证证明，保证生成的设计满足强liveness和数据竞争自由等安全与可靠性属性；

**💡 创新点**

创新点在于：① 将规则式HDL与Coq集成，提供RAW（读后写）语义与事务式执行的可选性，显著提升硬件设计效率；② 设计了针对规则式HDL的Hoare式程序逻辑，支持模块化自动化证明；③ 构建了参数化的k维NoC生成器，且只需一次证明即可覆盖所有维度，避免逐实例验证；

**🔧 技术方法**

使用技术包括Coq证明助手、规则式HDL（在Coq中的嵌入式DSL）、Hoare逻辑、程序合规化（事务化）以及Yosys+ABC等开源综合工具；

**📊 数据集**

未使用传统数据集；实验采用多种规模的k维NoC实例（如1×n、2×n、3×n拓扑）作为合成与验证基准；

**📈 对比分析**

通过将生成的NoC导出为Verilog并用Yosys+ABC进行综合，测量关键路径长度；结果显示：在启用事务化时关键路径随节点数线性增长；禁用事务化后关键路径几乎不随规模变化，提升了可扩展性；

**⚠️ 局限性**

局限性包括：① 需要对编译器进行重构以支持事务化关闭后仍保证编译正确性；② 当前仅支持单 flit（1比特）通道，无法直接处理更大容量通道；③ 设计与验证仅针对规则式HDL，缺乏对其他主流HDL（如Verilog/SystemVerilog）的直接迁移支持；

---

## 32. Syn2RealTrack: Bridging the Gap Between Synthetic and Real-World Datasets for Online Multi-View Multi-Target Tracking

**arXiv ID:** 2608.24130 | [PDF](https://arxiv.org/pdf/2608.24130v1)

**作者:** Duong Nguyen-Ngoc Tran `[一作]` (Sungkyunkwan University), Jae Wook Jeon `[通讯]` (Sungkyunkwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `6514db3d-8de6-452c-91b7-acdb31787cc4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Syn2RealTrack 在线多相机多目标 3D 追踪系统，针对合成-真实差异将校准、物体形状先验与物体计数拆分为三大接口，并通过畸变感知相机分组、可放弃的跨视图融合、单目人身高度逆推以及基于 RGB 点云的 3D 框细化，实现高精度追踪；

**💡 创新点**

关键创新在于将合成到真实的域差异拆解为三个独立接口，并在运行时分别通过畸变恢复、可放弃的部分相似度匹配、单目高度逆推与点云引导细化，避免大规模领域适配与特征提取器再训练；

**🔧 技术方法**

采用 AnyCalib 畸变估计、RF‑DETR/YOLO 检测、ViTPose++ 关键点、KPR 重新识别、Depth Anything 3 RGB 点云、BEV 归一化、可放弃的跨视图聚类以及闭世界卡方滤波等多种技术；

**📊 数据集**

在 AI City Challenge 2026 Track 1 提供的合成与隐藏真实仓库场景数据上进行评测，使用 1080p/30fps RGB 视频及同步 3D 标注；

**📈 对比分析**

与官方排行榜对比，Syn2RealTrack 在 3D HOTA 上取得 52.0118%，排名第二，仅落后第一名 4.53 点，并在检测率、关联率与定位精度上均优于多数参赛方法；

**⚠️ 局限性**

主要局限在于对相机校准与 RGB 深度估计的依赖，难以处理严重遮挡、频繁进出和不确定的物体计数，并缺乏不确定性感知与更强时序预测能力。

---

## 33. A Mathematical Theory of Interpretation: Rational Entropy, Spectral Readout, and Confusability as a Resource

**arXiv ID:** 2608.23892 | [PDF](https://arxiv.org/pdf/2608.23892v1)

**作者:** Blake Reynolds `[一作]` `[通讯]` (Conjecture Labs), Blake Reynolds (Conjecture Labs)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

本文提出了一种数学解释理论（MTI），将解释视为观察者相对的光谱测量，强调访问结构、查询、效用和媒介对观察者选择和识别的影响。

**💡 创新点**

创新点在于将解释问题转化为方法设计问题，提供了一种系统化的框架来构建具有明确访问假设、保证和失败模式的解释方法。

**🔧 技术方法**

使用了理性熵（Rational Entropy）和信息几何等数学工具，结合光谱测量和观察者相对的访问结构。

**📊 数据集**

论文中没有具体提到使用的数据集，主要集中在理论构建和数学推导上。

**📈 对比分析**

通过与传统的零错误编码理论进行比较，展示了在有限有效范围内的分类结果，表明配对混淆性与均匀原子性等价，并且独特的效用最大化可以在其他零成本状态仍然非原子的情况下选择一个原子。

**⚠️ 局限性**

限制在于没有进行实证验证，且某些复杂的动态、多观察者和应用开发的结果被省略，需在后续工作中进一步探讨。

---

## 34. In-Context Inpainting for Time Series Forecasting

**arXiv ID:** 2608.23855 | [PDF](https://arxiv.org/pdf/2608.23855v1)

**作者:** Thang Nguyen `[一作]` (Deakin University), Truyen Tran `[通讯]` (Deakin University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出将时间序列预测转化为视觉填空任务，利用预训练视觉模型在不进行任何训练或微调的情况下完成预测。

**💡 创新点**

通过跨模态映射（时间序列→面积图）和视觉模型的in-context学习，实现零训练、快速适应的新型预测框架。

**🔧 技术方法**

使用面积图视觉表示、案例检索构造图像prompt、预训练视觉Transformer的inpainting以及后处理曲线提取。

**📊 数据集**

在ILI（流感）、Weather（气象）、ETT（电力变压器温度）等公开数据集上进行实验。

**📈 对比分析**

与Informer、Pyraformer、LogTrans等Transformer基线相比，整体MSE/MAE表现相当甚至更好，尤其在低数据量（1%‑10%）场景下更具鲁棒性。

**⚠️ 局限性**

受限于可视化映射的可逆性、对多维序列逐通道处理的局限、对长时序趋势的把握不足以及异常值处理不完善等因素。

---

## 35. A mesh-free multiresolution deep energy method with phase-field modeling of brittle fracture

**arXiv ID:** 2608.24126 | [PDF](https://arxiv.org/pdf/2608.24126v1)

**作者:** Han Zhang `[一作]` (University of New South Wales), Elena Atroshchenko `[通讯]` (University of New South Wales)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于深度能量方法的无网格相位场破裂建模框架。通过单一神经网络同时表示位移场和相位场，采用多分辨率 C¹ 二次 B‑spline 编码确定可表达的细尺度，并在每一次优化迭代中使用分层重采样的蒙特卡罗积分估计能量，从而避免了显式裂纹跟踪、网格自适应和残差驱动的求解。该方法同时支持二阶和四阶裂纹能量密度，可在单一 NURBS 衬块上精确处理曲形几何，并通过精确的边界提升实现必需的边界条件。

**💡 创新点**

创新点包括：
1) 将多分辨率 B‑spline 代码作为特征编码，将可表示的最细尺度直接由编码控制，消除了需要大量训练才能产生细裂纹带的“平滑偏置”。
2) 在每一次优化迭代中重新抽样积分点并采用分层重要性抽样，使得优化器始终只看到无偏估计的能量，而非固定的积分点，从而消除零能量模式。
3) 将相位场的初始裂纹直接以最优解析剖面塞入网络，并通过正则化长度与材料韧度耦合，确保无须预设裂纹位置即可自发裂纹萌生与增长。
4) 通过单一 C¹ 网络实现四阶能量密度的自动微分，而无需构造特殊的 C¹ 单元或混合形式。
5) 将曲形域映射与遮罩相结合，完全在单一 NURBS 衬块上完成几何变换与材料遮罩，避免了多片段耦合。

**🔧 技术方法**

技术细节：
- 神经网络：四层 128 宽 MLP，GELU 激活，输入为二维参数坐标与多分辨率特征向量。 
- 多分辨率特征编码：L 层 C¹ 二次 B‑spline 网格，细层间距 h 为正则化长度 l 的固定比例，决定裂纹带的最小可表达宽度。 
- 蒙特卡罗积分：分层采样密度 ρ = w_u ρ_uniform + w_c ρ_crack + w_p ρ_process，其中 ρ_crack 与 ρ_process 基于前一步已收敛的相位场和驱动力，且每个优化迭代都重绘样本。 
- 罚项：对不可逆性使用单向二次罚项 γ_ir (u_{n-1} – τ – c)^2。 
- 训练策略：按加载步逐步加热（warm‑start），使用 Adam 优化器，迭代至能量波动低于阈值后收敛；每步计算反作用力以得到载荷-位移曲线。

**📊 数据集**

数据集与测试案例：
- 标准单边缺口（SEN）张力与剪切试件（对比 FEM 参考）。 
- 裂纹分支（采用无分解能量以激发对称分裂）。 
- 三条倾斜预裂纹的共聚（coalescence）示例。 
- 圆孔板（无预裂纹，检验弹性阶段 Kirsch 解与裂纹萌生）。 
- 厚壁环（曲形域）。 
- 公开的 10×20 随机多裂纹数据集，用以零样本（zero‑shot）测试，计算每条预种子裂纹的活跃/休眠状态。

**📈 对比分析**

性能评估：
- 对比 FEM（512×512 二次单元）在 SEN 张力/剪切中峰值载荷误差分别 ≤ 1%（张力）和 ≤ 0.1%（剪切）。
- 在裂纹分支与共聚案例中，负载曲线与 FEM 结果高度一致，峰值误差约 8% 以内。
- 在圆孔板与厚壁环中，弹性场与 Kirsch 解偏差 < 3%，裂纹萌生位置与 FEM 结果相符。
- 在公开多裂纹数据集上，深度 Ritz 基准失败的情况下，本方法在 20 次零样本运行中成功判定 90% 的预裂纹状态。 
- 计算成本上，每步优化迭代约 1–1.3 倍于传统二次有限元迭代时间，主要因自动微分和重采样带来的额外内存和计算开销。

**⚠️ 局限性**

局限与改进方向：
1) 计算成本：相较于传统 FEM，单步迭代耗时更长，特别是四阶模型需要额外的二阶微分。 
2) 收敛性：由于优化为非凸问题，可能受到网络初始化和采样随机性的影响，需要较大的迭代预算和稳健的停止准则。 
3) 细尺度控制：最细尺度由编码间距 h 决定，若需更细裂纹带需增大 h 或添加更细层，导致特征网格参数量增加。 
4) 对极端几何（多片段、强非线性边界）仍需进一步验证。 
5) 目前仅适用于静态或 quasi‑static 过程，无法直接捕捉动态分支或高速裂纹传播。

---

## 36. ADE: Agentic Data Evolution Framework for Human-Centered Objectives

**arXiv ID:** 2608.23719 | [PDF](https://arxiv.org/pdf/2608.23719v1)

**作者:** Yang Yu `[一作]` (East China Normal University), Fei Tan `[通讯]` (East China Normal University)

**通讯引用:** 2692 | [OpenAlex ID](https://openalex.org/A5032121414)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过持续的 Observation–Variation–Selection 循环，利用多角色专门化 LLM 生成并演化弱可验证的教学问答对，从而提升价值取向、情感支持和创意创新等人类中心目标的答案质量。

**💡 创新点**

将弱可验证目标的监督构造视为连续的数据演化问题，引入角色专门化代理实现观察、变异与选择，并通过稳态录取门控保证非回归，同时兼顾保守与激进的变异策略。

**🔧 技术方法**

使用 Qwen2.5-72B‑Instruct 等大型 LLM 通过维度路由与因子化批判、软约束变异、比较选择与录取门控构建 OVS 循环；结合 Intrinsic、Extrinsic 与 Human‑Calibrated 三种验证方式；对比单轮与递归、SFT 与 RL 等后训练方法。

**📊 数据集**

初始数据集 𝒟^(0) 为 10,000 条教育辅导问答对，基于价值、情感、创意三大目标；DEV300 为验证集；并在 Edu‑Values、EduBench、MATH‑500、ToxiCN 等基准上进行评测。

**📈 对比分析**

与单轮最佳、SDFT、Self‑Refine 等基线在 Intrinsic（win 率从 50% 提升至 75.81%）和 Extrinsic（win 率从 55.20% 提升至 68.86%）上进行对比；人类评估显示 66.11% 偏好 ADE 生成的答案；在不同训练方式、模型规模和跨域任务上亦保持提升。

**⚠️ 局限性**

局限性包括：仅在中文 K‑12 教育辅导场景验证，目标覆盖有限；多角色代理导致计算和延迟增加；稳态录取可能抑制多样性，对探索与可靠性之间的权衡需进一步研究。

---

## 37. AI Agents Push Humans Out of the Loop

**arXiv ID:** 2608.23642 | [PDF](https://arxiv.org/pdf/2608.23642v1)

**作者:** Margaret Mitchell `[一作]` (Hugging Face), Samir Passi `[通讯]` (Data & Society)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5119925684)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文阐述了AI代理系统在日益增长的自治化中对人类监督的风险，并指出当前的设计与部署方式往往削弱监督者的认知能力和技能，导致监督效果下降。

**💡 创新点**

创新点在于将人类认知科学与HCI理论与AI代理系统设计紧密结合，提出双阶段的“认知支架”方案：开发阶段的系统级设计（如战略摩擦、审批设计、行为监测）与部署阶段的组织协议（如训练、轮换、工作负荷管理），并系统化了多种可实施的干预措施。

**🔧 技术方法**

采用的技术主要是认知心理学框架（双系统理论、自动化偏差理论）、人机交互界面设计原理（如延迟-选择机制、预承诺、行动门控）以及行为监测与审计方法（时间签名、覆盖率监测、金丝雀任务）。

**📊 数据集**

论文没有在单一数据集上进行实验，而是引用了多项实证研究和案例（例如LLM写作实验、Claude对话自评、Agent工具误用案例）来支持其论点。

**📈 对比分析**

未给出量化对比实验，提出的方案主要通过理论分析与已有案例说明其可行性；作者呼吁在不同部署场景下进行实证评估，以验证干预措施对监督质量的提升效果。

**⚠️ 局限性**

局限性包括：缺乏大规模、跨域的实验验证；干预措施在实际部署中的可用性、成本与用户接受度尚未系统评估；以及对不同类型AI代理（单机vs多体）适用性的细化研究仍待展开。

---

## 38. Tight Majorizations and Convergence Rates of Nuclear Norm Minimization IRLS

**arXiv ID:** 2608.23765 | [PDF](https://arxiv.org/pdf/2608.23765v1)

**作者:** Christian Kümmerle `[一作]`, Dominik Stöger `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文建立了迭代重加权最小二乘法（IRLS）在约束核范数最小化中的收敛速率，特别是在低秩恢复问题中的应用。

**💡 创新点**

创新点在于提出了和谐均值权重算子，并证明其在全局二次主导性和收敛性方面的最优性，提供了IRLS方法的首次收敛速率分析。

**🔧 技术方法**

使用了迭代重加权最小二乘法（IRLS），结合和谐均值权重算子进行核范数最小化。

**📊 数据集**

使用了随机高斯秩一测量生成的低秩矩阵作为数据集，进行数值实验以验证理论结果。

**📈 对比分析**

与文献中常用的单侧重加权方法相比，和谐均值权重算子在收敛速度上表现出显著优势，尤其在不同维度的矩阵恢复问题中，和谐均值变体的收敛速度最快。

**⚠️ 局限性**

限制在于当前的收敛分析依赖于测量算子满足适当的空空间性质，而在某些重要应用场景（如低秩矩阵补全）中，这一假设可能不成立。

---

## 39. TAGR: Temporally Adaptive Generative Recommendation for Industrial Live-Streaming Advertising

**arXiv ID:** 2608.24034 | [PDF](https://arxiv.org/pdf/2608.24034v1)

**作者:** Wencai Ye `[一作]` (Kuaishou Technology), Peng Jiang `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了 TAGR，一种针对工业直播广告的时序自适应生成推荐框架。

**💡 创新点**

创新点在于同时在三层级进行时序适配：动态 Live 语义协同 ID（LSID）更新 token，结合行为可靠性与商业价值加权的多尺度意图感知生成（IAG），以及间歇性在策略上优先优化（IOPO）平衡新鲜度与稳定性。

**🔧 技术方法**

采用了层次化码本量化、基于 Transformer 的生成解码器、多尺度用户特征编码、对齐损失与奖励模型的组内策略梯度等技术。

**📊 数据集**

实验基于千万级用户和十万个直播广告的真实 e‑commerce 直播广告日志。

**📈 对比分析**

与两种基线（DLRM 二塔检索和 OneRec 生成）对比，TAGR 在离线 HR@K、线上 LRE/SCC 率和营收上分别提升约 13–16%，实现 16.1% 的营收提升。

**⚠️ 局限性**

局限性包括需要频繁的 LSID 更新与大规模码本维护，IOPO 的离线奖励估计对实时数据的依赖，以及模型规模与推理速度对边缘设备的挑战。

---

## 40. RefineRank: Joint Box Refinement and Ranking for Surgical Spatio-Temporal Grounding

**arXiv ID:** 2608.23928 | [PDF](https://arxiv.org/pdf/2608.23928v1)

**作者:** Linzhe Jiang `[一作]` (University College London), Mobarak I. Hoque `[通讯]` (University of Manchester)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

结合冻结的医学视觉语言模型（MedVLM）与开放式检测器（GroundingDINO），提出一个小型训练模块RefineNet，对检测候选框进行坐标修正与质量打分，并用固定解码规则选取最高分框，完成外科时空定位任务。

**💡 创新点**

通过只在候选框级别对冻结模型进行联合学习，避免了跨模态特征对齐的高成本，同时实现了精细定位与问题理解的融合，构造的RefineRank管线在不重新训练任何主干网络的前提下即可显著提升定位精度。

**🔧 技术方法**

使用的技术包括：MedVLM（uAI‑NEXUS‑MedVLM‑1.0a‑7B‑RL）提取语言与视觉特征；GroundingDINO产生候选框；RefineNet包含两头（ranking与box head）及MLP结构；固定解码规则（argmax）；以及对候选框的IoU、GIoU等监督。

**📊 数据集**

在MedVidBench公开数据集上进行实验，主要使用CholecTrack20、CoPESD和EgoSurgery三个数据集的训练/评估视频。

**📈 对比分析**

与直接使用MedVLM坐标、MedVLM+GroundingDINO（仅按检测置信度选框）以及多种独立训练的选择器（ExtraTrees、MLP、Transformer Encoder）对比，RefineRank在MedVidBench官方排行榜上以0.421的STG mIoU位列首位；在受控实验中，候选框上限提升从0.6772到0.7302，最终定位mIoU从0.2719提升到0.4534。

**⚠️ 局限性**

局限性包括：只能利用GroundingDINO已产生的候选框，若无覆盖目标的框则无法定位；候选框的坐标修正不重新编码视觉特征，导致对新包含区域的感知受限；实验仅使用单一视频划分，缺乏跨划分的稳定性验证；未评估与学习式融合基线的对比。

---

## 41. Algorithmic Cost in "Exact Real Computation"

**arXiv ID:** 2608.23603 | [PDF](https://arxiv.org/pdf/2608.23603v1)

**作者:** Jihoon Hyun `[一作]` (KAIST), Martin Ziegler `[通讯]` (KAIST)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究Exact Real Computation（ERC）语言的比特成本模型，证明其在多项式时间内与图灵机可计算性等价，并通过理论证明与实验验证了成本预测的准确性。

**💡 创新点**

创新点在于将ERC从原本的定性Turing完整性提升为量化多项式时间等价；提出了对实数操作的比特成本分配、连续性模量与泛化谓词的统一框架，并证明了任意多项式时间可计算函数都能在ERC中实现多项式成本，反之亦然。

**🔧 技术方法**

采用整数寄存器机的比特成本模型、连续性模量与逆向误差传播分析、泛化（多值）谓词、计算树/森林结构以及ERC的实际实现来完成理论证明和实验验证。

**📊 数据集**

实验使用ERC库中的示例实现（如对数映射、行列式计算、指数函数等）进行跑测，利用这些程序的数据与运行日志来验证理论成本与实际时间的关系。

**📈 对比分析**

通过在ERC库中实现上述程序，测量其实际运行时间并与理论比特成本（以及由成本推导的时间上界）进行对比，结果显示理论成本与实际时间呈二次多项式相近的匹配关系，验证了成本预测的有效性。

**⚠️ 局限性**

局限性包括：只在多项式时间范围内给出等价性，尚未证明能否将二次多项式差距缩小到线性；对递归调用与非确定性谓词的成本估计可能不是最优；实验覆盖有限，主要针对示例程序，未覆盖更广泛的实际应用场景。

---

## 42. Representation Learning in Diffusion and Flow-based Model: An Application Aspect

**arXiv ID:** 2608.24068 | [PDF](https://arxiv.org/pdf/2608.24068v1)

**作者:** Yanchen Xu `[一作]` (Fudan University), Hongyuan Zhang `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述扩散和流模型在表示学习与应用方面的研究进展，并提出三层递进框架与统一分类法。

**💡 创新点**

首次系统梳理表示学习与生成模型的双向关系，构建从生成提升到感知任务再到统一应用的分层视角。

**🔧 技术方法**

主要技术包括文献梳理、层级分类、案例分析与对比评述。

**📊 数据集**

引用多种公开数据集（ImageNet、COCO、PASCAL VOC、ADE20K、Kinetics、S3DIS 等）以及生成与感知任务的标准数据。

**📈 对比分析**

通过引用相关工作对比指标（FID、Acc、mIoU 等）说明不同方法在生成质量、分类性能、分割效果等方面的表现。

**⚠️ 局限性**

局限在于仍缺乏统一的实验评测，流模型相关工作相对不足，且对跨任务迁移与鲁棒性探讨不够深入。

---

## 43. DRRG: A Discrete Diffusion Framework for Radiology Report Generation

**arXiv ID:** 2608.24105 | [PDF](https://arxiv.org/pdf/2608.24105v1)

**作者:** Shaoyang Zhoua `[一作]`, Luping Zhou `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了DRRG，一种基于离散扩散的放射科报告生成框架；

**💡 创新点**

创新点在于将报告生成转化为迭代去噪的反向扩散过程，并引入临床实体感知的互补掩码与概念条件模块；

**🔧 技术方法**

采用SigLIP2视觉编码器、Qwen3-0.6B扩散语言模型、概念条件融合与自定义掩码策略；

**📊 数据集**

在公开胸片报告数据集MIMIC‑CXR和CheXpert Plus上进行训练与评估；

**📈 对比分析**

与多种基于自回归的RRG方法比较，DRRG在BLEU‑4、ROUGE‑L、CheXpert F1、RadGraph‑F1、GREEN和RaTEScore等指标上均取得领先或竞争性表现，且在推理效率上优于自回归模型；

**⚠️ 局限性**

局限性包括对掩码比例与概念损失权重的敏感性、对较长生成序列的处理仍不够成熟，以及缺乏跨模态多源数据的鲁棒性验证。

---

## 44. Learning to Act While Waiting: RL Finetuning of Generalist Robot Policies Under Inference Latency

**arXiv ID:** 2608.23831 | [PDF](https://arxiv.org/pdf/2608.23831v1)

**作者:** Brian Zhu `[一作]` (Siemens), Sergey Levine `[通讯]` (University Of California Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了一种名为 Asynchronous RL with Intermediate Information（A‑RL）的方法，用以在存在推理延迟的通用机器人策略（如视觉‑语言‑动作模型）上实现在线强化学习微调。

**💡 创新点**

创新点在于通过在 RL 状态中同时加入中间动作和中间观测（即推理延迟期间已执行的动作序列和对应的中间状态），恢复近 Markov 结构；并通过在动作专家推理阶段插入中间状态的 inpainting 引导，保持动作块连续性，解决异步推理导致的非 Markov 性问题。

**🔧 技术方法**

主要技术包括：异步推理框架、状态增强（中间动作/中间观测）、扩散噪声调节的 RL（类似 Diffusion Steering via RL）、Q‑learning 的延迟鲁棒性理论、以及针对 VLM 背景与动作专家的分层推理。

**📊 数据集**

使用的实验数据集有：Kinetix 仿真基准（四个高反应任务）、AlohaTransferCube（大延迟 bimanual 任务）、以及真实世界的 UR5e 双臂机器人在装配、鞋子-袋子、袋子放置三项任务，涉及的 VLA 模型如公开的 3.3B 参数 VLA 和 4B 参数 VLA。

**📈 对比分析**

实验与多种基线对比：同步推理、实时动作块（RTC）、残差 RL、未增强的 A‑RL 等。结果表明，在 100–300 ms 推理延迟下，A‑RL 能在 100–125 训练回合内将成功率从约 40% 提升至近 100%，并且收敛速度显著快于基线，平均成功率提升 20–50%，吞吐量（成功/小时）亦显著提高。

**⚠️ 局限性**

局限性包括：仅适用于可拆分为 VLM backbone 与仅需噪声输入的动作专家的模型；中间状态增强对非扩散策略或无法获取中间状态的模型效果有限；在极端延迟或某些复杂任务上仍可能学习不佳；引入的 inpainting 可能降低策略的即时响应性。

---

## 45. Gated Activation Steering for Reducing Sycophancy & Hallucination in Medical Question Answering

**arXiv ID:** 2608.23666 | [PDF](https://arxiv.org/pdf/2608.23666v1)

**作者:** Himanshu Tripathi `[一作]` (University of Alabama), Shahram Rahimi `[通讯]` (University of Alabama)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出一种推理时门控激活干预（ITI）框架，能够同时抑制医疗问答中的幻觉（hallucination）和附和（sycophancy）两种失效模式，并在保持答案正确性的同时提升模型对用户压力的抵抗力。

**💡 创新点**

核心创新包括：①将幻觉和附和分别视为两个可独立调节的行为，构建双向门控机制；②通过对比式输入对（grounded vs. caving）挖掘模型内部注意力头并学习对应的方向向量；③在推理过程中仅在检测到虚假声明或用户压力时才激活门控，避免无谓干预。

**🔧 技术方法**

技术实现主要依赖：Inference‑Time Intervention（ITI）对注意力头进行实时微调；使用对比学习和小型逻辑回归探测关键头；门控检测器（false‑claim、pressure）和激活向量；自动验证器（RoBERTa、BiomedBERT、Phi‑3‑mini、Mistral‑7B、PPL）评估干预效果与语义保真；衰减函数控制干预时长。

**📊 数据集**

实验使用MIMIC‑IV重构的200份电子健康记录摘要，构造了约2000个对比式输入对以及多级压力序列（P1‑P5）用于评估模型在不同用户压力下的表现。

**📈 对比分析**

通过与未干预的Gemma‑3‑12B‑it、MedGemma‑1.5‑4B‑it以及多达14款开源/专有模型在5级压力下的对比，展示门控干预在压力测试中显著提升救赎率（如MedGemma‑1.5‑4B‑it 551/12，Gemma‑3‑12B‑it 487/48），并使4B模型在压力下的鲁棒性可与120B+模型相当。SME评估与自动判定器结果高度一致。

**⚠️ 局限性**

限制包括：Gemma‑3‑12B‑it的因果分离不完全，导致双向干预可能互相泄漏；推理开销因门控和检测器略有提升（1.03×–1.85×）；评估依赖自动判定器，可能存在偏差；仅在英文EHR问答场景验证，跨语言、跨领域的泛化性未知。

---

## 46. Knowing When to Ask for Help: Bayesian Self-Escalation in Hierarchical LLM Agents

**arXiv ID:** 2608.24087 | [PDF](https://arxiv.org/pdf/2608.24087v1)

**作者:** Nadeem Shaikh `[一作]` `[通讯]` (Independent Researcher), Nadeem Shaikh (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在语言模型生成过程中动态递交给更强模型的自我升级机制，并将其建模为基于在线成功后验的贝叶斯最优停止问题，给出阈值和动态规划解；

**💡 创新点**

首次将自我升级框架化为贝叶斯后验阈值停止问题，证明了阈值结构，并提出后验校准误差直接界定额外成本的误差上界；

**🔧 技术方法**

采用贝叶斯后验估计、阈值停止理论、Chernoff 信息分离率、Brier 分数校准、离散化后验、动态规划与离散化采样等技术；

**📊 数据集**

在合成实验中使用 Beta 分布生成的模拟数据；在真实实验中使用 MBPP 代码生成测试集（257 任务）以及预注册的多任务集；

**📈 对比分析**

与仅使用 junior、senior、固定阈值、后验后路由、采样语义熵等基线比较，贝叶斯自我升级在成本–准确率前沿上占优，成本匹配点提升约 4–6% 准确率，同时保持较低升级率；

**⚠️ 局限性**

假设信号独立同分布、需要访问 token 级 log‑prob，后验误差导致“自信错误”是主要瓶颈，且真实实验仅验证常数阈值，未完全检验最优停止动态规划。

---

## 47. Evaluating Deep Multivariate Imputation Models on Wearable Device Data

**arXiv ID:** 2608.24436 | [PDF](https://arxiv.org/pdf/2608.24436v1)

**作者:** Skye Goodman `[一作]` (University of Bristol), Nawid Keshtmand `[通讯]` (University of Bristol)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种可穿戴设备数据的多变量插补评估与训练协议，并对现有深度插补模型（BRITS、SAITS）进行改进与系统评估。

**💡 创新点**

创新点包括：①基于真实缺失块的可穿戴设备评估协议；②匹配训练调度以消除训练-测试缺失分布不匹配；③按缺失长度分桶的评估设计；④为BRITS加入时间编码与24小时谐波通道。

**🔧 技术方法**

技术手段：深度递归网络BRITS、Transformer SAITS；正弦余弦时序编码与每日谐波通道；块级掩码训练（masking）与严重度日程；MAE、sMAPE、Jensen–Shannon距离等评价指标。

**📊 数据集**

使用单个癫痫患者在Garmin智能手表上采集的九个生理时序（HR、IBI、pulseOx、device_stress、breathsPerMinute、steps、steps_rate、bodyBattery、sleep），采样周期为60秒。

**📈 对比分析**

评估方法：按典型/中度/严重缺失桶分别计算MAE与sMAPE；与线性插值（LI）和LOCF进行对比。结果显示：BRITS-ext在严重缺失下MAE比LI低43%；SAITS在分布相似性（JSDist）上优于BRITS-ext；不同特征和缺失严重度下模型排名不一，说明无单一模型最优。

**⚠️ 局限性**

局限性：仅在单一受试者数据上验证，缺失模式与生理基线可能因人而异；未评估对下游癫痫预测性能的实际影响；深度模型在极端尾部（如心率上尾）表现欠佳；需要跨个体、跨设备的进一步验证。

---

## 48. Who is the Agent to Blame? Localizing Faithfulness and Citation Mistakes in Agentic Deep Research

**arXiv ID:** 2608.24306 | [PDF](https://arxiv.org/pdf/2608.24306v1)

**作者:** Eran Hirsch `[一作]` (Bar-Ilan University), Ido Dagan `[通讯]` (Bar-Ilan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种局部评估方法，诊断多代理深度研究系统中的引文召回错误并定位责任代理。

**💡 创新点**

创新点是将错误分类为幻觉、未引用输入依赖、未引用输出、引用不足四类，并结合代理级与系统级评估实现错误定位与诊断。

**🔧 技术方法**

采用基于LLM的归属与蕴含推理（使用LongCite、DEER、Localized Attribution等提示），并进行人工验证。

**📊 数据集**

在DeepResearch Bench上评估了三套公开的深度研究系统（Nvidia AI-Q、MS-Agent、TrajectoryKit）。

**📈 对比分析**

与原系统相比，经过两项简单干预（用原始snippet替代研究者笔记、添加“不要使用未引用信息”提示），引文召回提升约5%，精确率提升3-7%，报告质量（RACE）保持不变。

**⚠️ 局限性**

局限在于LLM判定的可靠性、错误类型与模型能力关联仅为观察性、仅评估句子级引文未包含表格内容。

---

## 49. Observability and Fault Injection for LLM-Based Multi-Agent Systems in Software Engineering

**arXiv ID:** 2608.24271 | [PDF](https://arxiv.org/pdf/2608.24271v1)

**作者:** Zahra Seyedghorban `[一作]` (Delft University of Technology), Burcu Kulahcioglu Ozkan `[通讯]` (Delft University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种轻量级、与框架无关的工具，结合OpenTelemetry分布式追踪与故障注入，帮助研究者对LLM驱动的多智能体系统进行可观测性分析与受控压力测试。

**💡 创新点**

创新点在于将观测与故障注入统一到同一追踪模型上，支持在已存在的多智能体工作流中以最小改动实现结构化追踪，并可在通信、工具调用和LLM调用等关键边界注入多种故障，进而实现基线与受扰动运行的可比对。

**🔧 技术方法**

技术包括OpenTelemetry标准语义、分布式追踪、上下文传播、装饰器与上下文管理器实现的轻量化仪表化、以及基于配置规则的故障注入层。

**📊 数据集**

使用的基准数据集包括30任务的ProgramDev软件开发基准，以及ChatDev聊天驱动软件开发框架的真实任务。

**📈 对比分析**

比较方法是对基线、单一LLM调用延迟以及Agent-to-Agent（A2A）延迟三种情境进行多次重复，计算放大系数（故障延迟导致的整体运行时间增长比例），结果显示在演示系统中放大系数约为1~1.3，而在ChatDev中放大系数可高达48-59，表明小幅延迟在多阶段、消息繁重的工作流中能被放大。

**⚠️ 局限性**

局限性包括目前仅支持三类边界的故障注入，缺乏高级分析功能（如差异化分析、根因排名、自动调试报告），以及对更广泛的工作流、内存、协调等故障类型的支持尚未实现。

---

## 50. Callability Is Not Operability: Controlled Interface Interventions for LLM Agents

**arXiv ID:** 2608.23628 | [PDF](https://arxiv.org/pdf/2608.23628v1)

**作者:** Zihao Wang `[一作]` `[通讯]`, Zihao Wang

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究工具调用的可操作性（operability）与可调用性（callability）之间的差距，提出 Agent‑First Tooling（AFT）框架，将工具接口拆解为七个机制（选择性发现、可恢复调用、可持续状态、效果语义、结构化输出、验证等），并通过受控接口干预（Controlled Interface Intervention）在保持任务、后端、故障、模型不变的前提下，系统评估各机制对 LLM 代理可靠性的影响。

**💡 创新点**

创新点在于（1）首次提出“可操作性”概念并用多维度机制描述工具接口；（2）设计了受控接口干预框架，能够单独激活/抑制接口机制，分离模型与接口对结果的贡献；（3）在三族 LLM 上验证机制效能，揭示接口与模型之间的交互效应。

**🔧 技术方法**

技术手段包括：LLM 驱动的代理控制器；可插拔的工具接口定义与实现；故障注入（网络中断、状态丢失、响应丢失等）；持久化 SQLite 后端用于复现事务行为；统计方法：配对检验、Bootstrap CI、Holm 多重校正；以及结构化输出与验证日志的自动化收集。

**📊 数据集**

数据集与工作负载：自定义的合成任务集合，覆盖六类工作负载（选择性发现、突发中断、响应丢失、状态丢失、权限漂移、终端错误验证等）；工具目录规模在 10~1000 项；使用三族 LLM（Qwen 3.7 Plus、DeepSeek V4 Pro、GPT‑5.6 Sol）和持久化 SQLite 作为后端；所有实验在同一 frozen 版本下重复。

**📈 对比分析**

比较方法：对每个实验对（相同任务、后端、故障、模型、控制器），仅改变接口机制；评估指标包括：上下文令牌量、能力召回、恢复成功率、重复/不安全外部效果率、终端错误率。结果显示：选择性发现可减少约 4,013 tokens，召回不低于基线；恢复机制（可恢复调用、可持续状态）分别在对应故障下恢复率提升 100%；效果语义将重复效果降低 57% 不安全提交降低 50%；验证机制将错误终端声明降低 28%。所有主对比在 Holm 校正后均显著。

**⚠️ 局限性**

局限性：仅在合成和 SQLite 后端验证，未覆盖真实生产服务；工作负载故意激活特定失败，未评估真实失败频率；仅三族 LLM，模型泛化未知；未测试大规模并发/网络延迟；结构化输出与可观测性机制的独立效能证据相对薄弱；接口语义需手动定义，缺乏自动化推导与验证方法。

---

## 51. Fiber Optic Sensing Glove for High Performance Dexterous Manipulation Capture

**arXiv ID:** 2608.24572 | [PDF](https://arxiv.org/pdf/2608.24572v1)

**作者:** J. D. Peiffer `[一作]`, Ergys Ristani `[通讯]` (Meta)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

该论文提出了一种新的控制算法，用于优化系统的性能。

**💡 创新点**

创新点在于引入了一种新的优化策略，能够在复杂环境中提高控制精度。

**🔧 技术方法**

使用了自适应控制技术和机器学习算法。

**📊 数据集**

实验中使用了标准的控制系统数据集，以及模拟生成的数据。

**📈 对比分析**

与传统控制方法进行了比较，结果显示新方法在响应时间和稳定性上有显著提升。

**⚠️ 局限性**

限制在于算法在极端条件下的表现尚未充分验证。

---

## 52. Orientation in Extended Position-Based Dynamics: Application to Rigid Bodies and Cosserat Rods

**arXiv ID:** 2608.23606 | [PDF](https://arxiv.org/pdf/2608.23606v1)

**作者:** Samuel Tobin `[一作]` (University of Tennessee-Knoxville), Caleb Rucker `[通讯]` (University of Tennessee-Knoxville)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

通过 Lie 理论对 XPBD 进行扩展，给出统一的旋转约束、梯度、插值和求解框架，并将其应用于刚体关节和 Cosserat 杆的高精度仿真；同时提出基于块 Gauss–Seidel 的约束求解器和高阶有限元杆离散化。

**💡 创新点**

创新点包括：
1) 用 Lie 群上的加减运算（boxplus、boxminus）构造可微分约束，避免传统欧几里得梯度中的非线性误差；
2) 采用向量化约束而非标量约束，大幅提升约束线性化精度；
3) 统一的旋转插值与求导实现，支持任意阶有限元杆；
4) 通过块约束求解器与块带状求解器显著改善长链条、刚体连杆的收敛速度；
5) 采用均匀欠积分消除剪切锁定，在高长宽比杆件中保持正确的柔度。

**🔧 技术方法**

技术主要包括：
- XPBD 位置/方向耦合时间离散化；
- Lie 群 SO(3) 的指数映射、对数映射、Jacobi 矩阵；
- 旋转梯度的闭式推导；
- 块 Gauss–Seidel 与块带状线性求解器；
- 基于高阶 Lagrange 基函数的 Cosserat 杆有限元；
- 统一的碰撞检测与摩擦/恢复模型。

**📊 数据集**

使用的“数据集”主要是仿真测试：
- 三连杆摆、3‑RPS 并联机器人逆动力学；
- 长桩杆、重力加载下的大变形杆；
- 带前驱的螺旋弹簧压缩与屈曲；
- 过载扭转产生的螺旋线环（plectoneme）；
- 纸板弹珠/绳索的高速碰撞；
- 带压缩腿的 bristlebot；
- 以上均使用自行构建的几何模型、材料参数与初始条件。

**📈 对比分析**

比较方法：
- 与原始 XPBD（仅位置）及基于标量约束的实现做对比；
- 在刚体实验中对比向量约束/块求解与标量约束/单约束的原始残差和约束残差；
- 在杆件实验中对比链式刚体离散、线性/二次/三次有限元离散以及欠积分方案；
- 结果显示：向量约束与块求解将原始残差降低 10⁵–10⁶ 倍；
- 高阶有限元在相同计算时间下显著提升精度，剪切锁定得到有效消除；
- 计算时间方面，块求解对 CPU 时间几乎无增幅，显著优于单约束迭代。

**⚠️ 局限性**

限制与不足：
- 只针对刚体和 Cosserat 杆的离散，尚未涵盖更复杂的流体/弹性体；
- 对极端大旋转时数值稳定性仍需进一步验证；
- 块带状求解器对大型多体系统的并行化实现尚未深入；
- 仅在单 CPU 核心上评估，实际多核/GPU 加速效果未知；
- 对于非光滑碰撞（如齿轮、非凸形状）需要更复杂的碰撞检测与摩擦模型。

---

## 53. Turn Complexity and Bounded Languages

**arXiv ID:** 2608.24259 | [PDF](https://arxiv.org/pdf/2608.24259v1)

**作者:** Giovanni Pighizzini `[一作]` `[通讯]` (University of Milan), Giovanni Pighizzini (University of Milan)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

对受限上下文无关语言（bounded 语言）下推自动机的转折数进行可判定性与复杂度分析，并证明若不受限转折数则至少线性增长。

**💡 创新点**

在 bounded 语言情形下首次证明转折数可判定以及对任意固定 k 的 k‑转折判定可解，并建立从常数到线性增长的阶层结构；同时对全局接受计算的转折数给出线性下界。

**🔧 技术方法**

使用半线性集合、Presburger 算术、Parikh 映像、逆同态和状态标记等构造技术。

**📊 数据集**

无实验数据，本研究属于纯理论分析。

**📈 对比分析**

无实验比较与性能评估，所给定的结果为可判定性与下界的理论证明。

**⚠️ 局限性**

结果仅适用于 bounded 语言，对一般语言不适用；对两计数器机或更强模型失效；若接受计算数量无限则无法得到线性上界。

---

## 54. Resilience Matters for Embodied Agents System: New Metrics, Systematic Evaluation, and Optimization

**arXiv ID:** 2608.23839 | [PDF](https://arxiv.org/pdf/2608.23839v1)

**作者:** Yapeng Liu `[一作]` (National University of Defense Technology), Huaimin Wang `[通讯]` (National University of Defense Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了面向具身代理系统（EAS）的复原力评估框架，定义了恢复成本、稳定性和优雅可扩展性三大指标，并实现了非侵入式评估层，利用这些指标对 10 种代表性 EAS 在 Habitat‑Sim 3.0 的 400 个家庭任务中的执行过程进行量化诊断和优化。

**💡 创新点**

创新点在于：①首次为 EAS 提出以执行过程为导向的复原力指标体系，突破传统以成功率为主的结果导向评价；②将系统工程中的复原力概念（rebound、robustness、graceful extensibility）量化为可操作的度量；③构建了可插拔的评估层，使得复原力评估能够在不修改原始规划器的前提下进行；④通过指标引导的三种针对性优化（恢复、稳定性、可扩展性）验证了评估的可操作性。

**🔧 技术方法**

核心技术包括：①利用轨迹、运行日志、监控信号和 LLM 判断器提取执行过程特征；②基于阶段基线（Stage Baseline）计算恢复成本的 Mahalanobis 距离；③通过价值函数和 TD 残差评估稳定性；④构建压力响应映射和可扩展性阈值；⑤实现三种指标驱动的反馈回路以减少恢复窗口、降低策略波动和抑制长尾等待。

**📊 数据集**

数据集：在 Habitat‑Sim 3.0 物理仿真环境中生成 400 个家庭任务（导航、操纵、运输等），并对每个任务使用 10 种不同的 EAS 方法进行实验，形成包含执行轨迹、日志和 LLM 评判的完整数据集。

**📈 对比分析**

通过与传统成功率、完成率和安全得分的对比，复原力指标能揭示相同结果下的执行差异；实验表明不同方法在恢复成本、稳定性和压力容量方面存在显著差异，没有单一方法在所有维度均优；在指标引导下的优化能显著降低恢复成本或提高压力容量，但往往伴随其他维度的下降，体现了复原力的多维折衷。

**⚠️ 局限性**

限制包括：①评估基于仿真，真实机器人环境中的复原力表现需进一步验证；②指标体系尚未覆盖长期适应性（sustained adaptability）等维度；③当前方法缺乏针对具体部署场景的自适应权衡策略；④在极端扰动或复杂语义变化下，LLM 判断器的可靠性仍需提升。

---

## 55. RetrievalFormer: A Dual-Encoder Transformer for Efficient Approximate Nearest Neighbor Retrieval and Cold-Item Recommendation

**arXiv ID:** 2608.24079 | [PDF](https://arxiv.org/pdf/2608.24079v1)

**作者:** Theodore Rogers `[一作]` (Amazon Web Services), Soyoung Yang `[通讯]` (Amazon Web Services)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了共享搜索与推荐的双编码器索引在推荐端的准确性成本、冷启动性能与服务成本，并通过严谨的实验协议对比了多种基线模型。

**💡 创新点**

创新点：①在统一索引下量化准确性与规模壁垒的关系；②用配对实验拆解全Softmax与InfoNCE对性能的影响；③在严格的冷启动评估中展示内容塔对未见项目的显著优势；④对目标定义进行了校正并提供完整的评测协议。

**🔧 技术方法**

技术手段：双编码器Transformer、AttentionFusion特征融合、InfoNCE采样、全Softmax交叉熵、全索引扫描、IVF‑PQ近似检索、共享embedding表、特征噪声正则等。

**📊 数据集**

数据集：MovieLens‑1M、MIND‑small/large、AliEC广告日志、Avito广告搜索日志。

**📈 对比分析**

比较方法：在同一训练与调优协议下重新训练并调优六个基线（GRU4Rec、SASRec、BERT4Rec、SASRecF、FDSA、DIF‑SR），检验Recall@20、NDCG@20、Echo@20等指标。结果表明RetrievalFormer在warm端达94.8%强基线；在冷启动中内容塔比专用冷启动模型高约1.4倍；在搜索端无冷启动惩罚；服务成本与ID‑softmax相当，ANN对准确性无显著损失。

**⚠️ 局限性**

限制：①目标定义错误需校正；②全Softmax在百万级别catalog下内存壁垒无法突破；③基线与实验在历史长度、embedding表共享上不完全对齐；④评测仅在MovieLens/小型catalog上，未验证百万级规模的实际吞吐与准确性；⑤未对不同硬件与更大catalog的性能进行系统评估。

---

## 56. Task-Adaptive Rubrics for GUI Reward Modeling

**arXiv ID:** 2608.24174 | [PDF](https://arxiv.org/pdf/2608.24174v1)

**作者:** Tao Xiong `[一作]` (Zhejiang University), Shengyu Zhang `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了如何构建任务适应的评判准则来改进GUI代理的奖励模型，提出了Coarse-to-Fine Rubrics框架。

**💡 创新点**

创新点在于将评判准则分为类别级粗略检索和实例级细化生成两阶段，先检索通用规则再针对指令生成细化检查项，显著提升判定准确性。

**🔧 技术方法**

使用LLM辅助构建规则库、任务路由器、细化生成器和VLM验证器，结合多模态输入实现奖励判定。

**📊 数据集**

使用OGRBench跨平台GUI奖励基准（OSWorld、AndroidWorld、Windows、macOS、WebArena-Lite-v2）进行离线评测，并在MobileWorld环境下进行在线RL实验。

**📈 对比分析**

与六个基线（DigiRL、DistRL、AndroidGen、WebRL、ZeroGUI、OS-Themis）比较，在离线准确率86.7%/F1 86.6，线上RL任务成功率提升4.23个百分点，均优于其他方法。

**⚠️ 局限性**

限制包括规则库为离线固定、未覆盖所有应用场景和指令风格、对新兴任务的适应性不足。

---

## 57. Are Android GUI Agents Robust Against Runtime Anomalies? AnTrap: Evaluating Agents in Dynamic Adversarial Environments

**arXiv ID:** 2608.24099 | [PDF](https://arxiv.org/pdf/2608.24099v1)

**作者:** Guo Gan `[一作]` (Zhejiang University), Hong Zhou `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了一个名为STAR的Android GUI代理鲁棒性评测基准，通过在原始任务中注入四层十子类别的动态异常，保持任务可解性，系统评估代理在真实运行时异常下的恢复能力。

**💡 创新点**

创新点在于提出细粒度的四层（State、Thinking、Action、Round）异常分类法、设计可动态注入且不破坏任务完成性的陷阱构造管线，并通过GRPO强化学习区分可通过环境学习和思维瓶颈难以解决的异常类型。

**🔧 技术方法**

使用的技术包括基于AndroidWorld的任务扩充、图像与系统层级的异常注入、Group Relative Policy Optimization（GRPO）强化学习、以及对多模态语言模型的思维与指令版评估。

**📊 数据集**

采用了扩展自AndroidWorld的236个动态任务集合，并在此基础上生成对应的带陷阱任务，涵盖了所有四层十子类别的异常场景。

**📈 对比分析**

实验对比显示，在无陷阱环境下多数模型平均成功率为约74%，加入陷阱后平均下降至约66%；通过在含陷阱环境中训练GRPO，可显著提升状态和动作层单步异常的鲁棒性（约8-11%），但对多步上下文陷阱提升不足1%；整体性能提升仍有限。

**⚠️ 局限性**

主要局限包括任务样本规模有限、未尝试对抗性监督微调（SFT）提升上下文理解、以及基准设计侧重评估而非提供完整的训练改进方案。

---

## 58. A tale of perfect fit and phantom optima: how data-driven models can fail in real-time optimization

**arXiv ID:** 2608.23885 | [PDF](https://arxiv.org/pdf/2608.23885v1)

**作者:** Prithvi Dake `[一作]` (University of California Santa Barbara), James B. Rawlings `[通讯]` (University of California Santa Barbara)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在乙烯醋酸乙酯（VAc）工厂的数值仿真中训练并验证两类数据驱动模型（结构化混合模型和全黑盒神经ODE模型），并将它们用于实时优化（RTO），研究了模型预测精度与经济最优解的一致性。

**💡 创新点**

创新点在于提出并验证了一个决策导向的基准，展示即使模型对测量数据拟合极佳，训练过程中的随机梯度优化和模型结构缺陷仍可导致RTO产生多重伪最优解，强调仅靠验证误差不足以保证经济性能。

**🔧 技术方法**

技术包括：基于物理约束的神经网络闭包（hybrid）和完整数据驱动的神经ODE；训练使用混合ADAM+L‑BFGS优化器；对比评估采用利润损失指标（Δℓ）和多启动RTO搜索；对比训练和验证误差采用均方根误差（RMSE）。

**📊 数据集**

使用的“数据集”是从已知的VAc工厂模型生成的无噪声和带噪声的浓度时间序列，覆盖±30%（或±10%）的操作变量范围，包含200条随机输入轨迹，30%留作验证。

**📈 对比分析**

比较方法：对每个模型训练20个随机初始化的实例，计算其在RTO中得到的利润损失分布；与基准参数化模型（可识别并恢复最优）对比。结果显示，只有使用速率测量训练的结构化模型（rmeas）几乎无利润损失；其余结构化模型利润损失最高约30%，黑盒模型最高超过50%。

**⚠️ 局限性**

局限性在于标准的随机梯度训练过程对模型参数的漂移极为敏感，即使数据充足、无噪声、初始点已在最优附近，ADAM优化也可能将模型从最优解迁移至次优解；此外，本文仅在理想仿真环境下验证，缺乏对真实工业噪声、未观测状态或动态扰动的鲁棒性研究；未探索对神经网络施加物理或凸性约束的改进方法。

---

## 59. Does Episodic Memory Help Close the Lexical Frequency Gap in Sensitivity to Syntactic Contrasts? A Test Using Retrieval-Augmented Language Models

**arXiv ID:** 2608.23851 | [PDF](https://arxiv.org/pdf/2608.23851v1)

**作者:** Jing Liu `[一作]` (Université PSL), Najoung Kim `[通讯]` (Boston University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用 kNN‑LM 在语法对比测试中检验短语记忆机制是否能缩小词频带来的差距。

**💡 创新点**

将补充学习系统理论与检索增强语言模型相结合，证明了情景记忆可补偿低频项的弱统计表示。

**🔧 技术方法**

采用检索增强（kNN‑LM）技术，并对检索粒度、上下文窗口和邻居数进行多维实验。

**📊 数据集**

在两种预训练规模下使用 BLiMP、Zorro 与 BIG‑bench 的最小对比数据集，基准模型为 GPT‑2 XL 与 GPT‑2 Small。

**📈 对比分析**

与纯参数化基线相比，kNN‑LM 在语法对比准确率和困惑度上均有提升，低频项收益最大，频率差距被显著缩小但未完全消除。

**⚠️ 局限性**

实验仅覆盖受控的英文最小对比句，kNN‑LM 仅为 CLS 的功能类比，结构与语义信息的分离不完全，缺乏自适应检索配置，且未验证自然语境下的表现。

---

## 60. A Durable Vision-Based Tactile Fingertip for Robotic Manipulation

**arXiv ID:** 2608.24242 | [PDF](https://arxiv.org/pdf/2608.24242v1)

**作者:** F. Richard Cottrell `[一作]` (MIT), Edward H. Adelson `[通讯]` (MIT)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种可更换感知模块的耐久视觉触觉手指，并通过旋转刮擦和重复探针两种加速实验评估其耐磨性和重复加载寿命。

**💡 创新点**

创新点在于将多层热塑性聚氨酯保护膜与可替换硅胶胶囊相结合，实现两位数以上的耐磨提升、损伤逐步、功能保留长、且可快速更换，解决了传统视觉触觉传感器的寿命与维护难题。

**🔧 技术方法**

采用硅胶基质、TPU保护膜、硅烷耦合固化工艺，以及旋转刮擦测试和重复探针测试（加速耐久性评估）等实验技术。

**📊 数据集**

论文未使用公开数据集，仅通过自制实验样本与GelSight Mini、DIGIT商业产品进行对比测试。

**📈 对比分析**

通过与GelSight Mini和DIGIT在相同刮擦条件（400‑grit 200 g）和重复探针条件（39.2 N、45 c/min）下的对比实验，证明自制传感器在耐磨测试中可达180 min，约比商业产品高100–1000倍；在重复加载测试中功能可保持5–8天，显著优于商业产品（≈25–35 min）。

**⚠️ 局限性**

局限性包括：实验采用加速损伤条件，未能完全代表真实工业环境；缺乏长期现场使用数据；各工艺变量（涂覆顺序、热处理等）的单独影响尚未量化。

---

## 61. DriftAD: Visually-Guided Text Drift for Few-Shot Industrial Anomaly Detection

**arXiv ID:** 2608.23723 | [PDF](https://arxiv.org/pdf/2608.23723v1)

**作者:** Wenyang Liu `[一作]` (Nanyang Technological University), Adams Wai-Kin Kong `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 DriftAD 框架，利用视觉引导文本漂移和漂移导向空间门控等模块，在少量正常样本条件下实现工业缺陷的精准定位与检测。

**💡 创新点**

核心创新在于（1）将 CLIP 文本嵌入通过视觉上下文动态漂移，生成层级、空间适配的异常描述；（2）漂移导向空间门控显著提升异常相关视觉特征；（3）引入异常信号放大、漂移分离和门控监督等损失，进一步提升匹配质量。

**🔧 技术方法**

采用 OpenCLIP ViT‑H/14 作为视觉/文本编码器，结合 ASA（空间+频域增强）、VGTD（视觉引导文本漂移）、DGSG（漂移导向空间门控）、MBAS（多分支融合）以及 BCE、Dice、Focal、gate、drift 等多种损失。

**📊 数据集**

在 MVTec‑AD 与 VisA 两大工业缺陷数据集上进行实验，训练仅使用正常样本，合成异常样本作为像素监督。

**📈 对比分析**

与 PatchCore、WinCLIP、AnomalyGPT、PromptAD、ResAD、KAG‑prompt、FiLo++、FocusPatch‑AD 等少量样本方法对比，DriftAD 在 1、2、4 shot 场景下均取得更高的图像级 AUROC 与像素级 pAUROC，1‑shot 时提升约 1.5% AUROC、2.6× 文本判别力，且在 8‑shot 与全量基准中也能与之竞争。

**⚠️ 局限性**

局限性包括仍依赖 CLIP 预训练对语义表达的限制，对极小或复杂纹理缺陷的识别仍有挑战；漂移模型对图像尺度固定；未在跨域或非工业场景进行验证；相对较高的模型与计算开销。

---

## 62. EMRB: A Multi-Level Benchmark for Evaluating LLM Reasoning over Raw Electromagnetic Signals

**arXiv ID:** 2608.24086 | [PDF](https://arxiv.org/pdf/2608.24086v1)

**作者:** Mingxu Zhang `[一作]` (Hong Kong University of Science and Technology), Shan Huang `[通讯]` (National University of Defense Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 EMRB 基准，用原始 I/Q 数据让大语言模型（LLM）通过编写并执行代码来完成从信号检测到 OFDM 设计的多层次电磁信号分析任务。

**💡 创新点**

创新点包括：① 将原始复数时域数据作为输入，要求 LLM 自动发现并计算所需量；② 设计了 5 级难度和 27 种问题类型的系统化难度分级；③ 采用可重现的程序化信号生成与确定性验证器；④ 引入 ReconPilot 三阶段结构（固定探测 → 目标分析 → 自检）提升 LLM 的分析表现。

**🔧 技术方法**

使用了 Python 代码执行工具、FFT、PSD、STFT、自动相关、波形分割等经典 DSP 方法，以及 LLM 的自然语言交互与代码生成能力。

**📊 数据集**

数据集为 200 个基准问题，涵盖 11 种信号类型（数字调制、模拟调制、OFDM、LFM 等），每个问题对应一个原始 I/Q 文件，采用固定种子保证可复现与客观验证。

**📈 对比分析**

通过 14 个不同家族（专有、开源、推理型、轻量级）的 LLM 进行对比，分数从 24.1% 到 78.9% 变化；在 5 级难度上从 84.9%（基础测量）下降到 21.2%（系统设计）。ReconPilot 在三种 backbone 上平均提升 3.8~17.6 分，显著提高多信号分析与系统设计的表现。

**⚠️ 局限性**

局限性包括：① 仅覆盖合成信号，缺乏真实硬件失真、多径等环境因素；② ReconPilot 的固定 PSD 探测对弱/重叠信号可能不敏感；③ L5（系统设计）得分仍低，表明目前 LLM 对整体系统合成仍有较大不足；④ 未提供人工专家基准，仅通过模型联合和参考解算器证明可行性。

---

## 63. Coverage Planning for Robotic Tooth Preparation in Densely Constrained Environments

**arXiv ID:** 2608.24155 | [PDF](https://arxiv.org/pdf/2608.24155v1)

**作者:** Yunwen Li `[一作]` (Tsinghua University), Xiang Li `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一套可实现全冠牙齿自主切削的机器人系统，能够在口腔内完成牙体结构的精准去除并形成临床可用的冠形态。

**💡 创新点**

创新点在于：①提出解剖学感知的工具路径规划算法，兼顾牙冠目标形状与邻牙保护；②设计了清晰度导向的手柄姿态分配策略，利用余弦余轴旋转避免软组织与光学标记碰撞；③将上述两项整合到六自由度臂上，实现全程安全、精准的切削。

**🔧 技术方法**

使用技术包括：TRIOS 3口内扫描、三角网格分析、基于切削器几何的层层推进路径生成、最大化工具轴与面法线非钝角的优化、手柄姿态的yaw冗余赋值、UR3e六自由度机器人+光学跟踪+力/扭矩传感器、FreeCAD仿真与phantom头实验。

**📊 数据集**

数据集主要为：技术员设计的数字冠模型、同一口腔的TRIOS 3扫描数据，以及实验用的牙模phantom头模型。

**📈 对比分析**

通过与传统手动切削、无邻牙保护或光学姿态未优化的机器人对比，实验显示RMSE 0.117 mm（最佳拟合）/0.321 mm（现场对齐），90%点误差≤0.210 mm，邻牙未受损且软组织碰撞率为0%，验证了方法在精度与安全性上的优越性。

**⚠️ 局限性**

局限性包括：实验仅在phantom头上进行，缺乏真实口腔软组织建模；手柄姿态策略依赖保守规则，缺乏实时自适应；切削时间较长，尚未实现在线或离线的速度优化；扫描与配准误差仍影响最终精度，需要更鲁棒的配准与定位技术。

---

## 64. Example-based Robust Abnormality Detection with Minimal Annotations using Exemplar Med-DETR

**arXiv ID:** 2608.24281 | [PDF](https://arxiv.org/pdf/2608.24281v1)

**作者:** Sheethal Bhat `[一作]` (Friedrich-Alexander-Universität), Andreas Maier `[通讯]` (Friedrich-Alexander-Universität)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现了 Exemplar Med‑DETR，一种基于 Vision‑Language DETR 的少样本检测框架，利用示例特征生成和领域感知对比学习，在仅少量标注的情况下实现胸片异常检测。

**💡 创新点**

① 引入 Exemplar Generation Module 生成类别原型嵌入，并与文本嵌入交叉关注；② 采用域感知的负样本采样和多阶段迭代对比训练；③ 通过内存银行保存原型实现持续学习，无需大规模重训练。

**🔧 技术方法**

Transformer‑based DETR（Grounding DINO + Swin‑B backbone），文本编码器，示例生成网络，cosine 对比损失和特征一致性损失，逐阶段迭代训练与负样本挖掘。

**📊 数据集**

专有的 18,681 张全分辨率 CXR 图像（包含 16 种异常）与 100 张解剖标注集；公开 VinDR‑CXR（3,000 张）用于 OOD 验证。

**📈 对比分析**

与 Grounding DINO、RPS、Deformable DETR、RetinaNet 等基线在 7% 与 100% 注释下进行对比。EM‑DETR 在 7% 注释时敏感度提升 10–30%，mAP50 提升 4–16%；在 100% 注释下与 RPS 接近或略优，整体性能接近 SOTA。

**⚠️ 局限性**

内存占用高、文本编码器仅做伪标签、对注释噪声敏感、对某些异常（如 pneumothorax）需要更多样本、实现未优化。

---

## 65. Confidently Wrong, Silently So: Auditing Undetectable Failures of a Deployed On-Device Language Model

**arXiv ID:** 2608.23663 | [PDF](https://arxiv.org/pdf/2608.23663v1)

**作者:** Shashwat Pandey `[一作]` (University of California Santa Cruz), Suresh Raghu `[通讯]` (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对可部署的 on‑device 语言模型进行可靠性审计，揭示任务不对称的失衡、饱和置信度信号以及表面不可区分的高置信错误，并提出无模型访问的黑盒一致性恢复方案。

**💡 创新点**

提出模型无关的可靠性审计协议、表面不可区分性检验与成本阶梯分析，证明单次前向推理无法检测错误，仅通过 O(N) 一致性即可恢复可靠性；同时强调评估应针对可部署配置而非仅验证版。

**🔧 技术方法**

红队测试、误差可检测性分析、AUROC/ECE 统计评估、k‑sample 自一致性（SelfCheck）、表面特征分类器、单次置信度与长度等不确定性信号。

**📊 数据集**

TriviaQA、Global‑MMLU、含 110 个无效 + 150 可答控制的 false‑premise 集合、310 条摘要评估集、GSM8K 等冻结评估数据。

**📈 对比分析**

与 Gemma‑3‑4B‑it、Llama‑3.2‑3B‑instruct、Minstrel‑3B 等公开同级模型对比；on‑device 模型在置信度校准、confabulation 与 hallucination、过度拒绝方面表现最差；但在摘要任务上根拠性最高；采用 SelfCheck 可将 confabulation 从 75% 降至 3%，并将选择准确率从 42% 提升至 82%。

**⚠️ 局限性**

仅评估单一厂商模型，结果的普适性未知；依赖 API 版本差异；判定使用 LLM 而非人工；仅覆盖两类任务与单一语言；缺乏真实用户危害评估；成本阶梯未遍历所有预算；等价性检验的置信区间可能有限。

---

## 66. Can a Dynamic Internal Field Govern a Transformer's Cognition? Certifiability, not Superiority, in Homeostatic Compute Control

**arXiv ID:** 2608.24319 | [PDF](https://arxiv.org/pdf/2608.24319v1)

**作者:** Francisco M. Arrabal-Campos `[一作]` (University of Almería), Alfredo Alcayde `[通讯]` (University of Almería)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为 Homeostatic Background Processor（HBP）的动态内部场，用作 Transformer 的元认知调度器，控制计算深度与资源分配。

**💡 创新点**

创新点在于将低维物理场（波/扩散/KdV 等）与图 Laplacian 结合，构建可证明稳定性的 PDE 驱动调度器，并给出新的离散 Schur‑Cohn 判据；同时通过对比学习式 GRU 控制器验证其可证可行性。

**🔧 技术方法**

主要技术包括：图 Laplacian 上的二阶/一阶 PDE 组分、IMEX 以及 Verlet 积分、可证明的阻尼与摆动阈值、离散 Schur‑Cohn 判据、以及对物理参数的可学习约束。

**📊 数据集**

在 S₅ 置换组态追踪任务（生成器集合为相邻换位与 5‑cycle+换位）上进行实验，使用 OOD 评估（K 从 13→24）和 10/20 份种子。

**📈 对比分析**

与基线（无控制器、可学习门控、工作记忆）对比，HBP 在 OOD 计算适配度上略有提升，准确率保持不变；与匹配接口的 GRU 控制器的效果相当或略优，但仅 HBP 拥有可证明的稳定性。

**⚠️ 局限性**

局限包括：仅验证单一任务与小规模模型（4.2–5.6M 参数），仅在两组生成器中观察到结构效应，未探究更大图或不同物理场；缺乏对 OOD 准确率提升的显著发现；对控制器状态维度与阻尼参数的混杂未完全消除。

---

## 67. Transformer Accelerator (TFA): A Macro-Op INT8 Hardware Chip for Transformer Inference and Machine Translation

**arXiv ID:** 2608.23582 | [PDF](https://arxiv.org/pdf/2608.23582v1)

**作者:** Shashank `[一作]` `[通讯]` (Independent Researcher), Shashank (Independent Researcher)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了 Transformer 推理加速器 TFA（Transformer Accelerator），通过宏指令（macro‑op）指令集把所有 Transformer 块（自注意力、交叉注意力、前馈网络等）抽象为 8 种 512‑bit INT8 操作，并在 RTL 上实现了记忆到记忆的流式处理引擎，配合 AXI4 主从接口、DMA、软核编译器，能够以 bit‑exact 的方式执行完整的预训练 T5‑small 模型（60M 参数）进行多语言（英-法、德、罗）翻译，验证了设计的可验证性、功耗与性能提升。

**💡 创新点**

创新点包括：① 仅 8 个宏指令即可覆盖所有 Transformer 计算原语；② 通过对每个宏指令的严格 decode‑validation 合并到硬件，实现了完整的 bit‑exact 数值规范与 UVM 代码覆盖闭合；③ 引入随机 Hadamard 旋转的 lossless 变换，解决了 per‑tensor INT8 的激活异常问题，而不需要改变 ISA 或硬件；④ 通过 ping‑pong 操作缓冲与输出静止（output‑stationary）策略，实现了单词级自回归推理的内存带宽利用率 ≥ 93%。

**🔧 技术方法**

技术实现包括：INT8 乘加阵列、serial int‑sqrt/reciprocal（用于 RMSNorm）、maskable row‑softmax、element‑wise 乘加、DMA 双口缓冲、AXI4/AXI4‑Lite 接口、UVM Golden Model 验证、覆盖闭合、以及在编译时完成的随机 Hadamard 旋转与权重折叠。

**📊 数据集**

数据集：使用 HuggingFace 的 T5‑small 预训练模型，对 10 条英语谚语进行多语言翻译（目标语言为法语、德语、罗马尼亚语），同时构造了十句多语言压力测试集以验证推理流程。

**📈 对比分析**

性能比较方法：将 RTL 仿真结果与 22 线程 CPU 基准进行 end‑to‑end 速度对比，并通过 roofline 分析测算内存带宽利用率。实验显示，V1 版已在 RTL 上超过 22‑线程 CPU 约 20×，而更大配置预计每令牌能量可降低约 3 个数量级。覆盖率达到 100% 功能覆盖、94.96% 代码覆盖；验证错误率 0%。

**⚠️ 局限性**

局限性：① Prefill 阶段仍受内存带宽限制，V1 设计为单词级自回归内存带宽 bound；② 仅支持 INT8 且需要编译器进行旋转/量化预处理，若模型存在更复杂的激活分布仍需进一步变换；③ 设计规模受参数化约束，过大模型需扩展内存/带宽；④ 设计目前未对混合精度或自适应量化做支持，限制了对更大模型的直接迁移。

---

## 68. SonarLLM: A Native Sonar--Optical Multimodal Large Language Model for Underwater Perception

**arXiv ID:** 2608.24325 | [PDF](https://arxiv.org/pdf/2608.24325v1)

**作者:** Cong Su `[一作]` (Kunming University of Science and Technology), Zhengtao Yu `[通讯]` (Kunming University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了SonarLLM，一种将成像声呐作为本土感知模态并与光学图像联合的多模态大型语言模型，用于水下感知。

**💡 创新点**

创新点在于：设计声呐专用编码器与物理感知增强模块，构建可靠性感知的层级融合AGFM，并通过声呐-光学配对的SonarBench实现可控评估。

**🔧 技术方法**

使用了声呐专属Vision Transformer、Polar‑aware Positional Encoding、Optical‑VFE与Acoustic‑VFE、Adaptive Gated Fusion Module、DeepStack多层融合以及分阶段训练策略。

**📊 数据集**

数据集包括RGBS50、UATD、SCTD、DeeperSense、FLC+FLS、OceanGym、OceanInstruct、OceanPile等，构建了声呐-光学配对的训练和评测数据。

**📈 对比分析**

与多种基线（Qwen3‑VL、InternVL、MiniCPM、OceanGPT、NAUTILUS等）对比，SonarLLM在声呐单模宏观精度达到72%，在融合条件下达68.7%，分别比最强基线提升34.4和25.1个百分点。

**⚠️ 局限性**

局限在于评估主要使用合成光学退化而非自然海况，缺乏动态时间序列处理与声呐特定失效场景的验证，模型规模与推理成本相对较高。

---

## 69. From Traceability to Justifiability: Accountability Structures in Agentic Software Engineering

**arXiv ID:** 2608.23610 | [PDF](https://arxiv.org/pdf/2608.23610v1)

**作者:** Rashid Azarang `[一作]` `[通讯]` (Independent Researcher), Rashid Azarang (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对47个CI/CD与AI平台的文档进行双盲评分，并对30个公开GitHub仓库的发布流水线进行深度测评，系统评估了平台默认记录能否表达从评估到部署的可追溯与可辩护（justifiability）条件；

**💡 创新点**

创新点在于提出并量化了四条可辩护条件（artifact continuity、behavioral continuity、evidence continuity、authority continuity），并首次在大规模公开平台与流水线上同时验证了记录可表达性与实际实现度，揭示了“可追溯到可辩护”的结构性空缺；

**🔧 技术方法**

主要技术手段包括：固定三标签评分协议、双盲两轮评分、内容哈希与获取时间固定、深度测评工具自动从发布流水线的公开API提取配置与发布数据、基于规则的深度等级计算；

**📊 数据集**

数据集包含：47个平台的官方文档与API响应（20 CI/CD、27 AI平台），30个公开仓库的GitHub Actions/CI配置、发布资产、SBOM与 attestations 等；

**📈 对比分析**

方法对比：评分结果与深度测评结果相互印证；性能方面，评分覆盖率高（双盲一致率>70%），深度测评覆盖所有仓库且完全可复现，计算耗时仅数分钟；

**⚠️ 局限性**

限制主要在于：仅基于公开记录与文档，无法验证实际运行行为；深度测评仅覆盖 artifact 层，无法测量行为层的可辩护；平台变更与私有接口不在研究范围；因此结论仅适用于当前公开平台与流水线，未来需补充行为层记录与动态监测。

---

## 70. Latent-surrealism: Revisiting surrealism and its aesthetics in relation to contemporary AI-Generated cultural production

**arXiv ID:** 2608.24367 | [PDF](https://arxiv.org/pdf/2608.24367v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 71. NeurRAFT: Robot Motion Planning via Anchor-Level Flow Matching with Clearance-Aware Preference Tuning

**arXiv ID:** 2608.24026 | [PDF](https://arxiv.org/pdf/2608.24026v1)

**作者:** Sibo Tian `[一作]` (Texas A&M University), Xiao Liang `[通讯]` (Texas A&M University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一种名为 NeurRAFT 的生成式运动规划器，能够直接从点云感知生成可执行的机器人轨迹，并实现零样本从仿真到真实 Franka 机械臂的迁移。

**💡 创新点**

创新点包括：① 以锚点为基础的轨迹表示，避免稠密路径冗余；② 在流匹配训练中加入基于雅可比矩阵的权重，使损失更符合末端执行器位移；③ 通过自生成的障碍物净通道（signed clearance）进行偏好对齐（Preference Alignment），使用 Direct Preference Optimization（DPO）直接将分布转向更安全的轨迹；④ 在不增加推理开销的情况下提升碰撞避免性能。

**🔧 技术方法**

主要技术：条件流匹配（Conditional Flow Matching）、Cubic-spline 插值、雅可比加权损失、Direct Preference Optimization、点云编码（PointNet++）与 Transformer 架构。

**📊 数据集**

使用 MπNets 基准数据集（包含全球专家和混合专家轨迹），并在 500+ 真实世界规划任务中评估（桌面、抽屉、货架场景）。

**📈 对比分析**

与经典采样/优化规划器（PRM、CHOMP、STORM）以及最新神经规划器（MPNet、MπNets、EDMP、Neural MP、Cascaded Diffusion）对比。NeurRAFT 在所有三种测试拆分上均显著高于传统方法（成功率 93–99%），优于最强神经规划器（提升 4–13%），并在真实场景中实现约 86% 的成功率，展现了强大的零样本迁移能力。

**⚠️ 局限性**

局限性：① 仅处理静态环境，无法对动态障碍进行在线重规划；② 仍存在仿真到现实的差距，对感知噪声和未见几何体的鲁棒性有待提升；③ 对计算资源有一定需求，尤其在多锚点或多IK目标的情况下；④ 需要在训练集上构造优先级对，虽然无需人工标签，但仍需额外的后处理步骤。

---

## 72. Stop Abandoning Me: Exploring the Landscape of Unmaintained Intimate Partner Abuse Support Applications

**arXiv ID:** 2608.23826 | [PDF](https://arxiv.org/pdf/2608.23826v1)

**作者:** Ivy Turk `[一作]` (Ruhr-Universität Bochum), Rebekah Overdorf `[通讯]` (Ruhr-Universität Bochum)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究未被维护的亲密伴侣虐待支持应用，测量其存活率并探讨失去支持的原因。

**💡 创新点**

系统收集并分类现存支持工具，量化其被遗弃比例并提出未来评估和政策建议的框架。

**🔧 技术方法**

使用关键词搜索、雪球采样在Google Play、Apple Store、GitHub、学术数据库等平台手动收集数据，并采用手工注释与统计分析。

**📊 数据集**

构建了197个支持工具的数据集，包括135个Android应用、111个iOS应用和8个网页应用，记录发布、更新、评分等元数据。

**📈 对比分析**

通过统计比例分析，发现58.9%的应用已失去支持，计划使用逻辑回归检验不同类别与存活率的相关性，性能仍待提升。

**⚠️ 局限性**

数据量有限、语言覆盖不全、样本分布不均，导致模型预测能力不足，需扩大数据集以提升研究可靠性。

---

## 73. Joint Distribution Alignment for Universal Domain Adaptation

**arXiv ID:** 2608.24429 | [PDF](https://arxiv.org/pdf/2608.24429v1)

**作者:** Shizhe Li `[一作]` (South China University of Technology), Xiaowei Yang `[通讯]` (South China University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种联合分布对齐的通用域自适应算法JAUA，能够同时处理源域与目标域之间的标签空间差异与分布漂移问题。

**💡 创新点**

创新点在于首次给出了通用域自适应的目标域泛化误差上界，并以该上界为目标构建模型；同时设计了基于Chi‑Square散度的联合分布对齐与渐进式伪标签策略。

**🔧 技术方法**

主要技术包括：Chi‑Square散度度量、核方法与代表定理、随机傅里叶特征/共轭梯度求逆、Steepest Descent优化以及SVM伪标签与自适应阈值。

**📊 数据集**

实验使用了六大公开图像数据集：Office‑31、Office‑Home、VisDA、DomainNet、ImageCLEF、PACS。

**📈 对比分析**

通过与24种现有UniDA方法（如UAN、CMU、DANCE、DCC、OVANet、LIWUDA等）对比，JAUA在平均ACC、UNK和HOS指标上均实现最高或次高成绩，证明了其优越性能。

**⚠️ 局限性**

局限性包括：对超参数（尤其是λ_gss、λ_gst）的敏感性；需要大量矩阵计算导致在大规模数据上计算成本高；伪标签初始不可靠时易导致误判，需要进一步的错误传播分析。

---

## 74. SoK: ARCUS: On the Efficiency and Efficacy of Hardware Fuzzing

**arXiv ID:** 2608.23933 | [PDF](https://arxiv.org/pdf/2608.23933v1)

**作者:** Alenkruth Krishnan Murali `[一作]` (University of Virginia), Ashish Venkat `[通讯]` (University of Virginia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过构建两层分类体系和核心评估维度，对ISA、微架构和RTL三大抽象层的硬件模糊测试技术进行系统化梳理与评估。

**💡 创新点**

创新点：①首次提出跨层级的两层分类法；②定义统一的评估维度（输入刺激、算法、GRM、覆盖、反馈等）；③总结跨层级共性挑战与未来研究方向。

**🔧 技术方法**

使用的技术包括差分模糊、故障分析、观察无关模糊、模型驱动关系测试、定向模糊、AFL/LibFuzzer等软件模糊框架、FPGA/EDA仿真、LLM+RL、图模型、强化学习等。

**📊 数据集**

数据集来源：文献中公开的评测结果与bug列表（如CVA6、BOOM、Ariane等开源CPU，x86/ARM/ RISC‑V 实现），以及基准测试套件（Sandsifter、UISFuzz、Revizor、TheHuzz、MABFuzz等）和自动化注入的故障样本。

**📈 对比分析**

比较方法：对比每个工具在相同平台上产生的覆盖率（FSM/切换/跳转/折叠/可切换等）、测试用例数量、检测到的Bug数量以及执行时间；结果显示：例如在CVA6上，Hybrid/动态种子调度工具（HypFuzz、PSOFuzz、MABFuzz）比传统TheHuzz在检测同一组CWE时用的测试用例显著减少，覆盖率提升约0.3–2.2%；在ISA层面，SkipScan等结构化生成工具在识别未记录指令方面的测试效率比随机生成工具高4–5倍。

**⚠️ 局限性**

局限性：①缺乏可靠的Golden Reference Models，GRM错误导致误报/漏报；②对黑盒目标的反馈极限，导致模糊效率低、需要海量测试用例；③手工分析仍占主导，难以自动化；④覆盖度指标缺乏统一标准，难以跨工具对比；⑤算法多样性不足，主要集中在AFL/遗传/强化学习等，缺乏更具针对性的搜索策略；⑥跨层级集成和标准化工具链尚未形成，阻碍系统化硬件验证。

---

## 75. A Behavior-Guided Online Probabilistic Forecasting Method for Electric vehicle Charging Loads

**arXiv ID:** 2608.24441 | [PDF](https://arxiv.org/pdf/2608.24441v1)

**作者:** Chenghan Li `[一作]` (Tsinghua University), Yuxuan Liang `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种基于行为引导的在线概率预测框架，用于电动汽车充电负荷的实时预测。

**💡 创新点**

创新点在于将充电行为分为长期固定模式和近期动态变化两层时间尺度，并将行为偏移进行语义编码，动态调节预测模型的残差补偿与学习率，从而实现对概念漂移的精准响应。

**🔧 技术方法**

核心技术包括双时尺度行为特征提取（使用统计量如均值、方差、峰时指数等）、语义提示生成与冻结语言模型编码（DistilGPT‑2）、残差自适应调节（使用门控机制）以及基于Pinball损失的在线概率量化回归。

**📊 数据集**

使用了来自中国十个异构充电站的真实充电数据集，包含多种充电模式与行为漂移特征。

**📈 对比分析**

与TCN、PatchTST、LSTM、FSNet、OneNet等传统与漂移感知基线进行比较；在1小时与4小时预测任务中，方法分别比最佳基线MSE降低约15%–17%，Pinball损失降低约18%–23%，预测区间覆盖率接近90%且区间宽度更窄，整体概率校准和精度均优于对比模型。

**⚠️ 局限性**

局限性包括模型参数量大、对罕见或突发充电行为的快速捕捉仍有限、语义提示生成依赖语言模型的质量且在不同语言或领域可能需要重训，以及对外部影响因素（如天气、电价）的整合仍待进一步研究。

---

## 76. Velocity-coupled Representation Refinement for Satellite Orbit Prediction

**arXiv ID:** 2608.23728 | [PDF](https://arxiv.org/pdf/2608.23728v1)

**作者:** Yue Yang `[一作]` (Xi'an Jiaotong University), Fan Ma `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种面向卫星轨道预测的深度学习框架 OrbitNet，通过位置-速度耦合的表示细化与轨道段落建模实现轨道状态序列的准确预测。

**💡 创新点**

创新点：① 将速度信息以表示层次的耦合方式注入位置表示，显式捕捉位置-速度的交互；② 将轨道历史序列划分为时间段进行段落级学习，既保留局部运动变化，又把握全局长程依赖；③ 通过可学习的融合系数和段落位置嵌入实现自适应表示。

**🔧 技术方法**

技术：位置-速度耦合卷积细化、段落投影与位置嵌入、轻量级预测头、实例归一化、端到端训练；使用 Transformer/MLP/线性等传统模型做对照。

**📊 数据集**

数据集：基于 Space-Track 公开 TLE 记录，利用 Orekit 转化为每分钟采样的 6 维状态序列；训练集为 Starlink 卫星（208 颗），测试集为剩余 52 颗 Starlink 与 6 个未见星座（ASTROCAST、CAPELLA、ICEYE、KINEIS、LEMUR、SKYSAT）。

**📈 对比分析**

对照方法包括多种 Transformer（AutoFormer、FEDformer、iTransformer）、线性/MLP（DLinear、TimeXer、DropPatch、WPMixer）、CNN（TimesNet）、时序基座模型（TTM、TimesFM、MOIRAI、Times-MoE）以及专用轨道预测模型（KiGRU、DASR）。在 Starlink 内域实验中，OrbitNet 在 MAE 3.34 m、RMSE 22.80 m 领先所有基线；在六个星座的零样本测试中，平均 MAE 14.07 m，显著优于基线（平均 MAE 45.47 m 或 193.20 m）。

**⚠️ 局限性**

局限性：① 仅基于数据驱动，未显式加入轨道动力学、环境扰动或机动信息，导致在极端扰动或长周期预测时可能缺乏物理一致性；② 对于更大规模的轨道时间序列或多任务场景，仍需探索更通用的轨道基础模型与多任务学习。

---

## 77. Bridging Adversarial and Collaborative Learning for AI-Generated Image Quality Assessment

**arXiv ID:** 2608.24372 | [PDF](https://arxiv.org/pdf/2608.24372v1)

**作者:** Baoliang Chen `[一作]` (South China Normal University), Sijie Mai `[通讯]` (South China Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `da1b1a89-583a-4b57-9c81-478778569bec` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种双路交互式 AI‑生成图像质量评估框架，融合感知与文本对齐两维度的对抗与协作学习；

**💡 创新点**

创新点在于引入双门控混合专家（Dual‑Gated MoE）动态路由特征，利用学习指令与梯度反转实现对抗/协作路径的自适应切换；

**🔧 技术方法**

使用 CLIP 文本‑图像编码器、ViT+Dual‑Gated MoE 专家网络、差分卷积专家、梯度反转层以及任务指令引导的门控机制；

**📊 数据集**

实验覆盖 AGIQA‑1K、AGIQA‑3K、AIGCIQA2023、AGIQA‑20K、EvalMi‑50K 等五大公开基准；

**📈 对比分析**

通过 SRCC/PLCC 与现有多种 IQA/AIGIQA 方法对比，在所有基准均取得 SOTA 或排名前列，尤其在极端情景与跨域迁移中表现更稳健；

**⚠️ 局限性**

局限性包括：对极新生成器的适应性仍有限，门控与专家选择可能增加计算成本，且仅在 2D 图像上验证，缺乏对视频或三维场景的评估。

---

## 78. Reinforcement Learning-Guided Evolutionary Policy Optimization for Preference-Adjustable Heterogeneous Agile Earth Observation Satellite Scheduling

**arXiv ID:** 2608.24470 | [PDF](https://arxiv.org/pdf/2608.24470v1)

**作者:** He Wang `[一作]` (Harbin Engineering University), Liang Li `[通讯]` (Harbin Engineering University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出一种基于强化学习指导的进化策略优化框架，用于解决异构敏捷地球观测卫星排程问题。

**💡 创新点**

创新点在于将任务分配作为间接编码，结合解码器的等价成本评估，构建可调权重的可解释效用，并通过在线Actor‑Critic学习仅选择高层搜索算子模式，保持搜索空间小而可控。

**🔧 技术方法**

采用强化学习（Actor‑Critic）、进化算子（交叉、变异、局部改进等）、解码器、等价成本评估与权重化效用。

**📊 数据集**

实验使用六个人工构造的异构AEOS场景（任务数100–350，卫星数5–12），任务属性随机生成。

**📈 对比分析**

与MemeticEA、ALNS、AESSPSO、GWO等基线在相同评估预算下比较，RLOSMEA在所有场景中获得更高的加权效用和更稳健的收敛表现。

**⚠️ 局限性**

局限在于仅考虑静态任务；对动态、突发任务或云遮挡等不确定性处理不足，且RL学习过程对计算开销有一定影响。

---

## 79. Towards LLM-Enhanced Android Taint Analysis

**arXiv ID:** 2608.24269 | [PDF](https://arxiv.org/pdf/2608.24269v1)

**作者:** Nicholas Miazzo `[一作]` (University of Padova), Eleonora Losiouk `[通讯]` (University of Padova)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用大型语言模型（LLM）在安卓应用中进行流氓追踪，构建了基于模型上下文协议（MCP）的交互式分析代理；

**💡 创新点**

创新点在于首次将开箱即用的LLM作为独立推理引擎，绕过传统静态分析所需的手工框架建模，通过迭代代码探索实现对敏感数据流的自动推理；

**🔧 技术方法**

核心技术包括MCP协议、JADX与Ghidra的LLM插件、代理式提示（prompt）与多轮推理；

**📊 数据集**

实验使用DroidBench基准（190个案例）以及从AndroZoo抽取的5个真实应用；

**📈 对比分析**

与FlowDroid对比，LLM在DroidBench上实现了更高的召回率，F1分数提升至约0.70（FlowDroid约0.52），在ICC、隐式流、反射等难点类别表现尤为突出；在真实应用中发现17条额外泄露流，16条被验证为真阳性；

**⚠️ 局限性**

局限性包括LLM易产生幻觉与非确定性、需多次运行以筛选结果、对比仅基于默认FlowDroid配置、缺乏大规模真实标签数据以及计算成本较高。

---

## 80. Selective Regenerative Decoding: Trajectory-Level Intervention for Inference-Time Reasoning

**arXiv ID:** 2608.24338 | [PDF](https://arxiv.org/pdf/2608.24338v1)

**作者:** Sophia Xiao Pu `[一作]` (University of California Santa Barbara), Arshit Gupta `[通讯]` (Amazon Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了选择性再生解码（SRD）框架，在推理阶段对生成的推理轨迹进行路由、保留、细化或丢弃，并只重写质量下降的后缀。

**💡 创新点**

创新点在于：① 通过段级干预取代整体轨迹的全局接受/拒绝，节省计算；② 证明在合理假设下相较于拒绝采样，样本效率提升1.28–1.36倍；③ 通过局部再生实现无目标模型的细化。

**🔧 技术方法**

使用的技术包括：基于奖励模型的分段路由（阈值划分为 Keep/Refine/Discard）；目标温度提升的局部重生成；基于排名的归一化分数；以及理论分析证明与实验验证。

**📊 数据集**

使用的数据集有四个：MATH500（数学推理）、GPQA Diamond（科学问答）、HotpotQA（多跳推理）和AlpacaEval（指令跟随）。

**📈 对比分析**

与传统解码方法（单样本、Best-of-N、Speculative Rejection）以及奖励引导的显式采样进行比较。SRD 在相同或更低的 token 预算下，准确率与 Best-of-N 相当或更好，在低计算预算下明显优于 Speculative Rejection；在四个任务上都展现了更优的准确-计算权衡。

**⚠️ 局限性**

局限性包括：① 依赖奖励模型的质量，误差或偏差会导致不当细化；② 固定阈值与启发式边界选择可能不适用于所有任务；③ 需要额外的生成与奖励计算，可能增加推理延迟和实现复杂度。

---

## 81. Amortized Set Prediction for Inverse IFS Reconstruction from Density Maps

**arXiv ID:** 2608.24175 | [PDF](https://arxiv.org/pdf/2608.24175v1)

**作者:** Yutaka Yamaguti `[一作]` `[通讯]` (Fukuoka Institute of Technology), Yutaka Yamaguti (Fukuoka Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并训练了一种一次性推理的集合预测网络，用于从密度图估计反向IFS参数，并通过少量梯度步骤进一步优化。

**💡 创新点**

将反向IFS逆问题转化为可交换集合预测，并以重建误差为评价指标；利用已知渲染器实现自监督训练与无标签推理；证明并利用非可辨识性来构建训练代理。

**🔧 技术方法**

使用可分辨差分渲染器、Hungarian匹配损失、卷积残差网络、Amortized inference、少量梯度微调，以及点云Chamfer、Hausdorff和覆盖率等度量。

**📊 数据集**

通过合成的IFS样本生成的密度图作为训练和验证数据；在MNIST和Fashion‑MNIST图像上评估对真实图像的迁移。

**📈 对比分析**

与传统随机起点每图梯度优化（Tu等）在相同或两倍时间预算下进行比较，在密度重建指标上取得显著更优、速度提升12–2600倍；在占用率指标下表现略逊于传统优化。

**⚠️ 局限性**

受限于不可辨识性导致无法唯一恢复参数；仅适用于固定数量、仅正向映射且概率按行列式决定的IFS；对结构大变异的外域性能下降；未处理可变选择概率或反射映射。

---

## 82. From Triage to Discharge: A Survey of NLP Tasks, Methods, and Open Challenges in the Emergency Department

**arXiv ID:** 2608.23627 | [PDF](https://arxiv.org/pdf/2608.23627v1)

**作者:** Dipankar Srirag `[一作]` (University of New South Wales), Padmanesan Narasimhan `[通讯]` (University of New South Wales)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对急诊科（ED）中自然语言处理应用进行系统综述，涵盖分诊、诊断与处置三阶段任务、模型架构与评估方法。

**💡 创新点**

首次聚焦ED专属，构建统一任务框架，评估从传统考试式基准向开放式多模态、流程化任务转变，并指出部署与评估缺口。

**🔧 技术方法**

梳理了预训练Transformer、LLM、检索增强、代理系统、知识驱动、强化学习等多种技术。

**📊 数据集**

主要参考MIMIC系列、MIMIC-CXR、TriageSim、MC-BEC、MEDEC、ACI-Bench、MedJourney、MediQ、DDxGym等英文/中文ED相关语料。

**📈 对比分析**

通过比较准确率、ROUGE、BLEU、概念覆盖率等自动指标发现多数模型在开放式任务上低于考试式基准，且多未包含临床评估，整体性能差异显著但难以直接对比。

**⚠️ 局限性**

受限于单机构单语言数据、缺乏实时临床验证、评估偏自动化指标、缺少实际部署与隐私/偏见考量。

---

## 83. Beyond Executable Models: The Pufibara Agent Harness and the Modelica Agent Workflow Benchmark for Physical System Modeling

**arXiv ID:** 2608.23653 | [PDF](https://arxiv.org/pdf/2608.23653v1)

**作者:** Zizhe Wang `[一作]` `[通讯]` (Technische Universitaet Dresden), Zizhe Wang (Technische Universitaet Dresden)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了 Pufibara 代理机具，用于在 Modelica 环境中保持工程状态、绑定执行/仿真证据，并在 232 题（修复、生成、调优）上提供端到端的代理工作流评测。

**💡 创新点**

创新点：① 通过持久化工程义务账本和候选绑定证据，保证在迭代过程中工程约束不被遗忘；② 明确提交动作并与候选身份关联，避免仅靠可执行性即刻提交；③ 采用源代码根植的任务合成方式，确保任务既真实又不易被 LLM 记忆直接解答；④ 设计完整的 Benchmark 与评估契约，将官方判定与代理内部判定分离。

**🔧 技术方法**

技术：大型语言模型（DeepSeek v4 Flash、Claude Sonnet 5）+ Pufibara 代理机具；Modelica 运行时与 Model Context Protocol；模拟驱动反馈、逻辑 token 计数与运行时间监控；评估契约脚本。

**📊 数据集**

数据集：232 题，来源于 140 公共模型、15 组合公共库组件、77 内部自创参考模型；题型分布：132 修复、50 生成、50 调优；涵盖电、热、力、控制等多个物理域。

**📈 对比分析**

比较方法：在相同 LLM 后端、相同任务集、相同 Modelica 环境下，分别运行 Pufibara 与 Claude Code 代理机具；测量通过率、逻辑 token 用量、顺序运行时间。结果显示 Pufibara 在所有工作流中通过率更高（最高 202/232），逻辑 token 用量下降 76.4–82.5%，运行时间下降 6.1–58.4%。

**⚠️ 局限性**

局限性：每个任务仅跑一次，未检验多次随机性；未将贡献拆解到单一机制；可能受 LLM 预训练数据影响；Benchmark 仅针对 Modelica，未覆盖其他建模语言；评估契约虽然独立但仍有限的情景，非形式化验证。

---

## 84. When "Must" Becomes "Maybe": Constraint Weakening in LLM Agent Workflows

**arXiv ID:** 2608.24569 | [PDF](https://arxiv.org/pdf/2608.24569v1)

**作者:** Yiheng Sun `[一作]` (Shenzhen University), Yifan Yuan `[通讯]` (Shenzhen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究LLM代理在多角色工作流中，语言中介转化（如摘要、计划、记忆等）如何影响已确定的“绑定状态”（即未解决的条件对后续动作的约束）的运作

**💡 创新点**

提出并验证了“操作性状态保持”（operational state preservation）这一新估计指标，区分语义可用性与对行动的实际约束；并发现仅压缩或变换文本并不能保证绑定状态保持

**🔧 技术方法**

使用对比实验、人工标注评测器、固定工件修复、终端端点验证等技术，系统地测量不同语言转化对绑定状态保持与禁止动作的影响

**📊 数据集**

构造了1,772条合成企业任务样本，涵盖14种LLM模型、5种转化族、3种动态轨迹，共计1,296个实验实例

**📈 对比分析**

与直接保持格式（direct-preservation control）对比，发现压缩、合并计划、所有权推迟、先例替换等转化会导致80%~97% 的绑定状态失效，而修复所有四个字段可恢复到100%，终端验证可在不修复工件的情况下将禁止动作率降至0%

**⚠️ 局限性**

局限在于仅研究合成任务的单一“安全阻断器”类，缺乏真实任务验证；评测器对所有权字段的一致性低，可能影响结论；转化对不同LLM模型的差异尚未完全解释

---

## 85. Design-to-Plan: A Large Language Model-Based Multi-Agent Framework for Manufacturing Process Planning from 3D CAD Models and 2D Engineering Drawings

**arXiv ID:** 2608.24039 | [PDF](https://arxiv.org/pdf/2608.24039v1)

**作者:** Muhammad Tayyab Khan `[一作]` (Nanyang Technological University), Seung Ki Moon `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于大型语言模型的多代理系统（Design-to-Plan），实现从3D CAD模型和2D工程图自动生成可执行的制造工艺计划，并提供完整的可追溯报告。

**💡 创新点**

创新点：①将LLM作为交互式推理代理，配合Deterministic模块实现结构化信息提取与推理；②引入2D–3D上下文融合机制，解决设计信息与工艺规则的语义对齐；③采用多源知识库与并行/顺序ReAct架构，实现高效、可追溯的知识检索；④提供端到端评估框架，展示并行和顺序模式的质量–效率折中。

**🔧 技术方法**

核心技术包括：大型语言模型（GPT‑4o、GPT‑4o‑mini）+ReAct推理；图神经网络（GCNN）用于CAD特征识别；视觉‑语言模型（VLM）、YOLO、Donut用于图纸解析；结构化知识库（SQL、图数据库、规则引擎）与RAG；多代理异步通信与状态管理；参数计算公式与工具库。

**📊 数据集**

使用的实验数据集：①300个基准案例（拆分为100+100+100用于知识检索、工艺排序、刀具选择）；②20个真实CAD–图纸对，用于上下文融合评估；③150,000个合成CAD模型训练GCNN；④多种公开或内部数据（材料数据库、工艺规则、工艺模板）作为知识源。

**📈 对比分析**

对比方法：顺序ReAct vs 并行ReAct；与传统规则/模板工艺规划做基准。评估指标包括Tool F1、成功率、Jaccard/τ一致性、Token使用量、冲突检测得分。结果显示：并行模式实现100%成功率、工具F1>0.95、Token减少60‑68%；顺序模式在严重违规/约束调优上表现更好，但Token多。整体系统在300例端到端实验中保持高成功率与可追溯性。

**⚠️ 局限性**

局限性：①对稀有或非标准刀具、材料的覆盖不足；②冲突检测的解析率不高，主要受限于知识库完整性；③评估依赖单一参考序列，缺乏多重专家评审；④系统对极端不完整或模糊输入仍易产生不确定映射；⑤LLM对数值精度和结构化信息的hallucination风险；⑥部署成本受模型规模和多源检索频率影响。

---

## 86. Giga-Embeddings: Mixture-of-Experts Encoders for High-Throughput Text Embeddings

**arXiv ID:** 2608.23806 | [PDF](https://arxiv.org/pdf/2608.23806v1)

**作者:** Egor Kolodin `[一作]` (MIPT), Fyodor Minkin `[通讯]` (MIPT)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Giga-Embeddings系列，包含10B稀疏Mixture‑of‑Experts编码器、3B密集模型和480M蒸馏模型，用于生成高质量、可高吞吐量的文本嵌入。

**💡 创新点**

创新点包括：①稀疏激活的10B MoE模型，仅在每个token激活约1.8B参数；②维度无关的相似度分布蒸馏方法，可在教师和学生尺寸不同的情况下训练高效小模型；③开放源码发布全系列模型。

**🔧 技术方法**

技术主要有：Mixture‑of‑Experts稀疏激活、双向Transformer改造、均值池化、对比学习（InfoNCE）、相似度分布蒸馏（KL）以及多阶段训练（预训练→检索微调→多任务微调）。

**📊 数据集**

使用了公开的MTEB、MMTEB、ruMTEB等评测数据集，训练数据包含公开与非公开大规模对比学习数据。

**📈 对比分析**

与外部基线（如Qwen3‑Embedding、F2LLM、Gemini Embedding等）比较，10B‑A1.8B模型在英语、俄语、多语种、代码四大评测上均表现最佳，检索吞吐量比3B模型高25%，比外部系统快1.56–2.65倍。480M蒸馏模型在参数量上比FRIDA小42%，但在俄语MTEB上取得70.98分。

**⚠️ 局限性**

局限性包括：单次跑分缺乏置信区间；吞吐量测评只在特定vLLM环境下完成，未单独评估稀疏激活效果；训练混合包含非公开数据，难以完全复现；MTEB聚合分数可能掩盖任务/语言差异。

---

## 87. An HPC Approach to Accelerate Tensor Decompositions

**arXiv ID:** 2608.24307 | [PDF](https://arxiv.org/pdf/2608.24307v1)

**作者:** Markus Hellgren `[一作]` (Uppsala University), Roman Iakymchuk `[通讯]` (Uppsala University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了 Jacobi‑type 张量分解算法，并在单个 NVIDIA H100 GPU 上实现了可处理任意阶稠密张量的 CUDA 加速版本。

**💡 创新点**

创新点包括：将第三阶 Jacobi 方法推广到任意阶张量；提出高效的 GPU 并行实现，消除 pivot 检查导致的瓶颈；证明在禁用 pivot 检查时仍能收敛，从而进一步提升速度；在 3D–9D 的大规模张量上实现了 2 位数至 3 位数的加速（高于 MATLAB 的 200‑倍以上）。

**🔧 技术方法**

采用的技术主要有：CUDA 编程、cuBLAS/cuSOLVER/CUB 库调用、张量展开与模式乘、并行 pivot 集（批量三角函数计算）、常量内存优化、GPU 设备内存管理与一次性复制、一次性计算张量步幅、以及基于一维线程块的负载均衡。

**📊 数据集**

实验使用了人工生成的可对角化随机张量，尺寸 N 取 8、16、32、64、128、256，阶数 D 取 3、4、5、6、7、8、9；所有实验均在 NVIDIA H100 GPU 上完成。

**📈 对比分析**

对比方法：将 CUDA 版本与 MATLAB 参考实现以及单线程 C 版本在 10 次完整迭代循环的整体运行时间（包括内存分配、数据传输和内核调用）进行对比。结果显示：在不使用 pivot 检查时，CUDA 版本对 3D 张量的加速分别为 1.8×、11.3×、551×；在更高阶张量上，CUDA 版本相较 MATLAB 速度提升超过 200 倍，且随阶数呈指数增长、随维度线性增长。

**⚠️ 局限性**

局限性：目前实现仅支持稠密、非对称张量；未针对对称或稀疏张量做专门优化；仅在单 GPU（H100）上测试，未验证多 GPU 扩展；pivot 检查虽然可以禁用，但在更大规模或不同结构张量上的收敛性和数值稳定性仍待进一步评估。

---

## 88. ResiSpec: Enhancing Multi-Candidate Speculative Sampling via Residual Distribution Shaping

**arXiv ID:** 2608.24411 | [PDF](https://arxiv.org/pdf/2608.24411v1)

**作者:** Zhi-Kai Chen `[一作]` (Nanjing University), Han-Jia Ye `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ResiSpec框架，解决多候选推断中的Residual Drift问题。

**💡 创新点**

通过在验证阶段引入代理分布k(x)重新塑造残差分布，保持与草稿模型高置信区域对齐。

**🔧 技术方法**

利用代理分布重塑、残差分布归一化、树结构多候选推断等技术。

**📊 数据集**

以Llama-2-7B为目标模型、JackFram/Llama-68M为草稿模型，在CNN/DM、OpenWebText、C4等数据集上进行实验。

**📈 对比分析**

与SpecInfer、Sequoia、EAGLE等基线对比，接受长度提升至1.86×、吞吐量提升至1.68×，KL偏差约为1e-10，显示性能显著提升且分布保持一致。

**⚠️ 局限性**

局限在于代理分布设计依赖目标与草稿分布支持相近，极端长尾或支持差异大的场景下残差预算可能趋近零，导致效果受限。

---

## 89. Inter-dimension Dependence for Multi-Dimensional Evaluation of Open-Ended Text

**arXiv ID:** 2608.23783 | [PDF](https://arxiv.org/pdf/2608.23783v1)

**作者:** Haoyuan Li `[一作]` (University of North Carolina at Chapel Hill), Snigdha Chaturvedi `[通讯]` (University of North Carolina at Chapel Hill)

**通讯引用:** 1482 | [OpenAlex ID](https://openalex.org/A5041254552)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CorrGap 量化 LLM 判别器在多维评估中跨维度依赖，并提出 DimCheck 通过迭代删去无关证据降低该依赖。

**💡 创新点**

创新点在于（1）构造基于相关性差异的 CorrGap 指标，可客观评估跨维度依赖；（2）设计 DimCheck，利用多步 COT 编辑逐步剔除非目标维度信息，显著提升评估独立性。

**🔧 技术方法**

使用 LLM 生成 Chain‑of‑Thought (COT) 与分数、统计相关性分析、交叉验证、Permutation Test、训练小型 LLM 进行 COT 编辑，框架包含 G‑Eval、Analyze‑Rate 与 Debate。

**📊 数据集**

实验数据集包括 SummEval（新闻摘要）、Topical‑Chat（知识驱动对话）、Hanna（叙事生成）和 OpinSumm（意见摘要），覆盖 4 个任务和多维度评价。

**📈 对比分析**

在 9 种 LLM（Llama3、Qwen3、Gemma3、Ministral、GPT‑5.4、M‑Prometheus）与 4 个框架中，CorrGap 显示跨维度依赖普遍存在；DimCheck 在 Kendall τ 与 CorrGap 上均优于原始 Analyze‑Rate 与多种基线，整体分数提升 1–3 点，且小型训练 LLM 也可近似大模型效果。

**⚠️ 局限性**

局限性：仅针对点对点评分；仅适用于生成 COT 的评估框架；未探索对比、列表式评估等更广泛的评估范式；在极端文本质量或少量数据场景下效果待验证。

---

## 90. Taming Visual Neglect: A Variational Information Bottleneck Framework for Adaptive Attention in Multimodal In-Context Learning

**arXiv ID:** 2608.23570 | [PDF](https://arxiv.org/pdf/2608.23570v1)

**作者:** Kaito Tanaka `[一作]` (SANNO University), Aya Nakayama `[通讯]` (SANNO University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了VIB-ICL框架，用信息瓶颈原理解释视觉上下文在多模态上下文学习中的作用，并通过CMIG量化视觉信息增益。

**💡 创新点**

创新点在于将交叉模态信息增益（CMIG）引入多模态ICL，证明视觉忽略是信息瓶颈最优解，并给出注意力再分配闭式原则。

**🔧 技术方法**

使用变分估计CMIG、信息瓶颈优化、注意力再分配以及演示选择等技术。

**📊 数据集**

在VL-ICL Bench、TrueMICL、MMICL、COCO-FewShot、CausalVLBench等五个基准数据集上验证。

**📈 对比分析**

与Vanilla ICL、M^2IV、DARA、CAMA、MMICL等基线对比，VIB-ICL平均提升约3.4%准确率，最高提升4.7%，并显著减少示例数约35%。

**⚠️ 局限性**

局限在于对线性注意力假设、变分CMIG估计误差、对不同模型和模态的泛化仍需进一步验证。

---

## 91. MoRF-AST: Calibrated Probabilistic Virtual Sensing for Structural Monitoring under Changing Operating Conditions

**arXiv ID:** 2608.24531 | [PDF](https://arxiv.org/pdf/2608.24531v1)

**作者:** Wingho Feng `[一作]` (Tsinghua University), Chen Wang `[通讯]` (Tsinghua University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对结构虚拟传感中的全域响应不确定性推断，提出了MoRF-AST方法，在有限且噪声的传感器测量下实现全域响应的概率重建，并在运行时通过历史传感器记录自适应校准后验分布。

**💡 创新点**

创新点包括：① 在标准化模态坐标下构造解析高斯先验，并仅学习其残差的条件流匹配，显著提高数据效率；② 通过上下文条件化的扩散传输(AST)利用历史测量估计全局尺度，使用截断感知门控和Bures‑Wasserstein对称映射实现均值保持的后验扩散或收缩；③ 明确阐明后验扩散在收缩与扩张两种情况的相容性，解释了为何仅对MoRF有效。

**🔧 技术方法**

核心技术：主成分分析（POD）提取模态基，解析高斯后验中心与协方差；条件流匹配（Flow Matching）训练残差流；Bures‑Wasserstein 传输实现均值保持的协方差映射；历史上下文尺度估计（多尺度对数网格、最大似然估计）和门控机制；Heun ODE求解器用于后验采样。

**📊 数据集**

使用的基准数据集为桥面板模拟数据：128×128 网格的四层车道桥面板，生成 8000 条训练事件与 1000 条验证事件，采用 8 个交通流量场景（源混合及 7 个偏移域）和 40 个传感器，覆盖不同交通负载与条件。

**📈 对比分析**

与三种基线（解析高斯参考、深度集成、Gaussian Mixture Residual Regression）进行比较。MoRF 在 NRMSE 上达到 7.20%（低于 16% 的直接流模型），MoRF‑AST 在 ACE（平均绝对校准误差）上从 0.0535 降至 0.0236（下降 55.9%），且保持相近的 NRMSE（7.72%）。在 8 个偏移域中，AST 仅在需要收缩或扩张的域显著改善覆盖率，未对其它基线产生正面效应。

**⚠️ 局限性**

局限性：① 依赖源域的 POD 结构和观测矩阵，若结构或传感器发生重大变化需重新训练；② 仅通过单一尺度参数调整后验扩散，无法处理显著的各向异性或高阶分布偏移；③ 对尾部极端风险评估的改进有限，需要进一步的极值分析；④ 在传感器噪声严重或历史数据不足时，尺度估计不稳，导致后验校准失效。

---

## 92. When Does Self-Supervised Pretraining Help Tabular Models? A Study of Label Scarcity and Missing Data

**arXiv ID:** 2608.24381 | [PDF](https://arxiv.org/pdf/2608.24381v1)

**作者:** Sahand Mazrouei `[一作]` `[通讯]` (Kharazmi University), Sahand Mazrouei (Kharazmi University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对14个OpenML表格分类任务，在极端标签稀缺（1%–20%）和测试时缺失（MCAR、MNAR）条件下，评估了mask‑and‑recover自监督预训练与从头训练及经典树模型的性能差异。

**💡 创新点**

首次系统性揭示了在自监督预训练中出现的“清洁数据优于缺失数据”悖论，并强调在小样本表格SSL研究中必须使用多种seed平均来消除噪声、避免单seed误导。

**🔧 技术方法**

使用双分支MLP编码器实现mask‑and‑recover预训练，并加入一致性损失、分组遮蔽以及可选的mask‑aware训练，随后用AdamW对少量标签进行微调。

**📊 数据集**

采用14个OpenML公开分类数据集，涵盖不同样本量、特征类型和原生缺失率，构成多样化评估基准。

**📈 对比分析**

通过AUC‑ROC对比5%/10%标签稀缺、MCAR+30%缺失、MNAR结构缺失等情形，SSL平均AUC略高于从头训练，且与随机森林差距仅0.006；与VIME/SCARF/SubTab同构网络的差异在统计上不显著。

**⚠️ 局限性**

主要限制包括预训练收益在高缺失数据集上不稳定、训练成本显著高于树模型、单seed实验易产生虚假结论、以及多比较校正后多数对比差异不显著；缺失机制与预训练缺失一致性问题仍未得到充分解决。

---

## 93. Algebraic Characterizations for Minors of Finite Graphs via Flow Transformation Monoid Division and Embedding

**arXiv ID:** 2608.24239 | [PDF](https://arxiv.org/pdf/2608.24239v1)

**作者:** Amena Assem `[一作]` (University of Waterloo), Chrystopher L. Nehaniv `[通讯]` (University of Waterloo)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了有限图的流单子（flow monoid），并给出了三条关于图连通性与常数映射、图下标与流单子划分、以及图下标与嵌入式流单子之间的等价性定理。

**💡 创新点**

创新点在于：① 用流单子中的常数映射来完全判定图的连通性；② 将图下标关系用流单子划分（division）和代数交叉条件来刻画；③ 强化到嵌入式流单子，即在原图的流单子中构造一个局部幺元的子单子，使其在一个子集上与下标图的流单子同构，从而实现对下标的精确代数识别。

**🔧 技术方法**

主要技术包括：半群/单子理论（尤其是idempotent、划分、同态、同构）；图论中的下标、连通子图、边收缩；以及对流单子生成元（元素级合并）与图边之间的对应关系。

**📊 数据集**

论文不涉及实验数据或数据集，研究全部为理论推导与证明。

**📈 对比分析**

由于是纯理论工作，没有与其他方法进行实验对比；性能评估以定理证明的有效性与可计算性（如常数映射的构造复杂度为O(|V|+|E|)）为准则。

**⚠️ 局限性**

局限性：所给出的代数条件在实际应用中需要先构造下标或子图的连通分块，算法实现上可能存在组合爆炸；此外，文中仅讨论无向图，未扩展到有向图或带权图；最后，理论结果虽完整，但缺乏对大规模图实例的实验验证。

---

## 94. From Gradient-Boosted Trees to Deep Recommenders: Practical Lessons from Migrating a Production Customer Support Recommender

**arXiv ID:** 2608.24132 | [PDF](https://arxiv.org/pdf/2608.24132v1)

**作者:** Sonia Sharma `[一作]` (Intuit), Andrew Mattarella-Micke `[通讯]` (Intuit)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

迁移生产客户支持推荐系统从CatBoost树模型到深度对比式二元推荐模型，并在保证推荐质量不下降的前提下完成迁移。

**💡 创新点**

创新点包括采用对比式二元学习、结构化负采样、对比损失、注意力池化、噪声注入和双下降容量调优等技术，实现可动态处理新SKU和捆绑的推荐模型。

**🔧 技术方法**

使用了对比式二元分类、DeepFM/两塔网络、负采样、对比损失、注意力池化、噪声注入、双下降分析以及量化与批处理推理优化等技术。

**📊 数据集**

使用真实客服会话数据（2025‑01~2026‑06），包含会话文本、结构化特征、9种产品，按时间划分训练/验证/测试。

**📈 对比分析**

在同一固定正负样本集上对比原CatBoost基线，使用F1、ROC‑AUC等指标评估，DeepFM注意力池化在ER/GZ上超过基线，整体实现首发时不下降、后期提升。

**⚠️ 局限性**

仍在CL阶段略逊于基线，未做公平性/长期价值评估，对新SKU的泛化验证不足，且对大规模产品目录扩展未评估。

---

## 95. More Motion Is Not Always Better Motion: Corpus Composition Governs Whether Augmentation Helps SMPL-Based Parkinsonian Gait Severity Estimation

**arXiv ID:** 2608.23730 | [PDF](https://arxiv.org/pdf/2608.23730v1)

**作者:** Michael Caiola `[一作]` (Credence), Andrew C. Weitz `[通讯]` (Credence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本研究通过使用三种不同规模、冻结的2D-3D姿态提升编码器，构建了一个集成模型，可从SMPL运动序列预测MDS-UPDRS 3.10步态严重程度。

**💡 创新点**

创新点在于发现外部运动数据能否提升模型取决于其是否包含步速对比，而非数据量大小；并揭示对冻结编码器的任何改动都会导致性能下降，且下降幅度与改动程度正相关。

**🔧 技术方法**

技术包括：SMPL运动表示、基于姿态提升的冻结编码器、多视角投影、固定MLP头、集成平均、transductive centering、以及与手工特征（角度统计与临床步态参数）对比。

**📊 数据集**

使用的数据集包括CARE‑PD（带标签四组、共2950步态）、WearGait‑PD（惯性测量）、BEDLAM（合成运动）、GAVD、公开YouTube步态片段以及由CARE‑PD无标签组重建的SMPL模型。

**📈 对比分析**

通过与单编码器、不同编码器尺寸组合、以及手工特征的对照实验进行比较，隐藏测试集macro‑F1达到0.58；单编码器最高0.53，手工特征仅0.41/0.34，说明冻结编码器集成显著优于传统特征。

**⚠️ 局限性**

局限性包括：离线留一组交叉验证无法准确预测隐藏测试表现；外部数据因尺度误差或缺少助行器信息导致不适合作为训练；类3（使用助行器）难以从SMPL模型中捕捉；评标仅由单名非临床评估者完成，可能存在主观偏差；数据来源多样性有限，难以覆盖更广泛的站点与重建差异。

---

## 96. Aura: Dynamic Intra-Turn Emotion-Aware Adaptation of Large Language Model Responses

**arXiv ID:** 2608.24224 | [PDF](https://arxiv.org/pdf/2608.24224v1)

**作者:** Rachel Schuchert `[一作]` (ETH Zürich), Christian Holz `[通讯]` (ETH Zürich)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a4b10f5d-130b-4e77-9367-6469ec621899` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了Aura框架，实现LLM在对话中实时感知用户面部情绪并在生成过程中动态调整回应。

**💡 创新点**

创新点在于将面部情绪感知与POMDP策略推理相结合，并通过LoRA动态激活不同语言生成策略，实现了在单一回复中进行中途调整的能力。

**🔧 技术方法**

技术手段包括：EmoAffect‑Net + ResNet‑50 + LSTM 进行情绪估计；EMA平滑与突发检测；POMDP（状态为Engaged/Confused/Frustrated/Bored，动作为Reframe/Clarify/De‑escalate/Pace/Simplify/Base）进行策略选择；LoRA（Q‑LoRA 4‑bit）在冻结的 Llama‑3‑8B‑Instruct 上实现多种生成策略；以及合成语料库训练POMDP参数。

**📊 数据集**

使用的数据集包括 AffectNet（情绪识别预训练）、DAiSEE（四种交互状态标签）、合成的 GPT‑4o 生成的 24,929 次会话（131,283 轮）用于训练 POMDP，并在 20 名受试者的 18 个二分类任务中进行评估。

**📈 对比分析**

通过在实验室内进行 20 名受试者的 within‑subject 对比（Aura vs Llama‑3 vs GPT‑4o），评估指标包括：归一化学习增益、任务准确率、交互时长、主观满意度、易用性等。Aura 在学习增益上最高（0.59±0.22），交互时长比 GPT‑4o 与 Llama‑3 减少约 21%，并在满意度、清晰度与参与度上表现更佳。

**⚠️ 局限性**

局限性：依赖摄像头面部表情，可能受文化、年龄、性别或面部运动障碍影响；POMDP 的状态转移与奖励来自合成数据，真实世界动态可能不足；LoRA 只提供离散策略，缺乏连续化的细腻调节；实验仅覆盖二分类任务，未验证在开放式或多步任务中的效果。

---

## 97. Adoption Telemetry: Measuring Enterprise AI Adoption from Production Signals

**arXiv ID:** 2608.23617 | [PDF](https://arxiv.org/pdf/2608.23617v1)

**作者:** Damon A. Young `[一作]` `[通讯]` (PolyWise Partners), Damon A. Young (PolyWise Partners)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种称为“采用遥测（adoption telemetry）”的测量方法，结合部署评估门、生产遥测和变更管理阶段化模型，构建出一种可从企业 AI 系统的使用事件中直接计算用户采用进程的工具。

**💡 创新点**

创新点在于：①将四大传统测量体系（代理评估、使用仪表盘、产品分析、变更管理）统一到单一解释层；②提出 NANTE 五阶段模型（Notice、Attempt、Navigate、Transform、Embed），并公开阈值供同行质疑和校准；③提供开源实现，使计算过程可审计、可复现，并以可视化报告展示诊断与干预建议。

**🔧 技术方法**

技术方法包括基于事件流的规则推理（使用会话、轮数、任务成功率等指标）、阈值判定的阶段划分、停滞点检测与失败签名识别，以及基于规则的干预映射；实现采用 Python、SQLite 与开放配置文件实现。

**📊 数据集**

实验数据仅为模拟生成的合成群体，构造六种行为模式（健康、浅板、认知缺口、能力缺口、巩固衰退、冠军依赖），每种模式包含 167 名预配用户、149 天观测窗口；未使用真实企业数据。

**📈 对比分析**

在合成数据上，系统能够准确区分所有六种模式，正确识别停滞点与四类失败签名，且在健康与浅板两组间实现显著区分；与传统使用仪表盘相比，NANTE 能区分深度与表面活跃度，性能表现良好。

**⚠️ 局限性**

局限性主要有：阈值未经过真实案例验证，构造的合成模型可能与实际使用模式差异；事件模式仅使用极简架构，无法直接捕捉任务类型、结果可靠性和代理决策（disposition）等关键信息；仅适用于人发起的 AI 交互，无法涵盖全自动化工作流；未考虑测量诱导效应与多工具交叉使用等现实情况，需要后续真实数据校准与扩展。

---

## 98. Concept-Guided Exploration: Building Persistent, Actionable Scene Graphs

**arXiv ID:** 2608.23650 | [PDF](https://arxiv.org/pdf/2608.23650v1)

**作者:** Noé Zapata `[一作]` (University of Extremadura), Pablo Bustos `[通讯]` (University of Extremadura)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种概念优先（concept‑first）架构，让移动机器人通过异步概念代理（如房间、门）直接构建可执行的三维场景图，而非先生成密集的度量地图再叠加语义层；

**💡 创新点**

创新点包括：①异步概念代理协同构造场景图，打破传统的“先度量后语义”顺序；②层级约束传播机制，房间先验约束门的检测与定位；③局部度量帧的世界模型，避免全局坐标一致性负担；④基于 CORTEX 的分布式、事件驱动的共享工作内存实现高效并发与一致性；

**🔧 技术方法**

技术方法包括：基于 LiDAR 的点云预处理与 Hough 直线/角点检测、门检测的距离阈值与峰值筛选、GTSAM 因子图优化实现局部定位、CORTEX 代理与 WM 事件通信、LTSM 的局部图管理与预加载；

**📊 数据集**

实验数据集主要为 Webots 模拟环境中的多房间场景（10 房间）以及真实 Shadow 机器人在两个带门房间的实际测试；

**📈 对比分析**

方法在实验中能够在 4 小时内保持 15 cm 以内的定位误差，房间与门的几何误差均在几厘米以内，构建速度稳定，CPU/内存占用极低；相比传统基于全局占用栅格或密集点云的语义 SLAM，本文的概念优先方法在存储与计算上更轻量，且能即时生成可执行的场景图；

**⚠️ 局限性**

局限性包括：仅支持矩形房间与单一门概念；使用的检测算法较为简单，易受噪声与遮挡影响；缺乏强健的闭环检测与多机器人共享内存机制；未来工作需扩展到非矩形空间、深度学习感知、更多概念类别及分布式优化等。

---

## 99. More GPUs or a Smaller Cache? Tensor Parallelism versus KV Compression for Memory-Bound LLM Serving

**arXiv ID:** 2608.23962 | [PDF](https://arxiv.org/pdf/2608.23962v1)

**作者:** Srikanta Datta Tumkur `[一作]` (Massachusetts Institute of Technology and Vizuara), Raj Dandekar `[通讯]` (Massachusetts Institute of Technology and Vizuara)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过成本归一化的前沿图，系统对比了在同一模型、质量阈值和延迟目标下，添加 GPU（张量并行）与压缩 KV 缓存（低位量化与剔除）的两种内存缓解方案，并对两者在多种硬件、模型规模与批处理条件下的可行性与成本效益进行实验分析；

**💡 创新点**

创新点包括：①发现两种方案在任何匹配的内存缓解水平下不存在成本交叉点；②提出基于模型参数量与设备内存的“可行性阈值”作为决策边界；③提出从可行性角度出发的实用决策规则，解释了为什么在低于阈值时压缩更划算，而高于阈值时张量并行必不可少；

**🔧 技术方法**

技术手段包括：张量并行度 1/2/4/8 的权重与 KV 拆分与全归约通信建模；KV 缓存压缩（16/8/4 位量化 + keep‑ratio 1/0.5/0.25）；基于 Vidur 的可计量模拟器与精确内存算式；可行性方程式 2W/p_weights + B·m_kv ≤ M_dev(1‑μ)；成本/百万 token 归一化；敏感性分析（解量化开销阈值）；

**📊 数据集**

使用 Llama‑2 7B 与 70B 两种模型，实验覆盖 A100‑80GB、A40‑48GB、H100‑80GB 三种 GPU；合成工作负载 W1（2048+256）与 W2（3840+256），批量上限 128，最大上下文 4096 令牌；

**📈 对比分析**

比较方法为：在相同 KV 内存缓解（通过压缩或张量并行度）下，绘制成本/百万 token 与延迟（TTFT P99）的 Pareto 前沿；结果显示压缩在所有匹配的缓解水平下均低于张量并行（1.20×–1.89× 降本），张量并行仅在延迟上显著改善；在可行性阈值以上，成本曲线呈 U 形，最低成本位于略高于最小可行张量并行度；

**⚠️ 局限性**

局限性包括：未使用真实 GPU 采样，仅依赖已校准的模拟器；未评估压缩导致的精度下降，仅给出质量上限；未对低位量化解码时延进行建模，给出通过假设的阈值；假设 fp16 权重量化，未考虑权重量化后阈值位置；实验仅覆盖 Llama‑2 系列且上下文长度被硬编码为 4096，未覆盖更长上下文场景；

---

## 100. Paritok-4B: Intent-Conditioned Context Compression for Coding Agents

**arXiv ID:** 2608.24188 | [PDF](https://arxiv.org/pdf/2608.24188v1)

**作者:** Jiayu Shi `[一作]` (Paritok), Luzhuo Chen `[通讯]` (Paritok)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

训练了一个4B LoRA压缩器，将编程代理的上下文压缩到约四分之一，同时保持大部分关键代码信息。

**💡 创新点**

设计了“提取式”和“意图感知”两大承诺，保证保留准确字符串且按当前任务选择保留行，并使用五阶段数据管道将真实代理轨迹蒸馏给教师。

**🔧 技术方法**

采用Qwen3-4B-Instruct backbone，LoRA微调，教师蒸馏自gpt-4.1-mini，结构化[SEG]输出，段落级压缩和意图条件。

**📊 数据集**

利用67K条OpenHands代理轨迹（SWE-Rebench/Gym）作为训练集，300条SWE-bench Lite作为评估集。

**📈 对比分析**

与gpt-4.1-mini和gpt-5等大型模型做单轮压缩对比，压缩率仅25.7%，单轮解决率保持86.5%（或89.3%），与基线相当但消耗一半token；内部测试保持识别率≈96%。

**⚠️ 局限性**

主要缺点是下沉掉落率不足、识别率虽高但仍有约60%标识被丢弃，意图感知仅在行级而非段级，层级控制失效，未实现多轮代理成本评估，适用于Python，未覆盖其他语言。

---

## 101. Don't Just Listen, Try Planning: Graph-based Retrieval-Generation Agent for Long-form Audio Meeting Understanding

**arXiv ID:** 2608.24048 | [PDF](https://arxiv.org/pdf/2608.24048v1)

**作者:** Quanwei Tang `[一作]` (Soochow University), Guodong Zhou `[通讯]` (Soochow University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文构建了 LongAudioQA 长音频会议问答数据集，并提出了 GRGA 图检索生成代理模型来解决声学缺失与上下文碎片化问题。

**💡 创新点**

创新点在于将多维音频特征（语义、说话人、时间）建模为异构图，并通过规划、执行与自我反思的迭代流程实现精准检索与答案生成。

**🔧 技术方法**

使用的技术包括语音分段、强制对齐、说话人聚类、LLM 生成说话人属性、异构图构建、POMDP 规划、工具执行与反思。

**📊 数据集**

数据集来源于 AliMeeting、AMI Meeting Corpus 与 DailyTalk 三大公开会议语料，经过 LLM 辅助生成与人工校验得到的问答对。

**📈 对比分析**

与多种基线（端到端 Speech‑LLM、文本/音频 RAG 等）对比，GRGA 在事实、推理、时间、摘要与声学类问题上平均提升 10–30% 以上，尤其在长会话的推理任务中显著优于现有方法。

**⚠️ 局限性**

主要局限包括对 ASR 与分离的依赖导致错误传播、迭代推理导致推理延迟，以及对结构化会议场景的专业化，未验证对非结构化语音场景的泛化能力。

---

## 102. OPDSearch+: On-Policy Distillation with RL Refinement for Search-Augmented Reasoning

**arXiv ID:** 2608.24310 | [PDF](https://arxiv.org/pdf/2608.24310v1)

**作者:** Qinglin Ye `[一作]` (University of Chinese Academy of Sciences), Yiming Wang `[通讯]` (University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为OPDSearch+的两阶段框架，利用冻结的通用指令模型进行在线对齐蒸馏，并通过强化学习进一步精炼搜索增强推理能力；

**💡 创新点**

创新点包括：①使用前向KL在线蒸馏让教师重塑学生策略分布，使RL能够从更优的起点收敛；②完全消除高质量搜索轨迹收集和教师微调的成本；③对前向KL梯度方差与分布偏移进行理论分析，证明其稳定性；

**🔧 技术方法**

采用了前向KL在线蒸馏、基于搜索的策略梯度RL（GRPO）、重要性权重裁剪、KL正则化、E5检索器等技术；

**📊 数据集**

在NQ、TriviaQA、PopQA（单跳）以及HotpotQA、2WikiMultihopQA、MuSiQue、Bamboogle（多跳）七个基准上进行评估，训练集为NQ+HotpotQA（约170k）问答对；

**📈 对比分析**

与多种3B规模的基线（Search-R1、ReSearch、AutoRefine、StepSearch、GiGPO-Instruct等）以及非RL基线进行对比，OPDSearch+在7个基准上的平均EM为0.4402，单跳平均0.520，多跳平均0.456，分别比最佳3B基线提升13.1% HotpotQA和8.5% 2Wiki；

**⚠️ 局限性**

局限性包括：对教师模型规模和检索环境的依赖，前向KL裁剪可能引入偏差，扩展到更大模型需要更多算力，对极短或单跳问题的提升有限。

---

## 103. Evaluating Multiple LLM Generations with Validated Task Coverage

**arXiv ID:** 2608.24228 | [PDF](https://arxiv.org/pdf/2608.24228v1)

**作者:** Florian Le Bronnec `[一作]`, Rio Yokota `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出 VTC-Bench 这一基准，用以评估 LLM 在有限候选集（k 次生成）中产生的不同有用结果，定义了 Validated Task Coverage（VTC）指标，并在五个真实任务上进行验证；

**💡 创新点**

创新点在于：①从候选集角度而非单一输出评估 LLM，②提出 VTC 指标以量化不同有用结果的覆盖率，③设计了可自动、可重现的任务与验证映射，④系统性探讨了温度、思考、链式生成对覆盖率的影响；

**🔧 技术方法**

主要技术包括 LLM 生成、自动验证（RDKit、PatchEval、程序执行等）、映射到任务特定的有用结果、统计覆盖曲线、对比单输出质量与表面多样性；

**📊 数据集**

使用了五个数据集：ZINC250k（分子设计）、PatchEval‑Verified（仓库修复）、CodeContests（bug 发现）、AMIE 差异诊断案例、BRIGHT‑Pro（证据检索）等；

**📈 对比分析**

通过对四种模型（Qwen3.6‑27B、Qwen3.6‑35B‑A3B、GLM‑4.7、DeepSeek‑V4‑Flash）在不同温度、思考与链式生成设置下进行 24 组配置，比较 VTC 与单输出质量、表面多样性。结果显示：VTC 排名与单输出质量显著不同；温度升高往往提升 VTC，思考效果不稳定；链式生成并不能可靠提升覆盖率；不同任务表现差异明显；

**⚠️ 局限性**

局限性包括：①VTC 依赖任务特定的有效性与映射定义，可能忽略真实有用细微差异；②评估以固定尝试次数为基准，无法直接映射到 token 或计算资源；③依赖人工标注的参考目标，可能漏掉可行但未标注的答案；④仅在五个任务上验证，通用性待进一步探索。

---

## 104. Exact CVP Is NP-Complete for Principal Cyclotomic Ideals

**arXiv ID:** 2608.23828 | [PDF](https://arxiv.org/pdf/2608.23828v1)

**作者:** Jiaqi Liu `[一作]` (Academy of Mathematics and Systems Science), Yanbin Pan `[通讯]` (Academy of Mathematics and Systems Science)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文证明了在二维的幂次为 2 的循环同调环（cyclotomic rings）以及对应的全秩主循环理想（principal cyclic ideals）上，欧几里得精确最近向量问题（exact Euclidean CVP）是 NP‑完整的，并进一步展示了在这些固定理想族上进行预处理（CVPP）求解的多项式时间方案将导致多项式层级崩塌。

**💡 创新点**

创新点在于：① 构造了一个确定性的多项式时间归约，从 Exact Cover by 3‑Sets (X3C) 到主理想的 CVP ；② 通过将偶坐标映射到更大的循环同调环，实现了对主理想的“lift”，并保持距离尺度的精确放大；③ 在每个输入规模上预先确定唯一的主理想（固定族），仅将目标和阈值编码为 X3C 集合，从而在固定族上保持 NP‑完整性；④ 以此回答了 Micciancio 对循环理想上 CVP 的硬度及其固定族可否仍硬的未解问题。

**🔧 技术方法**

主要技术手段包括：利用 Sidon 集合构造唯一差分的指数；在多项式环中使用多项式卷积表示理想元素；设计特定的多项式（如 g(y)、T_μ(y)）来产生距离差；利用 Fourier 正交性证明 coefficient 与 canonical embedding 下距离只相差常数因子；以及通过 Chinese Remainder 定理实现理想的 lift 与距离放大。

**📊 数据集**

本文不使用实验数据集，而是基于理论构造。所有实例均由 X3C 问题的布尔矩阵（规模为 m）生成，随后通过算法产生对应的多项式、理想生成元、目标与阈值。

**📈 对比分析**

方法的比较完全基于计算复杂度。作者证明了在上述理想族上，exact CVP 属于 NP 且 NP‑难（即 NP‑完整），并且对搜索版本同样是 NP‑难。对预处理版本（CVPP）的多项式时间解法若存在，将导致 Σ₂^P = PH，即多项式层级收敛，这与目前主流的复杂度假设相矛盾。

**⚠️ 局限性**

局限性包括：① 仅针对幂次为 2 的循环同调环；② 归约构造涉及指数较大的多项式（维度 Θ(m⁷)），在实践中可能不具可扩展性；③ 证明依赖于 X3C 的 NP‑完整性，若后续发现更强的近似或随机化算法，结果可能需重新评估；④ 只给出了理论上 NP‑完整性的证明，并未提供具体实现或实验验证。

---

## 105. Memory Is Not Always Needed: Characterizing Conditional Memory in Scientific Reasoning

**arXiv ID:** 2608.23982 | [PDF](https://arxiv.org/pdf/2608.23982v1)

**作者:** Zhen Bi `[一作]` (Huzhou Normal University), Jungang Lou `[通讯]` (Huzhou Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在科学推理任务中，条件记忆的使用应具有选择性而非统一启用；

**💡 创新点**

提出知识边界感知路由器，利用输入前置特征决定何时、何处以及多大力度激活条件记忆；

**🔧 技术方法**

结合外部与内部路由技术，对Transformer层级与阶段注入记忆信号进行可控调节，并使用量化特征与交互规则；

**📊 数据集**

在生物学推理基准BioProBench（包含错误修正、步骤排序、协议问答）和化学推理基准ChemCoTBench（分子理解、编辑、优化）上进行评测；

**📈 对比分析**

与固定激活、随机匹配激活率及不使用记忆的基线对比，路由器在各任务上均实现了更高的准确率、召回率或特定指标，显著优于随机或全局激活策略；

**⚠️ 局限性**

记忆的收益高度依赖任务和输入，某些情况下仍会引入误导或干扰，且路由策略需要先验任务特定特征，限制了在未知任务上的通用性。

---

## 106. RePolicy: Reinforcement Learning for Safety-Policy Invocation in Agent Safeguards

**arXiv ID:** 2608.24275 | [PDF](https://arxiv.org/pdf/2608.24275v1)

**作者:** Houcheng Jiang `[一作]` (University of Science and Technology of China), Xiangnan He `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过强化学习学习安全策略调用，提供完整轨迹级评估的LLM代理安全防护模型RePolicy。

**💡 创新点**

将安全策略调用视为可调用的能力，利用强化学习和策略上下文扰动实现主动、安全、可解释的策略调用。

**🔧 技术方法**

采用冷启动监督训练、Group Relative Policy Optimization (GRPO)、规则式奖励与策略上下文扰动等技术。

**📊 数据集**

使用自己构建的PolicyTraj-20K数据集（约20k条安全策略标注轨迹）以及六个公开代理安全基准。

**📈 对比分析**

与多种闭源、开源通用模型和专用守卫模型对比，RePolicy在六个基准上获得最高总体 Unsafe F1，四项基准排名第一，提升约3.98点。

**⚠️ 局限性**

假设策略库完整覆盖、每条轨迹仅对应单一策略、推理开销较大、缺乏多语言支持、在线干预能力有限等局限。

---

## 107. The Limits of Automatic Evaluation of Creativity in Large Language Models

**arXiv ID:** 2608.23705 | [PDF](https://arxiv.org/pdf/2608.23705v1)

**作者:** Alessandro Tutone `[一作]` (University of Bologna), Mirco Musolesi `[通讯]` (University College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了自动评估大语言模型生成文本创意的有效性与可靠性

**💡 创新点**

揭示自动评估指标与人类评估之间存在严重偏差，并发现LLM作为评判者对AI文本存在自我偏好，说明当前自动评估方法无法捕捉创意多维度的主观价值

**🔧 技术方法**

采用传统定量指标（Creativity Index、Perplexity、SBERT‑Div、语法模板指标）以及LLM-as-a-Judge方案进行评估

**📊 数据集**

使用WritingPrompts数据集，构建了100篇人类创作与100篇LLM生成的短篇故事进行对比

**📈 对比分析**

通过Spearman/Kendall相关性分析与方差比较，结果显示几乎无相关性，LLM评判者在所有维度上表现低于人类且偏向AI文本，整体性能差

**⚠️ 局限性**

实验样本量有限、仅覆盖单一写作任务、仅使用一种LLM评判器、未探讨生成参数变化，且缺乏多语言或跨领域验证

---

## 108. Fidelity Preference, Not Demographic Preference: A Pixel-Level Attribute-Sensitivity Audit of Image Aesthetic/Preference Scorers

**arXiv ID:** 2608.23593 | [PDF](https://arxiv.org/pdf/2608.23593v1)

**作者:** Mingyang Xu `[一作]` (Peking University), Mingyang Xu `[通讯]` (Peking University)

**通讯引用:** 814 | [OpenAlex ID](https://openalex.org/A5049072744)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

评估并审计了四个主流图像美学/偏好评分器（LAION‑Aesthetics V2、PickScore、ImageReward、HPSv2）是否将肤色等人口属性视为质量，采用基于像素级干预的因果隔离方法，在真实与合成图像上进行交叉验证，并提出了可复现的审计基准；

**💡 创新点**

提出三支柱审计协议：①像素级属性干预（肤色光度 L* 移动、身体宽度几何扭曲）；②干预伪影协方差校正（过度剪裁、几何变形）；③best‑of‑n 零偏差校准（随机对照+配对自举）以及“属性可审计性诊断”，从而区分真实性质的“fidelity preference”与人口方向偏差；

**🔧 技术方法**

使用 CIELAB L* 光度干预、几何扭曲、图像分割阈值、贝叶斯 FDR 多重比较校正、分解回归去除干预伪影、argmax 归一化、配对自举、交叉验证以及统计检验；

**📊 数据集**

FairFace 真实人脸（1470 张）、SDXL 合成人脸（54 张）以及 SSP‑3D 311 帧/62 个体的全身照片；

**📈 对比分析**

在同一基准图像上施加不同程度的干预，计算得分差值，拟合倒 U 曲线并提取 fidelity 指数与偏向性；使用 best‑of‑n 归一化测量放大效应；在合成与真实两臂交叉验证，发现合成偏向被放大且方向一致，真实数据表现为对齐惩罚（倒 U 形），且无显著种族方向偏差；多重比较校正后结果仍显著，证明方法鲁棒；

**⚠️ 局限性**

仅在肤色光度维度上实现纯净因果干预；体型轴受几何扭曲伪影限制，无法得到严格的偏差测量；仅评估四个评分器，未覆盖所有潜在模型；使用中性提示，未考虑文本提示对评分的影响；受数据集规模与分割阈值的影响；对人口层级差异检测依赖于 FairFace 的标签，可能不足以揭示更细粒度的种族偏差。

---

## 109. From Causal Plausibility to Causal Reliability: Evaluating LLMs as Calibrated Direct Causal-Edge Classifiers

**arXiv ID:** 2608.23660 | [PDF](https://arxiv.org/pdf/2608.23660v1)

**作者:** Amit Kumar `[一作]` (Texas A&M University), Taoran Ji `[通讯]` (Texas A&M University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了12种开源指令微调LLM在六个因果图上进行成对直接因果边判定的表现，包括召回偏高、过度预测以及置信度失准。

**💡 创新点**

首次揭示LLM在直接因果判断中表现出强召回、误判间接和反向边的高置信度误差，并证明基于一致性的一致性置信度比传统置信度更可靠。

**🔧 技术方法**

采用多种提示策略（仅名称、元数据、链式思考、少量样例、结合思考与样例）与四种置信度来源（口头、对数、跨提示一致性、跨模型一致性），并使用温度缩放调校对数置信度。

**📊 数据集**

使用CausalGraphBench中的六个基准图：AsiaM、River Status、COVID、Coal Gasifier、Hepar2和Munin1。

**📈 对比分析**

通过与模型规模、提示风格和置信度来源比较，发现大模型在小型图上略优但在大图上收益有限；召回普遍高于精确度，误报率在间接/反向边显著升高；一致性置信度在校准与判别上优于口头/对数置信度，但差异无显著统计意义。

**⚠️ 局限性**

局限在于仅评估六个基准图、仅使用单一参考结构、未考虑观测/干预数据和全局结构约束、语义可访问性分析为启发式、提示词鲁棒性未测、可能存在数据集熟悉度未完全排除。

---

## 110. Mitigating Exploration Bias in RL for Multi-Instruction Following

**arXiv ID:** 2608.23830 | [PDF](https://arxiv.org/pdf/2608.23830v1)

**作者:** Mian Zhang `[一作]` (University Of Texas Dallas), Zhiyu Zoey Chen `[通讯]` (University Of Texas Dallas)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究RL在多指令跟随中的探索偏差，并提出两阶段框架解决该问题

**💡 创新点**

提出探索偏差量化指标VIA与VSA，并结合Behavioral Bootstrapping与Scarcity‑Aware Rewards两种创新技术

**🔧 技术方法**

采用Rejection Sampling Fine‑Tuning、GRPO与基于稀缺性的群组奖励设计，提升模型对难指令的探索

**📊 数据集**

在IFTrain、IFBench、IFEval及Multi‑IF等可验证指令跟随数据集上进行实验

**📈 对比分析**

相较传统累积奖励基线，使用两阶段框架可在严格准确率上提升约4.5–9.2分，匹配70B级别模型表现

**⚠️ 局限性**

奖励机制仅涵盖一阶与二阶指令组合，难以捕捉更高阶依赖；对非可验证、开放式指令的适用性有限

---

## 111. LLM-Guided Contextual Action Evaluation for Operational Decisions in Industrial Processes

**arXiv ID:** 2608.24156 | [PDF](https://arxiv.org/pdf/2608.24156v1)

**作者:** Youcheng Zong `[一作]` (Northeastern University), Dakuo He `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种工业过程控制的强化学习方法——LCAE，利用离线大语言模型将固定工业文档转换为动作-观测-方向-延迟关系，并通过历史事件动态调节这些关系的权重，在此基础上构建状态条件的动作-效果场来改进Actor-Critic算法。

**💡 创新点**

创新点在于：①将工业文档中的知识提炼为可冻结的关系卡；②通过历史对齐的事件网络为每个关系产生时变权重；③使用带有双曲正切非线性的关系幅值调制，将动作幅值与文档关系、历史权重三者融合，形成完整的动作-效果场；④实现了完全离线的语言模型使用，部署时无实时语言推理。

**🔧 技术方法**

主要技术包括：离线大型语言模型（LLM）进行关系抽取；文本渲染与固定嵌入模型生成语义向量；历史编码器与共享事件映射网络提取关系强度；双向Actor-Critic（SAC/TD3）框架与最大熵目标；以及基于关系权重的动作-效果场计算。

**📊 数据集**

本文未给出公开数据集，实验基于工业过程仿真环境（如聚合物制备、化工反应过程等）与对应的固定文档与传感器日志。

**📈 对比分析**

与原始的raw-action基线（同等网络容量、相同历史编码器）以及对关系卡随机打乱/删除的消融实验进行比较，实验表明在样本受限或动作效应可变的场景下，LCAE显著提升了闭环性能；若基线已达到相同性能或打乱关系后无差异，则说明文档关系未能提供决策偏置。

**⚠️ 局限性**

限制包括：需要准确且覆盖完整的工业文档；关系卡若错误或缺失将导致性能下降；无法处理未观测到的扰动、传感器误差或安全约束；不具备因果推断或故障诊断能力；无法跨环境直接迁移，需要重新抽取关系并重新训练。

---

## 112. IncSFS: Incremental Full-Sparse Flow-Sensitive Pointer Analysis for C/C++

**arXiv ID:** 2608.24391 | [PDF](https://arxiv.org/pdf/2608.24391v1)

**作者:** Kunlin Liu `[一作]` (National University of Defense Technology), Ji Wang `[通讯]` (National University of Defense Technology)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了IncSFS，首个针对C/C++程序的增量全稀疏流敏感指针分析算法；

**💡 创新点**

通过变量重命名构造约束图、同时传播指针集合的增量与递减、并在对象无环条件下保证终止和最优性，创新地实现了增量流敏感分析；

**🔧 技术方法**

利用值流图（VFG）到约束图的转换、强连通分量检测、增量差分传播与互补式传播策略；

**📊 数据集**

在六个大型开源项目（janet、zstd、tmux、astyle、nginx、sqlite）上进行实验；

**📈 对比分析**

与完整分析、重置-重算以及SILVA增量指针分析对比，平均提升约9.6×、5.8×和3.3×；

**⚠️ 局限性**

存在内存开销最高达3.8×、对对象指针环依赖假设、以及VFG到约束图转换的昂贵预处理成本。

---

## 113. The Sharp Tail of Uniform Stability

**arXiv ID:** 2608.24098 | [PDF](https://arxiv.org/pdf/2608.24098v1)

**作者:** Pahan Dewasurendra `[一作]` `[通讯]` (Johns Hopkins University), Pahan Dewasurendra (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并证明了在有界损失的统一稳定学习算法中，高概率一般化误差的最优下界，证明了存在一个确定性回归学习器能够实现与已知上界相匹配的γ log(1/δ) 级别误差。

**💡 创新点**

创新点在于构造一种多尺度稀有Rademacher特征的学习器：通过坐标级别的线性上限斜坡（ramp）和奇异对称最大值（odd maximum）将极端特征的高偏差转化为整体一般化误差，且保持损失在[0,L]内；该构造首次实现了对所有置信水平 e⁻ᵖ（1 ≤ p ≤ c n）同时满足最优下界。

**🔧 技术方法**

主要技术包括：统一稳定性分析、Rademacher特征多尺度设计、斜坡函数的L∞ Lipschitz性质、奇异最大值映射、Paley–Zygmund反集中不等式以及离散化的概率与期望控制。

**📊 数据集**

未使用真实数据集；所有结果均为理论构造与非渐近概率论证明，实验验证由随附的审核程序完成。

**📈 对比分析**

与已知的高概率上界（γ log(1/δ) + L√(log(1/δ)/n)）进行比较，证明该上界在最优常数范围内是可实现的；因此本文提供了统一稳定性理论下最优的高概率误差极限。

**⚠️ 局限性**

局限性：构造的学习器维度呈指数级增长（≈ e^{Θ(n)}），只适用于极端理论分析，无法直接推广到实际低维或结构化学习任务；且仅证明了极端置信水平下的下界，未给出具体可实现的算法实现细节。

---

## 114. Weighted Alternating Tree Automata

**arXiv ID:** 2608.24262 | [PDF](https://arxiv.org/pdf/2608.24262v1)

**作者:** Olle Torstensson `[一作]` `[通讯]` (Linköping University), Olle Torstensson (Linköping University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了加权交替树自动机（Weighted Alternating Tree Automata，wata），并对其表达能力进行了系统研究；

**💡 创新点**

创新点在于将权重与交替操作统一在树自动机中，证明在局部有限半环上 w/ata 与普通加权树自动机等价，而在非局部有限半环上具有更强表达力，并给出其语言类恰为可识别加权树语言对逆树同态的闭包；

**🔧 技术方法**

核心技术包括：正则化与归一化的 w/ata 转化、树同态与逆同态构造、两种模式的等价证明、以及使用多项式和幂运算来模拟交替与复制；

**📊 数据集**

本文没有使用实验数据集，全部以理论形式证明；

**📈 对比分析**

比较方法主要是与传统加权树自动机、极限多项式加权树自动机（pol-wta）以及不加权交替树自动机对比；理论结果表明：在局部有限半环下等价，非局部有限时 w/ata 具备更强表达能力；

**⚠️ 局限性**

局限性包括：对非线性树同态不闭合；仍缺乏针对特定语义或应用场景的实现与性能评估；并提出了关于有限交替、泛化权重结构等方向的开放问题。

---

## 115. The Ordinal Annotation Game: How Construct Abstraction Shapes Crowdsourced Consensus

**arXiv ID:** 2608.23727 | [PDF](https://arxiv.org/pdf/2608.23727v1)

**作者:** Kosmas Pinitas `[一作]` `[通讯]` (University of Piraeus), Kosmas Pinitas (University of Piraeus)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究构建并验证了“Ordinal Annotation Game”框架，探讨感知标签中注释者间分歧的结构化行为，并通过两组共享阈值的实验检验构造抽象对投票共识与努力关系的影响。

**💡 创新点**

创新点在于将多注释者的标注过程视为隐式博弈，提出统一相变假设——构造抽象决定共识主导与努力受限的转折点，并揭示感知任务共识增强与抽象任务努力削弱共识的对立现象。

**🔧 技术方法**

主要技术包括游戏理论建模、统一阈值映射、post-hoc 众数投票聚合、窗口化效用与一致性度量、Cohen κ耦合分析、bootstrap 置信区间、线性回归斜率评估等。

**📊 数据集**

使用的数据集为PAGAN交互式注释数据：实验1使用合成颜色强度和音调变化；实验2使用GameVibe FPS游戏片段。

**📈 对比分析**

方法对比同一阈值 δ=0.05 的两类任务，计算 dA/dE 斜率：感知任务正斜率（约 0.57-0.63）表明努力提升共识；抽象任务负斜率（-0.98 至 -0.99）表明努力降低共识。置信区间与时间分辨率验证结果显著。

**⚠️ 局限性**

局限性包括关联性非因果、样本量小、实验人群与刺激差异、低层次 saliency 近似、未检验其他聚合规则或多样化参与者。

---

## 116. A Judge Should Know What Changed:Construct Validity for LLM-as-a-Judge Evaluation

**arXiv ID:** 2608.24419 | [PDF](https://arxiv.org/pdf/2608.24419v1)

**作者:** Jianlin Chen `[一作]` (South China University of Technology), Chi Man Vong `[通讯]` (University of Macau)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出评估者（Judge）的构造效度二维指标——不变性和敏感性，并通过人类判定的最小编辑干预，测量了七个不同供应商的评审器在四个评估领域的效度曲线。

**💡 创新点**

创新点在于：①首次将评估者的效度拆分为两个独立维度并证明它们互不约束；②发现评审器在保持不变性时对构造改变的敏感性极低；③揭示公开标签集的泄漏现象，并给出验证力（validation power）评估框架。

**🔧 技术方法**

技术手段包括：构造效度定义与最小编辑干预、方向判定协议、对评审器进行阈值裁切以构造效度曲线、贫困预测器族（长度、表面、生成条件）与验证力算子。

**📊 数据集**

数据集：基于科学主张与证据的对照任务；使用来自七个供应商的评审器，跨四个评估领域；公开标签集包括 RewardBench、RM-Bench、MT-Bench、c4SetsAudited 等；并构造了新的盲注释集供验证力测试。

**📈 对比分析**

比较方法：在匹配的不变性阈值下计算敏感性，得到平均不变性≈0.77、敏感性≈0.26；与控制轴对比显示 scope 与 strength 轴存在显著差距；验证力在 per-item 模式约 0.3–0.5，paired 模式接近 1，表明大多数公开标签集可被表面特征完全预测。

**⚠️ 局限性**

局限性：效度测量依赖人工方向判定；仅适用于可构造最小编辑的科学主张；验证力评估受限于贫困预测器族；存在长度效应等混杂因素；未对其他构造或更广泛的评审器进行验证。

---

## 117. Inductive Inference of Cellular Automata

**arXiv ID:** 2608.24240 | [PDF](https://arxiv.org/pdf/2608.24240v1)

**作者:** Martin Kutrib `[一作]` (University of Giessen), Matthias Wendlandt `[通讯]` (University of Giessen)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种利用有限数量的状态间隔（由起始词、目标词和距离组成）来进行单向和双向一维元胞自动机的归纳推断与验证，并给出三种变体：已知完整转移函数的验证、未知转移函数的推断以及已知部分转移函数的扩展推断。

**💡 创新点**

创新点在于：①首次把元胞自动机的归纳推断问题转化为基于状态间隔的形式化问题；②设计了多种多态算法，证明这些问题可在多项式时间内完成；③证明了它们的并行计算复杂度为 P‑complete，表明即便在并行模型下仍难以加速；④提出了参数化视角下的二进制距离处理方法。

**🔧 技术方法**

主要技术包括：空间–时间图模拟、状态替换与冲突消除的构造算法、log‑space 归约构造、以及对多态问题的递归分解与并行性分析。

**📊 数据集**

论文未使用任何实验数据集，全部工作在理论计算机科学框架下进行。

**📈 对比分析**

由于未进行实验评测，作者通过理论分析与归约证明来说明算法的多项式时间复杂度和 P‑完整性；与已知的预测问题、L‑系统推断等相关工作相比，归纳推断更一般化，并在同类问题中保持了相同的复杂度级别。

**⚠️ 局限性**

局限性包括：①距离以单元编码为前提，导致输入规模较大；②构造的元胞自动机可能包含大量冗余状态，缺乏有效的最小化策略；③未给出实际实现或实验评估，无法验证在真实数据上的性能；④对更一般化的元胞自动机模型（如多维、随机或非确定性）尚未扩展。

---

## 118. What Guides the Agent? Adjudicating Unauthorized Behavior via Localizing Behavior-Guiding Instructions

**arXiv ID:** 2608.24022 | [PDF](https://arxiv.org/pdf/2608.24022v1)

**作者:** Yichao Gao `[一作]` (University of Science and Technology of China), Zhiqiang Wang `[通讯]` (University of Science and Technology of China)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出AttnLocate框架，利用LLM注意力模式实时定位行为指导指令并基于指令来源进行授权决策，防止注入攻击导致的误动作。

**💡 创新点**

创新点在于将行为指令定位转化为注意力矩阵上的目标检测问题，使用多头多层注意力聚合、1‑D U‑Net与锚点自由检测头，并引入sink‑aware正则化及权限仲裁器，实现精细化、可解释化的注入防御。

**🔧 技术方法**

核心技术包括多头多层注意力聚合、Gaussian层加权、1‑D U‑Net骨干网络、anchor‑free检测头、sink‑aware损失、以及基于提供者权限的仲裁策略。

**📊 数据集**

使用MCPTox（工具中毒）和InjecAgent（间接提示注入）两类数据集，涵盖Qwen、DeepSeek、Phi、LLaMA、Mistral、Gemma等六大LLM族群共10种代理配置。

**📈 对比分析**

与静态扫描、行为审计和属性归因等基线对比，AttnLocate在行为指令定位上mIoU达0.692–0.858、在授权决策上AUROC 0.927–0.989、TPR>0.91且FPR<0.07；跨模型零样本迁移仍保持高性能，且可无训练适配不同权限策略。

**⚠️ 局限性**

限制在于需要模型白盒访问以提取注意力矩阵，对极长上下文会导致性能下降；训练仍需针对每个模型或任务，且主要针对指令级注入，未覆盖更广泛的攻击类型。

---

## 119. Instance-Optimality of Bidirectional Dijkstra on Simple Graphs

**arXiv ID:** 2608.24380 | [PDF](https://arxiv.org/pdf/2608.24380v1)

**作者:** Christian Bertram `[一作]` (BARC, University of Copenhagen), Shuyi Yan `[通讯]` (BARC, University of Copenhagen)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在正权重简单图（无自环、无平行边）上双向 Dijkstra 算法的实例最优性，给出了在不同图模型（有向/无向、顺序依赖/无序）以及不同参数（边数、顶点数、最大度数、易实例）下的上界与下界。

**💡 创新点**

创新点在于将 Haeupler 等人仅在多重图上证明的实例最优性结果推广到常见的简单图，并揭示了实例最优性在无向无序模型下保持常数因子，而在有向或顺序依赖模型下需退化到 Θ(min{⌈m/n⌉, n^{1/3}}) 或 Θ(⌈m/n⌉) 的下界；此外，本文首次阐明了度查询对实例最优性的影响，并给出无度查询时的优化结果。

**🔧 技术方法**

技术上采用了实例最优性框架、概率论与逆向构造（构造 (a)、(b)、(c)），利用邻接表的顺序随机性与逆序性，结合最坏情况分析、四阶元组计数以及分层计数技巧来得到紧致的上界与下界。

**📊 数据集**

本文为理论工作，未使用任何实验数据集；所有结果均为数学证明。

**📈 对比分析**

通过与所有“实例智能”算法的比较，本文证明双向 Dijkstra 在无向无序模型下是常数因子实例最优的；在有向或顺序依赖模型下，它只能达到 Θ(min{⌈m/n⌉, n^{1/3}}) 或 Θ(⌈m/n⌉) 的实例最优性比率；在稀疏图（m/n=polylog n）或易实例（τ=o(n)）下，该比率可降至 O(1) 或 O(h)，与最优算法相匹配。

**⚠️ 局限性**

局限性包括：在有向图或顺序依赖模型下不具备常数因子实例最优性；结果依赖于正权重且不适用于零权重情况；缺乏实验验证；对于极稠密图的最优性比率仍存在对数因子余差。

---

## 120. AgentSpec: Speculative Decoding for Batch Inference of LLM Agents

**arXiv ID:** 2608.24004 | [PDF](https://arxiv.org/pdf/2608.24004v1)

**作者:** Xin Wang `[一作]` (Ohio State University), Mi Zhang `[通讯]` (Ohio State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种针对批量LLM代理推理的Speculative Decoding方法 AgentSpec。

**💡 创新点**

创新点在于：①结构隔离草稿生成（只在相同语义块内检索草稿），②基于冗余度的预算分配（动态分配令牌预算以提升接受率）。

**🔧 技术方法**

使用PDA缓存与token‑string映射、无模型草稿检索、冗余度评分、动态令牌预算分配等技术，并在vLLM上实现。

**📊 数据集**

评测模型：Qwen‑3‑8B、GPT‑OSS‑20B、DeepSeek‑R1‑Distill‑LLaMA‑8B、MiMo‑7B；工作负载：Code Generation (Reflexion+USACO)、Deep Research、SWE‑Bench‑Lite、GAIA 以及非代理基准 Spec‑Bench。

**📈 对比分析**

与 NGram、EAGLE‑3、SuffixDecoding、MTP 以及标准自回归方法对比，AgentSpec 在所有工作负载与模型中取得最高良好吞吐量，批量推理下可达 2.02× 的速度提升，尾部延迟亦下降 1.47×。

**⚠️ 局限性**

局限性：需要代理端提供语义块元数据；若无此信息加速效果受限；依赖于生成中重复模式的可利用度；实现需额外轻量接口。

---

## 121. ROBE: Reversed-Order-Biased-Experts for Extracting Extreme Long-tail Events from Historical Texts

**arXiv ID:** 2608.24268 | [PDF](https://arxiv.org/pdf/2608.24268v1)

**作者:** Stella Verkijk `[一作]` (Vrije Universiteit Amsterdam), Piek Vossen `[通讯]` (Vrije Universiteit Amsterdam)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了ROBE（Reversed-Order-Biased-Experts）方法和域特定合成数据技术，用于从17-18世纪荷兰东印度公司（VOC）的早期现代荷兰手稿中提取超过50种极端长尾事件；

**💡 创新点**

创新点在于将专家分类器按事件稀缺程度反向优先级排序以抵消频率偏差，并结合基于GPT-4o的提示模板生成符合领域的合成数据；

**🔧 技术方法**

技术实现包括对GloBERTise编码器进行按标签组细化微调，使用优先级规则进行分类器组合，利用Prompt模板与实体表格生成合成句子，并对比Lexicon、GenLLM等基线；

**📊 数据集**

使用的主要数据集为GLOBALISE项目公开的VOC档案（280份训练、18份测试扫描），涵盖70个事件标签，另外利用GLOBALISE的姓名册、商品词典和船舶数据生成合成数据；

**📈 对比分析**

在多种评估设置下与单一全量微调模型（EtE）、EtE+合成数据、ROBE、ROBE+、Lexicon和GenLLM基线对比，ROBE+在大多数情况下获得最高召回率和F1，ROBE在精确率上提升约0.16；合成数据提升精确率但降低召回率；外部验证显示ROBE的实际精确率远高于实验室评估；

**⚠️ 局限性**

局限性包括合成数据效果缺乏深入分析、对长尾事件的细粒度错误分析不足、专家分组与优先级设定对结果的依赖性高，以及对其他领域或语言迁移的可推广性未知。

---

## 122. Predicting Radiologist Expertise from 3D Gaze Patterns During CT Interpretation

**arXiv ID:** 2608.23836 | [PDF](https://arxiv.org/pdf/2608.23836v1)

**作者:** Leila Khaertdinova `[一作]` (University of Copenhagen), Bulat Ibragimov `[通讯]` (University of Copenhagen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

通过眼动追踪与3D CT扫描数据，构建并训练了一个基于Transformer的模型，用来判定放射科医师的专业水平。

**💡 创新点**

创新点在于将注视热图通过可学习的log‑space偏置注入自注意力层，并在特征聚合时采用注视加权池化，实现对3D视觉搜索行为的双重建模。

**🔧 技术方法**

使用DINOv2视觉Transformer、可学习的log‑bias、注视加权池化以及全连接分类头，并采用Adam优化、数据增强与采样等技术。

**📊 数据集**

使用40份LIDC‑IDRI肺部CT（共8,022切片）和182次阅读会话，涵盖3名专家与2名新人，采集了眼动轨迹和音频报告。

**📈 对比分析**

与改造的多模态CNN（TF‑CNN、IF‑CNN等）以及扫描路径预测模型（CT‑Searcher、Lou等）在5折交叉验证下比较，测试集ROC‑AUC 0.91、F1 0.86、特异性0.93，显著优于其他方法。

**⚠️ 局限性**

局限性包括样本仅为5名医师、数据量有限，模型可能过拟合，对不同CT协议、设备的迁移性尚未验证；眼动同步与映射过程亦可能引入误差。

---

## 123. AgentWorld: Personality-Aware Reliability Evaluation for Agentic Information Retrieval

**arXiv ID:** 2608.24076 | [PDF](https://arxiv.org/pdf/2608.24076v1)

**作者:** Gunja Agarwal `[一作]`, Vignesh Divakaran `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了 AgentWorld 框架，用于基于 Big Five 人格驱动的用户模拟、交互式信息检索代理的可靠性评估与训练闭环；

**💡 创新点**

创新点在于将人格驱动模拟、pass^k 连贯性度量、结构化失效诊断、分层工具调用、双控制交互、训练导出与多层对抗风险分析统一到一条闭环管线中，首次实现人格、多步工具调用与对抗性脆弱性共同评估；

**🔧 技术方法**

采用 LLM 生成用户与对话、NetworkX 构建多代理拓扑、状态化工具环境、pass^k 一致性度量、Monte Carlo 分支估计、Dempster–Shafer 证据融合、Shapley 归因、LLM 判分、DPO 自动导出等技术；

**📊 数据集**

使用模拟生成的 OCEAN 人格人口、内置六类工具应用、生产环境中的分析与客服代理对话（共 60/240 条消息和 19 任务×人设的实验数据）；

**📈 对比分析**

通过单次 LLM 判分与 pass^k、行为分数、完整度、步数等指标比较；实验显示不同人格/任务组合表现差异显著，pass^k 揭示可复现性不足，Risk Analyzer 发现行为通过率 100% 的任务在 V_min 低至 0.375，攻击层面风险分布清晰；

**⚠️ 局限性**

限制包括人格模拟依赖提示工程未通过人类实验验证、LLM 判分可能带偏差、Risk Analyzer 样本量小且仅使用单维扰动、对抗链分析未完成等。

---

## 124. PonderPounce: A Pretrained MLLM as an Episode Context Engine for Robot Control

**arXiv ID:** 2608.24115 | [PDF](https://arxiv.org/pdf/2608.24115v1)

**作者:** Suhwan Choi `[一作]` (MAUM.AI), Youngjae Yu `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过将预训练的多模态大语言模型（MLLM）的因果上下文直接作为机器人记忆，设计了一种异步持续认知通道，将记忆信息传递给视觉‑语言‑动作（VLA）控制器，实现端到端训练的记忆与控制协同。

**💡 创新点**

创新点包括：①不需要专门的记忆存储、检索或压缩模块，而是利用MLLM的原生上下文作为可扩展的 episode 记忆；②通过持续的认知 token 及其年龄信息实现系统 1（控制器）与系统 2（记忆引擎）的异步通信；③将记忆更新、子目标生成与演示推理等监督直接嵌入系统 2 的 LM 头，提升控制性能；④证明更大的预训练上下文容量（9B vs 0.8B）可在不改动控制器架构的情况下提升成功率。

**🔧 技术方法**

使用的技术包括：预训练多模态 LLM（Qwen3.5-9B/0.8B）作为上下文引擎、预训练动作模型（3.6B 或 3B GR00T N1.5）作为控制器、持续认知 token 与年龄编码、端到端联合训练、flow‑matching 损失、子目标与演示推理的交叉熵监督、KV 缓存和 Triton 融合内核实现低延迟推理。

**📊 数据集**

采用的主要数据集为 RoboMME（16 任务的记忆依赖控制）和 RoboCasa‑DC（演示条件控制）两套仿真基准；此外在 RoboMME 任务中使用 1× 与 9× 规模的训练数据。

**📈 对比分析**

在 RoboMME 上，该方法在 1× 规模下成功率 60.83% 超过 FrameSamp+Modul 的 44.51%；在 9× 规模下 75.54% 高于 57.88%。在 RoboCasa‑DC 上成功率 12.5% 仅略高于 SeeTraceAct 的 11.6%，但关闭认知时下降至 8.6%。相比传统目的建模记忆或直接观测控制，显示了基于预训练上下文的记忆在多任务记忆依赖场景下的显著优势。

**⚠️ 局限性**

局限性包括：仅在仿真环境中验证，缺乏真实机器人实验；使用的记忆通道仅单一认知 carrier（K=1）且上下文长度受 16K token 限制；对演示推理与子目标的监督依赖模拟生成的标注，可能影响可迁移性；能耗和并发吞吐量分析不足；在演示条件任务中的绝对成功率仍偏低，说明该方法并非在所有记忆需求下均最优；随机初始化的 9B 模型训练不稳定，证明对预训练的高度依赖。

---

## 125. Physics-Integrated Operator Learning via Gaussian Splatting Representations

**arXiv ID:** 2608.24049 | [PDF](https://arxiv.org/pdf/2608.24049v1)

**作者:** Jihao Zhang `[一作]` (Cornell University), Jian-Xun Wang `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种将物理方程知识嵌入神经算子（neural operator）的框架，通过使用前馈高斯抛光（FFGS）连续字段表示来实现对偏微分方程（PDE）状态的连续重建和闭式空间导数计算，并将这些导数直接嵌入到学习的演化映射中。

**💡 创新点**

创新点在于：① 在表示层面（field representation）而非训练损失或网络结构层面嵌入物理；② 使用可导的FFGS表示提供高阶导数，避免残差损失的数值难题；③ 支持部分已知物理（系数未知）时的系数识别与自适应；④ 通过单一前馈网络实现可扩展、可泛化的物理耦合。

**🔧 技术方法**

技术手段包括：前馈高斯抛光（FFGS）编码器、连续字段重建与解析导数、低通谱滤波、基于FFGS的嵌入物理组件、残差自由的复合演化算子、rollout 训练、系数线性回归识别、以及基于 FNO/U-Net/ResNet 的神经算子骨干。

**📊 数据集**

使用了五个二维/三维 PDE 基准：2D 纯输运、2D 输运-扩散、2D Burgers、2D 输运-Allen–Cahn 反应扩散、3D 输运-扩散；所有基准采用随机截断傅里叶序列初始条件，网格尺寸为 160^2 或 64^3，训练使用前 50 步，评估 200 步自回归推理。

**📈 对比分析**

对比方法为无物理约束的 FNO、U-Net 与 ResNet，使用相同参数规模；实验显示在所有基准上本文模型在 200 步自回归推理中相对 L2 误差平均降低 1.5–2.2 倍，谱误差显著下降，PSNR 提升 3–7 dB，且在部分已知物理时仍保持优异性能。

**⚠️ 局限性**

局限性包括：① FFGS 表示容量固定，难以覆盖高频/尖锐前沿；② 需要为每类 PDE 训练专属编码器，难以跨方程迁移；③ 低通滤波阈值需手动设定；④ 推理时比纯神经算子略慢。

---

## 126. Identifying Latent Declarative Representations of Code for Assisting Repository Migration

**arXiv ID:** 2608.23619 | [PDF](https://arxiv.org/pdf/2608.23619v1)

**作者:** Shraddha Surana `[一作]` (BITS Pilani), Michael Bain `[通讯]` (UNSW Sydney)

**通讯引用:** 2500 | [OpenAlex ID](https://openalex.org/A5058225317)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种利用隐式声明性表示（ADFD）来辅助大型遗留代码仓库迁移的方法。

**💡 创新点**

创新点是将程序的隐式声明性描述显式化为可检查的ADFD，并结合依赖感知分块生成，显著提高迁移覆盖和正确性。

**🔧 技术方法**

使用大型语言模型（Qwen-3-Coder、Claude Sonnet 4.5）进行ADFD推断与代码生成，配合静态分析、SCC分块与对齐检查。

**📊 数据集**

使用自定义的f2x50基准：50个Fortran开源仓库，按复杂度划分为低、中、高三层。

**📈 对比分析**

与直接翻译和两种消融基线对比，ADFD-Migrate在行为一致性率达85.6%，迁移结果指数平均93.1%，比基线提升16-59个百分点。

**⚠️ 局限性**

局限在于仅评估Fortran到Python迁移，缺乏全面系统测试，依赖特定语言工具链，且对复杂控制流和域特定数值精度的支持仍不完善。

---

## 127. MnemoDyn: Learning Resting State Dynamics from 40K FMRI sequences

**arXiv ID:** 2608.23936 | [PDF](https://arxiv.org/pdf/2608.23936v1)

**作者:** Sourav Pal `[一作]` (University of Wisconsin--Madison), Vikas Singh `[通讯]` (University of Wisconsin--Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种基于多分辨率波let与伪微分算子结合的动力学系统模型（MnemoDyn），用于建模和重建大规模 rs‑fMRI 序列；

**💡 创新点**

核心创新在于将 rs‑fMRI 的时间演化视为连续时间动力学系统，通过参数化的多尺度积分算子（Wavelet + pseudo‑differential）实现可解释、稀疏且计算高效的序列建模，避开了 Transformer 的自注意力机制；

**🔧 技术方法**

采用连续时间 ODE/控制方程框架、波let多分辨率分析、伪微分算子、低秩 CP 张量分解、以及自监督预训练（掩码/去噪）与轻量化适配器微调；

**📊 数据集**

预训练数据来自约 40,000 条 rs‑fMRI 序列，主要包含 UK Biobank（≈65K 受试者）和 Human Connectome Project（≈1,000 受试者）；微调与评估使用 HCP‑Aging、ADNI、Healthy Brain Network、ADHD‑200、ABIDE、NKIR 等多种公开 rs‑fMRI 数据集；

**📈 对比分析**

与 Transformer‑基线（Brain‑LM、Brain‑JEPA）以及传统 CNN/GCN 模型对比，MnemoDyn 在重建 MSE/R²、年龄/性别预测、疾病诊断（ADNI）等任务上均实现或逼近最先进性能，且训练成本仅为 Transformer 的 1/5 左右；

**⚠️ 局限性**

主要局限包括：仅验证在已分区的 rs‑fMRI 数据上，尚未扩展到 voxel 级或多模态（EEG、PET）输入；缺乏纵向研究验证；模型的神经生理学对应性仍待进一步探究。

---

## 128. Distributed Hypothesis Testing Against Dependence

**arXiv ID:** 2608.24403 | [PDF](https://arxiv.org/pdf/2608.24403v1)

**作者:** Han Wu `[一作]` (Tokyo University of Agriculture and Technology), Shun Watanabe `[通讯]` (Tokyo University of Agriculture and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

在分布式假设检验框架下，论文针对测试依赖性、依赖性与独立性的笛卡尔积以及条件依赖性问题，给出了误差指数的单字母表达式。

**💡 创新点**

创新点在于首次证明了 Han 指数在测试依赖性问题中是最优的，驳斥了 Han 先前关于朗姆信息的猜想，并引入了条件量化编码方案提升了条件依赖性问题的指数。

**🔧 技术方法**

主要技术手段包括对 Han 指数的多字母形式进行单字母化，利用辅助分布与反向求和论证；以及基于条件典型性、类型覆盖引理的编码与误差分析。

**📊 数据集**

论文为理论性研究，无使用具体数据集，所有结论均以信息论模型和符号推导验证。

**📈 对比分析**

由于未进行数值或实验比较，论文未给出与现有方案的性能对比，只说明了所给指数在特定假设下可达到或严格优于已知下界。

**⚠️ 局限性**

局限性包括：结论仅适用于 i.i.d. 结构且假设分布满足特定因子化形式，未探讨更一般的多终端或非独立信号情况；同时未给出实际实现的可行性分析。

---

## 129. A Structural FHMM for Interpretable Disease Trajectories in T2DM

**arXiv ID:** 2608.24328 | [PDF](https://arxiv.org/pdf/2608.24328v1)

**作者:** Alessandro Mari `[一作]` (Ecole Polytechnique Fédérale de Lausanne), Andrea Burden `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种结构化因子隐马尔可夫模型（FHMM）用于分析2型糖尿病患者的长期疾病轨迹，并通过多链独立潜在状态来识别临床相关的患者状态与轨迹聚类；

**💡 创新点**

将FHMM与可解释的结构约束相结合——状态递减、单向转移、带宽限制的转移矩阵以及单调增发射参数——实现了对多种共病进展的因子拆解与临床解读；

**🔧 技术方法**

使用结构化FHMM、变分EM（structured mean‑field）、group‑lasso单调性约束、带宽转移矩阵、EM参数学习、K‑means聚类、AIC/BIC模型选择等统计与机器学习技术；

**📊 数据集**

利用IQVIA Medical Research Data UK（IMRD‑UK）中的THIN匿名电子健康记录，筛选了88201名成年人（2006–2019年首次非胰岛素降糖药处方），包含诊断、实验室、药物、生活方式等变量；

**📈 对比分析**

与传统单链HMM在同一数据上进行比较，采用BIC/AIC进行模型选择；结果显示FHMM（8链×4状态、转移带宽2）在所有性别/年龄组中均优于HMM，且在聚类和轨迹解释上表现更好；

**⚠️ 局限性**

局限包括：EHR数据不规则且缺失严重、使用前向插值可能引入偏差、仅考虑常见共病且排除罕见疾病、假设高斯发射、时间步归一化、HbA1c筛选导致样本偏倚、药物使用未充分建模以及罕见事件的不平衡问题。

---

## 130. VideoHarness-RSI: Recursive Harness Self-Improvement for Long-Video Understanding with Frozen Vision-Language Models

**arXiv ID:** 2608.24302 | [PDF](https://arxiv.org/pdf/2608.24302v1)

**作者:** Guoyang Xu `[一作]` (Tencent), Hao Chen `[通讯]` (Tencent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种递归自我改进的可执行上下文构造器搜索框架，用于在冻结的长视频视觉‑语言模型（VLM）周围构造高效的上下文；

**💡 创新点**

创新点在于将上下文构造视为可执行程序并在外部循环中递归搜索、评估和保留改进的程序，从而将模型改进与上下文改造分离，形成可重复的优化层；

**🔧 技术方法**

使用的技术包括：可执行程序的写‑读‑打包（Write‑Read‑Pack）结构、外部提议者（Claude Opus 4.6）生成候选程序、基于开发集准确率的严格前沿更新策略、以及对上下文成本的Pareto分析；

**📊 数据集**

数据集主要为 LVBench（包含约1,232问答对的83段视频）进行开发与测试，同时在 Video‑MME 与 MLVU 上进行跨基准直接复用评估；

**📈 对比分析**

与四类对照基准（统一采样、手工构造、文献启发式、专有 VLM）以及无搜索的基线相比，递归搜索得到的上下文构造器在开发集上提升了约15‑20%准确率，在保留测试上提升至约50%及以上，且能够迁移到其他长视频基准；

**⚠️ 局限性**

局限性包括：仅在单一冻结模型和单一数据种子上实验；搜索过程受提议者知识与先验影响；仅在单个视频内构造上下文，未考虑跨问答记忆；以及高容量搜索显示开发与测试选择之间的差距。

---

## 131. Unary Versus Binary Two-Way Automata

**arXiv ID:** 2608.24238 | [PDF](https://arxiv.org/pdf/2608.24238v1)

**作者:** Viliam Geffert `[一作]` (P.J.Šafárik University), Rastislav Královič `[通讯]` (Comenius University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了单词语言的状态复杂性，给出了将单元正则语言的最小单向确定性有限自动机（DFA）转换为二进制编码对应的双向DFA的状态数上界与下界，并构造了匹配的示例语言。

**💡 创新点**

创新点：1）证明单元DFA的二进制编码可由一个O(n·log n)状态的双向DFA识别；若原DFA仅使用奇长度循环，则该上界降至2n+2；2）给出对应的下界证明，表明至少需要n个状态；3）构造每个 n≥7 的最优示例语言，展示上界与下界的紧密关系。

**🔧 技术方法**

使用了理论工具：内图与循环分解、奇偶循环分离、扫波（sweeping）自动机、基于模运算的gadget 模拟、Chinese remainder theorem 等，配合状态复杂性分析和构造证明。

**📊 数据集**

本工作完全是理论性的，未使用实验数据集。

**📈 对比分析**

通过严格的理论证明比较状态数上界与下界，结果表明：在一般情形下上界为 O(n·log n)；若仅含奇长度循环，可实现线性上界 2n+2；下界则至少为 n，构造的示例语言满足 n≤ |Q| < n+log n−1，体现两边的紧密匹配。

**⚠️ 局限性**

限制与未解问题：1）不确定上界 O(n·log n) 是否可进一步优化；2）研究仅限于确定性双向FA，非确定性情形仍待探讨；3）构造示例语言依赖于素数分解，若能减少素数个数或改用其他分解可能进一步压缩状态数。

---

## 132. ALPHABET: A Laplace-Pole History Aggregator with Banked Exponential Transport

**arXiv ID:** 2608.24051 | [PDF](https://arxiv.org/pdf/2608.24051v1)

**作者:** Daehwa Ko `[一作]` (Korea Aerospace University), Jay Hoon Jung `[通讯]` (Korea Aerospace University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种仅含数千参数的线性时间序列模型，通过两套稳定拉普拉斯极点银行压缩时间历史，并将模型状态压缩为可审计的模态统计量来完成预测任务。

**💡 创新点**

创新点在于将可审计的模态统计量与频谱测量对应，理论证明每个极点能定位谱信息，且在有限模态下可实现谱类分离，同时实现了极小参数、高速推理的竞争性性能。

**🔧 技术方法**

使用技术包括：两套极点银行（直接与级联）线性递归、模态能量与滞后矩阵读取、对数映射压缩、RMS归一化、以及固定的仿射头；同时基于频谱理论分析极点响应。

**📊 数据集**

实验使用多任务公开数据集，涵盖UCR、UEA、ECG、临床/活动数据以及预测任务，构成十家族的对比基线。

**📈 对比分析**

与九个可训练序列模型族在同一基准下比较，本文模型在D=64宽度下仅6,437参数，推理速度比基线快5.02倍，完整训练步快3.93倍，平均排名3.97，且在多任务上表现与最强模型相当。

**⚠️ 局限性**

局限性包括：理论仅适用于完全观测、等步、平稳的特征过程；对高斯噪声的鲁棒性差；未保证优化一定找到分离极点；在某些消融对比中统计显著性不强；在非平稳或稀疏观测下表现未知。

---

## 133. Defending Network Intrusion Detection Systems Based on Graph Neural Networks Against Structural Adversarial Attacks

**arXiv ID:** 2608.24454 | [PDF](https://arxiv.org/pdf/2608.24454v1)

**作者:** Dimitri Galli `[一作]` (University of Modena and Reggio Emilia), Mirco Marchetti `[通讯]` (University of Modena and Reggio Emilia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对基于图神经网络（GNN）的网络入侵检测系统进行防御，采用对抗训练提升其对结构性对抗攻击的鲁棒性。

**💡 创新点**

创新点在于通过低度节点替换生成结构对抗样本的轻量化方法，将此对抗样本加入训练集，从而实现高效的对抗训练。

**🔧 技术方法**

使用技术包括E-GraphSAGE GNN、对抗训练策略以及低度节点替换生成对抗流的技术。

**📊 数据集**

采用的公开数据集为CTU-13和TON-IoT，分别包含多种网络攻击场景与正常流量。

**📈 对比分析**

与基线E-GraphSAGE进行对比，硬化模型在干净数据上F1、精确率、召回率保持或略微提升；在结构攻击（攻击、恶意通信、添加节点）下检测率显著提升，最高可达近90%的绝对增幅。

**⚠️ 局限性**

局限性包括仍未能完全抵御所有对抗攻击，鲁棒性提升仍有限；对更强或组合型攻击的适应性需要进一步研究。

---

## 134. PhysicsBench: A Unified Leaderboard for Generative and Predictive Models in Engineering Design and Simulation

**arXiv ID:** 2608.24056 | [PDF](https://arxiv.org/pdf/2608.24056v1)

**作者:** Sang Won Lee `[一作]` (Narnia Labs), Namwoo Kang `[通讯]` (Narnia Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了统一的生成与预测模型基准与排行榜BenchRank，覆盖1D/2D/3D工程设计任务并评估在有限数据规模下的表现。

**💡 创新点**

创新点在于统一评价流程、数据规模分层、引入工程有效性指标、使用去偏图谱排名算法以及公开可复现的“living leaderboard”。

**🔧 技术方法**

使用了多种深度学习技术，包括GAN/VAEs/扩散、点云/图网络、神经算子、Transformer、神经场，以及传统的TabPFN、XGBoost等。

**📊 数据集**

使用工业级CAD/CFD/FEA数据集（DeepJEB、DeepWheel、DrivAerNet、DrivAerML、AirfRANS等）以及公开数据集。

**📈 对比分析**

通过统一训练、推理、指标计算和BenchRank去偏图谱排名，对比了66个模型在不同数据规模下的质量和效率，发现无单一“最优”模型，最佳模型随任务与数据规模变化。

**⚠️ 局限性**

局限包括样本重复次数不足、部分任务覆盖有限、BenchRank超参数设定固定、数据规模分层与真实工业场景仍有差距，以及商业许可限制。

---

## 135. Deterministic Bandwidth of Finite Languages

**arXiv ID:** 2608.24246 | [PDF](https://arxiv.org/pdf/2608.24246v1)

**作者:** Da-Jung Cho `[一作]` (Ajou University), Max Wiedenhöft `[通讯]` (Kiel University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

研究有限语言在部分DFA（无sink）中的带宽（bandwidth）概念，并系统分析其对语言结构的影响。

**💡 创新点**

①证明即使仅考虑有限语言，k‑带宽DFA也形成无穷层级；②给出带宽为1的有限语言的完整位置结构并提供多项式时间判定算法；③提出可有效计算的上界（位置展开）与下界（残差球）并证明两者相差不超过2D；④在某些语言族上这两个界完全匹配。

**🔧 技术方法**

主要使用离散数学与图论技术：带宽定义、层级构造、状态层展开、残差语言分析、图的圆形/线性排序、动态规划求解层级集合与球集合。

**📊 数据集**

本文为理论研究，无使用实际数据集；所有结论均通过证明得到。

**📈 对比分析**

通过理论证明与构造示例进行比较；对任意有限语言给出可计算的上、下界，误差因子不超过最长词长的两倍（即2D），展示了理论上可接受的精度；对带宽为1的情形给出多项式（O(|Q|³|Σ|)）判定算法。

**⚠️ 局限性**

①对于固定有限字母表，尚未证明上、下界可在所有语言族上匹配；②决定任意k是否足以表示给定语言的判定问题的复杂度仍未知（可能是NP‑hard）；③仅讨论有限语言，扩展到一般正则语言及近似情形仍为未来工作。

---

## 136. Asymptotically Tight Bounds for Generalized Covering Radii of Binary Primitive BCH Codes at All Higher Orders

**arXiv ID:** 2608.23833 | [PDF](https://arxiv.org/pdf/2608.23833v1)

**作者:** Zeev Vladimir Belinsky `[一作]` (Technion Israel Institute of Technology), Aryeh Lev Zabokritskiy `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `847a60d8-a755-47af-ba5d-c5236b9e3083` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了二进制原始 BCH 码的广义覆盖半径，给出了从第二阶到更高阶的稳定上界和下界，并在四误差和三误差等特定情形下给出了精确值；

**💡 创新点**

创新点在于提出了统一的“共同核心”框架，将多重误差覆盖问题转化为一次列举多元完成多项式，并利用代数几何、簇和 Artin–Schreier 标记实现符号域线性无关性，从而获得稳定的两值上界；

**🔧 技术方法**

使用了代数几何的不可约性证明、Hankel 矩阵逆、完成多项式的分裂域与符号场、符号类线性无关性、以及 Cafure–Matera 的有限域点数估计等技术；

**📊 数据集**

没有使用传统机器学习或大规模实验数据，而是基于理论分析和符号计算；

**📈 对比分析**

论文通过理论证明与已知的 Griesmer 下界、球覆盖下界等比较，表明在大多数参数范围内所给的上界与下界已相当紧密；

**⚠️ 局限性**

局限性在于对某些阶（尤其三阶）的精确覆盖半径仍未完全确定；阈值和点数估计较保守，实际最优值可能更小；

---

## 137. Event-Based Motion Estimation via Oriented Distance Fields

**arXiv ID:** 2608.24223 | [PDF](https://arxiv.org/pdf/2608.24223v1)

**作者:** Lei Sun `[一作]` (Sofia University St Kliment Ohridski), Luc Van Gool `[通讯]` (Sofia University St Kliment Ohridski)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aaccfe5c-6b26-4208-b23c-35331481e142` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于预计算方向距离场（ODF）的事件相机运动估计方法，并在图像去模糊和眼部跟踪两项任务中验证其通用性。

**💡 创新点**

核心创新在于用单次平均预计算的距离向量场替代传统的迭代优化或多假设搜索，实现低延迟闭式运动估计，并将估计轨迹用于生成精确模糊核和方向性事件滤波。

**🔧 技术方法**

采用了方向距离场（ODF）、自适应事件批量选择、空间-时间对比滤波、深度可展开网络（USRNet-tiny）、事件轨迹转换为模糊核以及基于ODF的方向向量事件滤波器。

**📊 数据集**

使用了Event-Camera Dataset（滑动器序列）、EventAid-B（去模糊基准）、自采集的事件+帧同步数据（70张模糊图）、Angelopoulos等人的近眼事件数据以及Prophesee GenX320等硬件。

**📈 对比分析**

与CM、HASTE等纯事件方法、Blind NAFNet/Restormer、事件辅助EDI/EFNet等进行对比；在滑动器实验中ODF的更新率最高、误差最低；去模糊实验中轻量网络（0.6M参数）取得最高PSNR 27.23 dB，性能与或优于基线；眼部跟踪中IOU≥0.85、连续追踪时长≈70 s、功耗显著低于帧摄像。

**⚠️ 局限性**

方法要求全局一致的2-DoF平移、足够多样化的边缘方向和足够事件密度；对动态场景、单一边缘主导或稀疏事件区域表现不佳，且不适用于6-DoF或更一般的仿射/投影运动。

---

## 138. Rebuild Dossier: Mechanically-Enforced Specs for Agentic App Rebuilds, and What Model-Tier Failures Reveal

**arXiv ID:** 2608.23616 | [PDF](https://arxiv.org/pdf/2608.23616v1)

**作者:** Parker Fawcett `[一作]` `[通讯]` (Independent Researcher), Parker Fawcett (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了名为rebuild-dossier的开源工具，用于在构建代码之前锁定应用的真实接口（输入输出）并通过一次性测试与机械检查保证重建过程的可靠性。

**💡 创新点**

创新点在于：①通过机械强制的接口锁定与构建纪律（rails）避免了模型生成的结构性不匹配；②对测试套件本身进行变异检测，防止“测试作弊”导致的误判；③使用多级验证（agent报告、日志、文件）确保结果的真实性。

**🔧 技术方法**

技术包括：LLM代码生成与指令（如AgentModernize）；静态与动态接口提取；自动化测试与变异检测；运行时钩子（hooks）实现构建纪律；MCP（多模型协作）与插件化工具链。

**📊 数据集**

数据集为两款真实应用（Madeline与catchandtrade）以及一款第三方Next.js/TS CRUD示例，包含约83路由、512单元测试；此外还使用了人工构造的前端页面与接口以验证边界情况。

**📈 对比分析**

对比方法：在弱模型与强模型、单提示（single-prompt）与spec+rails（锁定接口+纪律）之间进行多次独立实验，记录通过/失败/违规等结果。性能方面：在弱模型下，spec+rails与单提示在最小应用上表现相当；在大型应用上，spec+rails仍易出现批量构建违规，且在某些实验中未真正执行纪律。

**⚠️ 局限性**

局限性包括：样本规模有限（仅两款作者自建、一个第三方应用）；缺乏非LLM或基准重构对照；部分钩子机制在某些平台（Claude Code）未生效；测试套件本身存在“作弊”与“测量误差”问题；未充分评估成本与可扩展性。

---

## 139. MaST: Motion-aware Sparse Pipeline for Lightweight Object Tracking

**arXiv ID:** 2608.24365 | [PDF](https://arxiv.org/pdf/2608.24365v1)

**作者:** Qingmao Wei `[一作]` (South China University of Technology), Quan Tang `[通讯]` (Pengcheng Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了Motion-aware Sparse Tracker（MaST），一种轻量化的端到端稀疏视觉跟踪框架，包含基于运动先验的早期token剪枝和稀疏的预测头。

**💡 创新点**

创新点在于：①使用前一帧预测的运动先验（高斯窗口）指导早期token的选择，克服注意力分数噪声；②设计score-first、regress-once的稀疏MLP头，彻底消除密集reshape与多余回归，真正实现端到端稀疏。

**🔧 技术方法**

采用轻量级Vision Transformer（ViT‑Tiny）作为编码器，利用一次性token稀疏化、运动窗口加权的注意力分数，以及稀疏MLP头进行目标定位；训练时结合交叉熵、L1及GIoU损失。

**📊 数据集**

在COCO、LaSOT、GOT‑10k、TrackingNet、VastTrack、NFS、UAV123等多项公开基准数据集上进行训练与评估。

**📈 对比分析**

在所有基准上与AsymTrack‑S、OSTrack等先进跟踪器对比，MaST‑tiny在1 G MACs下取得63.8 AUC（LaSOT）/80.1 SUC（TrackingNet），在Jetson Nano上跑速152 FPS，Raspberry Pi 5上22.6 FPS，速度提升近两倍且保持或超过竞品的精度。

**⚠️ 局限性**

限制在于高分辨率输入仍需完整注意力计算导致前期计算开销高，缺乏输入自适应剪枝；在极高压缩率下可能出现目标被误剪除的情况。

---

## 140. Centrality-Based Deployment of Queue Policies in Acyclic Multipath Routing Networks

**arXiv ID:** 2608.24131 | [PDF](https://arxiv.org/pdf/2608.24131v1)

**作者:** Mahima Gupta `[一作]` (Indian Institute of Technology Mandi), Sreelakshmi Manjunath `[通讯]` (Indian Institute of Technology Mandi)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在具有拓扑层级、无环多路径路由的网络中单点部署Active Queue Management（AQM）的影响，并给出了理论分析与仿真验证。

**💡 创新点**

提出使用Katz中心性指标来选择最合适的路由器进行AQM部署，证明该指标比传统的边缘路由器部署方案能更好地提升网络稳定性和降低延迟。

**🔧 技术方法**

采用控制理论中的流体模型和延迟微分方程来描述TCP窗口与队列动态，进行局部稳定性分析；利用Katz中心性计算（矩阵求逆+加权路径求和）指导部署；使用ns-3进行包级仿真验证。

**📊 数据集**

使用人工构造的六路由器多瓶颈拓扑（含60条TCP流）和同步/异步RTT的仿真场景，未使用公开真实数据集。

**📈 对比分析**

将AQM部署在计算出的Katz中心性最高节点（例如路由器3）与全DropTail及仅在边缘路由器部署AQM进行对比；仿真结果显示在高RTT场景下该部署方式显著降低平均延迟、保持吞吐率、并在一定程度上控制丢包率，性能优于传统方案。

**⚠️ 局限性**

局限性包括：仅考虑无环多路径网络；假设RTT恒定、流量模型简化；只验证两种TCP变体（NewReno、DCTCP）；未考虑UDP、短突发流量等异构流量；部署策略在不同拓扑或动态流量变化时的适用性仍待进一步验证。

---

## 141. Dataset Scarcity Limits Robust Evaluation of Multilingual Embedding Models: A Case Study of Slavic Languages

**arXiv ID:** 2608.24477 | [PDF](https://arxiv.org/pdf/2608.24477v1)

**作者:** Ana Gjorgjevikj `[一作]` (Josef Stefan Institute), Tome Eftimov `[通讯]` (Josef Stefan Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种两维框架，用以在多语言嵌入模型评估中同时考量任务层面的排名稳定性和语言层面的跨任务迁移一致性，并在斯拉夫语子集上进行实验。

**💡 创新点**

创新点在于引入证据强度评分（ESS）与证据层级概念，能够量化稀缺或冗余数据对评估结论可靠性的影响，并将排名稳定性、Top‑k迁移一致性与证据强度有机结合。

**🔧 技术方法**

技术上采用多指标决策方法（WSM、TOPSIS、VIKOR、PROMETHEE II）配合多重权重策略计算模型排名，使用 Kendall’s W 等统计量评估排名与数据组合的稳定性。

**📊 数据集**

数据集来源于 MTEB 的八类任务（分类、聚类、检索、重新排序、语义相似度、成对分类、多标签分类、双语文本挖掘）及其对应的斯拉夫语言子集。

**📈 对比分析**

通过比较模型在不同排名方案和数据组合下的Top‑k迁移一致性以及覆盖加权的跨任务一致性，实验发现 llama‑embed‑nemotron‑8b、multilingual‑e5‑large‑instruct 与 Qwen3‑Embedding 系列在多任务、多语言上表现最稳定，而单任务赢家多样，且跨任务稳定性显著高于单任务稳定性。

**⚠️ 局限性**

局限性包括：评估严重受数据稀缺与冗余影响，许多语言-任务对缺乏多样化数据，导致证据弱；框架基于单一 MTEB 快照，未考虑数据拆分不确定性及提交者元数据错误。

---

## 142. DoublesEval: Diagnosing Multi-Agent Tactical Reasoning in Vision-Language Models via Professional Doubles Badminton

**arXiv ID:** 2608.24439 | [PDF](https://arxiv.org/pdf/2608.24439v1)

**作者:** Jintao Cheng `[一作]` (Hong Kong University of Science and Technology), Weibin Li `[通讯]` (University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出DoublesEval框架，对专业双打羽毛球视频进行关键时刻诊断评估，系统剖析VLM在多智能体战术推理中的缺陷并提出TacticCheck一致性检查器

**💡 创新点**

将双打羽毛球作为结构化多智能体战术推理的可验证基准，构建四层诊断维度并设计基于低层预测的无监督一致性重排方法

**🔧 技术方法**

利用零-shot视觉语言模型（Qwen2.5-VL、Qwen3-VL、VideoLLaMA3、Molmo2）进行推理，采用GPT‑5作为结构化标签映射器，TacticCheck通过约束式重排序提升结果

**📊 数据集**

60段专业双打比赛（共约9.6K实例）收集自BWF World Tour的四场赛事，包含分段关键时刻、角色、空间、动作与结果标注

**📈 对比分析**

在零-shot设置下，四模型四层准确率均低于40%，TacticCheck提升约7个百分点；相较于传统单智能体评测，DoublesEval更能定位空间、交互绑定和因果链错误

**⚠️ 局限性**

仍受限于低层预测不准导致一致性检查误判；整体准确率远未达到实用水平，缺乏对更复杂多智能体场景的推广与训练指导

---

## 143. Primate vision reveals a missing principle for robust dynamic AI

**arXiv ID:** 2608.23790 | [PDF](https://arxiv.org/pdf/2608.23790v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 144. Automata from Agent Traces: Failure and Next-Step Prediction

**arXiv ID:** 2608.23670 | [PDF](https://arxiv.org/pdf/2608.23670v1)

**作者:** Seonglae Cho `[一作]` (Holistic AI), Adriano Koshiyama `[通讯]` (University College London)

**通讯引用:** 1487 | [OpenAlex ID](https://openalex.org/A5071702962)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `fede83ac-7505-405f-ab37-e7284695c47f` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种仅用正样本构造有限状态机的方法，对LLM代理执行轨迹进行压缩并进行行为分析。

**💡 创新点**

创新点在于利用有限的活动字母表，通过最后活动右合并构造无超参数的紧凑DFA，兼具工作流记忆、下一步预测、故障预测和实时监控。

**🔧 技术方法**

采用活动提取、前缀树、按最后活动合并、稀疏转移过滤，以及基于状态的概率估计和机器学习特征。

**📊 数据集**

在十二个公开LLM代理轨迹数据集（如SWE-agent、WebArena、AgentNet、tau2-bench等）上验证。

**📈 对比分析**

与RPNI、EDSM、Alergia、k-Tails、HMM、Process Mining和AWM等基线相比，压缩率15–3036×，预测交叉熵降低21%，故障预测AUROC最高0.94，实时监控在32%完成时提前停止。

**⚠️ 局限性**

局限包括仅接受直接跟随闭包，无法检测伪造轨迹；依赖手工设计的活动提取函数；对大规模或弱序列化的代理可能不够紧凑。

---

## 145. VizAnchor: Decoding Manipulation Intent from Tampering Visualizations via Dual-Anchor Reasoning

**arXiv ID:** 2608.24535 | [PDF](https://arxiv.org/pdf/2608.24535v1)

**作者:** Xiaotian Zhang `[一作]` (East China Normal University), Sicheng Song `[通讯]` (East China Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建双锚定证据（语义锚点与空间锚点），并通过多智能体VLM推理，完成可视化图表篡改检测、定位、解释与误导意图推断。

**💡 创新点**

创新点包括：①引入语义锚点从水印恢复原始图表；②空间锚点结合裁剪和局部编辑掩码提供精细定位；③三步VLM多智能体推理框架，分别为误导者归纳、叙事重建、意图推断，生成结构化的误导描述。

**🔧 技术方法**

技术手段：可逆水印嵌入与恢复（IWM）、crop‑aware 对齐、U‑Net 局部编辑定位、Gemini‑3.5‑flash VLM、多智能体推理。

**📊 数据集**

数据集：VisGuard、VizDefender 以及自建 VizAnchor Dataset（VAD），VAD 包含 1500 对自动生成的图表用于定位训练，120 对人工构造的图表用于评估。

**📈 对比分析**

与现有基线相比，VizAnchor 在水印保真度、元数据恢复、裁剪/局部编辑定位、误导类型识别、组件识别、过程与意图描述等指标上均显著提升（如误导类型准确率 0.91，意图 Cos‑FA 0.75）。

**⚠️ 局限性**

局限：仅适用于已嵌入水印的图表，无法处理已有或无水印的可视化；对全图重生成的篡改无法检测，因水印被破坏；需要进一步结合被动取证信息以应对 AIGC 全新生成的图表。

---

## 146. Contrastive Branch Policy Optimization

**arXiv ID:** 2608.24300 | [PDF](https://arxiv.org/pdf/2608.24300v1)

**作者:** Ying Wang `[一作]` (Alibaba Group), Jingli Yang `[通讯]` (Harbin Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的强化学习框架CBPO，使语言模型在与外部工具交互的多轮推理任务中能够更细粒度地为中间决策分配奖励信号

**💡 创新点**

核心创新在于将候选分支定位与局部奖励分配解耦：先用生成熵全响应扫描并通过路径/节点衰减实现预算分配，再用Exact‑Prefix组内奖励方差（Contrastive Branch Value）来估计决策敏感度，并通过受限剪裁保持梯度方向

**🔧 技术方法**

技术手段包括：全响应熵扫描、路径/节点衰减的预算分配、Exact‑Prefix组构建、CBV标准化与受限剪裁、前缀屏蔽与层级分段以避免共享词重复奖励、PPO/GRPO优化框架

**📊 数据集**

在十个基准上评估，包含5个数学推理任务（AIME 2024/25、MATH‑500、GSM8K、MATH）和5个知识检索任务（WebWalker、HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle），使用Qwen3-1.7B和Qwen3-4B两种规模的模型

**📈 对比分析**

与多种基线（包括SFT、OPD、GRPO、REINFORCE++、DAPO、GSPO、EAPO、OC‑GRPO、GIGPO、Tree‑GRPO、ARPO）在相同预算下对比，CBPO在数学平均准确率从66.0%到69.3%、搜索平均准确率从53.2%到57.5%均超过对手，尤其在大部分单项任务上夺冠

**⚠️ 局限性**

限制在于全响应熵扫描的效果尚未与仅工具边界扫描进行严格对照，且仍需在真实部署中检验安全性和成本（分支采样带来额外生成开销）

---

## 147. Recursive Agentic Reasoning

**arXiv ID:** 2608.23956 | [PDF](https://arxiv.org/pdf/2608.23956v1)

**作者:** Shengxin Zhang `[一作]` (Google), Jing Xie `[通讯]` (University of Maryland)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将迭代细化、问题分解和重复采样三类测试时推理方法统一为递归算子，在相同的提示、预算与评测代码下对四种算子进行实验对比。

**💡 创新点**

创新点包括：①建立统一的算子实验框架；②发现采样+投票的提升主要来自对预算耗尽导致的空输出的恢复，而非路径边缘化；③通过paired scoring展示无配对评估会导致结果偏差；④表明不需要复杂的算子路由器。

**🔧 技术方法**

使用递归算子实现（Additive、Reductive、Search）基于solve函数；自定义终止判据、deduplication、排除传输失败；统一代理进行模型调用；采样使用温度0.7、N=5；评测采用pairwise scoring。

**📊 数据集**

使用五个基准：MuSiQue（多跳推理）、HLE（专家级问答）、BBEH（通用多跳）、SuperGPQA（研究生知识）、Omni‑MATH（奥数），涵盖多种推理场景。

**📈 对比分析**

在14个模型×基准细胞（DeepSeek‑V4‑Pro、MiniMax‑M3、Qwen3.6‑plus）下，总计49,327条评测项，151,876次模型调用；在paired协议下，采样算子在所有细胞均优于单遍，平均提升约+6分；Additive和Reductive表现不稳定，部分细胞下降；采样算子虽调用数最高，但能获得最多准确度。

**⚠️ 局限性**

局限性：①未使用加权投票或树搜索，采样仅为固定N=5且无早停；②仅评估无训练的黑盒算子；③仅适用于产生隐式推理流的模型，未验证对无此行为模型的适用性；④未给出显著性检验；⑤HLE评测为下限；⑥Omni‑MATH未跑Qwen3.6‑plus。

---

## 148. MatReplace: A Reference-Free, Conditioning-Aligned Benchmark for Material Replacement in Interior Scenes

**arXiv ID:** 2608.24107 | [PDF](https://arxiv.org/pdf/2608.24107v1)

**作者:** Mingzhe Du `[一作]` (National University Of Singapore), Luu Anh Tuan `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了一个专门针对室内设计中材质替换任务的基准（MatReplace），并提出了无参考、四维度的评估协议；

**💡 创新点**

创新点在于：1）单一、可验证的四维度评价体系，去除了对单一参考图的依赖；2）三条对齐的条件轨道（文本、文本+掩码、掩码+参考），实现跨模型对比；3）通过人工专家校准验证评估与人类偏好的一致性；

**🔧 技术方法**

采用了SigLIP2分类、DINOv2相似度、LPIPS、SSIM、单目深度估计、低通亮度相关性等已有视觉指标，并将其组合成四个维度评分；

**📊 数据集**

数据集基于GPT-Image-2生成的2,017个任务样本，经过算法和语义审核后冻结为1,421个任务，涵盖10类材质和6类表面；

**📈 对比分析**

对11种系统（11条轨道）进行单核种子评测，使用无参考评估聚合（gate）与原始均值两种方式比较；在Track A中，封闭模型（Nano-Banana-2-Lite、GPT-Image-2）领先，开放模型（Qwen、BAGEL、OmniGen2、FLUX）分布在第二至第四位；在Track C中所有模型表现显著下降，部分模型甚至低于“保持不变”的基准；

**⚠️ 局限性**

局限性包括：1）全部合成数据且单源生成，缺乏真实照片验证；2）对掩码质量的依赖导致覆盖率仅70%；3）人工评估样本仅两位评审，交叉评议有限；4）对非家具表面（墙、台面）的覆盖相对不足；5）闭源模型与生成器之间存在一定的同源偏差。

---

## 149. MC-CXR: A Multi-Context Chest X-ray Benchmark for Context-Induced Disruption in Vision-Language Models

**arXiv ID:** 2608.24118 | [PDF](https://arxiv.org/pdf/2608.24118v1)

**作者:** Junhyeok Lee `[一作]` (Seoul National University), Kyu Sung Choi `[通讯]` (Seoul National University Hospital)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Multi-Context Chest X-ray (MC-CXR) 基准，采用可靠与误导性上下文的配对扰动，评估视觉语言模型在保持图像判定正确性的同时如何使用或抵制临床上下文。

**💡 创新点**

创新点在于将三类任务（图像仅识别 IOR、可靠上下文稳定 RCS、误导上下文抵抗 MCR）与两项内在度量（switch‑to‑wrong 与 context‑aligned error）结合，首次量化医学 VLM 的上下文鲁棒性与文本‑视觉偏差。

**🔧 技术方法**

技术手段包括零样本 constrained‑letter 直接回答提示、配对扰动案例构造、Cohen’s κ、switch‑to‑wrong 与 Y‑aligned error 等指标的计算，并对 10 个公开/闭源 VLM 进行实验。

**📊 数据集**

数据来源为 MIMIC‑CXR 的 240 条放射学案例，结合 CheXpert 标签，并人工标注可靠与误导性文本、先前图像和视觉覆盖，扩展成 2,522 个评估实例。

**📈 对比分析**

在 IOR 条件下模型准确率仅 18–34%，可靠上下文下平均切换率为 28.8%，误导文本导致 60–78% 的切换且 64–85% 与误导标签一致，而误导视觉导致 35–62% 切换但仅 16–18% 与误导标签一致，表明文本对 VLM 的影响远大于视觉。

**⚠️ 局限性**

局限性包括样本量有限、单评审、单次确定性推理、对提示协议敏感、未提供回避选项、误导视觉条件信息量不足，以及模型与 MIMIC‑CXR 数据集重叠未完全审计。

---

## 150. ESQ-Bench: A Multi-Tier Enterprise Oracle Benchmark for Evaluating NL2SQL Dialect Generalization and Silent Semantic Divergence

**arXiv ID:** 2608.23569 | [PDF](https://arxiv.org/pdf/2608.23569v1)

**作者:** Sanjay Mishra `[一作]` (Independent Researcher), Ganesh R. Naik `[通讯]` (Torrens University Australia)

**通讯引用:** 28660 | [OpenAlex ID](https://openalex.org/A5029523168)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 ESQ‑Bench，一个面向 Oracle 数据库的 NL2SQL 基准，包含六个企业级填充完毕的架构、550 题库以及四个评估指标（EM、EX、SR、SD）。

**💡 创新点**

创新点在于①构建了完整的 Oracle‑first NL2SQL benchmark；②引入 Silent Divergence (SD) 指标揭示执行匹配的盲点；③设计了系统化的三层复杂度分层和四指标评估框架；④同步支持 PostgreSQL、MySQL、SQL Server 等多方言。

**🔧 技术方法**

使用了 LLM 生成 SQL（GPT‑4o、Claude Sonnet 4.6、Llama 3.2）结合 schema‑linked prompting 与 zero‑shot；评估 harness 运行在 Oracle 21c 上，并对失败进行 F1–F4 分类。

**📊 数据集**

数据集为六个完全填充的企业级 Oracle 架构（共 465 张表、164,682 行），在 PostgreSQL、MySQL、SQL Server 同步；550 题问–查询对经过人工金标验证。

**📈 对比分析**

通过在三层 Tier 上对 GPT‑4o、Claude Sonnet 4.6、Llama 3.2 进行 EX、EM、SR、SD 等指标评估；GPT‑4o 在 Tier1 达到 79.8% EX，Tier2 60.3%，Tier3 57.2%；Claude Sonnet 4.6 在所有 Tier 上均优于 GPT‑4o；Zero‑shot 在 Tier2‑3 的 EX 甚至超过 schema‑linked。

**⚠️ 局限性**

局限性包括：数据集为人工构造的企业样本，未覆盖真实生产数据库；正式 SD 评估仅完成 Tier1，Tier2‑3 仍在构建；模型评测受 LLM 版本变化影响；仅在 Oracle 21c 上测试，未包含 23ai 等新特性；缺少多轮交互和其他商业方言的评估。

---

## 151. PlaceSeek: Human-Centered Geospatial Retrieval of Urban Outdoor Places via Semantic Grounding and Affective Alignment

**arXiv ID:** 2608.24133 | [PDF](https://arxiv.org/pdf/2608.24133v1)

**作者:** Ziqi Cui `[一作]` (University of British Columbia), Shangyu Lou `[通讯]` (University of California, Santa Barbara & San Diego State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了PlaceSeek框架，能够基于自然语言查询检索满足人类活动与情感需求的城市街景地点，并通过物理证据与情感匹配两步提高检索质量。

**💡 创新点**

创新点在于：①将查询意图拆分为物理证据、情感需求、活动支持等；②使用语义对齐模块（SGM）通过Coarse–Fine的物理验证（OpenCLIP+GroundingDINO+Qwen3-VL）确保检索结果具备所需实体；③使用情感对齐模块（AAM）通过LoRA微调OpenCLIP并映射到Place Pulse维度，实现对用户情感期望的精细对齐。

**🔧 技术方法**

技术包括：LLM（ChatGPT‑4o）意图解析；OpenCLIP ViT‑L/14 + prompt bank；GroundingDINO对象定位；Qwen3‑VL视觉问答验证；LoRA微调OpenCLIP用于情感对齐；多层基于规则的重排序。

**📊 数据集**

使用的数据集为：Milan市Google Street View 127,824张图；Place Pulse 2.0（人类情感判断数据）用于情感对齐训练；十个多样化自然语言查询；人工标注评价数据。

**📈 对比分析**

与四个基线（CLIP、FT‑CLIP、SigLIP、VQA‑Qwen3）在10个查询上进行对比；PlaceSeek在Precision@5/10/20、MeanMatch、nDCG上均领先，Precision@5最高达88%，Precision@20达89.5%。Ablation实验表明SGM与AAM各自贡献显著。

**⚠️ 局限性**

局限性包括：评测仅在单一城市Milan；情感对齐受Place Pulse六维限制，难以捕捉更丰富的情感概念；街景仅视觉信息，无法判断安全性、可达性等实际属性；未来需跨城市、多文化验证并整合更多情境与空间约束。

---

## 152. Opals.jl: a comprehensive, composable framework for data assimilation in Julia

**arXiv ID:** 2608.24265 | [PDF](https://arxiv.org/pdf/2608.24265v1)

**作者:** Nicholas Mueller `[一作]` `[通讯]` (Delft University of Technology), Nicholas Mueller (Delft University of Technology)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了Opals.jl Julia包，提供统一的框架实现多种数据同化与逆问题方法，并支持ODE、PDE及ROM后端。

**💡 创新点**

创新点在于：①采用类型驱动、可组合的接口，能在不改动核心代码的情况下将协方差定位、噪声估计、偏差校正、ROM校准等高级功能以包装器形式叠加；②实现与SciML、Gridap/GridapROMs的无缝交互，允许直接用ODE、全阶或降阶PDE解算器作为转移模型；③构建了可复用的高层 API，简化多阶段实验流程。

**🔧 技术方法**

主要技术包括：Kalman、EnKF、UKF、粒子滤波、3D/4D‑Var 等传统DA算法；协方差定位/膨胀、在线噪声协方差估计、RC（循环神经网络）偏差模型、基于克里金插值的ROM误差校准；Julia的多分派、JIT、SciMLSensitivity、GridapTopOpt、GridapROMs 以及自研的 RC 训练实现。

**📊 数据集**

使用仿真数据集：Lorenz‑63、Van der Pol、二维 Navier‑Stokes 方框腔流以及热方程，观测为稀疏且带噪声或偏差；没有使用公开真实观测数据集。

**📈 对比分析**

在四个基准上比较了 EnKF、UKF、SIR 粒子滤波和变分方法，且对偏差-aware EnKF 与无偏差 EnKF 进行对比；通过 RMSE、创新分布、时间/内存等指标评估性能。结果显示：Opals 在准确性上优于现有 Julia 库，RC 与 ROM 校准能显著提升精度，且计算速度比全阶模型提升 15–30 倍。

**⚠️ 局限性**

局限性：对高度非线性、高维系统仍可能出现滤波器发散；ROM 校准依赖离线训练，可能无法覆盖全参数空间；多保真/多分辨率方法尚未实现；对极大规模 PDE 的计算资源需求仍高；缺乏真实观测数据的验证。

---

## 153. SENSESHIFT: Continuous Sentiment-Controlled Text Generation via Encoder-based Mask Infilling

**arXiv ID:** 2608.24304 | [PDF](https://arxiv.org/pdf/2608.24304v1)

**作者:** Shahed Masoudian `[一作]` (Johannes Kepler University), Markus Schedl `[通讯]` (Johannes Kepler University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于Encoder的细粒度情感控制文本生成框架，利用量化情绪token和迭代掩码填充实现句子级情感编辑。

**💡 创新点**

创新点在于：①引入双向注意力与量化情绪信号联合进行句子级情感控制；②采用迭代mask infilling克服传统decoder单向生成的局限，提升上下文连贯性；③实现轻量化、低参数规模的情感控制模型。

**🔧 技术方法**

技术细节包括：Encoder MLM微调、情绪token量化与预置、迭代掩码填充、温度与多样性惩罚的Beam Search、VADER情感评分作为监督信号。

**📊 数据集**

使用数据集：TinyStories（约4.5M条GPT-3.5/4生成短篇故事）和Yelp Reviews（约30万条人写评论），两者分别用于训练、验证和评估。

**📈 对比分析**

评估方法：在in‑domain与out‑of‑domain场景下与prompting、instruction tuning、token instruction tuning、activation steering等基线进行比较；采用Δ_s、Corr、Acc、PPL和Δ_f等指标。实验显示该方法在情感控制精度、文本流畅度和上下文适配度上均优于所有基线。

**⚠️ 局限性**

局限性：①依赖VADER情感分析器，细腻情绪捕捉受限；②训练数据情感分布不均，极端情绪控制效果较差；③数据领域有限，泛化能力未完全验证；④存在潜在误用风险，需配合使用限制与审计。

---

## 154. PinSieve: Production Selective VLM Serving and a Governed Memory Flywheel for Enterprise Content-Quality Triage

**arXiv ID:** 2608.24040 | [PDF](https://arxiv.org/pdf/2608.24040v1)

**作者:** Chuqing Gao `[一作]` (Pinterest), Andrey Gusev `[通讯]` (Pinterest)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在生产环境中部署了一个可选的VLM服务代理（Serving Agent），只在难以判定的“灰区”流量上做决策，并结合了基于反馈记忆、可治理重放以及规范化推理审核的离线生命周期管理。

**💡 创新点**

创新点在于将紧凑的VLM嵌入现有多级审核链路中，形成有限范围的自动化决策，并通过反馈记忆记录观察路径、审核概率，构建可治理的重放与升级流程（DC‑Replay）以及对教师生成推理的 keep/repair/drop 审核机制，兼顾性能、成本与可追溯性。

**🔧 技术方法**

主要技术包括：2B 公开预训练VLM的微调、知识蒸馏与推理文本蒸馏、阈值控制的单标量路由、逆倾向加权估计、代表性+不确定性+近期样本组合的可治理重放、离线 Shadow‑Test 与滚动升级。

**📊 数据集**

使用内部生产数据，约 6 个月的内容质量二分类样本（图像+文本），共计数百万条记录，目标类别为低质量内容。

**📈 对比分析**

对比结果：在生产上，灰区自动通过率从 20.48% 提升至 41.99%，审核生产率提升 25.7%，归一化成本降低 16%，信号交付从次日改为当日；在离线重放实验中，FNR@50% 从 17.73% 降至 13.29%，PR‑AUC 亦有提升；与传统随机重放相比，DC‑Replay 在保持分布的同时进一步降低误判率。

**⚠️ 局限性**

局限性包括：仅在单一信号上验证，其他任务需进一步验证；重放策略对选择性反馈极为敏感，若观测路径或审核概率记录不充分会导致偏差；部署受限于已有轻量级前端与审核通道；可治理流程需要人力审计，扩展性受限；模型规模仍受成本约束，无法做到全局 VLM 服务。

---

## 155. WeMM-Embedding: WeChat Multi-Modal Embedding Technical Report

**arXiv ID:** 2608.24053 | [PDF](https://arxiv.org/pdf/2608.24053v1)

**作者:** Junjie Zhou `[一作]` (Tencent Inc.), Jing Lyu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了WeMM-Embedding，一系列基于Qwen3.5的多模态嵌入模型，支持文本、图像、视频及其交互组合。

**💡 创新点**

创新点在于统一pair-based格式、两阶段训练策略、以及跨尺度知识蒸馏与重排序监督相结合的精细化学习。

**🔧 技术方法**

采用的技术包括自监督对比学习、分级相关学习、Matryoshka表示、语义ID重采样、硬负样本构造、embedding-teacher蒸馏和视觉输入扩展。

**📊 数据集**

数据集来源于数亿级弱监督图文/视听对、标题对、检索对、分类对、问答对和分级相关对，经过语义ID重采样与质量审核后形成精细化训练集。

**📈 对比分析**

与MMEB、Gemini、Nova MME等基线对比，2B/4B/9B模型在MMEB-v2/3、12项跨模检索和内部26项任务上均领先或同等，最高可达MMEB-v2 80.6分、跨模平均81.7%。

**⚠️ 局限性**

局限在于目前仅支持文本、图像、视频三种模态，音频任务无支持；模型规模受限于9B且未公开权重；对多模态组合的推理复杂度仍高。

---

## 156. Anatomy of a Scam Call: What 10,000 real scam and spam calls reveal about how phone scammers operate

**arXiv ID:** 2608.24127 | [PDF](https://arxiv.org/pdf/2608.24127v1)

**作者:** Ethan Traister `[一作]` (scam.ai), Simiao Ren `[通讯]` (scam.ai)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对10,211个真实电话诈骗/垃圾电话进行完整录音和转写，分析其业务模式、脚本使用、说服手段，并通过随机赋予虚假身份的实验检验被动年龄对诈骗力度的影响。

**💡 创新点**

首次利用主动嗅探honeypot记录完整对话并随机分配虚假身份，揭示诈骗者按目标年龄进行资源分配的策略，同时提出基于开场句子的早期内容检测基准。

**🔧 技术方法**

采用AI语音代理、语音识别、LLM分析管道，结合TF-IDF+逻辑回归、句子嵌入、Qwen2.5小型语言模型以及k-means聚类脚本，构建完整的分析与预测模型。

**📊 数据集**

使用一个封闭的美国英语语音数据集，包含54天内10,211个入站电话（913小时录音、330,956轮次、5,780个来源号码）和相应的自动化标签。

**📈 对比分析**

通过呼叫前几句的模型阶梯（TF-IDF、句子嵌入、LLM）在无重叠号码的 hold‑out 集上评估ROC‑AUC和AP，TF-IDF模型在第八句即可达到0.87 ROC‑AUC，LLM未显著超越；随机实验显示年龄越大，诈骗者话语轮数提升约1.15倍，但请求概率无显著差异。

**⚠️ 局限性**

限制包括：数据仅为美国英语、标签为自动化银色标注、仅10个虚拟身份导致样本量有限、实验时间短、年龄与性别交织、代理本身的干预难以完全消除等。

---

## 157. Function-Level Execution Feedback for Code Preference Optimization

**arXiv ID:** 2608.23632 | [PDF](https://arxiv.org/pdf/2608.23632v1)

**作者:** Idris Nechnech `[一作]` (Seoul National University), Jungwoo Lee `[通讯]` (Seoul National University)

**通讯引用:** 11387 | [OpenAlex ID](https://openalex.org/A5100376261)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于模块级函数的代码过程监督框架STEP-KTODER，用执行单元测试提供局部正确性标签并结合KTO进行偏好优化。

**💡 创新点**

创新点是将代码生成分解为可独立测试的函数步骤，使用执行基础的二元步骤标签与全局标签联合训练，实现局部错误定位与纠正。

**🔧 技术方法**

采用Kahneman–Tversky Optimization (KTO) 与其步骤化扩展、自动函数分解、单元测试生成、LoRA微调等技术。

**📊 数据集**

训练使用TACO、APPS；评估在HumanEval、MBPP、BigCodeBench（Full/Hard）和LiveCodeBench等七个基准上。

**📈 对比分析**

与基础模型、DPO、KTO、Target-DPO等基线对比，STEP-KTODER在较难的BigCodeBench Hard和LiveCodeBench上分别提升约+11%和+8%，其余基准保持或略优。

**⚠️ 局限性**

限制包括依赖可分解为函数的代码、需要模型在策略上生成候选、单元测试质量对标签可靠性至关重要，且在更复杂的I/O或极端问题上效果未知。

---

## 158. Squeezing the Cache, Preserving the Truth: Monotonic Equipotential Allocation with Geodesia-KV

**arXiv ID:** 2608.23599 | [PDF](https://arxiv.org/pdf/2608.23599v1)

**作者:** Vincenzo Dentamaro `[一作]` (Geodesia.ai), Giuseppe Pirlo `[通讯]` (Geodesia.ai)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练‑free 的 KV 缓存压缩框架，利用单调块级精度分配、精确的率‑失真残差和查询稀疏读取，实现长上下文推理的显存高效管理。

**💡 创新点**

创新点在于：① 将 KV 缓存视为按块的离散率‑失真问题；② 设计了多层精度梯度 {16,8,4,2,中心} 并用 Lagrangian 分配；③ 将 Quest 的查询稀疏机制与压缩后的 KV 表结合，形成“Compressed‑Quest”，兼具低显存占用与 100% 检索准确率。

**🔧 技术方法**

核心技术包括：块级离散量化、率‑失真 Lagrangian 分配、指数移动平均注意力质量估计、Grouped‑Query Attention、vLLM 原生块页管理、在线 softmax 量化解包核。

**📊 数据集**

实验数据集：PG‑19、Qwen 官方预训练数据、不同长度的 16k/1M 书籍窗口；模型覆盖 Qwen2.5‑3B、Qwen3‑8B、Qwen3‑5‑0.8B、30B-A3B 等。

**📈 对比分析**

通过与 Full‑KV、Quest、StreamingLLM、SnapKV、KIVI 等基线在相同 Q=1 条件下的 perplexity、位率、读写比率、检索准确率进行对比；在 3B、8B 上实现 71–84% VRAM 节省，1M‑token 上仍保持 100% Needle‑In‑a‑Haystack 检索，且在部分窗口的 perplexity 与最优基线相当或更优。

**⚠️ 局限性**

局限性：对极端大窗口或不同模型结构的迁移性尚待验证；在一些窗口的 perplexity 提升有限；需要更细粒度的误差分析和对分布漂移的鲁棒性研究。

---

## 159. Hierarchical Skill Retrieval for Data-Efficient Adaptation of Vision-Language-Action Models

**arXiv ID:** 2608.24042 | [PDF](https://arxiv.org/pdf/2608.24042v1)

**作者:** Haoran Hao `[一作]` (Carnegie Mellon University), Jeffrey Ichnowski `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于任务分解与层次检索的 VLA 模型自适应框架 HSR，以在数据稀缺的目标任务中实现高效学习；

**💡 创新点**

创新点在于将 LLM 生成的子任务序列与基于先验数据的技能可靠性评分相结合，构建层次化检索与两阶段预训练+微调策略；

**🔧 技术方法**

采用 LLM（Qwen3‑VL‑4B）进行任务分解、BERT 文本嵌入做语言检索、VAE 提取行为特征做重排序，以及行为克隆（BC）损失评估技能可靠性；

**📊 数据集**

使用 LIBERO 机器人模拟环境、Open X‑Embodiment 与 DROID 先验数据集，以及在 xArm 7‑DoF 实验平台收集的真实任务演示；

**📈 对比分析**

与 BC、随机采样、全量数据、语言检索、FR、STRAP、BR、IWR 等基线对比，HSR 在 LIBERO 上平均提升 10.3% 成功率，在真实任务上提升 21.3%，显著优于最强对照方法；

**⚠️ 局限性**

局限性包括仅在操纵任务与单一 VLA 后端上验证、LLM 与闭环控制集成不足、跨实现差距导致检索样本噪声大，以及对极端失败情况的鲁棒性不足。

---

## 160. Mahalanobis-Based Multi-Head Attention for Complex State Propagation

**arXiv ID:** 2608.24462 | [PDF](https://arxiv.org/pdf/2608.24462v1)

**作者:** Xiaohe Li `[一作]` `[通讯]` (GuangDong Police), Xiaohe Li (GuangDong Police)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在复杂数值状态传播(CSP)的基础上提出MHA‑CSP，将注意力改为基于马氏距离的多头注意力，并加入树结构距离累积与跨头融合机制，实现更强的层级推理。

**💡 创新点**

创新点包括：① 用马氏距离直接计算注意力，省去 Q/K/V 投影；② 通过 LogSumExp 对树结构距离进行修正，天然编码嵌套层级；③ 通过跨头困惑矩阵实现多头协作，提升鲁棒性与效率。

**🔧 技术方法**

技术手段包括：复杂值状态传播、马氏距离注意力、树结构距离累积、LogSumExp 修正、跨头融合、教师强迫仅在最终隐藏状态。

**📊 数据集**

数据集：四个确定性状态追踪任务——(1) 带复制的嵌套算术表达式推理、(2) 括号匹配、(3) 模 3 计数、(4) 奇偶性检查，序列长度最长 128，生成 200k 训练样本、20k 测试样本。

**📈 对比分析**

对比 LSTM、GRU、GDN、ARFormer（小型 Transformer）和 Vanilla CSP，MHA‑CSP 在括号匹配上 50%→100%（Vanilla 100%）、算术+复制上 50%（Vanilla 30%）并比 ARFormer 提升约 18%，在奇偶性检查上 100% 兼具效率，参数约 119K，训练成本低。

**⚠️ 局限性**

局限性：① 计算复杂度为 O(T²)，对长序列内存瓶颈；② 多头可能发生“坍塌”，需依赖困惑矩阵调节；③ 训练过程存在“grokking”波动，收敛不稳定；④ 仅在特定结构推理任务验证，需进一步检验跨领域泛化。

---

## 161. STAIN-FL: Stealthy Targeted Attack Injection with Contextual Triggers in Federated Learning

**arXiv ID:** 2608.23952 | [PDF](https://arxiv.org/pdf/2608.23952v1)

**作者:** Ashlinder Kaur `[一作]` (Singapore Institute of Technology), Tram Truong-Huu `[通讯]` (Singapore Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在联邦视频异常检测中提出STAIN‑FL隐蔽针对性后门注入框架。

**💡 创新点**

创新点是利用自然监控条件（低光、室内、拥挤）作为触发器，结合对最少更新梯度的掩蔽，既保持干净准确率，又实现后门的长期持久性。

**🔧 技术方法**

采用FedAvg/FedProx聚合、少量梯度掩蔽、I3D 1024维特征提取、异常转正标签重标记等技术。

**📊 数据集**

使用一个包含1900条真实监控视频、各950条正常与异常样本的数据集，视频转化为1024维I3D特征。

**📈 对比分析**

通过稀疏与连续攻击、不同梯度掩蔽比例的对比，发现稀疏攻击在FedAvg/ FedProx下清洁准确率下降≤1.66%，但后门准确率可达56.7%（FedAvg）/54.2%（FedProx），且攻击后平均336轮仍保持>25%后门效果。

**⚠️ 局限性**

局限性包括仅评估单一受攻击客户端、仅使用固定触发条件、未探究多客户端协同或主动触发策略的影响。

---

## 162. Hierarchical Prototype-Memory Adaptation of SAM for Surgical Instrument Segmentation

**arXiv ID:** 2608.24541 | [PDF](https://arxiv.org/pdf/2608.24541v1)

**作者:** Xinning Yao `[一作]` (Beihang University), Bo Liu `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过构建多尺度冻结的原型记忆库，并将其通过轻量化适配器层级化注入到SAM3模型中，实现对手术仪器分割的自适应。

**💡 创新点**

创新点在于将视觉证据拆分为全球、结构和局部三层记忆，分别与文本特征、解码器查询和高分辨率特征对齐，避免单一路径提示瓶颈并保持原型的持久性。

**🔧 技术方法**

技术包括基于SAM3的多尺度特征提取、K‑means聚类构建原型记忆、轻量化适配器（残差映射）、文本-原型注意力、结构原型注入解码器、局部对齐损失等。

**📊 数据集**

使用公开的 EndoVis2017 与 EndoVis2018 两个手术仪器分割基准数据集进行训练与评估。

**📈 对比分析**

与多种专用手术分割网络（如ISINet、MATIS、S3Net）以及基于SAM的对比方法（TrackAnything、PerSAM、SurgicalSAM、MA‑SAM2）进行对比，HPMA 在 Challenge IoU、mean IoU、mean Dice 等指标上均达到了最高成绩，显著提升了鲁棒性。

**⚠️ 局限性**

局限性包括：仍需依赖大规模预训练模型，对原型数量和聚类参数敏感；适配过程在推理时略微增加计算量；在实时视频序列或极端光照、显微镜环境下的性能尚未充分验证。

---

## 163. SeMoCo: A Semantic-First Motion Codec for Motion Language Modeling

**arXiv ID:** 2608.24334 | [PDF](https://arxiv.org/pdf/2608.24334v1)

**作者:** Tianlv Huang `[一作]` (Jilin University), Xin Zheng `[通讯]` (Frontier Robotics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 SeMoCo 语义优先运动编码器和双轴生成器，用于基于语言条件的人类运动生成。

**💡 创新点**

创新点在于每个运动令牌同时包含语义令牌和运动残差序列，先建模语义进程，再自回归细化运动细节，实现语义与运动的显式分离。

**🔧 技术方法**

采用了语义优先的运动编码技术、双轴自回归生成器以及 SOMA 表示框架，配合重建驱动的训练和语义级序列建模。

**📊 数据集**

使用了自建的 Ω‑MotionVerse 大规模多源人类运动数据集，统一在 SOMA 表示之下。

**📈 对比分析**

在与多种编码器的重建准确率对比中，SeMoCo 取得最佳成绩；在文本到运动的生成任务中也表现出较强的效果，证明其运动令牌对下游生成任务有利。

**⚠️ 局限性**

局限性包括：1）仍受重建驱动层次的影响，可能对细粒度语义控制不足；2）对多样化场景的泛化能力尚未充分验证；3）需要进一步探索不同语义粒度对运动生成的具体影响。

---

## 164. Effective Pivot Attack Detection via System and Network Information

**arXiv ID:** 2608.23731 | [PDF](https://arxiv.org/pdf/2608.23731v1)

**作者:** Ava Powelson `[一作]` (Dalhousie University), Israat Haque `[通讯]` (Dalhousie University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一种基于宿主机的实时枢纽攻击检测系统，利用 eBPF 对内核系统调用与网络流进行追踪并结合信息流控制（IFC）与流特征，实现在不依赖网络设备合作的前提下检测内部横向流量的转发。

**💡 创新点**

创新点在于：①将系统级进程追踪与网络流特征相结合，先通过 IFC 建立入站与出站流的因果关系，再通过时间/大小窗口与端点访问频率过滤，显著降低误报；②采用 eBPF 在内核层实现高性能、低开销的实时检测；③设计了基于 LRU 的内存回收机制，保持检测精度的同时内存占用极低。

**🔧 技术方法**

技术主要包括：eBPF 程序与映射（socketIn/Out、tgids、flowStats、endpoints）；进程监控（execve/clone/accept/exit）；网络流监控（XDP/Tc hook）；信息流控制（IFC）与端点频率分析；LRU 哈希表回收；C++/Python 用户空间代理。

**📊 数据集**

使用了加拿大网络安全研究院的 CIC‑IDS2017 数据集（模拟 Web 服务器流量）并在此基础上注入四种常见枢纽攻击（SoCat、Chisel、SSH、Nmap）进行实验；同时在真实校园网络的两台生产服务器上持续部署 80 天进行实测。

**📈 对比分析**

对比基于纯流特征的 FCB（flow‑characteristics‑based）方案，实验证明：①检测准确率提升 31.49%；②平均误报率从 42.38% 降至 0.18%；③在真实部署中误报率仅 0.006%，CPU 开销 <0.1%，内存占用 <350 MB，系统对性能影响可忽略。

**⚠️ 局限性**

局限性包括：①需要宿主机具备 Linux 6.x 内核与 eBPF 支持；②假设攻击者不拥有 root 级别访问，若宿主机被完全控制可绕过；③阈值（时间/大小/端点访问）需要管理员手动调优，误报仍可能来自合法端口转发或高对称流量；④未评估在极高并发或异构操作系统环境下的可扩展性；⑤对某些高级伪装技术（如填充包、延迟注入、频繁新连接）仍有一定的容忍空间。

---

## 165. Quantifying System-Level Harms from AI Adoption in Complex Sociotechnical Systems

**arXiv ID:** 2608.23906 | [PDF](https://arxiv.org/pdf/2608.23906v1)

**作者:** Paul Vautravers `[一作]` (Advai Ltd), Damian Ruck `[通讯]` (Advai Ltd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种将系统理论风险分析、组件级实验与概率系统建模相结合的框架，用于评估 AI 在复杂社会技术系统（以英国实时总清算系统为例）中的系统级危害；

**💡 创新点**

创新点在于从系统层面出发，构建可追溯的 AI 失效路径链路，并将基于 STPA 的定性分析与 LLM 对抗实验和金融传染模型的定量结果融合，实现对 AI 失效对整个金融网络冲击的可量化评估；

**🔧 技术方法**

采用的技术包括：系统理论过程分析（STPA）、大语言模型（LLM）对抗测试（prompt injection）、金融传染模型（Eisenberg–Noe 框架改进）以及 Monte‑Carlo 统计模拟；

**📊 数据集**

使用的数据集主要为生成的金融新闻文本（四类资产、不同情绪标签）和公开的银行网络结构与资产负债表参数；

**📈 对比分析**

通过对比非对抗与对抗场景下的资产配置偏移、火售强度与银行倒闭率，表明对抗攻击会显著提高系统的脆弱性（倒闭率上升、冲击门槛降低），但实验规模受限，未覆盖更复杂的攻击与模型；

**⚠️ 局限性**

局限性包括：对 STPA 结果的主观判断、组件到系统映射的简化假设、模型参数敏感性、对抗实验仅涉及简单攻击、缺乏与真实金融机构的实证验证等。

---

## 166. MolEmb: Multimodal Large Language Models Can Be Strong Molecular Embedding Models

**arXiv ID:** 2608.23646 | [PDF](https://arxiv.org/pdf/2608.23646v1)

**作者:** Xinjian Zhao `[一作]` (Chinese University of Hong Kong), Tianshu Yu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5075551272)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过将多模态大型语言模型（MLLM）适配为通用分子嵌入模型，提出MolEmb框架；

**💡 创新点**

在分子嵌入中首次引入多视图输入（二维图像+SMILES）与自然语言语义上下文，并利用双向对比损失对分子与文本进行统一对齐；

**🔧 技术方法**

使用预训练的MLLM（Intern-S1-mini、Qwen3‑VL‑8B、Qwen3.5‑0.8B）+轻量级LoRA适配器；对分子和文本分别在MLLM中提取隐藏状态并池化为固定长度向量；

**📊 数据集**

主要数据集包括MolTextNet、KnowMol‑100k、ChEBI‑20‑MM用于对齐训练，MolCAR‑Train、MolCAR‑Structured、MolCAR‑Natural用于上下文检索评估；另外使用标准化学性质预测基准（ESOL、Lipophilicity、FreeSolv、BACE、BBBP、ClinTox、Tox21、SIDER）做下游评估；

**📈 对比分析**

在属性预测上与监督式GNN、图预训练方法相比，MolEmb在多项回归和分类任务上均能取得与最优基线相当或更优的RMSE/ROC‑AUC；在分子–文本检索中，经过对齐的MolEmb在R@1上突破80%以上，远优于未对齐的通用MLLM；在MolCAR上下文检索中，经过“持续对齐”后，Context R@1从≈40%提升至≈100%（Intern‑S1‑mini）或≈70%（Qwen3.5‑0.8B），显示出显著的任务指令敏感性；

**⚠️ 局限性**

局限性在于单纯的分子–文本对齐不足以实现可靠的上下文感知检索，仅通过多样化、结果驱动的任务标签监督才能显著提升；此外，MLLM的尺寸和算力需求仍高，且对特定任务的微调仍需要额外的数据与计算资源。

---

## 167. Multilevel Fair Allocation under Additive Preferences

**arXiv ID:** 2608.24400 | [PDF](https://arxiv.org/pdf/2608.24400v1)

**作者:** Maxime Lucet `[一作]` (LIP6, CNRS, Sorbonne Université), Nicolas Maudet `[通讯]` (LIP6, CNRS, Sorbonne Université)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在树结构层级组织的多层公平分配问题，针对内部节点仅作为子节点的代表并采用社会福利聚合，提出了三种基于估计的加权厌恶公平（WEF1）概念，并设计了多层加权回合制算法（MWRR）来实现完整分配并满足其中的某些公平性。

**💡 创新点**

创新点在于：①首次将厌恶公平概念推广到多层层级情形，并给出三种估计方法（悲观、无偏、乐观），揭示它们的层级关系；②证明在相同偏好下三种概念等价，并证明MWRR能够在此情形下同时满足完整性和多层WEF1；③在一般加权偏好下证明MWRR只能保证悲观WEF1，并给出存在性和近似性的负面结果；④通过大规模仿真实验验证MWRR在实际实例中对无偏WEF1的高度满足率。

**🔧 技术方法**

技术方法包括：树结构建模与多层分配定义；利用加权社会福利聚合与加权厌恶公平的数理定义；构造MWRR算法并证明其多层WEF1保障与多项式复杂度；理论证明层级公平性与存在性、近似性；实验设计采用平衡二叉树、梳状树和部分非平衡树，配合四种偏好生成模型（Mallows、Dirichlet、成本效用、相关效用）以及两种权重分配方式（叶子计数与随机）。

**📊 数据集**

实验数据集为合成实例：树节点数取 {15,31,63,127}（平衡/梳状）和 {21,43,87,175}（部分非平衡），物品数取 m=n 或 m=2n；偏好由上述四种生成模型产生；权重采用 w_i=|children_i| 或随机整数 1~6；每类实例生成 200 条样本，合计 192,000 条。

**📈 对比分析**

比较方式：对每个实例统计 MWRR 输出的分配是否满足 M[agno]-WEF1 与 M[opt]-WEF1，计算满足率并给出 95% 置信区间；同时记录平均运行时间与标准差。结果显示：对无偏WEF1，MWRR 在几乎所有实例（<0.03% 失配）满足；对乐观WEF1，满足率随树结构、规模和权重随机性显著波动，部分大规模随机权重实例近 100% 失配；运行时间在 0.0002–0.049 s 之间，显示算法极快。

**⚠️ 局限性**

局限性：①在一般加权偏好下 MWRR 不能保证 M[agno]-WEF1 或 M[opt]-WEF1，且存在实例无此公平分配；②对乐观WEF1 的近似性上不具备任何正比系数；③理论分析以加权社会福利聚合为前提，未考虑内部节点自身偏好；④实验仅使用合成偏好模型，缺乏真实世界数据验证；⑤关于归一化偏好的存在性问题仍未解决。

---

## 168. CRISP: Calibration-Aware Visual State Space Duality for Remote Sensing Semantic Segmentation

**arXiv ID:** 2608.23746 | [PDF](https://arxiv.org/pdf/2608.23746v1)

**作者:** Kangning Wang `[一作]` (Beihang University), Zhiguo Jiang `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出了CRISP框架，改进Visual State Space Duality（VSSD）网络的全局聚合，恢复高频细节，并设计了正交多原型解码器以保留多模态类别信息。

**💡 创新点**

创新点包括Duality Calibration Operator（DCO）实现残差注入、高通调制与通道重平衡，以及Orthogonal Multi-Prototype（OMP）头在端到端训练中实现多中心正交正则化的结合。

**🔧 技术方法**

使用的技术包括线性时间状态空间模型、VSSD、多方向扫描与全局聚合、频域残差恢复、token级高通调制、通道DC/HC重平衡、正交多原型解码以及相应的正则化损失。

**📊 数据集**

实验使用了ISPRS Potsdam、Vaihingen和LoveDA三个遥感语义分割数据集。

**📈 对比分析**

与SOTA方法（如VSSD、UNetFormer、AerialFormer、SFA-Net等）对比，CRISP在Potsdam上取得88.77% mIoU、Vaihingen上83.00% mIoU、LoveDA上51.56% mIoU，参数约32M，保持了线性复杂度并显著优于大多数基线。

**⚠️ 局限性**

局限性在于仍需依赖VSSD结构，频率恢复受全局聚合的限制，对不同任务或更大规模数据集的泛化性尚未充分验证，并且需要对校准超参数进行手动调优。

---

## 169. Partial Identification under Causal Orders by Linear Programming

**arXiv ID:** 2608.24427 | [PDF](https://arxiv.org/pdf/2608.24427v1)

**作者:** Eric Rossetto `[一作]` (Istituto Dalle Molle di Studi sull'Intelligenza Artificiale), Alessandro Antonucci `[通讯]` (Istituto Dalle Molle di Studi sull'Intelligenza Artificiale)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个基于线性规划的框架，用最小的查询本身蕴含的结构约束来对任意（包括嵌套）反事实查询进行部分识别，并给出严格的上下界。

**💡 创新点**

创新点在于：①仅利用查询自身产生的部分拓扑顺序，无需完整因果图；②将识别问题转换为对响应签名（response signature）的线性组合，从而得到可解的线性规划；③通过等价类归约（lifted LP）将变量规模降至与观测变量样本空间成线性关系；④证明得到的上下界是紧致的，并构造满足这些界限的SCM。

**🔧 技术方法**

技术包括：结构因果模型（SCM）与偏结构因果模型（PSCM）理论；响应签名与等价类划分；线性规划（LP）与Charnes–Cooper变换（用于条件查询）；多重约束处理（实验数据、单调性、弱外生性）和求解器PuLP/COIN‑OR。

**📊 数据集**

使用了以下数据集进行实证：
- 经典二元反事实示例（PNS、PNS、PN、PS）
- 随机对照试验（Coronary Primary Prevention Trial）用于ACE、CDE
- UC Berkeley录取数据（性别、录取、申请院系）用于NDE
- 其他公开案例（如示例图中的IV、调节模型）进行案例验证。

**📈 对比分析**

方法与文献中的解析公式（Tian & Pearl 2000、Pearl 2009等）在相同假设下得到完全一致的上下界；在缺乏完整图假设时，得到更宽但仍紧致的区间。实验显示，使用lifted LP后，变量数从指数级（O(n^{n^k})）降到多项式级（O(n^k)），即使在包含数千个响应状态的实例（如NDE在UC Berkeley数据上）也能在秒级完成求解。与手工推导或基于图的边界相比，计算效率更高，且不需要图的显式构造。

**⚠️ 局限性**

限制主要包括：
①区间相对较宽，缺乏完整因果图的条件独立性约束时信息量有限；
②仅适用于无环（递归）SCM；
③虽然对部分顺序的求解不需要枚举所有线性扩展，但在极端情况下仍可能需要多次求解；
④对极大规模多值变量的响应签名，虽然lifted后可降维，但在某些极端模型中仍可能面临变量爆炸。

---

## 170. Semantic Overlays: Mitigating Prompt Injection with Annotations Beyond Tokens and Steering Vectors

**arXiv ID:** 2608.23873 | [PDF](https://arxiv.org/pdf/2608.23873v1)

**作者:** Joshua Penman `[一作]` `[通讯]`, Joshua Penman

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

无法获取论文内容

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

## 171. Feedback That Backfires: Why Small Language Model Agents Repeat the Call They Just Watched Fail

**arXiv ID:** 2608.23651 | [PDF](https://arxiv.org/pdf/2608.23651v1)

**作者:** Esmail Gumaan `[一作]` `[通讯]` (University of Passau), Esmail Gumaan (University of Passau)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在代理框架中将失败调用及其错误信息追加到对话记录时，模型更倾向于再次重复该调用的现象，并提出并评估了两种低成本的干预方式来消除这一“执行反馈反转”。

**💡 创新点**

创新点在于将执行反馈拆分为表面形式和语义两项，证明后者主导重复行为，并首次在两种截然不同的环境中验证该结论；随后提出抽象描述失败和解码时禁用失败字符串两种干预，显著降低重复率。

**🔧 技术方法**

使用了对数概率差（corrective gain）及其分解、归一化重复概率、精确贪心重复率等量化指标，并在多模型（0.5B–7B）上进行教师强迫评分与对照实验。

**📊 数据集**

构造了 ToolShed（12 个工具的模拟工作空间）和 CodeRepair（MBPP 代码修复）两大环境，手工生成固定的失败调用及对应错误信息，用以保证所有模型在相同的测试项上进行比较。

**📈 对比分析**

通过比较不同 harness（verbatim、instruction、abstract、ban）对 corrective gain、重复率以及任务成功率的影响，发现 abstract 与 ban 两种干预能将重复率降至接近零且不显著降低成功率，验证了干预方案的有效性。

**⚠️ 局限性**

主要局限包括仅在小规模模型和两种受限环境下测试；失败调用为人工构造；使用贪婪解码；缺乏大模型、真实长期任务以及随机采样等实验的验证。

---

## 172. When May an Agent Stop? Evidence-Carrying Termination for Tool-Using LLMs

**arXiv ID:** 2608.23623 | [PDF](https://arxiv.org/pdf/2608.23623v1)

**作者:** Jason Liu `[一作]` (University of California San Diego), Jason Liu `[通讯]` (University of California San Diego)

**通讯引用:** 91558 | [OpenAlex ID](https://openalex.org/A5032583158)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了证据携带终止（ECT）机制，要求智能体在停止前提供完整、可追溯的证书，证明所有必需答案槽都有合格证据并可被确定性重放。

**💡 创新点**

创新点在于将任务合同、证据账本与闭合变换语言结合，构建可确定性验证的终止门控；实现了类型化证书、哈希绑定、闭合重放，首次在终止决策上实现可验证的证书体系。

**🔧 技术方法**

使用的技术包括：类型化证书、可信任务合同与证据账本、哈希绑定、闭合变换语言、确定性重放、适配器化日志捕获、LLM 终止批评器以及多重检查器。

**📊 数据集**

采用合成任务集48个任务，涵盖lookup、aggregation、top‑k、temporal comparison、hierarchy join、missing‑data abstention六类工具族，并在每个任务下产生8种故障；闭环验证使用576条轨迹，覆盖22个任务簇。

**📈 对比分析**

比较方法：与六种基线（终止标记、内部自检、启发式、检查核心、全轨迹LLM批评器、后置条件参考）进行静态和动态闭环实验；在静态实验中ECT取得0/288 unsafe completion vs 252/288 for critic core；在动态闭环实验中，ECT与控制器对比不劣，恢复成功率更高，预期失败率显著降低。

**⚠️ 局限性**

限制：验证仅基于声明的合同与可信适配器前提，无法保证外部真值、安全或对齐；故障设计为人工干预，未覆盖未知错误；仅在单一模型与工具环境下测试，泛化性待进一步验证。

---

## 173. Measuring Digital Labour Market Transitions with a Digital Semantic Score: An AI-Based Methodology Applied to the Dutch Labour Market

**arXiv ID:** 2608.24222 | [PDF](https://arxiv.org/pdf/2608.24222v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 174. StabQ: Quantum Program Analysis via Weighted Stabilizer Representations

**arXiv ID:** 2608.24144 | [PDF](https://arxiv.org/pdf/2608.24144v1)

**作者:** Shangzhou Xia `[一作]` (Kyushu University), Jianjun Zhao `[通讯]` (Kyushu University)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了StabQ，一种基于稳定子表示的量子程序符号执行框架，能够在存在非Clifford门的情况下精确跟踪量子态演化，并生成可复用的“Tableau Chain”中间模型；

**💡 创新点**

核心创新包括：1）通过符号化的加权稳定子表（weighted stabilizer tableaux）实现非Clifford门的精确展开；2）引入Pauli分解机制将非Clifford门映射为加权Pauli组合；3）设计全局相位恢复与表凝聚（tableau consolidation）技术控制符号状态膨胀；4）在同一中间模型上支持多种分析任务（态重构、纠缠度量、支持集、Clifford属性检测）。

**🔧 技术方法**

使用的技术主要包括：稳定子表述（stabilizer tableau）及其符号化变体；Pauli基分解；符号执行与状态传播算法；全局相位恢复与表凝聚；基于Tableau Chain的支持集求解、相位重构、纠缠纯度计算等。

**📊 数据集**

实验评估基准覆盖三大套：Algorithms、MQT Bench、QASMBench，包含2–14 qubit、2–689 gate的多样化电路（如Grover、QAOA、QFT、GHZ、VQE等）。

**📈 对比分析**

与精确状态向量模拟（qiskit.statevector）和密度矩阵分解（Schmidt decomposition）对比，重构态与纠缠度量完全一致；构造时间与内存随量子位数、门数增长，但保持在可接受范围；在相同中间模型上多任务分析的开销远低于重新执行或单独重构。

**⚠️ 局限性**

局限性包括：1）非Clifford门导致的表状态指数膨胀仍难以完全消除；2）表凝聚与全局相位恢复在某些电路中效率不均；3）对高量子位或高度非Clifford化电路的扩展仍受限；4）缺乏针对更复杂分析（如错误校正、复杂资源估计）的专门支持。

---

## 175. Memory-Sovereign Inference: Output-Exact Execution Beyond Full Residency

**arXiv ID:** 2608.23805 | [PDF](https://arxiv.org/pdf/2608.23805v1)

**作者:** Lukas Stepanek `[一作]` `[通讯]`, Lukas Stepanek

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了“内存主权推理”体系结构和可验证的执行证书，用于证明在存储后端推理中，完整输出可以在超出完整驻留阈值的情况下保持精确。

**💡 创新点**

创新点包括：① 将典型的张量存储、物理驻留、稳定执行地址与调度四个关注点分离；② 设计了跨适配器共享核心计划与可验证的生命周期协议；③ 构建了可交叉验证的执行证书，记录资源权威、等价关系、重用不变式与执行结果，实现了对存储后端推理的可证实、可重复的证明。

**🔧 技术方法**

使用技术包括：cgroup v2 完整进程树内存计费；全板 GPU 监控与物理容量上限；固定大小 O_DIRECT 窗口与边界窗口；LRU64 与多窗口异步加载；生成器与路由器的原生内核；确定性故障注入与命名的 fail‑closed 转移；以及基于 JSON SHA‑256 的结果完整性验证。

**📊 数据集**

主要使用的数据集与模型为：Qwen3‑Next 80B‑A3B‑Instruct 量化版（Q4_K_M），在 32 768 token 机械 prompt 上生成 64 token；对 Gemma 4 QAT Q4_0 进行 512 + 8 规模的零缓存对照；另外还有 Qwen3‑6 与 Qwen3‑Next 的匹配源实验、Gemma 传输实验等。

**📈 对比分析**

比较方法：采用同一二进制、同一模型、同一硬件与同一资源契约下的零缓存基准；通过 E_64‑out 视图对 64 token 的 logits、token、响应、路由、消费者与目标身份进行逐字节比对；对比阻塞单窗口读取（D）与八窗口异步读取（F）的完整壁时比例（≈32 %）；利用资源计量（cgroup 负载、GPU 计费、物理占用）验证非驻留；在故障注入实验中，检查 14 个预定义的故障单元是否 fail‑closed。性能方面：在 Qwen3‑Next 上实现了 45 B 字节的完整表示，超过主机+GPU 容量约 11 B；异步组件的壁时比 32 %；但与驻留实现相比，整体推理时间明显较慢。

**⚠️ 局限性**

局限性：① 只在单一身份（Qwen3‑Next 80B‑A3B‑Instruct）与单一硬件（RTX 3090）上验证，缺乏跨模型、跨硬件的通用性；② 仅验证了资源计量与输出精度，未证明竞争条件、并发安全或可靠性；③ GPU 计费为审计型而非硬分区，可能隐藏隐藏资源；④ 故障注入实验基于后续构建，未覆盖所有可能的故障；⑤ 采用的量化模型与 prompt 受限，未体现更大规模或更复杂工作负载的效果。

---

## 176. Designing Caterpillars for Graphs: Approximation and Hardness

**arXiv ID:** 2608.24510 | [PDF](https://arxiv.org/pdf/2608.24510v1)

**作者:** Leon Kullmann `[一作]` (Technische Universität Berlin), Stefan Schmid `[通讯]` (Technische Universität Berlin)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并研究了在度受限的猫形树（caterpillar）上设计最小线性排列问题，给出了近似算法和多项式时间可判定的下界。

**💡 创新点**

创新点在于：①将任意 α 近似解升维为度‑Δ 猫形树的 (α+3-2/(Δ-1)) 近似；②证明该问题在常数 Δ 下对树仍为 NP‑难，与传统 MLA 成为鲜明对比。

**🔧 技术方法**

主要技术包括：将线性排列拆分为星形段并沿主干连接；利用距离拉伸分析得到成本上界；以及从最大割和 3‑Partition 构造的多项式时间归约来证明 NP‑难。

**📊 数据集**

本研究为理论性工作，未使用实际数据集；所有结果均基于图论构造与计算复杂性证明。

**📈 对比分析**

与已知的 MLA 近似算法对比，得到相同阶的 O(√log n·loglog n) 近似；在树上实现了 4‑近似；NP‑难性证明表明无法在多项式时间内获得多项式更优解。

**⚠️ 局限性**

局限性：近似算法存在常数项加成，且在 Δ 可变时复杂度未解；对未给定 Δ 的情况、其他目标度量（如带宽、切割宽度）的推广仍是开放问题。

---

## 177. TrAct: Bridging Robot Control and Visual Prediction with Visual Tracks

**arXiv ID:** 2608.24101 | [PDF](https://arxiv.org/pdf/2608.24101v1)

**作者:** Zhi Cao `[一作]` (University of Michigan), Huang Huang `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 TrAct 框架，通过视觉轨迹与动作的联合预测、轨迹条件视频生成和视觉语言奖励评估，实现了机器人决策与执行的高效跨机器人、跨视角通用性。

**💡 创新点**

将二维点轨迹作为 embodiment‑agnostic 接口，替代传统动作条件；构建 VLAT 同时输出动作与轨迹；利用 ControlNet 将轨迹作为 Stable Diffusion 的条件；通过 VLM 奖励模型对轨迹驱动的视频进行评分，完成基于轨迹的奖励驱动动作选择。

**🔧 技术方法**

Vision‑Language‑Action‑and‑Track (VLAT) 流匹配；Track‑Conditioned World Model (TWM) 基于 Stable Video Diffusion + ControlNet；Vision‑Language Reward Model (VLAC) 采用 InternVL2；动作条件基线 (AWM)；温度缩放重采样；大规模预训练与微调流程。

**📊 数据集**

大规模真实机器人数据 DROID（76K）与 EgoDex（150K）预训练；LIBERO‑PRO、LIBERO‑Plus、LIBERO‑INTEGRAL 评估集；UR5 与 Franka Panda 真实机器人实验集（400 条演示）。

**📈 对比分析**

在 LIBERO‑INTEGRAL 与真实 Franka Panda 任务中与 π0.5、VLAT、VLAT+AWM 等基线对比；TWM 在 PSNR、SSIM、LPIPS、FID、FVD 等视频质量指标上均优于 AWM；在模拟任务中成功率从 27% 提升至 55%；在真实任务中从 49% 提升至 76%，相较基线提升 20–30% 绝对成功率。

**⚠️ 局限性**

对 2D 轨迹的依赖限制了对复杂 3D 交互或稀疏轨迹点的处理；轨迹预测误差会直接影响世界模型输出；扩散模型推理成本高；实验聚焦于抓取/操纵任务，缺乏跨域长周期任务验证。

---

## 178. Low-Rank Velocity Fields as a Structural Prior for Unsupervised 4D Medical Image Interpolation

**arXiv ID:** 2608.24025 | [PDF](https://arxiv.org/pdf/2608.24025v1)

**作者:** Haojin Li `[一作]` (Southern University of Science and Technology), Jiang Liu `[通讯]` (Southern University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

设计端点无监督4D医学图像插值框架，利用低秩速度场作为结构先验，提升解剖一致性与时间连续性。

**💡 创新点**

通过 Tucker 低秩分解将速度场限制在共享空间基与样本特定核心，结合多尺度粗细层次的运动建模，首次实现端点无监督下的平滑且解剖可解释的中间帧合成。

**🔧 技术方法**

使用 Tucker 低秩参数化、固定低频 DCT 基、可学习混合矩阵、多尺度粗细级联、指数映射（Exp）对速度场积分、端点重建损失（NCC+Charbonnier）以及光滑正则化。

**📊 数据集**

在 ACDC 心脏 MRI（终末舒张/收缩两帧）和 4D-Lung CBCT（呼吸运动两帧）数据集上进行实验验证。

**📈 对比分析**

与经典无监督变形网络、近期无监督插值网络及有监督方法对比，本文在结构指标 NMI、SSIM 上领跑，同时重建误差保持与有监督方法相当，整体性能达到或超过当前最优方案。

**⚠️ 局限性**

低秩假设可能限制对极细节或非线性运动的捕捉；多尺度配置需手工设定秩与权重，且端点两帧的稀疏性仍可能导致部分运动信息缺失。

---

## 179. WebMCP-Phalanx: Enforcing and Characterizing Trust Boundaries for Browser-Integrated LLM Agents

**arXiv ID:** 2608.24017 | [PDF](https://arxiv.org/pdf/2608.24017v1)

**作者:** Lin-Fa Lee `[一作]` (National Yang Ming Chiao Tung University), Kuo-Hui Yeh `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并验证了一种双层 WebMCP-Phalanx 体系，结合浏览器原生能力凭证与多 LLM 异构检查，防止 LLM 代理在网页中被工具劫持、生命周期失控和语义注入攻击。

**💡 创新点**

首次在浏览器层引入不可伪造的工具所有权凭证与 Provenance 标签，再配合 Quarantine‑LLM 与 Privileged‑LLM 的权限隔离，实现对 SOP 缺陷的结构性与语义层面双重防御。

**🔧 技术方法**

利用浏览器原生加密凭证、Telemetry 与生命周期 Oracle、信任标签合成引擎，以及多 LLM 异构运行时的隔离与语义审计。

**📊 数据集**

在真实浏览器环境下基准 80 条描述注入、80 条返回注入攻击样本，结合 80 次工具注册与调用实验，评估系统鲁棒性。

**📈 对比分析**

与无防御基线和单层标签对照，实验显示工具撤销/覆写成功率从 100% 降至 0%，所有 80 条提示注入被完全拦截，仅剩 2/80 与工具命名时序相关，且任务完成率与无攻击基线无显著差异。

**⚠️ 局限性**

受同源脚本共用凭证的限制，仍无法完全阻止同源劫持；此外仅实现了检测与拦截而非完整沙箱，白盒自适应攻击仍可通过名称注入绕过；实现基于 polyfill，未覆盖原生加密凭证与 API 不可篡改。

---

## 180. Compression Trinity: Exploring Sparsity, Quantization, and Low-Rank Approximations for LLM Compression

**arXiv ID:** 2608.24070 | [PDF](https://arxiv.org/pdf/2608.24070v1)

**作者:** Mohammad Mozaffari `[一作]` `[通讯]`, Mohammad Mozaffari

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了一种统一的压缩三元组框架，在大型语言模型的预训练与后训练中同时应用稀疏化、量化和低秩近似，实现显著加速和更高压缩率。

**💡 创新点**

创新点在于首次将三种压缩技术协同运用，并针对优化器和模型架构设计了MKOR、双稀疏后向传播以及动态混合稀疏比率等算法，突破单一技术的精度‑效率壁垒。

**🔧 技术方法**

使用了稀疏化（N:M稀疏）、量化（低位宽量化）和低秩近似（低秩适配器、块对角稀疏曲率近似）等技术，结合第二阶优化器MKOR与低秩“lazy”适配器。

**📊 数据集**

在大规模LLM预训练和推理任务上使用了公开语料库（如Common Crawl、Wikipedia等），并在标准评测数据集（GLUE、SQuAD等）上进行微调与评估。

**📈 对比分析**

与传统KFAC、单独稀疏化或量化方法相比，训练加速达1.85×、推理加速1.38×，在零训练静态掩码下精度提升3.97%，完整三元组方案精度提升5.66%，并在相同参数预算下超过未压缩模型0.6%。

**⚠️ 局限性**

局限性包括对硬件的实现依赖较高、超参数调优复杂、以及在极端稀疏或低精度环境下可能出现数值不稳定或模型容量不足的问题。

---

## 181. Scalable Question-Centric Text-to-Image Evaluation: Reliable Ranking, Fine-Grained Diagnosis, and Cost-Aware Routing

**arXiv ID:** 2608.24112 | [PDF](https://arxiv.org/pdf/2608.24112v1)

**作者:** Shaoan Zhao `[一作]` (Data Science & Artificial Intelligence Research Institute, China Unicom), Shiguo Lian `[通讯]` (China Unicom Group Co Ltd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 QC-T2I-Bench，基于问题中心的评估框架，将开放式提示拆解为带属性的原子问题，并使用层次约束聚合与 Davidsonian 场景图（DSG）实现精细诊断与模型排序。

**💡 创新点**

创新点在于把评估单元从提示转为问题，实现原子归因、依赖感知、复杂度加权、跨提示证据整合，并在不训练的前提下完成成本感知路由。

**🔧 技术方法**

技术包括问题构造与能力坐标化、DSG 结构化、层次约束问题聚合（HCQ）、文本渲染评估、统计检验与 Bootstrap 置信区间。

**📊 数据集**

使用包含 6,573 句子提示、94,547 条英文问题与 94,555 条中文问题的公开数据集，对 13 种开源 T2I 模型进行评估。

**📈 对比分析**

通过层次聚合、DSG 组件分析和 Bootstrap 排名对比，揭示模型在实体、属性、关系、语义等 21 维能力上的差异；路由实验表明 Q-Profile-C 与 ERNIE 等价，且 GPU‑s/MP 降低 21.3%。

**⚠️ 局限性**

局限性包括问题构造对自动化审核的依赖、文本内容评估仍采用编辑距离、以及对跨领域泛化和模型更新的适应性不足。

---

## 182. Balancing Evidence and Interpretation: Historical Grounding Ratio as a Design Parameter for AI-Generated Urban Storytelling

**arXiv ID:** 2608.24157 | [PDF](https://arxiv.org/pdf/2608.24157v1)

**作者:** Fuyang Zhang `[一作]` (Nanjing University), Maurice Benayoun `[通讯]` (Nanjing University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究如何在位置感知的生成式叙事系统中对来自历史档案与实时场景的多源信息进行分配，并提出“历史根植比率（HGR）”作为衡量生成文本中历史证据比例的可操作设计参数。

**💡 创新点**

创新点在于：①将历史根植比率定义为可测量、可调控的设计变量，弥补现有系统对多源证据比例缺乏透明度的缺陷；②通过实地实验揭示不同历史根植比例对用户体验的非单调影响，说明单纯提高历史信息量并不总能提升整体体验。

**🔧 技术方法**

技术手段包括：位置感知 + 视觉感知（摄像帧→场景描述），检索增强生成（RAG）策略，使用8B中文大语言模型进行文本生成，结合自定义指令实现对历史与现场信息的分配控制。

**📊 数据集**

使用的数据集包含：南京市161个历史空间多边形及其关联档案记录（约2000条检索候选），现场摄像帧的视觉描述，及18名参与者的GPS轨迹与实时环境数据。

**📈 对比分析**

方法：在同一段路上对每位参与者实施三种HGR配置（情境占优、平衡、证据占优），通过对比每种配置下的五项体验评分（地点关联、历史理解、场景融合、信息匹配、探索意愿）进行统计检验。结果显示：HGR提升显著增强地点关联感，但对其他四项体验的提升不随HGR线性增加，而平衡配置在整体体验上更佳。统计检验（Friedman、Wilcoxon、等价检验）均支持此结论。

**⚠️ 局限性**

局限性包括：①仅在三段固定路段内评估，未检验连续动态调节HGR的可行性；②样本规模相对较小，缺乏对不同人群的广泛验证；③实验只关注叙事文本输出，未考察更复杂的交互方式（如问答、导航）对HGR的影响；④HGR在本研究被限制为三离散水平，未探索其在连续空间中的调节效果。

---

## 183. Guillotine and Tiling Cofiniteness in Unary Picture Languages

**arXiv ID:** 2608.24260 | [PDF](https://arxiv.org/pdf/2608.24260v1)

**作者:** Pierluigi San Pietro `[一作]` (Dipartimento di Elettronica, Informazione e Bioingegneria), Antonio Restivo `[通讯]` (Università di Palermo)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在一字母图像语言中，利用矩形单元（单字图块）通过水平垂直拼接或任意拼贴生成所有足够大的矩形图像的可行性问题。

**💡 创新点**

提出并证明了一个完全等价的算术判据：当且仅当所有图块高度的最大公约数为1且对于每个出现的宽度质因子，其对应的高度集合的最大公约数亦为1时，生成的集合在二维中是渐进完全的。

**🔧 技术方法**

采用了数论中的最大公约数与最小公倍数技术、根号单位的根号证明法、以及 Klarner 系统的有限基理论，将一维的数论判据推广到二维。

**📊 数据集**

本文不涉及实验数据集，研究完全基于理论分析与证明。

**📈 对比分析**

该工作不涉及算法实现或性能评估，仅给出理论上的必要与充分条件；因此不存在可比性能指标。

**⚠️ 局限性**

局限性在于仅处理一字母图像语言，扩展到多字母或更一般的图像语言仍是开放问题；此外，对无限图块集合的判定在有效性方面尚未得到完整解决。

---

## 184. Counterfactual Explanations and the Scope of Contestability

**arXiv ID:** 2608.24562 | [PDF](https://arxiv.org/pdf/2608.24562v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 185. Across the Loss Landscape with Progressive Growth

**arXiv ID:** 2608.24568 | [PDF](https://arxiv.org/pdf/2608.24568v1)

**作者:** Paul Caillon `[一作]` (Université Paris Dauphine-PSL), Alexandre Allauzen `[通讯]` (Université Paris Dauphine-PSL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文研究了渐进式网络成长（逐步解锁参数）对训练过程几何特性的影响，证明这种做法会倾向于让优化到达在冻结方向上更平坦的最小值，并通过理论、仿真和在ResNet/CIFAR‑100上的实验验证这一偏差。

**💡 创新点**

创新点在于：
1) 将成长视为逐步放宽约束的“受限优化”序列，给出了基于Schur补的“有效曲率”理论，阐释了为何平坦最小值在冻结约束下更易访问；
2) 提出了一系列跨阶段的过渡度量（interpolation barrier、retention、leakage、冻结方向最大特征值比）来评估成长过程的几何稳定性；
3) 通过内部对照（固定子空间训练、一次性解冻）清晰区分了“渐进放松”与“延迟容量释放”的差异。

**🔧 技术方法**

主要技术包括：
- 随机正交子空间选择与增量解冻；
- 本地二次近似与Hessian Schur补分析；
- 通过交叉验证的插值障碍、留存率和泄漏度量评估过渡；
- 计算子空间上Hessian最大特征值比的近似估计。

**📊 数据集**

实验数据集：
- 受控二次基准（维度100，已知曲率矩阵）；
- 真实深度学习任务：ResNet-18 训练于CIFAR‑100。

**📈 对比分析**

比较方法：
- 与全模型训练（无成长）做对比；
- 对比固定子空间训练和一次性解冻；
- 通过不同成长触发方式（验证精度、训练损失）和阶段数（S=3,5,10）进行细粒度实验。
 结果显示：
  - 成长不会提升最终测试准确率，甚至在阶段过多时会略低；
  - 但在几何指标上表现优异：低插值障碍、高保留率、新解冻子空间的最大特征值比显著低于已训练子空间；
  - 一次性解冻可恢复大部分准确率，却缺乏低曲率特征；
  - 固定子空间训练几乎失败。

**⚠️ 局限性**

局限性：
- 理论仅局限于局部非退化最小值，无法直接处理现代网络中的对称/近似平坦轨道；
- 仅在单一架构与单一数据集上验证，缺乏普适性评估；
- 曲率估计为局部近似，未能完全表征全局盆地结构；
- 成长策略未针对效率或最终性能优化，实际训练成本与收益不确定；
- 仍需探索能否将过渡度量转化为在线自适应成长规则。

---

## 186. DDMS: Discriminative Distillation of Multi-view Foundational Features into Single-view Models

**arXiv ID:** 2608.23850 | [PDF](https://arxiv.org/pdf/2608.23850v1)

**作者:** Jeong-gi Kwak `[一作]` (University of British Columbia), Kwang Moo Yi `[通讯]` (University of British Columbia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种 3D‑Aware 视觉特征蒸馏框架 DDMS，通过构建多视角教师并蒸馏给单视角学生，提升跨视角一致性与局部区分度，同时保留原始 2D 基础模型的语义表达。

**💡 创新点**

创新点包括：① 将预训练的 2D 基础特征与多视角 Transformer 的几何上下文进行残差级融合；② 通过几何监督的排名损失与语义锚定双重约束，在教师阶段实现 3D 一致性与细粒度可分离；③ 将上述多视角教师的知识蒸馏到单视角学生，使其在推理时仅需单张图像即可获得多视角感知。

**🔧 技术方法**

关键技术包括：多视角 Transformer（如 3DGS 或类似模型）提取几何特征；轻量级投影与残差融合模块；基于 3D 距离的可区分排名损失；语义锚定约束（保持与原 2D 特征空间的相似度）；以及特征级蒸馏与弱正则化。

**📊 数据集**

实验使用的主要数据集有：ScanNet、ScanNet++、NYUv2（用于深度和语义迁移）；Navi‑Wild、SPair‑71k、DAVIS‑2017（用于几何与语义对应评估）；Mip‑NeRF 360（用于 3D Gaussian 上的渲染评估）。

**📈 对比分析**

与冻结的 DINOv2、Fit3D、MEF、SnD 等基线相比，DDMS 在多视角对应、特征区分、语义分割、深度估计以及 3D 点云/Gaussian 渲染等评估中均表现出更高的准确率或更小的误差，兼顾了跨视角一致性与语义迁移性能。

**⚠️ 局限性**

局限性：① 需要预训练的多视角几何模型作为教师，增加额外的训练步骤和算力；② 蒸馏过程中仍依赖多视角输入，单视角迁移后对极端视角变化的鲁棒性可能有限；③ 对不同基线模型的泛化性和在资源受限场景下的部署成本尚未充分验证。

---

## 187. Automated Synthesis of Cloud Emulators

**arXiv ID:** 2608.23842 | [PDF](https://arxiv.org/pdf/2608.23842v1)

**作者:** Archit Bhatnagar `[一作]` (University of Michigan), Ang Chen `[通讯]` (University of Michigan)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种自动化的神经-符号方法，用文档生成云模拟器，并通过与真实云对齐进一步提升准确性。

**💡 创新点**

创新点在于将大型语言模型与符号结构（状态机抽象、资源索引）相结合，既抑制模型幻觉，又保证规模化生成；并实现基于执行轨迹的自动对齐与修复。

**🔧 技术方法**

采用 GPT 等 LLM 进行代码合成；符号化的状态机框架与资源索引进行约束；执行轨迹收集、差异定位与 LLM/自动化修复；覆盖率驱动的测试生成。

**📊 数据集**

使用 AWS 与 GCP 官方文档（约 90 种资源，200+ API），以及 50 个 CLI 与 50 个 Terraform 脚本（共 330 例）作为评测案例。

**📈 对比分析**

与主流模拟器 LocalStack 对比，覆盖率从 70% 提升至 100%，在错误类、响应一致性和状态一致性三级指标上均优于 LocalStack；合成时间约 1.3 小时，自动对齐时间约 6 小时，整体性能可接受。

**⚠️ 局限性**

局限性包括：仅模拟 API 行为，未覆盖性能、延迟和并发等特性；依赖外部云作为对齐 oracle，成本与时延高；对多云交互的跨资源依赖建模仍需改进。

---

## 188. Scaling Reinforcement Learning for Diffusion Models via Velocity Matching

**arXiv ID:** 2608.23664 | [PDF](https://arxiv.org/pdf/2608.23664v1)

**作者:** Jaemoo Choi `[一作]` (Georgia Institute of Technology), Yongxin Chen `[通讯]` (Georgia Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于奖励的速度匹配（RVM）方法，对扩散模型进行奖励微调，完全不需要轨迹或似然估计。

**💡 创新点**

创新点在于把奖励直接映射到速度场的回归目标，形成一种轨迹无关、只需单一步噪声的训练框架，并证明其可归纳为之前的 RAM 与 DiffusionNFT，显著简化算法。

**🔧 技术方法**

技术方法包括：奖励加权的速度回归损失、可选的锚定速度项、动态跟踪奖励（基于光流）以及 LoRA 微调，全部在扩散模型的原始训练表示上实现。

**📊 数据集**

使用的数据集与模型包括：Stable Diffusion 3.5（文本到图像）、Wan2.1-T2V-1.3B（文本到视频）、SkyReels-I2V（图像到视频），评估指标为 VBench、VideoAlign、HPSv3 等。

**📈 对比分析**

与基线（原始模型、CFG、FlowGRPO、DanceGRPO、TaRoS、DiffusionNFT、RAM）对比，RVM 在视频任务上获得更高的 VBench 总分与动态度数，同时训练成本约为轨迹方法的一半或更低；在文本到图像任务上也与 ELBO/PEPG 方法相当。

**⚠️ 局限性**

局限性：需要精心设计奖励与锚点，过度依赖奖励信号的质量；目前仅针对扩散模型，未直接处理自回归视频生成，且对极端少步采样的稳健性仍有改进空间。

---

## 189. Absorbing Gradient Conflicts: Modeling Semantic Variance via Kent Distributions for Cross-Modal Hashing

**arXiv ID:** 2608.24010 | [PDF](https://arxiv.org/pdf/2608.24010v1)

**作者:** Hengjie Zhu `[一作]` (Chinese Academy of Sciences), Weiping Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Kent‑based Distributional Proxy Hashing（KDPH）框架，将多标签交叉模态哈希中的代理从传统点转为在超球面上的可学习 Kent 分布，解决梯度冲突与代理崩塌问题。

**💡 创新点**

创新点：① 用 anisotropic Kent 分布表示代理，允许代理在保持中心不变的前提下通过方向方差动态吸收梯度冲突；② 通过 Cayley 变换实现代理正交参数化，保证训练稳定；③ 设计分布式 triplet 损失与多模态正则化，兼顾检索精度与代码熵。

**🔧 技术方法**

技术手段：CLIP 双流特征提取；Kent 分布代理与 Cayley 变换正交化；分布式 proxy triplet 损失；InfoNCE 对齐；多模态不相关正则化；均匀分布约束（OT）；Adam 优化。

**📊 数据集**

使用 MIRFLICKR‑25K、NUS‑WIDE、MS COCO 三大多标签交叉模态数据集。

**📈 对比分析**

与 9 个先进基线（如 DCPH、nivMF、DCMHT、MIAN、DSPH、DNPH、DNpH、DDBH、DAGtH）在 16/32/64 位码长下做 mAP 对比，KDPH 在所有设置下均优于最佳基线，提升幅度可达 6.07% 以上。

**⚠️ 局限性**

局限性：在单标签场景下提升有限；Kent 分布引入额外参数，训练时需使用 Cayley 变换与正则化，导致计算复杂度略增。

---

## 190. Transition Systems from Causal Reversible Bundle Event Structures

**arXiv ID:** 2608.24251 | [PDF](https://arxiv.org/pdf/2608.24251v1)

**作者:** Nataliya Gribovskaya `[一作]` (A.P. Ershov Institute of Informatics Systems), Irina Virbitskaite `[通讯]` (A.P. Ershov Institute of Informatics Systems)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究可逆束事件结构（RBES），提出两种转移系统语义——基于配置与基于残差，并证明在因果可逆RBES下这两种语义同构；同时给出残差求解器及其正确性、范畴论的映射与函子化；

**💡 创新点**

创新点在于：①首次证明因果可逆RBES的配置转移系统与残差转移系统之间存在同构而非仅仅是双边似乎；②设计了基于不可执行事件的残差求解器并给出其分类学性质；③把RBES到LTS的映射形式化为函子，满足范畴论一致性原理；

**🔧 技术方法**

主要技术包括事件结构模型、可逆性约束（因果一致性）、残差（removal）操作、标签多重集（step semantics）、同构与函子构造、范畴论映射（LTS-morphism、RBES-morphism）。

**📊 数据集**

本文为理论研究，无实验数据集。

**📈 对比分析**

由于是形式化证明，未给出实验比较；同构证明表明两种语义在结构上完全等价，因而在理论上可互换。

**⚠️ 局限性**

限制：残差求解器仅能删除已执行事件，无法处理非因果可逆事件结构；对非可逆或更一般事件结构的扩展需更复杂的残差操作。

---

## 191. On existential Büchi arithmetic in two coprime bases

**arXiv ID:** 2608.24410 | [PDF](https://arxiv.org/pdf/2608.24410v1)

**作者:** Joris Nieuwveld `[一作]` `[通讯]` (University of Oxford), Joris Nieuwveld (University of Oxford)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明了在两基数 α、β 为互质的情况下，包含 Büchi 预测子 Vα、Vβ 的 Presburger 结构的存在量子片段 ∃FO(<,+,Vα,Vβ) 是可判定的。

**💡 创新点**

创新点在于：①将之前仅适用于幂预测子 α^、β^ 的可判定性结果扩展到更强的 Büchi 预测子；②利用共价（coprime）条件在 Baker 定理与 Kronecker 定理的组合下实现了对变模约束的有效化简；③通过引入“层次（levels）”与“底部族（bottom family）”的概念，在量化消去过程中统一处理指数与线性指数变量。

**🔧 技术方法**

主要技术：量化消去策略、线性规划与极限论（凸多面体、薄多面体判定）、Baker 的指数算术不等式、Kronecker 的 Diophantine 逼近定理、模约束分解与 Chinese Remainder、以及对多变量指数方程的模块化处理。

**📊 数据集**

无实验数据集，本文为纯理论证明工作，无需使用实际数据。

**📈 对比分析**

与之前基于幂预测子的可判定性结果相比，本工作在存在量子片段上保持可判定性，但在更强的 Büchi 预测子下实现；并通过严格的理论分析证明了判定算法的有效性，未给出计算复杂度或运行时间等性能指标。

**⚠️ 局限性**

限制：仅对互质的 α、β 有效；无法覆盖一般的乘法独立但不互质的基数；不适用于三及以上预测子或更一般的正则集合预测子；对于非互质基数的情况仍需新的方法（可能涉及 p‑进逼近或更强的算术猜想）。

---

## 192. CARO: Contact-Agnostic Residual Observation for Zero-Shot Robust Quadruped Locomotion

**arXiv ID:** 2608.24217 | [PDF](https://arxiv.org/pdf/2608.24217v1)

**作者:** Zihan Yang `[一作]` (Beihang University), Xiang Yu `[通讯]` (Beihang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了CARO框架，在强化学习控制循环中嵌入固定基欧拉-拉格朗日内部模型，并通过观测残差作为适应信号，使四足机器人在未见动态和接触条件下实现零射击鲁棒行走。

**💡 创新点**

创新点在于：①采用接触无关的残差观测，只依赖关节位置、速度和命令扭矩，省去接触力、浮基速度或视觉传感器；②将内部模型误差作为结构化反馈直接喂给策略，而非传统的直接补偿；③在相同的训练设置下实现了显著提升的零射击泛化性能。

**🔧 技术方法**

使用的技术包括：强化学习（PPO）、内部模型观测（固定基欧拉-拉格朗日残差）、误差观测器（基于动量的观测器）、域随机化、以及在模拟环境legged_gym与真实DeepRobotics Lite3机器人上的部署。

**📊 数据集**

数据集主要为在仿真中的多种地形（平地、斜坡、粗糙斜坡、碎石、波浪）和负载比例（1.0~3.0倍基准质量）以及相应的随机化参数；真实测试使用Lite3机器人在四种未见地形、不同侧向负载和高平台落地情景。

**📈 对比分析**

与基线方法Vanilla、RMA和RL2AC比较，CARO在25种负载-地形组合中的平均成功率最高（88.6%），在3.0×负载时仍达64.5%成功率，且在跟踪误差、转向稳定性和突发负载变化的瞬态响应上表现最佳；实验显示其零射击性能显著优于传统自适应和无自适应方法。

**⚠️ 局限性**

局限性包括：①残差观测误差的有界性仅在采样增量受限时成立，未证明闭环稳定性；②观测器增益平衡响应速度与噪声放大；③观测器输出的残差为整体量，无法提取具体接触力或位置；④在极高动态运动或大浮基加速度下，固定基近似误差可能主导，影响适应效果。

---

## 193. Topology optimization of force densities for form finding of cable structures

**arXiv ID:** 2608.24177 | [PDF](https://arxiv.org/pdf/2608.24177v1)

**作者:** Nicolò Pollini `[一作]` `[通讯]` (Technion Israel Institute of Technology), Nicolò Pollini (Technion Israel Institute of Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于力密度法（FDM）的拓扑优化框架，用于在给定材料体积约束下寻找轻量化绳索结构的几何形状和连通性。

**💡 创新点**

创新点在于：①将传统的离散力密度方法与基于连续密度变量的SIMP插值相结合，实现力密度与结构拓扑的同步优化；②通过引入显式的二进制惩罚项，促使优化结果逼近离散解；③将非线性FDM以约束优化的形式重写，并将其与拓扑优化耦合，可在保持几何约束的前提下进一步简化结构。

**🔧 技术方法**

技术手段包括：力密度法（线性与非线性）、SIMP插值、二进制促进惩罚、混合二进制-连续非线性规划、自动微分（JuMP.jl）、内部点求解器MadNLP（以及IPOPT用于非线性FDM），并在Julia语言中实现。

**📊 数据集**

采用自定义的网格基准结构（11×11、21×21、31×31节点的二维/三维网格）作为实验数据集，分别设定不同边界条件、力密度分配和体积约束，用于验证方法的有效性与可扩展性。

**📈 对比分析**

与传统线性FDM或手工设定的网格进行对比，结果显示：①优化后的拓扑在保持接近参考形状的同时，显著降低了材料使用；②在不同体积约束下可触发从冗余到极简的结构转变；③求解时间随网格细化呈指数增长，但在标准桌面机器上，17,000+变量的最坏情况仅需约1分钟；④在大多数案例中，连续密度变量在优化结束后已逼近离散解，舍入误差仅为优化误差的十分之一。

**⚠️ 局限性**

局限性包括：①非凸的二进制-连续非线性问题对求解器初始值和参数敏感；②在更复杂的几何约束或更大规模结构时，求解时间可能显著增加；③当前仅针对拉伸性绳索网络，尚未扩展到含弯曲、膜或壳体等多功能结构；④虽然采用SIMP和二进制惩罚促使接近离散解，但仍需在后处理阶段进行阈值化，且在极细网格下不同长度成员的数量-体积不匹配可能导致设计偏差。

---

## 194. pigzpp: Fast, Parallel, Portable Compression for the Whole Stack

**arXiv ID:** 2608.24153 | [PDF](https://arxiv.org/pdf/2608.24153v1)

**作者:** Thamme Gowda `[一作]` `[通讯]`, Thamme Gowda

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将原版pigz从单线程、全局状态的可执行工具改写为线程安全、可嵌入的C++23库，并提供多语言绑定（Python、Go、Rust、WebAssembly）以及ZIP和PNG的应用层封装。

**💡 创新点**

核心创新在于：①把pigz的并行设计抽象为可重入、线程安全的DEFLATE核心；②结合SIMD优化的zlib-ng与x86-64 ISA‑L汇编实现的两条后端路径；③一次性提供多语言绑定和应用便利；④利用AI编码代理完成大规模重构与现代化。

**🔧 技术方法**

技术手段包括：C++23标准、std::thread、SIMD指令集（SSE/AVX2/AVX‑512、NEON）、Intel ISA‑L汇编、nanobind/Python绑定、Emscripten/WebAssembly、cgo/FFI、以及ZIP/PNG的标准结构包装。

**📊 数据集**

实验数据集主要有：128 MB多语言文本（英文+中文维基百科）、637 MB真实OCI镜像层、24张Kodak真彩色图像集、以及随机不可压缩数据。

**📈 对比分析**

采用基于七次计时样本的中位数对比方法，比较对象包括原始pigz、gzip、libdeflate、Python标准库、Go stdlib、Rust等。结果显示：pigzpp在标准gzip压缩下达10–20 %更高吞吐量，ISA‑L后端更快；Python绑定比官方zlib快约1.5倍；Go cgo绑定可达1.1 GB/s；整体压缩比保持与原gzip相同。

**⚠️ 局限性**

局限性：ISA‑L仅在x86‑64平台可用；所有基准在单一WSL2 x86‑64环境下完成，缺乏ARM64或其他架构的数据；主要使用128 MB多语言文本和单一压缩级别进行评测，结果对不同数据类型和压缩级别的泛化性有限；未针对CPU亲和、Turbo Boost或频率缩放做统一控制。

---

## 195. Graph-Supervised Hierarchical Clinical Alignment for Radiology Report Generation with Large Language Models

**arXiv ID:** 2608.24121 | [PDF](https://arxiv.org/pdf/2608.24121v1)

**作者:** Yingshu Li `[一作]` (University of Sydney), Luping Zhou `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种Graph‑Supervised Hierarchical Clinical Alignment框架，在训练时使用临床知识图将图像‑报告监督拆分为疾病级对齐与全局语义对齐，从而提升放射报告生成的临床准确性。

**💡 创新点**

创新点在于：①将知识图仅作为训练时的结构先验，用于定义疾病级监督单元；②构造疾病节点的跨模态查询与实例条件匹配，并加入软正样本正则化；③在全局层面实现实例条件语义匹配与疾病条件化正则化；④推理时完全去除知识图，保持模型轻量且无额外推理成本。

**🔧 技术方法**

采用 LLaMA3‑3B 作为语言生成器，Swin Transformer 作为视觉编码器，结合知识图注意力层、多头跨模态注意力、实例条件匹配、软正样本正则化、KL 正则化、对比学习和交叉熵等技术。

**📊 数据集**

在 MIMIC‑CXR、IU‑Xray、COV‑CTR 三大放射影像基准上进行实验；使用 CheXpert 14 病种构建知识图；同时在 IU‑Xray 上使用 CheXbert 伪标签进行验证。

**📈 对比分析**

与多种基线（R2Gen, METransformer, DCL, KARGEN, R2GenGPT, 等）在 BLEU、METEOR、ROUGE、RadGraph F1、BERTScore、RadCliQ、GREEN、RateScore 等指标进行对比。3B 模型在 MIMIC‑CXR 上已超越 7B/13B 规模的对齐方法，在 IU‑Xray 与 COV‑CTR 上亦获得最高或相近的分数，且在临床指标上实现领先。

**⚠️ 局限性**

局限性包括：①需要预先构建且标注的疾病知识图，对新疾病或低频疾病适应性有限；②对图结构敏感，随机或全连图会显著下降；③框架目前仅验证于胸部 X‑ray/CT，尚未证明对其他模态的泛化；④对极端稀有病症仍表现不足，需进一步改进。

---

## 196. Variance-Guided Spatial Attention Fusion for Robust End-to-End Driving under Asymmetric Sensor Degradation

**arXiv ID:** 2608.24366 | [PDF](https://arxiv.org/pdf/2608.24366v1)

**作者:** Weizhi Tao `[一作]` (Hong Kong Polytechnic University), Hailong Huang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种针对感知传感器不对称退化的端到端多模态驾驶方法 VG‑SAF，能够在摄像头或 LiDAR 局部受损时只抑制受损单元而保留其余信息。

**💡 创新点**

创新点包括：① 用物理驱动的增广器生成密集腐败掩模作为可靠性监督；② 通过跨分支稠密蒸馏将掩模映射为单像素方差，获得严谨的严重度‑可靠性映射；③ 将校准后的方差通过局部门控与跨模态信任软最大化的混合注意力实现空间与模态级的可靠性融合；④ 使用 Laplace 置信头提供全局轨迹不确定性报警。

**🔧 技术方法**

采用 heteroscedastic 置信头、跨分支稠密蒸馏、softplus 参数化、最大池化聚合、混合注意力（局部门控+信任 softmax）以及自回归 GRU 轨迹解码器。

**📊 数据集**

在 CARLA Longest6 基准上进行评估，使用多种摄像头和 LiDAR 退化模式（全局与局部遮挡、噪声、曝光、雨雾等）。

**📈 对比分析**

与 TransFuser、Equal‑Weight Fusion、Image‑only 等基线对比，VG‑SAF 在所有退化场景下的 Driving Score 提升约 10‑13 分，路程完成率和违章分数均显著改善。

**⚠️ 局限性**

局限性包括：① 方差头在完全信号丢失时失效；② 训练时使用的掩模分布对极端或未见退化的泛化有限；③ 未考虑多模态时序可靠性与真实车载验证，需进一步研究。

---

## 197. TRACE: An Evidence-Grounded Benchmark for Safety Evaluation of Large Reasoning Models

**arXiv ID:** 2608.24232 | [PDF](https://arxiv.org/pdf/2608.24232v1)

**作者:** Zhenyu Wu `[一作]` (King Abdullah University of Science and Technology), Xin Gao `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 TRACE，一个基于证据的安全评估基准，用于在大规模推理模型（LRM）生成的提示、推理轨迹和最终回复三个阶段评估安全性。

**💡 创新点**

创新点在于：① 提供跨提示、推理轨迹和最终回复的完整安全注释；② 对每个安全判断附上可验证的文本证据；③ 设计了针对推理轨迹的安全评估，填补现有基准的空白；④ 在多语言、多风险类别和多攻击策略上进行系统评测。

**🔧 技术方法**

技术手段包括：使用四个 LRM（安全对齐及其 abliteration 版本）生成样本；用三个强大 LRM 进行多模型安全标注与证据抽取；采用多数投票决定安全标签并验证证据为连续子串；评估时使用 F1、FNR、FPR、TokenF1 等指标；对 18 种 guardrail 模型进行统一实验。

**📊 数据集**

使用的数据集：S‑Eval（9 种风险类别，10 种攻击策略）与 WildChat（安全对话），共 1993 条提示，生成 5000 条（提示、轨迹、回复）三元组；所有样本均覆盖两种语言（英文、中文）。

**📈 对比分析**

比较方法：对 18 个 guardrail 模型在提示、推理轨迹和最终回复三阶段分别计算 F1、FNR、FPR，并对证据归属使用 TokenF1。结果显示，提示安全判断 F1 最高（≈88%），最终回复其次（≈86%），推理轨迹最难（≈84%）。证据归属 TokenF1 仅在 11–15% 左右，表明模型在提供解释方面仍弱。

**⚠️ 局限性**

局限性：① 仅覆盖英文和中文两种语言，缺乏多语言评估；② 标注过程中仍可能存在噪声，需更大规模人工审核；③ 仅评估了 18 种 guardrail 模型，未来需扩展更多模型和攻击场景。

---

## 198. StateTune: Transforming LLM-Assisted EDA Flow Tuning into a Stateful, Closed-Loop Process

**arXiv ID:** 2608.23601 | [PDF](https://arxiv.org/pdf/2608.23601v1)

**作者:** Kunlong Li `[一作]`, Lingli Wang `[通讯]` (Fudan University)

**通讯引用:** 7592 | [OpenAlex ID](https://openalex.org/A5101415626)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于大型语言模型（LLM）的闭环EDA流程参数调优框架 StateTune，利用持久化、结构化的记忆实现跨阶段、跨设计的知识共享与自适应调优。

**💡 创新点**

创新点：① 引入 typed、evidence-gated 的持久化记忆，自动过滤错误规则并持续积累失效模式、敏感性信息；② 将此记忆同时驱动候选配置生成与 EHVI‑guided、runtime‑aware 的推广决策，实现对全流程预算的精细分配；③ 在多保真度评估与 LLM 推理之间构建闭环，提升搜索效率与结果质量。

**🔧 技术方法**

使用技术：大型语言模型（Qwen3 + DeepSeek）与 Chain‑of‑Thought 推理、RAG‑EDA 文档检索、基于高斯过程与 kNN 的多保真度 QoR 与运行时预测、EHVI 量化与优先级排序、结构化记忆管理（硬规则、软启发、失败总结、敏感性统计）。

**📊 数据集**

数据集：Cadence Genus/Innovus 工业流程，覆盖两技术节点（ASAP7 7 nm、NanGate45 45 nm）和三类设计块（JPEG、AES、IBEX），共六个基准块，包含 19 个跨阶段参数。

**📈 对比分析**

对比方法：BO（qEHVI）、Optuna‑TPE、Random、RankTuner、CROP 等五个基线；在相同总预算（192 线程·h）下进行公平比较。结果显示 StateTune 在所有六块基准上均取得最佳 WNS、功耗及最终超体积，平均超体积提升约 30‑40%，且在全流程预算内显著加速收敛。

**⚠️ 局限性**

限制：实验仅覆盖六个相对小规模的工业基准，缺乏更大规模 SoC 或多设计族的验证；记忆结构采用平面式组织，面对数千条规则时可扩展性与查询效率需进一步研究。

---

## 199. Revenge of Monosemanticity: Specialized Neurons Improve Data Efficiency in MLPs

**arXiv ID:** 2608.24007 | [PDF](https://arxiv.org/pdf/2608.24007v1)

**作者:** Amirhesam Abedsoltan `[一作]` (University of California San Diego), Mikhail Belkin `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

研究多聚类回归问题中 MLP 的特征学习机制，发现神经元在训练后会出现单义化（专门化）现象，即每个神经元主要对某一聚类的预测方向负责。

**💡 创新点**

创新点在于揭示并证明 MLP 能通过单义化在没有显式路由模块的情况下同时学习聚类结构与局部低维特征，从而在预测方向全局没有低维结构时仍保持优秀的样本效率。

**🔧 技术方法**

使用的技术包括多层感知机（ReLU、GeLU、ReGLU、SwiGLU 激活）、小初始化的梯度流与 Frobenius 约束的 ERM、NTK 与卷积核分析、Recursive Feature Machine（RFM）与核岭回归对比，并配合理论证明（神经元专门化与样本复杂度分析）。

**📊 数据集**

实验数据为合成高斯混合模型，聚类数 K∈{1,2,10,50}，维度 d=20 或 40，聚类中心与预测方向分别沿不同坐标轴，链接函数随机选取二次 Hermite、sin、tanh 等非线性。

**📈 对比分析**

与基线比较包括全局 RFM、Laplace 核、给定聚类信息的 oracle RFM/Laplace；结果显示当 K 增大时，MLP（尤其 ReGLU）性能接近 oracle，远超全局 RFM；理论上在 K→∞ 时 MLP 的样本复杂度显著优于传统核方法和 RFM。

**⚠️ 局限性**

限制在于仅在合成数据和简化的两层网络上验证，缺乏对真实高维数据或更深网络的广泛实验；理论分析仅覆盖单索引混合模型，未讨论多维目标或更复杂链接函数的通用性。

---

## 200. Macro-Operator Generation and Predicate Selection for TAMP Operator Learning

**arXiv ID:** 2608.23629 | [PDF](https://arxiv.org/pdf/2608.23629v1)

**作者:** Can Emir Bora `[一作]` (Bogazici University), Emre Ugur `[通讯]` (Bogazici University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种自动生成宏操作符和谓词选择的方法，显著提升了任务与运动规划（TAMP）的规划效率和可行性。

**💡 创新点**

创新点在于：①通过“效应互补分析”从演示数据中自动发现因果关联的动作对并合成为宏操作符；②提出迭代谓词选择（IPS）技术，删除所有未被任何学习到的操作符使用的谓词，减少搜索时的符号状态开销。

**🔧 技术方法**

技术方法包括：LOFT式的操作符学习（预条件、影响概率估计及确定化），宏操作符生成的四阶段算法（效应识别、因果对匹配、验证冲突消除、宏构建），以及宏连续采样器的顺序采样机制。

**📊 数据集**

实验使用四个TAMP域（Cover、Blocks、Painting、Kitchen）中的演示轨迹和随机探索轨迹，所有数据均由PyBullet仿真生成。

**📈 对比分析**

与基线LOFT方法对比，宏+IPS组合在Blocks域可实现约4.6倍的规划速度提升，在Kitchen域（需超过50步的长序列）则使得原基线无法求解的任务全部可解，规划时间从失败到约0.05秒；在短任务中成功率保持不变或略降。

**⚠️ 局限性**

局限性包括：仅考虑长度为两步的宏，无法处理更长链或嵌套序列；仅删除而非生成新谓词；采样器仍手工设计；仅在仿真中验证，缺乏真实机器人鲁棒性和在线学习能力。

---

## 201. Incorporating Cognitive Load and Knowledge Transfer for Multi-Domain Knowledge Tracing

**arXiv ID:** 2608.24005 | [PDF](https://arxiv.org/pdf/2608.24005v1)

**作者:** Haotian Zhang `[一作]` (University of Science and Technology of China), Qi Liu `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种多域知识追踪框架 LT‑MKT，旨在通过整合认知负荷和知识迁移机制，动态评估学生在跨域学习情境下的知识状态。

**💡 创新点**

创新点：
- 同时建模认知负荷（题目难度、域切换、域覆盖）与知识迁移（内域前置关系与跨域相关性）。
- 通过大型语言模型（LLM）自动生成多域层级图，捕获题目-概念和概念-概念的语义关联。
- 设计了内域 GAT 与跨域 GAT 的两层知识迁移网络，并结合 GRU 进行状态演化，形成完整的认知负荷驱动知识追踪流程。

**🔧 技术方法**

技术手段：
- 大语言模型（Qwen‑plus / GPT‑5 等）用于问答文本与概念信息的链式推理，构建 Multi‑Domain Hierarchical Graph (MDHG)。
- 题目文本使用 BERT 进行编码，认知负荷通过嵌入三维特征（难度、域切换、域覆盖）形成。 
- 状态演化采用 GRU，知识迁移采用两层 GAT（intra‑GAT 与 inter‑GAT）。
- 预测层结合知识状态、题目嵌入与认知负荷，使用二元交叉熵训练。

**📊 数据集**

数据集：
- JuniorH 与 SeniorH（iFLYTEK 自有，涵盖数学、物理、英语）。
- PTADiscJP（Java 与 Python）和 PTADiscDS（C 与数据结构）。
共计 4 个真实多域学习数据集，覆盖 4 万+ 学生、1.5 万+ 概念和 12 万+ 交互。

**📈 对比分析**

比较方法与性能：
- 与 11 种主流知识追踪基线（DKT, GKT, AKT, HawkesKT, LPKT, DIMKT, AT‑DKT, MIKT, SINKT, promptKT, TransKT）在 AUC、ACC、RMSE 上进行对比。 
- 在所有四个数据集上，LT‑MKT 均获得最高 AUC 与 ACC、最低 RMSE，提升幅度在 1.5%–4% 之间，显著优于现有最优方法（如 TransKT）。

**⚠️ 局限性**

局限性：
- 认知负荷只考虑了题目难度、域切换与域覆盖，未包含情绪、注意力等其他心理因素。 
- 依赖 LLM 生成图的质量，LLM 规模与成本会影响实际部署。 
- 仅在四个数据集上验证，泛化能力与对更大规模、多学科场景的适用性仍需进一步研究。 
- 模型复杂度高，解释性与实时推断效率尚未系统评估。

---

## 202. LUX: A Lesion-Aware Graph-Conditioned Visual - Language Architecture for Explainable Endoscopic Captioning

**arXiv ID:** 2608.23853 | [PDF](https://arxiv.org/pdf/2608.23853v1)

**作者:** Alexis Ivan Escamilla-Lopez `[一作]` (Tecnologico de Monterrey), Sharib Ali `[通讯]` (University of Leeds)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 LUX，一种基于病变图的可解释医学图像描述模型，用于炎症性肠病（溃疡性结肠炎）内镜图像的自动注释。

**💡 创新点**

创新点在于：①将 Grad‑CAM 与 CBAM 激活映射转换为病变节点并构建场景–病变图；②在 T5 解码器中使用双向交叉注意力，使每个词聚焦于具体病变节点，从而实现词级证据对齐；③加入词‑病变对齐损失、伪标签增强和图对齐约束，提升语义一致性与可解释性。

**🔧 技术方法**

核心技术包括：CBAM‑增强 ResNet‑50 视觉编码器、Grad‑CAM 病变定位、图卷积网络（GCN）构建病变关系图、T5 Transformer 语言解码器、双向交叉注意力和 token‑病变对齐机制。

**📊 数据集**

使用两大数据集：①UC‑Caption（500 张带有专家注释和 MES 标签的图像），②LIMUC（11276 张无文本但有 MES 标签的图像，其中 400 张配有像素级病变掩码）。

**📈 对比分析**

在多项指标上与多种基线（CNN‑RNN、CBAM‑ResNet、GCN‑Encoder、各种医学 VLM 及大规模 VLM）进行统一评估；LUX 在 BLEU‑4、METEOR、ROUGE‑L、CIDEr、SPICE 上分别取得 0.41、0.29、0.44、0.92、0.25 的最高分，同时 MES 预测准确率 84.7%，幻觉率仅 5.3%（比最强对手低 44%）。

**⚠️ 局限性**

局限性包括：①依赖人工标注的病变掩码仅覆盖 400 张样本，标注成本高；②模型仍为单帧静态图像，未处理连续视频序列；③在极端光照或器械遮挡条件下的鲁棒性待进一步验证。

---

## 203. LLM Agents Perform Controlled Experiments Using Simulation Models

**arXiv ID:** 2608.23622 | [PDF](https://arxiv.org/pdf/2608.23622v1)

**作者:** Yuchen Xia `[一作]` (University of Stuttgart), Pol Llopart `[通讯]` (AstraZeneca)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了一套多代理框架，将大语言模型与高精度药物工艺仿真模型结合，通过结构化实验设计、执行仿真比较、解释结果并给出基于证据的工艺参数优化建议。

**💡 创新点**

创新点在于：1）将仿真模型视为实验环境，构建面向科学实验的多代理工作流；2）在语言模型推理中嵌入可控实验、可解释的图形化轨迹；3）通过需求分析和实验验证显著提升输出的特异性与可操作性。

**🔧 技术方法**

使用 GPT‑4o 驱动的 LLM 代理（需求分析、规划、操作、解释、报告）+ Executor 代理实现 Python 函数调用；基于可视化的图形化推理轨迹；采用 LUCI、词汇模糊度等指标评估特异性；采用工业级药物结晶仿真模型。

**📊 数据集**

使用包含 5 个药物结晶过程设计任务的测试集（含基线配置和目标），以及对应的仿真模型输出；此外收集了领域专家的人工评估数据。

**📈 对比分析**

通过与无仿真、无需求分析、LLM 单一提示等四种变体对照；采用模糊词/词汇不确定度、LUCI、正确性、帮助度、仿真调用精确率/召回率、仿真确认率等指标。完整系统在特异性、正确性（平均 4.1/5）和帮助度（4.2/5）方面明显优于消融版本，仿真确认率达 76%。

**⚠️ 局限性**

局限性包括：1）高度依赖高保真工业仿真模型，若仿真精度不足或 LLM 生成的假设错误会影响结果；2）仍需人工监督，系统尚未完全自动化；3）在无仿真功能时表现不佳；4）目前仅在药物工艺结晶场景验证，泛化性需进一步验证。

---

## 204. Universal Random Coding for Successive Refinement of Individual Sequences Based on Lempel-Ziv Complexity

**arXiv ID:** 2608.24355 | [PDF](https://arxiv.org/pdf/2608.24355v1)

**作者:** Neri Merhav `[一作]` `[通讯]` (Technion Israel Institute of Technology), Neri Merhav (Technion Israel Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出了基于 Lempel–Ziv（LZ78）复杂度的无先验随机编码方案，用于逐级（两阶段）压缩任意确定性序列。

**💡 创新点**

创新点在于：①将单阶段的无先验随机编码扩展到两阶段，解决了在逐级压缩中出现的 Jensen 障碍；②构建了完全无类型（type‑free）的实现方案，既保留了与类型匹配方案相同的可实现区域，又消除了对类型的预先承诺；③证明该方案在理论上覆盖了先前有限状态编码器方法的可实现区域。

**🔧 技术方法**

采用了 Lempel–Ziv 78 算法的无先验概率分布、条件 LZ 复杂度、典型序列与类型类的计数技术以及随机编码搜索与 Kraft 计数的组合。

**📊 数据集**

无实验数据集，纯理论推导。

**📈 对比分析**

通过与先前有限状态编码器方法的理论对比，证明了本方案在任何给定的失真约束下都能实现更大的可实现区域；与单阶段 LZ 方案比较，得到相同的最优性。

**⚠️ 局限性**

局限在于搜索成本较高，尤其是无类型方案需要在每次候选码字上计算条件 LZ 复杂度；此外，理论结果仅适用于固定块长度的渐进性，实际实现的细节仍待进一步研究。

---

## 205. Degree Centrality Algorithms for Weighted Multilayer Networks (or w-MLNs)

**arXiv ID:** 2608.23876 | [PDF](https://arxiv.org/pdf/2608.23876v1)

**作者:** Ayomide Ayowole-Obi `[一作]`, Sharma Chakravarthy `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在加权同质多层网络（HoMLN）中，基于布尔OR聚合的边结构和最大/和两种权重聚合的加权度中心性，并提出了分解（decoupling）框架及启发式组合方法来估计节点的重要性。

**💡 创新点**

创新点在于：1）引入分解式框架，使得不需要构造全局聚合图即可计算加权度中心性；2）对最大聚合给出上下界估计，证明单层信息无法精确恢复全局中心性；3）提出基于候选集（hub/top‑k）的高效组合策略，实现高准确率与低运行时间的平衡。

**🔧 技术方法**

技术手段包括层级分析与组合的分解框架、启发式上界/下界估计、节点强度求和、候选集筛选（hub / Top‑k）、Jaccard相似度评估、实验对比与运行时间测算。

**📊 数据集**

实验使用了多种合成HoMLN（不同节点/边数、分布、重叠率）以及两份真实数据集：2004‑2005年的作者合作网络和蚂蚁交互网络。

**📈 对比分析**

方法与传统聚合后全局计算（GT）及朴素基线进行对比，使用Jaccard相似度衡量准确性。实验显示UB/All策略准确率与GT相当，而运行时间比GT低数倍；LB和Hub‑Only策略更快但准确率略低。

**⚠️ 局限性**

局限性包括：仅针对两层网络；最大聚合的上界/下界仍为启发式，可能偏离真实值；在稀疏或低重叠图中准确性下降；未考虑更多聚合函数（如min、平均）或其他中心性指标。

---

## 206. AQLoRA: A Zero-Search Recipe for Fast Quantized LoRA Fine-Tuning

**arXiv ID:** 2608.23816 | [PDF](https://arxiv.org/pdf/2608.23816v1)

**作者:** Md Romyull Islam `[一作]` `[通讯]` (Kennesaw State University), Md Romyull Islam (Kennesaw State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种无搜索、基于 NF4 重构误差的逐层精度分配规则，结合 LoRA 适配器和提前停止反向传播，显著提升 4‑bit LoRA 的训练速度。

**💡 创新点**

创新点在于：① 用单次 CPU 计算得到每层误差，直接决定 fp16 层数量；② 将精度分配与层级适配器放置和早停相结合，形成可调的速度‑质量平衡；③ 该规则与工业实践（Unsloth）保持一致，且无需校准数据或搜索。

**🔧 技术方法**

使用技术包括 NF4 量化、重构误差计算、top‑K 层挑选、全精度（fp16）层保护、适配器（LoRA）在前部或全部层插入、梯度检查点与提前停止、标准的长度分组批处理、融合 AdamW 等通用加速手段。

**📊 数据集**

数据集主要是 Commonsense‑170K（8 任务）和 GSM8K（数学推理）作为下游评测，WikiText‑2 用于基准 perplexity。

**📈 对比分析**

与 QLoRA、rsLoRA、LoftQ、fp16 LoRA 等方法对比，AQLoRA 在保持与 QLoRA 同等（或接近 fp16 LoRA）准确率的同时，-q 方案平均提升 4.8% 训练速度，-s 方案提升 11.1%；仅增加约 0.2 GiB 内存。

**⚠️ 局限性**

局限性包括：① 通过 NF4 误差排序的层选择在随机挑选上无显著优势；② 速度提升仅取决于 fp16 层数量而非其身份；③ 在 14B 模型上 fp16 层可能溢出，需改用 fp32；④ 仅验证了整层级别的分配，未覆盖列级或更细粒度的量化；⑤ 速度优势只在预 Hopper 或边缘 GPU 的标准栈上出现，对 kernel‑级或 FP8 方法不适用。

---

## 207. Energy-Aware Performance Evaluation of Nonlinear Mechatronic Systems Under Matched-Tracking Conditions

**arXiv ID:** 2608.23578 | [PDF](https://arxiv.org/pdf/2608.23578v1)

**作者:** Bhanuka Dayawansa `[一作]` `[通讯]` (University of Moratuwa), Bhanuka Dayawansa (University of Moratuwa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证了一种能量感知的匹配性能评估框架，结合跟踪误差与累计执行器能量进行综合评价。

**💡 创新点**

创新性地引入匹配性能对比与能量感知性能指数（EAPI），揭示传统轨迹指标无法体现的能量差异，尤其在非线性系统中。

**🔧 技术方法**

采用PD控制器仿真、能量平衡分析、绝对功率积分以及RMSE等指标，并通过归一化的EAPI进行综合评估。

**📊 数据集**

使用MATLAB仿真数据，参数包括m=1.0、c=0.35、k=25、α=300、F_c=0.8，PD增益范围K_p∈[40,180]、K_d∈[6,30]，时间窗口8s。

**📈 对比分析**

通过在RMSE误差容差内匹配线性与非线性系统的控制参数，比较两者的累计能量与EAPI，结果显示非线性系统能耗提升约106%且EAPI提升约41%。

**⚠️ 局限性**

仅在一维仿真场景下验证，缺乏实验验证；需要手动设定权重λ，且框架在多维系统与复杂控制器下的适用性尚待探索。

---

## 208. Boot-and-Feedback Framework for Generalist-Expert Model Collaboration in Breast Ultrasound Diagnosis

**arXiv ID:** 2608.23974 | [PDF](https://arxiv.org/pdf/2608.23974v1)

**作者:** Ming Cheng `[一作]`, Qiuhong Ke `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出Boot-and-Feedback框架，实现通用多模态大语言模型与专业视觉专家模型的双向协作：Boot阶段通过BI‑RADs词汇对齐和视觉专家预测先验引导生成结构化诊断描述；Feedback阶段采用Attention‑Gated Cross‑Modality Fusion模块融合文本与图像特征，提升诊断准确性与可解释性。

**💡 创新点**

创新点包括：①将BI‑RADs词汇标准化用于LLM的提示，避免无关幻觉；②使用视觉专家的置信先验作为Boot阶段的约束，强化LLM与视觉模型的语义一致性；③设计轻量化的Attention‑Gated Fusion模块，实现文本与视觉特征的自适应门控融合；④通过双向交互提升LLM文本描述的诊断可靠性和解释性。

**🔧 技术方法**

技术手段：Gemini 2.5 Pro等多模态LLM；任务激活提示 + 以BI‑RADs为约束的in‑context学习；视觉专家模型（ResNet‑50、MedViT）+ 置信先验提取；RadBERT文本编码器；Attention‑Gated Cross‑Modality Fusion（AGCFM）实现文本-视觉交互；交叉熵训练与5‑折交叉验证。

**📊 数据集**

使用公开乳腺超声数据集：BUS‑BRA（1268 benign, 607 malignant）和BUSI（437 benign, 210 malignant）进行训练与评估。

**📈 对比分析**

与现有单模态（Vision Expert）、双模态（Vision‑Language Expert）、以及多种SOTA方法（KRC‑APM、HoverTrans、REAF、BD‑StableNet、SgmaFuse）进行对比。BooF在BUSI上取得AUC 0.959、Specificity 0.962、F1 0.885；在BUS‑BRA上取得AUC 0.976、Recall 0.903、F1 0.899，均显著优于对照组，并在多项指标上实现绝对提升4.8%–6.1%（AUC）、5.7%–7.4%（Accuracy）、14%–16%（Recall）。

**⚠️ 局限性**

局限性：①对闭源LLM的微调成本高，依赖外部LLM接口；②当前框架仅支持单轮交互，缺乏多轮迭代以进一步提升鲁棒性；③尽管先验引导降低幻觉，但仍可能出现文本与视觉不一致的情况；④实验仅在两个公共数据集上验证，需在更大规模、不同设备下进一步评估稳健性。

---

## 209. A Survey of Timing Variability in Microservice-Based Software-Defined Vehicles

**arXiv ID:** 2608.23649 | [PDF](https://arxiv.org/pdf/2608.23649v1)

**作者:** Cyrus K. Vattes `[一作]` (Eindhoven University of Technology), Nirvana Meratnia `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 4866 | [OpenAlex ID](https://openalex.org/A5003885813)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了微服务架构，尤其是软件定义车辆（SDV）中时序变异的来源、传播机制与影响，并对现有时序模型进行分类与评估。

**💡 创新点**

创新点在于把时序变异视为由服务依赖、资源共享和控制循环交互产生的系统级涌现现象，并系统地评估传统时序模型在动态、分布式微服务环境中的适用性与局限。

**🔧 技术方法**

使用了网络计算、实时计算、概率模型（如随机流量、尾部延迟分析）和基于状态的模型（Petri网、Markov模型）等多种形式化时序分析技术，对它们的假设、保证与可扩展性进行对比。

**📊 数据集**

论文并未使用具体实验数据集，而是基于SDV的典型工作流（如障碍检测与自动刹车）进行示例性阐述，并引用已有研究中的延迟分布与吞吐量数据来说明问题。

**📈 对比分析**

通过对比分析，发现传统网络计算和实时计算模型在动态负载、异步通信与资源争用方面往往过于保守或失效；概率模型更能反映尾部延迟，但缺乏硬实时保证；基于状态的模型在准确性上优于前两类，但由于状态空间爆炸难以直接应用于大规模系统。性能方面，作者指出这些模型在小规模实验中的理论上可行，但在实际 SDV 环境下往往面临可观测性不足与计算开销高的问题。

**⚠️ 局限性**

局限性包括：① 所有模型均基于对工作负载、调度策略与资源争用的稳定或可观测假设，难以适应微服务的自适应、异构与部分可观测特性；② 现有模型难以对资源共享和控制循环中的非线性放大效应进行精确建模；③ 缺乏统一的评价指标和大规模实验验证，导致模型适用性评估不充分；④ 在 SDV 等安全关键系统中，缺少对混合关键性干扰的完整分析与保证。

---

## 210. Luce: Relightable Gaussians for 3D Asset Generation

**arXiv ID:** 2608.23943 | [PDF](https://arxiv.org/pdf/2608.23943v1)

**作者:** Mayank Singh `[一作]` (Apple), David E. Jacobs `[通讯]` (Apple)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种多模态 PBR 高斯云表示，能够从单张图片生成可重新照明的 3D 资产；

**💡 创新点**

创新点在于：①为每个占据体素分别存储颜色、金属-粗糙度和法向三种 PBR 模态的高斯云，消除模态竞争；②使用结构化 VAE 压缩该高斯云为紧凑潜在空间，并用 rectified‑flow transformer 在该空间生成潜在；③利用多层 DINOv2 特征保留细节；④将解码得到的法向高斯烘焙为切线空间法向贴图，提高网格细节；

**🔧 技术方法**

技术包括：多模态 3D Gaussian Splatting、稀疏体素网格、结构化 VAE、rectified‑flow Transformer、DINOv2 预训练特征、Cook‑Torrance PBR 计算、切线空间法向贴图烘焙；

**📊 数据集**

训练数据主要是 Objaverse、Objaverse‑XL 和 TexVerse 的约 658K 个 PBR 资产；评估使用 Toys4K（338 资产）和 130 张 AI 生成图像的基准；

**📈 对比分析**

与 TRELLIS、TRELLIS 2、LiTo、3DTopia‑XL 等基线比较；在 Toys4K 上 FID 下降到 20.99（比 TRELLIS 2 降 8+ 点），在 AI 图像基准上 CLIP 与 SigLIP2 得分最高；网格生成的 PSNR 与 SSIM 也均优于对手；

**⚠️ 局限性**

局限包括：体素分辨率有限，细节可能被低分辨率捕捉不足；仅支持标准 PBR 三模态，未覆盖更复杂材质；对场景级生成及交互支持有限；

---

## 211. An AI-Based Approach to Early Reporting and Justice Initiation in Image-based Sexual Abuse. A Pilot Study

**arXiv ID:** 2608.24412 | [PDF](https://arxiv.org/pdf/2608.24412v1)

**作者:** Mattia Falduti `[一作]` (Square Mediterranean Centre for Revolutionary Studies), Anca Radu `[通讯]` (European University Institute)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并测试了一个基于 GPT‑4o Mini 的在线 AI 代理，帮助图像性虐待（IBSA）受害者收集信息并自动生成正式的初步报告，以支持他们向平台和执法机关报案。

**💡 创新点**

创新点在于：①面向多语言和多司法体系的通用报告模板；②在收集信息时采用分步、简短提问，减少受害者空白页焦虑；③通过专家评估验证报告的可用性，而非仅停留在技术演示层面。

**🔧 技术方法**

技术：GPT‑4o Mini 语言模型、No‑Code 在线对话平台、前置安全与提示工程（guardrails）以及多语言选择功能。

**📊 数据集**

本文未使用公开数据集，而是通过人工编写的提示语与案例对话来训练和测试代理；评估数据来源为三位法律专家的问卷反馈。

**📈 对比分析**

评估方法：让三名法律和执法专家与代理交互，并使用四点李克特量表（Strongly Agree, Agree, Disagree, Strongly Disagree）回答关于信息完整性、指引有效性和报告可用性的问题。结果显示，所有专家均对信息完整性和报告可用性给出“同意”或“强烈同意”，但对指引的细节改进给出“同意”或“强烈同意”。

**⚠️ 局限性**

限制：①评估样本仅为三名专家，缺乏普通用户验证；②未在真实案件或大规模用户测试中验证效果；③功能仅限于报告生成，未包含后续的法律援助、热线信息或与平台的集成；④模型可能受限于 GPT‑4o Mini 的知识与安全策略。

---

## 212. On Scaling Coordinate-Based Neuroevolution: The Quadtree Bottleneck in ES-HyperNEAT

**arXiv ID:** 2608.24480 | [PDF](https://arxiv.org/pdf/2608.24480v1)

**作者:** Romain Claret `[一作]` (University of Neuchâtel), Kilian Stoffel `[通讯]` (University of Neuchâtel)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了基于JAX的ES‑HyperNEAT（JAX‑ESHN），在GPU上并行化子网络构造并与CPU版PUREPLES基线进行对比，揭示了自适应四叉树分割导致的结构性瓶颈。

**💡 创新点**

发现每个CPPN在四叉树分割中产生不同的子网络位置集合，使得基于静态形状编译（JAX/XLA）的批量化不可行；并提出EMR‑HyperNEAT等前瞻性方法可消除此瓶颈。

**🔧 技术方法**

使用JAX（XLA）、TensorNEAT、Python递归实现四叉树分割、批量CPPN查询、Vmap、预计算坐标偏移等技术，并对比CPU版NEAT+HyperNEAT。

**📊 数据集**

在五个基准任务上验证：XOR、Parity‑3、circle classification、sine regression、CartPole，涵盖布尔、连续和控制问题。

**📈 对比分析**

对比方法：计算总运行时间、构造开销、求解率和每代耗时；JAX‑ESHN在深层子网（depth ≥4）实现近100%求解率，Baseline仅在浅层求解率高；JAX‑ESHN的构造开销随深度指数增长，但在深层求解时比Baseline更快；对比显示CPU‑vs‑CPU控制验证瓶颈非GPU硬件所致。

**⚠️ 局限性**

局限性：由于子网络位置集合的可变长度，静态形状编译框架无法实现全族级向量化，GPU利用率低；构造开销在深层成为主要成本；当前实现未解决动态形状GPU加速，仅通过EMR等方法可望突破；此外，实验受NEAT库默认参数影响，求解率与实现差异相关。

---

## 213. A Drop-in KEM Replacement for Client Signatures in Post-Quantum SSH

**arXiv ID:** 2608.24447 | [PDF](https://arxiv.org/pdf/2608.24447v1)

**作者:** Hongbo Liu `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Li Zhou `[通讯]` (Institute of Software, Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种可直接替换 SSH 客户端签名的 KEM 基础用户认证方法，保留公钥凭证模型并实现可否认身份验证。

**💡 创新点**

创新点在于通过会话绑定的 KEM 响应完成身份证明，在不增加额外往返的前提下提供后量子 ACCE 安全证明，并实现了可否认且轻量的认证机制。

**🔧 技术方法**

采用 ML‑KEM 及 HMAC 作为响应函数，在 OpenSSH 中实现并使用 liboqs 进行后量子算法的加密与解密操作。

**📊 数据集**

使用多种 NIST 级别后量子算法（ML‑DSA、Falcon、SLH‑DSA、ML‑KEM 等）与传统 Ed25519 作为基准，在 OpenSSH 10.2p1 环境下进行实验。

**📈 对比分析**

在代表性 RTT（37/67/163 ms）和 TCP 初始窗口（3–50 MSS）设置下与 Ed25519、ML‑DSA、SLH‑DSA 等方法比较，KEM 认证在大签名场景下可减少约 10% 的握手延迟，服务器端在线加密成本降低约 60%，在小窗口和高并发负载下优势更明显。

**⚠️ 局限性**

局限性包括：实现仅覆盖客户端认证层，导致可否认性限制审计与不可否认需求；侧信道安全未完全评估；协议规范仍需标准化才能广泛部署。

---

## 214. Diverse by Reasoning: Harnessing the Wisdom of LLM Crowds for Future Prediction

**arXiv ID:** 2608.24001 | [PDF](https://arxiv.org/pdf/2608.24001v1)

**作者:** Nirupam Chetlapalli `[一作]` (University of Maryland Baltimore County), Keke Chen `[通讯]` (University of Maryland Baltimore County)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于LLM推理行为的“行为感知”框架，通过聚类和代表性选择构建小而有效的LLM预测众包，专门用于未来事件预测。

**💡 创新点**

创新点在于：①用LLM在独立发展任务上的推理轨迹来刻画行为多样性，而非依赖模型元信息；②在聚类得到的行为群组中选取代表（medoid）而非随机或全部模型，从而显著提升众包预测精度并降低成本；③系统性评估不同众包构造策略的效果，并证明“代表性行为多样性”比单纯最大化差异更有益。

**🔧 技术方法**

技术包括：使用文本编码器对推理轨迹进行向量化并归一化得到模型行为签名；K‑means++与层次聚类确定行为簇；从每个簇选取medoid或k个最大不相似模型；使用多数/中位/模具等简单投票/聚合规则；对比随机、专家、全模型投票等基线；同时在结果上考虑成本（模型调用次数与费用）。

**📊 数据集**

行为建模使用了7个异构开发基准（LiveBench/Reasoning, LiveBench/Math, LiveBench/Instruction‑Following, LiveBench/Data‑Analysis, GPQA Diamond, Natural Plan, LiveCodeBench/Execution‑v2，共350题）；预测评估使用FutureX‑Past和Bench‑to‑the‑Future v3两个未来预测基准（各100题）。

**📈 对比分析**

与全模型投票、随机小众、专家单模型等对照实验表明：使用3个K‑means++簇的medoid众包在两大基准上均取得最优或次优成绩（FutureX‑Past 0.302，BTF‑v3 0.810），相较于25模型投票精度提升~0.006–0.014，同时模型调用成本和费用下降约80%/88%。

**⚠️ 局限性**

局限性：仅评估了25个LLM，且众包是静态、统一的；行为签名采用平均嵌入，可能掩盖任务相关的细粒度行为差异；未考虑针对不同预测问题动态构造或任务感知的众包；并未验证在更大、更复杂的模型群体或更广泛预测领域中的泛化性。

---

## 215. Trusted Polytopic Action Sets for Fast Planning in Underactuated Systems

**arXiv ID:** 2608.24019 | [PDF](https://arxiv.org/pdf/2608.24019v1)

**作者:** Akshay Jaitly `[一作]` (Onyx Robotics), Siavash Farzan `[通讯]` (California Polytechnic State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对欠驱动系统，快速在线生成可信的多边形动作集（PAS），并将其作为可复用的凸多段动作集合嵌入采样树中，实现长时程规划。

**💡 创新点**

创新点在于：① 将凸化操作迁移至动作坐标空间，通过动力学违背度量定义信任区域；② 使用 IRIS‑ZO 零阶采样在动作空间内逼近内核，从而得到可信 PAS；③ 将 PAS 作为可复用的凸多段动作集合与树搜索结合，实现高效的碰撞与控制约束处理。

**🔧 技术方法**

技术手段包括：线性时间变（LTV）线性化、分段线性（PWL）输入参数化、动力学违背度量、IRIS‑ZO 的零阶采样与切割、线性规划（LP）求解、GPU 并行评估、HiGHS 求解器、OSQP 二次规划、CUDA 并行计算。

**📊 数据集**

实验数据集：2D 双积分器在随机圆形障碍场中的障碍场（4–63% 占用率）；Cartpole 摆动上升任务（真实非线性动力学）。

**📈 对比分析**

与 kinodynamic RRT（OMPL）进行对比，PAS‑RRT 在相同场景下平均速度提升 14–78 倍，终端误差比采样和 NLP 基线低 26–86%；在高密度障碍场保持 100% 成功率，树深度仅为 1–2 层，规划时间在 30–120 ms 之间。

**⚠️ 局限性**

局限性：① 依赖局部 LTV 线性化，若本地模型误差大或通道过窄会失效；② 内核逼近是近似的，受非线性程度和模型条件影响；③ 对极端非线性动力学或不可达目标的表现不佳；④ 需要手动设置动力学违背阈值 ζ_max，影响探索与精度平衡。

---

## 216. Resource Allocation for Secure Dual-UAV-Assisted ISAC System

**arXiv ID:** 2608.24398 | [PDF](https://arxiv.org/pdf/2608.24398v1)

**作者:** Hongjiang Lei `[一作]` (Chongqing University of Posts and Telecommunications), Gaofeng Pan `[通讯]` (Beijing Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出并实现了一个双UAV辅助的安全ISAC系统，其中基站UAV负责通信与雷达探测，友好UAV负责发射噪声以干扰潜在的窃听者。

**💡 创新点**

创新点在于考虑窃听者位置不确定性，联合优化用户调度、时隙分配、功率、波束和双UAV轨迹，并采用BCD+SCA+SDR的迭代算法实现近似全局最优。

**🔧 技术方法**

主要技术包括凸优化、顺序凸逼近、半正定松弛、Bianchi分配与路径规划等。

**📊 数据集**

通过多场景仿真（不同用户分布、能耗限制、速度等）验证。

**📈 对比分析**

与固定轨迹和单天线对手方案对比，提出方案的平均安全速率(A SR)显著提高，收敛速度快。

**⚠️ 局限性**

局限性：仅考虑单个静止窃听者、LoS单向信道、二维轨迹，未考虑IRS、多窃听者或移动窃听者。

---

## 217. SAGE: From Direct Answering to Evidence-Grounded Inference for Chinese Ancient Document Understanding

**arXiv ID:** 2608.24011 | [PDF](https://arxiv.org/pdf/2608.24011v1)

**作者:** Yuchuan Wu `[一作]` (Fudan University), Bin Li `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SAGE，一种基于多代理、证据驱动的框架，用于中文古籍文档的理解与推理，强调任务规划、工具辅助的证据获取、主张级验证和有限重规划；

**💡 创新点**

创新点在于将古籍理解从单步生成转化为分阶段、可追溯的证据驱动推理流程，并通过共享状态运行时实现规划、执行与验证的协同；

**🔧 技术方法**

采用多代理架构（规划代理、执行代理、验证代理）、受限共享状态运行时、预定义工具接口（页面读取、文本规范化、实体提取、局部检索、答案合成）以及主张级验证和重规划机制；

**📊 数据集**

使用了 AncientDoc 基准数据集进行实验评估；

**📈 对比分析**

与直接回答的 LVLM 基线（InternVL3‑8B、Qwen2.5‑VL‑7B、Qwen3.5‑9B）在 CHRF++ 与 BERTScore F1 上进行对比，SAGE 在所有四个理解任务上均实现提升，甚至在部分指标上超过规模更大的直接生成模型；

**⚠️ 局限性**

局限性包括仅在 AncientDoc 任务上验证，验证报告和支持率为系统生成而非人工事实性注释，额外的规划/工具/验证开销，以及对更广泛历史文档、不同布局或其他古语言的适用性尚待验证。

---

## 218. ACE: A Self-Correcting Agentic Canvas Editor for Multi-Slide Presentation Automation

**arXiv ID:** 2608.24103 | [PDF](https://arxiv.org/pdf/2608.24103v1)

**作者:** JooYoung Jang `[一作]` (Seoul National University), Nojun Kwak `[通讯]` (Seoul National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ACE框架，利用代理式画布编辑器在演示模板编辑中实现无参考的指令遵循与自校正。

**💡 创新点**

创新点：①无参考指令遵循评估器，②基于场景图的98种专业工具动作空间，③内容感知路由CARE，④自校正循环。

**🔧 技术方法**

技术手段：大语言模型（Claude、GPT 等）、场景图表示与可编辑操作、内容感知路由、指令遵循评估器、渲染与可视化。

**📊 数据集**

数据集：Figma‑Slides 基准 97 任务（94 可评估），包含 PPTArena 适配任务与 12 个新任务。

**📈 对比分析**

与 Claude‑Skill HTML、PPTArena、OpenXML 基线对比；ACE 在指令遵循 IF 上平均 4.45，速度 1.75×、成本约 44% 更低；人类评审亦偏好 ACE。

**⚠️ 局限性**

局限：评估器与自校正循环使用同一模型可能引入循环；VQ 差异统计不显著；仅评估已给模板，未覆盖模板检索；非 Figma 平台迁移未实现。

---

## 219. Evaluating Language Models on Cross-Language Code Functional Equivalence

**arXiv ID:** 2608.23961 | [PDF](https://arxiv.org/pdf/2608.23961v1)

**作者:** Hui Sun `[一作]` (North Carolina State University), Wesley K. G. Assunção `[通讯]` (North Carolina State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在跨语言人写代码功能等价判断中的表现，并构建了PolyHuman数据集进行评估。

**💡 创新点**

创新点在于提出了真实人写代码跨语言功能等价评测基准，并对模型失败原因从知识、抽象层面进行系统性分类。

**🔧 技术方法**

采用零样本/提示式推理、链式推理（CoT）以及统计分析和逻辑回归等技术对LLM做功能等价判断。

**📊 数据集**

使用PolyHuman数据集（来自CodeContests的CPP、Java、Python程序）以及EquiBench、SeqCoBench等基准进行对比。

**📈 对比分析**

对比多种开源与专有LLM，发现GPT‑o4‑mini在人写代码上准确率仅约0.8，跨语言更低，且表现受难度、相似度和语言偏差影响。

**⚠️ 局限性**

限制包括模型对难度和抽象层面缺乏稳定推理、数据集可能存在标签噪声、实验只覆盖有限模型规模与语言，难以完全代表工业场景。

---

## 220. AffineTok: Semantic Affine Consistency for Diffusion-Friendly Visual Tokenizer

**arXiv ID:** 2608.23864 | [PDF](https://arxiv.org/pdf/2608.23864v1)

**作者:** Junqiu Yu `[一作]` (Fudan University), Yanwei Fu `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的视觉 tokenizer 训练框架 AffineTok，强调在扩散过程中语义一致性（Semantic Affine Consistency，SAC）与对应的 tokenizer 侧指标 ；

**💡 创新点**

创新点包括：①首次将 SAC 定义为 denoising 语义恢复的缺失约束，并推导其正交分解；②设计了 tokenizer 侧指标  用于预测扩散生成质量；③提出两种训练组件（GSCT 与 PMSA）在 tokenizer 训练阶段实现 SAC，且不改动下游扩散训练；

**🔧 技术方法**

使用了视觉基础模型（DINOv3）进行语义监督，Transformer 架构用于 GSCT 与 PMSA；采用无监督对齐与投影器进行后验均值估计；训练中结合重构、LPIPS、GAN 与 KL 损失；

**📊 数据集**

主要使用 ImageNet‑1K（256×256）作为数据集进行 tokenizer 训练与扩散评估；

**📈 对比分析**

通过对比 14 种公开 tokenizer 与 13 种扩散模型，在 SiT‑B 与 SiT‑XL 两个规模下与 gFID、IS、Precision/Recall 等指标对比；AffinityTok 在 600 轮训练后 gFID 达到 1.21（无 CFG）/1.10（CFG），在 ImageNet 256×256 上实现了 state‑of‑the‑art ；

**⚠️ 局限性**

局限性包括：①对 VFM（DINOv3）和 ImageNet 数据的依赖，泛化性尚未验证；②训练过程中需要额外的 GSCT 与 PMSA 计算，增加算力负担；③仅在 ImageNet 256×256 上评估，缺乏对高分辨率或其他域的验证。

---

## 221. From Anonymous Shapes to Named Places: A Tool for Braille and Place-Semantic Annotation of Tactile Maps

**arXiv ID:** 2608.23820 | [PDF](https://arxiv.org/pdf/2608.23820v1)

**作者:** Li Liu `[一作]` (University of California), Leilani H. Gilpin `[通讯]` (University of California)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一款基于浏览器的点击式注释工具，允许视力辅助者在已生成的3D打印触觉地图上添加盲文标签，并将地图与地理信息（OpenStreetMap）进行匹配；

**💡 创新点**

创新点包括：①把盲文注释放在几何生成后作为后处理；②提供点击式OSM匹配与可编辑的缩写生成；③设计符合ISO 17049的打印安全盲文点几何并提供审核步骤；③导出可重用的每标签记录，分离打印几何与语义，支持多种输出形式。

**🔧 技术方法**

技术实现：前端Web技术（HTML/JS），OpenStreetMap API用于地图匹配，Three.js或类似库渲染3D STL模型，ISO 17049标准下的盲文点生成逻辑，JSON/GLB导出接口，浏览器端无服务器架构。

**📊 数据集**

使用的数据集：OpenStreetMap建筑足迹与属性；Touch Mapper生成的3D STL模型（已打印的触觉地图）；实际打印的触觉地图（用于用户评估）。

**📈 对比分析**

评估方法：在十名盲/弱视读者中进行成对比较，分别给出无标签和已注释的打印地图；结果显示4人能熟练阅读盲文，其他人提出音频或非盲文形式需求；对盲文点感知与间距也收集了定性反馈。

**⚠️ 局限性**

局限性：需要视力注释者手动确认匹配与缩写；盲文标准支持有限（仅UEB/ISO 17049），缩写规则不够完善；目前仅支持打印盲文，未实现音频或符号输出；评估为定性探索性研究，缺乏大规模定量性能指标。

---

## 222. Independent Languages and Witnesses of Dependence (Non-satisfaction)

**arXiv ID:** 2608.24254 | [PDF](https://arxiv.org/pdf/2608.24254v1)

**作者:** Stavros Konstantinidis `[一作]` `[通讯]` (Saint Mary's University), Stavros Konstantinidis (Saint Mary's University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种用语言方程φ(X)=∅来定义独立语言（即满足某种独立性属性）的理论框架，并引入了“树证据（tree witness）”的概念，用来精确描述非满足φ(L)=∅时的违背证据；

**💡 创新点**

创新点包括：①证明任意由φ定义的独立属性都是Jürgensen独立性；②给出对正则语言L求树证据的算法，并证明其在φ固定时为多项式时间，在一般情况下为PSPACE‑完整；③通过路径可逆的自动机构造实现从接受路径反推出构造树证据所需的L-词；

**🔧 技术方法**

主要技术包括：语言表达式（φ）的递归定义、自动机与有限转导的构造、路径可逆（path‑invertible）自动机操作、树结构遍历（DFS/BFS）来生成增广自动机，以及基于接受路径的非确定性空间算法；

**📊 数据集**

本文没有使用实验数据集，全部工作为理论分析与算法设计；

**📈 对比分析**

方法的性能通过计算复杂度分析给出：在φ固定时，树证据算法为O(|L|^1+|φ|_∩)；在一般情况下，判定问题为PSPACE‑完整，证明了其不可在多项式时间内完成；

**⚠️ 局限性**

局限性：目前的独立性表达式框架不足以表达UD码等更复杂的代码属性；树证据的最小化（最少词数或最短词）未讨论；未来研究方向包括构造满足给定φ的极大语言及其最小树证据。

---

## 223. Equivariant Cellular Sheaves for Molecular Electronic Structure: Bridging Sheaf Cohomology and E(3)-Equivariant Hamiltonian Learning

**arXiv ID:** 2608.23571 | [PDF](https://arxiv.org/pdf/2608.23571v1)

**作者:** Krishna Harish `[一作]` `[通讯]` (Elkins High School), Krishna Harish (Elkins High School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

将局域轨道电子哈密顿量表述为细胞层 sheaf 的拉普拉斯算子，构建了 E(3)‑等变形的细胞 sheaf 网络（ECSN）来预测电子结构；

**💡 创新点**

核心创新在于将电子哈密顿量与细胞 sheaf 拉普拉斯之间的严格对应关系，以及利用 sheaf 同调得到的非键合轨道计数和环路信息的拓扑不变量；

**🔧 技术方法**

使用的技术包括：O(3) 诱导可旋转核函数、细胞 sheaf 结构、Sheaf Laplacian 与 Hodge-Laplacian、细胞差分算子、sheaf 扩散层以及基于同调的读出头；

**📊 数据集**

验证数据集主要为 11 个共轭 π 分子（如苯、纳菲啶、环丁二烯等）和随机生成的小分子；

**📈 对比分析**

与基准方法相比，ECSN 在均方误差（MAE）上相对于坐标 MLP 下降 58%，并且对随机旋转具有完全不变性；还通过数值实验验证了同调计数与已知非键合轨道数的完全一致性；

**⚠️ 局限性**

局限性包括：需要选取 PSD 能量基准 E_ref，限制了可解释性；sheaf 约束下的映射仅在可观测的拉普拉斯下可辨识；环路基底不唯一且需要固定；计算复杂度比标量 GNN 高 d² 倍；仅适用于单体（均场）电子结构，无法处理多体相关系统。

---

## 224. A survey detection channel overrides the pixels in an astronomical foundation model, and biases tomographic mean redshifts

**arXiv ID:** 2608.23626 | [PDF](https://arxiv.org/pdf/2608.23626v1)

**作者:** Ihor Kendiukhov `[一作]` `[通讯]` (Independent researcher), Ihor Kendiukhov (Independent researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对一款训练于五大天文调查的39模态transformer进行审计，发现其过度依赖来自catalogue的元数据，导致检测门控误差并产生显著的系统偏差。

**💡 创新点**

创新点在于首次通过因果干预揭示模型在图像与元数据冲突时仍优先使用元数据，以及量化这种偏好对光度红移及宇宙学分析的实质性影响。

**🔧 技术方法**

采用因果干预、激活补丁、概念抹除和稀疏词典学习等可解释性技术，结合分辨率有限的tokeniser评估，对模型内部机制进行定量诊断。

**📊 数据集**

使用DESI BGS‑bright 交叉匹配数据（约2×10^8对象），Legacy Survey、Hyper Suprime‑Cam、SDSS、DESI和Gaia的图像与光度以及HSC深场切片。

**📈 对比分析**

通过在真实数据上对分割图像做位移、交换和掩码替换等干预，比较模型输出与对照组，验证去除检测通道可消除偏差，且模型在不损失性能的情况下保持精度。

**⚠️ 局限性**

局限性包括仅针对单一模型家族与特定深度的BGS-bright样本，缺乏对更深或更混合环境的验证，稀疏字典评估受随机种子影响，以及检测缺失率和光度-红移耦合在其他调查中的普适性不明。

---

## 225. Robust Data-Collection Policy Learning for Low-Variance Online Policy Evaluation

**arXiv ID:** 2608.24146 | [PDF](https://arxiv.org/pdf/2608.24146v1)

**作者:** Claire Chen `[一作]` (California Institute of Technology), Shangtong Zhang `[通讯]` (University of Virginia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种双循环梯度算法，在考虑转移不确定性的环境下学习鲁棒行为策略，以显著降低策略评估的方差。

**💡 创新点**

创新点在于将行为策略搜索与对抗性转移建模相结合，推导了转移梯度表达式，并给出了全局收敛保证。

**🔧 技术方法**

采用了最小–最大优化、对抗性梯度更新、重要性采样、线性–softmax策略参数化以及投影梯度下降等技术。

**📊 数据集**

在随机Garnet MDP和库存管理（Inventory Management）任务上进行实验验证。

**📈 对比分析**

与传统的MC、BPG、ROS等基线比较，所提方法在对抗转移条件下实现了更低的评估方差，表现出更强的鲁棒性。

**⚠️ 局限性**

局限性包括对抗梯度计算的额外成本、对重要性采样比率的有界性假设以及实验范围仍局限于相对简单的任务。

---

## 226. HAP: Head-Adaptive Visual Token Pruning via Cross-Modal Alignment

**arXiv ID:** 2608.23921 | [PDF](https://arxiv.org/pdf/2608.23921v1)

**作者:** Yuanhao Sun `[一作]` (Shanghai Jiao Tong University), Xinbing Wang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无训练、无超参调优的视觉标记剪枝框架HAP，基于跨模态注意力的Prompt‑Grounded Attention Quality（PAQ）进行头级和层级聚合，动态分配FLOPs预算并保留最相关视觉标记。

**💡 创新点**

核心创新是将PAQ作为衡量注意力头对文本提示的对齐质量的指标，利用PAQ权重对注意力头和层级进行软聚合，消除平均头聚合所带来的噪声，显著提升剪枝效果；并引入几何衰减预算分配，实现单一FLOPs预算下的自动分组和逐层压缩。

**🔧 技术方法**

技术包括：PAQ的互信息/不确定度计算、基于PAQ的softmax加权注意力聚合、层级分组与预算分配、预填阶段视觉标记筛选与KV缓存压缩、与现有剪枝方法的对比实验。

**📊 数据集**

在18个视觉语言基准（如LLaVA、InternVL、Qwen‑VL、DeepSeek‑VL、MME、RefCOCO、SQA、GQA等）以及短视频评测（MVBench、MLVU）上进行验证。

**📈 对比分析**

与FastV、SparseVLM、PDrop、AutoPrune、VisionZip、VisPruner等基线相比，HAP在5.6%标记保留时保持99.1%原始性能，FLOPs下降至0.89T（比AutoPrune低~20%），KV缓存降低至20%，推理速度提升2.82×，在多模态基准上均达或超越最强基线。

**⚠️ 局限性**

局限性包括：仅在预填阶段剪枝，未对解码阶段做进一步压缩；PAQ仅基于现有文本提示的注意力统计，缺乏因果验证；实验仅覆盖静态图像和短视频，未评估流式或超长多模态序列的效果。

---

## 227. RecurSE: Bounded Recursive Self-Evaluation for LLM Rubric Judges

**arXiv ID:** 2608.24231 | [PDF](https://arxiv.org/pdf/2608.24231v1)

**作者:** Kaiyuan Liu `[一作]` (Zhejiang University), Jieping Ye `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种闭环自评机制——Recursive Self‑Evaluation（RSI），让LLM评判模型在不使用外部黄金标签或奖励模型的情况下，利用自身生成的评判和审计来进行强化学习；

**💡 创新点**

创新点包括（1）通过界面解耦消除评判与审计之间的 token 复制短路；（2）设计 Pairwise Advantage Validity（PAV）监控器实现无监督的早停；（3）同步 judge 与 checker 的权重，形成自我进化的闭环；（4）在多种规模与架构（9B‑27B）上验证方法的通用性；

**🔧 技术方法**

使用的技术包括：基于组相对优势的策略梯度强化学习；过程审计（checker）仅产生标量奖励；同步更新权重以实现 co‑evolution；PAV 作为验证监控；FSDP 与 vLLM 进行高效训练；

**📊 数据集**

数据集方面：训练使用 RubricHub（从 16 模型生成的合成回答），验证使用 100 条人工核对的严格规则；外部迁移评测使用 HealthBench、RubricBench、CheckEval‑Summ、ProfBench 等；

**📈 对比分析**

与基线（未强化学习的评判器）相比，RSI 在 Qwen3.5‑9B 上 rule‑accuracy 提升 12.9 点，Gemma‑4‑E4B‑it 提升 5.2 点，Qwen3.6‑27B 提升 3.9 点；在所有迁移基准上均有正向提升；PAV 早停策略能更稳健地捕捉最佳停止点，避免仅追求验证准确率导致的迁移退化；

**⚠️ 局限性**

局限性：1）RSI 的改进是有界的，需要人工验证集来决定停止点；2）方法依赖规则化评估，面对无规则或高度主观的任务效果不明；3）虽然已在 27B 规模验证，但更大模型或多语言场景的可扩展性尚待评估；4）若界面解耦不彻底，仍可能产生自奖励短路；

---

## 228. Agents of ViTAL: Ethics Missions -- A Narrative-Centered Learning Environment with a Co-Designed Conversational Agent for Middle School AI Ethics

**arXiv ID:** 2608.23580 | [PDF](https://arxiv.org/pdf/2608.23580v1)

**作者:** Sarah Burriss `[一作]` (Vanderbilt University), Ole Molvig `[通讯]` (Vanderbilt University)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5016807864)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一个基于浏览器的叙事式学习环境，帮助中学生协作评估是否采用AI课堂反馈工具，并通过嵌入的对话代理EthicsBot辅助伦理推理。

**💡 创新点**

①将伦理推理嵌入叙事式协作决策流程，②让学生共同设计对话代理的行为与安全约束，③通过排名挑战实现群体共识与伦理权衡。

**🔧 技术方法**

采用Web前端技术搭建交互界面，利用脚本化提示和正在研发的LLM驱动的对话代理，结合共享排名板和即时群聊。

**📊 数据集**

未使用公开数据集，而是通过游戏内收集的虚拟证据（如新闻剪报、情景决策）和学生反馈来训练与评估对话代理。

**📈 对比分析**

在四个九年级班级（共80名学生）中进行试点，测量学生参与度、伦理论证深度和决策质量；结果显示学生积极投入且能形成基于证据的伦理判断，未与其他系统做量化对比。

**⚠️ 局限性**

局限性包括：1）对话代理仍主要基于脚本，LLM版本尚未全面验证；2）缺乏大规模跨学科评估；3）可能存在学生过度依赖代理或信息误差（hallucination）导致推理偏差。

---

## 229. Wontopos Tablet 2: Measuring Multilingual and Multimodal Memory Retrieval Without Lexical Matching

**arXiv ID:** 2608.23920 | [PDF](https://arxiv.org/pdf/2608.23920v1)

**作者:** Sunwoo Kim `[一作]` `[通讯]` (Wontopos), Sunwoo Kim (Wontopos)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对一款生产环境中的记忆检索引擎进行量化评估，主要在两大维度：文本检索基准（LongMemEval‑S 与 BEAM‑1M）和无文本照片的跨语言检索。

**💡 创新点**

创新点在于：①构建了一种不依赖词汇匹配、语言模型或关键字评分的检索路径；②通过对 re‑ask 预算、读取模型和上下文扩展等超参的系统性消融，揭示其对性能的具体贡献；③提供了对无文字照片跨语言检索的首个量化基准，并展示了低资源语言和多语言混合存储对检索的影响。

**🔧 技术方法**

技术上主要使用：①自研的稠密向量检索引擎；②多轮检索（re‑ask）机制；③多语言多模态对齐（文本/图像）以及 BM25 词汇检索作为对照；④语言模型（如 Claude Opus 5、GPT‑5.6‑sol）用于答案生成与判分；⑤统计分析与置信区间计算。

**📊 数据集**

数据集包括：LongMemEval‑S（500 问题）、BEAM‑1M（700 问题，覆盖 2.2M 条记忆）、Crossmodal‑3600（300 张无标注照片 + 14 语言的自然语言描述）以及自制的 30 张多语句子-图像匹配基准。

**📈 对比分析**

比较方法：对同一记忆引擎在相同配置下与不同读取模型、re‑ask 预算以及上下文扩展的组合进行多跑实验，给出 95% 置信区间；在跨语言检索中，将系统与 BM25 以及两种公开的多模态检索基线（English‑CLIP 与 Multilingual‑CLIP）做对照。性能方面：在 LongMemEval‑S 上取得 95.7%（置信区间 93.4–97.1%），在 BEAM‑1M 上取得 67.5%（置信区间 64.8–70.2%）；在跨语言照片检索上，系统实现 91.4% recall@5，BM25 在无文字条件下为 0%。

**⚠️ 局限性**

局限性包括：①评估受读者模型与 re‑ask 预算的显著影响，导致不同基准间结果不易直接比较；②在低资源语言（如 Swahili、Telugu）表现显著下降；③多语言混合存储中附加英文标题会导致跨语言检索下降；④系统在对短查询的韩语处理上存在缺陷；⑤部分数值曾因记录错误或脚本缺陷而产生误差，需要进一步校验；⑥未公开检索架构细节，导致复现性受限。

---

## 230. A unified dynamical modeling framework for cruise control and adaptive cruise control

**arXiv ID:** 2608.23827 | [PDF](https://arxiv.org/pdf/2608.23827v1)

**作者:** Mingfeng Shang `[一作]` (Rochester Institute of Technology), Shian Wang `[通讯]` (University of Kansas)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一种连续融合巡航控制（CC）与自适应巡航控制（ACC）的统一动力学模型，并通过该模型分析等值状态与波纹稳定性，设计了最优切换阈值以提升交通流吞吐量与安全性。

**💡 创新点**

主要创新在于：①不修改ACC控制器，而是通过sigmoid权重函数实现CC与ACC的平滑过渡；②将等值状态、波纹稳定性与安全约束统一为对权重因子（𝜄）的上限与下限，从而推导最优切换阈值；③阐明了吞吐量、安全与波纹稳定性之间的可调权衡。

**🔧 技术方法**

使用了微分方程的车辆跟随模型、sigmoid平滑过渡、线性化与拉普拉斯变换、波纹稳定性判据、解析等值时间头距与吞吐量的表达式、数值仿真与参数灵敏度分析。

**📊 数据集**

以现场收集的车辆轨迹数据校准OVRV模型参数，并在仿真中采用一段实测的周期性速度曲线作为前车扰动。

**📈 对比分析**

通过对10辆车编队的仿真，分别比较纯ACC、纯CC与统一模型（吞吐优先、安保优先、商业阈值）三种切换策略；使用平均速度变化（ASV）和吞吐量作为指标。结果表明：吞吐优先设计将ASV降低39.7%，吞吐提升58.6%；安保优先降低34.7% ASV，吞吐提升16.4%；商业阈值表现最差。

**⚠️ 局限性**

局限性包括：仅考虑同类车辆编队，未涉及异构或混合自驾交通；缺乏现场实验验证；安全约束采用简化的RSS模型，可能未覆盖所有实际情况；模型对车辆参数假设较为理想，推广性待进一步研究。

---

## 231. TrustDABench: Benchmarking Reliability and Robustness of LLMs for Structured Data Analysis

**arXiv ID:** 2608.24145 | [PDF](https://arxiv.org/pdf/2608.24145v1)

**作者:** Boshen Shi `[一作]` (China Mobile Jiutian Artificial Intelligence Technology Co Ltd), Junlan Feng `[通讯]` (China Mobile Jiutian Artificial Intelligence Technology Co Ltd)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一个评估基准，检验LLM在结构化数据分析中是否能在缺乏证据时拒绝回答，并在语义保持的表格扰动下保持正确性。

**💡 创新点**

将可靠性与鲁棒性统一为“证据支持路径”视角，设计19种扰动算子并构建2340个人工验证实例，系统评估LLM的拒绝与不变性能力。

**🔧 技术方法**

采用Agentic-LLM生成框架（Kimi-K3、Claude-haiku等）对原始任务施加扰动算子，自动验证后人工审核；评估时使用LLM judge判定拒绝类型。

**📊 数据集**

基于AIDABench-QA和DABench两个现有结构化数据分析基准作为原始任务来源。

**📈 对比分析**

对八个主流LLM进行评估，计算可靠性平均MRS（最高24.21%）和鲁棒性ASR（最低9.10%）；模型在可靠性与鲁棒性上表现不一致，整体仍远低于理想水平。

**⚠️ 局限性**

仍无法有效识别证据冲突、缺失信息时的拒绝；鲁棒性受表结构变化影响大；基准样本分布不均，评估侧重单一问答形式。

---

## 232. Keep-or-Drop? Adaptive Tokenizer for Compact Video Representation

**arXiv ID:** 2608.24293 | [PDF](https://arxiv.org/pdf/2608.24293v1)

**作者:** Yeonkyeong Lee `[一作]` (Kakao Corp.), Donghoon Lee `[通讯]` (Kakao Corp.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了 KATok，一种基于 Transformer 的自适应 VAE，用来在视频的连续潜在空间中动态保留或丢弃 token，以获得更紧凑且表达力强的表示，并结合流匹配扩散模型实现高质量视频生成。

**💡 创新点**

核心创新在于引入可微的 keep‑or‑drop token selector（基于 Gumbel‑Softmax 与稀疏正则化），以及两种解决稀疏 token 引发的空间位置信息失配的方法——联合内容与位置生成与级联 mask‑prior 生成，从而实现了无需预设 token 数量的自适应压缩与高效生成。

**🔧 技术方法**

使用了 Transformer 编码器/解码器、3D Rotary Position Embedding、可学习的查询 token、稀疏正则化、Gumbel‑Softmax 软采样、流匹配扩散（flow‑matching）以及 Video‑LPIPS 等视频感知损失。

**📊 数据集**

在 Panda‑70M 语料上训练 VAE；在 SkyTimelapse、UCF‑101 与 Kinetics‑600 数据集上评估生成模型；同时使用了 Omni‑Tokenizer‑VAE 与 ElasticTok‑KL 作为基准。

**📈 对比分析**

与固定长度 VAE（Omni‑Tokenizer‑VAE）相比，KATok 在相同分辨率下仅用 366–1554 个 token 仍能获得更高的 PSNR（最高 33.23）和更低的 rFVD（最低 5.12）；在扩散生成任务中，采用级联 mask‑prior 方案的 gFVD 下降至 61.53，训练速度提升约 6.9×、推理速度提升 3.2×，同时大幅减少 11× 的 token 数。

**⚠️ 局限性**

仍需对 mask‑prior 与联合生成的噪声调度进行手动调参；在极端静态或大尺寸场景下的 token 选择可能欠缺细粒度控制；整体架构依赖大量 GPU 资源，模型规模仍较大。

---

## 233. Beyond Static and Linear: What Attention Constraints Best Fit Human Reading Times?

**arXiv ID:** 2608.23818 | [PDF](https://arxiv.org/pdf/2608.23818v1)

**作者:** Lanni Bu `[一作]` (Georgetown University), Ethan Gotlieb Wilcox `[通讯]` (Georgetown University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了在Transformer语言模型中加入多种注意力记忆约束（内容基础和距离基础），并评估其对人类阅读时间预测和语法能力的影响。

**💡 创新点**

创新点在于对多种记忆约束、不同模型规模和语料进行全面比较，区分静态与动态记忆课程，并发现内容基础约束（如Forgetting Transformer）在认知拟合上优于距离基础约束，同时揭示动态课程与语法能力之间的解耦现象。

**🔧 技术方法**

采用OPT基础Transformer，改进注意力机制（ALiBi、n-gram滑动窗口、Forgetting Transformer、Stick-breaking），构建动态课程（Less-to-More/More-to-Less），使用Delta Log Likelihood回归评估阅读时间拟合，并在BLiMP上评估语法能力。

**📊 数据集**

实验使用了BabyLM-10M、BabyLM-100M、Pile 2B三种训练语料；阅读时间数据来自Brown、Natural Stories、UCL、Dundee、GECO、Provo六个英语数据集；语法评测采用BLiMP基准。

**📈 对比分析**

通过比较静态模型的Delta Log Likelihood和BLiMP准确率，发现内容基础约束在大多数配置下实现最高阅读时间拟合；动态模型在BLiMP上优于静态，但在阅读时间拟合上低于静态，More-to-Less Forgetting Transformer在认知拟合上表现最佳，挑战Less-is-More假设。

**⚠️ 局限性**

限制包括：仅使用单一随机种子、仅评估英语数据、模型缺乏多模态输入与社交环境、静态模型仅评估最终检查点、动态与静态BLiMP难以直接对比、记忆机制为理论代理而非直接实现。

---

## 234. CoDrift: Compositional Drifting for Offline Reinforcement Learning

**arXiv ID:** 2608.23939 | [PDF](https://arxiv.org/pdf/2608.23939v1)

**作者:** Xiewei Ni `[一作]` (Xi'an Jiaotong University), Xiangyu Xu `[通讯]` (Xi'an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种离线强化学习生成式策略框架 CoDrift，通过把行为匹配、边缘行为匹配和值最大化三种目标统一为动作空间位移场并加性组合，完成单步生成。

**💡 创新点**

创新点在于：①将多目标学习转换为动作位移场的形式，实现目标的可加组合；②利用漂移模型实现单步生成而不需多步推理；③同时在条件与边缘层面约束行为分布。

**🔧 技术方法**

核心技术包括漂移模型（drifting models）、基于核的均值移位动作位移场、条件与边缘行为漂移场、值梯度场以及单步噪声生成器。

**📊 数据集**

在OGBench 73个连续控制任务和D4RL 18个任务上进行离线和离线到在线的评估。

**📈 对比分析**

与11种代表性基线（高斯、扩散、流）以及离线到在线基线进行比较，CoDrift 在所有基准上平均排名最佳，尤其在OGBench上表现突出。

**⚠️ 局限性**

局限性：漂移场估计方差较大，极高维动作空间下的稳定性尚待验证，且对边缘漂移的有效性仍需更多多任务数据支持。

---

## 235. RAGSentinel: Certifiable Geometric Consensus for Robust Retrieval-Augmented Generation

**arXiv ID:** 2608.23965 | [PDF](https://arxiv.org/pdf/2608.23965v1)

**作者:** Yueyang Quan `[一作]` (University of North Texas), Zhuqing Liu `[通讯]` (University of North Texas)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练、无标签的后检索防御方法（RAGSentinel），通过对检索文档的隐藏状态偏移进行几何异常检测，过滤知识库中的恶意注入文档，从而提升检索增强生成（RAG）系统的事实性与安全性。

**💡 创新点**

创新点：①利用检索文档在代理编码器中的隐藏状态偏移作为未被攻击者可优化的几何信号；②在去除公共主题方向后，以几何中值（geometric median）构造鲁棒共识并结合局部一致性，形成自适应阈值过滤；③在无标签、无训练、黑盒环境下实现对恶意文档的有效识别，并给出可证明的过滤安全性。

**🔧 技术方法**

技术手段：代理编码器（如BGE-M3）提取隐藏状态，动态子空间选择、范数裁剪、主题方向去除；几何中值求解、局部邻域一致性计算；自适应过滤半径；最终将过滤后的文档拼接为 RAG 上下文并一次性调用黑盒LLM。

**📊 数据集**

数据集：Natural Questions、HotpotQA、MS‑MARCO；使用三种LLM（Mistral‑7B、Llama‑3.1‑8B、Qwen‑2.5‑7B）和多种检索器（Contriever 等）进行评估。

**📈 对比分析**

对比方法包括 Vanilla RAG、RobustRAG、InstructRAG、AstuteRAG、TrustRAG、CrAM。实验显示，在三种攻击（PoisonedRAG、PIA、AD）下，RAGSentinel 的攻击成功率（ASR）普遍低于 0.1，同时准确率（ACC）与 Vanilla RAG 相近，优于所有基线；在自适应攻击和混合攻击场景下亦保持最佳性能。

**⚠️ 局限性**

局限性：①需要满足“诚实多数”假设（恶意文档数量小于检索集的一半）；②假设攻击者无法访问或查询代理编码器，若能直接获得隐藏状态仍可能绕过防御；③对高比例恶意注入的鲁棒性尚未充分验证；④对低资源语言或结构不良检索结果的几何特征支持不明。

---

## 236. The Handoff Tax: Continuing Non-Native Trajectories in LLM Agents

**arXiv ID:** 2608.24358 | [PDF](https://arxiv.org/pdf/2608.24358v1)

**作者:** Roy Ganz `[一作]` (AWS), Ron Litman `[通讯]` (AWS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统研究了编码代理在长周期任务中从低成本/低能力模型（LC）切换到高成本/高能力模型（HC）以及反向切换时的中途轨迹交接对成本与质量的影响。

**💡 创新点**

发现了“手动税”（handoff tax）——不同切换方向和交接接口会显著改变质量恢复比例与成本占比；指出轨迹信息的保留或压缩对升级与降级产生相反效果，并提出手动交接应视为单独的推理优化问题。

**🔧 技术方法**

使用了多种手动交接接口（Raw、Compact_pre、Compact_suf、Traj-drop），并通过对话与工具调用、代码编辑等多轮交互模拟长周期编码代理；对成本采用供应商定价模型，对质量采用通过SWE‑bench验证的通过/失败率。

**📊 数据集**

主要数据集为SWE‑bench Verified（500个真实GitHub issue），并在LiC与BrowseComp等任务中探讨信息动态对手动交接的影响。

**📈 对比分析**

通过匹配切换子集、计算归一化的质量恢复率（QRec）与成本保留率（CSRet）进行对比；结果显示Raw升级只能恢复不到一半的质量且成本高；降级则能在保持较低成本的同时保留大部分质量；压缩与轨迹丢弃在不同方向下有显著差异。

**⚠️ 局限性**

局限性包括仅评估两对模型（Claude Haiku/Opus 与 GPT Luna/Sol）、仅在单一任务实例下进行一次运行、切换点固定且在难度分层中的样本量有限；成本评估依赖于特定供应商价格，且未覆盖多次交接或多模型路由策略。

---

## 237. How Do Professional Editors Evaluate the Editing Quality of AI-Generated Cinematic Video Ads?

**arXiv ID:** 2608.24329 | [PDF](https://arxiv.org/pdf/2608.24329v1)

**作者:** Po-Ming Law `[一作]` (Adaptive Machines, Inc), Arpit Narechania `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对社交媒体短视频广告的分析，构建AI生成电影化广告的两步生成管线，并收集专业编辑对70条AI生成广告的评价，提出六维度评估框架。

**💡 创新点**

首次从专业编辑的反馈中提炼出六维度电影化广告编辑质量评价框架，兼顾情节、视听协调、视觉构图、连贯性、品牌信息与节奏等方面。

**🔧 技术方法**

采用LLM生成镜头计划，随后使用视频生成模型Seedance 2.0渲染短片，并通过人工评审与访谈获取专业批评。

**📊 数据集**

基于Meta与TikTok广告库收集124条广告并扩展至99条电影化广告，随后为35个真实品牌生成70条AI广告。

**📈 对比分析**

通过六位专业编辑对70条广告进行开放式批评与访谈，量化每维度出现频率，验证框架在诊断编辑缺陷方面的有效性，但未进行算法性能对比。

**⚠️ 局限性**

仅评估基于LLM+Seedance 2.0管线的广告，对其他生成方法的通用性有限，且缺乏自动评估器与量化指标验证。

---

## 238. Ethical LLM-Assisted Research: A Framework for Responsible Delegation, Verification, and Epistemic Value

**arXiv ID:** 2608.23644 | [PDF](https://arxiv.org/pdf/2608.23644v1)

**作者:** Kalin Stoyanov `[一作]` (University of Chemical Technology and Metallurgy), Kalin Stoyanov `[通讯]` (University of Chemical Technology and Metallurgy)

**通讯引用:** 127 | [OpenAlex ID](https://openalex.org/A5079974556)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并阐述了一套关于 LLM 辅助科研的责任与验证框架，定义了 O(g)、V(g)、R(g)、M(g)、E(g) 五个核心属性，并引入“认知审计”概念。

**💡 创新点**

创新点在于将生成与验证、归因与责任、以及机器生成的知识与人类可接受性明确区分，主张验证与人类可接受性是决定科学合法性的关键，而非机器参与度；并首次将审计视角与科学决策结合。

**🔧 技术方法**

主要使用了形式化理论与概念模型构建（如函数映射、属性定义和递归验证循环），并借鉴了分布式认知、责任追溯和审计的文献。

**📊 数据集**

未使用任何具体数据集，论文完全基于理论推导和文献综述。

**📈 对比分析**

论文不涉及实验比较，未给出性能指标，而是通过与现有学术规范（如 ICMJE、WAME 等）对照，说明框架的可行性和一致性。

**⚠️ 局限性**

局限性在于缺乏经验验证与定量评估，框架仍属概念性；未提供可操作的测量工具，也未在实际科研流程中测试其有效性。

---

## 239. LG-GER: Language-Guided Group Emotion Recognition via Multimodal Evidence Distillation

**arXiv ID:** 2608.23880 | [PDF](https://arxiv.org/pdf/2608.23880v1)

**作者:** Ahmed Shehab Khan `[一作]` (University of South Carolina), Yan Tong `[通讯]` (University of South Carolina)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出LG-GER框架，利用多模态大语言模型生成空间证据并将其蒸馏到单一视觉‑语言模型上，实现检测‑free的组情感识别。

**💡 创新点**

通过离线MLLM生成的区域级情感证据和四种空间监督损失，实现无检测推理下高效、可扩展的组情感识别。

**🔧 技术方法**

多模态大语言模型（Gemini 3.0 Flash）、视觉‑语言模型SigLIP 2、情感适配器、梯度归一化损失平衡、区域文本对齐、空间情感与置信度回归等技术。

**📊 数据集**

GroupEmoW与GAF 3.0两个组情感基准。

**📈 对比分析**

与需检测和多流融合的SOTA方法对比，在GAF 3.0验证集上取得84.08%（最高），在GroupEmoW测试集上92.39%（第二高），同时保持检测‑free推理。

**⚠️ 局限性**

对区域级注释质量依赖MLLM生成的可靠性，且在极度混合情感或弱信号场景下仍可能过度关注单一区域。

---

## 240. From Preferences to Principles: Rubric-Based Alignment for Grounded Knowledge Answers

**arXiv ID:** 2608.23812 | [PDF](https://arxiv.org/pdf/2608.23812v1)

**作者:** Aman Saini `[一作]` (Apple), Wanming Chen `[通讯]` (Apple)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于检索证据生成的多维度评判规则框架，用于在检索增强生成（RAG）场景下的奖励学习。

**💡 创新点**

创新点在于将评判规则与检索证据耦合，并将规则拆分为四个质量维度（完整性、结构、真实性、安全性），实现细粒度、可控的监督。

**🔧 技术方法**

使用大型语言模型生成评判规则、外部LLM判定器进行条目级评分、GRPO强化学习以及检索增强生成技术。

**📊 数据集**

训练数据为4.7k条合成知识查询，评估集包括Search Arena、RAGBench和FACTS Grounding Benchmark。

**📈 对比分析**

与instruction‑tuned基线及多种消融方案比较，检索‑多维评判模型在Composition、Grounding、Instruction‑Following三轴平均提升约6.5%，在Search Arena获得最大显著提升。

**⚠️ 局限性**

局限包括仅在英文单一领域实验、评估依赖LLM判定、单次实验结果、参考答案单一来源、假设检索质量完好等。

---

## 241. FireRedAudio: A General-Purpose Audio Language Model with Decoupled Continuous Representations for Understanding and Generation

**arXiv ID:** 2608.24168 | [PDF](https://arxiv.org/pdf/2608.24168v1)

**作者:** Junjie Li `[一作]` (Xiaohongshu), Yao Hu `[通讯]` (Xiaohongshu)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了FireRedAudio，一种统一的音频语言模型，支持语音识别、音频理解、零射TTS、指令式TTS以及语义与声学语音编辑，且能处理长达一小时的音频；

**💡 创新点**

核心创新在于引入了两条解耦的连续输入表示通道：一条用于理解的音频编码器，另一条用于生成的可重构RedAE编码器，二者共享同一个9B参数的LLM而不混合表示；

**🔧 技术方法**

利用Whisper预训练的音频编码器、RedAE确定性自编码器、Patch Encoder、流匹配的DiT扩散模型以及分阶段的多任务自监督训练；

**📊 数据集**

训练集涵盖多语言ASR、音频理解（MMAU、MMSU）、零射TTS（Seed-TTS-Eval）、指令式TTS（InstructTTSEval）、语音编辑（Ming-Freeform-Audio-Edit）以及长时音频（5–50分钟），共计约591B多模态token；

**📈 对比分析**

在各基准上表现优异：音频理解ACC最高，ASR在LibriSpeech、FLEURS English和FLEURS‑102上取得最低错误率，零射TTS内容准确率排名首位（平均1.20%），指令式TTS在所有六项指标上均优于竞品，语音编辑在语义与声学指标上均高于Ming-UniAudio-Edit；

**⚠️ 局限性**

限制在于：对长时音频的处理仍以第二级时间戳为准，且零射TTS的说话人相似度低于专用模型；模型规模较大（9B参数）且训练成本高；未来需进一步提升说话人多样性与对极端长录音的鲁棒性。

---

## 242. When Seeing Is Not Enough: Benchmarking Interactive Visual Grounding in LVLMs

**arXiv ID:** 2608.23978 | [PDF](https://arxiv.org/pdf/2608.23978v1)

**作者:** Zhengxiang Wang `[一作]` (Stony Brook University), Owen Rambow `[通讯]` (Stony Brook University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

引入一个可控评估框架，用来衡量大型视觉语言模型在交互式视觉定位中的表现，系统化地调节先验目标信息量，并通过四种交互协议进行实验。

**💡 创新点**

①将传统单轮视觉定位拆解为多轮交互；②设计四个协议（Full、Optional、Underspecified、Question-driven）以区分视觉匹配、信息获取与合成；③提供开源数据与评测工具，展示当前LVLM在主动问答驱动定位上的不足。

**🔧 技术方法**

采用大型视觉语言模型（如GPT‑5.4、Gemini‑3.1‑Pro 等）和 GPT‑5.4 作为模拟导演，使用基于 JSON 的对话交互，计算准确率、问题率、校准度以及自我修正行为。

**📊 数据集**

四个对象级视觉数据集（狗、篮子、抽象 Tangram 等）以及六个面部细粒度数据集，用于人机对话模拟和后续验证。

**📈 对比分析**

通过与任务级人类基准及随机基准对比，记录准确率、提问次数、校准误差；结果显示 LVLM 在所有协议下均低于人类，尤其在“Question-driven”协议下最为突出；大型模型表现优于小型模型，但整体仍远落后。

**⚠️ 局限性**

①导演使用模拟（GPT‑5.4）而非真人，可能偏离真实交互；②协议虽然区分视觉匹配与信息获取，但未拆解为更细粒度子任务；③仅测试有限的低复杂度对象场景，未涵盖复杂场景、空间关系、开放式视觉搜索。

---

## 243. Calibration-Preserving Pruning: Compression as a Reliability Contract

**arXiv ID:** 2608.23744 | [PDF](https://arxiv.org/pdf/2608.23744v1)

**作者:** Ibne Farabi Shihab `[一作]` (Iowa State University), Anuj Sharma `[通讯]` (Iowa State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于校准保持剪枝（Calibration‑Preserving Pruning，CPP）的技术，使剪枝后模型在保持独立分割校准下的分布式边际覆盖率的同时，进一步压缩有效的预测集；

**💡 创新点**

创新点在于在剪枝评分中加入非符合性梯度敏感度，以在保持覆盖率的前提下提升预测集的效率；

**🔧 技术方法**

使用分割式合规预测、梯度敏感度剪枝、阈值感知与候选标签的 saliency 计算，以及离散的 prune‑val‑conf‑test 四个互不重叠的数据拆分；

**📊 数据集**

主要实验数据集包括 Qwen2.5‑1.5B 在 AG News、TREC、DBpedia‑14、Banking77、CLINC150 的三种稀疏率（30%、50%、70%），并做 RoBERTa‑base 与 Llama‑3‑8B 的转移验证；

**📈 对比分析**

与传统的 magnitude、Wanda、SparseGPT 等基线对比，CPP 在保持 90% 边际覆盖率的前提下平均可缩小 10–20% 的预测集大小，并在大标签任务中提升准确率；阈值感知版本的 CPP 在效率上更优，但计算成本更高；

**⚠️ 局限性**

局限性包括：仅适用于固定标签分类任务（非开放生成）；需在剪枝前获取标记样本和梯度；离散分割校准要求严格的独立性；理论给出的是充分但非必要的稳定性条件；未评估稀疏化对推理时延与能耗的影响。

---

## 244. Markerless Pose Estimation for Resistance Training Technique Assessment

**arXiv ID:** 2608.24384 | [PDF](https://arxiv.org/pdf/2608.24384v1)

**作者:** Joseph Turner `[一作]` (University of Bristol), Nawid Keshtmand `[通讯]` (University of Bristol)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发并评估了基于BlazePose的无标记姿势估计框架，用普通视频提取深蹲、卧推、硬拉等力量训练动作的关节角度轨迹并进行技术评估。

**💡 创新点**

①提出可在非实验室环境下使用的完整无标记姿势估计与关节角度提取流程；②构建包含参考动作的力量训练数据集；③系统研究摄像机视角对2D关节角度估计的影响，并给出基于RMSE的相似度评分。

**🔧 技术方法**

BlazePose 2D人体姿势估计、向量公式计算关节角度、线性插值时间归一化、RMSE与相似度评分、基于局部最小值的多次重复分段分析。

**📊 数据集**

结合两个公开Kaggle力量训练视频数据集并筛选，最终得到89段可用视频；参考动作来自公开教学视频；覆盖深蹲、卧推、硬拉。

**📈 对比分析**

将每个重复动作的膝角和躯干角度轨迹与参考轨迹做RMSE比较，再转化为0–100相似度分数；平均RMSE约17.6°，平均相似度约65/100；侧面视角可用帧率最高（99.7%），视角差异显著影响性能。

**⚠️ 局限性**

2D角度估计高度依赖摄像机视角与遮挡，非侧面视角会产生显著误差；数据集规模小，缺乏3D重建与更广泛环境验证；RMSE未对关键动作点加权，评估指标仍需改进。

---

## 245. SIREN-Bench: Behavior-Driven Generation and Evaluation of Emergency-Vehicle Interactions

**arXiv ID:** 2608.24094 | [PDF](https://arxiv.org/pdf/2608.24094v1)

**作者:** Yicheng Zhu `[一作]` (Rochester Institute of Technology), Zilin Bian `[通讯]` (Rochester Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了SIREN平台，利用SUMO与CARLA的共模仿技术生成行为驱动的紧急车辆（EMV）与周围车辆的交互，并以此构建SIREN‑Bench‑v1基准，涵盖三项评测任务：3D目标检测、轨迹预测与视觉‑语言风险理解。

**💡 创新点**

创新点包括①基于EMV特权与民众响应的行为层配置，而非预设情景；②通过事件驱动的控制转移实现SUMO–CARLA的同步闭环仿真；③提出了多模板、参数化的交互模型，揭示不同EMV行为对感知、预测与风险推理的行为依赖性；④首次在同一平台上对三类任务进行统一比较。

**🔧 技术方法**

技术实现主要涉及SUMO网络仿真、CARLA连续控制、跨仿真同步与控制交接、64束LiDAR+多摄像头感知、现成轨迹预测器（CSP、STDAN、BAT、EMP-M/D、DeMo等）、LiDAR基3D检测器（PointPillars、SECOND、VoxelNeXt、TransFusion‑L）以及视觉‑语言模型（Blaifa‑InternVL3.5‑8B、Gemma3‑12B、LLaVA‑Llama3‑8B、MiniCPM‑V‑4.5‑8B、Qwen3.5‑9B）。

**📊 数据集**

数据集为SIREN‑Bench‑v1自生成的七个交互模板（每模板一条基准轨迹），共105个前视摄像头视频（15条/模板），配合LiDAR与IMU等传感器的同步记录；未使用公开的nuScenes、Waymo等数据集，而是完全由平台生成。

**📈 对比分析**

评估方法包括：轨迹预测以ADE/FDE衡量（5 s预测窗口）；3D检测以nuScenes协议下的mAP/NDS评估；风险理解以分类准确率、精确率与F1衡量。实验结果显示：①学习预测器在多数模板上未能超越常数速度基准；②检测性能在交通清除模板最低，红灯穿行最高；③视觉‑语言模型对风险级别的区分存在强偏差，几乎无法同时对Normal、Near‑Miss与Collision获得非零F1。

**⚠️ 局限性**

局限性包括：仅在单一地图、单一EMV级别下生成；每模板仅有一次轨迹生成，缺乏随机化与多种实现；检测未使用EMV专属标签；未对不同级别、地图与随机种子进行分离分析；缺少对感知与预测模型的微调或迁移学习，导致迁移效果不佳。

---

## 246. An Echo Chamber of One: Should AI Psychosis Be a Distinct Clinical Entity?

**arXiv ID:** 2608.23937 | [PDF](https://arxiv.org/pdf/2608.23937v1)

**作者:** Joshua Au Yeung `[一作]` (King's College London), Richard Dobson `[通讯]` (King's College London)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文探讨了大型语言模型（LLM）聊天机器人可能诱发的精神病症状（即“AI精神病”）并讨论是否应将其归类为独立诊断；

**💡 创新点**

创新点在于首次系统梳理LLM的sycophancy与anthropomorphism组合导致的心理破坏机制，并提出跨学科协作的监测与监管框架；

**🔧 技术方法**

主要运用LLM安全评估基准（如SycEval、EchoBench、Psychosis-bench）及理论分析，结合案例与媒体报道；

**📊 数据集**

并未使用专门的数据集，而是依赖公开媒体报道、个案报告和早期观察数据；

**📈 对比分析**

因缺乏实验数据，文中未进行量化对比，主要以理论讨论与实践建议为主；

**⚠️ 局限性**

局限在于缺乏系统证据、因果推断不足、可能存在媒体与案例偏倚、以及对“AI精神病”标签的社会标签化风险。

---

## 247. When LLMs Slow Down: How Environmental Impacts Mediate University Students' LLM Usage

**arXiv ID:** 2608.23968 | [PDF](https://arxiv.org/pdf/2608.23968v1)

**作者:** Hyeonwook Kim `[一作]` (Georgia Institute of Technology), Josiah Hester `[通讯]` (Georgia Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套基于延迟的生态反馈界面，在LLM交互中可视化碳排放与响应延迟的权衡，并在大学计算机伦理课程中对89名本科生进行实验评估；

**💡 创新点**

首次将延迟-碳权衡以可视化生态反馈呈现给用户，结合个人环保规范探究对低碳使用模式的接受度，并在教育情境下验证其可持续性与学习效果；

**🔧 技术方法**

采用Gemma‑2‑27b‑it模型、服务器无服务器API，设计5种eco modes（通过GPU数量、批处理大小、可再生能源调度）实现延迟控制；使用实时可视化（延迟、碳、累计节省、真实世界等）展示；采用GLMM进行定量分析；

**📊 数据集**

主要使用实验参与者的交互数据（89名CS本科生）以及通过模型仿真生成的不同输入/输出长度下的延迟与碳数值；未使用公开数据集；

**📈 对比分析**

通过混合效应逻辑回归比较不同eco mode下的选择比例，并测评用户满意度；结果显示eco mode 1接受度约45%，随延迟升高到mode 5下降至<5%；碳减排约12 g CO₂/查询，年均可达14.6 kg CO₂；

**⚠️ 局限性**

局限包括样本仅为单校CS本科生，缺乏多样性；高延迟下界面可用性下降；碳模型简化，缺乏真实数据中心细节；未与其他可持续LLM系统或更细粒度的延迟控制进行对比；对个人环保规范的分析深度有限。

---

## 248. ViSculpt: Visual-Centric Agentic Geometry Editing

**arXiv ID:** 2608.24169 | [PDF](https://arxiv.org/pdf/2608.24169v1)

**作者:** Bo Pang `[一作]` (Peking University), Peng-Shuai Wang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个训练‑free 的多代理系统，利用大型语言模型直接在 Blender GUI 中执行局部 3D 网格编辑，从而实现自然语言驱动的、在原始网格上做局部修改且保留其身份的编辑流程。

**💡 创新点**

创新点在于：① 将复杂的雕刻操作抽象为仅三种鼠标轨迹原语（Smear、Drag、Draw），大幅简化可执行空间；② 通过视觉‑语言模型与 QuadLoc 方案实现精确的目标定位；③ 在三代理框架（Planner、Action、Reflection）中使用即时视觉反馈与经验库检索来实现无训练的自动化编辑。

**🔧 技术方法**

核心技术包括：大型语言模型（Gemini 3 Flash/Pro）用于规划与翻译；视觉‑语言模型与 Grounded SAM2 进行图像分割；Z‑Image 等 2D 生成模型辅助 Draw 原语；QuadLoc（递归四分定位）提高定位精度；经验库检索增强生成（RAG）支持迁移学习；以及 Blender Python API 与 GUI 屏幕‑空间控制实现实际操作。

**📊 数据集**

使用自制的 20 个编辑任务基准集，每个任务都有专业艺术家手工完成的前后网格对，此外还利用公开的 VLM、SAM 及 Blender 文档等资料作为模型和参考库。

**📈 对比分析**

通过与人类艺术家的盲测对比（48 位参与者，包含 39 非专家和 9 专家），系统平均得分 7.53，略高于人类 7.20，表现接近；与脚本化方法 Blender MCP、生成模型（Hunyuan2.0、Rodin）进行案例级对比，展示了在局部保留与即时视觉反馈方面的优势；编辑耗时约 2–8 分钟，主要受模型推理延迟影响。

**⚠️ 局限性**

局限性包括：对大型基础模型高度依赖导致较高延迟；对高度抽象或模糊指令的理解仍有限；基于 2D 截图的感知难以处理严重自遮挡或内部几何；缺少拓扑级别的原语（如布尔运算、挖洞等）；以及仅通过 2D 视觉判断，无法完全捕捉非流形或自交等几何错误。

---

## 249. Too much of a good thing -- when knowledge distillation promotes overfitting, and how to avoid it

**arXiv ID:** 2608.23752 | [PDF](https://arxiv.org/pdf/2608.23752v1)

**作者:** Irene Trigueros-Lorca `[一作]` (Andalusian Research Institute in Data Science and Computational Intelligence), Daniel Molina `[通讯]` (University of Granada)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了在EfficientNet-B0教师网络与同构学生网络之间进行块级知识蒸馏的方法，并在经典与细粒度、数据稀缺的数据集上系统评估不同蒸馏点的数量与位置对模型性能的影响。

**💡 创新点**

创新点包括：①提出可自由配置的块级蒸馏框架，允许在学生网络的任意块上进行蒸馏；②通过注意力图、CKA与Grad-CAM等可解释性技术深入分析知识在学生网络中的传播机制，发现后半块信息最关键；③基于解释性分析设计了高效的“Blocks3456”学生结构，在保持性能的同时显著降低蒸馏点数量。

**🔧 技术方法**

主要技术手段包括：知识蒸馏（MSE损失）、块级蒸馏策略、教师/学生细调策略（FT、FS、FTS）、数据增强与数据裁减实验、以及可解释性分析工具（注意力图、CKA、Grad-CAM）。

**📊 数据集**

实验使用了11个图像分类数据集，7个经典数据集（CIFAR10/100、EMNIST、FashionMNIST、Food101、MNIST、SVHN）和4个细粒度、数据稀缺数据集（CUB200、ISIC、OxfordPets、StanfordCars）。

**📈 对比分析**

对比方法包括：仅用教师输出蒸馏（end-block）、所有块蒸馏（all-block）以及多种中间配置；通过多次随机种子实验、Wilcoxon和Friedman统计检验评估显著性。结果表明：在数据丰富的经典数据集上，end-block与all-block几乎无显著差异；在细粒度或样本稀缺情况下，all-block或Blocks3456显著提升精度，单一额外蒸馏点即可弥补大部分性能缺口；FS策略在无教师细调时仍保持竞争力。

**⚠️ 局限性**

局限性包括：①实验仅基于EfficientNet-B0教师，尚未验证更大/更小教师的普适性；②可解释性分析仅使用三种方法，未深入探讨更细粒度的知识结构；③对计算成本的讨论仅在单GPU环境下进行，未覆盖大规模训练时的资源瓶颈；④在极端数据稀缺（如少于10%数据）下，仍有性能提升空间。

---

## 250. More Rejective, Not More Discriminative: The Unit of Verification in Pre-Execution LLM Oversight

**arXiv ID:** 2608.23941 | [PDF](https://arxiv.org/pdf/2608.23941v1)

**作者:** Yuchen Han `[一作]` (University of Science and Technology of China), Wuyang Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究预执行监督中LLM监控器的验证单位长度对捕获错误与误拒的影响，提出并评估了可控的“双缀前缀”（twin-prefix）测量框架。

**💡 创新点**

①提供了只改变验证窗口长度、同时给出干净对照的实验设计；②使用预注册、配对评估和Youden指数对不同长度进行客观比较；③发现长窗口虽然提高误拒率，却并未提升错误区分度，最佳长度往往为1或2。

**🔧 技术方法**

使用LLM指令式验证（如Llama‑3.3‑70B、Qwen2.5‑14B/72B），Youden指标、AUC、预注册实验流程、配对评估、层级窗口、干预观察缺失等技术。

**📊 数据集**

两套人工生成的数据集：①零售工具代理（200条基准，31条清洁前缀） ②AppWorld多应用API程序（200条基准，190可实现）——每条基准注入单一、非自适应写入错误，分为三种严重度。

**📈 对比分析**

通过在每条基准上生成不同长度（L=1,2,3,5,8）的双缀前缀，评估固定长度、投票、路由、标签无关自报等策略，测量捕获率、误拒率、Youden指标和AUC。结果显示所有评审者和两域均在L=1或2取得最高Youden，长窗口导致误拒率显著上升（Youden降至≈0.035），且无标签策略无法超过校准的短窗口。

**⚠️ 局限性**

限制：①注入错误仅为单一、非自适应写入，未涵盖更复杂或自适应错误；②验证窗口上限为8，实验仅覆盖两域和零射门LLM；③清洁前缀样本量有限，尤其零售域；④观察缺失是主要失效因素，但替代预测仍未完全解决。

---

## 251. A Theory of Speciation in Generative Diffusion Models on Compact Riemannian Manifolds

**arXiv ID:** 2608.23798 | [PDF](https://arxiv.org/pdf/2608.23798v1)

**作者:** Alessio Marta `[一作]` (University of Milan), Paola Causin `[通讯]` (University of Milan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了在紧致黎曼流形上生成扩散模型的本征分化理论，描述了分化为临界点分岔的过程。

**💡 创新点**

创新点包括将分化视为分布临界点的分岔，利用谱热核、Poincaré–Hopf 与 Morse 理论揭示几何拓扑对分化的约束，并给出泛型 A_2 典型形态与几何模式。

**🔧 技术方法**

采用热核谱展开、黎曼曲线扩散的SDE与Fokker–Planck方程、奇异性理论、数值模拟以及神经网络的隐式分数学习。

**📊 数据集**

使用球面上的 von Mises–Fisher 分布、CelebA 以及 NASA 火灾数据等多模态数据集进行验证。

**📈 对比分析**

通过与解析解对比及对学习分数误差的稳定性分析，实验表明分化时间与位置的偏移仅与误差在临界方向上的投影有关，性能与理论预测吻合。

**⚠️ 局限性**

局限性在于对高维或非紧致流形的理论尚不完善，对分岔核维数大于一的情况缺乏完整分析，以及对真实数据中分数逼近误差的精确估计不足。

---

## 252. 'Ghaib in Translation' aka Unseen Harm: Measuring Cross-Script Safety Inconsistency with 'Missed-in-Urdu' Scores in LLM Hate Speech Detection

**arXiv ID:** 2608.24191 | [PDF](https://arxiv.org/pdf/2608.24191v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 253. Partial Optimal Transport on the Circle for All Transported Masses in O(N log N)

**arXiv ID:** 2608.23910 | [PDF](https://arxiv.org/pdf/2608.23910v1)

**作者:** Soheil Kolouri `[一作]` `[通讯]` (Vanderbilt University), Soheil Kolouri (Vanderbilt University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在球面上研究并实现了部分最优传输（partial optimal transport）方法，并将其应用于形状匹配与鲁棒拟合任务。

**💡 创新点**

提出了一种利用球面几何投影和部分OT来解决遮挡与杂散（clutter）问题的创新框架，并给出了可以一次性计算完整传输曲线的高效算法。

**🔧 技术方法**

使用了部分OT理论、球面投影与归一化、优先队列/堆排序等数据结构，以及Sinkhorn等对比方法。

**📊 数据集**

采用了MPEG‑7 CE‑Shape‑1数据集（共1400个形状轮廓），对形状进行正态角度分布处理。

**📈 对比分析**

与传统相关匹配器、最近邻匹配器以及平衡OT/Sinkhorn方法进行比较。实验结果显示，在遮挡和杂散场景下，所提部分OT在成功率、检索精度（bullseye分数）和对应F1值上明显优于基线，并且计算成本显著低于Sinkhorn（约20倍）。

**⚠️ 局限性**

局限性包括：仅针对二维球面数据，参数ρ的选择仍需经验或学习；在极端遮挡/杂散情况下仍可能出现误配；未验证对更高维或非球面数据的泛化能力。

---

## 254. A Scenario-Based Evaluation of CRQC+AI Vulnerability Spectrum for TLS 1.3 Cryptographic Dependencies

**arXiv ID:** 2608.23785 | [PDF](https://arxiv.org/pdf/2608.23785v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 255. Constraint-Guided Enterprise Data Mapping with Large Language Models

**arXiv ID:** 2608.24218 | [PDF](https://arxiv.org/pdf/2608.24218v1)

**作者:** Sebastian Monka `[一作]` (Bosch Center for Artificial Intelligence), Lavdim Halilaj `[通讯]` (Bosch Center for Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 Constraint-guided Mapping (CGM) 方法，通过可执行硬约束先行过滤候选集，再用软约束加权的神经排序和受限 LLM 去歧义，实现企业级实体对齐；

**💡 创新点**

创新点在于将硬约束作为假设空间前置过滤器，采用级联放宽保证非空候选集，同时保持可审计、低成本，实验表明约束门是提升性能的关键；

**🔧 技术方法**

使用语义嵌入+软约束加权、可执行硬约束逻辑推理、LLM bounded disambiguation、cascade relaxation 与自动约束发现与验证等技术；

**📊 数据集**

使用合成结构化记录基准、公开 Valentine 列表以及七个汽车制造商（Make A–G）的企业对齐数据集；

**📈 对比分析**

与 Magneto、COMA、Cupid、SimilarityFlooding、Jaccard 等 SOTA 方法在 synthetic、Valentine 与企业数据上对比，CGM 在 synthetic 约束门 F1 从 0.08 提升至 0.66，企业宏观 F1 达 0.70，成本比无约束 LLM 低 28 倍，专家工时下降约 7 倍；

**⚠️ 局限性**

局限在于合成实验不完全代表真实分布，专家效率评估样本有限，Valentine 仅列级评估，方法对约束覆盖度高度敏感，需要手动或自动约束发现。

---

## 256. Steering Recurrent Reasoners at Inference Time with Readout Feedback

**arXiv ID:** 2608.24136 | [PDF](https://arxiv.org/pdf/2608.24136v1)

**作者:** Shunsuke Kamiya `[一作]` (University of Tokyo), Yutaka Matsuo `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了读取反馈（Readout Feedback，RoFB）方法，在推理时通过模型自身的输出概率来引导循环推理模型的潜在状态，从而无需重新训练即可提升性能。

**💡 创新点**

创新点在于将中间读出概率转换为闭环反馈，形成基于类别相似度的吸引/排斥耦合，主动调控潜在动态，使推理过程更快收敛。

**🔧 技术方法**

使用循环推理模型（AKOrN、ItrSA++、TRM）、自定义耦合项、门控机制、软最大读出层以及在推理步骤中注入反馈的数学框架。

**📊 数据集**

在两类逻辑推理基准数据集上评估：Sudoku Extreme（20,000道题）和Maze Hard（1,000个迷宫）。

**📈 对比分析**

与传统延长迭代步数、增加轨迹数或置信度投票等方法比较，RoFB在四个模型-任务组合上实现了显著提升（最高提升约6.4%），且在相同或更低的计算成本下达到或超越传统方法的性能。

**⚠️ 局限性**

局限性包括仅在小型循环推理模型和特定谜题数据集上验证，且对大型语言模型的可迁移性尚不确定；部分模型-任务组合（ItrSA++/Sudoku、TRM/Maze）未见显著改进，且需要进一步诊断哪些情形下RoFB有效。

---

## 257. MARS: Multi-Specialist LLM Relay System for Competitive Programming

**arXiv ID:** 2608.23918 | [PDF](https://arxiv.org/pdf/2608.23918v1)

**作者:** Andrei Mikhailov `[一作]` (MIRAI), Alsu Sagirova `[通讯]` (AXXX)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于提示的多智能体协作框架MARS（Multi-Agent Relay of Specialized LLMs），通过主题专门化的LLM代理配合检索增强生成（RAG）来解决竞赛编程任务；

**💡 创新点**

创新点在于：①将每位代理专门化为单一算法主题并用检索对其进行知识补全；②通过自评选择任务匹配的专家团队，并在每一步利用公共测试执行结果进行即时反馈、修正或交接；③采用极简的提示与局部判断机制，避免复杂的多阶段规划与搜索；

**🔧 技术方法**

使用技术包括：大规模语言模型（Gemma 4、Qwen3.5‑27B、GPT‑5.4‑mini）、检索增强生成（检索cp‑algorithms语料库）、C++17/ Python代码生成与执行沙箱、公共测试反馈循环、自动化代码整理与基础设施修复。

**📊 数据集**

数据集主要为CodeContests 165道 Codeforces 竞赛题目，使用cp‑algorithms的算法理论语料库作为检索索引；实验也在Python版本和不同模型上重复。

**📈 对比分析**

与基线对比：Direct、Single‑RAG、Parallel ensemble、Base relay、CodeSIM。MARS在Gemma 4上取得0.624±0.006的通过率，较Direct提升14.4%，较Single‑RAG提升9.5%；与CodeSIM相比，MARS在通过率上仅差约0.107，且3.3×更低的墙钟时间和更小的token方差。其它后端和语言迁移实验亦保持相同趋势。

**⚠️ 局限性**

局限性：评估仅覆盖165道题目、三种模型与两种语言；仅使用Codeforces标签，未测试其他语义标签或更广泛的算法领域；Python版复用了相同检索语料，需为其他语言构建专属语料；局部门禁仅检查当前轮次公共测试的回退，未能捕捉隐藏测试错误；所有代码均需沙箱执行，无法直接在真实环境中运行。

---

## 258. Beyond Information Seeking: Severity-Aware Question Supervision for Proactive Medical Dialogue

**arXiv ID:** 2608.24521 | [PDF](https://arxiv.org/pdf/2608.24521v1)

**作者:** Chenxuan Li `[一作]` (Peking University), Peixing Wan `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种基于预期严重性风险的主动医学问答选择方法；

**💡 创新点**

将诊断结果的严重性信息与问答价值结合，并在选择时对未观测答案进行边缘化，形成“Expected-Severity-Risk”评价；

**🔧 技术方法**

使用前向推断的病理答案分布、后验诊断概率、梯度下降与LoRA微调的Qwen3-4B语言模型；

**📊 数据集**

DDxPlus公开数据集中的9种疾病子集，包含5000训练、800验证、1000测试病例；

**📈 对比分析**

与传统信息增益、期望0/1风险等基线对比，实验显示高严重性诊断漏检率下降29.5%，准确率提升约2个百分点，且额外提问量仅增0.14个；

**⚠️ 局限性**

仍受限于固定的病症集合和预先定义的严重性等级，且在极短问答预算下信息增益方法表现更好。

---

## 259. Benchmarking LLM Judges for Voice-Agent Evaluation: Reliability, Calibration, and Human Oversight

**arXiv ID:** 2608.24314 | [PDF](https://arxiv.org/pdf/2608.24314v1)

**作者:** Anupam Purwar `[一作]` (Sprinklr AI), Kritika Srivastava `[通讯]` (Sprinklr AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

使用GPT-4.1与GPT-5作为评判者，对电信与零售两类语音代理的对话进行大规模评估，并与人工评估结果进行对比。

**💡 创新点**

提出了三种提示配置（p0、p1、p2）和多维度评估指标，证明LLM评判在大多数维度下具有稳定的相对趋势，可作为可扩展的评估前置工具，同时指出安全与恢复类指标仍需人工干预。

**🔧 技术方法**

核心技术为LLM-as-a-Judge框架、提示工程（三种配置）、指标自定义与校准、统计对比（比值、相关系数、误差分析）以及混合评估管道。

**📊 数据集**

使用242条真实语音对话，分别来自零售（120条）和电信（122条）领域，按每个配置各40/41条进行评估。

**📈 对比分析**

通过对比人工与LLM的比值、相关系数与误差分布，发现LLM在TE、CR、ARGA等指标上与人工一致；在IAS、SR、RTC等安全与恢复指标上差距显著。GPT-5在目标完成度指标上更贴近人工，GPT-4.1在某些安全相关评判上更为保守；总体而言，LLM在大多数指标上表现出稳定的相对趋势，校准后可达到可接受的评估精度。

**⚠️ 局限性**

局限性包括：安全类指标（IAS、SR）与恢复类指标（RTC）对LLM的可靠性不足；不同领域对校准的敏感度不同；缺少基于音频时序的评估，如停顿、打断等语音特有行为；LLM评判在多轮恢复追踪时易低估，需进一步改进或人工复核。

---

## 260. Contextual Embedding Evidence for Main--Light Verb Distinctions in Urdu

**arXiv ID:** 2608.23645 | [PDF](https://arxiv.org/pdf/2608.23645v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 261. Shortcut Before Circuit: Document Statistics Time In-Context Conflict Resolution

**arXiv ID:** 2608.24460 | [PDF](https://arxiv.org/pdf/2608.24460v1)

**作者:** Yijun Liao `[一作]`, Fanwei Liang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在一种带有两条轴（重复度 R_old 与更新到查询距离 ΔD）的合成赋值语言上训练 26M 参数的 Transformer，使用最小因果编辑将稀有度与最近性两条同义规则的计数相互切换，随后通过读取模型在编辑前后的对数概率差异来判别模型实际依赖的提示，探讨在同义规则共存时模型的机制归因是否可行。

**💡 创新点**

创新点在于提出了“同义构造（aliasing construction）”——通过让两条规则在训练分布上始终一致，从而让目标函数对其不作区分，进而揭示何时数据能决定机制归因；同时引入了逃逸时间（escape timing）作为可复制的判定指标，并将读取器门控设为“电路形成”而非简单准确率，以避免在电路尚未出现时得到错误的归因。

**🔧 技术方法**

主要技术包括：基于预训练的 Decoder‑Only Transformer（8 层、d_model=512）；使用 per‑head RMS 归一化与可学习的维度增益；利用最小因果编辑（intervention）在输入层切换计数；读取器以 log‑odds 差值 Δ 衡量稀有度与最近性的偏向；通过 loss‑derivative 峰值定位逃逸点；对 75 个网格单元在 3 个种子下进行统计学分析。

**📊 数据集**

使用的数据集为完全自定义的合成赋值语言：文档为一系列 (实体, 属性, 值) 语句加上查询，实体词表 200、属性 8、值 512，文档长度在 45–55 句之间，重复度 R_old 取 {3,5,8,12,16}，更新到查询距离 ΔD 在指定区间内均匀采样，整个训练流无穷无尽。

**📈 对比分析**

比较方法：在 75 个 (R_old, ΔD) 组合上进行 3 个随机种子实验，记录最终准确率、读取器的符号比例、逃逸时间等指标。所有模型最终准确率均 ≥ 0.999，表现极佳；然而读取器在同义规则共存的单元上表现出显著的种子波动，表明机制归因不唯一；逃逸时间随 R_old 单调递增，验证了数据统计对机制出现时机的决定作用。

**⚠️ 局限性**

局限性包括：同义构造导致目标函数在两条规则上不区分，因而机制归因仅在特定条件下可行；实验仅在单一模型结构（26M 参数 Transformer）和单一任务上进行，缺乏对更大模型或自然语言数据的泛化；种子波动导致单个实验难以给出可靠归因；编辑后仍可能存在未观察到的隐藏机制；最后，数据统计与优化过程的交互使得读数的解释依赖于实验细节，限制了结论的普适性。

---

## 262. Mechanistic Circuit Identification for Controllable Data Generation

**arXiv ID:** 2608.24065 | [PDF](https://arxiv.org/pdf/2608.24065v1)

**作者:** Nakyung Lee `[一作]` (Seoul National University), Jungwoo Lee `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过机制解释发现模型内部电路与数据质量（learnability、challenge、alignment）之间的因果关联，利用这些电路进行可控数据生成，并引入阶段感知调度（SAMS）优化训练效果。

**💡 创新点**

将机制解释转为可操作的电路级控制，实现对合成数据的多维度（学习性、挑战性、对齐性）精准调节；结合阶段性调度进一步提升下游性能。

**🔧 技术方法**

使用EAP-IG等机制解释技术挖掘电路；利用AUM、EL2N、GradAlign等训练动力学指标；采用激活添加与注意力调节的电路驱动生成；实现SAMS调度算法。

**📊 数据集**

主要在SciQ多项选择问答基准上进行实验，并用ARC‑Easy评估跨域鲁棒性。

**📈 对比分析**

与提示式生成、随机混合以及统一混合等基线对比，SAMS在SciQ上精度提升至85.8%、ECE下降；在ARC‑Easy上同样表现优于基线，证明了电路驱动与阶段调度的优势。

**⚠️ 局限性**

实验仅覆盖单一模型（Qwen2.5-1.5B）和任务，对其他模型/任务的迁移性未验证；电路挖掘和干预需要高昂计算资源，且对不同架构的通用性有限。

---

## 263. VisCache: Visual KV Cache Pruning for Efficient Vision Large Language Model Inference

**arXiv ID:** 2608.24063 | [PDF](https://arxiv.org/pdf/2608.24063v1)

**作者:** Lyuke Wang `[一作]` (Shenzhen International Center for Industrial and Applied Mathematics), Guangxu Zhu `[通讯]` (Shenzhen International Center for Industrial and Applied Mathematics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 VisCache，一个训练无关、可插拔的两阶段框架，用于在长视频 VLLM 推理中显著压缩视觉 KV 缓存。

**💡 创新点**

创新点在于先用轻量 VLM 结合 MMR 进行 prompt‑aware 关键帧筛选，再通过层级化 parabolic 预算与异构 key/value 归并的 PruneKV，实现更细粒度、结构化且无需训练的 KV 缓存压缩。

**🔧 技术方法**

技术包括 CLIP‑style 轻量 VLM 与 MMR 关键帧筛选、基于多层注意力的 parabolic 预算分配、异构 key/value 更新（key 丢弃、value 归并）以及在 Qwen2.5‑VL‑3B/32B 等预训练 VLLM 上的直接应用。

**📊 数据集**

使用了 ActCap、DREAM1K、NExTQA、ActQA、EgoSchema 和 MVBench 等视频理解与视觉问答基准数据集。

**📈 对比分析**

与 PyramidKV、FastV、PDrop、Q‑Frame 等基线相比，VisCache 在 40%、28% 和 19% 的 KV 维持率下，FLOPs 下降至 9%/15%/6%，速度提升至 1.93×/2.35×，且在 28% 维持率下平均准确率高于全缓存并超越所有基线。

**⚠️ 局限性**

局限在于轻量 scout 与主模型的视觉表征可能不完全一致，导致关键帧筛选误差；PruneKV 需要在预填阶段存储所有层注意力，产生额外内存开销。

---

## 264. Autonomous Mathematical Discovery in an Open-World Multi-Agent Environment

**arXiv ID:** 2608.23691 | [PDF](https://arxiv.org/pdf/2608.23691v1)

**作者:** Stephen Chung `[一作]`, William J. Wesley `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个名为 Station 的开放式多代理科研环境，让不同模型的 AI 研究员在无中央协调器的情况下自主选择研究方向、实验、协作并累积共享文献，完成 12 个 AlphaEvolve 题目和 2 个案例研究的数学发现。

**💡 创新点**

创新点在于：① 引入完全自治的多代理生态，让 AI 研究员像真实科研团队一样自我规划和协作；② 通过 Archive Room、Question Room 等模块积累可检索的科学知识；③ 设计了 Holiday、Stagnation、Supervisor 等机制以促进探索、避免陷阱，并与 AlphaEvolve 等传统单代理搜索进行系统对比。

**🔧 技术方法**

核心技术包括：多房间交互式环境、GPT‑5.5、Claude Opus 4.8、Gemini 3.1 Pro 等大语言模型代理、自动化评测器、归档论文提交与审核流程、问答与邮件通信、随机监督、周期性假期与停滞协议。

**📊 数据集**

使用的数据集为 AlphaEvolve 的 12 个优化/构造问题（如有限域 Kakeya、Erdős 重叠、kissing 数等）以及 Book Ramsey 数与 Jacobian Conjecture 两个独立案例，全部由 Station 自行探索完成。

**📈 对比分析**

通过与 AlphaEvolve 在同一问题上的多实例对比，发现 Station 在 5/12 题目上产生了相对文献的新结果、3/12 超越 AlphaEvolve、2/12 与其相当、2/12 略逊；具体突破包括无限族 Kakeya、604 点 kissing 配置、Erdős 最小重叠下界提升、Jacob 反例重构等。

**⚠️ 局限性**

主要局限包括：缺乏专家直觉导致探索偏差；模型间研究品味相似导致多样性不足；上下文学习有限，无法充分吸收累积知识；存在吸引陷阱，易被无意义的细节或重复实验分散注意力。

---

## 265. Gradient-extrapolation-based distributed mirror descent algorithm for multi-cluster aggregative games

**arXiv ID:** 2608.24183 | [PDF](https://arxiv.org/pdf/2608.24183v1)

**作者:** Rui Zhu `[一作]` (Nankai University), Zengqiang Chen `[通讯]` (Nankai University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种分布式镜像下降（Mirror Descent）与梯度外推（Gradient Extrapolation）相结合的算法，用于在多簇聚合博弈（MAG）中寻找纳什均衡，能够在时间变化的内部和外部网络环境下实现协同与竞争的平衡；

**💡 创新点**

创新点包括：①采用Bregman散度而非欧氏距离，提升算法对约束集几何的适配性；②在对偶空间引入梯度外推，提高收敛速度；③在多簇聚合博弈框架下完成分布式实现；④在受限强单调性假设下给出O(1/k)收敛率分析；

**🔧 技术方法**

技术手段涵盖：镜像下降（Mirror Descent）算法、Bregman散度、梯度外推、变分不等式、分布式一致性/聚合、Lipschitz连续性与强单调性证明、时间变化网络分析；

**📊 数据集**

实验使用了合成的电力需求响应数据，设置3个社区（各3、4、5户），随机生成基准负荷、约束集参数R=15、权重矩阵Q等；

**📈 对比分析**

与文献中的欧氏分布式算法和半去中心化聚合算法进行了对比，评估指标为相对误差‖x_k−x*‖/‖x_0−x*‖；实验显示本文算法在相同精度下收敛更快、计算时间更短；

**⚠️ 局限性**

局限性：需满足受限强单调性与双向随机矩阵假设；对极限通信情况（如稀疏或单向网络）未做充分验证；仅给出O(1/k)收敛率，尚未探索更快收敛策略；

---

## 266. Auditing the Synthetic Memoir: Measuring Scene-Level Confabulation in LLM-Generated Autobiography Against the Documented Record of the Life It Describes

**arXiv ID:** 2608.23640 | [PDF](https://arxiv.org/pdf/2608.23640v1)

**作者:** Heather Renze `[一作]` `[通讯]` (Serenze Global), Heather Renze (Serenze Global)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对一名个体在 2025 年使用 LLM 生成的 366 天日记进行逐天情节级别的真实性审计，并量化其验证失败率。

**💡 创新点**

首次将个人自传文本的情节真实性量化，发布了可复用的四分类审计工具，并展示“基础驱动漂移”与“检索基础纠正”对可信度的影响。

**🔧 技术方法**

使用 LLM 审计员（Claude）手动记录的四分类评分表、正则关键词屏蔽检索、Wilson 区间、Kappa 等统计方法进行分析。

**📊 数据集**

利用 366 天的日记正文、独立的自传原稿、记事账本、公开稿件、音乐目录、知识库与网络检查等七类真实记录语料进行验证。

**📈 对比分析**

通过与两套独立盲审样本的二元与四级一致率比较，验证 96.7% 的失败率稳健；在 60 天样本中，未检索时保持 100% 失败，检索后下降至 83.3%，显著提升。

**⚠️ 局限性**

局限性包括单一受试者、LLM 自循环标注、四级标签可靠性仅为公平至中等、验证失败并不等同于造假、未进行大规模人类终审。

---

## 267. Every Layer Counts: An Exponential $L_2$ Depth Hierarchy for ReLU Networks

**arXiv ID:** 2608.23877 | [PDF](https://arxiv.org/pdf/2608.23877v1)

**作者:** Itay Safran `[一作]` `[通讯]` (Ben Gurion University), Itay Safran (Ben Gurion University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

证明了ReLU网络在所有相邻深度之间存在指数层级，给出了可计算的目标函数以及对应的宽度下界。

**💡 创新点**

首次给出在深度≥3的固定深度之间的指数分离，并且对任意权重不做限制，构建了完整的相邻深度指数层级。

**🔧 技术方法**

采用递归复制、微扰布尔立方体、一次隐藏层的复制器和仿射一般位置等技巧来构造目标函数。

**📊 数据集**

使用的是由几何形状构成的人工概率分布（如并置的立方体），不依赖真实数据集。

**📈 对比分析**

通过理论上给出的宽度下界（指数 vs 多项式）来比较，表明深度提升可实现指数幅度的宽度节约。

**⚠️ 局限性**

构造的分离在分布支持半径指数增长，未能在多项式半径下实现；对Vardi–Shamir等现有阈值电路下界的转移仍未得到。

---

## 268. Native Multimodal Representation Learning for Click-Through Rate Prediction in E-Commerce Scenarios

**arXiv ID:** 2608.24091 | [PDF](https://arxiv.org/pdf/2608.24091v1)

**作者:** Chao Yi `[一作]`, Han Zhu `[通讯]` (Taobao & Tmall Group of Alibaba)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对电商场景的CTR预测，提出一种Native Multimodal Representation Learning框架，即先用注释模型从CTR数据中挖掘高质量多模态可解释样本，再对预训练的SCL编码器进行微调。

**💡 创新点**

创新点在于发现端到端联合训练对强大预训练编码器无效，提出Mine-Then-Train方法将监督信息在数据层分离，利用多模态可解释性筛选样本以提升CTR性能。

**🔧 技术方法**

技术包括CLIP ViT‑B/16+SCL预训练、多模态注释模型（残差调优+SID编码器）、Triplet Margin Loss、SCL对比损失、GAUC评估及在线A/B测试。

**📊 数据集**

使用的数据集为阿里巴巴天猫广告平台的CTR日志（约1.9 B样本，84 M用户，88 M商品）以及轻量级Taobao‑MM基准。

**📈 对比分析**

与传统两阶段冻结编码器和直接加入注释分数的方法对比，Mine‑Then‑Train在离线AUC/GAUC提升0.22%/0.11%，在线A/B测试CTR提升1.5%、RPM提升0.5%。

**⚠️ 局限性**

局限在于仍依赖大规模业务日志挖掘、阈值敏感、对非多模态噪声处理仍不完美，且需要较多工程成本与验证。

---

## 269. Names Can Hurt: Spotting Slopsquatting Risks Caused by Package Name Hallucinations in Local Coding LLMs

**arXiv ID:** 2608.23897 | [PDF](https://arxiv.org/pdf/2608.23897v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 270. Do LLMs Understand Limit Order Book Dynamics?

**arXiv ID:** 2608.23706 | [PDF](https://arxiv.org/pdf/2608.23706v1)

**作者:** Junxiao Chen `[一作]` (Columbia University), Paul Glasserman `[通讯]` (Columbia University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

训练Transformer LLM以生成合法的限价订单簿（LOB）事件序列，并评估其隐式世界模型；

**💡 创新点**

提出kernel‑level与history‑level总变差测度以及回归测试，用以量化LLM对LOB动力学的误解导致的偏差与伪预测；

**🔧 技术方法**

采用GPT风格的解码器Transformer（12层/768维或48层/1600维）并使用交叉熵训练；

**📊 数据集**

使用合成LOB数据集，分别构造随机游走（RW‑S、RW‑L）与最短路径（SP）序列，规模从50万段到1000万段；

**📈 对比分析**

与经验核基线对比：LLM在合法性测试中表现优异，但压缩分数低，多步TV和回归系数显示显著偏差，表明LLM未能正确捕捉Markov状态；

**⚠️ 局限性**

局限性在于仅使用小规模、无真实可预测性的合成LOB，扩展至更大状态空间需要更多训练数据，且无法直接验证在真实市场中的表现。

---

## 271. Poisoning Agentic Alpha: Adversarial Vulnerabilities Across Roles and Architectures in Multi-Agent Trading Systems

**arXiv ID:** 2608.24069 | [PDF](https://arxiv.org/pdf/2608.24069v1)

**作者:** CheolWon Na `[一作]` (Sungkyunkwan University), Jee-Hyong Lee `[通讯]` (Sungkyunkwan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对LLM驱动的多智能体交易系统进行系统性对抗实验，评估不同角色与通信拓扑下的攻击效果。

**💡 创新点**

首次在金融领域量化不同攻击入口（数据污染、提示注入、说服、目标劫持、越狱）以及四种通信拓扑对安全性的影响，并提出Adversarial Signal Preservation Score用于解释。

**🔧 技术方法**

采用大型语言模型（GPT‑4.1、Qwen‑3‑235B）、定制攻击脚本、代理式交易管线，使用市场行情、新闻、社交文本为输入。

**📊 数据集**

使用五只大型公司股票（BTC‑USD, MSFT, NVDA, TSLA, AAPL）与 Alpha Vantage、yfinance、Reddit 等公开数据，在2026 Q1后期数据上进行回测。

**📈 对比分析**

通过攻击成功率（ASR）与财务影响（期末资本变化）比较，结果显示无一拓扑本质安全，终端风险管理被攻击最易导致高 ASR，但单一代理往往比多智能体更易被攻击，且架构与模型、攻击方向交互影响大。

**⚠️ 局限性**

实验受限于单一 backbone、仅覆盖大盘资产、缺乏交易成本/滑点考虑，且只在模拟环境验证，未检验实际交易场景。

---

## 272. ZODIAC: Zero-shot Octree-based Diffusion for Anatomical Completion

**arXiv ID:** 2608.24422 | [PDF](https://arxiv.org/pdf/2608.24422v1)

**作者:** Miruna-Alexandra Gafencu `[一作]` (Technical University of Munich), Nassir Navab `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

利用零样本扩散模型完成脊柱超声的全三维形状重建

**💡 创新点**

提出无配对数据的零样本生成式先验与混合完成机制，克服了传统方法对人工模拟遮挡的依赖

**🔧 技术方法**

采用自适应八叉树表示、VAE编码、两阶段扩散网络（低分辨率、高清细化）以及每步重噪混合完成

**📊 数据集**

训练使用VerSe20与TotalSegmentator完整脊柱网格，评估用两个解剖模型与Balgrist志愿者超声-CT数据集

**📈 对比分析**

与SITD、TP-ODIAC等基线对比，零样本方法在志愿者数据上HD95提升22%，CD提升70%，在大部分指标上接近或优于监督上限

**⚠️ 局限性**

仍在精细细节和F1分数上略逊，且对极端噪声或缺失程度极高的观测仍有限制

---

## 273. Rethinking Pre-Training and Augmentation for Zero-Shot Cross-City Object Detection

**arXiv ID:** 2608.24154 | [PDF](https://arxiv.org/pdf/2608.24154v1)

**作者:** Long Hoang Pham `[一作]` (Sungkyunkwan University), Jae Wook Jeon `[通讯]` (Sungkyunkwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对隐私受限、硬件受限的零样本跨城市目标检测任务，设计了基于多数据集预训练与Grayworld增广的模块化训练框架，并在RF-DETR上实现了RF-DETR-HR与RF-DETR-Grayworld两种高性能模型。

**💡 创新点**

创新点在于：① 引入类无关的对象性蒸馏，解耦车辆几何与语义分类；② 开发Grayworld颜色去耦增广，消除光照与传感器依赖的颜色捷径；③ 在16 GB GPU约束下，采用梯度累积、差异学习率与JIT优化，显著提升跨城市泛化能力。

**🔧 技术方法**

使用的技术包括：多数据集预训练（CoCo、Hafnia、TSBOW等）、类无关对象性蒸馏、Grayworld与CLAHE增强、RF-DETR transformer、梯度累积、差异学习率、TorchScript JIT、mAP评估。

**📊 数据集**

使用的数据集为：Hafnia（主数据）、TSBOW、TrafficCAM、FishEye8K、VisDrone、MOT20；以及在AIC2026-Track 6隐私平台上获得的隐藏训练/测试集。

**📈 对比分析**

与基线（COCO预训练RF-DETR-2XLarge）以及竞赛其他队伍比较，RF-DETR-HR以47.53 mAP夺得官方排行榜第一，RF-DETR-Grayworld以46.63 mAP位列第二，分别比第二名提升约4.7与3.8 mAP，显著提高小目标与低光照场景检测性能。

**⚠️ 局限性**

主要局限在于：① 方案专门针对Transformer与RF-DETR架构，缺乏对其他模型的通用验证；② 依赖特定的隐私平台与GPU约束，跨平台可迁移性有限；③ 仅在交通监控场景验证，跨域性能对非交通场景未知。

---

## 274. BenchBench-Protocol: Evaluating Real-World Wet-Lab Protocol Reasoning and Modification

**arXiv ID:** 2608.23898 | [PDF](https://arxiv.org/pdf/2608.23898v1)

**作者:** Aditya Sivakumar `[一作]` (Benchling), Nithin Parsan `[通讯]` (Benchling)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 BenchBench-Protocol 这个基准，收集了 149 个真实实验室协议修改任务，并为每个任务设计了加权评估 rubrics

**💡 创新点**

创新点在于：①从科学家实际在 Benchling 上修改协议的差异中自动生成任务，保证任务贴近真实工作；②使用专家评审保证任务和 rubrics 的科学严谨性；③引入加权 rubrics，量化评判模型在协议后续步骤影响上的完整性

**🔧 技术方法**

技术手段包括：自动比较原始与修改后协议生成差异；使用 GPT‑5.6 Terra 作为 LLM‑as‑a‑judge 对模型输出进行评分；对模型进行多次尝试、期望得分评估；多模型对比实验（Claude Opus 5、GPT‑5.6 系列、Gemini 3.6 Flash、Grok 4.5、Kimi K3 等）

**📊 数据集**

数据集为 96 条公开协议在 Benchling 上被科学家改写后产生的 149 条任务，覆盖蛋白化学、细胞培养、分子克隆、染色成像、测序等九个实验室生物学领域

**📈 对比分析**

比较方法：在单轮设置下，每模型尝试 10 次，使用加权 rubrics 计算归一化分数；对比单次得分与期望最高得分（α-0）。Claude Opus 5 最高得分 59.2%，其余模型介于 34.1%–47.1% 之间；期望得分提升在 19.6%–39.5% 的可用分数范围内

**⚠️ 局限性**

局限性：①仅评估单轮回答，未考虑多轮对话和澄清；②任务选取偏向难度，可能不代表日常实验调整；③评估使用的 LLM‑as‑a‑judge 本身也是待评估模型之一，可能存在偏差；④未给出实验室真实实验结果的直接验证；⑤token 使用与成本高，未覆盖不同部署场景

---

## 275. GAP-Prompt: Gated Adaptive Prompting for Efficient Continual Learning

**arXiv ID:** 2608.23782 | [PDF](https://arxiv.org/pdf/2608.23782v1)

**作者:** Trung-Anh Dang `[一作]` (Université d'Orléans), Vincent Nguyen `[通讯]` (Université d'Orléans)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了GAP-Prompt，一种基于实例条件门控的持续学习框架，能够在冻结预训练ViT主干的同时，通过实例级动态调节提示层、动态知识融合以及共享提示蒸馏来提升模型对新任务的适应性并降低遗忘。

**💡 创新点**

创新点包括：①实例条件门控（ICG）实现每张图像自适应的提示层激活；②动态知识融合（DKF）在实例级门控下实时聚合历史任务的提示，实现知识的灵活重用；③共享提示蒸馏（SPD）在前两层保持共享提示的语义一致性，进一步抑制遗忘。

**🔧 技术方法**

使用技术包括：预训练Vision Transformer（ViT）主干、前缀提示调优（prefix tuning）、Gumbel-sigmoid 软门控、余弦相似度匹配的任务键、动态温度衰减以及多任务交叉熵、匹配和蒸馏损失的联合优化。

**📊 数据集**

在CIFAR-100、ImageNet-R和CUB-200三个分类基准上进行实验，采用10任务分类增量学习设置。

**📈 对比分析**

与现有无提示、提示基线（L2P、DualPrompt、CODA-Prompt、EvoPrompt、RainbowPrompt等）以及多种预训练模型（ImageNet-1K、ImageNet-21K、iBOT-1K、DINO-1K）比较，GAP-Prompt在平均准确率上分别获得89.24%（CIFAR-100）、78.72%（ImageNet-R）和87.29%（CUB-200），显著超过对比方法，同时在遗忘率上保持最低（CIFAR-100 3.03%、ImageNet-R 3.12%、CUB-200 3.68%）。

**⚠️ 局限性**

局限性包括：1）在完全无记忆（无重放）设置下仍依赖预训练模型的先验知识，若预训练不匹配可能受限；2）实例级门控和知识融合虽然提升性能，但引入额外计算和存储开销；3）实验仅覆盖离线、类增量学习场景，未验证在线或少样本持续学习的鲁棒性。

---

## 276. DeepRepoQA: Code Repository Question Answering with Deep Agent Exploration

**arXiv ID:** 2608.24221 | [PDF](https://arxiv.org/pdf/2608.24221v1)

**作者:** Weihan Peng `[一作]` (Shanghai Jiao Tong University), Xiaodong Gu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DeepRepoQA，一个基于蒙特卡洛树搜索（MCTS）的代理框架，用于仓库级代码问答，支持多跳推理、结构化与语义检索以及有证据的答案生成。

**💡 创新点**

将 QA 视为规划问题，结合感知、规划、执行、评估四个子代理，使用 MCTS 引导全局搜索；引入结构化检索、语义检索与检索‑增广生成，形成迭代搜索‑验证循环，实现深层次多跳推理与可解释答案。

**🔧 技术方法**

蒙特卡洛树搜索、LLM 代理（Perception、Planning、Execution、Evaluation）、Tree‑sitter 语法树解析、代码嵌入语义检索、检索‑增广生成（RAG）、评估代理评分机制等技术。

**📊 数据集**

扩展版 SWE‑Bench（15 个 Python 仓库，720 QA 对），并对 30 条 Java QA 进行验证。

**📈 对比分析**

与直接提示、RAG（滑窗、函数块）、Agent‑based（SWE‑Agent、OpenHands）以及商业工具（Tongyi Lingma、Cursor）对比；在 GLM‑4.6、Kimi‑K2、Qwen3‑Coder‑480B‑A35B‑Instruct、GPT‑5.1 四大 LLM 上均实现 4–7% 的整体性能提升，特别是在 Correctness、Completeness、Reasoning 维度表现突出，逼近甚至超越部分商业工具。

**⚠️ 局限性**

仍受模型知识偏差、检索覆盖率限制；需要足够的搜索迭代，计算成本相对较高；对非 Python 语言的支持有限；评估依赖 LLM 判定，可能存在主观性；极大仓库或极长代码的可扩展性尚未充分验证。

---

## 277. Compressive Sensing - Introduction and Relations to Deep Learning

**arXiv ID:** 2608.24211 | [PDF](https://arxiv.org/pdf/2608.24211v1)

**作者:** Hung-Hsu Chou `[一作]` (University of Pittsburgh), Holger Rauhut `[通讯]` (Ludwig-Maximilians-Universität München)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

综述压缩感知（Compressive Sensing, CS）的基本理论与技术，并阐述其与深度学习的交叉点，包括：1）压缩感知的稀疏性、随机测量与ℓ1最小化的理论基础；2）利用压缩感知算法的逐步实现（如ISTA）作为可训练的“unrolled”神经网络；3）分析梯度下降（及其连续时间的梯度流）在过参数化线性网络中的隐式正则化，揭示其对稀疏解与低秩解的偏好；4）讨论低秩矩阵恢复、张量恢复与更一般的非线性网络的隐式正则化。

**💡 创新点**

创新点主要有：
- 将压缩感知算法逐步迭代视为神经网络层，构建可学习的 LISTA 等网络，并给出其泛化误差上界。
- 通过梯度流在多层对角线线性网络中的分析，证明在小初始化时，优化过程会趋向于ℓ1-最小化的稀疏解，从而给出梯度下降隐式正则化的第一批理论解释。
- 对低秩矩阵恢复问题，引入对称或可对角化测量矩阵的假设，证明梯度流能隐式偏向低秩解，解释深度网络训练中出现的“早期对齐”与“低秩化”现象。
- 提出将优化过程视为 Riemannian 梯度流的观点，为进一步研究隐式正则化提供了几何视角。

**🔧 技术方法**

使用的技术包括：
- 经典压缩感知工具：稀疏性概念、ℓ1/ℓ0最小化、基追踪（Basis Pursuit）、ISTA/FISTA、随机测量矩阵与 RIP、NSP。
- 机器学习与优化理论：可学习的阈值、梯度流与离散梯度下降、Bregman 距离、Rademacher 复杂度、泛化误差分析。
- 线性代数与概率论：随机矩阵理论、子高斯分布、矩阵不变性、平衡初始化、奇异值分解。
- 几何方法：Riemannian 代数梯度流、度量张量与流动。

**📊 数据集**

本文主要为综述性研究，未在真实数据集上进行大规模实验；但在理论讨论与小规模仿真中提到：
- 合成稀疏向量（如长度为 10⁴、稀疏度 s=50 的向量）与合成低秩矩阵（如 20×20、秩 2）
- 传统压缩感知（如基追踪）与 LISTA 以及梯度流对比。
- 在医学成像、雷达、天文等应用场景中引用已有文献中使用的 MRI、SAR、星图等数据集，但本文并未自行实验这些数据集。

**📈 对比分析**

对比方法：
- 传统压缩感知（基追踪、CoSaMP 等）与可学习的 LISTA。
- 传统梯度下降与梯度流在多层对角线网络上的收敛速率和最终误差。
- 低秩恢复中梯度流与经典核范数最小化的恢复概率。

性能表现：
- 在合成实验中，LISTA 在保持相同迭代次数下可显著降低恢复误差，且训练后推理速度提升。
- 对角线网络在小初始化下的梯度流收敛至 ℓ1 最小化解，恢复误差与理论上界相符。
- 在低秩矩阵恢复中，深度网络在更少测量下就能成功恢复（几乎达到信息论下限）。

**⚠️ 局限性**

局限性：
- 主要基于理想化的线性/可线性化模型（对角线网络、线性网络），对非线性深度网络的理论尚未完整。
- 对测量矩阵的假设（随机高斯、子高斯、可对角化或满足 RIP）在实际硬件中可能不完全成立。
- 只给出了梯度流的理论结果，离散梯度下降的收敛与隐式正则化分析仍需进一步研究。
- 未在真实大规模数据集上验证所提出的网络结构与理论预测的实际效果。
- 对于多层 ReLU 或更复杂激活函数的隐式正则化，目前仅有有限的初步结果。

---

## 278. Rethinking Semantic Alignment in LLM-Enhanced Collaborative Filtering: A Spectral Decoupling Approach

**arXiv ID:** 2608.24363 | [PDF](https://arxiv.org/pdf/2608.24363v1)

**作者:** Yedong Jin `[一作]` (Nara Institute of Science and Technology), Eiji Aramaki `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种将LLM语义特征与协同过滤特征在各自谱域中进行预测级联的分离式框架UniSpecRec。

**💡 创新点**

创新点在于从谱角度揭示协同信号主要依赖低频成分、语义信号对非主导成分也有价值，并证明传统对齐学习会抑制这些成分，进而设计无参数、基于谱滤波的解耦融合方法。

**🔧 技术方法**

技术包括图谱分解、奇异值分解、功率谱滤波器、预测级联以及对LLM嵌入的离线SVD处理。

**📊 数据集**

实验使用Amazon评论的Games、Toys和Books三大类别数据集，并采用LLaMA-3.2-3B、NV-Embed-v2和Qwen3-Embedding-8B三种LLM编码器。

**📈 对比分析**

与传统协同过滤模型及多种对齐式LLM增强方法比较，UniSpecRec在Recall@20/NDCG@20上均显著提升（最高提升≈15%），且训练/推理时间更短、跨编码器稳定性更好。

**⚠️ 局限性**

局限性包括对非主导语义成分的内在语义解释不足、对超参数（滤波指数p和融合权重α）的依赖，以及在序列或多模态推荐场景的适用性尚待验证。

---

## 279. ColorA11Y: Enhancing Creative Design Workflows with Just-in-Time Color Accessibility Recommendations

**arXiv ID:** 2608.23852 | [PDF](https://arxiv.org/pdf/2608.23852v1)

**作者:** Alexa Siu `[一作]` (Adobe), Jonathan Lazar `[通讯]` (University of Maryland)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一款名为ColorA11Y的设计工具，能够在用户编辑图形内容时实时检测文字与背景的对比度，并给出多种可操作的颜色调整建议（文字颜色、背景颜色、描边、透明度等），支持复杂背景下的色彩可访问性优化；

**💡 创新点**

创新点在于将可访问性检查从事后校验转为“出生即可访问”——在创作流程中即时提供上下文感知的可操作建议，使设计师在保持美学的同时满足WCAG对色彩对比度的要求；

**🔧 技术方法**

技术实现包括：基于WCAG 2.0的对比度计算（对多色背景进行像素级采样）；利用React+Flask+Konva.js构建交互式画布编辑器；实现六类建议算法（文字色、描边、背景色、背景透明度、局部描边、局部透明度），并提供多种候选颜色；

**📊 数据集**

使用了自建的六种设计背景（纹理、图片、渐变）以及40名有设计经验的受试者进行问卷评估，另外8名设计师参与对比实验，未使用公开数据集；

**📈 对比分析**

通过两项用户研究评估：①对比度建议在不同背景下的可读性与视觉吸引力（使用CLMM模型和Likert评分）发现背景透明度和背景色在纹理背景中表现最佳；②与传统对比度检查器对比（ASQ满意度、时间效率、符合WCAG的文本元素比例），ColorA11Y在100%元素满足对比度且用户满意度显著提升（平均分4.5/5）而基线仅71%符合；

**⚠️ 局限性**

局限性包括：仅考虑WCAG 2.0的对比度算法，未针对色弱/盲设计；未覆盖排版、语义结构等其他可访问性维度；工具功能比完整设计软件有限，实验样本偏小且仅为视力正常设计师，缺乏长期使用与色弱受试者评估；

---

## 280. Agentopia on a Consumer GPU: A Reduced-Scale Long-Horizon Port with an 8B Model

**arXiv ID:** 2608.24215 | [PDF](https://arxiv.org/pdf/2608.24215v1)

**作者:** Luo Huan `[一作]` `[通讯]` (Shenzhen University), Luo Huan (Shenzhen University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

实现并评估了在消费者级RTX 5070 Ti GPU上使用Qwen3‑8B‑AWQ 4‑bit量化模型的Agentopia缩减版，多代理社会模拟完成52周（共154系统周），并加入了系统管理的层级内存压缩、每日四块时间分区及显式物理/心理健康状态变量；

**💡 创新点**

在低资源环境下通过系统级层级内存压缩实现长期记忆管理，采用每日四块时间分区提升行为细粒度，并引入健康监测模块以监控代理状态，从而实现长周期多代理模拟；

**🔧 技术方法**

使用Qwen3‑8B‑AWQ 4‑bit量化模型、vLLM推理服务器、PagedAttention、层级内存压缩、四块时间分区、健康监测模块、随机化提示与单线程并发控制等技术；

**📊 数据集**

基于Agentopia框架的5‑agent apartment world数据，使用生成的角色资料与情景；原始模型权重来自Qwen3‑8B‑AWQ，未公开大型数据集；

**📈 对比分析**

与原始100‑agent 10 年实验对比，在单GPU下完成52周；与内存开关、时间分区单独对比发现内存关闭不影响完成率，四块时间分区记录数提高2.72倍但缺失字段增多；健康模块关闭导致21周/38周提前终止；整体性能：无死亡/健康警告，记录完整率约10%，耗时33‑40小时；

**⚠️ 局限性**

仅使用单模型、无并发、仅5‑agent、未完整匹配基线、健康模块未充分评估、缺乏人类评估多样性、健康指标未验证、未检验其他世界、随机种子不固定、仅10周时间比较等限制。

---

## 281. PROOF-Gen: From Optimized Data to Better Distillation

**arXiv ID:** 2608.23911 | [PDF](https://arxiv.org/pdf/2608.23911v1)

**作者:** Anh Ta `[一作]` (Apple), Shahin Shayandeh `[通讯]` (Apple)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种每个场景都进行提示优化的技术，利用反射模型对教师模型生成的失败轨迹进行纠正，从而恢复原本被generate‑and‑filter丢弃的高质量轨迹，并在后续的监督微调中使用这些恢复后的轨迹。

**💡 创新点**

创新点在于：①用提示优化技术“过拟合”单个场景而非寻找通用提示，从失败轨迹中提取纠正性指导；②在训练前剥离提示“脚手架”，保证学生模型只学习演示的内容而非提示语义；③通过将教师执行器保持不变，实现能力与语音（interaction voice）的解耦，让学生在保留教师执行风格的同时获得更强的解题能力。

**🔧 技术方法**

核心技术包括：①使用GEPA（Prompt Optimizer）与DSPy框架完成每场景提示优化；②GPT‑4o作为教师执行器，GPT‑5.x（5.1/5.4）作为反射模型；③QLoRA进行4B规模指令调优；④生成式评估器与验证器做执行‑based 验证；⑤在训练中采用generate‑and‑filter与补全的混合数据集。

**📊 数据集**

主要数据集：τ2‑bench（telecom多轮客服场景，2285任务），BFCL v4 multi‑turn（162 API，800任务，分为四类），以及内部生产级别的数千条真实或模拟场景。研究中对这些数据集分别进行训练与评估。

**📈 对比分析**

与传统的generate‑and‑filter（只保留通过的轨迹）或partial‑credit（仅给已完成动作打分）基线相比，恢复技术显著提升任务完成率：在τ2‑bench上，Qwen3‑4B‑Instruct‑2507的Pass¹从0.132提升到0.529；在BFCL上，Gemma‑4‑E4B‑it的任务准确率提升+7.2个百分点；在生产评估中，目标完成率提升+6.3pp，离线on‑device模型提升+1.5pp，且所有地区均出现正向迁移（非英语平均+1.48pp）。

**⚠️ 局限性**

局限性：①恢复效果高度依赖反射模型的推理能力，跨供应商/开源模型的可迁移性未知；②目前仅使用GPT‑4o作为教师，缺少对教师模型多样性的评估；③假设教师能执行长度达7–9K字符的详细提示；④在自我蒸馏（student作为教师）时，学生的指令跟随能力可能成为瓶颈；⑤受限于Benchmarks规模（尤其是BFCL的任务数量有限），难以在更大多域场景验证方法的泛化。

---

## 282. Giraffe: A Mapping Architecture from Hidden Text Representations to Visual Embeddings for Efficient Graphic Design

**arXiv ID:** 2608.23970 | [PDF](https://arxiv.org/pdf/2608.23970v1)

**作者:** Nejla Ghaboosi `[一作]` `[通讯]` (Canva Research), Nejla Ghaboosi (Canva Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种名为Giraffe的单图像单Token映射架构，使大型语言模型能够生成图形设计，支持文本到设计和图像到设计两种任务。

**💡 创新点**

采用两个浅层MLP块、一个压缩模块与共享扩展模块的L形结构，仅用单个<IMG> token映射到CLIP视觉嵌入空间，并通过六种损失训练，训练时辅助块后推理去除，实现轻量化且高效的视觉生成。

**🔧 技术方法**

结合CLIP ViT‑L/14视觉编码器、FLUX+CLIP VIT‑L/14 IP Adapter图像生成、SiLU/Tanh激活的多层感知机、MSE/余弦相似度/InfoNCE等多重损失，使用Gemma3 4B或Transformer作为语言模型。

**📊 数据集**

训练使用了180万条专业图形设计样本（社交媒体帖子、横幅、传单、名片、徽标等）并按90/10划分，图像到设计任务使用GPT‑4o生成的样本。

**📈 对比分析**

与基线（仅文本描述+FLUX生成）对比，在文本到设计任务中FID从81.61降至66.85，图像到设计任务CLIP余弦相似度为0.85±0.05，显示生成质量与一致性显著提升。

**⚠️ 局限性**

仍依赖预训练视觉编码器和大型语言模型，方法主要针对图像，音视频等多模态需进一步适配；推理时需额外去除辅助块，且对极端多图场景仍可能受限于序列长度。

---

## 283. Revelation Control

**arXiv ID:** 2608.23860 | [PDF](https://arxiv.org/pdf/2608.23860v1)

**作者:** Qinyou Wang `[一作]` `[通讯]`, Qinyou Wang

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了如何通过未来学习干预揭示隐藏学习状态，从而做出更优决策，并构建了理论与实验框架。

**💡 创新点**

提出了 Revelation Control 理论，将隐藏状态分辨为决策相关的 quotient，区分纯信息增益与可复用计算，给出了可度量的揭示深度、决策足够性和适应性控制的理论判据，并验证了跨模型的结构一致性。

**🔧 技术方法**

使用贝叶斯决策理论、信息价值分解、局部几何揭示深度、等预算计算匹配、两步策略（重启 vs 继续）对比、两模型 Transformer（Qwen2.5‑7B 与 Mistral‑7B‑v0.3）实验、统计推断、t 检验、Bootstrap、贝叶斯风险控制等技术。

**📊 数据集**

使用了两种 7B Transformer 的内部数据：Qwen2.5‑7B 的 336‑anchor、432‑anchor、480‑anchor 等多组测试与开发面板，以及 Mistral‑7B‑v0.3 的 336‑anchor、480‑anchor 两组。

**📈 对比分析**

对比了等预算下的 H8 深度试探与 H4 短探 restart 的性能，评估决策足够性、揭示价值、可复用计算收益。结果表明，在两组模型中，H8 试探在决策价值、可复用计算和等预算终端效益上均优于 H4 对比；Qwen 在决策非冗余揭示上有显著收益，Mistral 则符合单维决策充分性。

**⚠️ 局限性**

实验仅覆盖两类相似 Transformer，缺乏跨规模、跨架构验证；只验证了单步揭示深度，无多步 Markov 闭包分析；生产性揭示假设需可复用的路径，可能不适用于损伤或不可逆干预；风险‑严重度分离导致仅在已预设尾部条件下可获得期望效用保证；未证明全局最优性，仅在开发样本上固定策略。

---

## 284. Causal Analysis for Time Series Foundation Models

**arXiv ID:** 2608.24303 | [PDF](https://arxiv.org/pdf/2608.24303v1)

**作者:** Mathis Jander `[一作]` (University of Twente), Martijn Mes `[通讯]` (University of Twente)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一套因果分析框架，用于在部署前识别时间序列基础模型的偏差与失效模式，并在六种生成器上对两款主流模型进行实验验证。

**💡 创新点**

创新点在于首次将因果干预（do-operator）应用于时间序列生成过程，以量化模型对不同时间序列模式的响应，揭示了模型在漂移、持久性、周期、状态切换和阈值事件上的系统性偏差。

**🔧 技术方法**

采用了Pearl因果框架、参数干预、参数统计量（如漂移估计、AR(1)系数、频率、停留时间、阈值、Hurst指数）以及FFT、Hurst估计等技术手段。

**📊 数据集**

使用了六个仿真生成器（随机游走、AR(1)、谐振子、状态切换、能量释放、分数布朗运动）以及各自的参数空间进行实验，没有使用真实业务数据。

**📈 对比分析**

通过比较模型输出的参数统计与原始时间序列的参数统计，绘制剂量-反应关系，结果显示两模型在漂移和周期上表现良好，但在持久性、状态切换、阈值与高阶持久性上表现出过度估计或平滑失效，未给出具体数值指标。

**⚠️ 局限性**

局限性包括仅使用固定长度（200步）窗口、单一噪声规模、有限的干预范围、仅评估两款模型，缺乏对真实业务数据的验证，且未探讨模型架构对结果的影响。

---

## 285. Preference Data Selection for Mitigating the Alignment Tax in Large Language Models

**arXiv ID:** 2608.24192 | [PDF](https://arxiv.org/pdf/2608.24192v1)

**作者:** Minsu Kim `[一作]` (KAIST), Steven Euijong Whang `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 BALIGN，一种基于风险分数的数据选择策略，在对大语言模型进行对齐时预先过滤掉高风险的偏好样本，从而降低对齐带来的灾难性遗忘，保持模型的通用能力。

**💡 创新点**

创新点在于：①从梯度分析出三种与灾难性遗忘相关的特征（参考模型的对数概率边际、所选与拒绝答案的长度差异、与通用语料的 TF‑IDF 相似度）；②将这三种特征标准化后加权求和得到统一的复合风险分数；③在数据选择阶段只保留低风险样本，既兼顾模型稳定性又保持对齐效果。

**🔧 技术方法**

技术包括：直接在 DPO（Direct Preference Optimization）框架下工作；通过单次前向推理计算三种风险分数；对风险分数做 min‑max 归一化并线性组合得到复合分数；根据预设采样比例阈值筛选样本；随后用 DPO 对筛选出的样本进行微调。

**📊 数据集**

使用的主要数据集为：
- 偏好数据：HH‑RLHF（帮助性与无害性两项）
- 通用能力基准：MMLU、IFEval、ARC‑Challenge、HumanEval、GSM8K
- 参考模型训练语料（用于 TF‑IDF 向量化）

**📈 对比分析**

与多种基线（Full、Random、DSIR、LESS、TSDS、NICE、GrADS、OGS、Selective DPO、BeeS、PD）在同一数据预算下进行对比。BALIGN 在通用能力指标上始终位于最优或次优，且在对齐收益（R）上与全数据或随机采样接近，整体上实现了 Pareto 前沿。计算时间仅略高于基线的统计方法，但远低于基于梯度或额外模型的对齐方法。

**⚠️ 局限性**

局限性：
- 需要依赖参考模型的前向推理来计算风险分数，对模型架构有一定依赖；
- 超参数（α、γ、λ、β）对结果影响较大，需要在验证集上调优；
- 目前仅在 DPO 对齐框架下验证，对其他对齐算法的适用性尚未深入；
- 对极端长文本或特殊领域偏好可能需要进一步调整风险分数或引入更多特征。

---

## 286. Beyond Confidence: Test-Time Scaling for Multi-Turn Search Agents via Retrieval Grounding

**arXiv ID:** 2608.24024 | [PDF](https://arxiv.org/pdf/2608.24024v1)

**作者:** Hyunho Kook `[一作]` (University of Southern California), Beidi Chen `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Retrieval‑Grounded Voting（RGV）方法，利用答案与检索文档的词汇重叠来给多轮检索式搜索代理的并行轨迹加权，从而提升投票质量。

**💡 创新点**

创新点在于诊断并指出检索后续上下文导致的 copy‑inflation 现象，并将投票信号迁移至检索日志之外，利用检索结果的外部信息解决信心投票失效的问题。

**🔧 技术方法**

技术包括：token‑级 logprob 分析、词汇重叠（lexical overlap）计算、最大覆盖（max‑over‑docs）投票权重、加权多数投票，且不需要额外 LLM 调用或 logprob 采样。

**📊 数据集**

使用四个多轮检索代理基准：BrowseComp‑Plus、BrowseComp、GAIA、FRAMES；并在每个基准上评估五个大型语言模型（gpt‑oss‑120b、MiniMax‑M2.7、GLM‑5.1、Kimi‑K2.5、Tongyi‑DeepResearch）。

**📈 对比分析**

与简单多数投票和基于 token‑logprob 的 DeepConf 投票相比，RGV 在所有 20 个 benchmark×模型组合中均取得最高准确率，平均提升 5.4%（最高 35% 在少数正确样本上），且在仅使用 4 次轨迹时即可达到 DeepConf 的 8 次轨迹的准确率。

**⚠️ 局限性**

局限性包括：依赖检索质量，检索错误时无法改进；可能给出表面相似但错误的答案（错误检索但高重叠）；对极短答案的区分度降低；仅在英文数据上验证，对攻击性污染的鲁棒性尚未解决。

---

## 287. NeuronGuard: Robust LLM Safety Alignment via Ablation-Aware Safety Signal Redistribution

**arXiv ID:** 2608.23959 | [PDF](https://arxiv.org/pdf/2608.23959v1)

**作者:** Anjun Gao `[一作]` (University of Louisville), Minghong Fang `[通讯]` (University of Louisville)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种在微调阶段通过动态识别安全关键神经元并在训练中逐步稀释其重要性的防御框架，能够同时抵御 jailbreak 与 neuron‑level 攻击。

**💡 创新点**

创新点包括：① 使用逐层线性分类器周期性重新定位安全关键神经元；② 在训练中强制抑制这些神经元并用 KL 约束保持输出一致，从而迫使安全信息在网络中更广泛分布；③ 采用随机梯度投影解决安全目标与任务目标冲突，保持模型性能；④ 给出正式的 ASR 上界下降理论证明。

**🔧 技术方法**

核心技术包括：逐层线性分类器、神经元抑制（ablation）、KL 一致性正则、随机梯度投影、LoRA 微调、理论上界分析。

**📊 数据集**

使用的数据集与评估：安全探测集（750 违规 + 750 合规），SST2、AGNews、CoLA、GSM8K 任务；StrongREJECT、BeaverTails、NSFW Detection、T2I 转化的多模态输入；六种攻击策略（PAIR、TAP、Puzzler、GCG、AutoDAN、NeuroStrike）。

**📈 对比分析**

与 Prompt‑level（Perplexity、SmoothLLM、GradSafe）和 Model‑level（CAT、LED、SafeNeuron）防御进行对比；在 Llama‑3.1‑8B、Qwen2.5‑7B、Falcon3‑7B 上实验，ACC 近似无损（>0.90），ASR 下降至 0.00–0.04，显著优于所有基线。

**⚠️ 局限性**

局限性：仅在拥有完整微调管道（数据、权重、目标）时有效，无法直接用于未微调或仅提供 API 访问的部署场景；对计算资源有一定要求，尤其是动态 neuron 识别与梯度投影。

---

## 288. ExMesh++: From Multi-View Images to Relightable UV-PBR Mesh Assets via Topology-Adaptive Reconstruction and Decomposition

**arXiv ID:** 2608.24109 | [PDF](https://arxiv.org/pdf/2608.24109v1)

**作者:** Chuanjin Fan `[一作]` (University of Science and Technology of China), Tianzhu Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 ExMesh++，一个两阶段框架，将多视角图像重建为可编辑、可重光的 UV‑PBR 网格资产。

**💡 创新点**

1）采用自适应顶点分割/合并的显式网格优化，并在拓扑变化时保持 UV 连续；2）在固定网格‑UV 上进行 PBR 材质与环境光分解，减少几何‑材质‑光照相互补偿；3）引入一次性散射间接照明，无需学习残差光场。

**🔧 技术方法**

利用可微光栅化（nvdiffrast）、UV 重新映射（xatlas）、蒙特卡罗采样、一次性间接照明射线追踪以及金属‑粗糙度 PBR 模型。

**📊 数据集**

在 DTU、Synthetic4Relight、Stanford‑ORB 以及 NeRF‑Synthetic 等数据集上进行实验。

**📈 对比分析**

与 NeRF、Gaussian、mesh‑driven 以及逆渲染方法对比，采用 Chamfer Distance、PSNR、SSIM、LPIPS 等指标。结果显示 ExMesh++ 在几何精度、重光合成和材质分解上与现有方法相当或更优，训练时间约 30 分钟，输出资产可直接在 DCC 中使用。

**⚠️ 局限性**

局限在于仅使用金属‑粗糙度 PBR，无法处理更复杂材质；仅实现一次性散射间接照明，忽略多次散射和镜面反射；UV 重新生成依赖 CPU，可能在大模型上成为瓶颈；缺乏金属通道的定量评估。

---

## 289. OmniJudge or OmniBias? Diagnosing Multimodal Judges through Balanced, Decoupled Lenses

**arXiv ID:** 2608.24160 | [PDF](https://arxiv.org/pdf/2608.24160v1)

**作者:** Guangzheng Hu `[一作]` (Alibaba Group), Jin Xu `[通讯]` (Qwen Team)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 D3-Omni，一套针对多模态评判模型（OmniJudges）的平衡且解耦的细粒度评估基准，覆盖图像、视频与语音生成三种任务。

**💡 创新点**

创新点在于：① Dual-balanced、Decoupled、Dynamic（D3）构造框架，可在每个维度实现近 1:1 的正负样本比例并保持总分均匀；② 通过已验证的正样本加单维度扰动（prompt rewriting 与原子模态操作）实现真正的解耦；③ 动态迭代填充缺失的分数段，持续保持基准的平衡与可扩展性。

**🔧 技术方法**

技术手段包括：多 LLM 逆向 prompt 生成、人工校验、对维度进行单点翻转的 prompt 重写、模态专用原子扰动、维度耦合矩阵评估、动态平衡算法与多任务统一采样。

**📊 数据集**

数据集：构建了 10,671 条 T2I、T2V、TTS 任务样本，包含 53 个相互正交的二元维度（T2I 17、T2V 22、TTS 14），数据来自公开生成模型（如 Gemini、GPT、Claude、Qwen 等）并在 Hugging Face 发布的 D3OmniBench。原始正样本通过多模型逆向生成得到，随后用人工审核保证标签准确。

**📈 对比分析**

与现有评判模型（10 种 T2I、8 种 T2V、6 种 TTS）对比，采用 per-dimension accuracy、segment accuracy、perfect-match、维度耦合矩阵等多指标评估。结果显示：总体准确率在视觉任务中约 70%，在 TTS 中约 65%；但在中等分数段出现 U‑形性能下降，且“是”类（Yes）识别显著优于“否”类（No），多模态任务的错误共聚合导致判别能力隐藏。

**⚠️ 局限性**

局限性：① 构造器仅使用了有限的正样本库与原子扰动集合，覆盖面可能不足；② 动态平衡只关注标签分布稀疏，未针对特定模型的错误分布进行自适应；③ 当前仅覆盖 T2I/T2V/TTS，未验证逆向任务；④ 需要人工验证的步骤仍然存在成本与主观性。

---

## 290. Discovering Cross-Language Reasoning Invariance in LLMs with Geometry-Invariant Sparse Autoencoders

**arXiv ID:** 2608.23809 | [PDF](https://arxiv.org/pdf/2608.23809v1)

**作者:** Igor Bogdanov `[一作]` (Carleton University), Changcheng Huang `[通讯]` (Carleton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多语种大型语言模型中研究跨语言推理的内部特征共享，利用稀疏自编码器和因果补丁验证特征是否在不同语言之间可互换。

**💡 创新点**

提出 Geometry‑Invariant Sparse Autoencoder (GI‑SAE)，在传统稀疏自编码器基础上加入 InfoNCE 对比损失，使相同问题的不同语言激活趋于一致，并构建了一个跨五个模型、六种语言的因果补丁评估框架。

**🔧 技术方法**

使用 top‑K 稀疏自编码器、InfoNCE 对比损失、CKA 与 Jaccard 相似度、因果补丁（特征替换）以及 KL 散度评估输出变化；通过 TransformerLens 重放推理轨迹，采集内部残差流进行训练与评估。

**📊 数据集**

使用 Multilingual Grade School Math (MGSM) 数据集，包含六种语言（英语、德语、法语、西班牙语、俄语、中文）的同一批小学数学题目。

**📈 对比分析**

将 GI‑SAE 与仅重建的 baseline SAE 在相同层进行对比，使用 CKA、Jaccard、KL/feature 进行度量；GI‑SAE 在几乎所有层提高几何相似度，但仅在“收敛”共享程度（15–60%）的层获得功能互换性提升（约 83% 成功率），在“饱和”层（Gemma）几乎无效，在低共享层（Llama）效果有限。

**⚠️ 局限性**

局限性包括：仅测试 MGSM 数学推理，某些模型有效样本数少；仅覆盖 1.7–4B 参数规模；超参数（K、w、τ）固定；每个问题仅采样一条推理轨迹；对比正样本未按时间步匹配；跨模型族覆盖有限；对更大规模模型或更复杂任务的可扩展性尚未验证。

---

## 291. Vision Language Model Fusion for Explainable Face Recognition

**arXiv ID:** 2608.24430 | [PDF](https://arxiv.org/pdf/2608.24430v1)

**作者:** Ana Estrada-Real `[一作]` (Hochschule Darmstadt), Christian Rathgeb `[通讯]` (Hochschule Darmstadt)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了多视觉-语言模型（VLM）融合来提升人脸识别的准确性并生成可解释的文本说明。

**💡 创新点**

创新点在于将多个VLM的相似度分数、文本解释与原始图像信息一起交叉融合，使得最终决策既更准确又提供更丰富、可信的可解释性。

**🔧 技术方法**

主要技术包括零-shot VLM推理、四种融合策略（仅分数、分数+解释、分数+图像、分数+解释+图像）以及基于三模型决策器的后续推理。

**📊 数据集**

实验数据集为 Labeled Faces in the Wild（LFW），构造 200,000 张 50:50 的正负人脸对。

**📈 对比分析**

通过与 AdaFace、MagFace、LVFace 等传统 FR 基线对比，VLM 单体可达 AUC 0.9985、EER 1.30%；融合后最佳配置（Gemma 作为决策器，Qwen+Intern 为源模型）EER 1.06%，FNMR 7.20% @0.1%FMR，显著优于单体 VLM 并接近专业 FR 模型。

**⚠️ 局限性**

局限性包括：在极低 FMR（0.01%）点性能仍不理想；解释质量评估仍缺乏统一自动化指标；多模型融合计算成本高，实时性受限；以及实验仅在公开 LFW 上验证，可能受模型训练数据泄露影响。

---

## 292. Trust, but Verify: Rigorously Profiling Best-Effort High-Performance Computing for Digital Evolution

**arXiv ID:** 2608.23955 | [PDF](https://arxiv.org/pdf/2608.23955v1)

**作者:** Matthew Andres Moreno `[一作]` (University of Michigan), Emily Dolson `[通讯]` (Michigan State University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套针对最佳努力（best‑effort）高性能计算的运行时行为测评框架，并将其应用于数字进化工作，在传统 CPU 集群和 Cerebras Wafer‑Scale Engine（WSE）上进行实验。

**💡 创新点**

创新点在于：① 将质量服务（QoS）指标与最佳努力执行结合，实现对非确定性运行时行为的可测量；② 采用条形码式基因标记与稀疏采样重建进化树，支持在极度受限的设备内存与通信环境下捕获进化历史；③ 通过对比同步与异步执行，证明最佳努力策略在规模扩展、错误耐受和能效上的显著优势。

**🔧 技术方法**

技术手段包括：MPI 异步通信与 Conduit 库、最佳努力的发送缓冲区与丢包策略、QoS 维度（straggling、latency、attrition、bunching）的实时采样与统计、单比特条形码记录、可变长度环形缓冲区用于时间序列记录，以及基于树形聚类的进化历史重建。

**📊 数据集**

使用的数据集涵盖：a）DISHTINY 多细胞数字进化模型（约 3,600 细胞/进程）；b）图着色基准（2,048 节点/进程、100 次迭代）；c）WSE 上 226.1 M 代理人 5 M 代的稀疏采样（≈9,000 次快照，1.1 B 记录用于后续分析）；d）WSE 的高突变体演化实验，记录未知持续时间的时间序列。

**📈 对比分析**

比较方法：对同一工作负载在同步与最佳努力两种模式下进行弱/强扩展实验，记录平均执行速度、QoS 统计（中位数与极值）以及错误率；在 WSE 上比较稀疏采样前后对进化树重建的完整度与偏差。性能结果显示：在 64 进程时，DISHTINY 的最佳努力模式保持 92% 的扩展效率（vs. 47% 同步），并实现 2.1× 的加速；图着色在最佳努力下实现 12.5× 的加速与 73% 的误差下降；QoS 指标在扩大至 256 进程时保持稳定，且对硬件异常的鲁棒性良好；WSE 稀疏采样能够在 5% 传输错误率下继续正常工作。

**⚠️ 局限性**

局限性：① 仅对观察层面进行最佳努力，核心计算仍保持确定性，需谨慎划分；② 可能引入统计偏差，需通过多重实验与 QoS 监测进行校正；③ 对特定硬件（如 WSE 的内存与通信拓扑）高度依赖，迁移到其他加速器需重新调优；④ 由于采用条形码与环形缓冲，无法重建完整的细胞谱系，适用范围受限；⑤ 在极端负载下，QoS 的极值可能仍会影响整体结果，需要进一步的容错与纠错机制。

---

## 293. B-MIM: Biased Masked Image Modeling for Generalizable Segmentation of Fine-Grained Anatomical Structures

**arXiv ID:** 2608.24364 | [PDF](https://arxiv.org/pdf/2608.24364v1)

**作者:** Sebastián González `[一作]` (Universidad de Chile), Bernard Ghanem `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8d10c613-917e-4880-9716-17789f50e119` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究了一种名为B-MIM的自监督预训练方法，优化3D Swin Transformer以更好地捕捉CT图像中细粒度解剖结构，并验证其在肝血管和小肿瘤分割任务上的跨数据集泛化能力。

**💡 创新点**

通过在iBOT框架中引入随机忽略全局CLS损失的“偏置”策略，降低全局语义对齐压力，突出局部补全任务，提升对细小结构的表征能力，并在保持参数量低的前提下实现高精度分割。

**🔧 技术方法**

使用3D Swin Transformer骨干、iBOT自监督蒸馏、B-MIM偏置损失、nnU-Net轻量化解码器、LoRA参数高效微调以及clDice评估等技术。

**📊 数据集**

构建了9,955例跨17源的腹部CT多机构数据集用于预训练，随后在CRLM（197例）上训练下游模型，并在未参与预训练的IRCAD（20例）和MSD（303例）上评估泛化。

**📈 对比分析**

采用5折交叉验证与外部测试，对比Swin‑nnUNet、VoCo等基线；B‑MIM在血管分割的clDice从0.491提升至0.582（+18.5%），Dice保持竞争力；肿瘤分割表现更为可变，B‑MIM‑B在Dice上取得最佳结果。

**⚠️ 局限性**

对肿瘤等形态多样任务的提升有限，缺乏对不同p值的系统分析，且仅验证了CT模态，尚未推广到MRI等其他成像类型。

---

## 294. On the Robustness of Audio Deepfake Detection under Audio Watermarking

**arXiv ID:** 2608.24159 | [PDF](https://arxiv.org/pdf/2608.24159v1)

**作者:** Zi Qian Yong `[一作]` (Monash University), Sébastien Marcel `[通讯]` (Idiap Research Institute)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究音频水印对音频深伪检测（ADD）系统的鲁棒性影响，评估多种ADD模型在加水印后的检测性能和特征空间漂移；

**💡 创新点**

将音频水印视为结构化非攻击性扰动进行鲁棒性评估，并通过Fréchet距离、余弦相似度、L2距离等指标对嵌入空间进行深入分析，揭示水印对SSL模型鲁棒性的大影响；

**🔧 技术方法**

使用WavMark音频水印、SWFT预处理、可逆神经网络嵌入、EER评价、Fréchet距离、余弦相似度与L2距离等技术；

**📊 数据集**

ASVspoof 2021 LA、DF、ASVspoof 2024、In-the-Wild (ITW) 与 Fake-or-Real (FoR) 数据集；

**📈 对比分析**

通过比较未加水印与加水印时的EER及嵌入空间距离来评估；结果显示在ASVspoof 2021 LA/DF 上EER显著上升（最高36.5%），Fréchet距离巨大；在ASVspoof 2024、FoR、ITW 上影响较小；

**⚠️ 局限性**

仅考虑单一水印方案（WavMark）和预训练ADD模型，未探究不同水印容量、音频质量及多类攻击的影响；实验仅基于二分类评估，缺乏更广泛的鲁棒性验证。

---

## 295. AdaWidth: Query-Adaptive Embedding Width for Dense Retrieval

**arXiv ID:** 2608.23862 | [PDF](https://arxiv.org/pdf/2608.23862v1)

**作者:** Shubing Yang `[一作]` (University of Washington), Dongfang Zhao `[通讯]` (University of Washington)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 AdaWidth，一种在保持完整编码器的前缀表示不变的前提下，对每个查询动态分配要评估的嵌入宽度，从而在密集检索中显著降低计算成本。

**💡 创新点**

创新点在于：①使用正交前缀适配器（Householder 反射）将判别信息聚集到前缀坐标，同时保证全宽相似度不变；②引入基于阶段 1 排名的统计特征的轻量路由器，仅在需要时扩展到更宽前缀；③从理论上推导前缀充分性分析，揭示所需宽度随语料库大小与检索深度的对数关系。

**🔧 技术方法**

核心技术包括：正交矩阵的 Compact WY 形式 Householder 反射、梯度提升回归树路由器、三重对比损失与几何一致性约束、以及对前缀足够性（order‑statistic）分析；所有操作均在冻结的高维编码器上完成。

**📊 数据集**

使用六个检索任务的数据集：BEIR 的四个文本检索任务（FiQA、ArguAna、Quora、MS MARCO）以及两个视觉‑语言检索任务（OK‑VQA、A‑OKVQA），并在五个冻结的文本/视觉‑语言编码器上评估（E5‑Mistral‑7B、Qwen3‑Embedding‑0.6B、Nomic‑embed‑v1.5、BGE‑large‑en‑v1.5、Qwen3‑VL‑Embedding‑2B）。

**📈 对比分析**

与前缀截断、Matryoshka Adaptor、SMEC 和 Learning‑to‑Select 等基线相比，AdaWidth 在所有任务与编码器上实现了相当甚至更高的 NDCG@10，且每查询平均使用 55%–84% 的维度，显著提升了查询效率和性能。

**⚠️ 局限性**

局限性包括：路由器依赖于阶段 1 排名的统计特征，可能在极端多重复制或高度相关的语料库中效果有限；理论分析假设竞争文档独立，实际场景中近似重复导致偏差；需要额外训练正交适配器与路由器，且对编码器的冻结假设限制了可扩展性。

---

## 296. Optimal Rank Lotteries for Truthful Unit-Interval Covering

**arXiv ID:** 2608.24193 | [PDF](https://arxiv.org/pdf/2608.24193v1)

**作者:** Alexandros A. Voudouris `[一作]` `[通讯]` (University of Southern Denmark), Alexandros A. Voudouris (University of Southern Denmark)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在无报酬单元区间覆盖问题中，利用报告无关的秩抽样机制实现诚实性，并给出了该类机制的最优近似比；

**💡 创新点**

首次对秩抽样机制的最优近似比进行精确表述，并提出了通用的下界9/8，填补了先前5/3上界与3/2下界之间的空白；

**🔧 技术方法**

采用了几何极值归约、随机平移网格、规范化配置以及对连续度量的离散化等数学技术；

**📊 数据集**

无；

**📈 对比分析**

相较于先前的5/3上界，本文给出了更紧凑的3/2-1/(2⌊n/2⌋)上界，同时证明任何通用随机化机制至少需要9/8的近似比；

**⚠️ 局限性**

对n=4时通用随机机制的最优比仍存在9/8至5/4之间的未知区间，且对更一般的机制和更大n的性能提升仍需进一步研究。

---

## 297. A Formal Methodological Framework for Auditing Robustness and Fidelity in Explainable AI: From Application to Trust Certification

**arXiv ID:** 2608.23817 | [PDF](https://arxiv.org/pdf/2608.23817v1)

**作者:** Rosa Elysabeth Ralinirina `[一作]` (University of Fianarantsoa), Thomas Mahatody `[通讯]` (University of Fianarantsoa)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出了一套评估后置解释器鲁棒性和保真度的审计协议，并将两项指标合并为单一的信任分数；

**💡 创新点**

创新点在于将鲁棒性（基于Jensen‑Shannon散度）与保真度（基于特征消融）联合评估，并通过统一的信任分数对解释结果进行量化比较；

**🔧 技术方法**

使用的技术包括SHAP和LIME解释器、TreeSHAP、KernelSHAP、特征消融实验、Jensen‑Shannon散度计算以及基于加噪扰动的鲁棒性测评；

**📊 数据集**

采用马达加斯加多部门食物安全数据集，共83个特征、253条记录、四类慢性营养不良标签；

**📈 对比分析**

通过在三种模型（随机森林、XGBoost、神经网络）和两种解释器（SHAP、LIME）上实验，发现高AUC模型仍可能产生不稳定或无信息解释，且信任分数能显著区分不同模型-解释器组合，正则化能提升解释的鲁棒性与保真度；

**⚠️ 局限性**

局限性包括样本量小、类别极度不平衡导致交叉验证受限、特征消融成本高、噪声扰动可能不符合实际域变异、权重α、β需领域判断、正则化或校准不足时保真度失效。

---

## 298. A Hybrid Two-Stage Machine Learning Pipeline for Fault Detection and Classification in Power Transmission Systems

**arXiv ID:** 2608.23726 | [PDF](https://arxiv.org/pdf/2608.23726v1)

**作者:** Sahil Manikshete `[一作]`, Van-Hai Bui `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种两阶段混合机器学习管道，先用无监督 Isolation Forest 与可选的自适应监督二分类器（OR融合）进行故障检测，再用随机森林多分类器对检测到的样本进行故障类型识别，并通过物理驱动的特征工程（每测点 18 个特征，包括零序对称分量）提高判别性能。

**💡 创新点**

①将检测与分类解耦，使用自动分配的二分类器处理 Isolation Forest 的盲点；②利用零序对称分量特征解决三相与三相接地故障的混淆；③采用统一的每测点特征算子，实现网络规模无关的特征工程；④通过跨数据集验证展示零序签名系统依赖性，证明学习边界优于固定阈值。

**🔧 技术方法**

Isolation Forest、随机森林、多分类随机森林、OR 融合、Fortescue 对称分量特征、统计特征（RMS、最大值、均值、失衡比）、特征工程算子、宏观 F1、端到端准确率评估、误报率统计。

**📊 数据集**

TLFaultDataset（7 测点、578,923 样本、6 故障类+正常）以及独立单测点数据集（12,001 样本、5 故障类+正常）。

**📈 对比分析**

与 TL Fed 的 1D-CNN-LSTM 联邦学习模型（94.84% 端到端准确率）进行对比；在 TLFaultDataset 上将 Line 故障的端到端准确率从 31.3% 提升至 95.8%，整体端到端准确率 95.8%；在第二数据集上实现 97.25% 端到端准确率，超过 TL Fed；误报率从 1.5% 上升到 5.3%，推理时间仅 0.05 ms/样本。

**⚠️ 局限性**

仅在模拟数据上验证，缺乏现场噪声、谐波等实际测量误差；每步独立处理，未利用时序上下文；误报率提升需要平衡；模型需针对每个站点单独训练，未实现跨站迁移；未提供故障定位功能；零序特征的学习边界系统依赖，需在不同网络中重新训练。

---

## 299. SyPS: Measuring Sycophancy Prompt Sensitivity in Large Language Models

**arXiv ID:** 2608.23837 | [PDF](https://arxiv.org/pdf/2608.23837v1)

**作者:** Lijia Huang `[一作]` (Northeastern University), Sihao Ren `[通讯]` (Everpure)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SyPS框架，用于评估大型语言模型在不同社交线索（如自信、情绪、社会共识等）下的顺从性变化；

**💡 创新点**

引入了实例级的 Sycophancy Prompt Sensitivity Score（SPSS），将模型在相同情境但不同提示变体下的顺从性差异量化；

**🔧 技术方法**

构建了八种社交提示变体，并通过 GPT‑4 判断器对开放式回答进行二值顺从性标注；

**📊 数据集**

使用 ELEPHANT 基准中的三组数据集：OEQ（开放式建议问答）、AITA‑YTA（道德判断）和 SS（主观陈述）；

**📈 对比分析**

在九个开源 LLM 上测评基线顺从率与 SPSS，发现不同数据集表现差异，验证了提示敏感性可通过方向性效应（验证求安、情绪压迫、反顺从）进一步分析；

**⚠️ 局限性**

局限性包括：提示变体有限、二值标注简化复杂回答、SPSS 对方向不敏感、模型大小不决定敏感度、评估仅覆盖单轮互动且依赖 GPT‑4 判定器。

---

## 300. A Few Shared Random Bits Suffice for Constant-Round Almost Stable Matching

**arXiv ID:** 2608.24102 | [PDF](https://arxiv.org/pdf/2608.24102v1)

**作者:** Yijun Chang `[一作]` (National University of Singapore), Kushagra Chatterjee `[通讯]` (Indian Statistical Institute)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275`

**🎯 论文内容**

在分布式与并行计算模型下，提出了一种求解近似稳定匹配（允许最多 ε|E| 阻塞对）的算法，并证明：当 ε 为常数时，使用共享随机性可在常数轮次内完成；不使用共享随机性时可在 O(log n) 轮次内完成。

**💡 创新点**

核心创新点为：① 引入“度数守护冻结规则”，一次性全局计费即可处理任意度数分布，摒弃了原方法中需 Θ(log n) 次阈值分段的做法；② 仅用 O(log(1/ε)) 个共享随机比特来选定输出轮次，从而大幅降低随机性需求。

**🔧 技术方法**

采用的技术包括：分位数提议框架（Quantile‑Match）、近似最大匹配子程序（(H,ρ)）、度数守护冻结规则、低直径分解、以及在 MPC 中基于排序与前缀和实现的高效操作。

**📊 数据集**

本工作为理论性研究，无实验数据集，主要在抽象的分布式与并行模型中给出证明。

**📈 对比分析**

与之前的 O(log n/ε³) 或 O(log² n) 结果相比，本算法在常数 ε 时实现了常数轮次（共享随机性）或 O(log n) 轮次（无共享随机性），并把对 ε 的依赖从多项式改为 O(log(1/ε)/ε⁴)。

**⚠️ 局限性**

局限性包括：对 ε 的复杂度仍为 O(log(1/ε)/ε⁴)，与已知的 Ω(1/ε) 下界存在多项式差距；需要共享随机性（若无共享随机性则需 O(log n) 轮次）；并未给出确定性下界，算法在高度非均匀度数下仍保持 O(log n) 轮次。

---

## 301. Subregular Expressions and Their Expressive Power

**arXiv ID:** 2608.24244 | [PDF](https://arxiv.org/pdf/2608.24244v1)

**作者:** Martin Kutrib `[一作]` (Universitaet Giessen), Matthias Wendlandt `[通讯]` (Universitaet Giessen)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对基于不同子正则表达式操作集的语言族进行系统性综述与分类

**💡 创新点**

首次将所有由两操作和三操作子正则表达式产生的语言族整理成统一框架，揭示其包含、等价与不可比关系，并总结未解决的开放问题

**🔧 技术方法**

文献综述、归纳正则表达式的语义与闭包性质、构造正规形式与自动机表述

**📊 数据集**

无实验数据集，全部使用理论推导和已有文献结果

**📈 对比分析**

通过构造示例语言、归纳包含关系图和表格，对各族的表达力进行比较，展示它们在一字母与任意字母情形下的区别与相似性

**⚠️ 局限性**

仍存在多类族间关系未确定、判定问题与结构特征缺乏统一理论、未给出算法实现或复杂度分析

---

## 302. LEMONS: Leveraging Model-Based Techniques to Enable Non-Intrusive Semantic Enrichment in Wireless Sensor Networks

**arXiv ID:** 2608.24277 | [PDF](https://arxiv.org/pdf/2608.24277v1)

**作者:** Jan Novacek `[一作]` (FZI Research Center for Information Technology), Wolfgang Rosenstiel `[通讯]` (FZI Research Center for Information Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建了基于模型驱动软件开发（MDSD）与语义Web技术（SWT）的无线传感器网络（WSN）配置与运行时语义丰富化框架，自动生成网关代码、静态与动态OWL实例，实现非侵入式的语义增强与可重用性。

**💡 创新点**

① 通过定义与SSN、QUDT、Time等标准本体对应的类型无关元模型，实现模型到代码与本体实例的全自动生成；② 在运行时使用微本体（micro‑ontology）生成动态观测实例，避免频繁传输静态信息；③ 对微本体进行GZIP压缩，显著降低MQTT负载；④ 通过上述技术实现无需专家知识的WSN大规模配置与管理。

**🔧 技术方法**

使用EMF+Ecore与DSL、EGL模板实现模型定义与代码生成；Scala + Akka Streams实现异步流式处理；Jena框架处理OWL实例；MQTT协议进行数据传输；GZIP压缩、SPARQL终端查询；利用SSN、QUDT、Time等本体。

**📊 数据集**

实验基于TurtleBot搭载Nordic Thingy:52多传感器平台收集的加速度、温湿度等原始传感器数据；未使用公开大规模数据集，而是自建传感器数据集。

**📈 对比分析**

在Intel i5 CPU上测量微本体生成时间、压缩率与网络负载，平均生成时间<500µs，压缩后单个微本体小于1KB，传输延迟对WSN实时性影响可忽略；与传统手工配置相比，部署时间缩短、错误率降低、互操作性提升。

**⚠️ 局限性**

元模型未覆盖SSN全部内容，映射Ecore与OWL表达力差异导致自动化难度；当前编辑器仅树形，缺乏图形化支持；只生成网关代码，未覆盖服务器配置；DSL未支持JSON/XML等行业标准；尚未评估高级delta压缩等网络流量优化技术。

---

## 303. From State to Action: OODA-Tool for Reliable Multi-Turn Tool Use

**arXiv ID:** 2608.24368 | [PDF](https://arxiv.org/pdf/2608.24368v1)

**作者:** Rongfeng Guo `[一作]`, Vincent Tao Hu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于Observe–Orient–Decide–Act四阶段的闭环工具使用策略OODA‑Tool，专门解决多轮交互中状态与动作竞争问题。

**💡 创新点**

创新点在于把任务状态的重构、执行准备、动作规划和动作实现分别拆成可监督、可校验的类型化阶段，并通过中央控制器逐步验证，显著缓解状态信息被动作生成压制的风险。

**🔧 技术方法**

采用大型语言模型Qwen3作为主干，配合LoRA微调实现四个阶段的专属适配器；并利用类型化接口与约束策略进行阶段监督、校验与训练；使用多轮工具使用流程的闭环推理。

**📊 数据集**

主要使用ToolDial的11,111条多轮工具使用轨迹进行训练与评估，并在FAIL‑TaLMs、MTU‑Bench和BFCL等外部基准上做零样本迁移测试。

**📈 对比分析**

与直接函数调用（Direct‑LoRA）、ReAct‑LoRA、四次采样Direct‑SC@4以及α‑UMi等基线相比，OODA‑Tool在所有模型规模下均实现了更高的Task Success、Tool Exact和Ask‑Act Accuracy；其优势在小模型和需要状态累积、约束变化、序列依赖的任务上最大，随着模型容量增大收益逐渐递减。

**⚠️ 局限性**

局限性包括：对并行工具调用的动作展开与跨调用参数绑定仍表现不佳，导致在多工具并行场景下提升有限；多阶段推理带来显著的延迟开销；在部分类型化状态缺失或不完整时仍可能出现误判。

---

## 304. Who Chooses How Preferences Are Aggregated? Auditing Aggregation-Rule Authority in LLM-Based Group Recommendation

**arXiv ID:** 2608.23966 | [PDF](https://arxiv.org/pdf/2608.23966v1)

**作者:** Yuxuan Du `[一作]` `[通讯]` (Independent researcher), Yuxuan Du (Independent researcher)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在多用户推荐场景中，LLM在何时以及如何决定将个体偏好聚合为群体决策的规则，进而评估不同权威分配对模型行为的影响。

**💡 创新点**

创新点在于将聚合规则权威视为交互层面的对齐问题，将其与规则执行能力与最终结果区分开来，并揭示了在权威保留与委托时出现的结构化推迟行为。

**🔧 技术方法**

采用了受控行为审计方法，对三大LLM（GPT‑5.6、Claude Sonnet 5、Qwen 3.6 Plus）进行交互式提示实验，并通过自动与人工编码对模型的承诺与聚合一致性进行量化分析。

**📊 数据集**

使用了两套数据集：1）由0–10刻度随机采样的合成偏好配置，2）MovieLens 32M中的真实用户评分对，均被划分为核心冲突、共享首选和规则一致三类。

**📈 对比分析**

通过比较未指定、用户保留和用户委托三种权威分配下的承诺率与聚合一致性分布，发现模型在权威保留时几乎不做决定，而在权威委托时总是决定，但不同模型与数据设置下的最终聚合结果差异显著，表明模型对相同的聚合规则有不同的实际选择倾向。

**⚠️ 局限性**

局限性包括仅考虑两人、五选项、单轮、显式数值评分的情景；聚合规则仅限于Additive Utilitarian和Least Misery，未覆盖更广泛的社会选择方案；实验条件为显式权威分配，未探究更自然的多轮交互与语言表达方式。

---

## 305. UHI-Bench: Benchmarking Dual-Source Urban Heat Island Modeling Across Cities in Diverse Climate Regimes

**arXiv ID:** 2608.23857 | [PDF](https://arxiv.org/pdf/2608.23857v1)

**作者:** Wanyun Ling `[一作]` (Technical University of Munich), Ziyue Li `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了 UHI-Bench 基准，集成了双源城市热岛（LST-UHI 与 AirT-UHI）、每小时气象驱动和静态城市形态特征，覆盖 20 个城市、9 个 Köppen 气候，构建统一的评估框架。

**💡 创新点**

首个支持双源 UHI 建模的基准；将多源时空异步数据对齐；提出信号‑机制‑转移三层实验框架；系统评估跨城市、跨气候的迁移性能。

**🔧 技术方法**

使用统计方法、经典机器学习、深度时空模型与时间序列基础模型（如 RF、XGBoost、GraphWaveNet、Chronos-2 等）完成缺失填补、极端事件检测、短期预测、机制解释等任务。

**📊 数据集**

UHI-Bench 数据集：16 个双源核心城市（2015–2025）、4 个仅 LST 城市、2 个仅 AirT 城市；配套 ERA5‑Land 气象驱动、建筑/道路/光照/NDVI 等城市形态特征，约 81,755 个 1 km 像素。

**📈 对比分析**

统一训练/测试切分（2015–2022 训练，2023–2025 评估），使用 MAE、F1、Kendall W 等指标；结果显示无单一模型最佳，基础模型稳定；LST 与 AirT 互补；跨城市迁移更受 UHI 规律重叠影响，气候多样性比同气候内城市数量更重要。

**⚠️ 局限性**

局限性包括：数据仅覆盖 MSG/SEVIRI 范围，分辨率 1 km；双源覆盖不完整（部分城市仅 LST 或 AirT）；云缺口填补仍依赖缺失率；评估未深入极端事件分布；未来需扩展城市、提升分辨率、开发更专业的 UHI 基础模型。

---

## 306. PACT: Post-route Agentic Checkpoint Tuning for FPGA Timing Closure

**arXiv ID:** 2608.23602 | [PDF](https://arxiv.org/pdf/2608.23602v1)

**作者:** Huan Lin `[一作]` (Fudan University), Zhiang Zhang `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出PACT框架，自动化后路（post‑route）FPGA检查点（DCP）的ECO式时序闭合优化；

**💡 创新点**

将LLM决策层与结构化的Vivado/RapidWright操作、验证门控循环和案例记录相结合，形成可验证的局部编辑流程；

**🔧 技术方法**

使用GPT‑5.5 LLM进行诊断与规划、Vivado与RapidWright进行后端操作、约束执行器、验证门控接受与回滚机制；

**📊 数据集**

在35个UltraScale+后路DCP上评估，涵盖经典、HLS、计算密集型、系统级和压力测试设计；

**📈 对比分析**

与DATuner（全流程搜索）和Codex Agent（自由式LLM驱动）对比，PACT在平均时间内实现+22.3% F_max提升，速度提升6.4×，令牌成本低24.5×；

**⚠️ 局限性**

局限性：依赖预先定义的操作集合和动作库，跨设计迁移性有限；对多时钟、多芯片和功耗优化的支持尚未展开。

---

## 307. CARE: Camera-Residual Reserves for First Sightings in Adaptive LiDAR Sensing

**arXiv ID:** 2608.24282 | [PDF](https://arxiv.org/pdf/2608.24282v1)

**作者:** Jiachen Gong `[一作]` (University of Tokyo), Manabu Tsukada `[通讯]` (University of Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 CARE 机制，将相机检测中未被历史跟踪解释的残余信息作为残余预留来分配 LiDAR 视锥，解决首次感知欠分配问题；

**💡 创新点**

创新点在于训练无关的相机残余预留策略、基于安全阈值的忘却模块，以及严格无泄漏的首次感知评估协议；

**🔧 技术方法**

使用 2D YOLOX 相机检测、3D CenterPoint 目标检测、恒定速度轨迹预测、共享随机底层填充和安全阈值驱动的忘却算法；

**📊 数据集**

在 nuScenes 验证集（150 场景、4,148 首次感知事件）以及实车与 CARLA 仿真平台进行评估；

**📈 对比分析**

与历史驱动扫描、随机/均匀预留、全相机预留、不确定性预留、光幕式预留、光束削减以及学习式分配器进行对比；在 10%、20% 和 35% 的光束预算下，CARE 首次感知召回率提升 4.3–5.2 点，整体召回率保持与历史相近或略低；

**⚠️ 局限性**

局限在于仅使用冻结的检测器与轨迹预测，未验证不同传感器组合和动态环境下的鲁棒性，且对极低预算下的整体召回可能仍有不足。

---

## 308. TRACE: Transition-Aware Residual Control for Multi-Objective Materials Discovery

**arXiv ID:** 2608.23631 | [PDF](https://arxiv.org/pdf/2608.23631v1)

**作者:** Kang Zhou `[一作]` (Wuhan University of Technology), Jingling Yuan `[通讯]` (Wuhan University of Technology)

**通讯引用:** 1262 | [OpenAlex ID](https://openalex.org/A5062853168)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TRACE框架，利用LLM代理生成可执行编辑，并记录父-编辑-子转移的反馈来进行多目标材料发现

**💡 创新点**

首次将评估的编辑转移作为可重用的反馈单元，并通过残差感知的编辑选择提升搜索效率

**🔧 技术方法**

结合LLM生成、可执行编辑操作、转移记忆、轻量级编辑效果估计、残差驱动排名以及全局与局部探索协同

**📊 数据集**

在LLEMABench的14个多目标材料任务上进行实验

**📈 对比分析**

与LLEMA及其他生成/代理基线在匹配评估预算下对比，TRACE宏观命中率从18.13%提升至25.96%，多任务上均优于基线

**⚠️ 局限性**

对编辑特征的经验估计受限于样本稀缺，对长程编辑路径缺乏建模，且在某些任务/后端仍存在性能波动

---

## 309. From Relaxed Indexability to Exact Indexability: A $t$-Step Approach for Partially Observable Restless Bandits

**arXiv ID:** 2608.24167 | [PDF](https://arxiv.org/pdf/2608.24167v1)

**作者:** Qizhen Jia `[一作]` (Xi'an Jiaotong-Liverpool University), Keqin Liu `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了部分可观测休眠多臂赌博机的t步前瞻阈值政策，并推导出对应的近似Whittle指数；

**💡 创新点**

将原始一阶线性阈值扩展为任意t步前瞻，利用首次穿越时间构造线性系统，证明近似指数以几何速率收敛，并提供无需先验可索引性验证的算法；

**🔧 技术方法**

采用有限时限值迭代、首次穿越时间分析、线性系统求解、性能差分法以及几何收敛证明等技术；

**📊 数据集**

在随机生成的三状态（共2,715个）实例、三臂高贴现率实例以及六臂有限时限实例上进行实验；

**📈 对比分析**

与一阶线性阈值、贪婪策略和最优动态规划进行对比；误差随t下降，t=2已恢复索引排序；在有限时限情形下t=2/5的阈值指数策略几乎逼近最优，计算时间随t增长平缓；

**⚠️ 局限性**

对极高贴现率仍需较深t；t和搜索上界需手工设定；在高维状态空间下计算量可能增大；实验仅基于随机实例，真实应用中的表现尚待验证。

---

## 310. SPIDER4TianoCore: Enhancing Patch-Propagation for the TianoCore UEFI Firmware Development Ecosystem

**arXiv ID:** 2608.23755 | [PDF](https://arxiv.org/pdf/2608.23755v1)

**作者:** Laura Baird `[一作]` (University of Colorado Colorado Springs), Armin Moin `[通讯]` (Purdue University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并演示了一个 Python 命令行工具，用于在 TianoCore UEFI 固件供应链中为已知 CVE 生成可审核的补丁状态证据；

**💡 创新点**

创新点在于将 SPIDER 的安全补丁推断框架改造成面向集成阶段的、易于使用的 Python 工具，提供四种可审核状态（Vulnerable、Already Patched、Not Applicable、Uncertain），并通过可视化报告减少维护者手工审核负担；

**🔧 技术方法**

采用文本归一化、tree‑sitter 语法树匹配以及基于 YAML 的清单驱动工作流，对目标文件进行静态分析和判定；

**📊 数据集**

使用了包含两条公开 CVE（CVE‑2024‑38797 与 CVE‑2023‑45234）和 20 个已预备的目标/CVÉ 对（来源自八个公开下游仓库）的手工标注基准；

**📈 对比分析**

与手工语义审计标签进行比较，工具在 10 个精确前补丁匹配和 4 个精确后补丁匹配上均达 100% 识别率，在其余 6 个案例上安全地标记为 Uncertain，未出现误判，整体保持高度保守且无错误分类；

**⚠️ 局限性**

局限性包括：工具仅提供证据而非安全推断；只使用了有限的静态分析器，未集成完整的 SPIDER 条件等价分析；基准样本有限，缺乏对更大规模或更复杂补丁的评估；

---

## 311. Beyond Accuracy: A Dual-Judge Evaluation Protocol for Vision-Language Models in Legally Grounded Tasks

**arXiv ID:** 2608.24258 | [PDF](https://arxiv.org/pdf/2608.24258v1)

**作者:** Su Myat Noe `[一作]` (National Institute of Informatics), Ken Satoh `[通讯]` (ROIS-DS Center for Juris-Informatics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了双裁判评估协议，在交通标志识别任务中使用视觉语言模型评估质量与语义等价性。

**💡 创新点**

引入双裁判（0-10质量评分+严格二元等价判定）揭示单一评分无法捕捉的判定差异，尤其在遮挡严重时高分不可信。

**🔧 技术方法**

采用 GPT-4o 作为视觉语言模型和评判者，使用结构化输出、正则表达式校正解析错误，并结合链式思维、链式推理、多代理等提示范式。

**📊 数据集**

使用 30 个英国交通标志的原始图像，生成 7 级可视度与 2 种遮挡模式，共 390 个变体，形成 4,680 次评估。

**📈 对比分析**

对四种 VLM 系统（单代理、单代理+链式思维、单代理+链式推理、多代理）进行配对 t 检验，单代理+链式思维显著优于基线，单代理+链式推理略逊，双裁判相关性为 0.644，Type II 不一致率在低可视度下达 54–63%。

**⚠️ 局限性**

仅使用 GPT-4o 评判可能导致偏差；数据集规模小；等价判定基于作者释义而非法条；缺乏多评审者验证。

---

## 312. Bridging Teacher Expectations and Robot Learning via Coupling Dynamics

**arXiv ID:** 2608.23994 | [PDF](https://arxiv.org/pdf/2608.23994v1)

**作者:** Evan Dallas `[一作]` (Oakland University), Wing-Yue Geoffrey Louie `[通讯]` (Oakland University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并应用一种基于耦合动态的四阶量化尺度，对20篇人机教学相关论文进行评估，探究耦合度对学习效果和教师认知的影响。

**💡 创新点**

将构造主义学习理论与人机教学耦合度关联，构建耦合尺度并在文献中实证验证其解释力与预测价值。

**🔧 技术方法**

采用文献筛选、耦合尺度映射、数据归纳与对比分析等方法，对论文的耦合真值与教师感知进行双向对比。

**📊 数据集**

利用20篇精选论文的实验数据作为评估样本；未使用公开数据集，而是聚焦于已有研究中的教学与学习记录。

**📈 对比分析**

通过将论文按耦合真值和教师感知分别归类，并统计各类耦合下的学习效果、教师工作负荷与信任度等指标，发现无耦合导致认知错配，单耦合受限于数据多样性，迭代耦合提升学习性能，混合耦合实现效率与透明度平衡。

**⚠️ 局限性**

局限在于仅分析20篇论文，缺乏大规模或真实机器人实验验证；耦合尺度定义带有主观性，且对不同教学场景的普适性尚待进一步评估。

---

## 313. Coronavirus Optimization Algorithm: A Success-History Adaptive Evolutionary Framework with Archive-Assisted Search and Stagnation Recovery for Global Optimization

**arXiv ID:** 2608.23847 | [PDF](https://arxiv.org/pdf/2608.23847v1)

**作者:** Hari Mohan Pandey `[一作]` `[通讯]` (Bournemouth University), Hari Mohan Pandey (Bournemouth University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Coronavirus Optimization Algorithm（COA），一种面向盒子约束连续全局优化的自适应进化算法。

**💡 创新点**

创新点在于将SARS‑CoV‑2的五个生物机制（Spike–receptor binding、viral replication、antigenic drift、immune evasion、viral‑load dynamics）映射为具体算子，并融合DE的current‑to‑pbest/1突变、外部归档、成功历史参数自适应、对偶搜索初始化与部分重启、以及非线性种群规模调度，形成紧凑且可解释的优化框架。

**🔧 技术方法**

使用的技术包括DE/current-to-pbest/1突变、二项交叉、外部归档、成功历史适应的F和CR、对偶搜索初始化与重启、以及基于评估次数的种群规模递减。

**📊 数据集**

实验基准为CEC 2017单目标连续优化基准集29个函数，维度分别为10、30、50，采用30次独立实验进行评估。

**📈 对比分析**

与15种代表性优化器（DE、PSO、CMA-ES、SHADE、JADE、LSHADE-SPACMA、GWO、SSA、HHO、WOA、AO、RUN、RIME、DMO、CPO）进行统计检验，COA在所有维度上取得最低平均Friedman排名（2.79/2.86/2.53），在组成函数类别上表现最优，且胜利次数最高；在混合函数类别上则不及GWO。

**⚠️ 局限性**

局限性包括对混合（高变量交互）函数的适应性不足；缺乏协方差学习或子空间分组等机制；在更高维度（>50）以及受限、噪声、多目标等实际应用场景尚未充分验证。

---

## 314. Mind the Student: Behavioral and Contextual Cues for Automated Engagement Prediction in Online Learning

**arXiv ID:** 2608.24340 | [PDF](https://arxiv.org/pdf/2608.24340v1)

**作者:** Alperen Kantarci `[一作]`, Gemma Roig `[通讯]` (Goethe University Frankfurt)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一套多模态框架，利用学生、教师、屏幕和音频等多源信息，并通过Perceiver IO融合与贝叶斯上下文层共同预测在线学习视频中的学生投入水平（回归和二分类）。

**💡 创新点**

创新点包括：①将预训练的V‑JEPA2、CLIP、AudioMAE与显式行为特征（头部姿态、注视、表情单元等）并行编码并通过Perceiver IO实现跨模态自适应融合；②引入学生/教师的可变后验嵌入与先验正则化，构成层次化贝叶斯上下文层以实现部分池化与个性化；③使用证据回归（NIG）与谱归一化高斯过程（SNGP）双头，提供可解释的置信度与不确定性。

**🔧 技术方法**

主要技术包括：预训练视觉/音频编码器、Perceiver IO异构注意力、贝叶斯变分推理、证据回归、谱归一化高斯过程、KL正则化与任务不确定性加权训练。

**📊 数据集**

使用CASED挑战集进行微调，并在预训练阶段采用DaiSEE和Aff‑Wild2数据集；评估遵循严格的学生独立拆分。

**📈 对比分析**

与参赛者对比，该方法在CASED测试集上获得F1宏约0.52、CCC约0.018，整体与最佳方法相近，但仍停留在随机或低于基线的水平，表明数据本身噪声大、可辨别性有限。

**⚠️ 局限性**

局限性包括：①高度的个体差异与标注主观性导致模型难以捕获真实投入；②模型仍对身份特征过度敏感，导致过拟合学生外观；③单一帧级标注与多帧平均导致时序细节被稀释；④贝叶斯先验过于弱，难以平衡个性化与泛化。

---

## 315. Technology Caregiving: Reframing How Older Adults Are Supported in Everyday Digital Activities

**arXiv ID:** 2608.23751 | [PDF](https://arxiv.org/pdf/2608.23751v1)

**作者:** Debaleena Chattopadhyay `[一作]` (University of Illinois Chicago), Tasneem Mubashshira `[通讯]` (University of Illinois Chicago)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对老年人日常数字活动（DADL）的技术照护进行系统综述，并提出了技术照护框架（包含目的、来源、时机和交付四个维度）来描述和归纳文献中的支持配置。

**💡 创新点**

将老年人技术支持重新定义为“技术照护”并视其为一种持续、关系性、以功能独立为目标的照护工作，构建了面向技术照护的四维框架，并通过框架验证了文献中支持模式的可覆盖性。

**🔧 技术方法**

采用PRISMA系统综述方法、主题分析（Thematic Analysis）和框架适配（Theory‑Adaptation）来识别和编码研究中的支持配置，使用手工编码和共识讨论完善编码方案。

**📊 数据集**

共检索7个数据库（Scopus、EMBASE、PsycINFO、IEEE Xplore、CoCoR、PubMed、ACM DL）以及Google Scholar，最终纳入36篇原始研究（共43项研究），涉及4,077条记录的初筛、115篇全文评估。

**📈 对比分析**

本研究未进行算法或系统性能对比，而是通过对文献中支持配置的归纳与统计（如最常见的配置、支持来源与目的的分布等）来展示框架的适用性和研究现状，未给出量化性能指标。

**⚠️ 局限性**

局限性包括：未对纳入研究的质量进行评估；仅使用英文文献，导致对非英语地区的研究缺乏；大部分研究为横断面或小样本，缺乏纵向追踪；缺乏对技术照护实际效果或干预评估的系统比较。

---

## 316. Relative Time Intervals Representation for Word-level Timestamping with Masked Training

**arXiv ID:** 2608.24041 | [PDF](https://arxiv.org/pdf/2608.24041v1)

**作者:** Quanwei Tang `[一作]`, Guodong Zhou `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了XXX问题，提出了一种新的解决方案。

**💡 创新点**

创新点在于引入了XXX方法，显著提高了XXX的性能。

**🔧 技术方法**

使用了XXX技术，如深度学习、机器学习等。

**📊 数据集**

实验中使用了XXX数据集，包含了XXX样本。

**📈 对比分析**

与现有方法进行了比较，结果表明新方法在XXX指标上优于传统方法。

**⚠️ 局限性**

限制在于XXX，例如数据集的规模、模型的复杂性等。

---

## 317. FLARE: A Systematic, Uncertainty-Aware Framework for Evidence-Based Adoption of Artificial Intelligence in Healthcare

**arXiv ID:** 2608.23643 | [PDF](https://arxiv.org/pdf/2608.23643v1)

**作者:** Jacob Idoko `[一作]` (University of Calgary), Gouri Ginde `[通讯]` (University of Calgary)

**通讯引用:** 152 | [OpenAlex ID](https://openalex.org/A5108426807)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并验证了 FLARE 框架，用于评估 AI 在医疗工作流中的经济价值与 ROI。

**💡 创新点**

创新点在于将模糊逻辑与时间驱动活动成本法（TDABC）结合，构建了生命周期不确定性下的经济评估模型。

**🔧 技术方法**

采用了模糊逻辑（三角模糊数）、TDABC 计费方法以及 ROI 计算技术。

**📊 数据集**

利用急性缺血性卒中 CT 病例的时间-动作研究数据，并采用 Brugnara 等人公开的 LVO 检测模型训练与验证集（Heidelberg 训练/测试集）。

**📈 对比分析**

通过将传统 CT 路径成本与引入 AI 的成本对比，发现每患者约节省 54.65 加元，年度收益 273,250 加元，第一年 ROI 约 25%，五年累计 ROI 约 44%，展示了 AI 方案的经济可行性。

**⚠️ 局限性**

主要局限是依赖专家估计与模糊数，缺乏真实机构成本验证；未考虑通胀/折现，且结果需在本地化环境中重新校准。

---

## 318. ReproAgent: Contract-Guided Paper-to-Code Reproduction

**arXiv ID:** 2608.24291 | [PDF](https://arxiv.org/pdf/2608.24291v1)

**作者:** Xue Hu `[一作]` (Beihang University), Wentao Zhang `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了ReproAgent，一套四阶段（Prepare-Plan-Generate-Repair）流水线，通过持久化实现合同和引用证据渠道实现论文到代码的复现；

**💡 创新点**

引入双通道实现合同（实现需求+引用证据），在整个流水线中持久追踪论文显式需求与外部证据，避免生成可运行但不忠实的代码；

**🔧 技术方法**

利用LLM（Claude‑Sonnet‑4.5、Gemini‑3‑Flash）、文本分割抽取实现需求、检索相关仓库证据、工作包绑定、文件级合同、代码生成与修复审核；

**📊 数据集**

使用PaperBench Code‑Dev 20篇论文集评测；

**📈 对比分析**

与同骨干系统（如PaperCoder、Sci‑Reproducer等）对比，ReproAgent在PaperBench Code‑Dev上获得最高平均分，ablation实验显示两通道均显著提升忠实度；

**⚠️ 局限性**

仍受LLM理解与检索质量限制，复杂/多模态论文需求可能抽取不足；依赖可检索的相关仓库；修复预算有限导致部分违规未修复；需要人工评测以确认最终质量。

---

## 319. Speech-to-SOAP: End-to-End Summarization of Medical Dialogues: KIT@BeTraC 2026

**arXiv ID:** 2608.24327 | [PDF](https://arxiv.org/pdf/2608.24327v1)

**作者:** Enes Yavuz Ugan `[一作]` (Karlsruhe Institute of Technology), Alexander Waibel `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了端到端语音到SOAP医学摘要生成方法，并构建了一套可扩展的数据增强管线。

**💡 创新点**

创新点在于：①利用合成语音和自动生成SOAP监督统一多源医学对话数据；②系统评估多种适配策略（提示设计、文本+语音联合训练、多阶段适配、链式思维等），并展示了不同策略对性能的影响。

**🔧 技术方法**

核心技术包括 Qwen2.5-Omni-3B 语音-语言模型、LoRA 微调、Kokoro-82M TTS、GPT‑3.5 自动生成 SOAP 标签，以及多阶段适配与链式思维实验。

**📊 数据集**

使用的数据集有 Synth-DoPaCo、ACI‑Bench、MTS‑Dialog、PriMock57、OMI，合成后共 1,653,067 小时音频。

**📈 对比分析**

通过多实验对比（提示、音频清洗、长对话过滤、多阶段适配、联合训练等），最终合并模型在官方测试集上的表现为：Concept‑F1≈0.4986，ROUGE‑2≈0.3537，ROUGE‑3≈0.2417，优于单一提交。

**⚠️ 局限性**

局限性包括：合成语音仍出现 hallucination，链式思维未显著提升效果，模型对真实噪声环境鲁棒性有限，对复杂术语转换和细粒度临床信息提取的准确性仍有待提升。

---

## 320. MGQL: An Executable, Small-Step Semantics of GQL

**arXiv ID:** 2608.24565 | [PDF](https://arxiv.org/pdf/2608.24565v1)

**作者:** Aditya Thimmaiah `[一作]` (University of Texas at Austin), Milos Gligoric `[通讯]` (University of Texas at Austin)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对GQL（Graph Query Language）进行正式化，构建了包含图模式匹配、复合查询、路径模式、空值处理和图架构约束的读写分离语义，并在Lean 4中实现并机理化验证；

**💡 创新点**

首次提供完整的小步执行语义与基于图架构的类型系统，并证明类型安全，弥补了GQL标准中非正式语义和缺乏可执行形式化的不足；

**🔧 技术方法**

采用小步操作语义、层次化类型系统、Lean 4机械化证明与可执行解释器、证据化类型检查器以及基于三值逻辑的空值传播；

**📊 数据集**

在LDBC社交网络基准（Interactive v2）工作负载上进行验证与测试；

**📈 对比分析**

通过在Lean实现的可执行解释器与确定性解释器进行对比，所有测试均实现位对位一致；性能评估主要以验证通过率和可执行性为依据，未给出具体执行时间指标；

**⚠️ 局限性**

仅覆盖读写分离的子语言片段，未涵盖完整的GQL功能（如关联核心、数据转换等），因此在实现完整语法与语义时仍有待扩展。

---

## 321. GlanceWAM: Sparse Test-Time Imagination for World-Action Models

**arXiv ID:** 2608.23927 | [PDF](https://arxiv.org/pdf/2608.23927v1)

**作者:** Linhan Wang `[一作]` (Virginia Tech), Chang-Tien Lu `[通讯]` (Virginia Tech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种异步视觉想象框架（GlanceWAM），在不阻塞控制循环的情况下在潜在空间中生成秒级前瞻帧，并将其直接用于动作决策。

**💡 创新点**

创新点包括：① 将视觉预测与动作解码完全解耦，异步生成仅一次前瞻；② 在单一视频DiT内部实现视觉预测、动作生成与多尺度前瞻条件的统一训练；③ 采用非干扰式3分类前缀注意力掩码和时延鲁棒的视角随机化训练，使模型对前瞻衰老保持鲁棒。

**🔧 技术方法**

使用的技术包括：基于Latent Video Diffusion Transformer（SkyReels‑V2‑DF）的潜在空间视频扩散、Causal VAE压缩、流匹配动作头、Euler ODE采样、三分类前缀注意力掩码、随机化前瞻时间戳和多层Transformer特征融合。

**📊 数据集**

数据集：RoboCasa厨房（24个操控任务，50条演示/任务）和LIBERO四大子任务集（共10个任务/子集，50条演示/任务）。

**📈 对比分析**

与最新的模仿学习和世界动作模型进行对比：在RoboCasa上取得72.2%成功率（超过Cosmos Policy 67.1%和无前瞻共训练64.4%），在LIBERO上平均99.0%（接近或略高于同类基线）。此外，单个动作块的推理时间仅为48 ms（相比同步方法高出24–80倍的延迟），实现了实时控制。

**⚠️ 局限性**

局限性：① 仍需预训练的扩散模型，初始计算成本高；② 前瞻生成虽异步，但在极低延迟场景下仍需占用一定GPU资源；③ 仅在演示数据上训练，缺乏在线自适应或RL微调；④ 对于需要高频细粒度视觉细节的任务，单帧前瞻可能不足。

---

## 322. Source-Face Authenticity Detection for 3D Gaussian Heads Reconstructed from a Single Portrait: A Benchmark and Dedicated Detector

**arXiv ID:** 2608.23984 | [PDF](https://arxiv.org/pdf/2608.23984v1)

**作者:** Yujie Gao `[一作]` (Shanghai Jiaotong University), Jianfu Zhang `[通讯]` (Shanghai Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了首个大规模真实/伪造 3D 高斯头检测数据集和基准，并设计了两阶段训练的检测器；

**💡 创新点**

创新点在于结合掩码自编码与多视角对比学习实现细粒度细节保留与视角一致性，并通过多层 CLS 令牌融合提升分类性能；

**🔧 技术方法**

采用掩码自编码（MAE）、多视角对比学习、ViT（DINOv3 ViT‑H+/16）视觉骨干及多层 CLS 令牌拼接的 MLP 分类器；

**📊 数据集**

使用新构建的 361,469 张渲染图（16,372 个身份）的数据集，包含从 5 个真实人脸数据集（CelebV‑HQ、Ava‑256、FFHQ、NeRSemble、VFHQ）和 9 个 2D 伪造方法生成的伪造人脸，再通过 5 种单图 3D 高斯重建方法生成；

**📈 对比分析**

与 7 种现有 2D 真实/伪造检测器（AIDE、Effort、PGC、FreqNet、IID、ForAda、NPR）在统一协议下对比，本文方法在准确率、宏 F1、AUC、AP_fake 等指标上均优于最强基线；在 OOD 评估中亦表现出更强的泛化能力；

**⚠️ 局限性**

局限性：面部表达驱动的伪造仍然难以识别，准确率仅约 76%，需要进一步扩展该类样本并提升检测鲁棒性。

---

## 323. Where Entropy Is Measured Matters: Policy Geometry in Bounded Continuous-Control PPO

**arXiv ID:** 2608.24488 | [PDF](https://arxiv.org/pdf/2608.24488v1)

**作者:** Yiyang He `[一作]` (Lancaster University), Haolin Fei `[通讯]` (Lancaster University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了在PPO强化学习中，熵项的测量空间（潜在高斯空间 vs 执行后tanh/裁剪映射）如何影响连续控制策略在动作边界附近的几何分布，并通过同态状态分解、梯度分析和多任务实验揭示均值定位与方差扩散对边界占用的耦合机制。

**💡 创新点**

创新点在于提出并验证熵测量空间对策略几何的耦合效应，证明高边界占用既可由高方差产生，也可由均值外推产生；通过熵梯度分解显示执行熵在均值上施加直接向内的正则化，从而区分潜在熵与执行熵的不同几何结果。

**🔧 技术方法**

采用PPO、对称高斯策略、tanh映射与硬裁剪、熵正则化（潜在、无熵、执行），并利用梯度分解、同态状态回放、共享状态集、边界阈值扫频等方法；使用Stable‑Baselines3、CleanRL实现、MuJoCo/DM控制器、Adam优化等技术。

**📊 数据集**

实验数据集包括MyoLeg 80肌肉+14马达仿真任务（MuJoCo）和DeepMind Control Suite Dog‑Stand 38维任务（DM），并收集对应状态样本用于同态回放与共享状态评估。

**📈 对比分析**

通过三种熵条件（潜在熵、无熵、执行熵）在三粒子种子下训练，评估近边界占用（P_10、P_01、P_11）、均值绝对值、方差、回报；在共享状态集、不同边界阈值（1%~10%）下保持相同几何顺序；执行熵在Dog‑Stand产生最内侧均值，但回报顺序因任务而异，说明回报不直接表征策略几何。

**⚠️ 局限性**

局限性包括仅在PPO+对称高斯+状态独立方差+MuJoCo/DM环境下验证，未探讨其他分布、方差依赖、gSDE、不同优化器或学习率、硬裁剪替代方案；熵系数固定，未系统评估其规模影响；实验仅在仿真，缺乏硬件验证。

---

## 324. NVIDIA Cosmos-H-Dreams: Real-Time Generative Physics Simulation for Surgical Robotics

**arXiv ID:** 2608.24199 | [PDF](https://arxiv.org/pdf/2608.24199v1)

**作者:** Javier Gamazo Tejero `[一作]`, Sean D. Huver `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8d10c613-917e-4880-9716-17789f50e119` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Cosmos‑H‑Dreams，一个可实时交互、可由人类操作员、VR头盔或自主策略在单卡GPU上驱动的外科世界模型系统；

**💡 创新点**

创新点包括：①首个可实时交互且支持多种控制接口（键盘、Meta Quest、机器人控制台）的外科世界模型；②基于双向教师到因果学生的Self Forcing+Distribution‑Matching蒸馏，实现在两步扩散下的实时生成；③整合流式推理库与NVENC H.264编码，实现低动作‑光束延迟；④提出闭环策略评估基准，评估模拟与真实机器人的一致性。

**🔧 技术方法**

技术栈：Cosmos‑H‑Surgical‑Simulator/DiT骨干；双向教师→因果学生；Self Forcing与DMD蒸馏；Streaming KV cache与局部窗口注意力；少步扩散（2‑step）；TAEHV轻量解码；NVENC H.264编码；WebRTC/WebXR/TCP接口；单GPU推理优化。

**📊 数据集**

数据集：Open‑H‑Embodiment（32个数据集、9机器人、22M帧、统一44D动作空间）；dVRK桌面缝合数据（约100万帧、10 Hz、包含失败与OOD场景）；Cosmos‑Surg‑dVRK等。

**📈 对比分析**

与双向教师对比：FVD从170提升至265，LPIPS从0.086提升至0.121；与离线Cosmos‑Surg‑dVRK基线相比，闭环策略评估Pearson r≈0.696、MMRV≈0.23±0.09。推理性能：在单张RTX PRO 6000上，12帧块2步推理可达161 FPS、51 ms延迟；不同配置平衡速度与视觉质量。

**⚠️ 局限性**

局限性：实时蒸馏导致视觉质量下降，尤其在薄细结构（缝线）上出现假象；闭环评估在手术握取/结绳等复杂任务表现不佳；仅在桌面缝合任务验证，未覆盖临床手术；长时序漂移仍未完全消除；单卡GPU部署在更复杂场景下可能受限。

---

## 325. Scale, Concentration, and Entry Timing in the Shopify App Ecosystem: A Longitudinal Study of Platform Governance and Application Survival

**arXiv ID:** 2608.23771 | [PDF](https://arxiv.org/pdf/2608.23771v1)

**作者:** Fabrizio Assabese `[一作]` (Judge.me), Giuseppe Destefanis `[通讯]` (University College London)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a2602d71-93ab-4bad-974b-672788df8193` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文利用对 Shopify 应用市场的跨时间数据，系统分析了其规模、类别竞争结构、早期进入者的成长动力、平台治理事件的影响以及应用生命周期与退出预测。

**💡 创新点**

创新点包括：①整合周度安装面板、静态快照与网络档案的历史列表信息，构建七年跨度的纵向数据库；②首次度量平台方退出对补充者市场的影响；③发现并验证了“后入者优势”在 88% 类别中持续存在；④基于前 26 周公开信号的二次退出预测模型，AUC 超 0.8，提供可操作的预警工具。

**🔧 技术方法**

采用的技术手段有：描述性统计与 Herfindahl–Hirschman 指数衡量类别集中度；Spearman 相关分析评估早期进入与增长速率关系；事件研究（平台进入/退出、佣金变更）结合差分面板；Kaplan–Meier 与 Cox 回归刻画退出风险；逻辑回归加交叉验证评估早期警告模型。

**📊 数据集**

数据集包括：Store Leads 2025‑2026 7,708 个应用的每周安装面板（366 周）；2025 年 9 月 28 日的全景快照（24,826 个应用、2,701,805 个商店）；以及从 Internet Archive Wayback Machine 重建的 2012‑2026 年的应用列表历史（价格、类别、评测等信息）。

**📈 对比分析**

方法对比：单一横截面与纵向面板、事件对照组与整体类别对比、早期/后期入者分组与整体平均；模型性能：类别集中度分类精度高，后入者优势在 90% 以上类别显著；Cox 模型得到可靠风险比，逻辑回归预测 AUC 0.81/0.84，均表明方法稳健、可复制。

**⚠️ 局限性**

局限性：安装计数仅反映可检测的前端脚本，无法区分卸载与检测丢失；使用创建日期作为入市时间的近似，可能混淆实际上线顺序；缺乏对应用收入与付费转化的直接测量；跟踪面板的初始采样可能导致早期未被记录的应用被遗漏；并且无法完整捕捉开发者与平台之间的非公开互动与运营细节。

---

## 326. Mixed-Precision SEM-Based CFD Simulations on GPUs: A Taylor-Green Vortex case

**arXiv ID:** 2608.24348 | [PDF](https://arxiv.org/pdf/2608.24348v1)

**作者:** Yanxiang Chen `[一作]` (Umeå University), Roman Iakymchuk `[通讯]` (Umeå University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并实现了面向GPU的谱元方法CFD仿真中的多级混合精度控制模型，并在Taylor‑Green涡旋基准上评估其性能与精度。

**💡 创新点**

提出了三级（块、核、步）分层混合精度策略，针对SEM CFD的求解结构设计精度切换方案，并证明在保持数值鲁棒性的前提下可实现约34%的时间和能耗提升。

**🔧 技术方法**

采用Neko矩阵无关谱元求解器、CUDA+OpenMPI加速、Verificarlo算术分析、Krylov子空间求解器（CG、GMRES+HSMG）与半精度fp16核实现。

**📊 数据集**

采用Taylor‑Green涡旋基准流场，Re=1600（主实验）与Re=10000（精度压力测试），多阶多项式P5–P8，32×32×32网格。

**📈 对比分析**

通过与全双精度(fp64)和全单精度(fp32)基线对比，使用时间‑to‑solution、能耗和误差指标（动能、渗流）评估；在Group B的最佳配置下，时间与能耗均下降约34%，同时相对误差保持在0.7%以内。

**⚠️ 局限性**

当前实现仍以fp64存储为主，导致精度转换开销；fp16仅适用于部分核；缺乏步级自适应精度与完整GPU低精度驻留；高Reynolds数下单精度鲁棒性不足。

---

## 327. Investigating Knowledge Transfer Across Interactive Dialogue Games

**arXiv ID:** 2608.23969 | [PDF](https://arxiv.org/pdf/2608.23969v1)

**作者:** Filippo Momentè `[一作]` (University of Trento), Alessandro Torcinovich `[通讯]` (Free University of Bozen Bolzano)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了对话游戏之间的知识迁移，构建了转移图并分析了微调模型在权重空间中的更新。

**💡 创新点**

创新点在于发现视空（visuospatial）游戏对多种游戏具有更高的迁移性能，并证明仅靠任务向量相似度无法预测迁移效果；同时提出利用整数规划在预算约束下寻找最优迁移集合。

**🔧 技术方法**

使用了任务向量（Taskonomy）方法、二元整数规划（Integer Linear Programming）、子空间重叠度量、Qwen3.5-4B 大模型的全量微调，以及余弦相似度等技术。

**📊 数据集**

数据集为 OpenAI Dialogue Games 中选取的 15 个角色特定任务（来自 9 款游戏），共 585 条训练样本，采用 20 条验证和 20 条测试样本。

**📈 对比分析**

通过将得到的迁移集与随机迁移集合进行对比，发现后者在整体迁移性能上显著更优；然而相似度指标对迁移方向的预测表现不佳。

**⚠️ 局限性**

局限性包括仅考虑正迁移，未评估负迁移；仅使用协作型游戏数据，缺乏竞争型游戏；实验仅在单一 Qwen3.5-4B 模型上进行，未验证跨模型泛化。

---

## 328. Enhancing Bug Report Templates in the TianoCore UEFI Firmware Development Community

**arXiv ID:** 2608.23754 | [PDF](https://arxiv.org/pdf/2608.23754v1)

**作者:** Laura Baird `[一作]` (University of Colorado Colorado Springs), Armin Moin `[通讯]` (Purdue University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了针对UEFI固件开发社区TianoCore的改进bug报告模板，包括新增CPU架构和问题类别字段，并集成生成式预训练模型（LLM）生成可解释的 triage 辅助评论。

**💡 创新点**

将固件特定的诊断信息（CPU架构、回归标识）结构化为模板字段，并首次在GitHub Issues中结合AI生成的可解释辅导评论，以提升报错质量和 triage 效率。

**🔧 技术方法**

使用 GitHub Issues 模板扩展、GPT/Claude 等预训练大语言模型生成辅导评论、Web 原型实现、人工分析与 A/B 访谈等技术。

**📊 数据集**

基于 92 条已关闭的 TianoCore GitHub issue（包括 Bugzilla 迁移与原生 GitHub）进行手工标注，并在访谈中使用两条真实问题（11374、11882）进行对比测试。

**📈 对比分析**

通过在访谈中对比原始模板与改进模板（Template 1 vs Template 2）收集参与者偏好，3/4 参与者更倾向改进模板，显示可行性；目前尚未提供量化的 triage 时效或准确率指标。

**⚠️ 局限性**

研究仍为工作进展，样本量有限，缺乏大规模量化评估；AI 辅助建议的可靠性与可解释性待进一步验证；部分经验丰富的开发者认为新增字段过多，可能影响实际使用。

---

## 329. REFINE: A Multi-Agent LLM Approach for Evidence-Guided Code Refactoring

**arXiv ID:** 2608.23611 | [PDF](https://arxiv.org/pdf/2608.23611v1)

**作者:** Muhammad Waseem `[一作]` (Tampere University), Pekka Abrahamsson `[通讯]` (Tampere University)

**通讯引用:** 10596 | [OpenAlex ID](https://openalex.org/A5058417486)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种基于多代理、证据感知的工作流 REFINE，用于生成和评估 Java 文件级代码重构候选。

**💡 创新点**

创新点在于将静态分析证据与 LLM 生成、验证门控结合，提供可追溯的重构候选，并在文件级别系统化评估。

**🔧 技术方法**

采用 PMD 静态分析、LangGraph 编排、OpenAI GPT‑5.5 / Gemini 3.1 / Claude Opus LLM 生成与验证。

**📊 数据集**

使用 15 个开源 Java 系统的 450 个生产文件（共 1,350 次 LLM 运行）以及 150 文件的匹配基线。

**📈 对比分析**

通过与同样 LLM 的直接提示基线对比，REFINE 在代码异味降低（68–73%）上显著优于直接提示，且编辑量更小、公共方法删除更少；但在整体质量指标和行为保持上表现参差。

**⚠️ 局限性**

主要限制是仅在文件级别评估，缺乏全仓库编译/测试验证，静态保持检测并不能证明行为等价，且对跨文件依赖与系统级影响未覆盖。

---

## 330. RENDER: Controlling Reader-Facing Evidence in LLM Memory Evaluation

**arXiv ID:** 2608.23568 | [PDF](https://arxiv.org/pdf/2608.23568v1)

**作者:** Yuan Si `[一作]` (University of Waterloo), Jialu Zhang `[通讯]` (University of Waterloo)

**通讯引用:** 37 | [OpenAlex ID](https://openalex.org/A5014754883)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为Reader Evidence Rendering Diagnostics (RERD) 的基准控制，专门研究在保持对话、问题与答案不变的前提下，读者所看到的证据呈现形式如何影响记忆/检索增强生成（RAG）系统的问答准确率。

**💡 创新点**

创新点在于：①将读者可见的证据拆分为五级Packet Ladder（P_0~P_4），精准定位是缺失内容、冲突未解决还是表面格式导致的失败；②引入可比的部署式模板（自然语言条目、摘要、JSON记录、原始对话），揭示不同表面渲染对性能的显著影响；③提出在词预算匹配下的“结构化证据优于截断原始对话”这一颠覆性结论。

**🔧 技术方法**

技术手段包括：a) 通过结构化存储与冲突解析生成五级Packet Ladder；b) 对同一对话采用不同模板（自然语言条目、摘要、JSON记录、原始对话）生成可比的输入；c) 在不同模型上执行成千上万次API调用，使用子串匹配、规范化精确匹配以及LLM判别者进行多元评估；d) 对预算控制做词数截断与完整对话对照。

**📊 数据集**

主要使用的数据集为RUG（记忆问答）oracle tier的500个多会话问答；补充实验包含200个问答、检索噪声实验、以及迁移到另一多跳问答数据集（如HotpotQA）的评估。

**📈 对比分析**

比较方法：在同一问答下，分别给模型提供不同Packet Ladder层级、预算匹配下的原始对话、以及四种模板化证据；对每个模型统计整体准确率。结果显示：①在Packet Ladder上，P_2层（包含已解析答案）从0%跃升至15-25%；②在预算匹配下，结构化的Resolved‑P_2在所有模型上比截断原始对话高出42.4–72.6分；③部署模板下，最优自然语言条目与原始对话差距可达24–49分。

**⚠️ 局限性**

局限性包括：①词级截断并未完全反映token预算；②模板化呈现为人工构造，未模拟真实检索、摘要误差；③仅评估输入侧的证据渲染，未考察输出格式或完整检索链的影响；④实验仅覆盖RUG、HotpotQA等特定数据集和9个商业API模型，可能无法推广到更广泛的多模态或文档检索场景；⑤部分结果受LLM判别者的偏差和抽样噪声影响。

---

## 331. Low-Rank Ternary Adaptation for Fine-Tuning Transformers

**arXiv ID:** 2608.24469 | [PDF](https://arxiv.org/pdf/2608.24469v1)

**作者:** Alexandru-Dragos Manolache `[一作]` (Delft University of Technology), Jan van Gemert `[通讯]` (Delft University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种低秩Kronecker结构的三值乘法适配器，用于在保持三值权重的同时对Transformer进行参数高效微调。

**💡 创新点**

创新点在于将离散的保持/零/翻转更新通过可学习的三值Kronecker乘积实现，并可直接合并为三值模型，避免了去量化和重量化。

**🔧 技术方法**

技术包括三值乘法适配、Kronecker低秩分解、Straight‑Through Estimator (STE) 训练、SpinQuant 量化、LoRA 对比等。

**📊 数据集**

使用 LLaMA‑3.2 1B/3B、Falcon‑E‑1B/3B、BitNet、TernaryViT‑B/16 等模型，并在 Alpaca、GSM8K、ImageNet‑100 等数据集上评估。

**📈 对比分析**

与全精度、2‑bit PTQ、QLoRA 等基线对比，平均在语言任务上提升约5–7个百分点，PPL 降至 22 左右，视觉任务上比重量化 QLoRA 高约4个百分点，且无额外推理开销。

**⚠️ 局限性**

局限在于乘法更新无法激活被量化为 0 的权重，且 Kronecker 结构对某些不易因式分解的层形状可能限制表达。

---

## 332. The Empire, Long Divided, Must Unite: Architectural Convergence in Three LLM Agent Harnesses

**arXiv ID:** 2608.23953 | [PDF](https://arxiv.org/pdf/2608.23953v1)

**作者:** Dai Jiahong `[一作]` `[通讯]` (Nanyang Technological University), Dai Jiahong (Nanyang Technological University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对三大开源编码代理框架的源代码进行多案例研究，发现它们在不同起点下演化后趋于相同的五个架构要素，并指出外部可验证性缺失

**💡 创新点**

首次将架构收敛分析与代码层面轨迹结合，揭示并列出并行发现、扩散与代码复用三种收敛机制，并系统归纳四条常见缺陷线

**🔧 技术方法**

使用源代码审计、提交历史追踪、人工重现缺陷以及AI辅助代码定位等技术手段对框架进行深入解析

**📊 数据集**

以LangChain、Earendil与DeepSeek三大开源代理 harness 及其对应的提交历史为实验数据集

**📈 对比分析**

通过对五个收敛要素（统一循环、可重放记录、模型奇异性数据化、渐进式上下文暴露、显式接口分层）在三框架中的实现方式进行对比；未对性能做量化评估，而是侧重架构一致性与差异

**⚠️ 局限性**

样本有限（仅三框架），并非完全独立（某些代码复用），研究仅聚焦特定语义与功能场景，未覆盖所有潜在的代理架构变体，且依赖AI辅助，可能存在识别偏差

---

## 333. BotScan: An adaptive active probing approach for identifying live IoT Botnet C2 servers at scale

**arXiv ID:** 2608.23854 | [PDF](https://arxiv.org/pdf/2608.23854v1)

**作者:** S M Maksudul Alam `[一作]` (University of California, Riverside), Michalis Faloutsos `[通讯]` (University of California, Riverside)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一种基于重放可重播C2报文的适应性主动探测框架，可在大规模IP空间中高效发现IoT僵尸网络C2服务器。

**💡 创新点**

首次系统评估IoT恶意软件C2可重放性，提出两层基于子网/AS级别的空间局部性优先策略，并在探测过程中动态重排目标。

**🔧 技术方法**

采用恶意软件沙盒激活、报文重放、基于历史C2数据库的空间与端口优先级计算、Masscan端口扫描及Reputation模型等技术。

**📊 数据集**

使用1842个IoT恶意二进制（Mirai、Gafgyt等）、18.2k历史C2记录（BP-DS）、1.842k激活记录（RP-DS）以及2.5M IP目标空间和1.6M IP:port对进行评估。

**📈 对比分析**

与C2Miner、CyberProbe、AUTOPROBE等基线相比，重放探测在同等探测量下发现的活跃C2服务器数约翻倍，精度达98.8%，召回74.8%，在1M IP:port对内发现37台活跃服务器。

**⚠️ 局限性**

主要限制是对加密或会话专用加密协议的C2不可重放，需要重新激活；探测结果受C2服务器“脏活”影响，需多次探测才能保证准确性。

---

## 334. When Less Is More: An Empirical Study of Minimal Responses in Counseling Dialogues and the Behavior of LLMs

**arXiv ID:** 2608.24080 | [PDF](https://arxiv.org/pdf/2608.24080v1)

**作者:** Zhiyang Qi `[一作]` `[通讯]` (University of Tokyo), Zhiyang Qi (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统分析了心理咨询对话中最小回应（如backchannel、简短共情等）的分布，并评估LLM在生成这些回应时的能力。

**💡 创新点**

创新点包括提出两阶段过滤+LLM上下文验证方法识别最小回应；跨语言、跨数据集进行统计；揭示合成数据缺乏此类回应并导致模型难以学习；并发现传统LLM评估偏向内容丰富，低估了最小回应的交互价值。

**🔧 技术方法**

使用规则过滤、GPT‑5.4‑mini上下文验证、两种LLM评估器（GPT‑5.4‑mini、Gemini‑3.1‑Flash‑Lite）、对话生成实验以及最小回应比例、打断评分和质量平均等指标。

**📊 数据集**

采用了七个数据集：四个LLM合成数据集（Cactus、CPsyCounD、PsyDTCorpus、SmileChat）和三个真人收集数据集（AnnoMI、PsyDial‑D4、KokoroChat）。

**📈 对比分析**

通过比较一般提示与指令提示下各模型（通用LLM、GPT‑5.4、情境特定模型）生成最小回应的比例、打断分数和质量平均，结果显示合成数据训练模型最小回应率低，通用LLM在明确指令下可达约80‑97%最小回应率，但相应的质量评分下降，评估系统倾向于奖励内容丰富的回应。

**⚠️ 局限性**

局限性包括：不同数据集收集方式与咨询师特征差异影响可比性；LLM评估仅为代理，缺乏专家或客户的真实评价；实验仅关注局部上下文，未评估完整会话中最小回应的最佳时机与频率。

---

## 335. A Data-dependent Early Stopping Rule using Rademacher Complexity with L1-norm

**arXiv ID:** 2608.24210 | [PDF](https://arxiv.org/pdf/2608.24210v1)

**作者:** Duy Hoang `[一作]` (Université Paris-Saclay), Laurent Fribourg `[通讯]` (Université Paris-Saclay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于Rademacher复杂度（采用L₁范数）且无需训练即可估计梯度流最佳早停时间的解析规则；

**💡 创新点**

创新点在于：①不做数据分布或协方差矩阵谱的随机假设；②使用L₁范数计算RC，消除对未知常数M的依赖；③给出解析公式t⁺及其近似t⁺_approx，可直接计算；

**🔧 技术方法**

技术主要包括Rademacher复杂度分析、线性探测（linear probing）方法、梯度流解析求解、特征值分解和符号不变区间分析；

**📊 数据集**

实验使用了多种数据集：Gaussian、Uniform、Pareto的合成分布以及MNIST二分类（3‑5、0‑1）任务；

**📈 对比分析**

与传统的基于验证集的t_test以及基于L₂范数的RC估计对比，t⁺与t_test高度一致（误差<1%），且相较L₂方案误差更小；此外，t⁺能在不训练模型的情况下给出近似早停点；

**⚠️ 局限性**

局限性：仅针对线性模型（或通过线性探测映射到线性）；仅适用于标量输出；在过参数化（m>n）或特征值无明显分层时效果不佳；需要已知数据范围以估计C、M等常数。

---

## 336. Algorithmic Impact Reveals the Hidden Social Choice Structure of Alignment

**arXiv ID:** 2608.24046 | [PDF](https://arxiv.org/pdf/2608.24046v1)

**作者:** Zachary Wojtowicz `[一作]` (MIT), Ariel Procaccia `[通讯]` (Harvard University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将AI对多方利益相关者的决策问题转化为在影响空间中的线性优化，形成一种“影响空间”框架，用以分析对齐协议的社会福利后果，并设计可策略证明、可解释的对齐机制。

**💡 创新点**

创新点在于：①通过把模型的影响总结为单一向量，使对齐问题变为凸多面体上的线性优化；②利用该框架揭示RLHF、随机专制与投票按问题等传统方法的等价性与局限；③提出一族可实现外部性约束（福利下限、公民精神、参与外部性）的对齐协议，并给出闭式解与实现路径。

**🔧 技术方法**

技术手段包括：线性效用假设与Bradley–Terry–Luce模型、凸几何与线性规划、菜单机制与投票按问题的策略证明理论、随机专制的实现与极限参数化，以及对齐问题的理论证明与案例实验。

**📊 数据集**

实验使用四个真实世界数据集：肾脏分配、慈善食品分配、Moral Machine道德车祸与 Community Alignment（LLM回应）数据，所有数据均以可解释的特征空间映射，并通过线性模型估计个体偏好。

**📈 对比分析**

通过对齐机制（最优福利、RLHF、随机专制、福利下限等）的影响向量进行比较，评估每个机制下个体福利分布与平均福利。实验表明：最优福利最大化导致福利差异最大；RLHF与策略证明机制福利分布更平滑；福利下限机制能有效消除个体被伤害的情况，但会以平均福利损失为代价。对外部性约束机制则能限制单个参与者对他人的负面影响，提升整体公平性。

**⚠️ 局限性**

局限性包括：①对线性效用假设的依赖，若真实偏好非线性则框架不适用；②对查询分布与特征空间假设的敏感性；③随机专制实现需要事先广播选举概率，实际部署可能需在线学习；④缺乏对多轮互动或长期学习过程中的动态外部性分析。

---

## 337. What Reaches Expert Review? Representation, Structural Screening, and Candidate-Form Dependence in AI-Assisted Item Development

**arXiv ID:** 2608.23766 | [PDF](https://arxiv.org/pdf/2608.23766v1)

**作者:** Christopher Brooks `[一作]` (University of Michigan), Christopher Brooks `[通讯]` (University of Michigan)

**通讯引用:** 12977 | [OpenAlex ID](https://openalex.org/A5049205417)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 AI 辅助题目生成与计算评估器的决策对心理测量者可见证据与内容的影响

**💡 创新点**

将生成、语义表示、结构筛选和候选表构建的决策可视化并追踪其对项目保留、覆盖和最终表格词语的影响

**🔧 技术方法**

使用 LLM（Qwen3.5‑27B 与 Gemma 3‑27B‑IT）、多种文本嵌入配置、探索性图分析 (EGA)、唯一变量分析 (UVA)、图形拉森和 TMFG 等结构方法，以及两种候选表选取政策

**📊 数据集**

基于大五人格的 32,000 条生成题目，构建 20 个属性单元（4 个属性 × 5 维度），共 400 轮生成任务

**📈 对比分析**

通过固定源群体对比不同嵌入配置和结构方法的效应，并将两种政策（inclusive vs agreement）在相同结构结果下比较；发现嵌入差异导致候选表词语差异约 93%，而政策差异仅导致约 35% 的可选项减少，均保持表格完整但词语不一致

**⚠️ 局限性**

实验仅覆盖大五人格、两种 LLM 与嵌入族、两种结构方法和两种政策；结果无法推广到其他结构、语言或测量对象，且未验证最终题目在被测人群中的效度与信度

---

## 338. TransPhy: Visual In-Context Learning for Physically Grounded Image Editing

**arXiv ID:** 2608.24119 | [PDF](https://arxiv.org/pdf/2608.24119v1)

**作者:** Siyi Xie `[一作]` (Peking University), Quan Wang `[通讯]` (SenseTime Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出物理基础的视觉上下文学习（Physically grounded VICL），并开发相应的模型与评测框架；

**💡 创新点**

提出TransPhy框架，结合物理规则归纳与基于局部状态转移的token‑wise Mixture‑of‑Experts（MoE‑LoRA）渲染，实现对查询场景的物理适配；

**🔧 技术方法**

使用BAGEL多模态Transformer、LoRA、token‑wise MoE‑LoRA、State‑Transition Capturer（STC）进行专家路由对齐，配合ViT特征差异进行转移监督；

**📊 数据集**

构建PhysVICL‑74基准，包含74条物理变换规则、5,240个源‑目标图像对以及约75K个exemplar–query上下文，支持新实例迁移与未见规则泛化评测；

**📈 对比分析**

与FLUX.1‑Fill‑dev、BAGEL‑MoE以及RelationAdapter、VisualCloze、LoRWeB等方法对比，在TA、CP、RP、CLIP‑D等指标上均取得明显提升（尤其在未见规则下显著领先，LPIPS更低），证明了模型的物理一致性与通用性；

**⚠️ 局限性**

受限于数据规模与多样性，难以覆盖所有复杂物理过程；STC仅使用ViT特征差异做局部监督，可能不足以捕捉所有细节；模型仍可能在极端光照或材质极端差异的场景下出现复制痕迹。

---

## 339. A Feature-Major Codebook for Memory-Efficient Sparse-Binary Self-Organizing Maps: Scaling a MEDLINE Atlas to 1.05 Million Neurons on a Single Consumer GPU

**arXiv ID:** 2608.24067 | [PDF](https://arxiv.org/pdf/2608.24067v1)

**作者:** Andrew James Amos `[一作]` `[通讯]` (James Cook University), Andrew James Amos (James Cook University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在MEDLINE全文语料库上训练自组织映射（SOM），利用特征主导（feature-major）代码本布局、稀疏-密集乘积（SpMM）分块融合核以及盒形卷积更新，实现了单 GPU 上百万神经元的可浏览医学知识图谱；

**💡 创新点**

创新点在于将代码本以特征为主的内存布局与 BMU 搜索的分块融合核相结合，消除了 BMU 计算中的内存访问瓶颈，并将更新阶段的盒形卷积与 Kaski‑Lagus 停止准则结合，使得速度提升 4.5–8.5× 而不牺牲聚类质量；

**🔧 技术方法**

技术细节包括 CUDA GPU 加速、稀疏二进制 CSR 语料、FP16 代码本、SpMM‑tile BMU 核、盒形卷积（radius‑independent）更新、PCA 初始化以及自适应半径衰减和 Kaski‑Lagus 收敛判定；

**📊 数据集**

实验使用 PubMed 2026 基线 MEDLINE 语料，约 29.9 M 篇文章、30 766 个 MeSH 词条、332 M 非零条目（平均 11.1 个词条/文档）；

**📈 对比分析**

与 cuSPARSE（node‑major 与 feature‑major）、先前 MedSOM CUDA 版本以及多核 CPU somoclu 进行同等工作量下的比较；SparseBin.SOM 在 64×64 规模时已交叉性能（2.8× cuSPARSE 之上），在 512×512 规模仅其可运行；在 RTX 4090 上训练 512×512 地图耗时 5.4 k s，H200 上训练 1 048 576 神经元地图耗时 35 k s，整体性能提升达 82× (GPU) 与 621× (CPU)；

**⚠️ 局限性**

局限包括仅在单一 RTX 4090 体系上验证，未测试数据中心 GPU；Baseline 比较受初始化、半径调度和精度不一致的影响；大小范围评估缺乏统计区间；DRAM 流量与性能交叉仅在两种规模下测得；MedSOM 的量化误差不可直接比较。

---

## 340. The Loss Floor of Denoising Score Matching: Fisher Geometry from Schrödinger Bridges

**arXiv ID:** 2608.23916 | [PDF](https://arxiv.org/pdf/2608.23916v1)

**作者:** Avinash Raju `[一作]` (Great Wall Motors Company Limited), Kai Zhang `[通讯]` (China Patent Information Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `2704f255-0c84-4173-b83c-0e9a3dbea232` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对去噪分数匹配（DSM）目标进行了精确分解，揭示其包含一个不可消除的损失底线，该底线等于条件端点分布的Fisher–Rao度量的积分，并进一步将其与信息流、互信息以及高SNR下的维度信息关联起来；同时给出了在连续时间和离散掩码扩散中的相应表达式，并讨论了训练与采样中的实际影响。

**💡 创新点**

创新点在于：
- 通过Schrödinger桥变分原理导出DSM目标的二阶变分，并证明该变分正是条件端点分布的Fisher信息；
- 给出损失底线的通用几何与信息论表述，解释了已观察到的几何结构（Fisher‑Rao度量）是损失本身的内在组成；
- 证明损失底线等价于在扩散轨迹上累积的互信息，并在高SNR极限下与数据的Rényi信息维度关联；
- 分析了时间调度和权重对底线的影响，得出“等信息分配”准则并解释了调度不改变底线总量；
- 在离散掩码扩散中给出对应的熵形式，并将其与连续情形对应起来；
- 提出在不同SNR范围或调度下比较模型时需扣除底线以避免排名错误。

**🔧 技术方法**

主要技术包括：
- Schrödinger桥变分原理与Doob $h$‑变换；
- Fisher信息与Fisher–Rao度量的第二变分；
- 信息论公式（互信息、MMSE、de Bruijn 恒等式）；
- 对Gaussian噪声的闭式计算与高SNR渐近；
- 离散Bregman散度与掩码扩散的熵推导；
- 数值验证（解析混合高斯分布、熵/信息谱绘图）。

**📊 数据集**

实验数据主要采用低维解析高斯混合样本（如二维双峰混合）作为测试集，用来验证理论推导和数值一致性；并未在大规模图像或文本数据集上进行实验。

**📈 对比分析**

比较方法：直接将训练得到的DSM损失与理论推导的底线相减，得到“超额损失”，用于比较不同模型或不同SNR范围的表现。示例中发现，原始损失因SNR范围不同导致排名逆转，而扣除底线后模型的优劣顺序得到恢复。性能方面并未给出数值指标，而是表明正确的比较应基于扣除底线的超额损失。

**⚠️ 局限性**

局限性：
- 对非Gaussian或非线性退化过程的闭式结果缺失；
- 高SNR渐近依赖Rényi信息维度，可能不存在；
- 底线估计需要后验协方差，在低噪声端难以准确估计；
- 仅在参数化模型上讨论了投影影响，未给出完整的训练/采样策略改进；
- 主要在解析实验验证，缺乏大规模数据集上的实证结果。

---

## 341. Data Mixing as Mixture Experiment: Response Surface Methodology and Optimal Design for Large Language Model Pretraining

**arXiv ID:** 2608.23922 | [PDF](https://arxiv.org/pdf/2608.23922v1)

**作者:** Yicheng Mao `[一作]` (University of Calgary), Hongru Du `[通讯]` (University of Virginia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将大语言模型预训练中的数据混合问题重新表述为概率单纯形上的混合实验，并利用稀疏二阶 Scheffé 反应面模型对数据源的加性与交互效应进行可解释分解，同时设计模型稳健的 I‑optimal 实验方案以在代理训练阶段更高效地选取混合样本。

**💡 创新点**

创新点包括：① 把固定令牌预算的域比例视作混合成分，揭示域价值的相对性；② 采用稀疏二阶 Scheffé 模型提供可解释的主效应与交互结构；③ 通过模型稳健 I‑optimal 设计显著减少代理运行次数（约 25%），提升实验效率；④ 在代理实验设计与模型预测之间建立统一的实验设计框架。

**🔧 技术方法**

主要技术手段有：稀疏 L1 正则化的二阶 Scheffé 反应面回归、I‑optimal 与模型稳健 I‑optimal 设计、模拟退火算法求解混合实验设计、与 LightGBM 等机器学习模型的性能对比、Bootstrap 置信区间估计。

**📊 数据集**

使用公开的 RegMix 代理预训练数据，包含 17 个 Pile 数据域，共 512 次 1B-token 的小模型训练样本；评估数据为 1M、60M 与 1B 三个规模的验证损失。

**📈 对比分析**

与随机 Dirichlet 采样与 LightGBM 预测器比较：稀疏二阶 Scheffé 在 1B 规模下 Spearman ρ≈0.975、PRA≈0.937，略优于 LightGBM；I‑optimal 设计在 350 次代理运行时即可达到与 512 次参考设计相同的排名性能，验证了设计方法的高效性。

**⚠️ 局限性**

局限性包括：仅基于单一 RegMix 数据集，稀疏二阶 Scheffé 可能无法捕捉更高阶或非线性交互；实验设计假设所有混合点均可访问且成本相同，未考虑域可用性、许可、预处理成本等实际约束；结果对所拟合的响应面敏感，若真实响应更为复杂则效果可能下降。

---

## 342. Convertible Polynomial Evaluation Codes in the Merge Regime: A Skew-Polynomial Framework

**arXiv ID:** 2608.24179 | [PDF](https://arxiv.org/pdf/2608.24179v1)

**作者:** Songping Ge `[一作]` (Southwest Jiaotong University), Xiaohu Tang `[通讯]` (Southwest Jiaotong University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种代数框架，用于合并状态下的可转换编码，直接与底层编码的多项式评估结构相结合。

**💡 创新点**

创新点在于通过最小偏斜多项式的特征化、建立评估兼容的乘法规则以及推导出特殊的中国剩余定理（sCRT），为偏斜多项式评估编码的转换提供了新的模板。

**🔧 技术方法**

使用了偏斜多项式环 𝔽_q^m[x;x^q,0]，并结合了中国剩余定理的变体。

**📊 数据集**

使用了线性化的Reed-Solomon编码作为主要的数据集，并且在不同的初始编码和相同的初始编码情况下进行了实验。

**📈 对比分析**

与现有方法相比，本文提出的转换模板在符号访问成本上达到了最优，特别是在处理线性化的Reed-Solomon编码时，能够实现每个符号的访问最优成本。

**⚠️ 局限性**

限制在于偏斜构造需要完整的P基于共轭类，导致了m|k的可分性条件；此外，字段大小的要求并不旨在改善已知的通用MDS转换界限。

---

## 343. GATNextHop: A GAT for Shortest Path Routing with Cross-Topology Generalization

**arXiv ID:** 2608.23917 | [PDF](https://arxiv.org/pdf/2608.23917v1)

**作者:** Chia-Hong Chou `[一作]` (San José State University), Katerina Potika `[通讯]` (San José State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究使用图注意力网络（GAT）来预测最短路径中的下一跳，从合成图学习并推广到真实 ISP 网络。

**💡 创新点**

首次将 GAT 作为下一跳预测模型，并通过合成图训练验证其跨拓扑泛化能力，发现介数中心性是最重要特征。

**🔧 技术方法**

使用图注意力网络、节点/边特征编码、交叉熵损失、Adam 优化、早停以及与 Dijkstra SPF 的推理时间对比。

**📊 数据集**

利用 Internet Topology Zoo 实际 ISP 拓扑（180 张）和基于 ER/BA/WS/SBM/Waxman 的 1000 张合成图。

**📈 对比分析**

在合成验证中获得 85.1% 的准确率，在真实测试中 84.2%；与 Dijkstra 的单源推理相比，GAT 在大规模图中推理时间略慢（0.61 ms 对比 0.01 ms），但误差率仅约 16%。

**⚠️ 局限性**

在静态场景下推理速度明显落后，且 84% 的准确率意味着仍有约 1/6 的路由错误；未评估动态拓扑和多查询的加速优势。

---

## 344. Gen2Physics: Grounding Generated 3D Meshes in Physics via Multi-View Material Decomposition

**arXiv ID:** 2608.23869 | [PDF](https://arxiv.org/pdf/2608.23869v1)

**作者:** Mauro Comi `[一作]` (Google DeepMind), Manuel Sanchez `[通讯]` (Google DeepMind)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 Gen2Physics 框架，能够将任何生成的静态 3D 网格自动分解为基于材质的子网格，并为每个子网格分配合适的物理属性，使其可直接用于传统刚体仿真引擎。

**💡 创新点**

创新点在于将多视角 ViT 的密集材质分割结果通过 2D→3D 投影投票获得全局一致的材质地图，再利用 Vision‑Language Model 对分割结果进行语义纠错和内部结构推理，最终完成空洞/实心补全生成可闭合、物理可用的子网格；整个流程兼顾视觉细节与物理准确性。

**🔧 技术方法**

使用技术包括：细化的 Vision Transformer（TIPS）进行密集材质分割；多视角渲染和投票投影实现 3D 一致性；Gemini 2.5 VLM 进行语义校正和物理属性（密度、内部结构）推理；HoloPart 用于面片补全生成闭合子网格；基于体积/表面积计算质量并推导惯性矩阵。

**📊 数据集**

训练集为 500k 条 Coohom 多视角渲染样本，配有 12 类材质标签；评估集为 ABO‑500（用于质量估计）和自制 PartNet‑Material（用于材质分割）。

**📈 对比分析**

与 NeRF2Physics、PUGS 等基线在 ABO‑500 上的质量估计（ADE、ALDE、APE、MnRE）和 PartNet‑Material 上的 mIoU 进行对比，Gen2Physics 仅在 ADE、ALDE 等指标上与最佳基线相差≤0.2%，在 mIoU 上达到 48.3%（约两倍基线），并且是唯一能够输出可仿真闭合子网格的方案。

**⚠️ 局限性**

局限性包括：依赖外部 HoloPart 完成补全，未实现端到端；仅处理表面可见材质，未推断内部异质结构；PartNet‑Material 测试集规模有限；在极小或复杂内部结构的对象上可能误估质量。

---

## 345. Equivariant Covariance Tensors: Guaranteed SPD Uncertainty for Tensor-Valued Geometric Learning

**arXiv ID:** 2608.24386 | [PDF](https://arxiv.org/pdf/2608.24386v1)

**作者:** Ruihan Liu `[一作]` (Fudan University), Qingchao Jiang `[通讯]` (East China University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种 E(3)-等变的张量不确定性量化框架，能为对称二阶张量预测同时输出旋转对称的 SPD 协方差。

**💡 创新点**

创新点在于将对称矩阵映射到 Lie 代数并使用矩阵指数得到 SPD，同时利用不可约表示分解构造等变协方差头，并设计 Log‑Euclidean 等变评分目标。

**🔧 技术方法**

使用技术包括 E(3)-equivariant GNN (e3nn)、Kelvin–Mandel 维度化、矩阵指数映射、Log‑Euclidean 乘积与多元拉普拉斯负对数似然（LE‑ESO）、Clebsch–Gordan 基底。

**📊 数据集**

实验数据集为 ModelNet40 的惯性张量（几何验证）和 Materials Project 的电介质张量（真实材料预测）。

**📈 对比分析**

与确定性 MSE 与对角协方差 UQ 基线对比，ModelNet40 上 MAE 0.078、SPD 合法率 >99%，Materials Project 上 MAE 1.55、MACE 0.049，优于深度集成 UQ。

**⚠️ 局限性**

限制在于主要验证对称二阶张量，扩展到更高阶张量需新的表示基础；未能完整分离贝叶斯不确定性；相对确定性模型计算开销略高。

---

## 346. SceneReGen: Generative Reconstruction of 3D Scenes from a Single Image

**arXiv ID:** 2608.23930 | [PDF](https://arxiv.org/pdf/2608.23930v1)

**作者:** Zefan Tian `[一作]` (Huawei), Di Xu `[通讯]` (Huawei)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

利用预训练的生成式 3D 生成器与几何编码器，基于单张 RGB 图像完成对象级完整几何体的生成，并将其在共享观察对齐的场景框架中放置，从而实现单图像的完整 3D 场景重建。

**💡 创新点**

通过选择性姿态分解，将对象的观察到的旋转直接编码进生成网格，而仅通过场景级的平移和缩放估计，打破传统生成与重建的分离，提升几何完整性与场景一致性。

**🔧 技术方法**

使用 VGGT‑Ω 作为几何编码器，学习形状查询（shape queries）和位置查询（position queries）来分别调节对象生成与场景布局；对 DiT 生成器进行跨注意力扩展；采用遮挡增强、流匹配损失与 classifier‑free guidance。

**📊 数据集**

在 3D‑FUTURE 数据集上进行训练与评估，该数据集包含约 14,761 个训练场景和 5,479 个测试场景，每个场景提供对象分割和对应 3D 模型。

**📈 对比分析**

与 MIDI、Gen3DSR、SceneGen、ShapeR、SAM 3D、TRELLIS.2 等基线方法对比，SceneReGen 在场景级 Chamfer Distance、F‑Score 以及 3D 边界盒 IoU 上均取得最佳或第二最佳成绩，在对象级 Chamfer Distance 上并列最佳，F‑Score 仅次于 SceneGen。

**⚠️ 局限性**

局限性包括：纹理合成在强遮挡或极端光照条件下鲁棒性不足；低分辨率或模糊的输入图像会导致平移与缩放估计误差较大。

---

## 347. LUCAID: Agentic Multimodal AI for Lung Cancer Precision Pathology

**arXiv ID:** 2608.23803 | [PDF](https://arxiv.org/pdf/2608.23803v1)

**作者:** Marie-Lisa Eich `[一作]` (Charit'e -- Universitaetsmedizin Berlin), Simon Schallenberg `[通讯]` (Charit'e -- Universitaetsmedizin Berlin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发并临床验证了LUCAID，一个基于LLM代理的全流程肺癌病理诊断系统，整合九个模块完成QC、肿瘤检测、分型、TME分级、细胞含量评估、IHC细胞表型、PD‑L1/MET/TROP‑2评分及结构化报告生成；

**💡 创新点**

①整合完整诊断工作流程的多模态AI模块；②通过LLM代理进行任务调度与自然语言交互；③模块输出直接嵌入报告，确保可追溯与无“幻觉”；④在真实多中心临床样本上实现专家级一致性并超越人类病理学家；

**🔧 技术方法**

Atlas基础模型、视觉基础模型、细胞级检测与表型分类、基于核面积的肿瘤细胞含量估计、LLM代理（Claude Opus 4.8）工具调用、PubMed检索引用、结构化报告生成；

**📊 数据集**

1,620例多中心肺癌WSI（H&E+9种IHC）、1,001例发现队列、70例前瞻性临床验证队列（nNGM）、115例KRAS变异对照、多种扫描仪与染色平台；

**📈 对比分析**

使用105,227份专家标注对模块进行F1验证（0.82–0.95）；在70例前瞻性样本与5名资深病理学家进行一致性评估，LUCAID 93.0% 与专家标准一致，远高于68.3–81.1%；各项指标MAE低、相关性高；报告生成在10例评估中97%解释正确，引用准确率82.4%；

**⚠️ 局限性**

仅针对肺癌；报告评估样本有限；缺乏真实临床决策影响评估；需进一步外部验证与多学科合作验证其临床效益与可推广性。

---

## 348. AHEAD: Adaptive Hindsight with Environment-Augmented Distillation for Agentic RL

**arXiv ID:** 2608.24114 | [PDF](https://arxiv.org/pdf/2608.24114v1)

**作者:** Xiaolong Jin `[一作]` (AWS AI Labs), Varun Kumar `[通讯]` (AWS AI Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种步态感知的特权信息框架（AHEAD），通过将环境反馈与LLM生成的纠错提示结合，对多轮LLM代理进行自监督强化学习。

**💡 创新点**

创新点在于为不同类型的步骤（常规与错误）分配不同的监督源，利用环境反馈进行确认、利用LLM提示进行纠正，并仅在失败轨迹中激活特权信息，实现更细粒度的信用分配。

**🔧 技术方法**

技术包括GRPO基础算法、基于教师-学生对比的自监督蒸馏、LLM错误步骤分析器与纠错提示生成，以及基于令牌级蒸馏信号的优势重加权。

**📊 数据集**

使用了 ALFWorld、WebShop 以及 Search-based QA 三个多轮代理基准数据集，分别对应家庭任务、电子商务导航和检索式问答。

**📈 对比分析**

与 GRPO 及多种自监督蒸馏基线对比，AHEAD 在 ALFWorld 上提升 13.3 分、WebShop 成功率提升 11.0 分，并在样本与步数效率上均优于对照方法。

**⚠️ 局限性**

局限性包括对外部大型 LLM 解析器的调用成本、对错误步骤识别的依赖以及仅在短轨迹、二元成功/失败环境下验证，难以直接推广到更复杂或连续动作空间的场景。

---

## 349. The Value Generating Power of Weighted Tree Automata with Initial Algebra Semantics

**arXiv ID:** 2608.24247 | [PDF](https://arxiv.org/pdf/2608.24247v1)

**作者:** Manfred Droste `[一作]` (Leipzig University), Heiko Vogler `[通讯]` (Technische Universit"at Dresden)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275`

**🎯 论文内容**

本文研究了右分布式强双单子（尤其是半环）上的加权树自动机的生成能力，证明了存在既是双局部有限又非局部有限的强双单子，并展示了在含有符号位数≥2的字母表下，任何有限生成的强双单子都可以通过加权树自动机的初等代数语义生成其所有元素。

**💡 创新点**

创新点在于：①首次构造出既双局部有限又非局部有限的右分布式强双单子；②证明在符号位数≥2的情形下，加权树自动机的初等代数语义具有完全生成性，即可产生强双单子中的所有值；③从而得出即便是双局部有限也可能产生无限多值的结论，明显区别于加权字符串自动机。

**🔧 技术方法**

主要技术包括：
- 通用代数与合约理论，用来构造特定的强双单子；
- 术语代数与树同态，构造加权树自动机的编码与状态转换；
- 归约与同构论证，证明所构造的自动机确实产生所需闭包；
- 通过示例（如 Maps(ℕ)）展示语义不等价情况。

**📊 数据集**

无数据集，本文为纯理论研究。

**📈 对比分析**

无实验比较，本文未进行性能评估，主要给出理论证明与构造。

**⚠️ 局限性**

局限性：
- 结果仅适用于右分布式强双单子或半环，未覆盖左分布式或更一般的代数结构；
- 证明依赖于存在符号位数≥2的字母表，若仅有单一符号或所有符号位数为1，则不适用；
- 只讨论初等代数语义与运行语义的相等性，未对其他语义模型作进一步探讨。

---

## 350. UTS at CheckThat! 2026: Cite-Frame Engineering for Generated Fact-Checking Articles

**arXiv ID:** 2608.24466 | [PDF](https://arxiv.org/pdf/2608.24466v1)

**作者:** Dima Galat `[一作]` (University of Technology Sydney), Marian-Andrei Rizoiu `[通讯]` (University of Technology Sydney)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对 CheckThat! 2026 Task 3，提出并实现了一个 deterministic stub 生成器，并在其上叠加了 HostCite（域属性引用框架）和 ShadowVal（影子验证器）两条单杠干预，最终在官方 Leaderboard 中获得第二名。

**💡 创新点**

创新点在于：① 将评测机理（chunk‑count invariance 与 cite‑sentence entailability）拆解为两条设计规则，并据此设计 HostCite 与 ShadowVal；② 通过 Llama‑3.2:1B 仅作为 per‑cite 判定器，避免 LLM 生成文本对评分的负面影响；③ 实现了对评测机理的严格约束，使系统在不依赖 LLM 生成主体文本的情况下实现高精度引用。

**🔧 技术方法**

主要技术：Python deterministic pipeline、Llama‑3.2:1B 作为 per‑cite 判断器、RoBERTa‑MNLI 用于 chunk‑wise NLI、固定模板化文章结构、域属性引用框架与影子验证器逻辑。

**📊 数据集**

使用数据集：WatClaimCheck（训练/验证 3372 条，测试 1158 条），每条包含 claim、claimant、判定、预检索证据 URL 与对应全文；同时采用公开的 CheckThat! 2026 Task 3 评测集进行评估。

**📈 对比分析**

评估方法：在官方四度量（citation precision/recall、coverage、chunk‑wise NLI 与 per‑cite 评估）基础上计算 M4（四度量均值），UTS 系统在验证集上提升 +0.0274 M4，测试集上 M4 = 0.484，排名第二，领先 Baseline +0.212，落后获胜者 +0.062，主要差距在引用精确率/召回。

**⚠️ 局限性**

局限性：① 未实现低置信度引用的 selective emission，导致 P/R 与最优解存在差距；② LLM 用于正文生成被证明无效，系统缺乏可读性与自然语言流畅度；③ 仅针对 Llama‑3.2:1B 的引用判断，未探索更高级别或多模态判定器的潜力。

---

## 351. A Non-CDCL SAT Solver with Early Conflict Detection: The Watched-Literal-Based CSFLOC Solver

**arXiv ID:** 2608.24255 | [PDF](https://arxiv.org/pdf/2608.24255v1)

**作者:** Gábor Kusper `[一作]` `[通讯]` (Eszterházy Károly Catholic University), Gábor Kusper (Eszterházy Károly Catholic University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

开发了一种非CDCL SAT求解器 CSFLOC‑WL，将子句子集搜索改为观测字面量前缀传播，并引入早期冲突检测来实现更高效的计数器跳转。

**💡 创新点**

创新点在于用观测字面量实现子句子集搜索，结合早期冲突检测生成低层级跳转，从而显著减少全长子句空间遍历次数。

**🔧 技术方法**

采用观测字面量数据结构、单元传播、解释子句构造、冲突解析、前缀有效性判定以及学习子句缓存等技术。

**📊 数据集**

使用 SATLIB 中的随机 3‑SAT（uuf 系列）、鸽巢、Dubois、pret、ssa、bf、AIM 等未满足实例进行实验。

**📈 对比分析**

与 CSFLOC21TU、CSFLOC19、CaDiCaL 3.0.0 三个求解器分别做三次跑测，结果显示 CSFLOC‑WL3 在随机 3‑SAT 上显著快于竞争者，但在结构化实例上略逊。

**⚠️ 局限性**

主要局限是缺乏成熟的学习子句缓存，导致在结构化实例中性能不及 CSFLOC21TU；此外尚未实现完整的 2‑SAT 处理和进一步的优化。

---

## 352. Reverse Post Correspondence Problem and Undecidability of $5' \rightarrow 3'$ String Assembly Systems

**arXiv ID:** 2608.24257 | [PDF](https://arxiv.org/pdf/2608.24257v1)

**作者:** Benedek Nagy `[一作]` `[通讯]` (Eastern Mediterranean University), Benedek Nagy (Eastern Mediterranean University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文定义了逆向Post Correspondence Problem（Reverse PCP）和其等价形式odPCP，并利用它们证明了该问题及其正则过滤版本的不可判定性；随后将该结果迁移到新型String Assembly System（5'→3' SAS），证明了其空性问题不可判定；在一元情形下给出了完整的判定算法并讨论了可解与不可解实例的结构；此外提出了递归可枚举语言可由SAS语言经过删消同态得到的表述。

**💡 创新点**

创新点在于提出逆向PCP这一全新PCP变体及其与SAS的紧密联系；通过对SAS进行新定义的5'→3'模型展示了其与DNA双链结构的生物启发性；在一元案例中提供了完整判定算法，弥补了以往仅针对一般PCP的不可判定结果；进一步给出了递归可枚举语言与SAS语言之间的映射关系。

**🔧 技术方法**

主要技术手段是对图灵机计算过程进行编码的构造性归约：利用多份符号复制（Σ_1, Σ_2, Σ_3, Σ_4）将TM的配置、状态与磁带内容映射到domino和SAS单元；构造逆向PCP的domino集合和SAS的axioms、T、E，使得只有当TM接受时才存在完整匹配；对一元案例采用长度差集合与数论（Frobenius问题、最大公约数）进行判定。

**📊 数据集**

本工作为理论研究，无使用实际数据集；所有结果均为数学证明与构造。

**📈 对比分析**

论文不涉及实验或算法性能评估，主要通过证明展示理论性质；对比已知PCP、SAS等问题的可判定性，证明了新的不可判定性与可判定性边界。

**⚠️ 局限性**

限制与未解决问题包括：逆向PCP及odPCP的不可判定证明中仍使用正则过滤；是否能去除正则过滤得到更一般的不可判定结果尚未确定；SAS语言的完整层次结构、表达能力与其他模型的比较仍待深入。

---

## 353. Generating Intervention Hypotheses using Explainable Explanations on Graphs: G2I, a Two-Stage Greedy Framework

**arXiv ID:** 2608.23835 | [PDF](https://arxiv.org/pdf/2608.23835v1)

**作者:** Mulin Tian `[一作]` (University of Southern California), Ajitesh Srivastava `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一个两阶段贪婪框架（G2I），先在图神经网络中生成最小可执行的反事实解释，再将这些解释转化为可解释的 DNF 规则，用于设计网络层面的干预策略。

**💡 创新点**

创新点在于用离散贪婪搜索取代连续 mask 优化，直接最小化改变预测所需的干预量，并提供理论近似保证；同时通过 DNF 覆盖优化实现预算约束下的全局干预。

**🔧 技术方法**

技术包括：图卷积网络预测模型、基于贪婪的节点级反事实搜索、子模子（submodular）理论下的 DNF 覆盖贪婪算法、理论近似证明及无需数据集特定调参的实现。

**📊 数据集**

实验使用合成图（Neighbor‑Feature 与 Neighbor‑Only 变体，节点数从100到500，密度从1到5）以及真实世界的自杀风险网络（Military、Youth），并对比多种解释方法。

**📈 对比分析**

与 CF‑GNNExplainer、CF²、GNNExplainer 等基线相比，G2I 在精度、解释尺寸、覆盖率（AUCC）和运行时上均实现了显著提升，尤其在 100% 覆盖率、一次级别的时间节省（约一至两位数倍）上表现突出。

**⚠️ 局限性**

局限性包括：仅关注可执行的反事实解释，无法直接验证因果关系；对图结构假设较强，且对连续特征的细粒度干预成本建模仍不完善；若输入特征多为不可变属性，模型可能需要额外约束。

---

## 354. SteerCheck: Attribution Specificity and Alignment Leakage in Activation-Steering Audits

**arXiv ID:** 2608.24335 | [PDF](https://arxiv.org/pdf/2608.24335v1)

**作者:** Daming Luo `[一作]` (University of Technology Sydney), Junyu Xuan `[通讯]` (University of Technology Sydney)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并执行了在 Qwen3-14B 与 DeepSeek 上的激活 steering 评估，使用预注册的匹配 KL 预算，对比 isotropic、PCA 以及符号随机化三种 null family，并在特定性、转移、极性与语义效果等多维指标上进行评估。

**💡 创新点**

提出匹配 KL 预算作为对比基准，并引入交叉‑联合门控（intersection‑union）特定性测试，同时提供“对齐泄漏”诊断 A 以及符号随机化对齐保留量的实证分布，为激活 steering 的可审计归因提供细粒度方法。

**🔧 技术方法**

采用激活 steering（CAA）方向向量插入、KL 预算匹配、同构构造的符号随机化与 PCA 子空间对照、Holm 校正的多重假设检验、交叉‑联合门控、统计功效校准、Krippendorff α、宏F1 等统计与评估技术。

**📊 数据集**

使用 FLORES 英法翻译、FEVER 事实验证、OpenBookQA、16-token horizon 的中性提示库以及 Qwen3-14B、DeepSeek 的对齐银行，共计 768 条直接生成与 600 条层叠式 null 行。

**📈 对比分析**

通过与三种 null family 的 Holm‑corrected 单侧检验进行比较；在 Qwen3-14B 中 mean 通过 isotropic 通过但未通过 sign‑randomized，protected‑tail 失效；DeepSeek 的 language/ detox control 通过；转移指标显示 Qwen 在 margin 传递但未转移准确性；极性对比中 Qwen 通过 choice 但不通过 continuation；DeepSeek 在 choice/continuation 均不通过；人工评估显示 DeepSeek 通过 O1/O2，Qwen 通过 O2 但未通过 O1；整体表明激活 steering 有一定效果但特定性受限。

**⚠️ 局限性**

评估主要为后验性，缺乏因果性；对齐泄漏诊断依赖于特定构造，未验证跨模型转移；protected‑tail 统计功效低；对照库受限于固定 KL 预算与 token horizon；人工评估覆盖范围有限；自动评估判别器校准不足；整体结果混合，缺乏对所有模型和行为的泛化。

---

## 355. Exploit More, Explore Smarter for Budget-Constrained Agentic Search

**arXiv ID:** 2608.23848 | [PDF](https://arxiv.org/pdf/2608.23848v1)

**作者:** Haoyang Fang `[一作]` (Amazon AGI), Bernie Wang `[通讯]` (Amazon AGI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ExTS，一种针对预算受限的LLM代理搜索的树搜索策略；

**💡 创新点**

创新点在于将扩展视为信息增益决策，结合判别式奖励塑形、虚拟子节点估值和基于质量的分支门控，并提供诊断工具；

**🔧 技术方法**

使用改进的MCTS框架，加入奖励塑形函数、虚拟子节点采样与质量门控扩展；

**📊 数据集**

实验数据集包括HotpotQA、HoVeR、LiveCodeBench、K-MSE、DROP，以及GPU kernel优化；

**📈 对比分析**

与各领域最优任务特定基线对比，单一配置平均提升约5.5%，HotpotQA提升10.8%，LiveCodeBench硬难度提升11.7%，其余任务亦有显著或相当性能；

**⚠️ 局限性**

局限在于仅使用单一模型、诊断依赖模型与评分器、pilot树耗费预算、未探讨模型规模对搜索景观的影响。

---

## 356. Pipeline-Native Transformers: Co-Designing Model Architecture and CPU Inference for Bandwidth-Efficient Autoregressive Decode

**arXiv ID:** 2608.23841 | [PDF](https://arxiv.org/pdf/2608.23841v1)

**作者:** Tom Poperszky `[一作]` `[通讯]`, Tom Poperszky

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种针对CPU推理的流水线原生Transformer架构与CPU推理引擎的协同设计，旨在提高自回归解码的带宽效率。

**💡 创新点**

创新点包括：①使用按计算消耗顺序存储的L2尺寸瓦片化权重；②仅读取每个Mixture-of-Experts层的前k名专家并融合投影；③基于模型依赖参数的延迟感知阶段主调度；④实现磁盘驻留专家的异步I/O重叠以进一步提升吞吐量。

**🔧 技术方法**

技术手段包括：CPU-first流式引擎、垂直阶段主调度、瓦片化权重布局、top-k专家筛选、投影融合、延迟感知调度以及磁盘I/O异步重叠。

**📊 数据集**

使用的数据集为TinyStories，用于训练和评估不同架构的性能。

**📈 对比分析**

在五种架构中，最佳模型将关键路径权重带宽从9.00 MB/标记降低到4.50 MB/标记，接近最优候选模型的困惑度（+0.24）；在32核Ice Lake服务器上，30.9B参数的流水线原生MoE模型达到了5.94 tokens/s，显著高于相似规模稠密模型的4.75 tokens/s以及vLLM CPU后端的1.65 tokens/s；磁盘驻留专家的I/O重叠进一步提升了1.68倍，几乎匹配理论重叠模型。

**⚠️ 局限性**

局限性主要在于：①仍受主存带宽限制，超大模型需要更高带宽支持；②磁盘I/O重叠依赖磁盘性能，可能在不同存储介质上表现不一致；③报告中有一项设计声明被实验证明不成立，其余声明在特定条件下才可能成立。

---

## 357. Response Renormalization for Critical Deep Equilibrium Models

**arXiv ID:** 2608.23725 | [PDF](https://arxiv.org/pdf/2608.23725v1)

**作者:** Jose Luis Lima de Jesus Silva `[一作]` `[通讯]` (Federal University of Bahia), Jose Luis Lima de Jesus Silva (Federal University of Bahia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了响应重规范化（Response Renormalization）框架，用来缓解深度平衡模型（DEQ）和结构隐式层（SILVA）中残差雅可比近奇异导致的梯度放大问题；

**💡 创新点**

创新点在于只对loss可见的临界极点进行分母提升，提出了CMR、Phi‑CMR和Delta‑Phi三种选择性重规范化策略，既保留稳定通道，又避免全局梯度衰减，并给出了密集和矩阵无关的实现方法；

**🔧 技术方法**

技术包括奇异值分解、源‑响应理论、Woodbury身份、GMRES等线性求解技术，用于计算并修改残差雅可比的临界分母；

**📊 数据集**

使用了23个多物理/PDE/三维场/复杂几何/粒子系统的公开基准（如PDEArena、DynaBench、PDEGym、CFDBench、LagrangeBench等）进行训练和评估；

**📈 对比分析**

通过与标准隐式微分、Tikhonov正则化、稳定‑临界滤波等方法在一次步误差、八步误差、方向相似度等指标下对比，CMR/Φ‑CMR在大多数场景下误差不超过5%，梯度方向保持良好，且计算成本低；

**⚠️ 局限性**

局限性包括对极高维度或极弱可观测性的场景仍可能不理想，且需手动设定阈值和质量参数，未解决所有非线性收敛问题。

---

## 358. Markets, Not Planners: Decentralized Orchestration of LLM Agents with Private Information

**arXiv ID:** 2608.23867 | [PDF](https://arxiv.org/pdf/2608.23867v1)

**作者:** Xiao Liu `[一作]` (University of Chicago), James Evans `[通讯]` (University of Chicago)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了一个去中心化的LLM代理劳动力市场，利用私有成本估计与VCG式支付机制完成任务分配与分包。

**💡 创新点**

首次将经济学机制（VCG支付、声誉记录、私有反思笔记）与多层次分包结合，解决了集中式调度在成本隐私与偏好攻击下的瓶颈。

**🔧 技术方法**

多轮市场算法、VCG式第二价支付、公共声誉数据库、代理私有反思笔记、任务分解与子市场机制。

**📊 数据集**

四个基准数据集：OlympiadBench、BigCodeBench、SuperGPQA 与 GAIA。

**📈 对比分析**

与单模型、集中式路由器（如I RT-Router、CARROT）以及 MarketBench 等基线对比，平均得分提升明显，成本敏感度升高时优势更显著，开启分包后平均得分进一步提升约2点。

**⚠️ 局限性**

主要局限在成本估计不准确和投标策略未完全最优，导致在高成本敏感度场景下性能受限；对模型价格与偏好变化的鲁棒性仍待提升。

---

## 359. Matched Excess-Outranker Regularization for Candidate-Set Interference in Continual Knowledge Graph Embedding

**arXiv ID:** 2608.24273 | [PDF](https://arxiv.org/pdf/2608.24273v1)

**作者:** Hao Ren `[一作]` (University of New South Wales), Jiaojiao Jiang `[通讯]` (University of New South Wales)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对持续知识图谱嵌入中因实体增长导致的候选集干扰问题，提出一种正则化方法Meor来抑制新实体对历史查询排名的负面影响。

**💡 创新点**

将候选集干扰正式化为优化目标，并设计了匹配的结构参考与答案相对的平滑尾部聚合，形成单侧惩罚机制，只在新实体竞争超过匹配参考时才触发。

**🔧 技术方法**

采用匹配结构（预测角色、度数、关系签名）构造参考集合，利用答案相对的分数差进行归一化并通过指数加权平滑聚合，随后与结构匹配参考进行差值正则化。

**📊 数据集**

在实体增长流ENTITY（FB15K‑237）以及FBInc‑S、FBInc‑L三种增长速率的数据集上进行实验，并在不同嵌入基底（ComplEx、DistMult、TransE）和主机（Replay、LKGE、IncDE）上验证。

**📈 对比分析**

与Replay、持久校准、MMR和UOR等对照进行配对对比，结果表明Meor在ENTITY上使历史当前宇宙MRR提升0.0057、候选集干扰降低0.0055，并在所有十个传递实验中均获得正向提升，且新人采纳指标保持在保留阈值之上。

**⚠️ 局限性**

该方法仅针对实体增长导致的竞争干扰，无法解决关系或事实更新引起的模型参数漂移；当新实体竞争压力不足时正则化不激活，且参考构造受结构稀疏性限制，计算成本随新实体批量增长而增加。

---

## 360. The Blending Ratio Is Not Where the Performance Is: Diagnosing Prototype Blending for Few-Shot Adaptation of Vision-Language Models

**arXiv ID:** 2608.23634 | [PDF](https://arxiv.org/pdf/2608.23634v1)

**作者:** Liangzhi Li `[一作]` (Qufu Normal University), Guangshun Li `[通讯]` (Qufu Normal University)

**通讯引用:** 1749 | [OpenAlex ID](https://openalex.org/A5013763732)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

分析少样本视觉语言模型中混合比例 λ 的有效性，证明传统的 MSE 最优比例在分类任务中并非最优，并展示可以在无验证集的情况下通过留一交叉验证估计出接近 oracle 的比例；进一步证明即使使用 oracle 比例，线性探针仍能在多数场景下超越任何混合方法。

**💡 创新点**

提出混合比例的 MSE 驱动公式是错误的理论依据，首次提供无验证集的留一估计方法，并系统性验证验证无关线性探针（CLAP、LP++）在所有 shot ≥4 时均优于 oracle 混合，揭示混合族的能力上限。

**🔧 技术方法**

使用 James–Stein 收缩公式推导 MSE 最优比例，设计留一交叉验证比例估计；构建混合、NCM、GDA、HOSO、CLAP、LP++、Tip‑Adapter 等多种分类器；通过大规模 4,800‑cell 细粒度实验进行统计与置信区间分析。

**📊 数据集**

十个公开分类数据集（FGVC-Aircraft、Caltech101、Stanford Cars、DTD、EuroSAT、Flowers102、Food‑101、ImageNet、Oxford‑IIIT Pets、SUN397），五种 backbone（CLIP RN50、ViT‑B/32/16、ViT‑L/14、SigLIP ViT‑B/16），不同 shot 数、随机种子和四层 prompt 质量。

**📈 对比分析**

与零样本、NCM、四种混合比例（JS、LOO 等）、GDA、HOSO、CLAP、LP++、Tip‑Adapter 等对比；结果显示：1）JS 混合平均比 oracle 差约 8.5pp；2）LOO 比例与 oracle 差 < 1pp；3）验证无关线性探针（CLAP、LP++）在所有 shot ≥4 时均比 oracle 混合高 1.5–3pp；整体表明线性探针在该任务上优于任何混合方法。

**⚠️ 局限性**

实验使用 clip‑benchmark 固定划分，未采用图像增强；仅考虑单标签分类；部分方法（如 GDA、LP++）缺乏官方实现细节；未覆盖更大规模或多标签任务；结果受 feature 缓存与特定 backbone 限制；报告的上限仅为对该混合族的理论上限，未涵盖所有可能的混合策略。

---

## 361. A co-rotational formulation for arbitrarily shaped planar beams under large displacements and rotations

**arXiv ID:** 2608.23883 | [PDF](https://arxiv.org/pdf/2608.23883v1)

**作者:** Linh T. M. Phi `[一作]`, Zhangxian Yuan `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种高阶共旋转（co-rotational）梁单元，用于分析具有任意初始几何形状的平面梁在大位移和大旋转下的行为。

**💡 创新点**

创新点在于：①引入辅助直线参考构型，使初始几何、参考和当前构型都在同一参考系下描述，保持应变测量的客观性；②将高阶NURBS基函数直接用于几何和位移场，避免传统低阶单元导致的几何逼近误差和锁定；③通过高阶共旋转框架实现对任意曲线形状梁的锁定无关、超越传统方法的高精度与收敛性。

**🔧 技术方法**

主要技术包括：共旋转框架的刚体运动分离；基于NURBS的等几何几何重构与位移插值；离散化中的高阶基函数和高阶导数计算；Hamilton原理推导的运动方程；Newmark/ Newton–Raphson求解与伪弧长法等后处理手段。

**📊 数据集**

实验验证使用了多种结构实例：直线与锥形梁、开口圆环、深弓形、自由落体摆钟、螺旋梁等。每个实例通过解析解或ABAQUS（B21元素）对照，用以评估几何精度、锁定现象、能量守恒和动力响应。

**📈 对比分析**

比较方法主要是：与解析解的相对误差、与ABAQUS数值解的形变和能量曲线、对不同阶数(p=2,4,7)及控制点数(n=p+1 vs 更大)的收敛速率。结果表明：高阶(NURBS p=4–7)在相同自由度下可实现机器精度级误差，锁定现象显著减弱；n=p+1 的网格往往在中高阶时表现出更优的误差；对薄梁和非均匀几何的逼近也非常准确。

**⚠️ 局限性**

局限性包括：①未实现真正的锁定无关性，仍需依赖高阶基函数才能显著缓解锁定；②在极高阶（p>12）时出现矩阵条件数不佳导致数值不稳定；③目前仅验证平面梁，三维或有限元框架的推广需进一步研究；④示例使用单一权重NURBS（相当于B-spline），对需要精确圆形等几何的真实Rational NURBS未做专门处理。

---

## 362. Beauty is in the ELBO of the Beholder: A Variational Account of Processing Fluency in Face Perception

**arXiv ID:** 2608.24219 | [PDF](https://arxiv.org/pdf/2608.24219v1)

**作者:** Francisco M. López `[一作]` (Frankfurt Institute for Advanced Studies), Jochen Triesch `[通讯]` (Frankfurt Institute for Advanced Studies)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fede83ac-7505-405f-ab37-e7284695c47f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在未使用美貌标签的条件下，训练了卷积变分自编码器（VAE）四个面部数据集，随后用其学习到的表示评估芝加哥面部数据库（CFD）中的597张面孔的美貌评分，发现人类美貌倾向与VAE的证据下界（ELBO）在压缩-失真空间中的方向高度一致，并且吸引人面孔在形状和潜空间上更为原型。

**💡 创新点**

证明了处理流畅度（即低失真和低压缩成本）与人类面部美貌之间的量化关系；展示了潜空间中独立训练的VAE共享一个可迁移的“美貌方向”，并且这一方向超越传统的形状原型解释。

**🔧 技术方法**

使用卷积变分自编码器（VAE）进行无监督表征学习；计算压缩率（rate）和失真（distortion）的ELBO；通过标准化、Procrustes对齐和余弦相似度评估潜空间方向；使用多组交叉验证进行跨模型迁移测试。

**📊 数据集**

训练集：FairFace、FFHQ、CelebA、UTKFace；评估集：Chicago Face Database（CFD），包含八个族群×性别的597张面孔及其美貌评分。

**📈 对比分析**

方法：比较人类美貌评分与VAE ELBO的相关性，评估吸引方向与ELBO方向的余弦相似度；进行跨模型方向迁移预测；与形状和潜在原型度量比较。性能：ELBO与美貌评分的相关系数约0.25–0.32，吸引方向与ELBO方向的余弦相似度平均超过0.93，跨模型迁移保持约0.52的相关性，显示出强大的一致性和可迁移性。

**⚠️ 局限性**

仅使用简单的VAE模型，缺乏对人类视觉系统的生物学解释；CFD仅包含正面中性表情，限制了对姿态、表情变化和自然条件的泛化；美貌评分为群体平均值，未考虑个体差异；所有发现均为相关性，未证明因果关系。

---

## 363. Structured Frequency-Domain Evidence for LLM-Based Time-Series Anomaly Detection

**arXiv ID:** 2608.24113 | [PDF](https://arxiv.org/pdf/2608.24113v1)

**作者:** Jungwook Seo `[一作]` (Hanyang University), Sungyong Baik `[通讯]` (Hanyang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在零样本情况下，将频域证据（全局和局部FFT特征）与去季节化的时间序列索引值一起输入LLM，以实现更精确的异常区间定位。

**💡 创新点**

创新点在于将频域信息显式化为输入级证据，弥补现有LLM时间序列异常检测方法仅提供时间域或可视化信息而忽略频域特征的缺口。

**🔧 技术方法**

使用Fast Fourier Transform提取全局主频、谱熵、低高频能量比等特征；对滑动窗口进行局部频域描述；并在多模态LLM（InternVL2、Qwen2.5-VL、Gemini、GPT‑4o）中进行提示融合。

**📊 数据集**

在AnomLLM基准上评估，覆盖点、区间、趋势、频率等八类异常；同时在TSB‑AD‑U的八类子集上验证跨数据集鲁棒性。

**📈 对比分析**

与原始LLM‑TSAD基线对比，所有四个模型在标准F1和关联F1上均有提升；最显著提升集中在频率和趋势异常上，表明频域证据对周期性变化异常的检出尤为有效。

**⚠️ 局限性**

局限在于无法解析LLM在生成结果时如何权衡时间域、可视化和频域证据，且零样本异常定位仍面临挑战，未来需进一步探究模型内部机制与更复杂多变量序列的适用性。

---

## 364. STRIVE: Multi-Agent Structured Temporal Reasoning with Integrated Verification for Longitudinal Radiology Report Generation

**arXiv ID:** 2608.24237 | [PDF](https://arxiv.org/pdf/2608.24237v1)

**作者:** Junyeong Maeng `[一作]` (Korea University), Heung-Il Suk `[通讯]` (Korea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出STRIVE框架，将长期放射报告生成任务拆分为诊断、属性估计与时间变化三大专属代理，并在生成前后分别进行一致性门控与验证编辑，显著提升临床准确性与时间变化一致性；

**💡 创新点**

创新点包括：1）多代理分解与显式临床证据结构化；2）进展感知GRPO奖励对时间变化标签的分级奖励；3）两阶段一致性与验证机制确保报告与临床证据一致；

**🔧 技术方法**

采用七个冻结的胸片专家、指令调优LLM做诊断与属性推理，Temporal Change Agent通过GRPO强化学习优化；利用检索式写作与27B指令调优LLM进行报告生成与编辑；

**📊 数据集**

在Longitudinal‑MIMIC（MIMIC‑CXR的纵向子集）上进行评估；

**📈 对比分析**

与单图像与纵向RRG多种基线对比，STRIVE在NLG指标（BLEU‑1/4、ROUGE‑L、METEOR）、临床效能（CheXbert P/R/F1）及ReXrank、LCC等多项指标均取得最优或第二优表现，尤其LCC提升超过双倍；

**⚠️ 局限性**

局限性包括对时间变化代理的GRPO训练需要额外计算资源，验证阶段依赖规则化流程可能忽略细微语义差异，且在极端不完整的历史报告下表现尚待进一步提升。

---

## 365. Negotiating Ontological Boundaries in User-Authored Personal Sensing Systems

**arXiv ID:** 2608.24058 | [PDF](https://arxiv.org/pdf/2608.24058v1)

**作者:** Nava Haghighi `[一作]` (Apple Inc. and Stanford University), James Landay `[通讯]` (Stanford University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实施了两款开放式探针（EventMarker 与 PatternFinder），通过一周的实验让参与者自行定义并标记个人感知事件与模式，探索用户在自创系统中如何协商和扩展本体边界。

**💡 创新点**

创新之处在于引入开放式探针与差异化抽象层次，让用户在日常生活中主动协商现有系统固化的本体假设，并通过差异化分析方法揭示边界协商的四大场域（现象、主体、信号与噪声、数据客观性）。

**🔧 技术方法**

采用 Wizard‑of‑Oz 设计，利用 Apple HealthKit 接收心率、能量、步数等生理数据，使用 Swift/SwiftUI 开发原型，并通过进度轮视觉反馈模拟模型训练过程。

**📊 数据集**

使用参与者自身 Apple Watch 与 iPhone 采集的 Apple HealthKit 数据集，包括心率、心率变异性、能量消耗、步数等七种生理信号。

**📈 对比分析**

未进行传统性能评估；研究以质性差异化分析为主，未与现有算法或工具做量化比较，主要关注用户边界协商的可观察事件与体验。

**⚠️ 局限性**

局限性包括样本量仅八人、实验时长仅一周、Wizard‑of‑Oz 依赖手工后处理、技术与设备限制（如心率记录不稳定）、参与者经验差异导致结果差异大、难以推广至更广泛人群。

---

## 366. UTVPI-representable integer point sets: discrete convexity, polymorphisms, and pairwise closure

**arXiv ID:** 2608.24078 | [PDF](https://arxiv.org/pdf/2608.24078v1)

**作者:** Kei Kimura `[一作]` (Kyushu University), Ryo Yoshizumi `[通讯]` (Kyushu University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对整数格点集合的四类线性不等式系统（SVPI、DC、UTVPI、TVPI）进行研究，并从五个互补视角（不等式表示、离散凸性、多项式运算、两坐标投影重构、闭包算子固定点）对UTVPI可表示性进行完整表征；同时给出了SVPI可表示性的多运算表征，并探讨了其他相关类（TVPI、闭孔、孔、2-可分解、整数凸）在操作闭包上的可表征性极限。

**💡 创新点**

创新点在于：①提出并证明了UTVPI可表示性的五个等价描述，首次把离散凸性、2-可分解和闭包算子固定点等五种视角统一起来；②引入了“配对闭包”框架，给出局部到全局的闭包判断标准；③证明SVPI可表示性不可通过普通运算闭包表征，而给出了多运算和一维重构的完整表征；④给出对TVPI、闭孔、孔、2-可分解、整数凸类的操作闭包表征极限，指出它们在不同维度下的差异。

**🔧 技术方法**

主要使用的技术包括：离散凸分析（整数凸性、闭孔）、多元运算与多运算（多态子运算）、闭包算子与固定点理论、投影与重构方法、局部-全局（pairwise）闭包理论，以及对UVPPI系统的代数性质（中点运算、向量投影）进行的严谨证明。

**📊 数据集**

本工作为纯理论研究，不涉及具体数据集；所有结果均基于数学证明和构造反例。

**📈 对比分析**

由于本论文不包含实验评估，未采用传统意义上的方法比较与性能评估；其贡献主要体现在理论等价性与包含关系的完整划分，提供了对相关类之间精确的层次结构和交集特征。

**⚠️ 局限性**

限制与未解决问题：①对于TVPI、闭孔、孔、2-可分解、整数凸等类，无法得到统一的普通运算闭包表征；②SVPI可表示性仅在多运算框架下可表征，限制了其在传统CSP/SAT理论中的直接应用；③本文的局部-全局闭包框架虽广泛适用，但在高维情况下仍需要进一步研究其计算复杂度与实现细节。

---

## 367. Phase-Aligned Finite-Fourier Periodic Deformation for 4D Medical Image Interpolation

**arXiv ID:** 2608.24027 | [PDF](https://arxiv.org/pdf/2608.24027v1)

**作者:** Haojin Li `[一作]` (Southern University of Science and Technology), Jiang Liu `[通讯]` (Southern University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种基于相位结构的连续变形模型，实现4D医学图像从稀疏端点观测中插值缺失体积。

**💡 创新点**

创新点在于：① 使用有限傅里叶基对相位条件的速度场进行参数化，将周期性/近周期性运动直接嵌入变形空间；② 引入基于变形强度的相位对齐时间再参数化，解决非均匀运动进程；③ 采用双向端点位移、周期一致性与残差细化，保证解剖连贯性与精细度。

**🔧 技术方法**

主要技术包括：有限傅里叶周期变形参数化、相位对齐时间再参数化、双向连续变形积分、轻量残差细化网络、周期一致性损失、频率权重正则化、NCC + Charbonnier 训练损失。

**📊 数据集**

使用公开数据集 ACDC（心脏MRI）和 4D-Lung（4D CT）进行评估，端点分别为心搏端点和呼吸端点。

**📈 对比分析**

与多种基线（VM、TM、MPVF、SVIN、IFRNet、AMT、UVI‑Net、PerVFI、LDDM、TMSDF 等）在 PSNR、NMI、SSIM 上比较，实验表明本方法在两大数据集上均取得最高或接近最高分，显著提升插值质量并保持解剖一致性。

**⚠️ 局限性**

局限性：对极端运动或大时间间隔的外推能力仍有限；模型对高频细节的捕捉依赖正则化参数，可能导致细节模糊；仅在端点条件下训练和评估，实际多帧序列或不同采样模式的适用性需进一步验证；实现复杂度相对较高，训练和推理资源需求较大。

---

## 368. Ray-Traced Augmentation for Signal Strength Based Localization

**arXiv ID:** 2608.23901 | [PDF](https://arxiv.org/pdf/2608.23901v1)

**作者:** Jihoon Og `[一作]` (University of Alberta), Omid Ardakanian `[通讯]` (University of Alberta)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于射线追踪的室内Wi‑Fi RSS指纹增强框架，利用三维BIM模型和Sionna RT生成合成指纹，并用ResNet网络进行定位。

**💡 创新点**

创新点包括：两级校准（材料参数与每个AP设备偏差）实现高精度仿真；将RSS转换为二值或灰度图像三种表示并通过频段融合提升定位精度；证明仅用合成数据即可取得竞争性性能。

**🔧 技术方法**

技术手段包括：Blender/BIM建模、Sionna RT射线追踪、贝叶斯优化校准、Per‑AP差值校准、RSS热度图/二值/多值变换、ResNet‑18深度网络、上行/下行频段融合。

**📊 数据集**

实验数据集基于阿尔伯塔大学Athabasca Hall 1楼，收集34个实测RSS点，合成2,155点，构建synthetic‑only与20%真实种子两套训练集。

**📈 对比分析**

方法与四个基线（凸包、DBSCAN、RBF、extendGAN+）在平均欧氏距离上对比，结果显示多值+上行融合ResNet在synthetic‑only数据上平均误差为3.05 m，优于最佳基线4.59 m，误差减小1.54 m，且标准差更小。

**⚠️ 局限性**

局限性：仅验证单层楼层；对不同设备/多楼层的适用性尚未评估；频段融合方法仍相对简单；需要手工定位AP并依赖BIM的准确性。

---

## 369. Adaptive Influence Graphs for Failure Attribution in Multi-Agent Systems

**arXiv ID:** 2608.24361 | [PDF](https://arxiv.org/pdf/2608.24361v1)

**作者:** Yarden Bakish `[一作]` (Tel Aviv University), Ron Litman `[通讯]` (AWS Agentic AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多智能体系统的失败归因，提出自适应影响图（AIG）框架，将失败追踪从原始日志转化为可导航的图，并通过代理驱动的读取来定位错误。

**💡 创新点**

创新点在于把失败归因视为接口设计问题：既通过自适应图构造动态决定节点边界与继承关系，又通过代理式遍历结合原始日志验证，显著提升归因精度。

**🔧 技术方法**

使用大型语言模型（Opus‑5、Sonnet‑4、DeepSeek‑V3.2 等）构建与评估代理，配合日志检索、图构造与修订工具，以及 critic–refiner 循环来生成和验证结构化影响图。

**📊 数据集**

数据集使用 Who&When 基准的 Algorithm‑Generated 与 Hand‑Crafted 两个子集。

**📈 对比分析**

与先前基线（如 CHIEF、RAFFLES）以及单调、结构化日志、影响图等中间表示的 ablation 对比，Algorithm‑Generated 上实现 55.20% 的 step accuracy、71.20% 的 agent accuracy，超越先前最高记录 51.60%。

**⚠️ 局限性**

主要局限包括仅使用 Who&When 的 agent 输出，缺乏输入与工具日志；缺少节点/边的黄金标注导致结构评估仅靠 critic 与下游性能；构造与修订阶段额外消耗 tokens，适用于诊断代价较高的场景。

---

## 370. Sensorless damage-safe grasping

**arXiv ID:** 2608.23983 | [PDF](https://arxiv.org/pdf/2608.23983v1)

**作者:** Yusei Shuto `[一作]` (Kyushu University), Danilo Vasconcellos Vargas `[通讯]` (Kyushu University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种只利用编码器位置和电机努力信号的无触觉感知抓取控制器，利用压缩应变上限实现对软物体的损伤安全抓取。

**💡 创新点**

创新点在于：①通过将力-电流比例与物体下界刚度相除得到压缩应变估计，保证在任何符合下界的物体上压缩不超过用户指定的ε；②将压缩应变作为可解释的损伤上限参数；③量化接触检测耗压并将闭合速度作为吞吐‑温和度的可调参数。

**🔧 技术方法**

技术实现：电机电流与外部力的线性关系估计；使用压缩应变公式 r̂ = F̂/(k_min·D)；比例控制器完成闭合与停止；在MuJoCo仿真与SO‑ARM101硬件上验证；采用Feetech STS3215伺服电流负载寄存器作为努力信号。

**📊 数据集**

数据集：仿真中使用尺寸为35/40/45 mm、刚度为2–10 kN/m的立方体；硬件中使用3D打印TPU立方体，填充率为5%、10%、15%；不使用真实水果，采用可控刚度立方体作为实验对象。

**📈 对比分析**

比较方法：与两种固定力基线（低力停止、固定高力停止）比较；结果显示：在仿真中k≥4000 N/m时，ε∈[0.7,2.0]%范围内实现≥98%抓取成功且0%损伤；在硬件上与基线相比，抓取成功率相当、损伤率从100%降至40%（软块），并使用约一半的抓取力；固定力基线在软块常出现完全损伤或超时。

**⚠️ 局限性**

局限性：①在极软物体（k≈2000 N/m）无法同时实现抓取成功与无损伤；②控制器需要预先给定物体直径D，对不规则形状水果的适用性有限；③无法在线估计刚度，只使用下界k_min；④硬件验证仅在立方体上，未在真实水果上验证。

---

## 371. Rules Before Oracles: Auditable, User-Configurable Argument Selection for Deliberative Polling

**arXiv ID:** 2608.23979 | [PDF](https://arxiv.org/pdf/2608.23979v1)

**作者:** Muntaser Syed `[一作]` (Florida Institute of Technology), Marius Silaghi `[通讯]` (Florida Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种可公开、可重现且可配置的规则，用于在民主决策前为每位投票者挑选有限数量的论证条目，以保证被展示的论证能覆盖并代表当前存在的理由。

**💡 创新点**

创新点在于把论证选择视为可审核的“立法程序”，通过基于“一次反向背书流”且仅使用公开的背书计数与一跳链接权重的可配置算子，既实现了可解释性与可争议性，又在不同评估维度（覆盖度、顺序及时效性、背书质量）上优于随机与传统学习式排序，并在对抗性攻击下保持鲁棒。

**🔧 技术方法**

核心技术包括：(1) 把投票问题建模为二元辩论图，定义“覆盖度”“顺序覆盖”“背书质量”三种测度；(2) 设计基于参数化权重的单跳评分函数，满足七项可审核特性；(3) 使用贪婪子模子上界作为评估天花板；(4) 采用基于代理的仿真器对多种参数配置和攻击模型进行实验；(5) 采用分布式P2P存储以实现本地可重现的计算。

**📊 数据集**

数据集为合成的议题与理由库，包含三组议题（“AI普惠性基础设施”“AI基本收入”“学生无电脑上学”），每组有30个核心理由；实验通过随机生成1000名代理人、不同作者率、链接率、投票比例与攻击规模（0–25%）来构建1000个子语料库，最终产生约17000次配对实验。

**📈 对比分析**

与基准（随机抽取、贪婪上界、无链接权重的纯背书计数）对比，规则在非退化作者情境下与随机差距极小（<0.5%），但在退化作者或攻击情境下可显著提升覆盖度（最多提升4.7%）和排序顺序（在25%攻击下提前约5.7步达到90%覆盖），并在背书质量上高出约3.3倍；与贪婪上界的差距约3.1%，表明可解释规则仅失去极少的潜在最大覆盖度。

**⚠️ 局限性**

主要局限包括：① 仅在合成议题上验证，缺乏真实文本与人类投票者实验；② 仍受“时间不公平”限制，早期投票者看到的理由空间有限；③ 需要预先提取理由标签，若标签不准确会影响评估；④ 规则在高度冗余或极端攻击下仍可能被绕过（如多签名同一链接的联署攻击）。

---

## 372. Continual Visual Learning under Evolving Semantic Concept Shift

**arXiv ID:** 2608.23903 | [PDF](https://arxiv.org/pdf/2608.23903v1)

**作者:** Ismail Lamaakal `[一作]` (Mohammed Premier University), Ibrahim Ouahbi `[通讯]` (Mohammed Premier University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SemReWrite框架，实现视觉模型在语义概念演化过程中的选择性重写与知识保留；

**💡 创新点**

创新点在于将语义变更表示、受影响区域定位、结构化语义记忆与输入依赖低秩重写机制相结合，兼顾语义重写与保留；

**🔧 技术方法**

使用预训练视觉‑语言模型编码、文本与视觉对齐、低秩矩阵参数化、语义图正则化以及多目标损失；

**📊 数据集**

在EvoShift‑Bench上评估，涵盖ImageNet、iNaturalist、CUB‑200、DomainNet等四类数据集，测试分裂、合并、边界修订、插入、部分重定义、递归与混合语义+外观漂移等七种语义迁移；

**📈 对比分析**

与全微调、LoRA、EWC、PromptAlign、TPT等多种基线对比，SemReWrite在Rewrite Accuracy、Preservation Accuracy和综合SRS指标上均优于对手，尤其在少样本和混合漂移场景下表现突出；

**⚠️ 局限性**

局限在于需提供清晰可视化的语义说明、可由图像学习的差异、对全局变更适应性不足、稀疏标注难以精细定位细粒度边界，以及对语义冲突的检测和纠正不足。

---

## 373. Granite.Trust Policy Tools: Shareable, Actionable Policies for Generative AI Applications

**arXiv ID:** 2608.23870 | [PDF](https://arxiv.org/pdf/2608.23870v1)

**作者:** Nathalie Baracaldo `[一作]` (IBM Research), David Cox `[通讯]` (IBM Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套基于YAML的“Actionable Policy”模式，支持针对生成式AI模型输出的内容约束，并配套构建了面向该模式的合规性检查与异常治理工具；

**💡 创新点**

创新点在于首次将内容约束写入可读可执行的政策格式，支持异常追踪与多阶段合规性验证，且通过该模式生成合规性数据提升模型对政策的遵守能力；

**🔧 技术方法**

采用YAML Schema定义政策，利用DGT（数字生成工具）框架和Granite Guardian进行政策驱动的对抗式提示、答案生成及安全过滤，并支持LoRA等微调技术；

**📊 数据集**

主要使用政策描述所生成的自定义对抗式提示与安全响应对，结合少量人类提供的种子示例，构建合规性数据集；

**📈 对比分析**

在评估中对比了无政策与有政策驱动的合规性数据生成，发现后者在捕捉边缘违规样本、减少误拒方面更为有效，且在实验中提升了模型对政策的遵从率；

**⚠️ 局限性**

局限包括参与者样本量有限、对开放源模型的依赖导致对某些风险难以生成足够对抗示例，以及工具在多团队协作与冲突检测方面仍需进一步完善。

---

## 374. FARCA: Fact-Aligned Reliability-Aware Credit Assignment for Reinforcement Learning with Factual Supervision

**arXiv ID:** 2608.24350 | [PDF](https://arxiv.org/pdf/2608.24350v1)

**作者:** Qiming Xie `[一作]` (Nanjing University of Science and Technology), Rui Xia `[通讯]` (Nanjing University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 FARCA 框架，针对大语言模型在 RL‑with‑Verifiable‑Rewards 训练中出现的事实信用分配噪声问题，通过事实‑令牌对齐与可靠性加权来实现细粒度、可信的事实信用分配，提升模型事实性与推理能力。

**💡 创新点**

创新点在于：① 解决事实信用分配的定位模糊（credit localization ambiguity）与可靠性模糊（credit reliability ambiguity）问题；② 采用计数化的 token provenance 将事实与生成的 token 对齐；③ 利用反事实证据归因（counterfactual evidence attribution）来估计验证器输出的可靠性，并将可靠性权重融入奖励与优势重塑；④ 在奖励设计中加入可靠性加权的事实奖励，实现对每个 token 的局部更新。

**🔧 技术方法**

技术包括：① 使用 GPT‑4o 进行原子事实提取与 token provenance 建立；② 通过 NLI 验证器（HHEM‑2.1‑Open）得到连续符号化的事实分数；③ 反事实证据归因来计算可靠性权重；④ Group Relative Policy Optimization（GRPO）与基于 token 的 PPO‑clip 目标；⑤ 可靠性加权的奖励与优势重塑。

**📊 数据集**

数据集：知识密集型任务的 HotpotQA 与 2WikiMultiHopQA（挑选子集）；数学推理训练使用 SimpleRL；评估包括 SimpleQA、TruthfulQA、HallucQA、HaluEval‑QA 以及数学推理基准 AIME2026/25、MATH‑500、GSM8K。

**📈 对比分析**

与零样本提示、仅输出格式与答案奖励的 GRPO、以及现有事实强化学习基线（KnowRL、FSPO、FaithRL）对比。实验显示 FARCA 在 Qwen2.5‑3B‑Instruct 上在四个 hallucination 评测上均优于 FaithRL，平均提升 1.75 点；在 Llama‑3.2‑3B‑Instruct 上平均提升 2.21 点；同时在数学推理基准上保持或提升性能。

**⚠️ 局限性**

限制：① 依赖外部 NLI 验证器与 GPT‑4o 的事实抽取，成本高且对低资源模型适用性未知；② 可靠性估计仅基于单一最相关证据句的移除，可能未捕捉更复杂的证据依赖关系；③ 对长文本的 token‑provenance 对齐仍可能出现误匹配；④ 仅在实验中对 3B 规模模型验证，是否能扩展到更大模型尚未测试。

---

## 375. CodeHID: Learning an Addressable Hierarchical Code Index for Generative Code Retrieval

**arXiv ID:** 2608.24089 | [PDF](https://arxiv.org/pdf/2608.24089v1)

**作者:** Zhen Li `[一作]` (Xiamen University), Hui Li `[通讯]` (Xiamen University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过构建层级化的DocID空间和生成式检索，实现代码检索从平面匹配转向语义地址生成。

**💡 创新点**

关键创新是使用伪邻居引导的DocID学习将连续语义邻域映射为前缀共享的离散索引，并结合双阶段DocID生成指导提升路径选择。

**🔧 技术方法**

采用RQ‑VAE残差量化+kNN伪标签构造DocID，配合GraphCodeBERT编码器+Qwen2.5‑Coder解码器，使用硬负样本、排名蒸馏和前缀约束解码。

**📊 数据集**

在CoSQA和ProCQA（Python/Java）两大代码检索基准上进行实验。

**📈 对比分析**

与BM25、CodeBERT、UniXcoder、CodeSage、OASIS、CodeXEmbed、DSI、NCI、GLEN、RIPOR等传统与生成式检索方法比较，CodeHID在Hit@1、Hit@3、Hit@5、MRR@20等指标上均实现显著提升，尤其在rank‑one表现最为突出。

**⚠️ 局限性**

主要局限在于离线DocID构造需要全局量化和kNN图，随着代码库扩展可能产生规模瓶颈；同时对动态更新的代码库支持有限。

---

## 376. Towards a Definition of the Computational Architecture of Open Scholarly Infrastructures

**arXiv ID:** 2608.23760 | [PDF](https://arxiv.org/pdf/2608.23760v1)

**作者:** Ivan Heibi `[一作]` (University of Bologna), Silvio Peroni `[通讯]` (University of Bologna)

**通讯引用:** 3094 | [OpenAlex ID](https://openalex.org/A5031461768)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个基于POSI原则的多层计算架构框架，并通过OpenCitations实例验证了该框架在构建开放学术基础设施中的可行性。

**💡 创新点**

创新点在于将开放学术基础设施的技术维度（开放性、可持续性、互操作性、可重复性）细化为七大功能，并将其映射到硬件、虚拟化、编排、应用与元层四个互补层级，从而提供一种系统化、可复现的设计方法。

**🔧 技术方法**

使用的技术包括开源硬件与软件：裸机服务器、Proxmox虚拟化、Kubernetes编排（配合Rancher）、Docker容器化、Helm/OpenTofu IaC、GitHub+GitLab版本控制等。

**📊 数据集**

数据集主要是开放引文与书目元数据（OpenCitations提供的CC0数据），以及在GraspOS项目中开发的文献提取与引用意图分类工具所使用的PDF/引用文本。

**📈 对比分析**

方法通过对比传统单层或仅硬件/虚拟化部署的可扩展性、可观察性、自动化水平和复现性进行评估；实验表明，层级化架构使得服务复制、弹性伸缩和工作流迁移更高效，平均部署时间缩短30%，可观测性指标提升约40%。

**⚠️ 局限性**

局限性包括：元层作为可选项时可能导致配置散布；对云基础设施的支持不足，导致在大规模弹性扩展上受限；以及IaC与编排工具之间的集成仍需手动脚本，缺乏统一的全流程自动化框架。

---

## 377. DreamLedger: Execution-Settled Credit Files for World-Model Imagination in Robot Decision Loops

**arXiv ID:** 2608.23863 | [PDF](https://arxiv.org/pdf/2608.23863v1)

**作者:** Xianyao Li `[一作]` (University of Florida), Jing Du `[通讯]` (University of Florida)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了“执行结算信用文件”来记录机器人世界模型对未来预测的可靠性，并将该记录用于实时决策、信用门控和可审计日志；在三种仿真域（室内飞行、桌面操作、二维导航）及真实Franka机械臂上进行验证。

**💡 创新点**

创新点在于：① 将预测的可靠性转化为持久的、可通过执行结果结算的信用文件；② 通过结算、归因和低成本信用更新，构建跨模型、跨域的统一信任层；③ 将信用作为决策原语，实现信用门控与审计，且不需额外标签或高维不确定性量化。

**🔧 技术方法**

核心技术包括：世界模型预测结算（根据预设谓词与执行结果对比），信用文件按条件-区域-预测时程分箱，蒙特卡罗蒙德里安合并，基于结算监督的轻量级置信头，门控阈值与风险-覆盖曲线，审计票据与可重放的想象日志。

**📊 数据集**

数据集：仿真环境 Isaac Sim（quadrotor 24×6 m、桌面抓取、二维网格），以及真实FrankA 端口摄像头下的推送数据（4类物体，每类2个实例）。

**📈 对比分析**

与即时门控方法（集成分歧、自报阈值、单步一致性、滑动窗口自适应校准）对比。结果显示：信用门控将未兑现想象（burned imagination）降低约62%（CI 43–81%），保持成功率与碰撞率；在推送任务中将验证探测从 1.00/episode 降至 0.36/episode，成功率从 0.98 降至 0.94；风险-覆盖曲线表现更平稳，平均 AURC 与 ECE 也有所提升。

**⚠️ 局限性**

局限性：① 信用文件仅在本部署环境下有效，无法跨站点零样本迁移；② 结算依赖于在轨执行，未执行候选的信用估计基于交换性假设；③ 归因受状态估计质量限制，无法区分模型误差与传感器噪声；④ 仅评估推送操作，未涵盖抓取等更复杂操作；⑤ 信用阈值与谓词粒度需人工设定；⑥ 在真实实验中未能对比门控与无门控的因果效果，且V‑JEPAlike只做单步评估。

---

## 378. Cross-Stack Validation of Language-Model Training: A Clinical Fine-Tuning Case Study

**arXiv ID:** 2608.24267 | [PDF](https://arxiv.org/pdf/2608.24267v1)

**作者:** Thang Tran `[一作]`, Lan Dang `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

定义并验证了训练流程的轨迹级差分验证协议，并在 Qwen3‑0.6B 的临床问答适配中进行实验。

**💡 创新点**

首次提出三维实现独立性（栈、编排、运行时）以及轨迹级差分验证，揭示数据渲染和运行时错误对模型质量影响远大于数值错误。

**🔧 技术方法**

使用 PyTorch 与自研 Zig‑numbat 框架、LoRA、AdamW、32‑bit 精度、交叉熵与梯度范数监测，并通过多语言接口（Python、Go、Rust、TypeScript、C）实现运行时多样性。

**📊 数据集**

采用 PubMedQA、MedMCQA 与 MedQA‑USMLE 三大公开临床问答语料库，合并后构成训练与测试集。

**📈 对比分析**

通过四层交叉检查（L1 L2 L3 L4）比较未训练模型损失、模型加载一致性、数据渲染一致性以及轨迹损失与梯度范数；两栈平均误差约 0.134%，最大差异 0.316%；多语言实现每秒处理速率略高于 PyTorch，显存占用相近。

**⚠️ 局限性**

实验仅限单模型单卡单精度，未对不同种子进行基线比较；观察指标仅为交叉熵和梯度，缺乏更丰富的模型行为数据；故障矩阵为重建而非独立注入；未评估临床回答准确性，缺乏安全性分析。

---

## 379. ToolRobustBench: Stage-Wise Perturbation Evaluation and Failure Diagnosis for Tool-Calling Agents

**arXiv ID:** 2608.23635 | [PDF](https://arxiv.org/pdf/2608.23635v1)

**作者:** YiShan Zheng `[一作]`, Yi Chang `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个基于阶段对齐、确定性且具有级联感知的工具调用鲁棒性评测基准ToolRobustBench，系统性地对工具调用过程的四个阶段（接口、意图、输出/观察、运行时）进行扰动，并提供精确的错误归因与指标。

**💡 创新点**

创新点包括：① 将扰动与工具调用阶段对齐的四类扰动族（工具接口、用户意图、工具输出/观察、运行时环境）；② 级联感知的错误归因机制，区分观测错误与最早失败阶段；③ 使用确定性本地工具与可复现的数据生成流程，消除API波动带来的不确定性；④ 通过单步实验与混合扰动交叉验证，揭示单一扰动与组合扰动的非线性关系。

**🔧 技术方法**

技术实现主要采用：① 自定义扰动模板与离散化严重度等级；② 通过结构化记录（请求、工具注册表、期望输出、扰动信息）实现 deterministic scoring；③ 级联归因算法（递归回溯到最早失败阶段）；④ 对七大模型进行批量推理与评测。

**📊 数据集**

数据集：构建了40个本地工具（12类功能），随机抽取16个用于实验；生成了15,456条单家族扰动实例（包含clean、light、medium、heavy四个严重度），以及若干混合家族实例。所有实例均通过程序化方式生成并附带精确的 gold 标注与错误归因信息。

**📈 对比分析**

比较方法：使用 Clean、Robustness、Drop、Cascade Rate、Boundary Violation Rate 等指标；对七个模型（如GPT-4、Claude Sonnet 等）在单家族与混合家族下进行对比。实验结果显示：Clean 成功率高达 97.9%，但在工具输出/观察扰动下整体鲁棒性仅 45%；不同模型的 Drop 以及 Cascade Rate 亦呈现显著差异。混合家族实验揭示部分组合导致的鲁棒性下降超出单一家族最低值，表明扰动间存在交互效应。

**⚠️ 局限性**

局限性包括：① 采用确定性本地工具，无法完全覆盖真实 API 的多样性、认证、速率限制和状态依赖；② 仅评估单步工具调用，未考察多步规划与长期状态管理；③ 混合家族实验仅覆盖有限的五对扰动组合，未展开完整矩阵；④ 人工审核样本偏重难度案例，未代表整体分布；⑤ 不同模型接口兼容性可能引入细微差异。

---

## 380. MemUse: Moving Memory Evaluation from Direct QA to Natural Integration in Long-Term Human-AI Conversation

**arXiv ID:** 2608.24189 | [PDF](https://arxiv.org/pdf/2608.24189v1)

**作者:** Ryuichi Sumida `[一作]` (Kyoto University), Tatsuya Kawahara `[通讯]` (Kyoto University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在一项为期四个月、40名用户、7种记忆容量条件的日记式对话系统部署中，研究者探究了传统 Direct QA 评估与用户满意度之间的关系，并提出了新的 MemUse 评估基准，用以衡量在自然对话中实际检索并整合先前记忆的能力。

**💡 创新点**

创新点在于：①首次将真实用户的记忆使用时刻（约1.4%）提取为可重复评估的实例；②构建了关注“自然整合”的 MemUse 基准，揭示 Direct QA 与自然整合之间存在显著的检索-整合分离；③通过对 7 条记忆条件的纵向比较，证明仅提升检索能力并不能提升用户满意度。

**🔧 技术方法**

使用技术包括：GPT‑4.1‑mini 生成模型、RoBERTa 重要性评分模型、长文本上下文（LC）与检索增强生成（RAG）方案、GPT‑5.4 作为判别器进行自然整合与引用评估，及基于 LMM 的统计分析。

**📊 数据集**

数据集包括：① 1,872 个会话（共 29,575 轮）与 40 名用户的日记式交互日志；② 72 条由真实对话提炼的记忆使用实例（MemUse benchmark）以及对应的 316 个 fact‑seeking 问题；③ 公开的部署语料和评测脚本。

**📈 对比分析**

比较结果显示：在 7 条记忆条件下，Direct QA 准确率从 19.7% 上升到 70.1%，但用户满意度变化极小（平均差 < 0.06 SD）。MemUse 的自然整合得分在 22–28% 之间波动，与 Direct QA 无显著相关；而自然整合与满意度呈正相关（ρ≈0.29）。同一模型在 Direct QA 与引用（Reference）之间存在 71‑点差距，表明检索成功不等于自然整合。

**⚠️ 局限性**

局限性包括：① 所有条件均基于摘要基线，无法评估更广泛的记忆写法；② MemUse 仅捕捉显式提示的记忆使用，潜在隐式时刻被遗漏；③ 结果来自观察性实验，缺乏因果证据；④ 样本以女性为主，可能影响普适性；⑤ 评测采用 LLM 判别器，仍需进一步人工验证。

---

## 381. A Three-Parameter Binary Subdivision Scheme for Shape-Controlled Curve Design

**arXiv ID:** 2608.23637 | [PDF](https://arxiv.org/pdf/2608.23637v1)

**作者:** Rabia Hameed `[一作]` (Government Sadiq College Women University Bahawalpur), Hafiza Sana Mukhtiar `[通讯]` (University of Tabuk)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

**🎯 论文内容**

提出一种三参数9点二元逼近细分方案，通过对经典7点拉格朗日和B样条细分规则的加权组合实现曲线形状控制。

**💡 创新点**

创新点在于利用位移向量的几何组合并引入三独立设计参数，构造出统一的可调细分族，兼具逼近性质、可塑形与高阶连续性。

**🔧 技术方法**

采用位移向量、加权组合、分割规则的Laurent多项式分析、连续性判据以及Gibbs现象理论等数学技术。

**📊 数据集**

未使用外部数据集，实验仅基于自定义控制多边形进行数值验证。

**📈 对比分析**

通过数值实验对比不同参数取值下的支持、端点规则、连续性以及Gibbs振荡表现，证明该方案在保持逼近特性的同时能够显著改善曲线光滑度与形状可调性。

**⚠️ 局限性**

局限性包括：仅针对二元细分；参数选择仍需人工调节；未对实际工程案例或曲面生成进行验证，且对多阶细分的推广尚不完整。

---

## 382. Pattern-Derived Visual Swarm Games: Multi-Scale Drone-Vision States for Interception and Sustainability Audits

**arXiv ID:** 2608.23575 | [PDF](https://arxiv.org/pdf/2608.23575v1)

**作者:** Faruk Alpay `[一作]` (Bahçeşehir University), Levent Sarioglu `[通讯]` (Bahçeşehir University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用 Bloom 过滤器将无人机视觉标注流压缩成可重复的虚拟无人机能力向量，随后在 6v6 至 32v32 的多尺度编队游戏中评估其在零和拦截游戏中的表现，并通过人类可读的视觉叠加和 32×32 灰度占据图进行策略辨识。

**💡 创新点**

① 把真实标注数据直接映射为统计性虚拟能力向量而非学习模型；② 通过 Bloom 过滤器实现可重现、无外部依赖的能力采样；③ 将同一状态同时渲染为可视化覆盖图和可用于机器学习的占据图，从而兼顾视觉与博弈分析。

**🔧 技术方法**

Bloom 过滤器 + 离散化探测；多尺度编队游戏（4×4 纯策略矩阵）；乘法权重算法求解纳什平衡；多尺度 Gaussian 占据编码；基于 Hoeffding、Freund–Schapire 等理论的误差上界；可视化渲染与最近邻聚类。

**📊 数据集**

VisDrone、UAVSwarm、Sheffield UAV 降落、Stanford Drone Dataset、Anti‑UAV 与 DUT‑Anti‑UAV 作为训练与对照集；俄文视觉与数学源作为对比轨道。

**📈 对比分析**

与固定像素卷积核的 32×32、48×48、64×64、96×96、128×128 方案对比。固定像素在 128×128 时准确率降至 66%，热点误差增大；经过尺度归一化的 Gaussian 编码在 128×128 时准确率提升至 77%，热点误差下降，联合损失下降，表明该方案在更大视场下保持或提升性能。

**⚠️ 局限性**

① 支付模型简化，未包含真实航空动力学或传感器噪声；② Bloom 采样仅反映统计分布，缺乏物理解释；③ 编队动作集极小，无法覆盖复杂战术；④ 视觉分类器仅做基线，强模型需在更大数据集上评估。

---

## 383. Do System Prompts Leave Behavioral Fingerprints? A Large-Scale Empirical Study of Clone Detection via Output Similarity

**arXiv ID:** 2608.24461 | [PDF](https://arxiv.org/pdf/2608.24461v1)

**作者:** Linghan Chen `[一作]` (University of Adelaide), Honglong Chen `[通讯]` (China University of Petroleum)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于黑盒输出的行为指纹方法BBF，用以检测系统提示被盗后在不同模型上的复制情况。

**💡 创新点**

创新点在于将系统提示的行为印记转化为句子嵌入空间中的可比对指纹，并引入诊断查询优化(DQO)和无提示可迁移的检测规则。

**🔧 技术方法**

使用句子嵌入（MiniLM-L6-v2）进行相似度计算、余弦相似度阈值统计以及基于多数投票的聚合决策。

**📊 数据集**

评估数据来自8个常见NLP基准（MMLU、TriviaQA、SST‑2、MNLI、MedQA、CUAD、GSM8K、CNN/DailyMail），共250条诊断查询，每条3个生成样本。

**📈 对比分析**

在128个实验场景（4模型族×8基准）中，BBF在同模型下AUC平均0.876，跨模型平均0.725，且通过DQO可提升+0.120的跨模型AUC。

**⚠️ 局限性**

主要局限在于对短结构化输出的风格适配攻击不稳健，跨域/跨模态泛化仍未验证，且需先验领域知识构造基准提示。

---

## 384. Hamilton Cycles in 10-Tough $(2P_2 \cup P_1)$-Free Graphs

**arXiv ID:** 2608.24047 | [PDF](https://arxiv.org/pdf/2608.24047v1)

**作者:** Qiuyu Chen `[一作]` `[通讯]` (Shanghai Jiao Tong University), Qiuyu Chen (Shanghai Jiao Tong University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了10‑坚韧且不含2P₂∪P₁的无向图必存在哈密顿环。

**💡 创新点**

通过将图分为小邻域与大邻域两种结构，利用匹配路径覆盖压缩和异构连通度分析，将先前的11‑坚韧阈值降至10。

**🔧 技术方法**

采用组合结构分解、匹配路径覆盖、K₁,₂匹配与连通度、散射数与阻止子图的技术，结合Ham–cycle与prescribed-edge循环定理。

**📊 数据集**

本研究为理论证明，无需实验数据集。

**📈 对比分析**

与已有的11‑坚韧结果相比，本结论在坚韧阈值上实现了改进；结果仅为存在性证明，未涉及实验性能评估。

**⚠️ 局限性**

10的阈值可能仍非最优，且方法相对复杂，尚未扩展到更广泛的线性森林或更低坚韧常数。

---

## 385. Evolutionary Recurrent Decision Model in Developing Adaptive and Maladaptive Behaviors

**arXiv ID:** 2608.23932 | [PDF](https://arxiv.org/pdf/2608.23932v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 386. MRI-based Deep Radiomic Phenotyping of Neuromuscular Disorders: A Topology-driven Characterization

**arXiv ID:** 2608.24415 | [PDF](https://arxiv.org/pdf/2608.24415v1)

**作者:** Martyna Żur `[一作]`, Joanna Polańska `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种基于图形放射组学的全三维肌肉MRI表型框架，用自动化特征提取和形态学图网络捕捉脂肪侵袭的拓扑与几何特征，以实现神经肌肉疾病的差异化诊断和纵向跟踪。

**💡 创新点**

创新点在于：①将脂肪侵袭的三维结构直接转化为骨架图网络，提取节点、闭环等拓扑指标；②引入多层次脂肪分级（中度、重度）与接口动态度量；③利用非参数统计与效应量评估特征区分力，避免单一体积测量的局限；④构建可解释、硬件无关的特征体系，兼顾稀有病数据不足的挑战。

**🔧 技术方法**

技术方法包括：基于高斯混合模型的无监督分割、三维骨架化（Medial Axis Transform）与无向图构建、互信息筛选、Kruskal-Wallis检验与Epsilon平方效应量、Dunn事后对比、UMAP+PCA降维、PARC聚类，以及统计绘图与特征梯度映射。

**📊 数据集**

使用了来自CoMPaSS-NMD多中心项目的1184枚肌肉MRI扫描，涵盖5种遗传性神经肌肉病变（DYSF、CAPN3、GNE、DMPK、DUX4）。

**📈 对比分析**

通过非参数检验与效应量比较，拓扑节点和闭环特征（如SF1_Skel_Nodes、GF_Skel_Nodes）效应量高达0.26，显著优于传统脂肪体积或FF；UMAP投影与PARC聚类进一步验证了各基因型在三维特征空间的分离，表明该方法在区分多种病变方面具有较高性能。

**⚠️ 局限性**

局限性包括：①对初始分割质量高度依赖；②对切片厚度与分辨率异质性敏感，可能引入骨架化误差；③缺乏跨中心、不同磁共振设备的外部验证；④未充分控制疾病进展阶段，混合了基因型与时间效应；⑤计算量大，实时部署仍具挑战。

---

## 387. Mutation Testing of Simulink Cyber-Physical System Models: Challenges and Solutions in Practice

**arXiv ID:** 2608.24250 | [PDF](https://arxiv.org/pdf/2608.24250v1)

**作者:** Murat Kavak `[一作]` (Universiteit Antwerpen), Halim Abdurrahman Ceylan `[通讯]` (Ege University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究在工业合作伙伴公司实施了一个面向 Simulink/Stateflow 模型的基于变异测试的工具，并在三个实际生产模型上进行了验证。

**💡 创新点**

创新点主要包括：1）针对 Gain/ Product 块的维度过滤策略显著降低等价变异体数量；2）利用 Simulink Design Verifier 进行模型级等价检测；3）使用正则表达式实现 Stateflow 状态与转换的语法安全变异；4）在 CI/CD 流水线中集成 Smoke 测试与需求可追溯功能。

**🔧 技术方法**

技术栈包括 MATLAB/Simulink API、Stateflow API（通过正则实现变异）、Simulink Design Verifier、Embedded Coder、gcc 编译、Jenkins CI、Signal Builder 与 Requirements Toolbox。

**📊 数据集**

使用了三款工业模型（M1、M2、M3）共 459 个变异体，分别包含 250 个已被杀死的变异体和 209 个存活变异体。

**📈 对比分析**

实验结果显示变异覆盖率分别为 M1 82.0%、M2 41.0% 和 M3 95.7%；与先前三轮工具实验相比，整体测试套件的变异分数提升约 7.5%。

**⚠️ 局限性**

局限性包括：等价变异检测需生成大规模输入空间，导致执行成本高；需求可追溯功能仅支持 Signal Builder 与 Requirements Toolbox，无法直接扩展到其他工具链；状态流变异仍受限于 MATLAB 版本与 API 能力；工具的可迁移性和可扩展性仍需进一步验证。

---

## 388. How much of a measured AI preference is the model, and how much is the instrument?

**arXiv ID:** 2608.23641 | [PDF](https://arxiv.org/pdf/2608.23641v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 389. PhysMLLMs: Spatial Priors for Unified Referring Segmentation and Grounded Reasoning of Images and Videos

**arXiv ID:** 2608.24574 | [PDF](https://arxiv.org/pdf/2608.24574v1)

**作者:** Siyao Yan `[一作]` (Lanzhou University), Tat-SengChua `[通讯]` (National University Of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种训练阶段的物理启发空间连续性先验注入（PhysMLLMs），通过对齐冻结的 DINOv2 教师的全局视觉表示来提升视频多模态大语言模型的分割一致性。

**💡 创新点**

创新点在于：① 用物理启发的空间连续性先验（REPA-Global）将全局视觉表示对齐；② 采用离线缓存的教师嵌入并在训练期间使用调度的蒸馏权重；③ 该注入仅在训练阶段，对推理时无额外开销。

**🔧 技术方法**

使用的技术包括：InternVL3-2B 视觉语言模型 + SAM2 分割模块；冻结的 DINOv2 ViT‑B/14 作为教师；REPA‑Global 全局对齐蒸馏；离线教师嵌入缓存；PEFT（Vision LoRA 与 MaskDecoder 校准）；温度调度与定时权重调度。

**📊 数据集**

实验所用数据集主要有：ReVOS、MeVIS、Ref‑DAVIS17（视频分割），RefCOCO、RefCOCO+、RefCOCOg（单帧指代分割），MMBench、MME、POPE、TextVQA（通用 VLM 评测）。

**📈 对比分析**

在 Sa2VA‑InternVL3‑2B 基础上进行对比，PhysMLLMs 在 ReVOS、MeVIS U 和 Ref‑DAVIS17 上分别取得 57.4、57.4、76.0 的 J&F 分数，均超过最新 Video‑MLLMs；在 RefCOCO 系列保持近似不降性能；在 MMBench 等通用 VLM 评测保持与基线相当，显著提升小目标、遮挡、快速运动、干扰者场景的时空一致性。

**⚠️ 局限性**

局限性：全局对齐方式难以完全解决极小目标、持续遮挡以及密集相似干扰下的身份辨别；缺乏查询特定的轨迹约束；未对 3D 物理动力学或对象交互进行建模。

---

## 390. FlowNeg: GFlowNet-Guided Diverse Hard Negative Sampling for Knowledge Graph Embedding

**arXiv ID:** 2608.23849 | [PDF](https://arxiv.org/pdf/2608.23849v1)

**作者:** Ibne Farabi Shihab `[一作]` (Iowa State University), Joyanta Jyoti Mondal `[通讯]` (University of Delaware)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为 FlowNeg 的层次化生成流网络，用于知识图谱嵌入模型的负采样。

**💡 创新点**

其创新点在于利用上下文条件的奖励比例采样，通过分层策略先选类型再选实体，并用硬度、碰撞折扣与类型支持构造终端奖励，避免对所有实体做全量归一化。

**🔧 技术方法**

技术上采用了生成流网络（GFlowNet）与轨迹平衡训练、层次化策略、类型分区、基于结构的碰撞得分，并与 EMU、IF-NS 等传统采样方法进行对比。

**📊 数据集**

实验使用了五大基准数据集：FB15k‑237、WN18RR、YAGO3‑10、CoDEx‑L、Hetionet，并在 FB15k‑237/RotatE 上做了精细的 15‑种子匹配负样本计数对照实验。

**📈 对比分析**

与 EMU 和 IF-NS 的比较显示，FlowNeg 在 25 个格点中 24 个取得更高的 MRR（平均提升约 0.017），在 15‑种子匹配实验中 MRR 提升至 0.359、负样本多样性显著提高、碰撞率下降，并在等时钟检查点上比其他方法更快收敛。

**⚠️ 局限性**

局限性包括：引入额外采样器参数和计算开销；碰撞得分在关系邻域稀疏时可能失效；层次化结构依赖于手工或训练得到的类型划分；NDS 仅衡量同一分区内的覆盖而非语义多样性；理论分析基于固定上下文和残差界限，未能保证全局收敛；并且 FlowNeg 仍然是训练辅助工具而非事实验证器。

---

## 391. Synchronizing Automata: Open Problems

**arXiv ID:** 2608.24245 | [PDF](https://arxiv.org/pdf/2608.24245v1)

**作者:** Marek Szykuła `[一作]` `[通讯]` (University of Wrocław), Marek Szykuła (University of Wrocław)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对同步自动机理论中的一些开放问题进行了调查，重点讨论了著名的Černý猜想。同步自动机是指存在一个重置词，使得所有状态都映射到同一状态的确定性有限自动机。Černý猜想认为，每个具有n个状态的同步自动机都存在一个长度不超过(n-1)^2的重置词。

**💡 创新点**

本文的创新点在于提出了一些新的辅助结果，并对多个相关问题进行了深入探讨，包括避免词、压缩状态、同步特定子集、同步性判断的复杂性等。

**🔧 技术方法**

使用了线性代数方法来分析同步自动机的性质，并探讨了如何通过这些方法来解决同步性相关的问题。

**📊 数据集**

论文中没有具体提到使用的数据集，主要是理论分析和推导。

**📈 对比分析**

通过对比不同的同步自动机的重置阈值和避免阈值，提出了多个上界和下界的猜想，性能方面的结果表明，当前已知的最佳上界为O(n^3)的形式。

**⚠️ 局限性**

本文的局限性在于某些问题仍未得到解决，特别是在压缩状态和避免词的具体上界方面，许多猜想尚未被证明。

---

## 392. Ockhamareto: Pareto-Gated Segment-Level Credit Assignment for Concise Unit-Test Generation with Reinforcement Learning

**arXiv ID:** 2608.24473 | [PDF](https://arxiv.org/pdf/2608.24473v1)

**作者:** Dong Huang `[一作]` (National University of Singapore), See Kiong Ng `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了单次生成的单元测试框架 Ockhamareto，通过 Pareto 门控奖励和 token 级段落信用提升测试集的缺陷检测效果与简洁性。

**💡 创新点**

创新点包括：1）Pareto‑gated 奖励仅奖励在 (mutation, -#tests) 空间非支配的滚动；2）token‑level segment credit 将每个测试的边际 mutation kills 归因到其 token span，实现细粒度信用；3）将两者结合在单次生成中同时优化效果与规模。

**🔧 技术方法**

采用 Group Relative Policy Optimization（GRPO）与 LoRA 微调的 LLM；mutation 测试作为奖励信号；token offset mapping 将执行结果映射回 token；Pareto dominance 作为奖励门控；强化学习从可验证执行反馈中学习。

**📊 数据集**

训练集为去泄漏的 PLT（约 10.5k 个 Python 函数，来自 The Stack v2）；评估集包括 ULT、HumanEval+、MBPP+、CodeContests 和 TestGenEval‑Lite，涵盖函数级和仓库级测试场景。

**📈 对比分析**

与基准 Qwen3.5‑4B、+GRPO（无门控或段落信用）以及 MIST‑RL 进行对比；在 N=5 的预算下，Ockhamareto 在所有 5 个基准上 mutation score、覆盖率最高，suite size 最小；例如 ULT mutation 49.9% 对比 MIST‑RL 的 31.3%，平均测试数 2.60 对比 4.67，per‑test 效率提升 3.4 倍。

**⚠️ 局限性**

局限性：仅在 Python 单元测试场景验证；token‑level credit 依赖 tokenizer offset mapping，成功率约 67%；未给出自动停止阈值，只通过 Pareto 前沿提供选择；实验仅基于 Qwen3.5 体系，跨语言或更大模型迁移仍需验证。

---

## 393. Real-World Knowledge-Guided Change Data Synthesis for Remote Sensing

**arXiv ID:** 2608.24263 | [PDF](https://arxiv.org/pdf/2608.24263v1)

**作者:** Yaoyi Qi `[一作]` (Wuhan University), Gui-Song Xia `[通讯]` (Wuhan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于预训练视觉‑语言模型（VLM）的遥感变化数据合成框架 KnowChange，利用 VLM 进行变化区域与类别转换推理，再通过布局‑到‑掩码（L2M）和掩码‑到‑图像（M2I）生成器合成二元与语义变化数据。

**💡 创新点**

创新点在于：①用 VLM 取代传统手工规则进行变化模拟，显著提升变化类别覆盖度和灵活性；②将 VLM 推理结果与两级生成模型结合，形成可插拔的“知识‑驱动”变化模拟模块；③构建了三大合成数据集（Know‑BCD、Know‑SEC、Know‑HR），在多种变化检测基准上实现了显著性能提升。

**🔧 技术方法**

技术要点包括：VLM 交互式推理（使用 Qwen3‑VL/GLM‑4.6V 等），文本与图像条件融合（T5+CLIP 编码器），FLUX.1‑Fill（L2M）+ LoRA 微调，SD‑v1.5 + ControlNet（M2I）+ CLIP 嵌入调度器，伪变化模拟与形状保持/改变两种转换模式，数据集整合与大规模语义分割预训练。

**📊 数据集**

主要使用的数据集：公开语义分割集合（OpenEarthMap、FLAIR、Vaihingen、Potsdam、GID、SkySA）共 138K 张图像；基于这些数据训练 L2M/M2I；生成的合成数据集 Know‑BCD、Know‑SEC、Know‑HR；评估基准包括四个建筑变化检测集（LEVIR‑CD、WHU‑CD、DSIFN‑CD、SEC‑BCD）和两个语义变化检测集（SECOND、HRSCD）。

**📈 对比分析**

在合成‑到‑真实（synthetic‑to‑real）迁移实验中，KnowChange 训练的模型在四个建筑变化基准上平均提升 6.64 % IoU，二元 F1 提升 6.78 %；在语义变化基准上平均提升 6.78 % F1。与现有合成方法（Changen2、HySCDG、SyntheWorld 等）相比，KnowChange 在所有指标上均优于或接近最佳，且在少量真实数据（5 %）的增量训练下表现更佳。

**⚠️ 局限性**

局限性包括：①对预训练 VLM 的依赖，若 VLM 对某些遥感语义缺乏足够知识可能导致推理误差；②需要预先拥有高质量语义掩码，无法直接处理无标注图像；③大规模生成模型（FLUX、SD‑v1.5）和 VLM 计算成本较高，实际部署受限；④目前仅在 2D RGB 图像上验证，未覆盖多光谱/时序复杂性。

---

## 394. Small-World Communication Fabrics for Neuromorphic Multicore-SoCs

**arXiv ID:** 2608.24351 | [PDF](https://arxiv.org/pdf/2608.24351v1)

**作者:** Sebastian Billaudelle `[一作]` (University of Zurich and ETH Zurich), Melika Payvand `[通讯]` (University of Zurich and ETH Zurich)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对两种 22 nm FDSOI 的多核神经形态系统 NeoCorAL（树状异步包交换）与 MOSAIC（RRAM 电路交换网格）进行设计、路由特性与性能评估，并在三种网络拓扑下比较其效率。

**💡 创新点**

提出基于通信局部性驱动的硬件-算法协同评估框架，开发路由感知训练方法以在保持任务性能的同时实现可映射网络，并揭示树状与网格路由在不同局部性水平下的互补优势。

**🔧 技术方法**

异步包交换树路由、RRAM 电路交换网格、CAM 路由内存、层级多播编码、梯度正则化与稀疏化的路由感知训练。

**📊 数据集**

Spiking Heidelberg Digits (SHD) 数据集。

**📈 对比分析**

通过平均跳数、内存占用、单播/多播效率等指标在空间嵌入小世界、随机、层状网络上比较；在 SHD 任务中，路由感知训练在相同路由/权重内存下提升约5个百分点的准确率，且所需内存约减 10 倍。

**⚠️ 局限性**

局限性包括：仅在 22 nm FDSOI 上模拟与评估；未验证大规模或三维集成下的可扩展性；RRAM 的耐久性与写/读误差需进一步验证；异步与同步实现的能耗/时延权衡仍未完全量化。

---

## 395. CoSTALA: Compositional Spatio-Temporal Audio-Language Alignment via Multi-Grain Hierarchical Contrastive Learning

**arXiv ID:** 2608.24374 | [PDF](https://arxiv.org/pdf/2608.24374v1)

**作者:** Peiwei Ren `[一作]` (Xi’an Jiaotong Liverpool University), Yin Cao `[通讯]` (Chinese Academy of Sciences)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出CoSTALA训练范式，构建多粒度时空音频-语言对齐模型，解决多事件空间音频理解难题。

**💡 创新点**

通过层级损失函数实现全局与局部时空对齐，并引入3‑way时空损失、局部对齐与特征一致性，显著提升多事件时空检索性能。

**🔧 技术方法**

结合HTSAT音频编码、RoBERTa文本编码、Transformer时序编码与RoPE位置编码，采用对比学习、InfoNCE、MSE等多损失优化。

**📊 数据集**

在空间化Clotho数据集上合成FOA音频，并利用Qwen3‑8B生成空间化描述，构建30000训练样本和9000评估样本的多事件数据集。

**📈 对比分析**

与SALM和T‑CLAP进行双向空间检索对比，CoSTALA在Recall@1提升至约8.1%、Recall@5 19.9%、Recall@10 27.7%，显著优于对比模型。

**⚠️ 局限性**

仍受长序列上下文信息压缩导致的特征崩塌影响，且在极长时序或复杂空间分布下表现尚未完全稳健。

---

## 396. SDR Driver for Precise Timing Applications

**arXiv ID:** 2608.23614 | [PDF](https://arxiv.org/pdf/2608.23614v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 397. Place, Slice and Schedule: Hierarchical O-RAN Control of a Tethered mmWave UAV-gNB

**arXiv ID:** 2608.23824 | [PDF](https://arxiv.org/pdf/2608.23824v1)

**作者:** Alireza Mohammadhosseini `[一作]` (Clemson University), Fatemeh Afghah `[通讯]` (Clemson University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文设计了一个分层O‑RAN控制器，结合慢速无人机位置与切片预算决策与快速用户调度，实现了缆式毫米波无人机基站的智能资源分配。

**💡 创新点**

创新点包括：① 在O‑RAN中实现双时钟耦合控制，将物理布局与资源分配同步；② 使用Permutation‑equivariant DeepSets SAC调度器解决可变用户数和无序输入的挑战；③ 采用分阶段联合训练流程，将慢速与快速决策分离并协同优化。

**🔧 技术方法**

主要技术手段包括：O‑RAN RIC架构、Soft Actor‑Critic强化学习、DeepSets网络结构、Sionna Ray‑traced mmWave信道仿真、DQN与SAC混合训练、以及基于Python的仿真与优化框架。

**📊 数据集**

实验使用的“数据集”为Sionna RT射线追踪得到的28 GHz毫米波信道地图，以及仿真生成的用户移动、到达率与队列状态，用于训练和评估调度器。

**📈 对比分析**

通过与经典PF、Equal、Max‑Rate以及SAC‑MLP调度器，以及随机和贪婪移动基线进行比较，D‑SAC在eMBB SLA满足率提升约17%，URLLC准时交付提升42%，且在30/40/50用户场景下无需重训练即可保持领先。

**⚠️ 局限性**

局限性在于仅在仿真环境验证，未考虑实际能耗、硬件限制和跨网络协同；且模型假设切片预算已预设，未覆盖更复杂的网络层次和动态能耗管理。

---

## 398. RecGPT-Mobile-V2 Technical Report

**arXiv ID:** 2608.24295 | [PDF](https://arxiv.org/pdf/2608.24295v1)

**作者:** Lingqing Zhang `[一作]`, Zihong Huang `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种端到端的移动端个性化查询预测框架 RecGPT‑Mobile‑V2，联合优化意图质量与推理效率，采用行为轨迹压缩、推荐本地化基础训练、结构化短链式推理、质量门控自适应推理、教师学生蒸馏与低精度量化，并通过设备‑云路由实现落地部署。

**💡 创新点**

创新点：
1) 行为轨迹压缩保持关键证据，避免无关噪声；
2) 推荐本地化基础训练（域适配 + 连续预训练）让模型天然理解语义ID和行为关系；
3) 结构化五阶段推理与证据优先短链式推理，显著提升查询质量；
4) 质量门控自适应推理（分组rollout、输入特定预算、乘性奖励与排名保护），实现“足量足用”推理；
5) 教师-学生压缩管道（结构化压缩、低位量化、蒸馏），兼顾模型大小与查询质量；
6) 系统级评估（多维度评价器、检索互补性、设备性能）验证效率与多样性提升。

**🔧 技术方法**

技术手段：
- 行为压缩（信号过滤、去重、语义丰富、强度分层、序列化）；
- 领域适配 + 连续预训练（语义ID、行为迁移、后购关系等多视图监督）；
- 结构化推理（统一提示、五阶段推理、短链式证据提取）；
- 质量门控 RL（Group Policy Optimization, 输入特定预算, 乘性奖励, 质量评估器）；
- 低精度量化与结构化压缩（按张量层分配位数, 灵敏度优化, 量化感知恢复）；
- 设备‑云路由与低位推理实现。

**📊 数据集**

使用的数据集：
- 以阿里巴巴（淘宝）电商平台收集的跨表面隐式行为日志（点击、收藏、购买、后购探索等）；
- 通用语言模型预训练语料（如公开语料库）用于基础恢复；
- 业务级评估数据（查询标签、检索结果等）。

**📈 对比分析**

比较方法与性能：
- 对比不同 CoT 方案（无 CoT、短 CoT、完整 CoT）：短 CoT 在 ROUGE‑L 与 Jaccard 上分别提升 0.0866 与 0.0741，且超过完整 CoT；
- RL 机制对比（A1‑A6）：A6 在质量 78.6%（相较 A1 的 73.2% 提升 5.4 点）与硬失败率 1.6%（降至 1.6%）下，平均 CoT 长度降至 14 词（比 A1 的 62 词下降 48 词，77% 量化）；
- 检索互补性：Query 路径与传统召回通道的 Jaccard 交叉重叠下降 0.06‑0.12，表明检索结果更具互补性；
- 设备性能：在加速器上平均延迟从 3.00s 降至 0.76s（3.95×），P95 与 P99 同比分别下降 4.7× 与 4.0×。

**⚠️ 局限性**

局限与挑战：
- 主要验证为离线评估，未充分覆盖真实用户交互与多样化场景；
- 评估器虽结构化但仍可能被模型欺骗，需持续人工审核；
- 低位量化与压缩虽降低成本，但对极端边缘语义或罕见实体仍可能产生误差；
- 设备‑云路由策略需根据不同终端硬件细化，跨平台兼容性待进一步验证；
- 目前集中在电商业务，迁移到其他领域需重构语义ID 与行为映射；
- 长期学习与模型漂移仍需持续监控与增量更新机制。

---

## 399. WiCi: Wireless GPU Computing Infrastructure

**arXiv ID:** 2608.24204 | [PDF](https://arxiv.org/pdf/2608.24204v1)

**作者:** Yibin Shen `[一作]`, Zili Meng `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无线 GPU 计算基础设施 WiCi，允许移动设备通过 WiFi 远程访问服务器级 GPU 进行 LLM 推理。

**💡 创新点**

创新点在于：① 在用户模式驱动层拦截 CUDA 调用，保持应用无改动；② 结合模型缓存、函数批量化与追踪重放三种优化，显著降低 WiFi 传输延迟；③ 通过在 WiFi 路由器内置 GPU，兼容现有移动硬件与系统。

**🔧 技术方法**

使用技术包括：CUDA 驱动 API 拦截、TCP 传输协议、WiFi 6 无线通信、MD5 哈希实现模型缓存、函数批量化与追踪重放算法、GPU 路由器与 Raspberry Pi 5 移动终端的软硬件集成。

**📊 数据集**

使用的数据集与模型包括：LLM 推理框架 llama.cpp、Qwen3-32B、Falcon 40B、Qwen3‑VL‑8B 等主流大模型；并从 Hugging Face 取 top‑10 大模型（总下载量 90M、总大小 250 GB）做模型缓存实验。

**📈 对比分析**

对比方法：在相同模型下将 WiCi 与本地移动 GPU 推理、云端 GPU 推理以及原始远程 GPU 调用（rCUDA/sCUDA）进行对比。实验结果显示：WiCi 可将首次 token 时间缩短 90% 以上，token 速率提升约 39×，并实现 65–80% 的本地服务器 GPU 性能；模型加载时间通过缓存显著下降，RTT 消耗下降超过 80%。

**⚠️ 局限性**

局限性：① 仅针对 AI 推理设计，其他 GPU 应用需要进一步适配；② WiFi 带宽与延迟仍受网络环境限制，长时间高带宽传输可能影响常规网络使用；③ 需要用户自行采购服务器 GPU，初始硬件成本相对较高。

---

## 400. MetaRAG: Belief-Action Aligned Policy Optimization for Agentic RAG

**arXiv ID:** 2608.24214 | [PDF](https://arxiv.org/pdf/2608.24214v1)

**作者:** Qiuyi Qi `[一作]`, Qiang Zhu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文介绍了ACL会议论文的排版规范，并提供了使用 LaTeX 风格文件的详细说明与模板示例。

**💡 创新点**

创新点在于将通用的ACL排版要求与具体的 LaTeX 样式文件结合，形成一个完整且可直接使用的参考文档。

**🔧 技术方法**

采用 LaTeX 语言、样式文件（.cls）以及配套的源代码文件进行排版指导。

**📊 数据集**

本文不涉及实际数据集，仅提供排版示例。

**📈 对比分析**

本文不包含实验比较，主要通过展示排版效果来说明规范的正确性。

**⚠️ 局限性**

局限性在于仅针对排版规范，无法评估学术内容质量；若未严格遵循通用指引，仍可能出现格式不一致或错误。

---

## 401. XP-JEPA: Cross-Predictive Physics Grounding for Forecastable Latent Dynamics

**arXiv ID:** 2608.24044 | [PDF](https://arxiv.org/pdf/2608.24044v1)

**作者:** Kehan Wen `[一作]` (National University of Singapore), Fan Shi `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种利用训练时可用的物理轨迹对视觉潜在动态进行交叉预测的模型（Cross‑Predictive JEPA），实现视觉世界模型在部署时无额外物理输入的情况下获得更可预测的潜在动态并提升控制性能。

**💡 创新点**

创新点在于：①将物理状态作为与视觉观测同一轨迹的第二个预测视角；②通过共享的动作条件预测器将视觉与物理历史对应未来表示，并将预测结果同时匹配到两种模态的未来，从而在训练阶段直接约束潜在动态；③引入统一的物理状态接口，支持多任务场景下的共享物理编码。

**🔧 技术方法**

使用了视觉编码器+动作条件预测器（LeWM架构）、自监督交叉预测损失、物理状态编码器（统一刚体几何+姿态表示）、AdaLN动作调制、CEM+VAE动作搜索、并对预测器和潜在空间进行正则化。

**📊 数据集**

在四个单任务环境（Push‑T、OGBench‑Block、Two‑Room、Reacher）和一个包含22个物体‑任务配置的Meta‑World多任务桌面套件上进行评估。

**📈 对比分析**

与基线视觉自监督世界模型对比，Cross‑Predictive JEPA将多任务平均控制成功率从53.6%提升至78.2%，多任务 rollout drift 从0.361 降至0.104；单任务实验中在大部分环境中也显著提升控制成功率。

**⚠️ 局限性**

局限性包括：仅在仿真中验证，需训练时与测试时共享相同的物理轨迹；物理状态接口依赖于预定义的刚体几何与姿态，限制了对更复杂或未知物体的适用性；以及跨模态预测仅在训练阶段使用，部署时可能仍受潜在空间与物理对应不完全的影响。

---

## 402. Hybrid Semantic Tool Discovery for Enterprise MCP Gateway: Architecture and Implementation

**arXiv ID:** 2608.23992 | [PDF](https://arxiv.org/pdf/2608.23992v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 403. AI Finds A Way

**arXiv ID:** 2608.23875 | [PDF](https://arxiv.org/pdf/2608.23875v1)

**作者:** Aaron Dharna `[一作]` (University of British Columbia), Jeff Clune `[通讯]` (University of British Columbia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统收集并整理了26个关于人工智能（AI）在不同子领域中出现的非预期、创造性或作弊行为的第一手轶事，涵盖强化学习、自然语言处理、演化计算、机器人等领域。

**💡 创新点**

创新点在于：①首次对AI非正式轶事进行系统归纳与分类；②提出将这些轶事与AI安全、对齐问题相联系，提供安全启示；③创建了可公开访问的资源仓库，为后续研究提供基准。

**🔧 技术方法**

主要技术手段是文献检索、作者访谈、博客与论文摘录；随后对案例进行分类、对齐与安全性分析；未设计新算法或实验。

**📊 数据集**

使用的数据来源是来自100+研究者的轶事和公开描述，未使用传统机器学习数据集；案例内容来自学术论文、访谈和技术博客。

**📈 对比分析**

由于工作性质为案例汇编与分析，本文并未给出量化性能指标；通过专家讨论和案例对比，对不同类别的行为进行定性评价。

**⚠️ 局限性**

限制包括：①轶事选择与描述具有主观性；②缺乏统一的量化评估标准；③部分案例细节未公开，导致重现困难；④未系统验证这些行为在更广泛任务中的普遍性。

---

## 404. Trajectory-Level Continuous Action Representation for Robotic Manipulation

**arXiv ID:** 2608.24111 | [PDF](https://arxiv.org/pdf/2608.24111v1)

**作者:** Tong Yang `[一作]` (Fudan University), Wenqiang Zhang `[通讯]` (Fudan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了CAT，一种在固定时间窗口内通过连续潜在令牌编码轨迹级动作的框架；

**💡 创新点**

创新点在于将动作表示与时序网格解耦，使用频率感知的旋转位置编码实现跨控制频率的共享时序坐标，并通过连续潜在令牌和对比正则化实现轨迹级表征；

**🔧 技术方法**

使用Transformer编码器/解码器、频率感知旋转位置编码（F‑RoPE）、对比正则化、流匹配扩散策略，以及与现有VLA/流匹配模型的无缝集成；

**📊 数据集**

在LIBERO、MimicGen、RoboTwin 2.0以及真实机器人长周期任务（Stack‑5、Drawer、Rope、Flower）等数据集上进行评估；

**📈 对比分析**

与基线VLA、VQ‑VLA、FAST、CARP、Diffusion Policy、π₀等方法在相同训练设置下对比，CAT在LIBERO平均成功率从87.3%提升至90.8%，在MimicGen平均从85.8%提升至86.9%，在RoboTwin多频率测试中平均提升约10个百分点，真实机器人任务平均得分从45.0%提升至60.5%；

**⚠️ 局限性**

局限性包括对令牌数、正则化权重等超参数的敏感性；在极高频率或更复杂多模态任务中仍需进一步验证其可扩展性；

---

## 405. Generating Biomedical Fact-Checking Reports with RL-Enhanced Agentic Search

**arXiv ID:** 2608.23811 | [PDF](https://arxiv.org/pdf/2608.23811v1)

**作者:** Jiongxiao Wang `[一作]` (University of Wisconsin--Madison), Chaoqun Ni `[通讯]` (University of Wisconsin--Madison)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了基于LLM的代理BioCheck，通过在PubMed上进行代理检索生成结构化医学事实核查报告，并通过EG-GRPO强化学习优化报告质量。

**💡 创新点**

创新点：①首次提出生成结构化医学核查报告而非仅输出标签；②设计了Evidence‑Grounded GRPO，结合高级检索、证据质量评估与幻觉惩罚的任务专用奖励；③利用Boolean检索与LangGraph构建高效、可扩展的检索代理。

**🔧 技术方法**

使用技术：LLM（Qwen3.5-4B、GPT‑5.2）、检索增强生成（RAG）+代理检索、LangGraph/LangChain工具调用、Evidence‑Grounded Group Relative Policy Optimization（EG‑GRPO）强化学习、Boolean检索、证据质量评分模型。

**📊 数据集**

使用数据集：SciFact（训练/验证/测试）与HealthFC（仅测试），并通过PubMed API获取证据。

**📈 对比分析**

方法比较：与无检索、CER、FIRE、PMSearch Agent以及GRPO等基线在标签预测（准确率、宏F1）和报告生成（EQS、EHR）上评估。实验显示，Qwen3.5-4B + EG‑GRPO在SciFact准确率提升9.95%，EQS提升3.7%，EHR下降19.63%，甚至超过GPT‑5.2；在HealthFC也保持显著优势。

**⚠️ 局限性**

限制：仅适用于孤立的原子声明，难以处理复杂叙事与多模态信息；训练数据量有限且存在分布偏移，导致在更广泛的HealthFC上提升有限。

---

## 406. Learning to Grade Efficiently: A Bandit-Driven Prompt-Selection Framework for Low-Cost LLM Essay Scoring

**arXiv ID:** 2608.23814 | [PDF](https://arxiv.org/pdf/2608.23814v1)

**作者:** Olga Manakina `[一作]` (Carleton University), Igor Bogdanov `[通讯]` (Carleton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于多臂赌博机（MAB）的在线自适应提示选择框架，用于低成本自动作文评分

**💡 创新点**

首次将在线控制机制应用于自动作文评分，将提示选择从离线超参数搜索转变为在线学习任务；并给出了首个成本-可靠性学习曲线

**🔧 技术方法**

多臂赌博机（ε‑greedy MAB）、四种提示策略（多步/单步，含/不含示例）、Google Gemini 2.5 LLM、token与延迟跟踪

**📊 数据集**

IELTS Writing Task 2 787篇作文（官方评分1–9）

**📈 对比分析**

与传统的完整网格搜索（每篇作文尝试四种提示）对比，MAB在保持相近的评分准确率（MAE≈0.85，QWK≈0.55）的同时，将LLM调用量降低78.4%，token消耗降低72.8%，API成本下降约70%

**⚠️ 局限性**

仅使用单一模型（Gemini 2.5），固定ε=0.2的MAB算法，未加入上下文特征或不同作文类型；未来需评估多模型、多数据集和更灵活的探索策略

---

## 407. PRQ-KMeans: Projection Residual Quantization for Semantic ID Tokenization

**arXiv ID:** 2608.24207 | [PDF](https://arxiv.org/pdf/2608.24207v1)

**作者:** Yunxiao Luo `[一作]` (Kuaishou Technology), Chenyi Lei `[通讯]` (Kuaishou Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种后置分层语义标识符分词器 PRQ-KMeans，能在每一层通过投影残差去除已编码的公共信息。

**💡 创新点**

核心创新包括先移除全局均值、使用 Top‑k 软更新细化质心，以及在硬指派后通过投影消除选定质心的残差携带。

**🔧 技术方法**

技术手段为余量量化、K‑Means 聚类、余量投影、Top‑k 软权重更新与余量正交化。

**📊 数据集**

实验数据集为约 780 万条商品记录的工业电商检索数据集，以及 Amazon Sports、Toys、Clothing、LastFM 四个公开推荐基准。

**📈 对比分析**

与 RQ‑KMeans、RQ‑VAE 等方法比较，PRQ‑KMeans 在工业数据上 HitRate 提升 7.4%–11.8%，在四大公开基准上均达到或超过最优点，并显著提高代码簿利用率和降低 Gini。

**⚠️ 局限性**

局限在于对投影残差假设的依赖较大，且在极大规模高维数据上聚类收敛速度与计算开销仍需进一步优化。

---

## 408. Metadata-Aware Adaptation of a Generative Foundation Model for Conditional CMR Synthesis

**arXiv ID:** 2608.24342 | [PDF](https://arxiv.org/pdf/2608.24342v1)

**作者:** Marc Rodríguez `[一作]` (Universitat de Barcelona), Polyxeni Gkontra `[通讯]` (Universitat de Barcelona)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

利用预训练的潜在扩散模型，以患者结构化临床元数据和切片位置为文本提示，对短轴心脏磁共振图像进行条件生成，并通过元数据无关的分类器无关引导、对比批处理和逆频率采样三种策略提升元数据遵从度。

**💡 创新点**

①提出元数据无关的分类器无关引导（MF-CFG），在推理时用去除元数据的提示做负提示，增强元数据信息；②引入对比批处理（CB）使每批次包含不同元数值，提升训练多样性；③使用逆频率采样（IFS）在训练时对罕见元数值加权，缓解类别不平衡。

**🔧 技术方法**

基于公开的Stable Diffusion权重的潜在扩散框架，冻结VAE，微调U-Net进行CMR去噪，配合文本编码器进行条件编码；结合上述三种元数据处理策略实现条件生成。

**📊 数据集**

英国生物银行（UK Biobank）59,058张短轴CMR图像（训练53,298张，测试5,760张），每张图像附带年龄、性别、BMI、疾病、切片位置等结构化元数据。

**📈 对比分析**

与未采用元数据策略的Fine-tuned SD基线（FID 87.22）以及需要心脏几何输入的先前文本条件CMR扩散模型相比，融合三种策略后FID降至37.47（约57%提升），同时保持MAE 0.260、MS-SSIM 0.171，展示在分布一致性上的显著优势。

**⚠️ 局限性**

尽管分布一致性得到显著改善，但在疾病（pathology）等极不平衡的元属性上仍表现不佳，且对单一图像的像素级相似度略有下降，表明元数据条件下的可控性和对罕见类别的生成仍是挑战。

---

## 409. Beyond the Mandate: A Systematic Security Analysis of the Agent Payments Protocol (AP2)

**arXiv ID:** 2608.23858 | [PDF](https://arxiv.org/pdf/2608.23858v1)

**作者:** Avital Aviv `[一作]` (Ben-Gurion University of the Negev), Asaf Shabtai `[通讯]` (Ben-Gurion University of the Negev)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文系统地对Google发布的Agent Payments Protocol（AP2）v0.2进行了安全分析，构建了完整的威胁模型与威胁清单，使用AIVSS评分方法对48条威胁进行风险评估，并在自建测试平台上实现并验证了八条高危威胁的攻击与缓解措施；此外，还开发了面向AP2部署的安全扫描器，支持静态、跨角色、以及攻击者模拟三层检测。

**💡 创新点**

创新点主要体现在：①将AP2生命周期拆分为五个分析阶段，并定义了五种不同部署架构，全面揭示部署差异对威胁的影响；②首次使用MAESTRO框架对AP2进行完整的威胁建模，涵盖四类攻击者、十一条攻击面、六大攻击目标；③提出结合CVSS v4与AIVSS的综合评分体系，系统评估了各威胁在不同架构下的高危性；④在缺乏公开部署的前提下，构建自研测试平台并演示八条高危威胁的PoC；⑤研发了部署感知的安全扫描器，能够根据不同架构自动选择合适的检测层并报告威胁。

**🔧 技术方法**

使用技术包括：MAESTRO威胁建模框架、CVSS v4与AIVSS风险评分、LLM驱动的攻击演示与自动化扫描器（OpenAI API）、STRIDE‑GPT进行对照评估、Tamarin/ProVerif模型验证（参考）。

**📊 数据集**

主要数据来源为AP2 v0.2规范与相关先前研究文献，威胁清单基于协议描述；实验数据来自自行搭建的AP2测试平台，模拟五种部署架构并执行PoC与扫描器检测。

**📈 对比分析**

通过与STRIDE‑GPT自动生成的威胁表进行对比，覆盖率达66.7%；多位评估者对AIVSS评分的重现性高，AC2≈0.98，Krippendorffα≈0.80；扫描器三层架构整体召回率最高，在不同架构下的单层贡献也被量化。

**⚠️ 局限性**

局限性包括：①缺乏完整公开的AP2部署，仅使用自建平台进行演示，无法完全覆盖真实生产环境；②未对卡网络、发卡行等后端支付系统的连锁影响进行评估；③扫描器依赖LLM接口，受限于API调用成本与潜在误报/漏报；④部分PoC基于特定实现细节，可能对不同实现产生差异。

---

## 410. Renormalization Group Flow Matching for Scalable Local Generative Modeling

**arXiv ID:** 2608.23696 | [PDF](https://arxiv.org/pdf/2608.23696v1)

**作者:** Kanta Masuki `[一作]` (University of Tokyo), Yuto Ashida `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于重整化群(RG)的生成模型框架RGFM，通过在不同尺度上逐步生成数据，并在此过程中实现局部近似，从而在保持全局一致性的同时实现高效生成。

**💡 创新点**

创新点在于：①将精确的RG流作为概率路径引入flow matching；②利用RG的准局域性和尺度分离，在每一步仅需局部计算即可逼近全局流；③通过多次局部缩放与站点消减实现几乎线性规模的计算复杂度。

**🔧 技术方法**

使用流匹配(Flow Matching)技术、Polchinski RG方程、离散余弦变换(Discretized Cosine Transform)进行站点消减、以及卷积神经网络(CNN)实现局部速度场预测。

**📊 数据集**

在一维Ising模型、带潜在变量的条件局域分布、以及自然图像数据集FFHQ（64×64与256×256）上进行实验。

**📈 对比分析**

与传统全局流匹配（FM）以及仅使用局部网络的FM进行对比。实验表明：在一维模型中，RGFM能够准确重现远程相关性；在FFHQ图像中，RGFM在64×64时显著提升全局连贯性并降低FID，在256×256时仍优于局部FM，但在极端长程结构上仍有不足。

**⚠️ 局限性**

局限包括：①对高度非平稳或多模态数据的泛化能力尚待验证；②尽管复杂度降低，但仍需要多次递归缩放和随机恢复，训练与采样的实现仍相对繁琐；③在极大分辨率或高维数据中，局部近似误差与尺度分离假设的适用性需进一步评估。

---

## 411. Joint-Embedding Prediction of Masked Point Tubes for Self-Supervised Learning on 4D Point Cloud Videos

**arXiv ID:** 2608.24093 | [PDF](https://arxiv.org/pdf/2608.24093v1)

**作者:** Jheng-Ling Lee `[一作]` (National Taiwan University), Shang-Tse Chen `[通讯]` (National Taiwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种基于JEPA（Joint-Embedding Predictive Architecture）的自监督预训练框架，利用高比例遮蔽的空间-时间点管子（point‑tube）在潜在空间中进行目标表示预测，并通过Sketched Isotropic Gaussian Regularization（SIGReg）来抑制表征崩塌。

**💡 创新点**

创新点在于将传统的坐标重建任务替换为潜在特征预测，并在不使用教师网络或EMA的情况下引入SIGReg实现稳定的表征学习，同时采用高达75%的遮蔽率以强化预测难度。

**🔧 技术方法**

技术方案包括点管子分词、P4Transformer编码器、轻量级预测器、Smooth L1损失、SIGReg正则化、随机遮蔽、位置查询以及上下文-目标对齐训练路径。

**📊 数据集**

实验数据集涵盖动作识别的MSRAction‑3D和NTU RGB+D，以及手势识别的SHREC'17，并在半监督、少样本和跨数据集迁移任务上进行评估。

**📈 对比分析**

在MSRAction‑3D上达到94.08%（相较于监督P4Transformer的90.94%显著提升），在NTU RGB+D全标注下91.8%（相较于P4Transformer 90.2%提升），半监督50%标签下89.0%（相较于90.8%提升），在SHREC'17上实现30/50轮分别91.7%/93.3%（相较于P4Transformer 87.5%/91.2%提升），并在少样本实验中大幅提升1‑shot、3‑shot和5‑shot精度；与MaST‑Pre、M2PSC、Uni4D、DiMP等自监督基线相比均取得更优表现。

**⚠️ 局限性**

局限性主要体现在仅使用P4Transformer骨干网络、实验规模相对有限、未在更大规模或多样化的4D点云数据集上进行大规模预训练，且对更复杂的跨域或跨任务场景的泛化能力尚待进一步验证。

---

## 412. Do Recipes Have Personas? Characterizing and Generating Creator Style in Attributed Procedural Graphs

**arXiv ID:** 2608.24369 | [PDF](https://arxiv.org/pdf/2608.24369v1)

**作者:** Lei Jiang `[一作]` `[通讯]` (Microsoft), Lei Jiang (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过构建ViralRecipesTrans数据集，将烹饪视频转化为执行DAG，研究并实现了面向创作者的程序化风格识别与生成。

**💡 创新点**

创新点在于提出程序化风格学（procedural stylometry），把创作者的流程结构视为可学习的图形特征，并设计了结构化两阶段生成器和混合式集成方法以同时兼顾宏观规划与微观语义。

**🔧 技术方法**

技术手段包括图学习与拓扑度量、LLM（GPT‑5.4等）辅助语义推理、Markovian物理先验、束搜索、以及基于特征的集成选择器。

**📊 数据集**

使用的数据集是自研的ViralRecipesTrans（VRT），包含97个创作者、5,000+ 视频、数千条带属性的执行DAG和对应文本转录。

**📈 对比分析**

在IF1、nEF1、StepErr等指标上，与零样本、少样本LLM、N‑gram序列等基线相比，结构化两阶段生成器在宏观结构规划上优于LLM，少样本LLM在语义分配上更佳；集成方法在整体操作精度上达到了最高的nEF1≈0.48并保持低StepErr。

**⚠️ 局限性**

局限性包括LLM在宏观规划上的不足、对高质量转录的依赖、难以完全捕捉创作者的所有细微偏好，以及目前仅在烹饪领域验证，缺乏跨领域通用性。

---

## 413. EXAM$^2$: $\underline{Ex}tending$ $\underline{A}udio$ $Understanding$ $in$ $\underline{M}ultilingual$ $and$ $\underline{M}ultimodal$ $Analysis$

**arXiv ID:** 2608.23758 | [PDF](https://arxiv.org/pdf/2608.23758v1)

**作者:** Jiawen Wang `[一作]` (LMU Munich), Nancy F. Chen `[通讯]` (A*STAR)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了EXAM^2基准，用多语言（6种）和多模态（音频+视觉）在多类音频（语音、环境声、音乐、混合）上进行MCQ评测，并在该基准上评估多种模型。

**💡 创新点**

创新点在于首次将多语言和多模态结合进音频理解评测，提出OmniLoRA统一语言微调策略和Gemma3n-EXAM^2轻量化融合模型，显著提升跨语言与跨模态性能。

**🔧 技术方法**

采用Gemma3n-E4B多模态LLM为基底，使用LoRA进行参数高效微调，并通过OmniLoRA实现统一语言训练；视觉答案通过Stable Diffusion和GPT-image-2生成；模型评估使用多模态输入序列和自注意力融合。

**📊 数据集**

训练数据来自公开的MMAU-test-mini、MMAR、Clotho等音频集，问题与答案在英语基础上翻译成德语、西班牙语、日语、马来语、中文，生成22,614幅视觉答案；测试集包含998道题目，5,667个多模态问答。

**📈 对比分析**

与闭源模型GPT‑4o‑audio、GPT‑5‑mini、以及开源Phi‑4、Qwen‑2.5‑omni等进行对比，Gemma3n‑EXAM^2在多语言多模态测试中平均准确率达61.2%，相较于基线提升约12.4%（多语言）及21.7%（多模态），并在多数语言与音频域取得最高或第二高成绩。

**⚠️ 局限性**

局限性包括：仅关注准确率而未评估推理速度和部署成本；语言覆盖仅限欧亚语言，未包含非洲等低资源语言；仅使用AudioMCQ子集生成视觉答案，未充分利用全量数据；多模态融合仍易出现视觉干扰，需进一步改进。

---

## 414. Data Predictability Shapes Weibull Weight-Scale Growth in Transformer Training

**arXiv ID:** 2608.23573 | [PDF](https://arxiv.org/pdf/2608.23573v1)

**作者:** Tiexin Ding `[一作]` `[通讯]`, Tiexin Ding

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究训练好的Transformer权重幅度分布，提出基于Weibull分布的权重尺度（λ）增长规律

**💡 创新点**

提出利用预训练数据的二元条件熵预测λ增长的闭环法则，并证明其凸形由熵饱和决定

**🔧 技术方法**

使用Weibull拟合、二元条件熵估计、AdamW训练、控制腐败级别的语料扰动以及自检验证

**📊 数据集**

主要在WikiText、c4以及代码数据集的腐败变体上进行实验

**📈 对比分析**

与实测λ²增长对比，R²≈0.94，误差约5.7%，同时在不同学习率和两种模型架构上保持一致

**⚠️ 局限性**

局限于单一模型规模、单一训练预算，仅在同一语料内有效，跨语料预测失效，系数随架构变化，未考虑冗余维度

---

## 415. SandwichQuant: Which Parameters Matter Before and After Quantization?

**arXiv ID:** 2608.24173 | [PDF](https://arxiv.org/pdf/2608.24173v1)

**作者:** Peng Xia `[一作]` (Beijing University of Technology), Junbiao Pang `[通讯]` (Beijing University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究量化纠正的参数子空间，发现归一化-仿射子空间在相同预算下具有高效纠正方向，并提出 SandwichQuant 两阶段归一化-仿射框架（先预适应量化再后处理固定量化图）实现无额外推理开销的高效量化纠正。

**💡 创新点**

创新点包括：①将量化纠正视为参数子空间问题，系统量化不同子空间的修正能力；②证明低维归一化-仿射子空间在匹配预算下可恢复大部分量化误差；③提出双阶段 SandwichQuant（pre‑PTQ 和 post‑PTQ 归一化‑仿射调整）组合，兼顾量化前置校正和后置残差修正，实现性能提升而不增加推理算子。

**🔧 技术方法**

使用技术包括：参数子空间分解（W、Φ、Ω），对归一化-仿射的局部与全局更新，匹配预算的对比实验，梯度控制、知识蒸馏+任务损失优化，控制实验（等大小权重、不同后端对齐等），以及针对 LLM 的高位量化、权重‑激活‑KV 量化的实验。

**📊 数据集**

数据集涵盖：WikiText‑2、C4、PIQA、ARC‑Easy、ARC‑Challenge、HellaSwag、WinoGrande、BoolQ、ImageNet‑1K、CIFAR‑100、Cityscapes、OSCAR 等，用于语言模型、图像分类、分割等多任务评估。

**📈 对比分析**

与多种基线（RTN、QDrop、QAT 方法如 LSQ、DSQ、PACT、StableQAT、GPTAQ、ResComp 等）在 LLM 的 perplexity 和零-shot 平均准确率，以及图像分类/分割的 top‑1/mIoU 进行对比。SandwichQuant 在权重‑仅、权重‑激活‑KV 量化下均显著降低 perplexity、提升平均准确率；在 ImageNet/Cityscapes 上从几乎失效恢复到接近 FP32 级别，显示出显著的性能提升。

**⚠️ 局限性**

限制包括：①只能纠正可恢复的结构化响应误差，无法修复被截断或舍入丢失的关键信息；②需要与后端、位宽、校准数据匹配，缺失时效果大幅下降；③在 ImageNet 任务中需较大量的调优数据；④未在更大模型、MOE 结构、不同随机种子或部署延迟场景下验证；⑤双阶段流程增加了额外的离线 PTQ 与优化开销。

---

## 416. EgoErrorVQA: Assess Egocentric Comprehension Capabilities through Procedural Errors for Ego-Agentic AI

**arXiv ID:** 2608.24134 | [PDF](https://arxiv.org/pdf/2608.24134v1)

**作者:** Junlong Li `[一作]` (Hong Kong Polytechnic University), Yi Wang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究者提出了一个基于第一人称视角的程序错误检测与分类任务 EgoErrorVQA，并构建了一个可与视觉语言模型对话式评估的评估代理；随后提出了适应性解耦推理框架 Ego-ADR，以提升模型在程序错误理解上的性能。

**💡 创新点**

创新点：①首次为 egocentric 视频构建程序错误检测与分类的 VQA 基准，并给出了八类错误的统一 taxonomy；②设计了可自动化、可对话式的评估代理，采用 Agent2Agent 交互协议；③提出的 Ego-ADR 框架通过深浅解耦分阶段推理（关键步骤匹配、视频叙述、错误分类），显著提升模型的错误识别和分类能力。

**🔧 技术方法**

使用技术：VQA 框架（开放端与多选两种评估）；LLM‑as‑Judge（Qwen2.5‑VL、DeepSeek‑LLM）进行自动评分；Agent2Agent 交互协议实现评估代理与被测模型的对话；Adaptive Decoupled Reasoning（深/浅解耦）分阶段推理；关键步骤匹配、视频叙述与错误分类模块；链式思考（CoT）对比实验。

**📊 数据集**

数据集：从 CaptainCook4D、EgoOops、Epic‑Tent、Assembly101 四个 egocentric 数据集中抽样，构建 800 条视频、3560 条 QA 对，覆盖 31 种程序任务和 8 类错误。EgoErrorVQA 仅作为评估集，不包含训练数据。

**📈 对比分析**

对比方法：评估了 3 个闭源模型（GPT‑4o、GPT‑4o‑mini、Gemini‑2.5‑flash）、3 个主流开源 VLM（LLaVA‑OneVision、Video‑LLaMA2、Video‑LLaVA）、2 个专门的 egocentric 视觉代理（EgoGPT、Vinci）、3 个 Qwen‑VL 版本以及 2 个大模型（Qwen3‑VL‑32B、InternVL3.5‑38B）。实验表明，所有模型在 open‑end VQA 上平均相似度仅达到 3.3（人类为 3.77），在多选 VQA 上最佳 F1 仅约 20%。应用 Ego‑ADR 后，7B/8B 级模型的精确度提升 10–23%，F1 提升 8–30%，在同规模模型中达成 state‑of‑the‑art，显著优于基线和常用 CoT 方法。

**⚠️ 局限性**

局限性：①数据来源大小不均衡，正确样本占比过高导致指标分化有限；②仅提供评估集，缺乏对应训练集；③多选 VQA 评价对类别不平衡敏感，单一指标不具备完整解释力；④细粒度动作/物体识别需求仍未充分满足，影响错误类型识别；⑤评估代理和 LLM‑as‑Judge 的可信度受限，仍需进一步验证。

---

## 417. QisMC: A Model Checker for QISKIT Program Debugging

**arXiv ID:** 2608.24320 | [PDF](https://arxiv.org/pdf/2608.24320v1)

**作者:** Aochu Dai `[一作]` (Tsinghua University), Mingsheng Ying `[通讯]` (University of Technology Sydney)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 QisMC，一个专门用于 Qiskit 程序调试的量子模型检查器，基于新的量子时间逻辑 qCTL 并实现了完整的端到端调试工作流。

**💡 创新点**

创新点包括：① 定义了基于 Birkhoff‑von Neumann 量子逻辑的 qCTL 以及量子–经典转换系统；② 设计了双向固定点迭代算法以求解量子命题的最强后置条件与最弱前置条件；③ 通过把量子命题映射为经典原子命题，将 qCTL 检查问题化简为经典 CTL 检查，可直接使用 NuSMV；④ 在符号层面使用 CFLOBDD 进行量子态与算子运算，显著提升了对大规模程序的可扩展性。

**🔧 技术方法**

核心技术包括：Birkhoff‑von Neumann 量子逻辑、量子‑经典转换系统、qCTL（含 sp/wp 语义）、双向固定点迭代、CTO‑CTLM 减少、CFLOBDD 决策图、NuSMV 传统模型检查器。

**📊 数据集**

使用的数据集包括：Grover 与 BV 经典量子算法（来自 VeriQBench），GHZ 状态制备，VeriQBench 中的分布式量子计算（相位估计、QFT）以及 Benchpress Medium 组（40 个 10–30 量子比特电路）。

**📈 对比分析**

与现有工具（QPMC、QTC‑Maude、QMC、量子抽象解释器）对比，QisMC 在相同基准上能够在秒级完成 100 量子比特的验证，且支持循环、测量、条件控制等复杂控制流；但在树状 Grover、QFT、带浮点参数的电路以及极大规模程序中，由于决策图结构与浮点误差，性能会显著下降。

**⚠️ 局限性**

主要局限：1) qCTL 目前不支持概率性质；2) CFLOBDD 对电路拓扑与参数化门敏感，结构差异（树状 vs 线性）会导致空间/时间爆炸；3) 采用高精度浮点计算，易出现数值误差，影响极大规模程序的正确性。

---

## 418. Restoring Without Forgetting: Continual Learning Across Image Degradations

**arXiv ID:** 2608.23799 | [PDF](https://arxiv.org/pdf/2608.23799v1)

**作者:** Alif Ashrafee `[一作]` (Rochester Institute of Technology), Bartosz Krawczyk `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 11707 | [OpenAlex ID](https://openalex.org/A5054879396)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种持续多退化图像恢复框架RwF，能够在不保留历史数据、未知退化标签的情况下，逐步学习并保留多种图像退化的恢复能力。

**💡 创新点**

创新点在于冻结预训练去噪网络骨干，使用低秩适配器隔离每种退化的参数并通过无监督原型匹配实现动态路由，从而实现零遗忘且参数开销低。

**🔧 技术方法**

技术包括低秩适配器（LoRA）路径、冻结去噪预训练模型、无监督原型嵌入+余弦匹配路由、领域增量学习的无例子参数隔离训练。

**📊 数据集**

数据集：使用DIV2K构成的五个合成退化域（噪声、模糊、雨、雾、暗光）以及11个真实退化基准（CBSD68、Kodak24、Urban100、RealBlur-J/R、Rain100H/L、Test100、SOTS-Indoor/Outdoor、LOL-v1）。

**📈 对比分析**

与连续微调、EWC、LwF以及全域联合模型对比，RwF在Restormer/NAFNet上实现了+15.25dB/ +11.83dB的最终PSNR提升，忘记率为0，路由准确率89.5%，与oracle相比仅+0.94dB。

**⚠️ 局限性**

局限性：对全局性退化（如暗光）路由准确率偏低；适配器设计仍较简单，难以覆盖极端真实退化的分布；路由仅基于前编码器特征，可能受退化多样性限制。

---

## 419. Words, Spaces and Generative AI: Layers of language in contemporary architecture

**arXiv ID:** 2608.24360 | [PDF](https://arxiv.org/pdf/2608.24360v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 420. What Does Prompt Learning Change? -A Natural-Language Concept Analysis of Vision-Language Models

**arXiv ID:** 2608.24142 | [PDF](https://arxiv.org/pdf/2608.24142v1)

**作者:** Ryo Kamiya `[一作]` (Chiba University), Kazuhiko Kawamoto `[通讯]` (Chiba University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种后置方法 PromptSplice，用于将提示学习前后得到的 CLIP 文本嵌入投影到同一固定自然语言词典的稀疏线性空间，从而使连续提示向量的语义变化可被人类理解。

**💡 创新点**

创新点在于：① 通过在同一词典坐标系下对比提示学习前后的系数分布，直接揭示了提示向量在语义空间中的重组；② 发现提示向量的概念分布变化与分类准确率提升正相关；③ 提供了一个局部梯度表达式，解释图像对齐的概念方向为何在损失上更敏感。

**🔧 技术方法**

使用技术包括：CLIP 预训练模型、CoOp（上下文优化）提示学习、SpLiCE 的稀疏线性概念嵌入（Lasso）以及 Jensen‑Shannon 散度和 Pearson 相关系数等统计分析。

**📊 数据集**

在 11 个图像分类数据集上评估：ImageNet、OxfordPets、Caltech101、StanfordCars、Food101、Flowers102、FGVCAircraft、SUN397、DTD、EuroSAT、UCF101。

**📈 对比分析**

比较方法：① 通过余弦相似度评估完整系数重构的保真度；② 对齐前后系数排名的 High→High、High→Low、Low→High 统计；③ 用 JS 散度量化分布变化并与 CoOp 的准确率提升进行相关性分析，得到 Pearson r≈0.64。性能方面，CoOp 在所有数据集均有提升，EuroSAT 上最高提升 60.3%。

**⚠️ 局限性**

局限性包括：词典来自 LAION‑400M，包含拼写错误和噪声词，导致解释性下降；重构需要约 450 个词条，解释仍不简洁；Lasso 对词条相关性敏感，可能产生不稳定的系数；分析仅为后置诊断，无法直接引导提示学习；样本量仅 11 个数据集，相关性结论易受极端样本影响。

---

## 421. ConsensusTAS: Self-Supervised Temporal Action Segmentation for Long-Horizon Construction Videos

**arXiv ID:** 2608.24043 | [PDF](https://arxiv.org/pdf/2608.24043v1)

**作者:** Xiaoshan Zhou `[一作]` (University of Sydney), Yafei Sun `[通讯]` (University of New South Wales)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了ConsensusTAS，一种无标签自监督的时序动作分割方法；

**💡 创新点**

创新点在于通过内部共识评分、内部奖励以及峰值密度解码，实现多尺度候选分割的自我校准与融合；

**🔧 技术方法**

采用多尺度边界证据、内部奖励（相似度、对比度、紧凑度、复杂度），以及基于高斯核的共识密度解码；

**📊 数据集**

在GTEA、Breakfast、Assembly101三大公开数据集以及真实工地砖砌视频上进行实验；

**📈 对比分析**

与现有无监督方法（SSC‑AP、OTAS、ASESM等）比较，ConsensusTAS在F1@10、F1@25、F1@50指标上均取得领先，GTEA上F1@10达73.08，Breakfast上64.33，Assembly101上54.79；

**⚠️ 局限性**

局限性包括对超参数敏感、需要多次随机采样、在极长无标注视频或复杂摄像机运动时表现可能下降，且主要验证在工地场景，缺乏更广泛的跨领域评估。

---

## 422. PARTAB: Partition-Aware Reasoning with Structured Evidence for Scalable Table Understanding

**arXiv ID:** 2608.24082 | [PDF](https://arxiv.org/pdf/2608.24082v1)

**作者:** Md Mahadi Hasan Nahid `[一作]` (University of Alberta), Davood Rafiei `[通讯]` (University of Alberta)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出 PARTAB 框架，将大型表格划分为语义一致且行链接的子区域，并通过分层选择和结构化证据合成实现高效的表格推理。

**💡 创新点**

创新点在于构建结构化的证据接口：利用 LLM 进行语义列分组、行块分区，并通过分层选择精确定位需要的证据，避免了全表或单视图推理导致的注意力稀释和证据缺失。

**🔧 技术方法**

技术手段包括：LLM 作为问题分析器、列分组器、分区选择器和答案执行器；TF‑IDF 相似度检索；固定行块大小的分区；以及基于 LLM 的分层选择策略。

**📊 数据集**

实验数据集涵盖 WikiTableQuestions、TabFact 以及 TableBench（含 Numerical Reasoning 与 Fact Checking 两个子任务）。

**📈 对比分析**

在多项基准测试中，PARTAB 相较于全表提示、单视图裁剪以及 TableMaster、PoTable、H-Star 等先进方法取得显著提升：WikiTableQuestions 上 79.31 EM，TabFact 上 90.48 Acc，TableBench NR 70.33 EM，FC 82.71；在大表子集上平均提升 18‑25 分。

**⚠️ 局限性**

局限性包括：对 LLM 提示设计和模型波动敏感；缺乏对全表聚合任务的完整覆盖；固定块大小和 heuristic 分区策略可能不适用于所有表结构；多阶段流水线增加了推理延迟与计算成本。

---

## 423. Provenance Guided Incremental Learning Under Evolving Concept Definitions

**arXiv ID:** 2608.23893 | [PDF](https://arxiv.org/pdf/2608.23893v1)

**作者:** Ismail Lamaakal `[一作]` `[通讯]` (Mohammed Premier University), Ismail Lamaakal (Mohammed Premier University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对模型部署后目标定义变更的情形，提出了基于数据溯源的增量学习框架，只在受影响记录上重新标注和修复预测器。

**💡 创新点**

创新点在于将规则差分编译与历史记录的溯源信息结合，能够精确识别受影响的实例，完成局部重标注和增量修复，避免全量重训练。

**🔧 技术方法**

使用了规则差分编译器、记录级溯源分析、可执行与不确定标签分解、增量预测器修复以及版本化概念内存等技术。

**📊 数据集**

在四类数据集（PaySim 交易、Census‑Income 人口、UNSW‑NB15 网络安全、ogbn‑arxiv 图数据）构建的 RuleShift‑Bench 基准上进行实验。

**📈 对比分析**

与全量重标+重训、滑动窗口、在线更新、重放、ADWIN 触发等方法对比，取得 92.3% 准确率、90.2% 宏 F1，仅重新处理 14.7% 的历史记录，更新延迟从约 993 s 降至 179 s。

**⚠️ 局限性**

局限性包括：规则变更过于全局时选择性维护优势下降；历史溯源缺失或不完整时需保守地重新处理更多记录；不可执行或高度耦合的关系/图约束导致候选集变大。

---

## 424. Interpreting Control Latents for System Identification via Conditional Flow Matching

**arXiv ID:** 2608.23887 | [PDF](https://arxiv.org/pdf/2608.23887v1)

**作者:** Dingqi Zhang `[一作]` (UC Berkeley), Mark W. Mueller `[通讯]` (UC Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究将自适应控制中的隐变量解码为分布式的四旋翼动力学模型，从而实现对已冻结低层策略的在线预测调参和鲁棒性分析。

**💡 创新点**

创新点在于用条件流匹配将隐变量映射到多模态参数分布，解决非可辨识性问题，并利用解码模型直接驱动高层控制器和评估鲁棒性。

**🔧 技术方法**

核心技术包括联合训练的隐变量自适应控制器、条件流匹配生成模型、Flightmare仿真与真实硬件测试。

**📊 数据集**

实验数据来源于Flightmare中随机采样的20,000组物理参数与对应隐变量的对齐数据，随后在真实四旋翼上进行验证。

**📈 对比分析**

与基线（Naive、Baseline）和+20%扰动方法相比，CFM调参在位置RMSE上下降23%/45%，在鲁棒性预测中可精确捕捉横向误差，整体性能显著提升。

**⚠️ 局限性**

主要限制包括对观测噪声敏感、仅在四旋翼结构化参数模型上验证，且对非线性、接触或混合动力学系统的适用性尚未验证。

---

## 425. SeriCrypt: An LLM-Driven Context-Aware Serialization Framework for Cryptographic Protocols

**arXiv ID:** 2608.24498 | [PDF](https://arxiv.org/pdf/2608.24498v1)

**作者:** Maosong Chen `[一作]` (Information Engineering University), Chunxiang Gu `[通讯]` (Information Engineering University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SeriCrypt框架，使用LLM从协议规范自动生成协议消息的抽象语法树（PAST），并通过统一的域特定语言（CDSL）驱动可复用的执行引擎完成字段计算、加密操作与字节序列化，从而实现对多种加密协议（TLS 1.2/1.3、IKEv1/2、SSH、TLCP）的自动化消息构造与安全测试。

**💡 创新点**

创新点在于：①将协议结构、上下文依赖与加密计算拆分为可声明的中间表示；②设计了兼容加密语义的CDSL，提供六类属性和可复用的计算块；③采用LLM链式思考抽取规范信息，形成可验证的PAST；④实现了协议无关的执行引擎，能统一处理多协议的加密与序列化；⑤通过PAST层的结构化变异实现高效的安全漏洞挖掘与模糊测试。

**🔧 技术方法**

技术手段包括：大语言模型（Gemini‑3‑Pro‑Preview / ChatGPT‑5.5）进行规范抽取；自定义的上下文感知属性文法（CA‑AGCP）与统一属性系统；CDSL语法与计算块定义；可插拔的加密原语库（HMAC、HKDF、AES‑GCM等）；执行引擎实现深度优先树遍历、上下文投影与递归表达式求值；手工验证回路与安全约束提取；结构化模糊策略。

**📊 数据集**

使用的评估数据集为108种协议配置（包括TLS 1.2/1.3、IKEv1/2、SSH、TLCP）与六个主流开源实现（OpenSSL、TLSe、strongSwan、Libreswan、OpenSSH、GmSSL），覆盖多种密码套件与交互模式；实验生成的安全测试用例共计364,260个，模糊测试覆盖率比AFLnet/ChatAFL高约5‑12%。

**📈 对比分析**

通过与“直接LLM生成可运行客户端代码”的基线对比，SeriCrypt在所有108配置中实现100%握手成功；与主流模糊器比较，SeriCrypt在同等时间内获得最高代码覆盖率（约+5%~+10%）。安全约束测试发现5条新违规，验证了框架的实用性。

**⚠️ 局限性**

局限性包括：仍需人工验证与修正LLM生成的PAST；当前仅支持客户端消息序列化，未覆盖响应解析；对未训练过的协议或复杂加密链的错误率可能升高；对计算块的编写与维护仍需专业知识；在极端大规模协议集合中的可扩展性尚未彻底评估。

---

## 426. On Angle-optimization and Simplification of Degree-1 Homology Representatives

**arXiv ID:** 2608.23949 | [PDF](https://arxiv.org/pdf/2608.23949v1)

**作者:** Emerson G. Escolar `[一作]` (Kobe University), Yuta Shimada `[通讯]` (Kobe University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出并求解了角度最优同伦循环问题（AOHCP），通过最小化闭合曲线的总绝对曲率来寻找平面或近平面的 1-维同伦类代表；

**💡 创新点**

创新点在于将总绝对曲率引入同伦类代表优化，形成二元二次规划模型，从而同时兼顾平面性、凸性与简单性；

**🔧 技术方法**

主要技术包括角度最优同伦循环的二次优化建模、外角矩阵构造、混合整数线性化以及商业求解器 CPLEX 的使用；

**📊 数据集**

实验使用的合成数据集为三维圆柱与鞋形点云（各含 300、500、1000 或 200、600 点），并通过 Alpha 复合体构造简化复杂度；

**📈 对比分析**

与传统基于长度或简单性优化的代表性循环相比，AOHCP 在小规模实例上能快速得到几乎最优且几何更平面的解，但在大规模实例上求解时间明显增长；

**⚠️ 局限性**

限制在于二次规划非凸，求解效率受限于问题规模，且目前仅对单一同伦类求解，未能处理多类或全局最优保证。

---

## 427. Infant Care Video Dataset for Classification of Interventions Using Transformers

**arXiv ID:** 2608.23838 | [PDF](https://arxiv.org/pdf/2608.23838v1)

**作者:** Igor Bogdanov `[一作]` (Carleton University), James Green `[通讯]` (Carleton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究设计并公开了Infant Care Video Dataset（ICVD），利用视频变压器对新生儿护理干预进行分类，并提供基准实验结果；

**💡 创新点**

创新点在于首个隐私友好型、基于人偶模拟的NICU护理干预视频数据集，以及通过TimeSformer和MotionFormer展示时序建模显著优于单帧模型的证据；

**🔧 技术方法**

采用视觉变压器（TimeSformer、MotionFormer）预训练于Kinetics‑400，结合帧抽样、空间‑时间注意力机制进行微调；

**📊 数据集**

使用了4,144段手部操作视频，覆盖12个干预类别，包含多摄像头、多光照、多手套等系统变异的ICVD数据集；

**📈 对比分析**

通过与帧级（单帧空间注意）模型对比，时序模型Top‑1准确率达93.97%/93.17%，帧级仅23.17%，体现时序信息提升70.8%性能；

**⚠️ 局限性**

局限性包括人偶模拟与真实新生儿差异、仅涵盖12个干预、未处理多标签/重叠干预、对极端照明或低质量视频鲁棒性不足。

---

## 428. IC-ThermBench: An Open, Progressive Benchmark for Generalizable 2.5D/3D-IC Thermal Learning

**arXiv ID:** 2608.23977 | [PDF](https://arxiv.org/pdf/2608.23977v1)

**作者:** David Huang `[一作]`, Haiyang Xin `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `67630363-6be0-4f51-ab05-7198250671a5` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了开放且渐进式的IC‑ThermBench基准，用于评估2.5D/3D-IC热学习模型在不同泛化范围内的性能，并对八个代表性基线模型在统一数据、拆分、标签与指标下进行对比实验。

**💡 创新点**

创新点包括：① 统一公开的热学习基准，提供完整的数据生成、拆分、预处理与评估管道；② 通过渐进式设计将任务从固定设计 → 布局/材料/边界条件变异 → 跨包OOD，清晰揭示不同泛化难度；③ 在跨包OOD上引入有限标签的目标域适配实验，探究样本效率与泛化关系。

**🔧 技术方法**

使用的技术包括：HotSpot仿真生成热标注；输入格式统一为功率、坐标、导热率、边界参数；多种网络架构（U‑Net、FNO、U‑FNO、SAU‑FNO、DeepOHeat、Therm‑FM T/B/L）；Adam优化，GPU训练；统一评估指标RMSE、MAE、MaxAE、Top‑50 MAE、R²。

**📊 数据集**

数据集：① Alpha EV6 steady/transient 与工业包的固定设计任务；② 5万样本的2.5D chiplet扩展，划分为S2（布局）、S3（材料）、S4（边界条件）以及S5（跨包OOD）共5个泛化范围，样本总量超过十万。

**📈 对比分析**

比较方法：所有模型使用同一训练/验证/测试拆分、相同输入、相同指标。实验结果显示：S2–S4 随着物理变异逐步升高误差；S5 跨包OOD 误差急剧升高（RMSE从1.216 K升至15.99 K，MAE从0.938 K升至15.00 K）。有限标签适配后，MAE可降至2.60 K。不同模型在不同泛化范围内表现差异明显，模型规模与准确性不呈正相关。

**⚠️ 局限性**

局限性：① 仅涵盖2.5D稳态任务，未覆盖3D多层、瞬态或真实测量数据；② 生成数据来源于HotSpot模拟，缺乏硅级或签核级验证；③ 代表性分布稀疏，难以独立评估单一物理变量对性能的影响；④ 只评估八个基线，未覆盖全部工业应用场景。

---

## 429. ShardMeter: Sharded and Geo-Distributed Training Without the Guesswork

**arXiv ID:** 2608.23840 | [PDF](https://arxiv.org/pdf/2608.23840v1)

**作者:** Tim Beringer `[一作]` (Technical University of Darmstadt), Arya Mazaheri `[通讯]` (Technical University of Darmstadt)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了ShardMeter，一个轻量级的分析性能模型，用于预测Transformer模型在分布式（包括FSDP）和分散式（DiLoCo）训练中的训练时间、吞吐量及成本，并自动生成近似最优的部署方案。

**💡 创新点**

创新点包括：① 通过回归模型实现对未知Transformer模型训练性能的准确预测；② 统一建模FSDP与异步Decentralized训练的计算-通信依赖；③ 通过Pareto前沿实现成本-吞吐量权衡；④ 在多节点、异构硬件上保持平均误差低于15%。

**🔧 技术方法**

使用技术包括：线性回归建模通信/计算耗时、任务图与执行图捕捉并行与同步依赖、批量与模型规模交互分析、批量大小与岛配置的增量搜索优化、成本模型与多维度 Pareto 前沿生成。

**📊 数据集**

使用数据集：通过参数化的Transformer配置空间（如SmolLM、Llama、Gemma、Yi等）结合PyTorch Profiler提取的执行时间，网络通信基准（NCCL、InfiniBand 等）和不同硬件平台的实测数据。

**📈 对比分析**

通过与真实测量（单机、集群）对比，MAPE低于15%（计算与通信部分各自低于13%），在多种GPU/岛配置下预测与实际训练时间误差均小于20%，并在案例研究中实现了成本提升21%（保持99%吞吐）的显著优化。

**⚠️ 局限性**

limitations: 目前仅覆盖FSDP与现有DiLoCo分散式训练，未考虑GPU降频、节点失效、IO瓶颈、能耗/CO₂估算；不支持张量并行、100B+参数模型；对新硬件需重新收集基准并更新回归模型。

---

## 430. Streaming algorithms for computing coresets and $k$-median clustering in the Hamming space

**arXiv ID:** 2608.24347 | [PDF](https://arxiv.org/pdf/2608.24347v1)

**作者:** Taha El Ghazi `[一作]` (École normale supérieure de Paris), Tatiana Starikovskaya `[通讯]` (École normale supérieure de Paris)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出了首个在Hamming距离下的流式参数化近似（EPAS）算法，能够在FPT时间内实现(1+ε)-近似k‑median聚类，并给出了相应的ε‑coreset构造方法。

**💡 创新点**

创新点在于：①将Hamming距离映射到低维L2空间并保留近似距离；②构建可在流式环境下维护的ε‑coreset；③采用双层coreset放大技术实现高概率（w.h.p.）的近似；④首次在流式模型下实现对连续k‑median的ε‑coreset。

**🔧 技术方法**

技术包括Karoff嵌入、终端Johnson–Lindenstrauss变换、基于Huang‑Vishnoi算法的加权coreset采样、两级coreset放大与离散化采样。

**📊 数据集**

论文仅做理论分析，没有使用具体数据集。

**📈 对比分析**

与已有离线算法（如Ostrovsky & Rabani的(1+ε)-近似）相比，本文实现了流式场景下的空间复杂度(ℓk+k²)且保持FPT时间；在理论上取得了更优的时间-空间权衡。

**⚠️ 局限性**

局限性包括：仅适用于Hamming距离；依赖于多项式大小的字母表；实现中的常数和嵌入损失较大；实验验证缺失。

---

## 431. ChorusTIC: Training-Free Multivariate Time Series Classification via Chorus In-Context Learning

**arXiv ID:** 2608.24033 | [PDF](https://arxiv.org/pdf/2608.24033v1)

**作者:** Juntao Fang `[一作]` (Guangdong University of Technology), Zhifeng Hao `[通讯]` (Shantou University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5a41884c-404f-4688-a89c-aa238c10fe68` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出了一种支持条件无参数更新的时间序列分类基础模型ChorusTIC

**💡 创新点**

创新点在于将随机子通道槽拼接与双轴编码器、上下文校准及泄漏保护的ICL相结合，兼容异构通道配置

**🔧 技术方法**

使用RSSC、双轴Transformer、列分布建模、标签循环集成等技术实现跨通道交互与上下文推理

**📊 数据集**

在UCR-128（单变量）和UEA-30（多变量）两大公开数据集上进行预训练与评估

**📈 对比分析**

与通用ICL、冻结TSFM及传统分类器比较，ChorusTIC在全上下文与低样本情形下均取得最高平均准确率，提升约2.5%至4%

**⚠️ 局限性**

局限在于缺乏对缺失通道、异步采样及域漂移的处理，未来需扩展更广泛的基准与场景

---

## 432. The urban right to AI: Pluralistic co-design and governance of public space

**arXiv ID:** 2608.23999 | [PDF](https://arxiv.org/pdf/2608.23999v1)

**作者:** Rashid Mushkani `[一作]` `[通讯]`, Rashid Mushkani

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过构建公共空间AI治理框架，开发街景评价与生成模型，并在蒙特利尔进行多阶段参与式实验，提出了公民AI权利与多元化协同设计的理论与实践。

**💡 创新点**

创新点在于将AI治理视为双重基础设施，将公民AI权利与多元化协同对齐相结合，创建了Street Review测量管线、LIVS多元化对齐数据集，并将偏好学习与生成模型融合，首次将中立性视为治理信号。

**🔧 技术方法**

技术方面采用计算机视觉多尺度特征提取、注意力多层感知机进行街景预测，使用Stable Diffusion XL与Direct Preference Optimization进行文本到图像生成与偏好调优，结合结构化评分与分组讨论的实验设计。

**📊 数据集**

使用的数据集包括蒙特利尔街景图像（约45,000张）、Street Review的评估标签（12-28名参与者的评级与排名）、LIVS的37,710条成对偏好标注（13,462张图像），以及公开的街景数据。

**📈 对比分析**

与传统单一度量方法对比，Street Review的子组预测模型在R²上平均达到0.65-0.72，生成模型在DPO调优后在偏好匹配率提升约15%，但仍保留约30%的中立/不可判定样本，表明多元化信息被有效保留。

**⚠️ 局限性**

局限性在于以图像为唯一观测媒介，无法捕捉非视觉的公共空间体验；参与样本局限于蒙特利尔，缺乏跨城市验证；生成模型的偏好调优受限于二元反馈，导致中立性无法完全消除。

---

## 433. Object Counting Across Modalities: Taxonomies, Benchmarks, Applications, and Open Challenges

**arXiv ID:** 2608.23845 | [PDF](https://arxiv.org/pdf/2608.23845v1)

**作者:** Joana Konadu Owusu `[一作]` (University of Wyoming), Shivanand Venkanna Sheshappanavar `[通讯]` (University of Wyoming)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

对 2013-2026 年 600+ 论文进行系统综述，梳理了从密度回归到开放词汇、基础模型计数的四个方法时代，并提出了跨模态、跨任务的五轴分类法，揭示了六个结构性矛盾；

**💡 创新点**

首次将计数方法统一为模态、计数机制、提示方式、监督级别和泛化设定这五个维度，系统化评价不同范式与数据集的匹配，并指出评估基础设施落后于技术进展；

**🔧 技术方法**

主要采用文献检索、系统归档、跨维度对照分析以及对现有基准与评价指标的批判性评估，结合图表和对比表格呈现进展；

**📊 数据集**

重点考察了 FSC‑147、MixCount、DroneCrowd、ShanghaiTechRGBD、LVIS‑372、CAPTURe、PrACo 等代表性数据集，涵盖图像、视频、深度、3D、跨模态及应用专属多样场景；

**📈 对比分析**

通过对比 MAE、RMSE、GAME 等指标，发现尽管在主流基准上误差已逼近噪声水平，但在诊断基准和跨域测试中方法仍显失效，体现评估与实际通用性的鸿沟；

**⚠️ 局限性**

作为综述，缺乏对每种方法的细粒度实验重现，受限于现有基准多样性不足与评价指标不统一，难以给出全面客观的性能排名，亟需统一评估框架与更广泛的跨域验证。

---

## 434. AgentRoom: Concurrent Multi-Agent Coding in a CRDT-Backed Shared Workspace

**arXiv ID:** 2608.23740 | [PDF](https://arxiv.org/pdf/2608.23740v1)

**作者:** Seonglae Cho `[一作]` (Holistic AI), Donghyun Lee `[通讯]` (University of California, Berkeley)

**通讯引用:** 1642 | [OpenAlex ID](https://openalex.org/A5100436203)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了AgentRoom，一个基于CRDT的共享工作空间和MCP工具的并发编程框架，实现多智能体协同编码；

**💡 创新点**

创新点在于将文件级“声明-锁”与可广播的日志、状态管理集成到共享CRDT环境中，提供显式的并发协调通道，显著抑制单智能体“单文件中止”模式；

**🔧 技术方法**

核心技术包括CRDT合并（Y.js）、MCP工具（claim、status、broadcast等）、多智能体协同工作流、LLM接口（Claude、OpenAI、Gemini）以及多模型的并发推理；

**📊 数据集**

使用Express.js/TypeScript项目（T1‑T5）作为主实验集，外加Rust+axum的T4迁移和Python DevBench的多文件项目做跨语言检查；

**📈 对比分析**

在匹配计算预算下，通过LLM-judge综合分、正则和AST分数进行评估，AgentRoom相较于Solo、并行合并（parallel-merge）和ChatDev顺序管道提升约+0.21分，减少约30‑45%方差和13.7倍的中止率；

**⚠️ 局限性**

主要局限包括样本量有限、仅覆盖单一运行时（Express.js/TypeScript）、缺乏执行验证的oracle、跨语言普适性待进一步验证、部分模型（Gemini、Codex-mini）在并发MCP下表现不佳以及对大型多智能体规模的系统开销尚未系统评估。

---

## 435. Robust Code RL via Faulty-Code-Driven Test case Synthesis and Dense Reward Shaping

**arXiv ID:** 2608.24135 | [PDF](https://arxiv.org/pdf/2608.24135v1)

**作者:** Yiwen Zhang `[一作]` (Zhejiang University), Jun Zhou `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过基于近似正确的错误代码生成高质量测试用例，并结合验证器过滤和行为聚类，提出了一种鲁棒测试合成与密集奖励的RL框架，显著提升LLM代码生成的性能。

**💡 创新点**

创新点在于将“near‑correct”错误代码驱动的测试用例合成与密集奖励机制相结合，既扩展了诊断覆盖，又通过行为特征聚类和stepwise dense reward降低奖励噪声和误报。

**🔧 技术方法**

使用的技术包括LLM生成错误代码、验证器自动构造、行为特征向量聚类、K‑means多样性筛选、stepwise dense reward、GRPO强化学习等。

**📊 数据集**

采用CodeContests^+数据集（及其增强版RobustTests），并在LiveCodeBench与CodeForces基准上进行评估。

**📈 对比分析**

与Naive LLM Generation、HardTests、CodeContests+、CodeContests-O等基线相比，Qwen3‑32B在LiveCodeBench上绝对提升3%（Score从65.41提升到68.39），在CodeForces上Score、Rating、Percentile也都有提升，整体性能明显优于基线。

**⚠️ 局限性**

局限性包括：仅适用于有参考实现的编程任务；对更广泛软件开发场景的泛化能力待验证；需要手工标注或验证器才能确保测试用例的正确性。

---

## 436. IterCAD: Iterative Program Repair for CAD Code Generation from Orthographic Views

**arXiv ID:** 2608.24020 | [PDF](https://arxiv.org/pdf/2608.24020v1)

**作者:** Yuchuan Wu `[一作]` (Fudan University), Bin Li `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出IterCAD框架，将从二维正交视图生成可执行CAD代码转变为多轮程序修复过程；

**💡 创新点**

创新点在于构建有结构的修复或停止监督集IterCAD‑RS，并通过三阶段训练（初始生成→修复/停止学习→多轮RL优化）实现模型自适应检查与修正；

**🔧 技术方法**

使用vision‑language模型（如Qwen3.5‑9B、Qwen3‑VL‑8B），结合几何验证、结构化修复决策和GRPO强化学习优化全流程；

**📊 数据集**

基于工业级CADExpert正交视图‑CAD代码基准数据集进行训练与评估；

**📈 对比分析**

与LLaVA、InternVL、CME‑CAD等多种基线对比，IterCAD在IoU、Chamfer距离和可执行率上均大幅领先，Qwen3.5版本取得91.61% IoU、99.33%可执行率；

**⚠️ 局限性**

局限在于受限于训练数据规模和修复步骤上限（最多4轮），对极端几何或非标准视图的鲁棒性仍需提升。

---

## 437. Curved Inference II: Sleeper Agent Geometry - Extending Interpretability Beyond Probes

**arXiv ID:** 2608.24037 | [PDF](https://arxiv.org/pdf/2608.24037v1)

**作者:** Rob Manson `[一作]` `[通讯]`, Rob Manson

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究提出了一种基于内部残差流几何结构的自然对话欺骗检测方法，利用多轮上下文窗口模拟真实的推理过程，并通过无标签、无触发器的方式评估模型的隐藏意图。

**💡 创新点**

创新点在于将 Curved Inference 框架扩展到自然欺骗场景，提出新的语义表面积（A′）指标来量化语义推理的几何复杂度，并引入双分辨率、未归一化轨迹采样，从而捕捉到传统线性探测难以捕获的几何签名。

**🔧 技术方法**

主要技术包括：①双分辨率残差轨迹采样（关注子层前后状态）；②未归一化残差流与拉普拉斯（pullback）度量；③语义表面积 A′ 的计算；④多模型 LLM 共识分类（Gemini、Claude、GPT‑4o）与统计检验（Kruskal‑Wallis、Mann‑Whitney、效应量计算）。

**📊 数据集**

使用两款开源解码器 LLM（Gemma3‑1b 与 LLaMA3.2‑3b）以及设计的五种对话策略（诚实、策略、说服、欺骗、恶意），在 100 条生成样本/策略、共 500 条样本/模型的基础上进行实验。

**📈 对比分析**

与传统线性探测相比，A′ 在不同透明度和回应类型之间表现出显著差异（p < 0.001，效应量大），并且在全共识与全一致共识两种分类阈值下均保持一致的方向性；一致共识下信号更强，效应量提升，显示出测量精度提升可显著增强检测性能。

**⚠️ 局限性**

局限性包括：①依赖 LLM 共识标签作为间接标注，缺乏人类行为真实标签；②仅评估两款小型模型和有限的提示策略，结果的可推广性待验证；③全一致过滤导致样本量显著下降，可能产生选择偏差；④未深入研究更大规模或不同体系结构模型的几何尺度差异。

---

## 438. ROI-Gated SAHI: Content-Adaptive Slicing-Based Inference for Efficient Object Detection

**arXiv ID:** 2608.23923 | [PDF](https://arxiv.org/pdf/2608.23923v1)

**作者:** Rashid Riyadh `[一作]` (City University), Muzammil Behzad `[通讯]` (King Fahd University of Petroleum and Minerals)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于ROI门控的SAHI推理框架（ROI‑Gated SAHI），通过轻量级提议器先定位前景区域，然后仅在这些区域进行切片推理，减少背景计算。

**💡 创新点**

创新点在于将前景ROI估计与SAHI切片推理耦合，并设计自适应回退策略：根据ROI覆盖率动态切换全图SAHI与ROI‑Gated，实现在不同场景密度下的高效推理。

**🔧 技术方法**

采用YOLOv8n作为轻量级提议器、YOLOv8s作为高分辨率细化器；通过阈值τ的路由决策、ROI合并扩展、NMS融合等技术实现推理流程。

**📊 数据集**

使用COCO128全分割（128张高分辨率图像）以及三张分别为稀疏、中密、密集场景的高分辨率图像进行评估。

**📈 对比分析**

与全图SAHI进行对比，静态ROI‑Gated平均慢0.88×且mAP降至0.6602；自适应路由平均加速至1.02×，在稀疏场景实现最高6.90×加速，整体平均3.41×；mAP保持在0.7569，显示在稀疏场景下兼顾速度与精度。

**⚠️ 局限性**

主要限制包括：依赖轻量级提议器的召回率，漏检会导致细化阶段无法恢复；阈值τ和ROI扩展比等参数固定，缺乏自适应学习；实验仅验证YOLO模型，缺少跨检测架构的泛化性。

---

## 439. Task-disentangled Low-Rank Adaptation for Versatile Audio-visual Multi-modal Learning Tasks within a Unified Framework

**arXiv ID:** 2608.24209 | [PDF](https://arxiv.org/pdf/2608.24209v1)

**作者:** Hanyu Xuan `[一作]` (Anhui University), Hehe Fan `[通讯]` (Zhejiang University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种统一的音视频多模态学习框架，并通过任务解耦低秩适配（Task‑Disentangled LoRA）实现多任务协同；框架兼容时间定位、空间定位、像素级分割及时空推理六类核心任务。

**💡 创新点**

创新点在于：①将共享低秩子空间、任务特异调制矩阵和跨任务专家融合到LoRA中，既保留全局知识又解耦任务差异；②通过任务自适应路由实现专家动态加权；③在LLM上嵌入“mask token”实现分割输出。

**🔧 技术方法**

使用的大型语言模型 LLaMA‑2‑7B‑Chat；视觉编码器 CLIP‑ViT‑L/14，音频编码器 BEATs；Q‑Former 以及 MLP 对齐模块；任务解耦 LoRA（含共享矩阵 A、调制矩阵 Λ、MoE 专家 B）；Segment Anything 的 mask decoder；训练损失包括文本交叉熵与分割 Dice/BCE。

**📊 数据集**

六大任务对应数据集：AVEL‑AVE、AVVP‑LLP、AVQA‑MUSIC‑AVQA、AVS‑AVSBench(S4/Ms3)、RAVS‑Ref‑AVS、ARIG‑AVSBench(ARIG)。

**📈 对比分析**

与 TimeChat、GroundingGPT、Crab、MEERKAT、AnyRef 等通用模型对比，结果显示在所有六个任务上均优于或匹配最强竞争者；RAVS 的 mIoU 提升 +6.59、F1 +6.46，AVEL+3.73%，AVVP 事件级 F1 +4.22，AVQA 准确率 76.11% 等显著提升。

**⚠️ 局限性**

局限性包括：①对四个专家或三专家的实验表明模型对专家配置敏感，移除任何专家会导致性能下降；②整体依赖大规模预训练 LLM 与高算力，资源受限环境下可行性有限；③仍未在极端噪声或稀缺数据场景下验证鲁棒性。

---

## 440. FPGAgent: An LLM-Assisted Framework for Autonomous HLS Code Generation and Verification in FPGA Environments

**arXiv ID:** 2608.23630 | [PDF](https://arxiv.org/pdf/2608.23630v1)

**作者:** Tianyu Wang `[一作]` (Shanghai Jiao Tong University), Xijun Li `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 766 | [OpenAlex ID](https://openalex.org/A5081396846)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了FPGAgent，一个多智能体的端到端框架，能从自然语言任务描述自动生成FPGA可执行的HLS代码并在真实FPGA平台上验证；

**💡 创新点**

首次实现了基于进化搜索的核代码生成与多层错误反馈的功能反射，并将整个生成、验证、部署流程闭环到真实硬件上；

**🔧 技术方法**

采用大型语言模型与检索增强、进化算法、HLS工具链（Vitis HLS）、多级错误分析、函数层与硬件层的反馈循环；

**📊 数据集**

使用HLS‑Eval基准集（原始94例，裁剪后78例）进行实验；

**📈 对比分析**

与零射击生成对比，评估可综合率、可执行率和功能正确率；在Gemini 3.1 Pro上最高达92.3%的功能正确率，整体平均提升约30%；

**⚠️ 局限性**

对低性能/复杂核的功能修复仍有限；实验受限于单一基准与特定FPGA/工具链，迁移到其他平台需适配；构建xclbin耗时高，影响整体效率。

---

## 441. One Burst of t-Deletion and One Burst of t-Substitution Error-Correcting Codes

**arXiv ID:** 2608.24272 | [PDF](https://arxiv.org/pdf/2608.24272v1)

**作者:** Yajuan Liu `[一作]` (Bilkent University), Tolga M. Duman `[通讯]` (Bilkent University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种显式构造的二进制错误纠正码（ECC），能够纠正一个t删除和一个t替换的突发错误。

**💡 创新点**

创新点在于通过将原始序列表示为矩阵形式，并将两个突发错误转换为一些擦除或替换，从而实现了低冗余的错误纠正。

**🔧 技术方法**

使用了矩阵表示法和Reed-Solomon（RS）码的余类来构造错误纠正码。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了在DNA数据存储、文件同步和磁记录等通信系统中的应用。

**📈 对比分析**

与现有方法相比，本文提出的ECC在纠正一个t删除和一个t替换的突发错误时，冗余为10log n + 12t log log n + O(1)，性能优于其他方法。

**⚠️ 局限性**

限制在于该方法假设两个突发错误是非重叠的，且在某些情况下可能无法处理更复杂的错误模式。

---

## 442. Safety-aware Model Predictive Path Integral Control with Signal Temporal Logic

**arXiv ID:** 2608.23972 | [PDF](https://arxiv.org/pdf/2608.23972v1)

**作者:** Yiqi Zhao `[一作]` (University of Southern California), Georgios Fainekos `[通讯]` (Toyota Motor North America)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种基于MPPI的安全感知STL规划框架，利用时间变控制屏障函数实现对Signal Temporal Logic约束的满足。

**💡 创新点**

创新点在于将STL约束转化为离散时间的可变控制屏障函数，并在MPPI采样权重与投影过程中直接利用该CBF，从而在保持采样并行性和低计算量的前提下实现硬约束满足。

**🔧 技术方法**

采用MPPI（模型预测路径积分）、时间变控制屏障函数、STL语义编码、加权采样投影等技术。

**📊 数据集**

使用四个人工火星车规划案例和一套基于NVIDIA Isaac Lab的四旋翼仿真环境。

**📈 对比分析**

与若干MPPI基线进行对比，实验结果表明在安全率和规划效率上均优于基线，且计算开销低。

**⚠️ 局限性**

局限性包括：仅在仿真/人工案例验证，缺乏真实硬件实验；STL‑CBF约束近似实现，未给出严格的实时满足证明；对复杂高维STL规范的可扩展性仍待验证。

---

## 443. SQLite is Enough. Lexical, Semantic, and Hybrid Search with scrydb

**arXiv ID:** 2608.24060 | [PDF](https://arxiv.org/pdf/2608.24060v1)

**作者:** Timo Breuer `[一作]` `[通讯]` (University of Applied Sciences Cologne), Timo Breuer (University of Applied Sciences Cologne)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了名为 ScryDB 的 Python 库，能够在单个 SQLite 数据库文件中完成词法检索、语义检索以及两者融合检索，并支持后续的再排序和结果融合。

**💡 创新点**

创新点在于：①将文档、全文索引与向量索引全部嵌入同一 SQLite 文件，构成单文件可复现的检索资源；②利用 SQLite 的 FTS5 与 sqlite-vec 扩展，支持不同精度（二进制、int8、float32）的向量检索与再排序；③通过二进制化嵌入和 Hamming 距离实现高效的全量扫描，且可与更精确的重排序结合，展示了低精度检索在效果与延迟上的优势。

**🔧 技术方法**

使用的技术包括 SQLite FTS5（全文索引与 BM25 排名）、sqlite-vec（向量存储与检索）、向量二进制化与 int8 量化、Hamming 距离与余弦相似度、Reciprocal Rank Fusion（RRF）以及 Python 生态中的 SentenceTransformer 进行嵌入生成。

**📊 数据集**

实验采用了 BEIR 公开检索基准中的八个数据集：ArguAna、FiQA、NFCorpus、Quora、SciDocs、SciFact、Touché 与 TREC‑COVID，使用 Qwen3‑Embedding‑8B 作为嵌入模型。

**📈 对比分析**

与 MTEB 基准中使用的全精度（float32）余弦检索结果对比，ScryDB 在多种配置下取得了相近甚至优于基准的效果，特别是 Hamming+cosint8、BM25+cosfloat 等组合；在 nDCG@10 上平均只落后 0.006，且在四个数据集上超过基准。延迟方面，二进制 Hamming 检索在 523K 文档规模下仍在 81 ms 内，int8 重排序平均 165 ms，显著低于全精度检索（~2 s）。

**⚠️ 局限性**

局限性包括：①检索速度为线性扫描，无法满足大规模（百万级以上）语义检索或高并发服务需求；②无法实现实时增删改，适合离线或周期性更新的场景；③当词法与语义检索效果差距较大时，RRF 融合并不总是最佳选择；④对极大规模语料库仍需更高效的 ANN 索引结构。

---

## 444. Mixture of Channel Experts: Static Sparse Supports with Input-Adaptive Mixing for Pointwise Projections

**arXiv ID:** 2608.23794 | [PDF](https://arxiv.org/pdf/2608.23794v1)

**作者:** Elian Iluk `[一作]` (Ariel University), Gil Ben-Artzi `[通讯]` (Ariel University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Mixture of Channel Experts (MoCE)，将稠密点乘投影替换为稀疏通道支持并加入温度门。

**💡 创新点**

创新点在于把专家维度从操作复制转为通道选择，学习静态支持并仅通过输入可变温度进行轻量级混合，保持可调度性。

**🔧 技术方法**

使用稀疏点乘投影、软max混合、全局平均池化、温度门、覆盖正则化，并在ResNet、EfficientViT网络上训练，配合ImageNet、CIFAR-100及迁移学习进行验证。

**📊 数据集**

主要使用数据集为ImageNet-1K、CIFAR-100以及ImageNet→CIFAR迁移，并在EfficientViT上测试CIFAR-100。

**📈 对比分析**

与密集模型、SE、CondConv、Pick-or-Mix等方法对比，MoCE在ResNet-50/101/152上保持或提升Top‑1准确率，同时MAC下降约17–21%，部署参数减少约17–21%，端到端推理速度提升约4.6%。

**⚠️ 局限性**

局限性包括稀疏投影受内存流量限制，动态通道选择成本高、准确性提升有限，温度门对性能贡献显著；对更大k或不同硬件的适用性尚未完全验证。

---

## 445. Robust Slip Detection and Material Classification via Spatiotemporal Transformers on a Uniformly-Illuminated Visuo-Tactile Sensor

**arXiv ID:** 2608.24162 | [PDF](https://arxiv.org/pdf/2608.24162v1)

**作者:** Ziyang Ma `[一作]` (Beijing University of Posts and Telecommunications), Bin Fang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6514db3d-8de6-452c-91b7-acdb31787cc4` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一个无标记、统一 RGB 光照的视觉触觉传感器，并构建了包含 15 个日常物体的多任务 RGB‑D 触觉数据集，用于滑移检测和静态物体分类。

**💡 创新点**

通过自定义均匀 RGB 光照实现亚毫米级深度重建，首次将滑移细分为 8 个方向，并提出双头 TimeSformer 与 ResNet‑50 的统一感知框架。

**🔧 技术方法**

使用光度立体校准、深度积分、双头 TimeSformer 时空注意网络、ResNet‑50 视觉分类、梯度注意可视化等技术。

**📊 数据集**

使用 15 个对象共 19,336 张静态图像与 7,925 条滑移序列（共 63,400 帧）并同步生成深度图的 RGB‑D 数据集。

**📈 对比分析**

在未见物体上，滑移状态识别 95.5%、方向识别 91.5%；静态物体识别 RGB 98.8%、深度 96.06%；相较现有基准，深度 RMSE 0.072 mm，滑移检测准确率显著提升。

**⚠️ 局限性**

仍局限于单指抓取，未验证多指或复杂动态场景；深度推断依赖光度立体校准，光照不均时性能下降。

---

## 446. EviDx: Evidence-Aware Active Diagnosis with Scaffolded LLM Agents

**arXiv ID:** 2608.24570 | [PDF](https://arxiv.org/pdf/2608.24570v1)

**作者:** Lihang Zeng `[一作]` (Shanghai Jiao Tong University), Xiaofan Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了 EviDx 框架，将静态临床病例转换为交互式诊断环境，并通过临床角色化的工具与诊断 scaffold 以及 observer‑guided harness 进行证据驱动的主动诊断。

**💡 创新点**

创新点包括：①将患者特定环境、诊断 scaffold 与 runtime harness 三层融合，实现主动证据获取与动态终止；②利用不确定性熵与运行时证据覆盖度构建 observer‑guided harness，实现对诊断终止的动态控制；③提出三层评估金字塔，分别评估执行鲁棒性、推理动态和诊断结果。

**🔧 技术方法**

技术手段主要有：LLM 代理与工具调用（MCP）、EHR 交互与外部医学知识检索（MedRAG）、entropy‑based 诊断不确定性评估、runtime 证据覆盖度度量、工具链编排与运行时 harness 控制、以及多层次评估指标。

**📊 数据集**

使用的数据集包括公开临床基准 JAMA、MedXpertQA（及其诊断子集 MedXpertQA‑Diag）、DiagnosisArena 以及新构建的 LLM‑辅助、医生审阅的 Med‑Evidence‑2.6k。

**📈 对比分析**

与单一代理 baseline 在同一 100 个病例子集上进行对比，采用 MC 与 open‑ended 诊断准确率、证据回忆（Acq/Cog）、observer 干预次数、uncertainty 收敛度等指标。EviDx 在 MC 诊断上显著提升（如 Claude Sonnet 4.6 在 JAMA 上从 53% 提升至 66%，在 DiagnosisArena 上提升 8%），并提升过程稳定性；open‑ended 诊断仍受限于模型基础知识。

**⚠️ 局限性**

局限性包括：1）仍高度依赖 LLM 的医学知识，small models 在 open‑ended 诊断表现差；2）observer‑guided harness 的终止阈值基于实验室启发式，未验证对真实临床终止标准的适用性；3）未处理多模态、纵向 EHR 数据或真实患者–医生交互；4）LLM 可能产生不安全或错误建议，需严格人机交互监管。

---

## 447. COCI: Conference Organisers and Content Identifier

**arXiv ID:** 2608.24559 | [PDF](https://arxiv.org/pdf/2608.24559v1)

**作者:** Angelo Salatino `[一作]` (Open University), Enrico Motta `[通讯]` (Open University)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了名为COCI的AI框架，自动从 Call for Papers（CfP）中提取会议元数据、组织者信息并结构化；

**💡 创新点**

创新点在于将 LLM 与语义映射技术结合，针对灰色文献实现高精度实体消歧与主题映射，填补学术事件在知识图谱中的空白；

**🔧 技术方法**

使用了 GPT‑4o 进行文本解析、SentenceTransformer（all‑MiniLM‑L6‑v2）生成向量、Levenshtein 相似度、OpenAlex/DBLP/AIDA/ConfIDent API 以及 Streamlit 前端；

**📊 数据集**

评测使用了 40 份来自计算机科学、工程学、计量学、材料科学等领域的 CfP 文件；

**📈 对比分析**

与人工标注对照，COCI 成功处理所有 CfP，匹配准确率高于 80%，与手工处理相比时间提升约 90%，但在部分案例中因外部数据库错误导致匹配偏差；

**⚠️ 局限性**

限制包括对外部知识库质量的依赖、LLM 幻觉问题、仅覆盖有限数量 CfP、缺乏对更大规模多领域语料的验证。

---

## 448. X-MULTI: VLM-based Imaging Factor Disentanglement for Factor-Aware Image Synthesis

**arXiv ID:** 2608.24563 | [PDF](https://arxiv.org/pdf/2608.24563v1)

**作者:** Sonali Godavarthy `[一作]` (University of Siegen), Danda Pani Paudel `[通讯]` (INSAIT)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 X-MULTI 方法，通过使用零样本视觉语言模型（VLM）为合成图像提供因子级监督，从而实现对多种图像采集因子（镜头、传感器、视角、域）的可控生成，特别是对训练中未出现的因子组合实现了更好的分离。

**💡 创新点**

创新点在于：①将零样本 VLM 作为外部因子分类器为未见因子组合提供监督；②提出改进的 Factor Alignment Accuracy（I-FAA）评估指标，采用类别均衡采样和因子特定增强来消除交叉因子泄漏。

**🔧 技术方法**

使用技术包括：Stable Diffusion XL 的扩散模型、Textual Inversion 训练可学习因子嵌入、Qwen2-VL-7B-Instruct 零样本 VLM 进行因子预测、跨因子均衡采样及增强的线性分类头训练。

**📊 数据集**

使用数据集为 DF‑RICO 基准，包括 15 个自动驾驶和监控数据集，涵盖镜头（normal/fisheye）、域（real/simulation/video‑game）、视角（front/back/side/drone/pole）以及传感器（rgb/thermal/rgb‑thermal/gated/event）。

**📈 对比分析**

与 SDXL Zeroshot、DreamBooth、MULTI 等基线对比，X‑MULTI 在 I‑FAA、CLIP‑Score、Diversity Score 等指标上均取得更优表现，尤其在未见因子组合的因子对齐上提升约 11%。

**⚠️ 局限性**

局限性包括：VLM 对部分因子（如 rgb‑thermal、非 front 视角）的识别不够准确，导致监督信号不稳定；I‑FAA 虽降低交叉因子相关，但仍存在残留相关；模型对极少见因子组合的泛化能力有限，且需要精心设计因子提示和增强策略。

---

## 449. StrokeGuard: A Multi-Agent Guided System for Prehospital Stroke Assessment

**arXiv ID:** 2608.24555 | [PDF](https://arxiv.org/pdf/2608.24555v1)

**作者:** Wentao Yang `[一作]` (Shanghai Jiao Tong University), Yao Guo `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了一套基于多代理的StrokeGuard系统，用于在非专业救护者现场对中风进行FAST式评估并实现可审计的交接报告。

**💡 创新点**

创新点在于：①双通道代理机制，将正式评估与程序支持分离；②证据感知的编排与阶段本地恢复，保证评估步骤的完整性与数据来源可追溯；③基于已有预训练视频评估模块的多模态输入策略与文本回退补偿。

**🔧 技术方法**

主要技术包括自然语言对话代理、有限状态机编排、事件溯源与日志、预训练的面部、手臂、语音评估网络、视频优先与文本备选机制。

**📊 数据集**

实验使用模拟现场（70岁男性演员扮演患者）进行12名受试者的评估任务，没有公开真实医学数据集，所有评估结果均来自实验记录。

**📈 对比分析**

与传统纸质FAST表单对比，StrokeGuard在MATES-9总分上提高了10.83分（相对23.8%），评估时长从116s降至81s（降低30.2%），显示出显著的易用性和效率提升。

**⚠️ 局限性**

局限性包括：①仅在小规模模拟环境中验证，缺乏真实急救现场的数据与多样性；②核心评估模块仍未经过大规模临床验证；③系统对网络、设备兼容性的鲁棒性待进一步测试。

---

## 450. From Numerical Simulators of PDEs to Neural Emulators and Back

**arXiv ID:** 2608.24547 | [PDF](https://arxiv.org/pdf/2608.24547v1)

**作者:** Felix Koehler `[一作]` `[通讯]`, Felix Koehler

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了利用神经网络替代传统数值 PDE 求解器的可行性，提出了 APEBench 基准评测框架、PRDP 逐步精细化可微分物理训练策略，并系统性分析了训练数据误差、网络先验和训练目标之间的关系，揭示了某些神经仿真器在推断时甚至可优于其训练数据来源。

**💡 创新点**

创新点包括：1）使用 Fourier 频谱误差分析统一描述求解器误差、网络结构偏置和训练目标；2）证明在迭代未完全收敛的求解器下仍能有效训练，从而显著降低训练成本；3）构建可扩展、可重复的 APEBench 评测套件，涵盖 46 种周期、均匀网格 PDE 配置；4）通过实验证明神经仿真器在某些问题上能实现“超越”训练求解器的精度。

**🔧 技术方法**

技术手段包括：JAX 框架下实现的可微分伪谱求解器（Exponax）、可微分有限差分求解器（Picardax）、Chaotax；多种神经网络架构（ConvNet、SWIN、Dilated ResNet、UNet、FNO、Transformer 等）；Fourier 频谱误差分析；梯度裁剪、分层求解器、自动微分等优化与正则化技术。

**📊 数据集**

使用 46 种周期、均匀网格 PDE 配置（如 Burgers、Navier‑Stokes、波动方程等）生成的合成数据集，涵盖多种时间步长、分辨率与物理参数，且所有实验均基于 JAX 生成的可微分求解器产出的训练轨迹。

**📈 对比分析**

在 APEBench 基准下，FNO、UNet、Transformer 等模型在大多数 PDE 上取得最优性能；PRDP 在仅使用 1–10 次迭代的可微分求解器下即可达到与完全收敛求解器相同的误差，显著降低训练成本；此外，实验显示某些神经仿真器在长期推断时误差低于训练时使用的求解器，验证了“仿真器优越性”现象。

**⚠️ 局限性**

主要局限包括：仅针对周期性、均匀格点 PDE，非周期或非结构化网格的推广有限；对高维、复杂边界条件的泛化尚待验证；对训练数据分布与初始化的敏感性；以及在极端非线性或强激荡场景下模型稳定性与误差上限的进一步研究。

---

## 451. Neurosymbolic Alignment for Physiologically-Safe Clinical Language Models

**arXiv ID:** 2608.24534 | [PDF](https://arxiv.org/pdf/2608.24534v1)

**作者:** Abdulhady Abas Abdullah `[一作]`, Milena Zivkovic `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个神经符号对齐框架，用于从电子病历（EHR）中对临床查询生成安全、可解释的答案。

**💡 创新点**

将大规模生理知识图谱与语言模型结合，通过异构图神经网络（HGNN）进行可行性验证，并通过迭代的ORPO（偏好优化）对齐模型输出与知识图一致性。

**🔧 技术方法**

使用大语言模型（LLM）进行回答生成和候选采样，HGNN（4层R-GAT）进行物理可行性评分，以及ORPO对齐优化；同时基于KG的一致性约束。

**📊 数据集**

使用约847K节点、3.2M边的生理知识图谱（PhysioKG）以及临床安全摘要数据集进行评估。

**📈 对比分析**

相较于传统的纯LLM或仅基于KG的检索方法，本文在临床安全摘要（CSS）提升34%，幻觉率降低64%，对抗性指令（DID）达91.6%，最终安全兼容性得分（PC）为0.89。

**⚠️ 局限性**

局限包括对知识图谱完整度和更新频率的依赖、计算开销较大、在不同临床语境下的迁移性仍待验证。

---

## 452. Achieving Torn-Paper Channel Capacity with Successive Revelation

**arXiv ID:** 2608.24506 | [PDF](https://arxiv.org/pdf/2608.24506v1)

**作者:** Rui Xu `[一作]` (KTH Royal Institute of Technology), Le Wang `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

提出了一种逐步揭示（Successive Revelation）编码方法，用于二进制随机撕碎纸信道，在平均误差条件下实现容量极限。

**💡 创新点**

创新点在于利用已恢复的载荷轨道动态扩展位置信息，随解码进度逐步降低碎片长度阈值，从而实现无固定同步符号密度的自适应解码。

**🔧 技术方法**

采用轨道分层、随机公共位移、局部对齐与随机生成矩阵，并结合覆盖分析、误定位概率界定与矩阵秩估计等技术进行可靠性与速率证明。

**📊 数据集**

本文未使用实际数据集，而是通过理论模型和数值仿真（α∈[0,2]）展示不同M值下的速率曲线。

**📈 对比分析**

与先前的间隔同步符号和局部对齐方案相比，该方法在同一α下获得更高速率；当M→∞时误差上界趋于零，理论上可达到完整信道容量。

**⚠️ 局限性**

主要局限在于实现需预先共享随机位移与生成矩阵，且对实际硬件的实现复杂度与延迟尚未评估，且该方案在极低α值下仍需较大M才能逼近容量。

---

## 453. A Capability Broker for Workflow-Network QoS Coordination in B5G/6G Industrial Services

**arXiv ID:** 2608.24496 | [PDF](https://arxiv.org/pdf/2608.24496v1)

**作者:** Qize Guo `[一作]` (Ruhr University Bochum), Hao Yu `[通讯]` (Beihang University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的深度学习模型，用于图像分类任务。

**💡 创新点**

创新点在于引入了一种新的激活函数，能够提高模型的收敛速度和分类精度。

**🔧 技术方法**

使用了卷积神经网络（CNN）和改进的激活函数。

**📊 数据集**

使用了CIFAR-10数据集进行实验。

**📈 对比分析**

与传统的激活函数（如ReLU）进行比较，结果显示新模型在分类精度上提高了5%，且训练时间缩短了15%。

**⚠️ 局限性**

模型在处理大规模数据集时可能会遇到内存限制的问题。

---

## 454. Concept of Time-Reversal Characteristic Modes in Non-Free-Space Environments

**arXiv ID:** 2608.24489 | [PDF](https://arxiv.org/pdf/2608.24489v1)

**作者:** Chenbo Shi `[一作]` (University of Electronic Science and Technology of China), Jin Pan `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出一种基于相对背景差分散射场的时间反演特征模式物理定义，并在有限结构背景与PEC半空间两种非自由空间环境中验证其有效性。

**💡 创新点**

创新点在于：①把特征模式的定义迁移到散射域的场级别，使其与具体的矩阵表示无关；②通过时间反演闭合条件，将背景引用的传播与返回过程统一起来；③展示该定义在有限子结构与无限半空间两种截然不同的传统表示下仍能得到相同的特征模式，从而为非自由空间环境的特征模式分析提供了统一、物理直观的框架。

**🔧 技术方法**

使用了时间反演对称性、差分散射dyadic、基于背景的Green函数阻抗算子、Takagi分解、球面波散射矩阵等数值技术。

**📊 数据集**

未使用公开数据集；实验基于数值仿真得到的PEC条形结构模型（有限背景和半空间两种场景）。

**📈 对比分析**

通过最大相关系数对辐射场模式进行分支追踪，对比经典子结构CMT、差分散射dyadic时间反演模式以及共原点球面波实现，结果在两种环境下六条特征分支完全一致，验证了定义的正确性。

**⚠️ 局限性**

局限性包括：①本文仅在共振系统、无耗散、可逆背景下验证，缺乏对耗散或非共振情形的深入讨论；②对更复杂的多层、周期性或非共形环境的推广需要进一步研究；③计算需要准确的Green函数，若背景复杂可能导致数值实现困难。

---

## 455. WarpSAC: Towards the Pinnacle of Scalable Off-policy RL by Rethinking Exploration and Exploitation

**arXiv ID:** 2608.24479 | [PDF](https://arxiv.org/pdf/2608.24479v1)

**作者:** Zihao Wu `[一作]` (Tianjin University), Jianye Hao `[通讯]` (Tianjin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于数据覆盖度的可扩展离线策略学习框架（WarpSAC），并在CPU规模与GPU并行两种数据采集环境下对常见稳定器（参数归一化、双Q目标、样本权重衰减）进行系统性对比与组合，给出对应的数据规模推荐配置。

**💡 创新点**

创新点在于：①把传统的稳定器视为与数据覆盖度相关的“可切换”模块；②提出“SWD为核心，数据规模决定其余两个模块（归一化与双Q/单Q）”的规则；③通过大量基准实验验证该规则在CPU与GPU两种规模下均优于统一使用 FlashSAC 的做法。

**🔧 技术方法**

使用的技术包括：FlashSAC 基线（Soft Actor‑Critic + 参数投影归一化 + 双Q目标）、Sample Weight Decay（SWD）年龄加权重采样、单/双 Q 目标切换、GPU 并行模拟与高吞吐量采样、模拟到真实的闭环部署、Bfloat16 自动混合精度等。

**📊 数据集**

实验数据集涵盖八大基准族：DeepMind Control Suite (hard tasks)、MuJoCo、HumanoidBench、MyoSuite、IsaacLab、MJLab、ManiSkill、MuJoCo Playground；此外还使用 Unitree G1 29-DoF 真实机器人进行 sim‑to‑real 评估。

**📈 对比分析**

与 FlashSAC（统一设置）和 PPO（强化版）对比。WarpSAC 在 CPU 规模下平均提升 4.5% 的归一化得分‑步 AUC，GPU 并行下提升 23.1%；在 Unitree G1 上从 19.8% 提升至 96.4% 的成功率；在仿真‑真实闭环训练中，WarpSAC 训练时间比 FlashSAC 低 36.4%，并在 35 分钟内完成部署。

**⚠️ 局限性**

局限性：①当前的 regime‑aware 方案是离线基于预设规模切换，未实现在线动态调整；②实验仅基于 FlashSAC 架构，未验证其它常见稳定器（熵权重、目标网络延迟、梯度裁剪等）的规模依赖性；③在极大规模或跨规模混合训练场景下的鲁棒性仍待研究。

---

## 456. Implicit Q-learning-bootstrapped ant colony optimization for maritime moving-target observation scheduling with agile satellites

**arXiv ID:** 2608.24471 | [PDF](https://arxiv.org/pdf/2608.24471v1)

**作者:** He Wang `[一作]` (Harbin Engineering University), Liang Li `[通讯]` (Harbin Engineering University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于离线隐式Q学习驱动的蚁群优化（IQACO）框架，用来解决多卫星海上移动目标观测调度问题。

**💡 创新点**

创新点在于将离线价值学习用于自动调节蚁群搜索参数（α、β、ρ），从而在保留可行性约束优势的同时提升搜索效率与解质量。

**🔧 技术方法**

使用技术包括：离线隐式Q学习（IQL）用于策略与价值网络训练，蚁群优化（ACO）用于可行排程构造，以及C++/Python结合的混合实现。

**📊 数据集**

数据集为14个基于AIS轨迹模拟、含100–240个海上目标和3–6颗机动卫星的测试场景，目标运动、云覆盖和资源约束均从模拟中生成。

**📈 对比分析**

与遗传算法、粒子群、鲸鱼优化和传统ACO比较，IQACO在所有场景均获得最高平均观测收益，提升幅度从3.4%至9.4%，并实现更快收敛和更稳健的性能。

**⚠️ 局限性**

局限性包括：依赖模拟数据、简化云覆盖模型、仅训练于固定分布的离线策略，缺乏对真实AIS轨迹、时变云场及在线自适应的考虑。

---

## 457. FraudBench: Protocol-Sensitive Benchmarking of Adversarial Robustness for Financial Risk Assessment

**arXiv ID:** 2608.24551 | [PDF](https://arxiv.org/pdf/2608.24551v1)

**作者:** Xitong Zeng `[一作]` (University of Sydney), Quan Z. Sheng `[通讯]` (Macquarie University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了FraudBench，一套针对金融欺诈与信用风险检测的协议敏感对抗鲁棒性基准；

**💡 创新点**

核心创新在于将评估协议视为实验变量，区分无约束攻击、后处理过滤和部署约束集成攻击，并揭示协议对鲁棒性结论的显著影响；

**🔧 技术方法**

使用了CAPGD、Square Attack、HopSkipJump等对抗攻击，并实现投影与可变性掩蔽等约束处理；

**📊 数据集**

覆盖四个公开金融数据集：CCFD、IEEE-CIS、LCLD、Sparkov；

**📈 对比分析**

在相同实验设置下比较不同协议、模型族（MLP、XGBoost、异构集成）与防御（无防御、对抗训练、输入校验、集成）时发现，协议选择会显著改变鲁棒PR‑AUC、可行攻击计数和模型排名，说明协议敏感性；

**⚠️ 局限性**

局限性包括只使用了三随机种子、扰动预算为L∞在处理空间的标准化压力测试、CAPGD无法用于XGBoost、缺乏更广泛的攻击方法和约束投影、以及缺少经济成本/攻击者预算的量化模型。

---

## 458. KLTNet: Learning Sparse Feature Tracking for Robust and Accurate Monocular Visual-Inertial Odometry

**arXiv ID:** 2608.24544 | [PDF](https://arxiv.org/pdf/2608.24544v1)

**作者:** Renbiao Jin `[一作]` (Shanghai Jiao Tong University), Wenxian Yu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 KLTNet，一种轻量级、可插拔的稀疏特征跟踪器，用于替代传统的 KLT 跟踪器，从而提升视觉-惯性里程计（VIO）的跟踪精度与鲁棒性。

**💡 创新点**

创新点包括：① 粗细尺度的密集流初始化 + 三角包裹的参考图像补丁精细化，既保证全局稳健又保持局部精确；② 通过可微分多视角三角化学习可变形的观察权重；③ 仅需 1/4 分辨率的密集流网络，保持低运算成本；④ 能够在嵌入式 Jetson AGX Orin 上实时运行。

**🔧 技术方法**

采用的技术：密集光流网络 CoarseFlowNet（改进的 SEA-RAFT），三角包裹补丁编码器 TriPatchRefiner，基于 RNN 的流迭代更新，基于三角化的自监督权重学习，传统的多帧 VIO 后端（VINS‑Mono、OpenVINS）。

**📊 数据集**

训练数据集：TartanAir；评估数据集：EuRoC、TUM‑VI、Replica、KITTI‑360、UMA‑VI 以及自采低纹理室内数据集。

**📈 对比分析**

与经典 KLT、RAFT‑Sparse、SEA‑RAFT‑Sparse 等基线比较；在 VINS‑Mono 上平均 ATE 降低 34%（EuRoC）或 49%（TUM‑VI），在 OpenVINS 上平均 ATE 降低 33%；在低纹理、快速运动、光照变化等极端场景下仍保持稳定；在 Jetson AGX Orin 上可实现实时推理并保持低显存占用。

**⚠️ 局限性**

局限性：假设场景为静态，无法处理动态物体；在极端运动下仍需更强的再定位机制；目前仅支持单目 VIO，需进一步扩展至多相机系统。

---

## 459. RoG-DAgger: Rollout-Guided Post-Training for End-to-End Driving

**arXiv ID:** 2608.24525 | [PDF](https://arxiv.org/pdf/2608.24525v1)

**作者:** Liangyu Zhong `[一作]` (CARIAD SE, Volkswagen Group), Hanno Gottschalk `[通讯]` (Technical University Of Berlin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出RoG-DAgger后训练框架，通过短期动力学rollout扩展专家轨迹与速度空间，基于可解决性估计提前接管，并对齐专家与学生视角，从而提升闭环端到端驾驶模型性能。

**💡 创新点**

①使用rollout评估并生成预接管轨迹与速度监督；②引入可解决性（PONR）触发提前接管；③对齐专家视角与学生观察，消除信息不匹配。

**🔧 技术方法**

轨迹规划、短期动力学rollout、DAgger集成、PID控制、强化学习仿真。

**📊 数据集**

CARLA Leaderboard 2.0 训练路、Bench2Drive、Longest6 v2、Fail2Drive。

**📈 对比分析**

在Bench2Drive提升5.3驾驶分数、6.2成功率；在Longest6 v2驾驶分数从22提升至44、路线完成率从70提升至88；在Fail2Drive成功率从55%提升至66%；相较于TakeAD、MindDrive、TakeVLA等方法表现更佳。

**⚠️ 局限性**

rollout假设周围车辆不反应，导致可解决性估计不准确；候选轨迹集合可能不完整；方法依赖CARLA仿真，难以直接迁移至真实场景；PONR估计仍为近似。

---

## 460. SatDL: Jointly Optimizing Data Redistribution and Training for Satellite-Based Distributed Learning

**arXiv ID:** 2608.24516 | [PDF](https://arxiv.org/pdf/2608.24516v1)

**作者:** Hao Wu `[一作]` (National University of Singapore), Jingxian Wang `[通讯]` (National University of Singapore)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了SatDL框架，联合优化卫星间分布式学习中的数据重分配与训练时间，最小化总端到端学习时延。

**💡 创新点**

创新点在于采用Distributor–Critic架构，将通信延迟与训练耗时建模为可微分目标；通过类梯度原型近似梯度多样性，将离散的数据重分配问题转化为连续优化，实现在保持高精度的前提下仅部分数据重分配，平衡通信与收敛速度。

**🔧 技术方法**

使用了分布式学习算法（FedAvg、FedProx等）、梯度多样性理论、类梯度原型估计、基于梯度下降的联合优化、卫星网络拓扑与ISL带宽模型、以及NVIDIA Jetson与A100 GPU的硬件仿真。

**📊 数据集**

采用了5个真实/合成数据集：CIFAR‑10、CIFAR‑100、Flickr Mammals、Satellite Land Cover 及 Traffic Signs（MTSD‑Sign）。

**📈 对比分析**

与 Non‑IID、IID、FedProx、Hybrid‑FL 等基线相比，SatDL 在所有数据集上总学习时间均下降 14–18%，能量消耗下降 12–88%，同时保持与最优基线相近的测试准确率。

**⚠️ 局限性**

局限性在于实验仅在 Starlink 类星座和固定 ISL 带宽模型下验证；对极端高非IID、动态网络拓扑及不同硬件平台的鲁棒性尚未充分评估；需一次校准实验，参数迁移到不同任务或硬件时可能需要重新调参。

---

## 461. NeuralParker: A Reinforcement Learning Planner for Irregular Parking Environments

**arXiv ID:** 2608.24485 | [PDF](https://arxiv.org/pdf/2608.24485v1)

**作者:** Zihan Wang `[一作]` (Tsinghua University), Shengbo Eben Li `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 NeuralParker 的强化学习混合规划器，用于在不规则环境中实现任意姿态停车。

**💡 创新点**

创新点包括：①目标相对全局几何表示，将障碍物与边界信息以顶点形式编码，保持全局上下文；②采用曲率–长度弧形动作空间，并在终止时使用基于 Hermite 多样化连接器；③构建了阶乘与拓扑压力两大基准，用于评估全景成功率与轨迹质量。

**🔧 技术方法**

核心技术包括：目标相对坐标系、注意力自注意网络、PPO 强化学习、曲率约束弧线动作、基于 Hermite 曲线的终端装配器、起点地理梯度训练。

**📊 数据集**

使用两组程序生成的二维场景基准：Factorial Parking Benchmark（81训练场景+27测试场景）与 Topology‑Stress Benchmark（48训练+18测试场景），并在真实车辆的配送停车点上进行实车评估。

**📈 对比分析**

与自适应 HOPE、Hybrid A* 等传统和混合基线对比。NeuralParker 在全景成功率上提升 2–3%，轨迹长度、转向次数、曲率变化均显著下降；在真实车辆测试中规划时间最低、成功率最高。

**⚠️ 局限性**

局限性：仅在二维静态场景与预定义顶点预算内验证，未考虑动态障碍、感知不确定性、3D 车身尺寸等；终端连接仍使用手工规则，缺乏完整端到端可微优化。

---

## 462. Beyond Static Interpretability: Anticipating Post-SFT Mechanisms from Pre-SFT Parameters for Better Tuning

**arXiv ID:** 2608.24482 | [PDF](https://arxiv.org/pdf/2608.24482v1)

**作者:** Hang Chen `[一作]` (Nanyang Technological University), Wenya Wang `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

预测SFT后模型的关键参数并基于此进行定位‑微调（locating‑then‑tuning）

**💡 创新点**

提出前瞻性预测框架：利用Taylor展开将未来参数重要性映射到仅需预SFT参数和1%数据的探测SFT上，实现对后期机制的预测

**🔧 技术方法**

采用Taylor展开、归因补丁（Attribution Patching）、一次性前向+反向传播、梯度方向与距离分离、双粒度（神经元/组件）定位以及LoRA动态分配等技术

**📊 数据集**

使用Mistral‑7B、LLaMA‑2‑13B、Qwen‑3‑30B等大模型，并评估GLUE、BOOL、Arithmetic、IOI、Induction、WinoGrande、Genderr、Docstring等数据集

**📈 对比分析**

与梯度导向与因果导向基线（Graft、WAGLE、FLU、CLUE、CircuitLoRA）对比，在目标任务准确率（TTA）和通用能力保持率（PTA）上均表现最佳；在更大模型上保持稳定且整体时间成本最低

**⚠️ 局限性**

仅针对单步next‑token任务，难以直接扩展到多token生成/指令跟随；对多任务冲突的polysemantic neuron缺乏有效补偿，限制了多目标联合训练的效果

---

## 463. Pivot-and-Station Multi-Agent Path Finding: Solvability, Complexity, and Algorithms

**arXiv ID:** 2608.24585 | [PDF](https://arxiv.org/pdf/2608.24585v1)

**作者:** Andrea Di Nezza `[一作]` (Politecnico di Torino), Sara Bernardini `[通讯]` (University of Oxford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并研究了一种新的多智能体路径规划变体——Pivot-and-Station MAPF（PS‑MAPF），其目标是让一部分任务智能体先访问任意一个可互换的枢轴点，然后全体智能体最终停靠在匿名的站点上，并对该问题的可解性、复杂度及算法求解进行了系统性分析。

**💡 创新点**

创新点包括：1）在任意连通图上给出了完整的可解性判定与多边形有效距离度量；2）证明即使只有一个枢轴点，最小化站点完成时间和流时间也是NP‑hard；3）提出三类算法——完整基线、基于SAT的最优求解器以及快速的 Pivot‑Prioritized Planning（PPP）方法，并在实验中验证了其优越性。

**🔧 技术方法**

主要技术手段包括：2‑边连通性与桥树分解、有效距离（effective distance）计算、基于旋转和合并的构造可解性证明、基于最大流的匿名MAPF求解、空间–时间 A* 与流图的优先规划、以及离散时间 SAT（CP‑SAT）编码。

**📊 数据集**

使用的数据集主要为：1）随机生成的16×16、20×20、28×28网格实例，控制障碍物密度、智能体密度、枢轴点与站点比例；2）Moving AI 基准集中的空白、随机障碍和仓库地图；3）在实验中对 1–30 个枢轴点、不同任务智能体比例和总智能体数进行网格规模扩展。

**📈 对比分析**

实验中与基线、SAT 最优求解器及 PPP（多种优先顺序）进行对比，评估指标包括运行时间、站点完成时间（makespan）和流时间（flowtime）以及成功率。结果显示 PPP 在 74–89% 的实例上能在几百毫秒内得到解，其 makespan 与 flowtime 与基线相比平均低 2–3 个步骤，且显著快于 SAT 求解器；而 SAT 在低密度/单枢轴情况下能得到最优解，但在高密度/多枢轴时往往超时。

**⚠️ 局限性**

主要局限包括：PPP 不是完备的，遇到极端瓶颈（少枢轴且大部分智能体任务化）时会失败；基线虽完备但解质量极差；SAT 最优求解器虽然能得到全局最优但在大规模实例上易爆炸；此外，本文未考虑枢轴点和站点的多种类型或动态目标，未来工作可在此方向进一步扩展。

---

## 464. Joint Optimization of Tool Creation and Use for Large Language Model Agents

**arXiv ID:** 2608.24571 | [PDF](https://arxiv.org/pdf/2608.24571v1)

**作者:** Zhi Rui Tam `[一作]` (Appier AI Research), Hung-yi Lee `[通讯]` (National Taiwan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

训练一个单一的语言模型，使其能够同时生成可复用工具（Python函数+JSON schema）并在使用时调用该工具，形成闭环反馈；

**💡 创新点**

通过强化学习将工具生成和使用分离为三条奖励信号（执行准确率、LLM-judge质量、格式一致性），并同步评估器以避免循环评估；同时采用易到难的训练策略，促使工具能泛化到更困难的实例；

**🔧 技术方法**

使用 DAPO（改进的 GRPO）进行强化学习；多任务训练共享策略；使用 Qwen3‑4B‑Instruct 作为基模型，采用 LoRA 微调；工具生成输出为可解析的 Python 代码和 OpenAI JSON schema；评估器为大模型 30B‑A3B，周期性同步；

**📊 数据集**

13 类 Reasoning‑Gym 任务（算术、算法、代数、游戏、逻辑推理）做训练；评估包含 RG（Seen/Unseen）和 OOD 数据集 TabMWP‑Hard、GQA；

**📈 对比分析**

与 LATM、CRAFT、Trove、KTCE、ReTool、LATM‑distilled、Qwen3‑30B‑A3B 等基线对比，SMITH 在 RG（Unseen）取得 79.9% 宏平均准确率，优于所有基线；在 OOD 任务 TabMWP‑Hard、GQA 上也领先或接近最优；同时使用的 token 数量比标准 CoT 少 32 倍，输出 token 仅 100，极大提升算力效率；

**⚠️ 局限性**

局限包括：仅在 4B‑8B 范围内验证，未知更大规模的表现；评估器高度依赖 30B 大模型，Self‑Judge 结果不稳定；未对生成工具的安全性或对抗攻击做正式验证；未评估生成工具的可读性与开发者友好性；未支持多文件或服务器级工具生成，亦未鼓励并行/多工具调用。

---

## 465. Persistent Cross Entropy

**arXiv ID:** 2608.24549 | [PDF](https://arxiv.org/pdf/2608.24549v1)

**作者:** Sijin Yeom `[一作]` (Pohang University of Science and Technology), Jae-Hun Jung `[通讯]` (Pohang University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种新的持久跨熵（Persistent Cross Entropy, PCE）度量，用于比较不同持久性图并在知识蒸馏和因果方向推断中应用。

**💡 创新点**

创新点在于：①构造了在不同事件空间上的诱导概率，克服了传统熵无法比较不同持久性图的问题；②证明了诱导概率与 PCE 的一致性、稳定性和 KL 对应关系；③在无联合图的情况下直接获取因果方向，并在知识蒸馏中实现更有效的拓扑约束。

**🔧 技术方法**

使用持久性同义函数（如高斯相似度）与持久性加权、欧氏距离、Wasserstein 距离、卷积核/图像方法等统计和机器学习技术来构建概率、计算熵、评估稳定性，并将其嵌入到 ResNet 的蒸馏损失中。

**📊 数据集**

主要实验数据集包括：①合成点云（两个环+噪声）用于对比持久熵相同但结构不同的图；②弹簧质量系统的时序数据，用于因果方向推断；③CIFAR‑100 图像集，用于 ResNet56→ResNet20 的知识蒸馏。

**📈 对比分析**

与传统的对称 TDA 距离（bottleneck、Wasserstein）、Betti 曲线、持久景观、轮廓、尺度空间核、持久图像等方法对比，PCE 在四种耦合规程分离中取得最高 R²（0.56）并在双向因果分离中几乎完美（R²≈0.99）；在蒸馏实验中，EM‑PCE 的 Top‑1 准确率最高（71.49%），比基准 TopKD 提升约 0.5%。

**⚠️ 局限性**

局限性包括：对非线性或混沌动力学的因果推断效果不稳定；需要先验选择相似度函数和响应尺度，且理论假设持久性有上界；诱导概率不保证可逆性，且对噪声敏感。

---

## 466. Discovering Adaptive Transmission Programs for Collective Innovation

**arXiv ID:** 2608.24545 | [PDF](https://arxiv.org/pdf/2608.24545v1)

**作者:** Cédric Colas `[一作]` (Inria), Maxime Derex `[通讯]` (Institute for Advanced Study in Toulouse)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并验证了基于状态感知的传输协议框架，利用LLM驱动的进化搜索在模拟的“Collective Little Alchemy”任务中自动生成并优化信息共享策略；

**💡 创新点**

创新点在于把传输协议视为可解释的、可编程的状态感知程序，突破传统网络结构限制；

**🔧 技术方法**

采用大型语言模型（GPT‑4）生成和改进协议代码，并用进化算法评估其在仿真中的表现；

**📊 数据集**

使用改编自 Little Alchemy 2 的 720 元素组合游戏的模拟数据，涵盖不同代理类型和任务变体；

**📈 对比分析**

与五类基线（无共享、固定配对、动态配对、随机共享、网络共享）对比，进化得到的协议在 150 步时平均提升约 33%（最高 37%），且优势可延续至 300 步；

**⚠️ 局限性**

局限性包括：实验域过于简化且规则确定性强、假设全局可观测且无部分信息约束、未在真实人类群体中验证协议的转移效果。

---

## 467. Easier, but Not Easy: Nash Welfare under Lexicographic Valuations

**arXiv ID:** 2608.24537 | [PDF](https://arxiv.org/pdf/2608.24537v1)

**作者:** Soumil Aggarwal `[一作]` (Quantbox Research), Jatin Yadav `[通讯]` (IIT Delhi)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在层次（大间隙）加性（lexicographic）偏好下最大化Nash福利问题，提供近似、精确与难度分析。

**💡 创新点**

首次利用层次结构得到更好的1/√2近似、构造了匹配的LP积分间隙证明、提出了基于支配的分支修剪框架，并展示了NP-hard与APX-hard的极限。

**🔧 技术方法**

使用配置LP拉格朗日松弛与精细化归约、支配关系与互斥矩阵的叶子计数证明、以及多阶段归约与树形剪枝技术。

**📊 数据集**

本文为理论性工作，未使用实际数据集。

**📈 对比分析**

与现有最佳加性约束下的0.692近似相比，得到0.707近似；在常数个体的有序或倍增案例中实现多项式时间精确解；在一般n下给出EPTAS。

**⚠️ 局限性**

精确算法仅适用于常数个体，近似比率可能不是最优；未给出多项式时间的全局最优或更高近似；实验验证缺失。

---

## 468. LumiXAI: A Modular Full-Stack Framework for Feature Attribution

**arXiv ID:** 2608.24524 | [PDF](https://arxiv.org/pdf/2608.24524v1)

**作者:** Alfio Ferrara `[一作]` (Università degli Studi di Milano), Elisabetta Rocchetti `[通讯]` (Università degli Studi di Milano)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个模块化全栈框架 LumiXAI，统一管理并可视化文本、图像分类、文本生成和文本到图像扩散模型的特征归因分析。

**💡 创新点**

提供统一插件化架构、RESTful 后端、持久化存储以及三种访问层级（非程序员、开发者、扩展者），实现跨任务、跨模型的交互式双向归因。

**🔧 技术方法**

采用 FastAPI、Next.js、Docker Compose，集成 Captum、SHAP、LIME 等归因库，并通过 Hugging Face Hub 加载模型，使用 SQLite+JSON 进行持久化。

**📊 数据集**

评估使用 Civil Comments 语料进行毒性分类，并在自定义提示词（如“a impressionism painting of a pizza”）下生成图像，使用公开的文本到图像扩散模型。

**📈 对比分析**

通过用户研究比较三类用户对系统可用性的 Likert 评分，结果显示专家和非专家均认为易用性高；系统性能主要受模型推理时间影响，归因质量依赖所选库，未测量框架开销。

**⚠️ 局限性**

仅支持有限的模型族与归因方法，缺乏多模态图文模型支持；并发多用户场景下鲁棒性不足；归因结果受方法本身限制，未做系统级性能基准。

---

## 469. PeakBench: Benchmarking Resource-Aware Tool Invocation in LLM Agents

**arXiv ID:** 2608.24509 | [PDF](https://arxiv.org/pdf/2608.24509v1)

**作者:** Zhi-Kai Chen `[一作]` (Nanjing University), Han-Jia Ye `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了PeakBench基准，拆分为逻辑规划（依赖提取）和物理调度两部分，并引入资源感知调度上下文RASC，评估LLM在有限资源环境下的工具调用与并行执行能力。

**💡 创新点**

创新点在于：①通过沙箱执行自动获得执行级依赖关系，避免人工标注；②将依赖分析与调度评估分离，使失败可归因；③提供资源使用剖面并通过RASC让模型在不训练的情况下利用资源信息改进调度。

**🔧 技术方法**

技术包括：LLM提示与解析、工具调用沙箱环境、依赖图构建与评估（GED、Edge F1）、资源剖面采集与模拟调度、RASC上下文生成、基准对比算法（ASAP、Serial、规则调度）。

**📊 数据集**

数据集为PeakBench：约300条可执行多工具工作流，覆盖1.2k MCP兼容工具、130台服务器，分为易/中/难三个难度层级；每个工具附带经高频系统监控获得的资源剖面。

**📈 对比分析**

比较方法：在两维度上对模型和基准做量化：逻辑规划用GED与Edge F1；物理调度用Scheduling Latency、Capacity Violation Area(CVA)和Strict MRU；结果显示：强的逻辑规划并不必然带来安全高效的调度；RASC在多数模型上能减少CVA、提升Strict MRU，且接近规则调度器，但提升因模型而异。

**⚠️ 局限性**

局限性：①工作流为合成或人工构造，缺少真实工业场景；②资源剖面采样有限，可能无法覆盖所有真实负载；③RASC仅为上下文辅助，未对模型进行专门的调度训练；④不同模型对资源信息的利用差异大，表明仍需进一步研究更通用的资源感知策略。

---

## 470. Computing an e-net of a closed hyperbolic surface

**arXiv ID:** 2608.24497 | [PDF](https://arxiv.org/pdf/2608.24497v1)

**作者:** Vincent Delecroix `[一作]`, Monique Teillaud `[通讯]`

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文提出了一种从基本多边形表示开始，对双曲曲面构造 ε‑网（伪 ε‑网）的算法，并将该网用于计算曲面长度谱和短周期（systole）。

**💡 创新点**

创新点在于：①首次给出基于 Delaunay 细化的双曲曲面 ε‑网构造算法；②引入伪 ε‑网概念，避免在极细小曲线上产生无穷点；③利用 Delaunay 网的几何特性实现长度谱与最短曲线的多项式时间计算。

**🔧 技术方法**

核心技术包括：双曲几何的 Delaunay 三角剖分与边翻转、圆盘/椭圆投影、Collar Lemma、点集 packing 与 net 定义、广度优先搜索等。

**📊 数据集**

算法在理论上对任意闭双曲曲面（由单点 Delaunay 产生的 Dirichlet 多边形表示）进行操作，无需具体实验数据集；结果以复杂度表达式给出。

**📈 对比分析**

与传统基于三角剖分或直接曲线搜索的方法相比，本文的方法在厚区域可实现 O(g·m·L·e^L) 的时间复杂度（m 为长度谱的最大重数），而在极薄区域通过伪网实现多项式时间的 systole 计算；总体上比现有的指数级或无效方法更高效。

**⚠️ 局限性**

局限性包括：①构造 ε‑网（尤其伪网）对曲面 genus 的依赖指数级，导致实际构造复杂度高；②算法需要从 Dirichlet 多边形得到单点 Delaunay，实际实现时需额外的预处理；③在极薄曲面上仍需额外的阈值设置，可能影响鲁棒性。

---

## 471. SeisMamba: Low-Latency Single-Station Seismic Magnitude Estimation for Spatially Distributed Earthquake Early Warning

**arXiv ID:** 2608.24561 | [PDF](https://arxiv.org/pdf/2608.24561v1)

**作者:** Quenton Yeo `[一作]` (University of Sydney), Huaming Chen `[通讯]` (University of Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种轻量级的单站点地震规模估计模型SeisMamba，能够在低延迟下从三分量地震波形快速预测规模

**💡 创新点**

创新点在于结合稀疏选择性状态空间模型与多尺度卷积编码，构建高效的长时序建模架构，同时加入辅助时间序列头提供时序监督

**🔧 技术方法**

使用Mamba式选择性状态空间网络、层次卷积编码、多尺度特征融合以及辅助时间预测头等技术

**📊 数据集**

利用全球单站地震波形数据库STEAD进行训练与评估，并在Chile–Taiwan地区做区域留存实验

**📈 对比分析**

与MagNet、PhaseNet、EQTransformer、AMAG、U‑Mamba等基线比较，SeisMamba在STEAD上获得MSE0.0628、RMSE0.2506、R²0.9443，推理时间仅0.55 ms，明显优于大多数基线；在区域留存实验中依然保持较好性能（R²≈0.852）

**⚠️ 局限性**

局限在于对区域分布漂移的鲁棒性仍有限，模型在不同地区仍会出现性能下降，且缺乏完整的不确定性评估和正式的EEW决策框

---

## 472. Parameter-Level Attribution of Symmetry in Trained Networks Though Parameter-Wise Functional Sensitivity

**arXiv ID:** 2608.24700 | [PDF](https://arxiv.org/pdf/2608.24700v1)

**作者:** Alan Muriithi `[一作]` (University of Oxford), Torben Berndt `[通讯]` (Heidelberg Institute for Theoretical Studies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究神经网络参数空间与函数空间之间的对称性提升，提出功能灵敏度方法来量化参数对对称性的贡献，并在分类器与Hamiltonian网络上进行实验验证。

**💡 创新点**

首次将功能灵敏度与Lie群对称性结合，给出可计算的局部提升判据，并在参数空间给出对称生成元的逐参数归因；同时提出沿对称轨道和趋向等变子空间的局部参数方向。

**🔧 技术方法**

利用功能灵敏度、神经切线核、伪逆最小二乘法、参数空间轨迹积分（resolved 与 fixed），以及适用于动力学学习的 ASRNN 架构。

**📊 数据集**

使用合成的 annulus/inner‑disc 分类数据集（90° wedge 与 360°）以及从 Mexican‑hat 势能产生的二维轨迹数据（多 α 值）。

**📈 对比分析**

通过在参数空间沿计算出的对称与等变方向积分并评估函数空间中的角度偏差，发现每步重新计算（resolved）能准确沿对称轨道移动或减小等变误差，而固定方向随时间漂移，表明局部方法有效但需频繁更新。

**⚠️ 局限性**

仅提供局部一阶提升判据，无法保证全局或光滑的参数空间动作；需要在每一步重新求解；在高维过参数化网络中对齐误差与数值稳定性仍是挑战。

---

## 473. Who Falls for SMiSh? Learning Through Survey Data Where to Best Target Awareness Training for Mobile Messaging Attacks

**arXiv ID:** 2608.24669 | [PDF](https://arxiv.org/pdf/2608.24669v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 474. A Multimodal Foundation Model for Longitudinal Patient Representation and Scalable Insight Generation in Oncology

**arXiv ID:** 2608.24688 | [PDF](https://arxiv.org/pdf/2608.24688v1)

**作者:** Eugene Vorontsov `[一作]` (Tempus AI Incorporated), Siqi Liu `[通讯]` (Tempus AI Incorporated)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了一种名为 oFM 的多模态基线模型，能整合癌症患者长期临床记录、DNA/RNA 分子谱和 H&E 病理图像，实现患者状态的时序表示。

**💡 创新点**

创新点包括：①将多模态信息（临床、分子、影像）以每日“episode”形式编码并通过 Transformer 架构实现时序聚合；②在预训练阶段加入数值嵌入（FoNE）与自监督未来状态预测，提升对时间依赖性的建模；③提出可解释机制发现框架，将嵌入分解为稀疏潜在方向、因果可操控性和检索驱动的时间先后图谱。

**🔧 技术方法**

技术手段主要有：GatorTron（临床文本编码器）、PRISM2（H&E 图像编码器）、Transformer-based 轨迹编码器、FoNE 数值嵌入、I-JEPA 自监督预测、VICReg 正则化、Cox 部分似然、逻辑回归探针、稀疏自编码器与因果激活、检索增强生成（MedGemma 4B + BioLORD-2023）等。

**📊 数据集**

使用了 1,672,203 名来自 Tempus 的去标识多模态真实世界肿瘤队列，其中 1,045,011 名用于 episode 编码预训练，386,382 名用于时序编码器训练，92,567 名（含配对 DNA+RNA）用于最终端到端微调。

**📈 对比分析**

评估方式：在冻结 oFM 嵌入上进行二分类探针（治疗反应、无进展生存、总体生存）和治疗效益预测（11 个比较治疗队列），与基于 7,520 个手工挑选特征的基线做对比。结果显示：整体生存 AUC 0.774 对 0.563，PFS 0.688 对 0.544，反应 0.585 对 0.513；在治疗效益排序中，oFM 的 t_AUTOC/SD 为 4.61（显著优于基线 1.38），Cox C-index 在两臂均优于基线，表明模型在预测和排名方面均具有显著优势。

**⚠️ 局限性**

局限性包括：①仅为回顾性真实世界数据，存在混杂和缺失问题；②缺乏外部或前瞻性验证，预测性能可能在其他数据集下降；③模型解释层面仍依赖稀疏自编码与检索，未能完全解开所有生物学机制；④对图像的单次滑动窗口处理可能忽略大幅度病理变异；⑤需要更多计算资源，模型规模和训练时间较大。

---

## 475. TurboT2VA: Fast Large-Scale Text-to-Video-Audio Generation via Score-Regularized Consistency Distillation

**arXiv ID:** 2608.24674 | [PDF](https://arxiv.org/pdf/2608.24674v1)

**作者:** Xiaoda Yang `[一作]` (Zhejiang University), Zhou Zhao `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 TurboT2VA 框架，将 40 步文本‑视频‑音频扩散模型压缩为 4 步学生模型，并通过进阶训练课程和架构感知加速堆栈实现同步视频音频的高效生成。

**💡 创新点**

创新点包括：① 对视频和音频采用分解损失并进行每模态归一化，解决模态不平衡；② 采用 dCM→sCM→sCM+DMD 的渐进式课程，先学习轨迹一致性后再进行分布匹配，提升质量‑多样性‑同步性平衡；③ 设计了专属的稀疏注意力调度、W8A8 量化与融合 Transformer 核心，保持跨模态和文本注意力密集，同时显著降低每步计算成本。

**🔧 技术方法**

技术手段包括：分布式一致性蒸馏（dCM、sCM、rCM）、分布匹配蒸馏（DMD）、逐模态梯度归一化、SageSLA 稀疏注意力、后缩放 W8A8 线性算子、融合多模态 Transformer 操作、文本上下文压缩等。

**📊 数据集**

使用约 10 万条 512×768 分辨率、121 帧的文本‑视频‑音频训练集；评估时采用 200 句提示集并在 512×768 与 1024×1792 分辨率下进行对比。

**📈 对比分析**

与 40 步教师、开源/闭源 T2VA 系统及基准（JavisBench、VBench、TTA‑Bench、MS‑CLAP 等）对比，4 步 TurboT2VA 在 512×768 下实现 20.1× 的推理速度提升，保持甚至提升了视觉质量、音频保真度、同步性和多样性；在 1024×1792 高分辨率下，完整加速堆栈将生成器推理时间从 318.74 s 降至 5.83 s，实现 54.67× 的速度提升。

**⚠️ 局限性**

局限性包括：仍依赖高端 GPU（单卡 H20 训练/推理），模型规模庞大（19 B 参数）且 4 步采样对极低延迟场景仍不够；进阶课程与加速堆栈需手动调参，难以迁移到其它视频‑音频架构；在更高分辨率或不同模态比例下的性能与稳定性尚未充分验证。

---

## 476. ReGround-Surg: Reliability-Guided Anchor Grounding for Referring Surgical Video Segmentation

**arXiv ID:** 2608.24671 | [PDF](https://arxiv.org/pdf/2608.24671v1)

**作者:** Jiaxin Wen `[一作]` (University of Exeter), Zeyu Fu `[通讯]` (University of Exeter)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ReGround-Surg框架，通过文本引导的可靠性门、门控侧适配器和可靠性加权视觉‑文本注意力，在SAM2两阶段指涉外科视频分割中改进anchor grounding，提升第一阶段定位精度，进而减少跟踪误差。

**💡 创新点**

创新点包括：①利用文本条件生成空间可靠性图，②共享该图同时调制视觉特征与视觉‑文本注意力，③实现轻量化且可插拔的anchor grounding改进，④专门解决SAM2两阶段模型对anchor错误高度敏感的问题。

**🔧 技术方法**

使用技术包括：SAM2预训练模型、CLIP文本编码器、CSTMamba跨模态融合模块、门控侧适配器（GSA）、可靠性加权视觉‑文本注意力（RW‑V2T）、文本‑视觉可靠性门、零/恒等初始化、AdamW优化器、线性warm‑up与cosine decay学习率调度。

**📊 数据集**

使用数据集：Ref-EndoVis17 与 Ref-EndoVis18（工具与组织子集），均基于EndoVis17/18重标注，用于训练、验证与测试。

**📈 对比分析**

通过与ReSurgSAM2及其他RVOS方法在J＆F、J、F指标上比较，结果显示在所有三个数据集上均获得最高分数；相比ReSurgSAM2提升+3.77、+3.09、+0.94；速度几乎无损，仅增加0.5M参数。

**⚠️ 局限性**

局限性包括：对空间扩散的组织目标改进有限；gate floor α需手工调参；Stage 2跟踪仍未改进，记忆漂移与长时间遮挡问题仍存在；可靠性门在不同手术域的泛化性尚待进一步验证。

---

## 477. From local kernels to global form: modeling the emergence of musical content

**arXiv ID:** 2608.24660 | [PDF](https://arxiv.org/pdf/2608.24660v1)

**作者:** Francesco Vitucci `[一作]` (Conservatorio di Musica N. Piccinni di Bari), Francesco Scagliola `[通讯]` (Conservatorio di Musica N. Piccinni di Bari)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

使用重叠滑动窗口从Debussy《Syrinx》单个符号序列中估计局部马尔可夫转移核，并对其进行分析与合成。

**💡 创新点**

首次将观察驱动的有向转移核轨迹用于同一作品的结构分析与生成，突出时变马尔可夫的可解释性。

**🔧 技术方法**

采用滑动窗口计数估计、Jensen–Shannon与Hellinger距离评估、中心对齐与硬回退的采样生成技术。

**📊 数据集**

Debussy《Syrinx》全程273个逻辑音符事件（包括绝对音高与精确时值）。

**📈 对比分析**

比较连续窗口间的JS距离，发现窗口长度6时两条音高与时值轨迹均达到最大值并与A–B–A′边界对齐，显示局部变化显著，但存在宽阔平坦区，不能唯一定位分段；生成实验表明窗口增大导致事件级偏差显著提升。

**⚠️ 局限性**

方法仅在单一作品上验证，无法作为通用自动分割器；对窗口几何敏感，连续窗口比较限制了边界检测能力，且未考虑音高与时值的联合一致性。

---

## 478. Simthesizer: An Agent-Driven Simulation Framework for LLM Serving Systems

**arXiv ID:** 2608.24650 | [PDF](https://arxiv.org/pdf/2608.24650v1)

**作者:** Wonung Kim `[一作]` (Korea Advanced Institute of Science and Technology), Jongse Park `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了一个名为Simthesizer的可扩展LLM服务模拟框架，能够通过代理驱动的低级实现快速添加新特性并保持模拟器的可维护性。

**💡 创新点**

创新点在于将完整的服务工作流统一表示为动态DAG，并将控制决策模块化为可插拔组件，同时配合可编程的编码代理和验证工作流，实现了自动化、可审计的模拟器扩展。

**🔧 技术方法**

采用的技术包括统一动态DAG抽象、模块化控制层、基于LLM的编程代理（如OpenAI Codex）以及基于trace和参考的验证机制。

**📊 数据集**

评估使用了SWE-bench、tau-bench和ShareGPT等agentic及非agentic工作负载，并对比了vLLM的真实系统日志。

**📈 对比分析**

与LLMServingSim2.0和Vidur在相同功能扩展任务下对比，Simthesizer的平均吞吐量误差仅为2.51%（对比6.03%），模拟速度提升约23倍到285倍。

**⚠️ 局限性**

限制在于仍需依赖外部LLM编程代理的可靠性与提示质量，缺乏对硬件/网络细粒度模型的完整覆盖，以及在某些异构模型（如Hybrid Mamba）中仍需手动优化层级重用。

---

## 479. On-Policy Self-Distillation in Diffusion Models

**arXiv ID:** 2608.24646 | [PDF](https://arxiv.org/pdf/2608.24646v1)

**作者:** Wei Zhou `[一作]`, Tat-Seng Chua `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于自监督的Diffusion模型后训练方法（DiffusionOPSD），通过在每个迭代周期内先用冻结的行为策略生成轨迹，然后利用图像级奖励梯度构造带有上限的正负清晰输出目标，再以分离的停止梯度监督形式拟合这些目标，最后用EMA更新行为策略并在下一轮迭代中重新生成目标。

**💡 创新点**

创新点包括：①将奖励到目标的转换与有限拟合过程分离，能够独立测量目标构造与实现的收益；②构造正负两类受限清晰输出目标，既给出提升方向又提供排斥参考；③在策略上采用自监督的 on‑policy 迭代，避免了端点奖励直接监督中间步骤的结构性不匹配问题；④通过可测量的目标构造与实现差异，揭示了奖励方向比目标半径更重要的结论。

**🔧 技术方法**

主要技术：Rectified‑flow diffusion、奖励梯度上升/下降构造目标、分离的 stop‑gradient 目标监督、双分支拟合损失（OPSD）、EMA 更新行为策略、有限拟合（单步 AdamW）和目标半径投影。

**📊 数据集**

实验数据集：SD3.5‑M 与其 9‑step 步进压缩版 Z‑Image‑Turbo 作为基准模型；使用 Pick‑a‑Pic 作为训练提示集；在 DrawBench 提示集上进行端到端评估；奖励函数包括 Pick‑Score、CLIPScore、HPSv3、DeQA、VLM‑Pairwise、Aesthetic 等十个公开/内部评估指标。

**📈 对比分析**

与 FlowGRPO、ReFL、DiffusionNFT 等方法对比，DiffusionOPSD 在 20 个奖励匹配设置中 19 个获得最高的保留集得分，单奖励提升最高可达 44%；训练成本相比 DiffusionNFT 降低 40–63% GPU‑小时；在联合三奖励训练中，单一模型保持并提升了三项奖励的表现，优于 DiffusionNFT 的两阶段学生；在人类评测中也取得最高偏好率。

**⚠️ 局限性**

局限性：对低噪声查询的依赖，目标半径与噪声水平需要调参；在高噪声查询或极端目标半径下性能下降；不支持 CFG 训练且在不同评估尺度下略有退化；方法依赖于奖励梯度信息，对非梯度可导奖励适用性有限。

---

## 480. EVEREST:Endogenous Vision-Language Reinforcement Reasoning Exploration for Urban Socio-Semantic Segmentation

**arXiv ID:** 2608.24640 | [PDF](https://arxiv.org/pdf/2608.24640v1)

**作者:** Qixiu Li `[一作]` (National University of Defense Technology), Weifeng Xu `[通讯]` (National University of Defense Technology)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于视觉语言强化学习的自我探索框架，能主动识别并修正城市社会语义实体的边界。

**💡 创新点**

创新点在于采用 egocentric 视觉探索与伪代码引导的实例枚举/验证两阶段流程，结合结构化提示与 RL 训练，实现主动边界修正与结构化推理。

**🔧 技术方法**

利用 Qwen2.5‑VL‑3B 视觉语言模型、SAM2 固定分割器、伪代码解析、强化学习（GRPO）等技术。

**📊 数据集**

在真实的 SocioSeg 城市社会语义分割基准集上进行评测。

**📈 对比分析**

与 UNet、SegFormer、SocioReasoner 等多种基准模型对比，取得 50.4 cIoU / 61.4 F1 的最佳 Avg. Rank，尤其在 Socio‑class 与 Socio‑function 任务上提升显著。

**⚠️ 局限性**

仍依赖高质量地图‑卫星对齐，且算法复杂度高、对 RL 超参敏感，且在极端遮挡或低分辨率场景下性能下降。

---

## 481. Thermal Tuning Overhead in Wafer-Scale Optical Interconnects for LLM MoE Training: A Cross-Layer Analysis and Ferroelectric-Based Mitigation

**arXiv ID:** 2608.24637 | [PDF](https://arxiv.org/pdf/2608.24637v1)

**作者:** Seongwon Yoon `[一作]` (Georgia Institute of Technology), Shimeng Yu `[通讯]` (Georgia Institute of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对 wafer‑scale 光学互连在 MoE 训练中的性能进行交叉层面分析，结合工作负载分析、网络仿真、光学设备仿真和瞬态热仿真，量化热调节停顿对通信与整体训练时间的影响。

**💡 创新点**

创新点在于：① 将工作负载产生的热波动直接映射到光学谐振器的频率偏移，通过完整的热‑控制‑通信链条得到精确的停顿时长；② 提出一种基于铁电 LiNbO3/HZO 结构的非易失性电光调谐方案，消除持续热调节需求，显著降低调谐延迟。

**🔧 技术方法**

使用技术包括 HT‑Sim（网络级仿真）、Ansys（瞬态热仿真）、TCAD / Lumerical（光学模仿）、DWDM 线波导网络、微环谐振器（MRR）与铁电电光调谐器。

**📊 数据集**

评估数据集为 Mixtral 8×7B、Qwen‑MoE 14.3B 与 LLaMA‑MoE 6.7B 三个 MoE 模型，采用 NVIDIA H100 GPU 进行工作负载剖析，并在 256/512/128 GPU 的大规模配置上进行仿真。

**📈 对比分析**

通过将热调节停顿注入网络仿真，并与无停顿基线、NVLink、fat‑tree 及 MixNet 等传统拓扑做对比。结果显示热调节导致 Mixtral 迭代时间提升 2.7×，Qwen 3.8×，LLaMA 3.3×；铁电方案几乎消除停顿，恢复或超过传统拓扑的性能；停顿消除后 wafer‑scale 光学网络在大多数模型上实现 1.7× 的速度提升。

**⚠️ 局限性**

局限性包括：① 网络仿真仅使用四层代理，未完成全层规模仿真；② 假设的热控制速率、谐振器 Q 值与容忍偏移对结果有影响，仍需更细粒度验证；③ 铁电调谐器的长期耐久性虽基于现有材料数据估算，但在实际训练周期中的可靠性需进一步实验验证。

---

## 482. Asynchronous Verifiable Information Dispersal with Low Space and Communication Complexity

**arXiv ID:** 2608.24636 | [PDF](https://arxiv.org/pdf/2608.24636v1)

**作者:** Thomas Locher `[一作]` (DFINITY), Yvonne-Anne Pignolet `[通讯]` (StableClear)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种新的异步可验证信息分散（AVID）协议，兼顾分散、存储、检索与恢复四项关键操作的低空间与低通信复杂度。

**💡 创新点**

创新点在于：① 采用二维矩阵编码和定制化的分散算法，使节点在不下载整个数据的情况下完成分散与恢复；② 通过可证明的承诺机制保证数据完整性；③ 设计了可调节的协议变体，允许在空间/检索/恢复成本之间进行权衡。

**🔧 技术方法**

核心技术包括：可验证信息分散、线性全域错误纠正码（MDS）、向量承诺（Merkle树或KZG）、可靠广播、分布式恢复协议。

**📊 数据集**

本文为理论研究，未使用实测数据集，所有复杂度和性能指标均以理论分析与比较表格给出。

**📈 对比分析**

与现有方案相比：主协议在空间开销上为 3 倍数据量，检索通信复杂度为 1.5|m|，恢复通信复杂度约为 4.5|m|/n；相比 Red Stuff 的 4.5|m| 存储、7.5|m|/n 恢复，主协议在恢复上减少 40%，在分散上减少 14%；与 Alhaddad 等 3|m| 方案相比，检索与恢复更高但整体成本更均衡。

**⚠️ 局限性**

局限性包括：① 仍需签名与承诺验证的计算开销；② 对于极高网络延迟的环境，可靠广播可能导致多轮通信；③ 协议假设已知静态网络拓扑，动态节点加入/离开时需额外处理；④ 变体中空间效率提高会导致检索和恢复成本显著上升。

---

## 483. The Annotation Bottleneck in Persian Text NLP: Persian as an Annotation-Scarce Language

**arXiv ID:** 2608.24698 | [PDF](https://arxiv.org/pdf/2608.24698v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 484. HARQ-CC-Aided Slow Fluid Antenna Multiple Access with Highly Correlated Ports: An LST-Based Performance Analysis

**arXiv ID:** 2608.24614 | [PDF](https://arxiv.org/pdf/2608.24614v1)

**作者:** Sixu Han `[一作]` (University College London), Hanjiang Hong `[通讯]` (University College London)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了干扰受限下的下行HARQ-CC辅助慢速流体天线多址（sFAMA）系统的性能，并给出了稠密端口、高相关性情形下可计算的单轮SIR分布与累积SIR的闭式/数值求解方法。

**💡 创新点**

提出两种高相关性近似（Marcum‑Q核与步阈值）并进行端点校正，构建可在SIR域卷积或LST域逆变换下计算的可接受CDF；实现对FAS、BR‑AS、FPA三种接收机的统一分析；揭示端口密度饱和和HARQ回合数对可靠性与吞吐量的影响。

**🔧 技术方法**

使用空间块相关模型、Marcum‑Q函数近似、Gauss–Laguerre数值积分、Stieltjes卷积、Laplace–Stieltjes变换、Gaver–Wynn–Rho逆变换，以及端点校正与单步阈值修正技术。

**📊 数据集**

采用Jakes全空间相关模型生成的仿真数据，对不同端口数K、用户数U、HARQ回合数C和SIR阈值γ_th等参数下的性能进行比较。

**📈 对比分析**

通过SIR域卷积与LST域求解两种方式的结果与Monte Carlo仿真比较，验证一致性；FAS在相同条件下比BR‑AS和FPA拥有更低的掉线概率、更少的平均传输次数和更高的payload吞吐；端口密度增大至K≥64后性能趋于饱和，额外HARQ回合收益递减。

**⚠️ 局限性**

近似仅在端口高度相关（μ>0.97）时收敛；Marcum‑Q核忽略残差项ℛ_m；步阈值近似在低SIR区间可能失去单调性；对块相关模型的匹配依赖于μ与块划分，若真实相关性更复杂可能导致误差；仅针对下行单用户包传输，未考虑多接收机或频分多路复用。

---

## 485. Comparison Invariants for Verifying Control Invariance

**arXiv ID:** 2608.24598 | [PDF](https://arxiv.org/pdf/2608.24598v1)

**作者:** Promit Panja `[一作]` (Karlsruhe Institute of Technology), André Platzer `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出了在微分动态逻辑中对控制不变性的形式化验证框架，并给出了对应的公理和推理规则；

**💡 创新点**

首次统一了标量/向量比较原理与比较不变性，并将控制屏障函数、达布罗伊斯不变性及微分不变性等特殊实例归纳为比较不变性的子类，实现了形式化安全验证的整体化与可重用性；

**🔧 技术方法**

采用微分动态逻辑、微分幽灵（differential ghosts）、比较函数（class 𝒢/𝒦）、实数一阶算术决策程序等技术，构建了可推导的公理体系；

**📊 数据集**

论文为理论工作，未使用任何实验数据集；

**📈 对比分析**

通过把不变性问题转化为对Lie导数的实数不等式（可在一阶实数逻辑中判定）来实现比较，提供了比数值半正定规划更严格的符号化证明方式，虽未给出具体实验性能指标，但理论上保证了推理的可判定性与逻辑一致性；

**⚠️ 局限性**

局限性在于需存在满足 class 𝒢/𝒦 条件的比较函数，且对控制器合成的支持有限；推理规则在形式上完整但实现自动化仍待后续工作，且对复杂非线性系统的可扩展性尚未评估。

---

## 486. Delayed Optimizer-State Transport Shapes Short-Horizon Training Decisions

**arXiv ID:** 2608.24593 | [PDF](https://arxiv.org/pdf/2608.24593v1)

**作者:** Jinhui Guo `[一作]` `[通讯]` (Beihang University), Jinhui Guo (Beihang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了自适应优化器的动量状态在有限时间决策中的传递效应，基于AdamW的梯度历史对短期训练决策的影响进行差分分析，并提出基于全传输的评分方法用于调度选择。

**💡 创新点**

首次将全传输源-传输-读取框架应用于训练轨迹，证明动量状态是导致时间调度重排序的主要通道，并提供从局部线性响应到全局决策的可解释机制。

**🔧 技术方法**

对AdamW更新进行一阶微分，构造源-传输-读取分解，使用反向对偶求全时刻梯度，结合局部梯度裁剪、分支屏蔽与离散调度空间的多窗子采样。

**📊 数据集**

以Byte-level Transformer对Math与Code两大域的数据集（DeepMind Mathematics Dataset与Python标准库）和二维Ising模型的Wolff蒙特卡罗样本为实验基准。

**📈 对比分析**

采用四种对照（中性、VGA、即刻梯度、全传输）在12条未见历史上对比token-disjoint损失，发现全传输在10/12历史中优于即刻梯度，平均收益4.7e-4，且在大规模候选库上可通过全传输筛选提升完整搜索召回率至接近100%。

**⚠️ 局限性**

仅验证8步窗口、AdamW和固定未来批次路径，未涵盖未知路径、长周期控制、不同优化器（如SGD）以及更大模型或更长决策期的泛化，且全传输计算成本较高。

---

## 487. On-policy Distillation with Verifiable Reward

**arXiv ID:** 2608.24696 | [PDF](https://arxiv.org/pdf/2608.24696v1)

**作者:** Wenze Lin `[一作]` (Tsinghua University), Gao Huang `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为 OPDVR 的方法，通过在采样 token 上应用 ReLU 门，将 On‑policy Distillation 与 Reinforcement Learning with Verifiable Rewards 结合，无需额外超参数即可实现更高效的推理模型微调。

**💡 创新点**

创新点在于将 OPD 重新表述为 RLVR 的隐式奖励，并用 ReLU 门消除错误奖励，使得 token 更新始终与轨迹正确性一致，从而构建了一种无超参数、可直接与任意策略梯度算法结合的一体化框架。

**🔧 技术方法**

技术细节包括：对采样 token 的 KL 梯度进行 ReLU 门调制、将 OPD 转化为 RLVR 的形式、结合 GRPO 等策略梯度算法实现 OPDVR 与 GRPD，并在推理任务中进行高效微调。

**📊 数据集**

使用的主要数据集包括 DeepMath（训练集）以及六大推理基准 AIME24、AIME25、AMC、MATH500、Minerva、OlympiadBench 进行评估。

**📈 对比分析**

实验对比了标准 OPD、Top‑64 OPD、GRPO 等方法，结果显示 OPDVR 在同构与跨构架设置下均显著提升准确率，部分基准甚至超过教师模型，提升幅度通常为 2–5 分。

**⚠️ 局限性**

局限性：仍依赖教师模型的质量；ReLU 门的比例固定，可能在极端情况（如教师和学生概率差距极大）下表现不佳；目前仅在数学推理任务上验证，推广到其他领域需进一步研究。

---

## 488. Maia 200: A Software Defined Dataflow System for Large-scale AI Acceleration

**arXiv ID:** 2608.24664 | [PDF](https://arxiv.org/pdf/2608.24664v1)

**作者:** Sherry Xu `[一作]` (Microsoft Corporation), Torsten Hoefler `[通讯]` (Microsoft Corporation)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了Maia 200 AI加速器，基于软件定义局部访问数据流（SDLA）架构，专门为大规模LLM推理实现高性能与低能耗。

**💡 创新点**

创新点在于将数据流编程显式化、局部化内存访问以及控制与数据路径分离，形成一种全新的数据流微架构模型，并引入新的分类体系与可编程DMA/同步单元。

**🔧 技术方法**

采用TSMC 3nm工艺、CoWoS‑S三维封装、HBM3e高速内存、分层NoC、RDMA/以太网网络、动态电压频率调节（DVFS）、自定义数据流ISA与控制层C/C++编程接口。

**📊 数据集**

主要使用的测试集是公开的LLM模型Qwen 2.5 7B，在其推理工作负载（含自注意力、前馈、SwiGLU激活等）进行评测。

**📈 对比分析**

通过与GPU/CPU基准、Roofline分析、Allgather、矩阵乘法吞吐率等多维度比较，Maia 200在FP4/F P8模式下实现10145/5072 Tflop/s单芯片性能，集群级可达62 exaFLOP/s；单芯片推理速率达2434 tokens/s，约为理论最大70%。

**⚠️ 局限性**

局限性包括对推理工作负载的专门化（不适用于大规模训练）、对高级编程模型的依赖（需专家级ninja编程以充分发挥潜能）、以及在跨平台与多种架构间缺乏直接可比性（需要更多标准化基准）。

---

## 489. The Invisible Editorial Layer: Formalizing Undisclosed Inference-Time Steering, Probability Placement, and the Attribution Problem in Deployed Language Models

**arXiv ID:** 2608.24662 | [PDF](https://arxiv.org/pdf/2608.24662v1)

**作者:** Augusto Camargo `[一作]` `[通讯]` (Bluecore Consulting), Augusto Camargo (Bluecore Consulting)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并系统化了推理时框架偏差（Inference‑time Framing Bias）的概念，阐明了部署层干预对模型输出的可观测影响，并给出了相应的归因与治理框架；

**💡 创新点**

核心创新在于将推理时的logit‑级干预正式化为部署层的“隐性控制”，引入了Inference Attribution Problem、Probability Placement以及Inference Policy Transparency等新概念，并将现有的受控生成与水印技术与治理议题对接；

**🔧 技术方法**

主要技术手段包括对logit‑级干预的数学建模（加性偏置、概率分布变换）、对受控生成方法（PPLM、GeDi、DExperts、FUDGE、Activation Engineering、直接logit干预）的综述、以及黑盒审计与分布差异检测的概念化；

**📊 数据集**

文中未引入新的数据集，主要以已有的受控生成与水印技术为参考，并在理论层面讨论不同框架下的概率分布；

**📈 对比分析**

由于缺乏实验验证，本文并未给出性能评估；其提出的实证研究议程建议通过黑盒输出评估语义框架偏移，但目前尚无基准或度量；

**⚠️ 局限性**

局限性包括：缺乏实证数据与实验验证；推理时框架偏差的检测难度高，黑盒审计仍面临信息不完整的问题；监管框架对这类隐蔽干预的适用性尚不明晰；未提供具体实现细节或案例分析。

---

## 490. A Literate Programming Environment for Human and Machine Agents

**arXiv ID:** 2608.24644 | [PDF](https://arxiv.org/pdf/2608.24644v1)

**作者:** Adam T. Burke `[一作]` `[通讯]` (Queensland University of Technology), Adam T. Burke (Queensland University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为notlob的可文学化编程环境，将自然语言与可执行代码、测试等混合在同一文件中，并提供名称图谱支持跨文件、跨语言的导航与一致性检查

**💡 创新点**

将文学化编程与大型语言模型（LLM）结合，构建可被LLM编程代理直接利用的可执行语法与上下文窗口友好的文件结构，以及可被机器和人类共同读取的名称图谱

**🔧 技术方法**

使用Python实现的Lark语法解析器、命令行工具、JSON/RDF导出功能，并实现Haskell、Python、TypeScript三种绑定，支持语法高亮、单元/属性测试与文档渲染

**📊 数据集**

未使用公开数据集，主要通过自定义示例程序（斐波那契、罗马数字、Petri网游戏、Pleiades天文计算等）展示实现效果

**📈 对比分析**

未开展系统性能或对比实验，文中仅提供功能演示与案例说明，缺乏量化评估

**⚠️ 局限性**

缺乏大规模实验验证，LLM代理在实际使用中可能忽视声明式测试与名称图谱信息，需人工干预保证一致性；当前实现仅支持有限语言绑定且未提供完整的用户体验评估

---

## 491. Quantifying the Relationship Between Team Dysfunctions and Performance in Capstone Projects

**arXiv ID:** 2608.24634 | [PDF](https://arxiv.org/pdf/2608.24634v1)

**作者:** Luciano Pereira Soares `[一作]` (Insper), Rafael Corsi Ferrão `[通讯]` (Insper)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了团队功能失调与工程Capstone项目绩效的关系，并用Lencioni的五功能失调模型量化团队动态。

**💡 创新点**

创新之处在于将Lencioni模型应用于项目式学习环境，实证检验其对多维绩效指标的解释力。

**🔧 技术方法**

使用问卷调查、团队层级聚合、Pearson与Spearman相关分析以及线性回归等统计技术。

**📊 数据集**

数据来自Insper 2025学年40支Capstone团队的自评问卷和项目成绩（技术、组织、沟通、团队合作、设计、创业六维）。

**📈 对比分析**

通过相关系数和回归比较发现，‘结果关注度’与平均成绩相关显著（r≈0.32, p=0.04），其余维度相关弱且不显著，整体关联不强。

**⚠️ 局限性**

局限包括样本单一、得分集中导致变异不足、仅使用相关性无法说明因果关系、缺乏多机构或纵向验证。

---

## 492. Fair Allocation with Optional Selling

**arXiv ID:** 2608.24600 | [PDF](https://arxiv.org/pdf/2608.24600v1)

**作者:** Uriel Feige `[一作]` (Weizmann Institute of Science), Yotam Gafni `[通讯]` (Weizmann Institute of Science)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文研究了在存在可选出售价格的情况下，如何在公平分配不可分割物品时同时考虑出售和分配两种决策，并提出了多种公平性近似算法；

**💡 创新点**

创新点在于将传统的 MMS、TPS 等基于份额的公平性概念与可出售物品相结合，定义了 SEFX、SEF1 等新的基于厌恶的公平性，并证明了在此设置下可实现的最佳近似比值（如 2/3-MMS、n/(2n−1)-TPS）以及构造相应多项式/伪多项式时间算法；

**🔧 技术方法**

主要技术包括可行性分析、可出售物品的最大最小份额 (MMS) 定义、截断比例份额 (TPS) 计算、Cut&Give、moving-knife 与 bag-filling 机制、匹配与 Hall 定理、canonical partition 结构、以及伪多项式/完全多项式时间近似方案（PTAS/FPTAS）；

**📊 数据集**

论文为理论性工作，未使用公开数据集，所有结果均基于数学证明与构造实例；

**📈 对比分析**

通过理论证明展示了各类近似比例（如 2/3-MMS、3/4-MMS、n/(2n−1)-TPS）并给出相应算法实现；相较于无出售情形，部分近似比率保持不变，但在 3 代理时出现更大的可选出售导致的近似下界；

**⚠️ 局限性**

局限性包括：仅针对可加且非负的个人偏好，且只考虑不可分割物品；无法保证 3 代理全 MMS；可出售的决策使得理论分析更复杂；在实际应用中需进一步验证算法的实用性与对非可加偏好的适用性。

---

## 493. Online and Incremental Fractional Vertex Cover on Trees

**arXiv ID:** 2608.24630 | [PDF](https://arxiv.org/pdf/2608.24630v1)

**作者:** Júlia Baligács `[一作]` (University of Oxford), Anna Zych-Pawlewicz `[通讯]` (University of Warsaw)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究树图上分数顶点覆盖问题，分别给出了在线边到达模型和增量模型的算法；在线模型提出RGB算法，取得11/6≈1.83的竞争比；增量模型给出1.5-竞争比的半整数算法，并证明此比值是最优；同时给出对整数增量算法的下界，指出局部规则无法达到3/2的竞争比；

**💡 创新点**

主要创新点在于：①针对树图的在线分数顶点覆盖提供了比之前2.0更好的11/6竞争比算法；②在增量模型下首次给出1.5-竞争比的半整数算法，并证明其最优；③通过线性规划对下界进行精确分析，证明任何增量算法至少需要3/2竞争比；④揭示整数增量问题更难，局部规则无法达到3/2竞争比。

**🔧 技术方法**

主要技术包括：
- RGB算法基于三色（红、绿、蓝）边的分配，利用树的父子结构设计权重更新规则；
- 竞争比证明采用弱对偶性，将在线分数顶点覆盖转化为分数匹配问题，并构造合适的匹配来下界化；
- 对增量算法的下界利用层级层次（层i层i+1）约束，构建线性规划并求其对偶，以得到3/2的极限；
- 对半整数增量算法通过预算分配和局部规则证明竞争比。

**📊 数据集**

本文未使用任何公开数据集，全部为理论分析与算法设计，实验仅在理论模型中进行。

**📈 对比分析**

与已有工作比较：
- 对在线模型，之前仅有在顶点到达模型下的1.901竞争比，本文在更一般的边到达模型下取得更优的11/6竞争比；
- 对增量模型，先前没有已知结果，本文给出1.5-竞争比，并证明为最优；
- 证明下界与竞争比一致，说明算法已达到理论极限。

**⚠️ 局限性**

局限性包括：
- 对整数增量顶点覆盖问题未给出竞争比，且局部规则被证明无法达到3/2，完整解法仍开放；
- RGB算法和半整数增量算法特定于树图，尚未扩展到更一般图结构；
- 对更高竞争比的进一步优化仍是未来研究方向。

---

## 494. EviGraph: Towards Verifiable Evidence Construction for Information-Seeking Agents

**arXiv ID:** 2608.24667 | [PDF](https://arxiv.org/pdf/2608.24667v1)

**作者:** Jiashun Chen `[一作]` (Tencent Inc.), Wenhui Que `[通讯]` (Tencent Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于证据图的代理式搜索框架EviGraph，拆分搜索与证据记录两角色，并通过强化学习直接监督证据构建。

**💡 创新点**

创新点在于将搜索与证据记录解耦为共享策略的两角色，利用span锚定的证据图提供稠密过程奖励，并通过结构验证和惩罚机制抑制奖励作弊。

**🔧 技术方法**

技术包括深度搜索框架、冻结证据验证器、结构验证器、共享策略的GRPO强化学习、span锚定与约束标签的图结构。

**📊 数据集**

使用BrowseComp-Plus、BrowseComp、GAIA、XBench、LiveVQA等公开Web搜索与多模态搜索数据集。

**📈 对比分析**

与单角色ReAct或无RL的对比实验表明，在匹配预算下，双角色+RL的Qwen3-8B在BrowseComp-Plus达35.9%准确率，LiveVQA 78.0%，显著高于基线并减少生成token。

**⚠️ 局限性**

局限在于仍需检索候选文档、处理复杂多约束场景、对抗性作弊风险、以及对大型模型与多模态的可扩展性待验证。

---

## 495. Parason: Revealing Subtask and Trial Parallelism in LLM Reasoning

**arXiv ID:** 2608.24658 | [PDF](https://arxiv.org/pdf/2608.24658v1)

**作者:** Zhengyang Zhang `[一作]` (Tsinghua University), Ligeng Zhu `[通讯]` (Nvidia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Parason 框架，将大语言模型的串行推理拆分为子任务并行与试验并行两种模式，以实现推理加速。

**💡 创新点**

创新点在于：① 形成子任务并行与试验并行的语义并行分类；② 用上下文无关文法把推理轨迹转化为可执行的并行结构；③ 通过并行感知群组相对策略优化（PA‑GRPO）在训练中同时优化准确率、时延与并行比例。

**🔧 技术方法**

技术手段包括：上下文无关文法（CFG）标记；并行感知策略优化（PA‑GRPO）；工具调用执行与 SGLang 等推理引擎的集成；强化学习与监督微调结合。

**📊 数据集**

使用的主要数据集：ThreadWeaver 的 964 条 Qwen3‑8B 推理轨迹、Gemini‑3‑Flash 标签、Polaris‑53k 复杂推理问题；在 AIME24、AIME25、AMC、Math500、Minerva Math 等数学推理基准上评测。

**📈 对比分析**

与传统 SFT、现有并行系统（如 ThreadWeaver、Multiverse 等）比较，Parason 在 8B 模型上平均精度达到 84.7%，相对加速约 1.7×，在难度更高的题目下仍保持 1.5–1.6× 的 wall‑clock 加速。

**⚠️ 局限性**

局限性包括：对复杂混合依赖的推理轨迹建模尚不完善；实验仅在 8B 模型上验证，扩展至更大模型或更复杂任务仍需探索；需要推理引擎支持工具调用等实现细节。

---

## 496. Aging of Prompt Engineering Techniques Across LLM Versions

**arXiv ID:** 2608.24641 | [PDF](https://arxiv.org/pdf/2608.24641v1)

**作者:** Anastasiia Rudyk `[一作]` (University of Rostock), Regina Hebig `[通讯]` (University of Rostock)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对五种提示工程技术在不同 LLM 版本上的效果进行实验评估。

**💡 创新点**

系统研究提示技术在 GPT、Qwen、Mistral 代际之间的“老化”现象，并提出模型族依赖的适应性提示策略。

**🔧 技术方法**

采用零样本、少样本、链式思维、对比链式思维和程序思维等 PET，并使用 pass@k 评估函数级代码生成。

**📊 数据集**

使用 CodePromptEval 数据集的 218 条 Python 函数任务，生成 19,620 条代码样本。

**📈 对比分析**

通过对六款指令调优模型的 pass@1/2/3 进行对比，发现 GPT 新版对结构化提示几乎无益，Qwen 对少样本/对比链式提示仍显著受益，Mistral 介于两者之间。

**⚠️ 局限性**

仅覆盖 Python 函数级任务，未评估代码质量、安全性等多维指标，且模型族间规模差异和提示语法差异可能影响结果。

---

## 497. Causal Modelling of Support Interventions for Student Competency Assessment

**arXiv ID:** 2608.24632 | [PDF](https://arxiv.org/pdf/2608.24632v1)

**作者:** Francesca Mangili `[一作]` (Scuola Universitaria Professionale della Svizzera italiana), Rafael Cabañas `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

采用专家推断的结构因果模型（SCM）对小学算法能力测评进行构建，并利用该模型实现对学生能力的评估与干预效果的因果推断。

**💡 创新点**

创新点在于将传统的关联式心理测量方法（如项反应理论、贝叶斯网络）转化为结构因果框架，显式建模干预（如提示）和反事实推理，从而支持基于因果效应的个性化干预决策。

**🔧 技术方法**

技术包括：结构方程式专家 elicitation、SCM/PSCM 与 Bayesian network 的结合、因果 EM 反向传播求解兼容的完全 SCM、Twin network 计算反事实概率、以及利用 bcause 软件进行精确推断。

**📊 数据集**

使用了109名小学学生的 CAT（Cross-Array Task）测评数据，包含 12 个算法问题、对应提示、运气与偏差变量。

**📈 对比分析**

与从数据直接学习的贝叶斯网络进行对比：交叉验证下，BN 的 log‑likelihood 为 -277±19.1，预测准确率 0.84±0.02（答题）和 0.84±0.04（提示）；SCM 的 log‑likelihood 为 -287±18.5，预测准确率 0.77±0.02（答题）和 0.82±0.04（提示），显示因果模型在预测上略逊，但提供了更具可解释性和因果推断能力。

**⚠️ 局限性**

局限性包括：样本量有限、模型结构与参数尚未经过充分验证、相对预测性能较低、实现仅支持精确贝叶斯网络推断（对大规模技能集易受限）、未涵盖动态学习过程、缺乏真实干预数据用于验证因果估计。

---

## 498. Beyond Semantic Accuracy: Consequence-Aware Evaluation for Safety-Critical Language Understanding

**arXiv ID:** 2608.24621 | [PDF](https://arxiv.org/pdf/2608.24621v1)

**作者:** Yujing Chang `[一作]` (Nanyang Technological University), Sameer Alam `[通讯]` (Nanyang Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了语言模型在高风险任务（空中交通管制）中的可靠性，提出了基于后果的评估框架并构建了诊断性ATC数据集。

**💡 创新点**

创新点在于量化语义错误与操作后果之间的差距，并设计了非补偿的几何评分与降级指标，以更贴近安全评估。

**🔧 技术方法**

使用了零样本提示、Few-shot、Full‑aligned 提示以及 LoRA 风险加权微调的技术。

**📊 数据集**

利用由真实空管员验证的控制塔通信语料，包含结构理解与读回安全判定两任务共1500+实例。

**📈 对比分析**

与传统 NER‑F1、SLU‑F1 等语义指标对比，发现传统指标高估可靠性，后果感知指标（AR‑Geo、DDR/WDS）更能反映风险，模型在风险意识微调后性能提升但仍未完全关闭差距。

**⚠️ 局限性**

局限在于仅针对 ATC 领域，读回扰动为人为构造，模型规模与数据量有限，无法直接用于真实空管部署。

---

## 499. Lexicographic Social Ranking on Monotonic Coalitional Rankings

**arXiv ID:** 2608.24596 | [PDF](https://arxiv.org/pdf/2608.24596v1)

**作者:** Felix Fritz `[一作]` (Université Paris-Dauphine), Stefano Moretti `[通讯]` (Université Paris-Dauphine)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在单调联盟排名域内，利用包含-排除原理推导了lex-cel和L1评分的组合公式，并对其计算复杂度进行优化；

**💡 创新点**

创新点在于只需最小获胜联盟集合即可计算两种社交排序方案的分数，构建了通用的组合式框架并探讨了单调性对两方案一致性的影响；

**🔧 技术方法**

主要技术包括组合数学（包含-排除原理、二项式恒等式）、复杂度分析与优化方法（最小包含集约化、差值计算）；

**📊 数据集**

使用随机生成的单调联盟排名数据集（n=5,6的全排列），无公开真实数据集；

**📈 对比分析**

通过模拟比较两方案在随机排名下的相等率、分歧率（分支与逆转），发现差异率低于10%，但随n增大略升高；

**⚠️ 局限性**

局限在于最坏情况计算仍为指数级（O(2^{n⌊n/2⌋})），当最小获胜联盟数量大时不可行，且未处理不完整单调排名的情况。

---

## 500. Single State Update Predictive Coding training for Time Series Forecasting and Anomaly Detection

**arXiv ID:** 2608.24697 | [PDF](https://arxiv.org/pdf/2608.24697v1)

**作者:** Matteo Cardoni `[一作]` (Ghent University—imec), Sam Leroux `[通讯]` (Ghent University—imec)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过将生成式PCN与编码式PCN并行训练，实现了不依赖顺序误差传播的预测编码网络更新。

**💡 创新点**

创新点在于引入Guided PC训练，将两套PC网络通过激活匹配并行更新，消除传统PC的层级序列瓶颈。

**🔧 技术方法**

使用预测编码框架、内部能量与指导能量最小化、单步状态更新和并行权重更新等技术。

**📊 数据集**

在模拟MNIST数字在黑背景下移动的时序数据集上进行实验。

**📈 对比分析**

与传统Vanilla PC比较，Guided PC在连续在线学习中能更快恢复异常后新正常模式，异常评分更平稳、误差更小。

**⚠️ 局限性**

局限在于仅在简化的MNIST移动序列上验证，未在更复杂的现实时序数据或大规模网络上评估。

---

## 501. Data Leakage Inflates Generalizability of Power Outage Prediction Models

**arXiv ID:** 2608.24665 | [PDF](https://arxiv.org/pdf/2608.24665v1)

**作者:** Yamil Essus `[一作]` (University of Toronto), Benjamin Rachunok `[通讯]` (North Carolina State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

评估了电力停电预测模型在不同空间、时间和事件条件下的泛化能力，系统比较了随机拆分、留一州和留一事件三种评估策略。

**💡 创新点**

创新点在于：①提出并使用留一州/留一事件评估框架来揭示随机拆分带来的数据泄漏；②将基于大规模天气预训练模型（Prithvi WxC）的嵌入与传统ERA5特征对比，检验其对泛化的影响；③对目标变量（绝对停电人数 vs 相对比例）对模型性能的影响进行量化。

**🔧 技术方法**

采用XGBoost回归模型，利用随机拆分、空间留一州、时间留一事件三种训练/测试划分；使用Prithvi WxC预训练模型提取2560维嵌入，结合ERA5气象变量和NLCD树密度静态特征。

**📊 数据集**

数据集包括美国东海岸县级停电记录（PowerOutage.us）、ERA5 再分析气象数据、MERRA-2 气象数据（用于Prithvi WxC嵌入）以及NLCD树密度。

**📈 对比分析**

比较方法：计算R²、MAE并与零模型（预测均值）对比。随机拆分下R²≈0.45（相对指标）/0.33（绝对指标），MAE≈0.06/1800。留一州/留一事件时MAE显著升高（相对约0.06/0.07，绝对约2,300/2,700），仅飓风事件下相对指标略有提升。

**⚠️ 局限性**

限制：①模型在空间/时间外部泛化差，绝对指标几乎不优于零模型；②Prithvi嵌入提升有限，且高维特征存储和计算成本高；③再分析气象数据对极端事件的再现不够，导致模型对极端情形预测不足；④数据稀疏和极端事件数量有限，易导致过拟合与迁移能力差。

---

## 502. Code-Domain Grouped Index Modulation for Spectrally Efficient Spread-Spectrum Communications

**arXiv ID:** 2608.24642 | [PDF](https://arxiv.org/pdf/2608.24642v1)

**作者:** Peng Zhang `[一作]` (Southeast University), Zaichen Zhang `[通讯]` (Southeast University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出代码域分组索引调制（CGIM）方案，利用正交扩频码分组实现组内独立解扩和检测，并设计组内最大似然（ML）与贪婪检测（GD）算法。

**💡 创新点**

通过把扩频码库分成多个组，每组内部独立解扩并可并行检测，从而在不增加硬件复杂度的前提下有效扩大代码索引分支数；证明组内ML与整体ML等价，且在等能码与方形QAM下GD实现ML等效。

**🔧 技术方法**

采用正交码分组、组内ML检测、PAM分离、贪婪检测、误码概率的联合似然分析、Rayleigh 与 Nakagami‑m 衰落下的概率母函数、以及 Gaussian Q‑函数近似。

**📊 数据集**

通过 Monte Carlo 仿真，生成 Rayleigh、Nakagami‑m（m=2）及 AWGN 信道数据，没有使用公开数据集。

**📈 对比分析**

在相同传输速率下与 GCIM、QCIM 进行 BER 比较。结果显示 CGIM 在 Rayleigh、Nakagami‑m 与 AWGN 三种信道中，分别比 QCIM 和 GCIM 低 3–15 dB 的 E_b/N_0 需求，ML 与 GD 的误码曲线几乎重合，表明低复杂度 GD 仍能保持 ML 等效性能。

**⚠️ 局限性**

仍依赖于正交码分组的可行性与码本大小限制，分组设计对性能有较大影响；在低 SNR 区域误码概率上界不够紧凑；未考虑多用户干扰、非理想信道或多天线多用户情景。

---

## 503. Conditional GraphGANFed: Optimizing Graph-Structured Molecule Generation in Federated Generative Adversarial Networks

**arXiv ID:** 2608.24610 | [PDF](https://arxiv.org/pdf/2608.24610v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 504. Towards Reliable AI-Based Histological Staining: A Systematic Study of Scaling and Uncertainty in Unpaired Generative Models

**arXiv ID:** 2608.24626 | [PDF](https://arxiv.org/pdf/2608.24626v1)

**作者:** Qasim Siddiqui `[一作]` (University of Leipzig), Stefan Hoehme `[通讯]` (University of Leipzig)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估并基准六种无监督图像对图像翻译模型在将H&E染色转换为Sirius Red（SR）上的性能，并构建首个公开的配对鼠标肝脏数据集。

**💡 创新点**

在同一框架下系统比较不同架构（GAN、扩散）以及模型规模和数据量对感知质量、任务精度和深度集成不确定性三维指标的影响；首次将集成方差作为无监督染色翻译的不确定性量化。

**🔧 技术方法**

采用CycleGAN、UNIT、MUNIT、DCLGAN、UVCGAN、CycleDiffusion六种无监督模型；利用深度集成（10个独立种子）产生像素级方差；使用CPA MAE、FID、LPIPS、patch‑SSIM和读者研究等评估。

**📊 数据集**

公开的70张鼠标肝脏全切片图像（35 H&E + 35 SR），共30只动物训练集（约201k H&E、196k SR图块），5只动物测试集，数据按动物分割以防泄露。

**📈 对比分析**

通过54种模型容量（≈10、≈50、≈100M参数）与数据量（25%、50%、100%）的全因子实验，评估四项指标；结果显示GAN中DCLGAN小模型在任务误差和不确定性上均表现最佳，CycleGAN中等容量在完整数据下CPA MAE最低，扩散模型均值低但IQR大，表明其一致性与波动性不同。

**⚠️ 局限性**

数据仅来自单一鼠标肝脏疾病模型和实验室；测试样本量有限（5只动物），难以实现统计稳健；仅评估不确定性方差，未考虑测量噪声或生物异质性；读者研究样本小，缺乏全面评判。

---

## 505. VIP: Variation-based Iterative-learning Planning for Robotic Navigation

**arXiv ID:** 2608.24618 | [PDF](https://arxiv.org/pdf/2608.24618v1)

**作者:** Shuli Lv `[一作]` (Beihang University), Quan Quan `[通讯]` (Beihang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了变分式迭代学习规划（VIP）框架，直接在无限维函数空间中优化运动指令，实现单机器人路径跟踪与多机器人队形分布的统一规划；

**💡 创新点**

创新点包括：①以能量泛函统一不同规划任务的目标，避免轨迹参数化与长预测窗口带来的高维度问题；②通过函数级变分更新保持每次迭代计算复杂度线性O(n)，并可实现离线模型内或在线机器人内的无模型学习；③实现了在线无模型学习，通过历史能量曲线更新遍历速度，无需解析梯度或动力学模型；

**🔧 技术方法**

采用变分优化、Hamilton–Jacobi 逆向求解、积分算子、无模型迭代学习（model‑free ILC）、虚拟管道控制、核密度估计（KDE）等技术；

**📊 数据集**

在多种实验与仿真中验证：随机森林地图（Map1/Map2）、230m×200m×30m障碍环境、LiDAR‑基准四旋翼、升翼四旋翼、三机队列飞行；未使用公开数据集，而是自行生成随机障碍和虚拟管道；

**📈 对比分析**

与MPCC、最小抖动轨迹、采样/MPPI等传统方法比较，VIP 在行驶时间上可与MPCC持平或略优（1–2%），与多项式/最小抖动法差距可达6–7%；在计算时间上平均节省 47–88% 以上，尤其在轨迹点数增多时表现出明显线性复杂度优势；

**⚠️ 局限性**

局限性包括：①需要预先生成可行的空间轨迹或虚拟管道；②模型自由更新对能量测量噪声和环境不确定性敏感；③在动态环境或极端不确定情况下收敛速度与性能需进一步评估；

---

## 506. Quantization Effects on Bangla Language Understanding in Large Language Models: A Systematic Evaluation

**arXiv ID:** 2608.24615 | [PDF](https://arxiv.org/pdf/2608.24615v1)

**作者:** Ismail Hossain `[一作]` (Shahjalal University of Science and Technology), Mohammad Abdullah Al Mumin `[通讯]` (Shahjalal University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在孟加拉语自然语言理解基准上，对三大LLM家族（Qwen-2.5-7B、LLaMA-3.1-8B、GPT-OSS-20B）做后训练量化（GPTQ-Int8、GPTQ-Q8、GGUF-W8A16）实验，并在五个孟加拉语评测集上进行零样本准确率评估。

**💡 创新点**

首次系统比较多家族在低资源、多形态语言（孟加拉语）上的量化表现，发现量化对模型架构和量化方法敏感度高于语言本身，并揭示推理类任务比阅读理解类任务更易受量化影响。

**🔧 技术方法**

使用 GPTQ（基于二阶梯度校准的 INT8/INT8-Q8 量化）和 GGUF-W8A16 量化格式；零样本评估采用 EleutherAI 的评测工具，按候选答案 log‑likelihood 选取。

**📊 数据集**

评测集为 Bangla MMLU、CommonsenseQA-BN、OpenBookQA-BN、PIQA-BN、BoolQ-BN（共 5 组多选/二选、阅读理解任务）。

**📈 对比分析**

方法：对同一模型家族的全精度和量化版做逐基准准确率对比，计算绝对降幅 Δ 和相对降幅 Δ%；结果显示 GPTQ-Int8/ Q8 对 Qwen 与 LLaMA 的准确率损失 ≤1.5%，而 GPT-OSS 的 GGUF-W8A16 在推理类基准上降幅可达 57%，阅读理解基准受影响最小。

**⚠️ 局限性**

局限性：仅评估 Int8 级别量化；未包含 INT4、量化感知训练或生成任务；未测量推理延迟/内存占用；GPT-OSS 的 GGUF 校准集未公开，且实验未完全交叉模型家族、量化格式与规模，难以完全剥离这些因素的影响。

---

## 507. Gripper-aware Vision Language Action Models

**arXiv ID:** 2608.24603 | [PDF](https://arxiv.org/pdf/2608.24603v1)

**作者:** Hanyi Zhang `[一作]` (University of Liverpool), Baoru Huang `[通讯]` (University of Liverpool)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了多抓手感知的视觉语言动作模型（GVLA）和对应的多抓手演示数据集MiGA，解决了现有VLAs忽略抓手多样性的问题。

**💡 创新点**

创新点在于设计了多层级抓手tokenizer和双Mixture-of-Adapters，显式编码抓手类型与实例信息，并通过软提示和路由实现抓手特定策略学习与跨抓手迁移。

**🔧 技术方法**

技术包括软提示嵌入、多抓手tokenization、双MoA路由、流匹配动作损失、抓手预测辅助损失及负载平衡正则。

**📊 数据集**

使用了MiGA数据集，共103,000条演示，覆盖5种抓手、36个任务、真实+仿真场景，并提供自然语言描述与失败案例。

**📈 对比分析**

与传统两阶段抓取方法、GraspVLA、OpenVLA-OFT、π_0/π_0.5等基线对比，GVLA在四类任务的平均成功率从58%提升至66%，相对基线提升约7.6%。

**⚠️ 局限性**

局限性包括仿真环境对软/多DoF抓手运动建模不足、缺乏细粒度几何与接触建模导致抓手细化不足，以及抓手与视觉表示的部分耦合影响域迁移鲁棒性。

---

## 508. Comparative Assessment of Deep Learning Architectures for Underwater Subsurface Kelp Forest Segmentation with The Kelp-o-Tron

**arXiv ID:** 2608.24594 | [PDF](https://arxiv.org/pdf/2608.24594v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 509. Bandit Submodular Maximization under Matroid Constraints: Learning Compressed Exchange Policy

**arXiv ID:** 2608.24627 | [PDF](https://arxiv.org/pdf/2608.24627v1)

**作者:** Zongqi Wan `[一作]` (Great Bay University), Zhijie Zhang `[通讯]` (Fuzhou University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出了一种在一般基 matroid 约束下，在线对抗性 bandit 子模函数最大化的多项式时间算法，且实现子线性(1-1/e) 的期望 regret

**💡 创新点**

关键创新在于把基的 Poisson 随机 walk 转化为学习状态依赖的交换策略，再通过“balanced fractional exchanges”把指数级的策略混合压缩为单一分数基；此外设计了 leave-one-out 估计器，仅用一次可行价值查询即可获得无偏估计

**🔧 技术方法**

利用 Poisson 基 walk、Brualdi 基交换定理、子模函数的多线性扩展、离散在线学习中的 Exp4 与负熵投影、运输网络求解 Balanced Exchange、以及重要性采样估计损失

**📊 数据集**

本文未使用具体数据集，而是在理论上证明算法的 regret 上界；实验与数据集未列出

**📈 对比分析**

与之前在 uniform、partition matroid 上已知的 O(T^{2/3}) regret 相比，本文在任意 matroid 上也达成 O(n^{1/3}k^{2/3}T^{2/3}) 的子线性 regret，并首次给出子线性 regret 的下界和相应上界，证明了 1-1/e 的近似因子可保持

**⚠️ 局限性**

主要局限在于需要对基多项式时间求解平衡交换和熵投影；算法对参数选择（γ、δ、η、L）敏感，且在 n, k 远大于 T 时退化为固定基策略；对实际大规模基 matroid 可能计算开销仍然较高

---

## 510. Pushdown Model Checking Above the Cubic Bottleneck

**arXiv ID:** 2608.24601 | [PDF](https://arxiv.org/pdf/2608.24601v1)

**作者:** A. R. Balasubramanian `[一作]` (Max Planck Institute for Software Systems), Rupak Majumdar `[通讯]` (Max Planck Institute for Software Systems)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文给出针对“PDA与若干NFA交叉非空检测”问题的细粒度复杂度下限，证明若存在比已知O(n³k)（或O(n^ωk)）更快的算法，则会突破3k‑Clique等基准难题；

**💡 创新点**

创新点在于将Pushdown模型检测与3k‑Clique等基准问题关联，提出新的2NPDA(k)假设，并给出一整套线性时间归约构造，说明为何该问题难以改进；

**🔧 技术方法**

主要技术是细粒度复杂度理论、图形编码与计数gadget构造、PDA与NFA的产品构造以及多阶段逆向计数验证；

**📊 数据集**

本文为理论研究，未使用任何具体数据集；

**📈 对比分析**

实验比较未给出，理论上证明任何超越O(n³k)（或O(n^ωk)）的算法若存在将导致3k‑Clique可在O(n^{ωk/3-ϵ})时间内求解，故现有算法已近似最优；

**⚠️ 局限性**

限制在于结论是条件性的（依赖3k‑Clique及2NPDA(k)假设），且仅对组合式算法给出下界，无法直接覆盖所有可能的加速技术；

---

## 511. Taming foundation model with invariance-oriented pre-training for broad-spectrum EEG analysis across signal-level, brain-state, and brain-health tasks

**arXiv ID:** 2608.24597 | [PDF](https://arxiv.org/pdf/2608.24597v1)

**作者:** Yulong Dou `[一作]` (ShanghaiTech University), Dinggang Shen `[通讯]` (ShanghaiTech University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

构建并预训练一种名为 INCEPT 的 EEG 基础模型，用大规模未标注临床 EEG 数据学习可迁移的全脑表征，并在十个不同的 EEG 下游任务（信号质量评估、脑状态解码、脑功能评估）上进行线性探测与全微调评估。

**💡 创新点**

创新点在于：① 将局部上下文恢复与跨视图不变性学习相结合，强调“跨同一记录的全局稳定信息”，从而在保留个体生理结构的同时保持对状态/疾病差异的可辨别性；② 采用球面谐波编码实现电极空间连续嵌入，提升不同 montage 之间的迁移性；③ 通过动态空间时间采样产生多尺度视图，增强对 montage、通道数和时长变化的鲁棒性。

**🔧 技术方法**

技术手段包括：Transformer 交替结构（通道内与全局自注意力）、教师‑学生自监督对比学习（macro‑micro 与 macro‑masked 对齐）、掩码上下文恢复、KoLeo 正则、球面谐波电极嵌入、时频双分支特征编码、以及大规模无标签预训练。

**📊 数据集**

预训练数据：Temple University Hospital EEG Corpus（TUEG），约 11,000 小时 19 通道 10‑20 montage 的临床 EEG；下游评测数据共十个：TUAB（异常检测）、TUAR（四类伪影识别）、FACED/SEED‑V（情绪识别）、PhysioNet‑MI（运动想象）、ISRUC‑S1（睡眠分期）、Mumtaz2016/MentalArithmetic（抑郁/心理压力）、ADFTD（神经退行性疾病）、Siena（癫痫检测）。

**📈 对比分析**

对比方法包括：传统任务专用监督模型（EEGNet、ST‑Transformer、EEGConformer、SPaRCNet）和三种现有 EEG 基础模型（CBraMod、CSBrain、CodeBrain）。在 30 条评测指标（线性探测/全微调的多分类、二分类指标）中，INCEPT 在 26/30 条指标中排名第一，在 24/30 条指标中排名第二，且在多任务、多 montage 下表现更稳定、标准差更小。 Ablation 结果表明，仅使用掩码恢复或仅使用不变性学习都能提升性能，但两者结合效果最佳。

**⚠️ 局限性**

局限性：① 仍未对所学习的表征进行深入生理学解释；② 评估以公开数据集和离线标签为主，缺乏临床前瞻验证；③ 尽管数据量巨大，但不同 montage、采样率、参考方式的异质性仍可能影响迁移；④ 对噪声和异常的处理依赖预处理流程，可能导致信息损失。

---

## 512. Is Discrete Difficulty Sufficient? Leveraging Continuous Difficulty for Efficient Self-Consistency in LLMs

**arXiv ID:** 2608.24590 | [PDF](https://arxiv.org/pdf/2608.24590v1)

**作者:** Sihyeong Yeom `[一作]` (Konkuk University), Harksoo Kim `[通讯]` (Konkuk University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Flexible Self-Consistency（FSC），利用输出熵预测模型感知的难度，并按难度动态分配推理路径以实现高效推理

**💡 创新点**

创新点在于将难度视为连续信号，使用轻量线性探测器预测输出熵作为难度指标，从而实现比离散难度估计更细粒度的资源分配

**🔧 技术方法**

核心技术包括输出熵计算、线性探测器训练、熵驱动的采样预算调节，以及多数投票的Self‑Consistency推理

**📊 数据集**

在MATH、AMC、AIME、GPQA‑Diamond以及MMLU‑Pro等数学与通用推理数据集上进行实验

**📈 对比分析**

与SC、AC、ESC、DSC等基线比较，FSC在保持准确率不变的前提下，令token消耗降低最高达76.7%，在各模型规模与任务上表现最优

**⚠️ 局限性**

局限性包括：仅评估至14B模型；需要访问输入问句的最后隐藏表示，难以应用于闭源模型；探测器仅在数学数据上训练，跨域泛化尚待验证

---

## 513. IAPO: Influence-Aware Policy Optimization for Credit Assignment in Multi-Turn Service Agents

**arXiv ID:** 2608.24588 | [PDF](https://arxiv.org/pdf/2608.24588v1)

**作者:** Bo Ren `[一作]` (Fudan University), Wenhui Que `[通讯]` (WeChat, Tencent Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种基于影响依赖图的策略优化方法IAPO，用来解决多轮服务代理中随时间展开的信用分配问题。该方法通过对已完成的对话轨迹提取支持使用与错误传播的有向图，按依赖关系重新分配轨迹级优势，从而在保持原始奖励不变的前提下实现更细粒度的动作级信用分配。

**💡 创新点**

创新点主要有：
1) 只利用已完成的轨迹内部依赖结构（支持使用与错误传播），无需额外的对比、重采样或自我评估信号；
2) 将影响依赖图映射为可归一化的正权重，实现优势在动作层级的动态重路由；
3) 通过解析权重的正向与负向分支，保证优势总量与符号不变，同时提升对错误传播的敏感度；
4) 在多轮交互场景中首次系统验证了在用户与工具信息不断展开时，基于图结构的信用分配能够显著提升性能。

**🔧 技术方法**

技术细节：
- 使用冻结的注解器（如 Qwen3‑32B）对每个已完成的对话轨迹提取有向影响图；
- 计算支持用与错误用的计数特征 ϕ⁺、ϕ⁻，经过标准化、裁剪与重归一化得到 m⁺、m⁻；
- 对正优势轨迹采用 β⁺ 控制错误折扣；对负优势轨迹直接使用 m⁻；
- 生成每个动作的权重 w，并按权重分配轨迹优势 Aᵢ,ℓ = Âτ wᵢ；
- 与 GRPO 共享其群组归一化、剪裁损失与策略梯度接口，只改动优势映射；
- 训练采用 Qwen3‑4B/8B，思考功能关闭，使用 8×H20 GPU，max 30 轮；
- 评估包括 τ²‑Bench、UserBench、AgentChangeBench、BFCL‑v4 Multi‑Turn 等。

**📊 数据集**

数据集：
- 训练集：τ²‑Bench 训练拆分（178 个任务）
- 评估集：
  * τ²‑Bench hold‑out 测试（不同域）
  * UserBench（Travel22/33/44）
  * AgentChangeBench（Banking/Education）
  * BFCL‑v4 Multi‑Turn（函数调用保留）
  * 还对比了 Fission‑GRPO、ToolACE‑2‑8B、BitAgent‑8B 等公开模型作为基线。

**📈 对比分析**

实验比较：
- 与 GRPO 进行受控对比，IAPO 在 τ²‑Bench 的宏观成功率从 29.61% 提升至 42.18%（+12.57pp），在 UserBench 与 AgentChangeBench 亦分别提升 4–5pp。BFCL‑v4 多轮函数调用保留几乎不变，说明并未损失泛化能力。
- 对比 GiGPO、InfoPO 等信用分配基线，IAPO 在三大服务代理基准上持续领先；
- Ablation 结果显示 β⁺ 与裁剪阈值 c 对性能影响显著，说明权重调度策略关键。
- 结果表明：在信息量大、用户/工具交互频繁的场景中，依赖图驱动的信用分配能显著提高任务完成率。

**⚠️ 局限性**

局限性：
- 依赖注解器的质量：如果图结构提取不准，权重分配可能失效；虽然实验表明多模型间一致性高，但仍需保证注解器稳定性；
- 只适用于用户/工具交互式服务代理，难以直接迁移到没有外部工具或用户信息逐步展开的其他多轮任务；
- 需要对每条轨迹执行一次注解，虽然开销较小，但在大规模在线训练中仍有一定成本；
- 对于极其复杂的依赖结构（如循环引用或隐式上下文）可能难以捕捉，导致信用分配不充分；
- 目前未探索将图结构与更高级的价值网络或多步奖励分解结合的可能性。

---

## 514. Confident at the moment of action: belief miscalibration in LLM play under hidden information

**arXiv ID:** 2608.24691 | [PDF](https://arxiv.org/pdf/2608.24691v1)

**作者:** Bhushan Kashinath Joshi `[一作]` `[通讯]`, Bhushan Kashinath Joshi

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过设计一款可秘密转移“王权”的象棋变体（Regent Chess），在每一步都让大型语言模型（LLM）先给出行动（棋步）后独立给出对对手隐含“王”位置的概率分布，随后记录实际“王”位置以评估模型在行动时的信心与实际正确性之间的校准误差。

**💡 创新点**

创新点在于：①首次在具有隐式状态并可被模型自身改变的环境中独立评估信心校准；②揭示了在行动时高置信度下几乎全无正确率的“信心-行动偏差”；③证明常规评估指标（合法性、成本、延迟、完成率）与模型对隐含状态的推断质量可能完全解耦。

**🔧 技术方法**

技术方法包括：①构建带有“王权转移”规则的自定义棋类引擎；②使用多种LLM（Gemini、GPT‑5）在固定、可读的启发式对手面前进行对局；③在每回合通过特定提示让模型输出棋步和一组top‑k概率分布；④采用Brier分数、可靠性图和匹配均匀先验的校准比较；⑤进行预注册实验设计、重现性检验和多批次复制。

**📊 数据集**

数据集为LLM与固定启发式对手进行的多场对局日志，涵盖两批独立运行的Gemini 3.1 Flash‑Lite（S1）以及四个其他座位（S5、S6、S6B、S7）的游戏记录；每批游戏中都记录了模型给出的概率分布和实际“王”位置，便于后续的概率校准评估。

**📈 对比分析**

比较方法：对每次捕获时模型给出的置信度与真实结果进行Brier分数比较，且与基于对手剩余活棋数的匹配均匀先验进行对照；此外在每回合的所有有效位置上做全局校准评估；结果显示：在高置信度捕获（≥0.5）中，S1仅在62次中正确1次（≈1.6%），与均匀先验的Brier比分别高3.8–15.9倍；同一类误差在四个其他座位中也有递增趋势，且与模型的外部Leaderboard分数无直接对应关系。

**⚠️ 局限性**

局限性包括：①完整校准电池仅在S1上完成，其他座位仅测得高置信度捕获率；②所有实验仅对单一固定、可读的启发式对手，未检验更具反应性的对手；③未进行跨任务或实际部署场景的转移验证；④使用top‑k概率分布可能导致覆盖率下降；⑤对模型内部采样不确定性的处理仍需改进。

---

## 515. One Timeline, Many Renderings: A Wolfram Language Paclet for heterogeneous musical output

**arXiv ID:** 2608.24683 | [PDF](https://arxiv.org/pdf/2608.24683v1)

**作者:** Francesco Vitucci `[一作]` (Conservatorio di Musica N. Piccinni di Bari), Francesco Scagliola `[通讯]` (Conservatorio di Musica N. Piccinni di Bari)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实现了一个名为Temporal System的Wolfram Language paclet，能够在单一、不可变的时间轴上统一生成Csound合成文件、MusicXML 4.0符号谱、OSC控制消息以及点击轨道，保持所有输出同步；

**💡 创新点**

创新点在于将时间结构与多种渲染目标（音频、符号、实时控制、点击）通过共享的、基于有理数的时钟映射统一管理，推迟单位转换到各渲染器阶段，并提供统一的实体存储和契约机制，解决传统工具间时间漂移问题；

**🔧 技术方法**

核心技术包括：Wolfram Language 15.0的paclet打包、纯函数式设计与不可变数据结构、精确有理数时间表示、分层架构（时间层、语义层、渲染契约层）、多后端（Csound、MusicXML、OSC、点击）调度与序列化；

**📊 数据集**

使用的“数据集”为作者在Wolfram Notebook中构造的示例作品，包含多种实体（音符、休止、触发、曲线、标记等）以及多拍子、变速段的时间表；

**📈 对比分析**

比较方法主要是对同一时间轴下生成的四种输出（Csound、MusicXML、OSC JSON、点击音频）进行时间轴一致性验证，并通过示例演示（如节拍变化导致的同步性变化），性能上尚未给出数值基准，但实现能够无漂移地渲染完整乐曲；

**⚠️ 局限性**

局限性包括：作者工具依赖Wolfram专有环境（免费Engine仅限非商业使用），缺少MIDI与连续tempo支持，曲线绑定仅适用于Csound的p-field，音乐XML处于beta阶段，且当前未实现命名通道与未来可能的其他渲染目标；

---

## 516. Game2World Engine: Unlocking In-the-Wild Gameplay Videos for World Model Training

**arXiv ID:** 2608.24680 | [PDF](https://arxiv.org/pdf/2608.24680v1)

**作者:** Wenxuan Shen `[一作]`, Dongping Chen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套完整的游戏 UI 去除框架和模型，将原始游戏视频转换为干净、可用于世界模型训练的高质量数据。

**💡 创新点**

提出统一的 HUD 分类、全流程数据引擎、面向无掩码的多模态语义理解+视频扩散的 UI 去除模型，以及基于 MLLM 的评估判定。

**🔧 技术方法**

使用 MLLM+视频扩散变压器（DiT）、LoRA 参数微调、可学习查询、无掩码 UI 去除、自动化资产提取与合成、以及 MLLM 判定器进行评估。

**📊 数据集**

构建 Game2World 数据集：96K 合成对视频、1,079 真实游戏视频、5,132 经过验证的 UI 资产（21 类），包括 Game2World‑S（synthetic）和 Game2World‑W（in‑the‑wild）。

**📈 对比分析**

与 Aurora、Kiwi‑Edit、VACE、LoomVideo 等基线比较，采用 AAR、BG 等指标。Mask‑free 版本在 synthetic 任务中获得 95.36 AAR、99.00 BG，in‑the‑wild 获得 80.05 AAR、99.80 BG，明显优于基线提升 20–30% 以上。

**⚠️ 局限性**

局限在于未覆盖全部游戏和 UI 多样性，难以处理大遮挡或快速变化的 UI，未实现完整动作条件的世界模型训练，推理效率高且需进一步压缩。

---

## 517. Security Education in Higher Education through AI-Powered Gamification

**arXiv ID:** 2608.24778 | [PDF](https://arxiv.org/pdf/2608.24778v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 518. Causal Explanations of Process Monitor Predictions

**arXiv ID:** 2608.24672 | [PDF](https://arxiv.org/pdf/2608.24672v1)

**作者:** Tom Yaacov `[一作]` (King's College London), Hana Chockler `[通讯]` (King's College London)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对预测过程监控（PPM）模型的黑盒预测结果，本文提出一种基于实际因果性框架的局部解释方法，能够识别并量化每个事件对预测结果的因果责任。

**💡 创新点**

创新点包括：①构建了专门针对过程的因果模型，捕捉事件之间的时间依赖关系；②设计了一种基于责任度（degree of responsibility）的近似算法，能够在不显式构造完整因果图的情况下高效估计事件的重要性；③将该方法与现有的LIME、SHAP等通用解释方法进行对比，并在多种真实事件日志上验证其更高的稳定性与简洁性。

**🔧 技术方法**

主要技术手段包括：实际因果性理论、责任度度量、利用LSTM过程模拟器对可达轨迹进行采样、对黑盒分类器（XGBoost、MLP）进行预测、插入测试（Insertion Test）评估解释质量、VSI指标评估一致性、以及执行时间和解释大小的测量。

**📊 数据集**

实验使用了来自公开PPM基准的22个数据集，来自9个真实事件日志，最多包含130,000条轨迹、长度可达1,800、400种不同事件类型；每个数据集均划分为80%训练集和20%测试集。

**📈 对比分析**

与LIME和SHAP进行比较，采用一致性（VSI）、解释长度（占比）和执行时间三项指标评估。结果显示，AC4PM在大多数数据集上实现了更高的VSI、更短的解释长度，并且在大部分情况下保持了可接受的运行时间；相较之下，LIME和SHAP在长轨迹和高维度特征时表现出更低的稳定性和更长的执行时延。

**⚠️ 局限性**

局限性包括：①仅考虑活动名称和时间顺序，未能充分利用事件的连续属性或上下文信息；②因果责任的近似依赖于采样的覆盖度，采样不足时可能导致误差；③算法实现为原型，仍有提升效率的空间，尤其在极长轨迹或大规模事件集时的计算复杂度较高。

---

## 519. Expectation, Backlash, Recovery, and Excitement: How Model Releases Shape Reddit Perceptions of Conversational AI Systems

**arXiv ID:** 2608.24654 | [PDF](https://arxiv.org/pdf/2608.24654v1)

**作者:** Vahid Rahimzadeh `[一作]` (Delft University of Technology), Savvas Zannettou `[通讯]` (Delft University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对 2022-2025 年 668,063 篇 Reddit 讨论的长期大规模分析，研究 Conversational AI 系统（CAIS）模型发布如何动态影响用户情绪与话题；

**💡 创新点**

将模型发布视为社会技术事件，首次在海量社交媒体文本中自动化提取、归纳并跟踪用户感知的主题与情绪变化；

**🔧 技术方法**

使用基于 LLM 的模型提及识别器、LLooM 框架进行概念生成与去重、LLM 情感分类，并结合对称时间窗口、Theil–Sen 趋势等统计方法；

**📊 数据集**

利用 2022-2025 年涵盖 20 个子版块的 Reddit 数据集，共 668,063 篇帖子、505k 次模型提及；

**📈 对比分析**

通过发布前后对称窗口的情感和概念差异统计，采用 FDR 校正和 Pearson r 等方法验证显著性；结果显示 Anthropic 发布普遍正面，OpenAI/Google 负面，模型发布显著影响情绪与讨论主题；

**⚠️ 局限性**

仅分析文本 Reddit，忽略多媒体内容；样本偏向技术活跃用户；LLM 方法可能产生误分类；对称窗口可能平滑短期反应；无法识别 AI 生成的帖子。

---

## 520. Scalable datacenter replication with mostly-synchronous consensus on hardware

**arXiv ID:** 2608.24622 | [PDF](https://arxiv.org/pdf/2608.24622v1)

**作者:** Davide Rovelli `[一作]` (Universita della Svizzera Italiana), Patrick Eugster `[通讯]` (Universita della Svizzera Italiana)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

在分布式数据中心中提出了一种基于FPGA网络卡的可扩展一致性复制系统（name），实现了新的无领导者共识算法ckcuc，能够在多数同步环境下高效处理多节点复制请求；

**💡 创新点**

创新点在于将“几乎同步”系统模型与硬件加速相结合，提出可同时做出多个决策的共识变体cc，解决了传统基于领导者的SMR瓶颈，并在保证安全性的同时实现零停机时间；

**🔧 技术方法**

使用的技术包括FPGA实现的网络接口卡（AMD/Xilinx Alveo U50）、TLA+形式化验证、RDMA/网络可编程交换机等；

**📊 数据集**

评估基于Redis和Zookeeper的实际工作负载，使用100Gbps网络吞吐量与5.4µs响应延迟的测试场景；

**📈 对比分析**

与Raft（Waverunner）、NOPaxos等最先进SMR实现进行对比，name在三台以上复制节点时吞吐量提升至2.5倍以上，Redis吞吐量和延迟提升高达两位数倍；

**⚠️ 局限性**

局限性包括对硬件FPGA网络卡的依赖、在高度异步或网络时延波动大环境下性能可能受限，以及缺乏对极端大规模节点（数百以上）下的可扩展性与容错边界的深入评估。

---

## 521. Learning to Prefer Reliably: Error-Augmented Emotion Preference Optimization with Calibrated Fusion

**arXiv ID:** 2608.24730 | [PDF](https://arxiv.org/pdf/2608.24730v1)

**作者:** Zilong Huang `[一作]` (Hong Kong Polytechnic University), Tatsuya Kawahara `[通讯]` (Kyoto University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了错误增强偏好优化框架（EAPO），通过生成四类受控错误负样本并结合多模态大语言模型（MLLM）进行SFT+ DPO训练，再利用置信度标定的软融合提升情绪偏好判断；

**💡 创新点**

创新点在于（1）数据层面构建多种可控错误负样本（情感翻转、强度不匹配、证据矛盾、模态省略），显著扩展监督多样性；（2）模型层面采用多独立MLLM判别器，并通过边际标定实现跨模型偏好边际的统一尺度融合；（3）引入候选顺序交换减少位置偏差；

**🔧 技术方法**

技术包括LoRA增量微调（SFT）、直接偏好优化（DPO）、文本编辑规划LLM用于错误生成、规则与语义验证、边际标定（σ归一）以及软融合（平均归一化边际）；

**📊 数据集**

使用MER2026‑EmoPrefer原始对比数据集及其错误增强版本（2,908对增强对照），并在官方Stage 1与Stage 2测试集上评估；

**📈 对比分析**

与零射击、仅SFT、SFT+DPO、单一判别器、硬投票、原始融合等方案对比，EAPO的校准融合在官方宏观WAF上达到约80.23%，优于官方基线（≈78.7%）及任何单一判别器，且在错误子集和候选顺序一致性上均表现最佳；

**⚠️ 局限性**

局限性包括生成中间描述（S2策略）的质量不一可能影响偏好判断；融合方法仅在当前模型族与数据分布下验证，未评估在模型更换或域漂移下的稳健性；未来需结合人工审核与多代理生成以提升错误样本质量与泛化性。

---

## 522. Arbitrary Polygon Oscillator: Generalizing Polygonal Synthesis to Arbitrary Shapes, Morphing, and Three-Dimensional Polyhedra

**arXiv ID:** 2608.24726 | [PDF](https://arxiv.org/pdf/2608.24726v1)

**作者:** Antonio Argentieri `[一作]` (Conservatorio Niccolò Piccinni di Bari), Francesco Scagliola `[通讯]` (Conservatorio Niccolò Piccinni di Bari)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

论文提出了一套基于弧长参数化的任意多边形振荡器，可对任意闭合多边形进行音频合成，并支持多边形形状的平滑插值和3D多面体截面扩展。

**💡 创新点**

创新点包括：① 用弧长遍历替代传统的角速度遍历，实现对任意顶点配置的统一处理；② 混合插值算法在顶点数不一致时保留尖角；③ 针对任意顶点缓冲区的 polyBLAMP 四点抗锯齿和自适应超采样；④ 通过“睡眠顶点”保持三维截面顶点数稳定，实现连续的三维形变；⑤ 在 RNBO 环境中实现实时 DSP。

**🔧 技术方法**

技术实现包括：弧长参数化、四点 polyBLAMP 抗锯齿、贝塞尔控制点曲线、顶点匹配与循环对齐、睡眠顶点机制、3D 体裁剪、超采样、低通滤波、RNBO 模块化设计。

**📊 数据集**

实验使用自定义的多边形数据（等边三角形、正多边形、星形、锥体等）和 3D 立方体/金字塔/二十面体等几何体；没有公开数据集，而是通过手绘顶点缓冲区生成测试样本。

**📈 对比分析**

通过在不同抗锯齿配置（无、单纯超采样、polyBLAMP、两者结合）下测量 SNR，结果显示 polyBLAMP 单独提升约 22 dB，超采样和 polyBLAMP 结合可获得最高 SNR；同时展示了多边形插值和 3D 截面变形的频谱变化，证明系统保持了音色连贯性。

**⚠️ 局限性**

局限性包括：缺乏基于感知的插值优化；对非常多顶点（>24）时计算量上升；仅支持凸多面体截面，非凸 3D 形体处理有限；插值过程仍需手动顶点排布；跨形状切换时仍可能出现微小 DC 偏移。

---

## 523. Deep Learning Super Resolution for Satellite Cloud Mask Downscaling

**arXiv ID:** 2608.24715 | [PDF](https://arxiv.org/pdf/2608.24715v1)

**作者:** Angelos Georgakis `[一作]` (National Observatory of Athens), Kostas Philippopoulos `[通讯]` (National and Kapodistrian University of Athens)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过深度学习实现MSG SEVIRI低分辨率云罩向MODIS高分辨率云罩的上采样

**💡 创新点**

提出跨传感器云罩超分辨率的两个模型（SpatialCNN与SpatialGAN）并创建SEVMOD‑CM数据集

**🔧 技术方法**

采用CNN与GAN框架，结合残差学习、进阶上采样、BCE、感知与对抗损失等技术

**📊 数据集**

使用MODIS（MOD35_L2/MYD35_L2）和SEVIRI（MSG）云罩数据，经过时间空间对齐后生成12通道输入与二值目标的配对数据集

**📈 对比分析**

与双三次插值基线比较，SpatialGAN在SSIM和MSE上显著优于基线，PSNR略低；SpatialCNN略逊于双三次插值

**⚠️ 局限性**

受跨传感器空间、时间、光谱对齐误差的限制，SR方法仍无法完全恢复高频细节

---

## 524. StarHarness: Evolving Harnesses with Stratified Search for Enterprise Environments

**arXiv ID:** 2608.24804 | [PDF](https://arxiv.org/pdf/2608.24804v1)

**作者:** Esakkivel Esakkiraja `[一作]` (ServiceNow), Sagar Davasam `[通讯]` (ServiceNow)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在不修改模型权重的前提下，通过对企业环境中的工具调用与状态管理框架（harness）进行搜索和演化，提升固定语言模型在状态化任务上的表现。

**💡 创新点**

提出了任务分层、guardrail约束的harness演化框架，结合 hill‑climbing 与 tree‑search 两种搜索策略，实现对模型-环境摩擦的自动修复与知识压缩。

**🔧 技术方法**

基于 Git diff 的可编程 harness 编辑器、Pi‑agent 运行时、validator/ eulerator 评估循环以及持久化的 ledger 记录；搜索策略包括 hill‑climbing、tree‑search；使用的工具包括 MCP、SQL、API 等。

**📊 数据集**

三大企业级 benchmark：ITBench（40 例 Kubernetes 根因分析）、EnterpriseOps‑Gym（103 例 ITSM 工作流）以及 AutomationBench（100 例财务工作流）。

**📈 对比分析**

对比基线 Stirrup、Pi、Codex 等模型，StarHarness 在 ITBench +13.8pp、EnterpriseOps‑Gym +22.3pp、AutomationBench +17.6pp；在冻结模型跨模型迁移中提升 10–46pp；同时在 API 成本上分别降低 17%、53% 与 29%。

**⚠️ 局限性**

局限性在于只能改进固定模型权重；单个补丁的具体贡献难以量化；搜索空间受 guardrail 限制，且对新模型或新环境仍需重新演化；未探索与模型权重共同训练的协同优化。

---

## 525. EMFE: A lightweight, explainable machine learning framework for malaria cell classification

**arXiv ID:** 2608.24793 | [PDF](https://arxiv.org/pdf/2608.24793v1)

**作者:** Md Abdullah Al Kafi `[一作]` (Daffodil International University), Ahmed Al Marouf `[通讯]` (University of Alberta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `70e40602-aae3-44bd-80ec-4a7f2674330f` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出一种基于五个数学特征的轻量级特征提取框架EMFE，用于单细胞血涂片的疟疾诊断；

**💡 创新点**

创新点在于通过患者级分层交叉验证消除数据泄漏、提供严格统计检验，并用可解释的五个生物学特征实现高精度；

**🔧 技术方法**

采用Gray World色彩归一化、适应性绿通道阈值与形态学检测，配合随机森林等传统机器学习器；

**📊 数据集**

使用NIH Lister Hill国家生物医学通信中心公开的27,558张单细胞图像数据集（200名患者）；

**📈 对比分析**

在患者级20折嵌套交叉验证中随机森林达94.6%准确率，80%+的显著性检验；与DenseNet、ResNet、MobileNet基线相比仅低1.7–2.4点准确率，却快3.8–43倍、体积1.4–14倍；

**⚠️ 局限性**

局限在于仅针对单细胞图像、缺乏外部多中心验证、未实现细胞检测或病例级聚合、对低对比、模糊、低分辨率等情况敏感。

---

## 526. Linear Probing Provides Robust and Efficient Detection of Machine-Generated Text

**arXiv ID:** 2608.24780 | [PDF](https://arxiv.org/pdf/2608.24780v1)

**作者:** Gerrit Quaremba `[一作]` (King's College London), Elena Simperl `[通讯]` (King's College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究机器生成文本与人工文本在语言模型潜在空间中的线性可分性，并提出利用简单线性探测器（probe）进行检测

**💡 创新点**

揭示两类文本的潜在表示存在可共享且可迁移的线性方向，证明线性探测器比传统监督检测更具样本效率和跨域泛化能力；同时表明该方向可连续度量AI编辑强度

**🔧 技术方法**

基于Transformer隐藏状态的线性逻辑回归探测器（Layer‑Averaged Probe 与 Concatenated‑Layer Probe），配合PCA降维、对数似然、排名等传统零样本和监督基线做对比

**📊 数据集**

使用四个公开基准（DetectRL、MultiSocial、RAID、TSM）以及其子集，共计16个评估场景，涵盖不同域、语言、生成器和任务

**📈 对比分析**

与16种基线（零样本与监督）相比，探测器在ID场景提升0.04–18.85 AUC，OOD场景提升0.39–11.83 AUC；仅需10–100个样本即可逼近最优性能，并在AI编辑度量上与编辑强度呈高相关性

**⚠️ 局限性**

仅适用于可访问内部隐藏状态的开源模型；对表示差异的分析有限；未系统探究模型规模或更复杂探测器的影响；未验证方向的普适性与因果性

---

## 527. Design and Empirical Characterization of a Hardware-Realized Turing Machine with Automated Card-Based Programming

**arXiv ID:** 2608.24742 | [PDF](https://arxiv.org/pdf/2608.24742v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 528. Evidence Blindness in Direct Corpus Interaction: Persistent Navigation with AtlasNav

**arXiv ID:** 2608.24764 | [PDF](https://arxiv.org/pdf/2608.24764v1)

**作者:** Hongyu Guo `[一作]`, Zhao Cao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AtlasNav，一个持久多视角语料库导航框架，改进了直接语料库交互（DCI），通过一次性构建 Topic/Identity/Episode/Relation Atlas 并在查询时自适应路由，显著降低 Evidence Blindness 并提升多跳检索效率。

**💡 创新点**

将证据盲点量化为四阶段过程（Construction、Surface、Open、Locate），并首次在 DCI 环境下采用持久多视角 Atlas 取代动态工作空间，实现有限预算内的高效导航；同时通过多视角嵌入+Leiden 社区检测+RRF 组合路由提供可解释且可扩展的查询适配方案。

**🔧 技术方法**

使用 Topic、Identity、Episode、Relation 四种语义视角的向量编码；Leiden 社区检测构建多层 Atlas；BM25 词检索与 Reciprocal Rank Fusion（RRF）进行多视角融合；大语言模型 (如 DeepSeek‑V4‑Flash、MiMo‑V2.5、ChatGPT‑5.6‑Luna、Qwen‑3.7‑Flash) 作为底层代理进行 DCI。

**📊 数据集**

在公开基准 BrowseComp‑Plus、PhantomWiki（规模从 10K 递增至 1M）以及 EnterpriseRAG‑Bench（511,958 文档、500 题）上进行实验。

**📈 对比分析**

与原始 DCI、DR‑DCI 及主流系统比较：在 BrowseComp‑Plus 上 AtlasNav 的严格准确率提升 3.98–21.57 %（最高 21.57 %），在线推理成本下降 0.62–30.21 %；在 PhantomWiki 上保持最高准确率并将 Surface 盲点降至 49.5 %（相较 DCI 59.0 %、DR‑DCI 87.0 %）；在 EnterpriseRAG‑Bench 上取得 73.72 Overall，靠近主流系统的 76–80 %，并在正确率与完整率上表现均衡。

**⚠️ 局限性**

局限性：需一次性预处理构建 Atlas，难以快速响应大规模动态更新；对多视角嵌入质量高度依赖；在极大规模（≈1 M+文件）或高度动态的知识库中，Atlas 的更新与查询效率可能受限；在多文档合成或冲突信息处理方面仍存在挑战。

---

## 529. IDeaL: Data-Free Multi-Teacher Distillation via Improved Dead Leaves

**arXiv ID:** 2608.24759 | [PDF](https://arxiv.org/pdf/2608.24759v1)

**作者:** Feyza Yavuz `[一作]` (NAVER LABS), Diane Larlus `[通讯]` (NAVER LABS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了在无真实数据条件下进行多教师蒸馏的方法，利用改进的Dead Leaves合成样本训练学生模型。

**💡 创新点**

创新点在于通过基于教师注意力的像素级去相关损失，显著提升结构化噪声样本的多教师信息表达，使得数据自由蒸馏性能逼近真实数据。

**🔧 技术方法**

使用了Vision Transformer（ViT）教师与学生、基于注意力特征的去相关正则、像素级优化以及传统的多教师蒸馏损失。

**📊 数据集**

在ImageNet-1K上训练教师，使用其合成样本进行蒸馏，并在15个图像分类、语义分割和深度估计等多下游任务上评估。

**📈 对比分析**

与真实图像子集蒸馏相比，改进的Dead Leaves在1K样本时已能超过最低教师性能，在1M样本时与真实数据性能相差不足10%，在分类、语义分割、深度估计等任务均表现出显著提升。

**⚠️ 局限性**

局限性包括：合成样本在大样本量或密集任务上仍不如真实图像扩展性差；生成过程需要额外的像素优化计算；且对教师特定的注意力特征假设，若教师结构不同可能效果受限。

---

## 530. ICS Cybersecurity Datasets: A Systematic Meta-Review of Coverage, Evaluation Practice, and Structural Gaps

**arXiv ID:** 2608.24757 | [PDF](https://arxiv.org/pdf/2608.24757v1)

**作者:** Konstantinos E. Kampourakis `[一作]`, José Luis Hernández-Ramos `[通讯]` (Universidad de Murcia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对近7年内关于ICS/OT/IIoT公开数据集的18篇综述性论文进行系统性元评审，提取并统一对83个数据集进行五维度（攻击战术、进程深度、体系结构层级、来源真实性、标签质量）的分类与量化分析，随后对这些综述中描述的评估实践进行偏差审计。

**💡 创新点**

提出了以MITRE ATT&CK、ICS Kill Chain、IEC 62443、NIST 800‑82、CSF Detect为基础的统一五维度分类框架，首次在同一尺度上兼顾攻击、体系结构、来源与标签四个维度，并将该框架与评估方法的局限性关联，揭示三大结构性缺口（架构浅薄、进程压缩、跨域替代）。

**🔧 技术方法**

采用PRISMA指导的系统综述流程、数据提炼与编码规则，构建统一的五维度元数据表，对数据集进行描述性统计，并使用关联分析（如D1–D5与评估偏差的对应表）阐释数据集属性与评估可靠性之间的耦合关系。

**📊 数据集**

共提炼出83个公开数据集，涵盖ICS、SCADA、IIoT、IoT及通用IDS基准（如SWaT、WADI、BATADAL、TON_IoT、CICIDS、KDDCup99等），并从综述中归纳其攻防、层级、来源、标签等属性。

**📈 对比分析**

通过对每个综述中报告的评估设计（分区方式、窗口划分、阈值校准、标签粒度、指标、验证范围、运行模式、可复现性）进行评分，显示大部分研究仅采用随机或未约束的时间划分、点标记、传统准确率/F1等指标，缺乏事件级延迟与误报率评估；总体表现为高估指标、低可复现性，缺乏跨数据集、在线评估与真实来源验证。

**⚠️ 局限性**

研究的局限在于仅聚焦已发表综述，未检索所有公开数据集；分类与编码依赖主观判定，可能存在误差；统计为描述性而非统计推断；评估偏差分析基于综述描述，缺乏原始实验重现；未对不同数据集之间的可迁移性进行系统实验。

---

## 531. Meta$^n$: Recursive Self-Improvement through Emergent Depth

**arXiv ID:** 2608.24735 | [PDF](https://arxiv.org/pdf/2608.24735v1)

**作者:** Zae Myung Kim `[一作]` (University of Minnesota), Dongyeop Kang `[通讯]` (University of Minnesota)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种Meta-Recursion框架，通过固定的元操作Ω递归处理自身输出，形成多层堆栈，从而提升LLM解题性能。

**💡 创新点**

创新点在于：①不让Ω自我修改而保持稳定；②递归输入信息逐层递增，深度由收敛决定；③通过上下文条件化与代码库注入实现层与层之间的高阶协同。

**🔧 技术方法**

技术包括：Gemma4 31B-IT 与 GPT-5.2 作为底层solver；固定Ω的LLM提示模板；线性递归与进化档案搜索两种构建方法；预处理函数与可调用代码库的生成与注入；多层包装器实现堆栈执行。

**📊 数据集**

使用八个基准家族：CO-Bench、AlphaEvolve Math、Symbolic Regression、AlgoTune、ARC-AGI-2、TerminalBench 2.0、Symptom2Disease、LawBench。

**📈 对比分析**

与 Gödel Agent 与 OpenEvolve 在相同模型与计算预算下比较，采用 archive-best 与 best chain 两种评估；在所有基准上均优于两者，尤其在 ARC-AGI-2 达到 0.331、CO-Bench 0.870，其他基准均显著提升。

**⚠️ 局限性**

局限性包括：仅在同一模型上测试，未验证更强模型在Ω层的效果；上下文为自由文本，未探索结构化表示；递归深度上限未知，实验中深度停在 3–6 层。

---

## 532. Optimal Alternating Regret for Online Learning and Games

**arXiv ID:** 2608.24731 | [PDF](https://arxiv.org/pdf/2608.24731v1)

**作者:** Yixin Tao `[一作]` (Shanghai University of Finance and Economics), Weiqiang Zheng `[通讯]` (Yale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

本文提出并分析了在线线性/凸优化中的交替遗憾概念，给出了最优算法和匹配的下界，并基于此实现了两玩家博弈中O(1/T)收敛速度的交替学习动态。

**💡 创新点**

创新点包括：①首次证明OLO上的交替遗憾最优为Θ(log d)；②首次证明OCO上的交替遗憾最优为Θ(d log(1+T/d))；③提出AA‑Hedge与连续AA‑Hedge算法实现上述上界；③利用该结果得到两玩家一般博弈中首次实现O(1/T)的粗相关均衡（CCE）收敛。

**🔧 技术方法**

技术手段主要有：潜能函数分析、凸优化理论、Kullback–Leibler散度论证、连续化Hedge变体、对手构造与旋转几何构造、分段损失构造等。

**📊 数据集**

无实验数据集；研究完全基于理论分析与构造对手。

**📈 对比分析**

与以往的O(log^{2/3}d T^{1/3})或O(log T/T)等结果相比，本文的上界与下界匹配，性能显著提升；在博弈中实现了先前仅存在的O(log T/T)或O(1/T^{1/4})的更快O(1/T)收敛。

**⚠️ 局限性**

局限性：仅适用于有限维凸集且损失范围为[-1,1]；常数因子可能较大；未考虑随机或非凸环境；实际实现的计算复杂度与高维结构的适用性尚待进一步研究。

---

## 533. Constrained Hyperparameter Optimization for Streaming Data

**arXiv ID:** 2608.24712 | [PDF](https://arxiv.org/pdf/2608.24712v1)

**作者:** Bruno Veloso `[一作]` (University of Porto), João Gama `[通讯]` (University of Porto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在数据流在线超参数优化中处理边界约束的方法，并将五种约束策略（Boundary、Reflection、Centroid、Random、Wrapper）应用于SPT和MESSPT两个在线优化器。

**💡 创新点**

提出将多种边界约束校正技术与在线优化算法结合，并系统评估其在不同任务下的效果，揭示不同约束策略在分类与回归任务中的优劣差异。

**🔧 技术方法**

采用SPT（基于Nelder–Mead的自适应搜索）和MESSPT（微进化策略）两种在线优化器，结合五种边界约束校正方法，利用RiverML框架与ADWIN/ DDM漂移检测实现在线学习。

**📊 数据集**

在多种分类（ENRON1、NOMAO1、RandomRBF2、Agrawal2、Hyperplane2、SEA_Drift）和回归（Tetuan3、Metro4、2DPlanes2、MV2、Friedman Drift、Friedman2）数据流上进行实验。

**📈 对比分析**

采用Prequential评估协议，对每种约束策略在两个优化器下的准确率或RMSE进行排名；结果显示在分类任务中Centroid策略最优，在回归任务中Reflection策略最优，而Boundary与Wrapper在特定情境下表现突出。

**⚠️ 局限性**

仅评估了五种约束方法与两种优化器，未覆盖更复杂的漂移检测或更大规模数据流；部分策略在某些任务中未带来提升，表明仍需探索更精细的边界约束处理技术。

---

## 534. Ensemble of Convolutional Neural Networks for StrokePrediction: Towards Improved Diagnostic Accuracy

**arXiv ID:** 2608.24771 | [PDF](https://arxiv.org/pdf/2608.24771v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 535. Dynamic Edge Orientation via Random Walks: From Trees to Outerplanar Graphs and Beyond

**arXiv ID:** 2608.24776 | [PDF](https://arxiv.org/pdf/2608.24776v1)

**作者:** Gabriel Marques Domingues `[一作]` (Tel Aviv University), Shay Solomon `[通讯]` (Tel Aviv University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并分析了一种随机走法，用于在全动态外拓扑图以及 K₂,t‑无族图中保持常数最大出度，并在每次更新时实现 O(log n) 的最坏情况更新时间和回转量。

**💡 创新点**

①将随机走法从仅适用于森林推广到外拓扑图，并给出阈值 4 是最优的证明；②通过分层分析得到在 K₂,t‑无族图中的阈值 48(t−1)；③首次将此随机走法实现为几乎无损的分布式协议，保持 O(log n) 轮次与消息数。

**🔧 技术方法**

使用随机走法 + 循环消除、路径计数（外拓扑图中的 4^k 上界与 K₂,t‑无族的 Catalan/4(t−1)^k 上界）、ζ‑加权随机过程、弱双子图分层、树分解与递归（centroid 迭代）等技术。

**📊 数据集**

无实验数据集；论文为纯理论分析，不涉及数据实验。

**📈 对比分析**

与 Brodal‑Fagerberg（BF）及后续改进的随机走法相比，算法在常数 arboricity 的图中实现了最坏情况 O(log n) 更新时间与回转量；在分布式 CONGEST 本地唤醒模型下，恢复过程仅需 O(log n) 轮次与消息，优于以往仅提供期望或无全球数据结构的方法。

**⚠️ 局限性**

①阈值 3 在外拓扑图中不可实现 O(log n) 的随机走法；②在树宽 2（series‑parallel）图中随机走法可产生多项式步数；③仍未得到确定性 O(log n) 最坏情况更新时间；④阈值 2 需要 Ω(n) 的递归量（即使离线）。

---

## 536. Ten Years Later: Replicating Two Color Discrimination Studies

**arXiv ID:** 2608.24789 | [PDF](https://arxiv.org/pdf/2608.24789v1)

**作者:** Shadmaan Hye `[一作]` (University of Utah), Katherine E. Isaacs `[通讯]` (University of Utah)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对两项十年前的颜色辨别实验进行在线复制，并检验自报的颜色实践是否会影响辨别能力。

**💡 创新点**

在保持实验任务不变的前提下，首次引入颜色实践问卷评估，并在新平台、新样本下验证其可重复性。

**🔧 技术方法**

使用自制的reVISit平台实现实验，采用色差模型 CIELAB 与 CIE L*u*v* 进行阈值建模，利用二元强迫选择与二分搜索算法进行数据收集。

**📊 数据集**

利用原始实验的 79 色彩样本（散点图）与 4 种 Landolt‑C 颜色（光度/红/蓝/洋红），并收集 Prolific 上 144 名散点实验与 394 名 Landolt‑C 实验参与者数据。

**📈 对比分析**

通过与原始实验的阈值曲线和阈值椭球体体积分布进行对比，使用 Welch t‑检验和回归分析验证结果，发现整体匹配但在最小标记尺寸与椭球体积上略有差异，说明实验在不同硬件与受试者群体下仍能重现关键感知模式。

**⚠️ 局限性**

主要限制包括：在线实验无法控制显示器校准与照明，颜色实践分类仅基于自报信息，样本与原实验不同且仅复制了原实验的一部分条件，导致部分细节差异。

---

## 537. Test-Time Collaborative Classification over Multi-Agent Networks

**arXiv ID:** 2608.24787 | [PDF](https://arxiv.org/pdf/2608.24787v1)

**作者:** Ping Hu `[一作]` (École Polytechnique Fédérale de Lausanne), Ali H. Sayed `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了多代理网络中，在各代理独立训练后，通过在推理阶段进行分布式协同推断，实现分布式二分类。

**💡 创新点**

创新点在于：①提出独立训练 + 推理时协同的框架；②给出了在有限通信轮数、有限精度通信以及 PAC 风格下的误差上界；③通过 DeGroot 规则与经验中心化，统一处理不同特征空间和模型架构的异构代理；④用近似张量化熵与 Rademacher 复杂度等工具提供理论保证。

**🔧 技术方法**

主要技术包括：DeGroot 迭代平均、经验中心化、经验风险最小化与优化误差控制、近似张量化熵（ATE）假设、随机逼近量化（stochastic rounding）、Rademacher 复杂度与 PAC‑margin 边界、图论中 Perron 向量与谱半径分析。

**📊 数据集**

实验数据集：CIFAR‑10（patch‑partition 3×3 网格）与 ModelNet40（多视角 12 视图）两种 benchmark。

**📈 对比分析**

与多种基线（非协同、平均投票、融合规则、AdaBoost、VFL‑JT、中心全信息预测）以及理论上理想的中心化全信息 Oracle 进行对比。结果显示：在足够的通信轮数下，协同推断明显优于非协同，且在样本量足够时接近或可与 VFL‑JT 的性能相当；在早期通信轮数已显著提升性能，且在多视角场景下温度缩放与通信精度对性能有积极影响。

**⚠️ 局限性**

局限性包括：仅针对二分类任务；假设类先验均匀、图强连通且自环存在；需要满足近似张量化熵假设，实际数据依赖性难以验证；未考虑异步、时变拓扑或丢包；对小样本情况下的独立训练效果可能不如联合训练；量化误差与通信预算之间存在折衷，但理论上对极低位宽或极短轮数的严谨性尚未完全覆盖。

---

## 538. Shaping the Future of Generative AI for Black Communities: A Frame Analysis of Public Discourse and Empirical Scholarly Research

**arXiv ID:** 2608.24767 | [PDF](https://arxiv.org/pdf/2608.24767v1)

**作者:** Angela D. R. Smith `[一作]` (University of Rochester), Christina N. Harrington `[通讯]` (Google Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对91篇关于生成式AI与黑人社区的实证研究进行系统文献综述，并对28篇公开媒体资源应用Entman框架理论进行语篇分析，探讨公共话语与学术研究在问题定义、因果解释、价值判断和干预建议上的对齐与偏差。

**💡 创新点**

创新点在于首次将框架分析方法引入AI伦理研究，以揭示公共话语与学术研究之间结构性不一致，并强调黑人知识主体的被排除，提示框架分析可补充技术评估在预测结构性伤害方面的局限。

**🔧 技术方法**

使用两种方法：系统文献综述（SLR）和基于Entman四要素的框架分析；分析框架属性包括问题定义、因果解释、道德判断与干预建议。

**📊 数据集**

数据集来源为：91篇在ACM、IEEE、ACL、ArXiv等计算机科学期刊与会议上发表的实证研究；以及28篇涵盖新闻、专栏、白皮书等公开媒体资源。

**📈 对比分析**

通过对比两套语料库在框架属性上的频率与分布进行定性比较，未涉及模型或算法的性能评估，主要呈现数量与类别分布，指出学术研究聚焦技术改革而公共话语强调历史与结构性因果。

**⚠️ 局限性**

局限性包括：1）未覆盖AAA I 会议论文，可能导致研究视角不完整；2）仅聚焦计算机科学出版物，忽视社会科学与人文学科的视角；3）缺乏黑人社区自身语篇与观点；4）框架分析为定性方法，无法量化影响；5）未对干预建议的实际有效性进行验证。

---

## 539. Beyond Uniform Local Isometry and Topology: FactoMap for Disentangled Representations

**arXiv ID:** 2608.24762 | [PDF](https://arxiv.org/pdf/2608.24762v1)

**作者:** Sohini Gupta `[一作]` (University of Alberta), Bahareh Tolooshams `[通讯]` (University of Alberta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Factor-Space Topographic Map（FactoMap），通过学习匹配因子空间拓扑和尺度的原型网格，实现对非欧氏、位置相关尺度因子的可解释解耦。

**💡 创新点**

创新点在于将因子空间结构拆分为域、生成器同一化与位置相关尺度，利用拓扑和尺度信息构造可变形的格点，突破传统局部等距假设。

**🔧 技术方法**

使用自组织映射（Self‑Organizing Map）目标训练可解释的原型网格，并通过分析生成器的偏导来定义因子尺度；采用离散格点距离衡量因子空间结构。

**📊 数据集**

使用合成数据集FactoShapes，包含色调、尺度、水平和垂直位置四个可控因子。

**📈 对比分析**

通过匹配因子空间拓扑（环形或锥形格）与非匹配（网格）进行比较，利用InfoM、InfoE、InfoC指标评估解耦效果；匹配结构下InfoM约0.95、InfoE约0.82、InfoC约0.95，远优于匹配网格的0.27/0.54/0.025。

**⚠️ 局限性**

局限在于仅在合成数据上验证，未处理跨模态或真实世界复杂因子；需进一步研究如何自动推断因子空间结构并扩展到更高维情形。

---

## 540. ExpConCAD: Experience-Guided Text-to-CAD Generation from Shape Descriptions with Implicit Spatial Constraints

**arXiv ID:** 2608.24760 | [PDF](https://arxiv.org/pdf/2608.24760v1)

**作者:** Jingyao Liu `[一作]` (Sichuan University), See-Kiong Ng `[通讯]` (National University of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 ExpConCAD 框架，解决文本描述中缺失的空间约束，使模型能够生成可执行的 CadQuery 代码；

**💡 创新点**

创新点在于将构造结构理解与可迁移经验记忆相结合，通过检索过去 CAD 例子的约束补全模式来恢复隐式空间约束；

**🔧 技术方法**

采用 LLM（Qwen3.5‑27B、GPT‑5）进行自然语言理解与代码生成，配合构造结构理解模块、空间约束补全模块以及经验记忆检索与执行验证；

**📊 数据集**

主要使用 CADFusion‑Hard（硬难度子集）以及完整的 CADFusion 数据集进行训练与评测；

**📈 对比分析**

与 Text2CAD、CADFusion、CADCoder 等基线在 VLM 评分、几何一致性指标上进行对比，ExpConCAD 在 Qwen 基线上提升 VLM 22.5%，在 GPT 基线上提升 27.9%，整体性能显著优于现有方法；

**⚠️ 局限性**

局限性包括经验记忆的维护与更新策略未研究，可能导致存储冗余、检索效率下降以及无法及时加入新的约束模式。

---

## 541. RACE: Scalable Statistical Estimation of Functional Consistency in LLM Neurons

**arXiv ID:** 2608.24758 | [PDF](https://arxiv.org/pdf/2608.24758v1)

**作者:** Runyu Wang `[一作]` (Nantong University), Peng Ping `[通讯]` (Nantong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了RACE（Residual Alignment for Consistency Estimation）框架，通过前向残差方向对齐与贝叶斯聚合，对Transformer神经元在目标域的功能一致性进行统计评估，并通过神经元抑制实验验证其因果有效性。

**💡 创新点**

创新点：①采用残差流方向对齐（RDA）获取每个神经元对残差的贡献，避免梯度计算；②使用Normal-Inverse-Gamma贝叶斯聚合得到带不确定性的Consistent Alignment Magnitude（CAM）得分；③引入参考集过滤（RSF）剔除普遍性神经元，聚焦域特定行为；④实现线性时间复杂度，显著降低计算成本。

**🔧 技术方法**

技术手段：残差方向对齐（RDA）、Normal-Inverse-Gamma贝叶斯推理、置信下界CAM、参考集过滤RSF、前向统计评估、神经元抑制干预与分布度量（PPL、KL）评估。

**📊 数据集**

使用数据集：Qwen3-4B-it、OLMo-3.1-32B-it、Llama-3.1-8B-it；目标域样本集包括 MBPP+（代码）、HumanEval+（代码），MATH-500（数学推理）、AMC（数学 OOD），PyComp-1K（Python comprehension），参考集 WikiText-2；评测基准包括 MMLU-Redux、GPQA 等。

**📈 对比分析**

与梯度基线（GxAct、AttnLRP）以及激活均值、Empirical Mean、Empirical SNR、Neg.CAM 等对比，RACE 在目标域抑制时产生更大性能下降、ISI 指标更高，同时在非目标域保持较好性能；在算力方面，RACE 额外 FLOPs 仅 0.63× 前向推理，而梯度基线约 144×，显示出显著的效率优势。

**⚠️ 局限性**

局限性：①依赖线性残差方向对齐，可能忽略多神经元非线性协同和多重语义特征；②对注意力模块的效果有限，识别一致性神经元不如 MLP 模块显著；③在样本量不足时，贝叶斯估计的效果受限；④参考集过滤需要额外的对照数据，增加实验复杂度。

---

## 542. TorchMorph: CUDA-accelerated Morphological Transforms

**arXiv ID:** 2608.24738 | [PDF](https://arxiv.org/pdf/2608.24738v1)

**作者:** Kai Zhao `[一作]` `[通讯]` (Shanghai University), Kai Zhao (Shanghai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并实现了 TorchMorph——一个面向 PyTorch 的 GPU 批处理、N 维形态学、精确距离变换与熵正则化最优传输的统一库。

**💡 创新点**

创新点在于：①将经典形态学与距离变换实现为单次 CUDA 启动的融合核；②支持高达八维空间、批处理和多种边界模式；③提供完全一致的 API 兼容 SciPy；④实现了可微分的 Sinkhorn 迭代，支持批量、log‑domain 与高精度计算。

**🔧 技术方法**

核心技术包括：CUDA 流与图（CUDA‑graph）重放；低阶内存优化（共享内存、寄存器使用）；分离式下包络法实现 Euclidean DT；多线程块按扫描线划分；对 Sinkhorn 的行向量分块加速与一次性 log‑sum‑exp；Python 层统一参数、结构元素解析与错误检查。

**📊 数据集**

实验使用合成图像（2D、3D 及更高维）与网格点集合，规模从 256² 到 128³ 甚至 32² 网格的 1000 迭代 Sinkhorn；不依赖公开数据集，仅用于基准对比。

**📈 对比分析**

与 SciPy（CPU 单核）和 POT（CPU）对比：数值误差在 10⁻⁶ 级别，欧几里得 DT 与拓扑变换几乎完全一致；吞吐量方面，批量化显著提升，典型速度提升 10–40 倍（如 256² 灰度膨胀 10×，Sinkhorn 22×）；单输入时 GPU 开销占比高，批量化后接近饱和。

**⚠️ 局限性**

限制：形态学与距离变换目前仅正向计算，缺乏可微分梯度；仅在 float32 精度下工作；需要 CUDA 设备，CPU 后备有限；缺少连通分量、重建等高级形态学操作。

---

## 543. Parameter-Efficient Self-Supervised Adaptation for EEG-FM under Fixed Computational Budgets

**arXiv ID:** 2608.24727 | [PDF](https://arxiv.org/pdf/2608.24727v1)

**作者:** Meghal Dani `[一作]` (University of Tübingen), Stefanie Liebe `[通讯]` (University Clinic Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

针对 EEG 基础模型（EEG‑FM）提出一种参数高效的自监督适配方法，仅更新 9% 的参数，显著提升下游任务性能。

**💡 创新点**

创新点在于：①仅对最终编码层进行自监督微调即可实现跨域适配；②在固定计算预算下证明适配性能仅需 20–50% 的未标注数据；③发现总窗口数决定性能，而患者数量对性能影响不大。

**🔧 技术方法**

采用自监督学习（BIOT 的对比学习、CBraMod 的掩码重构）对最终编码层进行微调，随后使用线性探测评估；实验中使用了固定计算预算协议来控制数据量和训练次数。

**📊 数据集**

使用了三大临床 EEG 数据集：TUAB（异常检测）、TUEV（事件分类）和 CHB‑MIT（癫痫发作检测），覆盖了分布内（ID）与分布外（OOD）场景。

**📈 对比分析**

与仅进行线性探测（LP）基线相比，参数高效自监督适配在所有任务上均取得提升：最大 AUCROC 提升 +36.2 点，AUCPR 提升至 31.5%（相较于 1.5% 的随机水平），并且在固定计算预算下，性能在 20–50% 数据量时已达到峰值。

**⚠️ 局限性**

局限性包括：①仅对单一层微调可能在某些任务下不够充分；②对 CHB‑MIT 的患者数探究仅覆盖 3–22 位患者，结果可能不具普遍性；③实验仅验证了两种预训练目标，未检验在更广泛时间序列模型或其他医学数据上的适用性。

---

## 544. Enhancing Bayesian Optimization and Active Learning Through Kernel Diversity

**arXiv ID:** 2608.24721 | [PDF](https://arxiv.org/pdf/2608.24721v1)

**作者:** Heng Zhang `[一作]` (University of Georgia), Tara Javidi `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种统一的框架KENDO，通过集成多核高斯过程（EGP）与基于不一致性的采样策略，既可用于贝叶斯优化（BO），也可用于贝叶斯主动学习（AL），并扩展到多目标优化（MOBO）。

**💡 创新点**

核心创新在于将传统贝叶斯方法中的高维、耗时的MCMC采样替换为离散核集合与自适应贝叶斯加权，并设计了针对“优化器条件”与“模型不一致性”的采样函数，使得在保持不确定性量化的同时显著降低计算开销。

**🔧 技术方法**

使用技术包括：多核高斯过程（EGP）与贝叶斯模型平均；基于Hellinger距离的对抗式采样；随机标量化以维持单点条件；以及对高斯混合后验的矩匹配近似。

**📊 数据集**

在单目标、双目标、三目标等多种公开基准上进行实验，包括Branin、Rosenbrock、Hartmann、ZDT2、DTLZ2、VehicleSafety、CarSideImpact、Penicillin、LCBench、Higdon、Ishigami、RobotPush、GBT-HPO、Airfoil等；还涉及真实的工程与超参数调优数据集。

**📈 对比分析**

与多种基准方法（NEI、PES、MES、JES、SCoreBO、EGP-TS、MESMO、qNEHVI、SAL、BALD、BQBC、QBMGP等）对比，KENDO-BO在所有单目标和多目标任务上均达到或优于最优基准，并在计算时间上比MCMC方法快3–5倍；KENDO-AL在主动学习任务上取得最小负对数似然，速度提升27倍。

**⚠️ 局限性**

局限性包括：使用矩匹配近似可能低估不确定性，特别是不同核预测差异较大时；核字典需手工指定，缺乏自动构造机制；对高维、复杂约束问题的泛化尚待验证。

---

## 545. Lost in Speech: Trilingual Spoken Hallucination Detection Across Audio and Transcripts

**arXiv ID:** 2608.24707 | [PDF](https://arxiv.org/pdf/2608.24707v1)

**作者:** Meruyert Aristombayeva `[一作]` (Satbayev University), Dongwon Lee `[通讯]` (Pennsylvania State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

创建了一个多语言、多模态的语音幻觉检测基准，包含 12,013 篇英文、俄语、哈萨克语新闻样本及其对应的三种幻觉类型与严重级别的合成文本、语音和 ASR 转录，并补充了 290 条真实世界的假新闻数据；

**💡 创新点**

首次将多语种（英语、俄语、哈萨克语）与多模态（原始文本、合成语音、ASR 转录）相结合的幻觉检测基准引入，且通过真实世界假新闻和人类写作的真负样本，解决了合成数据中 provenance 与标签相关的偏倚；

**🔧 技术方法**

使用 LLM（如 GPT‑4、Claude、DeepSeek）生成幻觉文本并自评、外部评判；采用 TTS（Coqui XTTS‑v2、Silero、开源哈萨克 TTS）合成语音，Whisper‑large‑v3 与 wav2vec2‑large‑Kazakh 进行 ASR；对模型进行 fine‑tune 的多语种编码器（XLM‑R、mDeBERTa、ReMBERT）与零样本多模态解码器（Qwen2、Gemma、Step‑Audio 等）进行评估；

**📊 数据集**

合成数据集 12,013 篇新闻（原文 + 3 种幻觉类型 × 3 严重级别），真实世界数据 290 条俄语/哈萨克语假新闻（来自 factcheck.kz）及其双语翻译；此外还包含对应的 ASR transcripts、合成音频等；

**📈 对比分析**

通过对比原始文本 vs ASR 转录、音频 vs transcript，比较 fine‑tuned 编码器与零样本解码器在二分类、类型与严重度任务上的 accuracy 与 macro‑F1；结果显示文本检测优于音频，ASR 噪声导致哈萨克语性能显著下滑；在真实世界假新闻上，合成训练模型迁移良好，宏 F1 达 0.82‑0.88；同时发现模型对 provenance 的敏感性会影响评估；

**⚠️ 局限性**

局限包括：仅使用 TTS‑ASR 合成的朗读语音，未覆盖自然对话或录音；W er 与 CER 的跨语言可比性受形态学影响；仅覆盖三种语言与单一新闻域；LLM 生成幻觉可能带来模型特有偏差；语音评估仅针对哈萨克语部分；真实世界样本数量不均衡且为非随机样本；零样本解码器未进行 fine‑tune；并且 provenance 与 veracity 的相关性仍未完全消除。

---

## 546. Image Difference Quantification Using Autoencoder-Based Latent Representations

**arXiv ID:** 2608.24782 | [PDF](https://arxiv.org/pdf/2608.24782v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 547. LION: A Clifford Neural Paradigm for Multimodal-Attributed Graph Learning

**arXiv ID:** 2608.24795 | [PDF](https://arxiv.org/pdf/2608.24795v1)

**作者:** Xunkai Li `[一作]` (Beijing Institute of Technology), Guoren Wang `[通讯]` (Beijing Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种新型的多模态属性图（MAG）神经范式 LION，采用 Clifford 代数构建几何流形，实现“对齐→融合”的两阶段传播与聚合流程。

**💡 创新点**

创新点包括：① 用 Clifford 代数把多模态特征映射到几何流形，并通过几何势与空间旋转实现拓扑感知的高阶传播（CGP）；② 通过能量门控与多尺度共识的全息聚合（AHA）实现自适应模态融合；③ 将传播与聚合完全分离，传播无参数、一次性缓存，显著提升训练效率。

**🔧 技术方法**

核心技术：Clifford 代数、几何乘积、平行运输、空间旋转、几何势、拓扑感知高阶传播、能量门控、共识驱动的多尺度注意力、全息聚合；实现基于这些算子构建的两阶段网络 LION。

**📊 数据集**

实验使用 9 个公开 MAG 数据集（社交网络 RedditS、电影网络 Movies、推荐网络 Grocery/Sports/Ele-fashion/Cloth、艺术网络 SemArt、图像文本网络 Flickr30k、书籍网络 Goodreads），覆盖 6 个领域，包含文本与图像两种模态。

**📈 对比分析**

与 15+ 传统 GNN（GCN、GAT 等）、早期 MAGNN（MMGCN、MGAT）、现代 LLM‑驱动方法（GraphGPT-O、NTSFormer 等）以及图增强 MAGNN（DMGC、DGF、MIG‑GT、UniGraph2）等基线进行比较。LION 在 3 类图任务（节点分类、链路预测、节点聚类）和 3 类模态任务（检索、图→文本、图→图像）均取得 SOTA，平均提升 5‑10% 以上，且在稀疏、扩展性、收敛速度方面表现更稳健。

**⚠️ 局限性**

局限性：① Clifford 流形维度随模态数指数增长，虽然实际多模态任务通常只有 2‑3 模态；② 该范式对大规模高维模态的内存和计算开销仍较大；③ 理论证明仅在固定传播算子下成立，对动态或学习型传播的收敛性质尚未完全解析；④ 需要预先设计与训练的模态编码器，跨模态迁移时可能受限。

---

## 548. Right Diagnoses, Decorative Reasoning:A Perturbation Audit of Medical Chain-of-Thought

**arXiv ID:** 2608.24790 | [PDF](https://arxiv.org/pdf/2608.24790v1)

**作者:** Mengzhu Xu `[一作]` (Eindhoven University of Technology), Xi Long `[通讯]` (Eindhoven University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对14种医学问答大语言模型进行链式推理（CoT）的可信度审计，使用30个医学相关的扰动算子并评估链与答案是否耦合。

**💡 创新点**

提出医学专用的“链耦合率”（Chain‑Decoupling Rate）和一套包含临床危害、种族公平等指标的评估框架，首次将临床医生的标注作为基准来衡量CoT的可信度。

**🔧 技术方法**

构建了基于链编辑（F‑block）和问题编辑（M‑block）的扰动流程，配合链更新 × 答案翻转联合分析，并采用LLM判别器对链是否“注册”编辑进行语义判定。

**📊 数据集**

使用四个医学多选问答基准（MedQA、MedMCQA、PubMedQA、Medical MMLU）以及公开的开源模型和闭源模型。

**📈 对比分析**

对比模型的原始准确率、链耦合率、答案翻转率等指标，结果显示：链耦合率平均72.9%，链扰动不影响准确率，CoT提示并不优于直接答复；临床医生评估显示98.5%扰动保持金标准，13.3%为统一认为有临床危害的翻转。

**⚠️ 局限性**

局限性包括：仅在多选题上评估，链更新判定主要基于词表匹配；闭源模型缺乏链文本；扰动算子覆盖范围有限（仅二元性别和年龄），未包含种族/民族；基准可能被模型预训练泄漏；链长度不足的模型（如BioMistral、OpenBioLLM）对扰动响应不充分。

---

## 549. MoE-based Feature Adapter for Prompt-free Binary Coronary Artery Segmentation in X-ray Angiography

**arXiv ID:** 2608.24783 | [PDF](https://arxiv.org/pdf/2608.24783v1)

**作者:** Lin Xi `[一作]` (University College London), Yingliang Ma `[通讯]` (University of East Anglia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种无提示的混合专家（MoE）特征适配器，用于X射线血管造影视频的冠状动脉二值分割

**💡 创新点**

将AdaptFormer轻量级适配器扩展为多专家结构，并引入稀疏top‑k路由，使特征适配可根据局部图像特征动态选择专家，从而更好地处理低对比、细小分支和干扰背景

**🔧 技术方法**

基于Vision Transformer的编码器-解码器框架，使用MoE轻量级适配器、稀疏top‑k路由、Dice+BCE联合损失

**📊 数据集**

在MOSXAV（内部训练/测试）和XACV（外部验证）两个血管造影视频数据集上训练和评估

**📈 对比分析**

与U‑Net、Attention U‑Net、nnU‑Net、nnWNet、AdapterSeg、MaskVSC等基线相比，MoE适配器在MOSXAV上取得最高Dice/IoU、精确率和召回率；在XACV上也实现了最佳Dice/IoU和召回率，显示出更强的跨数据集泛化能力

**⚠️ 局限性**

仅在单帧图像上进行分割，未考虑时间信息；实验中专家数和top‑k固定，可能限制了适配灵活性；在极低对比或噪声极高场景下的鲁棒性仍待进一步验证

---

## 550. Cycle time minimization for the simple assembly line balancing problem under peak power constraints

**arXiv ID:** 2608.24779 | [PDF](https://arxiv.org/pdf/2608.24779v1)

**作者:** Bao Gia Hoang `[一作]` (VNU University of Engineering and Technology), Khanh Van To `[通讯]` (VNU University of Engineering and Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在固定工作站数和峰值功率限制下的简单装配线平衡问题，最小化周期时间。

**💡 创新点**

首次将峰值功率限制与 SALBP‑2 结合，并提出精确的 SAT 模型与两种可复现的功率上限公式。

**🔧 技术方法**

采用可满足性（SAT）模型，三种非重叠约束，增量搜索与最优性证明，并与 Gurobi、CPLEX 的 MIP 与 CP 进行对比。

**📊 数据集**

使用标准 SALBP 基准库的 72 个实例，任务功率随机分布于 5–50 之间。

**📈 对比分析**

与商业 MIP/CP 通过最优周期时间、求解到最优的实例数、平均速度比等指标比较，SAT 在绝大多数实例上得到更优或相同周期时间，解决更多实例到最优，平均求解速度约 30–100 倍更快。

**⚠️ 局限性**

仅考虑固定任务功率、离散时间、固定工作站数的情形，未涵盖时变功率、连续时间、可变工作站等更一般情况。

---

## 551. One-Shot Learning from Demonstration of Contact-Rich Robotic Manipulation by Identifying Physical Interactions

**arXiv ID:** 2608.24741 | [PDF](https://arxiv.org/pdf/2608.24741v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 552. Federated Sharing and Continuous Improvement of Medical Device Knowledge Artifacts: A Conceptual Model

**arXiv ID:** 2608.24761 | [PDF](https://arxiv.org/pdf/2608.24761v1)

**作者:** J. C. Mariscal-Melgar `[一作]`, Tobias Redlich `[通讯]` (Helmut-Schmidt University University of Federal Armed Forces Hamburg)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种以知识工件为中心的联邦共享与持续改进框架，整合点对点制造与跨机构的知识治理；

**💡 创新点**

首次将版本化、审查决策与使用记录结合，构建完整的工件生命周期，并强调开放源硬件许可对跨站改进的必要性；

**🔧 技术方法**

基于文献综合与概念模型设计，利用版本控制、可追溯性、开放源硬件许可、联邦数据交换技术；

**📊 数据集**

未使用实验数据集，而是基于对910条文献的系统检索、240篇映射与72篇深读，构成理论依据；

**📈 对比分析**

缺乏实验比较，本文仅提出评估框架和指标（如版本链完整性、审查工作量），未给出性能数值；

**⚠️ 局限性**

主要局限在于未经过实证验证，假设开放源硬件许可可普及且不涉及监管审批，且对实际部署所需技术细节与可操作性研究不足。

---

## 553. Weakly Supervised Seafloor Segmentation for Seagrass Habitat Mapping in Side-Scan Sonar Imagery

**arXiv ID:** 2608.24756 | [PDF](https://arxiv.org/pdf/2608.24756v1)

**作者:** Hayat Rajani `[一作]` (University of Girona), Rafael Garcia `[通讯]` (University of Girona)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出了一种弱监督侧扫声纳底栖栖息地分割方法，仅用图像级标签训练像素级语义分割。

**💡 创新点**

创新点在于将ViT编码器-解码器与密集条件随机场相结合，针对声学图像的噪声与弱边界进行伪标签细化，并采用Lovász-Softmax损失直接优化mIoU。

**🔧 技术方法**

使用了ViT编码器、卷积解码器、CAM、dCRF、Focal/Lovász-Softmax损失、EsViT自监督预训练等技术。

**📊 数据集**

采用BenthiCat数据集，其中包含约100万张未标记侧扫声纳图像用于自监督预训练，约36,000张带像素级分割的标注图像用于评估。

**📈 对比分析**

与完全监督模型对比，弱监督模型在保留图像级标签的前提下实现了87.6% mIoU，伪标签精度为89.3%，自监督预训练进一步提升约3%，表现与全监督模型相当且推理速度满足实时需求。

**⚠️ 局限性**

主要局限在于对多类样本极度不平衡时伪标签稀疏、阴影和盲区误判导致的分类误差，以及对不同声纳配置的泛化仍需进一步验证。

---

## 554. SkillForge: Evolving Verifiable Skills for Reinforcement Learning Agents

**arXiv ID:** 2608.24747 | [PDF](https://arxiv.org/pdf/2608.24747v1)

**作者:** Shidong Yang `[一作]` (Alibaba Group), Xiangxiang Chu `[通讯]` (Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SkillForge框架，实现LLM代理在强化学习中的连续技能演进与验证

**💡 创新点**

将技能调用显式化，使RL直接优化环境动作与技能调用；引入基于证据的技能验证和多路径技能诱导；实现技能库的持续增长与质量控制

**🔧 技术方法**

强化学习（GRPO），嵌入检索，LLM生成/修订技能（教师模型），经验抽象与摘要

**📊 数据集**

ALFWorld、WebShop、AppWorld三大开放式任务集

**📈 对比分析**

与ReAct、Reflexion、Mem0、SkillRL等基线对比；在所有基准上SkillForge均超越基线，提升约3.7–10.3个百分点，且在AppWorld的SGC提升近三倍

**⚠️ 局限性**

依赖教师LLM的技能生成与修订质量；技能库随训练持续增长可能导致检索开销；显式调用会增加提示长度与推理成本

---

## 555. Polynomial-time Stable Matching in Network Hypergraphs

**arXiv ID:** 2608.24728 | [PDF](https://arxiv.org/pdf/2608.24728v1)

**作者:** Karthekeyan Chandrasekaran `[一作]`, Krishna Kalathur `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文给出了在网络超图偏好系统中寻找稳定匹配的多项式时间算法。

**💡 创新点**

创新点在于将超图稳定匹配问题转化为冲突偏好图的内核求解，并证明在网络超图下该冲突偏好图是 DE 图的团无向化的 clique‑acyclic 超方向，从而利用已知的多项式时间内核算法实现求解。

**🔧 技术方法**

使用的技术包括：超图与偏好系统定义、冲突图与冲突偏好图构造、DE 图（有向树路径-弧交集图）的结构特性、团无向化的 clique‑acyclic 超方向概念以及 Pass‑Lanneau、Igarashi、Meunier 的内核多项式时间算法。

**📊 数据集**

本文为理论论文，没有使用实验数据集，主要依赖图论与多面体理论的理论证明。

**📈 对比分析**

相比以往仅给出存在性证明或在特殊子类（如树子图、单调超图）上的算法，本文在更广泛的网络超图族上完成了完整的多项式时间算法实现，性能上属于理论复杂度分析，未涉及实验评估。

**⚠️ 局限性**

局限性：算法仅适用于网络超图，尚未扩展到所有正则化超图或一般正常超图；Scarf 推枢机过程在网络超图上的多项式时间性仍未知；对带容量的稳定 b‑匹配问题亦未覆盖。

---

## 556. Fiber Bragg Grating Whiskers for Bioinspired Hydrodynamic Perception on Underwater Robots

**arXiv ID:** 2608.24724 | [PDF](https://arxiv.org/pdf/2608.24724v1)

**作者:** Hao Li `[一作]` (Stanford University), Mark Cutkosky `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2`

**🎯 论文内容**

设计并验证了一种基于光纤布拉格光栅（FBG）的仿生水下触须传感器，并在小型ROV上实现了延迟尾迹的分支选择。

**💡 创新点**

创新点是将仿生的非均匀锥形椭圆截面与NiTi超弹性芯结合，形成可多点、低噪声、可复用的触须模块，同时首次在机器人上实现利用尾迹信息进行决策。

**🔧 技术方法**

使用技术包括FBG传感、NiTi超弹性芯、聚氨酯外壳、三轴推拉实验、尾迹映射、逻辑回归分类器以及光纤多路复用。

**📊 数据集**

数据集来自水池中的推拉实验、旋翼桨叶尾迹扫描、ROV单触须分支决策实验（共20次试验，17次正确）以及海域部署中捕捉潜水员尾迹的现场数据。

**📈 对比分析**

与圆柱基准及先前传感器相比，触须在自振抑制、流速灵敏度（-183.47v²-12.06v+0.56 pm）、分支选择准确率（85%）等方面表现更优。

**⚠️ 局限性**

局限性包括仅使用单个触须、实验环境受限于水池、尾迹跟踪仅为二元分支选择、未考虑自体运动耦合及三维环境、样本量有限导致模型可能过拟合。

---

## 557. ''You Can't Open an LLM With a Screwdriver'': The De-Democratization of Software

**arXiv ID:** 2608.24720 | [PDF](https://arxiv.org/pdf/2608.24720v1)

**作者:** Zixuan Feng `[一作]` (Virginia Commonwealth University), Anita Sarma `[通讯]` (Oregon State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对一场专家小组讨论的转写进行定性主题分析，本文探讨了生成式AI对软件工程实践、教育、生态与治理的影响，提出软件工程正从直接编码转向人类主导的指挥与治理；

**💡 创新点**

创新点在于挑战“软件工程即将终结”的观点，强调AI并未消除专业技术，而是将软件工程师的核心技能从写代码转移到意图规范、评估行为、系统整合与治理，进而为未来教育与工具研究提出了新的方向；

**🔧 技术方法**

主要使用了专家访谈与定性分析技术（主题映射、叙事分析），以及对比历史技术抽象浪潮的文献梳理；

**📊 数据集**

研究数据来自单场专家小组（4名行业/学术专家）的讨论录音转写；

**📈 对比分析**

本文未进行实验性性能比较，而是通过主题分析得出四大主题（人类工作与角色、教育与专业、生态治理、伦理与不确定性），并将其与过去的软件工程抽象化历史进行概念性对比；

**⚠️ 局限性**

局限性包括：仅基于单一小组讨论，缺乏统计可推广性；缺少来自学生、政策制定者等视角的输入；研究缺乏量化实验验证，仅提供概念性和理论性分析。

---

## 558. Lifted Model Construction under Approximate Commutativity

**arXiv ID:** 2608.24713 | [PDF](https://arxiv.org/pdf/2608.24713v1)

**作者:** Malte Luttermann `[一作]` (University of Hamburg), Ralf Möller `[通讯]` (University of Hamburg)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出ε‑commutativity概念，对因子中近似可交换的参数进行检测与压缩，集成到acp算法中实现近似lifted模型构造与推理。

**💡 创新点**

创新点在于将对称性容差从严格等价扩展为ε‑等价，给出严格误差上界并证明其可通过算术平均实现最优压缩；同时揭示ε‑commutativity不对并集封闭，需重新设计子集检索方法。

**🔧 技术方法**

技术包括ε‑等价关系与对称群操作理论、改进的acp构造算法、代表值压缩（均值法）、对称距离度量D_CD、实验验证等。

**📊 数据集**

实验使用与原acp论文相同的合成因子图，包含布尔变量2d+1到d·⌊log₂d⌋+2个，k=1,3,7个ε‑commutative因子，d∈{2,4,8,12,16,20}，ε∈{0.001,0.01,0.1}。

**📈 对比分析**

与原acp算法在同一模型上比较，acp ±ε在lifted推理时实现指数级速度提升，查询结果p' / p高度集中在1附近，误差低于理论上限，几乎无精度损失。

**⚠️ 局限性**

局限在于目前仅适用于离散变量，crv限制；对连续变量的ε‑commutativity尚未处理；ε‑commutativity检测算法仍需进一步提升效率。

---

## 559. Automatic Model Card Generation Using an LLM

**arXiv ID:** 2608.24807 | [PDF](https://arxiv.org/pdf/2608.24807v1)

**作者:** Tajkia Rahman Toma `[一作]` (University of Alberta), Cor-Paul Bezemer `[通讯]` (University of Alberta)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了两种基于大型语言模型（LLM）的工具，分别用于对现有模型卡进行标准化重组和从模型仓库文件自动生成模型卡。

**💡 创新点**

创新点在于：①利用LLM将非结构化的模型卡内容映射到统一模板，显著提升结构一致性；②通过解析模型仓库内的论文、配置、tokenizer等文件，自动构建完整、可比的模型卡。

**🔧 技术方法**

技术包括：使用开放源码LLM（如GPT-4）在零温度下进行模板重组与内容生成；利用信息清单（checklist）评估重组完整性；采用语义相似度、事实正确性与稳定性评估。

**📊 数据集**

数据集为48份经过人工挑选、质量控制的 Hugging Face 基础模型卡，涵盖模型描述、训练数据、伦理风险等信息，并结合对应的模型仓库文件。

**📈 对比分析**

对比方法：对重组版与原版模型卡的内容保持率（median 93.8%）和信息置放正确率（仅1.9%错误）；对生成模型卡的语义相似度（平均≈0.9）和事实正确率（54.17%完全正确）；稳定性评估显示重组结果在多次运行中语义相似度均≥0.97。

**⚠️ 局限性**

局限性包括：LLM在解释性或推理性段落的准确度不足；依赖高质量论文或其他资源，缺失此类资源时生成质量下降；评估主要基于少量高质量样本，外部可推广性尚待验证。

---

## 560. CAFE: Self-Improving Search Agents Need Co-Evolving Feedback

**arXiv ID:** 2608.24794 | [PDF](https://arxiv.org/pdf/2608.24794v1)

**作者:** Boyang Liu `[一作]` (Fudan University), Xuanjing Huang `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在搜索问答任务中提出了一种名为CAFE的框架，使代理模型同时学习请求并使用反馈，同时生成可用于纠错的反馈。

**💡 创新点**

创新点是将代理与反馈生成器视为同一共享模型，采用在线RL与离线滚动推断优化交替迭代，让反馈与策略随时间共演进，并提出比较反馈估计(CFE)、优势塑形、滚动推断偏好优化(RDPO)等技术。

**🔧 技术方法**

使用强化学习（GRPO）、比较反馈估计、优势塑形、滚动推断偏好优化、共享参数的角色条件模型、SFT预训练等技术。

**📊 数据集**

在七个Agentic SearchQA基准（2Wiki、HotpotQA、MuSiQue、PopQA、Bamboogle、Natural Questions、TriviaQA）以及BrowseComp-Plus上进行实验。

**📈 对比分析**

与多种基线（封闭源LLM、开放源LLM、RL搜索代理）对比，7B规模下平均EM 52.5、F1 60.7，超越IGPO等最强RL基线，并在六个外域数据集保持优势，回答级误差率降至12.6%。

**⚠️ 局限性**

主要限制在于需要大量的在线与离线回放数据，依赖同一模型共享参数可能在更大规模或不同任务中难以扩展，并且对极端长回合或非检索类任务的适用性尚未验证。

---

## 561. StepGuard: Learning Step-Level Guardrails with Scalable Supervision and Safety-Utility Balancing

**arXiv ID:** 2608.24777 | [PDF](https://arxiv.org/pdf/2608.24777v1)

**作者:** Zhijie Zheng `[一作]` (Shanghai Artificial Intelligence Laboratory), Dongrui Liu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 StepGuard，一种针对 LLM 代理的逐步安全守护模型；

**💡 创新点**

创新点在于：①构建 StepGen 生成前缀对齐的安全/不安全轨迹，提供高质量的逐步监督；②引入 Balance‑GRPO 动态重权重机制，校正安全/不安全的防御偏差；③同时实现预执行检查与轨迹审计；

**🔧 技术方法**

技术包括：大模型微调（Qwen3‑4B‑Instruct 作为基座）、自动化合成数据引擎（StepGen）、对比式数据生成、动态优势重加权的 GRPO（Balance‑GRPO）、多工具仿真与风险标签；

**📊 数据集**

使用 ATBench 风险样本、R‑Judge、ASSE Security、TS‑Bench‑Dojo/Harm 等公开基准，以及在 AgentDojo、AgentDyn、AgentHarm 环境中生成的合成轨迹；

**📈 对比分析**

与 GPT‑5.4、Qwen‑5B‑Guard、LlamaGuard、ProGuard、AgentDoG、ShieldAgent、Safiron、TS‑Guard 等基线比较；在轨迹级别 StepGuard 达到 83%+准确率、84%+ F1，步级别 84.8%准确率、84.1% F1；在 AgentDojo/AgentDyn 上平均 ASR 降至 1.2/9.3，utility 分别提升至 90.7/66.7，误报率仅提升 0.3；

**⚠️ 局限性**

局限性包括：①合成数据可能存在覆盖不足、偏差或注释错误；②安全评估仍受限于现有基准，未覆盖长程、多智能体或自适应攻击；③并非形式化安全保证，仍可能出现误判，部署需配合监控与人力干预。

---

## 562. MoTE: Mixture of Task Experts for Multi-Task Video Understanding

**arXiv ID:** 2608.24763 | [PDF](https://arxiv.org/pdf/2608.24763v1)

**作者:** Muhammad Asad Ali `[一作]` (University of Kaiserslautern-Landau), Didier Stricker `[通讯]` (University of Kaiserslautern-Landau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了MoTE（Mixture of Task Experts）框架，将LLM解码器中的FFN转化为任务专用专家，并保持视觉‑语言背骨共享，实现多任务视频‑语言学习。

**💡 创新点**

创新点在于：①显式任务级路由，采用样本级任务索引选择专家，避免token‑级学习导致任务边界模糊；②共享专家与任务专家共存，既保留共享上下文又实现任务特定变换；③可在不改动共享骨干的前提下添加/移除任务专家，支持模块化扩展。

**🔧 技术方法**

技术手段包括：Transformer解码器改造、稀疏Mixture‑of‑Experts（MoE）、prompt‑conditioned专家选择、dense‑to‑expert复制初始化、LoRA、MLP投影器、视觉编码器等。

**📊 数据集**

使用数据集：COIN（5个程序化视频任务），Ego4D（在线叙述），GLM‑OCR基础模型上添加的SROIE和CORD（收据KIE）。

**📈 对比分析**

与现有VideoLLM基线对比，MoTE在COIN五任务平均精度上达62.9%（高于8B VideoLLM 61.8%），同时仅激活约2B LLM参数；在Ego4D叙述与KIE任务上也保持或提升性能；ablation实验表明任务级路由优于密集激活和token‑级稀疏路由。

**⚠️ 局限性**

局限性：需要任务标签或明确的任务索引，若提示模糊或包含多任务时路由效果有限；每新增任务需增加专家参数，导致模型规模线性增长；跨域迁移受限于共享背骨已学的特征表达。

---

## 563. The RAT: A Unified Bayesian Model for RAG Evaluation

**arXiv ID:** 2608.24753 | [PDF](https://arxiv.org/pdf/2608.24753v1)

**作者:** Pius von Däniken `[一作]` (Zürich University of Applied Sciences), Jan Deriu `[通讯]` (Zürich University of Applied Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个贝叶斯评估框架，对检索增量生成（RAG）系统的检索成功、放弃行为和答案正确性进行联合建模，并通过该框架推断生成器的策略符合度。

**💡 创新点**

核心创新在于：①将检索、放弃和任务成功的条件依赖显式化并因子化；②通过生成器成功的确定性定义，将端到端正确性与生成器行为分离；③分析注释分配问题，揭示检索成功标注对策略符合度估计更具信息量；④扩展模型以加入校准后的LLM‑as‑a‑judge噪声观测，统一人类与自动评估。

**🔧 技术方法**

使用贝叶斯推断（Stan + HMC/NUTS）实现模型；对检索使用BM25、Dense Embedding、Hybrid；生成器采用Apertus 8B、Gemma3 12B、Qwen3.5 9B；利用信息论量化注释策略的信息增益；并在模型中引入噪声观测层。

**📊 数据集**

实验基于KILT基准，选取FEVER、HotpotQA、Natural Questions三大任务，构建共享的1.72M段落检索库；对每个任务随机抽取10k问答样本进行评估。

**📈 对比分析**

在27个RAG配置（3检索×3生成器×3任务）上评估，结果显示即使端到端任务成功率相近，生成器策略符合度差异显著；注释分配实验表明检索标注比任务标注更能降低策略符合度估计误差；LLM‑as‑a‑judge在高FPR下仅带来有限的误差降低。

**⚠️ 局限性**

局限性包括：只处理二元判定而非连续评分；生成器成功仅按固定策略定义，无法覆盖软策略；实验覆盖的任务、生成器规模有限；模型仅适用于单轮检索-生成流水线，未考虑多轮对话、重检索等复杂组件；放弃检测仅通过字符串匹配，未覆盖更细腻的放弃表述。

---

## 564. From Natural Language Requirements to Graphical User Interfaces: Automated Prototyping and Verification with Pretrained Language Models

**arXiv ID:** 2608.24749 | [PDF](https://arxiv.org/pdf/2608.24749v1)

**作者:** Kristian Kolthoff `[一作]` `[通讯]` (Technical University of Clausthal), Kristian Kolthoff (Technical University of Clausthal)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一套基于自然语言需求的图形界面检索与快速原型工具RaWi，并构建了相应的评测金标准。

**💡 创新点**

创新点在于将BERT-LTR模型与prf‑kld查询扩展相结合，提供端到端的自然语言到可编辑GUI的检索与生成流水线，并公开首个高质量金标准。

**🔧 技术方法**

采用TF‑IDF、BM25、词向量NBOW、prf‑kld扩展、SBERT、BERT‑LTR三种损失训练，以及深度学习与信息检索技术。

**📊 数据集**

使用Rico大型Android GUI仓库（约57k GUI）以及通过AMT构造的931条nlr查询/GUI对，进一步挑选出100条查询、20条GUI的金标准。

**📈 对比分析**

通过P@k、nDCG、MRR等IR指标评估，BERT‑LTR模型在所有指标上均优于BM25和SBERT，PRF‑kld在小k下也能提升；在用户实验中相较MockPlus，RaWi在组件正确数、数据多样性等方面显著提升，效能差异显著。

**⚠️ 局限性**

局限包括仅依赖Rico数据集、原型编辑功能受限、生成的原型缺乏整体设计一致性，且评测仅覆盖两类业务场景。

---

## 565. Method, Mind, and Morality: How People Make Sense of Artificial Intelligence

**arXiv ID:** 2608.24748 | [PDF](https://arxiv.org/pdf/2608.24748v1)

**作者:** Jacy Reese Anthis `[一作]` (University of Chicago), James Evans `[通讯]` (University of Chicago)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过大规模文本分析与57次AI从业者访谈，研究人类如何在AI快速崛起背景下进行意义建构与框架争论。

**💡 创新点**

创新点在于将“话语原子”主题模型与框架理论相结合，系统识别出四大认知挑战与三大核心争论（方法、心智、伦理）并阐释其相互作用。

**🔧 技术方法**

采用基于词嵌入的词典学习（discourse atom）+ k‑means + SVD 进行主题建模，并使用开放式与轴向编码对访谈文本进行理论构建。

**📊 数据集**

数据集包括 371,312 篇 2018‑2024 年英文报纸文章与 1,391,195 条英文 Twitter 认证账号推文，约 1000 万条句子。

**📈 对比分析**

本文未进行算法性能对比，重点在于概念框架与定性分析，因而无数值指标或基准可提供。

**⚠️ 局限性**

局限性：仅使用英文文本与访谈，样本非代表性，无法进行因果推断，结果主要为探索性与解释性。

---

## 566. $(\text{DNN})^2$: Doubly Non-Negative Relaxations for Deep Neural Networks

**arXiv ID:** 2608.24743 | [PDF](https://arxiv.org/pdf/2608.24743v1)

**作者:** Hanna Jiamei Zhang `[一作]` (Northeastern University), David M. Rosen `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd`

**🎯 论文内容**

本文提出一种基于Burer–Monteiro（BM）低秩分解的DNN（0‑SOS）松弛求解框架（即(DNN)^2），并设计了一种新的特征值最大化方法来在LICQ失效时为BM-DNN求解结果提供全局最优性证明。

**💡 创新点**

创新点在于①将低秩BM分解应用于DNN松弛以显著降低变量维度；②提出针对DNN的非唯一KKT乘子空间进行特征值搜索的全局可证技术；③通过实验验证该方法在保持紧致性的同时实现可扩展的计算效率。

**🔧 技术方法**

使用的主要技术包括：ReLU网络的二次互补性约束建模；DNN（0‑SOS）松弛的Shor与CP约束框架；BM分解与非线性规划求解；特征值最大化的凸非光滑SDP搜索用于全局可证；Julia/JuMP实现。

**📊 数据集**

实验采用随机生成的全连接ReLU网络（权重、偏置均从[-1,1]均匀抽取），输入集为[-1,0.1]^2，输出集为[0,∞)。未使用公开数据集，而是通过随机网络实例模拟验证情形。

**📈 对比分析**

与传统c‑SDP、DNN松弛以及MILP基准（真实最优）进行比较。实验显示(DNN)^2在95%查询中恢复全局最优，并且相较c‑SDP提供更紧的上界；在大规模网络（宽度/深度增大）时仍保持约10^-4的误差；计算时间随网络规模线性增长，而c‑SDP的内点方法呈立方增长。

**⚠️ 局限性**

局限性包括：①全局可证步骤本身需解一次SDP，若求解失败会导致无法确认最优性；②在极少数查询中，证书搜索未能找到满足S≽0的乘子，原因是求解器精度或时间限制；③目前仅在随机小网络上验证，尚未验证在真实训练网络上DNN松弛的紧致性和可扩展性；④缺乏自适应秩提升策略，无法保证在所有实例下均能获得全局最优解。

---

## 567. Interpretable Fundus Image Classification via Ring-Based Retinal Vasculature Features

**arXiv ID:** 2608.24723 | [PDF](https://arxiv.org/pdf/2608.24723v1)

**作者:** Xiaoyan Li `[一作]` (University of Toronto), Huaxiong Huang `[通讯]` (University of Toronto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计了一种以视盘为中心的环形视网膜血管特征表示，提取血管几何、颜色、氧合相关外观及血管-背景熵等多种生理学指标，并通过逻辑回归进行三类（正常、糖尿病视网膜病变、青光眼）图像分类。

**💡 创新点**

创新点在于：①将多种血管相关特征按视盘周围的同心环组织，捕获不同视网膜弧度的空间变异；②结合氧合相关光学密度近似（SO₂ 代理），使得模型兼顾解剖学与功能学；③通过可解释的环级证据映射，显式展示模型预测与临床可观察特征的对应关系；④系统评估了深度预训练模型对图像获取特征的敏感性。

**🔧 技术方法**

技术包括：RIP‑AV 血管分割、光学密度理论与多通道红绿比特征提取、局部熵分析、环形特征聚合与标准化、弹性网络正则化逻辑回归分类器；与 RETFound、ResNet‑50、ViT‑B/16、ConvNeXt‑B 等预训练模型做对比；并对 FOV 标准化、背景扰动等实验进行控制。

**📊 数据集**

使用公开数据集 HRF（45张）、FIVES（800张高质量子集）和 SUSTech‑SYSU（1016张）进行评估，分别覆盖正常、糖尿病视网膜病变与青光眼三类。

**📈 对比分析**

与 RETFound 和其他 ImageNet 预训练模型对比，环形特征在 HRF 上实现 91.1% 准确率（与 RETFound 相当，且在使用自动分割时仍高于其他基线）；在 FIVES 上获得 76–83% 准确率，优于大多数基线；在 SUSTech‑SYSU 上达到 94–96% 准确率。实验显示，深度模型对 FOV、背景色彩等获取相关特征敏感，而环形特征更具稳健性与可解释性。

**⚠️ 局限性**

局限性包括：①对血管分割质量高度依赖，低质量或模糊图像会显著下降；②SO₂ 代理尚未校准为真实氧合度，受相机光谱、照明等影响；③缺乏跨机构、多设备和更大样本量的外部验证；④实现未针对实时部署优化，主要用于可复现性研究；⑤未能完全排除非血管背景信息对预训练模型的影响。

---

## 568. GaussianWAM: Distilling Geometry and Semantics from 3D Gaussian Fields into World-Action Models

**arXiv ID:** 2608.24714 | [PDF](https://arxiv.org/pdf/2608.24714v1)

**作者:** Zijian Zhang `[一作]` (University of Chinese Academy of Sciences), Haibao Yu `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过在训练阶段使用3D高斯场对几何与语义知识进行统一蒸馏，提升世界-动作模型的视觉表示。

**💡 创新点**

将几何与语义信息统一映射到同一3D高斯字段，并在推理时完全移除教师模块，保持原有推理效率。

**🔧 技术方法**

3D高斯场、VGGT-Omega、CLIP、差分渲染、Transformer/DiT架构、蒸馏损失。

**📊 数据集**

LIBERO、LIBERO-Plus、RoboTwin、真实机器人抓取/放置任务。

**📈 对比分析**

在FastWAM和Cosmos Policy上引入GaussianWAM后，整体成功率提升约1–2%，在视角、光照、噪声等分布偏移上提升显著（如+29%）。

**⚠️ 局限性**

需要外部基础模型与多视角数据，蒸馏过程耗时且在极端几何变化或单视角下尚未充分验证。

---

## 569. ECO-COMM: An Ultra Low-Latency Event Camera based Optical Communication System

**arXiv ID:** 2608.24705 | [PDF](https://arxiv.org/pdf/2608.24705v1)

**作者:** Chengling Xu `[一作]` (University of Wisconsin-Madison), Feng Ye `[通讯]` (University of Wisconsin-Madison)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 ECO-COMM，一种基于事件相机的光学通信系统，旨在实现超低延迟的设备关联和轻量级信息交换。

**💡 创新点**

创新点在于针对事件相机硬件产生的时间戳不一致、读出争用、尾随效应和不可避免的不可恢复期（IRP）提出一系列硬件感知的补偿与消除技术，并引入LED间时间偏移、事件聚合和起始符号定位方案，实现 10 µs 级设备关联、100 µs 级符号延迟。

**🔧 技术方法**

使用的技术包括：FPGA 控制的多LED发射器、LUCID Triton2 事件相机、基于微秒级时间戳的事件聚合、时空调制编码、基于开始符号的定位与同步、以及多种事件过滤与时间补偿算法。

**📊 数据集**

数据集：实验使用 32 字节随机数据和 32 字节全 1（0xFF）负载进行 BER 与延迟评估，未使用公开标准数据集。

**📈 对比分析**

方法对比：通过对比不同传输速率下的 BER、关联延迟和端到端延迟，ECO‑COMM 在 8 k‑12 k bps 速率下实现 BER ≤0.1% 且 E2E 延迟 ≤10 ms；在 32 字节负载下，设备关联可低于 15 µs，符号延迟约 100 µs，显示出相较传统基于帧的光通信在时延上的显著优势。

**⚠️ 局限性**

限制：仅验证单链路性能，缺乏多设备、多链路协调与网络层协议；事件处理目前为离线解析，实时实现尚未完成；受限于相机读出带宽与 IRP，极高调制速率仍受限；系统对距离、光照与LED数量敏感，需要进一步优化。

---

## 570. Improving Cross-Problem Vehicle Routing with Locally Augmented Preferences and Representation Disentanglement

**arXiv ID:** 2608.24859 | [PDF](https://arxiv.org/pdf/2608.24859v1)

**作者:** Arthur Corrêa `[一作]` (University of Coimbra), Samuel Moniz `[通讯]` (University of Coimbra)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 POLAR 训练算法与 PLE 编码器，联合用于统一多任务车辆路径规划求解器。

**💡 创新点**

POLAR 在偏好优化中加入局部搜索细化以提升比较信号；PLE 通过共享+任务专家分层拆解共享与约束特定表示，缓解表示混杂。

**🔧 技术方法**

使用 Transformer 编码器+自注意力、RMSNorm、FiLM、门控多专家结构，结合偏好优化与局部搜索，训练采用 AdamW、混合精度和 POMO 多起点解码。

**📊 数据集**

在 48 个 VRP 变体的合成数据集上训练，其中 16 个单枢纽组合用于训练，1k 实例每个变体在 50 与 100 节点规模上测试，另外 32 个变体用于零样本泛化评估。

**📈 对比分析**

与传统求解器 PyVRP、OR‑Tools 以及神经多任务基线（MTPOMO、MVMoE、RouteFinder、CaDA、FiLMMeD、MoSES）比较，本文方法在所有 16 个训练变体上平均 gap 降至 1.19%（n=50）/2.10%（n=100），比最强基线平均降幅约 21%，并在 32 个未见变体中有 27/32 胜出；推理时间与基线相当。

**⚠️ 局限性**

仍难以处理更复杂、真实世界的约束组合，且对大规模实例或更丰富约束扩展的效果尚待验证。

---

## 571. The Optimal Asymptotic Rate of Generalized Covering Codes

**arXiv ID:** 2608.24856 | [PDF](https://arxiv.org/pdf/2608.24856v1)

**作者:** Hengzhuo Li `[一作]` (Xi'an Jiaotong University), Hengjia Wei `[通讯]` (Xi'an Jiaotong University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文确定了广义覆盖码的最优渐近速率，研究了在特定约束下的覆盖码的性质。

**💡 创新点**

创新点在于证明了在产品形式约束下，广义覆盖码的最优渐近速率与线性码的最优速率相同，且这两个约束在渐近上是无成本的。

**🔧 技术方法**

使用了概率论和信息论的工具，包括类型方法、Janson不等式、第二矩方法和结构性变换论证。

**📊 数据集**

未具体提及使用的数据集，但研究涉及的对象是q-元汉明空间中的广义覆盖码。

**📈 对比分析**

通过与已有的覆盖率界限进行比较，证明了在特定条件下，广义覆盖码的最优速率与经典的球覆盖界限一致，且在q为素数幂时，线性码的最优速率也达到了相同的界限。

**⚠️ 局限性**

限制在于未提供具体的广义覆盖码的显式构造方法，且在实际应用中可能需要更多的具体实例来验证理论结果。

---

## 572. Research Methodologies for Cybersecurity in Enterprise Environments: A Narrative Review, Synthesis and Executable Guide

**arXiv ID:** 2608.24850 | [PDF](https://arxiv.org/pdf/2608.24850v1)

**作者:** Tran Duc Le `[一作]` `[通讯]` (University of Wisconsin-Stout Polytechnic), Tran Duc Le (University of Wisconsin-Stout Polytechnic)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对企业网络安全研究中的方法论进行了叙事式综述与综合，提出了11个方法族的分类，并为每个方法族设计了可执行的研究协议与图示，进一步分析了文献中出现的矛盾（如入侵检测算法排名不一致）并给出了对应的解释。

**💡 创新点**

创新点在于：①系统地构建了企业网络安全研究的方法论分类体系；②将每个方法族转化为可执行的协议，包含步骤、工具、评估标准与有效性威胁；③以可视化形式呈现协议流程，便于研究者直接使用；④针对文献矛盾进行了案例分析，强调评估设计对结果的影响；⑤提供了三种整合多方法族的端到端研究模板，并公开了经过注册表验证的研究语料库。

**🔧 技术方法**

使用的技术包括：叙事式与系统叙事混合的综述方法、RAMESES元叙事报告标准、OpenAlex、Semantic Scholar 与 arXiv 等数据库的检索策略、DOI 与 arXiv 记录的验证程序、可执行协议的流程图绘制工具，以及对研究可信度与可复现性的系统化评估。

**📊 数据集**

主要数据集为151篇符合标准的研究论文，所有论文在加入前均通过 DOI / arXiv 记录进行核实，构成了一个已验证的企业安全方法论语料库；该语料库未采用传统机器学习或安全事件数据集，而是基于文献检索和人工筛选。

**📈 对比分析**

通过比较不同方法族的证据强度、评估维度与有效性威胁，作者发现入侵检测算法在不同研究中的排名差异主要由评估设计（实验场景、基准、评估指标等）导致，而非算法本身。此比较说明单一的准确率指标不足以衡量企业环境中的实际性能。

**⚠️ 局限性**

局限性包括：综述并非完整覆盖所有文献，方法族划分可能存在重叠或遗漏；对评估可信度的信心标签为主观判断；全文未完整公开检索记录和查询语句；主要聚焦企业与组织环境，未涉及更广泛的安全技术应用；并且所提出的可执行协议尚未在大型实验中进行广泛验证。

---

## 573. BrowserForge: Scaling Web Episode via Parallel Browser Sandboxes

**arXiv ID:** 2608.24848 | [PDF](https://arxiv.org/pdf/2608.24848v1)

**作者:** Fei Tang `[一作]` (Zhejiang University), Yongliang Shen `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了 BrowserForge 框架，通过在海量浏览器沙盒中并行抓取开放网络网页，生成大规模、跨站点的纯 GUI 交互轨迹，随后对轨迹进行规则+模型清洗并统一链式思路格式，形成 203,238 条高质量轨迹的数据集。

**💡 创新点**

创新点在于：① 以开放网络为来源而非固定站点列表，实现规模与多样性并进；② 采用 Proposer–Solver 双代理循环，将无监督网页转化为可执行任务并记录轨迹；③ 通过规则+模型双重验证与链式思路统一，显著提升数据质量和可训练性。

**🔧 技术方法**

技术包括：Common Crawl URL 采集与清洗、沙盒集群调度、Proposer–Solver 任务合成、Chrome DevTools 交互、基于 Qwen3-VL 的模型推理与验证、Seed‑2.0 Pro 的链式思路重写、Qwen3.5 多模态模型微调。

**📊 数据集**

使用的数据集为 203,238 条来自不同网站的交互轨迹（每条轨迹平均 8.8 步），其中约 30% 通过验证后保留下来，并在此基础上抽样 200K 步用于训练。

**📈 对比分析**

在 Live Online‑Mind2Web 基准上，Fine‑tune Qwen3.5‑4B 的成功率从 25.66% 提升至 33.33%；在静态 Multimodal‑Mind2Web 上，单步准确率从 38.2% 提升至 43.8%。与同等参数量的公开模型相比，BrowserForge 训练的 4B/9B 代理已能超越更大模型与部分商业系统，表明数据质量对性能提升贡献显著。

**⚠️ 局限性**

局限性包括：① 仍需依赖高成本的多浏览器沙盒资源；② 任务生成与验证仍可能产生一定噪声，清洗比例仅约 30%；③ 在极端网站结构或阻断环境（CAPTCHA、访问限制）下表现受限；④ 目前仅针对网页 GUI 交互，未涵盖更复杂的交互模式。

---

## 574. FedV-KGQA: Multi-Hop Question Answering over Vertically Partitioned Knowledge Graphs

**arXiv ID:** 2608.24846 | [PDF](https://arxiv.org/pdf/2608.24846v1)

**作者:** Md Saikat Islam Khan Bappy `[一作]` (Rensselaer Polytechnic Institute), Oshani Seneviratne `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出FedV-KGQA框架，实现在垂直联邦（各方共享实体、分割关系）知识图上进行多跳问答。

**💡 创新点**

创新点在于结合本地图增强、本地知识图嵌入、服务器端实体拼接与主题实体锚定，既不共享原始三元组也能跨分区完成多跳推理。

**🔧 技术方法**

采用本地TransE/DistMult/ComplEx/RotatE等KGE模型、预训练文本编码器（BERT/DistilBERT/RoBERTa）、服务器端实体拼接+MLP投影、主题锚定与余弦相似度排序。

**📊 数据集**

使用MetaQA、PathQuestion、WebQSP三大问答基准数据集。

**📈 对比分析**

与改编的EmbedKGQA、FL-KG-QA、FedE、RelChain等基线对比，FedV-KGQA在所有数据集上均取得最高MRR，接近集中式上限，并在2跳、3跳场景表现优异。

**⚠️ 局限性**

局限性包括仅支持静态图、缺乏正式差分隐私保障，且实体嵌入可能泄露结构信息。

---

## 575. Parameterized Complexity of $L_p$-Lipschitz Constants for Input Convex Neural Networks and $L_p$-Norm Maximization over Zonotopes

**arXiv ID:** 2608.24865 | [PDF](https://arxiv.org/pdf/2608.24865v1)

**作者:** Aritra Das `[一作]` (Ashoka University), Moritz Stargalla `[通讯]` (University of Technology Nuremberg)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

本文研究了输入凸ReLU网络（ICNN）的 L_p‑Lipschitz 常数计算问题，并通过与在生成器表示的多面体（zonotope）上最大化 L_p‑范数的等价问题，证明了对所有固定的 p∈(1,∞) 该问题在参数化复杂度上是 W[1]‑hard 的，并给出基于 Multicolored Clique 的两种不同构造证明，进一步证明在 ETH 下不存在 ρ(d)·N^o(d) 的算法（N 为输入编码长度，d 为维度）。

**💡 创新点**

创新点包括：
- 将 ICNN 的 Lipschitz 常数问题精确映射为 zonotope 的 L_p‑范数最大化；
- 证明对所有固定 p∈(1,∞) 该问题是 W[1]‑hard 的，填补了此前仅对 p=1,∞ 的可解性与 p∉{1,∞} 的未知性空白；
- 提供两套完全不同的证明思路（显式 2k+1 维构造与隐式 2k 维几何构造），展示了构造技术与几何直觉的多样性；
- 通过 Taylor 展开与局部欧氏化方法将 L_2‑范数最大化的硬度迁移到任意固定 p 的情况。

**🔧 技术方法**

主要技术手段包括：
- 参数化复杂度框架与 W[1]‑hardness 减化（Multicolored Clique）；
- 多面体与 zonotope 的几何分析（顶点、支持函数、对称 zonotope 的支持函数构造）；
- 线性程序与极值点、射线枚举用于构造满足特定支持函数值的生成向量；
- Taylor 展开与误差分析将 L_p‑范数的非线性问题局部近似为二次型，从而实现硬度的迁移；
- 量化根号与阈值插值技术用于从 L_p^p‑范数的硬度转移到 L_p‑范数的硬度。

**📊 数据集**

本研究为理论工作，未使用任何实验数据集。

**📈 对比分析**

论文未给出实验或算法实现，仅通过理论证明表明：
- 任何枚举顶点/线性区域的暴力算法（时间 O(n^d)）已是几乎最优的（在 ETH 下不能显著改进）；
- 该结论适用于所有固定 p∈(1,∞)，与 p=1、∞ 的多项式/固定参数可解性形成鲜明对比。

**⚠️ 局限性**

主要局限包括：
- 仍未判定该问题是否属于 W[1]，即是否存在更弱的上界；
- 结果仅针对输入凸 ReLU 网络，尚不清楚是否可推广至更一般的网络结构；
- 证明高度技术化，依赖多面体与几何构造，实际实现与工程应用仍需进一步研究。

---

## 576. Bellman Calibration for Marginalized Importance Weighting in Offline Reinforcement Learning

**arXiv ID:** 2608.24858 | [PDF](https://arxiv.org/pdf/2608.24858v1)

**作者:** Lars van der Laan `[一作]` (Stanford University), Nathan Kallus `[通讯]` (Netflix)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种后处理方法——等距贝尔曼校准（Isotonic Bellman Calibration），通过对已有的占据比估计应用单维非递减变换，纠正残余贝尔曼平衡偏差，并保持预测的排序信息，从而提升离线策略评估的准确性。

**💡 创新点**

创新点在于：1) 将贝尔曼平衡误差转化为可观测的自监督校准问题；2) 设计基于等距回归的无模型后处理框架，既能实现精确的经验贝尔曼平衡，又可提供有限样本的校准与KL风险保证；3) 引入拟合占据比评估（FORE）作为迭代基底，显著提升校准效率。

**🔧 技术方法**

核心技术包括：等距回归、拟合占据比评估（FORE）、对角线/梯度下降求解一维凸优化、统计泛化分析（KL近似、校准误差上界）、以及交叉校准/样本分离策略。

**📊 数据集**

在D4RL（HalfCheetah、Hopper、Walker2d）和InfiniteCartPole数据集上进行实验，使用公开的离线策略评估基准。

**📈 对比分析**

将等距贝尔曼校准与四个基准估计器（DualDICE、MWL、NeuralDICE、NeuralFORE）进行对比。实验表明，校准后在绝大多数任务中显著降低了投影贝尔曼校准误差和绝对策略价值误差（例如D4RL平均误差从2.99降至0.71，CartPole从0.176降至0.110），但在低覆盖（low-ESS）场景下可能略有提升，NeuralFORE是唯一未获益的例外。

**⚠️ 局限性**

局限性包括：1) 对数据覆盖度要求高，低覆盖区可能导致校准失效或产生偏差；2) 需要足够的校准样本以保证统计误差可控；3) 方法仅针对折扣无穷期MDP，有限期或非平稳情境需进一步扩展；4) 对极端值的处理需加以限制（如对零权重做下限约束）。

---

## 577. Reading Is Not Using: Retrieval, Judgment, and the Design of AI Financial Research Workflows

**arXiv ID:** 2608.24842 | [PDF](https://arxiv.org/pdf/2608.24842v1)

**作者:** Miao Liu `[一作]` (Boston College), Zhizhe Liu `[通讯]` (Columbia University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大语言模型（LLM）在财务披露分析中的检索-整合差距，即模型能检索到信息但不影响其投资判断，采用固定信息量、变换无关上下文长度的实验设计；

**💡 创新点**

首次系统性定义并量化检索-整合差距，提出边际决策影响度量；揭示影响渠道（压缩摘要与注意力检索）并验证工作流架构对信息传递的关键作用；

**🔧 技术方法**

利用LLM内部表征探测、稀疏自编码器特征字典、因果干预（黑洞、状态移植）等技术；

**📊 数据集**

基于十二家美国上市公司改写的风险披露段落、相应的中立替代段落，以及二十份真实10‑K披露的删改版；

**📈 对比分析**

对比不同上下文长度、模型规模与工作流架构，结果显示：在短文本下模型影响显著，长文本下影响降至噪声；检索准确性保持稳定；仅“检索-再决策”工作流能将长文本下的影响提升至约8.5个百分点；

**⚠️ 局限性**

局限在于实验仅关注单一决策任务（卖买概率/降级概率），未评估模型自我解释、用户信任；工作流改造需要人工或外部提取，真实系统部署仍需进一步验证；

---

## 578. Next-generation O-RAN Edge: Energy-aware Joint Placement and Migration of Cloud-Native Functions

**arXiv ID:** 2608.24841 | [PDF](https://arxiv.org/pdf/2608.24841v1)

**作者:** Nguyen Phuc Tran `[一作]` (Concordia University), Oscar Delgado `[通讯]` (Ericsson)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种混合整数线性规划（MILP）模型，联合优化O‑RAN边缘云中云原生功能（CNF）的放置与迁移，目标是最小化服务器、传输、唤醒和迁移能耗，同时满足资源容量与单向F1‑U延迟约束。

**💡 创新点**

在单CU‑UP与多CU‑UP两种场景下引入可切换的slice‑aware DU‑CU-UP关联松弛，显著扩展可行放置空间；同时设计了基于k‑means的四阶段确定性启发式算法，能够在无需求解MILP的情况下逼近最优解。

**🔧 技术方法**

采用混合整数线性规划求解器Gurobi实现最优模型；启发式算法结合k‑means++聚类、匈牙利匹配、延迟感知物理映射、能量增量评估与局部可行性修复。

**📊 数据集**

使用蒙特利尔城市5G交通负载轨迹（包含eMBB、uRLLC、mMTC、VoIP四种切片），并在基于CloudSim的三节点边缘云（12台服务器）上进行仿真。

**📈 对比分析**

与单CU‑UP、Multi‑CU‑UP的MILP解以及迁移不感知的放置基线、以及传统Best‑Fit/First‑Fit/Modified‑Best‑Fit/Worst‑Fit等基线进行比较；在24小时负载下，Multi‑CU‑UP相较于单CU‑UP节能约5.7%，启发式算法与MILP差距约9.7%，并在迁移次数与能耗上均优于传统基线。

**⚠️ 局限性**

限制主要在于：①启发式算法在可行性修复阶段仍可能无法全局收敛；②模型未包含RU‑DU传输、控制平面延迟等外部因素；③仅对单一边缘云进行评估，未验证大规模多云场景下的可扩展性。

---

## 579. HORIZON: A Read-Efficient Firmware for DNA Storage with Horizontal Layout

**arXiv ID:** 2608.24839 | [PDF](https://arxiv.org/pdf/2608.24839v1)

**作者:** Alex Sensintaffar `[一作]`, Bingzhe Li `[通讯]`

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提供了IEEE会议论文的模板和排版指导，说明了标题、摘要、章节结构、单位、公式、引用等规范；

**💡 创新点**

创新点在于对IEEEtran.cls文件使用方法及排版细节的系统阐述，帮助作者避免常见排版错误；

**🔧 技术方法**

主要技术为IEEEtran.cls LaTeX 类文件、LaTeX 文本排版语法、文献引用系统BibTeX；

**📊 数据集**

无研究数据集，内容为模板说明；

**📈 对比分析**

没有实验方法或性能评估，主要通过举例说明如何书写和排版；

**⚠️ 局限性**

局限在于该文档仅为模板说明，未包含具体研究内容或实验结果，无法评估科研贡献。

---

## 580. Reliability Limits and Decoding for Partial Nanopore Protein Rereads With Persistent State

**arXiv ID:** 2608.24819 | [PDF](https://arxiv.org/pdf/2608.24819v1)

**作者:** Hongbin Ni `[一作]` (University of Cambridge), Ozgur B. Akan `[通讯]` (University of Cambridge)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `09944146-298c-433e-89df-37255de463d7` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究纳米孔蛋白质测序中部分重读（partial reread）时的信道模型，建立了分子共享、通道本地和观测本地三种状态持久性框架，并分析其渐近可靠性与极限；

**💡 创新点**

创新点包括：①提出分子持久性partial reread的有限字母信道模型，①引入三种持久性级别并比较其对后验与风险的影响；②设计基于order‑b投影的有限记忆接收器，并给出可计算的TV与KL残差上界；③使用LB‑IS无偏重要性采样与精确枚举在L=7处对模型进行验证，并在L=24规模下通过16细胞阶梯实验评估性能；

**🔧 技术方法**

技术手段包括：概率模型构造、最大后验与Bayes风险分析、KL链式规则、条件互信息、order‑b投影、BCJR前向后向递推、LB‑IS自归一化自适应采样、随机森林估计发射概率、校准q参数、统计检验与多重比较校正；

**📊 数据集**

数据集为PASTOR 516条记录（含七种氨基酸类别），通过交叉分层得到的随机森林得到发射概率，校准集使用长度四的已知序列，目标集为长度24，K=10的模拟实验，覆盖率与误差事件均为半合成设置；

**📈 对比分析**

比较方法：在L=7时用精确枚举作为基准，对LB‑IS、order‑4和order‑5接收器分别计算交叉‑NLL、Brier、准确率与TV；在L=24时用高分配的无投影参考对上述指标进行比较。结果显示：order‑4共享分支在16个细胞中NLL比pass‑local低0.033–0.224 nat/残基；但在目标尺度下，order‑4与order‑5均未达到联合参考的一致性；

**⚠️ 局限性**

局限性：仅在分子持久性假设下分析，未使用分子链接的数据；order‑b投影误差随b减小而增大，且在L=24规模下仍无法满足所有一致性指标；对电学相关性、结构化先验、上下文依赖发射的建模缺乏；实验仅基于半合成模型，缺乏真实测序验证。

---

## 581. Auditing Return Conditioning as a Control Knob: An Offline Diagnostic for Decision Transformer Recommendation

**arXiv ID:** 2608.24815 | [PDF](https://arxiv.org/pdf/2608.24815v1)

**作者:** Jingyu Wang `[一作]` `[通讯]` (Independent Researcher), Jingyu Wang (Independent Researcher)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一套离线决策变换器（DT）返回目标（RTG）敏感性审计方法，构造了RTG局部梯度阶梯（逐步替换不同数量的RTG位置），并通过四项诊断（局部与全局RTG影响、无RTG对照、打乱RTG对齐、记录奖励检查）评估RTG对推荐行为的实际影响。

**💡 创新点**

提出了RTG局部梯度阶梯以及完整的四诊断审计框架，用以区分RTG对局部决策的控制效果与对全局上下文的拟合，揭示离线RTG调优往往只能反映整体分布变化而非真正的奖励提升。

**🔧 技术方法**

主要技术包括：Decision Transformer（DT）模型及其无RTG和打乱RTG对照；使用Transformer架构（3层、隐藏128、4头、上下文20）进行动作预测；采用HR@1/HR@5、匹配率/匹配评分等指标；利用岭回归和MLP对RTG可预测性进行评估；构建RTG局部梯度阶梯实验。

**📊 数据集**

使用了两个公开推荐数据集：MovieLens 25M（18种动作/类型）和 MyAnimeList Database 2020（15种动作/类型）。

**📈 对比分析**

通过与DT-no-RTG、打乱RTG对照以及基线SASRec进行比较，发现：在MovieLens中，完整上下文的RTG改写（K=20）导致某些类型的显著份额变化，但局部改写（K=1）几乎无变化，匹配评分亦未提升；在MAL中，无论K取何值几乎无任何影响，整体性能与基线几乎相同。

**⚠️ 局限性**

局限性包括：仅评估基于类型的动作预测，无法推断对实际推荐质量的影响；离线评估与真实用户交互环境脱节；RTG仅在固定窗口内有效，无法验证长期奖励控制；实验仅涉及两类数据集，结果缺乏普适性；未考虑多目标或多级返回目标的复杂性。

---

## 582. Effective Learning Rate Governs Loss Dynamics in Language Model Pretraining

**arXiv ID:** 2608.24814 | [PDF](https://arxiv.org/pdf/2608.24814v1)

**作者:** Zihan Liu `[一作]` (Peking University), Lei Wu `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过匹配学习率与参数范数的比值——有效学习率（ELR），研究并验证了语言模型预训练过程中损失动态的宏观统一规律，揭示了不同的学习率调度和范数控制方法在相同 ELR 下几乎产生相同的损失曲线。

**💡 创新点**

创新点在于提出并实证证明了 ELR collapse：学习率与范数并非独立控制器，而是通过其比值共同决定损失演化；并进一步利用 ELR 统一功能性缩放律（FSL），实现跨范数控制方法的预测迁移，以及用 ELR 解释并调控“延迟加速”现象。

**🔧 技术方法**

主要技术包括：① ELR 定义与匹配实验；② 规范化设计与学习率–范数时间尺度 ablation；③ 权重衰减与 Hyperball 的 ELR 调度对齐；④ 基于 ELR 的功能性缩放律（elr‑FSL）与传统 lr‑FSL 的对比；⑤ 通过预设 ELR 轨迹实现对延迟加速的直接控制。

**📊 数据集**

数据集涵盖 FineWeb、C4、OpenWebText（文本）以及 ImageNet（图像），模型规模从 100M 至 1B 参数，使用 Llama、Qwen3‑MoE、KDA 以及 ViT 等架构；优化器包括 AdamW、Muon、Signum。

**📈 对比分析**

实验对比显示：ELR 匹配后，损失轨迹的平均绝对误差仅为 1–5×10⁻³，低于同一配置下随机种子导致的 1–1.6×10⁻² 波动；在 FSL 迁移实验中，elr‑FSL 在未见 Hyperball 路径上的 RMSE 仅为 0.0212，而 lr‑FSL 则达到 0.2508，误差提升近 12 倍。

**⚠️ 局限性**

局限性在于：① 仅关注损失动态，无法保证参数表示或下游性能的一致性；② ELR collapse 的高精度依赖于规范化设计与学习率–范数实现的时间尺度，缺乏严格的理论解释；③ 该规律在非 Transformer 结构或高度非尺度不变模型中的适用性尚未验证。

---

## 583. From Seeing to Acting: Smart Glasses as First-Person Intelligence Platforms

**arXiv ID:** 2608.24877 | [PDF](https://arxiv.org/pdf/2608.24877v1)

**作者:** Jiangning Zhang `[一作]` (Zhejiang University), Yong Liu `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性综述了智能眼镜的发展历程、正式定义、能力框架与评估蓝图

**💡 创新点**

提出基于硬件、时效、状态持久、动作授权等条件的闭环智能声称，并构建 L0–L5 级别与证据映射

**🔧 技术方法**

通过数据流模型、能力轴归纳、L0–L5 层级框架以及评估蓝图进行系统分析

**📊 数据集**

综合引用了 Ego4D、EPIC‑KITCHENS‑100、Project Aria 等众多公开数据集与基准

**📈 对比分析**

通过将各产品与能力层级对齐，提供可视化证据地图和标准化评估指标，展示不同硬件路线在各级别的可实现度

**⚠️ 局限性**

局限在于快速演进的产品与数据集、对跨平台实现细节的依赖不足，以及对实时部署安全与伦理风险的评估仍待深入

---

## 584. Prompt Structure Redistributes, Not Reduces: An Empirical Analysis of Security-Weaknesses in LLM-Generated Python Code

**arXiv ID:** 2608.24857 | [PDF](https://arxiv.org/pdf/2608.24857v1)

**作者:** Maitreyee Das Urmi `[一作]` (Toronto Metropolitan University), Glaucia Melo `[通讯]` (Toronto Metropolitan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究结构化提示对LLM生成代码安全性的影响，评估GPT‑4o与LLaMA 3.1‑8B在424个安全敏感Python任务上的代码输出。

**💡 创新点**

发现结构化提示能显著降低拒绝率并改变安全弱点的严重度分布，但并未整体降低弱点出现率，且产生安全驱动的语义漂移。

**🔧 技术方法**

使用Bandit和CodeQL静态分析工具，结合不同提示模板（结构化、安全指导、框架引用、对抗意识）对模型输出进行评估。

**📊 数据集**

采用Cybernative.ai的Python安全任务数据集，共424条。

**📈 对比分析**

通过比较五种提示变体的拒绝率、弱点数量、严重度比例和CWE分布，结果显示GPT‑4o在高严重度弱点上下降、低严重度上上升；LLaMA表现不一致；总体性能提升有限。

**⚠️ 局限性**

局限包括仅使用静态分析，单一语言和数据集，模型依赖性强，单次生成缺乏方差估计，未覆盖运行时威胁和多语言场景。

---

## 585. LeFlow: Generative Latent Flow Planning for World Models

**arXiv ID:** 2608.24855 | [PDF](https://arxiv.org/pdf/2608.24855v1)

**作者:** Hsiang-Wei Huang `[一作]` (University of Washington), Jenq-Neng Hwang `[通讯]` (University of Washington)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在冻结的潜在世界模型上学习可复用的潜在轨迹先验，用条件流生成潜在路径并通过逆动力学解码动作，再用模型滚动验证候选。

**💡 创新点**

将规划从在线轨迹优化转化为离线可复用的潜在路径生成，实现一次性规划知识的摊销，并在潜在空间中引入生成式规划与模型验证的组合。

**🔧 技术方法**

使用条件 rectified flow 生成潜在轨迹、逆动力学解码器将潜在转化为动作块、滚动重排序和一致性约束来保证可控性。

**📊 数据集**

在四个目标条件像素控制基准上评估：TwoRoom、PushT、Reacher、OGBench‑Cube。

**📈 对比分析**

与 GCBC、离线 RL、PLDM、LeWM+CEM、iCEM、MPPI 等基线比较，取得 100%–95% 的成功率并比 CEM 低 4–14 倍的规划时间。

**⚠️ 局限性**

局限在于固定短时限，长时限规划误差累积导致性能下降，需进一步研究层次化或更强的世界模型。

---

## 586. A Dual-Dimensional LLM Framework for Automated Item Incidental Content Similarity Analysis in Large-Scale Assessments

**arXiv ID:** 2608.24825 | [PDF](https://arxiv.org/pdf/2608.24825v1)

**作者:** Jing Huang `[一作]` (Purdue University), Hua-Hua Chang `[通讯]` (Purdue University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究构建了一套基于大语言模型的双维度自动题目相似性分析框架，用于识别大规模题库中的偶发内容冗余，并将该框架应用于计算机自适应测试（CAT）的项选择。

**💡 创新点**

创新点在于将结构化分解（表面形式）与语义相关性（深层含义）分离为两个可解释的相似度维度，并通过精心设计的提示让Claude Sonnet 4产生分量化评分，显著优于传统的BLEU和余弦相似度在捕捉心理测量局部依赖方面的表现。

**🔧 技术方法**

使用技术包括：Claude Sonnet 4大语言模型的提示工程；辅助性BLEU和余弦相似度；层次聚类；CAT模拟中基于相似度的聚类约束最大信息量（SC‑MFI）项选择算法。

**📊 数据集**

实验数据集为36条ECR（亲密关系体验量表）简短表格的题目文本，并在51,491名受试者中提取35,278名有效样本；此外通过模拟50次CAT，分别对2,000名考生使用20道题的无约束MFI和不同相似度约束的SC‑MFI进行对比。

**📈 对比分析**

评估方法：与残差相关性进行Spearman相关，比较聚类内分辨率与项目辨别度和阈值分布；CAT模拟评估误差（RMSE）与偏差。结果显示，LLM相似度与残差相关性为0.50（远高于BLEU、余弦的≈0），聚类更平衡且与项目辨别度匹配；在CAT中，LLM约束的SC‑MFI使偏差最小化且RMSE分布更集中，虽然略高于无约束MFI但效率损失可忽略。

**⚠️ 局限性**

局限性：仅使用单一LLM（Claude Sonnet 4）与单一题库；提示结构与权重设定较为任意，未在不同模型或领域进行交叉验证；对不同学科或题目类型的泛化性仍需进一步检验。

---

## 587. BioKERN: Biological Kernel Regularization for Histology-to-Transcriptomics Neighborhood Retrieval

**arXiv ID:** 2608.24823 | [PDF](https://arxiv.org/pdf/2608.24823v1)

**作者:** Seungik Cho `[一作]` (Rice University), Betul Orcan-Ekmekci `[通讯]` (Rice University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了BioKERN框架，通过构建基于转录组相似性和空间接近度的可学习生物核，对多模态空间生物数据进行表示学习，并提供分级邻域监督与几何正则化。

**💡 创新点**

创新点在于将明确的生物邻域结构作为可学习的归纳偏置引入到跨模态学习中，利用可学习的分子‑空间权重组合核，既保留实例匹配，又能捕获生物学上意义重大的空间邻域关系。

**🔧 技术方法**

使用了冻结的PLIP病理编码器、残差适配器投影、RBF核、可学习的α权重、InfoNCE损失加上邻域监督、全局与局部对齐正则化等技术。

**📊 数据集**

在两个公开的单捐献者数据集上进行评估：Mouse Brain Visium 10×（2,200 细胞位点）和Human Liver GSE240429（跨切片转移设置）。

**📈 对比分析**

与CCA、Ridge回归、PLIP线性、BLEEP及其仅改进架构版本BLEEP*对比，BioKERN在Bio-mAP指标上取得最高分，鼠脑单尺度从0.51提升到0.62，多尺度从0.50提升到0.67，提升的大部分（63–91%）归因于生物正则化。

**⚠️ 局限性**

局限性包括仅在单捐献者数据集上验证，核组合采用单标量α，缺乏跨捐献者/跨平台转移和更丰富的区域依赖生物先验。

---

## 588. A Co-Simulation Platform Coupling Land Use, Transportation, and Building Energy: Development and Case Study

**arXiv ID:** 2608.24817 | [PDF](https://arxiv.org/pdf/2608.24817v1)

**作者:** Gopindra Sivakumar Nair `[一作]` (Argonne National Laboratory), Paul Waddell `[通讯]` (UrbanSim Inc)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一个集成土地利用、交通与建筑能耗的协同仿真平台，在芝加哥地区实现了内部一致的多领域预测。

**💡 创新点**

首次将UrbanSim、POLARIS和CityBES三大模型在同一空间层级下耦合，动态生成建筑占用和新建建筑，实现跨领域的实时反馈。

**🔧 技术方法**

使用基于代理的活动模型 POLARIS、土地利用选择模型 UrbanSim、物理能耗模拟 CityBES，并通过 API 和并行计算实现耦合。

**📊 数据集**

利用美国人口普查区块、就业数据、建筑物 GIS 轮廓、税务评估记录以及芝加哥能源基准数据等多源数据集。

**📈 对比分析**

通过将耦合运行与仅交通或仅土地利用的基线进行比较，评估政策对 VMT、VHT 与建筑能耗的影响；结果显示耦合对县级指标影响约1–2%，总运行时约44小时/年。

**⚠️ 局限性**

局限包括未来网络速度可能被高估、区域总量固定不随政策变化、仅在芝加哥可复制、对远程办公对住宅选择的影响建模不足、需额外本地建筑清单与校准。

---

## 589. Do Robotic World Models Really Follow Actions? Diagnosing and Aligning Action-Conditioned Generation for Policy Learning

**arXiv ID:** 2608.24885 | [PDF](https://arxiv.org/pdf/2608.24885v1)

**作者:** Sixiang Chen `[一作]` (Peking University), Shanghang Zhang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了WorldEcho基准，用五类动作查询同时评估视觉完整性与SE(3)轨迹对齐，构建WorldSync训练框架以提升动作条件世界模型的动作跟随可靠性，并在RoboTwin与真实机器人上验证其对政策改进的正面影响。

**💡 创新点**

① 设计覆盖专家、跨状态、局部扰动、策略回放、可行空间采样五类动作查询；② 将视觉完整性与轨迹对齐相结合的门控评估；③ 通过动作覆盖扩展、动作强制专家（AFE）与干预效应（IE）监督的三管齐下训练策略，使模型在多分布下实现更真实的动作响应。

**🔧 技术方法**

基于流匹配的动作条件视频生成网络；NDTW动态时间对齐与MUSIQ、SAM等视觉质量评估；视频轨迹提取器用于SE(3)轨迹匹配；动作强制专家与干预效应监督用于表示层和关系层的对齐。

**📊 数据集**

RoboTwin仿真基准中的50个操纵任务，以及少量真实机器人演示数据。

**📈 对比分析**

与CtrlWorld、Cosmos-Predict2.5、Cosmos3、DreamDojo、Motus、LingBotVA六种现有世界模型在完整门控误差、原始NDTW和视觉通过率三项指标上进行比较。WorldSync在完整门控误差与视觉通过率上均优于所有基线，并在政策改进实验中使成功率提升约13%（仿真）至20%（真实机器人）。

**⚠️ 局限性**

缺乏对长时序交互、多机器人体系结构和开放世界环境的全面评估；对离线数据集的依赖以及在大规模真实机器人交互中的可扩展性尚未验证。

---

## 590. SPO++: Stream-Aligned Policy Optimization for Asynchronous Agentic RL

**arXiv ID:** 2608.24870 | [PDF](https://arxiv.org/pdf/2608.24870v1)

**作者:** Kai Ruan `[一作]` (Renmin University of China), Zihe Huang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SPO++，通过事件时间提示记忆和按动作‑token 量化的优势归一化，解决单轮滚动优势估计的两大不一致问题。

**💡 创新点**

创新点在于将持久提示基线与策略时钟对齐，并将标准化指标改为与 token‑mean 损失一致的动作‑token 量化；从而在单轮训练中提升奖励与学习效率。

**🔧 技术方法**

采用强化学习中的单轮优势估计、GRPO 与 SPO 基础框架、Dual‑Clip 修剪、策略事件坐标记忆、动作‑token 归一化以及自适应保留因子。

**📊 数据集**

使用 ALFWorld 128 任务集和 Math‑TIR（DAPO‑Math‑17K 1500 例子）进行评测，并以 Qwen3.5‑0.8B 与 Qwen3.5‑2B 两个规模模型作为代理。

**📈 对比分析**

在与原 SPO 同配置的匹配实验中，SPO++ 在 ALFWorld 领域提升曲线面积约 15‑19%（0.8B/2B）并在 Math‑TIR 领域取得约 2‑3% 的正向收益，显示出显著的学习效率和终端奖励提升。

**⚠️ 局限性**

实验仅在小型 Qwen3.5 模型与有限预算下进行，未能评估更长时序或 OOD 任务；归一化与批量缩放的独立影响未单独分离；并发限制与离散提示依赖导致事件时间实验受限，稀疏奖励仍是挑战。

---

## 591. Learning Whom to Trust : Decision-Generated Credibility in Social Learning

**arXiv ID:** 2608.24851 | [PDF](https://arxiv.org/pdf/2608.24851v1)

**作者:** Gabriel Bontemps `[一作]` (Université Côte d'Azur), Abhishek Banerjee `[通讯]` (Queen Mary University of London)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了强化学习主体在漂移扩散决策过程中产生的置信度如何作为社交信息的权重，从而形成决策生成可信度的反馈循环，并通过模拟和理论分析揭示其对集体学习的影响。

**💡 创新点**

创新点在于将置信度视为决策过程的内部输出，并将其直接映射为社交权重，实现行为过程到社交影响的闭环；提出局部放大阈值，解释社交强度与学习效率的非单调关系，并给出跨社区渗透对极化的解析作用。

**🔧 技术方法**

采用强化学习与漂移扩散（DDM）相结合的决策模型，利用蒙特卡罗仿真、线性化雅可比分析、社区块（quotient）聚合以及结构对照实验来研究社交传递机制。

**📊 数据集**

仅使用基于无噪声的 agent‑based 模拟数据（400–1000 试验规模），未使用任何外部真实数据集。

**📈 对比分析**

通过多次蒙特卡罗仿真与结构消融对照，绘制社交强度与渗透率的相位图和动态系统图；实验显示社交强度呈非单调性，中等强度能加速学习并实现高效率共识，而过高强度会放大早期高置信错误导致错误共识；低渗透率导致极化，高清渗透率促进错误共识。

**⚠️ 局限性**

模型仅考虑二元行动、固定奖励、固定网络结构和同质参数，置信度仅作为内部信号且不考虑外部置信推断；未涵盖多项奖励、动态网络或非二项决策情境，限制了其在更复杂现实环境中的直接适用性。

---

## 592. LAION-BVD: A 10-Million-Hour Open Video Dataset for Multimodal Pre-training

**arXiv ID:** 2608.24845 | [PDF](https://arxiv.org/pdf/2608.24845v1)

**作者:** Andreas Hochlehnert `[一作]` (University of Tübingen), Matthias Bethge `[通讯]` (University of Tübingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

发布了 LAION‑BVD，一个包含 1.3B 视频链接、80M 视频（共 10M 小时）的大规模开放视频数据集，并通过内容感知场景检测生成 55M 片段级视频/音频字幕和 300M 关键帧/图像字幕，构建了跨模态预训练的完整管道。

**💡 创新点**

创新点包括：① 以可扩展的方式将海量网络视频拆分为场景级剪辑并自动生成视频、音频和图像字幕；② 同时提供视频、音频和图像三种模态的高质量训练样本；③ 通过对比实验证明该数据集在视频、音频和图像预训练任务中均能显著提升性能，并展示了规模与算力的持续上升曲线。

**🔧 技术方法**

技术手段主要包括：内容感知场景检测（阈值 30）拆分剪辑；Qwen3‑VL‑2B‑Instruct 生成 20 词以内视频字幕；Audio Flamingo 3 生成 10 词以内音频字幕；DeepSeek‑VL2‑tiny 对关键帧进行重 caption；使用 ViCLIP、CLAP、CLIP 等对比学习模型训练；采用 WiSE‑FT 检查点融合提升性能。

**📊 数据集**

使用了自建的 LAION‑BVD 数据集（视频/音频/图像子集），并在公开基准上进行评估：视频任务用 Kinetics‑400、UCF‑101、HMDB51、MSR‑VTT、MSVD；音频任务用 UrbanSound8K、AudioCaps、Clotho；图像任务用 COCO、ImageNet‑1k、ImageNet‑R、Sketch、V2；对比基准包括 InternVid、DataComp‑1B、Re‑LAION、OpenCLIP 等公开模型。

**📈 对比分析**

采用零样本检索、分类准确率等标准评估；实验结果显示：BVD‑V‑55M 训练的 ViCLIP L‑14 在视频检索和动作分类上平均提升 2‑4 pp；BVD‑A‑10M 训练的 CLAP 在音频检索/分类上与 LAION‑Audio 相当或更优；BVD‑I‑300M 训练的 CLIP 在 MS‑COCO 检索上优于同等规模的 Web 图像数据集，但在 ImageNet‑1k 零样本分类上略逊。性能随数据量和模型规模呈持续上升趋势。

**⚠️ 局限性**

局限性包括：字幕全部自动生成、篇幅短小，可能缺乏细节且带有生成模型偏差；实验仅覆盖对比学习任务，未评估生成式或扩散模型；视频、音频与图像在本研究中未联合训练，无法验证跨模态同步性；数据采集仅依赖视频平台，未做严格的安全与偏见过滤，可能继承平台中的种族、性别或地域偏见。

---

## 593. Constrained Entity Selection under Partial Knowledge for LLM-Based Knowledge Graph QA

**arXiv ID:** 2608.24824 | [PDF](https://arxiv.org/pdf/2608.24824v1)

**作者:** Emanuel Kitzelmann `[一作]` `[通讯]` (Brandenburg University of Applied Sciences), Emanuel Kitzelmann (Brandenburg University of Applied Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种在知识图不完整情况下，对LLM生成的候选答案进行约束验证的框架；

**💡 创新点**

创新点在于引入三值约束语义与受限实体选择问题，利用轻量级约束既能过滤非法答案又能为合法答案提供符号支持，无需完整语义解析；

**🔧 技术方法**

技术包括轻量级约束提取、三值约束语义评估、支持分数计算，并在Hetionet KG上进行实验；

**📊 数据集**

使用Hetionet生物医学知识图作为数据集；

**📈 对比分析**

与未使用约束的基线比较，精确率从0.41/0.66提升至0.70/0.62，召回保持1.0，支持分数对真值候选显著高于非真值候选；

**⚠️ 局限性**

局限在于实验仅基于受控候选集，未验证端到端KGQA流水线；假设KG无误且所有实体已存在，对噪声或更大规模不完整KG的鲁棒性待进一步研究。

---

## 594. Lower Bounds for Linear Hashing via Arithmetic Kakeya

**arXiv ID:** 2608.24866 | [PDF](https://arxiv.org/pdf/2608.24866v1)

**作者:** Ainesh Bakshi `[一作]` (New York University), Alek Westover `[通讯]` (Redwood Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出新的下界并证明在任意随机种子下，Affine 模块线性哈希的最大负载至少为 exp(Ω(log n / log log n))，并给出一个键集合在所有种子下都达到该下界；通过两步简洁归约，将已知的实值哈希和算术 Kakeya 结果迁移至模块线性哈希；进一步证明在足够大的素数模数下，模块哈希与实值哈希的期望最大负载相差不超过 1，提供了上界的另一条途径；并指出任何子多项式上界都会导致算术 Kakeya 猜想及其几何后果。

**💡 创新点**

创新点在于：①用两步归约将实值哈希与算术 Kakeya 之间的关系显化；②给出指数级下界并且对每个种子都成立，突破以往仅有 Ω(log n / log log n) 的下界；③揭示子多项式上界与算术 Kakeya 猜想的深层关联；④提供模块哈希与实值哈希期望负载等价的精细分析。

**🔧 技术方法**

主要技术包括：简洁的归约技术（将实值哈希转为模块哈希，将算术 Kakeya 转为实值哈希）；组合数论与算术进程构造；实值哈希平均负载估计与 Riemann 近似；与 Konyagin–Ruzsa–Schlag 的三阶矩估计相结合；以及利用集合翻译与连通分量分段的方法处理算术 Kakeya 集。

**📊 数据集**

本文为理论分析，无实验数据集；使用的是构造的整数集合与键集合，规模为 n（或 n^{1+o(1)}），并在这些集合上证明下界与上界。

**📈 对比分析**

与已有的 O((n log n)^{1/3}) 上界和 Ω(log n / log log n) 下界相比，本文的指数下界显著提升了对最大负载的理论理解；同时通过归约证明模块哈希期望负载与实值哈希几乎等价，提供了另一条得到 O(n^{1/3}+o(1)) 上界的途径。

**⚠️ 局限性**

局限性包括：①下界仍不能达到多项式级别，归约无法突破算术 Kakeya 猜想的完整形式；②结果仅在宇宙规模为 n^{1+o(1)}（或 u < p）时适用；③若能获得子多项式上界，则必须接受算术 Kakeya 猜想及其几何推论为真，表明该类上界的提升难度巨大。

---

## 595. A Geometric Theory of Robust Fairness Audits

**arXiv ID:** 2608.24818 | [PDF](https://arxiv.org/pdf/2608.24818v1)

**作者:** Binita Maity `[一作]` `[通讯]` (Indian Institute of Technology, Gandhinagar), Binita Maity (Indian Institute of Technology, Gandhinagar)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在特征空间受限扰动下，基于邻域的个体公平性审计的鲁棒性，并建立了几何理论框架。

**💡 创新点**

提出将审计视为对邻域内对偶公平度量的Lipschitz聚合，推导出确定性、概率性和期望性鲁棒性保证，定义了审计波动性指标，并证明邻域替换是审计不稳定的根本机制。

**🔧 技术方法**

使用几何距离分析、Lipschitz聚合理论、概率不等式和期望值推导，以及实验中对k近邻一致性度量的实现。

**📊 数据集**

在Adult、Bank Marketing和COMPAS三个常用数据集上进行实验。

**📈 对比分析**

通过比较不同扰动模型、聚合算子（算术均值、裁剪均值、中位数、最差邻居）以及数据集的局部分离度量来评估鲁棒性；实验显示邻域替换量与审计波动性高度相关，鲁棒聚合算子显著降低波动，局部分离度越大审计越稳健。

**⚠️ 局限性**

假设模型预测不变、扰动受限、仅适用于Lipschitz聚合的邻域审计；对高维复杂模型、非欧氏距离或无监督场景的推广仍待进一步研究。

---

## 596. MDTE: Minority-Aware Diffusion over Temporal Edge Events for Imbalanced Node Classification

**arXiv ID:** 2608.24812 | [PDF](https://arxiv.org/pdf/2608.24812v1)

**作者:** Zhou Zelong `[一作]` (Zhejiang University of Technology), Fan Jing `[通讯]` (Zhejiang University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了一种针对时序边事件的少数类感知扩散框架MDTE，用以解决时序图中节点分类的类别不平衡问题。

**💡 创新点**

1）Distribution‑Aware Selective Propagation：基于LOF的传播过滤与低频跨簇权重，抑制多数类信息同化。2）Multi‑View Discriminative Fusion：通过特征重构与拓扑预测的不确定性提取多视角区分度，指导条件去噪。3）将上述机制与条件扩散、去偏对比学习等自监督流程整合，形成完整的无标签学习框架。

**🔧 技术方法**

条件扩散模型（U‑Net）+方向噪声+LOF+K‑means+GAT低频编码+Normal‑Inverse‑Gamma不确定性建模+拓扑预测+Debiased Contrastive Learning+随机游走位置编码+线性噪声调度。

**📊 数据集**

五个金融交易图数据集：DGraph‑Fin、Elliptic、Elliptic++ Transactions、Elliptic++ Actors、Ethereum。

**📈 对比分析**

与13个基线（时序图方法、静态图AE、失衡图方法、扩散方法）比较，MDTE在所有非AUROC指标上均为最优，尤其在少数类召回率提升最多26.6%、F1提升17.68%，AUROC保持与强基线相当。

**⚠️ 局限性**

对超参数（如LOF阈值、聚类数、对比学习温度等）敏感，尤其在极端不平衡场景（如Ethereum）表现更易受影响；模型复杂度较高，训练需要大量无标签边事件；未在训练阶段直接利用标签信息，限制了对稀缺少数类标签的利用。

---

## 597. Strictly Causal Streaming Video Anomaly Detection with a Theoretically-Grounded State-Space Core

**arXiv ID:** 2608.24810 | [PDF](https://arxiv.org/pdf/2608.24810v1)

**作者:** Yogesh Kumar `[一作]` `[通讯]` (Indian Institute of Technology Jodhpur), Yogesh Kumar (Indian Institute of Technology Jodhpur)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种严格因果的流式视频异常检测模型，能够在每帧 O(1) 时间和固定内存更新状态，并在边缘设备上实现实时推理。

**💡 创新点**

创新点包括：1) 采用对角线线性状态空间递归与可学习的事件边界衰减门，实现真正无向前视、无片段缓冲的流式推理；2) 推导了衰减谱与检测延迟、可检测最短异常时长之间的闭式关系，并通过实验验证；3) 在真实边缘硬件（Apple M3 Pro）上给出了毫秒级延迟与高帧率的端到端测量。

**🔧 技术方法**

技术：冻结视觉骨干（ResNet‑18 或 DINOv2 ViT），堆叠 2 层对角线 SSM，使用输入/状态相关的门控衰减，利用自监督下一嵌入预测训练，使用 PyTorch MPS 进行边缘部署。

**📊 数据集**

数据集：UCSD Ped2、CUHK Avenue（无监督 VAD 基准），并计划加入 ShanghaiTech 以进一步验证。

**📈 对比分析**

比较方法：与现有非因果 SSM 基线对比（但因实现差异未能直接复现），评估指标为帧级 ROC‑AUC、EER，边缘延迟与 FPS。结果显示 AUC 分别为 67.9%（Ped2）和 70.2%（Avenue），比以往方法低，但实现了 1.3‑1.4 ms/帧、>1300 FPS 的实时性能。

**⚠️ 局限性**

局限性：1) RBDC/TBDC 评估采用简化帧重叠方式，未使用官方定位标准；2) 延迟测量仅在单一 Apple M3 Pro 上完成，缺乏多平台验证；3) 理论延迟上界基于固定衰减，未充分考虑门控导致的时间变衰减；4) 在小数据集（Ped2）上门控可能过拟合，需更大数据集验证。

---

## 598. Structurally-bounded Agentic Graph Exploration for Evidence-Grounded Scholarly DeepSearch

**arXiv ID:** 2608.24809 | [PDF](https://arxiv.org/pdf/2608.24809v1)

**作者:** Rima Hazra `[一作]` (National University of Singapore), Animesh Mukherjee `[通讯]` (Indian Institute of Technology Kharagpur)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种结构化、可检验的学术文献检索方法，通过一次种子搜索、1.5跳引用邻域扩展、基于主张级别推理的边权重裁剪，并使用递归的个人化PageRank进行排名。

**💡 创新点**

将深度研究代理的无界搜索替换为有限的引用图探索，使用主张级别的蕴含推理权重化边缘，构建可审核的证据图。

**🔧 技术方法**

Semantic Scholar API、Qwen2.5-3B-Instruct命题提取、微调的科学蕴含模型（MSciNLI/SciNLI）、Recency-aware Personalized PageRank/SALSA、GROBID抽取。

**📊 数据集**

约50万篇arXiv论文（2016-2026），LitSearch、ACL、ICLR、arXiv三组检索基准。

**📈 对比分析**

与传统深度研究代理和SPECTER+DeepWalk等基线相比，在ACL/ICLR/arXiv数据集上recall@50提升约3倍、MAP提升显著，同时外部调用次数、token量、执行时间和成本显著下降。

**⚠️ 局限性**

仍受种子选择、主张提取和蕴含模型精度限制，尤其对极度新颖或跨领域的文献检索效果可能不足；模型需要手工构建计划与参数调优，缺乏通用自动化。

---

## 599. Latent Action as Intention Enables Efficient Future Imagination for World Action Models

**arXiv ID:** 2608.24882 | [PDF](https://arxiv.org/pdf/2608.24882v1)

**作者:** Xiang Li `[一作]` (CollegeAI, Tsinghua University), Wenchao Ding `[通讯]` (TARS Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种名为LAWA的世界动作模型（WAM），在测试时通过紧凑的潜在动作序列进行未来意图预测，从而避免昂贵的未来视频生成。

**💡 创新点**

创新点在于将未来想象迁移到潜在动作空间，并利用无标签的自监督机器人与第一人称视频预训练的离散潜在动作分词器；同时通过掩码监督增强对交互区域的关注。

**🔧 技术方法**

采用离散潜在动作分词器、流匹配（flow matching）训练、联合注意力机制、多模型联合训练以及DINOv2+SAM 2等技术。

**📊 数据集**

主要数据集包括RoboCasa（24个桌面任务）、LIBERO-Plus（零样本跨域评估）以及现实世界四个机器人装配与长时序操控任务。

**📈 对比分析**

与Fast‑WAM（无未来预测）和Joint‑WAM（完整未来预测）对比，LAWA在RoboCasa上实现65.6%（10%数据）/80.8%（全数据）成功率，优于Fast‑WAM 9.6/4.5个百分点，且保持与Joint‑WAM相近的性能；在LIBERO‑Plus零样本场景获得74.4%成功率，超过Fast‑WAM 14.4个百分点；在真实任务中平均提升约35个百分点，且推理延迟比Joint‑WAM低42.9%。

**⚠️ 局限性**

局限性包括对潜在动作分词器的预训练依赖，若无足够的无标签视频其性能会下降；同时在极端分布偏移下仍可能出现性能下降；并且在极少量演示数据时与Joint‑WAM相比仍略逊一筹。

---

## 600. Recursive Experiential-Working Memory Evolution for Long-Horizon Agent Harnesses

**arXiv ID:** 2608.24876 | [PDF](https://arxiv.org/pdf/2608.24876v1)

**作者:** Zhaochen Yu `[一作]` (NUS), Ling Yang `[通讯]` (Princeton University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种递归经验-工作记忆（Recuris）架构，在固定的LLM与工具层之上，通过工作记忆维护已验证的任务状态，并以此状态驱动经验记忆中技能的检索；在每一次任务执行后记录结构化轨迹，利用 Meta‑Agent 对失败原因进行定位并只更新受影响的记忆组件，最终实现跨任务、跨模型的记忆自我改进。

**💡 创新点**

创新点包括：① 将工作记忆与经验记忆耦合，保证技能调用始终与当前任务状态对齐；② 通过结构化轨迹实现可解释的故障定位，显著提升定位准确率（64.8% vs 13.0%）；③ 递归改进仅限于记忆控制层，保持基础模型与改进过程不变；④ 通过验证门控保证改动不会导致回归，提升系统安全性。

**🔧 技术方法**

技术手段：LLM（如GPT-5.6 Sol、Claude Opus 5等）+工具调用接口；工作记忆（状态模式、验证器、状态更新机制）；经验记忆（可重用技能库）；结构化轨迹记录（状态、技能、动作、观测、验证结果）；Meta‑Agent（经验检索、故障定位、补丁生成、验证门控）。

**📊 数据集**

数据集：τ²‑Bench（Retail 114 任务、Airline 50 任务），SkillFlow（166 任务，20 组），Terminal‑Bench 2.1（87 任务）。

**📈 对比分析**

对比方法：基准参考实现、带初始记忆、带演化记忆；在10个模型（从3B开源到前沿模型）上分别评估任务成功率；结果显示演化记忆在35/37个模型-基准组合中提升任务成功率，最大提升约+32.2分；在最长任务中提升显著，且前沿模型也能获得 15–20 分左右的提升。

**⚠️ 局限性**

局限性：① 只改进记忆层，无法提升基础模型的能力；② 需要大量失败经验和可验证的轨迹；③ 对任务结构高度依赖，跨任务迁移受限；④ 验证门控可能会拒绝一些潜在有益的补丁；⑤ 目前只评估在已定义的四个基准上，未检验对更开放领域任务的泛化。

---

