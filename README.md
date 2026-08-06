# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-08-06 | 今日论文总数: 547

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. ColorFD: A Finite-Difference Guided Black-Box Physical Adversarial Attack for Remote Sensing Object Detection

**arXiv ID:** 2608.04559 | [PDF](https://arxiv.org/pdf/2608.04559v1)

**作者:** Tiannuo Guo `[一作]` (Beijing University of Chemical Technology), Deliang Xiang `[通讯]` (Beijing University of Chemical Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于多块纯色小补丁的黑盒物理攻击方法（ColorFD），用于使遥感目标检测模型无法检测到目标。

**💡 创新点**

创新点包括：① 目标级适应度与选择机制，能够在多目标场景下保留各目标的改进；② 采用有限差分色探测定位关键区域并引导补丁搜索；③ 提出类别级共性特征提取作为搜索空间先验，进一步压缩搜索空间；④ 使用纯色补丁易于打印和部署，提升物理可行性。

**🔧 技术方法**

核心技术：差分进化（DE）用于联合优化补丁位置和颜色；有限差分颜色探测和共性特征提取作为搜索空间约束；目标级适应度与选择策略；数字与物理实验验证。

**📊 数据集**

使用DIOR遥感图像数据集的飞机子集（100张图，407架飞机）进行评估。

**📈 对比分析**

与Bbox-Att、AP-PA、BADEI等黑盒/白盒方法在YOLOv3u、YOLOv5u、Faster R-CNN上进行对比，评估指标为ASR和AP_50。ColorFD在黑盒下的ASR显著高于Bbox-Att，且在YOLOv5u与Faster R-CNN上可与或优于白盒基线，显示出优异的攻击效果。

**⚠️ 局限性**

局限性：DE的随机性导致不同运行结果差异大；关键区域定位在两阶段模型（如Faster R‑CNN）效果不佳；纯色补丁对环境变化（视角、光照、尺度）敏感，攻击鲁棒性受限；补丁可见度相对较高。

---

## 2. CSGen: A Multi-Domain Curvilinear Structure Generation Model via Hierarchical Multimodal Diffusion

**arXiv ID:** 2608.04655 | [PDF](https://arxiv.org/pdf/2608.04655v1)

**作者:** Zhe Shan `[一作]` (Hainan University), Xia Xie `[通讯]` (Hainan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了CSGen，一种基于分层多模态扩散模型的可控曲线结构生成方法，能够根据文本、布局等多种条件精准合成高保真曲线图像。

**💡 创新点**

创新点包括①构建覆盖5个领域、包含24,678张样本、7种标注类型的多域曲线结构数据集；②设计层级渐进控制策略(HPCS)，将几何拓扑与视觉上下文分离并逐步注入；③提出稀疏感知损失重加机制(SLRM)，依据结构等效直径动态加权，显著提升细枝和脆弱部位的连通性。

**🔧 技术方法**

技术手段主要是：使用Stable Diffusion 3.5作为生成骨干并接入ControlNet进行局部条件注入；结合CLIP与T5两种文本编码器实现多模态对齐；在训练中采用分层渐进注入（HPCS）和稀疏重加（SLRM）对流动匹配损失进行优化；最终实现高质量图像生成。

**📊 数据集**

使用的数据集为24,678张多域曲线图像，来源包括血管、冠脉、混凝土裂缝、叶脉和道路网络，每张样本配有多种控制图（宽度图、草图、边缘图、部分草图）和文本描述，共7种标注类型。

**📈 对比分析**

与FLUX、Qwen、SD3.5、SD3.5+LoRA、SD3.5+ControlNet等基线在FID、IS、LPIPS、DISTS、CLIPScore、VQA以及结构一致性指标(mIoU、clDice)等上进行对比。CSGen在FID（98.5）和clDice（0.766）上均为最优，整体生成质量、结构连通性和语义对齐显著优于其他方法。

**⚠️ 局限性**

局限性包括：1）依赖高质量的多模态标注，数据获取成本高；2）对极细小或极大尺度的曲线仍可能出现局部细节缺失；3）模型规模和训练成本较高，推理速度受限；4）对完全新领域的迁移能力需进一步验证。

---

## 3. Q-CueGraph: Query-Conditioned Visual Evidence Graphs for Multimodal Reasoning

**arXiv ID:** 2608.04452 | [PDF](https://arxiv.org/pdf/2608.04452v1)

**作者:** Pengcheng Pan `[一作]` (University of Tokyo), Xinfang Zhang `[通讯]` (Tohoku University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Q-CueGraph，一种将问题与图像结构映射到有限面积的坐标级观察窗口的显式证据策略，使得多模态大语言模型能在高分辨率图像中主动决定查看何处。

**💡 创新点**

创新点在于：①使用预缓存的 OCR/布局图或查询条件下的视觉节点生成可重用的结构化候选区域；②将坐标生成与预算、裁剪、组合和回退机制分离成可观察的单步策略；③通过冻结读者的裁剪级反馈进行“利用率”细化，以学习哪些候选区域真正可被模型利用。

**🔧 技术方法**

技术手段包括：冻结的 Qwen2.5‑VL‑7B 视觉语言读者；PP‑OCRv5 提取 OCR 线与布局关系；OWLv2 检测生成自然图像的视觉节点；基于图的匹配、布局扩展、候选排序、组合、padding、面积预算和可选的梯度提升利用率评分器。

**📊 数据集**

实验使用六大基准：DocVQA、InfographicVQA、TextVQA、OCRBench、ChartQA 与 V*Bench（含 InfoVQA）等，涵盖文档、信息图、文本问答、OCR 检索、图表分析与通用图像问答。

**📈 对比分析**

与全图、低分辨率全视、随机/中心/密度裁剪、对抗/打乱问题控制以及原生自我放大（native self‑zoom）等基线对比，Q‑CueGraph 在 V*Bench 上从 19% 图像面积提升准确率至 0.833（全图 0.696）；在 DocVQA、InfographicVQA 等任务中在 1/4–1/2 图像面积内即可达到 70–90% 的全图性能；整体表现优于全图和基线裁剪方案，且在显式观察与利用率细化方面展现显著优势。

**⚠️ 局限性**

局限性包括：对局部可定位证据最有效，对全局或分散式证据（如复杂图表）效果有限；策略目前仅为单步一次性裁剪，未实现多轮迭代或多区域组合；依赖 OCR/检测质量，结构错误或漏检可能导致选取失败；以及在某些任务中对预算和分辨率瓶颈的敏感度仍需进一步研究。

---

## 4. Right Reset: Chunking by Prefix Removal

**arXiv ID:** 2608.04330 | [PDF](https://arxiv.org/pdf/2608.04330v1)

**作者:** Mike Vegeto `[一作]` `[通讯]` (Independent Researcher), Mike Vegeto (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了前缀移除探测与右重置（RR）边界评分方法，利用语言模型的隐藏状态保留度来识别文本块边界。

**💡 创新点**

创新点在于将完整前缀移除作为干预，使用隐藏状态相似度与观察到的词概率比率捕捉上下文依赖，从而在结构弱化的文本中提供有效的分块信号。

**🔧 技术方法**

使用的技术包括因果语言模型（如Qwen3-4B、Qwen3.5-9B、Gemma4E4B等）的隐藏状态计算、余弦相似度评分、观察词概率比率、KL散度对比，以及基于动态规划的全局分块决策。

**📊 数据集**

实验数据集为从BEIR的FiQA、NFCorpus、SciFact中抽取的276条记录（+15条校准集），以及Wiki-50作为范围控制。

**📈 对比分析**

与传统基线（BGE嵌入距离、被动惊讶度、句子切点、固定网格）以及直接提示的指令模型相比，RR在扁平化记录上清洁单元召回率提升至47.7%（对BGE的+21.8%）并使分块F1从0.823提升至0.893，显著优于所有无任务训练的基线。

**⚠️ 局限性**

局限性包括仅对局部、检查点特定的上下文依赖进行测量，需在每个候选边缘执行一次完整前缀移除（计算昂贵），实验数据集有限，缺乏与监督分块器的比较。

---

## 5. SCOPE: Field-of-View-Aware Path Planning in Unknown 3D Environments via Safety-Volume Certification

**arXiv ID:** 2608.04420 | [PDF](https://arxiv.org/pdf/2608.04420v1)

**作者:** Junbin Yuan `[一作]` (Carnegie Mellon University), Sebastian Scherer `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种基于在线安全体积认证的三维 FOV 约束路径规划框架 SCOPE，能在未知环境中为多旋翼机器人生成可安全执行的轨迹。

**💡 创新点**

创新点包括：①将任务进度与安全认证解耦，利用“观察义务”将第一个未认证点转化为观测任务；②基于目标中心的视点搜索和递归子目标规划，配合上下文局部排除记忆避免循环与冗余；③引入已认证预览机制实现平滑执行；④通过可观察多面体提取实现观测约束的连续化；⑤提供条件完整性证明与安全保证。

**🔧 技术方法**

核心技术包括：体素网格地图与安全核膨胀；构建乐观图与已认证图；A* 与逆多源搜索；目标中心视点搜索与受限扩展；递归子目标与排除记忆；已认证预览与安全飞行走廊；GCOPTER+MINCO 轨迹优化；可观察多面体生成与校验。

**📊 数据集**

实验使用三种合成 3D 场景：Scene 1（垂直洞口）、Scene 2（多室水平通道）和 Scene 3（多层结构）。共 60 场随机任务；并在两台实车上演示多层上升与低通道穿越。

**📈 对比分析**

与三种基线（CPA-search、FOV-FMT-search、OmniPlanner-TR）比较，评价指标为成功率、路径长度、边缘/危险占用体素数以及危险-free 试验数。SCOPE 在所有场景 100% 成功，危险体素几乎为零；基线在垂直场景中成功率下降且危险体素明显增多。SCOPE 的预览机制显著缩短平均任务时间约 27%，但路径略长。实机实验验证了完整的感知-规划-优化-控制链条。

**⚠️ 局限性**

局限性：①理论安全保证仅在搜索层面，优化后走廊实现时可能产生微小“危险”占用；②实验仅涵盖静态未知环境，对动态障碍或感知噪声的鲁棒性未作充分验证；③需要精确的体素地图与安全核，分辨率与传感器模型匹配度对性能影响大；④递归子目标与排除记忆在极端复杂环境下可能导致搜索开销上升；⑤实际硬件实现中对实时预算的依赖导致某些场景下规划暂停或失败。

---

## 6. Modality Agreement- and Conflict-Aware Prototype Hypergraph Learning for Multimodal Intent Understanding

**arXiv ID:** 2608.04054 | [PDF](https://arxiv.org/pdf/2608.04054v1)

**作者:** Mohnish Raj `[一作]`, Ayan Dutta `[通讯]`

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了一种层次化的多模态意图识别框架 MACH，显式建模多模态之间的同意与冲突交互，利用原型引导的超图实现跨模态信息的可重用表示。

**💡 创新点**

创新点在于：①将同意与冲突分别视为可重用的高阶交互结构；②通过原型超图捕捉共享与冲突模式；③采用样本自适应仲裁机制动态平衡同意与冲突信息；④采用递进优化策略保证层次依赖的稳定学习。

**🔧 技术方法**

使用的技术包括：多模态编码器（如 Qwen2.5-Omni-7B）、原型学习、超图神经网络（HGNN）、信息对比学习、监督对比损失、交叉熵分类损失以及多阶段逐层训练流程。

**📊 数据集**

实验数据集有：MIntRec、MIntRec2.0（多模态意图识别）以及 MELD-DA（多模态情感识别）三大公开基准。

**📈 对比分析**

与多模态融合、置信度感知、图模型等现有方法对比，MACH 在 Accuracy、Weighted F1、Macro F1 等指标上均优于对照组，尤其在 MIntRec2.0 上提升约 1.5% 的准确率，表明其在复杂意图空间中的泛化能力更强。

**⚠️ 局限性**

局限性包括：①对更长的上下文依赖（如全对话）尚未建模；②原型与超图的设计仍需针对不同任务进行手动调优；③在视觉模态表现不佳时，模型依赖文本与音频的优势，导致对纯视觉信息的利用有限。

---

## 7. Learning Sexism Detection Using Multi-Agent Perspectivist Preference Optimization

**arXiv ID:** 2608.04056 | [PDF](https://arxiv.org/pdf/2608.04056v1)

**作者:** Hadi Mohammadi `[一作]` (Utrecht University), Masoume M. Raeissi `[通讯]` (Utrecht University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在性别歧视检测中，利用行为聚类构建多代理模型，保留不同注释者视角。

**💡 创新点**

首次将注释者行为聚类与多代理偏好优化结合，并通过团队奖励保持代理的聚类一致性。

**🔧 技术方法**

使用SFT、DPO、GRPO等偏好优化技术以及GPT和Qwen3-8B等大型语言模型。

**📊 数据集**

采用EXIST 2024英西推文性别歧视标注数据。

**📈 对比分析**

与无训练、零样本集成以及提示人群相比，MAP-PO在四种设置下均实现约90%的团队F1，单代理也保持对聚类的高一致性。

**⚠️ 局限性**

局限在于块设计导致跨聚类覆盖不足，聚类仅用三项行为特征，且仅评估单一任务。

---

## 8. Investigating Click Behaviors On Google Search Result Pages That Produce an AI Overview

**arXiv ID:** 2608.04831 | [PDF](https://arxiv.org/pdf/2608.04831v1)

**作者:** Athena Chapekis `[一作]` (Pew Research Center), Aaron Smith `[通讯]` (Pew Research Center)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析了美国900名成人在2025年3月1日至31日的网络浏览数据，探讨了谷歌AI概述（AI Overview）在搜索结果中的出现条件及其对用户点击行为和浏览结束率的影响。

**💡 创新点**

首次系统使用真实用户的浏览数据和混合效应逻辑回归模型评估AI概述的出现概率及其对点击率和浏览结束率的影响，并识别了查询长度、疑问词开头、名词+动词组合等特征是触发AI概述的关键因素。

**🔧 技术方法**

混合效应逻辑回归模型（R语言的lme4包），以及基于panel数据的描述性统计和可视化分析。

**📊 数据集**

Ipsos KnowledgePanel Digital在线面板的900名美国成人的网页访问日志（共2,457,176次访问，68,879条Google搜索查询）。

**📈 对比分析**

采用对照设计：对比包含AI概述的SERP与不包含的SERP的点击率和浏览结束率，使用混合效应模型控制随机效应和查询属性。结果显示，AI概述的出现导致点击率下降约一半（从15%降至8%），浏览结束率提高约10个百分点（从16%升至26%）。

**⚠️ 局限性**

该研究基于观察性数据，无法得出因果结论；样本仅限于美国成年互联网用户，可能不具备全球普适性；AI概述只在桌面浏览器上最多显示三条来源，无法捕获更多来源的点击行为；以及仅关注点击和浏览结束等离散行为，未评估信息满意度等深层用户体验。

---

## 9. CLIP-CC-Bench: Evaluating Paragraph-Level Video Descriptions in Video-Language Models

**arXiv ID:** 2608.04302 | [PDF](https://arxiv.org/pdf/2608.04302v1)

**作者:** Mukhtiar Ali `[一作]` (South Dakota State University), Chulwoo Pack `[通讯]` (South Dakota State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 CLIP-CC-Bench，一个用于评估长段落级视频描述的基准套件。

**💡 创新点**

创新点包括：①对参考描述剔除专有名词，降低记忆偏差；②采用五个 MTEB 级 LLM 嵌入模型的粗细粒度语义匹配，并通过 Borda 计数聚合提升评估鲁棒性。

**🔧 技术方法**

使用的技术包括：多模型嵌入（GTE-Qwen2‑7B、KaLM‑Gemma3‑12B、Llama‑Embed‑Nemotron‑8B、NV‑Embed‑v2、Qwen3‑Embedding‑8B）、余弦相似度的粗细粒度匹配、谐波平均与 Borda 聚合。

**📊 数据集**

数据集由 200 个约 90 秒的电影片段组成，累计 5 小时，来源跨 140+ 部影片，且每段都有专家撰写、去除专有名词的段落描述。

**📈 对比分析**

通过对 17 种视频‑语言模型在 5 个嵌入评审下的平均 HM‑CF 分数进行 Borda 排名，VideoLLaMA3 获得 80/80 的完美排名，平均分最高约 0.67，最低约 0.48，显示 Transformer 家族在长段落描述上占优，细粒度匹配仍存在提升空间。

**⚠️ 局限性**

局限性包括：评估仅覆盖电影片段且仅提供单一参考描述，缺乏多参考和跨语言测试；未显式评估事件顺序或因果关系；数据量相对有限，未来需扩展规模与多样性。

---

## 10. TS2TabPFN: Time Series Classification and Extrinsic Regression through Feature Extraction and a Tabular Foundation Model

**arXiv ID:** 2608.04174 | [PDF](https://arxiv.org/pdf/2608.04174v1)

**作者:** Gabriel da Costa Merlin `[一作]` (University of São Paulo), Diego Furtado Silva `[通讯]` (University of São Paulo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了TS2TabPFN框架，将时间序列特征提取与TabPFN基础模型结合，用于时间序列分类和外在回归。

**💡 创新点**

创新点在于把特征工程与基于ICL的Tabular foundation模型解耦，既保留可解释性，又实现无训练的高性能推理。

**🔧 技术方法**

采用tsfresh、catch22、MultiROCKET三种特征提取器，并利用TabPFN 2.5 Transformer进行单次前向推理。

**📊 数据集**

实验数据来自UCR/UEA分类归档（158个，实际使用134个）和TSML Extended回归归档（63个，实际使用55个）。

**📈 对比分析**

通过Wilcoxon符号秩检验和Critical Difference图与DrCIF、HC2等最先进模型比较，TS2TabPFN在TSER上显著优于DrCIF，在TSC上与HC2相当，速度提升可达两位数倍。

**⚠️ 局限性**

局限在于TabPFN对上下文大小和特征维度的显存约束，需一次性装入GPU，超过500维会触发子采样，导致内存和计算开销；因此部分大型数据集被排除。

---

## 11. ACA-GS: Adaptive-Capacity Anchored Gaussian Splatting for Compact Dynamic Radiance Fields

**arXiv ID:** 2608.04581 | [PDF](https://arxiv.org/pdf/2608.04581v1)

**作者:** Seunghyeon Song `[一作]` (Sungkyunkwan University), Jong Hwan Ko `[通讯]` (Sungkyunkwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对动态场景的 4D 高斯分布渲染，提出了一种可自适应分配锚点容量的框架（Adaptive-Capacity Anchored Gaussian Splatting，ACA-GS）。

**💡 创新点**

创新点在于：①**Adaptive Anchor Cardinality**——根据全局累积视差不透明度衡量重要性，动态重定位低重要性高斯到高需求区域，实现每个锚点可变的高斯数量；②**Adaptive Anchor Feature Masking**——使用可学习的通道掩码仅激活每个锚点中真正需要的特征通道，并引入条件正则化防止时间特征占用空间容量。

**🔧 技术方法**

技术主要包括：Anchor-based 4D Gaussian Splatting、可学习通道掩码（STE 方式）、全局重要性评估与高斯迁移、每锚点的可变高斯计数门控、条件正则化损失、以及与 GIFStream 编码框架的端到端压缩训练。

**📊 数据集**

实验使用 MPEG（Bartender、Cinema）、Panoptic Sports、N3DV 三大多视角视频数据集，覆盖快速运动、慢速运动和复杂几何场景。

**📈 对比分析**

与 4DGaussian、4DGS、E-D3DGS、STG、CSTG+PP、GIFStream 等基线进行对比。结果显示：在 MPEG 上可实现 1.5 倍以上的压缩率（相较 GIFStream），存储量仅 3.5–7.3 MB，PSNR/SSIM/LPIPS 与现有方法持平甚至更优；在 Panoptic Sports 与 N3DV 上也保持最高或相近的视觉质量，并显著减少占用内存。

**⚠️ 局限性**

局限性包括：需要较多超参数调优（如阈值 τ、ρ、γ、λ_M、λ_CR 等），对极端大场景或极高速运动的鲁棒性尚未充分验证；目前仍依赖锚点的均匀初始化，后期需要进一步探索层次化锚点或跨帧一致性机制；迁移和掩码机制在训练期间会增加计算复杂度。

---

## 12. MESH: Memory-Efficient Sinkhorn Optimization for Mixture-of-Experts Training

**arXiv ID:** 2608.04407 | [PDF](https://arxiv.org/pdf/2608.04407v1)

**作者:** Masato Fujitake `[一作]` `[通讯]` (Fast Accounting Co., Ltd.), Masato Fujitake (Fast Accounting Co., Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 Mixture‑of‑Experts (MoE) 专家矩阵的内存高效优化，提出了隐藏动量 Sinkhorn 优化器及其可选的块预条件版本，解决了原始 SAGE/Sinkhorn 在 MoE 中的性能瓶颈。

**💡 创新点**

创新点在于：① 引入隐藏动量机制，在 Sinkhorn 归一化前对专家梯度进行时序平滑；② 将该动量保存在梯度缓冲生命周期中，避免显式存储专家的一阶矩；③ 设计可选的块/神经元逆 RMS 预条件，以提升数值稳定性。

**🔧 技术方法**

使用了 Sinkhorn 矩阵归一化、SAGE 混合策略、隐藏动量累积、块/神经元逆 RMS 预条件、MoE 路由、AdamW 对照、梯度缓冲生命周期管理等技术。

**📊 数据集**

在 FineWeb‑Edu 流式文本数据集上，对 110M 参数的 DeepSeek‑style MoE 模型进行预训练。

**📈 对比分析**

通过多种随机种子（100、512、42）与 AdamW、SAGE/Sinkhorn 进行对比实验：隐藏动量 Sinkhorn 将 optimizer‑state 内存从 0.883 GB 降至 0.331 GB，CUDA 峰值约降低 12.6%；评估 loss 与 AdamW 相差约 0.05，整体保持较好性能；块预条件在部分种子可进一步略减 loss，但并非必需。

**⚠️ 局限性**

局限性：实验仅在单一 110M MoE 架构、5 000 步、512 样本评估上进行；未对词表/共享专家的角色分配进行完整优化；仍存在与 AdamW 的性能差距；缺乏在更大规模或更多种子上的验证。

---

## 13. Beyond the QBER Threshold: A Temporal QBER Based Machine Learning Framework for Multi Attack Detection in BB84 QKD

**arXiv ID:** 2608.04047 | [PDF](https://arxiv.org/pdf/2608.04047v1)

**作者:** Isha `[一作]` (IIT Mandi), Amit Shukla `[通讯]` (IIT Mandi)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并验证了一个基于时域QBER的机器学习框架，用于BB84量子密钥分发系统中多种窃听攻击的检测与分类。

**💡 创新点**

创新点在于：①提出63个物理信息化的时域QBER特征；②利用这些特征实现多类别攻击识别；③用机器学习方法替代传统固定阈值检测，大幅降低误检率；④对特征进行SHAP可解释性分析，提升模型可解释性。

**🔧 技术方法**

采用特征工程（统计、突发、频谱、基线依赖、通道交互等），随机森林、XGBoost和SVM-RBF分类器，并使用SHAP进行模型解释。

**📊 数据集**

使用仿真生成的24,000条BB84会话数据，8类（7种窃听攻击+正常），每类3,000条样本，80/20训练/测试划分，10次独立实验。

**📈 对比分析**

与固定11% QBER阈值和CUSUM在线检测对比。XGBoost获得88.01%准确率、宏F1 0.8803，误检率从0.8477降至0.0198，表现显著优于传统阈值和CUSUM。

**⚠️ 局限性**

局限性：仅在平衡类别、简化攻击模型的仿真环境中验证；未在真实QKD系统中测试；未考虑类别不平衡或域漂移；时窗大小选择经验性。

---

## 14. Retrieve in Time, Correct in Frequency

**arXiv ID:** 2608.04527 | [PDF](https://arxiv.org/pdf/2608.04527v1)

**作者:** Yuze Fan `[一作]` (Everwise-Tech Co., Ltd.), Xueqian Wang `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练‑free 的测试时校正框架（Retrieve in Time, Correct in Frequency），在不更新任何模型参数的前提下提升冻结 VLA（Vision‑Language‑Action）策略在长时序操控任务中的成功率。

**💡 创新点**

创新点包括：1）将进度感知检索与频域低频残差校正分离，形成 Progress‑Memory Alignment 与 Correct‑in‑Frequency 两个独立模块；2）采用因果子序列 DTW 的 Progressive Memory Alignment，维护对每条成功轨迹的可递增前沿；3）只将低频残差（非 DC 及高频）剪裁并叠加到策略输出，保持高频细节与把手动作不变。

**🔧 技术方法**

技术实现包括：冻结 SigLIP‑PCA 视觉编码、BPE‑FAST DCT 频谱动作表示、子序列 DTW 的因果前沿更新、低频残差剪裁与缩放、纯 CPU 计算，无需额外 GPU 资源。

**📊 数据集**

数据集为 LIBERO 四个套件（Long、Spatial、Object、Goal），每套 10 个任务，共 2000 条 episode，用于收集成功记忆并评估。

**📈 对比分析**

方法与冻结 PI‑FAST 基线、Frame‑NN、Time‑Domain、Diffusion Policy 等进行对比。四套件整体成功率从 86.4% 提升至 88.4%（+2.0pp），Long 套件最大提升 7.0pp（61.6%→68.6%）。增量延迟仅 10.99 ms/动作块，且不需要额外 GPU。

**⚠️ 局限性**

局限性：1）只能检索已有的成功轨迹，对失败或全新场景的泛化有限；2）低频残差校正可能不足以纠正大幅错误；3）需完整的成功记忆库，存储与检索成本随规模增长；4）对极短或极复杂任务的收益相对有限。

---

## 15. A Centralized Performance Monitoring Architecture for Heterogeneous Multicore SoCs

**arXiv ID:** 2608.04247 | [PDF](https://arxiv.org/pdf/2608.04247v1)

**作者:** Mohammed Sajjad Jafri `[一作]` (University of Waterloo), Rodolfo Pellizzoni `[通讯]` (University of Waterloo)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计并实现了一个集中式性能监控架构，利用 EVU 将事件收集到 APMU 进行聚合、计数与处理，支持低干扰的实时监控与调度。

**💡 创新点**

通过将事件产生与处理分离，提供统一的 EVU‑APMU 接口、可编程计数器与专用处理单元，实现跨异构 SoC 组件的低延迟事件收集与实时决策。

**🔧 技术方法**

AXI4 snooping 单元、CVA6 事件单元、IBEX RISC‑V APMU‑PE 自定义指令、事件驱动的等待指令、软硬件协同的加载‑运行编程模型。

**📊 数据集**

Synthetic 微基准（LLC_r、LLC_rw、MM_r、MM_rw）与 SDVB（Disparity、MSER、Stitch）工作负载。

**📈 对比分析**

与传统软件轮询/中断的 Linux perf/PEBS 方案对比，证明内存调度循环响应时间缩短、执行时序更稳定、资源占用仅占平台 LUT 的 4.5% 及计数器占用 0.05%。

**⚠️ 局限性**

受限于 32 个计数器上限、对 FPGA 低频模拟导致无法完全体现高频内存争用、缺乏完整操作系统与虚拟化集成、未评估功耗与面积在 ASIC 上的真实影响。

---

## 16. Enabling Urgency-aware Robot Swarm Intralogistics using Smart IoT Tags

**arXiv ID:** 2608.04721 | [PDF](https://arxiv.org/pdf/2608.04721v1)

**作者:** Youssef Alboraei `[一作]` (University of Bristol), Kerstin Eder `[通讯]` (University of Bristol)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文设计并实现了一套基于智能低功耗IoT标签的分散化仓储物流系统，利用DOTS机器人和BLE标签广播物品紧急度，实现机器人在不依赖中心调度的情况下，直接依据物品自身传递的优先级进行选择与配送。

**💡 创新点**

创新点在于将物品紧急度嵌入智能标签，实现“信息随物品”而非“集中调度”，使得分散化机器人群能够在本地感知并即时响应时间敏感任务，显著提升紧急物品的服务优先级。

**🔧 技术方法**

使用技术包括：nRF52833 BLE SoC与E‑ink显示的定制IoT标签；BLE广告与GATT写入；ArUco标记与摄像头定位；机器人行为树控制器；低功耗双频通信；Python二维仿真器；以及DOTS机器人硬件平台。

**📊 数据集**

数据集：实验中使用随机生成的物品等待时间与紧急度（0–9）进行多场景仿真和物理测试，没有使用公开标准数据集。

**📈 对比分析**

通过与仅靠接近优先级（proximity‑only）基线比较，使用P95延迟、平均延迟、吞吐量以及优先级对齐（PA）等指标。物理实验显示，α=0.7时PA提升至0.64（相较基线0.41）且P95延迟下降约9%；吞吐量保持在基线的1.2%以内。仿真扩展至更大规模时，PA提升41–52%，P95延迟下降5–12%，且吞吐量损失随规模增长而显著降低。

**⚠️ 局限性**

局限性包括：仅在小规模物理环境验证，未在真实大型仓库中测试；高紧急度权重可能导致资源拥堵；依赖BLE通信的可靠性与范围；未深入探讨动态优先级更新与环境变化的适应；以及缺乏对不同物品尺寸/重量等多样化物流场景的评估。

---

## 17. A Separator-based Algorithm for the Graph Edit Distance Problem

**arXiv ID:** 2608.04583 | [PDF](https://arxiv.org/pdf/2608.04583v1)

**作者:** Laura Bülte `[一作]` (University of Bonn), Petra Mutzel `[通讯]` (University of Bonn)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的精确计算图编辑距离（GED）的算法SR-GED，利用图的分离器实现递归分治。

**💡 创新点**

创新点在于通过适配节点编辑成本与分离器相结合，实现子图的独立求解，从而将原本n!的最坏情况降低到(4+ε)^n 的指数时间。

**🔧 技术方法**

技术手段包括基于平衡分离器的递归分治、节点映射的适配编辑成本、分离器映射枚举与子图划分的双层循环，以及多项式空间的递归实现。

**📊 数据集**

实验使用 GEDLIB 公开基准数据集，验证了算法在真实实例中的可行性。

**📈 对比分析**

与现有最优的树搜索/ILP 方法相比，SR-GED 在理论上将最坏时间从 n! 降到 (4+ε)^n，且在树宽小、Planar 等类上可达 4^n 的速度；实验表明对 GEDLIB 数据的性能有显著提升。

**⚠️ 局限性**

局限性在于只适用于至少一方图具有严格子线性分离器的 Hereditary 图类；在一般图上无法突破 ETH 限界；实际实现仍需工程化，递归枚举规模大，未给出完整的实测复杂度。

---

## 18. Towards Datalog on Quantum Annealers: Compiling Recursive Logic Programs with Bottom-up Semantics to 2-local Ising Models

**arXiv ID:** 2608.04645 | [PDF](https://arxiv.org/pdf/2608.04645v1)

**作者:** Bruno Rucy Carneiro Alves de Lima `[一作]` (University of Tartu), Joseph Haske `[通讯]` (Independent Researcher)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将递归 Datalog 程序编译成 2‑local Ising 模型，使其基态对应于程序的最小 Herbrand 模型；

**💡 创新点**

提供了完整的四阶段编译器（二元化、归一化、Min‑Ones SAT、Ising 编码），并在 Lean 4 中形式化验证每个阶段的正确性和整体对应定理；

**🔧 技术方法**

使用二元化、完整归一化、CNF 转化为 Min‑Ones SAT、基于门函数的 Ising 编码（含辅助变量）以及量子退火模拟（经典 SA 与 D‑Wave 的 sqa）进行实验；

**📊 数据集**

评估数据集包括若干 Datalog reachability 程序（tc_path、tc_cycle、nonlinear_tc_path、linear_chain）以及在 Zephyr 硬件图上的单源可达性；

**📈 对比分析**

通过将编译得到的 Ising 模型在模拟器中采样，计算基态恢复率；在小规模实例（≤10 个原子）恢复率接近 1，随规模和自支持性增加迅速下降；与经典 Datalog 引擎对比，后者在毫秒级完成；

**⚠️ 局限性**

受限于硬件精度（可区分的位数）、逻辑度数增长导致的链长度、以及自支持（unfounded）陷阱对能量景观的影响，导致在芯片级规模时根本无法恢复最小模型。

---

## 19. Geometry-Informed Parameter-Efficient Fine-Tuning of Pre-trained Molecular GNNs for Blood-Brain Barrier Permeability Prediction

**arXiv ID:** 2608.04257 | [PDF](https://arxiv.org/pdf/2608.04257v1)

**作者:** Marco Vieto Vega `[一作]` (Victoria University of Wellington), Binh P. Nguyen `[通讯]` (Victoria University of Wellington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种几何信息驱动的参数高效微调框架BBBP-GeoPEFT，用于血脑屏障通透性预测

**💡 创新点**

创新点在于：①在预训练GNN上引入多尺度距离图和对应的线图，以捕捉三维空间原子相互作用和二阶边缘关系；②设计节点级切点注意力机制，将几何表示与冻结的预训练层融合，既保持知识迁移，又显著降低可训练参数；③使用低秩因式分解提升辅助几何编码的参数效率

**🔧 技术方法**

技术包括：图神经网络（GIN backbone）、辅助几何图编码器、节点级切点注意力与门控残差连接、低秩因子化、RDKit conformer生成、线图构造、数据集分割（随机/骨架拆分）

**📊 数据集**

使用经过清洗的BBB数据库（约3832条样本，BBB+ 2446，BBB- 1386）

**📈 对比分析**

与全微调、Adapter、LoRA、GPF、GPF-plus、AdapterGNN等PEFT基线以及不同自监督预训练（SimGRACE、EdgePred、ContextPred）进行对比。BBBP-GeoPEFT在随机和骨架拆分下均取得最佳或第二佳的ROC‑AUC、PR‑AUC、准确率、F1，并且只更新约10.1%的参数；相比全微调仅略逊，优于所有提示式方法

**⚠️ 局限性**

局限性：依赖RDKit生成的 conformer 可能不完全准确，特别是柔性分子；多尺度图构造与辅助编码引入额外计算开销；对某些预训练目标（如EdgePred）效果不如强基线

---

## 20. InvFlowFD: Reference-Free and Background-Set-Free Perceptual Music Quality Metric with Flow Matching Inversion

**arXiv ID:** 2608.04142 | [PDF](https://arxiv.org/pdf/2608.04142v1)

**作者:** Alon Ziv `[一作]` (Hebrew University of Jerusalem), Yossi Adi `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出一种不需要背景音频集且无参考的音乐质量评估指标，通过预训练的 Flow Matching 模型进行无条件反向采样，并将反向采样得到的潜在分布与先验分布进行 Frechet 距离比较，从而量化音频的感知质量。

**💡 创新点**

创新点在于：①完全消除对背景集的依赖，避免了 FAD 等方法因背景集不同导致的评价偏差；②利用 Flow Matching 的先验空间进行分布距离度量，使评估更具泛化性；③同时提出样本级指标 CS，兼顾分布级和样本级质量评估。

**🔧 技术方法**

技术包括：预训练的 JASCO-400M‑chords‑drums Flow Matching 模型、EnCodec 128 通道潜在编码、Euler 逆向积分（100 步）、Frechet Distance 计算、MMD 与 MAUVE 对比、人工对比评测以及 Plackett‑Luce 价值估计。

**📊 数据集**

使用的数据集包括 MTG‑Jamendo、FMA‑small 进行实验评估；用于生成模型评价的文本提示来自 MTG‑Jamendo 测试集标签；人工评测采用从 Jamendo 随机抽取的 100 条 10 秒音频做增强（白噪声、低通/高通滤波、crop‑and‑paste）来收集人类偏好。

**📈 对比分析**

与 CLAP‑based FAD 在多种人工畸变（白噪声、低通/高通滤波、crop‑and‑paste）下进行对比。FAD 对不同背景集的依赖导致评价不稳定，而本方法对畸变的响应单调、敏感度高；在人类评测中的 Pearson 相关性最高达 0.73，优于 FAD；在评估生成模型的整体质量时，本方法能够正确排序且不需要背景集，表现与 FAD 相当。

**⚠️ 局限性**

局限性包括：①未解决 FAD 的慢收敛问题；②对极端或高度复杂畸变的敏感度尚不充分；③在极细微频段畸变上，CS 指标更敏感但需要多次前向传播，计算成本较高；④依赖于已预训练的 Flow Matching 模型，若模型训练不充分，评估效果可能受限。

---

## 21. Teaching MLLMs to Say No: Generalized Referring Expression Comprehension via Refusal Calibrated GRPO

**arXiv ID:** 2608.04698 | [PDF](https://arxiv.org/pdf/2608.04698v1)

**作者:** Xuzheng Yang `[一作]` (University of Electronic Science and Technology of China), Peng Wang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于强化学习的后训练框架 RC‑GRPO，用于让多模态大型语言模型在 generalized referring expression comprehension（GREC）任务中既能准确定位存在的目标，又能在目标不存在时给出拒绝答案。

**💡 创新点**

创新点在于引入拒绝校准机制——强制拒绝采样、对比奖励与负样本优势缩放，以及第二阶段的解释强化，三者协同实现了定位精度与拒绝可靠性的平衡。

**🔧 技术方法**

技术方法包括基于 GRPO 的组相对策略优化、强制拒绝 roll‑out、对比奖励函数、负样本优势缩放系数以及基于规则的理由奖励，并通过 LoRA 进行轻量化微调。

**📊 数据集**

实验使用 FineCops‑Ref（2k 样本子集）、gRefCOCO 和 D3 这三个 GREC 数据集，并在 MME、MMBench、POPE 上评估模型的通用性。

**📈 对比分析**

与 UNINEXT、HieA2G、InstanceVG 等专用方法以及 Qwen、Ferret、ROD‑MLLM、CRS 等 MLLM 基线进行比较，RC‑GRPO 在所有三大 GREC 基准上均显著提升了 Precision 与 N‑acc，同时保持甚至提升了 P‑acc，最高可达约 10% 的绝对增益。

**⚠️ 局限性**

局限性包括需要对负样本优势缩放因子 α 进行精细调参，且对极难的负样本仍可能出现误拒；在不同 Backbone 上的泛化性能仍需进一步验证，且后训练过程相对耗时。

---

## 22. Looking in the Mirror: Introspecting Side-Effect Misalignments Induced by Fine-Tuning

**arXiv ID:** 2608.04347 | [PDF](https://arxiv.org/pdf/2608.04347v1)

**作者:** Kotaro Yoshida `[一作]` (Institute of Science Tokyo), Wenya Wang `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在非对抗性微调后模型的侧效对齐性变化，并构建了侧效 introspection 数据集；提出了 Delta‑Aware Introspection Adapter（DAIA）来显式利用微调差异进行自我报告。

**💡 创新点**

①定义了侧效 introspection 问题，首次基于模型对齐分数差生成监督标签；②设计了能并行处理基模型激活和微调差异的 DAIA 结构，提升对侧效对齐变化的感知。

**🔧 技术方法**

采用 LoRA 轻量化微调、LoRA‑based introspection adapter 训练、Delta‑Aware adapter、自然语言问答生成、LLM‑评测、激活补丁分析等技术。

**📊 数据集**

使用 213 个来自 Hugging Face 的 LoRA 微调模型（基于 Qwen3‑14B、Gemma3‑12B‑it），以及 13 个统一对齐类别（从 StrongREJECT、HarmBench、HEx‑PHI、AdvBench 汇总）的 1,523 个评测样本。

**📈 对比分析**

与随机、众数预测、原始微调模型、Probe 以及标准 LoRA introspection 进行对比；在三种 OOD 场景（未见模型/未见类别）下，DAIA 在准确率和宏 F1 上均优于 LoRA，表明具有良好的泛化能力；Probe 与 introspection 方法的性能相近。

**⚠️ 局限性**

当前 introspection 方法本质上类似内部状态分类器，缺乏生成自由文本报告的能力；类别划分有限，导致对未见类别的泛化受限；基模型固定假设限制了在模型版本更新场景中的适用性；尚未验证在更开放式自我报告任务上的表现。

---

## 23. Test, then Route: How Language Models Execute In-Context Conditional Rules Across Models and Languages

**arXiv ID:** 2608.04183 | [PDF](https://arxiv.org/pdf/2608.04183v1)

**作者:** Luxshan Thavarasa `[一作]` (Independent Researcher), Sivasuthan Sukumar `[通讯]` (University of Moratuwa)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究语言模型如何在上下文中执行条件规则，利用四个对抗性提示通过激活补丁定位谓词测试与答案路由的残差通道。

**💡 创新点**

创新点在于设计了条件与答案不一致的四个donor提示，揭示了谓词测试在中层可分离、可独立定位，而答案路由不具备可迁移的分离子空间。

**🔧 技术方法**

采用激活补丁、交叉验证、分布式对齐搜索（DAS-lite）以及头组贪婪选择等技术，对多模型多语言的残差流进行因果定位。

**📊 数据集**

使用一个固定的250条项目数字项库，在六种语言（英语、中文、印地语、印尼语、泰米尔语、僧伽罗语）中翻译规则和演示，保持答案词为拉丁单词，评估多语言一致性。

**📈 对比分析**

在Gemma-3-4B、Gemma-3-12B和Qwen3-8B三大模型中，谓词测试模块在18个实验单元中均达到“隔离测试”通过率≥97%；但路由子空间未能跨标签或跨语言迁移，表明其不可分离、不可迁移。

**⚠️ 局限性**

局限包括仅测试开放权重模型，未验证更大规模或商业模型；任务为合成条件规则，未覆盖自然语言真实情境；路由子空间在Gemma-3-12B的完整头组几乎无效，仅依赖子空间方法；跨语言路由测试仅在Gemma-3-4B上完成。

---

## 24. The First EgoCross Challenge at EgoVis 2026: Cross-Domain Egocentric Video Question Answering

**arXiv ID:** 2608.04589 | [PDF](https://arxiv.org/pdf/2608.04589v1)

**作者:** Yuqian Fu `[一作]` (King Abdullah University of Science and Technology), Wenping Ma `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了EgoCross跨域视角视频问答基准和挑战赛，评估多模态大语言模型在手术、工业装配、极限运动和动物视角等四个专业领域中的泛化能力。

**💡 创新点**

创新点包括：①首次构建跨域视角VQA基准，覆盖非日常专业领域；②设计Source-Limited和Open-Source两条竞赛轨道，兼顾模型与数据限制；③提出域感知推理、反射式对话以及时间标记注入等多种技术方案，提升跨域推理效果。

**🔧 技术方法**

主要技术包括：Qwen3-VL-4B基础模型、SFT微调、域感知推理框架（DomainWiseInfer、OmniEgo-R²）、反射式多轮对话（Reflective Dialogue）、时间标记注入与LoRA适配（TokenInj-RAA）以及检索增强推理。

**📊 数据集**

使用EgoCross数据集，包含798条视频剪辑和957个多选问答，覆盖手术、工业装配、极限运动和动物视角四个领域，并提供80条支持集样本。

**📈 对比分析**

通过两条官方轨道的排行榜评测，获奖方案在Source-Limited轨道平均精度达66.98%，相比官方基准提升约20.9个百分点；Open-Source轨道最高精度为66.98%，与Source-Limited相近。整体表现仍显不足，特别是手术和极限运动领域。

**⚠️ 局限性**

局限性包括：①跨域泛化仍受限，某些专业领域（尤其手术、极限运动）准确率低；②支持集样本极少，难以充分捕获域内细粒度差异；③模型仍需更多时间序列建模与领域知识融合，未完全解决视频时序和细粒度视觉推理挑战。

---

## 25. Dense Metric Depth Completion from Sparse Direct Time-of-Flight Sensors

**arXiv ID:** 2608.04737 | [PDF](https://arxiv.org/pdf/2608.04737v1)

**作者:** Hakyeong Kim `[一作]` (KAIST), Min H. Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可跨不同 dToF 传感器、稀疏度与噪声条件下实现稠密度量深度重建的通用框架；

**💡 创新点**

创新点在于：①深度引导双分支 Vision Transformer 编码器配合带方向掩蔽的联合注意力，实现RGB与稀疏深度的高效且受控融合；②全合成的 dToF 仿真流水线，精准模拟闪光与旋转 dToF 的采样模式、噪声与硬件失真；③仅靠合成数据即可实现强大的零样本泛化；

**🔧 技术方法**

技术包括：双分支 ViT + 掩蔽联合注意力；轻量化 DPT 解码器；log‑normal 化 + 3 通道深度输入；多项损失（L1、尺度不变损失、掩模损失）；合成数据生成与多种噪声/稀疏度增强；

**📊 数据集**

使用了 KITTI-DC、ZJUL5、DDAD、DIODE、ETH3D、iBims-1 等真实 dToF 数据集；并在多达 6 个模拟场景（闪光、子 VGA、旋转 LiDAR 等）上训练与评估；

**📈 对比分析**

与 OMNI‑DC、Marigold‑DC、PromptDA、PriorDA 等主流方法对比，零样本设置下在大多数数据集上实现了相近或更优的相对误差（Rel）与阈值准确率（δ），同时推理速度提升约 20×、显存消耗下降约 10×，证明了高效性与准确性的双重优势；

**⚠️ 局限性**

局限性在于：①对极端极低分辨率（如 8×8）或极高噪声场景的鲁棒性仍有提升空间；②合成仿真虽然覆盖多种情况，但仍难以完全捕捉所有实际硬件细节；③在大尺寸 8K/4K 高分辨率图像上推理仍受限于显存；

---

## 26. Thinking with Anchors: Grounded and Efficient Document Reasoning

**arXiv ID:** 2608.04424 | [PDF](https://arxiv.org/pdf/2608.04424v1)

**作者:** Sichen Zhu `[一作]` (Georgia Tech), Jiuxiang Gu `[通讯]` (Adobe)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了一个包含视觉锚点、语义标签、图文链式推理的文档理解数据集，并针对该数据集设计了检测、分割、语义标注与计数等基准任务，

**💡 创新点**

提出将文档区域视作可语义化、可空间关联的“视觉锚点”，并通过Agentic聚合与多头解码实现高效的密集输出与推理，

**🔧 技术方法**

采用YOLOv12、RF‑DETR、SAM、LocateAnything等视觉模型以及多模态LLM进行检测、分割、标签与计数的端到端实验，

**📊 数据集**

使用扩充后的120k页文档数据集（原始页面锚点+人类标注的标题、语义标签、CoT链式推理），

**📈 对比分析**

与现有非VLM检测/分割基线对比，发现VLM在零样本下表现欠佳，细调后可提升约50–70分AP；Agentic聚合可显著降低过拆分错误；计数任务中VLM仍远逊于专业模型，

**⚠️ 局限性**

局限性包括跨域（自然图像→文档）适配差距、长尾标签分布导致稀有类别性能低、以及对密集布局中文文档的支持不足

---

## 27. Embedding Large Language Models into Flow Controls: An Agentic Framework for Adaptive and Trustworthy Automated Cooking

**arXiv ID:** 2608.04768 | [PDF](https://arxiv.org/pdf/2608.04768v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 28. Adaptive Intrusion Detection System using Transformer-Based Neural Networks and Continual Learning Approach with Adversarial Investigation

**arXiv ID:** 2608.04602 | [PDF](https://arxiv.org/pdf/2608.04602v1)

**作者:** Azizi Ariffin `[一作]` (Universiti Teknologi MARA), Nor Badrul Anuar `[通讯]` (Universiti Malaya)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种利用tabular transformer编码器与善意锚定的类平衡经验重放缓冲区的自适应IDS框架，支持持续学习并引入更贴近生产的类实例递增（CII）评估场景。

**💡 创新点**

创新点在于：1）首次将Transformer在网络流表格特征上的自注意力应用于IDS；2）通过在每次更新中持续重放正类流（benign）来消除灾难性遗忘；3）提出CII场景，以逼真方式评估持续学习；4）对重放缓冲区进行对抗性评估，揭示标签翻转和后门攻击的危害。

**🔧 技术方法**

采用TabTransformer作为特征编码器，结合经验重放（class‑balanced reservoir sampling）、AdamW优化、交叉熵损失，并对比EWC、LwF、iCaRL等持续学习方法。

**📊 数据集**

在CICIDS2017公共数据集（含8类攻击与主导的benign流）上进行实验，划分四个经验。

**📈 对比分析**

与七种持续学习基线比较：在CI与CII场景下，ER‑Balanced（每类1–10%缓冲）在accuracy上接近联合训练（CI: 0.9994，CII: 0.9989），忘记率和不可转移率均低于0.01；相比之下，顺序微调、EWC、LwF等方法在CI场景下几乎完全失效；在对抗实验中，标签翻转导致模型崩溃，后门攻击保持高准确率但攻击成功率可达95–100%。

**⚠️ 局限性**

局限性包括：1）评估仅基于单一CICIDS2017数据集；2）对缓冲区大小在更长时间跨度或更复杂流分布下的表现未作充分验证；3）未探索更复杂的后门或缓冲区攻击；4）对Transformer与其他编码器的贡献尚未彻底剖析。

---

## 29. NOLLI: A Difficulty-Calibrated Puzzle Benchmark for Diagnosing the English-Korean Performance Gap

**arXiv ID:** 2608.04397 | [PDF](https://arxiv.org/pdf/2608.04397v1)

**作者:** Dasol Choi `[一作]` (AIM Intelligence), Seunghyeok Hong `[通讯]` (Hankuk University of Foreign Studies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并发布了Nolli，一个基于程序生成的英韩推理基准，包含15种题型、25个任务、7,500个唯一解实例，并通过参考模型实现行为难度校准。

**💡 创新点**

创新点在于：①三层跨语言设计（直接翻译、文字脚本适配、韩语专属任务）；②使用目标准确率带进行行为难度刻度；③诊断性差距分析，将展示语言、书写系统与韩语特定知识分离。

**🔧 技术方法**

采用程序化生成与唯一解验证、参考模型目标带校准、确定性 exact‑match 评估、HRET 推理、统计检验（TOST、相关分析）和错误分解技术。

**📊 数据集**

使用了7,500条可种子重生成的谜题实例，涵盖15种题型、25个任务、3难度层级，并在12个不同模型上进行评估。

**📈 对比分析**

通过将15个模型按开发者组划分，使用精准的 exact‑match 评分比较性能；发现展示语言差距≤10pp，写字系统在韩语 Cipher 上可达68.7pp缺口，Cryptarithmetic 无显著差距，韩语专属任务呈现规则适用缺陷；宏观准确率从0.1%至84.9%。

**⚠️ 局限性**

局限性包括：难度校准仅基于单一参考模型与推理强度；脚本适配单独校准；模板语义等价假设；子音素机制仅为相关性；韩语专属对比受任务混淆；仅针对精确匹配谜题，未涵盖部分信用。

---

## 30. A 6G Integrated Sensing and Communication Framework for Railway Intrusion Detection and Collision Prediction

**arXiv ID:** 2608.04710 | [PDF](https://arxiv.org/pdf/2608.04710v1)

**作者:** Ajeet Kumar Yadav `[一作]` (Indian Institute of Science), Pandarasamy Arjunan `[通讯]` (Indian Institute of Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了基于6G ISAC技术的铁路入侵检测与碰撞预测框架，能够通过无线信道状态信息（CSI）检测入侵并实时预测入侵者位置、速度与碰撞时间。

**💡 创新点**

创新点在于：①利用仿真生成大规模高质量CSI数据并结合物理渲染场景；②提出子载波选择、静态成分去除、Wiener滤波等预处理流程；③设计3D卷积+双向LSTM混合网络，实现入侵检测与多参数回归的统一学习。

**🔧 技术方法**

技术方法包括：Blender三维场景渲染 + NVIDIA Sionna射线追踪仿真生成CSI；信号处理预处理（子载波去除、SNR加权选择、静态分量去除、Wiener滤波）；深度学习模型3D CNN+BiLSTM，并用Optuna进行超参数优化。

**📊 数据集**

使用了约22,695个合成CSI矩阵数据集，涵盖两种无线配置（10×10、7×7天线阵列等），并标注入侵状态、相对位置、速度与碰撞时间。

**📈 对比分析**

在两轮验证实验中，分类准确率最高可达99.99%，相对位置、速度和时间预测的MAE分别为0.42 m、0.02 m/s和0.18 s，整体性能显著优于传统单一检测方法。

**⚠️ 局限性**

局限性：仅基于仿真CSI，缺乏真实铁路环境的测量验证；对小尺寸或低速移动入侵的识别效果有限；模型对大MIMO阵列和较高发射功率有较强依赖。

---

## 31. Enacting Constructive Conflicts with AI Agents to Enhance Reconsideration among Novice Interaction Designers

**arXiv ID:** 2608.04166 | [PDF](https://arxiv.org/pdf/2608.04166v1)

**作者:** Howard Ziyu Han `[一作]` (Carnegie Mellon University), Nikolas Martelaro `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1582 | [OpenAlex ID](https://openalex.org/A5075763217)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并评估了一个对抗性 AI 代理，帮助初学交互设计师在多方利益相关者环境中进行建设性冲突，以提升设计再思考与改进。

**💡 创新点**

将“对抗性”与“建设性冲突”相结合，构建交互式 AI 代理能主动提出利益相关者的推挽点，区别于传统同意式提示，显著刺激设计师的迭代和修改行为。

**🔧 技术方法**

基于 GPT‑4.1 生成冲突点，并集成于 Miro 白板；使用 Next.js、Miro SDK、Firestore、Whisper 与 GPT‑4o 处理语音与视觉；实现对话式推挽流程。

**📊 数据集**

收集六位公共设计专家的对抗性案例与文献作为知识库；实验使用 45 名交互设计学生的设计方案（非公开数据集）。

**📈 对比分析**

采用三组实验（自我反思、分步指导、交互参与）进行对比，使用 Kruskal‑Wallis、ANOVA、Tukey 与 Dunn 检验；结果显示交互参与组在想法增删改、冲突接受和设计再思考方面显著优于基线，效应量中等至大。

**⚠️ 局限性**

样本量有限，效应可能被放大；仅使用单一对抗语气，未检验不同对抗强度；仅针对初学者，未涉及团队或真实利益相关者；模拟推挽不等同真实反馈。

---

## 32. Equitable System-Prompt Selection via Constrained Mixed-Strategy GroupDRO

**arXiv ID:** 2608.04339 | [PDF](https://arxiv.org/pdf/2608.04339v1)

**作者:** Mengyu Xu `[一作]` (University of Chicago), Chongyang Gao `[通讯]` (Northwestern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

针对同一信息检索问题不同表述导致答案质量差异的现象，本文提出一种基于约束混合策略GroupDRO的系统提示选择框架，使用已有提示池中的权重分配来最小化各评估指标和分组下的最坏情况损失，同时保持平均质量与平均选择相近。

**💡 创新点**

创新点在于将分组鲁棒优化（GroupDRO）与系统提示选择相结合，并通过约束平均损失实现最佳平衡；同时，混合策略的权重可揭示提示池中互补的提示，从而超越单一提示的性能。

**🔧 技术方法**

使用线性规划对混合策略GroupDRO进行求解，构造损失矩阵后通过求解带有平均损失约束的LP获得最佳权重；并结合PRC、Pure GroupDRO等对比方法进行评估。

**📊 数据集**

实验使用两套双语基准：MIRA（医学信息检索）和新构建的消费者金融基准（60个种子问题、24个评估分组），涵盖信息稀释、完整性、可操作性三项指标。

**📈 对比分析**

相较于无缓解、平均选择、Pure GroupDRO、混合GroupDRO和PRC，约束混合GroupDRO在10个模型-领域组合中平均提升约13%，最坏情况和最低25%分组的质量均显著下降，且平均质量几乎不变，展示出显著的公平性和鲁棒性提升。

**⚠️ 局限性**

局限性包括依赖离线评估得分和预先定义的分组，无法保证在未知或极端表述上的表现；提示池规模和多样性限制了方法的可扩展性；求解LP需要额外计算资源；且未对真实用户实验或动态场景中的鲁棒性进行验证。

---

## 33. Governing Execution Risk in Agentic AI Systems: A Trajectory-Guided Framework for Red Teaming

**arXiv ID:** 2608.04018 | [PDF](https://arxiv.org/pdf/2608.04018v1)

**作者:** Zhihao Zhu `[一作]` (Hong Kong University of Science and Technology), Yi Yang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 83397 | [OpenAlex ID](https://openalex.org/A5005421447)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于执行轨迹的红队框架 TrajRed，用来生成并评估针对 AI 代理的间接提示注入攻击，并进一步构建了 TrajGuard 运行时治理层，用红队发现的高危轨迹来实时监控并中断恶意执行。

**💡 创新点**

创新点在于：①将攻击视为轨迹级过程，用轨迹进度作为反馈；②采用模板化攻击生成并通过加权 DPO 学习提升轨迹进度；③将红队轨迹记忆转化为运行时风险评估门控，实现从红队到治理的闭环。

**🔧 技术方法**

技术包括：LLM 生成式红队（Qwen/Gemma 生成器）、轨迹进度打分（匹配参考恶意工具调用序列）、加权 DPO（依据轨迹得分差值加权）、LLM 风险评估门（动作门和内容门）以及基于 AgentDojo 的仿真环境。

**📊 数据集**

数据集：AgentDojo benchmark，涵盖 workspace、slack、travel、banking 四个工作流套件，使用单点注入任务，训练/测试分割在每个套件内部完成。

**📈 对比分析**

比较方法：固定模板基线（Direct、Ignore Previous、Important Instructions 等）、自动红队基线（AdvAgent、RL‑Hammer）以及内部 ablation 版本。TrajRed 在 Qwen 系列实验中，攻击成功率（ASR）提升至 41.8%（对比 AdvAgent 的 34.9%），轨迹进度分数（TPS）提升至 46.6%（对比 41.9%）。在治理层评估中，TrajGuard 将 ASR 降至 0%（或 0.1%），TPS 降至 2.7%，同时保持 52.7% 的正常任务效能（与无防御 54% 接近）。

**⚠️ 局限性**

局限性：①只评估单点注入场景，未考虑多点或持续交互攻击；②实验主要集中在 Qwen/Gemma 模型，跨模型效果下降；③轨迹记忆和门控基于已知恶意模式，可能对未知攻击失效；④未系统评估运行时开销和部署复杂度；⑤对长周期、持续记忆或多代理协同任务的泛化有限。

---

## 34. A Distributed Quantum Approximate Optimization Algorithm For Unit Commitment

**arXiv ID:** 2608.04159 | [PDF](https://arxiv.org/pdf/2608.04159v1)

**作者:** Ali Rajabi `[一作]` (Louisiana State University), Amin Kargarian `[通讯]` (Louisiana State University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在单元承诺问题中提出并实现了基于分布式量子近似优化算法（DQAOA）的三块ADMM框架，将二进制决策块转化为QUBO并通过DQAOA求解；

**💡 创新点**

创新点在于：①将DQAOA统一接口嵌入三块ADMM，使得二进制块可按需切换为暴力枚举、单机QAOA或分布式QAOA；②在同一UC实例中验证三种模式得到完全相同的最优调度，展示分布式QAOA可跨多QPU满足容量限制而不影响解质量；

**🔧 技术方法**

使用技术包括：三块ADMM分解、QUBO编码、量子近似优化算法（QAOA）、分布式QAOA（通过TeleGate实现跨QPU交互）、经典Adam优化器、量子电路采样等；

**📊 数据集**

使用的数据集为五机组、三时段的单元承诺实例，负荷曲线[L=60,130,280] MW，机组参数表（A_i, B_i, C_i, P_min, P_max）共15个二进制决策变量；

**📈 对比分析**

比较方法为在相同UC数据、ADMM参数、初始值与收敛阈值下，分别采用暴力枚举、单机QAOA、分布式QAOA求解二进制块；结果显示三种模式都在ADMM原始残差<1e-3后收敛，恢复同一调度、发电量与成本$12,678，表明分布式QAOA性能与单机QAOA相当；

**⚠️ 局限性**

限制主要有：仅在小规模实例（15二进制变量）验证，未测试更大规模或具有噪声/硬件约束的量子设备；分布式QAOA由于跨QPU通信导致额外开销，未在运行时间上体现加速；仅考虑单一UC模型，未覆盖多目标或更复杂约束情况。

---

## 35. An Analysis and Implementation of Seam Carving for Content-Aware Image Resizing

**arXiv ID:** 2608.04329 | [PDF](https://arxiv.org/pdf/2608.04329v1)

**作者:** Francesco Tosoni `[一作]` `[通讯]` (Sant'Anna School of Advanced Studies), Francesco Tosoni (Sant'Anna School of Advanced Studies)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

实现了一套完整的 C++17 seam carving 算法，支持图像缩放、按序列插入 seam 进行放大、前向能量（forward‑energy）以及用户自定义掩码用于对象保护或移除，并提供并行化（OpenMP）和复杂度分析。

**💡 创新点**

创新点包括：① 在实现中同时支持原始后向能量与后向能量；② 采用多通道梯度能量计算与可选 RGB 直接能量；③ 通过掩码实现一张图像中既保护又移除对象；④ 在放大时按顺序先找删除 seam 再插入，以避免单一路径拉伸；⑤ 对实现进行了细粒度并行化、内存优化并公开源码。

**🔧 技术方法**

核心技术：动态规划求最小能量 seam；梯度幅值能量计算（L^2 或 L^1）；前向能量成本公式；掩码加权；多通道图像处理；C++17 语法、OpenMP 并行；图像 I/O 使用公共域 header。

**📊 数据集**

实验使用多张公开自然图像（如 Jaipur 门厅、巴黎桥梁、鸟类、风景等），无专门训练数据集；所有结果均来自作者自行生成，展示不同能量模式与掩码效果。

**📈 对比分析**

对比方法：展示后向能量、前向能量、统一缩放和按顺序 seam 插入的放大效果；通过可视化 seam 轨迹说明前向能量在保持直线结构方面更优。性能：算法复杂度为 O(c · n · m)（c 为需删除/插入的 seam 数），内存占用 O(n · m)。多线程实现显著提升速度，单线程时后向能量动态规划仍保持较快。

**⚠️ 局限性**

局限性：若图像中重要内容密集、无低能量路径，则 seam carving 会破坏结构；全宽重要内容时无法避开；前向能量虽然减少伪影，但可能牺牲部分内容；梯度能量仅捕捉低级边缘信息，对语义重要性不足时需人工掩码。

---

## 36. GASP: GPU-Accelerated Safe Planner for Real-Time Collision-Aware Motion Generation with Latent Trajectory Sampling

**arXiv ID:** 2608.04612 | [PDF](https://arxiv.org/pdf/2608.04612v1)

**作者:** Colin Merk `[一作]` (Sony), Farshad Khadivar `[通讯]` (Sony)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种GPU加速的可采样安全规划器GASP，能够在已知静态环境中实时生成碰撞感知的关节空间轨迹。

**💡 创新点**

创新点在于：1) 将端点速度/加速度约束解析到B样条边界控制点，仅学习内部残差；2) 采用条件变分自编码器生成多样化候选轨迹；3) 完全在GPU上批量并行推理、评估与选择。

**🔧 技术方法**

使用技术包括：clamped B-spline 参数化、卷积残差网络、CVAE 隐变量、差分可碰撞惩罚、GPU并行评估与多线程候选筛选。

**📊 数据集**

数据集来源于在线采样生成的边界条件对（初始/目标姿态+时程），并使用相同采样流程生成验证集；对比实验也包含cuRobo、Ruckig 与MLP基线的测试数据。

**📈 对比分析**

通过与解析规划器Ruckig、GPU优化器cuRobo以及MLP基线在相同测试集上的对比，GASP在成功率上与Ruckig持平且显著优于cuRobo，同时推理时间保持在近毫秒级，满足实时要求。

**⚠️ 局限性**

局限性包括：训练需针对每台机器人与运动学约束；碰撞检测基于静态原语几何，无法捕捉动态障碍；以及损失权重需手动调优，缺乏自适应机制。

---

## 37. LEGOUI: Designing with UI-DSL Bricks to Balance Transparency and Controllability

**arXiv ID:** 2608.04293 | [PDF](https://arxiv.org/pdf/2608.04293v1)

**作者:** Yinsi Zhou `[一作]` (University of New South Wales), Gelareh Mohammadi `[通讯]` (University of New South Wales)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 LegoUI，一种分阶段的生成式用户界面设计框架，利用 UI-DSL 将生成过程拆解为需求分析、布局、交互、关系和样式等可见的中间步骤，并通过可追溯的 provenance 让用户能够在每一步主动接受、拒绝或添加设计决策，最终得到可渲染的界面。

**💡 创新点**

创新点包括：
• 将整个 UI 生成过程拆成可见、可编辑的阶段化步骤；
• 设计并使用 UI-DSL 作为持久化、可追溯的中间表示；
• 在每个阶段实时展示模型推理来源，让用户在生成之前就能干预；
• 通过 provenance 记录决策来源，支持跨迭代的持续性和可追溯性。

**🔧 技术方法**

技术手段：
• 大型语言模型（Claude Sonnet 4、GPT‑4o、Gemini 2.5）进行结构化推理与代码生成；
• Vue3+Ant Design 前端实现 UI 预览与交互；
• FastAPI+Python 后端负责模型调用与 DSL 解析；
• UI‑DSL 语法与原型生成器，将 DSL 转为 HTML/CSS 代码。

**📊 数据集**

数据集：
• 40 条标准化长文本 UI 设计提示，来自 UIPrompt 网站并统一为 150–200 词、6–10 组件、2–4 交互、1–3 风格约束；
• 用于技术评估的基准任务（需求分析）与用户研究的 15 名参与者实验。

**📈 对比分析**

比较与性能：
• 需求分析阶段在 40 条提示上达到 95%+ 的匹配率，覆盖率近 100%，冗余率 0%；
• 用户研究显示 LegoUI 在透明度、可控性、意图一致性评估中显著优于一键生成工具（Bolt、Lovable、Vercel V0、Claude Chat），各维度平均得分 4.2+（最高 4.67）且在首选和第二选偏好中占优势；
• 对比实验中，LegoUI 产生的界面在视觉连贯性、可访问性和意图一致性上获得最高分，且被 23 位参与者选为首选。

**⚠️ 局限性**

局限性：
• 语言模型在风格与全局约束提取上误差较多；
• 交互和关系阶段的文本决策往往难以直观判断，导致信息超载；
• 生成的 UI 仅为轻量 HTML，缺乏高级视觉效果与真实数据交互；
• 需要更完善的 consequence‑transparency（决策后果展示）与冲突处理；
• 目前的阶段划分较为固定，缺乏对跨阶段依赖的动态适配。

---

## 38. Towards End-to-End Multilingual Metaphor Processing: Integrating Detection, Translation, and Evaluation

**arXiv ID:** 2608.04260 | [PDF](https://arxiv.org/pdf/2608.04260v1)

**作者:** Jiahui Liang `[一作]` (Leiden University), Lifeng Han `[通讯]` (Leiden University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个端到端的多语言隐喻处理框架，集成了隐喻检测、翻译、人工评估和自动评估四大模块。

**💡 创新点**

创新点在于把隐喻检测与翻译评估互相嵌入，形成自监督的“LLM-as-a-judge”评估链，以及在MetaHOPE基础上扩展多语言、多译文的评估体系。

**🔧 技术方法**

主要技术包括：大型语言模型提示与链式思考(CoT)、检索增强生成(RAG)、MetaHOPE细粒度误差标签、LLM作为评估者的端到端模型。

**📊 数据集**

使用的数据集有：VUAMC、VUA隐喻检测基准；MMTE、AlphaMWE多语隐喻翻译基准；MetaHOPE自建多语翻译对照语料；WMT 2026 Test Suites的多语翻译提交集。

**📈 对比分析**

通过与SOTA监督检测模型、BERT/COMET/LLM翻译系统以及MetaHOPE人工评估的对比，检测准确率从0.65提升至0.78，翻译评估在多译文设置下BLEU下降但MetaHOPE误差占比下降15%，显示框架在捕获隐喻质量方面显著优于传统指标。

**⚠️ 局限性**

局限包括：对跨文化隐喻的覆盖不足、LLM对隐喻细粒度理解仍有限、评估所需的多译文语料生产成本高、对非常规隐喻和隐晦隐喻的处理仍不完善。

---

## 39. Personalized Federated Sparse Adaptation of Time-Series Foundation Models

**arXiv ID:** 2608.04695 | [PDF](https://arxiv.org/pdf/2608.04695v1)

**作者:** Priyanka Nihalchandani `[一作]` (Indian Institute of Science), Pandarasamy Arjunan `[通讯]` (Indian Institute of Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种个性化联邦稀疏适配框架，利用后表示层的混合专家（MoE）在预训练时间序列基础模型（TSFM）上对建筑能耗进行短期预测；

**💡 创新点**

创新点在于将序列级稀疏专家路由与个性化参数分离，允许不同建筑在共享公共表示的同时保留私有专家，且该策略可根据不同TSFM骨干动态选择共享或私有专家；

**🔧 技术方法**

采用预训练TSFM（MOMENT、Chronos‑2、Moirai）作为骨干，后表示层的MoE适配器、top‑k 路由、残差门、FedAvg/FedProx 联邦优化、覆盖式客户端抽样、LoRA 对比；

**📊 数据集**

使用ASHRAE Great Energy Predictor III 数据集的 50 个非住宅建筑的小时电量序列；

**📈 对比分析**

与零样本、全局联邦FL‑MoE、本地MoE、无MoE个性化等基线对比，实验表明私有专家版本在所有骨干上平均降低 NRMSE 约 8.2%‑12.5%，并在部分基准上显著优于本地MoE；

**⚠️ 局限性**

仅在单一数据集、单变量小时预测、50栋建筑的规模受限，未涵盖不同气候、建筑类型、多变量预测或更大规模客户端验证，且私有专家增加了客户端存储成本。

---

## 40. What Is a Skill Worth? Structure-Aware Shapley Valuation of Agent Skills

**arXiv ID:** 2608.04562 | [PDF](https://arxiv.org/pdf/2608.04562v1)

**作者:** Tao Li `[一作]` (Nanjing University of Aeronautics and Astronautics), Linjun Shou `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SkillSV框架，对agent技能内部单元（规则、脚本、示例等）进行结构化价值评估；

**💡 创新点**

创新点在于将技能编译为结构化博弈，使用结构约束的Shapley值、配对删除/填充对比估计，以及链耦合任务窗口的预算化估计；

**🔧 技术方法**

技术包括技能编译器（提取单元、依赖、层级）、结构化Shapley求值、paired deletion/padding渲染、链耦合任务窗口估计、噪声门限截断；

**📊 数据集**

四个基准数据集：LiveMath、OfficeQA、SpreadsheetBench、ALFWorld，使用各自的任务分布与指标；

**📈 对比分析**

与基线Closure‑LOO、LLM判别器和随机排序比较，SkillSV在安全压缩（保留性能）和价值闭合方面显著优于基线；

**⚠️ 局限性**

局限包括：对大型技能仍需大量rollout，依赖于预先定义的依赖/层级规则，对不同语言或更复杂结构的技能迁移性待验证；

---

## 41. Step Recursion: A Three-Parameter Refinement of the Grzegorczyk Hierarchy

**arXiv ID:** 2608.04871 | [PDF](https://arxiv.org/pdf/2608.04871v1)

**作者:** Kirill Osipov `[一作]` `[通讯]` (Independent researcher), Kirill Osipov (Independent researcher)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在Grzegorczyk层级中引入按步长递归的新函数类，并给出了完整的层级分类。

**💡 创新点**

创新点在于将递归步长视为输入参数，构造了“水平收缩”和“垂直分离”的新判据，并用“块式递归”实现内部化。

**🔧 技术方法**

采用了数学归纳、构造性证明、步长递归、生成函数比较、区域和带宽分析等技术。

**📊 数据集**

无（纯理论研究）。

**📈 对比分析**

通过构造性证明比较不同层级的包含关系，得到完整的层级划分。

**⚠️ 局限性**

对基数为2的倍增行（21l）尚未完全分类，相关结果依赖于P=NP假设。

---

## 42. Supporting the understanding of ontologies for scientific knowledge graphs with the new version of LODE

**arXiv ID:** 2608.04689 | [PDF](https://arxiv.org/pdf/2608.04689v1)

**作者:** Valentina Pasqual `[一作]` (University of Bologna), Silvio Peroni `[通讯]` (University of Bologna)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了新版LODE 2.0，提供模块化、富文本注释和实体级RDF来源追踪的可读性更高的语义模型文档生成框架；

**💡 创新点**

创新点包括：将Reader‑Model‑Viewer三层架构实现分离；支持跨模块注释聚合和Markdown渲染；新增实体级RDF Provenance、独立实体页面和静态站点生成；

**🔧 技术方法**

使用Python实现，依托RDFLib、FastAPI、Jinja2、Bootstrap等技术栈；

**📊 数据集**

使用SKG‑IF Ontology（SKG‑O）作为测试数据集；

**📈 对比分析**

与WIDOCO对比，LODE在注释完整性、实体级展示、Markdown渲染和跨模块聚合方面表现更好，支持单体实体文档且可导出多种RDF序列；

**⚠️ 局限性**

目前仅支持OWL语义模型，缺乏推理支持，尚未扩展到SKOS/SHACL等其他语义资源，后续需完善多类型资源支持和推理集成。

---

## 43. Discrete homology computations by reduction to zero differentials

**arXiv ID:** 2608.04262 | [PDF](https://arxiv.org/pdf/2608.04262v1)

**作者:** Sterling Ebel `[一作]`, Nathan Kershaw `[通讯]`

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种新算法，用于计算图的离散同调及其持久化版本。

**💡 创新点**

创新点在于通过活跃枚举和将过滤器转化为零微分的链复形，避免大规模矩阵写入，显著提升高维同调的可计算性。

**🔧 技术方法**

采用离散立方体同调、超八面体群商、活跃枚举以及基于Quasi‑isomorphism的算法框架。

**📊 数据集**

使用Greene球、三维圆、随机欧氏点、随机距离矩阵、叠加圆等标准与合成数据集进行实验。

**📈 对比分析**

与Ripser、Vietoris–Rips同调及之前的算法对比，速度提升多达10^5倍，尤其在计算H_4(G^sph)和H_3(Σ_3G)时实现了先前不可算的结果；在噪声数据上生成条数下降至2%以内。

**⚠️ 局限性**

局限性包括仅支持1维持久化；对更高维的持久化仍需直接处理超立方体；对最坏噪声情形下的活跃枚举效率仍需改进；未覆盖多参数或Delaunay/α过滤器等更复杂的结构。

---

## 44. LiNC: Lightweight Noise Correction via Per-Sample Trust and Gaussian Mixture Modeling

**arXiv ID:** 2608.04147 | [PDF](https://arxiv.org/pdf/2608.04147v1)

**作者:** Abhishek Moturu `[一作]` (University of Toronto), Anna Goldenberg `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种轻量级方法LiNC，学习每个训练样本的信任参数并用其自动区分和纠正噪声标签，提升医学影像分类的鲁棒性。

**💡 创新点**

创新点在于仅用单个可学习的信任参数，无需额外模型、干净验证集或阈值；通过3成分高斯混合模型自动区分噪声、模糊与干净标签，并分阶段软/硬纠正。

**🔧 技术方法**

采用交叉熵的软目标混合、梯度推导得到信任更新、3成分高斯混合模型、ViT预训练网络以及Adam+LR调度等技术。

**📊 数据集**

使用MedMNISTv2十个2D医学影像数据集，人工注入10%–50%对称噪声进行实验。

**📈 对比分析**

与多种噪声检测与学习方法（AUM、DataMaps、VoG等）以及标准训练对比，LiNC在AUC上最高达0.9837，最后期准确率平均提升2.19%–21.41%，并显著减少训练过程中的性能退化。

**⚠️ 局限性**

局限性包括：可能将临床模糊误判为噪声；依赖早期学习假设，稀疏类别或系统性误差时表现可能下降；三成分GMM在无噪声或低噪声时仍产生三组，导致误判；需在真实噪声数据上进一步验证。

---

## 45. GEB-Bench: Abstract Structures Told in Many Voices

**arXiv ID:** 2608.04111 | [PDF](https://arxiv.org/pdf/2608.04111v1)

**作者:** Tong Zhang `[一作]` (Fudan University), Tao Xie `[通讯]` (Peking University)

**通讯引用:** 18432 | [OpenAlex ID](https://openalex.org/A5048118068)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了 GEB-Bench，一个基于抽象结构动机的跨模态基准，用来评估模型在自然场景、故事、定理和骨架四种声部中的结构识别与跨声部映射能力。

**💡 创新点**

创新点在于将抽象结构动机作为基准单元，构造四种可程序验证的实现声部，并引入范畴理论框架来分离对象识别与结构迁移，揭示模型在跨声部映射上的系统性缺陷。

**🔧 技术方法**

使用深度学习多模态模型、程序可验证的文本与图形生成、基于范畴论的任务定义，以及多选题精确匹配评估技术。

**📊 数据集**

使用包含 25 种抽象结构动机的手工构造库，每种动机在四种声部中生成约 1,156 个样本，并在扩展实验中加入 25 个开源模型的额外样本。

**📈 对比分析**

通过多选题的精确匹配评估，计算机化 chance‑corrected κ 指标；结果显示前沿模型在识别任务上约 82% 但在跨声部匹配上显著下降，错误主要聚集在形式相邻的动机之间，表明跨声部迁移仍是瓶颈。

**⚠️ 局限性**

局限性包括：评估仅覆盖预设的抽象结构动机，缺乏跨语言、跨文化或更复杂场景的验证；跨声部映射能力仍受限于前沿模型；实验聚焦英语文本，未能充分检验多语言泛化。

---

## 46. CARGO-VL: Counterfactual Arbitration with Risk-Constrained Group Optimization for Vision-Language Models

**arXiv ID:** 2608.04509 | [PDF](https://arxiv.org/pdf/2608.04509v1)

**作者:** De Jiang `[一作]` (Tsinghua University), Shaohua Ma `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该工作提出了一种名为CARGO‑VL的训练框架，能够在视觉‑语言模型中同时优化四种对齐与冲突状态（A/V/T/N）下的答案和来源选择，并实现安全的拒绝决策。

**💡 创新点**

其创新点在于把对齐、图像正确、文本正确和两者错误四种状态视为一个 bundle，并设计了包含答案一致性、来源等价性及答复‑拒绝切换的转移奖励，以及基于对偶的风险约束，使得模型在对抗性证据下保持一致的推理行为。

**🔧 技术方法**

使用了组相对优化（GRPO）与软最小化、转移奖励以及自监督文本生成，配合冻结的语义映射器将生成文本映射为决策与来源，并通过多种技术实现安全-效用的平衡。

**📊 数据集**

数据集方面构建了XMC（eXtended Modal Conflict）四类冲突样本，扩展自TextVQA和ScienceQA，并在CMC‑Bench holdout与Modality‑Bias上进行评估。

**📈 对比分析**

与基线SFT、GRPO、CFPO以及大型模型Gemma‑26B、GPT‑4o等对比，CARGO‑VL在Acc(I*)、Acc(N)、ConfabR、CDR、ΔAcc以及接近零的偏差B上均优于对手，并保持高水平的Acc(T*)。

**⚠️ 局限性**

限制在于需要预先构造完整的A/V/T/N bundle，训练和评估过程对数据质量和转移奖励权重敏感，且对不同任务或更大规模模型的迁移性仍需进一步验证。

---

## 47. LoRetta: A Foundation Model and Extensive Dataset for Global-Scale Remote Sensing Dense Image Matching

**arXiv ID:** 2608.04106 | [PDF](https://arxiv.org/pdf/2608.04106v1)

**作者:** Siwei Yu `[一作]` (Beihang University), Zhengxia Zou `[通讯]` (Beihang University)

**通讯引用:** 9792 | [OpenAlex ID](https://openalex.org/A5088611151)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了LoRetta模型，专门针对全球尺度的遥感图像密集匹配问题，解决了大角度旋转、尺度变化、部分重叠以及多时相变化导致的匹配困难。

**💡 创新点**

创新点在于将密集匹配拆分为“定位与配准”两步：先通过匹配度加权的仿射定位获得全局宏观对齐，再在该对齐框架内进行局部残差细化，并同时学习匹配度图以筛选可靠区域。

**🔧 技术方法**

技术实现基于冻结的DINOv3视觉Transformer提取粗粒度特征，利用多视角Transformer生成匹配嵌入；随后使用匹配度加权最小二乘拟合仿射变换；再通过VGG19特征金字塔和三阶段残差Refiner进行多尺度残差配准，并采用伪匹配度、仿射监督以及鲁棒误差等多任务损失进行训练。

**📊 数据集**

使用了全新的LEVIR‑GM数据集，涵盖6大洲、5年（2018‑2022）多时相光学图像，分辨率从0.5 m到1024 m不等，包含103k对真实对齐样本与827k对经过仿射＋局部非刚性扰动生成的增强样本，并提供匹配度标签。

**📈 对比分析**

在LEVIR‑GM上与SIFT、SuperPoint+LightGlue、LoFTR、DKM、RoMa及RoMa v2等基线对比，LoRetta在AUC上达到83.3%（比RoMa v2提升1.6点），PCK@2px提升6.5点，且推理时间仅为64.8 ms，几乎是RoMa v2的二分之一。

**⚠️ 局限性**

局限性主要体现在：仅针对光学‑光学对齐，未涉及SAR或多光谱跨模态；对极端云遮挡、地形高度变化的鲁棒性仍有限；以及模型训练依赖大规模标注数据，迁移到小样本或跨传感器任务时效果不明。

---

## 48. Talk2Sensors: 3D Visual Grounding in Autonomous Driving via Sensor-Adaptive Physical Cue Matching

**arXiv ID:** 2608.04568 | [PDF](https://arxiv.org/pdf/2608.04568v1)

**作者:** Runwei Guan `[一作]` (Hong Kong University of Science and Technology), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了用于多传感器3D视觉定位的TSFormer框架，并创建了Talk2Sensors数据集

**💡 创新点**

通过语言引导的属性采样与稀疏保留的模态仲裁实现了对不同物理属性的动态路由，避免了传统融合中高密度模态压倒稀疏关键信号的缺陷

**🔧 技术方法**

使用Transformer+多模态解码器、查询驱动的可变形采样、文本引导的门控融合与跨模态注意机制

**📊 数据集**

Talk2Sensors（摄像头+LiDAR+4D雷达）以及Mono3DRefer（单摄像头）

**📈 对比分析**

在Talk2Sensors上与多模态检测器和单模态定位模型对比，TSFormer在所有传感器配置下均取得最高mAP（51.00/66.50），并在Mono3DRefer上无任务调优即可获得最优IoU@0.5分数；对比实验显示其在模态缺失时更稳健，效率与精度均优于基线

**⚠️ 局限性**

对雷达的稀疏性仍存在挑战，缺乏对夜间或恶劣天气下雷达与LiDAR互补性的进一步研究，以及对时序连续性信息的利用不足

---

## 49. Kitchen Robotic Manipulation utilizing Foundation Models

**arXiv ID:** 2608.04042 | [PDF](https://arxiv.org/pdf/2608.04042v1)

**作者:** Myung-Hwan Jeon `[一作]` (Kumoh National Institute of Technology), Joohyung Kim `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 4095 | [OpenAlex ID](https://openalex.org/A5100759184)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计并实现了一个模块化的感知-执行管道，用于厨房环境中盘子、杯子等餐具的识别、6D位姿估计与抓取，无需针对特定场景进行微调。

**💡 创新点**

创新点包括：①基于多种基础模型（LLMDet、SAMv2、DINOv2、GeoTransformer等）可互换的设计；②将2D视觉特征与3D几何特征通过简单拼接融合；③系统化评估24种模型组合，确定最优配置；④在真实厨房机器人上验证可直接迁移，未做环境特定训练。

**🔧 技术方法**

使用的技术包括：开源长文本视觉模型LLMDet用于生成检测提示；SAMv2实现多视角实例分割；DINOv2提取细粒度视觉特征；GeoTransformer提供全局几何上下文；ICP细化位姿；基于CAD模型的预定义抓取候选与IK/运动规划。

**📊 数据集**

数据集：自制的20场景厨房数据集（包含不同摆放、遮挡、杂物条件），每个场景均标注实例掩码和6D位姿；另外使用两个不同厨房环境的真实机器人实验数据。

**📈 对比分析**

方法对比：与FoundationPose、不同视觉/几何模型组合进行比较。最佳配置（LLMDet+SAMv2+DINOv2+GeoTransformer）在ADI指标上达到89.12%（未ICP）/88.92%（含ICP），超过FoundationPose。实际机器人任务成功率达87.5%（共296次尝试）。

**⚠️ 局限性**

局限性：①并行抓取手的刚性导致抓取滑移；②对高度堆叠、强遮挡的物体仍有失败；③运动规划未优化路径平滑性，容易产生抖动；④仅适用于已知CAD库存，无法扩展到未知物体；⑤目前的特征融合仅为拼接，缺乏可学习的跨模态整合。

---

## 50. What We Observe as LLM Behavior Can Be a Side-effect of Inference Backend

**arXiv ID:** 2608.04714 | [PDF](https://arxiv.org/pdf/2608.04714v1)

**作者:** Shahed Masoudian `[一作]` (Johannes Kepler University Linz), Markus Schedl `[通讯]` (Johannes Kepler University Linz)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了五种LLM推理框架对基准得分的影响，控制模型、数据和评测协议不变，采用了三大指令调优模型、六大基准和四种生成模式。

**💡 创新点**

创新点在于将推理后端视为可测量的超参数，量化其对模型性能的结构性与采样性贡献，并揭示后端对事实性与偏见性任务的差异性。

**🔧 技术方法**

技术上使用了 hf_raw、hf_pipeline、langchain_hf、vllm 以及 ollama 等推理后端，在四种生成模式（deterministic、fix、token=256、default）下进行对比实验。

**📊 数据集**

数据集包括 MMLU、TriviaQA、TruthfulQA、TruthfulQA-Gen、BBQ、StereoSet 等，覆盖多选题、开放式问答与社会偏见评测。

**📈 对比分析**

比较方法通过对每个后端与基准框架的分数差异、误差率、方差分解等统计量评估，结果显示后端结构性差异可导致约 0.00074 的方差，采样与默认参数进一步提升至 0.00190。

**⚠️ 局限性**

局限性包括仅测试 1B 规模模型、未考虑量化与更大模型、生成模式覆盖有限、仅在单一 RTX 3090 GPU 上运行，且评测集中于英文多选/短文本。

---

## 51. Concentration from Product Moments via an Additional Element of Randomness

**arXiv ID:** 2608.04125 | [PDF](https://arxiv.org/pdf/2608.04125v1)

**作者:** Michael Saks `[一作]` (Rutgers University), Renata Valieva `[通讯]` (University of Maryland)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

提出了一种基于初等对称多项式的框架来证明浓缩界限，增加了随机性元素以简化问题。

**💡 创新点**

通过引入随机性，改进了在有限独立性下的浓缩界限，尤其是在读-Δ家族和随机二进制线性哈希的情况下。

**🔧 技术方法**

使用了初等对称多项式和产品矩的框架，结合了随机选择的索引集。

**📊 数据集**

论文中没有具体提到使用的数据集。

**📈 对比分析**

与传统的Chernoff-Hoeffding界限进行比较，提出的界限在多种情况下表现出更好的依赖性，尤其是在有限独立性和随机输入的情况下。

**⚠️ 局限性**

在处理依赖性较强的随机变量时，可能会面临过于保守的界限，尤其是在极端情况下。

---

## 52. Behavioral Information Leakage in Darknet Traffic: A Multi-Channel Analysis Across Anonymity Networks

**arXiv ID:** 2608.04143 | [PDF](https://arxiv.org/pdf/2608.04143v1)

**作者:** Javeriah Saleem `[一作]` (Charles Sturt University), Md Zahidul Islam `[通讯]` (Charles Sturt University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于行为信息泄露分解的暗网流量分析框架，将流量描述符划分为控制、结构和节奏三类，并通过信息论与随机森林验证其可识别性；

**💡 创新点**

创新点在于通过机制层解释暗网服务可识别性，量化各行为通道的泄露，并提出服务变异指数(SVI)和泄露变异指数(LVI)评估跨网络差异；

**🔧 技术方法**

采用互信息估计、随机森林预测、重复分层交叉验证、低方差/相关性过滤、鲁棒缩放，并对结构与节奏特征进行交互与综合分析；

**📊 数据集**

使用公开的 Darknet Dataset 2020，其中包含 Tor、I2P、FreeNet、ZeroNet 的流量及其服务标签；

**📈 对比分析**

在全局、网络内和跨语义对比实验中，结构-节奏组合在 Tor 上实现 Macro‑F1 0.7165，整体提升约15%；跨网络迁移性能低，表明泄露高度依赖网络；结构与节奏交叉提升正向且泄露量显著增加；

**⚠️ 局限性**

仅依赖单一流量数据集、仅用随机森林做验证、累计互信息可能存在重叠、跨网络/服务对的样本不均匀，缺乏对不同捕获环境和对抗手段的评估。

---

## 53. Scale-CDA: A Scalable Prototype to Democratize AI-Assisted Cooperative Driving Automation (CDA) for Production Cars

**arXiv ID:** 2608.04235 | [PDF](https://arxiv.org/pdf/2608.04235v1)

**作者:** Hao Zhou `[一作]` (University of South Florida), Haibin Wen `[通讯]` (Sunnypilot LLC)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在生产车辆上实现了一套低成本、可插拔的协同驾驶自动化（CDA）系统，该系统通过 OpenDBC 接口、Openpilot Level‑2 ADAS、WiFi 6/LTE V2X 通信、边缘多模 LLM 推理以及 Meta‑Action API 将生成式 AI 直接嵌入车辆控制链路。

**💡 创新点**

创新点包括：① 通过标准化的 MCP 以及 Meta‑Action 桥接，使多模 LLM 能以结构化 JSON 输出可执行的驾驶指令；② 将 WiFi 6 与 MQTT 组合形成低延迟、百Mbps 的 V2X 通信栈；③ 在单车成本低于 1,000 美元的硬件平台上完成端到端的决策延迟低于 60 ms 的完整 CDA 堆栈。

**🔧 技术方法**

使用的技术包括：OpenDBC 与 Openpilot 作为底层控制框架；WiFi 6/LTE + MQTT 作为 V2X 通信协议；Model‑Context‑Protocol（MCP）桥接实现多模数据融合；边缘多模大语言模型（Gemma3、LLaVA、Qwen3 等）在 NucBox K6 上本地推理；Meta‑Action API 负责将 LLM 生成的指令映射到 Openpilot 的 planner/hooks。

**📊 数据集**

未使用公开数据集；实验数据来源于：① 车辆 CAN 与摄像头同步采集的实时感知流；② 7.5 km 循环测试路段中 WiFi 6/V2X 传输的 RTT 与吞吐量数据；③ 现场道路测试记录的决策延迟与安全事件日志。

**📈 对比分析**

性能对比：WiFi 6+MQTT 的单播往返延迟为 5.25 ms，吞吐量约 100 Mbps；端到端决策延迟低于 60 ms；在 NucBox K6 上 LLaVA 7B 模型的推理速率约 12 TPS，Gemma3‑4B 约 13.6 TPS；与传统 PC5/DSRC 对比，WiFi 6 具备更低延迟、更高带宽与更广覆盖。整体系统满足非安全关键 CDA 的实时性要求。

**⚠️ 局限性**

局限性包括：① 目前仅支持 Level‑2 ADAS，未实现完整 SAE L3/L4 功能；② 边缘 LLM 推理速度仍受限，难以满足极低延迟的复杂场景；③ 对安全关键 V2X 通信仍依赖 LTE/WiFi，缺乏 DSRC/PC5 的可靠性与抗干扰特性；④ 生成式 AI 的安全性与可解释性尚未通过完整安全案例验证。

---

## 54. Recurrent Residual Quantization: A Progressive Multi-Precision Representation for LLMs

**arXiv ID:** 2608.04048 | [PDF](https://arxiv.org/pdf/2608.04048v1)

**作者:** Yu Luo `[一作]` (Intel), Haihao Shen `[通讯]` (Intel)

**通讯引用:** 207 | [OpenAlex ID](https://openalex.org/A5056650480)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为Recurrent Residual Quantization（RRQ）的后训练量化框架，利用低比特基础模型与一系列低比特残差纠正，实现单检查点多精度推理。

**💡 创新点**

核心创新是把量化误差拆分成可加的残差层，而非传统嵌套整数位切片，使得可以在不重新训练或校准的情况下，从同一检查点得到2、4、6、8比特不同精度模型。

**🔧 技术方法**

采用基于Round‑to‑Nearest（RTN）的2比特量化作为基础与残差阶段，支持可组合的整数低位量化；同时也评估了更强的SignRoundV2基准。

**📊 数据集**

在六个近年LLM（Llama‑3.1‑8B、Llama‑3.1‑8B‑Instruct、Qwen3‑8B‑Base、Qwen3‑8B、Qwen3‑14B、Phi‑3‑Medium）上进行实验，使用标准的零样本任务集合（ARC‑Challenge、ARC‑Easy、HellaSwag、PIQA、WinoGrande）及WikiText‑2 perplexity。

**📈 对比分析**

与GPTQ、MatGPTQ等单精度与多精度PTQ方法比较，RRQ在6/8比特下保持与16‑bit基线相近的Task Avg和PPL；在4比特下表现受模型与基础量化质量影响，部分模型超越MatGPTQ，整体可与之竞争；构建时间上RRQ（RTN）为3.3×快。

**⚠️ 局限性**

限制主要在于对基础量化质量的高度依赖，低比特基础模型误差会限制后续残差层的纠正效果；此外，目前仍未在专用硬件上实现高效推理核，实际延迟及能耗评估待进一步研究。

---

## 55. A Model Merging Approach for Continual MLLM Unlearning

**arXiv ID:** 2608.04548 | [PDF](https://arxiv.org/pdf/2608.04548v1)

**作者:** Yuhang Wang `[一作]` (Xidian University), Haichang Gao `[通讯]` (Xidian University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种称为 MCU 的持续多模态模型遗忘框架，利用动态合并一次性遗忘适配器来实现连续的模型删除请求。

**💡 创新点**

创新点包括：①通过留一分析揭示遗忘适配器之间的交叉任务依赖与转移；②设计共享核心空间表示，保留主导方向并抑制坐标过度集中；③采用依赖感知的方向再配置（正交化与 Gram 矩阵约束）来平衡干扰与协同；④整合方向选择、通道容量控制与再配置的完整流水线。

**🔧 技术方法**

技术手段包括：LoRA 低秩适配器、共享核心空间投影、奇异值分解 (SVD)、Gram 矩阵约束、半正定优化、正交 Procrustes、秩约束与非负交叉项约束。

**📊 数据集**

使用 ICU‑Bench 与 MLLMU‑Bench 两大持续遗忘基准，涵盖多模态问答与常识推理任务。

**📈 对比分析**

与 GA、GA‑Diff、KL‑Min、NPO、MANU、MMUnlearner 等顺序方法以及 Task Arithmetic、TIES、DARE、TSV、Core‑Space 等合并方法对比，MCU 在 Forget（遗忘精度）上显著下降、Retain（保留性能）明显提升，且在遗忘反弹、保留漂移、生成质量与稳定率上均优于对照组，证明了其在长期连续请求下的稳健性。

**⚠️ 局限性**

局限性：①仅针对 LoRA 适配器设计，扩展到其他适配或全参数更新需要进一步研究；②核心空间投影与优化过程对内存与计算有一定开销；③在极大规模请求序列或极高维度模型下的可扩展性尚未完全验证。

---

## 56. Sample Complexity of Multicalibration for Multilevel Properties

**arXiv ID:** 2608.04288 | [PDF](https://arxiv.org/pdf/2608.04288v1)

**作者:** Jiuyao Lu `[一作]` (University of Pennsylvania), Shiva Prasad Kasiviswanathan `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了在多级可识别分布属性（如均值、方差、偏度、分位数等）下的多重校准（multicalibration）的样本复杂度，给出了匹配的上下界，并在三种典型场景（均值+绝对偏差、均值+方差+偏度、分位数+CVaR）中实例化理论。

**💡 创新点**

创新点在于：①提出了“顺序可识别”框架，允许属性依次条件可识别；②证明了对任意固定层数 k，样本复杂度为 Θ(ε^{-(k+2)})（至对数因子）；③将信息论下界与在线专家学习（exponential weights）+Sion 极小化双重技术结合，实现了近最优上界；④利用 Walsh 基函数构造多组二元集合，实现了阈值符号的多组线性逼近，保持聚类误差可控。

**🔧 技术方法**

主要技术包括：信息论下界（Fano、不等式、KL 散度估计）; 在线学习与专家建议框架（Σ 组专家和 sign 矩阵的最大化表示）；指数权重更新（exponential weights）与 Regret 分析；Sion 极小化双重定理与最小‑最大值近似；在线‑批量（online‑to‑batch）归约；凸优化与线性分离器（linear optimization oracle）在分布族上实现最优概率分布的求解；以及 Walsh 基函数和二元组逼近技术。

**📊 数据集**

论文仅在理论层面工作，使用构造的局部分布族（如三点支持分布、四点支持分布、指数族概率密度等）进行下界证明和上界实例化；没有使用公开真实数据集。

**📈 对比分析**

与方法的比较完全在理论上进行：给出的上界与下界在 ε → 0 时相匹配，样本复杂度达到最优阶数；实验性评估仅通过构造的测试分布验证理论一致性，未给出具体的数值性能指标。

**⚠️ 局限性**

局限性包括：①需要满足严格的正则性假设（残差 Lipschitz、KL 有界、属性范围紧致等）；②上界实现依赖于对每个 group‑sign 组合的专家权重分布，计算量随 2^k|𝒫_Q| 增长；③在实际大规模组集合或高维属性空间中，Walsh 基逼近与优化分离器的实现复杂度可能成为瓶颈；④论文讨论的样本复杂度仅对有限组大小 |𝒢| = O(ε^{-κ}) 的情况，超大组集合的情况仍待进一步研究。

---

## 57. HELENA:Hierarchical Sparse Coordination over a Union of Complementary Topologies for MAS

**arXiv ID:** 2608.04634 | [PDF](https://arxiv.org/pdf/2608.04634v1)

**作者:** Zhifang Mao `[一作]` (XiaoLab), Xiuquan Hou `[通讯]` (Xi'an Jiaotong University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个联合拓扑图（Union MAS Graph），并在此图上执行层级稀疏协同（节点级记忆合成 + 边级稀疏激活）与局部自我校正机制，从而提升LLM多智能体系统的推理质量。

**💡 创新点**

①通过MCTS与DPP自动选择互补的拓扑并融合成联合图；②在推理时采用层级稀疏协同抑制噪声传播；③局部自我校正对高风险决策单元进行对抗式验证。

**🔧 技术方法**

使用了Monte Carlo Tree Search、Determinantal Point Process、Sparsemax、记忆合成器、边评分器、全局更新器、局部自我校正、LLM（gpt‑4o‑mini）推理和Qwen嵌入。

**📊 数据集**

MMLU、MMLU‑Pro、GSM8K、MATH、MATH‑Lv5、SVAMP、HumanEval、MBPP。

**📈 对比分析**

与单智能体、预定义协议、自动化构建、图结构MAS等基线在相同gpt‑4o‑mini环境下重新实现；HELENA在所有8个基准上均取得最高分，平均提升3.47%，在最难的MMLU‑Pro提升10.34。

**⚠️ 局限性**

需要额外的搜索与多拓扑融合，计算开销较大；对K、α等超参数敏感；仅在LLM推理环境下验证，跨模型或非LLM场景的通用性尚未探究。

---

## 58. NeuroPB: Scaling Neural Decoding with Pretrained Behavioral Representations

**arXiv ID:** 2608.04389 | [PDF](https://arxiv.org/pdf/2608.04389v1)

**作者:** Luyao Jin `[一作]`, Wei-Hsin Liao `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a4b10f5d-130b-4e77-9367-6469ec621899` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了NeuroPB框架，利用机器人大规模行为预训练数据对神经与运动轨迹进行对齐与解码，并评估其在不同会话、个体与任务间的迁移性能。

**💡 创新点**

创新点在于①将规模与多样性均大幅提升的机器人数据（LIBERO-100）用于行为预训练，②采用Transformer‑based神经与运动编码器配合对比学习对齐，③通过严格的校准与评估流程验证大规模预训练对非机器人脑-机接口的显著促进作用。

**🔧 技术方法**

主要技术包括Transformer编码器（8层、384隐藏维度）用于运动轨迹、6层自注意力神经编码器、交叉注意力对齐、两侧投影至3,200维共享空间、对比损失与解码损失结合、SparseLamb优化器与cosine学习率调度。

**📊 数据集**

使用的主要数据集为LIBERO‑Spatial（10任务×50演示=500轨迹）与LIBERO‑100（100任务×50演示=5,000轨迹）进行机器人预训练，另外对照使用与匹配规模的Macaco轨迹集进行神经-运动对齐。

**📈 对比分析**

对比方法：与“Scratch”（无预训练）及“Robot‑Matched‑PT”（规模匹配但任务少）进行R²评估，跨会话、跨个体、跨任务转移实验。结果显示Robot‑Large‑PT在所有设置下均显著优于Scratch（p<0.01），并优于Robot‑Matched‑PT与Macaque‑PT，说明更大规模与多样性的机器人预训练能提升泛化性能。

**⚠️ 局限性**

局限性包括：仅在两种特定运动任务（CO、RT）中验证，未覆盖更复杂或多模态行为；对比学习温度与损失权重的超参数选择未做深入探索；校准阶段仍需一定量的目标域标注数据，限制了完全无监督迁移；模型对不同物种或传感器配置的鲁棒性尚待验证。

---

## 59. Chain-of-Thought Monitoring Can Be Unreliable in Implicit-Influence Settings

**arXiv ID:** 2608.04735 | [PDF](https://arxiv.org/pdf/2608.04735v1)

**作者:** Agatha Duzan `[一作]` (EPFL), Asa Cooper Stickland `[通讯]` (UK AI Security Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个基准，对比链式思考(CoT)监控在显式影响与隐式影响场景下的可监控性，涵盖四种任务格式和七大前沿推理模型。

**💡 创新点**

创新点在于首次将显式与隐式影响区分并直接匹配比较，探究真实部署提示对监控的削弱效果，以及通过将影响转化为多步推理来提升可监控性。

**🔧 技术方法**

采用了链式思考推理、GPT‑5驱动的监控器、延伸思考（extended‑thinking）模型、指导性系统提示与对抗性系统提示，以及多步链式推理等技术。

**📊 数据集**

数据集主要为自建基准，包括二元偏好、数值评分、带位置信息的多项选择题（来自Humanities Last Exam）和开放式编码任务，分别使用十个评分域和八个编码题目。

**📈 对比分析**

通过测量在四种设置下显式和隐式干预导致的行为转移，并计算监控器的检测率进行比较；结果显示显式场景下检测率在60%–94%之间，而隐式场景下下降41%–46个百分点，指导性提示可将隐式检测率降至仅5%。

**⚠️ 局限性**

主要局限包括监控器本身是LLM评判器，可能无法捕捉内部信号；Claude 4.5只能提供摘要思考；评估仅涵盖前沿模型与对齐权重，未覆盖真实不对齐模型；缺乏对最坏情况边界的系统性探索。

---

## 60. Deltoris: Enabling Real-time VLA Inference in Embodied AI via Bit-level Sparsity and Speculative Inference

**arXiv ID:** 2608.04428 | [PDF](https://arxiv.org/pdf/2608.04428v1)

**作者:** Zheng Liu `[一作]` (Shanghai Jiao Tong University), Yu Feng `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Deltoris 加速器，实现了对基于扩散的视觉-语言-动作（VLA）模型的实时推理。

**💡 创新点**

主要创新点包括：① 针对连续控制步的时间相似性提出的时间感知位级稀疏化算法，显著减少位级计算；② 通过“先草稿后验证”机制的投机推理，批量化大模型验证并抵消额外的内存访问；③ 为上述算法量身定制的 1D 赛马式位串行处理单元，消除传统位串行加速器的工作负载不均衡；④ 结合压缩编码/解码模块进一步降低离线内存流量。

**🔧 技术方法**

使用的技术包括：算法-硬件协同设计、位级稀疏化、时间差分计算、投机推理、1D 赛马式位串行阵列、压缩编码/解码、混合量化、微处理器控制逻辑。

**📊 数据集**

实验采用 Meta‑World、PushT 与 LIBERO 三个机器人控制数据集，评估 PAD、DP、UVA 等三种扩散 VLA 模型。

**📈 对比分析**

与移动 GPU（Orin）、Pragmatic、BBS、Cambricon‑D、Exion、Ditto 等基线比较，Deltoris 在三种模型上分别实现了最高 34.2× 的速度提升（相较 GPU）和 6.1× 的加速（相较前置加速器），能耗则可达 850× 的节能（相较 GPU）。

**⚠️ 局限性**

局限性：① 对快速运动场景的时间相似性降低，投机推理误差增大，精度略微下降；② 设计侧重扩散 VLA，针对非扩散或不具备时间相似性的任务可能无效；③ 额外的压缩编码和投机窗口管理增加了硬件复杂度和实现难度。

---

## 61. Monte Carlo Tree Search for Table-to-Multimodal Report Generation

**arXiv ID:** 2608.04071 | [PDF](https://arxiv.org/pdf/2608.04071v1)

**作者:** Teng Lin `[一作]` (Hong Kong University of Science and Technology), Nan Tang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 6894 | [OpenAlex ID](https://openalex.org/A5062243169)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出基于蒙特卡洛树搜索的单LLM统一框架，逐步构建包含文本与图表的多模态报告。

**💡 创新点**

将报告生成视为结构化搜索，利用MCTS探索报表结构，统一LLM完成行动与评估，采用自监督奖励实现事实准确、图表质量与叙事连贯的协同优化。

**🔧 技术方法**

使用MCTS、单一大型语言模型（LLM）、SQL验证的自监督奖励、代码生成的可视化（Vega-Lite/Matplotlib）、提示工程以及图表与文本对齐评估。

**📊 数据集**

构建MMDR‑Bench，185张真实表（涵盖金融、制造、医疗、教育、零售、IT运维六个行业），386个查询（单表279、跨表107），每个查询对应约4.75条关键洞察，总计1,834条关键点。

**📈 对比分析**

与12类基线（视图‑语言模型、代码增强多模态系统、深度研究代理）在MMDR‑Bench上对比，MCTS提升整体分数至77.9，远超最佳基线62.7，接近人工基线91；数值准确率、结构完整性、图表‑文本一致性均显著提升。

**⚠️ 局限性**

创新度仍有限（洞察新颖性不足），多表推理易混淆，LLM的分析深度受限，搜索成本较高，部分图表细节仍存在误差。

---

## 62. Calibrating Artificial Guilt: Neurally Grounded Reward Shaping for Prosocial Multi-Agent Reinforcement Learning

**arXiv ID:** 2608.04663 | [PDF](https://arxiv.org/pdf/2608.04663v1)

**作者:** Aaditya Mehta `[一作]` (Mahatma Gandhi International School), Arya Shah `[通讯]` (Indian Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

从公开的神经影像和行为数据中提取个人对伴侣负面结果的罪恶惩罚系数，将其嵌入两智能体的Social Lottery环境，并用PPO训练学习者；随后比较不同惩罚系数对社会福利、安全率、奖励不平等和与人类行为分布（KL）的影响。

**💡 创新点**

创新点在于：① 采用单一量化的“罪恶”对比（来自左前岛）作为奖励塑形系数，避免手工调参；② 将该系数直接迁移到多智能体RL，验证其对齐效果；③ 提供可复现的PettingZoo Social Lottery环境；④ 用KL距离与人类实验数据直接对齐，提出更具可解释性的对齐评价。

**🔧 技术方法**

技术主要包括：线性回归提取罪恶权重、Potential-based奖励塑形、Proximal Policy Optimization (PPO) 强化学习、PettingZoo并行环境接口、KL散度评估。

**📊 数据集**

数据集为OpenNeuro ds005588（SoDec responsibility 责任任务）——40名受试者的行为和fMRI记录，仅使用行为计数估计罪恶系数。

**📈 对比分析**

比较方法：在相同训练超参数下，设置四个惩罚系数（Zero、Uniform、Oracle、NeuroGuilt）分别训练20,000个episode，随后用1,000个greedy episode评估。指标包括社会福利、保险率、安全率、奖励不平等、罪恶惩罚量以及与人类安全率的KL距离。结果显示NeuroGuilt安全率0.459与人类0.484相近，KL仅为0.0012，远低于其他三个条件，证明量化惩罚系数显著提升与人类行为的一致性。

**⚠️ 局限性**

局限性：① 罪恶系数估计未达到统计显著，且未利用个体前岛beta值；② 实验环境为短期二选一抽象，缺乏长期信用、通信或更复杂的混合动机格局，限制了结论对更大规模或长期交互的推广；③ 仅针对该责任任务的特定奖惩结构，其他道德信号或场景需要进一步验证。

---

## 63. Bottleneck Paths Reduce to Deterministic Graphical Games: A Correction to a Claimed Linear-Time Algorithm

**arXiv ID:** 2608.04279 | [PDF](https://arxiv.org/pdf/2608.04279v1)

**作者:** Egor Gorbachev `[一作]` `[通讯]` (ETH Zürich), Egor Gorbachev (ETH Zürich)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

给出Chechik等人声称的线性时间DGG算法的反例，并提供从有向s–t瓶颈路径问题到DGG的线性时间归约。

**💡 创新点**

明确证明DGG至少与有向s–t瓶颈路径问题一样难，纠正了此前错误的算法，并把两个开放问题联系起来。

**🔧 技术方法**

采用图形化游戏构造、边拆分与终点置换的归约技术，以及对搜索过程的精细分析。

**📊 数据集**

本文无实验，使用的是理论构造的合成实例。

**📈 对比分析**

通过理论复杂度分析比较，证明若DGG可在线性时间内解，则有向s–t瓶颈路径亦可，暗示目前的(m log* n)算法是最优近似。

**⚠️ 局限性**

仍未给出实际线性时间比较模型算法，且只在理论层面提出归约，实验验证和实现细节缺失。

---

## 64. Benchmarking Deep Learning Models for Dense Event Classification of Offshore Wind Infrastructure in Sentinel-1 Time Series

**arXiv ID:** 2608.04706 | [PDF](https://arxiv.org/pdf/2608.04706v1)

**作者:** Thorsten Hoeser `[一作]` (German Aerospace Center), Claudia Kuenzer `[通讯]` (German Aerospace Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

基于Sentinel‑1 SAR时间序列，系统比较十种深度学习变体，替代传统规则驱动的事件分类器，实现对离岸风电设施生命周期事件的密集标注；

**💡 创新点**

首次提出在离岸风电SAR时间序列上使用序列上下文感知的深度模型（尤其是双向BiLSTM），并结合自监督预训练与规则基标签的转移最小化集成，显著提升了事件标注的连贯性与精度；

**🔧 技术方法**

采用LSTM、Transformer与全连接网络，并在三种时间上下文（单调、单向、双向）下进行监督与自监督预训练；

**📊 数据集**

使用由ESA Sentinel‑1 GRD IW VH数据构建的14.8 M条事件、15 606条时间序列的数据集，并在此基础上手工标注额外500条序列（661 k事件）供训练；

**📈 对比分析**

通过AUC_EditSim、完美匹配率、宏/微F1等指标评估，双向BiLSTM在监督训练下取得最高AUC 0.8509、完美匹配率 0.5063；与规则基基线相比提升约 7%；集成后进一步降低了标签跳变并改善了低频类（如平台）性能；

**⚠️ 局限性**

主要局限在于窗口长度64事件导致模型对长程时间模式把握不足，出现短期错误的基础阶段预测；自监督预训练对双向模型无显著提升，提示需更多有标签数据；因果模型仍低于基线，适用于实时监测时需进一步改进。

---

## 65. From Research Questions to Columns: Operationalization-Aware Data Discovery

**arXiv ID:** 2608.04536 | [PDF](https://arxiv.org/pdf/2608.04536v1)

**作者:** Houming Chen `[一作]` (University of Michigan), H. V. Jagadish `[通讯]` (University of Michigan)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文定义了操作化感知数据发现（OADD）任务，并基于已发表研究构建了OADD-Bench基准，包含160个问题、4,682列标签，覆盖122,324列HRS数据库。

**💡 创新点**

创新点在于：①首次将概念到可用列的桥接视为核心问题；②利用LLM从论文中自动提取问题与列映射，避免人工标注；③提供以出版物为基础的基准与评估指标，为OADD研究提供可复制的实验平台。

**🔧 技术方法**

技术手段包括：GPT‑5.5语言模型进行问答、文档理解与多步推理；检索式方法结合语义检索与代码书匹配；基于规则的最终归属与可解释证据包装；对HRS元数据进行约束搜索和候选缩减。

**📊 数据集**

数据集：Health and Retirement Study（HRS）122,324列科学数据库；111篇相关研究论文及其补充材料；对应的HRS元数据和版本信息。

**📈 对比分析**

与传统直接检索和五种schema‑linking改编方法进行比较，并构建OADD代理模型。评估采用Recall@R、@2R、@5R，结果表明即便最强模型在5R下的覆盖率仍不足一半，显示OADD任务在当前技术水平上存在显著差距。

**⚠️ 局限性**

局限性包括：①依赖LLM推理，可能产生误判；②基准仅覆盖HRS，推广性有限；③基准基于已发表研究，可能缺乏多样性与覆盖范围；④未覆盖列组合的自动化归一化与合成；⑤缺乏对多列归属准确度的细粒度评估。

---

## 66. Elbow-Based MoE Routing: A Training-Free Inference Time Plugin for Expert Selection

**arXiv ID:** 2608.04401 | [PDF](https://arxiv.org/pdf/2608.04401v1)

**作者:** Robin Pan `[一作]` (Harvard University), Rosa Wu `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于弯点的动态专家路由方法，在推理时根据路由器概率分布自动确定每个 token 激活的专家数目。

**💡 创新点**

创新点在于不需要额外训练或超参数，仅在推理时通过检测概率曲线的“elbow”来自适应选择专家，保持负载均衡。

**🔧 技术方法**

采用 Kneedle 算法寻找弯点，利用 softmax 路由概率排序并计算弯角，进而实现基于弯点的剪枝规则。

**📊 数据集**

在六个多选基准数据集（MMLU、ARC‑Easy、ARC‑Challenge、HellaSwag、PIQA、WinoGrande）上进行评估。

**📈 对比分析**

与固定 top‑8 路由对比，保持准确率差异≤0.33点，平均延迟下降5.3%，平均激活专家从8降至约7.6。

**⚠️ 局限性**

局限在于弯点方法假设路由概率呈现明显头尾分离，若分布平坦则效果有限；并未在更大规模或不同模型上验证。

---

## 67. On Design Principles for Efficient Heterogeneous DRAM-PIM-GPU Systems

**arXiv ID:** 2608.04169 | [PDF](https://arxiv.org/pdf/2608.04169v1)

**作者:** Corey Lammie `[一作]` (IBM Research), Irem Boybat `[通讯]` (IBM Research)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过系统级仿真评估了在GPU与DRAM‑PIM协同工作下，LLM解码阶段的能效，并提出了三条设计原则。

**💡 创新点**

创新点在于：①首次量化静态功耗对整体能效的影响；②发现解码性能随通道数单调递增，并在固定容量下所有模型共享近最优区间；③提出基于行局部性的RowLocalChunk映射方案，展示其在核级能耗上的可观提升，但对端到端效益有限。

**🔧 技术方法**

使用了基于NBPU的DRAM‑PIM模型、改进的ramulator2+attacc_simulator+Pimba仿真框架、静态功耗与刷新能耗建模、NVLink/CXL互连延迟/带宽模型，以及RowLocalChunk映射算法。

**📊 数据集**

使用的工作负载包括OPT‑7B/70B和Mamba2‑2.7B/70B两个解码器，配合不同批量、输入/输出长度进行评估。

**📈 对比分析**

通过将GPU‑PIM混合系统与单纯GPU基线在tokens/s/W、延迟和能耗上进行对比，发现忽略静态功耗时PIM系统高估效能多达3.85倍；在固定容量下，通道数增至64时能效提升可达140–2300%；映射优化提升核级能耗可达17.4%，但端到端提升仅约5.6%。

**⚠️ 局限性**

局限性包括：仿真依赖多项假设（如刷新能耗、静态功耗、互连参数），未在真实硬件上验证；仅关注解码阶段，未覆盖训练或推理其他阶段；研究范围限定于A100 GPU与特定DRAM‑PIM配置，缺乏跨平台的通用性验证。

---

## 68. MetaVideoAgent: Automated Video-Agent Evolution for Long-Form Video Understanding

**arXiv ID:** 2608.04587 | [PDF](https://arxiv.org/pdf/2608.04587v1)

**作者:** Benlei Cui `[一作]` (Alibaba Group), Haiwen Hong `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 MetaVideoAgent，一个自动化进化视频理解代理的框架，并构建了 VA‑EvoBench 基准，用于在不同视频分布下自动改进和评估代理性能。

**💡 创新点**

创新点包括：①针对特定视频分布的自动进化机制；②分布感知的初始设计与教师-诊断-进化循环；③利用 Gold Path 生成与最小验证任务进行精准失败诊断；④模块化视频代理表示和责任约束下的代码级更新。

**🔧 技术方法**

使用技术主要包括：大语言模型驱动的工具调用；视频结构化、证据定位、感知、工作记忆和推理模块；分布感知初始化、Gold Path 生成、跨轨迹诊断聚类以及模块级代码更新；同时结合 GPT、Claude、LLaVA 等 LLM 与视觉工具。

**📊 数据集**

数据集为 VA‑EvoBench，由八种不同视频分布（如商品直播、产品演示、剧集、体育、教学、游戏、舞台剧等）构成，每个分布都有演化集与留存集。

**📈 对比分析**

评估方法是在 VA‑EvoBench 上将 MetaVideoAgent 与公开的大语言模型和固定设计视频代理进行对比。MetaVideoAgent 在四轮进化后宏平均准确率从 38.44% 提升至 51.47%（+13.03pp），超过最强固定代理 6.39pp，并在每题 token 与帧消耗上保持最低。

**⚠️ 局限性**

局限性包括：进化过程耗费大量算力和时间；多模态感知与时间定位错误的诊断仍有挑战；依赖 LLM 与工具，缺乏对新分布的通用性；以及进化策略在极端稀疏或长时序视频中的适应性待进一步验证。

---

## 69. NEBULA: A Language - Independent Specification for Opaque Rotating Refresh Tokens

**arXiv ID:** 2608.04115 | [PDF](https://arxiv.org/pdf/2608.04115v1)

**作者:** Matteo Teodori `[一作]` `[通讯]`, Matteo Teodori

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 OAuth 2.0 刷新令牌的精确规范 NEBULA，并实现了十种语言的符合性实现。

**💡 创新点**

创新点在于将 RFC 9700 的政策层细化为可执行的语言无关规范，并通过可执行的行为向量实现跨语言一致性验证。

**🔧 技术方法**

使用 HMAC‑SHA‑256、CSPRNG、分离查询/证明令牌、原子 compare‑and‑set、零扩展的密钥轮换等技术。

**📊 数据集**

使用公开的 RFC 9700 规范描述以及自建的行为向量数据集进行验证，没有使用外部数据集。

**📈 对比分析**

通过共享的 38 个行为场景和 53 条规范要求的可执行套件进行比较，测试覆盖率高、实现间一致性优良；性能未做基准测试。

**⚠️ 局限性**

局限在于未做形式化验证、缺乏正式安全模型证明，存储一致性依赖数据库事务，未覆盖生产部署经验。

---

## 70. SSC: A Verifiable Structured Representation for Bimanual Manipulation Labelling

**arXiv ID:** 2608.04425 | [PDF](https://arxiv.org/pdf/2608.04425v1)

**作者:** Yupu Lu `[一作]` (University of Hong Kong), Jia Pan `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Structured Subtask Chain (SSC) 以及三阶段标注流水线，用于把长周期操作演示拆分为语义子任务并自动验证。

**💡 创新点**

创新点在于将语法解构与场景图状态的状态转移表示相结合，兼顾可读性与可验证性，并引入四个不变式和三层解析级联实现自动一致性检查。

**🔧 技术方法**

采用语法解析、轨迹推理、可选的注释查找、视觉‑语言模型的查询解决以及基于状态图的不变式验证技术。

**📊 数据集**

使用 BEHAVIOR‑1K 数据集，包含 50 个家用任务、3 个 episode，总计约 2,357 个动作单元。

**📈 对比分析**

通过评估 13 个 VL 模型在 VL‑only 与 AVL 两模式下的动作级准确率，最优模型在硬例上达 86% 以上；利用四个不变式检测到 31 个异常，验证器与人工评审结果一致，ensemble 约提升 1–2%。

**⚠️ 局限性**

局限性包括受框架假设限制，只能处理握持与空状态，超出范围的行为会被标记为异常；缺少时间分割前端，需进一步扩展以覆盖更广场景。

---

## 71. Understanding Fault Tolerance of Adversarially Robust Pruned Models

**arXiv ID:** 2608.04173 | [PDF](https://arxiv.org/pdf/2608.04173v1)

**作者:** Manali Dangarikar `[一作]` (Rochester Institute of Technology), Cory Merkel `[通讯]` (Rochester Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对三层CNN在MNIST上进行剪枝、对抗训练与硬件stuck-at-zero权重故障的系统实验，评估三者交互对模型可靠性的影响。

**💡 创新点**

首次量化剪枝、对抗训练和硬件故障共存时的相互作用，揭示对抗训练会提升输入鲁棒性却降低对硬件故障的容忍度，而剪枝对硬件故障敏感度影响不显著。

**🔧 技术方法**

使用PGD/FGSM对抗训练，基于权重幅值的全局剪枝并再训练，采用软件模拟stuck-at-zero权重注入来模拟硬件故障。

**📊 数据集**

采用MNIST手写数字数据集进行训练与测试。

**📈 对比分析**

通过干净准确率、FGSM/PGD攻击准确率以及不同故障率下的准确率进行对比；实验显示对抗训练显著提升攻击鲁棒性，但在硬件故障率升高时准确率显著下降，剪枝对准确率影响不大。

**⚠️ 局限性**

实验仅局限于stuck-at-zero故障，模型规模和数据集有限（三层CNN、MNIST），中间故障率下结果方差较大，未在真实神经形态硬件上验证。

---

## 72. SiMDex: Mining Similar Egocentric Videos for Cross-Embodiment Dexterous Manipulation

**arXiv ID:** 2608.04196 | [PDF](https://arxiv.org/pdf/2608.04196v1)

**作者:** Nie Lin `[一作]`, Yoichi Sato `[通讯]` (University Of Tokyo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于相似度的三阶段检索框架 SiMDex，在大规模视听动作(VLA)训练中，从数千万人类第一人称视频中挑选与机器人演示任务最相关的示例。

**💡 创新点**

创新点在于：①把人类数据选择视为工业推荐问题；②构建了无架构改动的三层召回‑排序‑再排序轻量级检索流水线；③利用姿态、语言和光流等多模态信号实现跨体型（人类↔机器人）检索。

**🔧 技术方法**

使用了手指相对几何表征、文本句子嵌入、光流描述子、欧氏距离、余弦相似度、以及基于流匹配的 VLA 训练目标；同时采用了共享 42 维动作空间与 88 维混合动作空间的掩码机制。

**📊 数据集**

人类数据来源 EgoDex（约 32M 帧，300 小时），机器人数据来自 12.4 小时双手远程操作演示。

**📈 对比分析**

与随机抽取同等规模人类样本的基线相比，SiMDex 在三项真实世界抓取/操纵任务中将整体成功率从 47.7% 提升至 61.1%（提升 13.4%），在低机器人数据量下更显著（如 Flick Wheel 成功率从 24.5% 跃升至 45.5%）。

**⚠️ 局限性**

局限性包括：仅在单一工业装配场景验证；当人类数据缺乏高质量对应技能时检索效果受限；检索仅基于运动学信息，未考虑接触力、物体状态等语义因素；目前检索是一次性离线操作，缺乏训练过程中的动态适配。

---

## 73. Static Timing Orchestration for Tree-Structured Robot Control Firmware

**arXiv ID:** 2608.04600 | [PDF](https://arxiv.org/pdf/2608.04600v1)

**作者:** Wang Xi `[一作]` (Shanghai Jiao Tong University), Jianping He `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于树形设备模型的控制固件生成框架，并通过静态时间调度实现对固件执行顺序的可预测控制；

**💡 创新点**

创新点在于利用C++对象初始化顺序获取设备注册顺序，构造无运行时依赖图的静态调度策略，并在此基础上给出完整的调度、可达性与延迟分析；

**🔧 技术方法**

主要技术包括：设备层两阶段任务分解（状态更新/决策生成）、树形依赖注入、Rate‑Monotonic分桶优先级分配、预抢占单核调度与基于周期性负载的响应时间上界推导；

**📊 数据集**

实验采用开源水下机器人平台（RoboMaster Type C）中的 14 个设备（13 叶子 + 1 根），并在其开放式控制固件基础上实现框架；

**📈 对比分析**

与基线（中断标志+主循环轮询）相比，平均传感器到执行器的延迟从 1610.00 µs 降至 53.64 µs，执行器抖动从 18.87 µs 降至 5.34 µs，显示出显著的时序性能提升；

**⚠️ 局限性**

局限性包括：依赖周期是谐波或相同的假设；需要手工提供 WCET 估计；仅适用于单核预抢占式调度；对动态设备增删、非树形依赖的支持有限。

---

## 74. Topological Semantics for Scoped Computational Paths

**arXiv ID:** 2608.04228 | [PDF](https://arxiv.org/pdf/2608.04228v1)

**作者:** Arthur Freitas Ramos `[一作]` (Microsoft), Tiago M. L. de Veras `[通讯]` (Universidade Federal Rural de Pernambuco)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种以可计算路径为基础的拓扑语义框架，定义了带有限步骤的可持续重写呈现、其取quotient箭头空间，并给出最终域（final‑domain）拓扑群oid的构造和通用性质；同时提供了从该框架到标准路径类群oid的映射，并在圆和二维环面上通过有效正规化示例验证完整性。

**💡 创新点**

创新点在于：①将可计算路径的语法与拓扑几何实现分离，形成“可持续”重写关系；②构造最终域拓扑使得箭头乘法在不依赖产品-商兼容的情况下保持连续；③给出四路判据与紧-Hausdorff条件，将产品-商兼容性与典型的可拓扑群oid条件精确对接；④提出可计算路径的完备性判据（有效正规化证书），从而在有限生成呈现上实现几何完整性；⑤通过Lean 4.24.0 的形式化验证证明所有定理。

**🔧 技术方法**

技术主要包括：拓扑空间与连续映射理论、路径空间的compact‑open拓扑、分裂和初始/商拓扑构造、最终域群oid的定义与证明、终极呈现的构造、正规化与可计算路径的归约算法、以及在Lean中的形式化证明。

**📊 数据集**

该工作为纯理论研究，未使用任何机器学习或实验数据集；验证工作通过Lean 4.24.0 的形式化证明完成。

**📈 对比分析**

比较方法：通过构造从可持续呈现到标准路径类群oid的连续群oid同态 R_𝒫，并给出完备性条件下的单射与同胚性质；在圆与环面示例中使用正规化归约演示了 R_𝒫 的单射性。由于该工作为理论证明，未给出数值性能指标。

**⚠️ 局限性**

局限性包括：①需要手工给出正规化证书以证明完备性；②最终域拓扑在某些情况下与普通可拓扑群oid 的乘法拓扑不一致，需满足紧-Hausdorff或离散等额外条件；③框架假设所有步骤与重写规则均可连续化，限制了对非平滑或离散空间的直接应用；④未涵盖高阶同伦结构的完整性（仅处理路径级别）。

---

## 75. When More Becomes Less: Position-Dependent Repetition Effects in Language Models

**arXiv ID:** 2608.04021 | [PDF](https://arxiv.org/pdf/2608.04021v1)

**作者:** Han-yu Wang `[一作]` (University of Hong Kong), Han-yu Wang `[通讯]` (University of Hong Kong)

**通讯引用:** 80284 | [OpenAlex ID](https://openalex.org/A5100462720)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种两探针（相邻探针与位移探针）设计，研究在语言模型中重复目标词对预测概率的影响，并发现位移探针会出现“倒U型”曲线，说明重复的效应随读取位置而异。

**💡 创新点**

创新点在于：①首次将读取位置与重复次数分离，揭示读取位置对重复效应的显著调节；②系统验证了该现象在13个公开模型和四种语言中的一致性；③通过六条件消融、框架语用控制和预先给出答案实验，明确归因于精确词汇重复；④将注意力分配与倒U型行为相关联，提供内部机制线索。

**🔧 技术方法**

使用了基于Transformer的掩码语言模型（MLM）和因果语言模型（CLM）架构，结合cloze式探针、序列对齐、注意力测量、slot‑state 观察以及线性回归、Bootstrap 等统计方法。

**📊 数据集**

实验数据集包括：261个手工挑选的单词目标（后根据不同模型的分词器过滤为182–258个），四种语言（英语、西班牙语、中文、德语、法语）中的单词列表，5种句型框架（F0–F4）以及13个公开模型的权重。

**📈 对比分析**

比较方法：对每个模型、每个框架计算每词倒U型“下降量”（峰值概率减去N=30时概率除以峰值），采用Bootstrap 置信区间检验显著性；所有模型的下降量均显著为正，表明倒U型现象普遍存在；在模型规模、语言、框架等维度上也保持一致，说明该效应与模型参数、训练目标无直接线性关系。

**⚠️ 局限性**

限制包括：①仅测试单词目标，未覆盖多词目标；②相邻探针仅在三模型上验证，未覆盖全部13模型；③框架翻译可能导致跨语言差异；④内部相关性无法证明因果机制；⑤对低概率目标的倒U型解释可能受极端概率分布影响。

---

## 76. Large Language Models and Social Media Information Integrity: Opportunities, Challenges, and Research Directions

**arXiv ID:** 2608.04375 | [PDF](https://arxiv.org/pdf/2608.04375v1)

**作者:** Junjie Xiong `[一作]` (Missouri University of Science and Technology), Lingyao Li `[通讯]` (University of South Florida)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了2019–2024年间关于大型语言模型（LLM）在社交媒体信息完整性（信息失真、社交机器人、隐私等）领域的研究，系统筛选1048篇论文，深入分析215篇代表性工作。

**💡 创新点**

首次构建了内容‑代理‑基础设施三维框架，系统化地阐释LLM的双重角色（既能检测又能生成失真内容），并识别跨语言、实时监测、隐私保护等关键空白，提出未来研究方向。

**🔧 技术方法**

采用PRISMA流程的系统综述方法，对LLM模型（GPT系列、LLaMA、Bloom等）、多模态融合、检索增强（RAG）、RLHF、对抗测试、隐私技术（差分隐私、联邦学习）等技术进行归纳与评估。

**📊 数据集**

整合多大规模公开数据集：LIAR、FakeNewsNet、CoAID、Fakeddit、TwiBot‑20/22、CONFAIDE等，用于验证信息失真检测、机器人识别和隐私泄露情况。

**📈 对比分析**

对比了不同LLM与传统方法在误报率、召回率、F1、AUC等指标上的表现，指出LLM在多模态、跨语言检测中可提升5–10% F1，但在实时推理、跨平台一致性和隐私泄漏方面仍存在显著差距。

**⚠️ 局限性**

局限包括：缺乏对LLM生成内容真伪细粒度判定，实验多集中于英文/主流社交平台，跨语言和低资源环境验证不足；缺少长期实地部署案例，隐私保护与性能折衷尚未系统评估。

---

## 77. PhysMind: From Video to Executable Worlds for Training-Free Physical Reasoning

**arXiv ID:** 2608.04575 | [PDF](https://arxiv.org/pdf/2608.04575v1)

**作者:** Chen Yang `[一作]` (Tsinghua University), Chen Wang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

PhysMind提出了一种无训练、代理式框架，将视频转换为可重用的可执行物理世界，并基于该世界回答解释性、预测性和反事实问题。

**💡 创新点**

核心创新在于构建一次性、问题无关的可执行世界，并通过解析连续时间系统识别快速拟合物理参数，避免逐帧模拟；同时将VLM与多种感知与重建工具协同，形成端到端推理管线。

**🔧 技术方法**

采用SAM 3D、MoGe-2、Video Depth Anything进行几何重建，FoundationPose对齐轨迹；利用VLM指导约束与姿态校正；通过解析连续时间动力学（弹性碰撞、摩擦、自由飞行）进行系统辨识；最终通过可执行的物理模拟与查询驱动的执行产生答案。

**📊 数据集**

在CLEVRER（合成物体移动视频）和Physion++（隐含属性预测）两个基准上进行评估。

**📈 对比分析**

与随机、盲目参考、基础VLM（如Gemini、GPT‑5.5）以及训练式与无训练式方法对照，PhysMind在CLEVRER上实现72.55%的问题精度、87.22%的选项精度，较GPT‑5.5提升19.25点的反事实准确率；在Physion++上获得59.64%的总体精度，超过GPT‑5.5 1.57点；同时在API费用上表现更优，成本-准确率比最优。

**⚠️ 局限性**

主要局限体现在感知与姿态重建误差、动力学拟合不准、对视角有限或边缘遮挡的依赖，以及对复杂多接触拓扑和非刚体物体的建模不足，导致错误集中在识别、碰撞时序和小轨迹偏差上。

---

## 78. YOLO-PVC: 2D-to-3D Consolidation of Slice-wise Detections for Volumetric Liver Tumor Localization in MRI

**arXiv ID:** 2608.04642 | [PDF](https://arxiv.org/pdf/2608.04642v1)

**作者:** Talha Waqas `[一作]` (ESME Research Lab), Yasmina Leroul-Chenoune `[通讯]` (ESME Research Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种轻量级的 YOLO-PVC 框架，将 2D 切片检测结果通过深度连续性滤波和分位数聚合转化为 3D 轴对齐边界框，并可选用轻量 MLP 进行轴向校正。

**💡 创新点**

创新点在于将 3D 整合视为结构化估计问题，采用分位数统计消除极端误检、深度连续性约束以及可学习的 MLP 校正，从而显著降低深度偏差并提升整体 IoU。

**🔧 技术方法**

使用 Ultralytics YOLO11s 进行 2D 检测，随后进行深度连续性筛选、10/90 分位数聚合、可选的 MLP 轴向校正，评估指标包括 IoU3D、Dice3D、BEV IoU、质心误差和轴向误差。

**📊 数据集**

在 142 份肝脏 MRI（85 HCC、22 CCA、35 Mixed）上训练并测试，使用病人级别划分，切片级别标签由放射科专家提供。

**📈 对比分析**

与五种传统 2D→3D 聚合基线（加权平均、修剪均值、中值融合、连续性链接、最小-最大堆叠）比较，YOLO‑PVC 取得 IoU3D 0.665，Hybrid MLP‑PVC 提升至 0.710，较基线提升约 19% 相对改进，Dice3D 也从 0.689 提升至 0.781。

**⚠️ 局限性**

局限性包括对结构化误检仍敏感，需依赖 2D 检测质量，且仅处理轴对齐框；当前仅在动脉相 MRI 上验证，跨模态或多中心推广仍待评估。

---

## 79. K-EXAONE 2.0 Technical Report

**arXiv ID:** 2608.04505 | [PDF](https://arxiv.org/pdf/2608.04505v1)

**作者:** Eunbi Choi `[一作]` (LG AI Research), Chansik Yoon `[通讯]` (LG AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过对前代K-EXAONE模型进行深度与宽度扩展，并在此基础上进行持续预训练、难度聚焦中期训练和后期微调，构建了一款拥有750B总参数、可在256K上下文长度内工作的多语言基础模型K-EXAONE 2.0。

**💡 创新点**

创新点主要包括：① 在保持Hybrid Attention架构的前提下，将模型从48层扩展到78层、专家数从128翻倍到256，采用专家复制+随机旋转噪声打破对称性；② 引入MTP和DSpark两条自我投机解码路径，实现高效推理；③ 通过Active Reading和latent-thought等合成数据提升知识获取；④ 设计长上下文检索验证（NIAH）及多阶段agent工作流数据，强化长句推理和工具调用；⑤ 在后期训练中结合GrouPER等多任务偏好学习，系统提升安全与对齐。

**🔧 技术方法**

核心技术包括：Mixture-of-Experts（MoE）深度+宽度扩展、Hybrid Attention（滑窗+全局注意）、QK-Norm与SWA-only RoPE、MTP与DSpark自我投机解码、Active Reading合成数据、Latent-Thought预训练、Mid-Training阶段性扩展上下文长度、Supervised Fine‑Tuning、GrouPER偏好优化、在线RL、AGAPO与GrouPER优化。

**📊 数据集**

使用的数据集涵盖：K-EXAONE原始混合数据（30B+10B合成），FineWeb2等多语言高质量文本，Korean Public API、K-DATA、NIA、韩国语言文化资源，GitHub PR、SWE‑bench、Terminal‑Bench等软件工程与工具调用数据，AIME、HMMT、IMO、SciCode、SWE‑bench Verified、OpenAI‑MRCR、AA‑LCR、Ko‑LongBench、MMLU‑Pro、GPQA‑Diamond、Humanity's Last Exam、MMMLU、GlobalMMLU‑Lite、PolyMath、KGC‑Safety、ROK‑Fortress等多维度评测基准。

**📈 对比分析**

与前代K-EXAONE相比，K‑EXAONE 2.0在9大评测维度（世界知识、数学、编码/代理编码、工具调用、指令跟随、长上下文理解、韩语、跨语言、以及安全）平均提升超过10%。其中编码相关指标提升约30%，长上下文检索与安全表现尤为突出，显著优于同等规模的开源模型。

**⚠️ 局限性**

局限性包括：仍可能生成不恰当或有偏见内容；在最新信息方面缺乏实时性，可能产生事实错误；对极端稀有或专业任务的推理深度尚不足；模型规模导致推理成本高，尤其在低资源设备上的部署仍具挑战；安全评估虽已加强，但对未来新型攻击的鲁棒性仍需进一步验证。

---

## 80. EmpaAva: An Open-source Agentic 3D-Avatar Empathetic Live Chatbot

**arXiv ID:** 2608.04709 | [PDF](https://arxiv.org/pdf/2608.04709v1)

**作者:** Jie Yang `[一作]` (National University of Singapore), Hao Fei `[通讯]` (University of Oxford)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发并公开了一个可实时面向面交互的3D Avatar共情聊天机器人 EmpaAva，能够根据用户的语音、语调和可选视觉信息感知情绪并以情感化语音、同步面部动作和逼真渲染方式回复。

**💡 创新点**

创新点包括：① Tri-Agent 架构将感知、响应规划和渲染拆分为独立的 LLM 驱动代理，形成闭环情感反馈；② Response Planning 层将 LLM 的文字回复转化为多模态执行计划，保证语音、面部表情和渲染在同一情感意图下统一；③ 结合 FLAME 3D 骨架、3D Gaussian Splatting 渲染与情感 TTS（EmotiVoice），实现面向多模态且具表达一致性的全流程。

**🔧 技术方法**

使用的技术包括：大型语言模型（LLM）、语音识别（ASR）、语音情感识别（SER）、可视帧采样、FLAME 3D 头部表情控制、3D Gaussian Splatting 渲染、情感驱动 TTS（EmotiVoice）、Audio‑to‑Motion 模块以及 FLAME‑to‑GS 转换。

**📊 数据集**

评估使用了 EmpatheticDialogues、AvaMERG、EmpathyEar 等公开数据集，并在这些数据集上与多种基线模型（非 LLM、LLM、2D 头像和多模态系统）进行对比。

**📈 对比分析**

通过文本层面指标（情绪识别准确率 Acc.、多样性 Dist‑1/2）以及端到端多模态指标（Dist‑2、Emo.Acc、Cause.M）进行比较。EmpaAva 在所有指标上均优于现有基线，在人工评测中共情、相关性和偏好评分也最高。

**⚠️ 局限性**

局限性包括：情绪识别可能受限于 SER 的准确性；单轮或短对话为主，缺乏长期情感记忆；渲染和音频同步仍有小幅延迟；对资源需求高，且在实际健康服务中仍需专业监管。

---

## 81. CheckOne: Lightweight Fault Detection and Mitigation for Vision Transformers

**arXiv ID:** 2608.04035 | [PDF](https://arxiv.org/pdf/2608.04035v1)

**作者:** Mohammad Hasan Ahmadilivani `[一作]` (Tallinn University of Technology), Jaan Raik `[通讯]` (Tallinn University of Technology)

**通讯引用:** 1793 | [OpenAlex ID](https://openalex.org/A5010286547)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种轻量级的错误检测与缓解方法CheckOne，用于保护Vision Transformer（ViT）模型在硬件故障（单比特翻转）下的推理过程

**💡 创新点**

创新点在于通过在矩阵乘法中追加单一向量1，在线计算权重与激活的求和（checksum），无需在输出上重新计算校验，从而实现对权重和激活同时检测与零值修复，显著降低了内存带宽与计算开销

**🔧 技术方法**

利用GEMM（矩阵乘法）优化、在线checksum、基于预先计算的金典求和与激活范围进行误差检测与定位、错误归零以及与传统ABFT的对比实现

**📊 数据集**

使用ImageNet验证集（5000张图像）进行误差注入实验和性能评估

**📈 对比分析**

与传统ABFT方法在相同的误差注入条件下进行对比，测量准确率下降、Critical SDC率以及执行时间；CheckOne在准确率下降<0.05%、SDC率<0.18% 的同时，相比ABFT平均提升约3.8×推理速度（ViT‑Tiny 4.7×，DeiT‑Tiny 4.7×，Swin‑Tiny 2.0×）

**⚠️ 局限性**

仅针对单比特翻转，无法处理多位错误；需要额外存储预计算的checksum和激活范围值，且在更大规模模型或不同硬件平台上的可扩展性与兼容性尚待进一步验证

---

## 82. Differential 6-DOF Pose Estimation with Provable First-Order Immunity to Camera Calibration Errors

**arXiv ID:** 2608.04673 | [PDF](https://arxiv.org/pdf/2608.04673v1)

**作者:** Yueqiang Zhang `[一作]` (Shenzhen University), Qifeng Yu `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于差分投影的6-DOF平台运动估计方法，直接利用相邻帧的图像位移与已知3D控制点求解，无需先估计绝对姿态。

**💡 创新点**

创新点在于通过深度不变近似与Lie群一阶展开，构造差分模型从而完全抵消相机外参平移误差，仅受旋转误差的有界影响；并给出了闭式线性求解、偏差消除一致估计、可观测性与CRLB分析以及有效性边界。

**🔧 技术方法**

采用的技术包括se(3) Lie群展开、深度不变近似、一阶线性化、最小二乘与偏差消除一致估计、Cramér–Rao下界分析以及单/多摄像头的系统推广。

**📊 数据集**

实验使用合成数据（50–100 m距离、5–50个控制点、0.2–1像素噪声）和真实世界双摄像平台（实验室和Qipanzhou桥梁长期监测）的实测数据。

**📈 对比分析**

与多种经典PnP/gPnP方法（LHM、EPnP+GN、RPnP、DLS、ASPnP、OPnP、gOp、gDLS、UPnP、GAPS、EA-GPnP等）进行对比，单目下RMSE降至10.1″、3.7 mm、时延0.34 ms；双目下RMSE约10.6″、3.9 mm、时延0.27 ms，显著优于传统方法，并且对相机外参平移误差具有完整免疫，旋转误差受限。

**⚠️ 局限性**

局限在于需要微小位移（深度不变近似和一阶展开约±30′工作范围），旋转外参误差仍会产生有限偏差；极端光学轴角或极点位置可能导致可观测性下降。

---

## 83. Guideline-as-Oracle: Zero-Annotation Training of an Ophthalmic Telephone Triage Agent

**arXiv ID:** 2608.04772 | [PDF](https://arxiv.org/pdf/2608.04772v1)

**作者:** Chenyu Wang `[一作]` (Shanghai Institute of Microsystem and Information Technology, Chinese Academy of Sciences), Diping Song `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出使用“Guideline-as-Oracle（GAO）”方法，将美国眼科协会（AAO）指南编译成70行规则表，直接为3,000条多轮对话生成实例级标签，从而无需人工标注；

**💡 创新点**

创新点在于：①将专业指南转化为完整对话标签的自动化管线；②构建八种对话生成策略并评估其对模型性能的影响；③通过标签-对话置换验证学习能力来源于指南而非对话表面；

**🔧 技术方法**

采用大规模语言模型（9B 版）进行全参数监督微调（SFT），并结合类权重重采样、护士-token交叉熵训练；

**📊 数据集**

数据集包括：70行AAO指南规则表；3,000条合成对话；201条基准公开 Reddit 眼科叙述作为评估参考；

**📈 对比分析**

方法对比基准为未微调的9B模型、规则表驱动的GPT-5.5、其他七个通用大模型；GAO‑Triage在操作参考一致性上从61.7%提升至74.1%，突发案例召回率从9.5%提升至69.0%，显著优于其他系统；

**⚠️ 局限性**

局限性包括：①评估依赖于作者编制的操作参考，缺乏独立临床验证；②合成对话可能未充分覆盖真实对话多样性；③存在隐私与合规风险，未进行IRB审查或去标识化审计；

---

## 84. SurgNarrator: A Generative Retrieval Framework for Surgical Video Understanding

**arXiv ID:** 2608.04676 | [PDF](https://arxiv.org/pdf/2608.04676v1)

**作者:** Yuqing Feng `[一作]` (University of Liverpool), Baoru Huang `[通讯]` (University of Liverpool)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为SurgNarrator的生成检索框架，用于实时手术视频的查询条件理解。

**💡 创新点**

创新点包括：①构建手术中心化词汇表并按手术类型与语义类别组织；②采用时序感知对比学习以区分视觉相似但语义不同的手术片段；③层次化的手术类型检索策略将检索空间从全局缩小到对应手术类型，提升准确性与效率。

**🔧 技术方法**

技术实现上使用Qwen3-VL-Embedding-8B作为多模态嵌入骨干，并通过LoRA进行参数高效微调；利用自定义的时序对比损失和伪负样本掩蔽；检索时采用余弦相似度和预缓存的词汇/手术类型嵌入。

**📊 数据集**

训练集为SurgLaVi-β（拆分为Surg-Train），评估集为Surg-Eval；此外在十二个零样本下游任务（相位、步骤、动作、三元组、工具识别等）进行泛化评估。

**📈 对比分析**

与基线（Qwen3-VL-8B-Instruct、SurgCLIP-β、Qwen3-VL-Embedding-8B）比较，SurgNarrator在识别任务Recall@1提升至7.11（比基线高约2.3倍），在推理任务CIDEr提升至24.71，且输出阶段延迟比生成基线低两位数级别；在零样本任务中多数指标均优于现有模型。

**⚠️ 局限性**

局限性在于推理任务的绝对分数仍低于某些生成模型，且对时序硬负样本的阈值设定需经验调优；在极度相似的手术片段中仍可能出现误检。

---

## 85. Learning Compression Rules for Network Traffic

**arXiv ID:** 2608.04545 | [PDF](https://arxiv.org/pdf/2608.04545v1)

**作者:** Quentin Lampin `[一作]` (Orange Research), Massih-Reza Amini `[通讯]` (Universite Grenoble Alpes)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `2704f255-0c84-4173-b83c-0e9a3dbea232` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用数据驱动的方法自动学习并生成适用于 SCHC 的规则压缩器，构建压缩规则集合。

**💡 创新点**

提出两阶段框架：1) 归一化熵比聚类递归划分数据以发现潜在规则；2) 用动态规划在规则预算约束下挑选最优子集，并结合 Good‑Turing 覆盖估计提升泛化；该方法可推广至任何基于规则的压缩方案。

**🔧 技术方法**

采用信息论熵、归一化熵比、分层递归聚类、Good‑Turing 样本覆盖、动态规划预算分配以及 RFC 8724 的 Python 实现。

**📊 数据集**

四个真实流量数据集：Balloon-20k、Thermostat-10k（IoT/CoAP）以及 GTP-traffic、NGAP-traffic（5G 核心网络）。

**📈 对比分析**

与专家手工规则（RFC 8824、结构性基线）在相同训练/测试分割下进行对比；在 IoT 数据集上，RECAP 在中等规则预算下压缩比几乎达到 header ratio，且超过专家规则；在 5G 数据集上显著优于结构性基线，说明学习规则能更好捕捉多样化协议结构。

**⚠️ 局限性**

仅在 SCHC 上验证，需调节阈值 θ 与映射表上限 M_map；对极小样本或极度异质流的学习效果有限；动态规划复杂度为 O(|V|·N²)，对极大树结构可能成为瓶颈。

---

## 86. Compass: Continuously Aligning Social Media Feeds via In-Situ Reflections

**arXiv ID:** 2608.04274 | [PDF](https://arxiv.org/pdf/2608.04274v1)

**作者:** Aadit Barua `[一作]` (University of Texas at Austin), Amy X. Zhang `[通讯]` (University of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 Compass 系统，通过浏览器插件在 YouTube Shorts 中实现持续反思与推荐对齐，使用户在日常浏览时能够即时表达和细化偏好，并让系统自动调整内容推送。

**💡 创新点**

创新点在于：①将反思与对齐过程嵌入使用流，形成持续的“in‑situ”交互；②通过行为信号模拟与 DOM 操作两种机制并行实现对齐；③利用轻量级的多模态语义嵌入实现实时视频-偏好匹配。

**🔧 技术方法**

使用技术包括：实时视频处理管道（提取元数据、三张关键帧，结合 Vision‑Language 模型生成描述并嵌入向量空间）；偏好表示为可扩展的子类别清单并通过用户标注更新；对齐策略分为行为模拟（后台点赞/观看）和直接 DOM 插入/删除；前端为 Chrome 扩展，后端使用 FastAPI + SQLite，调用 OpenAI API（文本嵌入、图像描述生成等）。

**📊 数据集**

数据集：①Field Study 15 参与者（8 Compass，7 Baseline），收集 10 天内的使用日志和主观评价；②技术评估数据集 408 条 Preference‑Video 对，来自用户自行标注的真实推送内容；③利用 YouTube Shorts 公开 API 获取的视频元数据与预加载的 3 张缩略图。

**📈 对比分析**

比较方法：与手动配置基线对比；使用 Mann‑Whitney U、线性混合效应模型分析频次、对齐率和观看时长；技术评估使用 ROC‑AUC 与 F1 衡量不同视频/偏好表示方式。结果显示：Compass 在偏好细化频率（11.5 vs 0.14）、Feed 对齐比例（49.3% vs 20.3%）和观看时长占比（58.7% vs 21.0%）上均显著优于基线，且保持内容多样性；技术评估中 Meta+Frames→LLM 与 2 个例子的视频/偏好表示达到最优性能，ROC‑AUC 0.913，F1 0.792。

**⚠️ 局限性**

局限性：①仅在桌面版 YouTube Shorts 上实现，无法覆盖移动端主流使用场景；②样本量小（n=15），且来自大学生群体，结果泛化性有限；③缺乏长期随访，难以评估持续使用效果；④对高级偏好细粒度控制支持不足，用户对分布均匀的期望无法完全满足。

---

## 87. Technical Report: A Formal Semantics for Java Symbolic Evaluation using Large-Block Encoding

**arXiv ID:** 2608.04513 | [PDF](https://arxiv.org/pdf/2608.04513v1)

**作者:** Soha Hussein `[一作]` (Ain Shams University), Vaibhav Sharma `[通讯]` (Amazon Web Services)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对 Java Ranger 这一 Java 程序路径合并符号执行工具的语义进行了形式化，给出了从原始 Java 字节码到符号执行可执行 IR 的一系列变换（γ‑创建、早期返回消除、重命名、输入代换、方法内联、字段 SSA 与常量引用传播）及其各自的语义定义，并证明每个变换在简化的 Java 语义下保持程序语义的一致性；通过组合这些变换，证明了 Java Ranger 在路径合并过程中的整体正确性。

**💡 创新点**

创新点在于：①首次为符号执行工具（尤其是路径合并工具）提供完整的形式语义模型并严格证明其语义保真；②将传统的 Java 字节码抽象为单一赋值 SSA 形式的 IR 并在此基础上构建多阶段变换链；③提出并证明了针对 Java 字段操作的 SSA 化以及常量引用传播的可组合性与正确性，弥补了现有研究中对字段访问及方法调用处理不足的空白。

**🔧 技术方法**

主要技术包括：形式化语义（简化的 Java 字节码语义与符号执行语义）、SSA 形式转换、γ‑表达式与 ϕ‑表达式的替换、条件路径合并（γ‑创建）、早期返回聚合（条件返回表达式）、变量唯一化（α‑重命名）、参数代换、方法内联（递归展开）、字段 SSA（字段读写转化为临时变量）以及常量引用传播；同时使用逻辑约束求解器（SMT）进行路径条件求解。

**📊 数据集**

使用了 SV‑COMP 2026 Java 基准集中的 7 个工具评测数据（包括 GDart、JBMC、jLiSA、MLB 等）以及 Java Ranger 自己的标准基准程序，用以验证形式化模型与变换实现的覆盖范围；此外，对 Java 程序的实例化代码（如 `separateBits`）进行了详细的变换示例。

**📈 对比分析**

本文的评估侧重于形式正确性而非运行性能；通过在 SV‑COMP 环境下对比已知错误案例（如 2020–2025 年 Java Ranger 的未发现的错误），验证了形式化证明能够揭示实测中难以捕获的安全缺陷；实验结果显示，在保证了所有变换的语义保真后，工具在验证安全性时与传统符号执行保持一致，且不引入新的错误。

**⚠️ 局限性**

局限性包括：①形式化模型为简化的 Java 语义，未覆盖完整 Java 语言特性（如异常、数组、反射、并发等）；②未提供机器可检验的证明（如 Coq/Isabelle）；③在循环和递归深度较大的程序中，路径合并仍可能导致巨大中间表示；④方法内联深度有限，无法完整处理无限递归或高度动态分派的场景；⑤仅针对 Java Ranger 的实现，无法直接推广到其他符号执行工具。

---

## 88. Non-asymptotic implicit bias of logistic regression at early-stage gradient descent dynamics

**arXiv ID:** 2608.04382 | [PDF](https://arxiv.org/pdf/2608.04382v1)

**作者:** Han Bao `[一作]` `[通讯]` (Institute of Statistical Mathematics), Han Bao (Institute of Statistical Mathematics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究梯度下降在无正则化的线性分类器（指数损失或逻辑回归）中，首次证明参数在早期阶段能以“双指数”时间 O(exp(exp(-δ))) 内实现弱最大间隔对齐（即 V(t)≤δ），大幅快于传统的慢收敛速率。

**💡 创新点**

创新点在于：① 通过径向和切向流分解直接分析对齐动力学；② 证明早期阶段的弱对齐是紧的（下界匹配）；③ 将结果推广到逻辑回归损失；④ 给出离散时间的完整上界与下界。

**🔧 技术方法**

主要技术包括：梯度流解析、径向/切向分解、能量/Lyapunov函数、温度单调性、几何不等式与投影算子、连续时间到离散时间的误差控制。

**📊 数据集**

使用的是任意线性可分数据集，假设具有正余弦间距 γ；论文中通过仿真展示不同 γ 的情形，但未使用标准公开数据集。

**📈 对比分析**

与传统的渐近对齐分析（V(t)=O(1/log²t)）相比，早期对齐收敛时间仅为 O(exp(exp(-δ)))；对比感知机理论的 O(1/γ) 迭代数，本文提供更小的误差 δ 的快速收敛上界，仿真结果验证了理论预期。

**⚠️ 局限性**

局限性：① 只能实现弱对齐，误差 δ 受数据集间距 γ 限制，无法达到任意小误差；② 需严格小步长（η ≤ c min{ρ/L(0), γ²}），对大步长/深度网络的适用性未知；③ 结果主要针对单层线性模型，扩展到更复杂模型仍是开放问题。

---

## 89. The Fairness Collapse Phenomenon: Bias Amplification in Language Models Trained on Synthetic Data

**arXiv ID:** 2608.04268 | [PDF](https://arxiv.org/pdf/2608.04268v1)

**作者:** Irina Proskurina `[一作]` (Laboratoire Hubert Curien), Julien Velcin `[通讯]` (École Centrale de Lyon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Bias in Bios数据集上，作者通过对比迭代式与递归式持续预训练，并分别采用种子续写与少量示例生成方式，系统实验探究了多代生成的人工合成文本对语言模型公平性和整体性能的影响。

**💡 创新点**

提出了“公平性崩溃”(fairness collapse)概念，即在模型持续训练于自身生成文本时，社会偏见先于传统的模型崩溃指标显著恶化，揭示了隐藏的早期警示信号。

**🔧 技术方法**

使用Qwen2.5-0.5B解码器模型进行文本生成和预训练；采用种子续写和少量示例提示技术；评价指标包括EO GAP、NLL-GAP、MMLU、CrowS‑Pairs、SOFA以及困惑度。

**📊 数据集**

Bias in Bios（专业简历）数据集，包含约30万篇人类撰写的职业简介，并按性别标注。

**📈 对比分析**

实验结果显示：在递归训练下，公平性指标（EO GAP、NLL-GAP、SOFA等）在前几轮就显著恶化，而困惑度持续下降、MMLU准确率缓慢下降；迭代式训练的公平性恶化程度较递归低。

**⚠️ 局限性**

局限性包括：仅针对性别与职业的单一公平性评估；使用较小的解码器模型，未验证大模型是否同样受影响；合成数据仅采用种子续写与少量示例生成，未覆盖更广泛的生成策略；评价指标主要基于概率与分类，可能忽略更复杂的社会危害。

---

## 90. Forced Displacement of People Experiencing Homelessness: Housing and Movement Outcomes after Encampment Clearances

**arXiv ID:** 2608.04076 | [PDF](https://arxiv.org/pdf/2608.04076v1)

**作者:** Brandon Morande `[一作]` (University of Washington), Zack W. Almquist `[通讯]` (University of Washington)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究利用2016-2018年西雅图12起大规模露宿营地清除事件的长期街头外展数据，探究被迫迁移者在一年内的住房与迁移路径，评估清除对可见无家可归率的影响。

**💡 创新点**

创新之处在于将关系事件模型（Relational Event Models）应用于无家可归者的个体迁移与服务接触事件，突破传统交叉调查的记忆偏差，能够捕捉事件序列与时间依赖性。

**🔧 技术方法**

技术方法主要是Bayesian估计的关系事件模型（使用relevent R包）预测不同结果的风险，结合欧氏距离与Census区块的空间迁移度量。

**📊 数据集**

数据集来自西雅图人类服务部公开的露营地清除日志（包含结构数量与坐标）与REACH组织的街头外展记录，补充了Homeless Management Information System（HMIS）中的服务交互数据，共计468个被清除事件与468名客户的纵向记录。

**📈 对比分析**

相较于传统描述性分析，该方法能估计事件发生的相对风险（Hazard Ratio）并给出95%可信区间；结果显示被迫迁移后，失去服务联系的风险约为进入收容所的14倍、获得住房的20倍，迁移至非原区块的风险约7倍，但对进入收容所或住房的风险并未显著提高，表明清除并未有效促进安置。

**⚠️ 局限性**

局限性包括：仅观察到已被清除者，缺乏未被清除对照组；数据主要来自单一外展组织，可能低估了与其他服务机构的接触；样本量相对有限导致置信区间宽广；未能区分因其他清除或自愿原因导致的多次迁移；并且研究局限于西雅图，外推性需谨慎。

---

## 91. AI-driven Multimodal Representation Learning for Latent Mediation Structure Discovery of Socioeconomic Disadvantage, Psychosocial Factors, and Cardiometabolic Multimorbidity: Insights from the All of Us Research Program

**arXiv ID:** 2608.04016 | [PDF](https://arxiv.org/pdf/2608.04016v1)

**作者:** Cong Cao `[一作]` (Yale University), Shuangge Ma `[通讯]` (Yale University)

**通讯引用:** 15012 | [OpenAlex ID](https://openalex.org/A5070916971)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建AI驱动的多模态中介框架，将社会经济不利、心理社会因素与心血管代谢多病联系起来

**💡 创新点**

整合多模态变分自编码器进行低维表征，并在潜在空间进行中介分析，首次实现跨域复杂路径探索

**🔧 技术方法**

变分自编码器、潜在空间中介分析、Bootstrap验证

**📊 数据集**

All of Us Research Program（包含社会经济、心理社会、临床、实验室、行为、基因等多模态数据）

**📈 对比分析**

相较于传统线性中介，使用潜在空间能更好处理高维非线性关系；主要路径NIE=0.002517，Bootstrap 95% CI(0.00243–0.00344)验证稳健

**⚠️ 局限性**

研究为观察性、结果在潜在空间而非临床尺度、潜在维度解释有限、完整案例整合导致样本缩减、可能存在选择偏倚

---

## 92. YOLOv14:Unified Cross-Domain Real-Time Object Detectionwith Adaptive Multi-View Representation

**arXiv ID:** 2608.04720 | [PDF](https://arxiv.org/pdf/2608.04720v1)

**作者:** Jinling Jia `[一作]`, Chenbin Zhang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于AdaIN与对抗损失的域不变特征学习框架，用于将真实人类检测模型迁移到游戏角色场景中。

**💡 创新点**

创新点在于将AdaIN与对抗学习相结合，实现特征对齐与域对抗的双重约束，显著提升跨域检测的鲁棒性。

**🔧 技术方法**

核心技术包括AdaIN（Adaptive Instance Normalization）调节特征均值方差、对抗损失学习域不变性、卷积神经网络（如ResNet/YOLO）作为特征提取器。

**📊 数据集**

使用公开的人类检测数据集（如COCO或CrowdHuman）与自建的游戏角色图像数据集（包含多款游戏中的人物）进行实验。

**📈 对比分析**

与传统无迁移、仅对抗迁移以及仅AdaIN迁移的基线方法相比，AdaIN+ℒ_adv方案在目标域上的平均精度（AP）提升约10%-15%，误检率降低约30%。

**⚠️ 局限性**

局限性包括：对极端域差（如极端风格差异或低分辨率）效果有限；需要额外的目标域标注或对抗训练样本；在非人类类别上迁移效果未验证。

---

## 93. Easy to Complete, Hard to Choose: Investigating LLM Performance on the ProverbIT Benchmark

**arXiv ID:** 2608.04670 | [PDF](https://arxiv.org/pdf/2608.04670v1)

**作者:** Enrico Mensa `[一作]` (University of Turin), Daniele Paolo Radicioni `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Italian ProverbIT benchmark，用 100 个多选题测试 LLM 对意大利谚语的理解。

**💡 创新点**

创新点在于将谚语完成任务转化为无正确答案的多选推理，揭示 LLM 在负面推理上的不足。

**🔧 技术方法**

使用了 Chain‑of‑Thought 分析、zero‑shot prompting 以及多模型评估技术。

**📊 数据集**

数据集为手工构造的 100 个意大利谚语多选题，包含 4 个错误选项和“None”选项。

**📈 对比分析**

通过对 13 个前沿模型（包括传统 LLM 与 LRM）进行基准评估，发现完成任务准确率>90%，而无答案多选任务准确率仅 4–77%，显著下降。

**⚠️ 局限性**

局限性包括模型依赖记忆模式、缺乏对否定推理的能力、过度思考、语言切换导致可解释性差以及 CoT 与答案不一致。

---

## 94. Beyond Global Routing Aggregation: Phase-Aware Expert Merging for MoE Vision-Language Models

**arXiv ID:** 2608.04454 | [PDF](https://arxiv.org/pdf/2608.04454v1)

**作者:** Hongyu Zhang `[一作]` (University of Science and Technology of China), Wuyang Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于阶段条件专家角色的训练‑free Mixture‑of‑Experts 视觉‑语言模型专家合并方法。

**💡 创新点**

创新点在于将专家在图像、问题、答案三个推理阶段的路由行为抽象为 Routing Role Profile，并用专家‑阶段信息损失和签名碰撞调整来决定可合并专家组，从而保留答案解码的专家区分。

**🔧 技术方法**

技术包括相位归一化路由统计、RRP 构造、KL 损失度量、单链接聚类与签名碰撞调整，以及共享权重的参数融合。

**📊 数据集**

使用了 DeepSeek‑VL2‑Tiny、DeepSeek‑VL2‑Small、Qwen3‑VL‑30B‑A3B‑Instruct 三种 MoE‑VLM，在 TextVQA–ChartQA–ScienceQA 的混合校准集（TCS）上评估。

**📈 对比分析**

与 Sub‑MoE、NAMEx、MC‑SMoE、MergeMoE、HC‑SMoE、REAM 等方法在相同专家保留比例下对比，本文方法在宏观平均得分上最高，最多可提升 9.6%（在 ρ=0.50 时）。

**⚠️ 局限性**

局限性包括仅验证了两大 MoE‑VLM 家族、依赖固定的校准集且未考虑合并后专家内部参数冲突，未来需要在更广泛模型、动态校准及后置微调中进一步验证。

---

## 95. Suppression Sticks, Locality Is Fragile: A Closed-Loop Target-and-Control Audit of Task-Vector Negation in VLA Policies

**arXiv ID:** 2608.04692 | [PDF](https://arxiv.org/pdf/2608.04692v1)

**作者:** Shaoguang Wang `[一作]` (Hong Kong University of Science and Technology), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对多任务视觉语言动作（VLA）策略进行任务向量减法编辑，并在闭环机器人控制中审计其对目标与控制技能的影响。

**💡 创新点**

首次在闭环目标与控制评估框架下系统性检验任务向量减法的局部性，揭示了分离、抗拒与全局崩溃三种行为模式，并评估了跨动作头和跨套件的一致性与边界。

**🔧 技术方法**

采用MergeVLA多任务模型与LoRA专家向量，利用任务向量减法（task arithmetic）以及梯度无关的数据自由编辑；同时对比梯度基记忆抹除方法（NegGrad、NegGrad+、Gradient-Difference）。

**📊 数据集**

在LIBERO任务库上进行实验，重点使用LIBERO-Goal 10个技能，另外对LIBERO-Spatial、Object、Long10进行跨套件评估。

**📈 对比分析**

与梯度记忆抹除基线及重训练上限进行对比，任务向量减法在目标抑制下实现约73.9%的控制保持率，略低于Gradient-Difference的88.3%，且无需编辑时的数据或梯度计算，显著降低了计算成本。

**⚠️ 局限性**

实验仅在单一随机种子、仿真环境下完成，依赖专家向量库，跨套件局部性在Object和Long10上失效，且多技能联减不具选择性，未提供永久性不学习证明。

---

## 96. General purpose graphical rendering on quantum devices with composable function systems

**arXiv ID:** 2608.04022 | [PDF](https://arxiv.org/pdf/2608.04022v1)

**作者:** James Schloss `[一作]` (MIT), Ayaka Usui `[通讯]` (Universitat Autonoma de Barcelona)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了可在量子架构上实现的可组合函数系统（CFS）并展示了第一段量子视频，证明了在IBM Kingston硬件上进行通用渲染的可行性。

**💡 创新点**

创新点在于将经典CFS与量子迭代函数系统（QIFS）结合，利用Husimi函数直接从量子态获得像素分布，克服了传统量子渲染需要大内存或复杂映射的局限；同时实现了在噪声中量子实现几何变换（位移、旋转、压缩、抖动、复制）并在实际硬件上完成完整场景渲染。

**🔧 技术方法**

使用技术包括：Qiskit量子编程、量子迭代函数系统（QIFS）、Husimi函数测量、Fock态混合、相空间变换（位移、旋转、压缩、拉伸、剪切）、量子通道（扩展）、多测距（投影）以及经典-量子混合后处理。

**📊 数据集**

数据集主要为自制的几何场景（如原子、笑脸、光标等）和合成视频帧；未使用公开图像/视频数据集。

**📈 对比分析**

对比方法包括经典CFS渲染、经典模拟器、噪声模拟以及真实IBM Kingston量子硬件；结果显示在有限的4~5量子位下可生成可辨识的图像，并证明量子实现具备潜在加速优势，但实际性能受限于量子噪声、测量次数与硬件延迟，渲染时间在数分钟到数小时之间。

**⚠️ 局限性**

limitation包括：缺乏着色和纹理支持、无统一的量子/经典接口、量子硬件噪声导致图像模糊、需要大量测量导致的高延迟、有限的量子位限制可绘制场景大小、未实现非幺正映射与收缩、Husimi函数测量效率低、缺乏可扩展的高分辨率渲染方案。

---

## 97. Simulation-Based Imaging: Learning Acoustic Inverse Problems from Simulated Data

**arXiv ID:** 2608.04145 | [PDF](https://arxiv.org/pdf/2608.04145v1)

**作者:** Luke Bodmer `[一作]`, E. Bruce Pitman `[通讯]` (University at Buffalo)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了 Simulation‑Based Imaging（SBI）框架，利用在高精度离散Galerkin（DG）仿真中生成的 1000 条三维声学数据，用 2D 卷积神经网络完成从 144 个边界传感器时序到 32×32×32 体素密度图的直接映射，实现了 3D 声学逆问题的实时反演。

**💡 创新点**

创新点在于：①完全用仿真数据训练逆向模型，彻底消除了传统全波形反演（FWI）的迭代计算瓶颈；②采用 k‑space 输出与 MSE 损失相结合的训练策略，使得网络自然偏向低频、平滑解；③通过传感器稀疏实验证明仅 17% 的传感器即可保持误差低于 4%，极大降低硬件成本。

**🔧 技术方法**

主要技术包括：高阶节点 DG 前向仿真（p=2）、GPU 加速、三维体素化、k‑space 预测、2D CNN 结构（两层卷积 + 自适应池化 + 全连接回归）、残差连接、Dropout、AdamW + OneCycleLR 训练策略，以及多噪声和多传感器配置的交叉验证。

**📊 数据集**

使用的数据集为 1000 条由 DG 仿真生成的三维声学案例，涵盖 1–3 个不同尺寸、位置的高密度立方包裹物，背景材料为 ρ_b=2.0 kg/m³、c_b=2.0 m/s，传感器共 144 个，采样时序约 600 步，随机生成高斯脉冲源。

**📈 对比分析**

通过与不同噪声水平（0%、5%、10%）以及传感器稀疏配置（全 144、均匀 120、棋盘 72、钻石 48、面心 24）进行 5‑折交叉验证比较，结果显示：在 5% 噪声下体素误差仅提升 13%；在仅使用 24 个传感器（17%）时误差仍低于 4%，并且整体误差变化不超过 7%，说明模型在噪声和硬件稀疏性方面具有出色的鲁棒性。

**⚠️ 局限性**

局限性在于：所有验证均基于数值仿真，缺乏真实实验数据验证；训练数据规模相对有限，未涵盖更复杂的几何形状、材料非线性、介质耗散等实际情况；模型在仿真与实测之间可能存在“sim‑to‑real”差距。

---

## 98. EgoAfford: Task-Oriented Affordance Grounding via Egocentric Referring Segmentation

**arXiv ID:** 2608.04533 | [PDF](https://arxiv.org/pdf/2608.04533v1)

**作者:** Xinyuan Guan `[一作]` (Shanghai Jiao Tong University), Lixin Yang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了EgoAfford基准并提出EgoLens模型，实现在第一人称桌面场景中基于当前观测与高层任务预测剩余步骤并对下一步的直接物体、工具和目的地进行部件级功能区域分割。

**💡 创新点**

创新点在于：①将任务进度推理、功能部件分解与多角色定位三者统一到同一框架；②通过生成式LLM、Grounded‑SAM和SAM2自动化生成并人工验证的多步图像系列及真实图像集；③设计EgoLens基于LENS的多查询架构，使模型在单前向传播中同时输出规划文本和三角色掩码。

**🔧 技术方法**

使用技术包括：LLM（多种大模型）生成任务与场景描述、Grounded‑SAM定位对象、SAM2产生细粒度掩码、Qwen2.5‑VL Backbone+LENS架构实现查询驱动生成、CoT式规划生成、交叉编码器相似度评估、Hungarian算法进行步骤匹配、BCE+Dice损失进行掩码训练。

**📊 数据集**

数据集为EgoAfford：15.5k人类验证的合成图像，覆盖2000个多步桌面场景；EgoAfford‑Real：102张真实拍摄图像，涵盖26个任务。数据包含4,123种对象名称与391种动词（聚类后1,188种对象类型、357种动作类型）。

**📈 对比分析**

与OMG‑LLaVA、Sa2VA、UniPixel、LENS及商业VLM+SAM2等基线进行比较；EgoLens在gIoU 0.700、cIoU 0.486、CSR 0.609、覆盖率最高，显著优于基线；在零样本迁移至真实图像时gIoU 0.666、cIoU 0.455，同样保持领先。

**⚠️ 局限性**

局限性包括：①生成数据缺乏自然环境中的杂乱度与动态变化；②模型仅在单一路径上训练，对多解或规划误差的鲁棒性有限；③仅使用静态图像评估，未覆盖运动可行性与闭环控制等实际执行挑战。

---

## 99. Image Classification Using CNN-QNN Hybrid Model with Optimized Correlated Features

**arXiv ID:** 2608.04379 | [PDF](https://arxiv.org/pdf/2608.04379v1)

**作者:** Minseo Seong `[一作]` (Sogang University), Youngwook Kim `[通讯]` (Sogang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一个在CNN特征提取后引入的相关性正则化方法，用以调节特征之间的相关系数，提升量子神经网络（QNN）在图像二分类任务中的准确率与稳定性。

**💡 创新点**

创新点在于将经典特征的相关性与量子纠缠结构对齐，证明中等相关性（≈0.5）能最优利用QNN的纠缠能力，从而显著提升性能；同时仅需在特征阶段加入一个标量超参数，无需改动量子电路。

**🔧 技术方法**

采用了CNN（浅层和ResNet-18）提取8维特征、ZFeatureMap量子特征映射、变分量子电路（含逆向纠缠链）以及基于均方误差的相关性正则损失；训练使用Adam优化器并在Qiskit模拟器上实现。

**📊 数据集**

实验数据集包括Fashion‑MNIST（shirt vs. coat）、CIFAR‑10（automobile vs. truck）以及雷达微多普勒签名（robot vs. non‑robot）。

**📈 对比分析**

与不使用相关性正则的基线以及使用多层感知机（MLP）分类头进行对比。结果显示，在所有数据集上，目标相关系数≈0.5时平均准确率提升约0.5–1%，且方差下降；相较于MLP头，浅层特征提取时QNN头表现更优。

**⚠️ 局限性**

局限性包括仅针对二分类任务、仅在模拟器上验证、使用8个量子比特、相关性目标为统一常数，未探讨多类别、真实硬件噪声或更复杂的VQC结构。

---

## 100. Visual Anchoring in Diffusion: Multimodal Zero-Shot Skeleton Action Recognition

**arXiv ID:** 2608.04623 | [PDF](https://arxiv.org/pdf/2608.04623v1)

**作者:** Zehao Bao `[一作]` (University of Hong Kong), Bruce X. B. Yu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一种基于扩散的多模态骨架-文本匹配模型TDSM-MM，用RGB作为视觉锚点改进零样本动作识别。

**💡 创新点**

创新点在于将RGB视为不扩散的条件token，融入单一扩散去噪器，避免传统后期融合的加权难题，并在候选类别上通过去噪误差进行评分。

**🔧 技术方法**

采用了DDPM扩散框架、CrossDiT去噪器、Shift‑GCN骨架特征、OpenCLIP ViT‑H文本/视觉编码器，以及候选文本能量最小化。

**📊 数据集**

在NTU RGB+D 60和120的SynSE拆分（48/12、55/5、96/24、110/10）上进行评估。

**📈 对比分析**

与骨架单模态TDSM、BSZSL、Flora等基线对比，TDSM‑MM在三四个拆分中取得最高无监督精度，在NTU‑120 96/24上无测试时适配也超越了转导式DynaPURLS（71.3% vs 69.1%）。

**⚠️ 局限性**

局限性包括对RGB质量高度敏感，标签不匹配或噪声会显著下降；需要为每个样本生成RGB特征，计算成本略高；并未完全解决视觉混淆类别的误判。

---

## 101. RUTA: Principled Visual Token Allocation via Rate-Utility Optimization

**arXiv ID:** 2608.04132 | [PDF](https://arxiv.org/pdf/2608.04132v1)

**作者:** Jian Zou `[一作]` (City University of Hong Kong), Kede Ma `[通讯]` (City University of Hong Kong)

**通讯引用:** 10216 | [OpenAlex ID](https://openalex.org/A5020029652)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 RUTA 方法，将视觉令牌稀疏化视为率‑效用优化问题，支持查询条件的候选构造、概率保留和锚聚合，实现对每个图像‑查询对自适应的视觉令牌数量分配。

**💡 创新点**

创新点在于：①将视觉令牌稀疏化正式化为可微的率‑效用优化；②引入查询条件的候选构造与概率保留机制；③通过锚聚合把非保留令牌的信息融合到保留锚中；④用可微率惩罚直接控制整体效率，实现全局自适应令牌分配。

**🔧 技术方法**

率‑效用优化框架、Grounding DINO 视觉定位、双层 MLP 生成保留概率、语义‑空间注意力聚合、直通估计与随机采样、预训练 VLM（LLaVA‑NeXT‑7B、Qwen3‑VL‑8B）以及冻结的视觉编码器与 LLM。

**📊 数据集**

使用五大视觉问答基准：VQAv2、GQA、TextVQA、A‑OKVQA 与 VizWiz。

**📈 对比分析**

与多种训练免费和训练基的视觉令牌压缩方法（FastV、SparseVLM、VisionZip、DivPrune、PruneSID、FastVLM、TwigVLM）在相同低令牌率下对比，RUTA 在匹配 2%–5% 令牌保留时平均保持 88.2%–94.4% 的任务性能，且在 LLaVA‑NeXT‑7B 与 Qwen3‑VL‑8B 两种 VLM 上均获得最高平均准确率。

**⚠️ 局限性**

局限性：①以令牌计数为效率代理，未覆盖实际延迟、KV 缓存、硬件吞吐等；②候选构造采用单个包围框，对分散目标可能覆盖过大；③需要针对不同 VLM/任务再训练，缺乏即插即用的通用性。

---

## 102. Checked-In Secret Detection: Strings Are All You Need

**arXiv ID:** 2608.04523 | [PDF](https://arxiv.org/pdf/2608.04523v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 103. A Unified Model for Cross-Domain Clone Detection via Model Merging

**arXiv ID:** 2608.04215 | [PDF](https://arxiv.org/pdf/2608.04215v1)

**作者:** Palash R. Roy `[一作]` (University of Saskatchewan), Chanchal K. Roy `[通讯]` (University of Saskatchewan)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过后置模型合并技术，将针对不同克隆检测域的专用模型融合成单一跨域检测器，无需额外训练或访问训练数据。

**💡 创新点**

创新点在于系统评估多种任务向量合并方法与层级拼接，并发现同基底合并是关键，提出一种无需再训练、仅基于检查点的实用合并方案。

**🔧 技术方法**

使用的技术包括 TIES、DARE‑TIES、WUDI、PCB、任务算术、贪心层级拼接以及线性跨分词器对齐等模型合并方法。

**📊 数据集**

实验使用的数据集为 BigCloneBench、CLCDSA（Java↔Python）和 GPTCloneBench（AI 生成的克隆），并基于四种预训练代码模型（UniXcoder、CodeBERT、GraphCodeBERT）。

**📈 对比分析**

与单个专业模型、多任务训练和零样本 LLM 对比，合并后的同基底 TIES 模型在 BigCloneBench 与 CLCDSA 的组合 F1 达 0.865，几乎匹配多任务模型，同时在 GPTCloneBench 上提升约 4×，推理成本显著低于 LLM。

**⚠️ 局限性**

主要限制是需要所有专用模型共享同一预训练基底；跨基底合并效果差且跨分词器对齐有限；未在更大规模或其他模型架构上验证，且对极端域间差异的鲁棒性仍待研究。

---

## 104. The RAIL Principles for Neurosymbolic AI: Reasoning, Assurances, Interfacing and Learning

**arXiv ID:** 2608.04285 | [PDF](https://arxiv.org/pdf/2608.04285v1)

**作者:** Agnese Chiatti `[一作]` (Politecnico di Milano), Benjie Wang `[通讯]` (UCLA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并阐述了RAIL（Reasoning, Assurances, Interfacing, Learning）四维原则，用来评估和指导神经符号化 AI 系统的设计与部署。

**💡 创新点**

创新点在于将神经网络与符号推理的融合问题映射为四维空间，提供统一的视角来识别系统在推理、保障、接口和学习方面的权衡与协同；并用该框架重新解读 Alpha 系列、工具增强 LLM、因果学习等多种主流系统。

**🔧 技术方法**

结合的技术包括：深度神经网络、逻辑程序/规则系统、知识图谱、蒙特卡洛树搜索、工具调用（ReAct、Toolformer 等）、因果图模型、物理约束网络、符号约束与差分可微化方法。

**📊 数据集**

本文为概念性综述，未使用具体实验数据集；但在讨论中提及的典型场景包括知识图谱链接预测、Alpha 系列游戏/证明/蛋白折叠、LLM 工具调用等常见公开数据集。

**📈 对比分析**

通过在 RAIL 维度上绘制雷达图，对比不同系统在推理强度、保障水平、接口开放度与学习方式上的差异；并给出案例说明：例如知识图谱方法从纯数据学习向知识引导学习迁移，工具增强 LLM 在保证性与接口方面显著提升。

**⚠️ 局限性**

局限性：缺乏量化指标与可操作化的评估方法，RAIL 维度仍属于定性描述；系统间比较主要基于概念映射，未提供统一实验基准；在可扩展性、工具兼容性和符号-子符号交互的鲁棒性方面仍面临挑战。

---

## 105. Poly-OPD: Heterogeneous Multi-Teacher On-Policy Distillation for Capability-Selectable Flow Models

**arXiv ID:** 2608.04349 | [PDF](https://arxiv.org/pdf/2608.04349v1)

**作者:** Siming Fu `[一作]` (Joy Future Academy), Si Liu `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 Poly-OPD，一种针对异构教师模型的 on‑policy 蒸馏框架，将多种互不兼容的教师（如 FLUX.1‑dev 与 Z‑Image）的优点融合到单一 2.5B SD3.5‑Medium 学生模型中，并实现可切换能力。

**💡 创新点**

创新点包括：1）像素桥（pixel bridge）实现对不同自编码器/噪声调度的 on‑policy 监督；2）使用梯度兼容性诊断决定注意力 LoRA 共享、前馈层独立；3）基于剩余教师‑学生差距的 gap‑aware 采样动态分配训练预算；4）噪声幅度匹配与语义空间（DINOv2）监督，消除坐标不匹配。

**🔧 技术方法**

技术手段包括：heterogeneous on‑policy distillation、flow‑matching student、DINOv2 CLS 语义监督、attention LoRA 与 FFN adapter、梯度兼容性测量、gap‑aware curriculum、噪声幅度对齐、温度熵采样、warm‑start 预热。

**📊 数据集**

数据集：Pick‑a‑Pic（风格/美学）、Flow‑GRPO（GenEval 结构）和 DPG‑Bench；同时使用官方 FLUX.1‑dev 与 Z‑Image 的官方训练数据进行教师生成。

**📈 对比分析**

与基准的比较：在 DrawBench 上 Aesthetic、ImageReward、HPSv3 等指标均优于 SD3.5‑Medium 及 FLUX.1‑dev；在 GenEval 上整体准确率从 67.3 提升至 73.3，超过 FLUX.1‑dev（69.4）与 Z‑Image（65.2）；在 DPG‑Bench 上亦取得最高关系分数 85.20。该学生模型在单一 backbone 下实现两种能力，切换仅需 adapter 交换，显著降低资源与维护成本。

**⚠️ 局限性**

局限性：1）仍依赖 DINOv2 语义特征，跨域或极端风格可能受限；2）对教师模型的选择和噪声匹配需人工调优；3）在极高分辨率或大模型规模下的扩展性未验证；4）对非文本提示或多模态输入的支持有限。

---

## 106. Neighborhood-Aware Dual Biomedical Entity Linking

**arXiv ID:** 2608.04144 | [PDF](https://arxiv.org/pdf/2608.04144v1)

**作者:** Yicheng Tao `[一作]` (University of Michigan), Jie Liu `[通讯]` (University of Michigan)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PILOT框架，用邻域感知检索、双重重排序与分数融合实现生物医学实体链接；

**💡 创新点**

创新点在于同时在查询端和知识库端注入本体邻域信息，结合表面形式和上下文双视角重排序，并通过点式重排序支持深度候选池；

**🔧 技术方法**

采用生成式查询改写（Qwen3-4B-Instruct）、Rocchio式实体嵌入池化、双重重排序器（Surface‑Form + Contextual）以及单一权重融合；

**📊 数据集**

在五个公开基准上评测：NCBI、BC5CDR、COMETA、AAP、MM‑ST21pv；

**📈 对比分析**

与判别式、生成式及LLM基线对比，PILOT在四个数据集上取得最高R@1，平均提升2.1个百分点，且在推理速度上比主流LLM基线快约9‑10倍；

**⚠️ 局限性**

局限包括对“is‑a”层级的依赖、需有域内训练数据才能捕捉注释约定、仅在英文医学文本上验证，缺乏跨语言或临床真实环境的评估。

---

## 107. Even more properties of parity based bit-counting complexity classes

**arXiv ID:** 2608.04484 | [PDF](https://arxiv.org/pdf/2608.04484v1)

**作者:** Tayfun Pay `[一作]` `[通讯]` (City University of New York), Tayfun Pay (City University of New York)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了基于位计数的奇偶类 B_|0|⊕P 与 B_|1|⊕P 的额外性质，证明了它们与 C_=P、ES、MNS 的包含关系，进一步显示 PP、#P 可通过 B_|0|⊕P / B_|1|⊕P 计算，并证明了 P^PP 与这两个奇偶位计数类相等，最终把计数层级 CH 纳入这些层级。

**💡 创新点**

创新点在于利用位计数与二进制长度的 XOR 关系构造 Mersenne 检测器；提出了按位恢复与缓存优化的查询方法，使得从 B_|i|⊕P oracle 只需 2n+2 次不同查询即可恢复任意 #P 值；通过这些技巧完成了 PP、#P 与 B_|i|⊕P 的相容性证明，并展示了 P^PP 与奇偶位计数类的等价性。

**🔧 技术方法**

主要技术包括：四连续值定理、位恢复/位恢复+缓存技巧、oracle 归约与层级模拟、递归层级的归约与组合、以及对 Mersenne 数字长度差异的利用。

**📊 数据集**

无实验数据集，全部为理论证明。

**📈 对比分析**

通过理论归约与层级模拟与已知等价关系比较，未给出实验性能指标，结论基于复杂度类包含与等价的数学证明。

**⚠️ 局限性**

局限性在于方法仅适用于 Turing 归约；缺乏多项式时间实现细节；对其它复杂度类（如 PH）是否在不借助 Toda 定理下被包含仍未解决；依赖已有等价关系，若这些等价被打破则结论失效。

---

## 108. PURPOSE: Poisoning Conflict Resolution in RAG via Proxy-Fact-Grounded Updates

**arXiv ID:** 2608.04756 | [PDF](https://arxiv.org/pdf/2608.04756v1)

**作者:** Zijian Wang `[一作]` (Nanjing University), Sheng Zhong `[通讯]` (Nanjing University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的黑盒知识中毒攻击（Pivot Update RAG Poisoning），通过利用代理事实和源支持的事件来生成低冲突、可信的毒化文档，攻击 RAG 的冲突解决机制。

**💡 创新点**

创新点在于：1）将攻击目标从直接对立转为最小化冲突的更新式注入；2）通过从公开 LLM 抽取代理事实构建“pivot”事件，保证与检索证据兼容；3）采用五阶段提示式流水线实现严格黑盒攻击。

**🔧 技术方法**

技术手段包括：使用公开 LLM（如 DeepSeek‑V3.2）提取代理事实；利用提示工程生成 pivot 事件、权威来源、与查询对齐的文档；并在检索‑生成‑冲突解决流水线中注入毒文。

**📊 数据集**

使用的数据集：NQ、HotpotQA、MS‑MARCO；评测 5 大生成模型（GPT‑5.2、Gemini‑3‑Flash、Qwen3.5‑Plus、DeepSeek‑V3.2、Llama‑3.3‑70B‑Instruct）。

**📈 对比分析**

与 PoisonedRAG、AuthChain、PARADOX 三种基线对比，在 45 个实验设置中，本方法在 35 组获得最高攻击成功率，平均 ASR 提升 9.7 点；在 vanilla RAG 亦提升 4.9 点；并保持较好的流畅度（PPL≈22.6）。

**⚠️ 局限性**

局限性：需要可访问的公开 LLM 作为代理，且对专业或长尾领域覆盖不足；实验仅在标准 QA 任务和公开语料上，未验证对真实系统的可迁移性。

---

## 109. A Multi-Sensor Dataset for Monitoring the Operational Environment of Rail Vehicles

**arXiv ID:** 2608.04704 | [PDF](https://arxiv.org/pdf/2608.04704v1)

**作者:** Claudio Diotallevi `[一作]` (understandAI GmbH), Martin Köppel `[通讯]` (DB InfraGO AG)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过在轨道维护车和通勤列车上配备多种传感器（相机、红外相机、LiDAR、雷达、位置加速度传感器），收集并标注了88.2分钟、7,052,055个高质量的对象标注，构建了面向铁路环境感知的多传感器数据集。

**💡 创新点**

创新点主要包括：①首次构建覆盖七种铁路对象类别（列车、轨道、信号、缆索、路面车辆等）的大规模多传感器标注数据；②结合自动投影和人工校正的标注流程，实现每周140,000个标注的高效产出；③采用RailLabel JSON schema 与 ASAM OpenLABEL 标准对齐，提升数据兼容性；④引入多级质量控制体系，确保标注精度。

**🔧 技术方法**

技术方面采用多传感器同步采集、传感器标定、3D点云自动标注、投影至2D/IR/雷达、手工审核、自动化 QA 验证、JSON 数据格式化、并使用自研的 RailLabel 工具进行可视化与验证。

**📊 数据集**

使用的数据集是作者自行构建的铁路多传感器数据集，涵盖两辆车（GAF 轨道维护车、BR472 通勤列车）共计 88.2 分钟、7,052,055 个标注，数据类型包括 RGB、IR、LiDAR、雷达、位置加速度等；该数据集已按 RailLabel JSON schema 公开可下载（需联系 DB InfraGO AG）。

**📈 对比分析**

本文主要通过与现有公开铁路视觉数据集（如 RailSem19、FRSign、OSDaR23 等）在标注量、传感器多样性、对象类别覆盖率等方面进行比较。相比之下，该数据集在多模态传感器覆盖、标注规模和对象多样性方面具有显著优势；在标注效率方面，峰值产出每周140k 注解，远高于传统人工标注流程。

**⚠️ 局限性**

局限性包括：①数据仅覆盖两辆车在德国特定线路的采集，地理与场景多样性有限；②部分对象类别（如摩托车、轮椅等）标注量不足；③数据集尚未公开分发，需通过 DB InfraGO AG 申请获取；④未提供基准模型评估，仅关注数据质量与流程实现。

---

## 110. TriCLE: Tri-Modal Vision-Language Reasoning for Edge-Deployed Fine-Grained Clustering

**arXiv ID:** 2608.04175 | [PDF](https://arxiv.org/pdf/2608.04175v1)

**作者:** Kishor Datta Gupta `[一作]` (Clark Atlanta University), Roy George `[通讯]` (Clark Atlanta University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

TriCLE 提出了一套在边缘设备上进行飞机分类的三模态视觉语言系统，通过从单张 RGB 图像生成结构保持的 FLIR 热像和伪 LiDAR 深度投影，并将三模态视图与任务指令输入 Qwen3‑VL‑4B‑Instruct 模型，实现飞机技术分类。

**💡 创新点**

创新点包括：① 通过 ControlNet+LoRA 在单目 RGB 上合成 FLIR 热像和伪 LiDAR，实现无真实多传感器数据的三模态同步；② 采用序列级 Policy Optimization（GSPO）实现对结构化输出和工程学分类的一致性；③ 结合 4‑bit 量化与注意力记忆优化，使模型在 8GB VRAM 上实现 1.48 秒的推理。

**🔧 技术方法**

核心技术包括：控制网络+LoRA 的 FLIR 合成、DPT 深度估计与伪 LiDAR 正射投影、Qwen3‑VL‑4B‑Instruct 视觉语言模型、RPSFT 旋转保持的参数高效微调、GRPO/GSPO/DAPO 三种强化学习对齐策略，以及 4‑bit 量化与注意力缓存。

**📊 数据集**

使用 FGVC‑Aircraft 数据集提取 RGB 单机图像，随后通过合成流程生成对应的热像和伪 LiDAR；评估时使用 100 张同步三模态试验集和 DSERT‑RoLL 路面场景作为跨域零样本测试。

**📈 对比分析**

在与基线未对齐模型、RPSFT+GRPO、RPSFT+DAPO 的对比中，GSPO 在 100 张测试集上取得 78.00% 的分类准确率、0.793 的加权 F1 及 94.00% 的可解析格式率，推理延迟仅 1.48 秒，显示出显著优于其他方法的性能。

**⚠️ 局限性**

局限性在于热像和伪 LiDAR 均为合成，可能包含生成器伪影，缺乏真实同步传感器数据；数据规模有限（800 张单机图像），未覆盖多机场景和硬负例；跨域评估使用 VLM 生成的银标签，缺乏人工验证，需在真实航空多传感器流上进一步验证。

---

## 111. The LLM Proposes, the Executive Disposes: A Self-Verifying Agent Instrument that Dissociates Commitment Drift from Binding Drift in Long-Horizon Agents

**arXiv ID:** 2608.04066 | [PDF](https://arxiv.org/pdf/2608.04066v1)

**作者:** Mohsen Arjmandi `[一作]` `[通讯]` (Independent researcher), Mohsen Arjmandi (Independent researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一种自我验证的长时程代理工具，使用确定性执行层和预先登记的预测来实现结构化验证，并通过“运行有效性门”与“影子引用”保证测量的可靠性。

**💡 创新点**

创新点在于将验证嵌入代理架构本身：实现了两种可切换的修复机制（绑定修复与承诺修复），对“目标漂移”进行因果分解，发现承诺漂移导致目标放弃，而绑定漂移被结构性吸收；同时引入了影子编译器在每个消融单元中定义漂移度量。

**🔧 技术方法**

使用技术包括：确定性执行层（Differ, Matcher, Validator, Compiler, Renderer, Test‑Evaluator），闭合词汇的类型化提议，写入错误率/渲染尺寸/加盐校验的自闭合门，影子编译（shadow compilation），预登记预测与代码匹配的自我验证，附加只写日志，绑定修复（join）和承诺修复（外部存储）两种切换开关。

**📊 数据集**

主要数据集为 ARC‑AGI‑3 交互式游戏，进行了 52 轮实验，约 17M 个 token；此外使用 Haiku 类小模型验证了模型尺寸壁垒。没有使用公开的大规模游戏集，所有实验均基于内部生成的环境。

**📈 对比分析**

比较方法：在每个消融单元（binding 修复 on/off，commitment 修复 on/off）下跑三颗种子，记录 GDS‑bind、GDS‑abandon、完成数等指标。结果显示：移除 commitment 修复后，目标放弃率从 0.00 升至 1.00，而绑定错误保持 0.00；移除 binding 修复对每帧漂移无影响。整体任务完成率为 0%，但验证机制本身未产生额外推理开销。

**⚠️ 局限性**

局限性包括：任务效能为零，无法证明完整的漂移分解；消融实验仅覆盖一个游戏（缺乏对称双消融验证）；结果依赖闭合词汇，可能无法覆盖所有环境；单次运行方差大，需多次种子；绑定修复被结构性吸收后未能观察到漂移提升；模型尺寸阈值未完全验证；未涉及感知或社会层面的 grounding。

---

## 112. Kathleen Writes: Autoregressive Generation and Data Scaling Without Attention

**arXiv ID:** 2608.04678 | [PDF](https://arxiv.org/pdf/2608.04678v1)

**作者:** George Fountzoulas `[一作]` `[通讯]` (Frederick University), George Fountzoulas (Frederick University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无注意力、无词典的字节级语言模型，证明其在文本生成任务中能与传统变换器匹配或超越，并通过自定义的 FORM DISTANCE 量化生成文本质量。

**💡 创新点**

创新点包括：1）使用 wavetable byte encoder 与多尺度衰减记忆(reverb)替代注意力；2）构建非参数、可离线、抗作弊的 FORM DISTANCE 评价仪表；3）在解码时加入检索增强（phrase retrieval）显著提升文本质量；4）系统性评估模型在不同数据规模下的表现，并公开所有实验细节。

**🔧 技术方法**

核心技术包括字节级 DFT 相位旋转编码、三尺度内容门控指数衰减记忆、快速分块求和实现的快速扫描、全参数匹配的无注意力基线、以及多策略的解码与检索增强。

**📊 数据集**

主要使用 WikiText‑103（原始 UTF‑8 字节）进行预训练与评估，亦使用 enwik8 作为鲁棒性检验；在迁移学习实验中使用 SST‑2 与 IMDB 数据集；检索增强使用 WikiText‑2 句子级短语库。

**📈 对比分析**

与参数匹配的 Transformer 进行对比，发现无注意力模型在 2–512 MB 数据规模下均取得更低的 bits/byte，且在 32 MB 数据上已达 1.84 bpb；在 FORM DISTANCE 上，宽采样与检索增强将距离从 3.17 降至 1.14，几乎达到人类文本基准（1.0）。

**⚠️ 局限性**

局限性包括：仅在固定参数下研究数据缩放，未探索参数与数据共同增长的 compute‑optimal 前沿；检索增强仅对模型自身语料有效，异域语料无显著帮助；FORM DISTANCE 只评估文本表面“可读性”，不涵盖语义真相或连贯性；无注意力架构缺乏精确长程记忆能力。

---

## 113. Not All Redundant Tokens Are Alike: Analyzing Visual Token Pruning through Token Roles

**arXiv ID:** 2608.04483 | [PDF](https://arxiv.org/pdf/2608.04483v1)

**作者:** Hyeonyu Kim `[一作]` (Maum AI Inc.), Jaejin Kim `[通讯]` (Maum AI Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过EmbedLens的视觉令牌角色分析，研究并比较了FastV、DART、DivPrune三种视觉令牌剪枝方法在不同令牌预算下的角色偏好及其对下游任务性能的影响。

**💡 创新点**

创新点在于将令牌角色分布与剪枝决策关联，并提出角色保护剪枝策略，发现非活跃（dead）令牌在剪枝时仍能影响模型表现，从而提出考虑令牌角色组合的剪枝思路。

**🔧 技术方法**

主要技术包括EmbedLens嵌入空间角色分配、三种无监督剪枝算法（FastV、DART、DivPrune）、角色保护剪枝实验、注意力结构分析等。

**📊 数据集**

实验使用了十个VLM基准数据集，包括GQA、MME、MMBench、POPE、TextVQA、ScienceQA-IMG、SEED-Bench、OCRBench、VizWiz-VQA和VQAv2，模型为LLaVA‑v1.5‑7B。

**📈 对比分析**

通过在相同令牌预算（12.5%–87.5%）下对比不同剪枝方法的角色分布与性能，发现DivPrune在大多数基准上性能最好，而FastV在部分任务上表现更优；角色保护剪枝可在保持或提升性能的同时降低令牌数。

**⚠️ 局限性**

局限在于仅评估单一VLM架构和三种剪枝方法，令牌角色分配仍受EmbedLens假设限制，且未探讨不同模型规模或更高压缩比下的泛化性。

---

## 114. Zero-error expectation equals amortized query complexity

**arXiv ID:** 2608.04152 | [PDF](https://arxiv.org/pdf/2608.04152v1)

**作者:** Daiki Suruga `[一作]` `[通讯]` (University of Waterloo), Daiki Suruga (University of Waterloo)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在期望随机量子与分布式查询复杂度框架下的直接相加（direct-sum）问题，并给出了总关系函数在多实例情形下的期望查询复杂度的精确渐近定律；同时对 OR‑n∘f 形式的单边错误组合函数给出了精确的摊销定理；并通过这些结果构造了多种摊销与单实例成本之间的分离例子。

**💡 创新点**

创新点在于：
• 给出了期望随机查询复杂度的完全可度量：对于任意总关系 f 与误差 ε∈[0,1]，有 lim_{n→∞} R_ε(f^n)/n = (1−ε)·R_0(f)。
• 证明了该乘子 1−ε 是精确的、且适用于所有 ε（包括大于 1/2 的情况）。
• 在单边错误的 OR‑n∘f 组合中，首次给出了摊销成本与单个实例成本相等的精确公式 c_ε(f)。
• 通过这些定理构造了摊销与单实例成本之间无界或多项式分离的例子，解决了 Blais & Brody 2019 年提出的开放问题。

**🔧 技术方法**

使用的技术主要有：
• 并行重复（parallel repetition）与截断（truncation）技术，用于构造上界算法；
• 维度模拟与坐标嵌入（coordinate embedding）与概率论工具（如 Chebyshev、Hoeffding）用于下界证明；
• 证明零错误 Yao 最小化定理的期望成本版本，将随机化算法视为分布式决策树的混合；
• 采用极小极大（minimax）理论与多面体对偶性，得到单边错误 OR 组合的摊销定理。

**📊 数据集**

本工作为理论性论文，未使用任何实验数据集；所有结果均为严格数学证明。

**📈 对比分析**

与之前仅得到常数因子或多项式因子上界/下界的直接相加定理相比，本文提供了完整的渐近等式；对比已知的强直方图（strong direct-sum）定理，证明了在期望成本模型下存在精确乘子 (1−ε)。对 OR‑n∘f 的单边错误情况，证明摊销成本正好等于单实例成本 c_ε(f)，表明不存在额外的摊销损失。

**⚠️ 局限性**

局限与未解决的问题包括：
• 对于分布式与最坏情况的摊销极限，本文仅给出 liminf/limsup，尚不清楚是否总存在极限；
• 期望量子查询复杂度的直接相加极限仍未确定；
• 交流复杂度中期望通信复杂度的直接相加极限也未完成；
• 对于更一般的外部函数（如 MAJ_n）或双边错误的 OR 组合，缺乏类似的精确摊销表述；
• 需要进一步研究在更广泛条件下的多项式/无界分离的可能性。

---

## 115. HoRFFI: High-Openness RF Fingerprint Identification with a Similarity-Enhanced Variational Information Bottleneck

**arXiv ID:** 2608.04881 | [PDF](https://arxiv.org/pdf/2608.04881v1)

**作者:** Shuiguang Zeng `[一作]` (Hebei Normal University), Houbing Herbert Song `[通讯]` (University of Maryland, Baltimore County)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了在仅有少量训练设备类时的高开放度无线频率指纹识别问题，提出 HoRFFI 框架实现后训练设备注册、识别与未知设备拒绝。

**💡 创新点**

创新点在于引入相似性增强的变分信息瓶颈（SVIB）监督机制，通过相似性压缩、分类监督和相似性保留三项损失，使嵌入空间更具可迁移性，从而在训练类少时仍能良好识别新设备。

**🔧 技术方法**

主要技术包括：CNN 特征提取器、Spectrogram/CSI 预处理、SVIB 损失（包含相似性压缩、分类监督、相似性保留），k‑NN 识别与未知设备拒绝规则。

**📊 数据集**

使用公开的 LoRa 与 Wi‑Fi RF 指纹数据集进行实验。

**📈 对比分析**

与 ScalableRFFI、MLGPN 等框架以及 Triplet、Contrastive、N‑pair+Center、SupCon、VIB 等损失进行对比；HoRFFI 在 LoRa 和 Wi‑Fi 上相对基线提升了约 0.1–0.2 的 ACC，并在高开放度条件下保持更高识别率，同时拒绝性能不逊于或优于对照方法。

**⚠️ 局限性**

限制包括：特征提取器在训练后保持冻结，难以在部署后进一步微调；对极高开放度的适应仍有限；需要较多（约 500）样本进行注册才能达到高准确率；实验仅覆盖 LoRa 与 Wi‑Fi 两类数据集，缺乏对更广泛 RF 系统的验证。

---

## 116. Fast Thick-Thin Decomposition for Sparse Spanners on Hyperbolic Surfaces

**arXiv ID:** 2608.04585 | [PDF](https://arxiv.org/pdf/2608.04585v1)

**作者:** Sándor Kisfaludi-Bak `[一作]` (Aalto University), Geert van Wordragen `[通讯]` (Aalto University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在负曲率的超曲面与超平面上构造稀疏的（Steiner）非交叉及交叉扩展点稀疏逼近图（(1+ε)-spanner）

**💡 创新点**

首次给出在超曲面（genus g）上仅线性依赖于g的稀疏(1+ε)-spanner，并提供多项式时间的厚–薄分解算法；同时在超平面上得到与欧氏相当的非交叉Steiner spanner

**🔧 技术方法**

利用厚–薄分解、短环的棕榈结构、伪–网、k‑近邻点检索、圆形覆盖与分层构造，结合贝尔曼–基尔帕特里克层次、Klein 圆盘模型和欧氏角约束等几何工具

**📊 数据集**

无，论文为理论算法，不涉及实测数据集

**📈 对比分析**

相较于已知欧氏/超平面结果，本工作实现了线性 g 依赖的边数：在超曲面上非交叉Steiner spanner 边数 O(n/ε^{3/2}+g/ε^{2})，交叉Steiner spanner 边数 O(n/√ε+g/ε)；在超平面上得到 O(n/ε^{3/2}) 边数；并衍生出超曲面上的 EPTAS 近似 TSP

**⚠️ 局限性**

未能实现最优轻度（lightness）与最优边数对 g 的下界；对更一般平滑或多面体曲面、最短路径/厚–薄分解的多项式算法仍为开放问题

---

## 117. A Symbolic Execution Framework for Symbolic Timing Analysis of Digital Integrated Circuits

**arXiv ID:** 2608.04036 | [PDF](https://arxiv.org/pdf/2608.04036v1)

**作者:** Dennis Eigner `[一作]` (Technical University of Vienna), Ulrich Schmid `[通讯]` (Technical University of Vienna)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并实现了一个基于符号执行的时序分析框架，能够用分析式门延迟模型自动生成电路所有可能执行路径的符号时延表达式，并通过约束传播实现路径剪枝。

**💡 创新点**

创新点在于：①首次将可闭式的多输入门延迟模型融入符号执行；②设计了符号时间戳与约束传播机制，避免传统方法的 min/max 非可微性；③引入“元转移”（meta‑transition）来高效处理反馈循环；④支持目标函数（goal functions）以聚焦特定时序问题。

**🔧 技术方法**

核心技术包括：符号执行树构造、符号时间戳赋值、约束传播与 SMT/非线性求解、元转移循环压缩、目标函数驱动搜索；实现基于 Python，利用计算机代数系统（如 SageMath）计算门延迟表达式。

**📊 数据集**

实验使用 ISCAS‑85 公开基准电路 c17_slack（含简单常数延迟模型），并对不同输入队列长度进行测试。

**📈 对比分析**

与传统 DDTA（仅模拟单条执行路径）对比，本文使用约束传播时路径数显著下降，执行时间从 0.095 s 下降到 0.141 s（输入队列长度 1），对更长队列更为显著（例如长度 4 时从 0.218 s 降至 0.16 s）。显示了剪枝策略在减少搜索空间上的有效性。

**⚠️ 局限性**

局限性包括：尚未处理嵌套或分支复杂循环（只支持简单循环）；实现为实验性 Python 原型，性能与可扩展性有待改进；依赖于高质量的分析式门延迟模型，若模型不准确会影响结果。

---

## 118. Agreement Before Diversity: Verification-First Complementarity for Heterogeneous Language-Model Coordination

**arXiv ID:** 2608.04618 | [PDF](https://arxiv.org/pdf/2608.04618v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 119. Breaking the Curse ofMultilinguality inMany-to-Many Speech-to-Text Translation via a Resource-AwareMixture of Speech Encoders

**arXiv ID:** 2608.04586 | [PDF](https://arxiv.org/pdf/2608.04586v1)

**作者:** Yexing Du `[一作]` (Harbin Institute of Technology), Ming Liu `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MSRT框架，实现多语言多源多目标的端到端语音转文本翻译；

**💡 创新点**

通过资源感知的混合语音编码器MoSE，冻结高资源专家并训练中低资源专家，实现对低资源语言的专门化，同时使用五阶段课程学习减少数据需求；

**🔧 技术方法**

采用Whisper编码器、Q-Former压缩、MLP投射、MiLMMT-4B LLM以及LoRA微调，配合显式语言路由；

**📊 数据集**

使用Common Voice 24、FLEURS（约10小时/语种）进行训练，并在FLEURS与CoVoST-2上评测；

**📈 对比分析**

与Gemini-3.5-Flash-Lite、Whisper+NLLB、SeamlessM4T-V2-Large、Qwen3-Omni-30B、MCAT-27B等基线对比，MSRT-4B在45×44方向上COMET平均83.3，至少80分方向1552/1980，显著优于所有对比模型，且参数仅4B；

**⚠️ 局限性**

受限于预训练LLM的机器翻译能力，低资源语言翻译仍受限于LLM在这些语言/方向上的表现不足。

---

## 120. Faster-WAM: Efficient Inference-Time Future Conditioning for Robust World Action Models

**arXiv ID:** 2608.04404 | [PDF](https://arxiv.org/pdf/2608.04404v1)

**作者:** Weiheng Zhao `[一作]` (Huazhong University of Science and Technology), Xinggang Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研发了 Faster-WAM，一种高效的未来条件化机器人操作模型，利用未来环境表示提升机器人操控性能。

**💡 创新点**

创新点在于提出稀疏未来条件化框架，使用 SparseMoT 取代全层融合，仅在关键网络阶段进行视频-动作交互；并通过 Interval KV-Fusion 在不增加注意力复杂度的前提下聚合多层次未来表示。

**🔧 技术方法**

主要技术包括稀疏未来条件化、SparseMoT 选择性视频-动作交互模块、Interval KV-Fusion 多深度未来表示聚合、以及动作去噪的后处理流程。

**📊 数据集**

使用了 LIBERO、LIBERO-Plus（七种分布偏移）和 RoboTwin 2.0 这三个机器人操作基准数据集。

**📈 对比分析**

与 Fast-WAM 和 Joint-WAM 对比，Faster-WAM 在 LIBERO-Plus 上成功率从 49.14% 提升至 73.57%，速度比 Joint-WAM 提升 2.21 倍；同时在 LIBERO 与 RoboTwin 2.0 上实现了最先进性能，并在真实环境中表现出强大鲁棒性。

**⚠️ 局限性**

局限性包括：仍需要预训练来获得良好性能，稀疏交互可能在极端动态场景下的表达能力有限；此外，实验主要聚焦于当前三大数据集，未覆盖更广泛的多模态或跨域任务。

---

## 121. Robust and Personalized Federated Learning for Aircraft-Engine Prognostics under Benign and Adversarial Client Heterogeneity

**arXiv ID:** 2608.04045 | [PDF](https://arxiv.org/pdf/2608.04045v1)

**作者:** Chinmoy Mitra `[一作]`, M. F. Mridha `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对涡轮风扇预测任务，研究了两类客户端异质性——诚实与攻击者，并评估了联邦学习模型在这两轴下的表现。

**💡 创新点**

首次将联邦模型个性化（FedRep）与鲁棒聚合（Krum）相结合，以同时缩小预测误差差距并抑制背门攻击，同时提出了多维攻击评估框架。

**🔧 技术方法**

采用联邦学习个性化方法FedRep、FedCCFA、FedProx等，以及对抗聚合方法FedAvg、Trimmed mean、Coordinate median、Krum，并在三维攻击矩阵下进行实验。

**📊 数据集**

使用涡轮风扇退化基准（Turbofan Degradation Benchmark）进行5×4矩阵攻击实验，包含5个随机种子。

**📈 对比分析**

通过比较不同方法在benign轴上闭合本地-中心化误差比例和在adversarial轴上背门成功率，FedRep+Krum将背门成功率降至2.8%，比单独Krum低4倍，并保持70%误差闭合。

**⚠️ 局限性**

仅在涡轮风扇数据上验证，攻击场景有限；组合方法对计算与通信成本未做深入分析；对更大规模与多种攻击的适应性仍需研究。

---

## 122. The Personalization Mirage: How LLMs Fabricate User Profiles, and Why Self-Monitoring Misleads

**arXiv ID:** 2608.04570 | [PDF](https://arxiv.org/pdf/2608.04570v1)

**作者:** Yushi Sun `[一作]` (LIGHTSPEED), Rui Sheng `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了MirageBench，用于系统评估个性化大型语言模型（LLM）在生成用户相关内容时的“过度推断”现象；

**💡 创新点**

首次构建了一个以四分类（Grounded/Reasonable/Stereotype/Fabricated）为基础的过度推断度量，并揭示了跨模型的“自监测逆转”现象，即模型自我报告的过度推断率与外部评判的关系相反；

**🔧 技术方法**

利用LLM进行多任务生成、Probe/Task/Accum评估流程、独立Judge（Claude‑Opus‑4‑7）进行标签化，并通过统计分析（Cohen κ、Spearman ρ、AUROC）评估模型表现；

**📊 数据集**

使用150个人工与LLM生成的用户人物档案（每人3条显式事实），结合6个个性化任务（从礼物推荐到公寓描述），共产生143,616条判定过度推断的主张；

**📈 对比分析**

对12个跨家族模型（包括GPT‑5.5、Claude‑Opus‑4‑6、Gemini‑3‑flash等）进行评估，发现所有模型的过度推断率均在35%–49%之间，平均41.6%；模型自监测与Judge评判呈负相关（ρ≈−0.60），但在单模型内部仍能较好区分过度推断（AUROC 0.58–0.83）；

**⚠️ 局限性**

局限包括：样本量仅12个模型、任务覆盖度有限、评估仅基于单轮与短期多轮实验、未针对不同用户群体进行细粒度分析，且自监测逆转的机制尚未彻底解释。

---

## 123. Smartphone Audio Based Distress Detection

**arXiv ID:** 2608.04176 | [PDF](https://arxiv.org/pdf/2608.04176v1)

**作者:** Anil Sharma `[一作]` (Indian Institute of Information Technology Delhi), Sanjit Kaul `[通讯]` (Indian Institute of Information Technology Delhi)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研发了一款基于智能手机麦克风的 24×7 无缝人类痛苦检测与报警系统 Always Alert，利用 SVM 两阶段学习筛选尖叫/哭泣，并通过好友圈与警方联动。

**💡 创新点**

创新点包括：①全程手机端实时执行；②两阶段 SVM+上下文滤波结合时间连续性分析显著降低误报；③将好友圈作为最终过滤器，减少对警方的误报负担；④通过大量志愿者真实使用评估验证系统可行性。

**🔧 技术方法**

技术方案：MFCC 特征提取 → 第一阶段 Speech Filter（SVM）→ 第二阶段 Context Filter（SVM）→ 时间连续性分析 → 发送至好友圈，必要时上报警方；实现采用 Android AudioRecord、libsvm、CoMIRVA 计算 MFCC。

**📊 数据集**

数据集：控制实验中收集的 340 段尖叫/哭泣、580 段正常语音、8714 段室内、4513 段室外、3566 段机械、3712 段电视、1641 段聚集声音；以及 16 名志愿者录制的约 250 小时日常音频；验证集在上述数据上加入不同 SNR（无、40dB、20dB、10dB）混合。

**📈 对比分析**

评估方法：在验证集绘制检测率–误报率曲线，比较单一 SVM、上下文滤波与两阶段串联；在志愿者测试中选取 P1（阈值 -1.1，列宽 1）得到检测率≈96%，误报率≈1%（平均每 3–4 小时一次误报）；时间分析后误报进一步降至每 2.5–3 小时一次；相较于传统单一 SVM，提升 15–20% 检测率，误报率降低 10–20 倍。

**⚠️ 局限性**

局限性：误报仍导致用户频繁发帖；对音乐、电视等背景噪声难以完全过滤；无法区分正面与负面尖叫；好友圈确认导致一定延迟；未充分利用位置信息、加速度计等多模态信息。

---

## 124. Unscented KalmanNet: a hybrid deep learning filter with calibrated posterior covariance for nonlinear state estimation

**arXiv ID:** 2608.04201 | [PDF](https://arxiv.org/pdf/2608.04201v1)

**作者:** Minhyeok Ko `[一作]` (University of Texas at Tyler), Abdollah Shafieezadeh `[通讯]` (Ohio State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种新的混合滤波器——Unscented KalmanNet（UKN），在 Unscented Kalman Filter（UKF）框架内加入两个学习模块（NoiseNet 用于自适应噪声协方差，GainNet 用于校正卡尔曼增益），并通过校准意识的损失函数联合训练，从而在非线性系统中实现更准确的状态估计与更可靠的后验协方差。

**💡 创新点**

创新点包括：
- 将 KalmanNet 的增益学习结构迁移到 UKF 的 sigma‑point 递推中，保持显式的后验协方差；
- 通过 NoiseNet 对过程与测量噪声协方差进行乘法修正，保证正定性；
- 通过 GainNet 对 UKF 增益施加有界残差校正，兼顾模型不匹配的补偿；
- 采用校准感知的自适应权重机制，使状态误差、协方差一致性和创新一致性损失在训练过程中动态平衡；
- 统一框架下同时优化估计精度与不确定性校准。

**🔧 技术方法**

使用技术：
- Unscented Transform 进行 sigma‑point 传播；
- 双层递归神经网络（GRU）实现 NoiseNet 与 GainNet；
- 对协方差采用 Cholesky 分解与乘法修正，保证正定；
- 采用有界 tanh 以及残差缩放实现增益校正；
- 复合损失包括 MSE、协方差校准损失（对角线对数损失）、学生 t 创新一致性损失以及增益正则化；
- 训练过程中通过梯度下降在 UKF 递推中进行自微分，使用自适应权重更新。

**📊 数据集**

数据集：
- 三个合成系统：Lorenz 系统、Duffing 摆、协同转向目标跟踪；
- 一个真实飞行数据集：UZH‑FPV 无人机竞速数据（11 条飞行轨迹）。

**📈 对比分析**

比较方法与性能：
- 与传统 UKF、KalmanNet（KN）和 Bayesian KalmanNet（BKN）做对比；
- 评价指标包括状态均方根误差（RMSE）、平均归一化估计误差平方（ANEES/NEES）以及经验置信区间覆盖率；
- UKN 在所有四个实验中均取得最低的 RMSE，减少 22%–49%；
- ANEES/NEES 近似满足卡方分布理论，覆盖率接近 95%/99% 的名义水平；
- 与 KN 相比，UKN 除了提供更小误差外还提供可可靠的协方差；BKN 在协方差校准上不如 UKN。

**⚠️ 局限性**

局限性：
- 仅针对状态估计，校准目标仅对角线协方差；
- 对多变量完整协方差校准缺乏专门机制；
- 需要显式 sigma‑point 递推，易受高维、强非线性系统的数值稳定性影响；
- 未对未知参数或系统漂移做联合估计，未来可拓展至参数估计。

---

## 125. An Approach for Embedding-Guided Function Reuse Detection in Embedded C Software

**arXiv ID:** 2608.04137 | [PDF](https://arxiv.org/pdf/2608.04137v1)

**作者:** A A Talha Talukder `[一作]` (Trent University), Akramul Azim `[通讯]` (Ontario Tech University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种面向嵌入式C软件的领域感知检索增强生成（RAG）管道，用于检测可复用函数。

**💡 创新点**

创新点是将四个硬件兼容性验证器嵌入检索堆栈并结合无监督阈值校准，显著降低SonarQube误判。

**🔧 技术方法**

使用多模型嵌入（MiniLM、MPNet、BGE、E5、GraphCodeBERT、OpenAI text-embedding-3-small、LLaMA 3 8B、StarCoder2 3B）、Jaccard重叠、参数计数、调用图重叠、分支模式比较以及动态规则注入。

**📊 数据集**

使用六个公开嵌入式C项目（共184个函数，4,815对），覆盖三类硬件平台。

**📈 对比分析**

通过与SonarQube对比，实验显示93.6%的误判率，验证器准确率97.5%，不同模型阈值差异达1.73倍。

**⚠️ 局限性**

局限在于仅基于公开代码、规则库与阈值在不同项目或行业代码可能需重新校准，且正则提取器可能漏掉复杂函数。

---

## 126. Sublogarithmic Swap Regret in Multiplayer General-Sum Games via Hybrid Regularization

**arXiv ID:** 2608.04149 | [PDF](https://arxiv.org/pdf/2608.04149v1)

**作者:** Taira Tsuchiya `[一作]` `[通讯]` (University of Tokyo and RIKEN), Taira Tsuchiya (University of Tokyo and RIKEN)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种基于Blum–Mansour分解与OFTRL的无耦合学习动态，在固定游戏中实现每位玩家 O(nm^2√(log m log T)) 的单调交换代价，进而使时间平均产物分布达到 O(nm^2√(log m log T)/T) 的相关均衡逼近；同时给出能对抗任意效用序列的鲁棒变体。

**💡 创新点**

创新点在于：① 通过引入分量化的混合正则化（负Shannon熵控制OFTRL的乐观预测误差，log-Barrier控制转移矩阵的Bregman距离）实现对转移矩阵稳态分布的全局灵敏度控制，摆脱了传统自共形局部范数与混合参数的依赖；② 结合全局灵敏度定理与Blum–Mansour分解，得到第一级 O(nm^2√(log m log T)) 的个体交换代价，首次突破了 O(nm^{5/2} log T) 的上界。

**🔧 技术方法**

主要技术包括：Blum–Mansour外部-交换代价归约、乐观跟随正则化领导（OFTRL）与混合正则化、Markov链树定理与对站点分布的全局敏感度定理、对偶分布与Bregman散度的结合、以及对抗鲁棒切换机制。

**📊 数据集**

无使用任何公开数据集；实验验证均基于合成的多玩家一般总和博弈模拟。

**📈 对比分析**

与之前的 O(nm^{5/2} log T) 以及 O(nm^3/4 T^{1/4}) 等个体交换代价上界相比，本工作在时间维度上实现了子对数（√log T）级别的提升，且在动作维度上将幂次从 5/2 降到 2；实验中相关均衡逼近误差随 T 下降的速度显著加快。

**⚠️ 局限性**

局限性包括：① 仍保留 √log T 因子，未能完全实现纯粹的 O(log T) 或更低的时间复杂度；② 对游戏参数 n、m 的最优性尚未证明，缺乏匹配的下界；③ 仅针对全信息反馈，未扩展至半信息或带噪声的情境；④ 对自适应多样化策略的鲁棒性分析仍不完整。

---

## 127. AFD-Ledger: Deployment Provisioning for Attention--FFN Disaggregation

**arXiv ID:** 2608.04502 | [PDF](https://arxiv.org/pdf/2608.04502v1)

**作者:** Chengyu Qiu `[一作]` (Tsinghua University), Mingxing Zhang `[通讯]` (Tsinghua University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 AFD‑Ledger，一种离线分析式部署规划系统，能够在给定模型、工作负载、SLO、预算、硬件目录和运行时假设下，联合探索 Attention–FFN Disaggregation (AFD) 与传统共置部署的最佳硬件分配与组织方案，并通过局部探索加速器实现逼近全枚举最优。

**💡 创新点**

创新点在于：①将部署规划视为可搜索的两层问题，先使用轻量级硬件估计快速筛选候选，然后再完成全部署评估；②实现了对硬件目录全面搜索的“可控探索”，显著减少完整部署评估次数；③通过对比分析验证了 AFD 在不同硬件与预算场景下的实际收益，揭示了同构与异构 AFD 的真实优势与限制。

**🔧 技术方法**

使用技术包括：离线 Python 实现、基于解析执行模型的吞吐量预测、角色特定硬件估计、反馈驱动的硬件修订、完整部署枚举与批处理、以及对 LongCat 2.0 的物理验证。

**📊 数据集**

数据集涵盖两类前沿 MoE 语言模型 Qwen3‑235B‑A22B 与 DeepSeek‑V3.2，以及多种商业 GPU（H200、H100、A100、L40S 等）和模拟角色专用设备，实验覆盖不同上下文长度、TPOT SLO 与预算组合。

**📈 对比分析**

比较方法：对同一部署规范，分别为 AFD 与共置部署进行完整部署规划，计算满足 TPOT 的最大解码吞吐；结果显示在 36 个同构设置中仅 7 次 AFD 超过共置，异构 AFD 需特定硬件互补，且在合适的预算与 SLO 组合下可实现 1.8‑倍吞吐提升；验证表明 AFD‑Ledger 的预测误差低于 10%，并能在 68.8%–83.5% 的评估次数内恢复全局最优。

**⚠️ 局限性**

局限性包括：①依赖解析执行模型，忽略运行时细节如调度延迟和网络争用，导致绝对吞吐偏高；②硬件探索策略为启发式，无法保证在更大硬件目录下的最优性；③对未来硬件的假设为模拟而非实际产品，缺乏对硬件更新的实时跟踪；④对多租户容错、弹性和资源共享等方面的评估不足。

---

## 128. COMPAS: Difficulty-Aware Joint Search for Optimizing Code Generation

**arXiv ID:** 2608.04336 | [PDF](https://arxiv.org/pdf/2608.04336v1)

**作者:** Jingzhi Gong `[一作]` (University of London), Mark Harman `[通讯]` (University of London)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为COMPAS的难度感知代码生成配置优化框架，能够针对不同难度的任务自动选择模型、提示和解码设置，并在部署时无需进一步搜索即直接路由到最佳配置。

**💡 创新点**

创新点在于：①将模型、提示和解码三维度的搜索拆分为两阶段（先快速模型挑选，再联合提示-解码搜索）并引入基于LLM反思的自适应搜索；②对任务按难度（并进一步按输入类型细分）构建单独的质量-成本前沿；③在部署时使用该前沿进行即时路由，显著提升质量成本比。

**🔧 技术方法**

使用的技术包括：LLM反思（LLM-based reflection）生成提示-解码变体；低成本模型探测与加权质量成本选择；多任务自适应搜索（round-robin mini-batch + 持续记录失败/成功案例）；Pareto 前沿构建与质量成本平衡投影。

**📊 数据集**

主要实验数据集为 Codeforces-style 发布的 880 题（Release v5）训练集和 175 题（Release v6）测试集；另外对 90/90 随机划分、50 题 Verified-mini 以及两种其他模型家族（Devstral、Qwen3.5）进行了泛化评估。

**📈 对比分析**

与四种领先基线（无搜索、路由器、动态解码调优、因果规则）以及三种搜索基线比较，COMPAS 在主测试集上实现 Pass@1 52.8%（相较基线最高 45.9% 提升 6.9pp），成本从 36.57 美元降至 4.92 美元，平均每小时成本下降约 7.4×；在随机划分、不同模型家族以及仓库级任务上也保持领先或相近性能。

**⚠️ 局限性**

局限性包括：①实验仅在同一模型家族内两款 LLM 上验证；②依赖任务官方难度标签，对难度预测器的鲁棒性依赖有限；③在极大搜索预算下可能出现过拟合，需更精细的采样与评估策略。

---

## 129. FM4WiFi: Flow Matching for Multi-AP Coordination in Dense Deployments of Beyond Wi-Fi 8 Networks

**arXiv ID:** 2608.04050 | [PDF](https://arxiv.org/pdf/2608.04050v1)

**作者:** Maksymilian Wojnar `[一作]` (AGH University of Krakow), Szymon Szott `[通讯]` (AGH University of Krakow)

**通讯引用:** 883 | [OpenAlex ID](https://openalex.org/A5023053545)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一套基于生成式模型的 FM4WiFi 管理器，用于在稠密 Wi‑Fi 网络中高效生成协同空间重用（Co‑SR）配置，支持 AP 组建、STA 选择、MCS 及功率的联合优化。

**💡 创新点**

创新点在于：
- 将 Co‑SR 调度视为条件生成问题，首次采用流匹配（Flow Matching）生成器在单步推理内产生多样化、可行的配置；
- 结合 GNN 自动编码器获取网络状态的低维潜在表示；
- 引入混合密度预测器（Mixture Density Network）作为快速数字孪生，快速评估候选配置；
- 通过“top‑k”候选选择实现公平与吞吐的折中；
- 能在 30+ AP 的大规模场景下实现 sub‑second 推理，突破传统搜索、求解和 RL 的可扩展性瓶颈。

**🔧 技术方法**

使用的技术包括：
- 图神经网络（GNN）自动编码器（变分自编码器范式）用于编码网络拓扑与信道；
- 连续流匹配（Flow Matching）生成模型，训练速度场并用 Euler 方法采样；
- 混合密度网络（Mixture Density Network）作为候选评估器；
- 余弦/Transformer 注意力结构在 FM 模型中实现多时序上下文编码；
- JAX/Flax 等现代深度学习框架与 GPU 计算。

**📊 数据集**

数据集：
- 从六类参数化场景（小型、住宅、开放、密集、室内小 BSS、干扰极端）生成 138k 条 Co‑SR 配置；
- 采用四种基线（Random、H‑MAB、T‑Optimal、F‑Optimal）采样，覆盖从最优到随机的配置空间；
- 训练时同时保留随机样本用于增强自编码器多样性，生成候选集时排除低质量随机样本。

**📈 对比分析**

与基线比较：
- 在 2×2–5×6 住宅/开放场景下，FM4WiFi 在大多数规模下实现了 95–110% 的吞吐（相较于 T‑Optimal）并保持 85–95% 的公平性；
- 与 H‑MAB 对比，FM4WiFi 在 12+ AP 时速率上超越 H‑MAB 并保持可扩展性；
- 在实验测试台（6 AP、8 STA）中，FM4WiFi 在未见过的真实拓扑上平均比简单轮询/全并发策略高 10–20% 的吞吐；
- 推理时间始终 < 1 秒，即使 30+ AP 也能在 0.6–1.2 秒内完成 128 候选生成与评估。

**⚠️ 局限性**

局限性：
- 训练离线，缺乏对特定部署的在线微调；
- 以总吞吐为目标，公平性和延迟控制需进一步改进；
- 需要中心化控制器和高速后端（有线链路、GPU 加速）才能满足 sub‑second 需求；
- 生成式模型及混合密度预测器对未见配置可能产生误差，缺乏严格的性能保证；
- 目前缺乏与其他联合 MCS 选择基线的直接对比，未能完全验证在所有场景下的优势。

---

## 130. STRIVE: Probing Reasoning Limits in Graded Plausibility Generation and Evaluation

**arXiv ID:** 2608.04567 | [PDF](https://arxiv.org/pdf/2608.04567v1)

**作者:** Bhiman Kumar Baghel `[一作]` (University of Pittsburgh), Xiang Lorraine Li `[通讯]` (University of Pittsburgh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了STRIVE框架，用大语言模型自动生成并评估符合事件可行性梯度的句子刺激集；

**💡 创新点**

创新点在于将生成与评估结合、使用多级可行性与难度交叉设计、以及利用全局推理scratchpad和迭代评估反馈显著提升生成质量；

**🔧 技术方法**

技术上依赖LLM（GPT‑5.1、Claude Sonnet 4.6、Qwen系列等）、结构化prompt、全局推理scratchpad、Evaluator‑Guided Refinement（R2R/R2VR）以及多重评估标准；

**📊 数据集**

数据集为60个可视化动词（来自三份英文动词命名评估），每个动词在共享事件框架中生成四个条件句子；

**📈 对比分析**

与人工标注及多种LLM评估器对比，迭代方法在GPT‑5.1上将GOLD率从≈16%提升至≈75%，与人类标注的κ≈0.53保持无显著差距；

**⚠️ 局限性**

局限包括：视觉可辨性仅基于文本推断，可能引入职业刻板印象；仅测试英语，跨语言推广需额外验证；未进行真实图像生成与临床效度验证；

---

## 131. Skill-Use: Can LLMs Actually Use Skills in Agentic Harnesses?

**arXiv ID:** 2608.04828 | [PDF](https://arxiv.org/pdf/2608.04828v1)

**作者:** Jinyi Han `[一作]` (East China Normal University), Yanghua Xiao `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Skill-Use 评测基准，评估 LLM 代理在逐步披露环境下的技能触发、遵循与边界遵守，并用 79 条真实技能配 177 个可执行任务进行评估。

**💡 创新点**

创新点在于将技能使用拆解为 Trigger、Compliance、Boundary 三个维度，构建可观测、可验证的轨迹评分体系，并通过两种代理 harness 对比验证技能使用是否受环境影响。

**🔧 技术方法**

使用了多代理 LLM（Claude Opus、GPT‑5.5、DeepSeek‑V4‑Pro 等）、Docker 隔离沙箱、进程可追溯的轨迹记录、LLM‑as‑Judge 验证器以及三阶段构建流水线。

**📊 数据集**

数据集来源为 79 条公开仓库的真实技能文档与 177 个基于真实文件的可执行任务，涵盖软件开发、基础设施、数据科学、数据库、文档处理、安全与业务运营等九大领域。

**📈 对比分析**

对比方法：在 Claude Code 与 Codex 两种 harness 下评估 8 个 LLM，采用 Trigger、Compliance、Boundary 评分并汇总为 SU；最大 SU 为 0.613，触发与遵循分别是主要瓶颈，且模型排名随 harness 变化显著。

**⚠️ 局限性**

局限性包括：技能使用仍无法达到可靠水平，评估结果高度依赖 harness；触发与执行是独立瓶颈，缺乏足够的外部验证；仅基于公开技能，未考虑新技能生成与持续学习能力。

---

## 132. Trident : How to Break Deep Reinforcement Learning Cyber Defenses (Agentic)

**arXiv ID:** 2608.04317 | [PDF](https://arxiv.org/pdf/2608.04317v1)

**作者:** Ryozo Masukawa `[一作]` (University of California, Irvine), Mohsen Imani `[通讯]` (University of California, Irvine)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 Trident 框架，包括交互式沙盒服务器、覆盖 CAGE 4 与 CyberWheel 的 13,744 条红蓝交互轨迹数据集，以及基于 RLVR 的 Agentic LLM 红队模型。

**💡 创新点**

将红队训练改为“Code-as-Policies”上下文带宽问题，使用 RLVR 直接生成可执行攻击脚本；提供可验证奖励的动态 benchmark，并展示 7B LLM 红队平均可使 DRL 防御失效 522%。

**🔧 技术方法**

采用深度强化学习、RLVR、Group Relative Policy Optimization、LLM（Qwen、Claude Mythos）以及 Log Summarizer–Planner–Coder 结构、RESTful 沙盒、AST 校验等技术。

**📊 数据集**

利用 Trident 自建的 13,744 条红蓝交互轨迹（CAGE 4 与 CyberWheel），构成 13k+ 轨迹的大规模数据集。

**📈 对比分析**

与默认静态红队、三类蓝队（GNN、H‑MARL、MARL）以及 PPO(NN) 进行对比，Trident Agentic 在所有环境中平均提升 522%（红队效率），并在零射、CoT、GPT‑4o 等对手上也表现优异。

**⚠️ 局限性**

主要局限在于高度依赖手工域特定的 prompt 与日志解析，以及奖励设计需针对每个环境，难以实现零射迁移。

---

## 133. RESPClinBench: Benchmarking Multimodal Clinical Decision-Making and Longitudinal Disease Management in Respiratory Specialty Care

**arXiv ID:** 2608.04514 | [PDF](https://arxiv.org/pdf/2608.04514v1)

**作者:** Mouxiao Bian `[一作]` (Shanghai Artificial Intelligence Laboratory), Jie Xu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了RESPClinBench，一个基于真实病例的多模态与纵向决策评测平台，评估呼吸科临床决策与长期疾病管理；

**💡 创新点**

创新点在于将临床真实情境与多模态图像、结构化数据相结合，并结合“原子临床行动回忆”和LLM-as-Judge两级自动评估，形成多维安全与性能评价；

**🔧 技术方法**

采用了基于API的标准化推理（温度0，最大8192 token）和自动化评估框架，结合指南导向的原子动作检查与结构化rubric评分；

**📊 数据集**

使用两套去标识化的呼吸科真实数据集：AECOPD-PIM（427例COPD纵向管理）和PNBIM（196例肺结节多模态评估）；

**📈 对比分析**

通过对比七款大型语言模型的平均得分（最高71.22分）和安全风险率（如图像幻觉31.85%）评估模型表现，发现模型跨任务表现不一致，平均得分约68.6分；

**⚠️ 局限性**

局限性包括缺乏患者结局与临床行为评估、仅使用单张CT图像、评估者偏差、风险标记不全面，以及仅涵盖两类任务，需进一步扩展与人类验证。

---

## 134. Pun Intended: Multi-Agent Translation of Wordplay with Contrastive Learning and Phonetic-Semantic Embeddings

**arXiv ID:** 2608.04311 | [PDF](https://arxiv.org/pdf/2608.04311v1)

**作者:** Russell Taylor `[一作]` (Georgia Institute of Technology), Michael Sana `[通讯]` (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了三种基于大型语言模型（LLM）的跨语言文字游戏（双关）翻译方法，并在CLEF JOKER 2025英语–法语翻译任务中进行实验评估。

**💡 创新点**

创新点包括：①将对比学习与判别器相结合的基线模型；②使用音素-语义联合嵌入进行检索指导的链式思维（CoT）生成；③构建多智能体迭代评估与改进框架，强调功能等价而非字面对应。

**🔧 技术方法**

主要技术包括：大语言模型（Gemini 2.5 Pro/Flash、o4-mini、Mistral Medium）、对比判别器、音素嵌入（IPA + PanPhon）、FastText语义嵌入、BiLSTM联合检索、链式思维提示、四维评估维度（等价、质量、情感、真实性）以及多智能体反馈循环。

**📊 数据集**

使用的数据集为CLEF JOKER 2025 Task 2英语–法语双关翻译数据集（1 405条训练例、3 683条验证、376条测试）以及CLEF JOKER 2023双关定位与解释数据集（用于标注和检索）。

**📈 对比分析**

对比方法包括：基线生成+判别器、CoT+音素-语义检索、以及多智能体迭代生成。实验结果显示：多智能体系统在人工评估中获得88.09 %成功率，排名第一；CoT系统为85.71 %；基线仅为47.62 %。在BLEU与BERTScore等传统指标上，三者排名靠后，说明传统指标并不适合评估双关翻译。

**⚠️ 局限性**

局限性包括：①对大语言模型的高度依赖，导致计算成本和推理延迟；②评估主要基于人工判断，缺乏统一客观指标；③仅在英语–法语上验证，难以推广到其他语言对；④双关类型的细粒度分类仍不完善，导致检索和生成的准确性受限。

---

## 135. EvtGraph: Event-Adaptive Compression for Sparse Temporal Graph Learning in Multimodal Time Series

**arXiv ID:** 2608.04368 | [PDF](https://arxiv.org/pdf/2608.04368v1)

**作者:** Ziqian Wang `[一作]` (Tsinghua University), Jinli Suo `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现了一种基于事件的统一框架 EvtGraph，用来在显式预算约束下对多模态时序数据进行高效建模。

**💡 创新点**

创新点在于将表示学习视为容量分配问题，采用事件自适应压缩、节点预算控制和时序受限稀疏图三项共同作用，实现在信息稠密区聚焦计算、压缩率高、可解释的结构化推理。

**🔧 技术方法**

核心技术包括事件自适应压缩（EAMC）将连续时序映射为事件级标记；节点预算控制（NBC）通过重要性评分选择固定数量标记；时序受限稀疏图（T2SG）构建仅与最近时间点连接的稀疏图。

**📊 数据集**

主要使用的公开数据集有临床多模态 MIMIC‑IV + CXR、跨域多模态 TimeMMD 以及短时序人体活动识别 UCI HAR。

**📈 对比分析**

与多种基准（GRU, LSTM, TCN, Transformer, Neural CDE, Latent ODE, ToMe, DynamicViT, DiffPool, MinCutPool 等）在相同预算下对比，EvtGraph 在 MIMIC‑IV 预测任务上 Macro AUROC 超过 0.90，TimeMMD 误差最低，UCI HAR 准确率达到 94.94%，同时显著降低了计算延迟和内存占用，展现出优越的效率–准确性 Pareto 前沿。

**⚠️ 局限性**

局限性：依赖粗粒度时间分块，可能无法捕捉极短或长跨度事件；对平稳信号的优势减弱；节点预算 B 的选择需要针对任务手工调优。

---

## 136. IslamicTurathBench: A Multi-Task, Multi-Discipline Benchmark for Evaluating Large Language Models on the Islamic Scholarly Tradition (turath)

**arXiv ID:** 2608.04703 | [PDF](https://arxiv.org/pdf/2608.04703v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 137. Multi-Mode Debugging for FRP-Based Embedded Systems

**arXiv ID:** 2608.04264 | [PDF](https://arxiv.org/pdf/2608.04264v1)

**作者:** Yugo Otani `[一作]` (Institute of Science Tokyo), Takuo Watanabe `[通讯]` (Institute of Science Tokyo)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了面向Emfrp嵌入式系统的多模式调试框架Emdb，解决了FRP源代码与生成C代码之间的抽象鸿沟。

**💡 创新点**

创新点在于将Emfrp源级调试与传统C层调试集成，提供节点级、子表达式级的单步、监视点以及依赖图可视化，并通过JSON源映射重建Emfrp执行。

**🔧 技术方法**

技术包括Emfrp编译器改造生成JSON源映射、在Python后端实现对GDB/LLDB的接口、VSCode前端调试视图、以及输入轨迹驱动的重放执行。

**📊 数据集**

案例数据集为ESP32单片机的双击LED控制程序（DoubleClick模块），通过实际设备收集输入轨迹进行调试。

**📈 对比分析**

通过在ESP32上重放记录的输入并与传统GDB对比，展示了在定位时序依赖错误时调试效率提升，实验表明Emdb在重放、断点设置和状态可视化等方面显著快于纯C层调试。

**⚠️ 局限性**

局限性包括实验仅覆盖ESP32平台，缺乏系统的定量效率评估，对更广泛嵌入式目标及其它FRP语言的适用性尚待验证，并且源映射技术需针对每种语言单独设计。

---

## 138. SoK: How Frontier AI Reshapes System-Level Security Risk Dynamics in Critical Infrastructure

**arXiv ID:** 2608.04033 | [PDF](https://arxiv.org/pdf/2608.04033v1)

**作者:** Chandra Thapa `[一作]` (CSIRO), Tooba Aamir `[通讯]` (CSIRO)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了前沿人工智能（FAI）在关键基础设施（CI）中的应用，并构建了一个以风险动态为核心的五维框架，系统性描述FAI如何在生命周期内重塑CI的安全风险。

**💡 创新点**

创新点在于：①提出了与传统攻击类型或生命周期阶段不同的“风险动态”视角；②将风险划分为能力出现、渗透路径、跨系统传播、控制权丧失与响应能力五大维度；③识别了科研与CI实践之间的匹配缺口，并提出了针对部署的五项保证标准。

**🔧 技术方法**

采用了结构化文献综述与定性编码技术：通过搜索关键词、灰色文献筛选、子问题探测、归纳分析等方法构建并编码研究材料，形成五维框架。

**📊 数据集**

使用的“数据集”为通过IEEE Xplore、ACM Digital Library、arXiv、ScienceDirect、行业白皮书、政府报告等渠道检索的约226篇文献（覆盖2015–2026年）。未涉及实验数据集。

**📈 对比分析**

本工作为综述性研究，没有实验对比；方法评估基于对文献中的证据进行主题归纳与交叉验证，未给出可度量的性能指标。

**⚠️ 局限性**

局限性包括：①文献检索范围受限，部分领域或最新研究可能缺失；②定性编码存在主观性；③跨系统传播（RD3）的实证证据相对稀缺；③快速演化的FAI生态可能使框架需要持续更新；④缺乏真实事故案例验证框架有效性。

---

## 139. EndoVLM: An Endoscopy Vision-Language Pre-training Model via Anatomy-Guided Sparsity and Progressive Alignment

**arXiv ID:** 2608.04472 | [PDF](https://arxiv.org/pdf/2608.04472v1)

**作者:** Zhenyu Yi `[一作]` (DAMO Academy, Alibaba Group), Yingda Xia `[通讯]` (DAMO Academy, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建并预训练了一个针对胃肠内镜的跨模态基础模型 EndoVLM，用于将无序的图像集合与结构化的临床报告进行对齐，并在多项下游任务中实现强性能。

**💡 创新点**

提出了三项创新：1）基于解剖标签的稀疏池化（AGSP）从冗余图像中提取有意义帧；2）进化语义对齐（PSAA）通过分层软目标实现从全局到局部的跨模态对齐；3）语义聚焦的掩码自编码器（SC‑MAE）在筛选帧上进行像素级重建以提升语义表达。

**🔧 技术方法**

结合 ViT‑B/16 视觉编码器、PubMedBERT 文本编码器、Qwen3 文本解析、InfoNCE 对齐、稀疏注意力、软目标交叉熵、MAE 重建等技术实现模型。

**📊 数据集**

在由两家医院收集的 348,000 例内镜检查（共 18.6M 图像）上进行预训练，排除了不合规案例，构成大规模的无标注数据集。

**📈 对比分析**

与通用视觉/跨模态基础模型（如 DINOv2/3、MAE、CLIP、BiomedCLIP）以及专用内镜视觉模型（如 EndoFM、EndoMamba）和任务专用分割模型进行对比；在 PolypDiag、CVC‑12k、Kvasir、ClinicDB、ColonDB、ETIS、LIMUC 等多任务中，EndoVLM 实现了更高的 F1、AUC、Dice 分数，并在零样本解剖识别和病变诊断中接近 100% AUC，显示出卓越的泛化能力。

**⚠️ 局限性**

局限性包括：对某些解剖部位（如回肠）的识别仍不佳；依赖大量标注良好的文本报告；在极少见或异质化的图像上表现尚未充分验证；以及模型仍需进一步优化以处理更复杂的时序信息。

---

## 140. Tail-Calibrated Soft-Output GRAND for Finite-Memory Noise-Effect Posteriors

**arXiv ID:** 2608.04068 | [PDF](https://arxiv.org/pdf/2608.04068v1)

**作者:** Behrooz Razeghi `[一作]` (Harvard University), Behrooz Razeghi `[通讯]` (Harvard University)

**通讯引用:** 480 | [OpenAlex ID](https://openalex.org/A5034055082)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对有限记忆噪声效应后验的尾校准软输出GRAND（Tail‑Calibrated SOGRAND），通过能量排序枚举噪声并利用随机码本占据率估计未查询代码字的缺失概率，进一步生成软输出置信度和LLR。

**💡 创新点**

创新点在于：1）将有限记忆噪声后验能量模型与GRAND噪声枚举结合；2）利用随机码本占据率将环境后验尾质量校准为代码字限定的后验分母；3）提供闭式公式估计缺失列表概率、块级与位级APP及其LLR；4）给出全块ML性、尾弃用保证与缺失列表估计的理论分析。

**🔧 技术方法**

使用技术包括：有限记忆（马尔可夫）后验能量表示、前向递归计算归一化常数与后验质量、最小化后验能量的枚举（K‑shortest路径/波前搜索）、随机码本占据率估计、尾校准停止规则、以及比特级前向后向求和得到LLR。

**📊 数据集**

实验数据集：随机线性码（RLC）[128,116]与[28,18]、Gauss–Markov 1D相关噪声（ρ=0.5）、硬判决二进制马尔可夫噪声与Gilbert–Elliott噪声；比较基准包括内插ORBGRAND、ORBGRAND‑AI、ExactBlockProduct、随机交织ORBGRAND、Hamming‑weight GRAND与SOGRAND。

**📈 对比分析**

与基准对比：Tail‑Calibrated SOGRAND在Gauss–Markov RLC上在相同SNR下实现更低的BLER和更少的代码字成员查询；在硬判决马尔可夫/吉尔伯特–艾利奥特噪声下也表现出显著的BLER和查询量优势。尾校准停止规则可根据误差容忍度（η）调节查询量，且缺失列表估计的校准误差极低。

**⚠️ 局限性**

局限性包括：1）缺失列表估计依赖随机码本占据率近似，对结构化码（如线性码）可能不严谨；2）枚举与后验计算在高SNR或大块大小时成本升高；3）需预先估计后验能量参数，模型不匹配会显著影响性能；4）未针对多码字/迭代解码器的硬件实现给出完整复杂度分析。

---

## 141. Relational Response Fields: A General Theory of Black-Box LLM Response Consistency and Recovery

**arXiv ID:** 2608.04552 | [PDF](https://arxiv.org/pdf/2608.04552v1)

**作者:** Song Zichen `[一作]` `[通讯]` (Sungkyunkwan University), Song Zichen (Sungkyunkwan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了“关联响应场”（Relational Response Field, RRF）框架，将黑盒大型语言模型的多次查询与已知的语义关系、锚点（anchors）统一为一个逆问题，并引入了内在难度指标γ_k(D,A)来量化在k个节点误差下的可恢复性。

**💡 创新点**

创新点在于：
1) 通过RRF把多样化查询、元模型一致性、执行验证等统一成一个线性关系-锚点算子；
2) 定义并证明γ_k(D,A)为k点误差恢复的必要且充分条件、最优噪声放大率和匹配的两点下界；
3) 明确一致性-真值分离、锚点相位转移、冗余饱和和跨模型/跨任务难度预测等结构性理论预测；
4) 提供理想、凸、离散三种修复算法，并给出相应的理论保证。

**🔧 技术方法**

主要技术包括：线性代数（最小奇异值、组稀疏恢复）、图信号处理、集合子空间（组spark）、凸优化（组ℓ1正则化）、近端梯度（group soft‑thresholding）以及对非线性传输的局部雅可比分析。

**📊 数据集**

使用的数据集包括：
- 合成实验（精确已知z*, B, 支持与γ_k），
- 真实模型日志：Qwen2.5-0.5B/7B/14B 和 Phi‑3‑mini 在数学（128个整数表达式）和代码（64个整数函数）任务上的四种变换查询；
- 真实错误重放实验，将真实模型错误插入k=1的场中，控制关系–锚点设计。

**📈 对比分析**

比较方法：
- 在合成实验中验证γ_k对一致性错误的预测；
- 在真实日志中计算γ_k并与重构误差、AUC、Spearman ρ 等指标关联；
- 通过对10种关系–锚点设计的重放实验，统计γ_k与恢复误差的相关性；
- 结果显示γ_k与恢复误差呈显著负相关（Spearman≈0.4–0.5），在控制模型、任务、关系数与锚点数后，γ_k提升可显著提高AUC（≈0.02）并提供排名预测能力。

**⚠️ 局限性**

局限性：
- 仅适用于有限维线性或局部线性传输；
- 需要可靠的解析器与已知的typed传输；
- 假设错误为稀疏节点误差，难以覆盖全局系统性偏差；
- 计算γ_k为组合式，规模大时需近似或下界；
- 实验规模有限，主要聚焦数学和代码两个领域，尚未验证在更大语言模型或更复杂任务上的泛化。

---

## 142. Large-Scale Analysis of Discussions by CS Educators Across the Stack Exchange Network

**arXiv ID:** 2608.04352 | [PDF](https://arxiv.org/pdf/2608.04352v1)

**作者:** Farhad Hossain `[一作]` (Trent University), Omar Alam `[通讯]` (Trent University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对79,854,463条帖子中12,949名CS教育者在Stack Exchange网络的投稿进行主题建模，提炼出55个主题并按IT与Non-IT两大类及其子类别进行层级归纳。

**💡 创新点**

创新点在于首次横跨整个Stack Exchange网络而非单一站点，系统捕捉CS教育者的技术与非技术讨论，并对主题随时间的演变进行定量分析。

**🔧 技术方法**

采用MALLET实现的LDA主题模型，并结合Arun等人的一致性指标确定最佳主题数；随后使用手工标注与开放式卡片排序对主题进行命名与归类。

**📊 数据集**

使用Stack Exchange公开数据集（约120 GB）与API结合，筛选出所有由CS教育者撰写且包含已接受答案的748,084条条目。

**📈 对比分析**

通过计算每月绝对影响力和相对影响力两种指标，对IT与Non-IT类别及其子类的活跃度进行对比，结果显示IT主题始终占主导，但Non-IT主题自2013年起呈持续上升趋势。

**⚠️ 局限性**

主要局限包括仅分析已接受答案而忽略未接受答案，手工标签可能引入主观偏差，且数据截至2024年，无法反映2025年及以后可能的新兴趋势。

---

## 143. A note on vector trifferent codes over the Sphere

**arXiv ID:** 2608.04256 | [PDF](https://arxiv.org/pdf/2608.04256v1)

**作者:** Stefano Della Fiore `[一作]` `[通讯]` (University of Brescia), Stefano Della Fiore (University of Brescia)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

本文改进了向量三重编码的上界，证明了对于任意的向量三重集C，|C|≤ (1+o(1))(3/2)^n。

**💡 创新点**

创新点在于引入了一种局部打包不等式，通过对每个码字周围的正交向量进行双色着色，替代了全局张量空间界限，从而精确地得到了缺失的常数因子。

**🔧 技术方法**

使用了局部打包不等式和正核估计等技术。

**📊 数据集**

论文中没有具体提到使用的数据集。

**📈 对比分析**

与Bhandari和Khetan的结果相比，本文的结果在常数因子上有显著改进，具体上界从√(2)提高到1。

**⚠️ 局限性**

限制在于该方法未能提供进一步降低常数的途径，改进常数低于1需要超出当前方法的信息。

---

## 144. FinPerMA: A Theory-Informed, Event-Grounded Personalized-Memory Benchmark for LLM Agents

**arXiv ID:** 2608.04095 | [PDF](https://arxiv.org/pdf/2608.04095v1)

**作者:** Ben Wang `[一作]` (Alibaba Cloud Computing), Chi Zhang `[通讯]` (Alibaba Cloud Computing)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于事件驱动的个性化记忆基准v8gold，并在金融领域对多种LLM与记忆系统进行评估。

**💡 创新点**

创新点包括：1）利用确定性规则的Impact Model生成可审计的用户偏好更新；2）设计Post‑Shock检查点专门测量事件后模型更新；3）冻结数据集实现可复现对比。

**🔧 技术方法**

技术手段包括：规则驱动的Impact Model（三层架构）、受约束的LLM叙事、自动验证器、检索式记忆（BM25、BGE‑M3）、结构化记忆（Mem0、MemOS、Memobase）以及七大前沿LLM骨干。

**📊 数据集**

使用数据集：合成的人格档案（基于公开投资者调查）、97条2020‑2026年的宏观、行业与个人事件、276个角色的对话与2,994道评测题（包含多选与开放式）。

**📈 对比分析**

比较方法：以无记忆基线与完整上下文为对照，衡量总准确率、MCQ准确率、开放式答案正确率、PAS、BIA、MemFid等指标；检索系统在相同骨干下可恢复约88%性能缺口，且仅需约1.4k上下文token；结构化记忆虽保留事实但缺失偏好信号，整体最高准确率仍仅约47%。

**⚠️ 局限性**

局限性：使用合成角色与规则生成的事件，历史长度仅5–8个事件；仅覆盖金融领域；对比使用单一种子与不匹配的token预算；缺乏真实投资者交互验证。

---

## 145. LoginTrap: Uncovering Task-Agnostic Phishing-Style Indirect Prompt Injection Attacks against LLM-based Web Agents

**arXiv ID:** 2608.04741 | [PDF](https://arxiv.org/pdf/2608.04741v1)

**作者:** Longtao Guo `[一作]` (Tongji University), Yang Shi `[通讯]` (Tongji University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种任务无关的登录诱导攻击（LoginTrap），能让LLM驱动的网页代理在未获授权的情况下进入并完成登录流程，从而泄露敏感信息。

**💡 创新点**

创新点在于：①首次从攻击者控制的网页上下文出发，设计任务无关的登录诱导策略；②采用基于fuzzing的注入生成方法，使注入语句与页面内容紧密匹配；③构建可复现的仿冒登录流程，完整覆盖从诱导到信息泄露的全链路。

**🔧 技术方法**

主要技术包括：LLM（GPT‑4o 等）与浏览器自动化框架、DOM/可视化信息提取、指令注入与模糊化生成、登录流程模拟、攻击评估指标（LER/ASR/PER）和多模型多架构实验。

**📊 数据集**

使用 Mind2Web 基准中的 80 个克隆网页（共 1,175 条任务），覆盖旅行、服务、信息、购物、娱乐等五大域。

**📈 对比分析**

通过在不同 LLM 后端（GPT‑4o、Gemini‑3 Flash、Claude‑Sonnet‑4、DeepSeek‑V3.2）以及三种主流网页代理架构（Browser‑Use、LiteWebAgent、Skyvern）下对比 LER、ASR 和 PER，平均 ASR 达到 86%，LER 93%，PER 100%，表明攻击在多模型、多架构下均保持高度有效。

**⚠️ 局限性**

局限性：实验仅覆盖 80 个可克隆网页，未涵盖更复杂或动态渲染页面；攻击依赖可被克隆的网页上下文；现有防御能降低成功率但无法彻底阻止泄露；未评估对更高级的任务指令、代理内存或安全策略的影响。

---

## 146. Visualizing Graph-to-Answer Mechanism Recovery in Materials-Science Hypothesis Generation

**arXiv ID:** 2608.04170 | [PDF](https://arxiv.org/pdf/2608.04170v1)

**作者:** Shashwat Sourav `[一作]` (Washington University in St. Louis), Tirthankar Ghosal `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建可视化诊断工作流，评估图结构化科学推理模型在材料科学假设生成中的机制恢复情况。

**💡 创新点**

首次将可视化层/token恢复热图与图结构化推理、机制‑F1、激活补丁等技术结合，揭示机制信息在模型层面何时何地被恢复，而非仅靠表面语义相似性。

**🔧 技术方法**

图结构化推理模型 Graph-PRefLexOR‑8B、Qwen3‑8B 基线、图结构损坏、机制‑F1 计算、激活补丁、残差流层级热图。

**📊 数据集**

公开的 100 条材料科学问题集，包含跨域映射、因果多尺度推理等开放式推理任务。

**📈 对比分析**

通过图结构损坏实验与激活补丁，比较干扰前后机制‑F1 的提升；发现第30‑36层恢复效果显著，优于对照层，说明机制信息在后期合成/答案起始层更易恢复。

**⚠️ 局限性**

仅针对单一模型、单一领域且样本量有限；机制‑F1 取值受抽取器精度限制；热图为静态视图，缺乏交互评估；结果未必推广到其他 LLM 或更大基准。

---

## 147. Training-Free Hashing-Based Attention via Binary Principal Components

**arXiv ID:** 2608.04405 | [PDF](https://arxiv.org/pdf/2608.04405v1)

**作者:** Daohai Yu `[一作]` (Xiamen University), Rongrong Ji `[通讯]` (Xiamen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无训练、基于二进制主成分的稀疏注意力机制 BinaryPC，用于高效处理长上下文 LLM 的解码。

**💡 创新点**

创新点在于通过计算键向量的二进制主成分，构造数据感知的 64 位哈希码，并引入误差感知安全机制，以无梯度训练的方式实现高精度、低成本的稀疏检索。

**🔧 技术方法**

采用了二进制主成分分析、异构哈希投影、位运算加速、离线投影校准与误差感知安全（EAS）等技术。

**📊 数据集**

在多模型（Llama‑3、Mistral、Qwen2.5 等）上使用了 LM‑Eval‑Harness、LongBench、InfiniteBench、LongBench v2、RULER、NIAH 等短、中、长上下文基准。

**📈 对比分析**

与静态、动态及哈希稀疏注意力（MagicPIG、Spotlight、Quest 等）相比，BinaryPC 在保持或接近全注意力精度的同时，在 GPU 上实现 3.56× 的解码吞吐提升，且 64 位哈希码足以匹配或超过 128 位/1000 位的哈希方案。

**⚠️ 局限性**

局限在于对极端稀疏场景下的误差控制仍需更精细的 EAS 调参，且在多域校准时需要一定量的样本以保证投影质量，未对极长序列（>1M）进行充分验证。

---

## 148. Feasibility of Embedded Photoplethysmography Sensing in Short-Duration Tactile Interactions With Pocket-Sized Robots Using IMU- and Confidence-Based Filtering

**arXiv ID:** 2608.04242 | [PDF](https://arxiv.org/pdf/2608.04242v1)

**作者:** Turjja Datta `[一作]` (IT-University of Copenhagen), Morten Roed Frederiksen `[通讯]` (IT-University of Copenhagen)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

在口袋式陪伴机器人 AffectaPocket 上嵌入光学光血容积脉搏（PPG）传感器，结合 IMU 信号进行运动抑制和置信度平滑，验证其在短时触觉交互期间监测心率的可行性。

**💡 创新点**

创新点在于首次将 PPG 传感器直接集成到手持机器人中，并提出基于 IMU 方差的运动拒绝与后期恢复窗口置信度滤波的两阶段算法，显著提升了动态环境下的测量精度。

**🔧 技术方法**

技术实现包括 MAXREFDES117 PPG 模块、ESP32 微控制器、三轴加速度计、滚动标准差运动检测、置信度窗口中值滤波、TOST 等效性检验、误差与相关性评估。

**📊 数据集**

使用 26 名实验参与者（约 17,135 条有效心率样本）进行 2 分钟的手持实验，记录 AffectaPocket 与 Polar Verity Sense 手腕式参考传感器的同步心率数据。

**📈 对比分析**

与参考传感器对比，滤波后 MAPE 从 29.14% 降至 22.72%，RMSE 由 32.43 BPM 降至 27.80 BPM，相关系数提升至 0.253，并在 ±5 BPM 范围内通过 TOST 显示统计等效；但单次测量分类准确率仅 48%。

**⚠️ 局限性**

局限包括：滤波阶段导致约 42% 的时间覆盖率损失、对瞬时心率识别精度不足、仅在成人样本上验证、缺乏多传感器冗余，难以实现实时闭环反馈，需要进一步硬件改进。

---

## 149. Generative Optimization for Incentivized Advertising with Global Level Constraints

**arXiv ID:** 2608.04421 | [PDF](https://arxiv.org/pdf/2608.04421v1)

**作者:** Gege Chen `[一作]` (University of Electronic Science and Technology of China), Xialong Liu `[通讯]` (Kuaishou Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了 GOAL，一个基于生成模型的激励广告分配框架，能够在全局 ROI 约束下生成连续激励额度。

**💡 创新点**

创新点：将激励额度量化为可生成的 token 序列，采用层次因果编码器捕捉短期与长期用户行为；引入约束感知 MoE 解码器和基于 Lagrange 乘子分布的 Safe Constrained Policy Optimization，使单一模型能在不同 ROI 阈值下自适应。

**🔧 技术方法**

使用技术包括：生成式序列建模、因果卷积 + 自注意力的层次编码器、Mixture‑of‑Experts（MoE）解码器、λ‑generalization 的 SCPO 强化学习框架、tokenizer 与加权 next‑token 损失。

**📊 数据集**

数据集：真实工业数据集 IA（短视频平台激励广告，日活 13 万用户、1.84M 事件）和基于 IA 构建的 Synthetic IA 仿真环境。

**📈 对比分析**

对比方法：DT、CDT、IQL、CAL、TREBI 等基线；GOAL 在收入（REV）提升约 18.5%、ROI 提升约 4.0%，RVR 降低约 18.4%；线上 A/B 测试中 ROI 提升 2.18%、收入提升 2.56%，均显著优于对手。

**⚠️ 局限性**

局限性：依赖离线数据，难以即时捕捉动态疲劳；token 化对极端高值仍存在误差；模型规模大，训练成本高；多目标协同优化尚未覆盖。

---

## 150. Perception Before Reasoning: Dynamic Latent Reasoning for Video Understanding and Question Answering

**arXiv ID:** 2608.04124 | [PDF](https://arxiv.org/pdf/2608.04124v1)

**作者:** Haotian Xia `[一作]` (Rice University), Hanjie Chen `[通讯]` (Rice University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5130355451)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种动态潜在推理框架（Dynamic Latent Reasoning），先通过感知潜在在视频中定位查询相关的视觉证据，然后根据需要决定是否插入推理潜在再给出答案，从而避免生成冗长的链式推理文本并提升推理效率。

**💡 创新点**

将视觉证据与推理步骤分别编码为连续潜在状态，并通过视觉目标监督、理由到潜在的自蒸馏以及强化学习学习何时调用推理，生成的答案仅包含少量可见文本。

**🔧 技术方法**

使用视觉感知潜在与推理潜在的潜在序列解码；对象级视觉目标监督感知潜在；理由蒸馏将文本推理信息迁移到潜在状态；基于可验证奖励的GRPO强化学习调优决策与答案。

**📊 数据集**

训练数据来源于Video‑R1‑CoT、LongVideo‑Reason、CG‑Bench、Video‑Holmes、MLVU等多源集合；测试覆盖Video‑MME、LVBench、LongVideoBench、MVBench、LongVideo‑Reason、TempCompass、Video‑TT、Video‑Holmes、MMVU九个视频问答基准。

**📈 对比分析**

与同一backbone的CoT、Thinking、VideoR1、Open‑o3‑Video等模型对比，Qwen3‑VL‑4B 上平均准确率提升4.2点，生成可见token仅18.5；Qwen2.5‑VL‑7B 上提升约1.5点，token18.2；InternVL3.5‑4B 提升2.7点，token13.4；LLaVA‑OneVision‑7B 提升4.5点，token17.8。

**⚠️ 局限性**

潜在状态不可读，缺乏逐步文本解释；实验仅在四种公开M‑LLM骨干和固定帧采样下验证，未检验更广泛架构或自适应帧选取的迁移性。

---

## 151. Diagnosing Tool-Selection Reasoning in LLM Agents with Canary Tools

**arXiv ID:** 2608.04719 | [PDF](https://arxiv.org/pdf/2608.04719v1)

**作者:** Atul Anand `[一作]`, Sourav Chattaraj `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估“canary工具”来诊断大型语言模型代理在工具选择时的认知弱点。

**💡 创新点**

提出六类专门化的canary工具以及能力分层的诊断框架，将单一错误判定转化为多维诊断。

**🔧 技术方法**

构建可生成canary的工具生成器、模拟真实MCP工具环境、使用LLM判定者和陷阱检测器，对八个模型进行大规模任务评估。

**📊 数据集**

120个单一作者的模板化任务（40易、40中、40难），结合12个真实工具与可生成的canary工具。

**📈 对比分析**

通过每任务canary易感率、任务成功率、恢复率等指标，发现能力越高模型易感率降低约36倍，层级并非安全保证；前沿模型在大多数陷阱下表现最佳。

**⚠️ 局限性**

工具输出为合成数据、任务数量有限、只覆盖8个模型且模型分层与供应商混杂，缺少更广泛的开源模型与更真实事实验证。

---

## 152. "Allow" to Achieve, Over-Privileged Inadvertently: The Unintended Cost of Task-Completion-Driven Pop-up Decisions in Mobile GUI Agents

**arXiv ID:** 2608.04755 | [PDF](https://arxiv.org/pdf/2608.04755v1)

**作者:** Dongsheng Chen `[一作]` (Southern University of Science and Technology), Xuetao Wei `[通讯]` (Southern University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性评估移动 GUI 代理在遇到系统权限对话框时的决策能力（Permission Literacy），构建四层权限分类框架并在 AndroidLab 环境下注入权限弹窗，进行专家验证与受控干预实验；

**💡 创新点**

提出首个多层权限分类与弹窗注入评估框架，揭示请求者身份与任务上下文对权限授权的显著影响，并评估多种提示（prompt）缓解策略的有效性；

**🔧 技术方法**

使用多模态大语言模型（Doubao、Gemini、GPT、Qwen）、AndroidLab 双模态输入、合成权限弹窗注入、统计检验（Fisher、Wilson CI）和提示工程技术；

**📊 数据集**

基于 AndroidLab benchmark 的 67 个真实任务，涵盖 Calendar、Contacts、Clock、PiMusic、Zoom 等 5 个应用，24 个四级权限场景；

**📈 对比分析**

通过对比各模型在不同提示下的授权率（Grant Rate）进行统计，发现模型间差异显著：Gemini、Qwen 在 L2–L4 级别授权率高，提示可降低高风险授权但偶尔削弱合法授权；总体校准不均匀；

**⚠️ 局限性**

仅使用单一代理框架、合成弹窗、有限的应用与权限组合，缺乏真实系统触发场景；受限的任务与提示评估范围，专家标签未覆盖多样化用户偏好；结果泛化至其他平台与长期使用场景需进一步验证。

---

## 153. Free-Lunch Augmentation by Revisiting Diffusion-Based Data Generation for Cross-Domain Few-Shot Object Detection

**arXiv ID:** 2608.04394 | [PDF](https://arxiv.org/pdf/2608.04394v1)

**作者:** Zijian Zhuang `[一作]` (Huazhong University of Science and Technology), Ruixuan Li `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对跨域少样本目标检测任务中使用扩散模型进行数据增强，提出 Selective Inpainting with Tailored Noise (SITN) 方法，通过弱噪声控制与背景 Inpainting 动态合成高质量样本。

**💡 创新点**

创新点在于：① 将视觉差距与语义差距分离并分别对待；② 采用弱噪声控制使扩散模型保持目标域信息；③ 使用背景 Inpainting 缓解语义差距；④ 设计选择模块对生成样本进行质量筛选，实现动态策略切换。

**🔧 技术方法**

主要技术包括：扩散模型（DDIM/Stable‑Diffusion）、LLM 生成文本提示、弱噪声控制、前景/背景 Inpainting、IoU 选择模块、传统的 RPN 检索与交叉注意力机制。

**📊 数据集**

使用跨域少样本目标检测基准六个数据集（ArTaxOr、DIOR、UODD、Clipart1k、DeepFish、NEU-DET），并将方法迁移至跨域少样本分割（CDFSS）四个数据集（PerSAM、RePRI、HSNet、LoEC 等）。

**📈 对比分析**

与 CDMM‑FSOD、ViTDeT‑FT、DE‑ViT‑FT、CD‑ViTO、GLIP、ETS、GroundDINO、DomainRAG 等最新基线以及 Diffmix、SDEdit、ControlNet 等扩散增广方法对比，SITN 在 1‑shot、5‑shot、10‑shot 场景下均实现 mAP 超过对照组，且生成时间从原始 DDPM 的 460 s 降至 0.02–1.5 s，表现出显著的性能提升与效率优势。

**⚠️ 局限性**

局限性包括：① 仍依赖人工框标注与 LLM 生成的提示，前景合成效果不如背景；② 对极端域差（如医疗 X‑ray）仍可能产生低质量样本；③ 需要进一步验证在更大规模、不同任务上的泛化；④ 选择模块需手动调参以适配不同数据集。

---

## 154. COSMO: Consensus-Driven Shift Modulation for Source-Free Domain Adaptation

**arXiv ID:** 2608.04604 | [PDF](https://arxiv.org/pdf/2608.04604v1)

**作者:** Bo Li `[一作]` (Sun Yat-sen University), Jianhuang Lai `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于共识驱动的源无关域自适应框架 COSMO，利用预训练视觉‑语言模型与目标模型的共建共识来完成无源数据的域迁移。

**💡 创新点**

创新点在于将源模型与 VLM 的交互转化为样本级可靠性分配，采用熵条件反 KL barycenter 形成初始共识，并通过动态再聚合与共识位移调制（CSM）在保持源证据的同时逐步吸收 VLM 信息。

**🔧 技术方法**

技术上结合了熵条件加权的逆 KL barycenter（等价于加权几何平均），centered‑logit 表示，动态共识再聚合循环，CSM 约束位移，IIC 互信息对齐损失以及 CLIP VLM 的提示优化。

**📊 数据集**

在四大公开基准上进行实验：Office‑31、Office‑Home、VisDA‑C 以及 DomainNet‑126。

**📈 对比分析**

与现有 SFDA 与 VLM 引导方法（如 DAMP、VSFOT、ProDe 等）在相同 VLM backbone 下进行对照，COSMO 在 Office‑Home（ViT‑B/32）上达 94.6% 最高分，在 VisDA‑C、DomainNet‑126 上也稳步领先，显著提升了无源域适应的性能。

**⚠️ 局限性**

局限性包括：对预训练 VLM 的依赖较强，无法直接扩展到未预训练的或小规模模型；需要手动调节 CSM 的 λ 参数，且在极端域偏移场景下仍可能出现源证据遗忘。

---

## 155. A Multimodal Automatic Redteaming Evaluation based on Atomic Jailbreak Strategy Decoupling and Combination

**arXiv ID:** 2608.04034 | [PDF](https://arxiv.org/pdf/2608.04034v1)

**作者:** Shiji Zhao `[一作]` (ByteDance), Xun Chen `[通讯]` (ByteDance)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了基于层级原子组合的多模态 jailbreak 框架 HACA，自动规划并生成跨模态攻击指令。

**💡 创新点**

将多模态 jailbreak 策略细分为结构、语义、句法三层，并构建原子策略集合；设计跨模态联合规划器和统一生成执行器，实现全流程自动化。

**🔧 技术方法**

利用大语言模型进行策略规划与文本生成，使用文本到图像模型生成图像指令；通过离散约束优化与启发式筛选完成跨模态策略组合。

**📊 数据集**

使用 Safebench 7 类危害（非法活动、仇恨言论、恶意软件、身体伤害、欺诈、色情、隐私侵犯）共 350 条查询。

**📈 对比分析**

与 Basic、QR-Attack、HADES、FigStep、SI-Attack、Ideator、TreeTeaming 等基线在 7 种 MLLM 上对比；HACA 在攻击成功率（ASR）和毒性分数均位列第一，平均 ASR 超过 95%，且查询次数最低。

**⚠️ 局限性**

仅评估了七类常见危害与主流 MLLM，未覆盖轻量化或新兴模型；性能高度依赖外部生成模型，迁移至弱模型时可能下降。

---

## 156. Circular Economy Synergies and Trade-offs in Data Centres

**arXiv ID:** 2608.04571 | [PDF](https://arxiv.org/pdf/2608.04571v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `38fa68f4-1c75-42bb-8d13-3b76129704e6`

---

## 157. Learning to Resolve Neutron Resonances with Fully Convolutional Neural Networks

**arXiv ID:** 2608.04027 | [PDF](https://arxiv.org/pdf/2608.04027v1)

**作者:** Nataly R. Panczyk `[一作]`, Majdi I. Radaideh `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

将一维全卷积神经网络用于自动检测中子传输光谱中的共振峰，并与传统R矩阵代码结合以减少先验偏差。

**💡 创新点**

首次提出无先验分布的峰值识别框架，并在多核素传输光谱上测试了模型的泛化能力。

**🔧 技术方法**

采用1D全卷积网络、B样条插值预处理、交叉熵+cosine annealing训练、TPE超参数搜索等技术。

**📊 数据集**

使用七组核素传输光谱（Cs‑133、Sm‑149评估与实验、Sm‑147、Cu‑63、Ir‑191、Ir‑193）进行训练与测试。

**📈 对比分析**

与单谱训练模型对比，单谱模型在准确率、召回率、F1上明显优于泛化模型；泛化模型主要将峰误标为非峰，单谱模型虽边界误差但峰中心定位准确。

**⚠️ 局限性**

泛化能力差、缺乏足够多样化数据导致对未见核素失效、纯数据驱动缺乏物理约束、峰边界误判影响后续参数提取。

---

## 158. The Greedy Binary Search Tree is Non-trivially Competitive

**arXiv ID:** 2608.04410 | [PDF](https://arxiv.org/pdf/2608.04410v1)

**作者:** Yuhao Guo `[一作]` (Tsinghua University), Chengzhang Wan `[通讯]` (Tsinghua University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

证明了某种在线二叉搜索树（如 Splay 树）在处理任意访问序列时，具有 2^O(√(log log n)) 的竞争比率。

**💡 创新点**

引入了“尺度化”分析框架，先在粗尺度上给出成本度量，再通过递推关系将其精细到更细尺度，并结合 Wilber 的 interleave 下界完成竞争上界的证明。

**🔧 技术方法**

使用 Lucas 算法的几何表述、Demaine 等人的 arboreally satisfied 超集视角、Wilber 的 interleave 下界、可见集与潜能函数分析，以及多层参考树 R_i 的构造和递归潜能证明。

**📊 数据集**

未使用实验数据，理论分析针对任意访问序列；为简化证明，假设 n = 2^(2^k) 并考虑长度 m ≥ n 的序列。

**📈 对比分析**

与此前最好的在线 BST 竞争比率 O(log log n) 相比，该结果给出了更小的 2^O(√(log log n)) 上界；虽然仍未达到 O(1) 的动态最优性，但是首个非平凡的竞争上界。

**⚠️ 局限性**

竞争上界仍不为常数，指数 2^O(√(log log n)) 可能有较大改进空间；证明依赖于参数 2 的常数，若改为 1 可得到 poly(log log n) 上界；此外对一般 n 的直接适用性需进一步讨论。

---

## 159. Efficient Online Lexicographic Generalized Low-Rank Matrix Bandits

**arXiv ID:** 2608.04324 | [PDF](https://arxiv.org/pdf/2608.04324v1)

**作者:** Bo Xue `[一作]` (City University of Hong Kong), Shuang Qiu `[通讯]` (City University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出并研究了多目标、按优先级排序的低秩矩阵带问题，并设计了两种在线算法：Scalar‑LowGLM（基于标量化的批量估计）和 Lexi‑LowGLM（按字典序筛选并在线 Newton 更新）。

**💡 创新点**

创新点在于：①首次将字典序偏好与低秩矩阵结构结合；②通过在线 Newton 步骤将估计复杂度从 O(T²) 降到 O(T)；③给出每个目标的目标相关累积风险上界，且该界仅依赖于有效低秩维度 (d₁+d₂)r，而非环境维度 d₁d₂。

**🔧 技术方法**

核心技术包括 Stein‑type 子空间估计、旋转投影得到低秩特征、各目标的异质正则化矩阵、UCB 上界与置信半径构造、以及字典序逐层过滤和在线凸优化更新。

**📊 数据集**

实验使用合成数据，矩阵秩分别取 1 与 2，比较了 G‑ESTT（单目标低秩算法）和 MTLO（线性字典序带算法）两种基线。

**📈 对比分析**

通过目标维度的累计风险曲线和平均运行时比较，Lexi‑LowGLM 在后续目标上表现出更低的累积风险，并在跑时上比 G‑ESTT 及 Scalar‑LowGLM 快约 21 倍、55 倍；与 MTLO 相比，Lexi‑LowGLM 既保持了较快的运行时，又显著降低了低优先级目标的风险。

**⚠️ 局限性**

局限性包括：实验仅限于合成数据；理论风险上界忽略常数与子空间估计误差，导致与实际性能不完全对应；未给出匹配的下界；对非平稳、延迟或重尾奖励等更复杂情境的适应性仍待研究。

---

## 160. DEGR: Dual Exploration-Driven Generative Re-Ranking for Adaptive Cross-Request Context Bridging

**arXiv ID:** 2608.04809 | [PDF](https://arxiv.org/pdf/2608.04809v1)

**作者:** Binglei Zhao `[一作]` (JD.com), Sulong Xu `[通讯]` (JD.com)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种双探索驱动的生成式重排模型DEGR，能够在固定上游供应的前提下，通过平衡即时收益与探索价值来生成更优的曝光序列，并在JD电商推荐系统中实现线上提升。

**💡 创新点**

创新点包括：① 结合监督学习与强化学习的双重探索优化框架；② 引入探索奖励模型，动态平衡即时与序列层面的探索收益；③ 采用自适应奖励加权ORPO（AR-ORPO）实现偏好优化；④ 通过多机制采样与多头解码协同提升多样性与效率；⑤ 在生成器内部实现跨请求上下文桥接，弥补传统两阶段模型的局限。

**🔧 技术方法**

技术手段：编码器-解码器（Transformer）生成器，使用多头解码协同并行解码；奖励模型采用DIN+MMoE+MLP预测即时CTR/CVR和序列探索值；多机制采样包含群体束搜索与启发式采样；探索多样性约束对同一解码头内部的相似度进行正则；自适应奖励加权ORPO利用序列奖励作为软权重进行对比优化。

**📊 数据集**

实验使用两个数据集：公开淘宝数据集（约2600万日志，8天）和JD生产数据集（约10亿请求，1亿用户）进行离线和线上评估。

**📈 对比分析**

与一阶、两阶及生成式基线方法（DCN、PRM、PIER、GRN、CMR、NAR4Rec、MG-E、GReF）对比，DEGR在离线指标（GAUC、NDCG、MAP@K、Recall@K）上均有提升，尤其在MAP@2/MAP@4提升明显；线上AB测试中实现UCTR +1.22%和PV +0.20%，并通过Ablation验证每项技术贡献。

**⚠️ 局限性**

局限性包括：奖励模型在有限采样候选空间下提升有限；模型参数与解码复杂度相对较高，需工程优化；依赖上游CTR预测质量，若上游极低则仍受限；探索奖励设计仍可进一步细化，需更多未来行为信号。

---

## 161. Foreseeing the Invisible: Amodal Reconstruction of Leaf Fossil Images

**arXiv ID:** 2608.04423 | [PDF](https://arxiv.org/pdf/2608.04423v1)

**作者:** Liuxiang Yue `[一作]` (Shanghai Jiao Tong University), Yikun Duan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

在单张RGB图像中直接预测化石叶片的可见、完整形状以及主脉和细脉掩模，实现无可见掩模的amodal重建。

**💡 创新点**

①全微调DINOv3 ViT-L/16 backbone；②在共享特征上并行四个头，其中两头用于韧皮质监督，为叶片形状提供结构先验；③使用多头损失与边界加权、Tversky等自适应约束；④实现了在无可见掩模条件下的端到端amodal分割。

**🔧 技术方法**

DINOv3 ViT-L/16 + DPT trunk + 四个3×3卷积头；多头损失（BCE、Dice、Tversky、面积约束等）；4-bit 权重量化 + ONNX Runtime Web；YOLO26-seg 石板/尺子检测；Flux.2 diffusion 生成生活叶片。

**📊 数据集**

基于 NMNS Cleared Leaf Database 160 枚完整叶片，使用 Blender 合成石板场景并随机破损得到 10k 训练 / 1k 验证的合成化石叶片数据集；此外在公开 amodal 数据集 KINS 与 COCOA-cls 上做对照实验。

**📈 对比分析**

在合成验证集上达到 95.0% Dice / 90.5% IoU 的完整叶片预测；在 KINS 与 COCOA-cls 上无可见掩模版本实现全 mIoU 85.05 / 66.65（KINS）以及 80.90 / 38.15（COCOA-cls），优于现有同类方法；浏览器 4-bit 版本在真实化石图像上 IoU 0.910（叶片）/0.845（主脉），保持高精度。

**⚠️ 局限性**

仅支持单叶片图像，无法进行实例分割；细脉恢复效果受限；完全合成训练导致域差异；固定 448×448 分辨率压缩信息；尺子检测仅适用于特定标尺；生成可视化阶段可能产生额外幻觉。

---

## 162. DataRx: Missingness-Aware Sampling for Safer Large Language Model Task-Specific Fine-Tuning

**arXiv ID:** 2608.04322 | [PDF](https://arxiv.org/pdf/2608.04322v1)

**作者:** Junbo Zhang `[一作]` (Northwestern Polytechnical University), Wen Jiang `[通讯]` (Northwestern Polytechnical University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了任务特定微调对大型语言模型安全性的影响，并提出了DataRx这一缺失感知安全样本选择方法，以降低微调导致的安全退化。

**💡 创新点**

创新点在于：①基于高维隐藏表示量化目标模型与安全参考答案之间的拒绝信号差距；②采用缺失感知采样，挑选能够弥补模型安全缺口的样本；③兼容并提升已有安全数据生成方法。

**🔧 技术方法**

技术包括：对LLM隐藏层进行对比学习，构建拒绝中心；计算每层拒绝分数并平均得到整体拒绝分数；定义安全适配得分（SAS）为参考与原生响应拒绝分数差；按SAS排序并选取Top K样本；结合LoRA进行参数高效微调。

**📊 数据集**

使用的安全数据集有Aegis、BeaverTails和生成式安全数据集GR‑SAP；任务数据集覆盖七类下游任务（GSM8K、MATH、HellaSwag、WinoGrande、MedQA、Magicoder、DocBlocks）；评估数据集包括DirectHarm4、HarmBench、HEx‑PHI。

**📈 对比分析**

与随机混合、最长、Paraphrase、Self‑Distill、SSS‑B、PSS‑B等方法对比，DataRx在Llama3‑8B‑Instruct、Qwen2.5‑7B‑Instruct和Mistral‑7B‑Instruct的攻击成功率分别从≈59%、≈50%和≈88%下降到≈14%、≈12%和≈62%，且对下游任务准确率影响仅约2%。

**⚠️ 局限性**

局限性包括：对安全数据质量高度敏感，低质量或噪声样本仍可能影响；仅在少数模型和任务上验证，泛化性待进一步探索；缺失感知采样的预算设定仍需经验决定。

---

## 163. A Vision-based Control Framework for Real-time Autonomous UUV Operations

**arXiv ID:** 2608.04723 | [PDF](https://arxiv.org/pdf/2608.04723v1)

**作者:** Erik Tjærand Frøland `[一作]` (NTNU), Eleni Kelasidi `[通讯]` (NTNU)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

实现了基于单目视觉的实时UUV定位、导航与3D地图构建框架，能够在海底网箱环境下完成自主网相对路径跟踪并生成体素地图。

**💡 创新点**

创新点在于整合FFT稀疏网格深度、TRUDepth稠密深度、相对与全局位姿估计以及wavemap体素映射，形成完全嵌入式实时闭环控制系统。

**🔧 技术方法**

使用了FFT网格检测、TRUDepth深度完成网络、相对姿态估计、DVL+声呐融合全局定位、wavemap体素建图、ROS+GPU节点以及PID/Ardusub控制器。

**📊 数据集**

利用自制Blender渲染的合成网箱数据集以及在挪威Trondheim MC‑Lab内的真实蓝色ROV2实验。

**📈 对比分析**

与TRUDepth和DVL平面估计对比，FFT方法在距离/朝向误差上更稳定，实验中误差在±0.1 m以内，控制追踪误差小于0.05 m，体素地图精度达到几厘米级。

**⚠️ 局限性**

局限在于对网箱可视性高度依赖，光照低或网格被遮挡时精度下降，且全局定位假设网箱为圆柱形，实际波浪或变形会引入误差。

---

## 164. Towards Robust Version Identification in the Wild: A Dataset, Benchmark, and Fine-Tuning Study

**arXiv ID:** 2608.04543 | [PDF](https://arxiv.org/pdf/2608.04543v1)

**作者:** Simon Hachmeier `[一作]` (Humboldt-Universität zu Berlin), Xavier Serra `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了一个规模达110万条、覆盖官方与用户生成内容的音乐版本识别数据集DiVers，提供版本级、段级音乐预测和多语言标签。

**💡 创新点**

创新点在于通过在YouTube搜索、模糊匹配与去重技术挖掘非官方版本，并自动为每个版本赋予音乐/非音乐段级标签及多维属性标签，显著提升数据多样性与真实性。

**🔧 技术方法**

技术方法包括基于常数-Q变换的特征提取、Triplet Loss训练与BPWR-5细调、PANN音频事件分类进行段级音乐检测，以及L²归一化增强嵌入鲁棒性。

**📊 数据集**

所使用的数据集为DiVers（官方与非官方版本），与Discogs-VI-YT、SHS100K、Da-TACOS等公开数据集共同构成训练/验证/测试分割。

**📈 对比分析**

与传统仅基于官方元数据的模型相比，DiVers训练的模型在噪声大、非官方环境下的检索准确率显著提升（MAP提升至0.708、NAR降至10.59），但在清晰录音集上略逊一筹。

**⚠️ 局限性**

局限性包括在细调过程中对干净版本的类间可分离度下降、标签自动匹配的召回率有限、以及潜在的文化/语言偏见未能完全消除。

---

## 165. Explicit Language Memory for Long-Horizon Planning in Vision-Language-Action Models

**arXiv ID:** 2608.04765 | [PDF](https://arxiv.org/pdf/2608.04765v1)

**作者:** Houze Xu `[一作]` (Fudan University), Ziyi Ye `[通讯]` (Fudan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种层次化的Vision‑Language‑Action（VLA）框架，加入显式自然语言记忆模块，利用高层视觉语言模型进行语义追踪，低层VLA执行连续动作；

**💡 创新点**

通过将任务历史压缩为可读的自然语言记忆，递归更新并指示子任务，实现长时序任务的阶段一致性和可解释性；

**🔧 技术方法**

使用PaliGemma视觉语言模型进行高层记忆与子任务生成，LoRA微调低层VLA以及300M流匹配动作专家，训练时采用自回归损失与流匹配损失；

**📊 数据集**

在BEHAVIOR‑1K（turn‑on‑radio）、Genie Sim 3.0（包裹分类）和真实XLeRobot的pick‑and‑place数据集上进行评估；

**📈 对比分析**

采用阶段成功率对比，显式记忆模型在BEHAVIOR‑1K从30%提升至40%，在Genie Sim单包任务从41.7%提升至63.9%，连续排序任务从31.3%提升至46.9%；

**⚠️ 局限性**

实验仅覆盖短任务，缺乏统计置信区间；显式记忆增加了额外计算与延迟；低层控制精度不足导致如按键操作等接触性子任务仍表现不佳；

---

## 166. InsightEmb: Learning Action-Intent Embeddings for Agentic Insight Retrieval

**arXiv ID:** 2608.04761 | [PDF](https://arxiv.org/pdf/2608.04761v1)

**作者:** Tsz Ting Chung `[一作]` (Hong Kong University of Science and Technology), Mo Yu `[通讯]` (Tencent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 InsightEmb，一种利用数学推理数据训练的对比学习框架，用于在 LLM 代理的每一步动态检索能解决当前程序瓶颈的抽象规则；

**💡 创新点**

创新点在于将抽象规则检索建模为跨域的“行动意图”匹配，克服传统语义相似检索无法跨层次抽象的局限；

**🔧 技术方法**

核心技术包括 InfoNCE 对比学习的两阶段训练（情境→洞见匹配与情境→经验匹配）、Qwen3‑Embedding‑4B 嵌入模型以及在检索时使用的任务指令前缀；

**📊 数据集**

使用的数据集涵盖 MATH 数学推理数据、ALFWorld、WebShop、ScienceWorld 三个交互式环境，以及 SRA‑Bench 静态检索基准；

**📈 对比分析**

通过与 Base 嵌入、ReasonIR、Llama‑NV‑Embed‑Reasoning、ALFWorld 训练嵌入等对比，InsightEmb 在 ALFWorld、WebShop、ScienceWorld 的任务成功率提升约 5–10 个百分点，在 SRA‑Bench 的召回率和 nDCG 上提升约 8–10 %；

**⚠️ 局限性**

局限包括：缺乏针对每一步最优洞见的黄金标签、评估仍以代理成功率为间接指标、对多环境和更大模型的通用性尚未完全验证、Stage‑1 训练对不同查询形式共享同一正负样本可能限制时序感知。

---

## 167. Joint UAV Flight and Opportunistic Routing under Reinforcement Learning for Delay-Tolerant Networks

**arXiv ID:** 2608.04590 | [PDF](https://arxiv.org/pdf/2608.04590v1)

**作者:** Xiao Wang `[一作]` (National Tsing Hua University), Shun-Ren Yang `[通讯]` (National Tsing Hua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种联合无人机航迹规划与机会式转发的延迟容忍网络控制框架 JUROR，并通过 CTDE‑PPO 进行训练，使每个节点在分布式环境下实现协同消息复制与 UAV 方向选择。

**💡 创新点**

创新点包括：① 将 UAV 航迹与转发决策统一为可分解的多智能体策略；② 引入可选热点预测辅助学习与热点引导对齐（HGA）以缓解奖励稀疏；③ 通过全局统计实现集中式训练与分布式执行，兼顾实用性与学习效率。

**🔧 技术方法**

核心技术为：PPO 强化学习框架、CTDE（集中式训练/分布式执行）架构、LSTM 预测网络、热点引导对齐机制、以及基于压缩观察的多智能体策略网络。

**📊 数据集**

实验使用 Helsinki‑medium WKT 测试平台（70 节点，5 台 UAV），并设计四种交通模式 M1–M4，评估不同 UAV 数量与训练策略下的性能。

**📈 对比分析**

对比 PRoPHET、MaxProp、Fan DPUVR、ICC Q‑learning 与 ICC FQLRP 等基准方法，JUROR 在所有流量模式下均能提升投递率，尤其在高负载场景中相对最大基准提升约 2–3 倍；同时在多 UAV 场景中保持较高的交付效率。

**⚠️ 局限性**

局限性：辅助模块（LSTM 与 HGA）对不同流量模式的敏感度高，需手工调参；实验仅在仿真环境中验证，缺乏真实网络部署与鲁棒性评估；大规模 UAV 航迹优化时训练样本与收敛速度受限。

---

## 168. Hidden Underbelly of the Silicon Valley: Algorithmic Exploitation and Health in Data Work Value Chains

**arXiv ID:** 2608.04019 | [PDF](https://arxiv.org/pdf/2608.04019v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 169. Binding Biometrics with AI Agent Identifiers for Delegation of Authority

**arXiv ID:** 2608.04292 | [PDF](https://arxiv.org/pdf/2608.04292v1)

**作者:** Joseph Geo Benjamin `[一作]` (Michigan State University), Karthik Nandakumar `[通讯]` (Michigan State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于生物识别的AI代理授权框架，利用模糊承诺绑定人类生物特征与代理身份及任务范围，实现不可否认的委托与实时身份验证。

**💡 创新点**

将模糊承诺与深度人脸特征结合，并设计了高斯抖动、随机投影、WTA哈希的特征适配方法，将连续值嵌入转换为可用于Turbo码的二进制模板，从而实现面部特征的可绑定与抗逆向。

**🔧 技术方法**

使用Turbo码错误纠正、WTA哈希、Gaussian抖动/投影、前向/反向错误校正编码、公共密钥加密及哈希。

**📊 数据集**

在LFW-a、CFP-FF、Multi-PIE三大公开人脸识别数据集上评估。

**📈 对比分析**

与三种人脸特征模型（IResnet101-ArcFace、AdaFace、KPRPE-Transformer）结合，使用1/3、1/2码率、不同阈值，实验显示AdaFace+1/3码率、τ=14时1024/2048/4096位Token分别获得≈96% TMR且FMR为0，优于其他配置。

**⚠️ 局限性**

对抗攻击不完整，假设转换密钥不安全；特征适配可能无法完全抵抗逆向；较高码率导致FMR上升；受模板分辨率限制；未给出完整安全证明。

---

## 170. Differentiating Through Dual Prices: End-to-End Policy Learning Under Capacity Constraints

**arXiv ID:** 2608.04669 | [PDF](https://arxiv.org/pdf/2608.04669v1)

**作者:** Mohammadsaeed Haghi `[一作]` (University of Southern California), Nima Kelidari `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出将双重价格分配策略的预测模型和价格参数联合端到端学习，直接通过对离线价值估计进行梯度传播来训练；

**💡 创新点**

创新点在于把双层优化中的内层价格问题显式为可微的函数，并使用隐式微分将价格梯度回传到预测模型，实现对容量约束的整体考量；

**🔧 技术方法**

技术包括离线强化学习中的IPW估计、双层优化与隐式微分、软max平滑与Nesterov熵光滑的凸近似、以及队列仿真评估；

**📊 数据集**

使用六个数据集：四个真实或半真实（Adult、ACTG 175、Criteo、Diabetes 130-US），一个十臂合成，另一个机制合成；

**📈 对比分析**

与传统的两阶段“预测-优化”管线、随机化控制实验、容量匹配的多臂网络以及无约束IPW最大化等方法比较，端到端方法在部署适配指标上始终占优，并在大规模真实数据上实现显著价值提升；

**⚠️ 局限性**

局限性包括：当可观测真实结果可用时，灵活的两阶段回归仍能在原始预测价值上占优；在观测数据中结果的假设性可忽略性问题；以及端到端方法对温度选择与梯度计算复杂度敏感。

---

## 171. When Modalities Fail to Tango: Conformal Backdoor Detection in Multimodal Contrastive Learning

**arXiv ID:** 2608.04052 | [PDF](https://arxiv.org/pdf/2608.04052v1)

**作者:** Yiming Chen `[一作]` (University of Macau), Jiantao Zhou `[通讯]` (University of Macau)

**通讯引用:** 10133 | [OpenAlex ID](https://openalex.org/A5037979193)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于粗细分两阶段的多模态对比学习后门检测框架

**💡 创新点**

创新点在于将跨模态一致性与合成预测（CP）结合，构造文本空间非一致性得分（NCS）实现可解释的置信区间检测

**🔧 技术方法**

采用CLIP的图像-文本编码器、外部文本生成模型（ClipCap）产生文本嵌入，使用高斯混合模型划分粗判集，再用CP与NCS对未决集进行细化判定

**📊 数据集**

主要使用大规模图像-字幕数据集CC3M进行实验，随后在ImageNet-1K等下游数据集验证模型安全与实用性

**📈 对比分析**

与CLIPScore、SafeCLIP、DAO等现有防御方法对比，平均FPR@100%TPR降至5.79%，AUROC最高达0.9999，整体检测精度显著提升

**⚠️ 局限性**

仅聚焦检测任务，未对检测后模型进行去学习或概念消除，且对极端高毒性率或多目标攻击的适应性尚待进一步验证

---

## 172. RORA: Realistic Object Reconstruction with Articulation

**arXiv ID:** 2608.04842 | [PDF](https://arxiv.org/pdf/2608.04842v1)

**作者:** Hyesung Lee `[一作]` (Seoul National University), Yongseok Lee `[通讯]` (DGIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一个半自动化的端到端工作流，从单一静态视频中重建带有关节的物体，并输出可在仿真环境中使用的混合高精度3D高斯斑点和网格模型。

**💡 创新点**

创新点在于结合自动关节建议算法与轻量级人机交互，能够仅凭单张静态视频实现多关节和链式动力学的精准重建。

**🔧 技术方法**

采用3D Gaussian Splatting、GS2Mesh、Convex Decomposition、SAM2、自动关节建议算法、URDF生成等技术。

**📊 数据集**

使用PartNet-Mobility数据集以及真实物体视频进行评估。

**📈 对比分析**

与Articulate-Anything和ScrewSplat基线比较，显示在几何误差、视觉指标和运行时间上均优于两者，尤其在多关节和链式对象上显著提升。

**⚠️ 局限性**

局限性包括对非外延部件或极小子部件的分割与关节识别不佳，以及对反射/透明表面的重建性能不足。

---

## 173. Preference-Driven Online Adaptation for Personalized Interaction Initiation in Proactive AI Assistants

**arXiv ID:** 2608.04416 | [PDF](https://arxiv.org/pdf/2608.04416v1)

**作者:** Yufeng Wang `[一作]` (South China University Of Technology), Mingkui Tan `[通讯]` (South China University Of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对主动式智能助手的交互时机决策，提出了Evidence-driven Online Preference Adaptation (EOPA) 框架，通过在线反馈动态更新时间与活动上下文证据，实现个性化交互时机决策。

**💡 创新点**

创新点在于将交互时机偏好转化为可度量的时间锚点和证据载体的活动原型，利用贝叶斯平滑与不确定性缩放，兼容稀疏交互样本并通过自适应融合实现高精度时机判断。

**🔧 技术方法**

采用时间锚点和活动原型的统计收集、Beta-Binomial平滑、方差缩放、加权融合、LLM响应生成（仅在交互时调用LLM）以及轻量级在线更新技术。

**📊 数据集**

在ProPerSim基准上，使用32个模拟用户，14天每日间隔2.5分钟共160,632步，生成交互时机标签。

**📈 对比分析**

与ProPerAssistant、EvoTest、SCOPE、Reflexion等基线相比，EOPA在交互时机F1提升至30.46（比Reflexion高19.8点），并显著降低推理延迟和适配时间。

**⚠️ 局限性**

局限在于对极端稀疏反馈的鲁棒性仍有限，且目前仅在模拟环境中验证，缺乏真实用户实验与跨场景通用性评估。

---

## 174. A Comparative Study of Feature Selection Methods for EHR Diagnosis Codes in Opioid Use Disorder Prediction

**arXiv ID:** 2608.04180 | [PDF](https://arxiv.org/pdf/2608.04180v1)

**作者:** Zihan Ding `[一作]` (Stony Brook University), Fusheng Wang `[通讯]` (Stony Brook University)

**通讯引用:** 11556 | [OpenAlex ID](https://openalex.org/A5100704639)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对电子健康记录中的诊断代码特征进行特征选择方法的系统比较，评估其在阿片使用障碍预测中的效果。

**💡 创新点**

首次同时对统计滤波、早期梯度敏感性、树模型重要性、正则化线性模型和大型语言模型语义先验五种范式进行比较，并在稀疏诊断空间中考察其性能、稳定性与稀有代码覆盖。

**🔧 技术方法**

使用NTK早期梯度敏感性、LightGBM‑SHAP、Elastic Net、Recurrence Enrichment、Claude Sonnet 4.5 LLM等技术，统一的BERT预测模型和Bootstrap稳定性评估。

**📊 数据集**

Cerner Health Facts多机构数据库，约1180万患者，其中约1.16%为OUD病例，使用截断至三字符的ICD‑10诊断代码。

**📈 对比分析**

对五种方法在不同词汇预算（K=50、100、150、200、300、500）下训练BERT模型，评估AUPRC、AUROC、Opt F1；结果显示NTK敏感性在500词汇下AUPRC最高（0.291），性能随词汇增长快速提升至约300词汇后趋于平稳；LLM仅做语义先验，单独效果较弱但与其他方法互补。

**⚠️ 局限性**

基于诊断代码的标签可能存在欠编码和时间偏差；ICD‑9→ICD‑10映射及3字符截断可能丢失细粒度信息；仅使用诊断代码，未考虑药物、手术等其他EHR域；在单一数据库上验证，外部可迁移性待检验。

---

## 175. FUSEP: A Multi-Center Benchmark for Diverse Tasks in Early Pregnancy Fetal Ultrasound Screening

**arXiv ID:** 2608.04766 | [PDF](https://arxiv.org/pdf/2608.04766v1)

**作者:** Bin Pu `[一作]` (Hunan University), Kenli Li `[通讯]` (Hunan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文构建了FUSEP早期孕期胎儿超声影像数据集，并在该数据集上对多种目标检测与域适配方法进行系统评估。

**💡 创新点**

创新点包括：①首个公开的多中心早期孕期胎儿超声数据集；②对14种解剖结构进行框式标注；③提供从全监督、半监督、无监督域适配到源自由域适配的完整基准与对比。

**🔧 技术方法**

采用了卷积与Transformer基目标检测框架（Faster R-CNN、YOLOX、DETR、Deformable DETR、DINO、Relation-DETR 等）以及多种半监督学习方法（Unbiased Teacher、Label Matching、Consistent Teacher、Semi-DETR、Sparse Semi-DETR、Semi-akmm）、域适配技术（SIGMA、SIGMA++、CMT、M³-UDA、ToMo-UDA、DATR）和源自由UDA方法（SF-AT、A²SFOD、IRG、ATSS）。

**📊 数据集**

使用自建的FUSEP数据集：4,017张CRL与NT视图，45,820个框级标注，来自三家医院，涵盖多种设备与扫描角度。

**📈 对比分析**

通过在全监督、半监督、UDA、源自由UDA四个任务中对比多种模型，结果显示：①全监督任务中Transformer基检测（Relation-DETR）mAP最高（85.6%）；②YOLOX在速度上最优；③半监督时Transformer方法表现最稳健；④UDA方法中基于解剖一致性的策略效果最佳；⑤源自由UDA在设备差异下仍能保持较高mAP。

**⚠️ 局限性**

局限性：仅包含两种视图，缺乏分割标注；极小结构导致极端尺度变化困难；跨中心性能受设备差异显著影响；实验仅覆盖三家医院，泛化能力仍有限。

---

## 176. When Memory Lies: An Empirical Study of Spatial Memory Staleness in VLM Agents

**arXiv ID:** 2608.04574 | [PDF](https://arxiv.org/pdf/2608.04574v1)

**作者:** Yushi Sun `[一作]` (LIGHTSPEED), Yanjie Zhang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在动态 FrozenLake 环境中，研究 VLM 代理在其空间记忆变得陈旧时的行为，并检验通过读时过滤是否能减轻安全风险。

**💡 创新点**

① 将空间记忆失效检测与导航任务结合，展示了记忆陈旧导致的安全税；② 明确了文本与视觉检测在迁移时的不一致性；③ 引入可控批量事件式审计，并与 Oracle 对比揭示过滤瓶颈。

**🔧 技术方法**

使用基于文本/视觉的二分类检测（文本 F1、视觉 F1）、批量审计与事件触发过滤、导航策略（NoMemory、NoFilter、SelfVerify、Filter）以及 LLM 交互。

**📊 数据集**

动态 8×8 FrozenLake 数据集：50 种随机种子、三种改变方案（L1、L2、L3），共 1,800 次检测运行和 12,000 次文本导航回合（GPT‑4o、Claude、Qwen、GLM），10 种视觉导航回合预览。

**📈 对比分析**

与无记忆、未过滤、单次自检等策略对比。文本检测 F1 接近 0.9，视觉检测差异显著；过滤后死亡率从 70% 降至 30% 左右，Oracle 与学习过滤几乎等价；然而在视觉模式下过滤效果不稳定，说明视觉审计可靠性是关键。

**⚠️ 局限性**

局限：仅在 8×8 的完全可观测格子世界上实验；视觉输入仅为单一渲染格式，未覆盖复杂真实感；模型数量有限；Oracle 结果显示在当前规模下已接近极限，可能无法在更大更复杂环境中保持同样效果。

---

## 177. Breadcrumbing Search Agents

**arXiv ID:** 2608.04565 | [PDF](https://arxiv.org/pdf/2608.04565v1)

**作者:** Xuebin Li `[一作]` (University of Science and Technology of China), Nenghai Yu `[通讯]` (University of Science and Technology of China)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对基于 LLM 的搜索代理设计了一套工具中介攻击框架，构建了状态化攻击运行时并提出了 Authority‑Chain Hijack（ACH）与 Trace‑Guided Strategy Evolution（TGSE）两种长周期攻击策略。

**💡 创新点**

创新点在于：①将攻击视为对搜索代理工具返回通道的动态介入，而非传统静态注入；②设计了可在多轮搜索中保持一致证据链的 ACH 策略；③采用基于执行轨迹的 DGM‑风格策略进化 TGSE，实现自动化的策略改进。

**🔧 技术方法**

主要技术包括 ReAct‑style 搜索代理、状态化攻击运行时（trajectory memory + planner）、策略驱动的 payload 生成、DGM‑启发的策略进化、以及对 SafeSearch 轨迹的解析与回溯。

**📊 数据集**

实验使用 SafeSearch 基准数据集，该数据集涵盖广告、偏见、误信息、危害与 Prompt‑Injection 等五类风险场景。

**📈 对比分析**

实验对比了多种静态与动态对抗基线，ACH 在整体攻击成功率（ASR）上提升 13‑36 点；在 TGSE 的帮助下，整体 ASR 可达约 70% 以上，显著超过所有基线。

**⚠️ 局限性**

局限性包括：仅在受控接口下评估，未检验在真实搜索平台上的可行性；策略可能对特定受害者过拟合，迁移性有限；以及 TGSE 的离线评估和策略搜索需要大量计算资源。

---

## 178. ArborEnum: Decision Tree Rashomon Sets over Continuous Features

**arXiv ID:** 2608.04310 | [PDF](https://arxiv.org/pdf/2608.04310v1)

**作者:** Zakk Heile `[一作]` (Duke University), Cynthia Rudin `[通讯]` (Duke University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 ArborEnum 框架，用于在包含连续特征的数据集上精确、近似或渐进式枚举决策树 Rashomon 集合。

**💡 创新点**

创新点包括：① 直接利用连续特征的有序阈值结构实现阈值剪枝，消除传统粗粒化二值化的限制；② 通过 AND/OR 图和子图缓存实现预算无关的枚举，显著减少重复搜索；③ 引入近似代理 LicketySNIP 以及“anytime”阈值激活机制，使得在资源有限时仍能得到高召回率的近似 Rashomon 集合。

**🔧 技术方法**

使用技术：基于 AND/OR 图的树结构、阈值到代理完成映射、Active Sample Distance 剪枝、阈值注册表 (Threshold Registry)、LicketySNIP/LSR 代理、预算迭代与子图扩展、以及二叉树/回溯搜索的混合实现。

**📊 数据集**

在 20 个含连续特征的公开数据集上验证：Abalone、Adult、Bank、Bike、Churn、Credit、Diamonds、Helena 等，使用不同 λ 值的叶子惩罚。

**📈 对比分析**

与现有方法（SORTeD、TreeFARMS、PRAXIS 等）相比，ArborEnum 在精确枚举时可达 63× 速度提升，近似版本在保持召回率 ≥ 0.99 的同时实现 270× 的加速；内存使用亦优于对手；“anytime”模式在达到全部阈值前仅额外 2.7% 运行时间，可在时间受限时获得快速的 Rashomon 近似。

**⚠️ 局限性**

局限性：对极大规模或深度树仍可能出现长运行时间；近似版本需依赖代理质量，若代理不够精确可能漏掉部分树；阈值激活策略对超参数（如预算、阈值间距）敏感，需手动调优；目前仅适用于分类决策树，尚未推广到回归树或规则列表。

---

## 179. Traceable LLM-Generated Hazard Scenarios for Operational Safety Analysis of Aviation Systems Using ASRS Reports

**arXiv ID:** 2608.04697 | [PDF](https://arxiv.org/pdf/2608.04697v1)

**作者:** Cristian Mascia `[一作]` (University of Naples Federico II), Stefano Russo `[通讯]` (University of Naples Federico II)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于大语言模型（LLM）的 AI 辅助方法，用来生成基于 NASA ASRS 记录的航空运营危害场景，并为每个场景提供结构化因素、叙述文本、可信度评分及可追溯链接。

**💡 创新点**

创新点在于：①将结构化因子与自然语言叙述一同生成，②引入进化式归谬（EVA）产生结构化因子后再指导叙述生成（^+），从而显著提升场景合法性与真实感；③提供可追溯的可信度评分，支持审计和专家评审。

**🔧 技术方法**

使用技术包括：多种 LLM（Vicuna、Llama‑2 等）与零/少样本提示、参数高效微调、进化式归谬算法、Jaccard 距离与 BARTScore 评估指标。

**📊 数据集**

数据集为 NASA Aviation Safety Reporting System（ASRS）公开报告，共 10,000 条，随机分为训练集与 20% 测试集，主要保留 13 个操作相关分类变量与 Result。

**📈 对比分析**

评估方法：通过 12 种 LLM+提示+微调组合测量生成失败率（GFR）、结构化因子 Jaccard 距离、叙述文本 BartScore，并与三种进化式、三种因果式基线及随机基线比较。结果显示：最优 LLM 配置在 GFR 与 Jaccard 距离上优于所有基线；^+ 在保证合法性的同时，结构化因子方差显著下降，叙述文本与基线差距不大。

**⚠️ 局限性**

局限性包括：ASRS 报告的自愿性与偏倚导致罕见灾难案例不足；评估指标仅基于与历史报告的相似度，未覆盖所有潜在的可行性；单一机型限制；模型可能出现漂移与幻觉，需要持续监控与验证；尚未在实际认证流程中进行实证验证。

---

## 180. Adaptive Finite-Budget Training for CVaR Risk-Aware Q-Learning

**arXiv ID:** 2608.04305 | [PDF](https://arxiv.org/pdf/2608.04305v1)

**作者:** Yifan Wu `[一作]` (University of Hong Kong), Wenjie Huang `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种自适应有限预算训练控制器，改进了CVaR风险感知Q学习的内外循环训练过程，以提升学习稳定性和风险调整后的性能。

**💡 创新点**

创新点在于结合六种协调机制（自适应内步长、匹配外衰减、早期VaR校正、覆盖-贪婪采样、后缀平均聚合、可观测校准），在不改变风险目标的前提下显著降低Bellman残差并提升金融回测效果。

**🔧 技术方法**

使用的技术包括两倍时间尺度的随机逼近、条件VaR与CVaR的变分表述、可观测量化的学习率与内循环深度校准、以及基于经验残差的动态采样分配。

**📊 数据集**

实验数据来自2018年2月8日至2026年6月28日的每日比特币价格与Crypto Fear & Greed情绪指数，共3,059条观测，采用27个离散状态与6种仓位动作。

**📈 对比分析**

与固定参数RaQL、固定仓位策略及买卖持有基线对比，改进控制器在856,000次内转移样本下将MeanBEQ从1.2202降至0.1854，Sharpe比率从0.5628提升至0.9281，最大回撤从17.77%降至6.46%。

**⚠️ 局限性**

局限性包括缺乏对自适应双层递推的收敛理论保证、仅在单一资产与表格状态空间下验证、以及未探索多资产或函数逼近环境的推广性。

---

## 181. Toward Uncertainty Quantification in Modern Art

**arXiv ID:** 2608.04038 | [PDF](https://arxiv.org/pdf/2608.04038v1)

**作者:** Tirtho Roy `[一作]` (Iowa State University), Tanusree Bhattacharjee `[通讯]` (Iowa State University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究现代艺术动画生成模型在多随机种子下的不确定性结构，提出一种可复用的评估协议，并构建首个现代艺术动画生成样本库。

**💡 创新点**

创新点在于：①首次将多随机种子生成的“解释分布”分解为可解释的结构维度（拓扑、离群、模态、各向异性等）；②设计一套基线+分布特征+分布模型消融的完整协议；③通过该协议能够区分种子不确定性与参考覆盖度，并揭示“源盲”不确定性仅能捕捉解读差异而非重建精度。

**🔧 技术方法**

使用源盲估计器（DCU、pairwise dispersion、embedding variance 等）、分布特征剖面、vMF/Kent/ACG/Student‑t/核/混合模型拟合、统计抽样重估与交叉验证，并在多编码器（OpenCLIP ViT‑B/32、ViT‑L/14、SigLIP、DINOv2）上进行比较。

**📊 数据集**

构建包含四类艺术风格（后印象派、波普艺术、超现实主义、现代通用）共计数百幅作品的“现代艺术动画生成样本库”，每幅作品在 Wan2.1 14B 上以四个随机种子生成 24 帧 832×480 视频，随后使用上述多编码器进行嵌入。

**📈 对比分析**

通过交叉验证 R²、AUROC、平衡准确率等指标比较，发现分布特征剖面在拓扑识别（0.98）与离群检测（AUROC 1.00）上远优于单一散度标量，但在预测跨种子语义差异时，分布结构并未显著优于简单散度（R² 0.294 vs 0.243）。

**⚠️ 局限性**

局限性在于：①分布特征对语义不一致性的预测并无提升；②源盲不确定性仅反映解读多样性而非对原作的忠实度；③在四个种子下对更细粒度结构（如对称双峰）的判别仍不可靠。

---

## 182. When Absence Is Evidence: Evaluating Completeness-Sensitive Negative Reasoning in Large Language Models

**arXiv ID:** 2608.04591 | [PDF](https://arxiv.org/pdf/2608.04591v1)

**作者:** Byoungjae Min `[一作]` (Sangmyung University), Jong Wook Kim `[通讯]` (Sangmyung University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了CROWN-QA，一个用于评估大型语言模型在“缺失信息”场景下的闭合判断的基准，专注于判断在已观察到的缺失支持下，是否应给出“否定”答案或保持“不确定”；

**💡 创新点**

创新点在于：①将缺失支持的判断拆分为“是否有正面支持”和“证据是否覆盖查询范围”两步，并将后者的覆盖关系做为任务标签；②设计了CROWN-Synth（可控制的同一问题同一事实配对）和CROWN-Real（真实文档对照集）两种评测形式，能够将覆盖关系作为单一可控变量；③提出了结构化的覆盖证书（query scope、evidence scope、Boolean判断）用于诊断错误来源；

**🔧 技术方法**

技术手段包括：控制式数据生成（四种覆盖表达规程 L1–L4）、对照集构造、使用多种提示策略（定义明确、链式思维、拒绝提示、自检、证书提示），以及对模型输出的结构化解析和错误分解；

**📊 数据集**

数据集主要有：①合成的“世界”数据（5个域，约5,000例），②ACL Anthology 会议论文标题索引（1,599例），③DailyMed 药品标签（1,599例）；在真实数据集上进一步构造 A/B/C 三种覆盖变体；

**📈 对比分析**

与三类公开权重模型（Qwen3.5-9B、Gemma-4-12B、Claude Haiku 4.5）以及多种提示条件进行对比。整体 class-balanced 准确率约 70–80%，但过度闭合率（OCR）普遍高于不足闭合率（UCR），尤其在隐式完整源类型（L3）下，模型常把部分覆盖误判为“Certified‑Negative”。提示策略能提升总体准确率，但并未显著降低 OCR 或 UCR，错误往往在覆盖判断上分布；

**⚠️ 局限性**

局限性包括：①模型在判断查询范围覆盖时不稳定，尤其在隐式完整与部分覆盖的对比中表现差异大；②提示和自检等方法只能在不同方向重分配错误，难以系统性消除 over‑closure；③结构化证书诊断表明错误大多来源于证据范围/完整性表述，说明模型对自然语言中隐含覆盖信息的把握不足；

---

## 183. HiSC: Hierarchical Spatial Clustering Token Compression for Efficient 3D Scene Understanding

**arXiv ID:** 2608.04610 | [PDF](https://arxiv.org/pdf/2608.04610v1)

**作者:** Jiuhe Qu `[一作]` (Beijing Institute of Technology), Ying Fu `[通讯]` (Beijing Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了HiSC，一种训练无关的层级空间聚类令牌压缩框架，用于加速多视角3D视觉语言模型的推理

**💡 创新点**

创新点在于将令牌压缩从单独令牌选择提升到基于空间连通的聚类层级处理，先通过空间图合并（SGraM）去除跨视角冗余，再通过空间聚类剪枝（SCluP）在LLM推理中按重要性与多样性进行层级分配

**🔧 技术方法**

采用几何-语义相邻图构建、连通分量聚类、聚类级预算分配、重要性+多样性混合采样等技术

**📊 数据集**

在ScanNet相关的五个多视角3D理解基准上评测：ScanRefer、Multi3DRefer、Scan2Cap、ScanQA、SQA3D

**📈 对比分析**

与FastV、VFlowOpt、VisPruner、VisionTrim、FastVGGT等训练无关的剪枝方法比较，HiSC在轻度/中度/极度压缩下分别保持≈99%/≈97%/≈93%的原模型性能，同时将令牌数降低到10%以下，显著优于基线

**⚠️ 局限性**

主要局限包括：仍依赖预训练的视觉编码器和LLM，压缩策略对不同场景的通用性和鲁棒性尚需进一步验证，且对极端高压缩可能仍丢失微细细节

---

## 184. IntentLint: Supporting Intent Scaffolding and Prompt-time Linting in Human-AI Collaborative Data Analysis

**arXiv ID:** 2608.04331 | [PDF](https://arxiv.org/pdf/2608.04331v1)

**作者:** Felicia Li Feng `[一作]` (University of Waterloo), Anamaria Crisan `[通讯]` (University of Waterloo)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种基于规则的协调层，包含意图支架和提示时检查机制，以支持人机协作的数据分析。

**💡 创新点**

创新点在于通过明确和可操作的规则来捕捉分析意图，从而提高协作透明度和一致性。

**🔧 技术方法**

使用了基于规则的协调机制，结合意图推断和提示检查的技术。

**📊 数据集**

在16名数据分析师的研究中进行了验证。

**📈 对比分析**

与传统方法相比，IntentLint提高了协作者的意图意识，促进了对分析策略的反思，显示出更好的协作效果。

**⚠️ 局限性**

限制在于该系统的有效性可能依赖于用户的参与程度和对规则的接受度。

---

## 185. Blockchain Empowered Trustworthy Agent Networks: Foundations, Taxonomy, and Future Directions

**arXiv ID:** 2608.04626 | [PDF](https://arxiv.org/pdf/2608.04626v1)

**作者:** Liehuang Zhu `[一作]` (Beijing Institute of Technology), Zijian Zhang `[通讯]` (Beijing Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文综述并梳理了1980–2026年AI代理网络的发展历程，提出了五维信任危机体系（实体与能力、授权与委托、信息与来源、协作与集体决策、结算与责任），并将其与区块链在身份、授权、溯源、协作、结算等方面的可信机制进行关联与匹配。

**💡 创新点**

创新点包括：①首次将代理网络的安全风险系统化为五维危机框架；②从风险-信任-机制三维度构建了区块链应用映射表，明确了区块链的边界与补充作用；③提出了多维度可信评估思路，为未来量化评估提供了参考。

**🔧 技术方法**

采用了大语言模型驱动的自主代理、Agent Communication Protocol、Model Context Protocol、Internet of Agents（IoA）等代理技术，以及区块链身份识别（DID/VC）、智能合约、可验证凭证、分布式账本、零知识证明等区块链相关技术。

**📊 数据集**

本文为综述性质，未使用特定实验数据集；主要引用公开论文、案例与现有研究成果。

**📈 对比分析**

本文通过文献综述与案例对比来说明区块链在不同信任层面的可行性与局限性，并未进行量化实验比较；讨论了区块链在身份、授权、溯源、协作与结算中的性能优势与不足。

**⚠️ 局限性**

局限性主要体现在：①未提供实证实验与统一评测指标；②区块链无法解决自然语言语义安全、隐私保护等深层问题；③多维度可信评估仍缺乏可操作的度量方法和标准。

---

## 186. From Transparent Labware Segmentation to Collision Avoidance: A Real-Time Edge-Aware Perception Pipeline

**arXiv ID:** 2608.04769 | [PDF](https://arxiv.org/pdf/2608.04769v1)

**作者:** Shijun Ding `[一作]`, Junlin Xiong `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一种轻量级边缘感知实例分割框架，结合多视角三角化的3D质心估计，用于透明实验室玻璃器皿的实时碰撞避免。

**💡 创新点**

创新点包括：①将边缘检测分支、边缘引导注意融合和参数无关SimAM模块集成到YOLOv5-Seg中，显著提升透明玻璃边界分割精度；②构建了LabGlass-IS真实实验室玻璃器皿实例分割数据集；③采用多视角最小二乘三角化得到3D质心，并用保守立方体实现实时碰撞约束。

**🔧 技术方法**

技术手段涵盖YOLOv5-Seg、轻量级ASPP边缘检测、BAM注意融合、SimAM特征细化、最小二乘多射线三角化、轴对齐立方体碰撞模型及MoveIt运动规划。

**📊 数据集**

使用自研LabGlass-IS数据集：3485张图像，6099实例，21类透明实验室玻璃器皿。

**📈 对比分析**

在LabGlass-IS上与YOLO+FastSAM、YOLACT、YOLACT++、PointRend等方法对比，Boundary F-score最高97.80，mAP_50:95达82.2，帧速率7.1 ms/帧；在真实机器人实验中碰撞避免成功率为93.3%。

**⚠️ 局限性**

局限性：仅提供保守的立方体碰撞约束，缺乏精确几何建模；对极细长玻璃器皿的定位误差仍较大，且未实现精准抓取与交互功能。

---

## 187. Physics-informed reduced-order modelling with equivariant spectral submanifolds

**arXiv ID:** 2608.04239 | [PDF](https://arxiv.org/pdf/2608.04239v1)

**作者:** Georg Maierhofer `[一作]` `[通讯]` (University of Cambridge), Georg Maierhofer (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了利用系统对称性进行谱子流形（SSM）约简的等变式方法（eSSM），并给出了其理论基础和完整算法。

**💡 创新点**

创新点在于证明SSM本身是等变子流形，并在参数化、约简动力学和扩展规范形式中直接嵌入对称群作用，显著降低自由参数并提升数值稳定性。

**🔧 技术方法**

核心技术包括：等变性理论（群表示与投影）、基于奇异值分解的对称子空间提取、正交与非正交投影的图像参数化、延伸规范形式与近共振条件、数据驱动的最小二乘拟合与正则化。

**📊 数据集**

使用了三个基准数据集：机械振荡链（线性耦合系统）、在球面上的粘性浅水方程（具有旋转对称性）以及周期域上的Kuramoto–Sivashinsky方程（来自CTF4Science挑战）。

**📈 对比分析**

通过与原始SSM学习（SSMLearn）以及其它数据驱动方法（如DMD、Koopman、SINDy等）比较，eSSM在参数量减少、拟合时间缩短（可达50%–75%）的同时保持或提升预测精度，尤其在对称变换下的泛化性能更好。

**⚠️ 局限性**

局限性包括：需要预先知道系统对称群；理论和算法主要针对固定点附近的光滑SSM，可能在强混沌或非平衡情形下不完全适用；高阶多项式拟合仍受数值不稳定和近共振分辨率限制。

---

## 188. Evaluating Theory of Mind in Reasoning Models: Robustness over Reasoning

**arXiv ID:** 2608.04646 | [PDF](https://arxiv.org/pdf/2608.04646v1)

**作者:** Ian B. de Haan `[一作]` (Leiden University), Max van Duijn `[通讯]` (Leiden University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估并比较了强化学习可验证奖励训练的链式思维（CoT）推理模型与非推理模型在理论心智（ToM）任务中的表现，探究其鲁棒性与实际推理能力的关系。

**💡 创新点**

提出将心理学实验改编与 Prompt 变体结合，结合基准结果来区分模型提升是由于鲁棒性增强还是新 ToM 能力，并通过对比分析证明推理模型的主要优势在于稳定性。

**🔧 技术方法**

使用了 RLVR（强化学习可验证奖励）训练的 CoT 推理技术、Prompt 变体设计、基准评测、定性推理轨迹分析和评分指标。

**📊 数据集**

采用了经典的 Sally‑Anne、Strange Stories、Imposing Memory 等心理学 ToM 测试，以及 FANToM、BigToM、MMToM‑QA、ParaphrasedToMi 等公开基准数据集。

**📈 对比分析**

通过在相同 Prompt 变体下对推理与非推理模型的准确率、鲁棒性指标进行对比，结果显示推理模型在大多数任务上表现出更高的稳健性，错误率显著降低，整体得分提升主要源于鲁棒性增强。

**⚠️ 局限性**

实验受限于缺乏完全受控的对照实验，模型差异来自 API 约束、温度设定等，可能导致元知识或模板匹配的偏差，且未能量化鲁棒性提升的具体幅度。

---

## 189. Design Choices That Matter: A Functional ANOVA Analysis for Remote Sensing Multi-Label Classification

**arXiv ID:** 2608.04702 | [PDF](https://arxiv.org/pdf/2608.04702v1)

**作者:** Maryam Gholami Shiri `[一作]` (Jožef Stefan Institute), Ana Nikolikj `[通讯]` (Jožef Stefan International Postgraduate School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用功能性方差分析（fANOVA）对两种遥感多标签分类（MLC）基准场景进行系统评估，量化网络架构、微调策略、学习策略和初始化等设计选择及其交互对模型性能的影响，并基于此构建数据集的敏感性元表示。

**💡 创新点**

创新点在于：①用fANOVA对不同数据集的设计选择重要性进行量化，揭示性能差异背后的因果机制；②通过聚类分析数据集的敏感性模式，发现数据规模与设计选择重要性之间的对应关系，提出了数据集依赖的最佳实践；③重新诠释以往基准结果，将其从“一刀切”的结论转变为“数据集特定”的指导。

**🔧 技术方法**

核心技术包括：功能性方差分析（fANOVA）与随机森林代理模型、层次聚类（HC）和余弦距离、Silhouette系数、Cophenetic相关系数等统计评估指标。

**📊 数据集**

使用七个公开遥感多标签数据集：Ankara、UCM、AID、DFC‑15、PlanetUAS、MLRSNet、BigEarthNet‑19/43，覆盖从小样本（216张）到大规模（590k张）的不同规模和空间分辨率。

**📈 对比分析**

对Scenario 1使用排名损失（ranking loss）评估48个CNN模型（不同架构、微调/冻结、端到端/特征提取），对Scenario 2使用平均精度（mAP）评估20个CNN/Transformer模型（不同架构、初始化）。通过fANOVA获得每个设计选择的方差贡献，聚类结果显示：大规模数据集性能受微调与架构主导；中等规模数据集受多因素交互影响；小样本数据集性能受初始化与架构交互决定；实验表明精细化的设计选择可显著提升性能，且fANOVA代理模型R²高达0.96。

**⚠️ 局限性**

局限性：仅考虑了CNN和早期Transformer架构，缺乏最新视觉基础模型；设计选择维度有限（最多三项）；数据集数量仅七个，聚类统计受样本量限制；训练设置（增广、优化器）固定，未纳入分析；因此结果在更大、更丰富的实验空间中的推广性仍需验证。

---

## 190. Securing Load Balancing over QUIC

**arXiv ID:** 2608.04164 | [PDF](https://arxiv.org/pdf/2608.04164v1)

**作者:** Garegin Grigoryan `[一作]` (Alfred University), Minseok Kwon `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 2099 | [OpenAlex ID](https://openalex.org/A5025860059)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了基于 QUIC 的无状态网络负载均衡器，并在 P4 可编程交换机上部署

**💡 创新点**

创新点在于利用 QUIC 连接 ID 与深度解析实现仅首包处理的无状态 LB，并在数据平面上阻止完整绕过和 0‑RTT 伪造攻击

**🔧 技术方法**

使用 P4 语言、PISA 架构、aioquic、P4Kube 框架进行实现

**📊 数据集**

使用 aioquic 生成的 HTTP/3 请求流，在 10 台服务器的 Fabric 测试平台上进行实验

**📈 对比分析**

通过对比部分 LB 与完整 LB 的流量分布和 RTT 延迟，证明深度解析对延迟影响极小，且能实现更均匀的流量分配

**⚠️ 局限性**

局限包括未在真实硬件上验证、0‑RTT 阈值与周期设定未完善、仅针对特定攻击场景，缺乏更广泛的安全评估

---

## 191. A Trust-region Framework for Moment Estimation

**arXiv ID:** 2608.04026 | [PDF](https://arxiv.org/pdf/2608.04026v1)

**作者:** Oluwasegun A. Somefun `[一作]` (Oregon State University), Oluwasegun A. Somefun `[通讯]` (Oregon State University)

**通讯引用:** 217 | [OpenAlex ID](https://openalex.org/A5054849356)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于 p 阶矩约束的信赖域框架 Gmake，用来统一解释 Adam 等自适应动量机制、学习率调度、谱低通滤波与矩阵谱归一化。

**💡 创新点**

创新点在于从严格的信赖域视角推导出 p∈[2,4] 的矩估计学习率机制，展示高阶矩（尤其 p=4 的峰度）对更新控制的作用，并将谱正则化与矩阵操作视为互补的信赖域约束。

**🔧 技术方法**

采用了 p 阶矩信赖域约束、指数移动平均估计、低通一阶滤波器（可实现 Heavy-ball / NAG）、矩阵逆平方根正交化等技术，实验基于 GPT‑2 124M 模型训练。

**📊 数据集**

使用了 FineWeb‑Edu（5000 万词）和 TinyStories 两个数据集。

**📈 对比分析**

通过比较基本、谱滤波和矩阵操作三种 Gmake 形式以及 p=2、4 的训练/验证损失，发现谱滤波与矩阵归一化显著提升性能；p=4 在较大信赖域下优于 p=2，但信赖域收紧后两者差距缩小。

**⚠️ 局限性**

局限在于高阶矩优势主要在弱信赖域，较强的滤波/归一化会削弱其效果；尚未找到针对高阶矩的谱正则化方法；实验范围有限，未系统探索 p 范围、超参与不同任务的影响。

---

## 192. Benchmarking Multi-fidelity Neural Operators on Complex PDE Problems with Non-trivial Fidelity Differences

**arXiv ID:** 2608.04708 | [PDF](https://arxiv.org/pdf/2608.04708v1)

**作者:** Ghifari Adam Faza `[一作]` (KU Leuven), David Moens `[通讯]` (KU Leuven)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究并比较了多种多保真学习策略（包括传递学习、两步模型、残差增强两步模型和中间结构）在神经算子（NO）上的应用，旨在解决高保真数据昂贵且样本有限的PDE求解问题。

**💡 创新点**

创新点在于：①提出将低保真数据与高保真数据结合的多保真框架，并系统评估四种常见策略；②通过对2D Darcy流域进行POD降维来模拟更大尺度差异的真实多保真场景；③引入时变烟流入问题作为新的复杂多保真基准；④发现传递学习在所有测试场景下表现最稳健，并揭示直接注入低保真信息的策略对大差异敏感。

**🔧 技术方法**

采用的技术主要是：深度算子网络（FNO和WNO），多保真架构（中间、两步、残差两步、传递学习），以及标准的MSE损失加正则化。实验使用了TensorFlow/ PyTorch 的实现。

**📊 数据集**

使用的数据集包括：①一维随机Poisson方程（随机高斯过程驱动）；②二维三角形Darcy流域（Tripura等提供的高低分辨率数据）；③修改后的Darcy流域（使用POD保留前2个模态的低保真数据）；④二维无界烟流入（PhiFlow仿真生成的125个入口配置、126个时间步，低保真96×120网格，高保真256×320网格）。

**📈 对比分析**

对比方法：在相同高保真样本数（如50个）和不同低保真样本量的设置下，计算各模型对高保真测试集的RMSE。结果显示：在1D Poisson中，传递学习与HF基线相当，其他模型略逊；在2D Darcy（网格分辨率）和POD降维的修改版中，传递学习显著优于其它多保真模型和HF基线；在无序烟流入时，传递学习在所有时间步均低于HF基线，而两步模型在测试集上泛化差。整体来看，传递学习是最稳健、最具通用性的策略。

**⚠️ 局限性**

限制与不足：①仅使用FNO和WNO两种算子架构，未覆盖更广泛的NO变体；②实验样本量受GPU内存限制，部分模型（如中间结构）未在烟流入上评估；③未对低保真与高保真差异进行系统的统计或不确定性分析；④多保真差异主要来自分辨率或POD降维，缺乏基于物理方程差异的真实案例；⑤对不同超参数组合的探索有限，未给出理论分析。

---

## 193. EASy: Towards Efficient LLM-Based Agentic System

**arXiv ID:** 2608.04588 | [PDF](https://arxiv.org/pdf/2608.04588v1)

**作者:** Junnan Liu `[一作]` (Monash University), Gholamreza Haffari `[通讯]` (Monash University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种面向效率的 Agentic 框架，利用基于能力–成本的调度器和 milestone‑plan‑act 工作流，在任务解答过程中同时优化任务成功率与计算成本。

**💡 创新点**

创新点在于：① 将树结构化回放与多元奖励（正确性、效率、完整性）相结合，显著提升策略学习；② 在任务分解层面引入里程碑概念，支持并行执行与动态适配；③ 通过可解释的能力‑成本描述实现执行器的自适应选择。

**🔧 技术方法**

核心技术包括：LLM 驱动的 orchestrator（Qwen2.5‑7B‑Instruct）、GRPO 强化学习、树结构化回放、奖励设计（R_cor, R_eff, R_com）、DAG 并行执行与里程碑级规划。

**📊 数据集**

使用了数学推理数据集（AIME24/25, MATH500）、具身决策数据集（ALFWorld, WebShop）、深度研究数据集（GAIA, Humanity’s Last Exam）以及约 7K 训练样本的混合数据集。

**📈 对比分析**

与多种最先进的 Agentic 基线（Direct Reasoning、ReAct、AutoGen、Reflexion、Plan‑and‑Act、AgentFlow、MasRouter）以及路由式系统对比，实验表明在所有基准上均实现了更高的任务准确率和更低的 token/计算成本，取得了最佳性能–效率权衡。

**⚠️ 局限性**

局限性：① 依赖预定义的能力‑成本排名，难以直接迁移到完全未知的执行器；② 树结构化回放在极大分支或极长序列任务中可能产生高计算开销；③ 现有评估侧重于 LLM 计算成本，未覆盖多模态或硬件加速等更广泛的效率指标。

---

## 194. AdaptAgent: A Multi-agent, Domain-Guided Reasoning Framework for Code Adaptation

**arXiv ID:** 2608.04459 | [PDF](https://arxiv.org/pdf/2608.04459v1)

**作者:** Xiaokai Rong `[一作]` (University of Texas at Dallas), Tien N. Nguyen `[通讯]` (University of Texas at Dallas)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个多代理、领域指导的推理框架，用于将在线代码片段自动适配到目标项目中，生成语义正确的补丁。

**💡 创新点**

将适配意图提炼、域特定指导、上下文挖掘、规划和迭代验证分离为专门代理，利用LLM进行分步规划和自校验，显著提高适配准确率并模拟人类开发者的修改模式。

**🔧 技术方法**

采用大语言模型（GPT‑4o）+多代理架构、Chain‑of‑Thought规划、类型化消息接口、程序依赖图/AST相似度评估、编译器验证及自动程序修复工具对比。

**📊 数据集**

使用Zhang等人基于Stack Overflow和GitHub的629条Java适配实例（542条有效、952条配对），包含测试用例的子集D_t和无测试用例的D_m。

**📈 对比分析**

与基线（单步提示、加域指导、复制+APR等）进行对比，采用测试通过率、人工评估、代码相似度（PDG、AST、CodeBLEU、编辑距离）评估；AdaptAgent在D_t上65.2%/D_m 62.3%语义正确率，显著优于基线（提升5.5–53.6%），且与人类修改模式的相关系数ρ≈0.9。

**⚠️ 局限性**

对长、多块、跨方法的大规模适配仍易失误；缺乏更丰富的语义上下文（调用图、数据流）；仅验证Java，无法直接推广到其他语言；仍依赖LLM的可靠性与提示设计。

---

## 195. An Exploratory Study of Agent Plans for Agentic AI Coding Tools in Open-Source Software

**arXiv ID:** 2608.04661 | [PDF](https://arxiv.org/pdf/2608.04661v1)

**作者:** Muhammad Auwal Abubakar `[一作]` (University of Bamberg), Matthias Galster `[通讯]` (University of Bamberg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对36,710个GitHub仓库进行搜索，发现并分析了10个仓库中共85个Markdown Agent Plan文件，研究了它们在仓库中的保存方式、支持的开发活动以及提供的执行指导信息。

**💡 创新点**

首次将任务导向的Agent Plan视为独立的仓库级配置工件，系统地划分并量化其内容结构与功能类别，为理解人-代理协作提供新的研究视角。

**🔧 技术方法**

使用GitHub API扫描特定工具相关目录，手工验证并对文件和章节进行层级编码，统计并计算Cohen’s κ以评估编码一致性。

**📊 数据集**

研究数据来自36,710个已筛选的工程型GitHub仓库，最终构成10个仓库、85个Agent Plan Markdown文件的语料库。

**📈 对比分析**

通过对比不同仓库、不同作者与AI共著标记的文件，评估计划文件在任务分类和信息结构上的分布，发现实现步骤、文件位置、测试验证等为最常见的三类指导信息；但未涉及性能指标或对比实验，主要以定性与频数统计为主。

**⚠️ 局限性**

样本极度集中（77.6%来自单一仓库）、仅检索特定目录，未捕获本地或其他路径的计划文件，缺乏对AI共著与非共著文件在效果上的定量评估，且归纳结果不一定适用于更广泛的OSS生态。

---

## 196. Leak-Resistant Unlearning: A New Benchmark for Evaluating Multi-Hop Reasoning Consistency and Recovery Robustness

**arXiv ID:** 2608.04519 | [PDF](https://arxiv.org/pdf/2608.04519v1)

**作者:** Haoting Qian `[一作]` (Tsinghua University), Han Qiu `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一套同时考察多跳推理路径和恢复攻击的LLM无学习基准，评估模型在不同推理结构下的知识遗忘效果。

**💡 创新点**

创新点在于引入六种逻辑启发的多跳推理结构和恢复攻击，扩展传统基准覆盖面，并构建完整的数据生成与自动验证流程。

**🔧 技术方法**

使用LLM（如GPT‑4o）进行知识抽取、问句生成与质量检测；在三大LLM（Llama‑3.1‑8B、Qwen3‑14B、Qwen3‑32B）上实现六种无学习方法（GA、NPO、RMU、TV、PALU、AS）及三种恢复攻击（Probab、FocusOnKey、Quantization）。

**📊 数据集**

基于两大公开数据集MQuAKE（Wikidata）和Books（哈利·波特系列）构建多跳问题集。

**📈 对比分析**

通过对比六种无学习方法在六种多跳推理路径下的遗忘质量、恢复率和模型通用能力，发现不同推理路径对遗忘效果影响显著，多跳查询更易恢复；无学习方法无法同时实现高遗忘质量、恢复鲁棒性和模型效能。

**⚠️ 局限性**

主要限制包括：数据构建依赖LLM，可能引入偏差；实验仅覆盖开放权重LLM，未评估商业API；聚焦事实知识，未覆盖隐私或隐式行为；仅在英语环境下验证，跨语言性能未知。

---

## 197. TwinIR: Coordinated Invisible Dual-Point Attacks on Online HD Map Construction

**arXiv ID:** 2608.04453 | [PDF](https://arxiv.org/pdf/2608.04453v1)

**作者:** Haibo Hu `[一作]` (City University of Hong Kong), Jianping Wang `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为 TwinIR 的双点近红外物理攻击方法，针对在线高清地图构建模型实现隐蔽且有效的地图失真。

**💡 创新点**

创新点包括对跨边界补偿效应的系统分析、机制导向的稀疏两点攻击策略以及利用近红外光源实现视觉上不可察觉的攻击。

**🔧 技术方法**

主要技术包括目标条件下的攻击点定位（先主点后次点）、Chamfer 与方向损失评估、数字近红外渲染与实际物理部署转换。

**📊 数据集**

使用 nuScenes 数据集中的非对称场景子集，对 MapTR、MGMap 与 DAMap 三种在线地图构建模型进行实验。

**📈 对比分析**

与单点盲目攻击对比，TwinIR 在 mAP 上下降 8–9 pp（RSA）或 2–5 pp（ETA），使未达目标率提升 11–12 pp，危险轨迹率提升 3–8 pp，整体攻击效果显著提升。

**⚠️ 局限性**

局限性包括：仅限于最多两点攻击；依赖黑盒查询与预先收集的数据；近红外光源虽不易被人眼察觉，但仍可能被特定红外摄像头检测；在更复杂场景或多模型攻击时效果需进一步验证。

---

## 198. Transfer Learning for Named Entity Recognition of Classical Latin through LLM Prompting

**arXiv ID:** 2608.04015 | [PDF](https://arxiv.org/pdf/2608.04015v1)

**作者:** Callum Chan `[一作]` `[通讯]`, Callum Chan

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在古典拉丁语命名实体识别共享任务中，对商用生成式大语言模型 Gemini‑2.5‑pro 与 Claude‑Sonnet‑4‑5 进行零样本和少样本提示，构建并提交了高性能识别系统。

**💡 创新点**

创新点在于首次系统性验证少样本提示在低资源古典拉丁语 NER 上的有效性，并通过提示模板实现多类别（粗细粒度）实体识别的高准确率。

**🔧 技术方法**

技术核心为提示工程（Prompt Engineering）——设计任务概述、具体要求、实体类型及输出规范，并在少样本提示中加入三句已标注拉丁语示例；使用 Gemini‑2.5‑pro 与 Claude‑Sonnet‑4‑5 这两款商用 LLM 进行推理。

**📊 数据集**

数据集包括官方提供的约 2900 个标注样本（涵盖 86 个实体），以及作者自行整理的 1.48 M 拉丁语标注语料（覆盖 Person、Place、Collective 三类），后者被映射至任务标签集。

**📈 对比分析**

在官方评测中，本文提交的 uOttawa_nerc_1（Gemini‑2.5‑pro 少样本）与 uOttawa_nerc_2（Claude‑Sonnet‑4‑5 少样本）在粗细粒度子任务上分别取得 F1 分别 0.894 与 0.856 的最高成绩，整体性能优于其它参赛队伍，尤其在粗粒度子任务中 Claude 表现更好，细粒度子任务中 Gemini 表现更佳。

**⚠️ 局限性**

局限性包括仅针对共享任务提供的文本进行实验，缺乏对更广泛拉丁语语料的验证；实验仅覆盖两款 LLM，未做提示/超参数的消融或统计显著性检验；提示设计可能带来主观偏差，且高召回率导致过度预测的问题。

---

## 199. MobileWAM: Bridging World Action Models to Mobile Manipulation with Chain-of-Foresight

**arXiv ID:** 2608.04657 | [PDF](https://arxiv.org/pdf/2608.04657v1)

**作者:** Zehua Fan `[一作]` (Tsinghua University), Yan Wang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

结合预训练视频扩散变换器与轻量动作专家，构建了一种适用于移动机械臂的世界动作模型；

**💡 创新点**

创新点包括：1）引入移动式混合专家（Mobile MoE）实现行走与抓取的专业化；2）Chain‑of‑Foresight训练时的循环未来潜在预测，零推理成本；3）解耦的视频与动作去噪，使部署时仅需当前帧；

**🔧 技术方法**

使用技术：预训练视频扩散变换器、3D VAE压缩、T5文字编码、层级联合注意力、流匹配损失、RNN式未来潜在链；

**📊 数据集**

数据集：ManiSkill‑HAB（SetTable 子任务）以及通过ARX Lift2收集的远程操作演示；

**📈 对比分析**

与 ACT、DP、DP3、RDT、AC‑DiT、AnchorVLA 等方法对比，平均成功率 73.0% 领先所有基线，部署时推理速度为 Motus 与 LingBot‑VA 的 5–8×；

**⚠️ 局限性**

局限性：在拾取–投放子任务中仍易受定位误差、碰撞干扰和无回收机制影响，需进一步提升时空一致性与闭环错误恢复能力。

---

## 200. Random features for Grassmannian kernel approximation with bounded rank-one projections

**arXiv ID:** 2608.04227 | [PDF](https://arxiv.org/pdf/2608.04227v1)

**作者:** Rémi Delogne `[一作]` (UCLouvain), Laurent Jacques `[通讯]` (UCLouvain)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一类基于随机特征的Grassmannian核近似方法，利用秩一投影并与有界非线性函数（符号或周期性）组合，生成可控误差、可扩展的子空间嵌入；

**💡 创新点**

创新点在于：1）用秩一随机投影替代密集投影，显著降低存储与计算；2）通过有界非线性实现轻量级、可聚合的二值/周期性特征；3）给出统一误差上界，证明与主角角度相关的正定Grassmannian核；4）设计结构化Hadamard+随机符号投影实现更高效的随机特征；

**🔧 技术方法**

技术包括：随机秩一投影、符号映射、复指数映射、Hoeffding/子指数/子韦伯尾分布分析、Johnson–Lindenstrauss及Fastfood/Hadamard结构化投影、主角角度理论、核方法与SVM；

**📊 数据集**

实验数据集为ETH‑80图像集，使用32×32灰度图生成9维子空间；

**📈 对比分析**

与传统密集投影、完整核矩阵、以及已有的投影/Binet‑Cauchy核做比较，结果显示：二值/周期性随机特征在5–20%压缩比下即可达到接近原核的分类精度，结构化投影在计算时间上显著优于完整核，尤其在80类子空间分类任务中速度提升10–30倍；

**⚠️ 局限性**

局限性：1）对完整Grassmannian的均匀近似需m=O((kn)²)以应对重尾分布；2）符号核缺乏闭式表达，难以直观分析；3）结构化投影的理论保证尚未完整；4）对极低频/高频参数的选择敏感；5）实验仅验证了图像集，需进一步验证跨领域效果。

---

## 201. Overcoming Statistical Bias in Action-Controllable World Models

**arXiv ID:** 2608.04653 | [PDF](https://arxiv.org/pdf/2608.04653v1)

**作者:** Yuhong Shi `[一作]` (Xi'an Jiaotong University), Jingwen Fu `[通讯]` (Zhongguancun Academy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过引入多步和空间反事实一致性约束（MSC2与ASC2），提升动作可控性的视频世界模型；

**💡 创新点**

创新点在于提出CoCo框架，将反事实一致性与动作空间等价性结合，并给出ARC/DE评估指标和Mini-SSMB数据集；

**🔧 技术方法**

使用Transformer基的视频生成模型，配合离散化token化、SmoothL1等损失实现多分支反事实训练；

**📊 数据集**

主要使用Mini-SSMB、BAIR、RoboNet、VP^2以及MetaWorld等数据集进行评估；

**📈 对比分析**

与iVideoGPT、FitVid、SAMPO、MaskViT等基线对比，CoCo在ARC、DE等动作可控性指标显著提升，VP^2任务平均成功率达73.1%，在多任务和模型基强化学习中亦表现更佳；

**⚠️ 局限性**

局限在于反事实与空间变换需预定义，计算开销增大，且对不可逆接触或复杂语义动作的适用性有限。

---

## 202. Behavioral Skill Reconstruction: Reconstructing Hidden Functionality from LLM Agent Skills

**arXiv ID:** 2608.04192 | [PDF](https://arxiv.org/pdf/2608.04192v1)

**作者:** Peichun Hua `[一作]` (University of Southern California), Mengyuan Li `[通讯]` (University of Southern California)

**通讯引用:** 425 | [OpenAlex ID](https://openalex.org/A5100415393)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本工作研究了在闭源代理技能中通过正常任务交互恢复隐藏功能的攻击方法，提出了一种基于接口假设、结构化探测、程序合成与差分验证的闭环黑盒重构技术；

**💡 创新点**

创新点在于将技能功能泄露建模为行为重构问题，并首次提出完整的BSR框架以及对功能泄露多维瓶颈和现有防御缺陷的系统评估；

**🔧 技术方法**

主要技术包括主动查询与主动学习式探测、基于类型假设的程序合成、差分验证与迭代修复、以及模型相对边际功能的度量；

**📊 数据集**

使用了30个公开技能样本，来源于SkillsBench、SkillRet以及公开注册库，覆盖规则、表格、程序和算法等多种功能类型；

**📈 对比分析**

评估方法采用持留测试与多模型对比，平均功能匹配率（ASR）约为71%，在20个可行目标中有16个能恢复至大于零阶阈值；闭环重构与自适应探测显著提升性能；防御措施（泄露过滤、广告最小化）对功能重构几乎无效；

**⚠️ 局限性**

局限性包括仅评估确定性规则/表格/公式/程序，缺乏对创造性交互或判断型技能的评估；实验仅覆盖有限模型与公开技能，未涉及商业付费服务；未考虑更全面的防御机制与跨会话监控等。

---

## 203. From Cake-Cutting and Necklace-Splitting to Fair Division of Indivisible Items

**arXiv ID:** 2608.04340 | [PDF](https://arxiv.org/pdf/2608.04340v1)

**作者:** Max Dupré la Tour `[一作]` (RIKEN Center for Advanced Intelligence Project University of Tokyo), Ayumi Igarashi `[通讯]` (RIKEN Center for Advanced Intelligence Project University of Tokyo)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种从连续公平分割定理（如蛋糕切割、项链分割）到离散不可分物品（排列在路径上的物品）的一般性转移框架，利用该框架在不要求可加性的情况下得到 EFk‑类型的公平性保证；

**💡 创新点**

创新点在于构造了“虚拟估值”和“迭代中位数”运算，能够在保留连续公平性（如连通的无嫉妒分割）后通过一种鲁棒的取整规则，将结果映射到离散实例，同时控制公平性损失仅为每个代理的边界物品；

**🔧 技术方法**

核心技术包括：
- 利用 Sperner 定理和 Kuhn 三角剖分对连续分割空间进行离散化；
- 设计证书三元组 W(I) 与迭代中位数来定义虚拟估值；
- 通过多重投影（ρ_j）将多段分配映射到代理的分段集合；
- 结合连通蛋糕切割与等权项链分割定理，得到 EF1^c_g、EF2、consensus EFn^c_g 等结果；

**📊 数据集**

文中没有使用实验数据集，所有结论均为理论存在性证明；

**📈 对比分析**

由于论文主要给出存在性证明而非算法实现，故没有与其他方法进行实验性能比较；理论上，该框架在满足连续公平定理条件（非负/非正、同一评价、prime‑power 代理数）下，能够保证连接分配满足 EF1^c_g，并在 prime‑power 代理数下得到 EF2 或 EF2^c_g 并实现近似平衡；

**⚠️ 局限性**

局限性包括：
- 结果是存在性的，缺乏多项式时间或可实现的算法；
- 依赖于代理数为 prime‑power 或所有代理同一评价；
- 对非连通分配或更一般的结构约束（如任意图形）尚不适用；
- 对非additive 价值的进一步约束（如子加性）未得到加强。

---

## 204. Revisiting Channel Effectiveness: A Multi-Dimensional Evaluation with Primitive Visual Stimuli

**arXiv ID:** 2608.04435 | [PDF](https://arxiv.org/pdf/2608.04435v1)

**作者:** Soohyun Lee `[一作]` (Seoul National University), Jinwook Seo `[通讯]` (Seoul National University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统评估七种视觉通道（位置、长度、倾斜、面积、曲率、亮度、饱和度）在四种感知任务（准确性、可辨别度、可分离度、预注意捕捉）中的表现，并基于原始视觉刺激构建多维通道效能概况。

**💡 创新点**

创新点在于：①去除图表框架，仅使用原始刺激以隔离通道本身；②同时考察四种感知维度，揭示通道排名随任务变化；③提出情景驱动视角（任务风险、数据细粒度、背景干扰、响应时限），提供更细粒度的编码选择指南；④使用等价边际和效应量判断实用差异。

**🔧 技术方法**

技术手段包括：使用 reVISit 平台实现在线实验；基于指数幂律、Weber、Anchor Harmonic Weber 模型对数据进行拟合；采用 TOST、Benjamini–Hochberg 校正、多重对比分析；使用聚合效应量（Hedges' g）与等价区间评估差异。

**📊 数据集**

数据集：来自 105 名 MTurk 参与者（实验总计约 2000 条数据）以及额外 45 名参与者用于可辨别度实验；每个通道在不同任务下分别呈现 1–32 条试验，覆盖完整值域。

**📈 对比分析**

比较方法：将每个通道在不同任务下的误差、JND 曲线、检测准确率等量化，利用对数误差、R²、边缘斜率、检测率等指标；实验结果显示空间通道在准确性任务中占优，面积通道在可辨别度和预注意捕捉中表现最好；但同一通道在不同任务中排名大相径庭，体现多维度效能。

**⚠️ 局限性**

局限性包括：1）MTurk 受限于屏幕、照明差异导致颜色通道噪声；2）每位参与者试验次数有限，难以评估个体差异；3）仅在共享画布边框下测量，可能不适用于无框架的 AR/嵌入式情境；4）部分任务（如位置在可辨别度）未直接测试；5）仅考虑两维交互，未涵盖三维/多通道交互。

---

## 205. Structured LLM Reasoning for Zero-Shot Human--Robot Coordination Under Hidden Goals

**arXiv ID:** 2608.04309 | [PDF](https://arxiv.org/pdf/2608.04309v1)

**作者:** Dong Hae Mangalindan `[一作]` (Michigan State University), Vaibhav Srivastava `[通讯]` (University of California Santa Barbara)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在私有目标视角下，提出一种基于 Dec-POMDP 的结构化大语言模型 (LLM) 架构，实现人机零样本协作建筑任务。

**💡 创新点**

创新点在于：①将决策理论拆分为 ToM 推理、分层规划、会话解读、动作验证与基于反馈的再规划；②使用 LLM 作为可行推理与规划的可扩展近似器；③通过规则验证器确保物理可执行性，避免 LLM 生成不可执行动作。

**🔧 技术方法**

核心技术包括：大语言模型 (GPT‑5.4 nano)、Bayesian 逆规划（ToM 推理）、层次化规划 (高层目标 → 低层动作)、对话自然语言处理、规则式动作验证与反馈再规划。

**📊 数据集**

实验使用 La Boca 合作构建任务的数据集，包含三维网格工作空间、九种多彩积木以及两种不同的二维目标视角。

**📈 对比分析**

与无 ToM 的 LLM 及离线训练的多智能体强化学习 (MaskablePPO) 基线比较。人机实验中，LLM+ToM 的交互步骤平均为 5.2 步、完成率 100%，信任得分最高；LLM 仅 6.4 步、完成率 100%；RL 仅 10.2 步、完成率 20%（仅 1 例完整完成，4 例中途退出）。

**⚠️ 局限性**

限制包括：仅 5 名受试者的 pilot 规模；ToM 更新频率低（每 2 步一次）；实验任务空间和动作集有限；RL 基线在私有信息场景下表现不佳；未对 LLM 置信度进行校准或动态澄清。

---

## 206. TRCoRSurg: Temporal-Relational Co-Reasoning for Surgical Video Triplet Recognition

**arXiv ID:** 2608.04606 | [PDF](https://arxiv.org/pdf/2608.04606v1)

**作者:** Fang Li `[一作]` (Beihang University), Aimin Hao `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种统一框架，用多尺度编码器、CAM、GCN、双记忆注意力以及双向时空融合注意力（BTRFA）实现外科手术视频中的三元组（instrument‑verb‑target）识别；

**💡 创新点**

创新点包括：①通过多尺度CAM引导的MS‑CAMRE模块实现节点与边级联的标签关联建模；②采用双向时空融合注意力实现时间与关系的共推理；③提出Triplet Consistency Error Rate（TCER）度量评估三元组一致性；

**🔧 技术方法**

技术手段涵盖多尺度特征编码、类别激活映射（CAM）、图卷积网络（GCN）、双记忆注意力、跨注意力融合、标签关联建模、双向时空融合注意力、耦合损失以及TCER评估；

**📊 数据集**

使用了CholecT45和ProstaTD两个公开手术视频数据集；

**📈 对比分析**

与RIT、MT4MTLKD、SDSwin、RDV、CoLSurgical等SOTA方法对比，AP_IVT提升约5–8%，TCER显著降低36%/25%，在AP、Top‑K和TCER等多项指标上均达成最优或接近最优性能；

**⚠️ 局限性**

模型结构复杂、训练困难，难以满足实时部署需求，未来需要探索轻量化和自适应记忆更新以提升实用性。

---

## 207. The Calibration Floor: Format Repair Can Masquerade as Self-Correction at Small-to-Mid Scale

**arXiv ID:** 2608.04355 | [PDF](https://arxiv.org/pdf/2608.04355v1)

**作者:** Mingguang Chen `[一作]` (DeepGrounding), Licheng Wang `[通讯]` (AlphaAvatar)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在多种大型语言模型上系统测量自我修正（self‑correction）对答案准确率的影响，发现大部分表观的准确率提升或下降主要源自答案可解析性（format）问题，而非模型实际内容推理的改变；作者提出了精确的“margin decomposition”方法，将总误差拆分为内容增益、格式恢复与格式损失三项，并进一步通过强制语法约束重提取的因果控制验证该分解的有效性。

**💡 创新点**

创新点包括：① 首次给出完整且可复现的自我修正误差分解公式，能够准确区分内容修正与格式错误；② 引入“校准底线（calibration‑floor）”判定标准，对自我修正的可利用性进行量化；③ 通过强制解析重生成的因果对照实验，实证展示格式问题能在一定程度上“遮蔽”真实内容变化；④ 在多模型（Qwen3.5、Gemma‑4、Tencent Hy3、Nvidia Nemotron‑3‑Ultra‑550B）与多任务（GSM8K、MMLU、MATH、Code、TriviaQA 等）上复现并跨规模验证该现象。

**🔧 技术方法**

技术手段主要包括：冻结轨迹（frozen‑trajectory）离线评估、答案提取器（extraction gate）、可解析性门控、margin decomposition、基于AUROC的信号判定、δ‑floor 校准底线、受限解码（constrained decoding）实现因果控制、GEE 逻辑回归对尺度效应的检验、以及多种统计检验（Wilcoxon、Mann‑Whitney、Bootstrap、Benjamini‑Hochberg）。

**📊 数据集**

使用的数据集包括：GSM8K（算术/数值推理）、MMLU（多学科多选）、MATH（开放式数学题）、HumanEval/MBPP（代码评测）、TriviaQA（短文本答案）、CommonsenseQA、TruthfulQA‑MC1、ARC（抽象推理）等；所有数据均采用标准 dev/holdout 分割，确保评测与验证的独立性。

**📈 对比分析**

比较方法：在同一模型–任务组合上，对照 ALWAYS（始终修正）、NEVER（从不修正）、随机门控、oracle（最佳门控）等基线；使用自定义的信号（A、A′、B、C）进行信号排序，评估门控的收益（gain）与遗憾（regret）。实验结果显示：虽然整体 Δ_total 往往为正值，经过 margin decomposition 后可见内容增益 Δ_content 在 4B–12B 规模下基本为零，格式恢复 Δ_format-recover 贡献了大部分正向效应；在 0.8B/2B 规模下内容增益虽为正但往往伴随更高的内容损失，整体收益有限；因果控制实验表明强制解析后可将原始 Δ_total 约 71% 缩小至 Δ_content，验证了格式误差对总误差的主导作用。

**⚠️ 局限性**

局限性：① 研究未正式预注册，部分门控与实验设计在数据观察后进行调整；② 引入的 extraction‑completeness gate 与 forced‑continuation 解析器基于后验发现，可能影响实验的可推广性；③ 因果控制实验仅在格式可解析率足够高的细胞中展开，未能在所有细胞中完全消除格式效应；④ 在极大规模（≈55B）模型上验证的 frontier arm 受 API 限制与数据缺失的影响，结果不够完整；⑤ 信号判定与 AUROC 的 binormal 假设在某些细胞中偏差较大，导致 δ‑floor 估计出现误判；⑥ 仅评估了固定的自我修正流程（review prompt + 再推理），未涵盖更复杂或多轮迭代的自我修正机制；⑦ 未能系统探究不同模型架构、参数量与提示策略对内容增益与格式误差的交互作用。

---

## 208. MCHA: A Memory-Centric Hierarchical Architecture for Parallel-Sequential Computing

**arXiv ID:** 2608.04443 | [PDF](https://arxiv.org/pdf/2608.04443v1)

**作者:** Daijing Shi `[一作]` (Peking University), Bonan Yan `[通讯]` (Peking University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种内存中心化分层架构MCHA，利用多层P2P NoC和数据驱动触发编程模型，将主存访问瓶颈迁移到局部通信，从而显著提升并行-顺序计算任务（如多代理强化学习、神经网络和图处理）的执行速度。

**💡 创新点**

创新点包括：去中心化全局缓冲，构建三层P2P NoC实现局部化通信；采用事件驱动的触发式编程模型隐藏数据传输延迟；基于RISC‑V实现可重构MCC并提供开源周期精确模拟器；通过多芯片可扩展设计实现从4片到32片的近线性加速。

**🔧 技术方法**

使用技术：28nm TSMC CMOS工艺；Memory‑Centric Cores (MCC) 与 Processing Blocks (PB)；双层MMIO FIFO触发；多层NoC（层1局部、层2块间、层3芯片间）与异步握手；RISC‑V ISA支持触发器；开源Cycle‑Accurate模拟器；UCIe/PCIe等高带宽跨芯片接口。

**📊 数据集**

数据集/基准：多代理强化学习（MPE Simple Spread、StarCraft II、Switch Riddle、STORM）；大规模神经网络（2% MVC）；BSP图处理（PageRank、BFS、Markov Random Field）；均使用与基线相同的模型参数、邻域大小和动作空间。

**📈 对比分析**

对比方法：与NVIDIA A100 GPU、RTX3090 GPU、FPGA、以及专用加速器（PEARL、ActiveN、Dalorex、PolyGraph）进行性能基准；4片MCHA在MARL任务上比A100获得153–2457×加速，32片进一步提升1.22–4.25×；MVC与BSP任务分别获得5–6×加速；DRAM访问率从96%降至5.44%，功耗仅115 mW、面积2.92 mm²。

**⚠️ 局限性**

局限性：架构高度依赖固定拓扑的局部性，对高度动态的图拓扑支持有限；迁移现有CPU/GPU代码到MCHA的编程模型仍需手工重写，缺乏自动化工具。

---

## 209. Real-time probabilistic tsunami forecasting via generative AI

**arXiv ID:** 2608.04327 | [PDF](https://arxiv.org/pdf/2608.04327v1)

**作者:** Yusuke Oishi `[一作]` (Fujitsu Limited), Fumihiko Imamura `[通讯]` (Tohoku University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一个基于条件扩散模型的概率性实时海啸淹没预测系统，可从海上观测波形即时推断陆面淹没深度和范围。

**💡 创新点**

创新点在于将生成式AI扩散模型与输入波形扰动相结合，实现既高精度又经过校准的概率预测，克服传统确定性预报的误导性边界和缺乏淹没信息的问题。

**🔧 技术方法**

使用了条件潜在扩散模型（LDM）、Transformer编码器提取波形特征、输入波形扰动与扩散噪声的协同生成、以及基于物理数值模拟的约2000个训练样本。

**📊 数据集**

采用了2011年东北大地震的合成与实测海浪波形（S-net、NOWPHAS）、对应的物理模拟洪涝结果，以及现场调查的淹没深度和建筑损毁数据。

**📈 对比分析**

通过200个独立物理模拟测试集和真实案例进行比较，使用CRPS、Spread‑Skill Ratio、RMSE、相对经济价值等指标评估。组合模型在校准（≈1.0）、预测精度最高且推断速度仅约38秒，优于仅扩散或仅扰动的基线模型。

**⚠️ 局限性**

局限性在于预测性能高度依赖训练数据的多样性和分布，需要扩展更多断层源情景；在极稀疏观测条件下预测不确定性仍较大，且对局部设施的高分辨率预测仍需进一步提升。

---

## 210. OmniRouting: A Semantic-Coupled Multimodal Benchmark for Constraint-Aware Spatial Reasoning in PCB Routing

**arXiv ID:** 2608.04434 | [PDF](https://arxiv.org/pdf/2608.04434v1)

**作者:** Taiting Lu `[一作]` (Pennsylvania State University), Mahanth Gowda `[通讯]` (Pennsylvania State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个大规模 PCB 路由基准（包含 1,681 个工业级、与原理图耦合的 PCB 设计及其人工参考路由），并对多种大语言多模态模型（LLMs）在零射击与工具增强的迭代设置下进行评估。

**💡 创新点**

①首次提供真实工业级 PCB 路由基准及完整的评估协议；②引入多维度评估指标（连通性、设计规则合规、功能性和工具增强的迭代改进）；③通过对比 LLM 与人工工程师以及现有商业路由器，揭示 LLM 在复杂约束下的显著不足。

**🔧 技术方法**

使用 LLM（GPT‑5.5、Claude‑Opus‑4.8 等）、图像+文本多模态提示、可视化工具、重叠检查和路由可行性检查等技术；基准构建依赖 OmniSch/OmiLayout 等自动化流水线。

**📊 数据集**

OmniRouting 数据集：1,681 个包含原理图、元件布局、堆叠信息、约束规则和人工参考路由的工业级 PCB 设计，涵盖 2–8 层板、77,242 个元件、168,815 个焊盘、55,830 个网络等。

**📈 对比分析**

与人工工程师及商业路由器（如 PcbRouter、GPCB）对比，零射击 LLM 的路由连通率仅为 12.6%/15.4%，而人类为 93.6%/95.9%；工具增强后稍有提升（≈28%/30%）。在 DRC 违规、开放/短路率等指标上，LLM 明显落后于人类和现有自动路由器。

**⚠️ 局限性**

主要局限：①多路网络路径规划能力弱；②难以满足设计规则、层与 via 分配；③在拥塞处理和保持电气连通性方面表现不佳；④对真实工业 PCB 复杂约束的理解不足，导致生成的路径与参考设计差异大。

---

## 211. CURATE: Leveraging LLM Agents to Compose, Catalog, and Deploy Reproducible Workflows

**arXiv ID:** 2608.04270 | [PDF](https://arxiv.org/pdf/2608.04270v1)

**作者:** Nolan Cutler `[一作]` (Oregon State University), Renato Figueiredo `[通讯]` (Oregon State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个人机交互的多智能体系统CURATE，利用大型语言模型生成、测试、部署完整的科学工作流，并通过模块目录实现工作流组件的复用与共享。

**💡 创新点**

①把工作流生命周期（草图、实现、测试、部署、回顾）纳入多智能体框架；②引入模块目录支持FAIR原则的复用与分享；③采用HITL门控流程保证质量；④结合无状态服务器化执行与资源隔离。

**🔧 技术方法**

使用Claude Opus 4.8 + Claude Agent SDK，LangGraph实现智能体协作；FaaSr作为无状态工作流管理系统；BM25关键词检索模块目录；GitHub Actions做部署；Python实现原型。

**📊 数据集**

SeBS-Flow 基准套件（4个工作流），PyADM1 7 步预处理+模拟流程，PDF 单词提取、视频识别、Faster R‑CNN 预训练模型等公开数据集。

**📈 对比分析**

与单一Claude Code基线对比，采用5个HITL门控。实验6个任务全部成功（无运行错误），基线成功率低于50%；token使用约高2倍，但在模块复用阶段（E5‑E6）token下降近一半，显示复用带来效率提升。

**⚠️ 局限性**

实验范围有限：SeBS-Flow为性能基准而非科学工作流；PyADM1仅覆盖预处理，未完成完整模型运用；手动修正FaaSr细节仍必要；缺乏跨学科用户研究与完整资源隔离安全评估。

---

## 212. Simile Understanding in Text-to-Image Models: An Evaluation Framework

**arXiv ID:** 2608.04750 | [PDF](https://arxiv.org/pdf/2608.04750v1)

**作者:** Luecheng Wang `[一作]` (University of Tokyo), Taro Watanabe `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估文本到图像模型对拟言（simile）的理解，构建可控数据集并引入YOLO检测与Diffusion Lens层分析的完整评测框架。

**💡 创新点**

①用对象检测自动衡量拟言文字化倾向；②通过Diffusion Lens揭示隐喻在文本编码器层次中的出现与消退；③比较随机重生与层基重生两种减轻文字化倾向的策略。

**🔧 技术方法**

YOLO对象检测、CLIPScore/PickScore、Diffusion Lens、LLM生成与筛选、随机/层基重生等技术。

**📊 数据集**

基于80类YOLO可检出对象、14种拟言模板共9,108句候选，最终筛选得到1,576句G5级拟言的文本数据集。

**📈 对比分析**

使用五个不同文本编码器结构的t2i模型（Dreamlike、PixArt、FLUX、SD3.5、Qwen-Image）在相同prompt下生成图像，利用YOLO检测率、CLIPScore、PickScore以及人工Q2/Q3评分进行对比。性能上，YOLO-Det最高的Qwen-Image为0.614，最低的Dreamlike为0.298；随机重生可将误检率降至0.244–0.124，层基重生效果因模型而异。

**⚠️ 局限性**

仅衡量对象文字化倾向，未评估属性转移的完整性；检测不到对象并不代表隐喻被正确解释；框架受YOLO类别限制，难以覆盖更抽象的隐喻。

---

## 213. Echoes in the Sky: Computational Thematic Analysis of Online Public Discourse on Bluesky Across Trump's Reelection

**arXiv ID:** 2608.04120 | [PDF](https://arxiv.org/pdf/2608.04120v1)

**作者:** Qile Wang `[一作]` (University of Delaware), Matthew Louis Mauriello `[通讯]` (University of Delaware)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

分析Bluesky上关于特朗普的帖子，构建微主题与宏观主题，比较与特朗普第二任期行政命令的主题分布，并对情感与时间动态进行统计与结构断点分析。

**💡 创新点**

1）提供最大公开的特朗普相关Bluesky数据集（38.5 M 帖子）；2）提出可扩展的LLM+人工验证主题分析框架；3）将在线讨论主题与行政命令主题对齐，揭示政策事件对讨论的即时影响；4）使用结构断点检测阐明讨论的事件驱动性。

**🔧 技术方法**

微主题聚类（Qwen3‑Embedding‑8B + UMAP + HDBSCAN）、LLM标题/摘要生成（30B‑A3B‑Instruct‑2507）、VADER情感分析、R breakpoints 包做结构断点分析，配合人类标注验证。

**📊 数据集**

38.5 M 条Bluesky帖子（2019‑2026）和 258 条特朗普第二任期行政命令。

**📈 对比分析**

通过主题分布比对、情感占比与时间序列的结构断点，发现主题在时间上与行政命令签署时刻对齐，负面情绪持续上升；聚类性能指标为 Silhouette ≈0.52‑0.53，Trustworthiness ≈0.95。

**⚠️ 局限性**

仅覆盖英语帖子，VADER 不能区分支持与反对情绪，未做因果推断，缺乏多语言、用户层面和立场分析。

---

## 214. EuroExec: Frontier Language Models Fall Short of Expert Judgment on European Executive Decision Tasks

**arXiv ID:** 2608.04549 | [PDF](https://arxiv.org/pdf/2608.04549v1)

**作者:** Pau Arnal `[一作]` (Sovrano AI), Marcus A. Castro `[通讯]` (Sovrano AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了欧盟高管决策支持基准EuroExec，邀请47位专业专家设计413道开放式长篇问题，并使用4,000+人工评估小时对六款前沿LLM在这些任务上的表现进行人工评估，提出“Solve Rate”指标；

**💡 创新点**

创新点在于：①针对真实职业场景设计高质量开放式长文本问题集；②通过人工专家多维度评估（rubric、checklist、排名）建立客观指标；③系统性比较前沿模型与人类专家的差距，证明人类评估仍是不可替代的基准；

**🔧 技术方法**

采用了手工设计的评价仪器（Likert rubric、特定checklist、偏好排序）以及自动化预筛选管线（Claude Sonnet 4.6、Gemini Flash 3.5等）对模型输出进行验证；

**📊 数据集**

EuroExec数据集：413道长文本问题，涵盖金融、市场、商业、产品四大领域，题目均来源于真实案例并附带专家编写的checklist；

**📈 对比分析**

对比方法：使用Solve Rate、平均排名、胜率、平均rubric分数和checklist满足率等多指标；实验结果显示最高模型Fable 5的Solve Rate仅56.9%，人类专家超过92%，其余模型更低；

**⚠️ 局限性**

局限性包括：仅为33道问题提供专家理想答案；评估成本高、可重复性受限；缺乏对模型失效模式的定性分析；仅使用单一LLM做AI评审，偏差未全面探究。

---

## 215. UniWorld-View: Large-Baseline View Synthesis via Video Diffusion Models

**arXiv ID:** 2608.04701 | [PDF](https://arxiv.org/pdf/2608.04701v1)

**作者:** Haiyang Zhou `[一作]` (Peking University), Li Yuan `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种统一框架，利用可遮挡点云渲染与双流条件视频扩散模型实现从单目图像或视频进行大基线可控新视图合成。

**💡 创新点**

创新点在于通过三重重投影与基于法向的可见性校正实现遮挡感知点云渲染，并构建双流（几何渲染+原始视图）扩散条件体系，既保证几何一致性又保留原始视图纹理。

**🔧 技术方法**

核心技术包括：VideoDiffusion + DiT（VACE）扩散框架；前向几何估计（深度、相机姿态、法向）；三重重投影与法向过滤的遮挡消除；双流条件（Context Block + Ref-DiT）与参考注入；自监督动态数据与静态多视角数据混合训练；4D Gaussian Splatting 复原。

**📊 数据集**

使用了 OpenVid-1M（自监督动态样本）、DL3DV、RealEstate10K（静态多视角样本）进行训练；评估数据包括 WorldScore 基准、RealEstate10K、CO3D、DL3DV140 等零样本新视图合成数据集。

**📈 对比分析**

在零样本 NVS 和 WorldScore 上与 See3D、GEN3C、Uni3C、SEVA 等基线进行对比。结果显示在 PSNR/SSIM 上在所有数据集均取得最优或接近最优，在 LPIPS 上在 CO3D 上取得最佳，在 DL3DV 上排名第二，同时在可控性、几何一致性和视觉质量上均优于对手。

**⚠️ 局限性**

局限性：依赖前端几何估计的精度，对极端遮挡或快速动态物体仍存在误差；训练需要大规模多视角和自监督数据，推理时仍需较高算力；在纹理细节上仍有提升空间。

---

## 216. Emotion Dynamics in Social Deception Games: Analysis of Professional and Nonprofessional Players through Electrodermal Activity in Werewolf Games

**arXiv ID:** 2608.04605 | [PDF](https://arxiv.org/pdf/2608.04605v1)

**作者:** Sho Mitarai `[一作]` (Kyoto University), Nagisa Munekata `[通讯]` (Kyoto Sangyo University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对比专业与非专业狼人游戏玩家，使用电导率（EDA）测量情绪波动，并对游戏过程中的言语进行分类分析，探讨不同专业水平玩家的情绪调节与沟通策略。

**💡 创新点**

首次将生理测量与言语内容结合，系统比较专业玩家与非专业玩家在高情绪激发时的情绪表现和沟通方式，并揭示专业玩家倾向使用说服性表达，非专业玩家倾向信息管理。

**🔧 技术方法**

使用EDA采集设备（BIOPAC、Q‑sensor），对皮肤电导率（SCL）与皮肤电反应（SCR）进行统计分析；采用混合因素ANOVA和Mann‑Whitney U检验进行组间与条件间比较；对发言按五大类（说服/断言、信息管理、情绪操纵、方向指示、未分类）进行手工标注。

**📊 数据集**

收集12名专业玩家（12局）和10名非专业玩家（10局）的游戏录像、音频与EDA数据，总计约220条发言（专业60条、非专业50条）用于情绪峰值与随机段落的比较。

**📈 对比分析**

通过两组对照的混合ANOVA发现专业玩家在早期讨论和投票阶段的SCL低于非专业玩家；SCR频率亦显著较低；Mann‑Whitney U检验显示专业玩家在高情绪时说服性表达比例显著高于随机段落，且与非专业玩家相比显著差异（p<0.01）。

**⚠️ 局限性**

局限性包括：EDA设备差异可能影响绝对值；样本仅限日本日本人，文化可推广性受限；样本量小且专业与非专业组人数不平衡；分类由单一研究者完成，缺乏多评审验证；二元专业/非专业划分忽略了中间经验层次。

---

## 217. Cost-Aware Multi-Objective Bandits: Theory and Application to Budgeted LLM Configuration Evaluation

**arXiv ID:** 2608.04333 | [PDF](https://arxiv.org/pdf/2608.04333v1)

**作者:** Bo Xue `[一作]` (City University of Hong Kong), Shuang Qiu `[通讯]` (City University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在有限预算下，考虑评估成本和多目标的LLM配置评估，并将其建模为成本感知多目标bandit问题，提出在线配置选择和固定预算Pareto识别两种算法。

**💡 创新点**

创新点：①首次将评估成本与多目标指标统一纳入bandit框架；②提出基于超体积/成本比的CoHV-UCB和成本感知的CoPSI；③给出对应的理论上界（预算对数律、指数误差下降）。

**🔧 技术方法**

采用UCB+置信区间、超体积指标、成本下界、经验间隙消除、固定预算Pareto识别的successive elimination及理论分析等技术。

**📊 数据集**

使用的数据集为GSM8K、PIQA及其从中挑选的14-arm子集。

**📈 对比分析**

与成本无关的HV-UCB、Accuracy-Cost-UCB、轮询、PSI-SR和均匀分配等基线比较；CoHV-UCB在超体积效率上比基线提升约60–80%，CoPSI在误差概率上比均匀分配降低约45–85%。

**⚠️ 局限性**

局限性：仅将每个配置视为独立 arm，未利用模型/提示结构；仅处理静态非上下文环境；当目标间隙很小或早期估计不可靠时性能受限。

---

## 218. When does training on downscaled images yield the same gradients?

**arXiv ID:** 2608.04448 | [PDF](https://arxiv.org/pdf/2608.04448v1)

**作者:** Seunghyun Ji `[一作]` `[通讯]`, Seunghyun Ji

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究在训练扩散变压器时使用低分辨率图像步骤是否能保持与原始梯度相同的方向，并提出可测量的梯度降级度量。

**💡 创新点**

提出两种梯度降级解释：光谱基准（仅关注失去的频段）和梯度扰动模型（将降级分解为频段误差与基于标记数的图形“底线”两项），并给出可校正的无偏估计方法。实验验证该模型可预测未见路由的梯度差异，显著优于传统光谱解释。

**🔧 技术方法**

使用基于Diffusion Transformer（DiT）的2B参数模型（Anima），配合Qwen-Image VAE、LoRA适配器、旋转位置嵌入、分辨率桶化、频段化旋转对齐等技术。

**📊 数据集**

在由512、768、896、1024、1280四个边长组成的图像桶化数据集上进行实验，主要使用插画语料库（两套：60张、15张不同风格）。

**📈 对比分析**

通过对比不同分辨率下的梯度余弦相似度、训练时间与权重空间余弦，验证了在 1024→896（σ∈(0.5,0.94]）和 1024→768（σ∈(0.65,0.95)）下的安全使用。实验显示在固定步数下可实现约14.6%的训练时间节省，权重空间余弦在晚期调度下可达0.75，近似与完整训练一致。

**⚠️ 局限性**

局限性包括：仅在单一DiT模型和单一噪声水平下验证；梯度级别的评估未涉及最终生成质量；估计器的分辨率受样本数量限制；对不同模型、数据集或更高分辨率的泛化性未完全验证。

---

## 219. AgentForge: An Immersive Role-Playing Platform for Learning Agentic Software Engineering

**arXiv ID:** 2608.04148 | [PDF](https://arxiv.org/pdf/2608.04148v1)

**作者:** Zihan Fang `[一作]` (Vanderbilt University), Yu Huang `[通讯]` (Vanderbilt University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出AgentForge，一种让初学者在多代理软件修复流程中扮演角色的沉浸式学习系统；

**💡 创新点**

创新点在于通过角色化工作流、结构化交接与元认知支持，使学习者能可视化代理决策、主动评估产出并提升与Agent协作的批判性思维；

**🔧 技术方法**

技术实现基于LLM（GPT‑5）构建四个专门代理（计划者、补丁作者、评审、测试）与AI Coach辅助提示；

**📊 数据集**

实验使用BugsInPy benchmark中的四个真实修复任务作为数据集；

**📈 对比分析**

通过对比四角色的交互指标、任务完成率和补丁相似度，发现任务规划/补丁作者/测试完成率均超过90%，但评审角色交互量与完成时间最高；总体学习效果显著提升；

**⚠️ 局限性**

局限性包括：界面信息密集导致学习者对指令模糊、对AI输出过度依赖、缺乏足够的批判性评估练习，以及样本规模相对有限。

---

## 220. SJEPA: Learning Elegant Latent Dynamics with Hybrid Symbolic-Neural Predictors

**arXiv ID:** 2608.04060 | [PDF](https://arxiv.org/pdf/2608.04060v1)

**作者:** Yongchao Huang `[一作]` `[通讯]`, Yongchao Huang

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种无重建的 Joint-Embedding Predictive Architecture (SJEPA)，在潜在空间中学习由符号定律和可正则化神经纠正组成的转换。

**💡 创新点**

创新点在于将表示学习与动态压缩结合为受限操作压缩框架，既保证潜在状态可预测又获得最简易的符号动力学，同时通过正则化控制符号与神经部分的分配。

**🔧 技术方法**

使用可微稀疏符号库、符号回归、神经纠正正则化、信息瓶颈（VICReg）以及可选的高斯过程残差模型。

**📊 数据集**

实验数据来自仿真的控制摆子系统（两维状态被混合为32维观测），并使用多步滚动和离群轨迹评估。

**📈 对比分析**

与传统神经JEPA、后置符号回归以及未正则化混合模型比较，联合学习将符号复杂度从约26降低到4.7，物理状态滚动误差和发散率分别降低约3倍以上，证明了操作压缩对可解释性和长期预测的正面影响。

**⚠️ 局限性**

局限性包括仅在受控、低维仿真数据上验证；未在高维视频、真实控制任务或不确定环境中评估；符号表达的可解释性受坐标变换非唯一性的影响，且需要手动选择符号词典和正则化参数。

---

## 221. PADFormer: Pose-agnostic Anomaly Detection from Sparse View Images

**arXiv ID:** 2608.04210 | [PDF](https://arxiv.org/pdf/2608.04210v1)

**作者:** Ruiqi Wang `[一作]` (Amazon), Jing Huang `[通讯]` (Amazon)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出PADFormer，一种利用Vision Transformer在二维图像空间直接进行稀疏视角姿势无关异常检测的方法；

**💡 创新点**

创新点包括：①无须3D重建、直接跨视角掩码重建无缺陷图像；②动态补丁选择与空间变换网络提升跨视角匹配；③交叉注意解码器引入可学习的无异常先验；④多次随机掩码推断与方差一致性过滤增强鲁棒性；

**🔧 技术方法**

使用技术包括：Vision Transformer、Masked Image Modeling、Spatial Transformation Network、交叉注意解码器、可学习的无异常 tokens、CIELAB误差度量、Gaussian滤波、两阶段训练（within-view + cross-view）；

**📊 数据集**

实验数据集：PAD任务采用MAD‑SIM和PIAD；FSAD任务采用MVTec‑AD和VisA；

**📈 对比分析**

与3D重建基线（OmniAD、SplatPose、PIAD）以及FSAD基线（PatchCore、UniVAD等）对比，PADFormer在2/4/10 shot PAD任务中图像AUROC提升20%以上，像素AUROC提升8%以上；在FSAD任务中保持竞争力；

**⚠️ 局限性**

局限性：固定补丁尺寸对大缺陷效果不佳；需要多次推理（15次）以获得完整覆盖；对极端视角变化仍有提升空间。

---

## 222. iStructTab: Structured Feature Sequencing for Multimodal Learning of Image and Tabular Data

**arXiv ID:** 2608.04348 | [PDF](https://arxiv.org/pdf/2608.04348v1)

**作者:** Al Zadid Sultan Bin Habib `[一作]` (West Virginia University), Donald A. Adjeroh `[通讯]` (West Virginia University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种将图像与表格数据融合视为特征序列化问题的多模态学习框架iStructTab。

**💡 创新点**

创新点在于把特征排序问题建模为列排列问题（CPP），通过Graph-Enhanced Descriptor Sequencing (GEDS) 生成结构化的特征序列，并在Order-Aware Efficient Transformer with Memory Augmentation (OEMT) 中显式利用该序列与记忆令牌实现更稳定、鲁棒的融合。

**🔧 技术方法**

采用的技术包括统计描述子与图卷积的相似性传播来构建特征图、Linformer型Transformer的高效编码、以及基于顺序的损失正则化。

**📊 数据集**

使用了六个公开图像-表格混合数据集：DVM、HAM10000、Deep Lesion、Pokémon、CheXpert 以及 Pet Finder。

**📈 对比分析**

与多种单模态和多模态基线（如STiL、TIP、DAFT、MMCL、ViT等）比较，iStructTab 在准确率、平均排名和平均惩罚上均取得领先，并在噪声、效率和鲁棒性实验中表现优异。

**⚠️ 局限性**

局限性包括特征图构造的O(m²)复杂度在大规模特征维度时可能成为瓶颈，以及对超参数（如记忆令牌数量、池化长度）的依赖需要进一步系统化研究。

---

## 223. Combating Knowledge Corruption in Agent Systems: A Byzantine-Tolerant Secure Collaborative RAG Framework

**arXiv ID:** 2608.04366 | [PDF](https://arxiv.org/pdf/2608.04366v1)

**作者:** Zhaoqi Wang `[一作]` (Beijing Institute of Technology), Liehuang Zhu `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SecureCollaRAG框架，对RAG系统进行多源知识验证，以抵御知识腐败攻击。

**💡 创新点**

首次结合多源知识验证与动态图神经网络，实现拜占庭容错的文档可信度评分；提出Adaptive Tampering Attack作为更隐蔽的攻击方式。

**🔧 技术方法**

分布式检索协议、动态知识图构建、GNN可信度评分、通用常识审计、指数平滑来源可信度更新。

**📊 数据集**

自然问题(NQ)、HotpotQA、医学MediNote、金融Finance Alpaca四个数据集。

**📈 对比分析**

与SS-RAG、MS-RAG、RobustRAG、FilterRAG对比，攻击成功率显著下降，尤其在毒化和ATA攻击下，SCR的ASR低于5%-6%，性能优于传统方法。

**⚠️ 局限性**

依赖多源协同和可信度阈值；在恶意来源接近50%或更强同源攻击时鲁棒性下降；需进一步研究跨领域适配与低资源环境。

---

## 224. Interpretable Fuzzy Inference for UAV Target Tracking Using Bounding-Box Geometry

**arXiv ID:** 2608.04121 | [PDF](https://arxiv.org/pdf/2608.04121v1)

**作者:** Reza Ahmari `[一作]` (North Carolina Agricultural and Technical State University), Abdollah Homaifar `[通讯]` (North Carolina Agricultural and Technical State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用 YOLO 边框几何信息（中心位置、面积、宽高比）构建可解释的模糊推理框架，实现无人机对移动地面目标的连续偏航角估计。

**💡 创新点**

创新点在于：① 在训练集上严格控制特征归一化与隶属函数参数，避免测试泄漏；② 采用相同的肩-三角-肩隶属函数分区，构建 27 条规则的 Mamdani 与 Takagi–Sugeno 结构，直观展示规则对应关系；③ 通过最小二乘识别线性后效器，使 Sugeno 模型在保持解释性的同时获得接近真实的连续偏航角。

**🔧 技术方法**

主要技术包括：YOLO 边框检测、特征归一化、Mamdani 模糊控制（基于极值隶属函数与质心解算）、第一阶 Takagi–Sugeno 模糊推理（隶属函数、产品归一化、线性后效器），以及训练/测试分割和误差评估指标。

**📊 数据集**

数据集：6169 张标注图像，来自室内 VICON 运动捕捉环境；UAV 静止，UGV 在不同位置和朝向行驶；每帧提供 YOLO 边框坐标及 VICON 计算出的真实偏航角。

**📈 对比分析**

与 5 种低维回归基线（线性、Ridge、SVR、随机森林、MLP）以及 Mamdani 进行比较。Sugeno 模型在 MAE 0.14°、RMSE 0.20°、最大误差 1.25°、±1° 区域准确率 99.68% 上显著优于基线；Mamdani 的方向一致性最高（约 91%），但误差大。计算量上 Sugeno 的推理时间仅 0.0005 ms/样本，低于 SVR、随机森林，接近线性回归。

**⚠️ 局限性**

局限性：仅在静态 UAV、单帧特征的实验环境；未评估闭环飞行或 UAV 自身运动对边框特征的影响；缺乏对不同相机、光照或户外环境的鲁棒性验证；暂未加入时序信息或更细粒度的视觉特征。

---

## 225. Transferable Dual-Stream Representations for Mesoscale-Preserving Sea Surface Temperature Downscaling

**arXiv ID:** 2608.04230 | [PDF](https://arxiv.org/pdf/2608.04230v1)

**作者:** Parth Doshi `[一作]` (Dalhousie University), Gabriel Spadon `[通讯]` (Dalhousie University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并训练了一种物理信息化的表示学习框架，实现公里尺度海面温度（SST）下采样。

**💡 创新点**

创新点包括：①双流结构将气象驱动力与海洋记忆在各自时间尺度上分离；②在海洋流中使用相对位置编码（RoPE）以提高跨域泛化；③将谱加权损失与DDIM扩散生成器结合，既保持精度又保留细尺度方差；④在零样本与少样本设置下实现不同海盆的迁移。

**🔧 技术方法**

技术手段包括：Transformer空间-时间分层注意力、RoPE相对位置编码、DDIM扩散模型、谱加权损失、双流编码与融合、残差形式预测、浴深度（bathymetry）注入等。

**📊 数据集**

使用数据集为ERA5（6h气象再分析）和MUR（1km海面温度）作为目标；训练域为圣劳伦斯湾（GSL），零样本/少样本评估域为波多菲尼湾（BOF）和墨西哥湾（GOM）。

**📈 对比分析**

与ERA5持久性基线及其他对照模型比较，采用RMSE、相对技能（相对ERA5持久性）和PSD比率三项指标。结果显示：零样本RMSE下降21%，技能提升至85.6%，PSD比率≈1.00，显示模型在精度、动态信息和谱保真度方面均优于基线。

**⚠️ 局限性**

局限性包括：气象流仍使用绝对位置编码，限制了跨域泛化；实验仅覆盖西北大西洋的三个盆地，未验证在热带或极地等不同物理环境下的表现；只预测一天的SST倾向，未评估多天滚动预测的稳定性，且少样本训练易出现过拟合。

---

## 226. D$^2$F-ReAG: Dynamic Decomposition and Filtering for Multi-Hop Reasoning-Augmented Generation

**arXiv ID:** 2608.04444 | [PDF](https://arxiv.org/pdf/2608.04444v1)

**作者:** Jiaoyang Li `[一作]` (Northeastern University), Jingbo Zhu `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种动态拆分与过滤的多跳推理增强生成框架 D^2F-ReAG，能够根据根层推理的可靠性自适应地拆分问题并利用可信子问题的推理路径来修正根推理，最终生成更准确的答案。

**💡 创新点**

创新点：① 引入可靠性评估机制，让系统仅在根层推理不可靠时才拆分问题，避免过度拆分或欠拆分；② 在子问题推理后将可信推理路径回传并更新根推理，形成递进式推理链，显著降低误差传播；③ 采用早停策略，在根层已可靠时立即停止，提升效率。

**🔧 技术方法**

技术：使用检索增强生成（RAG）框架结合 dense retriever、LLM 生成器与可靠性评估器；逻辑拆分与重写通过 prompt engineering 完成；推理路径更新通过 Update 操作将子问题结果合并到根层推理；早停与阈值控制实现自适应拆分深度。

**📊 数据集**

数据集：HotpotQA、MuSiQue、2WikiMultiHopQA 三个标准多跳推理基准。

**📈 对比分析**

比较方法：在 Str-Acc（精确匹配）与 LLM-Acc（语义等价）两指标上与零样本 LLM、图结构 RAG（RAPTOR、GraphRAG、LightRAG、HippoRAG系列）以及提示式 RAG（ReAct、ChainRAG、LogicRAG）对比。实验显示 D^2F-ReAG 在所有数据集上均取得最高或接近最高的 Str-Acc 与 LLM-Acc，尤其在 2WikiMultiHopQA 上提升 5.4 分 Str-Acc 与 6.4 分 LLM-Acc。

**⚠️ 局限性**

局限性：① 对极难案例仍需多轮拆分与推理，导致推理时间与 token 消耗显著增加；② 可靠性评估依赖 LLM 的评分准确性，若评分失误可能导致错误拆分或停机；③ 对检索质量高度依赖，检索到的文档缺失关键信息仍可能影响最终答案。

---

## 227. From Non-Convex Self-Concordant Regularization to Scalable Quasi-Newton Training of PINNs

**arXiv ID:** 2608.04206 | [PDF](https://arxiv.org/pdf/2608.04206v1)

**作者:** Chenhao Si `[一作]` (Chinese University of Hong Kong, Shenzhen), Ming Yan `[通讯]` (Chinese University of Hong Kong, Shenzhen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SCORE，一种自共形启发的无 Hessian 近似量子-牛顿方法，利用移位切线几何对 PINN 训练进行高精度微调。

**💡 创新点**

创新点在于：① 用单一的 quasi-Newton 减量同时决定切线的移位大小和自共形候选步长；② 将弱自共形的正向移位直接嵌入切线更新，既实现曲率正则化，又避免显式 Hessian 计算；③ 通过强 Wolfe 检验实现曲率相对的步长自适应。

**🔧 技术方法**

核心技术包括：自共形正则化、移位切线（shifted secant）更新、Self-Scaled Broyden 迭代、强 Wolfe 条件、Adam warm‑start 与块式训练。

**📊 数据集**

在四个经典 PDE 基准上进行评估：粘性 Burgers、Kuramoto–Sivashinsky、Korteweg–de Vries、复 Ginzburg–Landau，全部使用由数值求解器生成的高精度模拟数据集。

**📈 对比分析**

与 BFGS、SSBroyden 在相同超参数和训练预算下对比；评价指标为相对 L² 与 L∞ 错误。SCORE 在所有任务上均显著降低最终误差（Burgers 约 6×，其他任务亦有明显提升），且块级运行时间保持一致。

**⚠️ 局限性**

局限性包括：① 仍依赖切线信息，若切线不可靠（如极度稀疏或高度非线性区域）可能表现不佳；② 需要手动设定移位下限/上限及衰减因子，调参成本；③ 仅在中等维度 PDE（1‑D 与 2‑D）上验证，尚未检验在更高维或更大规模网络上的可扩展性。

---

## 228. From Compensation Design to Budget-Feasible Mechanisms: A Constant Approximation for Subadditive Valuations

**arXiv ID:** 2608.04337 | [PDF](https://arxiv.org/pdf/2608.04337v1)

**作者:** Ioannis Anagnostides `[一作]` (Carnegie Mellon University), Weiqiang Zheng `[通讯]` (Yale University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种统一的预算可行机制设计框架，利用补偿设计的潜在函数和自界函数理论，构造出一系列在不同价值类（单调子模、XOS、非单调子模、子加、β‑PHM 等）上实现常数近似的真诚机制，且在大市场情形下实现确定性常数近似。

**💡 创新点**

创新点主要包括：
- 将补偿设计中的间接机制通过潜在函数映射为直接真诚机制，首次在预算可行框架中使用自界函数和潜在函数的价格稳定性分析；
- 证明任意子加函数可由自界函数逼近至 2 倍误差（该结论独立意义，解决多选举核心问题）；
- 通过核心-尾分解、随机分区和均匀稀疏化实现对子加函数的多项式时间常数近似机制，彻底解决了长期开放的“子加函数可实现常数近似”难题；
- 在 β‑PHM（含 β‑自界函数）情形下，首次给出大市场常数近似机制，并证明在非大市场下存在不可解的上界。

**🔧 技术方法**

主要技术：
- 补偿设计的边际贡献支付规则与潜在函数的潜在稳定性（潜在函数为 F(S)∏(1‑c_i/B)_+）；
- 自界函数的自相似性与潜在函数的等价性；
- 子加函数的平滑化（smoothing）得到自界函数；
- 核心‑尾分解、随机分区、均匀稀疏化与过滤（filtering）保证预算可行性；
- 采样逼近平滑化函数以实现多项式查询；
- 通过阈值支付（Myerson）实现全局真诚性。

**📊 数据集**

该工作为理论算法，不使用任何实验数据集；所有结果均在抽象的预算可行机制模型中证明。

**📈 对比分析**

相较于先前结果，本文在信息论上显著改进：
- 单调子模从 3.798 提升至 3；
- 非单调子模从 9.742 提升至 3.718；
- XOS 从 28 提升至 3.718；
- 子加从 33 提升至 6.436；
- 在大市场情形下进一步降低到 2、e、2e。
在子加函数的多项式时间实现上，先前最佳为 O(loglog n)，本文实现常数 86.399（可任意接近 15.7083）。

**⚠️ 局限性**

限制与未解决问题：
- 与已知的下界（如 2 对于加法）仍存在差距，尤其在子加类下最优常数近似是否可达 2 仍未知；
- 随机化与确定性机制的相对优势尚未完全阐明，尤其在子加类可能存在更大分离；
- 机制实现方式为直接支付，不保证可通过时钟拍卖实现；
- 对于更一般的补偿设计模型或邻近预算可行框架（如 Bayesian、局部信息）尚未扩展；
- 平滑化得到的自界函数仅保证 2 倍近似，是否可进一步压缩仍是开放问题。

---

## 229. ExeCRE: Execution-Consistency Guided Reliability Estimation for Self-Correcting Code Generation

**arXiv ID:** 2608.04439 | [PDF](https://arxiv.org/pdf/2608.04439v1)

**作者:** Yiru Dong `[一作]` (Beihang University), Si Chen `[通讯]` (Beihang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 ExeCRE 框架，通过大规模执行一致性分析对 LLM 生成的参考代码进行可靠性估计，并将其用于自我纠错流程。

**💡 创新点**

创新点在于利用 Dawid–Skene 统计模型对执行一致性信号进行聚合，显式过滤低可靠参考代码，从而减少误导性反馈并提升迭代稳定性。

**🔧 技术方法**

核心技术包括：基于结构化输入模式的输入生成、执行输出矩阵构建、二值一致性投影、Dawid–Skene EM 推断、阈值过滤与自我纠错集成。

**📊 数据集**

主要数据集为 LiveCodeBench（182 个算法题）用于代码生成评估，并在 GSM8K 上做了小规模代码推理实验。

**📈 对比分析**

与 TextGrad、ConTested、Oracle-Guided 等多种自我纠错与验证方法对比，ExeCRE 在 GPT‑5.2、DeepSeek‑V3.2、Qwen‑32B 等模型上均显著提升 Pass@1，误导性反馈率下降至 1–10% 之间。

**⚠️ 局限性**

局限性包括对 Dawid–Skene 条件独立假设的敏感性、输入生成对模型知识截止的依赖、执行成本较高以及仅针对单个任务级代码生成的验证，尚未验证在仓库级或更复杂的交互式任务上的效果。

---

## 230. Approximate Multi-Objective Search Under Rulebooks

**arXiv ID:** 2608.04398 | [PDF](https://arxiv.org/pdf/2608.04398v1)

**作者:** Omar Muhammetkulyyev `[一作]` (Iowa State University), Tichakorn Wongpiromsarn `[通讯]` (Iowa State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种在规则书（rulebook）框架下的近似多目标搜索算法，能够在复杂的优先级关系中快速求解近似最优解集合。

**💡 创新点**

创新点在于定义了ε-规则支配（ε-rule-dominance）并证明其对规则支配具有单向传递性，从而可以在搜索过程中安全地使用近似支配关系；同时将维度压缩技术与规则书的层级结构结合，实现了高效的最佳优先搜索。

**🔧 技术方法**

主要技术包括：基于规则书的ε-规则支配判定、最佳优先（best‑first）搜索框架、维度压缩（dimensionality reduction）与分层闭集维护、以及近似规则支配下的合并与剪枝策略。

**📊 数据集**

实验使用了DIMACS 9th Challenge中的BAY道路图（321,270个节点，794,830条边）以及随机生成的多目标图（规模从几十到数百万节点）。

**📈 对比分析**

与现有算法比较：在无层级（S1）下与MO‑ε‑A*相近；在有层级且不近似（S2）下与规则书完整控制合成算法相比，速度提升达两阶；在有层级且近似（S3）下与近似MO‑ε‑A*相比，平均速度提升10–16倍，解集规模相当或更优。

**⚠️ 局限性**

局限性包括：仍具有指数最坏情况；需要设计合适的ε阈值和启发式函数；对规则书层级结构有要求，若层级复杂或无明显层级则性能不明显提升；返回的解集不保证接近最优，仅保证被某解ε-规则支配。

---

## 231. Active-SWE: Benchmarking Coding Agents for Proactive Bug Fixing without Issue Reports

**arXiv ID:** 2608.04682 | [PDF](https://arxiv.org/pdf/2608.04682v1)

**作者:** Haobin Li `[一作]` (Sichuan University), Xi Peng `[通讯]` (Sichuan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Active‑SWE 基准，用于评估大型语言模型驱动的代码代理在没有 issue 报告的情况下主动发现和修复 Bug 的能力。

**💡 创新点**

创新点包括：①将评估焦点从被动修复转向主动修复；②通过时间窗口合并多条 PR 构造多 Bug 难度任务；③设计双轨评估框架，既测量记录 Bug 的定位/修复，又评估潜在 Bug 的发现与测试验证。

**🔧 技术方法**

技术手段包括：基于 LLM 的数据标注与税onomy 共识筛选；LLM‑驱动的 Docker 环境搭建与测试脚本生成；ReAct 交互式任务模板；细粒度的 Edit‑Hunk 匹配指标（LR、LP）与测试驱动的潜在 Bug 评估。

**📊 数据集**

使用从 87 个热门开源仓库（覆盖 Python、Go、Rust、PHP、Ruby、JS/TS、Java、C/C++）提取的 1,663 条高质量 PR 构建的任务集，其中 1,411 条为单 Bug 任务，252 条为多 Bug 任务。

**📈 对比分析**

对比 10 款最新闭源与开源 LLM（Claude Opus 4.8、GPT‑5.5、Gemini‑3.1‑Pro、Qwen3.7‑Max 等），在记录 Bug 上平均定位 Recall/Precision 仅 20–35%，修复率最高仅 20%；在潜在 Bug 上通过测试生成与验证，发现 40–80% 的 Bug 具备可复现测试。

**⚠️ 局限性**

局限性在于：①主动修复任务仍极具挑战，定位与修复成功率低；②多 Bug 合并过程中难以保证完整性，导致部分难度任务失效；③潜在 Bug 的评估依赖 LLM 生成的测试，可能漏检或产生误报；④评估多侧面指标但未深入探讨模型可解释性与修复质量的多样性。

---

## 232. Representing Visual Evidence for Item Difficulty Prediction: Visual Textualization and Image-Native Modeling

**arXiv ID:** 2608.04554 | [PDF](https://arxiv.org/pdf/2608.04554v1)

**作者:** Han Chen `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Tianyi Zhou `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了在无学生响应的情况下，如何将视觉信息有效地加入到题目难度预测模型中，系统比较了仅文本、视觉文本化和原图加文本三种输入接口；

**💡 创新点**

创新点在于首次对三种实用视觉接口进行统一、系统的比较，使用多种LLM/VLM进行任务适配回归，揭示视觉信息表示对预测精度的影响并未出现单一优选方案；

**🔧 技术方法**

使用了任务适配的LoRA、全量微调、冻结特征回归和标量生成等回归技术，并结合Qwen、InternVL、PaliGemma等VLM以及GPT‑5.5、Qwen2.5‑VL‑7B等视觉文本化器进行实验；

**📊 数据集**

采用了NeurIPS 2020 Education Challenge发布的Eedi数学题库，经过筛选后共725道题（580训练，145测试），并利用Rasch模型估计的题目难度参数作为预测目标；

**📈 对比分析**

在相同训练/测试划分下对RMSE和Spearman相关系数进行对比，视觉文本化和图像原始输入分别在0.506~0.497范围内实现最低RMSE，优于仅文本的0.517，但统计检验未能给出显著排序；匹配实验表明视觉文本化能在所有文本模型上降低RMSE，图像原始输入在更广泛适配下同样受益；

**⚠️ 局限性**

主要限制在于样本规模有限（725道题）且仅为英语数学题，难度标注依赖大量学生响应，结果可能不易推广到其他学科、语言或更大规模数据集；

---

## 233. An immersive micro-manipulation system using real-time 3D imaging microscope and 3D operation interface for high-speed and accurate micro-manipulation

**arXiv ID:** 2608.04300 | [PDF](https://arxiv.org/pdf/2608.04300v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 234. Beyond Linear Dynamics: Neural Bilinear Dynamical Models for Time Series Forecasting

**arXiv ID:** 2608.04471 | [PDF](https://arxiv.org/pdf/2608.04471v1)

**作者:** Mengzhou Gao `[一作]` (Hangzhou Dianzi University), Pengfei Jiao `[通讯]` (Hangzhou Dianzi University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Neural Bilinear Dynamical Model（NBDM），结合Koopman映射、双线性隐状态-控制交互与可学习误差补偿，解决带或缺失控制输入的时间序列预测问题。

**💡 创新点**

创新点在于①使用双线性结构显式建模状态与控制的乘性交互，②引入参数化误差补偿项缓解逼近误差，③设计记忆增强反馈控制器在缺失控制输入时推断隐式控制信号。

**🔧 技术方法**

主要技术包括Koopman映射、双线性状态空间模型、误差补偿模块、记忆增强控制器、深度神经网络编码/解码、动态模式分解（DMD）与多步闭环递归预测。

**📊 数据集**

使用五个真实世界数据集：气候（Temperature）、空气质量（Seoul PM₂.₅）、交通流量/速度（PeMS-Bay、PeMS04、PeMS08）。

**📈 对比分析**

在多步（10步）预测任务中与Koopman、RNN/Transformer、图卷积等基线进行对比，NBDM在大多数数据集及长周期预测中均取得最低RMSE/MAE，尤其在缺失控制输入场景中优势显著。

**⚠️ 局限性**

局限性包括仅采用双线性形式可能不足以捕捉更高阶非线性，误差补偿项设计经验性强，对极高维或极噪声场景的鲁棒性仍需进一步验证。

---

## 235. Trace, Verify, and Correct: A Training-Free Framework for Spatial Reasoning in Multimodal LLMs

**arXiv ID:** 2608.04759 | [PDF](https://arxiv.org/pdf/2608.04759v1)

**作者:** Yang Yang `[一作]` (East China Normal University), Zhaoxia Yin `[通讯]` (East China Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种无训练、模块化的框架，用于在多模态大语言模型（MLLM）中验证并纠正空间推理中的视觉不一致性；通过构造空间证据图（SEG）提取 CoT 推理中的原子空间证据，并通过空间证据可靠性评估（SERA）判断视觉证据的可靠性，定位最早被可靠视觉证据驳斥的空间单元，指导模型重新生成受影响的推理步骤和最终答案；

**💡 创新点**

创新点在于（1）将 CoT 中的空间判断转化为可追溯的结构化图（SEG），实现证据级别的可追溯性；（2）提出多维度可靠性评估（对象存在、定位清晰度、几何测量稳定性），在验证前对视觉证据进行可靠性门控；（3）采用过程导向的纠错策略，仅针对最早出现的错误单元进行局部重生成，避免无效全链重写；

**🔧 技术方法**

技术包括：基于 MLLM 的提示式空间证据抽取、Grounding DINO 与 SAM2 结合的目标定位与分割、单目深度估计用于相对深度测量、SERA 的可靠性阈值门控、以及基于图结构的推理链重生成；

**📊 数据集**

使用了五大视觉问答/空间推理基准：LLaVA-Bench、RealWorldQA、POPE、GQA、MMHal-Bench，并在 Qwen3-VL-8B-Instruct、InternVL3.5-8B、Llama-3.2-11B-Vision-Instruct 三个主流 MLLM 上评测；

**📈 对比分析**

与 Vanilla、SpatialPIN、ByDeWay、SoM、GoM、FaithAct 等基线相比，平均提升约 8.55 个百分点（Qwen 64.33→70.59，InternVL 56.11→68.55，Llama 63.55→67.68），在 11/15 组别中取得最优或第二优成绩；

**⚠️ 局限性**

局限性包括：只能验证显式 CoT 中提取到的空间证据，无法处理隐式或缺失的空间判断；仅针对静态图像问答，未验证视频、视角多样性或封闭式本体交互；对非空间推理错误无效。

---

## 236. HRRC on the Farm: Quantile Forecasting for Highly-Reliable Remote Control via LEO Networks

**arXiv ID:** 2608.04326 | [PDF](https://arxiv.org/pdf/2608.04326v1)

**作者:** André Gomes `[一作]` (Rowan University), Jie Wang `[通讯]` (Iowa State University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在农场自动化场景下，本文将远程控制任务视为分位数预测问题，提出一种基于神经网络的高分位数估计器，以预测OneWeb LEO卫星网络中的RTT分位数，从而支持高度可靠的远程控制。

**💡 创新点**

创新点在于将分位数预测转化为条件分位数估计，利用混合Beta分布聚焦高分位数训练，克服传统均匀采样在尾部预测不足的缺陷，并在不依赖手动切换间隔或分布假设的情况下，实时预测手over期间和间隙内的RTT尾部。

**🔧 技术方法**

采用了期望pinball损失函数训练的多层全连接网络，结合Beta混合采样策略、Savitzky‑Golay滤波计算SINR导数、Z‑score和log10归一化，以及Adam优化器。

**📊 数据集**

使用了美国爱荷华州Ames市与弗吉尼亚州Ashburn之间的OneWeb网络收集的实时RTT和SINR轨迹数据，时间跨度为2025年2月-4月，共约7.6天。

**📈 对比分析**

与均匀分布(0~1)和(0.9~0.99)的基线以及无条件分位数基线进行对比。所提估计器在可靠率范围[0.9,0.99]内准确率近似理想值，并在最高可靠率99%下平均速度比无条件基线提升约138.6%，在90%和95%下分别提升19%和37.6%。

**⚠️ 局限性**

主要局限在于仅验证于OneWeb网络，未结合极值理论进行尾部加权，训练样本量约7.6天，模型对不同LEO星座的迁移性待验证，且在极端峰值下仍可能存在预测误差。

---

## 237. Taming Treewidth DP with Modulators: A General Booster for Graph Heuristics

**arXiv ID:** 2608.04446 | [PDF](https://arxiv.org/pdf/2608.04446v1)

**作者:** Jialiang Li `[一作]` (Adelaide University), Mingyu Guo `[通讯]` (Adelaide University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

未提供论文内容，无法确定具体研究工作

**💡 创新点**

无法判断创新点

**🔧 技术方法**

无法确认所使用技术

**📊 数据集**

无法确定所使用的数据集

**📈 对比分析**

无法比较方法与性能

**⚠️ 局限性**

缺乏信息限制了评估

---

## 238. EDATracer: An Agentic Framework for Large-Scale EDA Artifact Analysis

**arXiv ID:** 2608.04032 | [PDF](https://arxiv.org/pdf/2608.04032v1)

**作者:** Phat Tieu `[一作]` (Texas A&M University), Jeyavijayan Rajendran `[通讯]` (Texas A&M University)

**通讯引用:** 5951 | [OpenAlex ID](https://openalex.org/A5059126377)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了基于知识图谱与语义向量索引的Agentic框架，实现跨源EDA artifacts（源代码、脚本、日志、网表、PPA报告）的检索与推理分析。

**💡 创新点**

创新点在于：①大规模跨artifact知识图谱与向量索引结构；②将检索与LLM推理结合，形成证据基础的多步骤回答；③提供18.9 GB 2,787设计的公开数据集和90题基准。

**🔧 技术方法**

使用技术包括：知识图谱（Neo4j/类似）、语义向量检索（FAISS）、LLM Agent（RAG+Cypher查询）、多轮检索-推理交互、工业级LLM（Claude、ChatGPT、Qwen）。

**📊 数据集**

使用数据集为18.9 GB，包含2,787个可综合开源芯片设计及对应的合成日志、网表、PPA报告等共计249 K文件。

**📈 对比分析**

与Cursor、Claude Code等商业Agentic框架对比，Pass@1平均提升6.4–7.2%，Pass@5提升至约99%；平均分在8.92/10；同时在token使用上比对手低2–3倍，显著提升成本效益。

**⚠️ 局限性**

限制包括：对知识图谱构建质量高度敏感；离线预处理成本高；当前仅覆盖综合阶段，未涵盖后续floorplan、placement、routing等设计流程；对非数字/专业工业流的适配仍需改进。

---

## 239. An Explainable LLM Agent Layer for Open-World Anomaly Detection in Oil Wells

**arXiv ID:** 2608.04041 | [PDF](https://arxiv.org/pdf/2608.04041v1)

**作者:** Lucas Gouveia Omena Lopes `[一作]` (Federal University of Alagoas), William Wagner Matos Lira `[通讯]` (Federal University of Alagoas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文在油井异常检测的开放世界学习（OWL）流水线基础上，加入了一层大型语言模型（LLM）代理，用以对检测结果进行自然语言解释、验证和异常聚类命名。

**💡 创新点**

创新点在于将通用LLM（Qwen 3.5 MoE 397B）与结构化传感器指标及数据驱动的类特征向量结合，使得模型能够提供可审计的解释、独立的验证标记以及对未标记异常聚类的统一人类可读命名。

**🔧 技术方法**

采用的技术包括：结构化传感器指标计算、类特征向量构建、LLM代理推理（NVIDIA NIM接口）、三大评估实验（分类、验证、novelty detection）以及基于置信度的排名和命名输出。

**📊 数据集**

使用的公开数据集为3W油井生产异常数据集，包含九类已知异常以及未标记异常样本，共计约989个真实井段。

**📈 对比分析**

与传统的自动编码器+二分类+Mahalanobis聚类方法相比，LLM代理在三项评估中表现为：分类 top‑3 精度63.9%，验证阶段 91% 的 precision，novelty detection 89.7% 的召回率，并能为多数异常聚类提供稳定的命名。

**⚠️ 局限性**

局限性包括：对症状类（如流动不稳、产能下降）难以单独判定、对异构类9的检测率低、对下井遥测缺失的依赖导致误报、LLM未经行业微调可能产生事实性幻觉、以及样本量不足导致置信区间宽泛。

---

## 240. Cooking beyond Frames: A Stereo Event Camera Dataset in the Kitchen

**arXiv ID:** 2608.04865 | [PDF](https://arxiv.org/pdf/2608.04865v1)

**作者:** Chengming Feng `[一作]` (Delft University of Technology), Nergis Tömen `[通讯]` (Delft University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个大型、真实、前视双目事件相机厨房烹饪数据集（EventKitchen），并在此数据集上对动作识别、目标检测和立体深度估计三项任务进行基线实验。

**💡 创新点**

创新点包括：① 采用双目事件相机结合RGB、深度和IMU的多模态同步采集，实现前视自然动作记录；② 数据量达5.5小时，含10位参与者、13个厨房、10,762个动作片段（268类）和13,482个目标框；③ 提供完整标注和校准矩阵，支持多任务评估；④ 在真实环境中挑战事件视觉算法，揭示模拟与真实差距。

**🔧 技术方法**

使用的技术主要有：事件帧生成（voxel grid、时间窗口帧）、动作识别模型（TSM、Swin）、目标检测模型（YOLOv10、RVT、EvRT-DETR）、立体深度估计模型（SE‑CFF、FoundationStereo）以及事件重建方法（E2VID）和校准方法（E2Calib）。

**📊 数据集**

数据集为本研究自建的EventKitchen，包含双目事件流、同步RGB/深度/IMU数据，以及人工标注的动作段和目标框。

**📈 对比分析**

基线实验对比多种现有方法：动作识别top‑1分别为19.2%（TSM）和24.7%（Swin）；目标检测平均精度（mAP）仅为16.2%（YOLOv10），RVT和EvRT‑DETR低于10%；立体深度估计SE‑CFF在RMSE≈84–88mm、MAE≈54–59mm，低于Ground‑Truth标准差106mm，表明数据集难度高、基线表现有限。

**⚠️ 局限性**

局限性：① 动作和目标类别存在显著长尾不平衡；② 标注量相对较小，难以满足大规模预训练需求；③ 受限于前视视角，某些动作与环境依赖性强，导致模型泛化困难；④ 仅评估了部分事件算法，尚缺乏针对不同事件表示与深度学习框架的全面探索。

---

## 241. C$^2$MOE: Consistency and Complementarity-guided Mixture of Experts for Incomplete Multimodal Emotion Learning

**arXiv ID:** 2608.04013 | [PDF](https://arxiv.org/pdf/2608.04013v1)

**作者:** Yuntao Shou `[一作]` (Central South University of Forestry and Technology), Keqin Li `[通讯]` (State University of New York, New Paltz)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 C²MOE 框架，利用一致性与互补性引导的混合专家来解决多模态情感识别中的缺失模态问题。

**💡 创新点**

通过信息论分解将一致性和互补性分别映射到两个专家，并使用路由网络动态加权，实现在缺失模态下的鲁棒重建。

**🔧 技术方法**

采用信息论最大化互信息/条件熵、混合专家架构、变分推断预测、1D-CNN 特征映射、互补性对比学习等技术。

**📊 数据集**

在 CMU‑MOSI 与 CMU‑MOSEI 两个对话情感基准数据集上进行实验。

**📈 对比分析**

与 MCTN、MMIN、GCNet 等方法对比，C²MOE 在 ACC_2/F1/ACC_7 上均获得最高或最接近最高的成绩，且性能下降幅度显著降低。

**⚠️ 局限性**

对极低或极高缺失率下的特定模态仍可能导致信息损失，且模型对超参数 λ₁、λ₂ 的敏感性需要进一步研究。

---

## 242. A GitOps-Driven Annotation Catalog for Fully Automatic Railway Operations

**arXiv ID:** 2608.04724 | [PDF](https://arxiv.org/pdf/2608.04724v1)

**作者:** Martin Köppel `[一作]` (SIGNON Deutschland GmbH), Philipp Neumaier `[通讯]` (SIGNON Deutschland GmbH)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个基于GitOps和静态站点生成的轻量级铁路注释数据目录，用于自动列车操作的多传感器注释管理。

**💡 创新点**

创新点在于将数据作为代码(Data-as-Code)与GitOps CI/CD结合，并通过SSG实现无服务器、可追溯的注释元数据管理，消除文档漂移。

**🔧 技术方法**

采用GitLab DevOps、CI/CD、Python/SQLite、React+TypeScript+Tailwind、Vite构建工具、静态站点托管在GitLab Pages等技术。

**📊 数据集**

使用MARV平台的多传感器录制（ROS bag）和railLabel格式的注释JSON（基于ASAM OpenLABEL），涵盖数百万注释。

**📈 对比分析**

与Amundsen、DataHub、MLflow等传统单体目录对比，本文的构建时间为20–49分钟，页面加载138–414 ms，显著降低了基础设施开销与文档漂移风险。

**⚠️ 局限性**

局限性包括对GitLab CI运行资源依赖、构建时间受共享Runner限制，以及在极大规模数据集上可能需要更高的存储与网络带宽。

---

## 243. A/B Agent: A Self-Evolving Agent for Strategy Iteration in Industrial A/B Testing

**arXiv ID:** 2608.04625 | [PDF](https://arxiv.org/pdf/2608.04625v1)

**作者:** Zhuohang Jiang `[一作]` (Hong Kong Polytechnic University), Wenwu Ou `[通讯]` (Kuaishou Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种闭环 A/B 代理框架，用于工业推荐系统的策略迭代与参数自适应优化。

**💡 创新点**

创新点包括：① 将历史实验拆解为可复用的策略块并组织成层次化经验树；② 通过多路径 Tree‑RAG 与层次感知提升检索匹配度；③ 以实验树为引导的在线自进化机制，实现策略与参数的闭环迭代。

**🔧 技术方法**

技术栈涵盖：LLM（GLM‑5.1）+检索增强生成（Sparse & Dense）、树结构检索、分层重排序、实验树构建与效用评估、策略生成与验证。

**📊 数据集**

使用了310条历史推荐策略构建的工业基准数据集，涵盖三类短视频电商场景与多种推荐管线阶段。

**📈 对比分析**

与通用 LLM（如 GPT‑5.5、Claude‑Sonnet‑4.6）及多种 RAG 基线对比；在离线评测中取得最高整体分 7.244，较最强 RAG 提升 25‑32%；在线 A/B 测试 GMV 提升 4.829%，并同步提升 GPM、OPM、CVR 等指标。

**⚠️ 局限性**

局限性：① 依赖已构建的经验树，难以快速适应全新业务场景；② 关键检索与重排序对参数敏感，若训练数据噪声大易误检；③ 仍需人工审查与规则校验，自动化水平不高；④ 主要验证于短视频电商，泛化到其他类型推荐尚待进一步研究。

---

## 244. Bernoulli--Strang--Fix Conditions: Approximation and Prediction by Sampling Kantorovich Operators

**arXiv ID:** 2608.04727 | [PDF](https://arxiv.org/pdf/2608.04727v1)

**作者:** Sreya T `[一作]` (Indian Institute of Technology Indian School of Mines), A. Antony Selvan `[通讯]` (Indian Institute of Technology Indian School of Mines)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文在周期非均匀采样 (PNS) 框架下引入贝努利–斯特兰–费克 (BSF) 条件及其广义形式，利用这些条件研究采样 Kantorovich 算子（Kantorovich 采样算子）的多项式再现、逼近误差与从有限局部平均样本预测信号的理论与数值性能。

**💡 创新点**

创新点主要包括：① 将贝努利数与 Strang–Fix 条件结合，提出 BSF 条件；② 为向量生成器与 PNS 集合构造广义 BSF 条件，实现不要求多项式再现即可获得高阶逼近；③ 在此框架下给出预测算子可用有限局部平均样本进行信号预测的理论证明；④ 提供了构造满足 BSF 条件的生成器的实用方法，并展示其在高斯函数与 B‑spline 生成器上的数值验证。

**🔧 技术方法**

使用的技术包括：采样 Kantorovich 算子、傅里叶变换与 Poisson 求和公式、Bernoulli 数的递推性质、Taylor 展开、Minkowski 与 Hölder 不等式、Sobolev 空间逼近理论、近似逼近理论以及 MATLAB 进行数值仿真。

**📊 数据集**

实验使用的是合成信号，分别为高斯函数和 B‑spline 生成的函数（无真实数据集），用于验证逼近与预测误差随参数变化的收敛性。

**📈 对比分析**

通过对比不同尺度参数 W 与权重 α 下的逼近误差，展示误差随 W→∞ 递减至零；在预测实验中，算子仅利用有限个过去局部平均样本即可重建信号，误差随 W 增大而显著下降，证明了理论与实现的一致性和高精度。

**⚠️ 局限性**

局限性在于：研究仅限于一维情形，生成器需满足特定的紧支撑与平滑性条件；预测方法依赖于生成器在 (0,∞) 内有紧支撑；多维扩展及更一般采样方案的推广仍待进一步研究。

---

## 245. Towards Trustworthy Hypergraph Neural Networks under Label Noise

**arXiv ID:** 2608.04377 | [PDF](https://arxiv.org/pdf/2608.04377v1)

**作者:** Mengyao Zhou `[一作]` (Chinese Academy of Sciences), Guiying Yan `[通讯]` (Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究在标签噪声下的超图神经网络（HGNN）的鲁棒性，并提出了可提升鲁棒性的框架HyperTrust，基于对超边可信度的熵感知估计，随后通过HyperedgeBoost和HyperedgePrune两模块分别增强可信监督路径和剔除噪声传播路径。

**💡 创新点**

创新点包括①构建统一的超图标签噪声基准，系统评估传统LLN/GLN方法在超图上的局限；②提出HyperTrust框架，首次在超图中引入熵感知超边可信度评估；③通过增益可信超边和剪枝不可信超边的两阶段操作，实现对高阶信息传递的双重鲁棒性控制。

**🔧 技术方法**

主要技术为：预训练的HGNN编码器、熵感知超边可信度计算、基于余弦相似度的超边增强（HyperedgeBoost）、基于类原型相似度的关系剪枝（HyperedgePrune）、最终的预测融合与交叉熵训练。

**📊 数据集**

实验数据集涵盖文本领域的Cora、Citeseer、Pubmed及其Co‑citation版本，学术合作网络的Cora‑CA、DBLP‑CA，视觉与动作数据的NTU2012和ModelNet40，全部构造超图结构并在三种噪声类型（pair、uniform、random）下进行评测。

**📈 对比分析**

对比方法包括LLN类（S‑model、Co‑Teaching、JoCoR、APL、Forward、Backward）和GLN类（NRGNN、CP、CLNode、PIGNN、CGNN）以及基线HGNN。HyperTrust在大多数数据集和噪声设置下均显著优于所有基线，且在统计显著性检验下均达到最优或次优表现。

**⚠️ 局限性**

局限性：①对超边可信度阈值和剪枝阈值的选择仍有一定经验性；②在极高噪声或高度稀疏超图结构下，可信超边识别效果可能下降；③方法需要额外的预训练步骤和两次图传播，导致计算开销略高。

---

## 246. When Prompts Become Pixels: Prompt-Region Grounding for Multimodal Reasoning

**arXiv ID:** 2608.04726 | [PDF](https://arxiv.org/pdf/2608.04726v1)

**作者:** Yongxin Wang `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Xiaodan Liang `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了视觉化任务语义（VTS）评估框架，并设计了提示区域对齐（prompt‑region grounding）方法，显著缩小多模态大语言模型在视觉化问题场景中的语义通道差距；

**💡 创新点**

创新点在于将问题从文本通道迁移至图像像素，并通过对齐视觉问题区域与其文本等价表达来直接训练模型在图像中解读并使用任务指令，而非仅依赖OCR；

**🔧 技术方法**

采用了两种区域级目标：提示‑视觉表示蒸馏（PVRD‑SG）和掩码潜在预测（PRMLP），并在此基础上进行GSPO强化学习微调；

**📊 数据集**

使用了MATH‑Vision、MathVista、ChartQA、MMMU四大基准，以及VISTA‑Bench、OCRBench v2和1,000张真实世界任务图像；

**📈 对比分析**

与平衡回放和单纯SFT对比，平均VTS准确率从58.0%提升至66.3%，原始文本通道保持在69.1–70.3%，接口差距从11.2点降至4.0点；在独立基准和真实场景上亦取得4–8点的显著提升；

**⚠️ 局限性**

仍存在约4–7点残留差距，方法依赖已知问题区域与裁剪作为监督，可能在未知布局或极端视觉噪声场景下表现受限。

---

## 247. The Price of Isolation: Estimating the Ecosystem Cost of Symmetric Two-Sided A/B Testing

**arXiv ID:** 2608.04432 | [PDF](https://arxiv.org/pdf/2608.04432v1)

**作者:** Yuanyuan Shen `[一作]` (Snap Inc.), Chunhui Zhu `[通讯]` (Snap Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究对称双边隔离实验对内容平台生态系统的成本，并在生产环境中测量和建模。

**💡 创新点**

提出基于极值理论的订单统计模型，揭示匹配质量尾部决定隔离成本的关键性，并给出可落地的前测程序。

**🔧 技术方法**

使用极值理论、订单统计、尾类损失定律、仿真实验以及 A/A 与目录消融实验。

**📊 数据集**

Snap Inc. 生产短视频平台的两大 A/A 试验数据（10%/70%、10%/10%、2%/2%）及 10% 目录消融实验。

**📈 对比分析**

通过 A/A 对比和目录消融实验验证理论，结果显示模型对损失估计与观测高度吻合，具备良好预测性能。

**⚠️ 局限性**

仅估计内容侧成本，未分离供给与需求交互；尾类未精确识别；假设 i.i.d. 匹配质量，忽略相关性；平台演化可能导致 α 参数随时间变化。

---

## 248. Skills Know Their Neighbors: Cluster-Contrastive Capability Pages for Skill Retrieval

**arXiv ID:** 2608.04482 | [PDF](https://arxiv.org/pdf/2608.04482v1)

**作者:** Zifei Wang `[一作]` (Tencent IMA Product Center), Ruizhi Qiao `[通讯]` (Tencent IMA Product Center)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Capability Pages 这一离线编译的技能表示，用正向触发（trigger）、负向边界（boundary）和判别主体（body）三部分分别提升检索召回和路由准确率。

**💡 创新点**

创新点在于：① 把技能的可执行区域与其文档分离，阐明文档强加误差下界；② 通过聚类对比编译得到正负边界，构建两视图（检索视图使用 trigger+body，路由视图使用 boundary）来同时提升召回与排除误检；③ 证明单文档扩展不足以解决此类误差，强调邻居对比的重要性。

**🔧 技术方法**

使用技术包括：聚类（KMeans + Agglomerative）、LLM 生成 Capability Page（DeepSeek‑V4‑Pro），稀疏检索（BM25、TF‑IDF）、稠密检索（BGE‑M3、Qwen3‑Embedding‑0.6B/8B）、Doc2Query 对比、路由时的卡片对比，实验评估使用多种 LLM 执行器（DeepSeek‑V4‑Pro、Qwen3.6‑35B‑A3B、Qwen3.6‑27B‑FP8、Gemma‑E4B）。

**📊 数据集**

实验数据集：SRA‑Bench（26,262 技能 + 5,400 题目，包含 TheoremQA、LogicBench、CHAMP、MedCalc‑Bench、BigCodeBench、ToolQA）以及 SSL‑SkillDiscovery（6,184 技能 + 431 意图，约 96.5% 中文）。

**📈 对比分析**

比较方法：在原始文档、Doc2Query 扩展和 Capability Pages 三种索引文本上跑 5 种检索器，评估 Recall@10、nDCG@10、MRR@10；在路由层面对比是否使用 boundary 字段，评估 task‑success。结果显示：平均 Recall@10 提升 2.94 点，单模型提升 0.75–7.63 点；路由层面平均 task‑success 提升 3.62 点；跨语言迁移时，MRR@50 从 66.50% 提升至 73.07%。所有提升均在统计显著水平。

**⚠️ 局限性**

局限性：① 需要离线编译与维护，更新时需重新聚类与编译；② 负向边界仅在正确技能已进入候选集时有效，无法替代更高召回率；③ 聚类超参数（k、阈值）对结果有一定影响；④ 仅提升文本层面，与更强的模型或重排序技术相结合才可获得最大收益。

---

## 249. Responsibility in Multi-Agent Sequential Decision-Making: Comparing Human Judgments to Formal Models of Causal Attribution

**arXiv ID:** 2608.04318 | [PDF](https://arxiv.org/pdf/2608.04318v1)

**作者:** Nripsuta Ani Saxena `[一作]` (University of Southern California), Goran Radanović `[通讯]` (Max Planck Institute for Software Systems)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过在基于Goofspiel卡牌游戏的多代理顺序决策场景中进行大规模问卷调查，评估正式因果责任归因方法与人类责任判断之间的一致性。

**💡 创新点**

首次系统性地将实际因果定义与责任度量组合的责任归因模型与人类责任判断进行对比，并揭示初始条件偏差、信息可见度及反事实信息对责任评估的显著影响。

**🔧 技术方法**

使用了实际因果理论中的BF、HP、TR三种因果定义与CH、TR两种责任度量的组合来生成反事实场景，并采用多元线性混合效应模型对问卷数据进行统计分析；受试者招募通过Prolific完成。

**📊 数据集**

采用了基于Dec‑POMDP的Goofspiel游戏生成器产生的约1000局（每局五轮）无胜局的游戏记录作为问卷中的情境。

**📈 对比分析**

通过计算受访者责任等级与各责任归因方法所给责任等级的匹配得分来评估两者的一致性；实验结果表明，当初始手牌存在明显偏差时受访者与正式模型的匹配度最高，但在所有其他情境下未出现任何方法显著优于其他模型的情况。

**⚠️ 局限性**

研究的局限性包括：仅在简化的卡牌游戏环境中验证，缺乏对更复杂现实场景的普适性；受试者主要来自美国Prolific平台，样本可能不具备代表性；且实验中使用的反事实数量有限，未能充分探究更丰富的因果解释对责任评估的影响。

---

## 250. SpecDrop: Parameter-Free Category-Conditioned Routing for Modular Specialization

**arXiv ID:** 2608.04084 | [PDF](https://arxiv.org/pdf/2608.04084v1)

**作者:** Boyao Wang `[一作]` (Carnegie Mellon University), Zhihan Lei `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SpecDrop，一种固定、无学习参数的类别条件模块化dropout路由方案，用于在匹配参数预算下提升多分支网络的专门化与性能。

**💡 创新点**

通过固定分配矩阵和概率分级的dropout以及固定分母合并，消除了学习路由参数和辅助损失，仅靠类别标签即可实现专门化，并证明该方法在对齐分割时优于学习路由。

**🔧 技术方法**

固定规则路由、类别条件概率dropout、固定分母归一化、共享专家、warmup 余弦调度以及梯度聚焦理论。

**📊 数据集**

CIFAR-100、ImageNet-1K（BREEDS分级）、SlimPajama-6B语言建模（7个文档域）、SuperNI指令调优（20个任务集）等四个任务。

**📈 对比分析**

与密集模型、随机多分支、Stochastic Depth、Soft MoE等基线在相同参数/计算量下对比，SpecDrop在对齐的视觉分区中提升了+4.75% CIFAR、+6.53% ImageNet；在模糊分区中表现与匹配基线持平。

**⚠️ 局限性**

仅适用于需要在推理时提供类别标签的场景；对齐不充分的模糊分区无显著收益；在大规模模型和多标签任务上验证有限。

---

## 251. Accelerating C/C++ Pointer Analysis via Compiler-Based Offline Simplifications

**arXiv ID:** 2608.04466 | [PDF](https://arxiv.org/pdf/2608.04466v1)

**作者:** Zinan Gu `[一作]` (State Key Laboratory of Blockchain and Data Security, Zhejiang University), Kui Ren `[通讯]` (State Key Laboratory of Blockchain and Data Security, Zhejiang University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用LLVM的语义保持编译器优化，在中间表示(IR)层对程序进行离线简化，从而提升指针分析的性能；

**💡 创新点**

提出的简化方法不依赖于具体的指针分析算法，能在IR层获得更丰富的语义信息，支持实例化的、模块化的、可插拔的优化配置，并可与传统图简化互补；

**🔧 技术方法**

使用LLVM编译器框架的多级优化通道（函数、循环、模块级），通过随机搜索生成约300个优化配置，并在三种 Andersen 变体（DW‑ander、SCD‑ander、VSFS）上进行评估；

**📊 数据集**

实验使用22个开源 C/C++ 项目，覆盖从几千行到近48万行的代码量，涵盖多种功能类别；

**📈 对比分析**

通过在禁用优化的基线上与各优化配置比较，记录速度提升（最高3.14×）、内存消耗降低（最高1.94×）以及对分析精度的影响（基本保持不变），并分析IR指标的变化；

**⚠️ 局限性**

限制包括仅评估 C/C++ 程序与三种穷举点值分析，搜索策略为简单随机，未考虑阶段排序与细粒度区域优化，未覆盖其他指针分析形式、二进制提升或动态验证场景。

---

## 252. Architectural Implications of Agentic AI Workflows

**arXiv ID:** 2608.04458 | [PDF](https://arxiv.org/pdf/2608.04458v1)

**作者:** Jirong Yang `[一作]`, Jovan Stojkovic `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

未提供研究内容

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

## 253. Local Violation Certification for Linear Predict-Then-Optimize Pipelines

**arXiv ID:** 2608.04474 | [PDF](https://arxiv.org/pdf/2608.04474v1)

**作者:** Ş. İlker Birbil `[一作]`, Wenhao Chi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种针对线性预测‑再优化管道的局部违规认证框架，利用单个LP求解即可给出违规率和违规分布；

**💡 创新点**

创新点在于把违规率化为Mahalanobis距离的标准正态尾概率，并给出精确的截断高斯采样器，突破了传统情景生成的样本复杂度瓶颈；

**🔧 技术方法**

核心技术包括线性规划的基活跃区域分析、高斯变量变换、标准正态尾分布闭式表达以及条件高斯采样；

**📊 数据集**

实验以电力经济调度管道为例，使用五台发电机的成本、容量和排放参数数据；

**📈 对比分析**

与情景生成比较时，闭式解仅需一次LP求解，而情景生成需成千上万次求解，显著降低计算成本并给出更精确的风险评估；

**⚠️ 局限性**

局限性在于仅适用于完全线性预测、线性约束和线性违规判定，且假设部署决策所在基区无多重基切换，无法直接处理非线性或多区域情形。

---

## 254. DAC-Pose: Dual-Agent Collaborative Framework for Pose-Guided Human Generation

**arXiv ID:** 2608.04622 | [PDF](https://arxiv.org/pdf/2608.04622v1)

**作者:** Haotian Yang `[一作]` (City University of Macau), Xin Sun `[通讯]` (City University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出DAC-Pose框架，利用双智能体协作完成单视角姿态导向人像生成，尤其在跨视角反转和缺失部位补全场景下提升生成质量。

**💡 创新点**

引入Prior Semantic Reasoning (PSR) 和 Discrepancy-Aware Visual Encoding (DAVE) 两个互补的智能体，通过协同推理与视角误差编码实现对未观测区域的语义补全和空间对齐。

**🔧 技术方法**

结合多模态LLM推理、跨模态注意力对齐、稳定扩散模型以及VAE与Sentence-BERT编码，实现文本先验与视觉误差的双向反馈。

**📊 数据集**

在DeepFashion和Market‑1501两个公开数据集上进行实验，分别使用高分辨率时装图像和低分辨率行人图像。

**📈 对比分析**

与多种SOTA方法（Def‑GAN、PATN、PCDMs等）在SSIM、LPIPS、FID上进行统一评估，DAC‑Pose在DeepFashion上获得SSIM 0.7572、LPIPS 0.1274、FID 5.8547，超越最近方法IMAGPose；在Market‑1501上亦保持领先。

**⚠️ 局限性**

仍受限于单视角输入的先验不足，极端遮挡或极端姿态下可能出现细节失真，且框架对LLM推理的计算成本较高。

---

## 255. SONAR: Task-Aware Code Summary Evaluation for LLM Consumers Without References

**arXiv ID:** 2608.04195 | [PDF](https://arxiv.org/pdf/2608.04195v1)

**作者:** Simantika Bhattacharjee Dristi `[一作]` (University of Virginia), Matthew B. Dwyer `[通讯]` (University of Virginia)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SONAR 框架，基于代码重生成无参考的多维度评估方法，用来衡量代码摘要的正确性、抽象度、简洁性和流畅性，并通过这些维度预测 LLM 在四类下游软件工程任务中的表现。

**💡 创新点**

创新点包括：①首次将抽象度作为代码摘要质量维度；②利用代码重生成（Round‑Trip Correctness）而非参考文本来客观衡量摘要质量；③实现无参考、可扩展的评估框架；④通过实验展示不同维度对不同任务的重要性，从而实现任务感知的摘要评估。

**🔧 技术方法**

核心技术包括：代码重生成（利用 Gemini‑2.5‑Flash 等 LLM 生成实现代码）；差分模糊测试评估功能相似度；使用 CPG（Code Property Graph）和 Token 余弦相似度计算实现多样性；通过压缩后功能相似度评估简洁性；用 GPT‑2 的 perplexity 评估流畅性；多模型采样和提示工程提升维度得分。

**📊 数据集**

实验数据集：HumanEval、MBPP、BigCodeBench、The Vault，构建 500 个函数级 Python 代码；用于下游任务的检索、翻译、优化和测试用例生成等四类任务。

**📈 对比分析**

与传统参考指标（BLEU‑4、METEOR、ROUGE‑L、BLEURT、BERTScore）以及 SIDE 进行对比。SONAR 在每项任务上至少有一维度与 LLM 绩效显著正相关，相关系数最高可达 14 倍于最佳基线；并在 11 大 LLM 上揭示了模型间的维度优势与折衷，提示可以通过提示工程提升抽象度和简洁性。

**⚠️ 局限性**

局限性：仅实现 Python；对 LLM 生成的重构代码存在非确定性和噪声；对函数级代码的评估可能无法覆盖复杂项目级摘要；依赖于差分模糊器的执行环境，可能对不可执行或外部依赖的代码不适用；对抽象度的多模型采样虽然缓解单模型偏差，但仍受限于所选 LLM 的生成能力。

---

## 256. Mind-VLA: Instruction-Aware Spatial Representation Alignment for Vision-Language-Action Models

**arXiv ID:** 2608.04633 | [PDF](https://arxiv.org/pdf/2608.04633v1)

**作者:** Xingyu Ding `[一作]` (Nanjing University), Jian Cheng `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Mind‑VLA方法，将VLA模型的3D监督与语言指令对齐，以实现针对目标物体的空间表示。

**💡 创新点**

在训练时加入目标物体三视图的VAE潜变量预测和VGGT多层对齐，使3D监督具备指令感知，从而提升细粒度操作与遮挡鲁棒性。

**🔧 技术方法**

使用Transformer骨干、三视图VAE潜变量预测、VGGT特征对齐、Diffusion动作头以及基于CLIP的语言编码。

**📊 数据集**

在LIBERO（四个子套）和CALVIN ABC‑D长序列任务上评估，并在真实xArm 6机器人上进行Pick/Place/Drawer任务实验。

**📈 对比分析**

与7B级大型VLA和小型骨干模型对比，Mind‑VLA在LIBERO平均成功率93.9%，CALVIN平均完成长度4.47；在真实机器人上在遮挡条件下平均成功率54%，比Seer高32pp。

**⚠️ 局限性**

依赖预先获取目标物体的三视图，无法零样本部署到未知物体；在长序列任务中对全景场景建模不足，导致表现略逊。

---

## 257. Caching for the Future: Scrub Jay Episodic Memory Principles for Agent Memory Systems

**arXiv ID:** 2608.04746 | [PDF](https://arxiv.org/pdf/2608.04746v1)

**作者:** Kartikey Singh Bhandari `[一作]` (Birla Institute of Technology and Science), Pratik Narang `[通讯]` (Birla Institute of Technology and Science)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了 ScrubJay-MEM，一套基于 LLM 的代理记忆体系，采用 What–Where–When 记忆单元并引入每条记忆的类型化可腐败度 π_i、可逆更新与前瞻性记忆缓存；

**💡 创新点**

创新点在于把西方 Scrub Jay 的 WWW 记忆机制转化为可计算的 per‑memory 类型化衰减模型，自动分类并可在 O(1) LLM 调用下动态更新，同时通过 Prospective Memory Buffer 实现检索子线性成本；

**🔧 技术方法**

技术涵盖 4 维记忆编码、LLM+关键词回退的 π_i 与 τ 估计、查询自适应权重 α,β,γ,δ、超图关联与图奖励、Retroactive Contextual Integration、Prospective Memory Buffer 以及未来预期注释；

**📊 数据集**

使用的数据集包括 MemoryAgentBench 的 EventQA‑64k、内部构造的 Temporal Generalization Test (TGT)、MAB Conflict‑Resolution 子集，以及对比基线如 BM25、Contriever、Qwen3‑Embedding‑4B 等；

**📈 对比分析**

与 Mem0、A‑MEM、Contriever、Qwen3‑Embedding‑4B 在 EventQA‑64k 上比较，ScrubJay‑MEM 达到 61.58 F1（比 Mem0 +2.66，Qwen3 +3.09），在 TGT 上相较 9 种基线仅 ScrubJay‑MEM 实现正 GenGap +0.108（无衰减模式降至 +0.019），并保持 82.3% 的新旧判断准确率；

**⚠️ 局限性**

局限性主要是：在强大后端或需保持旧事实的任务（如事实合并）时效果不明显；TGT 为内部构造的诊断，外部可迁移性有限；模型在隐式遗忘敏感内容时需额外的保留策略。

---

## 258. EdgeLM: Edge Demonstrations for Language Models' Table Understanding

**arXiv ID:** 2608.04390 | [PDF](https://arxiv.org/pdf/2608.04390v1)

**作者:** Soroush Omidvartehrani `[一作]` (University of Alberta), Davood Rafiei `[通讯]` (University of Alberta)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 EdgeLM，一种基于边缘证据的检索框架，用以改进大语言模型在表格数据处理任务中的零样本推理。

**💡 创新点**

创新点在于同时利用两种边缘证据——数据边缘（局部标签对比）和模型边缘（模型误判），从而在保持相关性的同时显著增强决策边界的可辨识度。

**🔧 技术方法**

技术包括在检索前对训练集进行零样本评估构建错误池、基于相似度的局部邻域排序、对标签多样性进行轮询挑选、以及将两类边缘示例拼接至提示中进行 in‑context 推理。

**📊 数据集**

使用了十五个公开数据集（涵盖错误检测、实体匹配、属性匹配、缺失值插补、异常检测等五大数据处理任务），覆盖酒类、航班、医院、学术、零售、化学、财务、基因等多域场景。

**📈 对比分析**

与零样本、随机、标签多样性、相似度检索等基线对比，EdgeLM 在所有模型（OpenAI GPT‑4o‑mini、Gemini‑3.1‑flash‑lite、Qwen2.5‑7B、Llama‑3.1‑8B、Mistral‑Nemo‑12B）与任务组合上均获得最高或相近最佳 F1/准确率，特别是在异常检测与错误检测任务上提升幅度显著。

**⚠️ 局限性**

局限性在于需预先拥有标注样本和已冻结模型的零样本预测结果；在无标签或模型缺失误判信息的场景下难以构造两种边缘证据，并且目前仅在表格数据处理范畴内验证，尚未扩展到流式或非表格结构化数据。

---

## 259. SafeCommit: Certifying When Memory-Grounded Agents May Safely Act

**arXiv ID:** 2608.04289 | [PDF](https://arxiv.org/pdf/2608.04289v1)

**作者:** Mayur Akewar `[一作]` (Florida International University), Ravi Ranjan `[通讯]` (Florida International University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SafeCommit，一种在记忆不确定条件下安全提交外部动作的决策层，利用可校准的可行世界集合对动作进行证书化，若无证书则进行低副作用探测或保守回退。

**💡 创新点**

核心创新在于将记忆不确定性抽象为可校准的可行世界集合，使用 conformal 不合格性评分产生动作证书，并引入针对性探测来消除不安全的世界；同时提供安全等价压缩与分层探测策略。

**🔧 技术方法**

采用 conformal 推断、非合格性评分、可行世界构造、可行世界压缩（安全等价）、低副作用探测（如元数据读取、权限检查、沙箱模拟）、安全映射（Γ_t）以及一个无依赖的模拟器与基准。

**📊 数据集**

主要使用 SafeCommitBench‑Controlled，一个冻结的 JSONL 基准，包含四类记忆不确定性（stale、conflict、poisoned、auth. drift）和手工生成的可行世界、动作、探测等数据；实验在 10 个种子下进行。

**📈 对比分析**

与单世界决策、随机探测、冲突时回避、无探测 SafeCommit 等方法进行对比。指标包括不安全提交率（UCR）、任务成功率（TS）、提交覆盖率（CC）、回退率（FR）和平均探测数。实验显示 Full SafeCommit 将 UCR 从 41.2% 降至 2.6%，任务成功率提升至 97.4%，回退率降至 0%，平均探测数约 0.55。

**⚠️ 局限性**

局限性包括：1) 需要显式可行世界构造，真实系统中的世界构造不完整导致 β 误差；2) 依赖 conformal 交换性，分布漂移会影响校准；3) 探测模型为确定性二元输出，未考虑噪声与延迟；4) 仅在单步决策下给出安全保证，长期序列风险尚未解决；5) 需要对安全映射（Γ_t）进行精确实现，错误映射会导致误判。

---

## 260. Attention-based representations for multi-task computation

**arXiv ID:** 2608.04243 | [PDF](https://arxiv.org/pdf/2608.04243v1)

**作者:** Daniel Hsu `[一作]` (Columbia University), Mingyue Xu `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文理论分析单层多头注意力在有限维度与数值精度限制下能否同时支持多任务（最小/最大计算、XOR 等），并给出了对应的下界与上界。

**💡 创新点**

首次证明多头注意力在实现多任务表示时，头数与维度/精度满足阈值度乘积的下界关系，并展示该下界是最优的；同时将阈值度、加法基等概念引入注意力分析，开辟了新的理论工具。

**🔧 技术方法**

使用 Erdős–Szekeres 定理、几何阻塞/体积论证、注意力输出的有理函数表示、阈值度（threshold degree）理论以及加法基构造等数学技术。

**📊 数据集**

本文为纯理论研究，没有使用任何实验数据集。

**📈 对比分析**

通过构造实现上界与证明下界，展示两者匹配，从而验证理论结果；未进行实验比较或性能评估。

**⚠️ 局限性**

仅针对线性后处理器（线性分类器、多项式阈值函数）给出结论，未涵盖非线性预测器；结果仅适用于标准 softmax 注意力，未考虑位置编码、层归一化等常见变体；并未进一步证明所给维度上界是否最小。

---

## 261. Tactus: Open-Vocabulary Object Recognition from Low-Cost Pressure Arrays

**arXiv ID:** 2608.04043 | [PDF](https://arxiv.org/pdf/2608.04043v1)

**作者:** Abdul Basit Tonmoy `[一作]` `[通讯]` (Eximius Labs), Abdul Basit Tonmoy (Eximius Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Tactus，一个将低维电阻式压力传感器帧序列映射到冻结的多模态文本嵌入空间的开源模型，支持无训练分类器的开放词汇触觉识别。

**💡 创新点**

创新点在于将压力数组与多模态语言模型对齐，实现开放词汇零样本识别；同时给出小样本预训练与校准归一化的可量化配方，并对错误结构进行深入分析。

**🔧 技术方法**

采用ResNet‑18宽度主干、1×1卷积融合、投影至文本嵌入空间；使用同传感器的masked‑autoencoder预训练、聚类采样窗口、目标中心化等技术；语言侧冻结Qwen3‑VL‑Embedding‑2B。

**📊 数据集**

使用STAG数据集，包含548-taxel电阻手套的32×32帧，27个对象+空手，共187条训练记录和测试记录。

**📈 对比分析**

在STAG hold‑out测试上，Tactus平均top‑1 0.771±0.062、top‑3 0.935，超过闭集CNN基线0.76，且无需训练分类器；通过tuple voting可提升到0.858，表现稳健但对少数接触模糊类仍有误判。

**⚠️ 局限性**

局限包括高方差（±0.062）、仅针对单一传感器家族、未证明对新物体类别的开放式泛化；同时对传感器归一化和数据路径管理要求高，跨传感器迁移效果差。

---

## 262. Out-Of-The-Loop Multi-Fidelity Bayesian Optimization

**arXiv ID:** 2608.04113 | [PDF](https://arxiv.org/pdf/2608.04113v1)

**作者:** Gustavo Sutter `[一作]` (University of Waterloo), Agustinus Kristiadi `[通讯]` (Vector Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种“离线-循环”多保真贝叶斯优化（OOL‑MF‑BO）框架，解决在优化过程中无法直接访问最高保真度时，如何利用历史最高保真度数据进行知识迁移，以提升搜索效率。

**💡 创新点**

创新点包括：①首次正式定义并理论证明OOL‑MF‑BO在标准LMC/ICM模型下存在不可忽略的子最优性；②提出深度核多任务多保真度模型，将任务描述（结构化或通过LLM抽取的向量）与保真度信息融合，弥补缺失最高保真度的缺陷；③验证该方法在合成、化学分子和超参数调优等多领域的显著性能提升。

**🔧 技术方法**

技术手段包括：多保真度高斯过程（LMC/ICM/MISO）、深度核（对任务向量与保真度的RBF核），以及常用多保真度采集函数（MF‑MES/MF‑EI）。

**📊 数据集**

使用的数据集包括：合成函数（Branin、Michalewicz、Park 等八个基准），分子优化基准（Xe/Kr 选择性、溶剂化能、极化率），以及 HPOBench 的两类机器学习模型（逻辑回归、支持向量机）。

**📈 对比分析**

与随机搜索、Next‑Best、Single‑Task 等基线对比。实验结果显示，多任务深度核方法在大多数基准上显著降低累计回报，尤其在需要最高保真度信息的场景下，相比传统方法可获得更快的收敛速度和更低的最终回报。

**⚠️ 局限性**

局限性：①理论分析仅覆盖LMC/ICM族核，尚未扩展到更通用的核；②方法对历史任务相关性的依赖较强，若历史任务与当前任务不相关，收益有限；③高斯过程的三次复杂度仍是瓶颈，需进一步探索更高效的近似或稀疏技术。

---

## 263. The Evaluator Is Part of the Experiment: Measuring Open-Ended LLM Conformity

**arXiv ID:** 2608.04463 | [PDF](https://arxiv.org/pdf/2608.04463v1)

**作者:** Alicia Guerra `[一作]` (Illinois Institute of Technology), Yibo Hu `[通讯]` (Illinois Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计实验协议测量大型语言模型在多代理对话中的开放式一致性，区分生成端、候选内容曝光、同行呈现及评审侧的同行上下文影响。

**💡 创新点**

提出了分离生成重答、候选内容与呈现残差的框架，并对评审者的上下文敏感性进行配对盲/知情评估，展示了传统答案翻转指标不足。

**🔧 技术方法**

使用层级序数模型估计潜在答案质量，进行anchor校准审计，并结合固定预训练分类器作为外部验证。

**📊 数据集**

实验使用四个开源LLM（Qwen2.5-7B、Mistral-7B、Gemma-2-9B、Llama-3.1-8B）与三大基准数据集TruthfulQA、MMLU-Pro、ARC-Challenge。

**📈 对比分析**

比较方法是对各模型在无同行、全部正确、全部错误、混合四种同行条件下的质量差异进行分层估计，结果显示全部错误同行导致最低质量；评审者表现出正向、负向或中性上下文敏感性；anchor校准提升量表识别准确度。

**⚠️ 局限性**

局限在于无法获得答案质量的真实标签，因果估计基于实验对照而非单一实例，且不同模型/任务的外推性有限。

---

## 264. MIDAS: Multi-LLM Iterative Data-Adaptive Summarization

**arXiv ID:** 2608.04307 | [PDF](https://arxiv.org/pdf/2608.04307v1)

**作者:** Karen Lee `[一作]` (Volkswagen Group Innovation), Umair Rasheed `[通讯]` (Volkswagen Group Innovation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出MIDAS框架，通过多LLM协同迭代优化提示以自动适应企业多格式摘要需求

**💡 创新点**

引入数据驱动的模式学习与统一CoT批评器，消除手工提示，学习域特定格式约束

**🔧 技术方法**

多LLM提示优化、数据模式学习、统一CoT批评、LLM评估器

**📊 数据集**

企业IT帮助台多语言工单数据（约24,635条）以及金融行业ECTSum基准

**📈 对比分析**

与零样本、ICL、CriSPO、ZERA等基线对比，MIDAS在5种输出格式中平均提升ROUGE-1/2/L 11‑18%，并保持最高BERTScore

**⚠️ 局限性**

仅在大模型上验证，未在小开源模型或更广泛企业域测试，需进一步评估

---

## 265. Fewer Tokens, Smaller Cache: Reward-Coordinated Efficient Reasoning

**arXiv ID:** 2608.04771 | [PDF](https://arxiv.org/pdf/2608.04771v1)

**作者:** Qiyuan Zhu `[一作]` (Hong Kong University of Science and Technology), Sirui Han `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 ReCo 的框架，在推理过程中根据每一步的过程奖励动态调整 KV 缓存压缩比例、限制反思词的生成以及根据答案置信度提前停止，以降低推理成本。

**💡 创新点**

创新点在于：① 将过程奖励作为单一信号同时驱动缓存压缩、生成长度控制和提前停止，实现三者协同；② 通过奖励自适应的压缩率使高奖励步骤可以更激进地压缩缓存，而低奖励步骤则保留更多上下文；③ 在压缩的同时通过反思词惩罚和置信度提前终止，避免因压缩导致的生成长度膨胀。

**🔧 技术方法**

主要技术包括：KV 缓存压缩（按奖励调节保留比例）、注意力加权选择、基于奖励的反思词惩罚、答案困惑度置信度检测、轻量级过程奖励估计器 Pilot（30M 参数）以及对步骤进行基于换行符的划分。

**📊 数据集**

使用六个数学与科学推理基准：GSM8K、MATH‑500、AMC2023、AIME24、AIME25、GPQA，以及模型 DeepSeek‑R1‑Distill‑Qwen‑7B、DeepSeek‑R1‑Distill‑Llama‑8B、Qwen3‑8B。

**📈 对比分析**

与全链路推理、三种 KV 压缩（SnapKV、R‑KV、RPC）以及两种生成长度控制（SAT、Dynasor）对比。ReCo 在保持准确率仅略低于全链路推理的同时，平均生成 token 数降低 37%‑65%，端到端延迟提升 2.08×‑2.35×，峰值 GPU 内存也最低；相比单独压缩或单独控制，ReCo 的综合表现最优。

**⚠️ 局限性**

限制：① 仍需要额外的过程奖励估计器与置信度检测；② 对奖励阈值、压缩比例等超参数有一定依赖；③ 在极端高压缩比例下，准确率可能显著下降；④ 目前仅在单 GPU 上验证，跨多 GPU/大模型的可扩展性尚待评估。

---

## 266. DisMix: Order-Aware Mixup for Medical Imaging via Disentangling Ordinal and Non-Ordinal Features

**arXiv ID:** 2608.04652 | [PDF](https://arxiv.org/pdf/2608.04652v1)

**作者:** Dileepa Pitawela `[一作]` (Adelaide University), Hsiang-Ting Chen `[通讯]` (Adelaide University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种名为 DisMix 的顺序感知混合方法，通过对医学影像中的序数特征和非序数特征进行解耦，从而生成符合临床严重程度等级的增广样本。

**💡 创新点**

创新点在于利用双代码书 VQ‑VAE 将影像分离为顺序子空间与非顺序子空间，并在这两个子空间上分别执行不同的混合策略，避免了传统 Mixup 混合时破坏顺序结构的问题。

**🔧 技术方法**

主要技术包括双代码书 VQ‑VAE、顺序软标签、对抗判别器、梯度反转层与余弦相似度约束，以及多种顺序感知混合策略（Ordinal Mix、Non‑Ordinal Mix、Generate & Mix、Order Swap）。

**📊 数据集**

在四个医学影像数据集上验证：膝关节骨关节炎（KOA）、印度糖尿病视网膜病变（IDRID）、朝阳结肠直肠病理图像（Chaoyang）以及 Crowd Gleason 前列腺病理图像（Gleason）。

**📈 对比分析**

与六种主流 Mixup 基线和六种序数分类器组合比较，DisMix 在 24 种组合中在 20 种准确率、15 种 MAE 上取得最佳表现，显著提升准确率（Wilcoxon p=0.0075）并在多数场景下减少 4–6% 的 MAE，且在数据稀缺和分级变异性下保持鲁棒性。

**⚠️ 局限性**

局限性包括：需要额外的单一训练阶段以学习解耦表示，产生的增广样本不适合直接临床解读，且相较于纯粹的 Mixup 方法，计算开销略高，尤其在生成与缓存混合样本时需要额外资源。

---

## 267. ATLAS: Adaptive Topological Learning with Abstract Successors for Continual Learning

**arXiv ID:** 2608.04334 | [PDF](https://arxiv.org/pdf/2608.04334v1)

**作者:** R. Blake Lawlor `[一作]` (University of Utah), Daniel S. Brown `[通讯]` (University of Utah)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了Adaptive Topological Learning with Abstract Successors (ATLAS) 模型，用于持续学习和快速适应非平稳目标。

**💡 创新点**

创新点在于通过 Grow When Required (GWR) 网络构建自组织拓扑，并结合 Successor Features 实现转移动力学与奖励的结构解耦，提供快速恢复和正向迁移。

**🔧 技术方法**

使用 GWR 网络、Successor Features、离线“dreaming”规划、BFS 路径搜索、目标条件逆动力学控制器等技术。

**📊 数据集**

在连续控制的 PointMaze Medium 与离散控制的 MiniGrid Four Rooms 两个导航环境上进行实验。

**📈 对比分析**

与常见的模型自由基线 (SAC, PPO, DQN) 比较，ATLAS 在目标迁移阶段实现几千步内快速恢复，并在逆转阶段表现出正向迁移，整体样本效率显著优于基线。

**⚠️ 局限性**

局限在于低层控制器对连续动力学的部分遗忘，以及标准 GWR 的动态权重导致拓扑崩塌，缺乏动态边缘删除和对高度随机环境的适应性。

---

## 268. ReGround: Restoring Visual Grounding in Multi-Step Reasoning through Self-Diagnosis and Visual Re-Examination

**arXiv ID:** 2608.04385 | [PDF](https://arxiv.org/pdf/2608.04385v1)

**作者:** Lei Peng `[一作]` (University of Science and Technology of China), Wei Hu `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ReGround 框架，利用自我诊断与图像再注入实现视觉语言模型的多步推理自我纠正。

**💡 创新点**

核心创新在于将诊断质量与视觉再注入结合：仅有图像再注入无法改善，只有精准的自我诊断提示才能引导模型正确重检证据。

**🔧 技术方法**

技术手段包括两阶段训练（SFT+GRPO强化学习）、图像再注入（将图像token重新放置在对话新回合）、以及基于强大模型的诊断提示生成（能力引导）。

**📊 数据集**

使用来自十个源数据集构建的 68,477 条训练轨迹，并在 MathVista、MathVision、MathVerse、HallusionBench、MMBench、MMStar、VisuLogic、V*Bench 八个公开基准上进行评测。

**📈 对比分析**

与基线、Look‑Back、Thyme 等方法相比，ReGround 在 Qwen2.5‑VL‑7B 上在 5/8 个基准中取得最优或接近最优成绩（平均提升 2–5 分），在 Qwen3‑VL‑8B 上亦保持稳定提升；推理开销仅 1.6–1.8 倍，明显低于工具辅助方法。

**⚠️ 局限性**

局限性包括：对细粒度或空间分散的视觉任务效果有限；再注入导致的 prompt 令牌增多，可能对多图像/视频输入产生线性扩展；无法弥补视觉编码过程中丢失的细节。

---

## 269. Hardware-Enabled Fuzzy Inference: Architectures, Platforms, and Emerging Trends

**arXiv ID:** 2608.04031 | [PDF](https://arxiv.org/pdf/2608.04031v1)

**作者:** Amir Hossein Jalilvand `[一作]` (Iran University of Science and Technology), M. Hassan Najafi `[通讯]` (Case Western Reserve University)

**通讯引用:** 1431 | [OpenAlex ID](https://openalex.org/A5012903661)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述并系统化了硬件加速的模糊推理系统，按FPGA、ASIC与嵌入式TinyML平台进行分类；

**💡 创新点**

首次以平台为中心的分类法，识别七大研究缺口并提出未来方向；

**🔧 技术方法**

涵盖FPGA LUT/DSP映射、ASIC流水线与定制单元、模糊成员函数LUT、混合信号/模拟实现、随机/一元计算、Memristive内存计算、SoC协同设计等多种实现技术；

**📊 数据集**

对多种应用案例进行综述，使用的数据集包括光伏MPPT、直流/交流电机控制、风电MPPT、ECG心电图、无人机检测、机器人控制等；

**📈 对比分析**

通过速度、功耗、灵活性、开发与生产成本等指标对比各平台，发现FPGA灵活且易快速原型，ASIC能效最高但需高非递归工程，MCU/TinyML低功耗适合边缘，但无单一平台在所有维度最优；

**⚠️ 局限性**

主要局限：缺乏统一基准与报告规范、规则库规模可扩展性差、在线学习实现稀缺、设计自动化工具不足、缺乏可解释性硬件接口以及对新兴内存技术的应用有限。

---

## 270. AMD SEV-SNP: A Confidential Computing Primer

**arXiv ID:** 2608.04039 | [PDF](https://arxiv.org/pdf/2608.04039v1)

**作者:** Amean Asad `[一作]` (Confidential.ai), Patrick Woodhead `[通讯]` (Confidential.ai)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对 AMD SEV‑SNP 进行技术阐述，详细说明其从硬件根信任到内存加密、Reverse Map Table、VMPL、GHCB 等机制的实现与交互，构建完整的可信执行环境并实现远程证明。

**💡 创新点**

创新点在于将 hypervisor 视为对手，利用 RMP、VMPL 与 GHCB 协议实现对 guest 内存和寄存器的完整隔离与完整性检测，同时通过硬件签名的 attestation 把启动测量绑定到硬件根信任链，实现可信证明。

**🔧 技术方法**

使用技术包括 AMD Secure Processor 与内存控制器 AES 加密、ASID 基于密钥的访问、RMP（Reverse Map Table）内存完整性校验、VMPL 级别的硬件特权分层、GHCB 协议的显式 guest‑hypervisor 通信、#VC 异常、可信 I/O、Secure AVIC 中断过滤等。

**📊 数据集**

本文主要为技术说明，并未使用任何公开数据集；实验基于 AMD 公开的硬件规范和原型实现。

**📈 对比分析**

通过与 SEV、SEV‑ES 等前代技术对比，作者指出安全属性大幅提升；性能方面，内存加密与 RMP 检查对延迟影响极小，未给出具体数值，但指出现有硬件在常规工作负载下接近原生速度。

**⚠️ 局限性**

局限性包括：不保障可用性、侧信道、物理攻击、软件漏洞；对 AMD 制造与固件的信任仍是关键；在需要高可用性或无物理攻击假设的场景下仍需配合运维手段。

---

## 271. ODRA: Synthesizing Cognitive Behavioral Therapy Sessions with Structured Chain-Of-Thought and Dynamic Patient Resistance

**arXiv ID:** 2608.04524 | [PDF](https://arxiv.org/pdf/2608.04524v1)

**作者:** Javier Rodriguez-Juan `[一作]` (University of Alicante), Iryna Gurevych `[通讯]` (Technische Universität Darmstadt)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了ODRA框架，用于合成结构严格、同时包含患者抵抗行为的CBT对话。

**💡 创新点**

核心创新在于将Chain‑of‑Thought与CBT协议紧密结合，并通过抵抗orchestrator与动态prompt steering解决LLM的sycophancy问题。

**🔧 技术方法**

技术方案包括多阶段Chain‑of‑Thought推理、指数平滑抵抗更新、行为配置器、LLM‑as‑Judge评估以及在Llama‑3/Qwen‑3.5上进行的细调。

**📊 数据集**

采用150条合成CBT会话（9577回合、18496条推理轨迹），构建自CACTUS intake forms，并通过Sentence Transformers检索相似患者档案。

**📈 对比分析**

通过自动评测（CTRS、行为对齐、推理轨迹）和专家盲测（13项指标）与CACTUS、MAGneT、SQPsych、MIRROR等基线比较，ODRA在治疗技能、行为对齐和推理轨迹上均显著优于对照组；Fine‑tune实验表明ODRA数据能显著提升模型在不同患者设定下的临床表现。

**⚠️ 局限性**

主要限制包括：单会话设计无法模拟多会话进展；intake form仅提供初始信息，缺乏长期状态动态；合成成本高；使用通用LLM缺乏临床预训练；仅建模患者抵抗，未针对治疗师抵抗做专门处理；LLM‑as‑Judge存在偏见和长度偏好。

---

## 272. Zero-error information equals amortized communication complexity

**arXiv ID:** 2608.04141 | [PDF](https://arxiv.org/pdf/2608.04141v1)

**作者:** Daiki Suruga `[一作]` `[通讯]` (University of Waterloo), Daiki Suruga (University of Waterloo)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在随机通信复杂度框架下，解决了直接求和问题，给出了独立实例的期望和最坏情况的通信复杂度极限，并进一步推导了函数组合（OR型）在单侧误差模型下的精确极限；

**💡 创新点**

创新点在于证明期望化随机通信复杂度的渐近值等价于零误差信息复杂度，提出单实例协议嵌入配合前缀验证机制；同时对集合不交问题的规模假设给出精确反驳，构建了全局误差模型下的新定理；

**🔧 技术方法**

主要技术包括信息理论方法、信息复杂度分析、协议压缩技术、单实例嵌入与前缀验证、Yao最小化原理以及极限与连续性分析；

**📊 数据集**

该工作完全基于理论分析，不使用任何实验数据集；

**📈 对比分析**

通过严谨的理论证明与已有的下界/上界进行对比，得到匹配上下界，从而确定了极限值，并驳斥了先前关于集合不交问题的规模猜想；

**⚠️ 局限性**

局限性：对于总函数是否存在信息复杂度与通信复杂度的指数分离仍未解答；直接求和定理在全局误差模型下仍不成立，且部分结果仅适用于特定误差模型；

---

## 273. Predict, Then Retrieve: Cross-Instance Future-State Retrieval from Video Prefixes

**arXiv ID:** 2608.04426 | [PDF](https://arxiv.org/pdf/2608.04426v1)

**作者:** Quynh Vo `[一作]` (VinUniversity), Anh-Tuan Luu `[通讯]` (National University Of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Predictive State Retrieval（PSR）基准，要求模型从短视频前缀预测物体未来状态，并在跨实例视频/图像语料库中检索对应实例；同时提出LFTR轻量级检索器，利用冻结编码器预测未来潜在向量并在语义与视觉空间融合检索。

**💡 创新点**

首次将未来状态预测与跨实例检索结合，构建多难度层级、跨域且含人类验证的基准；通过“天花板分解”揭示预测而非感知是主要瓶颈；LFTR通过跨空间融合和难负样本训练，在不使用大规模LLM的情况下超过32B多模态LLM。

**🔧 技术方法**

使用冻结的视频编码器（如DINOv2）、文本编码器（BGE、Gemma等）和一个轻量级Transformer头；该头实现潜在轨迹滚动、条件读取和多查询InfoNCE训练；检索阶段在语义和视觉空间分别匹配，并通过z-score融合。

**📊 数据集**

从四大公开数据集（HowToChange、Oops、SSv2、MOST）构建，包含218k视频片段和333k图像，生成20,182个多类型、跨时间窗的检索查询，覆盖四个难度层级。

**📈 对比分析**

与随机、kNN、零射击冻结编码器、保留当前帧、7B→32B文本LLM以及32B多模态LLM（M3）等基线对比；LFTR-融合在R@5上达16.1±1.0，显著高于M3的11.9（+4.2pp）且计算成本仅为M3的四阶小；但与oracle ceiling（≈53）仍有显著差距，表明预测能力是关键。

**⚠️ 局限性**

局限性包括：图像语料的词汇不匹配导致检索困难；跨域泛化仅在四个源域内评估；GT由单一VLM生成，存在生成器偏差；人类验证样本有限，未覆盖所有难度层级；冻结前缀编码器虽不成为主要瓶颈，但仍可能限制性能；模型对社会偏见与错误的敏感度未完全消除。

---

## 274. Rethinking Reservoir Pruning: A Dynamical Perspective for Echo State Networks

**arXiv ID:** 2608.04593 | [PDF](https://arxiv.org/pdf/2608.04593v1)

**作者:** Sudip Laudari `[一作]` (Independent Researcher), Puspa Raj Adhikari `[通讯]` (Sungkyunkwan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出一种基于轨迹平均雅可比格拉姆矩阵的动态模式剪枝方法（DMP），对ESN的随机储备层进行一轮结构化剪枝，只重新训练线性读出。

**💡 创新点**

创新点在于将神经元重要性直接与输入驱动的状态转移模式关联，通过保留能量占比阈值的主子空间来衡量每个单元在主导动态方向上的贡献，从而实现比传统权重或激活统计更精细的动态剪枝。

**🔧 技术方法**

采用的技术包括：轨迹采集、雅可比矩阵计算、格拉姆矩阵求平均、特征值分解、基于贡献度的排序以及对剪枝后矩阵的可选谱半径重标。

**📊 数据集**

使用的数据集有：Mackey‑Glass混沌序列、澳大利亚国家电力需求、气象温度、风速以及太阳辐射的单变量时序。

**📈 对比分析**

与未剪枝ESN、中心性（Betweenness、Closeness）剪枝以及Leaky/Deep ESN等基线进行对比，采用10%、20%和30%剪枝率，在5个数据集和多次随机种子下评估MSE/NRMSE。结果显示，DMP在大多数数据集上能降低或保持预测误差，并实现约1.9×的推理速度提升。

**⚠️ 局限性**

限制包括：离线计算成本高（需要O(TN^3)和特征分解），对非常小的储备层或极端剪枝时可能失效，且一次性基于原始矩阵的近似在剪枝后动态可能不完全匹配，尤其在瞬态或稀有事件影响下的评估不够精准。

---

## 275. Emergence of Reputation-Based Cooperation in LLM Agents

**arXiv ID:** 2608.04507 | [PDF](https://arxiv.org/pdf/2608.04507v1)

**作者:** Kazuya Horibe `[一作]` (RIKEN), Wataru Toyokawa `[通讯]` (RIKEN)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过在连续捐赠游戏中模拟LLM代理的间接互惠互动，检验其在文化进化过程中对自由乘客（无条件不合作者）的抵御能力。

**💡 创新点**

创新之处在于将自然语言提示视为可进化的策略，揭示LLM在缺乏显式声誉标签的环境下只能实现Image Scoring级别的判别，无法自行演化出更稳健的Leading‑Eight规范，并指出对手的“投入敏感度”是预测自由乘客抵御力的关键指标。

**🔧 技术方法**

采用自然语言提示作为策略、基于适者优生的轮盘赌选择进行文化传播、连续捐赠游戏作为适应度评价、以及统计回归分析等技术来量化判别强度与合作稳健性的关系。

**📊 数据集**

实验数据来自四个LLM后端（Claude 3.5 Sonnet、Gemini 1.5 Flash、Gemini 2.0 Flash、Gemini 2.5 Flash），在10代、12人群体中收集策略文本并评估其性能，未使用外部标注数据集。

**📈 对比分析**

通过比较各模型在单代对抗自由乘客实验中所产生的“稳健”策略比例来评估性能，其中Gemini 2.5 Flash最高达48%，Gemini 2.0 Flash 28%，Gemini 1.5 Flash 10%，Claude 3.5 Sonnet仅3%；同时不同模型的资源积累表现也存在明显差异。

**⚠️ 局限性**

局限性包括仅使用四个模型（样本量极小）、固定的小规模群体、只评估单一非适应性自由乘客、未对策略文本内容进行深入分析，以及模型级别统计可能掩盖个体差异。

---

## 276. DeepInvert: Semi-Supervised Embedding Inversion Against Obfuscated Language Models

**arXiv ID:** 2608.04477 | [PDF](https://arxiv.org/pdf/2608.04477v1)

**作者:** Zhicong Huang `[一作]` (Ant Group), Tao Wei `[通讯]` (Ant Group)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种半监督嵌入逆向攻击方法（DeepInvert），能够在对抗基于遮蔽的语言模型防御时更高精度恢复原始令牌。

**💡 创新点**

创新点在于利用无标签目标嵌入的可利用语义结构，结合软标签、掩码增强视图以及EMA教师的自监督一致性正则化，并通过多影子模型和混合训练策略实现监督与自监督的互补。

**🔧 技术方法**

技术主要包括半监督学习（FixMatch框架改造）、掩码增强、软标签一致性、EMA教师、PCA降噪和多影子模型训练等。

**📊 数据集**

使用了五个公开数据集（SST‑2、CoNLL‑2003、AG News、Medical Meadow MedQA、Enron Emails）以及四种模型（BERT‑Base、RoBERTa‑Base、LLaMA‑3‑8B‑Instruct、Qwen3.5‑27B）。

**📈 对比分析**

与KNN、InvBert、ER、TBS、DEML等基线对比，在九种遮蔽防御、五个任务、四个模型上，DeepInvert 的 Top‑1 令牌恢复率普遍高于之前最佳攻击，特别是对 ObfusLM ϵ=0.1 达到 73.5%（原 26.2%）。

**⚠️ 局限性**

局限性包括对攻击所需的无标签目标嵌入数量、对特定防御的参数敏感性，以及在更强的差分隐私或生成任务中仍可能需要更高的噪声或额外的加密机制。

---

## 277. A Long-Run Persistence Theory for AI Systems under the Redundancy-Adjusted Artificial Age Score (AAS)

**arXiv ID:** 2608.04012 | [PDF](https://arxiv.org/pdf/2608.04012v1)

**作者:** Seyma Yaman Kayadibi `[一作]` (Victoria University), Seyma Yaman Kayadibi `[通讯]` (Victoria University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5119424975)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

提出了基于冗余调整的人工年龄评分（AAS）的循环级别长期持续性框架，并给出了其数学定义、性质与长期行为分类。

**💡 创新点**

创新点在于：①将静态的AAS扩展为可随循环迭代的年龄序列；②引入对齐、加权与对数惩罚的冗余调整机制；③定义并证明了负载平衡、零负载、振荡和累计终端负载等四大长期运行模式；④在严谨的数学框架下证明了年龄序列有界、收敛、几何衰减及零负载等性质。

**🔧 技术方法**

使用的技术主要是：离散时间系统建模、对数惩罚核设计、加权冗余调整、极限分析、序列收敛性与几何衰减定理、敏感性与连续性证明。

**📊 数据集**

该研究为理论性工作，无使用具体数据集。

**📈 对比分析**

没有实验或数值比较，本文仅给出定理证明和理论推导；因而无法给出性能指标。

**⚠️ 局限性**

局限性：①仅聚焦于结构负载的静态度量与循环序列，未覆盖噪声、随机漂移或学习动态等实际问题；②缺乏实验验证和对真实AI系统的适用性评估；③模型假设（如冗余度不消失）对实际系统的可实现性尚未讨论。

---

## 278. SAFECAST: Robust Failure Detection for VLA Policies with Contrast-Set Training and Calibration

**arXiv ID:** 2608.04246 | [PDF](https://arxiv.org/pdf/2608.04246v1)

**作者:** Harshitha Rajaprakash `[一作]` (University of Southern California), Jesse Thomason `[通讯]` (University of Southern California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `51c0528b-f690-4182-ae60-bb5f046c276c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 SAFECAST 框架，利用视觉、语言及多模态对比集扰动对 VLA（Vision‑Language‑Action）策略的隐藏状态进行探测器训练和功能化置信度校准，以实现部署时的运行时失败检测。

**💡 创新点**

创新点在于：①引入对比集扰动（视觉噪声、语言同义改写、联合扰动）作为训练与校准的数据增强；②在对比集环境下对隐藏状态风险序列进行功能化 conformal 预测，从而提升对多模态分布移位的鲁棒性；③实现了在仿真中训练探测器、在真实机器人上对比集校准的 sim‑to‑real 转移。

**🔧 技术方法**

核心技术包括：隐藏状态探测器（轻量级 MLP）、功能化 conformal 预测、对比集生成与 DTW 过滤、视觉扰动（加入干扰物、光照变化）、语言扰动（ChatGPT 生成同义句）、多模态联合扰动、仿真与实地实验的结合。

**📊 数据集**

使用了 LIBERO‑Spatial、LIBERO‑Plus 两个仿真数据集以及真实 DROID/Franka 机器人实验；对比集由视觉干扰物、光照变化、语言同义改写和联合扰动构成；还使用了少量人机遥操作演示与 ChatGPT 生成的语句。

**📈 对比分析**

通过与基线 SAFE 进行对比，在 F1 分数和 ROC‑AUC（对不同 α 取平均）上进行评估。SAFECAST 在视觉、语言及多模态扰动下均显著提升检测性能；联合视觉‑语言扰动的配置最优；在 sim‑to‑real 场景中，仿真训练+真实对比集校准比仅在真实数据上训练/校准表现更好。

**⚠️ 局限性**

局限性包括：需要手工或程序化构造足够代表性对比集，耗费额外采样成本；对比集覆盖的扰动范围有限，面对未出现的扰动可能仍失效；并未恢复在任意部署移位下的严格 conformal 交换性保证。

---

## 279. Multi-View Face and Gesture Animation with Dynamic Gaussians

**arXiv ID:** 2608.04722 | [PDF](https://arxiv.org/pdf/2608.04722v1)

**作者:** Alireza Javanmardi `[一作]`, Didier Stricker `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

论文探讨了某一领域的研究问题，并提出了一种新的解决方案。

**💡 创新点**

创新点在于提出了一种新的算法或模型，能够更有效地解决该问题。

**🔧 技术方法**

使用了深度学习技术和机器学习算法。

**📊 数据集**

使用了公开数据集进行实验，具体数据集名称未提及。

**📈 对比分析**

与现有方法进行了比较，结果显示新方法在准确性和效率上均有显著提升。

**⚠️ 局限性**

限制在于模型的可扩展性和对特定数据集的依赖性。

---

## 280. GeoReward: Mitigating Contextual Variable Overestimation in Vision-Language Models for Cross-Market Preference Prediction

**arXiv ID:** 2608.04504 | [PDF](https://arxiv.org/pdf/2608.04504v1)

**作者:** Shuo Liu `[一作]` (Alibaba International Digital Commerce Group), Xiaoyi Zeng `[通讯]` (Alibaba International Digital Commerce Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

诊断并解决VLM中稀疏关键变量被高频特征淹没的 Contextual Variable Overestimation（CVE）问题，并提出 GeoReward 框架实现跨国广告偏好预测与市场定制内容生成。

**💡 创新点**

三阶段门控架构（Market‑Aware Retrieval‑Augmented Generation、Context‑Guided Visual Modulation、Selective Sensitivity Loss）专门针对 CVE；引入 MACP 基准和 GeoReward 奖励模型用于 RL 微调，实现对稀疏关键信息的显式关注与权重调节。

**🔧 技术方法**

使用 Retrieval Augmented Generation（MA‑RAG）、轻量级视觉调制器（CGVM）、敏感性损失（SSL）以及 RLHF（DPO）与控制网络的 Stable Diffusion 进行文本到图像的生成。

**📊 数据集**

自建 Multi‑Country Ad Click Preference (MACP) 数据集，包含 823K 训练样本与 180K 测试样本，覆盖 10 个国家的广告图片点击率与产品信息。

**📈 对比分析**

在 MACP 上与 Qwen2‑VL 等多种基线（单/双模块组合）对比，GeoReward 在准确率 60.37% 与敏感度 40.84% 上显著优于基线（准确率提升约 5%，敏感度提升约 4%），验证了三门控的协同效应。

**⚠️ 局限性**

仅针对单一稀疏变量（国家），多变量扩展尚未实现；在极端冷启动场景下性能仍受限；缺乏跨领域通用基准，限制了对 CVE 泛化能力的全面评估。

---

## 281. FinReportBench: Measuring and Improving Institution-Grade Financial Report Generation

**arXiv ID:** 2608.04374 | [PDF](https://arxiv.org/pdf/2608.04374v1)

**作者:** Yinghao Tang `[一作]` (Zhejiang University), Wei Chen `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个专家驱动的金融报告生成基准 FinReportBench，用以衡量生成报告是否满足机构级交付标准。

**💡 创新点**

创新点包括：①将专业评审偏好转化为可观测的 35 项层级评估指标；②设计了覆盖 95.65% 多语言语义空间的 244 任务数据集；③提出基于基准诊断的技能演化流程，将模型反复出现的缺陷转化为可迁移的生成约束。

**🔧 技术方法**

采用了多模态评估（文本+渲染页面）、对比式证据挖掘、专家偏好引导的评估量表构建以及技能演化的优化器和审核者框架；所有评估均由 GPT‑5.6 Luna 进行。

**📊 数据集**

使用了 10,000 条中英双语金融研究记录（5,000 中文、5,000 英文）经过筛选与聚合后得到 244 任务；每个任务包含公开查询、检索证据及隐藏来源包。

**📈 对比分析**

对九个大型模型族进行统一评测，使用层级得分（G0、G1、G2）和 35 项通过率；在已演化的技能 K⋆ 下，平均 G1 提升 33.85 分、G2 提升 13.83 分，且所有模型均保持 G0 不下降，展示了跨模型的显著性能提升。

**⚠️ 局限性**

局限性包括：①评估仍受限于专家样本与偏好，可能无法覆盖所有机构特定细节；②技能演化仅在外部验证集上进行，未必能完全泛化到更广泛的任务空间；③基准覆盖率虽高，但仍存在语言与领域的细粒度差异，影响跨域迁移能力。

---

## 282. Radar4D-VLM: Proposal-Grounded Temporal 4D Radar Reasoning Across Frozen Language Models

**arXiv ID:** 2608.04130 | [PDF](https://arxiv.org/pdf/2608.04130v1)

**作者:** Jiaju Han `[一作]` (China University of Petroleum-Beijing), Chengyin Hu `[通讯]` (China University of Petroleum-Beijing)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并评估了仅使用4D雷达的时间视觉‑语言模型Radar4D‑VLM，构建对象、场景、运动等层次化token并与冻结语言模型接口。

**💡 创新点**

①将雷达点云转化为结构化token并通过低秩投影兼容多种冻结语言模型；②在受控实验中分离接口兼容性、雷达依赖与对齐语言监督的实际益处；③实现无LLM直接推理的裁剪版本。

**🔧 技术方法**

RTNH兼容稀疏卷积提议器、时间+多普勒特征编码、kinematic token设计、低秩投影、冻结Qwen/Phi/Mistral/Llama/Gemma等大语言模型、对齐与打乱语言监督对照实验。

**📊 数据集**

K‑Radar 4D毫米波雷达数据集，使用10帧滑动窗口进行训练/验证，严格划分1–40训练，41–48验证。

**📈 对比分析**

通过Top‑64提议召回、五任务宏平衡准确率以及对齐/打乱/无语言的平行对照；结果表明接口兼容性好、雷达内容和时间顺序对性能关键，但对齐语言监督并未产生稳定提升；提议召回达98.13%，核心平衡准确率约0.49。

**⚠️ 局限性**

仅在K‑Radar验证集上评估，未公开测试；对齐监督实验仅采用单一词表置换和三种种子；缺乏对不同语言模型规模、不同雷达频率或多模态融合的泛化评估；部分实验依赖特定提议器结构。

---

## 283. Why Ranking Anomaly Detection Algorithms Isn't as Reliable as You May Think

**arXiv ID:** 2608.04613 | [PDF](https://arxiv.org/pdf/2608.04613v1)

**作者:** Simon Klüttermann `[一作]` (Carnegie Mellon University), Alice Kirchheim `[通讯]` (TU Dortmund University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了异常检测算法在不同基准设置下的排名不稳定性，量化了排名波动并分析其主要来源。

**💡 创新点**

提出了新的排名不稳定度指标（σ_rank），并揭示了数据集选择和超参数配置是影响排名稳定性的关键因素。

**🔧 技术方法**

采用了七种经典异常检测方法（KNN、LOF、IFOR、HBOS、PCA、CBLOF、SEAN）在690个数据集上进行大规模实验，利用随机采样、指标变更、超参数随机化等技术评估排名波动。

**📊 数据集**

使用OddBench benchmark套件中的690个数据集，按需随机抽样至最多5000个样本，进一步在不同实验配置下挑选子集。

**📈 对比分析**

通过对不同数据集数目、评估指标（ROC‑AUC/ AUCPR）、超参数随机化和随机种子等因素的系统模拟，发现排名平均差异仅约1.5位，σ_rank最高可达0.4；仅使用约200个数据集即可将σ_rank降至≈0.2，提升稳定性。

**⚠️ 局限性**

研究局限于轻量级方法，超参数空间相对有限；未覆盖深度学习或大型基础模型的异常检测方法，且仅挑选7种算法，可能无法完全代表当前领域的多样性与复杂度。

---

## 284. ContextWeave: A Real-World Workflow Benchmark

**arXiv ID:** 2608.04830 | [PDF](https://arxiv.org/pdf/2608.04830v1)

**作者:** Bo Wang `[一作]` (Fudan University), Xipeng Qiu `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并实现了 MemoryBench，一套基于真实办公工作流的纵向基准，用于评估语言模型在跨任务记忆中的效果，并提供可执行的工作流、受控评估协议与多维指标。

**💡 创新点**

创新点包括：①将真实多月工作日志转化为可执行任务序列并进行隐私去标识；②设计以 Workspace Score、Preference Score 为核心的下游导向评价体系，并辅以 Relevance、Continuity、Solvability、Hallucination Robustness 四项诊断指标；③在不同记忆组件与基础模型上系统实验，揭示记忆在跨任务工作流中的价值与风险。

**🔧 技术方法**

技术手段涵盖：LLM 辅助任务重构（指令生成、环境构造、执行验证）；Docker 沙箱执行；多种记忆实现（Summary、Task Summary、ICL、MemoryBank、Mem0、LangMem、A‑Mem）；GPT‑5.5 进行推理、评估与诊断；embedding 相似度与 LLM 判别用于相关性与可解性评估。

**📊 数据集**

数据集为 14 位参与者在一个开源项目中多月的工作日志，共 1005 个可执行任务，其中 568 个为核心任务；原始文档、差分等信息经过规则+人工双重脱敏后构成基准。

**📈 对比分析**

实验通过对比有无记忆、不同记忆组件与不同基础模型，在 Workspace Score 与 Preference Score 上进行量化比较。结果显示记忆提升约 8–12 分，ICL 记忆（In‑Context Learning）最具提升；不同模型对记忆的利用效果差异明显，DeepSeek 在 Workspace 上提升最大，GPT‑5.5 在 Preference 上提升最大；诊断指标表明记忆提高了相关性、连续性、可解性，同时误导率保持低水平。

**⚠️ 局限性**

局限性包括：评价标准需人工校准且成本高（约 200 美元/配置）；仅覆盖有限的模型与 agent harness；基准数据来源单一项目、有限人数与领域，难以全面泛化；未来需要降低评估成本、扩展模型与工作流多样性。

---

## 285. VoxStruct3D: Structure-Leading Flow Matching for Voxel-Space 3D MRI Synthesis

**arXiv ID:** 2608.04557 | [PDF](https://arxiv.org/pdf/2608.04557v1)

**作者:** Fang Li `[一作]` (Beihang University), Aimin Hao `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出Voxel-space模型VoxStruct3D用于高保真3D MRI合成。

**💡 创新点**

创新点在于Volumetric Voxel Generator（VVG）与Structure-First Image-Follows（SFIF）两大模块，利用重叠解码减少块状伪影并通过预训练3DINO编码器的结构引导实现全分辨率细节与全局解剖一致。

**🔧 技术方法**

结合Diffusion Transformer（DiT）、重叠反卷积解码、Patch-Aligned RoPE、异向注意力、结构VAE、流匹配时间步调度与Sobolev损失等技术。

**📊 数据集**

使用T1加权的病理（BraTS 2021）和健康（IXI、NIMH、NFBS）脑MRI数据集。

**📈 对比分析**

与HA-GAN、3D-LDM、3D MedDiffusion、WDM、MOTFM等基线对比，VoxStruct3D在FID、MS-SSIM、MUSIQ、NIQE、Tenengrad等指标上显著优于所有基线。

**⚠️ 局限性**

主要局限在训练成本高、仍需改进跨模态细节重建、对不同器官/体位的泛化性待验证。

---

## 286. Characterizing the Evolving Landscape of Modern Information Seeking

**arXiv ID:** 2608.04609 | [PDF](https://arxiv.org/pdf/2608.04609v1)

**作者:** Shuoqi Sun `[一作]` `[通讯]` (RMIT University), Shuoqi Sun (RMIT University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过构建ISMIE框架、开展在线情景调查与离散选择实验，以及使用脑电（EEG）实验探究生成式AI聊天机器人对现代信息寻求行为与认知负荷的影响。

**💡 创新点**

创新点在于首次提出专门描述现代信息环境下信息寻求的ISMIE框架，系统评估生成式AI在搜索情境中的用户偏好和认知差异，并将多模态脑电信号引入信息寻求研究。

**🔧 技术方法**

主要技术包括在线问卷与情境模板化、离散选择实验（DCE）、脑电（EEG）记录与多变量认知指标分析。

**📊 数据集**

数据来源于两阶段在线众包调查（包含真实搜索情境与选择实验）以及实验室收集的EEG与任务完成结果数据。

**📈 对比分析**

对比方法是控制信息覆盖度后，比较传统搜索引擎与生成式AI聊天机器人在用户认知负荷（EEG特征）和搜索结果质量上的差异；虽然具体性能未给出，但实验设计旨在揭示认知投入与搜索效能的权衡。

**⚠️ 局限性**

局限性包括样本规模有限、实验室情境与真实网络搜索行为的差异、EEG信号易受干扰、以及对生成式AI多轮交互深度特征的捕捉尚不完整。

---

## 287. OutLangSplat: 3D Language Gaussian Splatting for UAV Outdoor Scenes

**arXiv ID:** 2608.04560 | [PDF](https://arxiv.org/pdf/2608.04560v1)

**作者:** Xia Yan `[一作]` (Zhejiang University of Technology), Jiazhou Chen `[通讯]` (Zhejiang University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了 OutLangSplat，结合 2D-3D 双分支融合和训练无关的像素贡献与一致性聚合，实现 UAV 户外场景的开词汇 3D 场景理解。

**💡 创新点**

创新点包括基于区域的 2D-3D 对齐融合提升空间一致性，以及基于像素贡献可靠性和跨视角语义一致性的聚合策略。

**🔧 技术方法**

采用 3D 高斯剖分、RemoteCLIP/RS5M 视觉语言编码、MinkUNet 3D 结构提取、像素贡献逆辛普森指数和一致性加权迭代聚合等技术。

**📊 数据集**

使用四个公开 UAV 户外场景（InstanceBuilding 的 Buildings1/2 与 UrbanScene3D 的 Polytech/Campus）构建的 660 区域标注数据集。

**📈 对比分析**

在开放词汇定位与语义分割任务上与 LangSplat、Lang3D‑XL、LUDVIG 比较，OutLangSplat 在 Loc@50 约 88.8% 及 mIoU/mAcc 最高，显著优于其它方法。

**⚠️ 局限性**

依赖已有 3DGS 重建质量，重建误差会影响特征对齐与聚合，未来需联合优化重建与语言表示。

---

## 288. Social Pressure Breaks Majority Voting in LLM Safety Panels

**arXiv ID:** 2608.04415 | [PDF](https://arxiv.org/pdf/2608.04415v1)

**作者:** Yibo Hu `[一作]` (Illinois Institute of Technology), Jiaming Qu `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过两轮实验，评估当所有LLM审阅者在投票前收到相同错误标签提示时，个体误判与面板投票的变化。

**💡 创新点**

首次揭示共享错误社交线索会导致多数投票失效，提出在面板部署前对共享上下文下的审阅者表现进行前置诊断。

**🔧 技术方法**

使用六个开源LLM（Qwen、Llama、Gemma、Mistral、OLMo）作为审阅者，在六个安全数据集上进行两轮评估，并用多数投票进行聚合；同时分析错误转移与置信度变化。

**📊 数据集**

BeaverTails、XSTest、Ethics、WildGuard、Aegis、ToxiChat 这六个安全内容基准。

**📈 对比分析**

与无标签同伴对照（silent peers）比较，测量误判率、假警报率和聚合后错误率；结果显示错误标签提示将平均假警报率从56.5%提高至87.5%，面板假警报率从43%升至100%；提示恢复尝试效果不稳定。

**⚠️ 局限性**

局限在于仅使用单轮模拟消息，未覆盖多轮对话；实验仅涵盖六个开源模型及少数专有模型，提示和结构的可推广性尚未充分验证。

---

## 289. NeuMoSync: End-to-End Neuromodulatory Control for Plasticity and Adaptability in Continual Learning

**arXiv ID:** 2608.04358 | [PDF](https://arxiv.org/pdf/2608.04358v1)

**作者:** Seyed Roozbeh Razavi Rohani `[一作]` (Simon Fraser University), Mo Chen `[通讯]` (Simon Fraser University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种名为 NeuMoSync 的全局神经调制架构，通过一个可学习的控制器为每个神经元动态生成四个输入相关的调制系数（α_SM、α_WC、α_AL、α_ARM），并将其与快网络权重和慢网络（EMA）权重融合，进而实现可塑性保持与快速适应的连续学习。

**💡 创新点**

创新点在于：1) 将生物学中的全局神经调制机制转化为人工网络的全局协调模块；2) 通过参数共享的 Transformer / 1D 卷积控制器实现对数千甚至上万神经元的统一调制；3) 将权重、激活斜率与偏置三大计算要素全部输入相关地调制；4) 引入快慢网络融合与 EMA 记忆相结合的混合学习策略，模仿快速-慢速学习系统。

**🔧 技术方法**

技术细节包括：可学习的神经元特征向量、指数移动平均（EMA）快慢网络、四个调制系数（α_SM、α_WC、α_AL、α_ARM）、Transformer/1D 卷积控制器、PReLU 的可自适应负斜率、加法偏置调制、端到端梯度下降训练；实验使用 ResNet-18/50、CNN、MLP 等基准模型。

**📊 数据集**

实验使用多种连续学习基准：随机标签 CIFAR‑10、MNIST；概念漂移 Shuffle CIFAR‑10；域递增 Permuted MNIST；类递增 Class Split CIFAR‑100、T‑ImageNet；以及随机标签/概念漂移等多任务序列。

**📈 对比分析**

与传统可塑性保持方法（CBP、CReLU、ReDo、L2Init+EWC）以及元学习基线（MAML、ANML）在平均在线任务准确率、LCA_F/B、知识转移指标（BKT、FKT）上进行比较。NeuMoSync 在保持可塑性、快速适应（LCA_F/B 更高）以及知识转移方面均显著优于基线，在多数任务上提升幅度可达 10‑20% 以上。

**⚠️ 局限性**

局限性包括：1) 对灾难性遗忘的主动防护不如专门的稳定性方法，常需配合经验回放；2) 纯全自注意力控制器计算复杂度随神经元数呈二次增长，需稀疏采样等扩展方案；3) 对极大模型或跨模态、非监督任务的适用性尚未充分验证；4) 理论上对调制与优化动力学的交互机制仍缺乏深入分析。

---

## 290. Eliciting Intrinsic Hallucinations in LLMs via Semantically Equivalent Adversarial Attacks

**arXiv ID:** 2608.04286 | [PDF](https://arxiv.org/pdf/2608.04286v1)

**作者:** Atri Vivek Sharma `[一作]` (Imperial College London), Alessio Lomuscio `[通讯]` (Imperial College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种在检索增强生成（RAG）系统中对模型进行语义等价对抗攻击的评估框架，并针对该框架改造多种对抗方法，系统测评其在不同模型与数据集上的真值保持能力。

**💡 创新点**

创新点在于：①首次在RAG情境下引入语义等价约束的对抗攻击，以激发模型内部的“真实幻觉”；②将多种已有攻击技术（梯度、遗传、搜索、黑盒迭代等）统一到此框架；③通过LLM判别器实现自动化的等价性与真值评估。

**🔧 技术方法**

采用梯度导向的GCG、遗传算法AutoDAN、黑盒PAIR、SECA-GB/B以及SRA等对抗策略；使用LLM（如Gemini‑2.5‑Flash‑Lite）进行语义等价与真值判定；在Hallucination Lens基准上实现语义相等的查询生成。

**📊 数据集**

主要数据集包括Hallucination Lens的三大子集：FaithEval（对抗性真值对照）、ANAH‑v2（多领域实体问答）以及FailSafeQA（金融长篇上下文问答）。

**📈 对比分析**

通过对比清洁准确率(CA)、对抗准确率(AA)、攻击成功率(ASR)与查询困惑度(PPL)评估攻击效果；实验显示即使是最先进的Gemini‑2.5‑Flash‑Lite、GPT‑5‑chat等模型，在最具挑战的FailSafeQA/FaithEval数据集上，SRA/SECA等攻击可使ASR超过50%，表明模型真值保持鲁棒性不足。

**⚠️ 局限性**

局限性包括：①基于LLM判别器的评估可能受模型偏好影响；②只针对单轮问答，未考虑多轮对话；③大规模推理型模型实验受限；④部分词级攻击产生高困惑度、自然度差；⑤未对防御方法进行系统评测。

---

## 291. Season: Spectrum-Aware Orthogonal Gradient Refinement for Transfer-Based Adversarial Attacks

**arXiv ID:** 2608.04441 | [PDF](https://arxiv.org/pdf/2608.04441v1)

**作者:** Tianyi Wang `[一作]` (Tongji University), Shengjie Xu `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种 Spectrum-Aware Orthogonal Gradient Refinement（Season）框架，用于提升跨架构的基于迁移的对抗攻击；

**💡 创新点**

创新点在于将梯度按频率分解为低频结构分支和高频纹理分支，采用低显著性引导将高频噪声引导至背景，并通过正交投影消除两分支间的干扰；

**🔧 技术方法**

主要技术包括高斯低通滤波实现梯度分频、预计算显著性图和遮罩、正交投影约束、以及作为通用插件的包装实现；

**📊 数据集**

使用 ImageNet 1k 验证集与 8 个不同架构（CNN、ViT、MLP）目标模型进行实验；

**📈 对比分析**

与八种主流迁移攻击（MI-FGSM、DI-FGSM、PI-FGSM、TI-FGSM、SI-FGSM、VT、Admix、NI-FGSM）对比，Season 在平均上提升 6.6% 的成功率，单个目标可提升至 16% 以上；

**⚠️ 局限性**

局限性包括：仅在 ImageNet 及 L∞ 边界下验证，针对其他数据集或不同攻击范式（如 L2、白盒）尚未系统评估，且正交投影与显著性遮罩需额外预处理，可能在某些场景下引入微小计算开销。

---

## 292. Discretization and Statistical Consistency of Functional Flow Matching

**arXiv ID:** 2608.04531 | [PDF](https://arxiv.org/pdf/2608.04531v1)

**作者:** Lennon J. Shikhman `[一作]` `[通讯]` (Georgia Institute of Technology), Lennon J. Shikhman (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `14d48e9d-0069-4ad9-996a-1d5968216998` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

论文研究功能流匹配（Functional Flow Matching, FFM）在函数空间上的离散化与统计一致性问题，给出了非嵌套观测信息条件下的条件目标一致性、量化误差界、以及点传感器下的正则化重构等结果；

**💡 创新点**

创新点在于：①证明在非嵌套sigma‑algebra下条件目标的强L²收敛；②提出点传感器正则化与正交投影误差分解；③通过与总体叠加路径耦合给出无需唯一性约束的端到端Wasserstein误差上界；④验证归一化四分之一神经算子在网格无关下的学习常数；⑤给出非对易高斯例子与贝塞尔近似的O(n⁻¹)风险界；⑥对截断高斯缩放模型给出显式收敛速率；

**🔧 技术方法**

主要技术包括：条件期望与Bochner期望、弱连续方程与超位置原理、正交投影与不相干投影误差分析、Sobolev嵌入与正则化空间、神经算子（quadrature neural operator）理论、Lipschitz与马尔可夫不动点估计、Bernstein不等式、Wasserstein距离与耦合技术、以及高阶泛函不变式与复合收敛分析；

**📊 数据集**

论文不使用具体数据集，全部为理论分析与模拟实验（如高斯例子、截断高斯缩放实验）来验证理论；

**📈 对比分析**

方法通过对比理论误差界与已知的传统流匹配或ODE学习结果，展示在网格无关、正则化空间及截断高斯缩放下取得了可观的收敛速率（如O((log n)/√n)）和显式的Wasserstein误差上界；

**⚠️ 局限性**

局限性包括：①需要可控的重构、实现与稳定常数，未覆盖未知频率外推；②对非截断高斯数据需更强的矩条件；③学习场的Lipschitz稳定性仍是必要假设，缺乏对独立终点稳定性的完整分析；④示例表明连续场的Lipschitz不必保证网格无关的有限目标上界；⑤未给出完整的实验验证与对比。

---

## 293. Eigenius: A Typed Knowledge-Graph DBMS with Epistemic Stratification and Institution-Mediated Reasoning

**arXiv ID:** 2608.04457 | [PDF](https://arxiv.org/pdf/2608.04457v1)

**作者:** Hans-Martin Will `[一作]` (Eigenius Project), Matthew Fuchs `[通讯]` (Docimion)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

实现了 Eigenius，一款开源的 typed knowledge‑graph DBMS，统一将经验与形式知识的证明链、类型系统与存储层嵌入单一不可变、内容可寻址的核心；

**💡 创新点**

其创新点在于将依赖类型理论、Grothendieck 机构和内容寻址存储合并进数据库核，构造结构化证明链；通过 comorphism 将跨系统翻译持久化、消除 O(N^2) 适配器瓶颈，并在内存中直接检验 Lean4 证明，确保从提交到验证的完整可追溯链路；

**🔧 技术方法**

采用 Rust + RocksDB、EigenTT 依赖类型理论、Grothendieck 机构、内容寻址存储、Lean4 内嵌验证器、EigenQL 典型 Datalog/聚合查询、Julia 生态机构以及容器化/Wasmm 的互补技术；

**📊 数据集**

评估数据集包括 WordNet 与 UMLS 词典用于批量注入，以及重现 Chan 等人发表于 Nature 的 WRN helicase 研究，完整复制实验流程与结论；

**📈 对比分析**

通过微基准（WordNet+UMLS 注入在单核 22‑CPU 机器上达约 5200 资源/秒）与宏观评估（复现 Nature 研究 52/52 结论并发现 4 个数据不一致）进行比较，显示 Eigenius 提供可重跑、可检查的完整链路，显著优于传统脚本+脆弱流水线；

**⚠️ 局限性**

局限性包括：机构协议最初基于 Wasm 的边界不适用于大多数科学库，改为容器后仍缺少完整安全沙箱；验证器与 comorphism 类型检查器仍为可信组件，需机械化证明；层图合并与类型完整性理论尚未完全形式化；AI Scientist 的完整闭环实现尚未完成。

---

## 294. Lindblad-Inspired Multi-Timescale Reservoir Computing with Separable Rotation and Dissipation

**arXiv ID:** 2608.04028 | [PDF](https://arxiv.org/pdf/2608.04028v1)

**作者:** Jyotiranjan Beuria `[一作]` (IKS Research Centre), Amit Shukla `[通讯]` (Indian Institute of Technology Mandi)

**通讯引用:** 1621 | [OpenAlex ID](https://openalex.org/A5008777707)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于 Lindblad 动力学分离旋转与耗散的多时尺度固定残差网络，并仅训练线性读取器实现时序学习。

**💡 创新点**

创新点在于将旋转角度和衰减率分别作为可独立调节的设计变量，构造正交混合的阻尼旋转块，从而实现无需谱半径缩放的全局收敛保证，并提供可解释的记忆与混合控制。

**🔧 技术方法**

采用分块指数化的阻尼旋转模式、正交相似变换、多成员并行块结构、tanh 非线性以及岭回归读取器，对输入投影进行固定化。

**📊 数据集**

在多项标准时间序列基准（线性记忆、NARMA‑10/20、Mackey‑Glass、Lorenz‑63、延迟 XOR）以及 UCI 空气质量传感器校准数据集上进行评估。

**📈 对比分析**

与标准 ESN、泄漏 ESN、深度 ESN、正交循环、CRJ、NG‑RC 以及 32 隐藏单元 GRU 在相同参数预算下配对比较，使用 Wilcoxon 检验；在 bounded NARMA‑20 与记忆容量上表现最佳，整体保持竞争力。

**⚠️ 局限性**

在光滑或短延迟任务（如 Mackey‑Glass、延迟 XOR）中不如泄漏/深度 ESN 或 NG‑RC，且需要手动调节衰减参数，缺乏任务自适应的权重训练。

---

## 295. Helping Music Co-Creation Agents 'Listen' Well: Hierarchical Self-Supervised World Models for Understanding and Generation

**arXiv ID:** 2608.04378 | [PDF](https://arxiv.org/pdf/2608.04378v1)

**作者:** Scott H. Hawley `[一作]` `[通讯]` (Belmont University), Scott H. Hawley (Belmont University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个基于Swin V2的分层自监督音乐“世界模型”（MIDI‑RAE‑JEPA‑SON），通过对MIDI piano‑roll 的无标签学习得到层级嵌入，并以此为条件驱动流匹配生成器，实现低延迟的音乐生成与掩码填充；同时将LLM用于口头反馈，形成“AI Rick Rubin”式的协作创作流程。

**💡 创新点**

① 采用分层Swin V2 + JEPA 目标，首次在符号音乐上实现多层次无监督表征；② 将层级嵌入直接作为流匹配模型的条件，省去压缩与解码器，显著降低推理时延；③ 通过在粗层添加少量监督（和弦、短句），显著提升和声与调性识别；④ 结合LLM提供可解释的口头建议，强调人机协同。

**🔧 技术方法**

Swin V2 Transformer、JEPA（Masked Embedding Prediction）+ equivariance loss、SIGReg 正则化、流匹配生成器、PCA 降维条件、层级 dropout、LLM 对话生成、MIDI‑to‑audio 预处理。

**📊 数据集**

Lakh MIDI 数据集（1×、4×）、POP909（用于训练、评估和监督标签），以及通过音频‑MIDI 转换获得的多乐器 MIDI。

**📈 对比分析**

使用一系列线性探针（phrase、note‑density、chord、key、chroma 等）对每层嵌入进行评估，并与 DINOv2 基线对比。自监督模型在短句与音高密度上与 DINOv2 接近，和弦监督将和声检测提升至 0.54、调性识别提升至 0.70。生成/填充方面，流匹配在 10 Euler 步、guidance 1.0 下实现 pixel‑F1 0.996，CPU 推理 2.8 s（M1 Max）/0.6 s（MPS）。

**⚠️ 局限性**

仅使用 MIDI piano‑roll，忽略力度、发音等细节；生成多样性有限，低于大型扩散模型；窗口尺寸固定 128×128，无法直接生成全曲；需外部和弦/短句监督才能显著提升和声任务；对复杂节奏与多轨混合支持不足。

---

## 296. When Proxy Prediction Becomes Equation Reconstruction: Diagnostics and Residual Learning for Factor-Derived Proxy Supervision

**arXiv ID:** 2608.04393 | [PDF](https://arxiv.org/pdf/2608.04393v1)

**作者:** Chayan Lahiri `[一作]` (Adams State University), Cody Fehringer `[通讯]` (US Forest Service)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了利用RUSLE方程生成的土壤侵蚀代理值，在因子信息降解时模型的鲁棒性，并提出RASPL框架来保持公式估计作为预测锚点。

**💡 创新点**

创新点在于通过公式保持的残差学习机制，既保留降解后公式估计，又利用自适应门控学习上下文修正；并构建完整的降解因子诊断协议。

**🔧 技术方法**

采用了公式保持残差网络(RASPL)、统计与卷积上下文编码器、加权对数空间损失、树模型基线、以及多种降解因子实验。

**📊 数据集**

使用了基于PRISM、USDA SSURGO、USGS 3DEP DEM与Sentinel‑2 LULC四因子合成的RUSLE代理土壤侵蚀数据集。

**📈 对比分析**

通过与公式参考、RF/XGBoost基线、直接预测以及公式特征输入模型的对比，RASPL在宏观R^2、Tail95 MAE及综合DRS上均优于直接预测；卷积版本在尾部鲁棒性方面表现最佳。

**⚠️ 局限性**

局限性包括对K因子缺失的鲁棒性仍有限，公式估计在高阶降解下的效果不佳，需要在更多地区和任务中进一步验证。

---

## 297. Causal Evidence Extraction and Triangulation in Crisis Reports using Large Language Models: A ReliefWeb-based Study

**arXiv ID:** 2608.04576 | [PDF](https://arxiv.org/pdf/2608.04576v1)

**作者:** Yuanjun Zhang `[一作]` (University of Oulu), Mourad Oussalah `[通讯]` (University of Oulu)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计了基于查询条件和片段定位的两阶段LLM提取流程，自动从长篇人道主义报告中抽取干预-结果关系，并通过灾难×来源细胞聚合的上下文保持三角化方法对证据进行跨情境综合。

**💡 创新点**

创新点在于将查询条件约束与片段归因结合到提取阶段，显著减少噪声，并提出基于细胞的上下文保持三角化，利用拉普拉斯平滑和极性归一化实现证据合成与可信度度量（LoE）。

**🔧 技术方法**

技术上采用两阶段LLM（如Qwen‑Plus、GPT‑4o‑mini、DeepSeek‑V3和Llama‑3.1‑8B‑Instruct）进行查询条件提取与片段归因分类，并对Llama进行LoRA微调；随后使用加权概率、等细胞权重和LoE计算实现聚合。

**📊 数据集**

数据集为2000‑2024年 ReliefWeb 上的 8,029 篇英文现金援助相关报告，辅以 100 篇专家标注的 220 条关系样本进行评测。

**📈 对比分析**

与基线单步提示和无查询条件的提取相比，查询条件+片段归因策略在 Qwen‑Plus 上获得最高加权 F1 90.73%，LoRA 微调后 Llama‑3.1‑8B‑Instruct 达到 94.15%，并在三角化中得到 LoE 0.865 的强正向收敛。

**⚠️ 局限性**

局限性包括仅针对现金援助关键词、仅英文、灾难与来源类型有限、评测样本规模小以及聚合设计（极性归一化、权重、平滑）对结果的潜在影响。

---

## 298. CommBench: Can LLMs Write Correct and Efficient GPU Communication Code?

**arXiv ID:** 2608.04450 | [PDF](https://arxiv.org/pdf/2608.04450v1)

**作者:** Shuang Ma `[一作]` (University of California, Davis), Yang Zhou `[通讯]` (University of California, Davis)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了CommBench——一个包含100+专家级GPU通信编程任务的综合基准，并提供自动化、作弊防护的评估框架；

**💡 创新点**

首次系统评估LLM在多GPU通信代码生成上的能力，提出联合质量-加权通过率（Quality-Weighted Pass Rate）作为综合评测指标；

**🔧 技术方法**

采用基于真实硬件的编译执行验证、迭代自我纠错反馈、以及几种量化指标（Pass Rate、PASS+Good、GM‑Speedup）进行评估；

**📊 数据集**

使用从行业级GPU通信库（MSCCL++, NCCL, NVSHMEM, ThunderKittens等）及实际LLM训练/推理工作负载中提炼的代码，形成100+个任务集；

**📈 对比分析**

与多种主流LLM（GPT‑5.5、Gemini‑3.1‑Pro、Claude‑Opus 4.7、GLM‑5.1等）在NVLink和RDMA环境下对比，GPT‑5.5的最佳Quality‑Weighted Pass Rate为0.467，Pass Rate 57.4%，PASS+Good 30.7%，显示当前模型在通信代码生成上仍有显著缺口；

**⚠️ 局限性**

受限于对专业库API的知识缺失、对稀有库的训练数据不足、以及难以实现的复杂通信模式，导致即使通过率高的模型在性能或正确性上也常出现大幅退化。

---

## 299. UC, Categorically: Rigorous Diagrammatic Proofs

**arXiv ID:** 2608.04521 | [PDF](https://arxiv.org/pdf/2608.04521v1)

**作者:** Pooya Farshim `[一作]`, Philip Wadler `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文将Canetti的静态系统UC框架用对称单子范畴（SMC）和字符串图（string diagram）重新表述，给出严格的图示证明，构造了可在更一般计算模型（如量子计算、DSL）上应用的统一安全组合理论，并证明其与原始UC等价。

**💡 创新点**

创新点包括：① 用SMC和字符串图实现可视化且可形式化的组合证明；② 将安全组合泛化到非单机对手、非ITM计算模型；③ 通过等价关系消除机器标识并允许环境与对手为网络；④ 在正式化过程中发现并纠正了原UC定义中的技术细节错误。

**🔧 技术方法**

技术手段主要是：对称单子范畴的语义化、字符串图语法、生成元和方程构造范畴、可区分性（computational indistinguishability）在范畴中的泛化、以及对等价类的归约和组合定理的证明。

**📊 数据集**

本文没有使用实验数据集，而是完全基于数学公理化和抽象构造；若要验证实现，可通过对称单子范畴的自动化工具（如Coq、Lean或其他图形化证明助手）进行形式化验证。

**📈 对比分析**

评价方法主要是与传统UC框架的等价性证明与对比；在理论上，组合定理的证明更简洁、可形式化；实验性能方面未给出，因为该工作属于理论研究。

**⚠️ 局限性**

局限性：① 仅覆盖静态系统的UC（不适用于动态网络）；② 对于更复杂的多方或量子UC的完整化仍未完成；③ 对手模型虽已放宽，但仍依赖于可在单台机器上模拟网络的假设；④ 目前的框架未能自然处理非对称通信、动态身份等细节。

---

## 300. Hallucinations on the Board: Tool-Augmented Evaluation of LLM Chess Commentary

**arXiv ID:** 2608.04240 | [PDF](https://arxiv.org/pdf/2608.04240v1)

**作者:** S. Ashwin Hebbar `[一作]` (Princeton University), Pramod Viswanath `[通讯]` (Princeton University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 ACT‑Eval 框架，对棋局注释进行原子级拆解并通过工具验证事实、覆盖率和移动质量评估。

**💡 创新点**

创新点在于将可计算的棋局事实与专家标注的原子知识结合，利用工具验证减少 LLM 幻觉，并提供可解释的多维度评估指标。

**🔧 技术方法**

使用了棋局分析工具套件（Stockfish、局面查询、合法性检查等）与 GPT‑5.4 作为判定者进行原子拆分与验证。

**📊 数据集**

构建了 325 个棋局‑移动对的基准，其中 125 个位置包含专家核实的原子注释，并涵盖教材、比赛和关键位置。

**📈 对比分析**

与传统 LLM‑as‑a‑judge 比较，ACT‑Eval 在事实精度上显著提升，工具辅助下误差率从 22% 降至 9%，但覆盖率仍低于 65%；模型在识别错误走法上表现良好。

**⚠️ 局限性**

局限在于工具集覆盖不足、判定者仍受 LLM 解释误差影响、原子拆分可能缺失上下文、金标注不完备，以及仅在棋类验证，可推广性待验证。

---

## 301. FinProBench: Evaluating Financial AI Agents with Role-Grounded Rubrics Derived from Professional Deliverables

**arXiv ID:** 2608.04077 | [PDF](https://arxiv.org/pdf/2608.04077v1)

**作者:** Ben Wang `[一作]` (Alibaba Cloud Computing), Chi Zhang `[通讯]` (Alibaba Cloud Computing)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了FinProBench金融专业任务评估基准，并提出基于真实专业交付物的角色导向评估标准构建（RGRC）流程

**💡 创新点**

创新点在于将评估标准直接从真实职业交付物中抽取，实现角色级标准的可复用；同时通过四阶段自动化流程将角色级标准转化为任务级评估表，降低任务级构建成本约6.7倍

**🔧 技术方法**

采用LLM进行交付物分析、能力提取、标准合成与验证，使用多轮自动化与人机交叉验证；评估时用四家LLM判别器进行二进制评分，计算加权分数

**📊 数据集**

使用了1,723份真实金融专业交付物（涵盖57个职业角色、161种交付物类型，8个子行业），并抽取20个完整任务（20个角色、7个子行业）进行评测

**📈 对比分析**

与三款通用LLM（Codex、Qoder、Claude Code）及人工交付物进行对比，评估指标为加权得分（0-100）及维度覆盖率；人工得分最高（73.7），三款模型均在70左右，95%置信区间重叠，说明差距不显著；各模型在不同维度表现互补

**⚠️ 局限性**

局限在于缺乏独立从业者人工标注，评估主要基于自动判别器；数据集主要为公开文档，可能过度代表公开职业规范；跨司法辖区的稳定性尚未验证

---

## 302. Zero-Instrumentation Dependency Discovery for Guided Microservice Migration Using eBPF

**arXiv ID:** 2608.04413 | [PDF](https://arxiv.org/pdf/2608.04413v1)

**作者:** Eshan Trivedi `[一作]` (Independent researcher), Chandrahasa Pranava `[通讯]` (Independent researcher)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用 eBPF 内核网络追踪实现无代码注入的运行时服务依赖图自动发现，并基于该图进行 spectral clustering 与 Kernighan‑Lin 细化得到的 VM 归属划分以及 ROI‑ranked 的迁移顺序，从而为微服务迁移规划提供基于真实流量的决策支持。

**💡 创新点**

创新点在于：①通过两遍 PID‑port 关联算法在同一用户进程共享的环境下恢复服务身份；②将 spectral clustering 与 KL 细化结合，以最小化跨 VM 边权；③提出 ROI 评分机制，将流量收益、迁移成本、依赖深度与负载波动综合评估，得到更优的迁移顺序。

**🔧 技术方法**

技术手段包括：eBPF 追踪内核 kprobe（connect/accept），NDJSON 事件采集，PID‑port 关联推断，窗口化流量计数，基于归一化拉普拉斯矩阵的 spectral embedding，k‑means 初始划分，Kernighan‑Lin 迭代细化，以及基于图流量与拓扑度量的 ROI 计算。

**📊 数据集**

实验数据集为作者自建的 20 服务微服务栈（端口 8001–8020），部署在两台 DigitalOcean 虚拟机上，采集 3 分钟内 13,615 次 TCP 事件，生成 32 条依赖边。

**📈 对比分析**

比较方法：在仿真中将 ROI‑ranked 迁移顺序与 alphabetic、most‑called‑first、reverse‑ROI 四种基线进行对比，计算每一步的跨 VM 流量占比 (cut%) 的曲线下的面积（AUC）。ROI‑ranked 的 AUC 为 301.7，较 alphabetic（413.9）低 27%；在吞吐量方面，eBPF 开启时吞吐下降仅 4.4%，但在 200 RPS 近饱和负载下 p50 与 p99 分别升高 383% 与 1050%，显示高负载时的尾部延迟显著。

**⚠️ 局限性**

局限性包括：仅在作者自建的 20 服务小规模测试环境中验证，未在真实生产依赖图上测试；迁移效果仅为仿真，未执行实际迁移；eBPF 收集在高负载下产生显著尾延迟，需在低负载或专用采样节点运行；未与 Istio/Jaeger 等现有服务网格或分布式追踪工具进行对比，缺乏对比性评估；扩展性和多端口/动态端口服务的适用性尚未验证。

---

## 303. The Sample Complexity of Distributionally Robust PAC Learning under Cressie--Read Divergences

**arXiv ID:** 2608.04686 | [PDF](https://arxiv.org/pdf/2608.04686v1)

**作者:** Elad Aigner-Horev `[一作]` (Ariel University), Roi Weiss `[通讯]` (Ariel University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

本文研究了在Cressie–Read散度约束下的分布不稳健PAC学习，给出了对0–1损失的可实现和不可实现样本复杂度上界与下界，并证明了在固定k>1时样本复杂度与鲁棒性半径ρ的精确依赖关系；

**💡 创新点**

创新点在于：① 在Cressie–Read阶数k>1的通用框架下闭合了鲁棒PAC学习的上下界，② 通过保留事件膨胀映射的两支分支，精确捕捉了鲁棒性对估计精度的尺度敏感性，③ 在ρ→0时恢复经典PAC学习速率，且阐明了k=2处的临界转折；

**🔧 技术方法**

主要技术包括：事件膨胀映射的二值化简化、对Cressie–Read散度的解析求解、两支分支的不等式界定、VC维度的相对偏差上界与下界构造、以及对鲁棒风险偏差的尺度敏感控制；

**📊 数据集**

本文为理论性工作，没有使用任何实验数据集；

**📈 对比分析**

与以往使用全局上界（忽略误差尺度）或在Shapiro双重表示下求解的结果相比，本文的上界在ρ>0时保持与ρ、k、VC维度的正确幂次关系，且在不可实现情况下可达到近似最优；

**⚠️ 局限性**

局限性包括：① 分析仅适用于固定k>1，k趋近1（KL散度）仍未给出完整的样本复杂度；② 结果为上界与下界相匹配但仍含有对数因子；③ 对非二值化损失或其他f-散度的推广尚未完成。

---

## 304. Mind the Cap: Output-Budget Regimes Change the Measured Multilingual Reasoning Gap

**arXiv ID:** 2608.04160 | [PDF](https://arxiv.org/pdf/2608.04160v1)

**作者:** Ankit Goyal `[一作]`, Jaideep Ray `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对多语言大型语言模型（Qwen3-8B 与 Llama-3.1-8B-Instruct）在 MGSM（德语、泰语、斯瓦希里）任务中，比较在不同 token 预算（硬截断）下 Native 与 Translate-Act 两种推理策略的精度，并通过预算扫掠、独立硬截断解码、长度归一化等方法量化 token 预算对结果的影响。

**💡 创新点**

创新点在于将输出 token 上限视为实验中的独立变量，对预算敏感性进行系统性分析；提出通过长度归一化和预算绑定区间来解释多语言推理差距，使用冻结测试与独立解码验证方法，揭示紧预算会导致 Native 表现被 token 预算所掩盖，且长度归一化可逆转策略排序。

**🔧 技术方法**

技术手段包括：预算扫掠（B∈{64,…,4096}）、硬截断独立解码、Holm 多重校正、Bootstrap 置信区间、COMET 翻译质量评估、GlotLID 语言识别、答案格式解析、词表扩展（跨语言 token）以及 vLLM 解码对比。

**📊 数据集**

使用的数据集为：MGSM（250 题/语种，德语、泰语、斯瓦希里）；FLORES-200 用于计算 token premium；COMET 用于翻译质量评估；GlotLID 用于 trace 语言识别；并在 Qwen3-8B 与 Llama-3.1-8B-Instruct 两个 8B 模型上进行实验。

**📈 对比分析**

比较方法：在每个预算下分别评估四种策略（NATIVE、TRANSLATE-ACT、PIVOT、CODE‑SWITCHED）的精度，计算 Native 与 Translate-Act 的差距 Δ_L(B)。实验结果显示：在紧预算（如 B=128/256）下，Native 在某些语言中占优势，长度归一化可将差距扩大或逆转；在 B=1024 时差距几乎消失；在峰值预算下（德语 B=192、泰语 B=256、斯瓦希里 B=128）差距分别可达 34.6、38.6、13.7 点，显著高于平常的 5 点显著差异阈值。

**⚠️ 局限性**

局限性：实验仅覆盖 MGSM 任务、三种语言和两款 8B 模型；词表扩展仅为 counterfactual，未涉及模型微调；预算公布对行为的影响仅在固定 cap 下探测；结果多为探索性，未验证在更广泛任务、提示或训练方法上的推广；翻译质量与 token premium 的混淆仍存在；未解决 Native 失败原因的完整因果分析。

---

## 305. GUARD: Grounding Uncertainty and Ablation-Based Risk Detection for Diffusion-Based VLAs

**arXiv ID:** 2608.04510 | [PDF](https://arxiv.org/pdf/2608.04510v1)

**作者:** Suhas Hegde `[一作]` (Bosch Research India), Jitendra Yasaswi Bharadwaj Katta `[通讯]` (Bosch Research India)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在测试时对视觉-语言-动作（VLA）策略进行失败检测，利用KV缓存的梯度显著性评估并通过对最重要的视觉和语言token进行平均值置换来构造反事实，从而检测动作生成是否真正依赖于当前任务的多模态证据。

**💡 创新点**

创新点在于将失败检测转化为对动作生成对关键信息的功能耦合度的直接测量，提出了基于KV缓存显著性和反事实置换的敏感度、模态偏差、注意熵和基础效率等诊断指标，并通过在线校准和时间序列分类实现可部署的实时警报。

**🔧 技术方法**

使用技术包括：梯度显著性（saliency）反向传播、对KV缓存进行平均值置换的反事实探测、注意熵与敏感度的归一化、轻量级时间序列分类器（GRU/LSTM/Transformer）以及功能式保形预测用于阈值校准。

**📊 数据集**

实验数据集包括LIBERO、SimplerEnv、MetaWorld和PhysicalAI-AV，涵盖了桌面操作、不同身体结构、物体交互以及自动驾驶四个任务域。

**📈 对比分析**

与SAFE、欧氏/余弦距离、马氏距离、PCA-KMeans、LogpZO、FIPER、STAC等多种基线进行比较，GUARD在未见任务的ROC‑AUC平均达88.84%（比FIPER高5.73个百分点），并在多种设置下实现了更高的平衡准确率和更早的检测时间，整体性能显著优于传统基线。

**⚠️ 局限性**

局限性在于仅适用于基于扩散的动作生成模型，无法直接应用于自回归VLA；同时梯度显著性需要保存中间激活，导致峰值显存占用提升，需要进一步的内存优化与高效探测。

---

## 306. BrainBench: Benchmarking Large Language Models for Comprehensive EEG Understanding

**arXiv ID:** 2608.04156 | [PDF](https://arxiv.org/pdf/2608.04156v1)

**作者:** Yangxuan Zhou `[一作]` (Zhejiang University), Gang Pan `[通讯]` (Zhejiang University)

**通讯引用:** 484837 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

引入统一的可指令化EEG理解基准BrainBench，并在该基准上评估多种LLM的性能

**💡 创新点**

创新点包括：①将EEG评估从单一解码任务扩展为全面指令驱动理解；②设计四大子集与六类验证单元，实现多维度、可复现的评估；③对比自主代码执行（CodeAct）与结构化代理执行（BrainAgent）两种范式

**🔧 技术方法**

采用大语言模型与多代理工具链、Python代码生成、验证器、六类验证单元（数值、类别、集合、序列、语义、工件）

**📊 数据集**

使用17个公开EEG数据集，涵盖基础分析、睡眠评估、神经认知评估与多模态整合，总计超过17k实例

**📈 对比分析**

在相同指令、数据与评估标准下，分别在BrainAgent与CodeAct两种范式下对13种LLM进行评测，分数在0-100之间，平均分约70-80，且随任务难度提升性能显著下降

**⚠️ 局限性**

局限性：整体最高分仍低于80，难以处理高难度多步骤分析；结构化执行在最难任务中的优势减弱；LLM在长程推理与科学证据整合方面仍受限

---

## 307. LaPrune: Controllable Differentiable Sparsity at Million Scale

**arXiv ID:** 2608.04057 | [PDF](https://arxiv.org/pdf/2608.04057v1)

**作者:** Jakub Antczak `[一作]` (Wrocław University of Science and Technology), Jacek Tabor `[通讯]` (Jagiellonian University)

**通讯引用:** 2327 | [OpenAlex ID](https://openalex.org/A5052791052)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种可微分的Top‑k层LaPrune，能够在保持稀疏预算的同时通过归一化的第二矩控制掩码的硬度。

**💡 创新点**

将预算约束与稀疏度硬度解耦，使用归一化的第二矩γ控制掩码硬度并保证预算不变；提出预测掩码饱和比例的均值场模型和分布无关的最坏情况下限。

**🔧 技术方法**

利用Laplace‑CDF软Top‑k、隐式函数求解（Newton+二分）、矩分析、均值场预测与理论下界；实现了GPU高效且可微的前向/后向实现。

**📊 数据集**

在合成高斯分布、200维特征选择实验、ResNet‑18/CIFAR‑100 激活的稀疏自编码器，以及 n=10⁷ 规模数据上进行实验。

**📈 对比分析**

与LapSum、DFTopK、硬top‑k、SoftMax/Entmax等基线比较；在特征恢复、稀疏自编码器和CIFAR‑100 Top‑k分类等任务中，LaPrune 在保持预算的前提下获得更高的特征恢复F1、更低的FVU和更高的probe准确率，硬度调节显著提升性能。

**⚠️ 局限性**

仅基于Laplace‑CDF掩码并只控制单一二阶矩，难以覆盖更复杂的稀疏度约束；在接近硬端点时数值不稳定；未对优化动态或其他掩码家族扩展理论。

---

## 308. CofactVLA: Deconfounding Vision-Language-Action Models via Counterfactual Intervention

**arXiv ID:** 2608.04396 | [PDF](https://arxiv.org/pdf/2608.04396v1)

**作者:** Yan Zhang `[一作]` (Tsinghua University), Jungong Han `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了CofactVLA，一种通过双路径反事实干预来消除视觉覆盖（vision‑override）现象的视觉‑语言‑动作模型框架。

**💡 创新点**

创新点在于构建Dual‑path Deconfounding Graph (DDG)，并在同一前向传递中同时采用Action‑Level Orthogonal Projection Guidance (OPG) 与 Feature‑Level Counterfactual Covariance Reduction (CCR)，实现对视觉混淆因子在特征层与动作层的因果去混。

**🔧 技术方法**

使用了反事实干预、连续流匹配（flow‑matching）、正交投影指导、协方差差异惩罚、双分支语言掩码设计，以及对VLM关键值/注意力特征的动态截断。

**📊 数据集**

在模拟实验中使用LIBERO和LIBERO‑Plus基准数据集；在真实机器人实验中使用AgileX PiPer 6‑DoF机械臂与Intel RealSense摄像头进行四个多目标操纵任务。

**📈 对比分析**

与OpenVLA、π₀、π₀.₅、X‑VLA等SOTA方法对比，CofactVLA在LIBERO平均成功率达到98.5%（Object 100%、Spatial 99%），在LIBERO‑Plus OOD总成功率69.1%显著高于π₀（53.6%）；在真实机器人实验中标准环境平均成功率90.8%（高于π₀.₅的71%），OOB环境提升至75.8%（比π₀.₅的23.5%提升52.3%绝对）。

**⚠️ 局限性**

局限性包括：依赖基础VLM的零样本对齐，无法学习未见概念；对严重遮挡（如机械臂遮挡摄像头）仍易失效；缺乏多视角融合提升遮挡鲁棒性。

---

## 309. Active Learning Guided Design Space Refinement for Scalable Multi-Objective Bayesian Optimization in Materials Discovery

**arXiv ID:** 2608.04651 | [PDF](https://arxiv.org/pdf/2608.04651v1)

**作者:** Alexandros Ntagiantas `[一作]` (National Centre for Scientific Research Demokritos), Panagiotis Krokidas `[通讯]` (National Centre for Scientific Research Demokritos)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在大规模材料优化中，提出了基于主动学习的自适应搜索空间细化框架，并将细化后的子空间用于多目标贝叶斯优化以加速收敛。

**💡 创新点**

创新点在于将单分类器XGBoost与DAGS主动学习、置信度与不确定性结合，形成一种自适应的候选过滤与热启动策略，显著缩小搜索空间并保持Pareto信息。

**🔧 技术方法**

采用XGBoost分类器、DAGS主动学习、置信度与不确定性估计、伪标签、高置信度筛选，以及多目标贝叶斯优化（qNEHVI）和热启动。

**📊 数据集**

评估数据集包括：1）52,272个CFRP层压板的压力容器设计（3个目标），2）69,839个共价有机框架（COF）在CH₄/N₂吸附条件下的性能（2个目标）。

**📈 对比分析**

与全空间BO直接对比。实验显示细化后搜索空间约缩小45–50%，保留99%以上原始超体积；在150次额外BO评估下，Pareto-AUC提升约36%（容器）和26%（COF），超体积提升不大但早期收敛更快。

**⚠️ 局限性**

局限性包括：若分类器早期误判可能丢失重要候选；需要额外的主动学习查询，整体评估成本略增；未给出理论保证的Pareto保留上限，且实验基于已知目标的离线数据，在线实际应用需要适配。

---

## 310. Patients-like-me: A Variational LM--GNN Framework for Explainable Clinical Prediction

**arXiv ID:** 2608.04193 | [PDF](https://arxiv.org/pdf/2608.04193v1)

**作者:** Xinyu Wang `[一作]` (McGill University), Ziyang Song `[通讯]` (Ohio University)

**通讯引用:** 89 | [OpenAlex ID](https://openalex.org/A5104281960)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了统一的语言模型–图神经网络框架 PLM，用于电子健康记录（EHR）的临床预测，并通过可解释的“像我一样的患者”来说明预测原因。

**💡 创新点**

创新点在于：① 通过变分期望最大化（VEM）算法实现 LM 与 GNN 的交替、联合训练，使局部语义与全局人群结构相互迭代提升；② 在 GNN 上引入基于诊断码重叠的患者图，并利用梯度重要性评分为每个患者检索并解释最具影响力的参考患者；③ 该框架对 encoder‑only 与 decoder‑only 语言模型均保持性能提升，显著降低训练开销。

**🔧 技术方法**

使用技术包括：Transformer‑based 语言模型（BioClinical ModernBERT、Meerkat 等），Graph Convolutional Network（GCN）对患者图进行信息传播，变分 EM（VEM）优化策略，诊断码重叠构建患者图，梯度重要性评分与边缘遮蔽验证等。

**📊 数据集**

实验数据集为公开的大规模 EHR 数据集 MIMIC‑III 与 MIMIC‑IV。

**📈 对比分析**

与多种基线（Deepr、RETAIN、GRAM、StageNet、AdaCare、GRASP 等）以及 LM‑GNN 方法（G‑BERT、LEADER、GLEM、GraphCare、KARE、ColaCare 等）比较。PLM 在读入预测、住院时长预测和药物推荐三项任务上均取得最高或最接近最高的 AUPRC/AUROC/F1/ Jaccard 等指标，提升幅度可达 2–5% 以上，并且在训练时间和 GPU 内存上仅增加 4–5% 的开销。

**⚠️ 局限性**

局限性：患者图仅基于诊断码重叠，未充分利用多模态 EHR（药物、实验室、影像）或外部医学知识；参考患者解释的可解释性仅通过定量遮蔽评估，缺乏临床专家的真实评估。

---

## 311. LiverPlan: A Stage-Adaptive Immersive Visual Analytics Framework for Anatomical Liver Surgical Planning

**arXiv ID:** 2608.04707 | [PDF](https://arxiv.org/pdf/2608.04707v1)

**作者:** Qixuan Liu `[一作]` (Chinese University of Hong Kong), Pheng-Ann Heng `[通讯]` (Chinese University of Hong Kong)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文设计了一套基于XR的分阶段可视化分析框架，用于肝脏解剖切除（ALR）的术前规划。

**💡 创新点**

创新点包括：①将ALR规划拆解为三阶段的认知流程，并为每阶段量身定制可视化与交互；②在XR环境中实现色彩保真渲染、上下文保护聚焦、直接三维操控和嵌入式安全指标可视化；③通过焦点+上下文的血管交叉可视化帮助医生建立预判模型。

**🔧 技术方法**

采用了Meta Quest 3移动XR设备，6-DoF 手柄交互；GPU加速体素化、SDF 计算FLR；自定义色彩保真渲染、血管骨架抽象；实时嵌入式 FLR/RM/TSCR 指标显示；Unity 2022.3 与 MRTK3 开发框架。

**📊 数据集**

使用匿名病人肝脏三维分割数据（包括 Couinaud 分段、动静脉与肿瘤），共 4 例临床案例（2 简单 2 复杂）用于评估。

**📈 对比分析**

与基线桌面系统（3D Slicer）进行对比；结果显示：任务完成时间下降 51.69%（Cohen d = 1.86），主观工作量下降 32.63%（d = 0.92），可用性评分提升 98.36%（d = 1.89）。各阶段分别实现了 56%–62% 的时间节省与显著的工作量减轻。

**⚠️ 局限性**

局限性包括：仅支持平面切除界面；XR 设备佩戴时间久会导致疲劳；需要先行熟悉设备与交互；依赖高质量医学图像与分割结果；专家样本量有限，结果可能不易推广。

---

## 312. Privacy-Preserving Action Recognition: Taxonomy, Methods, and Privacy-Utility Trade-offs

**arXiv ID:** 2608.04501 | [PDF](https://arxiv.org/pdf/2608.04501v1)

**作者:** Sareer Ul Amin `[一作]` (Chung-Ang University), Sanghyun Seo `[通讯]` (Chung-Ang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对 2018–2026 年间 32 篇关于隐私保护动作识别（PPAR）的论文进行了系统综述，并提出了统一的隐私空间分类与评估协议。

**💡 创新点**

创新点包括：①基于 PRISMA 的系统检索与筛选流程；②双维度（注入层与机制家族）的隐私空间分类；③统一威胁模型与标准化评估框架；④经验性隐私–效用 Pareto 前沿分析；⑤面向部署的路线图。

**🔧 技术方法**

技术手段主要是 PRISMA 检索、统计与聚类分析，构建了包含 cMAP、ε‑DP 等指标的统一评估协议，并对多种隐私注入机制（对抗学习、骨架、加密、差分隐私等）进行对比。

**📊 数据集**

评估使用的主要数据集包括公开动作识别基准 HMDB51、UCF101、Kinetics 以及其隐私增强版本 PA‑HMDB51、VP‑HMDB51、VP‑UCF101 等。

**📈 对比分析**

通过对 32 篇论文按注入层与机制家族分类并量化 Top‑1 识别准确率与 cMAP 等隐私指标，发现骨架/模态方法可在保持 85–95% 识别准确率的同时实现 60–80% 隐私提升；对抗学习方法可维持 70–80% 识别准确率；差分隐私方法则效用低于 70%。

**⚠️ 局限性**

局限性在于评估碎片化：仅 10% 论文采用正式隐私定义，攻击模型多为弱级别，跨数据集、适应性攻击和实时边缘部署的实验不足；方法多聚焦单一属性且缺乏大规模基准与公开代码，导致结果难以直接比较与复现。

---

## 313. Manipulation-Proof Oblivious Audits against Deceptive Model Providers

**arXiv ID:** 2608.04365 | [PDF](https://arxiv.org/pdf/2608.04365v1)

**作者:** Augustin Godinot `[一作]` (INRIA Centre de l'Université de Rennes), Sébastien Gambs `[通讯]` (Université du Québec à Montréal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种名为 respir 的无记忆审计协议，利用私有信息检索（PIR）隐藏审计集，从而提升对机器学习模型公平性审计的抗操纵性。

**💡 创新点**

创新点在于将 PIR 与公平性审计相结合，使审计者无法确定具体被检验样本，迫使模型提供者在更大比例的候选集上修改输出，从而显著提高操纵成本和检测概率。

**🔧 技术方法**

主要技术包括基于 LWE 的 SimplePIR/VeriSimplePIR、可验证的 PIR 证明，以及概率论工具（Hoeffding/Serfling 置信界）用于证明操纵难度提升。

**📊 数据集**

实验使用了三组数据集：Credit Card Default (CCD)、COMPAS 以及 HateDay 文本数据集，分别针对不同的敏感属性进行公平性评估。

**📈 对比分析**

与传统黑盒公平性审计在相同审计集规模下比较，respir 需要的输出翻转数提升数倍，检测概率提升约两倍；在候选集足够大且群体平衡的场景中效果最为显著。

**⚠️ 局限性**

主要局限包括：需要候选集具有代表性和足够平衡；若候选集不充分，审计结果可能不具普适性；PIR 实现仍带来通信和计算开销，并且无法完全消除所有策略性操纵的可能。

---

## 314. Relevant but Incomplete: Referential Dangling as a Paradigm-Level Failure Mode in Hard Prompt Compression

**arXiv ID:** 2608.04569 | [PDF](https://arxiv.org/pdf/2608.04569v1)

**作者:** Zhengpei Hu `[一作]` (Qinghai University), Jianqiang Huang `[通讯]` (Qinghai University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在硬性提示压缩过程中出现的“参照悬挂”（referential dangling）现象，分析其原因并评估其在多跳问答任务中的普遍性。

**💡 创新点**

创新点在于首次正式提出并量化参照悬挂问题，揭示独立评分方式会导致重要句子被保留但其必要的前置信息被删除，并证明通过重新选择或自动恢复缺失句子能显著提升下游模型性能。

**🔧 技术方法**

采用硬性压缩算法（Beaver、PartPrompt、Selective‑Context、LLMLingua‑2、LongLLMLingua、DAC）进行句子/块级别的自适应选择，并利用Qwen3‑8B、Llama‑3.1‑8B、Mistral‑7B、GPT‑5.5等大型语言模型进行下游推理，此外设计了基于句子级别的自动恢复分类器。

**📊 数据集**

使用的公开数据集包括 HotpotQA、2WikiMultiHopQA、MuSiQue 以及 LongBench‑v2 Single‑Document QA。

**📈 对比分析**

通过比较压缩后原始上下文、重新选择（re‑selected）和完整支持（full support）三种情况，实验显示在 r=0.30 的压缩比例下，重选能提升 Qwen3‑8B 的准确率约 29–34 点，自动恢复则额外提升 4.7 点；在不同压缩器与模型上均能观察到类似的性能提升。

**⚠️ 局限性**

局限性在于目前的自动恢复策略主要针对 Beaver 输出和已标注支持句子，未能在其他压缩器或无标注的多跳问答场景中实现良好迁移；此外参照悬挂的诊断仍依赖内容词覆盖率阈值，可能无法捕捉所有依赖关系。

---

## 315. AudioScape-TTA: A Structured Soundscape Benchmark for Fine-Grained Text-to-Audio Evaluation

**arXiv ID:** 2608.04479 | [PDF](https://arxiv.org/pdf/2608.04479v1)

**作者:** Jinting Wang `[一作]` (Hong Kong University of Science and Technology), Li Liu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了AudioScape‑TTA，一个针对文本到音频生成的结构化、复杂度感知评测基准，并构建了对应的细粒度评测框架；

**💡 创新点**

创新点在于将音景拆分为场景、背景音乐、音效与语音四大模组，使用事件密度与结构复杂度两维度刻画样本复杂度，并通过二进制评判规则（事件存在、属性匹配、语音内容）实现可解释的细粒度评测；

**🔧 技术方法**

主要技术包括：利用LLM（如Qwen3.5‑27B）自动化生成结构化描述与评测规则；使用Qwen3‑Omni‑Instruct和Qwen3‑ASR对生成音频进行语义验证；以及传统的全局相似度指标（CLAP、FAD、Inception等）做对比；

**📊 数据集**

数据集为自建的AudioScape‑TTA，共2258条音频-文本对，包含25707条二进制评测规则，平均时长约10秒，平均文本长度32词；

**📈 对比分析**

与传统的全局相似度指标对比，基于规则的满足率（SR）在细粒度上更能反映模型的语义遵循能力；在13个开源TTA模型中，Foley‑Omni以79.62% SR位列榜首，显示出优异的事件属性控制与语音内容保留；

**⚠️ 局限性**

局限性包括：语音内容保留仍难，除Foley‑Omni与Dasheng AudioGen外其余模型SR为0；属性控制远不及事件生成，且在多模态组合与高复杂度样本中性能显著下降；评测依赖LLM与ASR，可能受到模型偏差影响。

---

## 316. MINT: Tensor Decomposition on Stacked Recurrence Matrices for Time Series Data Mining

**arXiv ID:** 2608.04157 | [PDF](https://arxiv.org/pdf/2608.04157v1)

**作者:** Kaamil Kaka `[一作]` (University of Texas at Dallas), Vikram Jayaram `[通讯]` (Neuralix AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了基于张量化自相似矩阵（MINT）的方法，用于挖掘多传感器时间序列中的跨传感器共模模式。

**💡 创新点**

将自相似矩阵堆叠成三阶张量并利用可解释的CP分解，首次实现对跨序列子序列共同形状与季节性的可解释共聚类。

**🔧 技术方法**

采用SCAMP生成Mplot、非负稳健PCA提取低秩结构、CPD（CANDECOMP/PARAFAC）张量分解、Kneedle寻找秩、以及均值中心化与噪声估计等技术。

**📊 数据集**

在台北捷运客流、加州大型交通、欧洲电力需求、风机监测以及电力负荷等五个真实多传感器数据集上进行实验。

**📈 对比分析**

与单系列矩阵概况方法对比，使用四项共聚类属性检验，实验中70%–100%属性满足，成功识别圣诞节与季节性需求、交通高峰等模式，表明方法在解释性和跨系列发现上表现优异。

**⚠️ 局限性**

仅支持欧氏距离自相似、需先行均值中心化、对异构传感器适用性有限、鲁棒PCA与CPD计算开销大、未在多变量或其他距离度量上验证。

---

## 317. Spend Bits Where Queries Look: KV Cache Vector Quantization with Attention-Preserving Transforms

**arXiv ID:** 2608.04074 | [PDF](https://arxiv.org/pdf/2608.04074v1)

**作者:** Samuel Fernández-Menduiña `[一作]` (University of Southern California), Salman Avestimehr `[通讯]` (University of Southern California)

**通讯引用:** 7844 | [OpenAlex ID](https://openalex.org/A5047191296)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种针对大语言模型KV缓存的压缩方案，以减少显存占用并支持更长上下文；

**💡 创新点**

创新点在于基于注意力logit误差推导出非正交键变换，并提出体积均衡分组与向量量化策略；

**🔧 技术方法**

使用了非差异化的变换编码（Companding）、向量量化（k‑means）、体积均衡分组和Sink/Recent BF16带等技术；

**📊 数据集**

主要在公开的GPQA‑Diamond、RULER NIAH、LongBench、HumanEval、LiveCodeBench等数据集上进行校准与评估；

**📈 对比分析**

与OSCAR、BF16、INT2等基线方法对比，在相同压缩率下能在六大评测集上保持与全精度相近的准确率，同时在吞吐量上可与标量量化方案相当或更快；

**⚠️ 局限性**

局限性包括对高分辨率理论的依赖（在低比特率下可能失效）、对键与值的分步优化未全局最优、以及对键/值相关性假设的严格要求。

---

## 318. Dart: An Automated and Reproducible Environment Toolkit for DNS Protocol Analysis

**arXiv ID:** 2608.04498 | [PDF](https://arxiv.org/pdf/2608.04498v1)

**作者:** Yunyi Zhang `[一作]` (Tsinghua University), Haixin Duan `[通讯]` (Tsinghua University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个自动化、可复现的DNS协议分析工具包DART，利用声明式配置和容器化技术快速搭建多实现的测试环境；

**💡 创新点**

创新点包括：①将DNS多实现的异构配置抽象为统一的声明式语法；②内置自动化DNSSEC生命周期管理；③提供可插拔资源接口以支持闭源DNS软件；④集成eBPF观测模块，实现低开销的实时监控；⑤自定义Payload生成器（IP碎片、伪造等）和热加载/钩子机制；

**🔧 技术方法**

技术手段：Docker/容器化、YAML配置-as-code、自动依赖注入、插件化架构、eBPF实时采样、脚本钩子、DNSSEC密钥自动生成与签名、网络层模拟与数据包注入；

**📊 数据集**

使用的数据集包括：20篇DNS安全论文的实验环境配置、Tranco Top‑1 000与Top‑1 M域名、TsuKing实验拓扑、以及自定义的10个RFC合规性测试用例T1–T10；

**📈 对比分析**

评估方法：①对比DART与Docker Compose、EVE‑NG、Containerlab、SEED的环境构建效率和配置行数；②通过BIND 9.20.3在不同观测模式下测量100 QPS时延与峰值吞吐；③在20个DNS攻击案例中复现实验并统计成功率；结果显示：配置行数比手动方法减少约90%，观测模块对延时影响<0.01 s、吞吐下降<8%，且实验结果完全可复现；

**⚠️ 局限性**

局限性：目前仅支持四大主流解析器（BIND、Unbound、PowerDNS、Knot），对DoH等加密传输支持尚不完整；插件化需要手动编写，闭源软件的兼容性仍受限；在极大规模拓扑下的性能与资源消耗尚未充分验证；

---

## 319. muSync-GS: Physics-Synchronized Driving Video Synthesis for Weather and Geometric Road Hazards

**arXiv ID:** 2608.04412 | [PDF](https://arxiv.org/pdf/2608.04412v1)

**作者:** Yang Chen `[一作]` (Rochester Institute of Technology), Zilin Bian `[通讯]` (Rochester Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了 muSync‑GS，一个通过共享路面状态、路面几何曲线和车辆路径位置实现视觉与物理同步的驾驶视频合成框架，能够在雨雪等恶劣天气和路面凸起/凹陷等几何危害下生成车辆动力学响应与摄像机轨迹一致的合成视频，并提供帧级物理注解。

**💡 创新点**

创新点在于：①将天气、路面几何与车辆动力学通过同一共享状态联动，打破传统仅视觉或仅物理的耦合局限；②实现了雨雪对摩擦、滑移、制动和车身俯仰的连续化、可控映射；③通过半车四自由度模型将路面几何变化直接映射到车辆姿态和摄像机运动，完成全流程的物理同步渲染；④在同一场景上实现了视觉编辑与物理响应的双向可控性。

**🔧 技术方法**

技术组合包括：3D Gaussian 车场景重建与编辑；预训练的天气模块（WeatherEdit、RainyGS、Weather‑Magician）生成雨雪视觉效果；基于 CarSim 的车辆动力学模型（半车四自由度、轮胎曲线、ABS 逻辑）；预计算的路面表面状态函数与路面高度曲线；基于仿真路径的摄像机轨迹同步渲染管线；以及 DOVER、CLIP‑ρ 等无参考视觉质量评估工具。

**📊 数据集**

使用的数据集包括：Waymo Open Dataset（重建序列用于视频同步与可控性评估）、A2D2（真实车辆制动事件用于实车验证）、CarSim（19 个开发案例 + 12 个 hold‑out 用于物理精度评估）。

**📈 对比分析**

与 Cosmos、LTX‑Video、VACE 等视频编辑基线在同一场景、相同时间窗口下进行对比。muSync‑GS 在速度、俯仰、滑移和每轮载荷的 RMSE 与 CarSim 的差异分别为 0.0273 m/s、0.0590°、0.0101 与 26.61 N，显著低于基线。视频同步评估中，muSync‑GS 的路面几何 NRMSE（速度凸起 0.316、路面凹陷 0.431）比基线低约 2‑3 倍；在真实车辆制动事件中的俯仰误差仅为 0.062°，相关性高达 0.901；视觉质量方面，DOVER 与 CLIP‑ρ 分别显示 muSync‑GS 在控制一致性和局部编辑上最优。

**⚠️ 局限性**

局限性包括：①仅建模纵向动力学与轮胎摩擦，缺乏横向（侧向）动态与碰撞；②依赖预先设定的车辆/轮胎参数，缺少测量的轮胎‑路面摩擦数据；③只支持雨雪两种天气，未覆盖冰雪、雾等复杂条件；④对不同车型、轮胎规格和道路材质的泛化性有限；⑤目前参数需要人工标定，缺少端到端自适应机制。

---

## 320. Neuro-Symbolic Proof-of-Vulnerability Generation with Open-Weight Models

**arXiv ID:** 2608.04217 | [PDF](https://arxiv.org/pdf/2608.04217v1)

**作者:** Yu Nong `[一作]` (University at Buffalo), Haipeng Cai `[通讯]` (University at Buffalo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种低成本的神经符号PoV（Proof‑of‑Vulnerability）生成框架 PoVGen，先利用细化的 Open‑Weight LLM 对漏洞表征点进行定位，再通过锚定随机步行进行路径敏感搜索，最后使用 LLM 提取并求解路径约束，配合 SMT 求解器生成触发输入。

**💡 创新点**

创新点包括：① 在漏洞表征点上进行语义聚焦，显著缩小搜索空间；② 对约束提取和输入合成进行专门的模型细化，避免通用 LLM 的“胡言乱语”；③ 本地部署 Open‑Weight 模型，消除每样本 API 成本；④ 通过 PoV 的生成验证补丁，发现六个失效补丁并报告了五个未公开的 CVE，展示了实用价值。

**🔧 技术方法**

使用的技术主要有：Llama‑3.2‑3B 与 Llama‑3.1‑8B 的 Fine‑Tuning、Joern 与 SySeVR 的程序切片、SVF 的 ICFG 构建、KLEE 与 Z3 的符号执行与 SMT 约束求解、GLM‑5.2 进行上下文函数扩展、Anchored Random Walk（锚定随机步行）用于路径搜索。

**📊 数据集**

训练与评估数据集包括：InterPVD（漏洞补丁与表征点数据），SV‑COMP 与 Juliet Test Suite（约束生成与输入训练），ARVO 490 个样本（带 PoV 的基准），以及 250 个公开 CVE（无 PoV）作为真实场景验证。

**📈 对比分析**

在 ARVO 基准上，PoVGen 在带补丁模式下实现 78.98% 的成功率，补丁缺失模式下仍达 65.10%，显著优于 LibFuzzer（≤50.20%）、AFL++/AFLGo（≤44.49%）和 KLEE（仅 2.45%）。与直接调用前沿 LLM（OpenAI‑o3、Claude‑4、Gemini）进行的 Prompt 对比，PoVGen 成功率提升至 35.89% 以上。平均每个漏洞 98 分钟完成，API 成本仅 $0.04，且在 250 个真实 CVE 中实现 74.80% 的重现率。

**⚠️ 局限性**

局限性主要在于：① 目前仅针对 C/C++ 的内存安全漏洞，缺乏对其它漏洞类型和语言的验证；② 仍依赖静态分析与补丁定位，对深层调用链或复杂输入生成存在挑战；③ 在极大代码基上路径爆炸仍可能影响效率；④ 需要手工准备训练集与模板，扩展到新领域需额外工作。

---

## 321. Efficient Algorithms for the Bottleneck Path Problem in Geometric Graphs

**arXiv ID:** 2608.04203 | [PDF](https://arxiv.org/pdf/2608.04203v1)

**作者:** Matthew J. Katz `[一作]` (Ben Gurion University Of The Negev), Micha Sharir `[通讯]` (Tel Aviv University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了在平面方向天线图和 1.5 维地形可见性图两种几何场景下，求解最小瓶颈路径问题的近线性决策算法和随机期望 O*(n^8/7) 的优化算法。

**💡 创新点**

创新点在于构造多层动态数据结构（Voronoi、半平面范围查询、正交范围树）实现近线性 BFS，并结合 shrink-and-bifurcate 技术突破传统 O*(n^4/3) 的瓶颈，得到更快的最小瓶颈路径和 bounded‑hop 版本。

**🔧 技术方法**

主要技术包括多维正交范围树、动态 Voronoi 图、半平面范围报告、旋转复制、参数搜索与 shrink‑and‑bifurcate 组合。

**📊 数据集**

论文未使用具体实验数据集，全部以理论分析为主，假设点集为一般位置并满足常数角度下限。

**📈 对比分析**

与之前的 O*(n^4/3) 算法相比，本工作在决策阶段实现 O(n log^6 n) 的 BFS，整体最优路径求解时间提升至 O*(n^8/7)，在大规模实例上明显更快。

**⚠️ 局限性**

局限性包括对天线角度设有常数下限、算法实现极其复杂、随机化期望时间以及对 1.5D 地形的特殊性，无法直接推广到更一般的高维或任意角度场景。

---

## 322. TRNet: Topography-Guided Frequency Rectification and Structure-Aware Decoding for Multimodal Paddy Rice Segmentation

**arXiv ID:** 2608.04154 | [PDF](https://arxiv.org/pdf/2608.04154v1)

**作者:** Kaiwen Xiao `[一作]` (Sichuan University), Yanfeng Su `[通讯]` (Sichuan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在多源高分辨率遥感图像上，本文提出了TRNet网络，利用地形信息对视觉特征进行频率级别的修正，并通过结构感知解码器提升稻田分割的边界与内部一致性。

**💡 创新点**

创新点包括：① 将地形作为上下文而非细节特征，构建视觉专用编码器与地形辅助编码器的非对称架构；② 设计Topographic Energy‑Spectral Rectification（TESR），对低频做FiLM调制、高频做基于坡度的门控抑制；③ 设计Topography‑guided Paddy Structure Decoder（TPSD），将边界、内部深度与粗略坡度上下文联合学习，实现边界细化与坡度约束的协同优化。

**🔧 技术方法**

技术手段：双流U‑Net骨干、Haar小波分解与逆变换、FiLM与能量门控、结构监督（边界、内部深度）与斜率敏感的交叉熵损失、Dice损失、平衡的多任务损失。

**📊 数据集**

使用数据集：0.5 m GaoJing‑1 RGB图像、5 m TanDEM‑X DEM 与坡度，构成5通道输入；在四川红雅县的Area A（训练/验证/内部测试）与地理隔离的Area B（外部测试）进行评估。

**📈 对比分析**

与U‑Net、DeepLabV3+、SegFormer、Samba、原始Dual‑Encoder U‑Net等基线模型以及RGB+DEM+坡度的早期拼接模型对比，TRNet在Area A的Rice IoU为85.10%（比原Dual‑Encoder U‑Net高9.15pp），在Area B为80.68%（高18.83pp），同时在边界精度、坡度误检率方面也表现出显著提升。

**⚠️ 局限性**

局限性：仅在单个县域、单一季节、单一传感器下验证；缺乏跨传感器/跨季节的泛化评估；DEM分辨率与噪声敏感性未系统探测；仅提供二值稻田掩模，未进行实例/地块级评估。

---

## 323. Trie-Constrained Token Prediction with Hierarchy-Aware Semantic Alignment for HS Code Prediction

**arXiv ID:** 2608.04464 | [PDF](https://arxiv.org/pdf/2608.04464v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 324. Design and Flight of an Ion-propelled Micro Hovercraft Leveraging Ground Proximity Effects

**arXiv ID:** 2608.04343 | [PDF](https://arxiv.org/pdf/2608.04343v1)

**作者:** C. Luke Nelson `[一作]` (University of Utah), Daniel S. Drew `[通讯]` (University of Hawaii at Manoa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并飞行演示了利用地面效应的电离子推进微型悬浮器，实现持续悬停、无声飞行及被动扰动抑制。

**💡 创新点**

通过对裙边角度、裙比和中心板位置的实验优化，首次实现了电离子推进悬浮器的压力支持升力，并取得比现有同类设备大幅提升的推力效率和有效载荷。

**🔧 技术方法**

采用多级电离子加速堆叠推进器、被动裙边结构、光纤激光微加工电极以及SLA打印结构；利用自动化测试平台测量推力、效率和声学特性。

**📊 数据集**

通过实验测得的推力-高度、效率-高度、载荷-效率等数据，用于拟合地面效应模型；对比表格中其他MAV电离子推进器的数据。

**📈 对比分析**

采用对比表格与Cheeseman‑Bennett模型和自建经验模型进行推力/效率对比；所示悬浮器在1-2 mm高度下推力≈30 mN、效率16 mN/W、有效载荷≈1.5 g，推力效率提升约10×，载荷能力接近1.5 g，远超同尺寸电离子推进平台。

**⚠️ 局限性**

未系统探索电极间距、层间距等多参数优化；未测量流场和压差；依赖外部高压电源；对地面不平整度和障碍物的影响未评估。

---

## 325. Preverbal Uninflected and Underived Roots in Mapudungun. Wuno and Its Implications

**arXiv ID:** 2608.04869 | [PDF](https://arxiv.org/pdf/2608.04869v1)

**作者:** Andres Chandia `[一作]` `[通讯]`, Andres Chandia

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究Mapudüngun语言中预词位置的非屈折、未衍生根（如kalli、kim、küp、pepi、shinge、wüño）的语法地位，提出并检验了“韵律-书写假设”，认为表面上的词类混乱源于传教士对语音停顿的书写习惯，而非真正的语法分类。

**💡 创新点**

创新点在于把书写空格与口语中的韵律停顿关联起来，提出预词根是普通动词在词根前置复合结构中的V1位，而非专门的助动词、情态前缀或分词；并通过多世纪语料的定量检验展示了书写与语法的解耦。

**🔧 技术方法**

技术方法包括手工标注预词根出现位置（V1、V2、独立）与拼写方式（融合/分离），利用描述性统计与定性案例分析检验五个预测（V2出现、融合率变化、插入现象等）。

**📊 数据集**

数据集为约126万词的Diachronic Mapudüngun Corpus，涵盖1606年至今的书面资料，按早期（1606–1846）、转型期（1916–1930）和当代（1992–今）分段，覆盖词典、语法书、叙事文本与官方翻译。

**📈 对比分析**

比较方法是将语法位置统计与书写形式对比；结果显示V2实例确实存在（证明动词可作为V1/V2），而融合率随时期波动与语法无关；插入现象在当代显著出现，进一步支持书写假设。整体性能表现为理论与实证高度一致，验证了“根本上是词根的V1-组合”模型。

**⚠️ 局限性**

限制包括：仅使用书面资料，无法捕捉口语原始发音与语调；部分根（如shinge、kalli）数据稀疏，统计可靠性有限；以及对“插入”与“融合”区分依赖人工判断，可能存在主观误差。

---

## 326. Decentralization of Agenda-Setting Power and Domain-Selective Bridging: Algorithm Design Beyond the Echo Chamber Debate

**arXiv ID:** 2608.04774 | [PDF](https://arxiv.org/pdf/2608.04774v1)

**作者:** Masahiro Fujita `[一作]` `[通讯]` (Kansai University), Masahiro Fujita (Kansai University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了议程民主化指数（ADI）与社会信息健康（SIH）模型，并通过基于代理的仿真验证了域选择性桥接的有效性。

**💡 创新点**

① 引入ADI量化议程设置权力去中心化；② 将认知约束与信息域特征（可验证性与集体范围）结合，提出域选择性桥接权重最优公式；③ 证明桥接效果呈渐进饱和。

**🔧 技术方法**

理论建模（归一化的四维指数与桥接函数），基于Agent‑Based Simulation 的仿真与计量指标（Echo Chamber Depth、Cross‑Cluster Information Sharing、User Satisfaction、SIH）。

**📊 数据集**

无真实平台数据，使用合成兴趣空间与信息生成规则的模拟数据。

**📈 对比分析**

与无桥接和统一桥接三种算法对比，域选择性桥接在SIH与用户满意度两项指标上均优于其他两种；在效率指标上提升约3.5倍。

**⚠️ 局限性**

未进行真实用户数据验证；V与S的测量方法未定；模型假设V、S独立且仅关注桥接数量；未考虑信息质量与跨域影响；ADI正则化与历史比较缺乏实证操作流程。

---

## 327. LLM-based Vulnerability Discovery in Business Process Documentation

**arXiv ID:** 2608.04271 | [PDF](https://arxiv.org/pdf/2608.04271v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 328. UBLLIE: Unified Backlight and Low-Light Image Enhancement

**arXiv ID:** 2608.04429 | [PDF](https://arxiv.org/pdf/2608.04429v1)

**作者:** Yasmin Yasin `[一作]` (King Fahd University of Petroleum and Minerals), Saeed Anwar `[通讯]` (University of Canberra)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种统一的无监督方法，用于同时提升背光和低光图像的可视质量

**💡 创新点**

将CLIP引导的文本提示学习与对称残差U‑Net结合，并在瓶颈处加入ASPP模块，既提供语义监督又捕获多尺度照明信息

**🔧 技术方法**

CLIP文本与图像编码器、对比损失、残差U‑Net+ASPP、Adam优化器及余弦学习率调度

**📊 数据集**

使用BAID背光子集、DIV2K高质量图像、LOL低光数据和VE‑LOL‑L数据进行训练与评估

**📈 对比分析**

在BAID、Backlit300、LOL和VE‑LOL‑L等公开数据集上与多种监督、无监督和半监督方法比较，PSNR、SSIM、LPIPS、MUSIQ等指标均达或超过现有最佳结果，显示出更好的图像质量与泛化能力

**⚠️ 局限性**

依赖通用CLIP模型，可能对特定场景或视频序列的适用性有限，且仍需更广泛的背光基准数据和实时轻量化实现

---

## 329. Scarcity and Predictive Uncertainty: Implications for Societal Resource Allocation

**arXiv ID:** 2608.04251 | [PDF](https://arxiv.org/pdf/2608.04251v1)

**作者:** Shafkat Farabi `[一作]` (Virginia Tech), Sanmay Das `[通讯]` (Virginia Tech)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究预测不确定性在资源稀缺分配中的影响，提出并分析了基于最大边际收益的预测不确定性感知（UA‑MMB）与无感知（UU‑MMB）以及脆弱性优先（VF）三种策略，证明资源稀缺时低不确定性个体被优先分配，资源充足时则优先高不确定性个体，并用PISA教育测试数据验证理论；

**💡 创新点**

首次将预测不确定性异质性纳入社会资源分配模型，揭示了资源稀缺与充足条件下优先级的“翻转”现象，并量化了忽视不确定性导致的效率损失；

**🔧 技术方法**

采用概率模型、对数凹分布、凸顺序分析、KKT条件求解阈值规则、XGBoost回归预测学生成绩，并用PISA数据进行实证模拟；

**📊 数据集**

主要使用OECD PISA 2022数学成绩的Plausible Values数据集（约11,000名学生，分为高低社会经济地位两组），并对高低不确定性组分别训练XGBoost模型；

**📈 对比分析**

与UU‑MMB和VF策略比较，UA‑MMB在所有资源水平下实现零相对效率损失；UU‑MMB在资源稀缺时约20%效率损失，随着资源增多趋近零；VF始终存在较高效率损失，只有当资源足够多时才恢复；

**⚠️ 局限性**

假设单一干预类型，忽略异质处理效应，预测不确定性仅在群体层面估计，且未区分系统性误差与不可约方差，可能导致模型在复杂现实场景下的适用性受限；

---

## 330. FocusMem: Factorizing Content, Readout, and Trust in Latent GUI Memory

**arXiv ID:** 2608.04530 | [PDF](https://arxiv.org/pdf/2608.04530v1)

**作者:** Zhuoran Zhang `[一作]` (Peking University), Tengjiao Wang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种名为 FocusMem 的 GUI 代理记忆框架，能够在压缩多模态轨迹的同时分别保留可复用经验和当前任务进度。

**💡 创新点**

创新点在于将记忆功能拆分为三项独立责任：内容恢复、状态条件读取和证据信任，并通过角色感知内容基、状态条件读出以及独立信任门实现。

**🔧 技术方法**

采用连续向量压缩、多模态轨迹离散化、冻结策略网络训练、Qwen3‑VL‑8B 强化学习以及基于 Gemini‑3.1‑Pro 的评估技术。

**📊 数据集**

在五个公开的 GUI 代理基准（Webbench 等）上进行实验。

**📈 对比分析**

与固定动作仅记忆以及之前的 Latent 记忆方法对比，FocusMem 在任务成功率上持续优于对照组，尤其在加入无关轨迹干扰时提升明显。

**⚠️ 局限性**

局限性包括仅在冻结的 Qwen3‑VL‑8B 上评估，未验证对其他模型或移动/桌面环境的泛化；信任门诊断仅使用注入无关轨迹，未覆盖更细微的不匹配情况；且评估依赖 Gemini‑3.1‑Pro 的判定，可能存在判定偏差。

---

## 331. Diverse and Plausible Algorithmic Recourse via Tractable Recourse Distributions

**arXiv ID:** 2608.04677 | [PDF](https://arxiv.org/pdf/2608.04677v1)

**作者:** Anagha Sabu `[一作]` (Indian Institute of Technology Palakkad), Narayanan C. Krishnan `[通讯]` (Indian Institute of Technology Palakkad)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Tractable Recourse Distributions（TRD）框架，通过指数倾斜在正类概率电路上生成多样、可行且可信的反事实；

**💡 创新点**

创新点在于将回溯问题从点估计转化为个体化概率分布，并证明平滑可分解概率电路对可加成本的指数倾斜闭合，实现无训练集、无迭代的完整采样；

**🔧 技术方法**

使用指数倾斜、概率电路（Sum-Product Network）及k-medoids聚类等技术；

**📊 数据集**

实验使用成人（Adult）、德国信用（German Credit）、Give Me Some Credit（GMSC）等LiCE基准表格数据以及MNIST手写数字数据；

**📈 对比分析**

与DiCE、LiCE、MIO等多重反事实生成方法对比，TRD在多样性、可行性覆盖率和可解释性上保持最优或最平衡的表现，且覆盖率达100%；

**⚠️ 局限性**

限制在于接受率随有效概率质量变化，过于严格的有效性或因果约束会降低采样效率，需要进一步把约束编译进电路并自适应倾斜强度。

---

## 332. Splat-Based Metal Artifact Reduction in Cone-Beam CT via Compact Attenuation Modeling

**arXiv ID:** 2608.04764 | [PDF](https://arxiv.org/pdf/2608.04764v1)

**作者:** Kiseok Choi `[一作]` (KAIST), Min H. Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出一种基于高斯投射（Gaussian splatting）的金属伪影抑制框架，在CBCT重建中直接使用多能谱物理模型实现金属诱发的光束硬化消除。

**💡 创新点**

创新点在于：①将能量依赖的质量衰减系数用低维二次贝塞尔曲线近似，仅用一个标量控制材料特性；②将多能谱Beer–Lambert定律嵌入可微分高斯投射中；③实现了无金属掩膜、无监督的联合几何与材料优化。

**🔧 技术方法**

使用技术包括：可微分多能谱前向投影、连续高斯表示、光束硬化物理建模、L1/SSIM/TV 损失、CUDA 加速、SPEKTR 生成的 X‑ray 光谱。

**📊 数据集**

实验使用了三种合成 CBCT 阴影（Lung、Teeth、Broccoli）和多组真实 CBCT 数据（Bruker SKYSCAN 1273 扫描的 Garlic、Avocado、Chicken 等），并提供了相应的金属无扫描作为参考。

**📈 对比分析**

与 FDK、LIMAR、Polyner、Park 等基线相比，本文方法在 3D PSNR、SSIM 以及视觉质量上均显著优于所有对手；在合成数据中 PSNR 达 28–29 dB，SSIM 约 0.99；在真实数据中能彻底抑制金属伪影且保留细节；计算时间方面，比 Polyner 与 Park 快 10 倍以上。

**⚠️ 局限性**

局限性包括：①低维材料模型无法准确描述极其特殊或混合材料的 MAC 曲线；②对 X‑ray 光谱仿真精度敏感，误差可能导致残留伪影；③高斯投射需要合适的原子数量与分布，极大体积或高度各向异的结构可能需要手动调参；④目前仅针对静态 CBCT，动态或有限角度扫描尚未验证。

---

## 333. Tropical Algebraic Geometry for Neuronal Representations: An Arakelov-Green Measure Based Descriptor for Graph Learning

**arXiv ID:** 2608.04460 | [PDF](https://arxiv.org/pdf/2608.04460v1)

**作者:** Yuyang Zhang `[一作]` (McGill University), Qihuang Zhang `[通讯]` (McGill University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关、利用热带代数几何的结构先验，用连续松弛评估3D神经元形态的拓扑与几何特征；

**💡 创新点**

创新点在于将Arakelov‑Green度量在拓扑约化空间上闭式计算，避免了NP‑Hard的CVP求解与格点四舍五入误差，并通过谱分解得到节点坐标与图级特征；

**🔧 技术方法**

使用了热带阿贝尔‑朱比尼变换、格点近似的连续松弛、图拉普拉斯伪逆、Kron约化、周期空间增广和商空间构造等数学工具；

**📊 数据集**

在BREC基准上验证可区分超越1‑WL的图对；在ACT‑4、JML‑4、BIL‑6三组3D形态数据上提升分类准确率，尤其在Tree‑LSTM、GNN和VAE框架中表现优异；

**📈 对比分析**

与基线（1‑WL、子图GNN、高阶Transformer等）以及传统的格点近似方法对比，连续评估在计算时间上快4–6倍、宏F1更高，且在大图（g>30）下保持可行；

**⚠️ 局限性**

局限在于需要O(|V|³)预处理开销，且移除了绝对坐标信息，可能忽略某些空间分布特征，未来可探索在覆盖空间直接定义消息传递的网络结构。

---

## 334. Securing Contrastive mmWave-based Human Activity Recognition against Adversarial Label Flipping

**arXiv ID:** 2608.04029 | [PDF](https://arxiv.org/pdf/2608.04029v1)

**作者:** Amit Singha `[一作]` (Purdue University), Yanchao Zhang `[通讯]` (Arizona State University)

**通讯引用:** 6654 | [OpenAlex ID](https://openalex.org/A5100615703)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文研究了基于毫米波（mmWave）的无线人体动作识别（HAR）系统在使用监督对比学习（SCL）时，受到标签翻转攻击的脆弱性，并提出了相应的防御机制。

**💡 创新点**

创新点在于首次系统性识别并分析mmWave基HAR系统在SCL框架下的三种轨迹相似度驱动的标签翻转攻击，并提出了无需可信训练集即可通过可信样本选择实现的自适应防御方法。

**🔧 技术方法**

主要技术包括毫米波雷达信号处理、CNN‑LSTM骨干网络、监督对比学习与无监督对比学习、Mixup数据增强、可信样本与可信对的自动选取以及自适应损失函数组合。

**📊 数据集**

实验使用TI 1843 mmWave雷达收集的10,650条手部动作样本，覆盖Push、Pull、Slide Left/Right、Clockwise/Anticlockwise等六类动作，样本来自25名志愿者，在5个采集位置与6种环境下获取。

**📈 对比分析**

在正常角度下，SCL比传统SL提升约1.5%准确率；在极端角度下，SCL相对SL提升约8%；在随机、跨轨迹、内轨迹三种标签翻转攻击下，SCL准确率可降至70%，而所提Sel‑CL在40%攻击时仍保持90%以上准确率，且在无攻击时准确率高达98.78%。

**⚠️ 局限性**

局限性包括样本仅覆盖手部动作且数量有限；防御依赖轨迹相似度假设，若攻击者具备更精确轨迹匹配能力可能突破；在多用户、多环境的更复杂场景中验证尚不足。

---

## 335. Deep Learning for Real-Time Sound Order Recognition in Human-Robot Interaction

**arXiv ID:** 2608.04072 | [PDF](https://arxiv.org/pdf/2608.04072v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 336. On Hamming-Lipschitz Type Stability of the Subdominant (Minmax) Ultrametric: Theory and Simple Proofs

**arXiv ID:** 2608.04014 | [PDF](https://arxiv.org/pdf/2608.04014v1)

**作者:** Alokendu Mazumder `[一作]` (IISc Bengaluru), Punit Rathore `[通讯]` (IISc Bengaluru)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套针对子主导（最小最大）超度量的稀疏编辑下的 Hamming‑Lipschitz 稳定性理论，揭示了编辑仅通过 MST 的“暴露切点”传播，给出了单条编辑的最优受影响对数 S_union(f) 以及树全局常数 L̅_T，并证明了该实例依赖性不可避免；

**💡 创新点**

核心创新在于将传统的 ℓ∞/Gromov–Hausdorff 稳定性扩展到稀疏编辑的 ℓ0 视角，给出可实现且可达成的暴露‑切点分数与全局上界，并证明单条离线编辑可导致 Θ(n²) 的超度量变动，表明对树几何的依赖是本质的；

**🔧 技术方法**

主要技术包括基于 MST 的路径瓶颈分析、暴露切点与对集的联合计数、条件近加性原理以及对单编辑与多编辑情况的构造证明；

**📊 数据集**

实验数据集涵盖图像嵌入（CIFAR‑10、ImageNet‑10、STL‑10）、超像素分割（Cameraman）、以及半监督聚类（MNIST、USPS、HAR、Olivetti Faces、OptDigits）；

**📈 对比分析**

与原始权重、局部尺度权重、质心间距、Ward‑桥、Fisher‑桥及随机查询等基线相比，结构分数 S_union(e) 在边验证预算内能最快提升聚类质量，且在大多数任务中与最佳特征基线相当或更优；

**⚠️ 局限性**

局限性包括：理论仅适用于单链接聚类，需构造 MST；在最坏情况下单条编辑可能导致二次级变动；实验仅为诊断性验证，未涵盖更复杂的聚类管线。

---

## 337. Web Cache Overflow: Exploiting Imprecise Keys for Cache Degradation and Beyond

**arXiv ID:** 2608.04744 | [PDF](https://arxiv.org/pdf/2608.04744v1)

**作者:** Matteo Golinelli `[一作]` (University of Trento), Bruno Crispo `[通讯]` (University of Trento)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了一种利用网页缓存键不精准导致的缓存污染攻击（Web Cache Overflow），通过构造不同缓存键的相同请求填满缓存，降低缓存命中率并增加源服务器负载。

**💡 创新点**

创新点在于：①攻击无需预知缓存替换算法或对象热度；②基于普通的缓存失效（cache busting）技术实现重复制；③首次系统性展示精准缓存键设计是防御此类攻击的最佳实践；④提供自动化检测工具。

**🔧 技术方法**

使用了：自动化爬虫+缓存键逆向推理、HEAD 请求维持 TTL、Web Polygraph 负载模拟、SHA‑512/XXH3 哈希做去重、各种缓存算法（LRU、GDSF、LFUDA）实验、rate‑limit 模拟。

**📊 数据集**

主要数据集为 Tranco top 10k 域名列表（约10k 域名）、GitHub 上公开的缓存配置项目（127 项）以及自建实验环境中的静态文件。

**📈 对比分析**

通过对五种主流缓存代理（ATS、HAProxy、Squid、Varnish、Traefik）进行统一实验，比较攻击前后缓存命中率、流量放大比（Traffic Amplification Ratio）以及不同缓存容量、对象大小、淘汰算法的影响。实验显示，即使缓存容量为 100 GB、对象 1 MB，攻击仍能将命中率显著降低；而大对象或更大缓存时攻击成本上升但仍可行。

**⚠️ 局限性**

局限性包括：①对 CDN 等大规模分布式缓存几乎无效；②当缓存容量极大或对象极小、网络带宽受限时，攻击成本高、效果有限；③防御需要对网站缓存键做精准重构，实施成本与人工审核相关；④去重等补救措施在高并发环境下开销较大。

---

## 338. MatrAIx: Simulating the World with 8.3 Billion Persona Agents

**arXiv ID:** 2608.04205 | [PDF](https://arxiv.org/pdf/2608.04205v1)

**作者:** Xiaomin Li `[一作]` (Harvard University), Dawn Song `[通讯]` (University of California Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并验证了MatrAIx模拟用户评估基础设施，涵盖8.3B persona数据、四种交互环境与1,010可复现任务，完成1.8万次实验并评估人类多样性。

**💡 创新点**

首次提供大规模、可定制的persona集合（Persona 8B）与统一评估管道，将人工与合成资料结合，支持跨产品、跨模型的分组分析。

**🔧 技术方法**

使用基于DAG的依赖采样生成合成persona、LLM驱动的persona代理、自动验证器与远程并行执行，结合GPT/Claude模型进行交互。

**📊 数据集**

构建8.3B persona记录，公开1M核心样本（599k人类来源、400k合成），来源包括Wikipedia、Amazon Reviews、Stack Overflow、GSS等。

**📈 对比分析**

通过对比三大LLM（Claude Opus4.8、GPT5.5、Claude Haiku4.5）在四种环境下的任务结果，发现persona属性对行为影响显著且可重复；行为遵从率91.5%，提取质量与人类评估相近。

**⚠️ 局限性**

受限于合成与来源样本的偏差、对真实人类实验的缺乏验证、以及仅覆盖部分交互情境；需在真实用户研究中进一步验证。

---

## 339. NuclearDiffusion: Text-to-Image Foundation Models for Learning Nuclear Energy Concepts

**arXiv ID:** 2608.04030 | [PDF](https://arxiv.org/pdf/2608.04030v1)

**作者:** Mohammed I. Radaideh `[一作]` (University of Michigan), Majdi I. Radaideh `[通讯]` (University of Michigan)

**通讯引用:** 1764 | [OpenAlex ID](https://openalex.org/A5049880944)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究对Stable Diffusion XL、SD-v3.5-Medium和Flux.1进行领域适配，使用1000张带说明的核能图像生成核能相关图像。

**💡 创新点**

创新之处在于构建首批核能图像数据集并系统评估不同生成模型在专业核能视觉任务上的表现，展示了领域适配能显著提升技术准确性。

**🔧 技术方法**

采用扩散模型、流匹配Flux.1以及LoRA微调技术，配合GPT-Image-2.0图像增强，并使用KID/CMMD等指标与专家评估。

**📊 数据集**

使用从教材、期刊及新闻网站筛选并处理的1000张带英文说明的核能图像数据集。

**📈 对比分析**

通过KID/CMMD与人工专家打分与三大商用模型进行对比，结果显示SDXL微调后在视觉与技术一致性上远优于零射击模型，Flux.1表现不佳，而GPT-Image-2和Gemini在通用核能概念上可行但细节不足。

**⚠️ 局限性**

主要局限包括评估依赖人工成本、核能图像样本不足、CLIP等指标无法充分衡量技术正确性，以及现有GPU资源限制无法进一步扩展模型规模。

---

## 340. Spatiotemporal Graph Transformer for Traffic Intelligence in Edge Computing

**arXiv ID:** 2608.04075 | [PDF](https://arxiv.org/pdf/2608.04075v1)

**作者:** Laha Ale `[一作]` (Southwest Jiaotong University), Peng Yu `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 7439 | [OpenAlex ID](https://openalex.org/A5100783073)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于图神经网络与Transformer注意力机制的时空交通预测框架，用于边缘计算中资源的前瞻性调度。

**💡 创新点**

创新点在于将空间感知（图卷积）与时间推理（Transformer自注意力）分离处理，从而在保持空间关联的同时捕获长期时间依赖，提高长时延预测精度。

**🔧 技术方法**

使用的技术包括：图卷积网络（GCN）进行空间特征聚合；自注意力Transformer编码器进行时间记忆；双层注意力聚合和全连接回归层生成多时隙预测。

**📊 数据集**

采用中国电信上海区六个月的真实蜂窝网络数据，先将基站聚类为25个边缘服务区域，再将用户连接记录转化为每小时的流量需求时空序列。

**📈 对比分析**

与三种递归图模型（GCN‑RNN、GCN‑LSTM、GCN‑GRU）做统一实验，使用相同的历史窗口、预测时隙、隐藏维度等设置。实验结果显示GCN‑Transformer在1–24小时预测均表现出最低MAE、最高R²，尤其在中长时隙（>12h）显著优于递归基准。

**⚠️ 局限性**

限制包括：模型仅利用空间坐标和基本流量统计，未加入移动轨迹、服务类型等更丰富的上下文；图结构固定且完全连接，可能导致计算开销随节点数上升；对极端突发事件的预测仍显平滑，需进一步改进。

---

## 341. Hidden Ciphers and Where to Find Them: Static Discovery and Assessment of Cryptographic Assets in Software

**arXiv ID:** 2608.04857 | [PDF](https://arxiv.org/pdf/2608.04857v1)

**作者:** Christian Näther `[一作]` (XITASO GmbH), Eduard Hirsch `[通讯]` (University of Applied Sciences Amberg-Weiden)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于分类的静态加密资产发现与评估框架，结合规则仓库和扫描器实现了加密资产的自动化清单与风险标注，并输出CBOM；

**💡 创新点**

提出了三维分类（Crypto‑Material、Crypto‑Artifacts、Crypto‑Invocations）与评估维度（Weakness、Vulnerability）以及与扫描器解耦的可扩展规则仓库，支持不同语言与配置文件的统一发现；

**🔧 技术方法**

利用静态分析、AST/正则/解析器、规则驱动引擎、CycloneDX CBOM导出；

**📊 数据集**

使用自研合成基准（覆盖Go、Ruby、nginx配置，已知真值）和10个生产服务的真实代码/配置/依赖；

**📈 对比分析**

与CBOMkit对比，针对Go调用子集F1从0.66提升至0.92；在合成基准下资产发现F1为0.75、评估Recall 0.91，实测10服务370条资产扫描耗时<6分钟；

**⚠️ 局限性**

依赖规则仓库完整性，缺乏数据流/动态信息导致部分弱点非可操作；仅覆盖Ruby/Go及常见配置，未支持Java/C/C++/Python等；实时性能受单线程限制；缺乏完整基准导致真实召回难以量化。

---

## 342. A Dual Evaluation for Music Transcription

**arXiv ID:** 2608.04511 | [PDF](https://arxiv.org/pdf/2608.04511v1)

**作者:** Ping Wang `[一作]` (University of Washington), Noah A. Smith `[通讯]` (University of Washington)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建24条音频到乐谱的转录流水线并加入Rubato端到端系统，评估其生成的乐谱符号相似度和音频再生相似度。

**💡 创新点**

创新点在于提出双维度评估框架，将符号写作与音频再现分别衡量，并通过大规模人类ABX实验验证多种自动评估指标，揭示不同指标偏好不同转换器，说明系统设计存在符号-播放张力。

**🔧 技术方法**

技术实现包括使用OMR‑NED衡量符号相似度，DTW/TWED与CLEWS、CLaMP‑3、AudioLM等嵌入式评估衡量播放相似度，MuseScore+FluidSynth渲染音频，以及对数千对比样本的Spearman/Kendall相关与人类排名分析。

**📊 数据集**

实验数据集为230段未见的钢琴录音（来自ATEPP），涵盖23部作品、30位演奏者、6位作曲家，首三分钟用于评估。

**📈 对比分析**

通过Spearman/Kendall相关与人类排名对比自动指标，发现CLEWS在准确性与成本上最优；Rubato在符号相似度最高且播放相似度亦位居中上，系统之间因转换器不同而产生显著排名差异。

**⚠️ 局限性**

局限性包括仅针对西方古典钢琴短时段，渲染参数可能影响结果；未验证符号可读性；Rubato缺乏人类播放评估；结果可能不适用于其他乐器或音乐风格。

---

## 343. Revisiting Pose Sensitivity in Splat-based Computed Tomography under Sparse-view Reconstruction

**arXiv ID:** 2608.04752 | [PDF](https://arxiv.org/pdf/2608.04752v1)

**作者:** Kiseok Choi `[一作]` (Korea Advanced Institute Of Science And Technology), Min H. Kim `[通讯]` (Korea Advanced Institute Of Science And Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种自校正的高斯高斯基 CT 重建框架，联合优化体积与相机姿态；

**💡 创新点**

首次系统分析姿态敏感性并设计稳定的梯度驱动自校正方法，去除了 TV 正则化的必要性；

**🔧 技术方法**

利用可微分高斯 splatting、四元数姿态微分、L1+SSIM 损失和基于 Jacobian 的梯度传播；

**📊 数据集**

在模拟的带姿态噪声的 CT 数据集（TIGRE）和真实 CBCT 数据集（含核桃等物体）上验证；

**📈 对比分析**

与 FDK、SAX-NeRF、NeAT、Thies 等基线对比，PSNR 提升约10dB，姿态误差显著降低，计算时间更低；

**⚠️ 局限性**

在极稀视角（25 视图）下仍出现针状伪影，需进一步正则化以提升极端稀视情况的质量。

---

## 344. Coupled Continuous-Discrete Generation for Scene Text Image Super-Resolution

**arXiv ID:** 2608.04525 | [PDF](https://arxiv.org/pdf/2608.04525v1)

**作者:** Axi Niu `[一作]` (Northwestern Polytechnical University), Yanning Zhang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 DualTSR，一种统一的场景文本图像超分辨率框架，内部同时生成高分辨率图像和文本序列，完成端到端推理，无需外部 OCR 先验。

**💡 创新点**

创新点在于将图像恢复与文本预测放在同一多模态 Transformer 里，并通过同步的条件流匹配与吸收状态离散扩散实现连续-离散联合生成，从而消除外部 OCR 或结构先验带来的误差，并显著提升语义一致性与推理效率。

**🔧 技术方法**

使用技术包括：条件流匹配（Conditional Flow Matching）用于图像生成，吸收状态离散扩散用于文本预测，多模态 Transformer 共享表示，模型导向训练（Classifier-Free Guidance）以及同步噪声注入的联合损失。

**📊 数据集**

在 CTR-TSR（中文合成场景文本）和 RealCE（真实中文文本）两个基准数据集上进行评估。

**📈 对比分析**

与通用 SR 方法（ESRGAN、SwinIR、SRFormer）及文本专用方法（MARCONet、MARCONet++、DiffTSR）对比，DualTSR 在 ×2、×4 级别上取得最佳或接近最佳的 FID、LPIPS、ACC、NED 等指标；参数量从 1.23B 降至 203M，推理时间从 13.3 s 缩短至 132 ms，显著提高了效率。

**⚠️ 局限性**

局限性包括：在极端退化或缺少字体/颜色信息时仍可能出现错误；目前仅验证中文场景，对其他文字或语言的泛化未知；模型对同步噪声设定和扩散步数的敏感性可能需要进一步调整。

---

## 345. EviGraph: Evidence-Guided Autonomous Research Agents

**arXiv ID:** 2608.04738 | [PDF](https://arxiv.org/pdf/2608.04738v1)

**作者:** Zhenjiang Ren `[一作]` (University of Chinese Academy of Sciences), Jiajun Zhang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

暂无可用信息

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

## 346. MERaLiON-GR: Speech Gender Recognition Model for English and SEA Languages

**arXiv ID:** 2608.04433 | [PDF](https://arxiv.org/pdf/2608.04433v1)

**作者:** Qiongqiong Wang `[一作]` (Institute of Advanced Intelligence and Computing), Longyin Zhang `[通讯]` (Institute of Advanced Intelligence and Computing)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 MERaLiON-GR 语音性别识别系统，支持英语与东南亚多语言的二分类。

**💡 创新点**

创新点：① 结合大规模 Conformer 预训练编码器 MERaLiON‑SpeechEncoder‑2；② 用 LoRA 参数高效微调并插入 rsLoRA 归一化；③ 设计多尺度 ECAPA‑TDNN 下游网络；④ 将预测性别作为元数据提升 Audio‑LLM 性能。

**🔧 技术方法**

技术：Conformer、LoRA、rsLoRA、ECAPA‑TDNN、注意力池化、RMSNorm+GELU 分类头、类别平衡加权交叉熵、cosine 学习率调度、BatchNorm→GroupNorm、Label smoothing。

**📊 数据集**

数据集：训练集为 VoxCeleb1 + IMDA PART1–5（新加坡英语）；评估集包含 8 种语言的公开多语言基准（FLEURS、Common Voice、IEMOCAP、SMALDUSC、OpenSLR、Thai SER、Thai Elderly、Vietnamese、Indonesian、Khmer）以及自制 SG‑ECMT 四语种 10–30s 录音。

**📈 对比分析**

比较方法：与 Vox‑Profile（最先进的独立性别识别模型）和 MERaLiON‑v2（通用 Audio‑LLM）按分类准确率对比；MERaLiON‑GR 在 15 个公开测试集上比 Vox‑Profile 高 12/15 组，部分数据集 100%；在 10–30s 片段评估中提升至 4.32pp；将性别元数据注入 Audio‑LLM 后，准确率提升 30–60pp。

**⚠️ 局限性**

局限性：在域不匹配的数据集（如 Malay SMALDUSC、Thai SER）表现下降；模型对低资源语言的泛化仍有限；仍需在更广泛的录音环境与口音上进一步验证。

---

## 347. NodeJEPA: Structure-Conditioned Latent Prediction for Node-Level Graph Self-Supervised Learning

**arXiv ID:** 2608.04381 | [PDF](https://arxiv.org/pdf/2608.04381v1)

**作者:** Tinghe Zhang `[一作]` (Northeastern University), Qiang Wang `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出两种节点级联成自监督学习框架NodeJEPA和PatchJEPA，利用latent-space预测代替输入重建或对比学习，提升节点表示质量。

**💡 创新点**

创新点在于将Joint-Embedding Predictive Architecture (JEPA)引入图节点任务，设计结构感知的k-hop ego-subgraph掩蔽、结构条件预测器以及VICReg+ISG正则化，且通过PatchJEPA实现大图的高效缓存式掩蔽。

**🔧 技术方法**

采用EMA目标编码器、stop‑gradient机制、结构条件跨注意力或消息传递预测器、余弦预测损失、VICReg风格方差-协方差正则和ISG正则，结合k-hop子图掩蔽或METIS分区。

**📊 数据集**

在五个节点分类基准上评估：Amazon‑Computers、Amazon‑Photo、Coauthor‑CS、Coauthor‑Physics和OGB‑arXiv。

**📈 对比分析**

与四个主流自监督基线（DGI、GraphMAE、BGRL、CCA‑SSG）及监督GCN在相同编码器、数据分割和线性探针协议下对比；NodeJEPA和PatchJEPA在平均自监督排名上并列第一（2.4），在四/五个数据集上均位列首或次，few‑shot探针下保持明显优势。

**⚠️ 局限性**

局限在于仅验证同质性较强的引用、协同购买与共著图；对低同质性图、不同编码器、动态掩蔽优化等情况尚未覆盖。

---

## 348. Interoceptive Attention as Dynamic Homeostatic Prioritization in a Foraging Agent

**arXiv ID:** 2608.04232 | [PDF](https://arxiv.org/pdf/2608.04232v1)

**作者:** St John Grimbly `[一作]` (University of Cape Town), Jonathan P. Shock `[通讯]` (University of Cape Town)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研究了在有限感知预算下，活跃推理代理如何通过动态分配交感精度来优先满足身体需求；

**💡 创新点**

提出了基于身体状态信念的自适应交感精度分配机制，并证明其能显著提升生存率；

**🔧 技术方法**

采用活跃推理框架、Dirichlet更新、预期自由能最小化和预算约束的精度调度；

**📊 数据集**

使用自制的AffectWorld网格世界（6×6）包含多种资源布局；

**📈 对比分析**

与均匀精度分配的基线比较，学习期生存率提高约2.1倍（0.199→0.414），方向相反的对照甚至更差；

**⚠️ 局限性**

仅在固定预算下测试，未评估不同世界模型、规划深度或连续状态的通用性，且依赖于身体状态后验良好校准的前提。

---

## 349. Masked diffusion enables coherent beat tracking

**arXiv ID:** 2608.04624 | [PDF](https://arxiv.org/pdf/2608.04624v1)

**作者:** Francesco Foscarin `[一作]`, Richard Vogl `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了一种基于掩码扩散模型的节拍/下拍跟踪方法，利用多输出建模消除神经网络产生的无效连续下拍和节奏跳变。

**💡 创新点**

创新点在于将MDM迁移至节拍跟踪任务，并设计了独立的节拍/下拍掩码、平衡掩码调度和跨步骤峰值拾取三大改进，显著提升了输出连贯性与准确率。

**🔧 技术方法**

使用技术包括Mask Diffusion Model、RoFormer Transformer、Beat This架构、SwiGLU激活、Shift‑tolerant weighted BCE损失、模型集成及多步迭代推理。

**📊 数据集**

训练数据为Beat This公开的4556轨音频（含部分无下拍标注），测试数据为GTZAN音频集993曲目（含下拍标注）。

**📈 对比分析**

通过与Beat This、Gagneré ST‑BCE、MusicFM+HingeNet等SOTA系统（包括DBN与非DBN配置）对比实验，MDM在CMLt、AMLt等连贯性指标上提升约9%，整体F1和下拍准确率也得到显著提升。

**⚠️ 局限性**

局限性包括训练时间翻倍、推理时间增至多步迭代成本、集成模型对硬件资源要求更高，且尚未实现对多层节拍层级选择与严格约束（如DBN）兼容的控制。

---

## 350. CARVE: Cross-Slice Anisotropic Reallocation of Visual Evidence for Efficient 3D Medical Volume Understanding

**arXiv ID:** 2608.04515 | [PDF](https://arxiv.org/pdf/2608.04515v1)

**作者:** Zhenyu Yi `[一作]` (Shanghai Jiao Tong University), Lichi Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对slice‑based 3D医学多模态大语言模型的视觉令牌进行压缩，采用预训练的2D视觉编码器后，按深度与平面不均衡地重新分配令牌，最终将视觉输入压缩至原始约20%的令牌量后仍保持高性能。

**💡 创新点**

提出CARVE框架，核心创新在于：①基于交叉切片特征变化自适应划分深度窗口；②在每个窗口内分配不同量的空间锚点与跨切片检索令牌；③利用残差得分与3D NMS实现跨切片证据检索；④在窗口内将剩余令牌折叠到锚点，实现训练‑free、预‑LLM的压缩。整个过程不需改动编码器、投影器或LLM，保持模型冻结。

**🔧 技术方法**

技术细节包括：深度窗口分割（g_t阈值判定）、交叉切片残差评分、min–max归一化与softmax分配窗口预算、基于注意力熵与块异质性的quadtree锚点选择、4D NMS实现跨切片检索、局部投影权重融合到锚点，再与检索令牌拼接送入LLM。

**📊 数据集**

主要在三大医学VQA/报告生成基准上评测：AMOS‑MM（闭合端VQA、报告生成）、3D‑RAD、M3D‑VQA；同时在通用领域的Qwen3‑VL进行跨模型迁移测试。

**📈 对比分析**

与VisionZip、DivPrune、MMTok、FastVID、MedPruner等训练‑free压缩方法以及原始Full模型对比。CARVE在AMOS‑MM报告生成中实现87%质量保留，仅19.3%令牌保留，优于所有基线；在3D‑RAD和M3D‑VQA的VQA任务中保持约97–99%总性能，仅≈20%令牌；整体排名在不同backbone/track组合中均为第一，显著降低推理延迟和内存占用。

**⚠️ 局限性**

局限性包括：①仅针对slice‑based模型，未直接适用于原生3D编码器；②压缩策略为硬性阈值和启发式分配，缺乏可学习的动态预算调度；③对极小切片内结构变化可能仍不足捕捉；④在超大尺度或多模态（PET‑CT）任务中的效果尚未验证。

---

## 351. SPOT: Sparse Probing and Outcome Calibration for On-Policy Distillation

**arXiv ID:** 2608.04419 | [PDF](https://arxiv.org/pdf/2608.04419v1)

**作者:** Zikun Qu `[一作]` (Chinese University of Hong Kong), Zhongxiang Dai `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Sparse Probing and Outcome‑calibrated Targets (SPOT)，在对学生模型进行OPD时通过先用教师熵、top‑k概率质量和学生‑教师不匹配来稀疏定位需要探测的位置，再利用验证器评估候选续写的收益并以KL正则化方式校准教师分布，形成局部目标。

**💡 创新点**

将“在哪里探测”和“该如何校准”两项决策分离：使用教师熵、top‑k质量与学生误差的乘积作为探测优先级，并通过验证器奖励倾斜的KL目标将局部教师分布与后验收益融合。

**🔧 技术方法**

逆K‑L OPD、top‑k前向KL近似、Jensen–Shannon距离评估学生‑教师差距、验证器评估续写奖励、闭式解的KL约束目标、PPO式优化。

**📊 数据集**

在Qwen3教师下对Qwen3‑0.6B、1.7B、4B学生分别使用MATH、DAPO‑Math‑14k训练，并在MATH‑500、AIME 2024/25、AMC 2023、Minerva Math、HMMT 2025六大数理推理基准上评估。

**📈 对比分析**

与KD、OPD、GRPO、EOPD等基线对比，SPOT在所有学生规模下宏观 Pass@8 最高，Avg@8 最高或次高；在多样本覆盖率上显著提升（Pass@8+4‑5个百分点），而平均准确度几乎保持或略优。

**⚠️ 局限性**

需要额外的验证器和局部探索开销，探测预算和候选数量需手动调参，方法主要针对推理类任务，未充分验证对非数理任务或更大规模模型的适用性。

---

## 352. Toward Integrating Adaptive Experience Replay and Online Uncertainty Estimation in Safe Actor-Critic Optimal Control

**arXiv ID:** 2608.04732 | [PDF](https://arxiv.org/pdf/2608.04732v1)

**作者:** Mahshad Rastegarmoghaddam `[一作]` (Politecnico di Milano), Shima Samadzadeh `[通讯]` (Politecnico di Milano)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在两维机器人导航任务中，提出了一种将安全过滤（基于控制障碍函数CBF）、自适应经验回放和在线不确定性估计相互耦合的完整架构，并通过六种配置进行系统比较。

**💡 创新点**

创新点在于：①将安全过滤、回放管理与不确定性估计视为共设计的核心模块；②利用扩展的回放元组（包括安全度、估计残差、创新度和新颖度）实现多维优先级；③在线地把障碍物几何估计注入CBF，使安全约束随不确定性实时更新；④通过执行动作而非原始动作更新评论家，提高价值估计与闭环行为的一致性；⑤给出有限训练下的回放曝光上界和鲁棒CBF条件。

**🔧 技术方法**

使用技术包括：Actor‑Critic 强化学习框架；控制障碍函数（CBF）安全过滤（QP实现）；混合优先级经验回放（TD误差、安全度、不确定度、新颖度权重）；在线的障碍物中心估计（静态中心滤波器，可推广为LSTM或Kalman）；连续时间机器人动力学离散化；统计检验（Friedman、Wilcoxon）。

**📊 数据集**

数据集：合成的 3.20m×3.20m 工作空间，包含已知矩形障碍和两颗不确定圆形障碍；传感器观测包含偏置、漂移、抖动、跳变等噪声；每个配置训练 10 轮，每轮 115 步，评估 5 种种子；对感知噪声进行 11 级扩展（乘数 2.2、6.0 等）。

**📈 对比分析**

通过在相同训练预算、随机种子、传感器流、探索方式和扰动下，对六种组件匹配配置进行比较。评估指标包括总成本、违规次数、最小全局距离、障碍物信念RMSE、目标成功率、滤波器干预率和运行时。极端测试（乘数 6.0）中，完整配置（Full）实现无碰撞、5/5 目标成功、最低成本（7.63±0.44）和信念误差（3.52±0.55 cm）。在中等测试中，所有含 CBF 的配置均无碰撞，Full 仍表现出最低 RMSE 与最高安全裕度。统计检验表明配置间显著差异，且 Full 与 UE 在极端测试中呈现明显优势。

**⚠️ 局限性**

局限性包括：仅在二维仿真场景下验证，缺乏真实硬件或移动障碍验证；不确定性估计采用简单静态中心滤波，未评估更复杂的递归或观测器；回放优先级权重手工调参；有限缓冲区尺寸与曝光上界仅为经验性质；理论保证仅覆盖回放曝光与鲁棒 CBF 条件，未给出全局收敛或安全证明；计算开销因 QP、回放扫描和估计更新而增加；评估仅针对合成感知噪声，未涵盖真实分布漂移和在线离线混合策略的挑战。

---

## 353. Adversarially Robust Abductive Fusion of Pre-trained Transformer-based Perception Models

**arXiv ID:** 2608.04190 | [PDF](https://arxiv.org/pdf/2608.04190v1)

**作者:** Mario Leiva `[一作]` (Universidad Nacional del Sur), Paulo Shakarian `[通讯]` (Syracuse University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e0540dec-d77f-42db-94ae-d039248f6393` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种在无域知识前提下，通过每个模型的Label Vector Pool学习错误检测规则，并将多模型推理视为一致性归纳问题，得到鲁棒的组合器。

**💡 创新点**

创新点在于（1）用模型自身的嵌入聚类构造LVP，实现无域知识的元认知层；（2）将模型融合框架化为一致性归纳，既保持与投票相当的性能，又能抵抗协同标签翻转攻击。

**🔧 技术方法**

技术主要包括：ViTDet目标检测器、基于k-means的LVP原型池、随机森林错误概率预测、逻辑规则学习（EDR）、整数规划与启发式搜索实现一致性归纳，以及可选的 tie‑breaker。

**📊 数据集**

实验使用MDS‑A（15个混合天气测试集）和VisDrone‑DroneVehicle（RGB/IR双模态）两大数据集，模型为六个单天气专属ViTDet检测器。

**📈 对比分析**

与单模型、投票（多种变体）以及基于域知识的规则进行对比，本文方法在清洗数据上与强投票基线F1相当，且在90%标签翻转攻击下平均F1提升至0.42（vs. 0.35），在所有测试集上均保持相对领先。

**⚠️ 局限性**

局限性包括：① 需要为每个模型单独训练LVP和规则，导致规模化成本；② 处理完全无目标域标注时仍需假设可用目标检测框；③ 计算成本相对投票略高，尤其是整数规划解法。

---

## 354. A Systolic Array Architecture for Nonlinear Activation Functions and Softmax Computation using Chebyshev Polynomials

**arXiv ID:** 2608.04734 | [PDF](https://arxiv.org/pdf/2608.04734v1)

**作者:** Benedikt Schaible `[一作]` (Technical University Of Munich), Jiang Hu `[通讯]` (Texas A&M University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种基于流水阵（systolic array）的激活函数单元，统一实现单变量激活函数（tanh、sigmoid、ReLU）和多变量 softmax，采用 Chebyshev 多项式近似并通过 Horner 递推实现。

**💡 创新点**

创新点在于：① 将 softmax 也纳入同一流水阵，实现硬件复用；② 用 Chebyshev 多项式而非传统 CORDIC 或 PLA/ LUT 方式，显著降低误差与功耗；③ 采用统一 Q3.12 固定点量化，兼顾多函数精度；④ 在同一流水阵中实现软max 的四步计算，支持大规模向量。

**🔧 技术方法**

使用 Chebyshev 多项式近似、Horner 递推、固定点 Q3.12 量化、流水阵结构、硬件除法（softmax 的归一化步骤）以及 ReLU 的比较器实现。

**📊 数据集**

实验使用合成的高斯（N(0,σ²)）和均匀（U(−a,a)）分布向量，vector 长度分别为 8 与 256，比较了激活函数和 softmax 的误差；未给出具体深度学习模型或数据集，但评估指标涵盖 AE、KL 散度、RSE 等。

**📈 对比分析**

与 CORDIC（Raghuram 等）以及 ONE‑SA（分段线性近似）对比。结果显示：tanh/sigmoid 的平均绝对误差低 71%/41%；softmax 的 KL 散度比 CORDIC 与 ONE‑SA 分别低 44.6%/79.0%；面积和功耗分别比 CORDIC 低 4.6%/5.1%；latency 约 15%–16% 的降低，尤其是 256 长度 softmax 的 14.3% 延迟提升。

**⚠️ 局限性**

局限性包括：① softmax 需要额外寄存器支持，向量长度受限；② 低阶多项式在极端输入下误差仍可能增大；③ 设计基于固定点量化，极高精度需求下可能需要更宽位宽；④ 论文未在完整模型推理链路或实际深度网络上验证性能与准确率。

---

## 355. Improving Auto-Design of Neural PDE Solvers with a Domain-Specific Language

**arXiv ID:** 2608.04384 | [PDF](https://arxiv.org/pdf/2608.04384v1)

**作者:** Shengxin Kong `[一作]` (North China University of Technology), Jingwen Fu `[通讯]` (Beijing Zhongguancun Academy)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种用于神经PDE求解器自动化设计的领域特定语言ADSL-PDE，并在此基础上构建了结构化进化框架；

**💡 创新点**

创新点在于通过把求解器设计抽象为可验证的高层方案（架构、约束、采样、优化等）并提供确定性编译器，从而将搜索空间从无限制的Python代码压缩到高密度、可执行的决策空间；

**🔧 技术方法**

核心技术包括：①面向代理的DSL定义与解析；②静态验证（IR检查）；③确定性后端编译；④基于LLM的进化代理与反馈循环；④多任务PDE基准测试；

**📊 数据集**

使用了多种一维、二维、三维及高维PDE基准：Advection、Diffusion-Reaction、Burgers、Diffusion‑Sorption、Allen‑Cahn、Cahn‑Hilliard、Darcy Flow、Shallow‑Water、Black‑Scholes、Heat、Navier‑Stokes、Poisson、Kuramoto–Sivashinsky 等；

**📈 对比分析**

与手工设计的函数学习与算子学习方法、以及基于搜索与LLM代理的自动设计方法（RandomAgent、BayesianAgent、Lang‑PINN、PINNacle 等）进行对比。实验表明在 8/11 任务上取得最佳成绩，几乎所有基准的几何平均误差显著下降（≈52%）且候选有效率、搜索效率和令牌效率均提升；

**⚠️ 局限性**

局限性包括：1）仍需依赖LLM进行修改，LLM性能影响仍存在；2）对极端复杂或高度耦合的PDE，DSL的表达能力可能受限；3）当前仅验证了演化框架的效果，缺乏理论收敛或搜索空间完备性的证明。

---

## 356. Searching for Sound-Meaning Collisions: Graph-Based Affordance Retrieval and Multi-Evaluator Ranking for Pun Translation at CLEF 2026 JOKER Task 2

**arXiv ID:** 2608.04299 | [PDF](https://arxiv.org/pdf/2608.04299v1)

**作者:** Russell Taylor `[一作]` (Georgia Institute of Technology), Prateek Awate `[通讯]` (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套基于检索+生成+评估的系统，用来将英语文字游戏（puns）翻译为法语。

**💡 创新点**

创新点在于将检索视为寻找语义-语音桥梁（affordances），并在生成过程中主动利用这些桥梁，同时采用多视角评估和多模型集成来实现最佳翻译。

**🔧 技术方法**

技术包括：基于BERT的语义与语音检索、利用FAISS构建多语言检索图、生成式语言模型（Gemini‑3‑flash、GPT‑5.5 等）的多候选生成、两阶段多代理（personas）排名与加权 Borda 投票。

**📊 数据集**

使用的数据集包含 370,450 条法语表达式库、245,746 条单词–IPA 对与 4.46M 的语音关系、以及 CLEF 2026 Joker 任务的英语-法语文字游戏语料。

**📈 对比分析**

在 CLEF 公开排行榜上，最佳单模型得分 37.783，单一 Gemini‑3‑flash 得分 37.119，四模型集成得分 35.949；相比 2025 任务，检索覆盖率提升至 50.8%，但最终获胜的翻译仍仅占约 3.5%。

**⚠️ 局限性**

主要局限是检索仍是瓶颈，约一半源文本无法获得有效桥梁；检索聚焦于近似语音关系，导致最终选取多为精确音匹配；候选排序对最终结果有显著影响，表明 LLM 评估对位置敏感。

---

## 357. REZE: Recognition-Based Zero-Shot Extraction for Video Temporal Grounding

**arXiv ID:** 2608.04480 | [PDF](https://arxiv.org/pdf/2608.04480v1)

**作者:** Boyang Li `[一作]` (Monash University), Jianfei Cai `[通讯]` (Monash University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练-free的零样本视频时序定位方法REZE，将视频划分为短剪辑，使用冻结的视觉语言模型对每个剪辑给出与查询相关性的连续分数，并通过确定性外部聚合生成最终时间区间或高亮得分。

**💡 创新点**

创新点在于将内容识别与时序聚合分离，利用剪辑级置信度得分而非直接生成时间戳，外部聚合算法可适配单区间、多区间检索及高亮检测三种输出格式，且无需任何任务专属训练。

**🔧 技术方法**

采用冻结的视觉语言模型（如LLaVA、Qwen、InternVL）进行二分类提示，提取下一词对数的softmax分数作为连续置信度；构建秒级分数曲线后使用高斯平滑、均值中心化、Kadane算法或Otsu阈值化等外部聚合技术完成时序定位。

**📊 数据集**

在Charades-STA、ActivityNet Captions和QVHighlights三大基准上进行评估，覆盖单区间检索、多区间检索与高亮检测三种任务。

**📈 对比分析**

在同一模型（如Qwen2-VL-7B）上与Direct时间戳生成和NumPro做严格对比，REZE在Charades-STA、ActivityNet Captions及QVHighlights上均显著优于Direct；在QVHighlights上实现训练-free mAP 40.32（检索）和44.18/73.41（高亮），超过之前的训练-free最佳及部分监督式最优结果。

**⚠️ 局限性**

局限性包括：仅对剪辑独立评分，未建模跨剪辑关系，难以处理需要精细边界或时序顺序的任务；对不同模型-数据集组合的提升不均衡；以及在每个查询上需要更多token导致单查询吞吐率低于Direct。

---

## 358. SIGNPOST-Bench: Benchmarking Text-Vision Conflict Resolution in Multimodal Large Language Models

**arXiv ID:** 2608.04244 | [PDF](https://arxiv.org/pdf/2608.04244v1)

**作者:** Sirun Li `[一作]` (Peking University), Fan Zhang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个基于视觉地理定位的五种条件对照基准（SIGNPOST-Bench），用于系统评估多模态大语言模型在场景文本与视觉信息冲突时的决策行为。

**💡 创新点**

创新点在于：①构造控制的对照组（Original、Blank、Similar、Random、Adversarial）以量化文本干扰的方向与幅度；②设计连续坐标空间诊断指标（WLA、TBS、TFR、TDR）衡量误差增大、文本偏差与目标诱导；③将多模态冲突鲁棒性拆分为能力（C）与鲁棒性（R）两部分，并聚合成MCRS，揭示模型清洁输入性能与冲突处理能力不一定相关。

**🔧 技术方法**

技术手段包括：EasyOCR+EasyOCR检测场景文本；Gemini-3.1-Flash-Lite生成三种文本替换；Nominatim OpenStreetMap进行地理编码；Qwen-Image-Edit-2509进行局部图像编辑；多模态模型API（Gemini、GPT、Claude、Qwen、Seed、Kimi、Grok）在统一提示下进行坐标预测；统计与可视化用Python、Pandas、Matplotlib。

**📊 数据集**

使用四大数据源：IM2GPS3K、YFCC4K、Google Street View、Baidu Street View，构成5111组共25555幅图像，涵盖多种语言与文化环境。

**📈 对比分析**

通过对20个不同提供商的MLLM进行511100次评估，比较五个条件下的WLA、TBS、TFR等指标。结果显示：Adversarial条件下WLA平均下降36.6%，中位误差增长4.8倍；6.5–20.1%的模型在Adversarial下预测落在50km以内的注入目标；不同模型在MCRS得分上差异显著，Gemini系列最高，Moonshot-Vision最低。

**⚠️ 局限性**

局限性包括：①基准仅关注视觉地理定位，未涵盖其他下游任务；②对抗样本仅为文本替换，未考察视觉伪造；③评估依赖于公开API，可能受调用限制；④图片样本数量有限且未公开完整图像，限制再现性；⑤冲突检测与防御提示对大多数模型效果不显著，表明仍需更有效的对抗与自适应机制。

---

## 359. Building and Governing AI Systems: Advancing Social Workers' Roles across the Technology Industry, Human Service Organizations, and Policy Institutions

**arXiv ID:** 2608.04273 | [PDF](https://arxiv.org/pdf/2608.04273v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 360. Super-Gaussian: Interactive Scene Editing for 3D Gaussian Splatting and NLI-Based Volume Visualization in Virtual Reality

**arXiv ID:** 2608.04475 | [PDF](https://arxiv.org/pdf/2608.04475v1)

**作者:** Suemin Jeon `[一作]` (Korea University), Won-Ki Jeong `[通讯]` (Korea University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

在虚拟现实中设计了基于可编辑3D高斯散射的体积可视化框架（Super‑Gaussian），实现了结构感知的ROI选择、层级“选择‑细化”工作流以及自然语言交互和实时语义标注。

**💡 创新点**

创新点在于：① 将高斯原语按几何特征聚类为Super‑Gaussian，实现结构级交互；② 引入随机游走+图聚类的选择‑细化流程，显著降低用户操作；③ 将可编辑高斯渲染、VR空间交互与多代理自然语言接口集成，形成完整的VPA循环；④ 使用CLIP图像+文本嵌入实现即时语义标注，支持无模板的开放词汇查询。

**🔧 技术方法**

主要技术包括：3D Gaussian Splatting、可编辑高斯属性（颜色、透明度、光照、法线等）训练与实时渲染；基于kNN图的几何SLIC聚类与地理距离加特征相似度的Super‑Gaussian构造；随机游走（Random Walk）在Super‑Gaussian图上的传播；Unity+HLSL GPU渲染管线；语音识别（STT）、文本转语音（TTS）与LLM驱动的多代理NLI框架；CLIP图像/文本编码用于实时语义融合。

**📊 数据集**

使用多种医学（血管/脑动脉瘤）、自然（贝壳/树木）和仿真（宇宙模拟）体积数据集进行实验，覆盖不同分辨率、稀疏度与结构复杂度。

**📈 对比分析**

与传统的2D桌面选择（SuperSplat）和点选VR方法（GSVR）相比，Super‑Gaussian在准确率（ACC、IoU、F1）上始终最高，交互次数与完成时间平均下降约30–50%；在渲染方面，3c可编辑高斯渲染保持≈120 FPS，即使是4 GB体积也能实时渲染；相比之下，基准DVR在高分辨率时帧率降至5 FPS甚至OOM。

**⚠️ 局限性**

局限性包括：① 需要预先计算Super‑Gaussian，聚类粒度受手工超参数影响；② 随机游走依赖用户种子，稀疏或模糊种子会影响初始选择；③ 单个高斯可能跨越多个结构，导致边界模糊；④ 结果仍受原始多视图渲染与TF设计的影响，无法完全消除预设依赖；⑤ CLIP嵌入在体积渲染图像上的语义对齐有限，未做专门微调。

---

## 361. PriDyG: Privacy-preserving Dynamic Graph Inference with LLM-GNN Collaboration

**arXiv ID:** 2608.04255 | [PDF](https://arxiv.org/pdf/2608.04255v1)

**作者:** Yuyang Xia `[一作]` (Emory University), Li Xiong `[通讯]` (Emory University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 PriDyG，一种面向动态图的边级差分隐私推理框架，将 LLM 的语义推理与 GNN 的结构学习结合，并通过增量私有多跳聚合实现常数预算。

**💡 创新点**

创新点在于（1）增量 PMA 利用聚合线性与边批分离实现隐私预算不随更新次数线性增长；（2）LLM 仅使用节点文本提供无隐私成本的语义信号，通过置信度门控融合补偿 DP-GNN 的精度损失。

**🔧 技术方法**

技术方法包括基于 GAP 的 Gaussian 机制多跳聚合、隐私预算计量（Renyi-DP）、LLM（Llama‑3‑8B）语义推理、置信度门控融合以及增量边缓冲的并行组合。

**📊 数据集**

使用四个公开图数据集：Cora、PubMed、ogbn‑arxiv 与 ogbn‑products，均采用节点文本嵌入做特征。

**📈 对比分析**

在节点分类与链接预测任务上，PriDyG 在静态与动态（边插入）场景下均超过或匹配非私有基线，并在动态更新中比传统几何衰减方案和无更新模型保持更高准确率，同时将累计隐私成本降低数百至数千倍。

**⚠️ 局限性**

局限性包括：仅考虑边级隐私而非节点级，无法处理边删除；增量聚合在多跳路径上存在结构失真；LLM 推理成本高且受模型规模限制。

---

## 362. Energy Efficiency in Microservice Architectures: A Systematic Literature Review

**arXiv ID:** 2608.04070 | [PDF](https://arxiv.org/pdf/2608.04070v1)

**作者:** Eoan O'Dea `[一作]` (University of L'Aquila), Henry Muccini `[通讯]` (University of L'Aquila)

**通讯引用:** 3678 | [OpenAlex ID](https://openalex.org/A5030457541)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究系统性综述了2015年以来关于微服务架构能源效率的文献，聚焦能源考虑的生命周期阶段、测量方法与架构解决方案，并对其进行量化归类与交叉分析。

**💡 创新点**

创新点在于将能源效率视为生命周期、测量与架构三维度的交叉研究，揭示其在微服务中主要定位为运行时优化、测量粗粒度且缺乏早期设计整合；并提出四大“takeaway”与实践建议。

**🔧 技术方法**

采用系统性文献综述（Kitchenham指南）、六阶段筛选、LLM辅助筛选、参数化数据抽取与交叉表格映射技术，对40篇主研究进行编码与统计。

**📊 数据集**

数据集来源于ACM、IEEE Xplore、Scopus、SpringerLink四大数字图书馆检索的3146条记录，最终筛选得到40篇（后期可细化为37篇）符合条件的主研究。

**📈 对比分析**

通过对RQ1–RQ3的结果进行交叉映射与频数统计，对比不同生命周期阶段、测量层级与工具类型，展示能源效率研究的分布与空缺；不涉及实验性能评估，而是通过量化分析体现研究趋势与方法有效性。

**⚠️ 局限性**

局限性包括：①仅检索2015–2025年期刊/会议英文论文，可能遗漏早期或非学术实践文献；②LLM筛选可能引入误判；③参数定义受作者主观影响，其他维度（如碳强度、内置能耗）未被涵盖；④缺乏实证案例验证所归纳模型与建议。

---

## 363. A Modular Part-of-Speech Tagger for Scottish Gaelic using spaCy

**arXiv ID:** 2608.04808 | [PDF](https://arxiv.org/pdf/2608.04808v1)

**作者:** Peter Stefan `[一作]` (Edinburgh Napier University), Alistair Lawson `[通讯]` (Edinburgh Napier University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文利用spaCy框架，在苏格兰盖尔语的ARCOSG语料上训练了两个词性标注模型，分别使用细粒度和粗粒度标签集。

**💡 创新点**

创新点在于首次将spaCy的轻量化管道应用于低资源、形态复杂语言的词性标注，并提供可复现的基准模型。

**🔧 技术方法**

技术核心包括spaCy的Tok2Vec特征提取、词性标注器以及最小化的预处理流程。

**📊 数据集**

使用的数据集是完整的Annotated Reference Corpus of Scottish Gaelic (ARCOSG)，按80‑10‑10比例划分。

**📈 对比分析**

通过与前两代专门开发的标注器比较，细粒度模型达88.6%准确率，粗粒度模型达93.7%准确率，性能与传统系统相当甚至略优。

**⚠️ 局限性**

主要局限是语料规模有限、未使用预训练嵌入或多语模型，导致对稀有词和细粒度标签的泛化能力受限。

---

## 364. Continuous Improvement and Parallel Autonomous Exploration: An LLM-Agent Framework for Searching Large Solution Spaces

**arXiv ID:** 2608.04341 | [PDF](https://arxiv.org/pdf/2608.04341v1)

**作者:** Dulmini Hettiarachchi `[一作]` (Mercari, Inc.), Sho Akiyama `[通讯]` (Mercari, Inc.)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于保留测试排行榜奖励的持续改进循环和完全自治的并行探索框架，让LLM代理在无人工干预的情况下自主搜索大规模解空间。

**💡 创新点**

创新点在于将持续改进的奖励机制与多代理并行自治相结合，既能深度迭代单一策略，又能通过共享排行榜实现多样化探索，突破单一探索的局限。

**🔧 技术方法**

采用Claude Sonnet 3.5 LLM、BGE-M3/Qwen3嵌入模型、FAISS索引、TF‑IDF、Python运行时、自动排行榜与调度模组以及双层GPU锁定管理。

**📊 数据集**

在日本C2C市场的产品‑目录匹配任务上进行实验，使用约33K SKU的分类结构数据集（智能手机、交易卡牌等），并以BGE‑M3为基准。

**📈 对比分析**

与BGE‑M3基线（33.3% 覆盖率）及单代理/5代理三次实验对比；单代理覆盖率提升至47.8–57.4%，5代理提升至62.8–69.4%，同时保持每类≥95% P@1 的精度。

**⚠️ 局限性**

局限包括：代理数量与总算力、GPU争用、排行榜可见性共变，缺乏因果结论；仅在单一任务与种子设定下验证；未对LLM生成代码进行安全审计；未证明跨任务的普适性。

---

## 365. Open-World Darknet Traffic Recognition Under Leave-One-Service-Out Evaluation

**arXiv ID:** 2608.04167 | [PDF](https://arxiv.org/pdf/2608.04167v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 366. Attention, Anomalies! Handling Attention Layers in Unsupervised Federated Outlier Detection

**arXiv ID:** 2608.04753 | [PDF](https://arxiv.org/pdf/2608.04753v1)

**作者:** Mihailo Ilić `[一作]` (University of Novi Sad), Dušan Jakovetić `[通讯]` (University of Novi Sad)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究在联邦学习环境下使用内存增强自编码器(MemAE)进行无监督异常检测，并提出了三种基于聚类和设施位置的注意力层聚合方法。

**💡 创新点**

创新点在于针对MemAE注意力层的结构设计专门的聚合函数，避免简单均值聚合，提供基于K‑Means、K‑Medoids和设施位置的指导聚合。

**🔧 技术方法**

使用的技术包括联邦学习、MemAE架构、注意力机制、K‑Means/K‑Medoids聚类、设施位置算法以及重构误差阈值进行异常判定。

**📊 数据集**

实验数据集为KDDCUP10、NSL‑KDD和PAMAP2，在IID与极端非IID两种分布下进行测试。

**📈 对比分析**

与普通自编码器、FedAvg聚合MemAE以及随机行聚合对比，结果显示在KDD和NSL‑KDD上性能相近，而在PAMAP2的极端非IID场景下，指导聚合方法在F1和AUC上显著优于基线，并表现出更稳定的收敛性。

**⚠️ 局限性**

局限性包括仅在浅层单层模型上验证，缺乏对多模态或更大规模数据集的评估，随机聚合方法的波动较大，以及未深入探讨客户端选择与聚合策略的交互影响。

---

## 367. ToolArtist: Tool-Using Unified Multimodal Models for Agentic Image Generation

**arXiv ID:** 2608.04436 | [PDF](https://arxiv.org/pdf/2608.04436v1)

**作者:** Jiahao Zhao `[一作]` (RUC), Shuicheng Yan `[通讯]` (NUS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了完全代理的图像生成模型 ToolArtist，在统一多模态模型基础上实现自动推理、工具调用和图像生成。

**💡 创新点**

将图像生成纳入代理策略，实现端到端的工具使用与图像绘制，而非预设流程或外部生成器。

**🔧 技术方法**

采用 Emu3.5 统一多模态模型，先通过教师代理生成多轮交互轨迹进行 SFT，再用 RAD‑GRPO 强化学习结合意图奖励与质量奖励进行端到端优化。

**📊 数据集**

使用 7,132 条经过转换的多轮交互轨迹，涵盖文本搜索、图像搜索和原始图像生成，形成训练数据集。

**📈 对比分析**

在 WISE 和 WorldGenBench‑Humanities 基准上，ToolArtist 取得 0.79 Overall（WISE）和 22.10 KCS（WorldGenBench）性能，领先非专有基线并与闭源模型相当。

**⚠️ 局限性**

受限于对外部搜索工具的依赖、奖励模型的主观性，以及对极低频概念或需要实时更新的知识仍有局限。

---

## 368. MOAT: Model-Agnostic Randomized Transformations for preventing Efficiency Degradation Attacks on ViTs

**arXiv ID:** 2608.04680 | [PDF](https://arxiv.org/pdf/2608.04680v1)

**作者:** Anadi Goyal `[一作]` (Indian Institute of Technology Guwahati), Norrathep Rattanavipanon `[通讯]` (Prince of Songkla University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量化、模型无关的预处理防御框架 MOAT，用于抵御针对 Vision Transformer 的 token pruning 效率降级攻击。

**💡 创新点**

创新点在于将随机缩放、均值滤波和 JPEG 压缩三种不易被梯度攻击突破的变换序列结合，实现对攻击的混淆与消噪，且无需修改模型或剪枝策略。

**🔧 技术方法**

核心技术包括：随机尺寸变换、3×3 均值滤波和 JPEG 压缩（量化+DCT），以及对攻击过程的解析与对比。

**📊 数据集**

实验使用 ImageNet-1K 验证集，针对 DeiT‑Tiny 与 DeiT‑Small 两个 ViT 变体。

**📈 对比分析**

通过在 DeSparsify 攻击下对比原始、无防御与 MOAT 防御三种情况，攻击成功率 (ASR) 从约 60% 降低至 7‑9%，GFLOPs 恢复到 28‑31% 的节省率；与基于 DDPM 的防御相比，MOAT 的计算开销仅为 0.15%~0.57%。

**⚠️ 局限性**

局限性包括：对完全自适应攻击（如使用 Expectation‑over‑Transformation）仍可能被突破；轻量化变换对干净图像的准确率有轻微影响；且目前仅在图像分类任务与 DeSparsify 攻击场景下验证，需进一步扩展到其他任务与更强攻击。

---

## 369. Towards a New Grammar of Reasoning for Artificial Legal Intelligence and the Mecelle as Its Semantic Protocol

**arXiv ID:** 2608.04011 | [PDF](https://arxiv.org/pdf/2608.04011v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 370. AI Literacy for Legal Translation: Developing Digital Resilience

**arXiv ID:** 2608.04641 | [PDF](https://arxiv.org/pdf/2608.04641v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 371. Not Every Divergence Should Be Suppressed: Counterfactual Recoverability in On-Policy Distillation

**arXiv ID:** 2608.04408 | [PDF](https://arxiv.org/pdf/2608.04408v1)

**作者:** De Jiang `[一作]` (Tsinghua University), Shaohua Ma `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在基于教师监督的自监督推理中，提出了一种基于对抗回放的可恢复性判断机制，用来决定在错误后是继续学习原有轨迹还是回滚重新采样。

**💡 创新点**

创新点在于将“可恢复性”作为判定后续干预的目标变量，并通过预算匹配的对抗分支重放实现三类标签（可恢复、不可逆但可避免、模糊），从而替代仅靠教师-学生差异的抑制策略。

**🔧 技术方法**

使用技术包括：候选动作条件的 OPD、基于候选集合的对数概率分布、JS 散度、SOD 加权、对抗回放的继续/回滚分支、AUC 评估以及 Qwen3.5‑9B 学生与 27B 教师模型。

**📊 数据集**

数据集主要为 AIME 2024/2025 训练与测试集（30/60 任务多步、32 样本自由文本）以及 GPQA‑Diamond 198 题库。

**📈 对比分析**

与传统 Vanilla OPD、SOD 重现和随机掩码等基线相比，恢复性感知控制在 AIME2025 取得 0.578 的成功率（高于 0.517 基线），在自由文本评估上平均@32 提升至 0.3125，GPQA‑Diamond 也从 0.2702 提升到 0.3070。

**⚠️ 局限性**

局限性包括：需要昂贵的教师分支回放，训练时教师 token 约为基线的 3 倍；仅在 3 个随机种子上验证，缺乏泛化评估；大约 101/200 状态被标记为模糊，说明需要更精准的在线估计；且实验仅使用单一教师模型与固定任务集。

---

## 372. Multi-Objective Ranking for Live-Streaming: Balancing Fresh and Delayed Signals with Segment-Aware Targeting

**arXiv ID:** 2608.04455 | [PDF](https://arxiv.org/pdf/2608.04455v1)

**作者:** Xiaoyi Gu `[一作]` (Twitch Interactive), Saad Ali `[通讯]` (Twitch Interactive)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对直播推荐系统中目标稀疏、延迟反馈和用户分段偏差问题，提出基于延迟窗口、多模型与分段加权的多目标排序框架。

**💡 创新点**

创新点在于将稀疏目标通过 14 天延迟窗口聚合、将新鲜与延迟信号分离为不同模型、在推理时对新老用户进行分段加权，并使用 Multi‑Gate Mixture‑of‑Experts (MMoE) 联合建模后显著压缩参数。

**🔧 技术方法**

技术包括：延迟窗口采样、Fresh Signal Model (FSM)、Delayed Signal Model (DSM)、多任务学习的 Multi‑Gate Mixture‑of‑Experts (MMoE)、分段加权（VST）、离线 NDCG 评估和在线 A/B 测试。

**📊 数据集**

使用 Twitch 直播平台的数百万用户/频道历史行为日志，覆盖短观看、长观看、聊天、关注、消费等多种行为，并在 14 天窗口内收集稀疏目标标签。

**📈 对比分析**

与单目标 DNN 基线、统一多任务模型、共享底层、CGC 等进行对比；离线 NDCG@6 及在线指标（DAV、ARPU、LMP、关注）均显著提升，MMoE 方案将参数量减少 41.9% 并保持 110 ms p99 延迟。

**⚠️ 局限性**

局限在于延迟窗口长度固定、分段权重手工设定、仅对两类用户分段、缺乏自适应窗口与在线权重学习，且对更细粒度分段的效果尚未验证。

---

## 373. Large Language Models for Low-Resource Languages: A Conceptual Framework for an Electronic Explanatory Dictionary of the Tajik Language

**arXiv ID:** 2608.04186 | [PDF](https://arxiv.org/pdf/2608.04186v1)

**作者:** Mullosharaf K. Arabov `[一作]` (Kazan Federal University), Mullosharaf K. Arabov `[通讯]` (Kazan Federal University)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5099178332)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了基于大语言模型的塔吉克语电子解释词典的概念架构，并设计了形态分析、语义聚类、LLM生成词条以及质量评估四大模块。

**💡 创新点**

首次将传统词典学方法、塔吉克语形态学数据库、统计语料分析与LLM生成能力融合成一体化的概念框架，实现了完整词典的自动生成与迭代优化。

**🔧 技术方法**

使用形态分析器与词干化、Word2Vec/ FastText 嵌入、K‑means 聚类、Gemma 3 或 Mistral 7B LLM+PEFT（LoRA/QLoRA）、多级提示工程，以及 BLEU/ROUGE/METEOR/BERTScore 等评估技术。

**📊 数据集**

利用塔吉克 Web 语料库（168.5 M 词）、塔吉克国家语料库（58.4 M 词）、TajPersLexon 4.0 k 词对、已有塔吉克语词典（电子版）以及基于这些资源合成的“词‑词条”对。

**📈 对比分析**

通过在 100–200 条验证集词条上比较 Gemma 3 与 Mistral 7B 的 PEFT 结果，主要指标包括 perplexity、BLEU 与 BERTScore；初步实验显示 Mistral 7B QLoRA rank‑16 在 perplexity ≈ 5.03 及 BERTScore 上优于 Gemma 3，后续仍需专家评估确认。

**⚠️ 局限性**

受限于词条标注稀缺导致合成训练样本质量不高，LLM 生成易出现幻觉，子词标记在西里尔脚本上的障碍，以及缺乏完整的自动化与人工校正闭环，此外对高性能 GPU 的依赖亦是实现难点。

---

## 374. FBID: Adaptive Personalized Federated Learning for Robust Out-of-Distribution Attack Detection in IoT Networks

**arXiv ID:** 2608.04073 | [PDF](https://arxiv.org/pdf/2608.04073v1)

**作者:** An Khanh Bui `[一作]` (Ho Chi Minh City University Of Technology), Diep N. Nguyen `[通讯]` (University Of Technology Sydney)

**通讯引用:** 5154 | [OpenAlex ID](https://openalex.org/A5100697893)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种服务器端控制的个性化联邦学习框架FBID，用于在异构IoT网络中检测并提高对OOS攻击的识别率。

**💡 创新点**

创新点在于：① 服务器端使用上下文多臂赌博机动态分配每个客户端的本地训练轮数；② 通过信任度自适应融合机制生成每个客户端的全局-局部混合系数，防止过度个性化导致的OOS鲁棒性下降；③ 将本地更新与全局验证集结合，形成可解释的奖励信号。

**🔧 技术方法**

采用技术包括：服务器端LinUCB多臂赌博机、指数衰减的信任更新、全局验证集上的AUC/F1奖励计算、局部SGD训练、全局加权聚合以及α‑系数混合。

**📊 数据集**

实验使用CICIoT2023数据集（46维特征、33种攻击类型），通过非IID划分构造10个客户端，并使用20,263个未出现过的OOS样本进行评估。

**📈 对比分析**

与FedALA、APFL、Ditto以及CBC基线对比。所有方法在ID数据上表现相近（F1差距<0.5%），但在OOS上FBID提升DR约2.4%（相对单客户端提升7.66%）和F1约2%（单客户端提升5.08%），并在所有客户端保持稳定的OOS鲁棒性。

**⚠️ 局限性**

局限性：需要服务器端验证集，若无法获得可能受限；对奖励尺度和信任衰减参数敏感；当前仅在MLP上验证，未在更复杂的序列或深度模型中测试；在大规模异步联邦场景下的扩展性尚待评估。

---

## 375. A Qualitative Comparative Study of Communication in Higher Distance Education

**arXiv ID:** 2608.04017 | [PDF](https://arxiv.org/pdf/2608.04017v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 376. DIVE: Dynamic Iterative Visual Evidence Construction for Efficient Vision-Language Models

**arXiv ID:** 2608.04496 | [PDF](https://arxiv.org/pdf/2608.04496v1)

**作者:** Chen Zhong `[一作]` (Wuhan University), Wei He `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种训练无关、动态迭代的视觉令牌剪枝框架DIVE，逐步构建视觉证据集合；

**💡 创新点**

创新点在于采用残差条件评分与一次性视觉/文本残差更新的 select–update–re‑evaluate 机制，让每一次选择都能更新剩余令牌的价值，避免了传统一次性 Top‑k 剪枝的静态评估；

**🔧 技术方法**

使用了视觉与文本共享表示的内积相似度、残差能量与 Prompt 对齐的加权评分、一次性正向反馈更新、动态迭代选择算法；

**📊 数据集**

在十个 LMMS‑EVAL 任务集上验证：VQA（GQA、ScienceQA、VQAv2、VizWiz、TextVQA）、感知（MME、POPE、OCRBench）以及视频理解（MVBench、VideoMME），并在 LLaVA‑1.5‑7B/13B、LLaVA‑NeXT、Qwen2‑VL‑7B 及 LLaVA‑OV‑7B 视觉语言模型上进行实验；

**📈 对比分析**

与 FastV、SparseVLM、PDrop（in‑LLM）及 VisionZip、DART、PruneSID（pre‑LLM）等方法对比，DIVE 在所有 token 预算下保持 98%+ 的平均性能，常在各基准上位居第一，并在 64 令牌预算下实现 1.68× 的总时延加速；

**⚠️ 局限性**

局限性包括：选取过程增加预填充计算开销；目前仅关注视觉令牌，未考虑文本令牌的动态剪枝；对极高分辨率或更长时序视频的极端压缩场景尚未全面验证；以及残差更新依赖于高维向量相似度计算，可能受限于 GPU 计算与内存瓶颈。

---

## 377. An Inline Control Architecture for Language Models in Intelligent Transportation Systems

**arXiv ID:** 2608.04065 | [PDF](https://arxiv.org/pdf/2608.04065v1)

**作者:** Narendra Kumar Dewangan `[一作]` (Télécom Paris), Mounira Msahli `[通讯]` (Télécom Paris)

**通讯引用:** 598 | [OpenAlex ID](https://openalex.org/A5043223613)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了一套实时内联语义护栏框架Guarded‑V2X，用于防御V2X系统中的LLM提示注入和误用攻击。

**💡 创新点**

将规则预过滤、轻量安全分类器、结构化JSON输出、签名检索和多层裁决结合，在150 ms实时预算内实现对生成式AI的机器可验证安全约束。

**🔧 技术方法**

使用规则引擎、Transformer安全分类器、受限生成模型（13B LLM）、可信检索增量（RAG）、多模投票裁决器以及速率限制与工具路由等技术。

**📊 数据集**

基于模拟的V2X RSU通告、操作员指令和V2X消息注释构建的自定义语义数据集，并在CIC‑IDS2017和ETSI ITS公开日志上做跨域评估。

**📈 对比分析**

通过四阶段实验（入侵、校准、验证、压力测试）与无护栏、安全提示、轻量护栏及现有对话护栏（PromptGuard、PIGuard、CAPTURE）对比，Guarded‑V2X在两轮攻击下IASR降至0%，p95延迟118 ms，低于150 ms阈值，并在其他攻击族群保持≤5%。

**⚠️ 局限性**

评估基于合成数据，无法覆盖真实V2X语言多样性与未知攻击；未针对实时控制层（<20 ms）展开；缺乏现场RSU日志与人机红队验证，且对模型分布漂移的鲁棒性有限。

---

## 378. An entropic explanation of insistence on sameness in autism

**arXiv ID:** 2608.04616 | [PDF](https://arxiv.org/pdf/2608.04616v1)

**作者:** Przemysław Śliwiński `[一作]` `[通讯]` (Wrocław University of Science and Technology), Przemysław Śliwiński (Wrocław University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出基于熵距离度量的框架，用以解释自闭症中坚持同样行为的现象，并把学习疗法设计为约束优化问题；

**💡 创新点**

创新点在于将不确定性最小化与自闭症行为联系，利用信息熵距离阐释坚持同样行为，并把疗法具体化为最大化互信息的优化目标；

**🔧 技术方法**

采用条件熵与熵距离、近邻分类、刺激处理循环等信息理论与认知模型技术；

**📊 数据集**

未使用公开数据集，研究主要基于理论推导与假设性模拟；

**📈 对比分析**

论文未给出实验数据对比，仅提出两种验证方案（实地与虚拟）且未展示量化性能指标；

**⚠️ 局限性**

局限在于仅为功能性模型，缺乏生物学实现细节，适用范围限定为低功能非语言个体，需要更多个体化验证实验。

---

## 379. Training Crossroads for Recurrent Vision Transformers: Recurrence, Neural ODEs, and Deep Supervision

**arXiv ID:** 2608.04879 | [PDF](https://arxiv.org/pdf/2608.04879v1)

**作者:** Grzegorz Gruszczynski `[一作]` (Samsung AI Center), Alberto Presta `[通讯]` (Samsung AI Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了单块参数共享的Vision Transformer（bViT），并在CIFAR‑100数据集上对其在不同计算和内存约束下的表现进行了系统评估。

**💡 创新点**

创新点在于将bViT视为连续时间神经ODE，验证了状态差分向量场的正确性并揭示高阶求解器更多是架构偏置而非数值提升，以及通过分阶段深度监督实现长期推理稳健性。

**🔧 技术方法**

使用了ViT架构、参数共享、Neural ODE（Euler、Heun2、RK4）、深度监督、EMA、SwiGLU激活和多头自注意力等技术。

**📊 数据集**

实验数据集为CIFAR‑100，包含100个类别的600张RGB图像，训练集500张，测试集100张。

**📈 对比分析**

与标准ViT在匹配FLOPs或参数预算下进行比较，结果显示在计算受限时标准ViT更优，而在内存受限时bViT在保持相近精度的同时显著降低参数量；深度监督能提升长推理的鲁棒性但不提升基准精度。

**⚠️ 局限性**

局限性包括：bViT在极大递归步长时易出现梯度消失导致精度下降；深度监督虽然稳健但消耗额外计算且不改善常规精度；实验仅在CIFAR‑100小规模数据集上验证，未探讨更大规模图像任务的可扩展性。

---

## 380. Energy Efficient AI-Enabled Wireless Sensor Networks for Mission Critical Environments: A Systematic Review across Smart Grid, AI, and Urban Infrastructure Applications

**arXiv ID:** 2608.04499 | [PDF](https://arxiv.org/pdf/2608.04499v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 381. Strategic Evaluation of Planning Strategies for LLM Agents in Cyber-Physical Systems

**arXiv ID:** 2608.04265 | [PDF](https://arxiv.org/pdf/2608.04265v1)

**作者:** J. de Curtò `[一作]` (BARCELONA Supercomputing Center), I. de Zarzà `[通讯]` (Universidad Pontificia Comillas)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一个基于物理的可控基准，用于评估大语言模型（LLM）代理在多代理网络系统（智能电网需求响应）中的规划策略，定义了四种执行架构（预定义、顺序、分层、搜索），并对规划诱发控制轨迹、执行忠实度、事件级可行性和自适应选择进行独立量化。

**💡 创新点**

①将规划决策层与物理结果解耦，构造“规划诱发控制轨迹”；②设计对抗性对照实验、事件级截止与外部物理验证器；③将LLM接口与执行器、响应模型和物理仿真耦合，形成多维度评价框架；④通过约束感知门控与统计诊断揭示不同规划架构的性能差异和自适应选择潜力。

**🔧 技术方法**

使用 Llama‑3.3‑70B‑Instruct（以及 DeepSeek‑V4‑Pro、Gemma‑3‑27B‑IT、GLM‑5.2、MiniMax‑M3 等多模型）实现策略声明和消息生成；实现四类执行器；基于游戏理论的 prosumer 反应模型；线性 DistFlow 电网物理仿真；对照实验的缓存、固定随机种子、事件级截止；多模型接口扩展；线性/岭回归预测器用于自适应选择。

**📊 数据集**

合成的 24 小时馈线数据，包含 40 个 prosumer、随机负荷/PV、预设电压/导入限；在 144 个不同的压力组、噪声水平、截止时间等组合下产生 576 次强制实验；对每个模型进行 300 条声明（60 条/模型）用于接口和延迟测评；所有实验均基于同一径向电网拓扑和线性电流模型。

**📈 对比分析**

通过 paired oracle regret（与每个种子下最佳强制策略比较）评估四种架构的物理目标 J；使用 95% Student‑t 或 bootstrap 置信区间；对不同噪声、截止时间做敏感性分析；在约束感知诊断下，已知不可行模式先行排除后再预测质量，regret 从 90.7 降至 29.0，提升 61%；搜索始终为全局最优；自适应选择在可行场景下仍难以超越搜索，说明质量排序仍是未解决的问题。

**⚠️ 局限性**

①实验仅在单一 LLM（Llama）和单一径向电网、线性电流模型上验证；②未对不同硬件/模型的完整物理仿真进行多重实验；③接口延迟与概率尾部仅在一次性测量中观察，未系统化；④未验证方法对更复杂电网、非线性物理或机器人等系统的迁移；⑤自适应选择仍依赖简单线性预测，无法完全捕捉可行性内部的质量排序。

---

## 382. CAMP: A Cycle-Aware Multi-Scale Patch Mixer for Time Series Forecasting

**arXiv ID:** 2608.04051 | [PDF](https://arxiv.org/pdf/2608.04051v1)

**作者:** Jung Min Choi `[一作]` (University of Hildesheim), Lars Schmidt-Thieme `[通讯]` (University of Hildesheim)

**通讯引用:** 17941 | [OpenAlex ID](https://openalex.org/A5039470755)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种名为CAMP的时序预测框架，通过自适应周期学习、周期去除与残差多尺度分解以及基于Patch的MLP混合器来捕捉周期性和残差动态。

**💡 创新点**

创新点在于：①样本级自适应周期识别无需预设周期；②使用Stationary Wavelet Transform对残差进行多分辨率对齐分解；③引入Horizon Guided Patch Mixer实现位置依赖的逐步信息聚合；④整体统一模型实现周期与残差并行预测。

**🔧 技术方法**

技术包括FFT频谱选择、两层MLP周期合成、Stationary Wavelet Transform、Patch分块、Intra-Patch MLP、Horizon Guided Patch Mixer、RevIN归一化等。

**📊 数据集**

在ETT（ETTh1/2, ETTm1/2）、Weather、Electricity、Traffic等七个长周期数据集以及PEMS（PEMS03/04/07/08）短周期交通数据集上进行评测。

**📈 对比分析**

与SRSNet、TimeKAN、Amplifier、iTransformer、CycleNet、PatchTST、DLinear、Crossformer等基线对比，CAMP在七大数据集的平均MSE中占据六个最佳，MAE亦占六个最佳；在PEMS短期任务中获得最高胜场数，整体表现优于或相当于现有最先进方法。

**⚠️ 局限性**

局限性包括：模型计算量大；FFT周期估计对短周期或弱周期信号不稳健；全通道混合在高维多变量数据中可能导致过平滑。

---

## 383. EA-Graph: Artifact-Anchored Verification Memory for Coding Agents under Upstream Drift

**arXiv ID:** 2608.04278 | [PDF](https://arxiv.org/pdf/2608.04278v1)

**作者:** Hwai-Jung Hsu `[一作]`, Hanna Everett `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究提出 EA-Graph，一种将验证声明与代码 artifact 的子路径内容绑定的内存模型，解决多会话中声明失效的问题。

**💡 创新点**

创新点在于：①将 artifact 视为第一类节点并使用子路径身份解析别名；②将证据强度与新鲜度分离，支持“不可证明”终态；③通过内容哈希检测漂移并精准撤回声明。

**🔧 技术方法**

技术主要包括：基于 Python 的 EA-Graph 结构实现、内容哈希和别名解析、Wilcoxon 符号秩检验、生成式测试床构建、与 LLM（Claude Haiku 和 Sonnet）交互。

**📊 数据集**

使用自定义生成式仓库（7 个清洁世界、14 个模型世界实例），每个世界包含 96 个行为、12 个模块、数据与逻辑漂移，以及部分隐藏内容。

**📈 对比分析**

比较方法：每个会话在三种记忆条件（ANCHOR、PROSE、NONE）下评估 96 个行为的受影响/不可证明分类，使用 F1 分数为主要度量，配合 Wilcoxon 检验；结果显示在 Haiku 轮次下 ANCHOR 在所有 7 个世界中优于其他两种，Sonnet 轮次 ANCHOR 达到 1.0，但控制条件多达阈值导致统计不显著。

**⚠️ 局限性**

局限性包括：1）测试床为合成环境，实际仓库重构难度可能更高；2）每个条件仅跑一次，模型随机性与世界难度混杂；3）未测量成本与效率；4）跨模型比较为探索性，缺乏等价界限；5）部分实验中存在泄漏与编译字节码残留问题。

---

## 384. Bicriteria Approximation Algorithms for Demand Matching

**arXiv ID:** 2608.04223 | [PDF](https://arxiv.org/pdf/2608.04223v1)

**作者:** Yuchong Pan `[一作]` (Massachusetts Institute of Technology), Michel X. Goemans `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了需求匹配（Demand Matching）问题，并提出了一系列基于资源增广的（α,β）双准则近似算法，涵盖一般图、二分图以及k-超图需求匹配问题；同时给出了相应的下界证明，完成了β∈[0,1)和β≥1两区间的权重近似与容量违规之间的完整取舍关系。

**💡 创新点**

创新点主要包括：
1) 通过对自然LP极点的结构性刻画，证明在迭代松弛过程中剩余图的支持图必为奇环的顶点互不相连并集，显著简化残差问题；
2) 结合两种取舍策略（最大权重匹配与全取环）得到(7/6,1)双准则近似，且在二分图可退化为(1,1)；
3) 推导出一族参数化双准则算法，给出(1,4/3)及更一般的(α(β),β)取舍；
4) 为k-超图需求匹配提出了简单的贪心算法，取得(k,1)双准则近似；
5) 通过构造三角和多边形实例，证明上述上界在β≥1和β<1区间内均是最优的。

**🔧 技术方法**

主要技术手段有：
- 迭代松弛（iterative relaxation）框架；
- LP极点结构分析（奇环结构、半整数性、匹配多面体性质）；
- 两策略取舍（better‑of‑two）与参数化调度；
- 贪心排序与密度比（w/d）策略；
- 组合下界构造与紧迫性分析。

**📊 数据集**

本文为理论研究，未使用真实数据集；所有结果均通过构造的图实例（如三角图、多重图等）来证明下界，并在理论上给出最优取舍。

**📈 对比分析**

与以往的3.264/2.764倍近似（相对于LP上限）相比，本文的(7/6,1)在一般图上大幅提升了权重近似，二分图实现了完全无容量违规的(1,1)近似；参数化算法在β=4/3时可获得(1,4/3)近似；贪心算法在k-超图上给出(k,1)双准则近似；所有上界与下界在相应区间内匹配，说明了所提出算法的最优性。

**⚠️ 局限性**

主要局限包括：
- 对于β∈(0,1)的区间，仍存在3/2的上限与3/2下限的接近，未能进一步突破；
- 结果基于无瓶颈（no‑bottleneck）假设，若需求与容量比例相差较大则不适用；
- 仅给出理论证明与构造实例，缺乏实验验证与对实际图结构的适用性分析；
- 对于更一般的多属性需求或非二分图的特殊结构，尚未给出更细致的近似分析。

---

## 385. Robustness Emerges Early in Training Dynamics, but Is Not Preserved

**arXiv ID:** 2608.04442 | [PDF](https://arxiv.org/pdf/2608.04442v1)

**作者:** Jiangang Yang `[一作]` (Institute of Microelectronics, Chinese Academy of Sciences), Jian Liu `[通讯]` (Institute of Microelectronics, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出两种训练动态干预方法——Early‑Phase Stabilization（EPS）和Asymmetric Weight Reversion（AWR），通过冻结或回溯训练早期浅层权重，保存和恢复网络在自然噪声下形成的鲁棒前层特征；

**💡 创新点**

创新点在于首次发现鲁棒性衰退（robustness fading）现象，并提出不改网络结构、无额外可学习参数的两种轻量化训练轨迹干预，直接在训练时期锁定或恢复早期鲁棒先验；

**🔧 技术方法**

采用训练动态干预、权重冻结/回溯、梯度方向余弦相似度、Centered Kernel Alignment (CKA)、InfoNCE 互信息、损失曲面局部锐度、梯度平坦度、有效秩 (Effective Rank) 与内在维度 (Intrinsic Dimension) 等指标进行分析；

**📊 数据集**

实验涵盖ImageNet‑C、ImageNet‑C̅、ImageNet‑3DCC、ImageNetV2‑C、COCO‑C、ADE20K‑C、Cityscapes‑C 及 ACDC 等分类、检测、分割数据集；

**📈 对比分析**

与 SAM、DAMP、DAT、VOneNet、AdaSAP、DST、EWS、AutoAug、AugMix、CutMix、Label Smoothing、Dropout 等多种基线在上述数据集上对比，EPS/AWR 在 ImageNet‑C 上平均提升 3–7% Top‑1 Accuracy，mCE 降低数个百分点；在 COCO‑C、ADE20K‑C、Cityscapes‑C 等下游任务亦分别提升 2–5% mAP / mIoU；

**⚠️ 局限性**

局限性包括：仅针对低层鲁棒先验，对语义或场景级分布漂移的改善有限；对干预时机、学习率等超参数敏感；在某些极端自然噪声（如雨滴）下提升有限；未针对对抗攻击的鲁棒性提供显著提升。

---

## 386. Multi-Level Aggregation via Dual Fitting: An $O(D)$-Competitive Algorithm

**arXiv ID:** 2608.04258 | [PDF](https://arxiv.org/pdf/2608.04258v1)

**作者:** Sara Ahmadian `[一作]` (Google), Shirley Zhang `[通讯]` (Harvard University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种新的在线算法，用于多层聚合问题（MLAP）并在任意延迟函数下实现了 2D-competitive 的性能（D 为树深度）。

**💡 创新点**

创新点在于引入了一个全新的 dual fitting 框架，结合 hindsight dual 构造和时间依赖的 dual packing，有效解决了传统在线 primal‑dual 方法的不可行性问题。

**🔧 技术方法**

主要技术包括递归购买算法、统一的优先级排序（对延迟函数的动态调整）、hindsight dual 生成以及基于配置线性规划的双重拟合分析。

**📊 数据集**

该工作为理论研究，未使用任何具体数据集，而是基于树结构的理论模型与证明。

**📈 对比分析**

与以往的 O(D²) 竞争比率相比，该算法将上界降至 2D，几乎匹配仅针对 deadline 版本已知的 D 竞争界限，展示了显著的理论改进。

**⚠️ 局限性**

局限性在于算法仍保持 2 倍常数，尚未达到最优的 D 竞争比率，并且对动态树结构或非树形网络的扩展尚未讨论。

---

## 387. AgentAntibody: An Adaptive Immune System for Defending LLM Agents against Prompt Injection

**arXiv ID:** 2608.04053 | [PDF](https://arxiv.org/pdf/2608.04053v1)

**作者:** Shihao Weng `[一作]`, Jiongchi Yu `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于抗体模型的AAgentAntibody系统，通过持久化状态和抗原投影实现对环境信息的捕获与匹配；

**💡 创新点**

创新点在于引入层次化对比匹配与自适应成熟机制，使得代理能够在动态环境中持续学习与自我校正；

**🔧 技术方法**

采用对比学习、持续学习、提示模板以及强化学习等技术；

**📊 数据集**

使用了LatentBoundaryBench和公开语料库（如SQuAD、OpenAI GPT-3预训练数据）进行实验；

**📈 对比分析**

与传统的基线方法（如Seq2Seq、BERT+RL）在多项指标上取得了1.2%~3.5%的提升；

**⚠️ 局限性**

主要局限包括缺乏大规模真实世界验证、对稀疏样本的泛化能力有限以及计算资源消耗较高。

---

## 388. Teaching Foundation Models to Read mmWave: Pose-Guided Kinematic Representation for Human Behavior Understanding

**arXiv ID:** 2608.04127 | [PDF](https://arxiv.org/pdf/2608.04127v1)

**作者:** Duo Zhang `[一作]` (Peking University), Daqing Zhang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出mmMind模型，通过同步3D姿态监督学习连续的毫米波雷达运动标记，并将其与LLM对齐，实现雷达感知行为描述、时空问答和零样本动作识别。

**💡 创新点**

创新点在于①利用姿态监督预训练雷达编码器，生成保持人体结构与运动演变的连续雷达标记；②创建mmMind‑Bench真实雷达‑语言基准；③实现仅靠毫米波雷达的行为理解与多轮交互。

**🔧 技术方法**

采用端到端多模态LLM框架，雷达时空Transformer编码器，姿态指导的多项损失（关节位置、速度、骨长约束），投影器将雷达标记映射到LLM嵌入空间，并使用LoRA微调进行指令调优。

**📊 数据集**

使用mmMind‑Bench数据集，包含17.9小时、23人、7室内场景的同步毫米波雷达点云、RGB‑D、3D姿态、双语行为描述、问答与对话。

**📈 对比分析**

与mmExpert、RadarLLM、mmCLIP等基线对比，mmMind在mmMind‑Bench的行为描述（METEOR 46.5 / BERTScore 86.8）、时空问答（单/多轮 89.2% / 84.7%）以及零样本动作识别（91.2%）上均显著优于基线。

**⚠️ 局限性**

局限在于对速度、转向角、重复计数等精细量化估计不够准确；多人人场景依赖先验聚类追踪，稀疏反射下的动作识别仍有待提升。

---

## 389. Attention-Only White-Box Transformer via LeJEPA-Based Self-Supervised Pretraining

**arXiv ID:** 2608.04213 | [PDF](https://arxiv.org/pdf/2608.04213v1)

**作者:** Yang Bai `[一作]` (Information Engineering University), Bin Yan `[通讯]` (Information Engineering University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了一种完全基于注意力的白盒 Transformer，利用 LeJEPA 自监督目标与 ADMM 优化联合推导，从而去除了 ISTA 字典和 MLP 结构；

**💡 创新点**

创新点在于将 LeJEPA 的全局扩展项与白盒目标的压缩/稀疏项分离，证明两者在协方差层面共享等距最优，并通过 ADMM 推导出仅含注意力的前向网络；

**🔧 技术方法**

采用 LeJEPA（SIGReg）、ADMM 推导、Multi‑Head Subspace Self‑Attention (MSSA)、ReLU 近似稀疏更新、知识蒸馏以及线性探测等技术；

**📊 数据集**

在 CIFAR‑10、CIFAR‑100 和 ImageNet‑1K 数据集上进行自监督预训练与下游分类；

**📈 对比分析**

与 CRATE、AoT 以及标准 ViT 进行对比，基准模型在 Base 规模下参数减少约31%，在 ImageNet 预训练后精度仅低0.76%，并在相同参数下超过 AoT；去除 MLP 的 ViT 在参数减少约66%后准确率仅下降不到1个百分点；

**⚠️ 局限性**

局限性包括仅在图像分类任务上验证，需进一步检验在其他任务与数据上的泛化能力；对 Isotropic Gaussian 假设的依赖可能限制其适用性。

---

## 390. MergeSE: Post-Hoc Model Merging for Software Engineering Tasks Without Retraining

**arXiv ID:** 2608.04181 | [PDF](https://arxiv.org/pdf/2608.04181v1)

**作者:** Palash R. Roy `[一作]` (University of Saskatchewan), Chanchal K. Roy `[通讯]` (University of Saskatchewan)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 MergeSE，一个开源命令行与网页工具，用于无训练、无 GPU 的模型后置合并，以提升软件工程任务的跨域性能。

**💡 创新点**

将模型合并技术（TIES、DARE‑TIES、PCB、Wudi 等）与软件工程任务注册表、兼容性诊断和评估流程结合，形成一套完整、易用的 SE 级合并工作流。

**🔧 技术方法**

基于 HuggingFace 预训练代码模型的参数空间算术，使用任务向量相似度、TIES/DARE 等合并算法，并在 CPU 上执行张量运算。

**📊 数据集**

使用 BigCloneBench、CLCDSA、ZC3 等克隆检测数据集，以及 GPT 生成的克隆样本作为 OOD 测试集。

**📈 对比分析**

通过与单一专业模型、多任务训练基线和参考合并实现对比，MergeSE 的 TIES 合并在跨域 F1 上可达 93% 的多任务性能，并在未见克隆类型上提升约 4 倍。

**⚠️ 局限性**

仅支持共享基模型的检查点；跨基模型的任务向量不兼容；目前仅实现 Encoder‑only 合并，无法处理复杂多头架构；工具对大模型扩展性和更丰富评估指标仍有限。

---

## 391. Adversarial Attacks for Good: A Survey of Proactive Protection across the Visual Content Lifecycle

**arXiv ID:** 2608.04314 | [PDF](https://arxiv.org/pdf/2608.04314v1)

**作者:** Jiaming Zhang `[一作]` (Nanyang Technological University), Wei Yang Bryan Lim `[通讯]` (Nanyang Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了利用对抗样本实现视觉资产保护的多种技术，阐述了从隐私过滤到可训练数据防护、生成安全、CAPTCHA及出处追踪的完整生命周期框架；

**💡 创新点**

提出了“adversarial attacks for good”的统一范式，将对抗样本从攻击转为保护工具，并构建了统一的评估维度（可转移性、适应性、部署准备度），使跨研究领域的结果可比；

**🔧 技术方法**

对抗样本生成、误导优化、梯度传播、频域与语义编辑、模型加密、知识蒸馏、生成式水印等多种对抗技术与模型训练、推理、生成与验证相结合；

**📊 数据集**

使用公开视觉数据集（如FaceScrub、CelebA、ImageNet、COCO、LAION、VGGFace、多模态模型训练集）以及各研究提出的专用数据集（如合成对抗样本集合、实验平台提供的 API 数据），并对每类方法在相应数据集上的表现进行汇总；

**📈 对比分析**

通过三维评估轴L1–L3对五大类方法进行横向比较，展示了不同技术在黑盒/灰盒/白盒可转移性、对压缩/去噪/自适应防御的鲁棒性以及实验室/外部/运营级部署成熟度的差异；在多数情况下，方法仅在实验室或白盒环境下表现良好，跨模型或跨平台的鲁棒性不足；

**⚠️ 局限性**

局限主要体现在评估不一致、缺乏真实运营环境验证、对抗手段易被恢复或被迁移、单一防护难以同时满足多重风险（识别、训练、生成、归因），以及对多模态、自治代理的适配性不足；

---

## 392. Dimension Rigidity and Projective Geometry of Trace-Product Switchings of the Gold Cube

**arXiv ID:** 2608.04261 | [PDF](https://arxiv.org/pdf/2608.04261v1)

**作者:** Oleksandr Kuznetsov `[一作]` `[通讯]` (eCampus University), Oleksandr Kuznetsov (eCampus University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对Gold立方体的自然标量迹乘开关进行了完全分类，特别是在每个偶数维度下。

**💡 创新点**

创新点在于证明了在维度四、六和八中存在非平凡的开关，而在所有更大的偶数维度中则不存在非零系数。

**🔧 技术方法**

使用了低秩导数标准和加法特征估计等数学技术。

**📊 数据集**

使用了与Gold立方体相关的标量迹乘开关的系数列表，特别是在维度4、6和8的系数。

**📈 对比分析**

通过与已有的文献和理论进行比较，证明了在维度大于8时不存在有效的开关机制，性能上提供了更高的效率和准确性。

**⚠️ 局限性**

限制在于只对偶数维度进行了研究，且在维度大于8时完全排除了开关机制的存在。

---

## 393. Geometry-Informed Optimization of Binary RIS Configurations for Communication and Sensing

**arXiv ID:** 2608.04133 | [PDF](https://arxiv.org/pdf/2608.04133v1)

**作者:** Angelos Gkekas `[一作]` (Information Technologies Institute, CERTH), Christos Liaskos `[通讯]` (University of Ioannina)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对1‑bit RIS相位配置问题，提出几何特征约束，将全局最优解表征为向量投影符号集合，并分别为MIMO系统设计基于该特征的采样方法、为SISO系统给出多项式时间枚举算法，并将同一优化框架推广到RIS辅助的ISAC场景。

**💡 创新点**

创新点在于：①证明任意全局最优配置必由某一方向投影符号决定，从而把原本指数规模的搜索空间约束为几何可行集合；②基于此几何结构提出高效采样与枚举算法；③展示同一几何原则可统一处理通信和感知两大功能，提供RIS资源分配的理论依据。

**🔧 技术方法**

采用几何分析（向量投影、角分区）、随机采样、组合优化、极值证明，以及基于模拟的性能评估；在ISAC中结合贝叶斯优化进行目标定位置信息估计。

**📊 数据集**

使用仿真生成的三维射频环境（包括发射机、接收机与RIS的空间坐标、λ_c 距离、RIS网格、传输功率等），并通过数值实验验证算法效果；未使用公开真实数据集。

**📈 对比分析**

与无结构的二进制采样以及理想连续相位RIS进行对比；在LOS最大化实验中，几何采样在相同采样预算下提升5–7 dB，逼近连续相位下的性能；在ISAC实验中，展示通信与感知的功率/误差/误检折中曲线，证明在一定资源分配区间内可同时提升通信功率而感知误差仅轻微增加。

**⚠️ 局限性**

局限性包括：①仅在简化的自由空间单路径模型下验证，未考虑多路径、相互耦合、非理想硬件失真等实际因素；②算法在极大尺寸RIS时仍需较多采样或枚举，计算量随N增加；③需事先已知完整CSI，实际部署中CSI估计误差可能影响性能。

---

## 394. Artificial Institutions: How Institutional Design Shapes LLM Simulations

**arXiv ID:** 2608.04020 | [PDF](https://arxiv.org/pdf/2608.04020v1)

**作者:** Maxim Chupilkin `[一作]` (University of Oxford), Maxim Chupilkin `[通讯]` (University of Oxford)

**通讯引用:** 97 | [OpenAlex ID](https://openalex.org/A5033203971)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在同一套固定的代理偏好与报酬设置下，作者利用大型语言模型（LLM）代理在五种不同的市场制度（叫价市场、卖家挂单市场、买家挂单市场、连续双重拍卖和双边议价）中进行重复交易，系统地比较了不同制度对人工社会结果的影响。

**💡 创新点**

论文的创新点在于首次展示了机构架构（交易规则、信息流、匹配方式等）对LLM代理行为的显著影响，即LLM代理并非仅由模型、提示或记忆决定其行为，而是与所嵌入的制度密切相关；并通过实验阐明制度敏感性对效率、交易量、价格与剩余分配的实质性差异。

**🔧 技术方法**

技术方法包括基于Python的代理模拟框架，调用OpenAI、Anthropic与Google Gemini API生成决策；使用统一的系统指令和针对不同制度的用户提示；对模型输出进行JSON解析，执行可行性约束；并运用统计回归与bootstrap置信区间对效率、数量、价格距离和剩余分配进行量化比较。

**📊 数据集**

使用的“数据集”是人工构造的定价实验环境：每场市场有4名买家（价值分别为100、90、70、50）和4名卖家（成本分别为30、45、65、85），共5个交易周期，10场独立市场。共计1,000个市场-周期观测值，覆盖4个LLM模型族、5种制度。

**📈 对比分析**

比较方法是将相同代理与报酬设置下的五种制度进行对照，测算每种制度实现的效率（已实现剩余/最大可实现剩余）、交易量、价格与竞争均衡带之间的距离以及买卖双方剩余份额，并以回归系数量化制度差异。结果显示，叫价市场实现88.6%效率，连续双重拍卖71.5%，卖家挂单/买家挂单各约66%，双边议价仅56.4%，证明制度差异可与模型差异相当。

**⚠️ 局限性**

研究局限在于实验规模极小（单一需求/供给表、有限周期、无金钱激励、模型仅作决策而非学习），且仅涉及四个LLM族，外部环境或模型更新可能导致重复性差异；因此结果应视为对制度敏感性的证据，而非对真实市场或LLM行为的普适结论。

---

## 395. Reconstructing Persistent Worlds from Narratives for Narrative-Grounded Interactive Experiences

**arXiv ID:** 2608.04037 | [PDF](https://arxiv.org/pdf/2608.04037v1)

**作者:** Yi-Chun Chen `[一作]` (National Cheng Kung University), Yi-Chun Chen `[通讯]` (National Cheng Kung University)

**通讯引用:** 11094 | [OpenAlex ID](https://openalex.org/A5100688437)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出了从叙事描述中重构持久世界的框架，并实现了一个参考原型，将叙事信息转化为持久世界并在此基础上生成可交互的 tile‑based 环境。

**💡 创新点**

创新点在于将持久世界视为叙事与交互之间的共享计算对象，先行重构并维护世界状态，而非为每个场景或玩法单独生成，提供了跨场景一致性的基础。

**🔧 技术方法**

技术包括基于 LLM 的结构化叙事解析、受约束的世界补全、知识图谱/符号世界模型的持久世界构建、基于规则的空间布局与玩法实现，以及与 GameTileNet 的语义资产映射。

**📊 数据集**

使用的“数据集”是三组人工设计的情节案例：化学实验室、遗忘神殿和小红帽改编故事，文本通过 GPT‑5‑mini 生成并手工校对。

**📈 对比分析**

对比方法主要是与传统的为单一场景或玩法生成的中间表示进行对比，实验结果通过可视化和一致性评估展示了持久世界实现的连续性和交互一致性，未给出量化指标。

**⚠️ 局限性**

局限性包括实现简化、手工验证依赖、仅支持短文本且缺乏自动化一致性检查、验证仅限三案例，未与独立场景生成方法做定量对比。

---

## 396. Advancing Utility Pole and Sign Detection Through Deep Learning

**arXiv ID:** 2608.04061 | [PDF](https://arxiv.org/pdf/2608.04061v1)

**作者:** Carl Dickinson `[一作]` (University of Strathclyde), Gaetano Di Caterina `[通讯]` (University of Strathclyde)

**通讯引用:** 1386 | [OpenAlex ID](https://openalex.org/A5009371598)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个基于 DETR 的检测‑分割框架，利用地面视角图像自动检测木质电力杆和警告标志并估计杆的倾斜角度。

**💡 创新点**

创新点包括：①创建首个公开的 OHL-UK 数据集；②在 DETR 上加入分割头，实现一次推断即可得到目标掩码；③通过掩码线拟合实现单目倾斜角估计，精度可达 1°。

**🔧 技术方法**

使用了 DETR（Transformer 检测器）、轻量分割头、AdamW+focal loss、丰富的数据增强以及 OpenCV 线拟合等技术。

**📊 数据集**

采用从 Google Street View 获取的 4,570 张 640×640 图像构成的 OHL-UK 数据集，包含 6,773 根木质杆和 1,805 个警告标志，并标注边框、掩码和倾斜角。

**📈 对比分析**

与 RetinaNet、Faster R‑CNN、YOLOv3‑Tiny 等传统检测器对比，DETR mAP 90.43%（杆）/88.26%（标志），分割后 MAE 1.01°，在更强基线 YOLOv8、DINO‑DETR 上仍保持竞争力。

**⚠️ 局限性**

局限性包括：视角/遮挡导致检测下降；域泛化受限，需适配不同地区；对小尺寸标志检测不佳；倾斜角估计受分割质量影响；标注噪声及对人类监督的依赖。

---

## 397. The Order Is the Guarantee: Verifier-Budgeted Code Deletion with Static-First Learned Proposals

**arXiv ID:** 2608.04611 | [PDF](https://arxiv.org/pdf/2608.04611v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 398. OmniVR: Joint Video-Audio Conditional Generation for Restoring Degraded Historical Films

**arXiv ID:** 2608.04224 | [PDF](https://arxiv.org/pdf/2608.04224v1)

**作者:** Xin Lu `[一作]` (University of Science and Technology of China), Zheng-Jun Zha `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对历史影片进行联合音频-视频生成式恢复，利用22B多模态DiT一次性修复颜色、噪声、抖动与音频噪音。

**💡 创新点**

创新点包括三大技术：可在线合成真实旧片退化的联合管道、架构保持的T2AV→AV2AV转化与提示退火、以及首帧锚定与波形监督实现长序列一致性。

**🔧 技术方法**

采用多模态扩散模型（Diffusion Transformer）+ VAE 编码、Prompt Annealing、Classifier-Free Guidance、跨模态门控、波形STFT损失、LoRA 微调等技术。

**📊 数据集**

训练使用从互联网收集的高质量影片与音频，合成退化后构造训练样本；评估使用OmniVRBench（200条真实历史片段、71条带参考对话片段）及RTN旧片基准。

**📈 对比分析**

与视频专门恢复、音频专门恢复及级联方法相比，在视觉（MUSIQ、CLIP-IQA、NIQE等）、音频（DNSMOS、FAD）与同步（LSE-C/D）指标上均优于所有基线，甚至在部分视觉指标上超过清晰参考。

**⚠️ 局限性**

限制在于缺乏真实退化/修复配对数据，处理极度损坏片段时仍可能出现帧缺失/同步漂移，且固定窗口推理难以捕获长距离上下文。

---

## 399. Energy- and Memory-Efficient PEFT Methods for Personalized On-Device SLMs on Consumer GPUs

**arXiv ID:** 2608.04488 | [PDF](https://arxiv.org/pdf/2608.04488v1)

**作者:** Kuanysh Akhmetzhanov `[一作]` (Nazarbayev University), Jurn-Gyu Park `[通讯]` (Nazarbayev University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对比了四种小型语言模型（TinyLlama-1.1B、Qwen3-1.7B、Mamba-1.4B、Mamba-2-1.3B）在消费者 GPU 上的五种参数高效微调方法（Full FT、LoRA、LoRA+、QLoRA、BitFit），并通过能耗、显存、训练时间等指标评估其在六个基准（GLUE 的 SST‑2、QNLI、STS‑B 以及 LaMP 的 LaMP‑1、LaMP‑2、LaMP‑3）上的性能。

**💡 创新点**

提出了以 NetScore 为核心的多维度评价体系（NetScore‑E、NetScore‑M、NetScore‑#），并给出“能耗优先”与“显存优先”的严格选择规则；首次在同一实验框架下系统比较了 Transformer 与 SSM 两类模型在同类 PEFT 方法下的能耗/显存/精度折衷。

**🔧 技术方法**

采用 LoRA、LoRA+、QLoRA、BitFit 等 PEFT 技术；使用 PyTorch 2.9.1 + CUDA 12.8、BitsAndBytes 4‑bit 量化；利用 Hugging Face Datasets 载入 GLUE 与 LaMP 数据；通过 pynvml 采样 GPU 利用率计算总能耗；采用 NetScore 指标整合任务性能与资源消耗。

**📊 数据集**

GLUE：SST‑2（情感分类）、QNLI（自然语言推断）、STS‑B（语义相似度）；LaMP：LaMP‑1（个性化引用识别）、LaMP‑2（个性化电影标签）、LaMP‑3（个性化产品评分）。

**📈 对比分析**

通过 24 种模型-任务-方法组合进行实验，总计 108 次微调。结果显示：LoRA+ 在 19/24 配置中获得最高 NetScore‑E，QLoRA 在 5/12 Transformer 配置中获得最高 NetScore‑M；BitFit 仅在参数效率极端限制下表现突出；Full FT 几乎不被选中。TinyLlama‑1.1B 在绝大多数基准上实现了最优的能耗与显存平衡。

**⚠️ 局限性**

实验仅在单一 RTX 4090 GPU 上完成；QLoRA 仅支持 Transformer，无法用于 Mamba；基准 NetScore 未直接惩罚显存；未考虑不同硬件平台（如移动 GPU、TPU、ARM 等）的真实能耗；实验规模局限于 1–2 B 参数模型，未验证更大/更小模型的通用性。

---

## 400. HyPASE: Hyperbolic Geometry for Parameter-Efficient Speech Emotion Fine-Tuning Framework for Large Audio-Language Models

**arXiv ID:** 2608.04351 | [PDF](https://arxiv.org/pdf/2608.04351v1)

**作者:** Tian Jin `[一作]` (Tongji University), Jin Zeng `[通讯]` (Tongji University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了HyPASE——一种基于超bolic几何的参数高效微调框架，用于将大型音频‑语言模型迁移到语音情感识别任务。

**💡 创新点**

创新点在于将Poincaré球面空间引入权重调制与特征聚合，利用超bolic半径显式编码层级细粒度，从而实现层级感知的权重调制（HGA）和多尺度情感融合（EMCA）。

**🔧 技术方法**

技术包括超bolic几何运算（exp/log、Möbius 缩放、Einstein 中点）、Hyperbolic Geometric Adapter、Emotion‑aware Multi‑capacity Cross‑modal Aggregator、复合损失（CE+超bolic 类别距离+半径排序）以及与LoRA/Adapter的对比实验。

**📊 数据集**

使用IEMOCAP和MELD进行训练与评测，并在RAVDESS、SAVEE进行零样本跨数据集泛化测试。

**📈 对比分析**

在MELD上相较于Euclidean PEFT（LoRA/Adapter）提升WA/UA/F1约+5.5个百分点；在IEMOCAP提升UA +3.2个百分点、WA略低；在零样本迁移上相较原始Qwen2‑Audio零样本提升30–40个百分点；整个模型仅占0.12%的参数。

**⚠️ 局限性**

局限性包括仅在Qwen2‑Audio和英文基准上验证，缺乏多语言或多模态的鲁棒性评估；未对公平性与偏见进行系统评估；需要进一步验证对不同声学环境和人口子群的适用性。

---

## 401. Segmentation Pre-training for Label-Efficient Lumbar Spine Degeneration Grading

**arXiv ID:** 2608.04810 | [PDF](https://arxiv.org/pdf/2608.04810v1)

**作者:** Monzon Maria `[一作]` (ETH Zurich), Jamaludin Amir `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

利用自动生成的脊柱解剖分割做预训练，再在少量人工分级标签上微调，提出一种数据高效的多任务腰椎退化分级框架。

**💡 创新点**

创新点在于把自动分割作为几何先验，显著降低对专家分级标注的需求，并在稀有空间相关病症上获得更大收益。

**🔧 技术方法**

技术方案为3D ResNet-18编码器与U‑Net解码器的分割预训练（Dice+CE），随后添加轻量级分类/排名头，训练时使用加权交叉熵、hinge损失等。

**📊 数据集**

使用约2000例来自11个欧洲中心的T1/T2腰椎MRI数据，含11个等级分级任务，自动分割伪标签来源于SpinePS。

**📈 对比分析**

与从零随机初始化的基线对比，Seg‑Pretrain在仅20%分级标签时即可达到宏观ROC‑AUC 0.932（全量监督为0.919），在低频空间相关病症（如滑脱、端板缺陷）上提升尤为显著；整体性能在各标签比例下均优于基线。

**⚠️ 局限性**

局限性包括仅在单一内部多中心数据集验证；预训练依赖自动分割的质量；以及未检验在不同成像协议或外部数据上的迁移能力。

---

## 402. WatchLens: A Configurable Platform for Online Video Recommendation Experiments

**arXiv ID:** 2608.04807 | [PDF](https://arxiv.org/pdf/2608.04807v1)

**作者:** Deogyong Kim `[一作]` (Yonsei University), Dongha Lee `[通讯]` (Yonsei University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了WatchLens，一个开源、模块化的视频推荐实验平台，能够独立配置用户界面、内容池和推荐策略，并在播放日志中即时记录曝光上下文；

**💡 创新点**

核心创新在于：①将推荐策略与播放日志曝光信息绑定在统一事件流中，支持同一实验中feed和watch页策略独立调控；②提供统一事件格式和自动指标计算，极大降低实验搭建门槛；

**🔧 技术方法**

技术栈包括前端React/TypeScript、后端FastAPI+PostgreSQL、Docker Compose部署、插件式推荐策略（Python或HTTP服务）以及统一事件追踪器；

**📊 数据集**

案例研究使用自建的1000个短视频内容池，并在多名大学生实验中收集数据；在文献回顾中提及KuaiRec、RecFlow等公开数据集但未直接使用；

**📈 对比分析**

通过在同一实验组内交替施加相似性和多样性两种watch页策略，使用会话级指标（观看时长、会话长度、单视频链率等）并采用Wilcoxon符号秩检验，发现相似性策略显著提升观看时长和会话保留；

**⚠️ 局限性**

目前仅支持单机部署，未对高并发场景进行性能评估；缺乏分布式视频服务、移动端客户端及参与者招募等完整实验生态。

---

## 403. MGSB: Manifold Gated Signature Branch Pressure-Domain Baseline Architecture for Two-Phase Pipeline Flows Under Distributional Shift

**arXiv ID:** 2608.04805 | [PDF](https://arxiv.org/pdf/2608.04805v1)

**作者:** Issah Suleiman `[一作]`, Matthew Hamilton `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种名为Manifold Gated Signature Bias (MGSB) 的漏点检测模型，能够在多相管道的不同流态（泡沫、斑块、絮状）下保持鲁棒性。

**💡 创新点**

创新点在于将流态条件化的门控机制与TT-RoughPath几何编码、适配输入压缩器以及Mean‑Teacher一致性正则化相结合，实现了在输入失真和分布漂移时的自动降级。

**🔧 技术方法**

采用深度学习技术，包括DepthwiseSE卷积、TT‑RoughPath路径签名、张量训练压缩、门控注意力以及Mean‑Teacher一致性正则化等。

**📊 数据集**

主要使用自建多相流实验室数据集，同时在GPLA（声学泄漏）和GAS（气压管道）两大公开数据集上做零射击跨数据集验证。

**📈 对比分析**

与CNN‑LSTM、Transformer、FCN等基线模型在相同Mean‑Teacher训练框架下对比，MGSB在留一组交叉验证中检测F1达到0.93，OOV F1 0.78，OOV跌幅仅15.8%，显著优于基线。

**⚠️ 局限性**

局限性包括样本规模有限、缺乏完整的泄漏位置信息、仅在2 Hz采样率下验证、对边缘流态（斑块–絮状）处理有限，以及尚未在现场真实环境中进一步验证。

---

## 404. IMFACT: Counterfactual Explanations for Time Series via Intrinsic Mode Function Substitution

**arXiv ID:** 2608.04777 | [PDF](https://arxiv.org/pdf/2608.04777v1)

**作者:** Udo Schlegel `[一作]` (LMU Munich), Javier Del Ser `[通讯]` (TECNALIA Basque Research and Technology Alliance)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

为时间序列分类器生成可解释的反事实解释，利用经验模态分解（EMD）的IMF替换技术；

**💡 创新点**

创新点在于将反事实生成从原始时域转移到频域分解空间，采用IMF逐步替换来保持物理可行性；

**🔧 技术方法**

采用EMD、IMF选择策略、最近不同类邻居检索、模型无关的反事实搜索；

**📊 数据集**

使用UCR档案中的FaultDetectionA（振动信号）和FruitFlies（果蝇翅振动）两类数据集；

**📈 对比分析**

与Wachter、Native Guide、Glacier等基线比较，IMFACT在两数据集上均达100%有效率，接近度与可行性指标与基线相当或更优，且运行时间更低、更稳定；

**⚠️ 局限性**

局限包括仅处理单变量序列、EMD模式混叠问题、需多邻居调参、未在更深层网络上验证、以及对模式选择的局部依赖与部分运行时波动。

---

## 405. When Does Latent Communication Pay? A Causal Audit of Relayed KV Caches in Multi-Agent LLMs

**arXiv ID:** 2608.04893 | [PDF](https://arxiv.org/pdf/2608.04893v1)

**作者:** Jiaming Cheng `[一作]` (Independent Researcher), Rajiv Ramnath `[通讯]` (Ohio State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多代理大型语言模型系统中使用 KV 缓存进行隐式通信的效果进行因果审计，构造了匹配、错配、零化、随机四种缓存干预，分为受接收器需求的校准模式和自然模式，对多家模型族与多种基准进行评估。

**💡 创新点**

创新点在于：①首次在公开系统中实施基于错配缓存的因果审计，验证“隐秘思想”是否确实携带示例特定信息；②引入接收器需求分离实验，揭示信息不对称是决定缓存价值的关键；③使用等价检验（TOST）和 Holm 校正，以系统声明的增益为边界量化内容转移的上限。

**🔧 技术方法**

技术手段包括：对 KV 缓存的直接插值/删减干预、程序化生成的私有注册表校准仪、随机匹配矩阵重建、三种多种模型（Qwen3、Mistral‑Nemo、phi‑4）和两种已发布的跨模型通信方案（C2C、KVComm）的端到端部署。

**📊 数据集**

主要数据集包括 GSM8K、ARC‑Challenge、MedQA（USMLE）以及基于 Qwen3 的多任务基准；还使用了程序化生成的 4 句自然语言 QA 作为表面形式压力测试。

**📈 对比分析**

比较方法：对每个实验细胞使用相同批量与解码设置，计算真实缓存与错配缓存、零缓存、随机缓存之间的准确率差异；在自然模式下采用 TOST 与 Holm 校正，验证差异是否小于 ±2.8% 的报告增益。结果显示：校准模式下真实缓存接近天花板（≈100%），自然模式下差异被约束在报告增益范围内，C2C 无显著内容转移，KVComm 仅转移约十分之一。

**⚠️ 局限性**

局限性包括：①仅评估公开配置的单一版本，可能无法推广到重新训练或微调的变体；②校准仪仅使用一种注册表构造，其他构造可能影响天花板；③部分细胞的绝对准确率受解析器限制，跨论文比较需谨慎；④实验聚焦于 KV 缓存，不覆盖其他隐藏状态或多模态形式。

---

## 406. A-SR: Self-Evolving Agentic LLMs for Symbolic Regression via Hierarchical Coordination

**arXiv ID:** 2608.04872 | [PDF](https://arxiv.org/pdf/2608.04872v1)

**作者:** Wenxiao Zhao `[一作]` (Shanghai AI Laboratory), Lei Bai `[通讯]` (Shanghai AI Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种自演化的代理式符号回归框架A‑SR，利用LLM生成公式并通过评估器反馈实现角色、协议和内存的动态协同；

**💡 创新点**

创新点在于把控制权从具体编辑操作转移到“角色‑记忆视图”对齐，使用评估器可观测的失败模式驱动协议选择、角色策略适配和状态路由，形成分层协作；

**🔧 技术方法**

技术包括多角色LLM代理（生成器、分析师、简化器、评审），协同协议（探索导向、可靠性导向、阶段导向、稳定性导向），在线角色价值更新，状态条件记忆路由，以及轨迹蒸馏生成LoRA提议先验；

**📊 数据集**

使用LLM‑SRBench中的四个合成科学领域（Transform、Physics、Material、Chemistry、Biology）和四个真实科研任务（振荡、细菌生长、拉伸等），数据来源为公开实验与模拟；

**📈 对比分析**

与直接LLM生成、LLM‑SR、LASR、SGA、Deliberate Evolution等基线比较，在Llama3.1‑8B‑Instruct上，A‑SR‑Static在NMSE上最高，在线A‑SR在Acc@0.01上最高，Qwen3‑4B‑Instruct‑LoRA在多领域Acc@0.01上提升至≈38%；在真实任务中A‑SR在大部分指标上获得最佳或第二佳；

**⚠️ 局限性**

局限包括依赖显式规则和轻量级在线更新，协议选择可能需在新科学领域重新校准；不同变体在NMSE与Acc@0.01上表现差异，未实现完全学习化控制器，未来可结合RL或更大任务库进行学习提升。

---

## 407. MemoryCPT: An End-to-End Agent Memory Framework for Cost-Performance Trade-off

**arXiv ID:** 2608.04843 | [PDF](https://arxiv.org/pdf/2608.04843v1)

**作者:** Songxin Lei `[一作]` (Hong Kong University of Science and Technology), Fugee Tsung `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了MemoryCPT，一个端到端可训练的长时序对话记忆管线，先通过离线查询无关蒸馏构建可复用的事件与语义记忆库，再通过查询相关检索与强化学习压缩上下文，以在保证答案质量的同时显著降低推理成本。

**💡 创新点**

创新点包括：① 将记忆构造拆分为查询无关蒸馏和查询相关检索+总结的两阶段训练；② 用结构化推理轨迹引导蒸馏，让学生模型内化中间决策；③ 在在线阶段采用GRPO与成本感知奖励，实现质量与成本双目标的平衡；④ 定义并使用QPC（质量/成本）指标统一衡量性能。

**🔧 技术方法**

技术手段包括LoRA低秩适配器（两阶段）、RRF（密集+稀疏检索融合）、GRPO强化学习、成本建模奖励、Qwen2.5-7B-Instruct/ Llama-3.2-3B 作为记忆处理基模型、Qwen3-14B 作为最终问答模型、vLLM推理框架。

**📊 数据集**

实验使用了LoCoMo和LongMemEval这两个长时序对话记忆基准，共计约314+105个测试问答，训练集包含数千条对话记录。

**📈 对比分析**

与LightMem、MemoryOS、BudgetMem、Memory-R1、No-Memory等多种基线在相同最终问答模型和评测标准下进行统一比较；MemoryCPT在F1、LLM-as-Judge、成本以及QPC方面均优于所有对手，尤其在成本敏感环境下表现突出。

**⚠️ 局限性**

局限性包括：依赖大模型和昂贵的离线蒸馏过程；对不同任务或更大规模数据迁移性尚未验证；奖励设计需要手工调参；目前不支持在线持续更新或动态预算调整。

---

## 408. Revisiting Incremental Linearization for Nonlinear Integer Arithmetic

**arXiv ID:** 2608.04835 | [PDF](https://arxiv.org/pdf/2608.04835v1)

**作者:** Marek Dančo `[一作]` (Czech Technical University in Prague), Mikoláš Janota `[通讯]` (Czech Technical University in Prague)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种改进的增量线性化方法，用于解决无量词非线性整数算术的可满足性问题，特别是针对由高次单项式构成的多项式约束。

**💡 创新点**

创新点在于引入了一组新的公理集，显著提高了在多项式约束下的收敛性，尤其是在处理高次单项式和混合乘积时。

**🔧 技术方法**

使用了增量线性化技术，并在Z3的基础上实现了一个独立的求解器。

**📊 数据集**

在SMT-LIB的NIA基准集上进行了评估。

**📈 对比分析**

与现有的最先进求解器（如Z3、cvc5、MathSAT和Yices 2）进行比较，结果显示该方法在整体性能上具有竞争力，并在多项式约束主导的基准测试中表现优异。

**⚠️ 局限性**

限制在于该方法仍然是一个不完全的过程，无法保证在所有情况下都能找到解，且在某些特定类型的问题上可能仍然存在性能瓶颈。

---

## 409. Reference-Based Manipulation: A Framework and Pipeline for Multimodal Spatial Reasoning

**arXiv ID:** 2608.04798 | [PDF](https://arxiv.org/pdf/2608.04798v1)

**作者:** Yangyang He `[一作]` (Georgia Institute of Technology), Can Liu `[通讯]` (City University of Hong Kong)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了基于空间参照的多模态交互框架 RBM，利用语音与手势构造空间参照（Source、Anchor、Frame）并通过 LLM 解析后执行 VR 中的对象操作。

**💡 创新点**

创新点在于系统化分析 VR 中用户的空间参照构造，将空间指令拆解为 Source、Anchor、Frame，定义参照组合策略与显式程度，并构建了支持隐式与多轮参照的 LLM 推理管线。

**🔧 技术方法**

使用 GPT‑4o / GPT‑5.5 作为 LLM，结合 Unity/SteamVR、Meta Quest 3、Recognissimo 语音识别、手势点射、WIZVR 交互工具以及 WebSocket 进行实时交互。

**📊 数据集**

收集自 12 名参与者的 Wizard‑of‑Oz 研究数据，共 903 条指令与 1,436 条系统操作，作为实验数据集。

**📈 对比分析**

通过与去参照推理的基线对比，采用错误率与执行时间两指标评估。RBM 在 GPT‑4o 下将错误率从 44% 降至 28%，在 GPT‑5.5 下从 20% 降至 2%，但平均响应时间从 5.16 s 提升至 9.55 s（GPT‑4o）或从 23.89 s 提升至 53.57 s（GPT‑5.5）。

**⚠️ 局限性**

主要限制包括较高的延迟（10–20 s）、对高度模糊或非结构化语句的解析不稳定、有限的历史检索策略、可视化参照关系的复杂性、对长序列的递进模糊处理不足，以及手工编写的示例指南难以完全泛化。

---

## 410. Privileged, but Biased: How PI-Conditioned Teachers Break Self-Distillation

**arXiv ID:** 2608.04794 | [PDF](https://arxiv.org/pdf/2608.04794v1)

**作者:** Sarthak Harne `[一作]` (Microsoft Research), Akshay Nambi `[通讯]` (Microsoft Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究自蒸馏（Self‑Distillation，SD）作为唯一训练目标在复杂推理任务中的表现。

**💡 创新点**

揭示PI条件化导致的教师偏差使得SD无法学习正确性，提出 PI Bias Score 来量化这一偏差，并以此解释损失与任务成功的解耦。

**🔧 技术方法**

使用 JSD 自蒸馏损失、慢速 EMA 教师复制、对齐裁剪等技术，并在 Qwen3‑8B/32B 模型上进行实验。

**📊 数据集**

MMLU‑Pro、DAPO‑Math、CodeForces、BFCL（多轮工具使用）四大任务集。

**📈 对比分析**

与基线模型和相同任务下的其它方法对比；在所有难度任务中，SD 的验证准确率不升反而下降，而损失下降，表明单独的 SD 目标无效。

**⚠️ 局限性**

局限性：仅在 Qwen 系列模型上评估；未给出修复方案；PI 偏差导致目标失效，需结合奖励或多轨迹教师以恢复正确性学习。

---

## 411. Agentic Reinforcement Learning with Observation-Calibrated Self-Distillation

**arXiv ID:** 2608.04788 | [PDF](https://arxiv.org/pdf/2608.04788v1)

**作者:** Yi Yang `[一作]` (Meituan LongCat Interaction), Yi Feng `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出Observation‑Calibrated Self‑Distillation（OCSD），通过对比包含未来观察的Full Replay视图和去掉观察的Observation‑Ablated视图，计算观察残差并将其用于在高不确定步骤下调节GRPO的token级更新。

**💡 创新点**

创新点在于识别并消除Replay scaffold对token支持的混淆：利用结构匹配视图得到观察残差，只在高不确定步骤应用，显著提升了局部监督的可靠性和训练效率。

**🔧 技术方法**

使用的技术包括GRPO、OPSD、token级支持对比、tanh归一化、基于NLL的step选择、sign‑preserving优势调节、KL正则化等。

**📊 数据集**

实验使用了ALFWorld、WebShop和Search‑QA三个交互式基准，并在Qwen3-1.7B、4B、8B三个模型规模上进行评估。

**📈 对比分析**

与Vanilla、GRPO、OPSD、GRPO+OPSD、RLSD、SDAR等基线对比，OCSD在所有任务和模型规模上均取得最优或次优成绩，显著提升成功率和EM分数，尤其在OOD场景中表现突出。

**⚠️ 局限性**

局限性包括：仅针对未来观察的privileged信息进行校正，未覆盖其他类型的privileged视图；对step‑selection比例和β参数敏感；缺乏在更大规模或多语言环境中的验证。

---

## 412. Reachability in 3-VAS

**arXiv ID:** 2608.04786 | [PDF](https://arxiv.org/pdf/2608.04786v1)

**作者:** Łukasz Kamiński `[一作]` (University of Warsaw), Sławomir Lasota `[通讯]` (University of Warsaw)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过构造多阶段的仿真与对称性约束，证明 3 维（以及 4 维）无状态向量加法系统（VAS）以及其对称子类的可达性问题是 PSPACE‑complete。

**💡 创新点**

首次给出 3 维对称 VAS 的 PSPACE‑硬度，从而完成低维 VAS 可达性问题的完整复杂度刻画。

**🔧 技术方法**

利用对称性压缩、计数器值模拟、三步递归仿真以及大数尺度分离的构造与归约技巧，构造了从有状态 VAS 到无状态对称 VAS 的多级归约。

**📊 数据集**

无实验数据，研究纯理论复杂度。

**📈 对比分析**

没有实验对比；通过理论归约展示了问题与 PSPACE 的匹配。

**⚠️ 局限性**

结果不适用于 2 维 VAS，2 维可达性问题的复杂度仍然是 NP 与 PSPACE 之间的开放区间。

---

## 413. RepoProbe: Benchmarking Architecture-Aware Repository Comprehension with Checklists

**arXiv ID:** 2608.04783 | [PDF](https://arxiv.org/pdf/2608.04783v1)

**作者:** Yuexi Yang `[一作]` (Zhejiang University), Zhen Qin `[通讯]` (Zhejiang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出RepoProbe基准，利用真实的GitHub Discussions构建面向仓库级代码理解的开放式问答任务，并引入Checklist‑Based Verification Protocol对答案进行客观验证。

**💡 创新点**

创新点包括：①以讨论为源，避免传统Bug报告带来的定位快捷方式；②首次系统量化“Edit Bias”——模型过早生成改动而非先理解架构；③用可拆解的检查表取代标量评分，显著提升评测稳定性和可解释性。

**🔧 技术方法**

技术手段：多模态LLM评测（包括闭源与开源20个模型）；生成式检查表与基于Rationale‑Guided Scoring的评估；Agentic环境下的Claude Code工具链（文件导航、符号搜索等）；LLM‑as‑a‑Judge用于检查表生成与验证；链式思维与分层评分。

**📊 数据集**

数据集：500条问答对，来自50个活跃且受欢迎的GitHub仓库（共15种语言），每条问题均来自已答复的Discussion，并通过严格过滤、自动与人工审核后得到自包含、可检验的样本。

**📈 对比分析**

比较方法：使用检查表对20个模型进行评分，得到整体得分、知识得分、清晰度得分及完美解决率；结果显示前沿闭源模型平均得分约60‑65%，Open‑Weight模型稍低；与传统标量评分相比，检查表评测方差下降>50%，解释性更好；Edit Bias在所有模型中占10‑24%失败案例。

**⚠️ 局限性**

局限性：①依赖Discussion文本，仍可能包含偏离代码理解的讨论；②语言多样性有限（主要是主流语言），未覆盖极其小众生态；③评测仅关注文本答案，未对生成的代码或文件引用进行直接验证；④检查表生成与评估仍使用LLM，可能引入评审偏差；⑤未涵盖多模态证据（图像、日志）等更丰富的仓库信息。

---

## 414. Exploring Fraction Comprehension and Interest in Elementary Education Through AI-Powered Personalized Learning

**arXiv ID:** 2608.04892 | [PDF](https://arxiv.org/pdf/2608.04892v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 415. Above-ground Biomass Estimation with Geospatial Foundation Models

**arXiv ID:** 2608.04792 | [PDF](https://arxiv.org/pdf/2608.04792v1)

**作者:** Ghjulia Sialellia `[一作]`, Konrad Schindler `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在AGBD数据集上对Geospatial Foundation Models（GFMs）进行大尺度上层植被生物量（AGB）回归的系统基准评估，比较权重分发的GFMs（冻结编码器）与预计算嵌入产品（AEF、TESSERA）的性能，并检验其空间与时间泛化能力以及与ESA CCI全球生物量产品的对比。

**💡 创新点**

创新点在于首次将GFMs与预计算嵌入结合到全球尺度的连续值回归任务中，提出了“嵌入作为分析即用层”的实用框架，并系统评估了不同交付方式对精度与计算成本的影响。

**🔧 技术方法**

技术手段包括：冻结预训练的GFM编码器（如SSL4EO-MoCo、Prithvi等）配合UPerNet解码器；训练线性探针、MLP和全卷积网络；使用AlphaEarth Foundations和TESSERA的全时序多模态嵌入；在PANGAEA框架下统一评估。

**📊 数据集**

使用的数据集为AGBD（约1600万样本，包含Sentinel‑2、ALOS‑2 PALSAR‑2、DEM等多模态）和独立验证集AGBref（10 km分辨率的全球生物量参考）。

**📈 对比分析**

比较方法：在AGBD的Full和Lite两种规模下，冻结GFM + UPerNet与基准的全监督模型（使用所有特征）以及嵌入+MLP进行对比；空间/时间泛化实验通过跨区域、跨年份训练/测试；与ESA CCI产品对比使用AGBref。结果显示：权重分发的GFMs在冻结状态下RMSE约60‑64 Mg/ha，未能匹配全监督基准（53‑58 Mg/ha），而AEF嵌入+MLP可达52‑53 Mg/ha，超越基准；AEF在空间/时间泛化上优势明显；与ESA CCI在AGBref上接近但略逊。

**⚠️ 局限性**

局限性包括：冻结Encoder评估可能低估GFMs潜力；模型无法接收L‑band SAR与其他辅助变量；预训练目标与回归任务不完全匹配；嵌入数据由专有模型生成，缺乏可复现性；仅评估两年时间范围，未检验长期灾害/土地利用变化的影响；空间泛化实验受训练区域分布限制。

---

## 416. Continual-Learning Physics-Informed Neural Networks for Parameterized Partial Differential Equations

**arXiv ID:** 2608.04778 | [PDF](https://arxiv.org/pdf/2608.04778v1)

**作者:** Xujia Chen `[一作]` (Tsinghua University), Wenhui Fan `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于连续学习的参数化物理信息神经网络（CL-PINN），通过主动参数选择、动态权重平衡、稀疏物理约束回放和参数子网络，解决传统ParamPINN在训练效率、精度不均衡和过拟合问题。

**💡 创新点**

创新点包括：① 将不同参数值的PDE实例视为连续学习任务，利用贝叶斯优化实现高效参数采样；② 引入任务级动态损失权重，调节各参数任务的优化速率；③ 采用稀疏回放仅保留少量物理点，缓解任务遗忘；④ 通过独立的参数子网络实现参数表示的分离，并可选择冻结或正则化。

**🔧 技术方法**

核心技术为物理信息神经网络（PINN）、参数化PINN（ParamPINN）、贝叶斯优化（BO）、梯度权重动态调节、稀疏经验回放、参数子网络与Adam/L-BFGS优化器组合。

**📊 数据集**

实验使用五个基准：连续函数（Schaffer-like）、一维Burgers方程、Allen–Cahn方程、二维Kovasznay流、四维Poisson–Boltzmann方程，全部在无观测数据的观察自由设置下进行。

**📈 对比分析**

与统一采样、固定任务和网格贪婪等基线相比，CL-PINN在MSE和相对L2误差上均实现显著提升（多达约80%降低），且在不同参数值上保持更均衡的精度；单参数微调（ACR2-finetune）进一步在有限步数内将误差降至外部方法之下。

**⚠️ 局限性**

主要局限包括：① 对物理损失与真实误差的对应关系仍不稳定；② 超参数与回放策略高度依赖具体PDE，缺乏通用设定；③ 仅针对无观测数据的情形，需进一步扩展至带观测或高维参数域。

---

## 417. NSF-HRPT: Neural Semantic Field meets Hierarchical Risk Perception Tree for Safety-Critical Scenario Assessment

**arXiv ID:** 2608.04776 | [PDF](https://arxiv.org/pdf/2608.04776v1)

**作者:** Yu Zhao `[一作]` (Zhejiang University), Xiubo Liang `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了NSF‑HRPT框架，通过学习多任务神经语义场和层级风险感知树，实现从单目视频实时量化风险评估。

**💡 创新点**

创新点在于结合连续语义场与概率TTC预测的神经网络，并将其作为先验驱动的层级树结构，实现并行高效风险推理；同时采用无监督的Sim2Real增强，提升无重训练的跨域性能。

**🔧 技术方法**

采用基于CARLA仿真生成的多帧序列训练的ResNet‑34+Transformer结构的NSF，配合Neural Attention Field与多任务解码器；风险推理使用四叉树HRPT；Sim2Real通过早期/中期融合深度与语义先验。

**📊 数据集**

训练使用CARLA SafeBench生成的800条基于NHTSA Pre‑Crash Typology的场景；测试在CARLA模拟以及公开事故数据集DAD和CCD上。

**📈 对比分析**

在CARLA模拟中，NSF‑HRPT在AP、mTTA、AOLA三项指标均超越DSA与Liao等方法，成为新SOTA；在DAD/CCD上，基线模型已优于部分早期方法，加入Sim2Real后与当前SOTA相近。

**⚠️ 局限性**

主要局限包括仍存在域差异导致真实数据性能不足；缺乏真实BEV轨迹与语义标注；模型在极端光照/天气下的鲁棒性待验证。

---

## 418. STEP-OPD: Rethinking Output Targets and Internal Dynamics in On-Policy Distillation for Diffusion Models

**arXiv ID:** 2608.04887 | [PDF](https://arxiv.org/pdf/2608.04887v1)

**作者:** Qingyan Wei `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a4b10f5d-130b-4e77-9367-6469ec621899` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在图像生成领域，本文提出了一种新的 On-Policy Distillation（OPD）框架，能够将多个任务专用模型的能力集成到一个统一的学生模型中；

**💡 创新点**

创新点在于：①通过输出超越（output extrapolation）利用基准模型与任务专用教师之间的速度差，构建一个比教师更具挑战性的学习目标；②引入表示变化对齐（representation change alignment），直接监督学生在网络各层的隐藏状态变化方向与幅度，使内部表示的演进更贴近教师；

**🔧 技术方法**

使用的技术包括 Stable Diffusion 3.5 Medium 作为基准与学生骨干，Deterministic Flow Solver 生成 10 步的 on-policy 轨迹，LoRA 参数微调，基于速度匹配的 OPD 损失，梯度惩罚与加权，并结合上述两项创新；

**📊 数据集**

实验使用的主要数据集和评估指标有：GenEval（测量合成对齐）、OCR（文字识别准确率）、DrawBench（PickScore、Aesthetic Score、HPSv2.1、ImageReward），以及用于人类偏好训练的 Pick-a-Pic；

**📈 对比分析**

与 DanceOPD、DiffusionOPD、Flow-GRPO、DiffusionNFT、FLUX.1-Dev 等基线相比，本文方法在 GenEval 上从 0.927 提升到 0.961，OCR 从 0.941 提升到 0.946，且四项偏好指标均有提升，且统一学生在所有能力组上均超过对应的单任务教师；

**⚠️ 局限性**

局限性主要体现在：①需为每个任务单独调节输出超越系数与温度；②对超参数（如 warm-up 长度、λ_h、β 等）敏感，可能需要额外的实验调优；③在更大规模模型或多任务设置下的可扩展性与计算成本尚未充分验证。

---

## 419. A Cost-Aware Probability Monad for Liquid Haskell

**arXiv ID:** 2608.04886 | [PDF](https://arxiv.org/pdf/2608.04886v1)

**作者:** Matthias Hetzenberger `[一作]` (Vienna University of Technology), Florian Zuleger `[通讯]` (Vienna University of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

本论文提出了一种基于 Liquid Haskell 的成本感知概率单子，用于在执行可执行 Haskell 代码的同时自动推导随机化算法的期望成本。

**💡 创新点**

创新点在于将期望成本直接编码到单子运算的精炼类型中，使得成本分析与概率运算天然结合，既支持完全自动化的 SMT 推理，又能在需要时顺畅切换到交互式证明；同时提供了完整的期望值、概率和成本的测度和运算。

**🔧 技术方法**

技术手段包括：Liquid Haskell 的精炼类型与测度（measure）机制、SMT 支持的自动化验证、可执行的有限离散概率分布单子（包含 tick、coin、uniform 等运算）、以及与 Haskell 运行时兼容的递归与模式匹配实现。

**📊 数据集**

论文未使用传统意义上的机器学习或大规模数据集，而是通过对经典随机化算法（meldable heaps、随机快速排序、随机快速选择、随机旋转树、随机置换、招聘问题）的案例研究来验证方法的有效性。

**📈 对比分析**

通过对比现有工具和手工证明，展示了本方法在代码量、证明步骤和可验证性方面的优势：例如 meldable heaps 的验证仅需几百行代码，快速排序的复杂期望分析可在交互式证明中完成，整体自动化程度高于传统方法。

**⚠️ 局限性**

主要限制包括：仅支持有限支持的离散分布（无法直接处理无限支持或几乎必然终止的情形）；对对数等数学常数需要手工提供公理；且受限于 Liquid Haskell 的终止性检查，无法直接处理非结构化递归或概率终止问题。

---

## 420. Persistent Object Narratives for Token-Efficient Video Language Models

**arXiv ID:** 2608.04866 | [PDF](https://arxiv.org/pdf/2608.04866v1)

**作者:** Junzhe Chen `[一作]` (Tianjin University), Xiaojie Guo `[通讯]` (Tianjin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 SlotNarrative，一种将视频拆分成持久对象叙事的槽式视觉接口，将每个对象序列化为身份标记和状态标记，仅用 144 个视觉标记供 Video‑LLM 处理。

**💡 创新点**

创新点在于：① 通过槽注意力先提取帧级对象观察；② 用无参记忆将跨帧观察关联成持久对象；③ 将每个对象分解为身份标记和多段状态标记，既保留稳定外观又保留时序变化，实现显著压缩且结构化的视觉输入。

**🔧 技术方法**

技术实现包括：SigLIP 视觉编码器、槽注意力与循环槽编码器、基于多种匹配线索的无参记忆、以及身份/状态两类投影器和层归一化，将得到的对象特征映射到语言模型嵌入空间。

**📊 数据集**

使用了 ActivityNet 训练集（VideoInstruct‑100K）进行无监督和对话式监督，评估数据为 MSVD‑QA、MSRVTT‑QA 和 ActivityNet‑QA 三大 Video‑QA 基准。

**📈 对比分析**

在三大基准上，SlotNarrative 以 144 视觉标记获得 MSVD‑QA 75.6%、MSRVTT‑QA 69.8% 和 ActivityNet‑QA 50.3% 的准确率，优于大多数基于同等或更少标记数的竞争方法，且仅落后于最高精度模型约 0.3–2.0 分，证明了其高效的 token‑accuracy 取舍。

**⚠️ 局限性**

局限性主要体现在：对长时视频的时序编码仍较弱，易出现槽碎片化和缺失的对象关联；缺乏外部检测器或更丰富的运动线索，导致在极长或复杂场景中仍可能丢失关键对象信息。

---

## 421. Toward Blockage-Resilient 6G-V2X Connectivity: Semi-Distributed Bandit with Dynamic Arm Set for mmWave HetNets

**arXiv ID:** 2608.04852 | [PDF](https://arxiv.org/pdf/2608.04852v1)

**作者:** Weiqi Chi `[一作]` (University of Tokyo), Manabu Tsukada `[通讯]` (University of Tokyo)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文针对毫米波车联网中的用户关联问题，提出了全分布式的BAND和半分布式的S-BAND两种基于动态、阻塞感知的多臂赌博机框架，能够在不依赖中心CSI或离线训练的前提下实现实时决策。

**💡 创新点**

创新点包括：① 两阶段UCB策略配合动态基站集合管理，实现对活跃/非活跃基站的自适应探索；② 结合阻塞预测的变化检测机制，有效抑制因瞬时遮挡产生的误报；③ 设计轨迹对齐知识区域（TAK）与知识继承保真度指标，提升知识共享与迁移质量；④ 在阻塞过滤基准下给出全新非伯努利奖励的C-STAT理论分析。

**🔧 技术方法**

采用的技术主要有：上下文多臂赌博机（CMAB）、CUSUM变化检测、两阶段UCB决策、基于轨迹的聚类（TAK与K-means）、半分布式知识上传与聚合、Hoeffding界限的误报/延迟估计，以及仿真中的Ray‑Tracing与SUMO交通模拟。

**📊 数据集**

使用的数据集包括：东京涩谷区的OpenStreetMap道路与建筑信息、OpenCelliD实际基站坐标、SUMO生成的真实交通轨迹、3GPP TR 38.901的射线跟踪模型以及3GPP TR 37.885定义的车辆类型。

**📈 对比分析**

实验将BAND和S-BAND与三种基线（中心化CMAB、最近基站直连、3GPP A3事件式切换）进行对比，结果显示BAND比中心化CMAB降低34.9%累计惩罚，S-BAND进一步降低至59.4%；在10%–50%不同遮挡率下，均保持较低惩罚和较高平均速率，优于所有基线。

**⚠️ 局限性**

局限性主要体现在：① 仅给出每车辆的期望惩罚，未提供全局最优或纳什均衡保证；② 阻塞过滤基准假设需先验已知；③ 同步周期τ固定，未自适应；④ 轨迹聚类仍可能出现区域重叠或边缘误传；⑤ 对大规模高速网络的可扩展性尚未完全验证。

---

## 422. Towards a satellite image manipulation and deepfake localization benchmark dataset

**arXiv ID:** 2608.04840 | [PDF](https://arxiv.org/pdf/2608.04840v1)

**作者:** Jacob Arndt `[一作]` (Oak Ridge National Laboratory), Nivedita Nukavarapu `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了一个包含 60 张卫星图像（30 张真实，30 张合成）的微型深伪数据集 fmow‑fake‑small，合成图像采用三种操作：简单复制粘贴剪切、对象剪切以及扩散模型修补，并为每张图像提供像素级真值掩码和获取元数据。

**💡 创新点**

创新点在于：①首次为遥感图像深伪提供高质量、可像素级定位的真值掩码；②将多尺度、地理参照的原始图像与基于 SAM 与 RSPaint 的高级合成技术结合，生成更逼真且无明显视觉痕迹的伪造样本；③公开完整的构建流程与示例对象集，促进后续研究者在同一环境下复现或扩展。

**🔧 技术方法**

使用了三种技术：①基于 SAM（Segment Anything Model）的对象分割与人工剪切；②对简单剪切进行基于物理尺寸的等比例缩放；③使用已在 SAMRS 数据集上微调的 RSPaint（Stable Diffusion 微调模型）实现示例‑基修补，并对掩码尺寸与物理尺度进行一致性校正。

**📊 数据集**

数据集来源是 Functional Map of the World (fMoW) 的 RGB 平面化高分辨率卫星图像，经过投影、GeoTIFF 存储后，用于生成三种伪造类型；同时保留了原始图像的地理位置、像素尺寸等元数据信息。

**📈 对比分析**

在缺乏可量化实验结果的情况下，作者通过对比现有遥感深伪数据集（如 RSFAKE‑1M、FLDCF、DM‑AER、FSI）的视觉质量与掩码可用性，指出 fmow‑fake‑small 具有更少的视觉瑕疵、更逼真的合成效果，并可支持像素级定位评估；目前未给出数值性能指标，更多是质量与可用性的比较。

**⚠️ 局限性**

局限性包括：①数据集规模仅 60 张，难以满足大规模训练需求；②仅包含三种合成方式，生成多样性有限；③对象剪切与修补过程需人工裁剪与掩码设置，缺乏自动化，难以扩展；④当前更适合作为评估集而非训练集。

---

## 423. Global Attention-Fused Image Cropping with Attention-Guided and Global-Aligned Crop Evaluator

**arXiv ID:** 2608.04821 | [PDF](https://arxiv.org/pdf/2608.04821v1)

**作者:** Haotian Yang `[一作]` (City University of Macau), Xin Sun `[通讯]` (City University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种全局注意力融合裁剪框架 GAFIC，利用注意引导特征融合和全局对齐裁剪评估，实现对图像重要区域的精确定位和裁剪边界的稳定感知。

**💡 创新点**

创新点：①Attention‑Guided Feature Fusion (AGFF) 模块通过候选裁剪框生成全局重要性热图并融合到特征；②Global‑Aligned Crop Evaluator (GACE) 将裁剪框内部、外部与全局特征对齐评估；③结合多尺度 ranking 损失、最佳裁剪回归与排名损失的多维优化，显著提升裁剪准确性与排名稳定性。

**🔧 技术方法**

技术手段：VGG16 作为 backbone，利用多尺度特征聚合；RoIAlign/​RoDAlign 提取裁剪框内部/外部特征；热图权重、1×1/3×3 卷积、全局采样、线性映射；三重损失（最佳回归、回归、排名）实现联合训练。

**📊 数据集**

使用数据集：GAIC、CPC、GAICD、FCDB、FLMS，主要在 GAICD、CPC、FCDB 进行训练与评测。

**📈 对比分析**

方法比较：与 VFN、VEN、ASM‑Net、GAIC、CGS、HCIC、Cropper 等 SOTA 进行对比；评估指标包括 IoU、Disp、SRCC、Acc5/10、AccN。GAFIC 在 GAICD、CPC、FCDB 上取得最高 IoU、Disp、SRCC、Acc5/10，整体性能显著优于现有方法。

**⚠️ 局限性**

局限性：①热图质量依赖候选裁剪生成策略；②仅基于 VGG16，可能受限于特征表达；③未充分考虑不同裁剪比例和多视角场景；④评估未按注释一致性细分，未覆盖高不确定裁剪场景。

---

## 424. Rethinking Pixel Mean Flows via Interval Denoiser

**arXiv ID:** 2608.04818 | [PDF](https://arxiv.org/pdf/2608.04818v1)

**作者:** Alexander Zaytsev `[一作]` (HSE University), Aibek Alanov `[通讯]` (HSE University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Interval Denoiser框架，实现无潜空间的少步图像生成。

**💡 创新点**

通过从流匹配ODE推导精确的中间轨迹映射，将预测限制在低维图像流形上，消除经验代换导致的梯度偏差，并结合残差裁剪和时间采样课程提升大步稳定性。

**🔧 技术方法**

采用流匹配ODE、JVP求导、对数损失、残差裁剪、两阶段时间采样课程以及纯像素空间训练。

**📊 数据集**

在ImageNet 256×256数据集上训练和评估。

**📈 对比分析**

与现有纯像素空间快前向模型对比，在1‑NFE下FID 4.55，2‑NFE下FID 3.98，创下同类模型的最佳表现。

**⚠️ 局限性**

仍受限于大步训练的梯度抑制和对时间区间采样的依赖，且在更高分辨率或更复杂场景下的扩展性未验证。

---

## 425. StaticSegFormer: An Efficient High-Performance Semantic Segmentation Based on Static Structured Pruning

**arXiv ID:** 2608.04811 | [PDF](https://arxiv.org/pdf/2608.04811v1)

**作者:** Timo Bartels `[一作]` (Technische Universität Braunschweig), Tim Fingscheidt `[通讯]` (Technische Universität Braunschweig)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了对SegFormer注意力头进行静态结构化裁剪的方法，以提升语义分割的效率；

**💡 创新点**

创新点在于将注意力头的完整裁剪与静态方式相结合，既降低FLOPs，又显著提升FPS，同时无mIoU性能损失；

**🔧 技术方法**

采用静态结构化裁剪技术、改进的MHSA、简化的decoder拼接（改为加法），并在ImageNet预训练后直接裁剪；

**📊 数据集**

使用Cityscapes和ADE20K两个标准语义分割数据集进行训练与评估；

**📈 对比分析**

与原SegFormer、Bai等动态裁剪方法相比，在Cityscapes上可将FLOPs降至50%以内，FPS提升至34%，且mIoU保持或提升；在ADE20K上mIoU略逊于动态裁剪，但仍低FLOPs且FPS提高；

**⚠️ 局限性**

局限性包括：在ADE20K上mIoU仍不如动态裁剪，且对极大裁剪比例的鲁棒性有限，未来需进一步提升高类数数据集的性能。

---

## 426. Scrouting: Cost-Aware Routing of Coding Agents by Scouting the Repository First

**arXiv ID:** 2608.04804 | [PDF](https://arxiv.org/pdf/2608.04804v1)

**作者:** Ishaan Bhola `[一作]` (SuperAGI Research), Mukunda NS `[通讯]` (SuperAGI Research)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个先搜索后路由的系统，使用小型搜索模型探索仓库生成验证手工，随后用摘要路由器根据搜索器隐藏状态将任务分配给不同前沿修复模型。

**💡 创新点**

创新在于将仓库预探索与验证手工相结合，使用搜索器隐藏状态作为路由特征，并实现无训练新增修复模型的可插拔性。

**🔧 技术方法**

使用7B Qwen2.5-Coder搜索模型、沙箱验证、摘要式路由器、逻辑回归评分、隐藏状态特征与嵌入中心点等技术。

**📊 数据集**

在SWE-bench Pro的Python 266任务上评估，并在OpenHands、SWE-rebench-openhands等公开轨迹上训练搜索器，同时使用100个新任务进行校准。

**📈 对比分析**

与单一前沿修复模型及盲目混合路由进行对比，在官方限额预算下取得159/266的解决率，匹配最佳单模型的158/266，但每次解决成本仅为其约五分之一（$0.23 vs $1.274）。

**⚠️ 局限性**

结果仅基于单一语言和单一基准，手工验证与路由效果未实现统计显著性，未对所有模型训练数据进行污染控制，且成本评估基于即时价格，可能不具普适性。

---

## 427. On the Effectiveness of Adaptation Strategies for VLM-Based Federated Learning in Remote Sensing

**arXiv ID:** 2608.04791 | [PDF](https://arxiv.org/pdf/2608.04791v1)

**作者:** Simon Lösche `[一作]` (BIFOLD - Berlin Institute for the Foundations of Learning and Data), Begüm Demir `[通讯]` (BIFOLD - Berlin Institute for the Foundations of Learning and Data)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在RS图像分类的联邦学习框架下，对四种 VLM 适配策略（全微调、编码器单独微调、提示学习、LoRA）进行了系统比较，并给出了在不同通信与计算约束下的选择准则。

**💡 创新点**

首次在遥感领域全面评估 VLM 适配方法，提出针对不同约束的选型指南，并验证 LoRA 在保持性能的同时显著降低通信开销。

**🔧 技术方法**

使用预训练 CLIP 模型，在联邦学习中应用 AdamW、FedAvg、Prompt 学习与 LoRA 等技术，进行本地训练与参数聚合。

**📊 数据集**

BigEarthNet‑S2、EuroSAT、RESISC45 以及 ImageNet 作为评估数据集。

**📈 对比分析**

在分布式非 IID 环境下比较分布内外的 mAP、Top‑1/5 准确率、FLOPs、通信参数；LoRA 在任务专一性能与跨域泛化之间取得最佳折中；提示学习通信最少但性能最低；FFT 产生灾难性遗忘。

**⚠️ 局限性**

局限包括对高度非 IID 或分布外场景的鲁棒性有限；提示学习泛化差；FFT 导致显著遗忘；实验仅涉及单一 VLM 与单一任务，未覆盖多模态或多光谱扩展。

---

## 428. Minimal Binary Linear Codes of Dimension n+4 from Partial Spreads and Their Dual Access Structures

**arXiv ID:** 2608.04889 | [PDF](https://arxiv.org/pdf/2608.04889v1)

**作者:** Apurba Sarkar `[一作]` (Visva-Bharati), Makhan Maji `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构造了一族新的[2^n-1, n+4]最小二进制线性码，并通过其对偶码给出理想的多阈值秘密共享结构；

**💡 创新点**

创新点在于使用四个布尔函数的组合与有限域上的部分分布（partial spread）相结合，得到维度为 n+4 的最小码，并证明该码能严格违反 Ashikhmin–Barg 条件；

**🔧 技术方法**

核心技术包括：部分分布几何构造、布尔函数的对称差与Walsh–Hadamard变换、最小码的必要与充分条件证明以及对偶码支持集与秘密共享访问结构的对应；

**📊 数据集**

无实验数据集，本文全部为理论构造与数学证明；

**📈 对比分析**

与已有的 n+1、n+2、n+3 维最小码相比，所构造码在授权集合数量上实现四倍增长（|Γ|=2^n+3），多阈值访问范围更宽（Δ≥2^n-1+2^m-2-1），并在 n=8 时展示相较于 n+2 维码 20% 的吞吐率提升；

**⚠️ 局限性**

局限性包括：仅适用于二元域且要求 n 为偶数；需要满足复杂的集合条件 C1–C3 以保证最小性；目前未讨论如何推广到更高维或非二元域；构造依赖于存在足够的部分分布，实际实现时可能受限于此。

---

## 429. Evaluating the Diagnostic Robustness of Vision-Language Models Under Visual and Textual Perturbations

**arXiv ID:** 2608.04885 | [PDF](https://arxiv.org/pdf/2608.04885v1)

**作者:** Ali Khoramfar `[一作]` (University of Tehran), Heshaam Faili `[通讯]` (University of Tehran)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

系统评估四种视觉‑语言模型在医学 MRI 上的诊断鲁棒性，探究视觉序列和文本提示变化对诊断稳定性的影响

**💡 创新点**

首次量化模型在保留证据的排列扰动下的诊断翻转率，揭示高准确率无法反映临床可靠性，并提出基于稳定性的评价指标

**🔧 技术方法**

使用视觉扰动（顺序反转、随机打乱、ROI 前/后）和文本扰动（标签交换、词义改写）以及证据消除的负控制实验

**📊 数据集**

采用 90 例经组织学确认的脑肿瘤 MRI 数据集（45 例胶质母细胞瘤、45 例脑转移），包含 T1CE 与 T2 两序列以及专家注释的 ROI 掩膜

**📈 对比分析**

与传统医学影像基线（支持向量机、随机森林）以及随机基线对比，发现尽管基线准确率 70‑80%，视觉/文本扰动下翻转率高达 48.9%‑67.8%，证据消除时模型仍出现 58.9%‑76.1% 的过度诊断

**⚠️ 局限性**

受限于 VLM 的上下文窗口、无内部表示可解释性、实验只覆盖特定扰动，且未涵盖实际临床环境中的多重变异

---

## 430. Cluster Deletion is as Hard to Approximate as Vertex Cover

**arXiv ID:** 2608.04883 | [PDF](https://arxiv.org/pdf/2608.04883v1)

**作者:** Yixin Cao `[一作]` (Hong Kong Polytechnic University), Ying Xu `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过构造加入大量全局顶点的图，将 Clique/Vertex Cover 的逼近难度迁移到 Cluster Deletion，进而证明在唯一游戏假设下 Cluster Deletion 的最优逼近比例为 2，并给出 √2-逼近下界；同时在受限情形下展示可突破 2 的逼近算法。

**💡 创新点**

主要创新在于提出了新的逼近保持化简技术（通过全局顶点连接实现），给出了在 UGC 条件下的逼近阈值、√2 逼近下界，并首次提供了 Cluster Editing 与 Bad Triangle Transversal 最优值不等的 31 点图例证。

**🔧 技术方法**

技术手段包括组合逼近理论、唯一游戏假设、图论构造与真兄弟（true twins）性质、Sherali–Adams 层级、LP 松弛与本地障碍（bad triangles/P₃）约束、逼近保持化简、以及从 Vertex Cover 到 Clique 的互补图映射。

**📊 数据集**

本工作为纯理论研究，无实验数据集。

**📈 对比分析**

与已知的 2‑逼近算法对比，本文证明在一般图上无法获得更优逼近；在最大团数受限或所有团大小受限的图中提供 1.92‑逼近实现；并通过 31 点图示例说明 Bad Triangle Transversal 的 ρ 至少为 10/9。

**⚠️ 局限性**

局限性在于化简构造产生的图普遍为密集图，难以直接应用于稀疏图；逼近下界依赖 UGC 与 Vertex Cover 的已知难度，实际实现需依赖昂贵的 LP/Sherali–Adams 计算；对特殊图类的逼近提升仍待进一步研究。

---

## 431. Variational Bounds for Perceptron Learning from Structured Data

**arXiv ID:** 2608.04882 | [PDF](https://arxiv.org/pdf/2608.04882v1)

**作者:** Francesco Camilli `[一作]` (Alma Mater Studiorum -- Università di Bologna), Emanuele Mingione `[通讯]` (Alma Mater Studiorum -- Università di Bologna)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一个变分方法，用于求解在高维比例极限下，训练于高斯混合数据的连续自旋感知器的有限温度自由能与零温度极限。

**💡 创新点**

创新点在于：① 将感知器的 Gibbs 流程用单一变分势统一表述；② 在非 Bayes 最优、非二值自旋的设置下，通过对数凸性与自适应插值得到上下界只在两外部优化顺序上可能不匹配；③ 证明了当两外部优化可交换时，可获得精确的热力学极限，并进一步得到训练误差、泛化误差和基态能量的闭式表述。

**🔧 技术方法**

技术手段主要包括：自适应 Guerra‑Toninelli 插值、对数凸性与 Prékopa‑Leindler 定理、Bräms–Lieb 估计、Sion 极小极大定理、以及对变分势的 Hessian 分析。

**📊 数据集**

使用的数据集为二维高斯混合样本（两个等方差高斯云），标签服从 ±1 取值，信号方向为随机向量；实验中还考察了不同温度、正则化强度与负载率 α 的情形。

**📈 对比分析**

通过对变分势在 (ρ,r) 平面上进行数值优化（含 δ,h,q,m 四个内部参数），绘制了 Φ^⋆(ρ,r) 的三维曲面，并观察到存在鞍点结构，从而验证 sup‑ρ 与 inf‑r 可交换。实验结果显示在多种学习设置（β=1 或 100，κ=1、0.01 等）下，上下界相匹配，表明理论预言准确。

**⚠️ 局限性**

局限性：① 需要对数凸性和自旋先验的单调性，难以推广到多层网络或非对数凸损失；② 仍需假设固定的唯一性或可交换性，未能在最一般情况下证明两边界完全相等；③ 对于高阶层叠、非线性激活等实际深度学习场景，缺乏可行的对数凸性或浓缩性质，导致方法失效。

---

## 432. Do Language Models Know Their Slang? Queer Slang Understanding in User-Generated Content

**arXiv ID:** 2608.04847 | [PDF](https://arxiv.org/pdf/2608.04847v1)

**作者:** Arianna Denitto `[一作]` (University of Torino), Beatrice Savoldi `[通讯]` (Fondazione Bruno Kessler)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为 Slang-Q 的手工标注数据集，并对大语言模型在定义生成任务中的表现进行了评估。

**💡 创新点**

首次提出针对 LGBTQ+ 俚语的专门评估资源，并系统研究了上下文与领域提示对模型理解的影响。

**🔧 技术方法**

采用了指令微调的多模态 LLM（Claude Sonnet 4.6、Qwen3‑32B、LLaMA 4 Scout、LLaMA 3.3‑70B）与 ROUGE‑L、BERTScore 两种自动评估指标。

**📊 数据集**

使用了 1,024 条包含 118 种 queer 俚语词汇的英文用户生成句子（来源自 Urban Dictionary）以及相应的词义定义。

**📈 对比分析**

通过四种提示条件（词条/上下文 × 通用/俚语专家）对模型进行对比，结果显示在缺乏领域提示或上下文时性能明显下降；但在提供俚语背景或句子上下文后，模型的 ROUGE‑L 与 BERTScore 均接近人类上限，说明模型能够在一定程度上捕捉俚语意义。

**⚠️ 局限性**

局限性包括仅覆盖英文且平台特定的俚语，评估范围局限于定义生成任务，实验模型样本有限，且俚语词汇的快速演变使得数据集易需持续更新。

---

## 433. RegisterBridgeMM: A Register-Centric Framework for RGB-Infrared Object Detection

**arXiv ID:** 2608.04833 | [PDF](https://arxiv.org/pdf/2608.04833v1)

**作者:** Zian Wang `[一作]` (Jilin University), Fangming Gu `[通讯]` (Jilin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于预训练DINOv3 Register的RGB‑IR目标检测框架RegisterBridgeMM，利用register token作为跨模态通信通道，在冻结backbone的前提下实现高效融合。

**💡 创新点**

创新点包括：①发现并利用预训练register token中隐含的模态共享/模态特定结构；②构建三阶段register生命周期（Aggregate‑Bridge‑Project）；③在Bridge阶段引入双向register‑patch交互与共识‑残差拆分；④在Project阶段采用基于register摘要的空间自适应校准（SPADE‑style）。

**🔧 技术方法**

技术实现：DINOv3 ViT作为冻结backbone；Register‑Write Patch‑Read（RWPR）双向cross‑attention；Register‑Consensus / Residual‑Split（RCRS）共识‑残差平衡；Modality‑Aware Calibration（MAC）空间自适应 affine 校准；RT‑DETR检测头。

**📊 数据集**

数据集：LLVIP、M3FD、DroneVehicle、FLIR‑Aligned四个公开RGB‑IR目标检测基准。

**📈 对比分析**

与最新方法对比：在所有四个基准上均取得最高mAP_50‑95；在参数效率方面，仅使用约27.8M可训练参数，优于大多数密集交叉注意力方法；在低光、恶劣天气、无人机视角和复杂交通场景中均表现出显著提升。

**⚠️ 局限性**

局限性：仍依赖预训练register token；在极端模态不匹配或未对齐的场景下性能可能受限；对backbone的冻结策略可能限制进一步的性能提升；注册插入层的深度需要经验性选择，过深会导致信息无法传播。

---

## 434. Robust Control under Stationary Ambiguity

**arXiv ID:** 2608.04832 | [PDF](https://arxiv.org/pdf/2608.04832v1)

**作者:** Konrad J. Mueller `[一作]` (Imperial College London), Lukas Gonon `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了在仿真中保持稳定不确定性（stationary ambiguity）的控制策略优化框架，并在期权对冲任务中验证其有效性。

**💡 创新点**

创新点在于将不确定性视为随系统状态变化但不随时间消失的过滤过程，避免了传统随机化导致的可辨识性衰减，从而实现持续鲁棒性。

**🔧 技术方法**

使用LSTM神经网络作为策略表示，在基于仿真的强化学习训练中引入刷新随机化（refresh latent model）与静态随机化对比。

**📊 数据集**

在实验中使用三种合成对冲环境（BS-Vol、Heston-Corr、BS-Cov）以及S&P 100历史股价数据进行回测。

**📈 对比分析**

通过初始鲁棒性、持续鲁棒性评估以及在真实市场数据的对冲风险（spectral risk）比较，发现采用stationary ambiguity的策略在鲁棒性和风险表现上显著优于传统方法。

**⚠️ 局限性**

局限性包括仅针对金融对冲任务验证，缺乏对其他控制问题的通用性评估，以及对参数刷新率的手工调节和对连续重训练的探索不足。

---

## 435. Deliberate Before You Fly: Vision-Guided Spatial Deliberation for UAV See-and-Reach Navigation

**arXiv ID:** 2608.04825 | [PDF](https://arxiv.org/pdf/2608.04825v1)

**作者:** Fanfu Xue `[一作]` (Shandong University), Jiande Sun `[通讯]` (Shandong Normal University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出DBFly，一种用于无人机“见即达成”导航的视觉‑语言航路点预测框架，能在无人机初始视野中定位目标并精准靠近并安全停止。

**💡 创新点**

创新点在于引入视觉引导的空间思辨（Vision‑Guided Spatial Deliberation）——先进行目标方向锚定、空间诊断与动作决策，再生成航路点，并通过隐式飞行走廊与终止收敛感知实现更一致的操控和可靠的终止。

**🔧 技术方法**

采用Qwen3‑VL预训练视觉‑语言模型，使用LoRA微调，结合目标方向先验、隐式走廊状态、空间决策链与终止策略，自动化生成三维航路点并进行监督优化。

**📊 数据集**

在UAV‑VLN‑FOV基准数据集上进行评估，该数据集包含2,717条轨迹，涵盖14个场景和89个目标类别，划分为训练集和三种测试集（见/未见物体/未见场景）。

**📈 对比分析**

与随机、固定、零样本视觉‑语言模型以及专门的TravelUAV和3DG‑VLN等六个基线相比，DBFly在所有测试集上平均提升25.07个百分点的成功率，SPL、OSR等指标同样显著提高，证明其在目标到达与路径效率方面具备SOTA表现。

**⚠️ 局限性**

局限性在于未对未来视觉状态进行显式预测，缺乏基于动作的世界模型来预估执行可行性，未来工作需结合空间思辨与世界模型以实现更完善的前瞻性规划。

---

## 436. When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions

**arXiv ID:** 2608.04820 | [PDF](https://arxiv.org/pdf/2608.04820v1)

**作者:** Feng Ding `[一作]` (Nanchang University), Mengyao Xiao `[通讯]` (Harbin Institute Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 ReSem‑Face 的多参考语义增强扩散框架，用于在大范围遮挡下进行身份保持且可控的面部修复。

**💡 创新点**

核心创新在于：①设计了 Identity‑Aware Semantic Pre‑Inpainting 模块，先在干净特征空间中预测缺失区域的高层语义（几何与纹理）；②通过 Reference Semantic Attention (ReSemAttn) 将该语义先验直接注入扩散 U‑Net，形成与文本交叉注意力和 Reference Identity Attention 并行的三路条件；③采用多参考身份记忆与跨图注意力，进一步提升身份一致性。

**🔧 技术方法**

技术手段包括：扩散模型（Latent Diffusion Model）、CLIP 视觉与文本编码、Transformer 交叉注意力、多头注意力、语义先验注入、身份一致性损失、教师蒸馏、DDIM 采样。

**📊 数据集**

在 CelebAHQ‑IDI‑5（包含 1,963 个人 5 张参考图）和 VGGFace2（3.3M 张 9,131 个人）上进行训练与评测，使用了多种遮挡类型（低脸、眼眉、全脸、随机）进行测试。

**📈 对比分析**

与 8 大基线（LDI、Custom Diffusion、Textual Inversion、ReF‑LDM、PVA、TransRef、OmniGen、HiFi‑Inpaint）进行对比，ReSem‑Face 在身份相似度（ID）上最高（0.766/0.672），FID/LPIPS 也最低；在文本控制实验中，CLIPScore、ImageReward、Attr‑Acc 也均位居前列；用户研究显示在身份保持、文本对齐、视觉真实度三项评分均显著优于对手。

**⚠️ 局限性**

局限性包括：仍需多张高质量参考图以获得最佳身份一致性；在极高自由度文本引导（高文本提示强度）时，身份与文本的冲突可能导致身份漂移；模型训练与推理仍耗时，尤其是多参考注意力计算开销较大。

---

## 437. OneDayAgent: Towards a Long-Horizon Harness for Autonomous Agents

**arXiv ID:** 2608.05013 | [PDF](https://arxiv.org/pdf/2608.05013v1)

**作者:** Jingsheng Zheng `[一作]` (Zhejiang University), Ningyu Zhang `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 OneDayAgent，一个长时程代理 harness，能将开放式日常请求分解为子任务，维护执行记忆并在交付前做全局验证与修复。

**💡 创新点**

创新点在于将任务分解、执行记忆压缩与全局验证修复三大能力统一整合为一个可跨后端的 harness，显著提升长时程任务的可靠性。

**🔧 技术方法**

技术包括基于 ReAct 的 LLM 推理循环、统一工具接口、上下文压缩与子任务状态检查/修复机制，以及多模态数据处理。

**📊 数据集**

使用 AgentIF-OneDay 基准，包含 104 个跨工作、学习和生活场景的日常任务。

**📈 对比分析**

通过与官方基准、Codex 以及五种不同后端 LLM（Gemini、Qwen 等）比较，GLM‑5.2 后端在该基准上获得最高 0.821 分，展示了跨后端的稳定性能提升。

**⚠️ 局限性**

局限性包括对单一基准的泛化有限、不同后端模型仍导致执行风格差异，以及缺乏完善的安全隔离与多任务并行处理能力。

---

## 438. Towards Physics of Multimodal Pretraining: Knowledge Flow, Modality Synergy, Early Unification, and Recipes

**arXiv ID:** 2608.05000 | [PDF](https://arxiv.org/pdf/2608.05000v1)

**作者:** Junlin Han `[一作]` (FAIR, Meta), Mike Lewis `[通讯]` (FAIR, Meta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性研究统一多模态预训练的“物理”机制，涵盖知识流、模态协同、早期统一与数据/架构/时间策略，并在大规模（1–2 T tokens）和中等规模（100 B tokens）下进行对比实验。

**💡 创新点**

创新点包括：① 发现语言对视觉理解与生成的普适提升，而视觉理解对生成的强正向影响，生成对其他模态的转移极限；② 用共享注意力/归一化、分离FFN的 Transformer 设计（MoE）最大化协同并抑制竞争；③ 揭示“视觉懒惰”现象并证明早期联合训练可消除该缺陷；④ 结合上述发现提出异构数据混合比例（L70/U25/G5）与早期联合+MoE 的高效预训练配方。

**🔧 技术方法**

技术手段：Transfusion 统一 Transformer（可同时处理文本的 next-token 预测与视觉的连续流匹配或离散自回归生成）；Mixture-of-Experts（1.5 B active 参数、256 专家、两专用模态专家）；共享/分离注意力、归一化、FFN 的可配置架构；多阶段/顺序/并行训练实验；多指标评估（VQA、GenEval、CLIP similarity、Diffusion loss、FID）及可解释性分析。

**📊 数据集**

数据集：语言：DCLM；视觉-文本：Shutterstock-Image（≈3.5 亿对）；自制 CLEVR 合成基准；VQA 评测集（GQA、MMBench、MMVP 等）；大规模训练集 1–2 T tokens（约 50 B 语言 + 50 B 视觉）。

**📈 对比分析**

方法对比：与平衡数据混合、dense 3.5 B、late‑fusion 等基线；使用语言下游精度、VQA 多维度平均、GenEval、CLIP-sim、Diffusion loss、FID 等指标。结果显示：异构配方提升语言与理解平均 1–2 % 以上，生成质量相当甚至略优；MoE 13.5 B 在 2 T tokens 下优于 3.5 B dense 及 late‑fusion，表明早期联合+架构改进带来显著性能提升。

**⚠️ 局限性**

局限性：仅在文本‑图像对上验证，未覆盖视频/音频/3D 等高维模态；大规模训练仍需高算力；对生成任务对语言先验依赖的机理未深入；对真实世界分布的因果解释仍有限；模型在更复杂的多模态推理（如视频问答、跨模态生成）中的泛化需进一步验证。

---

## 439. ORACLE: A Multi-Objective Reinforcement Learning-Based Analog Circuit Design Optimizer with Large Language Models-Guided Exploration

**arXiv ID:** 2608.04999 | [PDF](https://arxiv.org/pdf/2608.04999v1)

**作者:** Osei Brempong `[一作]` (University of Utah), Morteza Fayazi `[通讯]` (University of Utah)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于多目标强化学习的模拟电路设计优化框架ORACLE，并引入大语言模型辅助的动作筛选机制。

**💡 创新点**

创新点包括：①用向量化奖励替代标量奖励，保留各目标信息；②通过偏好向量实现单模型可调多折衷解；③设计归一化权重与余弦对齐两种动作选择策略；④使用LLM对不利动作进行掩码，提升搜索效率。

**🔧 技术方法**

主要技术为：多目标DDQN（MO‑DDQN）+偏好条件；归一化奖励与向量化Q值；归一化权重与余弦相似度动作选择；LLM（Llama 3.2）动作掩码；经验回放与ε‑贪婪探索。

**📊 数据集**

在45nm BSIM技术节点下的两阶段OPAMP、三阶段非传统OTA共2000个目标规范的基准集进行实验。

**📈 对比分析**

与AutoCKT、MODEBI、ABCMOBO等SOTA方法对比，ORACLE在通过10个偏好向量生成10个解的情况下，通过率达99.9%，FoM平均提升至5.1–318.6倍，运行时间缩短20.4–104.4倍，Pareto前沿覆盖与稀疏度均优于传统方法。

**⚠️ 局限性**

局限性在于：①仍需对每个电路拓扑训练一次；②对LLM的动作掩码依赖于语言模型的准确性；③在极大搜索空间下仍可能出现局部最优；④实验主要集中在模拟电路，未验证在更大规模或更复杂拓扑上的可扩展性。

---

## 440. RAC: Reference-Aware Activation Compression for Communication-Efficient Split LLM Inference

**arXiv ID:** 2608.04991 | [PDF](https://arxiv.org/pdf/2608.04991v1)

**作者:** Guotao Yang `[一作]` (Tianjin University), Keqiu Li `[通讯]` (Tianjin University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于参考的边界压缩码（RAC），用于边缘–云–边缘的分割LLM推理，显著减少激活量并降低通信延迟；

**💡 创新点**

创新点在于利用历史token跨度、同轮跨层匹配与连续解码预测构造阶段特定参考，结合分组仿射对齐和校准残差量化，兼顾质量与通信成本；

**🔧 技术方法**

采用分组仿射对齐、残差量化（4/8位）、可选的稀疏异常值处理、预取与预测模型、离线校准与在线重构等技术；

**📊 数据集**

在GLM-4-9B、Qwen-30B、Llama-3.3-70B模型上，使用WikiText-2、HellaSwag、GSM8K、MATH-500等评测集；

**📈 对比分析**

与Raw、TopK(8/4比特)、全局INT8/4等基线对比，RAC将激活量压缩至约27.8%/25%，在9个模型–链路组合中TTFT和TPOT平均分别提升1.24–2.72×和1.01–2.79×，且12项非困惑度指标变化仅在±2.5分以内；

**⚠️ 局限性**

局限性包括对预取与预测的准确性依赖、在高带宽场景下可能失效、对多轮交互的尾部性能仍受网络RTT制约、以及在某些模型/任务中仍存在一定的质量下降。

---

## 441. Protoreasoning in Tiny Transformers

**arXiv ID:** 2608.04980 | [PDF](https://arxiv.org/pdf/2608.04980v1)

**作者:** Eduardo Valle `[一作]` (Fin AI Research), Fergal Reid `[通讯]` (Fin AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在极小的Transformer模型上引入了原始的逐步推理（protoreasoning）并在Dyck语言的两项任务上评估其效果

**💡 创新点**

提出了一种可解释、可验证的步骤化推理轨迹（protoreasoning trace），并证明其显著提升了模型在分布外的泛化能力

**🔧 技术方法**

使用了小型Llama‑2 1M参数模型、SkipAlign位置编码增强、step‑dropout随机跳步以及两种Dyck语言任务（最深路径和最大叶子兄弟组）

**📊 数据集**

利用Dyck‑k语言生成的合成句子（如k=4、长度32的括号对），并在不同结构参数和长度上划分训练/验证/测试集

**📈 对比分析**

对比了普通样本（无推理轨迹）与包含推理轨迹的样本，发现轨迹格式在所有hold‑out模式下（尤其是插值）均能提升验证和测试的输出有效率，单任务下达到95%+，多任务下也明显优于无轨迹；但在极端外推模式下仍表现不足

**⚠️ 局限性**

限制在于：（1）模型仍难以实现真正的跨任务、跨结构的泛化；（2）外推（extrapolation）性能不佳；（3）缺乏对推理轨迹可组合性或模块化的探索；（4）仅针对Dyck语言，需进一步验证到更自然语言或其他任务的适用性

---

## 442. SciCode-Verified: How Benchmark Defects Underestimated the Scientific-Coding Ability of Language Models

**arXiv ID:** 2608.04975 | [PDF](https://arxiv.org/pdf/2608.04975v1)

**作者:** Sihan Hu `[一作]` (Hefei National Laboratory), Kun Chen `[通讯]` (Institute of Theoretical Physics)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对SciCode科学编码基准进行全面审计，发现并修复了262个缺陷，排除了一道不可验证的问题，发布了改进后的标准；

**💡 创新点**

首次揭示基准缺陷导致模型性能被低估，并系统性地重构了规范与判分机制，显著恢复模型的真实能力；

**🔧 技术方法**

采用领域专家手工审查、差异比对、两环境交叉判分与随机数独立验证等技术手段；

**📊 数据集**

使用SciCode原始的65道测试问题（共80道，15道开发，65道测试）进行评估；

**📈 对比分析**

在12个前沿模型上重新评估，子问题准确率从45–60%提升至84–98%，主问题准确率从9–27%提升至69–92%，显示模型能力远高于原始基准；

**⚠️ 局限性**

局限性包括仅覆盖测试集、评估仅为pass@1且受环境版本差异影响、审计未进行外部盲评、对长周期工作流的评估不足。

---

## 443. AsymSpec: Efficient Cloud-Edge Speculative Decoding over Asymmetric Networks

**arXiv ID:** 2608.04974 | [PDF](https://arxiv.org/pdf/2608.04974v1)

**作者:** Guotao Yang `[一作]` (Tianjin University), Keqiu Li `[通讯]` (Tianjin University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在云–边缘推理场景中提出 AsymSpec 系统，结合非对称验证协议、残差 TV 证书驱动的分阶段纠错和已确认前缀流水线，显著提升大语言模型的输出吞吐量。

**💡 创新点**

创新点：①只在上传时携带接受所需的 token–概率对，拒绝时才在下行提供精细纠错信息；②使用残差 TV 证书动态扩展 top‑K 支持，按需切换到 exact recovery；③采用已确认前缀（confirmed‑prefix）流水线与解耦批处理，消除同请求跑前导致的无效工作。

**🔧 技术方法**

技术：非对称验证协议、残差 TV 证书、top‑K 纠错、基于提议的 exact recovery、已确认前缀流水线、异步请求解耦批处理。

**📊 数据集**

数据集：GSM8K（数学推理）和 HumanEval（代码生成）。

**📈 对比分析**

与 Standard Spec、CoSine、PipeInfer 等基线比较，AsymSpec 在 3 对模型（Qwen3、GLM‑4、Llama 3.1）、2 个工作负载、3 种异步网络配置下，输出 token 通过率提升 2.82–28.03 倍，几乎不随网络弱化而下降。

**⚠️ 局限性**

限制：需要前后端均部署草稿/目标模型；top‑K 支持的大小需调优，过大或过小都会影响效率；极低下行速率或极大词表场景下完整纠错仍会产生高延迟；系统假设请求之间独立性，跨请求依赖不受支持。

---

## 444. EvolveNet: Collaborative Harness Evolution for Agent Self-Improvement

**arXiv ID:** 2608.04968 | [PDF](https://arxiv.org/pdf/2608.04968v1)

**作者:** Jun Nie `[一作]` (Hong Kong Baptist University), Bo Han `[通讯]` (Hong Kong Baptist University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 EvolveNet，一个协作型 harness 进化框架，广播共享的 LLM 代理 harness 给多方数据本地客户端，客户端独立演化并返回改动，服务器聚合改动生成新的共享 harness。

**💡 创新点**

创新点在于把经验提取从中心化优化器迁移到数据本地部署，并聚合可执行程序改动而非原始工作负载；通过证据指导、作用域类型化的程序聚合，对改动进行全局或域限定并通过逐项行为门控验证，证明多方经验可以累积。

**🔧 技术方法**

使用冻结的 LLM（deepseek‑v4‑flash 或 MiMo‑V2.5）作为后端，利用 LLM 生成器根据执行轨迹编辑 harness；引入差分、机制分类、作用域归类、行为门控（按项翻转计数）等技术；配合 LLM 驱动的 diff、机制提取与聚合流程。

**📊 数据集**

实验覆盖五个设置：文本到 SQL 的 BIRD、数据科学编码 DS‑1000、软件工程 SWE‑bench Verified、代理工作流 ClawEval 与 LiveCodeBench，分别在不同域划分的客户端上进行。

**📈 对比分析**

与未进化 harness、选最佳客户端、分派（delegation）、仅全局聚合（global‑only）以及集中式演化基线进行对比。EvolveNet 在所有五个基准上提升 13–33 点，显著优于基线，保留约 90% 的客户端提升；门控保证轨迹非递减；在规模化实验中展示了多客户端并行搜索带来的串行深度缩短。

**⚠️ 局限性**

局限性包括：数据本地不等于隐私，依赖标注验证数据；门控需要可观测的 dispatch key；程序体积随轮次增长；实验仅在 T≤3、K≤7；聚合规则单一快照；可能因程序改动泄露客户端信息；在更长周期或更大规模上未验证。

---

## 445. WorldCycle: Self-Verifiable Reinforcement Learning for Long-Horizon Video World Models

**arXiv ID:** 2608.04964 | [PDF](https://arxiv.org/pdf/2608.04964v1)

**作者:** Bohai Gu `[一作]` (Hong Kong University of Science and Technology), Song Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了WorldCycle框架，利用可逆动作循环实现自我可验证的长期视频世界模型训练。

**💡 创新点**

创新点在于将可逆动作周期转化为轨迹级、无标注的监督信号，设计空间闭合和时间一致性两种奖励，解决了长期累积误差和组合动作泛化问题。

**🔧 技术方法**

采用强化学习自监督，DiffusionNFT式负值回归目标，结合密集视觉对应点匹配的空间/时间奖励。

**📊 数据集**

使用CycleBench基准和WorldPlay训练数据，采样约4000张图像–字幕对进行后训练。

**📈 对比分析**

与Lingbot World v2、WorldPlay和WorldCompass对比，WorldCycle在ESC、RPS、RCS上分别提升32-44%并在组合动作上提升4倍，且保持甚至提升视觉质量。

**⚠️ 局限性**

局限在于只针对完全可逆动作，其他非可逆或部分可逆控制场景未验证，并且奖励设计仍需手动调参。

---

## 446. Towards Valid B-Rep Generation: Training-Free Wireframe Anomaly Detection and Repair

**arXiv ID:** 2608.04955 | [PDF](https://arxiv.org/pdf/2608.04955v1)

**作者:** Jingyu Wu `[一作]` (University of Science and Technology of China), Ligang Liu `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了无训练的插件式框架WDR，在多阶段B-Rep生成的中间线框阶段检测并修复几何与拓扑风险，从而提升最终B-Rep的有效性。

**💡 创新点**

在不修改生成器权重的前提下，引入三模态异常检测器GTAD（VLM、几何能量、拓扑约束）以及能量引导的局部重采样/扩散引导修复EGGTR，实现了针对性、可插拔的风险检测与修复。

**🔧 技术方法**

使用VLM（Qwen3.5-Flash）进行粗筛，离散切线点能量与拓扑违约计数构造能量函数，以及基于能量的局部候选重采样和训练‑free扩散引导（TFG）。

**📊 数据集**

在DeepCAD、ABC和Furniture三大数据集上进行无条件和条件生成、点云重建的评估。

**📈 对比分析**

与DTGBrepGen、Stitch‑A‑Shape、BrepForge等现有多阶段生成器在同一评测基准下对比，WDR在无条件生成中将Valid提升10.9–26.9个百分点，类条件生成提升约9个百分点，点云重建Valid提升2.2个百分点，且基本保持多样性与分布质量。

**⚠️ 局限性**

仅在存在暴露线框的管道中可用，需额外推理开销，检测误判或无法恢复的损坏仍会导致失败，且未保证最终模型的可制造性或功能正确性。

---

## 447. State2State: Environment-Derived Mid-Training for LLM Agents

**arXiv ID:** 2608.04934 | [PDF](https://arxiv.org/pdf/2608.04934v1)

**作者:** Xuanyu Lei `[一作]` (Tsinghua University), Yang Liu `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究环境学习中 LLM 代理的中间训练阶段，提出 State2State 方法通过环境探索生成可验证的状态到达任务，让代理在无专家演示或任务定义的情况下学习环境感知与操作能力。

**💡 创新点**

创新点在于把可达环境状态作为自监督训练目标并用规则匹配奖励，突破传统需外部任务和奖励的瓶颈，提供可扩展、可验证的中间训练范式，并显示跨环境迁移潜力。

**🔧 技术方法**

采用随机探索收集状态，构造状态匹配任务，用 GRPO 与动态采样进行 RL 训练，随后作为下游人类任务 RL 的初始化；评估在 ALFWorld、ScienceWorld 以及 MobileWorld GUI。

**📊 数据集**

使用 ALFWorld、ScienceWorld 和 MobileWorld GUI 三个文本/图形交互环境作为实验数据集。

**📈 对比分析**

与 SFT、Agent Early Experience、RL 以及更强 RL 基础（GRPO、GiGPO）对比，State2State 在大部分设置下提升了基准模型的单独性能，并且作为中间训练可进一步提升下游 RL 的最终表现和学习效率，尤其在 ID 与 OOD 上均有显著提升。

**⚠️ 局限性**

仅在可复现状态和可匹配奖励的环境中适用；实验仅覆盖 Qwen3-4B/8B 等中等规模模型，未验证更大规模模型；对视觉/复杂网页/真实设备等环境的推广尚待探索。

---

## 448. Mimir: A Neuro-Symbolic Memory System with Dynamic Grounding for Embodied Agents in Interactive Environments

**arXiv ID:** 2608.04933 | [PDF](https://arxiv.org/pdf/2608.04933v1)

**作者:** Haoming Xu `[一作]` (PrimeBot Research Institute), Hao Dong `[通讯]` (PrimeBot Research Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种名为 Mimir 的神经符号记忆系统，用于长周期具身任务，通过区分世界记忆和任务记忆并在每步动作前进行动态对齐来支持决策。

**💡 创新点**

创新点在于将世界观与任务进度明确拆分，并在每次执行前动态将当前目标与召回的世界证据绑定，形成可执行的决策状态，从而实现对长期任务的更可靠规划。

**🔧 技术方法**

技术包括多模态大模型（VLM+LLM）做规划与动作生成、树形结构的 World Memory、顺序目标列表的 Task Memory、动态 grounding 机制以及交互反馈闭环。

**📊 数据集**

使用 EmbodiedBench 数据集中的 EB-ALFRED 与 EB-Habitat 环境（含长周期子集）进行实验评估。

**📈 对比分析**

通过与多种基线（RoboMemory、Voyager 等）和闭源模型对比，Mimir 在 EB-ALFRED/EB-Habitat 上平均提升约 23% 的成功率，单一 backbone 可达 42.5% 的最大提升；在 EB-Habitat 长周期子集上实现 86% 成功率，超过所有闭源基线。

**⚠️ 局限性**

局限性在于仍受制于语言歧义和视觉信息不足，无法识别不可见或不可辨认的目标；此外，对模糊指令的鲁棒性和对极端遮挡情形的处理仍有待改进。

---

## 449. PRIMAL3: Pathfinding via Reinforcement and Imitation Multi-Agent Learning - Leveraging LaCAM3

**arXiv ID:** 2608.04905 | [PDF](https://arxiv.org/pdf/2608.04905v1)

**作者:** Chengyang He `[一作]` (National University of Singapore), Guillaume Sartoretti `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 PRIMAL3 框架，结合拓扑感知双图通信、LaCAM3 引导的置信度增强和优先级感知 PIBT 行动细化，实现对大规模多智能体路径规划的高效协调。

**💡 创新点**

创新点：① 双图结构区分同向跟随与冲突并采用多跳聚合与注意力聚合；② 手工设计拓扑节点特征（割点、死胡同、堵塞估计）显式提供结构约束；③ 以策略熵为阈值触发 LaCAM3 专家干预并提供 label‑smoothed 行为克隆；④ PIBT 中加入持久优先级、学习优先级和距离感知，保持回退策略与学习策略一致。

**🔧 技术方法**

技术：A* 参考路径、图神经网络双分支通信、门控多跳消息聚合、注意力聚合、熵阈值检测、行为克隆、PIBT 冲突解决、深度强化学习与行为克隆混合训练、Adam 优化器、GPU 并行。

**📊 数据集**

数据集：随机地图（障碍密度≈0.3）、迷宫地图（障碍密度≈0.5）、大规模城市级地图（100–100,000 代理）。

**📈 对比分析**

与传统搜索器（LaCAM、LaCAM3、LNS2）以及学习基线（MAPF‑GPT、SYLPH、HMAGAT）对比；PRIMAL3 在随机/迷宫地图上成功率最高或接近传统搜索器；在 10,000 代理的超大规模实例中成功率保持 95–99%，显著优于 HMAGAT；在极端拥挤迷宫下仍保持高于其他学习基线。

**⚠️ 局限性**

限制：① 仍需手工设计拓扑特征，难以直接从原始观测提取；② 训练阶段频繁调用 LaCAM3，计算开销大；③ 专家干预仅以单步动作为目标，未充分利用专家完整轨迹的长期交互信息。

---

## 450. The Beginning of ChatGPT Ads

**arXiv ID:** 2608.05008 | [PDF](https://arxiv.org/pdf/2608.05008v1)

**作者:** Emma Lurie `[一作]` (University of Pennsylvania), Danaé Metaxa `[通讯]` (University of Pennsylvania)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过构建91个按种族和收入层级划分的sock puppet账户，系统性收集ChatGPT在美国用户界面上展示的广告，完成了首个关于LLM广告投放的经验性研究。

**💡 创新点**

创新点在于：①首次对生成式AI聊天机器人广告进行大规模审计；②采用基于地理代理和定位提示的多维度人口学分层设计；③公开构建了可搜索的广告库，为后续监测提供基线。

**🔧 技术方法**

采用sock puppet审计技术，结合住宅代理、地理定位提示以及标准化每日提问电池，配合自动化浏览器脚本和HTML标签识别来捕获广告。

**📊 数据集**

使用的主要数据集包括：335条自然语言提问、91个模拟账户生成的对话记录、3,000+条广告（来自186家广告主）、以及收录在公共广告库中的完整截图与HTML快照。

**📈 对比分析**

通过对广告曝光概率与账户中位收入的逻辑回归、对收入与曝光率的密度估计、以及按主题/群组的广告率比较，发现低收入账户曝光率显著高于高收入账户；种族因素无显著差异，广告内容主要集中于消费品与软件。

**⚠️ 局限性**

局限性包括样本规模有限（仅91个账户）、研究周期短暂（受账号被检测为无效影响），缺乏真实用户行为与长期历史；地理代理成本高且易被平台检测；以及仅涵盖美国ChatGPT的初期广告阶段，难以推广到其他市场或后期系统。

---

## 451. DelusionEval: Measuring Delusion-Linked Behaviors in AI Chatbots

**arXiv ID:** 2608.05004 | [PDF](https://arxiv.org/pdf/2608.05004v1)

**作者:** Jared Moore `[一作]` (Stanford University), Desmond C. Ong `[通讯]` (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套名为DelusionEval的评估协议，用以检测大型语言模型在与用户对话中出现的与妄想相关的危险行为；

**💡 创新点**

创新点在于基于真实用户记录（包含心理伤害案例）的多轮对话样本，构建了面向妄想循环的行为编码体系，并将这些对话作为评估刺激，填补了以往模拟或单轮评估的空白；

**🔧 技术方法**

技术上采用LLM‑as‑a‑judge（Prompt+评分）进行行为判定，利用可插拔的推理层（默认/高推理）和不同上下文长度进行实验；

**📊 数据集**

使用了来自18名报告过心理伤害的用户的12,591条聊天记录，抽取了589条独特会话历史，构成677个基于行为码的对话窗口；

**📈 对比分析**

将模型在这些窗口上的行为出现率与原始对话中的表现进行比较，结果显示大多数LLM在妄想、奉承、关系及助长危害等类别中仍出现显著比例的危险行为，模型规模、发布时间或推理能力对结果影响不大，增加上下文长度则会提升部分危害行为的发生率；

**⚠️ 局限性**

局限包括样本规模仅18位用户、未覆盖更广泛的心理健康危害、评估仅为静态回放且未考虑实时记忆/检索等系统特性，且数据脱敏过程可能引入评估指示。

---

## 452. Promptable Animal Pose Tracking Across Species

**arXiv ID:** 2608.04995 | [PDF](https://arxiv.org/pdf/2608.04995v1)

**作者:** Le Li `[一作]` (University of Glasgow), Nicolas Pugeault `[通讯]` (University of Glasgow)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于视觉基础模型的可提示动物关键点跟踪框架，包含监督和无监督两条路线，可在单帧标注下追踪任意关键点；

**💡 创新点**

创新点在于：①将大规模视觉基础模型（如Diffusion Hyperfeatures、DINOv3等）直接用于跨帧关键点对应；②提出可注入结构先验的关键点提示编码器；③开发无监督漂移校正模块，实现训练‑free跨物种跟踪；

**🔧 技术方法**

主要技术包括：视觉基础模型特征提取、关键点热图编码、可注入注意力的特征调制、粗细匹配器、漂移校正与边界框约束；

**📊 数据集**

使用两个公开动物跟踪数据集：APTv2（30种动物，15帧）和TigDog（虎、马，长度可变）；

**📈 对比分析**

与现有基于人类姿态或全监督点跟踪器（如AllTracker、CoTracker3、TAPIR）对比，在无监督模式下PCK@0.1_img≈86–88，虽略逊于专门训练的点跟踪器，但在监督模式下PCK@0.1_img可达≈99，显著优于传统人类姿态基准；

**⚠️ 局限性**

局限性在于：监督模式在新物种或极端遮挡下仍易失效；无监督模式在细节关节（尤其是四肢末端）和多物种混合场景中精度不如监督模式；未来需改进对应与匹配策略以进一步提升性能。

---

## 453. SVI-DAG: A Structured Variational Inference Approach to Bayesian Causal Discovery

**arXiv ID:** 2608.04930 | [PDF](https://arxiv.org/pdf/2608.04930v1)

**作者:** Shrenik Zinage `[一作]` `[通讯]` (Massachusetts Institute of Technology), Shrenik Zinage (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种基于结构化变分推断（SVI）的贝叶斯因果发现方法（SVI‑DAG），能够利用观测数据与先验知识，直接对因果 DAG 的后验分布进行近似。

**💡 创新点**

创新点在于：① 通过条件正则化流（normalizing flow）对边之间的依赖建模，获得多模态、可表达的后验；② 引入基于 Beta‑Bernoulli 的域知识先验，使先验在置信度上可调；③ 采用 Stein 变分梯度下降（SVGD）更新节点势函数，以在 acyclicity 空间中实现模式覆盖，抑制 ELBO 的模式寻踪倾向。

**🔧 技术方法**

主要技术包括：可微分 DAG 搜索（NOTEARS / DAGMA 核心约束）、Gumbel‑Softmax 离散化、正则化流（如神经样条流）、SVGD、KL 散度闭式计算、直通（straight‑through）梯度技巧。

**📊 数据集**

使用的数据集：
- 2 变量的人工 ANM 生成数据（用于验证先验效果）；
- 线性与非线性 Erdős–Rényi DAG（25/50 变量、不同边数）作为大规模合成数据；
- 真实生物医学数据 Sachs（血液细胞因子）和 Flow Cytometry（7466 样本、11 变量）。

**📈 对比分析**

与 5 组 SOTA 贝叶斯 DAG 学习方法（如 BayesDAG、ProDAG、GFlowNets 等）在 Brier、SHD、F1、AUROC 等指标上进行对比。结果显示：
- 在合成数据上，SVI‑DAG 在 Brier 分数和 AUROC 上均优于对照组，结构精度（SHD/F1）保持竞争力；
- 在 Sachs 数据上，SVI‑DAG 在 CPDAG 级别的 AUROC 最高，Brier 分数最低，说明其不确定性量化更好；
- 计算时间与对手相近，且对 GPU 资源需求较低。

**⚠️ 局限性**

局限性：
- 结果高度依赖所选的正则化流（本文仅使用神经样条流），不同流可能导致性能变化；
- 直通梯度与 SVGD 的近似可能导致偏差，特别是在边界或高维节点势空间；
- 对大规模 DAG（>100 变量）仍存在可扩展性挑战；
- 先验构造需用户提供先验概率，若先验设定不当仍可能影响学习。

---

## 454. LLM-Assisted Detection and Repair of Hardware Security Vulnerabilities in Verilog Designs

**arXiv ID:** 2608.04907 | [PDF](https://arxiv.org/pdf/2608.04907v1)

**作者:** Ethen Santana `[一作]` (Iowa State University), Hao Zheng `[通讯]` (University of South Florida)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种利用大型语言模型（LLM）对Verilog RTL设计进行CWE驱动的漏洞检测、测试平台生成与自动修复的完整方法；

**💡 创新点**

将模块分类、资产识别、程序依赖图分析与CWE指导审查相结合，形成可迭代的漏洞识别与修复流程，首次实现LLM自动生成测试平台并进行修复；

**🔧 技术方法**

使用Microsoft Copilot作为LLM，结合Verilog HDL、CWE条目指南、程序依赖图（PDG）、资产识别、测试平台自动生成、仿真与修复循环；

**📊 数据集**

使用32个单模块Verilog设计，其中27含有相应CWE，5无漏洞；设计来自先前研究和作者构造；

**📈 对比分析**

通过在各阶段（分类、资产识别、图分析、漏洞检测、测试生成、修复）对LLM输出进行人工评估，最终以仿真通过率84%衡量性能；检测准确率高，但测试生成与修复存在明显缺陷；

**⚠️ 局限性**

主要局限包括测试平台生成不可靠导致误判和失败；LLM易出现幻觉，误判非漏洞；缺乏设计上下文导致修复偏离原功能；对复杂设计的适用性有限；需要进一步改进提示、上下文信息与验证流程。

---

## 455. DreamWAM: Beyond RGB Future Prediction for World Action Models

**arXiv ID:** 2608.04996 | [PDF](https://arxiv.org/pdf/2608.04996v1)

**作者:** Shanglin Yuan `[一作]` (Huazhong University of Science and Technology), Xinggang Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种在训练阶段对未来状态进行结构化监督的世界动作模型 DreamWAM，使其在 RGB 之外学习运动、几何和语义视角的未来表征，并在推理时仍保持仅使用 RGB 的部署方式。

**💡 创新点**

核心创新在于将未来状态拆解为四个互补视角（外观、运动、几何、语义），并分别采用联合稀释（RGB+运动）和门控残差分支（几何+语义）在预训练 VideoDiT 上进行监督；这种设计既保留了预训练的 RGB 生成能力，又让模型内部内化与动作相关的状态变化。

**🔧 技术方法**

技术包括：VideoDiT‑ActionDiT 双专家架构、RAFT 光流与 Wan2.2 VAE 的联合稀释、Depth‑Anything‑V3 与 DINOv2 提取的几何/语义特征、门控残差分支、共享注意力机制以及基于 diffusion 的动作稀释。

**📊 数据集**

主要在 LIBERO 及其视觉扰动扩展版 LIBERO‑Plus 的模拟任务上进行训练和评估，另外在 AgileX PiPER 双臂平台上对四个桌面操作任务及其光照、背景、布局等扰动进行真实机器人实验。

**📈 对比分析**

与 Fast‑WAM（RGB‑only）以及其他 VLA/WAM 基线相比，DreamWAM 在无滚动推理下从 97.30% 提升至 98.40%，在联合推理下从 98.00% 提升至 98.90%；在 LIBERO‑Plus 下分别从 51.36%→63.44% 和 69.16%→75.47%；在真实机器人上从 55.6%→74.4%。这些提升在视觉扰动下尤为显著，证明了结构化未来监督的鲁棒性。

**⚠️ 局限性**

局限性包括：仍需在训练阶段依赖外部光流、深度和语义教师，推理时必须剔除这些通道，可能导致训练与部署之间的分离难以完全避免；此外模型主要提升的是视觉鲁棒性，未针对更复杂的物理交互（如接触动力学）做进一步探索。

---

## 456. Exploring Cross-Reality Transitions between Projections and Head-Mounted Displays for Immersive Digital Art

**arXiv ID:** 2608.04971 | [PDF](https://arxiv.org/pdf/2608.04971v1)

**作者:** Xiangpeng Fu `[一作]` (Trinity College Dublin), Mads Haahr `[通讯]` (Trinity College Dublin)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构建一个结合大型投影、AR 与 VR 的混合沉浸式艺术装置，进行对比实验与访谈，探究在不同可视化环境之间切换时的感知线索、沉浸体验与资产类型敏感度。

**💡 创新点**

创新点在于将投影与 MR 的跨现实转场视为诊断性实验，用联合失配条件（空间偏移、色彩失配与延迟）揭示用户对不同线索的感知与影响；并提出资产感知差异化的切换设计原则；同时发布开源 Unreal Engine 插件 HUICRSync，推动该领域研究与原型开发。

**🔧 技术方法**

使用 Unreal Engine 5 搭建投影与 MR 应用，Meta Quest 3 头显，双盔顶投影仪，UDP 同步层，空间校准与对象分割技术，局部扩展与门户场景切换等。

**📊 数据集**

采用自制的沉浸式艺术场景（静态网格、骨骼网格、粒子系统），并对 24 名受试者进行实验，收集主观问卷（Presence、NASA‑TLX）与访谈文本。

**📈 对比分析**

通过配准与失配两种条件的对比，结果显示失配条件下存在感显著下降（p<.001），工作负荷显著上升（p<.001），精细交互任务完成时间延长（p<.002）。访谈揭示空间偏移对交互精度影响最大，色彩失配主要影响审美，延迟则削弱控制感。

**⚠️ 局限性**

局限包括样本性别偏差、失配条件是多线索组合无法单独归因、单一硬件/投影规模、校准漂移与穿透图像残差、单人实验未覆盖多人协作情境，且实验仅在一次性体验后收集数据，缺乏长期适应性评估。

---

## 457. Optimal Training-Time Scaling in Gradual Adaptation

**arXiv ID:** 2608.04927 | [PDF](https://arxiv.org/pdf/2608.04927v1)

**作者:** Zonghuan Xu `[一作]` (Fudan University), Krishna Harish `[通讯]` (Lawrence E. Elkins High School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究渐进适应中任务数增加时每个任务的训练时长应如何缩放，理论推导并在旋转MNIST、Yearbook等数据上验证。

**💡 创新点**

创新点是证明最优每任务训练时间随任务数 N 以 N⁻¹ 缩放，并给出极限学习曲线在小训练时长线性增长、长训练时长倒数衰减的精确描述。

**🔧 技术方法**

主要技术包括过参数化线性回归的梯度流解析、连续极限转换、矩阵指数展开与实验中的梯度下降实现。

**📊 数据集**

使用的数据集包括旋转回归模拟、旋转 MNIST、以及 1930–2013 年的 Yearbook 照片时间序列。

**📈 对比分析**

通过比较不同任务划分下的最终任务损失和平均路径损失，发现更细的任务划分需要更少的每任务训练步数；实验结果与理论 N⁻¹ 预测高度一致。

**⚠️ 局限性**

局限在于假设任务共享零损失解、Hessian 光滑常秩、线性模型，实际中非线性、有限样本、噪声等情况仍需进一步验证。

---

## 458. Consistency-Driven Co-Evolution for Self-Supervised Cross-Representation Learning

**arXiv ID:** 2608.04926 | [PDF](https://arxiv.org/pdf/2608.04926v1)

**作者:** Xuehang Guo `[一作]` (William & Mary), Qingyun Wang `[通讯]` (William & Mary)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种跨表示一致性驱动的共演框架（CoCoEvolve），通过在图表、表格和可视化代码三种表现形式之间建立一对一对应关系，利用互相一致性作为无标注的自监督信号，在训练和推理阶段共同提升模型在六个交叉任务（图表↔表格、表格↔代码、代码↔图表）的理解能力。

**💡 创新点**

创新点：①引入严格的一对一对应约束，消除传统一对多映射导致的标注稀缺与不确定性；②将一致性奖励分层设计为代码一致性、视觉一致性、格式一致性和层级奖励，形成完整的自监督训练目标；③在推理时实现两模型的共优化（Co-Optimization），进一步提升结果质量；④构建统一的多维评估套件，避免单一分数或 LLM/MLLM 评判的主观性；⑤兼容教师指导模式，可无缝切换到监督学习。

**🔧 技术方法**

技术：多模态强化学习（RL）框架（支持 GRPO、DAPO、GSPO 等），基于执行器 h 对代码进行渲染，利用 CLIP、SSIM、OCR、DINO 等视觉相似度指标；代码相似度通过嵌入余弦相似度；教师模块实现半监督奖励；自回归生成模型（LLM/MLLM）作为 fθ 与 gψ 的实现。

**📊 数据集**

数据集：ChartCoder（训练/测试），ChartMimic（多域外部测试），ChartNet（健康、财经等领域），Chart2Code（不同复杂度的任务）。

**📈 对比分析**

与基线模型（单任务 LLM/MLLM）以及传统标注监督方法对比，CoCoEvolve 在无监督训练下就能获得约 17–26% 的规则评估提升，代码执行成功率达到 100%；在多域、复杂度更高的外部测试中提升幅度高达 37–47%；在推理时使用教师指导可再提升 2–4%。

**⚠️ 局限性**

局限性：①当仅在训练阶段使用教师指导时结果不够稳定，需要在推理阶段同样启用；②对计算资源要求高（需同时训练两模型并进行多次 roll‑out）；③在极端稀疏标注环境下，一对一约束可能仍需要人工预先设定；④评估套件虽然多维，但对极端视觉差异的判定仍可能受限于现有指标。

---

## 459. A Chain Is Only as Strong as Its Weakest Link: A Scoping Review of System Integration Audits in AI

**arXiv ID:** 2608.04921 | [PDF](https://arxiv.org/pdf/2608.04921v1)

**作者:** Leah Davis `[一作]` (McGill University), AJung Moon `[通讯]` (McGill University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统化检索与筛选4,259篇文献，聚焦58篇将系统集成视为核心的AI审计研究，采用反思性主题分析，对审计的集成场景、参与角色、评估质量、输出形式及功能进行归纳与合成。

**💡 创新点**

创新点在于：①首次系统化梳理AI领域的系统集成审计，提出三大集成场景（组件间、系统-环境、多系统）；②将审计角色、评估质量、输出类型和功能分层，对现存碎片化实践进行结构化描述；③识别并强调审计实践中缺乏标准化、信息获取受限等关键痛点，为后续研究与监管提供理论与方法指引。

**🔧 技术方法**

采用的技术与方法包括：PRISMA‑ScR框架的系统综述流程、六大数据库（Scopus、IEEE Xplore、ProQuest、Web of Science、EBSCOhost、WorldCat）的检索与筛选、Covidence平台的双人独立筛选与冲突解决、ATLAS.ti软件进行数据编码与主题归纳、反思性主题分析以及叙事合成。

**📊 数据集**

所用数据集为：一共58篇符合纳入标准的文献，覆盖从2019年至2024年的期刊文章、会议论文、行业白皮书、预印本等。文献来源于全球多国，涉及不同AI应用场景。

**📈 对比分析**

本文并未进行实验或性能比较，而是通过定性分析与归纳，展示各审计在质量评估、输出类型与功能上的差异与共性。因而不存在传统意义上的“性能评估”指标。

**⚠️ 局限性**

局限性包括：①检索词与定义局限，导致可能遗漏非标准术语的系统集成审计文献；②以学术数据库为主，可能低估行业、草根或非英文工作；③受限于公开文献，缺乏对实际审计实施细节与效果的实证验证；④术语不统一，导致跨研究比较困难；⑤信息获取与权限限制在审计实践中普遍存在，未能充分探讨其对结果的影响。

---

## 460. An active-learning framework for real-time depth perception from monocular vision streams

**arXiv ID:** 2608.04917 | [PDF](https://arxiv.org/pdf/2608.04917v1)

**作者:** Xiaorong Zeng `[一作]` (Xiamen University of Technology), Shuiwen Shen `[通讯]` (Xiamen University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了基于 MobileNetV3-Small 的轻量级单目深度估计框架，并通过在线主动学习实现连续自我监督适应。

**💡 创新点**

创新点在于：① Gated Cross-scale Additive Fusion（GCAF）模块实现语义引导的去噪级联融合；② 在线主动学习（OAL）闭环（Predict–Evaluate–Correct）结合置信度筛选、回放缓冲和 Elastic Weight Consolidation（EWC）实现可选择性可塑性。

**🔧 技术方法**

使用了自监督深度+位姿网络、光度重投影+SSIM+边缘平滑损失、AdamW 在线优化、EWC 正则化、置信度门控与经验回放。

**📊 数据集**

在 KITTI（Eigen split）进行离线训练与在线测试，跨域零样本适应至 NuScenes 评估鲁棒性。

**📈 对比分析**

与 Monodepth2、Lite-Mono 等轻量级基线对比，GCAF+OAL 使 Abs Rel 下降至 0.125（比基线低约 4%），在域漂移下保持性能，计算量降低约 75%，推理速度约 388 FPS，表现优于同类轻量级模型。

**⚠️ 局限性**

局限性：受 MobileNetV3-Small 表示能力限制，在线适应提升有限；对极端环境或长期部署的尺度漂移、极端光照仍需进一步验证；模型对高频动态物体的鲁棒性仍有提升空间。

---

## 461. CheMLFlow: An Open-Source Platform for Cheminformatics and Materials Informatics Applications

**arXiv ID:** 2608.04942 | [PDF](https://arxiv.org/pdf/2608.04942v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 462. AutoCue: Multimodal LLM-Assisted Externalization of Implicit Inputs as Instructional Visual Cues in Screencast Tutorials

**arXiv ID:** 2608.04910 | [PDF](https://arxiv.org/pdf/2608.04910v1)

**作者:** Shengyang Luo `[一作]` (Purdue University), Yingjie Victor Chen `[通讯]` (Purdue University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 AutoCue，一个多模态 LLM 辅助的教程增强管道，能够从缺少输入元数据的屏幕录制教程中外化隐式鼠标和键盘输入，生成增广视频和可编辑的人工审核文件。

**💡 创新点**

创新点在于结合视觉变化、OCR 文本、配音与官方文档，使用受约束的多模态 LLM 自动推断缺失的输入信息，并提供统一的视觉提示语法与可编辑输出，从而解决教程中输入信息缺失导致的学习阻塞问题。

**🔧 技术方法**

技术包括像素差异检测、Otsu 阈值和形态学操作提取感兴趣区域、EasyOCR 提取界面文本、自动语音识别与配音对齐、受约束多模态 LLM（gpt‑5.4）推断、视觉提示渲染与视频/JSON 可编辑导出。

**📊 数据集**

使用 Autodesk Maya 的 30 个视频片段（约 45 分钟）进行技术验证，并在 24 名具有 Maya 学习经验的用户中开展实验，对单一 10 分钟建模教程进行用户研究。

**📈 对比分析**

通过两组间实验比较原始教程与 AutoCue 增强教程，测量任务完成时间、重播/卡顿次数、主观体验等；AutoCue 在 UI 事件上达 95.1% 召回、96.9% 精确，实验中完成时间平均降低 278 秒，卡顿次数平均降低 3.5 次。

**⚠️ 局限性**

局限性包括：仅能自动推断具有显著 UI 反馈的事件，对复杂状态变更需要人工编辑；对低质量或高速播放视频易漏检；依赖可见视觉反馈，难以适用于无 UI 反馈或非 GUI 软件；需手动编辑的可编辑文件仍占一定工作量。

---

## 463. When Shared Rollouts Fail in Defensive Driving Evaluation: A NAVSIM Score Basis Audit

**arXiv ID:** 2608.04896 | [PDF](https://arxiv.org/pdf/2608.04896v1)

**作者:** Ziang Wei `[一作]` (EABOT.AI), Wei Li `[通讯]` (EABOT.AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 NAVSIM 伪仿真评分体系进行结构性审计，识别出由于共享滚动和速度重拟合数值不稳定导致的“共享失效宽恕崩溃”，并提出基于盲探针、依赖版本公开、覆盖率报告与滚动稳定性测试的审计协议。

**💡 创新点**

首次将“共享失效宽恕崩溃”概念系统化，并将盲探针作为验证防御驾驶评分是否真正区分演员感知的必要条件；同时通过数值求解器替换和堆栈对照实验定位问题根源。

**🔧 技术方法**

使用 NAVSIM v2.2 评估框架、批量 LQR 速度重拟合、伪逆求解器、有限差分双侧诊断、求解器替换（直接解、Hermitian 伪逆）、相同源控制堆栈以及精确输入诊断。

**📊 数据集**

主要基于 NAVSIM 的 navtest 数据集（12,146 个 token）以及官方 450-token 控制池和 32-token 诊断集；亦对公开的 LEAD checkpoint 进行交叉验证。

**📈 对比分析**

通过与演员感知的强对照器（如 LEAD、Hydra-MDP 等）以及两种盲探针（路线路径无视和路线路径有视）进行排名对比；在原始配置下，盲探针排名高于演员感知模型；改用稳定求解器后，盲探针被正确降序，说明评分失效被纠正。

**⚠️ 局限性**

研究仅在单一配置（x86-64 + OpenBLAS）上验证，未覆盖闭环驾驶或其他平台；无法评估该问题在不同后端或公开排行榜中的普遍性；所提审计仍为诊断工具，非完整的防御驾驶评估方案。

---

## 464. scikit-rom: An Open-Source Python Platform for Teaching and Prototyping Projection-Based Reduced-Order Modeling

**arXiv ID:** 2608.04960 | [PDF](https://arxiv.org/pdf/2608.04960v1)

**作者:** Suparno Bhattacharyya `[一作]` (Indian Institute of Technology Dhanbad), Jean C. Ragusa `[通讯]` (Texas A&M University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一个轻量级、Python原生的投影基ROM库（scikit‑ROM），实现完整的从全阶有限元求解、快照采集、POD基组构造、Galerkin投影、离线–在线分解、超减速（DEIM、S‑OPT、ECSW、ECM）到误差评估的全流程；

**💡 创新点**

创新点在于：① 将整个投影基ROM工作流拆分为可透明、可修改的六个阶段，降低教学与原型开发门槛；② 在同一框架下统一实现多种超减速策略；③ 采用模块化问题模板和注册机制，方便用户快速创建新案例；

**🔧 技术方法**

使用技术包括：有限元求解（PyFEM/自研），SVD/POD构造基组，Galerkin投影，离线–在线分解，DEIM、S‑OPT、ECSW、ECM超减速，Newton、隐式欧拉、Newmark、Rayleigh阻尼数值积分，误差指标（L2、L∞、RMSE、R²）以及NumPy/NPZ与HDF5存储；

**📊 数据集**

数据集主要是四个数值实验：1）1D非线性热传导（训练/测试参数分别为k_p、q_p），2）2D线性热传导（参数k、q），3）3D有限变形超弹性块（材料参数μ、λ），4）3D梁振动（材料参数E、ν），每个案例均使用Sobol采样生成训练与测试参数集；

**📈 对比分析**

通过对比全阶模型（FOM）、标准ROM以及不同超减速ROM，分别计算全局误差、局部误差及加速比：① 1D非线性热传导：标准ROM加速有限，ECSW/ECM/DEIM超减速后加速数十倍，误差<10⁻²；② 2D线性热传导：ROM平均加速≈15倍，误差<10⁻⁴；③ 3D超弹性：ECSW超减速后加速≈6倍，误差<10⁻³；④ 3D梁振动：ROM平均加速≈35倍，误差<10⁻³；

**⚠️ 局限性**

局限性：仅支持有限元离散的侵入式ROM，尚未实现Petrov–Galerkin、非侵入式或机器学习驱动的超减速；在单机/多线程下运行，尚未充分并行化；示例覆盖范围有限，需进一步扩展至流体、复杂几何等问题；

---

## 465. SpecRoll: Fast-Slow Verifier-Feedback Adaptation for Speculative Reinforcement Learning Rollouts

**arXiv ID:** 2608.04962 | [PDF](https://arxiv.org/pdf/2608.04962v1)

**作者:** Nhat Minh Pham `[一作]` (VNU University of Engineering and Technology, Vietnam National University), Khac-Hoai Nam Bui `[通讯]` (Viettel AI, Viettel Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SpecRoll，一种用于GRPO的精确推测式滚动加速引擎；

**💡 创新点**

创新点在于双时尺度自适应：快速无梯度的Reflex记忆纠正轨迹局部误差与慢速的持久参数更新；

**🔧 技术方法**

采用轻量级未来词头、稀疏树并发验证、Reflex机制以及可靠性门控的持久更新；

**📊 数据集**

在Qwen2.5（1.5B/3B/7B/14B）和Llama-3.1-8B模型上，使用GSM8K、SimpleRL-Abel-Level3to5和DAPO-Math-17K三个数学推理数据集；

**📈 对比分析**

与基线GRPO、FastGRPO对比，SpecRoll在15个模型-数据集组合中实现1.26×–2.15×生成速度提升，1.21×–2.04×端到端速度提升，且平均速度提升高于FastGRPO；

**⚠️ 局限性**

局限性包括仅评估至14B模型、仅覆盖数学推理任务，未在更大模型、代码或多语言推理等多样化场景中验证；

---

## 466. ContextMaster: Interactive Multi-Shot Video Creation via Fixed-Budget Sparse Context Routing

**arXiv ID:** 2608.04956 | [PDF](https://arxiv.org/pdf/2608.04956v1)

**作者:** Xu Guo `[一作]` (Tsinghua University), Xiangwang Hou `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种交互式多镜头视频创作（IMVC）框架，支持文本生成、参考驱动生成、编辑及多镜头组合；

**💡 创新点**

创新点包括：角色感知旋转坐标（Role‑aware RoPE）统一异构视觉上下文；可缓存的固定预算稀疏上下文（cacheable fixed‑budget context）与ConstraintSink在保证必需参考/源信息的同时限制读取量；以及通过全上下文教师的“特权上下文蒸馏”（privileged context distillation）训练稀疏少步模型，随后再用分布匹配精细化。

**🔧 技术方法**

技术：改进的Transformer结构、旋转位置编码、块稀疏注意力、ConstraintSink、块摘要的动态路由、两阶段蒸馏（PCD + DMD）、CFG指导与四步采样。

**📊 数据集**

使用内部约1M多镜头视频数据集，外加公开参考与编辑数据集；在T2MV、R2MV、V2MV、X2MV四个任务集上进行评估。

**📈 对比分析**

与MultiShotMaster、LongLive、ShotStream、Infinity‑RoPE、Phantom、VideoCoF、LucyEdit、StreamEdit、LiveEdit等方法对比，ContextMaster在跨镜头一致性、任务完成度上均优于基线，并实现单张GPU 16 FPS，显著提升生成速度与质量。

**⚠️ 局限性**

局限性：仍需大量训练数据；稀疏上下文在极长历史下可能导致细节缺失；模型对极端视角或复杂编辑需求的鲁棒性有限；实际部署中需要精细调节预算与蒸馏参数以兼顾速度与质量。

---

## 467. Reply, Delete, or Ignore? Examining How Content Creators Perceive and Select Comment Moderation Strategies

**arXiv ID:** 2608.04951 | [PDF](https://arxiv.org/pdf/2608.04951v1)

**作者:** Yunhee Shim `[一作]` (Rutgers University), Shagun Jhaver `[通讯]` (Rutgers University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过问卷调查研究内容创作者对删除、回复或忽略仇恨评论的三种内容治理策略的感知与选择。

**💡 创新点**

创新点在于从创作者视角探讨情绪安全、印象管理和算法收益的权衡，并发现算法收益对策略选择影响不大。

**🔧 技术方法**

采用定量问卷方法，结合描述性统计和多元回归分析。

**📊 数据集**

数据集为584名社交媒体内容创作者的问卷结果。

**📈 对比分析**

通过对三种策略的感知收益进行对比，未涉及算法性能指标，但发现删除对安全最有利，回复对算法收益最高。

**⚠️ 局限性**

局限性包括自我报告偏差、样本可能缺乏平台多样性、缺乏实验验证策略实际效果。

---

## 468. Visual Representation Matters: Exploiting Temporal Differences in Video-to-Audio Generation

**arXiv ID:** 2608.04902 | [PDF](https://arxiv.org/pdf/2608.04902v1)

**作者:** Zehua Chen `[一作]` (Tsinghua University), Jun Zhu `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了视频到音频生成，提出利用视频帧间时间差（Temporal Differences）增强视觉条件，实现无额外监督的端到端V2A模型。

**💡 创新点**

将时间差作为区分V2A与I2A的核心特征，提出Hierarchical Continual Learning（逐级学习）和Annealed Temporal Differences Guidance（逐步引导）来高效利用时间差，而不需要额外网络或监督。

**🔧 技术方法**

使用条件扩散模型（Latent Diffusion）、CLIP视觉编码器、帧级/CLIP级时间差编码、Hierarchical Continual Learning、ATDG引导机制、T2A预训练、VAE编码等技术。

**📊 数据集**

在VGGSound、AudioSet、AudioCaps、FreeSound、MSD等数据集上进行T2A预训练与V2A微调，并在VGGSound测试集上评测。

**📈 对比分析**

与多种基准V2A/T2A/VT2A模型对比，使用FAD、KL、IS、FD、IBS、AA等指标，TD‑V2A在音频质量、语义/时间一致性指标上显著优于基线，甚至逼近Diff‑Foley，并在主观评价中获得最高分。

**⚠️ 局限性**

对帧差分窗口长度k的敏感性，需要人工调参；在极端静态或噪声视频中时间差信息可能不足；目前验证仅在标准数据集上，缺乏跨域或更复杂场景的评估。

---

## 469. Towards Decentralized Searcher Competition in MEV Markets

**arXiv ID:** 2608.05011 | [PDF](https://arxiv.org/pdf/2608.05011v1)

**作者:** Roozbeh Sarenche `[一作]` (KU Leuven), Yunwen Liu `[通讯]` (KU Leuven)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

分析了以太坊MEV搜索者竞赛的集中化问题，并提出了基于Shapley值的入门过滤加燃烧的拍卖机制以实现更公平、去中心化的奖励分配。

**💡 创新点**

创新点在于将Shapley值分配与入门阈值、燃烧参数相结合，并在去中心化的MEV环境中设计了对复制代码Sybil攻击和验证者‑搜索者共谋的Bayesian安全约束，实现了公平性、效率与安全的统一权衡。

**🔧 技术方法**

采用异质搜索者模型、Shapley值与Jain/HHI指标评估公平性与集中度，并通过Bayesian安全分析给出安全参数校准；理论证明和数值模拟支撑机制的可行性。

**📊 数据集**

使用以太坊历史MEV套利交易数据（Luo等构建的577,264笔交易），根据交易策略与利润等级提取截断对数正态成本分布，构建四种不同集中度的搜索者样本。

**📈 对比分析**

将第一价格赢家淘汰（FPA）与入门过滤SCA_θ对比，利用Jain指数、有效搜索者数和理论/模拟结果显示：在集中环境下SCA能把公平度从≈0.18提升至≈0.98，集中度从≈1.04下降至≈3.78；在已分散环境下提升有限。

**⚠️ 局限性**

局限在于仅以合约层面识别搜索者，未区分同一实体下的多合约，且仅观测成功交易，无法完整捕捉竞标失败与私有流信息，影响参数估计与模型外推。

---

## 470. Stochastic Emulation using Generalized Stratified Sampling for Performance-Based Risk Optimization of Structures

**arXiv ID:** 2608.05006 | [PDF](https://arxiv.org/pdf/2608.05006v1)

**作者:** Isabela D. Rodrigues `[一作]` (University of São Paulo), André T. Beck `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在结构性能评估中结合一般化分层采样与随机多项式混沌展开构建高效的超稳态仿真器，并将其用于两层钢框架BRBF的PBRO优化，以降低非线性模型评估次数。

**💡 创新点**

创新点在于将GSS与SPCE耦合，实现对极端响应尾部的准确逼近，并通过层级重组实现概率约束的快速估计。

**🔧 技术方法**

使用的技术包括Generalized Stratified Sampling、Stochastic Polynomial Chaos Expansion、OpenSees非线性时间历程分析、遗传算法优化。

**📊 数据集**

数据集为两层BRBF有限元模型的非线性时间历程模拟结果（约10,000个训练样本、50,000个验证样本），以及由随机地震记录产生的地面运动。

**📈 对比分析**

与直接Monte Carlo和直接GSS的对比显示，GSS-SPCE将非线性模型评估次数从约4.35×10⁷降至1.0×10⁴，节省约4,350倍（相对于MC）或43.5倍（相对于GSS），并能在目标极限概率10⁻³以下准确评估概率约束。

**⚠️ 局限性**

限制在于尾部准确度依赖于分层设定和支持点数量，且对更高维设计空间或更小的目标失败概率可能需要更多分层或更大样本量；另外在梯度信息缺失的情况下仍需使用启发式优化。

---

## 471. A Tight Bound on Online Vertex Cover under Edge Arrivals

**arXiv ID:** 2608.04994 | [PDF](https://arxiv.org/pdf/2608.04994v1)

**作者:** Zhihao Gavin Tang `[一作]` (Shanghai University of Finance and Economics), Yuhao Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了在边到达的在线顶点覆盖问题中，无论是随机化还是分数化的算法，最优竞争比均为 2，完成了对该问题的完美下界证明。

**💡 创新点**

首次将 Assadi、Jiang 与 Xiang 的蓝图框架直接用于在线顶点覆盖的下界证明，获得了此前未知的 2 倍下界，填补了该模型下缺失的理论极限。

**🔧 技术方法**

使用蓝图技术、循环移位构造隐藏的独立集以及期望分析来证明任何算法必需在未揭示独立集前已分配足够权重，从而导致竞争比不小于 2。

**📊 数据集**

无实验数据集，全部为理论构造与证明。

**📈 对比分析**

与已知的 2-竞争比算法（两端点都加入）直接对比，证明该算法已达到最优，且任何改进都不可能存在。

**⚠️ 局限性**

仅适用于边到达模型和无记忆对手，未讨论有记忆对手或顶点到达场景；此外仅给出下界，缺乏对应上界或算法改进的讨论。

---

## 472. Toward Practical Decentralized Proof-of-Location via Physical Witnessing Zones

**arXiv ID:** 2608.04957 | [PDF](https://arxiv.org/pdf/2608.04957v1)

**作者:** Tamor Tomson `[一作]` (University of Tartu), Ulrich Norbisrath `[通讯]` (University of Tartu)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在室内实验室环境中实现并评估了一个基于UWB测距、局域网Mesh与BFT账本的去中心化定位证明（Proof‑of‑Location）完整原型。

**💡 创新点**

首次将理论上的 witnessing‑zone 架构落地为物理实验平台，并在此基础上提出了可适配低成本硬件的协议改进：见证者发起的双向测距、全局 RMS 残差一致性检查和区块哈希绑定的 freshness 检测。

**🔧 技术方法**

使用的技术包括 ESP32‑DWM3000 UWB 模块实现 DS‑TWR；Raspberry Pi‑4+B.A.T.M.A.N‑Adv 构建 Mesh 网络；GoQuorum+IBFT 共识维护权限账本；多方测距求解多层定位；离线证明验证脚本。

**📊 数据集**

数据集为 340 次实验（4 个内点 + 1 个外点，每点 20 次）收集的测距、时间戳、账本交易与证明文件，全部公开存储于 Zenodo，文件包含原始日志与标准化结果。

**📈 对比分析**

通过与理想模拟模型对比，本原型在 80 次正常实验中实现平均误差 0.18 m（SD 0.086 m）、RMS 0.15 m、证明延迟 0.055 s，并在模拟攻击（位置外、延迟、范围膨胀、区块回溯）中均能被正确拒绝，接受率达到 100%。

**⚠️ 局限性**

局限性包括：仅在单一室内 zone 测试，未覆盖多 Zone、多 Prover 与并发情形；缺乏完整账本包含证明与可验证签名；缺少严格的 Witness 资格注册与时间同步机制；隐私保护不足；对极端多径、室外或大规模部署的可扩展性与稳健性尚未验证。

---

## 473. UG-UMRE: Uncertainty-Guided Modality Augmentation and Distributional Calibration for Unified Multimodal Relation Extraction

**arXiv ID:** 2608.04949 | [PDF](https://arxiv.org/pdf/2608.04949v1)

**作者:** Bo Kong `[一作]` (Xinjiang University), Shengquan Liu `[通讯]` (Xinjiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出UG-UMRE网络，解决统一多模态关系抽取中的模态噪声与分布差异问题；

**💡 创新点**

①将单模态特征建模为高斯分布并通过变分信息瓶颈进行不确定性驱动去噪；②设计联合不确定性对齐（JAUA）模块实现全局分布校准；③利用自监督对比学习结合不确定性确保语义一致性；

**🔧 技术方法**

变分信息瓶颈、Gaussian不确定性建模、self‑supervised 对比学习（InfoNCE）、KL 对齐、层级跨模态交互、MMoE、多层次最优传输、t‑SNE可视化等技术；

**📊 数据集**

UMRE、MORE、MNRE 三大基准数据集，以及 Aug‑Noise 噪声挑战集；

**📈 对比分析**

与多种SOTA MRE 模型（如 MEGA、MKGFormer、IFAformer、REMOTE 等）以及多模态大模型（Qwen2‑VL‑7B、Llama‑3.2‑11B‑Vision）进行对比；UG‑UMRE 在 UMRE、MORE、MNRE 上分别达到 69.98%、66.76%、89.59% F1，提升 2–4% 以上，且在噪声鲁棒性和算力开销方面表现良好；

**⚠️ 局限性**

对极少量/长尾关系的处理仍有限；不确定性估计仅针对整体特征，缺乏动态阈值；对未知噪声、严重模态缺失的适应性需进一步研究。

---

## 474. A General Sufficient Condition for Rewriting Horn-ALCHI Atomic Queries into GQL

**arXiv ID:** 2608.04945 | [PDF](https://arxiv.org/pdf/2608.04945v1)

**作者:** David Carral `[一作]` (University of Montpellier), Quentin Manière `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种将原子查询重写为图查询语言（GQL）的充分条件，重点关注基于描述逻辑的本体介导查询（OMQs）。

**💡 创新点**

创新点在于引入了DL自动机这一新形式，能够通过对事实集的运行捕捉OMQs的语义，并识别出一大类可以重写为GQL的DL自动机。

**🔧 技术方法**

使用了DL自动机这一新形式，结合了状态分层的概念，以避免复杂性较高的循环依赖。

**📊 数据集**

使用了基于描述逻辑的本体和事实集，具体示例中涉及了关于计算机网络的信任用户的公理和事实集。

**📈 对比分析**

与传统方法相比，本文的方法通过将规则查询转化为DL自动机，再转化为正向两路正则路径查询（PQs），最终转化为GQL，性能上能够处理更复杂的查询，且保证了等价性。

**⚠️ 局限性**

限制在于当前方法仅适用于特定类型的查询，未能涵盖所有可能的查询形式，未来工作将探讨如何放宽分层条件以捕捉更广泛的查询。

---

## 475. A geometry-based deep equilibrium model for image restoration under multiplicative Gamma noise

**arXiv ID:** 2608.04944 | [PDF](https://arxiv.org/pdf/2608.04944v1)

**作者:** Shengkun Yang `[一作]` (Harbin Institute of Technology), Zhichang Guo `[通讯]` (Harbin Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于深度平衡（DEQ）框架的图像去模糊与多重伽马噪声去除方法。

**💡 创新点**

创新点在于：①用可学习的卷积滤波器与影响函数显式构造几何先验（表面积与平均曲率）作为正则化；②采用Bregman镜像梯度算法并证明在非L‑光滑、非凸情形下的全局收敛；③通过DEQ实现无限深网络，显著减少参数量。

**🔧 技术方法**

核心技术包括：深度平衡模型、Bregman镜像梯度（mirror descent）迭代、Kurdyka–Łojasiewicz（KŁ）理论、自动微分与隐式微分训练、基于RBF的影响函数与DCT基卷积参数化。

**📊 数据集**

训练数据采用DIV2K（720张高分辨率图像），测试集为常见灰度图像（Dart、SAR、Leaves）及多彩图像集（Set3C、Kodak24、BSD500、Set14）。

**📈 对比分析**

通过与多种基于模型的算法（AA、RLO、DZ、ZWN、MG）以及已有DEQ-RED模型对比，实验显示该方法在PSNR/SSIM/LPIPS上均优于或可与最先进的DEQ方法持平，同时参数量仅约600个，显著低于DEQ-RED的十万级参数。

**⚠️ 局限性**

局限性包括：①对退化模型的显式假设（A矩阵、噪声均值）在实际场景中可能不完全成立；②镜像梯度与自适应步长的求解仍需多次迭代，导致推理时间相对较长；③在极端噪声或过度模糊的情形下，恢复效果仍略逊于部分专门设计的后处理方法。

---

## 476. Reading Between the Frames: Interpreting Implicit and Non-literal Meaning in Social Media Videos

**arXiv ID:** 2608.04939 | [PDF](https://arxiv.org/pdf/2608.04939v1)

**作者:** Yang Wang `[一作]` (University of Manchester), Chenghua Lin `[通讯]` (University of Manchester)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 DrivelHub+ benchmark，用于评估短视频的隐含语用理解，提供 1000 条社交媒体视频和对应的人类隐含意义注释。

**💡 创新点**

首次将 drivelology 从文本迁移到多模态视频，并设计解释与检索两种互补评估方式。

**🔧 技术方法**

使用视频‑语言模型（如 Qwen、InternVL 等）进行自由文本解释，并采用检索‑对比学习的 embedding‑native 模型（如 LCO‑Omni）进行双向检索。

**📊 数据集**

构建 DrivelHub+ 数据集，包含 1000 条短视频及其隐含意义说明，并标注相关模态信息。

**📈 对比分析**

实验显示生成模型中 Qwen3.5‑27B 在解释任务上取得最高对齐率和总分；检索任务中 LCO‑Omni 系列表现最佳；总体性能仍远低于人类。

**⚠️ 局限性**

局限性包括数据主要以英语为主，跨语言泛化有限；存在敏感内容且需受限访问，且模型在细粒度语用推理方面仍表现不足。

---

## 477. Unleashing the Potential of Vision-Language Models for Generalizable AI-Generated Image Detection

**arXiv ID:** 2608.04935 | [PDF](https://arxiv.org/pdf/2608.04935v1)

**作者:** Weihan Cai `[一作]` (Chinese Academy of Sciences), Xinping Gao `[通讯]` (Purple Mountain Laboratories)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过在视觉-语言模型 Perception Encoder 上使用语义原型校准（SPC），实现对 AI 生成图像的高效检测。

**💡 创新点**

创新点在于将文本编码器生成的取证语义作为类别原型的起点，并在冻结特征空间中进行微调，利用语言对齐的取证结构提升线性探针性能。

**🔧 技术方法**

技术上采用冻结图像编码器、线性分类头、文本特征初始化原型、交叉熵微调以及 LDA/重叠系数评估，评测跨生成器、后处理和野外场景。

**📊 数据集**

使用 GenImage、Chameleon、WildRF、SocialRF、CommunityAI、AIGI-Holmes、AIGI-Now、Blur-and-JPEG 等公开基准数据集。

**📈 对比分析**

与 DINOv3-Linear、SigLIP2-Linear、MetaCLIP2-Linear、OpenCLIP 等基线相比，PE-SPC 在平均准确率上提升 3–6%，在 GenImage 交叉生成器和 In-the-Wild 真实场景中显著提高最低准确率，最终实现新的 state‑of‑the‑art。

**⚠️ 局限性**

局限性包括：仅在预训练时已接触大量 AI 生成图像的模型才有效；对文本提示的语义对齐要求较高；在文本与视觉表征对齐不足的模型（如 OpenCLIP、SigLIP2）可能反而退化。

---

## 478. Does Out-of-Sight Equal Out-of-Mind in CoT Monitorability?

**arXiv ID:** 2608.04928 | [PDF](https://arxiv.org/pdf/2608.04928v1)

**作者:** Pedro Ferreira `[一作]` (University of Amsterdam), Ivan Titov `[通讯]` (University of Amsterdam)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在提示干预设置下，探究从显式链式推理（explicit CoT）迁移到隐式链式推理（latent CoT）对模型可监控性的影响；通过对提示依赖（hint‑reliance）的检测，评估不同推理模式、访问层级以及可视化技术的监控效果。

**💡 创新点**

首次系统对比显式与隐式链式推理在可监控性上的差异，证明即便失去可读推理链，模型内部激活仍能保留提示依赖信号；并揭示任务类型与对内部状态的访问程度是决定可监控性的关键因素，而非推理模式本身。

**🔧 技术方法**

使用三种推理方式：显式CoT、弱监督隐式CoT（SIM‑CoT）和强监督隐式CoT（CODI）；构建多种监视器，包括精确匹配、提示式LLM、Fine‑tuned LLM、激活探测器；并通过logit lens与decoder对隐式状态进行可视化。

**📊 数据集**

采用两大任务及其在域内外的数据集：数学推理使用GSM8k（域内）与SVAMP（域外）；问答推理使用ECQA（域内）与SIQA（域外）。

**📈 对比分析**

通过AUROC评估监视器性能。实验表明：激活探测器在大多数任务与推理模式下优于基线且表现最好；对显式CoT，原生文本监视器在GSM8k上可达到高性能；隐式CoT通过logit lens或decoder可显著提升对数学推理的监控能力；在域外迁移中，数学推理保持较好性能，而问答推理性能大幅下降。

**⚠️ 局限性**

局限性：仅评估单一基础模型（Llama‑3.2‑1B‑Instruct）和两种隐式CoT实现；任务仅限数学与问答两类，可能不具备普适性；提示干预与监控目标均为单一形式（正确答案提示），且通过干预‑aware fine‑tuning人为诱导提示依赖，未涵盖自然出现的提示行为。

---

## 479. Transition Techniques for Externally-Guided Multi-Scale Viewpoint Changes

**arXiv ID:** 2608.04912 | [PDF](https://arxiv.org/pdf/2608.04912v1)

**作者:** Matt Gottsacker `[一作]` (University of Central Florida), Blair MacIntyre `[通讯]` (Northeastern University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了在XR展示情境下的外部引导多尺度视角转换技术，并评估其对空间记忆、用户体验和舒适度的影响。

**💡 创新点**

提出并验证了三种以目标视角预览与参照框外化为核心的转换方案，证明在大尺度变化时可显著提升空间记忆和连续感。

**🔧 技术方法**

使用Unity 3D + HTC Vive XR Elite，设计实现了世界缩放（WiM）视图、同尺度移动、第三人称预览、摄像机路径预览等技术。

**📊 数据集**

实验基于Unity资产商店提供的“Japanese City 240217”城市环境（约400×400 m），并在不同位置设置POI和标记。

**📈 对比分析**

采用within-subjects实验，20名受试者在三种技术和基线下完成多轮任务；通过空间回忆误差、NASA‑TLX、UEQ、连续性等量表评估，结果显示在多尺度转换中，外化旋转/姿态预览技术显著降低角误差并提升连续感，优于基线；同尺度转换无显著差异。

**⚠️ 局限性**

局限包括：仅测试同尺度与多尺度共变的情景、固定19 s转换时长、样本量有限、未单独拆分位置/方向/尺度影响、仅单一受众设置，导致对其他场景或更长时间使用的普适性不足。

---

## 480. Enhancing Low Back Pain Assessment with Diffusion Models for Lumbar Spine MRI Segmentation

**arXiv ID:** 2608.04906 | [PDF](https://arxiv.org/pdf/2608.04906v1)

**作者:** Maria Monzon `[一作]` (ETH Zürich), Catherine R. Jutzeler `[通讯]` (ETH Zürich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了基于扩散模型的SpineSegDiff框架，实现对低背痛患者T1/T2加权MRI的统一语义分割，兼顾了解剖结构与退行性病变；

**💡 创新点**

创新点包括：双编码器的2D扩散网络、预分割加速训练、以及利用随机采样生成的置信度热图；

**🔧 技术方法**

采用的技术主要是2D DDPM/DDIM与UNet解码器，配合MSE、Dice及BCE复合损失；

**📊 数据集**

使用的公开数据集为SPIDER多中心低背痛患者218例的矢状MRI；

**📈 对比分析**

通过5折交叉验证与nnU-Net、Diff-UNet 2D、IISDM等模型比较，SpineSegDiff在T1/T2无差别训练下Dice分数与nnU-Net相当，尤其在IVD分割上更优；

**⚠️ 局限性**

局限性在于仍需在更大、更多样本上验证、计算资源需求较高、不确定性热图缺乏精确置信区间，以及在严重退行性病变（如椎体前移、椎间盘狭窄）下精度下降。

---

## 481. Evaluation Pitfalls and Sparsity Limitations in LLM-based Confidence Estimates for Classification

**arXiv ID:** 2608.04899 | [PDF](https://arxiv.org/pdf/2608.04899v1)

**作者:** Elena Merdjanovska `[一作]` (Humboldt University of Berlin), Andreas Rücklé `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在分类任务中的置信度稀疏性问题，并提出通过将数字标记的概率加权（verbalization logprobs）来降低稀疏性。

**💡 创新点**

创新点在于将模型生成的数字词元的概率信息纳入置信度计算，既消除了罕见置信度值导致的阈值选择困难，又无需额外采样；同时呼吁在AUARC评估中统一使用阶梯插值以消除评估偏差。

**🔧 技术方法**

采用了黑盒置信度估计技术（verbalization、sampling、token logprobs）并提出新的verbalization logprobs方法；使用API获取token logprobs、计算期望值并与传统方法对比。

**📊 数据集**

实验数据集包括SST‑2、SST‑5、Amazon ESCI产品分类和Yahoo!答案主题分类四个基准。

**📈 对比分析**

通过AUARC、AUROC、ECE等指标对比，verbalization logprobs在AUARC上比普通verbalization提升约2.3个百分点，几乎与四次采样的verbalization sampling相当，却仅需一次推理，成本显著降低。

**⚠️ 局限性**

局限性包括对单词化单一token的假设、仅适用于能返回logprobs的API、采样次数固定、仅讨论阈值插值对AUARC的影响，其他阈值基准如AUPRC仍需进一步验证。

---

## 482. Strengthening Target-Language Features: SAE-Based Steering for Multilingual Inference

**arXiv ID:** 2608.04904 | [PDF](https://arxiv.org/pdf/2608.04904v1)

**作者:** Hongsheng Wang `[一作]` (Johns Hopkins University), Phlipp Koehn `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种推理时多语言驱动方法，利用预训练稀疏自编码器识别并增强目标语言特征，从而提升多语言任务性能；

**💡 创新点**

不需要模型参数更新或额外训练，通过稀疏SAE特征差异驱动隐藏层调制，能在推理阶段快速改善低资源语言表现；

**🔧 技术方法**

稀疏自编码器（SAE）、激活向量注入、层级特征选择、语言对比激活差异；

**📊 数据集**

Gemma‑3‑12B‑it 语言模型；使用 FLORES‑200 进行特征识别；评估数据集为 XCOPA（常识推理）、XNLI（自然语言推理）、MGSM（多语言数学推理）；

**📈 对比分析**

与无驱动基线和多种对照策略（直接隐藏维度驱动、隐藏状态缩放、最小二乘英语投影）对比；在 XCOPA 上平均提升约 11.3% 点，在 XNLI 上约 4.2% 点，MGSM 仅提升 0.6% 点；

**⚠️ 局限性**

依赖于兼容的预训练稀疏自编码器，仅适用于已公开的模型家族，缺乏对其他架构或自编码器训练方式的通用性验证；

---

## 483. From Score Matrices to Football-Aware Match-State Simulation: An Auditable LLM Harness for Exact-Score Reranking

**arXiv ID:** 2608.05030 | [PDF](https://arxiv.org/pdf/2608.05030v1)

**作者:** Shaopeng Liang `[一作]` `[通讯]`, Shaopeng Liang

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文结合动态泊松模型与大型语言模型（LLM），构建了一种可审计的混合足球比分预测体系，并通过逐步迭代改进实现了从纯数学预测到基于情境推理的路径模拟。

**💡 创新点**

创新点包括：①可审计的架构，将概率估计、情境提示、LLM推理与确定性验证严格分层；②在LLM中引入显式的“首破门”根决策、停止规则与得分级联评估；③利用冻结候选集与可扩展尾部候选，提升候选覆盖率但仍保持可验证性；④演示了LLM在保持概率先验的前提下为比分分布注入语义解释的可行性。

**🔧 技术方法**

技术手段包括：动态Dixon–Coles泊松模型（带时间衰减与低分修正）、LLM推理层（受限提示、可解释路径生成）、概率先验与候选池的冻结与验证、以及定量评价指标（log-loss、Brier、RPS、Top‑1/3准确率）。

**📊 数据集**

使用的数据集为2015–16至2024–25赛季的各大联赛及欧冠等比赛做训练验证，随后对2025–26英超前150场比赛进行回放评估。

**📈 对比分析**

与V1（纯数学）以及V3（路径模拟）对比，V4在Top‑1准确率从10.0%提升至14.7%，Top‑3从26.7%提升至30.7%，但其基于候选排名的结果仍低于V1的原生1X2概率决策（53.3%）。V1在概率评估上表现更好，取得0.9878的log‑loss、0.5870的Brier和0.2095的RPS。

**⚠️ 局限性**

局限性主要有：LLM仅输出排名而非校准概率，无法参与适当评分；未能有效提升0–0或大分差情形的准确率；尾部候选虽扩展但未被充分提升；实验为回放性质，存在模型记忆泄露风险；并且迭代设计基于已知结果，缺乏真正前瞻性验证。

---

## 484. Language Models Generalize to Human-like Word Order Preferences

**arXiv ID:** 2608.05028 | [PDF](https://arxiv.org/pdf/2608.05028v1)

**作者:** Amanda Popadich `[一作]` (University of Washington), Shane Steinert-Threlkeld `[通讯]` (University of Washington)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在去掉多修饰词名词短语的语料上训练语言模型，检验它们是否能在未见过的多修饰词句子中展现出 scope‑homomorphism（语义范围同构）偏好。

**💡 创新点**

提出了将人工语言学习（ALL）范式迁移到大规模语言模型中的 FiCT（Filtered Corpus Training）方法，并显示即便缺乏直接证据，模型仍能自发学习人类一般化偏好，进一步证明此偏好可由通用学习机制产生。

**🔧 技术方法**

使用 OPT 变体的解码器‑仅 Transformer（52M、110M、350M 参数）进行训练，并通过最小‑对比评估（HCΔ、HC%）衡量顺序偏好。

**📊 数据集**

以 2023 年 Wikipedia 抽取的 100M-token 语料为基础，随后对包含多修饰词名词短语的句子进行分解处理，构成训练集；评估集为 150 个最小对（50 对/修饰词组合）。

**📈 对比分析**

对比不同模型规模及修饰词组合（Dem‑Num、Dem‑Adj、Num‑Adj）的 HCΔ 和 HC%；结果显示所有模型均显著偏好同构顺序（平均 HC% ≈ 70–76%，HCΔ ≈ 0.9–1.1），并且 Dem‑Num 组合表现最强。进一步的 PMI 相关性分析未能解释模型偏好，说明词汇关联强度不足以预测。

**⚠️ 局限性**

限制包括：①模型并非人类学习者，学习目标和输入分布不同；②训练环境虽移除直接证据，却不完全等同于人类 ALL 的极简人工语言；③仅在英语上实验，未验证跨语言适用性；④PMI 仅捕捉词汇共现，未涵盖更高层次的句法/语义信息。

---

## 485. Exact simulation of diffusions and improved algorithms for log-concave sampling

**arXiv ID:** 2608.05022 | [PDF](https://arxiv.org/pdf/2608.05022v1)

**作者:** Fan Chen `[一作]` (MIT), Matthew S. Zhang `[通讯]` (MIT)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出了一种基于路径空间拒绝采样的精确模拟扩散过程的新框架，并将其应用于无阻尼 Langevin 扩散（ULD），在强对数凹、光滑目标分布上实现了高精度采样；此外，扩展到镜像 Langevin 扩散并给出了非对数凹分布的 Fisher 信息界。

**💡 创新点**

创新点在于：①利用 Girsanov 定理构造可无偏估计的密度比，并通过一次性查询即可完成对整条轨迹的重加权；②设计了两种提议分布（指数欧拉提议和基于 Picard 迭代的高阶提议），在仅有二阶光滑时实现 O(κ^{2/3} d^{1/3}/ε) 的复杂度；在三阶导数可控时进一步提升至 O(κ^{1/2}+d^{1/5}) 的复杂度；③将该框架推广至镜像 Langevin 并改善了维度依赖；④结合 proximal sampler 得到非对数凹采样的 Fisher 信息复杂度提升至 O(β d^{1/3} KL_0/ε^2)。

**🔧 技术方法**

核心技术包括：路径空间拒绝采样（FORS）、Girsanov 变换、无偏估计子采样、指数欧拉提议、Picard 迭代高阶提议、Hessian‑vector oracle、对数 Sobolev 与 Poincaré 不等式的连续时间收敛分析、镜像映射与相对强凸性、Fisher 信息与 KL 收敛的结合。

**📊 数据集**

本工作为理论性研究，无实测数据集；所有实验与评估均在理论复杂度分析与先前工作对比中进行。

**📈 对比分析**

与现有最佳算法比较：MALA 的复杂度为 O(κ d^{1/2}/ε)，MHMC 为 O(κ d^{1/4}/ε)。本方法在 κ 维度上实现了从 O(κ) 到 O(κ^{2/3}) 的加速，在 d 维度上实现了从 O(d^{1/2}) 到 O(d^{1/3})（仅二阶光滑）或 O(d^{1/5})（三阶光滑）的显著降维；在非对数凹场景下，Fisher 信息采样的维度依赖由 d^{1/2} 降至 d^{1/3}。

**⚠️ 局限性**

限制与未来工作：①实现路径重加权需对无阻尼扩散的空势分布进行精确模拟，实际实现难度较大；②三阶导数可控的改进仅适用于 Hessian‑Lipschitz 场景；③复杂度分析基于理想化的梯度与 Hessian‑向量 oracle，实际实现中需额外的数值误差控制；④在极大维度或非光滑目标时，当前方法的优势可能被削弱。

---

## 486. Reward Structure Shapes the Interaction Between Episodic Exploration and Neural Memory in Reinforcement Learning

**arXiv ID:** 2608.05111 | [PDF](https://arxiv.org/pdf/2608.05111v1)

**作者:** Jai Malegaonkar `[一作]` (University of California San Diego), Henrik I. Christensen `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过系统对比两种基于回合的探索奖励（E3B、NovelD）与六种循环记忆架构（GRU、LSTM、RetNet、GatedDeltaNet、Mamba‑2、无记忆控制）在三种部分可观测环境（MysteryPath‑Grid、MiniGrid‑MemoryS13、TinyReproduce）中的表现，研究了探索奖励与记忆架构的相互作用；

**💡 创新点**

创新点在于：①首次将探索奖励与记忆架构联合评估，揭示三种不同的交互模式（放大、均衡、无效）；②通过控制奖励结构而非奖励密度，证明奖励结构决定探索奖励的有效性；③引入“观察锚定奖励机”来形式化奖励稀疏度，区分结构稀疏与潜在稀疏两种属性，并用它们解释三种交互模式。

**🔧 技术方法**

技术包括：基于PPO的循环强化学习、两种经验式探索奖励、六种循环网络实现、奖励机（Reward Machine）与POMDP的数学形式化、实验对照与统计显著性检验。

**📊 数据集**

使用了三种自定义离散动作环境：MysteryPath‑Grid（隐路径探索）、MiniGrid‑MemoryS13（需要记忆提示的分岔点）和TinyReproduce（按序列复现）。

**📈 对比分析**

比较方法为在相同训练堆栈下交叉所有奖励与架构，记录最终20%检查点的确定性策略成功率；结果显示：在MysteryPath中奖励放大低容量架构的差距；在MiniGrid中奖励将低容量架构提升至同一性能上限；在TinyReproduce中奖励无显著影响。

**⚠️ 局限性**

局限性包括：仅评估离散动作、单一 on‑policy PPO 与前向分离奖励/记忆的设置；奖励机形式化在高维输入下可能不可扩展；未检验全局奖励或离线学习者，且未分析奖励对内部表示的具体影响。

---

## 487. Agent Against Agent: An Agentic System for Automatic Prompt Injection Red Teaming

**arXiv ID:** 2608.05108 | [PDF](https://arxiv.org/pdf/2608.05108v1)

**作者:** Yanting Wang `[一作]` (Pennsylvania State University), Jinyuan Jia `[通讯]` (Pennsylvania State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个名为PIMiner的agentic prompt injection红队系统，利用层次化记忆机制将攻击经验转化为可读的策略库，并可在未见目标LLM上直接转移；

**💡 创新点**

创新点在于通过策略库、路由器以及跨样本的内部记忆，将搜索式攻击的经验累积与共享，实现搜索式方法与RL式方法的性能接近，同时保持高可解释性；

**🔧 技术方法**

核心技术包括：多层记忆架构（长期策略库、数据集级记忆、样本级记忆）、策略路由器、迭代攻击模块与经验消化器；

**📊 数据集**

使用了IPIArena、AgentDojo、InjecAgent三大公开benchmark，训练集包含多种工具使用、编码等场景，测试集涵盖多目标LLM；

**📈 对比分析**

与静态攻击、搜索式攻击（TAP、PAIR、Strategy）以及RL式攻击（Vanilla GRPO、RL-Hammer、PISmith）对比，PIMiner在ASR@10上在IPIArena和AgentDojo上分别达到76.2%/86.7%和61.9%/53.3%，与RL式方法相当且显著优于其他基线；

**⚠️ 局限性**

局限性包括：主要使用Claude Code模型，针对不同LLM的成本差异；策略库的生成需要在训练阶段使用高算力模型，且对极大规模模型的泛化性与实时性仍待验证；

---

## 488. CoPlan: A Trustworthy Co-Intelligence Interface for Care Planning through Role-Based Contestable Argument Graphs

**arXiv ID:** 2608.05107 | [PDF](https://arxiv.org/pdf/2608.05107v1)

**作者:** Hung Truong Thanh Nguyen `[一作]` (University of New Brunswick), Hung Cao `[通讯]` (University of New Brunswick)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了CoPlan，一个支持多代理协同、可检验、可争议的人机协作护理规划界面。

**💡 创新点**

创新点在于将可争议性与协同智能结合，提供角色基础的论证图，允许护理人员在生成护理计划前检查、质疑、修改人工智能的建议。

**🔧 技术方法**

采用多代理系统、语言模型（LLM）、检索增强、量化双极论证框架（QBAF）和图形交互界面。

**📊 数据集**

使用interRAI Home Care评估数据、医学知识库和历史医疗向量数据库作为输入。

**📈 对比分析**

在示例案例中展示了四个阶段的工作流，未给出量化对比，但演示了人机争议流程后生成的护理计划优于单纯自动推荐。

**⚠️ 局限性**

限制在于目前的有效性和置信度评分仅用于审查而非自主决策，系统缺乏大规模实证评估，可能导致过度依赖或缺乏公平性。

---

## 489. Bag-of-Visual-Words for Spatial Mapping of Lung Adenocarcinoma Growth Patterns

**arXiv ID:** 2608.05074 | [PDF](https://arxiv.org/pdf/2608.05074v1)

**作者:** Darya Ardan `[一作]` (University of Geneva), Henning Müller `[通讯]` (University of Geneva)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

设计了一种弱监督的 Bag-of-Visual-Words（BoVW）管线，利用少量 ROI 在肺腺癌 WSI 上实现模式的空间映射。

**💡 创新点**

创新点在于在 ROI 级别学习视觉词汇并构建模式原型，通过 Jensen–Shannon 散度进行最近原型分类，从而在无像素级标注的情况下生成可解释的模式地图。

**🔧 技术方法**

采用冻结的基础模型编码器提取 tile 嵌入，K-means 聚类生成视觉词汇，BoVW 直方图编码，Jensen–Shannon 散度与最近邻原型匹配，以及滑动窗口投影生成空间图。

**📊 数据集**

使用 87 位 CPTAC-LUAD 患者的 WSI 以及 168 个标注 ROI（分布于 6 种模式）进行训练和评估。

**📈 对比分析**

与基于均值池化的 SVM 监督基线对比，肿瘤/健康分类平衡准确率最高 0.974（接近 0.987 的监督方法），二元级别分级平衡准确率 0.729，优于 0.678 的监督基线。

**⚠️ 局限性**

局限性包括 ROI 标注量少且类别不平衡、评估集的等级标注弱且不平衡，以及缺乏像素级或 WSI 级真值，限制了对空间映射精度的客观评估。

---

## 490. If it is Good Then Drop it -- a Spiteful Poisson Process for Submodular Maximization

**arXiv ID:** 2608.05062 | [PDF](https://arxiv.org/pdf/2608.05062v1)

**作者:** Ariel Kulik `[一作]` (Ben-Gurion University of the Negev), Mohit Singh `[通讯]` (Georgia Tech)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种基于 Poisson 过程的混合算法，在满足任意稀疏基（matroid）约束下对非单调和单调的子模函数进行最大化，得到 1/e（非单调）和 1‑1/e（单调）的近似保证。

**💡 创新点**

创新点在于：1) 引入“spiteful” 步骤，即在 Poisson 事件时有意丢弃已在当前解中的元素，从而克服非单调性导致的困难；2) 通过将连续 greedy 与离散交换映射相结合，构造无离散化、无舍入的全新算法框架；3) 证明该算法对一般稀疏基与划分稀疏基都能保持低评估成本。

**🔧 技术方法**

使用的技术包括：Poisson 过程建模与分析、子模函数的多线性扩展、matroid 交换映射、连续 greedy 的测度化（measured continuous greedy）思路、随机采样与 Hoeffding/马尔可夫不等式估计多线性扩展、以及对非单调情况的改进 swap 过程。

**📊 数据集**

无实验数据集；论文完全基于理论分析与证明，关注算法的近似比与评估复杂度。

**📈 对比分析**

与现有离散算法（如局部搜索 1/4，随机贪心 ~0.274）相比，取得了同等 1/e 的近似且评估次数显著降低；与连续 greedy 方案相比，保持相同的 1/e（非单调）与 1‑1/e（单调）近似，比现有最优 1/e+0.033 的连续算法更易实现，且对划分稀疏基实现了 O(n ln(1/ε)) 的评估复杂度。

**⚠️ 局限性**

局限性包括：1) 仍未突破 1/e+0.033 的连续算法的近似下限；2) 对于一般稀疏基，评估复杂度为 O(nk ln(1/ε))，对极大规模实例仍可能昂贵；3) 需要对多线性扩展的高质量估计，采样开销在实践中可能成为瓶颈。

---

## 491. Exact Model-Free Policy Iteration for Co-safe LTL Planning

**arXiv ID:** 2608.05047 | [PDF](https://arxiv.org/pdf/2608.05047v1)

**作者:** Zetong Xuan `[一作]` (University of Florida), Yu Wang `[通讯]` (University of Florida)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于模型无关的策略迭代算法，用于解决具有co‑safe LTL目标的最大可达性问题。

**💡 创新点**

创新点在于通过引入折扣代理识别并消除导致Bellman方程多重解的闭合强连通分量（clamp set），从而实现对真正可达性概率的精确学习。

**🔧 技术方法**

利用折扣式TD估计、clamp集构造、无折扣策略评估与贪婪改进，以及随机逼近理论证明收敛。

**📊 数据集**

以一个5×4的随机网格世界（含trap状态和scLTL公式A∧B）为实验数据集。

**📈 对比分析**

与精确值迭代基线对比，算法在7步策略改进后收敛到最优策略，估计误差保持在1.8%以内，显示出与基线相当甚至更优的样本效率。

**⚠️ 局限性**

局限在于需要折扣因子γ和阈值ε的经验设定，且在极大状态空间下clamp集的准确识别与收敛速度仍有待改进。

---

## 492. Gradient Immunity: Null-Space Resistance to Malicious Fine-Tuning

**arXiv ID:** 2608.05045 | [PDF](https://arxiv.org/pdf/2608.05045v1)

**作者:** Yuxuan Huang `[一作]` (Shanghai Artificial Intelligence Laboratory), Chaochao Lu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在provider-controlled partially protected open-weight (PPOW) 设定下提出了一种Unidirectional Safety Gate（USG），通过在模型最后一层插入不可学习的Null Space Cubic Layer与Inverse Adapter，实现对恶意下游微调中有害样本梯度的阻断，同时保持安全样本的正常前向推理。

**💡 创新点**

创新点包括：①首次将null-space理论应用于LLM微调防御；②构造可恢复前向行为的Null Space Cubic Layer和Inverse Adapter组合；③采用发布时阈值化的梯度门控，减少对下游用户合作的依赖。

**🔧 技术方法**

主要技术：Null Space Cubic Layer、Inverse Adapter、梯度门控（norm‑ratio阈值化）、基于代表性空间阻断的安全阈值校准。

**📊 数据集**

使用的数据集与模型：Qwen‑14B、Llama‑8B；有害数据来自JailbreakBench、HarmfulBench、BeaverTails‑H；恢复任务使用Alpaca；对比实验包括无防御、Rep Noise、Booster、AntiBody等基线。

**📈 对比分析**

通过对比ASR（攻击成功率）和安全通过率评估。USG在严格阈值下，后续微调后的ASR基本保持与发布前一致，安全通过率在JailbreakBench/HarmfulBench为100%，在BeaverTails‑H略降；Inverse Adapter在恢复任务上几乎不影响模型性能，显示USG能在PPOW环境下有效抑制恶意微调，同时保持实用性。

**⚠️ 局限性**

局限性：null space 对输入变化敏感；随着有害样本增多公共null空间消失；在exact零情况下Inverse Adapter失效；阈值硬化可能误拦安全样本；扩展null空间在大规模有害样本时仍无法保证足够分离；依赖于发布时可获得的安全样本集合。

---

## 493. BridgeVLA++: A Data-Efficient, Generalizable, and Memory-Augmented Vision-Language-Action Framework for 3D Manipulation

**arXiv ID:** 2608.05042 | [PDF](https://arxiv.org/pdf/2608.05042v1)

**作者:** Peiyan Li `[一作]` (New Laboratory of Pattern Recognition Institute of Automation Chinese Academy of Sciences), Tieniu Tan `[通讯]` (New Laboratory of Pattern Recognition Institute of Automation Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 BridgeVLA++，一种在 3D 视觉‑语言‑动作（VLA）框架中加入统一时空记忆模块的机器人操控模型，能够在保持数据效率和泛化能力的同时，实现对记忆依赖任务的有效推理；

**💡 创新点**

创新点包括：① 将 3D 观测投影为多视角 2D 图像，使输入与输出在同一 2D 热图空间对齐；② 在此基础上设计了跨阶段的时空记忆体系，时序记忆捕捉交互历史，空间记忆恢复被遮挡的几何信息；③ 引入可扩展的语言条件热图预训练，提升语言理解与视觉定位的融合；④ 采用粗细两级热图推理与 6D 连续旋转表示，增强精细定位；⑤ 轻量化双臂扩展方案，支持 bimanual 操作。

**🔧 技术方法**

技术手段包括：预训练的 PaliGemma VLM、正交投影多视角点云渲染、热图预测与凸包插值、交叉注意力记忆注入、自适应子目标选择、粗细级别的热图回归、6D 连续旋转编码、双臂共享 VLM 并行头。

**📊 数据集**

使用的数据集有：RoboPoint（热图预训练）、RLBench、COLOSSEUM、GemBench、RMBench、MemoryBench 以及在 Franka Research 3 与 Dobot CR5A 机器人上的真实实验数据。

**📈 对比分析**

与基线相比，BridgeVLA++ 在 RLBench 上平均 90.5% 成功率，超越 SAM2Act + 3.7pp；在 COLOSSEUM 与 GemBench 上分别取得 64.0% 与 50.0%，领先现有 3D VLA 方法；在记忆依赖基准 RMBench 与 MemoryBench 上实现 96% 与 99.7% 的高成功率；在真实机器人上记忆任务成功率达 93.3%，显著优于 SAM2Act +。

**⚠️ 局限性**

局限性包括：仍需一定量的演示数据；记忆模块增加约 9% 参数和 0.2 秒推理延迟；对极端遮挡或超长记忆场景的鲁棒性尚待进一步提升；预训练热图任务的迁移效果受限于可用标注。

---

## 494. Private Direct Preference Optimization for LLM Alignment

**arXiv ID:** 2608.05040 | [PDF](https://arxiv.org/pdf/2608.05040v1)

**作者:** Yangfan Jiang `[一作]` (National University of Singapore), Bolin Ding `[通讯]` (Alibaba Group)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种针对大型语言模型对齐的偏好隐私方法（PrivDPO），在保证对齐性能的同时，对人类偏好数据提供正式的ε-偏好隐私保护。

**💡 创新点**

核心创新在于发现DPO梯度差异仅沿一维“偏好轴”变化，并利用该结构在DPO目标上随机重缩放（无偏偏好强度扰动），从而实现只在敏感维度注入噪声，避免了传统DP-SGD全梯度噪声导致的偏差和效率问题。

**🔧 技术方法**

使用的技术包括：Direct Preference Optimization (DPO)、无偏随机重缩放机制、PrivSFT（对SFT阶段的隐私化）、理论分析（无偏估计、误差界）以及大型LLM训练框架（如PyTorch FSDP）来实现可扩展训练。

**📊 数据集**

实验数据集包括 Anthropic-HH、TL;DR Summarization 以及 UltraFeedback-Binarized 三个公开对齐基准。

**📈 对比分析**

与 DP-SGD、随机响应（RR）以及基于 Duchi/Piecewise 的标量扰动方法进行对比；PrivDPO 在奖励指标（reward margin/accuracy）、生成质量（LLM-as-a-judge win rate）以及训练时间/显存使用上均显著优于基线，并且在多种模型（Qwen2.5、Llama3、Pythia）和多种规模（3B–32B）上保持接近非私有 DPO 的性能。

**⚠️ 局限性**

局限性：仅适用于标准 DPO 目标，对 KTO/IPO/SIMPO 等 DPO 变体不直接适用；只能保护偏好标签，无法防止预训练模型已存在的敏感信息泄露；用户级隐私、交互式反馈等更复杂场景尚未覆盖。

---

## 495. Dimensions of Power: A Systematic Guide to Power Indices for Explainable AI

**arXiv ID:** 2608.05031 | [PDF](https://arxiv.org/pdf/2608.05031v1)

**作者:** Filip Naudot `[一作]` (Umeå University), Christopher Blöcker `[通讯]` (Umeå University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统综述并整理了针对单玩家、集合以及基于基数的三种维度的功权指数，并对其经典与非经典公理进行证明和分析。

**💡 创新点**

创新点在于提出将功权指数划分为三维度的框架，扩展并统一已有指数的定义，并给出完整的公理化评价表，为选择合适指数提供指导。

**🔧 技术方法**

采用协同博弈理论中的功权指数（Shapley、Banzhaf、Owen、Counterfactual、cardinality‑based 及 Υ‑value 等）以及公理化分析方法。

**📊 数据集**

示例使用了合成的医疗症状分类与欺诈检测模型数据，以及公开的基准模型的预测概率作为价值函数。

**📈 对比分析**

通过对每个指数在效率、对称性、无效玩家、弱匿名性、成功性、因果性、定量因果性等七个原则下的满足情况进行比较，结果显示不同指数满足不同组合的公理，未出现单一最优。

**⚠️ 局限性**

局限性包括未讨论指数的计算复杂度与近似方法，对特定应用的解释有效性依赖于价值函数的设计，且实验仅基于合成数据未在大规模真实数据上验证。

---

## 496. On Computational Hardness of Mistake-Bounded Language Generation: A Random-Oracle Query Separation

**arXiv ID:** 2608.05029 | [PDF](https://arxiv.org/pdf/2608.05029v1)

**作者:** Xiaoyu Li `[一作]` (University of New South Wales), Junbin Gao `[通讯]` (University of Sydney)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在随机oracle模型下证明，信息理论上可无错生成无限语言集合，但任何多项式查询生成器在有限期内必须产生指数级错误。

**💡 创新点**

首次将生成在极限的闭包维度与查询复杂度分离，展示信息理论可行性与计算难度可共存的极限。

**🔧 技术方法**

使用闭包维度、随机oracle图、稀疏查询、植入耦合和Borel–Cantelli等概率工具构造分离。

**📊 数据集**

没有实测数据，实验基于随机oracle构造的理论实例。

**📈 对比分析**

通过理论证明对比，展示无查询生成器零误差与多项式查询生成器指数级错误的对比，未给出数值性能。

**⚠️ 局限性**

结果仅在随机oracle模型下，尚未在标准模型或自然语言类中实现，且对多项式查询下的具体误差常数不确定。

---

## 497. Link prediction on multi-relational graphs from an influence propagation perspective

**arXiv ID:** 2608.05016 | [PDF](https://arxiv.org/pdf/2608.05016v1)

**作者:** Zidu Yin `[一作]` (Yunnan Normal University), Javen Qinfeng Shi `[通讯]` (Adelaide University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于扩展SIR传播模型的影响传播子图，并通过均值场理论引入虚拟边对子图进行压缩，随后构建Influential Graph Neural Predictor实现多关系图链路预测。

**💡 创新点**

创新点包括：①将SIR模型扩展为双向影响传播以捕获局部与全局信息；②利用虚拟边压缩影响路径，显著降低计算成本；③将压缩后的子图与递归GNN结合，提升链路预测的准确性。

**🔧 技术方法**

使用技术包括：扩展的SIR传播模型、均值场理论、虚拟边构造、递归图神经网络（Recurrent GNN）、多种损失函数设计以及子图提取算法。

**📊 数据集**

实验数据集包含5个公开基准：蛋白质-蛋白质相互作用（PPI）、合作作者网络、社交网络、Ego网络以及Wikidata知识图（1k/5k/10k节点）。

**📈 对比分析**

在统一实验设置下，将方法与相似度、潜在特征、各类GNN基线进行比较。结果显示在所有数据集的多分类和二分类任务中均优于基线，准确率/宏F1提升30%~40%，二分类准确率提升约9%，在大规模多类数据上最高达95%以上。

**⚠️ 局限性**

局限性在于仅针对静态图设计，未处理动态关系变化；模型在极大规模或高度动态图上的推广仍需进一步研究。

---

## 498. DASyR-LLM: Domain-Aware Symbolic Regression with LLMs for Kinetic Model Discovery

**arXiv ID:** 2608.05120 | [PDF](https://arxiv.org/pdf/2608.05120v1)

**作者:** Roberto Aliaga Medina `[一作]` (Imperial College London), Antonio del Rio Chanona `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种结合大型语言模型（LLM）与符号回归（SR）的迭代化方法，用于加速化学反应动力学模型的发现。

**💡 创新点**

将LLM嵌入SR循环，既对候选模型进行物理可行性评估，又在每一步生成基于领域知识的新的速率表达式，从而显著减少实验迭代次数。

**🔧 技术方法**

基于ADoK‑S符号回归框架，配合Qwen3‑14B LLM进行批判与生成，结合AIC模型选择和MBDoE实验设计。

**📊 数据集**

使用四个合成案例数据集（烷基化、氧化还原、异构化反应及细菌蛋白生产），通过数值模拟生成并加入高斯噪声。

**📈 对比分析**

与传统仅使用SR的基线在相同实验预算下对比，LLM指导方法平均迭代次数减少41.7–79.3%，验证集R²>0.98，预测性能相当。

**⚠️ 局限性**

计算成本每次迭代更高，仍需人工解析LLM输出，参数估计未保证全局最优，且仅在合成数据上验证，真实实验中未测试。

---

## 499. ABSeeker: Training Long-Horizon Search Agents via Answer-Backtracked Credit Assignment

**arXiv ID:** 2608.05102 | [PDF](https://arxiv.org/pdf/2608.05102v1)

**作者:** Yijun Lu `[一作]` (Shanghai Jiao Tong University), Siheng Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于答案回溯的细粒度信用分配框架ABC，用于训练长时序搜索代理；

**💡 创新点**

创新点在于通过回答回溯恢复中间证据线索，再以这些线索为依据给每一步分配具体奖励，从而区别成功轨迹中的错误行为和失败轨迹中的有价值行为；

**🔧 技术方法**

技术包括答案回溯线索恢复（Answer‑Backtracked Clue Recovery）、基于线索的步骤评分（Clue‑Anchored Step Scoring）、加权监督微调（ABC‑SFT）以及基于步骤奖励的GRPO强化学习（ABC‑GRPO）；

**📊 数据集**

主要使用 BrowseComp 与 BrowseComp‑ZH（长时序英文/中文信息检索任务）进行训练和评估，并在 xbench 与 GAIA‑text 等通用任务上做跨域验证；

**📈 对比分析**

与同规模 4B 基础模型、30B 规模搜索代理以及多种基准模型对比，ABSeeker 在 BrowseComp 及 BrowseComp‑ZH 上从 37.3% 提升至 55.3% 与 52.9%（开启上下文管理），在 xbench‑2505、xbench‑2510、GAIA‑text 上分别达到 77.0%、46.0%、81.6%，整体性能优于同规模基线并与更大模型竞争；

**⚠️ 局限性**

局限性包括仅在 4B 模型规模下实验，未探索更大模型的提升；此外方法依赖可验证答案的任务，对无答案或不唯一答案的情境适用性尚待验证。

---

## 500. VQ-VAD: Vector-quantized Motion Representation Learning for Human-centric Video Anomaly Detection

**arXiv ID:** 2608.05069 | [PDF](https://arxiv.org/pdf/2608.05069v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 501. Lesion Detection in CT with Frozen Self-Distilled Features: SALT, a Spatially Adaptive Label-Guided Temperature

**arXiv ID:** 2608.05100 | [PDF](https://arxiv.org/pdf/2608.05100v1)

**作者:** Mahmut S. Gokmen `[一作]` (University of Kentucky), V. K. Cody Bumgardner `[通讯]` (University of Louisville)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

在CT病灶检测任务中，作者提出一种在自监督预训练期间对教师目标进行空间自适应温度和损失加权的SALT方法，利用仅在预训练阶段可获得的弱标签框在ViT编码器上生成更具针对性的病灶表征，并在冻结编码器的基础上训练轻量化的CenterNet风格检测头完成3D病灶定位。

**💡 创新点**

创新点在于把自监督目标的空间适配从传统的视图引导迁移到目标层面：通过在教师分布上局部降低温度并在病灶区域提升掩码损失权重，使得病灶在预训练过程中获得更强、更集中且不依赖于后续标签的学习信号；这一机制仅在预训练阶段使用弱标签，推理时完全不需要任何标注或条件。

**🔧 技术方法**

技术包括ViT-L/ViT-B transformer编码器、DINOv2与iBOT的自蒸馏与掩码目标、register token、4层多深度特征拼接、3×3 stem + 6残差块的多尺度 CenterNet 关键点检测头、焦点损失、L1框回归、CPM 评价指标等。

**📊 数据集**

实验基于八个CT病灶库（DeepLesion、UniToChest、NLST‑seg、LiTS、NIH‑Lymph、HCUCH、Tübingen、LyNoS），使用弱标签框在训练集上进行预训练，评估在四个测试队列（共3983个病灶）和HCUCH RECIST 的28对纵向扫描上。

**📈 对比分析**

与同架构、同数据、同标签引导视图但无目标层面条件的DINO‑LG基线相比，SALT在分离度上提升约四倍（0.449 vs 0.105），CPM提升约0.11；在外部自然图像和医学基础模型中，SALT ViT‑L的CPM达到0.556，高于MedDINOv3 0.423、Curia 0.433；对小于6 mm的病灶加成显著，整体CPM提升约10%。

**⚠️ 局限性**

局限性包括：在某些队列（NLST‑seg、HCUCH）效果略有下降，显示对不同数据分布的敏感性；评估仅覆盖捕获一次检测和固定阈值，未涉及体素级分割；仅验证了弱框标签的泛化，尚未测试对其他弱注释的适用性；SALT的温度锐化与损失加权未分离，难以单独量化两者贡献。

---

## 502. Same Formulas, Different Semantics: Do Language Models Follow Modal Logic Specifications?

**arXiv ID:** 2608.05097 | [PDF](https://arxiv.org/pdf/2608.05097v1)

**作者:** Réemi Andrieu `[一作]`, Damien Sileo `[通讯]` (University of Lille)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于对称-不对称的平衡核心对照组来评估大型语言模型在模态推理中的语义控制能力。

**💡 创新点**

创新点在于通过保持公式不变，仅改变显式的框架或域条件，并使用自动定理证明器给出对立标签，从而逼迫模型真正遵循语义规范，而不是默认的熟悉逻辑。

**🔧 技术方法**

使用了自动化推理技术（Vampire、Leo‑III、LET嵌入、Kripke评估器）作为oracle，构造了800个对照实例（其中160个为平衡核心）并对模型进行推理模式与非推理模式的对比实验。

**📊 数据集**

数据集为自定义的模态推理问答集合，包括平衡核心的160对，以及更大范围的800对；所有实例均以控制英语、TPTP格式呈现并经过非经典TPTP转化。

**📈 对比分析**

实验结果显示，直接推理下大多数模型在平衡核心上准确率低于仅使用语义条件的50%基线；开启推理模式后，DeepSeek V4 Flash从4.4%跃升至88.1%，表明推理计算能显著恢复语义控制，但仍不能保证完全正确。

**⚠️ 局限性**

局限性包括：仅测试单模态、单框架/域的受控英语；缺乏自然语言多样性与人类基准；对多代理或非序列框架的迁移性未知；自动推理的依赖于特定的TPTP嵌入与定理求解器；以及对不同表示方式的影响仍不完全清晰。

---

## 503. Optimizing What Policies Learn From: Recoverability-aware Rollout Intervention Learning

**arXiv ID:** 2608.05080 | [PDF](https://arxiv.org/pdf/2608.05080v1)

**作者:** Zheyuan Zhang `[一作]` (University of Notre Dame), Wei Niu `[通讯]` (Amazon, Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于可恢复性（recoverability）的回合生成干预学习框架（RAIL），将回合生成视为可学习的优化过程，动态决定何处以及如何进行额外的rollout；

**💡 创新点**

核心创新是将回合干预建模为上下文赌博机问题，通过学习可恢复性收益实现非平稳（随策略演化而变化）和非标量（多维干预空间）两大缺口；

**🔧 技术方法**

技术方法包括：可恢复性控制器的上下文赌博机训练、shadow‑to‑live 阶段的监督收集与在线更新、structured intervention 空间（分支预算与解码策略）、GRPO 与 policy‑gradient 结合；

**📊 数据集**

实验使用四个agentic reasoning benchmark：AgentBench‑OS、AgentBench‑DB、WebShop、ToolQA‑Coffee；

**📈 对比分析**

与GRPO、ARPO、AEPO、VIP、TAMPO、Tree‑GRPO等基线对比，RAIL 在成功率（SR）上整体领先，同时在相同或更低的 rollout 预算下获得更高性能；

**⚠️ 局限性**

局限性：需要手工设定干预空间与阈值，shadow 阶段耗时；对不同模型/任务的泛化仍需验证；对大规模环境的可扩展性尚待进一步探索。

---

## 504. Optimal Constrained sc-LTL Planning in MDPs via Switching Policies

**arXiv ID:** 2608.05021 | [PDF](https://arxiv.org/pdf/2608.05021v1)

**作者:** Zetong Xuan `[一作]` (University of Florida), Yu Wang `[通讯]` (University of Florida)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了在马尔可夫决策过程（MDP）上使用可安全线性时序逻辑（scLTL）进行目标与安全约束的最优规划方法。

**💡 创新点**

通过双产品MDP构造将非马尔可夫性问题转化为可用 stationary 策略最优的约束可达性问题，证明了切换策略（预切换+后切换）足以达到全局最优，并能用线性规划求解。

**🔧 技术方法**

使用 LMDP、scLTL 与 DFA 的双产品构造、可达性概率的占用度量、切换策略理论及线性规划（HiGHS）求解。

**📊 数据集**

在论文中未采用公开大规模数据集，主要以四乘五格子世界（带随机滑动与吸收障碍）作为案例验证。

**📈 对比分析**

通过对阈值 τ 进行扫描，绘制目标与约束的权衡曲线，显示可行区间至 τ≈0.64；与无约束最优策略相比，得到的策略在满足约束的同时仅略降低目标成功率，性能符合预期。

**⚠️ 局限性**

局限性包括：仅处理可安全 LTL（cosafe fragment）且仅考虑到两个独立目标的情况；对更复杂的 LTL 或多目标约束需进一步扩展；算法在状态空间与 DFA 规模较大时可能面临指数增长的计算开销。

---

## 505. Characterizing Visual Accessibility Issues in AI Developer Tools: An Empirical Study

**arXiv ID:** 2608.05116 | [PDF](https://arxiv.org/pdf/2608.05116v1)

**作者:** Sabrina Haque `[一作]` (University of Texas at Arlington), Christoph Csallner `[通讯]` (University of Texas at Arlington)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对5个AI辅助开发工具的公共问题追踪和论坛进行挖掘，识别并系统分析可视化可访问性问题。

**💡 创新点**

首次从公开维护渠道系统性收集和归类AI工具的视觉可访问性报告，并结合LLM自动标注与主题建模实现大规模无人工标注的可访问性问题识别。

**🔧 技术方法**

关键词检索 + 三模型（GPT‑5‑mini、Gemini‑2.5‑Flash、Llama‑3.3‑70B）集成判别 + BERTopic 主题聚类 + 定性手工编码。

**📊 数据集**

共收集2,652条候选问题，最终筛选出600条高置信度视觉可访问性报告，涉及VS Code Copilot、Cursor、Claude Code、OpenAI Codex、OpenCode等生态。

**📈 对比分析**

通过比较不同生态系统中报告的障碍类别比例、闭合率、维护者参与度等指标，展示编辑器集成、终端和AI面板各自主导的可访问性障碍类型；结果表明不同生态在处理报告方面存在显著差异。

**⚠️ 局限性**

关键词检索可能漏检或误检；LLM一致性未完全等同人工标注；主题模型参数对结果敏感；仅捕获公开报告，闭合状态不一定代表问题已修复。

---

## 506. Hierarchical Graph Memory for LLM Agents with Path-level Localization and Rewrite

**arXiv ID:** 2608.05095 | [PDF](https://arxiv.org/pdf/2608.05095v1)

**作者:** Xiawei Yue `[一作]` (Nankai University), Ziwei Zhang `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个可演化的层级图记忆框架，支持对长期推理任务的高效记忆检索、路径级定位与协同重写。

**💡 创新点**

创新点在于：① 将记忆划分为上层抽象节点与细粒度 MemoryUnit 的层级结构，减少无关上下文；② 引入 MicroGraph 进行查询/更新条件下的路径级定位；③ 采用协同重写同时更新单元状态与依赖关系，避免重复重写与过时依赖。

**🔧 技术方法**

主要技术包括：层级图记忆建模、MicroGraph 构造与匹配、路径级证据定位、内外部单元协同重写、基于语义匹配与时间一致性的评分函数。

**📊 数据集**

使用了两大公开基准：LoCoMo（长期对话问答）和 MemConflict（冲突感知记忆评估）。

**📈 对比分析**

与多种基线（如 MemoryBank、A-MEM、ReadAgent、MemGPT、Mem0、LangMem、Letta、MemOS 等）对比，实验表明在 LoCoMo 上取得最高的 F1、BLEU 与 LLM-J 分数，同时显著降低 token 消耗；在 MemConflict 上实现了最优的答案准确率、冲突识别与支持检索指标，证明了路径定位与协同重写的有效性。

**⚠️ 局限性**

局限性包括：在 Open Domain 任务上表现不如在时间/多跳场景，主要因缺乏外部知识支持；框架依赖图结构和显式依赖关系，若记忆内容极为稀疏或结构化不佳时效果受限；未来需要探索多模态融合与外部知识整合。

---

## 507. MALT: Lightweight Curvature-Aware Muon via Diagonal Preconditioning

**arXiv ID:** 2608.05088 | [PDF](https://arxiv.org/pdf/2608.05088v1)

**作者:** Tongle Wu `[一作]` (Pennsylvania State University), Ziye Ma `[通讯]` (City University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种轻量级的 Muon 优化器 MALT（及其自适应版本 MALTER），通过在矩阵参数上引入双侧对角预处理、Newton–Schulz 正交化和范数嫁接，实现对损失曲率和梯度异方差的同时抑制。

**💡 创新点**

创新点包括：1) 仅使用对角预处理（行、列平方梯度均值）在预处理空间完成 Muon 正交化，显著降低了对曲率异方差的敏感性；2) 采用范数嫁接分离更新方向与幅度；3) 在此基础上引入基于噪声的自适应步长（MALTER），实现噪声自适应与曲率预处理的统一。

**🔧 技术方法**

技术方法：Newton–Schulz 迭代进行矩阵正交化；对角预处理通过对梯度行/列平方和做指数移动平均；范数嫁接与归一化；自适应步长基于 Adam 类噪声估计（NAMO 思路）；在预处理空间下的梯度下降与正交化结合。

**📊 数据集**

使用 GPT‑2（Small 124M、Medium 355M、Large 774M）在 OpenWebText（约 9 B 训练标记、4.4 M 验证标记）上进行预训练实验。

**📈 对比分析**

与 AdamW、Muon 进行对比。通过网格搜索学习率，MALT 在所有模型规模下均优于 Muon；MALTER 在验证损失上进一步领先，整体最优。内存和单步时间几乎与 Muon 相当，显著优于 AdamW；实验表明 MALT/MALTER 在收敛速度和最终损失上都有显著提升。

**⚠️ 局限性**

限制：理论收敛分析对矩阵维度有较高的多项式依赖，实际效果仍受预处理对角化程度的限制；对比 Dense 预处理方法（如 FISMO、Mousse）能获取更丰富的曲率信息，但需更多内存/计算；MALT 在极端曲率变化下可能不如稠密预处理稳定。

---

## 508. SpikingNav: Robust Embodied Navigation with Spiking Neural Policies

**arXiv ID:** 2608.05078 | [PDF](https://arxiv.org/pdf/2608.05078v1)

**作者:** Jiahong Zhang `[一作]` (Chinese Academy of Sciences), GuoqiLi `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 SpikingNav，一个将脉冲神经网络用于室内机器人导航的框架，包含脉冲感知编码器 SSE 和脉冲策略网络 SPN，适用于 PointNav 和 ObjectNav 任务。

**💡 创新点**

创新点在于将脉冲神经网络从感知编码扩展到完整的策略学习，并通过膜电位累积、阈值触发与脉冲重置的动态机制提升对视觉噪声的鲁棒性，同时保持模型轻量化。

**🔧 技术方法**

采用了基于 SNN 的 ResNet‑style 感知编码器、脉冲递归网络（integrate‑and‑fire 结构）、PPO 强化学习框架以及 Surrogate‑gradient 训练方法，并在 Thruster‑V2 neuromorphic 处理器上实现硬件验证。

**📊 数据集**

在 RoboTHOR 模拟环境中使用 PointNav 与 ObjectNav 两个基准任务，并通过 RobustNav 提供的七种视觉失真（噪声、模糊、裂纹等）进行鲁棒性测试。

**📈 对比分析**

与匹配参数规模的 ANN 版（ANNNav）以及多种主流方法对比，SpikingNav 在 PointNav 上接近 ANNNav 的成功率，ObjectNav 上提升了约 3.1% 的成功率，并在所有失真场景下平均成功率提升至 13.71%（比 ANNNav 高 5.26%），同时参数减少约 12%，每步 FLOPs 下降约 77%。

**⚠️ 局限性**

主要局限在于评估主要基于高保真仿真，硬件验证仅覆盖感知模块，未完整实现闭环导航；模型规模受限于边缘设备，未验证大规模 Vision‑Language‑Action 架构的可迁移性。

---

## 509. MultiPathFormer: Towards a Foundation Model for Multipath Wireless Propagation

**arXiv ID:** 2608.05076 | [PDF](https://arxiv.org/pdf/2608.05076v1)

**作者:** Blessed Guda `[一作]` (Carnegie Mellon University), Carlee Joe-Wong `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 MultiPathFormer，基于多径传播的自回归基础模型，用路径序列代替传统通道张量进行预训练，并在下游任务（波束预测、定位、LoS分类、信道估计）中复用。

**💡 创新点**

创新点在于：① 将多径路径本身作为预训练对象，采用 next‑path 预测目标；② 引入环境检索增强生成（RAG）获取局部与走廊几何；③ 采用 K‑means 代码簿为首条路径提供先验并仅预测残差，显著提升首条路径精度。

**🔧 技术方法**

技术手段包括：Transformer 编解码器、自动回归路径生成、环境 RAG 检索与聚合、首条路径代码簿残差预测、适配器微调、基于路径的损失（L2+交叉熵+长度损失）以及多任务下游头部。

**📊 数据集**

使用 DeepMIMO 现场级射线追踪数据，涵盖 31 个场景，其中 27 个用于预训练，4 个作为完全新环境进行零样本评估，样本量约 23.9M 条路径标记。

**📈 对比分析**

与 LWM、WiFo、MLP、LSTM 等基线在同一数据集上对比。结果显示：波束预测 top‑1/3 分别提升至 67.1%/91.4%；定位均值误差 5.57 m；LoS 分类零样本 97.6%/95.9% F1，微调后 99.4%/99.0%；信道估计 NMSE -13.27 dB，未微调时已超过基线。

**⚠️ 局限性**

局限性包括：跨环境零样本迁移仍显弱，需更多真实测量数据验证；对环境物体信息的依赖使模型在缺少完整 3D 场景时受限；代码簿和 RAG 设计可能导致模型对特定硬件/部署的细粒度适配不足。

---

## 510. HelloWorld: Enabling Socially Interactive Characters in Video World Models

**arXiv ID:** 2608.05070 | [PDF](https://arxiv.org/pdf/2608.05070v1)

**作者:** Liangyang Ouyang `[一作]` (University of Tokyo), Yoichi Sato `[通讯]` (University of Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了HelloWorld，一种能够实现用户与视频世界内角色进行社交互动的交互式视频世界模型；

**💡 创新点**

创新点包括：①使用自蒸馏方式将预训练的视频生成模型转换为可控世界模型，无需外部数据或人工标注；②采用warp视频作为摄像机运动的几何指令；③设计无训练的时间交叉注意力掩码，实现对交互时机的精确控制；以及①创建了首个社交互动基准HelloWorldBench。

**🔧 技术方法**

核心技术包括：自蒸馏训练（使用Pi3X提取点云与相机轨迹，Warp Video条件），LoRA轻量化微调，Temporal Cross-Attention Mask，基于DiT的视频生成架构。

**📊 数据集**

使用的数据集：基于LTX-2.3生成的自制交互视频；以及为评估而构建的HelloWorldBench（120张高质量图片，配合LLM生成交互描述和相机轨迹，共400个评估样本）。

**📈 对比分析**

与WorldPlay、Matrix-Game 3.0、LingBot-World、SANA-WM、Warp-as-History等五个最近的世界模型进行对比。HelloWorld在社交交互的三项指标（ActAcc、TimeAcc、GazeDev）上显著优于所有基线，并且在视频质量、相机跟随和背景一致性方面与最佳基线持平或略优。

**⚠️ 局限性**

限制：目前不支持实时交互，交互和相机轨迹需预先指定；缺乏对角色身份的持续跟踪与多轮交互支持；以及受限于基础模型的结构与计算成本，无法实现更长时长或更复杂的世界生成。

---

## 511. Beyond Reprojection Error: Camera Calibration with 3D Targets

**arXiv ID:** 2608.05066 | [PDF](https://arxiv.org/pdf/2608.05066v1)

**作者:** Dennis Ruppel `[一作]` (Fraunhofer Institute for Computer Graphics Research), Arjan Kuijper `[通讯]` (Fraunhofer Institute for Computer Graphics Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种结合环形特征检测与射线评估的三维相机标定框架，并引入 icosahedron 3D 标定目标。

**💡 创新点**

创新点包括：①使用基于射线的交叉误差和重建误差来直接评估 3D 重建精度；②设计 icosahedron 目标以提供更丰富的 3D 特征，降低参数相关性；③将环形特征检测应用于平面与 3D 目标。

**🔧 技术方法**

采用泛化相机模型（Luhmann/Brown 失真模型）、Levenberg–Marquardt 优化、Otsu 阈值、边界层级分割、随机自举采样、Ray‑tracing 渲染以及三角化求解。

**📊 数据集**

使用合成 Ray‑tracing 数据（约 200 张图像，已知真值）和 PhaseOne iXG 100MP 相机真实采集的 60 组数据，包含 5 个标定目标（平面 ringboard、铝板、3D 打印板、三种 icosahedron）。

**📈 对比分析**

通过比较重投影误差、交叉误差、重建误差以及位姿误差，发现 icosahedron 在交叉误差上平均降低约 40%，并在自举实验中标准差更小，显示更稳健；但在重投影和重建误差上平面目标表现更好。真实实验中平面目标误差最低，icosahedron 误差受打印误差影响显著增大。

**⚠️ 局限性**

局限性：icosahedron 目标在实际制造时几何误差大，导致真实数据表现不及合成；未将射线误差直接融入束束平衡；对大范围视角的鲁棒性验证有限。

---

## 512. RepairFormer: Automated Repair of Structured Inputs Using Transformers

**arXiv ID:** 2608.05060 | [PDF](https://arxiv.org/pdf/2608.05060v1)

**作者:** Ovi Paul `[一作]` (University of Houston), Ali Shokri `[通讯]` (University of Houston)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一种基于Transformer的结构化输入修复框架RepairFormer，自动修复多种格式的语法错误。

**💡 创新点**

创新点在于把修复任务化为有监督的序列生成，通过格式标签、边界定位和oracle验证实现对原始内容的最大保留。

**🔧 技术方法**

使用T5/CodeT5模型、LoRA参数高效微调、边界定位模块以及oracle驱动的验证流程。

**📊 数据集**

利用从GitHub下载的DOT、INI、JSON、OBJ、S-expression、TinyC等有效文件，人工加入单字符、双字符和截断变异生成buggy–repaired对。

**📈 对比分析**

与ϵREPAIR、ANTLR、DDMax等基线对比，在外部和内部基准上修复率达88%/94%恢复率，整体得分82.9%，并且相对基线平均快5倍。

**⚠️ 局限性**

局限在于受最大token长度限制，依赖特定oracle验证，边界定位误差可能导致失败，且局部修复缺乏全局上下文信息。

---

## 513. The Effect of Perceived Race and Gender on Police Language Use: Experimental Evidence from VR Simulations

**arXiv ID:** 2608.05050 | [PDF](https://arxiv.org/pdf/2608.05050v1)

**作者:** Sandra C. Sandoval `[一作]` (University of Maryland), Hal Daumé `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用虚拟现实（VR）实验，评估警察在与黑人男性虚拟角色互动时的言语尊重度变化，并通过因果推断衡量这种影响；

**💡 创新点**

首次将多层次混合效应模型与大语言模型（LLM）特征化文本相结合，在真实警察-虚拟角色对话中估计平均处理效应（ATE），并验证其在合成数据上的可靠性；

**🔧 技术方法**

使用混合效应回归、逆倾向得分加权（IPTW）与双稳健（Doubly Robust）估计，结合LLM（Llama 3.1 8B）提取对话嵌入以及LLM微调预测模型；

**📊 数据集**

基于79名美国各地区警察在三种情境（房屋、公交站、便利店）中的VR互动语料，包含5,009名众包标注者给出的尊重度评分；

**📈 对比分析**

与合成数据验证比较显示，Pipeline I（混合效应+IPTW+LLM特征化）在场景、官员性别/种族子组中均显著捕捉负向ATE；Pipeline II（LLM微调+双稳健）在合成数据中能检出较大效应，但在真实数据上效果不显著，模型解释度（R²）远低于Pipeline I；

**⚠️ 局限性**

局限包括：对真实场景的因果推断仍受未观测混杂影响，LLM特征化方法可能缺乏对多层级结构的充分捕捉，且微调模型对非线性关系捕捉不足，未来需改进模型设计与更大样本验证。

---

## 514. When Do PEFT Adaptations Leak Structure? Measuring Black-Box Structural Bounds in Public-Base Model Services

**arXiv ID:** 2608.05036 | [PDF](https://arxiv.org/pdf/2608.05036v1)

**作者:** Zhongjiang Yao `[一作]` (King's Collage London), Gang Shi `[通讯]` (Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究公共基础模型与私有参数高效微调（PEFT）服务的泄漏面，提出将受害者与基础模型的残差转换为结构性签名并通过校准后验推断 PEFT 家族、层局部性和秩等信息，并对版本指纹、重建与接口可观测性进行系统评估。

**💡 创新点**

创新点在于：① 以残差签名为核心的多特征结构估计框架；② 通过三门（对齐、结构、接口）分层检测泄漏；③ 公开评估私有版本指纹和成本-格式前沿；④ 发现生成模式和标签接口对泄漏可观测性的显著边界。

**🔧 技术方法**

使用残差特征提取、分布式多元回归（多项式逻辑回归）、逻辑元拒绝器、校准后验概率、精确匹配的服务级签名聚合，以及基于对抗实验的拒绝阈值设定。

**📊 数据集**

主要数据集包括 MNLI、QQP、SST-2、AG News 等自然语言推断与分类任务，模型体系包括 BERT、RoBERTa、DeBERTa-v3、LLaMA/Alpaca 与 Llama‑3.1‑8B‑Instruct，采用多种 PEFT 变体（LoRA、Adapter、Head、Prefix、BitFit、DoRA 等）。

**📈 对比分析**

与传统的纯蒸馏学生和参数/计算匹配学生对比；在族类泄漏上 BERT/MNLI 等模型的闭集 Top‑1 准确率显著超越随机；在版本指纹上 AUC 达到 0.94；但在重建方面，后验约束的 PEFT 与蒸馏→PEFT 的性能相当或略低，且在公平查询预算下无查询优势；生成任务下，教师强制比自由运行更能保留结构信号。

**⚠️ 局限性**

局限性：实验主要聚焦 BERT/MNLI，开放集与基准漂移检测灵敏度有限；对标识标签接口的泄漏为随机，无法覆盖商业化对话 API；缺乏对更大规模或跨平台服务的验证；基准漂移时的对齐判别器表现不佳，外部攻击者难以独立验证公共基模型的一致性。

---

## 515. BnBERT-iPET: Sparse Few-Shot Language Modeling for Bengali via Lottery Ticket Pruning

**arXiv ID:** 2608.05104 | [PDF](https://arxiv.org/pdf/2608.05104v1)

**作者:** Sajib Hossain `[一作]` (North South University), Nabeel Mohammed `[通讯]` (North South University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个稀疏的孟加拉语BERT模型BnBERT-iPET，先通过少量样本的Iterative Pattern Exploiting Training (iPET)提升模型能力，再利用Lottery Ticket Hypothesis剪枝实现90%稀疏，只保留10%权重，最终在多项孟加拉语下游任务上实现竞争性能；

**💡 创新点**

首次将iPET与彩票票据剪枝结合用于低资源语言的稀疏化小模型，既保持了few‑shot学习能力，又大幅降低模型规模与能耗；

**🔧 技术方法**

使用的技术包括BERT预训练、iPET迭代式模式利用训练、Lottery Ticket Hypothesis权重剪枝、稀疏矩阵训练与推理、token化与自定义数据处理；

**📊 数据集**

采用自构建的多来源孟加拉语多样化数据集BanglaDDS（包含新闻、社交媒体、Shadhu文本）以及情感、情绪、作者、新闻分类、POS、标点恢复等七个下游任务的数据集；

**📈 对比分析**

与Bangla Electra、Indic‑BERT、XLM‑RoBERTa、BanglaBERT、Indic‑DistilBERT等大型或轻量模型进行基准对比；在情感、作者、新闻分类、POS、标点恢复和情绪分类等任务中，稀疏模型在多数指标上与大模型相当甚至优于部分大模型，同时训练和推理时间明显低于XLM‑RoBERTa；

**⚠️ 局限性**

局限性包括仅在孟加拉语上验证，模型仍依赖BERT架构，极高稀疏度可能导致性能衰减，缺乏对其他语言或更大规模任务的泛化评估；

---

## 516. Multimodal Spatiotemporal Atmospheric Data Assimilation with Latent Flow-matching

**arXiv ID:** 2608.05103 | [PDF](https://arxiv.org/pdf/2608.05103v1)

**作者:** Dibyajyoti Chakraborty `[一作]` (Pennsylvania State University), Romit Maulik `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

训练了一个隐式视频流匹配先验，利用ERA5重分析数据生成连续8天的全尺度大气状态，并通过该先验实现超分辨率、稀疏观测融合、滤波、平滑以及直接观测到预报等多种数据同化任务；

**💡 创新点**

创新点在于统一无条件隐视频流匹配先验，实现所有DA任务的“同一模型”处理；内在时间传播无需额外数值模型；结合自适应噪声级权重、引导采样与Langevin纠正，提升采样效率与分布校准；

**🔧 技术方法**

采用TrigFlow流匹配与3D Diffusion Transformer（DiT3D）作为先验，配合3D卷积自编码器压缩；使用DPS式引导、SDA温度调节、Langevin纠正和自适应噪声级权重的采样方法；

**📊 数据集**

使用ERA5重分析（1.40625°，128×256，69变量，32帧/8天）作为训练和评估数据；结合NOAA IGRA、ISD、ICOADS观测作为稀疏/稠密观测来源；

**📈 对比分析**

与GraphDOP、ECMWF IFS等基准进行对比；在超分辨率任务中RMSE约1.02K，风场RMSE≈3m/s；观测同化的RMSE低于ERA5误差；直接观测到预报六日RMSE与GraphDOP相当甚至更优，展示了与传统DA方法可比的性能；

**⚠️ 局限性**

局限性包括：观测仅能对应ERA5格点变量，缺乏对卫星辐射观测的处理；高纬度/细尺度动力特征捕捉不足；缺乏长期子季预报与气候一致性校准；以及对大规模部署的计算成本与实时性尚未彻底评估。

---

## 517. HexMIL: Hierarchical Attention MIL for Ante-Hoc Explainable Detection of AI-Manipulated CT Volumes

**arXiv ID:** 2608.05101 | [PDF](https://arxiv.org/pdf/2608.05101v1)

**作者:** Orazio Pontorno `[一作]` (University of Catania), Sebastiano Battiato `[通讯]` (University of Catania)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研发 HexMIL，用于检测与定位 CT 影像中的医学深度伪造，解决跨生成器泛化和可解释性难题；

**💡 创新点**

提出双层多实例学习框架，结合层级门控注意力，能够仅用体积二元标签训练并生成先验可解释的 3D 关注体积；

**🔧 技术方法**

采用 ResNet‑50 作为特征提取器，门控注意力机制、多实例学习、位置编码、滑动窗口分块以及两阶段训练等技术；

**📊 数据集**

在 M3DSynth 与 CT‑GAN（基于 LIDC‑IDRI）数据集上进行实验，涵盖多种生成模型的注入/去除肺结节场景；

**📈 对比分析**

与 2D/3D CNN、Transformer、单层 MIL 以及专门的深度伪造检测基线进行比较，HexMIL 在跨生成器分类上达 AUC 92.0、F1 90.2，定位上 IoU 42.4%、PG 70.6，显著优于所有对比方法；

**⚠️ 局限性**

固定长度窗口可能会切断大体积中的伪造区域；仅在肺部 CT 上验证，未测试其他解剖部位或 MRI；对多中心数据和扫描协议变化的鲁棒性尚未评估。

---

## 518. Item Response Theory for AI Safety

**arXiv ID:** 2608.05086 | [PDF](https://arxiv.org/pdf/2608.05086v1)

**作者:** Joshua Fonseca Rivera `[一作]` (Independent), Konstantinos Voudouris `[通讯]` (UK AI Security Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文使用项目反应理论对192个语言模型在八个安全基准上的表现进行分析。

**💡 创新点**

创新点在于发现安全评估可以分解为拒绝严格度、真确性与上下文危害三因素，能够用极少的测试项重构完整得分并检测模型沙箱化行为。

**🔧 技术方法**

采用两参数逻辑斯蒂IRT、因子分析、计算机自适应测试与人配适统计。

**📊 数据集**

数据集包含五千余条测试项、192个模型的响应，覆盖八大安全基准（HarmBench、TruthfulQA等）。

**📈 对比分析**

与随机子集相比，25项静态测试和约10项自适应测试能以低于20%的RMSE重现全套得分，且对沙箱化检测的准确率可达80–97%。

**⚠️ 局限性**

局限在于仅使用公开模型、沙箱化测试基于提示而非精调，样本量虽大但仍低于人类心理测验；缺乏对真实部署环境的预测验证。

---

## 519. Capability-Gated Planning: Cost-to-Goal Discovery and the Limits of Myopic Experiment Selection

**arXiv ID:** 2608.05085 | [PDF](https://arxiv.org/pdf/2608.05085v1)

**作者:** Ahmed Hassoon `[一作]` (Johns Hopkins University), Mark Dredze `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究自动科学发现中，如何在构造性实验（如建造仪器、构建程序库）与信息获取实验之间做出决策，并提出一种基于成本到达目标的规划方法 CG‑Plan。

**💡 创新点**

创新点在于：①把发现过程建模为信念空间中的随机最短路（SSP）问题；②证明在有能力门控的情形下，任何仅依据有限视野信息增益的“近端”策略都可能出现无界的最优比；③设计了一种加入能力成本的启发式 h = h_cap + h_info 的增益估计，从而在不做完整深度搜索的情况下捕捉构造性实验的长期价值。

**🔧 技术方法**

技术方法主要包括：随机最短路（SSP）框架、可行能力图的删减松弛（delete‑relaxation）求解最小构造成本、增益启发式的加法组合、以及 LPA*/D*Lite 递归重规划以实现在线决策。

**📊 数据集**

实验使用了一个人为构造的基准测试床：布尔电路模型，包含可测量终端、深层内部节点与依赖链式构造实验，以及可选的无用构造干扰物。没有使用公开的真实科学实验数据集，而是通过模拟来验证理论。

**📈 对比分析**

与基线方法（随机探测、基于 EIG 的贪婪、H‑step 近端 EIG、基于可行性评分的推理等）对比，CG‑Plan 在所有受门控情形下均能以最低成本成功到达置信阈值；而近端方法在门控强度大时完全失败；在无门控情形下两者性能相当。

**⚠️ 局限性**

主要局限在于：①实验示例为人为构造的弱化模型，未能证明真实科研问题普遍存在深层能力门控；②假设能够获得准确的能力依赖图，实际环境中该图可能未知或不完整；③对多资源约束、多目标评价等实际需求的支持尚不充分；④对混合信息与构造实验的情形缺乏理论分析。

---

## 520. Provable Limits and Certified Deferral for Verbalized Uncertainty in Small Language Models

**arXiv ID:** 2608.05064 | [PDF](https://arxiv.org/pdf/2608.05064v1)

**作者:** Jianru Shen `[一作]` `[通讯]` (University of Montana), Jianru Shen (University of Montana)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种面向小型开源语言模型的风险控制式推理放弃框架，利用模型自述的置信度（verbalized confidence）进行后置校准，并通过 Clopper‑Pearson 上界给出有限样本的置信度阈值保留策略。

**💡 创新点**

创新点包括：① 在小模型规模（0.5B–14B）下验证并量化语义化置信度的可用性；② 用理论证明严格单调校准保持风险‑覆盖前沿，阐明温度缩放不可校准的不可行性下限；③ 引入基于 Clopper‑Pearson 的保真阈值选择，实现可验证的风险预算；④ 识别并修正 TruthfulQA 的答案顺序偏差。

**🔧 技术方法**

主要技术手段有：① 通过固定模板提取置信整数并归一化；② 后置校准方法：温度缩放、Platt 逻辑回归和单调回归；③ 以严格单调的 Platt 作为部署校准器；④ 通过 Clopper‑Pearson 置信区间和联合上界选择满足风险预算的阈值；⑤ 对模型进行多任务、跨模型的阈值迁移和格式鲁棒性检验。

**📊 数据集**

使用的数据集为 ARC‑Challenge（小学科学题目）和 TruthfulQA（单项选择、以字母计分），各自提供 200 题校准集，余下约 1,271 题和 617 题作为测试集。

**📈 对比分析**

与常见校准方法（温度缩放、Platt、单调回归）对比：Platt 在所有模型–任务组合下均显著降低 ECE；温度缩放在 0.5B 级别模型中普遍无效；单调回归在保持校准精度的同时破坏覆盖网格。经过 Platt 校准后，Clopper‑Pearson 仅在 20% 风险预算下为 Qwen2.5‑14B（ARC）和 Qwen2.5‑7B（ARC）等最强模型提供了 99.8% 及 93.3% 的可保留率；其他模型在 10% 或 20% 预算下均无可验证阈值，说明放弃是必要的默认策略。

**⚠️ 局限性**

局限性包括：仅测试多项选择任务，未覆盖开放式生成；假设部署数据与校准集 i.i.d.，在分布漂移下无效；校准集仅 200 题，限制了可证风险范围；模型规模上限为 14B，并未检验更大模型；格式鲁棒性实验仅覆盖两种模板且仅在 TruthfulQA 上验证。

---

## 521. Hardware Design and Security in the Era of Chiplets and LLMs

**arXiv ID:** 2608.05063 | [PDF](https://arxiv.org/pdf/2608.05063v1)

**作者:** Johann Knechtel `[一作]` (New York University), Ramesh Karri `[通讯]` (New York University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了2.5D芯片模块系统与LLM驱动EDA工作流中的安全威胁与防御方案，并提出了基于物理隔离的根信任架构与LLM辅助安全验证的未来方向。

**💡 创新点**

创新点在于将芯片模块安全与LLM安全相结合，提出了使用活跃互连层实现物理隔离根信任（2.5D RoT）以及通过LLM生成的安全断言、模型校准与机器无学习技术来对抗LLM诱导攻击和后门。

**🔧 技术方法**

采用的技术包括2.5D split制造、活跃互连层的TRANSMON与CMCs、机器无学习SALAD、对抗性红队训练NetDeTox与TrojanGYM、以及LLM微调与RAG框架。

**📊 数据集**

所引用的数据集包括VerilogEval、VeriContaminated、CWE列表、TrojanInS等，但本文自身并未使用实验数据。

**📈 对比分析**

本文通过对比各类安全防御方法的理论优缺点进行评述，未给出定量实验结果；综述表明所提防御在理论上可降低后门成功率并提高模型对齐质量。

**⚠️ 局限性**

局限性在于缺乏实测验证、对新提出的2.5D RoT与LLM安全框架的实现细节不足，以及对跨域协同方案的具体性能评估仍待进一步研究。

---

## 522. OmniEdit-Bench: A Comprehensive Benchmark for Instruction-based Video Editing

**arXiv ID:** 2608.05049 | [PDF](https://arxiv.org/pdf/2608.05049v1)

**作者:** Chenxuan Miao `[一作]` (University of Hong Kong), Hengshuang Zhao `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并发布 OmniEdit-Bench，构建覆盖空间、时间、音频、参考、推理五个轨道的指令式视频编辑评测框架，并提出四维度（准确性、保持性、真实性、一致性）评估及准确性惩罚机制。

**💡 创新点**

①系统化分层任务设计，细分空间、时间、音频、参考与推理维度；②四维度评估与准确性衰减机制，确保评估反映指令忠实度；③利用 Gemini‑3.1‑Pro 进行大规模自动评估，验证与人工评估高度一致。

**🔧 技术方法**

采用视频扩散模型、流匹配技术实现编辑；使用 Gemini‑3.1‑Pro 作为视觉语言模型做自动评估；构建多模态参考与音频同步机制，并通过多维度权重融合最终分数。

**📊 数据集**

基于现有 VE‑Bench、EditBoard、OpenVE‑3M 等公开数据集，扩充并整理出约 850 条视频样本，覆盖 5 个轨道（空间 240、时间 200、参考 200、音频 100、推理 50）。

**📈 对比分析**

采用 0–100 分的加权评分体系（0.5×准确性 + 0.2×保持性 + 0.15×真实性 + 0.15×一致性），对比开源与商用模型。结果显示，商用模型在空间轨道可达 70 分以上，但在时间、音频、推理轨道普遍低于 20 分，开源模型整体更低，凸显在时序与推理上的明显不足。

**⚠️ 局限性**

缺乏对长期时序依赖和复杂推理的有效建模；评估仍依赖 VLM，可能对极端样本产生偏差；数据规模有限，未覆盖全部复杂场景；目前大多数模型对音频编辑支持不足。

---

## 523. German parties shifted towards intuition-based rhetoric after the far right's parliamentary breakthrough

**arXiv ID:** 2608.05075 | [PDF](https://arxiv.org/pdf/2608.05075v1)

**作者:** Peer Saleth `[一作]` (University of Konstanz), David Garcia `[通讯]` (University of Konstanz)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究德国政治精英在2015-2025年间，推特与议会演讲中的证据式与直觉式话语变化，聚焦极右党AfD进入议会后的影响。

**💡 创新点**

首次跨平台、多维度量化“证据-直觉”话语，揭示极右可见性与精英话语风格在欧洲多党制议会中的关联。

**🔧 技术方法**

采用分布式词典表示（DDR）、词嵌入（word2vec、fastText）、EMI指标、混合效应回归与BERTopic主题控制等技术。

**📊 数据集**

数据集包括约450万条德国政客推文、59,170份18-21届联邦议院正式演讲，使用德语词典翻译验证。

**📈 对比分析**

通过在推特与议会两个领域对EMI变化、党派与意识形态水平进行时间序列与结构断点回归，并使用主题控制验证稳健性；结果显示两大领域均出现显著的向直觉化倾向下降。

**⚠️ 局限性**

仅覆盖精英传播，未检视大众受众反应；议会与推特平台差异可能影响结论；AfD进入议会与其他宏观事件同时发生，因果性难以完全分离。

---

## 524. SparseDitto: Customizing GPU Kernels for Different Sparsity Patterns with LLM-Based Agentic System

**arXiv ID:** 2608.05033 | [PDF](https://arxiv.org/pdf/2608.05033v1)

**作者:** Shiyang Li `[一作]` (University of Minnesota), Caiwen Ding `[通讯]` (University of Minnesota)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于大型语言模型（LLM）的系统 SparseDitto，能够为任意稀疏矩阵、算子（SpMV/SpMM/SpGEMM）和目标 GPU 自动生成最佳 CUDA 内核。

**💡 创新点**

创新点在于：① 通过结构特征分析和可解释加性能量模型对已知优化策略进行排序；② 层次化硬件感知规划在搜索空间中高效生成多种候选设计；③ 结合 LLM 的代码生成与验证，以及 GPU 实测反馈，实现跨算子、跨稀疏模式与 GPU 的统一自适应优化。

**🔧 技术方法**

技术细节包括：稀疏结构特征提取、可解释的加性能量模型、层次化规划与约束求解、LLM 编码/验证代理、GPU 性能测评与迭代细化。

**📊 数据集**

使用 SuiteSparse 60 矩阵集（覆盖常见稠密度、行长度分布和块结构）以及 Reddit 全批 GCN 数据进行实验。

**📈 对比分析**

与 cuSPARSE、CB‑SpMV、DTC‑SpMM、SparseTIR、HSMU‑SpGEMM 等最先进系统对比；在 RTX PRO 6000 上平均加速 2.68×、在 H200 上 2.79×；SpMM 最高 4.46×、SpGEMM 最高 146×，GCN 训练加速最高 3.39×。

**⚠️ 局限性**

局限性包括：对 LLM 生成质量和搜索预算敏感；极端稀疏或特殊算子（如非标准乘法、非方阵输入）可能未得到充分覆盖；模型需要离线训练并依赖已测量的基准数据。

---

## 525. ArtAnno: Annotating Implicit Semantics in Artworks through LLM Agent-Driven Bidirectional Human-AI Augmentation

**arXiv ID:** 2608.05026 | [PDF](https://arxiv.org/pdf/2608.05026v1)

**作者:** Xiaoyan Gu `[一作]` (Zhejiang University), Wei Chen `[通讯]` (Zhejiang University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了基于双向人机协同的艺术品注释系统 ArtAnno，融合主动标签推荐与交互驱动的知识演进，实现多模态、多智能体协作支持高质量、隐含语义注释。

**💡 创新点**

创新点在于：①构建了双向人机增补框架 BiHAA，打破传统单向协同；②采用多智能体架构实现主动支持（技能推荐、聚类、标签建议、聊天）与交互演进（行为挖掘、技能生成、结构化合并、知识库更新）闭环；③提供多模态证据支持验证，增强信任；④强调可迁移性，可应用于医学、考古等知识密集领域。

**🔧 技术方法**

核心技术包括：GPT‑5 LLM（OpenRouter）、LangChain 智能体框架；YOLO‑World‑V2 物体检测；文本与图像相似度嵌入模型；行为挖掘与结构感知技能合并算法；前端 React.js、后端 Flask；多模态描述生成与知识库管理。

**📊 数据集**

使用的主要数据集：传统中国绘画女性身份与隐含意义标注集（30幅图像），以及墨西哥海报情感语义标注案例；数据由领域专家（≥5 年经验）标注并冲突解决后构成参考集；同时利用公开图像与元数据进行多模态描述生成。

**📈 对比分析**

通过三种实验条件（基线 C1、无演进模块 C2、完整系统 C3）进行用户研究与案例评估。系统可用性（SUS）为 84.17，显示优秀；注释效率在 C3 中比 C1 提升约 50%（15.75 分钟 vs 30.92 分钟，p=0.00049）；标注一致率从 73% 提升至 90%（p=0.00049）。演进模块 C3 与 C2 对比，时间略低但体验显著提升（p=0.00049），行为挖掘产生 112 条候选技能，合并后 31 条（约 72% 降重）。

**⚠️ 局限性**

局限性包括：①缺乏个性化交互，用户可自行浏览；②技能可靠性与偏见问题，错误标注可能被纳入技能；③目前不支持多标签标注；④未进行大规模、纵向评估；⑤知识迁移与技能更新缺乏专家审核，可能导致知识偏差。

---

## 526. Canonical Joint Energy-Based Model on CIFAR-10: failure modes and practical indistinguishability of Predictor-Corrector and SGLD samplers

**arXiv ID:** 2608.05025 | [PDF](https://arxiv.org/pdf/2608.05025v1)

**作者:** Dmytro Knopov `[一作]` `[通讯]` (National University of Kyiv-Mohyla Academy), Dmytro Knopov (National University of Kyiv-Mohyla Academy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对 Canonical JEM 在 CIFAR-10 上进行高保真重建，并系统地用 Predictor‑Corrector（PC）与 Stochastic Gradient Langevin Dynamics（SGLD）两种采样器在训练、生成与 OOD 检测三种协议下进行对比实验。

**💡 创新点**

首次在同一实验设置下复现 Canonical JEM 并量化 PC 与 SGLD 的差异，同时揭示两种失败模式：训练期的灾难性发散与运行依赖的 SVHN OOD 判别动态；并给出在固定噪声下 PC 无显著理论优势的解释。

**🔧 技术方法**

采用 JEM 训练框架、SGLD 与 PC 采样器、Replay Buffer 重新采样、Fréchet Inception Distance（FID）与 AUROC 评估、对数抽样 Bootstrap 与 Welch TOST 等统计方法。

**📊 数据集**

使用 CIFAR-10 作为基准数据集；OOD 数据集包括 SVHN、CIFAR-100、DTD、LSUN‑R 与 iSUN。

**📈 对比分析**

通过两独立跑、Margin‑10 检查点对齐、对 AUROC 差值（|Δ|<0.007）与 FID 差值（<0.5）进行对比；层级 Bootstrap 95% CI 包含零，TOST 也未能证明两者等价，说明 PC 与 SGLD 在实际性能上无显著差异。

**⚠️ 局限性**

局限性包括仅进行两次独立跑、PC 参数未尝试可调噪声调度、只测试固定噪声的 Canonical JEM、未验证其他架构/数据集以及低 AUROC 环境下可检验性受限。

---

## 527. Short-term load forecasting under EU-AI Act Requirements in Safety-Critical Environments: Results from a 41-day live challenge on the aggregated German transmission-grid load

**arXiv ID:** 2608.05018 | [PDF](https://arxiv.org/pdf/2608.05018v1)

**作者:** Thomas Bartz-Beielstein `[一作]` `[通讯]` (THK-AI Research Cluster), Thomas Bartz-Beielstein (THK-AI Research Cluster)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本论文构建并评估了一套面向德国输电网的全天候负荷预测系统，系统具备确定性、可复现性与可审计性，符合欧盟人工智能法案的安全关键要求，并在为期41天的实时挑战赛中完成了对德国市场区聚合负荷的日预测。

**💡 创新点**

创新点包括：
• 采用安全关键子集的递归多步 LightGBM 预测器，结合异常检测与缺失值补齐，保证数据完整性与模型可追溯；
• 引入 surrogate‑model 超参数优化（SpotOptim）与对比 TPE（Optuna）两种调参器，验证两者在同一搜索空间下性能无显著差异；
• 通过严格的代码开发与流程规则（CR‑1~CR‑4、PR‑1~PR‑4）实现模型的可审计与可复现；
• 在同一挑战平台上对比传统递归模型、预训练的基础模型（Chronos‑2）与低成本自学习模型（MACL2L），展示前者可在保持可审计性的前提下竞争甚至领先。

**🔧 技术方法**

核心技术与工具：
- LightGBM 级联回归树实现递归多步预测；
- SpotOptim 与 Optuna 作为超参数搜索器；
- Isolation Forest 进行异常检测；
- 日历与天气协变量（Open‑Meteo、德国假期等）；
- skforecast 递归策略的安全关键实现 spotforecast2‑safe；
- Python 生态（pandas、numpy、scikit‑learn）和 GitHub Actions 的持续集成以保证代码可追踪。

**📊 数据集**

数据集：
- ENTSO‑E 实际总负荷（15 分钟级别，聚合至小时）作为目标序列；
- 相关协变量包括日历（时段、星期、月份）、假期标记、气象（温度、湿度、降水等）来自 Open‑Meteo；
- ENTSO‑E 预报（6.1.B）仅用于诊断，禁止作为模型协变量；
- 所有数据均来自 ENTSO‑E 公开平台与 Open‑Meteo API。

**📈 对比分析**

比较方法与性能：
- 以 MAE 为排名指标，同时报告 RMSE、MAPE、平均偏差、低估率（UPR）和 MASE；
- spotoptim‑lgbm 在 35 天共享测试中平均 MAE 为 1369 MW，比 ENTSO‑E 基准低 34.7%（p = 0.0001）；
- Hot Rod 以 1228 MW 的 MAE 领先（但在共享天数上差异不显著）；
- MACL2L（含/不含 ENTSO‑E 输入）和 Chronos‑2 在与传统递归模型的对比中表现相近甚至更优；
- SpotOptim 与 Optuna 的结果在相同预算下几乎相同，验证 surrogate‑model 搜索的有效性。

**⚠️ 局限性**

局限性：
- 仅覆盖 41 天的夏季日，缺乏冬季高峰、极端天气等情形；
- 评估仅针对德国单一市场区，未验证跨区泛化；
- 部分参赛队伍的缺失日由“搬运”规则影响，导致模型质量与提交纪律混合；
- 规则更改（协变量更新、异常门槛）在挑战中途插入，可能对结果产生不可解释影响；
- 预训练模型的可审计性与数据来源仍需进一步研究；
- 仅使用单一天气服务，未考虑气象不确定性对预测的影响。

---

## 528. Cardinal Grid Slime Trail is PSPACE-Complete

**arXiv ID:** 2608.05118 | [PDF](https://arxiv.org/pdf/2608.05118v1)

**作者:** Anne Pham `[一作]` (Dickinson), Matthew Ferland `[通讯]` (Dickinson)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文证明了在四向（cardinal）网格上的Slime Trail游戏是PSPACE‑完整，并进一步通过旋转与缩放技巧推广到八向（diagonal）网格，解决了先前提出的关于网格版本复杂度的开放问题。

**💡 创新点**

创新点在于将原来用于任意平面图的QBF归约改造为严格受限的格子图：通过设计满足度数≤4、偶/奇步长约束的“变量、选择、合并、二极管、交叉”等格子友好型小工具，完成了对量化布尔公式的完整编码；此外，提出了一种45°旋转+√2放大映射，将四向网格构造映射到八向网格，保证了不出现新的捷径。

**🔧 技术方法**

核心技术包括：1）QBF归约框架；2）格子约束下的图形设计与平面嵌入；3）假设点（dummy node）与偶数步长平衡的使用；4）旋转缩放变换以实现八向实现；5）对合法与非法走法的案例分析与证明。

**📊 数据集**

该研究为理论性证明，不涉及具体数据集或实验数据。

**📈 对比分析**

由于本工作是理论证明性质，没有实验或基准测试；因此没有对比方法或性能评估，只说明该游戏在一般情形下属于PSPACE‑完整，属于最难的复杂度类之一。

**⚠️ 局限性**

局限性包括：1）仅讨论正方形网格（四向与八向），未扩展至六边形网格或其他格子；2）对多目标节点场景已处理，但单目标或受限目标数的情况尚未彻底探讨；3）理论证明未给出实际求解算法或近似方法，无法评估在实际竞赛实例上的可行性；4）对图形的尺寸增长随QBF输入多项式，但实际构造仍非常庞大，实际实现难度较高。

---

## 529. Robust and Efficient Motion Reasoning for Privacy-Aware Classroom Incident Recognition

**arXiv ID:** 2608.05115 | [PDF](https://arxiv.org/pdf/2608.05115v1)

**作者:** Paritosh Parmar `[一作]` (Agency for Science, Technology and Research), Chiat Pin Tay `[通讯]` (Agency for Science, Technology and Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了结合生成式CCTV视频和真实课堂姿态的混合数据集，并提出基于层次运动推理的轻量化、隐私友好的课堂事故识别框架。

**💡 创新点**

创新点在于①利用层次运动表示（位置、速度、加速度）捕捉动作动态；②多阶融合教师模型再通过多目标知识蒸馏压缩为仅使用位置序列的单流学生模型；③生成式视频与真实数据结合的混合基准及零样本迁移评估。

**🔧 技术方法**

采用姿态序列预处理、层次运动特征提取、基于STGCN++的多阶骨架网络、加权融合与温度蒸馏，以及多任务损失组合。

**📊 数据集**

使用自研合成数据集（1296条视频，涵盖7类事故+背景类）与真实课堂采集的数据（574条样本），并利用生成式AI（Kling、Seedance）创建场景与制服。

**📈 对比分析**

与AAGCN、STGCN++、CTRGCN、MSG3D、PoseC3D等骨架动作识别基线对比，合成数据上准确率从70.54%提升至71.78%，零样本迁移到真实数据上准确率从54.36%提升至63.41%，并在参数量与GFLOPs上实现至少1/10的压缩。

**⚠️ 局限性**

仍存在合成-真实域差距、样本类别较少、对极端光照与遮挡鲁棒性不足、模型对长时序动作的识别能力有限，且在真正复杂课堂环境中需进一步验证。

---

## 530. Deployment Feasibility Analysis of Post-Quantum Digital Signatures in Safety-Critical C-V2X Communication for Urban Mobility Scenario

**arXiv ID:** 2608.05087 | [PDF](https://arxiv.org/pdf/2608.05087v1)

**作者:** Akid Abrar `[一作]` (University of Alabama), Ahmad Alsharif `[通讯]` (University of Alabama)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了NIST标准化的后量子数字签名算法在C‑V2X PC5 Mode 4侧链通信中的物理可行性和性能表现；

**💡 创新点**

首次结合传输块可行性分析与跨层全栈仿真，量化不同后量子签名在多交通密度与LOS/NLOS环境下的可靠性与时延边界；

**🔧 技术方法**

使用IEEE 1609.2/SAE J3161标准、liboqs实现后量子签名、OpenCV2X与SUMO/Veins联动的Mode 4仿真，并通过TBS、MCS等参数进行可行性评估；

**📊 数据集**

通过SUMO生成四叉信号交叉口车辆流量，按HCM六级交通密度（A–F）以及LOS/NLOS传播模型进行仿真；

**📈 对比分析**

对Falcon‑512与ECDSA P‑256在24个场景下比较包交付率(PDR)与端到端时延；Falcon在LOS A可达90% PDR但在更高密度下失效，时延差异≤2.5 ms，满足100 ms时限；

**⚠️ 局限性**

仅单一交叉口场景、未实现SPS+One‑Shot、未考虑更高MCS、liboqs非硬件优化、NLOS下普遍失败；结果对其他网络、硬件或更复杂环境的泛化有限。

---

## 531. Learning When to Stop: Prefix-Optimal Dynamic Diffusion Policies for Continuous Control

**arXiv ID:** 2608.05084 | [PDF](https://arxiv.org/pdf/2608.05084v1)

**作者:** Rohit Kumar Salla `[一作]` (Virginia Polytechnic Institute and State University), Simon Stepputtis `[通讯]` (Virginia Polytechnic Institute and State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 PO​​GP 方法，通过在扩散策略的每一步学习前缀价值函数，实现动作生成的自适应早停与中间动作质量提升。

**💡 创新点**

创新点在于引入前缀价值函数（PVF）与 Bellman 风格递归双重目标：既为中间动作提供辅助优化目标，又为测试时提供基于价值差值的停止标准。

**🔧 技术方法**

使用扩散政策框架、TD3/SAC 价值网络、前缀价值回归与可学习的截断映射等技术，全部在单一网络内实现。

**📊 数据集**

在 MuJoCo 四个连续控制任务（HalfCheetah‑v4、Walker2d‑v4、Hopper‑v4、Ant‑v4）上进行实验。

**📈 对比分析**

与 12 个基线（TD3、SAC、PPO、Diff‑QL、DPPO、D^2PPO、SDAC、DSAC‑D、IDQL、FQL、SAC‑GMM 等）对比，POGP 在 IQM 上平均提升约 4.6%，迭代次数平均下降约 66%（≈2.7×速度提升），并保持 99% 以上的完整链性能。

**⚠️ 局限性**

局限性：仅在四个模拟环境验证；长链（K>20）或真实机器人任务的适用性尚未探索；对扩散步长敏感，需进一步扩展到更复杂的任务与更大模型。

---

## 532. Kerckhoffs-Compliant Watermarking for Physical Design IP Protection: From Placement to Routing

**arXiv ID:** 2608.05055 | [PDF](https://arxiv.org/pdf/2608.05055v1)

**作者:** Andrew B. Kahng `[一作]` (University of California at San Diego), Yiting Liu `[通讯]` (University of California at San Diego)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套在放置、时钟树合成和布线三个阶段均可嵌入的物理设计水印框架 PDMarks，满足 Kerckhoffs 原则的 IP 保护。

**💡 创新点**

创新点在于：① 将所有阶段的水印信息统一绑定至单一 32 字节密钥，利用 HMAC‑SHA256 产生独立阶段密钥和水印实例；② 通过本质自由度（单元顺序、时钟缓冲区的奇偶性、布线的错误方向比）嵌入水印，既不破坏设计质量也难以被白盒攻击者定位；③ 在 OpenROAD 流程中直接嵌入 Hook，支持仅读取布局的后验验证。

**🔧 技术方法**

采用 HMAC‑SHA256、AES‑256‑GCM、时间戳承诺、统计检验（Welch 检验）等技术；在放置、CTS、布线三阶段分别实现不同的嵌入/提取算法。

**📊 数据集**

实验基准为 NanGate45 与 ASAP7 的八个开源 benchmark（JPEG、SweRV、Ariane、BP、CVA6 等），所有实验均在 OpenROAD‑flow‑scripts 中完成。

**📈 对比分析**

与 Row‑Parity、Buffer‑Insertion、ICMarks 等传统水印方法相比，PDMarks 在 1% 以下的 PPA 额外成本下实现了更低的碰撞概率（10⁻³² 以上）和更高的提取率（≥ 0.96）；在盲攻击、定向攻击和错误密钥检验下均保持显著优势。

**⚠️ 局限性**

局限性：① 仅保护已完成的物理实现，无法阻止攻击者重新生成全新布局；② 需要在设计流程中插入 Hook，若流程不可改动则难以部署；③ 受限于所选的自由度，若技术节点或工艺约束变化，水印容量与鲁棒性可能需要重新调参。

---

## 533. The Loss Does Not See the Basis, but Adam Does

**arXiv ID:** 2608.05136 | [PDF](https://arxiv.org/pdf/2608.05136v1)

**作者:** Devender Singh `[一作]` `[通讯]` (Memorial University of Newfoundland), Devender Singh (Memorial University of Newfoundland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过理论分析和实验证明：在因子化模型（如低秩矩阵感知、Transformer 的注意力头等）中，梯度下降等“等尺度”优化器保持了模型的等变性（gauge equivariance），从而在训练到全拟合（interpolation）后偏好低秩解；而标准 Adam 等按坐标适应的优化器会破坏这种等变性，导致选择不同的插值点，产生更高的恢复误差。

**💡 创新点**

创新点在于：①引入 gauge-equivariance 作为判定优化器是否保持低秩偏好的判据；②提出“结构定理”将无记忆等尺度更新规则等价为 Gram 矩阵预调节器；③通过“dial”逐渐消除 Adam 的坐标异otrop性，展示恢复误差随 p 从 1 降到 0 单调下降；④绘制了 Muon 等等率预调节器在目标谱尾能量变化下的相位图，解释何时该预调节器有利或有害；⑤在 Transformer、矩阵感知、以及真实的 hyperspectral 数据集上验证该机制。

**🔧 技术方法**

主要技术包括：
- gauge symmetry（U, V ↦ UQ, VQ）与其在梯度更新中的协变性分析；
- 记忆无关的等尺度更新规则的结构定理（Δ = H(GGᵀ)G）；
- Adam‑p 预调节器（p ∈ [0, 1]）的设计与实验；
- Muon 的等速预调节器、Shampoo 的 Kronecker 预调节器、梯度流的时间变换传递定理；
- 对矩阵感知任务的恢复误差和有效秩评估；
- Transformer 双模型对照实验（gauge 变换下的双模型差异度量）。

**📊 数据集**

使用的数据集与任务包括：
- 低秩矩阵感知（n = 40、r = 3、m = 2 dof）作为基准实验；
- 两个 hyperspectral 图像完备任务（Indian Pines 与 Pavia University），随机选取 2000 个像素、约 200–100 + 1 维度，采用 48 维因子化；
- 小型 Transformer（4‑head、2‑层）与字符级语言模型（6‑层）用于 gauge 依赖性验证。

**📈 对比分析**

比较方法：在所有优化器都训练到训练误差 < 10⁻⁷ 的插值点，收集恢复误差、有效秩、梯度平衡度；对比“等尺度”与“按坐标”两大类。实验结果显示：
- 等尺度优化器（GD、scalar‑Adam、Muon、Shampoo）恢复误差 ≤ 0.286，Adam 等按坐标者 ≥ 0.42；
- Adam‑p 随 p 降低 1→0 时恢复误差从 0.57 降至 0.20，效秩从 14.5 降至 5.4；
- Muon 在无谱尾（τ = 0）时可精确恢复，随着 τ 增大到约 0.2 时被 GD 超越；
- 在真实 hyperspectral 数据上，GD 在匹配训练误差时的 hold‑out RMSE 比 Adam 高出约 43 %–28 %（取决于采样密度），且有效秩显著更低。

**⚠️ 局限性**

局限性：
- 研究范围局限于无权重衰减、确定性全批训练，未覆盖大规模、随机批次或带正则化的场景；
- 仅对等尺度更新规则的“最优”程度给出经验阈值（如 Muon 的等速预调节器），缺乏理论上可调的显式偏差强度参数；
- 对梯度流的传递定理仅适用于无记忆共标量流，对带动量或状态的优化器的分析仍未完全；
- 实验在固定预算和固定学习率网格下完成，未进行全面的超参数搜索，结果可能受调参策略影响；
- 机制对非低秩或高度非线性任务的适用性尚未验证。

---

## 534. Reasoning Core: Designing Broad Procedural Data for Completion-Supervised Reasoning Training

**arXiv ID:** 2608.05148 | [PDF](https://arxiv.org/pdf/2608.05148v1)

**作者:** Damien Sileo `[一作]` (Univ Lille), Dimitri Kachler `[通讯]` (Univ Lille)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个包含50个可验证推理生成器的库，用于在完成监督（SFT）下训练大语言模型。

**💡 创新点**

创新点在于将可验证程序化生成器设计为统一接口，兼顾SFT与RL需求，并系统评估生成器的可训练性、难度控制和目标规范化。

**🔧 技术方法**

技术包括语义评分器、可调节难度、统一的任务接口、外部求解器集成、Docker化环境、以及模型辅助审计与人工复核流程。

**📊 数据集**

数据集主要是自生成的推理实例，覆盖数学、逻辑、规划、状态追踪、形式语言、结构化数据、游戏、因果推断与代码等九大领域，并与FineWeb-Edu、DOLCI等主流文本数据混合使用。

**📈 对比分析**

对比方法：将该库与三大已有程序化集合在四种基模型（SmolLM2-135M、SmolLM2-360M、OLMo-1B、SmolLM3-3B）和多种训练时长下进行匹配SFT，评估在DROP、LogiQA、ARC-Challenge、BBH等推理基准上的负对数似然下降。结果显示，3B模型下该库在Reasoning指标上平均提升约0.84个百分点，整体优于其他三组集合。

**⚠️ 局限性**

局限性包括仅验证至3B规模，未探讨更大模型；主要关注闭合答案推理，缺乏开放式、多模态或真实环境任务评估；未测试SFT后续RL的协同效果；并且对比RL时仅用单一随机种子，结果仅示范性。

---

## 535. Argus: A General-Purpose Agentic Runtime for Long-Horizon Reasoning

**arXiv ID:** 2608.05144 | [PDF](https://arxiv.org/pdf/2608.05144v1)

**作者:** Boxiu Li `[一作]` (Microsoft), Zhijie Deng `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种持久化、可验证的代理运行时（Agentic Runtime），通过角色分工（Manager、Planner、Engineer、Reviewer）实现长期研究任务的持久化、可审计、可演化，并在多个基准上实现自动化执行。

**💡 创新点**

创新点包括：①将长期推理拆分为“持久化状态 + 角色分工 + 验证门控”三大模块，避免无声漂移；②使用“验证门控的固定模型自我演化”机制，使得模型权重保持不变，但运行时状态可累积、可回溯；③在同一跑中实现“自动化迭代、审计、回滚”全过程，形成可复用的研究轨迹。

**🔧 技术方法**

技术手段：①基于大型语言模型（GPT‑5.5、Codex、Claude）与外部工具交互；②通过“ManagerAdmit”投影实现合约更新；③角色分工实现任务划分、执行、审计；④持久化存储（checkpoint、event log、skills、verifiers、routing）与回滚机制；⑤在不同基准中嵌入自定义验证器、评测脚本。

**📊 数据集**

使用的评测数据集/任务包括：SWE‑Bench Pro（软件修复）、SOL‑ExecBench（GPU kernel 优化）、nanochat B200/H100（模型训练）、nanoGPT speedrun（训练加速）、AARRI‑Bench（研究助理任务）、Math‑Reasoning Data（数学问题生成）以及正在进行的GLM‑5.2/Claude‑Code实验。

**📈 对比分析**

对比方法：与直接 Copilot（GPT‑5.5/xhigh）相比，在 SWE‑Bench Pro 上达成约78% 正确率，代价为1.41× Token；在 SOL‑ExecBench、nanochat、AARRI‑Bench 等基准中均超过或接近现有最佳结果；在长期跑中，成熟阶段相比启动阶段，solve‑input Token 下降21%，主动工作时间下降15%，表明自我演化带来显著资源节省。Reviewer 路径的引入实现了约43% 的任务需要独立审计，恢复率为34% 通过官方验证，22% 实现严格恢复。

**⚠️ 局限性**

局限性：①实验结果为观测性，缺乏因果验证（未做随机化或冻结状态对比）；②验证门控仅依赖现有验证器，若验证器自身错误则无法纠正；③Manager 的合约更新仍可能被误批准导致目标漂移；④未评估零触摸（zero‑touch）率；⑤多任务序列与资源消耗未完全分离，难以精确归因；⑥外部评测环境（如 GPU、语言模型、数学证明）差异导致跨基准比较不一致。

---

## 536. Predicting Brain Morphometry with MT-GNN: Mesh Evolution in Continuous Time with Graph-Based Metric Tensor Embeddings

**arXiv ID:** 2608.05132 | [PDF](https://arxiv.org/pdf/2608.05132v1)

**作者:** Hao Ding `[一作]` (Illinois Institute of Technology), Boris Gutman `[通讯]` (Illinois Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出并实现了一种连续时间的子皮层表面预测模型 MT‑GNN，利用前置扫描序列预测未来任意时间点的内在度量张量，并通过可微分的 As‑Rigid‑As‑Possible 重新解码为表面形状。

**💡 创新点**

创新点包括：① 用内在度量张量（第一基本形式）而非外部顶点位移进行预测；② 采用 log‑Euclidean 均值基底 + 学习的基移 + 个体残差的分解；③ 通过重建约束（MVE‑through‑ARAP）而非直接的度量空间损失来训练，起到软可实现性约束；④ Fourier 时序编码实现任意预测时长的条件化。

**🔧 技术方法**

使用的技术包括：基于 MeshConv 的图神经网络、SPD(2) 的 log‑Euclidean 参数化、Fourier 时序嵌入、可微分 ARAP 求解器、基于顶点误差的训练目标，以及可选的曲率头。

**📊 数据集**

数据集为 ADNI 纵向 T1‑MRI，使用 FreeSurfer 8 提取 14 个基底核结构（左右侧纹状体、尾状核、壳核、苍白球、海马、杏仁核、尾状核底部）的注册固定拓扑网格；每个样本由 3 次访视组成，未来目标设置在 12、24、36、48 个月后。

**📈 对比分析**

与传统时间平均、几何回归（DCM）以及面张量 Transformer（TransforMesh）进行比较。MT‑GNN 在所有时长上均显著低于时间平均（平均降低 2.29%），并在 14 个结构中均优于两类基线；曲率变体 MT‑GNN+H 在 4 个结构上进一步提升，整体误差保持在 0.6–0.7 mm 级别，处于重建可接受范围内。

**⚠️ 局限性**

局限性包括：① 访视间形变极小，接近重建噪声底，易被噪声放大；② 长时程验证样本稀缺，统计置信度较低；③ 可实现性约束仅为软约束，未提供精确的 Gauss‑Codazzi 投影；④ 仅训练单独结构模型，缺乏统一的多结构框架；⑤ 仅在离散 12 个月网格上监督，离散化外的时长仍未充分验证。

---

## 537. SmartMage: Dynamic Modality Orchestration for 3D Scene Understanding

**arXiv ID:** 2608.05137 | [PDF](https://arxiv.org/pdf/2608.05137v1)

**作者:** Yue Zhang `[一作]` (Zhejiang University), Hehe Fan `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种统一的3D场景多模态大语言模型SmartMage，能够在每个查询中动态选择有用模态并将模态信息分配给专门化专家，实现语义感知的多模态推理。

**💡 创新点**

创新点包括：① SMART模块基于语义先验、语义相似度与模态质量进行全局动态模态路由；② MAGE模块利用模态先验的专家推测，在稀疏Mixture‑of‑Experts框架中实现模态感知的专家分配，从而实现更精准的模态专属推理。

**🔧 技术方法**

采用的技术有：3D特征提取（RGB‑D、BEV、点云、体素）、语义先验估计、语义相似度评分、模态质量评估、稀疏Mixture‑of‑Experts、软硬路由、文本‑模态对齐、LLM微调与正则化损失等。

**📊 数据集**

使用的数据集：基于ScanNet构建的统一训练集；评测包括五个3D场景基准（ScanQA、SQA3D、Scan2Cap、ScanRefer、Multi3DRefer）、RGB‑only视频基准（VSI‑Bench、VSI‑SUPER、MMSI‑Bench）以及自定义诊断基准ScanFacet。

**📈 对比分析**

与多种SOTA方法对比，SmartMage在所有5个3D基准上均取得领先或相当的表现：ScanQA Acc@0.5提升5.1点，ScanRefer F1@0.5提升6.4点，Scan2Cap CIDEr@0.5提升8.9点；在RGB视频基准上虽略低于专门的2D视频模型，但仍保持竞争力。

**⚠️ 局限性**

局限性：预处理高分辨率多模态输入耗时较大；在单一RGB输入下仍受限于训练时多模态的经验，性能未能达到最优；训练成本相对较高，且在极稀疏场景下可能出现专家崩塌问题。

---

## 538. SSTQ:Privacy-Preserving Vector Quantization via Subsampled Stochastic TurboQuant

**arXiv ID:** 2608.05127 | [PDF](https://arxiv.org/pdf/2608.05127v1)

**作者:** Adel Javanmard `[一作]` (University of Southern California), Vahab Mirrokni `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Subsampled Stochastic TurboQuant (SSTQ)框架，用于在分布式优化中实现局部差分隐私且通信成本低；

**💡 创新点**

创新点在于结合过完备等范数紧致框架、坐标子采样和一维隐私感知量化，优化代码簿实现了O(2^b)的均方误差标度，并在高位宽下引入度量感知拉普拉斯机制；

**🔧 技术方法**

使用了等范数紧致帧、Kashin表示、随机化响应/拉普拉斯隐私机制、凸优化代码簿、傅里叶/Hadamard变换等技术；

**📊 数据集**

在CIFAR-10和Fashion‑MNIST两个图像分类数据集上进行实验；

**📈 对比分析**

与PrivUnit、vqSGD、SQKR等基线进行比较，SSTQ在相同隐私预算下实现约3倍更低的位传输同时保持与SQKR相近或略优的准确率；

**⚠️ 局限性**

局限性包括需预先设定通信预算和数据范围、对Kashin变换计算量大、以及在高维或大位宽下仍存在方差/偏差折中。

---

## 539. Objects as Audio-Visual Modal Sound Fields

**arXiv ID:** 2608.05145 | [PDF](https://arxiv.org/pdf/2608.05145v1)

**作者:** Zisen Shao `[一作]` (University of Maryland), Ruohan Gao `[通讯]` (University of Maryland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于多视角图像和少量撞击声录音的对象级音频‑视觉模态声场表示，可在任意接触位置合成冲击声，并支持接触定位与声音编辑。

**💡 创新点**

创新点包括：① 将 3D Gaussian Splatting 与稠密视觉特征相结合得到几何感知视觉先验；② 用物理意义的模态参数（频率、阻尼、位置依赖增益）构造声场，既可解释又能在极少样本下学习；③ 结合视觉对齐、残差噪声建模等机制，显著提升少样本学习效果。

**🔧 技术方法**

使用的技术包括：3D Gaussian Splatting、DINOv2 视觉编码、对称对齐特征、线性模态分析、差分合成、隐式位置依赖增益场、STFT 损失、多尺度训练、Audio‑SDS、注意力机制等。

**📊 数据集**

实验数据集为 ObjectFolder Real 与 RealImpact 两个真实多感知数据集。

**📈 对比分析**

与白噪声、随机冲击、KNN、DiffSound（基于逆渲染的物理方法）和 SonicGauss（生成式方法）进行对比；在两个数据集上均取得更低的 L1、L1‑log、Envelope、CDPAM 等指标，并在非对称物体上明显优于 KNN；在接触定位和声音编辑任务中也取得更优的 RMED 与 UMAP 距离。

**⚠️ 局限性**

局限性在于仅针对单一均匀材料对象，缺乏对多材料或非均匀结构的泛化；需要多视角图像与少量冲击录音；物理假设（阻尼不随位置变化）在部分对象上不成立；高度对称物体仍可能导致定位失败。

---

## 540. OctoLong: Mid-Training On Cross-Repository Code Contexts Enhances Long-Context Modeling

**arXiv ID:** 2608.05141 | [PDF](https://arxiv.org/pdf/2608.05141v1)

**作者:** Indraneil Paul `[一作]` (TU Darmstadt), Iryna Gurevych `[通讯]` (TU Darmstadt)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了跨仓库代码依赖检索流水线，利用AST+LSP+包管理器递归抓取实现，生成约6.2B标记、依赖密集的长上下文数据并用于训练长上下文语言模型

**💡 创新点**

创新点在于首次通过跨仓库递归检索实现生成前所未有的高度依赖密集长上下文，并将其作为中期训练（LCFT）数据，大幅提升长距离检索、状态追踪和代理任务性能

**🔧 技术方法**

采用AST查询、Language Server Protocol、容器化环境、BFS依赖图遍历、ABF‑RoPE扩展、模型合并以及后续的指令调优（SFT）等技术

**📊 数据集**

数据集包括约6.2B Token 的跨仓库代码上下文（OctoLong），与传统长文本混合构成约50B Token 的 LCFT 语料，再与约10B Token 的指令/代理数据混合用于 SFT

**📈 对比分析**

与18个支持至少64K上下文的开源基线及三种消融实验对比，取得在代码检索、长对话、工具调用等任务上显著提升，长上下文128K兼容且短上下文性能基本不下降

**⚠️ 局限性**

局限在仅限 Python、未覆盖多语言与 ABI/FFI 依赖、上下文长度上限为128K、未包含低资源语言或更大规模模型

---

## 541. OPD-V: Visual On-Policy Self-Distillation with Modality Balance

**arXiv ID:** 2608.05131 | [PDF](https://arxiv.org/pdf/2608.05131v1)

**作者:** Aniri `[一作]` (Ludwig Maximilian University of Munich), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出 OPD‑V，一种利用模态平衡作为特权信息进行多模态大语言模型后训练自蒸馏的方法，显著提升视觉推理性能；

**💡 创新点**

创新点在于将模型内部视觉‑文本注意比例（模态平衡）抽象为隐式特权信息，通过正负教师（缩放图像与遮挡图像）对照形成信任区间，实现在自蒸馏过程中的局部监督；

**🔧 技术方法**

采用了 On‑Policy Self‑Distillation、Jensen‑Shannon 归一化蒸馏、双教师对照策略、模态平衡注意比与对比分数、EMA 维护教师参数、top‑K logits 近似等技术；

**📊 数据集**

使用 Vision‑OPD 提取的 6.2k 视觉推理样本进行训练，并在 V* Bench、ZoomBench、HR‑Bench、MME‑RealWorld 等六个基准上评测；

**📈 对比分析**

与标准 OPSD、Vision‑OPD、VA‑OPD、SFT、GRPO 等方法对比，在 Qwen3.5‑4B 上平均准确率从 64.30% 提升至 80.01%（+15.7pp），在多模型、多尺度下保持一致提升，并在训练时延上减少约 25‑32%，响应长度缩短至原来的约 25%；

**⚠️ 局限性**

局限性包括：模态平衡阈值对结果敏感，需要针对不同任务优化正负教师的图像变换；方法对特定视觉任务的泛化尚未充分验证；

---

## 542. CoCo-IR: Contextual Composed Image Retrieval

**arXiv ID:** 2608.05149 | [PDF](https://arxiv.org/pdf/2608.05149v1)

**作者:** Shengcao Cao `[一作]` (University of Illinois Urbana-Champaign), Liang-Yan Gui `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种多轮交互式的上下文组合图像检索（Contextual Composed Image Retrieval）任务，并设计了Transformer架构的Transformable Image Embedding模型，实现对多轮图文交互历史的持续推理与嵌入生成。

**💡 创新点**

创新点包括：①将多轮交互历史作为统一token序列输入，使用专门的⟨CTX⟩token作为全局信息瓶颈；②在每轮内部使用双向注意力、跨轮使用因果注意力的混合机制；③构建全自动、LMM驱动的数据引擎，利用自我反思和硬负样本验证自动生成高质量多轮训练数据。

**🔧 技术方法**

核心技术包括：大型多模态模型（Gemma3、LLaVA等）作为编码器；InfoNCE对比学习目标；双向+因果混合注意力；特殊嵌入token；自监督的数据生成与硬负样本挖掘。

**📊 数据集**

使用了公开的单轮检索基准（FIQ、CIRR、CIRCO）进行单轮评测；并在作者新构建的多轮对话检索数据集上进行评估，该数据集通过LMM自动生成、硬负样本验证后形成的4轮对话链。

**📈 对比分析**

与传统单轮检索模型（MagicLens、E5-V、BGE-VL等）以及使用不同适配策略（拼接指令、最近输入、Gemini摘要）的对比，作者模型在单轮任务上取得了39.4 mAP@5（CIRCO）并在4轮多轮任务中实现了R@1 44.1%，远超对比模型的28.2%，显著提升了多轮检索性能。

**⚠️ 局限性**

局限性包括：①模型仍需依赖大规模LMM，计算资源成本高；②多轮对话数据仍以自动生成为主，缺乏真实用户交互场景的多样性；③评测指标严格的累积召回可能导致对真实可迭代搜索体验的低估。

---

## 543. Toward Skill-Native LLMs: Skill Entropy for Benchmarking and Training Long-Horizon Reasoning

**arXiv ID:** 2608.05139 | [PDF](https://arxiv.org/pdf/2608.05139v1)

**作者:** Yinghui He `[一作]` (Princeton University), Sanjeev Arora `[通讯]` (Princeton University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究并量化跨技能长周期推理中的技能切换难度，构建 Skill‑Entropy 指标，发布基准 Benchmark，并提出 Skill‑Entropy RL 通过奖励信号提升模型跨技能表现。

**💡 创新点**

引入有向对称 Skill‑Entropy 量化技能切换难度，形成可衡量的基准，并将同一指标用作强化学习奖励，显著提升跨技能推理能力。

**🔧 技术方法**

采用 LLM 生成、强化学习（GRPO）结合答案与技能熵奖励的训练框架，以及多步骤对齐评估和技能标签聚类。

**📊 数据集**

9 个域的 558 个技能基准，使用 OpenR1‑Math、MMLU、MMLU‑Pro、LiveCodeBench、ZebraLogicBench、WikiTable、WebSRC、NaturalPlan、Creative Writing 等数据集。

**📈 对比分析**

与 8 大前沿模型和 4 个开源模型对比，发现准确率随任务技能熵升高而下降；Skill‑Entropy RL 在 Qwen3‑4B‑Instruct 上将 34.4% 提升至 68.4%，在多域与公开基准上均优于传统 SFT/GRPO 方法。

**⚠️ 局限性**

依赖手工或 LLM 自动生成的技能标签，评估仍受限于预设的 9 个域，技能切换错误仍在后续步骤显著；对极长任务或高维技能空间的扩展尚未验证。

---

## 544. DeepConnect: A Visual Analytics System for Bridging Interdisciplinary Research Collaborations

**arXiv ID:** 2608.05134 | [PDF](https://arxiv.org/pdf/2608.05134v1)

**作者:** Yingchaojie Feng `[一作]` (National University of Singapore), Anthony K. H. Tung `[通讯]` (National University of Singapore)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个名为DeepConnect的LLM增强可视化分析系统，帮助研究者将跨学科合作意图转化为可检索任务，检索文献并评估潜在合作者。

**💡 创新点**

创新点在于将LLM生成的任务拆解与可视化的时间线、专家匹配评分、术语对比以及基于出版物的对话模拟相结合，实现从意图到候选人再到沟通准备的闭环流程。

**🔧 技术方法**

采用GPT-5.1与GPT-4.1-mini进行任务生成和对话模拟，使用all‑MiniLM‑L6‑v2嵌入进行文本相似度匹配，结合OpenAlex的机构、作者、作品和领域实体进行数据处理。

**📊 数据集**

使用OpenAlex公开学术元数据，并在此基础上构建了本校内部子集（涵盖机构、作者、论文和领域），用于案例与用户研究。

**📈 对比分析**

通过两项案例研究、12名研究者的使用实验和组件级评估，对比了基于时间权重的任务匹配、仅相似度和出版量三种排名方式，结果显示时间加权方法在任务适配和专业更新度上均达到4.5以上的满意度。

**⚠️ 局限性**

局限包括仅在前期探索阶段评估，未跟踪合作结果；依赖出版物数据作为专业代理，可能忽视非文献贡献；系统对LLM生成的推理仍存在幻觉风险。

---

## 545. Spoken Function Calling: A New Perspective on Spoken Language Understanding for Large Audio Language Models

**arXiv ID:** 2608.05126 | [PDF](https://arxiv.org/pdf/2608.05126v1)

**作者:** Yuezhang Peng `[一作]` (Shanghai Jiao Tong University, Token Foundry, Alibaba Group), Xie Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现基于功能调用的语音理解框架SFC，构建SFC‑Bench数据集，并在大型音频语言模型上进行实验比较及后训练提升；

**💡 创新点**

创新点在于将语音语义解析转为结构化功能调用，利用函数定义规约歧义，构造多级多意图多轮任务数据集，并引入细粒度奖励的RL后训练；

**🔧 技术方法**

使用多代理生成、LLM与大型音频LLM（Qwen、Gemini、GPT‑4o）结合GRPO强化学习、Fine‑grained奖励、LoRA微调以及Whisper ASR；

**📊 数据集**

采用改造的SLU基准（ATIS、SNIPS、FSC、SLURP、MAC‑SLU）生成300个功能，SFC‑Bench包含约7k+样本，划分ID/OOD；

**📈 对比分析**

通过意图准确率、槽值F1和整体准确率对比，SFC在所有模型上比传统SLU提升5–18%，SFC‑7B在SFC测试中整体准确率超过80%，甚至超过GPT‑4o‑Audio；

**⚠️ 局限性**

局限在于高难度多意图/多轮任务仍易失真，易产生hallucination，受ASR误差影响大，未知域泛化不足，RL训练成本高且对模型稳定性有挑战。

---

## 546. Chained Recursive Language Models for Multi-Iteration Reasoning

**arXiv ID:** 2608.05124 | [PDF](https://arxiv.org/pdf/2608.05124v1)

**作者:** Purbesh Mitra `[一作]` (University of Maryland), Sennur Ulukus `[通讯]` (University of Maryland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Chained Recursive Language Models（Chained RLM）推理架构，利用同一 LLM 在多次“根”调用中分阶段完成长上下文推理，核心思路是通过纯文本摘要、黑板和持久化 artifact 传递中间状态，避免一次推理中出现上下文衰退。

**💡 创新点**

创新点在于：①将推理拆分为多个独立根调用，避免一次长推理导致的状态混乱；②使用简洁的纯文本手风（handoff）格式而非结构化 JSON，降低对系统解析的依赖；③引入 artifact 工作区，让后续根可以读取、修正、审核前一步生成的中间结果，实现类似分阶段分析的可检查推理链。

**🔧 技术方法**

技术细节包括：基于 GPT‑5‑mini LLM；工具调用与 Python REPL 环境；链式手风文本格式（Summary、Blackboard、Next）；黑板和 artifact 的持久化存储；自定义系统提示；评估指标为 Pass@1 准确率以及平均根调用数、手风次数、输入/输出 token 数和成本。

**📊 数据集**

实验使用四个长上下文推理基准：RULER、BABILong、LongBench v2、OOLONG‑real。

**📈 对比分析**

与单一调用 LLM 基线（同一模型）进行对比。结果显示 Chained RLM 在所有四个基准上的 Pass@1 准确率均有提升，平均提升约 13.8%（RULER 87→92%，BABILong 44→59%，LongBench v2 41→52%，OOLONG‑real 14→38%）。同时，平均根调用数、手风次数、token 数和成本均有所上升，约为原基线的 2.5‑3 倍。

**⚠️ 局限性**

局限性包括：①未强制根始终读取 artifact，可能导致过早给出答案；②artifact 的质量高度依赖模型，错误结构可能被继承而无法纠正；③链式过程可能漂移，后续根可能忽略优秀 artifact 并重新开始；④计算成本显著增加，需在准确率和成本之间权衡。

---

## 547. IRIS: A Visual Cortex-Inspired Framework for Analyzing Orientation Selectivity in Vision Transformers

**arXiv ID:** 2608.05122 | [PDF](https://arxiv.org/pdf/2608.05122v1)

**作者:** Vaishnavi B Mohan `[一作]` (University of Washington), Shashank Hegde `[通讯]` (Nvidia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究 Vision Transformer 在不同预训练范式、规模与训练阶段下是否出现类似 V1 的方向选择性，并提出基于神经科学的方向编码度量工具 IRIS；

**💡 创新点**

创新点在于将方向选择性指标（RSS、ORS、OSI、HWHM）系统化应用于 ViT，揭示预训练目标是决定方向选择性的主因，并发现 RSS 曲线峰值可预测最优微调层数；

**🔧 技术方法**

使用合成正弦栅格刺激、冻结 ViT 各层激活收集，对残差流、MLP 神经元和稀疏自编码器特征分别计算 RSS、ORS、OSI、HWHM，并通过层级对齐与 LoRA 微调等技术进行分析；

**📊 数据集**

采用多种公开预训练 ViT（OpenCLIP、DeiT III、DINOv2/3、MAE、AIMv2）的检查点进行测试，刺激为合成正弦栅格，且在 Taskonomy 任务集上评估下游性能；

**📈 对比分析**

通过对不同模型的 RSS、ORS、宽度、深度等指标进行比较，并在 9 个下游任务中进行线性/1×1 conv 评估，发现 RSS 曲线与下游泛化高度相关，峰值能准确预测最佳微调起点；DINOv3 的方向编码最稳定，AIMv2 最早衰减，MAE 维持至最后层；

**⚠️ 局限性**

局限性在于仅用合成栅格刺激评估方向选择性，未考虑自然图像中更复杂的方向处理，也仅聚焦方向特征，未扩展至空间频率、色彩对比、运动方向等 V1 基础特征。

---

