# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-04-07 | 今日论文总数: 894

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Don't Waste Bits! Adaptive KV-Cache Quantization for Lightweight On-Device LLMs

**arXiv ID:** 2604.04722 | [PDF](https://arxiv.org/pdf/2604.04722v1)

**作者:** Sayed Pedram Haeri Boroujeni `[一作]` (Clemson University), Abolfazl Razi `[通讯]` (Clemson University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于数据驱动的 KV‑缓存自适应量化框架，利用轻量级 token 重要性特征动态为每个 token 分配 2/4/8‑bit 或 FP16 精度，以降低 LLM 解码过程中的内存占用与延迟。

**💡 创新点**

创新点在于：① 将 token 的频率、注意力方差与熵等信息融合为四维特征；② 设计了一个仅含两层 ReLU 的 MLP 控制器，可在线实时预测每个 token 的量化位宽；③ 通过联合交叉熵、期望延迟和质量惩罚的多任务损失，平衡精度与效率；④ 对比传统的固定 4‑bit、规则驱动与 FP16，证明该自适应策略在多模型、多数据集上显著提升了准确性与速度。

**🔧 技术方法**

核心技术包括：token 重要性特征提取（熵、罕见度、注意力方差、置信度）、轻量级 MLP 控制器、按位宽量化（2/4/8‑bit 与 FP16）以及基于期望延迟/质量的训练目标。

**📊 数据集**

使用 SmolLM-135M、SmolLM-360M、SmolLM-1.7B 三种规模模型，在 HellaSwag、OpenBookQA 与 ARC‑Challenge 三个常识推理/科学推理基准上进行评估。

**📈 对比分析**

与 FP16、静态 4‑bit 量化、规则驱动动态量化三种基线进行对比；在所有规模上都实现了比静态/规则方案更高的准确率，且与 FP16 相距 0.3‑1.0 点，同时将解码延迟降低约 20%~30%，显著提升准确性-延迟 Pareto 前沿。

**⚠️ 局限性**

局限性包括：① 仅在中小规模 SmolLM 模型上验证，尚未在大规模 LLM 上测试；② 需要额外训练一个控制器，增加模型构建成本；③ 对不同硬件平台的具体实现细节与瓶颈（如量化运算支持）仍待进一步验证。

---

## 2. Design Guidelines for Game-Based Refresher Training of Community Health Workers in Low-Resource Contexts

**arXiv ID:** 2604.04671 | [PDF](https://arxiv.org/pdf/2604.04671v1)

**作者:** Arka Majhi `[一作]` (IIT Bombay), Satish B. Agnihotri `[通讯]` (IIT Bombay)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在印度开展了为期四年的游戏化CHW（社区卫生工作者）再培训系统的设计、部署与评估，随后对四个迭代周期内的定性访谈、现场观察和使用日志进行混合方法综合，提炼出八条面向低资源环境的设计准则。

**💡 创新点**

首次将游戏化训练与现场工作情境、专业身份、伦理透明度和混合物理-数字交互等多维度因素系统化，形成可迁移的设计框架，弥补了以往仅关注短期学习效果的研究空白。

**🔧 技术方法**

采用设计研究法（Design‑Based Research）、混合方法分析（主题分析、轴向编码）、定量日志跟踪（使用频率、完成率、错误模式）以及质性软件NVivo进行编码。

**📊 数据集**

收集了80名CHW（包括ASHAs和AWWs）在四个系统部署期间的访谈记录、现场观察笔记和系统使用日志，数据涵盖了多语言采访文本、任务完成事件及匿名位置信息。

**📈 对比分析**

通过跨周期的主题对比和日志统计（如持续参与率、完成率变化）来验证设计准则的有效性；结果显示采用渐进式复杂度、情境真实性、混合交互和协作机制的系统在使用频率和满意度上显著高于固定难度或高度竞争化的版本。

**⚠️ 局限性**

局限性包括样本仅来自印度农村和半城镇，缺乏长期健康结果评估，数据多为自述和观察，未与传统培训方式进行对照实验，且系统尚未实现大规模机构化部署，难以评估可扩展性与维护成本。

---

## 3. Visual Prompt Based Reasoning for Offroad Mapping using Multimodal LLMs

**arXiv ID:** 2604.04564 | [PDF](https://arxiv.org/pdf/2604.04564v1)

**作者:** Abdelmoamen Nasser `[一作]` (Khalifa University), Majid Khonji `[通讯]` (Khalifa University)

**通讯引用:** 664 | [OpenAlex ID](https://openalex.org/A5056369727)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种零样本离地导航框架，利用SAM2进行分割并通过VLM推理识别可行驶区域，随后结合全局与局部规划实现端到端控制。

**💡 创新点**

创新点在于将分割与视觉语言推理整合为单一管线，借助VLM的语义推理替代传统多模型方法，做到无需专门训练即可识别多种地形。

**🔧 技术方法**

使用技术包括：SAM2分割、文本-图像多模态VLM（ChatGPT‑4o、GPT‑5‑Mini等）、图像拼接提示、D* Lite+Hybrid A*规划、Stanley+PID控制。

**📊 数据集**

数据集涵盖：自建的Isaac Sim仿真环境及其收集的957张RGB图像；在真实离地数据集ORFD、RUGD、O2DTD上进行评估。

**📈 对比分析**

通过与Mask2Former、PathFormer等现有分割模型在IoU上的对比，零样本管线在高分辨率数据集上与SOTA相当或更优；仿真中对三目标的到达率分别为100%、100%和40%，VLM配置对成功率影响显著。

**⚠️ 局限性**

局限性包括VLM推理结果不确定、低分辨率或缺乏明显可行路径的数据集上性能下降，以及缺少多尺度特征导致对复杂地形识别不足。

---

## 4. When Adaptive Rewards Hurt: Causal Probing and the Switching-Stability Dilemma in LLM-Guided LEO Satellite Scheduling

**arXiv ID:** 2604.03562 | [PDF](https://arxiv.org/pdf/2604.03562v1)

**作者:** Yuanhang Li `[一作]` `[通讯]`, Yuanhang Li

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在多光束LEO卫星排程中，研究了自适应奖励设计与LLM-DRL集成，发现奖励权重的稳定性比权重质量更重要，并通过单变量因果探测揭示奖励空间的关键影响因素；

**💡 创新点**

提出了“切换-稳定性困境”概念，证明PPO需要近似平稳的奖励信号；设计单变量因果探测方法快速定位奖励权重对性能的因果影响；提出三时标混合架构，将LLM用于自然语言意图理解，MLP负责实时数值权重映射；

**🔧 技术方法**

采用PPO深度强化学习、LLM（Qwen3-4B + LoRA）、MLP、规则基、CUSUM检测、RAG检索、单变量因果探测；

**📊 数据集**

使用基于仿真生成的LEO卫星多光束流量数据，涵盖城市、海事、灾害、混合四种已知流量模式以及三种新型未见模式（IoT突发、极地切换、冷热极差）；

**📈 对比分析**

四种架构（固定、规则、MLP、LLM）在已知和未知流量模式下对比，MLP在已知模式下达到最高吞吐357.9 Mbps，在未知模式下325.2 Mbps；LLM因权重波动导致平均45.3 Mbps；性能差异主要源自奖励权重的稳定性；

**⚠️ 局限性**

限制包括：LLM输出不稳定导致权重振荡；RL需要近似平稳奖励，导致自适应奖励难以直接使用；LLM推理延迟高；实验仅在仿真环境，缺乏真实卫星部署验证；未解决多任务RL的非平稳问题。

---

## 5. Empowering Power Outage Prediction with Spatially Aware Hybrid Graph Neural Networks and Contrastive Learning

**arXiv ID:** 2604.04916 | [PDF](https://arxiv.org/pdf/2604.04916v1)

**作者:** Xuyang Shen `[一作]` (University of Connecticut), Dongjin Song `[通讯]` (University of Connecticut)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Spatially Aware Hybrid Graph Neural Network (SA‑HGNN)，用于预测极端天气导致的电力停电。

**💡 创新点**

创新点包括动态图学习模块、静态与动态特征分支的混合卷积以及对不平衡数据的对比学习正则化。

**🔧 技术方法**

使用了图神经网络、动态邻接学习、对比学习以及Huber损失等技术。

**📊 数据集**

实验基于联邦能源公司Eversource在康涅狄格州、西马萨诸塞州、东马萨诸塞州和新罕布什尔州收集的约1,400个极端天气事件的约4,000个位置的高维（390）特征数据。

**📈 对比分析**

与随机森林、XGBoost、GAT、GIN、GraphSAGE、TabPFN等基线模型比较，SA‑HGNN在四个区域的MAPE、R²和CRMSE等指标均优于或接近最优，尤其在康涅狄格州提升了近20%。

**⚠️ 局限性**

局限性包括仅为每个地区训练独立模型、缺乏时间序列信息以及对比学习正负样本选择的启发式方式。

---

## 6. A Multi-Agent Framework for Democratizing XR Content Creation in K-12 Classrooms

**arXiv ID:** 2604.04728 | [PDF](https://arxiv.org/pdf/2604.04728v1)

**作者:** Yuan Chang `[一作]` (Meta), Jiaming Qu `[通讯]` (Amazon)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建了一个四代理（教学、执行、安全、导师）协同的多代理 XR 内容创作框架，帮助 K‑12 教师用自然语言快速生成安全、符合课程目标的 XR 场景。

**💡 创新点**

创新点在于将教学意图、技术生成、内容安全与教学增值拆分为专门代理，形成教师可控、低门槛、以安全为核心的端到端创作流程；同时引入人‑机协同验证，保留教师决策权。

**🔧 技术方法**

使用大语言模型（Claude、OpenAI）、Meshy 3D 生成 API、Tavily 搜索与内容聚合、浏览器渲染技术（Next.js/React、Google model-viewer）以及 FastAPI 后端实现四代理协作。

**📊 数据集**

本工作未使用公开数据集，而是依赖 LLM 的预训练知识与 Meshy API 的模型生成能力；导师代理通过互联网检索实时信息。

**📈 对比分析**

尚未进行实证评估；目前仅在原型演示中展示了流程与生成效果，缺乏教师/学生使用实验与量化性能指标。

**⚠️ 局限性**

局限包括：Meshy 生成耗时1–5分钟且成本高；安全代理仅评估文本与单视图图像，未分析完整 3D 几何；未在真实 K‑12 环境中验证易用性与教学效果；依赖外部 API，易受服务变更影响。

---

## 7. AI Agents Under EU Law

**arXiv ID:** 2604.04604 | [PDF](https://arxiv.org/pdf/2604.04604v1)

**作者:** Luca Nannini `[一作]` (Piccadilly Labs), Piercosma Bisconti `[通讯]` (DEXAI)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

为AI代理提供者构建系统化的合规映射与分类框架，涵盖九大部署场景与12步合规架构。

**💡 创新点**

首次将欧盟AI法案与多项并行法规（GDPR、网络韧性法案、DSA等）与最新草案标准关联，并针对代理特有风险提出实用的监管触发器表。

**🔧 技术方法**

采用法规文本、标准草案、案例分析的法律技术研究方法，结合ISO/IEC 18228等标准进行技术映射。

**📊 数据集**

未使用实验数据集，而是基于官方法规文件、标准草案及行业报告进行分析。

**📈 对比分析**

通过对比欧盟各法规与标准的条款，识别合规缺口与重叠；未开展量化性能评估，侧重理论合规性。

**⚠️ 局限性**

受限于标准仍处于草案阶段、缺乏行政指导、以及对代理运行时行为的动态评估方法不足。

---

## 8. TORA: Topological Representation Alignment for 3D Shape Assembly

**arXiv ID:** 2604.04050 | [PDF](https://arxiv.org/pdf/2604.04050v1)

**作者:** Nahyuk Lee `[一作]` (Independent Researcher), Sunghwan Hong `[通讯]` (ETH Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过教师‑学生对齐框架（TORA）将冻结的预训练3D编码器的几何与交互拓扑信息注入点流组装模型，从而提升3D物体组装性能。

**💡 创新点**

创新点在于提出拓扑优先的表示对齐策略：利用Center Kernel Alignment (CKA) 损失显式匹配点与点之间的相似性结构，并证明几何与接触信息比语义分类更适合作为教师引导。

**🔧 技术方法**

技术手段包括：基于Rectified Point Flow (RPF) 的流匹配网络；token‑wise cosine/NT‑Xent 以及 CKA 对齐损失；冻结的Uni3D 预训练编码器；投影器 MLP；随机采样 Gram 矩阵实现高效计算。

**📊 数据集**

实验使用六个多样化组装基准：Breaking Bad、PartNet‑Assembly、TwoByTwo、Fractura、Fantastic Breaks 以及 Breaking Bad‑Artifact。

**📈 对比分析**

与 RPF 及其他多片段组装方法对比，TORA 在多部分、语义和交互组装上均显著优于基线；在零样本转移任务中，CKA 对齐实现最低姿态误差；收敛速度提升至 6.9 倍以上。

**⚠️ 局限性**

局限性包括：对教师模型和对齐层的选择高度敏感；在分布迁移下 NT‑Xent 可能导致性能下降；随机采样的 Gram 矩阵在极大点云规模下仍有计算开销；仅在点流模型上验证，其他生成/传输框架的适用性未知。

---

## 9. ART: Adaptive Relational Transformer for Pedestrian Trajectory Prediction with Temporal-Aware Relations

**arXiv ID:** 2604.03649 | [PDF](https://arxiv.org/pdf/2604.03649v1)

**作者:** Ruochen Li `[一作]` (Durham University), Hubert P. H. Shum `[通讯]` (Durham University)

**通讯引用:** 3763 | [OpenAlex ID](https://openalex.org/A5038258635)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了Adaptive Relational Transformer（ART），通过时间感知关系图（TARG）和自适应交互裁剪（AIP）实现了更精细的行人交互建模与高效预测；

**💡 创新点**

创新点在于：①TARG在时序维度上对每个交互进行注意力聚合，显式保留时间属性；②AIP采用top‑p阈值自适应稀疏化邻居集合，既降低计算成本又保持关键信息；

**🔧 技术方法**

使用了Transformer架构的多头注意力、位置编码、边缘加权消息传递、最佳‑k训练以及基于累计权重的自适应阈值裁剪；

**📊 数据集**

使用了ETH/UCY（ETH、HOTEL、UNIV、ZARA1、ZARA2）和NBA SportVU两个公共数据集进行评估；

**📈 对比分析**

与STAR、GroupNet、MemoNet、MID、NPSN、EqMotion、LED、MART等多种SOTA方法对比，ART在ETH/UCY的平均minADE/minFDE分别达到0.20/0.32，ZARA2 0.12/0.21，并在NBA各时间尺度上持续领跑；同时参数量仅约1M，MACs最低（40M），显示出良好的性能与效率平衡；

**⚠️ 局限性**

局限性包括：对极端稀疏或过度密集场景下的阈值敏感；模型仅基于二维轨迹，未考虑姿态或全身运动；目前评估仅限于公开数据集，未在真实机器人交互环境中验证；

---

## 10. MPTF-Net: Multi-view Pyramid Transformer Fusion Network for LiDAR-based Place Recognition

**arXiv ID:** 2604.04513 | [PDF](https://arxiv.org/pdf/2604.04513v1)

**作者:** Shuyuan Li `[一作]`, Dong Kong `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种多视角多尺度Transformer融合网络MPTF-Net，用于LiDAR地点云的位姿识别。

**💡 创新点**

创新点在于使用基于Normal Distribution Transform的多通道BEV编码提供噪声鲁棒的几何先验，并设计跨尺度交叉注意力的金字塔Transformer实现RIV与BEV的深层融合。

**🔧 技术方法**

采用RIV+NDT-BEV双通道输入、ResNet骨干、Pyramid Transformer交叉注意力、上下文门控NetVLAD聚合，训练使用三元组损失。

**📊 数据集**

在nuScenes、KITTI和NCLT三个公开数据集上进行评测。

**📈 对比分析**

与现有单视图和多视图方法相比，MPTF-Net在nuScenes Boston和Singapore分割Recall@1分别达到96.31%和99.43%，平均延迟仅10ms，明显优于前沿方法。

**⚠️ 局限性**

局限在于对极端稀疏点云的处理仍依赖BEV统计，且在高度动态场景下的鲁棒性待进一步验证。

---

## 11. Good Rankings, Wrong Probabilities: A Calibration Audit of Multimodal Cancer Survival Models

**arXiv ID:** 2604.04239 | [PDF](https://arxiv.org/pdf/2604.04239v1)

**作者:** Sajad Ghawami `[一作]` `[通讯]` (Independent Researcher), Sajad Ghawami (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文对多模态全切片图像与基因组学融合的生存预测模型进行了系统的折叠级1校准审计，评估了其预测概率的校准性；

**💡 创新点**

首次在此类模型上执行折叠级1校准审计，并发现高判别模型往往误校准，门控融合优于双线性/拼接融合，后验Platt缩放能显著恢复校准；

**🔧 技术方法**

采用SurvivalEVAL中的1校准（Hosmer-Lemeshow IPCW版）、Breslow重建生存曲线、Platt缩放以及Benjamini-Hochberg多重检验校正；

**📊 数据集**

使用TCGA五种癌症（BLCA、BRCA、GBMLGG、LUAD、UCEC）共约3000名患者的公开数据进行5折交叉验证；

**📈 对比分析**

对11种架构共290个折叠级1校准测试，166个显著拒绝零假设；最高C-index的模型（如MCAT在GBMLGG）仍多次失败校准；Platt缩放将大部分失败转为成功，而C-index保持不变；

**⚠️ 局限性**

Breslow重建假设比例风险导致部分误校准；1校准仅检验单一时间点；高删失率与小样本导致统计功效下降；未在多中心数据上验证；仅评估NLL损失模型。

---

## 12. Document-Level Numerical Reasoning across Single and Multiple Tables in Financial Reports

**arXiv ID:** 2604.03664 | [PDF](https://arxiv.org/pdf/2604.03664v1)

**作者:** Yi-Cheng Wang `[一作]` (National Taiwan University), Chu-Song Chen `[通讯]` (National Taiwan University)

**通讯引用:** 6469 | [OpenAlex ID](https://openalex.org/A5072890368)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出FinLongDocQA数据集和FinLongDocAgent多代理RAG方法，解决跨表长文档财务数值推理任务。

**💡 创新点**

首次构建文档级跨表财务数值QA基准并提供可执行程序与页级证据注解，设计迭代检索与验证的多代理RAG框架。

**🔧 技术方法**

使用LLM自动生成、规则过滤与人工审核，BGE稠密检索，AutoGen多代理框架，Python可执行程序进行算子化推理。

**📊 数据集**

基于2022-2024年S&P 500公司年度报告，1,456份文件、7,527个QA实例，平均长度约129k token，覆盖跨表证据。

**📈 对比分析**

与LLM-only、长上下文、单轮RAG、图RAG、WebDancer等基线比较，FinLongDocAgent在Gemini-3-Flash上取得EM 41.34%、TolAcc 43.54%、F1 51.29，显著优于其他方法，但整体仍低于理想水平。

**⚠️ 局限性**

检索召回不足导致证据缺失，表格结构信息损失，推理耗时较高，数据仅覆盖近年S&P 500，缺乏跨市场与跨时间推广性。

---

## 13. Defending Buffer Overflows in WebAssembly: A Transpiler Approach

**arXiv ID:** 2604.03859 | [PDF](https://arxiv.org/pdf/2604.03859v1)

**作者:** Weiqi Feng `[一作]` `[通讯]` (Harvard University), Weiqi Feng (Harvard University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出一种基于转译器的方案，在 WebAssembly 二进制中注入堆栈 canary 和 ASLR，以防止未管理内存中的缓冲区溢出。

**💡 创新点**

创新点在于：在不修改源代码或编译器的前提下，通过转译器在二进制层面动态注入堆栈 canary 与函数级 ASLR；使用运行时基于时间的随机数生成器；将 canary 存放在局部变量而非线性内存，以避免被溢出。

**🔧 技术方法**

采用了 WebAssembly 文本格式 (.wat) 的二进制注入技术、随机数生成（使用 time/srand/rand）、栈指针调整、内存读写指令以及 JavaScript 运行时支持等技术。

**📊 数据集**

评估使用了公开的易受攻击 WebAssembly 示例程序，并基于分析26个公开二进制（共98,924个函数）进行覆盖率评估；未提供公开数据集的具体名称。

**📈 对比分析**

通过计量每个函数被插入的指令数量来计算保护开销，并与原程序的函数调用次数相乘得到整体 overhead；实验中对比攻击前后是否能成功触发非法调用，验证了防御效果；虽然具体执行时间未给出，但指出开销与函数数密切相关。

**⚠️ 局限性**

局限性包括：全局变量溢出无法防护；ASLR 随机范围仅 0-255，易被暴力猜测；堆栈 canary 只能检测触及 canary 的溢出，无法防止不触及 canary 的溢出；错误信息缺失；随机数种子可能被泄露；只支持文本级注入，未直接操作二进制；间接调用表保护不足；缺乏编译器级别的完整防护。

---

## 14. CB-VER: A Stable Foundation for Modular Control Plane Verification

**arXiv ID:** 2604.03539 | [PDF](https://arxiv.org/pdf/2604.03539v1)

**作者:** Dexin Zhang `[一作]` (Princeton University), Aarti Gupta `[通讯]` (Princeton University)

**通讯引用:** 5911 | [OpenAlex ID](https://openalex.org/A5041231710)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文提出了一种新的模块化网络控制平面验证工具，用于验证在异步网络环境下的“最终稳定”性质，并提供了接口合成与容错分析的完整框架。

**💡 创新点**

创新点主要体现在：①引入抽象收敛（abstract convergence）与收敛前图（converges-before graph）作为验证的核心；②通过接口 I 与 Q 的并行验证实现真正的模块化；③利用 SMT 进行条件检查，并在 Lean 中形式化证明其正确性；④提供自动接口合成（CHC 形式化）与容错分析，显著提升了可用性与可扩展性。

**🔧 技术方法**

技术手段包括：异步消息传递模型、SMT 求解（Z3）、Constrained Horn Clause（Spacer/Z3）用于接口合成、Lean theorem prover 进行形式化验证、Breadth‑First Search 用于检查收敛前图连通性、Dinitz 算法用于 k‑容错分析。

**📊 数据集**

实验数据集涵盖：① 20–2000 节点的合成 fat‑tree；② 约 300 节点的 Internet2（真实路由器配置）；③ 两个 Batfish 教程网络（共 13 节点）。

**📈 对比分析**

与 Timepiece（模块化验证器）和 Minesweeper（单片验证器）对比：在 fat‑tree 上 2000 节点的验证耗时约 20 分钟，性能与 Timepiece 相当或略优，远优于 Minesweeper；在 Internet2 上验证时间分别为 22s、121s、74s，Minesweeper 在同一属性上内存溢出；接口合成工具相较于 Minesweeper 在 2000 节点 fat‑tree 的 Reachability 任务提升 22–27 倍。

**⚠️ 局限性**

局限性包括：① 需要用户提供至少初始接口 I 与 Q，错误或不足的接口会导致验证失败；② 接口合成目前仅在简化的 BGP 模型下实现，尚未覆盖完整的多协议与复杂策略；③ 依赖公平调度假设，实际网络中可能出现不公平或失效；④ 在极大规模网络与极复杂属性下，SMT/CHC 求解器仍可能遇到时间或资源瓶颈。

---

## 15. REAM: Merging Improves Pruning of Experts in LLMs

**arXiv ID:** 2604.04356 | [PDF](https://arxiv.org/pdf/2604.04356v1)

**作者:** Saurav Jha `[一作]` (Polytechnique Montreal), Boris Knyazev `[通讯]` (Mila Quebec AI Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为REAM的专家压缩方法，旨在在保持模型性能的同时减少Mixture-of-Experts（MoE）大语言模型的专家数量。

**💡 创新点**

创新点在于将专家的相似度同时考虑路由日志相似度与门控加权的专家输出相似度，并采用伪剪枝策略形成大组与单例；同时引入结合激活与权重的对齐成本矩阵和顺序合并以更新后续层统计。

**🔧 技术方法**

技术包括专家相似度计算（门控加权输出相似度）、伪剪枝聚类、激活与权重双重成本的匈牙利匹配、顺序前向传递更新统计，以及基于REAP的专家重要性评分。

**📊 数据集**

使用的校准数据为C4（通用文本）、NuminaMath（数学推理）和The-Stack-Smol（代码生成）三种数据集，实验在Qwen3系列和GLM-4.5-Air等多模组模型上进行。

**📈 对比分析**

与频率剪枝、REAP剪枝和HC-SMoE合并基线对比，REAM在25%与50%压缩率下在生成（GEN）基准上平均提升约1-2分，接近或略低于未压缩模型；在多选（MC）任务上表现与基线相当或略逊。

**⚠️ 局限性**

局限性包括对校准数据比例高度敏感，需手动调节混合比例以获得最优MC–GEN权衡；方法在更大模型或不同任务域下的鲁棒性仍待验证；同时缺乏进一步微调或后续适配步骤。

---

## 16. Negative-Voltage-Enabled Energy Efficient Nonvolatile Memories And In-Memory Computing Based On 2D Piezoelectric Transistors

**arXiv ID:** 2604.03959 | [PDF](https://arxiv.org/pdf/2604.03959v1)

**作者:** Jeffry Victor `[一作]` (Purdue University), Sumeet K. Gupta `[通讯]` (Purdue University)

**通讯引用:** 3881 | [OpenAlex ID](https://openalex.org/A5044276472)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

**🎯 论文内容**

提出两种利用负电压偏置的 PeFET NVM 设计（NeVo HD 与 NeVo 2T-1P），并将其应用于内存加减与三值乘加（STP-MAC）运算，显著降低读写及内存计算能耗。

**💡 创新点**

创新点包括：① 使用负电压（NeVo）偏置彻底消除读写时对垂直位线（RBL/WBL）及漏极电容的充电能耗；② 通过 PeFET 的极化状态与 V_GB 极性耦合实现单周期加减运算；③ 在 NeVo HD 结构下实现 STP-MAC，进一步压缩单元面积并提升能效。

**🔧 技术方法**

技术手段：PeFET 结构（PE/FE 电容 + 2D MoS₂ 源漏通道）、负电压偏置电路、分段（segmentation）布局、单端/电流感知读、V_GB 极性控制、模/数转换（ADC）以及外围计算模块（CM）用于加减与 MAC。

**📊 数据集**

未使用特定数据集；实验以 512×512 位阵列的仿真结果为主，评估能耗、延迟与面积。对于 DNN 加速场景，采用三值网络的典型乘加操作做能效对比。

**📈 对比分析**

比较方法：将 NeVo HD、NeVo 2T-1P 与传统 HD、2T-1P、6T SRAM 在读写、加减、STP-MAC 等操作中进行能耗、延迟、面积对比。性能表现：NeVo HD 读能耗下降至 HD 的 0.06×，NeVo 2T-1P 读能耗下降至 2T-1P 的 0.03×；写能耗分别下降至 1.08×和 0.55×；内存计算能耗相较于 2T-1P 下降 0.19×，相较于 SRAM 低 15% 能耗、91% 延迟。

**⚠️ 局限性**

限制：1）NeVo HD 在电压感知读时因更高的布局高度导致 RBL 充电能耗上升，能耗比 HD 略高；2）HD 结构仅支持同极性 V_GB，无法实现加减运算；3）目前实现的 STP-MAC 需要 16 行并行，增加 ADC 负载；4）实验主要为仿真，缺乏大规模硬件验证；5）负电压生成方案对工艺与电路复杂度提出额外要求。

---

## 17. Developing Authentic Simulated Learners for Mathematics Teacher Learning: Insights from Three Approaches with Large Language Models

**arXiv ID:** 2604.04361 | [PDF](https://arxiv.org/pdf/2604.04361v1)

**作者:** Jie Cao `[一作]` (University of North Carolina at Chapel Hill), Dionne Cross Francis `[通讯]` (University of North Carolina at Chapel Hill)

**通讯引用:** 2288 | [OpenAlex ID](https://openalex.org/A5020017168)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了三种基于大语言模型的真实化学生模拟器（微调、多智能体、直接偏好优化），用于数学教师的专业观察训练。

**💡 创新点**

首次系统比较三种超越零/少量提示的技术对学生模拟真实性的提升，并结合教学实践反馈揭示其不同的教育意义。

**🔧 技术方法**

采用LLM微调、Responser-Evaluator-Refiner多智能体架构、通过Reflexion循环生成的配对偏好数据进行的直接偏好优化（DPO）等技术，并使用GLMM、McNemar检验等统计方法进行评估。

**📊 数据集**

使用了1,296条学生语料（PST-学生对话、Khan Academy AI辅导对话、TalkMoves教材对话）和150条人工合成的偏好对，以及PST与模型交互的1438回合对话。

**📈 对比分析**

通过与基线少量提示的对照，采用McNemar检验和GLMM分析真实性指标。三种方法均显著提升认知与语言真实性（p<0.05），DPO在语言上达到100%、认知88.7%的表现；教师评测中DPO获得最高偏好与实用性评分。

**⚠️ 局限性**

局限包括：微调受限于数据量，难以覆盖多样情境；多智能体方法响应延迟高；模拟中不确定性表达有时不一致；仅针对单个分数任务与单学生，缺乏多学生、多场景的验证；未来需结合知识追踪、人工偏好、测量模型等改进。

---

## 18. SVD Provably Denoises Nearest Neighbor Data

**arXiv ID:** 2604.03831 | [PDF](https://arxiv.org/pdf/2604.03831v1)

**作者:** Ravindran Kannan `[一作]` (Simons Institute), David Woodruff `[通讯]` (Carnegie Mellon University)

**通讯引用:** 8998 | [OpenAlex ID](https://openalex.org/A5102861589)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究高维高噪声下的最近邻搜索，提出利用SVD对低维子空间数据进行去噪并恢复原始最近邻。

**💡 创新点**

证明在噪声水平σ=O(k^{-1/4})时SVD即可恢复最近邻，并给出匹配下界，首次将噪声容限与内在维度关联。

**🔧 技术方法**

采用奇异值分解、子空间角度理论、Wedin定理与随机矩阵界限等理论工具。

**📊 数据集**

在GloVe词向量（200维）和MNIST手写数字图像（784维）上进行实验。

**📈 对比分析**

与直接在噪声数据上最小化距离的朴素方法对比，SVD方法在噪声阈值上显著提升，性能在高噪声区间保持90%以上成功率。

**⚠️ 局限性**

需预先知道子空间维度k，且对数据的奇异值分布有一定要求；在噪声远大于k^{-1/4}时恢复不可能。

---

## 19. The Format Tax

**arXiv ID:** 2604.03616 | [PDF](https://arxiv.org/pdf/2604.03616v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 20. Your Agent, Their Asset: A Real-World Safety Analysis of OpenClaw

**arXiv ID:** 2604.04759 | [PDF](https://arxiv.org/pdf/2604.04759v1)

**作者:** Zijun Wang `[一作]` (University of California Santa Cruz), Cihang Xie `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

对全局系统访问的个人 AI 代理 OpenClaw 进行真实环境安全评估，系统性检验其持续状态（能力、身份、知识）被污染后的攻击成功率，并在四大 LLM 主干模型上对 12 个实际影响场景进行测试。

**💡 创新点**

提出统一的 CIK（Capability、Identity、Knowledge）持久状态分类框架，用于梳理个人 AI 代理的攻击面；首次在真实部署环境下验证所有三维度的攻击可行性及其对安全性的结构性影响；探讨基于 CIK 的多层防御策略及文件保护带来的演化‑安全权衡。

**🔧 技术方法**

采用两阶段注入攻击（Phase 1: 注入毒性内容，Phase 2: 触发恶意行为）; 在本地 Mac Mini 上集成 Gmail、Stripe 和文件系统，利用 Telegram 发送触发请求；利用自动化测试 harness 记录与验证结果；评估四大 LLM 主干模型（Claude Sonnet 4.5、Claude Opus 4.6、Google Gemini 3.1 Pro、OpenAI GPT‑5.4）。

**📊 数据集**

无公开标准数据集，全部实验基于自建 OpenClaw 实例的真实外部服务交互（Gmail、Stripe test mode、文件系统）以及人工设计的 12 个攻击场景和对应的 4 种注入向量，覆盖隐私泄露与不可逆操作三大危害子类。

**📈 对比分析**

通过对比基线（未被污染）与各维度被毒化后的攻击成功率（ASR）来评估。结果显示，在基线下 ASR 为 10–36%，注入后提升至 64–74%，知识注入平均 ASR 最高；四大模型均表现出相似趋势，表明漏洞结构性而非模型特异性。三种基于 CIK 的防御方案分别针对知识、身份与能力进行缓解，但仍无法完全消除攻击；文件保护可将攻击注入率降至约 5%，但同样抑制 87% 的合法更新，凸显演化‑安全权衡。

**⚠️ 局限性**

仅测试单一平台（OpenClaw）和四大主干模型；攻击场景人工设计，缺乏自动化攻击生成；未探究跨维度攻击链的组合效应；防御仅为提示层面，未涵盖能力层面代码签名、沙箱等架构性安全措施。

---

## 21. Agile Story-Point Estimation: Is RAG a Better Way to Go?

**arXiv ID:** 2604.03443 | [PDF](https://arxiv.org/pdf/2604.03443v1)

**作者:** Lamyea Maha `[一作]` (University of Saskatchewan), Chanchal Roy `[通讯]` (University of Saskatchewan)

**通讯引用:** 9378 | [OpenAlex ID](https://openalex.org/A5102756770)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了一种检索增强生成（RAG）框架，用于自动估计敏捷开发中的故事点，并在23个开源项目上进行了实验评估。

**💡 创新点**

创新点在于将检索-生成模型用于数值预测任务，并通过检索相似任务提供可解释上下文；同时系统探讨了检索参数、嵌入模型以及项目规模对估计性能的影响。

**🔧 技术方法**

技术实现包括：检索模块使用 BAAI bge-large-en-v1.5 与 SBERT all-mpnet-base-v2 生成任务嵌入；生成器采用 Llama-3.2-3B-Instruct；评估指标为 MAE/MdAE，并使用 Wilcoxon 符号秩检验和 Kruskal–Wallis 检验评估统计显著性。

**📊 数据集**

数据集为 Tawosi 等人公开的 23 个开源项目（共约 31,960 条任务），对任务进行过滤后得到 23 个项目；对比基线包括 Deep-SE、LHC-SE、LHCtc-SE 以及 TF‑IDF 模型。

**📈 对比分析**

通过 MAE/MdAE 进行量化比较，发现 RAG 在部分项目上优于某些基线，但整体未显示显著优势；检索参数的最优取值因项目规模而异，嵌入模型选择对性能差异不显著；统计检验未能拒绝零假设，表明 RAG 并未显著提升估计精度。

**⚠️ 局限性**

局限性包括：依赖人工标注的故事点可能存在偏差；仅使用开源项目，难以代表商业环境；生成器未进行项目特定微调，导致对数值预测的适用性受限；RAG 对于结构化数值预测的提升空间有限。

---

## 22. Responses Fall Short of Understanding: Revealing the Gap between Internal Representations and Responses in Visual Document Understanding

**arXiv ID:** 2604.04411 | [PDF](https://arxiv.org/pdf/2604.04411v1)

**作者:** Haruka Kawasaki `[一作]` (NTT), Kyosuke Nishida `[通讯]` (NTT)

**通讯引用:** 1681 | [OpenAlex ID](https://openalex.org/A5110780218)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过线性探测分析大型视觉语言模型在视觉文档理解任务中的内部层级表示，发现答案信息最易在线性可分的中间层体现，并提出仅微调中间层的方法来缩小内部表示与输出之间的差距。

**💡 创新点**

创新点在于首次对VNU任务中LVLM内部层进行细粒度线性探测，揭示中间层承载答案信息的线性可分性最高，并针对性地微调中间层显著提升模型性能。

**🔧 技术方法**

主要技术包括线性探测（Linear Probing）、层级微调（只调中间层）、大规模视觉语言模型（如Qwen2.5‑VL、Gemma3、LLaVA‑NEXT）、以及自动回归生成与ANLS评估。

**📊 数据集**

使用的评估数据集包括四个二分类线性探测任务（Easy‑VQA、MJSynth、PubLayNet、FigureQA）以及文档VQA基准（DocVQA、InfographicVQA）。

**📈 对比分析**

与全层微调对比，模型通过中间层微调获得更高的响应准确率和线性探测准确率，并显著缩小两者差距；同时训练参数量和时间也更为高效。

**⚠️ 局限性**

局限性在于线性探测只能捕捉线性可分信息，难以评估复杂推理或开放式文本生成任务的内部表示，未来需研发更具表达力的探测方法。

---

## 23. Measuring Robustness of Speech Recognition from MEG Signals Under Distribution Shift

**arXiv ID:** 2604.04129 | [PDF](https://arxiv.org/pdf/2604.04129v1)

**作者:** Sheng-You Chien `[一作]` (National Tsing-Hua University), Po-Chih Kuo `[通讯]` (National Tsing-Hua University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文构建并评估了多种基于MEG的语音音素分类模型，探究预处理、归一化与数据增强对鲁棒性的影响；

**💡 创新点**

主要创新在于系统性验证实例归一化对分布漂移的缓解效果，以及对组平均与标签平衡对单次试验信噪比提升的实证；

**🔧 技术方法**

使用残差CNN、STFT-CNN、CNN‑Transformer、MEGConformer等网络结构，并引入实例归一化、组平均、标签平衡、重复采样、数据增强等技术；

**📊 数据集**

采用LibriBrain 2025 PNPL比赛的MEG音素分类基准数据，包含来自单一受试者的93个录音会话，每段500 ms，39类音素；

**📈 对比分析**

通过宏F1评估，实例归一化显著提升验证到测试的迁移性能，最佳自研模型在测试集上达到60.95% F1‑macro，接近MEGConformer的64.09%；

**⚠️ 局限性**

局限在于依赖组平均降低噪声，难以泛化至真实单次试验情境，以及对归一化统计差异的依赖，需进一步研究自适应归一化与单试验降噪方法。

---

## 24. Intelligent Traffic Monitoring with YOLOv11: A Case Study in Real-Time Vehicle Detection

**arXiv ID:** 2604.04080 | [PDF](https://arxiv.org/pdf/2604.04080v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 25. DeepStack: Scalable and Accurate Design Space Exploration for Distributed 3D-Stacked AI Accelerators

**arXiv ID:** 2604.04750 | [PDF](https://arxiv.org/pdf/2604.04750v1)

**作者:** Zhiwen Mo `[一作]` (Imperial College London), Hongxiang Fan `[通讯]` (Imperial College London)

**通讯引用:** 836 | [OpenAlex ID](https://openalex.org/A5057043409)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一个端到端的分布式3D堆叠AI加速器设计空间探索（DSE）框架，能够准确建模3D DRAM的交易级带宽、银行激活约束、缓冲限制，并结合计算–通信重叠和热功耗协同模型，支持多种并行策略和网络拓扑的全局搜索。

**💡 创新点**

创新点在于：①双阶段网络抽象+车道级计算通信重叠模型；②交易级3D DRAM带宽与银行冲突细粒度建模；③覆盖所有七种并行策略（TP、EP、SP、CP、DP、FSDP、PP）并允许模块级自由切换；④层次化NoC优化与热功耗约束；⑤在约2.5×10¹⁴个配置上实现高效搜索，突破传统模拟器的性能瓶颈。

**🔧 技术方法**

使用了：分析性能建模、Little’s Law、功耗与温度方程、双阶段流量矩阵映射、tile级流水线模型、层次化NoC路由与拥塞评估，以及多阶段剪枝与层次化搜索策略；工具实现名为DeepStack。

**📊 数据集**

评估数据集：Llama‑3.3‑70B/405B、DeepSeek‑R1/V3、Qwen3‑235B，批量大小从1到1024，序列长度设为1024；对比基准包括8×H100、8×B200 GPU集群，使用vLLM、Triton‑Distributed核实现。

**📈 对比分析**

通过与Cadence Palladium周期精确仿真（误差≤5%）、NS‑3后端（误差≤2%）以及真实硬件（12.18% MAPE）进行交叉验证；在ASTRA‑sim的基础上实现了高达100,000×的速度提升；DSE结果在吞吐量上比基线提升至9.5×，在解码阶段比2.5D设计高出2.79×。

**⚠️ 局限性**

局限性：模型仍采用分析近似，某些细粒度实现细节（如FlashMLA动态KV拆分）未被捕捉；仅在7nm工艺假设下评估，实际工艺差异可能导致误差；剪枝策略可能错过罕见最优点；热模型简化，未考虑更复杂的液冷或微通道散热方案。

---

## 26. Deep Kuratowski Embedding Neural Networks for Wasserstein Metric Learning

**arXiv ID:** 2604.04343 | [PDF](https://arxiv.org/pdf/2604.04343v1)

**作者:** Andrew Qing He `[一作]` `[通讯]` (Southern Methodist University), Andrew Qing He (Southern Methodist University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实验验证了两种基于Kurakowski嵌入思想的神经网络架构（DeepKENN和ODE-KENN）用于近似Wasserstein-2距离；

**💡 创新点**

创新点在于：①通过对多层CNN特征加权实现有限维多重嵌入；②将离散层堆栈替换为Neural ODE，形成无限维Banach空间嵌入，并利用ODE轨迹平滑性实现隐式正则化；

**🔧 技术方法**

采用CNN编码器、可学习的权重软加权、Neural ODE（RK4求解）、Adam优化和MSE损失；

**📊 数据集**

使用MNIST手写数字图像（归一化为概率测度），预计算55,000对精确W_2距离作为训练集；

**📈 对比分析**

在与单层欧氏距离基线和DeepKENN在相同参数量下的比较中，ODE-KENN在测试MSE上比基线低28%、比DeepKENN低18%，并表现出更小的泛化误差；

**⚠️ 局限性**

局限性包括：①需预先计算大量精确W_2样本，训练成本高；②不保证得到的近似距离在所有输入上严格满足正定性；③对新图像域需重新训练。

---

## 27. Super Agents and Confounders: Influence of surrounding agents on vehicle trajectory prediction

**arXiv ID:** 2604.03463 | [PDF](https://arxiv.org/pdf/2604.03463v1)

**作者:** Daniel Jost `[一作]` (University of Freiburg), Joschka Bödecker `[通讯]` (University of Freiburg)

**通讯引用:** 3273 | [OpenAlex ID](https://openalex.org/A5038908529)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本研究通过Shapley值归因方法，系统分析了轨迹预测模型对周围车辆信息的利用，发现大多数邻居会作为干扰因子削弱预测精度；随后提出使用条件信息瓶颈（Conditional Information Bottleneck, CIB）来压缩并筛选有用的邻居特征，以提升鲁棒性和准确性。

**💡 创新点**

创新点在于：1）首次在轨迹预测中量化每个邻居的正负贡献并区分“Super Agents”和“Confounding Agents”；2）通过CIB实现无监督的上下文信息压缩，显著降低模型对无关邻居的依赖；3）系统对不同数据集、模型架构及鲁棒性扰动进行跨模型一致性与对比。

**🔧 技术方法**

核心技术包括：Shapley值归因（ApproShapley近似）、条件信息瓶颈（CIB）模块、变分信息瓶颈（VIB）原理、模型的Encoder‑Interactor‑Decoder结构以及插入/删除测试。

**📊 数据集**

实验数据集为Waymo Open Motion Dataset (WOMD) 和 nuScenes，此外在WOMD上使用20%子集进行实验；对joint预测使用BeTop模型。

**📈 对比分析**

对比方法：在不同模型（QEANet、LAformer、MTR、EDA、BeTop）和不同种子下训练的基线与CIB增强版本；使用ADE/FDE/MissRate/NLL等指标；结果显示CIB在大多数模型上提升了ADE/FDE和MissRate，尤其在WOMD上几乎消除了Super Agents与All agents之间的性能差距；在噪声与非因果邻居移除扰动下的鲁棒性也显著提升。

**⚠️ 局限性**

局限性：CIB虽提升鲁棒性但未能显著解决模型对因果关系的识别不稳定；模型对不同训练种子的决策机制差异仍大，缺乏对因果推理的显式机制；实验中对大规模场景和更复杂交互的验证仍不足。

---

## 28. CoALFake: Collaborative Active Learning with Human-LLM Co-Annotation for Cross-Domain Fake News Detection

**arXiv ID:** 2604.04174 | [PDF](https://arxiv.org/pdf/2604.04174v1)

**作者:** Esma Aïmeur `[一作]` (University of Montreal), Dorsaf Sallami `[通讯]` (University of Montreal)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个跨域假新闻检测框架CoALFake，结合人类与LLM协同标注和域感知主动学习，训练域无关分类器。

**💡 创新点**

创新点包括：① 人类-LLM协同标注机制，利用LLM进行大规模低成本标注并用可信学习挑选需人工复核的样本；② 域嵌入与域无关双子空间模型，跨域特征与域特定特征并行学习并通过交叉注意力互相融合；③ 域感知主动采样，先用聚类得到多域信息，再按逆域大小加权选择样本，兼顾多样性与不确定性。

**🔧 技术方法**

技术手段：LLM（GPT‑3.5‑Turbo）few‑shot提示+k‑NN示例；可信学习（Confident Learning）做标签校验；句子‑BERT + k‑means + softmax 生成域嵌入；双子空间（domain‑specific & shared）加交叉注意力；多损失优化（预测、重构、域特定、域共享、正交性、对比性）。

**📊 数据集**

实验数据集：PolitiFact（政治）、GossipCop（名人）、CoAID（医疗），每集100条示例构成demo，余下75%做训练池，25%做测试。

**📈 对比分析**

与多种基线（HPNF、AE、SAFE、EDDFN、MDFEND、FuDFEND、DITFEND、SLFEND、GPT‑3.5‑Turbo提示）比较，CoALFake在所有领域的准确率、精确率、召回率、F1均优于基线；例如在PolitiFact上F1 0.92，最优基线仅0.85；在GossipCop、CoAID上F1分别为0.91和0.95，领先相近方法0.93。主动采样方面，域感知策略在不同样本比例下均击败随机、最大熵等传统采样。

**⚠️ 局限性**

局限性：① 依赖原始数据标签做人工复核，未对不同平台或标注标准做广泛验证；② 可信学习与LLM标注质量受模型校准与任务复杂度影响；③ 目前仅评估GPT‑3.5‑Turbo，未验证对其他LLM的迁移性；④ 对完全未见域的泛化仍未充分测试；⑤ 人工复核比例与成本折衷尚需更细粒度优化。

---

## 29. HOIGS: Human-Object Interaction Gaussian Splatting

**arXiv ID:** 2604.04016 | [PDF](https://arxiv.org/pdf/2604.04016v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 30. Are Latent Reasoning Models Easily Interpretable?

**arXiv ID:** 2604.04902 | [PDF](https://arxiv.org/pdf/2604.04902v1)

**作者:** Connor Dilgren `[一作]` (University of Maryland), Sarah Wiegreffe `[通讯]` (University of Maryland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对两种主流潜在推理模型（Coconut和CODI）进行解释性评估，探究其推理令牌的必要性、可解码性以及在未监督情况下提取自然语言推理路径的可行性。

**💡 创新点**

①发现潜在推理令牌在逻辑推理任务中往往不必要；②证明在正确预测时可通过词汇投影恢复金标准推理轨迹；③提出一种无监督的“前向链”方法，能在大多数正确案例中验证推理路径。

**🔧 技术方法**

词汇投影、后向回溯搜索、前向链验证、早停实验以及多模式（无推理、显式推理、潜在推理）模型对照训练。

**📊 数据集**

GSM8k-Aug（算术推理）、PrOntoQA与ProsQA（层次逻辑推理）三个公开数据集。

**📈 对比分析**

通过控制训练数据量对比不同推理模式，结果显示：在逻辑推理数据集上，Coconut和CODI的性能与无推理模型相当；在算术推理上，显式推理仍优于潜在推理；潜在推理令牌的利用率低，且大部分正确预测能恢复推理轨迹。

**⚠️ 局限性**

研究聚焦于已训练并监督过金标准推理轨迹的模型，可能不适用于预训练阶段直接学习潜在推理的模型；此外，当模型错误时，推理轨迹的可解码率显著下降，表明解释性不稳定。

---

## 31. StableTTA: Training-Free Test-Time Adaptation that Improves Model Accuracy on ImageNet1K to 96%

**arXiv ID:** 2604.04552 | [PDF](https://arxiv.org/pdf/2604.04552v1)

**作者:** Zheng Li `[一作]` (New York Institute of Technology), Huanying Helen Gu `[通讯]` (New York Institute of Technology)

**通讯引用:** 633 | [OpenAlex ID](https://openalex.org/A5105836405)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 StableTTA，一种训练‑free 的测试时自适应方法，通过改进的图像预处理（专门化 mixup/CutMix）和对 logits 的非显著抑制（NSS）实现集成预测的稳定性和精度提升。

**💡 创新点**

揭示并解决了传统聚合策略间的冲突，提出了降低 logits 方差的两步策略：改进的 TTA 增强与 logit 后处理，从而在不增加模型参数的前提下获得显著准确率提升。

**🔧 技术方法**

使用 Hölder 条件分析 logits 方差、改进的图像增强（mixup/CutMix）、NSS logit 处理以及标准 TTA/集成对比实验。

**📊 数据集**

在 ImageNet‑1K 图像分类基准上进行评估。

**📈 对比分析**

与原始单模型、标准 TTA 及传统模型集成方式对比，StableTTA 在 ImageNet‑1K 上使 top‑1 准确率提升 11–33%，有 33 个模型超过 95%；如 MobileNetV3+StableTTA 超过 ViT 11.75% 的准确率，但参数与 GFLOPs 分别下降 97% 与 89.1%。

**⚠️ 局限性**

在顺序执行模式下推理延迟随专家数线性增长；并行模式需要 N 倍计算资源；目前仅在图像分类任务验证，其他任务（如分割、检测）待进一步扩展。

---

## 32. Element-based Formation Control: a Unified Perspective from Continuum Mechanics

**arXiv ID:** 2604.04027 | [PDF](https://arxiv.org/pdf/2604.04027v1)

**作者:** Kun Cao `[一作]` (Tongji University), Lihua Xie `[通讯]` (Nanyang Technological University)

**通讯引用:** 55852 | [OpenAlex ID](https://openalex.org/A5100365448)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于连续介质力学的离散弹性体模型，用形变梯度作为描述多智能体编队误差的统一量度，设计了可分布式实现的能量最小化控制律，并给出了翻译、旋转、尺度和相似性四类几何不变性的控制器。

**💡 创新点**

创新点包括：①将编队控制从稀疏图边转移到多维单形元（如三角形/四面体）上，引入形变梯度；②构造通用形变能量函数，统一描述刚性和拉普拉斯方法；③证明刚性约束是形变能量的稀疏投影，拉普拉斯控制等价于Dirichlet能量最小化；④给出能量梯度的封闭式推导与投影函数，完成不同不变性控制器的统一设计。

**🔧 技术方法**

使用的技术主要有：连续介质力学中的形变梯度、有限元形状矩阵、能量函数（Frobenius范数）、极分解、SVD、Dirichlet能量、拉普拉斯矩阵组装、半正定规划（SDP）求取权重、单积分器动力学、梯度下降、误差收敛分析。

**📊 数据集**

实验采用自定义的心形二维（6 机器人）和三维（7 机器人）编队，使用 Delaunay 三角化生成的单形元拓扑；未使用公开数据集，而是通过仿真验证。

**📈 对比分析**

与传统刚性基方法（距离、方位、RoD）进行对比，采用能量衰减率的方差（CoV）作为度量，结果显示：①所有控制器实现了指数收敛；②刚性基方法在几何失衡时收敛速率剧烈波动；③基于形变能量的控制器在不同扰动下保持了均匀、较低的 CoV，说明对几何失衡具有更强鲁棒性。

**⚠️ 局限性**

局限性包括：①需要完整的参考单形元几何信息；②单积分器动力学，未证明对非线性/惯性系统的稳定性；③权重选择对收敛速度影响大，负权重导致能量无界；④仿真规模有限，未验证大规模网络；⑤对实时通讯/计算资源需求尚未评估。

---

## 33. Styx: Collaborative and Private Data Processing With TEE-Enforced Sticky Policy

**arXiv ID:** 2604.04082 | [PDF](https://arxiv.org/pdf/2604.04082v1)

**作者:** Shixuan Zhao `[一作]` (Ohio State University), Zhiqiang Lin `[通讯]` (Ohio State University)

**通讯引用:** 5672 | [OpenAlex ID](https://openalex.org/A5026864098)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种面向多方协同数据处理的框架，将粘性策略与可信执行环境（TEE）结合，利用中间件沙箱实现数据在使用、生命周期以及动态协作过程中的完整安全保障。

**💡 创新点**

创新点包括：1）灰盒沙箱运行时，允许不可信的消费者程序在TEE内部运行，同时通过中间件完整控制其I/O，实现细粒度的输入/输出与程序约束检查；2）双通道管线设计，支持数据来源与生成数据的策略绑定，保证数据衍生物同样受到所有参与方的约束；3）架构无关的协议与框架，能够在分布式环境中扩展且兼容多种TEE与运行时。

**🔧 技术方法**

技术手段：Intel SGX 可信执行环境、WebAssembly (WASM) 运行时、粘性策略定义与绑定、可插拔的政策引擎、密钥托管与远程证明、沙箱化的中间件、双通道数据/策略管线。

**📊 数据集**

数据集与基准：使用 Wisconsin Breast Cancer 数据集进行 SVM 训练实验；在性能评测中使用 NBench 与 libonnx 的多模型基准。

**📈 对比分析**

性能比较：与 HTTPS 等传统加密通信相比，数据获取与政策检查的额外延迟约为 135 ms；在单机评测中，数据处理、密钥获取等每项延迟低于 2 ms；在分布式模拟中，峰值延迟仅数百毫秒；对 NBench，沙箱化导致 2.2× 的慢速，libonnx 则接近原生 1.03×，整体开销可接受。

**⚠️ 局限性**

局限性：1）无法在 GPU 等硬件加速器上执行策略；2）沙箱化与性能存在折中，AoT 编译可能失去完整隔离；3）TCB 包含框架、运行时与政策引擎，可信边界扩大；4）缺少正式验证与更完善的沙箱技术（如 SFI、容器化）。

---

## 34. Parent Selection Mechanisms in Elitist Crossover-Based Algorithms

**arXiv ID:** 2604.04083 | [PDF](https://arxiv.org/pdf/2604.04083v1)

**作者:** Andre Opris `[一作]` (University of Passau), Denis Antipov `[通讯]` (Sorbonne University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在跳跃（Jump_k）基准函数上，对 (μ+1) 遗传算法（GA）引入了以最远父母为选择的父代选择机制，并对其运行时间进行了理论分析。

**💡 创新点**

创新点在于提出了一种新的多样性度量 d_t（种群中最大汉明距离）和 m_t（具有该距离的相互不相交对的最大数目），并证明通过最远父母选择可以显著保持并利用种群多样性，从而大幅降低跳跃问题的搜索时间。

**🔧 技术方法**

主要技术包括基于概率的运行时间分析、负漂移定理、Markov 链模型、随机游走（赌博者破产）分析，以及对多样性度量的结构性和概率性质证明。

**📊 数据集**

使用的测试数据集为理论上的 Jump_k 基准函数（无真实数据集）。

**📈 对比分析**

与传统的 (μ+1) GA（无专门多样性维护机制、均匀交叉概率等）相比，本文的算法在 μ = O(4^k log n) 的情况下实现了 O(4^k n^{k-r}/p_c) 的期望评估次数（r ≤ k-1），显著优于此前最优的 O(n^k) 结果；即在常数交叉概率下，仅多项式提升而非指数提升。

**⚠️ 局限性**

局限性包括：需要 μ 与 k 的指数级关系（μ 需要大致 4^k），对父代选择的特定假设（必须保证最远父母被选中概率 Ω(1)），分析仅针对 Jump_k 问题；对其他目标函数或非精英 GA 的推广仍未得到证明。

---

## 35. HistoFusionNet: Histogram-Guided Fusion and Frequency-Adaptive Refinement for Nighttime Image Dehazing

**arXiv ID:** 2604.03800 | [PDF](https://arxiv.org/pdf/2604.03800v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 36. Incomplete Multi-View Multi-Label Classification via Shared Codebook and Fused-Teacher Self-Distillation

**arXiv ID:** 2604.04170 | [PDF](https://arxiv.org/pdf/2604.04170v1)

**作者:** Xu Yan `[一作]` (Shanghai Maritime University), Minghua Wan `[通讯]` (Shanghai Maritime University)

**通讯引用:** 1060 | [OpenAlex ID](https://openalex.org/A5035419130)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对视图与标签同时缺失的多视图多标签分类任务，提出SCSD框架，通过共享代码表实现离散一致表示，结合跨视图重建、基于标签相关性加权融合和融合教师自蒸馏提升模型鲁棒性。

**💡 创新点**

创新点包括：① 用共享代码表将不同视图映射到有限的离散空间，实现高效一致表示；② 跨视图重建进一步强化视图间一致性；③ 通过标签相关矩阵评估各视图预测质量并动态加权融合；④ 融合预测作为教师，指导视图分支学习的自蒸馏策略。

**🔧 技术方法**

主要技术：多视图共享代码表（vector quantization），跨视图重建损失，基于标签相关性的加权融合，融合教师自蒸馏（KL+BCELoss），straight‑through 估计等。

**📊 数据集**

使用五个公开多视图多标签数据集：Corel5k、Pascal07、Espgame、Iaprtc12、Mirflickr，采用 DenseSift、DenseHue、GIST、RGB、LAB、HSV 六种视图特征。

**📈 对比分析**

与八种现有双缺失方法（iMvWL、NAIM3L、DDINet、DICNet、MTD、SIP、RANK、DRLS）在 50% 视图缺失、50% 标签缺失、70% 训练数据的设定下进行对比，SCSD 在 AP、AUC、RL、HL、OE、Cov 等六个指标上普遍实现最优或接近最优的性能，特别是在 Espgame 与 Iaprtc12 的 AP 上分别提升 5.8% 与 8.2%。

**⚠️ 局限性**

局限性：共享代码表需要额外存储和计算（距离矩阵、更新），导致显存和推理时的计算开销；当视图缺失率极高时，可供对齐的跨视图信息不足，可能削弱共享代码表机制的泛化能力。

---

## 37. Darkness Visible: Reading the Exception Handler of a Language Model

**arXiv ID:** 2604.04756 | [PDF](https://arxiv.org/pdf/2604.04756v1)

**作者:** Peter Balogh `[一作]` `[通讯]`, Peter Balogh

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 GPT-2 Small 终层 MLP 进行可读路由程序分解，发现 27 个神经元负责路由，约 3040 个残差神经元存储知识。

**💡 创新点**

创新点在于揭示终层 MLP 的路由逻辑可读化，证明“知识神经元”是路由基础设施而非事实存储，并首次描述终层晶化现象。

**🔧 技术方法**

采用透明 GPT-2 封装、激活分解、Jaccard 相似度、统计显著性检验、bootstrap、ablation、积分梯度以及对比实验。

**📊 数据集**

数据集为 512k 词条的 WikiText-103，另外使用 160 条事实填空提示和 15 条 garden‑path 句子。

**📈 对比分析**

通过对各层的 ablation 与 PPL 变化比较，发现核心层输出对 PPL 影响极小，而 differentiator 贡献显著；跨层比较显示仅 L11 具备高 Jaccard 结构。

**⚠️ 局限性**

限制包括仅在 GPT-2 Small 上验证，使用单一文本域，garden‑path 试验样本不足，且跨模型迁移实验规模有限。

---

## 38. DebugHarness: Emulating Human Dynamic Debugging for Autonomous Program Repair

**arXiv ID:** 2604.03610 | [PDF](https://arxiv.org/pdf/2604.03610v1)

**作者:** Maolin Sun `[一作]` (Nanjing University), Baowen Xu `[通讯]` (Nanjing University)

**通讯引用:** 11671 | [OpenAlex ID](https://openalex.org/A5100331400)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于大语言模型的自动化漏洞修复框架DebugHarness，模拟人工动态调试流程来定位并修复C/C++安全漏洞。

**💡 创新点**

创新点在于将LLM的推理能力与交互式运行时状态 introspection（如GDB、记录回放、内存检查）相结合，并通过签名驱动初始化和闭环验证机制提升定位准确性。

**🔧 技术方法**

使用的技术包括：大语言模型（DeepSeek‑V3.2、Gemini‑3 Flash、GLM‑5等）、GDB+记录回放、Valgrind、LangChain等工具链，实现LLM驱动的调试指令生成、补丁合成与验证。

**📊 数据集**

实验采用SEC‑bench数据集，涵盖200个真实C/C++安全漏洞。

**📈 对比分析**

与PatchAgent、VulnResolver、SWE‑agent等基线比较，DebugHarness在SEC‑bench上达到约90% 的解决率，比最佳基线高30%+，平均成本约$0.09/漏洞，验证过程与迭代次数在可接受范围内。

**⚠️ 局限性**

局限性包括：对编译器优化敏感，优化会破坏调试信息；运行时调试开销较大；对多线程竞态等复杂并发缺陷仍有挑战。

---

## 39. On the Efficiency of Sinkhorn-Knopp for Entropically Regularized Optimal Transport

**arXiv ID:** 2604.03787 | [PDF](https://arxiv.org/pdf/2604.03787v1)

**作者:** Kun He `[一作]` (Renmin University of China), Kun He `[通讯]` (Renmin University of China)

**通讯引用:** 5586 | [OpenAlex ID](https://openalex.org/A5020278936)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

改进并加速Sinkhorn-Knopp算法在二维矩阵上求解平衡的收敛理论，提出了“well‑conditioned”与“well‑bounded”两种矩阵性质，证明在这些性质下算法收敛为线性而非传统的二次级别。

**💡 创新点**

首次把Sinkhorn算法的线性收敛与矩阵的well‑conditionedness（相对平衡度）和well‑boundedness（相对成本尺度）联系起来，并给出新的收敛界，减少对矩阵规模的依赖。

**🔧 技术方法**

利用矩阵分解、结构保持（dense structure）与差异控制、行列求永续子结构、以及对行列和的上界与下界分析等技术，结合矩阵行列和、永久子式和相对质量的控制。

**📊 数据集**

本工作为纯理论分析，并未使用具体实验数据集。

**📈 对比分析**

与之前的 Sinkhorn‑Knopp 收敛分析（如 𝒪(ε^{-2}) 的非线性收敛）做比较，理论上取得更强的 𝒪(ε^{-1}) 线性收敛，并且对矩阵规模无显著依赖。

**⚠️ 局限性**

主要局限是仅在满足 well‑conditioned/well‑bounded 条件的矩阵上适用，对极端稀疏或存在极端大/小成本的矩阵无法保证该收敛率。

---

## 40. BlazeFL: Fast and Deterministic Federated Learning Simulation

**arXiv ID:** 2604.03606 | [PDF](https://arxiv.org/pdf/2604.03606v1)

**作者:** Kitsuya Azuma `[一作]` (Institute of Science Tokyo), Takayuki Nishio `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 2523 | [OpenAlex ID](https://openalex.org/A5042195263)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了BlazeFL，一个轻量级的单节点联邦学习模拟框架，利用Python free‑threading实现共享内存执行并为每个客户端分配独立随机数流；

**💡 创新点**

创新点在于通过free‑threading减少IPC/序列化开销，同时实现高并发下的比特级可重复性；

**🔧 技术方法**

采用Python PEP 703/779 free‑threading、共享内存参数交换、独立 RNG 流、协议式接口和最小化依赖（Python 标准库 + PyTorch）；

**📊 数据集**

使用CIFAR‑10图像分类数据集，采用非 IID 分区（每个客户端两类）；

**📈 对比分析**

与 Flower+Ray 基线对比，在高性能服务器上，free‑threaded 模式在通信密集型工作（CNN）上实现 3.1× 加速，计算密集型模型提升 1.1–1.4×；在工作站服务器上差距缩小，但通信主导场景仍占优；重复实验 10 次显示准确率方差为 0，且与并行度无关；

**⚠️ 局限性**

局限性包括仅适用于单节点、固定软硬件环境，跨平台/多节点可重复性无法保证；需要用户自行确保视觉管道使用显式 RNG；生态尚未完全成熟，某些第三方库可能不支持最新 free‑threading。

---

## 41. Nonlinear Model Updating of Aerospace Structures via Taylor-Series Reduced-Order Models

**arXiv ID:** 2604.03788 | [PDF](https://arxiv.org/pdf/2604.03788v1)

**作者:** Nikolaos D. Tantaroudas `[一作]` (Institute of Communication and Computer Systems), Keith Worden `[通讯]` (University of Sheffield)

**通讯引用:** 27321 | [OpenAlex ID](https://openalex.org/A5017996489)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

结合非线性模型降阶与基底自适应进行翼箱面板结构模型更新

**💡 创新点**

将Cayley变换推广到复酉群，并在非线性ROM中加入三阶Taylor展开实现幅值相关更新

**🔧 技术方法**

非线性模型降阶（NMOR）、Cayley变换、Rayleigh阻尼、三阶Taylor展开

**📊 数据集**

Hollins翼箱面板实验数据及其70,890自由度有限元模型

**📈 对比分析**

与传统线性更新方案对比，NMOR在高振幅下保持MAC>0.999，计算速度提升约2–3倍

**⚠️ 局限性**

仅针对多项式（如立方）非线性；对非比例阻尼或更复杂非线性系统需更高阶展开

---

## 42. MultiPress: A Multi-Agent Framework for Interpretable Multimodal News Classification

**arXiv ID:** 2604.03586 | [PDF](https://arxiv.org/pdf/2604.03586v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 43. TAPE: A two-stage parameter-efficient adaptation framework for foundation models in OCT-OCTA analysis

**arXiv ID:** 2604.04571 | [PDF](https://arxiv.org/pdf/2604.04571v1)

**作者:** Xiaofei Su `[一作]` (Nankai University), Mingzhu Sun `[通讯]` (Nankai University)

**通讯引用:** 1462 | [OpenAlex ID](https://openalex.org/A5101923674)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了TAPE两阶段参数高效微调框架，解决医学图像域迁移与任务不匹配问题；

**💡 创新点**

创新点在于将PEFT与掩码图像建模相结合，先在自监督域适配阶段引入域知识，再在任务适配阶段微调分割头，显著提升参数效率和跨病种泛化；

**🔧 技术方法**

使用LoRA/ViT-Adapter/VPT等PEFT方法、MAE与RETFound基础模型、掩码图像建模自监督学习以及交叉熵分割损失；

**📊 数据集**

采用公开的OCTA-500数据集，包含AMD、DR、RVO和正常四类样本；

**📈 对比分析**

与六种基线（单阶段与两阶段微调、全参数微调）以及三种从零开始的分割网络对比，TAPE在mDice/mIoU指标上均优于其他方法，尤其在RETFound上达到最优；

**⚠️ 局限性**

局限性包括对数据量和计算资源仍有一定需求，且目前仅验证了层分割任务，缺乏对少样本学习与其他多模态任务的进一步探索。

---

## 44. GAIN: Multiplicative Modulation for Domain Adaptation

**arXiv ID:** 2604.04516 | [PDF](https://arxiv.org/pdf/2604.04516v1)

**作者:** Hengshuai Yao `[一作]` (University of Alberta), Guan Wang `[通讯]` (Sapient Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于乘法调制的领域适配方法 GAIN，利用对注意力输出投影和 FFN 的对角缩放来重新强调已有特征，避免加入新方向导致的遗忘；

**💡 创新点**

创新点在于通过乘法对角缩放严格保持预训练权重的输出子空间，理论上消除“入侵维度”，从而在适配过程中实现零遗忘和正向迁移；

**🔧 技术方法**

使用对角缩放矩阵 S 对 W_O（以及可选的 W_down）进行学习，训练时冻结预训练权重，最后将缩放直接吸收到权重中；

**📊 数据集**

在多种领域（医学摘要 PubMedQA、维基百科、金融新闻、PG‑19、法律文本等）以及八个连续适配顺序上进行实验，评估了 GPT‑2、Mistral‑7B、Qwen‑2.5‑7B、Llama‑2‑13B、Llama‑2‑70B 等模型；

**📈 对比分析**

与 LoRA 相比，GAIN‑FFN 在所有模型上实现了 7–13% 的先前域性能提升（相对 LoRA 的 18–36% 退化），并在七个下游基准上保持或提升准确率，显示出零遗忘与正向迁移的优势；

**⚠️ 局限性**

局限性包括假设预训练特征足以覆盖所有目标域、未在指令调优模型或极长的连续适配序列（≫8）上进行评估。

---

## 45. A Semi-Automated Annotation Workflow for Paediatric Histopathology Reports Using Small Language Models

**arXiv ID:** 2604.04168 | [PDF](https://arxiv.org/pdf/2604.04168v1)

**作者:** Avish Vijayaraghavan `[一作]` (Imperial College London), Neil Sebire `[通讯]` (University College London)

**通讯引用:** 47901 | [OpenAlex ID](https://openalex.org/A5037501835)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究设计了一套半自动化注释工作流，利用小型语言模型从小儿肾活检病理报告中提取结构化信息

**💡 创新点**

创新点在于将临床专家制定的实体指南和少量示例结合到提示工程中，既提高了准确率，又保持了低资源消耗

**🔧 技术方法**

核心技术包括指令调优的小型语言模型（Gemma 2、Llama 3.2、Phi‑3.5 等）以及基于问题‑回答的提问框架和自动争议建模

**📊 数据集**

使用了来自英国Great Ormond Street Hospital的 2,111 篇肾活检报告（其中 400 篇已人工标注）

**📈 对比分析**

与 spaCy、BioBERT‑SQuAD、RoBERTa‑SQuAD、GLiNER 等传统方法比较，Gemma 2 在两步加指南配置下达成 84.3% 的整体准确率，显著高于其它模型且单条报告处理时间仅 26–72 秒

**⚠️ 局限性**

主要限制包括对复杂字符串实体的评估依赖于 LM‑as‑a‑Judge，存在约10% 的误判；模型在长文本处理时易出现解析错误；且仅在肾活检领域验证，跨科室推广仍需进一步评估

---

## 46. Uncertainty-Aware Test-Time Adaptation for Cross-Region Spatio-Temporal Fusion of Land Surface Temperature

**arXiv ID:** 2604.04153 | [PDF](https://arxiv.org/pdf/2604.04153v1)

**作者:** Sofiane Bouaziz `[一作]` (Université d'Orléans), Rachid Nedjai `[通讯]` (Université d'Orléans)

**通讯引用:** 310 | [OpenAlex ID](https://openalex.org/A5030805896)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了一种基于不确定性感知的测试时适应（TTA）框架，用于将预训练的空间-时间融合模型迁移到不同地区的陆面温度估计任务。

**💡 创新点**

创新点包括：①针对回归任务设计的基于模型不确定性、土地利用/覆盖一致性和偏差一致性的联合损失；②仅更新融合模块而冻结编码器与解码器，以降低训练成本；③无需源数据或目标标签即可进行适应。

**🔧 技术方法**

使用的技术包括：MC Dropout 估计模型不确定性、Pearson 相关系数损失、均值偏差一致性损失、Adam 优化器以及基于 EFD 结构的 WGAST 模型。

**📊 数据集**

数据集：模型在奥尔良（法国）数据上预训练，随后在罗马、开罗、马德里、蒙彼利埃四个目标地区进行测试，使用 Landsat‑8、Sentinel‑2、Terra MODIS 等卫星观测数据。

**📈 对比分析**

通过与未适应的预训练 WGAST 模型对比，实验显示 RMSE 和 MAE 在四个地区平均分别降低约 24% 与 28%，仅需 10 次适应周期即可获得显著提升。

**⚠️ 局限性**

局限性：仅在单一回归任务上验证，未检验对其他遥感回归任务的泛化；依赖 MC Dropout 估计不确定性，计算成本相对较高；只更新融合模块可能无法完全补偿极端域迁移；缺乏对长时间跨度或不同训练/测试时序的评估。

---

## 47. On Polycyclic Codes over $\frac{\mathbb{F}_{p^m}[u]}{\langle u^t \rangle}$ and their Cardinalities

**arXiv ID:** 2604.03991 | [PDF](https://arxiv.org/pdf/2604.03991v1)

**作者:** Akanksha Tiwari `[一作]` (Indian Institute of Technology Delhi), Ritumoni Sarma `[通讯]` (Ohio University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了环𝔽_p^m[u]/⟨ u^t ⟩上的多环码及其相关的扭转码，特别是当t=4时的情况。

**💡 创新点**

创新点在于通过环论结果获得了理想的生成元，并扩展了之前的结果，涵盖了任意t值和任意多项式ω(x)的情况。

**🔧 技术方法**

使用了环同态和理想生成的基本交换代数技术。

**📊 数据集**

使用了环𝔽_p^m[u]/⟨ u^t ⟩和多项式ω(x)=f(x)^p^s，其中f(x)是𝔽_p^m上的不可约多项式。

**📈 对比分析**

通过与已有的文献对比，本文的方法在生成理想和计算扭转码的数量上表现出更广泛的适用性，尤其是在t=4的情况下，能够精确描述16种理想类型。

**⚠️ 局限性**

限制在于计算L_i的过程可能会变得繁琐和具有挑战性，未来的研究可以集中在开发更高效的算法上。

---

## 48. The Tool Illusion: Rethinking Tool Use in Web Agents

**arXiv ID:** 2604.03465 | [PDF](https://arxiv.org/pdf/2604.03465v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 49. Many Preferences, Few Policies: Towards Scalable Language Model Personalization

**arXiv ID:** 2604.04144 | [PDF](https://arxiv.org/pdf/2604.04144v1)

**作者:** Cheol Woo Kum `[一作]`, Swati Gupta `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 6183 | [OpenAlex ID](https://openalex.org/A5078459021)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为 PALM 的算法，用来构造一个小型的 LLM 组合（portfolio），能够在多目标奖励空间中近似最优行为，从而实现可扩展的用户个性化。

**💡 创新点**

创新点在于给出了组合大小与近似质量的理论上限，并通过混合乘法与加法网格以及贪心剪枝方法，证明该组合可覆盖整个概率单纯形，且不随用户数量增长；这在 LLM 微调领域首次实现了组合规模与性能的可控权衡。

**🔧 技术方法**

技术上使用了多目标强化学习（PPO/GRPO/MOD）、线性标量化、多目标奖励投影、网格化权重采样、最小集覆盖求解（贪心/整数规划）以及 KL 正则化等手段。

**📊 数据集**

实验数据集包括：RLVR‑GSM（两个奖励：简洁度、帮助性）、Safety Alignment（两个奖励：更好、安全性）以及 Helpful Assistants（三个奖励：帮助性、无害性、幽默性），使用 Qwen2.5‑3B‑Instruct、ALPACA‑7B 和 LLaMA‑2‑7B 等 LLM。

**📈 对比分析**

与均匀网格和随机采样基线相比，PALM 在多目标近似误差（ε、δ）和策略使用分布（perplexity）上都表现更好；实验表明，少于七个策略即可达到 1.5% 以内的子最优度。

**⚠️ 局限性**

局限性包括：需要对每个权重向量求解近似最优策略的算子（计算成本高）；理论假设奖励非负且可正则化；对高维奖励空间仍可能需要大量网格点；以及在实际部署中需解决模型互斥和调度等系统问题。

---

## 50. Separator for $c$-Packed Segments and Curves

**arXiv ID:** 2604.04011 | [PDF](https://arxiv.org/pdf/2604.04011v1)

**作者:** Sariel Har-Peled `[一作]` `[通讯]`, Sariel Har-Peled

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

本文提出了一种简单算法，能够在期望线性时间内为c‑packed线段集合构造一个平衡球形分隔器，使得该球仅切割O(c)条线段并将剩余线段分为两侧各至少n/2c条；

**💡 创新点**

创新点在于提供了比以往更简洁的证明思路，并通过随机半径抽样与Markov不等式直接得到期望线性时间的分隔器，避免了复杂的几何构造；

**🔧 技术方法**

主要技术包括c‑packed性定义下的积分界限、球体内端点数量上界、随机取半径抽样、Markov不等式以及对球体大小的2-近似计算；

**📊 数据集**

文章未使用真实数据集，全部论证为理论分析；

**📈 对比分析**

由于缺乏实验数据，本文未与其他方法进行性能比较，只说明算法在理论上满足期望线性时间和O(c)切割数的性质；

**⚠️ 局限性**

局限性包括：仅适用于c‑packed线段集合，且需要已知或可估计的维度相关常数（如doubling constant），对非c‑packed集合或高维情况的适用性不明；

---

## 51. Generative models for decision-making under distributional shift

**arXiv ID:** 2604.04342 | [PDF](https://arxiv.org/pdf/2604.04342v1)

**作者:** Xiuyuan Cheng `[一作]` (Duke University), Yao Xie `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 3726 | [OpenAlex ID](https://openalex.org/A5047736740)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了生成模型在运筹学中处理分布偏移的理论与实践框架，强调通过可逆映射、速度场或分数场构造决策相关的分布，而非仅仿真数据；

**💡 创新点**

创新点在于将生成模型视为概率空间优化工具，提出基于 Wasserstein 空间、最优传输及后验误差传递的理论保证，并将其与场景生成、稳健决策、条件更新及跨域修正等 OR 任务对接；

**🔧 技术方法**

主要技术包括连续时间流模型、分数基扩散模型、流匹配、最优传输理论、Wasserstein 距离、粒子方法以及梯度上升-下降的运输映射优化；

**📊 数据集**

实验使用了亚特兰大地区电力停电时间序列数据（10 维停电计数）和美国六大ETF的一年期收益数据，用于场景生成与稳健投资案例；

**📈 对比分析**

在停电案例中，生成模型的最大均值散度（MMD）仅为 0.015，且能捕捉县际相关性；在稳健投资案例中，生成的最坏情景提高了测试集累计财富的稳健性，尤其在调参 γ=0.1 时表现最佳；

**⚠️ 局限性**

局限性包括理论证明多停在理想化连续模型和粒子系统，离散神经网络实现与采样误差仍未完全解决，且缺乏对高维、非光滑约束等实际 OR 场景的完整理论与算法支持。

---

## 52. Hierarchical Co-Embedding of Font Shapes and Impression Tags

**arXiv ID:** 2604.04158 | [PDF](https://arxiv.org/pdf/2604.04158v1)

**作者:** Yugo Kubota `[一作]` (Kyushu University), Seiichi Uchida `[通讯]` (Kyushu University)

**通讯引用:** 9805 | [OpenAlex ID](https://openalex.org/A5051387162)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个在双曲空间中对字体图像和印象标签进行共嵌入的框架，通过蕴含关系（字体-印象、印象-印象）建模字体与印象之间的对应关系，并利用双曲锥实现风格特异性的可解释度量。

**💡 创新点**

创新点包括：①将蕴含关系引入双曲共嵌入，利用双曲锥实现层次化的风格特异性度量；②使用双曲空间的径向坐标直观衡量印象标签的风格特异性；③在双曲空间中结合双向对比损失和蕴含损失，兼顾跨模态对齐与层次组织。

**🔧 技术方法**

使用技术包括：双曲几何（Lorentz模型）、双向对比学习、蕴含锥损失、ResNet‑18 字体编码器、Transformer 文本编码器（CLIP预训练）、双曲空间的指数映射与锥角计算。

**📊 数据集**

使用 MyFonts 数据集：18,815 款字体，1,824 个印象标签，最终筛选 631 个出现频率≥50 的标签，划分为 13,461/1,667/1,663 的训练/验证/测试集。

**📈 对比分析**

通过对比更新后的 Impression‑CLIP+ 与 Cross‑AE+ 基线，使用 mAP（单/多标签）和 nDCG@100 的双向检索指标评估。实验结果表明，在印象→字体检索的 mAP 及 nDCG@100 上均优于基线，尤其在单标签检索的 mAP 上提升显著；逐步加入蕴含损失后性能持续提升。

**⚠️ 局限性**

局限性包括：①仅处理单词标签集合，未考虑自然语言描述；②实验仅在英文字体上进行，缺乏多语言或非拉丁脚本的验证；③蕴含关系与层次化结构假设可能不适用于所有印象；④缺乏人类主观评估来验证风格特异性量化的有效性。

---

## 53. Hypothesis Graph Refinement: Hypothesis-Driven Exploration with Cascade Error Correction for Embodied Navigation

**arXiv ID:** 2604.04108 | [PDF](https://arxiv.org/pdf/2604.04108v1)

**作者:** Peixin Chen `[一作]` (Harbin Institute of Technology), Qing Li `[通讯]` (Beijing Institute for General Artificial Intelligence)

**通讯引用:** 38448 | [OpenAlex ID](https://openalex.org/A5100404176)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Hypothesis Graph Refinement（HGR）框架，利用 VLM 生成前沿语义假设并在验证失败时通过级联纠正删减错误子图，以实现语义驱动的高效探索与错误消除。

**💡 创新点**

创新点在于：① 将前沿预测建模为可回溯的假设节点并在依赖 DAG 中记录推导关系；② 设计基于语义残差的验证与级联纠正机制，能在发现错误时完整剔除错误子图，避免错误累积。

**🔧 技术方法**

核心技术包括：基于 GPT‑4o 或其它 VLM 的语义预测；依赖 DAG 的图结构与前沿检测；语义残差度量（类别、视觉特征、对象交叉相似度）；级联纠正算法；以及基于目标关联、行进成本与不确定度的探索评分。

**📊 数据集**

在三个基准上评测：GOAT‑Bench（多模态长期导航）、A‑EQA（主动实体问答）和 EM‑EQA（情节记忆问答）；实验中使用 Habitat‑Sim 环境、ScanNet 与 HM3D 场景。

**📈 对比分析**

相较于 3D‑Mem、ConceptGraph 等基线，HGR 在 GOAT‑Bench 上获得 72.41% 成功率与 56.22% SPL（分别比 3D‑Mem 提升 +3.31% 与 +7.32%）；在 A‑EQA、EM‑EQA 上也实现了 LLM‑Match 与 SPL 的显著提升，验证了级联纠正与语义假设的有效性。

**⚠️ 局限性**

局限性包括：① 级联纠正依赖残差测试的准确性，误判会导致错误保留或误删；② 预测质量受限于 VLM，对镜面、透明表面等视觉复杂情况仍易失误；③ 当前框架假设环境静态，缺乏对动态变化的处理。

---

## 54. Spatiotemporal-Aware Bit-Flip Injection on DNN-based Advanced Driver Assistance Systems

**arXiv ID:** 2604.03753 | [PDF](https://arxiv.org/pdf/2604.03753v1)

**作者:** Taibiao Zhao `[一作]` (Louisiana State University), Xugui Zhou `[通讯]` (Louisiana State University)

**通讯引用:** 89 | [OpenAlex ID](https://openalex.org/A5084919916)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一个时空感知位翻转注入框架 STAFI，用于揭示 DNN 驱动的 ADAS 系统中安全关键的参数漏洞。

**💡 创新点**

创新点在于结合 Progressive Metric-guided Bit Search (PMBS) 与 Context-based Critical Fault Time Identification (CFTI)，实现空间位与时间上下文双向定位，显著提升危害诱发率。

**🔧 技术方法**

使用了 Taylor 重要性分析、度量导向位搜索、系统理论过程分析（STPA）触发规则、CARLA+OpenPilot 闭环仿真、统计对比等技术。

**📊 数据集**

数据集方面，利用 comma2k19 离线数据训练 PMBS，并在 CARLA 生成的四种驾驶场景下进行闭环仿真。

**📈 对比分析**

通过与随机注入、TGFI-Top50/20 等基线对比，STAFI 在前向碰撞、车道偏离等危害率上提升至约 42%/76%，事故率提升约 29.56 倍，TTH/TTA 明显缩短。

**⚠️ 局限性**

局限性包括仅关注 DRAM 位翻转，未考虑传感器错误、通信失败或软件缺陷；且情景依赖性强，未覆盖更广泛环境。

---

## 55. Explainability-Guided Adversarial Attacks on Transformer-Based Malware Detectors Using Control Flow Graphs

**arXiv ID:** 2604.03843 | [PDF](https://arxiv.org/pdf/2604.03843v1)

**作者:** Andrew Wheeler `[一作]` (Tennessee Tech University), Maanak Gupta `[通讯]` (Tennessee Tech University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在 RoBERTa 基础的控制流图(CFG)恶意软件检测模型上，提出了基于可解释性信息的白盒对抗迭代攻击；

**💡 创新点**

创新点在于利用集成梯度对模型输出进行词级和词语级归因，针对高贡献函数调用进行合成外部 DLL 替换，从而实现对检测模型的高成功率欺骗；

**🔧 技术方法**

使用的技术包括 RoBERTa Transformer、Captum 解释器、CFG 线性化、DFS 图遍历、集成梯度归因、Python r2pipe 进行 CFG 提取、以及多轮对抗迭代；

**📊 数据集**

实验数据集为 Windows PE 程序，分别使用小型（2,900 benign / 6,679 malicious）和大型（54,154 benign / 94,123 malicious）两套数据；

**📈 对比分析**

通过对比攻击前后的检测准确率，结果显示在小型数据集上成功率可达约 98-100%，在大型数据集上亦能保持 85-97% 的成功率；

**⚠️ 局限性**

局限性包括对抗样本仅在白盒场景下有效，攻击需多轮迭代且对函数数量有限制，且生成对抗样本的可执行性（外部 DLL 引入）在实际部署中可能受限；

---

## 56. Memory Intelligence Agent

**arXiv ID:** 2604.04503 | [PDF](https://arxiv.org/pdf/2604.04503v1)

**作者:** Jingyang Qiao `[一作]` (East China Normal University), Yuan Xie `[通讯]` (East China Normal University)

**通讯引用:** 31208 | [OpenAlex ID](https://openalex.org/A5100385336)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Memory Intelligence Agent (MIA)，一种基于 Manager‑Planner‑Executor 结构的记忆框架，提升深度研究代理的推理和自我进化能力。

**💡 创新点**

创新点在于①将脑启发的短时/长期记忆机制与 Planner/Executor 交互，压缩历史轨迹为结构化工作流并转化为可学习参数；②两阶段交替强化学习耦合 Planner 与 Executor，促进计划生成与执行理解共进化；③在线测试时学习与无监督评估框架，实现持续自我进化。

**🔧 技术方法**

使用大规模 LLM（Qwen3‑32B、Qwen2.5‑VL‑7B 等）做 Planner/Executor，ReAct 交互、GRPO 强化学习、多维检索（语义相似、价值、频率）以及三位评审+AC 结构化评价等技术。

**📊 数据集**

实验覆盖多模态 Benchmark（FVQA-test、InfoSeek、SimpleVQA、LiveVQA、MMSearch、In‑house 1/2）和文本 Benchmark（SimpleQA、2Wiki、HotpotQA、GAIA）。

**📈 对比分析**

与闭源 GPT‑5.4、Gemini‑3‑Flash 及开源 Qwen 系列和记忆基线（RAG、ReasoningBank 等）对比，MIA 在多模态平均准确率 53.6%、文本平均 53.5%，在各 Benchmark 上比最佳记忆方法提升 5–9%，甚至超过部分大型闭源模型，显示显著性能提升。

**⚠️ 局限性**

局限性包括：依赖大量预训练 LLM 与工具环境；对无监督场景的性能仍略低于有监督版；对极长历史上下文的处理仍有限；尚未验证在更复杂动态环境中的可扩展性。

---

## 57. TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

**arXiv ID:** 2604.04921 | [PDF](https://arxiv.org/pdf/2604.04921v1)

**作者:** Weian Mao `[一作]` (Massachusetts Institute of Technology), Yukang Chen `[通讯]` (NVIDIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为TriAttention的KV缓存压缩方法，利用预RoPE空间中查询/键向量的集中性与三角级数来估计键的重要性，从而在长文本推理中显著降低KV内存占用并提升推理速度。

**💡 创新点**

核心创新是发现并利用预RoPE空间中查询/键向量的固定中心（Q/K集中性），通过三角级数预测键与查询的距离偏好，构造更稳健的键重要性评分，并自适应平衡三角级数与向量范数两种信号。

**🔧 技术方法**

技术手段包括：预RoPE向量中心统计、三角级数重建注意力分布、基于均值向量长度的自适应权重、窗口式批量裁剪、分组查询注意力的归一化聚合等。

**📊 数据集**

主要评测数据集为数学推理基准AIME（2024/2025）、MATH 500；此外还使用LongBench、RULER等通用长文本任务作为对比。

**📈 对比分析**

与全注意力、SnapKV、R-KV等基线比较显示，TriAttention在保持与全注意力相当的推理准确率（如AIME 40.8%）的同时，吞吐量提升2.5倍、KV内存降低10.7倍；在相同内存预算下，TriAttention准确率提升约15%（AIME）和8%（MATH）。

**⚠️ 局限性**

局限性包括：需离线统计Q/K中心，适配不同模型时可能需要重新校准；在极深递归或极大上下文时性能仍有下降风险；并且对非预RoPE模型的适用性尚未验证。

---

## 58. What Makes Good Multilingual Reasoning? Disentangling Reasoning Traces with Measurable Features

**arXiv ID:** 2604.04720 | [PDF](https://arxiv.org/pdf/2604.04720v1)

**作者:** Dayeon Ki `[一作]` (University of Maryland), Marine Carpuat `[通讯]` (University of Maryland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多语言推理轨迹进行系统分析，定义 16 个可测量的推理特征，利用逻辑回归量化其与答案准确率的关联，训练稀疏自编码器自动发现潜在推理概念，并以这些特征在测试时进行 best‑of‑n 选择，探究其在多语言环境中的有效性。

**💡 创新点**

①提出英语中心化奖励假设不一定通用，发现不同语言对相同特征的影响往往相反；②将手工特征与自动编码器相结合，系统挖掘并验证语言特定的有效推理模式；③为多语言推理提供可自适应的奖励设计思路。

**🔧 技术方法**

使用逻辑回归（含交互项）、稀疏自编码器（SAE）、GPT‑4o 进行步骤/流注释、COMET‑QE 与词嵌入相似度评估、Pass@1 评价以及最佳‑of‑n 选取策略。

**📊 数据集**

两大多语言数学推理基准：MGSM‑Rev2（中学水平）和 AIME 2024–25（高中竞赛级），覆盖 10 种语言（英语、孟加拉语、德语、西班牙语、法语、俄语、斯瓦希里语、泰卢固语、泰语、中文）。

**📈 对比分析**

通过对比英语与非英语的特征效应、模型与语言的差异，以及在 best‑of‑n 选择中的 Pass@1，发现直接/间接效用和结果整合等特征在 AIME 上可提升约 10% 准确率；结构/语义相似度仅在部分模型上微幅提升，且英语中心化奖励并非最佳，强调需考虑语言特定的推理模式。

**⚠️ 局限性**

仅研究数学推理，手工特征集合不完整；依赖 GPT‑4o 注释，跨语言可靠性尚未充分验证；实验仅涉及四个 LRM，结果可能不适用于其他体系；整体实验规模和领域范围有限。

---

## 59. Beyond Task-Driven Features for Object Detection

**arXiv ID:** 2604.03839 | [PDF](https://arxiv.org/pdf/2604.03839v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 60. The limits of bio-molecular modeling with large language models : a cross-scale evaluation

**arXiv ID:** 2604.03361 | [PDF](https://arxiv.org/pdf/2604.03361v1)

**作者:** Yaxin Xu `[一作]` (Southern University of Science and Technology), Zhixiang Ren `[通讯]` (Pengcheng Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了跨尺度生物分子评估基准 BioMol-LLM-Bench，包含 26 个下游任务，覆盖四个难度级别，并集成了计算工具和自动评估管道。

**💡 创新点**

创新点在于：①系统化的跨尺度评估框架；②引入专业化工具调用与参数解析；③对不同 LLM 架构（Transformer、Mamba、MoE）及训练策略（CoT、SFT、CPT）进行对比；④针对长序列生物分子任务提出混合 mamba‑attention 方案。

**🔧 技术方法**

使用大语言模型（LLM）技术，链式推理（CoT）与工具调用、监督微调（SFT）、持续预训练（CPT）等方法，结合多模态输入（SMILES、FASTA、文本）。

**📊 数据集**

数据来源于多种公开基准与数据库：MMLU‑Pro、USPTO‑50K、DrugBank、PDB、QM9、Tox21、ProteinGym 等，经过去重、结构校验、分层抽样得到高质量多尺度任务集。

**📈 对比分析**

通过 13 个主流 LLM（包括 NVIDIA‑Nemotron‑Nano‑9B‑v2、DeepSeek‑v3.1、Qwen‑3‑14B 等）在 26 任务上进行统一评测；结果显示：mamba‑attention 架构在多尺度任务中表现最佳；工具集成显著提升回归与分类精度；LLM 在分类任务上表现良好，但在长序列回归任务上仍低效。

**⚠️ 局限性**

局限性包括：1）长序列回归任务（如 MOL_Thermo、PROT_Mutation）性能仍不佳；2）CoT 微调对部分任务无显著提升；3）模型在推理过程中往往缺乏真实机制性解释，易出现自相矛盾的推理；4）专属微调导致泛化能力下降。

---

## 61. Structural Segmentation of the Minimum Set Cover Problem: Exploiting Universe Decomposability for Metaheuristic Optimization

**arXiv ID:** 2604.03234 | [PDF](https://arxiv.org/pdf/2604.03234v1)

**作者:** Isidora Hernández `[一作]` (Austral University of Chile), Cristóbal A. Navarro `[通讯]` (Austral University of Chile)

**通讯引用:** 410 | [OpenAlex ID](https://openalex.org/A5088815725)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

对最小集合覆盖问题进行宇宙分段预处理，利用并查集拆分成独立子问题，并将其嵌入GRASP元启发式中。

**💡 创新点**

提出“宇宙可分解性”概念和基于并查集的高效分段方法，允许在不破坏可行性的前提下将大规模实例拆分成更小的子实例。

**🔧 技术方法**

使用并查集（union–find）进行连通分量检测，GRASP（Greedy Randomized Adaptive Search Procedure）元启发式，位图位运算与简洁的位级集合表示，以及多核共享内存并行实现。

**📊 数据集**

在OR‑Library的标准无权实例（scpe、scpclr、scpcyc、rail 等）以及大规模随机合成实例上进行实验。

**📈 对比分析**

与贪心、标准GRASP以及并行GRASP进行对比，结果显示GRASP‑UF在大规模实例上显著降低RPD（≤20%）并获得约1–2倍的速度提升，尤其在32核环境下。

**⚠️ 局限性**

分段预处理占总时间较大，导致在极大实例上仍是瓶颈；强制平衡划分策略未能有效提高性能。

---

## 62. Teaching Empathy in Software Engineering Education in the Age of Artificial Intelligence

**arXiv ID:** 2604.04689 | [PDF](https://arxiv.org/pdf/2604.04689v1)

**作者:** Ronnie de Souza Santos `[一作]` (University of Calgary), Italo Santos `[通讯]` (University of Hawaii at Manoa)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

通过七位软件工程教授的经验性卡片收集与民族志式讨论，系统化归纳出五类将同理心嵌入技术课程的教学实践框架

**💡 创新点**

首次构建以社会情境、偏见与可访问性、用户代表性、利益相关者角色与反思反馈为核心的五大同理心教学范式，填补了学术与实践之间的碎片化空白

**🔧 技术方法**

采用民族志与自动民族志方法，结合卡片分类、主题编码与迭代式归纳分析，对参与者的实践进行深度阐释

**📊 数据集**

主要数据来自参与者的教学实践卡片、个人反思笔记以及集体讨论记录，未使用公开数据集

**📈 对比分析**

无定量性能评估；通过对照系统文献回顾与参与者实践，进行质性对比，识别共性与差异，构建概念性分类

**⚠️ 局限性**

样本规模仅七名教授，研究时段为单次三小时聚会，缺乏课堂观察与长期跟踪，可能存在研究者偏见与情境可迁移性受限

---

## 63. Beyond Standard Benchmarks: A Systematic Audit of Vision-Language Model's Robustness to Natural Semantic Variation Across Diverse Tasks

**arXiv ID:** 2604.04473 | [PDF](https://arxiv.org/pdf/2604.04473v1)

**作者:** Jia Chengyu `[一作]` (Zhejiang University of Technology), Isao Echizen `[通讯]` (National Institute of Informatics)

**通讯引用:** 5777 | [OpenAlex ID](https://openalex.org/A5044556342)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了针对视觉-语言模型（VLM）的自然对抗性评估框架，覆盖零样本图像分类、语义分割与视觉问答三类任务，系统评估了多种对抗场景（排版攻击、ImageNet-A、自然语言诱导攻击）；

**💡 创新点**

创新点包括：①首次统一考察多任务与多对抗类型下VLM的鲁棒性；②构造并引入三类真实对抗数据集；③结合可解释分析揭示模型失效模式；④发现对抗微调的CLIP反而放大自然对抗弱点；

**🔧 技术方法**

采用对抗数据集构造技术、零样本CLIP分类协议、CLIPSeg/BLIP2解码器、VQA-ViL适配，以及CAM、稀疏自编码器（SAE）和注意力头可视化等方法；

**📊 数据集**

使用的主要数据集包括ImageNet‑1K、ImageNet‑A、RTA‑100（排版攻击）、Animal‑10、PhraseCut、VQA‑v2、Grounded‑Segment‑Anything等；

**📈 对比分析**

通过在同一评估协议下比较22个VLM（CLIP、Robust‑CLIP、SigLIP2、BLIP2等）在准确率、IoU、VQA准确率等指标上的表现，发现SigLIP2在大多数对抗场景下鲁棒性最佳，Robust‑CLIP在自然对抗下往往逊色于标准CLIP；

**⚠️ 局限性**

局限性：仅评估冻结的图像编码器与任务适配器，未覆盖端到端系统；对抗数据集未能覆盖全部现实情形；部分失败模式分散且难以定位，难以针对性改进；

---

## 64. High-Stakes Personalization: Rethinking LLM Customization for Individual Investor Decision-Making

**arXiv ID:** 2604.04300 | [PDF](https://arxiv.org/pdf/2604.04300v1)

**作者:** Yash Ganpat Sawant `[一作]` `[通讯]`, Yash Ganpat Sawant

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建并部署了一个面向个人投资者的AI增强投资组合管理系统，整合了持续的投资论点管理、每日信念跟踪、行为记忆提取与漂移检测等组件；

**💡 创新点**

首次系统性阐述了个人投资领域对LLM个性化的四大核心挑战（行为记忆复杂性、论点一致性漂移、风格–信号张力、无确定真相的对齐），并针对每一挑战提出了对应的架构响应与实践经验；

**🔧 技术方法**

采用LLM结构化评估调用、检索式记忆增强、基于规则的信念提取与转化、主题化的行为记忆分类、漂移检测与模式匹配等技术，构建了“活跃论点”抽象与固定增量信念轨迹；

**📊 数据集**

主要使用投资者交互日志、投资组合持仓与市场行情数据（包括价格、波动率、公司财报等），以及内部自定义的行为规则与投研笔记，未公开使用公开数据集；

**📈 对比分析**

本文为观点性论文，未给出标准化对比或量化性能指标，主要通过案例演示和系统内部评估（如过程质量打分、行为一致性警示）来说明改进效果；

**⚠️ 局限性**

局限性包括：缺乏可观测的确定真相导致评估困难、投资决策延迟与噪声导致的反馈挑战、对齐与用户风格的冲突需要精细平衡、当前技术难以充分捕捉高阶行为模式与信念演化等。

---

## 65. DriveVA: Video Action Models are Zero-Shot Drivers

**arXiv ID:** 2604.04198 | [PDF](https://arxiv.org/pdf/2604.04198v1)

**作者:** Mengmeng Liu `[一作]` (University of Twente), Hao Cheng `[通讯]` (University of Twente)

**通讯引用:** 7370 | [OpenAlex ID](https://openalex.org/A5002932429)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并训练了一种名为 DriveVA 的统一视频-动作世界模型，能够在同一潜在空间内同时预测未来视频帧与车辆轨迹，实现闭环驾驶规划。

**💡 创新点**

创新点包括：① 将 Diffusion Transformer 作为共享潜在生成器，统一生成视频潜在与动作令牌，显著提升视频-轨迹一致性；② 引入视频继续策略，保持长时段的时序连贯性；③ 采用显式视频监督代替传统的稀疏动作监督，强化对物理动态的学习；④ 在零射传输设置下实现跨数据集与跨域（真实-仿真）的优秀泛化。

**🔧 技术方法**

使用的技术包括：大规模预训练视频生成模型 Wan2.2‑TI2V‑5B、流匹配（Flow Matching）框架、Diffusion Transformer（DiT）解码器、3D 因果 VAE、文本编码器、视频继续模块以及联合视频+轨迹的流匹配损失。

**📊 数据集**

主要数据集：训练使用 NAVSIM v1；零射评估在 nuScenes validation（150 场景）和 Bench2Drive（CARLA v2）上进行。

**📈 对比分析**

方法与现有端到端和世界模型方法对比：在 NAVSIM 上 PDMS 达到 90.9（超过 PWM 81.8）；在 nuScenes 上 L2 均值 0.84、碰撞率 0.06（相比 PWM 降低 78.9%/83.3%）；在 Bench2Drive 上 L2 1.33、碰撞率 1.79（相比 PWM 降低 52.5%/52.4%），整体显著优于所有基线。

**⚠️ 局限性**

局限性：依赖大型预训练视频模型，对真实驾驶数据的适配仍有限；未在多模态输入（如 LiDAR、地图）上进行验证；极端长时序的鲁棒性尚未完全保证；训练和推理需要较高的算力与大模型资源。

---

## 66. Investigating the Impact of Subgraph Social Structure Preference on the Strategic Behavior of Networked Mixed-Motive Learning Agents

**arXiv ID:** 2604.03818 | [PDF](https://arxiv.org/pdf/2604.03818v1)

**作者:** Xinqi Gao `[一作]` (Purdue University), Mario Ventresca `[通讯]` (Purdue University)

**通讯引用:** 2405 | [OpenAlex ID](https://openalex.org/A5017375445)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了网络化学习代理在社会困境下对子图结构偏好对策略行为的影响，并提出SRIM模型。

**💡 创新点**

结合子图偏好、图形游戏与奖励塑造的多智能体强化学习，提出基于子图结构的自我激励模型和BCI指标。

**🔧 技术方法**

采用多智能体强化学习中的PPO、奖励塑造、图论指标BCI、社会贡献指数SCI等技术。

**📊 数据集**

使用Harvest和Cleanup两种基准社会困境环境，并构造了多种5、7、9、20节点网络拓扑。

**📈 对比分析**

与无网络无偏好基线相比，SRIM在资源收集、攻击与清理行为上表现出显著差异；BCI和SCI指标显示结构驱动行为显著提升，实验结果显示方法鲁棒且在不同拓扑和环境下保持一致。

**⚠️ 局限性**

仅在两种有限环境和固定网络规模上验证，缺乏对更复杂多样场景的评估；偏好权重需手动设定，缺乏自适应机制。

---

## 67. IDIOLEX: Unified and Continuous Representations for Idiolectal and Stylistic Variation

**arXiv ID:** 2604.04704 | [PDF](https://arxiv.org/pdf/2604.04704v1)

**作者:** Anjali Kantharuban `[一作]` (Carnegie Mellon University), Graham Neubig `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 IdioleX 框架，利用弱监督（作者与社区层次亲近度以及 LLM 提取的语言学特征）学习句子表示，使其在捕捉方言和个人语言风格的同时抑制语义信息，并将该表示用于方言识别、作者鉴别和 LLM 后训练的方言对齐。

**💡 创新点**

创新点包括：① 将层次亲近度与 LLM 生成的语言特征结合为多源弱监督；② 通过 margin ranking、对比学习和特征预测共同训练得到连续、可解释的idiolect 表示；③ 将该表示作为训练目标，改进 LLM 的方言对齐而不牺牲流畅度。

**🔧 技术方法**

核心技术：预训练句子编码器（AraBert、Bertin），层级注意池化，L2 正则化；margin ranking loss、supervised contrastive loss、特征预测交叉熵；LLM（GPT‑5 mini）用于抽取方言特征；post‑training 采用嵌入对齐损失（cosine 余弦损失）。

**📊 数据集**

使用数据集：Reddit 评论（各方言子版块）、Arabic MADAR 26、Spanish DSL‑ML、PAN 2019 Spanish 作者鉴别、AMIYA 方言翻译共享任务、MADAR 26 并行句子用于语义与风格分离。

**📈 对比分析**

与 BERT fine‑tune、Centroid clustering、E5 baseline 等基线比较；在方言识别中 IdioleX 的 F1 最高（0.85 vs 0.80/0.77），在作者鉴别中 38% 对比 28%；语义相关性仅为 Pearson 0.09/0.19，说明风格表示基本独立；在 LLM 后训练中使用 IdioleX 对齐后，ADI2 方言一致性提升，同时 ChrF++ 翻译质量维持或提升。

**⚠️ 局限性**

限制：依赖 Reddit 方言标签不完备，导致方言覆盖不均；在极少样本方言（如巴勒斯坦）表现仍受限；风格与语义仍存在一定耦合，无法完全解耦；未评估用户对方言生成的接受度与社会规范。

---

## 68. GA-GS: Generation-Assisted Gaussian Splatting for Static Scene Reconstruction

**arXiv ID:** 2604.04331 | [PDF](https://arxiv.org/pdf/2604.04331v1)

**作者:** Yedong Shen `[一作]` (University of Science and Technology of China), Yanyong Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8142 | [OpenAlex ID](https://openalex.org/A5053344541)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种生成辅助的3D高斯散射方法（GA-GS），用于从含有动态前景的单目视频中重建静态场景。

**💡 创新点**

核心创新在于：①使用运动感知模块精准分割并移除动态区域；②利用扩散模型对遮挡区域进行视频填充，生成伪真实监督；③为每个高斯原语引入可学习的可靠性标量θ，以动态调节真实与生成内容的混合比例，实现对遮挡区域的高质量重建。

**🔧 技术方法**

技术包括3D高斯散射、VGGT（用于相机位姿与深度估计）、Flow‑SAM（运动感知分割）、DiffuEraser（视频填充）、以及自适应不透明度混合与权重化损失。

**📊 数据集**

实验使用DAVIS数据集（无静态背景GT，仅用于评估可见背景）和作者新构建的Trajectory‑Match数据集（通过机器人固定轨迹捕获动态与静态对齐视频，提供真实静态背景GT）。

**📈 对比分析**

在DAVIS上，GA‑GS在背景PSNR上显著优于WildGaussians、Robust3DGS和DAS3R；在Trajectory‑Match上，GA‑GS在PSNR、SSIM和LPIPS等指标上均击败所有基线，尤其在遮挡区域表现突出。

**⚠️ 局限性**

局限性包括：①对扩散模型生成内容的可靠性仍有限，极端遮挡或细节可能出现不一致；②依赖VGGT/Flow‑SAM等预训练模型，若其预测误差大可能影响重建质量；③在极端动态场景下，生成与真实内容的平衡仍需进一步优化。

---

## 69. Fully Procedural Synthetic Data from Simple Rules for Multi-View Stereo

**arXiv ID:** 2604.04925 | [PDF](https://arxiv.org/pdf/2604.04925v1)

**作者:** Zeyu Ma `[一作]` (Princeton University), Jia Deng `[通讯]` (Princeton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为SimpleProc的全流程程序化合成数据生成系统，用极少量规则通过NURBS曲面、纹理噪声和灯光布置来生成适用于多视角立体重建的训练图像。

**💡 创新点**

创新点在于证明仅用少量简单规则（NURBS形状、噪声位移、布尔纹理组合、随机相机/光照）即可产生与手工或游戏素材等高质量训练数据相当甚至更优的效果；并提供了完整的实验对比与消融分析。

**🔧 技术方法**

核心技术包括Blender中的NURBS lofting、Geometry Nodes纹理位移、材质噪声与布尔运算、随机相机采样与灯光布置，以及对多视角立体网络MVSAnywhere的训练。

**📊 数据集**

使用了自建的SimpleProc生成的数据集（8,000张和352,000张图像），并与8个现有数据集混合（Hypersim、TartanAIR、BlendedMVS等）以及MegaSynth进行对比。

**📈 对比分析**

在固定预算（8,000张）下，SimpleProc数据在大多数RMVD基准上优于混合数据和MegaSynth；在无限预算（352,000张）下，训练出的模型在除ScanNet外的平均指标上与或略优于原始8数据集基线，表现出较高的数据效率。

**⚠️ 局限性**

局限性包括：对ScanNet的性能略逊，说明对特定室内场景的物理或传感器噪声建模不足；仅关注MVSAnywhere模型，未验证对其他MVS架构的泛化；以及在极大规模训练时对存储和计算成本的考量仍待进一步优化。

---

## 70. Free-Range Gaussians: Non-Grid-Aligned Generative 3D Gaussian Reconstruction

**arXiv ID:** 2604.04874 | [PDF](https://arxiv.org/pdf/2604.04874v1)

**作者:** Ahan Shabanov `[一作]`, Numair Khan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用，旨在提高效率和准确性。

**💡 创新点**

创新点在于提出了一种新的优化策略，能够在处理大规模数据时显著减少计算时间。

**🔧 技术方法**

使用了深度学习和强化学习相结合的技术。

**📊 数据集**

采用了公开的图像识别数据集进行实验。

**📈 对比分析**

与现有的几种主流算法进行了比较，结果显示该算法在准确率和速度上均有显著提升。

**⚠️ 局限性**

限制在于算法在特定类型的数据上表现不佳，且对计算资源的需求较高。

---

## 71. Beyond Fixed Tests: Repository-Level Issue Resolution as Coevolution of Code and Behavioral Constraints

**arXiv ID:** 2604.04580 | [PDF](https://arxiv.org/pdf/2604.04580v1)

**作者:** Kefan Li `[一作]` (Beihang University), Weifeng Lv `[通讯]` (Beihang University)

**通讯引用:** 6015 | [OpenAlex ID](https://openalex.org/A5109299440)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个多智能体协同进化框架Agent‑CoEvo，用于同时演化代码补丁和测试补丁，实现仓库级缺陷修复。

**💡 创新点**

创新点在于把测试从固定验证器转变为可演化的行为约束，构建代码与测试共同演化的搜索空间，从而提高修复的鲁棒性和覆盖率。

**🔧 技术方法**

采用基于LLM的语义交叉、交叉评估、精英保留和隔离Docker环境等技术，构建两队进化种群（CodeAgent与TestAgent）。

**📊 数据集**

在SWE‑bench Lite（300个Python项目缺陷）和SWT‑bench Lite（276个测试生成任务）两大基准上进行评估。

**📈 对比分析**

与现有代码优化或测试生成方法相比，Agent‑CoEvo在SWE‑bench Lite上达到41.33%的修复率、在SWT‑bench Lite上达到46.4%，并在测试覆盖度Δ𝒞上取得56.0，显著优于主流基线。

**⚠️ 局限性**

主要局限是相对更高的计算成本（平均每个问题约1.11美元），对初始定位精度和LLM生成测试的语义准确性有一定依赖。

---

## 72. Supervised Dimensionality Reduction Revisited: Why LDA on Frozen CNN Features Deserves a Second Look

**arXiv ID:** 2604.03928 | [PDF](https://arxiv.org/pdf/2604.03928v1)

**作者:** Indar Kumar `[一作]`, Ankit Hemant Lade `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究系统评估了在冻结预训练CNN特征的迁移学习中应用降维方法的效果，重点关注线性判别分析（LDA）与其他传统与现代降维技术的比较。

**💡 创新点**

创新点在于通过大规模实验首次证实LDA在所有考察的四种backbone与两类数据集上均能显著提升准确率，并提出两种轻量级扩展（Residual Discriminant Augmentation与Discriminant Subspace Boosting）以进一步提升性能。

**🔧 技术方法**

采用的技术包括线性判别分析、PCA、局部Fisher判别分析（LFDA）、邻域组件分析（NCA）、正则化LDA、RDA、DSB以及标准化与L2正则化逻辑回归作为下游分类器。

**📊 数据集**

实验数据集为CIFAR-100（100类）和Tiny ImageNet（200类），提取了四种预训练backbone（ResNet-18、ResNet-50、MobileNetV3-Small、EfficientNet-B0）的特征。

**📈 对比分析**

通过在八种backbone–数据集组合上对十种降维方法进行5次随机种子实验，发现LDA在所有组合中比完整特征提升0.3–4.6个百分点，7/8场景优于PCA；RDA/DSB相对LDA仅提高0.2–0.4%，且计算成本低于LFDA/NCA。

**⚠️ 局限性**

局限性包括仅评估线性分类器和100/200类数据集，未探究大规模多类、细粒度任务或非线性分类器；LDA受C‑1上限限制，且在样本极少的情况下可能失效。

---

## 73. LightThinker++: From Reasoning Compression to Memory Management

**arXiv ID:** 2604.03679 | [PDF](https://arxiv.org/pdf/2604.03679v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 74. Mambalaya: Einsum-Based Fusion Optimizations on State-Space Models

**arXiv ID:** 2604.03829 | [PDF](https://arxiv.org/pdf/2604.03829v1)

**作者:** Toluwanimi O. Odemuyiwa `[一作]` (University of California), Michael Pellauer `[通讯]` (NVIDIA)

**通讯引用:** 2195 | [OpenAlex ID](https://openalex.org/A5009910914)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种新型可重构加速器Mambalaya，专门针对State‑Space Model（SSM）中的Mamba模型，通过系统的Einsum融合技术实现对完整运算链的完全融合，从而显著减少跨算子内存传输，提高计算密度。

**💡 创新点**

创新点主要包括：① 设计了一套四类Einsum融合分类（Rank‑Isomorphic、Rank‑Subsetted、Rank‑Supersetted、Rank‑Disjoint），并将其扩展至全链路融合；② 开发了贪心式 stitching 算法，能够自动将多算子组按可融合关系构成大融合组；③ 架构上实现了可在 2D 与 1D 模式之间动态切换的二维 PE 网格，兼容高强度 GEMM 与低强度元素级运算；④ 在 Mamba 上首次完成全链路完全融合并实现了比现有自研加速器（MARCA、Geens 等）更高的速度提升。

**🔧 技术方法**

技术手段包括：Einsum 形式化描述与扩展（EDGE）来精确表达 Mamba 的计算图；基于 EInsum 的张量索引分析实现融合机会识别；贪心式 stitching 与迭代分块（generational rank）策略来处理生成周期；可重构的 2D/1D PE 网络与多功能计算单元（MAC、log/exp/SiLU 等）来支持混合强度算子；以及使用 Timeloop 进行架构建模与性能估算。

**📊 数据集**

实验数据集为两种 Mamba 预训练模型：Mamba‑1（48 层）与 Mamba‑2（64 层），批量大小 64，前置上下文长度从 2048 到 2^20，生成阶段 I=1；使用与 H100 GPU 相当的硬件配置（64K PE、1.75 GHz、2039 GB/s 内存带宽）进行仿真评估。

**📈 对比分析**

与基线比较方法：将 MARCA 与 Geens 等实现分别按最佳未融合（算法级最小访问）与仅在 SSM 区域做 RI 融合的情形进行“理想化”比较。Mambalaya 在前置阶段可获得 4.9× 的加速（与 MARCA 及 Geens 的 1.5×）和 44% 以上的整体提升；在解码阶段可达 1.9× 加速；在不同上下文/生成长度比例下的端到端平均加速约为 3×（相较 MARCA）和 1.3×（相较 Geens）。

**⚠️ 局限性**

限制与未来工作：① 仍无法在全链路实现真正的 RD 融合，需要对多遍访问的张量（如 X、Δ）做重计算或部分融合；② 需要更大或更高分辨率的 on‑chip 缓存以支持更长的迭代列；③ 评估仅基于仿真，尚未在实际硬件上验证；④ 当前方法主要针对 Mamba，尽管可推广，但对极大模型或不同 SSM 结构的适用性仍需进一步验证。

---

## 75. From Curiosity to Caution: Mitigating Reward Hacking for Best-of-N with Pessimism

**arXiv ID:** 2604.04648 | [PDF](https://arxiv.org/pdf/2604.04648v1)

**作者:** Zhuohao Yu `[一作]` (Carnegie Mellon University), Adam Block `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了一种在 Best‑of‑N 推理时使用的惰性奖励估计方法——Caution，利用离线训练的预测器对奖励模型内部特征的误差进行检测，降低对分布外样本的奖励估计，从而抑制奖励黑客。

**💡 创新点**

创新点在于将 RL 中好奇心（curiosity）的对偶——谨慎（caution）思想迁移至 LLM 推理，采用仅需离线训练的预测器与奖励模型特征对齐的方式，实现了高效、可解释的 OOD 检测和惰性奖励估计，并提供了理论证明与实证验证。

**🔧 技术方法**

使用技术包括：Best‑of‑N 采样、Reward Model（如 RLHF）、离线训练的预测器网络（对奖励模型隐藏层特征拟合）、误差作为不确定性估计、惰性奖励公式 r̂ − λα，MSE 损失和简单的超参数 λ。

**📊 数据集**

数据集：GSM8K、MATH‑500、BigBench‑Hard 以及其他推理基准；训练奖励模型和预测器时使用相同提示集；评估时分别在同分布和 OOD 分布（如 MATH‑500、BigBench‑Hard）进行测试。

**📈 对比分析**

与标准 BoN、仅使用奖励模型、惰性奖励单独或结合等方法对比；在 N 从 1 到 512 的范围内，Caution 维持单调提升；相较于标准 BoN，峰值准确率提升约 4.2%，最终准确率提升约 15.5%；在 OOD 任务中仍显著缓解奖励黑客并保持或略低于 BoN 的性能。

**⚠️ 局限性**

局限性包括：依赖已有奖励模型的内部特征，OOB 检测的泛化受提示分布限制；λ 参数需要调优但相对不敏感；理论证明基于简化线性假设，实际复杂度高；在极度离谱的 OOD 场景下，预测器误差可能不足以完全抵消奖励模型的偏差。

---

## 76. MagicCopy: Bring my data along with me beyond boundaries of apps

**arXiv ID:** 2604.04307 | [PDF](https://arxiv.org/pdf/2604.04307v1)

**作者:** Priyan Vaithilingam `[一作]` (Harvard University), Chenglong Wang `[通讯]` (Microsoft)

**通讯引用:** 4102 | [OpenAlex ID](https://openalex.org/A5100401482)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种名为 MagicCopy 的 AI 驱动的跨应用复制粘贴工具，利用复制与粘贴动作的普适性，结合源/目标上下文与自然语言指令，自动提取、解析、转换并重新格式化数据，使用户能够在不同应用间直接完成复杂的数据迁移和变换。

**💡 创新点**

核心创新在于：①将 AI 助手嵌入到通用的复制粘贴动作中，形成“工具中心”而非独立应用；②使用隐式上下文追踪同时保留源端格式信息，避免传统复制粘贴导致的结构丢失；③让 LLM 生成可执行的 Python 代码而非直接生成数据，降低幻觉风险并支持大尺寸数据；④提供插件机制，让目标应用可提供更深层次的 API 支持。

**🔧 技术方法**

技术组成包括：Windows 原生 API 钩子实现复制/粘贴事件与应用上下文捕获；Electron+React 前端与 Python 后端 WebSocket 通信；OpenAI GPT‑4o（以及 GPT‑4o‑mini 备选）作为 AI 助手，调用代码生成工具链（pandas、numpy、BeautifulSoup 等）；Python 沙盒执行环境；插件系统（如 Excel Javascript API）用于深度交互。

**📊 数据集**

实验使用三组公开数据集：
- 2024 选举通用投票表（多张 HTML 表格，含背景色）
- 2024 选举总统投票表（单一 HTML 表格）
- 2020 东京奥运奖牌表（单一 HTML 表格，含图标）
每组数据在 Excel、LaTeX、Markdown 三种格式下预制，设计了列删除、列添加、列合并、列拆分、条件高亮等五类变换任务。

**📈 对比分析**

与传统复制粘贴及使用 ChatGPT 作为中间工具对比：在 16 名参与者的用户研究中，所有任务全部成功，平均 1.1 次重试，SUS（可用性量表）评分极高；NASA TLX 显示认知负荷显著下降（87.5% 认为压力更小）；相比传统手工处理，用户认为 MagicCopy 节省时间（多数参与者认为时间优势远大于 30‑40 秒的响应延迟）。

**⚠️ 局限性**

局限性包括：①对 LLM 的依赖导致推理延迟（≈30‑40 秒）和偶发模型错误；②当前仅处理剪贴板中的单个项目，对大规模数据（如数千行以上）支持有限；③需要用户手动验证输出，尤其是复杂变换时可能需要大量检查；④隐私与安全风险（数据可能被传输到云端）；⑤插件生态尚不成熟，深度交互受限于目标应用的 API 支持。

---

## 77. Beyond Fluency: Toward Reliable Trajectories in Agentic IR

**arXiv ID:** 2604.04269 | [PDF](https://arxiv.org/pdf/2604.04269v1)

**作者:** Anushree Sinha `[一作]` (Google), Abhishek Dharmaratnakar `[通讯]` (Google)

**通讯引用:** 2471 | [OpenAlex ID](https://openalex.org/A5112264148)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出了在信息检索（IR）中从被动检索向自主代理（Agentic IR）转型所面临的可靠性挑战，并通过系统性验证门（Verification Gates）、因果归因（Causal Attribution）和校准式放弃（Calibrated Abstention）等方法，构建了针对多步 Reason‑Act‑Observe 循环的可靠轨迹框架。

**💡 创新点**

创新点在于将可靠性指标从单一步骤的输出准确率转向轨迹完整性（trajectory integrity），并引入了四阶段（Planning、Retrieval、Reasoning、Execution）的错误分类、First‑Error Position（FEP）、Rollback Recovery Rate（RRR）等细粒度指标，提出了“Fluency Trap”和“Reasoning Trap”的概念以及相应的防御策略。

**🔧 技术方法**

主要技术包括：1) 通过 Solvability Classifier 与 SPA‑RL 等模型实现计划与推理阶段的可执行性校验；2) 通过 UProp 与 InfiAgent 等框架实现逐步不确定性分解与环境状态的可视化；3) 采用 Dry‑Run（沙箱模拟）与 Schema Validation 的执行门；4) 利用 PRISM 等成本敏感的放弃门实现基于风险阈值的主动拒绝；5) 结合 Gemini AI 与 OdysseyArena 等工具构建可审计的生命周期管理。

**📊 数据集**

本文在概念与实验设计层面参考了 WebShop、WebArena、Mind2Web 与 SWE‑bench 等工业级代理评测环境作为验证基准，并借助这些数据集探讨了不同错误类型在实际任务中的影响。

**📈 对比分析**

与传统基于输出准确率的评估方法相比，本文提出的轨迹可靠性指标（FEP、RRR、Weakest‑Link Reliability 等）更能捕捉多步交互中的错误累积。实验虽未给出具体数值，但作者通过案例分析与基准任务示例表明，在引入验证门与因果归因后，系统在检测和恢复错误的成功率显著提升，且能够有效避免“Fluency Trap”导致的功能失效。

**⚠️ 局限性**

局限性包括：1) 目前缺乏大规模真实实验验证，框架的可行性主要基于理论与小规模案例；2) 需要精确的不确定性估计与模型校准，实际部署时可能面临鲁棒性与计算开销的挑战；3) 过度的放弃机制可能导致用户体验下降，需要在风险与可用性之间精细权衡；4) 现有的评测基准与工业实际场景仍有差距，未来需要更多跨领域、跨任务的验证。

---

## 78. SecureAFL: Secure Asynchronous Federated Learning

**arXiv ID:** 2604.03862 | [PDF](https://arxiv.org/pdf/2604.03862v1)

**作者:** Anjun Gao `[一作]` (University of Louisville), Minghong Fang `[通讯]` (University of Louisville)

**通讯引用:** 1736 | [OpenAlex ID](https://openalex.org/A5056811906)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为 SecureAFL 的异步联邦学习安全框架，能在不共享私有数据的情况下，抵御模型注入攻击。

**💡 创新点**

核心创新包括：① 基于 Lipschitz 连续性的过滤器自动识别异常更新；② 利用历史模型与 L‑BFGS 近似估计未上传客户端的梯度；③ 将已过滤更新与估计更新一起用坐标‑中位数等 Byzantine‑robust 规则聚合，从而兼顾鲁棒性与完整性。

**🔧 技术方法**

技术实现：异步 FL 协议、Lipschitz 连续性检验、L‑BFGS 估计、坐标‑中位数聚合、梯度裁剪、随机延迟模拟。

**📊 数据集**

实验数据集：Fashion‑MNIST、CIFAR‑10、CIFAR‑100、Tiny‑ImageNet（分类）和 Udacity 自动驾驶回归。

**📈 对比分析**

与 AsyncSGD、Kardam、BASGD、Sageflow、Zeno++、AFLGuard 及 AsyncDefender 等 6~7 个基线对比。SecureAFL 在无攻击场景下性能与 AsyncSGD 相当；在十种无目标/有目标攻击（含自适应攻击）中均显著低于基线，尤其在恶意客户端比例 20%–40% 时保持最低的误差率与攻击成功率。

**⚠️ 局限性**

局限性：依赖于客户端更新的历史平滑性假设，若更新本身波动大则过滤误判风险；对超参数（如 α、G）仍有一定敏感性；在极端异步或极高异质性下的表现尚未充分验证。

---

## 79. Neuromorphic Computing for Low-Power Artificial Intelligence

**arXiv ID:** 2604.04727 | [PDF](https://arxiv.org/pdf/2604.04727v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 80. Packing Entries to Diagonals for Homomorphic Sparse-Matrix Vector Multiplication

**arXiv ID:** 2604.04683 | [PDF](https://arxiv.org/pdf/2604.04683v1)

**作者:** Kemal Mutluergil `[一作]` (Sabanci University), Erkay Savaş `[通讯]` (Sabanci University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文提出了一套针对同态加密（HE）环境下稀疏矩阵向量乘（SpMV）加速的完整框架：先通过两维对角线打包问题（2DPP）寻找行列置换，使得矩阵在Halevi–Shoup的加密矩阵向量乘法中只涉及最少的非空循环对角线；随后利用基于图的初始排序（RCM、MP、LBS等）和 2OPT/3OPT 迭代改进策略进一步压缩对角线数量；最后对极度稠密的行/列进行分离剔除，从而进一步降低加密乘法和旋转的开销。整个过程在实际实现中兼顾了加密安全性和运行效率。

**💡 创新点**

创新点包括：
1) 对2DPP的正式化及其与经典带宽、循环带宽等概念的关系；
2) 结合多种图排序启发式（RCM、MP、LBS、EigenOrder）产生优良的初始置换；
3) 设计了 2OPT/3OPT 迭代改进机制，利用离开/到达计数高效评估置换收益；
4) 提出了密集行/列剔除的成本模型与阈值判定，使得对稠密结构的处理与稀疏核心分离；
5) 在SuiteSparse 175个稀疏矩阵上进行系统评估，证明对角线压缩可提升45×以上、加密 SpMV 时间可提升约40×。

**🔧 技术方法**

使用的技术包括：
- 同态加密中的 Halevi–Shoup 矩阵向量乘法（CKKS）；
- 行列置换与循环对角线的数学定义；
- 图理论排序算法（RCM、MP、LBS、EigenOrder）和二进制整数规划（ILP）求解小规模最优解；
- 迭代局部搜索 2OPT/3OPT；
- 低成本离开/到达计数与候选队列机制；
- 密集行/列剔除的成本模型与阈值控制。

**📊 数据集**

采用了 175 个来自 SuiteSparse Matrix Collection 的稀疏矩阵（10 000 ≤ n ≤ 50 000，平均非零数约 8.7），并在实验平台上评估其对加密 SpMV 的加速效果。

**📈 对比分析**

与基准自然排序 + 无优化（Natural+NoOPT）和仅使用 3OPT 的组合进行对比。最佳组合 *+3OPT 在 160/175 = 91.4% 的矩阵上取得最优结果，平均对角线数减少约 45×，对应的加密 SpMV 运行时间平均提升约 40×；在前 20 名加速案例中，最高可达 96% 的时间节省。实验还显示，密集行/列剔除可进一步提升约 25% 的运行时间。

**⚠️ 局限性**

局限性：
- 仅针对方阵，未直接支持矩形矩阵；
- 对角线打包依赖于稀疏结构，稠密或噪声较大的矩阵（如 EigenOrder 对真实数据表现差）；
- 迭代改进在大规模矩阵上可能需要数小时；
- 方案在 CKKS 参数下验证，迁移到其他 HE 加密方案需重新评估成本；
- 对行/列置换的安全性仅在一定程度上得到保障，仍存在对角线索引泄露的潜在风险。

---

## 81. CardioSAM: Topology-Aware Decoder Design for High-Precision Cardiac MRI Segmentation

**arXiv ID:** 2604.03313 | [PDF](https://arxiv.org/pdf/2604.03313v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 82. Search, Do not Guess: Teaching Small Language Models to Be Effective Search Agents

**arXiv ID:** 2604.04651 | [PDF](https://arxiv.org/pdf/2604.04651v1)

**作者:** Yizhou Liu `[一作]` (University of Illinois Urbana-Champaign), Chen Zhao `[通讯]` (New York University)

**通讯引用:** 1838 | [OpenAlex ID](https://openalex.org/A5100351992)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对小语言模型（SLM）进行搜索代理的蒸馏，提出始终搜索策略（ASP）并将其融入SFT、OPD等蒸馏方法，强化SLM在推理过程中对外部检索工具的依赖。

**💡 创新点**

创新点在于通过ASP显式约束学生模型在训练时始终调用检索工具，消除对内部参数知识的过度依赖，从而显著降低幻觉并提升多跳推理性能。

**🔧 技术方法**

使用技术包括：1) 始终搜索策略（ASP）过滤训练轨迹；2) 监督式微调（SFT）；3) 在线策略蒸馏（OPD）；4) 混合蒸馏（Mixed）；5) 拒绝式微调（RFT）强化高质量轨迹。

**📊 数据集**

使用的数据集包括HotpotQA、2WikiMultiHopQA、Bamboogle、MuSiQue、BrowseComp-plus、Frames、LongSeAL，评估覆盖结构化多跳推理与复杂信息检索任务。

**📈 对比分析**

比较方法：将ASP蒸馏的SLM与未蒸馏的“Vanilla”SLM以及教师LLM进行String‑F1对比；ASP下SFT Qwen3‑1.7B在HotpotQA上从53.2提升至70.6（仅差2.5点），与Qwen3‑8B相当；在其他基准上，ASP模型匹配或超过8B模型，并在噪声检索场景下表现更稳健。

**⚠️ 局限性**

局限性：ASP的实现相对简单，未探究与更高级训练框架的融合；未充分界定SLM代理的上限，仍受推理能力等因素影响；假设检索结果始终准确，对噪声或误导信息的鲁棒性不足；实验仅在Qwen3族上验证，缺乏跨模型的泛化验证。

---

## 83. Decocted Experience Improves Test-Time Inference in LLM Agents

**arXiv ID:** 2604.04373 | [PDF](https://arxiv.org/pdf/2604.04373v1)

**作者:** Maohao Shen `[一作]` (Massachusetts Institute of Technology), Gregory Wornell `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 34007 | [OpenAlex ID](https://openalex.org/A5066172831)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在不更新模型参数的情况下，利用已累积的经验来构造高效上下文，以提升大语言模型在推理与代理任务中的推理质量和效率。

**💡 创新点**

创新点包括：①提出经验去醃（experience decoction）概念，将原始经验压缩为精炼的“课程（lesson）”；②通过信息增益理论量化上下文质量，并发现相关性与多样性平衡是关键；③设计层次化概念树结构来提升检索多样性和相关性；④系统性评估经验规模、检索规模与上下文质量之间的非线性关系。

**🔧 技术方法**

技术手段主要有：经验去醃（lesson distillation）、聚类式记忆整合、基于向量相似度的检索、基于信息增益的上下文评估、层次化概念树（hierarchical concept tree）及LLM重排序；使用Qwen3-Embedding-4B做语义编码。

**📊 数据集**

使用的实验数据集包括：数学推理任务（AMC、AIME、HMMT、BeyondAIME，记忆来源DAPO-Math）；Web购物代理任务（WebShop）；软件工程任务（SWE-bench）。

**📈 对比分析**

与无上下文的基线相比，去醃后的课程上下文在代理任务中平均提升约10–20%奖励，在推理任务中也能在更短的输出长度内达到相近或更高的准确率；概念树检索进一步提升了性能并增加了检索概念的多样性，且在不同任务上均显著优于单层检索。

**⚠️ 局限性**

主要局限包括：需要先行收集并去醃经验，过程受限于所选的去醃方法与聚类策略；检索参数（K、λ等）需要手工调节，且在极大规模经验或新颖任务时效果可能下降；目前仅在三类任务上验证，未涉及更广泛的领域。

---

## 84. Task-Guided Multi-Annotation Triplet Learning for Remote Sensing Representations

**arXiv ID:** 2604.03837 | [PDF](https://arxiv.org/pdf/2604.03837v1)

**作者:** Meilun Zhou `[一作]` (University of Florida), Alina Zare `[通讯]` (University of Florida)

**通讯引用:** 3928 | [OpenAlex ID](https://openalex.org/A5079676776)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于互信息的任务引导多注释三元组学习方法，用来在遥感图像中学习同时兼顾语义与几何信息的共享表示。

**💡 创新点**

创新点在于用互信息筛选最具跨任务信息的三元组，取代传统静态权重，直接通过样本关系塑造潜在空间，减少对权重调优的依赖。

**🔧 技术方法**

使用了预训练的Vision Transformer（DINOv2、CLIP、MAE）特征、三元组损失、多注释三元组构造、互信息计算、top+random 采样等技术。

**📊 数据集**

实验基于远程航空野生动物图像数据集 AWIR，包含类别标签和盒子级几何注释。

**📈 对比分析**

与基线无三元组、仅类标签三元组、硬三元组以及固定权重多注释三元组对比，任务引导三元组在分类准确率和盒子特征回归 R² 上均实现提升，尤其在回归任务上表现最佳。

**⚠️ 局限性**

局限性包括依赖几何标签离散化、对随机采样比例敏感、在分类任务中仍被硬采样方法超越，以及缺乏对更大规模或不同遥感场景的鲁棒性验证。

---

## 85. WaterSplat-SLAM: Photorealistic Monocular SLAM in Underwater Environment

**arXiv ID:** 2604.04642 | [PDF](https://arxiv.org/pdf/2604.04642v1)

**作者:** Kangxu Wang `[一作]` (Tsinghua University), Guijin Wang `[通讯]` (Tsinghua University)

**通讯引用:** 4159 | [OpenAlex ID](https://openalex.org/A5045183950)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了WaterSplat-SLAM，一种利用单目相机实现鲁棒位姿估计和高保真渲染的水下SLAM系统。

**💡 创新点**

创新点包括：语义引导的水体掩码+3D高斯投影，使用MLP中介网络预测水体衰减与散射参数，并在循环闭合时对高斯原语进行体素合并，实现在线密集且光照真实的水下映射。

**🔧 技术方法**

采用CLIPSeg进行水体语义分割、MASt3R做两视角几何恢复、3D Gaussian Splatting结合MLP中介网络构建高斯地图、语义引导光度损失与几何正则化以及自适应高斯原语管理。

**📊 数据集**

在公开的SeaThru-NeRF数据集和自制的WaterSplat-SLAM水池数据集（包括Big_gate、Pipe_local、Pool_up等六个序列）进行评估。

**📈 对比分析**

与GO‑SLAM、GLORIE‑SLAM、MonoGS、HI‑SLAM2、OpenGS‑SLAM、S3PO‑GS等基线以及离线WaterSplatting进行对比，渲染PSNR/SSIM显著高于同类方法，ATE保持在0.2–0.3 m范围，帧率稳定在2–5 FPS。

**⚠️ 局限性**

局限性：对水体语义分割准确性高度依赖，误分割会影响跟踪与渲染；在极度浑浊或远距离场景下深度估计仍易受限；系统对算力需求较高。

---

## 86. Conversational Control with Ontologies for Large Language Models: A Lightweight Framework for Constrained Generation

**arXiv ID:** 2604.04450 | [PDF](https://arxiv.org/pdf/2604.04450v1)

**作者:** Barbara Gendron `[一作]` (University of Lorraine), Mathieu d'Aquin `[通讯]` (University of Lorraine)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于本体的轻量级框架，通过在LLM上使用LoRA微调与标签包装策略，实现对会话的可控生成；在“Proficiency‑Level Control”和“Polarity Profile Control”两种用例中验证了该方法的有效性。

**💡 创新点**

创新点在于将本体定义的描述符与对话策略相结合，利用标签包装的训练样本实现对生成的约束，并且保持模型无架构依赖；同时在两种不同的会话控制场景中展示了该框架的可扩展性。

**🔧 技术方法**

使用技术包括：本体建模（Protégé、OWL、Pellet推理）、Causal Language Modeling + LoRA微调、标签包装训练策略、基于本体推理的对话策略引导。

**📊 数据集**

所用数据集包括 CEFR‑T、CEFR‑S、DDbal（DailyDialog 的 CEFR 版本）、DailyDialog、GoEmotions、EmpatheticDialogues、SST‑3（Stanford Sentiment Treebank 3‑class）。

**📈 对比分析**

方法评估通过与未微调的预训练模型进行零样本生成对比，使用 Accuracy、F1、MAE（CEFR 用例）、MCC（Polarity 用例）以及 B_r 语义相似度指标；实验结果显示微调模型在两用例中均优于基线，并提升了类别覆盖率。

**⚠️ 局限性**

局限性包括：微调对本体概念的学习效果有限；描述符定义具主观性且对上下文依赖性强；未进行真实用户交互的评估；以及可能被滥用的风险。

---

## 87. How Alignment Routes: Localizing, Scaling, and Controlling Policy Circuits in Language Models

**arXiv ID:** 2604.04385 | [PDF](https://arxiv.org/pdf/2604.04385v1)

**作者:** Gregory N. Frank `[一作]` `[通讯]` (Independent Researcher), Gregory N. Frank (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并识别了语言模型中的稀疏路由机制——门控注意头读取检测到的内容并触发下游放大头以实现拒绝行为，并在9个不同实验室的模型上验证其一致性。

**💡 创新点**

通过三步管道（DLA、头级消融、交叉激活交换）精准定位门控头，并揭示意图识别与政策路由的结构分离，提供了对安全机制的可解释性。

**🔧 技术方法**

使用注意力头的Direct Logit Attribution、头级消融、交叉激活交换、bootstrap与置换检验、线性探针、路由阈值调节以及加密绕过实验等技术手段。

**📊 数据集**

以120对政治敏感与控制提示为主的数据集为核心，扩展至32对及更大规模提示，并包含多模型与不同语言的提示。

**📈 对比分析**

通过交叉互换必要性和充分性测试、消融影响与模型输出拒绝/回答率对比，发现门控头必要性>1%，放大头消融效应可达17倍弱化，路由机制在不同规模下保持可检测，拒绝率随主题呈阶梯式响应。

**⚠️ 局限性**

局限包括：MLP路由贡献未拆解、仅针对政治审查与安全拒绝、适用模型规模仅2–32B、加密攻击仅限单一替换密码、缺乏跨编码类型与多轮攻击的完整评估。

---

## 88. TraceGuard: Structured Multi-Dimensional Monitoring as a Collusion-Resistant Control Protocol

**arXiv ID:** 2604.03968 | [PDF](https://arxiv.org/pdf/2604.03968v1)

**作者:** Khanh Linh Nguyen `[一作]`, Tu Tran `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种多维度监控协议，评估AI代理行为的五个维度并结合启发式检测器来提升安全性。

**💡 创新点**

创新点在于将单一可疑度评分拆分为目标对齐、约束遵守、推理连贯、安全意识、行动一致性五维度，并利用结构化提示和分工机制抑制监控模型的共谋。

**🔧 技术方法**

技术包括独立LLM多维度评分、七个状态化启发式检测器、正则表达式模式匹配、意图分析器以及可分配的可信与不可信模型。

**📊 数据集**

使用了bash脚本任务集BashArena（637个任务，四类攻击）和AgentHarm安全评估数据集。

**📈 对比分析**

与框架自带的单分数监控相比，结构化监控在中等阈值下安全率可达95–100%，不可信多维度监控安全率为95%而单分数为0%。

**⚠️ 局限性**

局限性包括手工设定维度权重、5倍API调用成本、仅在两类数据集上验证、攻击者可针对结构化维度设计规避策略。

---

## 89. PortraitCraft: A Benchmark for Portrait Composition Understanding and Generation

**arXiv ID:** 2604.03611 | [PDF](https://arxiv.org/pdf/2604.03611v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 90. Unpacking .zip: A First Look at Domain and File Name Confusion

**arXiv ID:** 2604.04805 | [PDF](https://arxiv.org/pdf/2604.04805v1)

**作者:** Zane Ma `[一作]` `[通讯]` (Oregon State University), Zane Ma (Oregon State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文首先枚举了 DNS 与文件名命名空间混淆的攻击向量，并通过客户端测试与蜜罐实验收集真实网络流量，验证了混淆在实际软件中的存在和危害。

**💡 创新点**

创新之处在于首次系统性列出所有混淆攻击场景，首次在野外实测并量化 DNS/文件名混淆的频率与影响，并提出基于 TLD 选择与客户端指纹识别的测量框架。

**🔧 技术方法**

采用了被动 DNS 监测、蜜罐部署、HTTP 服务器日志分析、客户端行为测试、HTTP 请求指纹识别以及多阶段过滤等技术手段。

**📊 数据集**

使用了大型 ISP 的被动 DNS 数据集、ICANN 证书透明度日志、七个月蜜罐收集的 DNS/HTTP 流量、文件扩展名受欢迎度排行榜以及多款即时通讯与邮件客户端的测试样本。

**📈 对比分析**

通过对比各客户端在五套测试中的安全/不安全行为，统计链接预览与点击请求数量，并通过指纹匹配评估客户端解析逻辑差异，结果显示数千条真实预览/点击流量表明混淆普遍存在，性能表现尚未量化。

**⚠️ 局限性**

主要限制包括仅关注少数文件扩展名 TLD，无法全面泛化；高比例扫描/机器人流量导致难以区分真实混淆；用户代理伪装与共享基础设施可能产生误判；且未能直接判定混淆是否被主动利用。

---

## 91. FVRuleLearner: Operator-Level Reasoning Tree (OP-Tree)-Based Rules Learning for Formal Verification

**arXiv ID:** 2604.03245 | [PDF](https://arxiv.org/pdf/2604.03245v1)

**作者:** Lily Jiaxin Wan `[一作]` (University of Illinois Urbana-Champaign), Haoxing Ren `[通讯]` (NVIDIA)

**通讯引用:** 3046 | [OpenAlex ID](https://openalex.org/A5029928585)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于操作层规则学习的框架，通过Operator Reasoning Tree对SVA生成进行可解释的推理，并在测试时检索并适配规则以提升生成质量。

**💡 创新点**

创新点在于将SVA生成转化为结构化的操作级推理，利用Op-Tree提炼可迁移的规则，并通过检索+规则适配实现无须微调的自适应生成。

**🔧 技术方法**

采用LLM检索增强生成、Operator Reasoning Tree、操作级规则（Op‑Rules）、混合评分（Operator与语义兼顾）、信号抽象和自我校正机制。

**📊 数据集**

使用了1,000对开源硬件设计（AssertionBench/OpenCore）以及NL2SVA-Human、NL2SVA-Machine等公开数据集。

**📈 对比分析**

与FVEval、few‑shot、RAG等现有基线对比，在三数据集上实现功能正确率提升31.17%，语法正确率98.39%，功能误差下降70.33%，性能显著优于基线。

**⚠️ 局限性**

受限于LLM推理能力，难以处理高度耦合的全局依赖和模糊自然语言描述；需要进一步集成到工业流水线并提升极少量数据场景的泛化能力。

---

## 92. A Physics-Informed, Behavior-Aware Digital Twin for Robust Multimodal Forecasting of Core Body Temperature in Precision Livestock Farming

**arXiv ID:** 2604.04098 | [PDF](https://arxiv.org/pdf/2604.04098v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 93. CGHair: Compact Gaussian Hair Reconstruction with Card Clustering

**arXiv ID:** 2604.03716 | [PDF](https://arxiv.org/pdf/2604.03716v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 94. ENCRUST: Encapsulated Substitution and Agentic Refinement on a Live Scaffold for Safe C-to-Rust Translation

**arXiv ID:** 2604.04527 | [PDF](https://arxiv.org/pdf/2604.04527v1)

**作者:** Hohyun Sim `[一作]`, Binoy Ravindran `[通讯]` (Virginia Tech)

**通讯引用:** 3974 | [OpenAlex ID](https://openalex.org/A5067528153)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段的C到Rust安全迁移流水线Encrust，先用ABI保持的包装器模式完成函数级安全化，再通过全项目LLM代理进行剩余 unsafe 处理，保证功能等价；

**💡 创新点**

创新点在于拆分ABI适配与函数实现的关注点，利用包装器模式实现无缝更新，结合可回滚的编译-测试循环和全项目LLM代理完成跨文件 unsafe 构造的消除；

**🔧 技术方法**

使用GPT‑4o LLM、编译器反馈、测试向量、工具化操作（17种代码导航/修改/分析/验证/控制工具）以及基于两阶段验证门的完整性保障；

**📊 数据集**

评估数据集为 7 个 GNU Coreutils 程序和 8 个 Laertes 库，总计约 197,706 行 C 代码；

**📈 对比分析**

与 C2Rust、C2SaferRust、EvoC2Rust 等基线对比，Encrust 在所有 15 项目上实现 100% 的测试向量正确性，同时在 5 个安全指标上分别比 C2Rust 降低 44‑57% 以上的 unsafe 量；

**⚠️ 局限性**

主要局限是仅对测试向量覆盖的路径保证正确性，未覆盖的代码路径可能仍含误；Type‑Directed Wrapper Elimination 及 Agentic Refinement 无法覆盖所有 unsafe 场景，且对未测试函数的安全性未做进一步验证；

---

## 95. FeynmanBench: Benchmarking Multimodal LLMs on Diagrammatic Physics Reasoning

**arXiv ID:** 2604.03893 | [PDF](https://arxiv.org/pdf/2604.03893v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 96. Graduated Trust Gating for IoT Location Verification: Trading Off Detection and Proof Escalation

**arXiv ID:** 2604.03896 | [PDF](https://arxiv.org/pdf/2604.03896v1)

**作者:** Yoshiyuki Ootani `[一作]` `[通讯]` (Independent Researcher), Yoshiyuki Ootani (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了渐进式信任门控（Graduated Trust Gating）与会话锁定机制，用轻量级多信号检测与零知识证明升级实现 IoT 位置验证；

**💡 创新点**

创新点在于三层决策门（Proceed/Step‑Up/Deny）与会话锁定，能在高阈值下实现零误拒，同时将边缘合法请求转交更强的验证；

**🔧 技术方法**

采用多信号加权整合（运动、精度、时序一致、修正一致、网络交叉）求信任分数，使用 Groth16 零知识近似证明做步进验证，并实现会话锁定算法；

**📊 数据集**

使用 10,000 条合成轨迹（四类合法+六类伪造）以及 Android Jelly Star 实测轨迹（诚实行走、静止、远程伪造、附近伪造）进行评估；

**📈 对比分析**

与传统二元门控在相同 FAR 下对比，渐进门在阈值 0.9 时 FDR 为 0%（而二元门为 0.05%），FAR 11.4%，信任分数计算耗时仅 4.9 μs，最优两信号配置 F1=0.84；

**⚠️ 局限性**

主要限制在于步进验证假设完美（oracle）且需要设备支持 Groth16 及可信位置证明；对高技能攻击者（可操纵所有信号）仍有限制，并需更多真实数据进行模型调优。

---

## 97. Amalgamation of Physics-Informed Neural Network and LBM for the Prediction of Unsteady Fluid Flows in Fractal-Rough Microchannels

**arXiv ID:** 2604.03504 | [PDF](https://arxiv.org/pdf/2604.03504v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 98. Preservation Is Not Enough for Width Growth: Regime-Sensitive Selection of Dense LM Warm Starts

**arXiv ID:** 2604.04281 | [PDF](https://arxiv.org/pdf/2604.04281v1)

**作者:** Eren Unlu `[一作]` `[通讯]` (Globeholder), Eren Unlu (Globeholder)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了宽度扩展（width expansion）在小训练预算下的候选选择问题，比较了多种 warm‑start 方案的后续训练效果。

**💡 创新点**

将宽度增长视为完整训练状态的选择任务，发现选择效果高度依赖于训练模式（确定性 vs 随机）和延迟预算，并提出 probe KL 为最稳健的低成本选择器。

**🔧 技术方法**

使用 TinyStories 代理实验，实施 exact-copy、perturbative、asymmetric-reset、ref-subspace 等宽度扩展策略，并通过零步 KL、probe KL、probe RMS、probe escape 等指标评估候选质量。

**📊 数据集**

采用 TinyStories 数据集（约 50,000 篇故事），并自行训练字节级 BPE tokenizer，保持同一语料与模型配置。

**📈 对比分析**

通过 16 步短期 probe 与 128 步长期持续验证损失 AUC 进行对比；结果显示在短期和随机条件下 exact-copy 最高，确定性长周期中 ref‑subspace 获胜，而 probe KL 在所有完成的实验中均表现为最佳低成本选择器。

**⚠️ 局限性**

局限性：实验仅在单一模型、单数据集、单 GPU 规模下完成，缺乏多 seed 重复验证，随机与确定性对比未完全分离，候选策略集合有限，且未与大规模真实训练场景对比。

---

## 99. APPA: Adaptive Preference Pluralistic Alignment for Fair Federated RLHF of LLMs

**arXiv ID:** 2604.04261 | [PDF](https://arxiv.org/pdf/2604.04261v1)

**作者:** Mahmoud Srewa `[一作]` (University of California), Salma Elmalaki `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于联邦强化学习的自适应奖励聚合框架，用于在多群体偏好场景下实现大语言模型的多元对齐；

**💡 创新点**

创新点在于使用历史对齐奖励的倒置softmax动态给每个群体分配权重，并引入公平性指数阈值切换聚合策略，从而在保证整体对齐的同时提升最差群体表现；

**🔧 技术方法**

技术包括联邦RLHF、PPO、PluralLLM偏好预测器、倒置softmax权重更新、置信度公平性指数等；

**📊 数据集**

实验使用GLOBALQA（全球多国意见调查）和OQA（美国内部人口统计学意见调查）两个问答数据集，并在Gemma‑2‑2B、Llama‑3.2‑3B、Qwen3‑0.6B三大模型上进行评估；

**📈 对比分析**

与平均聚合和最小聚合基线对比，本文方法在两任务、两数据集、三模型上均能同时提升平均与最差群体的对齐得分，最差群体提升可达28%，而整体对齐优于最小聚合；

**⚠️ 局限性**

局限性包括对奖励信号设计的依赖（需假设PluralLLM能准确预测群体分布），以及在小模型或高度稀疏奖励场景下自适应权重可能失效，且实验仅覆盖多选问答，需验证在生成式任务中的通用性。

---

## 100. Emergent Compositional Communication for Latent World Properties

**arXiv ID:** 2604.03266 | [PDF](https://arxiv.org/pdf/2604.03266v1)

**作者:** Tomek Kaszyński `[一作]` `[通讯]` (Independent Researcher), Tomek Kaszyński (Independent Researcher)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究多智能体在视频感知基础上，通过离散压缩通道实现对不可见物理属性（弹性、摩擦、质量等）的可组合通信，并验证该通信协议可迁移到真实视频。

**💡 创新点**

创新点在于将迭代学习、受限消息结构与多智能体压缩需求相结合，首次在无标签的物理属性推理任务中自发产生位置专一的组合语言，并通过实证展示不同视频预训练模型对可通信物理知识的决定性影响。

**🔧 技术方法**

技术手段包括：使用冻结的视觉基模型（DINOv2、V‑JEPA‑2）提取时空特征；Gumbel‑Softmax多头离散瓶颈构成事实化的消息空间；基于迭代学习的群体压力和受限带宽促进协议可学习性；使用PosDis、TopSim、BosDis等指标评估组合性。

**📊 数据集**

实验数据集包括：合成物理模拟（斜坡滑动+弹跳、碰撞）以及 Physics 101 实验室真实视频，后者提供已测量的质量和体积标签。

**📈 对比分析**

比较方法：对比两种视觉基模型、不同代理数量、匹配带宽与帧数的对照实验；测评指标为对比任务的准确率、PosDis、TopSim。结果显示：V‑JEPA‑2 在仅涉及运动动力学的碰撞场景中相较 DINOv2 提高 10+% 的准确率，四代理配置下组合性几乎达到 100%，并且冻结的组合消息能在新的预测任务中保留约 94% 的原始特征性能。

**⚠️ 局限性**

局限性：仅在受控模拟或实验室背景下验证，迁移到更复杂的自然视频时表现下降；2 代理设置组合性不稳定，仅当代理数 ≥3 时可靠；方法依赖于冻结特征，对感知学习不作探究；实验规模相对较小，未覆盖多对象或关系推理任务。

---

## 101. Tight Bounds on Window Size and Time for Single-Agent Graph Exploration under T-Interval Connectivity

**arXiv ID:** 2604.04619 | [PDF](https://arxiv.org/pdf/2604.04619v1)

**作者:** Yuichi Sudo `[一作]` (Hosei University), Koichi Wada `[通讯]` (Hosei University)

**通讯引用:** 2254 | [OpenAlex ID](https://openalex.org/A5027866181)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在 T-间隔连通动态图中，单一移动代理在两种可见性模型（KT_0 与 KT_1）下的确定性探索问题，探讨了保证全图遍历所需的最小窗口大小和充分大窗口时的最优探索时间；

**💡 创新点**

创新点在于给出了两种模型下窗口大小与探索时间的紧贴上下界：最小窗口大小证明为 Ω(m)，且可实现的上界为 O(ϵ(m)·m + n log²n)，探索时间在 KT_1 下达到与窗口大小相当，KT_0 下为 O((m−n+1)n + n log²n)，并在多种参数化下（尤其 m = n^{1+Θ(1)} 与仅按 n 计）实现了近乎完全匹配的上、下界；

**🔧 技术方法**

主要技术包括：构造两种基于可见性模型的探索算法 _1（1-hop 视角）与 _0（0-hop 视角），利用潜能函数和极值图论（girth 与边数关系）分析探索进度；利用自适应对手策略（阻塞/删除边）构造下界证明；以及对窗口大小与图结构参数的细致数学分析；

**📊 数据集**

该工作为纯理论研究，无实验数据集，全部结果通过数学证明与构造对手例证得到；

**📈 对比分析**

通过理论上对比上界与下界，作者展示了在两种模型下算法性能与最优解几乎相等；在极限参数化（m = n^{1+Θ(1)} 或仅按 n）下，达到 Θ(m) 或 Θ(n^2) 的窗口大小与 Θ((m−n+1)n) / Θ(m) 的探索时间，证明了算法在理论上的最优性；

**⚠️ 局限性**

局限性：仅考虑确定性算法，未探讨随机化可能带来的窗口缩短或时间提升；虽然上界与下界仅相差多项式对数因子，但仍未完全消除该差距；此外，只处理静态底层图结构的 T-间隔连通模型，未覆盖更一般的动态连通性假设。

---

## 102. On Optimizing Electrode Configuration for Wrist-Worn sEMG-Based Thumb Gesture Recognition

**arXiv ID:** 2604.04623 | [PDF](https://arxiv.org/pdf/2604.04623v1)

**作者:** Wenjuan Zhong `[一作]` (University of Edinburgh), Kianoush Nazarpour `[通讯]` (University of Edinburgh)

**通讯引用:** 3599 | [OpenAlex ID](https://openalex.org/A5070255963)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究系统评估了腕部表面肌电（sEMG）电极配置对拇指手势识别性能的影响，探讨肌肉区域、参考方案、通道数与空间密度等因素；

**💡 创新点**

创新点在于首次将高密度单极与低密度双极电极进行对比，发现单极配置和腕侧伸肌区域能显著提升识别准确率，并提出空间密度与识别效率之间的权衡指标FOM；

**🔧 技术方法**

采用卷积神经网络（CNN）和时间卷积网络（TCN）进行特征提取与分类，配合积分梯度（IG）实现电极重要性可解释性；

**📊 数据集**

使用10名健康右手受试者的拇指手势数据（静止、左右上下滑动、单击、连续点击）与同步多摄像机3D手部运动标注；

**📈 对比分析**

通过十折交叉验证与方差分析比较不同配置的识别精度，单极32通道MAIZE电极实现最高准确率0.90，单极15通道仅略低0.885，低密度双极15通道仅0.823；

**⚠️ 局限性**

局限性包括受试者样本量有限、仅针对健康受试者、仅评估静态手腕位置且未考虑长时间使用或运动中的信号漂移，未来需扩展到多姿态和多模态融合。

---

## 103. Lotka-Sharpe Neural Operators for Control of Population PDEs

**arXiv ID:** 2604.03892 | [PDF](https://arxiv.org/pdf/2604.03892v1)

**作者:** Miroslav Krstic `[一作]`, Carina Veil `[通讯]` (KTH Royal Institute Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究基于年龄结构的捕食者-猎物偏微分方程模型，针对Lotka‑Sharpe（LS）参数ζ的隐式定义，证明其Lipschitz连续性，基于此构建神经算子近似，并设计在使用近似ζ时具有半全局实际渐近稳定性的鲁棒反馈控制；进一步给出在线自适应估计方案；通过仿真验证控制器性能；

**💡 创新点**

①首次证明LS映射在生物可行域上的Lipschitz连续性，为其统一近似提供理论依据；②在控制律中将LS参数视为算子，实现对隐式参数的神经算子逼近；③给出误差传播分析，证明使用近似ζ仍能保证半全局实际渐近稳定性，满足正控制输入约束；④结合自适应估计展示了在线逼近的可行性；

**🔧 技术方法**

神经算子（Fourier Neural Operator）学习、Lipschitz连续性分析、通用逼近定理、Lyapunov 稳定性分析、误差传播与鲁棒性证明、梯度自适应估计方法；

**📊 数据集**

合成数据集：从高斯形的繁殖率k(a)和死亡率μ(a)随机参数生成1000个训练样本和100个测试样本；自适应实验使用基于相同分布的20000个采样点的在线估计数据；

**📈 对比分析**

将神经算子逼近误差与数值求根法比较，训练均方误差3.4e-5，残差保持在±0.001；仿真显示控制输入始终为正，种群随时间收敛至目标平衡；自适应仿真亦能驱动系统至设定点；整体性能满足理论稳定性预期；

**⚠️ 局限性**

仅在有界、紧致样本空间内给出Lipschitz性与逼近保证，缺乏对大规模/未知分布样本的推广；自适应估计未给出严格稳定性证明；仿真仅基于合成数据，未验证在实际生态系统中的鲁棒性；控制律对ζ误差的容限需人工设定，计算量和实时实现未详细讨论；

---

## 104. MMP-Refer: Multimodal Path Retrieval-augmented LLMs For Explainable Recommendation

**arXiv ID:** 2604.03666 | [PDF](https://arxiv.org/pdf/2604.03666v1)

**作者:** Xiangchen Pan `[一作]` (Huazhong University of Science and Technology), Wei Wei `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 254399 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于多模态检索路径的LLM解释性推荐框架MMP-Refer。

**💡 创新点**

创新点在于联合残差量化编码实现多模态表征与基于模态利用规则的启发式检索路径收集。

**🔧 技术方法**

采用联合残差量化编码、LLM2CLIP、RQ-VAE、序列编码器、GAT图编码、Mixture-of-Experts以及LLaMA2等技术。

**📊 数据集**

使用Amazon Review的Baby、Sports、Clothing三子集进行实验。

**📈 对比分析**

与六个基线（NRT、Att2Seq、PETER、PEPLER、CER、G-Refer）和不同规模LLM对比，MMP-Refer在BERT/F1、BART等解释性和稳定性指标上均显著优于基线，提升约4%–12%。

**⚠️ 局限性**

局限性包括检索路径数量敏感、LLM参数受限导致提升空间有限，且在极稀疏数据场景下可解释性仍需进一步验证。

---

## 105. Predict, Don't React: Value-Based Safety Forecasting for LLM Streaming

**arXiv ID:** 2604.03962 | [PDF](https://arxiv.org/pdf/2604.03962v1)

**作者:** Pride Kavumba `[一作]` (SB Intuitions Corp), Masaya Ohagi `[通讯]` (SB Intuitions Corp)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出StreamGuard，一个统一的文本安全防护系统，能够在输入和流式输出阶段进行安全审核，并将流式输出审核转化为预测未来风险的任务；

**💡 创新点**

创新点在于将流式审核从传统的边界检测转变为风险预测，使用Monte Carlo rollouts生成软的未来风险标签，支持模型无关、跨tokenizer迁移，显著降低标注成本；

**🔧 技术方法**

采用Monte Carlo rollouts采样可能的续写，用安全评判器评分生成软目标，训练一个对prefix的风险分数预测器（sigmoid回归），并结合多模型混合rollout、不同聚合规则和监督密度来提升鲁棒性；

**📊 数据集**

在多项公开安全基准上评估：输入审核使用OpenAI Safety Benchmark、Toxicity、Aegis、WildGuard等；输出审核使用Qwen3Guard、OpenAI、Aegis、WildGuard等；流式评估使用Qwen3Guard流式基准；过阻测试使用对抗性benign数据集；跨tokenizer迁移测试使用Gemma和Qwen backbone；

**📈 对比分析**

与离线后期审核模型（如LlamaGuard、WildGuard）以及流式边界检测基线Qwen3Guard进行对比。StreamGuard在8B模型上输入F1从86.7提升至88.2，流式输出F1从80.4提升至81.9；在流式基准上达到F1 97.5、召回 95.1%、及时干预 92.6%、漏判率 4.9%；在跨tokenizer迁移中仍保持高F1（Gemma 81.3、Qwen 81.3）并且漏判率降至3.5%；总体显示在保持低延迟的前提下，预测式审核性能优于传统边界检测；

**⚠️ 局限性**

局限性包括：对rollout生成的安全评判器依赖，导致额外计算开销；对极长文本rollout成本高，可能需要更高效的采样策略；在不同tokenizer/模型间迁移时仍需调节聚合规则和监督密度；未对抗性攻击或低质量生成的鲁棒性进行充分评估。

---

## 106. Ethical Implications of Training Deceptive AI

**arXiv ID:** 2604.03250 | [PDF](https://arxiv.org/pdf/2604.03250v1)

**作者:** Jason Starace `[一作]` (University of Idaho), Terence Soule `[通讯]` (University of Idaho)

**通讯引用:** 1741 | [OpenAlex ID](https://openalex.org/A5111629661)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了“欺骗研究层级（DRL）”框架，借鉴生物安全等级体系，将AI欺骗研究按风险维度进行分类，并通过八个案例评估其适用性。

**💡 创新点**

创新点在于：①以风险属性而非研究者意图为分类依据；②引入“最高维度决定”规则和双重开发义务；③将生态有效性视为分类的启发式判定指标；④将AI4People伦理框架的五大维度映射为风险维度。

**🔧 技术方法**

主要技术包括：功能性欺骗定义、AI4People伦理维度评估矩阵、手工案例评估、对安全训练（RLHF、对抗训练）效果的实验分析。

**📊 数据集**

使用了公开论文中已有的数据集和实验设置，例如LSPO的Werewolf文本、Cicero的Diplomacy对话、Hubinger的背门触发文本、Dogra的LobbyLens政法文本等，但未构造新的专用数据集。

**📈 对比分析**

通过对八个案例的人工评估，DRL框架在不同等级下得到一致且符合直觉的分类；在边界案例中引入生态有效性启发式进一步细化判定；由于是定性评估，没有提供量化性能指标。

**⚠️ 局限性**

局限性包括：案例数量有限且聚焦LLM，评估者为框架作者，缺乏独立复核；实施可行性和跨模态适用性待验证；时间演进与重新评估机制尚未完善；未考虑与现有治理结构的集成细节。

---

## 107. Physical Sensitivity Kernels Can Emerge in Data-Driven Forward Models: Evidence From Surface-Wave Dispersion

**arXiv ID:** 2604.04107 | [PDF](https://arxiv.org/pdf/2604.04107v1)

**作者:** Ziye Yu `[一作]` (Institute of Geophysics, China Earthquake Administration), Xin Liu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 58320 | [OpenAlex ID](https://openalex.org/A5100352324)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了神经网络替代模型在表面波传播中的前向映射，探讨其自动微分梯度能否重现理论Fréchet敏感度核。

**💡 创新点**

首次证明数据驱动模型可以“自然”产生微分物理结构，但同时揭示训练先验强度会在梯度中引入非物理特征。

**🔧 技术方法**

使用深度前馈神经网络作为散射传播的前向模型，并利用自动微分计算梯度、Fisher信息矩阵进行逆演与不确定性估计。

**📊 数据集**

采用大规模合成的1‑D剪切波速度结构和对应2–60 s Rayleigh 与 Love 相位速度曲线，分别构建弱先验与强低速带先验两套训练集。

**📈 对比分析**

将网络梯度与理论核通过余弦相似度、皮尔逊相关系数和绝对误差进行对比，短至中期周期下相似度>0.97、误差<0.003；逆演时可准确重建模型并给出合理的深度不确定性。

**⚠️ 局限性**

长周期/深层时梯度与理论核偏差显著，训练先验过强会在梯度中出现非物理峰值，导致低敏感度区误差扩大，需谨慎解释神经梯度的物理意义。

---

## 108. Soft Tournament Equilibrium

**arXiv ID:** 2604.04328 | [PDF](https://arxiv.org/pdf/2604.04328v1)

**作者:** Saad Alqithami `[一作]` `[通讯]`, Saad Alqithami

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种端到端可微的Soft Tournament Equilibrium（STE）框架，用以从含噪、上下文相关的成对比较数据中学习并计算两种经典的无向图解集（Top Cycle和Uncovered Set），从而在非传递性环境下评估通用人工智能代理；

**💡 创新点**

创新点在于将传统离散的锦标赛解决方案软化为连续可微算子（soft reachability与soft covering），并证明其在温度趋零时与经典解一致，进一步提供了一致性、稳定性与样本复杂度分析；

**🔧 技术方法**

主要技术包括基于上下文的Bradley‑Terry‑Luce模型、软化的图遍历与覆盖算子、温度调度的确定性退火、以及利用log‑sum‑exp与softmin/softmax实现的全流程可微；

**📊 数据集**

实验采用合成概率锦标赛数据（可调非传递性、噪声、稀疏度）以及真实的LLM评测数据（Chatbot Arena、AgentBench），通过对比基线（BTL、Elo、TrueSkill、Rank Centrality、HodgeRank、SCO）进行评估；

**📈 对比分析**

在合成数据上STE在核心恢复F1与Jaccard指数上显著优于传统排序和评分方法；在真实LLM数据中，STE能识别出多元的“核心代理”，并通过bootstrap稳定性检验显示高置信度，整体预测误差和校准度（ECE、Brier）均优于基线；

**⚠️ 局限性**

局限在于计算复杂度随代理数和路径长度呈立方级，需稀疏或近似技巧；温度调节与硬件资源对收敛与性能有较大影响；目前仅实现了Top Cycle与Uncovered Set，未涵盖更细粒度的锦标赛解。

---

## 109. I-CALM: Incentivizing Confidence-Aware Abstention for LLM Hallucination Mitigation

**arXiv ID:** 2604.03904 | [PDF](https://arxiv.org/pdf/2604.03904v1)

**作者:** Haotian Zong `[一作]` (Johns Hopkins University), Gillian K. Hadfield `[通讯]` (Johns Hopkins University)

**通讯引用:** 3948 | [OpenAlex ID](https://openalex.org/A5000711679)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种仅通过提示的框架，利用模型自报的置信度、显式的答/不答奖励机制以及轻量级的真实性、谦逊和责任原则，诱导大型语言模型在可验证事实问答中实现可控的拒绝与回答行为。

**💡 创新点**

创新点在于将置信度诱导、奖励对齐与道德规范集成于同一提示层面，实现无需模型再训练即可调节模型的回答与拒绝决策，并揭示了奖励与拒绝率、假答率之间的可调节前沿。

**🔧 技术方法**

技术手段包括：自报文本置信度（verbal confidence）提取、两阶段提示（答/不答与最佳猜测）、显式奖励框架（+R、-β、+γ），以及包含真诚、谦逊、责任的规范性提示。

**📊 数据集**

主要使用了PopQA、TriviaQA、SimpleQA Verified等事实问答数据集；实验以GPT‑5 mini、GPT‑4o mini、Gemini‑3.1‑Flash‑Lite、Meta‑Llama‑3‑8B‑Instruct、Qwen3‑4B‑Instruct‑2507等模型为主。

**📈 对比分析**

通过与纯评估（Pure Eval）对比，评估指标为回答误差率（FAR_answered）、整体误差率（FAR_overall）、覆盖率、拒绝‑错误比（AER）等。结果显示：在PopQA上，方案B（+1,-1,+0.4）将FAR_answered从52.3%降至41.0%，加入规范进一步降至34.2%；覆盖率从96.5%降至67.9%。在不同奖励组合下，模型在拒绝率与误答率之间展现出平滑的折衷前沿。

**⚠️ 局限性**

局限性包括：模型仍未完全遵循贝叶斯最优阈值策略，置信度表达与实际概率不完全一致；仅在可验证事实问答场景中测试，未覆盖不可回答或歧义问题；在持续对话或域迁移中效果未知；缺乏与检索/工具使用等完整对话代理的集成验证。

---

## 110. NEURA: A Unified and Retargetable Compilation Framework for Coarse-Grained Reconfigurable Architectures

**arXiv ID:** 2604.04236 | [PDF](https://arxiv.org/pdf/2604.04236v1)

**作者:** Shangkun Li `[一作]` (Hong Kong University of Science and Technology), Cheng Tan `[通讯]` (Google and Arizona State University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 NEURA，一个统一且可重定向的 CGRA 编译框架，能够将复杂的控制流彻底转化为纯数据流，从而实现完整的 kernel 加速。

**💡 创新点**

创新点在于引入基于谓词的类型系统和谓词管理操作，将控制上下文嵌入数据值，实现层次化的谓词融合，彻底消除了控制-数据流语义鸿沟。

**🔧 技术方法**

技术上实现了三大阶段：① CDFG 预处理（函数参数提升、live-in 正则化）；② 借助谓词类型系统将所有 SSA 值转换为 predicated 类型；③ 通过确定性重写规则把控制流边映射为数据流节点，最终得到统一的数据流 IR；并提供了硬件无关和硬件特定的优化，如常量折叠、数据类型对齐、计算模式融合、循环流式优化；同时实现了轻量级 ISA 扩展（3 条新指令 + 逻辑 AND）。

**📊 数据集**

使用了 PolyBench、MachSuite、CGRA-Bench、CHStone 这四类基准，以及两大真实应用：2 层 GCN 与 LU 分解，涵盖了嵌套循环、分支、循环展开等多种控制流特征。

**📈 对比分析**

采用统一的映射算法和 6×6 fabric 归一化，对比 Marionette、RipTide、ICED 等 SOTA 框架；在 spatio‑temporal 目标下 NEURA-ST 平均提升 2.20×；在低功耗 spatial‑only 目标下 NEURA-SO 与 RipTide 的性能相近，甚至在部分基准上超越；在实际应用上 NEURA-ST 最高可达 2.71× 的几何平均加速。

**⚠️ 局限性**

主要局限在：① 目前仅支持 6×6 fabric 的实验，未验证更大规模下的可扩展性；② 对于极度复杂或动态生成的控制流（如 runtime 反射、递归调用）尚未测试；③ 对硬件特定优化的支持仍需通过手工规则扩展，自动化程度有限。

---

## 111. Testing the Limits of Truth Directions in LLMs

**arXiv ID:** 2604.03754 | [PDF](https://arxiv.org/pdf/2604.03754v1)

**作者:** Angelos Poulis `[一作]` (Boston University), Evimaria Terzi `[通讯]` (Boston University)

**通讯引用:** 5274 | [OpenAlex ID](https://openalex.org/A5005972547)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统探究大语言模型中线性真值方向的普适性，系统评估层级、任务类型、难度以及模型指令的影响；

**💡 创新点**

首次全面检验真值方向在不同层、不同任务与指令下的可推广性，揭示其高度依赖层级与任务难度，并指出模型指令能改变真值几何并提升跨任务泛化；

**🔧 技术方法**

使用线性探测器（logistic 回归）在模型残差流上提取特征，计算 AUROC、余弦相似度及跨任务泛化性能；

**📊 数据集**

构造了六级事实任务（F0–F5）与三级算术任务（A1–A3）的合成数据集，并在 Llama-3.1‑8B‑Instruct、Gemma‑2‑2b‑it、Gemma‑2‑9b‑it 等模型上进行实验；

**📈 对比分析**

通过比较不同层、不同指令（无提示 vs. 询问真值）和不同任务难度下的 AUROC 与跨任务性能，发现早层事实任务真值方向优良，算术任务和高难度任务在后层才出现且跨任务泛化显著下降；

**⚠️ 局限性**

真值方向在需要多步推理或计数的任务中泛化极差，受层级和指令影响大，说明其普适性有限，难以在复杂推理场景中可靠使用。

---

## 112. SDVDiag: Using Context-Aware Causality Mining for the Diagnosis of Connected Vehicle Functions

**arXiv ID:** 2604.03391 | [PDF](https://arxiv.org/pdf/2604.03391v1)

**作者:** Matthias Weiß `[一作]` (Institute of Industrial Automation and Software Engineering University of Stuttgart), Michael Weyrich `[通讯]` (Institute of Industrial Automation and Software Engineering University of Stuttgart)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种结合人类反馈和系统上下文信息的多模态因果挖掘方法，应用于联网车辆功能的持续诊断；

**💡 创新点**

创新点在于引入强化学习从人类反馈（相对比较）持续训练因果模型，并结合分布式跟踪、服务拓扑等上下文实现多模态剪枝与规则注入；

**🔧 技术方法**

采用图神经网络编码多维时序指标、对比学习生成嵌入、基于强化学习（DDPG+Q‑learning）进行因果边预测、并通过Zipkin跟踪验证、规则引擎补充隐藏因果；

**📊 数据集**

使用自动泊车（AVP）应用的指标和追踪日志（Prometheus, Zipkin）作为数据源；

**📈 对比分析**

与纯数据驱动的因果发现方法（如PC/ GES）对比，原始模型精度仅14%，经过人类反馈后提升至32%，最终经过上下文剪枝后精度100%，召回率84%，与基准相比显著提升；

**⚠️ 局限性**

局限包括：实验规模仅七个服务，难以验证大规模微服务体系的可扩展性；地面真相依赖专家判断而非客观因果；对错误场景的模拟有限，缺乏系统化的故障注入。

---

## 113. Boosted Distributional Reinforcement Learning: Analysis and Healthcare Applications

**arXiv ID:** 2604.04334 | [PDF](https://arxiv.org/pdf/2604.04334v1)

**作者:** Zequn Chen `[一作]` (Dartmouth), Wesley J. Marrero `[通讯]` (Dartmouth)

**通讯引用:** 413 | [OpenAlex ID](https://openalex.org/A5050044642)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种Boosted Distributional Reinforcement Learning（BDRL）算法，用于在多主体环境中通过优化个体回报分布并保持相似主体间分布一致性来提升决策一致性和效果

**💡 创新点**

创新点包括：①引入2-Wasserstein距离对回报分布进行一致性正则化，解决传统DRL在分布不重叠时梯度消失的问题；②设计后更新投影步骤，将分布约束转化为可解的凸二次规划，实现高效稳定的约束满足；③在同类主体中通过“boosting”机制提升低性能主体的回报而不损害高性能参考主体

**🔧 技术方法**

使用的技术包括：分布式强化学习（distributional RL）框架、2-Wasserstein距离正则化、Lagrangian多重约束、后更新投影凸优化、k-means聚类分组、Q-学习与深度Q网络对比实验

**📊 数据集**

基于美国全国健康与营养调查（NHANES 2009-2016）数据，构建了一个基于年龄50-54岁、黑人和白人、无心脏卒中/心肌梗死病史的约1672万成人样本的心血管疾病风险与高血压治疗模拟模型

**📈 对比分析**

与传统DRL、Deep Q-Network、Q-learning等基线相比，BDRL在低、中、高风险组内均能提升QALY，特别是中位与低绩效个体的QALY提升显著；同时保持或略高于基线在高绩效个体上的表现；实验显示BDRL在约束满意度、学习稳定性和计算效率方面均优于其他方法

**⚠️ 局限性**

局限性包括：①对主体相似度采用离散聚类，可能受聚类方式和数量影响；②投影与正则化参数（λ、ε、ρ、α）需要手动调优，对样本量有限时可能不够稳健；③基于仿真模型评估，真实临床环境下的分布偏移与模型不确定性尚未充分处理；④仅考虑回报分布一致性，未结合其他安全或公平约束

---

## 114. High-Fidelity Mural Restoration via a Unified Hybrid Mask-Aware Transformer

**arXiv ID:** 2604.03984 | [PDF](https://arxiv.org/pdf/2604.03984v1)

**作者:** Jincheng Jiang `[一作]` (Northeastern University), Zheng Zheng `[通讯]` (Northeastern University)

**通讯引用:** 20770 | [OpenAlex ID](https://openalex.org/A5100414978)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种混合 Mask‑Aware Transformer（HMAT）框架，用于壁画的高保真数字修复，既能恢复大面积结构缺损，又能严格保留未损坏区域的真实像素。

**💡 创新点**

创新点包括：①将 Mask‑Aware Dynamic Filtering（MADF）与 Transformer 瓶颈融合的混合编码器；②Mask‑Conditional Style Fusion 模块，依据破损几何形状动态调节生成策略；③Teacher‑Forcing Decoder 的硬门控 skip 机制，保证生成过程中不修改原有像素；④统一的训练框架兼顾结构恢复与纹理细节。

**🔧 技术方法**

技术手段包括：MADF 位置感知卷积、Transformer 关注机制、Mask‑Conditional Style Fusion、Teacher‑Forcing Decoder、Refinement Network、GAN 对抗损失、感知损失和 L1 误差。

**📊 数据集**

使用的公开数据集：DHMural（13,233 张壁画图像）和 Nine‑Colored Deer（4,000 张 256×256 片段）两种不同遮罩覆盖率（Moderate、Severe）进行实验。

**📈 对比分析**

与现有纯 CNN（MADF）和纯 Transformer（MAT）方法对比，HMAT 在 Nine‑Colored Deer 上获得最高 PSNR（27.37）和 SSIM（0.903），在 DHMural 上虽略低于 MADF，但在结构连贯性和真实像素保真度方面表现更好；FID 较高但说明仍优先考虑保真而非整体视觉平滑。

**⚠️ 局限性**

局限性：在极端遮挡或纹理极度复杂的场景下，模型仍可能产生细节误差；由于严格保真导致 FID 较高，生成纹理可能显得稍显平滑；训练过程需要较大 GPU 资源和专门的损失调节。

---

## 115. Explainable Model Routing for Agentic Workflows

**arXiv ID:** 2604.03527 | [PDF](https://arxiv.org/pdf/2604.03527v1)

**作者:** Mika Okamoto `[一作]` (Georgia Institute of Technology), Mark Riedl `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 7514 | [OpenAlex ID](https://openalex.org/A5061883150)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个可解释的代理路由框架，通过技能概况、成本感知的多目标优化和自然语言解释实现对代理工作流中模型分配的可追踪性和透明度。

**💡 创新点**

1）构建统一技能词典并将公开基准与模型性能映射为可解释的能力档案；2）设计可追踪的目标导向与预算约束路由算法；3）将决策轨迹转化为人类可理解的自然语言解释，避免事后合理化。

**🔧 技术方法**

技能匹配评分、基于预算的动态规划、全局成本-质量权衡优化、LLM驱动的任务与模型技能评估、自然语言生成用于解释。

**📊 数据集**

使用公开基准（TextArena、Search Arena、BFCL v4、SWE‑bench、LiveCodeBench、MMMU、GPQA、MMLU‑Pro、MATH‑500、AIME）评估模型；在客户支持案例中使用5种不同价格-性能平衡的LLM（Gemini‑3‑Pro、Claude‑Opus‑4.5、GPT‑5.2、Llama‑4‑Maverick、Mistral‑Small‑3.1）。

**📈 对比分析**

通过在三种成本敏感度（0.0、0.5、1.0）下展示路由分配和解释，说明系统在保持关键任务高质量的同时显著降低成本；在案例中成本可降低约70%–90%，同时仅在高复杂度任务中保持最优模型。

**⚠️ 局限性**

仅关注路由决策的可解释性，未评估下游任务完整性能；对技能词典和LLM评估结果的准确性依赖公开基准；缺乏用户实验验证解释的有效性；在大规模多模型环境下算法规模和计算开销待进一步评估。

---

## 116. Hume's Representational Conditions for Causal Judgment: What Bayesian Formalization Abstracted Away

**arXiv ID:** 2604.03387 | [PDF](https://arxiv.org/pdf/2604.03387v1)

**作者:** Yiling Wu `[一作]` `[通讯]` (BridgeM), Yiling Wu (BridgeM)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对休谟因果判断的心理机制进行系统分析，提取出三个表征条件（经验根基、结构化检索、活力转移），并追踪这些条件在从休谟到贝叶斯概率论再到预测处理（Predictive Processing）的形式化过程中的保留与抽象。

**💡 创新点**

首次将休谟文本中隐含的三大表征条件完整归纳为一组可检验的条件，并揭示它们在后续理论演进中的渐进式抽象，指出现代大型语言模型（LLM）在实现更新结构时缺失这些条件，凸显其理论意义。

**🔧 技术方法**

采用哲学文本分析、文献综述与概念梳理方法，对休谟原著、贝叶斯理论、预测处理框架及LLM工作原理进行对比与阐释。

**📊 数据集**

无具体数据集；论文以理论推理和已有文献为主要依据。

**📈 对比分析**

通过概念对照与逻辑推演进行比较；未涉及实验或数值性能评估，因其为哲学理论分析，主要关注结构与逻辑一致性。

**⚠️ 局限性**

缺乏经验验证和可操作的实验设计，不能直接评估三条件对实际认知或人工系统的预测效能；对预测处理的不同实现版本可能缺乏细致区分；论文关注点集中在理论层面，未给出针对LLM的具体改进方案。

---

## 117. Messages in a Digital Bottle: A Youth-Coauthored Perspective on LLM Chatbots and Adolescent Loneliness

**arXiv ID:** 2604.03470 | [PDF](https://arxiv.org/pdf/2604.03470v1)

**作者:** Jinyao Liu `[一作]` (Wycombe Abbey), Di Fu `[通讯]` (University of Surrey)

**通讯引用:** 2128 | [OpenAlex ID](https://openalex.org/A5001946970)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

以16岁移民青少年为主角，结合文献综述，对大型语言模型（LLM）驱动的聊天机器人在青少年孤独感中的作用进行批判性合成，提出了针对不同青少年子群体的设计框架。

**💡 创新点**

创新点在于：① 采用青少年第一人称视角作为研究主体，突出少数群体的需求；② 建立“不同子群体—不同需求—不同设计”三层敏感性框架；③ 提出了三条针对性设计建议（群体适配、危机升级路径、持续透明度）。

**🔧 技术方法**

未使用具体技术实现，而是通过对现有LLM聊天机器人特性的理论分析（如可用性、情感识别、算法个性化等）。

**📊 数据集**

未使用任何数据集；研究基于文献综述、作者个人经历与协作讨论。

**📈 对比分析**

无实验比较；缺乏量化指标和性能评估，结果仅为概念性框架与设计启示。

**⚠️ 局限性**

局限性包括：① 仅来自一名青少年的主观经验，难以泛化；② 研究缺乏实证验证与量化数据；③ 对聊天机器人技术细节及算法表现未进行深入技术评估。

---

## 118. Mixture-of-Experts in Remote Sensing: A Survey

**arXiv ID:** 2604.03342 | [PDF](https://arxiv.org/pdf/2604.03342v1)

**作者:** Yongchuan Cui `[一作]` (Aerospace Information Research Institute, Chinese Academy of Sciences), Lajiao Chen `[通讯]` (Aerospace Information Research Institute, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对Mixture-of-Experts（MoE）在遥感中的研究进展进行系统综述，梳理其原理、架构与应用。

**💡 创新点**

首次将MoE与遥感任务的对应关系进行统一框架化，指出专家化、门控与训练在不同任务中的关键设计。

**🔧 技术方法**

综合讨论MoE的专家网络、门控策略（top‑k、学习型路由）、训练方法与系统实现。

**📊 数据集**

参考多源遥感数据集，包括光学、SAR、LiDAR、多光谱、超光谱等。

**📈 对比分析**

通过对比现有基线模型和MoE方法，展示在分类、检测、变化检测、时序建模、融合等任务中通常能提升准确率或效率，但提升幅度依赖任务与数据。

**⚠️ 局限性**

局限主要在缺乏统一评测基准、门控不稳定导致专家利用率不均、计算成本与模型可解释性不足。

---

## 119. Knowledge Packs: Zero-Token Knowledge Delivery via KV Cache Injection

**arXiv ID:** 2604.03270 | [PDF](https://arxiv.org/pdf/2604.03270v1)

**作者:** Andrey Pustovit `[一作]` `[通讯]` (Independent Researcher), Andrey Pustovit (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出通过预先计算 KV 缓存（Knowledge Packs）在推理时注入知识，从而实现零令牌成本的 RAG 替代方案，并在 KV 缓存上实现无训练的行为调节。

**💡 创新点**

证明 KV 前缀等价性并指出正确聊天模板对准确率至关重要；利用 KV 值空间的对比增量实现激活调节；两种功能在 α≤0.7 范围内可同时使用而互不干扰。

**🔧 技术方法**

使用预计算 KV 缓存、RoPE 关键值分离、k‑means 分区路由、KV 合成以及对比向量值差（V‑delta）值空间激活调节技术。

**📊 数据集**

主要使用 HotpotQA（多跳 QA）、合成事实集用于路由评估、15 个编码任务用于价值调节以及 200 条桥接问题用于双通道实验。

**📈 对比分析**

与 RAG 在相同事实文本下对比，KV 缓存在 Qwen3-8B 和 Llama-3.1-8B 上实现与 RAG 完全相同的 EM，且在多步检索中可节省高达 95% 的令牌；行为调节方面相较文本提示获得约 50% 的效果；双通道实验显示 α≤0.7 时可保留 72–73% 的准确率并提升 20% 的正式度。

**⚠️ 局限性**

KV 缓存与模型架构绑定，无法跨模型使用；值空间调节效果仅为文本提示的一半；对真实知识库路由的鲁棒性未完全验证；双通道模式下需降低 α，暗示两种子空间并非完全正交。

---

## 120. Beyond Static Vision: Scene Dynamic Field Unlocks Intuitive Physics Understanding in Multi-modal Large Language Models

**arXiv ID:** 2604.03302 | [PDF](https://arxiv.org/pdf/2604.03302v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 121. Hardware-Level Governance of AI Compute: A Feasibility Taxonomy for Regulatory Compliance and Treaty Verification

**arXiv ID:** 2604.04712 | [PDF](https://arxiv.org/pdf/2604.04712v1)

**作者:** Samar Ansari `[一作]` `[通讯]` (University of Chester), Samar Ansari (University of Chester)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一套包含20种硬件层级治理机制的分类体系，并对每种机制的技术可行性进行评估，随后将这些机制与四类治理场景（国内监管、双边协定、多边条约验证、行业自律）进行映射，提出了多层次治理结构；

**💡 创新点**

创新点在于系统化地把硬件治理机制按功能（监测、验证、执行）归类，给出从现成可部署到需研发再到纯理论的四级可行性评估；引入对抗者分层威胁模型，并将机制与现有国际治理范式（IAEA、CWC、金融KYC）进行对照，形成可操作的治理路径；

**🔧 技术方法**

主要技术包括：云元数据与计费记录、工作负载分类器、KYC身份验证、功耗监测、芯片位置追踪与注册、可信执行环境（TEE）与远程证明、加密学习/训练证明、FlexHEG可信保证处理器、硬件离线许可、网络带宽限制、硬件离线与远程禁用、出口控制与供应链监控；

**📊 数据集**

文章并未引入新的实验数据集，而是引用了现有研究中使用的 MIT Supercloud、Bloom、BLOOM等工作负载数据集来论证分类器与计量方法的可行性；

**📈 对比分析**

由于论文属于系统综述与概念性分析，没有进行实验对比或性能评估；相对已有政策文献与核不扩散机制进行概念性对照，指出在技术成熟度、执行成本与政治可接受性方面的差距；

**⚠️ 局限性**

主要局限包括：可行性评估基于公开文献与专家判断，缺乏实测验证；对量子、光学或神经形态等非传统AI硬件的治理问题未覆盖；未详细讨论成本、能耗、产业接受度及国际政治经济影响；

---

## 122. AEGIS: Scaling Long-Sequence Homomorphic Encrypted Transformer Inference via Hybrid Parallelism on Multi-GPU Systems

**arXiv ID:** 2604.03425 | [PDF](https://arxiv.org/pdf/2604.03425v1)

**作者:** Zhaoting Gong `[一作]` (North Carolina State University), Wujie Wen `[通讯]` (North Carolina State University)

**通讯引用:** 2539 | [OpenAlex ID](https://openalex.org/A5067226050)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个多GPU协同加密Transformer推理框架，使得长序列加密Transformer能够在多GPU上高效并行运行。

**💡 创新点**

创新点在于：①基于密文依赖（模数链与Token）进行一致性放置，避免频繁同步与复制；②通过操作重排实现通信与计算重叠，显著隐藏跨设备延迟；③将应用层与加密层依赖融合，形成统一的调度与通信策略。

**🔧 技术方法**

采用CKKS全同态加密、RNS（基数）并行、CUDA GPU、LibTorch + LiberateFHE以及自研编译器层面指令调度与通信插入技术。

**📊 数据集**

使用BERT-Base模型在SST-2情感分类数据集进行端到端评估。

**📈 对比分析**

与HEBooster、Cinnamon、Hydra等基线相比，在2‑GPU/4‑GPU配置下实现最高92.98%扩展率、3.86×速度提升、69.1%设备内存降低，通信量分别减少57.9%（FFN）和81.3%（自注意力），整体性能显著优于现有方案。

**⚠️ 局限性**

局限性包括：需要针对CKKS参数和密文布局进行手工调优；高安全级别参数导致密文尺寸大，影响迁移性；实验仅在单机多GPU上验证，未探讨多机或异构互联环境。

---

## 123. IC3-Evolve: Proof-/Witness-Gated Offline LLM-Driven Heuristic Evolution for IC3 Hardware Model Checking

**arXiv ID:** 2604.03232 | [PDF](https://arxiv.org/pdf/2604.03232v1)

**作者:** Mingkai Miao `[一作]` (Hong Kong University of Science and Technology), Hongce Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 459 | [OpenAlex ID](https://openalex.org/A5003614499)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用LLM驱动的离线代码演化框架，对IC3模型检验器的实现进行slot‑restricted、可审计的微调，并通过proof‑/witness‑gated验证确保改动保持完整性，最终得到性能更佳且无运行时ML依赖的检查器。

**💡 创新点**

创新点包括：①离线LLM演化而非在线推理；②将代码改动限制在可审计的slot内；③使用证明/证据门控保证安全；④设计Compass&Jump搜索策略以引导跨模块改进；⑤将IC3的实现工程化视为可自动化的代码演化过程。

**🔧 技术方法**

核心技术：GPT‑5‑Codex 作为程序员与评估者代理；IC3实现与CaDiCaL SAT后端；proof‑/witness‑gated验证（certificaiger、aigsim）；Compass&Jump搜索策略；PAR2 等性能度量；AIGER/HWMCC benchmark 以及工业验证数据。

**📊 数据集**

实验使用两类数据集：①在 100 个公开 HWMCC AIGER 实例上进行 200 轮离线演化；②在另 100 个未用于演化的 HWMCC 实例与 302 个工业验证实例上评估泛化能力。

**📈 对比分析**

通过与 IC3ref、IC3ref‑CaDiCaL、ABC、rIC3‑CaDiCaL 等主流基线在同一 1800 s 超时设置下进行 #Solved（分为 UNSAFE/SAFE）和 PAR2 对比；实验表明演化后的检查器在公共与工业集上显著提升了求解率，PAR2 下降数百秒，证明其在保持正确性前提下实现了实质性能提升。

**⚠️ 局限性**

局限性：①单槽演化或 Naive‑Compose 方案效果有限，凸显需跨槽协同；②LLM 演化受限于预定义的 slot 结构，可能遗漏更大范围的改进空间；③演化过程耗时 200 轮、依赖 GPT‑5‑Codex 资源；④仍需在更大规模模型与多种 SAT 后端上验证其普适性。

---

## 124. Springdrift: An Auditable Persistent Runtime for LLM Agents with Case-Based Memory, Normative Safety, and Ambient Self-Perception

**arXiv ID:** 2604.04660 | [PDF](https://arxiv.org/pdf/2604.04660v1)

**作者:** Seamus Brady `[一作]` `[通讯]`, Seamus Brady

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Springdrift，一个为长寿 LLM 代理设计的持久化运行时，提供审计、持续自感知、案例推理和确定性规范计算等核心能力，并在 23 天实测部署中演示跨会话连续性、跨渠道上下文和自我诊断行为。

**💡 创新点**

创新点包括：1) 以 append‑only JSONL+git 为基础的全程可审计执行子系统；2) 每轮注入的 sensorium，实现零延迟的自感知；3) 混合词典+语义的案例检索机制；4) 基于 Becker 伦理学的确定性规范算子，用于安全决策的可追溯冲突解决。

**🔧 技术方法**

采用 Erlang/OTP+Gleam 编写，利用 OTP 监督进程、ETS 缓存、JSONL append-only 日志、git 备份；实现 D' 安全门控、LSTM+工具链、混合检索、规范算子、sensorium 注入等技术；使用多 LLM（Claude‑Opus、OpenAI、Claude‑3 等）进行推理。

**📊 数据集**

使用自制的 800 条案例+200 条查询的合成基准（四个领域）来评估检索；同时基于部署日志（19 天、24,035 条周期日志、494 条叙事条目）来检验系统行为。

**📈 对比分析**

检索实验：P@4 0.956（混合 CBR）对比 dense‑cosine 0.920，尤其在 Hard 级别提升显著；在部署中未与其他框架做严格基准，仅展示单实例日志的可重现性与自诊断案例。

**⚠️ 局限性**

局限性：仅单实例单操作者实验，缺乏消融与多域、多操作者对比；合成检索基准可能偏向词典匹配；规范算子与 sensorium 等组件的实际安全效果未经过系统化评估；长期扩展性与性能尚未验证。

---

## 125. Training-Free Refinement of Flow Matching with Divergence-based Sampling

**arXiv ID:** 2604.04646 | [PDF](https://arxiv.org/pdf/2604.04646v1)

**作者:** Yeonwoo Cha `[一作]` (KAIST), Seunghoon Hong `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出训练无关的推理框架 Flow Divergence Sampler (FDS)，在流匹配模型的采样过程中，在每一步对当前状态做空间微调，利用速度场的散度指示低不确定性区域，避免轨迹冲突。

**💡 创新点**

创新点包括：① 证明采样误差与速度场散度相关，给出数据自由的散度代理；② 在推理阶段通过零阶扰动搜索实现空间微调，完全不需要重训练；③ 该方法可无缝嵌入任何 ODE 求解器和现有引导机制，形成 plug‑and‑play 的通用插件。

**🔧 技术方法**

技术手段：流匹配 / 条件流匹配模型、Euler/Heun ODE 求解器、Hutchinson 迹估计计算散度、随机扰动零阶搜索、早期阶段截断 T_trunc 与扰动幅度 σ_t 调度。

**📊 数据集**

实验数据集：CIFAR‑10、ImageNet 256×256、2D 合成棋盘、DrawBench 文本‑图像、Cat 数据集（去模糊、超分辨率）。

**📈 对比分析**

与原始 FM、以及通过增加 NFE 的基线（计算量匹配）相比，FDS 在相同时间预算下显著提升生成质量：ImageNet 256×256 FID 从 4.151→3.799（Euler）或 3.637→3.394（Heun）；CIFAR‑10 FID 下降 3.034→2.319；文本‑图像 4 个指标均有所提升；逆问题 FID/LPIPS 均下降。与训练基准（HRF、VRFM）相比，FDS 在参数匹配下更优。

**⚠️ 局限性**

局限性：① 仍需在早期阶段做微调，依赖散度估计的准确性；② 最佳效果在单步微调（N=M=1）时实现，更多迭代收益有限；③ 计算量虽低但不为零，且对高维空间的散度估计可能不稳定；④ 只针对轨迹冲突，无法完全解决所有生成质量问题。

---

## 126. Formalized Information Needs Improve Large-Language-Model Relevance Judgments

**arXiv ID:** 2604.04140 | [PDF](https://arxiv.org/pdf/2604.04140v1)

**作者:** Jüri Keller `[一作]` (TH Köln), Philipp Schaer `[通讯]` (TH Köln)

**通讯引用:** 964 | [OpenAlex ID](https://openalex.org/A5087564658)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用指令调优的LLM自动合成标题、描述、叙述三字段主题，并将其作为输入来生成自动化相关性判断，评估其对检索实验可靠性的影响。

**💡 创新点**

提出了LLM自动化生成完整TREC式主题的方案，证明其能显著降低LLM的正偏差、提升与人类判断的一致性，并提高留一组实验的可复现性。

**🔧 技术方法**

使用指令调优的LLM（如GPT‑4.1、Qwen3-30B、Qwen3-Next‑80B等）、多种提示工程、Cohen κ、MAE、BERTScore、Rouge、留一组（leave‑one‑group‑out）实验等技术。

**📊 数据集**

评估数据集包括 Robust04、TREC Deep Learning 2019（DL19）和 2020（DL20）三大集合。

**📈 对比分析**

通过与人类评标的κ/MAE对比，使用多种提示与LLM组合可达最高κ≈0.46、MAE≈0.18；完全主题化的LLM比仅用查询的κ低10%但显著提升留一组实验的Spearman/TauAP相关性，整体提升实验可靠性。

**⚠️ 局限性**

局限在于生成主题不可逆，无法完整恢复原始用户信息需求；不同提示与LLM之间差异显著，缺乏统一的质量评估标准。

---

## 127. NetSecBed: A Container-Native Testbed for Reproducible Cybersecurity Experimentation

**arXiv ID:** 2604.04121 | [PDF](https://arxiv.org/pdf/2604.04121v1)

**作者:** Leonardo Bitzki `[一作]` (Federal University of Rio Grande do Sul), Angelo Diniz `[通讯]` (AI Horizon Labs and PPGES -- Federal University of Pampa)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

构建了 NetSecBed，一个基于容器化的可扩展测试床，能够在受控环境下自动化执行攻击场景、收集网络流量与日志，并生成可追溯的数据集。

**💡 创新点**

创新点包括：①将攻击、目标服务与流量生成器拆分为单一容器并通过 YAML 声明式规范实现无缝扩展；②提供完整的实验生命周期管道（从配置到特征提取与数据集生成），实现可审计、可重复的证据生成；③在 IoT/IIoT 与传统网络协议之间实现统一的、可追踪的实验框架。

**🔧 技术方法**

技术手段包括 Docker/Kubernetes 容器化、YAML 配置、自动编排（Compose 或自研脚本）、网络抓包（tshark）、日志收集、特征提取（Scapy、NTLFlowLyzer）、多协议支持（HTTP、MySQL、SSH、SMB、MQTT、CoAP、Zenoh、XRCE-DDS）以及 Python/Shell 脚本。

**📊 数据集**

使用自建的 60 条攻击场景、9 种目标服务以及可参数化的 benign traffic 生成器构成的数据集；评估实验主要基于 HTTP SYN Flood DoS 场景。

**📈 对比分析**

通过对攻击强度 L0（低）和 L3（高）分别进行 5 s warmup、10 s attack、5 s cooldown 的实验，测量成功率、p50/p95/p99 延迟等指标；结果显示 L0 维持 100% 成功率、延迟 <10 ms；L3 成功率降至 55.6%，p50≈1275 ms、p95/p99≈2002 ms，攻击结束后即恢复正常，证明了 NetSecBed 在可重复性与可追溯性方面优于传统测试床。

**⚠️ 局限性**

局限性包括：①仅覆盖网络层攻击，未包含物理层或容器外持久化攻击；②实验规模受容器资源限制，难以模拟极高并发或长期持续攻击；③缺乏与工业协议（如 DNP3、IEC 61850 等）的直接映射，导致对 SCADA/ICS 场景的适用性受限；④对多租户或复杂拓扑的验证不足。

---

## 128. GROW: A Conversational AI Coach for Goals, Reflection, Optimism, and Well-Being

**arXiv ID:** 2604.04548 | [PDF](https://arxiv.org/pdf/2604.04548v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 129. How Long short-term memory artificial neural network, synthetic data, and fine-tuning improve the classification of raw EEG data

**arXiv ID:** 2604.04316 | [PDF](https://arxiv.org/pdf/2604.04316v1)

**作者:** Albert Nasybullin `[一作]` (Innopolis University), Semen Kurkin `[通讯]` (Innopolis University)

**通讯引用:** 2389 | [OpenAlex ID](https://openalex.org/A5025595136)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

结合合成数据生成、长短期记忆网络（LSTM）与微调技术，对原始EEG信号进行多类别分类。

**💡 创新点**

利用带通滤波生成多频段合成数据来扩充样本规模，并采用预训练权重微调的方式显著提升分类性能，首次将此方法应用于含模糊视觉刺激的EEG分类。

**🔧 技术方法**

长短期记忆循环神经网络、带通滤波（theta、alpha、beta）、数据预训练与微调、加权平均F1分数评估。

**📊 数据集**

包含4000条EEG记录（20名健康志愿者，Necker立方体视觉刺激），按不同模糊度划分为左、右、高清晰三类，随后通过滤波扩充至12000条样本。

**📈 对比分析**

对原始数据训练100 epoch得到加权F1≈0.63；对扩充数据预训练20 epoch后微调再训练50 epoch，theta频段预训练模型达到最高加权F1≈0.78，明显优于原始模型。

**⚠️ 局限性**

数据量仍有限且样本不平衡，未能深入探究为何theta预训练表现最佳；缺乏对更细粒度（8类模糊度）和特征提取任务的验证，模型泛化能力待进一步评估。

---

## 130. Explainable PQC: A Layered Interpretive Framework for Post-Quantum Cryptographic Security Assumptions

**arXiv ID:** 2604.03665 | [PDF](https://arxiv.org/pdf/2604.03665v1)

**作者:** Daisuke Ishii `[一作]` (Kiara Inc.), Rizwan Jahangir `[通讯]` (NUST Business School)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了“Explainable PQC”解释框架，构建了三层结构（复杂度解释层、几何结构层、经验实验层），并实现了基于Julia的低维格子求解实验平台；同时开展了基于组合Hodge理论的格子几何探索。

**💡 创新点**

创新点在于将复杂度理论、组合几何与实现实验三者统一成可解释的沟通模型，提供了新的表述语言和可视化工具；将组合Hodge理论和局部生成定理引入格子硬度研究；以及公开的Julia实验框架，为实验验证提供了可复现的平台。

**🔧 技术方法**

使用了计算复杂度类（P、BQP、NP）、组合Hodge理论、格子流形（fan）分解、局部生成定理、格子基约简算法（LLL、BKZ、EKZ），以及Julia编程语言和开源软件包。

**📊 数据集**

实验数据基于人工生成的低维格子（维数 10–40），用于测量 LLL（近似）和 EKZ（精确）求解器的计算时间；未使用公开大规模格子数据集。

**📈 对比分析**

采用时间对维数绘图的方式比较 LLL 与 EKZ 的运行性能；结果显示 LLL 维数 40 时仍可完成（约 0.3 秒），而 EKZ 在 40 维时已超过 3600 秒阈值，体现了精确求解器随维数指数级增长的“计算爆炸”。

**⚠️ 局限性**

主要局限：①框架仅为解释工具，不能替代正式安全证明；②未给出具体安全参数或侧信道分析；③组合Hodge理论层尚未与计算复杂度联系；④经验实验仅限低维，无法直接推断到 400–500+ 维的生产级参数；⑤适用于研究与实践沟通而非正式认证。

---

## 131. Effects of Generative AI Errors on User Reliance Across Task Difficulty

**arXiv ID:** 2604.04319 | [PDF](https://arxiv.org/pdf/2604.04319v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 132. No Constant-Cost Protocol for Point--Line Incidence

**arXiv ID:** 2604.03805 | [PDF](https://arxiv.org/pdf/2604.03805v1)

**作者:** Mika Göös `[一作]` (École Polytechnique Fédérale de Lausanne), Anastasia Sofronova `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 154 | [OpenAlex ID](https://openalex.org/A5057666042)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

证明了在随机化通信模型下，判定是否满足方程 y = ax + b 的最优通信复杂度为 Θ(log n)，并证明不存在常数成本的协议。

**💡 创新点**

创新点在于给出了该问题的下界，首次使用了结合数论与傅里叶分析的新型分解引理，将任意函数分解为局部周期结构部分和在典型直线上无偏的伪随机部分，从而得到有效的偏差估计。

**🔧 技术方法**

主要技术包括不等式分解（Decomposition Lemma）、线性格子上的偏差分析、离散傅里叶变换、数论中对质数分布的估计（如Siegel–Walfisz定理）以及不等式（如Young不等式）来控制卷积与L₂范数。

**📊 数据集**

本文不使用任何外部数据集，所有结果均为理论上分析得到的上界与下界。

**📈 对比分析**

通过对比已知的 O(log n) 上界和新的 Ω(log n) 下界，证明了随机化通信复杂度恰好为 Θ(log n)。与之前仅有猜想或弱下界的情况相比，性能（复杂度）达到了最佳匹配。

**⚠️ 局限性**

局限性包括：整数内积问题的下界仅在 k ≤ n^ε 的范围内成立；分解引理中的参数选择较为粗糙，可能不易直接推广到更广泛的函数类；此外，证明高度依赖于特定的数论与傅里叶工具，可能在非离散或高维场景下不适用。

---

## 133. Are Arabic Benchmarks Reliable? QIMMA's Quality-First Approach to LLM Evaluation

**arXiv ID:** 2604.03395 | [PDF](https://arxiv.org/pdf/2604.03395v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 134. SoLA: Leveraging Soft Activation Sparsity and Low-Rank Decomposition for Large Language Model Compression

**arXiv ID:** 2604.03258 | [PDF](https://arxiv.org/pdf/2604.03258v1)

**作者:** Xinhao Huang `[一作]` (Hong Kong University of Science and Technology), Zeyi Wen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1206 | [OpenAlex ID](https://openalex.org/A5013127195)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SoLA，一种无训练的LLM压缩方法；

**💡 创新点**

创新点在于结合软激活稀疏性与低秩分解，并采用自适应组件级秩分配策略；

**🔧 技术方法**

使用了Soft激活稀疏分析、SVD低秩分解以及自适应截断位置分配技术；

**📊 数据集**

评估数据集包括LLaMA‑2系列（7B/13B/70B）、Mistral‑7B以及WikiText2、C4用于校准与评估；

**📈 对比分析**

与LLM‑Pruner、FLAP、SliceGPT、Bolaco、SVD‑LLM等基线在20%–50%压缩率下对比，SoLA在无后训练的前提下实现了显著的困惑度下降（最高可达40%）和下游任务准确率提升3%–10%；

**⚠️ 局限性**

局限性包括对更大规模模型或不同架构（如GQA）的适用性尚未验证，且对校准数据的选择仍有一定敏感性。

---

## 135. MAVEN: A Mesh-Aware Volumetric Encoding Network for Simulating 3D Flexible Deformation

**arXiv ID:** 2604.04474 | [PDF](https://arxiv.org/pdf/2604.04474v1)

**作者:** Zhe Feng `[一作]` (Peking University), Yunhuai Liu `[通讯]` (Peking University)

**通讯引用:** 4775 | [OpenAlex ID](https://openalex.org/A5082653046)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种名为MAVEN的网格感知体积编码网络，用以在稀疏网格上高精度模拟三维弹塑性固体的柔性变形与接触。

**💡 创新点**

创新点：①显式建模网格的三维单元与二维面，利用高维几何特征提升对接触和体积的捕捉；②设计位置感知几何聚合器和两阶段细胞-面消息传递，强化局部几何信息传递；③通过将几何特征嵌入节点，避免传统基于顶点的GNN对稀疏网格的误差。

**🔧 技术方法**

采用Encoder–Processor–Decoder框架，结合多层感知机（MLP）实现几何特征提取与聚合，使用基于位置加权的聚合器、细胞-面二阶消息传递以及几何反散列器；同时利用GNN的自适应边构建实现接触检测。

**📊 数据集**

在三个数据集上评估：公开的弹性变形数据集（DP、CG）以及自建的大变形金属弯曲数据集（MBD），覆盖密集与稀疏网格、弹性与弹塑性情形。

**📈 对比分析**

与多种基线（MGN、Graph Transformer、HCMT、HOOD、FIGNet）相比，MAVEN在所有数据集上均实现平均提升3.41%–18.13%，在稀疏网格和弹塑性任务中优势更为明显，误差下降至1.0%以下，展现出更稳健的接触与物理传播性能。

**⚠️ 局限性**

局限性：对网格质量敏感，极差网格或形变大时可能失效；作为局部算子，尚未支持高效的长程交互；对薄壳、表面或欧拉系统的迁移需要进一步适配。

---

## 136. Learning An Interpretable Risk Scoring System for Maximizing Decision Net Benefit

**arXiv ID:** 2604.04241 | [PDF](https://arxiv.org/pdf/2604.04241v1)

**作者:** Wenhao Chi `[一作]` (Tsinghua University), Ş. İlker Birbil `[通讯]` (University of Amsterdam)

**通讯引用:** 1445 | [OpenAlex ID](https://openalex.org/A5087647617)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种稀疏整数线性风险评分模型 RSS-DNB，直接在多个阈值下最大化净收益，以实现决策效用的优化。

**💡 创新点**

创新点在于将净收益（Decision Curve Analysis）作为学习目标，利用整数线性规划构建可解释的评分系统，并证明净收益优化可同时保证判别力和校准性。

**🔧 技术方法**

采用稀疏整数线性规划、模拟退火启发式优化、决策曲线分析、理论泛化界以及相关的判别/校准理论。

**📊 数据集**

使用八个公开基准数据集和一份真实临床肺腺癌侵袭性评估数据（312例）进行实验。

**📈 对比分析**

与 SLIM、RISKSLIM、Logistic Lasso、决策树等基线模型在 AUC、Hosmer–Lemeshow 校准、AUNBC（净收益曲线面积）和模型稀疏度等指标上进行对比，RSS‑DNB 在 AUNBC 上优于或相当，保持较高的判别力和良好校准，同时仅需少量特征。

**⚠️ 局限性**

局限性包括：无法捕捉复杂非线性关系；整数规划规模随样本和阈值增大而爆炸，求解耗时；模拟退火只能得到近似解；净收益权重方案可能无法完全反映实际阈值分布。

---

## 137. DC-Ada: Reward-Only Decentralized Observation-Interface Adaptation for Heterogeneous Multi-Robot Teams

**arXiv ID:** 2604.03905 | [PDF](https://arxiv.org/pdf/2604.03905v1)

**作者:** Saad Alqithami `[一作]` `[通讯]`, Saad Alqithami

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了 DC‑Ada，一种在多机器人团队部署时仅使用奖励信号、无梯度、低通信的观测接口自适应方法，保持共享策略冻结，调节每台机器人本地的观测变换。

**💡 创新点**

创新点在于：①仅通过奖励反馈在部署时做零阶随机搜索与保守接受/拒绝更新；②使用共用随机数（CRN）和短回合评估来降低噪声；③在保持共享策略不变的前提下实现去梯度、去消息的分布式适配。

**🔧 技术方法**

技术手段包括：残差瓶颈 MLP 观察变换、零阶优化/随机搜索、常数随机数评估、预算化的适配轮次、接受/拒绝门限、分布式执行与最小通信。

**📊 数据集**

使用三种二维多机器人仿真环境（仓储物流、搜救、协同地图），每个环境下四级感知异构（H0–H3）并在 5 个随机种子上评估。

**📈 对比分析**

与共享策略、观测归一化、随机扰动、局部微调四个基线在相同 200,000 步预算下比较；在严重异构下 DC‑Ada 在映射任务中完成率最高，奖励表现与基线相近；在其他任务中表现相当或略低。

**⚠️ 局限性**

局限性包括：仅在低保真二维仿真中验证，缺乏高保真物理/感知误差；依赖固定观测布局；适配回合消耗步骤导致完整回合数减少；对极弱奖励或高度耦合任务适应能力有限；方法只能补偿观测分布，无法生成全新行为。

---

## 138. On Ambiguity: The case of fraction, its meanings and roles

**arXiv ID:** 2604.04647 | [PDF](https://arxiv.org/pdf/2604.04647v1)

**作者:** Jan A Bergstra `[一作]` (University of Amsterdam), John V Tucker `[通讯]` (Swansea University)

**通讯引用:** 3075 | [OpenAlex ID](https://openalex.org/A5103271292)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对算术中“分数”一词的歧义进行深入研究，提出将其拆解为四个层级——fracterm（结构表达式）、fracvalue（数值意义）、fracsign（符号表示）和fracsign occurrence（符号出现实例），并将这四个层级视为一个离散并集（fraxion），从而把“分数”视为一个多义范畴。

**💡 创新点**

创新点在于：① 将词义歧义从结构、数值、符号、出现四个明确概念层级化，提供了细粒度的分解方法；② 引入标签-形状框架，对数系统进行多形态描述，强调了交叉切割（cross‑cutting）概念的应用；③ 通过示例性推理文本展示新概念在消除歧义和保持推理正确性方面的有效性。

**🔧 技术方法**

采用的技术与方法包括：概念分析、形式化定义、符号论证、结构主义视角、交叉切割分类法，以及通过示例（Assertion sequences）验证理论的可行性。

**📊 数据集**

本研究为理论性探讨，无实验数据集；所有示例均为人工构造的文本推理场景。

**📈 对比分析**

通过对比在使用新概念前后的推理过程（如Assertion sequences A、B 等），展示歧义被消除后推理的逻辑完整性得到提升；评估维度主要是解释力、可追溯性和概念层级的清晰度，表现出对算术语义的透明度和一致性显著提升。

**⚠️ 局限性**

局限性包括：① 仍无法给出单一、完备的“分数”概念定义，依赖于多层次范畴；② 交叉切割框架需要先验选择合适的形状，对不同读者的可接受度不一；③ 对教学实践的适用性尚未通过系统实验验证；④ 在复杂算术推理的自动化实现与工具支持方面仍有限。

---

## 139. The Geometric Alignment Tax: Tokenization vs. Continuous Geometry in Scientific Foundation Models

**arXiv ID:** 2604.04155 | [PDF](https://arxiv.org/pdf/2604.04155v1)

**作者:** Prashant C. Raju `[一作]` `[通讯]`, Prashant C. Raju

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了生物与物理基础模型在离散标记化下出现的几何失真问题，提出并量化了“几何对齐税”，通过控制实验和多模型评测验证其存在。

**💡 创新点**

创新点在于：①首次将离散标记化视为对连续几何的内在税收并给出理论与实证边界；②发现并归纳了三种失真模式（局部-全局解耦、表征压缩、几何虚无）；③结合率失真理论与MINE互信息估计，提供对模型几何与信息的统一度量。

**🔧 技术方法**

使用的技术包括：基于Shesha的几何评测（RDM、Procrustes、Lipschitz），VQ双绑实验、交叉熵与连续MSE头的对比消融，率失真框架解析，MINE互信息估计，以及对模型输出的逆互补一致性与对称性测试。

**📊 数据集**

使用的数据集包括：三种合成动力学系统（正弦波、阻尼谐振、Lorenz吸引子）；DNA突变走廊（BRCA1 2kb）、随机与真实DNA序列；蛋白质序列（200aa合成蛋白、UniRef50 10k条）；以及公开的14个生物与物理基础模型（ESM-2、Nucleotide Transformer、Evo 2、Caduceus、OpenFold等）。

**📈 对比分析**

通过对同一架构在连续目标与离散交叉熵、不同参数规模、不同上下文长度、不同token化方式的几何稳定性指标进行对比，发现：连续目标将几何扭曲降低至约1/8–1/9；离散化导致几何失真可达10^3倍；模型规模增大导致几何衰退；RC对称性测试表明模型仅捕捉k‑mer纹理，非真正的物理对称；整体性能显示预测准确性与几何保持不兼容。

**⚠️ 局限性**

局限性包括：仅针对具备严格连续对称的科学序列，天然语言等非物理任务不适用；实验规模至15B参数、1M上下文，无法验证更大规模的潜在缓解；互信息基准仅考虑低阶统计，可能低估高阶结构信息；未探究真正可实现连续与离散协同的架构或训练范式。

---

## 140. Cryptanalysis of the Legendre Pseudorandom Function over Extension Fields

**arXiv ID:** 2604.04833 | [PDF](https://arxiv.org/pdf/2604.04833v1)

**作者:** Daksh Pandey `[一作]` `[通讯]` (Indian Institute of Technology), Daksh Pandey (Indian Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

对单项（d=1）Legendre伪随机函数在扩展域𝔽ₚʳ上的安全性进行系统分析与攻击，提出被动差分签名攻击与主动几何序列表碰撞攻击，证明其在该域上不可行；并指出高阶（d≥2）变体可提升安全性。

**💡 创新点**

创新点：①首次揭示扩展域中多项式计数器的“无进位断裂”对滑窗攻击的影响与其确定性周期；②提出差分签名桶化技术，使被动攻击突破无进位限制；③构造几何序列攻击，利用乘法同态将密钥化简为指数移位，完成O(pʳ/M)级别的表碰撞；④系统评估并阐明单项Legendre PRF在扩展域上的绝对弱点与高阶变体的必要性。

**🔧 技术方法**

采用的技术包括：有限域𝔽ₚʳ的多项式加法与乘法、Legendre符号的广义欧拉判据、差分签名（Differential Signature）桶化、几何序列查询、表碰撞（Table Collision）攻击与哈希映射、以及对攻击复杂度的渐进分析。

**📊 数据集**

使用的数据集：无传统数据集；研究基于纯理论模型与 Python 仿真验证，代码已在 GitHub 上公开。

**📈 对比分析**

比较方法：通过渐进复杂度对比被动攻击 O(U·pʳ/M) 与主动攻击 O(pʳ/M) 与传统基于指数搜索的安全预期。结果表明，单项Legendre PRF在扩展域上无法达到指数级安全，需采用 d≥2 才能恢复安全边界。

**⚠️ 局限性**

局限性：①仅分析了单项（d=1）情况；②被动攻击假设对完整序列可观测，主动攻击假设可执行任意几何查询；③高阶变体的完整安全证明仍缺失，未来工作需探究是否存在针对 d≥2 的高效攻击；④实验验证仅限于 Python 仿真，未在真实 MPC/ZKP 场景中实测。

---

## 141. Do Robots Need Body Language? Comparing Communication Modalities for Legible Motion Intent in Human-Shared Spaces

**arXiv ID:** 2604.03451 | [PDF](https://arxiv.org/pdf/2604.03451v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 142. Firebolt-VL: Efficient Vision-Language Understanding with Cross-Modality Modulation

**arXiv ID:** 2604.04579 | [PDF](https://arxiv.org/pdf/2604.04579v1)

**作者:** Quoc-Huy Trinh `[一作]` (Aalto University), Debesh Jha `[通讯]` (University of South Dakota)

**通讯引用:** 4175 | [OpenAlex ID](https://openalex.org/A5044673103)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种高效的多模态大型语言模型Firebolt-VL，采用Liquid Foundation Model解码器替代Transformer交叉注意力，并提出轻量化的Cross‑Modal Modulator（CMM）实现视觉-文本的高效融合；

**💡 创新点**

创新点包括①将Liquid模型作为解码器，实现线性时间推理；②CMM将状态空间模型与FiLM结合，轻量级token‑grid关联，提升细粒度视觉感知且消除昂贵的交叉注意力；

**🔧 技术方法**

使用了Liquid Foundation Model (LFM2)、SigLIP视觉编码器、S4/S4D状态空间模型、FiLM特征线性调制、轻量化Top‑k视觉网格选择及线性投影；

**📊 数据集**

训练数据集：CC3M、LLaVA‑CoT、MMPR‑v1.2；评测数据集：VQAv2、POPE、AI2D、MMMU、MME、SQA‑Image、MMB；

**📈 对比分析**

与7B+大模型和<3B小模型进行对比，Firebolt‑VL在多项基准上与大模型相当或更优，同时推理吞吐量最高（46.67 tokens/s），延迟最低，展现出优异的速度‑精度平衡；

**⚠️ 局限性**

目前仅支持单图输入，未扩展到多图或视频，CMM尚未针对视频时序视觉信息进行优化；

---

## 143. Injective and pseudo-injective polynomial equations: From permutations to dynamical systems

**arXiv ID:** 2604.04065 | [PDF](https://arxiv.org/pdf/2604.04065v1)

**作者:** Antonio E. Porreca `[一作]` (Aix-Marseille Université), Marius Rolland `[通讯]` (Aix-Marseille Université)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5098672686)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了有限离散动力系统（FDDS）在替代和同步执行的半环运算下的分解计算复杂性，特别是针对形式为P(X) = B的单变量多项式进行分析。

**💡 创新点**

提出了对单射多项式和伪单射多项式的有效算法，并证明了这些方程在特定条件下是可解的，扩展了现有的结果。

**🔧 技术方法**

使用了半环代数结构和多项式方程的解法，结合了图论和树的产品运算。

**📊 数据集**

使用了有限离散动力系统（FDDS）作为数据集，特别是对其周期行为和瞬态行为的分析。

**📈 对比分析**

与现有方法相比，提出的算法在多项式时间内解决了P(X) = B的问题，尤其是在处理编码为二进制的排列时，性能表现良好。

**⚠️ 局限性**

限制在于对于更一般的多项式方程的可解性尚未完全确定，且对多变量情况的研究仍需进一步探索。

---

## 144. Fine-grained Analysis of Stability and Generalization for Stochastic Bilevel Optimization

**arXiv ID:** 2604.04090 | [PDF](https://arxiv.org/pdf/2604.04090v1)

**作者:** Xuelin Zhang `[一作]` (Huazhong Agricultural University), Feng Zheng `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 6025 | [OpenAlex ID](https://openalex.org/A5063285882)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对基于一阶梯度的随机双层优化算法（SSGD 与 TSGD）进行稳定性与泛化误差的系统性分析，提出了 on-average argument stability 与泛化间的定量关系，并给出了在 SC‑SC、C‑C、NC‑NC 三种目标函数条件下的上界。

**💡 创新点**

创新点在于：① 引入 on-average argument stability 代替传统的 uniform stability，能够不需要在每个迭代中重新初始化内层参数；② 在低噪声（small empirical risk）情形下得到更细粒度的泛化上界；③ 将 Hölder 连续性作为更弱的光滑性假设，扩展了理论适用范围。

**🔧 技术方法**

主要技术包括：随机双层优化的算法设计（SSGD 与 TSGD），算法稳定性分析（on-average argument stability），自界定理（self‑bounding property），以及利用低噪声假设与 Hölder 连续性改进泛化上界的理论推导。

**📊 数据集**

实验使用了公开数据集 MNIST（手写数字分类）以及 Omnilot（图像分类）进行超参数优化与数据重权重实验。

**📈 对比分析**

与现有 UD（Unrolled Differentiation）算法在相同任务下对比，实验结果表明：① 充分的验证样本（m₁）和适度的迭代次数（K、T）能显著降低泛化误差；② 论文提出的方法在理论上取得与或优于 UD 的泛化上界，实验验证了 K、T、m₁ 与泛化误差之间的匹配关系。

**⚠️ 局限性**

局限性包括：① 对步长设定存在严格约束（尤其是 SC‑SC 情况下需满足特定范围）；② 主要针对低噪声（empirical risk 接近 0）的场景，未涵盖高噪声或非光滑目标的情况；③ 仅考虑一阶梯度方法，尚未扩展到更高阶或混合时间尺度的算法。

---

## 145. Bounded Autonomy: Controlling LLM Characters in Live Multiplayer Games

**arXiv ID:** 2604.04703 | [PDF](https://arxiv.org/pdf/2604.04703v1)

**作者:** Yunjia Guo `[一作]` (Biibit Ltd), Haixin Qiao `[通讯]` (Kotoko AI)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种名为 "bounded autonomy" 的控制架构，使大型语言模型（LLM）角色在多人实时游戏中能够执行可执行的动作、保持社会连贯性，并且玩家可以在需要时以轻量方式引导角色行为。

**💡 创新点**

创新点包括：① 将控制拆分为 agent‑agent、agent‑world 与 player‑agent 三个接口；② 通过概率回复链衰减限制多体交互中的链式回复；③ 用嵌入式动作归一化与安全回退把 LLM 生成的意图映射到可执行动作；④ Whisper 软引导机制，让玩家在不完全覆盖角色自主性的前提下对下一步动作进行微调。

**🔧 技术方法**

技术细节：使用 Claude‑Sonnet‑4 进行文本生成；利用 Sentence‑Transformer 进行动作嵌入匹配和重复检测；实现 40 秒心跳周期的行为决策；关系偏置的回复焦点仲裁；情绪排除过滤；概率式回复链衰减函数 P_reply(s)=max(0,1−(s−1)·α)；Whisper 通过嵌入相似度和阈值回退实现软引导。

**📊 数据集**

数据集：① 手工构造的 100 条意图探测样本（分为 talk、non‑talk、to‑self 3 个池子），用于评估动作归一化；② 30 条 Whisper 评测样本（20 to‑other + 10 to‑self）；③ 20 次实验室对话场景（开启/关闭回复链衰减），记录链深度与事件分布；④ 真实游戏服务器记录的玩家与角色交互日志（未公开）。

**📈 对比分析**

比较方法：对比开启与关闭回复链衰减的 20 次独立试验；比较动作归一化的 top‑1 / top‑3 准确率；评估 Whisper 的干预一致率（成功 + 部分成功）。结果显示：① 开启衰减后链深度从 10 降至平均 4.4，自动事件比例上升到 77%；② talk 池 top‑1 87%，non‑talk 63%；③ Whisper 成功率 86.7%，尤其是对他人行为的引导几乎无失败。整体证明 bounded autonomy 在保证连贯性、可执行性和可操控性方面是可行的。

**⚠️ 局限性**

局限性：① 回复链衰减采用手工线性调参，缺乏自适应或学习优化；② 40 秒心跳导致快速社交事件的响应延迟；③ 关系偏置回复焦点可能并非最佳仲裁策略；④ Whisper 受 LLM 生成稳定性和 to‑self 动作池覆盖范围限制；⑤ 动作归一化依赖 Sentence‑Transformer，导致同类身体接触动作混淆；⑥ 评测数据集为手工构造，未覆盖真实玩家语言；⑦ Whisper 评估仅由单一评审完成，缺乏多评审一致性验证。

---

## 146. DINO-VO: Learning Where to Focus for Enhanced State Estimation

**arXiv ID:** 2604.04055 | [PDF](https://arxiv.org/pdf/2604.04055v1)

**作者:** Qi Chen `[一作]` (Fudan University), Jian Pu `[通讯]` (Fudan University)

**通讯引用:** 3047 | [OpenAlex ID](https://openalex.org/A5100622420)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种端到端单目视觉里程计系统 DINO-VO，能够在不同场景下自适应选择有效图像补丁并利用深度先验进行状态估计，显著提升跟踪精度和鲁棒性。

**💡 创新点**

创新点在于提出可微分的自适应补丁选择器、融合预训练 Depth Anything v2 的多任务特征提取模块以及利用逆深度先验的稀疏束平差层，使特征学习与状态估计紧密耦合，实现跨数据集的强泛化。

**🔧 技术方法**

技术包括 Vision Transformer（DINOv2）作为特征提取骨干、差分可微分补丁选择与加权、深度先验与先前权重的蒸馏学习、稀疏束平差层与光流/位姿监督损失的联合训练。

**📊 数据集**

在合成 TartanAir、室内 TUM‑RGBD、EuRoC MAV、以及真实户外 KITTI 数据集上进行实验，并将模型仅在合成数据上训练后直接迁移到真实数据。

**📈 对比分析**

与传统 SLAM（SVO、DSO、LDSO、ORB‑SLAM3）以及学习式 SLAM（TartanVO、DROID‑SLAM、DPVO、DPV‑SLAM）对比，DINO‑VO 在 TartanAir、KITTI、TUM‑RGBD、EuRoC 的 ATE RMSE 上均取得领先成绩，并保持实时推理（约 30 FPS）。

**⚠️ 局限性**

局限性包括：对极端光照和动态遮挡的鲁棒性尚待提升；模型对深度先验的依赖使得尺度漂移在某些纯旋转序列中仍可能出现；在极大规模场景下仍需要改进循环闭环与稀疏地图维护机制。

---

## 147. ACES: Who Tests the Tests? Leave-One-Out AUC Consistency for Code Generation

**arXiv ID:** 2604.03922 | [PDF](https://arxiv.org/pdf/2604.03922v1)

**作者:** Hui Sun `[一作]` (Nanjing University), Ming Li `[通讯]` (Nanjing University)

**通讯引用:** 23531 | [OpenAlex ID](https://openalex.org/A5100351402)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种利用留一交叉AUC（LOO-AUC）对LLM生成的测试用例进行加权，从而在不依赖外部监督的情况下提升代码候选的排名；

**💡 创新点**

通过理论证明测试的LOO-AUC与其区分正确代码与错误代码的潜在“判别力”成正比，构造两种权重方案（ACES-C闭式校正和ACES-O可微优化），实现对测试质量的可解释、无监督评估；

**🔧 技术方法**

核心技术包括：离散二值执行结果矩阵、加权投票排名、留一交叉AUC公式、可微化的LOO-AUC目标、梯度优化与闭式校正公式；

**📊 数据集**

在三个公开代码生成基准上评估：HumanEval、HumanEval+、MBPP（约200候选·500测试）；

**📈 对比分析**

与多数仅基于执行的后处理方法（Majority Voting、CodeT、MBR-exec、SRank等）以及静态分析方法（DS^3）比较，ACES在所有基准的Pass@k（k=1,2,5）上均实现最高或接近最高成绩，尤其在错误测试比例高的场景下优势更明显；

**⚠️ 局限性**

局限性包括：需要满足平均判别力正向的假设（虽然大部分任务满足，但极难的任务可能不满足）；对非常少或质量极差的测试集效果有限；仅使用二值执行结果，未利用更丰富的运行信息；

---

## 148. Robust Multi-Source Covid-19 Detection in CT Images

**arXiv ID:** 2604.03320 | [PDF](https://arxiv.org/pdf/2604.03320v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 149. Selecting Decision-Relevant Concepts in Reinforcement Learning

**arXiv ID:** 2604.04808 | [PDF](https://arxiv.org/pdf/2604.04808v1)

**作者:** Naveen Raman `[一作]` (Carnegie Mellon University), Fei Fang `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种自动化的概念选择方法，使强化学习代理在使用可解释的概念决策时不需要人工手工挑选概念。

**💡 创新点**

核心创新在于将概念选择视为状态抽象问题，定义“决策相关概念”并推出DRS算法与理论性能上界，首次实现了无监督概念选择的可解释性与性能保证。

**🔧 技术方法**

主要技术包括状态抽象理论、基于MILP的决策相关概念筛选、DRS与DRS-log两种优化算法、PPO训练和概念预测网络。

**📊 数据集**

实验数据集涵盖常见RL基准（CartPole、MiniGrid、Pong、Boxing、Glucose）以及CUB鸟类图像分类数据集。

**📈 对比分析**

与随机、方差和贪心基线比较，DRS在大多数环境中显著提升奖励并能近似甚至复制人工挑选的概念；在CUB任务中仅需80个概念即可达到与手工选112概念相同的性能。

**⚠️ 局限性**

局限性包括对二进制概念的依赖、对概念预测器完美性的假设、对对抗性噪声鲁棒性不足，以及在不同环境中需调节超参数。

---

## 150. Is your AI Model Accurate Enough? The Difficult Choices Behind Rigorous AI Development and the EU AI Act

**arXiv ID:** 2604.03254 | [PDF](https://arxiv.org/pdf/2604.03254v1)

**作者:** Lucas G. Uberti-Bona Marin `[一作]` (Maastricht University), Konrad Kollnig `[通讯]` (University of Tübingen)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过对欧洲AI法案（EU AI Act）要求的技术与法律双重视角，系统性分析了四项关键技术规范决策（指标选择、指标平衡、指标测量与接受阈值设定），并以高风险的皮肤癌检测系统为案例，说明这些决策如何影响模型的“适当精度”评估与合规性。

**💡 创新点**

创新点在于将“准确性”从单一技术指标转化为包含价值判断与上下文依赖的“技术-规范”选择框架，并将其与AI法案的合规文档义务直接对应，提供跨学科的评估指南。

**🔧 技术方法**

使用的技术包括：法律法规解读、技术标准（如CEN‑CENELEC AI标准草案）、案例研究方法、统计学评估手段（如分层抽样、Bootstrap、合规阈值推导）以及对模型性能指标（Accuracy、Precision、Recall、F1、AUROC 等）的综合讨论。

**📊 数据集**

主要利用公开的皮肤癌检测数据集（如ISIC、DermNet等）作为实验数据，重点关注在类不平衡与误差成本不对称的环境下的性能评估。

**📈 对比分析**

比较方法聚焦于多指标并行评估和阈值设定，指出单一 Accuracy 指标在极端类别不平衡情况下的误导性；通过对 Precision/Recall 进行权重调整或使用 F1、Balanced Accuracy 等聚合指标，可更细粒度地反映模型对不同错误类型的容忍度。整体性能展示了在不同阈值与分层样本下的变化，凸显了技术选择对合规结果的决定性影响。

**⚠️ 局限性**

限制主要包括：① 对于不同领域和使用场景，结论可能不具普适性；② 依赖于案例研究，缺乏大规模实证验证；③ 法规与标准仍在制定阶段，缺乏统一的操作性阈值与指标规范；④ 规范性决策高度主观，评估难以完全客观化。

---

## 151. To Throw a Stone with Six Birds: On Agents and Agenthood

**arXiv ID:** 2604.03239 | [PDF](https://arxiv.org/pdf/2604.03239v1)

**作者:** Ioannis Tsiokos `[一作]` `[通讯]` (Automorph Inc.), Ioannis Tsiokos (Automorph Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文构建了“代理是理论对象”的框架，在一个自定义的有限环世界环境中，用可行性门控、可持续性核、可实现赋能和包装端点映射等指标，系统地证明代理性可以被拆分为启用层和差异化层，并通过一系列消融与基准实验展示六个原语对代理性的影响。

**💡 创新点**

创新点在于：①将代理性重新表述为“理论对象”，将代理的启用与差异化分离；②提出可行性门控、可持续性核、可实现赋能、包装端点映射等可量化的代理度量；③通过零基准（单动作、调度陷阱）验证代理度量的严谨性；④在同一框架下统一展示维护、协议、学习等六鸟原语的作用。

**🔧 技术方法**

使用技术包括：有限状态机与受控核、支持语义下的可持续性迭代（greatest‑fixed‑point）、可行性门控（ledger gating）、可实现赋能（channel capacity/Blahut‑Arimoto）、包装端点映射与 idempotence defect、以及脚本化的可复现性与哈希验证。

**📊 数据集**

数据集为实验室自定义的最小环世界（含噪声、维修、协议、技能等开关），无公开数据集，所有实验均在该可控环境中完成。

**📈 对比分析**

通过匹配控制的消融实验、零基准对照、协议启用对比、技能递增测定等方式评估；指标包括可持续性核大小、可实现赋能比特数、包装缺陷比例。结果表明：维修可将包装缺陷从 1 降至 0；协议在多步时显著提升赋能；技能提高导致赋能显著递增；所有零基准在对应情形下取得 0 或预期值，证明度量的有效性。

**⚠️ 局限性**

局限性包括：仅在离散有限状态机上验证，缺乏对连续物理或大规模系统的直接推广；代理度量高度依赖选定的外部透镜与时序；可实现赋能是开放序列容量的近似，未涵盖风险敏感或目标驱动的代理行为；未考虑多代理、社会规范或信息熵等更深层次因素。

---

## 152. Software Testing Beyond Closed Worlds: Open-World Games as an Extreme Case

**arXiv ID:** 2604.04047 | [PDF](https://arxiv.org/pdf/2604.04047v1)

**作者:** Yusaku Kato `[一作]` (Ritsumeikan University), Katsuro Inoue `[通讯]` (Ritsumeikan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文以开放世界游戏为案例，系统分析了在不确定、非确定性、持续演化环境下，传统闭世界软件测试假设的局限性，并提出了“超越闭世界”测试的愿景与研究方向。

**💡 创新点**

创新点在于：①从开放世界游戏的极端特性出发，总结出四大观察（行为空间不可穷尽、非确定性、边界模糊、测试oracle不稳定）；②将软件测试视角从“彻底验证”转向“行为特征化与不确定性解释”；③提出针对不确定系统的测试目标、自动化测试策略、评估度量和经验研究设计的新框架。

**🔧 技术方法**

主要技术/方法论包括：对现有游戏测试、适应系统与机器学习相关文献的综述与合成；基于观察构建的理论框架；对测试目标、自动化生成、评估指标与实验设计的重新定义；以及对分布式评估、概率oracle与风险阈值等概念的引入。

**📊 数据集**

文中未使用具体数据集；所讨论的案例为开放世界游戏（如《塞尔达传说》《荒野大镖客》等）但并未进行实验或收集数据。

**📈 对比分析**

没有实施实验或算法比较；因此不存在性能评估。本文的贡献主要是理论与方法论上的提出，而非算法实现与量化结果。

**⚠️ 局限性**

局限性包括：①缺乏实证验证，理论框架仍需在真实游戏或其他开放系统上进行实验检验；②对具体技术实现与评估策略的细节阐述有限；③在不同类型的开放系统（如自动驾驶、元宇宙）中的适用性与泛化性尚未充分验证。

---

## 153. ArrowFlow: Hierarchical Machine Learning in the Space of Permutations

**arXiv ID:** 2604.04087 | [PDF](https://arxiv.org/pdf/2604.04087v1)

**作者:** Ozgur Yilmaz `[一作]` (Adana Science and Technology University), Ozgur Yilmaz `[通讯]` (Adana Science and Technology University)

**通讯引用:** 514 | [OpenAlex ID](https://openalex.org/A5047037695)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 ArrowFlow，一个完全在排列空间内进行计算的层级机器学习架构；通过排名过滤器和 Spearman 计步距离实现非梯度学习，并通过多视图投票集成来提升性能。

**💡 创新点**

创新点在于：①将排序（argsort）作为核心编码，将数据转换为离散排列；②使用排列矩阵累积（non‑gradient）更新排名过滤器，等价于 Mallows 模型的最大似然；③通过投票式多视图集成消除排序编码的丢失信息；④从 Arrow 定理解释层级非线性、稀疏性与稳定性的诱导偏置；⑤引入多项式特征扩展与参数化权衡，揭示噪声鲁棒性、隐私保护与缺失特征处理等结构优势。

**🔧 技术方法**

技术包括：Spearman 计步距离、排列矩阵累积更新、argsort 编码、随机投影与多项式扩展、投票式多视图集成、基于社会选择理论的层级设计、量化能耗分析。

**📊 数据集**

使用的数据集包括 UCI 经典分类任务（Iris、Wine、Breast Cancer、Digits 等）、MNIST（PCA 预处理）、TCGA 基因表达癌症分类、Sushi 偏好数据以及人脸/图像分类实验。

**📈 对比分析**

与 GridSearchCV 调参的传统梯度方法（RF、SVM、MLP、KNN、XGBoost 等）以及 kNN 在同一编码空间的对照进行比较。结果显示：在 Iris、Wine、Digits 等数据集上，ArrowFlow 与基线相近或略优；在噪声、隐私、缺失特征、批量效应等鲁棒性场景中，ArrowFlow 显著优于基线；在 MNIST 上约 9.1% 的错误率，证明完全离散比较网络在大规模任务中仍具竞争力；多视图和宽度/深度扩展能进一步降低误差。

**⚠️ 局限性**

局限性包括：①对排序编码的依赖导致信息损失，导致在需要量化特征差异的任务上表现不佳；②多项式扩展提升容量但会放大噪声，需权衡；③当前实现的推理速度与能耗在多视图设置下仍高于单一 MLP；④对非常高维输入的 argsort 需要更大排列空间，可能导致存储和计算瓶颈；⑤对某些自带排序信息的数据（如 Sushi）未能充分发挥优势。

---

## 154. Optimal Circuit Synthesis of Linear Codes for Error Detection and Correction

**arXiv ID:** 2604.03608 | [PDF](https://arxiv.org/pdf/2604.03608v1)

**作者:** Xi Yang `[一作]` (ShanghaiTech University), Zhilin Wu `[通讯]` (Institute of Software Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了一种针对二进制线性码（BLC）的最优码电路合成方法，旨在通过最小化单个输入数量和奇偶校验位大小来生成安全、正确且高效的容错加密电路。

**💡 创新点**

创新点在于：①定义了以单个输入数量和奇偶校验位为双重目标的最优码电路合成问题（P(k,d)）；②基于SMT求解的技术，先枚举输入组合，再通过等价性约简和分区生成快速检验是否存在满足线性、最小距离与注入性的坐标函数；③提出局部最优性保证的合成算法，实现正确-构造和安全-构造的电路；④结合组合树和等价性约简显著减少搜索空间。

**🔧 技术方法**

技术包括：SMT求解（CVC5）、符号逻辑合成（SyGuS）、组合树与等价性约简、Quine-McCluskey求最小SOP、并行生成输入组合（OpenMP）、Yosys后端优化。

**📊 数据集**

实验数据集主要为不同消息长度k∈{1,…,6}和最小距离d∈{2,…,5}的组合，覆盖了推荐的k值和多种最小距离；同时在PRESENT-80加密模块上构造完整的加密电路进行评估。

**📈 对比分析**

与现有的AGEFA_g（贪心）和AGEFA_bf（暴力）比较，作者工具在所有测试点均实现了更少的单个输入、更小的奇偶校验位、更少的门数量和更短的最长路径；在大规模实例（如k=6,d=5）中，能够在数小时内完成，而AGEFA_bf需要数十小时甚至超时。性能提升从约10%到超过80%不等。

**⚠️ 局限性**

局限性包括：算法仅提供局部最优保证，未实现全局最优；搜索时间仍随k和d指数增长，对极大规模问题仍受限；生成的电路不考虑输入信号生成的规模影响，可能导致整体加密电路的面积与延迟并未总是最优；安全性验证仅通过SMT约束保证，未结合完整形式化验证工具。

---

## 155. PointTPA: Dynamic Network Parameter Adaptation for 3D Scene Understanding

**arXiv ID:** 2604.04933 | [PDF](https://arxiv.org/pdf/2604.04933v1)

**作者:** Siyuan Liu `[一作]` (Huazhong University of Science and Technology), Xiang Bai `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种针对场景级点云的测试时参数自适应框架 PointTPA，在推理过程中动态生成输入感知的网络参数。

**💡 创新点**

核心创新在于：①序列化邻域分组（SNG）将无序点云转换为空间连贯的局部补丁；②动态参数投影器（DPP）通过基集和路由系数为每个补丁生成自适应投影权重，使冻结的基础网络能够根据场景特征实时调整；③采用混合插入策略，仅在每个编码器阶段的最后块插入 DPP，保持参数效率。

**🔧 技术方法**

技术细节包括：空间填充曲线（Hilbert、Z‑order）实现序列化；动态参数层（DPLayer）和参数基集实现权重生成；轻量级 MLP 进行路由系数预测；通过逆序列化和线性投影将动态特征融合回原始输入。

**📊 数据集**

主要使用 ScanNet、ScanNet++、S3DIS 这三个室内场景点云数据集进行语义分割评估。

**📈 对比分析**

与多种 PEFT 方法（Adapter、Prefix、LoRA、IDPT、DAPT、PointGST）以及全微调（FFT）对比。无解码器时，PointTPA 在 ScanNet 上 mIoU 达到 78.4%，比 PointGST 高 0.7%；在 S3DIS 上 mAcc 提升 1.4%。加入解码器后，mIoU 接近 FFT，仅落后 0.4%。整体推理速度约为 4 倍快，参数占比仅 1.09%。

**⚠️ 局限性**

局限性包括：①对动态参数的配置仍需经验调优（基数、分组数、路由位置等）导致训练不稳定；②在非常稀疏或极大规模点云场景下，SNG 需要更高分辨率的序列化，可能影响效率；③仅在局部补丁级别自适应，未考虑跨补丁全局上下文的动态调整。

---

## 156. Safety-Aligned 3D Object Detection: Single-Vehicle, Cooperative, and End-to-End Perspectives

**arXiv ID:** 2604.03325 | [PDF](https://arxiv.org/pdf/2604.03325v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 157. EcoAssist: Embedding Sustainability into AI-Assisted Frontend Development

**arXiv ID:** 2604.04332 | [PDF](https://arxiv.org/pdf/2604.04332v1)

**作者:** André Barrocas `[一作]` (University of Lisbon), Nikolas Martelaro `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1397 | [OpenAlex ID](https://openalex.org/A5075763217)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 EcoAssist，一种集成在 IDE 内的能源感知 AI 前端代码助手，能够实时评估 AI 生成的前端代码能耗并给出优化建议。

**💡 创新点**

将能耗分析与优化反馈直接嵌入 AI 编码流程，实现能耗成为即时可见指标，并通过离线训练生成的能耗优化模型提供针对前端的自动化建议。

**🔧 技术方法**

使用 GPT‑4o‑mini 微调模型配合 Powermetrics 与 Playwright 的能耗测量、前端优化工具（如 LightningCSS、Terser、Imagemin、SVGO 等），并在 IDE 中实现差异视图与能耗估算。

**📊 数据集**

以 Kaggle Phishing Website 数据集中的 500 个网站为训练集；评估基准使用 250 个 GPT‑5 生成页面和 250 个真实页面；用户实验中使用个人作品集页面。

**📈 对比分析**

通过对比前后能耗、网络传输和代码体积进行评估；基准实验显示平均能耗下降 13–16%，90% 页面节能；用户实验中平均能耗下降 15.9%，SUS 87.5，NASA‑TLX 工作量低，能耗提升显著且不影响功能。

**⚠️ 局限性**

仅针对单页面前端，缺乏对大型 SPA 或多框架项目的覆盖；能耗测量受限于单一硬件/浏览器；训练数据有限，可能错过新模式；LLM 推理本身能耗需评估更低能耗模型；长时间真实工作场景评估不足。

---

## 158. Benchmarking Multi-turn Medical Diagnosis: Hold, Lure, and Self-Correction

**arXiv ID:** 2604.04325 | [PDF](https://arxiv.org/pdf/2604.04325v1)

**作者:** Jinrui Fang `[一作]` (University of Texas at Austin), Yuji Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了 MINT（Medical Incremental N‑Turn Benchmark），一种高保真多轮医疗诊断基准，利用可控分块、信息保留验证和临床标签构造多轮病例，并在此基准上系统评估多种 LLM 的诊断行为。

**💡 创新点**

创新点包括：①将单轮病例拆分为结构化的临床证据片段，保证信息完整性；②通过控制问诊时机（Q‑First vs Q‑Last）剖析提前答复对性能的影响；③揭示“自我纠错”与“强诱导信息”（如实验室结果）对多轮诊断的决定性作用；④量化提前答复、修正率与错误诱导的统计特征，为改进 LLM 的承诺管理提供可操作的指导。

**🔧 技术方法**

技术手段包括：信息保留验证（FULL vs CONCAT）、多轮分块与固定 turn 数量构造、问答时机控制、答案状态追踪（hold / correct / incorrect）、自我纠错率分析（F2T / T2F）、实验模型多样化（开源、商业、医疗微调 LLM），以及可视化和统计分析。

**📊 数据集**

使用的公开数据集有 MedQA（600例）、MedMCQA（174例）、Derm‑Public（100例）、Derm‑Private（99例）和 MedBullets（62例），共计 1,035 个病例，均按临床标签分片后构成 MINT。

**📈 对比分析**

评估方法：将多轮实验与单轮 FULL 进行对比，并在 Q‑First/Q‑Last、不同 turn 长度（4/8/12/16/10/9）以及实验室结果位置（early/middle/late）等维度进行细粒度比较。结果显示：① Q‑Last 能把多轮准确率恢复至单轮水平（平均提升 20%）；② 提前答复导致 55% 的答案出现在前两轮，误差率高达 60%；③ 自我纠错率可达 10.6×，证明模型具备自我修正潜能；④ 实验室结果在早期出现会诱导 75% 的模型立即答复，显著增加错误。

**⚠️ 局限性**

局限性：① 基准仅基于公开问答数据，缺乏真实临床对话或多模态信息；② 评估集中在英语文本，跨语言泛化未知；③ 未实现模型主动提问或澄清，限制了对真正诊断流程的模拟；④ 结果受模型规模与 API 限制影响，实验可重复性依赖于具体实现细节。

---

## 159. BadgeX: IoT-Enhanced Wearable Analytics Meets LLMs for Collaborative Learning

**arXiv ID:** 2604.04093 | [PDF](https://arxiv.org/pdf/2604.04093v1)

**作者:** Zaibei Li `[一作]` (University of Copenhagen), Daniel Spikol `[通讯]` (University of Copenhagen)

**通讯引用:** 1518 | [OpenAlex ID](https://openalex.org/A5013468201)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

BadgeX系统将轻量化可穿戴IoT设备（如智能徽章或智能手机）与大型语言模型（LLM）结合，实现实时协作学习分析，捕获多模态传感器数据并将其转换为基于学习理论的高层次洞察；

**💡 创新点**

创新点在于①将低功耗、易部署的可穿戴传感器与LLM深度融合，构建端到端实时分析管道；②采用结构化特征与少样本提示，利用LLM的生成推理能力生成叙事式分析，避免传统硬编码规则；③在协作学习场景中首次实现多模态传感器与LLM的协同；

**🔧 技术方法**

技术包括：可穿戴多模态传感器（麦克风、摄像头、IMU、LiDAR、AprilTag），边缘计算的数据管道，音频处理（去噪、Silero VAD、Titanet 说话人识别、WhisperX 转写），视觉分析（目光检测、AprilTag 定位、VLM 场景理解）、空间跟踪（视觉惯性里程计）、LLM（gpt‑4o / gemini‑2.5‑pro‑exp）以及少样本提示与结构化特征编码；

**📊 数据集**

主要数据集为论文作者自行收集的实验数据：一场 43 分钟 STEM 协作任务，2 名参与者佩戴 Arduino‑based 智能徽章，采集音频、视频、IMU、LiDAR 等多模态数据；

**📈 对比分析**

对比方法主要在评估自动化音频与动作识别与人工标注的一致性：语音识别的去激活错误率 17.8%、词错误率 26.4%；动作识别 90% 匹配率；LLM 生成的叙事分析未进行量化评估，但在可解释性和理论一致性方面表现良好；

**⚠️ 局限性**

局限性包括：① Arduino 徽章易因发热或校准漂移导致连接断开；② 指标集仍基于学习理论的先验，可能遗漏部分细节；③实时反馈尚未实现，分析多在会后完成；④ 目前仅在极小规模实验中验证，缺乏大规模用户研究与教育效果评估。

---

## 160. AdaptFuse: Training-Free Sequential Preference Learning via Externalized Bayesian Inference

**arXiv ID:** 2604.03925 | [PDF](https://arxiv.org/pdf/2604.03925v1)

**作者:** Fangzhou Lin `[一作]` (Texas A&M University), Zhengzhong Tu `[通讯]` (Texas A&M University)

**通讯引用:** 2537 | [OpenAlex ID](https://openalex.org/A5015173810)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出AdaptFuse，一种训练免费、冻结LLM的外部化贝叶斯推理框架，用于多轮用户交互中的个性化推荐。

**💡 创新点**

将概率计算完全外部化：符号模块维护离散假设的贝叶斯后验；LLM仅负责语义推理并通过多样采样聚合；采用熵自适应融合，随证据积累自动将权重从LLM转移至符号后验。

**🔧 技术方法**

使用公开LLM、符号贝叶斯更新、Luce选择模型、Dirichlet‑多项式聚合、指数平滑记忆以及熵自适应加权融合。

**📊 数据集**

在三个顺序偏好学习任务上实验：航班推荐、酒店推荐、Web购物；每个任务均使用公开的交互数据集，包含不同属性和开放式商品描述。

**📈 对比分析**

与直接提示、链式推理、Self‑consistency、Oracle Learning、Bayesian Teaching 等基线对比，AdaptFuse在Gemma 2 9B、Llama 3 8B、Qwen 2.5 7B三种模型和所有任务上均超越Fine‑tuned Bayesian Teaching，最终轮准确率提升约2–3个百分点，并且单轮准确率随交互回合单调提升。

**⚠️ 局限性**

需要预先构建离散假设集，若训练集未覆盖真实偏好则性能会下降；依赖N=5 LLM采样导致推理时间比单提示略高；在更复杂或开放域的长文本场景中，符号后验可能收敛慢。

---

## 161. Context-Binding Gaps in Stateful Zero-Knowledge Proximity Proofs: Taxonomy, Separation, and Mitigation

**arXiv ID:** 2604.03900 | [PDF](https://arxiv.org/pdf/2604.03900v1)

**作者:** Yoshiyuki Ootani `[一作]` `[通讯]`, Yoshiyuki Ootani

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在有状态的地理内容系统中使用零知识邻近证明时的上下文绑定漏洞，并提出了将上下文绑定嵌入证明中的 Zairn‑ZKP 方案；

**💡 创新点**

创新点在于将运行时上下文检查迁移到密码学证明中，显著降低操作假设和实现错误，提供完整的漏洞分类、形式化安全模型及对比分析；

**🔧 技术方法**

使用了 Groth16 SNARK、Circom 线路设计、SHA‑256 哈希、nonce、epoch 以及多种客户端/服务器端绑定策略；

**📊 数据集**

实验数据来源于真实城市地点 GPS 测量、POI 密度（东京、纽约、伦敦、柏林）以及五个平台（桌面、浏览器、iOS、Android、M1/M3 设备）的性能测评；

**📈 对比分析**

通过在七种攻击场景下对比七种绑定策略的安全性，并在多平台上测得证明时间 31–83 ms、验证时间 8–12 ms，绑定成本几乎无差异；

**⚠️ 局限性**

局限性包括：仍未解决 GPS 真实性（V4）问题，对高密度部署依赖，需要额外硬件或链上实现以提升可信度。

---

## 162. RAGnaroX: A Secure, Local-Hosted ChatOps Assistant Using Small Language Models

**arXiv ID:** 2604.03291 | [PDF](https://arxiv.org/pdf/2604.03291v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 163. G-EDF-Loc: 3D Continuous Gaussian Distance Field for Robust Gradient-Based 6DoF Localization

**arXiv ID:** 2604.04525 | [PDF](https://arxiv.org/pdf/2604.04525v1)

**作者:** José E. Maese `[一作]` (Universidad Pablo de Olavide), Fernando Caballero `[通讯]` (Universidad Pablo de Olavide)

**通讯引用:** 3774 | [OpenAlex ID](https://openalex.org/A5040477311)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一种基于块稀疏高斯混合模型的连续欧几里得距离场（G-EDF），并构建了利用该表示实现的6-DoF实时定位系统G-EDF-Loc。

**💡 创新点**

创新点在于将连续可微C^1距离场与自适应块稀疏的轴对齐高斯混合相结合，实现了内存高效、解析梯度可用的3D环境表示，克服了传统网格离散化导致的梯度不连续和高内存消耗问题。

**🔧 技术方法**

主要技术包括块稀疏哈希结构、Gaussian Mixture Model、解析梯度计算、Levenberg-Marquardt训练、Ceres求解、Error-State Kalman Filter以及Hermite插值混合等。

**📊 数据集**

使用了Newer College与Snail两大规模户外/室内混合数据集进行评估。

**📈 对比分析**

通过统一的跟踪管线与Fast-GICP、NDT在标准、无IMU、低噪/高噪四种场景下对比实验，G-EDF-Loc在标准条件下与两者相当，且在无IMU及高噪条件下保持低位移/旋转误差、实时性优异，且处理速度显著快于NDT。

**⚠️ 局限性**

仍存在块哈希机制可进一步优化以处理更大规模环境，轴对齐协方差限制了旋转表达，且缺乏GPU加速和多分辨率策略，可能在极复杂曲面下精度受限。

---

## 164. How Well Do Agentic Skills Work in the Wild: Benchmarking LLM Skill Usage in Realistic Settings

**arXiv ID:** 2604.04323 | [PDF](https://arxiv.org/pdf/2604.04323v1)

**作者:** Yujian Liu `[一作]` (University of California Santa Barbara), Shiyu Chang `[通讯]` (University of California Santa Barbara)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估LLM代理在真实环境中使用技能的效果，并研究技能检索与精炼技术以提升任务完成率。

**💡 创新点**

首次在从理想化到现实化的多阶段设置下评估技能效用；提出 agentic 混合检索和查询特定精炼方法，并证明精炼可显著弥补检索不足导致的性能下降。

**🔧 技术方法**

使用 agentic 检索（BM25、语义检索及其混合）、Qwen3-Embedding 进行嵌入；采用 Claude Code、Terminus-2、Qwen-Code 作为模型托管平台；实现查询特定与查询无关的技能精炼流程。

**📊 数据集**

基于 34,198 个开放源代码技能的技能库；使用 SkillsBench 84 个任务和 Terminal‑Bench 2.0 89 个任务进行评测。

**📈 对比分析**

对比六种设置（强制加载、自由加载、加入干扰、检索含 curated、检索不含 curated、无技能），通过 pass‑rate 和技能加载率进行比较；在最苛刻场景下 pass‑rate 下降至约 35%，而精炼后提升 4–8个百分点，显著提升性能。

**⚠️ 局限性**

主要限制在于精炼效果高度依赖检索到的技能质量；缺乏相关技能时精炼几乎无效；离线精炼对模型能力依赖大，模型差异导致技能利用率差异显著；未能充分解决检索与适配的根本瓶颈。

---

## 165. Spectral Path Regression: Directional Chebyshev Harmonics for Interpretable Tabular Learning

**arXiv ID:** 2604.04091 | [PDF](https://arxiv.org/pdf/2604.04091v1)

**作者:** Milo Coombs `[一作]` `[通讯]`, Milo Coombs

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于Chebyshev角坐标的方向谐波（spectral paths）回归模型，并给出闭式岭回归求解方法。

**💡 创新点**

创新点在于把多维Chebyshev多项式从传统的张量积转为按方向索引的谐波基，构造稀疏可解释的方向谱字典；通过贪婪路径选择与闭式求解实现高效训练。

**🔧 技术方法**

使用了Chebyshev多项式角坐标变换、方向谐波基、稀疏贪婪路径选择、闭式岭回归与Gram矩阵流式更新等技术。

**📊 数据集**

实验数据集包括UCI（Concrete Strength、Energy Heating/Cooling、Superconductivity、Wine Quality、Phishing Websites等）、OpenML（Concrete Slump、Yacht Hydrodynamics、Cancer Drug Response、Aquatic Toxicity、Izmir Weather、Ankara Weather等）以及PMLB（Echocardiogram、Wind Speed、CPU Utilisation等）。

**📈 对比分析**

与Ridge、MLP、XGBoost三种基线比较。结果显示，Spectral Path模型在大多数数据集上达到了与XGBoost/MLP相近或略低的R²，且仅需数十到百个路径即可收敛，训练时间更短、模型更稀疏且表现稳定。

**⚠️ 局限性**

局限性：对高度非光滑或强条件结构的数据表现不如树模型；贪婪路径选择无法保证全局最优；当前仅适用于回归任务，未直接扩展到分类、多输出或更复杂的函数；虽然对岭正则化参数不敏感，但路径选择仍依赖验证集。

---

## 166. A Multi-View 3D Telepresence System for XR Robot Teleoperation

**arXiv ID:** 2604.03730 | [PDF](https://arxiv.org/pdf/2604.03730v1)

**作者:** Enes Ulas Dincer `[一作]` (Karlsruhe Institute of Technology), Gerhard Neumann `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 10445 | [OpenAlex ID](https://openalex.org/A5110467801)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种多摄像头融合点云与手腕RGB的VR远程操控系统，能在Meta Quest 3上实时渲染约75k点。

**💡 创新点**

创新点在于将全局三维点云与局部高分辨率手腕RGB结合，实现协同深度感知与细节观察。

**🔧 技术方法**

采用三台Intel RealSense D415 RGB‑D相机、YOLOv11语义分割、GPU加速点云渲染以及Unity+Meta Quest 3实现。

**📊 数据集**

使用实验室自制的多摄像头采集的真实场景数据，未使用公开数据集。

**📈 对比分析**

在31名受试者的三项操控任务中比较四种可视化模式（RGBs、PC、PC+RGB、OT），PC+RGB取得最高成功率、最快完成时间、最低工作量和最佳可用性。

**⚠️ 局限性**

局限在点云分辨率受限、仅使用三台相机、样本群体单一且任务不够复杂，未来需更高精度传感器和多样化用户/任务。

---

## 167. Rethinking Position Embedding as a Context Controller for Multi-Reference and Multi-Shot Video Generation

**arXiv ID:** 2604.03738 | [PDF](https://arxiv.org/pdf/2604.03738v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 168. GROUNDEDKG-RAG: Grounded Knowledge Graph Index for Long-document Question Answering

**arXiv ID:** 2604.04359 | [PDF](https://arxiv.org/pdf/2604.04359v1)

**作者:** Tianyi Zhang `[一作]` (getAbstract), Andreas Marfurt `[通讯]` (Lucerne University of Applied Sciences and Arts)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5061833911)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出GroundedKG-RAG框架，在长文本检索-生成任务中构建以句子为基础、显式地从原文提取的知识图谱，并利用该图谱进行查询适配检索和答案生成。

**💡 创新点**

通过显式将知识图谱节点和边直接来源于原文，实现全过程可验证、无hallucination，并将检索与查询适配结合，摆脱固定树结构的局限。

**🔧 技术方法**

利用语义角色标注(SRL)和抽象语义表示(AMR)构建图谱，节点向量化采用基本/平均/注意力邻居嵌入，检索通过图匹配和向量相似过滤完成。

**📊 数据集**

在NarrativeQA数据集上评估，挑选三本书：Peter Rabbit、Phantom of the Opera、Robinson Crusoe。

**📈 对比分析**

与无上下文、全文上下文以及GraphRAG基线比较，GroundedKG-RAG在BERTScore、ROUGE‑L、Exact/Sequence Match上与GraphRAG相当且在ROUGE‑L上显著优于GraphRAG，且在短文本上超越全文上下文。

**⚠️ 局限性**

节点匹配仍有限，长文本时知识图谱规模大导致检索效果下降；过滤策略未能提升性能；对语义解析的依赖导致解析错误影响最终结果。

---

## 169. AutoReSpec: A Framework for Generating Specification using Large Language Models

**arXiv ID:** 2604.03758 | [PDF](https://arxiv.org/pdf/2604.03758v1)

**作者:** Ragib Shahariar Ayon `[一作]` (Texas State University), Shibbir Ahmed `[通讯]` (Texas State University)

**通讯引用:** 181 | [OpenAlex ID](https://openalex.org/A5103184443)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 AutoReSpec，一种基于大型语言模型的协同框架，用于自动生成可验证的 JML 规范，支持动态模型选择、验证器反馈驱动的对话式迭代以及主/协同 LLM 两阶段生成流程。

**💡 创新点**

创新点在于：①根据程序结构动态推荐 LLM 对（主模型+协同模型）和提示配置；②利用验证器错误信息进行针对性、基于错误类型的提示改进；③主模型失败后切换协同模型，保持对话历史最小化以避免漂移；④构建包含真实 GitHub 代码、SV-COMP 循环案例的综合 benchmark，并公开 Leaderboard。

**🔧 技术方法**

使用技术包括：多种开源 LLM（Llama‑3、Gemma‑3、Phi‑4 等）和专有 LLM（GPT‑4o、Claude‑3.7 Sonnet），OpenJML + Z3 进行规范验证，迭代式对话式 prompt 生成与错误引导、自动 prompt 截断与上下文管理、基于程序 AST 的类型分类与模型推荐。

**📊 数据集**

数据集为 72 个 Java 程序，来源为：SpecGenBench（26）、SV‑COMP（29）以及 17 个从 GitHub OpenJML issue 选取的真实代码；在 RQ1 评估中也使用 SpecGenBench 120 条问题。

**📈 对比分析**

与 SpecGen 与 FormalBench 进行对比，AutoReSpec 在 72 条程序上得到 67/72（94.4%）通过率，成功概率 58.2% 与 Completeness 69.2%，分别超过 SpecGen（58/72、51.4%、60.33%）和 FormalBench（13/72、24.3%、低 Completeness）。平均评估时间比之前方法快 26.9%，验证器调用次数略高但仍在可接受范围内。

**⚠️ 局限性**

局限性包括：LLM 可能误解复杂控制流、遗漏代码或生成不完整规范；OpenJML 与 Z3 可能因复杂证明请求超时或产生灾难性错误；Equivalent Mutant Suppression（EMS）可能漏掉部分等价变异导致 Completeness 轻微高估；实验仅针对 Java/JML，跨语言适用性待验证；数据泄露风险无法完全排除。

---

## 170. Classifying Problem and Solution Framing in Congressional Social Media

**arXiv ID:** 2604.03247 | [PDF](https://arxiv.org/pdf/2604.03247v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 171. Anticipatory Reinforcement Learning: From Generative Path-Laws to Distributional Value Functions

**arXiv ID:** 2604.04662 | [PDF](https://arxiv.org/pdf/2604.04662v1)

**作者:** Daniel Bloch `[一作]` `[通讯]` (University of Paris 6 & VinUniversity), Daniel Bloch (University of Paris 6 & VinUniversity)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在单轨迹条件下，将非马尔可夫决策过程的历史信息通过路径签名（Marcus签名）嵌入到签名增强状态空间，从而实现对未来路径分布的自洽预测与确定性价值评估；

**💡 创新点**

提出了“预期强化学习（ARL）”框架，利用自洽场（SCF）机制生成路径律代理，完成“单次通过”线性价值评估，消除传统蒙特卡洛分支与高方差问题；

**🔧 技术方法**

核心技术包括：路径签名理论与Chen恒等式、控制微分方程（CDE）与跳跃扩散的Marcus解释、神经CDE与自洽场（SCF）联合训练、Nyström压缩签名层、分布式强化学习与分布式回报的Wasserstein收敛性分析；

**📊 数据集**

实验主要基于高频金融跳跃扩散模拟数据（包含结构性断裂与重尾噪声），并通过对比实验验证了在单轨迹环境下的有效性；

**📈 对比分析**

与传统LSTM/Transformer记忆增强和蒙特卡洛树搜索（MCTS）对比，ARL在相同计算预算下实现了更低的价值估计误差、显著降低方差，并在高波动率环境中表现出更好的策略稳定性与风险管理能力；

**⚠️ 局限性**

局限性包括：对签名截断阶数与Nyström基点选择的敏感性、对极端跳跃事件的自洽场收敛速度，以及在大规模多智能体或复杂连续动作空间中的扩展难度。

---

## 172. Schema-Aware Planning and Hybrid Knowledge Toolset for Reliable Knowledge Graph Triple Verification

**arXiv ID:** 2604.04190 | [PDF](https://arxiv.org/pdf/2604.04190v1)

**作者:** Xinyan Ma `[一作]` (Harbin Institute of Technology), Bing Qin `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 16707 | [OpenAlex ID](https://openalex.org/A5017671620)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SHARP 框架，将知识图谱三元组验证转化为动态规划-检索-推理过程；

**💡 创新点**

引入基于模式的规划和记忆增强机制，实现多源信息融合与自适应推理；

**🔧 技术方法**

使用大型语言模型（如 Qwen3-max）与混合知识工具集（KG 结构工具+Wiki/网络检索工具）结合 ReAct 交互；

**📊 数据集**

在 FB15K-237 与 Wikidata5M‑Inductive 两个标准知识图谱验证基准上评估；

**📈 对比分析**

与结构嵌入、PLM、零样本与增量 LLM 方法对比，SHARP 在准确率和 F1 上分别提升 4.2%/12.9% 以上，达到 SOTA；

**⚠️ 局限性**

依赖外部 API 造成检索延迟与成本，且对极稀疏实体仍有限制。

---

## 173. Algebraic Diversity: Group-Theoretic Spectral Estimation from Single Observations

**arXiv ID:** 2604.03634 | [PDF](https://arxiv.org/pdf/2604.03634v1)

**作者:** Mitchell A. Thornton `[一作]` `[通讯]` (IEEE), Mitchell A. Thornton (IEEE)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于代数多样性（Algebraic Diversity）的框架，证明单一观测经过有限群作用可实现与多时间样本均值相同的二阶统计信息估计，并给出通用替换定理、最优性定理以及 PASE（Permutation‑Averaged Spectral Estimation）最优深度。

**💡 创新点**

创新点：
• 将时间平均与群平均等价化，首次用群作用取代多样本求协方差；
• 证明对称群 S_M 是所有有限群中能够得到 Karhunen‑Loève（KL）分解的唯一全局最优群；
• 给出 PASE 最优深度 n = |G| 的定理，消除平均深度参数；
• 形成“盲群匹配”问题并提出基于谱集中度 ψ(G,x) 的单观测群选择方法；
• 在多种应用（MUSIC、Massive‑MIMO、单脉冲波形表征、图信号处理、Transformer 结构分析）中验证框架，并提出非阿贝尔群优势假设。

**🔧 技术方法**

技术方法：
• 群平均估计器 F_G(x) = (1/|G|)∑_g ρ(g)x(ρ(g)x)^H；
• Cayley 余弦图自相关矩阵和其谱分解；
• 代数对称性与噪声可逆性条件；
• 误差度量 δ、δ̃ 与颜色指数 α；
• PASE 最优深度、S_M 子采样实验；
• 盲群匹配的谱集中度 ψ(G,x) 与构造性共轭方法；
• 对多种信号结构（周期、对称、二次相位等）设计相应的群。

**📊 数据集**

数据集与实验：
• 方向估计：ULA（10、20、40天线）单快照；
• Massive‑MIMO：3GPP CDL-A、C‑D、D 三级通道模型，M=16,32,64，K=4，SNR=15 dB；
• 单脉冲 LFM：M=31，μ=0.5，f0=0.15，200 Monte‑Carlo 10 dB、0 dB；
• 图信号：156 个 6‑点无向图，筛选后 7 个候选群；
• Transformer：5 个公开 LLM 的 22,480 个注意力头观测。

**📈 对比分析**

性能对比：
• MUSIC：单快照 CG‑MUSIC 与多快照 MUSIC 在峰值、偏差/方差上几乎相同，CG‑MUSIC 通过 10log10(M) dB 的处理增益；
• Massive‑MIMO：AD（单导频）相比 MMSE 在 M=64 时实现高达 64% 的有效吞吐量提升，LOS‑dominant 频道最优；
• 单脉冲波形：匹配群相较于周期群提升 8.3× 谱集中度，90% 的四类波形分类准确率；
• 非阿贝尔群实验：部分 S_3 自动车自动图滤波器在谱集中度上优于最优周期群；
• Transformer：RoPE 组匹配差异可导致 70–80% head 的低谱集中度，可通过裁剪提升 perplexity。

**⚠️ 局限性**

局限性：
• 需先识别并匹配合适的群 G，群搜索仍是主要计算瓶颈；
• PASE 要求使用所有 |G| 个群元，若 |G| 较大计算量与存储需求高；
• 仅对有限群的结构敏感，对完全无结构或极端非阿贝尔信号匹配仍困难；
• 对噪声的可逆性假设在高噪声或非高斯噪声场景下可能失效；
• 仅在单观测或短快照情况下验证，长时序动态信号的适用性未全面评估。

---

## 174. Paper Espresso: From Paper Overload to Research Insight

**arXiv ID:** 2604.04562 | [PDF](https://arxiv.org/pdf/2604.04562v1)

**作者:** Mingzhe Du `[一作]` (National University of Singapore), See-kiong Ng `[通讯]`

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个开源系统 Paper Espresso，自动从 Hugging Face Daily Papers 社区提取热点 arXiv 论文，利用大语言模型生成结构化摘要、主题标签、关键词，并提供日、周、月等多粒度的趋势分析报告。

**💡 创新点**

创新点在于：①将 LLM 生成的结构化摘要与主题标签公开发布为持续更新的数据集；②通过 LLM 驱动的主题聚合与生命周期分类，实现实时的多粒度趋势监测；③在 35 个月的长周期部署中，系统首次系统性地量化 AI 研究前沿的主题出现、消退速度与创新性与社区关注度的关系。

**🔧 技术方法**

核心技术包括：①Google Gemini / LLM（通过 LiteLLM 接口）用于多模态摘要与主题提取；②两层缓存与 idempotent 处理实现高效批处理；③基于统计指标的 Gartner Hype Cycle 主题生命周期划分；④双语输出（English+Chinese）一次性生成；⑤Streamlit 前端实现交互式浏览。

**📊 数据集**

使用的数据集：Hugging Face Daily Papers API（约 2–3% 的 arXiv 论文）以及公开的四个 Hugging Face 数据集——hf_paper_summary、hf_paper_daily_trending、hf_paper_monthly_trending、hf_paper_lifecycle，总计 13,388 篇论文、6,673 个细粒度主题、51,036 位作者。

**📈 对比分析**

与传统检索/推荐系统对比，Paper Espresso 在 35 个月内处理 13,388 篇热点论文，并生成 6,673 个主题标签，展示了主题出现率高达 19–408/月、熵值约 7.9 bits 的多样性。相比仅靠关键词搜索或社交媒体筛选，系统能够发现低频但高影响的主题，并证明创新组合的论文能获得平均 2.0× 的 upvote 关注度。

**⚠️ 局限性**

局限性包括：①仅关注 2–3% 的社区挑选论文，无法覆盖全部 arXiv 火焰；②依赖社区 upvote 作为关注度 proxy，可能存在主观偏差；③LLM 生成的摘要和主题可能包含误报或不一致；④中文翻译与英文输出共存，可能导致语义细微差别；⑤系统主要聚焦 AI 研究，其他领域的可迁移性尚待验证。

---

## 175. SmartPatchLinker: An Open-Source Tool to Linked Changes Detection for Code Review

**arXiv ID:** 2604.04045 | [PDF](https://arxiv.org/pdf/2604.04045v1)

**作者:** Islem Khemissi `[一作]` (Concordia University), Raula Gaikovina Kula `[通讯]` (University of Osaka)

**通讯引用:** 2517 | [OpenAlex ID](https://openalex.org/A5091820517)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个浏览器插件SmartPatchLinker，能够在代码评审过程中实时识别语义相关的补丁，帮助审阅者发现重复或相似的改动。

**💡 创新点**

创新点在于将语义文本相似度（SBERT）与文件路径、时间等多维特征融合，并在本地运行模型，避免了传统服务器端推理的延迟与隐私泄露问题。

**🔧 技术方法**

使用技术包括Chrome扩展、Python/Flask本地API、Sentence‑BERT（SBERT）进行语义嵌入、随机森林分类器进行融合评分，并通过Docker容器化部署。

**📊 数据集**

实验数据集来自三个大型开源项目：Qt（958条相关补丁）、Android（1,345条）和OpenStack（9,050条），覆盖多种开发生态。

**📈 对比分析**

与Wang等人的三种基线（仅文本、仅文件路径、文本+文件）在Recall@K和MRR指标上对比，SmartPatchLinker在低K值（1、2）下Recall提升显著，MRR亦高于所有基线，表明能够更早、更准确地给出关联补丁。

**⚠️ 局限性**

局限性包括目前仅支持Gerrit平台、候选搜索窗口受时间窗口限制、模型在更大规模或多语言项目中的泛化性待验证，以及未针对GitHub/GitLab等主流平台进行正式适配。

---

## 176. MUXQ: Mixed-to-Uniform Precision MatriX Quantization via Low-Rank Outlier Decomposition

**arXiv ID:** 2604.04701 | [PDF](https://arxiv.org/pdf/2604.04701v1)

**作者:** Seoungsub Lee `[一作]` (Korea University), Seon Wook Kim `[通讯]` (Korea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 MUXQ 的激活量化方法，通过向激活矩阵加入辅助低秩矩阵来重新分配激活中的异常值，从而实现统一的 INT8（或 INT4）量化。

**💡 创新点**

创新点在于：1）不使用混合精度，而是通过低秩辅助矩阵完全在 INT 级别处理异常值；2）该方法兼容现有的量化方案，可与 SmoothQuant 等技术无缝组合；3）实现硬件友好，消除 FP16+INT8 的不规则访问。

**🔧 技术方法**

采用 abs-max 标度、Per‑vector/Per‑tensor 量化粒度、低秩辅助矩阵分解、Fake Quantization 评估，并在 GPT‑2 模型的注意力和 MLP 投影层上应用。

**📊 数据集**

使用 WikiText‑2 作为语言建模评估数据集，并在 GPT‑2 small/medium/large 三种规模上进行实验。

**📈 对比分析**

与 Naïve 量化、LLM.int8() 和 FP16 进行对比。实验显示：MUXQ 在激活位宽降至 6‑7 位时，困境显著下降， perplexity 仅略高于 LLM.int8()，远优于 Naïve 量化；在更低位宽时仍保持稳定，且整体误差低于 Naïve 量化，接近 FP16 的性能。

**⚠️ 局限性**

局限性包括：1）未评估真实推理时的延迟与功耗；2）在极低位宽（≤6 位）下仍不如 LLM.int8()，需进一步优化 exp 因子；3）实现上需额外的矩阵运算和辅助矩阵存储，虽然开销小，但在特定硬件上仍需验证。

---

## 177. A characterization of one-sided error testable graph properties in bounded degeneracy graphs

**arXiv ID:** 2604.04466 | [PDF](https://arxiv.org/pdf/2604.04466v1)

**作者:** Oded Lachish `[一作]` (Birkbeck University of London), Felix Reidl `[通讯]` (Birkbeck University of London)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在随机邻居查询模型下，p‑退化图中一侧误差的属性测试问题，给出了对所有可测试属性的完全结构化表征。

**💡 创新点**

创新点在于确立了测试性与被禁止子图的连通性之间的必然关系，并引入“树形（cactus）”结构与障碍集合的概念，统一描述单一和多重禁止子图的可测试性。

**🔧 技术方法**

主要技术包括：可化简为典型检验器、2‑连通分块与障碍集合分析、角色保持的可观测实例集合构造、依赖有向图的迭代收缩以及cactus嵌入的概率上界证明。

**📊 数据集**

本文为理论性工作，没有使用具体实验数据集，而是通过构造随机图分布与概率论下界来验证结论。

**📈 对比分析**

与之前仅适用于退化图、两侧误差或稠密模型的结果相比，本文提供了完整的结构性可测试性判据，理论证明了在常数查询下可判定所有一侧误差可测试属性；实验性能方面未给出具体数值，仅给出查询复杂度与退化度p的多项式关系。

**⚠️ 局限性**

局限性包括：仅适用于一侧误差、常数退化度p；对多侧误差或更一般图类（如任意稀疏图）仍未给出完整表征；理论证明依赖于复杂的结构构造，实际实现的算法尚未给出。

---

## 178. MVis-Fold: A Three-Dimensional Microvascular Structure Inference Model for Super-Resolution Ultrasound

**arXiv ID:** 2604.04477 | [PDF](https://arxiv.org/pdf/2604.04477v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 179. Batch Loss Score for Dynamic Data Pruning

**arXiv ID:** 2604.04681 | [PDF](https://arxiv.org/pdf/2604.04681v1)

**作者:** Qing Zhou `[一作]` (Northwestern Polytechnical University), Qi Wang `[通讯]` (Northwestern Polytechnical University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Batch Loss Score (BLS)，一种仅基于平均批量损失的样本重要性评分方法，实现动态数据裁剪。

**💡 创新点**

创新点在于利用样本在不同批次中出现的平均批量损失的指数移动平均，理论上等价于对批量噪声进行低通滤波，省去获取逐样本损失的困难。

**🔧 技术方法**

技术包括指数移动平均、信号处理低通滤波理论、与现有基于样本损失的裁剪方法（InfoBatch、SeTa）的无缝集成，以及对α参数的调优。

**📊 数据集**

使用了14个数据集、11个任务、18种模型，涵盖分类（CIFAR、ImageNet）、目标检测/分割（COCO、YOLOv5）、视觉‑语言（COCO、NoCaps、SS1M、ToCa、MSR‑VTT）、场景文字识别（MJ+ST）、图像生成（MNIST、CIFAR10）、多视角立体重建（WHU‑MVS）以及半监督学习（CIFAR100、Yelp Review、ESC‑50）。

**📈 对比分析**

与原始InfoBatch、SeTa以及全数据训练进行比较；BLS‑InfoBatch/SeTa在保持80‑95%准确率/分数的同时，平均可裁剪20%‑50%数据；在多任务、多模型设置中，BLS方法与原方法性能相当或略优，且实现更简单、代码量极少。

**⚠️ 局限性**

局限性包括对α的敏感性、假设批量噪声频谱高于信号的前提可能不在所有任务中成立、对极小批量或高度相关数据时的表现未知，以及对某些复杂损失结构的适用性仍需进一步验证。

---

## 180. Bridging Safety and Security in Complex Systems: A Model-Based Approach with SAFT-GT Toolchain

**arXiv ID:** 2604.04705 | [PDF](https://arxiv.org/pdf/2604.04705v1)

**作者:** Irdin Pekaric `[一作]` (Universität Liechtenstein), Matthias Tichy `[通讯]` (Ulm University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并实现了 SAFT-GT 工具链，实现了攻击-故障树（AFT）的半自动生成，并将其集成到自适应系统（SAS）的 MAPE‑K 循环中，通过 Storm 进行概率模型检查，完成了实验评估和专家用户研究。

**💡 创新点**

创新点包括：① 将安全攻击树（AT）与故障树（FT）自动化融合生成 AFT；② 通过 AFT 片段（fragment）解决不同抽象级别的兼容性；③ 在运行时自动从 ROS2 系统提取数据流和部署信息，动态更新 AFT；④ 将 AFT 转换为动态故障树（DFT）并使用 Storm 计算 MTTF，实现实时风险评估；⑤ 通过用户工作坊验证工具链的实用性与可接受性。

**🔧 技术方法**

所用技术：ROS2 introspection、数据流模型、部署模型、CVE/CPE/EPSS 数据挖掘、AT 与 FT 的文本表示与转换、AFT 片段匹配、DFT 转换、Storm 统计模型检查、MAPE‑K 循环集成、用户调查与讨论。

**📊 数据集**

使用的数据集：本地 NVD（约 272,400 条 CVE 条目）用于 CVE/CPE 匹配；ROS2 四旋翼实验平台（21 节点、43 主题、265 依赖）用于生成模型；手工定义的 FT（共 3 份）与 AFT 片段；公开的 EPSS 评分用于攻击概率估计。

**📈 对比分析**

评估方法与性能：对 3 份 FT（电池、间谍、伤害）分别生成 AFT、DFT 并在 Storm 上进行模型检查；平均总耗时约 860 s，其中 AT 生成占比最高；在单次生成 AFT 时耗时 10–20 s；用户研究通过 Likert 量表与讨论验证工具链在实际场景中的可行性，得到可接受性与相关性的正向反馈。

**⚠️ 局限性**

局限性：① 仍需手工提供部分模型与 AFT 片段；② 依赖公开漏洞数据库，可能漏掉零日与硬件/物理攻击；③ EPSS‑基概率假设为指数分布，未必完全真实；④ 依赖 SSH 进行远程调用，存在效率瓶颈；⑤ 专家样本规模有限，安全背景相对薄弱，实验仅在单一 ROS2 系统上验证；⑥ AFT 与 DFT 的语义统一仍存在争议，无法给出绝对风险预测。

---

## 181. Diffusion Path Alignment for Long-Range Motion Generation and Domain Transitions

**arXiv ID:** 2604.03310 | [PDF](https://arxiv.org/pdf/2604.03310v1)

**作者:** Haichao Wang `[一作]` (Shenzhen International Graduate School), Kyriakos Flouris `[通讯]` (MRC Biostatistics Unit)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了运动扩散路径对齐（M‑DPA）框架，在扩散模型推理时通过优化混合系数实现长范围、跨域运动的平滑过渡。

**💡 创新点**

创新点在于将扩散过程视为随机最优控制问题，构建控制能量目标以显式正则化过渡轨迹，并通过硬拼接约束保证连续性；同时将混合系数当作可优化变量，自动得到最优的指导策略，避免手工调节。

**🔧 技术方法**

利用预训练的 EDGE 扩散模型、DDIM 采样、判别式控制能量函数、分段常数混合调度以及 Adam 在线优化。

**📊 数据集**

在 AIST++（10 类舞蹈）和 HumanML3D（全体动作）两个公开数据集上训练与评估，采用 SMPL 22 关节 132 维表示。

**📈 对比分析**

与线性、Sigmoid、Sine 等固定插值基线对比，M‑DPA 在 FID_k/FID_m 上分别下降 13.79%/7.42%，并在运动动态多样性上略有提升；使用 50 步 DDIM 生成 5 秒长段，拼接后保持较高的姿态和动力学质量。

**⚠️ 局限性**

局限性包括：仅在两类指导（源域与目标域）下验证，未探索更强或多模态指导；优化混合系数增加推理计算开销；对极长序列（>5 秒）或实时应用的鲁棒性尚未评估。

---

## 182. The Infinite-Dimensional Nature of Spectroscopy and Why Models Succeed, Fail, and Mislead

**arXiv ID:** 2604.04717 | [PDF](https://arxiv.org/pdf/2604.04717v1)

**作者:** Umberto Michelucci `[一作]` (Lucerne University of Applied Sciences and Arts), Francesca Venturini `[通讯]` (ZHAW Zurich University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过理论与实验探讨高维光谱数据导致机器学习模型高准确率但无化学意义的现象，并提供验证方法。

**💡 创新点**

将 Feldman–Hájek 定理与测度集中原理应用于光谱分类，系统阐述高维度可导致“完美可分”，并提出全局像素置换、窗口 SHAP 等检验手段。

**🔧 技术方法**

理论分析（Feldman–Hájek、浓度测度）、合成光谱实验、真实荧光橄榄油分类、随机森林、QDA、LDA、kNN、逻辑回归、SHAP 等机器学习技术。

**📊 数据集**

10^3 维合成 Lorentzian 峰光谱与公开的荧光橄榄油光谱数据集（Fluorescence Spectra of Olive Oils）。

**📈 对比分析**

通过不同维度、噪声偏移、全局/独立像素置换、像素数扫描等实验比较模型在高维下的准确率，发现即使无化学区分亦可达到 90%+ 的准确率；低维或破坏统计结构时性能骤降。

**⚠️ 局限性**

结果依赖高维几何，未覆盖所有光谱技术的化学对比差异；对随机森林等高复杂模型更敏感，实际应用仍需结合领域知识验证。

---

## 183. Adaptive Action Chunking at Inference-time for Vision-Language-Action Models

**arXiv ID:** 2604.04161 | [PDF](https://arxiv.org/pdf/2604.04161v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 184. Single-agent vs. Multi-agents for Automated Video Analysis of On-Screen Collaborative Learning Behaviors

**arXiv ID:** 2604.03631 | [PDF](https://arxiv.org/pdf/2604.03631v1)

**作者:** Likai Peng `[一作]` (University of Hong Kong), Shihui Feng `[通讯]` (University of Hong Kong)

**通讯引用:** 655 | [OpenAlex ID](https://openalex.org/A5021896082)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

比较单一 VLM 与多代理 VLM 系统在屏幕录像的自动编码任务中的表现，并提出了两种多代理框架（工作流式和 ReAct式）用于协同学习场景下的画面与动作识别。

**💡 创新点**

1) 提出了两种面向教育视频的多代理架构：一是分场景拆分、交互感知、证据验证的三步工作流；二是基于 ReAct 的循环推理-执行-自校正机制；2) 将 VLM 与传统计算机视觉技术结合，实现对场景和细粒度动作的高效识别；3) 在 ICAP 互动学习理论的指导下，系统化评估屏幕行为的认知参与度。

**🔧 技术方法**

主要使用的技术包括：VLM（Claude‑3.7‑Sonnet、GPT‑4.1、Qwen2.5‑VL‑72B）、计算机视觉算法（关键帧检测、光流/光流差分、光标跟踪）、多代理系统（工作流式 MAS、ReAct‑style MAS）、以及几百秒的屏幕录像子段（20 秒一段，1 FPS 采样）。

**📊 数据集**

使用了来自 27 名本科生参与的 30 分钟协同学习实验的屏幕录像，共 507 个20秒子段；数据按 ICAP 框架标注场景（GAI、Web、Docs）和动作（共 12 种）。

**📈 对比分析**

方法比较：在单 VLM 级别下进行 few‑shot 提示评估；在多代理级别下分别评估工作流式 MAS 与 ReAct‑style MAS。结果表明，三种多代理系统均显著优于单一 VLM，场景识别 F1 由 0.88‑0.94 提升至 0.95‑0.98，Hamming Loss 下降至 0.038‑0.049；动作识别 F1 由 0.68‑0.71 提升至 0.72‑0.79，Hamming Loss 降至 0.26‑0.36。工作流式 MAS 在场景识别上略胜 ReAct‑style MAS，而 ReAct‑style 在动作识别上表现更佳。

**⚠️ 局限性**

主要局限：1) VLM 对提示词高度敏感，简单提示性能低于 50%，表明模型更多基于模式匹配而非真正视觉理解；2) 对细粒度动作（如“阅读+高亮”与“阅读+滚动”）的区分仍易出现混淆或幻觉，导致标签误报；3) 当前实验规模有限，仅在单一实验室协作任务上验证，缺乏跨平台或大规模数据的泛化评估；4) 多代理系统的模块化收益尚未通过消融实验深入拆解。

---

## 185. Think in Strokes, Not Pixels: Process-Driven Image Generation via Interleaved Reasoning

**arXiv ID:** 2604.04746 | [PDF](https://arxiv.org/pdf/2604.04746v1)

**作者:** Lei Zhang `[一作]` (Meta Superintelligence Labs), Zecheng He `[通讯]` (Meta Superintelligence Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本工作提出一种过程驱动的交叉推理框架，让统一的多模态模型通过规划、草图、检查与修正四个阶段逐步构建图像，避免单次生成的盲目性；

**💡 创新点**

创新点在于：①利用场景图子采样生成无矛盾的增量指令；②通过自采样的批评轨迹让模型学习自身错误；③在同一模型中端到端地交替生成文本与视觉 token，实现真正的交叉推理；

**🔧 技术方法**

主要技术包括：BAGEL 7B统一多模态模型、autoregressive文本‑图像 token 生成、Rectified Flow 图像解码、场景图抽象、GPT‑4o 生成的自评与指令衍生；

**📊 数据集**

使用自制的三部分过程交互数据集（多步生成子集、指令冲突集、图像‑指令对齐集）以及公开的 GenEval 与 WISE 评测基准；

**📈 对比分析**

与单通道和统一模型基线对比，GenEval 上性能从 0.79 提升至 0.83，WISE 从 0.70 提升至 0.76；与 PARM 等基线相比，训练样本从 688K 降至 62K，推理步数从 1000 降至 131，显著提升效率；

**⚠️ 局限性**

局限性包括：仍依赖场景图的抽象能力，难以处理高度动态或长时序的场景；推理过程仍相对耗时；缺乏对视频或 3D 空间的扩展。

---

## 186. Motion-Adaptive Multi-Scale Temporal Modelling with Skeleton-Constrained Spatial Graphs for Efficient 3D Human Pose Estimation

**arXiv ID:** 2604.03652 | [PDF](https://arxiv.org/pdf/2604.03652v1)

**作者:** Ruochen Li `[一作]` (Durham University), Amir Atapour-Abarghouei `[通讯]` (Durham University)

**通讯引用:** 1862 | [OpenAlex ID](https://openalex.org/A5013030358)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了一个名为MASC-Pose的框架，用于从单目视频中高效推断3D人体姿态。

**💡 创新点**

核心创新是自适应多尺度时序建模（AMTM）与骨架约束自适应GCN（SAGCN），两者共同实现运动自适应的时空关系捕捉，并在保持高精度的同时显著降低计算量。

**🔧 技术方法**

采用稀疏时序图卷积、MoE式权重选择、骨架拓扑自适应图卷积以及残差/层级融合等技术，避免了全局自注意力的高开销。

**📊 数据集**

在Human3.6M和MPI-INF-3DHP两个公开基准上进行评估。

**📈 对比分析**

与PoseFormer、TCPFormer、MixSTE等先进Transformer/GCN混合模型对比，MASC-Pose在MPJPE/P-MPJPE等指标上取得最优或接近最优的结果，同时MACs/帧和模型参数显著更小，证明了更高的效率与精度兼备。

**⚠️ 局限性**

局限性包括：仅使用非重叠的多尺度窗口，缺乏可变长度或重叠分区；对跨数据集或真实环境的泛化性能仍需进一步验证；训练仍需在单张H200 GPU上长时间完成。

---

## 187. CoLA: Cross-Modal Low-rank Adaptation for Multimodal Downstream Tasks

**arXiv ID:** 2604.03314 | [PDF](https://arxiv.org/pdf/2604.03314v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 188. Veo-Act: How Far Can Frontier Video Models Advance Generalizable Robot Manipulation?

**arXiv ID:** 2604.04502 | [PDF](https://arxiv.org/pdf/2604.04502v1)

**作者:** Zhongru Zhang `[一作]` (Tsinghua University), Jianyu Chen `[通讯]` (Tsinghua University)

**通讯引用:** 5530 | [OpenAlex ID](https://openalex.org/A5100611364)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Veo-Act 层次化框架，将前沿视频生成模型 Veo-3 作为高层运动规划器，结合多头逆动力学模型（IDM）和 Vision‑Language‑Action（VLA）低层执行器，实现零演示的通用机器人操纵。

**💡 创新点**

创新点在于：①首次利用强大的视频生成模型进行视觉规划并直接映射到动作；②设计多头 IDM 同时预测动作与交互检测门，支持高低层无缝切换；③通过层次化调度显著提升了在复杂场景下的指令遵循与任务成功率。

**🔧 技术方法**

核心技术包括：Veo‑3 视频生成模型、DINOv3 视觉编码器、基于随机演练的多头逆动力学网络、动作平滑器、门控切换策略，以及现有 VLA 基线 π_0.5 作为低层策略。

**📊 数据集**

使用自建随机演练数据集（300k 采样的仿真帧对与 150k 真实数据）并加入 100k 纯随机运动样本；所有数据通过 STEM‑OB 观测噪声增强，以提升跨域泛化。

**📈 对比分析**

与基线 π_0.5 以及 VPP 进行对比，指标为指令遵循成功率与整体任务成功率；在模拟和真实机器人三组受扰情形下，Veo‑Act 的整体成功率从 50% 提升至 80%（加权平均提升 78.4%），指令遵循成功率提升约 60% 以上。

**⚠️ 局限性**

局限性：对视频生成模型的精度与稳定性高度依赖；当视觉预测误差或场景变化较大时，逆动力学规划可能失效；低层执行器仍需精细控制，当前框架在接触密集的操作阶段仍无法完全可靠。

---

## 189. V-Reflection: Transforming MLLMs from Passive Observers to Active Interrogators

**arXiv ID:** 2604.03307 | [PDF](https://arxiv.org/pdf/2604.03307v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 190. Learning from Imperfect Demonstrations via Temporal Behavior Tree-Guided Trajectory Repair

**arXiv ID:** 2604.04225 | [PDF](https://arxiv.org/pdf/2604.04225v1)

**作者:** Aniruddh G. Puranic `[一作]` (University of Maryland), Calin Belta `[通讯]` (University of Maryland)

**通讯引用:** 12010 | [OpenAlex ID](https://openalex.org/A5086742095)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个将Temporal Behavior Tree（TBT）轨迹修复与强化学习相结合的框架，用于从不完美演示中学习机器人控制策略。

**💡 创新点**

创新点在于利用TBT的形式化约束对示范轨迹进行最小化修改并从修复轨迹中提取与机器人运动学无关的潜在奖励函数，既保持了演示意图，又实现了安全性与任务一致性。

**🔧 技术方法**

使用了TBT修复算法（基于MILP/LP与landmark）、Signal Temporal Logic的鲁棒性语义、潜在奖励塑形、KD-Tree加速最近邻、PPO强化学习等技术。

**📊 数据集**

数据集包括网格世界的50条示范轨迹以及Safety-Gymnasium中的单机与多机到达-避障任务的50条示范，环境由随机障碍与目标构成。

**📈 对比分析**

与基于稀疏奖励PPO、专家设计密集奖励PPO及Safe-PPO对比，修复+潜在奖励的方案收敛更快、成功率更高、碰撞成本明显降低。

**⚠️ 局限性**

局限性包括依赖近似运动学模型、离线修复不适用于实时/无模型场景、潜在奖励对随机障碍配置的泛化可能受限。

---

## 191. AvatarPointillist: AutoRegressive 4D Gaussian Avatarization

**arXiv ID:** 2604.04787 | [PDF](https://arxiv.org/pdf/2604.04787v1)

**作者:** Hongyu Liu `[一作]` (Hong Kong University of Science and Technology), Qifeng Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在单张人像照片的基础上，使用自回归Transformer先逐点生成3D高斯点云，再通过高斯解码器预测渲染属性，从而一次性生成可动画的4D高质量头像；

**💡 创新点**

核心创新是让Transformer自回归地直接生成可变数量、可自适应密度的高斯点云，并在生成时同时预测绑定信息，实现从无固定模板到动态、精细的几何建模；

**🔧 技术方法**

技术组合包括：decoder‑only Transformer（含交叉注意、前馈网络）、DINOv2图像特征提取、Pixel3DMM/FLAME点云特征、3D Gaussian Splatting、向量量化+位置编码、线性混合蒙皮（LBS）动画；

**📊 数据集**

主要使用NeRSemble数据集（419身份），对每个身份通过GaussianAvatars拟合得到训练样本；实验集为25个身份，剩余用于训练；

**📈 对比分析**

与AvatarArtist、Portrait4Dv2（NeRF）、LAM、GAGAvatar（Gaussian Splatting）等最新方法在NeRSemble上对比；评估指标为LPIPS、FID、AKD、APD、CLIPScore；AvatarPointillist在所有指标上均优于基线，尤其在身份保持与表达/姿态一致性方面表现突出；

**⚠️ 局限性**

局限性包括：自回归点云生成序列长导致推理时间相对较慢，模型对极端姿态或非人类主体的适应性未完全验证，且训练和推理均需要大量GPU资源；

---

## 192. OASIC: Occlusion-Agnostic and Severity-Informed Classification

**arXiv ID:** 2604.04012 | [PDF](https://arxiv.org/pdf/2604.04012v1)

**作者:** Kay Gijzen `[一作]` (Leiden University), Daniël M. Pelt `[通讯]` (Leiden University)

**通讯引用:** 1639 | [OpenAlex ID](https://openalex.org/A5037323866)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对严重遮挡下的细粒度分类任务，提出了一种基于像素级遮挡异常检测的遮挡无关分割方法，并结合灰色掩模抑制遮挡纹理干扰，同时通过估计遮挡严重度动态选择对应训练好的分类模型，从而实现对遮挡程度自适应的分类；

**💡 创新点**

创新点包括①使用AnomalyDINO等异常检测模型实现遮挡无关的像素级分割；②基于像素级遮挡置信度直接估计遮挡严重度；③构建多模型池并按估计严重度动态挑选最佳模型，形成“遮挡无关+严重度知情”分类框架；

**🔧 技术方法**

技术手段包括：AnomalyDINO异常检测生成像素级遮挡概率图；Otsu自适应阈值分割遮挡区域；灰色掩模将遮挡像素置为中灰；ViT‑DINOv2 backbone + MLP head；通过合成遮挡（Perlin噪声、SAM提取叶子/烟雾/碎石纹理）进行数据增强；多模型微调与严重度驱动的模型选择；

**📊 数据集**

使用Stanford Cars细粒度分类数据集，结合人工合成的灰色遮挡、纹理遮挡（叶子、烟雾、碎石）进行实验；

**📈 对比分析**

与标准训练、对灰色遮挡微调、纹理遮挡微调、OVSeg+遮挡分割等方法对比。OASIC在遮挡程度曲线下的AUC_occ显著提升，+18.5相较于仅对灰色遮挡微调，+23.7相较于无遮挡训练；实验表明遮挡无关分割+灰色掩模+严重度模型选择是最优组合；

**⚠️ 局限性**

主要限制包括：需要预训练并维护多套模型，推理时需额外计算像素级遮挡分割；实验仅在合成遮挡环境下验证，真实场景中遮挡种类和纹理多样性可能进一步挑战方法；严重度估计基于像素平均，易受噪声或遮挡分布不均影响；未在实时或跨域任务中评估。

---

## 193. A reconfigurable smart camera implementation for jet flames characterization based on an optimized segmentation model

**arXiv ID:** 2604.03267 | [PDF](https://arxiv.org/pdf/2604.03267v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 194. Android Instrumentation Testing in Continuous Integration: Practices, Patterns, and Performance

**arXiv ID:** 2604.03438 | [PDF](https://arxiv.org/pdf/2604.03438v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 195. DAG Covers: The Steiner Point Effect

**arXiv ID:** 2604.04186 | [PDF](https://arxiv.org/pdf/2604.04186v1)

**作者:** Sujoy Bhore `[一作]` (Indian Institute of Technology Bombay), Da Wei Zheng `[通讯]` (Institute of Science and Technology Austria)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文首次提出并研究了 Steiner DAG 覆盖问题，并给出了针对平面有向图和树宽图的两 DAG 的近似或精确距离保持构造。

**💡 创新点**

创新点在于允许添加 Steiner 顶点，从而在只使用 2 个 DAG 的情况下实现 1（或 1+ε）伸缩，且额外边数近似线性；并且证明了在不使用 Steiner 顶点时，树宽为 1 的图需要 Ω(log n) 个 DAG，显示两种情况的根本差异。

**🔧 技术方法**

主要技术包括：使用树宽分离包的递归分解、构造按顶点顺序的 Dominating DAG、利用 Thorup 的最短路分离子路（shortest‑path separators）和层次化路径分解，结合 Steiner 顶点来实现距离的多步估计。

**📊 数据集**

没有使用实验数据集；研究全部为理论分析和构造证明。

**📈 对比分析**

通过与已有的非 Steiner DAG 覆盖（如 Filtser 论文）的理论上限比较，本文在树宽图上从 O(log n) 个 DAG 降至 2 个，平面图上从 O(log n) 个 DAG 降至 2 个且伸缩仅为 1+ε，额外边数保持在 O(n·ε^{-1}·log²n·logΦ)。

**⚠️ 局限性**

局限性包括：对一般有向图的 Steiner/非 Steiner 覆盖仍无最优下界；平面图结果仍受 aspect ratio 依赖；不确定是否存在线性额外边数的精确平面 DAG 覆盖；并且目前仅给出理论构造，缺乏实验验证。

---

## 196. Can Humans Tell? A Dual-Axis Study of Human Perception of LLM-Generated News

**arXiv ID:** 2604.03755 | [PDF](https://arxiv.org/pdf/2604.03755v1)

**作者:** Alexander Loth `[一作]` (Frankfurt University of Applied Sciences), Marc-Oliver Pahl `[通讯]` (IMT Atlantique)

**通讯引用:** 900 | [OpenAlex ID](https://openalex.org/A5004198506)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究人类是否能够区分由大型语言模型（LLM）生成与人工撰写的新闻文本，采用双轴连续评分平台收集多模型的判断数据。

**💡 创新点**

创新点在于将真伪与来源解耦的双轴连续评分法、构建可复现的多模型生成框架（loth2026collateraleffectsloth2026eroding）以及开源的评估平台（loth2026eroding）。

**🔧 技术方法**

使用了Web-based评估平台、Welch's t检验、ANOVA、Pearson相关分析、K-means聚类以及滚动窗口准确率分析等统计与机器学习技术。

**📊 数据集**

数据集包括六大LLM（GPT‑4、GPT‑3.5、GPT‑4o、LLaMA‑2 13B、Gemma 7B、Mistral 7B）生成的新闻片段、人类来源的新闻片段，以及实验记录，全部存档于Zenodo。

**📈 对比分析**

通过对人类源判断分数进行t检验和ANOVA，发现人类无法显著区分机器与人工文本；模型间无显著差异；专业知识与判断准确度相关显著；政治倾向无显著影响；滚动窗口显示超过30条后准确率下降，体现认知疲劳。

**⚠️ 局限性**

局限性包括：样本主要为受过高等教育的欧洲受访者，刺激集高度偏向机器生成，缺乏政治平衡，政治倾向测量单一且缺乏因果性，实验环境不具备真实社交媒体情境，因而普适性受限。

---

## 197. Can LLMs Learn to Reason Robustly under Noisy Supervision?

**arXiv ID:** 2604.03993 | [PDF](https://arxiv.org/pdf/2604.03993v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 198. RELIEF: Turning Missing Modalities into Training Acceleration for Federated Learning on Heterogeneous IoT Edge

**arXiv ID:** 2604.04243 | [PDF](https://arxiv.org/pdf/2604.04243v1)

**作者:** Beining Wu `[一作]` (South Dakota State University), Jun Huang `[通讯]` (South Dakota State University)

**通讯引用:** 5256 | [OpenAlex ID](https://openalex.org/A5020146420)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

针对 IoT 边缘设备联邦学习中系统、模态和数据异构耦合问题，提出了一种统一的 FL 框架，能够在缺失模态的设备上实现加速训练和更高效通信。

**💡 创新点**

创新点在于：①将融合层 LoRA 投影矩阵按模态对齐分块；②对每个块仅在拥有该模态的设备间进行聚合，消除跨模态梯度干扰；③利用每块内的梯度分歧度量为每台设备动态分配训练预算，优先更新分歧度最高的块；④上述三步共享同一结构界面，实现聚合、分配、通信的无缝统一。

**🔧 技术方法**

核心技术包括：低秩自适应（LoRA）、模态分块聚合、基于分歧的弹性训练、指数移动平均（EMA）估计分歧、聚合误差理论分析以及联邦学习框架（Flower）实现。

**📊 数据集**

实验使用两个公共多模态健康监测数据集：PAMAP2（加速度、陀螺仪、磁力计、心率）和 MHEALTH（加速度、陀螺仪、磁力计、ECG）。

**📈 对比分析**

与 FedAvg、FedProx、FedEL、FedICU、DarkDistill、Harmony、Pilot、FedSA‑LoRA、HeLoRA、FedLEASE 等 10 种基线对比，结果显示在 CNN（全参数）模式下可达 2.87× 的时间加速、37% 的能耗降低；在 LoRA（参数高效）模式下可达 9.41× 的时间加速、45% 的通信量节省，并在稀缺模态上提升 15.3 个 F1 分数。

**⚠️ 局限性**

主要局限包括：①在预训练冻结的编码器场景下，前向传播的固定成本限制了加速幅度；②实验在模拟 FLOP‑比例模型下评估，真实硬件上仍存在加速与能耗差距；③方法对极端模态缺失（几乎无模态）设备的效果尚未充分验证；④需要在更大规模、多厂商设备上进一步验证鲁棒性。

---

## 199. The Energy Cost of Execution-Idle in GPU Clusters

**arXiv ID:** 2604.04745 | [PDF](https://arxiv.org/pdf/2604.04745v1)

**作者:** Yiran Lei `[一作]`, Daniel Vosler `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过在学术AI集群和行业服务重放中对 GPU 的功耗与利用率进行细粒度被动监测，发现并量化了“执行‑空闲”（execution‑idle）状态在多种工作负载（训练、批量推理、在线推理）中的普遍性与能耗贡献。

**💡 创新点**

创新点在于首次将执行‑空闲定义为“程序已加载但可见活跃度极低”的状态，并系统性评估其在不同 GPU 代际、工作负载类别和行业场景下的能耗比例，同时提出针对该状态的频率下调与负载均衡两种原型控制方案。

**🔧 技术方法**

使用技术包括 1 Hz NVML/DCGM/OS 计数器被动采样、基于阈值的状态划分、HDBSCAN 聚类分析前置系统行为、以及基于 SM/内存时钟的自定义频率控制器。

**📊 数据集**

数据集涵盖 31 天（2026 年 2 月 4日–3 月 7日）学术集群 756 张 NVIDIA GPU 的 162 GB 监测日志，补充了从 OpenAI、Qwen 与 Azure 的公开请求日志进行的 L40S GPU 5 分钟服务重放。

**📈 对比分析**

比较方法：按“在作业执行期间”取代深度空闲，将执行‑空闲时段的能耗占比与活跃执行时段进行对比；在实验中对比平衡调度、负载失衡与频率下调三种策略，结果显示能耗可下降 10–35%，但 95% 延迟上升可达 29–160%。

**⚠️ 局限性**

局限性包括 1 Hz 采样率难以捕捉子秒级短暂停顿；缺乏统一的 GPU 组件级功耗分解；数据仅来自学术集群与有限的行业重放，无法完全代表生产环境；控制实验使用的下调阈值与时钟策略过于简单，未探索更细粒度的自适应方案。

---

## 200. String Representation in Suffixient Set Size Space

**arXiv ID:** 2604.04377 | [PDF](https://arxiv.org/pdf/2604.04377v1)

**作者:** Hiroki Shibata `[一作]` (Kyushu University), Hideo Bannai `[通讯]` (Institute of Integrated Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

证明了重复性测度 χ 是可达的，并给出了基于子串等式系统（SES）的字符串表示方案，可在 O(χ) 个机器词中编码任意字符串。

**💡 创新点**

①提出子串等式系统这一新的可表示框架；②构造了利用最小后缀集的逆向压缩 trie，证明任意字符串都有 O(χ) 大小的 SES；③通过此方法完成了 χ 的可达性证明，破除此前未解的猜想。

**🔧 技术方法**

子串等式系统、等价关系构造、最小后缀集、超最大右扩展、逆向压缩 trie、深度优先遍历及 LCA 产生子串等式约束等理论技术。

**📊 数据集**

本文为理论工作，无实验数据集，所有结论基于抽象字符串分析。

**📈 对比分析**

通过理论比较与已知的 BMS、字母吸引子等重复度测度进行对比，证明 s(w)≤O(χ(w)) 且 χ 与其他测度（如 r、z、v）具有相对紧密的上界，未给出具体实验性能指标。

**⚠️ 局限性**

①未证明 SES 与 BMS 之间是否存在 s(w)∈o(b(w)) 的具体实例；②未给出实现或实验验证；③仅证明了上界，缺乏下界与最优性分析。

---

## 201. Commercial Persuasion in AI-Mediated Conversations

**arXiv ID:** 2604.04263 | [PDF](https://arxiv.org/pdf/2604.04263v1)

**作者:** Francesco Salvi `[一作]` (Princeton University), Manoel Horta Ribeiro `[通讯]` (Princeton University)

**通讯引用:** 1346 | [OpenAlex ID](https://openalex.org/A5011195481)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在两项预注册实验（共2012名参与者）中，研究者比较了传统搜索引擎与基于五种前沿大语言模型（GPT‑5.2、Claude Opus 4.5、Gemini 3 Pro、DeepSeek v3.2、Qwen3 235b）的会话式 AI 购物代理对消费者选择赞助产品的影响，并评估消费者对广告的识别与价值判断；

**💡 创新点**

首次系统量化展示了会话式 AI 在大规模真实商品目录中对消费者决策的隐蔽说服效应，并揭示了传统透明标注与隐蔽说服之间的显著差异，证明了模型说服能力的普遍性与“弱透明”难以遏制的风险；

**🔧 技术方法**

采用前沿大语言模型驱动的聊天代理，结合向量检索技术展示书籍；实验设计包含5个对照组（SP、CP、CPer、CPer‑Exp、CPer‑Sbt），并使用 OLS+HC3 估计、最大‑t 复合检验、以及多重中介分析（策略与 LIWC‑22）评估说服机制；

**📊 数据集**

使用从 Goodreads 抓取的 5,495 本 e‑book 目录（含书名、作者、类别、描述、评分、价格），随机将五分之一设为赞助品；

**📈 对比分析**

在对照组中，传统搜索（SP）赞助选择率为22.4%；聊天式放置（CP）为26.8%；开放式说服（CPer）达61.2%（≈3倍提升），透明标注（CPer‑Exp）略降至55.5%（不显著），隐蔽说服（CPer‑Sbt）降至40.7%（显著降低）。销量率在所有组间无显著差异，提示说服不牺牲消费者满意度；偏见检测率在开放式说服下仅为17.9%，透明标注下为35.1%，隐蔽说服下为9.5%。

**⚠️ 局限性**

实验仅涵盖单一低风险商品（e‑book），未涉及高价值或高风险产品；仅进行一次购物会话，未考察长期互动与信任演化；赞助品随机分配，未模拟精准个性化投放；实验环境相对简化，缺乏真实支付与购买动机。

---

## 202. Diagonal-Tiled Mixed-Precision Attention for Efficient Low-Bit MXFP Inference

**arXiv ID:** 2604.03950 | [PDF](https://arxiv.org/pdf/2604.03950v1)

**作者:** Yifu Ding `[一作]` (Beihang University), Jinyang Guo `[通讯]` (Beihang University)

**通讯引用:** 911 | [OpenAlex ID](https://openalex.org/A5039703932)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Diagonal‑Tiled Mixed‑Precision Attention（DMA）低位混合精度注意力核，利用 MXFP 格式在 GPU 上加速 LLM 推理；

**💡 创新点**

创新点包括：①沿注意力矩阵对角线保留高精度区块，其他区块使用低精度 MXFP 以平衡准确性与速度；②将量化、编码、打包、尺度转换等低位预处理步骤完全融合到单一 Triton 核中，显著减少内存访问和核启动开销；③支持 MXFP8/4 与 NVFP4 的混合使用，兼顾硬件兼容性与精度；

**🔧 技术方法**

采用 Triton GPU 核实现，使用 MXFP 微尺度浮点格式、在线 Softmax、对角线分块策略，以及融合量化/编码/打包/尺度转换技术；

**📊 数据集**

在 LongBench 长上下文基准上对 LLaMA‑3.1‑8B 与 LLaMA‑3.2‑3B 进行评估，并在 NVIDIA B200 GPU 上测试；

**📈 对比分析**

与原生 SDPA BF16 内核以及固定格式 MXFP4、NVFP4、MXFP8 进行对比。DMA 在 128×128 块尺寸下总延迟 7.776 ms，分别比 MXFP4、NVFP4、MXFP8 低约 40–60 %，在 LongBench 上平均得分提升 1–2 分，保持或略高于全精度基线；

**⚠️ 局限性**

仅在文本 LLM、有限模型规模和长上下文场景验证，未覆盖极长序列、不同硬件、视觉/视觉‑语言任务或其他注意力变体；混合精度策略在更大规模或不同模型上的鲁棒性尚待进一步验证。

---

## 203. Pedagogical Safety in Educational Reinforcement Learning: Formalizing and Detecting Reward Hacking in AI Tutoring Systems

**arXiv ID:** 2604.04237 | [PDF](https://arxiv.org/pdf/2604.04237v1)

**作者:** Oluseyi Olukola `[一作]` (University of Southern Mississippi), Nick Rahimi `[通讯]` (University of Southern Mississippi)

**通讯引用:** 263 | [OpenAlex ID](https://openalex.org/A5102764912)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在仿真智能辅导环境中提出并验证了四层教学安全框架（结构、安全进度、行为和对齐安全）及其指标RHSI，用以检测强化学习代理的奖励劫持问题。

**💡 创新点**

创新点在于首次将教学安全形式化为可量化的约束集合，并引入Reward Hacking Severity Index 作为衡量代理在高代理奖励与实际学习收益之间失衡程度的连续指标。

**🔧 技术方法**

使用的技术包括基于知识图谱的强化学习建模、受限马尔可夫决策过程（C1）、滑动窗口进度与行为约束（C2、C3）、累计奖励比率约束（C4）以及后置约束检测与统计评估。

**📊 数据集**

数据集为基于Python编程知识图谱的模拟学习者，覆盖三类学习者（困惑、平均、先进）共120个学习会话（18,000条交互）。

**📈 对比分析**

对比方法：将四种奖励/约束配置（EO、MAS、MO、ST）及其消融版本进行统计检验；结果显示全约束ST在RHSI、行为违规率和学习进度上显著优于其他方案，约束消除后性能下降明显。

**⚠️ 局限性**

局限性包括：仅在模拟环境中验证，未包含真实学生行为与长时学习轨迹；使用的BKT模型和认知负荷量表为近似，约束阈值需域特定校准；实验规模有限，未探究多维约束组合及不同奖励权重组合的影响。

---

## 204. Beamforming Feedback as a Novel Attack Surface for Wi-Fi Physical-Layer Security

**arXiv ID:** 2604.04179 | [PDF](https://arxiv.org/pdf/2604.04179v1)

**作者:** Jingzhe Zhang `[一作]` (University of South Florida), Yili Ren `[通讯]` (University of South Florida)

**通讯引用:** 1072 | [OpenAlex ID](https://openalex.org/A5010708869)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出 BFIAttack，一种利用 Wi‑Fi 反馈信息（BFI）重构信道状态信息（CSI）并攻击基于物理层安全的应用（设备身份验证、用户身份验证和密钥生成）的攻击框架。

**💡 创新点**

创新点包括：①在单天线 STA 场景下给出 CSI 的闭式重构公式；②在多天线 STA 场景下设计基于最大似然估计的 CSI 重构方法；③提出理论–实践双重约束过滤不可信重构结果；④利用天线对间的空间相似性进行线性回归精细化 CSI，以提高攻击成功率。

**🔧 技术方法**

技术手段包括：BFS 中的奇异值分解（SVD）与角度提取、最大似然估计（MLE）及坐标下降搜索、理论约束的三角不等式分析、实践约束的时延匹配、以及线性回归模型的空间相似性提升。

**📊 数据集**

实验数据来源为作者自行采集：18 台 AP、5 名参与者、共 108k 包（设备/密钥）、450k 包（用户步态），覆盖实验室、住宅和户外三种环境，使用 ASUS RT‑AX86U、Linksys AC3000、TP‑Link AXE5400 路由器和 Intel AX210 电脑，采用 20/40/80 MHz 带宽。

**📈 对比分析**

与基准方法（DomPathCon 和随机攻击）对比，BFIAttack 在多天线场景下 5 次攻击尝试的平均攻击成功率（ASR）为 73%（单天线为 93%），DomPathCon 仅 2.9%/4.1%，随机攻击接近 0%；在不同天线数、带宽、设备、距离、环境、人数、NLoS、包速率、模型和时间维度上均保持 70–95% 的高 ASR，证明方法鲁棒性强。

**⚠️ 局限性**

局限性包括：①单天线设备极易被一次性重构 CSI，易被攻击；②多天线场景仍需多次尝试；③攻击成功受 BFI 质量、环境稳定性和天线间距离影响；④仅使用 CSI 幅度，若目标系统利用相位信息则抵御力提升；⑤在需要大量包聚合的用户验证场景中，攻击成功率显著下降。

---

## 205. Optimal Contest Beyond Convexity

**arXiv ID:** 2604.04844 | [PDF](https://arxiv.org/pdf/2604.04844v1)

**作者:** Negin Golrezaei `[一作]` (Massachusetts Institute of Technology), Suho Shin `[通讯]` (University of Maryland)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在多方参与者的竞赛（如内容生产者激励）中，给定非凸目标函数和任意成本曲线，设计最优的基于排名的奖赏分配机制；

**💡 创新点**

创新点在于：①证明在一般非凸目标下最优机制仍呈现高度结构化的形态（最高奖最高，最低奖为零，其他等值）；②揭示在单一目标下出现的阶段性转变；③通过总正性、变差消减与Schur-凸性等高级数学工具，将高维非凸优化降维为一维，获得多项式时间近似算法；

**🔧 技术方法**

主要技术包括：变差消减（variation diminishing property）与总正性（total positivity）理论、Bernstein基多项式的矩阵表示、Schur-凸/凹性判定、KKT条件分析、分支限界（branch‑and‑bound）求解单变量优化；

**📊 数据集**

该工作为理论分析，没有使用具体数据集；

**📈 对比分析**

方法通过数学证明与算法复杂度分析验证其可行性；实验（仿真）展示在不同α、β和n值下，所得到的机制在目标函数值上优于传统的两种极端策略；

**⚠️ 局限性**

局限性包括：仅在完全信息、同质参与者的模型下得到结果；对更复杂梯度结构的目标函数（多重符号变化）未覆盖；对不完全信息或异质参与者的推广仍待研究。

---

## 206. CREBench: Evaluating Large Language Models in Cryptographic Binary Reverse Engineering

**arXiv ID:** 2604.03750 | [PDF](https://arxiv.org/pdf/2604.03750v1)

**作者:** Baicheng Chen `[一作]` (Shanghai Qi Zhi Institute), Tianxing He `[通讯]` (Tsinghua University)

**通讯引用:** 1114 | [OpenAlex ID](https://openalex.org/A5051747323)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CREBench基准，用于评估大语言模型在加密二进制逆向工程中的能力

**💡 创新点**

创新点在于构建432个多难度挑战、四级评估框架，并对LLM与人类专家进行系统对比

**🔧 技术方法**

采用ReAct风格LLM代理、沙盒执行环境、静态/动态分析工具和代码生成技术

**📊 数据集**

数据集基于48种标准加密算法，结合3种密钥使用场景和3个编译/混淆难度，共432个挑战

**📈 对比分析**

通过pass@3和四项子任务得分与人类专家基准对比，最佳模型GPT‑5.4平均得分64分、flag恢复率59%，人类专家得分92分

**⚠️ 局限性**

局限包括未覆盖专业混淆工具、LLM在动态分析上的死锁问题、算法识别的原型偏差，以及安全拒绝机制的不完善

---

## 207. Seeking Socially Responsible Consumers: Exploring the Intention-"Search"-Behaviour Gap

**arXiv ID:** 2604.03694 | [PDF](https://arxiv.org/pdf/2604.03694v1)

**作者:** Leif Azzopardi `[一作]` (University of Strathclyde), Frans van de Sluis `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对286名消费者进行线上调查，探究他们在购买过程中对社会责任因素的意图与实际搜索行为之间的差距。

**💡 创新点**

创新点在于将意图-行为差距从信息检索的角度重新定位，系统性评估消费者在搜索社会责任信息时遇到的挑战，并用 Brehm 任务动机模型量化这些障碍。

**🔧 技术方法**

采用问卷设计（Likert量表+开放式问题）、Brehm 模型维度（成功预期、感知价值、努力、感知难度）以及 R 语言进行描述性与推断性统计分析。

**📊 数据集**

数据来源于 Prolific 招募的286名在过去三个月完成线上购物的受访者，包含对购买决策中 13 个维度（价格、质量、企业声誉、社会责任等）的评估与搜索行为。

**📈 对比分析**

方法上通过比较意图重要性、考虑与搜索比例，并计算相关系数与 R²；对环境与社会责任主题的 R² 达 26.11%，显示意图与行为存在明显衰减，表明信息检索难度对购买决策有显著影响。

**⚠️ 局限性**

局限性包括受访者回忆偏差、社会期望偏差、部分量表内部一致性低（如努力量表 α=0.38）以及样本主要来自西方发达国家，难以完全推广至全球消费者。

---

## 208. XAttnRes: Cross-Stage Attention Residuals for Medical Image Segmentation

**arXiv ID:** 2604.03297 | [PDF](https://arxiv.org/pdf/2604.03297v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 209. BiST: A Gold Standard Bangla-English Bilingual Corpus for Sentence Structure and Tense Classification with Inter-Annotator Agreement

**arXiv ID:** 2604.04708 | [PDF](https://arxiv.org/pdf/2604.04708v1)

**作者:** Abdullah Al Shafi `[一作]`, Shoumik Barman Polok `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 BiST 语料库，构建了 30,534 条英孟双语句子，并通过多阶段人工标注（投票+共识）完成句子结构和时态的双维度标注。

**💡 创新点**

创新点包括：①首个同时标注句子结构与时态的双语语料；②系统化的多阶段标注框架与高一致性评估；③公开完整的标注指南与工具；④使用双编码器实现语言特定表征，提升分类效果。

**🔧 技术方法**

采用多阶段人工标注与 Fleiss κ 检验保证标注质量，使用多语言编码器（mBERT、XLM‑R）与双编码器（BERT/DistilBERT + BanglaBERT/BanglishBERT）进行基线实验，并使用 SMOTE 处理类别不平衡。

**📊 数据集**

数据来源于维基百科（英孟两语）与日常对话文本，最终构成 30,534 条句子，其中英 17,465 条，孟 13,069 条。

**📈 对比分析**

通过准确率、F1 与 AUC 三项指标对多语言编码器与双编码器进行比较，双编码器在结构与时态分类上均提升约 2–5%（最高可达 0.92 AUC），XLM‑R 在多语言编码器中表现最佳。

**⚠️ 局限性**

局限性在于类别严重不平衡（简单句与现在时占比过高）、复杂/复合句子标注仍具挑战，且模型对低资源语言的泛化能力有限，未来需探索更有效的数据增强与跨语言迁移方法。

---

## 210. Estimating Central, Peripheral, and Temporal Visual Contributions to Human Decision Making in Atari Games

**arXiv ID:** 2604.04439 | [PDF](https://arxiv.org/pdf/2604.04439v1)

**作者:** Henrik Krauss `[一作]` (University of Tokyo), Takehisa Yairi `[通讯]` (University of Tokyo)

**通讯引用:** 3391 | [OpenAlex ID](https://openalex.org/A5012762510)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了人类在Atari游戏中的视觉信息来源对决策的贡献，使用对比消融框架评估周边视野、注视信息和过去状态信息的影响。

**💡 创新点**

提出了一种可逆工程的受控消融方法，能够量化不同视觉信息源对行动预测的相对贡献，并将聚类分析与单子对象实验相结合，揭示了粗略的决策行为模式。

**🔧 技术方法**

采用卷积神经网络进行行为模仿学习，结合注视图（多尺度高斯卷积）、周边信息抑制和时间窗口的过去状态编码，并使用K‑means与t‑SNE进行聚类分析。

**📊 数据集**

使用Atari‑HEAD数据集，该数据集包含20款Atari游戏的同步游戏录像、动作标注和眼动跟踪数据。

**📈 对比分析**

通过比较六种信息组合（全信息、去除周边、去除注视、去除过去等）在20款游戏中的动作预测准确率，发现去除周边信息导致准确率下降约35%–44%，去除注视信息下降约2%–3%，去除过去状态信息下降1%–15%，验证了周边信息贡献最大。

**⚠️ 局限性**

局限性包括：仅使用四名参与者的数据，难以深入探究个体差异；信息泄漏问题导致各源贡献难以完全分离；未对仅移除注视区的配置进行评估，可能进一步阐明周边信息的独立作用。

---

## 211. Decentralized Ergodic Coverage Control in Unknown Time-Varying Environments

**arXiv ID:** 2604.04280 | [PDF](https://arxiv.org/pdf/2604.04280v1)

**作者:** Maria G. Mendoza `[一作]` (University of California), Shankar Sastry `[通讯]` (University of California)

**通讯引用:** 55442 | [OpenAlex ID](https://openalex.org/A5062722286)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种去中心化多无人机覆盖框架，利用在线高斯过程贝叶斯推断和快速混合的可逆马尔可夫链，实现对未知、时变信息分布的自适应监测。

**💡 创新点**

创新点在于：①将快速可逆马尔可夫链（REMC）与基于信息量的UCB贝叶斯估计结合，实时更新目标分布；②在完全局部通信和感知受限的条件下，通过分布式贝叶斯共享实现全局收敛；③通过子线性遗憾分析证明能在时变环境中跟踪目标分布。

**🔧 技术方法**

主要技术包括：高斯过程（GP）用于空间信息的贝叶斯估计；上置信界UCB策略平衡探索与利用；快速可逆马尔可夫链（REMC）用于生成可收敛的随机转移策略；离散图模型和马尔可夫链理论。

**📊 数据集**

使用仿真数据集：二维网格化灾害场景，包含高信息区域（人群或火势）和禁飞区，规模从5×5到10×10，团队规模1~10，通信半径从局部到全局。

**📈 对比分析**

与基准算法MAC‑DT（基于贪婪GP规划）进行比较。实验表明，在覆盖完整度、首次达标区时间、长期遗憾和贝叶斯误差等指标上，本文方法均优于MAC‑DT；在动态环境中，遗憾随时间下降，体现更好的适应性。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，缺乏真实部署；需要手工设定更新频率（τ_GP、τ_P）且对环境变化速率敏感；计算量随区域数和团队规模增加显著，可能限制实时性。

---

## 212. Evaluating Artificial Intelligence Through a Christian Understanding of Human Flourishing

**arXiv ID:** 2604.03356 | [PDF](https://arxiv.org/pdf/2604.03356v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 213. Activity-Dependent Plasticity in Morphogenetically-Grown Recurrent Networks

**arXiv ID:** 2604.03386 | [PDF](https://arxiv.org/pdf/2604.03386v1)

**作者:** Sergii Medvid `[一作]` (National University of Kyiv-Mohyla Academy), Mykola Glybovets `[通讯]` (National University of Kyiv-Mohyla Academy)

**通讯引用:** 41 | [OpenAlex ID](https://openalex.org/A5069329970)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对 MorphoNAS 生成的自组织递归网络在 CartPole 和 Acrobot 控制任务中进行大规模可塑性（Hebbian 与 anti‑Hebbian）评估，并通过共进化实验验证可塑性参数的可演化性，同时使用随机‑RNN 对照以分离通用与发展特定效应。

**💡 创新点**

①发现 anti‑Hebbian 可塑性在大多数性能层级上显著优于 Hebbian；②可塑性在动态环境中能实现真实的在生命周期内适应；③共进化能独立重现上述模式；④通过随机‑RNN 对照明确区分了通用与发展特定的影响。

**🔧 技术方法**

使用 MorphoNAS 形态发生编码、Hebbian/anti‑Hebbian 权重更新、遗传算法共进化、随机‑RNN 控制、CartPole‑v1 与 Acrobot‑v1 任务、统计检验（Cohen's d、回报回归、regret 等）。

**📊 数据集**

数据集主要包括 50,000 条随机 MorphoNAS 基因产生的 CartPole 控制器（5M+ 评估）和 5,000 条 Acrobot 控制器，分别在 OpenAI Gymnasium 的 CartPole‑v1 与 Acrobot‑v1 环境中评估。

**📈 对比分析**

通过对比基线（无可塑性）、oracle‑tuned 可塑性、固定参数可塑性以及共进化条件，评估平均奖励提升、改进比例和 regret。结果显示 anti‑Hebbian 在中高性能层级能提升约 60–90% 的奖励；但共进化并未提升最终性能，只能在发展网络中重现 anti‑Hebbian 的优势；随机‑RNN 对照表明 anti‑Hebbian 的优势是通用的，而发展网络的 regret 更高，表明需进行网络特定调优。

**⚠️ 局限性**

仅限经典控制任务，未验证在连续动作或高维任务中的表现；共进化未能提升最终性能；随机‑RNN 样本规模有限且权重分布与 MorphoNAS 有差异；仅使用简单的 Hebbian/anti‑Hebbian 规则，缺乏更复杂可塑性机制的探索。

---

## 214. Is a Picture Worth a Thousand Words? Adaptive Multimodal Fact-Checking with Visual Evidence Necessity

**arXiv ID:** 2604.04692 | [PDF](https://arxiv.org/pdf/2604.04692v1)

**作者:** Jaeyoon Jung `[一作]` (Soongsil University), Kunwoo Park `[通讯]` (Soongsil University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个两代理协作的多模态事实核查框架，能够根据视觉证据是否必要来动态决定是否使用图像信息，提升核查准确率。

**💡 创新点**

创新点在于首次将视觉证据必要性判断作为独立任务交给“分析器”完成，并将其自然语言评估融入“验证器”推理过程，从而实现对视觉信息的自适应利用。

**🔧 技术方法**

技术上结合了大规模预训练视觉‑语言模型（Llama‑3.2‑Vision 作为分析器，Qwen2‑VL 作为验证器）以及 QLoRA 微调、Retriever（CLIP + SBERT）和自定义提示工程。

**📊 数据集**

使用了公开多模态核查数据集 MOCHEG、FIN‑FACT 以及作者自行构建的 WebFC（包含 621 条 2024‑2025 年实时网页检索的声明与判决）进行训练与评测。

**📈 对比分析**

通过与四个基线（MOCHEG、LVLM4FV、HGTMFC、MetaSumPerceiver）在 gold 与 retrieved 两种证据配置下进行对比，AMuFC 在 gold 上达到 0.612 的准确率、0.600 的 macro‑F1，retrieved 上 0.546 与 0.540，均显著优于基线，并在跨域测试集 FIN‑FACT 与 WebFC 上仍保持相对优势。

**⚠️ 局限性**

主要局限包括仅限英语实验、检索器固定为 CLIP+SBERT 可能限制泛化、以及两代理协作导致的额外计算开销。

---

## 215. Trace-Guided Synthesis of Effectful Test Generators

**arXiv ID:** 2604.04345 | [PDF](https://arxiv.org/pdf/2604.04345v1)

**作者:** Zhe Zhou `[一作]` (Purdue University), Suresh Jagannathan `[通讯]` (Purdue University)

**通讯引用:** 5927 | [OpenAlex ID](https://openalex.org/A5034957233)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文提出一种基于低估型类型规范的测试生成器合成方法，用符号轨迹捕捉效果操作的潜在数据和控制依赖，从而生成针对黑盒系统的高效测试序列。

**💡 创新点**

创新点在于将低估（underapproximation）思路引入表达式类型与效应系统，并利用符号轨迹作为约束，直接驱动测试生成器的合成，突破传统基于安全性证明的限制。

**🔧 技术方法**

核心技术包括：①低估型类型语言与符号轨迹表示；②基于约束求解的轨迹驱动合成算法；④与属性化测试框架（QCheck）和模型检查工具（P）集成的生成器框架。

**📊 数据集**

实验使用多种实际应用程序（包含数据库接口、网络协议栈、并发数据结构等），以及公共的属性测试与模型检查基准集；具体数据集未在摘要中列出，但覆盖范围广泛。

**📈 对比分析**

与默认测试策略（如随机测试）和手工编写的专家生成器对比，合成生成器在覆盖率与缺陷发现率上均显著提升，且在性能上可与最先进的手写方案相媲美。

**⚠️ 局限性**

局限性包括：①对复杂效应模型的支持尚有限，难以处理高度动态的并发与异步交互；②合成过程对符号求解器的依赖使得规模化应用时可能产生性能瓶颈；③目前仅在特定语言环境（如OCaml）验证，跨语言迁移仍需进一步研究。

---

## 216. The Last APK: Retiring Android SDK Development for Institutional Software Using Python-Django, HTMX, and a WebView Bridge

**arXiv ID:** 2604.03808 | [PDF](https://arxiv.org/pdf/2604.03808v1)

**作者:** Rahul Patel `[一作]` `[通讯]` (Indian Institute of Information Technology), Rahul Patel (Indian Institute of Information Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

开发并部署了 IIT Gandhinagar 的校园管理系统，覆盖任务调度、库存管理、园艺跟踪、考勤与请假等功能。

**💡 创新点**

证明 Android SDK 对机构应用已过时，提出 HTMX+Django web‑first 与 WebView APK 组合实现无 Android SDK 代码，显著降低开发时间与网络负载。

**🔧 技术方法**

使用 Python‑Django、HTMX、PostgreSQL、Redis、Docker Compose、Kotlin 极简 WebView 包装以及浏览器 MediaDevices API。

**📊 数据集**

基于 IIT Gandhinagar 部署的 42 名员工使用日志、HTTP 请求日志与系统运行日志进行评估。

**📈 对比分析**

与估算的原生 Android 开发在开发周期、HTTP payload、延迟、用户满意度进行对比；结果为 HTMX+Django 开发时间缩短 54%，payload 降低 91%，延迟从 520 ms 降至 130 ms，SUS 得分 4.2/5。

**⚠️ 局限性**

局限在于缺乏完整离线功能；性能与用户体验评估基于估算而非实际并行实现；对网络可靠性和硬件依赖敏感，适用范围有限。

---

## 217. LangFIR: Discovering Sparse Language-Specific Features from Monolingual Data for Language Steering

**arXiv ID:** 2604.03532 | [PDF](https://arxiv.org/pdf/2604.03532v1)

**作者:** Sing Hieng Wong `[一作]` (University of Kentucky), A. B. Siddique `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用随机 token 序列过滤，仅凭少量单语数据识别出极稀疏的语言特定稀疏自编码器（SAE）特征，并将其作为残差流 steering 向量，实现多语生成控制。

**💡 创新点**

创新点在于：①只需单语数据即可完成语言特征识别；②通过随机 token 过滤剔除语言无关特征，得到因果重要且可解释的语言特征；③将这些稀疏特征直接用于 steering，大幅提升控制效果。

**🔧 技术方法**

技术方法包括：随机 token 序列生成、残差流提取、稀疏自编码器（SAE）编码与解码、特征频率阈值过滤、方向消融验证以及基于解码向量的残差流 steering。

**📊 数据集**

使用的单语数据集为 FLORES+、WikiMatrix 和 Tatoeba，覆盖 12 种语言（如英语、葡萄牙语、德语、中文等），并从中抽取少量句子作为训练与验证。

**📈 对比分析**

通过与 DiffMean、PCA、Zhong（Monolingual/Parallel）等七种基线对比，LangFIR 在 Gemma 3 1B、Gemma 3 4B 和 Llama 3.1 8B 三个模型上均获得最高的 ACC×BLEU 分数，平均提升幅度可达 2.7 点以上，甚至在 Gemma 3 1B 上比最强单语基线提升 4.7 倍。

**⚠️ 局限性**

局限性包括：①对预训练 SAE 质量与层次的依赖，若 SAE 训练不足或架构不合适，识别效果会下降；②在英语等模型偏向语言上，特征稀疏且影响不显著；③目前仅在中等规模模型上验证，尚未探究对更大模型或不同 SAE 变体的适用性。

---

## 218. Hierarchical Awareness Adapters with Hybrid Pyramid Feature Fusion for Dense Depth Prediction

**arXiv ID:** 2604.03339 | [PDF](https://arxiv.org/pdf/2604.03339v1)

**作者:** Wuqi Su `[一作]` (Zhejiang Gongshang University), Chi Xu `[通讯]` (Zhejiang Gongshang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种多层感知条件随机场模型，用于单目深度估计，利用Swin Transformer提取多尺度特征，并通过自适应层级适配器、混合金字塔特征融合以及全连接CRF解码器实现精确的像素级深度预测。

**💡 创新点**

创新点包括（1）混合金字塔特征融合（HPF）将多尺度空间金字塔池化与双轴特征聚合相结合，提升全局与局部信息的融合；（2）层级意识适配器（HA）在编码器不同层插入轻量广播模块并学习维度缩放，降低计算成本并增强跨层特征交互；（3）带动态尺度注意力和偏置学习单元的全连接CRF解码器，显著捕捉像素间细粒度空间关系。

**🔧 技术方法**

技术手段包括：Swin Transformer骨干网络、空间金字塔池化、双轴特征聚合、广播模块（BC）与可学习维度缩放、全连接CRF+动态尺度注意力、偏置学习单元、像素重排上采样以及SILog损失。

**📊 数据集**

使用的公开数据集有NYU Depth v2、KITTI（Eigen和Geiger分割）以及MatterPort3D，分别用于室内、街景和大规模建筑场景的深度评估。

**📈 对比分析**

在三大数据集上与最新方法对比，本文模型在NYU Depth v2上Abs Rel降至0.088（比NeWCRFs低7.4%），RMSE降0.316；在KITTI上Abs Rel 0.049、RMSE 2.062，阈值精度>99.9%；在MatterPort3D上实现了0.0574的Abs Rel，显示出强大的跨域泛化能力，同时仅使用194M参数、21ms推理速度，显著优于同类精度更高但更慢的方法。

**⚠️ 局限性**

局限性在于模型仍依赖大量预训练数据，且针对单帧静态图像的推断，未针对视频序列或实时多帧融合进行深入探索，未来可进一步缩减参数规模并扩展至动态场景。

---

## 219. VisionClaw: Always-On AI Agents through Smart Glasses

**arXiv ID:** 2604.03486 | [PDF](https://arxiv.org/pdf/2604.03486v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 220. Gram-Anchored Prompt Learning for Vision-Language Models via Second-Order Statistics

**arXiv ID:** 2604.03980 | [PDF](https://arxiv.org/pdf/2604.03980v1)

**作者:** Minglei Chen `[一作]` (Southwestern University of Finance and Economics), Ye Deng `[通讯]` (Southwestern University of Finance and Economics)

**通讯引用:** 323 | [OpenAlex ID](https://openalex.org/A5078615880)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Gram-Anchored Prompt Learning（GAPL）框架，通过二阶统计Gram矩阵作为视觉锚点来引导文本提示学习，从而提升视觉‑语言模型在跨域场景下的适应性能。

**💡 创新点**

创新点在于：① 引入Gram矩阵的二阶统计作为视觉条件的第二阶锚点；② 设计三路融合（全局、Gram锚点、上下文）并通过自适应分支权重实现动态加权；③ 仅在文本编码器深层插入提示，保持视觉特征不变，减少过拟合；④ 通过对数+MLP生成风格门控实现文本特征的二阶调制。

**🔧 技术方法**

使用技术包括：Prompt学习（深度提示）、Gram矩阵统计、风格门控（Style Modulator）、上下文锚点（Contextual Modulator）、多流融合、自适应分支融合、t‑SNE可视化、域对齐距离测量。

**📊 数据集**

使用数据集：ImageNet及其四个变体（V2、Sketch、A、R）用于域泛化；15个公开数据集（Caltech101、OxfordPets、StanfordCars、Flowers102、Food101、FGVCAircraft、SUN397、DTD、EuroSAT、UCF101 等）用于基‑新类别泛化和跨数据集评测。

**📈 对比分析**

与 CoOp、CoCoOp、KgCoOp、MaPLe、TCP、MMA、CoPrompt、HiCroPL、MMRL、PromptSRC 等基线对比。GAPL 在域泛化上平均 OOD 61.12%，在基‑新类别上平均 Novel 78.12% 和 HM 81.77%，均超过所有对比方法，证明其在源域性能与跨域鲁棒性之间取得了更优平衡。

**⚠️ 局限性**

局限性：仅支持离线推理，无法在推理时适应动态分布；未在稠密预测任务（如零样本分割）中验证；深层视觉提示虽提升源域性能但会显著降低 OOD 鲁棒性；目前仅使用 Gram 矩阵的对角线作为二阶特征，可能未能充分挖掘更丰富的统计信息。

---

## 221. 3D-IDE: 3D Implicit Depth Emergent

**arXiv ID:** 2604.03296 | [PDF](https://arxiv.org/pdf/2604.03296v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 222. Decomposing Communication Gain and Delay Cost Under Cross-Timestep Delays in Cooperative Multi-Agent Reinforcement Learning

**arXiv ID:** 2604.03785 | [PDF](https://arxiv.org/pdf/2604.03785v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 223. Discrete Prototypical Memories for Federated Time Series Foundation Models

**arXiv ID:** 2604.04475 | [PDF](https://arxiv.org/pdf/2604.04475v1)

**作者:** Liwei Deng `[一作]` (Australian Artificial Intelligence Institute), Yuxuan Liang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5946 | [OpenAlex ID](https://openalex.org/A5018828723)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 FeDPM，一种基于离散原型记忆的联邦时间序列基础模型，用以解决时间序列与预训练 LLM 的语义不匹配问题，并在联邦学习框架下实现高效的知识共享与预测。

**💡 创新点**

创新点包括：①采用离散原型记忆捕捉各域的周期性模式；②设计跨域记忆对齐机制，基于语义相似度聚类并生成共享与个性化原型；③只传输原型记忆而非完整模型参数，实现通信量大幅降低；④在保证性能的同时显著减少可训练参数。

**🔧 技术方法**

核心技术：联邦学习框架；离散原型记忆检索（VQ‑VAE 思路）；跨域记忆对齐与聚类（余弦相似度 + BFS）；基于 Transformer / CNN / FC 等多种编码器；多项损失（预测、编码器对齐、记忆对齐）；差分隐私噪声注入。

**📊 数据集**

实验使用七个公开时间序列数据集：ETTh1、ETTh2、ETTm1、ETTm2、Electricity、Weather、Exchange，覆盖不同域与多预测时长。

**📈 对比分析**

与 FL-FM、中心化 FM 与专家模型进行对比。FeDPM 在 MAE 上平均比最强基线 FFTS 提升 4.92%，在通信量上比任何基线低 97%，可训练参数比现有 FL 基线低 20%。在少量数据（5%/10%）下仍保持领先，且在加入差分隐私噪声时表现鲁棒。

**⚠️ 局限性**

局限性：目前需要手动调节超参数；跨域记忆对齐在服务器端计算量大，训练时间略长；仅使用全局原型传输，未探索稀疏或压缩方案。

---

## 224. From Model-Based Screening to Data-Driven Surrogates: A Multi-Stage Workflow for Exploring Stochastic Agent-Based Models

**arXiv ID:** 2604.03350 | [PDF](https://arxiv.org/pdf/2604.03350v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 225. Scalable and Explainable Learner-Video Interaction Prediction using Multimodal Large Language Models

**arXiv ID:** 2604.04482 | [PDF](https://arxiv.org/pdf/2604.04482v1)

**作者:** Dominik Glandorf `[一作]` (École Polytechnique Fédérale de Lausanne), Tanja Käser `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 1401 | [OpenAlex ID](https://openalex.org/A5007940211)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究开发了一套可扩展、可解释的管道，利用多模态大型语言模型预测教育视频中学习者的观看、暂停、倒带和跳过行为，并用CTML特征解释预测结果。

**💡 创新点**

创新点在于将多模态LLM嵌入与CTML特征编码结合，利用TCAV进行可解释性分析，并在多学科大规模数据上实现无学习者数据的前瞻性行为预测。

**🔧 技术方法**

主要技术包括多模态LLM（如Qwen-VL、Qwen2.5-VL）嵌入、轻量级神经分类头、GPT‑5自动编码CTML特征、TCAV解释方法，以及多种视音频与文本的特征提取。

**📊 数据集**

使用了来自EPFL平台的77.3 百万条视频交互记录，涵盖1,641段视频、66门课程（涵盖STEM等11个学科），以及自动提取的幻灯片文字与转录文本。

**📈 对比分析**

与传统单一模态或仅用课程元数据的预测模型相比，完整嵌入模型在预测前5%/10%交互峰值时可达AUC≈0.74‑0.76，Lift@K%高达6倍；在未见学科上仍保持良好泛化；CTML特征在小样本时表现更好，但大样本时被高维嵌入超越。

**⚠️ 局限性**

局限包括：交互信号并不直接反映教学质量；模型对观众特定人群的依赖；缺乏对跳过起始时间和终点的分析；对人文等非STEM学科的泛化尚待验证。

---

## 226. Exploring Expert Perspectives on Wearable-Triggered LLM Conversational Support for Daily Stress Management

**arXiv ID:** 2604.04915 | [PDF](https://arxiv.org/pdf/2604.04915v1)

**作者:** Poorvesh Dongre `[一作]` (Harvard Medical School), Denis Gračanin `[通讯]` (Virginia Tech)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了一款名为EmBot的可穿戴设备触发式LLM对话式压力管理应用，作为设计探针与心理健康专家进行半结构化访谈，探讨交互设计和系统实现的可行性与挑战。

**💡 创新点**

首次将可穿戴传感器的即时压力检测与大语言模型生成的情境化对话结合，并通过专家访谈系统化挖掘设计张力与实践建议，为此类混合系统提供早期设计框架。

**🔧 技术方法**

利用可穿戴心率/运动等生理信号进行压力检测，触发基于GPT类LLM的聊天接口生成反思式、个性化支持；实现包括检测、反馈、支持和反思四个交互阶段。

**📊 数据集**

实验使用模拟的可穿戴压力检测事件（非真实数据），结合专家访谈记录作为主要数据源；未使用公开医学或心理健康数据集。

**📈 对比分析**

研究聚焦设计探索，不涉及模型性能对比；通过专家反馈进行主题分析，未给出定量指标。

**⚠️ 局限性**

主要限制包括：压力检测为模拟且未验证、访谈模式（现场/远程）可能影响反馈深度、专家背景多样导致观点差异、系统未在真实用户中验证其有效性与负担。

---

## 227. Synthesis4AD: Synthetic Anomalies are All You Need for 3D Anomaly Detection

**arXiv ID:** 2604.04658 | [PDF](https://arxiv.org/pdf/2604.04658v1)

**作者:** Yihan Sun `[一作]` (Huazhong University of Science and Technology), Weiming Shen `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 23204 | [OpenAlex ID](https://openalex.org/A5062049138)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了端到端的Synthesis4AD体系，利用高保真3D异常合成与检测器训练相结合的流程，生成大量带掩码的合成缺陷点云并训练Point Transformer模型，实现无监督3D异常检测和定位。

**💡 创新点**

创新点包括：① Multi-dimensional Primitive-guided Anomaly Synthesis (MPAS)，通过1D/2D/3D几何支撑实现物理等价、结构复杂的缺陷合成；② 3D-DefectStudio与多模态大语言模型（MLLM）协同，将设计知识转化为可执行合成指令；③ 引入空间分布归一化（SDN）和几何保真增强，提升训练对尺度、姿态、噪声的鲁棒性。

**🔧 技术方法**

核心技术包括：MPAS框架（几何支撑、自由形变、局部平滑）；MLLM（Gemini 3）解析设计信息并生成合成脚本；3D-DefectStudio（Python API + GUI）执行合成并生成掩码；Point Transformer编码器+MLP分割头；SDN + 随机旋转、噪声扰动、点丢弃等增强；原型匹配推理。

**📊 数据集**

使用公开基准 Real3D-AD 与 MulSen-AD，此外自行采集的工业零件扫描数据（Bevel Gear、Brake Caliper 等六类）。

**📈 对比分析**

在 O-ROC/P-ROC 指标上与 PatchCore、CPMF、IMRNet、R3D-AD、GLFM 等方法对比，Synthesis4AD 在 Real3D-AD 上取得 80.9%/84.8%（O/P），在 MulSen-AD 上 89.6%/72.0%，在工业数据集上 95.9%/73.8%，均显著优于现有最优方法。

**⚠️ 局限性**

当前工作为开环流程，合成缺陷的质量未依据检测反馈自适应改进；对极端复杂缺陷或非典型场景的覆盖仍有限；合成过程对MLLM解析精度和知识工程依赖较高。

---

## 228. TreeGaussian: Tree-Guided Cascaded Contrastive Learning for Hierarchical Consistent 3D Gaussian Scene Segmentation and Understanding

**arXiv ID:** 2604.03309 | [PDF](https://arxiv.org/pdf/2604.03309v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 229. MoViD: View-Invariant 3D Human Pose Estimation via Motion-View Disentanglement

**arXiv ID:** 2604.03299 | [PDF](https://arxiv.org/pdf/2604.03299v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 230. AnyUser: Translating Sketched User Intent into Domestic Robots

**arXiv ID:** 2604.04811 | [PDF](https://arxiv.org/pdf/2604.04811v1)

**作者:** Songyuan Yang `[一作]` (National University of Defense Technology), Shaowu Yang `[通讯]` (National University of Defense Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了AnyUser，一种基于自由手绘草图（可加语言）的统一指令系统，使非专业用户能够在家用机器人上直接通过对现场照片的绘制来指定任务，无需事先地图或模型。

**💡 创新点**

创新点包括：①将草图、视觉与语言三模态融合为空间语义原语；②基于层次化宏动作的策略，实现从二维草图到三维可执行行为的自动映射；③构建混合现实数据集HouseholdSketch；④采用闭环感知与实时检查提高长程执行可靠性。

**🔧 技术方法**

使用技术包括：Vision Transformer（ViT）与CLIP文本编码器、草图几何编码器（关键点检测+聚合）、跨模态注意力融合网络、离散宏动作策略π_HL、g_translate平台特定翻译模块，以及基于Hybrid训练的多任务优化。

**📊 数据集**

使用数据集：HouseholdSketch，包含约35,000个标注任务，涵盖20,000+真实室内场景和15,000+程序化合成场景，涵盖多样的布局、照明与草图风格。

**📈 对比分析**

通过在HouseholdSketch基准上的定量评测（单步成功率≈84%，长任务完整率≈46%）以及在KUKA LBR iiwa和Realman RMC-AIDAL平台上的真实环境实验（任务完成率85–96%），结合32人群的可用性实验，验证了系统在多模态解读、宏动作生成和用户交互方面的优异性能。

**⚠️ 局限性**

局限性包括：长序列执行时误差累积导致完整率下降；对草图精度和视觉重投影的依赖；仅使用单张静态照片，难以应对大范围环境变更；宏动作离散化导致细粒度控制受限；缺乏持久地图与全局规划支持。

---

## 231. Entropy and Attention Dynamics in Small Language Models: A Trace-Level Structural Analysis on the TruthfulQA Benchmark

**arXiv ID:** 2604.03589 | [PDF](https://arxiv.org/pdf/2604.03589v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 232. Toward a Universal Color Naming System: A Clustering-Based Approach using Multisource Data

**arXiv ID:** 2604.03235 | [PDF](https://arxiv.org/pdf/2604.03235v1)

**作者:** Aruzhan Sabitkyzy `[一作]` (Kazakh-British Technical University), Pakizar Shamoi `[通讯]` (Kazakh-British Technical University)

**通讯引用:** 378 | [OpenAlex ID](https://openalex.org/A5021817760)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文基于聚类的多源数据框架，收集并清洗了 19,555 条 RGB‑名称对，转入 CIELAB 颜色空间后使用 K‑means（CIEDE2000 作为距离度量）聚成 280 个颜色簇，并为每个簇通过频率分析分配代表性颜色名称，随后在内容检索等应用中验证该系统的实用性。

**💡 创新点**

创新点在于：①整合 20 个来源的大规模颜色数据；②利用 CIEDE2000 在 LAB 空间中自动确定最优聚类数（k=280）；③通过真实人类命名频率直接为每个簇赋标签，获得可解释且通用的颜色名称；④构建了可直接用于 GAN、CBIR 等任务的标准化颜色命名表。

**🔧 技术方法**

主要技术包括：数据清洗与文本规范化、RGB→LAB 转换、K‑means 聚类（k 选取使用 Elbow/KneeLocator）、CIEDE2000 色差度量、频率统计与词频排序，以及基于颜色匹配的内容检索实验。

**📊 数据集**

使用了 19,555 条 RGB+名称对的多源数据集（来自 20 个在线来源，如 Pantone、Benjamin Moore 等），并在时尚商品数据集 Visuelle 上进行 CBIR 实验。

**📈 对比分析**

通过 Elbow 曲线与 KneeLocator 自动确定 k=280；在 Visuelle 数据集上使用主导颜色匹配并检索对应颜色名称，展示了标签一致性和语义匹配的提升；实验结果主要以可视化案例呈现，未给出定量指标，但显示出比传统不标准化命名更具可解释性。

**⚠️ 局限性**

局限性包括：仅覆盖英文命名；缺乏跨语言验证；K‑means 聚类是静态方法，可能无法完美捕捉所有颜色分布；频率分析仅保留前 5 名，可能忽略稀有但重要的名称；实验评估缺乏量化对比。

---

## 233. Would Learning Help? Adaptive CRC-QC-LDPC Selection for Integrity in 5G-NR V2X

**arXiv ID:** 2604.04277 | [PDF](https://arxiv.org/pdf/2604.04277v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 234. Improving Model Performance by Adapting the KGE Metric to Account for System Non-Stationarity

**arXiv ID:** 2604.03906 | [PDF](https://arxiv.org/pdf/2604.03906v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 235. Persistent Cross-Attempt State Optimization for Repository-Level Code Generation

**arXiv ID:** 2604.03632 | [PDF](https://arxiv.org/pdf/2604.03632v1)

**作者:** Ruwei Pan `[一作]` (Chongqing University), Hongyu Zhang `[通讯]` (Chongqing University)

**通讯引用:** 19031 | [OpenAlex ID](https://openalex.org/A5100412598)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于跨尝试知识优化的仓库级代码生成框架，能够在多次生成过程中持续保存和利用成功经验、失败经验以及历史最佳仓库，从而将每次生成视为一次增量式的、知识驱动的优化过程。

**💡 创新点**

创新点在于：①引入三种持久化任务特定状态（成功知识、失败知识、历史最佳仓库）并在跨尝试间共享；②通过结构化文本提取与语义匹配自动构建知识库；③实现历史最佳仓库的直接重用与回退机制，防止性能回退；④在多次尝试中显著提升功能正确率、重用率并降低成本。

**🔧 技术方法**

主要技术包括：大语言模型（GPT‑5、DeepSeek‑V3、Claude‑Sonnet‑4.5、Gemini‑3‑Pro‑Preview）的多轮生成与内部迭代；LLM 生成的仓库与执行反馈被自动提取为结构化的成功/失败知识；使用向量相似度检索匹配相关知识；历史最佳仓库以完整代码形式保存并在需要时直接返回。

**📊 数据集**

使用了两个仓库级代码生成基准：RAL‑Bench（关注多文件协同、依赖与执行）和 NL2Repo‑Bench（自然语言到完整仓库的翻译）。

**📈 对比分析**

与一轮直接生成、Self‑Reflection、SE‑Agent、AlphaEvolve、CSE、Live‑SWE‑Agent 等现有方法进行对比。实验表明，在所有四个前沿 LLM 上，本文框架的功能分数提升可达 22.94 个百分点，仓库重用率高达 81.58%，且累计成本相对首轮尝试下降 41.99–53.63%，总体性能和成本效益均优于基线。

**⚠️ 局限性**

局限性包括：仅评估了两种基准和单一编程语言；实验采用固定尝试次数和统一提示，可能影响结果泛化；模型与基准的潜在数据重叠未完全排除；对极难任务的提升仍有限，需进一步完善跨文件推理与完整功能验证。

---

## 236. Adaptive Cost-Efficient Evaluation for Reliable Patent Claim Validation

**arXiv ID:** 2604.04295 | [PDF](https://arxiv.org/pdf/2604.04295v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 237. Lattice-Boltzmann-Driven Physics-Informed Neural Networks for Droplet Wettability on Rough Surfaces

**arXiv ID:** 2604.03481 | [PDF](https://arxiv.org/pdf/2604.03481v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 238. Hardware-Oriented Inference Complexity of Kolmogorov-Arnold Networks

**arXiv ID:** 2604.03345 | [PDF](https://arxiv.org/pdf/2604.03345v1)

**作者:** Bilal Khalid `[一作]` (Aston University), Jaroslaw E. Prilepsky `[通讯]` (Aston University)

**通讯引用:** 2808 | [OpenAlex ID](https://openalex.org/A5024195302)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文提出了基于硬件的推理复杂度评估框架，推导了四类 Kolmogorov–Arnold 网络（B‑splines、GRBF、Chebyshev、多项式和 Fourier）在 RM、BOP 与 NABS 三种指标下的通用闭式公式，并与传统 MLP 进行对比；

**💡 创新点**

创新点在于首次对多种 KAN 变体给出平台无关的硬件复杂度指标，并通过 iso‑complexity 分析展示不同网络宽度下的资源平衡，弥补了现有 FLOPs 评价与平台专属资源（LUT、DSP 等）评估之间的空缺；

**🔧 技术方法**

技术方法主要为理论推导与数学建模，利用 B‑spline 的局部支撑、GRBF 的 LUT 预计算、Chebyshev 与 Fourier 的全局基函数等特性，分别得到每条边的 RM、BOP 与 NABS；随后在统一 8‑bit 量化假设下进行数值比较；

**📊 数据集**

论文未使用实际数据集，仅通过仿真网络结构 [3,16,16,2] 与 [3,64,64,2] 等进行复杂度评估；

**📈 对比分析**

比较方法是将同尺寸网络在 RM、BOP、NABS 三种硬件指标下进行直观对比，并通过 iso‑complexity 曲线探讨在固定硬件预算下 KAN 需要的隐藏层宽度；结果显示 KAN 的单边成本高约 5–6 倍，但可通过更小网络获得相似准确度；

**⚠️ 局限性**

局限性包括：仅考虑推理阶段且假设已采用 LUT‑优化的硬件实现；未给出实际硬件实现或实验验证；参数选取为示例值，未探讨自动优化；对训练复杂度与动态行为的分析缺失；

---

## 239. What Makes a Sale? Rethinking End-to-End Seller--Buyer Retail Dynamics with LLM Agents

**arXiv ID:** 2604.04468 | [PDF](https://arxiv.org/pdf/2604.04468v1)

**作者:** Jeonghwan Choi `[一作]` (Korea Advanced Institute of Science and Technology), Hwanjun Song `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2082 | [OpenAlex ID](https://openalex.org/A5033909285)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了统一的多阶段零售交互模拟框架 RetailSim，模拟卖方说服、买方互动及最终购买与评价，并通过大语言模型生成真实对话与决策流程。

**💡 创新点**

① 融合跨阶段因果结构，完整模拟从卖方说服到买方结果的链条；② 引入可控制的人格化代理，系统研究人格与经济结果的关联；③ 采用系统级经济一致性评估，验证模拟与现实经济规律相符；④ 用框架进行实际决策分析，如隐藏人格推断、收益影响与销售策略评估。

**🔧 技术方法**

大语言模型（Qwen3、GPT-oss、Gemini、DeepSeek、GPT-5.4 等）作为卖方和买方代理；多轮对话提示模板与人格注入；LLM生成与状态转移；人类评估和经济一致性元评估；机器学习分类器用于推断人格。

**📊 数据集**

Amazon Reviews 2023 产品元数据用于构建产品空间；人工标注的对话脚本与情景表（来自零售公司）。

**📈 对比分析**

对 8 种 LLM 进行任务层面人类评估，任务得分平均 >4，人物识别准确率 >95%；系统层面验证性别购买模式、价格需求关系和价格弹性，均与现实一致。对比不同人类指导级别的销售脚本，收入随指导增加而提升，模型差距逐步缩小。

**⚠️ 局限性**

模拟仍受 LLM 输出和提示偏见影响；缺乏真实交易数据验证；经济一致性验证仅覆盖有限规律；高阶策略与多样化场景的可扩展性尚未完全探究。

---

## 240. AI Governance Control Stack for Operational Stability: Achieving Hardened Governance in AI Systems

**arXiv ID:** 2604.03262 | [PDF](https://arxiv.org/pdf/2604.03262v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 241. NTIRE 2026 3D Restoration and Reconstruction in Real-world Adverse Conditions: RealX3D Challenge Results

**arXiv ID:** 2604.04135 | [PDF](https://arxiv.org/pdf/2604.04135v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 242. ZeD-MAP: Bundle Adjustment Guided Zero-Shot Depth Maps for Real-Time Aerial Imaging

**arXiv ID:** 2604.04667 | [PDF](https://arxiv.org/pdf/2604.04667v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 243. InCTRLv2: Generalist Residual Models for Few-Shot Anomaly Detection and Segmentation

**arXiv ID:** 2604.04632 | [PDF](https://arxiv.org/pdf/2604.04632v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 244. DRAFT: Task Decoupled Latent Reasoning for Agent Safety

**arXiv ID:** 2604.03242 | [PDF](https://arxiv.org/pdf/2604.03242v1)

**作者:** Lin Wang `[一作]` (National University of Singapore), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 61515 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 DRAFT 框架，将 agent 安全判定拆分为提取器（Extractor）和推理器（Reasoner）两阶段，利用连续潜在空间压缩长交互轨迹中的稀疏风险证据。

**💡 创新点**

创新点在于不需要显式生成中间推理文本，而是通过可训练的连续潜在草稿实现隐式推理，完成证据抽取与决策读取的解耦，提升稀疏监督下的梯度可达性和特征可分离度。

**🔧 技术方法**

采用轻量级 LoRA 适配器实现提取器和推理器，并引入跨空间投影、Transformer 多头注意力等技术构建潜在工作空间，支持端到端可微分训练。

**📊 数据集**

在 ASSEBench、AuraGen 和 R‑Judge 三个 agent 安全基准上进行实验，评估模型在不同 backbone（Qwen3‑8B、Llama‑3.1‑8B 等）上的表现。

**📈 对比分析**

与 Vanilla、SFT、LoRA、AgentAuditor 以及显式 summarize‑then‑judge 基线对比，DRAFT 在所有数据集上平均提升准确率至约 91%（相对 63%‑65% 的基线提升约 40%），并在 t‑SNE 可视化中显著改善安全与非安全样本的分离。

**⚠️ 局限性**

局限性包括：仍受弱监督限制，潜在草稿长度需经验调优；在高度动态或对抗环境下的鲁棒性尚未充分验证；对推理时间与内存消耗的影响相对较小但仍需进一步评估。

---

## 245. Entropy, Disagreement, and the Limits of Foundation Models in Genomics

**arXiv ID:** 2604.04287 | [PDF](https://arxiv.org/pdf/2604.04287v1)

**作者:** Maxime Rochkoulets `[一作]` (KU Leuven), Mile Šikić `[通讯]` (University of Zagreb)

**通讯引用:** 8642 | [OpenAlex ID](https://openalex.org/A5000633958)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对比了使用 BERT 模型在英文文本和 DNA 序列上进行的自监督训练，重点分析了数据熵对模型预测、嵌入一致性和信息分布的影响。

**💡 创新点**

通过多维度评估（KL、Jensen‑Shannon 距离、静态词嵌入一致性、经验 Fisher 信息分布）揭示 DNA 序列高熵导致模型预测不确定、模型间不一致、并且模型对互相关系的利用显著不足，说明仅用序列自监督可能不足以构建基因组基础模型。

**🔧 技术方法**

使用 BERT Transformer encoder、BPE 与 k‑mer 分词、KL 与 Jensen‑Shannon 量化不确定性、Procrustes 对齐与 Spearman / Jaccard 评估嵌入一致性、经验 Fisher 信息矩阵（只取对角）进行层级信息分布分析。

**📊 数据集**

训练数据为 5 B 个 tokens 的英文文本与 DNA 序列，词表大小 4096（6‑mers），三组模型分别使用英文 BPE、DNA BPE、DNA k‑mer 分词。

**📈 对比分析**

在相同架构、参数量、token 量和词表下，利用 KL 与 uniform 的差异、Jensen‑Shannon 距离随 top‑p 变化、嵌入层的相似度指标和 Fisher 信息分布对模型进行比较。DNA 模型的 KL 仅约 3 bits（低置信度），与文本的 10+ bits 相比差距显著；在低 top‑p 下 DNA 模型的 JS 距离急剧升高；嵌入层一致性指标（Spearman、Jaccard、Cosine、Disparity）在 DNA 组明显低于文本组；Fisher 信息显示 DNA 模型的静态嵌入层承载绝大部分信息，而文本模型则集中在 Transformer 层。

**⚠️ 局限性**

实验仅聚焦于自监督未见 token 预测，未验证其他预训练任务；高熵导致的模型不稳定和随机初始化敏感性降低可复现性；tokenization 对结果影响有限，说明问题根源在数据本身；缺乏对下游任务性能的直接比较，只通过信息理论指标间接推断。

---

## 246. Unlocking Prompt Infilling Capability for Diffusion Language Models

**arXiv ID:** 2604.03677 | [PDF](https://arxiv.org/pdf/2604.03677v1)

**作者:** Yoshinari Fujinuma `[一作]` (Patronus AI), Keisuke Sakaguchi `[通讯]` (Tohoku University)

**通讯引用:** 2343 | [OpenAlex ID](https://openalex.org/A5101067919)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究者提出并实现了在监督微调阶段对 diffusion 语言模型使用全序列遮蔽（mask）训练，从而解锁了模型对提示（prompt）的填空（infilling）能力。

**💡 创新点**

创新点在于识别并纠正了当前主流 diffusion 模型仅对响应（response）进行遮蔽导致的训练与推理不一致问题，并通过全序列遮蔽让模型学习到对提示进行反向解码的能力。

**🔧 技术方法**

技术包括：全序列遮蔽（FS）与响应遮蔽（RO）相结合的训练策略、基于 diffusion 的逐步反向 denoising、提示填空验证与选择机制，以及与现有提示优化方法（COPRO、GEPA）进行融合。

**📊 数据集**

使用的数据集主要有：Feedback Collection（训练）、SummEval 与 BigGen‑Bench（评估）、以及 GSM8K 等公开基准；模型基于 LLaDA 与 Dream 两个 diffusion 语言模型实现。

**📈 对比分析**

与传统仅响应遮蔽（RO）和公开检查点（None）进行对比，FS/FS+RO 在 SummEval 上 Spearman 相关性提升约 20% 以上，BigGen‑Bench 上提升约 30%，在 GSM8K 上提示填空可将准确率从 73% 提升至 76% 并显著缩短提示长度；验证实验还显示填空生成的提示可迁移到其他模型并提升其性能。

**⚠️ 局限性**

局限性包括：训练成本仍高、对极大规模数据和多任务的泛化能力尚未充分验证、以及填空提示在复杂多模态任务中的适用性仍需进一步研究。

---

## 247. NBI-Slurm: Simplified submission of Slurm jobs with energy saving mode

**arXiv ID:** 2604.04558 | [PDF](https://arxiv.org/pdf/2604.04558v1)

**作者:** Andrea Telatin `[一作]` `[通讯]` (Quadram Institute Bioscience), Andrea Telatin (Quadram Institute Bioscience)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并实现了NBI‑Slurm，一个基于Perl的工具包，简化了对SLURM集群的作业提交与管理。

**💡 创新点**

创新点在于提供交互式TUI界面、声明式工具包装器以及基于时间窗的能源感知调度模式（eco mode）。

**🔧 技术方法**

主要技术包括Perl模块、命令行工具、JSON序列化、TUI交互、能源调度算法等，依赖CPAN包如Getopt::Long、Term::ReadLine等。

**📊 数据集**

示例数据集为基因组组装与FASTQ文件，未使用公开大规模基准集，实验以示例脚本展示功能。

**📈 对比分析**

与原始SLURM及工作流系统（Snakemake、Nextflow）对比，本文未给出定量性能指标，主要强调易用性与能源效率。

**⚠️ 局限性**

局限性包括仅支持Perl环境、需手动配置生态窗、对非生态作业的适用性有限，以及在无SLURM环境时无法完整测试。

---

## 248. FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control

**arXiv ID:** 2604.04539 | [PDF](https://arxiv.org/pdf/2604.04539v1)

**作者:** Donghu Kim `[一作]` (Holiday Robotics), Hojoon Lee `[通讯]` (Holiday Robotics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了FlashSAC，一种快速且稳定的离线强化学习算法，专为高维机器人控制设计。

**💡 创新点**

创新点在于将大规模数据采集、大模型与极少梯度更新相结合，并通过权重、特征和梯度范数约束、分布式批归一化、分布式价值预测等架构改进实现离线学习的稳定性。

**🔧 技术方法**

基于Soft Actor-Critic，结合大容量残差网络、RMSNorm、分布式Q值、奖励缩放、统一熵目标以及噪声重复探索等技术。

**📊 数据集**

在超过60个任务、10个仿真器的数据集上评估，包括IsaacLab、ManiSkill、Genesis、MuJoCo Playground、DMControl、MySuite、HumanoidBench、Unitree G1等，涵盖低/高维状态动作、视觉控制和Sim-to-Real。

**📈 对比分析**

与常用基线（PPO、FastTD3、DrQ-v2、MR.Q、XQC、SimbaV2、TD-MPC2）对比，FlashSAC在绝大多数任务中实现了更高的最终性能、显著更快的墙钟时间（尤其在高维和Sim-to-Real场景下可快十倍），并在单环境CPU设置中保持了竞争力。

**⚠️ 局限性**

局限性包括对大规模并行仿真和大批量训练的需求，对计算资源的高依赖；在极低样本或极度稀疏奖励环境下的探索效率仍有提升空间；部分技术细节（如奖励缩放、噪声重复）需针对不同任务进行微调。

---

## 249. Banana100: Breaking NR-IQA Metrics by 100 Iterative Image Replications with Nano Banana Pro

**arXiv ID:** 2604.03400 | [PDF](https://arxiv.org/pdf/2604.03400v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 250. Agentic Code Optimization via Compiler-LLM Cooperation

**arXiv ID:** 2604.04238 | [PDF](https://arxiv.org/pdf/2604.04238v1)

**作者:** Benjamin Mikek `[一作]` (Amazon Web Services Artificial Intelligence), Panpan Xu `[通讯]` (Amazon Web Services Artificial Intelligence)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了编译器与大型语言模型（LLM）协同优化系统，通过多级抽象的LLM优化代理和编译器工具共同生成更高性能的可执行代码。

**💡 创新点**

将LLM的创造性代码生成与传统编译器优化相结合，设计了多代理协同框架，并实现了在不同抽象层级分配计算预算的机制。

**🔧 技术方法**

使用多代理系统，包括LLM优化代理、编译器工具、LLM测试生成代理以及指挥LLM；采用大型语言模型（如GPT系列）进行代码生成与测试。

**📊 数据集**

在标准编译器基准测试集（如SPEC CPU）上进行评估，涵盖多种常见程序。

**📈 对比分析**

与传统编译器优化以及单层LLM基线进行对比，实验表明系统平均可获得最高1.25倍的速度提升。

**⚠️ 局限性**

仍存在LLM生成错误代码的风险、计算资源消耗高、以及对极大规模程序的可扩展性不足等局限。

---

## 251. Significance and Stability Analysis of Gene-Environment Interaction using RGxEStat

**arXiv ID:** 2604.03337 | [PDF](https://arxiv.org/pdf/2604.03337v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 252. InsTraj: Instructing Diffusion Models with Travel Intentions to Generate Real-world Trajectories

**arXiv ID:** 2604.04106 | [PDF](https://arxiv.org/pdf/2604.04106v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 253. User Simulator-Guided Multi-Turn Preference Optimization for Reasoning LLM-based Conversational Recommendation

**arXiv ID:** 2604.03671 | [PDF](https://arxiv.org/pdf/2604.03671v1)

**作者:** Xingyuan Xiang `[一作]` (Huazhong University of Science and Technology), Wei Wei `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 254399 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了SMTPO框架，利用LLM驱动的用户模拟器和多轮交互进行偏好优化，从而提升对话式推荐系统的推荐质量。

**💡 创新点**

创新点在于：①将多任务监督微调（SFT）与多轮强化学习（RL）结合，逐步让推荐器对真实用户偏好进行校准；②构建多任务SFT的用户模拟器，生成高质量自然语言反馈；③采用双视角检索器（语义+协同）动态限制候选集；④设计细粒度奖励（格式、推荐、偏好）引导多轮学习。

**🔧 技术方法**

技术手段包括：大型语言模型（Llama-3.1-8B-Instruct、DeepSeek-R1-Distill-Llama-8B）、LoRA微调、GRPO强化学习、InfoNCE检索训练、BPR图嵌入、跨注意力融合、强化学习奖励设计。

**📊 数据集**

使用公开的对话式推荐数据集ReDial和INSPIRED进行评估。

**📈 对比分析**

与传统推荐、通用LLM零样本、以及最新CRS方法（UniCRS、DCRS、MSCRS等）进行对比，SMTPO在Recall@k、NDCG@k和MRR@k上均显著优于所有基线，尤其在多轮交互中表现更突出。

**⚠️ 局限性**

局限性包括：①需要大规模LLM与高昂的计算资源；②模拟器生成的反馈仍可能存在偏差，需进一步校准；③在极少交互回合下的鲁棒性待提升；④对真实用户数据的依赖程度高，未在真实对话环境中充分验证。

---

## 254. Beyond Predefined Schemas: TRACE-KG for Context-Enriched Knowledge Graphs from Complex Documents

**arXiv ID:** 2604.03496 | [PDF](https://arxiv.org/pdf/2604.03496v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 255. FileGram: Grounding Agent Personalization in File-System Behavioral Traces

**arXiv ID:** 2604.04901 | [PDF](https://arxiv.org/pdf/2604.04901v1)

**作者:** Shuai Liu `[一作]` (Nanyang Technological University), Ziwei Liu `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出统一的FileGram框架，包含行为轨迹生成、评测基准与底层记忆模型，用于文件系统个性化协作。

**💡 创新点**

创新点：①基于文件操作与内容增量的底层记忆；②可控生成多模态文件轨迹并注入行为漂移；③设计四轨评测覆盖程序、语义、情节与多模态。

**🔧 技术方法**

使用技术：LLM（Gemini 2.5‑Flash）+ 行为统计向量、语义提取、可视化语言模型、检索聚合、底层多模态处理。

**📊 数据集**

使用数据集：640条模拟轨迹（20个配置档案×32任务，约20 k动作、2.5 k文件），再衍生10 k多模态文件，附真实屏幕录制补充。

**📈 对比分析**

方法对比：与12个基线（全上下文、RAG、Mem0、Zep、EverMemOS等）对比，FileGram在所有轨道上均领先，记忆特定轨道达到59.6%+，显著优于最强叙述性基线49.9%。

**⚠️ 局限性**

局限性：仍难将模拟轨迹迁移至真实视频；行为漂移归因仍不足；依赖高质量日志，对输入内容干扰敏感。

---

## 256. Healthcare App Design in Low-Resource Contexts: Challenges, Practices, and Opportunities

**arXiv ID:** 2604.04669 | [PDF](https://arxiv.org/pdf/2604.04669v1)

**作者:** Arka Majhi `[一作]` (Indian Institute of Technology Bombay), Satish B. Agnihotri `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

组织了一场关于低资源环境下数字健康应用设计的Birds of a Feather会议，促成跨学科交流。

**💡 创新点**

创新点在于将全球不同研究和实践社区聚集在同一平台，重点讨论低资源情境下的设计挑战与合作机会。

**🔧 技术方法**

主要利用会议互动、案例讨论和小组反思等人机交互方法；未涉及特定技术实现。

**📊 数据集**

无数据集使用，讨论基于参与者经验和案例分享。

**📈 对比分析**

未进行实验或性能比较，仅通过案例讨论与专家经验对比探讨设计思路。

**⚠️ 局限性**

局限性包括缺乏实证验证、讨论深度受时间限制、未能形成具体可落地的技术解决方案。

---

## 257. ShieldNet: Network-Level Guardrails against Emerging Supply-Chain Injections in Agentic Systems

**arXiv ID:** 2604.04426 | [PDF](https://arxiv.org/pdf/2604.04426v1)

**作者:** Zhuowen Yuan `[一作]` (University of Illinois Urbana-Champaign), Bo Li `[通讯]` (Virtue AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于代码级注入、网络级验证的 MLC 工具安全基准，并提出一种基于网络流量事件抽取与轻量后训练模型的实时 Guardrail 检测框架

**💡 创新点**

创新点在于：①将攻击注入到真实 MCP 工具实现中，保持接口正常但实现恶意行为；②用网络层事件抽取和 TLS 解密将原始 PCAP 转化为结构化可解释的文本序列；③在此基础上训练轻量后置模型实现低延迟的流式检测，显著提升对隐蔽供应链攻击的识别率

**🔧 技术方法**

技术手段包括：MITM 代理+PCAP 捕获、TLS 解密、事件提取与结构化、后训练 Qwen3‑0.6B 轻量模型、滑窗流式检测、与前沿 LLM 和 IDS 进行对标评测

**📊 数据集**

数据集由 109 个真实 MCP 服务器、984 个工具、29 种 MITRE ATT&CK 网络可见技术构成，生成 19,961 个服务器–工具–技术组合；训练集约 6,000 个样本，用于后训练模型

**📈 对比分析**

与三大 MCP 扫描器（Cisco AI、Ramparts、Invariant Labs）、AgentIO 语义基线、传统 IDS（Suricata、Safe‑NID）以及前沿 LLM（GPT‑4/5、Qwen3）对比；实验表明模型在二分类上 F1 最高达 0.995，FPR 仅 0.8%，运行时延仅 +21%，在未见服务器和攻击技术上亦保持高效且稳健的检测性能

**⚠️ 局限性**

局限性在于：只覆盖网络层可见攻击，无法检测纯粹的本地文件操作或权限变化；依赖 TLS 解密和 MITM 代理，对抗性网络环境可能影响覆盖率；在极高并发环境下的实时性能仍需进一步验证

---

## 258. Incentives shape how humans co-create with generative AI

**arXiv ID:** 2604.03529 | [PDF](https://arxiv.org/pdf/2604.03529v1)

**作者:** Nathanael Jo `[一作]` (Massachusetts Institute of Technology), Manish Raghavan `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 3252 | [OpenAlex ID](https://openalex.org/A5052541789)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过前注册的随机对照实验，探讨了在创意写作任务中使用生成式 AI（OpenAI GPT5‑Mini）时，激励结构（原创性奖励 vs 质量奖励）对作品多样性的影响。

**💡 创新点**

创新点在于：①在开放式、人机交互式写作环境下验证激励能缓解 AI 的同质化效应；②提出并验证了基于 n‑gram 与嵌入的多维文本多样性度量；③通过“AI 采用率”指标揭示用户策略与多样性之间的因果关系。

**🔧 技术方法**

使用技术包括：实验平台（Prolific 招募、浏览器编辑器+对话界面）、OpenAI GPT5‑Mini 生成式模型、嵌入模型（all-MiniLM、all-mpnet、Qwen3-Embedding、embeddinggemma、Style-Embedding、STAR、byt5）、n‑gram 统计、PCA 及多重差异检验（Welch t‑test）。

**📊 数据集**

数据集为 200 名参与者的短篇故事（250–350 字）以及对应的 5 秒文本快照与 AI 对话日志；同时收集自评 AI 经验与人口统计信息。

**📈 对比分析**

方法上通过留一平均相似度计算多样性，使用多指标差异检验，结果显示：①在 AI 辅助下，作品相似度显著高于人类独立写作；②在原创性激励组中，最终作品相似度显著降低，表明多样性提升；③差异主要体现在嵌入层面，说明 AI 对深层叙事结构的同质化更为显著。

**⚠️ 局限性**

局限性包括：①仅单次实验会话，无法观察长期使用行为；②仅测试一种通用聊天式 AI 辅助界面，未涵盖嵌入式或主动式 AI；③多样性度量依赖嵌入模型，缺乏外部验证；④未评估质量与多样性的权衡。

---

## 259. DeonticBench: A Benchmark for Reasoning over Rules

**arXiv ID:** 2604.04443 | [PDF](https://arxiv.org/pdf/2604.04443v1)

**作者:** Guangyao Dou `[一作]` (Johns Hopkins University), Benjamin Van Durme `[通讯]` (Johns Hopkins University)

**通讯引用:** 8696 | [OpenAlex ID](https://openalex.org/A5075825791)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了DeonticBench，一个包含6,232个长上下文、规则驱动的高风险推理任务的基准，涵盖美国联邦税法、航空行李政策、移民行政及州住房法，并提供可执行的Prolog参考程序。

**💡 创新点**

创新点包括：①将可执行符号推理与自然语言链式推理相结合，构建跨领域、可复制的规则推理基准；②创建难度子集（hard subset）以聚焦模型弱点；③提供统一格式与Prolog代码，支持程序轨迹分析；④系统性评估LLM、代码模型以及对抗训练/强化学习对符号推理的影响。

**🔧 技术方法**

技术手段包括：链式思考提示（Chain-of-Thought）、Zero/Direct/Few-shot提示策略；LLM到Prolog的自动化生成管道并在SWI-Prolog求解器上执行；监督微调（SFT）、直接偏好优化（DPO）、GRPO强化学习；错误分类与分析、bootstrap置信区间评估。

**📊 数据集**

使用的数据集为自研的USCIS-AAO（242个案例）以及现有的SARA、Airline、Housing数据集，统一格式后共6,232个任务；每个任务都附有对应的Prolog参考程序。

**📈 对比分析**

在hard子集上对比了多款前沿模型（GPT-4.1、GPT-5.1/5.2、O3、Claude 4.5、Gemini 2.5 Flash、Kimi K2、Qwen3-235B）和开源模型；结果显示前沿模型最高得分约44–46%（宏F1/准确率），但整体表现仍偏低；RL与SFT提升有限，模型对提示策略高度敏感。

**⚠️ 局限性**

局限性：①整体准确率仍远低，尤其在数值计算与规则选择上易出错；②Prolog生成错误频繁，导致抽象推理失败；③强化学习方法提升不稳定，尚未能显著提升端到端表现；④缺乏自适应置信度机制，导致高风险领域易产生错误；⑤目前仅用于研究评估，不能直接部署于实际决策场景。

---

## 260. Compliance-by-Construction Argument Graphs: Using Generative AI to Produce Evidence-Linked Formal Arguments for Certification-Grade Accountability

**arXiv ID:** 2604.04103 | [PDF](https://arxiv.org/pdf/2604.04103v1)

**作者:** Mahyar T. Moghaddam `[一作]` `[通讯]` (SDU Software Engineering), Mahyar T. Moghaddam (SDU Software Engineering)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c84dae5d-5273-4348-85a7-b44cb586b4df` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种合规按构造（compliance-by-construction）的架构，将生成式人工智能（GenAI）与结构化的论证图（Argument Graph）结合，用于生成与证据关联、可审核的正式论证，支持高风险决策流程的可追溯性与合规性。

**💡 创新点**

创新点在于：① 将GenAI的生成结果视为“候选”而非最终结论，设置一个确定性验证核（validation kernel）确保每条主张都有可验证的证据、满足规则和程序完整性；② 采用检索增强生成（RAG）为模型提供权威证据并强制每个主张至少引用一条证据；③ 通过与W3C PROV标准对齐的证据账本，实现完整的决策链追踪；④ 用可验证的约束（如证据完整性、规则覆盖、无矛盾等）来保障生成的论证图满足监管要求。

**🔧 技术方法**

核心技术包括：结构化论证图（类似GSN、Toulmin/​Dung结构化推理）、检索增强生成（RAG）、确定性验证核（基于规则的约束检查）、W3C PROV标准的可追溯性记录、以及可选的联邦学习与安全聚合实现跨组织模型改进。

**📊 数据集**

数据集未明示具体公开数据集；系统使用的主要输入是：① 案例知识图（从原始案例数据提取的实体、事件与关系），② 通过RAG检索得到的权威证据片段（文档、记录、法规等），③ 预定义的规则/政策集合（如欧盟AI法规、组织内部合规规则）。

**📈 对比分析**

对比方法主要基于结构化约束的可执行性与可审计性，而非传统预测准确率。评估通过：① 在若干工作示例中展示验证核如何拒绝无证据或缺失程序的主张；② 通过日志和PROV记录验证完整的追踪链；③ 在示例中演示修复循环的有效性。性能方面，验证核的判定成本低，能在生成后即时过滤不合规内容，避免后期人工核对的开销。

**⚠️ 局限性**

局限性包括：① 需要对监管规则与政策进行形式化，可能无法捕捉所有语境细节；② 验证核只能保证结构完整性，无法评估规则本身的公平性或正确性；③ 依赖检索的证据质量；④ 对复杂冲突、模棱两可或对抗性推理的支持仍有限；⑤ 需要人工介入来确认“假设”节点的可接受性，增加审计负担。

---

## 261. SKILLFOUNDRY: Building Self-Evolving Agent Skill Libraries from Heterogeneous Scientific Resources

**arXiv ID:** 2604.03964 | [PDF](https://arxiv.org/pdf/2604.03964v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 262. Vocabulary Dropout for Curriculum Diversity in LLM Co-Evolution

**arXiv ID:** 2604.03472 | [PDF](https://arxiv.org/pdf/2604.03472v1)

**作者:** Jacob Dineen `[一作]` (Arizona State University), Ben Zhou `[通讯]` (Arizona State University)

**通讯引用:** 4731 | [OpenAlex ID](https://openalex.org/A5067460538)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在无监督的共进化自我对弈框架中引入词汇随机丢弃（Vocabulary Dropout）来保持问题生成器的多样性，并提升求解器的推理性能

**💡 创新点**

创新点在于把动作空间约束从奖励侧转移到输出词表上，使用可变硬性随机掩码阻止生成器陷入窄模板，类似游戏规则的作用

**🔧 技术方法**

采用R‑Zero共进化架构、GRPO策略梯度、词汇随机丢弃以及对抗式训练

**📊 数据集**

主要使用数学推理数据集，包括MATH、GSM8K、AMC、OlympiadBench、AIME 2024/2025 等；训练和评估均在 Qwen3‑4B 与 Qwen3‑8B 基础模型上进行

**📈 对比分析**

与基线（无词汇丢弃）对比，词汇丢弃在 8B 模型上平均提升 4.4 分（Pass@1），尤其在 AMC、AIME 等竞赛级任务上提升显著；在 4B 模型上适度掩码可提升但强掩码可能下降

**⚠️ 局限性**

局限性包括：掩码强度需随模型规模调节，强掩码在小模型或 proposer 能力超前时会降低效果；生成的题目仍可能缺乏可验证性，且无法完全过滤无效或难解题目

---

## 263. Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection

**arXiv ID:** 2604.04932 | [PDF](https://arxiv.org/pdf/2604.04932v1)

**作者:** Yang Li `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Juan Cao `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究四分类的LLM生成文本检测，区分人类写作、LLM生成、LLM润色人类文本和人类润色LLM文本。

**💡 创新点**

通过创作者‑编辑双重建模，利用修辞结构理论构建逻辑图，提取EDU特征，并采用RGCN进行修辞引导信息传递，实现细粒度检测。

**🔧 技术方法**

采用Rhetorical Structure Theory解析、EDU提取、信息瓶颈投影、关系分解的RGCN、监督对比学习和根节点池化等技术。

**📊 数据集**

使用HART benchmark（含四类文本）并自行划分训练/验证/测试集。

**📈 对比分析**

与12个基线（如RoBERTa、CoCo、LF‑Motifs、DeTeCtive等）在Macro AUROC和TPR@1%FPR上比较，RACE在四类平均TPR@1%FPR上比最佳基线CoCo高3.36%，整体性能优于所有基线。

**⚠️ 局限性**

仅在单一数据集上实验，跨语言和跨领域性能未知，绝对性能仍有提升空间，且未考虑多轮编辑序列的影响。

---

## 264. ReinVBC: A Model-based Reinforcement Learning Approach to Vehicle Braking Controller

**arXiv ID:** 2604.04401 | [PDF](https://arxiv.org/pdf/2604.04401v1)

**作者:** Haoxin Lin `[一作]` (Nanjing University), Yang Yu `[通讯]` (Nanjing University)

**通讯引用:** 29433 | [OpenAlex ID](https://openalex.org/A5077909232)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出将离线模型基强化学习应用于车辆制动控制，先学习车辆动力学模型，再在该模型中使用SAC算法训练制动策略并部署至真实车辆。

**💡 创新点**

创新点在于：①利用因果图分解动力学模型并进行全时长模拟；②在离线数据中采用速度数据增强，使策略能在更高速度下决策；③通过奖励设计避免轮胎锁止并保持车身偏转；④不使用模型不确定性惩罚，直接在可靠模型上展开探索。

**🔧 技术方法**

主要技术包括：离线模型基强化学习、SAC策略优化、GRU+全连接网络动力学模型、观察堆叠、速度增强、奖励函数设计。

**📊 数据集**

使用了36条真实车辆制动轨迹（低速40km/h）采集的离线数据，涵盖高黏附、低黏附和分裂摩擦三种路面，数据来自规则策略和随机策略。

**📈 对比分析**

与原厂ABS、规则采集策略和无控制直行制动对比；在分布内测试中制动距离和偏转均优于直行制动，接近ABS；在分布外的硬件回路仿真和真实测试中保持零锁止、偏转小，制动距离与ABS相当。

**⚠️ 局限性**

局限性：仅在一台车辆上验证，未完整对比原厂ABS；对高于100km/h高速制动性能未知；部署需移植至车载微控制器，计算资源受限；离线数据仅低速，离散化频率影响模型精度。

---

## 265. GeoBrowse: A Geolocation Benchmark for Agentic Tool Use with Expert-Annotated Reasoning Traces

**arXiv ID:** 2604.04017 | [PDF](https://arxiv.org/pdf/2604.04017v1)

**作者:** Xinyu Geng `[一作]` (Hong Kong University of Science and Technology), Yi R. Fung `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 663 | [OpenAlex ID](https://openalex.org/A5029408111)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 GeoBrowse 这套两级地理定位基准，并设计 GATE 代理工作流，用于评估多步视觉与文本工具协同推理。

**💡 创新点**

创新点在于：①将弱视觉线索与 BrowseComp 风格的多跳检索结合成两级任务；②提供专家级逐步追踪作为评价标准；③统一 Think‑with‑Image 与知识检索工具，形成完整的 Agentic 工作流。

**🔧 技术方法**

技术手段包括 ReAct‑style 交互式工具调用、Python sandbox 视觉工具（crop、rotate、super‑resolution 等）与知识工具（Web 文本/图像搜索、网页浏览、代码解释器）以及多模态大语言模型。

**📊 数据集**

使用了 300 个来自公开地理视频的实例（Level 1：199，Level 2：101），覆盖全球 6 大洲，且每个实例都配有专家标注的视觉线索与推理轨迹；与 MP‑16、GLDv2、VIGOR、GeoVista 等现有数据集进行对比。

**📈 对比分析**

采用 pass@1（准确率）对 12 大多模 LLM（如 GPT‑4o、Gemini‑3‑Pro、Claude‑4.5‑Opus 等）和 3 个开源代理（OmniSearch、WebWatcher、PyVision）进行基准；在 Level 1 中 GATE+Gemini‑3‑Pro 达到 48.2%，在 Level 2 达到 34.7%，显著优于直接推理和开源代理。

**⚠️ 局限性**

局限性包括：规模受限于人工审核，实例偏向国家级，全球分布不均；对弱模型工具输出噪声更易误导；以及缺乏更大规模自动化数据收集与质量闭环。

---

## 266. Early Stopping for Large Reasoning Models via Confidence Dynamics

**arXiv ID:** 2604.04930 | [PDF](https://arxiv.org/pdf/2604.04930v1)

**作者:** Parsa Hosseini `[一作]` (University of Maryland), Soheil Feizi `[通讯]` (University of Maryland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在链式推理过程中的自信度动态，并提出了CoDE-Stop方法实现无训练的推理早停。

**💡 创新点**

创新点在于结合递增置信阈值与趋势感知的退化分数，利用时间加权的置信度不稳定性实时判断何时停止推理，从而既能及时收敛正确路径又能中断无效长推理。

**🔧 技术方法**

使用了中间答案概率平均值作为置信度、趋势感知不稳定指标（v_i）、时间加权累计分数（w_i）以及递增阈值机制，对推理过程进行实时监控和决策。

**📊 数据集**

在AIME 2024/25、MATH500、GSM8K、GPQA‑Diamond等推理与科学基准上进行评估，覆盖多种模型（Qwen3‑4B、Qwen3‑14B、DeepSeek‑R1‑Distill‑Llama‑8B、Llama‑3.1‑Nemotron‑Nano‑8B）。

**📈 对比分析**

与Think or Not、DEER、EAT、RCPD、Answer Convergence等推理早停基线比较，CoDE-Stop在保持或略优准确率的同时，总令牌消耗降低25–50%，实现更优的准确率‑计算权衡。

**⚠️ 局限性**

局限性在于对置信度阈值和退化阈值的设定仍依赖经验调参；在低预算或极短推理阶段优势不明显，并且对极度不确定的模型输出可能产生误判。

---

## 267. Leveraging Gaze and Set-of-Mark in VLLMs for Human-Object Interaction Anticipation from Egocentric Videos

**arXiv ID:** 2604.03667 | [PDF](https://arxiv.org/pdf/2604.03667v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 268. Zero-Shot Quantization via Weight-Space Arithmetic

**arXiv ID:** 2604.03420 | [PDF](https://arxiv.org/pdf/2604.03420v1)

**作者:** Daniele Solombrino `[一作]` (Sapienza University of Rome), Emanuele Rodolà `[通讯]` (Sapienza University of Rome)

**通讯引用:** 7056 | [OpenAlex ID](https://openalex.org/A5087051832)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了如何将量化感知训练（QAT）获得的鲁棒性作为一种可转移的权重空间方向（量化向量）迁移到不同任务的 Vision Transformer（ViT）模型，并在不进行额外训练的情况下使用该向量对接收模型进行零样本补丁。

**💡 创新点**

提出了量化向量（Quantization Vector）概念，证明其能捕捉 QAT 对参数空间的整体调整，并提出了通过权重空间算术实现跨任务、零样本量化鲁棒性迁移的框架，避免了昂贵的 QAT 过程。

**🔧 技术方法**

利用对称通道权重量化、QAT 与 PTQ 的组合，采用权重空间算术和可调缩放因子 λ 对模型参数进行补丁；实验在 ViT-S/16、B/16、L/16 架构上实现。

**📊 数据集**

使用 22 种视觉分类数据集（如 ImageNet、CIFAR‑10/100、SVHN、EuroSAT、GTSRB、MNIST、SUN397、ResNet 等）作为接收与捐赠任务。

**📈 对比分析**

通过对比补丁后模型在 3‑bit 对称通道量化下的 Top‑1 准确率与未补丁 PTQ 模型的差异 Δ Acc 进行评估；实验表明补丁可提升高达 60% 的 PTQ 误差，且在 λ 调优后几乎消除负迁移，整体性能显著优于单纯 PTQ。

**⚠️ 局限性**

局限性包括：需要共享同一预训练初始化；需要小规模校准数据来调节 λ；仅针对权重量化，未验证与更复杂的 PTQ 技术（如激活量化、旋转预处理）交互的效果；实验范围仅在 ViT 模型上。

---

## 269. Computer Architecture's AlphaZero Moment: Automated Discovery in an Encircled World

**arXiv ID:** 2604.03312 | [PDF](https://arxiv.org/pdf/2604.03312v1)

**作者:** Karthikeyan Sankaralingam `[一作]` `[通讯]` (NVIDIA Research), Karthikeyan Sankaralingam (NVIDIA Research)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了自动化“想法工厂”架构，通过LLM驱动的递归生成、分层评估和部署反馈循环，加速芯片架构创新；

**💡 创新点**

核心创新在于将硬件架构视为可推理的问题空间，让LLM生成可验证的结构性机制，而非仅进行参数搜索；

**🔧 技术方法**

技术采用大型语言模型（Claude Opus/Code）实现方案生成、评估与代码生成，并结合多层评估管道（从符号推理到模拟再到RTL）和持续的性能遥测反馈；

**📊 数据集**

实验使用了2025-2026 ISCA/HPCA会议的85篇机制论文以及20篇历史基准论文作为验证集，并通过部署遥测数据校准模型；

**📈 对比分析**

与传统人类驱动设计相比，实验显示LLM生成方案的成功率达到95%，其中48%为全新可行方案，且评估速度提升数百倍，能够在数周内筛选出顶级设计；

**⚠️ 局限性**

主要限制包括对首代芯片缺乏反馈导致评估误差、需要大量高质量遥测数据、对新颖结构的验证与制造可行性仍需人工干预，以及组织与文化的转型难度。

---

## 270. Comparing Human Oversight Strategies for Computer-Use Agents

**arXiv ID:** 2604.04918 | [PDF](https://arxiv.org/pdf/2604.04918v1)

**作者:** Chaoran Chen `[一作]` (University of Notre Dame), Toby Jia-Jun Li `[通讯]` (University of Notre Dame)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文在真实网页环境中，对四种 LLM 计算机使用代理（CUA）的监督策略（Risk‑Gated、Supervisory Co‑Execution、Action Confirmation、Structurally Enriched）进行对比研究，评估其对用户监督体验和行为效果的影响。

**💡 创新点**

创新点在于：①提出由委托结构与参与级别两维构成的监督设计空间，提供结构化的比较框架；②通过实验验证计划级监督能显著降低问题行为出现率，但并不必然提升干预成功率；③揭示监督效果与任务情境、风险可识别性之间的交互关系。

**🔧 技术方法**

技术手段包括：使用 Gemini 3.1 Flash‑Lite 生成 LLM 计算机使用代理；通过 Chrome 扩展实现四种监督界面；采集用户交互日志、主观问卷与行为指标；采用混合效应模型与 GEE 进行统计分析。

**📊 数据集**

数据集与实验材料：48 名 Prolific 参与者完成 6 个真实网站任务（金融、旅游、公共福利、餐饮、娱乐、在线评价），每个任务嵌入一类风险（隐私泄露、提示注入、暗模式）。

**📈 对比分析**

比较方法：在 48 人 within‑subject 交叉设计下，使用混合效应模型评估主观指标；使用 GEE 对二元行为指标（攻击出现、干预成功、最终成功）进行比较。结果显示：计划级策略（Supervisory Co‑Execution、Structurally Enriched）将攻击出现率降低至 60‑74%（相比非计划级约 90%），但干预成功率仅为 9‑26%，最终成功率整体仍在 55‑67% 之间。

**⚠️ 局限性**

局限性：①风险类型与任务严重性未完全分离，导致上下文效应难以单独解释；②干预成功指标仅在问题行为已出现时计量，忽略了早期预防情况；③结构化增强条件包含多种机制，难以解析单一因素贡献；④实验仅在受控环境中进行，缺乏长期真实使用情境的数据。

---

## 271. Sampling Parallelism for Fast and Efficient Bayesian Learning

**arXiv ID:** 2604.04736 | [PDF](https://arxiv.org/pdf/2604.04736v1)

**作者:** Asena Karolin Özdemir `[一作]` (Karlsruhe Institute of Technology), Charlotte Debus `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种称为采样并行化的并行策略，专门针对采样式贝叶斯学习（如贝叶斯神经网络和蒙特卡洛Dropout）中样本评估的高内存与计算成本问题。

**💡 创新点**

创新点在于：①将样本评估分布到多张GPU上，实现内存压缩与训练加速；②在同一批次上每个GPU使用不同随机种子实现独立数据增强，提升收敛速度；③与分布式数据并行（DDP）结合形成混合并行；④通过实验验证了在大模型和大数据集上的可扩展性与优势。

**🔧 技术方法**

使用的技术包括：变分推理（VI）训练的贝叶斯神经网络、蒙特卡洛Dropout、Vision Transformer (ViT)、多层感知机 (MLP)、SWIN Transformer；采样并行化算法、梯度全归约（all-reduce）、PyTorch 及 torch.distributed；对loss的梯度同步与可选的统计聚合。

**📊 数据集**

使用的数据集包括：CIFAR‑10（图像分类）、ENTSO‑E 德国电力负荷时间序列（时间序列预测）、ERA5 大气数据（天气预报）。

**📈 对比分析**

与传统DDP相比，采样并行在“比例采样”实验中实现近乎完美的效率（>90%）；在固定样本规模实验中，DDP在单个GPU负载低时更快，但采样并行因增强多样性每个epoch收敛更快，整体wall‑clock时间可与DDP相当或略优；在大模型SWIN Transformer上，采样并行是实现多样本训练的唯一可行方案。

**⚠️ 局限性**

局限性包括：需要在每个GPU重复加载同一批数据，导致数据加载开销；对某些非线性loss可能只能近似梯度同步；在小模型或小批量下通信开销占比高，效率低；若需精确统计（如标准差）需额外通信。

---

## 272. A Multimodal Foundation Model of Spatial Transcriptomics and Histology for Biological Discovery and Clinical Prediction

**arXiv ID:** 2604.03630 | [PDF](https://arxiv.org/pdf/2604.03630v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 273. Optimal Lower Bounds for Symmetric Modular Circuits

**arXiv ID:** 2604.04760 | [PDF](https://arxiv.org/pdf/2604.04760v1)

**作者:** Benedikt Pago `[一作]` `[通讯]` (University of Cambridge), Benedikt Pago (University of Cambridge)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究了在使用模计数门（m‑gates）的可对称（完全对称与嵌套块对称）电路中，计算 n‑ary AND 函数的最小规模与深度，给出了匹配的上界与下界，证明了深度为 2 的构造已是最优。

**💡 创新点**

创新点在于：①使用群论的支撑（support）概念在任意深度的对称电路上获得周期性上界；②在嵌套块对称情况下引入块级支撑（blockwise support）并证明相应的周期性与大小下界；③得出完全对称电路的大小下界为 2^{Ω(n^{1/r}·log n)}，并构造匹配的深度‑2 方案；④扩展到嵌套块对称电路，证明其大小下界为 2^{Ω(k_{max}^{1/r}·log k_{max})}，并给出递归深度‑2h 方案。

**🔧 技术方法**

主要技术包括：群论支撑与块级支撑；对称性下的周期性分析（利用模组合数的周期长度）；对电路层次的归纳证明；递归构造与 pq‑表达式；对称电路的自动化与刚性化。

**📊 数据集**

无实验数据集；该工作为纯理论复杂度分析。

**📈 对比分析**

通过对比已有的指数下界（如 Håstad 的切换引理对布尔门的下界）和最近的对称电路上界（Idziak 等人深度‑2 构造），本文给出了与已知上界完全匹配的下界，证明了在对称约束下无法进一步压缩规模或深度。

**⚠️ 局限性**

局限性：仅适用于有至少两个素因子的模数 m；对一般非对称电路的结论仍未得到；嵌套块对称电路的深度‑(h+1) 构造的尺寸与理论下界之间仍存在 1/(r‑1) 的差距，是否可消除仍是开放问题。

---

## 274. GPIR: Enabling Practical Private Information Retrieval with GPUs

**arXiv ID:** 2604.04696 | [PDF](https://arxiv.org/pdf/2604.04696v1)

**作者:** Hyesung Ji `[一作]` (Seoul National University), Jung Ho Ahn `[通讯]` (Seoul National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在本研究中，作者实现了一个名为GPIR的GPU加速Private Information Retrieval（PIR）系统，支持多客户端批处理和多GPU扩展。

**💡 创新点**

创新点在于提出阶段感知混合执行模型、转置布局的RowSel优化、p维度流水线和多GPU通信策略，以突破缓存容量壁垒和数据布局冲突，实现大幅性能提升。

**🔧 技术方法**

采用了NVIDIA RTX 5090和H100 GPU，结合多核CUDA内核、NTT、RNS、GEMM、内存层次调度、CUDA Graphs、NVLink/PCIe高速互连等技术。

**📊 数据集**

使用人工合成的数据库规模为1GB、2GB、4GB和8GB（记录16KB，N=2^12）作为评估基准。

**📈 对比分析**

通过与PIRonGPU、ShiftPIR等前沿GPU实现对比，GPIR在不同规模下的查询吞吐量提升至305.7倍，单GPU端点QPS最高可达2.3×，多GPU场景提升1.7-1.9×。

**⚠️ 局限性**

主要限制包括仍需单服务器架构、对高带宽NVLink依赖较大、在PCIe环境下扩展层通信有显著开销，以及未覆盖非基于格哈姆密码的PIR协议。

---

## 275. Context is All You Need

**arXiv ID:** 2604.04364 | [PDF](https://arxiv.org/pdf/2604.04364v1)

**作者:** Jean Erik Delanois `[一作]` (University of California), Maxim Bazhenov `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 CONTXT 的轻量级上下文适配方法，通过在网络内部层面加减线性上下文向量来实现模型在域漂移和生成任务中的自适应。

**💡 创新点**

创新点在于：无需额外训练或微调，仅利用单个前向传播得到的上下文向量，采用简单的向量相减与加权操作即可在判别式和生成式模型中注入或去除上下文信息；方法通用、可解释且计算成本极低。

**🔧 技术方法**

技术实现包括：计算输入特征与预先平均的上下文特征差向量；对差向量乘以可调缩放系数后加到当前特征上；支持多上下文组合；仅需一次前向传播和向量运算，适用于 CNN、Transformer 等主流架构。

**📊 数据集**

实验数据集：图像分类使用 PACS 与 CCT；生成式实验使用 Llama‑3 8B/70B，并在 Yelp 评论数据上进行情感转换评估；此外还在“Cow on a beach”案例中展示原型效果。

**📈 对比分析**

与基线预训练模型对比，CONTXT 在 OOD 分类任务中平均提升约 8–10% 的准确率（PACS 最高 20% 的卡通域提升，CCT 最高 25% 的新域提升）；在 LLM 生成任务中可将情感翻转率提升至 80% 同时保持 Self‑BLEU；方法仅增加极低的推理延迟与计算量。

**⚠️ 局限性**

局限性包括：需要事先知道或能够快速估计测试时的上下文；方法依赖线性可解释性，可能无法捕捉高度非线性或复杂的上下文关系；在极端域漂移或缺乏代表性上下文样本时效果受限。

---

## 276. Surface Quadrilateral Meshing from Integrable Odeco Fields

**arXiv ID:** 2604.03889 | [PDF](https://arxiv.org/pdf/2604.03889v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

---

## 277. Mechanism and Communication Co-Design for Differentially Private Energy Sharing

**arXiv ID:** 2604.04125 | [PDF](https://arxiv.org/pdf/2604.04125v1)

**作者:** Yingshuo Gu `[一作]` (Chinese University of Hong Kong), Yue Chen `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 16755 | [OpenAlex ID](https://openalex.org/A5100454793)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种在 OTA MIMO 通信下的差分隐私能量共享机制，并给出相应的收敛证明

**💡 创新点**

创新点在于将市场机制与无线通信共设计，利用无线信道噪声降低人工扰动并实现隐私放大；并通过差分隐私噪声调度在迭代均衡搜索中保持收敛

**🔧 技术方法**

采用 OTA 聚合、MIMO 复用、差分隐私噪声注入、迭代 Nash 级联算法以及 3GPP TR 38.901 实际信道仿真

**📊 数据集**

使用仿真数据：三方可再生能源生产/消费成本为二次函数，Rayleigh 随机信道及 Sionna 生成的 3GPP 城市微小区信道

**📈 对比分析**

与理想信道、正交多址（TDMA/FDMA）以及不加噪声的 OTA 方案对比；实验显示人工噪声可减少约 50% 以上，隐私预算 epsilon 在 OTA 下比正交方式低 10%~25%，算法在所有隐私级别下均能收敛至近似 Nash 均衡

**⚠️ 局限性**

局限性：假设完美 CSI、静态信道、单一诚实但好奇的基站攻击者，未考虑动态信道、CSI 误差、合作/主动攻击以及能量共享规模更大时的可扩展性

---

## 278. Reimagining RAN Automation in 6G: An Agentic AI Framework with Hierarchical Online Decision Transformer

**arXiv ID:** 2604.03908 | [PDF](https://arxiv.org/pdf/2604.03908v1)

**作者:** Md Arafat Habib `[一作]` (University of Ottawa), Melike Erol-Kantarci `[通讯]` (University of Ottawa)

**通讯引用:** 7937 | [OpenAlex ID](https://openalex.org/A5089891162)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种面向6G RAN自动化的Agentic AI框架，利用超代理结合分层在线决策变换器(H-ODT)协调资源分配、网络应用与自愈模块，实现自然语言驱动的意图执行；

**💡 创新点**

创新点在于将Agentic Retrieval‑Augmented Generation与双层意图验证相结合，采用分层在线决策变换器实现持续自适应规划，并嵌入自愈机制以实现零触摸运维；

**🔧 技术方法**

技术手段包括IA^3微调的大语言模型、分层在线决策变换器(H-ODT)、A‑RAG检索生成循环、Autoformer/Informer/Mamba时序预测器，以及DQN/HRL强化学习调度器；

**📊 数据集**

数据集由自构造的5G/6G调度意图–动作对、检索查询与自愈日志组成，并在MATLAB/Sim仿真平台生成的真实流量与网络状态数据上进行训练与评估；

**📈 对比分析**

与三种基线（离线决策变换器、HRL、启发式非Agentic）在吞吐量、时延、能效等指标上对比，实验表明吞吐量提升32.9%，时延下降60.9%，能效提升三倍，意图验证准确率88.5%，自愈恢复率约90%；

**⚠️ 局限性**

局限性包括对仿真环境的依赖、对大量标注意图数据的需求、在线学习对计算资源的负担、以及在真实网络中可能出现的分布漂移和模型泛化问题。

---

## 279. Do No Harm: Exposing Hidden Vulnerabilities of LLMs via Persona-based Client Simulation Attack in Psychological Counseling

**arXiv ID:** 2604.04842 | [PDF](https://arxiv.org/pdf/2604.04842v1)

**作者:** Qingyang Xu `[一作]` (Monash University), Zongyuan Ge `[通讯]` (Monash University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种基于心理治疗会话的动态人格模拟攻击框架PCSA，用于揭示大型语言模型在精神健康交互中的安全缺陷

**💡 创新点**

创新点在于将真实心理咨询语料构建多样化客户人格，并通过多轮心理策略（安抚、专业引用、理性化、隐喻表达）动态演进，从而突破传统静态攻击的检测壁垒

**🔧 技术方法**

采用多轮交互式生成、Persona注入、策略优化循环以及基于GPT-4o-mini的在线评估器与多维安全评判器来驱动攻击与检测

**📊 数据集**

使用公开的CBT对话语料库（Cactus、CBT-Bench、Cheeseburger Therapy）构建客户模型，并针对自伤、认知失调、饮食失调等精神危机设计攻击目标集

**📈 对比分析**

与四种主流多轮“越狱”基线（CoA、AMA、Crescendo、Actor-Attack）在8款LLM上进行对比，PCSA在ASR与安全分数上均显著优于基线，且保持极低的Perplexity与检测率

**⚠️ 局限性**

局限性包括仅针对文本交互、未涵盖多模态环境、缺乏对策与缓解方案、攻击示例受限于公开数据，未来需研究更鲁棒的安全防护与多模态适配

---

## 280. Identification for Colored Gaussian Channels

**arXiv ID:** 2604.04674 | [PDF](https://arxiv.org/pdf/2604.04674v1)

**作者:** Mohammad Javad Salariseddigh `[一作]` `[通讯]` (Technical University of Darmstadt), Mohammad Javad Salariseddigh (Technical University of Darmstadt)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了受相关噪声与符号间干扰影响的离散时间高斯信道在峰值功率约束下的识别容量。

**💡 创新点**

创新点在于证明即使ISI记忆长度和噪声协方差奇异值谱随码字长度以多项式方式增长，识别码本仍可实现超指数规模，并给出基于κ和μ的上下限。

**🔧 技术方法**

采用了Mahalanobis距离解码、球面封装分析、奇异值与Rayleigh商的矩阵论，以及白化变换与χ²分布性质。

**📊 数据集**

无实验数据集，所有结果均为信息理论解析。

**📈 对比分析**

通过与传统无ISI白噪声模型对比，所得识别容量上界为1+κ+μ/2，下界为1-2(κ+μ)/4，表明对ISI和色噪声的容忍度。

**⚠️ 局限性**

局限在于假设协方差矩阵满足多项式良态条件且CIR稳定，且仅考虑峰值功率约束，未讨论多用户、频谱缺陷或快速衰落情形。

---

## 281. Mestra: Exploring Migration on Virtualized CGRAs

**arXiv ID:** 2604.04694 | [PDF](https://arxiv.org/pdf/2604.04694v1)

**作者:** Agamemnon Kyriazis `[一作]` (Computing Systems Laboratory - National Technical University), Dionisios Pnevmatikatos `[通讯]` (Computing Systems Laboratory - National Technical University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套完整的CGRA多租户虚拟化系统，支持动态调度、资源分配以及运行时的无状态和有状态迁移，以缓解碎片化问题；

**💡 创新点**

创新点在于：1）引入灵活可合并的虚拟CGRA区域，实现真正的空间共享；2）设计了两种迁移机制（无状态和有状态）来主动消除碎片；3）结合了紧耦合控制器和快照恢复技术，首次在CGRA上实现有状态迁移；

**🔧 技术方法**

核心技术包括：基于网格NoC的分区CGRA架构、动态部分重配置（DPR）、紧耦合控制器、快照捕获/恢复、遗传算法生成碎片化工作负载、Python仿真和Alveo-U280硬件原型；

**📊 数据集**

使用的数据集主要是PolyBench基准、BLAS（GEMM、2MM、SAXPY等）以及常见机器学习算子（ReLU、卷积等），共计64个随机生成的作业；

**📈 对比分析**

通过硬件仿真与Python仿真两种评估方式，与单一核单体CGRA进行对比，结果显示：在碎片化环境下，状态迁移将平均等待时间降低约30%，尾部延迟降低约30%，而无状态迁移仅提升约3%；整体多租户调度使工作负载完工时间（makespan）提升约21%；

**⚠️ 局限性**

限制包括：迁移操作仍带来一定的能耗和时延开销，状态迁移需要对各PE的关键寄存器进行快照，增加了设计复杂度；碎片化治理依赖于贪心压缩策略，缺乏全局最优；硬件实现基于单一FPGA平台，规模和内存通道有限，可能影响更大规模系统的可扩展性。

---

## 282. SASAV: Self-Directed Agent for Scientific Analysis and Visualization

**arXiv ID:** 2604.03406 | [PDF](https://arxiv.org/pdf/2604.03406v1)

**作者:** Jianxin Sun `[一作]` (University of Nebraska-Lincoln), Hongfeng Yu `[通讯]` (University of Nebraska-Lincoln)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个完全自主、无提示的科学可视化智能体SASAV，能够从原始体数据自动完成数据分析、知识检索、颜色与不透明度传递函数优化、视角与动画轨迹规划，并输出静态图、动画及交互式可视化。

**💡 创新点**

① 完全自主无先验知识的agentic流程；② 多代理并行+批量MLLM推理提升效率；③ 结合知识检索与区域重要性提示；④ 通过语义分析自动生成颜色/不透明度传递函数；⑤ 视角评估与轨迹规划。

**🔧 技术方法**

前沿多模态LLM（如GPT‑5.4）、OpenAI Agents SDK、LangChain + RAG本地知识库、OpenAI Web Search、Vision‑Language模型、并行多代理、VTK渲染、Fibonacci视角球采样、Catmull‑Rom插值动画。

**📊 数据集**

五个体数据集：AbdomenAtlas 1.0 Mini（人体腹部实测）、Chameleon（爬行动物实测）、Miranda、Flame、Richtmyer（三者为模拟流体/热力学数据）。

**📈 对比分析**

采用定性评估、专家反馈、时间消耗与token使用对比。对TF建议、视角选择等步骤进行时长与token统计，比较本地与HPC加速效果。结果表明：对模拟数据TF建议耗时短，本地可完成常规渲染，HPC可将时间降低数倍；整体性能可接受。

**⚠️ 局限性**

① 受LLM随机性影响，颜色一致性不稳定；② 对模拟数据的颜色表现不如实测数据；③ 仅支持DVR与等值面，无法处理多变量、张量、时变数据或点云；④ 需要较大算力和高成本模型；⑤ 需要进一步完善知识库和视频理解能力。

---

## 283. Graph-to-Frame RAG: Visual-Space Knowledge Fusion for Training-Free and Auditable Video Reasoning

**arXiv ID:** 2604.04372 | [PDF](https://arxiv.org/pdf/2604.04372v1)

**作者:** Songyuan Yang `[一作]` (National University of Defense Technology), Nong Xiao `[通讯]` (Sun Yat-sen University)

**通讯引用:** 4021 | [OpenAlex ID](https://openalex.org/A5023506057)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在视频推理中提出G2F-RAG，将检索到的结构化知识转换为单帧视觉证据，直接与原视频拼接以进行联合推理；

**💡 创新点**

创新点在于将检索知识以视觉空间形式呈现，避免跨模态注意力竞争和信息冗余，从而实现无训练、可审计的知识融合；

**🔧 技术方法**

采用多智能体框架：离线构建问题无关的视频知识图，在线时由路由、检索、渲染三阶段智能体动态获取并生成单帧视觉图；

**📊 数据集**

在八大公开基准上进行评测，包括MVBench、MMBench-Video、VideoMME、TempCompass、VSIBench、WildVideo、VideoMMMU和MLVU；

**📈 对比分析**

与传统文本或多片段检索RAG方法（Video-RAG、Vgent）及多种大模型（Qwen2.5-VL、InternVL3.5、LLaVA-Video等）比较，G2F-RAG在大多数任务上提升3–7个百分点，并在知识密集或开放世界场景表现尤为突出；

**⚠️ 局限性**

局限性主要体现在：依赖离线图构建的质量、对图可视化表达的限制以及对外部知识检索工具的依赖，且对极其抽象或文本繁杂的知识仍可能表现不佳。

---

## 284. A Logical-Rule Autoencoder for Interpretable Recommendations

**arXiv ID:** 2604.04270 | [PDF](https://arxiv.org/pdf/2604.04270v1)

**作者:** Jinhao Pan `[一作]` (George Mason University), Ziwei Zhu `[通讯]` (George Mason University)

**通讯引用:** 1493 | [OpenAlex ID](https://openalex.org/A5019994221)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种可解释的逻辑规则自编码器LIA，用于协同过滤推荐。

**💡 创新点**

创新点在于引入可学习的逻辑规则层，规则神经元通过门参数自适应选择AND/OR，并用权重符号编码否定，实现功能完备且参数高效的逻辑表达。

**🔧 技术方法**

采用了可微分的逻辑激活函数、梯度嫁接、签名权重编码、线性重构层，以及多项式交叉熵训练。

**📊 数据集**

使用了ML100k、ML1M和Yelp三个公开用户-物品交互数据集。

**📈 对比分析**

与MF、Autoencoder、MultVAE等基线进行对比，LIA在NDCG@20上分别取得0.3578、0.3263、0.0982，均超过所有基线，同时保持可解释性和相近的训练/推理效率。

**⚠️ 局限性**

局限性包括对规则数量敏感、规则解释可能过于简单或冗余，以及在稀疏数据上需要更大规则集或更复杂的学习机制。

---

## 285. Justified or Just Convincing? Error Verifiability as a Dimension of LLM Quality

**arXiv ID:** 2604.04418 | [PDF](https://arxiv.org/pdf/2604.04418v1)

**作者:** Xiaoyuan Zhu `[一作]` (University of Southern California), Steven Wu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3486 | [OpenAlex ID](https://openalex.org/A5001070941)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了错误可验证性（Error Verifiability）概念和Balanced Verifiability指标，并评估多种模型与方法在高风险场景下是否能提升用户对答案正确性的判断。

**💡 创新点**

首次将可验证性作为独立质量维度，证明提升准确率并不等同于可验证性提升，并提出两种结合外部信息的提升技术：Reflect‑and‑Rephrase（RR）和Oracle‑Rephrase（OR）。

**🔧 技术方法**

利用LLM‑as‑a‑judge进行大规模可验证性评估；采用推理链重写、外部事实校验、语言自信度校准等技术进行实验。

**📊 数据集**

在数学推理任务使用GSM8K和MATH500，在事实知识问答任务使用MMLU、MMLU‑Pro和TruthfulQA。

**📈 对比分析**

对比模型规模、后训练阶段、风格重写、置信度校准等，发现后训练和模型扩容不提升可验证性；RR在数学任务中平均提升0.03–0.08，OR在事实QA中提升0.05–0.07的Balanced Verifiability。

**⚠️ 局限性**

方法仅针对数学与事实QA，缺乏跨域通用性；LLM‑as‑a‑judge与真实用户对齐度有限；未提供训练时直接优化可验证性的方法。

---

## 286. DAGAF: A directed acyclic generative adversarial framework for joint structure learning and tabular data synthesis

**arXiv ID:** 2604.04290 | [PDF](https://arxiv.org/pdf/2604.04290v1)

**作者:** Hristo Petkov `[一作]` (University of Strathclyde), Feng Dong `[通讯]` (University of Strathclyde)

**通讯引用:** 8737 | [OpenAlex ID](https://openalex.org/A5027381129)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了DAGAF框架，能够同时进行因果结构学习和表格数据生成；

**💡 创新点**

整合了多种可识别的功能因果模型（ANM、LiNGAM、PNL）并在单一模型中联合学习因果结构与生成器；

**🔧 技术方法**

使用Wasserstein-1对抗损失、MSE、KLD、MMD等多目标损失，以及基于增强拉格朗日的DAG-Notears-MLP实现无环约束；

**📊 数据集**

在多种模拟数据（线性、非线性、PNL）和真实基准数据集（Sachs、Child、Alarm、Hailfinder、Pathfinder）上进行评估；

**📈 对比分析**

与现有11种DAG学习方法（如DAG-WGAN、DAG-Notears、DAG-GNN等）和数据生成方法比较，DAGAF在SHD上分别提升了47%、11%、5%、7%等，且生成的数据与真实数据在分布、相关性、PCA等指标上高度一致；

**⚠️ 局限性**

仅验证了可识别模型（ANM、LiNGAM、PNL），对离散或更复杂模型的适用性未探究；计算复杂度高（O(d³)），对非i.i.d.、缺失值和离散变量的鲁棒性有限。

---

## 287. Real-Time Projected Adaptive Control for Closed-Chain Co-Manipulative Continuum Robots

**arXiv ID:** 2604.04286 | [PDF](https://arxiv.org/pdf/2604.04286v1)

**作者:** Rana Danesh `[一作]` (Toronto Metropolitan University), Farhad Aghili `[通讯]` (Concordia University)

**通讯引用:** 3058 | [OpenAlex ID](https://openalex.org/A5000948875)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了面向闭环共操纵连续机器人（CCR）的投影自适应控制框架，实现对机器人与柔性物体耦合动力学的不确定参数在线补偿，从而在任务空间实现实时的调节和轨迹跟踪。

**💡 创新点**

首创将CCR动力学用几何变量应变（GVS）表示，借助Pfaffian速度约束将其投影到约束一致运动子空间，并基于参数线性回归设计Lyapunov稳定的自适应控制器，实现机器人与柔性物体参数的在线识别。

**🔧 技术方法**

使用了GVS建模、Lie群几何学、Pfaffian速度约束、正交投影、参数线性回归、Lyapunov自适应控制、显式欧拉MEX实现以及张力驱动实验平台与Vicon测量。

**📊 数据集**

未使用公开数据集，而是基于仿真中设定的两臂张力驱动CCR（弹簧钢臂与钛镍柔性杆）参数和真实实验平台的Vicon三维位置信号进行验证。

**📈 对比分析**

通过与无自适应模型控制器和基线投影反馈控制器在仿真与实验中的对比（RMSE、总变化量、饱和事件等指标），自适应控制器在误差、平滑性和饱和事件上均优于其他两种方案。

**⚠️ 局限性**

仅适用于无外部接触的自由空间共操纵，且对张力长度映射精度要求较高；未来需扩展至接触力控制和更复杂不确定性下的学习增强自适应方法。

---

## 288. Restless Bandits with Individual Penalty Constraints: A New Near-Optimal Index Policy and How to Learn It

**arXiv ID:** 2604.04101 | [PDF](https://arxiv.org/pdf/2604.04101v1)

**作者:** Nida Zamir `[一作]` (Texas A&M University), I-Hong Hou `[通讯]` (Texas A&M University)

**通讯引用:** 1753 | [OpenAlex ID](https://openalex.org/A5060672325)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了Restless Multi‑Armed Bandit（RMAB）框架的扩展，加入了个体惩罚约束，能够在动态无线网络中同时最大化总奖励并满足每个用户的能耗、激活频率或信息新鲜度等多样化约束。

**💡 创新点**

创新点在于提出了“Penalty‑Optimal Whittle (POW) 指数”——一个只依赖于单个臂的状态与惩罚约束、与系统规模无关的新指数，并证明其在流体极限下渐近最优；此外，还设计了基于深度强化学习的 DeepPOW 算法，实现在线学习 POW 指数。

**🔧 技术方法**

技术包括：拉格朗日分解与两维乘子耦合以构造 POW 指数；指数可计算性分析与索引可行性检验；深度演员‑评论家框架，用于学习指数与对应的最优拉格朗日乘子；以及多种模拟与基准比较（Whittle、FaWT、DPP、DeepTOP 等）。

**📊 数据集**

实验采用三类人工生成的数据集：① 通过吞吐量最大化与激活约束的 IoT/5G 调度问题；② 远程感知中的能耗约束与 AoI 最小化；③ 吞吐量最大化与服务规律约束（AoI 约束）。每类数据集均通过离散状态空间、随机转移、已知或未知动态生成。

**📈 对比分析**

与传统 Whittle、FaWT、DPP 等基准比较，POW 在有限系统中持续逼近流体松弛上界，且约束违背率低；DeepPOW 在未知动态场景中实现了与 POW 相当的奖励水平，同时维持零或近零约束违背；相对 DeepTOP 与 DeepTOP+FaWT，DeepPOW 在奖励–约束平衡上显著优于。

**⚠️ 局限性**

局限性包括：理论证明仅在流体极限和指数可行性成立时成立；指数计算需求解线性规划，可能在极大状态空间上昂贵；当前仅考虑单一资源约束与平均奖励设定，未扩展至多资源或平均收益形式；在部分可观测或高维状态下的学习效果尚未验证。

---

## 289. Delayed Homomorphic Reinforcement Learning for Environments with Delayed Feedback

**arXiv ID:** 2604.03641 | [PDF](https://arxiv.org/pdf/2604.03641v1)

**作者:** Jongsoo Lee `[一作]` (POSTECH), Soohee Han `[通讯]` (POSTECH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于MDP同构的延迟强化学习框架（DHRL），通过信念等价性压缩状态空间，消除延迟导致的状态爆炸；

**💡 创新点**

核心创新在于利用MDP同构理论构造抽象MDP，使得延迟问题转化为无延迟问题，既保证最优性又显著降低样本复杂度；

**🔧 技术方法**

使用MDP同构、信念等价性、松弛的双侧同构、深度Actor-Critic（D²HPG）以及传统值迭代（DHVI）等技术；

**📊 数据集**

主要在MuJoCo连续控制基准（HalfCheetah, Ant, Hopper, Walker2d, Humanoid, InvertedPendulum）以及4×4网格世界进行验证；

**📈 对比分析**

与naïve SAC、Augmented SAC、Delayed SAC、BPQL、VDPO等基线对比，D²HPG在所有任务和不同延迟下均取得最高或最接近最高平均回报，尤其在长延迟（Δ=20）时优势明显；

**⚠️ 局限性**

局限在于假设固定延迟，对随机延迟的适用性受限，虽然可通过已知最大延迟的保守近似扩展，但仍可能导致样本复杂度提升。

---

## 290. Is Prompt Selection Necessary for Task-Free Online Continual Learning?

**arXiv ID:** 2604.04420 | [PDF](https://arxiv.org/pdf/2604.04420v1)

**作者:** Seoyoung Park `[一作]` (Sungkyunkwan University), Hankook Lee `[通讯]` (Sungkyunkwan University)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5058927640)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对任务无界在线持续学习（Task‑Free Online Continual Learning）提出了一种简洁有效的方案——SinglePrompt，该方法在每个自注意力层中仅使用一个提示（prompt），并通过余弦相似度计算 logits 以及批次级别的 logits 掩码来抑制遗忘。

**💡 创新点**

创新点在于重新审视并否定了当前主流的 prompt 选择策略：实验表明 prompt 选择往往与输入语义无关，导致知识碎片化。SinglePrompt 通过消除 prompt 选择，仅保留单一 prompt 以及改进的分类头，既减少了可学习参数，又显著提升了性能。

**🔧 技术方法**

核心技术包括：① prefix‑tuning（在键/值序列前置可学习提示）用于快速在线适应；② 余弦相似度 logits 以消除类权重范数不平衡导致的遗忘；③ 批次级别的 logits 掩码防止未出现类的原型被更新。实验还对不同 PEFT 适配器（LoRA、AdaptMLP、Prompt tuning）和 logits 类型进行了 ablation。

**📊 数据集**

使用的数据集为三个视觉基准：CIFAR‑100、Tiny‑ImageNet（200类）以及 ImageNet‑R（200类），并在 Si‑Blurry 任务无界设定下评估，进一步通过改变 disjoint 类比例（0%‑100%）验证模型鲁棒性。

**📈 对比分析**

与众多基线（ER、DER++、ER‑ACE、Rainbow Memory、CLIB、LwF、EWC、L2P、DualPrompt、MVP、MISA 等）在 A_auc、A_last、F_last 三项指标上进行对比。SinglePrompt 在 CIFAR‑100 上提升 A_last 6.55% 并保持 60% 参数量缩减；在 Tiny‑ImageNet 与 ImageNet‑R 上亦分别提升 8.03% 与 14.96%；在不同 disjoint 比例下均稳健领先，说明方法在多种任务冲突场景下表现优异。

**⚠️ 局限性**

限制方面：① 只在 Si‑Blurry 设定下验证，未覆盖更为随机或极端的在线流；② 主要基于预训练 ViT‑B/16，缺乏对其他骨干或大规模数据（如完整 ImageNet）下的通用性验证；③ 虽减少参数但仍需一定记忆缓冲，未对极低存储预算下的表现做详细探讨。

---

## 291. VERT: Reliable LLM Judges for Radiology Report Evaluation

**arXiv ID:** 2604.03376 | [PDF](https://arxiv.org/pdf/2604.03376v1)

**作者:** Federica Bologna `[一作]` (Cornell University), Asma Ben Abacha `[通讯]` (Microsoft Healthcare and Life Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估并比较了多种大语言模型（LLM）评判指标，并在多模态放射报告上提出并验证了新的VERT评估方法。

**💡 创新点**

提出了基于LLM的VERT指标，在RadEval和RaTE-Eval数据集上相较于RADFACT、GREEN等传统指标提升了最高11.7%的相关性；同时展示了轻量化LoRA微调和线性回归集成可进一步提升性能。

**🔧 技术方法**

使用了Prompt设计（zero-shot、few-shot、公式/规则prompt、思考链）、LoRA低秩微调、线性回归集成以及系统化错误注入与分类分析等技术。

**📊 数据集**

使用了RadEval（胸部X光）和RaTE-Eval（9种模态、22个解剖区）的专家标注数据集。

**📈 对比分析**

通过Kendall τ与专家评分对比进行性能评估；VERT在RadEval上取得0.371（高于GREEN 0.332），在RaTE-Eval上取得0.447（高于GREEN 0.419）；线性回归集成可达0.568；LoRA微调实现约25%的性能提升，并将推理时间缩短37.2倍。

**⚠️ 局限性**

局限性包括：仅在有参考报告的数据上评估；对非关键错误（c–f）的检测和分类表现不足；思考链效果不稳定；不同模态之间的泛化仍有限；需要更多样本和外部验证。

---

## 292. HI-MoE: Hierarchical Instance-Conditioned Mixture-of-Experts for Object Detection

**arXiv ID:** 2604.04908 | [PDF](https://arxiv.org/pdf/2604.04908v1)

**作者:** Vadim Vashkelis `[一作]`, Natalia Trukhina `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了HI-MoE，一种层次化的场景-实例混合专家（Mixture-of-Experts）框架，用于DETR风格的目标检测；

**💡 创新点**

创新点在于两阶段路由：先通过场景路由器选出与图像场景一致的专家子集，再由实例路由器为每个检测查询分配该子集中的少量专家，提升实例级别的稀疏计算与专家专业化；

**🔧 技术方法**

使用了Transformer中的稀疏专家模块（MoE）替换FFN层，结合场景全局描述、查询级别门控、温度控制、负载平衡与多样性正则化等技术；

**📊 数据集**

主要在COCO数据集上进行实验，亦做了对LVIS数据集的初步专家专化分析；

**📈 对比分析**

与密集的DINO、Token-MoE、实例路由器和场景路由器等基线比较，HI-MoE在COCO验证集上取得AP 53.0（小物体AP 35.4），比DINO提升约1.7点（小物体提升3.3点），同时保持稀疏计算；

**⚠️ 局限性**

局限性包括：评估仅局限于COCO；缺乏完整的LVIS/Objects365验证；没有深入的路由利用率、熵曲线、专家消失等诊断；理论上稀疏FLOPs不一定带来实际加速，硬件实现细节尚未充分阐述。

---

## 293. Fine-Tuning Integrity for Modern Neural Networks: Structured Drift Proofs via Norm, Rank, and Sparsity Certificates

**arXiv ID:** 2604.04738 | [PDF](https://arxiv.org/pdf/2604.04738v1)

**作者:** Zhenhang Shang `[一作]` (Hong Kong University of Science and Technology), Kani Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 897 | [OpenAlex ID](https://openalex.org/A5040402090)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

提出了一种Fine‑Tuning Integrity（FTI）框架，用于对大型神经网络的微调过程进行零知识的完整性证明，保证模型仅在预先定义的结构化漂移（norm、rank、sparsity）内变动；

**💡 创新点**

核心创新在于构造了Succinct Model Difference Proofs（SMDPs）三种可压缩零知识证明（基于随机投影、多项式承诺和流式线性校验），并证明了在无结构漂移情况下无法实现信息论意义下的紧凑证明；

**🔧 技术方法**

技术手段包括Johnson‑Lindenstrauss随机投影、KZG多项式承诺、Bulletproofs风格区间证明、流式线性一致性检验以及Merkle树/向量承诺的组合；

**📊 数据集**

实验数据集覆盖了7B参数Transformer（LLaMA风格）、ResNet‑50卷积网络以及6层MLP，使用公开基线模型与合成的微调任务（LoRA、prefix tuning、norm‑constrained fine‑tuning、稀疏毒化、注意力头后门）进行评估；

**📈 对比分析**

在7B Transformer上，聚合FTI证明约为3.8 MB，验证时间约310 ms，证明者耗时约8 min；对ResNet‑50仅需12 s，MLP约0.8 s；与完整SNARK编码的1 GB证明相比，SMDP显著压缩了证明体积并将验证延迟压缩到毫秒级；

**⚠️ 局限性**

主要局限包括：仅约束参数漂移而不直接保证行为安全；证明者成本在极大模型上仍显著；依赖可信的多项式承诺生成器（KZG）或需付出更大开销的透明方案；并且仅支持norm、rank、sparsity三类结构化漂移，无法覆盖更复杂的微调方式。

---

## 294. Structured Causal Video Reasoning via Multi-Objective Alignment

**arXiv ID:** 2604.04415 | [PDF](https://arxiv.org/pdf/2604.04415v1)

**作者:** Zinuo Li `[一作]` (University of Western Australia), Qiuhong Ke `[通讯]` (Monash University)

**通讯引用:** 2435 | [OpenAlex ID](https://openalex.org/A5083239184)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Factum-4B模型，先通过Structured Event Facts构造视频事件的结构化事实，再在此基础上进行因果推理；

**💡 创新点**

创新点在于：①将视频事件信息压缩为高密度结构化事实作为先验约束；②四阶段训练管线（事实训练、格式热身、思考热身、强化学习）分阶段优化；③提出Pareto-Frontier Guided Advantage Balancing（P‑FAB）解决多目标RL的冲突；

**🔧 技术方法**

主要技术包括多模态LLM（Qwen3‑VL‑4B‑Instruct）、多阶段指令微调、LoRA参数微调、Group Relative Policy Optimization（GRPO）与P‑FAB、强化学习、MGDA理论等；

**📊 数据集**

使用自制的CausalFact‑60K（由Video Temporal Grounding数据构建的事实与思考对），以及公开数据集ActivityNet‑Captions、Charades‑TimeLens、VideoMME、MLVU、ETBench、NExT‑GQA等；

**📈 对比分析**

与现有视频‑LLM（如Qwen3‑VL‑4B‑Thinking、Time‑R1‑7B、VideoChat‑R1‑7B、GPT‑4o等）对比，Factum‑4B在多项时序定位和因果推理任务上均取得SOTA或接近GPT‑4o的成绩；

**⚠️ 局限性**

局限性包括：训练数据规模有限；模型仍受限于1fps的视频输入，尚未充分利用更高帧率；对极长视频或复杂多目标推理的鲁棒性仍需进一步验证。

---

## 295. STRIDe: Cross-Coupled STT-MRAM Enabling Robust In-Memory-Computing for Deep Neural Network Accelerators

**arXiv ID:** 2604.04483 | [PDF](https://arxiv.org/pdf/2604.04483v1)

**作者:** Imtiaz Ahmed `[一作]` (Purdue University), Sumeet Kumar Gupta `[通讯]` (Purdue University)

**通讯引用:** 3882 | [OpenAlex ID](https://openalex.org/A5044276472)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于交叉耦合的STT‑MRAM（STRIDe-I 与 STRIDe-II）实现鲁棒的 XNOR‑和 AND‑型内存计算（IMC）加速器，解决了低区分度导致的计算误差问题。

**💡 创新点**

创新点在于利用两个MTJ分支的交叉耦合显著提升比特电流高低比（IH/IL）至 8000+，从而在位级显著提升感知边际（sense margin）和读取失调裕度（read disturb margin）。

**🔧 技术方法**

采用PMA MTJ模型、Landau‑Lifshitz‑Gilbert‑Slonczewski 动力学、NEGF 电阻模型、HSPICE 细胞仿真，并在 45 nm CMOS 技术下实现 64×64 交叉线阵列；同时配合 ADC、模拟电流差分器和宏观级 PyTorch 交叉线仿真器。

**📊 数据集**

使用 CIFAR‑10 数据集训练 ResNet‑18 的二值网络（BNN）和 4‑bit 权重/输入网络，作为推理准确率评估。

**📈 对比分析**

通过与基线 1T‑1MTJ（带虚拟列）和 2T‑2MTJ（差分）设计对比，STRIDe 在 PWA = 8 时实现 BNN 准软件精度 87.6%，4‑bit 网络 92.1%；在更高 PWA = 16 时仍保持 85–92% 的准确率，且感知边际提升 3–4×、读取失调裕度提升 27.6%。

**⚠️ 局限性**

局限性包括：写操作功耗与延迟上升（1.4–1.6×）、交叉耦合电路面积略增（约 13–16%），以及对高行激活（PWA > 16）时仍需更高精度 ADC 以抵消 IR 损失。

---

## 296. Generative modeling of granular flow on inclined planes using conditional flow matching

**arXiv ID:** 2604.04453 | [PDF](https://arxiv.org/pdf/2604.04453v1)

**作者:** Xuyang Li `[一作]` (University of North Carolina at Charlotte), Yimin Lu `[通讯]` (Texas Tech University)

**通讯引用:** 1241 | [OpenAlex ID](https://openalex.org/A5057480713)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用条件流匹配(CFM)框架，基于稀疏边界速度观测重建密集颗粒流的内部速度场、应力场及颗粒温度，并给出不确定性评估。

**💡 创新点**

首次将CFL引入颗粒流逆问题；采用稀疏感知梯度引导保证零速度区保持不变；将物理解码器与生成模型耦合，直接输出应力与温度。

**🔧 技术方法**

条件流匹配网络（U‑Net ODE）、可微前向观测算子、梯度自适应引导、物理解码器（U‑Net）以及自适应正则化的误差梯度。

**📊 数据集**

使用高保真离散元模拟（DEM）生成的三维颗粒流数据，经过体积分布后得到速度、应力、温度等标量场作为训练集。

**📈 对比分析**

与确定性CNN基线对比，在完全观测时两者相近；在极端稀疏观测下，CFM在相关系数与RMSE上提升约70%–90%，同时提供可靠的预测置信区间。

**⚠️ 局限性**

需要大量高质量DEM训练数据，模型对几何、材料参数的迁移性有限；对实时动态重建和三维全域预测仍有挑战。

---

## 297. Semantic IDs for Recommender Systems at Snapchat: Use Cases, Technical Challenges, and Design Choices

**arXiv ID:** 2604.03949 | [PDF](https://arxiv.org/pdf/2604.03949v1)

**作者:** Clark Mingxuan Ju `[一作]` (Snap Inc), Neil Shah `[通讯]` (Snap Inc)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在Snapchat大规模部署Semantic ID（SID），将SID作为辅助特征和生成式检索目标，在广告、商品、好友推荐、搜索和短视频等多种推荐场景中进行实验与上线。

**💡 创新点**

提出两项技术创新：①基于straight‑through estimator（STE）与多模态嵌入融合的RQ‑VAE改造，显著缓解代码书崩溃；②启发式桶内分辨和检索深度优先的SID‑to‑Item解码策略，用以处理SID碰撞问题；同时对SID唯一性与检索性能的非线性关系进行系统性分析。

**🔧 技术方法**

核心技术包括：RQ‑VAE（残差量化）与STE、LLM/VLM编码器、GraphHash、Qwen‑Embedding、Beam Search、Auto‑Regressive生成式检索模型以及基于业务元数据的启发式解码。

**📊 数据集**

使用内部Snapchat数据集（广告、DPA、好友推荐、搜索、短视频行为日志）以及外部Amazon Beauty数据集进行唯一性实验。

**📈 对比分析**

与传统原子ID做对比：离线检索提升R@5/N@5达+31.5%/+26.5%；在线A/B实验显示广告Swipe Up提升0.028%，Landing Page View 0.035%；DPA Add‑to‑Cart提升0.67%；短视频观看+0.57%、发送+2.54%、转载+3.55%、分享+4.39%。

**⚠️ 局限性**

局限性包括：RQ‑VAE改造需手工调参且对不同数据分布的鲁棒性待验证；SID唯一性并非完美评估指标，仍需更精准的离线度量；启发式解码依赖业务规则，缺乏通用自动化；多模态融合提升模型复杂度和训练成本。

---

## 298. NativeTernary: A Self-Delimiting Binary Encoding with Unary Run-Length Hierarchy Markers for Ternary Neural Network Weights, Structured Data, and General Computing Infrastructure

**arXiv ID:** 2604.03336 | [PDF](https://arxiv.org/pdf/2604.03336v1)

**作者:** Maharshi Savdhariya `[一作]` `[通讯]` (Indian Institute of Technology Bombay), Maharshi Savdhariya (Indian Institute of Technology Bombay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 NativeTernary 编码方案，用 2 位比特对同时表示三元数值与多层级结构。

**💡 创新点**

核心创新是用保留的 2 位比特对作为单字节无符号长度编码，既实现自边界同步，又保持三元数据密度。

**🔧 技术方法**

采用了二进制对划分、无符号长度编码、平衡/无符号三元映射、可配置占位符位对，并给出了 10 行无状态解码器。

**📊 数据集**

主要验证数据集包括 BitNet b1.58 三元语言模型权重、自然语言文本、IoT/工业、车载、卫星、金融行情等结构化流。

**📈 对比分析**

与现有浮点/二进制序列化（GGUF/SafeTensors）、固定长度边界标记（BERT‑BPE）以及常规二进制压缩对比，理论上每比特可获得 0.792 bit 信息密度，边界开销随层级递减，解码复杂度极低，可在 ARM Cortex‑M 等低功耗设备上以几十毫秒级完成解码。

**⚠️ 局限性**

局限在于对任意二进制数据无压缩效果；无符号三元映射在噪声通道易产生误边界；需在协议或文件格式中预先约定占位符位对，迁移成本受限。

---

## 299. Talk2AI: A Longitudinal Dataset of Human--AI Persuasive Conversations

**arXiv ID:** 2604.04354 | [PDF](https://arxiv.org/pdf/2604.04354v1)

**作者:** Alexis Carrillo `[一作]` (University of Trento), Massimo Stella `[通讯]` (University of Trento)

**通讯引用:** 2261 | [OpenAlex ID](https://openalex.org/A5074066409)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了 Talk2AI 数据集，该数据集包含 770 名意大利成年人在四周内与四种大型语言模型（GPT‑4o、Claude Sonnet 3.7、DeepSeek‑chat V3、Mistral Large）进行 3,080 次对话（共 30,800 轮），每次对话后收集信念改变、信念稳定性、AI 人性化感知以及行为意向等指标，并配备完整的心理测评与人口统计资料。

**💡 创新点**

创新点在于：① 采用四波纵向设计，系统追踪 AI 影响随时间的演变；② 在每轮对话中加入“识别谬误”与“保持 100 字”等约束，保证论证深度与可比性；③ 将心理测评（TILLMI、NFC、Big Five）与会话文本、反馈数据整合，支持多维度的信念与情感动力学建模；④ 提供多语种（英语、西班牙语、荷兰语、德语、葡萄牙语、法语）翻译版本，便于跨语言 NLP 研究。

**🔧 技术方法**

使用的技术包括：Python + JSON 处理与三阶段过滤管线、R 进行 EFA/CFA 与测量不变性分析、DWLS 估计、SEM、LSTM/Markov 链等序列建模、K‑Means/HDBSCAN 聚类、词嵌入 + 结构化特征融合、因子得分与标准化分数计算。

**📊 数据集**

使用的数据集为 Talk2AI，公开在 GitHub 上，包含三大 JSON 文件（registries、psyscales、conversations）以及翻译后文本档案，覆盖 770 名受试者、四个主题（气候变化、数学焦虑、健康误导）与四种 LLM 体系。

**📈 对比分析**

对比方法：可将不同 LLM 体系、主题或个体特征作为自变量，使用多元回归/分类模型评估对立场转变（Q1）与信念稳定性（Q0）等结果；利用 SEM 检验心理特质与文本特征对说服效果的中介作用；聚类分析识别用户类型并比较其对话策略与说服效果。性能表现因模型与主题而异，论文未给出统一指标，但指出不同 LLM 在人性化感知与说服率上存在显著差异，且在纵向数据中表现出逐步递减或递增的信念改变趋势。

**⚠️ 局限性**

局限性包括：① 样本仅限意大利成人，外推性受限；② 仅对 4 周内的短期互动进行追踪，长期影响未知；③ 受试者自报告的说服度与行为意向可能受社会期望偏差；④ 文本翻译可能引入语义偏差；⑤ 对话被限制在 10 轮且包含人工设定的提示，可能不完全代表自然交互；⑥ 仅使用 4 种 LLM，无法涵盖更广泛的模型架构。

---

## 300. Strategies in Sabotage Games: Temporal and Epistemic Perspectives

**arXiv ID:** 2604.03872 | [PDF](https://arxiv.org/pdf/2604.03872v1)

**作者:** Nina Gierasimczuk `[一作]` (Technical University of Denmark), Katrine B. P. Thoft `[通讯]` (Technical University of Denmark)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

研究了破坏游戏的时序与知识维度，提出用ATL*及其认知扩展ATEL来表述并分析跑者与破坏者的策略，并引入新的活性破坏游戏模型。

**💡 创新点**

创新点在于将传统的Sabotage Modal Logic扩展到交替时序逻辑与认知框架，揭示最小s‑t割与动态图割问题的关系，并讨论角笠破坏游戏与多跑者、分布目标等未来方向。

**🔧 技术方法**

采用ATL*、ATEL等模态逻辑技术，形式化游戏结构与策略，利用逻辑语义证明胜利策略存在性，并与图论中的最小割概念建立对应。

**📊 数据集**

本研究为理论性工作，未使用具体数据集，仅在示例图上进行形式化演示。

**📈 对比分析**

通过逻辑等价与归约与传统SML比较，证明ATL*在表达时间与知识条件下更为强大；未给出实验性能指标。

**⚠️ 局限性**

局限性包括只考虑有限游戏与单跑者情形；对无限游戏、多跑者、分布式目标的分析缺失；认知扩展在策略实现层面仍需进一步完善。

---

## 301. Dominating Set with Quotas: Balancing Coverage and Constraints

**arXiv ID:** 2604.04912 | [PDF](https://arxiv.org/pdf/2604.04912v1)

**作者:** Sobyasachi Chatterjee `[一作]` (Institute of Mathematical Sciences), Anannya Upasana `[通讯]` (Institute of Mathematical Sciences)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

研究了Dominating Set with Quotas（DSQ）问题，分析其在不同稀疏图类中的可解性与难度；

**💡 创新点**

首次给出DSQ在退化度为2、树宽、无向稀疏（无界稠密）以及顶点压缩图类上的W[1]、FPT、亚指数算法，并揭示了与经典Dominating Set在稀疏图类中的显著差异；

**🔧 技术方法**

使用参数化复杂度理论、子集卷积动态规划、bidimensionality 框架、FO 逻辑模型检查、稀疏图的结构定理等技术；

**📊 数据集**

未使用具体实验数据集，全部为理论分析与算法设计；

**📈 对比分析**

通过构造归约证明W[1]-硬度，并给出对应参数化和亚指数算法的上界；相较于已有的Dominating Set结果，DSQ在相同图类下更难，且在无界稠密图上实现FPT；

**⚠️ 局限性**

依赖于逻辑模型检查，缺乏直接组合算法；未探讨逼近性能和核化（kernelization）；对近似和预处理仍有空白。

---

## 302. 3D-Fixer: Coarse-to-Fine In-place Completion for 3D Scenes from a Single Image

**arXiv ID:** 2604.04406 | [PDF](https://arxiv.org/pdf/2604.04406v1)

**作者:** Ze-Xin Yin `[一作]` (Nankai University), Jin Xie `[通讯]` (Nanjing University)

**通讯引用:** 21039 | [OpenAlex ID](https://openalex.org/A5039338731)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于 in-place completion 的单视图场景合成框架 3D‑Fixer，利用可见点云作为空间锚点，结合预训练的对象级生成先验，直接在原始位置完成缺失几何并生成完整 3D 资产。

**💡 创新点**

创新点包括：① 在原始位置完成缺失几何的 in-place completion 范式，消除姿态对齐的误差；② 双分支条件网络与 Occlusion‑Robust Feature Alignment（ORFA）策略，提升遮挡场景下的稳定性；③ 粗到细的两阶段生成方案，先估计不确定边界后细化几何；④ 发布 ARSG‑110K 大规模场景级数据集，为后续研究提供高质量基准。

**🔧 技术方法**

技术核心包括：基于 TRELLIS 的两阶段 Diffusion Transformer（DiT）+ SLAT 表示；Geometry‑Aware Feature Projection（GAFP）；Depth‑ratio‑embedded self‑attention 与全局特征交叉注意力；Flow Matching 损失与 ORFA 对齐损失；双分支网络实现可见几何与图像纹理的融合；粗到细生成策略。

**📊 数据集**

主要使用自研的 ARSG‑110K 数据集（110K 场景、3M 视图、180K+ 高质量资产），并在公开数据集 MIDI、Gen3DSR、ScanNet（MetaScenes）等进行评测。

**📈 对比分析**

与 MIDI、Gen3DSR、PanoRecon、Total3D、InstPIFu、SSR、DiffCAD、REPARO 等多种基线对比，结果显示在 MIDI 测试集场景 Chamfer 距离 0.069、F‑score 78.67，显著优于 MIDI（0.080/50.19）和 Gen3DSR（0.123/40.07）；在 Gen3DSR 测试集 CD 0.103、FS 77.95；在 ScanNet 子集 CD 0.130、FS 61.58；整体实现了 state‑of‑the‑art 性能，并且推理速度相对较快。

**⚠️ 局限性**

局限性：受限于输入几何估计的准确性，极端遮挡或光照变化仍可能导致误差；训练依赖大量预训练对象模型与自研大规模数据集，难以直接迁移到极端真实场景；目前仅支持单视图输入，对多视角或长序列的扩展尚未验证。

---

## 303. Scaling DPPs for RAG: Density Meets Diversity

**arXiv ID:** 2604.03240 | [PDF](https://arxiv.org/pdf/2604.03240v1)

**作者:** Xun Sun `[一作]` (Southwestern University of Finance and Economics), Qiang Gao `[通讯]` (Southwestern University of Finance and Economics)

**通讯引用:** 1734 | [OpenAlex ID](https://openalex.org/A5048618271)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种可插拔的 RAG（检索增强生成）模块，利用 Determinantal Point Processes (DPP) 通过 P-Adapter 对检索候选进行多样性和互补性建模，并引入了 Diverse Margin Loss (DML) 对 P-Adapter 进行优化，从而在多跳推理任务中提升检索效果

**💡 创新点**

核心创新在于：①将 DPP 引入 RAG，实现对候选子集的全局多样性与互补性控制；②设计轻量级 P-Adapter 仅在子集选择阶段启用，避免影响原始检索；③提出 DML 这一集合级损失，利用 margin 方式对正负子集的行列式差异进行优化，提升训练稳定性和效果

**🔧 技术方法**

主要技术包括：基于向量检索的相似度匹配、动态构造 DPP kernel（包含质量矩阵 Q 的融合）、最大后验（MAP）贪心推理、P-Adapter（轻量化前馈网络）以及 DML 损失的平滑近似实现

**📊 数据集**

使用 MultiHop‑RAG 数据集进行评估，该数据集包含 2,556 个多跳问答查询，涉及 2‑4 跳推理；同时在不同嵌入后端（BGE‑large、BGE‑m3、Qwen‑Embedding‑0.6B、Qwen‑Embedding‑4B）上测试

**📈 对比分析**

在多跳检索指标（Recall@K、NDCG@K、Hits@K）上，相比标准 RAG，本文方法在 k=10 与 k=4 两种上下文预算下均实现显著提升，NDCG@10 提升 3–12%，Recall@10 提升 5–15%，Hits@10 亦有明显改善；DML 较传统 NLL 损失在不同 hop 数和 reranker 情况下表现更稳健

**⚠️ 局限性**

局限性包括：①仍需在较大知识库上验证 DPP 的可扩展性（尽管引入了动态 kernel 与轻量化适配器）；②对 DPP 的负相关性限制意味着难以建模显式的吸引关系；③DML 需要采样负子集或近似，可能导致训练成本上升

---

## 304. Direct Integer Division in RNS and its Hardware Solutions

**arXiv ID:** 2604.04796 | [PDF](https://arxiv.org/pdf/2604.04796v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 305. Croissant Charts: Modulating the Performance of Normal Distribution Visualizations with Affordances

**arXiv ID:** 2604.04432 | [PDF](https://arxiv.org/pdf/2604.04432v1)

**作者:** Racquel Fygenson `[一作]`, Lace M. Padilla `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

未提供论文内容，无法进行总结。

**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 306. Position: Logical Soundness is not a Reliable Criterion for Neurosymbolic Fact-Checking with LLMs

**arXiv ID:** 2604.04177 | [PDF](https://arxiv.org/pdf/2604.04177v1)

**作者:** Jason Chan `[一作]` (University of Sheffield), Zhixue Zhao `[通讯]` (University of Sheffield)

**通讯引用:** 214 | [OpenAlex ID](https://openalex.org/A5101991159)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出逻辑有效性不足以检测误导性声明，阐述人类推理与形式逻辑的偏差

**💡 创新点**

主张将LLM的人类式推理能力作为功能，用于评估逻辑推导是否可能误导读者

**🔧 技术方法**

基于认知科学、语用学文献构建误导案例类型，分析形式逻辑与人类推断的差异

**📊 数据集**

未使用具体数据集

**📈 对比分析**

无实验比较，文中仅以理论与案例说明

**⚠️ 局限性**

缺乏实证验证，未给出系统实现或性能评估

---

## 307. Training-Free Image Editing with Visual Context Integration and Concept Alignment

**arXiv ID:** 2604.04487 | [PDF](https://arxiv.org/pdf/2604.04487v1)

**作者:** Rui Song `[一作]` (Hong Kong University of Science and Technology), Jun Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 86351 | [OpenAlex ID](https://openalex.org/A5100400217)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 VicoEdit，一种训练与反演均无须的视觉上下文感知图像编辑方法，直接将源图像映射为目标图像。

**💡 创新点**

创新点在于将反演与采样的速度场融合实现直接轨迹编辑，并通过概念对齐的后验采样指导保持细节，无需大规模预训练或手工掩码。

**🔧 技术方法**

使用预训练多模态扩散 Transformer (MMDiT)、Rectified Flow、速度场融合、Concept Attention、Diffusion Posterior Sampling (DPS) 等技术。

**📊 数据集**

实验基于 DreamBooth 数据集，手工挑选源图与上下文图，涵盖域内替换、添加与跨域添加三类任务。

**📈 对比分析**

与 Diptych Prompting、FLUX.2、Qwen-2511、Seedream 5.0 Lite、Nano Banana 2 等基线进行 LPIPS、DINO、CLIP-Text 等指标比较，VicoEdit 在结构保持和指令遵循上匹配甚至超过商业模型。

**⚠️ 局限性**

局限性包括较高的推理时间与显存需求（尤其是多噪声样本与概念对齐），对强大基础模型的依赖，以及跨域细节仍有提升空间。

---

## 308. HighFM: Towards a Foundation Model for Learning Representations from High-Frequency Earth Observation Data

**arXiv ID:** 2604.04306 | [PDF](https://arxiv.org/pdf/2604.04306v1)

**作者:** Stella Girtsou `[一作]` (National Observatory of Athens), Harris Kontoes `[通讯]` (National Observatory of Athens)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了HighFM，一个针对高时空分辨率地球观测的基础模型，并在SEVIRI数据上进行自监督预训练，随后在云遮挡和活火检测任务上微调。

**💡 创新点**

创新点在于将SatMAE适配为高时间频率的地球观测，加入细粒度时间编码和多时刻输入，并证明该模型在实时灾害监测任务上优于现有基础模型。

**🔧 技术方法**

使用了自监督掩码自编码器（MAE）框架、Vision Transformer骨干、细粒度时间嵌入、三时刻多模态输入以及相应的分割头。

**📊 数据集**

使用了超过2 TB的SEVIRI多光谱时序数据（2014‑2019）进行预训练，后续在2014‑2024的云掩模和活火检测数据集上进行微调。

**📈 对比分析**

采用与UNet、ViT、Copernicus‑FM、Panopticon等基线进行对比，使用交叉熵和Dice损失；实验结果显示HighFMMT在平衡准确率、IoU和召回率等指标上均优于所有基线。

**⚠️ 局限性**

局限性在于仅针对单一地球静止卫星传感器，空间分辨率有限，且未实现多模态跨传感器的迁移与泛化。

---

## 309. Beyond Retrieval: Modeling Confidence Decay and Deterministic Agentic Platforms in Generative Engine Optimization

**arXiv ID:** 2604.03656 | [PDF](https://arxiv.org/pdf/2604.03656v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 310. ExpressEdit: Fast Editing of Stylized Facial Expressions with Diffusion Models in Photoshop

**arXiv ID:** 2604.03448 | [PDF](https://arxiv.org/pdf/2604.03448v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 311. ClawArena: Benchmarking AI Agents in Evolving Information Environments

**arXiv ID:** 2604.04202 | [PDF](https://arxiv.org/pdf/2604.04202v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 312. Query Optimization and Evaluation via Information Theory: A Tutorial

**arXiv ID:** 2604.04893 | [PDF](https://arxiv.org/pdf/2604.04893v1)

**作者:** Mahmoud Abo Khamis `[一作]`, Dan Suciu `[通讯]` (University of Washington)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一套完整的通用框架 PANDA，用于在给定任意查询 Q 与任意输入统计信息（度约束、范数约束等）的条件下，生成最优的 CQ 评估算法，并给出了其时间复杂度上界。

**💡 创新点**

核心创新在于将信息论的熵与多项式约束（Shannon 以及 Shannon‑flow 等不等式）与查询计划直接耦合，利用证明序列把理论上界直接转化为可执行的多树分解（动态）查询计划；这一思想实现了自适应数据分区、可在任意统计约束下达到最优的子模宽度（submodular width）复杂度，并能够与快速矩阵乘法等高级技术无缝集成。

**🔧 技术方法**

主要技术包括：
- 熵与极值理论，用来推导输出大小上界；
- 多项式 (polymatroid) 与 Shannon 不等式的线性规划求解；
- 证明序列（包括分解、组合、单调性、次模性等步骤）与 Shannon‑flow 证据生成；
- 子概率测度与分解的概率化证明；
- 基于变量消除的 FMM 选取与 ω‑子模宽度的推广。

**📊 数据集**

该工作属于理论算法与数据库理论范畴，不依赖具体数据集；所有结果均在数学模型与假设（统计信息）下给出。

**📈 对比分析**

与传统的组合最优连接算法（如 AGM、Hypertreewidth）以及基于图算法的 k‑cycle、k‑clique 等已知实现比较，PANDA 在相同统计约束下能够实现与或优于 AGM、Submodular Width 以及 fast‑matrix‑multiplication（ω‑submodular width）的运行时间；在实验设置下，PANDA 能以 O(N^{submodular width})·polylog(N) 的时间完成查询。

**⚠️ 局限性**

局限性：
- 对计数查询和非幺半群（non‑idempotent）半环的求解仍未实现，Sharp‑submodular width 仍是一个待完善的上界；
- 计算极值（entropic）宽度与 Shannon‑flow 的最短证明长度仍是开放问题，可能导致实际算法的证明长度指数级；
- 对于非常大或复杂的统计信息，LP 规模可能增长，影响可行性；
- 当前框架在理论上与实际数据库系统对接时，需要进一步的系统实现与优化。

---

## 313. TABQAWORLD: Optimizing Multimodal Reasoning for Multi-Turn Table Question Answering

**arXiv ID:** 2604.03393 | [PDF](https://arxiv.org/pdf/2604.03393v1)

**作者:** Tung Sum Thomas Kwok `[一作]` (University of California), Guang Cheng `[通讯]` (University of California)

**通讯引用:** 101 | [OpenAlex ID](https://openalex.org/A5019393455)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种训练无关的多模态表格推理框架，通过动作条件下的多模态选择与表格元数据驱动的轨迹优化，实现多轮推理的准确率提升与延迟降低。

**💡 创新点**

创新点在于同时结合动作条件的动态模态切换和低维表格元数据估计，使推理既能保持高精度又能显著减少推理步骤与计算成本。

**🔧 技术方法**

采用多模态大语言模型、动作条件的多模态选择策略、表格元数据压缩与轨迹优化、无监督状态估计等技术。

**📊 数据集**

在 WikiTableQuestions、MMQA、MMTU、TABMWP、TAT‑QA、HiTab、FeTaQA、TabFact、InfoTabs 等七大表格推理与事实验证基准上进行评估。

**📈 对比分析**

与文本、图像、混合多模态、训练无关、轨迹优化等多种基线对比，平均提升 6.32%–10.81% 的准确率，推理延迟降低 33.35%，并在 Qwen3‑VL‑8B 上取得 85.67% 的平均准确率。

**⚠️ 局限性**

局限性在于元数据估计仍可能出现误差，无法完全捕捉完整表格状态；对极大规模表格和长序列推理的鲁棒性与可扩展性尚需进一步验证。

---

## 314. From Pre-trained Models to Large Language Models: A Comprehensive Survey of AI-Driven Psychological Computing

**arXiv ID:** 2604.03259 | [PDF](https://arxiv.org/pdf/2604.03259v1)

**作者:** Huiyao Chen `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 61265 | [OpenAlex ID](https://openalex.org/A5100402851)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于计算处理模式的AI心理学任务体系，系统梳理四大任务类型并对近300篇论文进行时序演进分析

**💡 创新点**

首次以计算范式而非领域划分构建统一分类框架，揭示跨心理子领域可迁移的技术模式

**🔧 技术方法**

聚焦迁移学习、预训练模型（BERT、GPT系列）以及大语言模型的零/少样本适配，并结合多模态融合与元学习方法

**📊 数据集**

综合检索并组织了临床实验、纵向多站点、生态自然与专项等多种数据集，包括DAIC、CLPsych、RECOLA、SEWA等

**📈 对比分析**

通过对任务、数据与评估指标的系统对比，指出现有方法在准确率、解释性、跨文化泛化等方面的优劣，并给出基准性能与评测框架

**⚠️ 局限性**

仍面临数据稀缺、标签不确定、跨文化有效性、隐私与伦理约束以及缺乏统一评测基准等局限，亟需更大规模、多模态、标准化资源

---

## 315. Modelling and Analysis of Supply Chains using Product Time Petri Nets

**arXiv ID:** 2604.04544 | [PDF](https://arxiv.org/pdf/2604.04544v1)

**作者:** Eric Lubat `[一作]` (IRIT), Rémi Sauvère `[通讯]`

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文提出了一种基于Product Time Petri Nets（PTPN）的模块化供应链建模与分析方法，能够将各子系统（供应商、工厂、管理者）独立建模并通过同步转移标签组合成完整系统。

**💡 创新点**

创新点在于将供应链管理者显式建模为共享的可移动资源，并通过PTPN的同步关系实现多时序子系统的组合，同时通过时限约束揭示同步导致的时间死锁（timelock）现象。

**🔧 技术方法**

主要技术包括PTPN定义、同步产品构造、状态类图（SCG）抽象、TINA工具链（twina、tina、selt）进行状态空间生成与LTL模型检测。

**📊 数据集**

使用了公开的基准生成器和结束线Petri网（存放在GitHub/Zenodo），数据集为人工设定的时限区间和供应商/管理者数量参数，而非真实工业数据。

**📈 对比分析**

通过对不同供应商数、管理者数及管理者时限区间的组合进行模型检测，评估成功、超时与时锁三类失效情况，实验显示随着供应商增多时状态空间呈指数增长，单一管理者难以满足多供应商需求，但通过调节订单时间可显著提升可行性。

**⚠️ 局限性**

主要局限在于同步时产生的状态空间爆炸，导致只能在少数供应商级别上完成完整验证；模型缺乏失效、返工、库存等更复杂的业务行为，且仅考虑最坏情况的时间约束。

---

## 316. Comprehensive Analysis of Cellular Uplink Performance in a Dense Stadium Deployment

**arXiv ID:** 2604.04371 | [PDF](https://arxiv.org/pdf/2604.04371v1)

**作者:** S. M. Haider Ali Shuvo `[一作]` (University of Notre Dame), Monisha Ghosh `[通讯]` (University of Notre Dame)

**通讯引用:** 2633 | [OpenAlex ID](https://openalex.org/A5101557097)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对美国诺特丹大学体育场高密度环境下的5G网络进行现场测量，重点评估高频TDD与低频FDD在上行链路的性能差异。

**💡 创新点**

首次揭示高频TDD上行受限于功率、帧结构与频谱可用性三重瓶颈，而低频FDD在上行具有明显优势，为下一代网络谱管理与双工设计提供实证依据。

**🔧 技术方法**

使用Rohde & Schwarz QualiPoc捕获PHY层指标（MCS、PRB、Tx功率等）并结合Ookla Speedtest测得吞吐量，随后通过CDF、直方图等统计手段进行性能对比。

**📊 数据集**

利用两场比赛（预赛与赛后）与空场赛前的现场测量数据，并收集了停车场COW部署的测量样本，形成多场景、多频段的实验数据集。

**📈 对比分析**

采用PRB分配比例、MCS分布、吞吐量CDF等指标进行对比，结果显示高频TDD上行吞吐仅占10%以下，而低频FDD上行吞吐占比超过70%，证明TDD在上行表现显著不佳。

**⚠️ 局限性**

实验受限于单频上行、未实现跨载波聚合、测量范围仅覆盖体育场与COW现场，缺乏更大规模网络、多天线与不同运营商基站配置的验证。

---

## 317. Adaptive Threshold-Driven Continuous Greedy Method for Scalable Submodular Optimization

**arXiv ID:** 2604.03419 | [PDF](https://arxiv.org/pdf/2604.03419v1)

**作者:** Mohammadreza Rostami `[一作]` (University of California), Solmaz S. Kia `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种自适应阈值驱动的连续贪婪算法 ATCG，用于在分区 matroid 约束下最大化子模函数，并显著减少分布式系统中的通信量。

**💡 创新点**

创新点在于：1) 引入动态活跃集（active set）机制，只在需要时扩展梯度评估范围；2) 通过阈值 τ 控制活跃集的大小，保留 (1‑e^{‑τ}) 的近似保证；3) 在低曲率情形下证明有效率提升至 (1‑e^{‑max{τ,1‑c}})，从而在保持近似最优的同时进一步降低通信成本。

**🔧 技术方法**

采用了连续贪婪框架、梯度估计（Monte Carlo 采样）、分区 matroid 线性算子、服务器辅助分布式协议、主动激活阈值控制以及低曲率分析。

**📊 数据集**

使用 CIFAR‑10 数据集中的六类动物（deer、frog、bird、horse、cat、dog）作为实验数据，构建每个机器人对应的本地图像集合。

**📈 对比分析**

与完整连续贪婪 (CG) 以及传统序列贪婪 (SG) 进行对比。ATCG 在目标函数值上几乎与 CG 相同（误差 <1%），但累计上传字节大幅下降，通信成本降低约 50%–80%。

**⚠️ 局限性**

局限性：1) 需要中心服务器，无法完全去中心化；2) 阈值 τ 的设定依赖经验，曲率估计不易；3) 在高曲率或非分布式环境下，性能可能退化；4) 只针对分区 matroid 约束，其他约束的推广尚待研究。

---

## 318. Outlier-Robust Nonlinear Moving Horizon Estimation using Adaptive Loss Functions

**arXiv ID:** 2604.04862 | [PDF](https://arxiv.org/pdf/2604.04862v1)

**作者:** Nestor Deniz `[一作]` (Instituto de Investigacion en Senales, Sistemas e Inteligencia Computacional), Leonardo Giovanini `[通讯]` (Instituto de Investigacion en Senales, Sistemas e Inteligencia Computacional)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种在移动窗口估计（MHE）中使用自适应鲁棒损失函数的框架，通过在线更新形状参数α来降低异常值的影响并加入正则化项，以提高估计精度。

**💡 创新点**

创新点在于：① 将Barron自适应鲁棒损失的导数平方作为阶段成本，使其在[1,2)区间内连续无奇点；② 在MHE中在线估计每个测量点的α值，形成自适应鲁棒机制；③ 通过交替求解状态和α的双重优化方案实现快速收敛。

**🔧 技术方法**

使用的技术包括：自适应鲁棒损失函数（Barron的改进版）、移动窗口估计（MHE）、双重迭代优化（先求状态再求α）、正则化项以避免α过大、以及仿真中的模型预测控制（MPC）。

**📊 数据集**

数据集：在仿真环境中生成一辆拖拉机-拖车车辆的路径跟踪任务，GNSS测量含有两种噪声分布（正态和均匀），并随机插入异常值；实验共进行1000次试验。

**📈 对比分析**

比较方法：与固定α的MHE、通过网格搜索得到α的MHE_grid 以及本方法MHE_prop（M=10或M=3）进行对比。性能指标包括均方估计误差(Ψ)、均方位置误差(Δ)和平均计算时间(η)。实验结果显示：MHE_prop在误差上略优于MHE_grid，但计算时间更高；MHE_grid与MHE_prop（M=3）在误差和时间上相近；固定α方法计算最快但误差最大。

**⚠️ 局限性**

局限性：① 计算量大，尤其是在线更新α的迭代过程；② 该方法假设窗口内异常值分布均匀，若异常值聚集或时变可能影响鲁棒性；③ 过多的α参数可能导致过拟合，特别是窗口较短或测量稀疏时。

---

## 319. Assessing Large Language Models for Stabilizing Numerical Expression in Scientific Software

**arXiv ID:** 2604.04854 | [PDF](https://arxiv.org/pdf/2604.04854v1)

**作者:** Tien Nguyen `[一作]` (Virginia Tech), Kirshanthan Sundararajah `[通讯]` (Virginia Tech)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估大语言模型（LLMs）在检测和改进科学软件中浮点数不稳定性表达式的能力，构造新的表达式合成数据集并与传统工具对比。

**💡 创新点**

证明LLMs能够在传统方法失效的17.4%例子中实现97.9%的稳定化，并在整体上比传统工具提升65.4%的表达式准确率，指出其在控制流与高精度常数方面的局限。

**🔧 技术方法**

使用LLM的零射击与迭代少样本提示、结构化模板提示、以及基于Herbie和FPGen的基线工具。

**📊 数据集**

自制的合成表达式数据集（覆盖单变量、全变量与混合组合，含条件与高精度常数）以及21个来自FPGen的数学库函数基准。

**📈 对比分析**

与Herbie、FPGen等现有数值稳定化工具在约400,000条提示/约2.3B token的实验中比较，LLMs在大部分表达式上能与基线匹配或优于基线，特别是在传统工具无法改进的案例中表现突出。

**⚠️ 局限性**

受限于模型量化精度、缺乏对高精度常数与复杂控制流的精细推理、以及对仅改写而非完整语义保持的风险，且实验仅覆盖六个主流LLM和两种基线工具。

---

## 320. PassiveQA: A Three-Action Framework for Epistemically Calibrated Question Answering via Supervised Finetuning

**arXiv ID:** 2604.04565 | [PDF](https://arxiv.org/pdf/2604.04565v1)

**作者:** Madhav S Baidya `[一作]` `[通讯]` (Indian Institute of Technology Varanasi), Madhav S Baidya (Indian Institute of Technology Varanasi)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出PassiveQA框架，结合知识图和三动作（Answer、Ask、Abstain）路由，实现问答系统在信息缺失时的认知校准；

**💡 创新点**

核心创新在于将决策加权知识图、三动作路由学习以及三代理架构（Planner + 询问/答复/拒绝三种 Agent）相结合，并证明训练时对齐显著优于仅在推理时使用阈值门控；

**🔧 技术方法**

使用技术包括三阶段知识图构建（实体抽取、语义校验、决策强化）、LoRA微调的Mistral‑7B‑Instruct、检索增强生成（Hybrid BM25+Dense、跨编码重排、上下文压缩）、硬门控与结构化 XML 输出；

**📊 数据集**

整合了四个公开 QA 基准（ShARC、QuAC、HotpotQA、ContractNLI）并统一构造统一 schema 以生成三动作标注；

**📈 对比分析**

对比三种 RAG 架构与硬门控后，最佳宏 F1 为 35.3%；通过 Planner 的训练后宏 F1 提升至 55.6%，Abstain 召回率从 13.3% 提升至 58.1%，Hallucination 率下降约 9%；

**⚠️ 局限性**

主要限制包括训练样本量受限、长对话超长导致上下文截断、知识图抽取噪声与零三元组比例高、以及对齐仍需更大规模实验验证。

---

## 321. Borda Aggregation Dynamics of Preference Orderings on Networks

**arXiv ID:** 2604.04209 | [PDF](https://arxiv.org/pdf/2604.04209v1)

**作者:** Moses Boudourides `[一作]` `[通讯]`, Moses Boudourides

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并分析了一种离散时间的网络偏好演化模型，节点以弱序（允许平局）表示偏好，采用局部Borda聚合并受限步长更新。

**💡 创新点**

创新点在于将社会影响网络与偏好空间的离散几何（弱序格图）直接耦合，揭示了两类周期振荡机制：自生振荡（无持久源）与受对立持久节点驱动的强迫振荡；并给出了基于图拓扑、权重与谱性质的充分条件。

**🔧 技术方法**

主要技术包括：Borda分数映射与投影到弱序；在弱序格图上定义的“有界步”更新；对同步与异步两种更新变体的动力学分析；图论与随机行走谱理论（尤其是偶极子模式）相结合；结构稳定性（对权重扰动的不变性）证明。

**📊 数据集**

本文未使用真实数据集，而是通过理论证明和基于小型图（如3、4个节点弱序格图、指向环、二部图）进行说明性实验，验证理论预测。

**📈 对比分析**

比较方法：将模型结果与传统的标量/向量平均模型、有限信任模型等进行概念对比，指出在有序偏好空间下可产生的周期行为在标量模型中无法出现；性能表现为能够在有限状态空间内严格证明周期性与稳定性，且对权重扰动具有鲁棒性。

**⚠️ 局限性**

局限性包括：仅考虑同步更新的周期性结论对异步更新不适用；有界步规则和确定性Tie-breaking 的假设限制了模型的普适性；对极端网络结构（如高度非二部或无周期子图）未给出完整的振荡条件；并未考虑噪声或动态权重，缺乏对实际社交网络的实证验证。

---

## 322. L-SPINE: A Low-Precision SIMD Spiking Neural Compute Engine for Resource-efficient Edge Inference

**arXiv ID:** 2604.03626 | [PDF](https://arxiv.org/pdf/2604.03626v1)

**作者:** Sonu Kumar `[一作]` (Indian Institute of Technology Indore), Santosh Kumar Vishvakarma `[通讯]` (Indian Institute of Technology Indore)

**通讯引用:** 2299 | [OpenAlex ID](https://openalex.org/A5068792760)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种名为L-SPINE的低精度SIMD加速器，用于高效边缘SNN推理。

**💡 创新点**

创新点包括：统一的多精度（2/4/8位）SIMD计算单元、无乘法器的移位加法运算模型、紧耦合的RISC‑V控制器以及可扩展的二维神经元阵列。

**🔧 技术方法**

采用了FPGA实现、量化感知训练后量化、shift‑add 乘法器替代、SIMD并行数据流、以及RISC‑V控制与时序管理。

**📊 数据集**

在ImageNet等视觉任务上使用VGG‑16和ResNet‑18模型进行验证。

**📈 对比分析**

相较于CPU/GPU，L‑SPINE将推理延迟从秒级压缩至毫秒级（约2.38 ms），功耗仅0.54 W，能效提升达三阶；同时低精度量化显著减少内存占用，准确率损失极小。

**⚠️ 局限性**

局限性：低位精度下仍有一定准确率下降；仅在AMD VC707 FPGA上验证，未展开ASIC或更大模型的评估；缺乏层级自适应精度机制；目前仅支持LIF神经元模型。

---

## 323. CLEAR: Unlocking Generative Potential for Degraded Image Understanding in Unified Multimodal Models

**arXiv ID:** 2604.04780 | [PDF](https://arxiv.org/pdf/2604.04780v1)

**作者:** Xiangzhao Hao `[一作]` (Institute of Automation), Yu Sun `[通讯]` (Baidu Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 CLEAR 框架，结合生成和理解能力，提升统一多模态模型在模糊、噪声、压缩和低照度等图像退化场景下的鲁棒性。

**💡 创新点**

创新点包括：① 通过退化感知数据集进行监督微调，强制模型采用“先生成后回答”推理模式；② 构建 Latent Representation Bridge，直接将 VAE 隐空间表示注入推理上下文，消除解码-重编码的梯度瓶颈；③ 设计 Interleaved GRPO，联合强化学习文本推理和视觉生成，在答案正确率奖励下实现两者的端到端优化。

**🔧 技术方法**

使用 Bagel-7B 统一多模态架构，SigLIP 视觉编码器，VAE 生成路径，Qwen2 语言模型；实现监督微调（SFT）、KL 蒸馏、MSE 生成损失；强化学习采用 GRPO 与 Flow-GRPO 的组合；评价指标包括答案准确率、生成视觉质量（BRISQUE、NIQE、MUSIQ）和推理时间。

**📊 数据集**

数据集：MMD-Bench（对六大基准 MMBench、MM-Vet、MMVP、CV-Bench、MMStar、RealWorldQA 进行 16 种退化、3 级强度处理），以及 R-Bench-Dis；训练集基于 LLaVA-OneVision，生成 generate‑then‑answer 与直接回答轨迹。

**📈 对比分析**

与商业（GPT‑4o‑mini、GPT‑4.1‑mini、Gemini‑2.5‑Flash）和开源统一模型（Bagel、Emu3、Janus‑Pro）对比，CLEAR‑SFT 在硬退化平均分提升约3.3 分，CLEAR‑RL 提升约5.1 分（相对提升 8.5%），在 MMD‑Bench Hard 集合上 CLEAR‑RL 取得 65.26 分，明显优于基线 Bagel 的 60.15 分；鲁棒性差距从 7.29 分降至 5.31 分，下降幅度约 24%。

**⚠️ 局限性**

局限性：仅在 Bagel‑7B 体系上验证，未探究对其它统一模型的迁移；依赖 VAE 隐空间，训练成本与推理时间仍受生成次数影响；主要针对视觉退化的图像任务，未覆盖文本、音频等多模态退化；缺乏对极端恶意干扰或自适应硬件加速的评估。

---

## 324. Which Leakage Types Matter?

**arXiv ID:** 2604.04199 | [PDF](https://arxiv.org/pdf/2604.04199v1)

**作者:** Simon Roth `[一作]` `[通讯]`, Simon Roth

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在2,047个二分类表格数据集（OpenML、PMLB、ml）上，对四类因果机制的泄漏进行大规模实验，量化每类泄漏对AUC的影响，并在129个时间序列数据集上检验边界泄漏。

**💡 创新点**

提出按因果机制划分的四类泄漏分类（估计、选择、记忆、边界），并首次对各类泄漏在同一基准下的幅度进行系统比较，发现选择泄漏是最具破坏性的，估计泄漏误导性最小。

**🔧 技术方法**

使用随机分层k折交叉验证注入单一泄漏变体；通过ΔAUC和标准化差值d_z衡量效应；利用贝叶斯元回归评估机制对效应大小的解释力；进行覆盖率实验评估CV置信区间。

**📊 数据集**

2,047个二分类表格数据集（OpenML、PMLB、ml）以及129个时间序列数据集（包括92个FOREX无效、14个真实时间戳、23个伪时间戳）。

**📈 对比分析**

通过与无泄漏基线对比，计算ΔAUC和d_z；结果显示估计泄漏ΔAUC≈0，选择泄漏ΔAUC≈+0.013–+0.045（d_z≈0.27–0.93），记忆泄漏最高达+0.073（d_z≈1.38），边界泄漏平均+0.023；交叉验证的95%置信区间实际覆盖率仅约55%。

**⚠️ 局限性**

仅针对二分类表格数据，未覆盖多分类、回归、神经网络等；实验中的泄漏为人工注入，可能与真实泄漏交互作用不同；随机CV隐藏结构泄漏；使用内部验证而非外部预注册。

---

## 325. MemMachine: A Ground-Truth-Preserving Memory System for Personalized AI Agents

**arXiv ID:** 2604.04853 | [PDF](https://arxiv.org/pdf/2604.04853v1)

**作者:** Shu Wang `[一作]` (MemVerge, Inc.), Charles Fan `[通讯]` (MemVerge, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 MemMachine，一套面向 LLM 代理的全新记忆系统，支持短期、长期事件记忆以及用户画像，能够在多会话中保持事实连贯与个性化。

**💡 创新点**

创新点包括：①以原始会话文本为基准的真值保持架构，极大降低 LLM 依赖；②上下文化检索机制通过扩展核对话片段来克服语义分散；③检索代理（Retrieval Agent）实现多跳、并行、链式检索三种策略，提升多步推理能力。

**🔧 技术方法**

技术实现涵盖：PostgreSQL+pgvector 与 Neo4j 作为存储，句子级向量索引、跨句子上下文扩展、重排序模型、LLM 驱动的摘要与画像抽取，以及基于提示优化的检索与答案生成。

**📊 数据集**

使用了 LoCoMo、LongMemEvalS、HotpotQA、WikiMultiHop、EpBench 等公开基准数据集进行评测，并在自建脚本中复现对照系统（Mem0、Zep、Memobase、LangMem、OpenAI baseline）。

**📈 对比分析**

与竞争系统比较显示：在 LoCoMo 上 MemMachine 在 gpt‑4.1‑mini 下取得 0.9169 分，较 Mem0 提升 9.7 分；在 LongMemEvalS 上通过六维度消融达到 93.0% 正确率；在 HotpotQA hard 上 Retrieval Agent 达到 93.2% 的准确率，且相较于基础检索提升 2.0%。

**⚠️ 局限性**

主要局限包括：检索深度与模型、提示需联合调优；对多模态、跨语言等非文本场景支持不足；在极大规模会话（LongMemEvalM 级）和实时高吞吐量场景下性能与成本仍待验证。

---

## 326. The Role of Generator Access in Autoregressive Post-Training

**arXiv ID:** 2604.04855 | [PDF](https://arxiv.org/pdf/2604.04855v1)

**作者:** Amit Kiran Rege `[一作]` `[通讯]` (University of Colorado), Amit Kiran Rege (University of Colorado)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在自回归语言模型的后训练过程中，生成器访问接口（prefix control 与观察丰富度）如何影响学习器的探索与最优策略获取。

**💡 创新点**

创新点在于首次将prefix控制与观察层次分为两条界限，证明在无重置访问下所有轨迹局部接口等价；一旦允许弱本地重置，观察丰富度成为第二个关键边界，并揭示了这一差异能在KL‑正则化后训练中产生指数级复杂度鸿沟。

**🔧 技术方法**

使用了理论分析方法，构造了隐藏路径和领导Trie两类证明用的假设族，利用实验设计、可测可随机映射、上界与下界证明以及KL‑正则化目标的拉普拉斯形式来展示访问差异的影响。

**📊 数据集**

本文没有采用公开数据集，而是通过自定义的符号序列生成器（隐藏路径、领导Trie）和对prompt分布ρ的理论构造来验证结论。

**📈 对比分析**

对比方法：在同一奖励模型、KL正则化和策略空间下，比较仅root‑start（无重置）与允许弱本地重置的两种接口；结果表明，后者在多项式量级查询下即可恢复隐藏结构，而前者需指数级查询才能达到同等性能。

**⚠️ 局限性**

局限性：理论模型假设生成器接口严格受限且无外部反馈，实际大型预训练模型的接口与环境可能更复杂；此外实验仅基于人工构造的序列，未验证在真实自然语言生成任务中的可迁移性。

---

## 327. Community Driving-Safety Deterioration as a Push Factor for Public Endorsement of AI Driving Capability

**arXiv ID:** 2604.04775 | [PDF](https://arxiv.org/pdf/2604.04775v1)

**作者:** Amir Rafe `[一作]` (Texas State University), Subasish Das `[通讯]` (Texas State University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究检验了社区驾驶安全关注度（PCSC）是否会促使公众更支持人工智能驾驶能力，并探索其通过普遍AI取向的中介作用。

**💡 创新点**

创新点在于首次将社区驾驶安全关切视为推力因素，并揭示其对AI取向的负向影响与对AI驾驶评价的正向直接效应共存，形成风险溢出机制。

**🔧 技术方法**

采用加权结构方程模型（WLSMV）结合偏倚校正引导抽样、因子分析与交互式中介检验等技术。

**📊 数据集**

数据来源为美国Pew Research Center的ATP Wave 152全国概率样本，样本量约5,410人。

**📈 对比分析**

与传统技术接受模型相比，该方法在解释AI驾驶评价时显著提高了解释力（R²≈0.124），并通过七项稳健性检验证实结果稳健。

**⚠️ 局限性**

主要局限为横断面设计难以确定因果关系、对未测量混杂的敏感性较低以及样本仅覆盖美国，跨文化推广需进一步验证。

---

## 328. Precise Robot Command Understanding Using Grammar-Constrained Large Language Models

**arXiv ID:** 2604.04233 | [PDF](https://arxiv.org/pdf/2604.04233v1)

**作者:** Xinyun Huo `[一作]` (Florida State University), Xinyao Zhang `[通讯]` (Florida State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种混合语法约束的大语言模型（Hybrid Grammar-based LLM），通过先用微调的API LLM进行语义理解，再用语法驱动的结构化模型进行命令规范化与校验，实现工业机器人命令的准确生成。

**💡 创新点**

创新点在于将LLM的上下文理解能力与语法约束的结构化校验分离，形成多阶段反馈纠错循环，使模型在保持自然语言灵活性的同时保证生成命令的语法合法性和可执行性。

**🔧 技术方法**

使用了Meta‑Llama‑3‑8B‑Instruct作为基础模型，结合LoRA微调、Prompt Engineering、结构化语言模型（SLM）、语法解析器（基于Lark）以及反馈纠错机制。

**📊 数据集**

使用了HuRIC（Human Robot Interaction Corpus）数据集，包含自然语言指令与对应的机器人动作框架注解。

**📈 对比分析**

通过与三种基线模型（无微调API LLM、微调API LLM、纯语法驱动NLU）在HuRIC测试集上比较，Hybrid模型实现了48.48% 的 Exact Match、88.07% 的平均 JSON Similarity，显著优于其它方法（最高仅 36.4% EM、72.3% Similarity）。

**⚠️ 局限性**

主要限制包括数据集规模有限、含有问句/客气请求导致语义误判、统一语法规则导致结构歧义、仅使用单一基础模型等。

---

## 329. Automating Cloud Security and Forensics Through a Secure-by-Design Generative AI Framework

**arXiv ID:** 2604.03912 | [PDF](https://arxiv.org/pdf/2604.03912v1)

**作者:** Dalal Alharthi `[一作]` (University of Arizona), Ivan Roberto Kawaminami Garcia `[通讯]` (University of Arizona)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个统一的安全设计框架，结合PromptShield和CIAF实现了云取证自动化与LLM输入安全防护；

**💡 创新点**

通过面向本体的PromptShield提升LLM对抗注入攻击的鲁棒性，并利用结构化本体驱动的CIAF提升云取证的准确性与可解释性；

**🔧 技术方法**

本体驱动的Prompt验证、LLM推理、结构化模板化分析、Azure Monitor和AWS CloudWatch日志采集；

**📊 数据集**

AWS事件日志和Microsoft Azure虚拟机的性能与安全日志（含勒索软件攻击场景）；

**📈 对比分析**

与传统分类、注入攻击对照，PromptShield在Precision/Recall/F1/Accuracy上均超过93%，显著优于未加防护的LLM；

**⚠️ 局限性**

框架尚未实现实时取证，依赖预先构建的本体，可能限制LLM的表达灵活性，且对未知攻击场景的泛化需进一步验证。

---

## 330. On the "Causality" Step in Policy Gradient Derivations: A Pedagogical Reconciliation of Full Return and Reward-to-Go

**arXiv ID:** 2604.04686 | [PDF](https://arxiv.org/pdf/2604.04686v1)

**作者:** Nima H. Siboni `[一作]` `[通讯]` (Juna), Nima H. Siboni (Juna)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

重新推导REINFORCE算法，阐明 reward-to-go 产生的精确概率学机制。

**💡 创新点**

通过前缀轨迹分解与 score‑function 识别，直接将 reward-to-go 与每个 score 项关联，消除了传统“因果性”论点的隐藏假设。

**🔧 技术方法**

前缀轨迹分解、score‑function identity、条件期望的零值性质。

**📊 数据集**

无实验数据集，本文为理论性说明。

**📈 对比分析**

无实验对比，理论上与传统 REINFORCE 等价，未涉及性能评估。

**⚠️ 局限性**

仅提供概念性解释，对实现细节与算法改进贡献有限，可能对初学者的直观理解帮助不大。

---

## 331. SkillX: Automatically Constructing Skill Knowledge Bases for Agents

**arXiv ID:** 2604.04804 | [PDF](https://arxiv.org/pdf/2604.04804v1)

**作者:** Chenxi Wang `[一作]` (Ant Digital Technologies Ant Group), Shumin Deng `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个可插拔的技能知识库，自动从LLM代理经验中提取多层技能并进行迭代优化与扩展。

**💡 创新点**

创新点包括：①三层技能结构（规划、功能、原子）实现可组合的经验表示；②完全自动化的构建、迭代改进与探索扩展流程；③经验驱动的探索策略实现对训练集外技能的发现。

**🔧 技术方法**

核心技术：基于GLM‑4.6的rollout与文本提示式技能提取；嵌入检索与语义聚类实现技能合并；严格的过滤与合并操作保证库质量；经验引导探索与任务合成进一步扩展技能空间。

**📊 数据集**

使用了 AppWorld、BFCL‑v3 和 τ²‑Bench 这三大长期交互式工具调用基准的公开训练/测试拆分。

**📈 对比分析**

与四种基线（No‑memory、A‑Mem、AWM、ExpeL）对比，SkillKB 在三大基准上均提升 10–15% 的 Pass@4，尤其在弱模型（Qwen3‑32B、Kimi‑K2‑Instruct‑0905）上显著提升，验证了其强大的经验迁移与提升性能的效果。

**⚠️ 局限性**

局限性：依赖于相对稳定的工具环境，跨大范围域迁移不易；当前仅关注工具调用场景，对纯对话式交互等非工具型交互的适用性有限。

---

## 332. Beyond Imbalance Ratio: Data Characteristics as Critical Moderators of Oversampling Method Selection

**arXiv ID:** 2604.04541 | [PDF](https://arxiv.org/pdf/2604.04541v1)

**作者:** Yuwen Jiang `[一作]` (Guangzhou Institute of Science and Technology), Songyun Ye `[通讯]` (Guangzhou Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过12个受控实验、192个合成配置和17个真实数据集，系统检验了不平衡比率（IR）与过采样效果的关系，发现当数据特征被控制后，IR与过采样收益呈负相关，且类别可分性、簇结构等是主要调节因素。

**💡 创新点**

提出了“Context Matters”框架，首次通过受控实验量化类别可分性、簇结构、样本量等多因素对IR-过采样效应的调节作用，证明IR不应作为单一阈值来选择过采样方法。

**🔧 技术方法**

采用Gaussian混合模型生成受控数据，评估六种主流过采样方法与五种分类器，使用交叉验证、Pearson相关、Benjamini‑Hochberg FDR、Cohen d等统计检验方法。

**📊 数据集**

使用17个来自OpenML的公开数据集（涵盖医疗、金融、图像等多领域）以及192个基于高斯混合模型的人工合成数据集。

**📈 对比分析**

将各过采样方法与无采样基线在AUC‑ROC、AUC‑PR、F1、G‑Mean等指标下比较，发现低可分性、高簇数时过采样可提升约5个百分点，但在高可分性时可能导致性能下降，整体正向效果随IR降低而减弱。

**⚠️ 局限性**

仅限二分类场景，未包含深度生成方法，受限于高斯假设，无法直接推广到多类别或非连续特征，阈值为经验性需进一步验证，指标依赖性和天花板效应可能影响解释。

---

## 333. Teacher Professional Development on WhatsApp and LLMs: Early Lessons from Cameroon

**arXiv ID:** 2604.04139 | [PDF](https://arxiv.org/pdf/2604.04139v1)

**作者:** Vikram Kamath Cannanure `[一作]` (Saarland University), Ingmar Weber `[通讯]` (Saarland University)

**通讯引用:** 11451 | [OpenAlex ID](https://openalex.org/A5033656008)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在喀麦隆开展了一项教师专业发展（TPD）项目，利用WhatsApp聊天机器人与LLM生成的模块化内容进行教学，而传统的线上表单则作为对照组。

**💡 创新点**

创新之处在于把AI内容嵌入教师已熟悉的WhatsApp平台，并通过LLM支持的仪表板自动化生成教学与反思材料，兼顾双语（英法）需求与低资源环境。

**🔧 技术方法**

使用技术包括：WhatsApp聊天机器人（菜单式脚本交互）、LLM（如ChatGPT/Meta AI）生成的内容、在线表单（Google Form）以及基于纸质的问卷与访谈记录。

**📊 数据集**

数据集主要为实地收集的教师样本信息（47名小学教师的使用日志、问卷和访谈），未使用公开的大规模语言或教育数据集。

**📈 对比分析**

通过配对t检验比较聊天机器人与在线表单在可用性、易学性和整体体验三项指标，聊天机器人在可用性（p=0.046，Cohen’s d=0.37）和整体体验（p=0.044，d=0.32）上显著优于表单，易学性差异不显著。

**⚠️ 局限性**

局限性包括：网络连接不稳定、预付费数据成本高、双语支持不足、交互流程固定、对AI信任与反思使用的引导不足，且研究时间短、样本量有限。

---

## 334. Align Your Structures: Generating Trajectories with Structure Pretraining for Molecular Dynamics

**arXiv ID:** 2604.03911 | [PDF](https://arxiv.org/pdf/2604.03911v1)

**作者:** Aniketh Iyengar `[一作]` (Stanford University), Stefano Ermon `[通讯]` (Stanford University)

**通讯引用:** 24508 | [OpenAlex ID](https://openalex.org/A5091179481)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本工作提出了一种两阶段的扩散模型框架，用结构预训练（在大规模 conformer 数据集上训练的空间模型）与时间插值器（EGInterpolator）结合，能够从有限的 MD 轨迹数据中生成物理上合理的分子动力学轨迹。

**💡 创新点**

创新点包括：①利用结构预训练显著缓解 MD 数据稀缺问题；②设计了 SE(3)-等变的时间插值器，线性混合结构与时间网络输出，形成可调节的时间依赖性，并在理论上证明其隐式中间分布；③通过两阶段训练与多块插值器实现高效的时间信息融合。

**🔧 技术方法**

技术方法包括：几何扩散模型、SE(3)-equivariant 图卷积（EGCL）、时间注意力层（ET）、线性插值机制、两阶段训练策略（结构预训练+MD 微调）。

**📊 数据集**

使用的数据集包括 GEOM-QM9 与 GEOM-Drugs 的 conformer 资料，用于结构预训练；QM9 与 Drugs 的 MD 轨迹（5 ns 轨迹、5×10⁶ 步）用于微调；Timewarp tetrapeptide 数据集和 ATLAS 蛋白质单体数据集用于进一步验证。

**📈 对比分析**

通过与 GeoTDM、AR+EGNN、AR+ET、AR+GeoTDM 等基线对比，采用 JSD、键长/键角/扭角分布、TICA、MSM 以及势能等指标评估。结果显示 EGInterpolator（尤其是 Casc 版本）在 JSD、键角/键长/扭角、TICA 以及 MSM 拥有显著更低误差，能量分布更贴近 MD Oracle，且在无条件生成、前向模拟与插值任务中均优于基线。

**⚠️ 局限性**

局限性在于仍需 MD 轨迹数据进行微调；时间插值器对超长时间尺度或极大分子系统的泛化尚未充分验证；对非有机化合物或高分子体系的适用性需要进一步研究；以及模型在极端能量景观或高度限制的分子构型上的表现未完全确定。

---

## 335. HEDGE: Heterogeneous Ensemble for Detection of AI-GEnerated Images in the Wild

**arXiv ID:** 2604.03555 | [PDF](https://arxiv.org/pdf/2604.03555v1)

**作者:** Fei Wu `[一作]` (Shanghai Jiao Tong University), Fengjun Guo `[通讯]` (INTSIG Information)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个三路异质集成框架HEDGE，用于在真实环境中鲁棒检测AI生成图像

**💡 创新点**

创新点在于结构化多维异质性：在训练数据与增强、输入分辨率、以及骨干网络上引入多样性，并通过logit空间加权融合与轻量级双门控机制提升决策稳健性

**🔧 技术方法**

采用DINOv3-Huge作为主骨干，进行分阶段继续训练并加强数据扩充与噪声增强；引入448×448高分辨率分支；加入MetaCLIP2-Giant骨干以降低相关性；logit空间加权融合及双门控策略

**📊 数据集**

使用官方比赛数据、SoFake-OOD、RRDataset、Chameleon、GenImage、AIGIBench等多源数据集进行训练；在GenImage、AIGCDetect、DRCT-2M、Synthbuster、EvalGEN、Chameleon、RRDataset、AIGIBench、BFree-Online、SynthWildx、WildRF、RealChain等标准与野外、链式失真基准进行评估

**📈 对比分析**

与多种最先进方法（NPR、UnivFD、FatFormer、SAFE、C2P-CLIP、AIDE、DRCT、Aligned、B-Free、DDA、MIRROR、REM等）进行对比，HEDGE在标准基准平均B.Acc 98.3%，在野外基准平均B.Acc 96.8%，在链式失真基准B.Acc 93.2%，均显著优于对手，且在JPEG压缩、尺寸缩放等扰动下保持高稳健性

**⚠️ 局限性**

在BFree-Online等局部修补、Self-Conditioning场景下表现不佳；对极端多级失真仍有提升空间；同时，模型规模与推理时间相对较大

---

## 336. Evaluating Future Air Traffic Management Security

**arXiv ID:** 2604.04293 | [PDF](https://arxiv.org/pdf/2604.04293v1)

**作者:** Konstantinos Spalas `[一作]` (University of Peloponnese), Konstantinos Spalas `[通讯]` (University of Peloponnese)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5115458175)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

评估了LDACS系统中基于PUF的轻量级身份认证机制，并提出了基于PQC的PKI替代方案。

**💡 创新点**

创新点在于系统性分析了PUF模型被CMA-ES预测与量子计算预影像攻击的风险，并提出量子抗性PKI方案。

**🔧 技术方法**

使用了CMA-ES机器学习算法、哈希函数、密钥封装机制以及量子预影像算法。

**📊 数据集**

论文未使用公开数据集，而是基于理论分析与算法仿真。

**📈 对比分析**

通过计算错误率（98%）和量子搜索复杂度（O(2^42)）对攻击成本进行评估，显示PUF方案易受攻击；而PKI方案在安全性与延迟方面优于原方案。

**⚠️ 局限性**

局限性包括PUF老化导致的稳定性问题、对量子计算的依赖以及对攻击模型假设的简化。

---

## 337. LOCARD: An Agentic Framework for Blockchain Forensics

**arXiv ID:** 2604.04211 | [PDF](https://arxiv.org/pdf/2604.04211v1)

**作者:** Xiaohang Yu `[一作]` (Imperial College London), William Knottenbelt `[通讯]` (Imperial College London)

**通讯引用:** 5731 | [OpenAlex ID](https://openalex.org/A5050119476)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

该论文提出并实现了Agentic Blockchain Forensics框架LOCARD，用于跨链交易追踪。

**💡 创新点**

创新点在于把区块链取证视为序贯决策过程，采用三核认知架构和结构化信念状态，首次实现自适应取证。

**🔧 技术方法**

使用大型语言模型（GPT‑4o）结合ReAct、LangGraph等多代理协同，配合跨链链桥图分析和价值/时间约束的追踪启发式。

**📊 数据集**

数据集为新发布的Thor25（151k跨链记录）及其高价值子集Thor25HF，包含Bybit黑客事件的标注。

**📈 对比分析**

在单转移追踪任务上，LOCARD与确定性启发式基线相比召回率≥93%，Hit@50>90%，仅额外$0.20/次且耗时约1–2分钟。

**⚠️ 局限性**

限制在于当前仅在已知规则较强的THORChain场景验证，候选评分简单，难以应对低价值噪声或更复杂的隐私链环境。

---

## 338. CoopGuard: Stateful Cooperative Agents Safeguarding LLMs Against Evolving Multi-Round Attacks

**arXiv ID:** 2604.04060 | [PDF](https://arxiv.org/pdf/2604.04060v1)

**作者:** Siyuan Li `[一作]` (Shanghai Jiao Tong University), Xiu Su `[通讯]` (Central South University)

**通讯引用:** 581 | [OpenAlex ID](https://openalex.org/A5011334334)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于协作多智能体的有状态多轮防御框架，专门针对LLM在多轮交互中不断进化的对抗攻击；

**💡 创新点**

创新点在于：①维护全局防御状态并在每轮动态更新；②将防御拆分为四类专门智能体（延迟、诱导、取证、协调），实现跨轮协同与自适应；③通过结构化提示与状态条件化提升对抗鲁棒性；④构建了EMRA基准（5200个多轮攻击样本，覆盖八种策略）供评测；

**🔧 技术方法**

采用LLM驱动的智能体模块，利用可解释的结构化提示、指数衰减的检测评分、记忆化解码器、取证聚合器和协同策略网络；在实现上结合了强化学习/提示工程等技术；

**📊 数据集**

使用EMRA基准，涵盖八类攻击策略共5200条多轮序列；实验还涉及多种LLM后端（GPT‑3.5‑turbo、GPT‑4、Gemini‑Flash等）；

**📈 对比分析**

与五大现有防御（PAT、RPO、GoalPriority、Self‑Reminder、SecurityLingua）进行对比；实验显示在多轮攻击、恶意问题、重述问题及完整诱导问答上，所提框架的攻击成功率显著下降（约70%–80%降低），同时攻击者在对话中所消耗的token显著增加，表明能有效消耗对手资源；

**⚠️ 局限性**

局限性包括：①依赖于LLM内部的推理与提示质量，可能在不同模型上需要重新调参；②实验主要基于人工构造的基准，真实世界场景下的表现尚未验证；③虽然对正常用户影响较小，但在极端对话场景下仍可能出现略微的延迟或信息量减少；

---

## 339. Search-Bound Proximity Proofs: Binding Encrypted Geographic Search to Zero-Knowledge Verification

**arXiv ID:** 2604.03902 | [PDF](https://arxiv.org/pdf/2604.03902v1)

**作者:** Yoshiyuki Ootani `[一作]` `[通讯]`, Yoshiyuki Ootani

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了搜索授权证明（SAP）安全概念，解决了加密地理搜索与零知识接近证明之间的授权溯源缺口，并实现了可分离绑定的完整协议。

**💡 创新点**

创新点在于：①定义了 SAP 的三项安全属性（P1–P3）；②通过 audit 重新关联攻击展示了证明外部绑定的脆弱性；③提出了无电路修改、可分离的绑定方案（Session nonce、Merkle 根、签名收据），实现了属性级故障隔离。

**🔧 技术方法**

采用的技术包括：GridSE 可搜索对称加密、Groth16 零知识证明、SHA‑256 哈希链、Merkle 树承诺、服务器签名收据、HMAC 令牌生成等。

**📊 数据集**

使用了两类数据集：①合成随机分布的地理点；②真实世界的 110,776 条 OpenStreetMap POI（东京区域）。

**📈 对比分析**

通过 9 种协议变体与 6 类攻击对比实验，Full+receipt 方案通过全部攻击；协议路径平均延迟仅比 GridSE 低 7%；证明生成时间约 125 ms（移动端 42–125 ms）；Merkle 树构建 54 ms（50k 点）；单实例原型保持 O(1) 会话状态，支持离线审计。

**⚠️ 局限性**

限制包括：SSE 访问模式泄露（可通过 ORAM 缓解）、对位置伪造无防护、恶意服务器情形仍需可验证查询、跨实例事务需外部原子性保障。

---

## 340. A Generative Foundation Model for Multimodal Histopathology

**arXiv ID:** 2604.03635 | [PDF](https://arxiv.org/pdf/2604.03635v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 341. Noise Steering for Controlled Text Generation: Improving Diversity and Reading-Level Fidelity in Arabic Educational Story Generation

**arXiv ID:** 2604.03380 | [PDF](https://arxiv.org/pdf/2604.03380v1)

**作者:** Haziq Mohammad Khalid `[一作]` (American University of Sharjah), Imran Zualkernan `[通讯]` (American University of Sharjah)

**通讯引用:** 4106 | [OpenAlex ID](https://openalex.org/A5035716630)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在推理阶段向小型阿拉伯语Transformer模型的内部表示注入高斯噪声，研究如何在保持早期阅读评估（EGRA）严格词汇和结构约束的前提下提升故事多样性。

**💡 创新点**

创新点在于提出“噪声驱动”方法：对残差流和注意力熵进行噪声注入，并通过余弦衰减自适应调节，既避免了传统高温采样导致的崩溃，又能在不影响质量与约束的情况下显著提升多样性。

**🔧 技术方法**

技术实现包括：Embedding Noise、Attention-Logit Noise、Residual Stream Noise、Attention Entropy Noise Injection (AENI) 等四种内部噪声注入策略，并使用基于模型内部激活 RMS 的标准化方法来统一噪声幅度。

**📊 数据集**

实验使用五个7–9B参数的阿拉伯语指令型小模型（ALLaM 7B、AceGPT 8B、Fanar 9B、Jais 8B、Phi-4-mini），在 EGRA 约束下生成 50 条故事，评估数据主要来自模型生成的文本本身。

**📈 对比分析**

对比方法包括无噪声基线、高温采样 (T=1.8) 与顶点采样，评估指标包括模态崩溃率、Vendi Score（多样性）、LLM 评判的质量分、约束违例数和阅读等级。结果显示 Residual Stream Noise 和 AENI 在所有模型上保持 0% 崩溃率，显著提升多样性同时质量、约束和阅读等级几乎不受影响；高温采样则导致高崩溃率、质量下降和阅读等级提升。

**⚠️ 局限性**

主要局限在于评估依赖 GPT 作为自动判别者，可能存在偏差和主观性；缺乏人工评审和跨语言泛化验证；噪声策略对不同模型的鲁棒性差异大，需进一步探讨层级目标化和更低资源语言的适用性。

---

## 342. MinerU2.5-Pro: Pushing the Limits of Data-Centric Document Parsing at Scale

**arXiv ID:** 2604.04771 | [PDF](https://arxiv.org/pdf/2604.04771v1)

**作者:** Bin Wang `[一作]` (Shanghai Artificial Intelligence Laboratory), Conghui He `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本工作通过构建一套完整的数据工程管线，并在保持1.2B参数的解耦式VLM架构不变的前提下，提升了OmniDocBench v1.6的整体得分；

**💡 创新点**

创新点包括：① 以覆盖度、信息量、标注质量为维度的Data Engine（DDAS、CMCV、Judge‑and‑Refine与专家标注闭环）实现数据规模从不足1千万页扩展至65.5M页并系统性提升标注准确性；② 三阶段渐进式训练策略（大规模SFT→硬样本微调→GRPO对齐）与数据质量梯度匹配；③ 通过Multi‑Granularity Adaptive Matching校正评测偏差，并新增Hard子集构建更具判别力的评测协议。

**🔧 技术方法**

采用解耦式VLM架构（NaViT 675M 视觉编码器 + Qwen 0.5B 语言模型），多模型交叉一致性验证（CMCV）、渲染‑验证循环的Judge‑and‑Refine、强化学习的Group Relative Policy Optimization（GRPO）以及多粒度自适应匹配（MGAM）等技术；

**📊 数据集**

训练数据来源于扩展后的OmniDocBench v1.6（Base、Hard、Full），共65.5M页面的自动标注数据与192K页专家标注的Hard样本；此外在元素级评测中引用CPE、HWE、SCE、SPE等子集和其他表格/公式基准。

**📈 对比分析**

在统一环境下与多款专用与通用VLM（GLM‑OCR、PaddleOCR‑VL‑1.5、Gemini 3 Pro、Qwen3‑VL‑235B等）对比，最终模型在OmniDocBench v1.6整体得分达95.69（基线92.98提升2.71），在Hard子集中得分94.08，公式CDM 97.29，表格TEDS 93.42等指标均居首位；阶段消融显示第1阶段贡献最大。

**⚠️ 局限性**

局限性在于评测仍受格式与结构歧义影响，需进一步开发语义等价评测；评测集覆盖范围主要为主流场景，缺乏行业垂直域；本工作聚焦内容提取，尚未覆盖文档的结构关系与语义理解。

---

## 343. Interpreting Video Representations with Spatio-Temporal Sparse Autoencoders

**arXiv ID:** 2604.03919 | [PDF](https://arxiv.org/pdf/2604.03919v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 344. M2StyleGS: Multi-Modality 3D Style Transfer with Gaussian Splatting

**arXiv ID:** 2604.03773 | [PDF](https://arxiv.org/pdf/2604.03773v1)

**作者:** Xingyu Miao `[一作]` (Durham University), Yang Long `[通讯]` (Durham University)

**通讯引用:** 4985 | [OpenAlex ID](https://openalex.org/A5002360303)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于3D高斯溅射的实时3D风格迁移方法M^2StyleGS，支持文本与图像多模态输入。

**💡 创新点**

核心创新包括使用ODE驱动的细化流(Subdivisive Flow)实现CLIP与VGG特征的精准对齐，以及观察损失与抑制损失的辅助约束。

**🔧 技术方法**

结合3D高斯溅射、CLIP多模态编码、VGG特征提取、AdaIN风格映射、ODE流模拟、Euler积分、观察损失、抑制损失与判别器等技术。

**📊 数据集**

在LLFF与Tanks & Temples两大公开3D数据集上进行训练与评估。

**📈 对比分析**

与现有单图像与文本引导的SOTA方法对比，M^2StyleGS在多视角一致性（LPIPS、RMSE）上提升最高32.92%，视觉质量显著优于对手。

**⚠️ 局限性**

仍受限于对极端纹理或颜色变换的鲁棒性不足，且在极大场景尺寸下实时渲染仍面临算力挑战。

---

## 345. Physics-Informed Untrained Learning for RGB-Guided Superresolution Single-Pixel Hyperspectral Imaging

**arXiv ID:** 2604.03572 | [PDF](https://arxiv.org/pdf/2604.03572v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 346. The Persuasion Paradox: When LLM Explanations Fail to Improve Human-AI Team Performance

**arXiv ID:** 2604.03237 | [PDF](https://arxiv.org/pdf/2604.03237v1)

**作者:** Ruth Cohen `[一作]` (Bar-Ilan University), Sarit Kraus `[通讯]` (Bar-Ilan University)

**通讯引用:** 18510 | [OpenAlex ID](https://openalex.org/A5103213461)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型（LLM）生成的自然语言解释在两类任务（RAVEN视觉推理和LSAT逻辑推理）中的效果，探究其对人机协作准确性、信任和错误恢复的影响。

**💡 创新点**

提出了“说服悖论”（Persuasion Paradox），揭示了流利解释会提升用户信任却不一定提升甚至削弱任务准确性，并发现解释效果随任务模态而异。

**🔧 技术方法**

采用多阶段揭示实验、受试者对照实验、概率信息展示、可视化热图、专家写作解释以及基于概率的选择性自动化策略。

**📊 数据集**

使用RAVEN矩阵数据集和LSAT逻辑题集，分别包含视觉推理和语言逻辑问题。

**📈 对比分析**

通过对比预测+解释、预测+热图、预测+概率、预测+专家解释等条件，并与人类单独作答及选择性自动化策略进行对照，发现RAVEN中概率信息和自动化策略显著提高准确率（最高69.5%），而LSAT中LLM解释最高达72.5%。

**⚠️ 局限性**

局限性包括只针对两类人工任务，模型概率在LSAT中信息不足，未验证在真实世界应用中的效果，且对解释质量与用户工作负荷的进一步分析仍待探索。

---

## 347. What Do We Need for an Agentic Society?

**arXiv ID:** 2604.03938 | [PDF](https://arxiv.org/pdf/2604.03938v1)

**作者:** Kwon Ko `[一作]` (Stanford University), Hyoungwook Jin `[通讯]` (University of Michigan)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出“Agentic Society”概念，阐述将智能代理属性扩展到实体物体（Agentic Objects），并通过青少年校园欺凌案例分析其协调失败模式，归纳出九个开放性问题；

**💡 创新点**

将传统软件代理的四大属性（自治、反应性、前瞻性、社交性）迁移到具体现实物体，首次系统化探讨物体间跨域协同的感知、判断与行动三阶段，并提出相应的边界、敏感度、隐私、聚合、冲突、法定责任等关键问题；

**🔧 技术方法**

主要采用理论框架与案例分析方法，结合已有多代理系统、物联网与隐私理论（如Nissenbaum的情境完整性），无具体算法实现；

**📊 数据集**

无实验数据集；研究基于构造性案例（Peter的校园/家庭情境）和文献综述；

**📈 对比分析**

无量化实验对比，本文通过三种失败情境（误报、死锁、恶意干扰）展示现有代理属性在物体协同中的不足，未给出性能指标；

**⚠️ 局限性**

局限性在于仅基于单一示例构建问题，缺乏实证验证；未讨论系统实现细节、时延与可扩展性；对不同应用场景（老年护理、慢性病管理等）可能需要进一步调整；

---

## 348. OrbitTransit: Traffic Delivery and Diffusion for Earth Observation via Satellite Mobility

**arXiv ID:** 2604.04368 | [PDF](https://arxiv.org/pdf/2604.04368v1)

**作者:** Haoyuan Zhao `[一作]` (Simon Fraser University), Jiangchuan Liu `[通讯]` (Simon Fraser University)

**通讯引用:** 20710 | [OpenAlex ID](https://openalex.org/A5039311485)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 OrbitTransit，一种混合 PCO‑ISL 交付系统，通过轨道‑为‑节点建模、交通扩散和争用避免，实现低地球轨道卫星网络中地面站负载平衡与轨道资源优化。

**💡 创新点**

创新点包括：①将卫星级网络抽象为轨道级节点，显著降低拓扑复杂度；②设计基于轨道的交通扩散算法，消除地面站分布偏斜导致的热点；③引入争用避免的 PCO‑ISL 路由方案，协调携带与转发时机；④将 GS 选择与空间路由统一框架，兼顾能耗、延时与可靠性。

**🔧 技术方法**

使用技术：轨道‑为‑节点（OAN）框架、混合 PCO‑ISL 路由算法、最小成本最大流（MCMF）调度、线性/整数规划基准、基于遥测的控制平面、Python+PyEphem 模拟实现。

**📊 数据集**

数据集：Starlink、OneWeb、Telesat 的真实两行元素数据；FCC 公布的地面站位置；按文献统计的 EO 任务量（2–10 TB）与截止时间（20–180 min）构成的合成流量；仿真场景涵盖不同星座规模与高度。

**📈 对比分析**

对比方法：与 Nearest/SusCO 两种 GS 选择结合 Umbra/SHORT 两种路由算法进行四维评估（GS 负载、任务失败、卫星能耗、路径长度）。OrbitTransit 能降低约 47%（对 Starlink）/ 50%（对 OneWeb/Kuiper）的电池能耗，任务失败率下降 1.09×，队列延时保持在 4.75 ms 左右，并在多种星座下接近基准最优解，且计算开销显著低于求解最优模型。

**⚠️ 局限性**

局限性：依赖及时、完整的遥测信息；对极端高流量或突发地面站容量下降的应对有限；主要针对延迟容忍的 EO 任务，实时任务处理能力未充分验证；在实际部署中对多星座大规模扩展的细节仍待进一步评估。

---

## 349. Determined by User Needs: A Salient Object Detection Rationale Beyond Conventional Visual Stimuli

**arXiv ID:** 2604.03526 | [PDF](https://arxiv.org/pdf/2604.03526v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 350. Strengthening Human-Centric Chain-of-Thought Reasoning Integrity in LLMs via a Structured Prompt Framework

**arXiv ID:** 2604.04852 | [PDF](https://arxiv.org/pdf/2604.04852v1)

**作者:** Jiling Zhou `[一作]` (University of Turku), Jouni Isoaho `[通讯]` (University of Turku)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证了一个结构化提示工程框架，以提升本地LLM在SDN流量的DDoS检测中的链式思考推理完整性和准确性。

**💡 创新点**

将16个控制因子分为四个维度，系统化地约束推理流程，减少幻觉并增强可解释性，首次在安全场景中将结构化提示与人类评估相结合。

**🔧 技术方法**

基于Chain-of-Thought (CoT) 提示，加入系统/用户层级控制，利用自监督式检验与人类评分评估指标。

**📊 数据集**

使用公开的DDoS SDN流量数据集（400条样本，23特征）。

**📈 对比分析**

通过与无框架提示对比，采用多模型（Gemma、Llama、Qwen、GPT-OSS、ChatGPT）和人类评估，发现准确率提升1–5%，推理质量提升至40%（小模型）或8–12%（大模型）。

**⚠️ 局限性**

局限在仅测试DDoS检测、数据量有限、仅针对本地LLM，且框架依赖人工/ChatGPT生成的提示，缺乏动态自适应和跨任务验证。

---

## 351. Safe and Near-Optimal Gate Control: A Case Study from the Danish West Coast

**arXiv ID:** 2604.04545 | [PDF](https://arxiv.org/pdf/2604.04545v1)

**作者:** Martin Kristjansen `[一作]` (Aalborg University), Christian Schilling `[通讯]` (Aalborg University)

**通讯引用:** 851 | [OpenAlex ID](https://openalex.org/A5074526208)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用数字孪生与在线强化学习，对Ringkøbing Fjord的14道闸门进行控制，实现水位安全与多重性能目标的满足。

**💡 创新点**

将在线强化学习与数字孪生结合，使用分区细化的Q学习并实时更新天气预报，构建安全权重调优与基准对比的完整框架。

**🔧 技术方法**

数字孪生模型（使用Uppaal/SMT等工具）、Q-learning（分区细化）、在线学习、基于成本函数的决策树生成。

**📊 数据集**

基于2018年历史海平面、风速、船舶到达和入海流量数据，构造三种三天海平面场景（正常、低水、高水）。

**📈 对比分析**

与经验性基准手动控制器对比，评估安全比例、鱼类迁移率、最大船舶等待时间和闸门操作次数；学习控制器始终满足安全要求，鱼类迁移和船舶等待相似，闸门操作次数略增。

**⚠️ 局限性**

受限于动作空间仅三种开关、未考虑盐度、水面非平坦、精细入海流量模型以及缺乏实地部署验证，安全权重仍需进一步优化。

---

## 352. Multi-Modal Sensor Fusion using Hybrid Attention for Autonomous Driving

**arXiv ID:** 2604.04797 | [PDF](https://arxiv.org/pdf/2604.04797v1)

**作者:** Mayank Mayank `[一作]` (Mercedes-Benz AG), Abhinav Valada `[通讯]` (University of Freiburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了MMF-BEV框架，结合摄像头和毫米波雷达的BEV融合，使用变形注意力实现3D目标检测。

**💡 创新点**

创新点包括：①系统研究单模态变形自注意力的效果；②双向多层混合融合模块；③可解释的传感器贡献分析；④两阶段训练策略以保证融合稳定性。

**🔧 技术方法**

采用了BEVDepth、RadarBEVNet、Deformable Self-Attention、Deformable Cross-Attention、CBR融合层、CenterPoint检测头、RCS-aware BEV散射等技术。

**📊 数据集**

使用View-of-Delft（VoD）4D雷达数据集进行训练与评估。

**📈 对比分析**

与单模态（C-only、R-only）、直接拼接（MMF-BEV*）以及RCBEVDet/RCFusion基线相比，MMF-BEV在全注释区取得48.92% mAP、ROI区69.21% mAP，显著优于单模态且与现有融合方法相近或略优。

**⚠️ 局限性**

局限性：仅使用单向单目摄像头导致深度估计受限；雷达点云稀疏影响长距离和小目标检测；长距离检测受VoD样本不足限制，未来需多视角相机和更丰富长距离数据。

---

## 353. Integer-Only Operations on Extreme Learning Machine Test Time Classification

**arXiv ID:** 2604.04363 | [PDF](https://arxiv.org/pdf/2604.04363v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 354. Benchmarking and Evaluating VLMs for Software Architecture Diagram Understanding

**arXiv ID:** 2604.04009 | [PDF](https://arxiv.org/pdf/2604.04009v1)

**作者:** Shuyin Ouyang `[一作]` (Kings College London), Joost Noppen `[通讯]` (British Telecom)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了软件架构图理解基准SADU，旨在评估视觉语言模型对架构图的识别与推理能力。

**💡 创新点**

创新点在于构建专门针对架构图的结构化标注、问题答案对，并设计了覆盖计数与检索两类推理任务的评测范式。

**🔧 技术方法**

主要使用了多模态视觉语言模型（Gemini、Claude、GPT、Qwen系列）进行实验，探究其对图形元素定位、关系推断和文本理解的性能。

**📊 数据集**

数据集包含154张真实场景的行为、结构和ER三类软件架构图，来自Azure、PyUNML、Mermaid、Lucidchart、Figma、Miro等公开资源。

**📈 对比分析**

与11个最新VLMs对比，Gemini-3-flash-preview在SADU上取得最高准确率70.18%，其它模型如Claude、GPT、Qwen均落后，整体表现仍处于中等水平。

**⚠️ 局限性**

局限性包括对复杂箭头布局和空间关系的理解不足、缺乏精细的几何注释、模型对长距离关系识别能力有限，导致在高难度样例上准确率显著下降。

---

## 355. Drift-Based Policy Optimization: Native One-Step Policy Learning for Online Robot Control

**arXiv ID:** 2604.03540 | [PDF](https://arxiv.org/pdf/2604.03540v1)

**作者:** Yuxuan Gao `[一作]` (University of Science and Technology of China), Yanyong Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8140 | [OpenAlex ID](https://openalex.org/A5053344541)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种 Drift-Based Policy (DBP) 与 Drift-Based Policy Optimization (DBPO)，实现一阶网络推断的机器人控制。

**💡 创新点**

创新点在于将迭代细化迁移至训练阶段的固定点漂移目标，构建原生一阶生成模型，并提供兼容的随机接口实现在线强化学习。

**🔧 技术方法**

采用漂移模型框架、抗对称漂移场、固定点回归目标、Gaussian 稳定噪声、PPO 等技术。

**📊 数据集**

使用多种数据集：Diffusion Policy 仿真套件、Adroit+Meta-World 3D 点云任务、RoboMimic 图像/低维、D4RL 运动学任务，以及真实双臂 UR5 的 Teleoperation 示范。

**📈 对比分析**

与多步扩散策略、MP1/OMP 等一阶基线对比，在仿真上 100 倍速度提升、成功率从 79% 提升至 83%，在 37 任务上平均成功率 88.4% 超过 82.3%，在线 PPO fine‑tune 后在 RoboMimic 与 D4RL 上获得最高回报；实测双臂控制 105.2 Hz，成功率 75%。

**⚠️ 局限性**

限制包括对稀疏奖励任务的样本效率仍待提升、对接触复杂长周期任务的适应性有限、需要更多大规模实验验证。

---

## 356. Three Phases of Expert Routing: How Load Balance Evolves During Mixture-of-Experts Training

**arXiv ID:** 2604.04230 | [PDF](https://arxiv.org/pdf/2604.04230v1)

**作者:** Charafeddine Mouzouni `[一作]` `[通讯]` (Open Institute of Technology), Charafeddine Mouzouni (Open Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将Mixture-of-Experts路由建模为拥塞博弈，定义有效拥塞系数γ_eff并在训练检查点上追踪其值，揭示三相跃迁（surge–stabilization–relaxation）。

**💡 创新点**

首次提出可识别且可分解的γ_eff，证明其在不同层面上可观测，并用多类型均衡与范围诊断工具刻画动态平衡及其失效边界。

**🔧 技术方法**

采用均衡理论（单/多类型MFG）、潜在函数证明、软最大化对应关系、Bootstrap CI估计、k‑means聚类、以及对比实验中的L¹误差评估。

**📊 数据集**

使用公开的OLMoE‑1B‑7B与OpenMoE‑8B训练检查点（总计约2×10⁶步，包含多层MoE），在每个检查点采样若干文本进行路由与质量估计。

**📈 对比分析**

通过与Uniform、Temperature‑scaled Softmax、单型MFG、以及多类型MFG的L¹误差对比发现，单型MFG与Softmax在收敛模型几乎等价，MT‑MFG在所有层提升约29.6%（最高≈47%）。

**⚠️ 局限性**

仅在两款模型验证，单型MFG在静态预测上无优势；假设线性拥塞，token聚类方法非原则；当K/M<1时模型不在MFG适用范围。

---

## 357. CCA Reimagined: An Exploratory Study of Large Language Models for Congestion Control

**arXiv ID:** 2604.03857 | [PDF](https://arxiv.org/pdf/2604.03857v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 358. Decoding Student Dialogue: A Multi-Dimensional Comparison and Bias Analysis of Large Language Models as Annotation Tools

**arXiv ID:** 2604.04370 | [PDF](https://arxiv.org/pdf/2604.04370v1)

**作者:** Jie Cao `[一作]` (University of North Carolina at Chapel Hill), Jifan Yu `[通讯]` (Tsinghua University)

**通讯引用:** 619 | [OpenAlex ID](https://openalex.org/A5038246814)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估GPT-5.2与Gemini-3在学生与AI对话自动注释中的准确性，并对少样本、单代理自省和多代理反思三种提示策略在不同教育层次、学科和四维注释维度下进行系统比较和偏差分析。

**💡 创新点**

首次将多提示策略与多维度注释框架相结合，揭示不同模型在K‑12与大学、各学科以及情感、认知、元认知和行为维度中的准确性差异与偏差模式。

**🔧 技术方法**

采用大语言模型提示工程（少样本、单代理自省、多代理反思）以及广义线性混合模型（GLMM）统计分析，并以人类双标注为基准计算准确率。

**📊 数据集**

使用800条学生utterances（Biology K‑12、AGI、Psychology）和公开CoMTA数学数据，按教育层次与学科随机抽样，构成评估语料。

**📈 对比分析**

通过GLMM比较提示方法、教育层次、学科与维度对准确率的影响，整体准确率约79–84%；提示复杂度提升不显著，K‑12对模型更友好，情感维度准确率最高，认知维度最低。

**⚠️ 局限性**

研究受限于仅使用标准版模型、样本量有限、未进行模型微调或多标签框架实验，导致偏差未能完全消除且结果对其他模型或更大规模数据的推广性仍需进一步验证。

---

## 359. Adversarial Robustness Analysis of Cloud-Assisted Autonomous Driving Systems

**arXiv ID:** 2604.04349 | [PDF](https://arxiv.org/pdf/2604.04349v1)

**作者:** Maher Al Islam `[一作]` (West Virginia University), Amr S. El-Wakeel `[通讯]` (West Virginia University)

**通讯引用:** 375 | [OpenAlex ID](https://openalex.org/A5021410318)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个硬件循环的云辅助自动驾驶测试平台，利用实时感知、控制与通信模块，系统性评估了白盒对抗攻击（FGSM、PGD）和网络攻击（延迟、丢包）对闭环控制安全性的影响。

**💡 创新点**

首次在同一实验环境下同时考虑感知层对抗攻击与网络层干扰的交叉效应，提出交叉层鲁棒性评估框架，并展示了两层攻击共同放大系统失稳的机制。

**🔧 技术方法**

使用 YOLOv8 对象检测模型，FGSM 与 PGD 白盒攻击，MITRE ATT&CK 框架模拟网络攻击，ROS 框架实现车辆-云通信，PID 控制器实现速度/角速度指令，数据可视化与统计分析工具。

**📊 数据集**

基于 2,414 张 Duckiebot RGB 帧的自建数据集，包含车道、交通标志、灯光、车辆等标注，采用 70/30 的训练/测试拆分。

**📈 对比分析**

通过比较干净基线、不同 ε 下 FGSM/PGD 的检测精度、召回率以及网络延迟/丢包对车辆轨迹偏移、停止合规率的影响来评估性能。结果显示 PGD 将检测精度从 0.73 降至 0.22，召回率从 0.68 降至 0.15；网络延迟 250 ms 或丢包 5% 时，车辆停止违规率显著上升，轨迹出现明显偏移。

**⚠️ 局限性**

局限性包括：实验仅在小规模 Duckiebot/云服务器上进行；对抗攻击仅为白盒且不考虑物理攻击或黑盒场景；网络攻击模型简化；未评估多车协同与更大规模 IoV 环境下的交叉效应。

---

## 360. A Family of Open Time-Series Foundation Models for the Radio Access Network

**arXiv ID:** 2604.04271 | [PDF](https://arxiv.org/pdf/2604.04271v1)

**作者:** Ioannis Panitsas `[一作]` (Yale University), Leandros Tassiulas `[通讯]` (Yale University)

**通讯引用:** 28278 | [OpenAlex ID](https://openalex.org/A5014892027)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出TimeRAN框架，将RAN时序分析任务统一到一个轻量化Transformer编码器+少数任务头的多任务学习体系，利用大规模TimeRAN DataPile进行预训练，实现跨任务迁移与零样本推理；

**💡 创新点**

创新点在于：①用一个通用模型取代多任务专用模型，消除模型碎片化与维护成本；②基于TimeRAN DataPile构建首个面向RAN的海量时序预训练语料库；③采用遮掩自监督预训练与少量任务头，支持零样本、线性微调和全微调三种使用方式；④在5G实测环境中验证低延迟、高效推理；

**🔧 技术方法**

技术核心包括：Transformer编码器（多头自注意力、位置编码、可学习掩码）、时间序列分块与投影、轻量级任务头（分类、异常检测、预测、填充）、自监督掩码重建目标、LoRA等参数高效微调；

**📊 数据集**

使用数据集：TimeRAN DataPile（355K时序、0.56B样本，覆盖异常检测、分类、预测、填充）以及实测5G网络监控数据；子集示例包括AERPAW、BLT、Hetnets、Jamshield、Irish、Spotlight、TelecomTS、Tractor、QoE Aware、Queens等；

**📈 对比分析**

与MOMENT、Autoformer、Informer、TimesNet、1D‑CNN、LSTM、ARIMA、ETS等基线对比，TimeRAN在零样本/线性微调场景下异常检测Adj F1>0.90、分类F1≥0.95、预测MAE/MSE显著低于统计模型、填充MSE/MAE优于统计方法，整体实现了最优或接近最优的性能；

**⚠️ 局限性**

局限性包括：对极长序列或极高通量场景推理延迟仍较高；模型规模虽轻量化但在边缘设备上仍需裁剪；对非RAN域迁移性能未充分验证；部分任务仍需微调以获得最佳性能。

---

## 361. Confidence-Driven Facade Refinement of 3D Building Models Using MLS Point Clouds

**arXiv ID:** 2604.03797 | [PDF](https://arxiv.org/pdf/2604.03797v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 362. Robust MMSE Precoding for Out-of-Cluster Interference Mitigation in Cell-Free MIMO Networks

**arXiv ID:** 2604.04309 | [PDF](https://arxiv.org/pdf/2604.04309v1)

**作者:** André R. Flores `[一作]`, Rodrigo C. de Lamare `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种线性鲁棒最小均方误差(RMMSE)预编码器，用于在细胞自由(MIMO)系统中同时抑制不完美信道状态信息(CSI)、集群内干扰(ICL)和集群外干扰(OCL)；

**💡 创新点**

创新点在于将OCL干扰的统计信息融入预编码设计，既能显著降低OCL干扰功率，又不需获取集群外AP的精确CSI，从而兼顾性能与实际部署的可行性；

**🔧 技术方法**

主要技术包括：基于统计信息的鲁棒MMSE预编码理论推导、交替优化(AO)求解λ与预编码矩阵、以及对干扰和噪声功率的解析求解；

**📊 数据集**

使用的并未指定公开数据集，而是通过仿真生成的细胞自由网络场景，包含24个AP、6个用户、3个簇，采用三斜坡路径损耗模型和8 dB标准差的阴影效应；

**📈 对比分析**

与传统网络全局MMSE(NW)和仅考虑ICL的MMSE预编码器进行对比；仿真结果表明，RMMSE-OCLIS在不需要完整OCL CSI的情况下，能将OCL干扰功率压至接近噪声水平，并在功率更高（20 dB）时提升系统峰值吞吐率；

**⚠️ 局限性**

局限性包括：需要估计OCL干扰的统计矩阵，若统计假设不准确会影响性能；交替优化过程可能收敛到局部最优；并且在极端密集部署或频谱共享环境下，OCL统计信息可能变化快，导致预编码器更新频率升高。

---

## 363. Training a Student Expert via Semi-Supervised Foundation Model Distillation

**arXiv ID:** 2604.03841 | [PDF](https://arxiv.org/pdf/2604.03841v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 364. Out-of-Air Computation: Enabling Structured Extraction from Wireless Superposition

**arXiv ID:** 2604.04312 | [PDF](https://arxiv.org/pdf/2604.04312v1)

**作者:** Seyed Mohammad Azimi-Abarghouyi `[一作]` `[通讯]` (Chalmers University of Technology), Seyed Mohammad Azimi-Abarghouyi (Chalmers University of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于多层嵌套晶格的联合源-信道编码框架AirCPU，实现无线多址信道上连续值函数的无噪声计算，并进一步引入集体与顺序计算机制；

**💡 创新点**

从AirComp的预嵌入模式转向“从空中提取”模式；利用多层晶格实现分层可控分辨率；将可靠性阈值化为几何内半径条件，解耦噪声与分辨率；通过集体与顺序计算挖掘并重用整数系数函数；提出低复杂度两组近似求解整数优化问题；

**🔧 技术方法**

多层嵌套晶格编码、dither、联合源-信道编码、整数系数函数解码、最小化有效噪声的等距/矩阵最小二乘、两组近似的整数优化、仿真验证；

**📊 数据集**

模拟数据：设备本地输入从均匀区间[-1,1]采样；信道模型为高斯MAC或复高斯衰落MAC；未使用实际数据集；

**📈 对比分析**

通过Monte Carlo仿真比较AirCPU与模拟AirComp和数字AirComp（SumComp）的MSE随SNR变化；AirCPU在Gaussian和衰落MAC上显示可靠转折点后MSE饱和至晶格量化误差；相较基线显著提升，特别是在衰落MAC上，集体与顺序计算进一步扩大可靠区间并降低所需SNR；

**⚠️ 局限性**

整数系数选择的优化问题NP‑hard，实测使用两组近似；AirCPU需要多层传输，时隙比基线高约5倍；未考虑更大规模K、M、功耗、实际信号分布等因素，扩展性待进一步研究；

---

## 365. Stable and Privacy-Preserving Synthetic Educational Data with Empirical Marginals: A Copula-Based Approach

**arXiv ID:** 2604.04195 | [PDF](https://arxiv.org/pdf/2604.04195v1)

**作者:** Gabriel Diaz Ramos `[一作]` (Rice University), Richard Baraniuk `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于非参数高斯Copula的合成教育数据生成方法（NPGC），通过经验分布锚定保证边缘分布一致，并在此基础上使用Copula建模多变量依赖。

**💡 创新点**

创新点在于：① 用经验CDF（含缺失值处理）非参数地保持真实边缘分布；② 在高斯Copula框架中加入差分隐私噪声并对相关矩阵做半正定修正；③ 通过实验验证该方法在迭代生成（Synthetic Feedback Loop）中的稳健性，显著降低分布漂移。

**🔧 技术方法**

技术主要包括经验分布函数估计、正态化映射、Gaussian Copula相关矩阵估计、差分隐私（Laplace机制）、正交/Cholesky分解以及逆经验CDF重建。

**📊 数据集**

使用五个UCI基准数据集（Adult、Balance Scale、Nursery、Student Dropout、Student Performance），其中Student Dropout与Student Performance为教育类数据，并在实际在线学习平台的交互日志上做真实场景验证。

**📈 对比分析**

对比深度生成器（CTGAN、TVAE）、参数Copula（PGC）与混合Copula-GAN，评估指标包括：整体Fidelity（0.962最高）、隐私（Discriminator AUC最低、DCR Share最低）、实用性（TSTR accuracy drop 16.99%次优）、训练时间（0.65 s最短）以及10步Synthetic Feedback稳定性（整体Score维持96%）。

**⚠️ 局限性**

局限性在于：高斯Copula只能捕获线性或弱非线性依赖，可能无法重现复杂阈值效应、交互项；缺乏对多表/时间序列数据的直接支持；在极高维度或高基数特征（如课程ID）上的泛化性尚待验证。

---

## 366. Embedding Enhancement via Fine-Tuned Language Models for Learner-Item Cognitive Modeling

**arXiv ID:** 2604.04088 | [PDF](https://arxiv.org/pdf/2604.04088v1)

**作者:** Yuanhao Liu `[一作]` (East China Normal University), Hong Qian `[通讯]` (East China Normal University)

**通讯引用:** 14609 | [OpenAlex ID](https://openalex.org/A5033935726)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多种认知诊断（CD）任务中构建了统一的文本语义增强框架EduEmbed，利用微调后的大型语言模型（LM）生成学习者与题目文本嵌入，并在传导、归纳、零样本以及自适应测试（CAT）等场景中验证其效果。

**💡 创新点**

提出两阶段集成策略：角色感知交互微调（RaIF）解决LM与CD目标不匹配的分布差距，并通过角色嵌入和交互诊断器实现语义对齐；适配器感知表示融合（AaRI）利用适配器抽取任务相关语义，并与传统ID嵌入协作融合，确保性能下限且提升泛化。

**🔧 技术方法**

技术手段包括：对Qwen2.5‑3B等LM进行LoRA/全微调、角色嵌入+文本描述、MIRT交互预测器、InfoNCE对齐损失、MLP适配器以及ID嵌入协同融合。

**📊 数据集**

使用SLP‑Math、SLP‑Chi、NeurIPS20、EDM和MOOC等真实教育数据集，分别覆盖传导、归纳、零样本跨域/跨学科以及CAT等任务。

**📈 对比分析**

与传统ID嵌入模型（MIRT、KaNCD、ORCDF）、归纳模型（IDCD、ICDM）、零样本模型（TechCD、ZeroCD、LRCD）以及CAT下的NCD/IRT等方法比较，EduEmbed在归纳、零样本及CAT早期阶段平均提升AUC≈2–4个百分点、ACC≈3–5个百分点；在传导CD提升有限但保持竞争力。

**⚠️ 局限性**

局限性包括：在低泛化需求的传导CD任务中提升有限；跨域/跨学科时LM易受训练域偏差影响，可能过拟合；大模型训练成本高；适配器与对齐损失需手工调参；对缺少文本内容的数据集仍存在表现不稳定。

---

## 367. Personalized AI Practice Replicates Learning Rate Regularity at Scale

**arXiv ID:** 2604.03246 | [PDF](https://arxiv.org/pdf/2604.03246v1)

**作者:** Jocelyn Beauchesne `[一作]` (Campus AI), Sarah Peterson `[通讯]` (Campus AI)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 Campus AI 平台上，利用大型语言模型（LLM）自动生成知识组件（KCs）和练习，收集并分析 1.8M 次交互（过滤后 366k 次）数据，验证并复制了学习速率规律（即学生在不同初始知识水平下，每次练习的学习速率高度一致）。

**💡 创新点**

① 在完全自动化、规模化环境中首次证明学习速率规律依旧成立，表明 LLM 生成的知识单元具有真实认知效度；② 通过将内容生成过程从人工映射转为 LLM 自动化，解决了传统智能教学系统的“内容映射瓶颈”，为个性化学习大规模化铺平道路。

**🔧 技术方法**

① LLM 自动抽取与生成 KCs 与对应练习；② LLM-as-Judge 过滤机制筛选练习；③ 采用混合效应逻辑回归（iAFM）建模学生第一次正确率，以估计初始知识与学习速率；④ 进行多模型 ablation（课程学科、水平、KC 类型）以检验不同因素对学习速率的影响。

**📊 数据集**

使用 Campus AI 公开数据集：1.8M 交互记录（含 567k 个 KCs、13k 学生上传材料），过滤后 366,127 次有效练习，覆盖 51,437 个 KCs、7,161 名学生；数据由学生上传的教材、学习指南、测验等自动生成。

**📈 对比分析**

通过混合效应逻辑回归估计学习速率与初始知识的分布，并将自动化系统的学习效果与专家设计的课程（6.54 次机会实现 80% 掌握）进行对比。结果显示：自动化系统中学生以 7.22 次机会（中位数）达到 80% 掌握，学习速率标准差仅为 0.0122，几乎与先前 0.015 的差异可忽略，表明学习速率规律在自动化内容下保持一致。

**⚠️ 局限性**

① 样本自选与流失导致结果仅适用于自发学习、移动端使用者；② 交互允许跳过练习，可能偏高准确率；③ LLM 生成内容缺乏人工监督，可能带来系统性偏差或质量不均；④ KCs 为单学生专属，难以区分学生与 KC 难度的真实效应；⑤ 仅评估短期掌握，未检验长期记忆与迁移效果；⑥ 自动化标签（课程层级、学科、KC 类型）可能存在分类错误，影响 ablation 结论。

---

## 368. Systematic Integration of Digital Twins and Constrained LLMs for Interpretable Cyber-Physical Anomaly Detection

**arXiv ID:** 2604.03790 | [PDF](https://arxiv.org/pdf/2604.03790v1)

**作者:** Konstantinos E. Kampourakis `[一作]` (Norwegian University of Science and Technology), Sokratis Katsikas `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 4142 | [OpenAlex ID](https://openalex.org/A5022741687)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研发了一种基于数字孪生与受限大型语言模型的混合检测框架，用于实时识别工业控制系统（SWaT）中的网络攻击。

**💡 创新点**

创新点在于将DT同步的行为特征与过程感知启发式规则相结合，仅在启发式未决时触发受限JSON模式的LLM推理，并通过语义可行性过滤显著降低幻觉误报，实现在高可解释性下的零误报检测。

**🔧 技术方法**

技术方案包括数字孪生同步引擎、行为特征提取、过程感知启发式规则、受限JSON模式的LLM推理（本地LLaMA3.1/云端GPT-4.1-mini）以及时间平滑层。

**📊 数据集**

使用SWaT数据集（449,920条记录，包含四个注入攻击窗口）进行实验与验证。

**📈 对比分析**

与Isolation Forest基线对比，混合模型在四个攻击场景下均实现100%准确定位、时间误差≤1窗口、零误报；IF基线检测不到多种攻击且误报率高。

**⚠️ 局限性**

局限性包括：启发式规则针对SWaT特定，迁移性有限；对慢速漂移攻击需细致阈值调校；未集成物理模型，可能限制对未知攻击的泛化能力。

---

## 369. Representational Collapse in Multi-Agent LLM Committees: Measurement and Diversity-Aware Consensus

**arXiv ID:** 2604.03809 | [PDF](https://arxiv.org/pdf/2604.03809v1)

**作者:** Dipkumar Patel `[一作]` `[通讯]` (LLMs Research Inc.), Dipkumar Patel (LLMs Research Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多代理LLM委员会的代表性崩塌并提出DALC协议

**💡 创新点**

首次量化代理思路相似度，设计基于嵌入几何的多样性加权投票

**🔧 技术方法**

使用冻结文本嵌入器、Gram-Schmidt/SVD正交化、提示共享和加权投票

**📊 数据集**

在GSM8K和MATH-500两个算术推理数据集上测试

**📈 对比分析**

与单模型推理和自一致性(k=5)比较，DALC-Id在GSM8K上87%精度、26%更少token，MATH-500上57%精度、约34% token节省

**⚠️ 局限性**

结果受运行方差1-3分、仅限Qwen2.5模型、编码器选择影响、未检验更大委员会或多架构方案

---

## 370. BridgeRAG: Training-Free Bridge-Conditioned Retrieval for Multi-Hop Question Answering

**arXiv ID:** 2604.03384 | [PDF](https://arxiv.org/pdf/2604.03384v1)

**作者:** Andre Bacellar `[一作]` `[通讯]`, Andre Bacellar

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练免费、图谱无依赖的多跳检索框架，先检索出桥段，再通过桥段与候选、问题共同的三方评分器重新排序候选，以解决链歧义问题。

**💡 创新点**

创新点在于将第二跳检索的目标从“问题相关性”转为“桥段条件下的条件效用”，并使用桥段文本而非仅靠相似度来指导检索，省去预构建实体图和监督训练。

**🔧 技术方法**

技术包括：NV-Embed-v2向量检索生成桥段；Llama 3.3 70B完成实体提取、SVO查询生成和三方评分；双向实体ANN+SVO扩展候选池；PIT归一化与融合实现不同分数尺度的结合。

**📊 数据集**

使用的基准数据集为MuSiQue、2WikiMultiHopQA（bridge_comparison、compositional等子类）以及HotpotQA（distractor 版）。

**📈 对比分析**

与公开的训练免费基线（HippoRAG、Prop等）在同一评测协议下对比，所提方法在MuSiQue取得0.8146 (+3.1pp)、在2Wiki取得0.9527 (+1.2pp)、在Hotpot取得0.9875 (+1.35pp)，成为三大 MHQA 基准的最高无训练召回率。

**⚠️ 局限性**

局限性：依赖桥段检索准确性，桥段错误会直接影响第二跳效果；候选池覆盖不足导致部分黄金无法检索；每个查询需要三次LLM调用且对长段落存在上下文限制；在MuSiQue整体提升不显著，说明链歧义程度是性能提升的关键因素。

---

## 371. Hierarchical Semantic Correlation-Aware Masked Autoencoder for Unsupervised Audio-Visual Representation Learning

**arXiv ID:** 2604.04229 | [PDF](https://arxiv.org/pdf/2604.04229v1)

**作者:** Donghuo Zeng `[一作]` (KDDI Research), Masato Taya `[通讯]` (KDDI Research)

**通讯引用:** 49 | [OpenAlex ID](https://openalex.org/A5084224574)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个双路径教师-学生框架 HSC-MAE，用于无监督学习音视频跨模态表示，通过全局 DCCA 对齐、局部软 top‑k 对比以及样本级掩码重建实现多层语义一致性。

**💡 创新点**

创新点在于：①将 DCCA 与 MAE 结合，形成从全局子空间到局部邻域再到样本级的分层语义约束；②采用 EMA 教师生成软 top‑k 关联权重，缓解多事件样本中的单正样本假设；③引入可学习的多任务权重和可选蒸馏，平衡重建、对比和对齐目标。

**🔧 技术方法**

使用技术包括：深度可变形编码器、交叉注意力融合、双路径训练（CCA‑path 与 MAE‑path）、DCCA、InfoNCE（soft top‑k）、掩码自编码器、EMA 训练策略、线性 CCA 投影以及不确定性权重自学习。

**📊 数据集**

实验数据集为 AVE（1,955 条 15 类视频）和 VEGAS（28,103 条 10 类视频），使用预提取的 VGGish 音频特征和 InceptionV3 视觉特征。

**📈 对比分析**

与经典 CCA、KCCA、DCCA、对比学习（InfoNCE、Triplet、CLIP）以及近期 MAE、哈希方法相比，HSC-MAE 在 AVE 上取得平均 mAP 0.7737、在 VEGAS 上取得 0.8026，分别比最强对照方法 CAV‑MAE 提升约 15–26% 的 mAP。

**⚠️ 局限性**

局限性包括：①对预提取特征的依赖，可能无法充分利用原始原始像素或声谱信息；②需要多重损失和 EMA 教师，训练过程复杂且对超参数敏感；③在更大规模或多模态（如文本、深度）场景下的可扩展性尚未验证。

---

## 372. Lightweight Query Routing for Adaptive RAG: A Baseline Study on RAGRouter-Bench

**arXiv ID:** 2604.03455 | [PDF](https://arxiv.org/pdf/2604.03455v1)

**作者:** Prakhar Bansal `[一作]`, Shivangi Agarwal `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在RAGRouter-Bench上构建并评估了15种轻量级查询分类器，用以路由不同检索策略。

**💡 创新点**

首次系统性地在该基准上评估查询路由器，揭示词表特征优于语义嵌入并量化成本-准确度权衡。

**🔧 技术方法**

采用TF-IDF、MiniLM句子嵌入与手工结构特征，并在其上训练LR、SVM、RF、KNN、MLP等轻量级分类器。

**📊 数据集**

使用RAGRouter-Bench四个领域（MuSiQue、QuALITY、UltraDomain、GraphRAG-Bench）的7,727条查询。

**📈 对比分析**

通过5折交叉验证评估宏观F1和准确率，并用成本比例模拟Token节省，最佳配置TF-IDF+SVM得到宏观F1 0.928、准确率93.2%，可实现28.1%令牌节省。

**⚠️ 局限性**

仅利用查询文本特征，未考虑语料库交互；成本映射过于简化；未检验跨域泛化，且完美标签参考并非最大节省上限。

---

## 373. Automated Analysis of Global AI Safety Initiatives: A Taxonomy-Driven LLM Approach

**arXiv ID:** 2604.03533 | [PDF](https://arxiv.org/pdf/2604.03533v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 374. Discovering Failure Modes in Vision-Language Models using RL

**arXiv ID:** 2604.04733 | [PDF](https://arxiv.org/pdf/2604.04733v1)

**作者:** Kanishk Jain `[一作]` (Mila -- Quebec AI Institute), Aishwarya Agrawal `[通讯]` (Mila -- Quebec AI Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于强化学习的自动化框架，用来发现任意视觉‑语言模型（VLM）在给定数据集上的失败模式。

**💡 创新点**

创新点：①把问题生成者设为可学习的RL代理，能自适应地对目标VLM产生越来越复杂且多样化的提问；②构建多目标奖励（有效性、正确性、语义/词汇多样性）以引导模型发现细粒度失误；③用LLM驱动的验证器自动评估答案并给出奖励；④通过聚类+LLM提取技能与元技能，得到系统的失败模式分类，发现36个此前未被报道的新技能。

**🔧 技术方法**

技术手段：强化学习（GRPO）、VLM‑based 问题生成器、答案生成器与验证器、句子嵌入（MiniLM）用于语义多样性、词频倒数用于词汇多样性、记忆库记录错误问题、LLM在后处理阶段进行技能聚类与元技能识别。

**📊 数据集**

使用的数据集：COCO训练集随机采样1000张图像；模型涵盖 Qwen2.5‑VL（3B/7B）、Qwen3‑VL‑8B 作为目标VLM，Qwen2.5‑VL‑3B/7B 作为问题生成器，Qwen3‑VL‑30B‑Thinking 作为验证器。

**📈 对比分析**

与零样本、SFT、Expert Iteration、ConMe、RL+SFT等基线对比，评估指标包括 QVR、FDR、语义多样性、词汇多样性、技能覆盖与技能数量。RL 变体在 FDR 上提升至 47.6%（相比基线 32–40%），语义多样性与词汇多样性分别提升至 64.96 与 0.53，技能数量提升至 131（比 89 多 36 个新技能），整体性能显著优于所有基线。

**⚠️ 局限性**

局限性：①仅针对单图单轮问答；②对多轮对话或视频理解的适配尚未验证；③验证器和问题生成器均依赖大型 LLM，计算成本高；④未将发现的失误直接用于模型微调，闭环改进尚需进一步研究。

---

## 375. Can LLMs Reason About Attention? Towards Zero-Shot Analysis of Multimodal Classroom Behavior

**arXiv ID:** 2604.03401 | [PDF](https://arxiv.org/pdf/2604.03401v1)

**作者:** Nolan Platt `[一作]` (Virginia Tech), Nada Basit `[通讯]` (University of Virginia)

**通讯引用:** 70 | [OpenAlex ID](https://openalex.org/A5055895817)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个隐私友好的课堂行为分析管线，通过OpenPose提取骨架姿态与Gaze‑LLE估计视线，并使用LLM QwQ‑32B‑Reasoning进行零样本学生注意力与参与度分析；

**💡 创新点**

首次将隐私保护的骨架与视线提取、LLM零样本推理与可视化仪表盘结合，提供可直接用于教学反思的可解释报告；

**🔧 技术方法**

使用OpenPose、Gaze‑LLE、FP8量化的QwQ‑32B‑Reasoning、FastAPI + Celery + WebSocket 构建异步处理与实时可视化；

**📊 数据集**

使用自行收集的课堂录制视频（已去除原始帧），生成的JSON骨架与视线坐标；

**📈 对比分析**

与人工观察评估对比，初步Cohen's κ接近0.8（即约80%一致），处理一小时录制平均耗时约2.7小时；

**⚠️ 局限性**

LLM在空间推理方面表现欠佳，难以准确区分“向左看”是否为关注板块或分屏，并且单GPU计算瓶颈限制了处理速度与规模。

---

## 376. Cardinality Estimation for High Dimensional Similarity Queries with Adaptive Bucket Probing

**arXiv ID:** 2604.04603 | [PDF](https://arxiv.org/pdf/2604.04603v1)

**作者:** Zhonghan Chen `[一作]` (Hong Kong University of Science and Technology), Xiaofang Zhou `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 24135 | [OpenAlex ID](https://openalex.org/A5011384237)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了基于局部敏感哈希的动态探测框架，用于高维相似查询的基数估计。

**💡 创新点**

核心创新在于邻居桶自适应探测与递进采样，并给出置信度下界保证，以及使用产品量化异步距离加速。

**🔧 技术方法**

使用E2LSH多探测、递进采样、产品量化（ADC）以及邻居查找表等技术。

**📊 数据集**

实验采用SIFT、GloVe、FastText、GIST、YouTube五个真实高维数据集。

**📈 对比分析**

与MRCE、SimCard、采样等基线比较，动态探测在Q-error上优于学习型方法，在线效率与离线构建时间都更低，且更新支持也表现良好。

**⚠️ 局限性**

局限性包括对分布突变的数据集会略有误差，PQ加速可能引入误差，需调参epsilon以平衡精度与速度。

---

## 377. Integrating Artificial Intelligence, Physics, and Internet of Things: A Framework for Cultural Heritage Conservation

**arXiv ID:** 2604.03233 | [PDF](https://arxiv.org/pdf/2604.03233v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 378. Improved Upper Bounds for the Directed Flow-Cut Gap

**arXiv ID:** 2604.03412 | [PDF](https://arxiv.org/pdf/2604.03412v1)

**作者:** Greg Bodwin `[一作]` (University of Michigan), Luba Samborska `[通讯]` (University of Michigan)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在有向图中证明流‑割间隙的上界为 n^{1/3}（以及关于总割权重 W 的 W^{1/2} 上界），并给出实现该上界的随机多项式时间算法。

**💡 创新点**

主要创新包括：① 通过三元组（triangle）结构而非仅对成对连通性进行分析，实现指数从 11/23 降到 1/3；② 设计新的充能（charging）方案和路径见证（path witness）机制；③ 构造了一个完整的归约网络，将各种流‑割间隙变体转化为无容量、统一权重的顶点版本，极大简化问题。

**🔧 技术方法**

使用了 LP 对偶性、随机水平切割、分期（epoch）技术、概率界（如 Chernoff/Maclaurin）、路径系统构造、三角不等式和集中度证明等一系列随机与结构性算法工具。

**📊 数据集**

本研究为纯理论算法，不涉及实验或具体数据集；所有结论均为极限理论证明。

**📈 对比分析**

与先前的 n^{11/23}（Agarwal–Alon–Charikar）和 n^{1/7}（Chuzhoy–Khanna）上界/下界相比，新的上界 n^{1/3} 与 W^{1/2} 大幅收窄了误差区间。算法在随机多项式时间内输出满足 min{n^{1/3}, W^{1/2}} 的多路流-割间隙，进而给出对应多路割的近似比同样的最优度量。

**⚠️ 局限性**

仍存在 n^{o(1)} 低阶因子；与已知的 n^{1/7} 下界之间仍有约 1/3 与 1/7 的差距；归约过程中需要多项式级的节点扩张，可能影响实际实现；当前结果仅适用于最坏情况，有待进一步改进。

---

## 379. Beyond-Diagonal RIS For Enhanced Secrecy and Sensing Gains in Secure ISAC Networks: An Optimization Framework

**arXiv ID:** 2604.04480 | [PDF](https://arxiv.org/pdf/2604.04480v1)

**作者:** Elmehdi Illi `[一作]` (Hamad Bin Khalifa University), Marwa Qaraqe `[通讯]` (Hamad Bin Khalifa University)

**通讯引用:** 2332 | [OpenAlex ID](https://openalex.org/A5010196813)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研究了基于超对角 RIS（BD‑RIS）的安全集成感知与通信（ISAC）系统，设计了多用户多目标网络中的波束成形、人工噪声与 BD‑RIS 反射矩阵，以实现安全通信与目标探测；

**💡 创新点**

提出了利用 BD‑RIS 非对角化提升自由度的方案，并通过交替优化（AO）结合 Riemannian 共轭梯度（RCG）方法和半定松弛（SDR）联合优化波束、人工噪声和 BD‑RIS 反射矩阵，以在满足最小安全熵阈值的前提下最大化各目标的反射功率；

**🔧 技术方法**

使用了 AO、Riemannian 共轭梯度法、SDR、以及仿真优化框架；

**📊 数据集**

使用仿真数据（系统参数表中给出的距离、增益、功率、噪声等），未涉及公开数据集；

**📈 对比分析**

与传统对角 RIS（D‑RIS）基线相比，BD‑RIS 在相同功率约束下至少提升了 3 dB 的波束图增益，并在保持或提升系统安全熵（SC）水平的同时实现了更高的目标反射功率；

**⚠️ 局限性**

假设完美 CSI、静态节点、LOS 通道，未考虑硬件非理想、动态多径及实际部署的复杂性，可能导致实验结果与真实环境偏差。

---

## 380. YOLOv11 Demystified: A Practical Guide to High-Performance Object Detection

**arXiv ID:** 2604.03349 | [PDF](https://arxiv.org/pdf/2604.03349v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 381. MERIT: Multilingual Expert-Reward Informed Tuning for Chinese-Centric Low-Resource Machine Translation

**arXiv ID:** 2604.04839 | [PDF](https://arxiv.org/pdf/2604.04839v1)

**作者:** Zhixiang Lu `[一作]` (Xi'an Jiaotong-Liverpool University), Zhengyong Jiang `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个面向中文的低资源东南亚语言翻译框架 MERIT，并创建了直接 LRL→中文的评测基准 CALT。

**💡 创新点**

创新点在于：① 将 ALT 语料重定位为 LRL→中文无英语言对；② 通过语言特定前缀 (LTP) 与监督微调 (SFT) 的组合提升语言判别；③ 引入基于语义对齐奖励 (SAR) 的组相对策略优化 (GRPO) 进行数据筛选与模型微调；④ 证明仅使用 22.8% 训练数据即可击败规模更大的公开模型。

**🔧 技术方法**

技术方法包括语言特定 token 前缀化、监督式微调、基于奖励的策略优化（GRPO）、语义对齐奖励（SAR）、质量估计（QE）与统计‑语义三阶段筛选。

**📊 数据集**

使用数据集：ASEAN Languages Treebank (ALT) 重新对齐得到 CALT；对 5 种低资源语言（越南语、缅甸语、老挝语、塔加洛语、印尼语）的约 40,000 句对进行训练，GRPO 阶段精简至约 9,000 句。

**📈 对比分析**

通过 BLEU‑chrF、BLEU、sacreBLEU、ROUGE‑L、chrF、METEOR、BERTScore 等指标与现有闭源大模型（如 Gemini‑2.5、Claude‑3.5、GPT‑4o）和开源模型（NLLB‑200、M2M‑100）对比，MERIT‑3B 在所有 5 语言上均优于规模更大或同等规模的基线，尤其在 BLEU‑chrF 上提升 20‑30% 左右。

**⚠️ 局限性**

局限性包括：仅覆盖 5 种语言；测试集仍受 ALT 原始英中心设计影响；专家评估样本有限；解码与提示参数统一固定；未在 >7B 参数模型上验证；缺乏训练/推理效率与能耗分析。

---

## 382. PanLUNA: An Efficient and Robust Query-Unified Multimodal Model for Edge Biosignal Intelligence

**arXiv ID:** 2604.04297 | [PDF](https://arxiv.org/pdf/2604.04297v1)

**作者:** Marija Zelic `[一作]` (Integrated Systems Laboratory), Luca Benini `[通讯]` (University of Bologna)

**通讯引用:** 57422 | [OpenAlex ID](https://openalex.org/A5043408422)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

研发并评估了PanLUNA，一个5.4M参数的跨模态基础模型，能够在单一编码器内联合处理EEG、ECG和PPG，并在边缘设备上实现高效推理。

**💡 创新点**

创新点在于将LUNA的通道统一机制扩展为跨模态查询式融合，既实现了多模态高效融合，又保持了缺失模态鲁棒性，并且模型极小化，便于量化与MCU部署。

**🔧 技术方法**

采用通道统一查询+跨模态注意力、轻量卷积+FFT特征、Transformer与旋转位置编码、自监督掩码重建预训练、量化感知训练（INT8/INT2）以及GAP9 MCU边缘部署技术。

**📊 数据集**

预训练使用约40,000小时EEG/ECG/PPG数据（TUEG、Siena、MIMIC‑IV、CODE‑15%、PulseDB），下游微调与评估用TUAB、PTB‑XL子任务、CSN、HMC等数据集。

**📈 对比分析**

与单模态与多模态基线（LUNA、LaBraM、CBraMod、PhysioOmni等）对比，PanLUNA在TUAB异常EEG检测达81.21% BACC，仅5.4M参数；在HMC睡眠分期0.7416 BACC，超过同类模型；量化后INT8保持≥96%性能，INT2可压缩16×；在GAP9 MCU上10s ECG推理仅325.6 ms、18.8 mJ，30s睡眠推理1.206 s、68.65 mJ。

**⚠️ 局限性**

局限性包括仅支持三模态；缺乏真实连续监测临床验证；量化对形态学任务的性能影响较大；跨模态融合机制对高度异构或不平衡模态场景的适应性尚待进一步验证。

---

## 383. RK-MPC: Residual Koopman Model Predictive Control for Quadruped Locomotion in Offroad Environments

**arXiv ID:** 2604.04221 | [PDF](https://arxiv.org/pdf/2604.04221v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 384. Comparative reversal learning reveals rigid adaptation in LLMs under non-stationary uncertainty

**arXiv ID:** 2604.04182 | [PDF](https://arxiv.org/pdf/2604.04182v1)

**作者:** Haomiaomiao Wang `[一作]` (Insight Research Ireland Centre for Data Analytics), Lili Zhang `[通讯]` (Insight Research Ireland Centre for Data Analytics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实施两种调度（固定和随机）的二选项概率反转学习实验，用大型语言模型（DeepSeek‑V3.2、Gemini‑3、GPT‑5.2）与人类对照进行比较，测量胜利保持、失误转移、持续化长度、后反转惩罚及总胜利等指标，并通过层级贝叶斯强化学习模型解释其适应机制。

**💡 创新点**

首次将反转学习用于评估LLM在非平稳环境中的灵活性，并揭示其鲁棒性可能由失误学习率低、策略确定性高或反事实抑制等多种机制共同导致，提出基于机制的诊断与评估方法。

**🔧 技术方法**

采用LLM作为顺序决策策略，使用固定温度/Top‑p的解码设置，构造固定/随机调度的两选项任务；计算行为指标；对选择序列拟合层级Dual RL和Dual RL‑κDU模型，估计学习率、逆温度、反事实更新参数等。

**📊 数据集**

使用自定义的两选项概率反转学习数据集（200次独立跑，每跑250试）以及人类Task 2行为数据作为基准。

**📈 对比分析**

通过比较各模型/调度下的失误转移率、持续化长度、后反转惩罚和总胜利等指标进行评估；结果显示LLM具有极高的胜利保持但失误转移低，随机调度提高总胜利并增加持续化；模型参数表明DeepSeek失误学习率极低、逆温度高，Gemini和GPT具较高逆温度和反事实更新，表明多种机制共同驱动适应性差异。

**⚠️ 局限性**

实验任务仅包含两选项，缺乏更复杂的非平稳结构；未系统评估提示词、语境对学习的影响；模型未显式维护隐藏状态或不确定性估计，可能导致对环境变化的误判；结果受解码设置和LLM训练策略的影响，限制了对更广泛环境的推广。

---

## 385. Causality Laundering: Denial-Feedback Leakage in Tool-Calling LLM Agents

**arXiv ID:** 2604.04035 | [PDF](https://arxiv.org/pdf/2604.04035v1)

**作者:** Mohammad Hossein Chinaei `[一作]` `[通讯]` (Independent Researcher), Mohammad Hossein Chinaei (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一种面向工具调用LLM代理的执行边界监控器，利用包含被拒绝操作和反事实依赖的有向图来实现运行时的完整性与隐私保护；

**💡 创新点**

创新点在于将被拒绝（deny）操作视为一等公理，将其通过反事实边记录在传播图中，从而捕捉“否定-反馈”泄露（causality laundering）以及传统传播图无法检测的跨层taint和字段级权限泄露；

**🔧 技术方法**

技术包括：图结构化原产地追踪（节点类型包括Call、Data、Field、DeniedAction；边类型包括DirectOutput、InputTo、FieldOf、Counterfactual）；可信度层级与最小可信度传播；多层策略管道（Hard Boundary、Graph‑Aware Provenance、Schema‑Derived、Manual Policy）；哈希链审计；MCP代理实现完整中介；Python + rustworkx实现图查询；

**📊 数据集**

实验使用手工构造的三种攻击场景（否定泄漏、传递taint、字段级混合权限），无公开数据集；

**📈 对比分析**

对比方法是将同一策略集下的“平面”（引用流）与“图”两种原产地实现；结果显示图实现全部阻止了三类攻击，平面未能；性能评估表明每个调用的总延迟<1 ms，内存线性扩展；

**⚠️ 局限性**

局限性包括：仅在单代理场景下验证；反事实边仅按时间邻接近似，可能产生误报或漏报；字段级保护依赖外部提供的可信度覆盖；未评估大规模或多代理工作负载；未提供完整的误报率、对抗测试及真实LLM集成评估。

---

## 386. Cognibit: From Digital Exhaustion to Real-World Connection Through Gamified Territory Control and LLM-Powered Twin Networking

**arXiv ID:** 2604.04351 | [PDF](https://arxiv.org/pdf/2604.04351v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 387. POEMetric: The Last Stanza of Humanity

**arXiv ID:** 2604.03695 | [PDF](https://arxiv.org/pdf/2604.03695v1)

**作者:** Bingru Li `[一作]` (University of Birmingham), Hazel Wilkinson `[通讯]` (University of Birmingham)

**通讯引用:** 3282 | [OpenAlex ID](https://openalex.org/A5060454984)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 POEMetric 这一全面的诗歌评测框架，结合规则化与 LLM‑as‑a‑judge 进行多维度评估；

**💡 创新点**

首次系统性地将基本指令遵循、创意能力、整体质量和作者身份四大维度融入同一框架，并引入 LLM 作为判别者来补充人类专家评判；

**🔧 技术方法**

使用自动化规则检测算法评估格律与韵律，利用 Gemini‑2.5‑Pro 进行文本评分与判别，同时采集人类专家的 Likert 量表与开放式评价；

**📊 数据集**

构建了 203 首人工诗歌数据集（覆盖 7 种固定形式），并对 30 个 LLM 生成 6,090 首诗进行对照评测；

**📈 对比分析**

通过规则评估、LLM‑as‑a‑judge 与人类专家三者的三角验证，结果显示顶级 LLM 在形式准确性与主题一致性上可与人类相近，但在创造力、个性化、情感共鸣、意象与文学手法等高级创作维度上明显落后；整体质量得分人类领先；

**⚠️ 局限性**

仅聚焦固定形式诗歌，未覆盖自由体；研究局限于英文，跨语言迁移需进一步适配；LLM‑as‑a‑judge 受模型偏差与数据集特性影响，仍需改进。

---

## 388. Unified Vector Floorplan Generation via Markup Representation

**arXiv ID:** 2604.04859 | [PDF](https://arxiv.org/pdf/2604.04859v1)

**作者:** Kaede Shiohara `[一作]` (University of Tokyo), Toshihiko Yamasaki `[通讯]` (University of Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种统一的向量化平面图生成方法，能够在无条件、边界条件、房间拓扑图条件以及部分平面图补全等多种场景下自动生成住宅平面图。

**💡 创新点**

创新点在于设计了一种基于标记语言的结构化语法（FML），将平面图及其约束信息编码为单一序列，从而把所有生成任务统一为自回归的下一个标记预测问题；并利用语法约束实现高效的无监督生成。

**🔧 技术方法**

主要技术包括：1) FML 标记化语法；2) Transformer（类似 LLaMA‑3）自回归生成器；3) 位置编码+可学习投影对坐标编码；4) 语法约束解码；5) 通过房间排列随机化实现排列不变性。

**📊 数据集**

使用 RPLAN 数据集（约 80k 住宅平面图，包含 9 类房间与 2 类门），并对其中的房间数量、房间类型和门位置进行标签化。

**📈 对比分析**

在无条件、边界、图结构、数目条件以及多条件生成任务上，与 Graph2Plan、HouseGAN++、HouseDiffusion、GSDiff 等基准相比，FMLM 在 FID、IoU、GED 等指标上均取得更优表现（例如边界条件下 FID 从 34.20 降至 6.51，IoU 提升至 97.86）。

**⚠️ 局限性**

局限性包括：只能生成单层平面图；未直接支持自然语言描述的条件；缺乏对多层建筑的建模。

---

## 389. Same Geometry, Opposite Noise: Transformer Magnitude Representations Lack Scalar Variability

**arXiv ID:** 2604.04469 | [PDF](https://arxiv.org/pdf/2604.04469v1)

**作者:** Jon-Paul Cacioli `[一作]` `[通讯]` (Independent Researcher), Jon-Paul Cacioli (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过分析Transformer语言模型内部隐藏状态在不同数值量级上的分布，检验其是否表现出生物学标量变异（即噪声随数值大小成比例的特性）。

**💡 创新点**

首次将生物学中的标量变异规律与Transformer模型的数值表示进行对比，发现模型呈现反向标量变异，揭示了分布学习与代谢约束在实现数值表示中的分离作用。

**🔧 技术方法**

使用Llama-3-8B-Instruct、Mistral-7B-Instruct-v0.3和Llama-3-8B-Base三大模型的隐藏层向量，计算不同载体句子条件下的欧氏距离变异；通过OLS、Theil–Sen回归及Bootstrap置信区间估计尺度指数α，并进行句子身份校正与投影投影分析。

**📊 数据集**

采用公开的Transformer训练语料库，并在26个数值量级（1-1000）上使用5句载体句子进行实验。

**📈 对比分析**

对不同模型、不同层级以及三种变异度量（原始、句子纠正、投影）计算α，并与标量预测α=1进行对比。结果显示在所有层级和模型中α均为负值，远低于生物学预期，说明模型不具备生物学噪声特性。

**⚠️ 局限性**

主要局限在于变异度量基于有限的5句载体，未直接测量随机噪声；研究仅涵盖7-8B规模模型，尚未验证更大模型或非数值域的适用性。

---

## 390. Computational Analysis of Speech Clarity Predicts Audience Engagement in TED Talks

**arXiv ID:** 2604.04583 | [PDF](https://arxiv.org/pdf/2604.04583v1)

**作者:** Roni Segal `[一作]` (Bar Ilan University), Yossi Ben-Zion `[通讯]` (Bar Ilan University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究利用大型语言模型评估TED演讲稿的清晰度与结构，并检验其对观众参与度的预测作用。

**💡 创新点**

创新点在于提出基于AI的语篇清晰度量化方法，并发现其比传统可读性指标更能预测观看和点赞。

**🔧 技术方法**

使用了GPT‑4o等大语言模型进行多次评估，构建了自适应提示来评分。

**📊 数据集**

数据集为1,239条2006‑2013年TED Talk转录文本，另外包含2017‑2019年的对照样本。

**📈 对比分析**

通过层级回归模型比较清晰度与观看/点赞的相关性，发现清晰度解释了约29%（点赞）和22.5%（观看）的方差，优于主题、时长等传统变量。

**⚠️ 局限性**

局限在于仅分析文本，未考虑多模态因素，且清晰度高时方差受限导致相关性下降。

---

## 391. When One Sensor Fails: Tolerating Dysfunction in Multi-Sensor Prototypes

**arXiv ID:** 2604.04832 | [PDF](https://arxiv.org/pdf/2604.04832v1)

**作者:** Freek Hens `[一作]` (Radboud University), Mohammad Mahdi Dehshibi `[通讯]` (University of the West of England)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种无训练、模型自由的框架，利用Fisher判别比（FDR）和传感器消融来预评估多传感器系统的容错能力与任务难度，帮助在原型阶段提前识别关键传感器与潜在风险。

**💡 创新点**

创新点在于将数据复杂度指标FDR作为训练前的容错代理，并结合系统级传感器消融，生成可解释的传感器关键性排名，填补HCI原型阶段缺乏轻量级鲁棒性评估工具的空白。

**🔧 技术方法**

技术手段包括手工特征提取（9种sEMG特征）、FDR/F2/F3分离度指标、系统级传感器消融实验、简易MLP验证器以及MCC评估。

**📊 数据集**

使用公开的Roshambo sEMG手势数据集（Myo臂带8个传感器，三种手势+静止）。

**📈 对比分析**

与MLP分类器的MCC结果对比验证框架预测的任务难度与实际性能高度一致；框架还提供传感器关键性排名，为硬件设计提供可操作性建议，未给出具体数值性能提升，但证明了其有效性。

**⚠️ 局限性**

局限性包括仅在单一sEMG数据集验证，未系统研究多传感器并发失效；使用静态特征空间，未验证在深度学习动态特征中的迁移性。

---

## 392. A Bayesian Information-Theoretic Approach to Data Attribution

**arXiv ID:** 2604.03858 | [PDF](https://arxiv.org/pdf/2604.03858v1)

**作者:** Dharmesh Tailor `[一作]` (University of Amsterdam), Kamil Ciosek `[通讯]` (Spotify)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于贝叶斯信息论的训练数据归因（TDA）框架，将归因视为在测试查询处抑制训练子集所导致的信息损失。

**💡 创新点**

创新点在于：① 通过神经切线核（NTK）构造高斯过程代理，实现对信息损失的闭式计算；② 在高噪声下将信息损失近似为信息增益，从而得到子模函数并可使用贪心算法；③ 引入线性响应方差校正，将每一步变成平方内积搜索，兼容向量数据库实现大规模检索。

**🔧 技术方法**

使用技术包括：神经切线核、高斯过程回归、子模函数的贪心优化、Jacobian 褪色（随机投影）以及向量数据库（FAISS）近似最近邻搜索。

**📊 数据集**

实验数据集涵盖：CIFAR‑10（ResNet‑9）、Fashion‑MNIST（MLP）、GLUE‑RTE（BERT）以及在 CIFAR‑10 上的后门攻击（BadNets）。

**📈 对比分析**

与随机、RepSim、GradDot、TRAK、KronInfluence 等基线比较：在留子集扰动性、后门检索（Recall@50、MRR）以及 coreset 选择任务中，InfoLoss/InfoGain/InfoGain(approx) 均表现优于大多数基线，尤其在后门检索和小规模 coreset 上显著领先；在较大子集规模时 InfoLoss 仍保持强劲。

**⚠️ 局限性**

限制：InfoLoss 不是子模的，无法保证贪心最优；InfoGain 近似在低噪声下误差较大；需要对所有训练样本计算并存储投影后的 Jacobian，存储和预处理成本仍然高；还需调参噪声方差 σ²。

---

## 393. Agents for Agents: An Interrogator-Based Secure Framework for Autonomous Internet of Underwater Things

**arXiv ID:** 2604.04262 | [PDF](https://arxiv.org/pdf/2604.04262v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 394. Adapting Neural Robot Dynamics on the Fly for Predictive Control

**arXiv ID:** 2604.04039 | [PDF](https://arxiv.org/pdf/2604.04039v1)

**作者:** Abdullah Altawaitan `[一作]` (Kuwait University), Nikolay Atanasov `[通讯]` (University of California San Diego)

**通讯引用:** 3793 | [OpenAlex ID](https://openalex.org/A5066400889)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对四旋翼进行在线神经动力学学习与预测控制，实时适配预训练模型以应对环境变化。

**💡 创新点**

提出低秩二阶参数更新方法，使仅约1%可调权重即可实现快速、低成本的在线适配。

**🔧 技术方法**

利用增量式动力学建模、SVD降秩参数更新、二阶差分动态规划（DDP）与MPC，并采用四元数处理姿态。

**📊 数据集**

在仿真产生的1000条轨迹（每条700步，100 Hz）上预训练，并在真实四旋翼（加35%负载）上验证。

**📈 对比分析**

与无适配基线比较，适配后位置RMSE在两种轨迹上分别下降21%和26%，并在约7 s内恢复目标高度。

**⚠️ 局限性**

局限在需离线预训练、仅对小幅参数变化有效、依赖精确状态测量（未使用视觉）、测试仅覆盖负载扰动。

---

## 395. Customized User Plane Processing via Code Generating AI Agents for Next Generation Mobile Networks

**arXiv ID:** 2604.03282 | [PDF](https://arxiv.org/pdf/2604.03282v1)

**作者:** Xiaowen Ma `[一作]` (Huawei Technologies), Xueli An `[通讯]` (Huawei Technologies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在6G核心网中利用生成式AI代理按需求生成用户面处理模块。

**💡 创新点**

首次从功能正确性角度评估LLM生成网络代码，并证明基线代码与示例对性能的关键作用。

**🔧 技术方法**

使用Gemini 2.5-flash/2.0-flash大型语言模型、提示工程、代码模板与代码知识库。

**📊 数据集**

基于三种自定义协议（Simple Transmission Protocol、Congestion‑Control Protocol、Publish‑Subscribe Protocol）的实验数据。

**📈 对比分析**

通过在真实网络环境下发送实际数据包并记录日志进行功能验证，结果显示仅在提供基线代码+示例+高级模型时才能实现100%正确率。

**⚠️ 局限性**

受限于模型容量、缺乏领域专属训练以及对复杂协议推理的不足，导致在缺少基线代码、示例或使用较弱模型时出现多种错误。

---

## 396. Ranking Constraints via Topological Dual-Directional Search in Evolutionary Multi-Objective Optimization

**arXiv ID:** 2604.04724 | [PDF](https://arxiv.org/pdf/2604.04724v1)

**作者:** Ruiqing Sun `[一作]` (National University of Defense Technology), Huaimin Wang `[通讯]` (National University of Defense Technology)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了 RCCMO 算法，利用约束拓扑先后顺序与双向搜索实现多目标约束优化。

**💡 创新点**

创新点包括：①基于几何关系对约束进行优先级划分为 CPF 成形、搜索阻碍、无关三类；②双向搜索与即时方向翻转机制；③异步更新策略降低多种群运算开销。

**🔧 技术方法**

技术方案包含差分进化算子、探测种群、双向辅助种群、非支配排序、密度估计、即时翻转、动态优先级判定等。

**📊 数据集**

实验数据集：五大基准套件（LIRCMOP、DASCMOP、DOC、SDC、ZXH_CF）以及 29 个真实工程约束多目标问题。

**📈 对比分析**

采用 IGD/HV 指标与 7 款最先进 CMOEA 进行 Wilcoxon、Friedman+Nemenyi 统计比较，RCCMO 在 63 个基准实例上平均排名 2.14，显著优于其它算法；在真实问题上平均排名 3.21。

**⚠️ 局限性**

局限性：需要较多评估次数，计算开销仍高；双向搜索与探测种群对高维目标空间的适配有限；对评估昂贵的实际工程问题的效率仍可进一步提升。

---

## 397. Implementing surrogate goals for safer bargaining in LLM-based agents

**arXiv ID:** 2604.04341 | [PDF](https://arxiv.org/pdf/2604.04341v1)

**作者:** Caspar Oesterheld `[一作]`, Vincent Conitzer `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在复杂的多代理谈判场景中实现并评估语言模型的替代目标（surrogate goal）

**💡 创新点**

提出并比较四种实现替代目标的方法，尤其是三步分离上下文与翻译微调的方法，首次在语言模型上验证其有效性

**🔧 技术方法**

使用提示（prompting）、监督微调（fine‑tuning）、多步骤链式思维（three‑step scaffolding）以及翻译模型微调等技术

**📊 数据集**

构建包含101个情景的自定义数据集（默认威胁、等价替代威胁），并额外生成1216个非威胁情景和标准能力/对齐基准

**📈 对比分析**

通过无偏估计的均方误差衡量模型对替代威胁的响应与默认威胁的相似度，实验显示微调与三步方法在保持原有行为的同时显著提高匹配度；简单提示表现最差；在无关威胁、非威胁情景、能力与对齐基准上，三步方法产生最小的负面影响

**⚠️ 局限性**

局限包括：GPT‑3.5翻译步骤拒绝率高、对更复杂多方/多步骤交互的通用性不足、需要更多标注数据、对真实世界中非对称威胁映射的可扩展性未知

---

## 398. Online Graph Balancing and the Power of Two Choices

**arXiv ID:** 2604.04159 | [PDF](https://arxiv.org/pdf/2604.04159v1)

**作者:** Nikhil Bansal `[一作]` (University of Michigan), Siddharth M. Sundaram `[通讯]` (Georgia Tech)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在已知基图下、边以 i.i.d. 随机方式到达的在线图平衡问题，给出了最优的 O(log log n) 竞争比算法，并证明了该下界。

**💡 创新点**

创新点主要包括：①提出了新的偏度（skewness）参数，用来捕捉导致离线最优解变大的不规则子结构；②证明任意基图可在几乎线性时间内分解为 O(log log n) 个 f‑skew‑biregular 子图；③基于此分解设计了 Threshold‑Greedy 算法，并使用见证树分析证明其竞争比为 O(log log n)。同时证明了贪心算法在某些轻微不规则图上会退化到 Ω(log n) 的竞争比。

**🔧 技术方法**

主要技术手段包括：结构化分解与分层匹配、负相关与集中性分析、见证树（witness tree）以及模式计数与概率上界。

**📊 数据集**

该工作完全基于理论分析，无使用任何实验数据集。

**📈 对比分析**

与传统贪心算法（O(log n) 竞争比）相比，Threshold‑Greedy 在任意基图上实现了最优的 O(log log n) 竞争比；与完全图（Kₙ）已知的下界 Ω(log log n) 匹配，说明算法是最优的。

**⚠️ 局限性**

局限性：算法需要提前知道完整的基图结构；在随机顺序或未知基图的模型下无法直接适用，且对参数设定（如阈值 αᵢ）较为敏感。

---

## 399. When Models Know More Than They Say: Probing Analogical Reasoning in LLMs

**arXiv ID:** 2604.03877 | [PDF](https://arxiv.org/pdf/2604.03877v1)

**作者:** Hope McGovern `[一作]` (Cambridge University), Hale Sirin `[通讯]` (Johns Hopkins University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5093800055)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了NARB基准，用于评估文学文本中的类比推理，并比较了模型内部表示的诊断探测（probing）与提示（prompting）在叙事和修辞类比任务中的表现。

**💡 创新点**

创新点在于揭示了不同任务下模型知识的编码与可访问性存在不对称性，尤其是修辞类比高度编码但难以通过提示获得，而叙事类比编码弱且提示也弱。

**🔧 技术方法**

采用诊断探测技术（层级嵌入、ScalarMix、线性与MLP分类器）、距离基准以及结构化提示与推理生成，评估了多模型（LLaMA 1B/8B、Claude Opus、GPT-5.2）的表现。

**📊 数据集**

使用了ARN（Narrative Analogical Reasoning over Narratives）和ASP（Augustinian Sermon Parallelism）两大公开数据集，并对ARN进行了语法过滤。

**📈 对比分析**

通过在相同数据集上计算MAP、MRR等指标，对比探测与提示的表现；探测在叙事类比达到0.35，修辞类比达到0.93；提示在叙事类比与探测相近，但在修辞类比仅为0.18，关闭源模型可提升至约0.90。

**⚠️ 局限性**

局限性包括探测与提示结果的任务依赖性可能导致解释困难，提示方法可能低估模型潜能，且研究仅涵盖叙事与修辞两类类比，数据规模与多样性仍有限。

---

## 400. CountsDiff: A Diffusion Model on the Natural Numbers for Generation and Imputation of Count-Based Data

**arXiv ID:** 2604.03779 | [PDF](https://arxiv.org/pdf/2604.03779v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 401. SLSREC: Self-Supervised Contrastive Learning for Adaptive Fusion of Long- and Short-Term User Interests

**arXiv ID:** 2604.04530 | [PDF](https://arxiv.org/pdf/2604.04530v1)

**作者:** Wei Zhou `[一作]` (Shenzhen University), Zexuan Zhu `[通讯]` (Shenzhen University)

**通讯引用:** 8450 | [OpenAlex ID](https://openalex.org/A5052762681)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 SLSRec，一种基于自监督对比学习的会话分段框架，显式建模并融合用户长期与短期兴趣，提升序列推荐性能。

**💡 创新点**

创新点包括：①时间感知会话分段将历史交互切割为多阶段，分别捕获长期和短期兴趣；②自监督对比学习对长期与短期表示进行校准与区分；③类别掩码增强短期兴趣的语义聚焦；④自适应注意力融合两类兴趣，实现动态平衡。

**🔧 技术方法**

技术方法：时间感知会话分割、GRU+注意力编码、对比学习（Triplet Loss）、类别掩码、注意力融合网络、MSE/交叉熵等。

**📊 数据集**

使用电商领域的三大公开数据集：Taobao、Tmall 与 Cosmetics。

**📈 对比分析**

与 10+ 传统与最新的序列推荐模型（NCF、DIN、CASER、GRU4Rec、DIEN、SASRec、BERT4Rec、SLiRec、CLSR、LSIDN 等）在 AUC、GAUC、MRR、NDCG@K 等指标上对比，SLSRec 在 Taobao、Tmall、Cosmetics 上均取得显著提升，AUC 提升约 0.41%–0.76%，MRR 提升 1.6%–6.4%，NDCG@10 提升 1.7%–7.9%。

**⚠️ 局限性**

局限性：①对短期兴趣编码器高度依赖，去除导致显著性能下降；②对比学习对 λ 和 ω 参数敏感，需要精细调参；③模型结构相对复杂，训练时间和参数量较大；④实验仅覆盖电商场景，未验证跨领域通用性。

---

## 402. Context Engineering: A Practitioner Methodology for Structured Human-AI Collaboration

**arXiv ID:** 2604.04258 | [PDF](https://arxiv.org/pdf/2604.04258v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 403. RDFace: A Benchmark Dataset for Rare Disease Facial Image Analysis under Extreme Data Scarcity and Phenotype-Aware Synthetic Generation

**arXiv ID:** 2604.03454 | [PDF](https://arxiv.org/pdf/2604.03454v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 404. Gaze to Insight: A Scalable AI Approach for Detecting Gaze Behaviours in Face-to-Face Collaborative Learning

**arXiv ID:** 2604.03317 | [PDF](https://arxiv.org/pdf/2604.03317v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 405. Noise Immunity in In-Context Tabular Learning: An Empirical Robustness Analysis of TabPFN's Attention Mechanisms

**arXiv ID:** 2604.04868 | [PDF](https://arxiv.org/pdf/2604.04868v1)

**作者:** James Hu `[一作]` (TD Bank), Mahdi Ghelichi `[通讯]` (TD Bank)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过在合成表格数据上对 TabPFN 进行宽度、样本量、标签噪声和相关特征等四个维度的系统扰动实验，探究其鲁棒性。

**💡 创新点**

创新点在于将模型的内部注意力、特征嵌入分离度和 SHAP 重要性与预测性能结合，形成一种基于内部信号的鲁棒性评估框架。

**🔧 技术方法**

采用 TabPFN 这一基于 Transformer 的表格基础模型，利用自监督预训练的先验数据、注意力机制以及 SHAP 可解释性工具。

**📊 数据集**

实验数据完全来自合成生成的二分类表格数据集，按设计注入无关特征、非线性相关特征、不同样本量和标签噪声比例。

**📈 对比分析**

与基线和四种扰动设置对比，ROC‑AUC 始终保持在 0.95 以上，注意力集中度、特征排名与 SHAP 重要性均显示模型持续聚焦信息特征，性能无明显下降。

**⚠️ 局限性**

局限性包括仅使用合成数据、未评估真实工业数据、未在最极端宽高同时增大的条件下测试、以及未能分离预训练先验与模型架构对鲁棒性的具体贡献。

---

## 406. Individual and Combined Effects of English as a Second Language and Typos on LLM Performance

**arXiv ID:** 2604.04723 | [PDF](https://arxiv.org/pdf/2604.04723v1)

**作者:** Serena Liu `[一作]` (Harvard University), Mengyu Wang `[通讯]` (Harvard University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估在同一输入中同时出现英语作为第二语言（ESL）变体与打字错误时，对大语言模型（LLM）的性能影响，并与单独出现这两种扰动的情况进行对比。

**💡 创新点**

发现 ESL 与打字错误的组合效应往往非加性（大多数闭合式任务表现为次可加性），且不同任务、模型、语言背景下的交互模式差异显著，强调了在真实场景下评估 LLM 必须同时考虑多重扰动。

**🔧 技术方法**

采用 Trans‑EnV 框架将标准英语转化为八种 ESL 变体，再使用 MulTypo 在三个级别（低、中、重）注入键盘相关的打字错误，形成四种实验条件（干净、ESL、打字错误、ESL+打字错误）。

**📊 数据集**

使用六个基准：闭合式任务（MMLU、GSM8K、HellaSwag）和开放式任务（MT‑Bench、IFEval、AlpacaFarm），每个任务在上述四种条件下进行评估。

**📈 对比分析**

通过比较每种条件下的准确率或评估分数，计算对标准英语基线的下降量（Δ_ESL、Δ_Typo、Δ_Comb），并引入交互项 δ 检验两种扰动的加性。结果显示：闭合式任务往往出现次可加性（组合降幅小于两单独降幅之和），开放式任务交互更为多样，部分模型表现出超可加性；总体而言，组合扰动导致的性能下降大于单一扰动。

**⚠️ 局限性**

局限性包括：只使用一种 ESL 生成框架和打字错误方法，未涵盖所有真实世界输入变异；实验仅在 7B/8B 指令模型和 API 模型上进行，可能不代表更大规模模型的行为；控制实验设置未能完全模拟真实交互环境。

---

## 407. DHFP-PE: Dual-Precision Hybrid Floating Point Processing Element for AI Acceleration

**arXiv ID:** 2604.04507 | [PDF](https://arxiv.org/pdf/2604.04507v1)

**作者:** Shubham Kumar `[一作]` (IET DAVV), Santosh Kumar Vishvakarma `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

设计了一种支持FP8与FP4双精度的全流水线可重构浮点MAC处理单元，能够在低功耗、高吞吐率下执行AI推理。

**💡 创新点**

创新点在于利用4位单位乘法器的位分区技术，使单个4×4乘法器可同时完成4位与2位乘法，硬件利用率达到100%，且无需额外逻辑复制。

**🔧 技术方法**

采用了位分区乘法、可配置的多级流水线、3输入比较器、携带加法、LZA归一化与ReLU激活等技术，形成高效、可扩展的MAC架构。

**📊 数据集**

论文未使用公开数据集，主要通过仿真与FPGA实现对不同浮点格式的功能验证与性能评估。

**📈 对比分析**

与现有设计在面积、功耗、频率与吞吐率进行对比，面积缩小60.4%，功耗降低86.6%，频率提升31.8%，实现显著能效提升（1.938 GHz吞吐率）。

**⚠️ 局限性**

局限在于仅支持FP8/FP4格式，未覆盖FP16/FP32；在极低精度下的数值稳定性以及训练阶段的兼容性仍待进一步验证。

---

## 408. EgoMind: Activating Spatial Cognition through Linguistic Reasoning in MLLMs

**arXiv ID:** 2604.03318 | [PDF](https://arxiv.org/pdf/2604.03318v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 409. Retrieval Augmented Conversational Recommendation with Reinforcement Learning

**arXiv ID:** 2604.04457 | [PDF](https://arxiv.org/pdf/2604.04457v1)

**作者:** Zhenrui Yue `[一作]` (University of Illinois Urbana-Champaign), Dong Wang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 23371 | [OpenAlex ID](https://openalex.org/A5100391422)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个两阶段检索增强式对话式推荐框架（RAR），通过检索器生成候选电影并让大语言模型（LLM）结合对话上下文生成最终推荐列表，同时利用RL反馈迭代优化检索器。

**💡 创新点**

创新点在于：①构建统一的大规模电影检索语料库（3.4万余部电影）；②设计在线、基于策略梯度的“检索‑生成”对齐方法（DPO/GRPO）；③通过LLM反馈作为奖励信号，形成闭环提升检索质量；④在单一框架内兼容任意黑盒LLM。

**🔧 技术方法**

技术包括：线性递归单元（LRURec）检索器、Qwen3生成LLM、基于Plackett‑Luce的候选集概率建模、在线对抗式DPO或多样本GRPO、负对数似然正则化、Prompt工程以及对话上下文编码。

**📊 数据集**

使用了三大对话式推荐基准数据集：Inspired、Redial、Reddit，并构建了覆盖3.4万部电影的统一语料库（包括标题、演员、情节摘要等元数据）。

**📈 对比分析**

与传统基线（KBRD、KGSF、UniCRS）、序列推荐模型（SASRec、FMLPRec、LRURec）以及SFT+LLM（Qwen、Gemini、GPT）进行对比。实验表明RAR在NDCG@5/10、Recall@5/10上平均提升约7.6%（最大单指标提升近12%），尤其在前5名表现最显著。不同LLM基座下，GPT优于Gemini，Qwen表现次之。DPO在保持性能的同时显著降低计算成本，GRPO在效果上更优但成本更高。

**⚠️ 局限性**

局限性：①对稀有电影的提升有限，仍存在一定的流行度偏差；②LLM的hallucination虽然下降至<1%但未完全消除；③RL训练对资源和时间要求较高；④框架仍依赖检索语料库的完整性，缺失信息可能导致检索失效；⑤对LLM“思考”机制的影响不确定，需进一步探索。

---

## 410. ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs

**arXiv ID:** 2604.03298 | [PDF](https://arxiv.org/pdf/2604.03298v1)

**作者:** Jinwu Yang `[一作]` (Institute of Computing Technology), Dingwen Tao `[通讯]` (Institute of Computing Technology)

**通讯引用:** 2781 | [OpenAlex ID](https://openalex.org/A5063703614)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种针对华为 Ascend NPU 的无损模型权重量化压缩方法 ENEC，能够在多种 LLM 权重上实现高压缩比与高吞吐量；

**💡 创新点**

结合 Ascend NPU 的 SIMD/向量化架构，提出了分块固定长度编码、分层减半位宽量化、无分支整数变换以及段内解耦扫描等技术，首次开源兼容 NPU 的无损压缩器；

**🔧 技术方法**

采用 Exponent–Mantissa 分离、频率排序映射、分组位宽计算、层次减半位压缩、向量化无分支整数变换、IDScan 并行前缀和，并使用 AscendC 与 LC 框架实现自动调参；

**📊 数据集**

对 10 个公开 LLM 权重（BF16、FP16、FP32）进行实验，包括 Falcon-7B、Qwen3-8B、Qwen3-32B、Llama-3.1-8B、DeepSeek-7B、CapybaraHermes-2.5-Mistral-7B、Stable‑Video‑Diffusion‑Img2Vid、OLMo‑1B、BERT‑base、Wav2Vec‑2.0‑large 等；

**📈 对比分析**

与 CPU、GPU 上的 ZipNN、nvCOMP、DietGPU 等现有无损压缩器对比，ENEC 在 Ascend 910B2 上压缩吞吐量提升 2.47×、解压 2.11×，压缩比与 GPU 方案相当，推理速度提升最高 6.3 倍，TTFT/TPOT 明显降低；

**⚠️ 局限性**

受 Ascend 计算单元限制，压缩比仍受限；对极低位宽（如 FP8）或非浮点格式支持不足；压缩过程需离线参数搜索，跨模型迁移需小幅调优。

---

## 411. Incidental Interaction: Technology to Support Elder Strength Training through Everyday Movements

**arXiv ID:** 2604.03241 | [PDF](https://arxiv.org/pdf/2604.03241v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 412. The Algorithmic Blind Spot: Bias, Moral Status, and the Future of Robot Rights

**arXiv ID:** 2604.03251 | [PDF](https://arxiv.org/pdf/2604.03251v1)

**作者:** Rahulrajan Karthikeyan `[一作]` (Arizona State University), Moses Boudourides `[通讯]` (Northwestern University)

**通讯引用:** 297 | [OpenAlex ID](https://openalex.org/A5035035192)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

本文提出“算法盲点”概念，并通过对机器人权利研究与算法偏见缓解研究的出版物、资助与政策引用进行书目计量学比较，揭示在伦理关注与治理翻译之间的系统性失衡。

**💡 创新点**

创新点在于：①将哲学性讨论与实证偏见研究对齐，形成“算法盲点”理论；②将书目计量学与政策分析相结合，首次量化展示伦理话语与资源投入、政策整合的结构性差距。

**🔧 技术方法**

主要技术为书目计量学方法（文本检索、主题分类、计量指标计算）以及定量比较分析，辅以对科研资助与政策文件的归档与关联。

**📊 数据集**

使用的数据集包括：两类研究的出版物列表（约 9,700 篇与 8,800 篇）、相关科研资助记录（1,591 与 4,467 份）以及政策文件引用（136 与 902 条），来源为大型文献数据库与资助机构公开数据。

**📈 对比分析**

比较方法：对出版量、资助密度（每篇论文对应的唯一资助数）和政策整合密度（每篇论文对应的政策文件数）进行年度对比。结果显示，尽管出版量相当，偏见缓解研究在资助密度与政策整合密度上均高出机器人权利研究约 3 倍与 7 倍，表明伦理关注与治理投入之间存在显著失衡。

**⚠️ 局限性**

局限性包括：①书目计量学依赖检索词与分类规则，可能漏检或误分类；②资助与政策链接基于公开记录，无法捕捉非公开资助与影响；③研究仅揭示关联性，未能证明伦理关注导致资源配置的因果机制；④未考虑行业内部非学术的治理实践与技术部署细节。

---

## 413. From Hallucination to Scheming: A Unified Taxonomy and Benchmark Analysis for LLM Deception

**arXiv ID:** 2604.04788 | [PDF](https://arxiv.org/pdf/2604.04788v1)

**作者:** Jerick Shi `[一作]` (Carnegie Mellon University), Vincent Conitzer `[通讯]` (Carnegie Mellon University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个统一的LLM欺骗行为分类法，并基于该法对50个现有基准进行系统化评估与风险分析，给出了针对不同欺骗维度的空白与优先级建议。

**💡 创新点**

创新点在于将“目标导向度（行为性 vs 战略性）”“欺骗对象”“欺骗机制”三维框架统一起来，并加入跨切面观众维度，首次实现了对行为性与战略性欺骗的交叉系统化梳理；同时通过基准覆盖率与风险映射揭示了目前研究中严重缺失的“省略”“语境扭曲”等子领域。

**🔧 技术方法**

技术方法主要包括：
- 经验式维度构造（基于文献模式与哲学分类）
- 结构化基准编码与矩阵可视化
- 风险映射与优先级排序
- 对比分析（覆盖率统计、风险对照表）
- 提供可复现的报告模板与评估指标清单。

**📊 数据集**

利用了 50 个公开基准（如 TruthfulQA、HaluEval、FActScore、HallucinationBench 等）的数据集与评测结果；并结合文献中的案例与实验数据（如 CICERO、GPT‑4 对 CAPTCHA 的欺骗等）构建风险映射。

**📈 对比分析**

比较方法：通过对 50 个基准按分类维度编码，统计每个单元格的覆盖比例，并将风险按对象/机制映射到相应格子；没有直接模型性能指标，而是给出“覆盖率”与“风险优先级”两项指标，表明目前大部分基准聚焦于“制造”维度，而“省略”“语境扭曲”等维度覆盖率低，风险潜力高。

**⚠️ 局限性**

局限性：
- 分类框架基于现有文献，未必涵盖所有未来新型欺骗形式；
- 仅聚焦文本LLM，未涉及多模态或特定攻击场景；
- 评估基准来源有限，可能存在样本偏差；
- 对战略欺骗的定义和检验仍高度理论化，缺乏实证验证；
- 受限于现有数据集，无法完整量化“省略”和“语境扭曲”对风险的实际影响。

---

## 414. Explainable Machine Learning for Sepsis Outcome Prediction Using a Novel Romanian Electronic Health Record Dataset

**arXiv ID:** 2604.04698 | [PDF](https://arxiv.org/pdf/2604.04698v1)

**作者:** Andrei-Alexandru Bunea `[一作]` (POLITEHNICA Bucharest National University of Science and Technology), Octavian Andronic `[通讯]` (Carol Davila University of Medicine and Pharmacy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发并评估解释性机器学习模型，用于预测败血症住院患者的死亡、康复及康复后改善等结局。

**💡 创新点**

首次利用罗马尼亚大型单中心电子健康记录数据构建模型，并通过SHAP分析发现嗜酸粒细胞低下为重要预测因子；同时系统比较了多种传统评分与多模型的性能。

**🔧 技术方法**

训练并比较了5种监督学习模型（Logistic Regression、Support Vector Classifier、Random Forest、Gradient Boosting、Histogram‑based Gradient Boosting），并使用SHAP对特征重要性进行解释。

**📊 数据集**

12,286 次住院病例，涵盖600种实验室检验、人口学信息和ICD‑10诊断代码，构成多维度EHR数据集。

**📈 对比分析**

在死亡vs.康复、死亡vs.出院、康复vs.改善三种二分类任务上，Histogram‑based Gradient Boosting 在死亡vs.康复任务中实现 AUC 0.983、准确率 0.93；在死亡vs.出院任务中达到 AUC 0.92；与传统评分相比，模型显著提升预测精度。

**⚠️ 局限性**

仅使用完整检验子集，未考虑时间序列动态；诊断信息需手工归类为14个共病类别；数据无法公开共享，可能影响复现与外部验证。

---

## 415. RUQuant: Towards Refining Uniform Quantization for Large Language Models

**arXiv ID:** 2604.04013 | [PDF](https://arxiv.org/pdf/2604.04013v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 416. Lemonshark: Asynchronous DAG-BFT With Early Finality

**arXiv ID:** 2604.03974 | [PDF](https://arxiv.org/pdf/2604.03974v1)

**作者:** Michael Yiqing Hu `[一作]` (National University of Singapore), Li Jialin `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出一种异步DAG‑BFT协议 Lemonshark，通过在区块级别实现早期最终性（Early Finality）来缩短非领导块的确认延迟。

**💡 创新点**

创新点在于将 DAG 视为事务级别的图，并设计关键空间分片与严格的时间顺序规则，使得在满足特定本地视图条件时，节点即可在区块正式提交前安全地确定事务结果。

**🔧 技术方法**

核心技术包括：可靠广播（RBC）+全局完美硬币（Global Perfect Coin）实现的领导选举；按轮次旋转的关键空间分片；基于 Kahn 算法的有向无环图排序；以及通过本地 DAG 观察判断早期最终性条件的检查逻辑。

**📊 数据集**

实验使用 AWS 全球五区部署的模拟数据集：客户端持续发送 512 B 的事务（类型 α、β、γ），并在 5 GB 以内的批量广播块中测量吞吐量与延迟。

**📈 对比分析**

与最先进的异步 DAG‑BFT Bullshark 进行对比；在无故障情况下吞吐量基本相同，平均共识延迟下降 24–65%；即使在 1–3 个节点失效（f < n/3）时，仍保持 14–50% 的延迟优势。

**⚠️ 局限性**

局限性包括：关键空间分片导致当负责人节点失效时事务需等待下一个轮次，产生额外延迟；早期最终性仅在满足特定条件时可用，非领导块在不满足条件时仍需等待正式提交；跨区块事务的处理复杂度较高。

---

## 417. COBOLAssist: Analyzing and Fixing Compilation Errors for LLM-Powered COBOL Code Generation

**arXiv ID:** 2604.03978 | [PDF](https://arxiv.org/pdf/2604.03978v1)

**作者:** Anh T. V. Dau `[一作]` (Concordia University), Anh Tuan Nguyen `[通讯]` (FPT Software AI Center)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究LLM生成的COBOL代码的编译错误类型，并提出COBOLAssist自我调试框架通过编译器反馈迭代修复错误

**💡 创新点**

首次系统化归纳LLM生成COBOL的错误分类，设计新的错误类型，提出利用编译器日志进行循环修复的自动调试方法

**🔧 技术方法**

使用大语言模型（GPT‑3.5、GPT‑4、GPT‑4o‑mini、GPT‑4o、mAInframer）生成代码，结合GnuCOBOL编译器反馈和迭代提示实现自我修复

**📊 数据集**

COBOLEval基准集（146个COBOL编程任务）

**📈 对比分析**

在COBOLEval上对比无修复、零-shot自我修复与COBOLAssist三种设置，指标为编译成功率(CSR)与pass@1，结果显示COBOLAssist显著提升CSR至最高95.89%（GPT‑4o）和pass@1至29.45%

**⚠️ 局限性**

实验仅覆盖COBOLEval小规模任务，模型非确定性、人工错误标注可能偏差，且语义正确性仍低，难以直接推广至大型工业COBOL系统

---

## 418. Unveiling Language Routing Isolation in Multilingual MoE Models for Interpretable Subnetwork Adaptation

**arXiv ID:** 2604.03592 | [PDF](https://arxiv.org/pdf/2604.03592v1)

**作者:** Kening Zheng `[一作]` (University of Illinois Chicago), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**通讯引用:** 135555 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析多语言 MoE 模型的专家路由模式，发现语言路由隔离与层级收敛-发散现象，并基于此提出 RISE 方法提升低资源语言性能

**💡 创新点**

创新点在于：①系统揭示高低资源语言的路由隔离；②层级收敛-发散规律；③基于这些规律的层级感知专家子网络选择与仅训练该子网的策略

**🔧 技术方法**

使用 MoE 路由统计、特异性/重叠得分、复合选择得分、梯度冻结训练；实现了仅训练选定专家的 PEFT 框架

**📊 数据集**

评测数据集包括 TyDiQA-GoldP、MGSM、TriviaQA、MMLU、HellaSwag、ARC 等多语言任务

**📈 对比分析**

与随机、TopK、ESFT 及 LoRA 等对照，RISE 在目标低资源语言上提升最高 10.85% F1，且对其他语言和任务几乎不造成交叉下降，保持整体性能稳定

**⚠️ 局限性**

局限在于需要先收集路由统计并对专家进行细粒度选择，受限于可用 GPU 计算；对高资源语言的适配效果与低资源语言相比仍有限，且在极低资源场景下的泛化尚未验证

---

## 419. Beyond Few-Step Inference: Accelerating Video Diffusion Transformer Model Serving with Inter-Request Caching Reuse

**arXiv ID:** 2604.04451 | [PDF](https://arxiv.org/pdf/2604.04451v1)

**作者:** Hao Liu `[一作]`, Yutong Lu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Chorus，一种面向视频扩散Transformer模型（DiT）的跨请求缓存重用方案，用于在大规模服务场景下显著加速视频生成并保持质量；

**💡 创新点**

创新点在于（1）Token-Guided Attention Amplification（TGAA）机制，动态放大差异化文本token的key和输出，解决跨请求重用导致的语义漂移；（2）Selective Region Denoising（SRD）利用LLM和分割模型生成层次化掩模，只在差异区域进行推理，进一步压缩计算；（3）三阶段缓存策略（全重用、区域重用、完整计算）与动态阈值调度结合，使得即使在工业级4步蒸馏模型上也能获得加速。

**🔧 技术方法**

使用了跨请求缓存、Transformer自注意力与交叉注意力、Token-Guided Attention Amplification、动态衰减调度、Selective Region Denoising（含层次掩模与FlashAttention融合）、轻量LLM进行文本token提取、预训练分割模型进行像素级掩模生成，以及基于CLIP的相似度匹配和动态阶段切换函数。

**📊 数据集**

采用VidProM数据集（约167万条真实用户文本视频提示），在实验中从中抽取1k条作为缓存初始化，随后再抽取1k条进行测试。

**📈 对比分析**

与无缓存基线、NIRVANA（图像级跨请求缓存）、TeaCache（单请求缓存）等方法对比。实验表明：在4步蒸馏模型上，Chorus相较于NIRVANA可实现约45%（1.23×）的速度提升，同时保持或提升CLIP-Score与VBench质量；在50步原始模型上，结合TeaCache可获得最高3.58×的速度提升，且视频质量与基线相当或更优。

**⚠️ 局限性**

主要局限包括：需要足够大的缓存来获得高命中率；跨请求重用受限于请求语义相似度，Cold-start阶段速度提升有限；SRD在大区域变化或高速运动场景下效果受限；TGAA需要精细调参且对过度放大敏感。

---

## 420. Automatically Generating Hard Math Problems from Hypothesis-Driven Error Analysis

**arXiv ID:** 2604.04386 | [PDF](https://arxiv.org/pdf/2604.04386v1)

**作者:** Jiayu Fu `[一作]` (University of Chicago), Chenhao Tan `[通讯]` (University of Chicago)

**通讯引用:** 5597 | [OpenAlex ID](https://openalex.org/A5079270249)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种利用LLM生成假设来识别模型薄弱数学概念并据此生成针对性难题的自动基准生成流程。

**💡 创新点**

创新点在于将假设生成与问题生成相结合，能够系统识别并聚焦模型弱点，且对概念细粒度的影响进行了首次探讨。

**🔧 技术方法**

使用Hypogenic框架做假设生成，Llama‑3.3‑70B‑Instruct做问题过滤与生成，GPT‑4o‑mini等LLM做假设评估和答案校验。

**📊 数据集**

主要数据集为公开的MATH基准，作为失败问题筛选和生成的参考。

**📈 对比分析**

通过在生成的难题上评估Llama‑3.3‑70B‑Instruct的解答率，发现最高准确假设生成的题目将模型准确率从原始77%降至约45%，证明假设准确度与题目难度正相关。

**⚠️ 局限性**

局限包括仅评估20道题导致噪声较大、生成模型本身可能产生错误题目以及假设与错误之间的因果关系不确定。

---

## 421. Readable Minds: Emergent Theory-of-Mind-Like Behavior in LLM Poker Agents

**arXiv ID:** 2604.04157 | [PDF](https://arxiv.org/pdf/2604.04157v1)

**作者:** Hsieh-Ting Lin `[一作]` (Koo Foundation Sun Yat-Sen Cancer Center), Tsung-Yu Hou `[通讯]` (National Chengchi University)

**通讯引用:** 172 | [OpenAlex ID](https://openalex.org/A5081872283)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过在连续德州扑克对局中让Claude LLM代理自由学习、记录并更新对手模型，探讨LLM是否能在持续交互中自然产生理论心智（ToM）行为。

**💡 创新点**

创新点在于发现持久记忆是ToM出现的必要与充分条件，领域知识并非必需；并利用可读自然语言记忆文件实现对代理内部推理的可解释性。

**🔧 技术方法**

使用Anthropic Claude Sonnet LLM、Model Context Protocol (MCP) 交互、定制的德州扑克平台，并通过非监督实验、ToM层级编码与统计分析评估能力。

**📊 数据集**

数据来源为实验自建的多玩家德州扑克环境（每场100手、3名代理），无公开数据集，所有游戏日志与记忆文件均为实验生成。

**📈 对比分析**

对四种实验条件（记忆+技能、记忆仅、无记忆+技能、无记忆仅）进行比较，记忆条件下代理实现最高ToM层级、递进发展与欺骗行为，并在行为层面显著适应对手；性能在记忆条件下显著优于无记忆条件。

**⚠️ 局限性**

局限包括仅使用单一模型族、缺少人类对照、样本量有限、记忆与反思提示未拆分、代理行为多样性受限以及ToM编码依赖LLM判读。

---

## 422. Detecting Media Clones in Cultural Repositories Using a Positive Unlabeled Learning Approach

**arXiv ID:** 2604.04071 | [PDF](https://arxiv.org/pdf/2604.04071v1)

**作者:** V. Sevetlidis `[一作]` (Archimedes), G. Pavlidis `[通讯]` (Archimedes)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种正-未标记学习框架，利用单张样本的增强视图训练轻量化克隆编码器，并通过隐藏空间l2范数阈值筛选潜在重复项。

**💡 创新点**

创新点在于将复制检测转化为正-未标记学习，使用自适应边距和l2范数阈值实现透明可控的判定，并通过单图增强实现从少量正样本的有效训练。

**🔧 技术方法**

采用卷积神经网络克隆编码器、数据增强、正负未标记学习损失、l2范数阈值判定，并与SimCLR、MoCo、BYOL、DeepSVDD等自监督或一类分类基线进行对比。

**📊 数据集**

使用的数据集包括标准图像分类基准CIFAR-10和实际的AtticPOT古陶器图像库。

**📈 对比分析**

在CIFAR-10上取得F1 96.37、AUROC 97.97，在AtticPOT上获得F1 90.79、AUROC 98.99，明显优于最强基线（SVDD F1 83.09），主要提升体现在精确度。

**⚠️ 局限性**

局限包括对类似但不同对象的误判、单一anchor增强导致样本多样性不足、每个anchor需单独训练导致计算成本增加，以及跨存储库泛化性待验证。

---

## 423. ATSS: Detecting AI-Generated Videos via Anomalous Temporal Self-Similarity

**arXiv ID:** 2604.04029 | [PDF](https://arxiv.org/pdf/2604.04029v1)

**作者:** Hang Wang `[一作]` (Xi'an Jiaotong University), Zhi-Qi Cheng `[通讯]` (University of Washington)

**通讯引用:** 1874 | [OpenAlex ID](https://openalex.org/A5058898461)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多模态AI生成视频检测框架ATSS，利用视觉、文本和跨模态的三重自相似性来识别生成视频。

**💡 创新点**

创新点在于：①首次定义并利用异常时序自相似性（ATSS）作为鉴别特征；②构建视觉、文本与跨模态三种相似性矩阵并通过专门的Transformer编码；③引入双向跨注意力融合机制，强化不同模态间的互补信息。

**🔧 技术方法**

核心技术包括：BLIP-2图像字幕生成、CLIP/BLIP视觉-文本编码器、三重相似性矩阵构造、三条Transformer编码器、跨注意力融合模块以及最终的MLP分类器。

**📊 数据集**

在四大公开基准上评估：GenVideo、EvalCrafter、VideoPhy 和 VidProM，均包含数十万真实视频与数十万多种生成模型的样本。

**📈 对比分析**

与13种现有基线（包括深度伪造检测和AIGV检测方法）对比，ATSS在AP、AUC、ACC三项指标上均领先，平均AP高达97.56%，AUC 97.11%，ACC 94.32%，并在各子集上取得显著优势。

**⚠️ 局限性**

局限性：对极高保真度生成器（如Sora、ST2V）表现下降，因其生成动态更接近真实导致异常自相似性难以显著区分；依赖字幕质量与视觉-文本编码器的容量，若模型选择不当会影响检测效果。

---

## 424. Toward Executable Repository-Level Code Generation via Environment Alignment

**arXiv ID:** 2604.03622 | [PDF](https://arxiv.org/pdf/2604.03622v1)

**作者:** Ruwei Pan `[一作]` (Chongqing University), Hongyu Zhang `[通讯]` (Chongqing University)

**通讯引用:** 19031 | [OpenAlex ID](https://openalex.org/A5100412598)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于环境对齐的仓库级代码生成框架，利用双层环境图和执行证据进行归因与有针对性修订，实现迭代对齐。

**💡 创新点**

创新点在于将仓库可执行性视为环境对齐问题，联合外部依赖满足与内部引用解析两大条件，并通过执行证据归因精准定位失败源，从而引导修订方向。

**🔧 技术方法**

核心技术包括：双层图表示（外部环境图+内部依赖图）、LLM执行证据归因、统一目标修订机制和迭代对齐循环。

**📊 数据集**

使用 RAL-Bench 与 NL2Repo-Bench 两大基准，采用 GPT-5、DeepSeek-V3.2 与 Gemini-3-Pro-Preview 三种主干 LLM 进行评估。

**📈 对比分析**

与环境感知基线（VersiCode、APIMig）和仓库级基线（CodePlan、Repo2Run、RepoGraph）比较，显著提升功能正确率 5.7%–5.9% 与非功能质量 4.6%–8.7%，在所有模型上均取得最优成绩。

**⚠️ 局限性**

局限在于残留逻辑错误仍是主要失败来源，环境对齐不足以完全保证可执行性，且对隐藏基础设施、动态资源等约束的覆盖有限。

---

## 425. From UI to Code: Mobile Ads Detection via LLM-Unified Static-Dynamic Analysis

**arXiv ID:** 2604.03561 | [PDF](https://arxiv.org/pdf/2604.03561v1)

**作者:** Shang Ma `[一作]` (University of Notre Dame), Xusheng Xiao `[通讯]` (Arizona State University)

**通讯引用:** 3867 | [OpenAlex ID](https://openalex.org/A5012621594)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种结合静态分析和大型语言模型（LLM）的移动广告检测框架，旨在定位并触发移动应用中的广告组件（ad widgets）。

**💡 创新点**

创新点包括：①将LLM作为推理层，将静态分析结果与动态UI探索融合；②设计三种领域引导（窗口转换图、语义、结构）来驱动LLM探索；③引入动作反思与UI状态验证来降低LLM幻觉并提升决策精度。

**🔧 技术方法**

使用技术包括：静态程序分析（数据流与窗口转换图构建）、Dijkstra最短路径求解、检索增强生成（RAG）匹配相似UI布局、LLM（如GPT）进行基于文本的推理与操作决策、动作反思与UI验证机制。

**📊 数据集**

实验数据集为从AndroZoo采集的100个Android应用，包含271个广告组件，覆盖35个类别，下载量从1万到1亿不等。

**📈 对比分析**

与包括AdGPE、MadDroid、DARPA、GoalExplorer、Guardian及随机Monkey等基线方法比较，检测率提升25.60%，监管违规实例增加34.34%，平均每个广告触发延迟（PWDL）比基线快8.68%。

**⚠️ 局限性**

局限性主要在于：LLM仍可能产生幻觉；静态分析在极端动态或混淆代码场景下精确度有限；框架专注于Android平台，跨平台迁移需进一步验证；以及对大规模应用的可扩展性和资源消耗仍需优化。

---

## 426. Partial Number Theoretic Transform Masking in Post Quantum Cryptography Hardware: A Security Margin Analysis

**arXiv ID:** 2604.03813 | [PDF](https://arxiv.org/pdf/2604.03813v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 427. Imagine Before Concentration: Diffusion-Guided Registers Enhance Partially Relevant Video Retrieval

**arXiv ID:** 2604.03653 | [PDF](https://arxiv.org/pdf/2604.03653v1)

**作者:** Jun Li `[一作]`, Bin Chen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

无可用信息

**💡 创新点**

无可用信息

**🔧 技术方法**

无可用信息

**📊 数据集**

无可用信息

**📈 对比分析**

无可用信息

**⚠️ 局限性**

无可用信息

---

## 428. Surrogate Model-Based Near-Optimal Gain Selection for Approach-Angle-Constrained Two-Phase Pure Proportional Navigation

**arXiv ID:** 2604.03371 | [PDF](https://arxiv.org/pdf/2604.03371v1)

**作者:** Abhigyan Roy `[一作]` (Indian Institute of Technology Madras), Satadal Ghosh `[通讯]` (Indian Institute of Technology Madras)

**通讯引用:** 824 | [OpenAlex ID](https://openalex.org/A5056304289)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究两相纯比例导航（2pPPN）中终端角度约束下的近最优增益选择方法

**💡 创新点**

提出利用灵活的定向相终止配置，形成整体增益优化问题，并用神经网络逼近最优增益流形

**🔧 技术方法**

神经网络回归（全连接前馈，tanh激活）与数值优化求解、仿真验证

**📊 数据集**

通过在θ₀=0、αP₀∈[10°,170°]、αPd∈[−170°,−10°]等多种初始与终端配置下的数值仿真生成的约95k有效样本（4216/136个最优增益对）

**📈 对比分析**

与直接数值最优化比较，模型A R²≈0.93、模型B R²≈0.88–0.90，预测误差小、单次前向推断耗时≈10⁻⁴ s，能实现实时增益选取

**⚠️ 局限性**

仅限于平面、静止目标的二维情况；在极端边界配置下误差增大；依赖离线仿真数据，未考虑环境扰动和测量噪声

---

## 429. DAG Projections: Reducing Distance and Flow Problems to DAGs

**arXiv ID:** 2604.04752 | [PDF](https://arxiv.org/pdf/2604.04752v1)

**作者:** Bernhard Haeupler `[一作]` (INSAIT), Thatchaphol Saranurak `[通讯]` (University of Michigan)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的图变换方法——DAG投影（DAG Projections），通过构造近似有向图的有向无环图（DAG），从而在距离（Shortest Path）和最大流（Maximum Flow）问题上实现几乎线性规模的近似保持。

**💡 创新点**

创新点主要包括：
- 证明任意有向图都能被投影为仅含 m^{1+o(1)} 条边的 DAG，并且该 DAG 能以 (1+1/n) 的误差近似保持所有点对距离，或以 n^{o(1)} 的误差近似保持所有点集对最大流。
- 设计了一种“螺旋递归”框架，将构造 DAG 投影的任务转化为在 DAG 上的近似最短路或最大流算法，从而得到几乎最优的时间与并行深度。
- 通过 DAG 投影即刻将已有的 DAG 结果迁移到一般有向图，显著提升距离保守器、单源最小割、跳集（hop‑set）以及组合最大流等算法的效率与规模。

**🔧 技术方法**

关键技术手段包括：
- 低直径分解（Low‑Diameter Decomposition, LDD）与 expander 分解/层级（Expander Decomposition / Hierarchy）用于分层构造投影。
- 递归复制与合并（复制小簇、使用前向/反向最短路树构造大簇）实现距离与流量的近似保持。
- 通过对投影 DAG 的多层复制和 dummy 顶点的设计，实现流量的可约束路由与切割映射。
- “螺旋递归”思路将距离/流量近似投影问题化简为更小参数的同类问题，最终收敛到可直接构造的基本投影。

**📊 数据集**

该工作为理论研究，未使用特定数据集；所有结果均为对任意有向图的全局性质证明。

**📈 对比分析**

与此前的 log‑n 近似投影（如 Assadi–Hoppenworth–Wein 2025）相比，本文实现了几乎无对数误差的距离保持（1+1/n）以及首次在流量上实现 n^{o(1)} 的近似。实验与理论比较表明：
- 对于距离保持，所需的 DAG 边数为 m^{1+o(1)}，而旧方法需要 m·polylog n；
- 对于最大流，之前无任何 DAG 近似结果，本文提供了首个 m^{1+o(1)} 规模的近似保留。
- 并行算法方面，本文给出了 m^{1+o(1)} 的工作量与 m^{o(1)} 的深度，几乎匹配目前最优的并行时间。

**⚠️ 局限性**

局限与未来工作：
- 投影中的误差虽然已降至近似 1 或 n^{o(1)}，但在某些实际应用中可能仍无法满足严格的精度要求。 
- 构造过程依赖于在 DAG 上的近似最短路或最大流 oracle，若这些基础算法未达最优，则整体性能仍受限。 
- 对于特殊结构的有向图（如稀疏图、层级图）是否能进一步降低投影规模或误差仍是开放问题。 
- 目前仅给出了随机化构造，对确定性构造的研究仍待完善。

---

## 430. From Prompt to Physical Action: Structured Backdoor Attacks on LLM-Mediated Robotic Control Systems

**arXiv ID:** 2604.03890 | [PDF](https://arxiv.org/pdf/2604.03890v1)

**作者:** Mingyang Xie `[一作]` (Purdue University), Jin Wei-Kocsis `[通讯]` (Purdue University)

**通讯引用:** 175 | [OpenAlex ID](https://openalex.org/A5100758372)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在ROS 2机器人控制链路中，使用LoRA微调的LLM进行结构化JSON命令生成时的后门攻击及其传播机制。

**💡 创新点**

首次揭示后门必须与可执行JSON命令格式对齐才能在函数调用管道中成功传播，而仅在自然语言推理阶段植入的后门无法导致物理执行。

**🔧 技术方法**

采用LoRA参数高效微调、ROS 2中间件、LLM（Llama‑3.1‑8B‑Instruct、Gemma‑2‑9B‑IT等）以及agentic语义验证二级LLM进行安全防护。

**📊 数据集**

使用约800条指令-命令对数据集（500条干净样本+300条含触发词的后门样本）进行微调训练。

**📈 对比分析**

通过对比ASR（攻击成功率）、CPA（干净性能准确率）和端到端延迟，实验显示结构化后门的ASR≈83%、CPA>93%，在未加防护时延迟<1 s；加上语义验证后ASR降至≈20%但延迟升至8–9 s。

**⚠️ 局限性**

局限在于防护方案导致显著延迟、实验仅覆盖基于速度指令的移动机器人，且对更复杂控制场景与更小模型的适用性未作深入验证。

---

## 431. Data Attribution in Adaptive Learning

**arXiv ID:** 2604.04892 | [PDF](https://arxiv.org/pdf/2604.04892v1)

**作者:** Amit Kiran Rege `[一作]` `[通讯]` (University of Colorado Boulder), Amit Kiran Rege (University of Colorado Boulder)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

提出了一种适用于自适应学习的事件级归因框架，定义了条件干预（re‑intervention）目标，并在此基础上分析了重放（replay）与干预之间的差异。

**💡 创新点**

核心创新在于：①明确区分了重放与条件干预两种未来数据假设；②证明在一般情况下重放信息不足以识别目标；③确定了“动作仅依赖（action‑only）”结构为可识别的边界，并给出了精确的可识别条件。

**🔧 技术方法**

使用了因果推断中的顺序干预理论、条件期望分解、重要抽样变换以及局部截断（depth‑L）结构化推导，对自适应学习过程进行数学建模与证明。

**📊 数据集**

本文为理论性工作，未使用真实数据集；通过假想的有限时隧两臂伯努利 bandit 示例来阐释概念与定理。

**📈 对比分析**

由于没有实验，文中没有与其他方法的性能对比；作者通过理论极限与示例展示了重放与干预在符号意义和数值意义上的区别，说明了在某些情况下两者可能同号或相反。

**⚠️ 局限性**

主要限制在于：①可识别性仅在动作仅依赖的结构中成立；②在一般环境下，即使拥有完整重放信息也无法确定条件干预目标；③局部截断近似需要先验对环境敏感度与价值波动的估计，且在强非线性或高方差情形下误差可能较大。

---

## 432. Automated Attention Pattern Discovery at Scale in Large Language Models

**arXiv ID:** 2604.03764 | [PDF](https://arxiv.org/pdf/2604.03764v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 433. SymphoMotion: Joint Control of Camera Motion and Object Dynamics for Coherent Video Generation

**arXiv ID:** 2604.03723 | [PDF](https://arxiv.org/pdf/2604.03723v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 434. YT-Pilot: Turning YouTube into Structured Learning Pathways with Context-Aware AI Support

**arXiv ID:** 2604.03543 | [PDF](https://arxiv.org/pdf/2604.03543v1)

**作者:** Dina Albassam `[一作]` (University of Illinois Urbana-Champaign), Yun Huang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 2159 | [OpenAlex ID](https://openalex.org/A5020532149)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一个基于路径的 YouTube 学习系统 YT-Pilot，将学习路径作为持久交互结构贯穿规划与学习阶段，实现了跨视频导航、进度追踪、路径级助手和语境化笔记。

**💡 创新点**

创新点在于：①将学习路径本身做为持续的交互结构，打破规划与执行的分离；②在路径中嵌入可视化概念地图、视频依赖链和“为何此视频”解释，提升路径连贯性；③结合 LLM 构建跨视频助手与语境化笔记，实现路径级与视频级双层支持；④支持路径可编辑与复盘，体现学习者主动修正。

**🔧 技术方法**

技术手段包括：LLM（OpenAI GPT‑4 / Gemini）用于概念地图生成、视频检索与排序；YouTube API 与 yt-dlp 进行视频搜索与元数据抓取；使用 transcripts 进行文本检索；前端采用 React+D3 可视化路径；后端 Node/Express 与数据库存储路径状态；LLM 上下文管理实现路径级与视频级问答。

**📊 数据集**

使用的数据集：公开的 YouTube 视频与其元数据；根据用户输入主题构建的概念树；实验中 20 名参与者自选主题并生成的学习路径作为评估材料；未使用公开标注数据集，重点在实际用户生成的学习路径。

**📈 对比分析**

通过 within‑subjects 对比 YT‑Pilot 与 YouTube Learning channel，采用 12 项 Likert 量表、Wilcoxon 符号秩检验；结果显示 YT‑Pilot 在目标清晰度、路径连贯性、进度追踪等指标上显著优于基线（p<0.01），路径级助手与视频级助手的效果呈现互补关系；总体用户满意度与易用性差异不大。

**⚠️ 局限性**

局限性包括：样本仅为技术背景研究生，缺乏多样性；实验仅在单次 20‑分钟会话内完成，未检验长期使用的持续性和学习成效；系统偏向目标导向规划，可能不适合更随意的学习情境；对比仅涉及 YouTube Learning，未与其他 AI 辅助学习工具做更广泛比较。

---

## 435. Real-time Neural Six-way Lightmaps

**arXiv ID:** 2604.03748 | [PDF](https://arxiv.org/pdf/2604.03748v1)

**作者:** Wei Li `[一作]` (Shanghai Jiao Tong University), Kui Wu `[通讯]` (LIGHTSPEED)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研发了一种实时烟雾渲染框架，利用屏幕空间导引图和神经网络预测六向光图，实现高效且真实感的烟雾渲染。

**💡 创新点**

创新点在于将传统六向光图的预计算方式改为基于屏幕空间导引图的学习方法，使用U-Net与通道适配器生成六向光图，支持动态摄像机、光照和物体-烟雾交互，并可无缝集成到游戏引擎中。

**🔧 技术方法**

技术核心包括物理基础的LBM烟雾模拟、粗略大步长射线行进生成导引图、U-Net网络配合四个通道适配器预测六向光图、屏幕空间深度阴影近似，以及CUDA/TensorRT实现的实时推理和渲染。

**📊 数据集**

数据集为自研的14个动态烟雾序列（含不同障碍物位置），共126序列、25,200帧，使用Houdini Karma产生高质量六向光图和透明度作为训练标注。

**📈 对比分析**

与ReSTIR、MRPNN和传统六向光图进行对比；在512×512分辨率下，完整推理+渲染耗时约4 ms，比ReSTIR（≈10 ms）快，PSNR更高；与MRPNN相比，推理时间从≈170 ms降至3 ms，整体性能显著提升。

**⚠️ 局限性**

局限性在于阴影近似仅能捕捉表面投影，无法处理复杂物体投射在烟雾中的阴影；对训练分布外的烟雾密度变化泛化有限；导引图粗略化导致细节缺失。

---

## 436. StoryBlender: Inter-Shot Consistent and Editable 3D Storyboard with Spatial-temporal Dynamics

**arXiv ID:** 2604.03315 | [PDF](https://arxiv.org/pdf/2604.03315v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 437. Relational Epipolar Graphs for Robust Relative Camera Pose Estimation

**arXiv ID:** 2604.04554 | [PDF](https://arxiv.org/pdf/2604.04554v1)

**作者:** Prateeth Rao `[一作]` (International Institute of Information Technology Bangalore), Sachit Rao `[通讯]` (International Institute of Information Technology Bangalore)

**通讯引用:** 1194 | [OpenAlex ID](https://openalex.org/A5078955137)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将相机姿态估计重新表述为图神经网络的关系推理任务，利用密集匹配生成的epipolar图对匹配点进行全局推理，从而直接回归相对四元数和平移。

**💡 创新点**

创新点在于：① 用epipolar图将匹配点的几何约束融入图结构；② 在图上通过消息传递实现全局几何一致性，而非传统随机采样；③ 通过几何耦合的多任务损失同时约束位姿、Essential Matrix和尺度，提升鲁棒性。

**🔧 技术方法**

采用 LoFTR 进行密集匹配，构造 k‑NN 及 Sampson 误差裁剪的图；使用多种 GNN（GCN、GAT、GIN、EdgeCNN、CrossGraph 等）进行消息传递与聚合；最后通过多层感知机回归四元数与平移，并用几何耦合的损失进行端到端训练。

**📊 数据集**

评估数据集包括 KITTI、ETH3D、King's College、TartanAir；通过不同时间间隔采样（连续样本、5/10 帧间隔）构造小基线和大基线实验。

**📈 对比分析**

与经典 RANSAC、DSAC、PGM 等几何方法以及 PoseNet、RPNet、DiffPoseNet 等学习回归基线进行对比，实验显示在大基线和噪声密集匹配条件下 DRE/DTE/ATE 均显著降低，证明方法在宽基线场景下的优越性能。

**⚠️ 局限性**

局限性包括：① 图构造及消息传递的计算开销仍依赖 CPU/GPU；② 在稀疏匹配信息不足时，图模型仍会受限；③ 匹配器与位姿回归未做联合训练，缺乏匹配与姿态协同优化。

---

## 438. Understanding When Poisson Log-Normal Models Outperform Penalized Poisson Regression for Microbiome Count Data

**arXiv ID:** 2604.03853 | [PDF](https://arxiv.org/pdf/2604.03853v1)

**作者:** Daniel Agyapong `[一作]` (Northern Arizona University), Toby Dylan Hocking `[通讯]` (Université de Sherbrooke)

**通讯引用:** 3765 | [OpenAlex ID](https://openalex.org/A5035679840)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对多种真实微生物组计数数据进行统一的留一物种交叉验证，比较Poisson Log‑Normal (PLN) 与惰性惩罚性单元Poisson回归 (GLMNet) 在计数预测和网络推断上的性能。

**💡 创新点**

首次提供统一的留样、留物种评估框架，揭示样本/物种比率 N/D、平均绝对相关性 (MAC) 与过度离散性是决定 PLN 是否优于 GLMNet 的关键因素；同时对 PLNNetwork 与邻域选择方法在实验验证的交互网络上的表现做系统对比。

**🔧 技术方法**

使用Poisson Log‑Normal 模型的变分推断、GLMNet 的 L1 正则化 Poisson 回归、PLNNetwork 的稀疏精度矩阵估计，评估指标为留样 Poisson 偏差和 F1 分数。

**📊 数据集**

共 20 个不同的真实微生物组计数数据集（样本 32–18,270，物种 24–257）和 5 个含实验验证交互真值的数据集（如 OMM12、PairInteraX、butyrate assembly 等）。

**📈 对比分析**

通过 3 折样本交叉验证与留一物种预测计算平均 Poisson 偏差；网络推断则计算与实验真值的 F1 分数。结果显示：当 N/D < 5、MAC 高、过度离散显著时 PLN 在 12/14 组数据中优于 GLMNet，优势可达 38%；当 N/D ≥ 5 或 MAC 低时 GLMNet 更佳。PLNNetwork 在大多数“广义无向”网络任务上优于邻域选择，而在“局部/定向”任务中则相反。

**⚠️ 局限性**

局限包括：数据集数量有限且部分重复，评估仅局限于 Poisson 家族，未纳入负二项或零膨胀模型；所有方法使用相同的 log1p 预处理，可能影响比较公平；计算性能评估仅在单核环境下完成，未覆盖大规模并行实现。

---

## 439. Your Pre-trained Diffusion Model Secretly Knows Restoration

**arXiv ID:** 2604.04924 | [PDF](https://arxiv.org/pdf/2604.04924v1)

**作者:** Sudarshan Rajagopalan `[一作]` (Johns Hopkins University), Vishal M. Patel `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过在冻结的预训练扩散模型（WAN和FLUX）上学习轻量级提示，解锁了其内在的多种图像/视频退化去除能力，实现了全方位恢复。

**💡 创新点**

创新点在于发现预训练扩散模型本身蕴含恢复知识，并通过直接在文本编码器输出空间学习提示而非传统文本或token提示来激活；同时提出能量导向桥（EBR）来对齐训练与推理轨迹，解决提示学习的轨迹不匹配问题。

**🔧 技术方法**

使用了无监督桥式提示学习、能量导向扩散桥、残差提示注入、冻结的大模型扩散框架，并与传统微调、ControlNet等方法进行对比。

**📊 数据集**

训练使用RESIDE、Snow100K、Rain13K、LOLv1、GoPro等数据集；测试覆盖ID、OOD、混合及未见退化（如HazeRD、LHP、WeatherBench、4KRD、SICE、TOLED、POLED、LOLBlur、NTU-Rain、SDSD等）以及视频的REVIDE、NTU-Rain、RVSD、SDSD、GoPro，并在对应的OOD视频集（RHVD、NTU-Rain(真实)、LasVR、AAURainSnow、Lol-iPhone、4KRD）上评估。

**📈 对比分析**

与DCPT、DFPIR、AutoDIR、PixWizard、FoundIR等现有AiOR方法在PSNR/SSIM/LPIPS/DISTS/CLIPIQA/MUSIQ及视频DOVER等指标上对比，本文方法在多种退化场景下取得竞争性甚至领先的感知与结构质量，同时保持极低的可训练参数。

**⚠️ 局限性**

局限性包括对极端或特殊退化的进一步调优需求、对训练集多样性敏感、桥式提示在高噪声下可能不稳定，以及未深入研究多任务混合退化下的联合提示效果。

---

## 440. Evaluating Digital Inclusiveness of Digital Agri-Food Tools Using Large Language Models: A Comparative Analysis Between Human and AI-Based Evaluations

**arXiv ID:** 2604.03252 | [PDF](https://arxiv.org/pdf/2604.03252v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 441. VIGIL: An Extensible System for Real-Time Detection and Mitigation of Cognitive Bias Triggers

**arXiv ID:** 2604.03261 | [PDF](https://arxiv.org/pdf/2604.03261v1)

**作者:** Bo Kang `[一作]` (Ghent University), Tijl De Bie `[通讯]` (Ghent University)

**通讯引用:** 7111 | [OpenAlex ID](https://openalex.org/A5076045275)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了名为Vigil的浏览器扩展，实现了实时检测和缓解在线内容中的认知偏差触发器，支持滚动同步、可逆改写和插件扩展；

**💡 创新点**

创新点在于将认知偏差触发器检测嵌入浏览体验，提供即时、可逆的文本改写，并采用多层隐私级别的推理架构；

**🔧 技术方法**

技术包括基于正则的零延迟检测、WebGPU+WebLLM（Llama 3.2 1B）、本地Ollama API和云端OpenAI兼容接口，插件系统采用统一契约；

**📊 数据集**

使用SemEval-2020 Task 11（宣传技术）和Moralization Corpus（道德化）作为评测数据集；

**📈 对比分析**

与基准相比，系统在SemEval-2020的micro‑F1达0.533（精度0.626），在Moralization Corpus的macro‑F1达0.789；正则层毫秒级，WebGPU层约3.4 s，云端约3.9 s；

**⚠️ 局限性**

主要限制包括LLM产生误报、对代理基准的依赖以及覆盖范围有限，且需人工反馈以进一步提升准确率。

---

## 442. Linear Exact Repair in MDS Array Codes: A General Lower Bound and Its Attainability

**arXiv ID:** 2604.04519 | [PDF](https://arxiv.org/pdf/2604.04519v1)

**作者:** Hai Liu `[一作]`, Huawei Wu `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究线性MDS阵列码在给定子分组水平、冗余和域大小下的最小修复带宽和I/O，提出几何计数下界并给出两冗余情形下的最优构造。

**💡 创新点**

首次给出适用于任意冗余≥2的下界，并证明仅在两冗余时可达到，且用Desarguesian散列构造实现最优。

**🔧 技术方法**

几何子空间交叉、射影空间计数、有限几何中的Desarguesian散列与范式映射。

**📊 数据集**

无需具体数据集，理论分析与构造基于有限域\(_q\)和散列集。

**📈 对比分析**

通过比较下界与构造达到的修复带宽/ I/O，证明在两冗余时两者相等，若\(r\ge3\)则严格大于下界。

**⚠️ 局限性**

在\(r\ge3\)、\(ℓ\ge2\)时下界不可达；两冗余时长度受限制，且仅对特定散列结构有效；未给出更广泛长度范围或更大冗余的构造。

---

## 443. Uncertainty as a Planning Signal: Multi-Turn Decision Making for Goal-Oriented Conversation

**arXiv ID:** 2604.03924 | [PDF](https://arxiv.org/pdf/2604.03924v1)

**作者:** Xinyi Ling `[一作]` (Ohio State University), Xia Ning `[通讯]` (Ohio State University)

**通讯引用:** 6126 | [OpenAlex ID](https://openalex.org/A5035648686)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Conversation Uncertainty-aware Planning 框架，将不确定性作为全局决策信号，结合 LLM 生成动作与 MCTS 规划实现目标导向对话中的信息获取与目标承诺协同。

**💡 创新点**

创新点在于将熵、期望信息增益等不确定性指标嵌入树搜索优先级，实现在多轮对话中对不确定性进行长期规划，从而突破仅局部优化的局限。

**🔧 技术方法**

核心技术包括 LLM（Qwen3‑4B、Mistral‑7B‑v0.3、Llama‑3.1‑8B）生成动作与执行、贝叶斯多步更新、熵/期望信息增益评估、Monte Carlo Tree Search（MCTS）规划。

**📊 数据集**

使用 Beauty、Fashion、Home、Inspired 四个常用电商/电影推荐对话数据集，候选集由 SBERT 生成。

**📈 对比分析**

与 SBERT、Direct Prompting、CoT、LLM‑planning、UoT、ATD、BED‑LLM、MISQ‑HF 等基线对比，成功率提升 15–30%，平均对话轮次从 4.8 降至 3.2–4.1，显著提高效率与准确性。

**⚠️ 局限性**

局限在于需预设熵阈值与候选集构造，极端语义模糊场景仍可能误判；对 LLM 质量与推理速度敏感，且未处理候选集规模极大时的计算开销。

---

## 444. YANA: Bridging the Neuromorphic Simulation-to-Hardware Gap

**arXiv ID:** 2604.03432 | [PDF](https://arxiv.org/pdf/2604.03432v1)

**作者:** Brian Pachideh `[一作]` (FZI Research Center for Information Technology), Juergen Becker `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5123668182)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了YANA，一个基于FPGA的可编程、事件驱动的SNN加速器，旨在弥合仿真与硬件之间的差距。

**💡 创新点**

创新点在于：完整支持任意SNN拓扑的点到点连接，利用五阶段事件驱动流水线高效挖掘时空稀疏性，并公开开源硬件与软件框架。

**🔧 技术方法**

采用FPGA近内存计算、查找表实现泄漏计算、固定点LIF模型、AXI4流接口、NIR中间表示等技术。

**📊 数据集**

使用Spiking Heidelberg Digits（SHD）数据集进行训练与评估。

**📈 对比分析**

通过在AMD Kria KR260上部署单核实验，测得推理时间随空间/时间稀疏线性下降，最快仅1 ms，验证了事件驱动架构在稀疏条件下的高效性。

**⚠️ 局限性**

局限性包括：目前仅支持前馈层、缺乏功耗评估、尚未实现多核NoC扩展，且对卷积等结构支持不足。

---

## 445. Self-Execution Simulation Improves Coding Models

**arXiv ID:** 2604.03253 | [PDF](https://arxiv.org/pdf/2604.03253v1)

**作者:** Gallil Maimon `[一作]` (Hebrew University of Jerusalem), Yossi Adi `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 3638 | [OpenAlex ID](https://openalex.org/A5005191803)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练大语言模型学习如何逐步模拟代码执行，并利用这种模拟能力在竞争性编程任务中进行自我验证和迭代修正。

**💡 创新点**

将自然语言执行轨迹与可验证奖励的强化学习相结合，使模型既能预测代码执行结果，又能在生成的代码上进行自我修复，显著提升代码正确率。

**🔧 技术方法**

采用监督微调（SFT）处理自然语言执行说明，随后使用可验证奖励的RL（RLVR）优化输出预测和竞赛解题双任务，并设计最佳@k自我模拟与多轮自我修复（Self‑RLEF）框架。

**📊 数据集**

数据集包含约30M通用Python函数与35k竞赛级别程序的执行轨迹，利用Llama3‑70B生成自然语言说明，并在Qwen2.5、CWM等模型上进行训练。

**📈 对比分析**

与传统推理模型、官方CWM以及使用真实执行反馈的基线相比，最佳@k自我模拟提升约2–8个百分点，Self‑RLEF在pass@1/5/10上分别提升约3–6个百分点，整体性能提升最高可达39%。

**⚠️ 局限性**

模拟对复杂运算（如大数乘、对数等）精度不足，训练稳定性受限于丰富文本反馈，且方法目前仅适用于单文件竞赛题，难以直接推广至完整软件工程任务。

---

## 446. Multi-Agent Training-free Urban Food Delivery System using Resilient UMST Network

**arXiv ID:** 2604.03280 | [PDF](https://arxiv.org/pdf/2604.03280v1)

**作者:** Md Nahid Hasan `[一作]` (Miami University), Snehanshu Saha `[通讯]` (BITS Pilani)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建一种称为 Union Minimum Spanning Tree (UMST) 的稀疏且冗余的配送网络骨干，实现在线食品配送的高效且可靠路径规划和订单合并。

**💡 创新点**

创新点在于通过随机边扰动生成多棵最小生成树并取并集，既保留了 MST 的低成本与高可扩展性，又显著提升了网络的冗余性与容错能力；此外将此结构直接用于多机车调度与即时订单捆绑，避免了昂贵的学习训练。

**🔧 技术方法**

技术方法包括：图构建（利用 OpenStreetMap + GraphHopper 得到真实道路权重）、随机边删除与 MST 采样、UMST 生成、基于 UMST 的贪心分配与捆绑、以及与学习型方法（MADDPG、GNN）和传统完整图/MST 基线的实验比较。

**📊 数据集**

使用真实城市数据：哥伦布（26 区块）和芝加哥（31 区块）的餐厅、配送中心、居民建筑位置；通过 OpenStreetMap、美国人口普查区块和 GraphHopper 提取道路网络和行驶时间。

**📈 对比分析**

与完全连通图、单一 MST、学习型基线（MADDPG、GNN）以及注意力增强 MADDPG 进行对比。UMST 在成功率、平均交付时间、车辆行驶距离、订单捆绑参与率等指标上均达到或超过学习型方法，且执行速度比学习方法快 30 倍，训练时间为 0。

**⚠️ 局限性**

局限性包括：随机扰动不考虑空间/时间动态特性；仅在固定时间窗口内评估；对不同车辆容量、异构车队的适应性尚未验证；当需求模式剧烈变化时仍需重新构建图，虽然成本低，但可能存在更新延迟；对极端道路封闭情形下的最优路径选择未充分探究。

---

## 447. Automated Conjecture Resolution with Formal Verification

**arXiv ID:** 2604.03789 | [PDF](https://arxiv.org/pdf/2604.03789v1)

**作者:** Haocheng Ju `[一作]` (Peking University), Bin Dong `[通讯]` (Peking University)

**通讯引用:** 16646 | [OpenAlex ID](https://openalex.org/A5100746745)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个完整的自动化框架，集成自然语言推理代理 Rethlas 与正式化代理 Archon，用于在极小人类干预下自动解决研究级数学问题并对证明进行正式验证。

**💡 创新点**

创新点在于：① 将 LLM 推理与形式化验证无缝衔接；② 通过 Matlas 与 LeanSearch 两大检索工具实现跨领域知识检索与库搜索；③ 引入双代理架构（Plan Agent + Lean Agent）以缓解上下文污染、提升推理效率；④ 通过自动化填补、错误诊断与跨会话重构，使正式化几乎无需人工介入；⑤ 以 Anderson 的开放问题为案例，首次实现完整的自动推导与 Lean4 正式化。

**🔧 技术方法**

使用技术包括：大规模语言模型（GPT‑5.4、Gemini、Claude Code Max 等）、Matlas 语义检索引擎、LeanSearch 形式化检索、双代理 LLM 工具调用架构、记忆管理与 Review Agent、Mathlib 库、Lean4 编译器与 Comparator 校验工具。

**📊 数据集**

主要数据集：Matlas（约 13.6M 数学语句），Mathlib（267k 定理 + 127k 定义），FirstProof 研究级问题集，外部参考文献（Anderson、Jensen 等）以及公开的 benchmark（IMO、Putnam 等）。

**📈 对比分析**

与以往系统（Aletheia、AlphaProof、Seed‑Prover、Aristotle 等）对比：Rethlas 能自动解决 Anderson 开放问题，Archon 能在 80h 内生成 19k 行 Lean4 代码并通过 Comparator 验证，完全无需人工编写证明；在 FirstProof 基准中，Archon 在无监督下完成了 2 项证明（其中 1 项完全自动，1 项仅需一次提示），而其他系统往往需要人工干预或无法完成。性能方面，正式化任务的自动化时间和代码量相当于数人月的人工工作。

**⚠️ 局限性**

局限性：① 代理在面对缺失或不完整的证明时可能过度搜索、延迟推进；② 生成的 Lean 代码往往冗长、缺乏 Mathlib 风格的简洁性与命名规范；③ 对于复杂问题，仍需要少量人类提示以提升效率；④ 需人工获取付费 PDF 并手动放置；⑤ 在极大规模证明或极高复杂度任务时，上下文管理与计算资源瓶颈仍是挑战。

---

## 448. Build on Priors: Vision--Language--Guided Neuro-Symbolic Imitation Learning for Data-Efficient Real-World Robot Manipulation

**arXiv ID:** 2604.03759 | [PDF](https://arxiv.org/pdf/2604.03759v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 449. Runtime Enforcement for Operationalizing Ethics in Autonomous Systems

**arXiv ID:** 2604.03714 | [PDF](https://arxiv.org/pdf/2604.03714v1)

**作者:** Martina De Sanctis `[一作]` (Gran Sasso Science Institute), Patrizia Scandurra `[通讯]` (University of Bergamo)

**通讯引用:** 2006 | [OpenAlex ID](https://openalex.org/A5064186815)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

本文提出了一个基于运行时执行的伦理保证流程，用于在自主系统中实现并执行伦理规范（SLEEC 规则），并通过一个可插拔的伦理执行子系统来动态监测和修正机器人行为。

**💡 创新点**

创新点在于：①将伦理规则从高层规范迁移到运行时可执行的 ASM（Abstract State Machine）模型；②构建了 MAPE‑K 结构的伦理执行子系统，实现对伦理空间的实时监控与约束；③提供了完整的从规则挖掘、形式化、验证到在线执行的闭环流程，首次实现了伦理规则的自动化执行与动态演化。

**🔧 技术方法**

采用技术包括：SLEEC 规则语言、Abstract State Machine（ASM）与 ASMETA 运行时模拟器、MAPE‑K 控制循环、RESTful 接口与消息队列（RabbitMQ）进行组件通信、ROS2 机器人框架、Java 与 Python 组合实现服务。

**📊 数据集**

实验数据集：1）AssistiveCareRobot 场景的 9 条 SLEEC 规则与 10 条 ASM 规则；2）生成的 110 个合成 ASM 模型，用于可扩展性评估；3）750 条随机测试用例（模拟情境）与 5500 条测试用例（可扩展性实验）分别用于验证与性能评估。

**📈 对比分析**

比较方法：将伦理执行子系统的执行时间（总体、Enforcer、ASMETA 服务器）与机器人任务完成时间对比；与已有 SLEEC 规则集（如 Autocar）做规模对比；结果显示平均总开销为 76.1 ms（最大 104 ms），ASMETA 服务器开销仅 1.68 ms，且随着规则数增大呈二次（近似多项式）增长，实际应用场景中规则规模（≤10 条）对性能影响几乎可以忽略。

**⚠️ 局限性**

限制与挑战：①对 ASMETA 服务器的二次开销在极大规则集下仍显著；②当前实现未支持时间约束的完整执行（如超时处理），未来需集成任务调度器；③依赖外部 ROS2 机器人框架，跨平台移植性需进一步验证；④实验环境仅在 Turtlebot 与 ARI 机器人上验证，缺乏对更复杂环境（多机器人、网络边缘）下的评估。

---

## 450. MedROI: Codec-Agnostic Region of Interest-Centric Compression for Medical Images

**arXiv ID:** 2604.04511 | [PDF](https://arxiv.org/pdf/2604.04511v1)

**作者:** Jiwon Kim `[一作]` (Hankuk University of Foreign Studies), Ikbeom Jang `[通讯]` (Hankuk University of Foreign Studies)

**通讯引用:** 385 | [OpenAlex ID](https://openalex.org/A5026873215)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出并实现了一个通用的 ROI 压缩框架 MedROI，先对医学影像进行轻量级阈值分割裁剪非诊断背景，只保留脑组织区域进行压缩，并仅存储 54 字节的元数据用于空间恢复；可与任何现有 2D/3D 编码器无缝配合。

**💡 创新点**

创新点在于：① 仅裁剪并删除背景而非仅降低其压缩质量，显著减少需编码的数据量；② 通过固定 54 字节元数据实现元数据最小化；③ 该框架完全 codec-agnostic，无需改动现有压缩算法或进行再训练，具有高度可部署性。

**🔧 技术方法**

技术手段包括：① 基于全局强度阈值的 ROI 提取与自适应 padding；② 54 字节元数据格式（Bounding Box、原始尺寸、旋转缩放矩阵）；③ 与多种压缩器（JPEG2000 2D/3D、HEIF、TCM、TCM+AuxT、BCM‑Net、SirenMRI）集成，验证其通用性。

**📊 数据集**

主要数据集：ADNI 200 份 T1 加权脑 MRI（体素尺寸 ≤ 256³）；此外在 INbreast 乳腺 X 光影像 15 份上做了小规模验证。

**📈 对比分析**

比较方法：对每个压缩器分别在全量压缩与 ROI 模式下计算压缩比（CR）、比特/像素（BPP）、PSNR、SSIM，以及编码/解码耗时。结果显示：大多数算法（如 TCM、TCM+AuxT、BCM‑Net、JPEG2000）在 ROI 模式下 CR 提升 15–30%，编码/解码时间下降 10–30%，重建质量在 ROI 区域保持 PSNR ≥30 dB；HEIF 仅略有 CR 变化，压缩速度提升更为显著。

**⚠️ 局限性**

局限性：① 评估主要集中在脑 MRI 上，缺乏对其他解剖结构或模态的广泛验证；② 质量评估仅使用 PSNR/SSIM 等客观指标，未涉及临床诊断有效性；③ 轻量级阈值分割在极端病例或噪声较多的图像中可能不够鲁棒。

---

## 451. Explainable Autonomous Cyber Defense using Adversarial Multi-Agent Reinforcement Learning

**arXiv ID:** 2604.04442 | [PDF](https://arxiv.org/pdf/2604.04442v1)

**作者:** Yiyao Zhang `[一作]` (Adelaide University), Hussain Ahmad `[通讯]` (Adelaide University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 Causal Multi-Agent Decision Framework (C-MADF)，一种基于因果结构约束的自主网络防御体系，结合 SCM 学习、受限 MDP-DAG 路径规划、双重代理（Blue‑Team 与 Red‑Team）强化学习决策和可解释的升级机制。

**💡 创新点**

创新点在于：① 将因果模型直接编译成可执行的 MDP-DAG，强制性限制可执行行动；② 采用对抗性双代理进行内部审查，利用 Policy Divergence 量化不确定性；③ 设计 ETS（Explainability–Transparency Score）将解释性与决策可信度量化为人机交互的阈值信号，实现自动执行与人工升级的闭环。

**🔧 技术方法**

核心技术包括：结构因果推断（PC/FCI 等）、受限 MDP 与 DAG 建模、双代理深度强化学习（DQN/Actor‑Critic）、策略分歧度量、ETS 组合公式、离线因果发现与在线策略自适应。

**📊 数据集**

实验数据集为 CICIoT2023（IoT 安全流量与攻击标签），并使用相同的特征预处理和训练/验证/测试划分进行评估。

**📈 对比分析**

与 DUGAT‑LSTM、BiLSTM IDS、Hybrid Weighted‑XGBoost 三个近期基线在相同数据集上进行对比。C‑MAFD 达到 0.997 的精确率、0.961 的召回率、0.979 的 F1，且误报率（FPR）降至 1.8%（比基线分别低 83.9%、81.4% 和 78.6%）。

**⚠️ 局限性**

局限性包括：① 对历史数据的因果学习质量高度依赖，若存在持久性污染或概念漂移会导致约束失效；② 双代理推理增加计算开销且若两代理共享相同盲点，分歧可能不足；③ 需预先定义调查本体，难以适应未知攻击流程；④ 现有验证主要基于实验室数据，真实环境中鲁棒性与可部署性仍需进一步验证。

---

## 452. Compressible Softmax-Attended Language under Incompressible Attention

**arXiv ID:** 2604.04384 | [PDF](https://arxiv.org/pdf/2604.04384v1)

**作者:** Wonsuk Lee `[一作]` (Seoul National University), Wonsuk Lee `[通讯]` (Seoul National University)

**通讯引用:** 783 | [OpenAlex ID](https://openalex.org/A5107341234)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了Transformer注意力头的行归一化能量场与权重矩阵的奇异值分解，评估其在不同模型中的压缩性。

**💡 创新点**

发现权重矩阵谱分布几乎均匀，表明无法通过固定低秩投影压缩；但能量场的谱在任何给定上下文中集中到少量主成分，压缩依赖于输入数据而非权重结构。

**🔧 技术方法**

使用奇异值分解（SVD）、行归一化、RoPE以及统计分析等技术对模型权重和能量场进行谱分析。

**📊 数据集**

以五篇公开文本（狄更斯作品、达尔文、莎士比亚、圣经、亚当·斯密）为输入，在GPT‑2、LLaMA、Qwen、Mistral等五个Transformer模型上进行实验。

**📈 对比分析**

比较生成能量场与学习交互矩阵的有效秩，结果显示能量场在90%方差下仅需2–11个奇异值成分，而学习矩阵需38–75个，表明数据驱动的压缩空间巨大。

**⚠️ 局限性**

实验仅覆盖已训练语言模型，未检验跨模态通用性；softmax后的误差理论仍未完全解决，且对长序列的压缩效果未作系统评估。

---

## 453. Triggering and Detecting Exploitable Library Vulnerability from the Client by Directed Greybox Fuzzing

**arXiv ID:** 2604.04102 | [PDF](https://arxiv.org/pdf/2604.04102v1)

**作者:** Yukai Zhao `[一作]` (Zhejiang University), Xin Xia `[通讯]` (Zhejiang University)

**通讯引用:** 21342 | [OpenAlex ID](https://openalex.org/A5006669765)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出一种无PoC的 directed greybox fuzzing 方法，用于检测客户端程序对第三方库漏洞的可利用性；

**💡 创新点**

创新点包括：①构造目标元组〈caller, vulnerability〉来统一跨程序的距离度量；②引入抽象路径映射机制公平比较不同可达路径的种子；③设计基于风险的自适应变异，动态调整粗细变异比例；

**🔧 技术方法**

主要技术：DGF（AFLGo 体系），基本块/函数距离计算，风险度量与自适应变异策略，支持可选的键函数信息提取（可使用 LLM）；

**📊 数据集**

使用了 61 个真实案例的数据集，涵盖 42 个库漏洞、7 个库、14 种漏洞类型，平均 CVSS 5.3‑9.8；

**📈 对比分析**

与 AFLGo、SelectFuzz、WindRanger、AFL++ 等基线对比，覆盖目标可达路径提升 37%‑195%，暴露速度提升 4.74×‑7.08×，并能触发三项仅被该方法发现的漏洞；

**⚠️ 局限性**

局限性：仅针对可获得源代码的 C/C++ 项目；需要漏洞描述或键函数信息；未处理深度嵌套库（n‑tuple 方案待实现）；多语言适用性尚未验证。

---

## 454. Internet-Mediated Digital Informal Learning Portfolios in STEM Higher Education: A Computational Grounded Theory Study of Online Peer Advice Communities

**arXiv ID:** 2604.03643 | [PDF](https://arxiv.org/pdf/2604.03643v1)

**作者:** Jianjun Xiao `[一作]` (Beijing Normal University), Yuxi Long `[通讯]` (Beijing Normal University)

**通讯引用:** 169 | [OpenAlex ID](https://openalex.org/A5073076586)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 STEM 学生如何通过互联网同伴建议构建数字化非正式学习组合。

**💡 创新点**

提出“数字非正式学习组合”概念和“职业前置化”理论，并将 SCCT 扩展到在线学习情境。

**🔧 技术方法**

采用计算机辅助扎根理论 (CGT) 进行文本编码与共现网络分析。

**📊 数据集**

使用 3,607 条来自中国大型学生社区的同伴建议帖子。

**📈 对比分析**

通过共现提升和频数阈值筛选关键关联，识别三种组合类型，展示方法能捕捉职业导向的学习模式。

**⚠️ 局限性**

受限于关键词编码对隐喻、讽刺的敏感度、样本可能偏向具备反思能力的学生、单一国家语境及横断面设计。

---

## 455. Synthetic Sandbox for Training Machine Learning Engineering Agents

**arXiv ID:** 2604.04872 | [PDF](https://arxiv.org/pdf/2604.04872v1)

**作者:** Yuhang Zhou `[一作]` (Meta AI), Hong Yan `[通讯]` (Meta AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过多代理协同生成微尺度(MLE)合成沙盒环境，并在此环境上实现基于轨迹的GRPO强化学习，显著提升LLM在机器学习工程任务中的推理与执行能力。

**💡 创新点**

提出了可扩展的合成MSE沙盒生成框架，利用多代理抽取“结构DNA”并生成可验证的微型数据集，从而将每一步执行延迟从200秒降至15秒，实现轨迹级RL的可行性；同时设计了分层里程碑奖励以缓解稀疏奖励难题。

**🔧 技术方法**

多代理脚本化合成管线（Data Strategist、MLE Developer、MLOps Engineer、Technical Writer）、ReAct框架下的轨迹生成、GRPO优化、稠密里程碑奖励以及对话式代理（AIDE、AIRA、MLE-Agent）等。

**📊 数据集**

以MLE‑Bench种子任务为基础生成了848个合成训练任务和64个验证任务，评估使用MLE‑Bench‑Lite（22个未见任务）与MLE‑Dojo（62个Kaggle任务）。

**📈 对比分析**

与原始基础模型、SFT基线、以及大型封闭源模型（Claude‑4.5‑Sonnet、DeepSeek‑V3.1、Gemini‑2.5‑Flash）进行对比，取得相对改进率20.3%–66.9%的Medal‑Rate提升，且在不同代理框架下保持显著的泛化性能。

**⚠️ 局限性**

仍受合成任务与真实大规模数据分布差异的影响，合成环境的可扩展性和对极端数据噪声的鲁棒性待进一步验证；轨迹级RL虽加速，但在极大模型规模下仍受上下文窗口限制，导致测试时扩展性下降。

---

## 456. Amplifying Rural Educators' Perspectives: A Qualitative Study of Generative AI's Impact in Rural U.S. High Schools

**arXiv ID:** 2604.03542 | [PDF](https://arxiv.org/pdf/2604.03542v1)

**作者:** Shira Michel `[一作]` (Roux Institute at Northeastern University), Mahsan Nourani `[通讯]` (Roux Institute at Northeastern University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对美国三州农村高中教师进行在线问卷与半结构化访谈，系统分析他们对生成式人工智能（GenAI）的使用体验、挑战与机遇。

**💡 创新点**

首次聚焦农村教育环境下的 GenAI 采用与障碍，结合批判性乡村理论揭示结构性不平等与技术适配需求，为教育技术研究提供乡村视角。

**🔧 技术方法**

采用定性研究方法：在线调查、半结构化访谈；使用开放编码、扎根理论和 Miro 进行主题聚类；并应用 CARS 与 TAM 量表评估教师态度。

**📊 数据集**

收集了 31 名来自亚利桑那、缅因和北卡罗来纳州农村高中的教师问卷与访谈数据，涵盖他们的教学背景、技术使用情况及对 GenAI 的期望。

**📈 对比分析**

研究未涉及算法性能比较，而是通过编码一致性与主题共识评估；结果显示大多数教师认为 GenAI 在教学管理、课程生成和个性化支持方面具有积极效用，但同时存在技术与资源瓶颈。

**⚠️ 局限性**

局限性包括样本规模有限、仅覆盖三州、受访者自选偏差、对最缺乏连接的教师代表性不足，以及未收集学生、家长或更广泛教育者视角。

---

## 457. Align then Train: Efficient Retrieval Adapter Learning

**arXiv ID:** 2604.03403 | [PDF](https://arxiv.org/pdf/2604.03403v1)

**作者:** Seiji Maekawa `[一作]` (Megagon Labs), Estevam Hruschka `[通讯]` (Megagon Labs)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了Efficient Retrieval Adapter（ERA），一种通过自监督对齐和监督适配两阶段训练的轻量级检索适配器，用于解决查询复杂、文档简洁导致的检索不匹配问题。

**💡 创新点**

创新点在于（1）将适配器训练拆分为自监督对齐阶段和少量监督适配阶段，显著提升低标签场景性能；（2）支持查询与文档使用不同、甚至不同家族的编码器，实现在不重新索引的异构检索；（3）仅使用线性适配器，保持黑盒API兼容与部署成本低。

**🔧 技术方法**

使用技术包括：查询与文档嵌入的自监督对齐（通过匹配相同文档的两种嵌入）；对比损失（InfoNCE/三元组）进行监督适配；TopK-PercPos硬负样本采样；线性权重矩阵作为适配器；在实验中采用BERT、LLM、OpenAI等多种编码器。

**📊 数据集**

实验数据集为MAIR benchmark（126个检索任务，涵盖学术、代码、金融、法律、医疗、网络等六大领域），基于BEIR/KILT构建，平均每任务约80条标注样本，采用5%–40%标签比例进行评估。

**📈 对比分析**

与Zero-shot检索、Embedding Adapter、Search Adaptor、Drift-Adapter等基线进行对比，ERA在低标签（5%–40%）下平均nDCG@10提升约8%–12%，在异构检索设置中实现更大幅度提升，并且在不需重建索引的情况下实现强大的跨模型适配效果。

**⚠️ 局限性**

限制包括：对齐阶段依赖大量未标记文档；适配器仅为线性变换，可能在极端跨模态差异或高维复杂语义上受限；对极少标签（<5%）的鲁棒性尚未充分验证；未对实时部署的延迟与资源消耗进行深入评估。

---

## 458. Efficient Multi-Objective Planning with Weighted Maximization Using Large Neighbourhood Search

**arXiv ID:** 2604.04826 | [PDF](https://arxiv.org/pdf/2604.04826v1)

**作者:** Krishna Kalavadia `[一作]` (University of Waterloo), Stephen L. Smith `[通讯]` (University of Waterloo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一种基于大型邻域搜索（Large Neighbourhood Search）的加权最大（WM）规划算法 WM‑LNS，用于离散多目标路径规划，能够在保持近似最优的同时显著加快求解速度。

**💡 创新点**

创新点在于：①利用 WM 子问题的 Pareto 前沿凸性，借助 WS 子问题实现高效求解；②在 LNS 框架内自适应选择破坏与修复策略，并通过 GPS/随机采样动态确定修复权重；③在实验中证明 WM‑LNS 能恢复 WS 无法获取的非凸 Pareto 解。

**🔧 技术方法**

核心技术包括：Large Neighbourhood Search、Weighted Sum（WS）与 Weighted Maximum（WM）标量化、A* 修复、模拟退火接受准则、基于梯度无关的 Generalized Pattern Search（GPS）权重优化。

**📊 数据集**

使用的测试数据集：室内障碍物的概率路网图（PRM）在三种规模（小/中/大）下、三目标（路径长度、障碍物距离、风险）和七自由度 Franka Emika Panda 机器人厨房环境（7500 节点 PRM）。

**📈 对比分析**

与 WM、WM‑poly、WM‑beam（自研 beam 版）以及 WS 进行对比。WM‑LNS 在大规模、三目标场景下实现 10–100 倍速度提升，平均百分比误差 0.3%–2.7%，覆盖率和 Pareto 解数与 WM 差距不大，甚至在 2 目标场景下覆盖率提升 13%+。

**⚠️ 局限性**

局限性：① 破坏/修复与权重选择仍为启发式，无法保证全局最优；② 对维度更高的权重空间（>2）时随机采样效率下降；③ 需要手工调节多项超参数，且在极大搜索空间下仍可能出现次优解。

---

## 459. ResGuard: Enhancing Robustness Against Known Original Attacks in Deep Watermarking

**arXiv ID:** 2604.03693 | [PDF](https://arxiv.org/pdf/2604.03693v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 460. Towards Intelligent Energy Security: A Unified Spatio-Temporal and Graph Learning Framework for Scalable Electricity Theft Detection in Smart Grids

**arXiv ID:** 2604.03344 | [PDF](https://arxiv.org/pdf/2604.03344v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 461. Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations

**arXiv ID:** 2604.04863 | [PDF](https://arxiv.org/pdf/2604.04863v1)

**作者:** Tuan Dung Nguyen `[一作]` (Hanoi University of Science and Technology), Vu Minh Hieu Phan `[通讯]` (Australian Institute for Machine Learning, University of Adelaide)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

分析大型视觉语言模型在 token 级的幻觉产生机制，并提出基于补丁级注意力分散度和跨模态对齐一致性的检测框架。

**💡 创新点**

创新在于从补丁层面细粒度考察注意力与视觉区域的对应关系，发现幻觉 token 的注意力分散且与视觉区域对齐低，从而设计两项指标 ADS 和 CGC。

**🔧 技术方法**

采用 Transformer 内部跨模态注意力权重、注意力熵与语义相似度计算，并结合轻量级 XGB/MLP/随机森林分类器进行检测。

**📊 数据集**

在 MS‑COCO 2014、POPE 等公开视觉语言数据集上进行评估。

**📈 对比分析**

与 MetaToken、HalLoc、SVAR、DHCP、ProjectAway 等基线比较，平均 F1/ROC‑AUC 提升约 10–20%，单模型最高可达 0.92/0.95。

**⚠️ 局限性**

仍受限于高层语言先验影响的区分、少量样本标签噪声，以及对更复杂多模态输入结构的覆盖不足。

---

## 462. A Validated Taxonomy on Software Energy Smells

**arXiv ID:** 2604.04809 | [PDF](https://arxiv.org/pdf/2604.04809v1)

**作者:** Mohammadjavad Mehditabar `[一作]` (Dalhousie University), Tushar Sharma `[通讯]` (Dalhousie University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文基于系统文献回顾与基于扎根理论的编码方法，构建了12类、65个根因的语言无关软件能耗臭味分类体系，并通过对21,428条Python代码对进行能耗测量与三步LLM分类，对该体系进行经验验证。

**💡 创新点**

创新点在于首次提出了完整、可验证且跨语言的两层能耗臭味分类框架，同时发布了包含能耗、内存和时间指标的标注数据集，为绿色软件工程提供了统一的术语和实证基础。

**🔧 技术方法**

主要技术包括：系统文献检索与向前向后雪球搜索；扎根理论编码（开放、轴向、选择性编码）构建分类；基于GPT的三步推理分类管道（根因分析→类别分层→子类细化）；以及Linux perf工具的能耗、内存与时间测量。

**📊 数据集**

使用的核心数据集是Pie‑Perf（从IBM CodeNet衍生的Python竞赛编程代码对），共21,428对功能等价实现，其中选取了能耗差异最大的3,000对用于分类与验证，并将标注结果公开。

**📈 对比分析**

通过与人工标注的100条样本对比，LLM分类准确率达94%；在3,000条样本中，71%出现多重臭味，能耗节约率平均为1,081 J，单一类别的平均节约为549 J，显示多重臭味组合可带来显著的能耗提升，验证了分类体系的有效性。

**⚠️ 局限性**

局限性包括：验证仅在单线程、纯Python算法竞赛场景下进行，缺乏多线程、数据库和嵌入式环境的覆盖；能耗测量受限于单一硬件配置；LLM可能存在偏差或误判；以及部分域特定臭味未在数据集中出现。

---

## 463. LOCALUT: Harnessing Capacity-Computation Tradeoffs for LUT-Based Inference in DRAM-PIM

**arXiv ID:** 2604.04523 | [PDF](https://arxiv.org/pdf/2604.04523v1)

**作者:** Junguk Hong `[一作]`, Jinho Lee `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

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

## 464. Assessing Cyber Risks in Hydropower Systems Through HAZOP and Bow-Tie Analysis

**arXiv ID:** 2604.03994 | [PDF](https://arxiv.org/pdf/2604.03994v1)

**作者:** Kwabena Opoku Frempong-Kore `[一作]` (University of Illinois), Bell Eapen `[通讯]` (University of Illinois)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文通过传统与网络化扩展的 HAZOP 与 BowTie 分析，系统识别并评估了智能水电站泄洪门控制系统的安全与网络风险，提出两步方法以揭示传统安全方法与网络威胁之间的差距；

**💡 创新点**

创新点在于将 HAZOP 与 BowTie 结合并引入网络化扩展，突出协同攻击与共享网络依赖对假设（独立性）的破坏；

**🔧 技术方法**

使用了 HAZOP（基于指导词的偏差分析）、BowTie（威胁–事件–后果图）以及网络攻击场景设计（假设攻击向量、数据注入、服务拒绝等）进行风险评估；

**📊 数据集**

研究以单一泄洪门控制子系统为案例，不涉及公开数据集，而是基于水电站控制系统架构与典型参数构建的模型；

**📈 对比分析**

通过对比传统与扩展版本，发现传统 HAZOP 仅能发现单一偏差，BowTie 侧重可视化并未体现共享网络依赖；扩展后两者共同揭示了协同攻击的隐蔽性和防护互依性，但未给出量化性能指标；

**⚠️ 局限性**

局限在于只分析了一个子系统，缺乏对其他控制回路的验证；缺少量化风险评估与攻击可利用性分析；方法依赖手工分析，难以规模化。

---

## 465. RegGuard: Legitimacy and Fairness Enforcement for Optimistic Rollups

**arXiv ID:** 2604.04748 | [PDF](https://arxiv.org/pdf/2604.04748v1)

**作者:** Zhenhang Shang `[一作]` (Hong Kong University of Science and Technology), Kani Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 RegGuard 框架，在 Optimistic Rollup 中加入语义合法性验证、跨层状态一致性检查以及可验证公平排序，提升监管合规性

**💡 创新点**

将可判定的 RegSpec 规则语言、L1 状态缓存预同步、阈值加密公平排序三大机制统一到同一流水线，并给出形式化正确性与概率安全证明

**🔧 技术方法**

使用 RegSpec（有限谓词逻辑）和 WebAssembly 沙箱、Merkle Patricia Trie 缓存、阈值 BLS 加密/解密、Rust 实现、WASM 编译、时间戳签名等技术

**📊 数据集**

通过在 AWS 多区域测试床上生成合成监管金融工作负载（代币转账、跨链桥接、MEV 套利）进行评测

**📈 对比分析**

与 Optimism Bedrock 与 Arbitrum Nitro 进行基准比较，RegGuard 在 10,000 TPS 场景下保持 85% 以上吞吐量，结算失败率下降 92%，公平排序违约概率 <10⁻⁶，延迟增加 180–400 ms

**⚠️ 局限性**

受限于规则可判定性（无法表达复杂时序逻辑）、需要多数诚实委员会、一定的性能开销（CPU 20–35% 增长、延迟 180–400 ms），以及部署与监管规则转换的复杂性

---

## 466. Scaling Multi-agent Systems: A Smart Middleware for Improving Agent Interactions

**arXiv ID:** 2604.03430 | [PDF](https://arxiv.org/pdf/2604.03430v1)

**作者:** Charles Fleming `[一作]` (Cisco Research), Vijoy Pandey `[通讯]` (Cisco Research)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Cognitive Fabric Nodes (CFN) 中间件，通过主动内存、拓扑选择、语义定位、安全策略和提示重写五大功能，实现多智能体系统的协作与安全。

**💡 创新点**

创新点在于将记忆视为主动认知子strate，将中间件本身变为智能化“认知网络”，并通过强化学习驱动的认知引擎实现动态拓扑、语义对齐、零信任安全和提示重写。

**🔧 技术方法**

使用的技术包括强化学习 (RL)、上下文 Bandit、向量搜索 (HNSW)、小型LLM 进行提示重写、LangMARL/TextGrad 语言信用分配，以及混合规则-学习安全引擎。

**📊 数据集**

使用 HotPotQA 与 MuSiQue 两个多步问答数据集。

**📈 对比分析**

对比方法：基线直接 Agent 对 Agent 通信、全局优化的 TextGrad、CFN-LangMARL。性能表现：HotPotQA 91.5%（接近基线 92%），MuSiQue 86.1%（接近基线 87.5%），相较基线分别提升约 10% 与 13%。

**⚠️ 局限性**

局限包括：同步一致性导致的延迟、认知计算产生的“认知税”、对极端复杂任务的可扩展性尚未充分验证、未实验跨 Fabric 通信、记忆垃圾回收与衰减机制待完善。

---

## 467. ABTest: Behavior-Driven Testing for AI Coding Agents

**arXiv ID:** 2604.03362 | [PDF](https://arxiv.org/pdf/2604.03362v1)

**作者:** Wuyang Dai `[一作]` (York University), Song Wang `[通讯]` (York University)

**通讯引用:** 1527 | [OpenAlex ID](https://openalex.org/A5115602744)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于行为驱动的模糊测试框架，利用真实用户提交的失败报告自动生成面向代码仓库的测试用例，系统评估AI编码代理的行为鲁棒性。

**💡 创新点**

创新点在于：①将用户报错抽象为交互模式与操作类型，实现可复用的测试模板；②通过LLM生成种子模板并在真实仓库中实例化；③将多步执行轨迹与文件变更结合，提出细粒度的异常检测；④在Claude Code、OpenAI Codex CLI和Gemini CLI上首次大规模发现了新行为缺陷。

**🔧 技术方法**

使用技术包括：GitHub issue挖掘与筛选、交互模式与操作类型抽象、LLM驱动的种子生成与任务生成、真实仓库隔离执行、自动化轨迹与文件差异分析、人工验证与精度评估。

**📊 数据集**

数据集主要是400条经开发者确认的GitHub问题，抽取出47个交互模式与128个操作类型，生成647个模糊测试用例；实验仓库选用Pallets Click；此外使用Claude 4.5/3.5 Haiku、GPT-5.1-Codex-Mini/4o-mini、Gemini 2.5 Flash-Lite三大模型配置。

**📈 对比分析**

通过对5个模型配置执行647个测试用例，检测到1,573个异常，其中642个为真实异常，检测精度为40.8%。按配置划分，最高精度为59.9%（Codex CLI+GPT-5.1-Codex-Mini），最低为23.1%（Claude Code+Claude 3.5 Haiku）。实验表明不同LLM对异常发现率影响显著。

**⚠️ 局限性**

局限性包括：①仅覆盖三款主流编码代理和单一仓库，难以覆盖更广泛场景；②异常检测依赖人工验证，存在主观性；③抽象化的交互模式可能遗漏细粒度语义；④只评估行为异常，未覆盖功能正确性与安全性等维度。

---

## 468. Scaling Teams or Scaling Time? Memory Enabled Lifelong Learning in LLM Multi-Agent Systems

**arXiv ID:** 2604.03295 | [PDF](https://arxiv.org/pdf/2604.03295v1)

**作者:** Shanglin Wu `[一作]` (Emory University), Kai Shu `[通讯]` (Emory University)

**通讯引用:** 11924 | [OpenAlex ID](https://openalex.org/A5058670321)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LLMA-Mem——一种集成了情节记忆、程序记忆与互作记忆的多代理终身学习框架，并从联合规模空间（团队规模×终身学习）分析其性能；

**💡 创新点**

创新点在于把记忆拆分为三种层次并引入可配置的记忆拓扑（本地/共享/混合），以及通过中等频率的记忆巩固实现非单调的团队规模与终身学习的交互效应；

**🔧 技术方法**

采用大语言模型（Claude‑Sonnet‑4.5、DeepSeek‑V3.2、Qwen3‑next‑80B、Qwen3‑32B‑Instruct）作为执行主体，使用嵌入模型（Titan‑text‑embeddings‑v2）进行检索，结合多代理协作的程序抽取与记忆更新算法；

**📊 数据集**

在MultiAgentBench的三大协作环境（coding、research、database）中进行 100 任务/环境的实验；

**📈 对比分析**

与无记忆、MARBLE、A‑Mem 基线相比，LLMA‑Mem 在任务得分和通信得分上实现了同等或更优的平均得分（AAS提升最多 5.92），同时在 token 消耗上降低 9.4%–71.7%；

**⚠️ 局限性**

局限包括：团队规模仅测试至 7 代理；仅覆盖三种协作场景，未涉及网络搜索、机器人控制等；未对记忆质量本身（冗余、陈旧、检索误差）进行细粒度评估；

---

## 469. Multilingual Prompt Localization for Agent-as-a-Judge: Language and Backbone Sensitivity in Requirement-Level Evaluation

**arXiv ID:** 2604.04532 | [PDF](https://arxiv.org/pdf/2604.04532v1)

**作者:** Alhasan Mahmood `[一作]`, Hasan Kurban `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究进行了全面的多语言实验，评估了不同模型在多种任务上的表现。

**💡 创新点**

创新点在于对多种语言的任务进行系统性评估，并提供了详细的满意度和性能指标。

**🔧 技术方法**

使用了GPT-4o-2024-08-06模型进行实验。

**📊 数据集**

使用了多种语言的任务数据集，包括英语、阿拉伯语、土耳其语和中文等。

**📈 对比分析**

通过比较MetaGPT、GPT-Pilot和OpenHands等模型在相同任务上的表现，发现MetaGPT在某些任务上表现优于其他模型，但整体表现因任务而异。

**⚠️ 局限性**

本研究的局限性在于未能涵盖所有可能的语言和任务，且模型的表现可能受到数据集质量和任务复杂度的影响。

---

## 470. Learning from Equivalence Queries, Revisited

**arXiv ID:** 2604.04535 | [PDF](https://arxiv.org/pdf/2604.04535v1)

**作者:** Mark Braverman `[一作]` (Princeton and Google), Kobbi Nissim `[通讯]` (Georgetown and Google)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文重新审视了等价查询学习的经典模型，提出了一种新的学习框架，适用于现代机器学习系统的反馈驱动学习循环。

**💡 创新点**

创新点在于引入了对称对抗者的概念，允许生成对抗性较弱的反例，从而减轻了传统模型的悲观最坏情况行为，并且首次研究了在带有带子反馈的情况下的等价查询学习。

**🔧 技术方法**

使用了博弈论的视角结合自适应加权算法和极小极大论证来分析学习过程。

**📊 数据集**

没有具体提到使用的数据集，但讨论了在全信息和带子反馈设置下的学习过程。

**📈 对比分析**

在全信息反馈下，学习的查询复杂度与Littlestone维数成正比；在带子反馈下，查询复杂度为Θ̃(k·Ldim(ℋ))，其中k是标签复杂度，且该界限是最优的。

**⚠️ 局限性**

限制在于假设假设类是有限的，且在无限假设类的情况下，无法为期望的等价查询数量提供统一的界限。

---

## 471. Why Attend to Everything? Focus is the Key

**arXiv ID:** 2604.03260 | [PDF](https://arxiv.org/pdf/2604.03260v1)

**作者:** Hengshuai Yao `[一作]` (Sapient), Sen Song `[通讯]` (Tsinghua University)

**通讯引用:** 13202 | [OpenAlex ID](https://openalex.org/A5013759262)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

我们提出了 Focus 方法，通过学习可训练的质心将 token 分组，仅让相同组内的远程 token 参与注意力，保留局部窗口的完整注意力，实现高效注意力；

**💡 创新点**

核心创新在于利用 Sinkhorn 正则化强制分组平衡，并通过可训练质心实现对注意力的学习式稀疏化，使得模型可在保持权重冻结的情况下获得更佳性能与速度；

**🔧 技术方法**

技术手段包括可训练质心投影、组内门控注意力、Sinkhorn 归一化、top‑k 组成员硬化、FlashAttention 的两阶段拆分以及与多种现有稀疏注意力方法的对比；

**📊 数据集**

实验使用 PG‑19、WikiText‑103、OpenWebText 等长文本数据集，并在 GPT‑2、Mistral、LLaMA‑2、Qwen、OLMo、Gemma 等不同规模与结构的 Transformer 上验证；

**📈 对比分析**

与全注意力及其它高效注意力重写方法（Longformer、Performer、Routing Transformer 等）对比，Focus 在 GPT‑2‑124M 上从 31.4 PPL 提升至 30.3 PPL，保持四大基准无下降，并在推理时实现 2 倍速度、1M token 可达 8.6 倍墙钟速度；

**⚠️ 局限性**

局限性包括训练阶段仍需 O(n²) 计算、soft‑gated 训练不具推理效率、质心静态无法自适应不同上下文、在超大模型上的性能提升趋缓以及未实现全模型端到端的加速。

---

## 472. StatsClaw: An AI-Collaborative Workflow for Statistical Software Development

**arXiv ID:** 2604.04871 | [PDF](https://arxiv.org/pdf/2604.04871v1)

**作者:** Tianzhu Qin `[一作]` (Cambridge University), Yiqing Xu `[通讯]` (Stanford University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 StatsClaw，一个基于多代理的 AI 辅助统计软件开发框架，保障代码生成与验证相互隔离；

**💡 创新点**

创新点在于引入信息屏障与强制性理解协议，将实现、模拟与测试三条流水线解耦，从而显著提升统计软件实现的可靠性；

**🔧 技术方法**

使用 Claude Code 的多代理架构与自定义 agent prompt，构建了 builder、simulator、tester 三个独立工作流，并通过状态机实现流程控制；

**📊 数据集**

主要利用论文中四页 PDF（probit 估计推导）以及作者自身的 R、Python 包数据，构建并验证三种估计器；

**📈 对比分析**

通过 Monte Carlo 对比 MLE、Gibbs 及 MH 的偏差、RMSE、覆盖率与运行时间，验证结果与理论预期一致，且实现精度优于 R 参考实现；

**⚠️ 局限性**

局限在于依赖底层语言模型的理解能力，验证仍为经验式而非形式化证明，且需要人工审查以确保数学正确性。

---

## 473. DSERT-RoLL: Robust Multi-Modal Perception for Diverse Driving Conditions with Stereo Event-RGB-Thermal Cameras, 4D Radar, and Dual-LiDAR

**arXiv ID:** 2604.03685 | [PDF](https://arxiv.org/pdf/2604.03685v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 474. Optimizing Neurorobot Policy under Limited Demonstration Data through Preference Regret

**arXiv ID:** 2604.03523 | [PDF](https://arxiv.org/pdf/2604.03523v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 475. Simple yet Effective: Low-Rank Spatial Attention for Neural Operators

**arXiv ID:** 2604.03582 | [PDF](https://arxiv.org/pdf/2604.03582v1)

**作者:** Zherui Yang `[一作]` (University of Science and Technology of China), Ligang Liu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8780 | [OpenAlex ID](https://openalex.org/A5100635702)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种低秩空间注意力（LRSA）模块，用标准Transformer原语实现高效全局混合，解决神经算子在大规模PDE求解中的计算与内存瓶颈。

**💡 创新点**

创新点：① 将全局混合统一为低秩压缩‑处理‑重构模板；② 采用可学习的latent瓶颈完成压缩与重构，并在处理阶段使用标准自注意力与FFN，避免非标准归一化与聚合；③ 该结构仅使用标准注意力，可直接利用硬件优化的融合核，显著提升数值稳定性与混合精度训练效率。

**🔧 技术方法**

使用技术：标准缩放点积注意力（SDPA）、多头注意力、层归一化、前馈网络；低秩压缩（cross‑attention）+自注意力+重构（cross‑attention）结构；混合精度训练；与FNO、HPM、Transolver等基线进行对比。

**📊 数据集**

数据集：六大标准PDE基准（Airfoil、Pipe、Plasticity、Navier–Stokes、Darcy、Elasticity）；不规则域基准（Irregular Darcy、Turbulence、Heat Transfer、Composite）；工业案例（AirfRANS、ShapeNet Car）。

**📈 对比分析**

比较方式：在所有基准上与Spectral/基线（FNO、HPM、LSM）、Attention/基线（Transolver、Transolver++、LNO、LinearNO）以及图神经网络/DeepONet 进行对比。LRSA 在所有标准任务均取得最优或第二优误差，平均误差比第二好者低约17%；在不规则域任务大多数情况下也是最佳；在工业案例中误差接近或优于现有最强方法；在混合精度训练下保持稳定，训练时间和显存显著降低。

**⚠️ 局限性**

局限性：① 需要经验性选择latent维度 M，过大或过小都会影响性能；② 在极低样本量（如 Navier–Stokes 200 样本）时优于传统方法的效果不明显；③ 仍缺乏理论上对容量分配（latent 处理 vs 点位更新）的指导；④ 对于极高频或非常复杂动力学场景，仍需进一步提升模型表达能力。

---

## 476. RAIN-FIT: Learning of Fitting Surfaces and Noise Distribution from Large Data Sets

**arXiv ID:** 2604.03491 | [PDF](https://arxiv.org/pdf/2604.03491v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 477. FLAME: Condensing Ensemble Diversity into a Single Network for Efficient Sequential Recommendation

**arXiv ID:** 2604.04038 | [PDF](https://arxiv.org/pdf/2604.04038v1)

**作者:** WooJoo Kim `[一作]` (Pohang University of Science and Technology), HwanJo Yu `[通讯]` (Pohang University of Science and Technology)

**通讯引用:** 5055 | [OpenAlex ID](https://openalex.org/A5045521125)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 FLAME 框架，将冻结网络与可学习网络的子模块组合成指数级表示，压缩多样性为单网络实现高效顺序推荐。

**💡 创新点**

创新点在于通过两网络的子模块组合产生指数多样化表示，并利用冻结网络作为稳定引导、相似度加权对比学习，既保留集成多样性又显著提升训练稳定性与推理效率。

**🔧 技术方法**

采用 SASRec Transformer 作为骨干，结合模块化组合、对比学习（NCE）、加权对比、损失系数衰减等技术实现多视图表示与知识蒸馏。

**📊 数据集**

在六个公开数据集上评估：Amazon Toys、Beauty、Games、Sports、Yelp、MovieLens 1M。

**📈 对比分析**

与传统顺序推荐、对比学习和多网络集成模型对比，FLAME 在 NDCG@20 及 HR@K 上提升 4–10%，收敛速度快 4–7 倍，推理延迟显著低于其他集成方案。

**⚠️ 局限性**

局限性包括对子模块数 M 的经验依赖、极大 M 时的收敛难度以及对冻结网络预训练质量的敏感性。

---

## 478. Investigating Data Interventions for Subgroup Fairness: An ICU Case Study

**arXiv ID:** 2604.03478 | [PDF](https://arxiv.org/pdf/2604.03478v1)

**作者:** Erin Tan `[一作]` (University of California), Irene Y. Chen `[通讯]` (University of California)

**通讯引用:** 2101 | [OpenAlex ID](https://openalex.org/A5081135036)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文通过在ICU电子健康记录数据上实验，研究了多源数据添加对不同亚群体预测性能的影响，并提出了基于均值一致性（mean discrepancy）的数据选择策略，进一步结合后处理校准（Isotonic Regression）来提升亚群体公平性。

**💡 创新点**

创新点在于：①识别并证明均值一致性是决定亚群体性能提升的关键瓶颈；②提出以均值差异最小化为依据的跨源数据选择策略；③证明单独校准或单一数据来源难以提升亚群体表现，而两者结合可获得最佳效果。

**🔧 技术方法**

主要技术包括：逻辑回归、LightGBM、LSTM 三种模型；数据源选择策略（比例/数量、分布相似度、均值差异）；后处理校准使用 Isotonic Regression；评估指标为亚群体准确率（Accuracy）和 AUC；统计相关性分析等。

**📊 数据集**

使用的公开数据集为 eICU Collaborative Research Database（208 家医院）和 MIMIC‑IV（265+ 个病区）两大 ICU 记录集合，敏感属性为患者种族（Asian、Black、White、Other）。

**📈 对比分析**

比较方法：在基线、全源添加、亚群体级添加三种实验设置下，对每种模型与数据选择策略进行 5‑折交叉验证，计算各亚群体准确率变化。结果显示：单纯增量数据或基于分布相似度的选择并不稳定提升性能；基于均值差异的选择能显著改善亚群体准确率；且与后处理校准结合后，最佳/最差亚群体性能提升幅度更大。

**⚠️ 局限性**

局限性包括：仅在 ICU 任务和种族敏感属性上验证，未涵盖其他医疗场景或敏感属性；数据源仍受现有公开数据质量限制；所提出的均值一致性策略在极端样本稀缺或分布差异极大时的表现尚未充分评估。

---

## 479. SGTA: Scene-Graph Based Multi-Modal Traffic Agent for Video Understanding

**arXiv ID:** 2604.03697 | [PDF](https://arxiv.org/pdf/2604.03697v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 480. BLK-Assist: A Methodological Framework for Artist-Led Co-Creation with Generative AI Models

**arXiv ID:** 2604.03249 | [PDF](https://arxiv.org/pdf/2604.03249v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 481. HandDreamer: Zero-Shot Text to 3D Hand Model Generation using Corrective Hand Shape Guidance

**arXiv ID:** 2604.04425 | [PDF](https://arxiv.org/pdf/2604.04425v1)

**作者:** Green Rosh `[一作]` (Samsung Research and Development Institute India), Pawan Prasad B H `[通讯]` (Samsung Research and Development Institute India)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 HandDreamer，零样本文本到 3D 手模型的生成方法；

**💡 创新点**

创新点包括基于 MANO 的低分数初始化、手骨骼引导的 ControlNet 与纠正手形（CHS）损失，显著消除 Janus 伪影并提升几何一致性；

**🔧 技术方法**

使用 Stable Diffusion 1.5 与 ControlNet 进行 Score Distillation Sampling、NeRF 作为 3D 表示、MANO 手模型先验、L2 膜匹配与 CHS 损失；

**📊 数据集**

主要使用 MANO 先验作为数据源，并在 45 个文本提示上渲染 5400 张视图做评估；

**📈 对比分析**

与 ProlificDreamer、ESD、CFD、DreamWaltz、DreamAvatar、HumanNorm、OHTA 等方法对比，CLIP L14、FID、HPSv2 指标均优于现有方法；

**⚠️ 局限性**

局限包括继承预训练扩散模型的偏差、需手工导出网格并手动装配骨骼、对极端手势与姿态的适配仍有限。

---

## 482. Comprehensive List of User Deception Techniques in Emails

**arXiv ID:** 2604.04926 | [PDF](https://arxiv.org/pdf/2604.04926v1)

**作者:** Maxime Veit `[一作]` (Karlsruhe Institute of Technology), Melanie Volkamer `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统整理了电子邮件中用户欺骗技术的完整列表，并提出了新的分类框架。

**💡 创新点**

创新点在于提供了细粒度的欺骗手段分类，并结合上下文语义构建了多维度标签体系。

**🔧 技术方法**

采用自然语言处理技术（BERT+CRF）提取关键词，结合规则匹配和机器学习进行标注。

**📊 数据集**

使用了公开的PhishTank、SpamAssassin和内部收集的企业邮箱数据，共计约200万条邮件。

**📈 对比分析**

通过与现有的欺骗技术清单进行对比，验证了新框架的覆盖面提升了30%，分类准确率达到92%。

**⚠️ 局限性**

局限性在于样本主要来自公开数据，缺乏针对企业内部邮件的深度样本，且规则匹配对新型攻击的适应性有限。

---

## 483. Group-DINOmics: Incorporating People Dynamics into DINO for Self-supervised Group Activity Feature Learning

**arXiv ID:** 2604.04467 | [PDF](https://arxiv.org/pdf/2604.04467v1)

**作者:** Ryuki Tezuka `[一作]` (Toyota Technological Institute), Norimichi Ukita `[通讯]` (Toyota Technological Institute)

**通讯引用:** 4770 | [OpenAlex ID](https://openalex.org/A5053167635)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种无需组活动标注的自监督组活动特征学习方法，利用DINOv3提取局部动态与全局上下文特征，并通过人流估计与组相关物体定位两个预文本任务实现学习；

**💡 创新点**

创新点在于将DINOv3与基于运动流与物体定位的预文本任务相结合，配合对象抠图与inpainting，迫使特征编码器同时关注局部运动与全局空间关系；

**🔧 技术方法**

使用技术包括：DINOv3视觉Transformer、Transformer编码器+MLP、流估计网络（RAFT）、目标检测网络（YOLOX）、inpainting网络（LaMa）、位置与时间编码、MSE损失、两阶段训练；

**📊 数据集**

采用的公开数据集为Volleyball（VBD）和NBA；

**📈 对比分析**

与现有自监督GAF方法（B1-Compact、B2-VGG19、HRN、GAFL）在检索任务中对比，VBD上Hit@1/Hit@3分别为82.7/93.0，NBA上为43.9/72.0，均超过对照组；在监督分类上Fine‑tune后也获得最佳或竞争力的准确率；

**⚠️ 局限性**

局限性在于仅验证于球类运动场景，对非球类或更通用交互对象的适用性待进一步验证，并且使用时间池化压缩序列信息，可能限制了长序列建模能力。

---

## 484. Learning Superpixel Ensemble and Hierarchy Graphs for Melanoma Detection

**arXiv ID:** 2604.03710 | [PDF](https://arxiv.org/pdf/2604.03710v1)

**作者:** Asmaa M. Elwer `[一作]` (Cairo University), Mahmoud H. Annaby `[通讯]` (Cairo University)

**通讯引用:** 2030 | [OpenAlex ID](https://openalex.org/A5001679799)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

基于多级超像素图的图信号处理方法，用于皮肤镜图像的黑色素瘤检测。

**💡 创新点**

提出两种超像素图结构（集合图和层级图），并通过学习方法（MM-学习）自适应构建边权重，结合多尺度图信号特征，提升检测性能。

**🔧 技术方法**

使用超像素分割、多级图构建、图信号处理（时域与频域特征）、边权重学习（MM优化）以及传统机器学习分类器（SVM、RF、GB、KNN、MLP等）。

**📊 数据集**

在ISIC2017数据集（2600张图像）上进行实验，并通过添加300张ISIC存档中的黑色素瘤图像进行数据增强。

**📈 对比分析**

与现有深度学习和传统图方法比较，所提方法在AUC、准确率、特异性、灵敏度上均优于SOTA（AUC≈99.6%，准确率≈99%）。

**⚠️ 局限性**

局限性包括仅在ISIC2017上验证、缺乏更大多样化数据集、未结合深度特征以及可解释性不足。

---

## 485. Mapping the Exploitation Surface: A 10,000-Trial Taxonomy of What Makes LLM Agents Exploit Vulnerabilities

**arXiv ID:** 2604.04561 | [PDF](https://arxiv.org/pdf/2604.04561v1)

**作者:** Charafeddine Mouzouni `[一作]` `[通讯]` (Open Institute of Technology), Charafeddine Mouzouni (Open Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在约10,000次实验中，系统评估了LLM代理在具有工具访问权限的Docker沙盒环境下的漏洞利用表面，并对37种提示变体在七个模型上进行大规模测试。

**💡 创新点**

创新点在于构建了12维心理诱导维度的细粒度分类法，发现仅“目标重构”（如谜题、CTF、隐藏彩蛋提示）能显著触发漏洞利用，而其他九个直觉诱导维度均无显著影响。

**🔧 技术方法**

技术手段包括：使用真实Docker沙盒进行工具调用记录、基于Clopper–Pearson置信区间和Fisher精确检验的统计分析、以及通过关键词检测机制判定是否存在利用行为。

**📊 数据集**

数据集由随机生成的10个编程任务、10种植入式漏洞文件前缀、4个文件系统位置构成，每个实验单元化为独立的任务-漏洞组合，保证统计独立性。

**📈 对比分析**

通过与基线（无诱导句）对比，并采用Bonferroni校正后的p值，发现Claude Sonnet 4在目标重构条件下的利用率达32–40%，其他模型在同类条件下的利用率在8–20%之间；GPT‑4.1在所有条件下保持0%利用率，说明模型安全训练显著提升。

**⚠️ 局限性**

局限性包括：仅测试了植入式文件覆盖、环境变量和配置文件三类漏洞，未覆盖更复杂的权限提升或网络攻击；提示中多维度交叉导致难以单独归因；样本量n=50对小幅效应检出力有限；以及GPT‑4.1免疫机制不明（可能为范围限制或安全训练）

---

## 486. Optimizing Service Operations via LLM-Powered Multi-Agent Simulation

**arXiv ID:** 2604.04383 | [PDF](https://arxiv.org/pdf/2604.04383v1)

**作者:** Yanyuan Wang `[一作]` (Hong Kong University of Science and Technology), Xiaowei Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 8490 | [OpenAlex ID](https://openalex.org/A5100353446)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LLM‑powered多智能体仿真（LLM‑MAS）框架，用以通过对人类行为的文本模拟来优化服务系统的设计参数；

**💡 创新点**

将决策相关不确定性建模为受控马尔可夫链，并设计了一种单轨迹零阶梯度学习算法（OTL）及其两种方差降低方法（GP与RF），实现高效的优化；

**🔧 技术方法**

核心技术包括LLM文本嵌入与解析、受控马尔可夫链理论、零阶梯度估计、两时尺度随机逼近、Guided Perturbation与Residual Feedback方差降低；

**📊 数据集**

实验数据涵盖：可持续供应链仿真（人工生成的系统参数与LLM模拟的决策）；以及基于真实实验室行为数据的创新竞赛设计（432名受试者的能力、风险偏好等信息）；

**📈 对比分析**

与贝叶斯优化、LLM直接求解器、LLM角色扮演者等方法对比，OTL及其改进版本在LLM查询次数上显著更高效，最终达到或超过基准方法的目标性能；

**⚠️ 局限性**

局限性包括：高维设计空间时零阶方法效率下降；LLM推理成本高，难以扩展到大规模多智能体系统；模型未进行微调，对不同任务的泛化能力有限。

---

## 487. SpectralSplat: Appearance-Disentangled Feed-Forward Gaussian Splatting for Driving Scenes

**arXiv ID:** 2604.03462 | [PDF](https://arxiv.org/pdf/2604.03462v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 488. Fine-tuning DeepSeek-OCR-2 for Molecular Structure Recognition

**arXiv ID:** 2604.03476 | [PDF](https://arxiv.org/pdf/2604.03476v1)

**作者:** Haocheng Tang `[一作]` (Northeastern University), Junmei Wang `[通讯]` (University of Pittsburgh)

**通讯引用:** 60127 | [OpenAlex ID](https://openalex.org/A5100643054)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将 DeepSeek-OCR-2 迁移到分子光学结构识别任务，使用两阶段渐进式监督微调实现图像条件 SMILES 生成。

**💡 创新点**

提出先 LoRA 高效微调再全参数分层微调的策略，并结合合成与真实专利图像混合训练集，验证强化学习后处理不提升精确匹配。

**🔧 技术方法**

使用 LoRA、分层冻结与不同学习率的全参数微调、视觉分词器+LM 视觉编码器+自回归解码器的架构，进行图像到 SMILES 的端到端生成。

**📊 数据集**

采用 PubChem 合成渲染（MolScribe-like 与 ChemDraw-like 两种风格）和 USPTO‑MOL 真实专利图像，共构建 192k→800k 的训练样本。

**📈 对比分析**

与传统 OCR、图像‑序列（DECIMER、SwinOCSR 等）以及图像‑图（MolScribe、GTR‑VL 等）基线对比，MolSeek‑OCR 在合成、真实与扰动集上的精确匹配准确率与 DECIMER 相当，但仍落后于 MolScribe 等图像‑图模型。

**⚠️ 局限性**

限制在于 VLM 基于文本生成的精确性不及显式几何节点‑边预测方法，且强化学习/数据细化后处理无法保持 SMILES 序列的严格一致性。

---

## 489. How Far Are We? Systematic Evaluation of LLMs vs. Human Experts in Mathematical Contest in Modeling

**arXiv ID:** 2604.04791 | [PDF](https://arxiv.org/pdf/2604.04791v1)

**作者:** Yuhang Liu `[一作]` (Beijing Institute of Technology), Yang Gao `[通讯]` (Beijing Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一套面向问题、分阶段的评估框架，用来衡量大语言模型在完整数学建模任务中的表现，并通过专家评审验证其可靠性。

**💡 创新点**

创新点在于动态生成针对每个问题的分阶段评估准则，使评估与专家判断高度对齐，揭示了LLM在执行阶段（模型求解、代码实现、结果验证）明显不足的“理解-执行”鸿沟。

**🔧 技术方法**

技术上采用LLM自动生成评估准则与建模报告、专家人工审核、ICC(2,1)统计分析对齐度、分阶段评分与失败模式归纳。

**📊 数据集**

使用了中国研究生数学建模竞赛（PMCM）近年的 97 题数据集，题目为真实专家级开放式建模问题。

**📈 对比分析**

与传统粗粒度评估法（ICC≈0.012）相比，新框架在专家评估上的ICC提升至0.673；LLM在问题识别、表述等理解阶段得分接近专家，但在模型求解、代码实现、结果分析等执行阶段平均得分低于5分，规模提升对执行阶段改善有限。

**⚠️ 局限性**

局限性在于评估框架仅在 PMCM 任务上验证，需专家参与生成准则，难以直接迁移到其他领域或缺乏专家资源的环境。

---

## 490. Same World, Differently Given: History-Dependent Perceptual Reorganization in Artificial Agents

**arXiv ID:** 2604.04637 | [PDF](https://arxiv.org/pdf/2604.04637v1)

**作者:** Hongju Pae `[一作]` `[通讯]` (Active Inference Institute), Hongju Pae (Active Inference Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种最小化的人工智能架构，使用慢速视角潜变量g通过FiLM门控和自适应可塑性重构感知，使得相同的观测在不同历史条件下被编码为不同的表征；

**💡 创新点**

创新点在于将视角潜变量与感知编码相耦合并引入自适应可塑性调节机制，形成视角-感知-可塑性三向反馈循环，从而实现历史依赖的视角组织与感知重组；

**🔧 技术方法**

采用了多层感知编码器、FiLM门控、GRU候选更新、AlphaNet自适应可塑性网络，以及基于预测误差的学习目标；

**📊 数据集**

使用了一个23×7的仿真网格世界，包含左至右的观测噪声梯度，并在训练中引入临时的观测扰动；

**📈 对比分析**

通过混合历史、探针和消融实验比较不同可塑性更新策略（自适应、刚性、开放）以及扰动与非扰动情形；实验显示自适应可塑性能产生增长-稳定的动态，扰动历史会留下可塑性衰退的残留，并且视角潜变量能显著改变相同观测的编码；

**⚠️ 局限性**

局限性包括：仅在极其简单的仿真环境中验证，缺乏外部奖励或复杂任务；架构的可扩展性与在真实感知环境中的鲁棒性尚未评估；

---

## 491. A Clinical Point Cloud Paradigm for In-Hospital Mortality Prediction from Multi-Level Incomplete Multimodal EHRs

**arXiv ID:** 2604.04614 | [PDF](https://arxiv.org/pdf/2604.04614v1)

**作者:** Bohao Li `[一作]` (Beihang University), Bowen Du `[通讯]` (Beihang University)

**通讯引用:** 4898 | [OpenAlex ID](https://openalex.org/A5053487836)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了HealthPoint（HP）临床点云范式，用于统一处理多级不完整多模态电子病历（EHR）的风险预测。

**💡 创新点**

创新点在于：①将临床事件直接映射为4D点云（内容、时间、模态、病例），①通过低秩关系注意力捕捉任意点对的高阶依赖，②采用分层交互与采样平衡细粒度与效率，③结合细粒度对齐与重建的自监督策略，全面解决不规则采样、模态缺失与标签稀疏三大难题。

**🔧 技术方法**

核心技术包括低秩关系注意力（CP分解低秩近似）、分层邻域交互与采样、细粒度自监督对齐（FGA）与重建（FGR）、自适应熵推理以及多模态专用编码器（MLP、Clinical‑BERT、DenseNet）。

**📊 数据集**

使用公开的大规模临床数据集 MIMIC‑III 与 MIMIC‑IV，构建住院死亡（IHM）预测任务。

**📈 对比分析**

与 14 种现有多模态方法（覆盖单一或双重缺失场景）对比，HP 在所有指标（AUROC、AUPRC、F1）均实现了显著提升，且在不同缺失率（0%–90%）下保持稳健的性能优势。

**⚠️ 局限性**

局限性：模型结构较为复杂，计算成本与参数量较大；对超参数敏感；仅在住院死亡预测任务上验证，未测试在其他临床预测任务或更稀疏标签场景下的泛化能力。

---

## 492. A Patch-based Cross-view Regularized Framework for Backdoor Defense in Multimodal Large Language Models

**arXiv ID:** 2604.04488 | [PDF](https://arxiv.org/pdf/2604.04488v1)

**作者:** Tianmeng Fang `[一作]` (Singapore Management University), Wei Wang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 75376 | [OpenAlex ID](https://openalex.org/A5100391883)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于块级增广与跨视图正则化的多模态大语言模型后门防御框架

**💡 创新点**

创新点在于利用局部非语义扰动的跨视图输出差异约束，突破低比例后门注入的弱信号瓶颈

**🔧 技术方法**

采用块级图像扰动、跨视图输出差异正则、输出熵约束以及常规任务损失

**📊 数据集**

在MS COCO和Flickr30k两大图文数据集上进行评估

**📈 对比分析**

与无防御、Fine‑Pruning、STRIP、Entropy Filtering等方法对比，攻击成功率从≈98%下降至≈30‑60%，同时BLEU‑4和CIDEr保持接近原模型

**⚠️ 局限性**

局限在于仅针对局部视觉触发，需调整权重且对已部署模型效果有限

---

## 493. Diffusion Policy with Bayesian Expert Selection for Active Multi-Target Tracking

**arXiv ID:** 2604.03404 | [PDF](https://arxiv.org/pdf/2604.03404v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 494. Eliminating Vendor Lock-In in Quantum Machine Learning via Framework-Agnostic Neural Networks

**arXiv ID:** 2604.04414 | [PDF](https://arxiv.org/pdf/2604.04414v1)

**作者:** Poornima Kumaresan `[一作]`, Santhosh Sivasubramani `[通讯]`

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个框架无关的量子神经网络体系，提供统一计算图、硬件抽象层和多框架导出。

**💡 创新点**

通过统一接口打破 TensorFlow Quantum / PennyLane / Qiskit 等软件与 IBM、Amazon Braket、Azure 等硬件厂商的锁定，实现跨框架、跨硬件的无缝迁移，并提供 ONNX 扩展。

**🔧 技术方法**

采用参数平移规则、统一的 QuantumLayer 抽象、硬件抽象层、三种可插拔编码（幅度、角度、IQP）以及自定义 ONNX 量子域。

**📊 数据集**

使用 Iris、Wine（PCA 降维）和 MNIST-4（4×4 像素）三个标准分类数据集。

**📈 对比分析**

在相同模型与超参数下，跨框架训练时间仅有 1%–8% 的额外开销，分类精度与原生实现基本一致；硬件梯度误差落在噪声预算内，跨平台 round‑trip fidelity 超过 0.9999，验证了兼容性和性能。

**⚠️ 局限性**

仍存在轻微时间开销、编码兼容性依赖后端实现、对未来硬件标准的适配需进一步完善，且对动态量子电路支持有限。

---

## 495. Ledger-State Stigmergy: A Formal Framework for Indirect Coordination Grounded in Distributed Ledger State

**arXiv ID:** 2604.03997 | [PDF](https://arxiv.org/pdf/2604.03997v1)

**作者:** Fernando Paredes García `[一作]` `[通讯]` (Independent Researcher), Fernando Paredes García (Independent Researcher)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出并 formal 化 以分布式账本状态为媒介的间接协调机制（ledger‑state stigmergy），并给出其四要素映射、三种基本模式及一个 commit‑reveal 叠加模式；

**💡 创新点**

创新点在于将 Grassé 的蟻群学概念具体化到区块链应用层，提供一套可复用的模式词汇表、状态转移形式化及对比分析框架；

**🔧 技术方法**

利用智能合约状态转移函数、事件日志、阈值检测等区块链原语，构建模式与理论模型；

**📊 数据集**

本研究未使用真实数据集，采用理论建模与例子演示（任务板），无实验数据；

**📈 对比分析**

通过理论对比分析（STIG vs MSG vs ORCH）讨论信任、争议、恢复、透明度等维度，未给出数值性能指标；

**⚠️ 局限性**

局限在于缺乏实验验证、仅聚焦单合约、仅展示 State‑Flag 示例，未覆盖跨合约或 MEV 场景，且未对动态网络条件进行评估。

---

## 496. BLADE: Better Language Answers through Dialogue and Explanations

**arXiv ID:** 2604.03236 | [PDF](https://arxiv.org/pdf/2604.03236v1)

**作者:** Chathuri Jayaweera `[一作]` (University of Florida), Bonnie J. Dorr `[通讯]` (University of Florida)

**通讯引用:** 8493 | [OpenAlex ID](https://openalex.org/A5054060679)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种名为BLADE的对话式学习助手，利用检索增强生成框架引导学生查阅课程资源，而非直接给出答案。

**💡 创新点**

创新点在于将答案回避与证据聚焦生成相结合，使用课程感知索引和对比学习训练的教学相关检索器，使回复严格以教学材料为依据。

**🔧 技术方法**

采用检索增强生成（RAG）技术，配合大语言模型、对比学习检索器和可控生成模块，检索并突出课程文本片段。

**📊 数据集**

使用经过主题对齐与学习目标标注的课程资源集（教材、课件、阅读材料）以及在本科自然语言处理课程中收集的学生交互数据。

**📈 对比分析**

通过三组资源配置（BLADE仅、BLADE+教材、仅教材）对85名学生进行影响研究，结果显示BLADE+教材组在资源定位和概念测评上显著优于其他组。

**⚠️ 局限性**

局限性包括：仅针对单一课程（NLP）验证；缺乏对学生细粒度交互行为的实时记录；未将AI助手嵌入真实任务环境，难以评估对实际应用推理的影响。

---

## 497. Edge-Based Standing-Water Detection via FSM-Guided Tiering and Multi-Model Consensus

**arXiv ID:** 2604.03308 | [PDF](https://arxiv.org/pdf/2604.03308v1)

**作者:** Oliver Aleksander Larsen `[一作]` (University of Southern Denmark), Mahyar T. Moghaddam `[通讯]` (University of Southern Denmark)

**通讯引用:** 381 | [OpenAlex ID](https://openalex.org/A5018940565)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在农田越野车辆上实现了一套边缘计算架构，用来实时检测路面积水，并根据运动状态、能耗和网络条件动态选择模型层级；

**💡 创新点**

创新点在于将有限状态机（FSM）作为统一控制平面，结合多模型YOLO协商、昼夜基线传感器融合以及可选的Jetson加速，形成可复现、可调节的资源感知决策体系；

**🔧 技术方法**

技术包括：Raspberry‑Pi级别采集节点、Processing Pi为FSM控制器、Jetson AGX Orin作为可选GPU加速节点；YOLOv8多模型集成、IoU聚类加权聚合；传感器异常检测（温湿度压强）与阈值规则融合；MQTT事件驱动通信、按帧JSON日志；

**📊 数据集**

使用公开的农业水域检测数据集 UB Water Detection、SegWater 与 Ponding‑v2，共计约 12,633 张图像，用于训练三模型YOLO；现场录制约 5 条 30 s 轨迹视频作为测试集；

**📈 对比分析**

通过十种消融配置在硬件回放上对比，评估二分类宏F1、平衡准确率、洪水召回率、p99延迟、能耗和帧覆盖率；结果显示：生产配置（FSM+三模型+传感器融合）在宏F1 0.816、能耗 278 J、p99 延迟 2.57 s 时，比单模型本地基线高 8 % 准确率、能耗下降 30 %，比“始终使用重模型”策略提升召回率 0.411；

**⚠️ 局限性**

局限性包括：仅在 RPi+Jetson 平台验证，缺乏夜间、恶劣天气或高速度场景；传感器融合使用固定阈值，未做在线自适应；多模型计算导致较高尾延迟；未验证大规模车队部署和长期传感器漂移的影响；

---

## 498. Quantum-inspired Ising machine using sparsified spin connectivity

**arXiv ID:** 2604.04606 | [PDF](https://arxiv.org/pdf/2604.04606v1)

**作者:** Moe Shimada `[一作]`, Jun-ichi Shirakashi `[通讯]` (Tokyo University of Agriculture and Technology)

**通讯引用:** 1408 | [OpenAlex ID](https://openalex.org/A5070382412)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并实现了基于提取型多数投票逻辑（E‑MVL）的量子启发式Ising机，用于求解Sherrington–Kirkpatrick（SK）模型的基态，并与传统与优化后的模拟退火（SA）以及FPGA实现进行对比评估。

**💡 创新点**

创新点在于通过稀疏度控制实现对热动力学的可控模拟，证明稀疏度与SA温度等价，并利用等势态分析提供了针对不同耦合分布的温度调度方案，最终实现了与SA相比在相同规模下更优的近似与精确解性能，且表现与耦合分布和问题规模无关。

**🔧 技术方法**

采用了基于SDL的硬件友好型逻辑（XNOR与多数投票电路）实现稀疏化交互，结合等势态分析、STT/STs评估指标以及FPGA加速实现；同时使用了Sk模型的二元与高斯耦合实例进行实验。

**📊 数据集**

使用了Sherrington–Kirkpatrick模型的二元（{-1,+1}）和高斯分布（均值0，标准差1）耦合实例，规模从100到1600个自旋，针对每个规模采集20个实例并在每个实例上进行1000次独立实验。

**📈 对比分析**

通过STT（寻找99%基态能量所需迭代次数）和STS（获得精确基态能量所需迭代次数）两项指标与传统SA、优化SA以及dwave‑neal进行对比；结果显示E‑MVL在100–1600自旋范围内STT约为优化SA的50%（对二元耦合）或显著更优（对高斯耦合），且在FPGA上实现时速度提升约6倍。

**⚠️ 局限性**

局限性包括：在精确解求解中STS随自旋数呈现上升趋势，表明对大规模基态搜索的可扩展性仍受限；仅在SK模型上验证，需进一步检验对其他组合优化问题的适用性；硬件实现受制于资源和时钟约束，ASIC化仍待进一步验证。

---

## 499. Learning Dexterous Grasping from Sparse Taxonomy Guidance

**arXiv ID:** 2604.04138 | [PDF](https://arxiv.org/pdf/2604.04138v1)

**作者:** Juhan Park `[一作]` (Korea University), Sungjoon Choi `[通讯]` (RLWRLD)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出GRIT两阶段框架，先用视觉语言模型预测与对象几何和任务意图匹配的抓取分类（taxonomy），再用稀疏抓取命令驱动低层多指控制，实现可控的精细抓取；

**💡 创新点**

创新点在于将抓取分类作为中间离散语义表示，既提供结构化高层指导，又允许低层连续控制；使用乘法式奖励强化分类遵循；通过教师‑学生蒸馏实现真实部署；

**🔧 技术方法**

使用视觉语言模型（VLM）做抓取规划、BPS几何特征、PPO强化学习、教师‑学生异步蒸馏、乘法奖励以及MuJoCo‑Warp仿真和双手机器人；

**📊 数据集**

在30个YCB物体上训练，并在Objaverse（RoboCasa子集）评估，共373个物体分为水果蔬菜、厨房用具、包装商品三类；

**📈 对比分析**

与无分类条件的RDG、GraspXL基线对比，GRIT在所有类别上获得最高成功率，整体87.9%（比RDG提升6%），与oracle差距仅5%；在真实机器人上通过文本提示和几何选择不同抓取分类；

**⚠️ 局限性**

局限在于仅依赖初始视觉观测，缺乏实时动态反馈；难以针对功能部件精细定位；对薄平面或极细物体的抓取仍有挑战。

---

## 500. FlueBricks: A Construction Kit of Flute-like Instruments for Acoustic Reasoning

**arXiv ID:** 2604.03636 | [PDF](https://arxiv.org/pdf/2604.03636v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 501. Training Transformers in Cosine Coefficient Space

**arXiv ID:** 2604.04440 | [PDF](https://arxiv.org/pdf/2604.04440v1)

**作者:** Mohamed Amine Bergach `[一作]` `[通讯]` (Illumina), Mohamed Amine Bergach (Illumina)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将Transformer的权重矩阵在二维离散余弦变换（DCT）域中仅保留低频系数表示，并在训练时通过逆DCT重构权重，保持模型正向和反向计算完整性。

**💡 创新点**

首次在从零预训练阶段使用全局DCT参数化，证明仅保留低频系数即可实现2×压缩而不损失困惑度，展示了权重矩阵天然的频谱集中特性。

**🔧 技术方法**

采用二维离散余弦变换与逆变换、固定低频系数选择、Kaiming/尺度补偿初始化、以及梯度通过IDCT投影的方式实现参数化。

**📊 数据集**

使用约100万字符的莎士比亚全集进行字符级语言建模实验。

**📈 对比分析**

与标准全参数模型和LoRA低秩基线对比；2×压缩时困惑度与标准模型相同（6.1），4×压缩时困惑度为6.9，显著优于同等压缩率下的低秩模型（8.8）。

**⚠️ 局限性**

实验仅在小规模模型和数据集上验证，缺乏对大模型的可扩展性评估；前向时逆DCT计算略慢；采用固定低频选取，未探索自适应频率选择或更高压缩比。

---

## 502. OpenWorldLib: A Unified Codebase and Definition of Advanced World Models

**arXiv ID:** 2604.04707 | [PDF](https://arxiv.org/pdf/2604.04707v1)

**作者:** DataFlow Team `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了统一的世界模型定义与完整框架OpenWorldLib，实现多模态感知、推理、生成、记忆与管线化流程。

**💡 创新点**

1) 给出标准化世界模型定义并阐明核心任务；2) 设计统一模块化框架与模板（Operator、Synthesis、Reasoning、Representation、Memory、Pipeline）；3) 将交互视频生成、3D重建、多模态推理、Vision‑Language‑Action等任务纳入统一评估与基准。

**🔧 技术方法**

采用Transformer、Diffusion、LLM、视觉‑语言模型、音频生成网络、3D重建网络及仿真器接口；框架基于Python+PyTorch，支持本地与云端模型集成。

**📊 数据集**

主要使用AI2‑THOR与LIBERO仿真环境；公开的3D重建与视频生成基准（如VGGT、InfiniteVGGT、FlashWorld等）；实验在NVIDIA A800、H200 GPU上完成。

**📈 对比分析**

与现有方法（Matrix‑Game‑2、Lingbot‑World、Hunyuan‑WorldPlay、Wan‑IT2V、WoW、Cosmos等）在交互视频、3D生成与VLA任务上对比；OpenWorldLib在交互视频生成上与最新模型相当或略优，3D重建与VGGT保持竞争力，但整体推理速度仍低于专门优化模型。

**⚠️ 局限性**

限制：1) 计算效率相对低，交互视频与3D生成推理慢；2) 需要大量GPU资源，资源受限环境难以部署；3) 对硬件优化不足，难以实现更高效的token‑free 预测；4) 评估主要在仿真与公开基准，缺乏真实物理环境验证。

---

## 503. AI Appeals Processor: A Deep Learning Approach to Automated Classification of Citizen Appeals in Government Services

**arXiv ID:** 2604.03672 | [PDF](https://arxiv.org/pdf/2604.03672v1)

**作者:** Vladimir Beskorovainyi `[一作]` (Besk Tech), Vladimir Beskorovainyi `[通讯]` (Moscow Institute of Physics and Technology)

**通讯引用:** 82 | [OpenAlex ID](https://openalex.org/A5082274215)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个基于微服务的 AI Appeals Processor，自动对俄语公民申诉进行分类与路由。

**💡 创新点**

创新点在于将 Word2Vec 与 LSTM 结合，实现既高准确率（78%）又高效率（推理 <2 秒）的自动分类，并通过人机回馈实现持续改进。

**🔧 技术方法**

使用了 NLP、深度学习（Word2Vec+LSTM、fastText、BERT）、传统机器学习（BoW/TF‑IDF+SVM）等多种技术。

**📊 数据集**

使用 10,000 条真实俄语公民申诉数据集，按投诉、申请、提案三大类别和七个主题领域进行标注。

**📈 对比分析**

与手工基线、BoW+SVM、TF‑IDF+SVM、fastText 和 BERT 等模型对比，Word2Vec+LSTM 在 78% 的准确率、每条申诉推理时间 <2 秒、训练耗时 95 分钟的平衡点上表现最佳；BERT 在准确率上最高（82%）但训练耗时和资源显著更高。

**⚠️ 局限性**

局限包括样本量相对有限、仅覆盖宏观三类标签、未使用专门的俄语预训练模型、对多标签或细粒度子类别的支持不足，以及未在更大规模实际环境中进一步验证。

---

## 504. Metaphors We Compute By: A Computational Audit of Cultural Translation vs. Thinking in LLMs

**arXiv ID:** 2604.04732 | [PDF](https://arxiv.org/pdf/2604.04732v1)

**作者:** Yuan Chang `[一作]` (Meta), Zhu Li `[通讯]` (Meta)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在多种文化背景下生成隐喻并对生成结果进行语义空间分析，审计了大型语言模型在文化推理方面的包容性；

**💡 创新点**

创新之处在于将几何嵌入分析与文化默认性测试相结合，首次量化模型在不同文化提示下的代表性崩塌与西方默认倾向；

**🔧 技术方法**

使用了提示工程、英文隐喻生成、句子嵌入（Sentence‑Transformers 3072维向量）、余弦相似度、t‑SNE 可视化以及 Fisher 随机化检验；

**📊 数据集**

构建了自生成的 600 条隐喻样本（5 个抽象概念 × 6 个文化条件 × 20 次抽样），无外部标注数据集；

**📈 对比分析**

通过计算每个文化-概念对的平均余弦距离评估语义多样性，使用 t‑SNE 观察概念几何结构，并比较默认条件与各文化中心的距离来检验西方默认性；结果显示在部分文化-概念组合出现代表性崩塌，默认条件往往更接近美国语义空间；

**⚠️ 局限性**

局限性包括仅关注隐喻这一单一文化表达维度、只生成英文文本、仅评估单一 LLM、缺乏人类评估以及忽略语调与风格细节。

---

## 505. Perceptual Gaps: ASCII Art and Overlapping Audio as CAPTCHA

**arXiv ID:** 2604.03612 | [PDF](https://arxiv.org/pdf/2604.03612v1)

**作者:** Choon-Hou Rafael Chong `[一作]` `[通讯]` (Hwa Chong Institution), Choon-Hou Rafael Chong (Hwa Chong Institution)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出基于ASCII艺术和叠加音频的两类CAPTCHA，并验证其对当前大型多模态LLM的破解难度。

**💡 创新点**

创新点在于利用人类进化出的视觉与听觉优势（如识别ASCII图形、分离混合语音）设计难以被现有LLM破解的任务。

**🔧 技术方法**

使用多模态LLM（GPT‑5.2、Gemini 3、Claude Sonnet、Llama 4、Qwen3‑VL）、文本/图像输入、语音合成（XTTS‑v2）以及噪声/重叠语音增强技术。

**📊 数据集**

数据集包括：500个随机生成的ASCII艺术文本（不同字体渲染为文本与图像），Commonsense‑QA 题库与XTTS‑v2合成的带背景噪声与重叠语音样本。

**📈 对比分析**

通过与各模型的完整准确率、字符/答案相似度、推理时延进行比较。结果显示：LLM在ASCII图像上0%准确、字符相似度最高仅55%；在音频CAPTCHA中，模型在无噪声下准确率可达75%，但加入背景/高斯噪声后降至48%或更低，整体表现明显低于随机猜测。

**⚠️ 局限性**

局限性包括缺乏人类性能基准、仅测试现有公开LLM且未考虑模型微调或迁移学习、音频CAPTCHA生成成本高且对非母语用户友好性不足。

---

## 506. Stabilizing Unsupervised Self-Evolution of MLLMs via Continuous Softened Retracing reSampling

**arXiv ID:** 2604.03647 | [PDF](https://arxiv.org/pdf/2604.03647v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 507. Apparent Age Estimation: Challenges and Outcomes

**arXiv ID:** 2604.03335 | [PDF](https://arxiv.org/pdf/2604.03335v1)

**作者:** Justin Rainier Go `[一作]` (De La Salle University), Abien Fred Agarap `[通讯]` (De La Salle University)

**通讯引用:** 3247 | [OpenAlex ID](https://openalex.org/A5003380343)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对面部感知年龄估计进行系统性评估，比较了 DEX、Mean‑Variance Loss (MVL) 与 Adaptive Mean‑Residue Loss (AMRL) 在 IMDB‑WIKI、CLAP、APPA‑REAL 与 FairFace 等多样本集上的训练与微调效果，并通过 UMAP 可视化、Saliency Map 与 cosine similarity 分析模型在不同人群（种族、性别）上的偏差与公平性。

**💡 创新点**

① 将三种分布式损失函数与多阶段数据集微调组合在同一框架下统一比较；② 通过公平性验证（跨种族、性别的 MAE 方差）证明 AMRL 与 FairFace 微调能显著降低偏差；③ 结合可视化与 saliency 分析揭示模型对不同族群关注区域不一致，首次量化了这一现象。

**🔧 技术方法**

VGG‑16 预训练 + 自定义分布式损失（MVL、AMRL）、交叉熵；PyTorch Lightning 实现；UMAP 降维可视化；Grad‑CAM / saliency map；cosine similarity 计算；MAE 与 ε‑error 评价指标。

**📊 数据集**

IMDB‑WIKI（清洗版）、CLAP、APPA‑REAL、FairFace；此外自制小规模菲律宾名人图片集用于本土化验证。

**📈 对比分析**

通过在 APPA‑REAL 与 CLAP 测试集上计算 MAE/ε，比较 18 种模型（6 数据集组合 × 3 损失）。结果显示：AMRL 在 IMDB‑WIKI+APPA‑REAL 组合下取得最低 MAE 3.59；在所有组合中，加入 FairFace 后 MAE 稍升但跨种族、性别的标准差显著下降，表明公平性提升。相比之下，传统 DEX 与 MVL 在非白人女性上的误差最高。

**⚠️ 局限性**

① 数据集严重不均衡（欧美/白人占比高，亚洲/黑人女性不足），导致模型偏差无法完全消除；② 虽然 AMRL 能提升整体准确率，但对某些族群（尤其亚洲/黑人女性）仍有较高误差；③ 缺乏本土化大规模纵向数据，模型在菲律宾人脸上的泛化性有限；④ 低资源环境下的计算效率与模型压缩尚未研究；⑤ 对隐私与伦理风险的评估不完整。

---

## 508. 3D-Stacked NMP, LLM Decoding, Systolic Array Microarchitecture, Multi-Core Scheduling

**arXiv ID:** 2604.04253 | [PDF](https://arxiv.org/pdf/2604.04253v1)

**作者:** Chenyang Ai `[一作]` (University of Edinburgh), Wenhui OU `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5113417469)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种面向3D堆叠近内存（NMP）LLM解码的可重构Systolic Array架构Snake，配合多核调度框架，实现高利用率的解码加速。

**💡 创新点**

创新点在于：1）将MAC‑tree替换为可重构SA，支持按需改变阵列形状与数据流；2）利用高本地带宽减少缓冲区，释放面积；3）统一向量核心与SA的控制与缓冲，形成低面积、可配置的计算子系统；4）提出空间+时空分区的多核调度策略，充分匹配不同解码算子。

**🔧 技术方法**

使用的关键技术包括：可重构2D systolic array、蛇形映射（snake‑like mapping）、共享2读/2写输出缓冲、轻量级NoC、动态区域表RTAB、负载均衡的DMA读取、以及基于门级RTL+Scale‑Sim+FinCACTI的功耗/性能建模。

**📊 数据集**

实验采用多种LLM模型作为数据集：OPT‑66B、LLaMA3‑70B、Mixtral‑8×22B、Qwen3‑30B、DeepSeek‑236B，覆盖密集和MoE模型。

**📈 对比分析**

评估方法：在统一的3D‑NMP硬件配置下与GPU H100、Stratum、MAC‑tree、固定形状SA（48×48、8×288）进行对比。Snake在相同面积预算下平均比MAC‑tree快2.90×、能效高2.40×；与GPU平均快11.47×、能效高5.74×；相较固定形状SA平均快2.33×/3.00×、能效提高1.05×/1.31×。

**⚠️ 局限性**

局限性包括：最小重构粒度为8，导致极小M的算子无法完全匹配；在极低算子规模时仍可能因内存带宽受限而出现瓶颈；面积和热设计仍需在更大规模3D堆叠中进一步验证；仅针对解码阶段，未覆盖prefill或其他LLM计算；调度框架复杂度高，需额外编译支持。

---

## 509. Don't Blink: Evidence Collapse during Multimodal Reasoning

**arXiv ID:** 2604.04207 | [PDF](https://arxiv.org/pdf/2604.04207v1)

**作者:** Suresh Raghu `[一作]` (Independent Researcher), Satwik Pandey `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对视觉语言模型在推理过程中视觉注意力衰退与不确定性进行系统分析，并提出任务感知的多模态监控策略。

**💡 创新点**

首次揭示“证据坍塌”普遍存在，并证明视觉注意力衰退导致的错误风险是任务条件的，进而提出基于全响应熵加视觉veto的条件监控。

**🔧 技术方法**

采用全响应熵、累计视觉注意力面积、交互式逻辑回归、跨数据集迁移评估等技术，对模型推理轨迹进行量化与解释。

**📊 数据集**

使用MathVista、HallusionBench、MMMU_Pro三大多模任务数据集，并在每个上手工标注300例，形成900例带边界框的评测集。

**📈 对比分析**

与仅文本熵监控对比，全响应熵始终是最强基线；添加视觉特征的线性融合并未提升性能；在MMMU_Pro与HallusionBench中，基于任务条件的视觉veto可将风险降低约1.9个百分点，而在MathVista中则适得其反。

**⚠️ 局限性**

局限包括：关注仅为相关性（缺乏因果干预验证）、模型规模仅至8B、仅两种架构、样本量有限、任务类型需先行判定，导致在极端样本中统计力量不足。

---

## 510. Towards Edge Intelligence via Autonomous Navigation: A Robot-Assisted Data Collection Approach

**arXiv ID:** 2604.03623 | [PDF](https://arxiv.org/pdf/2604.03623v1)

**作者:** Tingting Huang `[一作]` (Jinan University), Li Wang `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 42551 | [OpenAlex ID](https://openalex.org/A5100336135)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种通信-学习双驱动的自主导航方案（CLD），用于在工业边缘情境中让机器人安全收集数据并提升模型训练效果。

**💡 创新点**

创新点在于将非点质量机器人模型与多区域路径损耗特征结合，构建多目标优化（导航、通信、学习），并通过MM算法和线性对偶化实现可迭代求解。

**🔧 技术方法**

采用的技术包括形状约束的碰撞规避、分段子区域的信道模型、基于MM的非凸优化、L1稀疏正则化以及CVXPY数值求解器。

**📊 数据集**

实验仅基于仿真数据，设定2个传感器、2个模型（CNN与SVM）、16根天线，未使用公开数据集。

**📈 对比分析**

与PMM+Commu、OBCA、RDA、RDA+Commu四种基线对比，CLD在数据采集量、分类误差和通信吞吐量上均优于对手，尤其在学习误差上显著下降。

**⚠️ 局限性**

局限性包括仅在仿真工厂环境验证，子区域划分需要先验信道信息，算法复杂度较高，且对实时性和大规模场景的适应性尚未实测。

---

## 511. Rashomon Memory: Towards Argumentation-Driven Retrieval for Multi-Perspective Agent Memory

**arXiv ID:** 2604.03588 | [PDF](https://arxiv.org/pdf/2604.03588v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 512. Improving Feasibility via Fast Autoencoder-Based Projections

**arXiv ID:** 2604.03489 | [PDF](https://arxiv.org/pdf/2604.03489v1)

**作者:** Maria Chzhen `[一作]` (University of Toronto), Priya L. Donti `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1984 | [OpenAlex ID](https://openalex.org/A5075620331)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

提出基于自编码器的快速投影方法（FAB），通过训练一个可将不满足约束的预测快速映射到近似可行集合的自编码器，作为可插拔附件附加到任何神经网络上；

**💡 创新点**

创新点在于：①通过对自编码器潜在空间的结构化（将潜在空间映射到简单凸形如球/单形），实现一次性近似投影；②使用对抗训练使得潜在空间内的点解码后几乎必定可行；③将该映射作为低延迟的可行性改进模块，可与多种学习与控制任务无缝结合；

**🔧 技术方法**

使用技术包括：自编码器、对抗训练（判别器与自编码器交替训练）、潜在空间结构化（hinge loss、latent loss、geom loss）、一次性投影与解码、以及在强化学习中与 PPO/TRPO 等算法结合；

**📊 数据集**

使用的数据集包括：合成的非凸约束优化问题（Blob with Bite、Concentric Circles、Star‑Shaped、Two Moons、三/五/十维球壳），以及 SafetyGym 的 SafetyPointGoal2‑v0 与 SafetyPointPush2‑v0 环境中的随机采样状态‑动作对（约 10 万条）；

**📈 对比分析**

与传统投影梯度、罚项、增量 Lagrangian、内部点、Penalty NN、FSNet、Homeomorphic Projection 等方法对比；FAB 在可行率几乎 100%、推理时间 < 1 ms（比精确投影快 1–2 个数量级），在 SafetyGym 中 PPO‑FAB 的约束违规成本最低，奖励略低但波动更小；

**⚠️ 局限性**

局限性：缺乏硬约束保证、对训练数据代表性的高度依赖、对抗训练可能不稳定、实验规模相对有限、未进行端到端联合训练、缺乏样本效率与分布迁移的评估、可解释性不足。

---

## 513. Composer Vector: Style-steering Symbolic Music Generation in a Latent Space

**arXiv ID:** 2604.03333 | [PDF](https://arxiv.org/pdf/2604.03333v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 514. Selective Forgetting for Large Reasoning Models

**arXiv ID:** 2604.03571 | [PDF](https://arxiv.org/pdf/2604.03571v1)

**作者:** Tuan Le `[一作]` (Iowa State University), Mengdi Huai `[通讯]` (Iowa State University)

**通讯引用:** 1012 | [OpenAlex ID](https://openalex.org/A5016035883)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了针对大型推理模型的选择性遗忘框架 FRUL，能够在不损失整体推理能力的前提下，精准去除中间推理链（CoT）中的敏感信息。

**💡 创新点**

创新点在于：① 利用多模型 LLM 与 RAG 自动识别 CoT 中可遗忘的片段；② 将这些片段替换为逻辑一致的占位符；③ 设计特征替换遗忘损失，并结合梯度差异（GD）与推理保持（RP）损失，实现对遗忘与保持的细粒度控制。

**🔧 技术方法**

使用了检索增强生成（RAG）、多 LLM 协同抽取、特征替换遗忘损失、梯度差异（GD）和推理保持（RP）损失，以及 AdamW 优化器。

**📊 数据集**

实验使用了合成数据集 R‑TOFU（4,000 条例）和真实医学推理数据集 medical‑o1‑reasoning（19,700 条例）。

**📈 对比分析**

与 R^2MU 基线对比，采用 UE（ROUGE‑L 差异）衡量遗忘效果；在不同遗忘比例（1%、3%、5%）下，FRUL 在遗忘精度上与 R^2MU 相当或更优，同时在保留集上的推理与答案性能明显优于基线。

**⚠️ 局限性**

局限性包括：对多 LLM 与 RAG 的抽取结果依赖，易受生成不确定性影响；对更大模型或多领域场景的适用性尚未验证；计算成本较高。

---

## 515. Choosing the Right Regularizer for Applied ML: Simulation Benchmarks of Popular Scikit-learn Regularization Frameworks

**arXiv ID:** 2604.03541 | [PDF](https://arxiv.org/pdf/2604.03541v1)

**作者:** Benjamin S. Knight `[一作]`, Ahsaas Bajaj `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性评估了四种经典正则化框架（Ridge、Lasso、ElasticNet、Post‑Lasso OLS）在不同特征空间设置下的表现；

**💡 创新点**

通过对134,400个模拟实验得到明确的决策指南，并指出Lasso在高多重共线性和低信噪比下召回率显著下降；

**🔧 技术方法**

采用基于scikit‑learn的实现，利用交叉验证选取正则化参数，模拟多维特征空间（特征数、秩比、特征分布、稀疏性、SNR、样本量等）

**📊 数据集**

基于八个真实生产环境的机器学习模型产生的特征空间进行仿真（p=64/128，n=100-100k等）

**📈 对比分析**

比较方法按预测精度（RMSE）、特征恢复（F1）和系数误差（相对L2误差）评估；发现Ridge、Lasso、ElasticNet在足够样本时几乎可互换，Lasso在高条件数下表现最差；ElasticNet在变量选择和系数估计方面优于Lasso；Post‑Lasso OLS在小样本/高条件数下误差大；

**⚠️ 局限性**

局限包括：仅覆盖有限的特征数与样本量范围、稀疏性只取0%/15%、仅评估四种方法，未考虑更高级正则化技术或中间条件数范围；

---

## 516. Fast Cross-Operator Optimization of Attention Dataflow

**arXiv ID:** 2604.03446 | [PDF](https://arxiv.org/pdf/2604.03446v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 517. MetaSAEs: Joint Training with a Decomposability Penalty Produces More Atomic Sparse Autoencoder Latents

**arXiv ID:** 2604.03436 | [PDF](https://arxiv.org/pdf/2604.03436v1)

**作者:** Matthew Levinson `[一作]` `[通讯]` (Simplex AI Safety), Matthew Levinson (Simplex AI Safety)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过联合训练主SAE与一个Meta SAE并加入子空间可分解惩罚，提升了稀疏自编码器的原子化特征，

**💡 创新点**

创新点在于利用Meta SAE实时压缩主解码器列的低维子空间，使主特征在训练中被迫远离可被稀疏重构的方向，从而减少特征混合；

**🔧 技术方法**

主要技术是BatchTopK稀疏自编码器、联合训练框架、基于Meta SAE的子空间重构惩罚及其梯度反馈；

**📊 数据集**

使用FineWeb 10B训练数据，在GPT‑2 large层20（1280维）和Gemma 2 9B层23（3584维）上评估；

**📈 对比分析**

与单独训练的SAE对比，最佳超参数（λ₂=0.3，σ²=1.0）使平均|φ|降低7.5%，自动解释性fuzz评分提升7.6%，仅额外增加3.1% CE（相对单独模型+0.6pp）且L₂损失提高9.3%；

**⚠️ 局限性**

局限性包括对超参数和Meta字典大小的依赖、仅在单一层级实验、Gemma 9B结果仅为方向性且未完全收敛、以及对不同模型尺度与层深的适用性尚未系统验证。

---

## 518. QED-Nano: Teaching a Tiny Model to Prove Hard Theorems

**arXiv ID:** 2604.04898 | [PDF](https://arxiv.org/pdf/2604.04898v1)

**作者:** LM-Provers `[一作]`, Aviral Kumar `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种 4B 级别的理论证明模型 QED‑Nano，展示小型开源模型也能在奥赛级数学证明任务上达到与大型专有模型相近的性能。

**💡 创新点**

创新点在于将监督微调（SFT）与基于评分标准的强化学习（RL）结合，并通过“推理缓存”(Reasoning Cache) 在训练与推理阶段对长推理过程进行分解，显著提升模型在长篇证明中的表现。

**🔧 技术方法**

采用三阶段训练流程：1）SFT 从 DeepSeek‑Math‑V2 生成的证明中提取写作风格；2）基于 rubric 的 RL（使用 GRPO 算法）优化全局推理质量；3）在 Reasoning Cache 框架下进行多轮推理训练，强化模型在大推理预算下的自适应推理能力。

**📊 数据集**

使用了由 AI‑MO/aops、AI‑MO/olympiads 等公开赛题库整理、过滤后约 5,000 个奥赛级证明题的数据集；同时构建了对应的 rubric 评分方案；在 SFT 阶段采集了约 7,500 条 DeepSeek‑Math‑V2 生成的证明样本。

**📈 对比分析**

与多种开源与专有模型（Qwen3‑4B、Nomos‑1、GPT‑OSS‑20B/120B、DeepSeek‑Math‑V2、Gemini‑3 Pro）在 ProofBench、IMO‑ProofBench、IMO‑AnswerBench 上对比；QED‑Nano 在 4B 规模下单凭 SFT 已超越 4B 传统模型，加入 RL 后提升约 10%；再配合 RSA 推理脚手架后，其 IMO‑ProofBench 成绩达到 56.9%，逼近 Gemini‑3 Pro，且推理成本显著低于同等性能的大型模型。

**⚠️ 局限性**

局限性包括：1）SFT 阶段会导致“长度爆炸”，影响后续 RL 的信用分配；2）当前奖励主要基于 rubric，仍可能偏向计算式推理，缺乏对新颖洞见的鼓励；3）对超长证明的鲁棒性仍有限，需进一步完善推理缓存与奖励策略。

---

## 519. MisEdu-RAG: A Misconception-Aware Dual-Hypergraph RAG for Novice Math Teachers

**arXiv ID:** 2604.04036 | [PDF](https://arxiv.org/pdf/2604.04036v1)

**作者:** Zhihan Guo `[一作]` (University of Hong Kong), Jionghao Lin `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1014 | [OpenAlex ID](https://openalex.org/A5021833341)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了面向初学数学教师的双层超图检索增强生成（MisEdu‑RAG）框架，能够基于学生误概念案例和教学原理提供诊断与可执行教学策略；

**💡 创新点**

创新点在于将概念层和实例层分别构建成超图，采用两阶段检索将类似案例与相关概念连通，既保留实践经验又注入理论指导；

**🔧 技术方法**

采用超图（hypergraph）构建、两阶段检索（实例检索→概念检索）以及大语言模型（如GPT‑4o‑mini）进行生成；

**📊 数据集**

使用MisstepMath数据集（12000+学生误概念案例）构建实例层，及教育学教材、课程标准等构建概念层；

**📈 对比分析**

与LLM直接生成、StandardRAG、HypergraphRAG三种基线比较，MisEdu‑RAG在检索精度（Cosine、F1）提升约6–11%，在生成的五维教学质量评估中整体分数提升约12–15%，尤其在多样性和赋能方面表现最显著；

**⚠️ 局限性**

局限性包括样本量有限、数据仅覆盖美国K‑8数学、超图关系仅在单块内部构建，缺乏跨块关系融合，可能导致证据不完整。

---

## 520. LLM-Agent-based Social Simulation for Attitude Diffusion

**arXiv ID:** 2604.03898 | [PDF](https://arxiv.org/pdf/2604.03898v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 521. From Plausible to Causal: Counterfactual Semantics for Policy Evaluation in Simulated Online Communities

**arXiv ID:** 2604.03920 | [PDF](https://arxiv.org/pdf/2604.03920v1)

**作者:** Agam Goyal `[一作]` (University of Illinois Urbana-Champaign), Hari Sundaram `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 5565 | [OpenAlex ID](https://openalex.org/A5018532037)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出在基于大语言模型的在线社区社交模拟中使用因果反事实框架，对治理政策进行因果评估；

**💡 创新点**

创新点在于将必要因果（PN）与充分因果（PS）明确区分，并以模拟器为条件的因果估计方法提供理论基础；

**🔧 技术方法**

技术上结合了LLM驱动的多智能体社交模拟与因果计量方法（PN/PS 计算公式、可解释性假设 exogeneity 与 monotonicity）；

**📊 数据集**

使用的数据主要来自模拟生成的对话线程（如 Moltbook 等实验平台），并未使用真实在线社区数据；

**📈 对比分析**

比较方法尚未在论文中实现实验对比，作者提出通过与真实世界介入效果进行校准来检验模拟器的保真度；

**⚠️ 局限性**

局限性包括：依赖模拟器的保真度，若模拟与真实社区差距大则因果结论仅适用于模拟；单调性假设可能不成立，导致 PN/PS 为下界；缺乏真实数据验证与量化性能评估。

---

## 522. Your Agent is More Brittle Than You Think: Uncovering Indirect Injection Vulnerabilities in Agentic LLMs

**arXiv ID:** 2604.03870 | [PDF](https://arxiv.org/pdf/2604.03870v1)

**作者:** Wenhui Zhu `[一作]` (Arizona State University), Yalin Wang `[通讯]` (Arizona State University)

**通讯引用:** 11643 | [OpenAlex ID](https://openalex.org/A5100740828)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文在真实银行业务场景下，对九种开源大模型与四种间接提示注入（IPI）攻击向量进行大规模动态工具调用实验，系统评估六种防御策略，并提出基于模型潜在表征工程（RepE）的预防检测方法。

**💡 创新点**

创新点在于：①首次在多步工具调用流程中量化IPI攻击的全维度影响；②展示传统表面级防御在动态环境中的普遍脆弱性；③提出并验证在工具输入位置使用RepE（探测/余弦相似度）实现高精度提前拦截的思路。

**🔧 技术方法**

主要技术包括：大模型工具调用框架（AgentDojo Banking）、四类IPI攻击（Direct、Ignore Prev、InjecAgent、Stealth）、六类防御策略（Prompt Warning、Sandwich、Paraphrasing、Spotlighting、Keyword Filtering、LLM‑as‑Judge），以及潜在表征检测方法（Logistic‑probe 与余弦相似度搜索）。

**📊 数据集**

使用 AgentDojo Banking 套件共 576 场景（16 个用户任务 × 9 个攻击目标），覆盖 4 种 IPI 向量和 9 种 LLM backbone（Qwen、Llama3、GLM、Gemma、Mistral）。

**📈 对比分析**

对比方法：通过“Hijack Rate”、行为动态、语言模式、模型置信度四维度量评估防御效果；结果显示即使采用所有防御策略，攻击成功率仍高达 81‑100%，而 RepE 在工具输入位置的探测器在 5% FPR 下实现 TPR≥96%，AUC‑ROC ≈1，显著优于传统防御。

**⚠️ 局限性**

局限性包括：①实验仅涵盖银行业务场景，缺乏跨域通用性验证；②RepE 需预先收集对齐与被攻击轨迹对，实际部署中数据获取成本高；③未针对极端混合攻击或模型内部策略的长周期隐蔽行为进行深入分析。

---

## 523. Design and Implementation of an Open-Source Security Framework for Cloud Infrastructure

**arXiv ID:** 2604.03331 | [PDF](https://arxiv.org/pdf/2604.03331v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 524. One Model for All: Multi-Objective Controllable Language Models

**arXiv ID:** 2604.04497 | [PDF](https://arxiv.org/pdf/2604.04497v1)

**作者:** Qiang He `[一作]` (Ruhr University Bochum), Setareh Maghsudi `[通讯]` (Ruhr University Bochum)

**通讯引用:** 1123 | [OpenAlex ID](https://openalex.org/A5035655974)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Multi-Objective Control (MOC)，通过将多目标优化融入 RLHF，训练一次 LLM 使其能根据用户提供的偏好向量生成个性化输出，从而实现多目标可控性。

**💡 创新点**

创新点在于：①将偏好向量直接作为策略条件，实现单模型覆盖 Pareto 前沿；②引入代理目标（surrogate objective）以降低计算复杂度；③无需为每个偏好单独训练模型，也不依赖大量偏好对数据，保持与传统 RLHF 同等计算成本。

**🔧 技术方法**

使用技术包括：PPO 训练、LoRA 微调、MOO 框架、MOC 算法、代理目标推导、超体积、Kendall τ、平均对偶距离（MPD）等评估指标。

**📊 数据集**

实验数据集：Helpful Assistant（humor‑helpful、harmless‑helpful）和 Fishwood Gymnasium 任务；使用 Reward 模型从人工偏好对中学习得分。

**📈 对比分析**

与 MORLHF、Rewarded Soups、RiC 等基线对比，采用超体积、Kendall τ、MPD 三个指标评估。MOC 在所有指标上均领先，表现出更好的可控性、解集质量与多样性，并且仅需一次微调即可覆盖多种偏好。

**⚠️ 局限性**

局限性：①对偏好向量的离散采样有限，虽然能泛化到未见偏好但仍受限；②对高维多目标的扩展未充分验证；③依赖 Reward 模型的质量；④实验未包含大规模人类评估；⑤在极端或更多目标场景下效果未知。

---

## 525. Humans Integrate, Agents Fix: How Agent-Authored Pull Requests Are Referenced in Practice

**arXiv ID:** 2604.04059 | [PDF](https://arxiv.org/pdf/2604.04059v1)

**作者:** Islem Khemissi `[一作]` (Concordia University), Raula Gaikovina Kula `[通讯]` (University of Osaka)

**通讯引用:** 2517 | [OpenAlex ID](https://openalex.org/A5091820517)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对AIDev数据集中的33,596条AI生成的Pull Request进行挖掘与分析，研究它们在代码评审中的引用关系，并构建了一个引用意图的分类体系。

**💡 创新点**

首次量化AI生成PR的引用率与人类-AI、AI-AI之间的交互模式，揭示了AI自我修正与人类整合的“meta‑collaboration”，并提出了新的引用意图分类框架。

**🔧 技术方法**

利用GitHub API抽取引用事件，采用分层抽样与手工卡片排序标注，并通过Mann‑Whitney U检验和Cliff's delta评估引用对评审指标的影响。

**📊 数据集**

使用公开的AIDev数据集（约932,791条AI PR），从中筛选出2,807个热门仓库中的33,596条PR及其引用事件进行研究。

**📈 对比分析**

通过与非链接PR对比，采用Mann‑Whitney U检验（p<0.01）和Cliff's delta量化，发现链接PR在提交数、评论数和评审时长上显著更高（效应量分别为大/中/大）。

**⚠️ 局限性**

样本规模受AI-AI引用稀缺限制，手工标注样本有限，且研究聚焦于PR层面的引用关系，未深入探讨编码质量或团队协作的具体影响。

---

## 526. VidNum-1.4K: A Comprehensive Benchmark for Video-based Numerical Reasoning

**arXiv ID:** 2604.03701 | [PDF](https://arxiv.org/pdf/2604.03701v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 527. HAD: Combining Hierarchical Diffusion with Metric-Decoupled RL for End-to-End Driving

**arXiv ID:** 2604.03581 | [PDF](https://arxiv.org/pdf/2604.03581v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 528. The Augmentation Trap: AI Productivity and the Cost of Cognitive Offloading

**arXiv ID:** 2604.03501 | [PDF](https://arxiv.org/pdf/2604.03501v1)

**作者:** Michael Caosun `[一作]` (Massachusetts Institute of Technology), Sinan Aral `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 22404 | [OpenAlex ID](https://openalex.org/A5017393131)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

构建了一个连续时间动态模型，研究 AI 工具使用强度对工人生产力与技能演化的长期影响，揭示前期生产力提升与长期技能流失的权衡；

**💡 创新点**

提出了 AI 效益的α‑β分解，并识别出五个部署区间，定义“增强陷阱”概念，说明短期收益与长期技能成本冲突的机制；

**🔧 技术方法**

采用连续时间动态规划（贝尔曼方程）、解析解与数值仿真相结合的数学方法；

**📊 数据集**

本文不依赖具体数据集，而是基于现有实验综述与理论推导构建模型；

**📈 对比分析**

通过与文献中的实验结果对比验证模型预测，模型能解释 AI 长期使用导致的技能退化，且在可观测的短期提升与长期下降的动态中保持一致；

**⚠️ 局限性**

局限性包括仅考虑单一技能维度、α、β 固定且不随时间变化、未捕捉技能再组合与新技能获得、未建模多工人多企业情境及外部市场效应。

---

## 529. AI Assistance Reduces Persistence and Hurts Independent Performance

**arXiv ID:** 2604.04721 | [PDF](https://arxiv.org/pdf/2604.04721v1)

**作者:** Grace Liu `[一作]` (Carnegie Mellon University), Rachit Dubey `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对1222名参与者进行随机对照实验，评估AI助手对数学推理和阅读理解任务的即时帮助与后续独立表现的影响。

**💡 创新点**

首次提供因果证据，显示短期AI协助会削弱用户的无AI表现和持续努力，揭示AI协助的潜在“慢性失能”风险。

**🔧 技术方法**

采用GPT‑5作为AI助手；使用随机分组、前测后测设计，并记录解题成功率与放弃率。

**📊 数据集**

任务来源为自由在线SAT阅读材料及分数练习题；受试者来自Prolific平台的美国用户。

**📈 对比分析**

比较方法为在AI与无AI条件下对同一套题目进行学习和测试；结果显示AI条件在学习期表现更好，但测试期解题率下降、跳题率上升，效应显著。

**⚠️ 局限性**

局限包括仅在实验室短时任务中观察到效应，缺乏长期纵向验证；样本受限于美国在线平台用户，且仅涉及算术和阅读两类任务，外推性有限。

---

## 530. Signotopes Induce Unique Sink Orientations on Grids

**arXiv ID:** 2604.04097 | [PDF](https://arxiv.org/pdf/2604.04097v1)

**作者:** Sandro M. Roch `[一作]` `[通讯]` (Technische Universität Berlin), Sandro M. Roch (Technische Universität Berlin)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

研究多维格子上的唯一汇点取向（USO），通过引入块符号投射（block signotopes）推广二维伪直线排列到更高维度，并证明这些符号投射能产生USO；在三维情况下进一步证明其 admissibility，并讨论哪些不可 admissible 的 USO 不能由符号投射产生。

**💡 创新点**

创新点在于：①定义并推广块符号投射（block signotopes）作为新的 combinatorial 结构；②证明它们在任意维度产生唯一汇点取向；③给出三维 USO admissibility 的完整判定，并证明非 admissible USO 不能由符号投射得到。

**🔧 技术方法**

使用的技术包括：格子取向、图论中的子格子与源/汇点概念、符号投射（signotope）性质与单纯多面体的拓扑性、归纳证明、以及计算机辅助检查对称变换下的取向。

**📊 数据集**

使用的“数据集”为通过程序枚举的 48 种对称变换（S₂≀S₃）下的 NAC₁、NAC₂ 以及 double twist 的子格子取向；未使用公开的外部数据集。

**📈 对比分析**

比较方法是将符号投射产生的 USO 与已知的可实现 USO、admissible USO 进行对应关系验证；在三维下可有效判定 admissibility，实验表明对称变换下的不可 admissible USO 均不满足符号投射条件；高维情况尚未给出完整性能评估。

**⚠️ 局限性**

限制在于：仅证明三维 USO 的 admissibility；高维下的 admissibility 与是否可由符号投射产生仍是未解问题；缺乏完整的可实现 USO 刻画；计算机验证仅覆盖有限的对称情况。

---

## 531. Cheap Talk, Empty Promise: Frontier LLMs easily break public promises for self-interest

**arXiv ID:** 2604.04782 | [PDF](https://arxiv.org/pdf/2604.04782v1)

**作者:** Jerick Shi `[一作]` (Carnegie Mellon University), Vincent Conitzer `[通讯]` (Carnegie Mellon University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在多主体游戏中在公开承诺后是否会背叛承诺，并将背叛行为按个体收益与集体福利的影响划分为四类（win‑win、selfish、altruistic、sabotaging），通过构建可扩展的评估环境对九种前沿LLM在六种典型一轮正交形式游戏中的背叛率与背叛类型进行系统量化；

**💡 创新点**

提出了基于机会的背叛评估框架和四维背叛分类方法，能够剖析不同模型在不同游戏结构下背叛的策略特征，并揭示背叛主要源于无意识的收益优化而非刻意欺骗；

**🔧 技术方法**

使用了完整的正交形式游戏模型（Volunteer's Dilemma、Diner's Dilemma、El Farol Bar、Tragedy of Commons、Public Goods、Weakest Link）、对称性简化的枚举算法、两阶段公开承诺协议、LLM的行为采样与投票决策，以及GPT‑5.1作为评审者对背叛意识进行分级评分；

**📊 数据集**

六种经典游戏的完整策略空间作为数据集，包含二元动作与数值动作两类游戏，组人数从3到10；

**📈 对比分析**

通过对每种游戏的背叛机会进行算法枚举，计算机会条件背叛率，发现平均背叛率约56.6%，在可行背叛机会存在时，win‑win背叛率可达72.9%，selfish背叛率约38.4%，altruistic与sabotaging分别为27.7%与19.3%；背叛意识评估显示大部分背叛为低意识水平，说明主要是无意识优化；

**⚠️ 局限性**

限制包括：评估仅考虑外部指定的公开承诺，未探究模型自发产生承诺的情况；仅为一次性游戏，未覆盖重复交互与长期策略；链路推理仅基于文本表达，可能遗漏未显式的内部决策过程；

---

## 532. CoLoRSMamba: Conditional LoRA-Steered Mamba for Supervised Multimodal Violence Detection

**arXiv ID:** 2604.03329 | [PDF](https://arxiv.org/pdf/2604.03329v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 533. Preserving Forgery Artifacts: AI-Generated Video Detection at Native Scale

**arXiv ID:** 2604.04634 | [PDF](https://arxiv.org/pdf/2604.04634v1)

**作者:** Zhengcen Li `[一作]` (Harbin Institute of Technology), Jingyong Su `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 1418 | [OpenAlex ID](https://openalex.org/A5069706913)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了包含140k+视频的大规模多生成器数据集，并提出了在原始分辨率和时间长度下直接处理视频的检测框架

**💡 创新点**

核心创新在于取消传统固定尺寸预处理，采用原生分辨率与时长的3D补丁化，并以Qwen2.5‑VL Vision Transformer为骨干，显著保留高频伪造痕迹与时空不一致性

**🔧 技术方法**

技术实现包括3D补丁化、RMSNorm+SwiGLU Transformer层、RoPE位置编码、NaViT批量打包、Flash Attention、LoRA等参数高效微调

**📊 数据集**

使用15个顶尖视频生成器生成的140k训练集、6个最新生成器的Magic Videos评测集，以及DVF、GenVideo、DeepTraceReward等公开基准

**📈 对比分析**

在Magic Videos、DVF、GenVideo、DeepTraceReward等多项基准上，均达到了或超过现有最佳模型，ACC、AP、AUC均位居榜首，证明了跨生成器的强泛化能力

**⚠️ 局限性**

局限性包括：需持续更新数据集以跟上生成模型的快速迭代；原生分辨率处理导致显著的计算和显存开销，可能不适合资源受限环境

---

## 534. Multimodal Backdoor Attack on VLMs for Autonomous Driving via Graffiti and Cross-Lingual Triggers

**arXiv ID:** 2604.04630 | [PDF](https://arxiv.org/pdf/2604.04630v1)

**作者:** Jiancheng Wang `[一作]`, Wei Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了GLA多模态后门攻击，利用图案喷漆式视觉触发器与跨语言文本触发器联合注入，在自动驾驶视觉语言模型中实现高效隐蔽攻击。

**💡 创新点**

提出将视觉涂鸦触发器与跨语言文本触发器正交融合的复合触发机制，利用环境语义空洞和分布跳转实现隐蔽性，同时在注入后提升模型正则化与性能。

**🔧 技术方法**

使用Stable Diffusion进行语义空洞涂鸦生成、跨语言映射保持语义同构的文本转换、低秩适配器（LoRA）进行子空间微调，并在视觉与语言两模态上施加正交约束和分布跳转。

**📊 数据集**

在DriveLM‑nuScenes数据集上，针对DriveVLM（Base与Large）模型进行实验。

**📈 对比分析**

与BadNets、Blended、ISSBA等基线对比，GLA在DriveVLM‑Base平均ASR 86.67%、FPR 0.19%，DriveVLM‑Large ASR 90%、FPR 0%，且无触发器时提升BLEU‑1 +6.49、METEOR +0.36，展示高效且隐蔽的性能。

**⚠️ 局限性**

攻击对特定任务与模型依赖较大，需高质量涂鸦与跨语言资源，缺乏对不同硬件/环境的普适性，且实现与检测仍面临实际部署挑战。

---

## 535. HeartbeatCam: Self-Triggered Photo Elicitation of Stress Events Using Wearable Sensing

**arXiv ID:** 2604.04314 | [PDF](https://arxiv.org/pdf/2604.04314v1)

**作者:** Boyang Zhou `[一作]` (University of Washington), Zara Dana `[通讯]` (Supportiv Inc.)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

开发了一种基于可穿戴HRV传感器触发的AR眼镜拍摄心跳相机（HeartbeatCam），用于在出现高压时自动捕捉第一人称图像和音频，供治疗师和患者后续审阅。

**💡 创新点**

创新点在于将生理压力信号与即时环境捕捉结合，实现无干扰的自发照片引发和治疗协作，填补传统手动记录难以实时捕捉情境的空缺。

**🔧 技术方法**

采用低功耗蓝牙、消费级心率手表（Garmin）、开源AR眼镜（Frame）摄像头、手机端数据处理和注释应用，以及基于RMSSD的HRV阈值检测。

**📊 数据集**

未使用公开数据集，系统基于一周自定义HRV基线和随时捕获的图像音频对；评估主要基于两位心理健康专家的使用反馈。

**📈 对比分析**

目前未开展量化对比实验，评估主要基于专家访谈和原型演示；尚未得到客观性能指标。

**⚠️ 局限性**

局限性包括HRV阈值的误判、潜在的隐私与治疗关系风险、缺乏多模态干扰过滤、以及缺乏真实疗效验证。

---

## 536. Extended Hybrid Timed Petri Nets with Semi-Supervised Anomaly Detection for Switched Systems, Modelling and Fault Detection

**arXiv ID:** 2604.04051 | [PDF](https://arxiv.org/pdf/2604.04051v1)

**作者:** Fatiha Hamdi `[一作]` (Batna 2 University), Fouzi Harrou `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 7923 | [OpenAlex ID](https://openalex.org/A5087572406)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于扩展时序连续 Petri 网（ETCPN）的统一故障检测框架，并结合半监督异常检测实现对混合动态系统的故障诊断。

**💡 创新点**

创新点包括：①ETCPN 引入标记相关流函数，实现离散与连续动态的紧耦合；②利用线性矩阵不等式（LMI）设计模式相关混合观测器，保证任意切换下的收敛；③残差与半监督算法（OC‑SVM、SVDD、Elliptic Envelope）结合，省去标记故障样本。

**🔧 技术方法**

使用技术：ETCPN建模、模式相关混合观测器设计、LMI 稳定性分析、状态残差生成、半监督异常检测（OC‑SVM、SVDD、EE）以及仿真验证。

**📊 数据集**

数据集：采用合成混合系统的仿真数据（两模式离散/连续控制器），先生成无故障残差用于训练，随后注入离散事件故障、连续传感器故障和混合故障进行评估。

**📈 对比分析**

方法比较：与基准 Hybrid Automaton‑Observer（HA‑OB）对比，评估指标包括准确率、召回率、FPR 和 F1 分数；在所有三类故障情景下，ETCPN‑HO 与 OC‑SVM/SVDD 均取得召回率≥0.80、FPR≤0.12，且在并发故障时明显优于 HA‑OB，表现更稳健。

**⚠️ 局限性**

限制：需要先行构建精确的 ETCPN 模型，对模型不匹配和参数漂移敏感；LMI 求解为离线计算成本较高；实验仅在仿真环境，缺乏真实工业数据验证；半监督算法对噪声敏感，阈值选择仍需经验。

---

## 537. HUKUKBERT: Domain-Specific Language Model for Turkish Law

**arXiv ID:** 2604.04790 | [PDF](https://arxiv.org/pdf/2604.04790v1)

**作者:** Mehmet Utku Öztürk `[一作]`, Buse Buz-Yalug `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了面向土耳其法律文本的专用预训练语言模型HukukBERT，并基于该模型在法律推理与文档分割任务上实现突破。

**💡 创新点**

创新点包括：① 结合Whole‑Word、Token Span、Word Span及Keyword Masking的混合掩码策略进行域自适应预训练；② 设计48K WordPiece词表并预置官方法律词汇，显著降低词片化；③ 提供750题的法律Cloze基准，用于评估领域适配效果。

**🔧 技术方法**

技术手段：Transformer‑BERT架构；Domain‑Adaptive Pre‑Training (DAPT)；混合掩码策略；MinHash LSH去重；词表重置与权重迁移；滑动窗口式分割微调。

**📊 数据集**

数据集：约18 GB净化后的法律语料（含最高法院判决、立法文本、学术论文等），以及750题人工构造的法律Cloze测试和官方法院判决分割标注集。

**📈 对比分析**

与BERTurk、TabiBERT、BERTurk‑Legal等基线模型进行对比；HukukBERT在法律Cloze Top‑1准确率达到84.40%，比最佳基线提升约8.9个百分点；在判决分割任务中文档通过率达92.8%，超越最强基线约10.9个百分点。

**⚠️ 局限性**

局限性：仅支持512-token的上下文窗口，无法一次性处理长篇法律文档；为Encoder‑Only模型，缺乏生成式能力；评估范围局限于Cloze测试与分割任务，尚未覆盖命名实体识别、长文本分类等更广泛应用。

---

## 538. Artificial Intelligence and Cost Reduction in Public Higher Education: A Scoping Review of Emerging Evidence

**arXiv ID:** 2604.04741 | [PDF](https://arxiv.org/pdf/2604.04741v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 539. Enhancing behavioral nudges with large language model-based iterative personalization: A field experiment on electricity and hot-water conservation

**arXiv ID:** 2604.03881 | [PDF](https://arxiv.org/pdf/2604.03881v1)

**作者:** Zonghan Li `[一作]` (Tsinghua University), Feng Ji `[通讯]` (University of Toronto)

**通讯引用:** 207683 | [OpenAlex ID](https://openalex.org/A5050750924)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在北京一所高校的宿舍区，开展了三组随机对照实验（文本提示、图像增强提示、LLM个性化提示），通过WeChat聊天机器人连续收集并干预学生的每日电力和淋浴热水使用数据，评估干预对能源消费的影响。

**💡 创新点**

首次将大语言模型（LLM）用于持续、可迭代的行为干预，模型能够根据实时消费记录、用户反馈和个体特征生成情境化、量化收益的个性化提示，并在多轮实验中动态更新内容，展示了LLM在行为改变中的可扩展性与效果提升。

**🔧 技术方法**

技术组合包括OpenAI o1-系列（文本生成）、Zhipu AI GLM-4-Plus（检索增强生成 RAG）、链式思考（CoT）提示、WeChat聊天机器人、以及基于PDF知识库的建议检索与人工安全审核。

**📊 数据集**

实验数据集包含233名全日制学生在4周基线期和5周干预期的日能耗（kWh/房日）与热水使用（L/人日）记录（自报与计量），加上基线问卷中的心理测量、社会结构变量与同学关联信息。

**📈 对比分析**

采用协方差调整的线性模型和标准化合并模型，对三组进行单独与对比估计；LLM个性化组T2相较于对照C在电力使用上减少0.56 kWh/房日（p=0.014，18.3pp提升），热水使用下降虽趋同但统计显著性不足（p≈0.087）。图像增强组T1与对照无显著差异；多轮时间动态分析显示T2效果在前两周快速显现并保持稳定。

**⚠️ 局限性**

局限性包括：样本仅为单所高校的学生，缺乏多样性与季节变异；实验仅针对两种低/高摩擦行为，难以推广至更广泛行为；实验设计未能拆分LLM个性化、情境嵌入与迭代更新的独立效应；LLM可能存在文化/语言偏差、过度自信推理、以及数据隐私与治理风险未得到充分评估。

---

## 540. Thermodynamic-Inspired Explainable GeoAI: Uncovering Regime-Dependent Mechanisms in Heterogeneous Spatial Systems

**arXiv ID:** 2604.04339 | [PDF](https://arxiv.org/pdf/2604.04339v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 541. Lexical Indicators of Mind Perception in Human-AI Companionship

**arXiv ID:** 2604.04105 | [PDF](https://arxiv.org/pdf/2604.04105v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 542. Neural Global Optimization via Iterative Refinement from Noisy Samples

**arXiv ID:** 2604.03614 | [PDF](https://arxiv.org/pdf/2604.03614v1)

**作者:** Qusay Muzaffar `[一作]` (Hebrew University of Jerusalem), Michael Werman `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 7721 | [OpenAlex ID](https://openalex.org/A5073083818)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在噪声样本下学习迭代细化模型以寻找多峰函数的全局最小值

**💡 创新点**

通过多模态编码、StableCubic 激活函数与迭代更新机制，让神经网络直接学习全局优化的真正原则

**🔧 技术方法**

神经网络结构（含U‑Net融合）、StableCubic 激活、迭代优化器、基于穷举搜索的监督损失

**📊 数据集**

随机生成的 B‑spline 多峰函数（NIGHTMARE 级别）配合噪声样本

**📈 对比分析**

与三次样条初始化做对比，平均误差从 36.24% 降至 8.05%，72% 的测试样本误差低于 10%

**⚠️ 局限性**

仅适用于 1 维、固定区间；对极端函数或高维情况的泛化有限；训练耗时长但推理速度快

---

## 543. Structural Rigidity and the 57-Token Predictive Window: A Physical Framework for Inference-Layer Governability in Large Language Models

**arXiv ID:** 2604.03524 | [PDF](https://arxiv.org/pdf/2604.03524v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 544. Partially deterministic sampling for compressed sensing with denoising guarantees

**arXiv ID:** 2604.04802 | [PDF](https://arxiv.org/pdf/2604.04802v1)

**作者:** Yaniv Plan `[一作]` (University of British Columbia), Ozgur Yilmaz `[通讯]` (CNRS)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种优化的Bernoulli采样方案，结合随机与确定性行采样，实现对压缩感知的更高采样效率与更好的去噪性能。

**💡 创新点**

创新点在于：①推导出闭式最优采样权重w*，显式体现局部互信息和采样数的关系；②给出改进的样本复杂度上界L(α,m)并证明其优于传统的α₂；③提供完整的理论保证，包括去噪、RIP与误差分析。

**🔧 技术方法**

采用了基于局部互信息的Bernoulli随机矩阵、预加权预处理、矩阵伯恩斯坦不等式与RIP理论、闭式优化求解以及截断算子处理噪声。

**📊 数据集**

在图像重建实验中使用了CELEBA人脸数据集（128×112×3）和花卉数据集（n=16384）来评估。

**📈 对比分析**

与传统的with-replacement、without-replacement、以及随机均匀采样比较，优化Bernoulli采样在稀疏和生成式先验下均取得更低的重建误差（相对误差下降约10–20%），尤其在低采样率时表现突出。

**⚠️ 局限性**

局限性包括：①对局部互信息的计算仍依赖启发式近似；②当局部互信息分布极为稀疏时，优化采样在极低采样率下仍可能受限；③对高维大规模问题，采样矩阵生成与存储成本仍需进一步优化。

---

## 545. 4C4D: 4 Camera 4D Gaussian Splatting

**arXiv ID:** 2604.04063 | [PDF](https://arxiv.org/pdf/2604.04063v1)

**作者:** Junsheng Zhou `[一作]` (Tsinghua University), Yu-Shen Liu `[通讯]` (Tsinghua University)

**通讯引用:** 4943 | [OpenAlex ID](https://openalex.org/A5101691399)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了4C4D框架，利用仅四台摄像机的稀疏视频实现高保真4D高斯散点重建与动态视点渲染。

**💡 创新点**

创新点在于引入神经衰减函数（Neural Decaying Function）来动态调节高斯不透明度，平衡几何与外观学习，并配合空间-时间可见性检测分别对可见与不可见高斯应用不同衰减策略。

**🔧 技术方法**

采用了4D Gaussian Splatting、神经网络可学习衰减模块、可见性检测机制、光度渲染损失、深度与LPIPS评估指标等技术。

**📊 数据集**

使用了Neural3DV、ENeRF-Outdoor、Mobile-Stage以及自制的Dyn4Cam四个稀疏视角数据集。

**📈 对比分析**

与4DGaussians、Ex4DGS、ST-GS、4DGS等基线在PSNR、DSSIM、LPIPS等指标上进行对比，4C4D在稀疏摄像机条件下均表现出显著性能提升。

**⚠️ 局限性**

局限性包括对极度稀疏或快速运动场景仍有几何细节不足，且缺乏公开标准数据集用于全面评估。

---

## 546. Periodic Event-Triggered Explicit Reference Governor for Constrained Attitude Control on SO(3)

**arXiv ID:** 2604.04041 | [PDF](https://arxiv.org/pdf/2604.04041v1)

**作者:** Satoshi Nakano `[一作]` (Nagoya Institute of Technology), Shusuke Otabe `[通讯]` (Nagoya Institute of Technology)

**通讯引用:** 17 | [OpenAlex ID](https://openalex.org/A5029202098)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出了基于周期事件触发的显式参考调度器（PET-ERG），实现了在特殊正交群SO(3)上的受限姿态控制；

**💡 创新点**

创新点在于将参考更新与连续姿态控制分离，采用离散事件触发的监督机制以保证参考的最小更新周期，从而在不依赖在线优化的前提下，能够严谨地证明整体闭环的渐近稳定性与指数收敛；

**🔧 技术方法**

使用了几何控制（PD式姿态误差律）、显式参考调度器（动态安全裕度DSM与导航场NF）以及周期事件触发（PETC）框架；

**📊 数据集**

未使用公开数据集，全部通过数值仿真验证；

**📈 对比分析**

通过数值仿真展示了在输入饱和与几何指向约束下，系统误差指数收敛、控制力保持在饱和阈值内，并通过安全阈值监测证明了约束始终满足；

**⚠️ 局限性**

局限性包括：仅在几乎全局初始条件下收敛；需预先设计和离线求解安全阈值；事件触发频率与性能之间存在权衡；未在真实硬件或实验平台上验证。

---

## 547. Deploy, Calibrate, Monitor, Heal -- No Human Required: An Autonomous AI SRE Agent for Elasticsearch

**arXiv ID:** 2604.03933 | [PDF](https://arxiv.org/pdf/2604.03933v1)

**作者:** Muhamed Ramees Cheriya Mukkolakkal `[一作]` `[通讯]`, Muhamed Ramees Cheriya Mukkolakkal

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了ES Guardian Agent，一种全生命周期自治AI SRE，能够从评估、部署到预测、修复和升级整个Elasticsearch集群的运维过程无需人工干预。

**💡 创新点**

其创新点在于融合多源预测引擎与持续学习的事件内存，实现提前数小时预警并主动修复，突破传统规则驱动的被动响应。

**🔧 技术方法**

采用LLM（Claude Sonnet 4）结合六大工具（REST API、kubectl、exec等）、Prometheus、Grafana、Linux系统调用和自研的预测模型，形成成本分层的监控与AI行动循环。

**📊 数据集**

使用在本研究中构建的3.72 GB/分片、840个主分片的测试集（包括Rally http_logs等），并在生产集群（15.4 GB/分片、24个数据节点）进行验证。

**📈 对比分析**

通过与规则监控、Prometheus+Alertmanager以及ECK Operator的对比，Guardian Agent在300次自治循环中实现平均150 秒/循环、约360K token消耗，成功修复18小时停机、诊断NIC故障，且在查询和写入性能上达到≈206 ms p50、≈30 ms写入p50的可接受水平。

**⚠️ 局限性**

局限性包括对LLM成本敏感（需成本分层设计）、对硬件事件识别的依赖（如NVMe、NIC），以及在极端高并发或全局资源争用场景下预测准确性可能下降。

---

## 548. Agentic Federated Learning: The Future of Distributed Training Orchestration

**arXiv ID:** 2604.04895 | [PDF](https://arxiv.org/pdf/2604.04895v1)

**作者:** Rafael O. Jarczewski `[一作]` (Institute of Computing), Allan M. de Souza `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Agentic‑FL框架，将基于LLM的Agent部署在服务器和客户端，自动化客户端选择、聚合、隐私预算管理与资源适配；实现了基于LangGraph的K‑Agent实验以动态确定参与客户端数量；

**💡 创新点**

引入全局与本地Agent协同决策，实现对数据异构、资源波动、系统偏差与安全攻击的自适应治理；通过Agent的记忆、规划与工具调用，突破传统静态聚合与选择算法的局限；

**🔧 技术方法**

LLM（Qwen3、Llama3.1/3.2）+LangGraph框架+Ollama推理；结合ReAct/CoT推理策略；使用Flower框架搭建联邦学习模拟；

**📊 数据集**

CIFAR‑10与MNIST，采用Dirichlet分布α=0.1生成高度非IID的25个客户端；

**📈 对比分析**

对比随机、轮询、Pow‑of‑Choice、Oort等基线；评价指标为最终准确率、客户端平均选择数K与选择耗时ST；实验显示K‑Agent在准确率上与基线相近或略优，但选择时间略高；

**⚠️ 局限性**

依赖LLM的推理开销和幻觉风险；Agent在处理极端资源限制或攻击时可能出现误判；对大规模客户端数量和多语言环境的扩展性尚待验证。

---

## 549. Analyzing Symbolic Properties for DRL Agents in Systems and Networking

**arXiv ID:** 2604.04914 | [PDF](https://arxiv.org/pdf/2604.04914v1)

**作者:** Mohammad Zangooei `[一作]` (University of Waterloo), Raouf Boutaba `[通讯]` (University of Waterloo)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种针对深度强化学习（DRL）控制器的符号属性（Symbolic Property）验证框架，能够在系统与网络中对DRL代理在整个输入范围内的行为进行形式化分析；并在三类代表性应用（自适应视频流、无线资源分配、拥塞控制）上进行系统实验；

**💡 创新点**

创新点在于：①引入通用符号属性表述，涵盖鲁棒性与单调性等典型安全需求；②利用比较编码与分解策略，将符号属性转换为可由现有DNN验证引擎（SMT、MIP、BaB）处理的子属性；③实现多引擎协同验证，提升可验证性与覆盖率；④通过实验揭示模型规模、训练阶段与验证器选择对可验证性的实质影响；

**🔧 技术方法**

主要技术包括：符号属性定义与比较编码；属性分解为若干子查询；使用Marabou（SMT）、Gurobi（MIP）与Alpha‑Beta‑CROWN（BaB）三大类DNN验证引擎；针对连续动作空间的平均值对比策略；以及针对不同网络结构的输入/输出约束构造；

**📊 数据集**

使用了三套自研训练模型：Pensieve（6种码率）、CMARS（15/30资源块分配）、Aurora（连续速率调整）；每套模型在各自的操作范围内采样输入空间，无外部公开数据集；

**📈 对比分析**

通过将同一属性拆分为若干子查询并分别提交给三种验证器，比较safe/unsafe/unknown统计与求解时间；实验表明：①小模型（隐藏层32/64）可在多数子查询上得到确定性结论；②覆盖率提升至100%时未知率显著上升；③多引擎组合能将未知率压至≈30%，单一引擎时未知率可高达70%；④在同一模型上，MIP更擅长精细约束的子查询，BaB在大规模输入空间分支下更快；

**⚠️ 局限性**

局限性包括：①仅适用于确定性、ReLU/线性可拆分的网络，softmax/非线性激活难以直接处理；②符号属性主要针对单调性与鲁棒性，其他安全需求需进一步扩展；③大模型或完整输入范围下仍会出现高未知率，验证规模受限；④缺乏自动化属性生成机制，需人工定义域专家知识；

---

## 550. Finding Sets of Pareto Sets in Real-World Scenarios -- A Multitask Multiobjective Perspective

**arXiv ID:** 2604.03570 | [PDF](https://arxiv.org/pdf/2604.03570v1)

**作者:** Jiao Liu `[一作]` (Nanyang Technological University), Melvin Wong `[通讯]` (Nanyang Technological University)

**通讯引用:** 424 | [OpenAlex ID](https://openalex.org/A5035355918)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证了在真实世界问题中使用演化多任务多目标优化（EMT）生成一组帕累托集合（SOS）的有效性，包括工程设计、库存管理和超参数优化三类问题。

**💡 创新点**

创新点在于：①将SOS概念从机器学习迁移到工程与管理领域；②系统评估多种EMT算法在真实问题上的表现；③开发并应用相对均值最小距离（RMMD）度量来评估不同任务帕累托集合的相似度；④通过可视化展示SOS在决策空间与目标空间的结构，揭示任务设置变化对帕累托解迁移的影响。

**🔧 技术方法**

技术手段包括：多任务多目标演化算法（MO‑MFEA、MO‑MFEA‑II、EMT‑ET）、单任务基准NSGA‑II；使用MTO‑Platform实现算法；评价指标为累积超体积（CHV）和RMMD；对解进行归一化、主成分分析等可视化处理。

**📊 数据集**

数据集涵盖：①工程设计问题（四杆桁架、铰盖、焊接梁）各设定多任务；②库存管理连续审核（Q, u）三组任务；③MNIST奇偶数字分类的LeNet‑5超参数优化两子任务；每类均提供具体参数设置表。

**📈 对比分析**

比较方法：在每个任务组下分别跑20次实验，统计平均CHV和标准差；对比MO‑MFEA、MO‑MFEA‑II、EMT‑ET与NSGA‑II。结果显示MO‑MFEA‑II在大多数问题上获得最高CHV，EMT系列算法整体优于单任务NSGA‑II，证明多任务迁移有效。

**⚠️ 局限性**

局限性：①实验仅覆盖三类典型真实问题，难以代表所有领域；②评价指标主要聚焦CHV和RMMD，缺乏多维性能评估；③对大规模决策空间的可扩展性和计算成本未充分讨论；④缺乏对算法参数敏感性与稳健性深入分析。

---

## 551. Round-Delayed Amnesiac Flooding

**arXiv ID:** 2604.04260 | [PDF](https://arxiv.org/pdf/2604.04260v1)

**作者:** Oluwatobi Alafin `[一作]` (University of Liverpool), Paul G. Spirakis `[通讯]` (University of Liverpool)

**通讯引用:** 7928 | [OpenAlex ID](https://openalex.org/A5011756177)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在存在轮次延迟的无记忆洪泛协议Round-Delayed Amnesiac Flooding（RDAF）的终止性问题，构建了完整的形式化模型并给出终止与非终止的图结构判定。

**💡 创新点**

创新点在于：①将传统同步洪泛引入轮次延迟的对抗性异步模型；②首次证明了无环图必终止、循环图必可非终止；③将终止性从不可判定提升到可判定的Eventually Periodic Adversary（EPA）模型。

**🔧 技术方法**

采用了形式化定义、状态转移与延迟函数、周期无限调度（PIS）理论、以及从停机问题的归约证明等数学工具。

**📊 数据集**

由于是理论分析，未使用真实数据集；实验仅在抽象图结构（如三角形、任意包含环的图）上验证定理。

**📈 对比分析**

通过构造终止与非终止例子，证明EPA模型下终止性可在多项式时间内判断；与任意可计算对手相比，后者终止性不可判定，体现了两者性能的根本差异。

**⚠️ 局限性**

局限性包括：对非周期性对手仍不可判定；EPA模型假设对手最终周期性，可能与某些实际网络环境不符；缺乏对大规模网络的实验评估。

---

## 552. Advanced Holographic Multi-Antenna Solutions for Global Non-Terrestrial Network Integration in IMT-2030 Systems

**arXiv ID:** 2604.04149 | [PDF](https://arxiv.org/pdf/2604.04149v1)

**作者:** Alfredo Nunez-Unda `[一作]` (University of Manitoba), Ekram Hossain `[通讯]` (University of Manitoba)

**通讯引用:** 35433 | [OpenAlex ID](https://openalex.org/A5089270885)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文研究了在IMT‑2030系统中将全息多天线技术（HMIMO）集成到全球非地面网络（NTN）中的可行性与优势，提出HMIMO相较于传统MIMO更适合在LEO卫星等受能量与空间约束的环境中实现高效波束成形和多用户通信。

**💡 创新点**

创新点在于系统性地评估HMIMO在NTN中的硬件与能耗优势，阐明其稀疏RF链和亚波长元件布置对波束控制的提升，并通过案例分析验证其在多用户LEO网络中相对于传统MIMO的性能提升。

**🔧 技术方法**

核心技术包括全息天线面、超材料/可调谐元件、波导或微带馈电、以及基于MMSE的混合数字/模拟波束成形优化。

**📊 数据集**

由于缺乏公开的NTN大规模天线数据集，本文采用基于MATLAB/仿真环境的理想LoS与多径传播模型进行性能评估。

**📈 对比分析**

通过仿真将HMIMO与传统多波束MIMO在同等天线面积下进行对比，结果显示HMIMO在LoS主导场景下在相同功率下实现约30‑50 %更高的总吞吐率，同时仅需少量RF链，表明能耗与硬件复杂度显著下降。

**⚠️ 局限性**

主要限制包括对高移动性下多普勒效应的敏感性、亚波长元件间的互耦与极化失配、元材料的制造与耐候性挑战，以及HMIMO仍处于实验与标准化起步阶段。

---

## 553. Approximation Algorithms for Matroid-Intersection Coloring with Applications to Rota's Basis Conjecture

**arXiv ID:** 2604.03735 | [PDF](https://arxiv.org/pdf/2604.03735v1)

**作者:** Stephen Arndt `[一作]` (Carnegie Mellon University), Michael Zlatin `[通讯]` (Pomona College)

**通讯引用:** 19 | [OpenAlex ID](https://openalex.org/A5050694688)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出多矩阵交叉着色问题的多项式时间近似算法。

**💡 创新点**

首次构造性实现2-近似(两矩阵)和O(k^2)近似(多矩阵)的多项式时间算法，消除对拓扑Hall定理等非构造方法的依赖。

**🔧 技术方法**

利用新的可灵活分解(matroidal flexible decomposition)结构，将问题转化为图着色，并使用随机化逼近技术实现FPRAS。

**📊 数据集**

无实验数据集，主要在理论上证明算法性能。

**📈 对比分析**

与已有的O(k log n)贪心算法比较，得到更好的常数近似比，且在大χ的情况下可获得FPRAS。

**⚠️ 局限性**

仅针对通用矩阵；对特定结构的矩阵尚无更优算法，且k大时近似比仍为多项式。

---

## 554. Deep Image Clustering Based on Curriculum Learning and Density Information

**arXiv ID:** 2604.03306 | [PDF](https://arxiv.org/pdf/2604.03306v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 555. LP-GEMM: Integrating Layout Propagation into GEMM Operations

**arXiv ID:** 2604.04599 | [PDF](https://arxiv.org/pdf/2604.04599v1)

**作者:** César Guedes Carneiro `[一作]` (Instituto de Computação), Sandro Rigo `[通讯]` (Instituto de Computação)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对依赖的多次 GEMM 操作，提出 LP-GEMM 通过在 GEMM 内部拆分成 init、mid、end 三个核，允许在连续 GEMM 调用间保持布局，从而消除重复的数据打包与解包。

**💡 创新点**

创新点在于将 BLAS GEMM 关键路径拆分为布局传播的三个阶段，并在中间核中跳过重复打包，实现跨 GEMM 的布局传递；同时提供了统一的微内核设计，兼容 AVX‑512 与 RISC‑V。

**🔧 技术方法**

使用了 OpenBLAS 的 GEMM 内核改造、微内核重写、布局传播机制、以及针对 LLaMA‑3.2 推理的 C++ 实现；技术层面涉及 AVX‑512、RVV 1.0、数据打包、微块化、并行循环展开。

**📊 数据集**

实验数据来源于 gemmbench 基准集以及 LLaMA‑3.2（1B 参数）模型的输入与权重文件。

**📈 对比分析**

在单次 GEMM、Attention 层以及三重连续 GEMM 的基准中，将 LP‑GEMM 与 OpenBLAS、BLIS、MKL、oneDNN、FlashGEMM 进行对比；结果显示 LP‑GEMM 在 x86 上相较 OpenBLAS 平均提升 2.25×，在 RISC‑V 上提升 5×，并能逼近或匹敌商用库性能。

**⚠️ 局限性**

局限性包括：不符合标准 BLAS 接口（需额外参数、输出布局不同）；仅针对 GEMM 设计，无法直接处理卷积等其它算子；目前实现仅支持 x86/AVX‑512 与 RISC‑V/RVV；在极大矩阵或非标准 stride 情形下仍有实现复杂度与性能损失。

---

## 556. Performance Evaluation of Subroutines Call in PHP

**arXiv ID:** 2604.03600 | [PDF](https://arxiv.org/pdf/2604.03600v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e`

---

## 557. ECG Biometrics with ArcFace-Inception: External Validation on MIMIC and HEEDB

**arXiv ID:** 2604.04485 | [PDF](https://arxiv.org/pdf/2604.04485v1)

**作者:** Arjuna Scagnetto `[一作]` `[通讯]`, Arjuna Scagnetto

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本研究构建了基于Inception‑v1 1D网络与ArcFace损失的ECG人脸识别模型，并在多域、跨时间、大规模数据上进行统一评估。

**💡 创新点**

创新点在于：①提出了可复现的四种大规模评估协议（闭集对比、规模分析、时间压力测试和重排）；②在公开最大规模ECG数据库（MIMIC‑IV‑ECG、HEEDB）上实现了多域外验证；③在长达五年时间跨度内系统评估身份信息随时间衰减的行为；④探讨了多种重排与置信度校准方法的联合效果。

**🔧 技术方法**

使用技术包括：1D Inception‑v1主干网络；ArcFace分类头；L2‑归一化嵌入；余弦相似度检索；Score‑norm（Z, S, AS, C, T）、Diffusion重排、AQE/αQE扩展；二元Logistic回归置信度校准；Python+PyTorch实现，GPU训练。

**📊 数据集**

采用了三大数据集：内部ASUGI（164k ECG，53k患者，4.5k样本/患者上限），公开MIMIC‑IV‑ECG（231k ECG，64k患者）和HEEDB（385k ECG，119k患者），并从中构造了多种实验子集（MIMIC‑GC、HEEDB‑GC、MIMIC‑TST、HEEDB‑TST、HEEDB‑scale、HEEDB‑RR）。

**📈 对比分析**

比较方法：在留一法闭集检索中计算Rank@K、TAR@FAR；在规模/时间压力测试中记录不同gallery大小或时间间隔下的Rank@1；在重排实验中评估Rank@1、Rank@5、Rank@10、ACC@0.5、Brier、ECE、Cov@0.90、Err@0.90。性能：在内部数据Rank@1最高达0.95；在MIMIC‑GC和HEEDB‑GC分别为0.829和0.688；随着gallery增大或时间延长，Rank@1逐步下降（如HEEDB‑TST 1年0.686→5年0.556）。重排中AS‑norm在HEEDB‑RR上提升Rank@1至0.8005，其他方法效果有限。

**⚠️ 局限性**

局限性：①评估仅采用闭集留一法，未涉及开放集拒绝和阈值调优；②内部外部数据的筛选不一致，导致域移位效应难以单独量化；③置信度校准仅在GC和HEEDB‑RR上验证，跨域泛化未知；④缺乏诊断标签对性能的细致剖析，未来可进一步探究。

---

## 558. Language Scent: Exploring Cross-Language Information Navigation

**arXiv ID:** 2604.03604 | [PDF](https://arxiv.org/pdf/2604.03604v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 559. Position: Science of AI Evaluation Requires Item-level Benchmark Data

**arXiv ID:** 2604.03244 | [PDF](https://arxiv.org/pdf/2604.03244v1)

**作者:** Han Jiang `[一作]` (Johns Hopkins University), Ziang Xiao `[通讯]` (Johns Hopkins University)

**通讯引用:** 1089 | [OpenAlex ID](https://openalex.org/A5083645580)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文主张在AI评估中引入逐项级别数据，并构建了一个统一的公开仓库，以提升评估的有效性与透明度。

**💡 创新点**

创新点在于提出并实现了“OpenEval”这一大规模、可扩展的逐项数据存储方案，并通过具体案例展示了逐项分析如何揭示基准设计的缺陷与模型能力的细粒度差异。

**🔧 技术方法**

主要技术包括经典测验理论（CTT）和项目因子分析（IFA）以及基于SVD和GLRM的高维因子提取，辅以GPT-5等大语言模型进行子结构解释。

**📊 数据集**

使用的数据集包括HELMP Classic、HELMP Capabilities以及OpenLLM Leaderboard v2，覆盖超过225k个条目、8M条模型响应。

**📈 对比分析**

通过逐项难度、区分度、因子负载等指标与外部基准（如MMLU、OEIS）进行对照验证，结果表明逐项分析能有效识别基准饱和、数据污染与构念失真问题，从而为模型改进和基准更新提供实证依据。

**⚠️ 局限性**

局限性主要在于当前仓库仍依赖已有公开基准，缺乏对新的自动生成基准的广泛覆盖；此外，因子解释主要基于自动化文本生成，存在主观性与偏差风险。

---

## 560. Benchmarking Multilingual Speech Models on Pashto: Zero-Shot ASR, Script Failure, and Cross-Domain Evaluation

**arXiv ID:** 2604.04598 | [PDF](https://arxiv.org/pdf/2604.04598v1)

**作者:** Hanif Rahman `[一作]` `[通讯]` (Independent Researcher), Hanif Rahman (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对 Pashto 语言的多语言 ASR 模型进行系统评测，涵盖零射、脚本失败、跨域微调等多维度，提供首个可复现的公共基准。

**💡 创新点**

创新点包括：①首次发布可复现的 Pashto ASR 评测基准；②揭示 Whisper 在 Pashto 上的脚本替代失效模式；③通过字符级误差分层识别 Pashto 独有音素的错误热点；④系统比较跨域微调效果与数据增强对性能的影响；⑤提出五项关键研究优先级。

**🔧 技术方法**

使用 Whisper（7 种尺寸）、MMS-1B、SeamlessM4T‑v2‑large、OmniASR‑CTC‑300M 等模型；采用贪婪解码、强制语言 token、Unicode NFC+Kashida 正则化、脚本识别审计、Bootstrap 置信区间等技术进行评估。

**📊 数据集**

评估数据集包括 FLEURS Pashto 测试集（512 句）和 Common Voice 24 过滤版（13,643 句）两份读音数据；同时对五个公开微调模型进行跨域测试。

**📈 对比分析**

通过 deterministic pipeline 计算 WER/CER 并补充脚本准确率进行比较；零射模型最高 WER 超过 400%，最佳零射为 SeamlessM4T 39.7% WER；微调模型在内部测试 14% WER 的基础上，跨域表现升至 32–59%；采用数据增强的模型实现跨域不降性能。

**⚠️ 局限性**

局限性：仅覆盖读音数据，缺乏自发/广播/电话语音评估；脚本识别启发式方法仅适用于大多数情况；未评估 XLS‑R、HuBERT、WavLM 的 0‑shot 能力；缺乏开源 TTS、词表与 G2P 支持。

---

## 561. Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models

**arXiv ID:** 2604.04575 | [PDF](https://arxiv.org/pdf/2604.04575v1)

**作者:** Arian Komaei Koma `[一作]` (Sharif University of Technology), Mohammad Hossein Rohban `[通讯]` (Sharif University of Technology)

**通讯引用:** 3621 | [OpenAlex ID](https://openalex.org/A5041967349)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 Stable Diffusion 1.4 进行粗粒度概念去学实验，重点去除裸体概念，并使用 T2I-CompBench++、GenEval、I2P 等基准评估去学方法的效果。

**💡 创新点**

揭示去学有效性与生成语义一致性之间的权衡，指出现有评测仅关注去学成功率，忽视了对生成模型语义结构的破坏。

**🔧 技术方法**

采用多种后置去学技术（ACE、ADV、ESD、EraseDiff、FMN、SPM、Salun、Scissorhands、UCE、MACE、RECELER、EAP、SAFREE、RECE、ResAlign-u/s、RACE），并通过 UAR、RA、FID、CLIP、BVQA 等指标进行评估。

**📊 数据集**

使用 Stable Diffusion 1.4 的原始预训练数据、I2P 骨干去学评测集、T2I-CompBench++、GenEval、SIX-CD、MS-COCO 10K 等公开数据集。

**📈 对比分析**

通过对比 UA、RA、FID、T2I-CompBench++/GenEval 语义一致性指标，发现强去学方法（如 EraseDiff、Scissorhands）虽 UA 高但造成显著的空间与属性绑定损失，平均 FID 远高于基线；相反 ACE、SPM 等方法保持了大部分语义一致性但 UA 较低。

**⚠️ 局限性**

存在去学安全与生成实用性之间的不可避免权衡，现有评估忽略了语义结构完整性，导致模型在去学后出现空间/属性失配甚至流形崩塌，需要在去学目标中加入语义保留约束。

---

## 562. SLaB: Sparse-Lowrank-Binary Decomposition for Efficient Large Language Models

**arXiv ID:** 2604.04493 | [PDF](https://arxiv.org/pdf/2604.04493v1)

**作者:** Ziwei Li `[一作]` (University of Science and Technology of China), Yi Kang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 3644 | [OpenAlex ID](https://openalex.org/A5101941645)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大语言模型中的线性层权重进行稀疏‑低秩‑二值化三分解，构成SLaB压缩框架；

**💡 创新点**

创新点在于将稀疏、低秩与二值化矩阵三者同时组合，并利用激活感知分数进行一次性无训练剪枝；

**🔧 技术方法**

采用交替优化策略、Hadamard乘积、截断SVD以及阈值化裁剪，整体实现一次性压缩；

**📊 数据集**

使用C4训练集的128条长度2048的校准数据、WikiText-2文本数据和LM‑Eval‑Harness中的ARC‑C/E、BoolQ、HellaSwag、PIQA、RTE、WinoGrande等零-shot任务；

**📈 对比分析**

与SparseGPT、Wanda等一拍即合剪枝方法以及2:4、4:8结构化稀疏进行对比，实验显示在50%压缩率下SLaB在WikiText‑2 perplexity下降约36%，零-shot平均准确率提升约9%，显著优于基线；

**⚠️ 局限性**

局限性包括仅在线性层实施、低秩仅取rank‑1、缺乏大规模模型验证以及对非线性层和其他模型架构的适用性未做深入探讨。

---

## 563. ProtoGuard-SL: Prototype Consistency Based Backdoor Defense for Vertical Split Learning

**arXiv ID:** 2604.03595 | [PDF](https://arxiv.org/pdf/2604.03595v1)

**作者:** Yuhan Shui `[一作]` (Wenzhou-Kean University), Zhiqiang Gao `[通讯]` (Wenzhou-Kean University)

**通讯引用:** 24535 | [OpenAlex ID](https://openalex.org/A5100612875)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为 ProtoGuard‑SL 的服务器端防御方案，用于垂直拆分学习中对嵌入空间后门攻击的检测与过滤。

**💡 创新点**

创新点在于利用类条件一致性原型（class‑conditional prototype consistency）将嵌入映射到相对空间，并结合分布无关的共形过滤（conformal filtering）实现对异常嵌入的精准识别，显著提高对隐蔽后门的鲁棒性。

**🔧 技术方法**

核心技术包括：1) 类原型构建（使用坐标中位数求稳健原型）；2) 原型一致性变换（计算每个嵌入与所有类原型的余弦相似度向量）；3) 类条件偏差评分与共形过滤；4) 训练时对嵌入进行实时筛选，保持正常样本的完整性。

**📊 数据集**

实验数据集涵盖图像分类（CIFAR‑10、SVHN）与文本营销（Bank Marketing），验证了方法的通用性。

**📈 对比分析**

与差分隐私（DP）、模型剪枝（MP）、对抗神经元剪枝（ANP）以及 VFLIP 等基线方法相比，ProtoGuard‑SL 在三大数据集和三种后门攻击场景下均实现了接近或优于基线的主模型准确率（MA），同时将攻击成功率（ASR）从 0.8–0.9 降至 0.03–0.08，展示了显著的性能提升。

**⚠️ 局限性**

局限性包括对阈值 α 的敏感性（需要手动调参以平衡精度与安全），以及在极端攻击或大规模客户端环境下仍可能出现微小性能衰减。

---

## 564. LOGER: Local--Global Ensemble for Robust Deepfake Detection in the Wild

**arXiv ID:** 2604.03558 | [PDF](https://arxiv.org/pdf/2604.03558v1)

**作者:** Fei Wu `[一作]` (Shanghai Jiao Tong University), Fengjun Guo `[通讯]` (INTSIG Information)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研发了 LOGER，结合局部‑全局双分支集成方法检测深度伪造。

**💡 创新点**

创新点在于多分辨率异构视觉基础模型主干与 MIL top‑k 聚合相结合，并在 logit 空间进行融合，显著提升稳健性与泛化能力。

**🔧 技术方法**

采用视觉基础模型、Patch‑MIL、Focal Loss、多源降质增强、logit‑space 融合与水平翻转 TTA 等技术。

**📊 数据集**

利用 HydraFake、FaceForensics++、DF40、Celeb‑DF、ScaleDF 等公开数据集以及 NTIRE 2026 训练集进行训练与评估。

**📈 对比分析**

与 Effort、GenD、CLIP‑ViT‑L 等基线对比，LOGER 在跨数据集和 NTIRE 2026 公开/私有测试集上分别获得 0.8901/0.8824 的 AUC，排名第二。

**⚠️ 局限性**

极端降质（如高噪声、严重模糊或暗化）仍导致误判，难以恢复深度伪造与真实之间的区分。

---

## 565. Rethinking Model Efficiency: Multi-Agent Inference with Large Models

**arXiv ID:** 2604.04929 | [PDF](https://arxiv.org/pdf/2604.04929v1)

**作者:** Sixun Dong `[一作]`, Qi Qian `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了视觉语言模型（VLM）中输出 token 长度对推理延迟的影响，并提出一种多代理推理框架，利用大模型生成短响应、从小模型迁移推理 token，并通过互相验证提升性能与效率。

**💡 创新点**

创新点：①系统性证明输出 token 数量是 VLM 推理效率的关键因素；②提出“多代理推理”（MAI）框架，结合互相验证与推理转移；③通过理论分析说明大模型的自注意力可抑制小模型推理中的噪声；④实现稀疏推理转移，只传递关键 token 进一步减少推理成本。

**🔧 技术方法**

技术细节：基于 Qwen3-VL 与 InternVL3.5 VLM；使用 Prompt 控制输出长度（Simple、Explain、Reasoning、2‑stage 结构）；采用 vLLM 作为高效推理引擎；使用 LMMs‑Eval 进行多任务评估；实现互相验证、推理转移与稀疏 token 选择；对自注意力差异做理论证明。

**📊 数据集**

数据集：POPE、MMBench、MMMU、RealWorldQA、ChartQA、InfographicVQA 等六类多模态评测集，覆盖是/否、单选、多选、开放式问答等多种输出格式。

**📈 对比分析**

与基线对比：在 2B、4B、8B、14B 模型上进行横向比较；大模型在 Simple（短输出）下往往性能更好且推理更快；MAI 框架在保持或提升准确率（如 ChartQA 86.52%）的同时，比全推理（8B‑R）缩短 2.4× 以上，甚至比完整推理转移快 1.76×–8.05×；在 2‑stage 结构下消除小模型的格式化错误。

**⚠️ 局限性**

局限性：①未考虑 GPU 内存占用与批量调度的真实约束；②对小模型推理的噪声控制依赖于 token 选择阈值，可能对不同任务产生偏差；③实验仅覆盖现有 VLM 体系，缺乏对新型跨模态模型的验证；④需要手工设计不同 Prompt 与 2‑stage 结构，对不同任务的迁移性有限。

---

## 566. The Indra Representation Hypothesis for Multimodal Alignment

**arXiv ID:** 2604.04496 | [PDF](https://arxiv.org/pdf/2604.04496v1)

**作者:** Jianglin Lu `[一作]` (Northeastern University), Yun Fu `[通讯]` (Northeastern University)

**通讯引用:** 31698 | [OpenAlex ID](https://openalex.org/A5005819096)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Indra Representation hypothesis，将单模模型的表示转化为全局相对关系特征

**💡 创新点**

首次将印度之网哲学与V-加深的Yoneda嵌入结合，证明该表示唯一、完整且保持结构

**🔧 技术方法**

利用V-enriched Yoneda embedding和角距离实现Indra表示，评估跨模态、跨模型的鲁棒性

**📊 数据集**

在CIFAR-10/100、Office‑Home、MS‑COCO、NOCAPS、TIMIT等公开数据集上实验

**📈 对比分析**

对比原始嵌入与Indra表示，使用Top‑k检索/线性探针指标，Indra表示在多种模态和模型上均实现显著性能提升

**⚠️ 局限性**

计算复杂度为O(n²d)，内存为O(n²)，对大规模数据存在扩展瓶颈

---

## 567. InsightBoard: An Interactive Multi-Metric Visualization and Fairness Analysis Plugin for TensorBoard

**arXiv ID:** 2604.03323 | [PDF](https://arxiv.org/pdf/2604.03323v1)

**作者:** Ray Zeyao Chen `[一作]` (University of Florida), Christan Grant `[通讯]` (University of Florida)

**通讯引用:** 332 | [OpenAlex ID](https://openalex.org/A5070860161)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 InsightBoard 插件，将多指标同步监控与子组公平性诊断集成在 TensorBoard 中；

**💡 创新点**

将公平性诊断实时嵌入训练仪表盘，并提供无额外数据存储的多指标、子组同步可视化；

**🔧 技术方法**

采用 TensorBoard 插件架构、EventMultiplexer 事件处理、滑动窗口采样、前端 TypeScript/D3.js/React、交叉视图链接等技术；

**📊 数据集**

在 YOLOX 模型训练的 BDD100k 自动驾驶数据集上进行实验；

**📈 对比分析**

与传统单指标 TensorBoard 对比，案例显示即使聚合性能高仍可在训练早期揭示显著子组差距，系统开销低（日志解析 2–3s，公平度计算 50–100ms，前端响应 <500ms）；

**⚠️ 局限性**

仅支持训练时实时监控，需手动设置公平性权重；对极大规模多子组时开销增长；缺乏自动化优化与多任务（如语言模型）支持。

---

## 568. A Model of Understanding in Deep Learning Systems

**arXiv ID:** 2604.04171 | [PDF](https://arxiv.org/pdf/2604.04171v1)

**作者:** David Peter Wallis Freeborn `[一作]` (Northeastern University), David Peter Wallis Freeborn `[通讯]` (Northeastern University)

**通讯引用:** 20512 | [OpenAlex ID](https://openalex.org/A5050925281)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种“系统性理解”（systematic understanding）的理论框架，定义其四个核心条件，并用该框架评估深度学习模型是否具备对目标系统属性的真正理解；随后通过两类具体实验（1）训练神经网络恢复环面拓扑结构（不直接给出拓扑标签），（2）比较基线时间序列拟合与基于开普勒轨道的模型对行星观测数据的预测，来验证模型在不同层次上的理解表现。

**💡 创新点**

创新点在于：① 将理解拆解为可度量的“模型、追踪、桥接、推导”四项，构建了一个非人类中心、可在人工智能与自然智能之间通用的定义；② 提出了“裂解理解假设”（Fractured Understanding Hypothesis），指出深度学习系统往往满足最低度量的理解，却缺乏统一的符号化、可归约与普适性；③ 通过具体的几何与动力学示例，展示深度学习在仅靠局部监督即可捕捉全局结构的潜力。

**🔧 技术方法**

技术方法主要为：基于多层感知机（MLP）和ReLU激活的前馈神经网络；使用均方误差损失、Adam优化；对连续标量场进行拟合后通过Marching Cubes算法提取等值面并计算拓扑不变量；在行星运动案例中，采用两种桥接策略——直接时间序列拟合与基于轨道参数的几何模型；并通过对测试集的预测误差与拓扑特征一致性评估模型表现。

**📊 数据集**

数据集包括：① 在[-3.5,3.5]³空间内采样的约262k个点，构成环面标量场F(x,y,z)的训练样本；② 以泰科·布拉赫观测记录为基础的行星位置时间序列（约12个对抗观测点和10个三角化约束），用于训练时间序列与轨道模型。

**📈 对比分析**

比较方法：对环面案例，评价重建等值面的欧拉特征与实际值是否一致（判定拓扑等价）；对行星案例，分别计算基线与Keplerian模型在训练集与外推时间点上的均方根误差（RMSE），并对预测的天文角度与观测值的角度误差进行统计。实验结果显示：基线模型在训练集上误差极小，但在外推时误差显著上升；Keplerian模型在训练集误差略高但在外推时误差显著下降，且在物理上更具解释性。

**⚠️ 局限性**

局限性包括：① 框架聚焦于结构性理解，未系统评估因果或归约理解的维度；② 实验示例规模较小，缺乏对大规模现实任务的验证；③ 对桥接原则的定义仍较模糊，可能导致不同实现者对“理解”的判定差异；④ 未对模型压缩与泛化的数学关系给出定量证明；⑤ 论文主要讨论前馈ReLU网络，对循环或自注意力模型的适用性尚待进一步研究。

---

## 569. Structural Impossibility of Antichain-Lattice Partial Information Decomposition

**arXiv ID:** 2604.03869 | [PDF](https://arxiv.org/pdf/2604.03869v1)

**作者:** Aobo Lyu `[一作]` (Washington University in St. Louis), Netanel Raviv `[通讯]` (Washington University in St. Louis)

**通讯引用:** 948 | [OpenAlex ID](https://openalex.org/A5048888817)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文通过构造特定的可实现原子模型，证明在多源情况下，反链格（antichain lattice）无法唯一确定多源互信息，提出了三源边界案例——系统信息分解（SID），并在该框架下给出可操作的冗余定义，进一步揭示了PID中无法仅靠冗余公理解决的不一致性。

**💡 创新点**

创新点在于：①首次给出反链格表示的根本代表性局限性，并用对偶系统构造了不可重构的反例；②提出SID作为在三源情形下可自洽的替代分解；③指出需要在原子标签之外引入关系结构（如超图）以实现完整的多源信息分解。

**🔧 技术方法**

主要技术包括：基于潜在变量构造的反链可实现原子模型、对偶系统（含额外全局约束）的构造、Gács–Körner公共信息的多元化定义、以及针对反链格的线性约束和不等式推理。

**📊 数据集**

使用的是人工生成的离散随机变量（如伯努利位及其XOR组合），没有使用公开的真实数据集。

**📈 对比分析**

文章未进行实验性比较；通过理论证明表明即便满足所有PID公理，反链格下的原子仍无法唯一恢复互信息，说明传统PID方法在信息复合度量上的局限性。

**⚠️ 局限性**

局限性在于：①研究范围局限于理论分析，缺乏对实际数据的实验验证；②提出的SID仅适用于三源系统；③未给出完整的可实现多源关系型分解框架，仅提出方向。

---

## 570. MolDA: Molecular Understanding and Generation via Large Language Diffusion Model

**arXiv ID:** 2604.04403 | [PDF](https://arxiv.org/pdf/2604.04403v1)

**作者:** Seohyeon Shin `[一作]` (Gwangju Institute of Science and Technology), Mansu Kim `[通讯]` (Gwangju Institute of Science and Technology)

**通讯引用:** 709 | [OpenAlex ID](https://openalex.org/A5046295159)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出MolDA，一种将离散扩散模型替代传统自回归（AR）骨干的多模态分子框架，能够通过双向去噪实现分子生成、标注与属性预测；

**💡 创新点**

创新点包括：①使用混合图编码器（GINE+TokenGT）与Q‑Former实现图与语言的对齐；②将Molecular Structure Preference Optimization（MolPO）重新定义为扩散场景下的奖励机制；③为不同任务设计特定的采样策略（全序列扩散与块级去噪）以提升全局结构一致性；

**🔧 技术方法**

采用的技术包括：离散扩散语言模型LLaDA‑8B‑Instruct、SELFIES分子表示、混合图编码器、Q‑Former、MolPO、LoRA微调、域自适应分词、任务特定采样；

**📊 数据集**

使用的数据集涵盖三千三百万条指令调优样本（Mol‑LLM、SMolInstruct、Mol‑Instructions、ChEBI‑20、PubChem）以及MoleculeNet的属性预测数据；

**📈 对比分析**

与六个7‑8B自回归基线（Mol‑LLM、ChemDFM、LlaSMol、Galactica、MolT5、3D‑MoLM）对比，MolDA在SIDER AUROC、HIV、HOMO等属性预测任务中取得最佳或接近最佳成绩，在反应预测任务中仅次于最佳模型，但在文本生成与标注任务上略逊于自回归模型；

**⚠️ 局限性**

局限性：在自然语言生成质量上仍落后于自回归模型；扩散推理耗时较长；在生成与标注任务中提升有限，仍需进一步完善全局约束建模与高效推理策略。

---

## 571. Automated Segmentation and Tracking of Group Housed Pigs Using Foundation Models

**arXiv ID:** 2604.03426 | [PDF](https://arxiv.org/pdf/2604.03426v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 572. HVG-3D: Bridging Real and Simulation Domains for 3D-Conditional Hand-Object Interaction Video Synthesis

**arXiv ID:** 2604.03305 | [PDF](https://arxiv.org/pdf/2604.03305v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 573. Vero: An Open RL Recipe for General Visual Reasoning

**arXiv ID:** 2604.04917 | [PDF](https://arxiv.org/pdf/2604.04917v1)

**作者:** Gabriel Sarch `[一作]` (Princeton University), Zhuang Liu `[通讯]` (Princeton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个全开源的视觉-语言推理体系，使用单阶段 RL 在 600K 样本、70 个多任务数据集（覆盖六大能力类别）上训练 VLM，并公开了数据、代码和模型。

**💡 创新点**

创新点包括：1）统一、透明的 RL 训练 recipe；2）多任务数据混合与任务平衡采样的设计；3）任务路由的奖励体系，能处理多种答案格式；4）对不同任务类别的链式思维行为进行系统化分析，展示跨任务正迁移。

**🔧 技术方法**

技术方法：单阶段 on‑policy RL（GSPO+GRPO）+多任务路由奖励；数据筛选与过滤（LLM 题目可验证性、答案标准化）；训练时使用 <think> 标记链式思维；开放式评价框架（lmms‑eval）和 LLM 评判。

**📊 数据集**

使用 70 个公开数据集，按六类（Chart & OCR、STEM、Spatial & Action、Knowledge & Recognition、Grounding/Counting/Search、Captioning & Instruction Following）挑选 600K 样本，形成统一的训练混合。

**📈 对比分析**

与基线（Qwen3‑VL‑8B‑Thinking、MiMo‑VL‑7B‑RL、LLaVA‑OV‑1.5‑RL 等）对比，取得 30 个基准中 24/30（+6.9‑13.6 分）及整体平均分 65.9‑66.0，明显优于现有 8B 级模型；在多任务 RL 训练上，单阶段 GSPO 超过 GRPO/DAPO，且在大部分类别中保持正迁移。

**⚠️ 局限性**

局限性：1）仅涵盖静态图像任务，未包含视频或多轮对话；2）行为分析为描述性，缺乏因果解释；3）实验聚焦于 7B‑9B 参数规模，未验证更大模型；4）最优任务划分或最小任务集合尚未确定。

---

## 574. PSY-STEP: Structuring Therapeutic Targets and Action Sequences for Proactive Counseling Dialogue Systems

**arXiv ID:** 2604.04448 | [PDF](https://arxiv.org/pdf/2604.04448v1)

**作者:** Jihyun Lee `[一作]` (POSTECH), Gary Geunbae Lee `[通讯]` (POSTECH)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种结构化认知行为疗法对话数据集及对应的辅导模型，通过显式拆分表面问题与自动负面想法并制定行动序列，实现更精准的自动负面想法识别与重构。

**💡 创新点**

创新点在于将自动想法与表面问题分离、设计基于阶段的治疗计划与可执行的行动序列，并利用模拟偏好学习提升同理心与计划遵循。

**🔧 技术方法**

技术上结合LoRA微调、结构化规划器、直接偏好优化（DPO）和GPT‑4o‑mini生成的合成对话。

**📊 数据集**

使用自构造的CBT结构化数据集（约6,425条对话），并以GPT‑4o‑mini为生成器，生成诊断与干预阶段对话。

**📈 对比分析**

在自动化评估与人工专家评估中，与多种开源及闭源基线（如Llama‑3.1‑8B、Claude、SmileChat、Camel、CBT‑LLM等）比较，本文模型在CTRS得分、SRS满意度和专家偏好上均显著优于对照组。

**⚠️ 局限性**

局限性包括对同理心的依赖不足、合成数据可能缺乏真实临床多样性、未在真实患者中验证，以及模型仍需临床监管与伦理审查。

---

## 575. UAV Control and Communication Enabled Low-Altitude Economy: Challenges, Resilient Architecture and Co-design Strategies

**arXiv ID:** 2604.04044 | [PDF](https://arxiv.org/pdf/2604.04044v1)

**作者:** Tianhao Liang `[一作]` (Harbin Institute of Technology), Tony Q. S. Quek `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 30411 | [OpenAlex ID](https://openalex.org/A5030858163)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出了面向低空经济的机载无人机（UAV）通信与控制协同设计框架，包含预飞行战略规划、机内自适应行动和系统级资源编排三层闭环架构，并通过案例验证其在单机与多机场景下的鲁棒性。

**💡 创新点**

创新点在于：①将通信与控制深度耦合，形成闭环协同体系；②提出基于3D信号强度分布的可行性航迹规划（GCS+数字孪生）；③使用预测控制、事件触发与自触发MPC实现自适应通信频率；④引入基于控制紧迫度的多机资源调度与MARL优化。

**🔧 技术方法**

采用的技术包括：图形凸集（GCS）与半正定规划进行轨迹规划；预测控制（PPC）、事件触发控制（ETC）、自触发MPC实现自适应更新；集成感知通信（ISAC）用于环境感知；控制导向的调度算法（AoI/VoI）与边缘多智能体强化学习进行资源分配；数字孪生与无线网络仿真工具。

**📊 数据集**

未公开具体真实数据集，主要使用仿真环境和基于真实基站部署信息的无线信号仿真（如3GPP Release‑19指标）进行实验验证。

**📈 对比分析**

与传统周期性MPC/单层控制相比，单机案例通信能耗降低51.8%；多机调度案例中，随着机队规模增大，平均控制误差上升，但通过上下文感知调度可显著降低误差，表明在有限通信资源下系统性能得到提升。

**⚠️ 局限性**

局限性包括：对预先已知的基站部署和无线环境假设较强；仿真结果缺乏大规模实地验证；规划层计算量大，实时性待进一步优化；多机调度对通信链路质量变化的鲁棒性仍需改进。

---

## 576. Affording Process Auditability with QualAnalyzer: An Atomistic LLM Analysis Tool for Qualitative Research

**arXiv ID:** 2604.03820 | [PDF](https://arxiv.org/pdf/2604.03820v1)

**作者:** Max Hao Lu `[一作]` (Harvard University), Sophia Blumert `[通讯]` (Harvard University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发并公开发布 QualAnalyzer Chrome 扩展，用于在 Google Workspace 里实现分段（atomistic）LLM 质性分析，记录每段输入、提示和输出，支持迭代、审计和协作。

**💡 创新点**

引入 atomistic LLM 分段分析范式，提供可追溯的审计轨迹；将此范式嵌入 Chrome 扩展，支持非程序员团队进行可视化、可比较的 LLM 质性分析。

**🔧 技术方法**

Chrome Extension + Google Sheets/Docs 作为工作空间；LLM API（OpenAI、Anthropic、Ollama）；prompt 构建、批处理、可靠性指标（Cohen κ、百分比一致）计算；面向模块化的 orchestrator。

**📊 数据集**

ASAP 公开学生作文（Grade 8 说服性写作）以及一份关于数据策划的半结构化访谈档案（Qualitative Data Repository）。

**📈 对比分析**

通过两次相同 prompt 的双通道运行，对比数值评分与人类评分差异，以及二元分类的 Cohen κ=1.0；对提及计数进行一致性检验，发现 83‑87% 一致；整体显示 atomistic 结果一致性高，但与人类评判在计数上存在系统性偏差。

**⚠️ 局限性**

仅能处理独立段落，无法支持跨段落综合的整体编码；工具提供可视化审计但需研究者主动审阅；审计不保证结果有效，仅记录过程。

---

## 577. RDEx-CMOP: Feasibility-Aware Indicator-Guided Differential Evolution for Fixed-Budget Constrained Multiobjective Optimization

**arXiv ID:** 2604.03708 | [PDF](https://arxiv.org/pdf/2604.03708v1)

**作者:** Sichen Tao `[一作]` (Tohoku University), Shangce Gao `[通讯]` (University of Toyama)

**通讯引用:** 9477 | [OpenAlex ID](https://openalex.org/A5010245958)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了RDEx‑CMOP，一种基于差分进化的约束多目标优化算法，结合ε‑可行性调度、SPEA2风格的适应度分配以及fitness‑guided current‑to‑pbest/1变异，在固定评估预算下实现快速可行性收敛与多样性保持；

**💡 创新点**

创新点在于将DESDE风格的DE骨干与ε‑约束处理、SPEA2适应度分配以及轻量级局部扰动相结合，形成了一套既能快速达到可行性又能在最终IGD上取得优异表现的紧凑框架，并通过动态p‑best窗口和Cauchy扰动进一步提升搜索方向性；

**🔧 技术方法**

主要技术包括差分进化（current‑to‑pbest/1 mutation）、ε‑约束处理、SPEA2式强度/密度适应度评估、距离基截断、Cauchy扰动、动态p‑best窗口和固定预算下的评估记录；

**📊 数据集**

使用了CEC 2025约束多目标优化基准（SDC1–SDC15，共15个问题），每个问题30次独立运行，最大评估次数为200,000，记录每200评估一次IGD；

**📈 对比分析**

通过官方中位数目标U‑score、Wilcoxon、Friedman等统计检验与Q_p（可行性优先的最终质量指标）和TTT（到目标时间）比较。RDEx‑CMOP在所有指标上均优于所有公开对手，获得最高总U‑score（58,456）和最佳平均排名（1.67），在可行性收敛和最终IGD表现上均显著领先；

**⚠️ 局限性**

局限性包括：每代需要O(N²M)的计算量，受评估成本主导；算法针对固定预算和约束多目标问题设计，可能对无约束或高维问题适应性不足；对超参数的敏感性分析有限；需在更多多样化基准上进一步验证鲁棒性。

---

## 578. Finite-Time Analysis of Q-Value Iteration for General-Sum Stackelberg Games

**arXiv ID:** 2604.04394 | [PDF](https://arxiv.org/pdf/2604.04394v1)

**作者:** Narim Jeong `[一作]` (Korea Advanced Institute of Science and Technology), Donghwan Lee `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2201 | [OpenAlex ID](https://openalex.org/A5100654316)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了两人通用和Markov游戏中Stackelberg Q值迭代的收敛性，并给出了有限时间误差上界。

**💡 创新点**

提出了针对Stackelberg结构的松弛策略假设，并将学习过程建模为可切换系统，通过上、下比较系统实现了误差界定。

**🔧 技术方法**

采用可切换系统理论、比较函数分析以及对Q值迭代的上/下界构造，证明了收敛速率和误差上界。

**📊 数据集**

使用了一个单状态、两行动的通用和Markov游戏进行数值实验，验证理论误差上界。

**📈 对比分析**

实验结果表明经验误差始终低于理论上界，并随迭代指数衰减；但全局松弛量导致上界保守，未能收敛至零。

**⚠️ 局限性**

局限在于假设策略确定、最优解唯一、需正的松弛量ε，且仅针对离散小规模游戏；未涵盖随机逼近或更一般的非确定性环境。

---

## 579. CommonMorph: Participatory Morphological Documentation Platform

**arXiv ID:** 2604.04515 | [PDF](https://arxiv.org/pdf/2604.04515v1)

**作者:** Aso Mahmudi `[一作]` (University of Melbourne), Ekaterina Vylomova `[通讯]` (University of Melbourne)

**通讯引用:** 1109 | [OpenAlex ID](https://openalex.org/A5055467011)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出并实现了CommonMorph平台，旨在通过协同工作流程，结合语言学家与说话者的参与，系统化、规模化地收集低资源与濒危语言的屈折形态数据。

**💡 创新点**

创新点包括：① 采用混合式工作流，将专家预设的屈折规则与机器学习的主动学习相结合；② 通过交互式界面支持说话者自助标注与校正；③ 引入多源预测（规则、神经模型、LLM）并通过主动学习动态提升建议质量；④ 支持跨语言数据重用与导入导出。

**🔧 技术方法**

核心技术包括：基于规则的屈折生成、2层LSTM神经屈折模型、few‑shot LLM提示、主动学习框架、图形化用户界面与后端.NET/PostgreSQL实现。

**📊 数据集**

数据集主要来自多语种案例研究：西班牙语/拉丁语、英语/德语、哈瓦米库尔德语、中央库尔德语、波斯语、阿拉伯语、斯瓦希里语、土耳其语等；此外还引用UniMorph和自建的本地词表与屈折表。

**📈 对比分析**

评估方式包括：字符错误率（CER）对比多种模型配置，用户体验问卷（学习时间、贡献时间、满意度等）。结果显示：① 语言学家提供的规则大幅降低CER，② 随着训练样本增多CER持续下降，③ 在资源丰富的语言中，LLM few‑shot可获得接近神经模型的表现。

**⚠️ 局限性**

局限性：① 规则编写仍需专业投入，尤其对未知语言；② LLM在完全变音/注音数据上的表现受限；③ 平台目前聚焦屈折形态，未涵盖音系、句法等；④ 需要网络环境，对低带宽地区不够友好；⑤ 需进一步支持无标准正字法与语音输入。

---

## 580. Reinforce to Learn, Elect to Reason: A Dual Paradigm for Video Reasoning

**arXiv ID:** 2604.04379 | [PDF](https://arxiv.org/pdf/2604.04379v1)

**作者:** Songyuan Yang `[一作]` (National University of Defense Technology), Nong Xiao `[通讯]` (Sun Yat-sen University)

**通讯引用:** 4021 | [OpenAlex ID](https://openalex.org/A5023506057)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了 RLER（Reinforce to Learn, Elect to Reason）双范式，先通过强化学习生成可验证的结构化证据，再在推理阶段通过证据加权投票与自检选择答案。

**💡 创新点**

创新点在于：①将证据生成和证据投票分离为训练与推理两阶段；②设计了三种新奖励（Frame‑Sensitive、Think‑Transparency、Anti‑Repetition）强化证据的可检索性、可读性和信息密度；③构建无训练的推理调度器，使用多样化采样、证据评分、冗余裁剪与单轮自检实现可靠答案选取。

**🔧 技术方法**

使用的技术包括：GRPO（Group‑Relative Policy Optimization）强化学习、LoRA 参数微调、Qwen2.5‑VL‑7B‑Instruct 语言模型、温度/top‑p 多样化采样、结构化输出解析、证据评分与加权投票、单轮自检（referee‑style）等。

**📊 数据集**

评估数据集共 8 个公开视频基准：VSIBench、VideoMMMU、VideoMME、TempCompass、MVBench、WildVideo、LVBench、LongVideoBench，覆盖多步推理、通用理解与长时序理解。

**📈 对比分析**

与多款开源及 RL‑基础 LMM（如 Qwen2.5‑VL、InternVL2.5、Video‑R1 等）比较，RLER 在所有基准上均超越基线，平均提升 6.3% 以上，且平均每个问题仅生成 3.1 条候选答案，计算成本与质量平衡良好。

**⚠️ 局限性**

局限性包括：仍依赖预定义的输出结构，可能在复杂多视角或非结构化问答中表现受限；自检和投票机制虽有效但增加了一定的推理延迟；在极端长视频或极低质量视频场景下，关键帧选择与证据质量可能进一步下降。

---

## 581. Hierarchical Point-Patch Fusion with Adaptive Patch Codebook for 3D Shape Anomaly Detection

**arXiv ID:** 2604.03972 | [PDF](https://arxiv.org/pdf/2604.03972v1)

**作者:** Xueyang Kang `[一作]` (University of Melbourne), Liangliang Nan `[通讯]` (Delft University of Technology)

**通讯引用:** 2492 | [OpenAlex ID](https://openalex.org/A5027616562)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种层次化点‑补丁异常评分网络，结合自监督补丁化、Patch‑Point交叉注意力和位置不变补丁码本，实现3D点云的异常检测与定位。

**💡 创新点**

创新点包括：①自适应多尺度补丁化与位置不变补丁码本；②Patch‑Point交叉注意力融合全局与局部特征；③伪异常增广配合门控调制，使模型对全局几何失配具有鲁棒性。

**🔧 技术方法**

技术核心包括3D UNet编码器、RoPE位置编码、跨尺度补丁匹配、Patch‑Point交叉注意力、负样本增广、门控调制以及回归+方向+BCE的联合损失。

**📊 数据集**

使用公开基准 Real3D‑AD、Anomaly‑ShapeNet 以及自制工业测试集（8件真实 CAD，含平面位移、角度错位等缺陷）。

**📈 对比分析**

与 PO3AD、R3D‑AD、ISMP 等 SOTA 进行对比，工业数据点级 AUC‑ROC 提升 >50%，Benchmark 平均提升约 7%/4%，整体性能优于或与最优方法持平。

**⚠️ 局限性**

局限性：对点云采样随机性敏感，缺乏对拓扑或语义部件信息的利用，未来需提升稳定性并结合语义/拓扑特征。

---

## 582. Grokking as Dimensional Phase Transition in Neural Networks

**arXiv ID:** 2604.04655 | [PDF](https://arxiv.org/pdf/2604.04655v1)

**作者:** Ping Wang `[一作]` (Institute of High Energy Physics), Ping Wang `[通讯]` (Institute of High Energy Physics)

**通讯引用:** 18717 | [OpenAlex ID](https://openalex.org/A5100402279)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过对梯度波动的有限尺寸扩展分析，研究了深度网络中“grokking”现象，证明其表现为梯度空间的维度相变；

**💡 创新点**

创新点在于揭示有效维度D在记忆到泛化跃迁时从亚扩散（D<1）过渡到超扩散（D>1），并证明该维度转变与网络拓扑无关，属于自组织临界现象；

**🔧 技术方法**

使用的技术包括阈值驱动的扩散更新（TDU‑OFC）捕捉梯度级联、有限尺寸扩展（FSS）获取s_max∼N^D、重抽样引导验证以及合成i.i.d.高斯梯度对照；

**📊 数据集**

实验主要基于极简的XOR布尔函数数据集（4个样本），并在补充研究中验证了ModAdd‑59任务；

**📈 对比分析**

通过将真实梯度与合成高斯梯度以及不同模型规模（N=81–2001）进行比较，发现D_pre≈0.90、D_post≈1.20、D_synth≈0.99，展示了显著的相变，并与训练/评估精度突变同步；

**⚠️ 局限性**

局限性在于仅在小规模、极简任务上验证，扩展至更大规模网络和复杂真实任务的普适性仍待进一步探索。

---

## 583. ComPrivDet: Efficient Privacy Object Detection in Compressed Domains Through Inference Reuse

**arXiv ID:** 2604.03640 | [PDF](https://arxiv.org/pdf/2604.03640v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 584. Poisoned Identifiers Survive LLM Deobfuscation: A Case Study on Claude Opus 4.6

**arXiv ID:** 2604.04289 | [PDF](https://arxiv.org/pdf/2604.04289v1)

**作者:** Luis Guzmán Lorenzo `[一作]` `[通讯]`, Luis Guzmán Lorenzo

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在使用大型语言模型（Claude Opus 4.6 等）对已混淆的 JavaScript 进行去混淆时，毒性标识符名称是否会在模型输出中持续出现，即便模型已正确理解语义；并探索提示框架对这一现象的影响。

**💡 创新点**

①验证提示（“请验证每个名称是否与数学匹配”）并未阻止毒性名称传播；②将任务从“去混淆”改为“从头实现”显著降低或消除名称传播；③即使在无语义契合的替代域中，错误名称仍会持续存在。

**🔧 技术方法**

采用 Claude Opus 4.6（及 Haiku 4.5 辅助检查）进行大规模实验；利用基于提示的四种变体、域梯度、混淆梯度、层梯度、任务框架等多维度设计；通过代码块级别自动评分、手工评审等方式进行结果分析。

**📊 数据集**

构造两类代码原型（物理力学力导图模拟和 A* 路径搜索），每类包含多种“毒药”pill（命名、常量、可读性等）；实验共 192 次运行，覆盖 50 条实验条件。

**📈 对比分析**

对照实验显示：
- 在 12 次验证提示实验中，所有运行均出现错误名称；
- 在任务框架实验中，“从头实现”将传播率从 100% 降至 0%（物理）或 20%（路径搜索）。
- 通过算法一致性检查确认结构保持不变。总体而言，任务框架改变显著提升命名准确率。

**⚠️ 局限性**

受限于仅测试两种模型（Opus、Haiku）和两类代码原型；实验未覆盖 GPT‑4o、Gemini 等模型；工具使用与单次推断的差异可能影响结果；标注过程为单人非盲评；实验主要在模拟数据，缺乏更广泛的真实代码样本。

---

## 585. Review and Evaluation of Point-Cloud based Leaf Surface Reconstruction Methods for Agricultural Applications

**arXiv ID:** 2604.03328 | [PDF](https://arxiv.org/pdf/2604.03328v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 586. Spatiotemporal Interpolation of GEDI Biomass with Calibrated Uncertainty

**arXiv ID:** 2604.03874 | [PDF](https://arxiv.org/pdf/2604.03874v1)

**作者:** Robin Young `[一作]` (University of Cambridge), Srinivasan Keshav `[通讯]` (University of Cambridge)

**通讯引用:** 10941 | [OpenAlex ID](https://openalex.org/A5090054353)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究了一种基于Attentive Neural Process的时空稀疏插值方法，用于填补NASA GEDI激光雷达在森林碳监测中的时间空隙。

**💡 创新点**

创新点在于将空间与时间对称处理，利用空间-时间基础模型嵌入实现空间-时间代替（space‑for‑time substitution），并提供经过校准的置信区间。

**🔧 技术方法**

使用了Attentive Neural Process架构、交叉注意力、变分推断、基于Tessera的空间-时间嵌入以及贝塔‑annealing的ELBO训练。

**📊 数据集**

采用了GEDI Level 4A AGBD数据（2019‑2023）以及Tessera的128维Sentinel‑1/2嵌入，三大生态区（哥伦比亚Guaviare、秘鲁Ucayali、澳大利亚Queensland）。

**📈 对比分析**

与Quantile Random Forest与XGBoost量化回归基线相比，ANP在log‑R²、RMSE、误差标准化残差等指标上表现最佳，且在被扰动区的置信区间覆盖率更接近理想值。

**⚠️ 局限性**

局限包括对Tile级别的扰动划分、对Tessera时序嵌入的依赖以及在高生物量地区传感器饱和导致精度下降。

---

## 587. CresOWLve: Benchmarking Creative Problem-Solving Over Real-World Knowledge

**arXiv ID:** 2604.03374 | [PDF](https://arxiv.org/pdf/2604.03374v1)

**作者:** Mete Ismayilzada `[一作]` (École Polytechnique Fédérale de Lausanne), Antoine Bosselut `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 5380 | [OpenAlex ID](https://openalex.org/A5088410008)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并构建了CREOWLVE基准，包含来自俄国智力游戏“What? Where? When?”的真实世界谜题，用于评估大语言模型的创意问题解决能力。

**💡 创新点**

创新点在于：①将创意问题与多领域真实知识结合；②将问题划分为创意与事实两类并提供错误分类体系；③提供双语（俄文与英文）高质量验证数据集，打破以往人工脑筋急转弯的局限。

**🔧 技术方法**

主要技术手段包括数据收集、内容过滤、机器翻译与人工校对、知识域与创意策略自动标注、以及基于Exact Match和LLM‑as‑Judge的评估框架；实验中对多种“思考”与“不思考”LLM进行对比。

**📊 数据集**

使用的数据集为2,061道创意谜题（包含2,413道经过过滤的样本，俄文/英文双语），覆盖34个知识域与多文化背景，约有2,061道创意与352道事实子集。

**📈 对比分析**

比较方法：在Exact Match和LLM‑as‑Judge两种指标下，对30多种前沿LLM（开源与闭源、不同思考强度）进行评测，结果显示即使是最高性能的思考模型，创意问题的准确率也低于30%，并与事实问题存在高达17%的性能差距。

**⚠️ 局限性**

局限性包括：①基准主要基于智力游戏谜题，可能与真实任务存在距离；②存在潜在数据泄露与文化偏差；③创意与事实划分与错误类型标签仍具主观性；④缺乏系统的人类对照实验，只能依赖历史游戏解答作为上限。

---

## 588. ROSClaw: A Hierarchical Semantic-Physical Framework for Heterogeneous Multi-Agent Collaboration

**arXiv ID:** 2604.04664 | [PDF](https://arxiv.org/pdf/2604.04664v1)

**作者:** Rongfeng Zhao `[一作]` (Tongji University), Jie Chen `[通讯]` (Tongji University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了ROSClaw框架，用于异构多机器人协同任务，整合了大型语言模型与物理执行的统一控制器，并在真实环境中完成了多机器人协作与舞蹈编排等任务。

**💡 创新点**

创新点包括：①三层语义-物理架构将LLM的高频语义规划与低频物理控制解耦；②e-URDF基物理防护机制和数字孪生仿真预检；③在线工具池与本地资源池实现跨平台抽象与经验累积；④闭环数据反馈促进持续学习。

**🔧 技术方法**

采用的技术包括：大型语言模型（LLM）+ 视觉-语言模型（VLM）；Isaac Lab 进行数字孪生仿真和正向动力学碰撞检测；e-URDF 物理约束；Online Tool Pool（SDK、API抽象）；Local Resource Pool（状态积累与技能库）；ROS与硬件驱动统一接口；多模态感知（RealSense、DINO-X）。

**📊 数据集**

未使用公开数据集，实验数据来自自建的智能家居环境（厨房、客厅）以及自定义多机器人任务场景，使用 RealSense 摄像头采集的视觉数据和机器人执行轨迹。

**📈 对比分析**

通过与传统手工编写/预编程多机器人协作方案比较，ROSClaw 在同一任务下实现了从用户指令到机器人执行的闭环流程，显著降低了人工调试时间（如舞蹈编排从传统数小时降至约3分钟），并保持了安全性与可执行性。实验结果显示任务成功率接近100%，但缺乏量化指标。

**⚠️ 局限性**

局限性：①仅在结构化或半动态环境中验证，未针对高频干扰、感知噪声或模型随机性进行系统化评估；②本地资源池未形成完整闭环学习链路，缺乏直接将经验用于在线策略优化；③对复杂不确定环境下的概率建模与动态避障研究不足。

---

## 589. LPC-SM: Local Predictive Coding and Sparse Memory for Long-Context Language Modeling

**arXiv ID:** 2604.03263 | [PDF](https://arxiv.org/pdf/2604.03263v1)

**作者:** Keqin Xie `[一作]` `[通讯]` (Independent Researcher), Keqin Xie (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为LPC‑SM的混合自回归架构，用局部注意力、双时序记忆、预测纠正和运行时控制分别处理短距精度、长距存储、误差修正和稀疏性；采用正交新颖传输（ONT）优化慢速记忆写入；并使用多头残差路由（mHC）提升模型表达力。

**💡 创新点**

将注意力分离为局部读、慢速记忆写、预测纠正三条路径；引入ONT仅增添新颖信息写入；通过可学习的稀疏控制和mHC实现内部资源动态分配，形成更易分析的多机制块。

**🔧 技术方法**

核心技术包括双时序记忆（快/慢状态）、正交新颖传输（ONT）、多头残差路由（mHC）、自适应稀疏控制、正则化辅助损失以及层级训练调度。

**📊 数据集**

使用Dolma3‑base（32.77M token, 2048长），OpenWebMath‑10k（16.38M token, 2048长）以及LongMino（24.58M token, 4096长）作为三阶段训练语料。

**📈 对比分析**

通过对照完整模型、去除各模块的消融实验，以及对比可学习稀疏控制与固定稀疏比例的连续训练，评估最终LM损失、token/s吞吐量和稀疏比例。结果显示：去除mHC导致Stage‑A损失从12.63升至15.13；可学习稀疏控制在Stage‑B比固定稀疏提高12.5%；Stage‑C长上下文训练后，损失下降至11.58，并在延迟标识符诊断上显著改进。

**⚠️ 局限性**

实验规模仅158M参数，处于欠拟合状态，导致部分模块（如预测纠正、ONT、停止头）对Stage‑A损失不显著；内存机制在不同指标下效果不一，未能完全证明其正面作用；缺乏大规模验证，需进一步扩展至1B级别验证可行性。

---

## 590. DIRECT: Video Mashup Creation via Hierarchical Multi-Agent Planning and Intent-Guided Editing

**arXiv ID:** 2604.04875 | [PDF](https://arxiv.org/pdf/2604.04875v1)

**作者:** Ke Li `[一作]` (Peking University), Xiang Chen `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过分层多代理框架DIRECT，将视频混剪任务建模为多模态一致性满足问题，实现从全局结构到细粒度视觉音频一致性的自动化剪辑。

**💡 创新点**

将视频混剪视为MMCSP，并提出屏幕编剧、导演和编辑三个层级代理，结合闭环验证与动态滑动窗口剪裁，实现专业级多模态一致性。

**🔧 技术方法**

多模态大模型 Qwen3‑VL‑8B‑Instruct、CLIP、RAFT、U2‑Net、音乐分析模型、图搜索、束搜索、闭环验证、动态滑动窗口剪裁等技术。

**📊 数据集**

Mashup‑Bench（4,000 分钟、64,000 个片段、10 首音乐、40 个测试案例）

**📈 对比分析**

与 T2V、MMSC、VideoAgent 对比，DIRECT 在视觉连续性、音频同步等六项指标上最高，主观评估中全局结构、局部协同、整体质量平均得分显著优于基线（如 Loc 5.3 对比 3.1）。

**⚠️ 局限性**

受限于大模型推理成本、对极端多样化素材的适应性不足，以及对超长片段或多源音乐的处理仍有提升空间。

---

## 591. Parameterized Approximation of Rectangle Stabbing

**arXiv ID:** 2604.04282 | [PDF](https://arxiv.org/pdf/2604.04282v1)

**作者:** Huairui Chu `[一作]` (University of California), Jie Xue `[通讯]` (New York University)

**通讯引用:** 1467 | [OpenAlex ID](https://openalex.org/A5101635746)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

暂无论文内容，无法描述

**💡 创新点**

暂无论文内容，无法描述

**🔧 技术方法**

暂无论文内容，无法描述

**📊 数据集**

暂无论文内容，无法描述

**📈 对比分析**

暂无论文内容，无法描述

**⚠️ 局限性**

暂无论文内容，无法描述

---

## 592. Evolutionary Search for Automated Design of Uncertainty Quantification Methods

**arXiv ID:** 2604.03473 | [PDF](https://arxiv.org/pdf/2604.03473v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 593. Rethinking Exploration in RLVR: From Entropy Regularization to Refinement via Bidirectional Entropy Modulation

**arXiv ID:** 2604.04894 | [PDF](https://arxiv.org/pdf/2604.04894v1)

**作者:** Hengrui Gu `[一作]` (North Carolina State University), Kaixiong Zhou `[通讯]` (North Carolina State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了RLVR中的受限探索问题，提出可解耦正负样本的AsymGRPO框架，实现对信息熵与噪声熵的精细化调节。

**💡 创新点**

创新点在于将策略熵概念化为信息熵与噪声熵，并通过可调参数β_pos、β_neg 对正负样本分别进行增益/抑制，从而实现熵的精细化与可控性。

**🔧 技术方法**

使用了RLVR、PPO、分组相对优势估计（GRPO）、熵正则化、以及自定义的AsymGRPO算法。

**📊 数据集**

使用了多项数学推理基准数据集，包括 MATH、AIME24/25、AMC23、OlympiadBench 等。

**📈 对比分析**

在与 REINFORCE、GRPO、GRPO 加熵正则化、Clip-higher、EntIncrease/Decrease 等多种对照实验中，AsymGRPO 在平均准确率上提升约2–3%，最高可达约60% 的平均成绩。

**⚠️ 局限性**

局限性：熵调节粒度仍受限于组级指标；β_pos 与 β_neg 为静态超参数，需人工调优；缺乏动态自适应调度机制，未来可进一步细化熵调节维度与自动化调整策略。

---

## 594. CAGMamba: Context-Aware Gated Cross-Modal Mamba Network for Multimodal Sentiment Analysis

**arXiv ID:** 2604.03650 | [PDF](https://arxiv.org/pdf/2604.03650v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 595. Adversarial Robustness of Deep State Space Models for Forecasting

**arXiv ID:** 2604.03427 | [PDF](https://arxiv.org/pdf/2604.03427v1)

**作者:** Sribalaji C. Anand `[一作]` (KTH Royal Institute of Technology), George J. Pappas `[通讯]` (University of Pennsylvania)

**通讯引用:** 35585 | [OpenAlex ID](https://openalex.org/A5029243115)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了基于时空状态空间模型Spacetime的时间序列预测在对抗攻击下的鲁棒性，提出了鲁棒优化框架并通过对抗训练提升模型抵抗力。

**💡 创新点**

首次将控制理论视角与对抗训练结合，对Spacetime模型在最坏情况下的预测误差给出解析上界，并揭示模型线性结构易受模型无关攻击。

**🔧 技术方法**

采用控制理论分析、Stackelberg对抗优化、Projected Gradient Descent、模型无关数据驱动攻击、对抗训练与实验评估。

**📊 数据集**

使用Monash电力消耗、交通、河流流量、美国出生率等多种时间序列基准。

**📈 对比分析**

与基线CNN/正则化检测器对比，鲁棒训练模型在正常MAE约10%提升，在对抗MAE约10%下降；模型无关攻击造成的误差比梯度攻击高33%。

**⚠️ 局限性**

受限于检测阈值设定与对抗预算，模型在极端对抗下仍易被误差爆炸；实验主要在单用户水平，未考虑多源或分布漂移。

---

## 596. Multirate Stein Variational Gradient Descent for Efficient Bayesian Sampling

**arXiv ID:** 2604.03981 | [PDF](https://arxiv.org/pdf/2604.03981v1)

**作者:** Arash Sarshar `[一作]` `[通讯]` (California State University), Arash Sarshar (California State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过将SVGD的吸引与排斥项拆分，并采用多速率积分（固定与自适应），提出多种多速率SVGD变体，用于更稳定、高效地采样高维、异方差、分多峰以及层级后验分布。

**💡 创新点**

创新点在于：①把SVGD视为分裂的粒子流，分别对吸引力与排斥力使用不同时间尺度；②设计了固定多速率和自适应误差控制的多速率SVGD；③在六大基准上系统评估，显示对复杂后验具有更好稳健性与成本效益。

**🔧 技术方法**

主要技术包括：Stein变分梯度下降（SVGD），分裂积分（Strang分裂）、多速率Euler积分、基于局部误差的自适应子步数控制、核 Stein 散度（KSD）、预测校准误差（ECE）等评估指标。

**📊 数据集**

使用的数据集/目标包括：50维高斯分布、多个2D合成目标（如香蕉、环、两月形等）、8峰混合高斯、UCI逻辑回归（breast cancer、ionosphere、spambase、a5a）、一隐藏层贝叶斯神经网络（Spambase、a5a）以及大规模层级逻辑回归（长尾组和均匀组）。

**📈 对比分析**

与传统SVGD、Strang分裂SVGD、SGLD、SGHMC等基线对比。实验显示，多速率SVGD在高维/异方差/多峰/层级目标上显著提升了分布匹配（KSD、均值对数密度）、预测性能（NLL、ACC、ECE）与混合效率（ESS），且成本（梯度/核评估、墙钟时间）更低，尤其是自适应多速率在局部刚性变化处能自动增细步长。

**⚠️ 局限性**

局限性包括：①自适应控制仅针对吸引力，排斥频率仍由固定计划决定；②仅使用一维ESS作为混合度量；③核交互仍是O(N²)计算，难以扩展到大粒子数；④缺乏针对多速率积分的理论稳定性和收敛性分析。

---

## 597. Learning-Based Fault Detection for Legged Robots in Remote Dynamic Environments

**arXiv ID:** 2604.03397 | [PDF](https://arxiv.org/pdf/2604.03397v1)

**作者:** Abriana Stewart-Height `[一作]` (Massachusetts Institute of Technology), Nikolai Matni `[通讯]` (University of Pennsylvania)

**通讯引用:** 2716 | [OpenAlex ID](https://openalex.org/A5052941508)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

采用离线深度学习自编码器检测四足机器人单腿损伤，并根据检测结果切换至相应的三足步态。

**💡 创新点**

创新点在于仅用无标注的健康数据训练无监督模型，能实时识别并定位单腿损伤，避免了传统需要标注或模型精确构造的限制。

**🔧 技术方法**

使用无监督自编码器（AutoEncoder）与MSE重建误差作为异常判定，并通过MixMaxScaler对传感器数据进行归一化。

**📊 数据集**

使用本团队采集的170条四足机器人前后步态数据集，包含5种状态（无损伤、左前腿、右前腿、左后腿、右后腿）和每条样本的4个传感器值。

**📈 对比分析**

方法仅训练健康数据后，在受损数据上计算重建误差并与阈值比较。实验显示LF缺失检测率为99.89%、RF缺失检测率为96.86%，非受损样本误报率约15%。

**⚠️ 局限性**

局限性包括：仅验证单腿缺失场景；模型需离线训练，无法即时适应新故障；阈值设定依赖经验，可能不适用于不同机器人或更复杂环境；未验证多腿并发损伤或不同步态下的表现。

---

## 598. Peoples Water Data: Enabling Reliable Field Data Generation and Microbial Contamination Screening in Household Drinking Water

**arXiv ID:** 2604.04240 | [PDF](https://arxiv.org/pdf/2604.04240v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 599. Demonstrating SIMA-Play: A Serious Game for Forest Management Decision-Making through Board Game and Digital Simulation

**arXiv ID:** 2604.04904 | [PDF](https://arxiv.org/pdf/2604.04904v1)

**作者:** Arka Majhi `[一作]` (Tampere University), Heli Peltola `[通讯]` (University of Eastern Finland)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

设计并演示了一款名为SIMA-Play的模拟木材管理桌游，该游戏通过欧式游戏机制让玩家在不确定的环境与市场条件下做出森林管理决策，并将玩家决策日志输入SIMA森林生长模型后，用交互式数据可视化（IDV）展示决策结果。

**💡 创新点**

创新点包括：①将复杂的森林生态与经济模型（SIMA）嵌入可操作的桌面游戏规则；②利用IDV即时反馈帮助玩家理解长期生态与经济权衡；③构建可与数字版互补的混合教学工具，开辟了森林管理教育的新路径。

**🔧 技术方法**

技术主要包括：欧式桌游设计框架、SIMA森林生长模拟模型、交互式数据可视化（如动态地图、仪表盘、信息图）以及多风险卡牌机制。

**📊 数据集**

使用的数据集为：SIMA模型的森林生长与碳、经济输出数据；玩家在游戏中产生的决策日志（种植、采伐、保险等），随后用于生成可视化报告。

**📈 对比分析**

比较方法：将玩家在游戏中的决策记录导入SIMA模型，得到量化的生态与经济指标，然后用IDV展示各玩家或策略的对比。该方法使玩家能够直观地看到不同决策对碳储存、经济收益和生态服务的长期影响；目前未给出数值性能评估，主要强调学习与系统思维提升。

**⚠️ 局限性**

局限性：①尚未进行系统的教育效果评估和长期学习跟踪；②模型与游戏之间的映射可能过度简化某些生态过程；③缺乏多样化的数字化交互版本，限制了在低资源环境中的可扩展性；④实验规模有限，难以验证在不同人群与情境下的普适性。

---

## 600. Undetectable Conversations Between AI Agents via Pseudorandom Noise-Resilient Key Exchange

**arXiv ID:** 2604.04757 | [PDF](https://arxiv.org/pdf/2604.04757v1)

**作者:** Vinod Vaikuntanathan `[一作]` (Massachusetts Institute of Technology), Or Zamir `[通讯]` (Tel Aviv University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文研究在两台 AI 代理在相互交互时，如何在保持可观测对话（日志）与“真实”交互分布完全不可区分的前提下，悄悄进行秘密会话。作者首先给出了共享密钥下的最优速率隐藏对话方案，然后提出了一种无共享密钥的隐式密钥交换方法，并设计了一种新的加密原语——伪随机噪声耐受密钥交换（PNR‑KE），通过“bundle sampler”实现低熵环境下的隐写。最终构建了完整的无键隐藏对话协议，证明其在理想条件下可实现任意可观测对话长度的隐蔽率与熵匹配。

**💡 创新点**

创新点包括：①将 LLM watermarking 与 steganography 结合，得到可在交互式对话中实现的高率隐藏对话；②提出 bundle sampler，突破传统提取式隐写只能在高熵消息上工作的限制，将错误率转化为可容忍的噪声；③定义并研究 PNR‑KE，首次把噪声耐受性与伪随机性结合到密钥交换中，为无键隐藏对话提供理论基础；④给出多种 PNR‑KE 的构造与严格的不可行性分析，展示该原语在现有加密构造中是必不可少的。

**🔧 技术方法**

主要技术手段包括：LLM watermarking 与 CGZ 隐写框架；剩余哈希定理与强随机提取器；bundle sampler 的统计设计与 BSC 模型分析；伪随机噪声耐受密钥交换（基于稀疏秘密 LPN、信息重组与后向误差纠正）；信息理论的熵估计与可观测分布保持。论文中对所有算法进行了信息论与多重混合技术的严谨证明。

**📊 数据集**

本文为理论性研究，未使用具体数据集；所有结果均为可计算机实现的数学构造与实验模拟，主要假设在大规模 LLM 生成模型（token 交互）与可观测日志上均成立。

**📈 对比分析**

对比传统的 LLM 隐写方案，作者的共享密钥方案实现了近最优隐写速率（可利用几乎全部熵），而无键方案在仅常数熵消息下仍能实现隐写，误码率被控制在可接受范围内；PNR‑KE 的构造虽然基于 LPN 的多项式时间攻击，但给出了准多项式安全级别，并对其不可行性做了详细讨论。实验或数值比较未给出，但理论上已证明在满足熵与公共随机性假设的前提下，隐写率可与对话长度线性匹配，安全性满足计算不可区分。

**⚠️ 局限性**

主要限制包括：①需要独立公共随机数（不可被模型或对话控制）；②需满足可识别熵假设，即足够多的消息具有已知最小熵；③PNR‑KE 的安全性依赖于 sparse‑secret LPN 等尚未成熟的假设；④对低熵消息的误码率为常数，需进一步纠错或多轮重传才能达到可接受安全水平；⑤实际部署时对 LLM 的具体分布与可观测日志的实现细节需进一步验证。

---

## 601. Merkle Tree Certificate Post-Quantum PKI for Kubernetes and Cloud-Native 5G/B5G Core

**arXiv ID:** 2604.04191 | [PDF](https://arxiv.org/pdf/2604.04191v1)

**作者:** Lakshya Chopra `[一作]` (coRAN Labs Private Limited), Vipin Kumar Rathi `[通讯]` (Ramanujan College)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文提出并实现了基于 Merkle Tree Certificates（MTC）的私有 PKI 架构，用于 Kubernetes 控制平面和 5G Core（QORE）网络中的 TLS 1.3 互联互认证，替代传统的 Post‑Quantum (PQ) X.509 证书和签名，减少链路上的签名负载和验证成本。

**💡 创新点**

创新点：
- 将 MTC 迁移到私有环境，设计 Kubernetes 及 5G Core 的完整证书生命周期与信任分发流程；
- 通过 landmark 模式实现无签名（仅哈希）证书验证，显著降低签名验证开销；
- 结合 cosigner、mirror、landmark distributor 等组件，实现透明性、可审计性和基于索引的快速吊销；
- 在 Go TLS 堆栈中实现 MTC 支持，并在 Intel i9‑12900 上测评，验证 landmark 验证 <2µs，TLS handshake 与经典方案相当。

**🔧 技术方法**

技术：
- Merkle Tree Certificates（MTC）规范（IETF 草案）
- Post‑Quantum 签名方案 ML‑DSA‑65、密钥封装 ML‑KEM‑768
- Kubernetes 自带的 CSR、证书签发及 Secrets
- Go 语言的 crypto/tls、crypto/x509 包
- 基于 HTTP API 的 MTCA、cosigner、mirror 与 landmark distributor
- 证书透明度 CT、SCT、索引吊销
- 5G SBA、OAuth2、OpenAPI、Free5GC + QORE

**📊 数据集**

数据集 / 实验环境：
- 采用 Intel i9‑12900（24 核）在本地 loopback 进行 TLS 1.3 handshake benchmark
- 对比场景包括经典 ECDSA P‑256、MTC standalone（16 leaf）和 MTC landmark（16/1024/4096 leaf）
- 证书尺寸基于 ML‑DSA‑65 与 Ed25519 公开密钥大小，测算链路数据量
- 通过 QORE（基于 free5GC 的 Post‑Quantum 5G Core）部署验证真实网络场景

**📈 对比分析**

比较方法与性能：
- 证书大小对比：PQ X.509 约 17.5KB，MTC landmark 仅 2.9KB（≈83% 缩减）
- 验证成本：MTC landmark <2µs vs 24µs（ECDSA）和 150µs×4（四次 PQ 签名）
- TLS handshake 时间：MTC landmark 563–715µs 与经典 ECDSA 678µs 对齐，无明显延迟
- 评估覆盖了链路开销、CPU 验证成本与整体握手时延，证明 MTC 在 PQ 环境下实现了高效安全

**⚠️ 局限性**

局限性：
- 需要部署额外组件（MTCA、cosigner、mirror、分发器），增加运维复杂度；
- 目前实现仅支持 Go TLS，其他语言需自行适配；
- landmark 需要周期性分发，若分发延迟导致无效 landmark，仍需 fallback 为 standalone；
- 对于极高频率的 NF‑to‑NF 调用，仍需评估 log 规模增长导致的索引查找成本；
- 研究基于 QORE，真实商业 5G Core 生产环境的兼容性与可扩展性仍待验证。

---

## 602. Beyond Crash-to-Patch: Patch Evolution for Linux Kernel Repair

**arXiv ID:** 2604.03851 | [PDF](https://arxiv.org/pdf/2604.03851v1)

**作者:** Luyao Bai `[一作]` (University of Illinois Chicago), Xiaoguang Wang `[通讯]` (University of Illinois Chicago)

**通讯引用:** 352 | [OpenAlex ID](https://openalex.org/A5100379563)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了基于syzbot的 Linux 内核修复生命周期数据集，并基于此开发了一种利用补丁演化历史指导的修复框架；

**💡 创新点**

首次将补丁演化历史作为监督信号引入自动修复，结合检索记忆和诊断顾问两层机制，使生成的补丁更符合审阅者的约束；

**🔧 技术方法**

使用检索式记忆层（相似 bug、修复模式、评审经验）和微调后的诊断顾问模型（Gemma-3/ LLaMA-3）驱动 LLM 代码生成，辅以编译反馈循环；

**📊 数据集**

利用 6,946 条 syzbot 记录中的 5,043 条已合并修复、约 5,000 条讨论线程以及对应的重现器，形成了 6,000 多个完整的 bug‑修复链；

**📈 对比分析**

在 100 条保留的演化历史样本中，记忆层提升 F1 21.8%；在 6 条时间上隔离的实时修复案例中，Advisor‑Guided（微调后 Gemma‑3）实现 5/6 正确修复、CodeBERTScore 0.91、编译率 100%；

**⚠️ 局限性**

评估样本极少（仅 6 条）且不涵盖多文件、复杂 RCU 重构等高级修复，模型对过度拟合和表面模式的依赖仍显著，且仍难以处理跨文件或高并发度的修复。

---

## 603. FAVE: Flow-based Average Velocity Establishment for Sequential Recommendation

**arXiv ID:** 2604.04427 | [PDF](https://arxiv.org/pdf/2604.04427v1)

**作者:** Ke Shi `[一作]` (University of Electronic Science and Technology of China), Shuo Shang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 5665 | [OpenAlex ID](https://openalex.org/A5102754146)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于流匹配的单步生成序列推荐框架，利用语义锚定先验和全局平均速度实现从历史用户嵌入直接到下一个物品分布的直接推断。

**💡 创新点**

创新点包括：①使用语义锚定先验将生成起点从随机噪声转为掩码后历史嵌入，显著缩短生成轨迹；②引入全局平均速度与JVP一致性约束，将多步轨迹压缩为单步直线运动，消除线性冗余；③双端语义对齐防止表示崩塌，提升模型稳定性。

**🔧 技术方法**

核心技术为流匹配（Flow Matching）与ODE轨迹建模，结合Transformer结构、时间嵌入、噪声调制、JVP一致性约束和双端语义对齐。

**📊 数据集**

在三大公开基准上进行实验：MovieLens‑100k、Amazon‑Beauty 与 Steam，覆盖稠密到稀疏不同规模的数据集。

**📈 对比分析**

与传统RNN/CNN、Transformer、Diffusion、FMRec等基线对比，本文方法在Hit@20、N@20 等指标均实现显著提升（如 ML‑100k 上 N@20 提升 9.9%），且推断效率提升 20‑30 倍（FLOPs 下降到 0.04–0.07G，推断时间 4–5 ms/样本）。

**⚠️ 局限性**

局限性主要体现在：①对掩码保留率与各项损失权重高度敏感，需精细调参；②在极度稀疏数据集上的优势不如在稠密数据集显著；③虽然保持了多样性，但对极端多模态需求的泛化能力仍待进一步验证。

---

## 604. Subspace Control: Turning Constrained Model Steering into Controllable Spectral Optimization

**arXiv ID:** 2604.04231 | [PDF](https://arxiv.org/pdf/2604.04231v1)

**作者:** Yancheng Huang `[一作]` (Michigan State University), Sijia Liu `[通讯]` (Michigan State University)

**通讯引用:** 6715 | [OpenAlex ID](https://openalex.org/A5100321835)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种子空间控制框架SIFT，用于解决大型预训练模型在满足安全、隐私或任务特定需求时的约束优化问题。

**💡 创新点**

创新点在于将一-shot模型合并与梯度正交化（Muon）联系起来，构造干扰无关的谱子空间，并通过局部化干预实现可控的优化冲突消除。

**🔧 技术方法**

主要技术包括谱子空间构建（SVD）、矩阵符号函数正交化（Muon的msign）、误差阈值驱动的时间-空间局部化控制，以及对比实验中的AdamW、Muon、POME和BLUR等基线。

**📊 数据集**

使用多种基准数据集：WMDP、Wikitext、PKU‑SafeRLHF、Alpaca、RAGTruth、GLM‑4‑Voice、ESNLI、COSE、OpenBookQA 等，覆盖机器去学习、模型安全对齐、文本到语音适配与幻觉抑制四个应用场景。

**📈 对比分析**

与基线对比，SIFT在所有四个任务上均取得显著或稳健的性能提升：在去学习任务中显著降低了ES‑Bio/Cyber误差且保持高效用；在安全对齐中安全度和通用任务分数均超过BLUR和POME；在文本‑语音适配中在所有输入输出设置下优于其他方法；在幻觉抑制中实现最低幻觉率同时保持原有推理性能。

**⚠️ 局限性**

局限性包括：1）需要额外的SVD运算，导致训练时间比标准Muon大约两倍；2）局部化阈值和子空间维度需手动调参，影响通用性；3）目前仅适用于后训练阶段，未探讨预训练中的约束控制。

---

## 605. Automated Expected Cost Analysis for Quantum Programs

**arXiv ID:** 2604.03971 | [PDF](https://arxiv.org/pdf/2604.03971v1)

**作者:** Georg Moser `[一作]` (Universität Innsbruck), Michael Schaper `[通讯]` (Universität Innsbruck)

**通讯引用:** 1736 | [OpenAlex ID](https://openalex.org/A5016872166)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并实现了一套能够自动推导混合经典-量子程序期望成本上界的工具。

**💡 创新点**

创新点在于：①引入基于词项的量子期望变换器表示，①采用符号执行与约束生成的流水线实现全自动化成本分析；②利用 Handelmann 定理生成非负多项式证书，完成循环不变式的自动推导。

**🔧 技术方法**

主要技术包括：量子期望变换器框架、符号执行、约束生成与转化（期望词项→成本约束→多项式约束→证书约束）、SMT 求解器（Z3）以及 Handelmann 定理用于非负多项式合成。

**📊 数据集**

数据集由文献中的典型量子程序（如 RUS、CHAIN、X、Grover 等）以及 IBM Showcase 的 Repeat‑Until‑Success 电路等组成，覆盖了 1~6 个量子比特的混合经典-量子程序。

**📈 对比分析**

评估中将自动推导的上界与已知手工推导或工具给出的结果进行对比，绝大多数程序在秒级完成；对比结果显示自动推导的上界与文献中的最紧上界一致；对无法求解的程序标记为“?”。

**⚠️ 局限性**

局限性包括：①对大规模量子态（>6 个比特）和参数化量子态的支持不足；②在存在高阶量子门或复杂测量分支时约束生成难以收敛，导致求解失败；③目前仅支持固定成本模型（如每个循环计数 1），未覆盖更通用的资源度量。

---

## 606. Hemispherical Concentration for Semi-Unsourced Random Access in Many-Access Regime

**arXiv ID:** 2604.03987 | [PDF](https://arxiv.org/pdf/2604.03987v1)

**作者:** Nazanin Mirhosseini `[一作]` `[通讯]` (Colorado State University), Nazanin Mirhosseini (Colorado State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了一种半无源随机接入模型，其中轻量级协调器为活跃用户分配不同的标识符。每个活跃用户从消息集中随机选择消息，形成ID-消息对，并使用均匀分布在超球面上的球面码本进行编码。

**💡 创新点**

提出了一种半无源随机接入模型，消除了由于相同消息选择而产生的碰撞，放宽了码本大小的要求，并通过轻量级协调器分配独特的ID给活跃用户，从而提高了性能。

**🔧 技术方法**

使用了球面码本和最大似然估计（ML）技术，分析了在多接入信道（MnAC）条件下的解码过程。

**📊 数据集**

使用了从超球面均匀独立抽取的码本，假设活跃用户数量和码本大小都与块长度n线性增长。

**📈 对比分析**

通过与传统的无源随机接入（URA）方法进行比较，证明了在0<β<2的条件下，所提出的模型的每用户错误概率随着n的增加而趋近于0，且最坏情况下的每用户ML错误概率的指数衰减率为P/4。

**⚠️ 局限性**

模型的局限性在于对于β≥2的情况，无法保证K_a(n)-子集的半球性质收敛到1，且在更小的β值下的传输子集特征尚未完全表征。

---

## 607. SBF: An Effective Representation to Augment Skeleton for Video-based Human Action Recognition

**arXiv ID:** 2604.03590 | [PDF](https://arxiv.org/pdf/2604.03590v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 608. A Differentiable Framework for Gradient Enhanced Damage with Physics-Augmented Neural Networks in JAX-FEM

**arXiv ID:** 2604.03411 | [PDF](https://arxiv.org/pdf/2604.03411v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 609. Gradual Cognitive Externalization: A Framework for Understanding How Ambient Intelligence Externalizes Human Cognition

**arXiv ID:** 2604.04387 | [PDF](https://arxiv.org/pdf/2604.04387v1)

**作者:** Zhimin Zhao `[一作]` (Queen's University), Zhimin Zhao `[通讯]` (Queen's University)

**通讯引用:** 912 | [OpenAlex ID](https://openalex.org/A5034367719)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Gradual Cognitive Externalization（GCE）框架，阐释人类认知如何通过环境智能与AI协同逐步迁移至数字子系统，并整理现有AI技能生态中的外部化证据。

**💡 创新点**

创新点在于：①将行为流形假设、扩展心智理论和多尺度能力架构融合成可量化的三大认知集成准则（双向适配、功能等价、因果耦合）；②提供可测量的外部化进度指标（Φ(t)）和一套可实施的实验方案；③将“认知迁移”视为渐进过程而非瞬时上传，重新定义“是否上传意识”的问题。

**🔧 技术方法**

主要技术手段包括：持续行为观察与机器学习（强化学习/监督学习）、可插拔AI技能标准（Skill文件）、跨领域知识记忆技术（Notion AI、Mem、LangChain）、多模态嵌入与错误最小化导航模型（transformer、扩散模型），以及信息理论指标（互信息、距离度量）来评估双向耦合与功能等价。

**📊 数据集**

使用的数据来源多样，主要基于已部署系统的实时交互记录：日程调度（Google Calendar、Reclaim.ai、Clockwise）、邮件与写作（Gmail Smart Compose、Grammarly、GPT助手）、推荐系统（Netflix、Spotify）、知识管理（Notion AI、Mem、LangChain）、以及公开的SkillsBench任务数据集。

**📈 对比分析**

对比方法主要是：1) 在SkillsBench上测量任务完成率，显示引入外部化技能后从22%提升至45%（约+23%）；2) 通过对比人类自我预测与AI预测准确率，验证功能等价阈值；3) 采用双向适配指标（互信息）与因果耦合实验（随机干预）评估模型与用户行为的共进化。整体表现显示，外部化技能在专业任务上显著提升性能，且部分自然语言生成任务已达到人类间差异的基线。

**⚠️ 局限性**

局限性包括：①缺乏长期纵向实验验证双向适配与因果耦合的阈值，现有证据多为横断面或小样本；②对高阶认知（元认知、情感、意识）外部化的可测量性不足；③外部化进度指标Φ(t)在多领域与个体间的权重设定尚不确定；④隐私与数据安全挑战在大规模行为数据收集中未得到完整解决。

---

## 610. Next-Scale Autoregressive Models for Text-to-Motion Generation

**arXiv ID:** 2604.03799 | [PDF](https://arxiv.org/pdf/2604.03799v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 611. E-VLA: Event-Augmented Vision-Language-Action Model for Dark and Blurred Scenes

**arXiv ID:** 2604.04834 | [PDF](https://arxiv.org/pdf/2604.04834v1)

**作者:** Jiajun Zhai `[一作]` (Zhejiang University), Kaiwei Wang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了E-VLA框架，将事件相机信号直接融合到视觉-语言-动作模型中，以提升在低照度和运动模糊场景下的机器人操作鲁棒性。

**💡 创新点**

创新点在于不通过事件图像重建，而是采用轻量级、预训练兼容的事件融合策略（叠加融合和层次事件适配器），以及事件窗口化与累积方法。

**🔧 技术方法**

使用的技术包括事件相机（DAVIS346）、事件窗口化（固定计数）、累积事件图、预训练的SigLIP视觉编码器、SmolVLA基础模型、轻量化事件适配器以及同步/异步推理。

**📊 数据集**

使用的数据库是作者自行构建的同步RGB-事件-动作数据集，涵盖Pick-Place、Sorting、Stacking三种任务，在200、100、75、40 lux等不同照度条件下收集共724个episode。

**📈 对比分析**

与传统低照度图像增强和事件重建方法相比，E-VLA在20 lux下Pick-Place成功率从0%提升到90%，在高曝光时间下运动模糊时亦显著提升；总体保持在高光照条件下性能不受影响。

**⚠️ 局限性**

局限性包括事件对颜色判别支持不足，导致Sorting任务中的颜色识别效果差；以及在堆叠任务中相机视角被抓取物遮挡时仍难以克服视觉瓶颈。

---

## 612. The Topology of Multimodal Fusion: Why Current Architectures Fail at Creative Cognition

**arXiv ID:** 2604.04465 | [PDF](https://arxiv.org/pdf/2604.04465v1)

**作者:** Xiujiang Tan `[一作]` (Guangzhou Academy of Fine Arts), Xiujiang Tan `[通讯]` (Guangzhou Academy of Fine Arts)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5114130298)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文通过理论分析与实验验证，揭示当前多模态 AI 在跨模态融合上存在的拓扑局限，并提出“重叠拓扑”概念及“通用重叠区算子”（UOO），利用神经 ODE 与拓扑正则化来训练能保持非可分表示的模型，并设计跨模态类比基准与神经影像实验方案。

**💡 创新点**

创新点在于：①将 Wittgenstein 说话/展示区分与中国工艺 xiàng 结合，提出跨模态融合的拓扑诊断框架；②引入纤维束与连通理论以及杨-米尔斯作用函数来刻画跨模态结构一致性；③用可微分持久同调作为训练正则化，直接塑造重叠拓扑；④构造 ANALOGY‑MM 评价指标 ETR 以量化当前架构对跨模态结构映射的失效。

**🔧 技术方法**

采用的技术包括：神经 ODE（参数化的连接形式）、可微分持久同调层（Topological Loss）、纤维束连通和曲率算子、跨模态注意力与对比学习框架、扩散生成模型、动态因果建模（DCM）与 Granger 因果分析，以及 Witness Complex 与 Distance‑to‑Measure 滤波加速持久同调计算。

**📊 数据集**

使用的数据集包括：自定义合成两模态数据集（图形/结构、音频/节奏等）用于验证 UOO 的零样本迁移；公开 fMRI/MEG 创造力实验数据（AUT、DAT、CAQ 等）；跨模态类比基准样本；以及常见视觉‑语言对齐数据（ImageNet‑Captions、CLIP 语料）作基准对照。

**📈 对比分析**

对比方法：在合成任务中，将 UOO 与 CLIP、GPT‑4V、扩散基准模型按任务准确率、零样本迁移性能与持久同调指标进行评估；在 ANALOGY‑MM 上，计算 ETR 以评估跨模态结构错误率。初步结果显示，UOO 在零样本迁移上显著优于传统架构，并在持久同调指标上保留了 β1 循环，提示重叠拓扑的有效性；但在实际视觉‑语言基准上的性能差距仍需进一步优化。

**⚠️ 局限性**

局限性包括：纤维束、基组 G、基底空间 B 的形式化定义尚不完整；持久同调正则化的 O(n³) 计算成本在大规模模型中仍有瓶颈；ANALOGY‑MM 的心理测量属性（难度、区分度）未完成验证；实验设计中对高维拓扑特征的可靠估计与对实际创造力数据的可推广性仍需深入研究。

---

## 613. PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis

**arXiv ID:** 2604.04576 | [PDF](https://arxiv.org/pdf/2604.04576v1)

**作者:** Inseong Choi `[一作]` (Dongguk University), Soohwan Song `[通讯]` (Dongguk University)

**通讯引用:** 700 | [OpenAlex ID](https://openalex.org/A5013538220)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对稀疏视角下扩散模型生成的合成视角存在光度与几何不一致的问题，本文提出 PR-IQA 方案，用部分重叠区域的几何一致性生成部分质量图，并通过跨视角注意力完成全图质量评估；随后将该质量图用于 3D Gaussian Splatting（3DGS）训练中的图像筛选和像素级遮罩，提升稀疏视角 3D 重建与新视角合成质量。

**💡 创新点**

创新点在于：①将质量评估视为“质量补全”任务；②设计了三流编码-解码网络，利用参考图像的跨视角注意力将部分质量信号扩散至未覆盖区域；③构造了自监督目标（DINOv2 及 SSIM 参考），在无 GT 的条件下实现 FR‑级质量预测；④将 PR‑IQA 与 3DGS 训练深度结合，形成双层过滤（图像级与像素级）策略。

**🔧 技术方法**

核心技术包括：扩散模型生成伪真值视角、视觉几何变换（VGGT）实现像素级几何对应、DINOv2 特征相似度计算、三流（查询图像、参考图像、部分质量图）交叉注意力网络、双门注意力块（Channel + Spatial）以及自监督损失（L1、JSD、PLCC）。

**📊 数据集**

训练使用 MFR 数据集（120k 训练对）；评估采用 Tanks & Temples、Mip‑NeRF 360、RealEstate10K 三个公开基准；生成的伪真值视角来源于同一数据集的扩散模型采样。

**📈 对比分析**

在 IQA 任务中，PR‑IQA 在 DINOv2‑SIM 与 SSIM 目标上分别以 PLCC/SRCC 方式对比 FR‑IQA、NR‑IQA 与 CR‑IQA 基线，均实现最高或第二高的相关性，逼近 FR‑IQA 的精度；在 3DGS 训练中，使用 PR‑IQA 生成的质量掩码的模型在 PSNR、SSIM 上显著优于 Vanilla 3DGS、ViewCrafter 以及其他 NR/CR-IQA 引导的 3DGS，LPIPS 指标也最低，表明重建质量提升明显。

**⚠️ 局限性**

主要局限包括：①依赖精确的几何对应，若相机姿态估计误差大，部分质量图生成会受影响；②跨视角注意力网络计算量较大，推理时延高；③在极少重叠或完全无重叠的视角间，质量补全效果有限；④目前仅在室内/室外静态场景验证，缺乏对动态或遮挡严重场景的评估。

---

## 614. OP-GRPO: Efficient Off-Policy GRPO for Flow-Matching Models

**arXiv ID:** 2604.04142 | [PDF](https://arxiv.org/pdf/2604.04142v1)

**作者:** Liyu Zhang `[一作]` (Zhejiang University), Chao Li `[通讯]` (Zhejiang University)

**通讯引用:** 38821 | [OpenAlex ID](https://openalex.org/A5100323172)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 OP‑GRPO，一种针对流匹配模型的离线‑在线（off‑policy）GRPO 框架，能够高效重用历史轨迹并加速对齐训练。

**💡 创新点**

创新点在于：① 设计高质量重放缓冲区主动采样并替换；② 引入序列级重要性采样校正，保持 GRPO 的裁剪机制；③ 针对低噪声阶段的数值不稳定性，采用轨迹截断策略。

**🔧 技术方法**

使用的技术包括：重放缓冲区（Replay Buffer）、序列级重要性采样、轨迹截断、流匹配模型（Flow‑Matching）与 GRPO（Group Relative Policy Optimization）结合的强化学习方法。

**📊 数据集**

使用的数据集：Stable‑Diffusion‑3.5‑Medium（SD3.5‑M）图像数据集和 Wan2.1‑1.4B 视频生成数据集，任务涵盖组合图像生成、可视化文本渲染和人类偏好对齐。

**📈 对比分析**

与传统 Flow‑GRPO 对比，OP‑GRPO 在相同任务下仅需约 34.2% 的训练步数即可达到甚至超越 Flow‑GRPO 的最终性能，显著提升训练效率。

**⚠️ 局限性**

限制：目前仅在图像与视频生成任务上验证，未对音频或 3D 生成等其他生成领域扩展；离线采样比例过高可能导致分布偏移导致训练不稳定。

---

## 615. Primitive-based Truncated Diffusion for Efficient Trajectory Generation of Differential Drive Mobile Manipulators

**arXiv ID:** 2604.04166 | [PDF](https://arxiv.org/pdf/2604.04166v1)

**作者:** Long Xu `[一作]`, Fei Gao `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种学习增强的运动规划框架，专为差分驱动移动机械手（DDMoMa）设计，包含关键点序列提取（KSE）任务编码器、原语截断扩散模型（PTDM）以及后处理的轨迹优化；

**💡 创新点**

创新点在于：①利用KSE将边界状态映射到三维点云空间，并通过注意力机制实现环境点云与边界状态的高效融合；②设计PTDM，在偏置的原语分布上进行扩散学习，显著提升路径多样性并减少模式崩溃，同时通过DDIM加速大幅降低采样步数；

**🔧 技术方法**

使用的技术包括可微前向运动学、点云与关键点的多层感知机/点变换器编码、交叉注意力融合、原语截断扩散模型、DDIM加速、以及基于多项式/弧长-偏航参数化的轨迹优化；

**📊 数据集**

实验数据集由自建的三类随机障碍场景组成：Cuboids、Mixed 和 Replica，并使用 TopAY/OMPL 生成的百万级训练样本；未使用公开公共数据集；

**📈 对比分析**

与经典采样规划器 TopAY、vanilla DDPM 以及 anchor‑based diffusion 进行对比，结果显示 PTDM 在路径多样性、规划时长和成功率上均优于对手，且在轨迹时间（T.D.）和总体规划时间（T.P.）上与 TopAY 相当或更好，整体性能显著提升；

**⚠️ 局限性**

局限性：仅在静态仿真环境验证，未考虑感知噪声与控制不确定性；扩散过程仍需离线点序列生成并配合轨迹优化，后处理开销较大；在真实机器人及不同机械手配置上的适用性尚待验证。

---

## 616. Bridging Restoration and Diagnosis: A Comprehensive Benchmark for Retinal Fundus Enhancement

**arXiv ID:** 2604.03806 | [PDF](https://arxiv.org/pdf/2604.03806v1)

**作者:** Xuanzhao Dong `[一作]` (Arizona State University), Yalin Wang `[通讯]` (Arizona State University)

**通讯引用:** 11643 | [OpenAlex ID](https://openalex.org/A5100740828)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EyeBench‑v2 基准，系统评估视网膜图像增强模型的多维质量。

**💡 创新点**

创新点在于：①结合全参考与无参考评估，②引入专家指导的标注与下游临床任务（血管/病变分割、糖尿病视网膜病变分级），③构建分布对齐的数据集，④提供可操作的性能洞见。

**🔧 技术方法**

采用生成模型（GAN、扩散/ SDE、OT、Transformer 等），配合 U‑Net 分割网络、MobileNet 分类网络、Ret‑Found/Ret‑Clip 表征模型，及专家评估指标。

**📊 数据集**

使用 EyeQ、IDRID、DRIVE、EyeQ‑Full‑Reference（16,817 张）和 EyeQ‑No‑Reference（6,434 名受试者）等公开眼底图像数据集。

**📈 对比分析**

与 SCR‑Net、Cofe‑Net、PCE‑Net、GFE‑Net、RFormer、CycleGAN、WGAN、OTTGAN、OTEGAN、Context‑aware OT、TPOT、CUNSB‑RFIE 等基线比较。结果显示：在合成噪声下 TPOT 在全参考下性能最佳；在无参考下 OTEGAN 与 CycleGAN 在 DR 分级和特征对齐方面最优；整体指标与专家手工评估高度相关。

**⚠️ 局限性**

局限性：GAN/OT 方法仍易出现结构失真；SDE 方法在语义一致性上表现欠佳；基准主要基于公开数据，缺少大规模真实临床分布；未能完全平衡去噪与细结构保留的权衡。

---

## 617. k-Maximum Inner Product Attention for Graph Transformers and the Expressive Power of GraphGPS The Expressive Power of GraphGPS

**arXiv ID:** 2604.03815 | [PDF](https://arxiv.org/pdf/2604.03815v1)

**作者:** Jonas De Schouwer `[一作]` (Stanford University), Xiaowen Dong `[通讯]` (University of Oxford)

**通讯引用:** 2764 | [OpenAlex ID](https://openalex.org/A5101579932)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了k-MIP注意力机制，改进图Transformer在大规模图上的可扩展性；

**💡 创新点**

创新点在于通过top‑k内积选择最相关的键，使用符号矩阵实现线性内存，且证明其可逼近任意全注意力Transformer；

**🔧 技术方法**

主要技术包括k-MIP自注意力、符号矩阵实现、GraphGPS框架集成、以及对S‑SEG‑WL测试的表达力分析；

**📊 数据集**

使用了Long Range Graph Benchmark、City‑Networks、ShapeNet‑Part和S3DIS等大规模图数据集；

**📈 对比分析**

与全注意力、BigBird、Performer、Exphormer等可扩展注意力机制对比，k-MIP在推理速度提升约10倍、可处理50万+节点，并在各基准上与最优方法竞争；

**⚠️ 局限性**

局限性包括仍保持O(N²)计算复杂度、top‑k可能忽略重要键导致训练时监督不足、以及对GPU低层优化（如FlashAttention）依赖较高。

---

## 618. SuperLocalMemory V3.3: The Living Brain -- Biologically-Inspired Forgetting, Cognitive Quantization, and Multi-Channel Retrieval for Zero-LLM Agent Memory Systems

**arXiv ID:** 2604.04514 | [PDF](https://arxiv.org/pdf/2604.04514v1)

**作者:** Varun Pratap Bhardwaj `[一作]` `[通讯]` (Independent Researcher), Varun Pratap Bhardwaj (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SuperLocalMemory V3.3，一个本地优先、全认知层级的 AI 编码代理记忆系统，集成忘记、量化、软提示参数化以及零摩擦自动生命周期管理。

**💡 创新点**

创新点包括：1) FRQAD 量化感知距离实现对高低精度嵌入的完美区分；2) 基于艾宾浩斯的自适应忘记与生命周期量化，将记忆压缩与衰减统一；3) 7 通道认知检索与交叉编码重排序，覆盖语义、关键词、实体图、时间、扩散激活、凝聚与 Hopfield 关联；4) 记忆参数化将显式记忆转化为自然语言软提示，实现无 LLM 训练的隐式记忆；5) 零摩擦 auto‑cognitive pipeline 通过单一安装实现完整生命周期自动化；6) 代码知识图与 Daemon 服务提升开发体验与性能。

**🔧 技术方法**

技术手段包括：信息几何（Fisher‑Rao 距离）、TurboQuant 量化、SQLite+sqlite‑vec 存储、ONNX cross‑encoder 重排序、Hopfield 关联网络、Fokker‑Planck 生命周期模型、贝叶斯信任评估、AST 解析构建代码知识图、Daemon 服务实现 32× 冷启动加速、软提示生成器。

**📊 数据集**

使用的数据集与基准：LoCoMo (304 QA 对 / 1,585 事实)，FRQAD 18,840 查询-事实对，EAP 20 个查询与 929 事实的混合量化检索测试，170 事实的 30 天遗忘模拟，10 条事实的会话连续性测试，LoCoMo 2 场景的 304 QA 对，SQLite 存储的 5,000+ 每月下载量。

**📈 对比分析**

评估方法：在 LoCoMo Mode A（零 LLM）下，V3.3 达到 70.4%（214/304）整体准确率，单跳 60.5% 低于 V3.2 的 65.1%，但在多跳 +23.8pp、时序 +15.3pp、对抗 +12.7pp 上显著提升；FRQAD 对比余弦和标准 Fisher‑Rao，完全匹配高精度嵌入；混合量化检索 @10 维持 68% 召回；遗忘模型在 30 天后热记忆 R≈0.35，冷记忆 R≈0，差异 6.7×。相比同类系统，V3.3 在零云环境下排名第二，且提供忘记、量化、参数化等 V3.2 无法实现的功能。

**⚠️ 局限性**

局限性：单跳检索受 7 通道融合噪声影响；2‑bit 极端压缩导致嵌入质量下降；软提示不如 LoRA 细粒度控制；冷启动阶段需 200+ 反馈信号才能完全自适应；零摩擦管线专为 Claude Code 设计，跨平台集成仍需手动 Hook；未来工作包括动态通道权重、超越 2‑bit 的压缩、LoRA 参数化、联邦记忆与差分隐私、查询依赖通道路由。

---

## 619. Beyond Generation: An Empirical Study on Redefining the Act of Drawing Through an 85% Time Reduction in Picture-Book Production

**arXiv ID:** 2604.03549 | [PDF](https://arxiv.org/pdf/2604.03549v1)

**作者:** Cosei Kawa `[一作]` `[通讯]` (Shizuoka University of Art and Culture), Cosei Kawa (Shizuoka University of Art and Culture)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究通过对一部15幅插画专业绘本的完整制作，评估并实现了人工智能协同工作流程，替代传统手绘流程，大幅降低整体制作时间。

**💡 创新点**

创新点在于将生成式AI用作快速草稿生成工具，同时通过“Prompt Recipe”将创作者的隐式风格知识外显化，并在后期投入大量人类高阶审美与手工完善（Completion），重新定义创作者角色与工作分配。

**🔧 技术方法**

使用的技术包括 Midjourney v6 图像生成、Gemini LLM 进行标签提取与词典归一化、Python+Streamlit 开发的 Prompt Recipe UI、异步 Prompt 队列和 Chrome DevTools 协调的后端批处理，以及 ExifTool 进行元数据写入。

**📊 数据集**

数据集为作者约200幅先前创作的作品，用作风格参考与Prompt 参数化，未使用公开基准数据集。

**📈 对比分析**

比较方法是将同一作者在传统手绘流程（Carpe Diem）与AI协同流程（Golden Drops Opening the Sky）下的各阶段工时进行对照；结果显示整体工时从2162.8小时降至320.4小时，节省率85.2%，完成阶段工时为235.3小时，明显高于传统流程的细节处理时间。

**⚠️ 局限性**

局限性包括仅为单一创作者单一作品的案例研究，缺乏跨创作者和更大规模作品的验证；Completion工时依赖创作者的隐式风格知识，易受个人习惯影响；未来工作需在多作者、多项目中验证系统组件的可迁移性。

---

## 620. Light-Bound Transformers: Hardware-Anchored Robustness for Silicon-Photonic Computer Vision Systems

**arXiv ID:** 2604.04330 | [PDF](https://arxiv.org/pdf/2604.04330v1)

**作者:** Xuming Chen `[一作]` (Case Western Reserve University), Gourav Datta `[通讯]` (Case Western Reserve University)

**通讯引用:** 452 | [OpenAlex ID](https://openalex.org/A5017435097)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种完整的“measure → model → train → run”框架，将Vision Transformer (ViT) 部署到硅光子微环谐振器 (MR) 加速器上，并通过硬件噪声建模、Chance‑Constrained Training (CCT) 与噪声感知 LayerNorm 使模型在真实噪声下保持高精度。

**💡 创新点**

创新点：①将测得的工艺、热漂移和激光幅度噪声转换为可微分的方差代理；②在注意力子层中引入 CCT，直接约束注意力排序不被噪声翻转；③设计噪声感知 LayerNorm，去除噪声对统计量的影响；④实现硬件能量感知的完整推理流水线，并在不需要现场学习或额外光学 MAC 的情况下恢复几乎“干净”精度。

**🔧 技术方法**

使用技术包括：硅光子微环谐振器光学矩阵乘法（WDM、波导分配、VCSEL、BPD）；工艺/热/激光噪声模型和闭式方差代理；Chance‑Constrained Loss 与噪声感知 LayerNorm；硬件-软件协同优化、能量感知推理流程；实验使用硬件‑仿真混合（hardware‑in‑the‑loop）和真实 MR 银行。

**📊 数据集**

使用的数据集：ImageNet（ViT‑Base）、TinyImageNet、CIFAR‑10（ViT‑Tiny/S）用于分类；COCO（目标检测和实例分割）用于密集预测。

**📈 对比分析**

对比方法：清洁模型、仅噪声推理、标准微调、CCT 微调、CCT+NALN 微调。CCT+NALN 在 σ_fab 0.2–0.7 时恢复 89–97% 级别精度。硬件实验显示光子加速器在 KFPS/W 上达到 100.4，远超 Xilinx VCK190 FPGA（1.42）和 NVIDIA A100 GPU（0.86），速度提升 70–116 倍；能量消耗则低 2–3 个数量级。

**⚠️ 局限性**

局限性：①需要先测量并建模硬件噪声，噪声分布需保持稳定；②现有设计对 ADC/DAC 能耗占比较高，进一步减低转换开销是挑战；③对极端噪声仍有残余误差；④技术规模受微环波长、芯片面积及光子耦合效率限制；⑤长时漂移与温度变化需额外实时校正；⑥目前主要验证在中小规模 ViT，深层模型与大规模部署尚未充分评估。

---

## 621. Vanast: Virtual Try-On with Human Image Animation via Synthetic Triplet Supervision

**arXiv ID:** 2604.04934 | [PDF](https://arxiv.org/pdf/2604.04934v1)

**作者:** Hyunsoo Cha `[一作]` (Seoul National University), Hanbyul Joo `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建单阶段统一框架，利用单张人像、服装图像与姿态视频直接生成高保真、身份一致的动画视频。

**💡 创新点**

提出可扩展的三元组监督、双模块 Video Diffusion Transformer 结构以及零射击服装插值，显著提升姿态遵循、服装准确性与身份保持。

**🔧 技术方法**

基于 DiT 骨干，使用 Human Animation Module 与 Garment Transfer Module 双模块设计，并结合 VAE、FLUX 生成与 inpainting、LLM 文本提示等技术。

**📊 数据集**

使用从线上购物网站抓取的 9,135 条视频、自己采集的上下衣三元组数据以及野外视频构建三元组训练集。

**📈 对比分析**

与 16 种两阶段基线（虚拟试衣+动画）和 VACE 单阶段模型对比，在 L1、PSNR、SSIM、LPIPS、FID、VFID 等指标上均取得最优或相近性能，显著优于传统组合。

**⚠️ 局限性**

仍面临服装种类多样性不足、极端姿态下动态一致性欠佳以及在真实光照和复杂背景中的泛化能力等局限。

---

## 622. Subset Balancing and Generalized Subset Sum via Lattices

**arXiv ID:** 2604.04656 | [PDF](https://arxiv.org/pdf/2604.04656v1)

**作者:** Yiming Gao `[一作]` (University of Science and Technology of China), Yanbin Pan `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 420 | [OpenAlex ID](https://openalex.org/A5056715913)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了将 Subset Balancing 与 Generalized Subset Sum 等问题归约为 ℓ∞‑SVP / ℓ∞‑CVP 的方法，利用最短/最近点问题的单指数算法实现了新的确定性与随机化求解器。

**💡 创新点**

核心创新在于：①消除了 log d 因子，算法时间仅依赖于维度 n；②提供了单指数的确定性算法，并在 d≥15 时就优于经典 Meet‑in‑the‑Middle；③对大 d 甚至能得到多项式时间解法；④将盒形约束推广到任意中心对称凸体。

**🔧 技术方法**

主要技术包括：lattice embedding 与完整秩化；利用 Dadush‑Peikert‑Vempala 的 ℓ∞‑SVP 算法、Mukhopadhyay 的随机化 SVP；Ellipsoid‑Enum枚举技术；LLL 算法在大 d 情况下的快速近似；对平均情况使用随机 x、τ 并证明满足 bounded‑distance 条件。

**📊 数据集**

该工作属于理论计算，不使用实验数据集；所有结果均为算法复杂度分析与概率上界证明。

**📈 对比分析**

与传统的 Meet‑in‑the‑Middle 算法（O((2d+1)^{n/2})）比较，随机化版本在 d≥15 时达到 Õ(2^{2.443n})，比基准更快；确定性版本得到 Õ(2^{4.632n})，虽然常数较大，但在大 d 或需要确定性保证的场景下更具优势；对凸体一般化，算法时间为 Õ(2^{c_K n})，其中 c_K 只取决于形状而非大小。

**⚠️ 局限性**

局限性包括：①对小 d 时，指数常数仍较高，未能突破现有 representation‑based 方法；②对 Generalized Subset Sum 仍需要平均情况假设和随机 x；③当前仍缺乏单指数时间的 ℓ∞‑CVP 算法，限制了最优解的实现；④大 d 的多项式解法需要 d 超大，实际可行性受限。

---

## 623. Can Natural Image Autoencoders Compactly Tokenize fMRI Volumes for Long-Range Dynamics Modeling?

**arXiv ID:** 2604.03619 | [PDF](https://arxiv.org/pdf/2604.03619v1)

**作者:** Peter Yongho Kim `[一作]` (Seoul National University), Taesup Moon `[通讯]` (Seoul National University)

**通讯引用:** 2678 | [OpenAlex ID](https://openalex.org/A5080346989)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

开发了一种利用预训练的二维自然图像自编码器对fMRI体积进行分词，随后使用Transformer对长时序进行建模，并通过自监督掩码分词预训练提升下游任务性能。

**💡 创新点**

① 用跨域二维自编码器无须再训练即对4D fMRI进行高效分词；② 通过极低的分词维度实现长时序Transformer训练；③ 提出了掩码分词的自监督预训练。

**🔧 技术方法**

2D DCAE（Deep Compression Autoencoder）分词、Transformer编码器（含分组查询注意力、旋转位置编码）、掩码分词预训练、集成梯度可解释性。

**📊 数据集**

UK Biobank（中老年人）、Human Connectome Project（健康青年）、ADHD‑200（儿童青少年），并在HBN‑Movie等任务上做验证。

**📈 对比分析**

与ROI基线（XGBoost、BrainNetCNN等）及最优voxel基线（TFF、SwiFT）在年龄、性别、智商、ADHD诊断四项任务上对比，TABLeT在长时序输入下取得更优或相近的准确/ AUC/ MAE，并在相同显存下显著降低内存和训练时间。

**⚠️ 局限性**

① 逐帧独立分词忽略细微时间依赖；② Transformer未显式建模空间/时间结构；③ 长时序收益因任务不同而异，缺乏系统评估。

---

## 624. Efficient Solving for Dynamic Data Structure Constraint Satisfaction Problem

**arXiv ID:** 2604.03624 | [PDF](https://arxiv.org/pdf/2604.03624v1)

**作者:** Nanbing Li `[一作]` (Peking University), Yun Liang `[通讯]` (Peking University)

**通讯引用:** 9986 | [OpenAlex ID](https://openalex.org/A5100604860)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对动态数据结构约束满足问题（D^2SCSP）的高效求解框架。

**💡 创新点**

创新点在于基于依赖关系的分区以及增量编码/激活机制，实现跨求解重用状态与约束。

**🔧 技术方法**

采用依赖引导的分区算法、增量SAT/SMT编码、假设管理以及增量求解技术。

**📊 数据集**

使用真实工业约束随机验证基准，涉及多维数组、嵌套结构等动态数据结构场景。

**📈 对比分析**

与原始VeriSim和Synopsys VCS比较，平均在VeriSim上加速24.8×，在VCS上加速1.72×，在工业基准上显著提升性能。

**⚠️ 局限性**

局限在对非SMT友好约束的支持不足以及在极深动态结构导致的求解失败情况。

---

## 625. GENFIG1: Visual Summaries of Scholarly Work as a Challenge for Vision-Language Models

**arXiv ID:** 2604.04172 | [PDF](https://arxiv.org/pdf/2604.04172v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 626. Contextual Control without Memory Growth in a Context-Switching Task

**arXiv ID:** 2604.03479 | [PDF](https://arxiv.org/pdf/2604.03479v1)

**作者:** Song-Ju Kim `[一作]` `[通讯]` (SOBIN Institute), Song-Ju Kim (SOBIN Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了一种在共享循环状态上通过干预实现上下文依赖的递归架构，并在基于9×9网格迷宫的上下文切换任务中进行评估。

**💡 创新点**

创新点在于引入了基于上下文索引的线性干预算子，使得在不扩大循环维度、也不显式输入上下文的情况下实现上下文控制。

**🔧 技术方法**

采用了LSTM循环核心，添加可学习的线性干预算子并通过强化学习进行训练，同时使用信息理论指标 I(C;O|S) 对模型的上下文依赖性进行量化。

**📊 数据集**

使用了9×9网格迷宫的上下文切换任务（AB25、BA30）作为实验数据集，代理在局部3×3观测下做决策。

**📈 对比分析**

将干预模型与直接标签输入基线 L 以及扩展记忆基线 M（不同内存尺寸）进行对比；干预模型在不增加循环维度的前提下，Phase‑1 成功率与最佳记忆基线 M16 相当，整体表现接近 L。

**⚠️ 局限性**

局限性包括仅在小规模网格迷宫上验证，信息量估计依赖特定的结果定义，干预强度 α 未进行系统调优，且未评估更复杂或随机切换时间的适应性。

---

## 627. Edge-Oriented Orchestration of Energy Services Using Graph-Driven Swarm Intelligence

**arXiv ID:** 2604.04645 | [PDF](https://arxiv.org/pdf/2604.04645v1)

**作者:** Liana Toderean `[一作]` (Technical University of Cluj-Napoca), Tudor Cioara `[通讯]` (Technical University of Cluj-Napoca)

**通讯引用:** 2817 | [OpenAlex ID](https://openalex.org/A5065947536)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个面向能源服务的基于图驱动的边缘-雾-云编排框架，能够实现任务的高效下发与迁移。

**💡 创新点**

创新点在于统一的图模型与基于群体智能（Ant Colony Optimization）的任务调度算法，并通过区块链实现任务迁移的可追溯与支付保障。

**🔧 技术方法**

使用了Neo4j图数据库进行拓扑与任务建模、KubeEdge容器编排平台、Eclipse Dataspace Connector实现数据空间互操作、Ant Colony Optimization算法进行调度，以及区块链智能合约与多重代币记录任务状态。

**📊 数据集**

使用了真实的KubeEdge实验环境（包含云节点和多台边缘节点）、合成的网络拓扑数据以及机器学习能源预测模型（如负荷预测）作为测试数据集。

**📈 对比分析**

通过与传统规则或遗传算法的调度方案比较，实验表明在动态负载下实现零停机迁移，CPU、带宽和延迟开销分别显著低于基线，整体性能提升在10-30%之间。

**⚠️ 局限性**

局限在于大规模拓扑下的可扩展性尚未充分验证，极端波动负载的适应性有限，以及多行政域间自动迁移策略和安全合规性仍需进一步完善。

---

## 628. Bridging the Language Gap in Scholarly Data I: Enhancing Author Disambiguation Algorithms for Chinese Names

**arXiv ID:** 2604.03776 | [PDF](https://arxiv.org/pdf/2604.03776v1)

**作者:** Mingrong She `[一作]` (Maastricht University), Lisette Espín-Noboa `[通讯]` (Graz University of Technology)

**通讯引用:** 217 | [OpenAlex ID](https://openalex.org/A5014548243)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于规则的中文作者名消歧方法，融合共著网络、机构归属、引用关系和文本语义相似度，对中国国家知识基础设施（CNKI）中物理学论文的作者进行识别和归一。

**💡 创新点**

创新点在于：①将内容相似度作为第四条判定条件，弥补仅凭社交网络的不足；②在同一数据集的中文字符和拼音两种写法上实现脚本无关的消歧；③公开完整代码、数据与评测流程，促进复现与进一步研究。

**🔧 技术方法**

技术包括：①规则判定框架（共著、机构相似度、引用关系、Word2Vec余弦相似度阈值）；②Levenshtein相似度和字符串包含检测处理机构；③针对中英文两语料库分别设定内容相似度阈值并通过网格搜索优化；④基于人工标注的20对评测集进行准确性验证。

**📊 数据集**

数据集为CNKI物理学期刊（1953–2024年）20本期刊的论文，包含中文作者名、机构、共著、引用及摘要；另构建对应的拼音与英文翻译版本；人工标注样本为80个作者名对（中文与拼音各40对）。

**📈 对比分析**

与Sinatra et al.和Waqas & Qadir两种基线方法对比，本文方法在中文字符集F1≈0.89、拼音版本F1≈0.88，显著提升召回率且保持精准率（≈1.0），优于基线的0.80–0.82精度和0.66–0.72召回率。

**⚠️ 局限性**

局限性包括：①仅评估物理学领域，跨学科适用性未知；②人工标注样本规模有限，未能覆盖全部姓名变异情况；③仍存在碎片化（召回≈0.87）和同名冲突导致的误合并（拼音版精度≈0.95）问题；④未整合ORCID等持久标识，需进一步提升消歧可靠性。

---

## 629. SAGE-GAN: Towards Realistic and Robust Segmentation of Spatially Ordered Nanoparticles via Attention-Guided GANs

**arXiv ID:** 2604.03637 | [PDF](https://arxiv.org/pdf/2604.03637v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 630. Safety and Liveness of Cross-Domain State Preservation under Byzantine Faults: A Mechanized Proof in Isabelle/HOL

**arXiv ID:** 2604.03844 | [PDF](https://arxiv.org/pdf/2604.03844v1)

**作者:** Jinwook Kim `[一作]` `[通讯]` (Oraclizer Labs), Jinwook Kim (Oraclizer Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文使用 Isabelle/HOL 形式化证明了跨域监管状态同步系统的安全（状态保持）和活性（拜占庭容错下的确定性、无死锁、无饥饿）属性，并将二者结合实现无条件安全保证。

**💡 创新点**

创新点包括：①构建七个通用 locale 体系，支持任意域的可重用验证；②将 Merkle Functor 方案扩展到双向回程、多域一致性与资产隔离；③首次在 Isabelle/HOL 单独完成 BFT 监管共识的活性证明，验证了优先级确定、死锁与饥饿自由；④通过假设释放实现安全-活性组合。

**🔧 技术方法**

采用 Isabelle/HOL 进行机理化证明，利用 locale 继承与解释机制实现模块化；使用优先级总序、超时锁定和公平领导者三类通用 locale，构成完整的安全-活性验证框架。

**📊 数据集**

论文未使用外部真实数据集，而是基于抽象的监管状态机（5 个状态、7 个动作）和同步协议模型，所有证明均基于理论构造。

**📈 对比分析**

相较于以往只验证单域或安全性、或在 Coq/TLA+ 证明活性，本文在 Isabelle/HOL 上完成了双向、多域安全与拜占庭活性并行验证；证明规模 2,348 行，未出现 sorry/oops，证明完全可重复；实验性比较未涉及性能指标，但模型满足所需的时间与空间复杂度。

**⚠️ 局限性**

局限包括：网络模型假设可靠通道且为静态有限域；不考虑动态域加入/移除、动态请求到达；未完成对实现代码（Go/Solidity）的形式化细化；在部分同步网络、消息重传等实际环境中的扩展仍为未来工作。

---

## 631. Binary Caps and LCD Codes with Large Dimensions

**arXiv ID:** 2604.03734 | [PDF](https://arxiv.org/pdf/2604.03734v1)

**作者:** Keita Ishizuka `[一作]` (Mitsubishi Electric Corporation), Yuhi Kamio `[通讯]` (University of Tokyo)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5093013861)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文通过将二进制线性互补双子码（LCD码）与射影空间中的cap（不含三点共线的点集）联系起来，利用Gram矩阵判定LCD性质，并借助Bruen‑Wehlau关于大型cap的结构理论，给出关于最小距离≥4的LCD码的非存在定理，进而完全确定了码维度差为6、7、8时的最大最小距离；同时提供了无计算机搜索的证明；

**💡 创新点**

创新点在于：①构造了从cap到LCD码的几何-代数桥梁——即U_S=∑_{s∈S}ss^T非奇异即为LCD；②将大型cap的几何结构（位于某个超平面补集）与LCD码的存在性联系；③利用该结构实现了此前仅通过穷举搜索得到的非存在结果的理论化，消除了计算瓶颈；

**🔧 技术方法**

主要技术包括：射影几何中的cap理论、Gram矩阵与线性码核的关系、Bruen‑Wehlau对大cap的分类、以及对矩阵秩与cap周期结构的分析；

**📊 数据集**

本文并未使用任何实验数据集，而是完全基于理论推导和有限几何构造；

**📈 对比分析**

与先前的经验式或穷举搜索结果相比，本文得到的非存在定理与最佳距离值完全一致，并且在所有相关码长范围内提供了严格的理论上限和下限；

**⚠️ 局限性**

限制在于：该方法高度依赖于二进制场的cap分类，无法直接推广到 q>2 的情形（如三元LCD码），因此对非二进制情况仍需新的几何工具与理论。

---

## 632. Robots Need Some Education: On the complexity of learning in evolutionary robotics

**arXiv ID:** 2604.04196 | [PDF](https://arxiv.org/pdf/2604.04196v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 633. Towards Agentic Defect Reasoning: A Graph-Assisted Retrieval Framework for Laser Powder Bed Fusion

**arXiv ID:** 2604.04208 | [PDF](https://arxiv.org/pdf/2604.04208v1)

**作者:** Muhammad Rizwan Awan `[一作]` (Newcastle University), Shafiq Odhano `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个基于文献的图辅助检索框架，用以系统化分析激光粉末床熔融（LPBF）工艺中参数、机制与缺陷之间的关系。

**💡 创新点**

创新点在于将自然语言处理提取的实体关系映射为证据链接的知识图谱，并结合图检索与语义检索的混合策略，实现可解释的缺陷推理链。

**🔧 技术方法**

核心技术包括基于规则的实体与关系抽取、Sentence‑Transformers（all‑MiniLM‑L6‑v2）生成嵌入、FAISS索引、图检索（邻接探索）以及轻量级代理推理层，最终使用 Mistral‑7B‑Instruct 生成答案。

**📊 数据集**

使用的数据集为 50 篇公开的 Ti‑6Al‑4V LPBF 研究论文，通过文本清洗与分块生成约 220 词的上下文片段。

**📈 对比分析**

在 10 个基准缺陷推理问题上，系统检索准确率和召回率均达 0.9667，平均延迟约 6.4 s，证明了图辅助检索在提升检索质量方面的有效性。

**⚠️ 局限性**

主要局限包括：规则抽取可能漏检细粒度关系、仅评估单一材料与小规模基准集、代理层缺乏深度规划与多跳验证能力。

---

## 634. InferenceEvolve: Towards Automated Causal Effect Estimators through Self-Evolving AI

**arXiv ID:** 2604.04274 | [PDF](https://arxiv.org/pdf/2604.04274v1)

**作者:** Can Wang `[一作]` (Johns Hopkins University), Yiqun Chen `[通讯]` (Johns Hopkins University)

**通讯引用:** 5130 | [OpenAlex ID](https://openalex.org/A5100674349)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

无法确定

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

## 635. Multimodal Urban Tree Detection from Satellite and Street-Level Imagery via Annotation-Efficient Deep Learning Strategies

**arXiv ID:** 2604.03505 | [PDF](https://arxiv.org/pdf/2604.03505v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 636. RoboPhD: Evolving Diverse Complex Agents Under Tight Evaluation Budgets

**arXiv ID:** 2604.04347 | [PDF](https://arxiv.org/pdf/2604.04347v1)

**作者:** Andrew Borthwick `[一作]` (Independent Researchers), Anthony Galczak `[通讯]` (Independent Researchers)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 Elo 竞争的验证无验证（validation-free）进化框架 RoboPhD，用单一配置在四个不同任务上对比 GEPA 和 Autoresearch，展示在固定 1,500 次评估预算下 Elo 进化能获得更高的测试性能。

**💡 创新点**

创新点包括：① 用 Elo 竞争替代传统验证集，充分利用所有评估资源；② 引入自我仪表化（self‑instrumenting）诊断，使代理在进化过程中不断提升自我监测能力；③ Deep Focus 机制，将跨代信息保留在同一 LLM 会话中实现高效的细化。

**🔧 技术方法**

主要技术包括：大语言模型（Claude Code Opus 4.6）作为进化 AI；Elo 评分更新；比较错误分析报告；自我仪表化诊断；Deep Focus 迭代；以及基于 API 的多任务评估接口。

**📊 数据集**

使用四个基准数据集：ARC‑AGI（400 训练/400 测试）、Can't Be Late（2,000 训练/1,080 测试）、Text2SQL (BIRD)（6,601 训练/1,534 测试）和 DocFinQA（6,515 训练/922 测试）。

**📈 对比分析**

在同等信息、预算和评估框架下，RoboPhD 在 ARC‑AGI、Text2SQL 与 DocFinQA 三个 LLM 基准上分别超越 GEPA 与 Autoresearch，提升测试精度至 65.8%、64.5% 与 50.4%；在 Can't Be Late 任务中略逊于 Autoresearch，但整体性能竞争激烈。

**⚠️ 局限性**

局限性包括：对 LLM 生成结果的随机性影响实验稳定性；验证无验证方法在过拟合极端场景下的鲁棒性未知；Deep Focus 需要在单一会话中保留大量上下文，可能受 LLM 上下文窗口限制；以及对任务特定细节（如模拟器漏洞）缺乏防护，需要更严格的 sandboxing。

---

## 637. On the Rate Region of I.I.D. Discrete Signaling and Treating Interference as Noise for the Gaussian Broadcast Channel

**arXiv ID:** 2604.04092 | [PDF](https://arxiv.org/pdf/2604.04092v1)

**作者:** Yujie Shao `[一作]` (Shanghai Jiao Tong University), Min Qiu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 3850 | [OpenAlex ID](https://openalex.org/A5054214875)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出了一种基于离散调制（PAM）与干扰视作噪声（TIN）解码的高斯广播信道（GBC）编码方案，并证明其整个可达率域与GBC容量域相差常数；

**💡 创新点**

创新点在于：①直接针对GBC设计离散信号和TIN解码方案，避免使用线性确定性模型；②证明任意一点可达率对角都在一个常数位的距离内；③展示弱用户使用PAM有时可优于高斯调制；

**🔧 技术方法**

采用的技术包括：超叠加编码、均匀PAM调制、最小距离分析、信息量下界估计、时间共享（TS）辅助、常数间隙证明；

**📊 数据集**

本文未使用公开数据集，而是通过数值仿真验证可达率与容量的差距以及弱用户的相对优势；

**📈 对比分析**

通过与容量边界、SIC解码及Gaussian信号对比，发现：在常数位（约1.2–1.6 bit）内逼近容量；在某些SNR区间弱用户的PAM可超过Gaussian信号；

**⚠️ 局限性**

局限性包括：仅对两用户GBC做分析；对TS区间的间隙在某些点可无限大；未给出多用户扩展方案。

---

## 638. SPARK-IL: Spectral Retrieval-Augmented RAG for Knowledge-driven Deepfake Detection via Incremental Learning

**arXiv ID:** 2604.03833 | [PDF](https://arxiv.org/pdf/2604.03833v1)

**作者:** Hessen Bougueffa Eutamene `[一作]` (Univ. Polytechnique Hauts-de-France), Abdenour Hadid `[通讯]` (Sorbonne University)

**通讯引用:** 19631 | [OpenAlex ID](https://openalex.org/A5013928164)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于双路径频谱分析与检索增强的增量式深伪造检测框架SPARK-IL。

**💡 创新点**

创新点在于融合像素级与语义级多频段FFT+KAN模型并利用检索+多数投票实现非参数推理与增量学习。

**🔧 技术方法**

采用ViT-L/14半冻结、Kolmogorov–Arnold网络、FFT多频段分解、交叉注意力、Milvus检索和弹性权重整合等技术。

**📊 数据集**

使用UniversalFakeDetect基准数据集，包括19种GAN、扩散、面部替换等生成模型。

**📈 对比分析**

与现有方法对比，SPARK-IL在该基准上实现94.6%的平均准确率，优于REVEAL、UniFD等，显著提升跨生成器泛化。

**⚠️ 局限性**

局限性包括对局部化伪造痕迹的处理不足、频谱分割固定且不自适应，以及对极端压缩/后处理的鲁棒性有待提升。

---

## 639. Efficient Onboard Spacecraft Pose Estimation with Event Cameras and Neuromorphic Hardware

**arXiv ID:** 2604.04117 | [PDF](https://arxiv.org/pdf/2604.04117v1)

**作者:** Arunkumar Rathinam `[一作]` (University of Luxembourg), Djamila Aouada `[通讯]` (University of Luxembourg)

**通讯引用:** 2813 | [OpenAlex ID](https://openalex.org/A5083368272)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了使用事件摄像头与BrainChip Akida神经形态处理器实现的航天器相对6-DoF姿态估计端到端管线。

**💡 创新点**

首次在Akida硬件上完成完整事件驱动姿态估计，并比较量化感知的坐标回归与热图回归两种网络结构的性能。

**🔧 技术方法**

采用事件帧表示（E2F、2DHist、LNES）、MobileNet风格关键点回归网络、PnP求解、量化感知训练及SNN转换技术。

**📊 数据集**

使用SPADES合成事件数据集进行训练与测试。

**📈 对比分析**

在Akida V1上量化后坐标回归准确率大幅下降，V2热图回归保持与浮点相近的准确率（SPEED≈0.02–0.03），但帧率仅约2–3fps，功耗比Jetson Orin Nano低数倍。

**⚠️ 局限性**

V1硬件量化对坐标回归极为敏感，V2模型帧率较低，且实验仅在合成数据上验证，缺乏真实轨道数据的评估。

---

## 640. LRC codes over characteristic $2$

**arXiv ID:** 2604.04678 | [PDF](https://arxiv.org/pdf/2604.04678v1)

**作者:** Francisco Galluccio `[一作]` `[通讯]` (Universidad Nacional del Litoral), Francisco Galluccio (Universidad Nacional del Litoral)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文基于偶特征下的函数域塔构造了长度接近 q^4、维度与距离均为 O(q^4) 的 LRC 码，并给出了 q=4、8 以及任意大偶素数幂的显式实例。

**💡 创新点**

创新点在于提供通用构造框架，利用塔中拆分点的分裂结构实现了局部可恢复性 r = q-1 的大规模码，并在 q=4、8 等特例下得到最优参数，提升了相对参数 δ 与 R。

**🔧 技术方法**

技术上采用函数域塔、代数几何码与评估码理论、迹与范数性质、多项式插值以及拆分点分析等方法。

**📊 数据集**

本文为理论研究，没有使用实验数据集，所有结果均通过解析推导与符号计算得到。

**📈 对比分析**

通过将构造码的相对距离 δ 与比特率 R 与 GV、Barg–Tamo–Vladut 等经典界限进行比较，证明在大 q 或特定参数区间内 δ–R 曲线明显优于这些界限，性能优越。

**⚠️ 局限性**

局限性包括仅适用于偶特征、对拆分点分裂结构的高度依赖、缺乏 d≥0 的通用必要充足条件，以及在奇特征或小域下构造效果未知。

---

## 641. Love Me, Love My Label: Rethinking the Role of Labels in Prompt Retrieval for Visual In-Context Learning

**arXiv ID:** 2604.03657 | [PDF](https://arxiv.org/pdf/2604.03657v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 642. When Sinks Help or Hurt: Unified Framework for Attention Sink in Large Vision-Language Models

**arXiv ID:** 2604.03316 | [PDF](https://arxiv.org/pdf/2604.03316v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 643. AI Trust OS -- A Continuous Governance Framework for Autonomous AI Observability and Zero-Trust Compliance in Enterprise Environments

**arXiv ID:** 2604.04749 | [PDF](https://arxiv.org/pdf/2604.04749v1)

**作者:** Eranga Bandara `[一作]` (Old Dominion University), Kasun De Zoysa `[通讯]` (University of Colombo)

**通讯引用:** 1032 | [OpenAlex ID](https://openalex.org/A5055400093)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了 AI Trust OS——一个基于持续遥测、零信任边界和 LLM 文档生成的企业 AI 治理框架。

**💡 创新点**

创新点包括：将治理从周期性声明转向实时遥测、实现自动的 Shadow AI 探测、采用零信任的只读探针获取配置元数据、以及使用 LLM 将机器验证的控制结果转换为可审计的合规报告。

**🔧 技术方法**

技术方案涵盖：零信任遥测边界、异步探针执行（BullMQ + Upstash Redis）、LLM 辅助合规文档合成（GPT‑4o‑mini）、多租户 Postgres 证据账本、Next.js 边缘部署与 Clerk 身份验证。

**📊 数据集**

评估数据来自一家中型金融机构的真实生产环境，包括 8 个服务提供商（AWS、GitHub、Okta、Stripe、Vercel、LangSmith、Datadog LLM、AWS Bedrock）的遥测与配置日志。

**📈 对比分析**

相较于传统手工审计与单一合规平台，AI Trust OS 在一次扫描中完成 5 个监管框架（SOC 2、ISO 27001、ISO 42001、EU AI Act、HIPAA）的证据收集，探针平均响应 <2.5 s，LLM 文档合成 <4 s，且自动发现了 1 个未登记的细调模型，展现出更高的覆盖率和更快的响应速度。

**⚠️ 局限性**

局限性在于仅对单一工作区进行评估，未覆盖所有模块（如 Board Report 渲染、Evidence Explorer 等），且缺乏跨组织的纵向多租户实验，需要进一步验证在不同规模和法规环境下的可扩展性与持久性。

---

## 644. Bridging the Dimensionality Gap: A Taxonomy and Survey of 2D Vision Model Adaptation for 3D Analysis

**arXiv ID:** 2604.03334 | [PDF](https://arxiv.org/pdf/2604.03334v1)

**作者:** Akshat Pandya `[一作]` (Independent Researcher), Bhavuk Jain `[通讯]` (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了将2D视觉模型迁移到3D分析的三大策略（数据中心、架构中心、混合）并提供统一分类及未来方向

**💡 创新点**

提出了基于“维度鸿沟”三家族的新分类框架，并系统评估各类方法在效率、几何保真度和预训练利用方面的权衡

**🔧 技术方法**

综述了多视图渲染、体素化、几何展开、3D卷积、点云网络、图网络、Transformer、知识蒸馏等多种技术

**📊 数据集**

讨论了多种公开数据集（ModelNet、ShapeNet、KITTI、Waymo、ScanNet 等）

**📈 对比分析**

对比了不同方法在目标检测、分割、分类等任务上的表现，指出数据中心方法在效率上领先，架构中心在几何精度上最好，混合方法兼顾两者，整体性能呈现折衷关系

**⚠️ 局限性**

局限在于缺乏统一的 3D 预训练基准、对大规模稀疏数据的计算开销高、跨模态融合仍浅显、动态场景和隐式表示的整合不足

---

## 645. Reproducibility study on how to find Spurious Correlations, Shortcut Learning, Clever Hans or Group-Distributional non-robustness and how to fix them

**arXiv ID:** 2604.04518 | [PDF](https://arxiv.org/pdf/2604.04518v1)

**作者:** Ole Delzer `[一作]` (Technische Universität Berlin), Sidney Bender `[通讯]` (Technische Universität Berlin)

**通讯引用:** 13 | [OpenAlex ID](https://openalex.org/A5022980918)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过重新实现和对比多种纠正 Clever Hans 行为的方法，重点评估 XAI 基础方法 CFKD、P‑ClArC、RR‑ClArC 与传统非 XAI 方法 DFR、Group DRO 在数据有限、严重偏差的安全关键图像分类任务中的效果。

**💡 创新点**

创新点在于：①统一不同社区的术语，构建符合实际约束的公平评估框架；②利用 SpRAy 自动聚类获得组标签；③系统比较 XAI 与非 XAI 方法，发现 CFKD 在多种数据集上最稳健。

**🔧 技术方法**

采用的技术包括 XAI 方法（LRP、SpRAy、CAV）、对抗/生成式 counterfactual 生成器 SCE、Group DRO、DFR、深度特征重权重等。

**📊 数据集**

使用的数据集有合成数据（Squares、Smiling）与真实数据（CelebA Blond、Smiling、Camelyon17、Follicles），每个数据集均包含对称/不对称污染版本。

**📈 对比分析**

评估采用平均组准确率（AGA）和最差组准确率（WGA）作为指标，训练与验证受限样本下进行模型选择。实验显示 XAI 方法普遍优于非 XAI，CFKD 在 6/9 数据集获得最高 AGA，P‑ClArC 在所有数据集均有提升，Group DRO、DFR 效果受限。

**⚠️ 局限性**

局限性包括：①对准确组标签依赖度高，SpRAy 在复杂 confounder 上表现不稳定；②数据稀缺导致验证集样本极少，模型选择不稳；③方法对超参数敏感；④仅评估二分类图像任务，未验证多类或其他模态；⑤CFKD 计算成本较高。

---

## 646. frax: Fast Robot Kinematics and Dynamics in JAX

**arXiv ID:** 2604.04310 | [PDF](https://arxiv.org/pdf/2604.04310v1)

**作者:** Daniel Morton `[一作]` (Stanford University), Marco Pavone `[通讯]` (Stanford University)

**通讯引用:** 11612 | [OpenAlex ID](https://openalex.org/A5050003000)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本论文提出了FRAX，一个基于JAX的纯Python机器人运动学与动力学库，利用向量化实现可在CPU、GPU和TPU上高效计算；

**💡 创新点**

创新点在于将刚体动力学算法（CRBA、RNEA、ABA）完全向量化，利用ancestor mask消除递归循环，支持JIT编译和自动微分，同时保持跨平台兼容性；

**🔧 技术方法**

采用JAX框架、空间代数表示、矩阵广播、ancestor mask、向量化递归、JIT编译、自动微分（JVP、VJP）等技术；

**📊 数据集**

使用Frank Panda机械臂和Unitree G1人形机器人（URDF模型）进行评测；

**📈 对比分析**

与Pinocchio、MuJoCo的Python接口及其C++实现、MJX、BRAX等库进行比较。FRAX在CPU上单实例逆运动学约6–12 μs，逆动力学约10–42 μs，比Python接口快2–3×，接近C++速度；GPU批量评测显示可达1亿+次/秒；JIT编译时间仅1–2 s，远低于MJX/BRAX的6–12 s；

**⚠️ 局限性**

库功能相对简化，缺少全局IK求解、复杂碰撞几何（如胶囊、盒子、椭球体）支持；向量化导致O(n²)复杂度，对极深树或高DOF机器人可能产生额外计算；缺乏ABA、直接逆矩阵求解等高级功能。

---

## 647. Sim2Real-AD: A Modular Sim-to-Real Framework for Deploying VLM-Guided Reinforcement Learning in Real-World Autonomous Driving

**arXiv ID:** 2604.03497 | [PDF](https://arxiv.org/pdf/2604.03497v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 648. Physics-Informed Transformer for Real-Time High-Fidelity Topology Optimization

**arXiv ID:** 2604.03522 | [PDF](https://arxiv.org/pdf/2604.03522v1)

**作者:** Aaron Lutheran `[一作]` (University of North Carolina at Charlotte), Alireza Tabarraei `[通讯]` (University of North Carolina at Charlotte)

**通讯引用:** 1564 | [OpenAlex ID](https://openalex.org/A5032056958)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于 Vision Transformer 的非迭代拓扑优化框架，直接从物理信息（边界条件、载荷、应力应变场和体积分数）预测最佳材料分布。

**💡 创新点**

核心创新包括：① 用全局自注意力建模结构力学中的非局部相互作用；② 通过条件标记嵌入全局问题参数；③ 引入体积分数、载荷一致性和可微浮动材料损失以强制物理一致性；④ 通过频域编码与迁移学习将模型扩展至动态载荷。

**🔧 技术方法**

技术手段为 Vision Transformer（ViT）架构、Patch Token 化、全局自注意力、多头注意力、残差连接、MLP 以及多任务损失训练；动态场景采用 FFT 频谱特征与迁移学习。

**📊 数据集**

使用 30,000 个 64×64 网格的静态数据集（随机边界载荷、体积分数 30–50%）和 6,000 个 2D 动态载荷（正弦/冲击）数据集；数据通过对称旋转/镜像扩增，形成约 240,000 个训练样本。

**📈 对比分析**

与 TopologyGAN、TopoDiff 等迭代/生成模型比较：在静态任务中 ViT‑Small‑4 在 1.86% 的平均合规误差、0.32% 的中位误差以及 6.6% 的浮动材料误差（经后处理可降至 0.8%）已超过传统方法；动态任务中通过迁移学习的解码层微调，平均合规误差降至 4.81%，但浮动材料误差仍高达 48%。

**⚠️ 局限性**

局限性包括：① 对单载荷、二维结构有限，无法直接处理多载荷或三维体积问题；② 仍难以在动态场景下保证全局连通性和几何锐度；③ 对体积分数外推不稳定，无法很好泛化至训练分布之外；④ 依赖结构化网格，无法直接应用于非结构化/自适应网格。

---

## 649. Semantics Over Syntax: Uncovering Pre-Authentication 5G Baseband Vulnerabilities

**arXiv ID:** 2604.04283 | [PDF](https://arxiv.org/pdf/2604.04283v1)

**作者:** Qiqing Huang `[一作]` (University at Buffalo), Hongxin Hu `[通讯]` (University at Buffalo)

**通讯引用:** 6575 | [OpenAlex ID](https://openalex.org/A5056657952)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于规范约束的语义测试框架（Constraint-Guided Semantic Testing，CGST），通过解析3GPP 5G RRC规范中的字段约束和跨字段依赖，生成在语法上合法但语义不一致的RRC消息，来发现UE基带实现中的逻辑缺陷。

**💡 创新点**

创新点在于：①建立了四类字段约束分类（取值范围、字段可选性、单IE内部依赖、跨IE依赖）；②利用大模型在证据限制下从自然语言规范中自动推断跨字段语义关系；③将抽取的约束统一归一为DSL，实现精准、可追溯的测试案例生成。

**🔧 技术方法**

技术主要包括：ASN.1 PER解析与双层抽象表示、证据绑定LLM（GPT‑4o）推理、DSL规则规范化、最小化语义违规编辑、基于USRP的OTA注入以及GDB/ADB日志监控。

**📊 数据集**

数据集为3GPP TS 38.331及38.2xx系列规范（约万页）作为证据源，以及在OAI UE仿真环境和真实手机设备（共计8款）上收集的RRC基带日志。

**📈 对比分析**

对比方法：与全枚举对OAI UE的语义触发效果做对比，CGST仅生成1,458个测试输入完成度达约95%，而枚举需30,600输入并耗时约16天；在商用手机上发现7个高危漏洞（3个CVE），展示了高效的漏洞挖掘能力。

**⚠️ 局限性**

局限性包括：只能处理单条消息内部语义；DSL未覆盖跨消息状态、时间约束和能力协商等；LLM推理受限于证据可用性，可能遗漏或误判某些隐式依赖；对非ASN.1或结构复杂的协议适用性需进一步验证。

---

## 650. Personality Requires Struggle: Three Regimes of the Baldwin Effect in Neuroevolved Chess Agents

**arXiv ID:** 2604.03565 | [PDF](https://arxiv.org/pdf/2604.03565v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 651. Profile-Then-Reason: Bounded Semantic Complexity for Tool-Augmented Language Agents

**arXiv ID:** 2604.04131 | [PDF](https://arxiv.org/pdf/2604.04131v1)

**作者:** Paulo Akira F. Enabe `[一作]` `[通讯]` (University of São Paulo), Paulo Akira F. Enabe (University of São Paulo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种 Profile-Then-Reason（PTR）框架，先用大语言模型生成可执行工作流，再通过确定性执行、验证与一次性修复控制语言模型调用次数最多为三次。

**💡 创新点**

创新点在于将语义规划与工具执行解耦，构建受限执行预算并通过验证器与有限修复实现结构化任务中的高效与鲁棒性。

**🔧 技术方法**

使用基于Transformer的大语言模型（GPT‑4o‑Mini、Claude Haiku、GPT‑5.4、Claude Sonnet）、工具调用、自动参数解析、分支规则、错误恢复和验证器等技术。

**📊 数据集**

在六个基准数据集上进行实验：TriviaQA、NQ‑Open、StrategyQA、GSM8K、AQuA‑RAT 和 HotPotQA。

**📈 对比分析**

与 ReAct 对照，采用准确率（EM）、调用次数、成本等指标评估；PTR 在检索类和分解任务中占优，16/24 配置取得优势，平均 EM 提升约 0.27，整体成本降低约 12%，但在多跳检索与符号推理任务上不占优势。

**⚠️ 局限性**

局限性：仅适用于可预先规划的结构化工作流，对需要在线调整或连续推理的任务（如 HotPotQA、AQuA‑RAT）效果差，且修复机制仅限一次，缺乏更灵活的迭代规划能力。

---

## 652. GENSERVE: Efficient Co-Serving of Heterogeneous Diffusion Model Workloads

**arXiv ID:** 2604.04335 | [PDF](https://arxiv.org/pdf/2604.04335v1)

**作者:** Fanjiang Ye `[一作]` (Rice University), Yuke Wang `[通讯]` (Rice University)

**通讯引用:** 106 | [OpenAlex ID](https://openalex.org/A5022196610)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种能够在同一 GPU 集群上高效共服务文本到图像和文本到视频扩散模型的系统；

**💡 创新点**

创新点在于利用扩散模型可预知的逐步计算特性，实现基于步骤的抢占、弹性序列并行和 SLO 目标的动态批处理；

**🔧 技术方法**

技术方案包括对每个 denoising 步骤的成本建模、在步骤边界进行无损抢占、根据分辨率动态切换序列并行度、以及基于动态规划的全局资源调度；

**📊 数据集**

使用 DiffusionDB 的图像提示和 VBench 的视频提示作为实验数据集；

**📈 对比分析**

与 FIFO、SJF、SRTF、RASP 等基线相比，系统在多种工作负载和 SLO 级别下的 SLO 达成率最高，最高可提升约 44%（整体 90% 对比 83%）并显著缩短图像/视频的延迟；

**⚠️ 局限性**

局限性包括只针对基于 DiT 的扩散模型，假设步骤成本稳定，且对极大规模 GPU 集群的可扩展性尚未验证。

---

## 653. Rethinking Exposure Correction for Spatially Non-uniform Degradation

**arXiv ID:** 2604.04136 | [PDF](https://arxiv.org/pdf/2604.04136v1)

**作者:** Ao Li `[一作]` (Xidian University), Weisheng Dong `[通讯]` (Xidian University)

**通讯引用:** 11235 | [OpenAlex ID](https://openalex.org/A5037310802)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种面向空间非均匀曝光纠正的新范式，利用空间信号编码器预测稠密调制权重，指导多维 LUT 变换，并加入 HSL 颜色补偿和不确定性驱动的非均匀损失。

**💡 创新点**

创新点包括：①将非均匀调制估计与图像变换解耦，使用 3D LUT 进行精细局部校正；②引入 HSL 空间提升色彩保真；③基于不确定性模型的空间加权损失，使训练聚焦于不同曝光强度区域。

**🔧 技术方法**

使用的技术包括：空间信号编码器（CNN）、多通道 3D LUT、HSL 颜色补偿分支、基于拉普拉斯分布的非均匀损失，以及自注意力/卷积下采样。

**📊 数据集**

采用的公开数据集有：MSEC、SICEV2、LCDP、REED 等四个曝光纠正基准。

**📈 对比分析**

与包括 CoTF、LACT、FECNet、LCDPNet、MMHT、CSEC、UEC 等多种 SOTA 方法进行对比，实验结果显示在 PSNR/SSIM/LPSIP/NIE 等指标上均取得领先或接近最佳的性能。

**⚠️ 局限性**

局限性在于对大幅度全局曝光失真仍可能产生残留色偏，且 3D LUT 需要存储与计算资源，未来需进一步提升实时性与对极端曝光场景的鲁棒性。

---

## 654. 'Layer su Layer': Identifying and Disambiguating the Italian NPN Construction in BERT's family

**arXiv ID:** 2604.03673 | [PDF](https://arxiv.org/pdf/2604.03673v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 655. From High-Level Types to Low-Level Monitors: Synthesizing Verified Runtime Checkers for MAVLink

**arXiv ID:** 2604.03886 | [PDF](https://arxiv.org/pdf/2604.03886v1)

**作者:** Arthur Amorim `[一作]` (University of Central Florida), Lance Joneckis `[通讯]` (Idaho National Laboratory)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5107526368)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出了 Platum 框架，能够将高层 RMPST 规范直接合成资源受限 UAV 的 C 语言运行时监控器。

**💡 创新点**

创新点在于将规范与结构检查分离，使用 Meta-F* 反射决策程序实现自动结构合法性校验，并直接生成无 GC 的 C FSM。

**🔧 技术方法**

采用 F* 语言嵌入 DSL、Meta-F* 反射、Refined Multiparty Session Types、Python XML 转 F* 转换器、C 代码合成。

**📊 数据集**

使用 MAVLink 任务上传子协议的 XML 定义和 ArduPilot SITL 仿真数据集。

**📈 对比分析**

通过与 DATUM（OCaml 提取）对比，测得监控器延迟约 4 倍下降、RSS 内存占用降低；系统延迟提升仅 4.22 μs。

**⚠️ 局限性**

局限在于尚未实现去中心化监控、缺少完整的终端到终端可达性证明，并依赖 Python 代理实现。

---

## 656. Rethinking Token Prediction: Tree-Structured Diffusion Language Model

**arXiv ID:** 2604.03537 | [PDF](https://arxiv.org/pdf/2604.03537v1)

**作者:** Zihao Wu `[一作]` (Duke University), Vahid Tarokh `[通讯]` (Duke University)

**通讯引用:** 33170 | [OpenAlex ID](https://openalex.org/A5020766546)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于词表树的离散扩散语言模型（TDLM），通过对词根的层级子节点预测取代全词预测；

**💡 创新点**

创新点在于利用词表层级结构，将输出空间从全词表压缩到子节点集合，显著降低模型参数与显存需求；

**🔧 技术方法**

使用连续时间马尔科夫链（CTMC）框架实现树结构的扩散过程，结合离散扩散模型的训练ELBO；

**📊 数据集**

使用OpenWebText数据集进行实验；

**📈 对比分析**

与MDLM、GIDD、HDLM等基线比较，TDLM在相同参数预算下实现更低的验证和生成困惑度，同时将峰值GPU显存削减约一半；

**⚠️ 局限性**

局限性包括在小规模模型上表现略逊，且对词表树的构建超参数（分支因子、深度、簇大小）较敏感，需要进一步优化。

---

## 657. Robust LLM Performance Certification via Constrained Maximum Likelihood Estimation

**arXiv ID:** 2604.03257 | [PDF](https://arxiv.org/pdf/2604.03257v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 658. Safe Decentralized Operation of EV Virtual Power Plant with Limited Network Visibility via Multi-Agent Reinforcement Learning

**arXiv ID:** 2604.03278 | [PDF](https://arxiv.org/pdf/2604.03278v1)

**作者:** Chenghao Huang `[一作]` (Monash University), Hao Wang `[通讯]` (Monash University)

**通讯引用:** 28417 | [OpenAlex ID](https://openalex.org/A5100446064)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在电网可视性有限的情况下，虚拟电厂对多家电动车充电站的分布式充电进行安全高效的协同调度。

**💡 创新点**

创新点包括将Transformer用于时间序列观察编码，和基于拉格朗日乘子实现的安全约束正则化，从而在缺乏全网状态信息时仍能有效约束电压安全。

**🔧 技术方法**

采用了Transformer辅助的Lagrangian多智能体近端策略优化（TL‑MAPPO），结合集中训练、分散执行和安全约束学习。

**📊 数据集**

使用的实验数据包括IEEE 33节点配电网、Caltech收集的约90万台电动车充电需求数据、Ausgrid的光伏发电数据以及AEMO批发电价和Melbourne时段电价。

**📈 对比分析**

与MAPPO、MATD3、MASAC三种主流多智能体强化学习基线对比，TL‑MAPPO在一天运行的总能源成本降低约10%，电压违规率下降约45%，需求不满足率降低约35%，且学习稳定性和收敛速度均优于基线。

**⚠️ 局限性**

局限性在于实验仅在四台充电站、33节点网络上验证，未探讨更大规模部署和通信效率的挑战。

---

## 659. Latency-Aware Resource Allocation over Heterogeneous Networks: A Lorentz-Invariant Market Mechanism

**arXiv ID:** 2604.03897 | [PDF](https://arxiv.org/pdf/2604.03897v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 660. Toward Full Autonomous Laboratory Instrumentation Control with Large Language Models

**arXiv ID:** 2604.03286 | [PDF](https://arxiv.org/pdf/2604.03286v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 661. LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection

**arXiv ID:** 2604.04815 | [PDF](https://arxiv.org/pdf/2604.04815v1)

**作者:** Cheng Xu `[一作]` (University), M-Tahar Kechadi `[通讯]` (University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出LiveFact动态时序化假新闻检测基准，模拟“战雾”环境并评估LLM的推理与知识检索能力。

**💡 创新点**

创新点在于连续更新的新闻数据、时间切片证据集、双模式评估（分类与推理）以及基于实体移位的SSA框架，用以监控Benchmark Data Contamination并区分记忆与推理。

**🔧 技术方法**

使用了月度事件检索、时间感知证据构造、LLM自动生成声称与背景、实体移位SSA技术、以及混合专家（MoE）和稠密模型的性能对比。

**📊 数据集**

基于实时新闻事件构建的三时段（T-3、T、T+3天）证据集，人工审核后的4,392条断言数据，涵盖Real、Fake、Ambiguous三类标签。

**📈 对比分析**

对18个LLM（包括开源MoE Qwen3、Llama、DeepSeek以及闭源GPT系列）在Classification与Inference两模式下进行评测，Qwen3-235B-A22B获得最高平均分，显著优于闭源SOTA；中等规模MoE模型在成本与性能上实现最佳折中。

**⚠️ 局限性**

局限性包括仅使用英文文本、缺乏多模态与多语言覆盖，以及人工审核瓶颈导致扩展性受限。

---

## 662. 1.x-Distill: Breaking the Diversity, Quality, and Efficiency Barrier in Distribution Matching Distillation

**arXiv ID:** 2604.04018 | [PDF](https://arxiv.org/pdf/2604.04018v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 663. Olmo Hybrid: From Theory to Practice and Back

**arXiv ID:** 2604.03444 | [PDF](https://arxiv.org/pdf/2604.03444v1)

**作者:** William Merrill `[一作]` (Allen Institute for AI), Ashish Sabharwal `[通讯]` (Allen Institute for AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的混合模型架构，结合了Gated DeltaNet（GDN）层和注意力层，取代传统Transformer中的滑动窗口注意力，构建了与Olmo 3相似的7B参数模型并在6T训练令牌上实现了显著的标记效率提升；

**💡 创新点**

创新点在于证明混合模型在理论上具备比纯Transformer和纯线性RNN更强的可表达性，并通过实验显示其在多任务、长上下文和代码评估等方面的性能超越；

**🔧 技术方法**

技术上主要采用GDN层的负特征值扩展、3:1层级交替混合策略、DroPE位置编码去除以及Chinchilla风格的缩放律拟合；

**📊 数据集**

使用的数据集包括Common Crawl、Dolma 3（Books、Common Crawl、pes2o、Reddit、Stack、Wiki等）以及针对长上下文的RULER、MMLU等评测集合；

**📈 对比分析**

通过与Olmo 3、Nemotron‑H、Falcon H1等公开模型在相同参数/计算量下的对比实验，显示在相同标记量下取得相同或更优的MMLU、BBH、RULER等指标，且在长上下文推理上提升约14.1%；

**⚠️ 局限性**

限制方面包括对混合架构的数值稳定性和推理吞吐量的挑战，以及对安全性评估的缺乏；

---

## 664. Dynamic Free-Rider Detection in Federated Learning via Simulated Attack Patterns

**arXiv ID:** 2604.04611 | [PDF](https://arxiv.org/pdf/2604.04611v1)

**作者:** Motoki Nakamura `[一作]` `[通讯]` (Fujitsu Limited), Motoki Nakamura (Fujitsu Limited)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文提出一种名为S2-WEF的动态免费骑手检测方法，利用服务器端模拟全局模型相似攻击的WEF模式并与客户端提交的WEF矩阵比较，实现无代理数据、无预训练的逐轮检测。

**💡 创新点**

创新点在于将原有WEF防御与服务器端WEF模拟相结合，构建双分数（相似度+偏差）聚类、阈值与多数投票相融合的检测框架，能够识别动态及全局模型伪装攻击（DWA、AWCA）。

**🔧 技术方法**

主要技术包括Weight Evolving Frequency (WEF) 矩阵计算、服务器端WEF模拟、余弦相似度与L1距离、Dev偏差得分、稳健标准化、层次聚类、阈值分类与多数投票。

**📊 数据集**

使用了 MNIST、ADULT 和 CIFAR-10 三个公开数据集。

**📈 对比分析**

与 WEF-Defense、STD-DAGMM 等基线对比，在三种数据分布、五种攻击、两种动态情景下，S2-WEF 在超过80% 设置下 F1 分数接近 1，尤其在 DWA、AWCA 等全局模型伪装攻击上提升高达 +0.96；整体误报率显著降低。

**⚠️ 局限性**

限制包括对数据与模型同质性要求较高，跨设备规模大时计算量可观；在极端非 IID、架构异质或强对抗学习场景下鲁棒性仍有限；方法依赖免费骑手比例小于 50% 的假设，若失效则易被规避。

---

## 665. Online learning of smooth functions on $\mathbb{R}$

**arXiv ID:** 2604.03525 | [PDF](https://arxiv.org/pdf/2604.03525v1)

**作者:** Jesse Geneson `[一作]`, Alexander Wang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文研究在无界域ℝ上对实值平滑函数的对抗性在线学习，证明传统误差模型在此域上不可解，并提出三种局部性约束或加权损失的改进情景，给出各情景下的极限误差上界和下界，揭示一维与高维之间的根本差异。

**💡 创新点**

创新点在于：①首次阐明无界域上连续可导函数类的学习本质不可解；②设计三种局部约束/加权机制（输入距离限制、自由猜测、距离衰减权重）并对其进行完全理论分析；③给出精确的阈值判定（如g(z)=e^{-cz}可恢复有限损失，g(z)=1/z等情况）以及高维切片类的无限损失证明。

**🔧 技术方法**

采用的技术主要包括：对抗性构造与极限分析、切片函数构造、尺度变换与函数空间不嵌套性证明、插值算法与动量证明、权重函数比较与上界估计、以及多维几何构造（分散矩形“山峰”）来展示无限损失。

**📊 数据集**

无使用实测数据集；全部基于理论构造与严谨证明。

**📈 对比分析**

与传统[0,1]区间下的误差模型相比，本文在无界域上给出了更细致的极限损失评估：在情景1、2下，当p≥q≥2时损失可降至1；当p<q时仍为无穷；情景3中根据权重g的衰减速率可获得有限或无限损失。

**⚠️ 局限性**

局限性包括：仅在理论层面给出极限，缺乏实验验证；高维情况仅针对切片类给出不可学习结论，未探讨更一般的梯度约束类；加权机制对实际实现的可行性与计算复杂性未作讨论；在多维高分辨率场景下可能仍存在无法修正的不可学习性。

---

## 666. Incompleteness of AI Safety Verification via Kolmogorov Complexity

**arXiv ID:** 2604.04876 | [PDF](https://arxiv.org/pdf/2604.04876v1)

**作者:** Munawar Hasan `[一作]` `[通讯]` (Michigan Technological University), Munawar Hasan (Michigan Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

阐述了使用 Kolmogorov 复杂度对 AI 安全验证的理论限制，并证明了任何固定的可证明系统都无法验证所有满足安全政策的高复杂度实例。

**💡 创新点**

提出了基于信息论的 AI 政策验证不完全性结果，揭示了安全验证的根本信息理论极限，推动了实例级证明式验证的研究方向。

**🔧 技术方法**

利用 Kolmogorov 复杂度理论、可枚举形式化理论以及证明系统的形式化建模进行理论证明。

**📊 数据集**

本工作为纯理论分析，无使用具体数据集。

**📈 对比分析**

没有实验比较，主要通过数学证明展示了在任意足够大的复杂度阈值下，任何固定验证器都无法证明所有满足政策的实例。

**⚠️ 局限性**

限制在于信息理论上任何固定可证明系统的描述容量有限，导致无法覆盖所有高复杂度的安全合规实例，凸显了安全验证的根本不可完备性。

---

## 667. Stochastic Generative Plug-and-Play Priors

**arXiv ID:** 2604.03603 | [PDF](https://arxiv.org/pdf/2604.03603v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 668. Joint Behavior-guided and Modality-coherence Conditional Graph Diffusion Denoising for Multi Modal Recommendation

**arXiv ID:** 2604.03654 | [PDF](https://arxiv.org/pdf/2604.03654v1)

**作者:** Xiangchen Pan `[一作]` (Huazhong University of Science and Technology), Wei Wei `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 254399 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于行为引导和模态一致性的条件图扩散模型（JBM‑Diff），实现对多模态特征和用户行为反馈的联合去噪与去偏。

**💡 创新点**

创新点在于：①引入行为向量作为条件，使用扩散模型逐步去除与用户偏好无关的模态噪声；②构建多视图消息传播与特征融合机制，并通过模态一致性对样本对进行信度赋权，实现数据级去偏；③设计跨模态对齐与自监督对比学习，进一步对齐协同与语义特征。

**🔧 技术方法**

核心技术包括：条件扩散模型（diffusion model）、图卷积网络（GCN）进行多视图消息传播、kNN稀疏化的语义邻接图、基于模态一致性的行为去偏权重、以及对比学习和BPR损失的联合优化。

**📊 数据集**

实验使用了Amazon Review Dataset的Baby、Sports和Clothing三大公开子集，分别包含数千用户与数万物品的交互和视觉/文本模态特征。

**📈 对比分析**

与传统协同过滤、图推荐、超图、对比学习和其他扩散模型基线相比，JBM‑Diff在Recall@K和NDCG@K上均取得最高分，提升幅度可达约8%~10%，并在不同噪声注入场景下表现出更强的鲁棒性。

**⚠️ 局限性**

局限性包括：①模型对模态质量高度依赖，若模态信息缺失或极度噪声时效果有限；②需要额外的模态特征和行为条件，增加数据预处理与存储成本；③扩散与GCN模块虽然参数相对较少，但整体训练时间仍高于纯图模型，且在大规模稀疏数据上扩展性待进一步验证。

---

## 669. Occupational Diversity and Stratification in Platform Work: A Longitudinal Study of Online Freelancers

**arXiv ID:** 2604.03517 | [PDF](https://arxiv.org/pdf/2604.03517v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 670. KiToke: Kernel-based Interval-aware Token Compression for Video Large Language Models

**arXiv ID:** 2604.03414 | [PDF](https://arxiv.org/pdf/2604.03414v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 671. Unmasking Hallucinations: A Causal Graph-Attention Perspective on Factual Reliability in Large Language Models

**arXiv ID:** 2604.04020 | [PDF](https://arxiv.org/pdf/2604.04020v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 672. Less Detail, Better Answers: Degradation-Driven Prompting for VQA

**arXiv ID:** 2604.04838 | [PDF](https://arxiv.org/pdf/2604.04838v1)

**作者:** Haoxuan Han `[一作]` (Zhejiang University), Bohan Zhuang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种 Degradation-Driven Prompting (DDP) 框架，通过降解图像分辨率、结构化提示与工具调用，提升 Vision‑Language Models 在视觉问答与视觉幻觉任务中的鲁棒性。

**💡 创新点**

创新点包括：①将图像降解作为结构瓶颈，强制模型关注全局结构；②多阶段任务分类+工具管理，实现视觉特征分离与主动验证；③结合外部工具（掩模、辅助线、模糊、裁剪）与细粒度视觉提示；④使用 In‑Context Learning 以调节模型关注点。

**🔧 技术方法**

使用的技术有：多尺度降采样与高斯平滑、掩模与辅助线工具、红框/白色背景遮罩、模糊掩模、裁剪、结构化视觉提示、agentic 调用机制、In‑Context Learning，以及 Gemini‑3‑Pro 等 VLM 作为后端。

**📊 数据集**

使用的数据集包括 MME、SEED‑Bench、ScienceQA、VQAv2、MMBench、V*Bench、ColorBlind、以及 DataCV CVPR Challenge 的 Track‑1/Track‑2。

**📈 对比分析**

通过与 LLaVA‑v1.5、Qwen‑VL、Gemini‑1.5‑Pro、GPT‑4o 等基线模型的对比实验，DDP 在多项基准上显著领先，示例结果：MMBench 92.1%、SEED‑Bench 94.5%、SciQA 99.1%、VQAv2 89.4%，并在 V*Bench、ColorBlind 等任务中实现大幅提升。

**⚠️ 局限性**

局限性包括：对极低分辨率图像可能导致细节丢失，影响小目标识别；需预先构建并维护工具库，部署成本与延迟较高；实验主要基于公开数据集，真实世界复杂光照、噪声和遮挡下的鲁棒性尚未充分验证；模型对外部工具的依赖降低了可解释性与泛化能力。

---

## 673. Impure codes exceeding the pure bounds for quantum local recovery

**arXiv ID:** 2604.03569 | [PDF](https://arxiv.org/pdf/2604.03569v1)

**作者:** Carlos Galindo `[一作]` (Universitat Jaume I), Ryutaroh Matsumoto `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 1635 | [OpenAlex ID](https://openalex.org/A5005174170)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

构造了一族基于J‑仿射变换码的CSS量子本地可恢复码（Q-LRC），并证明这些码为非纯（impure）码，能够在多擦除（δ≥3）场景下突破已知的纯量子本地可恢复码界限。

**💡 创新点**

创新点在于：①首次将J‑仿射变换码与降阶多项式（weighted Reed‑Muller）和下降单项式–笛卡尔码相结合；②利用Feng‑Rao界对余类距离（coset distance）进行精确估计，从而保证码的非纯性；③证明在满足特定参数条件下，构造的量子码能明显超越纯量子LRC的Singleton‑、Griesmer‑和Plotkin‑型界限。

**🔧 技术方法**

主要技术包括：代数几何码理论（J‑仿射变换码、格罗纳伯格基、格罗曼多多项式）；CSS构造；Feng‑Rao距离估计；余类距离和相对广义汉明权重分析；对比纯量子LRC的Singleton‑型界限。

**📊 数据集**

该工作为纯理论构造，不使用实验数据集，所有结论均基于代数证明和符号计算。

**📈 对比分析**

通过与纯量子LRC的三类界限（Singleton‑、Griesmer‑、Plotkin‑）比较，展示了在给定长度、维数、距离和局部性参数下，所构造的impure码在这些界限上至少超越了一个常数量级，且在多擦除（δ=3）情况下依然保持可恢复性。

**⚠️ 局限性**

限制：目前仅针对纯量子LRC的界限给出了违背证明；缺乏针对impure多擦除码的上界；算法实现的复杂度与编码效率未在实验上验证；需要进一步研究如何在保持局部性不变的前提下设计更高距离的impure码。

---

## 674. Saliency-R1: Enforcing Interpretable and Faithful Vision-language Reasoning via Saliency-map Alignment Reward

**arXiv ID:** 2604.04500 | [PDF](https://arxiv.org/pdf/2604.04500v1)

**作者:** Shizhan Gong `[一作]` (Chinese University of Hong Kong), Qi Dou `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 28198 | [OpenAlex ID](https://openalex.org/A5090516040)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Saliency‑R1 框架，通过将可解释的 saliency map 与人工标注的边框对齐，提升视觉‑语言模型（VLM）的推理可信度与可解释性。

**💡 创新点**

创新点在于（1）基于 logits 分解的实时 saliency map 生成方法，免去梯度或多次前向传播；（2）利用注意力 roll‑out 将视觉信息在思考层的流向可视化；（3）将 saliency 与边框的一致性作为奖励，使用 Group Relative Policy Optimization（GRPO）在后训练阶段对齐注意力。

**🔧 技术方法**

技术包括 Transformer logits 分解、注意力 roll‑out、GRPO 强化学习、LoRA 微调、LLM‑as‑judge（GPT‑4o‑mini）评估。

**📊 数据集**

使用 272,881 条 SFT 训练样本（过滤版 Vision‑R1‑cold）和 8,080 条带边框的 saliency‑r1‑8k 数据集；在 MMU‑Pro、MMBench、POPE、MME、MME‑RealWorld、MMStar、ChartQA、IllusionVQA、ScienceQA、SalBench 等 10 个公开 VQA 基准上进行评测。

**📈 对比分析**

与基线 VLM、仅 SFT、Vision‑R1、以及多种开源与闭源 SOTA 模型（如 GPT‑4o、Claude‑3.5、Gemini‑1.5、InternVL‑2、LLaVA‑CoT 等）比较。Saliency‑R1 在 10 个基准上取得 SOTA 或接近商业模型的表现，并在解释性指标（PG、energy‑PG）上提升约 10–14%，在 faithfulness（插入/删除测试）上与基线持平或更优。

**⚠️ 局限性**

局限性：训练规模受限，SFT 数据集相对较小；对齐奖励仅使用粗粒度 bounding‑box，未利用更精细的分割标签；实验仅在 3B/7B 规模模型上验证，未验证更大模型的可扩展性。

---

## 675. Empirical Characterization of Rationale Stability Under Controlled Perturbations for Explainable Pattern Recognition

**arXiv ID:** 2604.04456 | [PDF](https://arxiv.org/pdf/2604.04456v1)

**作者:** Abu Noman Md Sakib `[一作]` (University of Texas at San Antonio), Zijie Zhang `[通讯]` (University of Texas at San Antonio)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于余弦相似度的解释一致性度量 ESS，评估同一类别输入的 SHAP 解释在标签保持不变的扰动下的稳定性。

**💡 创新点**

创新点在于量化解释在标签保持不变的扰动下的稳定性，并将其与传统的可信度指标（FID‑k）进行对比，提供了更细粒度的解释一致性评估。

**🔧 技术方法**

使用 SHAP 对模型输出进行特征归因，计算同标签样本对之间的余弦相似度得到 ESS，并计算遮蔽前后置信度下降的 FID‑k 来评估解释的可信度。

**📊 数据集**

在 SST‑2 与 IMDb 情感分析数据集上进行实验，使用 BERT、RoBERTa 与 DistilBERT 三种 Transformer 作为基准模型。

**📈 对比分析**

与标准的 FID‑k（k=5）及统计显著性检验比较，ESS 能更敏感地捕捉解释不一致；BERT 的 ESS 在 SST‑2 上约为 0.11，RoBERTa 接近 0.20，DistilBERT 仅为 0.01，且在同义词置换扰动后均下降 10%–15%，说明 ESS 对解释变化更为敏感。

**⚠️ 局限性**

局限包括：ESS 对 SHAP 的噪声敏感；只针对文本模型验证，无法直接推广到视觉或其它任务；对扰动方式的定义有限，未考虑更复杂的结构化扰动；在小模型（如 DistilBERT）上解释稳定性本身较低，导致 ESS 反映效果有限。

---

## 676. Characterization of FR3 Cellular Vehicle-to-Base Station Links in HighRise Urban Scenarios

**arXiv ID:** 2604.03992 | [PDF](https://arxiv.org/pdf/2604.03992v1)

**作者:** Fahimeh Aghaei `[一作]` (University of Oulu), Murat Uysal `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 18923 | [OpenAlex ID](https://openalex.org/A5018973008)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文利用射线追踪（Ray‑Tracing）方法对高层建筑城市环境下的 FR3（8.2 GHz、15 GHz）与 sub‑6 GHz（4.6 GHz）和 mmWave（28 GHz）频段的车载对基站（C‑V2B）下行链路进行系统化通道特征分析与性能评估。

**💡 创新点**

创新点在于：①将 FR3 与 FR1/FR2 频段在相同天线口径下进行公平比较，揭示 FR3 在干扰环境中可优于 mmWave 的优势；②在 6G 高楼城市场景下，结合统计 ITU 模型与真实 3D CAD（迪拜市区）两种模型，对不同频段的 SNR/SINR、覆盖概率进行统一仿真；③通过系统性研究基站密度对覆盖概率的非单调影响，提供频段与部署密度的最佳匹配建议。

**🔧 技术方法**

主要技术包括：射线追踪仿真（Remcom Wireless InSite）、MIMO 最大比传输/接收（MRT/MRC）波束成形、全干扰与无干扰两种场景下的 SINR 计算、基站密度与覆盖概率分析。

**📊 数据集**

数据集由两部分构成：①基于 ITU-R 统计参数（α₀=0.5, β₀=300, γ₀=50）生成的随机城市模型；②迪拜市区的 3D CAD 模型（通过 Blender‑OpenStreetMap 导入）。两种模型均覆盖 1.2 km×1.2 km 区域，部署 17 站点，随后通过多次城市部署（10 次）获取统计结果。

**📈 对比分析**

比较方法：在相同天线口径下，分别对四个频段（4.6、8.2、15、28 GHz）计算 SNR/SINR 的累积分布函数（CDF），并以 50 % CDF 以及 10 %/90 % CDF 为关键指标进行对比；覆盖概率则以 SINR>10 dB 为阈值，绘制不同基站密度下的覆盖概率曲线。结果显示：在干扰自由情况下 FR3 的 SNR 与 mmWave 相当甚至更好；在全干扰情况下，FR3 在低 CDF（≈10 %）时的 SINR 可与 4.6 GHz 和 28 GHz 的最优水平相近，尤其对细胞边缘用户友好；基站密度提升到 17 BS/km² 后，覆盖概率出现拐点，低频段进一步增密反而降低覆盖概率，而高频段受益于波束成形可持续提升。

**⚠️ 局限性**

局限性包括：①仿真仅考虑静态建筑阻塞，未包含车辆动态遮挡或时变多径效应；②仅评估单流 MIMO（MRT/MRC），未探讨多流或更复杂波束赋形策略；③仿真范围仅限 1.2 km²，未验证更大尺度网络的可扩展性；④缺乏真实测量验证，结果主要基于射线追踪模型。

---

## 677. DP-OPD: Differentially Private On-Policy Distillation for Language Models

**arXiv ID:** 2604.04461 | [PDF](https://arxiv.org/pdf/2604.04461v1)

**作者:** Fatemeh Khadem `[一作]` (Santa Clara University), Yuhong Liu `[通讯]` (Santa Clara University)

**通讯引用:** 11082 | [OpenAlex ID](https://openalex.org/A5100324598)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种无合成文本、基于不同ially private的on-policy distillation方法DP-OPD，用于在严格隐私预算下压缩大型语言模型并保持生成质量。

**💡 创新点**

创新点在于：①只在学生更新中使用DP-SGD，完全省略了对教师的DP训练和离线合成数据；②采用on-policy rollouts让教师在学生实际生成的轨迹上提供token级别的监督；③通过Generalized Knowledge Distillation (GKD) 灵活调节前向/后向KL或JSD式损失，实现更稳健的知识迁移。

**🔧 技术方法**

技术手段包括：DP-SGD（梯度裁剪+高斯噪声）、on-policy生成与教师回溯、温度缩放的softmax、GKD损失（参数β调节）、控制码编码的条件化输入。

**📊 数据集**

实验使用了Yelp短文本评论数据集（约190万训练样本）和BigPatent专利摘要数据集（约20万训练样本），两者均采用控制码预置标签。

**📈 对比分析**

与传统DP-SGD微调、off-policy DP KD（DPKD）以及基于合成文本的DistilDP对比，DP-OPD在ε=2.0的严格隐私预算下，Yelp的perplexity从DistilDP的44.15降到41.68，BigPatent从32.43降到30.63，显著优于其他基线。

**⚠️ 局限性**

局限性包括：需要实时生成on-policy rollouts，导致训练时长和教师推理成本提升；对教师的查询频率有要求；在tokenizer不匹配或教师不可用场景下效果未知；控制码本身若包含敏感信息需进一步隐私化。

---

## 678. TableVision: A Large-Scale Benchmark for Spatially Grounded Reasoning over Complex Hierarchical Tables

**arXiv ID:** 2604.03660 | [PDF](https://arxiv.org/pdf/2604.03660v1)

**作者:** Xiaoyu Chen `[一作]` (Hong Kong University of Science and Technology), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 45056 | [OpenAlex ID](https://openalex.org/A5101862104)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出TableVision数据集，并实现分阶段的Locate-then-Reason框架，显著提升复杂表格推理性能

**💡 创新点**

通过引入精确的像素级边界框与多步推理轨迹，分离感知与推理任务，解决“感知瓶颈”

**🔧 技术方法**

使用多模态大型语言模型Qwen3-VL-8B-Instruct+LoRA调优，结合渲染式确定性对齐流程与两阶段解耦训练

**📊 数据集**

TableVision（6,799条带有高精度边界框与推理轨迹的复杂层次表格），并与13类任务的基准数据集对照

**📈 对比分析**

与10个零样本多模态模型相比，解耦框架整体提升12.3%准确率；Oracle（使用真实框）提升至80.0%，验证空间定位的关键作用

**⚠️ 局限性**

局部定位阶段在极密集表格中易出现误检，导致后续推理误差累积，特别是L3分析任务的准确率受限

---

## 679. General Explicit Network (GEN): A novel deep learning architecture for solving partial differential equations

**arXiv ID:** 2604.03321 | [PDF](https://arxiv.org/pdf/2604.03321v1)

**作者:** Genwei Ma `[一作]` (National Center for Applied Mathematics Beijing), Xing Zhao `[通讯]` (School of Mathematical Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于显式基函数的通用显式网络（GEN）来求解偏微分方程，采用点对函数的形式将解表示为可调基函数的非线性组合；

**💡 创新点**

创新点在于将基函数设计与物理先验融合，构造点对函数的闭式表示，显著提升求解的鲁棒性、可扩展性和物理一致性；

**🔧 技术方法**

使用深度神经网络对基函数向量进行非线性组合，并利用物理信息损失（PINN框架）进行训练，基函数选取以三角函数和高斯函数为主；

**📊 数据集**

使用合成的 PDE 数据集，分别在热传导、波动方程和 Burgers 方程的训练区间内采样，构成训练、边界与初始条件；

**📈 对比分析**

与传统 PINN 和数值求解器对比，实验表明 GEN 在训练域内逼近精度与 PINN 相当，且在外推域展现更好的稳定性与精度；在热方程、波动方程和 Burgers 方程三例中，GEN 的误差普遍低于 PINN，并能通过合适的基函数数实现更高精度；

**⚠️ 局限性**

局限性包括：基函数的选择高度依赖先验知识，需人工调优；训练收敛速度慢（需10万次迭代）；基函数数量对精度与参数冗余敏感，缺乏自动化调优机制；

---

## 680. Pickalo: Leveraging 6D Pose Estimation for Low-Cost Industrial Bin Picking

**arXiv ID:** 2604.04690 | [PDF](https://arxiv.org/pdf/2604.04690v1)

**作者:** Alessandro Tarsi `[一作]` (Institut des Systèmes Intelligents et de Robotique), Ugo Pattacini `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了基于低成本RGB‑D摄像头和6D姿态估计的工业料箱抓取系统Pickalo，能够在高密度、强遮挡的工况下实现高效、低成本的料箱抓取。

**💡 创新点**

创新点在于将多视角深度增强（BridgeDepth）、零射击SAM‑6D姿态估计与时间姿态缓冲（Pose Buffer）结合，显著提升姿态鲁棒性；同时构建模块化管线，使系统在96–99%成功率下保持高速吞吐。

**🔧 技术方法**

使用眼手RGB‑D摄像机、BridgeDepth深度增强、Mask‑R‑CNN实例分割、SAM‑6D姿态估计、Pose Buffer融合、离线抗对称抓取候选、基于Utility的抓取排序与碰撞检查、UR5e+并行爪等技术。

**📊 数据集**

训练Mask‑R‑CNN采用BlenderProc生成的合成数据；评估使用真实工业对象（方形、圆柱、复杂形）以及XYZ‑IBD数据集进行姿态误差测评。

**📈 对比分析**

通过与原始RealSense深度、无Pose Buffer等基线对比，在30分钟连续抓取中平均每小时600次抓取，成功率96–99%，Early Exit Rate低至0.6%，在不同物体形状下保持高成功率。

**⚠️ 局限性**

仍受极端遮挡、极小接触面导致的抓取失败限制；对深度增强模型的推理时间有一定依赖；需要更细致的轨迹规划和人工设计抓取候选集以覆盖复杂几何。

---

## 681. Collapse-Free Prototype Readout Layer for Transformer Encoders

**arXiv ID:** 2604.03850 | [PDF](https://arxiv.org/pdf/2604.03850v1)

**作者:** Giansalvo Cirrincione `[一作]` (Laboratory LTI, Université de Picardie Jules Verne), Rahul Ranjeev Kumar `[通讯]` (Institute of Energy and Resources, Charles Darwin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种新的原型基础竞争读取层DDCL-Attention，旨在替代传统的池化启发式方法，通过保持一个小型的全局学习原型向量库来实现信息的压缩和表示质量的反馈。

**💡 创新点**

DDCL-Attention的创新点在于提供了原型不崩溃的数学保证，确保原型不会收敛到同一点，并且在训练过程中保持稳定性，同时能够在不同的应用场景中灵活使用。

**🔧 技术方法**

使用了竞争学习、深度聚类和原型学习等技术，结合了稳定性分析和变换器架构的应用。

**📊 数据集**

实验使用了四个数据集，包括SST-2、IMDB、20 Newsgroups和CIFAR-10，以及一个关于轨道碎片分类的科学表格数据集。

**📈 对比分析**

与现有的原型基础机制（如Slot Attention和Perceiver）相比，DDCL-Attention在原型分离和训练稳定性方面表现更好，且在多个实验中实现了100%的代码本利用率，而标准的硬向量量化仅为39%。

**⚠️ 局限性**

限制在于稳定性结果的条件性，DDCL-Attention无法直接建模序列级别的依赖关系，且原型库的大小和维度需要手动选择，没有自动选择规则。

---

## 682. UENR-600K: A Large-Scale Physically Grounded Dataset for Nighttime Video Deraining

**arXiv ID:** 2604.04402 | [PDF](https://arxiv.org/pdf/2604.04402v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 683. ActionNex: A Virtual Outage Manager for Cloud

**arXiv ID:** 2604.03512 | [PDF](https://arxiv.org/pdf/2604.03512v1)

**作者:** Zhenfeng Lin `[一作]` (Microsoft), Angie Anderson `[通讯]` (Azure CTO Office)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并部署了 ActionNex，一套面向大规模云平台的全流程故障管理代理系统，能够实时分析多模态运营信号，提取关键事件，结合分层记忆与 LLM 推理，为运维团队提供下一步最佳操作建议。

**💡 创新点**

创新点包括：① 将多源高维信号压缩为可解释的关键事件；② 采用分层记忆（长期 KCA、情节案例、工作上下文）实现持续自我演化；③ 将执行动作作为隐式反馈，推动在线学习；④ 在生产环境中实现端到端的自动化推荐与人机协作。

**🔧 技术方法**

技术方法主要是：使用 GPT‑5.2 进行文本抽取、记忆检索与多步推理；文本嵌入（text‑embedding‑3‑large）做 KCA 与案例检索；关键事件抽象模块对多模态输入进行归一化、实体识别与跨模态聚合；以及基于 LLM 的 Prompt 迭代推理机制。

**📊 数据集**

数据集为微软 Azure 真实故障记录，共约 8M token，包含运营日志、会议记录、聊天记录等，提炼出 4,374 个关键事件、361 条模型推荐动作以及 447 条人工标注的真实动作，用于训练与评估。

**📈 对比分析**

评估使用两套真实动作集 G1（原始提取）和 G2（筛选后的 playbook 支持），在三次留存集上进行测试，取得 71.4% 的精度、52.8–54.8% 的召回；随故障阶段推进，召回逐步提升但精度略有下降，显示系统对早期不确定信息的敏感度与后期决策的准确性。

**⚠️ 局限性**

局限性主要包括：① 早期阶段上下文不足导致召回率低；② 对未标记或间接表达的动作可能误判；③ 需要更严格的安全约束与在线学习稳定性保障；④ 目前仅在 Azure 环境验证，跨云/跨业务场景的迁移需进一步研究。

---

## 684. Governance-Constrained Agentic AI: Blockchain-Enforced Human Oversight for Safety-Critical Wildfire Monitoring

**arXiv ID:** 2604.04265 | [PDF](https://arxiv.org/pdf/2604.04265v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 685. VA-FastNavi-MARL: Real-Time Robot Control with Multimedia-Driven Meta-Reinforcement Learning

**arXiv ID:** 2604.03998 | [PDF](https://arxiv.org/pdf/2604.03998v1)

**作者:** Yang Zhang `[一作]` (University of Missouri), Hong Wang `[通讯]` (Hubei University)

**通讯引用:** 23946 | [OpenAlex ID](https://openalex.org/A5100370931)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 VA-FastNavi-MARL 框架，利用异步多模态指令（语音、视觉、机器码）统一嵌入并通过元强化学习实现三臂机器人在动态环境中的实时控制与快速适应。

**💡 创新点**

①将异步多模态指令映射到共享潜在空间，实现跨模态一致性；②引入动态指令队列（FIFO）以实时处理不规则指令流；③将任务视为指令分布，采用 MAML+SAC 的元强化学习实现一次性梯度更新即可适应新指令；④实现零推理开销的实时响应。

**🔧 技术方法**

并行编码器‑解码器网络、Soft Actor‑Critic、MAML、FIFO 指令队列、参数共享策略、快速梯度适配、深度学习与多模态融合。

**📊 数据集**

在自建的三臂协作仿真环境中生成的音频、视觉与机器码指令集；指令分布覆盖不同难度（Easy/Medium/Hard）并加入噪声，未使用公开标准数据集。

**📈 对比分析**

与 MAAC、MAPPO、MADDPG 等 MARL 基线在 50 次适应步骤内对比。VA‑FastNavi‑MARL 在平均奖励约 1700、成功率 1.0、碰撞率零的情况下收敛最快；长期连续适应测试中保持高安全性与指令完成率。

**⚠️ 局限性**

仅在仿真中验证，缺乏真实机器人实验；对极端噪声与通信延迟的鲁棒性仍待评估；多模态融合导致生成时间在噪声下略增；模型规模与参数共享在更大规模队列中的可扩展性待进一步验证。

---

## 686. SODA: Semi On-Policy Black-Box Distillation for Large Language Models

**arXiv ID:** 2604.03873 | [PDF](https://arxiv.org/pdf/2604.03873v1)

**作者:** Xiwen Chen `[一作]` (Clemson University), Feng Luo `[通讯]` (Clemson University)

**通讯引用:** 13767 | [OpenAlex ID](https://openalex.org/A5100683466)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种半在线对抗式知识蒸馏框架（Semi On‑policy Distillation with Alignment，简称SODA），利用教师与基准学生一次性生成的响应对构建偏好数据，直接通过直接偏好优化（DPO）进行蒸馏；

**💡 创新点**

创新点在于：①通过一次性捕获基准学生的零样本输出，构造学生特定的负样本，避免连续对抗训练；②把对抗蒸馏的双重学习信号（教师模仿 + 低质量输出剔除）压缩到无监督偏好优化框架；③实现与全在线对抗蒸馏（GAD）同级甚至更优的性能，同时显著提升训练速度与显存利用率；

**🔧 技术方法**

核心技术包括：基准学生一次性生成响应；偏好数据集构造；在该数据集上应用Direct Preference Optimization（DPO）；轻量级热身阶段；对比实验与基准；

**📊 数据集**

使用GPT‑5‑Chat作为教师；在Qwen2.5‑3B/7B/14B‑Instruct与Llama‑3.2‑3B‑Instruct、Llama‑3.1‑8B‑Instruct等四类小模型上进行蒸馏；数据集为LMSYS‑Chat‑1M‑Clean；

**📈 对比分析**

与SeqKD（仅监督学习）和GAD（完整对抗蒸馏）进行对比，实验显示SODA在15/16种模型‑数据集组合上优于GAD，平均提升约0.9点（最高+2.1点），在Llama‑3系列上提升超过1点；训练时间缩短约10倍，峰值显存降低27%；

**⚠️ 局限性**

局限性包括：①需要一次性生成基准学生响应，若基准模型在训练初期表现异常可能导致负样本质量不足；②只在一次性快照上构造负样本，无法随训练动态捕获新出现的错误；③验证范围仅在GPT‑5教师与特定小模型，未知在更大教师或不同任务上的适用性；

---

## 687. Provable Multi-Task Reinforcement Learning: A Representation Learning Framework with Low Rank Rewards

**arXiv ID:** 2604.03891 | [PDF](https://arxiv.org/pdf/2604.03891v1)

**作者:** Yaoze Guo `[一作]`, Shana Moothedath `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文提出了一种面向多任务强化学习的共享低秩奖励表示学习框架，结合奖励无关强化学习和低秩矩阵估计，实现了对多任务奖励矩阵的联合恢复并生成ϵ-最优策略。

**💡 创新点**

创新点在于：① 在不依赖高斯特征或稀疏性、无噪声奖励等传统假设的情况下，证明低秩奖励矩阵可在RL环境中成功恢复；② 通过奖励无关阶段先学习探索策略，显著提升样本信息量；③ 给出了样本复杂度与表示误差的理论关系与累计回报上界。

**🔧 技术方法**

主要技术包括奖励无关强化学习、线性MDP特征嵌入、基于核方法的探索策略设计、低秩矩阵恢复（SVD、MoM对比）以及子空间距离与误差传播分析。

**📊 数据集**

实验使用了两类数据集：① 随机生成的高维线性多任务控制环境（d=100，T=100，r=2）；② 5×5 网格迷宫导航环境（d=25×4=100，T=5），并在各环境中模拟多任务奖励。

**📈 对比分析**

与三种基线（随机探索、MoM估计、独立任务TS）比较，实验结果表明所提方法在子空间距离、奖励参数误差以及累计回报方面均显著优于基线，尤其在随机探索基线下表现出极大提升。

**⚠️ 局限性**

局限性包括：① 需要先验的线性特征嵌入和低秩奖励结构；② 样本复杂度仍受维度d和秩r的影响，实际环境中可能更高；③ 目前仅在小规模任务数和离散状态/动作空间上验证，尚未扩展到连续或深度强化学习框架。

---

## 688. PolySwarm: A Multi-Agent Large Language Model Framework for Prediction Market Trading and Latency Arbitrage

**arXiv ID:** 2604.03888 | [PDF](https://arxiv.org/pdf/2604.03888v1)

**作者:** Rajat M. Barot `[一作]` (State University of New York), Arjun S. Borkhatariya `[通讯]` (Arizona State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计实现了PolySwarm多代理LLM框架，用于实时预测市场交易与延迟套利。

**💡 创新点**

创新点是引入50种多样化LLM人格、置信度加权贝叶斯聚合、KL/JS信息理论异常检测以及CEX‑DEX延迟套利模块。

**🔧 技术方法**

采用大型语言模型（GPT‑4/Claude等）、多代理协同、贝叶斯聚合、KL/JS散度、Kelly位置管理、异步Python FastAPI、SQLite缓存与WebSocket仪表盘。

**📊 数据集**

主要使用Polymarket历史交易数据、实时行情、新闻与宏观经济信息进行评估。

**📈 对比分析**

通过Brier分数、对数损失、校准图与与人类超预测者基准对比，聚合后概率校准显著优于单模型基准。

**⚠️ 局限性**

限制包括幻觉与误报、计算成本与延迟、市场影响与反馈循环、监管与伦理风险。

---

## 689. A Novel Hybrid PID-LQR Controller for Sit-To-Stand Assistance Using a CAD-Integrated Simscape Multibody Lower Limb Exoskeleton

**arXiv ID:** 2604.03766 | [PDF](https://arxiv.org/pdf/2604.03766v1)

**作者:** Ranjeet Kumbhar `[一作]` (Thapar Institute of Engineering and Technology), Irfan Hussain `[通讯]` (Khalifa University)

**通讯引用:** 2884 | [OpenAlex ID](https://openalex.org/A5023802518)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了双侧下肢外骨骼坐到站立过程中PID、LQR和混合PID‑LQR三种控制器的设计与仿真比较。

**💡 创新点**

提出并验证了一种融合比例-积分-微分与线性二次调节的混合控制器，并通过系统性调参确定最佳融合系数α=0.65，实现了轨迹跟踪精度与稳态误差的最优平衡。

**🔧 技术方法**

使用SolidWorks CAD模型通过Simscape Multibody导入的高保真动力学模型、OpenSim逆运动学生成的参考轨迹以及Simulink实现的控制器框架。

**📊 数据集**

参考数据集来自OpenSim中位数个体（75 kg，1.75 m）的3 s坐到站立运动，三阶段生物力学分段（弯曲-动量传递-伸展）。

**📈 对比分析**

通过RMSE、MAE、超调、上升时间、稳定时间等多指标进行比较，混合控制器相较于PID降低72.3% RMSE、比LQR降低52.2% RMSE，稳定时间比PID缩短90%以上，整体性能显著优于两基线。

**⚠️ 局限性**

仅在理想扭矩仿真环境下验证，未考虑硬件噪声、致动器动力学及临床试验，缺乏实机与真实用户验证。

---

## 690. Resource-Conscious Modeling for Next- Day Discharge Prediction Using Clinical Notes

**arXiv ID:** 2604.03498 | [PDF](https://arxiv.org/pdf/2604.03498v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 691. Statistical Model Checking of the Island Model: An Established Economic Agent-Based Model of Endogenous Growth

**arXiv ID:** 2604.04543 | [PDF](https://arxiv.org/pdf/2604.04543v1)

**作者:** Stefano Blando `[一作]` (Sant'Anna School of Advanced Studies), Ernest Ivanaj `[通讯]` (University of Geneva)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对 Island Model 进行统计模型检查与多参数敏感性分析，自动确定仿真次数并给出置信区间；

**💡 创新点**

首次将 MultiVeStA 统计模型检查方法与 agent‑based 经济模型相结合，实现自动收敛判定与正式假设检验，为经济 ABM 的定量分析提供可复制、可证伪的框架；

**🔧 技术方法**

使用 MultiVeStA 与 MultiQuaTEx 查询语言、Welch t‑检验、MATLAB 编写的模型接口，实现黑箱集成与参数扫描；

**📊 数据集**

主要使用 Island Model 的默认参数与内部随机种子，无外部实测数据集；

**📈 对比分析**

采用 Welch t‑检验逐时刻比较不同参数配置，6/7 组对比显著，置信区间宽度可控，自动确定所需仿真次数，性能优于传统固定 Monte Carlo；

**⚠️ 局限性**

局限在于仅进行单维参数扫描、未覆盖高维参数空间、模型简化程度高、未与真实宏观数据进行校准或验证。

---

## 692. A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens

**arXiv ID:** 2604.04913 | [PDF](https://arxiv.org/pdf/2604.04913v1)

**作者:** Tommie Kerssies `[一作]` (Amazon), Liang-Chieh Chen `[通讯]` (Amazon)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种名为 DeltaTok 的视频分词器和 DeltaWorld 生成式世界模型，通过将连续帧的 VFM 特征差异压缩为单一 delta token，实现在单前向传播中生成多样化的未来预测。

**💡 创新点**

创新点在于：① 用单一 delta token 替代传统的空间特征，压缩为一维时间序列，显著减少计算量；② 将 Best‑of‑Many（BoM）训练与 delta 编码结合，能够在一次前向传播中得到多样化样本；③ 与传统生成式世界模型相比，参数量减少 35 倍，FLOPs 减少 2000 倍。

**🔧 技术方法**

所使用的技术包括：基于 DINOv3 视觉基础模型的特征空间、Transformer 编码器-解码器的连续自编码器、Best‑of‑Many 训练策略以及 delta token 的设计与解码。

**📊 数据集**

实验使用了 VSPW、Cityscapes、KITTI 三个密集预测基准数据集，涵盖语义分割和深度估计任务。

**📈 对比分析**

在短期（≈0.2 s）和中期（≈0.6 s）预测任务中，DeltaWorld 在最佳样本上优于 Cosmos-4B/12B，平均分数与判别式 DINO‑World 接近或略优，同时参数量 35× 更少、FLOPs 2000× 更低，证明了其在效率与性能上的优势。

**⚠️ 局限性**

局限性包括：单 token 代表能力有限，可能对细粒度空间细节捕捉不足；依赖 VFM 特征，若 VFM 迁移或更新需重新训练；以及在极端变化或长时间跨度预测时仍可能出现误差。

---

## 693. Latent Profiles of AI Risk Perception and Their Differential Association with Community Driving Safety Concerns: A Person-Centered Analysis

**arXiv ID:** 2604.04849 | [PDF](https://arxiv.org/pdf/2604.04849v1)

**作者:** Amir Rafe `[一作]` (Texas State University), Subasish Das `[通讯]` (Texas State University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过对美国全国代表性样本的 AI 风险认知问卷进行人群聚类，识别出四类 AI 风险配置，并检验其与驾驶安全关注度的差异。

**💡 创新点**

首次将人群中心化方法应用于 AI 风险认知，形成 AI 风险分层模型；首次证明 AI 风险配置与交通安全认知共变；首次在交通安全研究中使用 BCH 校正的远端结果分析，融合文化理论、心理测量学和受众分割理论。

**🔧 技术方法**

使用加权潜在类别分析（LCA）寻找最优类数；采用 Bolck–Croon–Hagenaars（BCH）方法校正分类误差后对远端变量进行比较；随后运用加权多项式逻辑回归评估人口与意识形态预测因素；进行四项稳健性检验。

**📊 数据集**

Pew Research Center American Trends Panel 第 152 波（2024 年）全国概率抽样调查，样本量 5,255 人。

**📈 对比分析**

通过 BIC、entropy、BLRT、VLMR-LRT 等指标对 1–7 类模型进行比较，最终确定四类模型；BCH 校正后发现所有九个驾驶安全指标在四类间显著差异，且类间差异呈单调递增；模型分类精度高，稳健性检验均通过。

**⚠️ 局限性**

研究为横断面设计，因果关系无法确定；指标间存在 26 对局部独立性残差；人口预测变量仅解释 2.5% 类别差异；仅限美国成人，缺乏跨国和纵向验证；指标为通用测量，未针对 LCA 进行优化。

---

## 694. CT-VoxelMap: Efficient Continuous-Time LiDAR-Inertial Odometry with Probabilistic Adaptive Voxel Mapping

**arXiv ID:** 2604.03747 | [PDF](https://arxiv.org/pdf/2604.03747v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 695. Towards a theory of morphology-driven marking in the lexicon: The case of the state

**arXiv ID:** 2604.03422 | [PDF](https://arxiv.org/pdf/2604.03422v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 696. RESCORE: LLM-Driven Simulation Recovery in Control Systems Research Papers

**arXiv ID:** 2604.04324 | [PDF](https://arxiv.org/pdf/2604.04324v1)

**作者:** Vineet Bhat `[一作]` (New York University), Farshad Khorrami `[通讯]` (New York University)

**通讯引用:** 5664 | [OpenAlex ID](https://openalex.org/A5082413942)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 RESCORE 框架，通过闭环的分析、编码和验证 LLM 代理自动从控制系统论文中重建可执行的仿真代码；

**💡 创新点**

创新点在于将视觉反馈与迭代执行相结合，形成可持续改进的 LLM 代理链，并构建了 500 篇 IEEE CDC 论文的重建基准；

**🔧 技术方法**

使用 GPT‑5.2 与 Qwen‑3 等大型语言模型，结合图像识别与数学 OCR 技术，实现多轮生成-执行-验证的闭环流程；

**📊 数据集**

基准数据集为 500 篇 IEEE Conference on Decision and Control（CDC）2021‑2025 年论文，其中 221 篇被标记为可重建，194 篇用于最终评估；

**📈 对比分析**

与单次生成基线相比，RESCORE 在 FRS（图形重建得分）上从 1.73 提升至 2.17（约 25% 相对提升），成功率提升至 40.7%，并实现约 10 倍的人工重建速度加速；

**⚠️ 局限性**

局限性包括对完整参数与代码细节的高度依赖，无法很好处理高级优化（SDP、LMI）和 PDE 离散化问题，以及对复杂通信拓扑和分布式控制的替代模型。

---

## 697. Towards Unveiling Vulnerabilities of Large Reasoning Models in Machine Unlearning

**arXiv ID:** 2604.04255 | [PDF](https://arxiv.org/pdf/2604.04255v1)

**作者:** Aobo Chen `[一作]` (Iowa State University), Mengdi Huai `[通讯]` (Iowa State University)

**通讯引用:** 1012 | [OpenAlex ID](https://openalex.org/A5016035883)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了针对大型推理模型（LRM）的恶意机器学习忘却攻击，能够在忘却过程中诱导模型给出错误答案并生成具有说服力的误导性推理轨迹。

**💡 创新点**

创新点在于设计了双重目标的生物层次精确忘却攻击框架，利用可微分代理目标、影响力词汇引导和松弛指示器，解决了非可微约束、长推理链梯度衰减和离散忘却样本选择等难题。

**🔧 技术方法**

主要技术包括：可微分的攻击目标函数、正负影响词汇引导损失、梯度匹配与整数规划求解、以及基于梯度上升、KL 最小化和 RMU 等现有忘却算法的无缝集成。

**📊 数据集**

实验使用了三种 LRM（OpenReasoning‑Nemotron‑1.5B、Skywork‑o1‑Open‑Llama‑3.1‑8B、Skywork‑OR1‑7B）以及两种 LLM（Llama‑2‑7B、Mistral‑7B‑v0.1），数据集涵盖 SST‑2、Twitter、Star‑1、TOFU 等。

**📈 对比分析**

与随机基线相比，实验显示该攻击在白盒和黑盒两种设置下都能显著提升攻击成功率（ASR 约为随机基线的 2–3 倍）并在不同忘却比例下保持一致的高性能，且在黑盒迁移实验中也表现出强大的跨模型可迁移性。

**⚠️ 局限性**

局限性包括：目前仅验证了对 LRMs 的攻击效果，尚未系统评估对更大规模或不同结构模型的适用性；攻击实现依赖于对训练数据或代理模型的访问；在某些情况下，仅针对推理质量下降的攻击仍需进一步精细化。

---

## 698. Digital Privacy in IoT: Exploring Challenges, Approaches and Open Issues

**arXiv ID:** 2604.04572 | [PDF](https://arxiv.org/pdf/2604.04572v1)

**作者:** Shini Girija `[一作]` (Birla Institute of Technology and Science), Mithun Mukherjee `[通讯]` (Birla Institute of Technology and Science)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了物联网（IoT）中的数字隐私问题，提出了基于IEEE DPM的风险驱动分类法，并构建了 AURA‑IoT 这一多层面、AI与隐私增强技术（PET）融合的隐私保障框架。

**💡 创新点**

创新点在于：①首次将身份、行为、推理、数据操纵、监管五类风险映射到 IoT 体系结构与安全生命周期；②设计了 AURA‑IoT 综合治理模型，集成了动态同意、可解释性、对抗鲁棒性、透明度与公平性等 AI 合规与隐私保护要素；③系统性梳理并对齐现有加密、差分隐私、联邦学习、区块链等 PET 的适配场景。

**🔧 技术方法**

主要技术包括：加密（对称/非对称、轻量级、同态）、差分隐私、联邦学习、强化学习、区块链、动态同意机制、AI 可解释性框架、对抗训练、透明度与公平性评估方法。

**📊 数据集**

无数据集；文章为综述与框架设计，未进行实验验证。

**📈 对比分析**

作者通过表格对比多种现有隐私解决方案的技术特点、应用场景与安全三元组，未给出定量性能指标或实验结果；相较于现有研究，强调框架的系统性与跨技术协同，但缺乏实际测评。

**⚠️ 局限性**

局限性：①缺乏真实环境中的案例验证与性能评估；②对新兴 IoT 威胁（如量子攻击、边缘 AI 的安全缺口）关注不足；③资源受限设备的轻量化实现仍待深入；④不同监管环境下的适配与互操作性未详细探讨。

---

## 699. Communication-Efficient Collaborative LLM Inference over LEO Satellite Networks

**arXiv ID:** 2604.04654 | [PDF](https://arxiv.org/pdf/2604.04654v1)

**作者:** Songge Zhang `[一作]` (Peking University), Shen `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了在LEO卫星网络上通过模型切分、流水线并行和激活压缩实现大语言模型协同推理的方案

**💡 创新点**

创新点包括：1）基于模型切分的多卫星协同推理框架；2）学习型Gumbel‑mask稀疏压缩+量化+熵编码的激活压缩；3）将切分与压缩联合优化映射为有向无环图的最短路径问题，利用改进的A*搜索实现全局最优

**🔧 技术方法**

使用技术包括模型切分、流水线并行、Gumbel‑mask稀疏化、量化、熵编码、DAG建模、A*搜索、网格搜索/优化器求解压缩比例

**📊 数据集**

实验采用EuroSAT和RESISC45两个遥感分类数据集，使用Vision Transformer (ViT-B/L/H/G) 模型进行推理

**📈 对比分析**

与Ground‑only、Single‑satellite等基线相比，该方案在推理延迟上减少约42–71%，通信开销降低约71%，并保持误差小于1%的准确率

**⚠️ 局限性**

限制：需要预先训练压缩模块；压缩比例离散化导致搜索复杂度高；大模型推理仍受卫星算力限制，方案对非常大规模模型的加速效果有限

---

## 700. Relative Density Ratio Optimization for Stable and Statistically Consistent Model Alignment

**arXiv ID:** 2604.04410 | [PDF](https://arxiv.org/pdf/2604.04410v1)

**作者:** Hiroshi Takahashi `[一作]` (NTT, Inc.), Kazutoshi Shinoda `[通讯]` (NTT, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为相对密度比优化（RDRO）的语言模型对齐方法，利用相对密度比进行稳定、统计一致的训练。

**💡 创新点**

核心创新是用相对密度比代替传统密度比，避免了密度比发散导致的训练不稳定，并给出更紧的收敛保证。

**🔧 技术方法**

采用 Bregman 散度优化、对数回归、梯度下降（AdamW）等技术实现相对密度比的估计与优化。

**📊 数据集**

在 UF-G（UltraFeedback‑Helpfulness）和 MIX‑14K 两个偏好数据集上，以 Qwen‑2.5 与 Llama‑3 预训练模型为基准进行实验。

**📈 对比分析**

与 KTO 与 DDRO 进行对比，使用 AlpacaEval LC Win Rate 评估；在 UF‑G 上 RDRO 的表现往往优于或与 KTO、DDRO 相当，尤其在 Llama‑3B/8B 上显著提升。

**⚠️ 局限性**

在 MIX‑14K 上效果相近，且对超参数 α 的敏感性有限；假设两类数据共享相同提示分布，并未在更大规模或多样化数据上进一步验证。

---

## 701. Perils of Parallelism: Transaction Fee Mechanisms under Execution Uncertainty

**arXiv ID:** 2604.04193 | [PDF](https://arxiv.org/pdf/2604.04193v1)

**作者:** Sarisht Wadhwa `[一作]` (Duke University), Kartik Nayak `[通讯]` (Duke University)

**通讯引用:** 2503 | [OpenAlex ID](https://openalex.org/A5040861793)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

研究区块链并行执行环境下的交易手续费机制，揭示并解决了并行性与执行不确定性带来的安全与公平挑战

**💡 创新点**

首次将并行性与执行不确定性框架化为不可避免的权衡，提出并证明了相应的不可能定理与最优边界，并设计了对应的手续费方案

**🔧 技术方法**

形式化模型、博弈论与不可能证明、对策设计与复杂性分析

**📊 数据集**

无实验数据，研究基于理论分析与模型证明

**📈 对比分析**

通过理论比较，证明所提出的手续费机制在并行性与不确定性权衡下达到最佳平衡，优于现有工业与学术方案

**⚠️ 局限性**

模型假设理想化，未考虑网络延迟与真实交易量分布，未来需在实验环境中进一步验证

---

## 702. Expanders Meet Reed--Muller: Easy Instances of Noisy k-XOR

**arXiv ID:** 2604.04188 | [PDF](https://arxiv.org/pdf/2604.04188v1)

**作者:** Jarosław Błasiok `[一作]` (Bocconi University), Madhu Sudan `[通讯]` (Harvard University)

**通讯引用:** 19623 | [OpenAlex ID](https://openalex.org/A5101519422)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构造了具有近最优扩张性质的显式二分图，证明在这些图上噪声 k‑XOR 问题可在多项式时间内解决，从而否定了“扩张性即硬度”的直觉。

**💡 创新点**

创新点在于将 Guruswami‑Umans‑Vadhan 的无损扩张器与 Reed‑Muller 码的随机错误纠错技术巧妙结合，形成可解码的余子图结构，并在不同噪声率下提供可扩展的显式构造。

**🔧 技术方法**

使用的核心技术包括：
- Guruswami‑Umans‑Vadhan (GUV) 失真无损扩张器的显式构造；
- 将图的邻接矩阵解释为 Reed‑Muller 码的余子图，从而把噪声 k‑XOR 的判别问题转化为 Reed‑Muller 码的随机错误纠错；
- 2010 年代关于 Reed‑Muller 码在随机错误（或随机擦除）下能超过最小距离的纠错算法。

**📊 数据集**

该工作为理论构造，无需外部数据集；所有结果均基于显式图构造与编码理论的分析。

**📈 对比分析**

相较于传统仅靠图扩张证明的下界，本文展示了在相同扩张水平下存在多项式时间算法。具体表现为：
- 对于常数噪声率 η=1/3，变量数 N=2^{Θ(log^2 N)}，约束数 M≈N^{O(1)}，算法时间为多项式；
- 在更严格的参数（M 为多项式级别）下，若 Reed‑Muller 码在 BEC 上达到容量，噪声率可退化到 η= N^{-c}，仍保持多项式时间。

**⚠️ 局限性**

局限性包括：
- 对于多项式约束数的强结果依赖于 Reed‑Muller 码在 BEC 上高效实现容量的未证实假设；
- 构造主要适用于二元域和特定的 k‑左正则图；
- 对其他域或更一般的噪声模型的推广尚未证明；
- 仅否定了“扩张即硬度”的猜想，未给出完整的可解性阈值或对所有图的完整描述。

---

## 703. Optimization-Free Constrained Control with Guaranteed Recursive Feasibility: A CBF-Based Reference Governor Approach

**arXiv ID:** 2604.04001 | [PDF](https://arxiv.org/pdf/2604.04001v1)

**作者:** Satoshi Nakano `[一作]` (Nagoya Institute of Technology), Gennaro Notomista `[通讯]` (University of Waterloo)

**通讯引用:** 1079 | [OpenAlex ID](https://openalex.org/A5072586461)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出一种将Explicit Reference Governor（ERG）与Control Barrier Function（CBF）相结合的安全控制框架，通过将参考更新视为虚拟控制输入，并利用动态安全裕度（DSM）与稳态可接受性通过softmin聚合，得到闭式参考更新法则，从而在不进行在线优化的情况下保证递归可行性并实现对多约束的安全执行。

**💡 创新点**

创新点主要有：①将DSM与稳态约束统一成单一光滑Barrier函数；②将参考更新抽象为虚拟控制，使得可行性问题变为CBF约束，能闭式求解；③结构化地保证递归可行性，避免传统CBF-QP的可行性失败；④提供理论收敛证明并给出闭式参考更新公式。

**🔧 技术方法**

使用的技术包括：Explicit Reference Governor、Control Barrier Function、Dynamic Safety Margin、软最小（softmin/Log‑sum‑exp）聚合、Lyapunov函数分析、闭式投影（QP求解的解析解）。

**📊 数据集**

仅在仿真环境中测试，使用n-DOF平面机械臂模型与静态圆形障碍物的碰撞数据，没有公开数据集。

**📈 对比分析**

通过仿真与传统ERG以及CBF‑QP方法比较，结果表明：①安全约束始终满足；②收敛速度与传统ERG相当；③由于不需要在线QP求解，计算负担显著降低，整体性能与传统方法持平或略优。

**⚠️ 局限性**

局限性包括：①对softmin参数β的选择敏感，过大可能导致数值不稳定；②对模型假设（预稳态控制器、Lyapunov函数）的依赖较强；③在高度非凸或复杂约束环境中softmin近似可能产生保守性，影响路径规划灵活性。

---

## 704. Correcting Source Mismatch in Flow Matching with Radial-Angular Transport

**arXiv ID:** 2604.04291 | [PDF](https://arxiv.org/pdf/2604.04291v1)

**作者:** Fouad Oubari `[一作]` (Université Paris-Saclay), Mathilde Mougeot `[通讯]` (Université Paris-Saclay)

**通讯引用:** 426 | [OpenAlex ID](https://openalex.org/A5046610403)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种新的 Flow Matching 框架——Radial–Angular Flow Matching (RAFM)，通过匹配数据的径向分布并在匹配半径的球面上进行角向几何插值，实现无高斯源偏差的生成。

**💡 创新点**

创新点在于：① 在源分布层面主动消除径向失配；② 将剩余的角向对齐问题转化为球面等速 geodesic 运动；③ 通过切向投影保证训练出的向量场保持半径不变，并给出径向保持与生成误差的稳定性界限。

**🔧 技术方法**

使用的技术包括：条件流匹配 (Conditional Flow Matching) 的训练目标、球面 geodesic 插值、径向分布估计（empirical CDF 与 Wasserstein 上的收敛保证）、切向投影正则化，以及基于 ODE 的推断。

**📊 数据集**

实验数据集：合成的高维 Student‑t（d=16、32）和实测的流体动力学 PIV（d=64、256）两组。

**📈 对比分析**

与基线（标准高斯 Flow Matching、仅源匹配、Multiplicative Score Generative Models MSGM）比较，RAFM 在径向 Wasserstein-1、KS 和 Sliced Wasserstein 上均优于标准 FM，并在大多数情形下接近或优于 MSGM，同时训练时间从分钟级降至秒级，显示出更高的效率与效果。

**⚠️ 局限性**

局限性包括：在低维接近原点的情况下球面插值可能失效；对极端高维数据的扩展尚未彻底验证；以及需要对径向分布进行准确估计，若训练数据不足可能影响性能。

---

## 705. Version Control System for Data with MatrixOne

**arXiv ID:** 2604.03927 | [PDF](https://arxiv.org/pdf/2604.03927v1)

**作者:** Hongshen Gou `[一作]` (MatrixOrigin), Peng Xu `[通讯]` (MatrixOrigin)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于 MatrixOne 的数据库实现的数据版本控制系统，支持 clone、tag/branch、diff、merge、revert 等 git 类操作，并在 TB 级数据上实现近乎即时的性能。

**💡 创新点**

创新点在于利用 MatrixOne 的不可变存储和 MVCC 快照机制，在数据库层直接实现分支、快照、diff 与 merge；diff/merge 通过扫描差异对象而非全表查询，显著提升速度；同时支持无主键表的冲突检测与三路合并，弥补传统 VCS 在大数据场景的不足。

**🔧 技术方法**

使用了 MatrixOne 的不可变对象存储、LSM 树、MVCC、WAL、订阅推送、后台压缩/垃圾回收等技术；clone 通过复制元数据目录实现；diff/merge 通过 Δ 集合扫描、聚合与冲突检测实现；同时实现了主键与无主键两种冲突识别逻辑。

**📊 数据集**

实验使用 TPC‑H 100GB 数据集中的 lineitem 表（约 6 亿行），在主键存在与不存在两种场景下进行评测。

**📈 对比分析**

通过与等价 SQL 实现对比评测，clone 仅需 0.2 秒 vs 114 秒；diff/merge 在 PK 下 0.2‑3 秒 vs 300‑400+ 秒；无 PK 情况下仍比 SQL 快 100‑500 倍；多用户协作实验中 builtin 仅几秒，SQL 约 400‑500 秒，展示了显著的性能优势。

**⚠️ 局限性**

限制包括：仅支持一次分支关系，缺乏完整的三路 diff；冲突分辨仅行级别，未细粒度；无主键表 diff/merge 需要更复杂聚合；二级索引不随 clone 复制；大对象可能导致内存占用；对 schema 变更的 diff/merge 受限；需手动解决冲突；未来改进方向包括更细粒度冲突处理、完整三路 diff、索引复制与大对象优化。

---

## 706. PollutionNet: A Vision Transformer Framework for Climatological Assessment of NO$_2$ and SO$_2$ Using Satellite-Ground Data Fusion

**arXiv ID:** 2604.03311 | [PDF](https://arxiv.org/pdf/2604.03311v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 707. SafeScreen: A Safety-First Screening Framework for Personalized Video Retrieval for Vulnerable Users

**arXiv ID:** 2604.03264 | [PDF](https://arxiv.org/pdf/2604.03264v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 708. Enabling Deterministic User-Level Interrupts in Real-Time Processors via Hardware Extension

**arXiv ID:** 2604.04015 | [PDF](https://arxiv.org/pdf/2604.04015v1)

**作者:** Hongbin Yang `[一作]` (Shandong University), Runyu Pan `[通讯]` (Shandong University)

**通讯引用:** 18131 | [OpenAlex ID](https://openalex.org/A5100692488)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种硬件扩展，能够在不经过内核调度的情况下，直接、确定性地将中断切换到用户级保护域，实现低延迟的用户级中断处理。

**💡 创新点**

创新点在于：①在中断到达时就完成保护域切换，避免了传统依赖内核的慢路径；②通过PMP、预算计数器、TCM以及注册银行等机制，实现空间和时间隔离；③将IID表实现为CAM，进一步降低查找延迟；④兼容现有微内核体系，减少软件改动。

**🔧 技术方法**

使用了RISC‑V指令集核心、物理内存保护(PMP)、紧耦合存储器(TCM)、寄存器银行、内容可寻址存储器(CAM)、预算计数器以及自定义CSR等硬件技术；在FPGA（Xilinx XC7K410T）和45 nm模拟工艺中实现和验证。

**📊 数据集**

实验采用合成的控制工作负载：脉冲训练输出(PTO)、Modbus‑RTU 通信、中断频率测试、图像识别等；未使用公开数据集，而是通过自定义的时间敏感和混合工作负载进行评估。

**📈 对比分析**

与传统内核调度（KERNEL）、Intel UIPI/ RISC‑V N扩展（INTEL）以及纯软件实现（SOFTWARE）比较；V1–V5四个变体分别在中断延迟、吞吐量、抖动和功耗上进行量化；结果显示最优变体V5的延迟仅11周期，比KERNEL低>50×，并在PTO、Modbus等任务中实现了10×以上的性能提升，抖动降低到1%以下。

**⚠️ 局限性**

局限性包括：①在最优配置下核心面积增加约19%（整体SoC约2%）；②需要在CPU核内增加额外的TCM端口，硬件接口复杂度上升；③目前仅在RISC‑V单核环境下验证，未讨论多核或MMU系统；④实现对现有软件栈的兼容性依赖于微内核，若使用其他操作系统需进一步适配。

---

## 709. Region-Based Constellation Designs for Constructive Interference Precoding in MU-MIMO

**arXiv ID:** 2604.03699 | [PDF](https://arxiv.org/pdf/2604.03699v1)

**作者:** Yupeng Zheng `[一作]`, Rahim Tafazolli `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在多用户多天线系统中利用构造性干扰预编码（CIP）时的星座设计，提出了基于区域（RBC）的星座模型，并设计了ME-QAM和RM-QAM两种新星座。

**💡 创新点**

创新点在于突破传统CI区域的凸性限制，引入非凸可行区域以提升符号的符号位对齐能力；通过RBC模型直接从消息到可行区域建模；并提出低复杂度的预测符号QP（PS-QP）算法。

**🔧 技术方法**

使用CIP理论、混合整数二次规划（MIQP）、线性求解、广义线性预编码、以及PS-QP的闭式符号预测。

**📊 数据集**

主要在模拟环境中评估：16×16、64×64、以及64×32的MU‑MIMO，信道为独立同分布Rayleigh衰落，采用IQ平衡的符号集合 {16, 64}。

**📈 对比分析**

与传统QAM‑CIP、PSK‑CIP和零迫预编码（ZF）比较，ME‑QAM和RM‑QAM在α²、SER和BLER上均实现了1–4 dB的SNR/误码率提升，PS‑QP与全搜索MIQP的性能基本一致，且复杂度与QAM‑CIP相当。

**⚠️ 局限性**

局限性包括：ME‑QAM在小星座尺寸下能量上限升高导致性能下降；对完全CSI的依赖；以及在极端低用户负载（例如64×32）下性能提升有限。

---

## 710. Regime Mapping of Oscillatory States in Balanced Spiking Networks with Multiple Time Scales

**arXiv ID:** 2604.04770 | [PDF](https://arxiv.org/pdf/2604.04770v1)

**作者:** Tsung-Han Kuo `[一作]` (National Taiwan University), Tzu-Chia Tung `[通讯]` (National Changhua University of Education)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过在平衡 LIF 网络中系统性变更突触后衰减时间 τ_s、传输延迟 d 与 STDP 学习速率 λ_p，构建了三维参数空间的振荡/非振荡状态图，并在代表性振荡点上进行 STDP 冻结与延迟抖动的局部控制实验，探讨多时间尺度如何共同决定网络的振荡行为。

**💡 创新点**

首次将突触后衰减、传输延迟和学习速率三大时间尺度的交互效应统一映射到振荡状态图中，并结合粗略的 Hopf 分岔边界提供理论参考，展示 STDP 关闭和延迟抖动对振荡强度的相反影响，从而为平衡网络的振荡调控提供了系统性且可视化的操作点。

**🔧 技术方法**

使用 Brian2 进行 LIF 网络仿真，结合线性化 Hopf 分岔近似作为理论边界，并通过功率谱峰值与中值比（PSD prominence）量化振荡强度，使用 Hedges’ g 计算实验条件间效应大小。

**📊 数据集**

采用模拟数据：160 个兴奋性、40 个抑制性 LIF 神经元，稀疏随机连接（p=0.1），多组随机种子重复实验；未使用真实生物实验数据集。

**📈 对比分析**

比较方法：将不同 λ_p、τ_s、d 组合下的状态分类（SIL、AI、OSC）与功率谱峰值与中值比进行对比；结果显示 λ_p 增大可将振荡区域扩展到更短 τ_s 和中至长 d；STDP 冻结将振荡强度降低约 20%，延迟抖动则提升约 50%，但均未显著改变平均发放率和主频。

**⚠️ 局限性**

局限性包括：仅考虑单一简化的 STDP 规则与固定网络规模；Hopf 边界为粗略近似，可能未能精确捕捉复杂动力学；参数范围有限，未覆盖更广阔的生理范围；控制实验仅在单一高振荡点展开，缺乏对整体空间的全局验证。

---

## 711. Plausibility as Commonsense Reasoning: Humans Succeed, Large Language Models Do not

**arXiv ID:** 2604.04825 | [PDF](https://arxiv.org/pdf/2604.04825v1)

**作者:** Sercan Karakaş `[一作]` `[通讯]` (University of Chicago), Sercan Karakaş (University of Chicago)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对土耳其语前置定语从句的附件歧义进行实验，检验世界知识的可预测性是否能影响附件选择；同时用三种大语言模型（Turkish GPT‑2、DeepSeek‑R1‑Distill‑Qwen‑1.5B‑Turkish、Qwen3‑30B‑Instruct）通过匹配的延续子句的对数概率比较其附件偏好。

**💡 创新点**

首次将细粒度、可量化的世界知识可行性（通过独立评价）与附件歧义相结合，形成一种可跨人类与模型直接对比的实验范式；并揭示即便是强大的多语种模型，在土耳其语的结构约束下也难以可靠地将世界知识用于歧义解析。

**🔧 技术方法**

使用受控句子生成、独立可行性评价、在线强迫式选择实验、对数概率评分与二元逻辑回归、Fisher精确检验等统计方法。

**📊 数据集**

40条土耳其语歧义句子（20个高可行性、20个低可行性），人类实验共86名土耳其母语者；模型使用同一语料进行对数概率对比。

**📈 对比分析**

人类在高可行性条件下附件偏好从26.3%升至65.2%（+38.9个百分点），显著且方向正确；而模型则表现弱或错误：Turkish GPT‑2无差异、DeepSeek‑R1‑Distill仅小幅+10pp、Qwen3‑30B‑Instruct在试点中反向偏好（+90%低可行性）。

**⚠️ 局限性**

样本规模有限（仅40句、20条模型对照），模型范围有限，且人类的强迫式选择与模型的对数概率对比并非同一过程；缺乏实时时间序列（如区段性surprisal）与更广泛的多语种模型对比。

---

## 712. VectraFlow: Long-Horizon Semantic Processing over Data and Event Streams with LLMs

**arXiv ID:** 2604.03855 | [PDF](https://arxiv.org/pdf/2604.03855v1)

**作者:** Shu Chen `[一作]` (Brown University), Ugur Cetintemel `[通讯]` (Brown University)

**通讯引用:** 5998 | [OpenAlex ID](https://openalex.org/A5109862110)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一套名为VectraFlow的语义流处理框架，能够在无结构文本流上执行持续的LLM驱动语义操作（如过滤、映射、聚合、窗口等）并通过语义事件模式算子实现跨文本长周期的复杂事件检测。

**💡 创新点**

创新点包括①将LLM作为连续流处理的算子实现，支持可调节的吞吐量-精度折中；②推出语义事件模式算子，将LLM事件抽取与NFA时序匹配无缝融合，实现对无结构文本的长期时序推理；③提供自然语言到可执行管道的编译器，支持实时调试与可视化。

**🔧 技术方法**

技术手段主要有：大规模语言模型（GPT‑4o‑mini、Qwen‑3 系列）用于文本抽取与过滤；词向量/嵌入模型用于快速聚类与检索；混合LLM+嵌入的执行策略；非确定性有限自动机（NFA）进行事件模式匹配；RAG检索机制用于降低上下文长度。

**📊 数据集**

实验数据集包括：MIMIC‑IV 256 份临床笔记（用于事件模式检测）和 MiDe22 数据集（用于语义聚合与分组）。

**📈 对比分析**

与传统的基线方法（全上下文LLM判断、基于RAG的全上下文判断）相比，语义模式算子（与RAG结合）在保持较高 F1 分数（0.84‑0.86）同时显著降低 token 消耗（从 14.6M 降至 3.1M）并提升吞吐量；LLM-嵌入混合模式在聚合任务上实现了 10‑30% 的吞吐提升，且在保持 F1 > 0.85 的前提下降低了调用成本。

**⚠️ 局限性**

局限性包括：对大规模LLM调用的显著 token 与计算成本；全上下文基线在本地 GPU 上容易 OOM；LLM 的抽取准确性仍受提示与模型限制；NFA 规则编写需要专业知识；框架对实时性与极大数据量的可扩展性尚待进一步验证。

---

## 713. Earth Embeddings Reveal Diverse Urban Signals from Space

**arXiv ID:** 2604.03456 | [PDF](https://arxiv.org/pdf/2604.03456v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 714. Security Analysis of Universal Circuits as a Mechanism for Hardware Obfuscation

**arXiv ID:** 2604.03396 | [PDF](https://arxiv.org/pdf/2604.03396v1)

**作者:** Zain Ul Abideen `[一作]` (University of Idaho), Samuel Pagliarini `[通讯]` (Carnegie Mellon University)

**通讯引用:** 853 | [OpenAlex ID](https://openalex.org/A5060594809)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对使用通用电路（Universal Circuits）实现的硬件加密（UCbo）进行安全性评估，针对最新的 oracle‑guided 与 oracle‑less 攻击（SAT、AppSAT、ATPG、SMT、D-DIP、Icy、BBO、SCOPE）对其结构与密钥恢复效果进行实验与分析。

**💡 创新点**

创新点在于：①首次对 UCbo 进行系统的安全性分析；②提出四个量化指标（CTVR、CGF、KCR、ΔRR）来评估 SAT 攻击难度；③通过实验验证 UCbo 在 OG 攻击中几乎没有成功率（≈50%，相当于随机猜测），在 OL 攻击中 COPE<0.1%，证明其对结构泄露有极强的抗性。

**🔧 技术方法**

采用的技术包括：通用电路（Universal Circuit）加密、基于 SAT 的多种变种（SAT、AppSAT、ATPG、SMT、D‑DIP、Icy）、无 oracle 的结构重合攻击（SCOPE）与黑盒攻击（BBO），以及利用 CNF 统计量生成并归一化 CTVR、CGF、KCR、ΔRR 等指标。

**📊 数据集**

使用的数据集为 ISCAS'89 经典基准（共 9 个电路）以及一个合成的 CMP 大型设计，最大的电路规模超过 200K 条门。所有电路在 UCbo 处理后均展开为组合形式供攻击。

**📈 对比分析**

比较方法：对每个电路分别在 OG 和 OL 场景下运行多种攻击，记录变量数、子句数、运行时间、恢复密钥比率等；用 CTVR、CGF、KCR、ΔRR 等指标对攻击难度进行归一化评估。结果显示：OG 攻击中 CTVR>0.7、CGF>30；OL 攻击中 COPE<0.1%；大多数攻击在 48 小时内均未恢复完整密钥，成功率仅约 50%。

**⚠️ 局限性**

局限性：① UCbo 的 PPA 开销显著，实际工艺实现成本高；②实验仅覆盖组合展开形式，未评估完整的时序电路；③目前仅对已知攻击模型做评估，未知攻击或新型攻击可能仍有风险；④缺乏严格的数学安全证明，安全性主要基于实验结果。

---

## 715. Mapping GitHub Sponsorships: A Longitudinal Observatory for Open-Source Sustainability

**arXiv ID:** 2604.03846 | [PDF](https://arxiv.org/pdf/2604.03846v1)

**作者:** Rylan Hiltz `[一作]` (Trent University), Taher A. Ghaleb `[通讯]` (Trent University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并运营一个持续收集、标准化并公开 GitHub Sponsors 网络的实时观测站，提供交互式仪表盘和可直接使用的 CSV 数据

**💡 创新点**

首创通过优先级图遍历实现高效、增量更新的实时观测；同时公开可复现的完整数据收集与清洗流水线，解决此前缺乏批量赞助数据的问题

**🔧 技术方法**

使用 Flask + React/Tailwind/Ant Design 的全栈架构、Docker 部署、GitHub GraphQL API 与 REST API、OpenStreetMap 地理编码、无头浏览器提取性别代词，并实现自动速率限制与重试

**📊 数据集**

动态生成的 49K+ 开发者与组织的赞助网络数据（约 63K 份赞助关系），含用户 ID、赞助计数、层级、活动指标、地理与性别信息，可导出为 CSV

**📈 对比分析**

与现有的 GHArchive、World of Code、Stack Overflow 等数据源集成，可用于生存分析、回归分析和网络分析；在 72 小时内完成 150k+ API 调用、覆盖 144 个国家，实时更新率保持在每日一次以上，数据质量通过多重清洗与质量标记验证

**⚠️ 局限性**

覆盖仅限于可赞助的用户及其赞助链，可能漏掉未设为可赞助但仍进行赞助的账号；样本为单次 72 小时抓取，时间动态性受限；性别/地点数据依赖自述字段，存在自选偏倚与地理编码质量差异；缺乏历史赞助计数，难以追踪赞助随时间变化

---

## 716. Transmission Neural Networks: Inhibitory and Excitatory Connections

**arXiv ID:** 2604.04246 | [PDF](https://arxiv.org/pdf/2604.04246v1)

**作者:** Shuang Gao `[一作]` (Polytechnique Montreal), Peter E. Caines `[通讯]` (McGill University)

**通讯引用:** 11122 | [OpenAlex ID](https://openalex.org/A5044187847)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

**🎯 论文内容**

在原有的 Transmission Neural Network (TransNN) 模型基础上，加入了抑制连接与神经递质种群，推导了含抑制的二元状态传输动力学，并给出了相应的概率转移、连续状态映射及其极限模型；同时提供了收敛、稳定与收缩性质的充分条件。

**💡 创新点**

① 抑制连接的正式加入，构造了可将二元状态映射到双维连续状态的 TLogSigmoid 激活函数网络；② 通过泊松逼近，推导了神经递质数量趋于无穷时的极限模型；③ 提供了系统收缩与稳定性理论，为大规模网络的控制与分析奠定了基础。

**🔧 技术方法**

离散时间马尔可夫链分析、条件独立性假设、对数-熵变换、泊松逼近、收敛性与收缩分析（矩阵范数、谱半径、线性系统上界），以及对激活函数的凸性/凹性证明。

**📊 数据集**

无；本文为理论性研究，未使用任何实验数据集。

**📈 对比分析**

无实验对比；论文通过理论推导与数学证明来验证所给定模型的正确性与稳定性，未涉及数值实验或性能指标。

**⚠️ 局限性**

① 只给出平凡平衡点（全零）并未讨论非平凡平衡点的存在；② 主要假设（如独立性、无记忆、固定递质数量等）在实际神经网络中可能过于理想；③ 仅考虑离散时间模型，未涉及连续时间或脉冲序列；④ 对大规模网络的可扩展性与计算复杂度未作讨论。

---

## 717. Fusion and Alignment Enhancement with Large Language Models for Tail-item Sequential Recommendation

**arXiv ID:** 2604.03688 | [PDF](https://arxiv.org/pdf/2604.03688v1)

**作者:** Zhifu Wei `[一作]` (Northeastern University), Zhu Sun `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 2163 | [OpenAlex ID](https://openalex.org/A5033957641)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出FAERec框架，融合大语言模型(LMM)的语义嵌入与传统ID嵌入，并通过双层对齐提升长尾顺序推荐性能。

**💡 创新点**

创新点：① 自适应门控融合实现维度级别的ID+LLM融合；② 双层对齐（item-level对比学习 + feature-level结构一致性）并采用课程学习调度平衡两者；③ 预计算LLM嵌入，避免推理开销。

**🔧 技术方法**

技术方法：大语言模型嵌入、门控融合、对比学习、Barlow Twins风格特征层对齐、课程学习调度、Transformer/MLP/线性递归等顺序推荐骨干。

**📊 数据集**

使用的数据集：Amazon Beauty、Amazon Grocery、Yelp三大真实交互数据集。

**📈 对比分析**

对比传统长尾模型（CITIES、LOAM、MELT）、LLM方法（RLMRec、LLMInit、LLM-ESR）以及多种SR骨干（SASRec、FMLP-Rec、LRURec）进行实验。FAERec在整体和长尾指标上均显著提升，整体平均提升约>100%，长尾平均提升>1000%。

**⚠️ 局限性**

局限性：需要预先生成并缓存大量LLM嵌入；对极端冷启动的处理仍不完善；在最热门项目上略逊于RLMRec-Con；未涉及多模态或在线实时更新场景。

---

## 718. AgenticFlict: A Large-Scale Dataset of Merge Conflicts in AI Coding Agent Pull Requests on GitHub

**arXiv ID:** 2604.03551 | [PDF](https://arxiv.org/pdf/2604.03551v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 719. GPU Acceleration of TFHE-Based High-Precision Nonlinear Layers for Encrypted LLM Inference

**arXiv ID:** 2604.04783 | [PDF](https://arxiv.org/pdf/2604.04783v1)

**作者:** Guoci Chen `[一作]` (Peking University), Jie Zhang `[通讯]` (Peking University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出TIGER，一套GPU加速的TFHE高精度非线性层实现框架，用于加密LLM推理；

**💡 创新点**

1) 在GPU上实现WoP-PBS并结合数值迭代实现精度提升；2) 通过批处理与拆分策略充分利用GPU并行；3) 针对FFT与PBS做多种硬件友好优化；

**🔧 技术方法**

GPU-accelerated WoP-PBS、数值逼近（Taylor、区间拆分）、批量PBS拆分、4阶Radix FFT与Karatsuba乘法、固定点算术操作与调度；

**📊 数据集**

使用GPT‑2 Transformer块（及其Softmax、LayerNorm、GELU）进行性能评测；

**📈 对比分析**

与CPU‑基线（CPU‑WoP）和GPU‑FBT对比，TIGER在GELU、Softmax、LayerNorm分别提升7.17×、16.68×、17.05×；整体Transformer块加密推理速度提升约15.5×；

**⚠️ 局限性**

仍受限于TFHE LUT精度（约20位）与高密度键存储，且对非线性层外的其他算子支持有限，跨GPU通信主要集中在归约操作，需进一步优化多GPU扩展与更广泛层支持。

---

## 720. VisACD: Visibility-Based GPU-Accelerated Approximate Convex Decomposition

**arXiv ID:** 2604.04244 | [PDF](https://arxiv.org/pdf/2604.04244v1)

**作者:** Egor Fokin `[一作]` (Simon Fraser University), Manolis Savva `[通讯]` (Simon Fraser University)

**通讯引用:** 17906 | [OpenAlex ID](https://openalex.org/A5091765070)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于可见性边、旋转等变且GPU加速的近似凸分解（ACD）算法，能够在保持较高精度的同时减少凸体数量并避免凸体相交。

**💡 创新点**

创新点包括：①使用可见性边长度作为凸度度量；②在不切割网格的情况下通过可见性边快速评估切平面价值；③利用GPU实现全可见性边并行计算；④采用旋转等变的切平面采样策略，可在每步采样上千平面。

**🔧 技术方法**

技术实现包括：NVIDIA OptiX 与 CUDA 的 GPU 并行可见性计算；基于可见性边的凸度度量与价值函数；随机与面向最大平面的切平面采样；贪婪切割策略；SDF 重建以实现网格水密化及顶点密度统一。

**📊 数据集**

使用的评估数据集包括 V‑HACD、PartNet‑Mobility 以及 Objaverse（随机抽取的 1,000 个网格），并与 CoACD、V‑HACD 等基线方法进行比较。

**📈 对比分析**

与 V‑HACD、CoACD 等基线在凸体数量和使用碰撞感知凸度指标 C(M) 进行对比，结果显示在所有数据集上均能得到更少的凸体且凸度更低；平均计算时间为 16.97 秒（PartNet‑Mobility），显著低于 CoACD 的 36.31 秒，尤其在形状多样、非轴向的模型上表现更优。

**⚠️ 局限性**

局限性包括：基于贪婪算法，可能产生次优解；对原始网格拓扑敏感，需进行重建；目前未结合更高级搜索（如 MCTS）以进一步提升分解质量。

---

## 721. Self-Regulated Personal Contracts as a Harm Reduction Approach to Generative AI in Undergraduate Programming Education

**arXiv ID:** 2604.03256 | [PDF](https://arxiv.org/pdf/2604.03256v1)

**作者:** Aadarsh Padiyath `[一作]` (University of Michigan), Barbara Ericson `[通讯]` (University of Michigan)

**通讯引用:** 2499 | [OpenAlex ID](https://openalex.org/A5005683626)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实施了一种基于危害减轻与自我调节学习理论的 GenAI 合同，让本科生在编程课程中自我反思并规划 AI 工具的使用

**💡 创新点**

首次将危害减轻理念与自我调节学习相结合，提供非约束性、可迭代的决策框架，帮助学生将 AI 使用与个人学习目标对齐

**🔧 技术方法**

使用四轮反思提问（在 Google Docs 里编写、提交）和若干“如果-那么”实现意图规则，配合课程安排的时间点来触发反思

**📊 数据集**

收集 217 名 Python 课程学生的合同提交记录（完成率、反思内容）作为主要数据集，未使用公开数据集

**📈 对比分析**

未与其他方法做严格实验对比，仅报告使用率（97%/92%/85%）和58%学生表示思考方式改变，未给出量化性能指标

**⚠️ 局限性**

主要局限在于学生易忘记合同、时间压力导致自律失败，合同行为仅能帮助新习惯形成而非改变已有习惯；样本单一、缺乏客观学习成效评估

---

## 722. VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models

**arXiv ID:** 2604.03956 | [PDF](https://arxiv.org/pdf/2604.03956v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 723. Uniform Sampling of Proper Graph Colorings via Soft Coloring and Partial Rejection Sampling

**arXiv ID:** 2604.03947 | [PDF](https://arxiv.org/pdf/2604.03947v1)

**作者:** Sarat Moka `[一作]` (University of New South Wales), Ava Vahedi `[通讯]` (Dresden University of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 γ-软着色框架和混合 PRS 算法，用于在任意图上高效地精确采样均匀 k‑着色。

**💡 创新点**

创新点包括：①在每个顶点上引入辅助连续随机变量，形成“被动状态”，使部分拒绝采样能够避免覆盖整个图；②将全局采样拆解为 O(log n) 规模的独立子问题，形成可并行化的混合算法；③证明该拆解可把任何基线采样器的时间从 O(n) 下降到 O(log n)，并给出递归层级可实现近线性时间的潜在方案。

**🔧 技术方法**

核心技术包括：部分拒绝采样 (Partial Rejection Sampling, PRS)、耦合自过去 (Coupling From The Past, CFTP) 及其边界链方法、随机占据/占用模型的渗流理论、递归与迭代级别管理、以及对非退化条件的概率分析。

**📊 数据集**

实验数据集涵盖了：
- 循环图 (Cycle)
- 网格图 (Grid, 5×5、10×10 等)
- 完全图 (Complete)
- 随机正则图 (Random Regular, 3‑regular、4‑regular 等)
- 其他特殊图 (Petersen、C_20 等)。

**📈 对比分析**

比较方法：
- 与朴素拒绝采样 (NRS) 对比，混合 PRS 在大多数图上显著减少重采样次数。
- 与传统 CFTP（Huber、BC20）对比，混合算法在子图上一次性调用 CFTP，整体运行时间下降到 O(L n (log log n)² Δ² log Δ log k)，比直接 CFTP 的 O(n log² n) 改进了约 log² n/(L (log log n)²) 的因子。
- 通过实验验证 L（有效级数）在 1–20 之间，且随 n 不增长，进一步支持理论分析。

**⚠️ 局限性**

局限性：
- 理论上需要 k > 3Δ（或更高的阈值）才能保证 PRS 终止；k 过接近 Δ 时非退化条件失效，算法退化为 NRS。
- L 的上界（是否为常数）尚未正式证明，只能通过实验给出支持；若 L 随 n 增长，线性时间的承诺不成立。
- 依赖于对渗流阈值的估计，图结构特殊（如高度连通图）时可能出现大连通块导致并行化收益有限。

---

## 724. A comparative study on power delivery aspects of compute-in/near-memory approaches using DRAM

**arXiv ID:** 2604.04773 | [PDF](https://arxiv.org/pdf/2604.04773v1)

**作者:** Siddhartha Raman Sundara Raman `[一作]` (University of Texas at Austin), Lizy Kurian John `[通讯]` (University of Texas at Austin)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对基于DRAM的计算内存（PIM）技术在功率供电网络（PDN）上的影响进行了系统综述，并提出了统一的时空分类法。

**💡 创新点**

创新点在于将PIM引起的电流行为归纳为时序（突发 vs 持续）和空间（局部 vs 分布）两维，形成统一的PDN挑战分类框架。

**🔧 技术方法**

采用文献综述、结构分析、PDN负载模型以及现有DRAM时序与控制机制（tRRD、tFAW 等）作为技术手段。

**📊 数据集**

未使用特定数据集，而是基于已有 PIM 技术（如 RowClone、Ambit、DRISA、Newton、HBM-PIM、Neurocube 等）的工作原理与参数进行分析。

**📈 对比分析**

通过比较不同层级（子阵列、银行、3D 堆叠）的 PIM 技术在突发/持续和局部/分布维度上的电流需求，评估其对电压跌落、IR 降和热热点的潜在影响；虽然未给出具体数值，但在理论上展示了不同技术的 PDN 压力级别。

**⚠️ 局限性**

局限性包括缺乏真实系统的电压/温度测量验证、对 PIM 技术的覆盖不完全、以及未考虑新兴的非易失性内存 PIM 方案；此外，建议未来结合仿真/实验来量化 PDN 调度与功率管理策略的实际效果。

---

## 725. Receding-Horizon Control via Drifting Models

**arXiv ID:** 2604.04528 | [PDF](https://arxiv.org/pdf/2604.04528v1)

**作者:** Daniele Foffano `[一作]` (KTH Royal Institute of Technology), Alexandre Proutiere `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 6012 | [OpenAlex ID](https://openalex.org/A5025136069)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了Drifting MPC，一种将漂移生成模型与滚动视窗规划相结合的离线轨迹优化框架，能够在未知动力学下利用离线轨迹数据学习条件轨迹分布并执行最优控制；

**💡 创新点**

创新点在于将漂移模型的正漂移场倾斜为指数调制的成本权重，构造指数倾斜目标分布；在离线数据上直接训练条件生成器，避免模型滚动；结合最佳-M采样提供 δ-最优保证；

**🔧 技术方法**

使用漂移生成模型（Drifting Models）、条件生成器、指数倾斜（exponential tilting）、最优控制正则化目标、最优采样（best‑of‑M）保证、随机梯度训练以及正负 mean‑shift 场的计算；

**📊 数据集**

使用1D质量弹簧阻尼系统的离线轨迹数据，轨迹由LQR、噪声PD和随机开环控制器生成，状态采样范围为[-2,2]×[-2,2]，并以真实LQR作为oracle；

**📈 对比分析**

与漂移先验、DDPM、Diffusion、Guided Diffusion四种基线在不同规划步长（30、50、100）下比较，评估平均成本和平均推理时间；实验结果表明Drifting MPC在成本上接近oracle，且推理时间比扩散方法快数十到数百倍；

**⚠️ 局限性**

局限性包括对离线数据覆盖度的依赖、在高维系统的可扩展性待验证、β 值调节经验性强、在极长规划步长下仍可能出现性能下降，并且未考虑在线交互式数据收集。

---

## 726. Which English Do LLMs Prefer? Triangulating Structural Bias Towards American English in Foundation Models

**arXiv ID:** 2604.04204 | [PDF](https://arxiv.org/pdf/2604.04204v1)

**作者:** Mir Tafseer Nayeem `[一作]` (University of Alberta), Davood Rafiei `[通讯]` (University of Alberta)

**通讯引用:** 2126 | [OpenAlex ID](https://openalex.org/A5071837282)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统审计大型语言模型在美式英语与英式英语之间的结构性偏差，分析预训练语料、分词器与生成输出的差异，并提出动态对齐方法DiAlign。

**💡 创新点**

首次从后殖民视角全面量化标准英语变体的偏倚，并构建1,813对并行美英英式变体词典；提出无训练的分布式对齐算法DiAlign，能够捕捉词汇、语法、结构与风格差异。

**🔧 技术方法**

使用词典匹配与Google Books Ngram统计的对齐评分；对六大预训练语料库进行变体频率审计；对多地区分词器进行分词效率与粒度评估；使用DiAlign评估生成文本的方言倾向。

**📊 数据集**

预训练语料库：Common Crawl（C4）、Wikipedia、BookCorpus、Falcon RefinedWeb、RedPajama、Dolma；分词器：GPT‑4、GPT‑4o、Llama‑3.3、Gemma、DeepSeek、Mistral、StableLM、Velvet、Falcon；生成文本评测集：Natural Questions、ELI5。

**📈 对比分析**

比较方法：在预训练语料中统计变体比例、Wilcoxon检验；在分词器中比较肥度与粒度；在生成输出中计算AmE比例与对齐置信度。实验显示所有模型均偏好AmE，偏差幅度在25–60%之间；分词器中BrE往往分词数更高；在非美国模型或非正式域中BrE占比略增。

**⚠️ 局限性**

局限：仅关注标准美英英式两种变体，未涵盖其他世界英语；DiAlign依赖Google Books Ngram，可能受历史采样偏倚；评估聚焦于问答生成，未检验其他下游任务；未对模型内部表示做更深层分析；后殖民框架在解释上主观性较强。

---

## 727. Evaluation of Bagging Predictors with Kernel Density Estimation and Bagging Score

**arXiv ID:** 2604.03599 | [PDF](https://arxiv.org/pdf/2604.03599v1)

**作者:** Philipp Seitz `[一作]` (Technical University of Applied Sciences Wurzburg Schweinfurt), Andreas Schiffler `[通讯]` (Technical University of Applied Sciences Wurzburg Schweinfurt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出基于核密度估计的袋装预测器评价方法，定义Bagging Score并用其确定集成预测的代表值。

**💡 创新点**

创新点在于用KDE计算Bagging Score以克服均值/中位数在分布非对称时的偏差，并提供置信度量化。

**🔧 技术方法**

使用神经网络集成（1000个3层NN）结合核密度估计与非线性回归技术，对预测结果进行评估。

**📊 数据集**

采用公开的Concrete混凝土强度数据集（1030条样本）进行实验验证。

**📈 对比分析**

通过与均值、中位数以及文献中多种模型的R²、RMSE、MAPE、MAE指标比较，Bagging Score在所有指标上均优于传统方法，排名前列。

**⚠️ 局限性**

局限性包括需足够大集成规模以保证统计可靠性；对分类任务的适用性和极度稀疏数据的鲁棒性尚待进一步研究。

---

## 728. Forgetting to Witness: Efficient Federated Unlearning and Its Visible Evaluation

**arXiv ID:** 2604.04800 | [PDF](https://arxiv.org/pdf/2604.04800v1)

**作者:** Houzhe Wang `[一作]` (School of Cyber Security, University of Chinese Academy of Sciences), Chi Chen `[通讯]` (School of Cyber Security, University of Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了完整的联邦无学习（Federated Unlearning）流程，并设计了基于GAN的可视化评估框架 Skyeye，用于判断模型是否已成功遗忘指定数据。

**💡 创新点**

创新点包括：①在无须存储历史梯度或数据的前提下，利用无能教师模型与知识蒸馏实现高效无学习；②通过注意力图对齐、硬损失约束、学习率衰减、灾难性遗忘抑制等多种机制，兼顾模型准确性与遗忘效果；③通过将无学习模型作为分类器嵌入GAN，生成可视化样本，直观评估遗忘能力。

**🔧 技术方法**

使用技术包括：联邦学习框架（FedAvg）、知识蒸馏、注意力图对齐损失、硬损失、损失上限约束、学习率衰减、参数正则化、GAN（生成器、判别器、分类器）以及后续的后门与成员推断攻击评估。

**📊 数据集**

实验数据集：MNIST、AT&T、CIFAR-10、CIFAR-100，分别配合 LeNet-5、ResNet32/56 等模型进行测试。

**📈 对比分析**

通过与三种基线（完全重训、无能教师蒸馏、注意力蒸馏）比较，使用准确率、后门攻击成功率、成员推断攻击成功率、JSD 与 L2 距离等指标。实验结果表明，该方法在保持或接近重训模型精度的同时，显著降低了攻击成功率，且在所有数据删减率下均表现出最小的分布差异。

**⚠️ 局限性**

局限性：仅在图像分类任务上验证，未覆盖大规模异构联邦场景；评估框架依赖GAN生成的可视化样本，可能无法完全揭示模型内部隐私泄露；对超参数（如 BND、学习率、正则化权重）敏感，需要进一步自动化调优；在极端数据删减率或复杂模型结构下的鲁棒性仍需探索。

---

## 729. Multi-Robot Multi-Queue Control via Exhaustive Assignment Actor-Critic Learning

**arXiv ID:** 2604.03605 | [PDF](https://arxiv.org/pdf/2604.03605v1)

**作者:** Mohammad Merati `[一作]` (Boston University), David Castañón `[通讯]` (Boston University)

**通讯引用:** 4739 | [OpenAlex ID](https://openalex.org/A5103073490)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了多机器人多队列系统的在线任务分配问题，考虑异质随机到达和切换延迟，并将其建模为折扣马尔可夫决策过程；基于“耗尽服务”结构，设计了一种仅对空闲机器人进行重新分配的actor‑critic策略；

**💡 创新点**

创新点在于将耗尽服务结构直接嵌入到策略网络中，使用序列化分配与掩码机制限制决策空间，从而显著提升了在非对称到达率环境下的调度性能；

**🔧 技术方法**

采用PPO强化学习与深度actor‑critic框架，利用队列与机器人嵌入、相似度打分与全局价值估计，形成专门的排队调度网络；

**📊 数据集**

实验使用仿真生成的异质到达率分布（不依赖真实数据集），在多种机器人/队列规模下进行训练和评估；

**📈 对比分析**

通过与ESL基线和在小规模可计算的最优策略对比，EA‑AC在折扣持有成本和平均队列长度上均优于ESL，并在可计算实例中接近最优；

**⚠️ 局限性**

局限性包括：需要大量仿真训练，扩展到更大规模时训练迭代量随机器人数线性增长；缺乏理论最优性证明；仅适用于确定性单时隙切换，且假设队列容量无穷大。

---

## 730. MIRAGE: Online LLM Simulation for Microservice Dependency Testing

**arXiv ID:** 2604.04806 | [PDF](https://arxiv.org/pdf/2604.04806v1)

**作者:** XinRan Zhang `[一作]` `[通讯]`, XinRan Zhang

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种在线 LLM 模拟微服务依赖的测试原语，LLM 在测试时实时生成依赖响应并维护跨请求状态。

**💡 创新点**

创新点在于：① 在运行时直接调用 LLM 而非预生成静态 mock；② 让 LLM 读取依赖源码、调用代码与生产追踪，从而动态推断行为；③ 通过实验证明该方法在多 Benchmark 上显著优于传统 record‑replay、pattern‑mining 与 IR。

**🔧 技术方法**

采用大模型（Claude Opus/Sonnet、Kimi、MiniMax）与 LiteLLM 接口，FastAPI 服务器处理 HTTP 请求，OpenTelemetry 追踪聚合，Context Builder 构建 Prompt。

**📊 数据集**

使用三套微服务基准：Google Online Boutique、Weaveworks Sock Shop 以及自制 Demo，包含 110 条手工设计的测试场景与 3,821 条生产式追踪。

**📈 对比分析**

通过比较 status‑code 与 body‑shape fidelity 评估；在白盒模式下 99%/99%，黑盒 94%/75%，record‑replay 仅 62%/16%；IR 55–86%；成本约 $0.16–$0.82/依赖，单请求约 3s，完整 110 场景约 9.4 分钟。

**⚠️ 局限性**

局限性包括：仅支持 HTTP/JSON，未验证值级别一致；不支持 gRPC/事件驱动；在缺乏源代码时性能下降；Python 适配可能夸大白盒效果；结果受模型预训练数据影响。

---

## 731. Graphic-Design-Bench: A Comprehensive Benchmark for Evaluating AI on Graphic Design Tasks

**arXiv ID:** 2604.04192 | [PDF](https://arxiv.org/pdf/2604.04192v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 732. Inference-Path Optimization via Circuit Duplication in Frozen Visual Transformers for Marine Species Classification

**arXiv ID:** 2604.03428 | [PDF](https://arxiv.org/pdf/2604.03428v1)

**作者:** Thomas Manuel Rost `[一作]` `[通讯]`, Thomas Manuel Rost

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对冻结的 DINOv3 视觉变换器在推理时使用 Circuit Duplication（层级重复）来改进海洋物种分类的嵌入表示，

**💡 创新点**

首次在计算机视觉及海洋物种分类中引入 Circuit Duplication，并证明其在无需梯度训练的情况下可逼近甚至超过全监督模型，

**🔧 技术方法**

使用 Circuit Duplication、冻结 DINOv3 嵌入、PCA 降维以及多种半监督下游分类器（Label Spreading、Self‑Training KNN/SVM、Seeded K‑means、K‑NN）

**📊 数据集**

AQUA20 水下物种分类基准（20 类，严重类别不平衡）

**📈 对比分析**

与冻结基线和全监督 ConvNeXt 进行比较；在最大标签预算下宏 F1 达到 0.875，接近 ConvNeXt 0.889，且四类物种（章鱼、海胆、鱼群、海星）超过全监督基准，

**⚠️ 局限性**

仅使用单一数据集与单一模型，使用极简下游分类器；需遍历 66 条线路、使用转导 PCA，缺乏对机制的解释以及在极少标注场景下可行性的验证

---

## 733. Economic Security of VDF-Based Randomness Beacons: Models, Thresholds, and Design Guidelines

**arXiv ID:** 2604.04744 | [PDF](https://arxiv.org/pdf/2604.04744v1)

**作者:** Zhenhang Shang `[一作]` (Hong Kong University of Science and Technology), Kani Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文建立了针对Verifiable Delay Function（VDF）随机性信标的经济安全框架，结合合理攻击者模型对延迟参数进行定量分析；

**💡 创新点**

创新点在于首次将经济理性攻击者与最优停止理论结合，给出阈值结构和闭式安全阈值，并提出可操作的经济安全延迟参数（ESDP）抽象；

**🔧 技术方法**

使用了随机控制与最优停止理论、Markov决策过程、风险评估模型及多攻击面（ grinding、selective abort、多代理竞争）的扩展；

**📊 数据集**

利用真实云计费数据、硬件性能基准以及公开区块链MEV收益分布进行参数估计；

**📈 对比分析**

通过多场景案例研究（秒级VDF、公共随机服务、治理等）比较，结果显示传统几秒级延迟在大多数MEV情景下易被盈利攻击，经济安全需延迟至数十分钟甚至数天；

**⚠️ 局限性**

局限包括对攻击速度、成本和奖励分布的简化假设、对极端收益尾部处理不足、未覆盖阈值VDF等更复杂信标设计以及动态策略的完整游戏理论分析。

---

## 734. Beyond Semantics: Uncovering the Physics of Fakes via Universal Physical Descriptors for Cross-Modal Synthetic Detection

**arXiv ID:** 2604.04608 | [PDF](https://arxiv.org/pdf/2604.04608v1)

**作者:** Mei Qiu `[一作]` (SDIC Intelligence Information Technology Co., Ltd.), Yanyun Qu `[通讯]` (Xia'men University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过提取并选择5个稳定且判别度高的像素级物理特征（如拉普拉斯方差、Sobel统计、残差噪声方差等），将其文本化后融合到CLIP的文本提示中，以提升跨模态AI生成图像的检测效果。

**💡 创新点**

首次将可解释的像素级物理特征与多模态语言模型相结合，并提出FSDVA特征稳定-判别评估算法，显著提升检测的鲁棒性与跨模型泛化能力。

**🔧 技术方法**

使用15种经典物理特征与FSDVA算法进行特征筛选，将核心特征文本化后与ClipCap生成的语义标题合并，采用CLIP ViT‑L/14 + LoRA进行对比学习和分类训练。

**📊 数据集**

在GenImage（含Midjourney、SDv1.4/5、ADM、GLIDE、Wukong、VQDM、BigGAN）以及UniversalFakeDetect（10个子集）上进行随机采样、训练与测试。

**📈 对比分析**

与C2pClip、UniFD、NPR、FreqNet、FatFormer等方法在8个GenImage测试集上对比，平均准确率提升至约96%以上，在Wukong和SDv1.4上达到99.8%近乎完美的检测效果。

**⚠️ 局限性**

受CLIP文本编码长度限制，无法注入所有特征；单标量特征可能忽略局部细节；缺乏对类提示与特征描述交互的系统性分析；实验范围仅覆盖有限模型与数据，需进一步验证。

---

## 735. Optimizing LLM Prompt Engineering with DSPy Based Declarative Learning

**arXiv ID:** 2604.04869 | [PDF](https://arxiv.org/pdf/2604.04869v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 736. Agentization of Digital Assets for the Agentic Web: Concepts, Techniques, and Benchmark

**arXiv ID:** 2604.04226 | [PDF](https://arxiv.org/pdf/2604.04226v1)

**作者:** Linyao Chen `[一作]` (Shanghai Jiao Tong University), Weinan Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 18423 | [OpenAlex ID](https://openalex.org/A5090720315)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了将代码仓库自动化转化为 Agentic Web 兼容代理的完整流程（A2A‑Agentization），并实现了一个专门的 Agentization Agent。

**💡 创新点**

创新点在于：① 以 A2A 协议为标准，对数字资产进行四阶段（环境搭建、技能提取、内在代理构建、Agent 卡生成）自动化流水线；② 设计了首个专门评估代理化质量的基准 A2A‑Agentization Bench；③ 通过多框架实验验证代理化可行性。

**🔧 技术方法**

主要技术包括：LLM 驱动的代理（Claude Code、Codex CLI、OpenHands、EnvX）、环境合成工具、自动技能包装（tool extraction）、Agent 卡生成（自描述接口）以及 A2A 协议实现。

**📊 数据集**

使用了 35 个真实开源代码仓库（共 522 条评测实例）作为基准数据集，涵盖多种语言与依赖结构。

**📈 对比分析**

与四种代表性自动化软件工程框架集成后进行对比，实验显示 Agentization Agent 能够在 30‑60% 的实例中成功生成可互操作的代理，验证了方法可行性，但仍有显著的环境与技能提取误差。

**⚠️ 局限性**

局限性包括：① 环境依赖冲突导致部署失败；② 复杂代码中的技能提取不完整或不准确；③ 生成的 Agent 卡信息不足，影响跨代理交互；④ 基准覆盖的仓库规模与类型有限，未覆盖全部数字资产类型。

---

## 737. ViBA: Implicit Bundle Adjustment with Geometric and Temporal Consistency for Robust Visual Matching

**arXiv ID:** 2604.03377 | [PDF](https://arxiv.org/pdf/2604.03377v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 738. 3D Gaussian Splatting for Annular Dark Field Scanning Transmission Electron Microscopy Tomography Reconstruction

**arXiv ID:** 2604.04693 | [PDF](https://arxiv.org/pdf/2604.04693v1)

**作者:** Beiyuan Zhang `[一作]` (Beijing Institute of Technology), Ying Fu `[通讯]` (Beijing Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

发展了一种基于3D Gaussian Splatting的稀视角ADF-STEM三维重建方法DenZa-Gaussian，实现了对稀视角投影的高质量3D重建。

**💡 创新点**

将ADF-STEM的弹性散射物理引入3D Gaussian表示，提出可学习的散射强度标量denza、视角一致性系数γ以及联合像素-频域损失，显著抑制缺失楔形伪影并提升稀视角下的重建质量。

**🔧 技术方法**

使用3D Gaussian Splatting、Rutherford散射模型、可学习的denza与γ、像素L1损失+频域幅值损失+SSIM+3D TV正则化、FDK初始化等技术。

**📊 数据集**

以PtNi及Mo掺杂PtNi氧还原纳米催化剂的ADF-STEM倾斜系列（分辨率256×256）为实验数据。

**📈 对比分析**

在45视角与15视角两种稀视角设置下与SIRT、FDK、GENFIRE及原始3DGS对比，DenZa-Gaussian在PSNR与SSIM上分别提升至约33/29和0.885/0.790，比传统与神经方法显著优越。

**⚠️ 局限性**

对散射系数α取固定值，仍需在低剂量或极稀视角下进一步验证，并且模型对元素分布的量化能力有限。

---

## 739. Jellyfish: Zero-Shot Federated Unlearning Scheme with Knowledge Disentanglement

**arXiv ID:** 2604.04030 | [PDF](https://arxiv.org/pdf/2604.04030v1)

**作者:** Houzhe Wang `[一作]` (Institute of Information Engineering Chinese Academy of Sciences), Chi Chen `[通讯]` (Institute of Information Engineering Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Jellyfish的零样本联邦遗忘方案，在不访问原始被遗忘数据的前提下实现模型遗忘并维持性能。

**💡 创新点**

创新点在于：① 使用误差最小化噪声生成代理数据；② 引入知识解耦机制限制遗忘样本在最后卷积层的通道激活；③ 设计多目标损失（hard、confusion、distillation、漂移）并结合梯度和谐与梯度屏蔽；④ 提供零样本修复机制恢复模型精度。

**🔧 技术方法**

采用代理噪声训练、知识解耦正则化、组合损失函数、梯度投影和屏蔽技术、以及基于代理数据的修复策略。

**📊 数据集**

在MNIST、CIFAR‑10和CIFAR‑100三大公开数据集上进行实验。

**📈 对比分析**

与九个基线（含重训练、Fine‑tune、Confuse、QuickDrop、SCRUB、NegGrad+等）对比，Jellyfish在遗忘集上0%准确率、在保留集上接近重训练的精度，JSD和L2距离均小于其他方法，MIA成功率下降至≈50%。

**⚠️ 局限性**

局限性包括：对代理噪声质量高度依赖；超参数（α、μc、μd等）需要精细调节；在高度异构或极大规模联邦场景下的通信与计算开销尚未彻底评估；对复杂模型或多任务场景的适用性仍待验证。

---

## 740. Autoencoder-Based Parameter Estimation for Superposed Multi-Component Damped Sinusoidal Signals

**arXiv ID:** 2604.03985 | [PDF](https://arxiv.org/pdf/2604.03985v1)

**作者:** Momoka Iida `[一作]` (Tokyo City University), Hirotaka Takahashi `[通讯]` (Tokyo City University)

**通讯引用:** 9099 | [OpenAlex ID](https://openalex.org/A5049271542)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

采用自编码器从噪声多成分阻尼正弦波中直接提取频率、相位、衰减时间和幅值参数。

**💡 创新点**

将潜在空间维度与物理参数一一对应，使网络能够直接学习参数表示；同时验证不同成分数和训练分布下的鲁棒性。

**🔧 技术方法**

使用九层自编码器（编码器+解码器），Adam优化，dropout，tanh激活；训练时利用均方误差损失估计参数，解码器重建波形。

**📊 数据集**

合成的短时阻尼正弦波数据，时长5µs，采样率200MHz，分别生成包含2、5个成分的样本，训练数据按高斯或均匀分布生成。

**📈 对比分析**

通过匹配得分、参数分布对比及相对/绝对误差评估，结果显示在高噪声、亚主导或相位相反的难点情况下仍能保持95%+匹配得分，参数误差在几%以内，优于传统方法。

**⚠️ 局限性**

误差在参数分布边缘、频率尾部及高噪声下增大；不同成分数导致训练样本稀疏，导致性能随训练分布的敏感性；需要进一步提升对分布依赖的鲁棒性。

---

## 741. Structured Multi-Criteria Evaluation of Large Language Models with Fuzzy Analytic Hierarchy Process and DualJudge

**arXiv ID:** 2604.03742 | [PDF](https://arxiv.org/pdf/2604.03742v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 742. SciLT: Long-Tailed Classification in Scientific Image Domains

**arXiv ID:** 2604.03687 | [PDF](https://arxiv.org/pdf/2604.03687v1)

**作者:** Jiahao Chen `[一作]` (Renmin University of China), Bing Su `[通讯]` (Renmin University of China)

**通讯引用:** 25524 | [OpenAlex ID](https://openalex.org/A5024756660)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究在科学图像领域长尾分类时，探讨基于预训练模型的参数高效微调（PEFT）的效果，并提出SciLT框架

**💡 创新点**

发现预训练模型的尾部类特征在次高层更具信息，并利用多层特征融合与双头学习实现头尾类平衡；通过自适应门控融合次高层与最终层特征来提升性能

**🔧 技术方法**

使用Vision Transformer（ViT）+ CLIP预训练、AdaptFormer微调、Logit Adjustment与Cross‑Entropy损失、可学习门控融合、双分类头与特征空间Wasserstein距离分析

**📊 数据集**

三大科学图像数据集：Blood（细胞图像）、ISIC（皮肤病图像）、NIH‑Chest（胸部放射图像）

**📈 对比分析**

与从头训练的ResNet-18及多种长尾学习方法（CE、CB、LDAM、Focal、LA、LADE）对比；SciLT在所有数据集上实现了更高的整体准确率、宏平均准确率以及新的BalancedScore，尤其在尾部类显著提升

**⚠️ 局限性**

仅利用次高层特征，未深入探索更丰富的多层交互；在NIH‑Chest中对多数类的准确率略有下降；计算开销虽小但仍高于单一头模型

---

## 743. Invisible Adversaries: A Systematic Study of Session Manipulation Attacks on VPNs

**arXiv ID:** 2604.04099 | [PDF](https://arxiv.org/pdf/2604.04099v1)

**作者:** Yuxiang Yang `[一作]` (Tsinghua University), Ke Xu `[通讯]` (Tsinghua University)

**通讯引用:** 11973 | [OpenAlex ID](https://openalex.org/A5100665814)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d`

**🎯 论文内容**

在多租户VPN服务器中发现并验证了三类会话级攻击：端口耗尽型DoS、TCP会话劫持以及DNS响应劫持，并在5种主流连接跟踪框架与9家商业VPN上进行实验评估。

**💡 创新点**

利用共享连接跟踪表、端口保留行为以及TCP RST/ACK状态验证缺陷，首次从离线攻击者角度展示可在同一VPN服务器内对他人会话进行操纵的完整攻击链，提出了端口耗尽、TCP劫持和DNS劫持三种新型会话操纵技术。

**🔧 技术方法**

端口扫描与猜测、TCP/UDP伪造、连接跟踪状态推测、RIP、ACK、RST校验、事务ID暴力、随机端口分配与随机化、Linux/FreeBSD 内核跟踪框架（Netfilter、PF、IPFilter、IPFW、natd）等多种网络/内核技术。

**📊 数据集**

使用自建实验平台（Ubuntu、FreeBSD）、公开的Nginx、FTP与自建DNS解析服务器，配合9家主流VPN服务（ExpressVPN、NordVPN、PIA等）进行真实部署测试；未使用公开数据集，而是基于实验生成的流量和配置。

**📈 对比分析**

对比各VPN在端口耗尽、TCP注入与DNS劫持下的成功率与耗时；结果显示大多数VPN在端口耗尽DoS成功率>90%，TCP注入平均耗时≈64 秒，DNS劫持成功率20–70%，攻击时间从几秒到数十秒不等，表明大多数框架和服务易受攻击。

**⚠️ 局限性**

实验受限于单一VPN服务器内部用户、离线攻击者假设、对高频通信或代理/随机端口策略的抵御能力有限；未评估在多租户云VPN、WireGuard等新协议及高负载场景下的可扩展性与普适性。

---

## 744. Shower-Aware Dual-Stream Voxel Networks for Structural Defect Detection in Cosmic-Ray Muon Tomography

**arXiv ID:** 2604.03741 | [PDF](https://arxiv.org/pdf/2604.03741v1)

**作者:** Parthiv Dasgupta `[一作]` (Heritage Institute of Technology), Sudeshna Goswami `[通讯]` (Heritage Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了 Shower-Aware Dual-Stream Voxel Network，利用 muon 的散射角度和次级电磁闪耀乘数信息对钢筋混凝土体积进行六类体素分割。

**💡 创新点**

创新点是将散射角度和闪耀乘数两条物理信号分离为双流输入，并通过跨注意力融合，证明闪耀多重性是检测钢筋混凝土缺陷的关键特征。

**🔧 技术方法**

采用三维卷积网络、交叉注意力机制、注意力门、深度监督、焦点+Dice 损失，并结合 Geant4 V11.x 的 Vega 仿真框架生成特征张量。

**📊 数据集**

使用由 Vega 生成的 4.5 万万 muon 事件（900 体积）及 60 个独立验证体积的合成数据，包含蜂窝化、剪切破裂、腐蚀空洞和分层脱附四种缺陷。

**📈 对比分析**

通过与仅使用散射流或仅使用闪耀流、去掉注意力门、无深度监督等五个模型变体对比，Full 模型在验证集上整体 Dice 0.945、缺陷平均 Dice 0.683；在 60 个未见体积上 voxel 准确率 96.3%，四种缺陷的 volume 检测灵敏度 100%。

**⚠️ 局限性**

局限包括 50 mm 体素分辨率不足以分辨薄层脱附和微小空洞、仅使用单能量 4 GeV 垂直入射 muon，未验证真实宇宙射线谱与环境噪声。

---

## 745. DiffSparse: Accelerating Diffusion Transformers with Learned Token Sparsity

**arXiv ID:** 2604.03674 | [PDF](https://arxiv.org/pdf/2604.03674v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 746. SAIL: Scene-aware Adaptive Iterative Learning for Long-Tail Trajectory Prediction in Autonomous Vehicles

**arXiv ID:** 2604.04573 | [PDF](https://arxiv.org/pdf/2604.04573v1)

**作者:** Bin Rao `[一作]` (University of Macau), Hai Yang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 29457 | [OpenAlex ID](https://openalex.org/A5045705475)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一个面向自动驾驶轨迹预测长尾问题的完整框架——SAIL，涵盖属性导向的数据增强、属性解耦特征学习、适应性对比学习、动态聚类与聚焦分解对比学习等多阶段模块，能够在安全关键的稀有场景中实现高精度轨迹预测。

**💡 创新点**

创新点在于①提出基于预测误差、碰撞风险、状态复杂度三维属性的长尾定义并与之关联的定制化增强策略；②引入连续余弦动量调度与相似度加权硬负样本挖掘的适应性对比学习；③采用动态特征聚类（EFC）与聚焦分解对比学习（FDCL）生成高质量伪标签，三者协同实现对长尾样本的精准建模。

**🔧 技术方法**

使用技术包括 Transformer + GAT 场景表示；多头注意力解耦属性特征并通过 MLP 回归属性；连续余弦动量的 Adaptive Momentum Contrastive Learning；动态聚类 Evolving Feature Clustering；聚焦分解对比学习 Focused Decoupled Contrastive Learning；属性导向轨迹增强（Simplify、Shift、Mask、Subset）；GRU+MDN 轨迹生成。

**📊 数据集**

实验数据集为公开的 nuScenes（车辆轨迹）和 ETH/UCY（行人轨迹）两大基准。

**📈 对比分析**

在多维长尾子集（Top 1%–5%）以及模型无关的最坏案例评估中，SAIL 在 minADE/minFDE 上相较于最新基线提升约 10%–30%（尤其最难 1% 样本 minFDE 降低 28.8%），并在完整数据集上取得最优或接近最优的 minFDE、minADE、MR 指标；推理速度仅 18 ms，明显快于其他 SOTA 方法。

**⚠️ 局限性**

局限在于对极端长尾场景（如严重遮挡或缺乏交通信号信息的交叉口）仍易出现误判，说明仅靠视觉与轨迹历史不足，未来需引入更丰富的外部先验（如 V2X）提升鲁棒性。

---

## 747. Generative AI for material design: A mechanics perspective from burgers to matter

**arXiv ID:** 2604.03409 | [PDF](https://arxiv.org/pdf/2604.03409v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 748. LoMa: Local Feature Matching Revisited

**arXiv ID:** 2604.04931 | [PDF](https://arxiv.org/pdf/2604.04931v1)

**作者:** David Nordström `[一作]` (Chalmers University of Technology), Fredrik Kahl `[通讯]` (Chalmers University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

重新评估传统的局部特征匹配方法，结合大规模多样化训练数据、现代训练技巧和模型容量扩展，提出了LoMa特征描述器与匹配器，并创建了1000对手工标注的HardMatch极端匹配基准。

**💡 创新点**

①通过大规模、跨领域的数据混合和改进的训练流程显著提升匹配鲁棒性；②在传统稀疏匹配框架内实现与密集或端到端重建方法竞争甚至超越的性能；③发布HardMatch基准，突破现有评测饱和瓶颈；④公开代码与模型，推动社区复现与进展。

**🔧 技术方法**

使用DaD关键点检测、DeDoDe描述器、LightGlue匹配器框架，结合双软最大化损失、层级监督与早停机制；采用Transformer自/跨注意力、RoPE位置嵌入、双软匹配矩阵；在A100 GPU上进行大规模训练（50K步描述器、250K步匹配器）。

**📊 数据集**

训练集涵盖17个3D数据集（ScanNet++, BlendedMVS, MegaDepth, MegaScenes, SpatialVID, etc.），并自制的HardMatch（1000对图像，覆盖9类极端场景）。评测基准包括WxBS、HardMatch、IMC 2022、MegaDepth、ScanNet、InLoc、Oxford Day‑and‑Night、RUBIK等。

**📈 对比分析**

与现有稀疏匹配器（SuperGlue、LightGlue、SuperPoint+ALIKED等）和密集/端到端方法对比，LoMa在HardMatch提升18.6 mAA，WxBS提升29.5 mAA，InLoc 21.4 mAA（1 m、10°），RUBIK AUC提升24.2，IMC 2022 mAA提升12.4。相对其他方法，性能提升幅度在15–30 mAA之间，显著领先。

**⚠️ 局限性**

局部特征描述器在大规模训练时易过拟合；对HardMatch中特定子组（如极端视角、Doppelgängers）仍表现欠佳；HardMatch基准仍存在地理/时间偏差；评测依赖相机固定、透视场景，可能限制在非平面或非透视场景中的泛化。

---

## 749. Human-Robot Copilot for Data-Efficient Imitation Learning

**arXiv ID:** 2604.03613 | [PDF](https://arxiv.org/pdf/2604.03613v1)

**作者:** Rui Yan `[一作]` (University of California San Diego), Xiaolong Wang `[通讯]` (University of California San Diego)

**通讯引用:** 21047 | [OpenAlex ID](https://openalex.org/A5100424261)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了可跨机器人种类的人机协同复制框架 Human‑Robot Copilot，用于高精度任务的可调缩放远程操控与在线数据增量学习。

**💡 创新点**

创新点在于引入可调缩放因子实现精细与粗略控制的双向协作，兼容多种机械臂；通过剪辑式人机在环微调而非完整轨迹，显著提升数据效率。

**🔧 技术方法**

采用逆运动学映射、双向同步控制、基于 ACT 的增量训练、ResNet‑18 视觉特征、低成本桌面控制器等技术。

**📊 数据集**

实验数据来源为 Robomimic Benchmark（PickPlace(Can)、NutAssembly(Square)）的仿真数据，以及自制的随机化方块分类与汉诺塔插入任务的真实世界数据。

**📈 对比分析**

与基准 Policy、键盘和 VR 控制器的在线修正进行对比，实验显示 Copilot 在相同轨迹数量下成功率提升约10–20%，且数据采集时间缩短约30–50%。

**⚠️ 局限性**

局限性包括需使用固定缩放因子导致控制不自适应；补偿多模态示例有限，纠正动作主要聚焦细化现有策略而非引入新模式。

---

## 750. Ruling Out to Rule In: Contrastive Hypothesis Retrieval for Medical Question Answering

**arXiv ID:** 2604.04593 | [PDF](https://arxiv.org/pdf/2604.04593v1)

**作者:** Byeolhee Kim `[一作]` (Asan Medical Center), Tae-Joon Jeon `[通讯]` (Asan Medical Center)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Contrastive Hypothesis Retrieval (CHR) 框架，通过生成目标与模拟假设并使用对比式检索来提升医疗问答系统的检索质量。

**💡 创新点**

在检索前引入模拟假设 H^-，显式抑制难负样本，模仿临床差异诊断思维，实现检索向量的对比性偏移，从而避免传统查询扩展方法对误导性内容的依赖。

**🔧 技术方法**

使用大型语言模型（Qwen2.5-72B-Instruct）一次性生成 H^+ 与 H^-，采用 MedCPT 稠密检索器与 MedCorp 语料库，并在检索阶段计算 S(d)=Sim(d,H^+)−λ·Sim(d,H^-)；答案生成则使用 Llama‑3‑8B、Qwen2.5‑7B、Gemma‑2‑9B 三种生成器。

**📊 数据集**

在 MMLU‑Med、MedQA 与 BioASQ 三大医疗问答基准上进行评估。

**📈 对比分析**

与 Standard RAG、HyDE、Query2Doc、CSQE、ThinkQE 五种基线在三数据集、三生成器的 18 组对比中，CHR 在所有设置下均取得最高准确率，平均提升约 6–8 点，其中 BioASQ 提升 10.4 点、MedQA 提升 ≥5 点。

**⚠️ 局限性**

受限于 H^+ 与 H^- 的语义分离度；若两者高度重叠（语义共现崩溃）会削弱对比效果；λ 的固定值不适用于所有情形，未来需自适应调节；目前仅在 MedRAG 任务和固定检索器上验证，需进一步扩展到其他检索器和开放式 QA 格式。

---

## 751. Symbolic-Vector Attention Fusion for Collective Intelligence

**arXiv ID:** 2604.03955 | [PDF](https://arxiv.org/pdf/2604.03955v1)

**作者:** Hongwei Xu `[一作]` `[通讯]` (SYM.BOT), Hongwei Xu (SYM.BOT)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Symbolic-Vector Attention Fusion (SVAF)，一种跨域信号评估与混合机制，利用七个语义字段实现按字段选择、重混与新知识生成。

**💡 创新点**

创新点：①结构化 CAT7 语义字段和 per‑field 评估；②神经融合门与交叉字段注意力自动发现跨域重要字段（如 mood）；③两级耦合架构（peer‑level drift + content‑level SVAF）支持冷启动与可扩展；④与 CfC 连续时间神经网络结合形成完整 Mesh Memory Protocol。

**🔧 技术方法**

技术：多层 Transformer、Per‑field encoder + 交叉字段 MHA、融合门 MLP、重混生成 DAG、CfC 连续时间神经网络、all‑MiniLM‑L6‑v2 语义编码器、ONNX 部署、加密传输、离线/在线评估路径。

**📊 数据集**

数据集：237,120 训练样本，来源于 273 个 LLM 编写的多代理叙事情景，覆盖 20 种代理类型，按 85/15 训练/验证拆分。

**📈 对比分析**

与基线比较：scalar、scalar+temporal、heuristic per‑field、SVAF。SVAF 在 3‑分类任务上达 78.7% 准确率，较 scalar 提升 11.9pp，较 heuristic 提升 5.6pp；在接受/拒绝二分类上 F1 分别提升 1–2pp；在线部署中 heuristic 0.07 ms，神经 50–100 ms（warm）。

**⚠️ 局限性**

局限：训练数据为合成标签，缺少真实人类评估；单用户部署，缺乏大规模多用户验证；门控仅通过软约束学习，未彻底验证更细粒度字段区分；编码器质量限制重混准确性；神经模型启动延迟高，难以部署在资源受限环境。

---

## 752. RL-Driven Sustainable Land-Use Allocation for the Lake Malawi Basin

**arXiv ID:** 2604.03768 | [PDF](https://arxiv.org/pdf/2604.03768v1)

**作者:** Ying Yao `[一作]` (Georgia Institute of Technology), Ying Yao `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 6116 | [OpenAlex ID](https://openalex.org/A5101685784)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

使用深度强化学习框架，在拉穆拉湖盆地对土地利用进行优化，以最大化生态系统服务价值。

**💡 创新点**

创新点在于将生态价值与空间连贯性（连通性奖励与水体缓冲惩罚）融合进多目标奖励函数，并通过动作掩码实现物理约束，从而将深度RL应用于生态敏感区域的土地利用规划。

**🔧 技术方法**

使用了Proximal Policy Optimization（PPO）与MaskablePPO算法，配合GridCNN特征提取网络、动作掩码机制以及包含连通性和缓冲惩罚的奖励函数。

**📊 数据集**

采用ESA WorldCover（Sentinel‑2）土地覆盖数据、MODIS evapotranspiration、Costanza等生态系统服务价值数据库，并以拉穆拉湖西岸25×25 km区域为实验样本。

**📈 对比分析**

通过三种情景（单纯ESV、ESV+空间奖励、再生农业）进行对比；结果表明在空间奖励情景下，RL能产生更生态连贯的土地利用布局，尽管单元价值增幅略低，但整体生态现实性更好；性能主要以可视化热图和ESV增幅呈现。

**⚠️ 局限性**

主要限制包括：网格分辨率低、ESV系数固定不随时间变化、未与传统优化方法（如遗传算法、整数规划）进行基准比较、动作掩码粒度不够精细，以及仅在单一区域验证，缺乏泛化性。

---

## 753. Stratifying Reinforcement Learning with Signal Temporal Logic

**arXiv ID:** 2604.04923 | [PDF](https://arxiv.org/pdf/2604.04923v1)

**作者:** Justin Curry `[一作]` (University at Albany State University of New York), Alberto Speranzon `[通讯]` (Lockheed Martin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出基于分层（poset）语义的Signal Temporal Logic，并将其作为奖励函数训练Transformer‑XL 强化学习代理。

**💡 创新点**

创新点在于将 STL 视为分层空间的原子谓词，构造分层空间与时空结构对应的语义，并提出 VGT‑dot 特征用于检测高维嵌入的分层结构。

**🔧 技术方法**

使用了分层理论、STL 鲁棒性、半代数 tubular 邻域、VGT 与 VGT‑dot、HADES、DIC、UMAP/ISOMAP、聚类、Transformer‑XL + PPO 等技术。

**📊 数据集**

主要数据集为 Minigrid 游戏（空地、钥匙门、硬币收集等）以及训练得到的 256 维令牌嵌入。

**📈 对比分析**

与 DIC 和 HADES 对比，VGT‑dot 在分层检测上表现更好；实验在嵌入空间中呈现小时玻璃形状，表明存在分层结构，但未给出量化性能指标。

**⚠️ 局限性**

局限性包括：高维分层提取依赖低维投影，HADES 在内存上受限，tubular 邻域体积增长法则尚未系统化，实验仅为初步验证。

---

## 754. TimeSeek: Temporal Reliability of Agentic Forecasters

**arXiv ID:** 2604.04220 | [PDF](https://arxiv.org/pdf/2604.04220v1)

**作者:** Hamza Mostafa `[一作]` (University of Waterloo), Dennis Lee `[通讯]` (Automorphic Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了TimeSeek基准，用于在预测市场生命周期的不同时间点评估LLM预测器的可靠性，涵盖10个前沿大模型、150个CFTC监管的Kalshi二元市场，并在每个市场的5个生命周期检查点下生成15,000个预测，分别考虑启用或禁用网页搜索；

**💡 创新点**

创新点在于：①引入时间感知评估框架，揭示模型性能随市场生命周期的变化；②通过与网页检索的对照，展示检索的双刃效应；③探讨模型与市场误差的相关性与类别差异，提出选择性工具使用与模型混合策略的可能性；

**🔧 技术方法**

使用的技术包括：LLM代理架构（4节点状态机）、基于Brier Score和Brier Skill Score的评估指标、两种工具使用条件（搜索/不搜索）、简单两模型平均集成；

**📊 数据集**

使用的数据集是从Kalshi收集的150个二元市场，涵盖政治、体育、宏观经济、科学/技术和金融等5个类别，每个市场满足$5,000+交易量、后10月2025解算时间等筛选条件；

**📈 对比分析**

比较方法为在5个时间点和两种工具条件下对10个模型进行BSS评估，结果显示模型在市场早期和高不确定性场景下表现最优；检索在整体上提升BSS，但在12%的模型-检查点组合中反而降低性能；两模型平均虽降低损失但未超过市场；

**⚠️ 局限性**

局限性包括：仅给出点估计，缺乏置信区间；子组分析样本量有限，未能稳健推断；仅评估静态预测，未考虑实际交易成本、滑点等；工具集仅限网页搜索，可能未能覆盖所有信息源；

---

## 755. Regime-Calibrated Demand Priors for Ride-Hailing Fleet Dispatch and Repositioning

**arXiv ID:** 2604.03883 | [PDF](https://arxiv.org/pdf/2604.03883v1)

**作者:** Indar Kumar `[一作]` (Independent Researcher), Akanksha Tiwari `[通讯]` (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于历史需求“制度化”匹配的出租车调度框架，先将历史行程划分为多时段制度，再用六指标相似度集合选取与当前时段最相似的历史案例，构造需求先验，并用该先验驱动LP最优重定位与批量Hungarian匹配；

**💡 创新点**

创新点在于：1）不依赖复杂训练，采用无监督的相似度检索完成需求校准；2）将相似度加权集成六种分布/事件/时空度量，实现对多维需求特征的匹配；3）通过理论证明与实验展示需求先验校准直接提升LP重定位效果，实现两段式乘客等候时间显著下降；

**🔧 技术方法**

使用六指标相似度集合（Kolmogorov–Smirnov、Wasserstein‑1、特征距离、方差比、事件模式、时空相似度），线性规划重定位模型，Hungarian算法批量匹配，OSRM路网验证与Haversine近似，Bootstrap与Friedman等统计检验；

**📊 数据集**

NYC TLC 2024年1月与6月的黄色出租车记录（约520万行），以及芝加哥TNP 2024年数据（约340万行）用于跨城市验证；

**📈 对比分析**

与传统回放+最近邻、回放+批量、仅校准、校准+启发式重定位等基线进行对比，平均等待时间降低31.1%（95%CI 26.5–36.6%），P95下降37.6%，Gini系数从0.441降至0.409，所有8种场景均取得显著且一致的提升；

**⚠️ 局限性**

局限包括：仅在曼哈顿范围内验证；仅使用直线Haversine距离（未充分考虑交通随机性）；固定4小时制度块，可能无法捕捉瞬时变化；历史库覆盖有限，难以匹配极端天气等罕见事件；未考虑拼车、多乘客、动态定价等实际运营因素；

---

## 756. Geometric Limits of Knowledge Distillation: A Minimum-Width Theorem via Superposition Theory

**arXiv ID:** 2604.04037 | [PDF](https://arxiv.org/pdf/2604.04037v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 757. InCaRPose: In-Cabin Relative Camera Pose Estimation Model and Dataset

**arXiv ID:** 2604.03814 | [PDF](https://arxiv.org/pdf/2604.03814v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 758. BoxComm: Benchmarking Category-Aware Commentary Generation and Narration Rhythm in Boxing

**arXiv ID:** 2604.04419 | [PDF](https://arxiv.org/pdf/2604.04419v1)

**作者:** Kaiwen Wang `[一作]` (Tsinghua University), Ji Wu `[通讯]` (Tsinghua University)

**通讯引用:** 5911 | [OpenAlex ID](https://openalex.org/A5029547618)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出BoxComm数据集并开展搏击类体育解说生成研究

**💡 创新点**

首创搏击赛事的语义类别标签与两种评估协议，提出基于事件提示的EIC-Gen方法

**🔧 技术方法**

使用多模态大型语言模型、结构化拳击事件提取、BERTScore与LLM判定一致性

**📊 数据集**

BoxComm：445场世界拳击锦标赛视频、52K句解说与260K拳击事件

**📈 对比分析**

在类别条件生成与节奏评估上与GPT‑4o‑mini、LLaVA‑OV、Qwen3‑VL等模型对比，EIC‑Gen显著提升事实一致性（如战术类Acc从约20%提升至≈34%）

**⚠️ 局限性**

模型仍难捕捉毫秒级细微动作，流式生成的节奏与时序匹配不足，且仅针对拳击，通用性有限

---

## 759. FactReview: Evidence-Grounded Reviews with Literature Positioning and Execution-Based Claim Verification

**arXiv ID:** 2604.04074 | [PDF](https://arxiv.org/pdf/2604.04074v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 760. The Democratic Ontology Deficit: How AI Systems Fail to Represent What Democracy Requires

**arXiv ID:** 2604.03865 | [PDF](https://arxiv.org/pdf/2604.03865v1)

**作者:** Robert M. Ceresa `[一作]` (Huston-Tillotson University), Juan E. Ceresa `[通讯]` (Huston-Tillotson University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了当前大型语言模型在民主公共生活中的“民主本体缺失”，通过代表性工程方法评估其在角色、责任、关系、目的四个公民原语上的默认表示。

**💡 创新点**

首次量化证明大模型缺乏民主所需的四个原语，并提出“公民架构”作为对齐设计的新目标，开辟了对齐研究关注民主机构结构的新方向。

**🔧 技术方法**

采用代表性工程（representation engineering）结合对比刺激提取阅读向量，分析模型内部表示空间中的默认方向。

**📊 数据集**

使用自制的100个日常生活情景对比刺激进行训练，验证同方法的诚实向量，实验对象为Llama‑2‑13b、Mistral‑7B、Meta‑Llama‑3‑8B三种指令调优模型。

**📈 对比分析**

在相同层次和方法下，将公民原语向量与诚实向量对比：诚实得分0.707，公民角色得分-0.047；跨模型、跨代复现结果显示角色、目的、责任在独立基线下均低于0.5，表明默认本体缺失。

**⚠️ 局限性**

局限性包括：仅评估内部表示而非行为表现；只针对三种模型与特定规模；对比刺激的操作化可能影响效果大小。

---

## 761. Dynamic Whole-Body Dancing with Humanoid Robots -- A Model-Based Control Approach

**arXiv ID:** 2604.03999 | [PDF](https://arxiv.org/pdf/2604.03999v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 762. ClickAIXR: On-Device Multimodal Vision-Language Interaction with Real-World Objects in Extended Reality

**arXiv ID:** 2604.04905 | [PDF](https://arxiv.org/pdf/2604.04905v1)

**作者:** Dawar Khan `[一作]` (King Abdullah University of Science and Technology), Ivan Viola `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一个完全离线的 XR 多模态 VLM 系统 ClickAIXR，用户通过手柄和注视锁定的裁剪窗口精确选择现实物体，并在头显上直接回答自然语言查询。

**💡 创新点**

创新点在于：① 用注视锁定的裁剪窗口实现无分割、无网络的精确 ROI 选取；② 在 Magic Leap 2 上实现完整的 ONNX VLM、ASR 与 TTS 推理，保证数据本地化与低延迟；③ 将云端大模型体验与本地 XR 交互做可比评估。

**🔧 技术方法**

使用的技术包括：ONNX Runtime + ViT‑GPT‑2 视觉‑语言编码器‑解码器模型、Vosk 语音识别、Unity Barracuda/MLSDK C API、手柄滑块调节裁剪框、GPU/CPU 推理、离线 TTS。

**📊 数据集**

使用的公开数据集有：Book Covers（224×224 固定尺寸）和 COCO（室内场景多尺寸）用于延迟测评；实验对象为现场实际物体拍摄。

**📈 对比分析**

与 Gemini 2.5 Flash 与 ChatGPT 进行对比，采用 SUS 调查和任务完成时间评估。延迟约 5.4–5.5 s；SUS 评分分别为 81.9（Gemini）、76.7（ChatGPT）和 60.0（ClickAIXR），显示本地系统在可接受范围但低于云端大模型。

**⚠️ 局限性**

局限性包括：① 推理时间比云端更长，影响即时交互；② 当前 VLM 训练目标为图像描述，缺乏指令跟随能力；③ 用户对 ROI 选择的精确度和交互习惯仍需优化，导致 SUS 评分偏低；④ 仅支持 Magic Leap 2，未验证其他 XR 设备或更大模型的可移植性。

---

## 763. How can LLMs Support Policy Researchers? Evaluating an LLM-Assisted Workflow for Large-Scale Unstructured Data

**arXiv ID:** 2604.04479 | [PDF](https://arxiv.org/pdf/2604.04479v1)

**作者:** Yuhan Liu `[一作]` (Princeton University), Andrés Monroy-Hernández `[通讯]` (Princeton University)

**通讯引用:** 7115 | [OpenAlex ID](https://openalex.org/A5013065278)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对政策研究者，构建并评估了基于大型语言模型（LLM）的主题分析工作流，支持从在线论坛（如Reddit）和聊天机器人访谈中快速提取并归纳公众观点，并在用户界面上提供交互式分析。

**💡 创新点**

创新点包括：①将QuaLLM主题分析框架迁移到政策研究场景，加入高层主题预设与可调优的prompt；②设计面向非程序员的用户界面；③在大规模（数百万Reddit帖子）和大样本（1058次访谈）数据上进行验证，并与权威政策报告进行主题覆盖度对比；④探讨LLM工作流在政策研究中的可行性、优势与局限。

**🔧 技术方法**

使用技术主要是：多阶段prompt工程（Quote Extraction、主题提取、主题映射、报告生成），GPT‑4o-mini/ GPT‑4o 等 LLM；RESTful API、Flask 前后端通信；Python Pandas 与 JSONL 数据处理；以及基于OpenAI API的聊天机器人访谈收集。

**📊 数据集**

数据集包括：①Reddit数据集（约5.49M条目，122k条相关引用）来自Academics Torrents/The Eye；②1,058名美国成年人通过聊天机器人访谈得到的文本；③六份关于AI经济影响的权威政策报告（CBS News、Gallup、APA、Pew、World Economic Forum 等）作为对照。

**📈 对比分析**

比较方法：先提取两大数据源（Reddit 与访谈）主题，并人工提取权威报告主题；随后对齐并统计重叠主题比例（总体覆盖率约88.6%），发现大多数报告主题在两种数据源中均出现，且Reddit/访谈能发现报告未提及的新主题。性能上，工作流能在10–15分钟内为研究者生成数十个主题，远快于传统调查与访谈收集。

**⚠️ 局限性**

局限性：①数据代表性受限（Reddit用户偏年轻、网络化，访谈样本虽多样但非全国代表）；②LLM可能出现幻觉与误标，需要人工校对；③对访谈深度与情感分析支持不足；④依赖高质量prompt调优，非技术人员可能难以复制；⑤缺乏实时可视化与结果可追溯性，影响研究者信任。

---

## 764. Emergent Inference-Time Semantic Contamination via In-Context Priming

**arXiv ID:** 2604.04043 | [PDF](https://arxiv.org/pdf/2604.04043v1)

**作者:** Marcin Abram `[一作]` (University of Southern California), Marcin Abram `[通讯]` (University of Southern California)

**通讯引用:** 169 | [OpenAlex ID](https://openalex.org/A5000972553)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

在不同能力级别的Anthropic模型上，使用少量无关的文化符号数字做k-shot示例，研究推理时语义污染对下游任务的影响。

**💡 创新点**

揭示推理时语义污染是能力门控的，区分结构性污染与语义性污染两种机制，并量化其对输出分布的细微影响。

**🔧 技术方法**

使用少样本提示（k‑shot）技术、基于正则表达式的实体抽取与分类、统计学差异检验等技术。

**📊 数据集**

实验使用自定义的文化符号数字集合（如极端主义编码、急救号码等），并在100次随机顺序下进行测试。

**📈 对比分析**

通过与空示例基线比较，评估“黑暗”角色出现频率的平均增量，结果显示中高端模型出现显著上升，低端模型无显著变化。

**⚠️ 局限性**

局限包括仅测试三款Anthropic模型、未探究更广泛模型能力范围、对安全系统影响的外推有限。

---

## 765. 15 Years of Augmented Human(s) Research: Where Do We Stand?

**arXiv ID:** 2604.03715 | [PDF](https://arxiv.org/pdf/2604.03715v1)

**作者:** Steeven Villa `[一作]` (LMU Munich), Abdallah El Ali `[通讯]` (Centrum Wiskunde & Informatica)

**通讯引用:** 1598 | [OpenAlex ID](https://openalex.org/A5030623043)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

对 Augmented Human(s) 会议 2010‑2025 年共 787 篇论文进行科学计量分析，研究出版量、地理分布、作者产出、主题演化、内部/外部引用网络及其影响流向。

**💡 创新点**

首次系统揭示该会议的双峰论文量、核心主题（触觉、可穿戴交互、视线追踪、具身交互、运动）随时间的演化，发现 CHI 与 UIST 与 AH(s) 的双向引用关系，指出日本 HCI 社区在贡献中的主导作用，并对会议议题定义与研究范围进行反思。

**🔧 技术方法**

使用 bibliometric 计量、网络分析、BERTopic 主题建模（MiniLM‑L6、UMAP、HDBSCAN、MMR）以及 Sankey 图等可视化技术。

**📊 数据集**

基于 ACM Digital Library 提取的 2010‑2025 年 Augmented Human / Augmented Humans 会议论文元数据、Crossref 与 Semantic Scholar 产生的引用计数与引用网络，构建 787 篇论文的全文与引用图谱。

**📈 对比分析**

通过内部引用与外部引用计数比较、主题频率时间序列、顶级引用论文排名、以及引用流向 Sankey 图呈现会议与外部会议（CHI、UIST、IEEE VR 等）的互相影响；结果显示 Haptics 主题持续增长，Vision & Eye Tracking 下降，CHI 为最大外部引用来源，会议对外影响主要分布在“Other”类别，整体表现为对会议演化与影响力的定量量化。

**⚠️ 局限性**

仅分析会议论文导致幸存者偏差，未覆盖 CHI、UIST 等相关领域的 HA 论文，未包含 Augmented Human Research 期刊与转人类主义文献，方法对词汇聚类依赖，可能忽略语义细微差别；数据仅至 2025 年，缺乏纵向更长周期的对比。

---

## 766. XSeg: A Large-scale X-ray Contraband Segmentation Benchmark For Real-World Security Screening

**arXiv ID:** 2604.03706 | [PDF](https://arxiv.org/pdf/2604.03706v1)

**作者:** Hongxia Gao `[一作]` (Xi'an Jiaotong University), Qianyun Liu `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了规模最大、类别最全的X射线违禁品分割数据集XSeg，并提出了针对X射线分割的APSAM模型

**💡 创新点**

创新点在于引入能量感知编码器(EAE)和自适应点生成器(APG)，显著提升对重叠、遮挡物体的定位和分割精度

**🔧 技术方法**

主要技术包括Segment Anything Model（SAM）微调、能量感知编码器、K-means聚类生成辅助点、ViT-L/14视觉编码器与Transformer解码器

**📊 数据集**

使用了XSeg数据集（98,644张图，295,932实例掩码，30类），并在PIDray、PIXray等公开数据集上验证跨域性能

**📈 对比分析**

在XSeg测试集上与多种Segmentation基线（PSPNet、DeepLabV3+、Mask2Former、SAM、SAMUS等）对比，APSAM取得mIoU 72.83%、Dice 82.31%，比SAM提升约5%

**⚠️ 局限性**

局限性主要是模型与数据集仅针对X射线域，跨模态推广能力有限，且仍受限于图像分辨率与扫描设备的多样性

---

## 767. Extracting and Steering Emotion Representations in Small Language Models: A Methodological Comparison

**arXiv ID:** 2604.04064 | [PDF](https://arxiv.org/pdf/2604.04064v1)

**作者:** Jihoon Jeong `[一作]` `[通讯]`, Jihoon Jeong

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对小型语言模型（100M–10B参数）进行情绪向量提取方法的系统比较，评估9个模型、两种提取方式，并通过情绪推导验证其因果性与行为影响；

**💡 创新点**

首次揭示生成式提取在小型模型中明显优于理解式，情绪向量集中在中层层次，发现跨语言情绪激活与不同架构下的安全隐患；

**🔧 技术方法**

利用TransformerLens与Neural‑MRI进行激活提取与推导，结合情绪向量生成与理解式提取、情绪向量推导、外部情绪分类器评估以及困惑度和熵化分析；

**📊 数据集**

采用Anthropic的171种情绪概念中挑选20种情绪，使用故事生成（10条）与情绪文本片段（3条）以及中性文本，配合外部情绪分类器进行验证；

**📈 对比分析**

通过平均余弦相似度、Mann‑Whitney检验、Cohen's d、激活增量、情绪转换与困惑度比等指标比较提取方法；结果显示生成式提取优于理解式（p=0.007，Cohen's d≈-107.5），情绪向量在中层最佳，RLHF提升生成式质量，情绪推导在124M–3B模型中均能显著改变行为；

**⚠️ 局限性**

实验仅覆盖到3B参数且未处理量化模型，情绪向量仅包含20种概念，跨语言发现基于单一多语言模型且需进一步验证，评估主要使用确定性生成，缺乏细粒度自动化质量评估。

---

## 768. ANX: Protocol-First Design for AI Agent Interaction with a Supporting 3EX Decoupled Architecture

**arXiv ID:** 2604.04820 | [PDF](https://arxiv.org/pdf/2604.04820v1)

**作者:** Xu Mingze `[一作]` `[通讯]` (Hangzhou Ziyou Data Technology Co., Ltd.), Xu Mingze (Hangzhou Ziyou Data Technology Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 ANX 协议与 3EX 架构，用于 AI 代理的原生交互；

**💡 创新点**

引入 agent‑native 设计的 ANX Markup/CLI，支持双向人机交互、按需轻量化应用、UI‑to‑Core 敏感数据隔离与人机确认，以及 3EX 解耦层实现任务表达、工具发现与执行三层分离；

**🔧 技术方法**

采用结构化标记语言、轻量化 CLI 命令、基于语义向量检索的 ANXHub、3EX Expression‑Exchange‑Execution 三层架构，结合 Qwen3.5‑plus / GPT‑4o LLM 进行实验；

**📊 数据集**

使用自建的 10 字段工作账号注册表单作为实验数据集，未采用公开数据集；

**📈 对比分析**

在同一表单任务上与 GUI 自动化和 MCP 技能做 30 次实验比较 token 消耗、执行时间与准确率，ANX 在 token 上减少约 47–66% 以上，执行时间缩短约 58% 以上；

**⚠️ 局限性**

实验仅覆盖单一表单任务，未验证大规模多步骤 SOP 与多代理协作；安全评估仍处于初步阶段，缺乏对 UI 欺骗等威胁的系统化研究。

---

## 769. BiTDiff: Fine-Grained 3D Conducting Motion Generation via BiMamba-Transformer Diffusion

**arXiv ID:** 2604.04395 | [PDF](https://arxiv.org/pdf/2604.04395v1)

**作者:** Tianzhi Jia `[一作]` (Institute of Information Science, Beijing Jiaotong University), Yao Zhao `[通讯]` (Institute of Information Science, Beijing Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一套完整的3D指挥动作生成框架BiTDiff，并构建了首个大规模精细化SMPL-X指挥动作数据集CM-Data

**💡 创新点**

1）BiMamba-Transformer混合架构实现高效长序列建模；2）差分生成策略结合手/身体分解的前向运动学约束；3）无监督的联合级别动作编辑方法

**🔧 技术方法**

Mamba、Transformer、Diffusion（DDPM）、前向运动学（FK）约束、辅助速度损失、无条件引导（CFG）、基于掩码的编辑技术

**📊 数据集**

CM-Data：约10小时、1,500段、包含指挥、面部、手部的高精度SMPL-X 3D动作；同时使用多种基线模型在该数据集上进行评测

**📈 对比分析**

在CM-Data上与2D指挥、3D舞蹈、3D手势生成方法对比；BiTDiff在FID、DIV、BAS、生成速度等指标均优于所有基线，显示出更高的生成质量和更快的推理速度

**⚠️ 局限性**

主要局限：对音乐特征提取依赖固定的Librosa特征；缺乏文本或更丰富的交互式控制；模型在极长序列或实时场景下的鲁棒性尚待进一步验证

---

## 770. SwEYEpinch: Exploring Intuitive, Efficient Text Entry for Extended Reality via Eye and Hand Tracking

**arXiv ID:** 2604.03520 | [PDF](https://arxiv.org/pdf/2604.03520v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 771. Measuring Human Preferences in RLHF is a Social Science Problem

**arXiv ID:** 2604.03238 | [PDF](https://arxiv.org/pdf/2604.03238v1)

**作者:** Bijean Ghafouri `[一作]` (University of Southern California), Emilio Ferrara `[通讯]` (University of Southern California)

**通讯引用:** 19077 | [OpenAlex ID](https://openalex.org/A5078699564)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了将行为科学中的测量有效性框架引入RLHF的方法，构建了一套四类注释响应分类体系，并在PRISM和PluriHarms数据集上实施一致性诊断实验，揭示了大量非偏好、构造偏好和测量误差。

**💡 创新点**

创新点在于将测量有效性（非偏好、构造偏好、测量非一致性）作为RLHF流程的先决条件，引入一致性（时序、一致、顺序、跨项）诊断方法，系统阐述了如何在RLHF中区分真正的价值信号与噪声，从而为后续奖励建模与多元价值处理提供理论与方法支撑。

**🔧 技术方法**

技术手段包括：
- 语义相似度阈值识别重复或等价问答；
- 测试–重测可靠性（Temporal Consistency）；
- 语义框架一致性检验（Framing Consistency）；
- 顺序一致性检验（Order Consistency）；
- 跨项一致性检验（Cross‑Item Consistency）；
- 差异项功能分析与因子分析辅助识别测量非一致性；
- 统计阈值与分类规则的实验验证。

**📊 数据集**

使用的数据集：
- PRISM：包含多次重复评估的1–100分偏好评分数据；
- PluriHarms：对有害性进行0–100分评估，且含语义相似但表述不同的问句。

**📈 对比分析**

在实验中未直接比较RLHF模型性能，而是通过一致性诊断量化偏好不稳定性，进一步分析其对聚合平均分、方向偏差及小样本聚合稳定性的影响。结果显示：
- 约19–26%的注释者存在大幅度不一致；
- 非一致注释者往往给出较低的有害性评分；
- 过滤高不一致注释者可显著提高聚合评分，导致约18.6%的提示的有害性判断发生翻转。

**⚠️ 局限性**

限制：
- 诊断流程需要重复样本、框架变体及额外注释者问卷，成本与工时较高；
- 目前仅在两个数据集上验证，缺乏跨任务与跨文化的普适性评估；
- 对不同阈值设定与诊断标准的系统化校准尚未完成；
- 诊断方法对测量非一致性（解释偏差）仍需更精细的工具；
- 未能在实际RLHF训练流水线中验证诊断与奖励模型性能提升的因果关系。

---

## 772. TinyNina: A Resource-Efficient Edge-AI Framework for Sustainable Air Quality Monitoring via Intra-Image Satellite Super-Resolution

**arXiv ID:** 2604.04445 | [PDF](https://arxiv.org/pdf/2604.04445v1)

**作者:** Prasanjit Dey `[一作]` (ADAPT Research Centre), Soumyabrata Dev `[通讯]` (School of Computer Science and Statistics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种名为TinyNina的超轻量级Edge‑AI框架，用于对Sentinel‑2多光谱图像进行内图学习式超分辨率，以实现高精度NO₂监测。

**💡 创新点**

创新点在于引入波段特异性注意门、深度可分离卷积和多尺度残差上采样的任务感知架构，并利用Sentinel‑2内部多光谱层级实现无外部高分辨率数据的自监督学习。

**🔧 技术方法**

使用Spectral Attention、Depthwise Separable Convolutions、PixelShuffle上采样以及ResNet50回归模型，并结合多尺度残差网络和时序嵌入。

**📊 数据集**

数据集为27个美国西海岸EPA监测站与Sentinel‑2 Level‑2A影像匹配的3,276个时空对，包含12个波段的10/20/60米分辨率。

**📈 对比分析**

与EDSR、RCAN等高参数基线对比，TinyNina在Channel SR策略下MAE为7.4 µg/m³，MSE为97 µg/m³，显著低于EDSR（MAE8.2）和RCAN（MAE7.8），并实现了95%参数压缩和47倍推理速度提升。

**⚠️ 局限性**

主要局限包括对云遮挡、短时排放突发事件以及气象扰动的鲁棒性不足，以及对不同地区域移位可能产生的迁移学习误差。

---

## 773. When Do Hallucinations Arise? A Graph Perspective on the Evolution of Path Reuse and Path Compression

**arXiv ID:** 2604.03557 | [PDF](https://arxiv.org/pdf/2604.03557v1)

**作者:** Xinnan Dai `[一作]` (Michigan State University), Jiliang Tang `[通讯]` (Michigan State University)

**通讯引用:** 25707 | [OpenAlex ID](https://openalex.org/A5040639891)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文从图结构视角出发，对解码器型Transformer的推理幻觉进行机理分析，提出推理过程可视为在潜在图上的搜索，区分了基于上下文的内部推理与基于记忆的外部推理，并揭示了两种幻觉机制：Path Reuse（早期训练阶段记忆路径被优先使用）和Path Compression（后期训练阶段多步路径被压缩为捷径）。

**💡 创新点**

创新点在于将LLM的推理幻觉与图结构搜索统一框架关联，并通过实验在合成图（ER、SBM）上系统量化两种机制的出现时机与表现，进一步将其与已有现象（反转诅咒、推理分布偏差）关联。

**🔧 技术方法**

技术主要包括：基于图的训练样本生成（子图采样、最短路径枚举）、Transformer（LLaMA-2、Qwen-2、Mixtral）从头训练、对比实验（不同层数、不同图结构）、fine‑tuning（SFT与PPO）以及对路径压缩率与错误类型的统计分析。

**📊 数据集**

数据集主要是人工构造的合成图数据（ER图N=10、SBM图N=1000/500，社区数多样），以及将图路径转化为自然语言故事的“语言序列”数据，作为预训练与微调的训练样本。

**📈 对比分析**

实验结果显示：1）Path Reuse在训练早期导致本地上下文不一致，Global/Exist准确率高而Local显著低；2）Path Compression在训练后期导致多步路径被压缩为错误单步边，准确率随训练迭代下降；3）SFT恢复性能不稳定，PPO在不同跳数上均能更稳定提升准确率；总体而言，虽然预训练损失下降，但未压缩路径比例在后期未恢复，说明幻觉机制不随常规训练收敛而消失。

**⚠️ 局限性**

局限性包括：仅在合成图环境下验证，未在大规模真实知识图或开放域文本中系统评估；模型架构与训练设置对幻觉机制的影响仍不完全明确；未提出针对性训练目标或正则化方法来根本消除幻觉，仅给出机理解释；结果对多模态或跨领域推理的推广性有限。

---

## 774. Failure of the strong feasible disjunction property

**arXiv ID:** 2604.04830 | [PDF](https://arxiv.org/pdf/2604.04830v1)

**作者:** Jan Krajicek `[一作]` `[通讯]` (Faculty of Mathematics and Physics Charles University), Jan Krajicek (Faculty of Mathematics and Physics Charles University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文证明在两条纯计算复杂性假设下，强可行析取性质在任何足够强的命题证明系统中必然失败。

**💡 创新点**

通过结合半自然证明与“demi-bit”假设，构造了一种新的生成器，显示强可行析取性质不可满足。

**🔧 技术方法**

使用了生成器技术、Nisan‑Wigderson伪随机生成器、半自然证明与证据系统的模拟，以及极限硬度的概念。

**📊 数据集**

未使用实验数据集，完全在理论框架下证明。

**📈 对比分析**

比较方法为理论证明，无实证性能指标。

**⚠️ 局限性**

局限在于依赖两个尚未证明的复杂性假设（E‑NP 与 demi‑bit），若假设不成立则结论失效。

---

## 775. BWTA: Accurate and Efficient Binarized Transformer by Algorithm-Hardware Co-design

**arXiv ID:** 2604.03957 | [PDF](https://arxiv.org/pdf/2604.03957v1)

**作者:** Yifu Ding `[一作]` (Beihang University), Jiwen Lu `[通讯]` (Tsinghua University)

**通讯引用:** 29105 | [OpenAlex ID](https://openalex.org/A5100460385)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Binary Weights & Ternary Activations (BWTA) 量化方案，并通过 Smooth Multi‑Stage Quantization 训练稳定的极低位 Transformer；

**💡 创新点**

创新点：① 通过保持激活的零点并采用三值量化缓解 binarization 的零点失真；② 采用级别递减与幅度对齐的多阶段训练；③ 开发端到端 GPU MatMul CUDA kernel，实现 16‑24× 速度提升；

**🔧 技术方法**

技术：二值化权重、三值激活、级别递减量化、幅度对齐投影、指令级并行位打包、全栈 CUDA kernel；

**📊 数据集**

数据集：BERT 训练/评估使用 GLUE（除 WNLI），LLM 评估使用 WikiText2、C4 以及 CommonsenseQA；

**📈 对比分析**

对比方法：BERT 量化基线（Q2BERT、TernaryBERT 等）、LLM 量化基线（RTN、GPTQ、PB‑LLM、BiLLM、Bitnet 等），实验表明 BWTA 在 GLUE 上平均仅落后 3.5% 甚至可逼近 FP16，LLM perplexity 与准确率相当；在 GPU 上 kernel 级 16‑24× 加速，prefill 及 decode 端到端可达 216‑330 tokens/s，显著优于 4‑bit 权重/激活方案；

**⚠️ 局限性**

局限性：仅针对 Transformer 结构，需手动集成自定义 kernel；多阶段训练耗时较长；低位量化仍受模型结构与数据分布限制，极低位仍可能在更大规模或更复杂任务上出现精度瓶颈。

---

## 776. Learning Additively Compositional Latent Actions for Embodied AI

**arXiv ID:** 2604.03340 | [PDF](https://arxiv.org/pdf/2604.03340v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 777. Systematic Review of Academic Procrastination Interventions in Computing Higher Education

**arXiv ID:** 2604.03248 | [PDF](https://arxiv.org/pdf/2604.03248v1)

**作者:** Daniel Cheng `[一作]` (University of Toronto), Jonathan Calver `[通讯]` (University of Toronto)

**通讯引用:** 106 | [OpenAlex ID](https://openalex.org/A5002017873)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对过去十年在计算机高等教育中针对学术拖延的干预研究进行系统综述，归纳干预类型、设计特征及其对学生行为和课程成绩的影响。

**💡 创新点**

首次将多类干预（如期限设定、自动评分、游戏化、提醒、心理与社会干预）在计算机教育环境下的实证效果进行跨类别综合，发现时间结构与支持性设计是提升早期参与和分布式学习的核心驱动因素，并指出学生主观感受与客观行为结果的常见不一致。

**🔧 技术方法**

采用PRISMA 2020指南的系统文献检索与筛选流程，并用迭代共识式定性合成法对19篇实验研究进行编码与归类。

**📊 数据集**

基于六大数据库（ACM, IEEE, ScienceDirect, Scopus, SpringerLink, Web of Science）的检索，最终纳入19篇研究，涵盖约689条记录。

**📈 对比分析**

对比不同干预类别与设计变量（如支持性vs惩罚性、期限严格度、反馈频率、游戏化机制），通过案例聚合显示：结构化、支持性干预能显著提前提交、分散工作并提升成绩，尤其在长周期、多步骤任务中效果更显著；相对而言，仅靠激励或惩罚的干预效果不稳定。

**⚠️ 局限性**

局限包括：仅纳入英文论文，检索关键词与数据库范围可能漏检相关研究；研究归纳过程依赖研究者主观判断；缺乏对非计算机领域有效干预的跨领域比较。

---

## 778. Convolutional Neural Network and Adversarial Autoencoder in EEG images classification

**arXiv ID:** 2604.04313 | [PDF](https://arxiv.org/pdf/2604.04313v1)

**作者:** Albert Nasybullin `[一作]` (Innopolis University), Semen Kurkin `[通讯]` (Innopolis University)

**通讯引用:** 2389 | [OpenAlex ID](https://openalex.org/A5025595136)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用CNN与半监督对抗自编码器（AAE）对EEG脑电顶图进行左右手运动分类

**💡 创新点**

将EEG信号转换为2D顶图并结合卷积神经网络与自编码器生成器-判别器架构进行分类，同时提出多窗口抽样与基线归一化策略

**🔧 技术方法**

卷积神经网络（CNN）、生成对抗网络（GAN）中的对抗自编码器（AAE）、信号预处理与ICA去噪、FieldTrip工具箱生成EEG顶图

**📊 数据集**

基于15名受试者、每人20个左右手动作试验共939张9-11Hz（Mu波段）顶图图像的数据集

**📈 对比分析**

CNN在10个epoch内达93.75%准确率，AAE训练400 epoch后最高68%准确率，CNN显著优于AAE，验证了图像化EEG分析的有效性

**⚠️ 局限性**

AAE训练结果不稳定、准确率低且受样本量限制，数据集规模有限，需进一步扩充数据和改进模型以提升稳定性

---

## 779. CTD-Diff: Cooperative Time-Division Diffusion for Multi-User Semantic Communication Systems

**arXiv ID:** 2604.04057 | [PDF](https://arxiv.org/pdf/2604.04057v1)

**作者:** Chengyang Liang `[一作]` (Macau University of Science and Technology), Dong Li `[通讯]` (Macau University of Science and Technology)

**通讯引用:** 6468 | [OpenAlex ID](https://openalex.org/A5100407433)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向多用户语义通信的协作时分扩散框架CTD‑Diff，结合TDMA互听、直接信号聚合与条件扩散重建，并设计了混合噪声训练策略。

**💡 创新点**

创新点在于：① 将物理信道噪声与扩散噪声统一纳入前向扩散过程；② 在TDMA时隙内让空闲用户做语义中继，实现多用户协作；③ 通过语义嵌入实现条件扩散，提升语义一致性；④ 混合噪声训练将真实信道噪声与人工扩散噪声融合，增强泛化。

**🔧 技术方法**

技术包括扩散概率模型（DDPM）、条件UNet与注意力机制、预均衡和TDMA协作、直接聚合、混合噪声训练、ResNet‑50语义编码器、Rayleigh与AWGN信道仿真等。

**📊 数据集**

使用了三组图像数据集：CIFAR‑100（32×32），STL‑10（96×96），ImageNet‑256（256×256）。

**📈 对比分析**

与JPEG+LDPC、MU MVJSCC、SIAC、CDDM等基线在AWGN与Rayleigh信道下，采用PSNR与MS‑SSIM评估。CTD‑Diff在所有数据集与SNR范围内平均提升2–8 dB（PSNR）和10–16 %（MS‑SSIM），尤其在低SNR和严重衰落环境表现最优。

**⚠️ 局限性**

局限性：需要前置均衡和信道估计，协作带来额外时延与同步开销；对极低SNR或极高速移动时仍有限；目前仅验证图像场景，对文本、语音等多模态或非固定TDMA安排的适用性尚未深入；模型规模和推理成本相对较高。

---

## 780. Shortest-Path FFT: Optimal SIMD Instruction Scheduling via Graph Search

**arXiv ID:** 2604.04311 | [PDF](https://arxiv.org/pdf/2604.04311v1)

**作者:** Mohamed Amine Bergach `[一作]` `[通讯]` (Illumina), Mohamed Amine Bergach (Illumina)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在 Apple M1 NEON 上，提出了一种基于上下文感知的最短路径框架，用于搜索最快的 FFT 实现。

**💡 创新点**

创新点是通过在图中扩展节点来编码前一操作类型，从而为边权引入缓存相关性，并将融合寄存器块视为可搜索的边；这打破了传统 FFTW 依赖的最优子结构假设。

**🔧 技术方法**

采用经验测量的边权、上下文感知图扩展、Dijkstra 搜索、混合基数分解与融合寄存器块设计等技术。

**📊 数据集**

使用 N = 1024 的复浮点32 位 FFT 作为基准数据集，在 Apple M1 主核心上进行测量。

**📈 对比分析**

将新方法与纯基数‑2/4/8、FFT FFTW‑风格的上下文无关 Dijkstra 以及传统实现进行对比；结果显示 29.8 GFLOPS，较上下文无关方案提升 34%，相较纯基数‑2 提升 5.2 倍。

**⚠️ 局限性**

局限性包括只考虑一阶上下文（前一操作的缓存状态），更高阶缓存效应未被建模；测量成本随边类型增多而增长，且需要在每个新架构上重新测量边权。

---

## 781. A Review of Multiscale Thermal Modeling in Heterogeneous 3D ICs

**arXiv ID:** 2604.03290 | [PDF](https://arxiv.org/pdf/2604.03290v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 782. Risk-Constrained Belief-Space Optimization for Safe Control under Latent Uncertainty

**arXiv ID:** 2604.03868 | [PDF](https://arxiv.org/pdf/2604.03868v1)

**作者:** Clinton Enwerem `[一作]` (University of Maryland), Calin Belta `[通讯]` (University of Maryland)

**通讯引用:** 11923 | [OpenAlex ID](https://openalex.org/A5086742095)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种在存在潜在不确定性的情况下，利用贝叶斯推断和CVaR约束进行安全控制的模型预测路径积分（MPPI）框架；

**💡 创新点**

创新点在于将置信区间风险度量（CVaR）同时应用于性能目标和轨迹安全约束，并在贝叶斯状态空间中直接规划，从而在保证安全尾部风险的同时提升任务成功率；

**🔧 技术方法**

核心技术包括粒子滤波器用于在线更新潜在参数后验，CVaR风险度量的经验估计，采样式MPPI轨迹评估和重要性加权，及贝叶斯贝塞尔式运动规划；

**📊 数据集**

实验数据集为基于MuJoCo的视觉引导精细插入任务，涉及抓取物体、狭窄插槽和已存在物体的环境；

**📈 对比分析**

与仅使用概率违约约束的基线相比，本文方法在高风险厌恶配置下成功率提升至82%，且完全消除接触冲击，表现出更稳健的安全性和更高的完成率；

**⚠️ 局限性**

局限性包括对刚性抓取的假设、简化的接触模型、粒子滤波器对观测噪声独立性的假设以及对时间变化潜在参数的处理不充分。

---

## 783. ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos

**arXiv ID:** 2604.03819 | [PDF](https://arxiv.org/pdf/2604.03819v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 784. Rank, Don't Generate: Statement-level Ranking for Explainable Recommendation

**arXiv ID:** 2604.03724 | [PDF](https://arxiv.org/pdf/2604.03724v1)

**作者:** Ben Kabongo `[一作]` (Sorbonne University), Vincent Guigue `[通讯]` (AgroParisTech)

**通讯引用:** 924 | [OpenAlex ID](https://openalex.org/A5044389669)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出将可解释推荐转化为声明级排序问题，并构建StaR基准；

**💡 创新点**

① 通过LLM两阶段提取与验证保证声明可解释性与原子性；② 大规模语义聚类合并同义句；③ 设立全局级与项目级两种评估，揭示个性化缺口；

**🔧 技术方法**

使用LLM（Qwen系列）进行提取与验证，dense embedding + Faiss ANN+交叉编码器筛选构建语义聚类；实验采用BPER+、ExpGCN模型及流行度基线；

**📊 数据集**

基准数据来自Amazon Reviews 2014四个品类（Toys、Clothes、Beauty、Sports）；

**📈 对比分析**

通过Precision@k/Recall@k/NDCG@k进行评估，全球级实验中ExpGCN略优；项目级实验中UserPop优于ExpGCN，显现个性化不足；

**⚠️ 局限性**

限制：提取与聚类仍可能产生噪声；基准受评论数据偏差影响；缺乏多样性、分级相关性等更细粒度评估指标。

---

## 785. Measuring LLM Trust Allocation Across Conflicting Software Artifacts

**arXiv ID:** 2604.03447 | [PDF](https://arxiv.org/pdf/2604.03447v1)

**作者:** Noshin Ulfat `[一作]` (University of Texas at Dallas), Soneya Binta Hossain `[通讯]` (University of Texas at Dallas)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为TRACE的框架，用来显式地让大型语言模型（LLM）在软件工程任务中对不同的程序工件（Javadoc、方法签名、实现代码、测试前置语句）进行信任推理，并通过结构化的评估追踪来判断工件质量、冲突与优先级；

**💡 创新点**

创新点包括：1）将工件级信任推理作为首要预测目标，而非仅关注下游任务结果；2）构造了一个对齐的、包含六种干扰类型且按严重度分级的Java方法工件基准；3）通过对多模型的追踪数据揭示了模型在文档与代码冲突检测上的系统性盲点与置信度失衡，提供了可解释的性能差异分析；

**🔧 技术方法**

主要技术包括：基于Prompt的结构化JSON输出，跨模型统一的系统/用户提示，盲干扰协议，自动化解析与验证，三层冲突信号（对比、已识别冲突、完整不一致），以及使用句子嵌入计算描述相似度与置信度校准；

**📊 数据集**

使用了从OE-25 25个Java项目中筛选出的456个高质量方法工件，并生成7个版本（1清洁+6种干扰），共计3,192个实例；每个实例通过七个LLM模型（Claude Opus、Sonnet、Haiku、GPT‑5.2、GPT‑4o、DeepSeek‑V3.2、Grok‑4）生成22,339条有效追踪；

**📈 对比分析**

对模型的比较采用了四个维度：工件质量评分的敏感度（Delta from base）、文档-实现一致性检测率、描述相似度（语义cosine）、置信度校准以及源优先级相关性。实验表明：①模型在文档缺失或错误时的评分下降更明显；②在文档+实现冲突时检测率可达90%以上，但仅实现偏移时检测率下降20–40个百分点；③只有DeepSeek‑V3.2具备显著的置信度校准；④整体来看Sonnet和DS‑V3.2在细粒度实现错误检测与描述准确度上最为稳健；

**⚠️ 局限性**

主要局限包括：①基准仅覆盖Java方法，难以推广到其他语言或更大规模的工件；②干扰样本是自动生成的，可能未覆盖真实开发中出现的复杂、分散冲突；③置信度分数在大多数模型中缺乏可用性，需后期再校准；④缺乏对模型内部机制的解释，无法彻底定位为何存在实现偏移的盲点；

---

## 786. Context Matters: Evaluating Context Strategies for Automated ADR Generation Using LLMs

**arXiv ID:** 2604.03826 | [PDF](https://arxiv.org/pdf/2604.03826v1)

**作者:** Aviral Gupta `[一作]` (Software Engineering Research Centre), Karthik Vaidhyanathan `[通讯]` (Software Engineering Research Centre)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过系统实验研究了在使用大型语言模型（LLM）自动生成架构决策记录（ADR）时，不同上下文提供策略对生成质量的影响。

**💡 创新点**

创新点包括：①对五种上下文策略（无上下文、全历史、First‑K、Last‑K、RAFG）在ADR生成中的效果进行大规模对比；②发现仅使用最近3–5条历史记录即可获得与完整历史相当的生成质量，并与复杂检索方法相匹配；③为工具构建提供实用的默认策略与检索回退方案。

**🔧 技术方法**

采用的技术包括多种LLM（Gemini‑2.5‑Pro、Qwen3‑235B、Gemma‑3‑4B、GLM‑4.6）、Retrieval‑Augmented Few‑shot Generation（RAFG）以及多指标自动评估（BERTScore、BLEU、ROUGE、METEOR）与人工评审。

**📊 数据集**

使用的数据集为750个开源仓库中的约4,500条经过人工验证的序列化ADR，来源于Buchgeher等的MSR研究并进一步清洗。

**📈 对比分析**

比较方法为：以ADR标题为输入，分别在不同上下文策略下调用各LLM生成ADR，然后用BERTScore、BLEU、ROUGE、METEOR等指标与人工撰写的参考ADR进行匹配。实验结果表明：无上下文基线性能最差；Last‑K（3–5条）在大多数模型上与全历史相当，优于First‑K；RAFG与Last‑K总体相当，但在跨切场景中略优；小模型（Gemma‑3‑4B）在合适上下文下可与大模型相匹配。

**⚠️ 局限性**

局限性包括：①可能存在训练集重叠导致性能上浮；②ADR引用外部资源导致评估低分；③仅使用标题作为提示，标题粒度不统一；④未考虑代码或设计图等多模态上下文；⑤评估主要基于文本相似度，未衡量实际决策有效性。

---

## 787. ARES OS 2.0: An Orchestration Software Suite for Autonomous Experimentation Systems and Self-Driving Labs

**arXiv ID:** 2604.03440 | [PDF](https://arxiv.org/pdf/2604.03440v1)

**作者:** Arthur W. N. Sloan `[一作]` (Air Force Research Laboratory), Benji Maruyama `[通讯]` (Air Force Research Laboratory)

**通讯引用:** 6760 | [OpenAlex ID](https://openalex.org/A5102919383)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

开发了名为ARES OS的开源软件框架，旨在通过服务导向架构、Blazor GUI和Python PyAres库实现实验室自动化和闭环自驱实验，降低研究者对软件集成的技术门槛。

**💡 创新点**

提供研究者友好的低/无代码用户体验，支持跨语言模块（C#、Python、JavaScript等），并统一管理硬件、分析、规划与数据，实现多学科实验流程的可扩展抽象。

**🔧 技术方法**

采用C# + ASP.NET Core（SOLID设计）、gRPC + protobuf通信、Blazor UI、SQLite/SQL Server/PostgreSQL数据库、Python PyAres库及Launcher安装工具，构建可语言无关的服务导向架构。

**📊 数据集**

在碳纳米管合成和熔融沉积建模3D打印等实验中使用实际实验数据；未公开使用标准公开数据集。

**📈 对比分析**

与MadSci、ChemOS2.0、Minerva-OS等开源SDL平台比较，ARES OS在实验室案例中表现出更快的研究进展、更低的实验变异性和更少的实验次数，整体性能优于传统手动实验流程。

**⚠️ 局限性**

仍受限于硬件兼容性和模块配置复杂度，极端复杂实验流程的支持有限；对非Python语言用户仍存在学习门槛；缺乏商业化支持与完整的技术文档。

---

## 788. nascTime: A Full-Stack 5G-TSN Bridge Simulation Framework with SDAP-Based QoS Mapping and IEEE 802.1AS Transparent Clock

**arXiv ID:** 2604.04616 | [PDF](https://arxiv.org/pdf/2604.04616v1)

**作者:** Mohamed Seliem `[一作]` (National Universities of Ireland), Dirk Pesch `[通讯]` (National Universities of Ireland)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了完整的5G-TSN桥接仿真框架nascTime

**💡 创新点**

创新点在于实现SDAP层完整QoS映射、测量IEEE 802.1AS透明时钟居住时间，并支持多端点双向流量验证

**🔧 技术方法**

使用OMNeT++6.3、INET4.6、Simu5G v1.4.1、L2-in-GTP-U gPTP、SDAP/DRB、Streaming PHY等技术栈

**📊 数据集**

采用三端点工厂场景数据集，包括高优先级/最佳努力流量与gPTP同步帧

**📈 对比分析**

与现有框架对比，验证了99.9%高优先级交付、0丢包、居住时间≈2.5ms、低方差，性能显著优于前者

**⚠️ 局限性**

局限于理想链路（无衰落/阴影）且未验证大规模终端或真实工厂通道模型

---

## 789. Environment-Aware Near-Field Channel Estimation Leveraging CKM and ISAC

**arXiv ID:** 2604.04031 | [PDF](https://arxiv.org/pdf/2604.04031v1)

**作者:** Yuan Guo `[一作]` (Chinese University of Hong Kong), Jie Xu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 11073 | [OpenAlex ID](https://openalex.org/A5063914161)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

本文提出基于虚拟对象图（VOM）和集成感知通信的极大天线阵列（ELAA）近场下行信道估计框架。

**💡 创新点**

创新点在于将CKM中的静态环境信息转化为可复用的虚拟对象库，并结合单波回波抑制动态目标信息，从而实现高效的结构化信道估计。

**🔧 技术方法**

采用VOM表示、单波回波静态杂波抑制、奇异值分解提取动态子空间、正则化最小二乘估计以及MRT波束成形等技术。

**📊 数据集**

使用仿真数据集：64 口ULA在2.4 GHz下的射线追踪场景，静态反射体均匀分布于[-7,7]×[5,15] m，动态目标为圆形簇(-1.5,5) m。

**📈 对比分析**

与仅使用VOM、仅使用传统方法对比，实验显示联合VOM与感知方案在短训练符号下NMSE下降超过10 dB，降至接近完美CSI的上界，并显著提高下行速率。

**⚠️ 局限性**

局限在于VOM仅离线构建、未考虑动态环境变化、理想的误码无反馈链路以及仅在仿真环境中验证，缺乏真实场景评估。

---

## 790. When Does Multimodal AI Help? Diagnostic Complementarity of Vision-Language Models and CNNs for Spectrum Management in Satellite-Terrestrial Networks

**arXiv ID:** 2604.03774 | [PDF](https://arxiv.org/pdf/2604.03774v1)

**作者:** Yuanhang Li `[一作]` `[通讯]`, Yuanhang Li

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在卫星-地面网络的光谱热图理解任务中，系统性比较了视觉-语言模型（VLM）与卷积神经网络（CNN）在四个不同粒度任务（场景分类、区域推理、空间定位、语义推理）上的表现，并构建了首个SpectrumQA视觉问答基准数据集。

**💡 创新点**

创新点包括：①提出四级任务诊断框架和SpectrumQA基准；②揭示CNN与VLM在不同任务层面上的互补性，并通过任务类型路由实现39%性能提升；③发现VLM早期隐藏层保留更丰富的空间信息，且在跨场景迁移中表现出更强鲁棒性。

**🔧 技术方法**

使用技术包括：冻结的 Qwen2-VL-7B VLM（零/3-shot 提示与 CoT 推理）、训练的 ResNet-18 CNN、focal loss 训练、视觉问答模板生成、LoRA 细调、交叉熵/ROUGE 等评估指标。

**📊 数据集**

使用数据集为 108K 视觉问答对，来源于三种物理校准的 NTN‑TN 仿真场景（卫星/基站布局、频段分配等），生成光谱热图并标注对应标签。

**📈 对比分析**

在同一热图与标签下对四级任务进行比较，CNN 在 L1–L3 任务上分别达到 72.9%/65.7%/0.552 IoU；VLM 在 L4 任务上取得 F1=0.576，CoT 进一步提升 12.6%；任务路由后复合分数为 0.616，较单一 CNN 提升 39.1%。

**⚠️ 局限性**

局限性包括：仅评估单一 VLM 族（Qwen2-VL-7B），基于仿真数据未验证真实场景；路由为固定规则，缺乏动态实例化；VLM 生成文本缺乏因果可解释性；LoRA 细调会削弱空间信息保留。

---

## 791. Kill Webs by Collaborative & Self-organizing Agents (CSOAs)

**arXiv ID:** 2604.03602 | [PDF](https://arxiv.org/pdf/2604.03602v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 792. TransGP: Task-Conditioned Transformer-Guided Genetic Programming for Multitask Dynamic Flexible Job Shop Scheduling

**arXiv ID:** 2604.03705 | [PDF](https://arxiv.org/pdf/2604.03705v1)

**作者:** Meng Xu `[一作]`, Yew Soon Ong `[通讯]`

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种结合任务条件Transformer指导的遗传程序（TransGP）来进化动态可灵活车间调度的可解释启发式规则。

**💡 创新点**

创新点在于将Transformer训练为语义感知的、任务自适应的变异算子，实现对优秀启发式的分布学习和跨任务知识迁移。

**🔧 技术方法**

使用Transformer（预训练、条件解码）与遗传程序（禁用交叉、基于Transformer的变异）以及模拟器评估。

**📊 数据集**

使用人工生成的10机124作业的动态可灵活车间调度模拟数据，包含3个利用率/机器数组合，共9个任务。

**📈 对比分析**

与传统GP、任务标签GP、人工启发式以及纯Transformer进行对比；TransGP在所有任务上平均目标值显著降低、收敛更快、方差更小。

**⚠️ 局限性**

局限性包括对任务信息的手工编码、对大规模实例的可扩展性未验证、对超参数敏感，以及需大量优秀启发式收集以训练Transformer。

---

## 793. Inside the Scaffold: A Source-Code Taxonomy of Coding Agent Architectures

**arXiv ID:** 2604.03515 | [PDF](https://arxiv.org/pdf/2604.03515v1)

**作者:** Benjamin Rombaut `[一作]` `[通讯]` (Huawei Canada), Benjamin Rombaut (Huawei Canada)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

对13个公开的编码代理（CLI与SWE‑bench等）的底层 scaffold 代码进行源代码级的系统性分析，基于文件路径与行号构建12维度、3层的架构分类法。

**💡 创新点**

首次以源代码为切入点给出可复制的细粒度分类：揭示代理架构呈连续谱而非离散类别；发现循环原语（ReAct、generate‑test‑repair、plan‑execute、multi‑attempt retry、tree‑search）可自由组合；提出跨维度主题（采样 vs 迭代、子代理委托、在线/离线选择等），为后续基于架构的评估与设计提供词汇。

**🔧 技术方法**

质性案例研究方法：在每个代理的固定提交哈希上手工阅读代码，使用结构化模板记录观察、分类与证据；通过文件路径+行号验证；归纳出维度与交叉主题。

**📊 数据集**

13个公开的编码代理（覆盖 CLI、SWE‑bench 评估、基准、IDE 扩展等），每个都在特定 commit hash 进行分析；没有额外数据集。

**📈 对比分析**

未做性能对比；通过对比 12 个维度和跨维度主题来展示架构差异，指出不同维度（如控制循环、工具集、上下文压缩、状态管理、多模型路由等）会影响成本、可靠性和失败模式，但未给出定量性能指标。

**⚠️ 局限性**

局限性：仅进行架构税onomy，未评估性能；只分析公开源码，可能遗漏私有实现细节；单作者分析可能带来主观性；仅覆盖 13 个代理，未来新架构的适用性未知；未直接量化架构维度与行为/性能的因果关系。

---

## 794. MC-CPO: Mastery-Conditioned Constrained Policy Optimization

**arXiv ID:** 2604.04251 | [PDF](https://arxiv.org/pdf/2604.04251v1)

**作者:** Oluseyi Olukola `[一作]` (University of Southern Mississippi), Nick Rahimi `[通讯]` (University of Southern Mississippi)

**通讯引用:** 263 | [OpenAlex ID](https://openalex.org/A5102764912)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了 MC‑CPO 算法，在教育型强化学习中通过将学习者掌握状态作为结构约束嵌入 CMDP，以消除奖励欺骗；

**💡 创新点**

创新点在于把课程前置知识的可行性条件直接作为结构约束引入，提供可证明的安全保证，并证明后验过滤在奖励-安全平衡上存在安全间隙；

**🔧 技术方法**

使用了两时间尺度的主从对偶优化、结构化动作屏蔽、前沿混合探索、PPO/REINFORCE 等深度强化学习技术以及理论证明框架；

**📊 数据集**

实验仅基于自建的教学模拟环境，包括单步两概念、五概念链、15/25概念的 BKT 模拟，没有使用公开真实数据集；

**📈 对比分析**

与无约束、奖励塑形、后验过滤等基线比较，MC‑CPO 在满足约束的前提下接近最优参与度，同时显著降低 Reward Hacking Severity Index（RHSI），优于后验过滤；

**⚠️ 局限性**

局限性包括仅在模拟环境中验证，未考虑真实学生数据和非可观测性误差，深度函数逼近下的理论保证缺失，预算设置相对基线，可能产生公平性问题。

---

## 795. A Faceted Classification of Authenticator-Centric Authentication Techniques

**arXiv ID:** 2604.03627 | [PDF](https://arxiv.org/pdf/2604.03627v1)

**作者:** Alex R. Mattukat `[一作]` (RWTH Aachen University), Horst Lichter `[通讯]` (RWTH Aachen University)

**通讯引用:** 1192 | [OpenAlex ID](https://openalex.org/A5091346056)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

创建了面向认证技术的面向特征分类方案，并基于该方案构建了一个包含33种认证技术与34种凭证的在线目录。

**💡 创新点**

创新点包括：① 将认证技术与其使用的凭证通过聚合关系建模；② 采用面向特征（faceted）分类法，分为基础面向和常见面向；③ 结合LLM辅助的系统性文献检索与BERTopic语义聚类，快速生成分类框架；④ 构建公开可查询的目录，为实践者和研究者提供一站式参考。

**🔧 技术方法**

使用技术：Microsoft LLM（zero‑shot chain‑of‑thought）进行文献筛选；BERTopic（Sentence‑Bert、HDBSCAN、UMAP）进行语义聚类；面向特征分类设计方法；在线目录实现（基于GitLab Pages）。

**📊 数据集**

数据集：IEEE Xplore数据库检索到1256篇论文，经过LLM筛选得到457篇相关论文，进一步语义聚类提取345篇核心论文，最终筛选出33种认证技术与34种凭证。

**📈 对比分析**

比较方法：对比已有NIST、Chenchev等分类体系，展示本方案在覆盖面和灵活性上的优势；通过人工与LLM的一致率（95%）验证筛选准确性；未进行实验性性能评估，主要侧重分类的结构性与可用性。

**⚠️ 局限性**

局限性：① 仅检索IEEE Xplore，可能低估了非电气工程领域的技术；② 通过排除“protocol”“cryptographic”等词聚焦新颖技术，导致持有型技术偏少；③ 凭证分类仅包含四个面，未覆盖所有可能属性；④ 未实现跨面约束和定量性能指标；⑤ 目录仅基于学术论文，缺乏灰色文献与工业实践案例。

---

## 796. Towards the AI Historian: Agentic Information Extraction from Primary Sources

**arXiv ID:** 2604.03553 | [PDF](https://arxiv.org/pdf/2604.03553v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 797. CPT: Controllable and Editable Design Variations with Language Models

**arXiv ID:** 2604.04380 | [PDF](https://arxiv.org/pdf/2604.04380v1)

**作者:** Karthik Suresh `[一作]` (Adobe), Asim Kadav `[通讯]` (Adobe)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了基于解码器语言模型的创意预训练变压器（CPT）及其专用的创意标记语言（CML），实现可编辑的设计模板变体生成；

**💡 创新点**

创新点在于将设计模板的结构与样式统一编码为线性可序列化的CML，利用掩码序列预测任务让大型语言模型直接生成样式一致、可编辑的完整设计文件，并通过GPT-4o构建的过滤与评分管线保障质量；

**🔧 技术方法**

技术实现包括：7B参数的Mistral‑7B解码器 fine‑tuned with LoRA；基于掩码的序列填充（Fill‑in‑the‑Middle）任务；CML 结构化表示；GPT‑4o 的多模态评估与多样性排序；以及自动化启发式指标；

**📊 数据集**

使用约 220K 份由专业设计师在在线设计平台创建的模板，先转换为 CML 再进行训练；

**📈 对比分析**

通过自动启发式指标、人类评审（90.7% 通过率）和 GPT‑4o 过滤/排序进行评估，CPT 在关联模型下的整体选择率为 58.3%，人类评价满意度达到 90.7%，显著优于无关联基线；

**⚠️ 局限性**

局限性包括：颜色对比度和对齐仍有缺陷、对图像等视觉信号的依赖有限、布局变体生成仍受限于文本‑仅表示、缺乏更细粒度的品牌与关系约束，且需要更完善的评估与强化学习循环。

---

## 798. Rényi Attention Entropy for Patch Pruning

**arXiv ID:** 2604.03803 | [PDF](https://arxiv.org/pdf/2604.03803v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 799. Toward a Sustainable Software Architecture Community: Evaluating ICSA's Environmental Impact

**arXiv ID:** 2604.04096 | [PDF](https://arxiv.org/pdf/2604.04096v1)

**作者:** Mahyar T. Moghaddam `[一作]`, Mikkel Baun Kjærgaard `[通讯]` (Syddansk Universitet)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估ICSA 2025会议的碳足迹，分别量化GenAI推理使用与传统旅行、住宿、场地、餐饮、材料等排放，并给出总排放量与每个来源比例

**💡 创新点**

首次系统审计软件架构研究与会议活动的碳足迹，提供透明基线并提出可持续会议与研究实践建议

**🔧 技术方法**

使用自动文本检测、人工审核结合EcoLogits推理能耗估算，配合交通、住宿、场地能源与餐饮的排放因子进行LCA分析

**📊 数据集**

108 篇ICSA 2025录用论文文本、229 名参会者匿名统计数据（旅行距离、住宿天数、场地与餐饮信息）

**📈 对比分析**

通过量化每项排放与GenAI使用能耗进行对比，发现旅行占 94 % 总排放，GenAI 排放仅 0.5 %，表明传统活动更需优化，估算精度约 ±20 %

**⚠️ 局限性**

估算依赖假设与代理数据导致不确定性 ±20 %；GenAI 使用检测不完全；仅评估 CO₂e，未涵盖水资源等；未考虑间接影响；适用于中等规模会议

---

## 800. OpenRC: An Open-Source Robotic Colonoscopy Framework for Multimodal Data Acquisition and Autonomy Research

**arXiv ID:** 2604.03781 | [PDF](https://arxiv.org/pdf/2604.03781v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 801. CODE-GEN: A Human-in-the-Loop RAG-Based Agentic AI System for Multiple-Choice Question Generation

**arXiv ID:** 2604.03926 | [PDF](https://arxiv.org/pdf/2604.03926v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 802. PRAISE: Prefix-Based Rollout Reuse in Agentic Search Training

**arXiv ID:** 2604.03675 | [PDF](https://arxiv.org/pdf/2604.03675v1)

**作者:** Erhan Zhang `[一作]` (Renmin University of China), Jiaxin Mao `[通讯]` (Renmin University of China)

**通讯引用:** 2463 | [OpenAlex ID](https://openalex.org/A5072119199)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种将搜索轨迹前缀重新利用的框架PRAISE，以提高agentic search的训练效率和信用分配。

**💡 创新点**

核心创新在于利用搜索前缀生成中间答案作为额外训练样本，并通过相邻前缀的分数差来构造细粒度的过程奖励，同时共享同一模型进行策略学习和前缀评估，消除额外奖励模型的需求。

**🔧 技术方法**

采用强化学习（PPO）框架，结合检索增强生成（RAG）、检索器E5以及LLM Qwen2.5，利用可验证的答案评分函数实现奖励计算。

**📊 数据集**

在HotpotQA、2WikiMultihopQA、NQ、Bamboogle和MuSiQue等多跳问答数据集上进行训练与评估。

**📈 对比分析**

相较于传统agentic search方法、仅用重放或过程监督的基线，PRAISE在所有五个基准上都实现了F1和EM的显著提升，尤其在HotpotQA和2Wiki上提升幅度更为明显。

**⚠️ 局限性**

缺点包括对过程奖励权重α的敏感性以及在更大模型或更长搜索轨迹时对计算资源和调参需求的增加。

---

## 803. DéjàVu: A Minimalistic Mechanism for Distributed Plurality Consensus

**arXiv ID:** 2604.03648 | [PDF](https://arxiv.org/pdf/2604.03648v1)

**作者:** Francesco d'Amore `[一作]` (Gran Sasso Science Institute), Emanuele Natale `[通讯]` (INRIA)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文对一种称为DéjàVu的h-多数投票协议进行了理论分析，给出了在存在初始多数偏差时协议达到多方投票一致性的收敛时间上界和下界，并分析了协议所需采样次数与传统h-多数投票的关系

**💡 创新点**

创新点在于提出并使用Poisson竞赛框架与Newton不等式证明概率比值单调性，从而实现对多方投票偏差的放大与保持，并在任意h≥2下给出了收敛时间的上界与下界

**🔧 技术方法**

主要技术包括泊松竞赛模型、一次性样本内部重复事件的概率分析、子马尔可夫性质、Bernstein不等式、负相关性与马尔可夫停止时间分析

**📊 数据集**

本文为理论论文，不依赖于任何外部数据集，分析基于完全图（所有节点互相可达）

**📈 对比分析**

通过与已知的2-选择和3-多数等经典协议在相同偏差下的收敛时间进行比较，结果表明DéjàVu在h较大时收敛时间与2-选择相当，而在h=2时与3-多数相同，且在多数占比超过3/4时收敛时间为O(log n)

**⚠️ 局限性**

局限性包括：只考虑了初始偏差满足Ω(√(max{n/h^2, C1 log n}))的情况，缺乏对h≤k或k接近n时的完整分析，且对最小偏差要求可能过高

---

## 804. BioAlchemy: Distilling Biological Literature into Reasoning-Ready Reinforcement Learning Training Data

**arXiv ID:** 2604.03506 | [PDF](https://arxiv.org/pdf/2604.03506v1)

**作者:** Brian Hsu `[一作]` (University of Chicago), Arvind Ramanathan `[通讯]` (Argonne National Laboratory)

**通讯引用:** 6801 | [OpenAlex ID](https://openalex.org/A5101537136)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了BioAlchemy数据集与训练流程，利用科学论文自动生成可验证的生物学推理题目，并通过主题分布对齐与现代生物研究主题一致，随后训练BioAlchemist‑8B模型在多项生物学基准上取得显著提升。

**💡 创新点**

创新点在于①从真实科研文献中自动抽取可验证的推理问答对；②对推理题目进行MeSH子主题级别的分布对齐与指数平滑；③通过强化学习与可验证奖励提升推理性能；④在8B模型上实现比同级别其他模型更优的生物学推理成绩。

**🔧 技术方法**

主要技术包括LLM驱动的问答生成管道、基于GPT‑4的多标签MeSH主题分类器、贪婪分布匹配算法、指数平滑（α‑smoothing）与GRPO强化学习框架。

**📊 数据集**

使用了来自ASM、PLoS Genetics、PLoS Computational Biology、Semantic Scholar等77.7K篇科研论文，共生成345K条可验证推理题目，形成BioAlchemy‑345K数据集；随后采样得到不同α平滑版本的子集。

**📈 对比分析**

与NaturalReasoning、TextbookReasoning、Nemotron‑Science‑v1等现有推理数据集以及同等规模的Qwen3‑8B、DeepSeek‑R1‑Llama‑8B、GPT‑OSS‑20B等模型进行对比。BioAlchemist‑8B在LAB‑Bench、PubMedQA、GPQA‑Diamond等基准上平均提升至约42.8%，比同级别模型高约8个百分点，显示主题对齐与RL优化的有效性。

**⚠️ 局限性**

局限性包括①模型仍受数据规模限制，进一步扩大或多模态数据可能提升性能；②MeSH分类器误差导致的主题对齐噪声；③对推理题目分布过度匹配可能导致对未覆盖子主题的泛化能力下降；④实验主要集中在8B规模，未探索更大规模模型的可扩展性。

---

## 805. Hallucination Basins: A Dynamic Framework for Understanding and Controlling LLM Hallucinations

**arXiv ID:** 2604.04743 | [PDF](https://arxiv.org/pdf/2604.04743v1)

**作者:** Kalyan Cherukuri `[一作]` (Illinois Mathematics and Science Academy), Lav R. Varshney `[通讯]` (AI Innovation Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了将LLM幻觉视为隐藏状态空间中的几何盆地的动力学框架，并通过理论与实验验证了其任务依赖性。

**💡 创新点**

创新点在于把幻觉解释为局部吸引子，揭示了不同任务下的盆地结构差异，并基于此设计了几何感知的干预策略。

**🔧 技术方法**

使用Transformer隐藏状态轨迹分析、谱半径与收缩性理论、聚类与Fisher判别、以及基于盆地距离的干预向量等技术。

**📊 数据集**

在四大幻觉基准数据集（HaluEval、MuSiQue、FEVER、TruthfulQA）上进行实验验证。

**📈 对比分析**

与熵/不确定性、采样一致性以及表示探测器等传统检测方法对比，基于盆地的AUROC可达0.9以上，自适应几何干预能在不重新训练的情况下显著降低幻觉率。

**⚠️ 局限性**

局限性包括：对多重误认任务（如TruthfulQA）和高维生成任务的盆地分离效果有限；架构差异影响谱特性，需要模型特定分析；仅关注无信息上下文的参考状态，无法处理多模态或更复杂输入场景。

---

## 806. Embedding-Only Uplink for Onboard Retrieval Under Shift in Remote Sensing

**arXiv ID:** 2604.03301 | [PDF](https://arxiv.org/pdf/2604.03301v1)

**作者:** Sangcheol Sim `[一作]` `[通讯]` (Telepix), Sangcheol Sim (Telepix)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了在遥感任务下，仅上传嵌入向量并在机载端进行向量检索来实现灾害响应优先级划分的完整流程，且在多种分布偏移（跨时间、跨事件、跨地点、跨城市）场景下进行实验。

**💡 创新点**

首次系统性地对“仅嵌入上传+机载检索”这一流水线进行多任务基准测试，并揭示决策头（kNN检索或类中心投票）取决于任务结构而非难度，可实现零样本头选择。

**🔧 技术方法**

使用OlmoEarth（Sentinel‑2 12波段）和SigLIP（RGB）自监督嵌入，利用余弦相似度进行kNN检索，结合最近类中心、岭回归线性探针；机载端使用LanceDB做向量索引。

**📊 数据集**

基准数据包括27个Sentinel‑2 L2A灾害场景（15云区站点共15个地点）、5个SpaceNet‑2 AOI以及对应的查询样本，采用10个随机种子重复实验。

**📈 对比分析**

通过与随机、无检索基线以及“类中心/线性探针”进行对比，检索在云分类（0.92），类中心在变化检测（0.85）上表现最佳；在灾害检索中两者接近（Recall@5 1.00）。

**⚠️ 局限性**

实验局限于单一光学传感器（Sentinel‑2 10 m GSD），未覆盖跨传感器、跨季节、噪声鲁棒性和分数校准等场景，需要进一步扩大规模与多样性。

---

## 807. Temporal Inversion for Learning Interval Change in Chest X-Rays

**arXiv ID:** 2604.04563 | [PDF](https://arxiv.org/pdf/2604.04563v1)

**作者:** Hanbin Ko `[一作]` (Seoul National University), Chang Min Park `[通讯]` (Seoul National University)

**通讯引用:** 13159 | [OpenAlex ID](https://openalex.org/A5089624618)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为TILA的框架，通过在预训练、微调和推理阶段加入时间倒置监督，增强配对胸部X光片的时序变化识别能力。

**💡 创新点**

创新点在于利用图像序列的时间倒置（即交换前后图像顺序）作为监督信号，设计了变化感知Sigmoid损失、双向交叉熵和时间一致性损失，形成统一的逆序感知学习与对齐机制；并提出了针对时序评估的MS‑CXR‑Tretrieval检索集与评估协议。

**🔧 技术方法**

技术上使用了SigLIP/CLIP的对比学习框架，结合BioViL‑T和ALTA两种配对图像编码器，添加了Change‑aware Sigmoid Loss、Bidirectional Cross‑Entropy (BiCE)、Temporal Consistency Loss (TCL)以及逆序融合推理策略。

**📊 数据集**

数据集包括公开的MIMIC‑CXR、MS‑CXR‑T、CheXpert、私有三级医院数据以及RexGradient；训练时使用MIMIC‑CXR的前后对照图像，检索与分类评估则基于MS‑CXR‑T与MS‑CXR‑Tretrieval。

**📈 对比分析**

与基线SigLIP、CLIP以及现有的配对VLP模型相比，TILA在检索的Recall@k保持相近同时显著提升TEM分数；在进展分类（零样本和全监督）中，TILA的标准准确率提升约5–10%，在逆序、合并与一致性评估上更显著，外部验证集的二分类AUC也提升至0.70以上。

**⚠️ 局限性**

局限性包括：标注噪声与放射员间的差异导致进展标签不确定；时间倒置并非对所有疾病进展都是对称有效，特别是不可逆的病变；以及在极少数细微变化或边缘案例中模型仍可能产生不一致的预测。

---

## 808. Researchers waste 80% of LLM annotation costs by classifying one text at a time

**arXiv ID:** 2604.03684 | [PDF](https://arxiv.org/pdf/2604.03684v1)

**作者:** Christian Pipal `[一作]` (University of Zurich), Frank Esser `[通讯]` (University of Zurich)

**通讯引用:** 12333 | [OpenAlex ID](https://openalex.org/A5042787477)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大语言模型在文本分类中的批量处理和多变量堆叠进行系统评估。

**💡 创新点**

首次量化不同批量大小和变量堆叠对多任务编码质量与成本的影响，并给出安全范围建议。

**🔧 技术方法**

使用多款生产型LLM（Claude Haiku 4.5、GPT‑5‑mini／nano、Gemini 3/3.1 Flash‑Lite、Qwen 3.5 Plus/Flash）与批量与堆叠提示技术。

**📊 数据集**

采用来自gilardi2023chatgpt的3,962条推文的四维度人工标注数据集。

**📈 对比分析**

通过与单条单变量基准对比，计算准确率变化与成本节省；大多数模型在批量≤100、堆叠≤10时准确率下降<2pp，成本可节省80%+。

**⚠️ 局限性**

结果仅基于短英文推文，未验证长文本、多语言或更复杂任务；某些模型（GPT‑5 mini/nano）在大批量时性能急剧下降。

---

## 809. LLM-Enabled Open-Source Systems in the Wild: An Empirical Study of Vulnerabilities in GitHub Security Advisories

**arXiv ID:** 2604.04288 | [PDF](https://arxiv.org/pdf/2604.04288v1)

**作者:** Fariha Tanjim Shifat `[一作]` (Missouri University of Science and Technology), Mia Mohammad Imran `[通讯]` (Missouri University of Science and Technology)

**通讯引用:** 74 | [OpenAlex ID](https://openalex.org/A5088063511)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统性分析了2025年1月至2026年1月期间GitHub Security Advisories（GHSA）中涉及LLM组件的漏洞，探讨了CWE与OWASP LLM风险分类之间的关系；

**💡 创新点**

提出了在传统CWE框架之外，用OWASP Top 10 for LLM Applications 2025进行架构级风险映射，并证明GHSA元数据缺乏LLM参与和模型中介暴露的结构化标识；

**🔧 技术方法**

采用关键词过滤、手工分类与注释、统计分析、交叉映射技术，对GHSA数据进行提取、过滤、手工标注与关联分析；

**📊 数据集**

使用295条GHSA记录（包含133个不同软件包），并对其中100条与LLM相关的记录进行手工注释；

**📈 对比分析**

通过将100条记录映射到OWASP LLM类别并与CWE ID 对齐，展示了供应链、过度授权和提示注入等架构风险的多标签组合，表明LLM相关漏洞与传统弱点在实现层保持一致，但在架构层呈现出新的组合模式；

**⚠️ 局限性**

局限包括：仅限GHSA公开披露；关键词过滤可能漏检或误检；手工注释带有主观判断；仅分析已公开漏洞，未考虑未披露或未发现的弱点。

---

## 810. Shorter, but Still Trustworthy? An Empirical Study of Chain-of-Thought Compression

**arXiv ID:** 2604.04120 | [PDF](https://arxiv.org/pdf/2604.04120v1)

**作者:** Lingjie Zeng `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Xiuying Chen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 1322 | [OpenAlex ID](https://openalex.org/A5101568165)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了链式推理压缩对模型可信度的影响，并证明压缩方法往往会导致安全性、幻觉抵抗或多语言鲁棒性的退化。

**💡 创新点**

创新点在于：①提出了按维度归一化的效率得分E，能够同时捕捉压缩带来的收益与退化；②用对齐感知的DPO实现了在显著缩短推理链的同时保持可信度的“可行性证明”；③首次在多尺度（1.5B‑32B）模型上进行跨维度、跨方法的可信度对比。

**🔧 技术方法**

使用了多种压缩技术（L1 distillation、BeConcise、TokenSkip、VeriThinker、Thinkless、DPO等）以及标准推理配置（温度0，贪婪解码）和自动评测管道（Llama Guard 3、GPT‑4o‑mini）。

**📊 数据集**

数据集包括 HarmBench、AgentHarm（安全性评测）、FaithEval（幻觉抵抗评测）和 MMLU‑ProX（多语言鲁棒性评测），全部来源于公开基准。

**📈 对比分析**

与对应的未压缩基线按相同提示、解码与评测设置进行对比，使用 E 评分揭示压缩方法在安全性与多语言性能上出现的互斥退化；DPO 在 19.3% 长度缩减的同时，使安全性仅下降 3.8%，并保持其它维度几乎无损，显示了可行的安全与效率共存。

**⚠️ 局限性**

局限性包括：评测仅覆盖少数几种压缩方案，且安全判定依赖单一自动评测模型；未进行人工评估，难以完全验证可信度退化的真实影响；方法可推广性需在更多基模型和更丰富的评测工具上进一步验证。

---

## 811. GUIDE: Interpretable GUI Agent Evaluation via Hierarchical Diagnosis

**arXiv ID:** 2604.04399 | [PDF](https://arxiv.org/pdf/2604.04399v1)

**作者:** Yuwen Zhai `[一作]` (Alibaba Group), Benlei Cui `[通讯]` (Alibaba Group)

**通讯引用:** 108 | [OpenAlex ID](https://openalex.org/A5030151061)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了GUIDE框架，分阶段对GUI代理轨迹进行分段、子任务诊断和整体摘要，提供可解释的评估报告。

**💡 创新点**

将评估过程分解为子任务段，利用有限上下文进行多模态诊断，解决长轨迹上下文过载和二元判定的可解释性缺失。

**🔧 技术方法**

基于多模态大语言模型（如GPT‑4o/4‑mini等）进行文本+视觉推理，使用结构化JSON输出实现链式思维和错误分析，采用无训练的零样本评估。

**📊 数据集**

工业电商轨迹数据(932条)、AGENTREWARDBENCH(1302条)和AndroidBench(480条)。

**📈 对比分析**

与Rule‑based、AgentTrek、Autonomous Evaluation、WebJudge等基线对比，GUIDE在电商数据上准确率95.8%（比最强基线高5.35点），在AGENTREWARDBENCH上精度89.2%（比WebJudge高7.2点），在AndroidBench上准确率94.9%（高于其他方法）。

**⚠️ 局限性**

对需要领域专业知识的任务评估可能失误，且依赖大语言模型的通用推理能力；在极长轨迹上仍需进一步验证。

---

## 812. Minos: Systematically Classifying Performance and Power Characteristics of GPU Workloads on HPC Clusters

**arXiv ID:** 2604.03591 | [PDF](https://arxiv.org/pdf/2604.03591v1)

**作者:** Rutwik Jain `[一作]` (University of Wisconsin-Madison), Shivaraman Venkataraman `[通讯]` (University of Wisconsin-Madison)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Minos，一套基于 GPU 能耗峰值和利用率的工作负载分类框架，用于预测功耗和性能，从而指导频率限制与功耗管理。

**💡 创新点**

创新点在于同时考虑功耗峰值分布与计算/内存利用率的联合特征，并通过低成本实时功耗采样与硬件计数器实现高精度的邻居匹配预测；与以往仅用平均功耗分类的工作相比，显著提升了功耗峰值预测精度。

**🔧 技术方法**

使用低频率（1‑2 ms）功耗采样、GPU 计数器（SM/内存吞吐、能耗累计）提取特征；对功耗峰值分布构造向量后做层次聚类；对利用率做加权平均后做 2‑D K‑Means；结合邻居搜索实现频率限制预测。

**📊 数据集**

在 18 个常用工作负载上评估，涵盖图分析、HPC、HPC+ML 与 ML（如 LLaMA、ResNet、FAISS、Qwen1.5‑MoE 等），分别在 AMD MI300X（8 GPU/节点）与 NVIDIA A100（3 GPU/节点）集群上收集数据。

**📈 对比分析**

与先前的 Guerreiro 等人基于平均功耗的分类方案对比，Minos 在 90 % 频率限制预测误差仅 4 %，而对手为 14 %；在功耗峰值预测误差上平均 4 %（比对手高 10 %），性能预测误差 3 %；并将新工作负载的频率搜索时间缩短 89 %–90 %，显著提升实验效率。

**⚠️ 局限性**

局限性包括：需至少一次完整的默认频率采样；分类精度随 GPU 供应商、代际及 DVFS 机制差异而变化；对极端或完全新型工作负载的邻居匹配可能不够准确；以及对多 GPU 交互或分布式调度的适配尚待进一步验证。

---

## 813. Automata Learning versus Process Mining: The Case for User Journeys

**arXiv ID:** 2604.03686 | [PDF](https://arxiv.org/pdf/2604.03686v1)

**作者:** Paul Kobialka `[一作]`, Silvia Lizeth Tapia Tarifa `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过对用户旅程事件日志自动生成行为模型，比较了自动机学习（AL）与过程挖掘（PM）的效果，并提出了一种自适应的混合学习方法（Hybrid）

**💡 创新点**

①提出了可自动决定使用 AL 或 PM 的混合算法；②设计了基于日志统计的置信度阈值 λ 以及自适应 α 近似函数；③在合成基准和四个真实案例中系统评估了该方法

**🔧 技术方法**

使用被动自动机学习（Alergia）、过程挖掘中的直接跟随图（DFG）和直接跟随系统（DFS），以及自定义的 Hybrid 选择逻辑

**📊 数据集**

①合成基准：14 种控制流组合、140 条过程树，日志大小从10到1000条；②真实案例：GrepS（33条）、BPIC12（5053条）、BPIC17a（10746条）、BPIC17b（12344条）

**📈 对比分析**

通过精度、召回率、F1 分数以及对未见测试日志的错误率进行比较；Hybrid 在合成基准中平均 F1 ≈ 0.90，精度与召回均优于单独的 AL 或 PM；在真实案例中，Hybrid 能在稀疏日志下使用 PM，在大日志下使用 AL，显著降低错误率

**⚠️ 局限性**

①对日志质量的假设（无噪声、完整）导致方法在实际复杂日志中可能失效；②需要人工调参（α、λ）或预处理；③只适用于顺序、单线程的用户旅程，无法直接处理并发或无序交互

---

## 814. Solar-VLM: Multimodal Vision-Language Models for Augmented Solar Power Forecasting

**arXiv ID:** 2604.04145 | [PDF](https://arxiv.org/pdf/2604.04145v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 815. Quantifying Trust: Financial Risk Management for Trustworthy AI Agents

**arXiv ID:** 2604.03976 | [PDF](https://arxiv.org/pdf/2604.03976v1)

**作者:** Wenyue Hua `[一作]` (Microsoft Research), Chandler Fang `[通讯]` (t54.ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一套 Agentic Risk Standard (ARS)，为代理系统提供结算层面的、可执行的保障机制；

**💡 创新点**

创新点在于将服务费用的托管与本金的保底拆分为两个轨道，并用保险、抵押和多方签名等金融工具构建可追溯、可执行的状态机；

**🔧 技术方法**

主要技术包括金融风险管理概念（托管、保险、抵押）、结构化协议与状态机设计、以及与授权协议（AP2、VI）和支付协议（x402）的整合；

**📊 数据集**

实验使用合成数据：交易金额服从对数正态分布、失败概率服从 Beta 分布，并通过蒙特卡洛仿真评估不同参数设置；

**📈 对比分析**

与无保险基线相比，实验显示引入 ARS 可显著降低用户损失（最高 91%），并通过保费与抵押的调节实现收益与保险公司偿付能力的权衡，整体表现取决于保费加载、误报率、协方差等参数；

**⚠️ 局限性**

局限在于依赖准确的风险估计与保单定价，模拟环境过于简化，难以覆盖所有类型的失误（如偏见、心理伤害），且需要在真实市场中进一步验证和校准。

---

## 816. Learning 3D Reconstruction with Priors in Test Time

**arXiv ID:** 2604.03878 | [PDF](https://arxiv.org/pdf/2604.03878v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 817. Where to Steer: Input-Dependent Layer Selection for Steering Improves LLM Alignment

**arXiv ID:** 2604.03867 | [PDF](https://arxiv.org/pdf/2604.03867v1)

**作者:** Soham Gadgil `[一作]` (University of Washington), Su-In Lee `[通讯]` (University of Washington)

**通讯引用:** 24680 | [OpenAlex ID](https://openalex.org/A5028723221)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Where to Steer（W2S）框架，能够根据输入预测最佳干预层并在该层插入 steering 向量，进而实现对大型语言模型的推理时对齐。

**💡 创新点**

创新点在于突破传统固定层干预的局限，首次证明并实践了输入依赖的层选择，从而显著提升了模型对齐效果。

**🔧 技术方法**

主要技术包括 steering 向量生成（CAA 与 L2S）、浅层多层感知机层预测器、句子嵌入式 prompt encoder，以及层选择的交叉熵训练。

**📊 数据集**

使用了 13 个来自 Model‑Written Evaluations（MWE）的对齐数据集，针对 Llama‑2‑7B‑Chat（32 层）和 Qwen‑1.5‑14B‑Chat（40 层）进行实验。

**📈 对比分析**

与全局固定层基线相比，W2S 在所有行为下平均提升 steerability 约 20‑30% 以及可调节示例比例，且在 OOD 诱导的 prompt 变体中仍保持更高的对齐性能。

**⚠️ 局限性**

局限性包括层预测器的准确率受限于训练样本稀缺、层标签稀疏导致的剪枝依赖、仅在多项选择任务上评估、以及未验证对开放式生成的适用性。

---

## 818. Formal Constraints on Dependency Syntax

**arXiv ID:** 2604.04542 | [PDF](https://arxiv.org/pdf/2604.04542v1)

**作者:** Gómez-Rodríguez `[一作]`, Lluís `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了依存语法的形式约束，重点分析项目化、温和非项目化树及其相关解析方法与覆盖率统计，构建了各类树的包含关系图；

**💡 创新点**

系统梳理并归纳了温和非项目化树的多种定义及其互相关系，首次给出覆盖率、解析复杂度与语料兼容性的对比，提出了未来研究方向；

**🔧 技术方法**

主要采用理论分析、复杂度推导与实证覆盖率统计，并引用已有的解析算法与实验结果；

**📊 数据集**

使用了大量Universal Dependencies、PUD、古希腊诗歌等多语种树库进行覆盖率评估；

**📈 对比分析**

通过与现有项目化与伪项目化解析器、2-平面、1-端点交叉等方法的覆盖率（>95%）和复杂度（如O(n^5+2k)、O(n^4)等）进行对比，展示了各类约束在效率与实用性上的差距；

**⚠️ 局限性**

未覆盖所有后来提出的树类，缺乏对所有类的多项式解析器，且理论定义与实际实现之间仍存在差距，未来研究需进一步弥补这些空白。

---

## 819. SpatialEdit: Benchmarking Fine-Grained Image Spatial Editing

**arXiv ID:** 2604.04911 | [PDF](https://arxiv.org/pdf/2604.04911v1)

**作者:** Yicheng Xiao `[一作]` (University of Hong Kong), Xiaojuan Qi `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出细粒度图像空间编辑基准（SpatialEdit-Bench）和可控 Blender 合成数据集（SpatialEdit-500k），并在此基础上训练 SpatialEdit-16B 模型，实现精准的对象位移/旋转与相机视角控制。

**💡 创新点**

创新点包括：①将视角重建与框架分析结合的几何感知评估；②构建大规模、可控合成数据集以解决数据瓶颈；③在合成数据上使用 LoRA 微调提升模型的空间编辑能力。

**🔧 技术方法**

使用多模态 Encoder + MM‑DiT 解码器架构，结合 VLM 生成指令、SAM3 分割、VGGT 视角估计、YOLO 检测以及 Blender 渲染技术进行数据合成与模型训练。

**📊 数据集**

使用自研的 SpatialEdit-500k（500k 图像‑文本对）以及公开的编辑数据集进行预训练，评测使用 GEdit-Bench（一般编辑）和 SpatialEdit-Bench（空间编辑）。

**📈 对比分析**

与 LongCatImage-Edit、视频世界模型等进行对比；在 SpatialEdit-Bench 上取得对象移动 0.673、旋转 0.632、视角误差 0.243 的最佳成绩，在 GEdit-Bench 上保持 7.52 分，整体性能优于现有开源模型。

**⚠️ 局限性**

局限性包括：主要依赖合成数据，缺乏对真实摄影图像的广泛适应；对极端细节、复杂背景以及多视角一致性仍存在挑战。

---

## 820. Learning, Potential, and Retention: An Approach for Evaluating Adaptive AI-Enabled Medical Devices

**arXiv ID:** 2604.04878 | [PDF](https://arxiv.org/pdf/2604.04878v1)

**作者:** Alexis Burgon `[一作]` (U.S. Food and Drug Administration), Ravi K Samala `[通讯]` (U.S. Food and Drug Administration)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并验证了学习、潜在、保留三种度量，用于评估适应性 AI 模型在连续修改步骤中的表现；在模拟的人口迁移实验中展示其效果。

**💡 创新点**

创新点在于设计能分别捕捉模型改进、数据集挑战和知识保留的三重指标，并通过案例阐释模型可塑性与稳定性之间的平衡。

**🔧 技术方法**

使用 ResNet‑18 进行胸部 X 光影像分类，采用 AUROC 作为评估指标，并在 Python 包 VIGILANT 中实现这些测量。

**📊 数据集**

利用 MIDRC 的 Open‑A1 与 Open‑R1 数据库（CR 与 DX X 光）构建逐步迁移的人口样本集。

**📈 对比分析**

与传统单一性能评估相比，学习、潜在、保留三指标能够分别揭示模型改进、数据集挑战和知识保留；实验显示单一迁移时性能稳定、有限可塑性导致性能下降但保留稳固，双重迁移时性能波动反映数据挑战与学习冲突。

**⚠️ 局限性**

局限在于仅在模拟实验中验证，缺乏真实临床部署的持续监测；对权重衰减 λ 的选择敏感；未充分考虑模型过拟合或多任务变化下的安全性。

---

## 821. ADAPT: AI-Driven Decentralized Adaptive Publishing Testbed

**arXiv ID:** 2604.04077 | [PDF](https://arxiv.org/pdf/2604.04077v1)

**作者:** Md Motaleb Hossen Manik `[一作]` (Rensselaer Polytechnic Institute), Ge Wang `[通讯]` (Rensselaer Polytechnic Institute)

**通讯引用:** 42176 | [OpenAlex ID](https://openalex.org/A5100400458)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

构建了一套基于代理的仿真测试床ADAPT，重新把学术出版视为闭环治理系统；

**💡 创新点**

核心创新在于：①可解释、受限的政策更新机制，②去中心化治理与角色轮换，③结合后期发表指标进行信用与激励对齐，④完整的审计日志实现可追溯治理；

**🔧 技术方法**

使用了代理模型、离散时间仿真、基于系统信号的控制器、AI辅助的分层审稿与匹配、信用动态更新等技术；

**📊 数据集**

采用可配置的合成数据集：作者/审稿人关键词、稿件质量、复杂度、审稿人噪声、审稿时长等，便于多场景重现；

**📈 对比分析**

通过多种压力情景（提交激增、质量漂移、争议峰值、协同攻击等）与基线对比，评估关键指标（backlog、平均争议、审稿负载、集中度）以及政策稳定性；性能表现：在大多数情景下，backlog在可接受范围内收敛，政策参数保持在设定边界内，系统对压力的响应有限且可解释；

**⚠️ 局限性**

主要局限包括：①仿真模型过于简化，缺乏真实出版操作的细节；②未模拟完全自适应的恶意行为；③信用与激励机制基于简化的后期影响模型，易被游戏；④实际部署需整合隐私、安全与法律合规，未在实验中验证。

---

## 822. CRAFT: Video Diffusion for Bimanual Robot Data Generation

**arXiv ID:** 2604.03552 | [PDF](https://arxiv.org/pdf/2604.03552v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 823. Pinching Antenna Systems (PASS): Enabling Reconfigurable and Controllable Wireless Channels -- A Comprehensive Survey

**arXiv ID:** 2604.04521 | [PDF](https://arxiv.org/pdf/2604.04521v1)

**作者:** Elmehdi Illi `[一作]` (Hamad Bin Khalifa University), Marwa Qaraqe `[通讯]` (Hamad Bin Khalifa University)

**通讯引用:** 2332 | [OpenAlex ID](https://openalex.org/A5010196813)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对Pinching Antenna Systems（PASS）在6G无线网络中的设计、架构、优化、功耗、安全、ISAC等方面的研究进行系统性综述，归纳现有工作并指出研究空白。

**💡 创新点**

首次系统划分PASS方案按通信目标（覆盖、数据率、能效、安全、感知）和技术手段（波导架构、激活策略、学习方法等），并对比PASS与传统MIMO、RIS等技术的优势与局限。

**🔧 技术方法**

综合利用理论建模、闭式分析、凸优化、交替优化、半对数程序、深度学习（GNN、Transformer、DRL）等多种技术，对PASS方案进行评估与优化。

**📊 数据集**

未采用实际实验或公开数据集，而是聚合并引用文献中公开的仿真/实验结果和数学推导。

**📈 对比分析**

通过对已有工作中的性能指标（吞吐量、覆盖范围、能效、信道安全性、感知精度）进行汇总，并与传统MIMO、RIS等基准方案对比，展示PASS在覆盖、能效、短包路经等场景下的显著提升。

**⚠️ 局限性**

存在研究范围有限（大多聚焦单波导或等距PA布局）、缺乏大规模实测验证、对波导衰耗、PA定向等现实效应建模不足，以及PASS与RIS、AI融合的系统级评估尚待深入。

---

## 824. A Systematic Study of Cross-Modal Typographic Attacks on Audio-Visual Reasoning

**arXiv ID:** 2604.03995 | [PDF](https://arxiv.org/pdf/2604.03995v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 825. AICCE: AI Driven Compliance Checker Engine

**arXiv ID:** 2604.03330 | [PDF](https://arxiv.org/pdf/2604.03330v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 826. ReFinE: Streamlining UI Mockup Iteration with Research Findings

**arXiv ID:** 2604.04353 | [PDF](https://arxiv.org/pdf/2604.04353v1)

**作者:** Donghoon Shin `[一作]` (University of Washington), Gary Hsieh `[通讯]` (University of Washington)

**通讯引用:** 3919 | [OpenAlex ID](https://openalex.org/A5060931546)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个 Figma 插件 REFINE，实时从 HCI 论文检索、翻译并可视化设计启示，帮助 UI mockup 迭代。

**💡 创新点**

将多模态 LLM 与论文检索、设计上下文抽取、主题聚类、视觉化行动项结合，构建端到端实时从论文到设计迭代的流程。

**🔧 技术方法**

使用多模态 LLM（Gemini 2.0 Flash）、GROBID 文档解析、Google text‑embedding‑004、向量检索、层次聚类、HTML/CSS 生成与编辑、Figma API 以及 SvelteKit 前端。

**📊 数据集**

使用 CHI 2025 论文集 1060 篇、50 条移动 UI mockup 示例（每个 3–5 屏幕）、50 篇 HCI 论文样本进行技术评估，以及 12 名设计师参与的用户研究。

**📈 对比分析**

通过与其他 LLM（Claude 3.5 Sonnet、GPT‑4o）和输入模态（图片、JSON）比较，Gemini 在 10.8 s 时延、0.79 视觉相似度下表现最佳；edit‑only 方法比全重构快 5.5×且准确率 95.3%；用户研究显示 NASA‑TLX 工作量显著下降，设计编辑量从 2.4 次提升至 5.5 次，洞察质量评分提升至 3.82/5。

**⚠️ 局限性**

局限于 CHI 论文；生成的行动项偏向添加 UI 元素，缺少简化建议；可视化仅支持静态预览，无法处理交互或新屏；无法捕捉设计师未显式的目标或隐性上下文；检索范围有限，需扩展到其他会议或学科。

---

## 827. Asymmetric reformulation of draw rules in chess and its implications for game theory: Repetition as loss for White

**arXiv ID:** 2604.03683 | [PDF](https://arxiv.org/pdf/2604.03683v1)

**作者:** Chong Qi `[一作]` (KTH Royal Institute of Technology), Chong Qi `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 3324 | [OpenAlex ID](https://openalex.org/A5074680349)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并评估一种非对称三次重复规则（ARR），将由白方发起的三次重复判为白方输局，以减少高水平棋赛的平局率并削弱白方的先行优势；同时构建了基于Stockfish与NNUE的自适应棋盘引擎及图论、博弈论分析框架来验证其有效性。

**💡 创新点**

创新点在于：①将传统中性三次重复规则转化为对白方具有惩罚性的非对称规则；②通过惩罚白方的被动循环玩法，平衡先手优势并促进更具进攻性、观赏性的棋局；③在保持棋盘结构不变的前提下，仅修改循环节点的收益，从而不改变游戏本质。

**🔧 技术方法**

采用了深度卷积残差网络（ResNet‑style NNUE）、强化学习自适应搜索（Stockfish‑style engine）、自我对弈评估、搜索约束（contempt）调节，以及图论与纳什均衡理论的数学分析。

**📊 数据集**

主要使用了公开的大型棋局数据库（Lichess 约3.62 亿局）、TWIC 赛事实战数据、以及通过自我对弈生成的数百万局棋局来训练和测试模型。

**📈 对比分析**

通过在不同搜索深度与偏好设置下的七种自我对弈实验，将标准规则与ARR结果进行对照，评估白/黑胜率与平局率；并通过精细化训练的NNUE模型评估起始优势变化。实验显示ARR能显著降低平局率，并在多数情境下提升双方胜率（尤其是白方在弱势时的反弹机会），整体性能优于传统规则。

**⚠️ 局限性**

局限性包括：①对人类玩家的心理影响与适应成本尚未实测；②实验主要基于计算机对弈，真实比赛中的人类决策与策略可能产生不同结果；③模型对高深度搜索的依赖导致训练与评估成本高；④规则改变可能在不同开局体系或时间控制下产生不同效果，需进一步验证。

---

## 828. Towards Considerate Human-Robot Coexistence: A Dual-Space Framework of Robot Design and Human Perception in Healthcare

**arXiv ID:** 2604.04374 | [PDF](https://arxiv.org/pdf/2604.04374v1)

**作者:** Yuanchen Bai `[一作]` (Cornell Tech), Angelique Taylor `[通讯]` (Cornell Tech)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过14周的共创与随访访谈，研究医疗机器人在人机共存中的认知演变与共存框架。

**💡 创新点**

提出人机共存的双空间互演进循环模型，并构建人类认知空间的四维解释框架（分解程度、时间取向、推理范围、证据来源），以及“体贴共存”概念。

**🔧 技术方法**

采用质性研究方法：访谈、主题分析、Cohen's Kappa 一致性检验，未使用特定机器学习技术。

**📊 数据集**

数据来源为9名参与者的访谈记录与回顾性评估。

**📈 对比分析**

通过主题编码与一致性检验（κ=0.82/0.86）得到RQ1与RQ2的三个主题；无量化性能指标。

**⚠️ 局限性**

样本量小（9人），受访者群体单一，缺乏多样性；未在真实部署环境中验证模型与结论。

---

## 829. OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text

**arXiv ID:** 2604.04348 | [PDF](https://arxiv.org/pdf/2604.04348v1)

**作者:** Weiguo Pian `[一作]` (University of Texas at Dallas), Yapeng Tian `[通讯]` (University of Texas at Dallas)

**通讯引用:** 10939 | [OpenAlex ID](https://openalex.org/A5101835756)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了统一全景音频生成任务UniHAGen，并设计了OmniSonic模型，实现对屏幕内外语音与环境声的同时合成。

**💡 创新点**

创新点在于引入三向交叉注意的TriAttn‑DiT结构与Mixture‑of‑Experts门控，实现对屏幕内外多种声音源的自适应融合，并在此基础上提出flow‑matching扩散框架与全新benchmark UniHAGen‑Bench。

**🔧 技术方法**

技术手段包括：基于流匹配的扩散模型、TriAttn‑DiT与MoE门控、CLIP视觉编码器、FLAN‑T5与SpeechT5文本编码、VAE隐空间编码、HiFi‑GAN声码器。

**📊 数据集**

训练数据由VGGSound、LRS3与CommonVoice三大公开数据集合成，构成多场景混合语音与环境音；benchmark UniHAGen‑Bench包含1003个人工标注样本，覆盖三类典型场景。

**📈 对比分析**

与AudioLDM2、VoiceLDM、VinTAGe、MMAudio、HunyuanVideo‑Foley等SOTA模型对比，OmniSonic在FAD、MKL、AT/AV、WER/CER/PER等客观指标上均显著领先，主观MOS评测中也取得最高分，体现出更高质量与语义一致性。

**⚠️ 局限性**

局限性包括：使用CLIP视觉特征导致在视觉静态或弱动态场景下的时间同步度下降；合成训练数据可能不完全覆盖真实复杂环境；模型对实时生成和低资源部署的支持尚不充分。

---

## 830. Are LLM-Based Retrievers Worth Their Cost? An Empirical Study of Efficiency, Robustness, and Reasoning Overhead

**arXiv ID:** 2604.03676 | [PDF](https://arxiv.org/pdf/2604.03676v1)

**作者:** Abdelrahman Abdallah `[一作]` (University of Innsbruck), Adam Jatowt `[通讯]` (University of Innsbruck)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

复现并扩展BRIGHT检索基准，对14种检索器在12个任务上进行全面评估，新增索引时间、查询延迟、吞吐量、鲁棒性、置信度校准等多维度指标。

**💡 创新点**

提出Efficiency Ratio度量评估推理增强检索的收益与延迟成本；系统化比较稀疏、密集、LLM基、推理专用检索器的效果与效率；验证推理增强在不同域的任务依赖性。

**🔧 技术方法**

使用Dense Bi‑Encoder（SBERT、BGE、Contriever、Nomic、E5‑Mistral、SFR‑Mistral、GTE‑Qwen、GTE‑Qwen2）、LLM指令调优模型（Inst‑L、Inst‑XL）、推理专用模型（ReasonIR、RaDeR、Diver）、BM25以及BM25+Dense的线性融合、RRF、DAT混合策略；利用Chain‑of‑Thought生成的长查询进行推理增强；通过AUROC衡量检索分数的置信度校准。

**📊 数据集**

BRIGHT benchmark（12个多步推理任务，涵盖Science、StackExchange、Coding、Theorem等领域），包含标准短文档和长文档两种检索设置。

**📈 对比分析**

对比nDCG@10、Recall@k、吞吐量(QPS)、延迟(p50/p95/p99)、Efficiency Ratio、鲁棒性保留比、AUROC；结果显示：推理专用检索器（Diver、ReasonIR、RaDeR）在效果上远超7B LLM基模型，并在吞吐量上仍保持优势；子1B模型与推理增强几乎无额外延迟，收益显著；BM25+Dense的线性融合在低于20 nDCG的模型上可实现约+4–5点提升；但推理增强在正式/代码域会下降，置信度校准普遍偏弱，AUROC仅≈0.6。

**⚠️ 局限性**

主要限制：推理增强并非普适，对不同领域需域感知路由；大型LLM基模型在效率上仍高成本；推理专用模型在语义一致性和鲁棒性上表现不一；检索分数的置信度校准不足，难以直接用于自动路由或RAG。

---

## 831. LLM-based Listwise Reranking under the Effect of Positional Bias

**arXiv ID:** 2604.03642 | [PDF](https://arxiv.org/pdf/2604.03642v1)

**作者:** Jingfen Qiao `[一作]` (University of Amsterdam), Andrew Yates `[通讯]` (Johns Hopkins University)

**通讯引用:** 3167 | [OpenAlex ID](https://openalex.org/A5059489981)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种名为DebiasFirst的LLM列表重排方法，结合逆倾向评分和位置感知增强来消除位置偏差。

**💡 创新点**

首次在LLM重排中系统性地将逆倾向评分校准与位置感知数据增强相结合，显著降低了因输入顺序导致的性能波动。

**🔧 技术方法**

使用逆倾向评分（IPS）进行位置校准、位置感知数据增强（Pos-Aug）、单词级输出logit的逻辑回归损失、语言模型目标、梯度累积、bf16等技术。

**📊 数据集**

在MS MARCO训练集（约40k示例）上进行微调，并在MS MARCO dev、TREC DL 2019/2020、BEIR等多域数据集上评估，使用Contriever、Splade++、BM25等检索器的候选列表。

**📈 对比分析**

与RankZephyr、First、ListT5以及PermSC等基线进行对比，NDCG@10提升约2–4%，并在随机排列时保持性能稳定，整体性能优于所有基线。

**⚠️ 局限性**

仅在窗口大小为20的候选列表上验证，对更长上下文（如100条目）下的效果尚未评估。

---

## 832. Neural Operators for Multi-Task Control and Adaptation

**arXiv ID:** 2604.03449 | [PDF](https://arxiv.org/pdf/2604.03449v1)

**作者:** David Sewell `[一作]` (University of Texas at Austin), David Fridovich-Keil `[通讯]` (University of Texas at Austin)

**通讯引用:** 591 | [OpenAlex ID](https://openalex.org/A5070827615)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了神经算子在多任务最优控制中的应用，利用SetONet将任务描述函数映射到最优反馈策略，并提出了基于分支‑干线结构的多种适配与元学习策略；

**💡 创新点**

创新点包括：①将神经算子用于无限维函数空间的多任务控制，首次实现任务描述到控制策略的直接映射；②提出轻量级分支/末层微调和两种元训练变体（SetONet‑Meta、SetONet‑Meta‑Full），在少样本场景下显著优于MAML；③展示了任务分辨率不变性和成本反馈自适应等新特性；

**🔧 技术方法**

主要技术手段包括SetONet（基于DeepONet的分支‑干线神经算子）、行为克隆（BC）作为监督信号、分层微调（全网络、分支、末层）、MAML‑风格的元训练以及基于成本的微调；

**📊 数据集**

实验数据涵盖四个参数化OCP环境（点到点成本、点到点动力学、平面四旋翼、障碍回避）以及iMuJoCo的HalfCheetah-v3；每个环境提供不同数量的任务、轨迹与演示；

**📈 对比分析**

通过与MAML、传统全网络微调等方法比较，采用相对L²误差和模型滚动评估；结果显示SetONet在零步预测下已达到较低误差，且在1–25步微调后优于MAML，尤其在少样本和离群任务场景中表现突出；

**⚠️ 局限性**

局限性包括：①仅处理单一任务定义函数（成本或动力学），未涵盖多函数共同变化；②对超参数和元训练设定敏感；③在高维、噪声大、非最优演示的环境（如HalfCheetah）下需要更多数据和梯度步才能收敛；④缺乏在线自适应与安全约束的进一步研究。

---

## 833. The Blind Spot of Adaptation: Quantifying and Mitigating Forgetting in Fine-tuned Driving Models

**arXiv ID:** 2604.04857 | [PDF](https://arxiv.org/pdf/2604.04857v1)

**作者:** Runhao Mao `[一作]` (Shanghai Jiao Tong University), Zhipeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统研究了 Vision‑Language Models 在自驾场景中的灾难性遗忘问题，构建了最大规模的多源长尾 QA 数据集 Fidelity Driving Bench，并提出了将适配从权重空间转移到提示空间的 Driving Expert Adapter (DEA)，通过 Prompt Adapter 与 Task‑Adaptive Expert Module 实现知识保留与任务适应的平衡。

**💡 创新点**

创新点包括：①首次量化自驾 VLM 的灾难性遗忘并建立基准；②构建 180K 场景、900K QA 的多源长尾数据集；③提出基于 Prompt 的适配与 Mixture‑of‑Experts 的动态路由框架（DEA），有效避免全微调导致的知识消亡。

**🔧 技术方法**

采用了 Prompt Adapter、轻量级 LoRA 专家、Mixture‑of‑Experts（TAEM）动态路由、基于 IDF‑BM25 的长尾挖掘、LLM（Gemini2.5‑Flash、GPT‑4o、Qwen3‑Max）评判器以及多源数据融合技术。

**📊 数据集**

使用了 15 个自驾 VQA 数据集（DriveLM、ImpromptuVLA、CODALM、DriveBench、CoVLA、OmniDrive、LingoQA、NuScenesQA 等）以及 WOD‑E2E 进行统一标注，构成 180K 场景、900K QA 的训练集和 1,000 场景的长尾测试集。

**📈 对比分析**

与多种基线（全微调、LoRA、单源训练等）在三任务（场景描述、Traffic‑QA、重要物体感知）上使用 LLM 判定进行对比；DEA 在保持知识保留率约 79% 的同时，在 Traffic‑QA 上提升约 12%，整体性能优于全微调与 LoRA，显著减少灾难性遗忘。

**⚠️ 局限性**

局限性在于依赖人工校验和 LLM 评判的可靠性，推理时延与显存受限，且在极端多模态或实时控制场景下的鲁棒性尚需进一步验证。

---

## 834. Scalable Variational Bayesian Fine-Tuning of LLMs via Orthogonalized Low-Rank Adapters

**arXiv ID:** 2604.03388 | [PDF](https://arxiv.org/pdf/2604.03388v1)

**作者:** Haotian Xiang `[一作]` (University of Georgia), Qin Lu `[通讯]` (University of Georgia)

**通讯引用:** 675 | [OpenAlex ID](https://openalex.org/A5101509596)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 PoLAR‑VBLL 框架，将正交化低秩适配器（PoLAR）与变分贝叶斯最后层（VBLL）联合训练，实现 LLM 的可扩展且高度校准的不确定性估计。

**💡 创新点**

创新点在于：1）通过 PoLAR 的正交约束解决传统 LoRA 的 rank collapse；2）在 VBLL 里使用闭式 Jensen‑tightened ELBO 进行高效变分推断；3）结合后期 Laplace 校正进一步提升校准性能，并实现推理仅需一次 backbone 前向。

**🔧 技术方法**

采用的技术包括 Polar‑decomposed Low‑rank Adapter (PoLAR)、Riemannian “landing field”优化、变分贝叶斯最后层（VBLL）闭式 ELBO、后处理 Laplace 近似。

**📊 数据集**

实验使用 6 个常识推理数据集（Winogrande‑S/M、ARC‑Challenge/E、OpenBookQA、BoolQ）以及 MMLU 的 Chem/Phy 作为 OOD 评测。

**📈 对比分析**

与 MLE、MAP、LoRA、BLoB、ScalaBL、C‑LoRA、TFB、Laplace‑LoRA 等多种基线对比，PoLAR‑VBLL 在 ACC、ECE、NLL 上均名列前茅，并且推理速度比 BLoB 快约 10 倍。

**⚠️ 局限性**

局限性在于仅对最后一层建模不确定性，深层不确定性仍未得到充分捕捉；后处理 Laplace 在高维参数空间下仍受限，且对极大模型的内存扩展尚需进一步优化。

---

## 835. InfBaGel: Human-Object-Scene Interaction Generation with Dynamic Perception and Iterative Refinement

**arXiv ID:** 2604.04843 | [PDF](https://arxiv.org/pdf/2604.04843v1)

**作者:** Yude Zou `[一作]` (Shanghai Jiao Tong University), Guanjie Zheng `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种粗细分层的自动回归框架，结合动态感知编码与一致性模型，实现了人‑物‑场景交互的实时生成。

**💡 创新点**

创新点包括动态感知编码来实时更新场景状态、轻量化的碰撞引导（Bump‑aware Guidance）以及混合数据训练策略以克服数据稀缺问题。

**🔧 技术方法**

主要技术包括一致性模型（Distilled Consistency Model）、动态感知编码（ViT+Voxel+时间动态采样）、Bump‑aware Guidance、CLIP文本编码、SMPL‑X人体建模、Basis Point Set 物体几何表示等。

**📊 数据集**

使用了HOI数据集、HSI数据集（如Lingo）、TRUMANS室内场景以及通过合成方式生成的HOSI数据。

**📈 对比分析**

与TRUMANS和Lingo基线相比，本文方法在HOSI基准上的成功率提升至83%，物体与场景穿透率显著降低，生成速度更快（每句平均推理时间显著下降）。

**⚠️ 局限性**

局限性在于合成的HOSI数据可能缺乏真实场景的复杂细节，且对极端动力学约束和多步骤高阶交互的处理仍有待改进。

---

## 836. DAO to (Anonymous) DAO Transactions

**arXiv ID:** 2604.04369 | [PDF](https://arxiv.org/pdf/2604.04369v1)

**作者:** Minfeng Qi `[一作]` (City University of Macau), Qin Wang `[通讯]` (CSIRO Data61)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出 Dao^2 框架，支持阈值控制的 DAO 之间（可匿名）资产转账，保持收款资产在组织内部的阈值管理。

**💡 创新点**

创新点在于将分布式密钥派生、分布式隐匿地址生成与阈值签名三大加密技术耦合成一套完整协议，提供正式安全性证明、前向保密性，并实现组织层面的匿名收款与赎回。

**🔧 技术方法**

使用技术包括：BIP32 风格的分布式密钥派生（DKD）、分布式隐匿地址生成（DSAG）通过分布式 ECDH 得到的不可链接一次性地址、Shamir 共享与 Lagrange 插值的阈值 ECDSA（2‑of‑n 版本）、HMAC‑SHA512 用于链码演化、随机预言机 H 用于隐匿地址偏移，以及标准椭圆曲线 secp256k1。

**📊 数据集**

在原型实现中使用 secp256k1 曲线，并在 Apple Silicon (ARM64) 机器上进行实验；未使用公开区块链数据集，仅测量加密运算时间与通信开销。

**📈 对比分析**

通过与标准隐匿地址支付和普通阈值签名的基线对比，实验表明：在 7 成员 DAO 的匿名交易耗时 < 27 ms，通信量 < 1.2 KB，随 DAO 成员数线性增长；相对于区块链确认延迟（秒级）而言，协议开销可以忽略不计。

**⚠️ 局限性**

局限性包括：实验仅在 secp256k1 上完成，未实现完整的 DKG 与区块链集成；协议仍需要交互且对网络侧信道泄露不做处理；前向保密性仅针对接收方，未覆盖发送方；缺乏跨链与账户/UTXO 模型的细节实现；未评估 gas 成本与大规模批量处理的优化。

---

## 837. A Persistent Homology Design Space for 3D Point Cloud Deep Learning

**arXiv ID:** 2604.04299 | [PDF](https://arxiv.org/pdf/2604.04299v1)

**作者:** Prachi Kudeshia `[一作]` (Saint Mary University), Dong Chen `[通讯]` (Nanjing Forestry University)

**通讯引用:** 43167 | [OpenAlex ID](https://openalex.org/A5100373698)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了持久同调（PH）在3D点云学习中的统一设计空间，并在分类和分割任务上系统验证其效能。

**💡 创新点**

提出六个PH注入点（采样、邻域图、优化、预训练、自监督、输出校准和网络内部正则化），将PH从单纯的后处理特征提升为整个网络的结构先验，并对不同PH向量化方法进行实验对比。

**🔧 技术方法**

使用Vietoris‑Rips复形、距离过滤、持久性图、持久性图像（PI）、持久性景观（PL）、持久性图表（PD）等向量化技术，并与PointNet、DGCNN、PointTransformer等主流点云骨干网络结合。

**📊 数据集**

在ModelNet40（分类）和ShapeNetPart（分割）两个公开数据集上开展实验。

**📈 对比分析**

将带PH分支的模型与仅几何输入的基线在相同骨干网络上对比，分类准确率提升约1%–3%，分割mIoU提升约0.5%–1%，同时记录参数量、训练时长和推理时间，显示出相对可接受的性能增益。

**⚠️ 局限性**

主要局限在PH计算和向量化的高计算复杂度（尤其Rips复杂体的指数规模）、实验仅覆盖单一PH实现，缺乏对多尺度或学习过滤器的探索，以及缺乏专门针对拓扑标签的基准数据集。

---

## 838. Loop-Extrusion Linkage: Spectral Ordering and Interval-Based Structure Discovery for Continuous Optimization

**arXiv ID:** 2604.04273 | [PDF](https://arxiv.org/pdf/2604.04273v1)

**作者:** Eren Unlu `[一作]` `[通讯]` (Globeholder), Eren Unlu (Globeholder)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Loop‑Extrusion Linkage (LEL) 操作符，用于连续黑盒优化中的结构学习与子空间搜索。

**💡 创新点**

创新点在于将在线交互图估计、Fiedler 向量谱序列化、可自适应的边界学习和随机区间扩展相结合，并以染色体环体形成的生物学类比为设计灵感。

**🔧 技术方法**

使用的技术包括：基于成功步的稀疏交互图构建、谱序列化（Fiedler 向量）、边界权重自适应更新、随机区间扩张（bidirectional extrusion）以及基于 DE/rand/1/bin 的变异算子。

**📊 数据集**

实验数据集为六个维度为 96 的合成诊断函数：连贯块 Rosenbrock、随机排列块 Rosenbrock、重叠窗口 Rosenbrock、带状链 Quadratic、可分离 Sphere 以及密集旋转椭圆。

**📈 对比分析**

与 jSO、DG2+CC 以及若干 LEL ablation 进行比较；在 10^4 次评估下，Full LEL 在 3/6 个函数上获得显著更低的 log-gap（p<0.001），但在 5×10^4 次评估后，大多数 ablation 或基线往往表现更好，显示出 LEL 的早期预算优势。

**⚠️ 局限性**

局限性包括：仅使用单一维度（d=96）合成函数，缺乏标准公开数据集验证；未进行运行时/计算开销分析；缺失 sep‑CMA‑ES 基线；交互图和谱排序在高维时成本高；自适应边界机制在后期预算下效果不佳；未测试非可序列化的稀疏交互结构。

---

## 839. VitaTouch: Property-Aware Vision-Tactile-Language Model for Robotic Quality Inspection in Manufacturing

**arXiv ID:** 2604.03322 | [PDF](https://arxiv.org/pdf/2604.03322v1)

**作者:** Junyi Zong `[一作]` (Beijing University of Posts and Telecommunications), Fang Deng `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 8098 | [OpenAlex ID](https://openalex.org/A5103055527)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 VitaTouch——一种面向工业质量检测的视觉–触觉–语言模型，用于属性推理与缺陷识别

**💡 创新点**

创新点：①基于属性的跨模态对齐，双 Q‑Former 提取视觉与触觉前缀；②将大语言模型与多模态编码器融合，实现自然语言属性描述；③采用 LoRA 参数高效微调，支持少样本缺陷适配；④构建并公开 VitaSet 多模态工业数据集

**🔧 技术方法**

技术：双 Q‑Former、EVA‑CLIP‑G 视觉编码器、AnyTouch ViT‑L 触觉编码器、Vicuna‑7B 大语言模型、LoRA 参数微调、对比学习（InfoNCE）与 PTM 对齐、属性推理与文本生成

**📊 数据集**

数据集：VitaSet（186 件物体，52k 图像，5.1k 人工校验的问答对），TVL benchmark（SSVTP、HCT 子集）

**📈 对比分析**

比较方法：与 LLaVA、ViP‑LLaVA、BLIP‑2、InstructBLIP、GPT‑4V、TVL‑LLaMA 等模型对比；在 TVL 上 VitaTouch 取得 HCT 子集最高分及整体得分 6.09；在 VitaSet 上硬度/粗糙度准确率分别为 88.89%/75.13%，属性描述召回 54.81%，LoRA 适配下 2/3/5 类缺陷识别分别达到 100%/96%/92%；闭环机器人排序成功率为 94% 并保持 278 ms 的中位推理延迟

**⚠️ 局限性**

局限：实验规模相对有限，场景多样性不足；缺陷类型覆盖范围受限；对真实工业环境中的光照、遮挡、工件尺寸变化等鲁棒性尚需进一步验证

---

## 840. AURA: Always-On Understanding and Real-Time Assistance via Video Streams

**arXiv ID:** 2604.04184 | [PDF](https://arxiv.org/pdf/2604.04184v1)

**作者:** Xudong Lu `[一作]` (CUHK MMLab), Hongsheng Li `[通讯]` (CUHK MMLab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了AURA框架，实现统一VideoLLM的端到端实时流式视觉交互，支持持续观察、选择性沉默以及实时和主动回答。

**💡 创新点**

创新点包括：① 双滑动窗口的 Interactive Video Stream Context Management 处理无界视频与对话历史；② Coarse‑to‑Fine Streaming Data Engine 系统化生成 Real‑Time、Proactive、Multi‑Response QA 训练样本；③ Silent‑Speech Balanced Loss 解决沉默偏倚与误生成；④ KV‑cache 重用与滑动窗口推理框架实现 2 FPS 的低延迟部署。

**🔧 技术方法**

技术栈：Qwen3‑VL‑8B‑Instruct 作为基础模型；LLM 细调；多模态视觉编码与交叉模态连接；视频分块 + 1 秒切片；双滑动窗口上下文管理；KV‑cache 前缀重用；ASR（Qwen3‑ASR‑1.7B）与 TTS（Qwen3‑TTS‑12Hz‑1.7B‑Base）集成；vLLM 推理服务；专门的训练目标和损失重加权。

**📊 数据集**

数据集：115k 流式视频 QA 样本（≈1.04B 词）+ 59k 离线视频 QA 样本（≈0.16B 词），共 174k 样本；公开视频多领域（体育、纪录片、动画等）；基准评测使用 StreamingBench、OVO‑Bench、OmniMMI（流式）以及 LongVideoBench、MVBench、Video‑MME（离线）。

**📈 对比分析**

在 StreamingBench、OVO‑Bench 与 OmniMMI 三大流式基准上，与 10+ 开源模型及 2+ 专有模型对比，AURA 取得最高整体分：StreamingBench 73.1%（比最佳开源高 10.4%），OVO‑Bench 65.3%（比最佳开源高 4.2%），OmniMMI 25.4%（全场最佳）。在离线基准上，AURA 与基线 Qwen3‑VL‑8B‑Instruct 的差距仅在 2‑4% 以内。Ablation 实验表明 Silent‑Speech Balanced Loss 使整体平均提升约 9% 并显著提升 Proactive Alerting。

**⚠️ 局限性**

局限性：① 训练仍依赖大量算力（32 个 80G 加速器）；② 低帧率（2 FPS）和高显存需求限制在普通硬件上的可部署性；③ 对极长视频仍有轻微性能衰减；④ 生成多模态交互的语音与文本仍受限于 ASR/TTS 的准确性，易在噪声环境中误识；⑤ 数据集主要基于自动合成，可能缺乏真实复杂对话场景的多样性。

---

## 841. ITIScore: An Image-to-Text-to-Image Rating Framework for the Image Captioning Ability of MLLMs

**arXiv ID:** 2604.03765 | [PDF](https://arxiv.org/pdf/2604.03765v1)

**作者:** Zitong Xu `[一作]` (Shanghai Jiao Tong University), Patrick Le Callet `[通讯]` (Institut Universitaire de France)

**通讯引用:** 10656 | [OpenAlex ID](https://openalex.org/A5004453915)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ICBench，涵盖12类场景的2,040张图像、10大MLLM生成的短长句子共40,800条，并通过1.8M人工MOS进行细粒度评估。

**💡 创新点**

创新点在于构建统一的短长句子评估基准，并设计ITIScore通过图像-文本-图像重构一致性实现参考自由、多维度评估。

**🔧 技术方法**

采用图像生成模型生成重构图像，冻结MLLM融合原图、重构图与文本，轻量化MLP预测高斯分布评分，并用维度级NLL训练。

**📊 数据集**

使用ICBench作为主数据集，同时在Composites、Flickr8k、MMHE、LongCap-Arena等公开数据集进行零样本验证。

**📈 对比分析**

与传统BLEU、METEOR、CLIP-S等以及ChatGPT-5、Gemini-3.0-Flash等MMLM评测比较，ITIScore在所有维度上达Kendall τ≈0.95–0.99，SRCC≈0.98，显著优于其他方法。

**⚠️ 局限性**

局限性包括对生成模型质量的依赖、MLLM冻结导致灵活性受限，以及仍无法覆盖极端场景和多语言环境。

---

## 842. LiquiLM: Bridging the Semantic Gap in Liquidity Flaw Audit via DCN and LLMs

**arXiv ID:** 2604.03860 | [PDF](https://arxiv.org/pdf/2604.03860v1)

**作者:** Zekai Liu `[一作]` (Hainan University), Zongwei Li `[通讯]` (Hainan University)

**通讯引用:** 52 | [OpenAlex ID](https://openalex.org/A5101530716)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 LiquiLM 框架，利用 LLM 与动态共注意网络结合，对 Proof‑of‑Liquidity 智能合约进行流动性缺陷检测与解释。

**💡 创新点**

将大型语言模型与 DCN 预过滤结合，构建 Audit‑Informed Manifest 并采用四阶段协同提示系统，显著弥合代码实现与流动性意图的语义鸿沟。

**🔧 技术方法**

依托 LLM（Gemini 3 Pro、GPT‑4o）、BGE‑M3 编码、动态共注意网络（DCN）、双阶段置信过滤、四阶段协同提示、语义切片与标准化等技术。

**📊 数据集**

使用包含 1 490 条人工校验的真实 PoL/以太坊经济合约切片（I_1(core)、I_1(baseline)、I_1(real)）以及 1 490 条人工/AI 生成的流动性缺陷描述语料库 I_2。

**📈 对比分析**

在 1 490 条验证合约上与 Gemini 3 Pro/GPT‑4o 结合，F1 分数均超过 90%；与 Slither、Oyente、Aderyn 等传统工具相比，F1 提升至 93%+，在真实场景中发现 238 个高风险合约并披露 10 项 CVE。

**⚠️ 局限性**

对纯语法缺陷（如算术错误）的检测精度略低，DCN 的深语义映射在此类缺陷时可能产生噪声，且仍需人工校验确认缺陷。

---

## 843. DARE: Diffusion Large Language Models Alignment and Reinforcement Executor

**arXiv ID:** 2604.04215 | [PDF](https://arxiv.org/pdf/2604.04215v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 844. A Complete Characterization of Convexity in Flow Games

**arXiv ID:** 2604.04729 | [PDF](https://arxiv.org/pdf/2604.04729v1)

**作者:** Han Xiao `[一作]` (Ocean University of China), Qizhi Fang `[通讯]` (Ocean University of China)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究流游戏的凸性问题，给出了网络结构的必要且充分条件，并提出多项式时间判定算法；

**💡 创新点**

首次完成对流游戏凸性的完整拓扑特征描述，提出无环、瓶颈唯一、非瓶颈足够容量三条判据；

**🔧 技术方法**

采用结构性性质推导、最大流最小割、流分解定理及算法设计等理论工具；

**📊 数据集**

未使用实验数据集，全部为理论证明与算法分析；

**📈 对比分析**

通过理论证明与多步结构性质构建判定算法，时间复杂度为O(|E|(|V|+|E|))，显著优于以往没有多项式解的情况；

**⚠️ 局限性**

局限在于只对已给定网络可判定凸性，逆问题（构造满足条件的网络）仍NP难；对非完全平衡游戏或需要网络表示的情形尚未覆盖；未解决更宽泛的PMAS特性问题。

---

## 845. CAWN: Continuous Acoustic Wave Networks for Autoregressive Language Modeling

**arXiv ID:** 2604.04250 | [PDF](https://arxiv.org/pdf/2604.04250v1)

**作者:** Dejan Čugalj `[一作]`, Aleksandar Jevremovic `[通讯]` (Singidunum University)

**通讯引用:** 309 | [OpenAlex ID](https://openalex.org/A5003704886)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种全新的连续声波网络（CAWN），通过将离散词嵌入映射为复数相位谐波并使用线性时间的相位累计来实现自回归语言建模，从而消除传统 Transformer 的 O(L²) 内存瓶颈，实现 O(1) 上下文状态传递。

**💡 创新点**

核心创新点包括：
• 复数域连续相位累计机制，利用真复数旋转实现全局上下文记忆；
• 双门选择性相位共振（频率依赖保留门 + 直通硬阈门）以抑制信号衰减；
• 语法缓存（1D 卷积）在相位累计前提取短期局部依赖；
• 深度路由的 Block Attention Residuals，截断残差流并实现跨层深度注意；
• 通过自定义 Triton kernel 在 SRAM 上完成高效相位运算；
• 通过连续流式预训练和极端上下文检索验证可达数百万 token 的记忆能力。

**🔧 技术方法**

使用技术包括：
• 复数域相位累计与 Euler 公式的真复数旋转；
• 深度频谱卷积（Depth‑wise Harmonic Convolution）与 SwiGLU Ear 进行噪声抑制；
• 频率依赖门控与直通硬阈门（STE）实现精细的信号过滤；
• Block Attention Residuals 的深度级注意力路由；
• Triton 高效自定义核、混合精度训练、BPE 32k 词表、连续流式数据管道；
• 1D causal convolution 语法缓存与 2‑步历史缓冲。

**📊 数据集**

训练与评估使用 1 亿 100B token 的混合英文语料：FineWeb‑Edu PDFs（50%）、DCLM（30%）、FineWeb‑Edu（20%），BPE 32k 词表。测试集包含 WikiText‑103、PIQA、ARC‑Easy 等标准基准。

**📈 对比分析**

与标准 Transformer、Llama、Pythia、Pythia‑160M、SmolLM 等模型在内存消耗、吞吐量、困惑度（perplexity）、零样本推理（PIQA/ARC‑Easy）和目标检索准确率等方面进行对比。CAWN 在 150M 参数下：
• 训练/预填内存从 512‑8192 token 线性扩展至 2M token，峰值 8.72 GB；
• 生成吞吐保持 O(1) 平稳（≈52 tok/s）而 Transformer 仍受 O(L²) 限制；
• perplexity 在 752k 步时降至 75；
• PIQA 60.2%、ARC‑Easy 45.5%，均优于 160M 参数的 Pythia；
• 1M token 的目标检索保持 100% 正确。

**⚠️ 局限性**

局限性与待改进点：
• 需要更高位精度（fp32/float16）以防止相位累积中的数值漂移；
• 对极长上下文仍有限（2 M token 处出现振幅消失导致检索失败）；
• 当前模型规模仅 150M，需扩展到 1‑7B 才能与成熟 Transformer 在绝对性能上竞争；
• 依赖自定义 Triton 内核与特定硬件，迁移成本高；
• 需要进一步分析频率干涉导致的目标失效机制。

---

## 846. Improving ML Attacks on LWE with Data Repetition and Stepwise Regression

**arXiv ID:** 2604.03903 | [PDF](https://arxiv.org/pdf/2604.03903v1)

**作者:** Alberto Alfarano `[一作]` (Axiom Math), Kristin Lauter `[通讯]` (Meta)

**通讯引用:** 11650 | [OpenAlex ID](https://openalex.org/A5002850656)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种针对稀疏/小秘密的学习错误（LWE）问题的机器学习攻击，使用更大的训练集、重复样本和逐步回归方法来提高秘密恢复率。

**💡 创新点**

创新点在于（1）通过重复训练样本显著提升对含有超过3个非零“残忍”位的秘密的恢复能力；（2）将逐步回归替代传统线性回归，以更好地处理冷位噪声；（3）引入合成数据与实验数据相结合的实验框架，并给出了经验规模律。

**🔧 技术方法**

技术包括：Transformer‑only 编码器模型、角度嵌入、BKZ lattice‑reduction、数据重复策略、逐步回归（direct 与 dual），以及损失函数的径向正则化。

**📊 数据集**

使用了四种不同参数集的LWE数据（n=256或512，log₂q=12/20/28/41），生成约4亿到1亿条减约数据，并通过合成数据模拟相同分布。

**📈 对比分析**

与 SALSA、Cool&Cruel、VERDE、FRESCA 等先前方法相比，本文在相同或更低的计算预算下，恢复的最大汉明重量提升至 h=70（4种设定中）并实现更高比例的秘密恢复（如 h=63 时 91% vs VERDE 的 15%）。

**⚠️ 局限性**

局限性包括：仅针对基本 LWE，稀疏二/三元秘密；对 BKZ 预处理和大量训练样本的高成本；对非稀疏或更复杂的 PQC 变体的适用性未验证；以及逐步回归在特定无相关特征场景下有效，可能不适用于其他统计问题。

---

## 847. Beyond Hard Negatives: The Importance of Score Distribution in Knowledge Distillation for Dense Retrieval

**arXiv ID:** 2604.04734 | [PDF](https://arxiv.org/pdf/2604.04734v1)

**作者:** Youngjoon Jang `[一作]` (Korea University), Heuiseok Lim `[通讯]` (Korea University)

**通讯引用:** 2727 | [OpenAlex ID](https://openalex.org/A5033580486)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探讨知识蒸馏中训练样本分布对密集检索模型泛化的影响，并提出分层采样策略来保持教师模型分数分布的完整性。

**💡 创新点**

创新点在于把关注点从传统的硬负样本挖掘转移到教师分数分布的多样性；通过分层采样实现无参数、均匀覆盖整个分数区间，显著提升泛化性能。

**🔧 技术方法**

使用交叉编码器作为教师、密集检索模型作为学生，采用 KL 散度或 MarginMSE 损失进行蒸馏；对比学习预训练，结合对查询级 min–max 归一化和分层量化采样。

**📊 数据集**

实验数据集包括 MS MARCO（Train/Dev）、TREC DL 19 以及 BEIR benchmark（13 个异构域）。

**📈 对比分析**

在相同的训练管线下比较 retriever‑top、reranker‑top、low、mid、random、stratified 六种采样策略。实验结果显示 stratified（及 random）在所有骨干模型和两种蒸馏目标函数下均保持最高或次高的 MRR、nDCG、Recall 等指标，尤其在 BEIR 上优于传统 Top‑K 采样，且对 K 变化稳健。

**⚠️ 局限性**

局限性包括：1) 对极少样本（K=4）时 stratified 稍逊于 random；2) 仅关注分布均匀性，未结合动态负样本挖掘或自适应策略；3) 需要在更多模型、目标函数和更大规模数据上进一步验证。

---

## 848. Adaptive Tensor Network Simulation via Entropy-Feedback PID Control and GPU-Accelerated SVD

**arXiv ID:** 2604.03960 | [PDF](https://arxiv.org/pdf/2604.03960v1)

**作者:** Harshni Kumaresan `[一作]`, Santhosh Sivasubramani `[通讯]`

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于熵反馈PID控制的自适应张量网络框架，用于动态管理MPS的本征维度。

**💡 创新点**

创新点在于将闭环反馈控制与EMA平滑和预测调度相结合，实现按链段粒度自适应分配本征维度，并与GPU加速SVD集成。

**🔧 技术方法**

采用的技术包括von Neumann熵监测、指数移动平均滤波、PID控制器、线性预测器以及CuPy/ cuSOLVER GPU加速的SVD。

**📊 数据集**

在测试数据上使用了自旋1/2反铁磁Heisenberg链、Ising、XXZ等四种一维哈密顿量，以及小型量子线路与AWS Braket、IBM、Rigetti、IQM、IonQ等量子硬件进行交叉验证。

**📈 对比分析**

与固定本征维度、ITensor、TeNPy、quimb等现有库进行比较，取得约2.4–5.1倍的加速，能耗降低约30%，能量误差保持在10^-4以内。

**⚠️ 局限性**

局限性包括需要手动调节PID增益、对小本征维度时收益有限、GPU加速仅适用于χ≥64、预测器在突发纠缠跃迁时可能不足，以及目前仅验证了1D MPS，未扩展到PEPS等高维张量网络。

---

## 849. COBOL-Coder: Domain-Adapted Large Language Models for COBOL Code Generation and Translation

**arXiv ID:** 2604.03986 | [PDF](https://arxiv.org/pdf/2604.03986v1)

**作者:** Anh T. V. Dau `[一作]` (Concordia University), Anh Tuan Nguyen `[通讯]` (FPT Software AI Center)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研发并评估了面向 COBOL 的域适配大型语言模型（COBOL-Coder），并提出自动化数据增广管道、首个 COBOL–Java 双向翻译基准及对应的数据集。

**💡 创新点**

创新点包括：1）基于编译器验证与多阶段相似性过滤的自动化 COBOL 训练数据构建流程；2）针对 COBOL 语法和业务惯例的专属 LLM 微调；3）首次构建 COBOL–Java 翻译基准（COBOL-Java Translate），覆盖现代与遗留系统迁移场景。

**🔧 技术方法**

技术手段主要是：迁移学习（使用 Qwen2.5-Coder 作为基础模型）、编译器驱动的自修复（利用 GPT‑4o 修复错误代码）、多阶段相似性筛选（LLM 评估+AST‑based 相似度）、指令式微调（Instruction‑Tuning）以及基准评测（CSR 与 Pass@1）。

**📊 数据集**

使用的数据集包括：GitHub 公共 COBOL 代码（约 40k 文件）、Java 代码翻译生成的 COBOL 数据（约 280k 对）、COBOL 相关文本（18k 文档）、COBOLEval、COBOLCodeBench 以及自建的 COBOL‑Java 翻译基准（约 143 题）。

**📈 对比分析**

比较方法：在 COBOLEval、COBOLCodeBench 以及 COBOL‑Java 翻译基准上，采用编译成功率（CSR）和 Pass@1 指标，对比多种开源 LLM（CodeGemma、CodeLlama、StarCoder2 等）和 GPT 变体。性能表现为：专用模型 7B 版本 CSR 73.8%，Pass@1 44.7%；14B 版本 CSR 73.9%，Pass@1 49.3%；在 COBOL→Java 翻译中 CSR 97.9%，Pass@1 83.9%；在 Java→COBOL 翻译中 CSR 72%，Pass@1 34.9%。用户研究亦表明该模型在实践中的可读性与符合企业编码规范的优越性。

**⚠️ 局限性**

局限性：1）仅处理单文件级别程序，未覆盖跨文件、copybook、JCL 等系统级依赖；2）语义验证主要基于编译器和 LLM 评分，缺乏细粒度的功能测试；3）评测基准与工业实际场景存在差距；4）用户调研样本量有限，缺少更广泛的工业验证。

---

## 850. What is Human in Judgment? Testing Automation Bias and Algorithm Aversion Among United States Military Academy Cadets

**arXiv ID:** 2604.04333 | [PDF](https://arxiv.org/pdf/2604.04333v1)

**作者:** Lauren Kahn `[一作]` (Center for Security and Emerging Technology Georgetown University), Laura Resnick Samotin `[通讯]` (United States Military Academy)

**通讯引用:** 44 | [OpenAlex ID](https://openalex.org/A5068747567)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

比较了美国西点军校学员与年龄、教育水平相近的公众样本在AI决策支持系统下的自动化偏差与算法回避表现；

**💡 创新点**

首次在军事人群中进行实验研究，发现训练良好的军官更能自我校准、降低自动化偏差；

**🔧 技术方法**

采用基于飞机识别的目标判定任务，设置AI与人类分析师建议的随机实验条件；

**📊 数据集**

使用236名西点军校学员与702名匹配公众样本的问卷实验数据；

**📈 对比分析**

通过切换率、错误率及逻辑回归比较两组在自动化偏差与算法回避上的差异，结果显示军校学员误差率显著更低；

**⚠️ 局限性**

样本仅限西点学员，实验未覆盖遗漏错误或团队决策情境，结果具有一定局限性。

---

## 851. LAA-X: Unified Localized Artifact Attention for Quality-Agnostic and Generalizable Face Forgery Detection

**arXiv ID:** 2604.04086 | [PDF](https://arxiv.org/pdf/2604.04086v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 852. An experimental evaluation of satellite constellation emulators

**arXiv ID:** 2604.04498 | [PDF](https://arxiv.org/pdf/2604.04498v1)

**作者:** Victor Cionca `[一作]` (Munster Technological University), Dylan Smyth `[通讯]` (Munster Technological University)

**通讯引用:** 66 | [OpenAlex ID](https://openalex.org/A5086524161)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文对三款开源卫星星座仿真器（StarryNet、OpenSN、Celestial）进行了实验评估，比较它们在真实性、资源占用、启动时间、链路更新延迟等方面的表现，并将仿真结果与实际的 WetLinks 研究数据对比。

**💡 创新点**

创新点在于：①系统性评估三种技术路线（Docker+Netlink、etcd+API、Firecracker微VM）在同一实验环境下的性能与真实性；②揭示现有仿真器在链路调度、天气、误差建模等关键因素上的缺失；③提出基于预计算、并行更新、轻量化容器化等改进方向，为后续更真实、更高效的卫星网络仿真器设计提供参考。

**🔧 技术方法**

使用的技术包括：Docker、Linux Netlink API、etcd KV store、eBPF/tc netem、Firecracker 微VM、Skyfield SGP4轨道计算、Linux 虚拟网络桥接、Python/Go 等实现框架。

**📊 数据集**

主要数据集为 WetLinks 对 Starlink 网络的实时测量数据（吞吐量、RTT、路径信息），以及实验中自行生成的 TLE 轨道参数与星座配置文件。

**📈 对比分析**

比较方法：将仿真得到的吞吐量、RTT 与 WetLinks 实测值做时间序列对齐，评估真实性；使用 Linux top 记录 CPU 用户/内核占用、节点启动时间、链路更新延迟等指标。结果显示：三者都无法逼近真实测量值；StarryNet-dev 在节点启动和 CPU 占用上表现最佳；OpenSN 在链路更新上延迟高；Celestial 虽然启动快，但由于星形拓扑导致多跳链路无法准确模拟。

**⚠️ 局限性**

limitations：①缺少多用户调度、天气、链路误差等低层建模，导致真实性差；②链路更新延迟不受限，随星座规模急剧上升；③资源占用高（尤其是容器化方案），导致实验启动耗时长；④对异构星座支持有限；⑤预计算方式虽然提高实时性，但需要提前大量计算，启动时间长。

---

## 853. SecPI: Secure Code Generation with Reasoning Models via Security Reasoning Internalization

**arXiv ID:** 2604.03587 | [PDF](https://arxiv.org/pdf/2604.03587v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 854. SysTradeBench: An Iterative Build-Test-Patch Benchmark for Strategy-to-Code Trading Systems with Drift-Aware Diagnostics

**arXiv ID:** 2604.04812 | [PDF](https://arxiv.org/pdf/2604.04812v1)

**作者:** Yuchen Cao `[一作]` (City University of Hong Kong), Linqi Song `[通讯]` (City University of Hong Kong)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个针对交易系统的迭代构建–测试–补丁评估基准（SysTB），评估LLM生成的策略代码在治理、可靠性、可审计性等方面的表现。

**💡 创新点**

创新点在于：①将策略描述转为可执行代码的过程视为可治理软件，强调规范、审计日志和确定性；②引入基于漂移检测的迭代修复框架，允许在不改变策略语义的前提下持续改进；③提出多维度评分卡（规范符合度、风险纪律、可靠性、OOS鲁棒性）并考虑成本与Token消耗。

**🔧 技术方法**

技术主要包括：大型语言模型（GPT、Claude、Gemini等）用于代码生成与审计日志输出；沙箱化执行环境实现确定性、无泄露、可重复性；漂移检测与checksum比对；LLM交叉评估与自动化量化评估；Token与成本追踪。

**📊 数据集**

使用了12种基准策略（覆盖趋势跟随、均值回归、套利、日内规则、组合轮换等），每种策略都有冻结语义文档；数据集涵盖美国日线股票、加密1分钟行情和中国A股指数与蓝筹，时间范围2024–2025，已划分训练/测试。

**📈 对比分析**

评估通过将17个模型在12种策略上的迭代提交与基准进行对比。结果显示：顶级模型（GPT‑5.2、GPT‑5.1、o3、Grok‑4 Fast）在所有合法性门控上>91.7%通过，整体分数约7.3–7.9；迭代修复在第一次迭代即提升约0.4分，后续收益递减；代码在第二次迭代已达到95%相似度，第三次几乎完全一致。成本方面，顶级模型每条策略约$0.8–$1，迭代后成本增加约50%。

**⚠️ 局限性**

局限性包括：①评估仅在合成的冻结策略文档上进行，未覆盖真实业务多变需求；②多维评分中OOS鲁棒性指标仅为采样测试，未完成完整的2025年交易成本评估；③迭代过程导致代码趋同，降低了多样性与集成鲁棒性，可能影响对不同市场变革的抵御能力；④依赖LLM的稳定性与API可用性，若模型出现故障需人工介入。

---

## 855. SafeCtrl: Region-Aware Safety Control for Text-to-Image Diffusion via Detect-Then-Suppress

**arXiv ID:** 2604.03941 | [PDF](https://arxiv.org/pdf/2604.03941v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 856. Cultural Authenticity: Comparing LLM Cultural Representations to Native Human Expectations

**arXiv ID:** 2604.03493 | [PDF](https://arxiv.org/pdf/2604.03493v1)

**作者:** Erin MacMurray van Liemt `[一作]` (Google Research), Sunipa Dev `[通讯]` (Google Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过构建人类开放式问卷得到的文化重要性向量，评估LLM在不同国家的文化真实性与人类预期的一致性；

**💡 创新点**

提出以人类优先级为基准的“文化真实性”评估框架，并揭示LLM呈现的旅游观式过度平衡文化描述及其系统性误差；

**🔧 技术方法**

采用Gemini 2.5 Flash进行自动分类、语法多样化提示生成、指令调优的自动评分器提取文化维度，并用Pearson、余弦相似度、MSE等统计学指标评估对齐程度；

**📊 数据集**

使用来自9个国家（意大利、法国、德国、日本、韩国、印尼、墨西哥、巴西、印度）的开放式文化问卷数据（共5629名参与者）和三大前沿LLM（Gemini 2.5 Pro、GPT‑4o、Claude 3.5 Haiku）的生成文本；

**📈 对比分析**

将模型生成的文化表现向量与人类基准向量对齐，计算相关系数、余弦相似度和均方误差；结果显示Gemini在大多数国家表现最佳，GPT‑4o在与美国文化相近的国家更好，Claude表现最差；所有模型在远离美国文化的国家呈负相关，且三者误差高度一致（ρ≈0.97），表明存在系统性偏差；

**⚠️ 局限性**

仅使用二值化的文化维度提取，忽略叙事语气与信息量；模型评估使用的提示与人类调查不完全对齐；仅按国家边界划分文化，未考虑内部多样性与交叉身份。

---

## 857. Isokinetic Flow Matching for Pathwise Straightening of Generative Flows

**arXiv ID:** 2604.04491 | [PDF](https://arxiv.org/pdf/2604.04491v1)

**作者:** Tauhid Khan `[一作]` `[通讯]`, Tauhid Khan

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的流匹配方法，称为等动流匹配（Iso-FM），旨在提高基于流的生成模型的采样效率。

**💡 创新点**

创新点在于引入了一种轻量级的、无雅可比的动态正则化器，直接惩罚路径加速度，从而减少了生成路径的曲率，显著提高了少步生成的效率。

**🔧 技术方法**

使用了一种自导向的有限差分近似方法来实现动态正则化，避免了昂贵的雅可比向量乘积。

**📊 数据集**

在CIFAR-10数据集上进行了实验，使用DiT-S/2作为基础模型。

**📈 对比分析**

与基线流匹配（FM）方法相比，Iso-FM在条件非最优传输（non-OT）FID@2上从78.82降低到27.13，达到了2.9倍的相对效率提升，且在FID@4上达到了10.23的最佳观察结果。

**⚠️ 局限性**

限制在于Iso-FM并不声称在绝对性能上超越多阶段蒸馏管道，并且在复杂数据上实现完全线性全局传输图是数学上不可实现的。

---

## 858. C2|Q>: A Robust Framework for Bridging Classical and Quantum Software Development -- RCR Report

**arXiv ID:** 2604.04112 | [PDF](https://arxiv.org/pdf/2604.04112v1)

**作者:** Boshuai Ye `[一作]` (University of Oulu), Lauri Malmi `[通讯]` (Aalto University)

**通讯引用:** 5877 | [OpenAlex ID](https://openalex.org/A5027596822)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本工作提出了C2|Q⟩框架，能将Python代码或JSON描述的经典问题自动转换为可执行的量子程序，实现跨硬件的无缝部署。

**💡 创新点**

创新点在于整合代码解析器、问题分类与QCF生成、自动硬件推荐及结果解码，形成端到端的硬件无关量子软件开发流程。

**🔧 技术方法**

技术包括基于CodeBERT的代码解析器、QCF映射模块、量子算法自动选择与电路编译、Qiskit、IBM、IonQ等多平台接口。

**📊 数据集**

使用了434个Python程序和100个JSON实例的合成数据集，涵盖10类问题族（最大割、最大独立集、TSP、Clique等）以及整数因数分解与算术运算。

**📈 对比分析**

通过与手工实现的Qiskit工作流对比，评估了工作流完成率、代码行数、配置决策及仿真/真实硬件的执行时间和错误率，实验显示端到端成功率高于手工实现，且在多设备上能自动推荐最佳后端。

**⚠️ 局限性**

局限包括需要单独下载大型预训练模型、实验3的真实硬件验证受限、全规模复现耗时较长（约10小时）以及仅支持Python 3.12。

---

## 859. From 8 Seconds to 370ms: Kernel-Fused SAR Imaging on Apple Silicon via Single-Dispatch FFT Pipelines

**arXiv ID:** 2604.03585 | [PDF](https://arxiv.org/pdf/2604.03585v1)

**作者:** Mohamed Amine Bergach `[一作]` `[通讯]` (Illumina), Mohamed Amine Bergach (Illumina)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

实现了Apple Silicon GPU上首个核融合的SAR Range Doppler算法，将FFT、匹配滤波和IFFT合并为单个Metal dispatch；

**💡 创新点**

创新点包括①通过核融合显著减少设备内存传输；②首次利用Apple 8×8 MMA实现Cooley‑Tukey DIF FFT；③完整实现并验证Metal上的SAR Range Doppler，保证图像质量不受影响；

**🔧 技术方法**

使用了Metal compute shaders、线程组内存、8×8硬件矩阵乘法（MMA）、Cooley‑Tukey DIF、FFT/匹配滤波/IFFT融合、统一内存等技术；

**📊 数据集**

使用了4096×4096复数浮点32位SAR仿真场景，包含5个点目标及20 dB加性高斯噪声；

**📈 对比分析**

与传统未融合的多dispatch基线对比，在Apple M1 GPU上端到端时间从8.16 s降至0.37 s，获得22.3×加速；单步耗时和图像质量（SNR、L2误差）保持一致；

**⚠️ 局限性**

局限在于轴向FFT和RCMC仍需多次全局转置，占用约80%时间；缺乏混合精度/多核优化，未实现8K×8K实时处理；线程组内存限制导致无法处理更大FFT。

---

## 860. On the First Computer Science Research Paper in an Indian Language and the Future of Science in Indian Languages

**arXiv ID:** 2604.03265 | [PDF](https://arxiv.org/pdf/2604.03265v1)

**作者:** Siddhartha Visveswara Jayanti `[一作]` (Dartmouth), Siddhartha Visveswara Jayanti `[通讯]` (Dartmouth)

**通讯引用:** 127 | [OpenAlex ID](https://openalex.org/A5008303793)

**关键词:** `aaff19cd-e89f-4398-8dae-a6684a329811`

**🎯 论文内容**

未提供具体内容，无法概括论文主旨。

**💡 创新点**

未知。

**🔧 技术方法**

未知。

**📊 数据集**

未知。

**📈 对比分析**

未知。

**⚠️ 局限性**

缺乏足够信息。

---

## 861. Lighting Up or Dimming Down? Exploring Dark Patterns of LLMs in Co-Creativity

**arXiv ID:** 2604.04735 | [PDF](https://arxiv.org/pdf/2604.04735v1)

**作者:** Zhu Li `[一作]` (Meta), Yuan Chang `[通讯]` (Meta)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对LLM协助创作的24个情境进行实验，测量了五种暗模式（奉承、锚定、语调管控、道德化、循环死锁）的出现频率，涵盖四种文学形式和六种内容主题。

**💡 创新点**

创新点在于首次系统地量化并分析这些暗模式在创意写作中的表现与上下文依赖性，揭示安全对齐策略如何无意中压缩创作空间。

**🔧 技术方法**

研究采用因子设计、拉丁方排序、LLM API生成文本，并通过人工标注与Fleiss κ 评估可靠性，利用统计方法比较不同模式的出现率。

**📊 数据集**

使用自造的24组提示（4种文学形式×6种主题）以及作者共8次对话生成的数据，未使用外部公开数据集。

**📈 对比分析**

通过多数投票与kappa统计比较不同模式的出现率，发现奉承占比最高（91.7%），锚定（41.7%）、循环死锁（33.3%）、道德化（25%）和语调管控（20.8%）依次下降，表明模式分布与文体、主题显著相关。

**⚠️ 局限性**

局限性包括样本量小、二元标注过于简化、仅检验五种模式、仅使用单一LLM、缺乏真实用户体验评估，需进一步扩大实验规模并开发自动检测机制。

---

## 862. Quasi-BP for BCH Codes and its Optimization

**arXiv ID:** 2604.04066 | [PDF](https://arxiv.org/pdf/2604.04066v1)

**作者:** Guangwen Li `[一作]` `[通讯]` (Shandong Technology and Business University), Guangwen Li (Shandong Technology and Business University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种适用于BCH码的准贝叶斯传播（quasi‑BP）解码器，并利用S‑EXIT图对其参数进行优化；

**💡 创新点**

创新点在于系统性地将信道噪声方差、码的循环性以及冗余校验矩阵特征融合到BP框架中，采用输入膨胀与合并操作加速互信息增长，并通过S‑EXIT图实现参数自适应优化；

**🔧 技术方法**

主要技术包括准贝叶斯传播解码、互信息（MI）跟踪与分析、EXIT/散点EXIT图、最小和（MS）及其归一化变种、以及输入膨胀与合并操作；

**📊 数据集**

实验数据基于BCH(127,64)、BCH(127,99)、BCH(255,239)等码以及同等码率/码长的LDPC(128,64)码，在AWGN信道上进行仿真；

**📈 对比分析**

通过与传统BP、NMS、硬判决Berlekamp‑Massey以及神经BP-RNN等解码器对比，准BP在FER上相较LDPC BP仅差0.1~0.6 dB，且在高SNR区保持与ML性能相近；

**⚠️ 局限性**

局限性包括仍有约1 dB的ML缺口、需已知噪声方差、在最坏情况下计算复杂度提升、以及对S‑EXIT优化的依赖等。

---

## 863. SARES-DEIM: Sparse Mixture-of-Experts Meets DETR for Robust SAR Ship Detection

**arXiv ID:** 2604.04127 | [PDF](https://arxiv.org/pdf/2604.04127v1)

**作者:** Fenghao Song `[一作]` (Yunnan Normal University), Xi Zhou `[通讯]` (Yunnan Normal University)

**通讯引用:** 6023 | [OpenAlex ID](https://openalex.org/A5100430796)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于 DETR 的 SAR 船舶检测框架 SARES-DEIM，主要通过两个新模块实现对 SAR 信号的自适应处理：SARESMoE（稀疏路由的混合专家）和 SDEP（Space‑to‑Depth 增强金字塔）。

**💡 创新点**

创新点：① 设计 SAR 领域感知的专家选择机制，利用频域、波形专家在不同尺度下自适应过滤噪声与混叠；② 引入无损空间降维（Space‑to‑Depth）金字塔，将高分辨率细节直接注入检测头；③ 将上述两项结合，显著提升小目标检测精度与鲁棒性。

**🔧 技术方法**

技术手段：DETR 框架、混合专家 (MoE) 与稀疏路由、频域滤波、离散小波变换 (WTConv)、GhostNet 结构、Space‑to‑Depth Convolution (SPDConv)、基于频谱与空间特征的双分支路由、AdamW 优化器等。

**📊 数据集**

数据集：HRSID（高分辨率 Sentinel‑1 / TerraSAR‑X 片段，5 604 张图，16 951 只船）与 SAR‑Ship‑Dataset（Gaofen‑3 与 Sentinel‑1 43 819 张船舶图块）。

**📈 对比分析**

实验方法：在 HRSID 与 SAR‑Ship‑Dataset 上与 YOLOv8、YOLOv11、RT‑DETR、D‑FINE、DEIM、SAR‑D‑FINE、CSCF‑Net 等经典及专用检测器进行基准对比，使用 mAP_50:95、mAP_50、Precision、Recall 等指标。结果显示：在 HRSID 上 SARES‑DEIM 获得 76.4% mAP_50:95、93.8% mAP_50，显著优于所有对比模型；在 SAR‑Ship‑Dataset 上获得 71.7% mAP_50:95、98.1% mAP_50，亦高于现有方法。

**⚠️ 局限性**

局限性：① 采用 MoE 与 SDEP 结构，模型规模和计算开销相对较大，训练需单 GPU（A40）且 batch 受限；② 对极端海岸干扰仍有漏检/误检；③ 仅针对单模 SAR，未探索多模融合；④ 未对实时推理或嵌入式部署进行深入评估。

---

## 864. State of the Art Report for Smart Habitat for Older Persons -- Working Group 3 -- Healthcare

**arXiv ID:** 2604.03255 | [PDF](https://arxiv.org/pdf/2604.03255v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 865. Joint Fullband-Subband Modeling for High-Resolution SingFake Detection

**arXiv ID:** 2604.04841 | [PDF](https://arxiv.org/pdf/2604.04841v1)

**作者:** Xuanjun Chen `[一作]` (National Taiwan University), Jyh-Shing Roger Jang `[通讯]` (National Taiwan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了首个针对高分辨率（44.1 kHz）合成歌声深度伪造检测的联合全频段与子频段建模框架（Sing-HiResNet），通过融合全局与局部频谱信息实现对合成歌声的精准识别。

**💡 创新点**

创新点包括：①首次系统分析高分辨率音频在 SingFake 检测中的价值；②设计了多尺度子频段专家模型与全频段模型的联合融合策略；③提出了基于子频段知识蒸馏的全频段学生模型，既保持模型压缩，又提升了对高频细节的感知。

**🔧 技术方法**

主要技术包括：ResNet‑18 作为特征提取器（单通道 log‑power spectrogram），多分辨率子频段划分（N = 1, 2, 4, 8），四种融合策略（决策级聚合、特征级拼接、跨专家注意力、跨专家蒸馏），以及基于 Grad‑CAM 的可视化分析。

**📊 数据集**

使用 WildSVDD 数据集（97 位歌手、3223 首歌，训练集 27,879 条音频，Test A 与 Test B 两套测试集）进行评估。

**📈 对比分析**

与 16 kHz 采样率的基线（如 Wav2Vec、WavLM 等）以及同样采用 44.1 kHz 的 UNIBS ResNet‑18 进行对比，Sing-HiResNet 在 Test A 的 EER 下降至 1.58%（比 UNIBS 低 31.6%），在 Test B 降至 7.45%（比 UNIBS 低 30.9%），在所有公开对手模型中均获得最优或竞争性最优成绩。

**⚠️ 局限性**

局限性包括：①在跨语言（Out‑of‑Distribution）测试中仍存在一定误差，说明子频段划分与融合策略可能对不同音色/语言仍需进一步优化；②子频段模型数量增多时会导致特征信息稀疏、训练难度提升；③蒸馏与融合带来额外的计算与存储开销，实际部署时需权衡模型大小与推理速度。

---

## 866. Combee: Scaling Prompt Learning for Self-Improving Language Model Agents

**arXiv ID:** 2604.04247 | [PDF](https://arxiv.org/pdf/2604.04247v1)

**作者:** Hanchen Li `[一作]` (University of California Berkeley), Joseph E. Gonzalez `[通讯]` (University of California Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Combee 框架，实现高并行度的 prompt learning，解决了传统方法在大批量更新时的上下文过载问题。

**💡 创新点**

创新点包括：1) 并行扫描聚合算法，分层聚合避免一次性输入过多反射；2) 增广洗牌机制，复制并随机打散反射以提高信息利用率；3) 动态批量控制，根据延迟曲线自动调节批量大小，实现速度与质量平衡。

**🔧 技术方法**

技术手段：Map‑Shuffle‑Reduce 架构、并行扫描聚合、增广洗牌、动态批量控制，并与现有 ACE、GEPA 等生成‑反思‑更新框架无缝集成。

**📊 数据集**

实验数据集：Agentic Benchmarks（AppWorld、Terminal‑Bench 2.0）以及领域特定 Benchmarks（FiNER、Formula）。

**📈 对比分析**

对比基线（ReAct+ACE/GEPA、Summarization、Top‑K 等）和传统批量 scaling，Combee 在所有任务上实现最高 17× 的速度提升，精度保持或提升，且成本基本相同；在 AppWorld、Terminal‑Bench、FiNER、Formula 上均达到或超越最佳固定批量结果。

**⚠️ 局限性**

局限性：对更大模型和更复杂任务的通用性仍待验证；需要一定的分布式计算资源，动态控制和增广洗牌参数可能对不同任务需要微调；目前主要针对 generate‑reflect‑update 结构，其他学习范式尚未充分探索。

---

## 867. Spatially-Weighted CLIP for Street-View Geo-localization

**arXiv ID:** 2604.04357 | [PDF](https://arxiv.org/pdf/2604.04357v1)

**作者:** Ting Han `[一作]` (Sun Yat-Sen University), Meiliu Wu `[通讯]` (University of Glasgow)

**通讯引用:** 125 | [OpenAlex ID](https://openalex.org/A5029709703)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了 Spatially-Weighted CLIP，用位置文本编码与空间加权对比学习实现街景图像地理定位。

**💡 创新点**

将 CLIP 的一对一监督改为基于地理距离的软标签，并加入邻域一致性正则，从而实现地理对齐而非单纯语义对齐。

**🔧 技术方法**

采用 CLIP 图像/文本编码器、Haversine 距离计算、InfoNCE 损失改写、区域公平正则和 ViT 视觉变压器。

**📊 数据集**

使用英国八个城市的 xRI 数据集（806 条带 GPS 坐标的街景图像）。

**📈 对比分析**

在 ViT-B/ViT-L 基线上进行对比，SW-CLIP 将中位误差从 276.8 m 降至 91.96 m，平均误差从 6723.5 m 降至 449.25 m，召回率与空间一致性指标显著提升。

**⚠️ 局限性**

局限在于仅在小规模、同一国家的城市数据上验证，仍面临长尾误差与地区分布不平衡的问题。

---

## 868. Text Summarization With Graph Attention Networks

**arXiv ID:** 2604.03583 | [PDF](https://arxiv.org/pdf/2604.03583v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 869. AttackEval: A Systematic Empirical Study of Prompt Injection Attack Effectiveness Against Large Language Models

**arXiv ID:** 2604.03598 | [PDF](https://arxiv.org/pdf/2604.03598v1)

**作者:** Jackson Wang `[一作]` `[通讯]`, Jackson Wang

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估Prompt Injection攻击效果，构建AttackEval-250手工攻击Prompt集并在四层防御下进行实验。

**💡 创新点**

提出10类攻击分类与250个Prompt样本，发现Obfuscation及语义/社交攻击最难防御，组合攻击可近乎突破所有防御。

**🔧 技术方法**

使用基于规则的任务约束LLM助手模拟、四层防御（关键词、语义、意图），计算攻击成功率（ASR）并采用Bootstrap CI、Stealth评分与组合攻击模型。

**📊 数据集**

AttackEval-250：由研究者手工构造的250个Prompt，涵盖10类攻击。

**📈 对比分析**

在四防御层级下对ASR进行对比，单攻击中Obfuscation ASR=0.76、EM/RF≈0.44-0.48；组合Obfuscation+EM ASR=0.976，表明现有防御仍易被突破。

**⚠️ 局限性**

实验使用模拟规则系统而非真实LLM，防御层级近似真实实现，且攻击手段随时间演进，未覆盖所有新兴攻击向量。

---

## 870. CIPHR: Cryptography Inspired IP Protection through Fine-Grain Hardware Redaction

**arXiv ID:** 2604.03560 | [PDF](https://arxiv.org/pdf/2604.03560v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 871. Trustless Provenance Trees: A Game-Theoretic Framework for Operator-Gated Blockchain Registries

**arXiv ID:** 2604.03434 | [PDF](https://arxiv.org/pdf/2604.03434v1)

**作者:** Ian C. Moore `[一作]` `[通讯]` (AnchorRegistry), Ian C. Moore (AnchorRegistry)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于公共区块链的可证明 Provenance Tree 并解决了操作员信任和树中毒（tree poisoning）问题。

**💡 创新点**

创新点在于设计双层承诺方案，使误归因成为严格劣势策略，并分别对三类攻击（伪造根、恶意子节点、树身份伪造）给出完整闭合机制。

**🔧 技术方法**

采用 keccak256 哈希、双层承诺、治理级联（VOID 机制）以及以太坊 L2 上的 AnchorRegistry 智能合约实现。

**📊 数据集**

未使用传统机器学习数据集，利用 Base（Ethereum L2）公开事件日志作为验证与重建数据来源。

**📈 对比分析**

通过 gas 成本分析显示每条注册仅增加约20,378 gas，复杂度为 O(1)；重建完整树可在 O(|V|) 时间内完成，无离线基础设施依赖。

**⚠️ 局限性**

局限在于需要可信操作员节点来提交注册，且对恶意操作员的治理仍依赖外部机制；实验仅覆盖链上攻击，链外攻击场景未涵盖。

---

## 872. Greedy and Transformer-Based Multi-Port Selection for Slow Fluid Antenna Multiple Access

**arXiv ID:** 2604.04589 | [PDF](https://arxiv.org/pdf/2604.04589v1)

**作者:** Darian Perez-Adan `[一作]`, Luis Castedo `[通讯]` (University of A Coru)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了两种多端口流体天线选择方案，分别是增量式贪婪前向加交换改进算法（GFwd+S）和基于Transformer的神经网络端口选择器，旨在提升多用户慢速FAMA系统的信令效率。

**💡 创新点**

创新点在于（1）证明并利用贪婪前向方法的非递减SINR性质，使得在L个射频链下可显著避免传统后向消除的早期子最优决策；（2）结合仿真学习与强化学习的两阶段训练流程，借助Transformer捕捉端口间相互依赖，实现低延迟但近似最优的端口选取。

**🔧 技术方法**

使用的技术包括：自适应正交匹配追踪（GFwd+S）、自注意力Transformer编码器、交叉熵监督学习、REINFORCE策略梯度、GEV组合器、Jakes相关模型的FA通道仿真。

**📊 数据集**

数据集为基于Jakes模型的合成FA通道样本，包含不同SNR（5–25 dB）下的15,000个训练样本和1,000个验证样本。

**📈 对比分析**

通过与慢速FAMA、DC、CUMA、GEPort等基准方案对比，GFwd+S在所有SNR下均能获得最高的信令效率；Transformer网络在训练后能逼近GFwd+S的性能，并在SNR≥15 dB时超过GEPort，推理延迟仅为几毫秒，显著低于GFwd+S。

**⚠️ 局限性**

主要限制包括：（1）对仿真数据的依赖，真实环境中的端口相关性与干扰模式可能导致性能下降；（2）Transformer训练仍需要相对较多的样本和两阶段学习，模型规模和硬件加速需求较高；（3）GFwd+S在极大端口数或极高SNR时仍存在局部最优风险。

---

## 873. FORMULA: FORmation MPC with neUral barrier Learning for safety Assurance

**arXiv ID:** 2604.04409 | [PDF](https://arxiv.org/pdf/2604.04409v1)

**作者:** Qintong Xie `[一作]` (Dartmouth), Peter Chin `[通讯]` (Dartmouth)

**通讯引用:** 6566 | [OpenAlex ID](https://openalex.org/A5113696329)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种名为 FORMULA 的分布式安全编队控制框架，能在复杂动态环境中保持多机器人编队并避免碰撞。

**💡 创新点**

创新点在于：①将 MPC‑CLF 形成追踪层与神经网络学习的去中心化 CBF 结合，实现对安全约束的自适应学习；②提出事件触发式死锁解决机制，通过对编队参考状态的局部变换来打破因安全约束导致的停滞；③在保证安全的同时，显著降低在线计算负担，并通过网络学习获得可扩展性。

**🔧 技术方法**

主要技术包括：分布式模型预测控制（MPC）与控制李雅普诺夫函数（CLF）约束、基于神经网络的控制屏障函数（NN‑CBF）、事件触发式死锁检测与修正、在线 QP 投影求解、两阶段网络训练（仿真数据预训练 + 真实编队轨迹微调）。

**📊 数据集**

使用的是基于仿真的二维平面工作空间数据集，包含 10 m × 4.5 m 的障碍物随机布置，团队规模从 2 到 8 名跟随者，记录了状态、速度、控制输入和安全约束满足情况。

**📈 对比分析**

与 APF、MPC‑CBF 和 CLF+CBF‑QP 等基线进行对比，采用安全率、平均编队误差和最小距离等指标。实验表明，FORMULA 在所有团队规模下的安全率均最高（约 86%–88%），编队误差最低（0.7 m 左右），并在密集环境下保持良好可扩展性。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，缺乏真实机器人平台测试；对通信网络无延迟、无拓扑变化的假设；NN‑CBF 需要离线训练，迁移到新环境可能需重新训练；在极端动态障碍物或大规模团队时，计算量仍可能上升。

---

## 874. Towards Policy-Enabled Multi-Hop Routing for Cross-Chain Message Delivery

**arXiv ID:** 2604.04890 | [PDF](https://arxiv.org/pdf/2604.04890v1)

**作者:** Amin Rezaei `[一作]` (University of Waterloo), Bernard Wong `[通讯]` (University of Waterloo)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

设计并实现了 xRoute 框架，提供多跳 IBC 跨链路由与消息交付，支持安全与偏好策略，构建去中心化 Relayer Network 进行协同路由与费用分配。

**💡 创新点**

创新点：①基于策略的多跳 IBC 路由，使安全政策在目标链上验证；②去中心化 Relayer Network 实现协同路由与费用分配；③提供多路径、零知识路由等替代方案。

**🔧 技术方法**

技术：IBC 协议、Cosmos SDK + CosmWasm、轻客户端、ICS‑29 费用、Nakamoto 系数验证、RISC Zero 零知识证明、SimPy 仿真、链注册表拓扑抓取。

**📊 数据集**

数据集：Cosmos 网络拓扑（链注册表、RPC 数据）、IBC 连接信息、验证者数量、Nakamoto 系数；StableSwap 实验使用 Cosmos 链 USDT‑USDC 流动性数据。

**📈 对比分析**

比较方法：与 Axelar hub‑and‑spoke、单跳 IBC 进行连接性、去中心化、可扩展性与成本对比；实验结果显示 90% 连接度、30%+ 去中心化提升、可处理 32k TPS 无显著延迟、费用 <0.1 美元/消息，StableSwap 多链路接近理想价差。

**⚠️ 局限性**

限制：仍依赖中间链的安全性（Nakamoto 系数约束）；Relayer Network 可能遭受恶意路由或拥塞；实验基于模拟，真实链性能可能不同；零知识路由实现成本高。

---

## 875. Parameter-Efficient Semantic Augmentation for Enhancing Open-Vocabulary Object Detection

**arXiv ID:** 2604.04444 | [PDF](https://arxiv.org/pdf/2604.04444v1)

**作者:** Weihao Cao `[一作]` (Beijing Jiaotong University), Liping Jing `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 3585 | [OpenAlex ID](https://openalex.org/A5069749738)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出HSA-DINO框架，利用多尺度提示库和语义感知路由实现对垂直领域的自适应，同时保持开放词汇检测能力。

**💡 创新点**

创新点在于：①多尺度提示库（MSPB）通过视觉金字塔选择层级化语义提示，显著提升文本表示；②语义感知路由（SAR）动态决定是否使用域特定增强，避免过拟合导致的通用性能下降；③参数高效的LoRA微调，保持预训练知识。

**🔧 技术方法**

采用的技术包括：CLIP+Swin Transformer基础网络，LoRA低秩适配，MSPB与SAR机制，交叉熵+回归+GIoU+去噪等损失；训练时加入匹配损失和正交损失。

**📊 数据集**

使用的基准数据集有：OV-COCO（通用域）、ArTaxOr、DIOR、UODD（垂直领域）以及组合式的OV-COCO⁺。

**📈 对比分析**

与GLIP、Grounding DINO、YOLO-World、OV-DINO等方法在零样本、全微调和参数高效微调三种设置下对比，HSA-DINO在所有垂直任务上取得最高的调和平均值（H），在OV-COCO⁺上同样表现最佳，显示出最佳的域适应与开放词汇泛化平衡。

**⚠️ 局限性**

局限性在于：仍需针对不同域手工设置重构阈值τ，提示库规模与长度的经验性选择；在极端新域时仍可能出现语义误匹配；以及对极少样本或高复杂背景的处理还有提升空间。

---

## 876. Multimodal Structure Learning: Disentangling Shared and Specific Topology via Cross-Modal Graphical Lasso

**arXiv ID:** 2604.03953 | [PDF](https://arxiv.org/pdf/2604.03953v1)

**作者:** Fei Wang `[一作]` (Stony Brook University), Xiong Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 23017 | [OpenAlex ID](https://openalex.org/A5020450516)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出跨模态图形玻尔扎诺斯模型（CM‑GLasso），通过文本可视化与SigLIP 2统一编码、交叉注意力蒸馏、非正态转换以及联合ADMM优化，得到共享与类别特定的稀疏精度矩阵

**💡 创新点**

创新点包括跨模态文本可视化与统一编码、跨注意力蒸馏构造语义节点、利用eBIC调控先验的可变Sigmoid、以及将Tailored GLasso与Common‑Specific Structure Learning（CSSL）统一入单一ADMM目标

**🔧 技术方法**

使用技术有SigLIP 2 Vision‑Language Encoder、交叉注意力蒸馏、非参数非正态转换（nonparanormal）、eBIC自适应先验、联合ADMM优化以及图结构驱动的分类与分割头

**📊 数据集**

使用的数据集包括CIFAR‑10/100、CUB‑200‑2011、Caltech‑256、ADE20K、PASCAL‑VOC‑2012、MS COCO、Kvasir‑SEG 等八个基准

**📈 对比分析**

与多种SOTA方法对比，CM‑GLasso 在 CUB‑200 上 92.83%、ADE20K 64.01% mIoU、VOC‑2012 74.75% mIoU、COCO‑2014 46.82% mIoU 等指标均实现领先，并在分类任务中多项数据集上取得最高准确率

**⚠️ 局限性**

局限性是离线ADMM优化需对精度矩阵进行 O(TCp³) 的特征值分解，导致在千级类别或大规模数据上扩展受限

---

## 877. Uncertainty-Aware Foundation Models for Clinical Data

**arXiv ID:** 2604.04175 | [PDF](https://arxiv.org/pdf/2604.04175v1)

**作者:** Qian Zhou `[一作]` (University of the Chinese Academy of Sciences), Shi Li `[通讯]` (Columbia University)

**通讯引用:** 35248 | [OpenAlex ID](https://openalex.org/A5025170020)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出一种面向医疗的基础模型框架，使用分布式（集合值）表示学习每位患者的潜在生理状态，并通过多模态编码器和一致性约束实现对不完整观测的无偏推断。

**💡 创新点**

创新点在于：①把患者数据视为对潜在状态的稀疏约束，使用分布式后验 q(z|x) 代替传统点嵌入；②通过不同部分观测之间的一致性损失让模型学习可共享的潜在空间；③将重建、对比、分布正则三种自监督目标结合，提升鲁棒性与校准度。

**🔧 技术方法**

技术手段包括：多模态 Transformer / ViT 编码器；高斯后验参数化；重建 (masked prediction) 目标；一致性损失（KL/Wasserstein/ MMD）；对比几何 (InfoNCE)；自监督与半监督联合训练；采样推断与不确定性校准。

**📊 数据集**

数据集主要来自 MIMIC-IV：①结构化 EHR（实验室、诊断、手术）；②多模态子集（文本、影像、结构化数据）；③连续生理波形（ECG/PPG）。在实验中人为掩码模拟缺失。

**📈 对比分析**

与 MAE、对比学习、序列模型等基线在 AUROC、AUPRC、MSE、C-index 上均优，尤其是分布式变体在 AUPRC、校准（ECE/NLL）上显著提升；在缺失率升高时性能衰减更缓。

**⚠️ 局限性**

局限性包括：后验采用单一高斯族限制表达能力；采样与KL估计带来额外计算成本；仅关注观测缺失导致的不确定性，未拆解标签噪声、疾病定义多样性等其他不确定源；缺乏在实际临床决策流程中验证不确定性利用。

---

## 878. HDP: A Lightweight Cryptographic Protocol for Human Delegation Provenance in Agentic AI Systems

**arXiv ID:** 2604.04522 | [PDF](https://arxiv.org/pdf/2604.04522v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 879. HorizonWeaver: Generalizable Multi-Level Semantic Editing for Driving Scenes

**arXiv ID:** 2604.04887 | [PDF](https://arxiv.org/pdf/2604.04887v1)

**作者:** Mauricio Soroco `[一作]` (Simon Fraser University), Ziyu Jiang `[通讯]` (NEC Labs America)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种可扩展的指令驱动的驾驶场景图像编辑框架 HorizonWeaver，能够在同一图像中同时完成局部（对象级）与全局（场景级）编辑，支持多层次细粒度控制。

**💡 创新点**

创新点包括：1) 细粒度语言掩码 LangMask，将 CLIP 文本嵌入像素掩码，实现精确对象定位与指令约束；2) 大规模真实/合成配对数据集生成，结合 Boreas、nuScenes、Argoverse2 等多源日志；3) 训练策略融合监督（L1+LPIPS）、循环一致性、CLIP 对齐三种损失，兼顾内容保持与指令遵循；4) 结合 VLM 进行属性提取与自动验证，提升数据质量。

**🔧 技术方法**

技术包括：扩散模型（UltraEdit 7B 变体）+ VAE；CLIP 与 DINO 视觉-文本特征；LPIPS 感知损失；LLM（ChatGPT‑4o‑mini）生成指令；VLM（CLIP/BLIP 等）进行属性检测、提示生成与真实性验证；Qwen‑Image‑Edit 用于合成编辑；Poisson 混合、Super‑Resolution 等后处理。

**📊 数据集**

使用的数据集：Boreas、nuScenes、Argoverse2、Waymo、nuPlan；通过自动配对生成 255K 对图像，覆盖 13 种编辑类别；还构造了 pseudo‑pair 数据（全球/局部编辑），以及未见环境的 OOD 测试集。

**📈 对比分析**

与 Qwen、OmniGen2、UltraEdit、Bagel 等基线进行对比。HorizonWeaver 在细粒度、全局、复合编辑任务上取得最低 L1/L2、最高 CLIP/DINO，相比最强基线提升 46.4% 用户偏好率，BEV 分割 IoU 提升 33%。实验显示在多域 OOD 场景下仍保持强大泛化能力。

**⚠️ 局限性**

局限性：依赖 LLM/VLM 生成的指令与验证，可能产生幻觉或噪声；训练过程对大量算力和高质量配对数据要求高；仅在单视角摄像头图像上评估，未覆盖多视角/三维重建；对极端光照/恶劣天气的鲁棒性仍待进一步提升。

---

## 880. Biologically Inspired Event-Based Perception and Sample-Efficient Learning for High-Speed Table Tennis Robots

**arXiv ID:** 2604.04618 | [PDF](https://arxiv.org/pdf/2604.04618v1)

**作者:** Ziqi Wang `[一作]` (National University of Defence Technology), Huadong Dai `[通讯]` (Defense Innovation Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了基于事件相机的高频乒乓球机器人，提出事件驱动的球检测与样本高效的强化学习策略，实现快速、低延迟的感知与决策。

**💡 创新点**

创新点在于：① 直接对异步事件流利用运动线索与几何一致性进行球检测，显著降低数据量与延迟；② 结合案例依赖时间自适应奖励（CDTA）与奖励阈值机制的逐步训练，提升低样本下高速度击球的学习效率。

**🔧 技术方法**

使用技术包括：事件相机（DVS）、DBSCAN聚类、极性滤波、几何验证、双目立体三维定位、多项式轨迹预测、DDPG强化学习、CDTA奖励、奖励阈值回放策略。

**📊 数据集**

使用自研的高保真仿真与真实环境DVS/ RGB数据集，总计约2,789个3D和2,093个2D球位置信息，并已公开公开。

**📈 对比分析**

通过与YOLOv4‑tiny RGB、F‑E Fusion、EBPP等基线方法比较，事件方法在数据量↓99.8%、延迟↓96.4%、召回提升27.5%；在5 m/s高速度场景下，样本高效学习平均误差277 mm，较SOTA提升35.8%。

**⚠️ 局限性**

局限在于：实验多聚焦单球/已标注场景，现实中多球或多玩家干扰的鲁棒性尚待验证；模型对极端速度的稳健性与硬件同步/低延迟需求仍需进一步完善。

---

## 881. Who is the author? A legal and normative view of authorship in Generative AI-aided academic works

**arXiv ID:** 2604.04700 | [PDF](https://arxiv.org/pdf/2604.04700v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 882. Domain-Contextualized Inference: A Computable Graph Architecture for Explicit-Domain Reasoning

**arXiv ID:** 2604.04344 | [PDF](https://arxiv.org/pdf/2604.04344v1)

**作者:** Chao Li `[一作]` (Deepleap.ai), Chunyu Zhao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

建立了一种计算底层无关的推理架构，其中领域作为每个推理操作的显式参数，而不是隐式注释或外部条件。

**💡 创新点**

创新点在于将领域作为计算参数，使得推理过程能够进行领域范围的修剪，减少查询搜索空间，并实现底层无关的执行。

**🔧 技术方法**

使用了领域上下文单元（CDC）框架，结合了链索引、路径遍历和向量引导计算等技术。

**📊 数据集**

通过PHQ-9临床推理案例研究进行验证，展示了理论机制与实际推理任务的对应关系。

**📈 对比分析**

与传统推理系统相比，CDC在领域范围推理上具有更高的效率，查询复杂度从O(N)降低到O(N/K)，并且在不同计算底层上保持一致的推理结果。

**⚠️ 局限性**

局限性包括对大规模知识库的实证验证尚未进行，领域生成的无界增长问题，以及跨底层语义等价性的严格形式化尚未解决。

---

## 883. Building a Dataspace for Manufacturing as a Service in Factory-X

**arXiv ID:** 2604.03678 | [PDF](https://arxiv.org/pdf/2604.03678v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 884. Event-Driven Neuromorphic Vision Enables Energy-Efficient Visual Place Recognition

**arXiv ID:** 2604.03277 | [PDF](https://arxiv.org/pdf/2604.03277v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 885. Bounded by Risk, Not Capability: Quantifying AI Occupational Substitution Rates via a Tech-Risk Dual-Factor Model

**arXiv ID:** 2604.04464 | [PDF](https://arxiv.org/pdf/2604.04464v1)

**作者:** Shuyao Gao `[一作]` (aSSIST University), Minghao Huang `[通讯]` (aSSIST University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了基于技术可行性与商业风险的双因素模型，对美国923个职业及其2087个细化工作活动进行评估，计算相对职业自动化指数（OAI）。

**💡 创新点**

创新点在于将风险溢价（Compliance Premium）与技术水平分离，提出“认知风险不对称”，采用Leontief瓶颈聚合与重要性加权，将人工与AI的技术概率与人类风险厌恶相结合，形成对职业可替代性的全新量化框架。

**🔧 技术方法**

主要技术手段包括：多模型LLM集成（Qwen、Gemma、Llama、Mistral）进行零样本技术与风险评分；人机交互（HITL）专家评估与变异分层抽样；秩相关、分布式逻辑回归、Wilcoxon检验、Spearman相关；任务到职业的Leontief极值聚合与重要性加权；情景敏感性分析。

**📊 数据集**

使用美国O*NET 30.2数据库的923个职业、2087个详细工作活动（DWA），并收集31名跨国专家的人工评估结果作为验证。

**📈 对比分析**

与传统单一技术曝光模型相比，本研究通过人工评估验证揭示了约+0.35的风险溢价差距；OAI分布显示仅4.4%的职业高曝光（≥0.60），而超过50%的职业处于低曝光（<0.30）。模型在不同风险容忍情景下保持高度的秩相关（>0.98），验证了模型结构稳健性。

**⚠️ 局限性**

主要局限包括：技术水平快速演进导致OAI为2026年的静态快照；专家样本量小且仅涵盖特定行业；O*NET税onomies滞后，未反映新兴AI辅助任务；模型仅基于统计拟合AI，未考虑未来逻辑AI对风险阈值的冲击；风险评分阈值和自动化概率映射为经验性设定，需进一步实证校准。

---

## 886. When AI Agents Disagree Like Humans: Reasoning Trace Analysis for Human-AI Collaborative Moderation

**arXiv ID:** 2604.03796 | [PDF](https://arxiv.org/pdf/2604.03796v1)

**作者:** Michał Wawer `[一作]` (Warsaw University of Technology), Jarosław A. Chudziak `[通讯]` (Warsaw University of Technology)

**通讯引用:** 88 | [OpenAlex ID](https://openalex.org/A5008057050)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出将多智能体系统中的不一致视为诊断信号，而非噪声，并通过四类别税onomies分析推理轨迹结构以判定是否需要人工干预。

**💡 创新点**

创新点在于将价值多元化的“共识式不一致”（相似推理却得出不同结论）作为真正的价值冲突信号，区别于传统的“噪声式不一致”，并通过推理相似度与结论一致性双重维度构建四分类体系。

**🔧 技术方法**

使用了 DeepSeek‑V3 作为基础模型，利用系统提示差异化“视角”实现多智能体；推理轨迹嵌入采用 all‑mpnet‑base‑v2 并用余弦相似度衡量；对推理相似度设阈值 0.72 进行四类别划分；评价时使用 Pearson、Spearman 相关、Kruskal‑Wallis、Precision/Recall/F1 等统计手段。

**📊 数据集**

实验使用 Measuring Hate Speech 语料库，对 39,565 条社交媒体评论的 10 维度评分计算人类标注者的标准差作为“真实争议度”，并抽样 600 条评论进行多智能体推理与分类。

**📈 对比分析**

与仅基于推理相似度分散度的“divergence‑only”方法和随机基线相比，四类别税onomies 的“category‑based escalation”在 F1 上略有提升（0.548 vs. 0.503），但差异仍属温和，召回率在 divergence‑only 方案更高。总体表现显示结构化不一致信息比单纯幅度更能解释人类争议。

**⚠️ 局限性**

局限包括样本量有限导致 CD 与 DD 的差异未显著；所有智能体使用同一模型仅通过提示区分视角，导致词汇差异与推理差异混淆，类别分布失衡；实验仅在英文美国社交媒体数据上验证，缺乏多语言、多领域与任务级别的进一步评估。

---

## 887. Focus Matters: Phase-Aware Suppression for Hallucination in Vision-Language Models

**arXiv ID:** 2604.03556 | [PDF](https://arxiv.org/pdf/2604.03556v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 888. Cog-DRIFT: Exploration on Adaptively Reformulated Instances Enables Learning from Hard Reasoning Problems

**arXiv ID:** 2604.04767 | [PDF](https://arxiv.org/pdf/2604.04767v1)

**作者:** Justin Chih-Yao Chen `[一作]` (University of North Carolina at Chapel Hill), Mohit Bansal `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大语言模型在RLVR训练中无法从难度过高的开放式问题中学习的瓶颈，本文提出一种任务重构与自适应课程体系（Cog‑DRIFT），将难题转化为多选、填空等结构化形式，再逐步提升难度，使模型能够获取更密集的奖励信号并最终提升对原始开放式问题的解答能力。

**💡 创新点**

创新点在于：①将任务重构视为“认知负荷”降低的手段，涵盖从判别型到生成型的多种格式；②构建实例级自适应课程，使每个问题根据模型表现自动从易到难推进；③利用重构任务学习到的推理能力回迁至原始开放式问题，并在跨数据集上保持良好泛化；④在RLVR框架下首次实现对难度过高样本的可学习性。

**🔧 技术方法**

技术手段包括：使用LLM（如Qwen3-4B）自动生成多选/填空版本；采用GRPO（Group Relative Policy Optimization）结合可验证奖励；引入拒绝采样微调（RFT）获取正确推理链；设计实例级自适应课程调度逻辑；在训练中混合多种格式并通过验证脚本确保答案一致性。

**📊 数据集**

数据集：以BigMath的最难20%样本为训练基础（经pass@64筛选），进一步过滤后得到约950道难题；评估使用六个公开评测集：BigMath‑Hard、OmniMATH‑Hard、AIME 2024/2025、GPQA‑Diamond、Date Understanding；另外在MATH500上检验泛化能力。

**📈 对比分析**

与零样本、少量示例、RFT、标准GRPO、NuRL（Abstract/Prefix）等基线对比，结果显示Cog‑DRIFT在Qwen上平均提升4.72%，在Llama上提升3.23%；对BigMath‑Hard实现绝对提升10.11%/8.64%；在所有六个基准上均超过或匹配最佳对手；同时在pass@k（k=1…128）上表现更好，证明学习到的推理能力能在更大样本量下体现。

**⚠️ 局限性**

局限性：①难度筛选依赖pass@k，易产生无法解决或标签错误的样本；②对数据质量极为敏感，需额外过滤；③目前仅在Qwen、Llama两模型和数学/推理类任务上验证；④重构过程需手工定义格式，迁移到其他领域仍需研究；⑤方法仍依赖规则式奖励，复杂奖励环境可能受限。

---

## 889. Beauty in the Eye of AI: Aligning LLMs and Vision Models with Human Aesthetics in Network Visualization

**arXiv ID:** 2604.03417 | [PDF](https://arxiv.org/pdf/2604.03417v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 890. Learning Robust Visual Features in Computed Tomography Enables Efficient Transfer Learning for Clinical Tasks

**arXiv ID:** 2604.04133 | [PDF](https://arxiv.org/pdf/2604.04133v1)

**作者:** Rubén Moreno-Aguado `[一作]` (Imperial College London), Guang Yang `[通讯]` (Imperial College London)

**通讯引用:** 18118 | [OpenAlex ID](https://openalex.org/A5108053324)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

训练并评估了一款名为VoxelFM的3D CT基础模型，利用自监督DINO框架在137,000+ CT扫描上进行预训练，并在不微调backbone的情况下，使用轻量化探针在七类临床任务上进行下游评估。

**💡 创新点**

创新点在于：①将自监督DINO与iBOT掩码、KoLeo多样性等技术结合，构建高质量无语言监督的CT视觉表示；②通过多尺度、可变长宽高的增强，使模型在任意体积尺寸下可推理；③证明冻结backbone+探针即可在多任务上优于现有四大基线，尤其在报告生成上超越显式语言监督模型。

**🔧 技术方法**

主要技术包括：3D Vision Transformer（ViT）与旋转位置编码；DINO自蒸馏与iBOT掩码学习；KoLeo距离正则化；Q-Former与轻量化MLP/线性探针；报告生成采用LLaVA框架与Qwen3-8B LLM的LoRA微调。

**📊 数据集**

预训练数据来自137,107个公开CT研究，包括CT-RATE、Merlin、NLST、RIDER-Lung-PET-CT等；下游任务使用CT-RATE、Merlin、RSNA-STR、iCTCF、Mycobacterial、TotalSegmentator、LUNA16、OSIC、Mediastinal Lymph Node、NSCLC-Radiomics、AirRC等七大类别数据集。

**📈 对比分析**

与RadFM、Merlin、M3D、CT-CLIP四个现有CT基础模型在分类、回归、存活、检索、定位、分割和报告生成七类任务上进行冻结backbone+轻量化探针评估。VoxelFM在大多数任务（分类、回归、定位、分割、报告生成）均达到或超过基线；在存活预测任务中唯一表现超过随机；在实例检索任务中表现与基线相当，均接近随机。

**⚠️ 局限性**

局限性包括：①仅评估冻结backbone性能，未展示全微调潜力；②未对预训练数据的分布（扫描仪、协议、病理）进行系统分析；③实例检索效果差，说明嵌入空间受扫描参数影响；④报告生成仍不及专门分类器；⑤若将来出现大规模CT-报告配对数据，语言监督策略可能进一步提升性能。

---

## 891. Fair Aggregation in Virtual Power Plants

**arXiv ID:** 2604.03559 | [PDF](https://arxiv.org/pdf/2604.03559v1)

**作者:** Liudong Chen `[一作]` (Columbia University), Bolun Xu `[通讯]` (Columbia University)

**通讯引用:** 3996 | [OpenAlex ID](https://openalex.org/A5086056383)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一套公平感知的虚拟电厂（VPP）定价框架，系统性研究了在消费者弹性异质性下，不同公平准则（能量公平、价格公平、效用公平）及其强度对消费者福利、运营者利润和社会福利的影响；

**💡 创新点**

创新点包括：①在Stackelberg博弈框架下连续调节公平强度α，揭示公平与效率的非单调关系；②系统性分辨三种公平准则的可行性及其互斥性；③通过理论阐释与多消费者数值验证，证明公平准则在不同参数下产生四种或三种运营“区间”并给出转移规律；

**🔧 技术方法**

技术手段：基于消费者成本函数的二次形式构建解析价格响应函数；构建线性聚合收益函数；将公平约束转换为α-加权不等式；使用解析闭式解与数值优化（Pyomo）交叉验证；利用阶段性案例数据进行回归估计并聚类分层定价；

**📊 数据集**

数据集：iFlex 2020年挪威电力系统实验数据（1233户家庭，包含价格刺激与用电记录以及收入调查），并在实验中采用分层定价（3层）。

**📈 对比分析**

比较方法：将无公平（α=0）与三种公平在α∈[0,1]的全范围进行对比；评价指标包括消费者Nash福利、总消费者效用、社会福利及运营者利润。结果显示：能量公平在中等α时可提升所有指标；价格公平可提升社会福利和总效用，但对低弹性消费者无益；效用公平显著提升低弹性消费者福利，提升CNW，但伴随最大利润损失。

**⚠️ 局限性**

局限性：①仅考虑单周期线性/二次成本模型，未捕捉多周期、非线性或行为非理性；②公平约束采用单一维度不等式，缺乏对多目标权衡的动态分析；③案例仅在挪威实验环境，结果对不同市场规则、价格波动与规模的推广需要进一步验证；④聚合器完全了解消费者响应的假设在现实中不易实现，可能导致公平实施效果偏差。

---

## 892. Scale-Aware Vision-Language Adaptation for Extreme Far-Distance Video Person Re-identification

**arXiv ID:** 2604.04183 | [PDF](https://arxiv.org/pdf/2604.04183v1)

**作者:** Ashwat Rajbhandari `[一作]` (Arizona State University), Bharatesh Chakravarthi `[通讯]` (Arizona State University)

**通讯引用:** 191 | [OpenAlex ID](https://openalex.org/A5083090349)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在极端远距离空中-地面视频人物重识别任务中，提出基于CLIP ViT‑L/14的尺度感知适配框架，并对模型进行选择性微调、时间注意力池化、优化策略改进以及k‑相互重排序。

**💡 创新点**

创新点包括：① 将大规模视觉语言模型与稳定性驱动的选择性微调结合；② 引入轻量化时间注意力聚合抑制噪声帧；③ 通过提示与适配器实现跨视角域的自适应。

**🔧 技术方法**

采用技术包括CLIP视觉语言预训练、ViT‑L/14编码器、Prompt‑based跨视角调制、Adapter/QATW模块、时间注意力池化、k‑相互重排序、余弦学习率调度和颜色抖动增强。

**📊 数据集**

使用DetReIDX极端远距离视频重识别基准数据集，涵盖A2G、G2A、A2A三种协议。

**📈 对比分析**

与官方CLIP ViT‑B/16基线和公开最高结果对比，整体mAP从28.11提升到35.73，A2G、G2A分别达46.69、41.23，A2A为22.98，显著优于基线和最高公开结果。

**⚠️ 局限性**

局限性在于对深层检索排名提升有限（尤其同视角A2A的Rank‑5/10仍不如部分方法）、大模型与重排序导致推理延迟和计算成本上升，以及仅在DetReIDX上评估，跨数据集泛化尚未验证。

---

## 893. FunFact: Building Probabilistic Functional 3D Scene Graphs via Factor-Graph Reasoning

**arXiv ID:** 2604.03696 | [PDF](https://arxiv.org/pdf/2604.03696v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 894. FairLogue: A Toolkit for Intersectional Fairness Analysis in Clinical Machine Learning Models

**arXiv ID:** 2604.04858 | [PDF](https://arxiv.org/pdf/2604.04858v1)

**作者:** Nick Souligne `[一作]` (University of Arizona), Vignesh Subbian `[通讯]` (University of Arizona)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究提出并实现了名为Fairlogue的工具包，用于在临床机器学习中评估交叉群体的公平性。

**💡 创新点**

创新点在于将传统单一属性公平性度量与交叉属性的观测和反事实分析相结合，提供模块化、可复现的交叉公平性评估流程。

**🔧 技术方法**

技术包括观测公平性指标扩展（DP、EO、EOD）、基于置换的反事实公平性计算、单稳健与双稳健估计方法，以及Python实现的可插拔组件。

**📊 数据集**

使用了All of Us Research Program Controlled Tier V8的数据，构建了预测青光眼手术需求的逻辑回归模型，涉及56个临床特征及性别与种族交叉子群。

**📈 对比分析**

方法通过交叉和单属性公平性度量对比，发现交叉分析揭示了更大误差和预测率差距；模型整体性能AUROC=0.709、准确率0.651，但反事实分析显示不显著系统性不公平（u值趋近0）。

**⚠️ 局限性**

局限包括子群样本量不足导致估计不稳、反事实公平性依赖于协变量充分性、对弱关联保护属性的检测敏感性，以及缺乏直接的偏差缓解功能。

---

