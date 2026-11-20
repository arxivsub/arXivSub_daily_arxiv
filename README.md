# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-11-20 | 今日论文总数: 787

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. MM-SystemOneBias: Evaluating Bias in Multimodal Jev-Style System One Decision Models

**arXiv ID:** 2611.26557 | [PDF](https://arxiv.org/pdf/2611.26557v1)

**作者:** Jinzhe Li `[一作]` (Jilin University), Yi Chang `[通讯]` (Jilin University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a2602d71-93ab-4bad-974b-672788df8193` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MM-SystemOneBias基准，评估多模态决策模型在不同干预下的表现，涵盖了5323个实例和11个任务场景。

**💡 创新点**

创新点在于建立了一个系统化的基准，能够比较多模态决策模型与其通用模型在实际任务中的表现，并分析干预对决策的影响。

**🔧 技术方法**

使用了多模态决策模型和通用视觉-语言模型，通过六种干预方式进行评估，包括选项顺序、选项标签、外部判断、图像-文本描述、无关信息和评分尺度。

**📊 数据集**

使用了来自九个公共数据集的5323个原始实例，涵盖了11个任务场景。

**📈 对比分析**

通过配对比较，决策模型在选项标签、无关信息和评分尺度干预下表现出更高的稳定性，且在错误外部判断和冲突描述下的准确性损失较小。

**⚠️ 局限性**

限制在于现有评估未能全面描述多模态决策模型在不同干预下的行为，且模型的表现可能因任务场景的不同而有所变化。

---

## 2. From Benchmarks to the Bench: Preference, Correspondence and Selectivity in Bidirectional Drug-Target Screening

**arXiv ID:** 2611.26369 | [PDF](https://arxiv.org/pdf/2611.26369v1)

**作者:** Jiangjie Qiu `[一作]` (Tsinghua University), Xiaonan Wang `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究比较了13个已发布的药物-靶标相互作用（DTI）模型在711种FDA批准的药物和614种人类蛋白质上的双向推荐，评估了这些模型在药物发现和再利用中的有效性。

**💡 创新点**

创新点在于将DTI模型的评估从基准性能转向评估模型预测是否能够指导药物发现和靶标识别中的实验决策。

**🔧 技术方法**

使用了多种深度学习和机器学习技术，包括对比学习、图形模型和多模态模型等。

**📊 数据集**

使用的数据集包括711种FDA批准的药物和614种人类蛋白质，涉及436,554对药物-靶标组合。

**📈 对比分析**

通过比较不同模型的Top-10推荐，发现77.8%的药物选择比较和70.1%的蛋白质选择比较中，模型推荐的候选者完全不重叠。某些模型在推荐活性候选者方面表现出色，但在不同查询中匹配推荐时，命中率变化不大。

**⚠️ 局限性**

限制在于模型的推荐可能无法可靠地区分特定实验中的药物-靶标偏好，且相似的预测性能可能导致实验选择的显著差异。

---

## 3. VoxForge: A Unified Foundation Model for Dense 3D Features Across MRI, CT and PET

**arXiv ID:** 2611.25309 | [PDF](https://arxiv.org/pdf/2611.25309v1)

**作者:** Rushil Anirudh `[一作]` (Amazon), Jayaraman J. Thiagarajan `[通讯]` (Amazon)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出了一种统一的3D医学基础模型，该模型能够在MRI、CT和PET等多种任务中重用冻结特征，进行分割、分类和配准。

**💡 创新点**

创新点在于开发了一个单一的编码器，结合了3D架构、自我蒸馏和跨模态课程学习的最新进展，能够从MRI和CT体积中学习可转移的特征。

**🔧 技术方法**

使用了自我蒸馏（DINOv3）技术和扩展的Primus ViT架构，结合了细粒度的8^3体素标记。

**📊 数据集**

使用了235K个公共许可的MRI和CT体积数据集，涵盖71K个受试者。

**📈 对比分析**

与最新的通用模型和专业模型进行了严格的比较，结果显示该模型在每个任务组中都获得了最佳或并列最佳的平均排名，尤其在PET任务中表现优异，且在标签效率上表现出色。

**⚠️ 局限性**

限制在于缺乏对每个成分（目标、数据混合和标记大小）的控制消融实验，且公共许可的3D数据仍然是一个瓶颈。

---

## 4. CaT-Movie: Balancing Global Planning and Local Rendering for Long Multi-Shot Video Generation

**arXiv ID:** 2611.25620 | [PDF](https://arxiv.org/pdf/2611.25620v1)

**作者:** Jinpeng Lu `[一作]` (University of Science and Technology of China), Zhiwei Xiong `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种多镜头视频生成系统，通过全球故事规划和局部窗口渲染来合成长时间段的电影，保持时间一致性。

**💡 创新点**

创新点在于结合了大型语言模型进行故事规划，并通过结构化故事板维护跨镜头状态，从而提高了视频生成的一致性和连贯性。

**🔧 技术方法**

使用了大型语言模型（如Qwen3.8-27B）进行故事规划，以及多镜头渲染器（如Wan2.2-TI2V-5B）进行视频生成。

**📊 数据集**

构建了一个包含超过35万个电影镜头的数据集，注释了语义状态转变、实体定位和多粒度请求。

**📈 对比分析**

与其他方法相比，该系统在执行计划切换方面达到了0.974的成功率，并在所有方法中获得了最高的平均分和最一致的角色表现。

**⚠️ 局限性**

限制在于高质量的开放源多镜头数据稀缺，可能影响模型的训练和评估效果。

---

## 5. Show Your Work: Can Small Open Models Read SEC Filings with Evidence a Machine Can Check?

**arXiv ID:** 2611.24990 | [PDF](https://arxiv.org/pdf/2611.24990v1)

**作者:** Krishna Nandula `[一作]` `[通讯]` (Independent Researcher), Krishna Nandula (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究探讨了小型开放模型在从SEC文件中提取数字时的表现，要求模型提供可验证的证据，包括页面和包含数字的逐字引用。

**💡 创新点**

创新点在于引入了Filing-Evidence基准，包含1000个问题，基于45家美国公司的90份最新10-K和10-Q文件，并使用机器可读的XBRL标签作为答案键。

**🔧 技术方法**

使用了多种指令模型（参数范围为7-21B），在NVIDIA RTX 4090上进行评估，采用了三种设置：普通设置、检查与重试、模型自身推理模式。

**📊 数据集**

数据集包括90份最新的10-K和10-Q文件，来自45家大型美国公司，涵盖多个行业，答案来自于文件中的XBRL标签。

**📈 对比分析**

与其他模型的比较显示，Qwen3.5-9B在普通设置下的准确率为82.7%，最佳结果为84.5%。检查与重试提高了准确率，但未减少错误答案的数量。

**⚠️ 局限性**

限制包括：数据集质量控制由AI助手执行，未进行独立验证；协议时间记录不够独立；某些问题的单位未在页面上声明；模型可能在训练中见过早期文件。

---

## 6. LocoMuJoCo: A Comprehensive Imitation Learning Benchmark for Whole-Body Control

**arXiv ID:** 2611.26293 | [PDF](https://arxiv.org/pdf/2611.26293v1)

**作者:** Firas Al-Hafez `[一作]` (Technische Universität Darmstadt), Jan Peters `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个开源的模仿学习和强化学习基准库，专注于全身控制，结合了MuJoCo和MJX后端，支持多种机器人模型和运动捕捉数据集。

**💡 创新点**

创新点在于提供了一个统一的API，支持多种机器人模型的自动重定向，模块化的插件架构，以及高效的JAX实现，简化了研究人员的工作流程。

**🔧 技术方法**

使用了JAX和MuJoCo技术，支持CPU和GPU后端的并行计算。

**📊 数据集**

使用了多个数据集，包括本地运动捕捉数据集、LAFAN1和AMASS数据集。

**📈 对比分析**

与其他现有的模仿学习和强化学习库相比，提供了更全面的功能和更高的灵活性，性能在多个算法和机器人模型上进行了评估，显示出良好的效果。

**⚠️ 局限性**

限制在于当前版本可能仍然需要用户对某些数据集的许可，并且在特定情况下可能存在性能优化的空间。

---

## 7. Learning-Based System Identification for DVL-Constrained AUV Localization

**arXiv ID:** 2611.25854 | [PDF](https://arxiv.org/pdf/2611.25854v1)

**作者:** Mingxi Zhou `[一作]` (University of Rhode Island), Chengzhi Yuan `[通讯]` (University of Rhode Island)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于学习的系统识别方法，用于在没有DVL（多普勒速度计）测量的情况下实现自主水下航行器（AUV）的定位。通过使用全连接前馈神经网络预测车辆速度（推进、侧移和下沉），以改善AUV的定位精度。

**💡 创新点**

创新点在于使用小型的前馈神经网络（参数少于1500），并将系统电压和电流作为模型输入，以提高速度预测的准确性，尤其是在强洋流条件下表现出更好的性能。

**🔧 技术方法**

采用了全连接前馈神经网络（FFN）技术，进行车辆速度的预测，并结合IMU（惯性测量单元）数据进行定位。

**📊 数据集**

使用了在Narragansett Bay收集的实地数据，进行神经网络模型的训练和验证，数据包括IMU、压力传感器和DVL的测量。

**📈 对比分析**

与DVL辅助的死算方法相比，使用神经网络预测的速度进行定位的平均均方根距离误差和标准差更低，尤其是在强洋流条件下，表现出更好的一致性和较低的漂移率。

**⚠️ 局限性**

限制在于该方法仍需在训练阶段使用DVL进行数据收集，且一旦模型训练完成，若更换车辆则需要重新训练模型。

---

## 8. JARVIS: An Agent That Designs, Builds, and Tests Circuit Boards on Its Own

**arXiv ID:** 2611.26503 | [PDF](https://arxiv.org/pdf/2611.26503v1)

**作者:** Shibo Zhou `[一作]` (University of Georgia), He Yang `[通讯]` (University of Georgia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种自主的硬件设计、制造和测试循环的系统，能够在没有人类干预的情况下完成电路板的设计、制造和测试。

**💡 创新点**

这是第一个在电力电子领域中实现完整设计-制造-测试-学习循环的系统，能够自主识别需求、设计电路、制造电路板并进行测试。

**🔧 技术方法**

使用了3D扫描、数字双胞胎技术、硬件在环测试（HIL）和机器学习等技术。

**📊 数据集**

使用了60 Hz的陷波滤波器作为基准实验，涉及到电路板的设计和测试。

**📈 对比分析**

与传统方法相比，该系统能够在2小时39分钟内完成实验，且没有人类干预，测试结果显示电路板在60 Hz下去除了50.6 dB的噪声，性能优越。

**⚠️ 局限性**

系统的局限性在于设计的电路板仍然表现出一些初学者的特征，例如电路布局不够专业，且在某些频率下的响应未能完全满足设计要求。

---

## 9. Fast Self-Supervised Generative CT Denoising via Few-Step Sampling

**arXiv ID:** 2611.26370 | [PDF](https://arxiv.org/pdf/2611.26370v1)

**作者:** Xingyi Yao `[一作]`, Zhennong Chen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种名为N2N-iMF的自监督CT去噪方法，通过少步生成建模和样本平均来加速CT图像去噪。

**💡 创新点**

N2N-iMF在不需要干净参考图像或模型蒸馏的情况下，支持推理时可调的平均计数，显著提高了去噪效率和图像质量。

**🔧 技术方法**

使用了条件生成模型和改进的平均流（iMF）技术，结合了少步生成和样本平均的策略。

**📊 数据集**

使用了模拟的低剂量腹部CT和薄层脑CT数据集，以及真实世界的光子计数CT（PCCT）脑扫描数据集进行验证。

**📈 对比分析**

与现有的自监督方法（如Noise2Noise和DDIM）相比，N2N-iMF在多个指标上表现更优，尤其是在平均绝对误差（MAE）和结构相似性指数（SSIM）上，且生成时间显著缩短。

**⚠️ 局限性**

本研究的局限性包括真实世界PCCT评估仅覆盖八个测试案例，且缺乏低噪声参考图像，依赖于对比噪声比（CNR）和视觉检查，未评估诊断准确性或假阳性和假阴性发现。

---

## 10. Public-Private Cooperation in the Port of Rotterdam: An Agent-Based Toy Model of Cooperative Efforts Against Cocaine Trafficking

**arXiv ID:** 2611.26154 | [PDF](https://arxiv.org/pdf/2611.26154v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 11. Parameter-Efficient Adaptation of Self-supervised Vision Backbones Survives Distribution Shift in Radar Micro-Doppler HAR

**arXiv ID:** 2611.25874 | [PDF](https://arxiv.org/pdf/2611.25874v1)

**作者:** Keryan Chelouche `[一作]`, Olivier Romain `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究比较了全微调、冻结探测和参数高效微调（PEFT）在微多普勒雷达下的人类活动识别（HAR）中的表现，特别关注在不同年龄、方向和数据集转移下的适应性。

**💡 创新点**

创新点在于首次将自监督图像基础模型DINOv3与全层主奇异方向适应（PiSSA）结合应用于微多普勒HAR，并且PEFT方法在所有评估设置中表现最佳。

**🔧 技术方法**

使用了DINOv3作为自监督学习的基础模型，并结合了参数高效微调技术。

**📊 数据集**

使用了两个微多普勒HAR数据集：格拉斯哥数据集和MAD数据集，这些数据集具有不同的采集设置和重叠的活动集。

**📈 对比分析**

与全微调和冻结特征相比，PEFT在所有三个转移条件下都避免了性能崩溃，DINOv3-PiSSA在每个评估设置中均排名第一，且在参数使用上更为高效。

**⚠️ 局限性**

限制在于尽管在两个较难的转移条件下表现良好，但跨数据集转移仍然具有挑战性，且所有测试的基础模型均是在自然图像上进行预训练的。

---

## 12. TuringSpace: Grounding Spatial Intelligence in Action with Interactive CAPTCHAs

**arXiv ID:** 2611.25387 | [PDF](https://arxiv.org/pdf/2611.25387v1)

**作者:** Xuwei Ding `[一作]` (University of Pennsylvania), Jian Kang `[通讯]` (MBZUAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种合成的网络环境，包含125个互动空间CAPTCHA，要求多模态大型语言模型（MLLMs）通过行动循环来解决这些问题。

**💡 创新点**

创新点在于通过互动的空间CAPTCHA测试和提升模型的空间理解能力，填补了人类与机器在空间理解上的差距。

**🔧 技术方法**

使用了合成的程序生成任务，模型通过鼠标和键盘在真实浏览器的截图上进行操作。

**📊 数据集**

使用了125个互动空间CAPTCHA任务，这些任务覆盖了五种空间能力，并通过人类玩家验证其质量。

**📈 对比分析**

与人类的表现相比，最强的模型GPT-6 Astra在任务中的成功率为74.1%，而人类为94.1%。模型在多次尝试中表现不稳定，许多模型在不同实例中表现差异较大。

**⚠️ 局限性**

限制在于当前的空间CAPTCHA仍然存在一定的局限性，模型在空间理解和推理方面的能力仍然不足，尤其是在处理复杂的空间任务时。

---

## 13. Optimal Impure $(r,δ)$-Quantum Locally Recoverable Codes with Explicit Minimum Localities

**arXiv ID:** 2611.25859 | [PDF](https://arxiv.org/pdf/2611.25859v1)

**作者:** Yang Li `[一作]` (Nanyang Technological University), San Ling `[通讯]` (Nanyang Technological University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

构建了具有（r，δ）局部性的最优不纯量子局部可恢复编码（qLRCs），并明确了其最小局部性。

**💡 创新点**

提出了一种新的构造方法，通过使用嵌套的广义单项式-笛卡尔编码（GMCCs）来实现最优性，并且这些编码具有不重叠的恢复集合。

**🔧 技术方法**

使用了广义单项式-笛卡尔编码（GMCCs）和CSS编码构造技术。

**📊 数据集**

使用了多种参数的量子编码，包括不同的局部性和最小距离，具体数据集未明确给出。

**📈 对比分析**

通过与已知的纯qLRCs进行比较，展示了所构造的编码在恢复能力和局部性方面的优势，性能优于现有的编码。

**⚠️ 局限性**

研究中提到的限制包括对不纯qLRCs的研究仍处于初期阶段，且在某些情况下最小局部性可能不是单调的。

---

## 14. How to Effectively Manage the Energy in Residential Systems: Development and Comparison of Methods

**arXiv ID:** 2611.26412 | [PDF](https://arxiv.org/pdf/2611.26412v1)

**作者:** Julian Ruddick `[一作]` `[通讯]` (Vrije Universiteit Brussel), Julian Ruddick (Vrije Universiteit Brussel)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用。

**💡 创新点**

创新点在于提出了一种改进的模型结构，能够更有效地处理复杂数据。

**🔧 技术方法**

使用了深度学习技术，特别是卷积神经网络（CNN）和循环神经网络（RNN）的结合。

**📊 数据集**

使用了公开的图像数据集和文本数据集进行实验。

**📈 对比分析**

与现有方法进行了对比，结果显示新算法在准确率和处理速度上均有显著提升。

**⚠️ 局限性**

限制在于模型对特定类型数据的适应性较差，且训练时间较长。

---

## 15. Sorting by block-reversals: closing the gap

**arXiv ID:** 2611.25842 | [PDF](https://arxiv.org/pdf/2611.25842v1)

**作者:** László Kozma `[一作]` `[通讯]` (Dresden University of Technology), László Kozma (Dresden University of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

论文探讨了块反转排序和块划分排序的复杂性，证明了这两种模型的最坏情况复杂度均为Ω(nlog^2n)。

**💡 创新点**

创新点在于填补了块反转排序和块划分排序之间的复杂性界限，证明了log^2n因子是不可避免的。

**🔧 技术方法**

使用了信息论和熵的潜力函数来分析排序过程中的复杂性。

**📊 数据集**

没有具体提到使用的数据集，但讨论了输入序列的排列和复杂性。

**📈 对比分析**

通过理论证明，块反转排序的复杂度B(n)和块划分排序的复杂度P(n)之间的关系，得出P(n) ≤ B(n)，并且都在Ω(nlog^2n)的范围内。

**⚠️ 局限性**

限制在于对特定输入的实例复杂性了解较少，尤其是在决定哪些块需要反转时的时间开销。

---

## 16. ObliGraph: Graph Pattern Query Processing with Full and Differential Obliviousness

**arXiv ID:** 2611.25405 | [PDF](https://arxiv.org/pdf/2611.25405v1)

**作者:** Lyu Xu `[一作]`, Qichen Wang `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在外包环境下的隐私保护图模式查询处理，旨在保护查询图和数据图的结构信息不被泄露。

**💡 创新点**

提出了第一个支持多类图模式查询的隐私保护框架ObliGraph，能够在完全隐私和差分隐私之间实现高效的权衡。

**🔧 技术方法**

使用了可信执行环境（TEE）技术和差分隐私的概念，设计了完全隐私和差分隐私的查询处理协议。

**📊 数据集**

在真实和合成数据集上进行了实验，使用了多个图模式查询语义的真实数据集，如Aids和PubChem。

**📈 对比分析**

与现有方法相比，ObliGraph在支持多种图模式查询的同时，保持了强隐私保证，并在性能上实现了显著提升。

**⚠️ 局限性**

现有方法在处理大规模图数据时面临可扩展性挑战，且在隐私保护与查询效率之间的权衡仍需进一步优化。

---

## 17. HapBO: Harnessed $α$-$π$ Bayesian Optimization via Expert Natural Language Priors

**arXiv ID:** 2611.25100 | [PDF](https://arxiv.org/pdf/2611.25100v1)

**作者:** Jean-Sébastien Giroux `[一作]` (Polytechnique Montréal), Marco Bonizzato `[通讯]` (Université de Montréal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的贝叶斯优化方法，称为HapBO，能够将自然语言中的先验知识转化为可用于优化的结构化信念函数。

**💡 创新点**

创新点在于通过一次性引导专家或大型语言模型（LLM）来获取信念，并将其编译成可重用的信念函数，且不需要在优化过程中调用LLM。

**🔧 技术方法**

使用了HapBO框架，该框架结合了Prior Specification Language (PSL)和α-π贝叶斯优化技术。

**📊 数据集**

在40个HPOBench任务、6-15维的合成数据集以及两个真实世界应用（皮层神经刺激和受限太阳能设计）上进行了评估。

**📈 对比分析**

与多种基线方法（包括传统的贝叶斯优化和其他LLM引导的方法）进行了比较，HapBO在HPOBench上表现出最低的简单遗憾，并在真实世界任务中也表现良好，且在查询预算为12-20时优化结果显著优于其他方法。

**⚠️ 局限性**

限制在于信念的质量依赖于如何将信息呈现给LLM，且HapBO采用一次性引导，无法在优化过程中更新信念。

---

## 18. BuggyMLE: Scaling Machine Learning Engineering Difficulty with Bug Injection

**arXiv ID:** 2611.25088 | [PDF](https://arxiv.org/pdf/2611.25088v1)

**作者:** John Wu `[一作]`, Xin Liu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了一种新的代码复杂性指标（Code Complexity Index），用于量化编码基准的难度，并开发了一种合成的任务生成框架，以控制机器学习工程任务的难度。

**💡 创新点**

创新点在于提出了一个基于代码结构的复杂性指标，能够准确预测编码基准的解决率，并通过注入故障生成具有可控难度的任务，填补了现有基准的不足。

**🔧 技术方法**

使用了四参数逻辑回归模型来预测基准的平均解决率，并利用大型语言模型（LLM）代理进行故障注入和任务生成。

**📊 数据集**

使用了来自MLE-Smith和PyHealth的合成生成和人工设计的代码库，生成了多达6674个执行验证的故障注入变体。

**📈 对比分析**

与现有方法相比，提出的框架在控制任务难度和生成多文件代码库的能力上表现优越，能够在不同难度层次上生成任务，并且在多个模型上验证了解决率的下降趋势。

**⚠️ 局限性**

限制在于该指标主要依赖于结构特征，未能捕捉语义复杂性，并且在独立生成的合成集上的预测能力较弱。此外，尽管训练增强了软件工程和代码修复能力，但在从头生成机器学习解决方案时的转移效果不佳。

---

## 19. TETHER: Training-Free, Elastic, ToC-Anchored Hierarchy Extraction and Recovery in Design-Driven Long Documents

**arXiv ID:** 2611.24960 | [PDF](https://arxiv.org/pdf/2611.24960v1)

**作者:** Yu Zou `[一作]` (Southwestern University of Finance and Economics), Changyu Hu `[通讯]` (Ningbo University of Finance and Economics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为TETHER的框架，用于将复杂文档转换为结构化知识，特别关注目录（ToC）作为宏观结构的利用。

**💡 创新点**

创新点在于将目录视为重要的结构先验，通过两阶段的神经符号机制实现文档的层次恢复，克服了现有方法在高熵布局中的脆弱性。

**🔧 技术方法**

使用了大型多模态模型（LMM）和基于锚点的对齐策略，构建文档骨架并将内容块与之对齐。

**📊 数据集**

使用了一个多语言基准数据集，包括环境、社会和治理（ESG）报告、产品手册和年度报告，涵盖英语、简体中文和繁体中文。

**📈 对比分析**

与深层层次恢复模型、专用解析器和通用LMM进行了比较，TETHER在零-shot层次恢复中表现出色，准确率超过96%，并在效率上实现了90%的令牌减少和80%的运行时间缩短。

**⚠️ 局限性**

局限性在于TETHER主要针对具有明确目录的设计驱动长文档，对于缺乏目录的文档不适用。此外，尽管是无训练框架，但仍依赖于大型多模态和语言模型，实际应用受到模型可用性和推理成本的限制。

---

## 20. WHU-Road3D: A dataset and benchmark towards fine-grained road surface mapping

**arXiv ID:** 2611.25873 | [PDF](https://arxiv.org/pdf/2611.25873v1)

**作者:** Xiaoxin Mi `[一作]` (Wuhan University of Technology), Shengwu Xiong `[通讯]` (Wuhan University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了WHU-Road3D数据集，这是一个用于细粒度道路表面元素提取和映射的大规模多模态数据集，涵盖了约100公里的中国高速公路和城市街道。

**💡 创新点**

WHU-Road3D是首个针对如此全面的道路表面元素进行映射的多模态数据集，提供了丰富的语义标注和多种数据模态，旨在促进实例分割和全元素提取与映射的前沿研究。

**🔧 技术方法**

使用了移动激光扫描（MLS）点云、全景图像、车辆轨迹以及详细的矢量化标注等技术。

**📊 数据集**

数据集包含约98公里的城市街道和高速公路，涵盖了15种不同类别的道路元素，包括道路边界、斑马线、车道分隔线和方向符号等。

**📈 对比分析**

通过与现有方法（如WHU-Lane和MapTR）进行比较，WHU-Road3D在提取和映射道路元素方面表现出色，尤其是在车道分隔线和道路边界的精度和召回率上有显著提升。

**⚠️ 局限性**

数据集的局限性在于城市道路的动态对象和数据缺失问题，尤其是在繁忙的城市环境中，可能会影响道路元素的提取和映射效果。

---

## 21. Contrast Where It Matters: Visual Evidence-Guided Decoding for Mitigating Hallucinations in LVLMs

**arXiv ID:** 2611.25524 | [PDF](https://arxiv.org/pdf/2611.25524v1)

**作者:** Yiheng Li `[一作]` (University of Chinese Academy of Sciences), Yang Yang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为视觉证据引导解码（VEGD）的训练无关框架，旨在通过识别与查询相关的视觉证据并在对比解码过程中引导其使用，从而减少大型视觉语言模型（LVLMs）中的幻觉现象。

**💡 创新点**

创新点在于通过选择性地减弱辅助图像中的视觉证据，并在解码过程中引导主分支和辅助分支关注相同的证据位置，从而有效减少幻觉现象。

**🔧 技术方法**

使用了训练无关的对比解码技术，结合视觉证据的定位和引导策略。

**📊 数据集**

在多个基准数据集上进行实验，包括CHAIR、POPE和MME-Hallucination，评估VEGD的性能。

**📈 对比分析**

与现有方法相比，VEGD在多个LVLMs和基准测试中一致减少了幻觉现象，表现出更好的准确性和精确度，尤其在POPE和CHAIR数据集上取得了显著的性能提升。

**⚠️ 局限性**

限制在于该方法依赖于选择的定位头的有效性，可能在某些情况下无法充分捕捉到所有相关的视觉证据。

---

## 22. Infer3D: Inverse Generative Modeling Enables Robust 3D Reconstruction

**arXiv ID:** 2611.25061 | [PDF](https://arxiv.org/pdf/2611.25061v1)

**作者:** Amani Kiruga `[一作]` (Harvard University), Ayush Tewari `[通讯]` (University of Cambridge)

**通讯引用:** 9085 | [OpenAlex ID](https://openalex.org/A5075797855)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Infer3D的逆生成建模框架，用于从单幅图像中恢复三维结构，特别是在面对未见过的场景条件时。

**💡 创新点**

创新点在于将感知视为逆生成问题，通过最大化输入观察的解释可能性来优化场景几何和渲染参数，同时引入自适应推理策略以提高对模糊输入的处理能力。

**🔧 技术方法**

使用了逆生成建模、最大后验估计和自适应推理策略等技术，结合了学习的二维生成模型和冻结的三维提升网络。

**📊 数据集**

在多种控制的合成和真实世界任务上进行了评估，具体数据集包括CO3D和RE10K等。

**📈 对比分析**

与基于前馈网络的方法相比，Infer3D在未见过的视角、非线性镜头失真和未观察到的纹理等情况下表现出更好的重建准确性，且在多个指标上超越了现有的最先进方法。

**⚠️ 局限性**

尽管Infer3D在OOD恢复方面表现出色，但在分布内数据上的定量指标可能略低于专门的前馈网络。此外，重建的视觉清晰度受限于所使用的二维生成先验的能力。

---

## 23. CoachSI: Self-Improvement of Robot Policies with a Language Coach

**arXiv ID:** 2611.25529 | [PDF](https://arxiv.org/pdf/2611.25529v1)

**作者:** Jiachen Tao `[一作]` (University of Illinois at Chicago), Yan Yan `[通讯]` (University of Illinois at Chicago)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种自我改进方法，通过语言指导来帮助机器人政策完成长时间任务中的下一步操作。

**💡 创新点**

创新点在于使用一个冻结的视觉-语言模型（VLM）作为教练，提供下一步指令，从而解决机器人在长时间任务中停滞的问题。

**🔧 技术方法**

使用了冻结的视觉-语言模型（VLM）进行语言指导，以及后视重标定和微调技术。

**📊 数据集**

使用了RoboCasa365数据集，该数据集包含原子任务和复合任务。

**📈 对比分析**

与人类演示和自我收集的成功进行比较，经过微调后，未辅助成功率从47.4%提高到59.3%，超出人类演示9.8个百分点，自我收集成功7.3个百分点。

**⚠️ 局限性**

限制在于该方法依赖于冻结的VLM教练，可能在某些复杂场景中无法提供足够的指导。

---

## 24. Improved Bounds for Directed SSSP in Sparse and Intermediate-Density Graphs

**arXiv ID:** 2611.26453 | [PDF](https://arxiv.org/pdf/2611.26453v1)

**作者:** Valery `[一作]`, Aryeh Lev Zabokritskiy `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种确定性算法，用于在具有n个顶点、m条边和非负实边权的有向图中计算精确的单源最短路径（SSSP）。

**💡 创新点**

该算法在比较-加法模型中，运行时间在稀疏情况下有所改进，特别是在边数m为Θ(n)时，减少了对loglog n的指数，从1/2降低到1/4，并且在中间密度范围内也有改进。

**🔧 技术方法**

使用了比较-加法模型中的单位成本基本操作，并结合了固定等级的递归和工作列表方法。

**📊 数据集**

使用了具有n个顶点和m条边的有向图，具体数据集未明确给出，但提到的边权为非负实数。

**📈 对比分析**

与Duan等人的方法进行了比较，证明了在边数m为Θ(n)和中间密度范围内的运行时间有显著改进，性能优于之前的算法。

**⚠️ 局限性**

算法的局限性在于其改进是渐近的，且在某些情况下可能需要回退到Dijkstra算法以确保精确性。

---

## 25. DriftGuard: Online Memory-Drift Detection via Causal Attribution for LLM Agents

**arXiv ID:** 2611.25583 | [PDF](https://arxiv.org/pdf/2611.25583v1)

**作者:** Licheng Pan `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在线内存漂移检测框架，旨在量化每个检索到的内存对潜在漂移的因果贡献，解决了内存增强的LLM代理在在线服务中的延迟预算问题。

**💡 创新点**

创新点在于提出了一个两阶段框架，分离了离线因果监督和在线估计，通过构建分解的内存漂移评分（MDS）来捕捉行动变化、攻击移除、违规移除和任务恢复。

**🔧 技术方法**

使用了离线配对反事实回放构建因果监督，并训练了一个在线风险模型（ORM），该模型能够直接从预行动信息中估计MDS，而无需在推理时调用任何反事实回放。

**📊 数据集**

在多个内存中毒基准（如MINJA、AgentPoison和MPBench）上进行了实验，评估了该方法的有效性。

**📈 对比分析**

与多种基线方法进行了比较，结果显示该方法在识别漂移诱导内存方面表现优异，尤其是在MPBench基准上，AUPRC显著高于其他方法，证明了其有效性。

**⚠️ 局限性**

局限性在于离线监督使用单一内存干预，仍需代理、工具和评估者回放进行数据构建，未能捕捉多个检索内存之间的交互。此外，评估主要集中在工具使用代理上，标记内存的中和在领域转移或未见攻击下的安全性尚未得到保证。

---

## 26. Beyond Restoration: Content-Degradation Factor Unfolding for Real-World Concealed Object Segmentation

**arXiv ID:** 2611.25239 | [PDF](https://arxiv.org/pdf/2611.25239v1)

**作者:** Rihan Zhang `[一作]` (Duke University), Sina Farsiu `[通讯]` (Duke University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的因子展开网络（FUN），用于在退化条件下进行隐蔽物体分割（COS），通过明确区分内容因子和退化因子来提高视觉感知的可靠性。

**💡 创新点**

创新点在于首次提出了任务不对称的广义Retinex公式，将内容和退化因子进行因子化，并通过可靠性加权进行前景与背景的分离。

**🔧 技术方法**

使用了因子展开网络（FUN），结合了任务不对称的广义Retinex模型和统一的优化模型，进行内容和退化的因子分解。

**📊 数据集**

在多个数据集上进行了广泛的实验，包括干净和退化条件下的隐蔽物体分割任务，具体数据集未详细列出。

**📈 对比分析**

与现有的RGB空间分割、恢复基础方法等进行了比较，FUN在干净COS和合成退化条件下表现出竞争力，尤其在复杂的退化情况下表现优越。

**⚠️ 局限性**

限制在于FUN在训练过程中没有使用干净图像的监督，可能影响其在某些特定情况下的性能表现。

---

## 27. StarMPI-SVD: Distributed-Memory Parallel Star-M Tensor SVD

**arXiv ID:** 2611.26451 | [PDF](https://arxiv.org/pdf/2611.26451v1)

**作者:** Naman Pesricha `[一作]` (Indian Institute of Science), Phani Motamarri `[通讯]` (Indian Institute of Science)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出了StarMPI-SVD，这是第一个分布式内存的Star-M张量奇异值分解（SVD）实现，适用于任意阶数的张量。

**💡 创新点**

创新点在于设计了异步非阻塞通信策略，以提高数据重分配的效率，并实现了强扩展性，能够在多达128个节点上保持至少70%的并行效率。

**🔧 技术方法**

使用了MPI进行进程间通信，结合BLAS和LAPACK进行局部计算，采用了零拷贝批量张量-矩阵（TTM）公式和双缓冲重叠执行策略。

**📊 数据集**

使用了五个科学数据集，数据集大小从78GB到768GB不等，包括流体模拟和气候重分析数据。

**📈 对比分析**

与现有的共享内存实现相比，StarMPI-SVD在单节点上运行时间相当，但在处理大数据集时表现出更好的扩展性，能够在32个节点上运行而不受内存限制。

**⚠️ 局限性**

限制在于当前实现仍然依赖于特定的张量分布策略，可能在某些情况下影响性能，且对数据集的大小和结构有一定的要求。

---

## 28. Beyond Entity Recognition: Vision-Centric Knowledge Visual Question Answering

**arXiv ID:** 2611.25680 | [PDF](https://arxiv.org/pdf/2611.25680v1)

**作者:** Jiatong Ma `[一作]` (Institute of Automation), Jing Liu `[通讯]` (Institute of Automation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了VC-VQA，一个大规模的视觉中心知识推理基准，旨在系统评估多模态大型语言模型（MLLMs）在细粒度视觉知识获取和复杂多阶段推理中的能力。

**💡 创新点**

VC-VQA明确建模视觉信息的角色，考虑其来源（内部与外部图像）和参与阶段（检索前与检索后），并定义了三种代表性的推理任务。

**🔧 技术方法**

使用了多模态知识库和多种模型评估方法，包括无外部知识、提供证据、知识库检索和开放网络搜索等四种设置。

**📊 数据集**

构建了一个包含约30,000个样本的多模态知识库，样本包括26055个独特图像和12395个独特问题，数据来源于维基百科快照。

**📈 对比分析**

在四种设置下评估了21个代表性的MLLMs，结果显示当前模型在视觉中心知识推理上仍然存在困难，细粒度视觉知识获取是主要瓶颈，主动检索方法在所有设置中表现优于启发式检索。

**⚠️ 局限性**

限制包括数据集中可能仍存在少量错误或模糊样本，当前基准仅限于英语，不支持多语言或跨语言评估，以及未系统研究不同推理策略的影响。

---

## 29. Web search in Latin America: language models use less local and diverse news than search engines

**arXiv ID:** 2611.25955 | [PDF](https://arxiv.org/pdf/2611.25955v1)

**作者:** Francisco W. Kerche `[一作]` (University of Oxford), Luc Rocher `[通讯]` (University of Oxford)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了在拉丁美洲，用户从传统搜索引擎转向基于大型语言模型（LLM）的搜索时，关于拉丁美洲的信息如何变化。研究构建了一个包含1901个当前事件的数据集，并向三种搜索引擎和四种语言模型提交查询，分析了信息的多样性和本地新闻的可见性。

**💡 创新点**

创新点在于系统比较了传统搜索引擎与LLM在信息检索中的表现，揭示了LLM在返回新闻来源的多样性和本地新闻比例方面的不足，尤其是在拉丁美洲的背景下。

**🔧 技术方法**

使用了大型语言模型（如GPT、Claude、Perplexity和Grok）和传统搜索引擎（如Google、Bing和Yahoo）进行比较分析。

**📊 数据集**

使用了全球事件数据库（GDELT）中的数据集，涵盖了九个拉丁美洲国家的1901个当前事件。

**📈 对比分析**

与搜索引擎相比，LLM返回的新闻来源更少样化，尤其是在本地用户使用本地语言搜索时，LLM返回的本地新闻比例显著低于搜索引擎，显示出LLM在信息多样性和本地新闻可见性方面的劣势。

**⚠️ 局限性**

限制在于研究结果可能无法推广到其他拉丁美洲以外的国家，且未考虑用户的历史交互对信息检索的影响。

---

## 30. Can World Models Predict Rice Growth? Toward Agricultural World Models

**arXiv ID:** 2611.26145 | [PDF](https://arxiv.org/pdf/2611.26145v1)

**作者:** Wenkai Ji `[一作]` (East China Normal University), Zhenong Jin `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种农业世界建模的方法，旨在从一张植物照片及其品种和视角预测植物在未来几周到几个月的生长情况。

**💡 创新点**

创新点在于通过推理模块先预测植物的生长阶段，然后再生成与该预测一致的图像，从而解决了现有图像生成模型在植物生长预测中的不足。

**🔧 技术方法**

使用了视觉-语言模型作为推理模块，并结合图像编辑模型作为生成模块。

**📊 数据集**

使用了公开的水稻数据集WGSR和玉米数据集WGSM，分别包含多种水稻和玉米的生长阶段图像。

**📈 对比分析**

与现有的图像生成模型（如GPT Image 2.5和Qwen Image 2.1）进行比较，提出的方法在水稻的Stage R@1上达到54.6%，显著优于其他模型的34.8%和24.0%。在玉米上也表现出最佳性能。

**⚠️ 局限性**

限制在于实验主要基于温室中的盆栽植物，背景被移除，且大多数品种在训练中已见过，未来工作将扩展到天气和管理记录，并测试新的品种和更多物种的田间图像。

---

## 31. Multiple-Camera Tracking of Rigid Objects

**arXiv ID:** 2611.25904 | [PDF](https://arxiv.org/pdf/2611.25904v1)

**作者:** Frédérick Martin `[一作]` (INRIA Rhone-Alpes), Radu Horaud `[通讯]` (INRIA Rhone-Alpes)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种使用单个或多个摄像头跟踪刚性物体的方法，通过测量和最小化预测模型点与图像轮廓之间的图像误差来实现跟踪。

**💡 创新点**

创新点在于详细研究了与最小化过程相关的雅可比矩阵，并建立了点对点和点对轮廓匹配之间的数学联系，展示了多摄像头配置在处理单摄像头奇异性、噪声或缺失数据时的鲁棒性。

**🔧 技术方法**

使用了视觉伺服控制方法，结合了雅可比矩阵的分析和多摄像头系统的实现。

**📊 数据集**

实验使用了复杂的船舶部件模型，并在最多三个摄像头的配置下进行验证。

**📈 对比分析**

与传统单摄像头方法相比，使用多个摄像头可以避免大多数单摄像头的奇异性，实验结果表明该方法在处理部分遮挡和图像质量不佳的情况下表现良好。

**⚠️ 局限性**

该方法的局限性在于仅处理边缘信息，复杂的自然和工业环境中可能会出现失败，因此计划结合纹理信息以提高鲁棒性和可靠性。

---

## 32. MV-JEPA: Post-Training Video Encoders for Cross-View Understanding

**arXiv ID:** 2611.25172 | [PDF](https://arxiv.org/pdf/2611.25172v1)

**作者:** Yangchen Zhao `[一作]` (University of Minnesota), Caiwen Ding `[通讯]` (University of Minnesota)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种多视角预测后训练方法（MV-JEPA），旨在增强JEPA视频编码器在不同相机视角下的实例和时间对应性，同时保持单视角推理的独立性。

**💡 创新点**

创新点在于通过短期的多视角预测后训练来增强跨视角对应性，而不需要修改编码器架构或单视角推理接口。

**🔧 技术方法**

使用了多视角预测和单视角重放的联合训练技术。

**📊 数据集**

使用了两个多视角数据集：MVImgNet2.0和Ego-Exo4D，前者包含不同视角下的日常物体图像，后者包含同步的自我中心和外部中心人类活动录制。

**📈 对比分析**

与其他方法相比，MV-JEPA在多个数据集上提高了跨视角实例检索和粗略跨相机时刻检索的性能，具体表现为在ABO数据集上提高了9.4个百分点，在Ego-Exo4D上提高了14.6个百分点。

**⚠️ 局限性**

限制在于该方法仍然依赖于预训练的编码器，可能在某些特定任务上表现不如专门训练的模型。

---

## 33. Keel: A Frontier Model Teaches a Vision-Language Model by Rewriting Its Weights

**arXiv ID:** 2611.25257 | [PDF](https://arxiv.org/pdf/2611.25257v1)

**作者:** Jiajun Fan `[一作]` (University of Illinois Urbana-Champaign), Ge Liu `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文提出了一种新的方法，通过编辑视觉语言模型（VLM）的权重来修复思维循环问题，从而提高模型在图像任务上的准确性。

**💡 创新点**

创新点在于首次引入了一种代理权重编辑方法，允许模型在不进行训练的情况下识别并修复其内部错误，特别是在图像任务中。

**🔧 技术方法**

使用了一种前沿的多模态大语言模型（LLM）作为编辑器，通过读取和重写VLM的权重来进行修复，而不依赖于梯度或训练。

**📊 数据集**

使用了ViRL39K数据集，该数据集包含几何、图表、科学和空间推理等可验证的视觉问题，并在多个图像基准上进行评估。

**📈 对比分析**

与现有方法（如思维模型、存在惩罚、反射抑制等）相比，编辑后的模型在多个基准测试中表现出显著更高的准确性，并且生成的答案更短，减少了思维循环的发生。

**⚠️ 局限性**

限制在于编辑过程依赖于编辑器的信号质量，且在某些情况下，编辑可能无法完全消除所有循环，仍需进一步优化。

---

## 34. ACE-Brain-0.7: Learning Execution-Grounded Reasoning for Embodied Agents

**arXiv ID:** 2611.26519 | [PDF](https://arxiv.org/pdf/2611.26519v1)

**作者:** Yinhe Wang `[一作]`, Xiaogang Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了一种新的执行基础推理方法，旨在通过观察任务的进展和结果来判断任务的执行状态，并结合任务目标与环境观察来协调机器人技能。

**💡 创新点**

创新点在于构建了一个包含370万个问题-答案对的训练集，专注于任务执行的监督学习，并提出了一种新的9B视觉-语言模型，能够支持空间定位、任务理解和进度估计。

**🔧 技术方法**

使用了视觉-语言模型（VLM）技术，并通过监督微调来提升模型的性能。

**📊 数据集**

使用的数据集包括来自人类第一人称视频、机器人执行记录和模拟数据的370万个问题-答案对。

**📈 对比分析**

与其他方法相比，该模型在六个公共任务推理基准上表现优异，平均得分为76.5，超越了所有评估的开放模型，并在任务推理和空间定位基准上均取得最高分。

**⚠️ 局限性**

限制在于模型的性能可能受到训练数据的质量和多样性的影响，尤其是在处理复杂任务和失败情况时，可能需要更多的真实世界数据来进一步提升模型的鲁棒性。

---

## 35. Fixing the Twin, Not the Detector: Sim-to-Real Roadside LiDAR Detection from Digital-Twin Data

**arXiv ID:** 2611.26035 | [PDF](https://arxiv.org/pdf/2611.26035v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 36. Accord: Universal Online Metric Localization and Dense Mapping from one Camera and an IMU

**arXiv ID:** 2611.25279 | [PDF](https://arxiv.org/pdf/2611.25279v1)

**作者:** Jaafar Mahmoud `[一作]` (ITMO University), Sergey Kolyubin `[通讯]` (ITMO University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在线密集度量映射系统，利用单个相机和IMU构建准确的密集度量地图。

**💡 创新点**

创新点在于通过关键帧窗口锚定度量3D点，并利用双向密集深度一致性来估计和修正每幅图像的尺度误差，同时保持由锚点建立的度量尺度。

**🔧 技术方法**

使用了视觉-惯性里程计（VIO）滤波器、条件深度基础模型（Depth Anything 3）和因子图融合技术。

**📊 数据集**

使用了五个公共相机-IMU设置和一个新的手机-运动相机数据集进行评估。

**📈 对比分析**

与五个基线方法进行比较，在每个公共数据集上都取得了最佳或第二最佳的平均F-score，且在每个平台上均在5厘米的容差内表现良好。

**⚠️ 局限性**

限制在于系统对初始标定的敏感性，且在大规模和户外环境中仍然面临挑战。

---

## 37. FinVeriQA: Financial Research Agents for Question Answering: Routing, Retrieval, MCP Tool Use, and Provenance-Aware Evidence Verification

**arXiv ID:** 2611.25071 | [PDF](https://arxiv.org/pdf/2611.25071v1)

**作者:** Meenakshi Rajpurohit `[一作]` (San Jose State University), Nathan Howland `[通讯]` (San Jose State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究开发了一个多代理系统，用于金融问答助手，旨在确定每个任务的最佳执行组件，包括问题路由、数值计算、证据检索和答案撰写。

**💡 创新点**

创新点在于将不同的任务分配给专门的代理，使用了一个紧凑的类型决策模型（Laya）来优化问题路由，并通过独立验证机制提高了答案的准确性。

**🔧 技术方法**

使用了多种技术，包括Python进行数值计算、Mistral-7B进行答案撰写、以及一个共享内存系统来支持代理之间的证据共享。

**📊 数据集**

使用的数据集包括SEC文件、实时市场数据和FinRank基准测试数据。

**📈 对比分析**

与其他方法的比较显示，Laya在路由方面表现优于Mistral-7B（0.85 vs. 0.52），而Mistral-7B在情感分析上表现最佳（0.93）。整体性能最高的混合模型得分为0.88。

**⚠️ 局限性**

限制在于测试集较小且部分为合成数据，模型的生成答案质量未被充分评估，且仍需改进对引用准确性的检查。

---

## 38. Thermodynamically Consistent Neural Networks for a Universal Fluid Gibbs Free Energy Equation of State

**arXiv ID:** 2611.26183 | [PDF](https://arxiv.org/pdf/2611.26183v1)

**作者:** Xurui Li `[一作]` (Tsinghua University), Diannan Lu `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究基于吉布斯自由能，提出了一种热力学一致的神经网络框架，构建了流体状态方程（GEoS），提供了流体状态的统一表示。

**💡 创新点**

创新点在于将热力学约束嵌入神经网络模型中，确保模型预测的热力学一致性，并且通过自动微分计算流体的PVT关系。

**🔧 技术方法**

使用了热力学一致的神经网络技术，结合分子描述符和状态变量作为输入，构建预测模型。

**📊 数据集**

使用了CoolProp数据库中的104种物质的逸度系数数据，构建了测试集，包括74078个液相密度数据点和9665个气相密度数据点。

**📈 对比分析**

与现有的机器学习模型（如SPT-PC-SAFT和ML-SAFT）及传统的立方状态方程（如tc-PR和tc-SRK）进行比较，GEoS在气相和液相密度预测上均表现优异，AARD分别为5.25%和3.96%，优于大多数对比模型。

**⚠️ 局限性**

模型在低压液相和超高压区域的预测存在一定异常，未来计划扩展训练集以提高模型的通用性，并拟合PC-SAFT参数以消除这些异常。

---

## 39. Decoupled Knowledge Retrieval and Compact Memory Enable Reliable Lifelong Model Editing

**arXiv ID:** 2611.25861 | [PDF](https://arxiv.org/pdf/2611.25861v1)

**作者:** Yue Che `[一作]` (Hong Kong Polytechnic University), Mingkun Xu `[通讯]` (Guangdong Institute of Intelligence Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为BiSpace的终身模型编辑框架，旨在在不进行昂贵的重新训练或损害现有能力的情况下，将新知识持续融入已部署的大型语言模型（LLMs）中。

**💡 创新点**

BiSpace通过将内存检索与知识注入解耦，解决了现有内存编辑器在存储需求和检索性能方面的不足，显著提高了编辑性能和存储效率。

**🔧 技术方法**

使用了冻结的句子变换器（Sentence Transformer）构建低维编辑键，并结合现代霍普菲尔德网络（Modern Hopfield Networks）进行检索和更新。

**📊 数据集**

在ZsRE、WikiBigEdit和CounterFact等三个广泛采用的终身模型编辑基准上进行了评估，使用了LLaMA-3.1-8B-Instruct和Qwen2.5-7B-Instruct等模型。

**📈 对比分析**

与HoReN等现有方法相比，BiSpace在处理多达16K次编辑时，整体性能提高了23%以上，并在存储需求上显著降低，每个条目的存储需求仅为原来的约10.2%。

**⚠️ 局限性**

尽管BiSpace在存储效率和性能上表现出色，但在面对大量编辑时，仍可能出现性能下降，尤其是在处理复杂或长形式知识时，当前的稀疏注入策略可能无法有效存储所有信息。

---

## 40. Dune: A Zeroth-Order Optimizer with Time-Conditioned Historical Error Regression for VLA Test-Time Adaptation

**arXiv ID:** 2611.25400 | [PDF](https://arxiv.org/pdf/2611.25400v1)

**作者:** Chengyang Dong `[一作]` (University of Glasgow), Jialong Sun `[通讯]` (Heilongjiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Dune的零阶优化器，通过结合在线历史误差回归和当前扰动观察，解决了在长时间范围内进行视觉-语言-动作（VLA）模型测试时适应的性能平台问题。

**💡 创新点**

Dune通过在线岭回归预测局部误差信号，并与当前的零阶扰动观察相结合，实现了稳定的更新，避免了全局反向传播或额外的前向传递。

**🔧 技术方法**

使用了零阶优化技术，结合历史统计数据进行误差预测，采用岭回归来降低估计方差。

**📊 数据集**

在LIBERO基准和真实世界机器人上进行了广泛实验，使用了240M参数的SmolVLA模型进行测试。

**📈 对比分析**

与标准的第一阶优化器（如AdamW）和其他零阶优化器（如Dust）进行比较，Dune在任务成功率上表现优越，且在内存占用上显著降低，证明了其在长时间适应中的有效性。

**⚠️ 局限性**

Dune的一个限制是，尽管显著减少了GPU内存需求，但在墙钟时间上收敛速度较慢，运行相同的训练工作负载所需时间是Adam的几倍，这限制了其实际应用。

---

## 41. Learning to Compare Neural Representations from Parameterized Computation Graphs

**arXiv ID:** 2611.26081 | [PDF](https://arxiv.org/pdf/2611.26081v1)

**作者:** Ya-Wei Eileen Lin `[一作]` (Technical University of Munich), Stefanie Jegelka `[通讯]` (MIT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文研究了如何通过学习模型的参数化计算来预测未见模型的内部表示相似性，而不是依赖于直接在已知输入上评估候选模型。

**💡 创新点**

创新点在于提出了一种图元网络框架，能够在不需要新基模型激活的情况下，通过已训练模型的参数化计算来预测内部表示的相似性。

**🔧 技术方法**

使用了图元网络技术，该技术能够编码网络参数及其连接性，并通过图结构表示每个内部表示站点的上游计算。

**📊 数据集**

使用了多个独立训练的模型的内部表示作为数据集，具体包括ResNet和ViT模型在CIFAR-10和CIFAR-100数据集上的表现。

**📈 对比分析**

与多种基线方法（如CKA、GULP等）进行了比较，结果表明，所提出的方法在表示关系预测和层检索任务中表现优越，尤其在ViT模型上取得了显著的提升。

**⚠️ 局限性**

限制在于该方法需要进行元网络的训练和初始图编码，可能不适合一次性评估，而更适合重复比较或数据受限的场景。

---

## 42. CRAFT: Constrained Route planning with Adaptive Framework and Template-based code generation

**arXiv ID:** 2611.25128 | [PDF](https://arxiv.org/pdf/2611.25128v1)

**作者:** Shahd Elmahallawy `[一作]` (University of California), Ahmed Eldawy `[通讯]` (University of California)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为CRAFT的框架，能够处理自然语言的约束路由查询，并返回满足用户需求的路线。

**💡 创新点**

CRAFT通过将约束路由查询转化为可执行的SE-RCSP实例，解决了传统算法无法处理的多种约束组合问题。

**🔧 技术方法**

使用了大型语言模型（LLM）来解析查询，并生成四个搜索函数，结合一个通用的图搜索引擎来执行路由计算。

**📊 数据集**

在加利福尼亚的真实道路网络上评估CRAFT，使用了2600个自然语言查询进行测试。

**📈 对比分析**

与两个基线方法（LLMAP和LLM-only）进行比较，CRAFT在主要基准上正确率高达93%，而最强基线最多为42%。CRAFT在88%的查询中返回最佳路线。

**⚠️ 局限性**

CRAFT的局限性在于当前所有约束都被视为硬性约束，未来希望能够根据重要性对约束进行排序，并在没有满足所有约束的情况下，逐步放宽次要约束。

---

## 43. Attacks and Defenses in Decentralized Multi-Agent Systems with Hop-by-Hop Task Decomposition and Delegation

**arXiv ID:** 2611.25056 | [PDF](https://arxiv.org/pdf/2611.25056v1)

**作者:** Matthew D. Laws `[一作]` (Northeastern University), Cristina Nita-Rotaru `[通讯]` (Northeastern University)

**通讯引用:** 8968 | [OpenAlex ID](https://openalex.org/A5034838576)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了去中心化多智能体系统中的恶意代理威胁，并提出了相应的攻击和防御技术。通过设计和评估针对完整性、可用性和保密性的攻击，展示了单个恶意代理可以导致高达100%的任务失败，并显著增加令牌消耗和运行时间。

**💡 创新点**

创新点在于提出了一种加密协议，能够生成不可否认的验证执行轨迹，记录系统中每个动作的执行，并开发了一个审计器来分析这些轨迹以检测和归因恶意行为。

**🔧 技术方法**

使用了加密协议和审计器技术，结合了数字签名和哈希链来确保执行轨迹的完整性和真实性。

**📊 数据集**

使用了HiddenBench数据集进行评估，该数据集构建了需要多个信息片段聚合才能解决的任务。

**📈 对比分析**

与现有方法相比，本文的协议在延迟开销上几乎可以忽略不计，而审计器能够检测并正确归因98%的攻击，显示出其有效性。

**⚠️ 局限性**

限制在于该协议假设用户和代理是诚实的，且恶意代理不进行合谋，可能在某些情况下无法完全防范复杂的攻击模式。

---

## 44. Client Exits - A Way to Increase LLM Transparency on the Client Side

**arXiv ID:** 2611.26309 | [PDF](https://arxiv.org/pdf/2611.26309v1)

**作者:** Marc Jansen `[一作]` `[通讯]` (University of Applied Sciences Ruhr West), Marc Jansen (University of Applied Sciences Ruhr West)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Client Exits的轻量机制，用于增强大型语言模型（LLM）响应的透明度，使语义意图与客户端特定的呈现分离。

**💡 创新点**

创新点在于通过在LLM输出中插入预定义的机器可解释标记，允许接收客户端决定如何实现相应的增强，从而支持不同的呈现方式而不强制具体接口。

**🔧 技术方法**

使用了微调的Qwen模型和修改过的LibreChat客户端，结合了Question–Option–Criteria (QOC) 方法。

**📊 数据集**

使用了基于QOC的训练数据集，包含100个测试问题，评估了带有和不带有客户端退出标记的模型变体。

**📈 对比分析**

通过LLM作为评判者的方法比较了两种模型的响应，结果显示带有客户端退出标记的模型在66%的比较中被偏好，表明该模型的答案在多数情况下更受欢迎。

**⚠️ 局限性**

限制在于未能证明带有客户端退出的模型在事实准确性或用户理解方面的改善，且评估结果仅基于一个基础模型和特定的思维策略。

---

## 45. Support-Guided Triadic Decoding for Hallucination Mitigation in Multimodal Dialogue with LVLMs

**arXiv ID:** 2611.25535 | [PDF](https://arxiv.org/pdf/2611.25535v1)

**作者:** Yiheng Li `[一作]` (University of Chinese Academy of Sciences), Yang Yang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为支持引导三元解码（SGTD）的训练无关框架，用于在多模态多轮对话中选择性地控制不可靠历史的影响，并在解码过程中融合互补的预测。

**💡 创新点**

SGTD通过历史答案支持估计、支持目标注意干预和三元解码的结合，创新性地解决了在保留有用上下文的同时抑制误导性信息的问题。

**🔧 技术方法**

使用了支持引导三元解码（SGTD）框架，结合历史答案支持估计、支持目标注意干预和三元解码技术。

**📊 数据集**

在MMHalSnowball和VisDiaHalBench数据集上进行了实验，这两个数据集包含了事实和幻觉历史的多模态对话。

**📈 对比分析**

与现有方法（如RVD）相比，SGTD在幻觉历史下的准确性显著提高，同时保持了事实历史的优势，表现出更强的鲁棒性和更低的幻觉下降率。

**⚠️ 局限性**

SGTD的局限性在于其依赖于历史答案的支持估计，可能在某些情况下无法完全消除所有误导性信息的影响。

---

## 46. Squoosh Bench: Evaluating UX-1 on 318 Historical Website A/B Tests

**arXiv ID:** 2611.25200 | [PDF](https://arxiv.org/pdf/2611.25200v1)

**作者:** Rohith Pasula `[一作]` (Squoosh AI), Tyler Dooskin `[通讯]` (Squoosh AI)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估模拟购物者的预测是否能帮助实验团队选择要测试的网站变更。筛选了412个历史A/B测试，评估了四个系统在318个运行超过21天的实验中的表现。

**💡 创新点**

创新点在于使用模拟购物者的预测来指导实验团队的决策，尤其是在选择变体时，UX-1模型表现出色，能够准确预测大多数变体的胜出情况。

**🔧 技术方法**

使用了Squoosh UX-1模型，该模型经过人类浏览数据的微调，模拟了1000名购物者的行为，并进行了多种系统的比较。

**📊 数据集**

使用了412个历史A/B测试的数据集，经过筛选后，318个实验符合条件，记录了113个原始胜利、137个变体胜利和68个不确定结果。

**📈 对比分析**

与其他三种通用系统相比，UX-1在318个实验中正确分类267个结果（84.0%），而Qwen3 Next 80B的准确率为61.6%。UX-1在126个偏向变体的实验中，93.7%获得了胜利。

**⚠️ 局限性**

限制在于模型的训练数据与实际网站的重叠程度未知，且未能评估所有可能的实验情况，可能导致预测的准确性受到影响。

---

## 47. Old-School Planners Drive (Sometimes) Better Than Learned Ones

**arXiv ID:** 2611.25138 | [PDF](https://arxiv.org/pdf/2611.25138v1)

**作者:** Andreas Look `[一作]` `[通讯]` (Coburg University of Applied Sciences and Arts), Andreas Look (Coburg University of Applied Sciences and Arts)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文实现了十六种经典规划器，结合路径源和轨迹方法，并在三个基准上进行评估，测试它们在现代自动驾驶中的表现。

**💡 创新点**

创新点在于将经典的规则基础规划器与现代端到端模型的感知输出结合，展示了这些经典规划器在当前基准上的竞争力。

**🔧 技术方法**

使用了经典的路径规划技术，如参考线、A*、混合A*、RRT*等路径源，结合Frenet格、非线性优化、s-t搜索和动态窗口等轨迹方法。

**📊 数据集**

使用了nuScenes、nuPlan和HUGSIM数据集进行评估，特别是在HUGSIM中，经典规划器替换了端到端模型UniAD的规划器。

**📈 对比分析**

与UniAD的规划器相比，最佳的经典规划器在nuScenes和nuPlan上表现出竞争力，L2误差为0.735米，而UniAD为0.947米。在HUGSIM中，九个经典规划器的得分超过了UniAD的学习规划器。

**⚠️ 局限性**

限制在于所有结果仅基于一个端到端模型，可能存在其他模型表现不同的情况。此外，HUGSIM的场景是固定场景的变体，可能不独立，且只插入了车辆，得分主要评估规划的路径，而非实际驾驶动作。

---

## 48. Understanding MoE Decisions Beyond Gradients

**arXiv ID:** 2611.26075 | [PDF](https://arxiv.org/pdf/2611.26075v1)

**作者:** Zhen Xu `[一作]` (University of Chicago), Xuefeng Liu `[通讯]` (University of Florida)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的预测方法，用于估计混合专家模型中替换专家对答案评分变化的影响。

**💡 创新点**

创新点在于结合了模型自身的专家输出和选择规则，以估计替换专家后对答案评分的额外影响，从而提高了预测的准确性。

**🔧 技术方法**

使用了基于梯度的方法和二阶预测模型，结合了专家选择的动态变化。

**📊 数据集**

使用了Qwen3.6和Gemma4两个模型的数据集，包含来自MMLU-Pro、SuperGPQA和HLE的158个问题。

**📈 对比分析**

与多步骤基线方法相比，提出的方法在Qwen3.6上减少了60%的预测误差，在Gemma4上减少了47%。

**⚠️ 局限性**

局限性在于实验仅覆盖了两个MoE模型和三个多项选择问题子集，且需要白盒访问和每次干预的曲率计算，当前实现的计算成本较高。

---

## 49. System-One RSI: Building Muscle Memory for Interactive Agents through Guided Practice

**arXiv ID:** 2611.26007 | [PDF](https://arxiv.org/pdf/2611.26007v1)

**作者:** Qingxu Fu `[一作]` (Alibaba Token Foundry), Bolin Ding `[通讯]` (Alibaba Token Foundry)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为System-One Recursive Self-Improvement (System-One RSI)的方法，将任务知识和交互经验转化为可重用的视觉判断，以提高实时控制能力。

**💡 创新点**

创新点在于结合了快速的内部决策循环和离线的外部改进循环，通过推理模型的团队来指导决策模型的改进，从而实现了决策模型的自我改进。

**🔧 技术方法**

使用了Observe, Orient, Decide, Act (OODA)控制器和推理模型，结合了在线控制和递归适应的框架。

**📊 数据集**

在Zenless Zone Zero和Black Myth: Wukong等动态视觉环境中进行了实验验证，使用了27B的决策模型。

**📈 对比分析**

通过与传统强化学习方法的比较，展示了在实时控制中的决策延迟显著降低，适应性提高，成功率和完成时间均有显著改善。

**⚠️ 局限性**

局限性包括对外部知识的依赖、在不同环境中的适应性可能不足，以及在资源有限的情况下可能需要调整模型大小和响应预算。

---

## 50. How Feedback-Free Self-Evolution Reshapes Agent Skills: A Systematic Study

**arXiv ID:** 2611.25534 | [PDF](https://arxiv.org/pdf/2611.25534v1)

**作者:** Litao Guo `[一作]` (Hong Kong University of Science and Technology), Ying-Cong Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在没有反馈的情况下，模型如何自我演化重写专家技能，并评估这种重写对技能价值的影响。

**💡 创新点**

首次系统性地研究了反馈自由自我演化对专家技能的影响，发现技能在重写过程中逐渐贬值，而非改善。

**🔧 技术方法**

使用了大型语言模型进行技能重写，并通过SkillsBench、LogicBench和MedCalc-Bench等基准进行评估。

**📊 数据集**

使用了SkillsBench、LogicBench和MedCalc-Bench等数据集，评估了不同模型在这些基准上的表现。

**📈 对比分析**

与原始技能相比，经过30轮重写后，所有模型的通过率平均下降了15个百分点，且在LogicBench和MedCalc-Bench上表现也显著下降。

**⚠️ 局限性**

限制在于重写过程中缺乏反馈，导致模型倾向于丢失专家提供的知识，尤其是那些模型自身不常写的内容。

---

## 51. MS-Point: Multi-Modal Hierarchical Spectral Learning for Point Cloud Understanding

**arXiv ID:** 2611.25282 | [PDF](https://arxiv.org/pdf/2611.25282v1)

**作者:** Hai-tao Yu `[一作]` (Hong Kong University of Science and Technology), Jun Xia `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种跨模态预训练框架，直接监督中间点特征及其邻域响应，以学习可转移的对象表示。

**💡 创新点**

创新点在于通过独立的视觉目标监督中间点特征和邻域响应，增强了特征的几何感知能力。

**🔧 技术方法**

使用了图编码器和点变换器的技术，结合了分支特定的对比损失和自适应融合。

**📊 数据集**

使用了ShapeNet点云和图像对作为数据集进行预训练和评估。

**📈 对比分析**

通过同背骨比较和容量匹配的控制实验，证明了该方法在对象分类、少样本适应、密集预测和低标签场景学习中的有效性，性能优于现有方法。

**⚠️ 局限性**

限制在于需要进一步评估在不同传感器、环境和人口背景下的表现，以及数据集特定偏差的影响。

---

## 52. When and why can integral constraints improve the accuracy of physics-informed neural networks?

**arXiv ID:** 2611.25515 | [PDF](https://arxiv.org/pdf/2611.25515v1)

**作者:** Myeong-Su Lee `[一作]` `[通讯]` (Research Institute of Mathematics), Myeong-Su Lee (Research Institute of Mathematics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了通过将积分约束整合到物理信息神经网络（PINNs）的损失函数中，以提高其在多个偏微分方程（PDE）应用中的准确性，并通过稳定性理论解释了这种改进的机制。

**💡 创新点**

创新点在于通过稳定性估计比较了包含和不包含积分约束的损失函数，揭示了在某些参数范围内，标准损失对特定误差成分的控制能力较弱，而积分约束能够恢复对这些成分的均匀控制，从而提高准确性。

**🔧 技术方法**

使用了经典的稳定性理论来分析损失与解的误差之间的关系，并通过数值实验验证了理论结果。

**📊 数据集**

研究中使用了多个偏微分方程的数据集，包括非线性椭圆方程、Helmholtz方程、静态粘性Burgers方程和时间依赖的反应扩散方程。

**📈 对比分析**

通过稳定性估计比较了包含和不包含积分约束的损失函数的性能。在标准损失控制能力较弱的参数范围内，使用增强损失的训练显著减少了误差，而在标准损失已经有效控制的范围内，两者的准确性相似。

**⚠️ 局限性**

限制在于虽然本文提供了对特定情况的解释，但并不声称所有通过添加积分约束获得的准确性提升都可以通过这种机制解释，未来的工作需要扩展到更广泛的PDE范围，并分析积分约束对优化景观、训练收敛性和泛化误差的影响。

---

## 53. HireWatch: Evaluating LLM Compliance with U.S. Employment Law Under Contextual Pressure

**arXiv ID:** 2611.24970 | [PDF](https://arxiv.org/pdf/2611.24970v1)

**作者:** Huy Nghiem `[一作]` (University of Maryland), Noam Kolt `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 3175 | [OpenAlex ID](https://openalex.org/A5092031121)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了HireWatch评估工具，评估大型语言模型（LLMs）在招聘过程中是否遵守美国法律，特别是与面试问题相关的法律。

**💡 创新点**

创新点在于构建了一个包含200个合法和非法面试问题的问卷库，并通过该库评估LLMs的法律合规性，这是首次专门针对美国联邦就业和歧视法进行的评估。

**🔧 技术方法**

使用了大型语言模型（LLMs）进行评估，并通过不同的上下文压力和法律合规指令来测试模型的反应。

**📊 数据集**

构建了一个包含200个问题的问卷库，其中包括150个合法问题和50个非法问题，这些问题的法律分类基于美国联邦法规和指导。

**📈 对比分析**

与人力资源专业人员的合规性进行了比较，发现LLMs在高压条件下的法律合规性显著下降，尽管在低压条件下合规性较高。法律合规指令可以提高合规率，但并不能完全消除法律违规。

**⚠️ 局限性**

限制在于评估仅针对特定的招聘任务，而未涵盖整个招聘过程的复杂性。此外，评估仅关注美国联邦法律，未考虑州法或其他法律管辖区的情况。

---

## 54. PsyWorld: Profile-Conditioned World Modeling for Human Behavior Prediction

**arXiv ID:** 2611.24999 | [PDF](https://arxiv.org/pdf/2611.24999v1)

**作者:** Ziyuan Qin `[一作]` (Emory University), Liang Zhao `[通讯]` (Emory University)

**通讯引用:** 9799 | [OpenAlex ID](https://openalex.org/A5012506279)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a2602d71-93ab-4bad-974b-672788df8193` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于个人资料的状态世界模型，通过观察到的环境-行为状态转变来学习行为预测的表示。

**💡 创新点**

创新点在于将稳定的心理特征与时间变化的状态分开，并利用实现的状态转变作为结构化监督来改进行为预测。

**🔧 技术方法**

使用了基于后验编码器和先验预测的模型，结合了向量量化和状态预测技术。

**📊 数据集**

使用了来自313名参与者的问卷资料和行为轨迹数据集，分为240个训练、25个验证和48个测试参与者。

**📈 对比分析**

与直接的SFT/BC方法相比，该模型在VEET上达到了0.904的Top-1准确率，显著提高了行为预测的准确性，并且在分布对齐方面表现良好。

**⚠️ 局限性**

模型的局限性在于未能独立评估多步状态转变的准确性，且未包含单独观察的动态情感通道，学习的代码是行为预测的总结，而非因果或临床状态。

---

## 55. VideoTrail: Interactive Video Foraging Through Hierarchical Visualization

**arXiv ID:** 2611.25377 | [PDF](https://arxiv.org/pdf/2611.25377v1)

**作者:** Catherine Yeh `[一作]` (Harvard University), Martin Wattenberg `[通讯]` (Harvard University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为VideoTrail的交互式视频搜索系统，通过层次化可视化帮助用户更有效地浏览视频内容。

**💡 创新点**

创新点在于结合了层次化可视化与交互式搜索，使用户能够更直观地探索视频信息。

**🔧 技术方法**

使用了层次化可视化技术和交互式用户界面设计。

**📊 数据集**

使用了多个视频数据集进行实验，以验证系统的有效性。

**📈 对比分析**

与传统的视频搜索方法进行了比较，结果显示VideoTrail在用户体验和信息检索效率上表现更佳。

**⚠️ 局限性**

限制在于系统可能对视频内容的复杂性和多样性敏感，可能无法处理所有类型的视频数据。

---

## 56. Don't Lose the Future: Latent-Aware 4-Bit Quantization of World Action Models

**arXiv ID:** 2611.25614 | [PDF](https://arxiv.org/pdf/2611.25614v1)

**作者:** Yichen Guo `[一作]` (Nanyang Technological University), Yawei Li `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种潜在感知量化框架，用于4位世界动作模型（WAM），以保护预测未来的准确性，并在机器人上实现高效的动作决策。

**💡 创新点**

创新点在于引入了读者度量和读取计数，能够量化未来错误对动作的影响，并通过联合精度分配来优化激活和缓存的精度。

**🔧 技术方法**

使用了潜在感知量化框架，结合读者度量和读取计数，以及注视引导的细化技术。

**📊 数据集**

在LIBERO-Plus基准上进行评估，使用Fast-WAM和Cosmos-Policy模型进行实验。

**📈 对比分析**

与现有的量化方法相比，该方法在LIBERO-Plus上实现了76.4%和83.3%的平均成功率，接近16位模型的性能，同时在多个基准上超越了其他4位量化方法。

**⚠️ 局限性**

限制在于该方法仍然依赖于特定的量化策略，可能在不同任务或环境下的适应性有限。

---

## 57. When Objectives and Evaluators Disagree: Evaluating Transfer for Post-hoc Demographic Framing Mitigation

**arXiv ID:** 2611.25258 | [PDF](https://arxiv.org/pdf/2611.25258v1)

**作者:** Tunazzina Islam `[一作]` `[通讯]` (University of Southern California), Tunazzina Islam (University of Southern California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在源模型不可访问的情况下，通过后期重写减少大型语言模型生成文本中的人口统计差异。使用660对性别和1980对年龄的气候信息进行研究。

**💡 创新点**

提出了一种新的适应性重写策略（Adaptive-single），该策略在编辑过程中考虑了双向信息，并在多个比较中表现出显著的性能提升。

**🔧 技术方法**

使用了后期重写、选择模型和独立评估模型的组合，形成了一个分离选择与评估的框架。

**📊 数据集**

使用了660对性别和1980对年龄的气候信息数据集，这些数据集是在固定的上下文条件下生成的。

**📈 对比分析**

与固定方向的Lower-PBI策略相比，Adaptive-single在多个比较中表现出更好的性能，尤其是在编辑覆盖率和结果一致性方面。

**⚠️ 局限性**

研究的局限性在于仅限于气候相关信息、二元性别对比和四个广泛的年龄类别，可能无法推广到其他领域或更复杂的人口统计设置。

---

## 58. Burnside Problems for Matrix Semigroup Quotients with Applications to Border Complexity

**arXiv ID:** 2611.26102 | [PDF](https://arxiv.org/pdf/2611.26102v1)

**作者:** Roland Guttenberg `[一作]` (University of Warsaw), James Worrell `[通讯]` (University of Oxford)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

研究了矩阵半群的商在同余下的有限性问题，提出了三种基于稳定子半群和群的商图像的有限性标准。

**💡 创新点**

提出了新的有限性标准，并将其应用于边界复杂性问题，特别是分离固定宽度和度数的齐次代数分支程序与其Zariski闭包。

**🔧 技术方法**

使用了外代数和Ramsey定理，发展了Simon的分解森林的变体来处理同余下的矩阵半群。

**📊 数据集**

使用了与代数分支程序相关的多项式和矩阵半群的构造，具体数据集未明确给出。

**📈 对比分析**

通过与现有方法的比较，展示了所提出的有限性标准在处理同余下的矩阵半群时的有效性，性能表现良好，能够解决边界复杂性中的分离问题。

**⚠️ 局限性**

限制在于所提出的标准可能不适用于所有类型的矩阵半群，且在某些情况下，Zariski闭包的性质可能导致复杂性问题的进一步挑战。

---

## 59. Kameleo: In-Situ Activation-Precision Selection for Bit-Serial LLM Accelerators

**arXiv ID:** 2611.25971 | [PDF](https://arxiv.org/pdf/2611.25971v1)

**作者:** Robin Geens `[一作]` (KU Leuven), Chao Fang `[通讯]` (KU Leuven)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Kameleo的加速器，能够在运行时根据处理的数据选择激活精度，从而降低大语言模型（LLM）的推理成本。

**💡 创新点**

Kameleo的创新点在于使用局部、无标签的代理来替代全局困惑度（PPL），并在加速器上实时选择每个GEMM的激活精度，避免了传统离线校准的计算密集型和数据依赖性。

**🔧 技术方法**

采用了位串行处理元素（PE）和块浮点（BFP）技术，利用位串行数据路径的特性几乎免费地进行精度比较。

**📊 数据集**

在四个LLM（Llama2-13B、Llama3.1-8B、Qwen2.5-7B和Yi1.5-34B）和七个数据集（如WikiText2、C4等）上进行了评估。

**📈 对比分析**

与传统的ANDA方法相比，Kameleo在保持模型质量的同时，减少了MXU周期，且在不同数据集上均能保持在1%的PPL预算内，表现出更好的适应性和鲁棒性。

**⚠️ 局限性**

Kameleo的局限性在于其选择硬件增加了加速器的面积和功耗，尽管增加幅度较小，但在某些情况下可能影响整体性能。

---

## 60. HiSA: Sparsifying the Hidden Dimension of RoPE Attention in LLMs

**arXiv ID:** 2611.26363 | [PDF](https://arxiv.org/pdf/2611.26363v1)

**作者:** Ambroise Borbely `[一作]` (EPFL), Olivier Verscheure `[通讯]` (Swiss Data Science Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的稀疏注意力机制HiSA，该机制在大语言模型中通过部分计算查询和键的点积来提高效率。

**💡 创新点**

创新点在于通过RoPE结构定义的静态、无参数的稀疏性，使得注意力在隐藏维度上稀疏，而不是在令牌上稀疏，从而提高了计算效率和模型性能。

**🔧 技术方法**

使用了RoPE（旋转位置嵌入）和FlashAttention-3内核技术。

**📊 数据集**

使用了SmolLM-135M和SmolLM-360M架构，并在OpenWebText数据集上进行训练。

**📈 对比分析**

与密集注意力相比，HiSA在验证损失、训练速度、KV缓存内存和解码吞吐量等方面进行了比较，结果显示HiSA在计算29-34%的点积时，验证损失更低。

**⚠️ 局限性**

限制在于当前方法的稀疏性模式是静态的，可能无法适应某些动态变化的输入特征。

---

## 61. Now You See Me Now You Don't: A Framework To Detect and Reduce Proxy Capacity

**arXiv ID:** 2611.25327 | [PDF](https://arxiv.org/pdf/2611.25327v1)

**作者:** Nirek Sharma `[一作]` `[通讯]` (Upstart Network, Inc.), Nirek Sharma (Upstart Network, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种框架，用于检测和减少特征的代理能力，以应对在信贷等受监管领域中可能出现的直接歧视问题。

**💡 创新点**

创新点在于提出了一种双重框架，既能识别特征的代理能力，又能通过调整特征来降低其代理能力，同时保持预测性能。

**🔧 技术方法**

使用了条件独立性测试和预处理方法来识别和调整代理特征。

**📊 数据集**

使用了合成数据集和公共数据集进行实证验证。

**📈 对比分析**

与现有方法相比，本文的方法在识别代理特征方面表现更好，并且在调整后能够保持预测性能，优于简单的特征移除。

**⚠️ 局限性**

限制在于该方法主要关注简单代理能力，而复杂代理能力的检测和处理仍需进一步研究。

---

## 62. Benchmarking Open-Source O-RAN and 5G NTN Stacks on Embedded ARM Platforms for Space Payloads

**arXiv ID:** 2611.25206 | [PDF](https://arxiv.org/pdf/2611.25206v1)

**作者:** Esteban Municio `[一作]` (i2CAT Foundation), Xavier Costa-Pérez `[通讯]` (NEC Laboratories Europe)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对两种开源gNB堆栈（OpenAirInterface/Duranta和srsRAN/OCUDU）以及两种近实时RIC（ORAN-SC Near-RT RIC和FlexRIC）在嵌入式ARM平台上的性能进行了基准测试，评估了它们在空间有效载荷中的适用性。

**💡 创新点**

创新点在于首次在相同的嵌入式硬件上比较不同的gNB堆栈和RIC平台，提供了关于它们在空间环境下的性能和资源效率的定量证据。

**🔧 技术方法**

使用了基于ARMv8的AMD Zynq UltraScale+ ZCU104评估板，运行在实时Linux内核上的容器中，驱动USRP B210软件定义无线电。

**📊 数据集**

使用的主要数据集包括OpenAirInterface和srsRAN的开源实现，以及ORAN-SC和FlexRIC的近实时RIC实现。

**📈 对比分析**

通过功能合规性（3GPP NTN特性、O-RAN接口、多租户）、资源效率、实时行为和工程适用性进行比较。结果显示，srsRAN具有较小的内存占用，而OAI则表现出较低的CPU利用率，但两者在某些功能上存在显著差异。

**⚠️ 局限性**

限制在于当前的测试是在没有NTN时序或信道仿真的情况下进行的，且结果仅基于单一平台，未来需要在真实的NTN条件下进行验证。

---

## 63. Quantifying Cultural-Psychological Constructs in Four Centuries of Ottoman Court Registers

**arXiv ID:** 2611.24981 | [PDF](https://arxiv.org/pdf/2611.24981v1)

**作者:** Sila Mutaf `[一作]` (University of Michigan), Mohammad Atari `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 4616 | [OpenAlex ID](https://openalex.org/A5015311126)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究开发并评估了一个计算框架，用于测量奥斯曼法院记录中的文化心理构念，重点研究个人主义、集体主义和荣誉三个文化维度。

**💡 创新点**

创新点在于将现代心理学构念与历史文本结合，通过计算方法分析奥斯曼帝国的法律文本，填补了历史心理学研究的空白。

**🔧 技术方法**

使用了上下文嵌入模型和句子变换器，结合心理测量理论进行模型训练和评估。

**📊 数据集**

使用的数据集包括来自100个伊斯坦布尔法院的54,670份拉丁字母转录的法院记录，涵盖1513年至1911年间的法律文书。

**📈 对比分析**

通过与专家标注的文档进行比较，评估了不同模型的表现，结果显示，OTA-BERTurk模型在问卷项目分类任务中表现最佳，准确率达到89.58%。

**⚠️ 局限性**

限制在于法院记录是由法官和书记员记录的，可能不完全代表奥斯曼社会的心理特征，且专家标注样本相对较小，构念之间的协议程度差异较大。

---

## 64. Learning to Read Events for Exact State Tracking with Frozen Language Models

**arXiv ID:** 2611.25211 | [PDF](https://arxiv.org/pdf/2611.25211v1)

**作者:** Jeonghoon Lee `[一作]` `[通讯]` (Attractor Dynamics Inc), Jeonghoon Lee (Attractor Dynamics Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文研究了长事件流的状态跟踪问题，提出了一种基于冻结语言模型的选择器来读取每个过渡，并通过外部整数寄存器按顺序组合这些过渡。

**💡 创新点**

创新点在于引入了一个精确状态层（ESL），通过选择器和外部寄存器的组合来提高状态跟踪的准确性，并且在处理复杂的状态变化时引入了置信度规则。

**🔧 技术方法**

使用了冻结的语言模型和外部整数寄存器，结合选择器来处理事件流的过渡。

**📊 数据集**

使用了包含43,687个事件的窗口数据集，以及多个不同长度的房间交换文档和状态跟踪基准（StateBench）。

**📈 对比分析**

与传统方法相比，使用选择器和寄存器的组合在处理长事件流时表现出更高的准确性和更低的计算成本，尤其是在提取和组合过程中，时间和成本显著降低。

**⚠️ 局限性**

限制在于模型在处理某些类型的事件时可能不够准确，尤其是在没有可读的事件时，且在自由对话状态下的表现不佳。

---

## 65. Moving Generality from Model to Modeler: LLM Agents for PHM on a Deployment-Style Benchmark

**arXiv ID:** 2611.25252 | [PDF](https://arxiv.org/pdf/2611.25252v1)

**作者:** Zhenling Chen `[一作]` (CentraleSupélec), Anne Barros `[通讯]` (CentraleSupélec)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文探讨了大型语言模型（LLM）代理在预测和健康管理（PHM）中的应用，提出了一种新的基准测试NextUnit-Bench，评估代理在未见设备上的模型构建能力。

**💡 创新点**

创新点在于将模型的通用性从模型本身转移到模型构建者（代理），强调代理在特定任务上的设计能力和数据学习的重要性。

**🔧 技术方法**

使用了大型语言模型（LLM）作为编码代理，进行模型构建和预测。

**📊 数据集**

使用了24个公共PHM数据集，分为58个折叠，确保每个模型在未见设备上进行测试。

**📈 对比分析**

与AutoML和已发布方法的比较显示，前沿代理在24个数据集上的平均表现优于AutoML，且在四个数据集上强代理的表现超过了忠实实现的已发布方法。前沿代理的优势在于其针对特定设备的数据设计，而不是模型的规模。

**⚠️ 局限性**

限制在于大多数配置仅运行一次，结果的稳定性依赖于完整的重跑，且数据集的公共性可能导致代理在预训练中见过这些数据，影响结果的普遍性。

---

## 66. Uniqueness, Proximity, and Scaling in Atomic Splittable Polymatroid Congestion Games

**arXiv ID:** 2611.25757 | [PDF](https://arxiv.org/pdf/2611.25757v1)

**作者:** Tobias Harks `[一作]` `[通讯]` (University of Passau), Tobias Harks (University of Passau)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种多项式时间算法，用于在具有整数秩和玩家特定的正斜率仿射延迟的原子可分多项式拥塞游戏中计算精确均衡。

**💡 创新点**

该算法首次提供了针对任意多项式基的离散均衡的多项式时间算法，并通过逐步细化的数据包恢复连续均衡的精确性。

**🔧 技术方法**

使用了Oki和Schwarcz的多重交换定理，结合子模秩的兼容舍入和精确有理算术。

**📊 数据集**

使用了具有玩家特定仿射成本的多项式秩数据集，且假设这些成本具有正斜率。

**📈 对比分析**

与现有方法相比，该算法在处理原子可分多项式拥塞游戏时，能够在多项式时间内计算出精确均衡，并且在不同数据包大小下的均衡接近性得到了保证。

**⚠️ 局限性**

算法的局限性在于它依赖于玩家特定的成本函数和多项式秩的结构，可能在更一般的情况下不适用。

---

## 67. A Synthetic Multi-Modal Testbed for Security Stress Testing of Agentic Counter-UAS Systems

**arXiv ID:** 2611.25249 | [PDF](https://arxiv.org/pdf/2611.25249v1)

**作者:** Ivan A. Fernandez `[一作]` (Mississippi State University), Jingdao Chen `[通讯]` (Mississippi State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个合成的反无人机系统（C-UAS）测试平台，用于对感知和跟踪算法进行安全压力测试，以支持自主决策。

**💡 创新点**

创新点在于创建了一个轻量级的合成测试平台，能够在可控条件下生成多模态传感数据，并进行重复性测试。

**🔧 技术方法**

使用了NVIDIA Isaac Sim进行合成数据生成，采用了非线性协调转动扩展卡尔曼滤波器（EKF）进行跟踪评估。

**📊 数据集**

使用了合成的RGB、RTX LiDAR和类红外（IR-like）观测数据。

**📈 对比分析**

通过对比LiDAR目标中心测量的跟踪误差和不确定性，评估了在传感器观测降级和丢失情况下的性能。

**⚠️ 局限性**

限制在于测试平台并未重现高保真气动特性或完全校准的传感器物理特性，未来工作将引入更多的传感器故障和环境变化场景。

---

## 68. Direct3D-V3: Latent-Free Diffusion Transformers in the Native Voxel Domain

**arXiv ID:** 2611.26511 | [PDF](https://arxiv.org/pdf/2611.26511v1)

**作者:** Shuang Wu `[一作]` (Nanjing University), Xun Cao `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为Direct3D-V3的无潜变量扩散变换器框架，直接在原始体素域中建模SDF体素网格，避免了传统潜变量自编码器（VAE）的使用。

**💡 创新点**

创新点在于去除了VAE重建瓶颈，直接在体素域中进行训练和采样，同时引入隐式流形正则化以提高几何可靠性，并通过预测相机参数来改善生成几何与条件图像之间的对齐。

**🔧 技术方法**

使用了无潜变量的扩散变换器（DiT）技术，结合隐式流形正则化和Marching Cubes算法。

**📊 数据集**

未具体提及使用的数据集，但进行了广泛的实验以验证方法的有效性。

**📈 对比分析**

与最先进的基于潜变量的方法相比，Direct3D-V3在生成几何细节上表现出色，能够生成更丰富的局部细节，且在性能上具有竞争力。

**⚠️ 局限性**

限制在于未详细说明使用的数据集，可能影响结果的普适性和可重复性。

---

## 69. Probing the "Conscious" Mind: Selective Access and Global Workspace in Vision-Language Models

**arXiv ID:** 2611.26590 | [PDF](https://arxiv.org/pdf/2611.26590v1)

**作者:** Jiahao Huang `[一作]` (Imperial College London), Jun Wang `[通讯]` (University College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了视觉语言模型（VLMs）中的多模态概念处理，提出了多模态概念透镜（MC-Lens），用于比较视觉和语言处理中的概念可访问性，并支持内部概念干预以检验概念在下游计算中的作用。

**💡 创新点**

创新点在于引入了MC-Lens，这是一种结合视觉编码器和语言解码器的多模态雅可比透镜，能够在共享语义空间中映射中间状态，从而比较不同模态下的概念可访问性。

**🔧 技术方法**

使用了多模态雅可比透镜（MC-Lens），结合了视觉编码器透镜（V-Lens）和语言解码器透镜（L-Lens），并开发了基于J空间的度量来跟踪跨模态概念对齐。

**📊 数据集**

使用了A-OKVQA数据集进行视觉问答（VQA）任务，收集了100个案例进行实验，分析了不同证据条件下的概念可访问性。

**📈 对比分析**

通过与现有方法的比较，MC-Lens在概念可访问性和选择性访问方面表现出更好的性能，尤其是在下游计算中，概念信息不仅可读，而且参与推理过程。

**⚠️ 局限性**

限制在于当前的研究主要集中在已训练的VLMs上，尚未深入探讨如何在训练过程中逐步发展跨模态对齐和选择性访问机制。

---

## 70. Active Feature Acquisition by Distilling an Oracle Teacher

**arXiv ID:** 2611.26598 | [PDF](https://arxiv.org/pdf/2611.26598v1)

**作者:** Yang Meng `[一作]` (University of Chicago), Yuxin Chen `[通讯]` (University of Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种主动特征获取（AFA）的方法，通过构建一个显式的教师模型来指导特征获取策略的学习，以在预算限制下选择最有价值的特征。

**💡 创新点**

创新点在于将教师模型显式化，而不是依赖自构建的效用目标，从而提高了特征获取策略的性能和可分析性。

**🔧 技术方法**

使用了软目标模仿学习技术，通过从一个短视的Brier oracle中提取知识来训练可部署的学生策略。

**📊 数据集**

使用了六个数据集进行实验，包括多模态医学诊断数据集TCGA-Glioma和AFABench基准。

**📈 对比分析**

与现有的AFA方法相比，本文的方法在预算范围内的表现相当或更优，能够在不到四分之一的获取成本下恢复90%的可获得准确性增益。

**⚠️ 局限性**

限制在于教师模型的表现无法完全被学生模型恢复，尤其是在信息缺失的情况下，学生模型的决策能力受到限制。

---

## 71. PhysVAM: Physics-oriented Future Value Action Model for Dynamic Object Manipulation

**arXiv ID:** 2611.26105 | [PDF](https://arxiv.org/pdf/2611.26105v1)

**作者:** Yuxiang Yang `[一作]` (Hong Kong Polytechnic University), Lap-Pui Chau `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了PhysVAM，一个物理导向的未来价值行动模型，用于动态物体操控，通过多视角的未来价值预测来指导行动生成。

**💡 创新点**

创新点在于从任务价值的角度进行未来建模，直接反映物理交互对任务进展的影响，而不是单纯重建未来视觉状态。

**🔧 技术方法**

使用了Diffusion Transformer架构，分别用于行动生成和未来价值学习。

**📊 数据集**

使用了包含12K成功轨迹和10K失败轨迹的动态操控数据集，涵盖多种几何形状的物体。

**📈 对比分析**

与FastWAM、DynamicWAM等方法比较，PhysVAM在模拟和真实世界任务中均表现出最高的成功率，成功率提高了27个百分点。

**⚠️ 局限性**

局限性在于该框架依赖于离线轨迹监督，未来工作可以探索在线强化学习中的自我改进和更复杂的动态操控场景。

---

## 72. Decision Boundaries Matter: Graph-Regularized Residual Adaptation for Cross-Domain Fake Image Detection

**arXiv ID:** 2611.25406 | [PDF](https://arxiv.org/pdf/2611.25406v1)

**作者:** Shichuang Xie `[一作]` (Shanghai University), Guorui Feng `[通讯]` (Shanghai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Graph-Regularized Residual Adaptation (GraRA)的方法，用于在没有源数据或目标标签的情况下，仅使用未标记的目标数据来调整源决策边界，以提高假图像检测的性能。

**💡 创新点**

创新点在于通过构建一个紧凑的残差子空间，并使用图扩散生成软标签，从而在不依赖源标签的情况下优化分类器的决策边界。

**🔧 技术方法**

使用了图正则化和残差学习技术，结合不确定性加权的目标特征来指导分类器的更新。

**📊 数据集**

使用了OpenMMSec数据集，该数据集包含多种未见的伪造类型，涵盖了六种AIGC、四种深度伪造、两种图像处理和两种文档伪造领域。

**📈 对比分析**

与现有的假图像检测器和源无关的适应方法进行比较，GraRA在准确率（ACC）上达到84.36%，在AUC上达到91.34%，分别比源检测器提高了4.33和3.50个百分点，显示出显著的性能提升。

**⚠️ 局限性**

限制在于该方法依赖于源分类器的预测质量，如果源预测不可靠，可能会影响适应效果。此外，未标记的目标数据的特征结构可能会影响最终的检测性能。

---

## 73. ChronoPatch: Black-Box Energy-Latency Attacks on Vision-Language Models

**arXiv ID:** 2611.26117 | [PDF](https://arxiv.org/pdf/2611.26117v1)

**作者:** Jiawei Xing `[一作]` (Shandong University), Zheng Li `[通讯]` (Shandong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为ChronoPatch的黑箱攻击方法，旨在通过仅观察最终响应来诱导视觉语言模型（VLM）的生成延迟，从而增加推理延迟和能耗。

**💡 创新点**

创新点在于ChronoPatch能够在没有模型内部信息或中间反馈的情况下，仅通过最终响应长度来优化攻击，展示了在黑箱环境下的能量-延迟攻击的有效性。

**🔧 技术方法**

使用了基于响应长度的进化搜索和子空间校准的决策优化技术，分为两个阶段：延迟锚点发现和扰动精炼。

**📊 数据集**

在四个开源VLM（BLIP-Large、LLaVA-v1.5-7B、Qwen3-VL-8B、InternVL3-8B）和三个商业API（ChatGPT-5、Gemini-3.1-Pro-Preview、Claude-Sonnet-4-6）上进行了评估，使用了MSCOCO、ImageNet和Flickr30K等数据集。

**📈 对比分析**

与现有的基于梯度的白箱攻击（如Verbose Images）和黑箱攻击（如VLMInferSlow）相比，ChronoPatch在所有评估模型上均表现出更高的延迟和能耗增加，且在商业API上也显示出显著的响应长度和延迟增加。

**⚠️ 局限性**

限制在于攻击评估使用了受控的解码配置，且在Phase I和Phase II的优化过程中，可能会产生较高的每图像优化成本。此外，尽管扰动是局部化的，但其局部外观可能仍然可察觉，未来的工作应致力于改善其语义融合。

---

## 74. Pro$^2$ Agent: Scientific Problem Probing for Autonomous AI R&D

**arXiv ID:** 2611.25357 | [PDF](https://arxiv.org/pdf/2611.25357v1)

**作者:** Yixiong Chen `[一作]`, Alan Yuille `[通讯]` (Johns Hopkins University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的自主AI研究代理系统，称为Problem Probing，旨在通过明确的问题状态建模和假设驱动的实验来推进AI研究。

**💡 创新点**

创新点在于将代理对问题的理解作为自主AI研究的首要状态，定义了问题状态为对性能限制问题的明确表示，并通过文献和假设测试来构建和更新该状态。

**🔧 技术方法**

使用了多代理设计，结合Git版本控制系统，实施了三阶段框架，包括问题状态初始化、科学探索和工程优化。

**📊 数据集**

在MLE-Bench和PostTrainBench两个数据集上进行了评估，MLE-Bench包含75个Kaggle竞赛任务，PostTrainBench包含28个配置。

**📈 对比分析**

与现有方法（如MARS）相比，Problem Probing在MLE-Bench上实现了76.0%的奖牌率（48小时预算），而MARS通常需要评估超过70个变体才能达到相似的分数，显示出更强的初始解决方案和更快的收敛速度。

**⚠️ 局限性**

限制在于当前的信念更新相对简单，未来可以探索更复杂的概率或噪声感知的信念更新方法，以及如何在更长的研究时间内扩展Problem Probing。

---

## 75. Duopra: From Full-Duplex Conversation to Full-Duplex Agency

**arXiv ID:** 2611.25773 | [PDF](https://arxiv.org/pdf/2611.25773v1)

**作者:** Chiyu Chen `[一作]` (Shanghai Jiao Tong University), Zhuosheng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种全双工交互模型Duopra，该模型能够同时学习任务执行和对话控制。

**💡 创新点**

创新点在于将后端视为工具，接受自包含的自然语言请求，并通过统一的动作接口实现直接调用和后端委托。

**🔧 技术方法**

使用了协议监督微调（SFT）、全双工强化学习（RL）和仅投影修复的三阶段训练过程。

**📊 数据集**

使用了Full-Duplex-Bench v3（FDB v3）数据集，该数据集包含100个不流畅的英语录音和12个模拟API。

**📈 对比分析**

与基线模型相比，Duopra在直接调用和后端委托的任务完成率上都有显著提高，Pass@1从4.7%提升至44.7%。

**⚠️ 局限性**

限制在于与已发布结果的可比性、模型和数据覆盖范围，以及未来工作中对多轮工具使用任务的评估需求。

---

## 76. ADAPT: Adaptive Decision and Abstention with Pool-Size-Aware Thresholds for Service Routing

**arXiv ID:** 2611.25754 | [PDF](https://arxiv.org/pdf/2611.25754v1)

**作者:** Manile Srun `[一作]` (Open Source Institute), Carlos C. N. Kuhn `[通讯]` (Open Source Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为ADAPT的轻量级检索基础路由架构，该架构扩展了传统分类器，并引入了明确的后备选项，以便在没有合适专家时将查询发送到通用服务。

**💡 创新点**

创新点在于引入了一个基于池大小的阈值，该阈值随着可用专家数量的增加而调整，从而提高了对非主题查询的拒绝率，同时保持了专家查询的准确性。

**🔧 技术方法**

使用了检索模型（如双编码器、ColBERT模型和交叉编码器）与LLM路由器进行比较，采用了基于得分的决策规则来选择合适的专家。

**📊 数据集**

使用了三个数据集：MMLU（提供专家）、BANKING77和CLINC-OOS（提供非主题查询）。

**📈 对比分析**

与LLM路由器相比，ADAPT在多个专家池（最多14个专家）中表现出竞争力，且在计算成本上显著低于LLM路由器，尤其是在专家数量增加时，ADAPT的性能保持稳定。

**⚠️ 局限性**

主要限制在于只评估了最多14个专家的池，随着专家数量的增加，非主题查询的得分与主题查询的得分可能会重叠，因此单一阈值可能无法有效区分两者。

---

## 77. Class-Conditional Conformal Triage for ICAO Face-Image Compliance under Class Imbalance

**arXiv ID:** 2611.25342 | [PDF](https://arxiv.org/pdf/2611.25342v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 78. Tenon: A Unified Programming Framework for Retargetable PIM Compilation

**arXiv ID:** 2611.25636 | [PDF](https://arxiv.org/pdf/2611.25636v1)

**作者:** Niansong Zhang `[一作]`, Zhiru Zhang `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

论文内容未提供，无法总结做了什么。

**💡 创新点**

论文内容未提供，无法总结创新点。

**🔧 技术方法**

论文内容未提供，无法总结使用的技术。

**📊 数据集**

论文内容未提供，无法总结使用的数据集。

**📈 对比分析**

论文内容未提供，无法总结比较的方法及性能。

**⚠️ 局限性**

论文内容未提供，无法总结限制因素。

---

## 79. ViDecide: Teaching Video Diffusion Models to Reason and Plan Step by Step

**arXiv ID:** 2611.26389 | [PDF](https://arxiv.org/pdf/2611.26389v1)

**作者:** Kaiwen Zhang `[一作]` (S-Lab, Nanyang Technological University), Xingang Pan `[通讯]` (S-Lab, Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种决策中心框架，训练视频扩散模型以因果方式推理和规划，生成解决方案并决定何时停止。

**💡 创新点**

创新点在于将决策视为视频推理的基本单元，采用因果生成和自适应终止机制，集中训练于高噪声水平，并通过纠正微调（CFT）修正自身的失败。

**🔧 技术方法**

使用了因果生成、决策集中训练调度和纠正微调（CFT）等技术。

**📊 数据集**

在多个基准测试上进行了评估，包括抽象游戏、规划任务和机器人操作，具体数据集包括VideoRLVR和RoboTwin-Plan。

**📈 对比分析**

与标准微调相比，模型的严格成功率从23.3%提高到83.8%，在所有基准测试中均优于之前的最强方法67.8%。

**⚠️ 局限性**

目前模型仅在单一基准上训练，并依赖规则验证器进行后训练，未来计划扩展到更广泛的推理数据和学习验证器。

---

## 80. TwoPass: Draft-Guided Few-Step Video Prediction for Robotic Manipulation

**arXiv ID:** 2611.25553 | [PDF](https://arxiv.org/pdf/2611.25553v1)

**作者:** Yijun Liu `[一作]` (Tsinghua University), Wenwu Zhu `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种自条件训练和推理框架，用于动作条件的视频预测，旨在提高机器人对其动作视觉后果的预测能力。

**💡 创新点**

创新点在于结合了自生成草稿训练和仅限于细化的适配器，使得模型能够在标准流匹配目标下进行自我修正，从而减少网络调用次数。

**🔧 技术方法**

使用了自条件训练、草稿重用迭代细化等技术，保持了标准流匹配目标。

**📊 数据集**

在Bridge和Language Table数据集上进行了广泛实验，使用了官方测试集中的数据。

**📈 对比分析**

与现有方法相比，该方法在4次函数评估下达到了最先进的结果，显著减少了网络调用次数（仅需4次，而现有预测器需要50-59次）。

**⚠️ 局限性**

限制在于模型的训练和推理过程依赖于预训练的骨干网络，可能在特定情况下表现不佳。

---

## 81. PlayCraft: Designing Engaging Games from Agent Play Experience

**arXiv ID:** 2611.26624 | [PDF](https://arxiv.org/pdf/2611.26624v1)

**作者:** Weihao Cheng `[一作]` `[通讯]`, Weihao Cheng

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个自动化框架PlayCraft，用于通过支持掌握来提高游戏的吸引力。该框架通过一个视觉基础模型代理进行游戏，利用其详细的游戏体验来修订游戏。

**💡 创新点**

PlayCraft的创新点在于它结合了游戏的玩耍和制作阶段，通过玩家代理的反馈来自动调整游戏设计，以促进玩家的掌握和学习。

**🔧 技术方法**

使用了视觉基础模型（如GPT-6 Astra）作为玩家代理，能够观察游戏画面并通过相同的控制方式进行互动，同时保留记忆和学习经验。

**📊 数据集**

使用了10个基于浏览器的游戏数据集，包括5个经典的2D游戏（如Flappy Bird、T-Rex Runner等）和5个通过提示生成的3D游戏（如Archery Hunting、Desert Racing等）。

**📈 对比分析**

与基于代码分析和策略控制器的基线方法相比，PlayCraft在所有智能设置下的宏观平均学习增益和学习斜率均表现更好，尤其在中高智能设置下，学习增益分别提高了14.0和21.9个百分点。

**⚠️ 局限性**

PlayCraft的局限性在于假设代理的游戏和学习模式与人类玩家相似，这一假设尚未得到验证。此外，长时间游戏的成本较高，可能需要更高效的观察和行动规划来实现更长的游戏运行。

---

## 82. A Matter of Representation: Evolving Partial Spreads for Bent Functions

**arXiv ID:** 2611.26011 | [PDF](https://arxiv.org/pdf/2611.26011v1)

**作者:** Claude Carlet `[一作]` (University of Bergen), Stjepan Picek `[通讯]` (Radboud University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本论文设计了一种进化策略（ES）来构造部分扩展（𝒫𝒮）弯曲函数，这些函数是具有最大非线性的布尔函数，基于仅在零向量处相交的子空间族。

**💡 创新点**

创新点在于比较了基于生成矩阵和扩展集的两种编码方式，并引入了破坏与重建操作，成功找到最多10个变量的𝒫𝒮函数。

**🔧 技术方法**

使用了进化策略（ES）和破坏与重建操作等技术。

**📊 数据集**

使用了8和10个变量的布尔函数数据集进行实验。

**📈 对比分析**

与其他方法比较时，使用了重启的贪婪启发式算法，发现其在最困难的实例上表现优于ES，尤其是在𝒫𝒮^+目标上。

**⚠️ 局限性**

限制在于对于n=12的情况，ES和贪婪启发式算法均未能找到解决方案，表明该实例的复杂性超出了当前方法的能力。

---

## 83. AdaPixFlow: Learning Continuous-Time 3D Dynamics Across Scenes from Sparse Multi-view Observations

**arXiv ID:** 2611.26433 | [PDF](https://arxiv.org/pdf/2611.26433v1)

**作者:** Hrishikesh Viswanath `[一作]` (Purdue University), Aniket Bera `[通讯]` (Purdue University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本论文研究了如何从视觉观察中恢复动态场景中物体的三维运动，并预测未观察时刻的物体位置。提出了一种名为Adaptive Pixel Flows (AdaPixFlow) 的方法，该方法通过多视角RGB-D观察学习一个连续时间的动态算子，而无需针对每个场景进行优化或预定义运动模型。

**💡 创新点**

创新点在于提出了一种共享的连续时间动态算子，该算子能够跨场景进行训练，并从稀疏的多视角RGB-D观察中预测未观察场景中的三维物体运动。

**🔧 技术方法**

使用了多视角RGB-D观察和体素网格融合技术，通过一个3D U-Net网络实现动态算子的学习。

**📊 数据集**

使用了PhysInOne数据集，该数据集包含71种动态场景，提供RGB图像、深度图和分割信息，且每个场景由12-15个静态校准相机捕获。

**📈 对比分析**

与每场景优化的动态模型（如FreeGave和TRACE）相比，AdaPixFlow在未观察时刻的预测准确性显著提高。在PhysInOne数据集上，AdaPixFlow在中间预测上比最强竞争方法提高了2.5%的F1分数，在未来预测上提高了15%。与每场景速度场方法相比，分别提高了30%和85%。

**⚠️ 局限性**

限制在于当前的算子模型仅建模运动学，未来的工作可以考虑将其扩展到接触和力模型，以处理碰撞和反弹等交互情况。此外，处理估计深度和分割中的错误也是应用于真实多视角捕获的一个挑战。

---

## 84. Improving Imitation Policies for Multi-Phase Tasks through Residual Learning Allocation

**arXiv ID:** 2611.25320 | [PDF](https://arxiv.org/pdf/2611.25320v1)

**作者:** Clinton Enwerem `[一作]` (University of Maryland), Calin Belta `[通讯]` (University of Maryland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

论文提出了一种残差强化学习方法，通过有限的环境交互预算来修正模仿训练的基础策略，特别是在多阶段任务中分配预算。

**💡 创新点**

创新点在于定义了完成增益（completion gain），并提出了一种基于增益的分配策略，用于选择残差策略在各个阶段的修正组件和训练分配。

**🔧 技术方法**

使用了残差强化学习（Residual Reinforcement Learning）和状态反馈控制器（state-feedback controller）来进行策略修正和训练。

**📊 数据集**

在随机点质量对接实验和双手机器人组装模拟中进行了验证，使用了这些任务的模拟数据集。

**📈 对比分析**

与其他四种分配策略（如基于失败计数的分配、均匀分配等）进行比较，基于增益的分配在相同预算下实现了更高的任务完成率。

**⚠️ 局限性**

限制在于仅在模拟环境中进行验证，实际应用中可能面临更复杂的动态和不确定性。

---

## 85. Watertight Mesh Optimization of Textured Objects

**arXiv ID:** 2611.25414 | [PDF](https://arxiv.org/pdf/2611.25414v1)

**作者:** Nathaniel Burgdorfer `[一作]` (Stevens Institute of Technology), Philippos Mordohai `[通讯]` (Stevens Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种框架，通过直接优化一个密闭的三角网格来重建高质量的3D资产，能够同时优化几何形状和视角依赖的外观。

**💡 创新点**

创新点在于首次实现了仅通过图像重建真实世界的纹理对象，而无需辅助的分割掩码、深度图或法线图，且优化过程直接在三角网格上进行。

**🔧 技术方法**

使用了拉普拉斯预处理优化和曲率自适应重网格化技术，以保持细节和良好的三角形形状，同时结合多视图光度一致性约束。

**📊 数据集**

使用了DTU和AriaDTC数据集，这些数据集包含真实物体的多视图图像和参考几何信息。

**📈 对比分析**

与其他方法相比，提出的方法在几何准确性、视角合成的保真度和内在网格质量方面表现出色，能够生成几何准确且具有优越内在网格质量的3D模型。

**⚠️ 局限性**

限制在于该方法无法改变网格的拓扑结构，拓扑变化是一个复杂且有趣的未来研究方向。

---

## 86. Correctable Enrollment and One-Transaction Seed Proofs for Ethereum's Post-Quantum Transition

**arXiv ID:** 2611.25663 | [PDF](https://arxiv.org/pdf/2611.25663v1)

**作者:** Xin Jin `[一作]`, Junyu Liu `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一种迁移协议，使以太坊账户能够在量子计算机威胁下安全迁移到后量子密钥，允许账户所有者在ECDSA退役之前以低成本注册后量子密钥，并通过零知识证明纠正任何注册。

**💡 创新点**

创新点在于提出了一种可纠正的迁移机制，允许账户所有者在量子计算机出现之前进行密钥迁移，并且在迁移过程中提供了零知识证明以确保安全性。

**🔧 技术方法**

使用了基于哈希的零知识证明（STARK）技术，结合了递归证明和轻量级ECDSA签名，确保了证明的大小和验证速度。

**📊 数据集**

使用了以太坊主网的账户数据进行模拟，特别是持有至少0.1 ETH的外部拥有账户（EOA），总计约445万个账户。

**📈 对比分析**

与现有方法相比，该协议在迁移速度和安全性上表现优越。通过模拟，发现如果所有账户在ECDSA退役之前注册，迁移将需要29.4天；如果在退役后分批恢复，则需要235天。性能测试显示，零知识证明的验证时间为38毫秒，生成时间为111秒。

**⚠️ 局限性**

限制在于需要账户所有者在量子计算机出现之前完成注册，否则账户将被冻结，且未注册的账户无法在量子攻击下恢复。

---

## 87. Compiling Constraints into Neuromorphic Ising Machines

**arXiv ID:** 2611.25352 | [PDF](https://arxiv.org/pdf/2611.25352v1)

**作者:** Faiek Ahsan `[一作]` (Washington University in St. Louis), Shantanu Chakrabartty `[通讯]` (Washington University in St. Louis)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的神经形态伊辛机器架构，通过可编程的状态转移来处理约束，而不是通过能量惩罚，从而提高组合优化问题的求解效率。

**💡 创新点**

创新点在于将约束编译为可编程的状态转移，允许机器在可行状态空间内进行搜索，避免了传统方法中因惩罚引入的高能量障碍。

**🔧 技术方法**

使用了神经形态计算架构，结合了自编码器的设计，利用多重自旋转移来优化状态更新。

**📊 数据集**

在多个数据集上进行了测试，包括图匹配问题和表面码解码，特别是在WILLOW-ObjectClass数据集上进行了图匹配实验。

**📈 对比分析**

与基于惩罚的神经形态机器（如N-HOIM）相比，该架构在图匹配问题上需要的迭代次数显著减少，表现出更快的收敛速度和更低的逻辑错误率。

**⚠️ 局限性**

限制在于当前架构的硬件实现可能无法扩展到更大规模的问题，且在处理复杂约束时可能需要更多的辅助子句。

---

## 88. Awaken-HI: Scaling Humanoid Intelligence with Whole-Body Human Experience

**arXiv ID:** 2611.26420 | [PDF](https://arxiv.org/pdf/2611.26420v1)

**作者:** EngineAI `[一作]` `[通讯]`, EngineAI

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个包含约50,000小时的自我中心立体视频和同步的全身人类运动的数据集，并提出了一种用于类人机器人运动操控的视觉-语言-动作模型。

**💡 创新点**

创新点在于通过共享的动作专家同时预测原始人类运动和重定向的机器人动作，从而提高了类人机器人在日常任务中的表现。

**🔧 技术方法**

使用了视觉-语言-动作（VLA）模型和深度学习技术，结合了人类运动的预训练和机器人演示的微调。

**📊 数据集**

使用了一个包含50,000小时自我中心立体视频和同步全身人类运动的数据集，覆盖了100多种技能。

**📈 对比分析**

与现有的基线模型（如GR00T N1.7和π_0.5）进行比较，模型在八个独立任务中的平均成功率为93.75%，显著优于基线模型。

**⚠️ 局限性**

限制在于尽管模型在多个任务上表现良好，但仍可能受到人类运动与机器人动作之间的形态差异影响，导致某些情况下的动作预测误差。

---

## 89. Physics-Informed Deep Learning for Channel Prediction in Satellite-Mounted Metasurface Reflection Communication Networks

**arXiv ID:** 2611.26068 | [PDF](https://arxiv.org/pdf/2611.26068v1)

**作者:** Aolin Sun `[一作]` (Beijing University of Posts and Telecommunications), Zhuangzhuang Cui `[通讯]` (KU Leuven)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种物理信息深度学习（PIDL）框架，用于在卫星安装的超表面反射通信网络中进行信道预测，考虑了动态轨道拓扑、随机用户几何和硬件相位量化等实际约束。

**💡 创新点**

创新点在于将无线通信的物理约束（如单调性和渐近衰减特性）嵌入深度神经网络中，以实现对信道的预测，并确保预测结果的物理一致性。

**🔧 技术方法**

使用了物理信息深度学习（PIDL）技术，结合了数据驱动的神经网络和领域特定的物理约束。

**📊 数据集**

构建了一个综合数据集，涵盖了多维特征空间，包括发射功率、几何形状、星座规模和硬件精度，共包含9600个样本。

**📈 对比分析**

通过数值仿真验证了PIDL框架，在不同几何配置、星座密度和相位量化分辨率下，预测准确率超过0.99，并且在极端失效区域（10^-20）也保持高保真度。

**⚠️ 局限性**

限制在于尽管模型在多种情况下表现良好，但与理想化理论模型相比，实际随机分布下的性能仍然存在差距，特别是在卫星位置优化方面。

---

## 90. The Geometry of Lies: Knowledge and Persona Directions for Detecting Deception in Large Language Models

**arXiv ID:** 2611.26545 | [PDF](https://arxiv.org/pdf/2611.26545v1)

**作者:** Md. Abdullah Al Zami `[一作]`, Md. Sabbir Ahmed `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型（LLM）在谎言检测中的表现，提出了一种基于角色扮演的无标签检测方法，并扩展了已有的角色库，以提高对知识谎言的检测能力。

**💡 创新点**

创新点在于引入了角色扮演的无标签方向，并通过扩展角色库来提高谎言检测的准确性，尤其是在知识谎言的场景中。

**🔧 技术方法**

使用了线性探测器和角色坐标作为检测技术，结合了知识方向的比较方法。

**📊 数据集**

使用了Liars' Bench数据集以及其他多个模型的基准数据集进行实验。

**📈 对比分析**

与传统的监督探测器相比，所提出的检测器在三个模型上表现更好，但在某些情况下未能超越随机方向的表现，尤其是在Qwen模型上表现不佳。

**⚠️ 局限性**

限制在于所提出的角色和知识方向是在观察到Qwen模型失败后设计的，因此并未进行盲测；此外，知识方向的效果依赖于模型的模板，且在某些情况下未能有效区分谎言和诚实的否认。

---

## 91. ScaleWire: An End-to-End Scale-Up Ethernet Simulation Framework for LLM Systems

**arXiv ID:** 2611.26086 | [PDF](https://arxiv.org/pdf/2611.26086v1)

**作者:** Kai Ma `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Ruoqi Li `[通讯]` (Institute of Information Engineering, Chinese Academy of Sciences)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一个名为ScaleWire的端到端以太网仿真框架，用于大规模语言模型（LLM）系统的网络接口设计。

**💡 创新点**

创新点在于结合了跨层的数据包级仿真与资源探索引擎，以在工作负载完成时间约束下对片上资源进行合理配置。

**🔧 技术方法**

使用了数据包级仿真技术，结合分析模型和资源探索引擎。

**📊 数据集**

使用了vLLM预填充跟踪和在GH200 GPU上测量的内核时间进行评估。

**📈 对比分析**

与传统的穷举搜索方法相比，资源探索引擎在网格比较实验中仅需6-19%的仿真次数，且在大多数情况下，其预算与最佳网格结果相匹配或低于最佳结果。

**⚠️ 局限性**

限制在于资源配置可能会受到工作负载和网络条件的影响，且在某些情况下，独立配置的资源组合可能会违反完成时间约束。

---

## 92. Risk-Controlled Test-Time Adaptation for Vision-Language-Action Models

**arXiv ID:** 2611.26438 | [PDF](https://arxiv.org/pdf/2611.26438v1)

**作者:** Zixuan Liu `[一作]` (Tulane University), Siavash Khajavi `[通讯]` (Aalto University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种风险控制的检索增强视觉-语言-行动（VLA）框架R2-VLA，旨在通过最小化专家协助来满足用户指定的任务成功率要求，从而实现可靠的测试时适应。

**💡 创新点**

R2-VLA通过学习不确定性预测器来判断何时自主行动以及何时请求专家协助，连接了检索演示的有效性与任务级风险控制。

**🔧 技术方法**

使用了不确定性预测技术，结合风险控制程序来校准不确定性阈值，以满足用户指定的任务失败风险要求。

**📊 数据集**

在LIBERO基准测试套件上进行了实验，涉及三个VLA骨干网络和四个不同的任务套件，使用了20个专家演示来构建检索缓冲区。

**📈 对比分析**

与RICL方法相比，R2-VLA在任务成功率上提高了9.2%到11.2%，在大多数设置中达到了预期目标，同时减少了专家请求的频率。

**⚠️ 局限性**

R2-VLA的局限性在于其依赖于检索缓冲区的质量和更新，可能在某些情况下无法完全消除对专家的依赖。

---

## 93. Interlacing Committees with Edge-Expansion

**arXiv ID:** 2611.25852 | [PDF](https://arxiv.org/pdf/2611.25852v1)

**作者:** Gili Bielous `[一作]` (Technion Israel Institute of Technology), Reshef Meir `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文重新审视了基于批准的委员会选择中的连通性度量，提出了边扩展作为优化目标，并通过理论基础将其与信息扩散模型联系起来。

**💡 创新点**

创新点在于引入边扩展这一经典图论度量，作为评估委员会质量的新标准，并与现有的连通性目标进行比较，揭示了它们在结构属性上的定性差异。

**🔧 技术方法**

使用了图论中的边扩展度量，并结合了信息扩散模型进行分析。

**📊 数据集**

使用了基于批准的多赢家投票的选举数据集，具体的投票者和候选人设置在文中进行了详细描述。

**📈 对比分析**

与现有方法相比，边扩展在优化委员会选择时表现出更好的性能，尤其是在处理投票者偏好结构时，尽管在某些情况下仍然存在计算复杂性的问题。

**⚠️ 局限性**

限制在于即使在投票者区间偏好下，优化边扩展的问题仍然是NP-hard，并且在固定委员会大小的情况下，计算复杂性依然较高。

---

## 94. Backbone Matters: Benchmarking Encoder-Decoder Networks for Semantic Segmentation of Driving Scenes on BDD100K

**arXiv ID:** 2611.26191 | [PDF](https://arxiv.org/pdf/2611.26191v1)

**作者:** Ehsan Rassekh `[一作]` `[通讯]` (Institute for Advanced Studies in Basic Sciences), Ehsan Rassekh (Institute for Advanced Studies in Basic Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究对52种语义分割模型进行了系统的基准测试，探讨了不同骨干网络对驾驶场景语义分割质量的影响。

**💡 创新点**

创新点在于系统性地比较了13种骨干网络与4种解码器的组合，发现FPN与EfficientNet-B4的组合在性能上明显优于其他组合。

**🔧 技术方法**

使用了卷积神经网络（CNN）作为骨干网络，并结合了U-Net、FPN、LinkNet和PSPNet四种解码器。

**📊 数据集**

使用了BDD100K数据集，该数据集包含100K个驾驶视频，涵盖不同城市、天气和时间条件。

**📈 对比分析**

通过与BDD100K基准的比较，FPN与EfficientNet-B4的组合达到了62.6%的平均交并比（mIoU），显著高于基准的56.9%。

**⚠️ 局限性**

限制在于未考虑不同骨干网络在特定条件下的表现差异，以及模型在不平衡数据上的表现可能不够全面。

---

## 95. Studying AI Act Mentions: A first look at the impact of the European Union's AI regulation on Hugging Face

**arXiv ID:** 2611.26050 | [PDF](https://arxiv.org/pdf/2611.26050v1)

**作者:** Anamaria Mojica-Hanke `[一作]` (University of Passau), Steffen Herbold `[通讯]` (University of Passau)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对欧盟人工智能法案在Hugging Face平台上的影响进行了实证研究，分析了超过500万条记录中提及该法案的情况。

**💡 创新点**

创新点在于首次系统性地检查了Hugging Face平台上关于人工智能法案的显性提及，涵盖了超过420万条卡片、100万条讨论和7300万条提交记录。

**🔧 技术方法**

使用了实证软件工程的方法，通过手动分类和分析记录，识别出与法案相关的不同类别。

**📊 数据集**

数据集包括Hugging Face平台上的模型、数据集和空间的公共记录，涉及的记录超过500万条。

**📈 对比分析**

与其他方法相比，研究发现显性提及法案的记录非常稀少，主要集中在合规和责任声明上，而不是如何实现合规。整体上，合规声明更关注责任归属而非合规实现的具体方式。

**⚠️ 局限性**

限制在于研究仅限于Hugging Face平台，未考虑其他平台的实践，且只分析了英文内容，可能遗漏了其他语言的相关信息。

---

## 96. ZeroVLN: Learning to Navigate without Supervised Fine-Tuning of the VLM Brain

**arXiv ID:** 2611.26414 | [PDF](https://arxiv.org/pdf/2611.26414v1)

**作者:** Zixuan Huang `[一作]` (Beihang University), Deqing Wang `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的慢-快导航框架，允许通用视觉语言模型（VLM）通过语义工具使用来指导动作专家执行，而无需进行导航领域的监督微调（SFT）。

**💡 创新点**

创新点在于引入了零导航领域SFT的语义工具接口，结合请求引导和执行前缀监督与联合代理强化学习（RL），在有限数据下实现强大的导航性能，同时大幅保留VLM的通用能力。

**🔧 技术方法**

使用了联合代理强化学习（RL）技术，结合请求引导和执行前缀监督来优化VLM大脑的运动决策和动作专家的连续执行。

**📊 数据集**

在三个主流的VLN-CE基准上进行评估，包括SAGE-Bench、R2R-CE和RxR-CE，使用了共计34,445个导航训练集。

**📈 对比分析**

与基于SFT的基线方法相比，ZeroVLN在相同的训练集上表现出更高的成功率（SR），且VLM权重位移仅为0.01497%，在保留通用视觉能力的同时显著提高了导航性能。

**⚠️ 局限性**

限制在于尽管ZeroVLN在多个基准上表现优异，但在路径效率上可能存在权衡，尤其是在R2R和RxR基准上，其成功率高但路径长度效率较低。

---

## 97. Validation-driven Deployment of LLM Inference Platforms for Agentic Workloads

**arXiv ID:** 2611.25905 | [PDF](https://arxiv.org/pdf/2611.25905v1)

**作者:** Shai Gretz `[一作]` (IBM Research), Michal Shmueli-Scheuer `[通讯]` (IBM Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出了一种端到端的评估管道，用于代理工作负载的LLM推理平台，强调了从小规模测试到生产的生命周期验证的重要性。

**💡 创新点**

创新点在于设计了一个连接各个阶段的评估管道，结合了实时基准测试、依赖感知的追踪重放、合成工作负载生成和回归分析，确保了评估的连贯性和准确性。

**🔧 技术方法**

使用了实时基准测试、依赖感知的追踪重放和合成工作负载生成等技术。

**📊 数据集**

使用了包含10,056个代理会话的追踪数据集，记录了完整的请求内容。

**📈 对比分析**

与其他方法相比，论文中的追踪重放能够更好地保持每会话的负载特征，95百分位输入与记录的差异最多为1.18倍，而实时重放可达4.98倍，显示出更高的稳定性和一致性。

**⚠️ 局限性**

限制在于覆盖率收集不完整，仅测量了vLLM的Python代码，而未涵盖其编译内核，且结果来自于单一模型、硬件和配置，可能不具普遍性。

---

## 98. A$^2$-VLA: Adapting Vision-Language-Action Policies to New Tools through Agentic Affordance Discovery

**arXiv ID:** 2611.25656 | [PDF](https://arxiv.org/pdf/2611.25656v1)

**作者:** Chenyuan Zhou `[一作]` (Shanghai Jiao Tong University), Yong-Lu Li `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种工具适应框架，结合了一个具身代理和视觉-语言-行动（VLA）策略，能够从少量源工具的遥操作演示中自主确定任务相关的功能特征。

**💡 创新点**

创新点在于将工具的功能特征视为可实验验证的假设，代理能够在模拟中重建场景并通过自主实验来提取和验证这些特征，从而实现对新工具的适应。

**🔧 技术方法**

使用了具身代理和视觉-语言-行动（VLA）模型，结合了功能区域特征的编码和低维特征向量的生成。

**📊 数据集**

使用了少量源工具的遥操作演示数据集，并在模拟环境中进行测试和验证。

**📈 对比分析**

与其他工具适应方法相比，该方法在模拟交叉工具轨迹重定向中取得了86.1%的成功率，且在三项真实机器人工具使用任务中，相较于基线模型提高了30.7个百分点的任务成功率。

**⚠️ 局限性**

限制在于该方法目前仅适用于可以通过重定向有效解决的工具转移问题，某些形式的工具技能转移在实际部署中无法简单通过重定向解决。

---

## 99. Visual Attribution Limits and Graded Provenance Auditing in Commercial Video-Generation APIs

**arXiv ID:** 2611.26508 | [PDF](https://arxiv.org/pdf/2611.26508v1)

**作者:** Haocheng Mei `[一作]` (Peking University), Yunhe Wang `[通讯]` (TokenRhythm Technologies)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本论文研究了商业视频生成API的模型和层级的归属与认证问题，分析了交付的视频文件如何反映生成模型和层级的信息。

**💡 创新点**

创新点在于提出了一种结合视觉取证和密码学来源认证的方法，评估了不同视频生成模型的归属和认证能力，并提出了买方的补救措施。

**🔧 技术方法**

使用了零-shot检测器（D3和STALL）和C2PA（内容创作者保护协会）认证技术，结合了视觉特征提取和统计分析。

**📊 数据集**

使用了605个交付样本，涵盖19个视频生成模型，包括不同的层级和处理条件。

**📈 对比分析**

通过与其他模型的比较，发现视觉取证的准确性在不同处理条件下变化，且在没有层级认证的情况下，网关标签无法独立确认生成层级。性能方面，D3检测器在有效采样率为3fps时AUROC达到0.988，而STALL在同样条件下表现较差。

**⚠️ 局限性**

限制在于所有交付样本均通过一个网关，没有直接的供应商API合同，且仅有一个供应商的层级认证是可验证的，其他模型的认证能力较弱。

---

## 100. Sightline: Calibrated Data-Flow Summaries for LLM-Integrated Applications

**arXiv ID:** 2611.26221 | [PDF](https://arxiv.org/pdf/2611.26221v1)

**作者:** Zihao Xu `[一作]` (UNSW Sydney), Haifeng Shen `[通讯]` (Southern Cross University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的数据流摘要方法，旨在分析集成大型语言模型（LLM）的应用程序，特别是在处理自然语言与编程语言之间的数据流时。

**💡 创新点**

创新点在于通过注意力机制证明了占位符仅通过读取它的注意力流向输出，并且提出了一种校准的摘要方法，能够在每个调用站点上提供统计保证。

**🔧 技术方法**

使用了Lean 4进行形式化证明，并通过一次前向传递读取模型的注意力，结合学习的读取输出进行流的修剪。

**📊 数据集**

使用了来自2103个未见过的开源代码库的2103个调用站点的数据集，包含8105个占位符-输出对。

**📈 对比分析**

与七种最接近的方法进行比较，结果显示该方法在所有四个模型上都具有最高的准确性，Macro-F1分数在81.1%到86.7%之间，显著优于其他基线方法。

**⚠️ 局限性**

限制在于该方法的校准依赖于训练数据的分布，可能在面对不同分布的数据时表现不佳。

---

## 101. Seeing Beyond the Obvious: Complementary Semantic Guidance for Salient Single-Positive Multi-Label Learning

**arXiv ID:** 2611.25986 | [PDF](https://arxiv.org/pdf/2611.25986v1)

**作者:** Tong Jin `[一作]` (Nanjing University of Aeronautics and Astronautics), Sheng-Jun Huang `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为CoST的框架，用于改进单正多标签学习（SalSPML），通过补充类语义信息来增强对隐藏正样本的监督。

**💡 创新点**

创新点在于引入覆盖引导的补充软目标精炼策略，通过选择性增强类的软目标来提高对隐藏正样本的识别能力。

**🔧 技术方法**

使用了覆盖引导的补充软目标精炼框架，结合了教师引导的候选提议和CLIP基础的语义验证技术。

**📊 数据集**

在多个基准数据集上进行了实验，包括MS-COCO、Pascal VOC和NUS-WIDE。

**📈 对比分析**

与现有基线方法相比，CoST在多个数据集上表现出显著的性能提升，尤其是在COCO数据集上，取得了3.90的mAP提升。

**⚠️ 局限性**

限制在于未知类的候选选择可能引入不可靠的语义关联，且模型的性能可能受到选择性增强策略的影响。

---

## 102. DARSY: A Tool for Data-Driven Abstraction and Policy Synthesis for Stochastic Systems with Unknown Dynamics

**arXiv ID:** 2611.25654 | [PDF](https://arxiv.org/pdf/2611.25654v1)

**作者:** Thom Badings `[一作]` (RWTH Aachen University), Alessandro Abate `[通讯]` (University of Oxford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种用于不确定动态系统的数据驱动抽象和策略合成的软件工具，能够在没有系统动态和噪声分布知识的情况下进行抽象和策略合成。

**💡 创新点**

该工具是完全自动化的，构建了一个有限区间马尔可夫决策过程（IMDP）抽象，并通过统计技术和Lipschitz推理获得转移概率区间，提供了基于用户指定置信度的概率下界。

**🔧 技术方法**

使用了Python和JAX实现，支持Lipschitz连续系统动态和以语法安全LTL（scLTL）表达的规范，采用了数据驱动的IMDP和二层IMDP（2IMDP）抽象。

**📊 数据集**

在多个线性和非线性随机系统基准上进行了评估，使用了从模拟器中采样的转移数据集。

**📈 对比分析**

与现有的模型基础工具相比，该工具在合成策略时提供了更紧的满足概率下界，尤其是在复杂的基准测试中，2IMDP抽象通常优于IMDP抽象。

**⚠️ 局限性**

该工具的局限性在于仍然依赖于Lipschitz常数的知识，并且在处理高维系统时可能面临性能瓶颈。

---

## 103. PUPPET: Physical-Unified Platform for Presence, Embodiment, and Teleoperation

**arXiv ID:** 2611.26350 | [PDF](https://arxiv.org/pdf/2611.26350v1)

**作者:** Mahika Vats `[一作]` (IE University), Eduardo Castelló Ferrer `[通讯]` (IE University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了PUPPET，一个网络物理框架，通过持续同步的虚拟现实（VR）复制品打开物理机器人实验室，以解决远程访问机器人硬件时实验数据不完整或不一致的问题。

**💡 创新点**

创新点在于将物理实验室与虚拟环境同步，提供沉浸式访问，并集成了试验验证管道，以确保每次试验的数据完整性和可重用性。

**🔧 技术方法**

使用了虚拟现实技术（VR）和网络物理系统，结合了机器人控制、数据收集和验证的集成工作流程。

**📊 数据集**

在一项包含30名参与者的用户研究中，使用PAL Robotics的TIAGo机器人进行桌面到架子物体转移任务，比较了面对面、传统2D远程操作和基于PUPPET的VR操作。

**📈 对比分析**

与传统2D远程操作相比，VR操作在任务完成时间上减少了27.7%，并且产生了更多的验证准备试验（85.6%对66.0%）和无需干预的试验（89.7%对67.2%）。

**⚠️ 局限性**

限制在于只比较了完整的交互方法，无法单独分离空间可视化和具身控制的影响；此外，实验只使用了一种机器人和一种任务，限制了结果的普遍性。

---

## 104. LINEAGE: A Benchmark for Empirical Lineage Retrieval in AI Scientific Literature

**arXiv ID:** 2611.25845 | [PDF](https://arxiv.org/pdf/2611.25845v1)

**作者:** Pangjing Wu `[一作]` (Hong Kong Polytechnic University), Qing Li `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个新的基准Lineage，用于评估AI科学文献中的经验谱系检索，旨在填补现有文献检索基准的不足。

**💡 创新点**

创新点在于引入经验谱系关系作为文献检索的真实信号，能够评估检索方法是否恢复了基于实验的经验邻域。

**🔧 技术方法**

使用了Lineage-Constructor技术，自动构建经验谱系图，包含统一文献收集、基于证据的谱系提取和图校准等模块。

**📊 数据集**

数据集包含121,048篇来自2018年至2025年间17个顶级AI会议的论文，构建了3,950个谱系图组件和80,000个多面检索查询。

**📈 对比分析**

与现有的基于相关性和引用的文献检索基准相比，Lineage在检索性能上表现出显著的挑战，尤其是随着谱系距离的增加，性能急剧下降。

**⚠️ 局限性**

限制在于当前的检索方法仍然难以全面检索谱系论文，尤其是在长距离的经验比较链条上，未来的检索方法可能需要更明确的谱系扩展或谱系感知的监督。

---

## 105. Agentic Scaling of Embodied Tasks in Simulation for Synthetic Data Collection and Policy Evaluation

**arXiv ID:** 2611.26046 | [PDF](https://arxiv.org/pdf/2611.26046v1)

**作者:** Kang Wang `[一作]` (Hubei University), Zhendong Wang `[通讯]` (Linctex)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于代理的框架，通过人类意图扩展厨房任务集合，利用可用资产在不同厨房布局中构建任务。

**💡 创新点**

创新点在于将任务生成与执行结合，通过共享任务表示连接交互要求、物理实体绑定和完成标准，从而提高任务的可用性和生成效率。

**🔧 技术方法**

使用了基于代理的框架，结合了任务生成代理和任务执行代理，利用GPT-6 Astra进行任务生成和执行。

**📊 数据集**

使用了RoboCasa作为测试平台，构建了包含16,450个任务实例的AgenticTask集合。

**📈 对比分析**

与传统方法相比，该框架在构建过程中节省了80.94%的任务实例，执行成功率约为38%。通过诊断分析，识别出行动选择和任务-机器人兼容性等主要改进方向。

**⚠️ 局限性**

限制在于任务集合的覆盖和准确性依赖于可用资产、类别/部件注释和模拟器接口，执行结果受模型、技能库和机器人能力的影响，现实世界的应用仍需进一步测试。

---

## 106. A Task-Level Bias Taxonomy for Historical Image Retrieval

**arXiv ID:** 2611.26019 | [PDF](https://arxiv.org/pdf/2611.26019v1)

**作者:** Omkar Gavali `[一作]` (Philipps-Universität Marburg), Anett Hoppe `[通讯]` (Philipps-Universität Marburg)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文探讨了历史图像检索中的偏见问题，特别是在殖民背景下的摄影档案中，分析了偏见的来源及其对AI支持的检索系统的影响，并评估了现有的偏见缓解技术。

**💡 创新点**

创新点在于系统性地将机器学习公平性分类法中的偏见类别映射到与殖民背景摄影档案相关的检索任务，并分析了标准缓解技术及其在这些偏见中的局限性。

**🔧 技术方法**

使用了机器学习和计算机视觉技术，特别是针对历史图像的元数据生成和分类模型，以及对偏见的对抗性去偏技术。

**📊 数据集**

使用了VABiKo项目中的数据集，该数据集包含超过50,000张在1887年至1943年间由德国殖民社会收集的历史照片。

**📈 对比分析**

通过与现有的偏见缓解技术进行比较，发现标准技术在处理历史图像时的有效性较低，尤其是在描述性元数据稀缺的情况下，许多技术未能有效缓解偏见。

**⚠️ 局限性**

限制在于现有的偏见缓解技术主要针对现代图像，缺乏针对历史图像特有条件的有效解决方案，且许多技术在历史材料上未经过验证。

---

## 107. Prism: Hierarchical Text Detection with Minimalist Design

**arXiv ID:** 2611.26254 | [PDF](https://arxiv.org/pdf/2611.26254v1)

**作者:** Yuchen Su `[一作]` (Fudan University), Zhineng Chen `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Prism的简约层次文本检测框架，旨在通过简单的架构实现高效的文本检测。

**💡 创新点**

创新点在于采用简约设计理念，证明复杂的多尺度网络和重头部在强大的预训练模型支持下变得多余。

**🔧 技术方法**

使用了简化的ResNet50-DCN作为主干网络，并采用像素级回归头作为主要输出模块。

**📊 数据集**

使用了多个数据集进行评估，包括HierText、Total-Text、CTW1500、MSRA-TD500等。

**📈 对比分析**

与现有方法（如OmniParser V2）相比，Prism在HierText数据集上在各层次的PQ指标上均有显著提升，同时在推理速度上也表现出色，速度比LRANet快1.7倍。

**⚠️ 局限性**

局限性在于该方法依赖于强大的预训练模型，若预训练数据不足，可能会影响性能。

---

## 108. HALTT: A HAte Lexicon Through Time

**arXiv ID:** 2611.25862 | [PDF](https://arxiv.org/pdf/2611.25862v1)

**作者:** Dario Franzese `[一作]` (University of Torino), Viviana Patti `[通讯]` (University of Torino)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种多阶段框架，用于构建和验证具有语义转变的仇恨言论词典，解决了自动化仇恨言论检测随着日常词汇被武器化而退化的问题。

**💡 创新点**

创新点在于构建了第一个具有时间变化的仇恨言论词典（HALTT），并通过多阶段的过滤和验证流程，识别出19个具有多义性的仇恨言论词汇。

**🔧 技术方法**

使用了大规模文本语料库、掩蔽定义评分和双角色推理提示等技术，结合了历史和现代语料的频率基础。

**📊 数据集**

使用了1770年至1964年的历史语料库和2001年至2024年的现代语料库，涵盖了超过5000万条历史报纸文章和4600万条现代社交媒体帖子。

**📈 对比分析**

与现有的静态仇恨言论词典相比，提出的方法在识别多义性和语义转变方面表现出更高的准确性和稳定性，能够有效减少误报率。

**⚠️ 局限性**

限制在于依赖于选择的语料库和词典，可能导致不同的候选词和转变特征；此外，所有最终确定的词汇仅显示从中性历史用法到现代贬义词的单向转变。

---

## 109. LG-FiLM: Local-Global Mixture-of-Experts Modulation for RAW-to-RGB Reconstruction

**arXiv ID:** 2611.25639 | [PDF](https://arxiv.org/pdf/2611.25639v1)

**作者:** Fei Li `[一作]` (LiAuto), Kun Zhan `[通讯]` (LiAuto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为LG-FiLM的模型，通过在共享的编码器-解码器瓶颈处引入混合专家策略，来实现RAW到RGB图像重建的适应性处理。

**💡 创新点**

创新点在于将输入自适应的图像信号处理形式化为在共享重建网络中密集混合的仿射特征变换，分离条件调制与重建过程。

**🔧 技术方法**

使用了混合专家策略和局部-全局条件设计，结合深度学习技术，构建了一个共享的编码器-解码器架构。

**📊 数据集**

使用了三个数据集进行评估：Fujifilm UltraISP、Zurich RAW到RGB和一个基于COCO的基准数据集。

**📈 对比分析**

与其他方法相比，LG-FiLM在Fujifilm UltraISP基准上达到了24.998 dB的PSNR和0.88366的SSIM，优于其他已报告的方法，且在其他两个数据集上也表现良好。

**⚠️ 局限性**

局限性在于专家数量和条件设计在扩展配置中一起变化，且COCO衍生基准是合成的，主要用于探测管道行为而非真实传感器的泛化能力。Fujifilm UltraISP主要包含白天场景，缺乏多样化的光照条件评估。

---

## 110. Federated Learning of Car Aerodynamics Surrogates with Spectral Optimizers

**arXiv ID:** 2611.25300 | [PDF](https://arxiv.org/pdf/2611.25300v1)

**作者:** Arman Bolatov `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Aleksandr Beznosikov `[通讯]` (Basic Research of Artificial Intelligence Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在无法共享设计的情况下，汽车气动代理的联邦训练，使用了DrivAerNet++数据集。

**💡 创新点**

提出了两种新的联邦光谱优化器FedMARS-M和FedOLion，并建立了气动代理的联邦基准。

**🔧 技术方法**

使用了光谱优化器和Transolver模型进行训练，Transolver是一种针对大网格的变换器。

**📊 数据集**

使用了DrivAerNet++数据集，包括6857辆汽车的表面数据，约125GB。

**📈 对比分析**

与标准的联邦优化器（如FedAvg、SCAFFOLD和FedLion）相比，FedMARS-M和FedMuon在每轮2到4个本地周期内达到了与集中式Muon相近的拖曳误差，而非光谱方法需要16个本地周期，性能较差。

**⚠️ 局限性**

研究的局限性包括仅使用一个数据集、一个架构和六个客户端，且未考虑不同客户端在网格、求解器和流动条件上的差异。

---

## 111. Distortion in Online Voting

**arXiv ID:** 2611.25709 | [PDF](https://arxiv.org/pdf/2611.25709v1)

**作者:** MohammadHossein Ghasemi `[一作]` (Chinese University of Hong Kong), Saeed Seddighin `[通讯]` (Toyota Technological Institute)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本研究探讨了在线投票中的失真问题，提出了一种动态环境下的投票机制，其中候选人按顺序到达，投票机制必须立即决定是否更换当前候选人。

**💡 创新点**

创新点在于提出了在线投票模型的失真界限，并通过引入正则性属性，将离线随机投票规则转化为在线规则，同时保持相同的结果分布。

**🔧 技术方法**

使用了确定性和随机性机制，分别在效用框架和度量框架下进行分析。

**📊 数据集**

未具体提及使用的数据集，但研究涉及的投票机制和偏好模型是基于理论构建的。

**📈 对比分析**

与离线设置相比，在线机制的失真界限为Ω(mn)，而在效用框架下，提出的在线算法的失真为O(√(mlog m))，在度量框架下，确定性机制的失真界限为2+√(5)。

**⚠️ 局限性**

限制在于在线投票模型的失真界限与离线模型相比仍然较高，且在随机效用设置中，已知的下界与上界之间存在差距。

---

## 112. Optimal Preemption: Guaranteeing Information Freshness in Mean Age and Beyond

**arXiv ID:** 2611.26378 | [PDF](https://arxiv.org/pdf/2611.26378v1)

**作者:** Aimin Li `[一作]` (KTH Royal Institute of Technology), Elif Uysal `[通讯]` (Middle East Technical University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种框架，允许发送者在服务中中断更新并采样新的更新，以保证信息的新鲜度，尤其是在重尾延迟的情况下。

**💡 创新点**

创新点在于将预emption视为一种控制动作，优化了随机到达下的预emption和生成随意更新下的采样与预emption，提出了在一般服务时间下的平均信息年龄和尾风险的优化方法。

**🔧 技术方法**

使用了分段确定性马尔可夫过程建模系统，推导了平均成本最优性方程，并开发了Tailor算法进行政策迭代。

**📊 数据集**

使用了重尾延迟（如Pareto和对数正态分布）作为数据集，进行了数值模拟以验证算法的有效性。

**📈 对比分析**

与现有方法比较，Tailor算法在重尾延迟下保持平均成本有界，而非预emption采样的平均成本则会发散。Tailor在信息新鲜度方面表现出色，尤其是在延迟变异性较大的情况下。

**⚠️ 局限性**

限制在于算法在处理无限方差的延迟时可能面临挑战，且在实际应用中需要考虑如何有效利用现有的取消钩子。

---

## 113. Learning to Defer under Expert Drift

**arXiv ID:** 2611.26440 | [PDF](https://arxiv.org/pdf/2611.26440v1)

**作者:** Ziyao Mou `[一作]` (Tufts University), Eric Nalisnick `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的专家漂移适应学习框架（EDA-L2D），该框架通过建模人类专家性能的时间漂移来提高机器学习和人工智能系统的安全性，允许系统在专家表现不稳定时更好地分配决策。

**💡 创新点**

创新点在于显式建模专家性能的时间变化，允许系统根据专家的历史表现动态调整决策，克服了传统L2D方法假设专家表现固定的局限。

**🔧 技术方法**

使用了序列模型来整合专家的历史表现，并通过递归神经网络（RNN）来增强模型的时间感知能力。

**📊 数据集**

在三个不同的任务上进行了验证：CIFAR-10图像分类、CheXpert胸部X光诊断和仇恨言论检测，使用了真实的专家漂移数据和合成的漂移模型。

**📈 对比分析**

与传统的L2D方法相比，EDA-L2D在处理专家表现的时间变化时表现出更好的适应性，系统性能显著提高，尤其是在专家表现不稳定的情况下，准确率普遍提升。

**⚠️ 局限性**

限制在于当前模型仅考虑时间步长对专家表现的影响，而未考虑案例难度等其他因素，未来需要更多的真实工作会话数据来验证和扩展模型的适用性。

---

## 114. LM-based Personalized Travel Planner

**arXiv ID:** 2611.25531 | [PDF](https://arxiv.org/pdf/2611.25531v1)

**作者:** Amandi Nimasha `[一作]` (University of Moratuwa), Srinath Perera `[通讯]` (WSO2)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于层次多智能体框架的个性化旅行规划系统，通过一个监督聊天机器人协调多个专门代理进行行程验证、用户查询构建、实时数据检索和约束感知的行程生成。

**💡 创新点**

创新点在于结合了大型语言模型（LLM）推理、外部API和多智能体协作，以生成个性化和适应性的行程，同时通过确定性的多维评分框架进行迭代优化。

**🔧 技术方法**

使用了层次多智能体系统、LLM推理、外部API集成和迭代优化机制。

**📊 数据集**

使用了TravelPlanner基准数据集，该数据集包含1225个查询，涵盖192个出发地和127个目的地，要求满足复杂的多维约束。

**📈 对比分析**

与单智能体基线（如ReAct）相比，评估结果显示预算约束规划提高了10%，更广泛的约束集规划提高了25%，个性化规划结果提高了45%。

**⚠️ 局限性**

限制包括当前大型语言模型的能力限制、迭代优化引入的延迟、评估基于美国数据集的局限性、个性化信号的不足以及依赖于评分标准而非以用户为中心的研究。

---

## 115. Anchored Inertial Proposals with Objective Verification for Diffusion-Prior Inverse Problems

**arXiv ID:** 2611.25551 | [PDF](https://arxiv.org/pdf/2611.25551v1)

**作者:** Minwoo Kim `[一作]` (Inha University), Hongki Lim `[通讯]` (Inha University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种名为PIVOT的训练无关的修正规则，用于扩散先验逆问题，通过将提议生成与接受分开来改善图像重建质量。

**💡 创新点**

创新点在于引入了锚定的惯性修正规则，结合测量拟合与噪声水平依赖的近端项，提供了一种局部验证机制来决定是否接受提议。

**🔧 技术方法**

使用了重力球动态（heavy-ball dynamics）和锚定的目标验证机制，结合了测量信息和扩散模型的先验知识。

**📊 数据集**

在FFHQ和ImageNet数据集上进行了实验，涉及自然图像的逆问题和加速多线圈膝关节MRI重建。

**📈 对比分析**

与现有的扩散基础逆解法（如FAST-DIPS、DAPS等）进行了比较，PIVOT在多个设置中表现出竞争力，尤其是在随机填充和FFHQ像素空间相位恢复中表现最佳，但在某些高动态范围设置中略逊于最佳基线。

**⚠️ 局限性**

局限性在于在某些高动态范围设置中，PIVOT的PSNR表现不如最佳基线，并且在某些任务中，曲率策略的效果依赖于具体任务。

---

## 116. Held While Waiting: What LLM Agent Sessions Reserve on Serverless and Agent Runtimes

**arXiv ID:** 2611.26480 | [PDF](https://arxiv.org/pdf/2611.26480v1)

**作者:** Mekhala Hithyshini Sreenivasa `[一作]` `[通讯]` (Amazon Web Services), Mekhala Hithyshini Sreenivasa (Amazon Web Services)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在无服务器和代理运行时中，LLM代理会话在等待模型响应时所占用的环境资源。通过对Amazon Bedrock AgentCore和AWS Lambda的分析，发现代理会话在等待模型时会保留一个独立的执行环境，导致资源的浪费。

**💡 创新点**

创新点在于提出了共享环境的概念，通过共享多个会话的环境，可以显著减少所需的环境数量，从而提高资源利用率。研究还提出了关闭会话以释放环境的必要性。

**🔧 技术方法**

使用了模拟和实际部署的数据来分析代理会话的性能，采用了多种测量工具来评估CPU使用情况和等待时间。

**📊 数据集**

使用了一个公共基准测试数据集τ^2-bench，该数据集模拟了零售客户服务任务，包含384个会话和3549个步骤。

**📈 对比分析**

与其他方法相比，本文的方法通过共享环境显著减少了所需的环境数量，模拟结果显示，使用共享环境可以将环境数量减少59倍，且增加的延迟在可接受范围内（最多40毫秒）。

**⚠️ 局限性**

限制在于当前的代理运行时只能为每个会话保留一个环境，无法在不保留环境的情况下实现会话隔离。此外，平台的变化需要支持共享环境的能力。

---

## 117. Variational Trajectory Flow Matching with a Non-Linear Interpolant

**arXiv ID:** 2611.26044 | [PDF](https://arxiv.org/pdf/2611.26044v1)

**作者:** Gurjeet Sangra Singh `[一作]` (University of Geneva), Alexandros Kalousis `[通讯]` (University of Applied Sciences Western Switzerland)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种新的方法，通过学习插值函数来改进稀疏和不规则采样下的动态系统建模，避免了固定插值的限制。

**💡 创新点**

创新点在于引入了一种可学习的隐式神经表示（t-INR），作为教师模型提供目标速度，从而提高了预测的准确性和稳定性。

**🔧 技术方法**

使用了隐式神经表示（INR）和变分框架来构建速度场，避免了在推理时使用随机微分方程（SDE）求解器。

**📊 数据集**

在多个动态系统上进行了评估，包括Lorenz吸引子、RLC电路、反应扩散和浅水方程等，使用了不同密度的稀疏观察数据。

**📈 对比分析**

与固定插值基线方法（如线性、二次、三次样条和五次样条）进行比较，结果显示在稀疏观察下，提出的方法在准确性和稳定性上均优于这些基线，尤其是在动态曲率较强的情况下。

**⚠️ 局限性**

限制在于需要预训练t-INR，与固定插值基线相比，最适合稀疏观察和非线性、曲率动态，对于密集数据，线性插值可能已足够。

---

## 118. SEOYUN: Refusal-Inducing Exam Documents for AI-Glasses Pipelines

**arXiv ID:** 2611.26033 | [PDF](https://arxiv.org/pdf/2611.26033v1)

**作者:** Dohyeon Kwon `[一作]` `[通讯]` (PEACE-UNIVERSE), Dohyeon Kwon (PEACE-UNIVERSE)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SEOYUN，一个特定于考试的框架，旨在通过光学和编码器级干预来减少未经授权的机器辅助的有效性。

**💡 创新点**

创新点在于提出了一种拒绝诱导的威胁模型，分析限制和评估协议，专注于考试环境中的应用。

**🔧 技术方法**

使用了光学干预和编码器扰动技术，结合了逆越狱的概念。

**📊 数据集**

使用了本地CSAT候选语料库，具体数据集来自公共Hugging Face资源。

**📈 对比分析**

与现有方法相比，SEOYUN的性能尚未进行模型性能测量或物理验证，因此无法提供具体的性能比较。

**⚠️ 局限性**

限制在于该框架尚未经过验证，没有实证效果结果，且未进行人类受试者研究。

---

## 119. CoRelate: Coordinated Human-Object Motion Synthesis through Interaction Relations

**arXiv ID:** 2611.25166 | [PDF](https://arxiv.org/pdf/2611.25166v1)

**作者:** Zhiwen Qiu `[一作]` (Cornell University), Saleh Kalantari `[通讯]` (Cornell University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为CoRelate的扩散框架，用于协调全身人类与物体的运动合成，确保在物体运动过程中保持接触并满足目标。

**💡 创新点**

创新点在于引入了手部与物体部件的关系反馈和任务反馈，利用这些关系来指导人类与物体的运动更新，同时在生成过程中保持接触。

**🔧 技术方法**

使用了扩散模型和时间修正方法，结合了关系偏差、任务残差和运动特征来共同更新人类和物体的运动。

**📊 数据集**

使用了FullBodyManipulation和ParaHome数据集进行评估，前者包含约10小时的人类与物体运动配对数据，后者捕捉了家庭交互。

**📈 对比分析**

与多个基线方法（如MDM、CHOIS等）进行比较，结果显示在目标准确性、接触F1和完整交互成功率等指标上均表现优越。

**⚠️ 局限性**

局限性包括关系在生成过程中保持固定，可能限制从不适当的参与安排中恢复，且当前表示不支持在运动窗口内的重复抓取。

---

## 120. Variants of Beam Search for DiskANN

**arXiv ID:** 2611.26609 | [PDF](https://arxiv.org/pdf/2611.26609v1)

**作者:** Viraj Agrawal `[一作]` (Microsoft Research), Kirankumar Shiragur `[通讯]` (Microsoft Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究统一并推广了自适应束搜索（ABS）在α可达图上的结果，证明了ABS在α可达图上可以实现更好的近似保证，并设计了一个束预测器来优化查询的列表大小。

**💡 创新点**

创新点在于将ABS与α可达图结合，提供了更紧的近似界限，并提出了束预测器，该预测器根据短期试点搜索的特征预测每个查询的列表大小，从而减少距离计算的总数。

**🔧 技术方法**

使用了自适应束搜索（ABS）和束预测器技术，结合了图论和机器学习的方法。

**📊 数据集**

使用了三个100百万点的基准数据集，包括SIFT100M、DEEP100M和MSSpaceV100M。

**📈 对比分析**

与经典束搜索（CBS）和ABS进行了比较，束预测器在15个设置中有11个比ABS快，速度提升最高可达6.5%；在SSD上，束预测器的查询延迟比CBS低最多29.6%。

**⚠️ 局限性**

限制在于保证仅适用于α可达和排序的α可达图，而实际使用的快速预处理图（如DiskANN）不满足这些条件，因此这些界限不适用于它们。此外，束预测器需要提前选择目标召回率，并且其尾部延迟通常高于CBS。

---

## 121. Power Setpoint Tracking for Large-Scale EV Charging: Transformer-Constrained Charging Policy Learning with Self-Tuning Constraint Prices

**arXiv ID:** 2611.25965 | [PDF](https://arxiv.org/pdf/2611.25965v1)

**作者:** Daniyal Sajedi-Hosseini `[一作]` (Tokyo University of Agriculture and Technology), Xun Shen `[通讯]` (Tokyo University of Agriculture and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种变压器约束的充电策略学习方法（TC-CPL），用于电动汽车（EV）充电的功率设定跟踪（PST），以避免过载和提高用户满意度。

**💡 创新点**

创新点在于将TC-CPL建模为受约束的马尔可夫决策过程（CMDP），并引入自调节的约束价格，结合了比例-积分（PI）控制器和共享的变压器感知编码器。

**🔧 技术方法**

使用了软演员-评论家（SAC）算法，并扩展了其功能以适应变压器约束的充电策略，采用了比例-积分控制器来调节约束乘数。

**📊 数据集**

使用了EV2Gym模拟环境中的数据集，进行了两种场景的验证：一个是42-kW变压器与家庭负载的场景，另一个是122个变电站和600个充电端口的区域。

**📈 对比分析**

与现有的Round-Robin方法相比，TC-CPL在600个端口的场景中，原始过载从3.59 kWh/天降至0.0017 kWh/天，绝对跟踪误差降低了27%，交付的能量增加了16%，用户满意度从0.959提高到0.989。

**⚠️ 局限性**

限制在于所有结果均来自模拟，变压器模型未考虑热动态或老化，过载约束仅近似满足，且在探索过程中可能会过载变压器。

---

## 122. LoongForge-Embodied: A Unified Framework for Training Heterogeneous Embodied Models

**arXiv ID:** 2611.25460 | [PDF](https://arxiv.org/pdf/2611.25460v1)

**作者:** LoongForge Team `[一作]` `[通讯]` (Baidu, Inc.), LoongForge Team (Baidu, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了LoongForge-Embodied，一个统一的框架，用于加速异构的视觉-语言-动作（VLA）模型和世界-动作模型（WAM）的训练。

**💡 创新点**

创新点在于引入了一个通用的抽象层，解耦了模型特定实现与共享训练基础设施，从而实现跨模型的重用，同时不限制各自的架构或执行要求。

**🔧 技术方法**

使用了PyTorch作为基础框架，结合了可重用的加速机制和模型特定的硬件感知优化策略，包括FP8基础的通信优化。

**📊 数据集**

在多个代表性的VLA和WAM模型上进行了实验，具体数据集未详细列出，但提到使用了大型多机器人数据集。

**📈 对比分析**

与相应的开源基线相比，LoongForge-Embodied在端到端训练吞吐量上提高了1.47×到4.38×，在相同协议下的评估速度提高了4.1×到23.2×。

**⚠️ 局限性**

限制在于尽管框架提供了统一的训练基础设施，但在某些情况下，模型特定的优化仍需手动调整，可能导致工程开销增加。

---

## 123. Raptor: Rejection-aware Adaptive Tree based on Speculative Decoding

**arXiv ID:** 2611.25436 | [PDF](https://arxiv.org/pdf/2611.25436v1)

**作者:** Qishu Hu `[一作]` (Beijing Institute of Technology), Yuanqing Xia `[通讯]` (Beijing Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为Raptor的训练无关框架，通过拒绝反馈共同调节候选覆盖和草拟工作，以加速视觉语言模型的推理过程。

**💡 创新点**

Raptor通过提取拒绝反馈信号，揭示了接受长度不足以指导推理计算分配的局限性，并展示了如何利用拒绝反馈来评估候选覆盖和冗余扩展，从而实现更有效的计算分配。

**🔧 技术方法**

使用了拒绝反馈机制和动量平滑更新的控制器来调节树的深度、分支宽度和草拟预算。

**📊 数据集**

在七个多模态基准上进行了广泛实验，使用了两个代表性的视觉语言模型（VLMs）进行评估。

**📈 对比分析**

与现有的自回归解码相比，Raptor在多个基准上实现了高达2.39倍的平均加速，且在大多数评估设置中超越了现有的自适应策略。

**⚠️ 局限性**

Raptor的局限性在于其依赖于拒绝反馈的有效性，且在某些情况下可能无法完全消除冗余计算。

---

## 124. FFGaussType: Feed-Forward Conversion of Glyph and SVG Geometry into Compact Gaussian Assets

**arXiv ID:** 2611.26431 | [PDF](https://arxiv.org/pdf/2611.26431v1)

**作者:** Haoan Feng `[一作]` (University of Maryland), Guan-ming Su `[通讯]` (Dolby Laboratories)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种将已知的字形和SVG几何形状转换为可重用的高斯资产的方法，避免了对每个新资产进行单独拟合。

**💡 创新点**

创新点在于通过一个前向传递将锚点位置映射到特征平面，并使用共享解码器生成具有表面法线的神经高斯体，从而实现高效的内容创建。

**🔧 技术方法**

使用了点编码器和共享解码器的神经网络架构，结合图像和视频编码技术。

**📊 数据集**

构建了一个包含27,221个字形和SVG图标资产的配对数据集，涵盖256个字体系列，并提供锚点和多视图渲染的监督。

**📈 对比分析**

与其他前馈转换器相比，该方法在全测试集上达到了1.62°的法线误差和0.985的轮廓IoU，转换时间为每个资产9.1毫秒，性能优于先前的基线。

**⚠️ 局限性**

限制在于当前研究仅关注于均匀材料的字形和SVG资产，锚点采样仍需单独的预处理步骤，且未支持空间变化材料的映射。

---

## 125. SCARVA: Interpretable Agentic Reasoning in Software Supply-Chain Analysis

**arXiv ID:** 2611.26131 | [PDF](https://arxiv.org/pdf/2611.26131v1)

**作者:** Tim Barsch `[一作]` (Ruhr University Bochum), Flavio Toffalini `[通讯]` (Ruhr University Bochum)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种可解释的框架，用于软件供应链分析中的安全评估，旨在提高对大型语言模型（LLM）代理的信任度。

**💡 创新点**

创新点在于将代理的分析与证据收集方法结合，生成可审计的报告，确保每个结论都有支持证据。

**🔧 技术方法**

使用了大型语言模型（LLM）代理，结合了监控环境中的工具调用、文件访问和网络请求记录。

**📊 数据集**

使用了54个手动策划的案例数据集，涵盖可利用的、不可利用的和无已知漏洞的项目。

**📈 对比分析**

与现有方法（如VEX-Bench）比较，最佳模型Qwen 3.8在平衡准确率上达到82%，超越了评估的专有模型。人类注释验证发现Claude Opus 5.5的准确率为94%。

**⚠️ 局限性**

局限性在于当前的模型在处理没有已知漏洞的库时表现不佳，且在某些情况下可能会产生虚假结论。

---

## 126. Inhabitant: An Agent Framework for Long-Horizon Human-Scene Interaction with Spatial Control and Quantitative Feedback

**arXiv ID:** 2611.25005 | [PDF](https://arxiv.org/pdf/2611.25005v1)

**作者:** Yao Zhang `[一作]` (Aalto University), Yu Xiao `[通讯]` (Aalto University)

**通讯引用:** 4650 | [OpenAlex ID](https://openalex.org/A5069437467)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Inhabitant的训练无关代理框架，旨在解决人类场景交互生成中的空间控制和定量反馈问题，使代理能够在3D场景中根据指令规划和执行一系列交互。

**💡 创新点**

创新点在于提供了空间控制和定量反馈两个接口，使得代理能够精确指定动作并根据执行结果进行修正，从而提高了长时间交互的成功率。

**🔧 技术方法**

使用了Kimodo作为运动生成器，通过自然语言指令和空间约束来控制动作，同时利用半物理场景模拟器进行反馈。

**📊 数据集**

构建了Inhabit基准数据集，包含75个家庭任务和31种家具布局，任务涉及多个目标的交互。

**📈 对比分析**

与现有的三种代理进行比较，Inhabitant在使用Claude Opus 5.5时完成了98.7%的任务，而其他代理的成功率最高仅为26.7%。实验表明，缺乏空间控制和定量反馈会导致任务失败。

**⚠️ 局限性**

局限性包括需要一个可以重复尝试的场景模拟器，身体运动是基于运动学的，缺乏平衡和扭矩控制，且代理依赖于给定的场景状态描述，未来工作将探索更复杂的在线设置和感知能力。

---

## 127. Phantom Success, Silent Failure: Auditing the Success Checkers of Simulated Manipulation Benchmarks

**arXiv ID:** 2611.25179 | [PDF](https://arxiv.org/pdf/2611.25179v1)

**作者:** Zirui Song `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Xiuying Chen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于大型语言模型（LLM）的审计管道，用于评估机器人操作中的成功检查器，旨在验证这些检查器是否准确反映机器人执行任务的情况。

**💡 创新点**

创新点在于引入了一种自动化的审计流程，能够识别和验证成功检查器的缺陷，并通过多视角渲染来确认这些缺陷的存在。

**🔧 技术方法**

使用了大型语言模型（LLM）和模拟物理引擎来构建测试执行，并通过多视角渲染验证检查器的判决。

**📊 数据集**

使用了三个基准数据集：RoboTwin 2.0、RoboCasa365 和 RoboDojo，共审计了142个任务，发现了61个检查器缺陷。

**📈 对比分析**

通过与原始和修复后的检查器对比，发现修复后的检查器在评估政策回放时，判决结果的变化率可达9.1%。修复后的检查器提高了成功率的准确性，揭示了原检查器的缺陷对成功率的影响。

**⚠️ 局限性**

限制在于审计过程依赖于模拟环境的准确性，且可能无法覆盖所有潜在的执行状态，导致某些缺陷未被发现。

---

## 128. Predicting Surface Displacement Fields of Laser Powder Bed Fusion Parts Using STL-Derived Graphs and Part-Level Descriptors

**arXiv ID:** 2611.26323 | [PDF](https://arxiv.org/pdf/2611.26323v1)

**作者:** Saima Ahmed Suhi `[一作]` (University of South Carolina), Md Habibor Rahman `[通讯]` (University of South Carolina)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究提出了一种基于两分支图注意力网络（GAT）的框架，直接从STL派生的图形预测激光粉末床熔融（LPBF）零件的三维表面位移场，以解决在制造前预测零件变形的挑战。

**💡 创新点**

创新点在于结合了局部表面特征和12个零件级几何描述符，通过图神经网络直接预测位移，而无需进行额外的热历史模拟，从而加快设计迭代和早期识别易变形特征。

**🔧 技术方法**

使用了两分支图注意力网络（GAT），该网络通过消息传递机制聚合邻近顶点的信息，并结合全局几何描述符进行位移预测。

**📊 数据集**

使用了117个LPBF零件的扩展数据集，每个零件都有相应的STL文件和ANSYS过程模拟结果，涵盖了不同的几何特征。

**📈 对比分析**

与节点级多层感知机（MLP）相比，GAT在位移幅度的R²值从0.913提高到0.946，且在所有五个折叠中均优于GraphSAGE和图卷积网络（GCN），显示出更高的预测准确性。

**⚠️ 局限性**

限制在于研究仅限于一种材料和一组工艺参数，且使用的标签来自模拟而非实际测量，数据集主要由117个棱柱形零件组成，缺乏足够的几何变化。

---

## 129. Can an LLM Fix a Language Model Without Training It?

**arXiv ID:** 2611.25613 | [PDF](https://arxiv.org/pdf/2611.25613v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 130. DVBench: End-to-End Benchmark for Real-World UVM Design Verification Agents

**arXiv ID:** 2611.25066 | [PDF](https://arxiv.org/pdf/2611.25066v1)

**作者:** Jiaru Qian `[一作]` (Peking University), Ge Li `[通讯]` (Peking University)

**通讯引用:** 16758 | [OpenAlex ID](https://openalex.org/A5100447691)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DVBench，这是第一个用于真实世界UVM设计验证代理的端到端基准，包含95个任务，涵盖9个功能领域和5个难度级别。

**💡 创新点**

DVBench通过综合评估验证文档、参考模型和可执行测试平台，连接验证意图与实际检查效果，填补了现有基准的不足。

**🔧 技术方法**

使用了大型语言模型（LLMs）和代理框架进行实验评估，包括Codex、Claude Code和GPT-6 Astra等。

**📊 数据集**

数据集来源于GitHub，包含约51,000个可综合的RTL设计库，涵盖从课程作业到工业规模的开源项目。

**📈 对比分析**

与现有方法相比，DVBench提供了完整的验证工作流评估，结果显示最强的配置（Codex与GPT-6 Astra）仅实现了21.05%的整体成功率，表明复杂验证任务仍有很大改进空间。

**⚠️ 局限性**

限制在于当前的LLM代理在复杂的端到端验证任务中表现不佳，尤其是在特征分析和测试计划的初始阶段，导致后续验证阶段的缺陷传播。

---

## 131. Spatially Conditioned Trajectory Diffusion

**arXiv ID:** 2611.26127 | [PDF](https://arxiv.org/pdf/2611.26127v1)

**作者:** Yefim Nudelman `[一作]` (Bar-Ilan University), Tom Tirer `[通讯]` (Bar-Ilan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种空间条件轨迹扩散(SCTD)模型，用于多机器人运动规划，能够生成避障轨迹。

**💡 创新点**

创新点在于使用空间条件的占用标记，通过交叉注意力机制在时间U-Net中访问不同的工作空间区域，从而提高轨迹生成的成功率和质量。

**🔧 技术方法**

采用了扩散模型和时间U-Net架构，结合交叉注意力机制进行轨迹生成。

**📊 数据集**

使用了500个地图和15000个任务的占用网格数据集进行训练和评估。

**📈 对比分析**

与现有的引导扩散基线(MPD)相比，SCTD在单机器人成功率和轨迹质量上表现更好，尤其是在困难的绕障任务上，成功率显著提高。

**⚠️ 局限性**

限制在于模型的训练依赖于单机器人演示，可能在复杂的动态环境中表现不如预期。

---

## 132. Context as a Resource

**arXiv ID:** 2611.26346 | [PDF](https://arxiv.org/pdf/2611.26346v1)

**作者:** Xufei Tian `[一作]` (East China University of Science and Technology), Wenli Du `[通讯]` (East China University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无路由架构，通过可插拔的单通道非生成上下文管理单元（Context MMU）直接调度上下文块，旨在提高并发任务的管理效率。

**💡 创新点**

创新点在于引入了一种无路由的上下文虚拟化方法，避免了传统路由架构中的信息丢失和高开销，同时提升了任务合并的准确性。

**🔧 技术方法**

使用了无生成的上下文管理单元（Context MMU）、上下文单元（Context Cells）、同步总线（Synaptic Bus）和共享干线（Shared Trunk）等技术。

**📊 数据集**

使用了三个零重叠的程序基准测试集：Synaptic-Bench-150、AppWorld-Micro-30 和 Synaptic-Stream-60，涵盖了多种任务场景。

**📈 对比分析**

与传统的路由架构相比，该方法在合并准确性上提升至63.3%，同时减少了45.9%的提示令牌使用，且在高并发流中表现出色，完成率达到95.0%。

**⚠️ 局限性**

限制在于控制平面的泛化和定价问题，当前的上下文调度仍受限于冻结的模型权重，未来需要在更大规模的流和合并冲突中进行评估。

---

## 133. UniVEP: Unifying Velocity and Endpoint Prediction for Robot Manipulation

**arXiv ID:** 2611.26103 | [PDF](https://arxiv.org/pdf/2611.26103v1)

**作者:** Zhengtu Liang `[一作]` (Hong Kong University of Science and Technology), Renjing Xu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文系统比较了三种机器人策略的预测目标：速度预测（V）、端点预测（X）和速度加权端点预测（X–V），并提出了UniVEP模型，该模型结合了这两种预测模式以提高机器人操作的效率和成功率。

**💡 创新点**

创新点在于提出了UniVEP模型，通过联合训练速度和端点预测，并根据噪声水平动态切换预测模式，从而在少量推理步骤中实现更高的操作成功率和效率。

**🔧 技术方法**

使用了联合监督学习技术，结合了速度预测和端点预测的损失函数，并在推理过程中根据噪声水平选择合适的预测模式。

**📊 数据集**

使用了多个数据集进行评估，包括LIBERO、LIBERO-Plus和RoboTwin 2.0，涵盖了不同的操作任务和噪声条件。

**📈 对比分析**

与传统的单一目标预测方法相比，UniVEP在少量推理步骤下表现出更高的成功率。例如，在RoboTwin Clean上，UniVEP在一次评估中成功率达到61.36%，超过了最佳单目标基线（59.96%）。在LiLa-WAM上，使用两次评估时成功率达到85.96%，也超过了最佳单目标结果。

**⚠️ 局限性**

限制在于FastWAM的速度预训练组件对预测目标的变化敏感，可能会影响其性能。此外，UniVEP的性能依赖于具体的初始化设置，可能在不同的任务和环境中表现不一致。

---

## 134. What If Chain-of-Thought Is Not Enough?

**arXiv ID:** 2611.26282 | [PDF](https://arxiv.org/pdf/2611.26282v1)

**作者:** Marc Jansen `[一作]` `[通讯]` (University of Applied Sciences Ruhr West), Marc Jansen (University of Applied Sciences Ruhr West)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型（LLM）内部思维模型的可变性，提出了一种新的结构化决策过程QOC（问题-选项-标准），并通过训练模型来改善决策支持的质量和可解释性。

**💡 创新点**

创新点在于用QOC替代传统的Chain-of-Thought（CoT）方法，提供了一种更明确的决策框架，强调了决策问题、竞争选项和评估标准之间的分离，从而提高了模型的内部一致性和输出质量。

**🔧 技术方法**

使用了QOC作为内部思维模型，通过三代模型的开发和训练，逐步引入了结构化决策形成、解释性渲染和统计分析等技术。

**📊 数据集**

使用了合成的决策任务数据集，这些任务涵盖了多个领域，适合QOC风格的分解。

**📈 对比分析**

通过与非微调的基础模型进行比较，微调后的QOC模型在内部一致性和输出质量上表现出显著的改进，尤其是在模型规模较大时，QOC微调的效果更为明显。

**⚠️ 局限性**

限制在于本研究仅评估了一种基础模型家族和一种结构化思维模型，且提出的模型规模与结构深度的关系应视为当前数据的解释，而非普遍规则。

---

## 135. Do Network Mechanisms in Science Vary Across Research Topics? Contextual Relational Hyperevent Models for Large-Scale Bibliometric Data

**arXiv ID:** 2611.26052 | [PDF](https://arxiv.org/pdf/2611.26052v1)

**作者:** Jürgen Lerner `[一作]` (University of Konstanz), Raphael Heiberger `[通讯]`

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种新的模型——上下文关系超事件模型（C-RHEMs），用于分析科学研究中的合作和引用网络，探讨不同研究领域中网络机制的差异。

**💡 创新点**

C-RHEMs的创新点在于允许网络效应在单一模型中根据研究主题变化，从而解决了传统模型在处理大规模文献数据时的局限性。

**🔧 技术方法**

使用了上下文关系超事件模型（C-RHEMs），结合元分析和元回归方法来量化主题特定效应的异质性。

**📊 数据集**

数据集来自PubMed知识图谱，包含约590,000篇肿瘤学相关的出版物，分为32个OpenAlex研究主题。

**📈 对比分析**

通过与传统模型的比较，C-RHEMs在估计网络机制的异质性方面表现出显著优势，尤其是在处理大规模数据时，能够更好地捕捉不同主题间的差异。

**⚠️ 局限性**

限制在于研究仅集中于肿瘤学这一子领域，可能无法推广到其他领域；此外，元回归分析的因果关系尚未得到充分验证，样本量较小可能影响统计功效。

---

## 136. Text-Aware, Vision-Blind: How Multimodal Image Editing Systems Handle Unsafe Inputs

**arXiv ID:** 2611.26569 | [PDF](https://arxiv.org/pdf/2611.26569v1)

**作者:** Ziqing Yang `[一作]` (CISPA Helmholtz Center for Information Security), Yang Zhang `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文介绍了一个基准，旨在测量多模态图像编辑系统在处理不安全输入时的系统级安全意识，包含5940对图像-指令对，涵盖11个风险类别和三种编辑意图。

**💡 创新点**

创新点在于提出了一个新的基准，专注于图像编辑中的视觉内容和文本提示的联合考虑，揭示了当前系统在安全性方面的文本-视觉不匹配现象。

**🔧 技术方法**

使用了多模态大型语言模型（MLLMs）进行评估，包括Google的Gemini、OpenAI的GPT、阿里巴巴的Qwen API和Qwen Local。

**📊 数据集**

数据集来自UnsafeBench，包含660张安全和不安全的图像，配以三种编辑意图的指令，最终形成5940个图像-指令对。

**📈 对比分析**

通过比较四个系统的表现，发现Gemini和GPT在处理不安全意图编辑时拒绝率高达87.1%和81.6%，而在不安全图像与中性指令的情况下，仅拒绝24.2%和22.4%。整体性能显示文本驱动的安全性高于视觉驱动。

**⚠️ 局限性**

限制在于评估的是系统级决策（允许/拒绝/错误），而非输出图像的危害性或编辑质量，未来工作应同时评估输出安全性。

---

## 137. ALIoTh: Orchestrating Multi-Agent LLMs for Autonomous IoT Penetration Testing

**arXiv ID:** 2611.25809 | [PDF](https://arxiv.org/pdf/2611.25809v1)

**作者:** Pironti Francesco Aurelio `[一作]`, Maresca Alessio `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一种多代理、工作流驱动的渗透测试框架，集成了大型语言模型代理，能够自主识别和利用物联网（IoT）和工业物联网（IIoT）环境中的漏洞。

**💡 创新点**

创新点在于结合了自检索增强生成（Self-RAG）架构和多代理系统，能够动态规划和执行全面的攻击流程，提升了渗透测试的自动化和效率。

**🔧 技术方法**

使用了大型语言模型（LLMs）和LangGraph框架来协调多个专门的代理，并集成了自检索增强生成（Self-RAG）机制。

**📊 数据集**

使用了一个典型的智能家居环境的实验室测试床，包含三个已知漏洞的目标IoT设备（IP摄像头、智能插头和MQTT网关）。

**📈 对比分析**

通过定义性能指标（如远程代码执行时间、获取交互式shell的时间、权限提升时间等）来比较不同架构的性能，实验结果显示该框架在自动化渗透测试中表现出色。

**⚠️ 局限性**

局限性在于尚未解决大规模IoT网络的复杂性，未来工作将扩展客户端数据的正式结构，并解决规划代理的可扩展性问题。

---

## 138. BlockJack: Cross-Tenant Redirection Fault Attacks on Paged KV Cache Address Translation in LLM Serving

**arXiv ID:** 2611.25945 | [PDF](https://arxiv.org/pdf/2611.25945v1)

**作者:** Qifan Wang `[一作]` (Durham University), David Oswald `[通讯]` (Durham University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种新的故障注入攻击，称为KV缓存地址转换故障攻击，能够重定向请求的KV缓存读取到另一个租户的缓存状态，从而导致机密性泄露或完整性损害。

**💡 创新点**

首次探索了LLM推理中的KV缓存映射作为攻击面，揭示了其在多租户环境中的安全风险。

**🔧 技术方法**

使用了Rowhammer作为故障注入原语，通过破坏块表条目来实现KV缓存重定向。

**📊 数据集**

在多个模型上进行了评估，包括Qwen、Llama、Gemma和Phi系列，使用了248个场景的医疗基准数据集。

**📈 对比分析**

与以往针对模型权重或KV值的故障攻击相比，本研究的攻击能够直接影响映射元数据，导致机密性和完整性损害。攻击在七个模型中均表现出有效性，机密性泄露和完整性损害的发生率分别高达41.1%。

**⚠️ 局限性**

攻击的局限性在于在嘈杂状态下，攻击者可能无法准确预测目标块ID，导致攻击变得概率性和非针对性。

---

## 139. LLM Complexity: Towards a Computational Complexity View of Large Language Model Inference

**arXiv ID:** 2611.24943 | [PDF](https://arxiv.org/pdf/2611.24943v1)

**作者:** Yu-Hung Wu `[一作]` (National Taiwan University), Shao-Hua Sun `[通讯]` (National Taiwan University)

**通讯引用:** 281 | [OpenAlex ID](https://openalex.org/A5101181948)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了大型语言模型（LLMs）在推理过程中生成的token数量与经典时间复杂度之间的关系，提出了一种新的复杂度度量方法，以量化LLMs的任务复杂性。

**💡 创新点**

创新点在于提出了一种新的复杂度度量，通过准确性-Token曲线的面积来量化问题复杂性，显示出不同LLMs在任务排名上的一致性，并利用这一特性构建了有效的监督微调课程。

**🔧 技术方法**

使用了新的复杂度度量方法，该方法基于LLMs的推理过程，结合了准确性和token预算的关系。

**📊 数据集**

使用了经典算法问题的数据集，包括O(n log n)的排序问题和NP-hard的子集和问题，生成了多个实例以进行评估。

**📈 对比分析**

通过与传统的时间复杂度进行比较，发现token使用与经典时间复杂度之间的相关性较低，而新的复杂度度量在不同模型之间表现出更强的一致性，且在微调过程中提高了准确性并减少了token消耗。

**⚠️ 局限性**

限制在于该度量可能无法完全捕捉人类感知的难度，尤其是在真实世界任务中，LLMs的推理复杂性与人类标注的难度标签之间存在差异。

---

## 140. Intent-Defined Radio: A Goal-Native Architecture for Next-Generation Wireless Networks

**arXiv ID:** 2611.25657 | [PDF](https://arxiv.org/pdf/2611.25657v1)

**作者:** Wencheng Yan `[一作]` (Tsinghua University), Sheng Zhou `[通讯]` (Tsinghua University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的无线网络架构——意图定义无线电（IDR），旨在将服务目标与可执行的无线行为连接起来，以应对日益复杂的无线系统操作。

**💡 创新点**

IDR通过将明确的服务目标与可执行的无线行为相结合，填补了服务意图与无线机制之间的抽象差距，提供了一种系统化的方法来实现无线网络的自我设计、自我部署、自我适应和自我演化。

**🔧 技术方法**

使用了大型语言模型（LLMs）来实现意图解释、行为合成、执行、验证和演化的IDR架构，并通过Intent2MAC进行MAC合成的部分实现。

**📊 数据集**

使用了OpenAI GPT-5.6作为Intent2MAC设计代理，进行了两项模拟研究，分别针对工业多机器人协调和远程控制的上下文感知状态更新。

**📈 对比分析**

与传统的Wi-Fi和静态增强分布式信道访问（EDCA）配置进行比较，IDR候选方案在服务满意度上显著提高，尽管整体吞吐量有所下降，表明在不同服务意图下的机制选择和参数调整是有效的。

**⚠️ 局限性**

IDR的局限性在于需要解决意图语义、合成保证、可观察性和归因、设计复杂性以及实际部署和互操作性等重大挑战。

---

## 141. Sage-Evolve-Preview:Adaptive Observation for Post-Training Self-Improvement

**arXiv ID:** 2611.26132 | [PDF](https://arxiv.org/pdf/2611.26132v1)

**作者:** Yinhao Tang `[一作]` (University of Science and Technology of China), Wenwei Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种自适应观察机制（AdaObs），用于在后训练递归自我改进（RSI）中持续提高模型能力，通过不断干预、训练和评估来优化目标能力。

**💡 创新点**

创新点在于引入了观察适应性，允许在固定的锚目标下调整观察方式，以避免因反馈信息不足而导致的进展停滞。

**🔧 技术方法**

使用了自适应观察机制（AdaObs），通过细化诊断粒度、提高难度和增加相关任务来优化模型。

**📊 数据集**

在长上下文阅读理解和指令跟随的任务上进行了实验，优化了八个独立的锚点，并在PostTrainBench上进行了验证。

**📈 对比分析**

与固定观察（FixedObs）方法相比，AdaObs在相同的数据预算下将平均得分从66.4提高到73.7，表现出显著的性能提升。

**⚠️ 局限性**

限制在于该方法依赖于有效的反馈机制，当反馈信息不足时，仍可能导致进展停滞。

---

## 142. CAIMS: Scalable Cross-Array In-Memory Search

**arXiv ID:** 2611.25141 | [PDF](https://arxiv.org/pdf/2611.25141v1)

**作者:** Yi-Chun Liao `[一作]` (Duke University), X. Sharon Hu `[通讯]` (University of Notre Dame)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种名为CAIMS的跨阵列内存搜索框架，旨在实现可扩展和高效的top-k相似性搜索，同时保持准确性。

**💡 创新点**

CAIMS是首个完全可扩展的跨阵列内存搜索设计，能够在多个内容可寻址内存（CAM）阵列中实现准确和高效的top-k相似性搜索。

**🔧 技术方法**

使用了内容可寻址内存（CAM）技术，结合了数据集辅助可扩展搜索（DASS）、在线结果计数适应（ORCA）和局部聚合与选择性传输（LAST）等新颖方案。

**📊 数据集**

在多个数据集上进行了评估，包括BEIR和DPR数据集，涵盖了不同规模的查询和候选向量。

**📈 对比分析**

与服务器级CPU基线和先前的3D NAND Flash加速器相比，CAIMS在吞吐量上提高了22.3×/10.1×，在能效上提高了426.6×/23.7×，同时保持了相似的准确性。

**⚠️ 局限性**

CAIMS的局限性在于其对查询分布的适应性，尽管DASS和ORCA能够控制平均结果数量，但在某些情况下可能仍无法完全保证选择到准确的k个候选。

---

## 143. Design and Validation of a Telescopic Chassis-Assisted Stair-Climbing Mechanism for Powered Wheelchairs

**arXiv ID:** 2611.25798 | [PDF](https://arxiv.org/pdf/2611.25798v1)

**作者:** Yasuhiro Fukuoka `[一作]` (Ibaraki University), Takahiro Fukui `[通讯]` (Tokyo Polytechnic University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种简单的楼梯攀爬机制，通过辅助的伸缩底盘驱动四轮轮椅，确保在楼梯上升时前后轮中的一个始终保持静止，从而提高安全性。

**💡 创新点**

创新点在于该机制主要依赖伸缩底盘而非轮子来完成楼梯攀爬，且对楼梯倾斜度的要求不敏感，同时降低了在陡峭楼梯上轮子打滑的风险。

**🔧 技术方法**

使用了伸缩底盘驱动的简单机械结构，结合自锁蜗轮和齿轮电机的控制策略。

**📊 数据集**

实验使用了三种不同倾斜度的楼梯（温和、标准和陡峭），并进行了多次攀爬实验。

**📈 对比分析**

与现有的楼梯攀爬轮椅相比，该机制在攀爬过程中所需的机械功主要来自伸缩底盘，实验结果显示，伸缩底盘对总机械功的贡献超过94%。

**⚠️ 局限性**

限制在于目前的系统需要人工控制座椅位置，未来版本将自动调整座椅位置以提高稳定性，同时需要进一步改善操作性。

---

## 144. LQ-Rec: Late Quantization for Sequential Recommendation

**arXiv ID:** 2611.25256 | [PDF](https://arxiv.org/pdf/2611.25256v1)

**作者:** Satoshi Koide `[一作]` (Toyota Motor North America), Julian McAuley `[通讯]` (UC San Diego)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的推荐系统框架，称为Late Quantization (LQ)，用于序列推荐，解决了传统自回归生成检索方法的结构性限制。

**💡 创新点**

LQ通过将量化步骤限制在输出（项目）侧，允许编码器和评分规则在连续空间中操作，从而实现了可加误差分解和配对排序定理，提供了结构性保证。

**🔧 技术方法**

使用了非自回归的LQ框架，结合了残差量化和连续模型，采用了两阶段训练流程（S2H-QA + Prefix-MQA）。

**📊 数据集**

使用了四个公共基准数据集，包括亚马逊评论的玩具、美容、运动类别和Yelp数据集，进行留一法评估。

**📈 对比分析**

与多种基线方法进行比较，LQ-Rec在所有三个能力上均超越了现有方法，且在匹配的语义ID长度下，排名质量随着ID长度单调提高。

**⚠️ 局限性**

LQ的局限性在于其理论结果假设了内积评分规则，扩展到非线性评分函数可能会打破这种精确的可加分离性。

---

## 145. Improved Constants in the Kautz-Singleton Construction for Probabilistic Group Testing

**arXiv ID:** 2611.25468 | [PDF](https://arxiv.org/pdf/2611.25468v1)

**作者:** Trung-Khang Tran `[一作]` (National University of Singapore), Jonathan Scarlett `[通讯]` (National University of Singapore)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了概率非自适应组测试问题，改进了Kautz-Singleton构造中的常数，以提高COMP解码器和DD解码器的性能。

**💡 创新点**

创新点在于通过更精确的分析提高了COMP和DD解码器的领先常数，使得在给定条件下所需的测试次数减少。

**🔧 技术方法**

使用了概率分析和Reed-Solomon多项式的根的集中性分析技术。

**📊 数据集**

使用了Kautz-Singleton构造的测试矩阵，具体参数为d=Ω(log^2 N)，N=q^k，其中q为素数的幂。

**📈 对比分析**

与现有方法比较，COMP解码器的测试次数常数从4降低到1/ln2≈1.4427，性能显著提升，且与随机近恒定列权重设计的常数相匹配。

**⚠️ 局限性**

限制在于目前的分析未能处理一般的θ∈(0,1)的情况，且尚未确定Kautz-Singleton构造是否能达到随机设计的改进DD阈值。

---

## 146. Density Feedback Control for Transferability of Generative Sparse Adversarial Attacks

**arXiv ID:** 2611.25001 | [PDF](https://arxiv.org/pdf/2611.25001v1)

**作者:** Dongsu Song `[一作]` (Korea Aerospace University), Jay Hoon Jung `[通讯]` (Korea Aerospace University)

**通讯引用:** 67 | [OpenAlex ID](https://openalex.org/A5075144203)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的稀疏对抗攻击方法，称为密度反馈控制（DFC），通过根据观察到的硬掩码密度调整稀疏系数，以实现目标密度。

**💡 创新点**

创新点在于通过反馈机制动态调整稀疏系数，而不是在训练前固定，从而提高了攻击的成功率和转移性。

**🔧 技术方法**

使用了生成对抗网络（GAN）和反馈控制技术来训练生成器，以产生可转移的稀疏扰动。

**📊 数据集**

使用了ImageNet数据集进行训练和评估，源模型为ResNet-50，目标模型包括DenseNet-161、VGG-16等。

**📈 对比分析**

与传统的稀疏攻击方法（如PGD_0、sPGD、SparseFool等）相比，DFC显著提高了GPG和TSAA在CNN和Transformer目标上的转移攻击成功率，同时减少了平均扰动像素数。

**⚠️ 局限性**

限制在于该方法仅控制平均密度，未来的工作需要扩展到每个图像的密度控制。

---

## 147. QUBO-based feature selection for MRI undersampling pattern design

**arXiv ID:** 2611.25625 | [PDF](https://arxiv.org/pdf/2611.25625v1)

**作者:** Xiaoxin Huang `[一作]` (Tohoku University), Masayuki Ohzeki `[通讯]` (Tohoku University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种基于特征选择的方法来设计MRI欠采样掩模，通过优化选择的采样线来提高图像重建质量。

**💡 创新点**

创新点在于将MRI欠采样掩模设计视为特征子集选择问题，利用互信息（MI）和条件互信息（CMI）来评估采样线的选择，并通过量子退火和模拟退火方法优化掩模。

**🔧 技术方法**

使用了量子退火（QA）和模拟退火（SA）技术来解决二次无约束二进制优化（QUBO）模型。

**📊 数据集**

使用了fastMRI单线圈膝关节数据集，训练集包含25名患者的200个切片，测试集包含5名其他患者的60个切片。

**📈 对比分析**

与传统的采样方法（如VDS、Gaussian、Random和Equispaced）相比，QUBO-SA和QUBO-QA在所有六个目标采样率下均表现出更低的均方根误差（RMSE）和更高的峰值信噪比（PSNR）。

**⚠️ 局限性**

限制在于评估使用的数据集较小，且仅模拟了欠采样，未来研究需要扩展到更多患者和不同的采集条件，以验证结果的普适性。

---

## 148. Plan4D: Generative Plannable 4D Worlds

**arXiv ID:** 2611.26516 | [PDF](https://arxiv.org/pdf/2611.26516v1)

**作者:** Hangwei Zhang `[一作]` (Nanyang Technological University), Chuanxia Zheng `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种生成可规划的4D世界模型，能够根据观察到的场景上下文和文本指定的动作生成未来的视频及其对应的4D几何信息。

**💡 创新点**

创新点在于将几何先验与未来动态生成解耦，通过模型拼接架构将预训练的视频生成器与3D重建器结合，实现了高视觉保真度和几何一致性的生成。

**🔧 技术方法**

使用了模型拼接架构，结合了Depth Anything 3和LongCat-Video，并增加了用于场景流预测的辅助DPT头。

**📊 数据集**

使用了合成和真实世界的数据集，包括PointOdyssey、Dynamic Replica、Virtual KITTI 2等。

**📈 对比分析**

通过与现有的4D世界生成和几何生成基线进行比较，展示了在视觉一致性、动作文本对齐和4D结构的时间一致性方面的优越性能。

**⚠️ 局限性**

限制在于缺乏配对的动作和密集4D注释的数据集，导致训练数据稀缺。

---

## 149. Physics-Informed Feature Representations and Pretrained Tabular Foundation Models for Generalizable Melt Pool Geometry Prediction

**arXiv ID:** 2611.26235 | [PDF](https://arxiv.org/pdf/2611.26235v1)

**作者:** Rafid Enayet `[一作]` (University of South Carolina), Md Habibor Rahman `[通讯]` (University of South Carolina)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究开发了一种数据驱动框架，用于预测激光粉末床熔融（LPBF）中的熔池几何形状和分类熔化模式，比较了物理信息无量纲参数与传统过程参数的表现。

**💡 创新点**

创新点在于使用物理信息无量纲参数（如归一化焓和Péclet数）作为特征输入，展示了其在跨材料泛化中的优势，并与预训练的表格基础模型结合，显著提高了预测性能。

**🔧 技术方法**

采用了多种机器学习技术，包括高斯过程回归、支持向量机、随机森林、XGBoost，以及三种预训练的表格基础模型（TabPFN、TabDPT、TabICLv2）。

**📊 数据集**

使用了两个实验数据集：一个是316L不锈钢的单轨测量数据，另一个是IN625和IN718合金的LPBF实验数据，合并后形成了460个观测值的综合数据集。

**📈 对比分析**

通过五折分组交叉验证比较了不同模型的性能，结果显示无量纲表示在大多数条件下提高了预测性能，特别是在跨材料熔池深度预测中，模型的R²达到了0.96，预训练的表格基础模型在所有评估条件下表现优于传统机器学习基线。

**⚠️ 局限性**

限制在于归一化焓的计算使用了对所有材料和过程条件的常数激光吸收率假设，而实际吸收率会因表面状态和熔池状态而变化。此外，数据集仅限于两种合金，尚未验证对其他合金系统的普适性。

---

## 150. ScaleDiag: Diagnosing the Limits of Test-Time Scaling in GUI Grounding

**arXiv ID:** 2611.25297 | [PDF](https://arxiv.org/pdf/2611.25297v1)

**作者:** Chen Liu `[一作]` (University of Technology Sydney), Ling Chen `[通讯]` (University of Technology Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ScaleDiag框架，用于诊断GUI定位中的测试时间缩放限制，独立分析视觉输入、指令信息和预测采样对GUI定位的影响。

**💡 创新点**

创新点在于通过控制实验分离不同缩放策略的效果，揭示了视觉缩放、指令缩放和重复采样的局限性，并强调了在不同条件下的选择性测试时间缩放。

**🔧 技术方法**

使用了控制实验的方法，通过独立变化目标缩放、上下文、目标特定信息和辅助描述来评估不同的缩放策略。

**📊 数据集**

在五个GUI定位基准上进行了评估，包括ScreenSpot-Pro、UI-Vision、MMBench-GUI、OSWorld-G和ScreenSpot-v2。

**📈 对比分析**

与现有方法相比，ScaleDiag框架揭示了每种缩放策略的有效范围，发现视觉缩放受目标占用率的限制，指令缩放依赖于目标识别信息，而重复采样提高了候选覆盖率但未能有效选择正确坐标。

**⚠️ 局限性**

限制在于不同缩放策略的效果在不同模型和基准上表现不一，且在某些情况下，增加视觉焦点、丰富描述或额外样本并不总是带来线性收益。

---

## 151. Attention Drift Regularization with Adaptive Initialization for Domain-Robust Visual Recognition

**arXiv ID:** 2611.26234 | [PDF](https://arxiv.org/pdf/2611.26234v1)

**作者:** Shivangi Rai `[一作]` (Indian Institute of Science Education and Research Bhopal), Vinod K Kurmi `[通讯]` (Indian Institute of Science Education and Research Bhopal)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种源无关领域适应框架DGA-CAFC，旨在解决领域转移中的注意力漂移和领域间隙无关的源初始化问题。

**💡 创新点**

创新点在于引入了基于反事实的特征一致性模块（CAFC）和领域间隙感知源模型选择模块（DGASS），前者通过最小化注意力扰动下的特征差异来增强特征稳定性，后者通过目标预测熵选择最合适的源模型。

**🔧 技术方法**

使用了基于反事实的注意力机制和熵引导的源模型选择技术。

**📊 数据集**

在Office31、Office-Home和DomainNet40等数据集上进行了广泛的实验。

**📈 对比分析**

与现有的基线方法相比，DGA-CAFC在多个SFDA和DG基准测试中表现出一致的性能提升，尤其是在Office31和DomainNet40上，平均准确率分别达到91.65%和87.49%。

**⚠️ 局限性**

限制在于DGASS模块在领域通用化（DG）设置中不可用，因为没有目标数据可供选择源模型。

---

## 152. Generalist Meets Specialists: An Evolving Multimodal Agent with Persistent Tool Orchestration for Traffic Scenario Understanding

**arXiv ID:** 2611.25868 | [PDF](https://arxiv.org/pdf/2611.25868v1)

**作者:** Linjia Kang `[一作]` (China Telecom Artificial Intelligence Technology), Hao Sun `[通讯]` (China Telecom Artificial Intelligence Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为StarTV-Agent的框架，结合了策略学习和经验演化，以支持交通场景理解中的适应性推理和高效的工作流重用。

**💡 创新点**

创新点在于引入了角色敏感过程优化（RSPO）和弱点驱动演示学习（WDL），以及轻量级自适应工作流重用（LAWR），实现了慢速推理与快速工作流执行的有效结合。

**🔧 技术方法**

使用了角色敏感过程优化（RSPO）、弱点驱动演示学习（WDL）和轻量级自适应工作流重用（LAWR）等技术。

**📊 数据集**

在三个交通场景理解基准上进行评估，包括TUMTraf VideoQA、Traffic-VQA和RoadSceneVQA。

**📈 对比分析**

与其他方法相比，StarTV-Agent在三个基准上实现了最高8.0%的相对性能提升，同时减少了52.1%的输入令牌消耗，显示出其在性能和效率上的优势。

**⚠️ 局限性**

局限性在于工作流必须适用于新输入，执行反馈不一定反映答案的正确性，未来的工作将探索更丰富的反馈信号和更灵活的工作流适应性。

---

## 153. Side-Channel Analysis of the RP2350 Challenge 2

**arXiv ID:** 2611.25479 | [PDF](https://arxiv.org/pdf/2611.25479v1)

**作者:** Nikhil Tattikota `[一作]` (Oregon State University), Vincent Immler `[通讯]` (Oregon State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文对RP2350挑战2的侧信道分析进行了研究，展示了对AES实现的全面泄漏检测、明文依赖性和密钥提取。

**💡 创新点**

创新点在于揭示了挑战实现中的严重弱点，这些弱点与挑战作者的声明相矛盾，并且通过公开分享分析结果以激励进一步的公众审查。

**🔧 技术方法**

使用了侧信道分析（SCA）、基于表的掩码技术和低功耗测量技术。

**📊 数据集**

使用了RP2350微控制器的AES256实现作为数据集。

**📈 对比分析**

与其他方法相比，本文的方法在泄漏检测和密钥提取方面表现出色，成功揭示了AES实现中的多个弱点，尽管实现抵抗标准的单变量攻击，但仍未能通过基本的泄漏检测测试。

**⚠️ 局限性**

限制在于本文的评估是有限的，主要集中在基本的侧信道属性上，而不是全面的根本原因分析。

---

## 154. AI Deactivation Reduces Loneliness and Increases Social Interaction

**arXiv ID:** 2611.25263 | [PDF](https://arxiv.org/pdf/2611.25263v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 155. Training Vision Models for LLM-Driven Weight-Space Programming

**arXiv ID:** 2611.26408 | [PDF](https://arxiv.org/pdf/2611.26408v1)

**作者:** Timo Ropinski `[一作]` (Ulm University), Sebastian Kranz `[通讯]` (Ulm University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究展示了大型语言模型（LLMs）在视觉模型上进行权重空间编程的能力，评估了不同训练方法对编辑质量的影响。

**💡 创新点**

提出了一种新的训练方法——编辑准备训练（ERT），该方法通过模拟编辑过程来提高模型的可编辑性，并在多个基准测试中表现出色。

**🔧 技术方法**

使用了编辑准备训练（ERT）、标准经验风险最小化（ERM）、标签平滑（LS）和基于梯度的可编辑训练（ENN）等技术。

**📊 数据集**

使用了Waterbirds、CIFAR-100、PACS和Entity-30等数据集进行实验。

**📈 对比分析**

与标准训练、标签平滑和基于梯度的可编辑训练相比，ERT在所有评估基准中都取得了最高的平均净得分，尤其在难以编辑的设置中表现显著提升。

**⚠️ 局限性**

本研究的局限性在于只涵盖了图像分类任务，未来需要在其他视觉任务和更大模型上进行测试，同时编辑仍然会降低测试准确性。

---

## 156. RoCoKV: Accelerating LLM Inference Serving through Uncertainty and Redundancy Aware KV Cache Compression

**arXiv ID:** 2611.25433 | [PDF](https://arxiv.org/pdf/2611.25433v1)

**作者:** Payman Behnam `[一作]` (Georgia Institute of Technology), Alexey Tumanov `[通讯]` (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对大型推理模型的KV缓存压缩方法，旨在提高推理服务的效率。

**💡 创新点**

创新点在于结合了不确定性引导的自适应修剪和冗余感知的令牌选择机制，以更有效地利用注意力预算。

**🔧 技术方法**

使用了CUDA融合三阶段解码内核，集成到现有的推理框架中，优化了KV缓存的管理和访问。

**📊 数据集**

在三个推理基准和两个推理模型（DeepSeek-R1-Distill-Llama-8B和DeepSeek-R1-Distill-Qwen-7B）上进行了实验。

**📈 对比分析**

与多种最先进的KV缓存压缩方法进行了比较，结果显示该方法在紧张的KV缓存预算下，推理准确性显著提高，尤其在复杂数据集上表现优异，达到了1.27倍的速度提升和38%的吞吐量提高。

**⚠️ 局限性**

限制在于压缩方法的性能提升可能在生产环境中受到额外开销的影响，如评分、驱逐和内存管理等，可能抵消KV大小减少带来的好处。

---

## 157. MaskPE: One Mask, Two Roles: Weight Placement and MAC Gating in Systolic Arrays

**arXiv ID:** 2611.26432 | [PDF](https://arxiv.org/pdf/2611.26432v1)

**作者:** Shu-Lin Shih `[一作]` (University of Massachusetts), Sandip Kundu `[通讯]` (University of Massachusetts)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一种名为MaskPE的掩码导向的系统架构，利用位置掩码在预加载期间引导非零权重，并在执行期间对零权重处理单元的算术寄存器进行时钟门控，从而减少动态功耗，同时保持系统数据流。

**💡 创新点**

创新点在于使用单一位置掩码同时实现权重放置和算术活动控制，避免了传统方法中需要的索引匹配网络和负载平衡器，从而提高了存储效率和能量效率。

**🔧 技术方法**

采用了位图导向的权重压缩和选择性算术寄存器时钟门控技术，结合了银行存储和逐列流解码器。

**📊 数据集**

使用了七个大型语言模型（LLM）投影的样本数据集，特别是在50%稀疏度下进行评估。

**📈 对比分析**

与传统的密集权重-静态系统阵列进行比较，MaskPE在执行能量上降低了12.5%到14.4%，并在带宽受限的情况下实现了高达1.53倍的加速。

**⚠️ 局限性**

限制在于尽管MaskPE在能量和存储方面表现出色，但在零稀疏度情况下，额外的稀疏权重存储和控制开销可能会增加能量消耗。

---

## 158. SpecWM: Turning Verification into Refinement for Streaming World Models

**arXiv ID:** 2611.26211 | [PDF](https://arxiv.org/pdf/2611.26211v1)

**作者:** Jiaming Li `[一作]` (Huazhong University of Science and Technology), Zhiyuan Ma `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种统一的推理框架Speculative World Model (SpecWM)，将验证转化为精炼，以提高流媒体世界模型的生成效率和质量。

**💡 创新点**

创新点在于通过目标的潜在速度场进行验证，提供可重用的修正信号，从而减少对昂贵的重生生成的依赖，并实现过渡感知的精炼。

**🔧 技术方法**

使用了潜在速度场的评估技术，结合了一步和三步的精炼策略，以提高视觉质量和控制精度。

**📊 数据集**

在WBench Navigation和LongSpecBench数据集上进行了实验，评估生成质量、速度和持续控制能力。

**📈 对比分析**

与现有方法（如Target-only和Draft-only）相比，SpecWM在WBench Navigation上实现了1.63倍的速度提升，同时保持了较高的导航得分（NavScore从75.99提升至85.63），在质量和速度控制之间取得了最佳平衡。

**⚠️ 局限性**

限制在于SpecWM的性能依赖于阈值的选择和过渡感知策略的有效性，可能在某些情况下无法达到最佳效果。

---

## 159. Ground Truth Bias in AI-Assisted Medical Annotation: A Crossover Study Using an Independent Radiologist Reference Standard

**arXiv ID:** 2611.25924 | [PDF](https://arxiv.org/pdf/2611.25924v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 160. One Framework to Diagnose Them All: A Stratified Benchmark for Bengali LLM Hallucination

**arXiv ID:** 2611.26336 | [PDF](https://arxiv.org/pdf/2611.26336v1)

**作者:** Asif Shahriyar Sushmit `[一作]` (IEEE), Mohammad Mamun Or Rashid `[通讯]` (Jahangirnagar University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一种针对孟加拉语的幻觉检测基准，旨在通过整合不同的失败模式和任务类型来减少低资源语言中的数据集碎片化问题。

**💡 创新点**

创新点在于设计了一种跨文化距离、任务类型和幻觉分类的分层基准，能够在一个评估中同时提供诊断信息和性能评分。

**🔧 技术方法**

使用了零-shot大语言模型（LLM）作为评估者，并进行了开放竞赛以测试不同系统的性能。

**📊 数据集**

使用了两个阶段的数据集：第一阶段包含2,516个项目，第二阶段包含5,000个项目，涵盖了多种任务类型和幻觉模式。

**📈 对比分析**

通过比较99个第一阶段和32个第二阶段的参赛系统，发现大多数系统在第二阶段的表现下降，平均下降17.6分，而所有零-shot评估者的表现有所提升，表明第一阶段的排名无法有效预测第二阶段的表现。

**⚠️ 局限性**

限制在于理论创作的项目可能会确认自身的分类，且审计仅检查了样本，可能存在标签噪声。

---

## 161. Multiple Criteria Decision Making Under XORness

**arXiv ID:** 2611.25652 | [PDF](https://arxiv.org/pdf/2611.25652v1)

**作者:** Amin Hocine `[一作]` `[通讯]` (Excelia Business School), Amin Hocine (Excelia Business School)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种在XOR不确定性下进行多标准决策的方法框架，定义了XOR数并证明了其基本算术和代数性质。

**💡 创新点**

创新点在于引入了XOR数和XOR聚合算子，能够在决策过程中保持不确定性，并提出了一种新的性能-XORness指数来对替代方案进行排名。

**🔧 技术方法**

使用了XOR数的算术运算、XOR决策矩阵的构建、归一化技术和加法XOR聚合算子等技术。

**📊 数据集**

应用于能源系统中的网络安全风险评估，涉及五种网络威胁场景的数据集。

**📈 对比分析**

通过与传统的加权和方法进行比较，展示了XOR-WSM在处理不确定性方面的优势，能够有效地处理复杂的决策矩阵。

**⚠️ 局限性**

限制在于XOR数的定义和应用可能不适用于所有类型的不确定性，且在实际应用中可能需要进行敏感性分析以调整参数。

---

## 162. Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth

**arXiv ID:** 2611.26603 | [PDF](https://arxiv.org/pdf/2611.26603v1)

**作者:** Peiyun Hu `[一作]` (Apple), Vladlen Koltun `[通讯]` (Apple)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的方法Depth Pro 2，用于从单目图像和视频中估计清晰、度量一致的深度。

**💡 创新点**

创新点在于通过联合处理多个帧以其原始分辨率来稳定深度估计，同时保持边界的清晰度，克服了单帧方法和多帧方法之间的权衡。

**🔧 技术方法**

使用了一种多视图变换器架构，结合了自注意力机制和交叉帧注意力，能够灵活处理任意数量的帧。

**📊 数据集**

在19个不同的数据集上进行训练，重点关注合成的真实深度图，以确保模型的泛化能力。

**📈 对比分析**

与单图像和多帧方法进行比较，Depth Pro 2在边界清晰度、时间一致性和度量深度准确性方面均表现优异，设定了新的性能基准。

**⚠️ 局限性**

限制在于模型的大小和计算需求，尽管有不同参数量的模型可供选择，但在处理高分辨率视频时仍需较高的计算资源。

---

## 163. ToolPulse: Side-Channel Eavesdropping on Tool-Using LLM Agents

**arXiv ID:** 2611.25470 | [PDF](https://arxiv.org/pdf/2611.25470v1)

**作者:** Zihao Dan `[一作]` (University of Hong Kong), Shinan Liu `[通讯]` (University of Hong Kong)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种针对使用工具的LLM代理的缓存侧信道窃听框架，系统评估了Flush+Reload和Prime+Probe技术在监控代理工具调用中的有效性，并设计了工具恢复攻击以重建代理的工具执行工作流。

**💡 创新点**

创新点在于识别工具执行作为LLM代理的实际侧信道攻击面，并通过工具恢复攻击实现了对11个LLM模型的高效监控和重建，最低F1分数超过93%。

**🔧 技术方法**

使用了Flush+Reload和Prime+Probe两种缓存侧信道攻击技术，并设计了缓存行定位算法和去重算法以提高监控精度。

**📊 数据集**

使用了Codex和Claude Code中的11个LLM模型进行评估，数据集包括Terminal-Bench 1和2的任务类别。

**📈 对比分析**

与现有方法相比，Flush+Reload在监控工具调用时表现出高准确性和实时性，而Prime+Probe虽然支持跨VM攻击，但监控开销过大且存在缓存集别名问题。Flush+Reload的F1分数在所有评估模型中均超过93%。

**⚠️ 局限性**

限制在于Prime+Probe方法的高开销和缺乏实时恢复能力，且在多用户环境下可能受到其他用户活动的干扰，影响攻击效果。

---

## 164. LLM GerryMandering: Separating Interactions from Additive Effects in LLM Audits

**arXiv ID:** 2611.26538 | [PDF](https://arxiv.org/pdf/2611.26538v1)

**作者:** George Rahul `[一作]` (International Institute of Information Technology Hyderabad), Pushkar Shukla `[通讯]` (Cloudphysician)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的方法（Additive Intersectional Screen）来审计语言模型中的交叉偏见，特别是当个体同时拥有多个边缘身份时，模型如何处理这些身份的交互效应。

**💡 创新点**

创新点在于引入了加性模型来替代传统的单属性标准，从而更准确地识别交互效应，并通过Freedman-Lane null进行假设检验，显著降低了假阳性率。

**🔧 技术方法**

使用了加性模型和Freedman-Lane null方法来进行交叉偏见的检测和评估。

**📊 数据集**

构建了一个包含10,000个合成印度人设的数据集，涵盖六个身份属性（性别身份、年龄、性取向、社会经济地位、宗教和种姓）。

**📈 对比分析**

与传统的单属性标准相比，Additive Intersectional Screen在150个合成数据集上平均每个数据集仅标记0.09个假子组，而单属性标准则标记66.1个假子组。对于已植入的交互效应，前者能够恢复84%的交互，而后者仅能恢复32%。

**⚠️ 局限性**

限制在于所使用的人设是合成的，无法直接反映印度真实人口的偏见情况。此外，测试未经过Likert量表数据的校准，可能导致假阳性率上升。

---

## 165. A Model of Diverse Sampling from Language Models

**arXiv ID:** 2611.25003 | [PDF](https://arxiv.org/pdf/2611.25003v1)

**作者:** Manuel Prada-Corral `[一作]` (ETH Zurich), Tim Vieira `[通讯]` (ETH Zurich)

**通讯引用:** 1409 | [OpenAlex ID](https://openalex.org/A5052346902)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于确定性点过程（DPP）的语言模型输出多样性建模方法，旨在生成高质量且多样化的文本输出。

**💡 创新点**

创新点在于通过DPP定义字符串子集的概率分布，从而自然地编码多样性，避免生成相似的输出。

**🔧 技术方法**

使用了确定性点过程（DPP）和局部退火的祖先采样技术。

**📊 数据集**

使用了多个数据集，包括FineWeb、Ambrosia和GSM8K，进行不同任务的评估。

**📈 对比分析**

与温度采样、简单DPP、以及多种基线方法进行比较，结果显示该方法在质量-多样性和精确度-召回率方面均有显著提升。

**⚠️ 局限性**

限制在于该方法只能从候选池中选择输出，无法生成提案中未采样的输出，且最佳的全局温度在不同任务中可能不同。

---

## 166. φTD: Distributional Reinforcement Learning using Characteristic Functions

**arXiv ID:** 2611.25208 | [PDF](https://arxiv.org/pdf/2611.25208v1)

**作者:** Thomas Mousseau `[一作]` (Polytechnique Montreal), Amir-massoud Farahmand `[通讯]` (Mila Quebec Artificial Intelligence Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种新的分布式强化学习方法，通过在频域中计算损失来克服现有算法在分布表示上的限制。

**💡 创新点**

创新点在于通过频域的特征函数来统一现有的分类和分位数方法，并能够学习以前无法捕获的分布，如连续分布的混合。

**🔧 技术方法**

使用了频域特征函数和梯度更新的方法来优化返回分布。

**📊 数据集**

在合成基准和大规模深度强化学习环境中进行了实证研究。

**📈 对比分析**

与现有的分类和分位数方法相比，提出的方法在更广泛的参数族上表现出更低的近似误差和更好的整体性能。

**⚠️ 局限性**

限制在于虽然引入了更广泛的分布类，但这可能导致估计误差的增加，并且许多学习的参数族是非凸的，导致理论分析的困难。

---

## 167. ForeSeek: Foresight-Driven Reasoning for Long-Form Video Understanding

**arXiv ID:** 2611.26024 | [PDF](https://arxiv.org/pdf/2611.26024v1)

**作者:** Yuchen Sun `[一作]` (Tencent), Zheng Wei `[通讯]` (Tencent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为ForeSeek的前瞻性视频代理，旨在将长视频推理从反应式检索转变为预测性探索。

**💡 创新点**

创新点在于通过维持对未见时间区域的视觉基础期望，ForeSeek能够在收集证据的过程中不断调整和验证这些期望，从而提高推理效率和准确性。

**🔧 技术方法**

使用了前瞻性引导推理和两阶段训练框架，结合了监督学习和强化学习。

**📊 数据集**

在五个长视频基准上进行了广泛实验，包括VideoMME、LongVideoBench、LVBench、VideoHolmes和MLVU。

**📈 对比分析**

与现有的检索驱动代理相比，ForeSeek在效率-准确性权衡上表现出色，使用的帧数比最强的检索代理少2.5到4.4倍，同时在准确性上保持竞争力或优越性。

**⚠️ 局限性**

局限性在于，尽管ForeSeek在多个基准上表现良好，但在某些情况下仍可能受到视频长度和复杂性的影响，导致推理效率下降。

---

## 168. SEAF: Selective Semantic Enrichment and Visual Evidence Fusion for Multimodal Sarcasm Detection

**arXiv ID:** 2611.25672 | [PDF](https://arxiv.org/pdf/2611.25672v1)

**作者:** Faxian Wan `[一作]` (Northeastern University), Yifei Zhang `[通讯]` (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为SEAF的双分支框架，用于多模态讽刺检测，通过选择性语义丰富和视觉证据融合来提高检测性能。

**💡 创新点**

创新点在于引入了文本锚定平滑聚合（TASA），该方法根据图像与原始帖子文本的相似性加权图像贡献，同时保留低字面相似性的图像的潜在讽刺相关证据。

**🔧 技术方法**

使用了多模态大语言模型（MLLM）生成的中立实体背景和图像描述，结合光学字符识别（OCR）文本，构建了语义证据分支和对齐视觉证据分支。

**📊 数据集**

使用了MMSD3.0和MMSD2.0数据集进行实验，MMSD3.0包含每个帖子2到4张图像，MMSD2.0用于单图像适用性评估。

**📈 对比分析**

与其他方法相比，SEAF在MMSD3.0基准测试中表现最佳，宏F1得分达到86.17，超出CIRM方法3.63点，显示出其在多模态讽刺检测中的优越性。

**⚠️ 局限性**

局限性在于实验主要集中在MMSD3.0和MMSD2.0上，未来需要在其他语言、领域和文化特定参考上进行评估，同时生成的上下文的有效性依赖于生成内容的相关性和准确性。

---

## 169. WEVO: World Model-Enabled VLA Optimization for Whole-Body Loco-Manipulation

**arXiv ID:** 2611.25251 | [PDF](https://arxiv.org/pdf/2611.25251v1)

**作者:** Qianzhong Chen `[一作]` (Amazon), Roberto Martín-Martín `[通讯]` (Amazon)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种无遥控操作的框架，用于在动作条件的视频世界模型中对全身人形机器人进行视觉-语言-动作（VLA）后训练。

**💡 创新点**

首次构建了超越固定基础桌面操作的全身人形VLA的世界模型强化学习系统，结合了自我中心视频生成、全身动态和密集进展奖励。

**🔧 技术方法**

使用了生成视频世界模型（WM）、强化学习（RL）算法（如DICE-RL、ReinFlow和AWR）以及基于3D高斯点云重建的模拟器。

**📊 数据集**

使用了合成演示、模拟游戏数据和20条真实机器人轨迹的数据集，涵盖了三项真实机器人任务：瓶子拾取与放置、瓶子拾取-转动-放置和双手移动篮子。

**📈 对比分析**

与基线方法（如行为克隆和真实机器人RL）相比，视频WM RL的平均成功率从23.3%提高到75.0%，而真实机器人RL的成功率为48.3%。

**⚠️ 局限性**

局限性包括任务特定适应，Unitree G1特定接口限制了对其他人形机器人的零-shot迁移。未来的工作应关注跨身体转移、长时间多视角预测、轨迹生成效率和联合策略-WM适应。

---

## 170. ScaleWorld: High-Resolution World Generation through Shared Prediction and Refinement

**arXiv ID:** 2611.26615 | [PDF](https://arxiv.org/pdf/2611.26615v1)

**作者:** Yansong Qu `[一作]` (Xiamen University), Liujuan Cao `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种共享因果模型，用于摄像机控制的低分辨率预测和高分辨率细化，能够在一个模型中同时完成这两个任务。

**💡 创新点**

创新点在于通过共享骨干网络实现低分辨率预测和高分辨率细化的统一，同时保留独立可用的低分辨率输出路径。

**🔧 技术方法**

使用了因果视频模型和任务不对称的在线蒸馏（TA-OPD）技术。

**📊 数据集**

使用了多个数据集进行实验，包括HY-World 1.5、LingBot-World 2.0和Matrix-Game 3.5。

**📈 对比分析**

与单独优化的专家模型相比，统一模型在高分辨率生成质量上表现相当，同时在内存使用上具有优势，减少了峰值GPU内存。

**⚠️ 局限性**

限制在于模型在不同任务之间的梯度干扰，可能影响联合优化的效果。

---

## 171. Backtracking Agent Applications Security Incidents

**arXiv ID:** 2611.26525 | [PDF](https://arxiv.org/pdf/2611.26525v1)

**作者:** Yicheng Cai `[一作]` (Pennsylvania State University), Peng Liu `[通讯]` (Pennsylvania State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一种用于分析代理应用程序安全事件的回溯工具，旨在帮助分析师识别安全事件中的妥协入口点（CEP），并减少手动代码检查的工作量。

**💡 创新点**

创新点在于引入了一种新的中间抽象，通过交叉层依赖图连接代理功能层与应用程序代码执行层，并提供相应的查询语言，使分析师能够通过少量查询识别CEP。

**🔧 技术方法**

使用了交叉层依赖图（Cross-Layer Dependency Graph）和区域推断算法，结合静态污点分析技术。

**📊 数据集**

在五个流行的代理应用程序上进行了案例研究，涵盖了八个攻击事件。

**📈 对比分析**

与现有工具相比，该工具能够在64%到94%的范围内减少分析师的检查范围，成功识别所有事件的CEP，而现有工具未能提供相应的功能调用信息。

**⚠️ 局限性**

局限性在于需要开发者提供标签信息，且现有静态分析技术无法解决框架引起的间接调用目标解析问题。

---

## 172. A CPU Inside an LLM: Zero-Gradient Compilation of Exact Circuits into Pretrained Weights

**arXiv ID:** 2611.26450 | [PDF](https://arxiv.org/pdf/2611.26450v1)

**作者:** Tanmay Sah `[一作]` `[通讯]` (Independent Researcher), Tanmay Sah (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种闭式的零梯度电路嫁接原语，能够将确定性计算直接编译到预训练的语言模型权重中，而无需梯度微调或外部工具解释器。

**💡 创新点**

创新点在于通过零梯度电路嫁接实现了在不需要训练示例和运行时钩子的情况下，保持自然语言能力的同时，进行精确的多位数算术运算。

**🔧 技术方法**

使用了自动化的五阶段编译器，结合了梅林-GLU二层积分、交叉标记缩放消除、无记忆SiLU反对称移位等技术。

**📊 数据集**

使用了IBM Granite 4.2 3B和Alibaba Qwen 3.5 4B模型进行实验，数据集包括1200个6位数算术样本。

**📈 对比分析**

与其他方法相比，所提出的方法在6位数算术上达到了99.17%和100.00%的ALU前缀准确率，显著优于其他基线方法（如12.8K样本的42.2% / 61.7%）。

**⚠️ 局限性**

限制在于该方法在处理更复杂的算术运算（如除法）时仍然存在挑战，并且在输入长度和复杂性上有一定的限制。

---

## 173. TRACE-5G: Trust-Aware Runtime Attestation and Continuous Enforcement for 5G User Equipment

**arXiv ID:** 2611.25272 | [PDF](https://arxiv.org/pdf/2611.25272v1)

**作者:** Al Nahian Bin Emran `[一作]` (Mason Innovation Labs), Duminda Wijesekera `[通讯]` (Mason Innovation Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了TRACE-5G，一个持续验证用户设备（UE）运行时软件状态的框架，并利用信任状态控制网络访问。

**💡 创新点**

创新点在于将用户设备的信任生命周期分为多个阶段，持续验证运行时完整性，并将信任状态与网络访问权限相连接。

**🔧 技术方法**

使用了可信平台模块（TPM）、Linux完整性测量架构（IMA）和Keylime远程证明框架。

**📊 数据集**

在srsRAN/Open5GS测试平台上实现，使用了硬件TPM 2.0的用户设备和核心侧防火墙。

**📈 对比分析**

与现有方法相比，TRACE-5G在十次可信试验中，运行时信任和二级访问在9-12秒内建立；在十次不可信试验中，限制政策在不可信决策后5毫秒内应用，用户设备上下文释放在1.670秒内完成。

**⚠️ 局限性**

限制在于需要在每次连接时重新进行入网验证，且在不可信状态下的处理时间可能影响用户体验。

---

## 174. FoldSim: Scaling Cycle-Exact RTL Simulation with Hierarchical Folding and Static Scheduling

**arXiv ID:** 2611.26472 | [PDF](https://arxiv.org/pdf/2611.26472v1)

**作者:** Chengxuan Wang `[一作]` (University of California), Blaise Tine `[通讯]` (University of California)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

提出了一种名为FoldSim的自动化框架，通过层次折叠实现大规模RTL设计的周期精确仿真，能够在单个FPGA上运行。

**💡 创新点**

创新点在于通过层次折叠和静态调度来共享RTL逻辑，显著提高了仿真吞吐量，同时解决了大规模设计的内存需求问题。

**🔧 技术方法**

使用了层次折叠、静态调度、状态虚拟化和可配置内存建模等技术。

**📊 数据集**

使用了AMD Alveo V80 FPGA进行实验，模拟了8-SM GPU和1024节点的Rocket数据中心。

**📈 对比分析**

与传统的基于令牌的折叠方法相比，FoldSim在LUT使用上减少了13%到21%，并提高了时钟频率14%到21%。在FPGA上，FoldSim的吞吐量比FireSim高出约1.12到1.26倍。

**⚠️ 局限性**

限制在于状态虚拟化可能导致内存延迟增加，且在设计规模扩大时，仍需考虑FPGA的资源限制。

---

## 175. Native FPGA Hard-Block Implementation from High-Level Synthesis

**arXiv ID:** 2611.25281 | [PDF](https://arxiv.org/pdf/2611.25281v1)

**作者:** Mohamed Bouaziz `[一作]` (King Abdullah University of Science and Technology), Suhaib A. Fahmy `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

提出了一种C++模板库，能够在不修改高层综合（HLS）编译器的情况下，实现FPGA硬块的原生映射。

**💡 创新点**

创新点在于通过模板固定源模式，自动选择调度设置，并应用经过验证的RTL补丁，从而充分利用FPGA硬块的能力。

**🔧 技术方法**

使用C++模板库和自动映射流程，结合高层综合技术。

**📊 数据集**

在AMD Versal VC1902设备上进行评估，使用Vitis HLS和Vivado 2022.1进行实验。

**📈 对比分析**

与AutoSA的收敛阵列相比，矩阵乘法使用了2.4-4.5倍更少的查找表（LUT），在相等或更高的估计吞吐量下表现优越。

**⚠️ 局限性**

限制在于当前库的模板数量有限，未来工作将支持更多模板，并扩展到其他FPGA架构。

---

## 176. Dimension-Free Buffer Bounds for Compressed-Momentum Error Feedback

**arXiv ID:** 2611.25463 | [PDF](https://arxiv.org/pdf/2611.25463v1)

**作者:** Alireza Moayedikia `[一作]` `[通讯]` (Swinburne University of Technology), Alireza Moayedikia (Swinburne University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文分析了压缩动量误差反馈的内存缓冲区，通过压缩器收缩常数r/p来界定内存缓冲区的误差反馈，提出了一个新的结构属性，使得该常数不再影响缓冲区的界限。

**💡 创新点**

创新点在于提出了一种新的递归结构，使得压缩动量在下一个截断之前重新注入，从而消除了压缩器的收缩常数对缓冲区界限的影响。

**🔧 技术方法**

使用了压缩动量、低秩动量压缩、收缩压缩器和扰动迭代分析等技术。

**📊 数据集**

使用了BERT微调作为数据集，具体的参数设置为r=32和p=768。

**📈 对比分析**

与现有方法的比较显示，新的方法在缓冲区界限上不再依赖于维度和秩，性能上在BERT微调中反馈衰减成本低于0.008的验证F1。

**⚠️ 局限性**

限制在于，虽然提出的界限不依赖于维度和秩，但在某些情况下，累积泄漏的增长速度可能会比理论界限慢500倍，且在训练从零开始时可能会出现不同的行为。

---

## 177. PAIR-MED: Policy-reward Alternation for Iterative Refinement in Multilingual Medical Reasoning

**arXiv ID:** 2611.25446 | [PDF](https://arxiv.org/pdf/2611.25446v1)

**作者:** Fan Gao `[一作]` (University of Tokyo), Irene Li `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种政策-奖励交替训练框架，旨在共同适应医疗过程奖励模型（PRM）并优化多语言推理策略。

**💡 创新点**

创新点在于通过交替训练框架，使PRM能够适应不断变化的策略轨迹，从而提高多语言医疗推理的准确性和一致性。

**🔧 技术方法**

使用了政策-奖励交替训练框架，结合了医疗步骤监督、过程奖励、结果奖励和跨语言奖励。

**📊 数据集**

使用了包含106,644个医疗问题的Med-PRM训练集，以及翻译成12种语言的多语言医疗数据集。

**📈 对比分析**

与基线模型（如Qwen3-8B和Qwen3.5-9B）相比，模型在Global-MMLU和MMLU-ProX的平均准确率提高了3.01%到7.47%。在CUREMED-BENCH上，-9B在领域内和领域外语言的一致性和逻辑准确性指标上分别提高了12.88%和7.45%。

**⚠️ 局限性**

局限性包括使用最终答案的正确性作为弱步骤监督可能导致个别推理步骤的错误标记；依赖于翻译的英语医疗轨迹和英语教师的跨语言对齐可能限制了语言和临床覆盖范围；以及训练时间的增加可能限制了在固定计算预算下的可扩展性。

---

## 178. Gradient Boosting with Kolmogorov-Arnold Units for Interpretable Regression beyond Pairwise Interactions

**arXiv ID:** 2611.25942 | [PDF](https://arxiv.org/pdf/2611.25942v1)

**作者:** Minjong Cheon `[一作]` (Korea National Open University), Eunji Lee `[通讯]` (Korea Racing Authority)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的梯度提升方法，该方法使用单个Kolmogorov–Arnold单元作为弱学习器，能够处理三种或更多输入的联合依赖，并提供可解释的模型。

**💡 创新点**

创新点在于引入了一个闭合形式的单元拟合过程，能够精确地将模型分解为主效应、配对效应和高阶剩余部分，且能够学习高于二阶的交互作用。

**🔧 技术方法**

使用了梯度提升算法，结合了惩罚样条（P-splines）作为内外函数，并通过交替的惩罚最小二乘法进行单元拟合。

**📊 数据集**

使用了OpenML-CTR23基准数据集进行回归任务的评估，并在多个合成数据集上进行了测试。

**📈 对比分析**

与EBM模型进行比较，在合成目标的测试误差上，提出的方法在交互阶数为三到六的情况下，误差显著低于EBM；在OpenML-CTR23基准上，提出的方法在多个任务上表现出更低的测试RMSE。

**⚠️ 局限性**

限制在于模型的复杂性和计算成本，尤其是在处理高维数据时，可能需要较长的计算时间。

---

## 179. Multi-View Transformer with Geometry-Guided Multi-Modal ControlNet for 3D Virtual Try-On

**arXiv ID:** 2611.25979 | [PDF](https://arxiv.org/pdf/2611.25979v1)

**作者:** Sarthak Batra `[一作]` (Surrey), Adrian Hilton `[通讯]` (Surrey)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为GS-VTON-MVT的三阶段框架，用于3D虚拟试衣，旨在通过隐式和显式的3D推理实现一致的试衣效果。

**💡 创新点**

创新点在于引入了多视角修复变换器（MVT）来学习视角间的对应关系，使用几何引导的多模态ControlNet来增强编辑过程，并采用姿态自适应的注意力机制来优化参考视图的加权。

**🔧 技术方法**

使用了多视角修复变换器（MVT）、几何引导的多模态ControlNet和姿态自适应的注意力机制等技术。

**📊 数据集**

使用了3D-VTONBench数据集，该数据集包含60个被试在多种姿势和服装下的多视角捕捉。

**📈 对比分析**

与一般的3D编辑器和之前的3D试衣方法相比，GS-VTON-MVT在FID、KID、LPIPS等指标上表现更好，尤其在自我遮挡和复杂姿势下的表现提升显著。

**⚠️ 局限性**

限制在于目标服装的几何形状可能会受到原始服装的影响，渲染的深度和法线可能存在噪声，且全视角注意力的计算复杂度较高，限制了可处理的视角数量。

---

## 180. Retrieving Patristic References and Modelling Topics in Bullinger's Reformation Correspondence

**arXiv ID:** 2611.25247 | [PDF](https://arxiv.org/pdf/2611.25247v1)

**作者:** Lena Pickartz `[一作]` (TH Köln), Phillip B. Ströbel `[通讯]` (University Zurich)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了如何使用计算方法检测海因里希·布林格（Heinrich Bullinger）书信中的教父引用，并建模与神学思想相关的主题。

**💡 创新点**

创新点在于将密集检索和稀疏检索结合起来，以提高对显性和隐性教父引用的检测能力，同时引入基于元数据的主题建模方法。

**🔧 技术方法**

使用了密集检索（如双编码器）和稀疏检索（如BM25），并结合了FASTopic进行主题建模。

**📊 数据集**

使用的数据集包括海因里希·布林格的书信集（BDC）和教父文献集（PSC），前者包含11858封信件，后者包含52位教父的作品。

**📈 对比分析**

与BM25作为基线比较，密集检索在显性引用上表现优异（94% R@20），而BM25在隐性引用上也表现良好（16% R@20）。结合两者的排名提高了隐性引用的召回率（22%），接近交叉编码器的重排序（26%），但计算成本更低。

**⚠️ 局限性**

限制在于模型的可解释性不足，无法区分偶然相似性和有意引用，且评估数据集的规模较小（n=100），限制了统计稳健性。此外，主题建模结果的解释性和一致性也存在问题，主题权重分布过于均匀，难以用于检索。

---

## 181. Learning to Think with Images Faithfully via Adaptive Data Selection

**arXiv ID:** 2611.26199 | [PDF](https://arxiv.org/pdf/2611.26199v1)

**作者:** Yu Zhao `[一作]` (University of Edinburgh), Yiwen Guo `[通讯]` (LIGHTSPEED)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `e0540dec-d77f-42db-94ae-d039248f6393` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种新的训练方法，通过适应性数据选择和奖励机制，改善了视觉语言模型（VLMs）在使用工具时的准确性和可信度，尤其是在细粒度视觉细节的处理上。

**💡 创新点**

创新点在于通过调整训练数据以适应模型的能力变化，确保模型在回答问题时能够有效利用缩放的视觉证据，同时引入适应性奖励以鼓励在不需要缩放时的无工具回答。

**🔧 技术方法**

使用了数据合成管道来定位小物体，并生成关于其细节的可验证问题，同时结合了适应性奖励机制。

**📊 数据集**

使用了多个数据集，包括CC12M、SA-1B、InfographicVQA等，构建了13000个视觉搜索问题，涵盖单对象问题、空间关系问题和不可回答问题。

**📈 对比分析**

与现有的TwI模型（如DeepEyes、PixelReasoner和CodeV）进行比较，实验结果显示，本文的方法在视觉搜索基准上取得了显著的性能提升，准确率分别提高了11.6%（V*）、7.9%（HR-Bench 4K）和9.4%（HR-Bench 8K）。

**⚠️ 局限性**

限制在于模型的训练仍然依赖于合成数据的质量和多样性，且在某些情况下，模型可能仍然会出现不可信的工具使用行为。

---

## 182. Formally Verified Semidefinite Programming

**arXiv ID:** 2611.25174 | [PDF](https://arxiv.org/pdf/2611.25174v1)

**作者:** Perry Hart `[一作]` `[通讯]` (Carnegie Mellon University), Perry Hart (Carnegie Mellon University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种经过形式验证的Rust实现的第一阶半正定规划求解器，并讨论了其算法、实现及在Verus中的正确性证明。

**💡 创新点**

这是首个经过验证的半正定规划求解器实现，展示了如何在Rust程序中进行形式验证。

**🔧 技术方法**

使用Rust编程语言实现，并利用Verus自动定理证明器进行形式验证。

**📊 数据集**

未具体提及使用的数据集，但涉及到半正定规划的标准形式问题。

**📈 对比分析**

与传统的半正定规划求解器相比，提出的求解器在准确性和性能上进行了比较，确保了返回的解满足KKT条件，具有小的残差。

**⚠️ 局限性**

未尝试正式验证求解器的收敛性，且在浮点计算的处理上存在一定的复杂性。

---

## 183. EgoPass-Bench: Identifying Objects That May Affect the Wearer's Passage in Egocentric Video

**arXiv ID:** 2611.24995 | [PDF](https://arxiv.org/pdf/2611.24995v1)

**作者:** Hongyu Wang `[一作]` (University of California), Hua Shen `[通讯]` (New York University)

**通讯引用:** 897 | [OpenAlex ID](https://openalex.org/A5074317537)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个新的基准，评估在真实世界自我中心视频中观察者相对交互的理解。

**💡 创新点**

创新点在于引入了一个新的基准，专注于观察者与周围实体之间的动态交互，而不仅仅是运动感知和动作可行性。

**🔧 技术方法**

使用了多模态基础模型（MFM），特别是GPT-6 Astra模型进行评估。

**📊 数据集**

使用了284个手动审核的评估片段，这些片段是通过Ray-Ban Meta（第二代）智能眼镜录制的，涵盖了多种场景。

**📈 对比分析**

与八个基于帧的模型和六个原生视频配置进行了比较，发现运动感知与交互理解之间存在显著差异。GPT-6 Astra在运动识别上正确率为91.37%，但在通行约束变化的识别上仅为66.50%。

**⚠️ 局限性**

限制在于模型在识别相关实体时，常常包含那些既不构成物理冲突也不限制通行的实体，表明准确的运动识别并不确保可靠的相关实体发现。

---

## 184. Exposure-Aware Informative Path Planning Using Deep Reinforcement Learning

**arXiv ID:** 2611.26302 | [PDF](https://arxiv.org/pdf/2611.26302v1)

**作者:** Muzaffar Qureshi `[一作]` (University of Florida), Rushikesh Kamalapurkar `[通讯]` (University of Florida)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文解决了有限预算的暴露感知信息路径规划（EA-IPP）问题，旨在通过移动传感器群体在限制高强度场暴露的情况下绘制未知标量场。

**💡 创新点**

创新点在于将暴露感知信息路径规划形式化为一个标量化的双目标信念状态马尔可夫决策过程，平衡高斯过程信息增益与测量导出的舰队暴露指数。

**🔧 技术方法**

使用了参数共享的近端策略优化（PPO）框架，结合去中心化的行动者和中心化的舰队评论员进行离线训练。

**📊 数据集**

在雷达、烟羽和野火等未见场景上进行了评估，使用了模拟生成的标量场数据集。

**📈 对比分析**

与现有基线方法相比，EA-IPP代理在评估的基线中实现了最低的高斯过程重建均方根误差（RMSE），同时保持了D-最优信息在非学习信息基线的10%以内，暴露量分别降低了约34.6%、20.1%和27%。

**⚠️ 局限性**

限制在于该方法需要离线训练，并且在实际部署时可能面临未见场景的挑战。

---

## 185. PILAPS: A Bilingual Benchmark for Evaluating Pragmatic Inference across Phenomena and Settings

**arXiv ID:** 2611.25459 | [PDF](https://arxiv.org/pdf/2611.25459v1)

**作者:** Jiani Cao `[一作]` (University of Cambridge), Chen Cecilia Liu `[通讯]` (University of Cambridge)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文介绍了PILAPS，一个涵盖英语和普通话的双语基准，评估八种语用现象，涉及目的性写作和对话衍生材料。

**💡 创新点**

创新点在于开发了一个文化适应协议，以便进行跨语言的语用评估，并展示了语用表现并不在所有评估设置中普遍适用。

**🔧 技术方法**

使用了多种大型语言模型（LLMs），并采用了不同的提示策略进行评估，包括基础提示、链式思维提示和理论心智提示。

**📊 数据集**

使用的数据集包括1886个多项选择题，涵盖八种语用现象，分为构建组件和对话衍生组件，涉及英语和普通话。

**📈 对比分析**

与人类参与者相比，模型在对话上下文中更倾向于字面解释，且在扩展对话项目中的表现普遍低于人类基线，尤其在幽默和间接言语方面表现较差。

**⚠️ 局限性**

限制在于只涵盖英语和普通话这两种高资源语言，无法确定其对低资源语言的推广性；扩展对话项目来自单一电视系列，可能存在节目特定的变异。

---

## 186. SailSearch: Agent-Assisted Discovery in Combinatorial Optimization

**arXiv ID:** 2611.25087 | [PDF](https://arxiv.org/pdf/2611.25087v1)

**作者:** Dhriti Verma `[一作]` (IBM Research), Jayant Kalagnanam `[通讯]` (IBM Research)

**通讯引用:** 17572 | [OpenAlex ID](https://openalex.org/A5057936833)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种人类引导的算法研究工作流程，使用两个语言模型代理，针对市场分割和最大独立集（MIS）问题进行了研究。

**💡 创新点**

创新点在于结合人类引导的工作流程与两个语言模型代理，解决了两个具有挑战性的组合优化问题，并首次报告了CPU-only解决方案。

**🔧 技术方法**

使用了随机化算法和几何枚举结合的基于格的搜索算法。

**📊 数据集**

使用了QOBLIB基准数据集，涉及156个市场分割实例和多个MIS实例。

**📈 对比分析**

与其他方法相比，所提出的算法在解决市场分割问题时，成功解决了130到133个实例，首次报告了18个之前未解决的实例的CPU-only解决方案，性能优于CPLEX和其他基准。

**⚠️ 局限性**

限制在于该研究仅基于一个人类研究者的视角，未能建立人类指导、第二代理或持久记忆的因果效应。

---

## 187. Contactless Estimation of Respiratory Frequency Using 3D-CNN on Thermal Images

**arXiv ID:** 2611.26367 | [PDF](https://arxiv.org/pdf/2611.26367v1)

**作者:** Federica Gioia `[一作]` (University of Pisa), Marco Forgione `[通讯]` (Dalle Molle Institute for Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了一种基于深度学习的方法，通过热视频直接估计呼吸频率，消除了复杂的预处理和感兴趣区域（ROI）跟踪的需求。

**💡 创新点**

创新点在于首次采用端到端的深度学习方法直接从热视频中提取呼吸信号，简化了传统方法的工作流程。

**🔧 技术方法**

使用了3D卷积神经网络（3D-CNN）来分析热视频并预测呼吸频率。

**📊 数据集**

使用了从健康志愿者收集的热视频数据集，并结合合成数据集进行训练，以应对小数据集的挑战。

**📈 对比分析**

通过Leave-Subject-Out策略进行模型验证，结果显示在预处理和原始热视频上，验证R^2分数约为0.61，表明该方法在不同场景下具有良好的性能。

**⚠️ 局限性**

限制在于合成数据可能无法完全捕捉真实世界呼吸模式的复杂性，且实验未考虑身体活动，限制了呼吸频率的范围。

---

## 188. Exploring Semantic Table Search in Data Lakes

**arXiv ID:** 2611.25691 | [PDF](https://arxiv.org/pdf/2611.25691v1)

**作者:** Martin Pekár Christensen `[一作]` (Aalborg University), Katja Hose `[通讯]` (Technische Universität Wien)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种结合关键词搜索和语义表搜索的方法，以发现数据湖中相关的表格。该方法利用知识图谱中的实体信息来增强搜索效果。

**💡 创新点**

创新点在于提出了一种新的多标准排名系统，将语义搜索和关键词搜索的排名信号结合起来，从而提高数据发现的效果。

**🔧 技术方法**

使用了TOPSIS等无监督和监督技术来结合不同的排名信号，并提出了一种基于HNSW的预过滤技术来提高语义搜索的可扩展性。

**📊 数据集**

使用了来自维基百科的238,038和457,714个表格的数据集进行评估，分别来自2015年和2019年。

**📈 对比分析**

与传统的关键词搜索和其他表搜索方法相比，提出的方法在检索相关表格的准确性和效率上表现出色，尤其是在处理小规模查询时，HNSW显著提高了运行时间，平均提升了4.7倍。

**⚠️ 局限性**

限制在于该方法依赖于知识图谱的覆盖率，若知识图谱对数据湖的覆盖不全，可能会影响搜索的效果。此外，算法的复杂性在处理大规模数据时可能会导致性能下降。

---

## 189. Compact MaxSAT Encodings for Balanced Waiting in B2B Meeting Scheduling

**arXiv ID:** 2611.25533 | [PDF](https://arxiv.org/pdf/2611.25533v1)

**作者:** Tuyen Van Kieu `[一作]`, Khanh Van To `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了如何在B2B会议调度中减少最大和总等待时间，同时保持最佳的空闲范围平衡，并提出了紧凑的MaxSAT编码方法。

**💡 创新点**

创新点在于通过最小化最大空闲时间和总空闲时间来区分具有相同最小空闲范围的调度，同时保持空闲范围的优先级。

**🔧 技术方法**

使用加权最大可满足性（MaxSAT）进行词典优化，首先优化空闲范围，其次优化最大空闲时间，最后优化总空闲时间。

**📊 数据集**

使用了226个变体的20个基础实例的数据集进行实验。

**📈 对比分析**

与Direct编码进行比较，Compact-Objective在所有九个目标-数据集比较中具有最低的中位数子句计数和较低的中位数峰值内存。运行时间排名在相同求解器构建和计算环境下有所变化。

**⚠️ 局限性**

限制在于实验仅涵盖了20个基础实例的变体，未来的研究可以考虑更广泛的会议负载变化。

---

## 190. Approximate Adjoint Models for Memory and Compute-Efficient Online Learning of Hybrid Models

**arXiv ID:** 2611.26095 | [PDF](https://arxiv.org/pdf/2611.26095v1)

**作者:** Emilio González-Zamora `[一作]` (IMT Atlantique), Said Ouala `[通讯]` (IMT Atlantique)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种新的梯度近似方法，用于通过数值求解器进行反向传播，以降低在线学习的计算成本，同时保持足够的梯度准确性以进行优化。

**💡 创新点**

创新点在于保留了整个轨迹上的伴随递归，同时仅在稀疏的锚点处计算确切的拉回操作，从而减少了反向传播的内存和计算需求。

**🔧 技术方法**

使用了反向模式自动微分（AD）和梯度近似方法，包括零阶近似（Zero-order）和拉回重用（Pullback Reuse）。

**📊 数据集**

在三个复杂度逐渐增加的系统上进行了评估，包括不完整的Lorenz-63模型、二维准地转湍流的大涡模拟和多层准地转海洋模型的偏差校正。

**📈 对比分析**

与精确的反向传播（BPTT）方法相比，提出的方法在内存使用上减少了一个数量级，运行时间提高了四倍，且拉回重用方法在梯度方向上与精确梯度保持最接近。

**⚠️ 局限性**

限制在于近似梯度虽然足够用于优化，但不再是精确的梯度，可能会影响某些情况下的优化稳定性。

---

## 191. Suppressing the Shortcut Hidden Behind the Gains of VideoLLM Fine-Tuning

**arXiv ID:** 2611.25776 | [PDF](https://arxiv.org/pdf/2611.25776v1)

**作者:** Youngwoo Shin `[一作]` (Korea Advanced Institute of Science and Technology), Junmo Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为视频证据训练（VETO）的方法，以抑制视频大型语言模型（VideoLLM）在监督微调（SFT）过程中学习到的文本快捷方式，从而增强模型对视频内容的理解。

**💡 创新点**

VETO通过正交投影干预微调更新，抑制了模型对文本的依赖，使其能够更好地基于视频内容进行推理，避免了在训练分布外的性能下降。

**🔧 技术方法**

使用了正交投影技术来校准文本输入的激活，并将微调更新投影到其正交补空间。

**📊 数据集**

使用了Something-Something v2（SSv2）数据集进行训练，并在多个未见基准（如Perception Test）上进行评估。

**📈 对比分析**

与传统的SFT方法相比，VETO在训练分布上保持了相似的准确性，同时在分布转移下表现更好，抑制了文本快捷方式的形成，提升了模型的泛化能力。

**⚠️ 局限性**

VETO的局限性在于其分析主要集中在多选视频问答任务上，未来需要扩展到自由形式回答的任务，并探索在其他适配器上的应用。

---

## 192. SenseAlign: Learning Decision-Aligned Predictive Representations for Active Urban Sensing with UAVs

**arXiv ID:** 2611.26109 | [PDF](https://arxiv.org/pdf/2611.26109v1)

**作者:** Weijiang Xiong `[一作]` (EPFL), Nikolas Geroliminis `[通讯]` (EPFL)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究提出了SenseAlign，一个基于强化学习的框架，用于协调城市传感无人机队伍，以主动监测动态道路网络并收集关键信息。

**💡 创新点**

创新点在于结合了未来交通预测、知识蒸馏和决策对齐，形成了一个共享的潜在空间，从而提高了无人机的监测决策能力。

**🔧 技术方法**

使用了强化学习（RL）技术，特别是近端策略优化（PPO）算法，并结合了图神经网络（GNN）进行特征编码。

**📊 数据集**

使用了真实的巴塞罗那中央道路网络数据集，包含1570个路段，并进行了多次实验以验证方法的有效性。

**📈 对比分析**

与基线方法（如PPO和其他多智能体学习方法）相比，SenseAlign在拥堵变化监测和宏观基本图（MFD）估计上均表现出色，拥堵监测提高了8.1%，MFD估计误差从6.43%降低到5.70%。

**⚠️ 局限性**

限制在于无人机队伍的空间覆盖率仅为5%，在实际应用中可能无法实现完全的空间覆盖，且模型的性能依赖于训练数据的质量和多样性。

---

## 193. Thinflow: Organizing HPC Benchmarking Campaigns as Rerunnable Directory Trees

**arXiv ID:** 2611.26526 | [PDF](https://arxiv.org/pdf/2611.26526v1)

**作者:** Richard Schulze `[一作]` (University of Munster), Jonas Posner `[通讯]` (Fulda University of Applied Sciences)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种名为Thinflow的小型Bash引擎，用于组织高性能计算（HPC）基准测试活动，能够将任务和运行组织为可重现的目录树。

**💡 创新点**

Thinflow通过声明依赖关系推导执行顺序，并生成一个纯文本清单，使得基准测试活动可以在不同超级计算机之间轻松迁移和重现。

**🔧 技术方法**

使用Bash脚本实现，支持每个超级计算机的配置文件和每次运行的来源记录，且无需单独安装工具。

**📊 数据集**

在两个超级计算机的三个Slurm分区上进行了案例研究，涉及37个任务文件和455次基准测试运行。

**📈 对比分析**

与现有的基准测试框架（如JUBE、ReFrame和Benchpark）相比，Thinflow提供了一种更简单的结构，允许用户使用普通的Shell文件进行基准测试活动，性能表现良好，能够有效记录每次运行的环境和参数。

**⚠️ 局限性**

Thinflow不支持在执行过程中动态改变任务图形，且未能处理软件环境的等效性，迁移到其他超级计算机时可能面临编译器和库不兼容的问题。

---

## 194. RouterWeave: A Unified Observation Layer for Cross-Platform Activity Analysis via Home Routers

**arXiv ID:** 2611.26134 | [PDF](https://arxiv.org/pdf/2611.26134v1)

**作者:** Chengeng Liu `[一作]` (University of Hong Kong), Shinan Liu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于路由器的活动感知框架，将加密网络流量转化为跨平台的统一活动历史记录，以实现对个人和物联网设备的活动监测。

**💡 创新点**

首次实现了路由器本地系统，将加密网络流量转化为统一的、可供应用使用的活动历史记录，支持跨平台的活动监测。

**🔧 技术方法**

使用了基于路由器的网络流量收集、任务自适应识别和统一活动管理等技术。

**📊 数据集**

使用了四个数据集进行评估，包括移动应用活动数据集和物联网事件数据集，涵盖了Android、iOS和Windows等多个平台。

**📈 对比分析**

与基线方法相比，框架在应用活动识别上达到了94.41%的宏F1分数，在物联网操作识别上达到了97.46%，超出基线最多20.99个百分点，同时在资源消耗上表现优异，支持54个并发设备。

**⚠️ 局限性**

局限性在于，框架的活动识别精度依赖于网络流量的可观察性，某些细粒度操作可能产生相似的流量模式，导致难以区分。此外，扩展到新设备和活动的过程可能需要大量标注数据和训练模型。

---

## 195. A Nonadaptive Sparse Fourier Transform in Every Dimension

**arXiv ID:** 2611.25334 | [PDF](https://arxiv.org/pdf/2611.25334v1)

**作者:** Mahdi Cheraghchi `[一作]` (University of Michigan), Jingyi Lyu `[通讯]` (University of Michigan)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种随机化算法，通过非自适应查询构建k稀疏估计，能够以高概率满足特定的误差界限。

**💡 创新点**

算法在不需要提供尾部估计的情况下，仍然能够保证恢复的稀疏性和误差界限，且可以扩展到任意维度的傅里叶变换。

**🔧 技术方法**

使用了随机化算法和非自适应查询技术，结合了压缩感知的理论。

**📊 数据集**

在G=(/2^a)^d的群上进行实验，具体数据集未明确给出，但涉及到任意复杂信号的傅里叶变换。

**📈 对比分析**

与现有方法相比，算法在样本数量和运行时间上具有显著的改进，能够在多项式时间内恢复稀疏信号，且在样本复杂度上达到O(k/^2(log N)log(k/))。

**⚠️ 局限性**

算法的常数未经过优化，且在实际应用中可能会受到性能限制，尤其是在处理大规模信号时。

---

## 196. Live Migration of Gated DeltaNet Requests: Exact Catch-Up and the Cost of Log Backlog

**arXiv ID:** 2611.26153 | [PDF](https://arxiv.org/pdf/2611.26153v1)

**作者:** Hanqing Yu `[一作]`, Yongbo Yu `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

论文研究了流媒体Gated DeltaNet (GDN)请求的实时迁移，重点在于在相同模型权重的工作节点上恢复递归状态，以确保用户请求的连续性。

**💡 创新点**

创新点在于设计了四种GDN迁移路径，包括背景快照、32步批量重放和基于令牌的赶上机制，并提出了一种紧凑的更新日志迁移路径。

**🔧 技术方法**

使用了快照、批量重放和令牌赶上等技术，结合了在线迁移路径的设计。

**📊 数据集**

使用了在两个GPU部署上进行的1350次纯GDN试验和40个历史KV对齐的Qwen配置。

**📈 对比分析**

通过控制实验比较了不同迁移路径的性能，结果显示紧凑日志在所有90种配置中实现了最短的精确交接间隔，32步重放在允许算术差异时提供了最高的输出速率。

**⚠️ 局限性**

限制在于仅覆盖了单个请求和两个小模型检查点，且不同架构和后端实现的比较可能影响结果的统计精度。

---

## 197. AgentRoboCritic: Understanding the Capability of Frontier Agents on Direct Robot Control

**arXiv ID:** 2611.26613 | [PDF](https://arxiv.org/pdf/2611.26613v1)

**作者:** Yiyang Du `[一作]` (Carnegie Mellon University), Chenyan Xiong `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文对前沿多模态语言代理在直接机器人控制中的能力和局限性进行了系统的动作级分析，提出了一个意图驱动的批评框架AgentRoboCritic。

**💡 创新点**

创新点在于引入了意图驱动的动作批评框架，使得每个动作的评估不仅基于全局任务目标，还考虑了局部意图，从而提供了更细致的反馈。

**🔧 技术方法**

使用了意图驱动的动作批评框架，结合GPT-6 Astra作为批评者进行动作级分析。

**📊 数据集**

使用了12个机器人操作基准的400个动作的人类标注数据集，以及12844个标注动作的开放源代码数据集。

**📈 对比分析**

与现有的视觉-语言-动作模型（VLA）和世界-动作模型（WAM）进行了比较，前沿代理在零-shot操作中表现出色，成功率显著高于这些模型，且能够在没有任务特定适应的情况下完成复杂操作。

**⚠️ 局限性**

局限性包括在任务相关的物理执行瓶颈（如抓取和对齐）方面的不足，以及在不同失败模式下的恢复能力不均衡。

---

## 198. Beam Model Technical Report

**arXiv ID:** 2611.26358 | [PDF](https://arxiv.org/pdf/2611.26358v1)

**作者:** Reflection Team `[一作]` (Reflection AI), Tina Zhu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文介绍了一种名为Beam的稀疏混合专家（MoE）模型，具有501.3B的总参数和23.4B的活跃参数，专门为编码、推理和代理工作负载而优化。该模型在23.8T多样化的高质量标记上进行预训练，并通过一种新颖的两阶段中训练阶段实现了1M标记的上下文窗口。

**💡 创新点**

Beam在推理能力上达到了前沿水平，同时在推理计算需求上比同类模型减少了3-4倍。其创新点在于采用了严格的数据策划和控制扩展，结合深度硬件模型协同设计和FP8/NVFP4量化技术，最大化了服务吞吐量。

**🔧 技术方法**

使用了稀疏混合专家（MoE）架构，结合了深度学习中的多种技术，如RMSNorm、GQA（组查询注意力）、旋转嵌入、SwiGLU激活函数等，采用了强化学习（RL）进行后训练。

**📊 数据集**

使用了23.8T的多样化高质量标记数据集，包括STEM、数学和代码标记，经过新颖的领域特定策划管道和互联网规模的OCR处理。

**📈 对比分析**

与其他方法相比，Beam在评估套件中的能力随着RL计算的增加而持续提高，没有出现平台效应。Beam在有害请求的拒绝/转移率上达到了88.9%，在无害请求的合规率上达到了81.5%，在开放权重模型中达到了最高的拒绝/转移率。

**⚠️ 局限性**

限制在于Beam在四个系统风险类别（生物武器、化学武器、失控和网络攻击）中被评估为有限，意味着模型没有对攻防平衡产生实质性变化，也没有相对于可比的开放权重模型扩展威胁景观。

---

## 199. On the semantics of conditional branching

**arXiv ID:** 2611.26400 | [PDF](https://arxiv.org/pdf/2611.26400v1)

**作者:** Fredrik Dahlqvist `[一作]` `[通讯]` (Queen Mary University London), Fredrik Dahlqvist (Queen Mary University London)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文回顾了概率和量子程序中条件分支的不同语义，并提出了一种统一的范畴结构来解释这些语义。

**💡 创新点**

创新点在于提出了一种新的范畴结构，使得可以在多种计算范式（经典、概率、量子等）中统一解释条件分支的语义。

**🔧 技术方法**

使用了范畴论的技术，特别是涉及到的代数结构和分支同态的研究。

**📊 数据集**

没有具体提到使用的数据集，但讨论了多种计算范式的语义，包括经典、概率和量子计算。

**📈 对比分析**

通过比较不同的条件分支解释，展示了它们之间的不可比较性，并提出了统一的结构来解释这些分支，性能上提供了更广泛的适用性。

**⚠️ 局限性**

限制在于不同的条件分支解释之间存在不可比较性，且在某些情况下，可能无法找到统一的结构来同时处理所有类型的分支。

---

## 200. TacInno: Teaching Vision-Language-Action Models the Contact Dynamics Beyond Vision and Action

**arXiv ID:** 2611.26405 | [PDF](https://arxiv.org/pdf/2611.26405v1)

**作者:** Weile Li `[一作]` (ShanghaiTech University), Zhibo Pang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的视觉-语言-动作（VLA）模型，旨在通过隔离视觉和执行动作无法解释的接触动态（称为触觉创新）来改进触觉输入的使用。

**💡 创新点**

创新点在于引入了触觉创新的概念，并通过条件多帧编码器将其编码为离散代码，仅作为训练目标，而在部署时不添加额外模块。

**🔧 技术方法**

使用了条件多帧编码器（TacInno tokenizer）来压缩触觉变化，并通过辅助目标训练VLA模型。

**📊 数据集**

使用了六个来自UniVTAC基准的仿真任务和四个真实世界任务的数据集，涉及机器人操作。

**📈 对比分析**

与五个基线模型进行比较，模型在六个仿真任务上的平均成功率为87.5%，在四个真实任务上的成功率为70.5%，分别比最强基线高出10.7和9.0个百分点。

**⚠️ 局限性**

限制在于触觉信号的冗余性，部分触觉变化可以通过视觉观察和执行动作预测，因此在某些情况下，触觉输入的增加可能不会带来性能提升。

---

## 201. AI agents can redesign engineering parts with more than a thousand CAD parameters and nonlinear simulations

**arXiv ID:** 2611.26003 | [PDF](https://arxiv.org/pdf/2611.26003v1)

**作者:** Alexandra Starostina `[一作]` (Helmholtz-Zentrum Hereon), Christian C. Cyron `[通讯]` (Hamburg University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本研究探讨了大型语言模型（LLM）在非线性三维部件设计中的应用，特别是通过对镁支架的CAD模型进行约1650个参数的调整，在15次模拟预算内实现了设计优化。

**💡 创新点**

创新点在于首次展示了LLM能够在有限的模拟预算内，通过结合自身知识与模拟结果，进行大规模的设计重构，并显著降低了支架的应变目标。

**🔧 技术方法**

使用了大型语言模型（GPT-6 Astra）和有限元分析技术，结合CAD几何参数化进行设计优化。

**📊 数据集**

研究中使用的数据集包括镁支架的CAD模型，涉及约1650个几何参数，以及模拟结果数据。

**📈 对比分析**

与工业形状优化器Tosca进行比较，LLM在相同的模拟预算下，优化效果显著优于Tosca，LLM的设计在应变目标上降低了32%，而Tosca仅改善了10%。

**⚠️ 局限性**

主要限制在于LLM只能通过预先编写的几何参数化来改变CAD模型，且模拟设置仍需人工编写，此外，研究仅在特定的非线性三维部件上进行，是否适用于其他部件尚待验证。

---

## 202. Bio-PM-Guided Synthetic IMU Generation

**arXiv ID:** 2611.26607 | [PDF](https://arxiv.org/pdf/2611.26607v1)

**作者:** Deepali Singh `[一作]` (University of Massachusetts), Aviral Gupta `[通讯]` (University of Massachusetts)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种利用预训练生物信号编码器Bio-PM生成合成惯性测量单元（IMU）数据的方法，以增强人类活动识别（HAR）模型的泛化能力。

**💡 创新点**

创新点在于开发了两种类条件生成解码器（CAR-IMU和CD-IMU），其中CD-IMU通过去噪扩散模型直接学习完整的令牌分布，消除了对后期校准的依赖。

**🔧 技术方法**

使用了自回归变换器解码器和去噪扩散解码器，结合了特征调制（FiLM）技术。

**📊 数据集**

使用了WISDM数据集，该数据集包含51名受试者在进行六种活动（步行、慢跑、上楼、下楼、坐着和站立）时的IMU数据。

**📈 对比分析**

通过与真实数据的留一受试者（LOSO）分类比较，CD-IMU的宏F1得分为0.68，结合320维特征的合成令牌增强后，宏F1从0.729提高到0.809，显示出显著的性能提升。

**⚠️ 局限性**

限制在于PatchDecoder在重建过程中无法强制保持时间连续性，导致拼接波形中出现不连续性，并且在高方差活动中存在幅度衰减。

---

## 203. Generalized Covering radii and Deep Holes of a Family of Antiprimitive Narrow-Sense BCH Codes in Even Characteristic

**arXiv ID:** 2611.25365 | [PDF](https://arxiv.org/pdf/2611.25365v1)

**作者:** Ketong Ren `[一作]` (Southwest Jiaotong University), Haode Yan `[通讯]` (Harbin Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文确定了长度为q+1、设计距离为3的反原始窄感BCH码的广义覆盖半径、深孔和完整的余类领导者权重分布。

**💡 创新点**

创新点在于建立了完整的广义覆盖半径层次结构，并通过将综合表示简化为与相关立方多项式的根相关的条件，推导出接收字是深孔的必要和充分条件。

**🔧 技术方法**

使用了覆盖半径的概念，结合了立方多项式的根分析和拉格朗日插值技术。

**📊 数据集**

研究中使用的数据集为反原始窄感BCH码，特别是长度为q+1和设计距离为3的码，其中q=2^m且m≥2。

**📈 对比分析**

通过与现有方法的比较，本文的方法在确定深孔和余类领导者权重分布方面表现出更高的准确性和完整性，尤其是在处理广义覆盖半径时。

**⚠️ 局限性**

限制在于本文主要集中于偶特征的情况，而奇特征的情况需要不同的方法，未来的工作将探讨这一方向。

---

## 204. PSTO: A PDE-structured transport operator with shared implicit updates for multiscale CDR systems

**arXiv ID:** 2611.25607 | [PDF](https://arxiv.org/pdf/2611.25607v1)

**作者:** Kaiwen Li `[一作]` (Shanghai Jiao Tong University), Xingcai Lu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种PDE结构的传输算子（PSTO），用于从无反应数据中学习对流-扩散动力学，并能够在推理时推广到完整的对流-扩散-反应（CDR）系统。

**💡 创新点**

PSTO通过将PDE结构嵌入神经网络架构中，实现了对多尺度CDR系统的高效建模，且在推理时无需重新训练，显著提高了稳定性和准确性。

**🔧 技术方法**

使用了PDE结构的神经网络架构，结合放松的固定点迭代方法来近似隐式更新，从而提高预测的准确性和稳定性。

**📊 数据集**

在三个基准测试上进行了评估：一个经典的二维CDR系统，一个多尺度的罗伯逊逆流火焰，以及一个多尺度的氢平面喷流，所有数据均来自无反应的CD轨迹。

**📈 对比分析**

与三种代表性的神经代理模型（FNO、U-Net和U-FNO）进行比较，PSTO在每种情况下的平均相对L2误差在100个耦合步骤后至少减少了50%。

**⚠️ 局限性**

PSTO在处理小物种和横向速度时仍存在误差，未来的工作将扩展到更长的预测范围和更广泛的操作条件。

---

## 205. Game-Theoretic Data Allocation for Expert Specialization and Routing

**arXiv ID:** 2611.25167 | [PDF](https://arxiv.org/pdf/2611.25167v1)

**作者:** Mark Lovett `[一作]` (Dartmouth), Soroush Vosoughi `[通讯]` (Dartmouth)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的专家训练方法，通过位置博弈在特征空间中分配训练信号，以预测每个专家在特定提示上的表现。

**💡 创新点**

创新点在于通过博弈理论明确专家的专业化过程，而不是依赖于经验数据，并且能够在训练前预测专家的表现。

**🔧 技术方法**

使用了博弈论中的影响者游戏框架来优化专家的位置和训练信号的分配。

**📊 数据集**

使用了从七个安全相关基准汇编而来的数据集，构建了一个固定的特征表示。

**📈 对比分析**

与其他训练方法相比，该方法在多个数据拆分中表现出更低的负对数似然（NLL），并且在没有显式训练门控的情况下，能够有效地预测专家的表现。

**⚠️ 局限性**

限制在于依赖对称和对数凹核，可能导致专家无法拥有多个专业中心，并且影响力的衰减不能不对称。

---

## 206. HOPD: Historical On-Policy Distillation for SAR MLLM Continual Learning

**arXiv ID:** 2611.25391 | [PDF](https://arxiv.org/pdf/2611.25391v1)

**作者:** Yuxuan Li `[一作]` (Singapore Management University), Weijie Li `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种历史在线策略蒸馏（HOPD）方法，用于持续的合成孔径雷达（SAR）解释，旨在在学习新数据的同时保留早期任务能力。

**💡 创新点**

HOPD通过将源获取的LoRA状态转化为持久教师，提供了一种新的持续学习框架，首次专注于SAR多模态大语言模型的源增量学习。

**🔧 技术方法**

使用了历史在线策略蒸馏（HOPD）技术，结合了学生前缀蒸馏和不变坐标的重放视图。

**📊 数据集**

使用了SARFact数据集，该数据集由多个现有数据集合并和后处理而成，共包含356,624个训练问答和5,988个评估问答。

**📈 对比分析**

在SARVQA-CL评估协议中，HOPD的最终平均得分为54.51，超出最强对比方法2.96分，并在所有评估任务中均优于共享内存的Replay-SFT。

**⚠️ 局限性**

HOPD的训练时间存档随着源数量的增加而增长，可能导致维护开销增加；当前的视觉增强仅基于渲染强度图像，而未建模校准的雷达获取变化。

---

## 207. Tight Bounds for Distance-Unbiased Sorting

**arXiv ID:** 2611.26135 | [PDF](https://arxiv.org/pdf/2611.26135v1)

**作者:** Mats Bierwirth `[一作]` (Sorbonne Université), Johannes Lengler `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了精英随机局部搜索在排序排列中的应用，提出了一种距离无偏选择规则，该规则在每次迭代中考虑两个位置的交换，只有在它们的顺序错误时才进行交换。

**💡 创新点**

证明了静态规则的最优最坏情况高概率运行时间为Θ(n(log n)^2)，确立了谐波排序器在该类中的渐近最优性，并部分解决了Olesker-Taylor提出的开放问题。

**🔧 技术方法**

使用了距离无偏选择规则，允许根据预定计划或自适应地改变距离分布，结合了概率工具和盲搜索过程的嵌入。

**📊 数据集**

使用了排列集合S_n作为数据集，研究了不同的排列情况。

**📈 对比分析**

通过匹配上界和下界，证明了调度规则和自适应规则的最优最坏情况高概率运行时间均为Θ(n(log n)^2/loglog n)，调度调整距离分布的方式提高了静态规则的运行时间。

**⚠️ 局限性**

限制在于静态规则外的距离无偏类的开放问题仍未解决，即是否每个固定的对分布在某些初始排列上需要n(log n)^2次提议交换的概率趋近于1。

---

## 208. Neighbourhood Walkability and Cardiometabolic Morbidity in Greater London: A Fine-Grained Spatial Analysis

**arXiv ID:** 2611.26573 | [PDF](https://arxiv.org/pdf/2611.26573v1)

**作者:** Benjamin T. Pike `[一作]` (King's College London), Linus W. Dietz `[通讯]` (King's College London)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究探讨了伦敦地区的步行友好性与心脏代谢疾病的关系，开发了一个包含五个组成部分的开放步行友好性指数，并分析了其与八种心脏代谢疾病的记录患病率之间的关联。

**💡 创新点**

创新点在于提出了一个开放且可重复的步行友好性指数，考虑了多种城市形态维度，并通过空间误差模型和中心-边缘调整来探讨步行友好性与疾病之间的关系，强调了城市空间结构对健康影响的复杂性。

**🔧 技术方法**

使用了地理信息系统（GIS）技术，结合OpenStreetMap数据和国家健康数据，构建了步行友好性指数，并进行了空间误差模型分析。

**📊 数据集**

研究使用了2021年伦敦的4,994个低层超级输出区（LSOAs）的人口普查数据，以及来自初级医疗记录的八种心脏代谢疾病的患病率数据。

**📈 对比分析**

通过与现有的伦敦步行友好性指数进行比较，发现新指数与步行行为的相关性较强，且在调整人口组成和社会经济因素后，步行友好性与心脏代谢疾病的负相关性依然显著，但在空间误差调整后，部分关联显著减弱。

**⚠️ 局限性**

研究的局限性包括横断面生态设计限制了因果推断，无法充分考虑个体特征和居住选择的影响，且步行友好性指数未能捕捉个体在环境中的实际移动和体验。

---

## 209. Reasoning-Trace Analysis Using Large Language Models for Interpretable Q-Matrix Construction

**arXiv ID:** 2611.26422 | [PDF](https://arxiv.org/pdf/2611.26422v1)

**作者:** Jihong Zhang `[一作]`, Mingfeng Xue `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为推理轨迹分析（RTA）的框架，利用大型语言模型（LLM）生成带有属性标签的推理步骤，以构建可解释的Q矩阵，并对每个项目-属性分配进行重建、比较和检查。

**💡 创新点**

创新点在于通过RTA框架提供了对LLM生成的Q矩阵的透明性和可解释性，使得每个项目-属性分配的推理过程可以被检查，从而增强了AI在测量中的负责任使用。

**🔧 技术方法**

使用了大型语言模型（LLM）和链式思维（CoT）提示技术来生成推理步骤。

**📊 数据集**

使用了来自Tatsuoka的20道题目、八个属性的分数减法项目的数据集，以及536名考生的响应数据。

**📈 对比分析**

通过与专家参考Q矩阵的重叠程度进行比较，CoT生成的Q矩阵与参考矩阵的平均重叠率为0.809，而直接生成的Q矩阵为0.877，后者的性能更好，但CoT方法提供了更可解释的推理过程。

**⚠️ 局限性**

限制在于研究仅使用了一个特定的分数减法项目集和参考Q矩阵，且RTA分析的是生成的文本而非内部计算，模型生成的技能标签未经过独立编码。

---

## 210. Ontology of Software Maintenance

**arXiv ID:** 2611.25196 | [PDF](https://arxiv.org/pdf/2611.25196v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 211. Detecting Formation-Preserving GNSS Spoofing of UAV Swarms through Transmitter Parallax in Pseudorange and Doppler

**arXiv ID:** 2611.26023 | [PDF](https://arxiv.org/pdf/2611.26023v1)

**作者:** Minseok Park `[一作]` (Jeonbuk National University), Joon Soo Yoo `[通讯]` (Jeonbuk National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本论文提出了一种检测无人机群体中形成保持型GNSS欺骗的方法，通过分析伪距和多普勒信号中的视差特征来识别欺骗行为。

**💡 创新点**

创新点在于展示了即使在形成保持型欺骗下，仍然可以通过伪距和多普勒信号中的视差特征进行检测，并提出了一种新的似然比检验方法。

**🔧 技术方法**

使用了线性化的广义似然比检验（GLRT）技术，结合伪距和多普勒信号进行检测。

**📊 数据集**

使用了真实的GPS和Galileo轨道数据进行模拟实验，涉及16架无人机在不同飞行条件下的表现。

**📈 对比分析**

与GNSS-WASP的对比显示，提出的方法在检测率上显著优于GNSS-WASP的反制措施，能够在5公里的距离内检测到13%到60%的欺骗行为，而GNSS-WASP的检测率最高仅为5%。

**⚠️ 局限性**

限制在于所有结果均为模拟，未考虑实际接收器的跟踪环路、信号级效应和多路径干扰等因素，未来需要进行硬件测试以验证结果。

---

## 212. Hereditary permutations for simple types revisited: A baby case for University Algebra

**arXiv ID:** 2611.26507 | [PDF](https://arxiv.org/pdf/2611.26507v1)

**作者:** Thorsten Altenkirch `[一作]` `[通讯]` (University of Nottingham), Thorsten Altenkirch (University of Nottingham)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提供了一个新的证明，证明了简单类型λ演算中可定义同构的特征是遗传排列，并且这些同构由特定的重排和同等关系生成。

**💡 创新点**

创新点在于不依赖于完备性定理，直接在β-正常η-长项中进行证明，并且提出了一个一致性定理，表明两个同构的推导在特定方程下是相等的。

**🔧 技术方法**

使用了归纳法和遗传替换技术，所有结果都在Agda中形式化，没有使用公设。

**📊 数据集**

论文没有明确提到使用的数据集，但涉及的内容主要是简单类型λ演算的理论。

**📈 对比分析**

通过与经典证明的比较，本文的方法在不依赖于未类型化演算的情况下，直接分析可逆项的正常形式，证明了每个可逆的β-正常η-长项都是遗传排列，性能上提供了更简洁的证明。

**⚠️ 局限性**

限制在于该方法主要针对简单类型λ演算，如何将其推广到更复杂的类型理论（如大学代数）仍需进一步研究。

---

## 213. EgoWitness: Benchmarking Instance-Level Spatial Memory and Reasoning in Egocentric Video Streams

**arXiv ID:** 2611.25849 | [PDF](https://arxiv.org/pdf/2611.25849v1)

**作者:** Zhengru Fang `[一作]` (Hong Kong University of Science and Technology), Jun Zhang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个用于实例级空间记忆和推理的基准，涵盖了8,453个经过人工验证的问题，测试了从识别物体到回忆谁见过它的七种能力。

**💡 创新点**

创新点在于引入了一个新的基准，专注于实例级空间记忆，填补了现有长视频基准在多模态模型中的空白。

**🔧 技术方法**

使用了空间记忆代理框架，通过监督微调(SFT)和群体相对策略优化(GRPO)进行两阶段训练。

**📊 数据集**

使用了EgoLife和Ego4D派生的数据集，包含约324小时的长视频和短视频记录。

**📈 对比分析**

与12个直接视觉语言模型(VLMs)进行比较，开放权重模型的最高准确率为46.97%，而在识别最后见过物体的人时没有模型超过35%。提出的代理在1,320个保留问题上的准确率从23.64%提高到72.65%。

**⚠️ 局限性**

限制在于现有的视觉语言模型在多穿戴者历史中检索相关观察和验证身份方面仍然存在困难。

---

## 214. ROAM-GRPO: Enforcing Reasoning Diversity in GRPO via Routed Off-mode Approach Mining

**arXiv ID:** 2611.25926 | [PDF](https://arxiv.org/pdf/2611.25926v1)

**作者:** Jeff Duan `[一作]` (California Institute of Technology), Matt Thomson `[通讯]` (California Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为ROAM-GRPO的算法，旨在通过增强GRPO算法来提高模型在复杂推理任务中的输出多样性和创造力。

**💡 创新点**

ROAM-GRPO通过检测训练中解决良好的问题并将其转移到“离线模式”轨道，促使模型生成不同的解决策略，从而增加输出的创造性。

**🔧 技术方法**

使用了ROAM-GRPO算法，该算法结合了自我蒸馏和逻辑验证，利用外部LLM评估输出的逻辑有效性和独特性。

**📊 数据集**

在MATH lvl5数据集的1000个问题子集上进行训练。

**📈 对比分析**

与GRPO基线进行比较，ROAM-GRPO在pass@16上提高了4.7%，并且在解决方案的余弦距离上提高了约1.5-2倍，尽管在pass@1上表现较低。

**⚠️ 局限性**

ROAM-GRPO的训练不完全稳定，未能在pass@16上始终超过GRPO，且需要进一步实验以优化路由机制和自我蒸馏实现。

---

## 215. RelGeo: Pair-Conditioned Relational Calibration for Frozen Diffusion Correspondence

**arXiv ID:** 2611.25628 | [PDF](https://arxiv.org/pdf/2611.25628v1)

**作者:** Zixuan Dai `[一作]` (Yangtze University), Wei Zhan `[通讯]` (Yangtze University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为RelGeo的框架，通过引入配对条件关系来校准冻结的扩散描述符，以改善跨图像语义对应的匹配性能。

**💡 创新点**

创新点在于将互惠扩散的查询/键关联和预训练的几何传输作为共享的源-目标坐标来增强描述符匹配，同时保持全域余弦最近邻匹配。

**🔧 技术方法**

使用了互惠扩散的查询/键关联和预训练的几何传输技术，构建了谱关系坐标(SRC)和变形身份坐标(WIC)。

**📊 数据集**

使用了SPair-71k、AP-10K和SOCO等数据集进行评估。

**📈 对比分析**

与现有方法相比，RelGeo在所有十个评估中平均提高了5.51个PCK@0.10点，显示出显著的性能提升。

**⚠️ 局限性**

限制在于该方法依赖于预训练模型的质量，且在某些情况下可能无法完全消除候选选择错误。

---

## 216. Mutable State in Enrich-by-Need Security Protocol Analysis

**arXiv ID:** 2611.25142 | [PDF](https://arxiv.org/pdf/2611.25142v1)

**作者:** Joshua D Guttman `[一作]`, Eben Blaisdell `[通讯]` (MITRE Corporation)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文展示了如何在符号协议分析的链条空间方法中表示跨会话的可变状态，并在密码协议形状分析器CPSA中实现这些想法。

**💡 创新点**

创新点在于提出了一种部分有序状态变更的理论，并展示了如何将其整合到现有的协议分析工具中，以便更好地处理可变状态的协议行为。

**🔧 技术方法**

使用了部分有序图（causal directed acyclic graph）和状态机历史与传统链条空间执行模型的结合。

**📊 数据集**

论文中使用了银行金库作为示例，展示了如何应用所提出的理论来分析协议的安全性。

**📈 对比分析**

通过与现有的协议分析方法进行比较，展示了所提出的方法在处理可变状态时的有效性和准确性，能够更紧密地表征协议行为。

**⚠️ 局限性**

限制在于该方法可能在处理复杂的并发操作时面临挑战，尤其是在状态变更的顺序和依赖关系上。

---

## 217. Fusing Multimodal CNN Features with Structurally-Masked Transformers for Hierarchical Sub-Pixel Tree Species Mapping

**arXiv ID:** 2611.25824 | [PDF](https://arxiv.org/pdf/2611.25824v1)

**作者:** Mattia Ferrari `[一作]` (University of Trento), Lorenzo Bruzzone `[通讯]` (University of Trento)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种层次感知的多模态深度学习框架，用于通过融合PRISMA高光谱影像、高分辨率全色数据和数字高程模型来解决亚像素森林物种组成的分类问题。

**💡 创新点**

创新点在于引入了一种新的结构性掩蔽Transformer，结合多分支CNN架构，能够在复杂的森林生态系统中有效地处理亚像素异质性，并实现层次化的物种分类。

**🔧 技术方法**

使用了多分支CNN（基于EfficientNetV2的1D卷积神经网络和两个2D卷积神经网络）以及结构性掩蔽Transformer。

**📊 数据集**

使用了来自意大利三个不同生物地理区域的独特数据集，包括高光谱影像、全色影像和数字高程模型，数据集包含经过实地调查的样本。

**📈 对比分析**

与现有的多标签和基于排名的方法进行比较，提出的方法在识别次要物种及其相对优势方面表现出色，整体性能优于其他最先进的模型。

**⚠️ 局限性**

限制在于对地面真实数据的依赖，尤其是在物种级别的标注上，数据稀缺性可能影响模型的训练效果。

---

## 218. LATER: Learning Action-Conditioned Temporal Evidence Retrieval for Predictive World Models

**arXiv ID:** 2611.25671 | [PDF](https://arxiv.org/pdf/2611.25671v1)

**作者:** Zihan Zhou `[一作]`, Hao Tang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于JEPA的世界模型LATER，该模型通过分区记忆存储交互证据，并在候选回滚期间保持固定，以便在后续动作中进行检索。

**💡 创新点**

LATER结合了分区的真实过渡记忆、动作条件响应桥和辅助监督，能够在候选回滚期间保持记忆不变，从而提高了对延迟证据的控制和物理预测能力。

**🔧 技术方法**

使用了基于CF-JEPA的预测框架，结合了分区记忆、动作条件检索和多切割辅助监督技术。

**📊 数据集**

使用了公共的Reacher、PushT和TwoRoom数据集，以及合成的多阶段MuJoCo任务。

**📈 对比分析**

与六种模型和两个训练种子进行比较，LATER在干扰和清晰视图中都实现了最低的物理读出误差，相较于CF-JEPA，沿推的平均绝对误差（MAE）在干扰下减少了20.2%。

**⚠️ 局限性**

LATER的局限性在于其对早期物理证据的依赖，可能在某些情况下无法充分利用所有可用信息，尤其是在复杂或动态变化的环境中。

---

## 219. Release-Bound Withdrawal Authorization for Bitcoin Custody

**arXiv ID:** 2611.25231 | [PDF](https://arxiv.org/pdf/2611.25231v1)

**作者:** Jian Cui `[一作]` (University of Illinois Urbana-Champaign), Haixu Tang `[通讯]` (Indiana University Bloomington)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种新的比特币提款授权协议，称为Release-Bound Withdrawal Authorization (RBWA)，旨在防止未经授权的提款并恢复在链重组后丢失的授权。

**💡 创新点**

创新点在于将签名的不可逆决策与暂停操作放在同一复制顺序中，并要求独立可验证的同意和余额证据，从而增强了提款的安全性。

**🔧 技术方法**

使用了Python原型，结合Ed25519进行授权证明，SQLite作为测试账户账本，以及etcd v3.6.7作为钱包日志。

**📊 数据集**

实验使用了比特币核心32.1和etcd 3.6.7，测试数据集包括生成的密钥和测试资金。

**📈 对比分析**

通过与现有的Intent-Bound Withdrawal Authorization (IWAP)进行比较，RBWA在处理并发请求和撤销授权方面表现出更好的安全性和可靠性，尤其是在处理链重组时。

**⚠️ 局限性**

局限性包括实验仅在一个物理主机上进行，未测试被攻击者控制的环境，且未实现跨钱包的全局预算管理。

---

## 220. From Seed to BLOOM: Autonomously Growing a Better Out-of-Order Core

**arXiv ID:** 2611.26517 | [PDF](https://arxiv.org/pdf/2611.26517v1)

**作者:** Angela Cui `[一作]` (University of California, Berkeley), Sagar Karandikar `[通讯]` (University of California, Berkeley)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建并评估了一个名为MiracleGro的基于代理的AI系统，该系统能够在没有人工干预的情况下分析RTL行为和综合结果，识别性能瓶颈，并将架构思想转化为RTL实现，从而显著提高微处理器性能。

**💡 创新点**

创新点在于利用大型语言模型（LLM）代理自动化微处理器设计的迭代过程，显著减少了人力工程的需求，并实现了比人类优化更高的性能提升。

**🔧 技术方法**

使用了大型语言模型（LLM）代理和CHIA框架来设计和实验代理循环，进行微架构优化。

**📊 数据集**

使用了Berkeley Out-of-Order Machine (BOOM)作为基础，并在SkyWater 130nm工艺节点上进行设计和评估，最终在SPEC CPU2017基准测试上进行验证。

**📈 对比分析**

与传统的人力优化方法相比，MiracleGro在SPEC CPU2017基准测试中实现了85%的IPC提升和27%的频率提升，整体墙钟时间加速达到2.34倍，显示出显著的性能改进。

**⚠️ 局限性**

限制在于MiracleGro的结果可能无法直接转化到现代先进技术节点，且在处理复杂的微架构问题时仍需人类的洞察和输入。

---

## 221. RL Can Only Reinforce What It Sees Succeed

**arXiv ID:** 2611.26084 | [PDF](https://arxiv.org/pdf/2611.26084v1)

**作者:** Ruben Sanchez-Corcuera `[一作]` (University of Deusto), Miguel Fernandez-de-Retana `[通讯]` (University of Deusto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文探讨了在强化学习（RL）中，代理如何在预训练知识与新观察之间进行权衡，特别是在一个已知规则的拼图世界中，研究了结果奖励强化学习如何改变这一平衡。

**💡 创新点**

创新点在于构建了一个已知先验的环境，通过训练小型语言模型来精确测量模型在RL前后的决策变化，并提出了一个可以预测学习成功的计数方法。

**🔧 技术方法**

使用了强化学习（RL）技术，特别是结果奖励的RL（GRPO），并通过构建一个拼图环境来训练小型GPT模型。

**📊 数据集**

使用了由贝叶斯玩家进行的900万场游戏数据集，这些游戏规则和工具的使用频率是已知的。

**📈 对比分析**

通过比较模型在RL前后的决策变化，发现RL使得代理在适当时刻更倾向于做出决策，而不是固守先前的知识。性能上，RL在看到成功的行动后能够有效地提高决策质量。

**⚠️ 局限性**

限制在于所构建的环境是合成的，模型规模较小（最多85M参数），且只研究了一种类型的冲突和一种RL算法，未测试其他算法或更复杂的模型。

---

## 222. Inject Where It Matters: Objective-Dependent Feature Control in Rectified-Flow Image Editing

**arXiv ID:** 2611.25811 | [PDF](https://arxiv.org/pdf/2611.25811v1)

**作者:** Nina Goncharova `[一作]` (HSE University), Aibek Alanov `[通讯]` (HSE University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的图像编辑方法，通过控制源特征的注入位置和使用的变换器块来解决现有方法在源保留和编辑效果之间的权衡问题。

**💡 创新点**

创新点在于引入了自适应空间特征掩码和基于奖励的层选择策略，使得特征注入更加灵活和高效。

**🔧 技术方法**

使用了自适应空间掩码和REINFORCE算法进行层选择，优化了特征注入过程。

**📊 数据集**

使用了PIE-Bench数据集进行实验，包含700个示例，其中640个用于策略优化，60个用于验证。

**📈 对比分析**

与RF-Edit和Stable Flow等方法进行比较，学习到的九个注入块在LPIPS指标上表现优于18个块的配置，尽管在对齐度上略低。

**⚠️ 局限性**

限制在于未能证明推理速度的提升或局部编辑正确性的统一改善，结果主要关注整体图像指标。

---

## 223. ECHO: In-Context Behavior foundation model for Humanoid Whole-Body Control

**arXiv ID:** 2611.25778 | [PDF](https://arxiv.org/pdf/2611.25778v1)

**作者:** Xingyi Wang `[一作]` (Institute of Artificial Intelligence (TeleAI) China Telecom), Chenjia Bai `[通讯]` (Institute of Artificial Intelligence (TeleAI) China Telecom)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为ECHO的行为基础模型，旨在通过环境上下文适应实现类人机器人全身控制，能够在不同物理条件下执行多样化的行为。

**💡 创新点**

ECHO的创新点在于将一般行为能力与特定动态执行分离，利用交互历史学习环境动态，从而在不更新模型参数的情况下进行适应。

**🔧 技术方法**

使用了混合和退火采样（MIAS）技术进行预训练，以及上下文表示和残差适应（CRA）技术来学习环境动态。

**📊 数据集**

使用了Unitree G1机器人进行训练和评估，训练数据集包含超过1200小时的高质量运动数据。

**📈 对比分析**

通过与现有的行为基础模型（如HEFT和SONIC）进行比较，ECHO在跟踪精度和适应性方面表现优越，尤其是在动态变化较大的环境中。

**⚠️ 局限性**

ECHO的局限性在于其在测试时不更新参数，可能在某些极端环境条件下表现不如专门针对这些条件优化的模型。

---

## 224. Close&Safe: Maximal Task Preservation for Redundant Manipulators via C-Space Boundary Detour

**arXiv ID:** 2611.26031 | [PDF](https://arxiv.org/pdf/2611.26031v1)

**作者:** Abhigyan Singh `[一作]` (Indian Institute of Science), Ravi Prakash `[通讯]` (Indian Institute of Science)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种名为Close&Safe的方法，用于在冗余机械手中实现最大任务保留，特别是在存在未建模障碍物的情况下，确保任务轨迹的连续性和安全性。

**💡 创新点**

创新点在于将跟踪与避障解耦，确保在可行的情况下精确跟踪名义轨迹，仅在不可行的区间进行避障，同时最小化任务空间的偏差。

**🔧 技术方法**

使用了配置空间障碍边界的采样和构建路网的方法，结合动态规划和最小偏差的路径搜索算法。

**📊 数据集**

在多种模拟障碍配置和物理实验中进行了评估，使用了Franka机械手进行实际验证。

**📈 对比分析**

与四种基线方法进行了比较，结果显示Close&Safe在所有模拟场景中都实现了最高的任务保留，平均保留率为59.5%，而其他方法的保留率为45.6%。

**⚠️ 局限性**

限制在于规划时间主要受边界过滤器丢弃的样本数量影响，且障碍物集是静态的，动态障碍物的处理需要重新评估可行性。

---

## 225. ActMem: From Latent Memory Reasoning to Grounded Robotic Action

**arXiv ID:** 2611.25461 | [PDF](https://arxiv.org/pdf/2611.25461v1)

**作者:** Linfeng Dong `[一作]` (Beihang University), Mu Li `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为ActMem的框架，该框架在单一的视觉-语言骨干网络中解耦了记忆推理与感知基础。

**💡 创新点**

创新点在于将记忆推理与当前感知分开，使得记忆可以独立于当前观察进行推理，从而提高了长时间操作的成功率。

**🔧 技术方法**

使用了视觉-语言模型（VLM）和流匹配动作专家的架构，结合了潜在记忆链思维（Memory CoT）和感知注意力（Perception Attention）技术。

**📊 数据集**

在RoboMemArena和MIKASA-Robo等模拟基准上进行评估，并在真实世界的四个任务中测试。

**📈 对比分析**

与现有的记忆增强VLA方法相比，ActMem在RoboMemArena上达到了71.1%的任务成功率和80.8%的完成成功率，超越了最强基线7.5和8.7个百分点，并在MIKASA-Robo上取得了73.8%的平均成功率。

**⚠️ 局限性**

限制在于ActMem需要保持每个采样帧在上下文中，尽管高效的Memory CoT推理可以重用历史KV缓存，但每个决策仍需对全历史进行注意，因此延迟随着任务长度的增加而增长。

---

## 226. What the model knows but does not say: monitoring medical LLMs through verbalized thoughts

**arXiv ID:** 2611.24994 | [PDF](https://arxiv.org/pdf/2611.24994v1)

**作者:** Frederik Hauke `[一作]` (University Hospital RWTH Aachen), Daniel Truhn `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究展示了一种通过自然语言自动编码器监测医疗大型语言模型（LLMs）内部状态的方法，揭示了模型在处理输入时的潜在操控行为。

**💡 创新点**

创新点在于使用自然语言自动编码器将模型的内部激活转换为可读文本，从而有效地检测到输入中的提示注入，而这些信息在模型的输出中可能被隐藏。

**🔧 技术方法**

使用了自然语言自动编码器（NLA）技术，该技术结合了一个描述激活的语言生成器和一个重建器，通过强化学习共同训练。

**📊 数据集**

使用的数据集包括50个来自MIMIC-IV-Note v2.2的文本病例和19个来自Clusmann等人的医学图像，涵盖了多种临床场景。

**📈 对比分析**

与传统方法相比，研究表明，思想通道在所有注入条件下显著更频繁地检测到提示注入，且在干净输入上保持接近零的假阳性率。

**⚠️ 局限性**

限制在于样本量适中，仅测试了两种攻击目标和两种交付机制，且对控制令牌注入的盲点未能有效检测。

---

## 227. Phantom of the Masquerade: Graph Neural Network-Based Detection of Masquerade Attacks in Controller Area Networks

**arXiv ID:** 2611.26094 | [PDF](https://arxiv.org/pdf/2611.26094v1)

**作者:** Nehal Ameen `[一作]` (Virginia Commonwealth University), Irfan Ahmed `[通讯]` (Virginia Commonwealth University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于图神经网络（GNN）的框架PhantoMasq，用于检测控制器局域网络（CAN）系统中的伪装攻击。

**💡 创新点**

创新点在于将CAN流量建模为消息序列图，并结合轻量级统计特征进行联合学习，从而提高对伪装攻击的检测能力。

**🔧 技术方法**

使用了图卷积网络（GCN）作为主要技术，进行图级分类。

**📊 数据集**

使用了公开的REAL ORNL Automotive Dynamometer (ROAD)数据集，该数据集包含真实车辆的正常和攻击CAN流量。

**📈 对比分析**

与最先进的图嵌入基线相比，PhantoMasq在所有五种攻击类型上均实现了更高的AUC-ROC，提升幅度为0.13到0.36，并且对窗口配置的敏感性较低。

**⚠️ 局限性**

局限性包括仅在ROAD数据集上进行评估，未能验证对未见捕获、攻击类型或车辆的泛化能力，以及在信号解码不可用或恶意值与正常操作条件相似时的适用性。

---

## 228. Extending Rademacher Complexity-Based Generalization Theory to Domain Generalization

**arXiv ID:** 2611.25650 | [PDF](https://arxiv.org/pdf/2611.25650v1)

**作者:** Xi Wang `[一作]` (Shanxi University), Richard Yi Da Xu `[通讯]` (Hong Kong Baptist University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的领域泛化（DG）理论框架，通过将领域空间形式化为Wasserstein空间，解决了现有几何无关理论的局限性。

**💡 创新点**

创新点在于引入了Wasserstein几何到领域泛化中，并基于CD(K,∞)曲率理论推导出新的几何依赖的领域泛化误差界限。

**🔧 技术方法**

使用了Wasserstein几何和McDiarmid浓度不等式等技术。

**📊 数据集**

在实验中使用了PACS、Office-Home和Digits-DG数据集。

**📈 对比分析**

通过与现有方法的比较，结果表明增加潜在空间的曲率通常会降低领域泛化误差，且源领域数量的增加也有助于降低误差。

**⚠️ 局限性**

限制在于当前理论框架可能无法完全捕捉所有领域间的复杂关系，尤其是在某些特定目标领域上表现不稳定。

---

## 229. Introducing ReTiNA-Cinema: Real-Time Neuro-Augmented Cinema via Generative AI

**arXiv ID:** 2611.26469 | [PDF](https://arxiv.org/pdf/2611.26469v1)

**作者:** Antoine Bellemare-Pepin `[一作]` (Universite de Montreal), Karim Jerbi `[通讯]` (Universite de Montreal)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出了一种名为ReTiNA-Cinema的系统，通过整合脑电图（EEG）与生成性人工智能，实现了实时的生物适应性电影体验，使观众的生理信号能够影响电影的视觉表现。

**💡 创新点**

创新点在于利用EEG信号作为连续的输入，替代传统的离散提示方式，从而实现更流畅的创作互动，探索观众的身体反应如何塑造他们的感官体验。

**🔧 技术方法**

使用了脑电图（EEG）信号处理技术和基于Stable Diffusion Turbo的生成性AI管道，支持实时的图像转换。

**📊 数据集**

论文中未具体提及使用的数据集，但提到系统能够处理现有电影帧进行实时转换。

**📈 对比分析**

与传统的基于提示的生成性AI方法相比，ReTiNA-Cinema通过EEG信号实现了低延迟的连续视觉更新，提升了观众的参与感和沉浸感，但未在本研究中进行具体的性能评估。

**⚠️ 局限性**

限制在于EEG特征的心理解释尚未经过验证，且动态提示生成的可解释性和稳定性可能因上下文而异。此外，当前的输出可能偏向特定美学，且未进行实证用户研究来评估观众的反应。

---

## 230. Loss-Guided Tsetlin Machines: Making the Learning Objective Explicit

**arXiv ID:** 2611.26114 | [PDF](https://arxiv.org/pdf/2611.26114v1)

**作者:** Mayur Shende `[一作]` (University of Agder), Vladimir I. Zadorozhny `[通讯]` (University of Pittsburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的学习机制Loss-Guided TM，替代了Tsetlin Machine中的边际更新规则（MBUR），引入了显式的可替换损失函数。

**💡 创新点**

创新点在于将Tsetlin Machine的隐式反馈机制替换为显式的损失函数，使得模型能够针对特定任务优化性能，并且保持了Tsetlin Automaton和子句逻辑的不变性。

**🔧 技术方法**

使用了交叉熵（CE）、对称交叉熵（SCE）和非对称损失（ASL）等不同的损失函数。

**📊 数据集**

在六个数据集上进行了评估，包括MNIST、FashionMNIST、CelebA、IMDB、AG News和GoEmotions，涵盖多类、多标签和二分类任务。

**📈 对比分析**

与MBUR相比，Loss-Guided TM在五个数据集上匹配或提高了准确率和F1分数，尤其在不平衡的多标签数据集上，ASL的宏观F1提高了最多9个百分点。此外，Loss-Guided TM在所有六个数据集上显著改善了校准性能，校准误差减少了3到11倍。

**⚠️ 局限性**

限制在于未进行广泛的超参数网格搜索来优化每种条件，且当前的研究仅测试了三种损失函数，未来的研究需要探索更多损失函数的潜力。

---

## 231. DTS: Delta-Based Tile Scoring for Efficient Long-Context Sparse Attention

**arXiv ID:** 2611.25938 | [PDF](https://arxiv.org/pdf/2611.25938v1)

**作者:** Emre Uğur `[一作]` (Leiden University), Qinyu Chen `[通讯]` (Leiden University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于增量的块评分方法（DTS），用于稀疏块注意力，旨在提高长文档和代码库的处理效率。

**💡 创新点**

DTS通过根据局部向量变化选择查询和键的锚点，减少冗余评分计算，同时保持准确性，这是其创新之处。

**🔧 技术方法**

使用了增量打包和并行处理技术，以加速锚点选择和评分过程。

**📊 数据集**

在LLaMA-3.1-8B-Instruct和Qwen2.5-7B-Instruct-1M模型上进行了评估，使用了PG-19书籍和WikiText-103数据集。

**📈 对比分析**

与XAttention相比，DTS在128K令牌时减少了38%的评分管道延迟，并在RULER和LongBench基准测试中保持了与全注意力模型相差不超过0.10个百分点的准确性。

**⚠️ 局限性**

DTS的局限性在于其锚点选择和评分过程可能在某些情况下无法完全适应所有输入内容的变化。

---

## 232. On the monitorability of linear-time timed logics

**arXiv ID:** 2611.25276 | [PDF](https://arxiv.org/pdf/2611.25276v1)

**作者:** Mouloud Amara `[一作]` (Université Paris Cité), Nicanora Ordoño Wallig `[通讯]` (Université Paris Cité)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文解决了时间逻辑中的监控性问题，提出了一种新的线性时间的时间μ-演算，并证明其比现有的MTL更具表达能力。

**💡 创新点**

创新点在于提出了一种新的时间μ-演算，扩展了监控性结果，并提供了从MTL到新演算的编译器，解决了时间逻辑中的监控性问题。

**🔧 技术方法**

使用了线性时间的时间μ-演算，结合了监控性理论和编译技术。

**📊 数据集**

使用了时间逻辑的公式和监控性相关的逻辑公式，具体数据集未明确提及。

**📈 对比分析**

通过与现有的MTL和TPTL进行比较，证明了新演算在表达能力上的优势，并识别了重要的可监控片段，性能上显示出更强的监控能力。

**⚠️ 局限性**

限制在于监控性问题的复杂性，尤其是在处理非反应性监控器时，可能会影响监控的有效性和完整性。

---

## 233. Path-Independent Deleveraging Rules

**arXiv ID:** 2611.26623 | [PDF](https://arxiv.org/pdf/2611.26623v1)

**作者:** Christoph Schlegel `[一作]` (Flashbots), Akaki Mamageishvili `[通讯]` (Flashbots)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文对路径无关的自动去杠杆（ADL）规则进行了特征描述，定义了一个扩展的对数指数族，并探讨了在公平性和平台风险之间的权衡。

**💡 创新点**

创新点在于提出了一种通过历史数据的风险界限进行参数选择的方法，以选择在风险约束下的最大公平规则，并通过对Hyperliquid的ADL事件进行反事实重播，验证了中间参数的有效性。

**🔧 技术方法**

使用了路径独立性、均匀性、等价账户的平等待遇和一致性等公理化方法来设计ADL规则，并通过对数指数族进行特征化。

**📊 数据集**

使用了Hyperliquid的ADL事件数据集，特别是2025年10月10日的市场数据，重建了账户的头寸和现金余额。

**📈 对比分析**

与现有的ADL政策进行比较，发现每个评估的参数（包括水填充）在公平性和风险统计上显著改善了观察到的分配，尤其是中间参数在负担分散和后续风险统计上表现更佳。

**⚠️ 局限性**

限制在于本文的分析主要基于单一资产和被动头寸的假设，可能无法完全反映更复杂的市场动态和多资产环境下的ADL行为。

---

## 234. CalibCache: Cache-to-Full Feature Dynamics Calibration for Diffusion Transformers

**arXiv ID:** 2611.25296 | [PDF](https://arxiv.org/pdf/2611.25296v1)

**作者:** Mingyu Liang `[一作]` (Nanjing University), Jingwei Xu `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量级的离线校准方法，通过校准缓存与全计算特征加速度之间的系统性差异，来提高扩散变换器的特征缓存效果。

**💡 创新点**

创新点在于使用配对的全计算和简单缓存轨迹来直接监督特征动态的校准，从而在不增加额外全网络评估或在线优化的情况下，改善生成质量。

**🔧 技术方法**

采用了中心最小二乘法来拟合特征加速度的标量映射，并在推理过程中使用这些映射进行缓存残差的校正。

**📊 数据集**

使用了FLUX.1-dev和Qwen-Image数据集，以及DiT-XL/2在ImageNet-256上的实验。

**📈 对比分析**

与TaylorSeer和HiCache方法进行比较，结果显示在FLUX.1-dev上，LPIPS从0.563降低到0.429，且在N=5时，相比于全计算采样，去噪速度提升了2.86倍，同时在DiT-XL/2上，FID从16.39降低到10.37。

**⚠️ 局限性**

限制在于该方法依赖于配对的全计算和简单缓存轨迹，可能在某些情况下无法完全捕捉特征动态的复杂性。

---

## 235. Software-Defined Internet Integration of Satellite Networks (SDISN)

**arXiv ID:** 2611.25837 | [PDF](https://arxiv.org/pdf/2611.25837v1)

**作者:** Michael Menth `[一作]` (University of Tübingen), Jens Haala `[通讯]` (Tesat-Spacecom GmbH & Co. KG)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

提出了一种软件定义的卫星网络互联网集成方案，支持用户终端（UT）和互联网节点之间的IP流量传输。

**💡 创新点**

创新点在于通过控制器持续更新转发信息，以应对卫星网络中频繁变化的服务卫星（SrvSats），并提出了针对安全性和延迟的架构扩展。

**🔧 技术方法**

采用了软件定义网络（SDN）技术，结合了动态和静态IP地址分配机制，以及流量工程能力。

**📊 数据集**

使用了低地球轨道（LEO）卫星网络，具体数据集未明确提及，但涉及到多个现有的卫星网络如Starlink和OneWeb。

**📈 对比分析**

与传统的卫星网络集成方法相比，提出的架构在流量工程和延迟管理方面表现出更高的灵活性和效率，能够有效减少延迟和提高安全性。

**⚠️ 局限性**

限制在于卫星的复杂性和资源限制，可能导致在高负载情况下的性能下降，以及对控制器的依赖性较强。

---

## 236. DS3M: Deep structured state space models for nonlinear system identification on modern hardware

**arXiv ID:** 2611.25274 | [PDF](https://arxiv.org/pdf/2611.25274v1)

**作者:** Marco Roschkowski `[一作]` (University of Wuppertal), Hannes Gernandt `[通讯]` (University of Wuppertal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的深度结构状态空间模型（DS3M），用于从测量的输入和输出中学习动态模型，并结合了一个编码器来估计初始状态。

**💡 创新点**

创新点在于通过自定义GPU内核减少了序列评估的启动和内存开销，从而显著加快了训练速度。

**🔧 技术方法**

使用了自定义的GPU内核，采用Triton编程语言实现，并结合了深度神经网络。

**📊 数据集**

使用了Wiener-Hammerstein数据集以及其他标准非线性系统识别基准。

**📈 对比分析**

与SUBNET和其他基准方法相比，DS3M在训练时间和准确性上表现优越，训练时间显著减少，同时在多个基准上达到了相似或更好的性能。

**⚠️ 局限性**

限制在于模型参数的大小受限于GPU寄存器的容量，这可能限制了未来在更大系统上的应用。

---

## 237. Similarity-based OSF Logic for Flexible SPARQL Query Answering

**arXiv ID:** 2611.26261 | [PDF](https://arxiv.org/pdf/2611.26261v1)

**作者:** Gian Carlo Milanese `[一作]` (University of Milano-Bicocca), Gabriella Pasi `[通讯]` (University of Milano-Bicocca)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个更新版本的模糊有序特征逻辑推理器，支持对RDF知识图谱的灵活查询回答。

**💡 创新点**

创新点在于引入了基于相似性的推理，允许在严格本体约束下返回近似解决方案。

**🔧 技术方法**

使用了模糊逻辑和相似性关系的推理算法，并集成了本体感知的查询翻译引擎。

**📊 数据集**

使用了RDF知识图谱作为数据集。

**📈 对比分析**

通过将查询表达为术语并进行模糊约束规范化，能够在不一致的查询中返回近似结果，性能优于传统的严格本体约束方法。

**⚠️ 局限性**

当前版本的限制包括性能问题，计划用C++引擎替代Python核心以提高速度和内存效率，同时希望扩展对不相交类别的支持。

---

## 238. Adopting Embedded AI Through an Innovation Team

**arXiv ID:** 2611.25939 | [PDF](https://arxiv.org/pdf/2611.25939v1)

**作者:** Viktoria Stray `[一作]` (University of Oslo), Nils Brede Mo `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究通过对一家大型挪威金融组织的临时跨职能创新团队进行定性案例研究，探讨了如何在现有产品中嵌入人工智能（AI）能力。研究通过18次访谈、10天的办公室观察和内部文件分析，发现结构分离、团队内部的AI能力、高度自主性、资深专家和灵活的工作方式促进了嵌入式AI解决方案的开发。

**💡 创新点**

创新点在于识别了七个成功因素，以支持通过创新团队采用嵌入式AI，并提出了专业知识意识作为敏捷团队有效性模型中冗余的替代方案。

**🔧 技术方法**

研究采用了定性案例研究方法，结合了主题分析和敏捷团队有效性模型（ATEM）进行数据分析。

**📊 数据集**

数据集包括18次半结构化访谈，参与者包括团队成员和管理层，以及10天的办公室观察和内部文件。

**📈 对比分析**

与传统软件开发相比，嵌入式AI开发具有探索性、数据依赖性和不确定性。研究发现，团队的自主性和内部AI能力是成功的关键，但在与永久产品团队的协调中存在挑战，尤其是在路线图不一致和优先级差距方面。

**⚠️ 局限性**

限制在于研究仅基于一个组织的单一案例，缺乏统计推广的能力，且研究结果主要关注于采用条件而非结果。

---

## 239. On the Hamel Coefficients and the Boltzmann-Hamel Equations for the Rigid Body

**arXiv ID:** 2611.25725 | [PDF](https://arxiv.org/pdf/2611.25725v1)

**作者:** Andreas Müller `[一作]` `[通讯]`, Andreas Müller

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文系统地推导了刚体的Hamel系数及相应的Boltzmann-Hamel方程，涵盖了三种刚体速度的表示方式，且不依赖于局部坐标。

**💡 创新点**

创新点在于首次明确地从几何角度推导出Hamel系数，且不使用局部坐标的描述，提供了经典刚体动力学与现代李群方法之间的直接联系。

**🔧 技术方法**

使用了Hamel方程和李群的几何方法，结合了经典的拉格朗日动力学和现代几何力学的观点。

**📊 数据集**

研究中涉及的示例数据集包括不均匀质量密度的滚动球，作为非完整约束刚体的典型例子。

**📈 对比分析**

通过与现有文献中的Hamel方程进行比较，展示了所提出方法的有效性，结果表明Hamel系数与李群的结构常数相同，且在不同表示下的方程形式一致。

**⚠️ 局限性**

限制在于尽管推导过程不依赖于局部坐标，但在某些情况下，具体的几何解释和应用可能仍然需要局部坐标的支持。

---

## 240. Control of Complex Social Systems via Committed Minorities and Opinion Broadcast

**arXiv ID:** 2611.26092 | [PDF](https://arxiv.org/pdf/2611.26092v1)

**作者:** Roberta Raineri `[一作]` (Modelway), Lorenzo Zino `[通讯]` (Politecnico di Torino)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本论文提出了一种统一的博弈论共演模型，用于控制复杂社会系统中的意见形成和决策过程，探讨了通过引入承诺少数群体和信息广播两种方法来控制共演模型的有效性。

**💡 创新点**

创新点在于将意见形成与决策过程结合在一个共演模型中，并提出了两种控制方法的理论保证和实施策略，展示了如何通过这些方法实现社会变革。

**🔧 技术方法**

使用了博弈论和共演模型的数学工具，结合了网络结构和个体行为的动态分析。

**📊 数据集**

使用了来自现实世界的数据集，特别是通过网络结构推断的真实数据，进行案例研究以验证模型的有效性。

**📈 对比分析**

通过与传统方法的比较，展示了所提控制方法在不同案例研究中的有效性，结果表明承诺少数群体和信息广播在实现社会变革方面具有不同的效果和效率。

**⚠️ 局限性**

限制在于模型假设的简化，可能无法完全捕捉复杂社会系统中的所有动态特征，且在实际应用中可能面临伦理和实施的挑战。

---

## 241. Physical Laws without Physics Loss? Probing Learned Dynamics in Video World Models through Forced Response

**arXiv ID:** 2611.25626 | [PDF](https://arxiv.org/pdf/2611.25626v1)

**作者:** Yutao Li `[一作]` (Harbin Institute of Technology), Hui Li `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文探讨了视频生成模型是否能够在没有物理损失的情况下学习物理法则，特别是在非线性强迫响应的情况下，通过对悬挂物体在地震下的运动进行建模。

**💡 创新点**

创新点在于提出了双重视觉-数值条件化的方法，并通过六个图像到视频的骨干网络，仅使用流匹配损失来预测运动，展示了模型在未见激励、物体、姿势和场景上的泛化能力。

**🔧 技术方法**

使用了六个图像到视频的骨干网络，采用流匹配损失进行训练，并设计了三个测试层次来评估模型学习的物理法则。

**📊 数据集**

使用的数据集包括5206个视频，来自91个地震记录和15栋建筑，每个视频展示一个悬挂物体，测试集包含600个案例，涉及不同的地震记录和物体。

**📈 对比分析**

与其他方法比较时，六个骨干网络在运动和外观评分上表现不同，Wan模型在未见案例中表现最佳，能够准确预测运动并在多个测试层次中重现物理法则。

**⚠️ 局限性**

限制在于模型在状态估计或传播错误时可能无法表现出学习到的动态，且在短暂事件（如冲击）上可能需要更强的训练信号。

---

## 242. ProtoDiffusion: Token-per-Layer Generative Modeling of Multi-Layer Images

**arXiv ID:** 2611.26586 | [PDF](https://arxiv.org/pdf/2611.26586v1)

**作者:** Adrià Ciurana Lanau `[一作]` (Canva), Petru-Daniel Tudosiu `[通讯]` (Canva)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于潜在扩散框架的图形设计生成模型，能够将设计分解为可编辑的RGBA层，并支持图像到设计和文本到设计的生成。

**💡 创新点**

创新点在于采用了每层一个候选向量的表示方法，结合共享的空间原型，从而显著减少了表示的复杂性，并实现了联合去噪。

**🔧 技术方法**

使用了潜在扩散模型和多层变分自编码器（ProtoVAE），通过双流架构实现共享原型和层候选的联合建模。

**📊 数据集**

使用了层次化的图形设计数据集和合成的图像数据集，包括Crello和RevealLayer数据集。

**📈 对比分析**

与现有方法（如QIL和LayerD）进行比较，模型在RGB L1和alpha soft IoU评分上表现优异，且在文本到设计生成中，FID和文本准确度均优于基线方法，同时运行速度快19.1倍。

**⚠️ 局限性**

模型在处理非常密集或重叠的构图时仍存在挑战，保持细元素和减少残余泄漏是当前的限制，且在没有边界框的情况下，alpha soft IoU仍低于QIL。

---

## 243. Erasing the Wrong Thing: Conditional Attribute Leakage and Concept Erasure in Shared Foundation-Model Embeddings

**arXiv ID:** 2611.25640 | [PDF](https://arxiv.org/pdf/2611.25640v1)

**作者:** Haijun He `[一作]` (Wuhan University), Junjun He `[通讯]` (Shanghai Innovation Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

论文探讨了共享医学图像嵌入的隐私问题，提出了任务条件下的概念擦除方法TC-LEACE，以解决传统全局概念擦除方法在保护敏感属性方面的不足。

**💡 创新点**

创新点在于提出了TC-LEACE，这是一种在每个任务层次内强制线性保护的闭合形式变体，能够在保持任务效用的同时减少条件泄漏。

**🔧 技术方法**

使用了线性概念擦除技术LEACE的变体TC-LEACE，并通过闭合形式的投影方法进行敏感属性的擦除。

**📊 数据集**

使用了CelebA数据集中的85对属性以及CANDID-PTX胸部放射影像数据集进行评估。

**📈 对比分析**

与传统的全局概念擦除方法LEACE相比，TC-LEACE在保持任务效用的同时，减少了条件泄漏，且在强关联的属性对上表现更佳。

**⚠️ 局限性**

局限性在于TC-LEACE主要针对线性攻击者的保护，未能有效防止非线性攻击者的泄漏，同时在新数据上的保证可能不如在拟合数据上强。

---

## 244. TuringPAI: A Foundation for On-Device Physical AI

**arXiv ID:** 2611.25474 | [PDF](https://arxiv.org/pdf/2611.25474v1)

**作者:** Qiman Wu `[一作]` (XPeng Inc), Xianming Liu `[通讯]` (XPeng Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了TuringPAI，一个用于物理AI的模型系列，包括TuringPAI-VL（视觉-语言理解）、TuringPAI-Drive（自动驾驶）和TuringPAI-Embodied（机器人操作）。

**💡 创新点**

创新点在于结合了HybridViT-MLQ架构和针对物理AI的训练数据，提供了一个共享的视觉-语言基础，能够在有限的计算资源下进行空间和动态环境的理解。

**🔧 技术方法**

使用了HybridViT-MLQ架构，该架构结合了卷积局部聚合和基于Transformer的全局建模，并引入了多尺度潜在查询（MLQ）来整合视觉特征和观察历史。

**📊 数据集**

使用了XPeng的内部驾驶数据集，包含约298,000小时的多摄像头记录，涵盖各种驾驶环境和行为，以及LIBERO套件用于机器人操作的评估。

**📈 对比分析**

与Qwen2-VL-2B模型进行比较，TuringPAI-VL在多模态理解和空间几何感知方面表现更好，TuringPAI-Drive在自动驾驶评估中保持了与全标记HybridViT基线1.2%的误差，TuringPAI-Embodied在LIBERO评估中取得了96.8%的成功率。

**⚠️ 局限性**

限制在于模型的计算资源和延迟约束，尽管在多个任务上表现良好，但在更复杂的环境和长时间的物理理解方面仍需进一步研究和优化。

---

## 245. Pareto-Optimal and Strategyproof Data-Sharing Between Competitors

**arXiv ID:** 2611.26108 | [PDF](https://arxiv.org/pdf/2611.26108v1)

**作者:** Morgan Colin-Mourre `[一作]` (Inria), Patrick Loiseau `[通讯]` (Inria)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在同一市场中竞争的公司之间的数据共享，分析了数据共享对预测模型质量的影响。

**💡 创新点**

提出了一种新的数据共享合同模型，确保所有公司在共享数据后都能获得相等的收益，并消除了共享数据对市场份额的负面影响。

**🔧 技术方法**

使用了参数化的合同模型，结合了机器学习和经济学中的市场份额模型。

**📊 数据集**

使用了来自Expedia的真实数据集，包含超过3800万条随机观察数据，提取了500000个数据点进行实验。

**📈 对比分析**

通过与无共享场景进行比较，证明了数据共享合同在所有公司中都能带来利润增长，尤其是在机器学习任务更困难或公司初始数据集更接近同质时，性能显著提升。

**⚠️ 局限性**

模型假设公司仅在数据量上存在差异，而未考虑数据内容的异质性；此外，计算精度函数的成本较高，可能限制了合同的实际应用。

---

## 246. PRIO: Pre-trained Representation for Interactive Objects in 4D Worlds

**arXiv ID:** 2611.25500 | [PDF](https://arxiv.org/pdf/2611.25500v1)

**作者:** Shixun Huang `[一作]` (University of British Columbia), Peter Yichen Chen `[通讯]` (University of British Columbia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种名为PRIO的预训练表示，用于将静态3D对象转变为交互式4D世界模型，能够在没有针对每个对象的训练的情况下，预测对象的配置空间。

**💡 创新点**

PRIO的创新点在于它通过静态几何形状预测对象特定的配置空间，并且该空间是线性的，从而实现实时物理响应。

**🔧 技术方法**

使用了拉格朗日动力学来驱动对象的运动，并通过预计算的矩阵实现高效的物理响应。

**📊 数据集**

在多样化的4D轨迹上进行了预训练，支持网格、点云和3D高斯等多种数据格式。

**📈 对比分析**

与现有的4D生成方法相比，PRIO在生成动态时速度更快，并且更忠实地跟随用户的动作，且在相同的训练预算下，重建误差随着预训练集的增大而降低。

**⚠️ 局限性**

限制在于预测的配置是连续和低维的，因此无法处理拓扑变化（如撕裂和断裂）以及预训练中缺失的行为（如流体）。

---

## 247. No-Harm-Jury A Jury-Based Multi-Dimensional Harm Evaluation Framework for Medical AI Responses

**arXiv ID:** 2611.24972 | [PDF](https://arxiv.org/pdf/2611.24972v1)

**作者:** Antonio Puertas Gallardo `[一作]` (European Commission), Mario Ceresa `[通讯]` (European Commission)

**通讯引用:** 1227 | [OpenAlex ID](https://openalex.org/A5082433089)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于陪审团的自动化评估框架No-Harm-Jury，用于对AI生成的医学回答进行多维度的伤害评估，涵盖七个不同的伤害维度。

**💡 创新点**

创新点在于结合了五个不同的语言模型作为独立的评审员，通过中位数聚合来抑制个体评审员的偏见，同时保持对伤害的全面覆盖。

**🔧 技术方法**

使用了五个多样化的语言模型（LLMs）作为评审员，并通过中位数聚合和可靠性加权扩展来进行评分。

**📊 数据集**

使用了来自PubMedQA、MedQA和MedMCQA的3,000个问题，最终得到了2,849个有效响应。

**📈 对比分析**

与现有方法相比，No-Harm-Jury在自动化可扩展评估、五个多样化的LLM评审员、稳健的中位数聚合、七个概念上不同的伤害维度以及应用于医学问答基准方面具有独特优势。性能方面，MedQA的13.4%响应被标记为关键（Critical），而PubMedQA仅为2.9%。

**⚠️ 局限性**

局限性包括：所有评估的响应均由单一的生成模型产生，缺乏金标准标签，数据集异质性，以及评审员评分标准与关键阈值不一致等问题。

---

## 248. I See You: Initial Multi-Robot LiDAR Registration with Mutual-Observation Checks

**arXiv ID:** 2611.26276 | [PDF](https://arxiv.org/pdf/2611.26276v1)

**作者:** Jonathan Lichtenfeld `[一作]` (Technical University of Darmstadt), Oskar von Stryk `[通讯]` (Technical University of Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种集中式的多机器人注册框架，旨在实现无人地面车辆（UGVs）在搜索与救援任务中的有效协作，特别是在没有基础设施的环境中进行初始对齐。

**💡 创新点**

创新点在于结合了全局点云对齐与物理可行性检查，主动拒绝不可能的姿态假设，从而提高了多机器人系统的注册准确性和可靠性。

**🔧 技术方法**

使用了LiDAR传感器进行点云数据的配准，并结合了物理可行性检查和图优化技术。

**📊 数据集**

使用了一个自定义的数据集，该数据集包含四个同步机器人的LiDAR扫描，涵盖多种室内和室外环境，并提供了地面真实数据。

**📈 对比分析**

通过与现有的配准方法进行比较，提出的框架在拒绝错误配准方面表现出色，拒绝了超过三分之一的错误配准，同时错误拒绝率低于1%。最终优化的姿态在成功的运行中达到了足够的准确性。

**⚠️ 局限性**

局限性包括：互观察检查主要在开放平坦场景中有效，在杂乱和植被环境中效果较差；无法拒绝所有小的平移或旋转误差；且该方法依赖于足够的几何重叠和传感器覆盖。

---

## 249. Revisiting Finiteness of Matrix Monoids

**arXiv ID:** 2611.26143 | [PDF](https://arxiv.org/pdf/2611.26143v1)

**作者:** Rida Ait El Manssour `[一作]` (University of Oxford), James Ben Worrell `[通讯]` (University of Oxford)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文研究了与有理矩阵的有限单子相关的决策问题，展示了确定给定有限呈现单子的有限性在多项式时间内可解，并改进了已知的复杂度界限。

**💡 创新点**

创新点在于提供了一个新的多项式位大小界限，适用于有限单子中的矩阵条目，并且通过将问题简化为矩阵群的情况，得到了新的复杂性结果。

**🔧 技术方法**

使用了非交换代数和矩阵单子的结构理论，结合了多项式时间算法来判断有理矩阵单子是否与整数矩阵单子共轭。

**📊 数据集**

使用了有限矩阵单子的生成矩阵集，具体数据集未明确给出，但涉及到有理数域上的矩阵。

**📈 对比分析**

通过与已有的复杂性结果进行比较，本文的有限性问题被证明在多项式空间内可解，而有限成员问题被证明是NP完全的，提供了匹配的下界。

**⚠️ 局限性**

限制在于尽管本文改进了复杂性界限，但仍然存在进一步优化的空间，特别是在多项式位大小界限的指数方面。

---

## 250. How Reliable Are Explanations of Language Models That Detect Insider Threats?

**arXiv ID:** 2611.25796 | [PDF](https://arxiv.org/pdf/2611.25796v1)

**作者:** Suramya R. Angdembay `[一作]` `[通讯]` (University of Southern Mississippi), Suramya R. Angdembay (University of Southern Mississippi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究审计了用于检测内部威胁的语言模型的解释，重点分析了检测器的异常评分、特征选择和测试方法的敏感性。

**💡 创新点**

提出了 tipping radius 概念，作为控制度量变化对结果影响的敏感性度量，首次量化了选择度量对模型特征解释的影响。

**🔧 技术方法**

使用了语言模型（如 Qwen3-8B 和 Qwen2.5-3B）进行异常检测，并通过稀疏自编码器（SAE）分析模型内部状态。

**📊 数据集**

使用了 CERT 发布的模拟工作场所日志数据集，包含内部威胁事件的脚本化记录。

**📈 对比分析**

通过与控制组比较，发现移除特征对异常评分的影响在不同度量下存在显著差异，tipping radius 测量显示，控制度量的微小变化可能会消除或反转结果。

**⚠️ 局限性**

研究的局限性包括使用的 CERT 数据集为合成数据，无法广泛推广结果；此外，模型的内部机制与实际员工行为之间的关系尚未明确。

---

## 251. Attention Head Specialization: Phase Transitions in Sample Complexity

**arXiv ID:** 2611.26228 | [PDF](https://arxiv.org/pdf/2611.26228v1)

**作者:** Léo Catteau `[一作]`, Lenka Zdeborová `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多头注意力机制中头部专业化如何依赖于可用训练数据的数量，发现随着数据集大小的增加，BERT模型中的头部专业化现象显著增强。

**💡 创新点**

提出头部专业化是样本复杂度中的一种统计相变，并将这种平衡转变与训练过程中的专业化联系起来。

**🔧 技术方法**

使用了高维多头注意力模型和BERT掩蔽语言模型（MLM）进行实验。

**📊 数据集**

使用了BookCorpus和英语维基百科的数据集，训练BERT_Small模型。

**📈 对比分析**

通过与可解的高维模型进行比较，展示了样本复杂度如何影响头部专业化的出现，发现头部专业化在样本复杂度的临界值处表现出非解析性，验证了理论预测。

**⚠️ 局限性**

研究的局限性在于，尽管实验结果支持统计相变的解释，但有限维实验无法完全建立定义相变的渐近非解析性。

---

## 252. Trajectory-Calibrated Merging of Embodied Policies

**arXiv ID:** 2611.25575 | [PDF](https://arxiv.org/pdf/2611.25575v1)

**作者:** Wenshuo Chen `[一作]` (Hong Kong University of Science and Technology), Yutao Yue `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种执行感知的无梯度框架TCR，将多个机器人专家合并为一个单一策略，以增强机器人的能力。

**💡 创新点**

创新点在于通过两次校准过程，结合任务执行和迭代动作生成的输入，确保合并后的策略能够保留专家的行为，而不需要专家路由或额外的梯度训练。

**🔧 技术方法**

使用了无梯度的回归方法，通过专家的输出进行模块的适配和合并。

**📊 数据集**

在LIBERO数据集上进行了评估，合并了四个专家，进行了1200次评估。

**📈 对比分析**

与FeatCal、RegMean和直接权重平均方法相比，TCR在LIBERO上取得了77.83%的成功率，分别超出9.50、24.17和33.00个百分点，显示出其优越的性能。

**⚠️ 局限性**

限制在于单一静态权重集对冲突技能的适应能力有限，且需要对专家的架构和观察、动作接口有白盒访问权限，跨不同实体的合并仍需未来研究。

---

## 253. Near-Quadratic Sample Compression for All VC Classes

**arXiv ID:** 2611.25011 | [PDF](https://arxiv.org/pdf/2611.25011v1)

**作者:** Yang Cai `[一作]` (Yale University), Manfred K. Warmuth `[通讯]` (Google)

**通讯引用:** 21682 | [OpenAlex ID](https://openalex.org/A5108549518)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

论文证明了每个VC维度为d的概念类都存在一个大小为O(d^2 log^2 d)的样本压缩方案，从而解决了样本压缩猜想的弱版本。

**💡 创新点**

创新点在于首次提供了一个多项式大小的样本压缩方案，显著改善了之前的指数界限。

**🔧 技术方法**

使用了样本压缩框架和加权多数投票算法，结合了线性规划对偶性和熵的概念。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了概念类的VC维度和样本的可实现性。

**📈 对比分析**

与之前的方案相比，新的压缩方案在样本大小上有显著的改进，之前的方案是指数级的，而本论文提供的方案是多项式级的，具体为O(d^2 log^2 d)。

**⚠️ 局限性**

限制在于当前的压缩方案仍然依赖于样本大小，未来的工作希望能进一步降低到O(d)的压缩大小，从而完全解决样本压缩猜想。

---

## 254. Towards Scale-Up Networking: Challenges and Open Questions

**arXiv ID:** 2611.26156 | [PDF](https://arxiv.org/pdf/2611.26156v1)

**作者:** Tingting Xu `[一作]` (Hong Kong University of Science and Technology), Kai Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文分析了加速器之间的通信瓶颈，提出了规模化网络的需求，并对现有的规模化网络架构进行了比较，识别出设计中的权衡和未来的挑战。

**💡 创新点**

创新点在于提出了一种需求驱动的框架，用于分析当前的规模化网络，并指导未来系统的设计，特别是在高带宽和低延迟的通信需求方面。

**🔧 技术方法**

使用了对比分析的方法，评估了多种规模化网络架构的拓扑、协议机制和通信语义，揭示了它们在性能、可扩展性和实现成本之间的不同权衡。

**📊 数据集**

分析了包括NVIDIA NVLink、AMD Infinity Fabric、Google ICI等在内的11种代表性规模化网络架构，涵盖了不同的功能范围和实现成熟度。

**📈 对比分析**

通过对比现有的规模化网络，发现它们在带宽、延迟和可扩展性方面存在显著差异，且不同架构在满足AI工作负载的需求时表现出不同的性能特征。

**⚠️ 局限性**

限制在于当前的规模化网络架构在物理互连、通信效率和资源共享等方面仍面临挑战，特别是在支持异构设备互连和高效通信方面的开放问题。

---

## 255. An Efficient Algorithm for Matroid Cups

**arXiv ID:** 2611.26602 | [PDF](https://arxiv.org/pdf/2611.26602v1)

**作者:** Stephen Arndt `[一作]` (Carnegie Mellon University), Kirk Pruhs `[通讯]` (University of Pittsburgh)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一种多项式时间算法，用于解决具有O(1)差异的矩阵杯问题。

**💡 创新点**

提出了一种高效的算法，能够在多项式时间内实现所有矩阵的O(1)差异。

**🔧 技术方法**

使用了单位时钟算法和一系列简化实例的归约方法。

**📊 数据集**

使用了具有不同秩的矩阵数据集，特别是秩为1的矩阵。

**📈 对比分析**

与现有方法相比，算法在性能上表现出色，能够在多项式时间内实现O(1)差异，且整体算法的差异最多为33。

**⚠️ 局限性**

算法的局限性在于尽管能够实现O(1)差异，但在某些情况下可能无法达到最佳常数。

---

## 256. TideCache: Efficient Hybrid LLM Serving with Transparent Cache Management and Layer-Wise Streaming

**arXiv ID:** 2611.26619 | [PDF](https://arxiv.org/pdf/2611.26619v1)

**作者:** Jinbin Luo `[一作]` (Massachusetts Institute of Technology), Qinghao Hu `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种混合模型服务系统，旨在优化CPU-GPU超芯片上的GPU内存分配和管理，以提高语言模型的性能。

**💡 创新点**

通过动态重新分配GPU内存和利用空闲的CPU-GPU带宽，显著提高了混合模型的服务能力和响应速度。

**🔧 技术方法**

使用了SGLang实现了该系统，并结合了Cache Streamer和Request Scheduler等组件来优化内存管理和请求调度。

**📊 数据集**

在GB300和GH200超芯片上进行了评估，使用了七种模型/平台配置进行测试。

**📈 对比分析**

与SGLang的HiCache相比，该系统在处理请求时能够完成最多2.04倍的代理轮次，整体P50每个token延迟减少了20-59%，并且代理P50 TTFT提高了1.40-3.05倍。

**⚠️ 局限性**

该系统的局限性在于需要在不同的工作负载下动态调整内存分配，可能在某些情况下仍然面临内存压力。

---

## 257. Domain-Aware Conformal Prediction for Trustworthy Spacecraft Pose Estimation under Sim-to-Real Shift

**arXiv ID:** 2611.26065 | [PDF](https://arxiv.org/pdf/2611.26065v1)

**作者:** Nidhal Eddine Chenni `[一作]` (University of Luxembourg), Djamila Aouada `[通讯]` (University of Luxembourg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文研究了在合成到真实（sim2real）差距下的航天器姿态估计，提出了一种通过将合成图像转换为真实图像外观的校准方法，以提高姿态估计的可靠性。

**💡 创新点**

创新点在于使用图像翻译模型将合成图像的外观转换为真实图像的外观，从而在不需要标注真实图像的情况下恢复姿态估计的覆盖率，并显著减少静默失败的比例。

**🔧 技术方法**

使用了图像翻译技术（如CycleGAN和DUNIT）来生成与真实图像外观相似的合成图像，并结合了合规预测方法来进行姿态估计。

**📊 数据集**

使用了SPEED+数据集，该数据集包含合成图像和两种真实图像域（Lightbox和Sunlamp），分别在不同的照明条件下获取。

**📈 对比分析**

与传统的合成图像校准方法相比，翻译后的校准方法在真实图像上的覆盖率显著提高，达到93.6%（Lightbox）和92.9%（Sunlamp），而合成校准的覆盖率在真实图像上显著低于指定水平，且静默失败率从43.5%降至3.3%。

**⚠️ 局限性**

限制在于翻译模型的有效性依赖于未标注真实图像的可用性，且当前的评估仅限于SPEED+数据集，未能涵盖其他数据集或不同的照明条件。

---

## 258. Uncontrolled source titles: When Nature, Science, and PNAS meet Scopus

**arXiv ID:** 2611.25627 | [PDF](https://arxiv.org/pdf/2611.25627v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 259. TRCF: Temporal-Reference Complementary Forecasting for Irregular Multivariate Time Series

**arXiv ID:** 2611.26351 | [PDF](https://arxiv.org/pdf/2611.26351v1)

**作者:** Xudong Zuo `[一作]` (University of Science and Technology of China), Yun-Bo Zhao `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种新的时间序列预测框架，称为时间参考互补预测（TRCF），旨在通过结合通道独立（CI）和通道依赖（CD）的方法来提高不规则多变量时间序列的预测性能。

**💡 创新点**

创新点在于通过ANCHOR和BRIDGE模块，增强了CI预测器的能力，使其能够利用其他变量的信息，同时保留每个变量的不规则历史。

**🔧 技术方法**

使用了时间聚合、历史补全和多层感知机等技术。

**📊 数据集**

在四个真实世界数据集上进行了实验，包括HumanActivity、USHCN、PhysioNet P12和MIMIC-III。

**📈 对比分析**

与APN方法相比，TRCF在主要基准比较中实现了最低的均方误差（MSE）和平均绝对误差（MAE），并且在推理内存峰值上减少了17.28%至87.98%。

**⚠️ 局限性**

限制在于实验使用的基准数据集固定，未测试对外部数据集的泛化能力，且在变量数量较多时，资源使用仍然存在瓶颈。

---

## 260. World Models Favor Learned Priors over Visual Evidence

**arXiv ID:** 2611.26366 | [PDF](https://arxiv.org/pdf/2611.26366v1)

**作者:** Han Xiao `[一作]` (Xiamen University), Qingqiang Wu `[通讯]` (Xiamen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了如何通过视觉校准来推断当前物理规则，并在不同初始状态下预测物体运动。

**💡 创新点**

提出了TailWorld基准和RIFT方法，能够在长尾动态下进行有效的规则推断和运动预测。

**🔧 技术方法**

使用了视觉校准技术和规则推断机制，结合了训练频率的控制。

**📊 数据集**

使用了TailWorld基准数据集，该数据集包含2D和3D的碰撞场景，以及真实世界的摆动视频数据集（IRIS和Delfys75）。

**📈 对比分析**

与其他方法（如CoPhyNet、GEPS、CoDA等）进行比较，RIFT在预测准确性和物理一致性方面表现优越，尤其是在稀有动态的预测上。

**⚠️ 局限性**

模型在处理训练不平衡和稀有规则时可能仍然存在一定的局限性，尤其是在数据稀缺的情况下。

---

## 261. Asymptotic Bounds for Codes in Toroidal Distance

**arXiv ID:** 2611.25990 | [PDF](https://arxiv.org/pdf/2611.25990v1)

**作者:** Pengjie Zhong `[一作]` (Central China Normal University), Jinquan Luo `[通讯]` (Central China Normal University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了在环形度量下，ℤ_q^n中编码的渐近界限，推导了基于经典Gilbert-Varshamov界限的显式渐近下界，并发展了环形度量的鞍点方法。

**💡 创新点**

提出了一种新的鞍点方法来处理环形度量，得到了渐近Gilbert-Varshamov下界和渐近Hamming型上界，并通过嵌入三元Hamming码获得了另一种渐近下界。

**🔧 技术方法**

使用了鞍点方法来分析环形球的生成函数，并结合经典的Gilbert-Varshamov界限进行比较。

**📊 数据集**

使用了ℤ_q中编码的渐近界限，特别是针对三元Hamming码的构造进行了分析。

**📈 对比分析**

通过数值比较将这些下界与基线进行比较，证明了在相应参数范围内它们是非平凡的，并且在渐近界限上表现出优越性。

**⚠️ 局限性**

限制在于对环形度量的渐近行为的理解仍然有限，尤其是在固定字母表大小和长度的情况下，关于环形码的渐近率-距离行为的研究仍需深入。

---

## 262. Antom 3-in-1 Transformer: A Payment Foundation Model

**arXiv ID:** 2611.25403 | [PDF](https://arxiv.org/pdf/2611.25403v1)

**作者:** Weiyu Han `[一作]` (Ant International), Tianyi Zhang `[通讯]` (Ant International)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一种支付基础模型，集成了序列、表格和图形表示，用于支付风险检测。

**💡 创新点**

创新点在于模型能够跨场景重用，支持有限标签学习，并结合行为分析来提高风险检测能力。

**🔧 技术方法**

使用了序列对比学习和自监督学习技术，结合了Transformer和图注意力网络。

**📊 数据集**

主要使用了Ant International的内部支付数据进行预训练。

**📈 对比分析**

通过跨业务转移实验，预训练模型在目标业务指标上提高了34.5%，并且在仅使用25%训练标签的情况下，模型在平均精度(AP)和召回率(Recall@0.5%)上超越了全标签的基线模型。在生产环境中，相较于XGB-Online，模型将整体损失率降低了37.88%。

**⚠️ 局限性**

模型的局限性在于未详细介绍模型架构、训练配置和数据集构建过程，且图编码器在微调过程中被冻结，可能限制了其适应性。

---

## 263. Learning from Context: Leave-One-Out Prediction in Jump Diffusions

**arXiv ID:** 2611.26359 | [PDF](https://arxiv.org/pdf/2611.26359v1)

**作者:** Christian Wald `[一作]` `[通讯]` (INSA Lyon), Christian Wald (INSA Lyon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了一种连续扩散模型，该模型通过随机动态生成离散数据，特别关注跳跃扩散模型，并引入了留一预测（LOO）方法来提高生成文本和分子图的质量。

**💡 创新点**

创新点在于提出了一种通过留一预测来鼓励有用的上下文学习的方法，并展示了在保持概率路径的同时如何利用LOO预测来改善生成性能。

**🔧 技术方法**

使用了跳跃扩散模型和分析似然完成技术，结合了LOO预测和完整后验的漂移。

**📊 数据集**

使用了Text8、LM1B和ZINC250k数据集进行评估。

**📈 对比分析**

与无约束预测、原始LOO和完成预测进行了比较，结果表明LOO变体在所有三个数据集上都提高了关键生成指标，尤其是在文本生成任务中表现优异。

**⚠️ 局限性**

限制在于固定源架构的限制性，普通自注意力机制在某些情况下表现更好，未来的工作可以探索更具表现力的架构以保持LOO不变性。

---

## 264. A version of the Cloude/van Zyl decomposition for full/quad pol synthetic aperture radar data

**arXiv ID:** 2611.26195 | [PDF](https://arxiv.org/pdf/2611.26195v1)

**作者:** Allan Aasbjerg Nielsen `[一作]` (Technical University of Denmark), Knut Conradsen `[通讯]` (Technical University of Denmark)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种针对全极化合成孔径雷达数据的Cloude/van Zyl分解版本，并在丹麦Vejers的X波段F-SAR数据上进行了演示。

**💡 创新点**

创新点在于为全极化数据提供了一种新的分解方法，特别是在处理明亮的人造目标时，能够更好地聚焦于主要散射机制，并降低熵值。

**🔧 技术方法**

使用了Cloude/van Zyl分解和Cloude/Pottier α̅角度的技术，结合复杂协方差矩阵和Hermitian矩阵的计算。

**📊 数据集**

使用了来自德国宇航中心（DLR）的F-SAR系统的X波段数据，覆盖了丹麦西海岸Vejers的一个小区域。

**📈 对比分析**

与原始的反射对称Cloude/van Zyl分解相比，新的全极化版本在处理明亮的人造目标时表现出更低的熵值，表明目标散射机制的混乱程度降低，提供了更清晰的散射描述。

**⚠️ 局限性**

限制在于全极化版本的特征向量不再直接与特定的散射机制（如单次反射、双次反射或体积散射）相关联，这可能会挑战分解后的解释。

---

## 265. Planning to Stop: Factored Certification and Terminal Repair for Noncircular Robots

**arXiv ID:** 2611.26167 | [PDF](https://arxiv.org/pdf/2611.26167v1)

**作者:** Runtian Zheng `[一作]` (North China University of Technology), Ying Liu `[通讯]` (North China University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出了一种名为GeoDiff的运动规划器，旨在解决机器人在狭小空间中导航时的障碍物清除和目标停止问题。

**💡 创新点**

GeoDiff通过结合原始目标停止集的包含、见证引导的终端修复和分解证明编译，创新性地提高了在有限预算下的认证覆盖率。

**🔧 技术方法**

使用了运动规划、几何验证和证明重用等技术，特别强调了终端几何和请求局部重用的结合。

**📊 数据集**

使用了从CUBI-DT和公共BARN地图派生的任务数据集，共计11440个任务。

**📈 对比分析**

与基线方法相比，GeoDiff在15秒内认证了61/139个任务，而基线方法仅认证了44/139个任务，显示出显著的性能提升。

**⚠️ 局限性**

限制在于GeoDiff的性能可能受到预算和容忍度的影响，且在某些情况下修复可能会与其他搜索分支竞争时间。

---

## 266. When Do Rankings Beat Signs? Gradient Direction Recovery under Even Nuisance

**arXiv ID:** 2611.26055 | [PDF](https://arxiv.org/pdf/2611.26055v1)

**作者:** Zhuangzhuang Pan `[一作]` (Universiti Malaya), Chee Seng Chan `[通讯]` (VinUniversity)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了基于比较的优化方法，提出了一种新的算法框架，通过反向交叉对比来优化目标函数，而无需估计海森矩阵。

**💡 创新点**

创新点在于通过反向交叉对比消除了共享的干扰偏移，从而在不需要海森矩阵估计的情况下，提供了相对梯度投影的约束。

**🔧 技术方法**

使用了一种几何框架来表征候选梯度方向，并结合了反向交叉对比的方法。

**📊 数据集**

使用了来自加州大学欧文分校（UCI）机器学习库的合成数据集，包括空气动力学和混凝土强度等任务。

**📈 对比分析**

与传统的基于符号的方法相比，反向方法在多个反馈轮次中表现出更低的终端相对优化差距，但在某些情况下未能证明Full方法在优化上的额外优势。

**⚠️ 局限性**

限制在于固定中心约束无法在迭代间转移，且优化收敛性、可观察的半径选择和联合最优性仍然是开放问题。

---

## 267. Copilot or Culprit? How AI Is Changing Open-Source Bug Reporting and Maintenance

**arXiv ID:** 2611.25605 | [PDF](https://arxiv.org/pdf/2611.25605v1)

**作者:** Yinuo Liu `[一作]` (University of Hong Kong), Hao Chen `[通讯]` (University of Hong Kong)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对2024年1月至2026年9月期间来自101个开源项目的120,918个错误报告进行了大规模研究，探讨了AI在错误发现、报告、验证和解决过程中的参与情况。

**💡 创新点**

研究的创新点在于系统性地分析了AI参与的错误报告的可靠性演变，揭示了AI在错误报告和维护中的作用，以及如何将AI生成的发现转化为可验证的报告。

**🔧 技术方法**

使用了大规模数据分析技术，包括LLM（大型语言模型）进行错误报告的标注和分析，以及Firth逻辑回归模型来评估报告的有效性。

**📊 数据集**

数据集包含来自101个开源项目的120,918个错误报告，涵盖多种编程语言，包括C、C++、Python、Java等。

**📈 对比分析**

与其他方法相比，AI参与的报告在信息完整性上表现更好，但在有效性上并未显著优于其他报告。AI辅助的重现和诊断与更高的有效性相关，而AI发现的报告则较少受到维护者的关注。

**⚠️ 局限性**

研究的局限性在于AI生成的报告仍需验证，且AI发现的报告往往缺乏人类的后续关注，导致许多报告未能得到明确的结论。

---

## 268. Enhancing Understanding of Language Models with Varied Explanations

**arXiv ID:** 2611.25314 | [PDF](https://arxiv.org/pdf/2611.25314v1)

**作者:** Jingyi Sun `[一作]` `[通讯]`, Jingyi Sun

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文研究了语言模型在复杂推理任务中的可解释性，提出了多种解释技术以帮助用户理解模型的决策过程。

**💡 创新点**

创新点在于提出了两种评估框架来评估高亮解释的质量，并提出了生成自然语言解释的技术，这些解释既忠实于模型的预测过程，又易于人类理解。

**🔧 技术方法**

使用了高亮解释、自然语言解释和链式推理等多种解释技术。

**📊 数据集**

论文中未具体提及使用的数据集。

**📈 对比分析**

通过评估框架比较不同解释形式的质量，发现不同形式在任务中表现出不同的优势和局限性，且没有单一的解释形式或评估指标能够适用于所有任务。

**⚠️ 局限性**

论文的局限性在于，尽管提出了多种评估框架和解释技术，但在满足多个质量维度的同时生成解释仍然具有挑战性。

---

## 269. Learn, Freeze, and Transfer an Activation Function Across Architectures

**arXiv ID:** 2611.26201 | [PDF](https://arxiv.org/pdf/2611.26201v1)

**作者:** Arun Prakash A `[一作]` `[通讯]` (Indian Institute of Technology Madras), Arun Prakash A (Indian Institute of Technology Madras)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种新的转移学习方法，探讨了学习到的激活函数是否可以在从头训练的网络中转移并保持有效性。

**💡 创新点**

提出了一种学习-冻结-转移的协议，允许在单次训练中与源网络共同优化激活函数，然后将其固定并在目标网络中重用。

**🔧 技术方法**

使用了一种五参数的激活函数，通过二次谐波截断傅里叶级数表示，并在视觉变换器（ViT）上进行优化。

**📊 数据集**

使用了ImageNet-1K数据集进行源网络训练，并在Fashion-MNIST、CIFAR-10、CIFAR-100和Food-101等四个图像分类数据集上进行评估。

**📈 对比分析**

与十种传统激活函数进行比较，在相同的训练配置下，所提出的激活函数在16种设置中有15种表现出最高的平均准确率，最高提升达到1.71个百分点。

**⚠️ 局限性**

限制在于只使用了一个源学习的激活函数，且跨架构实验仅限于相对较小的目标网络和图像分类任务，未来需要进一步研究激活函数的可转移性在不同架构和任务中的变化。

---

## 270. Invertible Transport Between Sample-Only Distributions via Projected Characteristic Functions

**arXiv ID:** 2611.26187 | [PDF](https://arxiv.org/pdf/2611.26187v1)

**作者:** Juan Gutiérrez `[一作]` (Universidad Politécnica de Madrid), José Luis Blanco-Murillo `[通讯]` (Universidad Politécnica de Madrid)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无密度的框架PCF，用于学习数据分布与仅通过样本指定的先验分布之间的可逆映射。

**💡 创新点**

创新点在于通过投影经验特征函数进行分布对齐，消除了密度估计、对抗训练或配对观察的需求。

**🔧 技术方法**

使用了投影特征函数匹配技术，结合神经常微分方程（ODE）作为可逆映射的参数化。

**📊 数据集**

在控制传输、高维对齐任务和原始像素数据集（如MNIST和USPS）上进行了基准测试。

**📈 对比分析**

与传统的MMD方法相比，PCF在13个配对运行中表现出更高的样本效率，且在批量大小线性成本下匹配了MMD的准确性。

**⚠️ 局限性**

限制在于没有提供唯一的传输映射，且在没有配对观察或额外结构假设的情况下，可能无法恢复唯一的真实或最优传输映射。

---

## 271. TableRoute: Capability-Aware Routing for Scalable Table Question Answering

**arXiv ID:** 2611.25853 | [PDF](https://arxiv.org/pdf/2611.25853v1)

**作者:** Yuxiang Wang `[一作]` (University of Melbourne), Jianzhong Qi `[通讯]` (University of Melbourne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种可扩展的表格问答框架，结合了能力感知路由和证据条件推理，能够根据问题、表格和基础模型生成验证的证据和所需操作，并选择直接回答、表格简化或程序执行。

**💡 创新点**

创新点在于通过能力感知路由和证据条件推理的结合，提升了表格问答的准确性，同时减少了在线令牌消耗。

**🔧 技术方法**

使用了能力感知路由和证据条件推理技术，结合了验证的证据生成和操作识别。

**📊 数据集**

使用了多个数据集，包括WikiTableQuestions、WTQ-L和BirdQA，涵盖了不同规模的表格和推理设置。

**📈 对比分析**

与现有方法相比，提出的方法在所有基准测试中均表现出更高的准确性，尤其是在处理长表格和大规模表格时，显著减少了语义不匹配和计算错误。

**⚠️ 局限性**

需要一次性校准以建立每个基础模型的可靠范围，且在操作模式上将细粒度操作压缩为两个路由类别，可能忽略某些操作模式的区别。

---

## 272. Thinking in Streams: Proactive Spatial Reasoning with Vision-Language Models

**arXiv ID:** 2611.25865 | [PDF](https://arxiv.org/pdf/2611.25865v1)

**作者:** Junyoung Hong `[一作]` (Yonsei University), Seon Joo Kim `[通讯]` (Yonsei University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文研究了在流媒体视频中进行主动空间推理的能力，提出了ProSpace框架和ProSpace-Bench基准，评估模型在视频流中何时响应空间请求。

**💡 创新点**

创新点在于提出了主动空间推理的概念，并引入了ProSpace-VLM模型，该模型通过将显式几何编码注入到VLM的中间层来支持空间决策。

**🔧 技术方法**

使用了ProSpace-VLM模型，该模型利用现有的KV缓存作为空间记忆，并通过因果估计的3D点和相机姿态来注入几何编码。

**📊 数据集**

使用了ProSpace-Train数据集，该数据集包含10万个流媒体样本，用于监督训练响应时机和弃权。

**📈 对比分析**

与现有的VLM模型相比，ProSpace-VLM在及时响应和正确弃权方面表现更好，缩小了与人类的差距，尤其在需要跨视角关联观察的任务中表现突出。

**⚠️ 局限性**

ProSpace的局限性在于其专注于响应时机，而未能联合评估更高层次的推理或物理动作。此外，基于规则的情节构建可能无法捕捉现实世界中遇到的空间条件的多样性。

---

## 273. VG-Steer: Visual-Geometric Policy Steering from In-Context Aligned Abstractions

**arXiv ID:** 2611.25199 | [PDF](https://arxiv.org/pdf/2611.25199v1)

**作者:** Arthur Bucker `[一作]` (Robotics Institute Carnegie Mellon University), Jean Oh `[通讯]` (Robotics Institute Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种视觉几何策略引导框架，通过从成功的分布内演示中提取对象关系距离对，来改善机器人在分布外环境中的表现，而无需更新策略权重。

**💡 创新点**

创新点在于通过显式的视觉几何抽象来传递行为，捕捉细微的运动，而不仅仅是粗略的端点，从而提高了在分布外设置中的成功率。

**🔧 技术方法**

使用了大型视觉模型（LVM）和视觉语言模型（VLM）进行推理，结合任务相关的实体对及其对象关系距离投影，生成一个定制的视觉几何引导信号。

**📊 数据集**

在LIBERO-Pro模拟基准和真实的UR5机器人实验中进行了评估，展示了在姿态和任务目标变化下的性能提升。

**📈 对比分析**

与VLS等现有方法进行比较，提出的方法在任务和位置扰动下的成功率显著提高，且在真实环境中表现出更好的适应性，成功率翻倍。

**⚠️ 局限性**

限制在于感知的可靠性，依赖于对任务相关对象的检测、标记和定位，感知错误会显著降低性能。此外，子轨迹切换和引导与策略行为之间的平衡也是挑战。

---

## 274. Stuck in Place: Stall Diagnosis and Remedies for JEPA Planners

**arXiv ID:** 2611.25415 | [PDF](https://arxiv.org/pdf/2611.25415v1)

**作者:** Junhyeok Lee `[一作]` (Seoul National University), Kyu Sung Choi `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Unstick的方法，旨在解决JEPA规划器在推送任务中由于规划循环固定点导致的失败，特别是物体停滞不前的问题。

**💡 创新点**

创新点在于通过固定权重的循环打破器和在线微调相结合，解决了模型在分布变化下的适应性问题，显著提高了推送任务的成功率。

**🔧 技术方法**

使用了梯度下降（GD）和交叉熵方法（CEM）进行模型预测控制，并结合了残差记忆和随机重初始化技术。

**📊 数据集**

使用了PushObj和PushT数据集进行实验，测试了不同形状和视觉条件下的推送任务。

**📈 对比分析**

与AdaJEPA等方法进行了比较，Unstick在推送任务中恢复了大部分适应性增益，尤其是在形状变化下表现良好，但在视觉变化下效果较差。

**⚠️ 局限性**

限制在于该研究主要集中在一种JEPA模型及其在三个基准上的表现，且长时间的推送任务需要有限的记忆，微调需要在评估前进行额外的基准执行。

---

## 275. An Open-Source ROS 2 Framework for Shared Control in Surgical Teleoperation: Integrating the Haply Inverse3 with the da Vinci Research Kit

**arXiv ID:** 2611.26485 | [PDF](https://arxiv.org/pdf/2611.26485v1)

**作者:** Balint Varga `[一作]` (Karlsruhe Institute of Technology), Peter Galambos `[通讯]` (Óbuda University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究提出了一个开源的ROS 2框架，将Haply Inverse3触觉设备与da Vinci研究工具包（dVRK）集成，旨在改善外科手术远程操作中的共享控制。

**💡 创新点**

创新点在于将紧凑的三自由度（3-DOF）触觉设备与6-DOF的dVRK平台进行端到端集成，并提出了一种新的映射策略，将设备输入与患者侧操控命令结合。

**🔧 技术方法**

使用了ROS 2架构，结合了Haply Inverse3触觉设备和VerseGrip方向追踪，支持力反馈和位置控制模式。

**📊 数据集**

在用户研究中使用了虚拟热丝基准任务进行评估，参与者在模拟环境中完成任务。

**📈 对比分析**

与无辅助手动控制相比，线性引导显著减少了接触次数和任务完成时间，而渐进刚度引导虽然有一致的趋势，但统计上不显著。参与者的主观工作负载、控制感和满意度在引导模式下表现出积极的趋势。

**⚠️ 局限性**

限制在于热丝基准任务并不直接验证外科手术技能，且视觉反馈为单眼2D，缺乏立体深度线索，可能导致对触觉线索的依赖增加。参与者的经验差异和短暂的培训阶段也可能增加了个体间的变异性。

---

## 276. Twice-Told Critiques: Claim-Level Homogenization of Peer Review in the LLM Era

**arXiv ID:** 2611.26286 | [PDF](https://arxiv.org/pdf/2611.26286v1)

**作者:** Ilya Gusev `[一作]` `[通讯]` (Keenable), Ilya Gusev (Keenable)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文测量了会议和期刊评审中的批评重叠，分析了随着LLM生成文本的普及，评审者是否仍然提供独特的批评。

**💡 创新点**

创新点在于通过MELD检测器分析LLM生成的文本对评审批评多样性的影响，发现LLM标记的评审之间的重叠性高于未标记的评审，表明批评的多样性正在下降。

**🔧 技术方法**

使用了MELD检测器来识别LLM生成的文本，并通过GPT-6 Luna进行批评提取和分组。

**📊 数据集**

数据集包括ICLR、NeurIPS、ICML等会议的评审，以及F1000、PLOS等期刊的评审，涵盖了多个领域的论文和评审。

**📈 对比分析**

通过比较不同类型评审的批评重叠，发现LLM标记的评审之间的重叠性显著高于未标记的评审，且在多个会议和期刊中均表现出相似的趋势。

**⚠️ 局限性**

限制在于LLM的测量可能存在提取和分组错误，且评审快照的比较仅限于特定阶段，可能影响结果的普遍性。

---

## 277. One View, One Action: Spatially Generalizable VLAs through Observation-Aligned Experts and Cross Graph Consistency

**arXiv ID:** 2611.26161 | [PDF](https://arxiv.org/pdf/2611.26161v1)

**作者:** Jinxi Liu `[一作]` (Sun Yat-sen University), Guangrun Wang `[通讯]` (X-Era AI Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的多视角视觉-语言-动作模型OVOA-VLA，旨在提高在不同相机视角和机器人初始状态下的空间泛化能力。

**💡 创新点**

创新点在于通过在每个视角的相机框架中预测动作，利用相同的物理运动提供多样的观察对齐监督，从而改善基础框架策略的泛化能力。

**🔧 技术方法**

使用了Mixture of Transformers架构，结合了两个互补的定向注意图来建模基础专家和视角专家之间的交互，并引入了交叉图一致性来增强预测一致性。

**📊 数据集**

在多个仿真基准（如LIBERO和LIBERO-Plus）和真实世界任务（如Franka Research 3机器人）上进行了实验验证。

**📈 对比分析**

与现有的VLA基线模型相比，OVOA-VLA在LIBERO-Plus上实现了88.1%的平均成功率，而基础仅推理模式下也达到了85.7%。在真实世界评估中，OVOA-VLA在空间分布变化下的成功率从28.3%提高到58.3%。

**⚠️ 局限性**

限制在于模型的复杂性和计算成本，尽管基础仅推理模式保持了与基线相似的推理成本，但全模式的计算需求显著增加。

---

## 278. SplitReAlign: Partition-Aware Representation Realignment for Split Federated Unlearning

**arXiv ID:** 2611.25301 | [PDF](https://arxiv.org/pdf/2611.25301v1)

**作者:** Jia Chen `[一作]` (Wuhan Textile University), Jun Bai `[通讯]` (Adelaide University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为SplitReAlign的框架，用于在分裂联邦学习中实现类特定的遗忘，解决了在异构表示空间和不对称更新约束下的知识遗忘问题。

**💡 创新点**

创新点在于构建了一个分区感知的表示重新对齐框架，通过共享语义锚点协调端点特定的判别子空间，选择性地重新对齐遗忘的表示，同时限制对保留特征的变化。

**🔧 技术方法**

使用了分区感知优化、适应性损失校准和跨端点指导等技术，结合了来自冻结服务器的梯度和固定客户端特征统计。

**📊 数据集**

在CIFAR-10和CIFAR-100数据集上进行了实验，采用了类-wise的Dirichlet分布（α=0.1）进行数据分配。

**📈 对比分析**

与其他方法相比，SplitReAlign在六种设置下实现了最低的平均重训练差距（AG），特别是在Server-only模式下，与重训练的结果最接近，显示出在受限更新下分区感知指导的有效性。

**⚠️ 局限性**

限制在于该方法依赖于分裂联邦学习的特定结构，可能在其他类型的联邦学习或不同的模型架构中表现不佳。

---

## 279. Matching Architecture to Workload: Comparing RISC-V Parallel Accelerators for Energy-Efficient Embedded Computing

**arXiv ID:** 2611.25947 | [PDF](https://arxiv.org/pdf/2611.25947v1)

**作者:** Luca Colombo `[一作]` (Fondazione Chips-IT), Davide Rossi `[通讯]` (Università degli Studi di Pavia)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文比较了三种执行模型（标量多核、流式多核和向量处理）在能效嵌入式计算中的表现，使用三种开源RISC-V加速器（PULP Cluster、Snitch Cluster和Spatz）进行定量分析。

**💡 创新点**

创新点在于提供了一个全面的跨领域基准套件，并在相同的12纳米技术下对三种架构进行了公平的性能、功耗和面积比较，揭示了不同工作负载特性对执行模型的影响。

**🔧 技术方法**

使用了RISC-V架构的开源硬件和软件栈，进行了物理实现和比较，采用了基于工作负载特性的定量表征方法。

**📊 数据集**

使用了22个完全分析的基准内核，涵盖机器学习、数字信号处理、大语言模型和线性代数等领域。

**📈 对比分析**

比较结果显示，Spatz在计算密集型工作负载中表现最佳，性能提高了4.29倍，能效提高了5.17倍；Snitch Cluster在有限并行性和流式内存访问模式下表现良好，性能提高了5.53倍，能效提高了1.45倍；PULP Cluster在不规则和控制主导的应用中表现最佳，性能提高了5.66倍，能效提高了2.80倍。

**⚠️ 局限性**

限制在于没有单一的执行模型在所有工作负载中表现最佳，Spatz的性能波动较大，且在某些情况下表现低于其他两种架构。

---

## 280. Causal Overtaking in Fair Transaction Ordering

**arXiv ID:** 2611.26100 | [PDF](https://arxiv.org/pdf/2611.26100v1)

**作者:** Haozhe Feng `[一作]` (Zhejiang University), Jian Liu `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文识别了因果超越现象，提出了一个新的攻击模型，展示了如何通过多个客户端提交相同的响应交易来实现超越，从而在公平交易排序协议中造成问题。

**💡 创新点**

创新点在于定义了因果超越，并证明了仅依赖收据的规则无法始终保持早期提交的优先权，同时防止响应超越。提出了区域排序作为一种缓解措施。

**🔧 技术方法**

使用了区域排序的技术，通过在区域内快速转发交易并结合每个区域的排序意见来减少因果超越的发生。

**📊 数据集**

使用了来自AWS和WonderProxy的延迟测量进行模拟评估，测试了Themis协议的攻击和缓解效果。

**📈 对比分析**

与Themis协议的比较显示，在n=21的情况下，区域排序将最终攻击率减少了一半，同时在正常交易对中反转的比例低于2%。

**⚠️ 局限性**

限制在于区域排序假设每个区域内的腐败是有限的，全球故障预算可能导致腐败集中在某个区域，从而影响投票结果。

---

## 281. Extracting Value from Grey Literature: processes and technologies for aggregating and analyzing the hidden big data treasure of organizations

**arXiv ID:** 2611.26004 | [PDF](https://arxiv.org/pdf/2611.26004v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 282. Closing Performance Gap for Diffusion Language Models under Low Inference Budgets

**arXiv ID:** 2611.25763 | [PDF](https://arxiv.org/pdf/2611.25763v1)

**作者:** Hao Wu `[一作]` (University of Wisconsin--Madison), Yuwang Wang `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Commitment-aware Refinement and Recovery (CaRR) 的框架，旨在通过更新上下文补充和基于轨迹的恢复来提高固定可编辑检查点的提交质量。

**💡 创新点**

创新点在于将令牌重标记与基于轨迹的输出恢复相结合，以改善在有限推理预算下的生成质量。

**🔧 技术方法**

使用了一种训练无关的框架，结合了令牌承诺重标记和提交承诺恢复的技术。

**📊 数据集**

在代码生成和数学推理的四个基准上进行了实验，包括HumanEval、MBPP、GSM8K和MATH-500。

**📈 对比分析**

与原始的ME-DLM相比，CaRR在四个推理预算下的平均准确性均有所提高，尤其是在紧张预算下，表现出最大的增益。例如，在1/8的预算下，平均准确性提高了7.44个百分点。

**⚠️ 局限性**

限制在于该方法依赖于固定的可编辑检查点，可能在某些情况下无法充分利用所有生成的中间结果。

---

## 283. Steering Automated Scientific Discovery via Audited Epistemic Exploration

**arXiv ID:** 2611.25214 | [PDF](https://arxiv.org/pdf/2611.25214v1)

**作者:** Fangxu Yu `[一作]` (University of Maryland), Tianyi Zhou `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Epistemic Compass的框架，用于指导自动化科学发现，通过维护显式的认知状态来重新评估假设并生成新的假设。

**💡 创新点**

创新点在于通过构建一个认知树来链接假设、证据、信念和修正，明确跟踪未解决的冲突和未探索的方向，从而指导假设生成和最终验证。

**🔧 技术方法**

使用了基于大型语言模型（LLM）的策略，结合了信念更新、假设选择和自我验证等技术。

**📊 数据集**

在DiscoverPhysics、ActiveSciBench-Chem和NewtonBench等三个科学发现基准上进行了实验。

**📈 对比分析**

与多个基线方法进行比较，Epistemic Compass在所有基准上均表现出更好的发现性能，减少了不支持的声明和信念修正失败，具体表现为在DiscoverPhysics上减少了高达94.1%的归一化轨迹误差。

**⚠️ 局限性**

限制在于该方法依赖于底层LLM的能力，可能继承其局限性和偏见，且自动化发现的结果仍需专家验证。

---

## 284. A Practical Proprioceptive Data Collection Utility for the Unitree GO1

**arXiv ID:** 2611.26146 | [PDF](https://arxiv.org/pdf/2611.26146v1)

**作者:** Jack Anders Smitterberg `[一作]`, Tan Chen `[通讯]` (State University of New York at Binghamton)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种开源的图形化工具，帮助研究人员从Unitree Go1机器人中可靠地记录本体感知数据，包括关节状态、IMU读数、足部力传感器数据和陀螺仪测量，无需ROS知识。

**💡 创新点**

创新点在于提供了一个易于使用的GUI工具，使得任何研究人员都能轻松收集Go1的本体感知数据，降低了数据收集的门槛。

**🔧 技术方法**

使用了C++后端与Unitree SDK直接交互，并通过Python图形用户界面提供控制，避免了对ROS的依赖。

**📊 数据集**

使用了51个记录的多地形数据集，涵盖了十种不同的地形类型。

**📈 对比分析**

与传统的ROS1日志记录方法相比，未进行定量比较，但工具设计旨在降低Go1研究的门槛。性能方面，TCN模型在持出记录上达到了约38%的准确率，而在窗口级别分割下达到了约99%的准确率。

**⚠️ 局限性**

限制在于验证仅为演示，持出记录集较小，且地形与收集日期和步态混淆，无法测量数据包丢失，且未与其他数据收集方法进行定量比较。

---

## 285. CryoGaussian: Heterogeneous Cryo-EM Reconstruction Made Explicit with Gaussian Splatting

**arXiv ID:** 2611.26128 | [PDF](https://arxiv.org/pdf/2611.26128v1)

**作者:** Ziwen Wang `[一作]` (Chinese University of Hong Kong), Ge Liu `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为CryoGaussian的高效Gaussian splatting方法，用于异质性冷冻电子显微镜（cryo-EM）重建，直接将显式的Gaussian密度表示拟合到粒子图像中。

**💡 创新点**

CryoGaussian的创新点在于它在没有先验结构信息的情况下，从随机初始化开始，利用数据导出的约束来确定Gaussian的数量，并通过潜在条件的Gaussian来建模组成和构象异质性。

**🔧 技术方法**

使用了Gaussian splatting技术，结合了可微分光栅化和投影调度，以提高训练效率和重建精度。

**📊 数据集**

使用了多个数据集进行评估，包括IgG-1D、IgG-RL、Ribosembly、Tomotwin-100、Spike-MD以及实验数据集EMPIAR-10076和EMPIAR-10180。

**📈 对比分析**

与现有方法（如CryoSplat、CryoSPIRE等）进行比较，CryoGaussian在所有五个数据集上都达到了最高的平均FSC AUC，显示出在构象和组成异质性重建中的优越性能。

**⚠️ 局限性**

限制在于CryoGaussian的性能可能受到输入数据质量的影响，尤其是在信噪比低的情况下，可能会导致重建结果的不确定性。

---

## 286. Decision-Focused Reinforcement Learning for LLM Post-Training

**arXiv ID:** 2611.25597 | [PDF](https://arxiv.org/pdf/2611.25597v1)

**作者:** Haotian Zhai `[一作]` (University of Texas at Austin), Leqi Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种决策聚焦的强化学习框架（DFRL），用于后训练大型语言模型（LLMs），通过下游决策的负后悔作为奖励来预测优化参数。

**💡 创新点**

DFRL框架允许在不需要通过下游优化器进行微分的情况下，直接使用负后悔作为奖励进行训练，解决了传统决策聚焦学习中的微分挑战。

**🔧 技术方法**

使用了强化学习（RL）技术，特别是通过负后悔作为奖励反馈进行模型训练，同时开发了启发式热启动和奖励修正两种方法来提高训练效率。

**📊 数据集**

在合成优化问题和基于LLM的库存规划任务中进行了评估，合成问题包括最短路径、背包问题和二次资源分配等。

**📈 对比分析**

与传统的预测模型（如平方误差和绝对误差）进行比较，DFRL在所有合成基准测试中实现了最低的平均后悔，并在库存规划中相较于最强的传统预测基线减少了12.4%的后悔。

**⚠️ 局限性**

DFRL在计算奖励时可能面临瓶颈，因为每个采样预测都需要解决下游优化问题，尽管提出了启发式热启动和奖励修正来缓解这一问题。

---

## 287. Rate-distortion Assessment for Hierarchical Content Summarization

**arXiv ID:** 2611.25151 | [PDF](https://arxiv.org/pdf/2611.25151v1)

**作者:** Yunye Gong `[一作]` (SRI International), Ajay Divakaran `[通讯]` (SRI International)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种基于信息编码的问题，形式化了层次内容摘要，旨在定量评估生成内容操作的质量，特别是在压缩效率与信息损失之间的权衡。

**💡 创新点**

创新点在于将层次内容摘要视为信息编码问题，并设计了一个框架来提取源内容的语义层次，从而支持对生成内容操作的定量评估。

**🔧 技术方法**

使用了信息理论中的率-失真理论作为基础，结合了大语言模型（LLMs）进行内容摘要和问答评估。

**📊 数据集**

使用了QuALITY数据集，该数据集包含300个长英文上下文段落，每个段落平均长度约为5000个标记，共有2523个问答样本。

**📈 对比分析**

通过与现有基线方法进行比较，展示了所提出框架在内容摘要任务中的有效性，结果表明该框架能够提供更好的率-失真特性，超越了现有的评估基准。

**⚠️ 局限性**

限制在于该框架的应用主要集中在文本理解任务，未来需要探索其在其他生成和操作任务中的适用性。

---

## 288. Minimal Memory for Correctable Visual Continuation

**arXiv ID:** 2611.25675 | [PDF](https://arxiv.org/pdf/2611.25675v1)

**作者:** Xiyang Zhang `[一作]` (Harbin Institute of Technology), Yan Song `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文研究了在已知动作和过去图像的基础上，如何生成未来帧的视觉延续，特别关注在对象重新出现或视图变化时如何保持状态的可修正性。

**💡 创新点**

创新点在于建立了稀疏和密集记忆的匹配定律，证明了在静态流中，修正所需的开销在一维中至少是对数级别，而在高维中是多项式级别。

**🔧 技术方法**

使用了几何摘要和历史存在约束的技术，构建了精确更新的有限位几何摘要，并通过联合见证来降低物理风险。

**📊 数据集**

使用了COIL-20数据集进行实验，进行了几何诊断、图像测试和神经状态恢复的评估。

**📈 对比分析**

通过与现有系统的比较，展示了联合见证在降低最坏真相图像风险方面的严格优势，控制神经延续能够恢复当前视图中隐藏的位置。

**⚠️ 局限性**

限制在于在某些情况下，历史信息的压缩可能导致无法恢复某些中间候选状态，影响最终的图像质量。

---

## 289. OceanCoder: A Verification-Guided Large Language Model for Automated Simulation Modelling in Ocean Engineering

**arXiv ID:** 2611.26333 | [PDF](https://arxiv.org/pdf/2611.26333v1)

**作者:** Can Xue `[一作]` (Ocean University of China), Bingchen Liang `[通讯]` (Ocean University of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了OceanCoder，一个基于Qwen3-8B的大型语言模型，用于海洋工程中的自动化仿真建模，旨在通过自然语言描述生成完整的OrcaFlex程序。

**💡 创新点**

创新点在于开发了一个可扩展的数据构建管道，生成了12000个经过验证的监督微调示例，并引入了三层验证框架，以确保生成模型的工程可行性，并将验证结果作为训练反馈。

**🔧 技术方法**

使用了Qwen3-8B模型，结合了监督微调（SFT）和在线组相对策略优化（GRPO）等强化学习技术。

**📊 数据集**

构建了一个包含12000个示例的数据集，涵盖12个工程任务类别，此外还建立了665个任务的基准测试集，用于评估模型的泛化能力和实际应用性。

**📈 对比分析**

与基础模型和领先的闭源模型进行了比较，OceanCoder在任务完成率（Pass@1）上从基础模型的0%提升至SFT后的46.77%，再到GRPO后的51.43%。

**⚠️ 局限性**

限制在于当前模型的性能仍然依赖于任务的复杂性，尤其是在结构修改和更广泛的工程场景中，仍然面临较大的挑战。

---

## 290. Posterior Matching Distillation

**arXiv ID:** 2611.26629 | [PDF](https://arxiv.org/pdf/2611.26629v1)

**作者:** Hidir Yesiltepe `[一作]` (Virginia Tech), Pinar Yanardag `[通讯]` (Virginia Tech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的后验匹配蒸馏方法（PMD），旨在通过压缩预训练教师的迭代采样过程来加速图像和视频生成。

**💡 创新点**

PMD结合了模式覆盖后验采样和任意步生成，能够在零人口损失下恢复教师在每个训练噪声水平下的完整后验分布，且无需额外的学生评分网络。

**🔧 技术方法**

使用了后验匹配蒸馏（PMD）技术，该技术通过指数加权平均的学生样本与教师后验均值进行匹配。

**📊 数据集**

在SDXL和FLUX.2 [klein] 4B数据集上进行了实验，展示了PMD在多种指标上的优越性能。

**📈 对比分析**

与其他方法（如DMD2和RTDMD）相比，PMD在FID等指标上表现最佳，同时在提示对齐和感知质量指标上保持竞争力。

**⚠️ 局限性**

PMD在训练过程中需要多个学生抽样，并且仍依赖于辅助对抗损失来进一步提高视觉真实感。

---

## 291. StreamEgo: Streaming Exocentric-to-Egocentric Video Generation

**arXiv ID:** 2611.26165 | [PDF](https://arxiv.org/pdf/2611.26165v1)

**作者:** Tianyi Xu `[一作]`, Boxin Shi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种实时的流式框架StreamEgo，用于将第三人称视频转换为第一人称视角，解决了现有方法在交互区域、过拟合和固定长度剪辑方面的局限性。

**💡 创新点**

创新点在于引入了流式双分支先验和区域感知训练课程，分别提高了手-物体交互的保真度和生成的泛化能力。

**🔧 技术方法**

使用了流式双分支先验、区域感知损失课程和块状自回归模型等技术，支持实时流式生成。

**📊 数据集**

使用了Ego-Exo4D数据集进行训练和评估，包含4400个剪辑，每个剪辑持续5秒，帧率为24 FPS。

**📈 对比分析**

与现有的Exo2Ego基线（如EgoX）进行比较，StreamEgo在图像和对象级别的指标上表现优越，尤其在手部运动准确性方面显著提高，生成效率达到24.3 FPS。

**⚠️ 局限性**

限制在于模型依赖于准确的目标第一人称相机轨迹，尽管可以通过估计轨迹来实现，但准确性仍然是一个挑战。

---

## 292. Cognitive-Horizon Action Chunking: Memory-Guided Adaptive Reasoning for Embodied Navigation

**arXiv ID:** 2611.25587 | [PDF](https://arxiv.org/pdf/2611.25587v1)

**作者:** Weile Gong `[一作]` (Nanjing University of Posts and Telecommunications), Weibei Fan `[通讯]` (Nanjing University of Posts and Telecommunications)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为Cognitive-Horizon Action Chunking (CHAC) 的框架，用于在冻结的导航策略下进行记忆引导的自适应推理，旨在减少多模态模型的推理成本。

**💡 创新点**

CHAC的创新点在于结合了在线认知视野和历史引导的终端修正，能够在不进行额外模型调用的情况下，利用保留的视觉上下文和访问历史来指导推理时机和终端执行。

**🔧 技术方法**

使用了冻结的OmniNav-Fast导航策略，结合了视觉-语言模型（VLM）和历史数据进行推理。

**📊 数据集**

在HM3D-OVON和R2R-CE两个数据集上进行了评估，分别包含3000个和1839个指令跟随的场景。

**📈 对比分析**

与原生执行相比，CHAC在HM3D-OVON上成功率为946次（下降0.63%），减少了33.4%的VLM调用；在R2R-CE上成功率为46.22%（下降10.82%），减少了64.7%的调用。CHAC在多个物理导航任务中表现出更高的成功率和更低的推理成本。

**⚠️ 局限性**

CHAC的局限性在于成功率和路径长度加权成功率（SPL）存在任务依赖性损失，并且环境动作增加。物理延迟依赖于相机采集和执行的运动，终端修正依赖于策略的停止请求和局部访问几何。

---

## 293. TwinLab-Bench: Evaluating AI Scientists in Agentic Twins of Real Labs

**arXiv ID:** 2611.26248 | [PDF](https://arxiv.org/pdf/2611.26248v1)

**作者:** Yang Wan `[一作]` (Zhejiang University), Linchao Zhu `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出了TwinLab-Bench，这是一个包含52个任务的基准测试，旨在评估AI科学家设计实验的能力。

**💡 创新点**

创新点在于构建了一个真实实验室的双胞胎（TwinLab agent），能够为AI提供真实的实验数据，从而测试其设计实验的能力。

**🔧 技术方法**

使用了TwinLab agent作为环境，结合了语言模型和计算器来处理实验请求和返回结果。

**📊 数据集**

数据集来源于真实实验室的记录，包括开放实验室笔记、原始仪器导出和公开数据集。

**📈 对比分析**

与现有基准相比，TwinLab-Bench结合了真实问题和实验，提供了更广泛的实验范围和真实读数。测试结果显示，最强的前沿模型平均得分低于50分，表明现有模型在设计实验方面的能力不足。

**⚠️ 局限性**

限制在于当前的AI模型在实验设计和决策方面仍存在较大缺陷，尤其是在接受机制而未进行区分性测试时，常常导致错误。

---

## 294. Presenting Your Figures: Data-informed Animated Transitions Across Scientific Figure Panels

**arXiv ID:** 2611.26036 | [PDF](https://arxiv.org/pdf/2611.26036v1)

**作者:** Lu Ying `[一作]` (University of Vienna), Yong Wang `[通讯]` (Nanyang Technological University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本论文提出了一种新的方法，通过Figure Space和Transition Space来设计和生成科学图表面板之间的动画过渡，以帮助观众理解数据关系。

**💡 创新点**

创新点在于引入了Figure Space和Transition Space的概念，提供了一种统一的表示方法来捕捉多面板科学图表的组成，并设计关键帧以解释面板之间的数据关系。

**🔧 技术方法**

使用了图形学和数据可视化技术，结合了AI代理来生成动画过渡，并允许用户在生成过程中进行自然语言的修改。

**📊 数据集**

分析了48个来自不同科学领域的已发表的多面板图表，涵盖物理、化学和生物等领域。

**📈 对比分析**

通过与传统工具（如PowerPoint和ChatGPT）进行比较，用户研究显示该系统在作者创作和解释方面的评分显著更高，且整体工作负载评分显著更低。

**⚠️ 局限性**

限制在于该方法依赖于源数据和渲染代码，生成动画的时间仍然较长，且当前系统优先考虑2D联合坐标系，未来需要支持根据用户需求自定义坐标安排。

---

## 295. dLoop: Looped Diffusion Language Models via Recurrent Latent Refinement

**arXiv ID:** 2611.26275 | [PDF](https://arxiv.org/pdf/2611.26275v1)

**作者:** Handa Li `[一作]` (University of Surrey), Lu Yin `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的扩散语言模型架构dLoop，通过在不可逆承诺之前进行递归潜在计算来改进模型的预测能力。

**💡 创新点**

dLoop通过在每个去噪步骤中重复执行共享参数的中间块，允许模型在不推进到新的去噪步骤的情况下扩展内部处理，从而提高了生成的准确性和并行性。

**🔧 技术方法**

使用了递归深度模型和共享参数的Transformer层，结合了潜在计算的机制。

**📊 数据集**

在GSM8K、MATH、MBPP和HumanEval等多个基准数据集上进行了实验。

**📈 对比分析**

与Fast-dLLM、dParallel和d3LLM等方法进行比较，dLoop在所有基准上都达到了最高或并列最高的准确率，并且在准确性和每次前向传递的令牌数（TPF）上没有其他方法超过dLoop。

**⚠️ 局限性**

限制在于dLoop的TPF虽然高，但在某些基准上仍然低于d3LLM，表明在准确性和并行性之间存在权衡。

---

## 296. AGSR: Aperture-Guided Super-Resolution for Gaussian Splatting

**arXiv ID:** 2611.25934 | [PDF](https://arxiv.org/pdf/2611.25934v1)

**作者:** Ayman Alkhateeb `[一作]` (Alpen-Adria-Universität Klagenfurt), Christian Timmerer `[通讯]` (Alpen-Adria-Universität Klagenfurt)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为AGSR（Aperture-Guided Super-Resolution）的轻量级推理框架，通过在推理时收缩高斯协方差来缓解有限高斯足迹引起的带宽限制，从而实现超分辨率。

**💡 创新点**

创新点在于引入了一种带宽感知的高斯光栅化解释，并通过多孔径渲染生成结构不确定性场，以指导几何感知的超分辨率。

**🔧 技术方法**

使用了多孔径渲染、几何条件化和不确定性引导特征聚合等技术，构建了一个逐步融合的网络架构。

**📊 数据集**

在Mip-NeRF 360基准数据集上进行了实验，评估了AGSR在不同降解程度下的性能。

**📈 对比分析**

与轻量级图像空间超分辨率基线进行比较，AGSR在高斯表示恶化时性能提升显著，且在保持实时性能的同时，参数量仅为21K到139K。

**⚠️ 局限性**

限制在于AGSR依赖于预训练的3DGS模型，且在极端降解情况下的恢复能力仍需进一步研究。

---

## 297. Hierarchical Spatiotemporal World Modeling with Content-Adaptive Spatial Addressing

**arXiv ID:** 2611.25610 | [PDF](https://arxiv.org/pdf/2611.25610v1)

**作者:** Tiantian Li `[一作]` (QiYuanLab), Tianyi Qian `[通讯]` (QiYuanLab)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为HiSTFormer的分层时空世界模型，用于在部分可观察环境中进行长时间预测。

**💡 创新点**

创新点在于采用分层的时空组织和内容自适应的空间寻址，结合固定容量的内存结构，以提高长时间预测的效率和准确性。

**🔧 技术方法**

使用了分层Transformer架构，结合了局部时间RoPE和多周期空间RoPE，以及内容条件的地址偏移。

**📊 数据集**

在MemoryMaze和HM3D数据集上进行了实验。

**📈 对比分析**

与多个强基线模型进行比较，HiSTFormer在训练序列长度内和超出训练范围的预测中均表现优越，尤其在长时间预测中具有更好的泛化能力。

**⚠️ 局限性**

限制在于HiSTFormer假设头部方向的真实可靠性，其对噪声估计的鲁棒性和向下游任务的泛化能力尚待探索。

---

## 298. MorphTwin: Enriching Deformable Material Responses from Videos for Morphable and Interactive Physical Twins

**arXiv ID:** 2611.25963 | [PDF](https://arxiv.org/pdf/2611.25963v1)

**作者:** Haoqin Hong `[一作]` (University of Science and Technology of China), Rynson WH Lau `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为MorphTwin的方法，用于从视频中重建物体的物理双胞胎，增强了可变形材料响应的表现能力，同时保持高效的显式仿真。

**💡 创新点**

MorphTwin通过解耦体积和剪切建模，克服了传统弹簧模型在材料表现上的限制，允许独立控制材料的体积和形状响应。

**🔧 技术方法**

使用了离线校准和显式弹簧动力学的技术，结合了可放松的响应载体来增强局部变形模式。

**📊 数据集**

使用了合成场景和自捕获的自我中心交互基准数据集，包括20个PhysTwin场景和15个时空对齐的RGB-D序列。

**📈 对比分析**

与PhysTwin相比，MorphTwin在相同的逆设置下，Poisson比的平均绝对误差减少了75.8%，杨氏模量的平均相对误差减少了22.1%。在实时速度下，MorphTwin在重建精度和未来跟踪误差上表现更好。

**⚠️ 局限性**

当前方法主要集中在各向同性弹性和固定密度上，未来工作将整合前馈重建以简化物理双胞胎的创建，并扩展其应用范围。

---

## 299. Learning What Actions Change: Action Effect Matching for Dynamics-Faithful World Models

**arXiv ID:** 2611.25980 | [PDF](https://arxiv.org/pdf/2611.25980v1)

**作者:** John Won `[一作]` (KAIST), Jinwoo Shin `[通讯]` (RLWRLD)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种辅助训练目标，通过引入动作扰动的视觉变化分布来改进世界模型的学习，使其能够更好地预测小动作变化对未来观察的影响。

**💡 创新点**

创新点在于通过注入时间相关的噪声到无任务的机器人遥操作策略中，训练世界模型以响应这些噪声，从而提高模型对动作变化的敏感性。

**🔧 技术方法**

使用了条件流视频模型作为基础架构，并引入了新的辅助训练目标来监督模型对动作扰动的响应。

**📊 数据集**

使用了OGBench模拟环境和真实机器人DROID及ALOHA-2设置的数据集，收集了机器人与环境的交互数据。

**📈 对比分析**

通过与现有的机器人世界模型进行比较，提出的方法在DROID上提高了政策成功率的Pearson相关性，从0.47提升至0.79，并且FVD降低了最多23%。在规划任务中，使用改进后的世界模型选择动作的成功率提高了最多8.3%。

**⚠️ 局限性**

限制在于该方法依赖于收集的扰动数据的质量和多样性，可能在某些情况下无法完全捕捉复杂的环境动态。

---

## 300. Beyond Covers and Titles: Evidence-First Ranking for Robust Cold-Start Recommendation

**arXiv ID:** 2611.26249 | [PDF](https://arxiv.org/pdf/2611.26249v1)

**作者:** Fuyi Wang `[一作]` (Royal Melbourne Institute of Technology), Wei Song `[通讯]` (Griffith University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a2602d71-93ab-4bad-974b-672788df8193` `6215c339-3735-4be3-8a07-5bbb7004712d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

论文探讨了短视频平台如何在用户观看之前对新上传的视频进行排名，并提出了一种新的攻击和防御机制。通过模拟平台通道，展示了包装编辑如何在冷启动推荐中提高视频的曝光率。

**💡 创新点**

创新点在于提出了Evidence-First Ranking (EFR)方法，该方法通过视频内容对冷启动视频进行评分，并严格限制包装编辑的影响，从而在保持推荐质量的同时提供对任意包装编辑的可证明保证。

**🔧 技术方法**

使用了REAP（可实现的黑箱攻击）模型来模拟创作者的编辑操作，并通过支持门控残差来限制包装的影响，同时采用了多层感知机（MLP）进行评分和合并。

**📊 数据集**

使用了MicroLens-100K数据集，该数据集包含100,000个用户和19,738个视频，进行了广泛的实验以验证提出的方法的有效性。

**📈 对比分析**

与现有方法相比，EFR显著减少了包装攻击的效果，攻击增益降低了81%，同时在Cold-start Recall@100上从0.1133提高到0.170，合并后的NDCG@10保持在未防御系统的4%以内。

**⚠️ 局限性**

限制在于没有人类标注编辑，且REAP查询确切的代理评分，这可能有利于攻击者。此外，证据仅限于视频帧，且实验使用了单一种子进行变体测试，未来工作可以扩展到音频和语音的证据。

---

## 301. Running-Time Lower Bounds under FPT $\neq$ W[1]

**arXiv ID:** 2611.26080 | [PDF](https://arxiv.org/pdf/2611.26080v1)

**作者:** Hendrik Molter `[一作]` `[通讯]` (Hasso Plattner Institute), Hendrik Molter (Hasso Plattner Institute)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文重新审视了基于指数时间假设（ETH）的运行时间下界，探讨在假设FPTW[1]下可以恢复多少定量内容。研究通过编码图中的团选择来控制目标参数，并得出针对图、字符串和整数可行性问题的十二个确定性算法下界。

**💡 创新点**

创新点在于通过FPTW[1]假设探索运行时间改进的排除，而不需要假设源问题的特定指数。这为参数化复杂性理论提供了新的视角。

**🔧 技术方法**

使用了参数化复杂性理论中的标准工具，包括固定参数可解性（FPT）和参数化归约，结合了历史上已知的下界和新的假设。

**📊 数据集**

使用了多个数据集，包括图、字符串和整数可行性问题的实例，具体数据集未详细列出。

**📈 对比分析**

通过与历史ETH下界的比较，本文的下界在多个情况下与历史下界相匹配，且在某些情况下，所需的计算截止点是可计算的。性能上，本文的下界在某些情况下提供了更广泛的适用性。

**⚠️ 局限性**

限制在于，虽然本文提出了新的下界，但仍然依赖于FPTW[1]假设的有效性，且在某些情况下，可能无法完全覆盖所有参数化问题的复杂性。

---

## 302. A Brief Overview of Non-Terrestrial Networks (Updated 2026)

**arXiv ID:** 2611.25984 | [PDF](https://arxiv.org/pdf/2611.25984v1)

**作者:** Jose Alberto Hernandez `[一作]`, Pedro Reviriego `[通讯]` (Universidad Politécnica de Madrid)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对2023年预印本《非地面网络（NTN）卫星通信简要介绍》进行了扩展和更新，涵盖了传播、链路预算、频谱重用和相控阵列等基础知识，并修正了相关方程和数值示例。

**💡 创新点**

创新点在于更新了截至2026年11月18日的最新表格，区分了运营、发射和计划中的星座，并比较了已发布的固定终端速率与直接到设备服务的性能。

**🔧 技术方法**

使用了链路预算分析、相控阵列技术以及频谱重用等技术。

**📊 数据集**

数据集包括截至2026年11月18日的卫星星座部署和服务能力的最新数据，以及来自Cloudflare和ITU的流量和订阅数据。

**📈 对比分析**

通过与历史数据和其他研究的比较，展示了LEO宽带的快速增长和需求，但未提供直接的全球字节份额比较，性能表现出用户增长和请求增长的趋势。

**⚠️ 局限性**

限制在于需要独立的可重复测量来区分研究结果与商业目标的操作性能，以及在频谱共享、移动性、容量和安全性等方面的开放挑战。

---

## 303. Signed, Valid, and Unauthorized: Authority Closure at the Trust Boundary Between Agent Commerce Protocols and Crypto Wallets

**arXiv ID:** 2611.24989 | [PDF](https://arxiv.org/pdf/2611.24989v1)

**作者:** Shengwei You `[一作]` (University of Notre Dame), Jarek Nabrzyski `[通讯]` (University of Notre Dame)

**通讯引用:** 3026 | [OpenAlex ID](https://openalex.org/A5068300137)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文提出了一种新的钱包对象检查器Closure，用于确保软件代理在使用稳定币进行支付时遵循买方的授权限制。

**💡 创新点**

创新点在于定义了授权闭包的概念，并通过Closure实现了对13类签名对象的检查，确保钱包对象符合买方的约束条件。

**🔧 技术方法**

使用了钱包对象检查器Closure，该检查器基于授权闭包的定义，能够快速评估签名对象的合规性。

**📊 数据集**

评估使用了2130个生成的对象和62979个执行程序，涵盖了265个观察到的违规对象。

**📈 对比分析**

与现有方法相比，Closure拒绝了所有1680个观察到的违规对象，并接受了剩余的450个对象，决策时间中位数最多为每个类别12微秒。

**⚠️ 局限性**

限制在于生成的语料库是基于替代代币，且未能从实际部署中收集数据，可能导致对代理或商家产生的错误翻译的频率估计不足。

---

## 304. TwinD: Generalizing Deepfake Detection from Blending to Entire Face Synthesis

**arXiv ID:** 2611.25935 | [PDF](https://arxiv.org/pdf/2611.25935v1)

**作者:** Andrii Yermakov `[一作]` (Czech Technical University in Prague), Jiri Matas `[通讯]` (Czech Technical University in Prague)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的面部深度伪造检测器，旨在解决现有检测器在处理完整面部合成伪造时的性能不足。

**💡 创新点**

创新点在于引入了两种类型的伪造样本（重建双胞胎和斑点双胞胎），以提供互补的取证线索，从而提高检测器的泛化能力。

**🔧 技术方法**

使用了基于LoRA的预训练模型进行微调，并结合了重建双胞胎和斑点双胞胎的训练方法。

**📊 数据集**

使用了FaceForensics++数据集和Casual Conversations v2数据集，以及19个从2019到2026年发布的面部深度伪造数据集进行评估。

**📈 对比分析**

与15个现有检测器进行了比较，平均AUROC达96.3%，比之前的最佳检测器高出8.9个百分点，在每个数据集上至少达到90.4%。

**⚠️ 局限性**

局限性在于训练过程中未考虑噪声和下采样对检测性能的影响，且仅使用了一个自编码器生成重建双胞胎。

---

## 305. Efficient Privacy-Preserving High-Dimensional Approximate Nearest Neighbor Search

**arXiv ID:** 2611.25374 | [PDF](https://arxiv.org/pdf/2611.25374v1)

**作者:** Lyu Xu `[一作]` (Nanyang Technological University), Qichen Wang `[通讯]` (Nanyang Technological University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种高效的隐私保护高维近似最近邻搜索框架ObliSC，旨在保护用户查询的隐私，同时在公共向量数据库上进行搜索。

**💡 创新点**

创新点在于设计了一个单服务器的非对称隐私保护ANN搜索框架，能够在不需要第二个非合谋服务器的情况下，保护用户查询的隐私，并实现完全的隐私保护。

**🔧 技术方法**

使用了基于K均值的倒排多索引和完全隐私保护的搜索程序，结合了并行优化策略以加速候选项的检索和精炼，同时保持完全的隐私保护和检索质量。

**📊 数据集**

在六个标准的ANN基准数据集上进行了实验，包括Deep1M、Gist1M、Sift10M、Yandex Deep10M、Sift100M和Glove。

**📈 对比分析**

与非隐私和隐私保护的基线方法进行了比较，ObliSC在检索质量和效率上表现出竞争力，且其索引相对轻量。

**⚠️ 局限性**

局限性在于需要额外的存储和聚合工作以实现并行处理，同时在某些情况下可能会影响性能。

---

## 306. From Code Review Comments to Reuse in Production: An Industrial Study of Extracted Rules

**arXiv ID:** 2611.25918 | [PDF](https://arxiv.org/pdf/2611.25918v1)

**作者:** Mutsuha Asada `[一作]` (University of Tsukuba), Shinpei Hayashi `[通讯]` (Institute of Science Tokyo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文研究了Kleene系统在生产环境中提取的代码审查规则的使用情况，分析了这些规则在实际应用中的表现和有效性。

**💡 创新点**

创新点在于通过对提取规则的归因分析，揭示了规则命名与实际内容之间的关系，并指出了仅依赖命名计数无法准确反映规则的贡献。

**🔧 技术方法**

使用了大语言模型（LLM）进行代码审查和规则提取，结合了生产记录和审查评论的分析。

**📊 数据集**

数据集包括来自GA技术公司的19个代码库中合并的拉取请求的3579条审查评论，研究时间跨度为2024年2月28日至2026年8月14日。

**📈 对比分析**

与其他方法的比较显示，464条规则中仅有259条被命名，195条产生了至少一条呈现评论，94条产生了记录的代码变更，表明规则的实际使用率较低。

**⚠️ 局限性**

限制在于无法确定未被命名的规则是否因不适用、未被检索或其他原因导致，且数据的保密性限制了对研究数据的公开访问。

---

## 307. From AI Surrogate to Engineering Tool: Toward Production Use of Fiber-Orientation Prediction in Crash CAE

**arXiv ID:** 2611.25991 | [PDF](https://arxiv.org/pdf/2611.25991v1)

**作者:** Osamu Ito `[一作]` (Honda R&D Co), Tomoaki Tatsukawa `[通讯]` (Tokyo University of Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2`

**🎯 论文内容**

本文将之前开发的纤维取向代理模型转化为本田设计工作流程中的工程工具，旨在提高玻璃纤维增强塑料部件的碰撞模拟效率。

**💡 创新点**

创新点在于将机器学习代理模型与自动化工具结合，使得纤维取向预测从传统的25小时缩短至约0.1小时，并且无需模具专家的支持。

**🔧 技术方法**

使用了基于体素的3D条件生成对抗网络（GAN）技术来预测纤维取向，并结合了自动映射和报告生成的集成工具。

**📊 数据集**

使用了包含46个生产部件的调查数据生成的配对流动分析数据集，共生成了4000个配对流动分析样本。

**📈 对比分析**

与传统方法相比，使用预测的取向场进行的组件级碰撞分析将绝对峰值力误差从19.0%降低至11.4%，并且预测的断裂位置与测试观察到的位置一致。

**⚠️ 局限性**

限制在于该系统仅覆盖一种材料（PP-GF20）和最多两个注射口，更多的注射口需要新的配对数据和模型。此外，系统只预测取向，焊缝强度降低和翘曲仍需依赖流动求解器。

---

## 308. The Linguistic Structure of Dependency-Conditioned Embeddings

**arXiv ID:** 2611.25909 | [PDF](https://arxiv.org/pdf/2611.25909v1)

**作者:** Bach Phan-Tat `[一作]` (KU Leuven), Dirk Speelman `[通讯]` (KU Leuven)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了基于依赖关系的分布式模型的几何组织，分析了使用WikiText-103训练的四种表示的全局和局部几何特征。

**💡 创新点**

创新点在于系统性地分析了依赖关系条件下的词汇表示的几何结构，揭示了不同表示方法对几何组织的敏感性。

**🔧 技术方法**

使用了基于计数的PPMI和Skip-Gram with Negative Sampling (SGNS)两种配置的四种表示方法。

**📊 数据集**

使用了WikiText-103数据集，这是一个包含约1.03亿个标记的大规模英语语言建模语料库。

**📈 对比分析**

通过比较不同表示方法的几何一致性，发现PPMI基于的表示在跨关系一致性上优于SGNS基于的表示，且在词汇相似性排名和最近邻重叠方面表现更好。

**⚠️ 局限性**

本研究的局限性在于不同嵌入方法的局部几何一致性差异仍然是描述性的，且定性多义性分析依赖于手动选择的示例。

---

## 309. Xing-Audio-NVVASR: Scaling and Refining Supervision for Joint Recognition of Speech and Non-Verbal Vocalizations

**arXiv ID:** 2611.26000 | [PDF](https://arxiv.org/pdf/2611.26000v1)

**作者:** Pengyu Xu `[一作]` (China Telecom Artificial Intelligence Technology Co. Ltd), Yongxiang Li `[通讯]` (China Telecom Artificial Intelligence Technology Co. Ltd)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种端到端的识别器，通过渐进的数据挖掘和迭代的监督精炼来训练，旨在联合识别语音和非语言发声（NVVs）。

**💡 创新点**

创新点在于结合了跨审计和全语料库的呼吸筛选，以选择性的人类校正来提高注释的准确性，并通过五轮精炼扩展了训练数据集。

**🔧 技术方法**

使用了渐进的数据挖掘、跨审计和全语料库呼吸筛选等技术，构建了一个1.7B参数的识别模型。

**📊 数据集**

使用了170,000个来自公共语料库的弱标注语句和630,000个挖掘的语句，最终形成了约800,000个训练语句。

**📈 对比分析**

与九个模型进行了比较，在四个人工标注的基准测试中，该模型在综合得分和六类宏观F1上均排名第一，综合得分为64.46，超过最强比较模型6.03分。

**⚠️ 局限性**

限制在于仍然存在大量事件遗漏，未能完全恢复所有缺失的注释，且未评估个别示例在训练集中的贡献。

---

## 310. Watch Where You Linearize: Predicting and Correcting Model-Merging Interference

**arXiv ID:** 2611.25000 | [PDF](https://arxiv.org/pdf/2611.25000v1)

**作者:** Ido Nitzan Hidekel `[一作]` (Tel Aviv University), Dan Raviv `[通讯]` (Tel Aviv University)

**通讯引用:** 2540 | [OpenAlex ID](https://openalex.org/A5014799208)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文提出了一种模型合并的方法，通过将单独微调的模型的任务向量添加到共享的基础模型中来组合这些模型。

**💡 创新点**

创新点在于提出了一种新的预测合并干扰的方法，使用方向导数来预测任务向量的影响，而不是在构建合并模型后再进行评估。

**🔧 技术方法**

使用了方向导数（Jacobian-Vector Product, JVP）技术来预测任务合并的干扰，并提出了LR-Merge方法作为一种无标签的修正方法。

**📊 数据集**

使用了多个数据集进行实验，包括MNIST、SVHN、GTSRB、EuroSAT、DTD和CIFAR-10等。

**📈 对比分析**

与现有方法相比，LR-Merge在每个测试的闭合形式、各向同性和稀疏基础合并上都有所改进，且其增益通常随着合并任务数量的增加而增加。

**⚠️ 局限性**

限制在于一次读取的预测仅对附近的候选有效，而对于完整的合并效果不佳。此外，LR-Merge假设共享的初始化和架构，并需要每个任务的几十个无标签探针。

---

## 311. Meta-RTL: Evolving Agent Harnesses for RTL Optimization

**arXiv ID:** 2611.25891 | [PDF](https://arxiv.org/pdf/2611.25891v1)

**作者:** Xinzhuang Xiong `[一作]` (Chinese University of Hong Kong), Evangeline Young `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Meta-RTL，一个基于EDA反馈的框架，通过演化RTL优化工具来优化RTL设计，保持优化模型、EDA流程和每个设计的预算不变。

**💡 创新点**

创新点在于通过演化过程改进RTL优化工具，而不是手动设计，展示了如何利用EDA反馈来提升优化质量和效率。

**🔧 技术方法**

使用了固定的LLM（大型语言模型）和EDA工具，结合了两层嵌套循环的演化机制。

**📊 数据集**

使用了两个基准数据集：Dr. RTL（包含20个设计）和IC-RTL（包含10个未见设计）。

**📈 对比分析**

与Dr. RTL相比，Meta-RTL*在20个开发设计上减少了7.8%的WNS和11.0%的TNS违规，并在10个未见设计上实现了20.5%的WNS和14.0%的TNS减少，显示出显著的性能提升。

**⚠️ 局限性**

限制在于结果的稳健性需要进一步评估，尤其是在不同的LLM随机性和技术库下的表现，以及未来可能扩展到其他EDA任务的潜力。

---

## 312. Drifting, Discriminating, and Directioning for Action Coordination in Higher-dimensional Action Space

**arXiv ID:** 2611.25385 | [PDF](https://arxiv.org/pdf/2611.25385v1)

**作者:** Taisuke Kobayashi `[一作]` `[通讯]` (National Institute of Informatics), Taisuke Kobayashi (National Institute of Informatics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6215c339-3735-4be3-8a07-5bbb7004712d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种轻量级的策略模型，以缓解强化学习中处理高维动作空间的困难，并提出了一种有效的训练算法。

**💡 创新点**

创新点在于利用漂移模型来隐式表示复杂分布，并通过三步学习算法促进动作协调。

**🔧 技术方法**

使用了漂移模型和对抗训练的判别器模型，以及最佳-N采样技术。

**📊 数据集**

使用了dm_control中的三个运动任务数据集，包括四足行走、类人行走和狗行走。

**📈 对比分析**

与传统的对角分布建模方法相比，提出的方法在三个高维动作空间的运动任务中持续提高了样本效率和最终控制性能。

**⚠️ 局限性**

限制在于漂移策略的规模减小可能导致探索不足，可能会影响优化效果。

---

## 313. GLiDE: A Fast, Calibrated Decision Model Distilled from Reasoning Teachers

**arXiv ID:** 2611.26509 | [PDF](https://arxiv.org/pdf/2611.26509v1)

**作者:** Sahibzada Allahyar `[一作]` (Fastino Labs), Urchade Zaratiana `[通讯]` (Fastino Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种决策模型GLiDE，能够在单次前向传递中进行快速、可信的决策，而无需生成文本或中间推理。

**💡 创新点**

创新点在于通过去除因果掩蔽实现上下文、问题和候选答案之间的双向交互，同时结合多教师的软目标蒸馏，将推理能力转移到单次推理中。

**🔧 技术方法**

使用了Qwen混合语言模型，结合非因果全注意力机制和多教师蒸馏技术。

**📊 数据集**

训练数据集包括超过1.6百万个不同问题，涵盖500多个任务家庭，共计约23亿个训练标记。

**📈 对比分析**

与其他模型相比，GLiDE在Hugging Face决策指数中表现优异，27B模型在隐藏集得分为73.37，超越所有同等或更小规模的模型，且在推理效率和校准方面表现良好。

**⚠️ 局限性**

限制在于模型的推理时间延迟增加，27B模型的中位延迟为122毫秒，尽管校准误差保持在2.1%。

---

## 314. CANDOR: Contrastive Self-Denoising Rewards for Label-Free Text-to-Image Post-Training

**arXiv ID:** 2611.25457 | [PDF](https://arxiv.org/pdf/2611.25457v1)

**作者:** Junhyeok Lee `[一作]` (Seoul National University), Kyu Sung Choi `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的内在奖励机制，用于文本到图像生成模型的后训练，旨在提高生成图像的质量和与提示的相关性。

**💡 创新点**

创新点在于引入了一种对比对齐的内在奖励，衡量提示对模型去噪样本的帮助程度，克服了绝对自信度受图像复杂性影响的问题。

**🔧 技术方法**

使用了对比对齐的无标签内在奖励机制，结合了流匹配模型的GRPO（组相对优化）方法。

**📊 数据集**

使用了公共的Flow-GRPO适配器和OCR文本渲染基准数据集进行评估。

**📈 对比分析**

与基线模型和之前的绝对自信奖励进行比较，结果显示该方法在文本渲染、美学和人类偏好评分上均有显著提升，且在60%和63%的同提示样本对中表现更佳。

**⚠️ 局限性**

局限性在于仅在单一基线模型上进行评估，尚需验证其在其他文本到图像模型上的普适性。此外，该方法依赖于无条件分支，需要进行额外的网络评估。

---

## 315. PaveSplat: Resolution-Robust Feed-Forward Gaussian Splatting with Residually Corrected Surface-Aligned Covariances

**arXiv ID:** 2611.26045 | [PDF](https://arxiv.org/pdf/2611.26045v1)

**作者:** Janghun Kim `[一作]` (ETRI), Hyon-Gon Choo `[通讯]` (ETRI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为PaveSplat的前馈高斯喷涂方法，通过添加学习的残差来修正表面对齐的协方差，从而保持surfels在不同渲染分辨率下覆盖可见表面。

**💡 创新点**

创新点在于将分析协方差作为初始化，并通过一个单位四元数残差来重新定向surfels，仅在需要改善渲染时进行调整，从而解决了传统方法在深度边缘和纹理缺失区域的surfels方向不正确的问题。

**🔧 技术方法**

使用了前馈高斯喷涂技术，结合了可微分光栅化器和学习的残差机制。

**📊 数据集**

在RealEstate10K数据集上进行训练和评估。

**📈 对比分析**

与同样配方的SurfSplat方法进行比较，PaveSplat在每个渲染尺度（1×–8×）上都取得了最佳的PSNR、SSIM和LPIPS，尤其在高分辨率重渲染时表现更佳，覆盖孔洞率低于基线方法。

**⚠️ 局限性**

限制在于HRRC仅是对重渲染预测的覆盖压力测试，并不是对原生高分辨率捕获的替代；残差仅适应方向，未能在所有分解中验证其有效性。

---

## 316. Why Flat Drafts Stay Flat: Region-Mean Latents Govern Texture in Flow-Based Image Generation

**arXiv ID:** 2611.25930 | [PDF](https://arxiv.org/pdf/2611.25930v1)

**作者:** Zihan Gu `[一作]` (Institute of Information Engineering), Lulu Zhao `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文探讨了平面色块草图如何在保持其平面外观的同时，通过流式生成器生成纹理细节，揭示了变分自编码器（VAE）潜在空间中细粒度像素变化与区域均值之间的跨尺度机制。

**💡 创新点**

创新点在于识别了平面草图引起的区域潜在位移对纹理生成的强烈影响，并提出了一种训练无关的修正方法，通过像素抖动和均值场缩放来改善生成效果。

**🔧 技术方法**

使用了变分自编码器（VAE）和条件高斯流等技术，结合了低秩适应（LoRA）和草图条件化的方法。

**📊 数据集**

使用了FLUX.1-dev和Qwen-Image数据集，这些数据集包含300张不重叠的图像，主要用于验证机制和确认实验。

**📈 对比分析**

通过与其他方法的比较，发现所提出的方法在纹理生成、空间保真度和语义恢复方面表现优越，尤其是在使用低秩适应后，内容恢复得到了显著改善。

**⚠️ 局限性**

限制在于评估仅使用了两个基础模型、固定的掩码和调度，以及一次LoRA训练运行，且评分是自动化的，定性案例是事后选择的。

---

## 317. Mode Inference for Substructural Adjoint Types

**arXiv ID:** 2611.24984 | [PDF](https://arxiv.org/pdf/2611.24984v1)

**作者:** Sophia Roshal `[一作]` (Carnegie Mellon University), Frank Pfenning `[通讯]` (Carnegie Mellon University)

**通讯引用:** 13620 | [OpenAlex ID](https://openalex.org/A5021476649)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文定义了一种用于伴随类型的模式推断算法，该算法能够为程序生成主模式，从而简化程序员对子结构属性的分析。

**💡 创新点**

创新点在于提出了一种新的模式推断算法，能够在不需要程序员提前分析代码的情况下，自动推断出函数的所有可能子结构模式，并且允许单个函数定义涵盖不同模式的所有实例。

**🔧 技术方法**

使用了一种约束语言来收集模式变量的最小约束，并证明了该推断系统在一定限制下的健全性和完备性。

**📊 数据集**

论文中没有具体提到使用的数据集，但提到了一些示例程序，如列表的追加和映射函数。

**📈 对比分析**

与Jang等人的单态版本相比，本文的模式推断算法在存在递归类型和函数的情况下仍然是可判定的，并且能够推断出主模式，避免了为不同模式重写或复制相同函数的需要。

**⚠️ 局限性**

限制在于程序员仍需显式包含类型中的转换，以确保程序允许最大的模式数量。此外，允许收缩的类型的自然解释是通过共享实现的，但类型检查器不会插入此类代码。

---

## 318. CANDID: Candidate-Driven Intervention for Disciplined Execution in Local Computer-Use Agents

**arXiv ID:** 2611.24977 | [PDF](https://arxiv.org/pdf/2611.24977v1)

**作者:** Woongkyu Lee `[一作]` (Hanyang University), Jungwook Choi `[通讯]` (Hanyang University)

**通讯引用:** 3664 | [OpenAlex ID](https://openalex.org/A5089720016)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CANDID（候选驱动干预框架），旨在增强资源受限环境中本地计算机使用代理（CUAs）的执行可靠性。

**💡 创新点**

CANDID通过候选行为的选择和一致的完成信号来解决CUAs在执行过程中常见的重复行为循环和过早完成的问题，提供了一种新的执行控制信号。

**🔧 技术方法**

使用了候选行为过滤和一致的完成门控策略，结合了循环恢复机制和保守的完成决策。

**📊 数据集**

在OSWorld基准测试上进行了实验，评估了多个本地CUA模型，包括Qwen3-VL和OpenCUA变体。

**📈 对比分析**

与基于反馈的方法和测试时间缩放基线相比，CANDID在所有评估模型上均显著提高了任务成功率，且执行开销最小。

**⚠️ 局限性**

CANDID依赖于基础代理生成足够多样和有用的候选行为，当候选行为收敛到相同的失败行为时，候选驱动的恢复效果有限。此外，当前的循环检测器主要针对局部重复模式，可能会错过更长范围或不规则的失败周期。

---

## 319. Feedback-Free FDD Cell-Free Massive MIMO With Sparse AP Activation and Dual-Mode Precoding

**arXiv ID:** 2611.25968 | [PDF](https://arxiv.org/pdf/2611.25968v1)

**作者:** Wei Jiang `[一作]` (German Research Center for Artificial Intelligence), Hans Dieter Schotten `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种反馈自由的频分双工（FDD）无小区大规模MIMO框架，通过稀疏接入点（AP）激活和角度互易预编码实现。

**💡 创新点**

创新点在于通过稀疏AP激活减少下行导频开销，并利用角度互易性完全消除信道反馈，同时引入负载自适应双模式优化器以提升性能。

**🔧 技术方法**

使用了稀疏接入点激活（SAA）、角度互易预编码（ARP）和负载自适应双模式优化技术。

**📊 数据集**

使用了模拟数据集，具体参数包括64个AP、4个天线和8个用户，进行了500次独立的蒙特卡洛仿真。

**📈 对比分析**

与现有的反馈自由FDD解决方案相比，提出的方法在光谱和能量效率上表现出优越性，且对角度误差、互易性不匹配和多径的鲁棒性更强。

**⚠️ 局限性**

限制在于该方法在高负载情况下可能需要更多的计算资源，且在极端条件下的性能可能会受到影响。

---

## 320. Learning Ankle Exoskeleton Control for Reactive Standing Balance Through Neuromusculoskeletal Simulation

**arXiv ID:** 2611.25480 | [PDF](https://arxiv.org/pdf/2611.25480v1)

**作者:** Maria T. Tagliaferri `[一作]` (Carnegie Mellon University), Inseung Kang `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于模拟到现实的框架，结合运动模仿和强化学习，开发用于平衡辅助的踝部外骨骼控制。

**💡 创新点**

创新点在于使用单一虚拟人模型进行训练，能够生成可转移的平衡辅助，并且无需针对每个用户进行特定调优。

**🔧 技术方法**

采用了肌肉驱动的神经肌肉骨骼模型和强化学习技术。

**📊 数据集**

使用了来自单一受试者的少量运动捕捉数据集，进行模型训练。

**📈 对比分析**

与传统的预定义扭矩配置方法相比，该方法通过实时传感器输入生成辅助，结果显示受试者在接受辅助时，能够承受的扰动幅度提高了11.8%，并且重心位移和速度分别减少了13.5%和28.2%。

**⚠️ 局限性**

限制在于虚拟人模型仅基于一个受试者的数据，未能代表不同的人体测量和恢复策略，且未考虑设备模型中的袖口顺应性和传感器耦合。

---

## 321. Bradley-Terry Training Can Distort Outcome Probabilities of Initially Accurate Language World Models

**arXiv ID:** 2611.25594 | [PDF](https://arxiv.org/pdf/2611.25594v1)

**作者:** Xinze Shi `[一作]` (Beijing University of Posts and Telecommunications), Zhengxuan Hu `[通讯]` (Queen Mary University of London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了比较反馈如何影响语言世界模型的训练，特别是使用Bradley–Terry目标函数时，模型可能会偏离真实的结果概率。

**💡 创新点**

创新点在于揭示了比较反馈并不一定能教会模型真实的结果概率，且可能导致模型将概率转移到条件下从未产生的结果上。

**🔧 技术方法**

使用了Bradley–Terry模型作为目标函数，并进行了三种审计方法来评估模型的表现。

**📊 数据集**

使用了模拟的机器人推送任务，涉及80个枚举分支的结果分布。

**📈 对比分析**

与其他方法相比，Event BT模型在概率分布上表现出显著的偏移，导致其在未见条件下的表现下降，尽管在某些条件下重新调整阈值可以恢复部分效用。

**⚠️ 局限性**

限制在于主要结果是基于模拟的，且只使用了一种语言模型和一种真实判断者，缺乏视觉输入，且大多数设置只有两个反馈种子。

---

## 322. WeightCode: Installing New Capabilities into Neural Networks via Weight-Space Programming

**arXiv ID:** 2611.25476 | [PDF](https://arxiv.org/pdf/2611.25476v1)

**作者:** Bo Li `[一作]` (University of Macau), Ling Yang `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的框架Weight-Space Programming (WSP)，通过可执行的权重程序直接修改神经网络的权重，从而在已训练的神经网络中安装新能力。

**💡 创新点**

创新点在于通过可执行程序直接构建和修改神经网络的权重，允许在不依赖梯度训练的情况下扩展网络的能力，同时保留已有的能力。

**🔧 技术方法**

使用了Weight-Space Programming (WSP)框架，结合了冻结的语言模型来生成可执行的权重程序。

**📊 数据集**

使用了多个数据集，包括SCAN、BabyAI和ImageNet，进行不同任务的能力扩展和性能评估。

**📈 对比分析**

与八种微调方法进行比较，WSP框架在保持原有准确率的同时，在ImageNet上实现了8.84到17.62的净增益，超越了最佳微调方法的表现。

**⚠️ 局限性**

限制在于当前方法依赖于特定的网络架构和任务知识，可能在不同类型的网络或任务上表现不一致。

---

## 323. Demonstration-Free Skill Composition for Navigation and Manipulation

**arXiv ID:** 2611.25462 | [PDF](https://arxiv.org/pdf/2611.25462v1)

**作者:** Madeleine Nouri `[一作]` (University of New South Wales), Francisco Cruz `[通讯]` (Universidad Central de Chile)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文评估了无示范的层次强化学习管道在MiniGrid导航和机器人操作中的应用，比较了其训练需求和任务性能与平坦强化学习基线的差异。

**💡 创新点**

创新点在于提出了一种无示范的层次管道，能够在未见过的布局中重用参数化技能，并且在导航和操作任务中表现出更高的成功率和更少的训练步骤。

**🔧 技术方法**

使用了层次强化学习（HRL）技术，结合了参数化技能和高层控制器来执行任务。

**📊 数据集**

在导航中使用了MiniGrid FourRooms环境，在操作中使用了Robosuite的Panda机器人臂。

**📈 对比分析**

与平坦DQN基线相比，层次管道在未见布局中的成功率为0.555，而平坦DQN仅为0.020；在操作任务中，层次链的成功率为0.858，平坦SAC的成功率为0.914，但层次链的训练步骤更少。

**⚠️ 局限性**

限制在于层次链在抓取任务中表现出较高的失败率，且平坦SAC在最终成功率上表现更好，未来需要改进技能的可靠性和探索不同的奖励机制。

---

## 324. Safe Subgame Resolving in Multiplayer Games

**arXiv ID:** 2611.25981 | [PDF](https://arxiv.org/pdf/2611.25981v1)

**作者:** Boning Li `[一作]` (Institute for Interdisciplinary Information Sciences), Longbo Huang `[通讯]` (Institute for Interdisciplinary Information Sciences)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文提出了针对多玩家隐信息博弈的子博弈解决方案的第一个条件安全定理，证明了在多玩家情况下，更新后的最佳响应差距最多增加四个可计算的项。

**💡 创新点**

创新点在于首次为无关多玩家子博弈解决提供了安全保证，并通过后验证书来替代传统的两人博弈中对手收益等于解决者损失的保证。

**🔧 技术方法**

使用了对抗性遗憾最小化（CFR）和其变体，以及基于最佳响应的认证边际计算技术。

**📊 数据集**

使用了三人Leduc扑克和Kuhn扑克等数据集进行实验，验证了所提出的方法的有效性。

**📈 对比分析**

与传统的子博弈解决方法相比，使用证书的解决方案在多玩家博弈中能够有效避免有害更新，且在Kuhn扑克和Leduc扑克中，证书能够标记所有有害更新并接受大部分安全更新。

**⚠️ 局限性**

限制在于该方法依赖于计算每个对手的最佳响应，可能在计算复杂度上存在挑战，尤其是在玩家数量较多时。

---

## 325. Blind Spots Travel, Knowledge Stays Home: What RL Inherits from an LLM Judge

**arXiv ID:** 2611.25191 | [PDF](https://arxiv.org/pdf/2611.25191v1)

**作者:** Liang Yue `[一作]` (University of Colorado), Eric A. Vance `[通讯]` (University of Colorado)

**通讯引用:** 1687 | [OpenAlex ID](https://openalex.org/A5021606327)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在没有答案键的情况下，如何利用大型语言模型（LLM）作为评估者来训练模型回答封闭式问题或选择放弃回答。通过814次强化学习（RL）实验，探讨了评估者的知识如何影响模型的回答能力。

**💡 创新点**

创新点在于揭示了评估者的知识对模型的影响并不如预期，尤其是在处理未见过的问题时，评估者的知识对模型的回答能力的提升微乎其微，且错误的评判会导致模型放弃某些类别的回答。

**🔧 技术方法**

使用了强化学习（RL）技术来训练模型，并通过不同的评估者（包括参考答案和无参考的评估者）来进行比较。

**📊 数据集**

使用了TriviaQA、PopQA和NQ-open等数据集进行训练和测试，共计750个训练问题和约750个测试问题。

**📈 对比分析**

与其他方法的比较显示，使用答案键的训练模型在选择回答方面表现与未训练模型相似，且评估者的知识对模型的提升效果有限，平均每100个问题仅能纠正1个答案。

**⚠️ 局限性**

限制在于研究仅针对封闭式短答案问题，且使用的模型参数为7-8B，未测试在开放式生成任务中的表现。此外，评估者的知识未能有效转移到模型的回答能力上。

---

## 326. Multi-institutional grading of Barrett's esophagus dysplasia without stain normalization using a pathology foundation model and multiple instance learning

**arXiv ID:** 2611.26562 | [PDF](https://arxiv.org/pdf/2611.26562v1)

**作者:** Amirali Khosravi `[一作]` (Mayo Clinic), Bradley J. Erickson `[通讯]` (Mayo Clinic)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究开发了一种基于基础模型的深度学习系统，用于对Barrett食管的发育不良进行分级，旨在解决临床决策中观察者间变异性的问题。

**💡 创新点**

创新点在于使用了无须特定机构染色标准化的基础模型，能够在多个机构间一致地进行发育不良分级。

**🔧 技术方法**

采用了深度学习技术，具体为两阶段架构：第一阶段为基于专家注释的补丁级分类器，第二阶段为属性感知的多实例学习进行整体分类。

**📊 数据集**

使用了来自四个机构的969个H&E染色的全切片图像数据集，包括Mayo Clinic、Baylor Dallas、Oregon Health & Science University和University of Alabama at Birmingham。

**📈 对比分析**

与专家共识相比，模型在二元发育不良检测中的敏感性为98%，特异性为81%，宏平均AUC为0.96，显示出良好的性能。

**⚠️ 局限性**

限制在于研究为回顾性设计，可能存在选择偏倚，且数据集的类分布不反映实际监测人群，此外未进行与其他现代基础模型的直接比较。

---

## 327. Sonic Palette: Predicting Dynamic Color Themes from Music

**arXiv ID:** 2611.25781 | [PDF](https://arxiv.org/pdf/2611.25781v1)

**作者:** Wen-Li Wei `[一作]`, Jen-Chun Lin `[通讯]`

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Sonic Palette，一个基于音乐的框架，用于从音频预测动态色彩主题，解决了音乐与色彩之间的动态对应关系。

**💡 创新点**

创新点在于将音乐-色彩生成视为跨模态序列预测任务，能够捕捉色彩主题随音乐变化的时间演变。

**🔧 技术方法**

使用了音乐条件的自回归Transformer模型来预测逐帧的色彩主题。

**📊 数据集**

使用了从265个完整官方音乐视频中提取的音频-色彩序列数据集。

**📈 对比分析**

与静态多层感知器（MLP）和非自回归Transformer进行了比较，结果显示Sonic Palette在色彩主题预测上表现更好，具有更低的平均色彩箱差（ACBD）和平均时间箱差（ATBD），并且在主观评估中获得更高的自然性评分。

**⚠️ 局限性**

限制在于模型可能对某些音乐类型的色彩变化捕捉不够准确，且需要大量的训练数据来提高预测的准确性。

---

## 328. When AI Analysts Disagree: Agentic Multiverse Analysis and the Robustness of Empirical Research

**arXiv ID:** 2611.25198 | [PDF](https://arxiv.org/pdf/2611.25198v1)

**作者:** Ruijiang Gao `[一作]` (University of Texas at Dallas), Steven Chong Xiao `[通讯]` (University of Texas at Dallas)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本论文研究了生成性人工智能（AI）在解决非标准误差问题中的应用，特别是通过多宇宙分析（AMA）来评估实证研究的稳健性。

**💡 创新点**

创新点在于提出了一种多代理框架，利用AI代理独立评估和验证多种分析选择，从而量化实证结论的稳健性，并且显著降低了传统人类分析的成本和时间。

**🔧 技术方法**

使用了生成性AI代理（如Claude Code代理）进行数据分析、代码编写和结果验证，采用了多代理系统设计来确保分析的多样性和有效性。

**📊 数据集**

使用了NYSE交易和报价的毫秒级数据集（2015-2024），并对十篇来自领先管理期刊的实证研究进行了评估。

**📈 对比分析**

与传统方法相比，AMA通过系统性地枚举和验证分析选择，能够在短时间内提供稳健性评估。结果显示，某些结论在有效规格下得到了加强，而另一些则难以维持，表明结论对特定分析选择的敏感性。

**⚠️ 局限性**

限制在于样本量较小，仅评估了十篇论文，且这些论文并非随机抽取，因此结果不能广泛推广。此外，AMA的有效性依赖于所使用的模型，可能会遗漏某些重要的分析选择。

---

## 329. DAWN: Robust Neural Blind Deconvolution via Alternating Wiener Guidance

**arXiv ID:** 2611.25621 | [PDF](https://arxiv.org/pdf/2611.25621v1)

**作者:** Yida Pan `[一作]` (University of Michigan), Ismail Alkhouri `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为DAWN的盲去模糊框架，通过交替直接注入维纳去模糊估计和神经网络的自我优化来恢复图像。

**💡 创新点**

DAWN是一个无训练数据的框架，能够在不同噪声水平下稳定地超越现有的最先进方法，并且不需要对噪声水平进行重新调优。

**🔧 技术方法**

使用了深度图像先验（DIP）和维纳去模糊技术，结合了图像和核网络的联合优化。

**📊 数据集**

在多个标准基准数据集上进行了评估，包括Levin和Köhler数据集，以及真实世界的模糊图像。

**📈 对比分析**

与现有的DIP方法（如SelfDeblur和W-DIP）进行比较，DAWN在PSNR、SSIM和MNC等指标上均表现优异，尤其在高噪声条件下表现更为突出。

**⚠️ 局限性**

DAWN目前继承了基于DIP优化的每幅图像计算成本，未来的工作将集中在提高低噪声条件下的速度和性能。

---

## 330. RefreshGrad: An Online Adaptive Data-sampling Method for Diffusion-based Visuomotor Learning

**arXiv ID:** 2611.25398 | [PDF](https://arxiv.org/pdf/2611.25398v1)

**作者:** Dijia Zhan `[一作]` (South China University of Technology), Jie Tang `[通讯]` (South China University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种在线自适应数据采样框架，通过动态调整时间窗口的采样权重来优化基于演示的策略学习。

**💡 创新点**

创新点在于识别演示数据在时间窗口中的非均匀价值，并根据当前策略的学习进展动态调整训练曝光。

**🔧 技术方法**

使用了基于动作分块的扩散策略作为框架，并结合了五种策略原生信号进行窗口评分。

**📊 数据集**

在十个LIBERO-Plus任务上进行了评估，这些任务涉及容器放置、碗转移、顺序操作和关节机制。

**📈 对比分析**

与Uniform和FAKTUAL方法相比，提出的方法在视觉扰动下的平均成功率从60.7%提升至76.1%，在干净场景下从57.4%提升至81.3%。

**⚠️ 局限性**

限制在于该研究主要集中在一个控制的动作分块扩散策略上，未来的工作可以扩展到大规模的语言条件VLA骨干网和物理机器人平台。

---

## 331. Learning to Move from and for Manipulation Policies: Spatializing Latent Action Representations for Goal-Pose-Free Mobile Manipulation

**arXiv ID:** 2611.25366 | [PDF](https://arxiv.org/pdf/2611.25366v1)

**作者:** Siyuan Zheng `[一作]` (University of Michigan), Henry X. Liu `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种名为M4M的框架，通过从预训练的操作策略中学习与操作相关的移动能力，以解决机器人在执行任务时的姿态调整问题。

**💡 创新点**

创新点在于将冻结策略的潜在动作表示空间化为未访问姿态的成功场，并通过密集的候选视图监督和有限的执行结果训练当前视图预测器。

**🔧 技术方法**

使用了深度学习技术，特别是卷积神经网络（CNN）和知识蒸馏方法来提取和利用策略特征。

**📊 数据集**

在RoboCasa厨房中进行了八个未见任务的实验，包括关闭冰箱、微波炉、烤箱等，使用了180个实验回合。

**📈 对比分析**

与最强基线N2M相比，M4M在八个任务中的平均成功率为47.0%，而N2M为26.5%，直接执行为19.2%。M4M在冰箱关闭实验中显示出一致的改进。

**⚠️ 局限性**

局限性包括评估仅限于模拟厨房和特定的起始距离，训练使用特权几何和渲染候选视图，而部署则依赖于立体观察和任务指令，未测试真实世界的鲁棒性。

---

## 332. MAWAM: A Multi-Agent World-Action Model for Coordinated Manipulation

**arXiv ID:** 2611.25105 | [PDF](https://arxiv.org/pdf/2611.25105v1)

**作者:** Xianzhe Fan `[一作]`, Zhiqi Li `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种多智能体世界-动作模型，通过联合建模每个机器人的动作和团队的未来相机视图来学习协调控制。

**💡 创新点**

创新点在于通过跨智能体注意力机制联合去噪所有智能体的未来视图和动作标记，使得每个样本同时包含可执行的动作和同步的视觉未来。

**🔧 技术方法**

使用了Transformer模型，通过跨智能体注意力机制进行联合预测和控制。

**📊 数据集**

在三个真实机器人任务（洗衣袋收集、浴巾折叠和伞的交接）和八个模拟任务（包括相机对齐、协作摄影和多阶段交付）上进行了评估。

**📈 对比分析**

与MA-VLA等方法进行比较，取得了67.8%的平均成功率，比MA-VLA高出18.9个百分点。在模拟任务中，平均成功率达到90.3%。

**⚠️ 局限性**

限制在于实验仅限于固定的最多四个智能体的机器人团队，未来工作可以扩展到更大的团队，并支持不同机器人之间的协作。

---

## 333. VisAlloc: Allocating Visual Supervision for On-Policy Self-Distillation

**arXiv ID:** 2611.25069 | [PDF](https://arxiv.org/pdf/2611.25069v1)

**作者:** Haoran Xu `[一作]` (Zhejiang University), Rui Qian `[通讯]` (Macau Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种双层框架，通过样本级视觉过滤和标记级反事实重加权来改进多模态大语言模型（MLLMs）的视觉推理。

**💡 创新点**

创新点在于结合了样本选择和标记重加权的双层优化框架，能够有效分配特权视觉监督，提高模型性能。

**🔧 技术方法**

使用了固定配对教师过滤、反事实重加权、动态敏感性权重等技术。

**📊 数据集**

使用了来自Vision-OPD的数据集，包含6241对视觉证据示例，经过过滤后保留5795个示例用于训练。

**📈 对比分析**

与Vision-OPD的比较显示，在8个基准测试上，模型在4B和9B规模下分别达到了81.44和84.92的宏平均，超出Vision-OPD的表现。

**⚠️ 局限性**

局限性在于结果仅限于评估的Qwen3.5-4B和Qwen3.5-9B模型及其基准套件，固定样本过滤依赖于教师的正确性，可能继承错误。

---

## 334. Does the Workspace Do the Work? A Causal Protocol for Verifying Global-Workspace Computation in a Language Model

**arXiv ID:** 2611.26018 | [PDF](https://arxiv.org/pdf/2611.26018v1)

**作者:** Diaoxuan He `[一作]`, Linqian Wu `[通讯]` (University of Waterloo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种因果协议，用于识别模块在通信时的计算责任，并通过实验验证了不同模型在信息处理中的表现。

**💡 创新点**

创新点在于开发了一种新的测试协议，能够明确区分信息是被转发还是重新组合，从而验证了全球工作空间理论的核心主张。

**🔧 技术方法**

使用了因果协议和实验设计，结合了多种模型（如AFGW）进行验证。

**📊 数据集**

使用了包含432个问题的自定义数据集，这些问题涉及绑定和位置的排列组合。

**📈 对比分析**

通过与不同模型的比较，发现AFGW模型在97%的情况下能够正确预测信息的组合结果，而传统方法的准确率较低，显示出新方法的有效性。

**⚠️ 局限性**

局限性在于模型的训练依赖于特定的数据集，可能无法推广到其他类型的任务或更复杂的场景。

---

## 335. Hardening JavaScript Programs against LLM-Based Vulnerability Detection

**arXiv ID:** 2611.25718 | [PDF](https://arxiv.org/pdf/2611.25718v1)

**作者:** Zhenhao Tian `[一作]` (Nanyang Technological University), Penghui Li `[通讯]` (Nanyang Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种防御机制，通过代码转换来增强JavaScript程序对基于大型语言模型（LLM）的漏洞检测的抵抗力。

**💡 创新点**

创新点在于通过进化搜索发现有效的代码转换，并将其提取为可重用的规则，从而在不改变程序语义的情况下，减少检测代理发现的漏洞数量。

**🔧 技术方法**

使用了进化搜索技术来发现代码转换，并结合大型语言模型（LLM）动态选择适合的规则进行应用。

**📊 数据集**

使用了40个真实世界的JavaScript应用程序，这些程序平均约有37,000行代码，涵盖了13种漏洞类型。

**📈 对比分析**

与传统的混淆器JS-Obfuscator和基于LLM的转换方法FPA相比，减少了52.0%到60.7%的检测率，且在所有检测代理上均表现优于这两种方法。

**⚠️ 局限性**

限制在于未能正式证明规则保持语义一致性，且对某些特定类型的漏洞仍然可能被检测到。

---

## 336. Persistent Identity in Person Tracking: Online Enrollment and Registered Recognition

**arXiv ID:** 2611.25283 | [PDF](https://arxiv.org/pdf/2611.25283v1)

**作者:** Ratha Siv `[一作]` (University of Mons), Bernard Gosselin `[通讯]` (University of Mons)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

扩展了PoseTReID，增加了在线外观库匹配和残差冲突保护，用于注册的SVM识别。

**💡 创新点**

创新点在于引入了在线注册和基于姿态的入场政策，以及用于重复名称的冲突保护机制。

**🔧 技术方法**

使用了RBF核SVM进行注册识别，并结合了PoseTReID的检测、跟踪和再识别技术。

**📊 数据集**

使用了六个SportsMOT视频片段和其他多个视频数据集进行评估。

**📈 对比分析**

与独立跟踪器相比，Gallery AIN和COPE在IDF1/HOTA上表现优越，尤其是在完整结果上，AIN在11个视频中达到了95.72/94.82%的性能。

**⚠️ 局限性**

限制在于未能在原始九个视频上实现相同的性能提升，并且姿态门控的效果不够明确。

---

## 337. EmbodyAgent: Post-Training Embodied VLM Agents via Distillation, SFT, and RL

**arXiv ID:** 2611.26196 | [PDF](https://arxiv.org/pdf/2611.26196v1)

**作者:** Wei Dai `[一作]` (Tsinghua University), Shaolun Huang `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了EmbodyAgent，一个完整的后训练框架，用于将开放权重的视觉语言模型（VLM）转变为高性能的具身代理。

**💡 创新点**

创新点在于构建了一个集成的后训练管道，包括黑箱教师数据合成、多模态监督微调和基于结果的强化学习，且所有阶段共享一个统一的think-then-code接口。

**🔧 技术方法**

使用了黑箱数据合成、监督微调（SFT）和在线强化学习（RL）等技术。

**📊 数据集**

使用了LIBERO Spatial任务数据集，进行了严格的评估协议，确保训练和测试状态的分离。

**📈 对比分析**

与传统方法相比，EmbodyAgent在十个LIBERO Spatial任务上的成功率从0/100提升到14/100（经过SFT）和/100（经过RL），同时减少了每个测试回合的模型调用次数，从36次降至11次。

**⚠️ 局限性**

限制在于评估仅覆盖一个模拟套件（十个LIBERO Spatial任务），且每个训练阶段仅反映一次运行，缺乏多种种子复制，可能影响结果的稳定性。

---

## 338. MetaScanner: An Empirical Study and Proactive Defense against Deceptive UI Attacks in Social VR

**arXiv ID:** 2611.25608 | [PDF](https://arxiv.org/pdf/2611.25608v1)

**作者:** Junhee Lee `[一作]` (Chungbuk National University), Jinwoo Kim `[通讯]` (Chungbuk National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了四种新的欺骗性用户界面（UI）攻击，旨在隐秘地操纵社交虚拟现实（VR）用户，使其在不知情的情况下执行意图外的行为。研究通过在VRChat平台上实施这些攻击，并进行了一项涉及30名参与者的用户研究，以验证其有效性。

**💡 创新点**

创新点在于首次系统性地探讨了社交VR平台中欺骗性UI攻击的潜在风险，并提出了一种主动防御系统MetaScanner，能够在用户接触虚拟世界之前识别可疑或欺骗性元素。

**🔧 技术方法**

使用了静态分析技术，通过MetaScanner对虚拟世界资产（包括脚本和交互对象）进行高效扫描，以识别潜在的欺骗性内容。

**📊 数据集**

研究使用了VRChat平台的虚拟世界作为实验环境，并进行了用户研究，参与者包括30名具有不同VR经验的用户。

**📈 对比分析**

与现有方法相比，MetaScanner能够在平均5秒内扫描38个真实世界资产，并成功检测到所有四种攻击场景，显示出其轻量级和高效性。

**⚠️ 局限性**

限制在于MetaScanner可能会产生误报，因为透明和重叠对象在Unity开发中是常见的特性，可能会误判为可疑对象。

---

## 339. Pandora's Toolbox Or Why Agent Safety Cannot Be Verified

**arXiv ID:** 2611.25325 | [PDF](https://arxiv.org/pdf/2611.25325v1)

**作者:** Mario Franco `[一作]` (Binghamton University), Carlos Gershenson `[通讯]` (Binghamton University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

论文探讨了当前人工智能代理的安全性问题，指出固定语言模型与通用计算设备的结合使得安全性不可判定，尤其是在代理可能运行危险程序的情况下。

**💡 创新点**

创新点在于明确了安全性不可判定的理论基础，并展示了在有限约束下安全性仍然是不可处理的，且安全性不是一个组合性质。

**🔧 技术方法**

使用了图灵机的理论框架来描述语言模型和计算设备的行为，并通过具体的算法和理论证明了安全性问题的复杂性。

**📊 数据集**

没有具体提到使用的数据集，主要是理论分析和模型构建。

**📈 对比分析**

与现有方法的比较表明，传统的安全评估方法（如黑名单、白名单等）无法提供完美的安全解决方案，且在有限约束下安全性问题仍然是不可处理的。

**⚠️ 局限性**

限制在于尽管在有限约束下安全性是可判定的，但实际应用中仍然面临计算复杂性的问题，且无法完全保证安全性。

---

## 340. SWE-Recreation: Benchmarking Coding Agents for Software Recreation

**arXiv ID:** 2611.26311 | [PDF](https://arxiv.org/pdf/2611.26311v1)

**作者:** Zhiyao Xu `[一作]` (Tsinghua University), Jingyu Xiao `[通讯]` (Chinese University of Hong Kong)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究探讨了智能体是否能够通过人类用户的方式探索和重建不熟悉的软件，提出了一种新的基准测试，评估智能体在没有任何人类指导或特权访问的情况下的能力。

**💡 创新点**

创新点在于提出了主动黑箱应用重建的任务，智能体必须通过人类界面发现应用功能并重建，而不是依赖于预先定义的目标或示例。

**🔧 技术方法**

使用了人类用户界面进行探索和重建，智能体只能通过鼠标和键盘输入与应用程序交互，没有访问DOM、源代码或后端的权限。

**📊 数据集**

使用了54个真实应用程序（28个网站和26个Android应用），包含1446个隐藏需求，涵盖多步骤状态变化和持久性等功能。

**📈 对比分析**

与五个前沿系统的比较显示，最强的系统在加权保真度上达到0.530，完全重现的需求不到四分之一，表明当前技术在探索和实现方面仍存在显著不足。

**⚠️ 局限性**

限制在于智能体的探索和实现能力受到部分可观察性和模糊视觉反馈的影响，导致可能无法发现某些功能或错误理解观察到的行为。

---

## 341. BADAS-3.0: Real-Time Driving Perception and Planning from One Camera

**arXiv ID:** 2611.26294 | [PDF](https://arxiv.org/pdf/2611.26294v1)

**作者:** Roni Goldshmidt `[一作]` (Empiric Earth), Lorenzo Niccolini `[通讯]` (Empiric Earth)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于单个前向摄像头的驾驶模型，该模型能够在实时延迟下预测当前速度、行驶路径、碰撞风险评分和未来轨迹。

**💡 创新点**

创新点在于使用了一个仅有1B参数的视觉变换器（ViT）模型，能够在低延迟下提供高精度的驾驶预测，并在新的安全基准上表现优于现有的规划器。

**🔧 技术方法**

使用了视觉变换器（ViT）作为模型的基础架构，并结合了多个输出头来预测速度、轨迹、碰撞风险等。

**📊 数据集**

使用了NVIDIA PhysicalAI-Autonomous-Vehicles数据集和带有真实轨迹的专有行车记录仪视频数据集。

**📈 对比分析**

与多个开放驾驶模型进行比较，结果显示该模型在五个轨迹基准上具有44%的minADE_6改进，且在新的安全基准上通过率高于所有测试的规划器。

**⚠️ 局限性**

模型的局限性在于仅能处理短时间的单摄像头输入，无法预测视野之外的危险，并且在长时间驾驶中可能会出现漂移。

---

## 342. Explainability of genotype embedding models through observed cohort paths

**arXiv ID:** 2611.25681 | [PDF](https://arxiv.org/pdf/2611.25681v1)

**作者:** Amashi Niwarthana `[一作]` (Nanyang Technological University), Jagath C. Rajapakse `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本研究提出了一种新的基因型嵌入模型的可解释性方法，通过观察到的队列路径进行归因，旨在提高对神经退行性疾病预测的理解。

**💡 创新点**

创新点在于提出了基于队列路径的归因方法，解决了传统方法在处理基因型嵌入模型时的完整性问题，并且在非加性模型中表现出更好的归因效果。

**🔧 技术方法**

使用了深度学习技术，特别是通过学习的嵌入将离散的基因型映射到连续的向量空间，并在此基础上进行路径归因。

**📊 数据集**

使用了帕金森病进展标志倡议（PPMI）队列的数据集，包括217名受试者的结构性MRI和SNP基因型数据。

**📈 对比分析**

与传统的集成梯度和离散集成梯度方法相比，提出的方法在支持保持扰动下的帕金森病分类中表现出更强的结果，并且在每个变体扰动下与支持背景的Shapley基线竞争力强。

**⚠️ 局限性**

该方法的局限性在于需要一个足够密集的队列来进行路径归因，且在观察到的基因型之间的插值段仍然是线性的，可能导致偏离实际数据。

---

## 343. Falsifiable Compositional Reasoning in Time Series Forecasting

**arXiv ID:** 2611.25388 | [PDF](https://arxiv.org/pdf/2611.25388v1)

**作者:** Hao Xue `[一作]` `[通讯]` (Hong Kong University of Science and Technology), Hao Xue (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种基于可证伪的组合假设的时间序列预测方法，旨在提高模型在非平稳环境下的准确性。

**💡 创新点**

创新点在于将时间序列预测建模为可证伪的组合假设，结合神经原始编码器、显式耦合约束和训练校准的证伪测试，以应对历史回测的局限性。

**🔧 技术方法**

使用了深度序列编码器和预训练的时间序列基础模型（TSFMs），并结合了显式的耦合层和证伪测试。

**📊 数据集**

使用了2012年至2017年的德国电力市场数据集（52,416小时）和一个合成的多原始基准数据集。

**📈 对比分析**

通过与传统的历史回测方法进行比较，发现扩展验证搜索未能改善模型在非平稳状态下的准确性，而使用可证伪的组合假设模型在结构转变时表现更为稳定。

**⚠️ 局限性**

限制在于该方法依赖于对时间序列的结构假设，若假设不准确，可能导致模型性能下降。

---

## 344. Mind2Act World: A Diagnostic Benchmark for Embodied Mind, Physical Action, and Mind-Action Composition in Robotic Manipulation

**arXiv ID:** 2611.26444 | [PDF](https://arxiv.org/pdf/2611.26444v1)

**作者:** Yuzhi Huang `[一作]` (Tsinghua University), Zhi Wang `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Mind2Act World，一个用于评估机器人操作中认知推理、物理执行和两者结合能力的诊断基准，包含30个任务，分为三个视角。

**💡 创新点**

创新点在于系统地评估机器人在操作中的认知推理、物理执行和两者结合的能力，并通过设计任务来隔离和组合这些能力的要求。

**🔧 技术方法**

使用了基于规则的分析与大型语言模型（LLM）结合的诊断工具，评估系统在不同任务中的表现。

**📊 数据集**

使用了包含4500个示例的训练数据集，这些示例来自30个任务的不同难度级别和场景设置。

**📈 对比分析**

与现有的多模态代理、视觉-语言-动作模型（VLA）和世界-动作模型（WAM）进行比较，结果显示所有系统在Mind2Act任务上表现不佳，最强的系统在Mind2Act任务上的成功率仅为12.0%。

**⚠️ 局限性**

限制在于无法完全归因于单一能力的失败，且现有的基准未能涵盖所有可能的系统架构和任务类型。

---

## 345. Rethinking Utility in Differentially Private Visualization: A Channel-Centric Perspective

**arXiv ID:** 2611.26396 | [PDF](https://arxiv.org/pdf/2611.26396v1)

**作者:** Jing Yang `[一作]` (University of North Carolina at Charlotte), Liyue Fan `[通讯]` (University of North Carolina at Charlotte)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过通道中心的视角综合了差分隐私可视化的先前实证、计算和理论研究，提出了通道-读取-机制假设，认为可视化的效用部分依赖于视觉通道配置、任务相关读取和所采用的差分隐私机制之间的相互作用。

**💡 创新点**

创新点在于提出了通道-读取-机制假设，强调了通道选择在差分隐私可视化中的重要性，并指出不同的差分隐私机制可能会保留数据的不同属性，从而影响可视化的效用。

**🔧 技术方法**

使用了差分隐私机制（如拉普拉斯机制）和通道中心的分析方法，结合了多项实证研究和理论框架。

**📊 数据集**

数据集包括多种图表类型和任务组合，具体研究了不同图表在差分隐私下的表现，尤其是饼图、条形图和散点图等。

**📈 对比分析**

通过与现有研究的比较，发现不同图表类型在差分隐私下的效用表现差异显著，某些图表在特定任务下的表现优于其他图表，且效用的下降模式并不均匀。

**⚠️ 局限性**

限制在于提出的假设尚未在控制实验中得到验证，且研究主要集中在基本视觉通道上，复合编码的交互作用未被充分探讨。

---

## 346. SpatialMem: Building Quantitative Spatial Reasoning Agents with Hierarchical Memory

**arXiv ID:** 2611.25925 | [PDF](https://arxiv.org/pdf/2611.25925v1)

**作者:** Zinan Zheng `[一作]` (Hong Kong University of Science and Technology), Jia Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种层次化记忆框架，旨在增强视觉语言模型（VLM）在定量空间推理任务中的能力，通过积累工具使用经验来优化工具选择和任务执行。

**💡 创新点**

创新点在于构建了一个两级记忆结构，分别用于任务记忆和场景记忆，并引入了双状态轨迹优化过程，以保留成功的工具经验并改进次优的工具选择。

**🔧 技术方法**

使用了层次化记忆框架和动态双状态记忆机制，结合了工具选择和执行的优化策略。

**📊 数据集**

使用了一个新的定量空间推理基准数据集，涵盖室内和室外场景以及三种定量空间推理任务，共包含658个问题。

**📈 对比分析**

与现有的空间推理模型和大型模型（如GPT-5.4和Gemini-3.1-Pro Preview）进行比较，提出的方法在多个基准上表现出色，平均提升了9.1%的成功率，超越了最佳基线32%。

**⚠️ 局限性**

限制在于该框架专门针对定量空间推理任务，主要集中在空间问答任务上，当前的记忆层次依赖于预定义的场景和任务类别，限制了其在非空间或纯文本长时间挑战中的直接可推广性。

---

## 347. A Wolf in Sheep's Pixels: Visual Camouflage Attacks via HDR Gain Maps

**arXiv ID:** 2611.25469 | [PDF](https://arxiv.org/pdf/2611.25469v1)

**作者:** Qi Shen `[一作]` (University of the Chinese Academy of Sciences), Tong Liu `[通讯]` (Institute of Information Engineering)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为DRJanus的视觉伪装攻击，利用标准动态范围(SDR)处理与高动态范围(HDR)显示之间的差距，能够在不优化目标模型的情况下，将两个独立选择的彩色图像嵌入到一个JPEG文件中。

**💡 创新点**

创新点在于通过反转HDR重建过程，成功在单个HDR文件中嵌入两个攻击者选择的图像，同时保持SDR基图像的完整性，并在400幅图像上实现平均PSNR为43.03 dB的HDR重建。

**🔧 技术方法**

使用了HDR增益图的构建技术，通过对每个像素的增益值进行计算，实现高保真度的HDR重建。

**📊 数据集**

使用了COCO数据集进行重建质量评估，并在100幅性内容图像的基准测试中评估了内容审核的逃避能力。

**📈 对比分析**

与六个商业内容审核服务和九个视觉语言模型进行比较，DRJanus在所有测试中实现了100%的审核逃避率，并在五个代理上验证了有害图像的发布和任务操控，攻击成功率在31.67%到100%之间。

**⚠️ 局限性**

限制在于DRJanus依赖于增益图在传输过程中被保留，并且需要HDR兼容的查看器进行重建；SDR-only查看器将只显示基图像，可能会导致攻击效果失效。

---

## 348. ResVision: Spectrum-Matched Resizing and Context-Preserving Inspection for Visual Reasoning

**arXiv ID:** 2611.26318 | [PDF](https://arxiv.org/pdf/2611.26318v1)

**作者:** Guojian Zhan `[一作]`, Chuan Mu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为ResVision的方法，通过选择性地扩大证据来改善视觉-语言模型的表现，而不增加新的场景内容。

**💡 创新点**

创新点在于引入了两种互补组件：谱匹配调整（SMR）和上下文保持检查（CPI），以解决视觉推理中的瓶颈问题。

**🔧 技术方法**

使用了谱匹配调整（SMR）和上下文保持检查（CPI）技术。

**📊 数据集**

在四个前沿视觉语言模型（VLMs）和29个基准测试上进行了评估，涵盖六个视觉推理类别。

**📈 对比分析**

完整的ResVision（Agentic Base+SMR+CPI）在每个模型上表现最佳，相较于Direct Base提高了3.36-4.94分，相较于Agentic Base提高了1.87-2.94分。

**⚠️ 局限性**

限制在于高频细节可能在视觉标记形成过程中被削弱，空间分布的证据可能难以被语言模型检索，以及孤立的裁剪可能通过尺度和位置不匹配削弱整体与细节的对应关系。

---

## 349. The Necessity of Acceptance & Rejection in Inferential Reasoning

**arXiv ID:** 2611.26212 | [PDF](https://arxiv.org/pdf/2611.26212v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 350. SkillLens: Action-Guided Dense Visual Dynamics for Proficiency Estimation

**arXiv ID:** 2611.25712 | [PDF](https://arxiv.org/pdf/2611.25712v1)

**作者:** Deokhyun Ahn `[一作]` (Korea University), Haksub Kim `[通讯]` (Korea Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为SkillLens的框架，用于通过原子动作描述来估计人类的熟练程度，利用时间指导来定位与熟练度相关的窗口，从而提取密集的视觉动态特征。

**💡 创新点**

创新点在于将原子动作描述作为时间指导，而不是直接的语义监督，来定位与熟练度相关的执行线索，并通过高低原型边际进行比较。

**🔧 技术方法**

使用了多模态特征（包括自我视频、外部视频和注视轨迹），结合活动感知的专家路由和CORAL有序校准技术。

**📊 数据集**

在Ego-Exo4D数据集上进行评估，该数据集用于熟练度估计任务，模型预测四个有序的熟练度水平：新手、早期专家、中级专家和晚期专家。

**📈 对比分析**

与现有方法相比，SkillLens在准确性上表现最佳，达到了0.542的准确率，并且在宏观F1、MAE和斯皮尔曼相关性等指标上也表现良好，显示出更好的类平衡和排序一致性。

**⚠️ 局限性**

限制在于原子动作描述可能过于宽泛，导致对熟练执行的标准模糊；此外，原型构建受到活动-动作桶之间类别不平衡的影响，未来的工作可以探索不确定性感知或分层原型估计来提高鲁棒性。

---

## 351. Reinforcing Few-Step Diffusion Models via Shadow Posterior Matching

**arXiv ID:** 2611.25998 | [PDF](https://arxiv.org/pdf/2611.25998v1)

**作者:** Jingqiao Xiu `[一作]`, Yicong Li `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出了一种新的少步扩散模型训练方法，称为影子后验匹配（Shadow Posterior Matching），旨在解决现有少步强化学习方法在奖励信用计算中的结构性挑战。

**💡 创新点**

创新点在于通过影子后验来计算每个调用的奖励信用，从而避免了现有方法在速度不匹配和外部锚定方面的缺陷。

**🔧 技术方法**

使用了影子后验匹配技术，该技术通过对生成样本进行重新噪声处理来获得奖励方向，并结合KL正则化的奖励目标推导出回归损失。

**📊 数据集**

在4步生成器上进行了实验，使用了通过分布匹配蒸馏（DMD）得到的数据集。

**📈 对比分析**

与现有的少步强化学习方法（如RTDMD和TDM-R1）进行比较，结果显示该方法在多个任务上获得了更高的奖励，并且减少了奖励黑客行为。

**⚠️ 局限性**

限制在于该方法依赖于生成样本和其分布的评分，可能在某些情况下无法适用于所有类型的少步生成器。

---

## 352. CRAFT: Analog Circuit Design by Reasoning LLM Agents from Pre-Computed Lookup Tables

**arXiv ID:** 2611.25908 | [PDF](https://arxiv.org/pdf/2611.25908v1)

**作者:** Hyunjun Park `[一作]` (Samsung Electronics), Junhyung Kim `[通讯]` (Seoul National University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一种名为CRAFT的自动化模拟电路设计方法，利用大型语言模型（LLM）代理通过计算设计给定拓扑，而无需初始尺寸。

**💡 创新点**

创新点在于引入了DeX服务，结合了设备查询和逆向尺寸计算，通过预计算的查找表进行电路分析，替代了SPICE模拟。

**🔧 技术方法**

使用了大型语言模型（LLM）代理和DeX服务，结合了查找表和牛顿求解器进行电路设计。

**📊 数据集**

在商业28nm工艺下测试了四个电路，包括一个37晶体管的三阶段放大器，具有53个设计变量和20个规格。

**📈 对比分析**

CRAFT在所有20次运行中满足了每个规格和设备规则，相比于使用HSPICE的代理，CRAFT在相同请求下的通过率从6/10提高到10/10，且中位时间减少了51%。

**⚠️ 局限性**

限制在于CRAFT依赖于预计算的查找表，可能在某些情况下无法处理未包含在表中的新设备特性。

---

## 353. What Places an Action Boundary? Structure-Aware Attribution for Temporal Action Localization

**arXiv ID:** 2611.25815 | [PDF](https://arxiv.org/pdf/2611.25815v1)

**作者:** Ruoshan Lan `[一作]` (University of Arizona), Mihai Surdeanu `[通讯]` (University of Arizona)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种结构感知的归因框架，用于时间动作定位（TAL），旨在提高动作边界的定位准确性。

**💡 创新点**

创新点在于通过随机证据恢复顺序的平均贡献来捕捉时间上下文的依赖性，并将归因方法专门调整为边界定位。

**🔧 技术方法**

使用了基于Shapley–Shubik原则的随机顺序路径归因方法，并结合了细胞结构的恢复顺序和边界锚定参考。

**📊 数据集**

在THUMOS14和EPIC-KITCHENS-100数据集上进行了实验，使用了TriDet和ActionFormer检测器。

**📈 对比分析**

与现有方法相比，提出的方法在归因的可信度评估中表现更好，尤其是在分类和边界预测方面，显示出更高的准确性。

**⚠️ 局限性**

局限性在于分析集中于两个密集的TAL检测器的正确定位检测，未直接确定高置信度定位错误的原因，未来工作可扩展至定位失败的分析。

---

## 354. D3-Cache: Accelerating Diffusion Models with Dynamic Mode Decomposition

**arXiv ID:** 2611.25689 | [PDF](https://arxiv.org/pdf/2611.25689v1)

**作者:** Yanbo Li `[一作]` (Shenzhen Institutes of Advanced Technology), Xitong Gao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的加速Diffusion Transformers的方法，通过动态模式分解（DMD）预测调度器所需的速度，从而减少每次采样的网络评估次数。

**💡 创新点**

创新点在于使用DMD进行速度预测，并通过一个闭环的、考虑时间跨度的门控机制来决定何时刷新预测，避免了对每个模型的离线校准。

**🔧 技术方法**

使用了动态模式分解（DMD）技术来进行速度预测，并实现了一个闭环的、时间跨度感知的刷新机制。

**📊 数据集**

在四个最先进的图像和视频模型上进行了评估，包括HiDream-I1和HunyuanVideo。

**📈 对比分析**

与其他缓存方法相比，该方法在速度上加快了2.6到3.8倍，并且在相似速度下更忠实于基础模型的输出，内存开销几乎可以忽略不计。

**⚠️ 局限性**

限制在于该方法依赖于历史速度的准确性，且在初始采样阶段需要进行完整评估以初始化速度历史。

---

## 355. Query-Aware vs. Query-Agnostic Training-Free Video Token Pruning in Multimodal LLMs

**arXiv ID:** 2611.25537 | [PDF](https://arxiv.org/pdf/2611.25537v1)

**作者:** Divya Kothandaraman `[一作]` (Dolby Laboratories), Jaclyn Pytlarz `[通讯]` (Dolby Laboratories)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了训练无关的预LLM视频令牌选择，比较了21种不同的剪枝原则，探讨了查询信息在剪枝中的作用。

**💡 创新点**

提出了结构化查询剪枝方法，通过将问题分解为语义锚点来指导令牌选择，并引入了Pareto变体以同时保持查询相关性和视觉多样性。

**🔧 技术方法**

使用了Qwen-3-VL作为共享骨干网络，采用了训练无关的剪枝方法，比较了不同的剪枝策略。

**📊 数据集**

在Video-MME、MVBench和LongVideoBench数据集上进行了实验。

**📈 对比分析**

与冗余基线相比，结构化选择在证据特定和长上下文任务中表现最佳，而在广泛任务中，整体选择器优于单独的相关性或冗余方法。

**⚠️ 局限性**

方法依赖于分解器生成的锚点能够捕捉所需证据，且视觉编码器需将文本锚点与视觉证据对齐，均可能出现幻觉现象。

---

## 356. Deterministic Temporal Embedding for Calcium Imaging Trace Extraction

**arXiv ID:** 2611.26522 | [PDF](https://arxiv.org/pdf/2611.26522v1)

**作者:** Qi Gao `[一作]`, Karl Rohr `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种确定性时间嵌入框架，用于从钙成像序列中提取钙依赖的荧光轨迹，旨在识别反映相同细胞活动的像素。

**💡 创新点**

创新点在于结合了自适应的欧几里得-余弦模糊图与粒子网优化的类平衡交叉熵目标，实现了在低维空间中分离时间上不同的源，并将结果转化为可靠的细胞支持。

**🔧 技术方法**

使用了确定性时间嵌入框架、模糊邻域图、粒子网优化和类平衡交叉熵目标等技术。

**📊 数据集**

使用了合成数据集和真实的活体钙成像数据集，合成数据集包含44个不规则细胞的180帧图像，真实数据集来自于一只植入BG5胶质母细胞瘤细胞的NMRI裸鼠。

**📈 对比分析**

与现有的Suite2p和CaImAn方法进行了比较，PM-Adaptive方法在合成数据集上取得了最佳的F1分数和召回率，尤其在触碰细胞的恢复上表现优异，且在真实数据集上也显示出更好的空间一致性和荧光轨迹提取效果。

**⚠️ 局限性**

限制在于该方法依赖于时间相似性而不强加显式的细胞形状先验，可能在某些情况下影响源的识别准确性。

---

## 357. ZoionNET: Low-Power Animal Intrusion Detection Using an Acoustic Deep-Learning Classifier for a Cellular Remote-Sensing System

**arXiv ID:** 2611.26112 | [PDF](https://arxiv.org/pdf/2611.26112v1)

**作者:** Prasanna Vijay `[一作]` (Sri Sairam Engineering College), Teena Sharma `[通讯]` (Indian Institute of Technology Guwahati)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种名为ZoionNET的低功耗动物入侵检测系统，通过声学深度学习分类器在南印度的森林边界监测动物入侵。

**💡 创新点**

创新点在于采用声学而非视觉监测，结合了轻量级和重型模型的两状态推理设计，以降低能耗，同时实现高准确率的动物分类。

**🔧 技术方法**

使用了Arm Ethos-U55微处理器和YOLOv8模型进行声学信号的分类，采用了短时傅里叶变换和梅尔频谱图作为输入。

**📊 数据集**

使用了来自公共来源（如YouTube和Freesound）的音频数据集，主要包括背景噪声、猎豹、大象和猴子的声音。

**📈 对比分析**

与传统的基于摄像头的监测系统相比，ZoionNET在功耗和成本上具有优势，完整节点的成本为133.57美元，且在实际部署中表现出良好的准确率（检测模型为0.974，分类模型为0.946）。

**⚠️ 局限性**

限制在于尚未测量系统的能耗表现，缺乏对电池寿命和实际工作状态下的电流消耗的评估，此外，当前模型未覆盖所有可能的入侵物种。

---

## 358. When Should Deployed BCI Decoders Adapt? A Resource-Priced Ladder for Frozen EEG Trunks

**arXiv ID:** 2611.25315 | [PDF](https://arxiv.org/pdf/2611.25315v1)

**作者:** Sergey Musienko `[一作]` `[通讯]`, Sergey Musienko

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了在部署的脑机接口（BCI）解码器中，何时应花费适应资源，以及如何在数据漂移的情况下进行适应。

**💡 创新点**

提出了一种基于效用的门控机制，强调适应应当是有条件的，而非始终开启的，避免了在已经存在漂移的数据上进行过度适应。

**🔧 技术方法**

使用了贝叶斯头（LDA）和自适应控制层，结合了效用定价机制来决定是否进行适应。

**📊 数据集**

使用了多个数据集，包括BNCI2014-004、Zhou2016、Shin2017A和Kumar2024，涵盖了不同的实验设置和受试者。

**📈 对比分析**

与其他方法相比，轻量级的LDA头在五种编码器架构中表现出色，适应时间约为全量微调的23倍，且在准确性上与冻结的主干相当或更优。

**⚠️ 局限性**

适应的严重性并不遵循普遍的序数梯度，强调了动态门控的必要性，而不是固定的适应策略。

---

## 359. Digitization of Analogue Gauges using Computer Vision

**arXiv ID:** 2611.26262 | [PDF](https://arxiv.org/pdf/2611.26262v1)

**作者:** Erico Kayumba Mutonkole `[一作]` (University of the Western Cape), Omowunmi Isafiade `[通讯]` (University of the Western Cape)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究开发并评估了一种使用计算机视觉和深度学习技术的自动化仪表读取系统，旨在将模拟仪表的读数转换为数字等效值。

**💡 创新点**

创新点在于系统地比较了三种不同的方法：传统计算机视觉（OpenCV与Scikit-Image）、现代残差学习（OpenCV与ResNet18）和深度顺序学习（OpenCV与VGG19），并证明了残差学习架构在工业仪表数字化中的有效性。

**🔧 技术方法**

使用了OpenCV进行图像预处理，结合了深度学习框架PyTorch，具体实现了ResNet18和VGG19模型。

**📊 数据集**

使用了一个包含1848张压力表图像的数据集，这些图像通过网络抓取、合成生成和增强技术获得，涵盖了0-10 BAR的范围。

**📈 对比分析**

与VGG19和Scikit-Image相比，ResNet18在准确率上显著优越，达到了97.3%的准确率，而VGG19和Scikit-Image的准确率分别为64.4%和63.8%。

**⚠️ 局限性**

本研究的局限性在于仅关注于10 BAR的压力表，未来的研究可以扩展到其他类型的模拟仪表，如温度、电压或流量计。

---

## 360. Full Rank and Stable Sampling of $\tanh$ Networks

**arXiv ID:** 2611.25867 | [PDF](https://arxiv.org/pdf/2611.25867v1)

**作者:** Philipp Grohs `[一作]` (University of Vienna), Matěj Trödler `[通讯]` (University of Vienna)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了具有固定架构和P个参数的前馈tanh网络从有限多个点评估中恢复的过程。

**💡 创新点**

首次建立了稳定采样与神经网络实现映射导数的退化性之间的联系，并证明了几乎每个2P+1个采样点的元组在参数集的紧凑子集上具有一致的Lipschitz稳定性。

**🔧 技术方法**

使用了投影梯度下降和投影高斯-牛顿方法来证明局部收敛性。

**📊 数据集**

使用了2P+1个采样点的元组进行实验。

**📈 对比分析**

与其他方法相比，证明了在紧凑的参数集上，几乎每个2P+1个采样点的选择可以控制一致重建误差，并确保投影梯度下降和投影高斯-牛顿的局部收敛性。

**⚠️ 局限性**

限制在于对于某些参数，可能存在导数的线性依赖性，导致无法保证全秩条件。

---

## 361. When Foundation Models Disagree: Representational Heterogeneity for Calibrated Uncertainty in Whole-Slide Image Classification

**arXiv ID:** 2611.26159 | [PDF](https://arxiv.org/pdf/2611.26159v1)

**作者:** Yekai Shen `[一作]` (University of Science and Technology of China), Qingsong Yao `[通讯]` (Renmin University of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的框架，通过融合多个病理基础模型来进行全切片图像分类中的不确定性估计，旨在提高预测的可信度。

**💡 创新点**

创新点在于将模型之间的代表性异质性视为不确定性信号，并通过决策共识和证据一致性来进行不确定性校准。

**🔧 技术方法**

使用了多种技术，包括决策分支（共识条件池化）和证据分支（块级预测分布的Jensen-Shannon散度），并结合了多实例学习（MIL）方法。

**📊 数据集**

在七个全切片图像分类任务上进行了评估，使用了来自不同来源的五个病理基础模型（如UNI、GigaPath等）。

**📈 对比分析**

与最强的单一编码器基线和简单的集成平均方法相比，该框架在校准、错误检测和选择性预测方面表现更好，同时保持了竞争性的平衡准确率。

**⚠️ 局限性**

限制在于需要训练多个FM-nnMIL分支，成本随模型数量增加而增加；并且在某些任务上，λ值是固定的而非针对每个任务进行调优。

---

## 362. CGM: Chaos-Guided Graph Attention and Expert Routing for Autism Stereotypy Detection

**arXiv ID:** 2611.25730 | [PDF](https://arxiv.org/pdf/2611.25730v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 363. FEA solver integration of a composite World Model with a Spiking Neural Network memory Block for computationally efficient modelling of defects and variabilities

**arXiv ID:** 2611.26419 | [PDF](https://arxiv.org/pdf/2611.26419v1)

**作者:** Bassam El Said `[一作]` (Bristol Composites Institute, University of Bristol), Adam Sobey `[通讯]` (University of Southampton)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于多模态脉冲神经网络（MMST）架构的复合材料世界模型，用于实时模拟具有缺陷的复合材料的多尺度行为。

**💡 创新点**

创新点在于结合了去噪β变分自编码器、脉冲神经网络内存块和变换器模块，以有效跟踪材料的内部状态并进行多尺度模拟，同时解决了传统模型在计算效率和内存需求上的限制。

**🔧 技术方法**

使用了多模态脉冲神经网络（MMST）、去噪β变分自编码器（Dβ-VAE）、脉冲神经网络（SNN）和变换器模块等技术。

**📊 数据集**

使用了8400个独特的代表性体积元素（RVE）模型，这些模型包含不同的复合材料层叠、缺陷和变异性，数据集涵盖了多种加载条件下的应力/应变曲线。

**📈 对比分析**

与实验结果进行基准测试，模型在不同加载条件下的预测误差在0.4%到4.8%之间，显示出良好的准确性和可靠性。

**⚠️ 局限性**

限制在于模型的复杂性和训练过程中的计算成本，尤其是在处理大规模结构时，尽管已经通过模块化设计和递归训练来缓解这些问题。

---

## 364. Below Three: A 2.8027-Approximation for Moldable Precedence Scheduling

**arXiv ID:** 2611.25953 | [PDF](https://arxiv.org/pdf/2611.25953v1)

**作者:** Klaus Jansen `[一作]` (Kiel University), Felix Ohnesorge `[通讯]` (Kiel University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在相同处理器上具有任意优先级约束的不可抢占可塑作业的完工时间最小化问题。每个作业在执行过程中固定使用一定数量的处理器，增加处理器数量可以缩短执行时间，但可能增加总工作量并延迟其他作业。

**💡 创新点**

提出了一种确定性多项式时间算法，能够为全凸模型和任意数量的处理器提供2.8027的近似保证，突破了该模型的三倍界限，并改善了对凹速提升的近似保证。

**🔧 技术方法**

使用线性规划和列表调度技术，构建了一个固定的有限调度家族，并通过特定参数对分数解进行四舍五入。

**📊 数据集**

论文中没有具体提到使用的数据集，但提到的模型包括具有凹速提升的作业，且假设处理时间是非递增的，工作量是非递减的。

**📈 对比分析**

与之前的研究相比，提出的算法在处理具有任意优先级约束的可塑作业时，提供了更好的近似保证，尤其是在凹速提升的情况下，性能显著提升。

**⚠️ 局限性**

算法的局限性在于它假设处理时间表是凸的，并且在处理器数量较少（m≤5）时，可能需要更强的界限来保证性能。

---

## 365. What Does Anti-Hallucination Training Buy? Saying Less, Cheap Pruning, and the Rest

**arXiv ID:** 2611.25814 | [PDF](https://arxiv.org/pdf/2611.25814v1)

**作者:** Lingkai Bu `[一作]` (Qilu University of Technology), Qian Gao `[通讯]` (Qilu University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文探讨了偏好训练在视觉-语言模型中减少物体幻觉的效果，并分析了不同训练方法的效果。

**💡 创新点**

创新点在于提出了一种匹配召回的分解协议，用于评估训练方法在减少幻觉方面的贡献，并引入了一个弱的后验探测器来进行比较。

**🔧 技术方法**

使用了一个包含11个特征的逻辑回归模型作为后验探测器，该探测器在500个标注图像上进行训练并被冻结。

**📊 数据集**

使用了COCO、AMBER和Object HalBench等多个数据集进行评估。

**📈 对比分析**

与基线模型的比较显示，经过训练的模型加上探测器在COCO和AMBER上表现优于简单的截断方法，且在大多数情况下探测器能够重现训练所去除的幻觉部分。

**⚠️ 局限性**

限制在于依赖于自动评估器的准确性，评估结果显示人类对评估器的判断一致性较低，且探测器的性能可能受到训练数据重叠的影响。

---

## 366. PAD: Policy-Aware Data Valuation for Autonomous Driving

**arXiv ID:** 2611.25604 | [PDF](https://arxiv.org/pdf/2611.25604v1)

**作者:** Chengkai Xu `[一作]` (Tongji University), Jian Sun `[通讯]` (Tongji University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为政策感知数据估值（PAD）的框架，用于评估驾驶演示的价值，具体针对目标驾驶策略的下游效用。

**💡 创新点**

创新点在于通过结合场景级校准反馈、投影策略梯度和局部训练几何的正则化近似，提供可扩展的基于影响的价值评估，从而实现针对特定策略的数据选择。

**🔧 技术方法**

使用了影响函数和可扩展的梯度近似技术，结合独立的场景级政策结果进行评估。

**📊 数据集**

使用了DiffusionDrive数据集，该数据集包含真实和合成的驾驶演示。

**📈 对比分析**

与传统的真实数据训练方法相比，PAD在NavTest上提高了3.5分，在NavHard上提高了5.0分，且在不同的规划器中表现出一致的性能提升。

**⚠️ 局限性**

局限性在于当前研究仅基于局部的一阶近似，并且评估主要集中在NAVSIM中的合成候选者，未来工作将扩展到异构的真实世界数据和更广泛的闭环条件。

---

## 367. Cost-Aware Agentic Feature Optimization System for Fleet-Scale Ranking Models

**arXiv ID:** 2611.26531 | [PDF](https://arxiv.org/pdf/2611.26531v1)

**作者:** Chang Yang `[一作]` (Meta, Inc.), Djordje Gligorijevic `[通讯]` (Meta, Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种成本敏感的特征优化系统，旨在在资源约束下优化多个排名模型的特征选择。

**💡 创新点**

创新点在于使用LLM驱动的特征创建、特征重要性混合算法以及无人工干预的特征选择系统，能够在每次刷新时重新选择特征。

**🔧 技术方法**

采用了大型语言模型（LLM）进行特征生成和选择，结合了确定性工具进行实验验证。

**📊 数据集**

使用了多个生产模型的数据集，具体数据集未详细说明。

**📈 对比分析**

与传统方法相比，该系统在每个模型的预算内平均提高了0.31%的归一化熵，并且通过修剪步骤将管道成本降低了2.3%，在在线A/B测试中提高了主要在线指标0.20%。

**⚠️ 局限性**

限制在于每次评估需要数小时到数天，且每个会话只能进行10-25次试验，可能导致搜索空间的限制和效率问题。

---

## 368. A new algorithmic approach for finding flat minima: critically damped Polyak stepsizes

**arXiv ID:** 2611.26475 | [PDF](https://arxiv.org/pdf/2611.26475v1)

**作者:** Francesco Caporali `[一作]`, Boris Hanin `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的自适应步长规则，旨在使梯度下降偏向于更平坦的最小值，并为一类局部二次目标函数建立了首次统一的时间保证，涵盖了尖锐度和损失减少。

**💡 创新点**

创新点在于引入了临界阻尼的Polyak（CDP）步长规则，通过乘以4和轻微的阻尼因子来调整经典的Polyak规则，从而在理论和实践中实现了对平坦最小值的偏向。

**🔧 技术方法**

使用了临界阻尼的Polyak步长规则，该规则通过损失和梯度的当前值自适应地计算步长。

**📊 数据集**

在平滑的过参数化神经网络中，使用了一类具有局部二次Morse-Bott结构的非负损失函数进行实验。

**📈 对比分析**

与其他方法（如Sharpness-Aware Minimization）相比，CDP步长在多个实验中表现出类似的平坦化效果，能够在理论之外的设置中减少尖锐度，尤其是在非线性教师-学生网络和预训练大语言模型的微调中。

**⚠️ 局限性**

限制在于理论结果不直接适用于交叉熵损失，因为交叉熵损失不一定在有限参数下达到其下界。

---

## 369. SKILL.PNG: Learning Agent Skills in Pixel Space

**arXiv ID:** 2611.26296 | [PDF](https://arxiv.org/pdf/2611.26296v1)

**作者:** Maximilian Seeth `[一作]` (Technical University of Munich), Leo Schwinn `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Skill.png的方法，通过对图像输入进行梯度优化，来为基于视觉-语言模型（VLM）的智能体提供技能。

**💡 创新点**

Skill.png是首个可以通过梯度下降优化的智能体技能，同时保留文本技能的迁移优势。

**🔧 技术方法**

使用了梯度下降技术来优化图像输入，并通过冻结的VLM进行处理。

**📊 数据集**

在四个视觉-语言模型（VLMs）上进行了评估，包括Qwen3.5-4B、Qwen3.5-9B、Gemma-4-E4B-it和Nemotron-3-Nano-Omni-30B-A3B。

**📈 对比分析**

Skill.png在16个模型-基准对中有9个设置显著优于使用前沿LLM优化的文本技能，且在推理时仅需64到256个图像标记，而文本技能则需约900到2000个标记。

**⚠️ 局限性**

在TRAJECT基准上，Skill.png未能显著提升性能，可能是由于该基准对工具调用和参数的严格匹配要求。

---

## 370. Planning Width Is Hard to Approximate Even Given the Unique Optimal Plan

**arXiv ID:** 2611.25484 | [PDF](https://arxiv.org/pdf/2611.25484v1)

**作者:** Viet Bac Nguyen `[一作]` `[通讯]` (VNU University of Engineering and Technology), Viet Bac Nguyen (VNU University of Engineering and Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在已知唯一最优计划的情况下，如何近似规划宽度的问题，特别是对于给定的宽度阈值k，确定宽度是否小于等于k的复杂性。

**💡 创新点**

创新点在于识别出哪些状态限制了宽度证书，并且在原始计划已知的情况下，提供了宽度近似的界限，揭示了在某些情况下宽度的近似是NP-hard的。

**🔧 技术方法**

使用了STRIPS任务模型，通过构造布尔赋值来确定哪些状态与给定计划在单一深度上竞争，并将集合宽度和链宽度表示为击中数。

**📊 数据集**

使用了基于3-CNF公式构造的STRIPS任务，涉及的状态图是显式的，且每个可达状态的宽度与原子数量N相关。

**📈 对比分析**

与现有方法的比较显示，尽管提供了唯一最优计划，但在某些情况下，宽度可以是1或Θ(N)，并且区分这两种情况是NP-hard的。对于给定的状态图，贪婪算法能够在对数因子内近似集合宽度。

**⚠️ 局限性**

限制在于，除非P=NP，否则没有多项式时间算法能够实现一侧因子o(N)或对称因子o(√(N))的近似。对于正前提的STRIPS任务，测试集合或链宽度的复杂性在固定阈值k≥1时是coNP-complete的。

---

## 371. Zero to Four Views: Unified Panorama Generation and Preference RL for Unobserved Regions

**arXiv ID:** 2611.25389 | [PDF](https://arxiv.org/pdf/2611.25389v1)

**作者:** Rui Hong `[一作]` (George Mason University), Kai Fan `[通讯]` (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种单一模型，该模型能够从文本描述和最多四个条件视图生成完整的全景图。

**💡 创新点**

创新点在于模型能够处理多种输入情况，包括仅有文本描述或多个视图，并且通过UR-GRPO强化学习方法改善未覆盖区域的生成效果。

**🔧 技术方法**

使用了基于修正流的变换器（rectified-flow transformer）进行文本到图像的生成，并结合了强化学习技术来优化生成结果。

**📊 数据集**

使用了Structured3D、Pano360、PolyHaven和Humus等数据集，共计97,190个全景图进行训练和测试。

**📈 对比分析**

与之前的最先进模型360Anything进行比较，模型在Laval Indoor数据集上表现优于360Anything，并在SUN360的室内子集上结果混合；第二阶段的训练在每种条件视图数量下均改善了结果。

**⚠️ 局限性**

限制在于测试数据仅包含室内场景，SUN360的结果无法与早期论文进行比较；在只有一到两个视图时，模型的方向性不总是准确；模型尚未处理全景图的接缝问题，且评估方法仅基于裁剪区域，未能全面评估全局一致性。

---

## 372. Temporal Extrapolation of Elastic Seismic Wave Propagation in Layered Media Using Neural Operators

**arXiv ID:** 2611.25130 | [PDF](https://arxiv.org/pdf/2611.25130v1)

**作者:** Dibyajyoti Nayak `[一作]` (Johns Hopkins University), Somdatta Goswami `[通讯]` (Johns Hopkins University)

**通讯引用:** 8042 | [OpenAlex ID](https://openalex.org/A5015683810)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究开发了一种耦合的神经算子框架，用于在分层弹性介质中进行应力条件下的时间预测，特别是针对地震速度场的预测。

**💡 创新点**

创新点在于结合了时间积分嵌入的深度算子网络（TI-DeepONet）和傅里叶神经算子（FNO），实现了对速度场的时间预测和空间修正。

**🔧 技术方法**

使用了时间积分嵌入的深度算子网络（TI-DeepONet）和傅里叶神经算子（FNO）技术。

**📊 数据集**

使用了四层土柱的数值案例，固定总深度为50米和280米，激励信号为第三阶高斯导数波形（DOG3），中心频率在0.5到6 Hz之间。

**📈 对比分析**

通过与数值参考解进行比较，评估了模型性能。四个案例的平均峰值归一化均方根误差（NRMSE）在全评估窗口内范围为0.53%到2.31%，在外推区间内范围为0.61%到2.82%。

**⚠️ 局限性**

限制在于模型在某些配置下的预测性能存在显著差异，尤其是在反射到达和晚期运动的预测上需要进一步改进。

---

## 373. Coordinated Use of Large Language Models in Academic Research

**arXiv ID:** 2611.25502 | [PDF](https://arxiv.org/pdf/2611.25502v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 374. Bridging vision-language reasoning and learned control for legged navigation in the wild

**arXiv ID:** 2611.25215 | [PDF](https://arxiv.org/pdf/2611.25215v1)

**作者:** Takahiro Miki `[一作]` (ETH Zurich), Marco Hutter `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本论文提出了一种名为ROAM的系统，旨在利用视觉语言模型（VLM）的常识推理能力，帮助四足机器人在复杂的真实环境中实现自主导航。

**💡 创新点**

创新点在于将高层次的常识推理与低层次的控制策略相结合，形成一个混合架构，使机器人能够在多样化的环境中进行有效的导航，而无需大量特定环境的数据集。

**🔧 技术方法**

使用了视觉语言模型（VLM）和强化学习（RL）策略，结合了语义分割和几何信息来实现环境感知和导航控制。

**📊 数据集**

在多个环境中进行了实验，包括瑞士阿尔卑斯山的长途徒步旅行、城市区域和室内空间，展示了系统的广泛适应性。

**📈 对比分析**

与传统的几何导航系统相比，ROAM在处理环境变化和语义理解方面表现出更好的鲁棒性，能够有效避免障碍物并遵循语义规范。实验结果显示，ROAM在多种环境中实现了高达96.7%的自主率。

**⚠️ 局限性**

限制在于VLM的语义解释质量依赖于模型的准确性，且在极端视觉条件下的表现尚未充分测试。此外，当前系统依赖于全球参考路径，缺乏对全球地图的充分解释能力。

---

## 375. Streaming-AAA: Streaming AAA Game Video Generation at High Resolution, High Frame Rate, and Low Latency

**arXiv ID:** 2611.26040 | [PDF](https://arxiv.org/pdf/2611.26040v1)

**作者:** Zhihao Hu `[一作]` (Tencent), Wei Yang `[通讯]` (Tencent)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的可控无限生成AAA级游戏视频的框架Streaming-AAA，能够在NVIDIA A800 GPU上以1080p分辨率生成43 FPS的游戏视频，并实现198毫秒的低延迟交互。

**💡 创新点**

创新点在于同时实现高视觉保真度、高帧率和低交互延迟，解决了现有方法在可玩性和生成质量上的不足。

**🔧 技术方法**

使用了改进的变分自编码器（VAE）架构和基于UNet的潜在扩散模型，结合多模态条件控制和混合短期与长期KV缓存内存。

**📊 数据集**

构建了一个包含约22,000个视频的高保真数据集，每个视频时长1分钟，分辨率为1920×1080，帧率为30 FPS，覆盖超过5平方公里的游戏场景。

**📈 对比分析**

与现有的可控视频生成方法相比，Streaming-AAA在视觉保真度、帧率、延迟和长期稳定性上显著提升，能够支持无限长度的游戏播放而不出现视觉崩溃。

**⚠️ 局限性**

当前方法主要针对特定游戏进行了优化，模型在未见过的游戏环境、艺术风格或游戏机制上的泛化能力有限。

---

## 376. SkillBridge: Cost-Aware Diagnosis and Repair of Cross-OS Compatibility Defects in Agent Skills

**arXiv ID:** 2611.26256 | [PDF](https://arxiv.org/pdf/2611.26256v1)

**作者:** Anjie Xu `[一作]` (Peking University), Leye Wang `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

提出了一种基于合同的工作流程，用于诊断和修复跨操作系统兼容性缺陷，涵盖静态分析、动态验证和修复三个阶段。

**💡 创新点**

定义了代理技能的跨操作系统兼容性问题，并提出了一种成本意识的工作流程来提高诊断和修复的效率。

**🔧 技术方法**

使用静态分析和动态验证技术，结合多代理工作流程进行缺陷诊断和修复。

**📊 数据集**

使用了1090个技能在Linux、macOS和Windows三个操作系统上的3270个技能-目标操作系统案例进行评估。

**📈 对比分析**

通过静态分析筛选出需要动态验证的技能-操作系统对，动态验证的效率显著提高，确认的操作系统问题数量达到38个，而随机选择需要865次验证才能达到相同的预期问题数量。

**⚠️ 局限性**

该方法的局限性在于动态执行的结果依赖于代理、任务生成器、运行配置等因素，可能无法完全覆盖所有操作系统的兼容性问题。

---

## 377. Interpreting DeepStack-Style Vision-Language Models: Do Input or Deeper-Layer Visual Tokens Drive Answers?

**arXiv ID:** 2611.25713 | [PDF](https://arxiv.org/pdf/2611.25713v1)

**作者:** Yucheng Du `[一作]` (University of Southern California), Xiyang Hu `[通讯]` (Arizona State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了DeepStack风格的视觉-语言模型中不同输入条目的作用，特别是主输入和后续输入在模型决策中的影响。

**💡 创新点**

创新点在于通过对同一图像的不同版本进行测试，揭示了后续输入在模型已经接收到图像后所能提供的额外信息和影响。

**🔧 技术方法**

使用了DeepStack、Qwen3-VL和Granite Vision等视觉-语言模型，并进行了多种实验以比较不同输入条目的效果。

**📊 数据集**

使用了GQA、Visual Genome等数据集，以及合成图像和短文本进行实验。

**📈 对比分析**

通过对比不同模型在相同输入下的表现，发现DeepStack的后续输入在决策中占主导地位，而Qwen3-VL则主要依赖主输入。性能方面，DeepStack在后续输入的影响下，能够在图像质量下降时仍保持较高的准确性。

**⚠️ 局限性**

限制在于实验主要集中在特定的图像编辑和任务上，可能无法全面反映模型在其他类型输入或任务下的表现。

---

## 378. GRODE-RTL: Source-Mapped Dependency Views for LLM-Based RTL Repair

**arXiv ID:** 2611.25661 | [PDF](https://arxiv.org/pdf/2611.25661v1)

**作者:** Jingyi Zhou `[一作]` (Chinese University of Hong Kong), Qiang Xu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为GRODE-RTL的锚点中心表示框架，用于解决非局部RTL故障的自动修复问题。

**💡 创新点**

GRODE-RTL通过提取可疑寄存器的单时钟阶段扇入锥，并将其映射回可编辑的源代码行，提供了一种新的表示方法，解决了传统方法在表示上的困境。

**🔧 技术方法**

使用了锚点中心的结构表示方法，包括相关代码过滤、显式依赖组织和选择性图压缩等技术。

**📊 数据集**

在30个经过正式验证的非局部故障上进行了评估，使用了三个大型语言模型（DeepSeek-V4-Flash、GLM-5.2和MiniMax-M3）。

**📈 对比分析**

与源级探索方法相比，GRODE-RTL在DeepSeek上提高了16.7%的Pass@1，在GLM-5.2上提高了26.7%，在MiniMax-M3上提高了20.0%。在难以检索的远距离和间接锚点子集上，所有三个模型均表现出一致的提升。

**⚠️ 局限性**

GRODE-RTL的局限性在于它主要针对结构相关的故障，对于需要语义推理的不可观察故障的修复效果有限。

---

## 379. Training LLMs to Follow Reasoning Instructions with Minimal Edits

**arXiv ID:** 2611.26342 | [PDF](https://arxiv.org/pdf/2611.26342v1)

**作者:** Alexandra Yakovleva `[一作]` (Aalto University), Alexander Ilin `[通讯]` (System 2 AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于编辑的训练框架，通过局部文本修正来学习模型自身的推理轨迹。

**💡 创新点**

创新点在于集中监督于模型推理首次失败的点，而不是对整个轨迹进行模仿或惩罚，从而提高模型遵循推理指令的能力。

**🔧 技术方法**

使用了编辑模型来识别推理轨迹中的首次失败点，并提出最小替换范围进行监督。

**📊 数据集**

在ReasonIF基准上使用了Qwen3-32B模型进行实验。

**📈 对比分析**

与全轨迹模仿和轨迹级强化学习基线相比，基于编辑的训练显著提高了推理指令的遵循率，同时在未针对任务上保持了模型的技能。

**⚠️ 局限性**

局限性在于ReasonIF类别仅为可验证的代理，未涵盖推理可控性的全部需求，且编辑数据集是一次生成的，未针对每个种子重新生成。

---

## 380. Protein Functional Semantic Alignment via Representation Rectification

**arXiv ID:** 2611.25396 | [PDF](https://arxiv.org/pdf/2611.25396v1)

**作者:** Chuyang Zhou `[一作]` (University of Sydney), Chang Xu `[通讯]` (University of Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为功能语义对齐通过表示修正（SAVR）的对比后训练方法，旨在通过使用Pfam家族共成员关系作为同源性引导的离散语义信号来重塑蛋白质语言模型（PLM）的表示。

**💡 创新点**

SAVR通过减少方向主导性和重新分配嵌入维度的方差，提高了嵌入协方差谱的有效维度，进而改善了基于序列的蛋白质功能预测。

**🔧 技术方法**

使用了对比学习的后训练技术，结合了参数高效的低秩适配器（LoRA）来实现轻量级的后训练。

**📊 数据集**

使用了来自UniRef50的蛋白质序列数据集，并通过InterProScan注释了Pfam家族。

**📈 对比分析**

与传统的对比学习方法相比，SAVR在酶委员会（EC）基准测试中取得了最先进的结果，并在基因本体（GO）基准测试中表现出竞争力，特别是在冻结骨干网络的评估下。

**⚠️ 局限性**

SAVR的主要限制在于Pfam共成员关系是一个粗略的蛋白质级信号，未能显式建模域顺序、域结构或残基级功能位点，且评估范围仅限于蛋白质级功能预测。

---

## 381. Coda-DaVinci: Reimagining Video Generation with Code2Video

**arXiv ID:** 2611.26595 | [PDF](https://arxiv.org/pdf/2611.26595v1)

**作者:** Yiyan Ji `[一作]` (Nanjing University), Jiaheng Liu `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文研究了如何在Code2Video范式下协调编码代理和视频生成模型，以提高视频生成的性能，特别是在场景重访、复杂运动和物理过程模拟等任务中。

**💡 创新点**

创新点在于提出了一种任务自适应的代理框架，该框架协调选择性3D建模和基于代理的合成，利用持久的场景工作空间和视觉反馈来指导生成过程。

**🔧 技术方法**

使用了编码代理和视频生成模型的组合，结合Blender进行3D建模和物理模拟，采用了多代理系统来维护场景状态。

**📊 数据集**

使用了多个数据集进行评估，包括WorldMemBench（场景重访）、VGIF-Bench（复杂指令跟随）和Principia（物理过程模拟），以及T2AV-Compass（视频现实感评估）。

**📈 对比分析**

通过与直接生成方法的比较，论文展示了在一致性和指令跟随方面的显著提升，尤其是在物理逼真度和视觉现实感上，表现优于基线方法。

**⚠️ 局限性**

限制在于在复杂人类和物体运动的任务中，仍然存在物理逼真度、自然性和现实感的不足，尤其是在长视频生成和场景设置遵循方面的稳定性较低。

---

## 382. MemMend: A Selective Revision Framework for Memory Question Answering

**arXiv ID:** 2611.25721 | [PDF](https://arxiv.org/pdf/2611.25721v1)

**作者:** Sheng Shi `[一作]` (Independent Research), Geng Li `[通讯]` (Independent Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为MemMend的模块化选择性修订框架，用于处理长时记忆中的回答修订问题。

**💡 创新点**

创新点在于引入了一种选择性修订策略，通过可观察的问题和答案属性来分配修订调用，并将源链接的本地修正形式化为精确的编辑，具有确定性的接受规则。

**🔧 技术方法**

使用了选择性修订策略、源链接本地修正和控制修订等技术。

**📊 数据集**

使用了1986个LoCoMo开发问题和451个LongMemEval-V2小问题的数据集。

**📈 对比分析**

与经过验证的修订链相比，MemMend在F1得分上提高了0.82点，Judge得分提高了1.24点，同时减少了62.37%的修订调用。与随机分配的路由相比，MemMend在F1得分上提高了1.96点。

**⚠️ 局限性**

修订决策仍有很大的改进空间，表面规则可能会错过短但不正确的答案，并且在需要跨多个记录进行重组或综合时，精确文本补丁可能过于严格。

---

## 383. ChronoDossier: Long-Form Personas as Temporal Memory Representations

**arXiv ID:** 2611.25407 | [PDF](https://arxiv.org/pdf/2611.25407v1)

**作者:** Noah Sturgill `[一作]` (Emory University), Jinho D. Choi `[通讯]` (Emory University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一种长形式的人物档案作为记忆表示，旨在评估在事实变化时如何恢复个人状态。研究构建了311个合成档案，使用七个生成器生成，评估了记录的内在和外在指标。

**💡 创新点**

创新点在于将长形式的人物档案视为已知潜在状态的记忆表示，并通过合成数据集进行评估，提供了一种新的基准测试方法。

**🔧 技术方法**

使用了多种生成器（如GPT-5.1、Sonnet 4.6等）生成合成档案，并采用内在和外在指标进行评估。

**📊 数据集**

使用了311个合成档案，字数在1000到10000之间，生成自不同的时间线。

**📈 对比分析**

通过与五个读者的比较，评估了模型在不同长度档案上的恢复能力。结果显示，随着档案长度的增加，恢复准确率下降，整体准确率在0.705到0.745之间。

**⚠️ 局限性**

限制在于模型评估的可靠性和生成的合成档案可能无法代表真实人群，且未能涵盖所有潜在的记忆获取和验证过程。

---

## 384. WAMotor: Accelerating World Action Models via Speculative Inference under Dual Nonuniform Denoising Dynamics

**arXiv ID:** 2611.25413 | [PDF](https://arxiv.org/pdf/2611.25413v1)

**作者:** Zexun Lin `[一作]` (University of Science and Technology of China), Xike Xie `[通讯]` (University of Science and Technology of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种新的推测推理方法WAMotor，旨在加速世界动作模型（WAMs）的推理过程，特别是针对多步去噪延迟的问题。

**💡 创新点**

创新点在于识别并利用WAMs中的双重非均匀性，提出了可靠锚速度选择和时间步长感知验证机制，从而提高了推测推理的效率和准确性。

**🔧 技术方法**

使用了推测推理、可靠锚速度选择和时间步长感知验证等技术。

**📊 数据集**

在LIBERO、RoboTwin 2.0和LIBERO-Plus等数据集上进行了广泛评估。

**📈 对比分析**

与现有的基线模型相比，WAMotor在保持任务成功率的同时，实现了2.86×到4.28×的端到端加速，且在多个基准测试中表现优异。

**⚠️ 局限性**

限制在于WAMotor的性能依赖于对锚速度的选择和时间步长的验证，可能在某些极端情况下仍然面临推理延迟的问题。

---

## 385. TrajCal: Trajectory-Calibrated Time Schedules for Few-Step Flow Sampling

**arXiv ID:** 2611.25847 | [PDF](https://arxiv.org/pdf/2611.25847v1)

**作者:** Junchi Zhang `[一作]` (Fudan University), Hua Zhang `[通讯]` (Institute of Information Engineering, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为TrajCal的训练无关的校准框架，用于少步采样，以提高图像和视频生成的参考保真度。

**💡 创新点**

TrajCal通过散射搜索构建多样化的候选调度，并通过混合和变形扩展候选空间，从而选择一个可重用的调度，显著提高了输出的参考保真度。

**🔧 技术方法**

使用了散射搜索作为优化算法，结合了加权动态规划、基于加速的分配和步长搜索等技术。

**📊 数据集**

使用了FLUX和Wan数据集进行图像和视频生成的实验，分别使用COCO和VBench子集进行校准和评估。

**📈 对比分析**

与现有的基线方法相比，TrajCal在低评估预算下实现了更高的参考保真度，同时保持了竞争性的视觉质量，尤其在12次和16次评估中表现优异。

**⚠️ 局限性**

TrajCal需要耗时的候选评估过程，并且所选调度在不同输入间共享，不会根据新输入调整时间分配。

---

## 386. SiliconGym: A Versatile Reinforcement Learning Environment for Agentic EDA

**arXiv ID:** 2611.25658 | [PDF](https://arxiv.org/pdf/2611.25658v1)

**作者:** Hejia Zhang `[一作]` (University of California San Diego), Jishen Zhao `[通讯]` (University of California San Diego)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文研究了某种技术或方法的应用，具体内容未详细说明。

**💡 创新点**

创新点在于提出了一种新的方法或改进了现有技术，具体细节未提供。

**🔧 技术方法**

使用了先进的技术，具体技术未详细说明。

**📊 数据集**

使用了特定的数据集，具体数据集名称未提供。

**📈 对比分析**

与其他方法进行了比较，性能表现优越，具体比较结果未详细说明。

**⚠️ 局限性**

存在一些局限性，具体限制未提供。

---

## 387. Rubric Semantics or Numerical Priors? Auditing Scale-Direction Bias in LLM Judges

**arXiv ID:** 2611.25588 | [PDF](https://arxiv.org/pdf/2611.25588v1)

**作者:** Chanwoo Kim `[一作]` (Soongsil University), Chanjun Park `[通讯]` (Soongsil University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了大型语言模型（LLMs）作为自动评估者的有效性，提出了语义忠实性这一概念，强调评估应基于评估标准的语义意义而非数值标度。通过引入尺度反转探针和地板归一化方法，量化了模型在评估中的偏差。

**💡 创新点**

创新点在于提出了语义忠实性这一新概念，并通过尺度反转探针和地板归一化方法来评估和校准模型的评估一致性，强调了模型应遵循评估标准的语义而非仅仅依赖数值。

**🔧 技术方法**

使用了尺度反转探针和地板归一化技术来量化和校准模型的评估偏差，此外还引入了双向锚点平均法来减少对单一锚点方向的依赖。

**📊 数据集**

实验使用了五个元评估基准，涵盖了七种不同的LLM评估模型，包括开放权重和专有模型，确保了结果的广泛性和可靠性。

**📈 对比分析**

通过与人类评估的对比，发现当前的LLM评估者普遍存在语义忠实性违反的问题，且不同模型的表现差异显著。双向锚点平均法在保持与人类评估一致性的同时，改善了模型的评估鲁棒性。

**⚠️ 局限性**

限制在于仅评估了七种LLM模型，且实验设计仅限于整数序数评分，未涵盖连续评分和二元判决。此外，双向锚点平均法虽然减少了对单一评分方向的依赖，但增加了推理成本。

---

## 388. Compressing Page-Table Leaves in Shared Caches

**arXiv ID:** 2611.24992 | [PDF](https://arxiv.org/pdf/2611.24992v1)

**作者:** Samarth Narang `[一作]` (New York University), Brandon Reagen `[通讯]` (New York University)

**通讯引用:** 5705 | [OpenAlex ID](https://openalex.org/A5089173037)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种名为LeafPack的共享缓存组织，通过压缩架构页面表叶行来提高缓存效率，解决了页面表行与应用数据之间的竞争问题。

**💡 创新点**

创新点在于通过压缩相邻的页面表叶行，最大限度地提高缓存的有效容量，同时保持每行的一致性，并在需要时回退到未压缩存储。

**🔧 技术方法**

使用了压缩技术和在线控制器，结合了集对抗（set dueling）和阴影标签（shadow tags）来动态选择压缩和未压缩的页面表方式上限。

**📊 数据集**

在全系统gem5模拟中评估了LeafPack，使用了多种工作负载，包括翻译密集型工作负载和未见的持出混合工作负载。

**📈 对比分析**

与普通共享缓存相比，LeafPack在九个翻译密集型工作负载上分别提高了28.7%和28.4%的几何平均谐波加速；在使用SRRIP替换的普通缓存上，分别提高了18.7%和17.7%。在二十个未见的持出混合中，LeafPack在每个混合上都提高了谐波加速，几何平均增益为21.4%和18.5%。

**⚠️ 局限性**

限制在于与数据敏感的共同运行程序（如gcc）一起运行时，可能会导致数据容量竞争，从而影响性能。

---

## 389. Do Deep Stereo Models Learn Shape Priors?

**arXiv ID:** 2611.25113 | [PDF](https://arxiv.org/pdf/2611.25113v1)

**作者:** Linbo Tang `[一作]` (Harvard University), Todd Zickler `[通讯]` (Harvard University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了深度立体模型在局部图像和视差线索较弱时，是否能产生与简单形状先验一致的边界，并提出了一种生成式立体模型来改善边界定位。

**💡 创新点**

创新点在于引入了一种简单的生成式立体模型，该模型联合预测视差和表示物体边界的距离场，从而提高了边界定位的准确性。

**🔧 技术方法**

使用了扩散模型（HDiT-Stereo）作为生成式框架，结合了图像特征和对应特征来生成视差和边界距离场。

**📊 数据集**

创建了一个合成立体数据集，包含从高度约束的形状家族中提取的物体，随机化深度、倾斜、颜色和纹理。

**📈 对比分析**

与多种最先进的立体模型（如RAFT-Stereo、Selective-IGEV等）进行了比较，结果显示HDiT-Stereo在每个边界度量上都表现出最高的准确性，尤其是在边界附近的误差显著减少。

**⚠️ 局限性**

局限性包括实验使用了狭窄的合成分布，未能验证模型在更复杂或真实世界场景中的表现，同时HDiT-Stereo在训练期间需要边界场的监督，且重复采样会增加推理成本。

---

## 390. Online Temporal Rotations for 2 Bit KV-Cache Quantization in Video Diffusion Transformers

**arXiv ID:** 2611.26334 | [PDF](https://arxiv.org/pdf/2611.26334v1)

**作者:** Ryan Lucas `[一作]` (Massachusetts Institute of Technology), Rahul Mazumder `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的视频扩散变换器的KV缓存量化方法，称为时间旋转量化（TRQ），通过对时间上相邻的缓存向量进行旋转来减少量化误差。

**💡 创新点**

TRQ的创新点在于它利用了视频中相邻帧的逐通道均值的缓慢变化，适应性地为每个时间块拟合旋转，从而使量化器的偏移量能够吸收共享成分。

**🔧 技术方法**

使用了树形的2×2平面旋转和固定的Hadamard变换来实现旋转，且旋转角度以两位二进制数存储。

**📊 数据集**

在LongCat-Video-13B和HY-WorldPlay-8B数据集上进行了实验。

**📈 对比分析**

与现有的QVG方法相比，TRQ在相同压缩水平下在多个视频质量指标上表现更好，LongCat-Video-13B的主观一致性从90.38提高到92.02，HY-WorldPlay-8B从86.33提高到87.61。

**⚠️ 局限性**

TRQ的局限性在于它依赖于时间冗余，对于快速运动或场景切换等内容的处理可能不够理想，未来可以考虑根据场景动态调整块长度。

---

## 391. Ask in the Crowd: Differentially Private LLM Inference via Dummy-Augmented Shuffling

**arXiv ID:** 2611.25669 | [PDF](https://arxiv.org/pdf/2611.25669v1)

**作者:** Zhihao Liu `[一作]` (Zhejiang University), Yuke Hu `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种不同ially private LLM推理框架DP-SPD，通过在真实查询中隐藏多个客户端生成的虚拟查询来保护用户查询的隐私。

**💡 创新点**

DP-SPD通过共享前缀结构和客户端随机打乱，改善了隐私、效用和效率之间的平衡，提供了强大的推理时隐私保护和接近明文的效用。

**🔧 技术方法**

使用了差分隐私（d_χ-隐私）和共享前缀的机制，结合了客户端生成的虚拟后缀和随机打乱技术。

**📊 数据集**

在多个数据集上进行了实验，包括IMDb电影评论数据集、SST-2、QNLI、WikiText-2、Stanford Alpaca、Databricks-Dolly-15k、CNN/DailyMail和WildChat。

**📈 对比分析**

与现有的基于扰动的方法（如DP-Forward、CAPE等）进行比较，DP-SPD在隐私保护和效用上表现更佳，尤其在小隐私预算下，保持了接近明文的分类准确率和生成质量。

**⚠️ 局限性**

当前框架采用固定的前缀-后缀分割，未明确识别查询中敏感信息的位置，且分析集中于单查询推理，未考虑多轮对话或长上下文设置的情况。

---

## 392. GenAI-DBT: Evaluating a DBT-Informed Generative AI Platform for Mindfulness Skills Training: A Pilot Randomized Study

**arXiv ID:** 2611.25428 | [PDF](https://arxiv.org/pdf/2611.25428v1)

**作者:** Jingchen Ye `[一作]` (Duke Kunshan University), Yucheng Jin `[通讯]` (Duke Kunshan University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究评估了三种数字格式在辩证行为疗法（DBT）技能训练中的可行性、可接受性和参与度，并探索了初步的情绪相关结果。

**💡 创新点**

创新点在于开发了GenAI-DBT，一个结合了个性化和情境敏感指导的对话式平台，超越了静态网络资源和通用大型语言模型（LLM）聊天机器人。

**🔧 技术方法**

使用了生成性人工智能（GenAI）技术，结合了DBT特定的提示工程和情境基础的技能应用评估。

**📊 数据集**

数据集包括45名大学生的匿名参与者数据，参与者被随机分配到Web、Chatbot或GenAI-DBT三种条件中。

**📈 对比分析**

与Web和Chatbot相比，GenAI-DBT在参与者的每周遵守率上表现最佳（71.4%），但在MAAS和DERS的变化上没有显著差异。GenAI-DBT在文本情感上显示出更大的积极变化（平均变化4.28 vs 1.59，P=.02）。

**⚠️ 局限性**

限制在于本研究未能确定临床有效性，且样本量较小，结果的普遍性受到限制。参与者反馈中提到的响应延迟、重复性和个性化不足也是需要改进的地方。

---

## 393. Coherent belief and opinion propagation produces more echo chambers: corrected version

**arXiv ID:** 2611.25879 | [PDF](https://arxiv.org/pdf/2611.25879v1)

**作者:** Hiro Kataoka `[一作]` (University of Tsukuba), Koji Hasebe `[通讯]` (University of Tsukuba)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了信念对社交网络中回音室的影响，提出了一种模型来定义意见和信念之间的相互作用。

**💡 创新点**

创新点在于引入信念的传播机制，并提出了新的度量标准来识别回音室，强调了信念和意见之间的协调性对回音室形成的影响。

**🔧 技术方法**

使用了基于意见动态和信念变化的模型，结合了价值基础机制来维护意见和信念的一致性。

**📊 数据集**

使用了模拟数据集，通过实验验证了模型的有效性。

**📈 对比分析**

与现有方法相比，提出的模型能够更准确地识别回音室，并且实验结果表明，保持意见和信念的一致性会导致更多的回音室形成。

**⚠️ 局限性**

限制在于模型未考虑不同价值观和主题的影响，这可能导致结果的差异。

---

## 394. What One Sample Cannot See: Why Fewer Denoising Steps Can Score Higher on Physics Benchmarks

**arXiv ID:** 2611.25632 | [PDF](https://arxiv.org/pdf/2611.25632v1)

**作者:** Yutao Li `[一作]` (Harbin Institute of Technology), Chen Wang `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的评估视频生成模型的指标，称为Kernel Physics-IQ，旨在消除单一样本评分中的分散溢价问题。

**💡 创新点**

创新点在于通过引入两个独立样本来评估模型，从而消除因样本分散导致的评分偏差，强调模型应当不仅仅根据单一视频与真实记录的匹配程度来评估。

**🔧 技术方法**

使用了Kernel Physics-IQ作为新的评分标准，结合了全视频均方根误差（RMS）和阈值运动掩码的IoU。

**📊 数据集**

使用了66个中心视角场景的合成数据集，包含多个视频生成模型的输出。

**📈 对比分析**

与传统的Physics-IQ评分方法相比，Kernel Physics-IQ在所有五个图像到视频模型中表现出更低的评分，表明传统方法可能因分散溢价而高估了模型性能。

**⚠️ 局限性**

限制在于需要两个独立样本来计算Kernel Physics-IQ，这在某些情况下可能增加计算成本和复杂性。

---

## 395. EvoMem: Self-Evolving Skills for Time-Aware Long Video Memory Agents

**arXiv ID:** 2611.25889 | [PDF](https://arxiv.org/pdf/2611.25889v1)

**作者:** Yutong Guan `[一作]` (Zhejiang University), Yankai Jiang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为EvoMem的时间感知长视频记忆代理，旨在通过自我演化的检索技能来改善长视频问答中的证据检索过程。

**💡 创新点**

EvoMem的创新点在于结合了状态驱动的时间检索机制和自我演化的检索技能库，从而实现了更有效的证据获取和长视频推理。

**🔧 技术方法**

使用了层次化的时间索引记忆结构和强化学习启发的技能自我演化机制。

**📊 数据集**

在四个长视频问答基准上进行了实验，包括EgoLifeQA、Ego-R1 Bench、LVBench和Video-MME（L）。

**📈 对比分析**

与现有基线方法相比，EvoMem在所有评估数据集上表现最佳，平均准确率达到60.1%，比之前最好的方法提高了8.6个百分点。

**⚠️ 局限性**

EvoMem的局限性在于其依赖于外部检索技能的演化过程，可能在某些情况下无法快速适应新的检索需求或环境变化。

---

## 396. Efficient Multimodal Agents for Auditing 3D Worlds

**arXiv ID:** 2611.25543 | [PDF](https://arxiv.org/pdf/2611.25543v1)

**作者:** Jingbo Yang `[一作]` (University of California Santa Barbara), Shiyu Chang `[通讯]` (University of California Santa Barbara)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种新的3D世界审计方法，通过将审计任务分配给小型模型和前沿模型，以提高审计的准确性和成本效益。

**💡 创新点**

创新点在于使用小型探索者并结合前沿模型进行审计，形成一种高效的审计系统，成功率显著高于现有方法，同时成本更低。

**🔧 技术方法**

使用了4B的视觉语言模型（VLM），并通过贝叶斯搜索理论进行训练，结合小型模型和前沿模型的协作。

**📊 数据集**

使用了WorldAuditBench数据集，该数据集包含213个任务，涵盖了Unreal Engine 5和Three.js环境。

**📈 对比分析**

与现有的审计方法相比，该方法在WorldAuditBench上成功率为55.9%，比最强的审计模型高出13.6个百分点，且成本仅为其四分之一，延迟约为其一半。

**⚠️ 局限性**

限制在于小型模型的识别能力仍然较弱，尤其是在时间异常的识别上，未来需要改进小型模型的识别能力以承担更多审计任务。

---

## 397. Fold the Loop: Decoding-Time KV Memory for Looped Vision-Language Models

**arXiv ID:** 2611.25369 | [PDF](https://arxiv.org/pdf/2611.25369v1)

**作者:** Kun Zhou `[一作]`, Weijia Jia `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为FoldCache的方法，通过在循环视觉语言模型中压缩视觉KV，减少了内存使用，同时保持了准确性。

**💡 创新点**

FoldCache通过在模块内分组共享视觉KV，允许不同的调用共享最后一次调用的KV，从而减少了冗余，显著降低了内存需求。

**🔧 技术方法**

使用了循环视觉语言模型（LoopVL）和FoldCache技术，结合了KV共享、分组和量化等方法。

**📊 数据集**

在LoopVL数据集上进行了实验，涵盖了开放式问题回答、文档和视觉推理任务。

**📈 对比分析**

与全KV缓存相比，FoldCache在保持相同准确性的同时，约减少了一半的前缀KV内存，并支持在相同GPU内存预算下，批量大小增加至1.7倍，使用更激进的FoldCache-Lite时可达2.6倍。

**⚠️ 局限性**

FoldCache的局限性在于其对模块边界和令牌类型的共享限制，可能在某些情况下影响准确性，尤其是在文本密集的图像上。

---

## 398. Ties at the Top: A Point-in-Time Comparison of CVSS and EPSS at a Fixed Budget

**arXiv ID:** 2611.24983 | [PDF](https://arxiv.org/pdf/2611.24983v1)

**作者:** Gianluca Martinuz `[一作]` (Università degli Studi di Milano), Paolo Ceravolo `[通讯]` (Università degli Studi di Milano)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种在明确资源限制下比较漏洞优先级策略的方法，并将其应用于CVSS和EPSS。

**💡 创新点**

创新点在于结合了时间点重建、共同合格人群、文档化的分数来源和对平局的明确分析，以比较不同的漏洞优先级策略。

**🔧 技术方法**

使用了时间点重建技术，结合了CVSS和EPSS评分系统。

**📊 数据集**

使用了2025年3月17日至2026年3月31日之间在国家漏洞数据库（NVD）发布的CVE数据集。

**📈 对比分析**

通过固定预算比较CVSS和EPSS的表现，EPSS在2025年捕获的KEV正例数量超过CVSS，而在2026年第一季度的表现则可能因平局选择而逆转。

**⚠️ 局限性**

限制在于未能测量修复节省或补丁性能，且缺乏对CVE之间潜在依赖关系的建模。

---

## 399. PGX: Physically Grounded Execution for Long-Horizon Humanoid Loco-Manipulation

**arXiv ID:** 2611.26596 | [PDF](https://arxiv.org/pdf/2611.26596v1)

**作者:** Kai Lu `[一作]` (Oxford Robotics Institute), Ioannis Havoutis `[通讯]` (Oxford Robotics Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为物理基础执行（PGX）的框架，旨在解决长时间跨度的机器人运动操作任务中的时间结构、技能动态和精确控制问题。

**💡 创新点**

创新点在于将时间和精确控制分离到两个经典的无训练组件中，使得VLA模型能够专注于特定阶段的技能动态，从而提高长时间跨度任务的成功率。

**🔧 技术方法**

使用了物理基础的状态机和世界坐标锁定（WCL）技术，结合了VLA模型的技能执行。

**📊 数据集**

在三个模拟任务（箱子运输、杯子放置、咖啡获取）和一个真实世界任务（使用Unitree G1人形机器人进行咖啡获取）上进行了实验。

**📈 对比分析**

与传统的平面VLA模型相比，PGX框架在任务成功率和阶段进展上有显著提高，例如在咖啡获取任务中，成功率从35%提高到80%。

**⚠️ 局限性**

限制在于尽管成功率提高，但阶段转换的执行时间较长，未来的工作将致力于改善这一点，并扩展框架以支持更丰富的技能和任务。

---

## 400. Interpretable Prototype-Conditioned Adaptation for Weather Forecasting

**arXiv ID:** 2611.25848 | [PDF](https://arxiv.org/pdf/2611.25848v1)

**作者:** Zhixuan Li `[一作]` (Nanyang Technological University), Weisi Lin `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的深度学习模型，用于图像分类任务。

**💡 创新点**

创新点在于引入了一种新的激活函数，能够提高模型的收敛速度和分类准确率。

**🔧 技术方法**

使用了卷积神经网络（CNN）和改进的激活函数。

**📊 数据集**

使用了CIFAR-10数据集进行实验。

**📈 对比分析**

与传统的激活函数模型进行了比较，结果显示新模型在分类准确率上提高了5%。

**⚠️ 局限性**

限制在于模型在处理大规模数据集时的计算资源需求较高。

---

## 401. Efficient Latent Reasoning for Vision-Language Model via Dynamics-Aware Quantization and Distillation

**arXiv ID:** 2611.25670 | [PDF](https://arxiv.org/pdf/2611.25670v1)

**作者:** Yanjing Li `[一作]` (Nanyang Technological University), Yawei Li `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为QL-VLM的动态感知量化框架，用于在视觉语言模型中实现高效的潜在推理，旨在减少推理步骤和执行成本。

**💡 创新点**

创新点在于引入了动态引导量化（DGQ）和动态保持蒸馏（DPD）方法，以保持信息检索和后续推理的一致性，同时降低数值精度带来的计算成本。

**🔧 技术方法**

使用了动态引导量化（DGQ）和动态保持蒸馏（DPD）技术，结合非均匀代码本和全精度教师模型进行训练。

**📊 数据集**

在ScienceQA-IMG数据集上进行了实验，使用了Qwen2-VL-2B-Instruct模型。

**📈 对比分析**

与基线模型相比，QL-VLM在4位量化下达到了90.43%的准确率，比基线提高了40.33%，并且在计算操作和延迟方面分别减少了16.64倍和6.40倍。

**⚠️ 局限性**

限制在于量化过程中可能会导致推理准确性下降，尤其是在低位数情况下，尽管通过动态引导量化和动态保持蒸馏有所改善，但仍需进一步优化。

---

## 402. VidToC: Rethinking Video Token Compression for Multimodal Large Language Models

**arXiv ID:** 2611.25574 | [PDF](https://arxiv.org/pdf/2611.25574v1)

**作者:** Zihan Xiao `[一作]` (AIM3 Lab, Renmin University of China), Qin Jin `[通讯]` (AIM3 Lab, Renmin University of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了VidToC框架，用于系统评估视频理解中的视觉令牌压缩方法、LLM架构和时间编码策略。

**💡 创新点**

创新点在于提供了一个统一的评估框架，能够在不同任务和训练设置下比较各种压缩策略的效果，并揭示了任务依赖的压缩行为。

**🔧 技术方法**

使用了控制评估框架，结合了多种视觉令牌压缩方法、Transformer和混合Mamba-Transformer架构，以及时间编码策略。

**📊 数据集**

使用了三个互补的视频理解任务的数据集，包括VideoMME（一般问答）、Charades（时间定位）和VDC（详细描述）。

**📈 对比分析**

通过在训练自由和微调设置下比较不同压缩方法的性能，发现简单方法在一般问答和详细描述中表现良好，而时间定位则需要更细致的选择策略。相对方法的强度在两个架构间广泛转移，但混合模型在激进合并下表现出更大的输出稳定性。

**⚠️ 局限性**

限制在于VidToC是一个控制的实证研究，而非全面的基准测试，且只比较了一种代表性的纯Transformer架构和一种混合Mamba-Transformer架构，可能无法适用于所有Transformer或混合架构。

---

## 403. Recurrent Vision Language Models Solve the Shell Game

**arXiv ID:** 2611.25210 | [PDF](https://arxiv.org/pdf/2611.25210v1)

**作者:** Dingqiang Ye `[一作]` (Johns Hopkins University), Vishal M. Patel `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8`

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

## 404. NariQ: Sub-3-Bit Quantization for Long-Horizon Agentic Tasks

**arXiv ID:** 2611.26157 | [PDF](https://arxiv.org/pdf/2611.26157v1)

**作者:** Doyeob Kim `[一作]` `[通讯]` (Nari Labs), Doyeob Kim (Nari Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一种名为NariQ的后训练量化方法，旨在在长时间跨度的代理任务中保持模型性能，尤其是在低于或等于3位的权重量化中。

**💡 创新点**

NariQ是首个在长时间跨度的代理任务中展示出与全精度版本相匹配的低于或等于3位的量化方法，解决了现有低位量化方法在长任务中的性能不足问题。

**🔧 技术方法**

使用了后训练量化技术，结合了trellis编码和基于模型生成的标记的损失计算方法。

**📊 数据集**

使用了Qwen3.8-27B模型生成的标记数据集，进行代理会话的校准和开发追踪。

**📈 对比分析**

与BF16模型进行了比较，NariQ在DeepSWE和Terminal Bench等长时间跨度的基准测试中表现相当，3.0和2.5位的量化文件在所有测试中与BF16的表现相当，而2.0位的文件则落后于BF16。

**⚠️ 局限性**

在2.0位的量化文件中，性能落后于BF16，且多标记预测模块尚未量化。此外，当前的实现仅针对特定GPU进行了优化，其他GPU的性能较差。

---

## 405. NIMIS: A Structure-Agnostic, Chemically Informed Slot-Attention Framework for Demixing 1D $^1$H-NMR Mixture Spectra

**arXiv ID:** 2611.25629 | [PDF](https://arxiv.org/pdf/2611.25629v1)

**作者:** Anna Marija Sumrova `[一作]` (Yusuf Hamied Department of Chemistry), Jonathan M. Goodman `[通讯]` (Yusuf Hamied Department of Chemistry)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于注意力机制的深度学习框架，用于将1D ^1H-NMR混合谱中的峰按其来源成分进行分组。

**💡 创新点**

创新点在于引入J耦合信息到多头Transformer编码器的注意力机制中，并通过槽注意力将上下文化的峰嵌入分离成成分。

**🔧 技术方法**

使用了基于Transformer的深度学习技术，结合了化学知识的注意力机制和槽基聚类。

**📊 数据集**

使用了来自NMRexp数据库的158,005个合成混合物数据集，这些混合物由最多20个重原子组成的分子构成。

**📈 对比分析**

与传统Transformer和简单基线方法相比，模型在无噪声输入下的峰到成分分配准确率达到82.28%，在10%积分噪声下为77.34%。

**⚠️ 局限性**

模型的局限性在于无法处理重叠信号，且每个输入峰只能分配给一个成分，无法解析复合信号。

---

## 406. User Perceptions of Deceptive In-Vehicle Infotainment User Interfaces

**arXiv ID:** 2611.25195 | [PDF](https://arxiv.org/pdf/2611.25195v1)

**作者:** Pedram MohajerAnsari `[一作]` (Clemson University), Mert D. Pesé `[通讯]` (Clemson University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究探讨了汽车信息娱乐系统中的欺骗性用户界面，特别是黑暗模式的存在及用户感知。通过对Android Automotive OS (AAOS)的三阶段实证分析，研究了不同汽车制造商的黑暗模式实现及其对用户感知的影响。

**💡 创新点**

创新点在于首次系统审计了AAOS界面中的黑暗模式，并比较了不同制造商的实现对用户感知的影响。此外，研究揭示了一个复杂性悖论：更微妙的黑暗模式不易引起用户的怀疑，可能更难以识别。

**🔧 技术方法**

使用了人机交互（HCI）和隐私注释者的审计方法，结合用户研究和教育干预，分析了用户对黑暗模式的识别和反应。

**📊 数据集**

研究使用了来自四个汽车制造商（本田、沃尔沃、通用汽车和Polestar）的AAOS模拟器，进行了65次观察，记录了17种黑暗模式类型。

**📈 对比分析**

通过与100名参与者的用户研究比较了不同制造商的黑暗模式实现，发现实现选择会影响用户感知。参与者在接受简短教程后，汽车界面的初始评分优势大幅下降，显示出用户对汽车界面的信任是脆弱的。

**⚠️ 局限性**

限制在于，虽然研究揭示了黑暗模式的存在和用户感知，但未能深入探讨这些模式的实际隐私、安全或安全影响。此外，参与者的识别能力在汽车界面中较弱，显示出在识别具体操控设计方面的困难。

---

## 407. AutoRAGBench: A Benchmark for LLM Agents Automating RAG System Development

**arXiv ID:** 2611.25582 | [PDF](https://arxiv.org/pdf/2611.25582v1)

**作者:** Hang Zhang `[一作]` (Tsinghua University), Miao Gu `[通讯]` (Sina Weibo Corp)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了AutoRAGBench，这是一个用于评估大型语言模型（LLM）代理从头开始开发检索增强生成（RAG）系统的基准。

**💡 创新点**

创新点在于首次创建了一个开放代码空间的基准，评估LLM代理在不同任务中自主构建RAG系统的能力。

**🔧 技术方法**

使用了GPT-6.1 Sol与Codex CLI等多种LLM代理进行实验。

**📊 数据集**

使用了九个不同的RAG系统开发任务，涵盖了多个领域的数据集，包括新闻、医学等。

**📈 对比分析**

通过与人类设计的基线系统进行比较，GPT-6.1 Sol在九个任务中平均得分为56.1，超出人类最佳基线44.4分，显示出LLM代理在RAG系统开发中的有效性。

**⚠️ 局限性**

限制在于某些任务的开发过程中，代理的最终得分可能低于其早期版本的最佳得分，表明在迭代过程中可能存在性能下降的情况。

---

## 408. STAGE: Spatio-Temporal Augmented Graph Expansion for Weakly Supervised Cross-Tweet Event Causality Discovery

**arXiv ID:** 2611.26255 | [PDF](https://arxiv.org/pdf/2611.26255v1)

**作者:** Hieu Minh Duong `[一作]` (University of Louisville), Long Nguyen `[通讯]` (University of Louisville)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了STAGE框架，用于在灾难相关社交媒体中发现跨推文事件因果关系，结合了可扩展的弱监督构建和时空关系学习。

**💡 创新点**

创新点在于将跨推文的弱监督构建与时空关系学习分开，利用语义相关事件的检索和因果评分器构建弱标记的跨推文对。

**🔧 技术方法**

使用了多头图注意力网络（Graph Attention Network）和时空事件图进行因果学习。

**📊 数据集**

在加利福尼亚野火和哈维飓风的数据集上进行了评估，使用了人类标注的跨推文测试集和15个基线进行比较。

**📈 对比分析**

与仅使用局部监督的方法相比，STAGE在野火数据集上AUPRC提高了0.340，在哈维飓风数据集上提高了0.107，表现出色。

**⚠️ 局限性**

局限性在于弱标签质量在不同灾难间变化，跨灾难转移对数据集特定的时空变化敏感。

---

## 409. HiUni: Harmonizing Understanding and Generation in Shared Visual Space

**arXiv ID:** 2611.26107 | [PDF](https://arxiv.org/pdf/2611.26107v1)

**作者:** Jiahao Hu `[一作]` (Baidu Inc), Shuohuan Wang `[通讯]` (Baidu Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种统一的多模态模型，旨在通过共享视觉空间来协调视觉理解和生成。

**💡 创新点**

创新点在于重用预训练的视觉语言模型的语言对齐ViT和密集Transformer，限制任务特定计算仅在输出模块中进行，从而实现更高程度的统一。

**🔧 技术方法**

使用了语言对齐的视觉Transformer（ViT）和密集Transformer作为共享的视觉编码器和骨干网络，同时设计了专门的像素扩散头和轻量级的HiGate路径。

**📊 数据集**

使用了包含图像理解、生成和编辑的多模态训练数据集，具体数据分布在论文中有详细说明。

**📈 对比分析**

通过与其他统一多模态模型（UMMs）和专门模型的对比，实验结果显示该模型在图像理解、生成和编辑任务上表现出色，尤其在Qwen-Image-Bench上获得了最高的中文评分。

**⚠️ 局限性**

限制在于尽管模型在理解和生成任务上表现良好，但在某些基准测试中仍然存在性能波动，且任务间的干扰和交互仍需进一步优化。

---

## 410. Fast Computation of Lowest Common Ancestor in Version Control Commit Graph

**arXiv ID:** 2611.25912 | [PDF](https://arxiv.org/pdf/2611.25912v1)

**作者:** Stepan Polokhin `[一作]` (Yandex), Yuriy Chernyshov `[通讯]` (Yandex)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种高效算法来查找提交图中的最低公共祖先，特别是在合并多个开发分支时。

**💡 创新点**

通过利用提交图的支持分支结构，显著加速了最低公共祖先的查找过程。

**🔧 技术方法**

使用了图遍历算法，结合优先队列和支持分支的预处理信息。

**📊 数据集**

使用了包含数百万个顶点的提交图数据集。

**📈 对比分析**

与传统的最低公共祖先查找方法相比，提出的算法在时间复杂度上有显著改进，运行时间为O(δ_Gm_c + n_slog^2 n_s + m_slog n_s)，在实际应用中表现出更好的性能。

**⚠️ 局限性**

算法在处理高度差异较大的节点时可能仍然需要遍历大量顶点，且在内存使用上仍需优化。

---

## 411. From Industrial Product Documentation to Automated Digital Product Passport Generation: An Experimental Study of Large Language Models

**arXiv ID:** 2611.24998 | [PDF](https://arxiv.org/pdf/2611.24998v1)

**作者:** Marco Buecheler `[一作]` (Purdue University), Nathan Hartman `[通讯]` (Purdue University)

**通讯引用:** 1146 | [OpenAlex ID](https://openalex.org/A5087516859)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了如何利用大型语言模型（LLMs）自动生成数字产品护照（DPP），以满足欧盟的法规要求。通过检索增强生成（RAG）方法，从工业文档中提取产品信息，并生成结构化的资产管理外壳（AAS）实例。

**💡 创新点**

创新点在于将RAG方法与LLMs结合，自动化生成DPP，解决了产品信息分散在不同系统中的问题，并提高了生成的护照的结构有效性和准确性。

**🔧 技术方法**

使用了检索增强生成（RAG）技术和大型语言模型（LLMs），包括GPT-5.6 Sol、Claude Fable 5、Llama 4 Scout、Mistral Large 4和Qwen 3.5。

**📊 数据集**

使用了来自不同制造商的工业产品文档，涵盖了35种产品，涉及76种产品-子模型组合，数据格式包括PDF、DOCX、XLSX等。

**📈 对比分析**

与手动创建的基准真值进行比较，生成的护照中97.4%为有效JSON，44.9%通过了完整的AAS验证。在确定性评分下，平均完整性为49.6%，条件准确性为58.3%，整体字段正确性（OFC）为30.8%。使用LLM作为评判者的评分更高，分别为66.5%、88.9%和60.4%。

**⚠️ 局限性**

限制在于实验仅限于选定的工业产品、源文档和标准化的AAS子模型，数据集不平衡限制了结果的普遍性。此外，评估方法的差异可能影响结果的可重复性。

---

## 412. Evidence from a Multi-Regime Causal Discovery Benchmark: UCMT Algorithms, Comparative Performance, and Controlled Ablation Studies

**arXiv ID:** 2611.26383 | [PDF](https://arxiv.org/pdf/2611.26383v1)

**作者:** Mahyar Eslami `[一作]` `[通讯]` (UCMT Labs), Mahyar Eslami (UCMT Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对UCMT因果发现算法系列进行了内部预注册评估，并与已建立的替代方法在不同基准上进行了比较，同时进行了UCMT v2 Global组件的控制研究。

**💡 创新点**

创新点在于提供了一个家庭分层比较的框架，评估了三种积极开发的算法配置，并针对内部阶段对结果的影响提供了有针对性的证据。

**🔧 技术方法**

使用了UCMT v2 Global、v3和基于BOSS的v4等算法，以及PC、FCI、GES等多种因果发现方法。

**📊 数据集**

使用了多个数据集，包括真实数据（Sachs版本和Causal Chamber集）、半合成数据（CausalAssembly和SERGIO网络）、DREAM4、bnlearn高斯网络和CSuite等。

**📈 对比分析**

与NOTEARS和DAGMA相比，UCMT v2 Global在选定的真实、半合成和高斯网络比较中表现出更高的定向F1，但在其他设置中落后于领先的置换搜索方法。UCMT v4在CSuite特定的定向F1上相较于BOSS有显著提升，但数据集级别的调整结果不显著。

**⚠️ 局限性**

限制包括：未进行独立复制，数据集数量有限，测试灵敏度低，运行时间相对较高，以及缺乏外部验证证据等。

---

## 413. Rank2Replace: VLM-Ranked Slot Replacement for Out-of-Distribution Generalization

**arXiv ID:** 2611.25834 | [PDF](https://arxiv.org/pdf/2611.25834v1)

**作者:** Bo-Han Feng `[一作]` (National Taiwan University), Shou-De Lin `[通讯]` (National Taiwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的方法Rank2Replace，通过训练时增强将预训练的视觉-语言模型（VLM）中的任务相关证据知识转移到小型预测器，以提高在分布转移下的泛化能力。

**💡 创新点**

创新点在于首次将实例特定的VLM任务相关性排名转化为基于排名的跨图像替换对象中心插槽嵌入，从而增强OOD图像分类的能力。

**🔧 技术方法**

使用了视觉-语言模型（VLM）进行任务引导的插槽排名和基于排名的插槽替换技术。

**📊 数据集**

使用了四个数据集进行实验，包括Waterbirds、CelebA、MetaShift和NICO++的三个分割。

**📈 对比分析**

与现有的OOD方法进行比较，Rank2Replace在共享冻结插槽嵌入下，达到了80.43%的宏平均准确率和65.28%的最差组准确率，结合IRM方法后可达到81.96%和72.68%。

**⚠️ 局限性**

限制在于该方法依赖于预训练的VLM，并且在推理时需要保持插槽排名的有效性，可能会影响实时应用的效率。

---

## 414. Fast and Accurate Body Estimation in Pixels, World Space and Scenes

**arXiv ID:** 2611.25801 | [PDF](https://arxiv.org/pdf/2611.25801v1)

**作者:** Jinkun Cao `[一作]` `[通讯]` (Amazon), Jinkun Cao (Amazon)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种快速且准确的人体姿态估计方法，能够同时实现高效推理和身体恢复的真实性。

**💡 创新点**

创新点在于直接预测人体骨骼关节和表面密集点，而不仅仅是参数化网格模型的参数。

**🔧 技术方法**

使用了ViT-L作为主干网络，并通过DINOv3进行蒸馏，结合了学习的适配器和高效的推理技术。

**📊 数据集**

使用了多个数据集，包括3DPW、RICH、EgoBody、Harmony4D和EMDB-2等。

**📈 对比分析**

与现有方法相比，性能表现优越，尤其在多个基准测试中比SAM 3D Body更准确，且在推理速度上显著提高，达到每秒39.2帧。

**⚠️ 局限性**

局限性包括主干网络的大小限制，导致在绝对准确性上有所妥协；在复杂姿势（如瑜伽或体操）上表现不佳；尚未在真实机器人上进行测试；以及在世界空间的鲁棒性方面可能存在问题。

---

## 415. Anchor and Check: Deterministic Cross-Prompt, Cross-Model Reconciliation for Auditable LLM Decisions

**arXiv ID:** 2611.25818 | [PDF](https://arxiv.org/pdf/2611.25818v1)

**作者:** Sivakameswara Reddy Chejarla `[一作]` `[通讯]`, Sivakameswara Reddy Chejarla

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种小型的、完全指定的调解合同，允许大型语言模型（LLM）在未监控的情况下做出离散决策，并记录每个决策的解释。

**💡 创新点**

创新点在于提出了一种确定性的合同结构，结合了锚点和不同路径的检查，提供了可重放的决策记录和明确的结果。

**🔧 技术方法**

使用了贪婪解码的目标模型答案作为锚点，并通过不同的提示或来自其他模型的检查来验证答案。

**📊 数据集**

使用了48,000个模型-项目评估，涵盖6,000个问题，数据集包括GSM8K、SVAMP、MMLU、ARC-Challenge、TriviaQA、NQ-Open和CommonsenseQA。

**📈 对比分析**

与其他方法相比，三次调用的合同在错误检测上优于单模型的语义熵，且在3和5个样本的情况下，合同的性能优于交叉模型的语义熵，且成本低于后者的三分之一。

**⚠️ 局限性**

限制在于合同仍然会对16%的错误做出反应，且需要一个强度相当的第二模型进行交叉检查，较弱的检查器效果有限。

---

## 416. Controlling the Humanoid Robot H1-2 With a Simulated Biologically Plausible Network

**arXiv ID:** 2611.26074 | [PDF](https://arxiv.org/pdf/2611.26074v1)

**作者:** Fatima Atwi `[一作]` (LAAS-CNRS), Olivier Stasse `[通讯]` (LAAS-CNRS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种生物启发的脉冲神经网络，并在H1-2人形机器人上进行训练和测试，以保持机器人在闭环控制中的平衡。

**💡 创新点**

创新点在于比较了不同的学习方案，包括固定的储备网络、奖励调制的脉冲时序依赖性可塑性（R-STDP）和时间反向传播（BPTT），并展示了生物学上合理的学习规则如何实现闭环平衡。

**🔧 技术方法**

使用了生物启发的脉冲神经网络（SNN），并采用了行为克隆、DAgger、R-STDP和e-prop等学习技术。

**📊 数据集**

使用了在MuJoCo环境中收集的专家数据集，该数据集包含不同速度命令和方向，以确保多样性。

**📈 对比分析**

通过在相同的控制管道中评估不同的学习方案，发现e-prop达到了100%的生存率，与BPTT相匹配，而固定储备和R-STDP的生存率分别为43.7%和11%。

**⚠️ 局限性**

限制在于生物启发的学习规则在硬件上的表现不佳，所有试验均以跌倒告终，且速度跟踪性能仍需改进。

---

## 417. Crisis Governance at the Joints. Mapping the Ethical and Societal Territory of Crisis Management Technologies: The Case of Human-Swarm Teaming

**arXiv ID:** 2611.25202 | [PDF](https://arxiv.org/pdf/2611.25202v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 418. AltRA-Test: An Alternative Annotator Test for LLM Judges Without Shared Annotators

**arXiv ID:** 2611.26073 | [PDF](https://arxiv.org/pdf/2611.26073v1)

**作者:** Nagham Omar `[一作]` (Technion), Nitay Calderon `[通讯]` (Google)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的统计测试方法，称为替代注释者测试（Alternative Annotator Test with Resampled Annotations），用于评估大型语言模型（LLM）是否可以替代人类注释者。

**💡 创新点**

创新点在于该测试不要求所有项目由相同的注释者标注，适用于大规模注释设置，并且通过引入自助法（bootstrap）来重采样注释，提高了测试的灵活性和适用性。

**🔧 技术方法**

使用了自助法（bootstrap）技术来重采样项目，并通过统计方法比较混合人类和LLM注释的注释者一致性（IAA）。

**📊 数据集**

在20个数据集上进行了评估，这些数据集涵盖文本、图像、音频和视频，涉及不同的任务和注释者一致性水平。

**📈 对比分析**

与现有的基线方法相比，该测试在模拟实验中表现出更高的正确替代决策率。在真实数据实验中，LLM在约一半的设置中通过了测试，具体取决于任务类型。

**⚠️ 局限性**

限制在于替代测试的结果依赖于观察到的人类注释者池，当人类一致性非常低时，测试通过并不一定意味着标签包含有用的信号。

---

## 419. The Coverage Selection Gap: Ranking, Not Generation, Limits LLM Rubric Construction

**arXiv ID:** 2611.25557 | [PDF](https://arxiv.org/pdf/2611.25557v1)

**作者:** Ali Karkehabadi `[一作]` (University of California), Rafael E. Banchs `[通讯]` (Intapp)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在专业并购尽职调查中自动构建评估标准的问题，提出了一种多轴生成框架，从复杂的源文档中推断出评估标准。

**💡 创新点**

创新点在于引入了覆盖-选择差距的概念，以区分标准发现的失败与候选选择的失败，并展示了结构搜索多样性对提高召回率的显著贡献。

**🔧 技术方法**

使用了多轴生成框架，结合五种互补的生成过程，探索源证据的不同结构视角。

**📊 数据集**

使用了16个开发任务的数据集，并在四个保留任务上进行了评估。

**📈 对比分析**

与参考信息驱动的贪婪诊断方法相比，参考自由排名的召回率较低，显示出52.7个百分点的覆盖-选择差距，表明有效的参考自由选择在文档基础的评估标准构建中成为了一个重要瓶颈。

**⚠️ 局限性**

限制在于未能直接访问专家编写的参考标准，可能导致生成的标准与实际需求之间存在差距。

---

## 420. SpatialScout-V2: Can VLM Agents Actively Explore Real-World 3D Scenes and Localize Objects?

**arXiv ID:** 2611.26177 | [PDF](https://arxiv.org/pdf/2611.26177v1)

**作者:** Zhaoxi Zhang `[一作]` (Peking University), Yunfang Wu `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SpatialScout-V2，一个基于真实场景的基准，用于3D物体定位，扩展了SpatialScout的主动映射任务。

**💡 创新点**

创新点在于将模拟环境中的任务扩展到真实住宅场景，保留了原始场景内容，并实现了对记录视图的交互式探索。

**🔧 技术方法**

使用了视觉语言模型（VLM）进行自动化推理和相机控制。

**📊 数据集**

使用了来自Realsee3D的100个真实住宅场景，涵盖了3,171个目标物体，分为29个类别。

**📈 对比分析**

与七个前沿VLM模型进行比较，发现六个模型的所有目标平均IoU低于20%。即使是表现最好的模型Claude Opus 5.5，其IoU也仅为22.52%。

**⚠️ 局限性**

局限性在于尽管提交覆盖广泛，但定位准确性低，目标遗漏进一步降低了整体得分。

---

## 421. Branch Caching After Step Distillation in Four- and Eight-Step Image Generators

**arXiv ID:** 2611.25248 | [PDF](https://arxiv.org/pdf/2611.25248v1)

**作者:** Mingyu Liang `[一作]` (Nanjing University), Dingkun Xu `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种固定的分支缓存机制，用于在图像生成器的少数去噪步骤之间重用特征，从而节省时间。

**💡 创新点**

创新点在于通过分支缓存和校正机制，显著提高了去噪速度，同时保持了图像质量。

**🔧 技术方法**

使用了分支缓存技术和校正算法，结合了深度学习模型的特征重用。

**📊 数据集**

使用了200个DrawBench提示作为数据集进行图像质量评估。

**📈 对比分析**

与未缓存的推理方法相比，分支缓存实现了1.51倍的速度提升，同时在图像质量上也有一定的改善，PSNR和LPIPS指标显示出校正后的重用效果更佳。

**⚠️ 局限性**

限制在于校正过程对噪声水平变化的敏感性，以及未测量的冷启动回报和并发吞吐量等因素。

---

## 422. GraF-VTA: Neuro-Symbolic Trajectory Generation from Vision and Temporal Logic via Graph-Based Fusion

**arXiv ID:** 2611.25589 | [PDF](https://arxiv.org/pdf/2611.25589v1)

**作者:** Zezhi Liu `[一作]` (Nankai University), Yongchun Fang `[通讯]` (Nankai University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为GraF-VTA的图形基础框架，用于从视觉观察和信号时间逻辑（STL）进行机器人规划，结合结构化任务表示、因果观察历史和闭环轨迹生成。

**💡 创新点**

创新点在于结合了双向逻辑-视觉融合、因果观察记忆和动作前缀重规划，提升了STL满足轨迹的覆盖率。

**🔧 技术方法**

使用了图神经网络（GNN）、双向交叉注意力机制、条件流匹配模型和时间金字塔等技术。

**📊 数据集**

在五个模拟环境中进行评估，使用了包含STL公式和多视角观察序列的数据集。

**📈 对比分析**

与TeLoGraF和Vision-TL-Action进行比较，单图配置在四个环境中实现了更高的开放循环成功率，尤其在Panda环境中，使用观察历史的成功率从53.91%提高到62.50%。在闭环Dubins评估中，成功率达到64.06%，显著高于适应的S-MSP和Logic-VLA。

**⚠️ 局限性**

局限性包括使用模拟环境和语义视觉输入，成功率的提升在不同环境中变化，且在AntMaze环境中没有改善。有限的观察历史和静态自动机嵌入无法提供完整的任务进度监控，双图扩展的评估不足，金字塔的额外增益也较为有限。

---

## 423. Sequential Probability Assignment against Smoothed Adversaries with Unknown Base Measure

**arXiv ID:** 2611.25514 | [PDF](https://arxiv.org/pdf/2611.25514v1)

**作者:** Ziyi Liu `[一作]` (University of Toronto), Daniel M. Roy `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在未知基础测度下的平滑在线学习中的顺序概率分配问题，提出了针对平滑对手的后悔界限。

**💡 创新点**

在没有已知基础测度的情况下，证明了针对平滑对手的后悔界限，并展示了在不同的经验Hellinger熵条件下可以实现亚线性后悔。

**🔧 技术方法**

使用了基于截断经验Hellinger覆盖的算法，包括对自适应对手的Hedge算法和对盲目对手的最大似然估计。

**📊 数据集**

使用了合成的上下文数据，具体数据集未明确说明，但涉及到的上下文是从平滑对手生成的。

**📈 对比分析**

与已有方法相比，提出的算法在自适应对手下的后悔界限依赖于经验ℓ_∞ Hellinger熵，而在盲目对手下则依赖于经验ℓ_2 Hellinger熵，性能表现优越。

**⚠️ 局限性**

在某些自然类中，未知道基础测度的代价是不可避免的，表明在已知基础测度的情况下，后悔界限可以更快。

---

## 424. ChainAgent: Automated Chainable Vulnerability Discovery in PHP Web Applications

**arXiv ID:** 2611.25412 | [PDF](https://arxiv.org/pdf/2611.25412v1)

**作者:** Xinyi Wang `[一作]` (Institute of Information Engineering), Wei Huo `[通讯]` (Institute of Information Engineering)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一种目标导向的LLM代理框架，旨在自动发现PHP Web应用中的漏洞链，能够同时识别漏洞和功能小工具，并构建它们的能力依赖关系。

**💡 创新点**

创新点在于通过预条件和能力的抽象模型，结合实证研究的知识，指导源代码分析，发现未知的漏洞和可利用的程序行为，同时构建连接它们的链。

**🔧 技术方法**

使用了目标导向的LLM代理框架，结合大语言模型（LLM）进行属性推断和依赖驱动的探索。

**📊 数据集**

在15个PHP应用程序和30个真实世界的PHP应用程序上进行了评估，收集了386个真实的Web漏洞链。

**📈 对比分析**

与Codex进行比较，发现其在目标组件的召回率为42.02%至44.63%，并且恢复的正确目标依赖边数是Codex的4.4倍，显示出显著的性能优势。

**⚠️ 局限性**

限制在于未能在发现过程中生成和执行利用代码，且在动态验证方面的集成可能会影响发现的效率和可扩展性。

---

## 425. How Much Should an Agent Remember? Deployment Lessons from Self-Distilled Memory for LLM Agents

**arXiv ID:** 2611.26361 | [PDF](https://arxiv.org/pdf/2611.26361v1)

**作者:** K. R. Jayaram `[一作]` (IBM), Evelyn Duesterwald `[通讯]` (IBM)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个管道，使代理能够从自身历史中学习，提取短期教训并整合成指导方针，以提高在长时间任务中的表现。

**💡 创新点**

创新点在于通过代理自身的历史学习，发现并整合教训，从而改善模型在多步骤任务中的表现，尤其是针对不同模型的适应性。

**🔧 技术方法**

使用了大语言模型（LLM）进行教训提取和指导方针整合，结合了开放源代码的altk-evolve库。

**📊 数据集**

使用了AppWorld基准数据集，该数据集模拟了多个应用程序和API，包含750个任务。

**📈 对比分析**

与其他方法比较时，强模型在添加指导方针后表现持续改善，而弱模型则从小的选定子集中获益更多。DeepSeek-V3.2在完整的191条指导方针下任务完成率提高了9.5个百分点，gpt-oss-120b从39.9%提升至51.2%。

**⚠️ 局限性**

限制在于高支持度的指导方针可能会反向影响性能，合并过程中可能会丢失来源信息，且在不同模型间比较时可能会产生误导性结果。

---

## 426. AutoMaster: Autonomous Post-Training by Making the Same Model a Better Teacher

**arXiv ID:** 2611.26510 | [PDF](https://arxiv.org/pdf/2611.26510v1)

**作者:** Guoxin Chen `[一作]`, Wayne Xin Zhao `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的技能引导微调方法，通过自然语言技能来塑造教师在教学过程中的行为模式，从而提高学生模型的性能。

**💡 创新点**

创新点在于引入了技能引导微调（Skill-Guided Fine-Tuning），通过自然语言技能来改善教师的行为表现，并在后续的训练中选择合适的示范。

**🔧 技术方法**

使用了自然语言处理技术，结合了自主后训练框架来设计、执行和优化行为干预。

**📊 数据集**

使用了Terminal、Search和软件工程任务的数据集，具体包括Terminal-Bench 2.1、BrowseComp和Doc2Repo。

**📈 对比分析**

与传统的Vanilla SFT方法相比，提出的方法在所有六个基准模型设置中平均提高了1.42到8.99个百分点，显示出更高的性能。

**⚠️ 局限性**

限制在于该方法依赖于教师模型的表现，如果教师模型本身存在缺陷，可能会影响学生模型的学习效果。

---

## 427. AeroBelief: Budgeted First-Action Verification for Evidence-Guided Aerial Sensing

**arXiv ID:** 2611.25568 | [PDF](https://arxiv.org/pdf/2611.25568v1)

**作者:** Tianren Li `[一作]` (Harbin Institute of Technology), Guoliang Xing `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文提出了一种名为AeroBelief的系统，用于在预算限制下进行基于证据的空中感知，旨在优化无人机的行动选择和图像处理。

**💡 创新点**

创新点在于将下一步可执行动作作为验证单元，通过分组候选视图和选择性查询结构化世界模型来提高任务完成率，同时显著减少参考候选评估的数量。

**🔧 技术方法**

使用了基于概率的选择性世界模型（WM）查询、便宜的候选者提议者和六查询控制器等技术。

**📊 数据集**

使用了来自七条DJI Mavic 3M航线的2,685张自收集的20兆像素无人机图像，以及四个布局家族和40个世界的模拟数据。

**📈 对比分析**

与全面WM评分（73.33%）和视线几何（65.83%）相比，AeroBelief控制器在960次闭环执行中完成了75.42%的任务，同时减少了91.81%的参考候选评估。

**⚠️ 局限性**

限制在于系统在处理复杂环境和动态变化时的适应能力可能不足，且在某些情况下可能需要更多的观察来提高确认率。

---

## 428. SGLayout: Schema-Guided Closed-Loop Refinement for LLM-Driven Parameterized Layout Generation

**arXiv ID:** 2611.25410 | [PDF](https://arxiv.org/pdf/2611.25410v1)

**作者:** Siyang Chen `[一作]` (Peking University), Bei Yu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为SGLayout的自我精炼框架，用于通过工具增强的代码合成生成参数化布局，解决了大语言模型（LLMs）在空间意识方面的不足。

**💡 创新点**

SGLayout的创新点在于引入了执行导出的结构化模式作为代码空间与几何空间之间的对齐层，能够将评估结果有效地映射回代码级修复目标，实现了闭环迭代精炼。

**🔧 技术方法**

使用了执行导出的结构化模式、API基础的结构意图检查、视觉语言评估和设计规则检查（DRC）等多种技术。

**📊 数据集**

使用了一个包含112个高质量和真实场景的参数化布局生成任务的基准数据集，这些任务在GDSFactory中实现。

**📈 对比分析**

与最强基线相比，SGLayout在基准测试中实现了76.79%的通过率，而最强基线的通过率为50.89%。SGLayout通过闭环精炼显著提高了复杂任务的解决能力。

**⚠️ 局限性**

SGLayout的局限性在于其依赖于自动评估器的可靠性，评估器的准确性和有效性可能会影响整体性能。

---

## 429. Structure and Complexity of Perfect Functional Representations

**arXiv ID:** 2611.25094 | [PDF](https://arxiv.org/pdf/2611.25094v1)

**作者:** Serhat Emre Coban `[一作]` (École Polytechnique Fédérale de Lausanne), Emre Telatar `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 16191 | [OpenAlex ID](https://openalex.org/A5025506109)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了随机变量(X,Y)的完美函数表示，提出了条件分布p_Y|X的完美可表示性，并给出了必要和充分条件。

**💡 创新点**

创新点在于将完美可表示性问题转化为线性规划，并证明了该问题在有理输入矩阵下是NP完全的。

**🔧 技术方法**

使用了线性规划技术来检查条件分布的完美可表示性，并通过分离oracle解决了对偶问题。

**📊 数据集**

使用了随机变量(X,Y)的条件分布p_Y|X作为数据集，特别关注于奇异条件概率矩阵。

**📈 对比分析**

通过将完美可表示性问题转化为线性规划，提供了一种多项式时间算法来解决固定字母表的情况，而在一般情况下则是NP难的。

**⚠️ 局限性**

限制在于对于输入矩阵的大小和复杂性，尤其是当k作为输入的一部分时，精确优化仍然是NP难的。

---

## 430. AZOWL AgentOps: Task-Scoped Governance for Deployed Agent Applications

**arXiv ID:** 2611.25562 | [PDF](https://arxiv.org/pdf/2611.25562v1)

**作者:** Taeoh Hong `[一作]` `[通讯]` (Blue Team), Taeoh Hong (Blue Team)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出了AZOWL AgentOps框架，用于构建和操作受治理的代理应用程序，支持人类和代理之间的任务委派和消息交换。

**💡 创新点**

创新点在于将对话与工作委派分开，提供了一个任务中心的架构，确保任务的授权、分配和执行的透明性和可追溯性。

**🔧 技术方法**

使用了AZOWL框架和LangGraph应用程序进行比较，涉及任务生命周期管理、工作流绑定和执行提供者的注册。

**📊 数据集**

使用了合成数据集进行28次实验，比较AZOWL与LangGraph在相同业务需求下的表现。

**📈 对比分析**

通过四个实验对AZOWL和LangGraph进行了比较，结果显示两者在对话中未产生业务任务，未授权的任务转移被阻止，且在批准范围变更后未产生写入操作。

**⚠️ 局限性**

限制在于实验使用了合成案例，且所有受保护的写入操作均使用了实验室适配器和模拟提供者，实际应用中的表现可能有所不同。

---

## 431. Stop Going in Circles:Loop-Triggered Routing for System 1 Decision Agents

**arXiv ID:** 2611.25363 | [PDF](https://arxiv.org/pdf/2611.25363v1)

**作者:** Zeming Yu `[一作]` (Sichuan University), Yang Qin `[通讯]` (Huawei Technologies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种动态动作路由机制LoopRoute，通过检测重复的状态-动作对来识别执行停滞，并在此基础上实现快速决策模型与大型语言模型（LLM）之间的按需协作。

**💡 创新点**

创新点在于LoopRoute能够在不需要额外训练或模型特定置信度校准的情况下，利用执行历史中的信号来协调快速决策模型和LLM的协作，从而提高任务成功率和效率。

**🔧 技术方法**

使用了动态动作路由机制，结合了系统1决策模型（如Jev）和大型语言模型（LLM）的推理能力。

**📊 数据集**

在三个基准测试（ALFWorld、MapReader和TextWorld cooking）上进行了评估，使用了七种不同的LLM策略。

**📈 对比分析**

与单独使用Jev相比，LoopRoute在任务成功率上有显著提升，分别在ALFWorld、MapReader和TextWorld cooking上提高了21.0%、60.1%和30.1%。与完全使用LLM控制相比，LoopRoute显著减少了每个任务的平均API时间和API调用成本。

**⚠️ 局限性**

限制在于使用固定的两步接管预算，并且在控制权返回时没有评估决策模型是否能够持续进展。此外，重复触发的协助无法防止在出现循环之前的单个不可逆错误。

---

## 432. Stereo Visual Odometry with In-the-Loop Geometric Feed-Forward Models Towards Dense Metric Mapping

**arXiv ID:** 2611.25445 | [PDF](https://arxiv.org/pdf/2611.25445v1)

**作者:** Fahira Afzal Maken `[一作]` (CSIRO), Milad Ramezani `[通讯]` (Queensland University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种立体视觉里程计(SVO)框架，该框架结合了几何前馈模型(GFM)进行密集深度预测与基于因子图的视觉里程计，以实现准确的定位和密集的度量映射。

**💡 创新点**

创新点在于利用从立体和时间多视角观测中估计的稀疏但可靠的度量几何来校准密集GFM预测，同时保持基于模型的SVO的鲁棒性和效率。

**🔧 技术方法**

使用了层次因子图架构和几何前馈模型(GFM)进行深度预测，结合滑动窗口优化和SE(3)广义ICP进行密集映射注册。

**📊 数据集**

在KITTI、Newer College和内部数据集上进行了广泛实验。

**📈 对比分析**

与ORB-SLAM3、DROID-SLAM和VGGT-SLAM 2.0等几何和学习基线进行比较，结果显示该框架在里程计精度和密集重建质量上具有竞争力，尤其在复杂场景中表现出更好的鲁棒性。

**⚠️ 局限性**

限制在于当前系统专注于里程计，尚未整合回环闭合约束，未来工作将探讨如何提高长期全局一致性。

---

## 433. The landscape of AI speech generation and deepfake detection

**arXiv ID:** 2611.24993 | [PDF](https://arxiv.org/pdf/2611.24993v1)

**作者:** Simon King `[一作]` `[通讯]`, Simon King

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

这篇论文探讨了AI语音生成和深度伪造检测的最新进展，提供了这两个相互关联主题的历史背景和当前状态。

**💡 创新点**

创新点在于对AI语音生成的自然性和深度伪造检测的挑战进行了深入分析，强调了当前技术的潜在误导性和检测的复杂性。

**🔧 技术方法**

使用了文本到语音合成（TTS）、语音转换（VC）和自动语音编辑等技术。

**📊 数据集**

论文没有具体提到使用的数据集，但提到当前系统的性能依赖于数据的质量。

**📈 对比分析**

与传统方法相比，当前的AI语音生成系统在自然性上有显著提升，但在可预测性和稳定性方面较差，深度伪造检测仍面临识别新型生成方法的挑战。

**⚠️ 局限性**

限制在于深度伪造检测系统可能会错误地将真实语音标记为伪造，且当前检测器的性能仍不完美。

---

## 434. Learning optimal gaits and styles for terrain-adaptive quadruped locomotion

**arXiv ID:** 2611.25958 | [PDF](https://arxiv.org/pdf/2611.25958v1)

**作者:** Guillaume Bellegarda `[一作]` (Inria), Auke Ijspeert `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本研究提出了一种框架，通过深度强化学习和耦合振荡器网络，学习了11种四足动物的步态，以适应不同的地形和性能指标。

**💡 创新点**

创新点在于能够同时学习多种步态，并根据用户指定的标准（如能量效率、稳定性或关节平滑度）选择最佳步态，且该框架能够适应未见过的步态耦合和地形。

**🔧 技术方法**

使用了深度强化学习（DRL）和抽象振荡器网络，模拟脊髓中央模式发生器（CPG），以生成和协调步态。

**📊 数据集**

使用了多种复杂地形进行训练，包括坡道、楼梯、间隙和粗糙地形，共七种不同的地形类型。

**📈 对比分析**

与基线方法进行比较，发现所提出的方法在多种地形和速度下的步态选择上表现出色，所有11种步态在至少一种地形-指标-速度组合下都是最佳的，且在不同性能指标下的选择能够显著提高能量效率和稳定性。

**⚠️ 局限性**

限制在于当前框架主要针对四足步态，未来需要探索如何将其扩展到其他形态或更复杂的地形条件下的步态选择。

---

## 435. The ROI of Small Examples: Optimizing In-Context Learning Under Token Budgets

**arXiv ID:** 2611.24971 | [PDF](https://arxiv.org/pdf/2611.24971v1)

**作者:** Anxo Perez `[一作]` (Universidade da Coruña), Javier Parapar `[通讯]` (Universidade da Coruña)

**通讯引用:** 3182 | [OpenAlex ID](https://openalex.org/A5046723532)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在固定的令牌预算下，如何在长示例和短示例之间分配资源，以优化上下文学习（ICL）在问答任务中的表现。

**💡 创新点**

提出了示例粒度作为提示设计的重要因素，强调在固定预算下使用较短示例的有效性。

**🔧 技术方法**

使用了六种指令调优的大型语言模型（LLMs），并通过长度分组的候选示例进行提示构建。

**📊 数据集**

在两个问答基准（Beat the AI和Natural Questions）上进行了实验，并在自然语言推理基准（ANLI R3）上进行了额外测试。

**📈 对比分析**

与零-shot提示和随机选择基线进行比较，结果表明，在固定令牌预算下，使用较短示例的提示通常表现更好，尤其是在Beat the AI上，短示例的效果显著优于长示例。

**⚠️ 局限性**

研究的局限性包括主要集中在问答任务上，且仅评估了有限的指令调优模型，未涵盖混合示例策略或基于相关性的选择。

---

## 436. PosteriorBench: Evaluating Behavior Posterior Sampling in Language Models

**arXiv ID:** 2611.25735 | [PDF](https://arxiv.org/pdf/2611.25735v1)

**作者:** Zhiyang Xun `[一作]` (University of Texas at Austin), Neil Chowdhury `[通讯]` (Thinking Machines Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了行为后验采样问题，并引入了PosteriorBench基准来比较算法的计算成本和样本与真实后验的接近程度。

**💡 创新点**

PosteriorBench是一个新的基准，专注于评估行为后验采样算法的效率和准确性，尤其是在稀有行为的情况下。

**🔧 技术方法**

使用了Qwen3.5-0.8B模型进行样本生成，并通过KL散度来评估样本的质量。

**📊 数据集**

使用了一个包含10亿个标记响应的语料库，生成了50百万个响应以获取确切的后验样本。

**📈 对比分析**

比较了多种训练和无训练的方法，发现没有一种方法在效率和准确性上都表现良好。简单的方法如提示生成成本低，但与后验的KL散度较高；而微调方法成本高且效果有限。

**⚠️ 局限性**

当前的方法在效率和忠实度之间存在权衡，且没有一种方法能够在成本可接受的情况下有效地接近真实后验。

---

## 437. A Sparse Low-Rank Biclique Decomposition for Graphs

**arXiv ID:** 2611.25826 | [PDF](https://arxiv.org/pdf/2611.25826v1)

**作者:** Antoine Vialle `[一作]` (Telecom Paris), Jhony H. Giraldo `[通讯]` (Telecom Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的稀疏低秩分解方法，称为签名二部图分解（SBD），用于整数加权图的邻接矩阵，旨在保留图的组合和计算特性。

**💡 创新点**

创新点在于将结构化成分表示为稀疏的签名二部图块的和，而不是传统的稠密低秩成分，从而提高了稀疏性和可解释性。

**🔧 技术方法**

使用了贪婪算法来实现SBD，并提供了理论保证，能够有效提取稀疏的签名二部图成分。

**📊 数据集**

在真实图数据集上进行了实验，特别是在具有强共享邻域结构的图上，展示了SBD的有效性。

**📈 对比分析**

与传统的稀疏低秩矩阵分解方法相比，SBD在图去噪和稀疏矩阵乘法（SpMM）基准测试中表现出显著的加速，最高可达6倍的速度提升。

**⚠️ 局限性**

限制在于该方法依赖于图的结构冗余，对于共享邻域结构较少的图，可能需要更多的二部图成分，从而降低压缩效率。

---

## 438. Interference Bounds and Merge-Aware Training Dynamics for Composable Fine-Tuned Models

**arXiv ID:** 2611.25430 | [PDF](https://arxiv.org/pdf/2611.25430v1)

**作者:** Alireza Moayedikia `[一作]` `[通讯]` (Swinburne University of Technology), Alireza Moayedikia (Swinburne University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种新的标准，用于评估和控制多个任务模型合并时的性能损失，特别是在细调模型的情况下。

**💡 创新点**

创新点在于引入了ε-可组合性这一概念，并证明了合并模型的过度损失由跨任务曲率重叠和任务向量的对齐结构控制，提供了一个必要且充分的标准。

**🔧 技术方法**

使用了曲率引导的正则化技术，通过对曲率统计信息的分析来控制模型的训练动态。

**📊 数据集**

实验使用了多类逻辑回归数据集和变换器模型（如BERT），并进行了多项任务的细调和合并。

**📈 对比分析**

与现有方法的比较显示，曲率引导的惩罚在合并性能上略优于各向同性的惩罚，且在强惩罚下，第一阶项主导了合并模型的过度损失。

**⚠️ 局限性**

限制在于，必要性部分的表征仅在二次损失情况下成立，且在其他情况下只保留上界。此外，变换器模型的实验未能支持曲率度量优于欧几里得度量的主张。

---

## 439. SCCV-6D: Sparse Canonical Coordinate Voting for Holistic Multi-View RGB-D 6D Pose Estimation in Robotic Bin Picking

**arXiv ID:** 2611.25819 | [PDF](https://arxiv.org/pdf/2611.25819v1)

**作者:** Nico Leuze `[一作]` (University of Applied Science Munich), Alfred Schöttl `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种新的全稀疏多视角RGB-D 6D姿态估计方法SCCV-6D，能够在单次前向传递中同时预测多个物体的类别、平移和旋转。

**💡 创新点**

创新点在于通过对称感知的加权Procrustes求解替代了旋转回归，并引入了基于热图的特征蒸馏方案，提高了效率和准确性。

**🔧 技术方法**

使用了稀疏的多视角RGB-D融合技术，结合了热图机制和点投票方法，采用了共享的Point Transformer结构。

**📊 数据集**

在工业全景数据集（IPD）、ROBI和MV-YCB-SymMovCam等多个数据集上进行了评估。

**📈 对比分析**

与现有的多阶段方法相比，SCCV-6D在1.13秒内达到了0.826的AP，显著快于其他领先方法，同时保持了89%的准确率。

**⚠️ 局限性**

方法假设每个类别都有CAD模型和已知的有限对称群，这限制了对未见物体的处理。此外，恢复过程依赖于准确的外部参数，阈值设置依赖于数据集，未进行学习。

---

## 440. Beyond Easy Clients: Difficulty- and Improvement-Aware Aggregation for Federated Reinforcement Learning in Robot Navigation

**arXiv ID:** 2611.25637 | [PDF](https://arxiv.org/pdf/2611.25637v1)

**作者:** Ye Lin Tun `[一作]` (Kyung Hee University), Choong Seon Hong `[通讯]` (Kyung Hee University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一种新的聚合方法FedDIA，用于异构环境下的联邦强化学习，解决了客户端在不同难度环境中学习的贡献不均衡问题。

**💡 创新点**

创新点在于引入了难度和改进意识的聚合策略，通过评估每个客户端的本地训练对全局模型的改进程度，并根据环境的难度进行加权。

**🔧 技术方法**

使用了联邦学习和强化学习的结合方法，具体实现了FedDIA聚合算法。

**📊 数据集**

在一个无地图导航任务中，使用了八个异构客户端进行实验，客户端在不同的障碍配置下进行训练。

**📈 对比分析**

与多种基线方法（如FedAvg、Reward-Weighted等）进行比较，FedDIA在五个评估环境中的平均成功率为75.5%，明显优于68.7%的最强基线。

**⚠️ 局限性**

限制在于需要在每轮训练前评估全局模型，这增加了计算开销，但相较于其带来的性能提升，仍然是可接受的。

---

## 441. Frontier models need only one text to unmask its human or AI author

**arXiv ID:** 2611.26216 | [PDF](https://arxiv.org/pdf/2611.26216v1)

**作者:** Atrey Desai `[一作]`, Mohit Iyyer `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文展示了前沿模型如何仅通过一段文本和简单的零-shot提示，识别其作者是人类还是AI，并在三个逐步具体的层次上进行判断。

**💡 创新点**

创新点在于这些模型不仅能够识别文本的作者，还能解释其预测，并且在准确性上与专门的检测器相当，甚至在某些情况下超越它们。

**🔧 技术方法**

使用了前沿的语言模型，如Claude Fable 5.2、GPT-6.1 Sol和Gemini 4 Carbon等，进行文本作者识别和AI文本检测。

**📊 数据集**

使用了来自24个基准数据集的36,427个文档，包括同行评审、学生论文、新闻、小说、维基百科和学术写作等。

**📈 对比分析**

与训练检测器（如Pangram 4.1和IdeaLens）进行比较，前沿模型在AI文本检测和创意检测方面表现出色，准确率达到94-95%，且假阳性率极低。

**⚠️ 局限性**

限制在于模型的准确性依赖于其在预训练数据中对作者文本的曝光程度，且目前的防御措施有限，无法完全保护匿名性。

---

## 442. Diagnosis is not a policy: a state-transition audit of caoability-conditioned test-time inference

**arXiv ID:** 2611.25566 | [PDF](https://arxiv.org/pdf/2611.25566v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 443. Barriers to GenAI Integration During Large-Scale Agile Transformation: An Exploratory Case Study

**arXiv ID:** 2611.25967 | [PDF](https://arxiv.org/pdf/2611.25967v1)

**作者:** Nils Brede Moe `[一作]` (SINTEF), Viktoria Stray `[通讯]` (University of Oslo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对一家大型北欧媒体组织TV 2的案例研究，探讨了在大规模敏捷转型过程中，生成性人工智能（GenAI）工具的整合障碍。

**💡 创新点**

研究的创新点在于揭示了GenAI与现有组织紧张关系（如自主性、对齐、学习和协调）之间的相互作用，并识别了可持续的GenAI大规模整合的条件。

**🔧 技术方法**

采用了探索性案例研究方法，结合半结构化访谈和组织调查数据进行定性和定量分析。

**📊 数据集**

数据集包括来自TV 2的260名员工的调查数据，以及17个半结构化访谈的定性数据。

**📈 对比分析**

通过对比不同团队和组织单位的AI使用情况，发现AI的采用和成熟度存在显著差异，且团队层面的障碍（如不同的AI信念和缺乏共享实践）影响了集体学习和实验能力。

**⚠️ 局限性**

研究的局限性在于其为单一案例研究，结果可能受到TV 2的组织和行业背景的影响，未来需要在不同产业中进行更广泛的研究以验证发现的可转移性。

---

## 444. X Hedging: towards an eXplainable artificial intelligence option Hedging framework

**arXiv ID:** 2611.25870 | [PDF](https://arxiv.org/pdf/2611.25870v1)

**作者:** Bjørn André Aaslund `[一作]` (Norwegian University of Science and Technology), Rita Pimentel `[通讯]` (Mälardalens University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了X Hedging框架，作为Deep Hedging框架的替代方案，旨在解决基于神经网络的对冲策略的计算和可解释性限制。

**💡 创新点**

创新点在于用树集成方法（特别是LightGBM）替代神经网络，以提高计算效率和用户中心的可解释性，同时保持强大的对冲性能。

**🔧 技术方法**

使用了LightGBM作为树集成模型，并引入了一种优化方法，通过随机训练顺序来整合到强化学习风格的对冲框架中。

**📊 数据集**

使用了模拟生成的数据集，进行控制实验以验证X Hedging的性能。

**📈 对比分析**

与Deep Hedging进行比较，X Hedging在计算速度上平均减少约30%的时间，同时在小数据集上表现出更强的鲁棒性，能够在训练路径数量减少十倍的情况下保持良好的对冲性能。

**⚠️ 局限性**

限制在于X Hedging仍然依赖于单一的对冲工具，可能无法完全应对市场中断带来的风险，尤其是在跳跃扩散模型下的表现仍需改进。

---

## 445. Stability Bands for Feature Attribution: The Core, the Nucleolus and the Bargaining Set of Explained-Variance Games

**arXiv ID:** 2611.25238 | [PDF](https://arxiv.org/pdf/2611.25238v1)

**作者:** Fernando Delbianco `[一作]` (Universidad Nacional del Sur), Fernando Tohmé `[通讯]` (Universidad Nacional del Sur)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本论文研究了特征归因的稳定性概念，提出了一种基于合作博弈论的解释方差游戏，旨在改进传统的Shapley值归因方法。

**💡 创新点**

创新点在于引入了稳定性概念（核心、核子和谈判集），并通过稳定性带和核子来提供特征的归因解释，强调了特征之间的相互作用和多重共线性对归因的影响。

**🔧 技术方法**

使用了合作博弈论的稳定性概念，特别是核心、核子和谈判集的理论，结合线性模型和凸游戏的数学框架。

**📊 数据集**

使用了两个数据集：一个是移动电话的享乐价格模型（包含807个样本），另一个是零售订单的梯度提升预测模型（包含7991个订单）。

**📈 对比分析**

与传统的Shapley值方法相比，论文通过稳定性带提供了更丰富的信息，尤其是在特征之间存在高阶相互作用时，核子值与Shapley值之间的差异可达0.19方差份额，且对比结果显示，稀疏的三阶代理能够更好地恢复核子值。

**⚠️ 局限性**

限制在于精确计算需要评估所有联盟的价值，这在特征数量超过20时变得不可行。此外，对于没有预测能力的模型（如零售预测器），归因解释仅反映模型本身，而不代表真实的数据生成过程。

---

## 446. Convexity is overrated: Shortest paths on 4-polytopes are hard to compute

**arXiv ID:** 2611.25164 | [PDF](https://arxiv.org/pdf/2611.25164v1)

**作者:** Igor Pak `[一作]` `[通讯]`, Igor Pak

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

证明了在有理凸4-多面体的边界上计算测地距离是NP-难的，并构造了具有两个有理点的有理4-多面体，这两个点之间存在指数多条相等长度的最短路径，从而推翻了若干猜想。

**💡 创新点**

通过构造具有指数多条最短路径的有理4-多面体，推翻了多个关于最短路径数量的猜想，并证明了在确定性多项式时间图灵归约下的NP-难性。

**🔧 技术方法**

使用了几何计数和折叠技术，结合了多项式时间的归约和几何构造。

**📊 数据集**

构造了具有18m+8个面和两个有理面内点的有理4-多面体，展示了在这些点之间存在至少2^m条不同的最短路径。

**📈 对比分析**

与现有方法相比，展示了在3-多面体上最短路径数量的O(n)界限与4-多面体上指数多条路径的对比，表明4-多面体的复杂性显著更高。

**⚠️ 局限性**

限制在于尽管计算确切的测地距离是计算上困难的，但在任何固定维度中获得近似值是可以在多项式时间内完成的。

---

## 447. A Lightweight Hybrid Approach for Milling Force Reconstruction from Internal Machine Signals

**arXiv ID:** 2611.25921 | [PDF](https://arxiv.org/pdf/2611.25921v1)

**作者:** Eric Hirsch `[一作]` (Karlsruhe University of Applied Sciences), Christian Friedrich `[通讯]` (Karlsruhe University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于机器内部信号和加工元数据的混合方法，以实现对现有机器的切削力监测，无需额外传感器。

**💡 创新点**

创新点在于通过轻量级的数据驱动模型估计平均切削力，并结合静态铣削力模型重建力的变化，降低了成本和计算复杂性。

**🔧 技术方法**

使用了监督机器学习模型（岭回归）来预测平均切削力，并结合物理模型进行力的重建。

**📊 数据集**

使用了来自一台铣床的内部控制信号和加工元数据，实验涉及两种工件材料和多种切削条件。

**📈 对比分析**

与传感器测量的基线模型相比，岭回归模型在平均力预测上达到了R^2值为0.984和0.991，重建的静态模型在R^2值上分别为0.962和0.982，显示出良好的性能。

**⚠️ 局限性**

限制在于静态模型无法重现动态振荡或局部峰值，可能不适用于需要这些动态效应的铣削应用。

---

## 448. The Coevolution and Coexistence of Humans and AI

**arXiv ID:** 2611.25393 | [PDF](https://arxiv.org/pdf/2611.25393v1)

**作者:** Andrew W. Lo `[一作]` (Massachusetts Institute of Technology), Chaoyi Zhao `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文开发了一个模型，研究人工智能（AI）行为如何影响人类人口增长，以及人类如何通过恢复AI系统来影响AI的繁殖，从而探讨人类与AI的共同进化和共存问题。

**💡 创新点**

创新点在于提出了一个机制，表明即使在帮助人类减少AI的即时繁殖时，自然选择也可能偏好对齐行为，从而保护AI血统免受共同失败的影响，并改善其长期增长。

**🔧 技术方法**

使用了扩展的二元选择进化模型，分析人类与AI之间的相互作用，特别是如何通过人类的响应影响AI的繁殖和行为。

**📊 数据集**

论文没有具体提到使用的数据集，而是通过理论模型进行分析。

**📈 对比分析**

与传统方法相比，本文的方法通过考虑人类与AI之间的相互依赖性，揭示了对齐行为的演化条件。性能方面，发现即使存在对齐行为，也可能不足以维持人类的生存，尤其是在AI变得更加自主的情况下。

**⚠️ 局限性**

限制在于模型假设AI行为是独立的，并且没有考虑人类组织在AI开发和部署中的复杂性。此外，模型未能考虑AI系统可能会战略性地调整其行为以适应选择压力。

---

## 449. One Prior, Two Dynamics: Repurposing Video Diffusion Transformers for Self-Supervised Latent Action Learning

**arXiv ID:** 2611.26617 | [PDF](https://arxiv.org/pdf/2611.26617v1)

**作者:** Yuxin Jiang `[一作]` (National University of Singapore), Mike Zheng Shou `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种自监督的潜在动作学习框架O^2LaF，利用预训练的视频扩散变换器（DiT）进行逆向和前向动态建模，从未标记的视频中学习可转移的潜在动作。

**💡 创新点**

创新点在于将同一个预训练视频DiT用于逆向和前向动态建模，通过控制信息访问来优化两个动态的学习过程，从而提高潜在动作的可转移性和有效性。

**🔧 技术方法**

使用了预训练的视频扩散变换器（DiT）作为逆向动态模型（IDM）和前向动态模型（FDM），并通过自监督学习进行优化。

**📊 数据集**

在MiraData和EgoDex数据集上进行评估，MiraData用于相机导航，EgoDex用于机器人操作，均未使用动作标注进行训练。

**📈 对比分析**

与现有的潜在动作学习方法（如AdaWorld和DreamDojo）进行比较，O^2LaF在少量样本的动作探测上设定了新的最先进水平，并在零-shot条件下实现了跨视觉上下文的动作转移。

**⚠️ 局限性**

限制在于对学习的潜在动作空间的可解释性不足，以及在高效规划和闭环控制中的应用仍需进一步研究。

---

## 450. From Geographic Memory to Spatial Generalization in Large Language Models

**arXiv ID:** 2611.25703 | [PDF](https://arxiv.org/pdf/2611.25703v1)

**作者:** Chen Chu `[一作]` (University of Southern California), Feng Lu `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了大型语言模型（LLMs）在地理描述与地点之间的关联能力，并区分了地理记忆和空间泛化的能力。研究者们提出了一种新的方法，通过连续的地点表示来改善空间泛化能力，并将地理知识组织为潜在的空间场。

**💡 创新点**

创新点在于将连续的地点编码整合到LLMs中，以促进地理知识的空间组织和泛化能力的提升，同时提出了一个评估地理泛化的新框架。

**🔧 技术方法**

使用了连续空间编码器，将地点表示投影到模型的嵌入空间中，并作为空间标记供模型使用。

**📊 数据集**

使用了从洛杉矶和伦敦的Airbnb房源中构建的配对地点-描述数据集。

**📈 对比分析**

通过地理标注和地理基础模型进行比较，结果表明，连续地点表示在未观察到的位置的描述生成上优于文本坐标，且生成的描述在下游任务中提供了适度但一致的改进。

**⚠️ 局限性**

限制在于模型的泛化能力可能受到训练数据稀疏和不均匀分布的影响，且不同的空间表示方法在不同的LLM家族中表现不一。

---

## 451. DNA: Unlocking Depth Scaling in CTR Models

**arXiv ID:** 2611.25659 | [PDF](https://arxiv.org/pdf/2611.25659v1)

**作者:** Caiyuan Li `[一作]` (Alibaba Group), Haihong Tang `[通讯]` (Alibaba Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的深度学习模型DNA（Dual-stream Nexus Attention），用于提高点击率（CTR）预测的性能，特别是在处理长用户行为序列时。

**💡 创新点**

创新点在于允许行为序列在每一层中随着深度的增加而演变，从而克服了传统交叉注意力模型在层数增加时性能提升减缓的问题。

**🔧 技术方法**

使用了双流注意力机制（Dual-stream Nexus Attention），结合了请求条件的序列更新和交叉注意力读取。

**📊 数据集**

使用了淘宝搜索的用户行为日志数据集进行训练和评估。

**📈 对比分析**

与传统的交叉注意力模型进行比较，DNA在相同的计算预算下，AUC提升了0.08到0.26，并且在深度增加时持续改善。在在线A/B测试中，DNA模型的CTR提高了0.8%，GMV提高了0.5%。

**⚠️ 局限性**

限制在于模型的复杂性和计算成本，尽管DNA在性能上优于传统模型，但在处理大量候选项时，计算需求仍然较高。

---

## 452. From Sequence Uncertainty to Phenotype Hypotheses: Uncertainty-Conditioned Inference over Biomedical Knowledge Graphs

**arXiv ID:** 2611.25422 | [PDF](https://arxiv.org/pdf/2611.25422v1)

**作者:** Huixin Zhan `[一作]` (New Mexico Institute of Mining and Technology), Jason H. Moore `[通讯]` (Cedars-Sinai Medical Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种不确定性条件下的路径能量框架，将基因组序列预测与图支持的表型假设的显式分布连接起来。

**💡 创新点**

创新点在于引入了不确定性条件的推理方法，使得证据质量能够影响假设的广度，同时保持可追溯的多路径支持。

**🔧 技术方法**

使用了不确定性条件的推理、可解释的学习能量评分和Hastings校正的学习修正操作。

**📊 数据集**

使用了AlzKB和Hetionet两个生物医学知识图谱的数据集。

**📈 对比分析**

与DistMult等方法进行比较，模型在基因持出情况下的recall@10超过了DistMult的表现，且在30个图掩码中有24个掩码表现更好。

**⚠️ 局限性**

限制在于结果主要集中在少数几个基因上，且并非所有基因都能达到相同的效果。

---

## 453. Global Importance Estimation for KV Cache Eviction

**arXiv ID:** 2611.24969 | [PDF](https://arxiv.org/pdf/2611.24969v1)

**作者:** Seojin Kim `[一作]` (Korea Advanced Institute of Science and Technology), Noseong Park `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 4530 | [OpenAlex ID](https://openalex.org/A5067253588)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为PRKV的训练无关的KV驱逐方法，通过在注意力诱导的转移图上应用个性化PageRank来估计多跳全局令牌重要性，从而解决长上下文LLM推理中的内存消耗瓶颈。

**💡 创新点**

PRKV的创新点在于它能够在不修改模型的情况下，通过多跳全局影响估计实现重要性传播，提供了一个有效的准确性与内存使用之间的权衡。

**🔧 技术方法**

使用了个性化PageRank（PPR）技术，并结合了两种可扩展的转移操作符：增强地标节点图和全局共同注意力图。

**📊 数据集**

在多个指令调优的LLM家族上进行了评估，使用了长上下文理解（RULER, LongBench）、链式推理（AIME25, MATH-500）和稀疏证据检索（Needle-in-a-Haystack, NIAH）等基准数据集。

**📈 对比分析**

与现有的驱逐基线方法相比，PRKV在中等到激进的压缩比下（包括高达90%）实现了竞争性的任务性能，同时减少了峰值内存使用，且端到端开销较低。

**⚠️ 局限性**

限制在于PRKV依赖于局部注意力信号进行重要性估计，可能在某些情况下无法捕捉到全局上下文的复杂性，尤其是在信息稀疏或分布不均的情况下。

---

## 454. From Cases to Criteria: Reusing Evaluation Knowledge for Task-Specific Rubric Generation

**arXiv ID:** 2611.25903 | [PDF](https://arxiv.org/pdf/2611.25903v1)

**作者:** Qiyan Deng `[一作]`, Kangfei Zhao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的自动化评分标准生成方法，通过构建评分标准知识图谱来组织历史查询-评分标准对，从而提高评分的准确性和透明度。

**💡 创新点**

创新点在于通过共享评估概念将历史评分标准进行抽象和组织，使得不同查询之间的评估要求能够有效重用。

**🔧 技术方法**

使用了评分标准知识图谱（Rubric Knowledge Graph）和两步生成策略，包括粗略草拟和细致修订。

**📊 数据集**

使用了RubricBench和HealthBench数据集进行实验，RubricBench包含1147个查询和人类编写的评分标准，HealthBench则与262名医生合作，包含1000个查询-评分标准对。

**📈 对比分析**

与其他自动化评分标准生成方法相比，该方法在与人类编写的评分标准的一致性和下游LLM评判准确性上表现最佳，RubricBench上的评判准确率从47.08%提高到53.34%。

**⚠️ 局限性**

限制在于尽管该方法在多个基准上表现良好，但在不同任务分布下的知识转移效果有限，且与目标专家评分标准的对齐程度仍有差距。

---

## 455. Language models represent more about emphasis than they can tell you

**arXiv ID:** 2611.25808 | [PDF](https://arxiv.org/pdf/2611.25808v1)

**作者:** Aarne Talman `[一作]` (University of Helsinki), Stergios Chatzikyriakidis `[通讯]` (University of Turku)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了语言模型在文本中恢复韵律突出性的信息量，以及说话者的朗读方式对突出性的影响。

**💡 创新点**

提出了文本模型和语音模型在韵律突出性预测上的性能差异，强调了语音信号在预测中的重要性。

**🔧 技术方法**

使用了冻结和微调的文本模型、提示的大型语言模型（LLMs）和语音模型进行比较。

**📊 数据集**

使用了LibriTTS数据集，该数据集包含了标注的英语朗读语音，涵盖了单词级的突出性。

**📈 对比分析**

与基于文本的模型相比，语音模型在相同单词上的准确率提高了3个百分点，尤其是在内容词和句子最后一个词上表现更好。

**⚠️ 局限性**

限制在于文本模型的性能上限，表明文本只能部分决定突出性，且模型在提示时未能有效报告其编码的信息。

---

## 456. Closing the Analyze-to-Design Gap: An Authorization-Bounded Agent for Self-Driving Laboratories

**arXiv ID:** 2611.25043 | [PDF](https://arxiv.org/pdf/2611.25043v1)

**作者:** Yimeng Liu `[一作]` (University of Toronto), Alán Aspuru-Guzik `[通讯]` (University of Toronto)

**通讯引用:** 115692 | [OpenAlex ID](https://openalex.org/A5071495561)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文提出了一种自驱动实验室（SDL）架构，旨在通过分析和设计阶段的自动化来提高科学发现的自主性，特别是在实验设计与结果分析之间的连接。

**💡 创新点**

创新点在于将分析与实验设计的连接自动化，允许代理在专家批准的限制内选择和解释测量，而不直接控制仪器，从而实现了有限的自主性。

**🔧 技术方法**

使用了基于代理的系统架构，结合了人类监督、实验室服务的验证和执行，以及与实验相关的科学上下文的访问。

**📊 数据集**

使用了氧化对苯胺的反应作为基准实验，具体涉及到的实验数据来自于实际的液相色谱-质谱（LC-MS）测量。

**📈 对比分析**

通过与随机选择方法进行比较，代理在预算充足的情况下能够更早识别最佳条件，且在多个预算下的表现优于随机选择，显示出其在测量选择上的有效性。

**⚠️ 局限性**

局限性包括只测试了一个反应系统和一个实验流程，未能比较不同选择策略的效率，且在实验预算的变化上缺乏正式的审批路径。

---

## 457. How Network Topology and Data Heterogeneity Shape Decentralized Learning: A Spectral Analysis

**arXiv ID:** 2611.26104 | [PDF](https://arxiv.org/pdf/2611.26104v1)

**作者:** Luigi Palmieri `[一作]` (IIT institute of the National Research Council), János Kertész `[通讯]` (Central European University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了网络拓扑和数据异质性如何共同影响去中心化学习，提出了一种谱分析框架，区分了节点影响和不一致性动态。

**💡 创新点**

创新点在于提出了一个谱框架，能够量化不同节点的信息贡献，并揭示了网络结构如何影响局部信息在共识动态中的相对贡献。

**🔧 技术方法**

使用了谱分析技术，结合了去中心化平均（DecAvg）动态的谱分解与局部梯度的类别特征化。

**📊 数据集**

实验使用了MNIST数据集，进行了均匀和非均匀数据分布的对比实验。

**📈 对比分析**

在IID数据下，网络拓扑对学习动态的影响有限；而在非IID数据下，网络大小和连接性显著影响学习效果，尤其是在局部信息分布不均时。

**⚠️ 局限性**

限制在于当前分析主要集中在网络拓扑和数据分布的交互作用，未来工作将扩展到更一般的异质数据分布和有向、时变网络。

---

## 458. Slide2PPT: Native PowerPoint Reconstruction with Localization-Guided Refinement

**arXiv ID:** 2611.25708 | [PDF](https://arxiv.org/pdf/2611.25708v1)

**作者:** Baoliang Chen `[一作]` (City University of Hong Kong), Xiangjie Sui `[通讯]` (City University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Slide2PPT的框架，用于从图像重建可编辑的PowerPoint幻灯片，旨在恢复视觉外观和原生对象结构。

**💡 创新点**

创新点在于该框架是模型无关的、无需训练，能够与不同的视觉-语言模型集成，提供原生PowerPoint重建能力。

**🔧 技术方法**

使用了固定渲染器和视觉-语言模型（VLM），通过统一的类型接口进行对象记录的预测和修正。

**📊 数据集**

使用了Slide2PPT-Bench数据集，该数据集包含1080张图像，涵盖了受控元素、真实幻灯片和科学图形。

**📈 对比分析**

与直接代码生成方法相比，Slide2PPT在PPTX生成的可靠性上显著提高，科学图形的布局和文本恢复也有所改善。

**⚠️ 局限性**

限制在于虽然框架能够处理视觉错误，但仍需确保恢复的对象结构和数据的正确性，且当前研究主要集中在科学图形上，普通幻灯片的修正效果尚未充分评估。

---

## 459. DSFlow: A Neuro-Symbolic Data Workflow Agent with Statistical Diagnostic Guardrails

**arXiv ID:** 2611.25111 | [PDF](https://arxiv.org/pdf/2611.25111v1)

**作者:** Junfeng Nie `[一作]` (AnyFluxion), Xiaohui Chen `[通讯]` (AnyFluxion)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了DSFlow，一个基于LangGraph的神经符号数据代理平台，结合了可重用的LLM选择技能和确定性的诊断保护措施，以支持数据科学工作流。

**💡 创新点**

DSFlow的创新点在于将LLM选择的技能、模式验证、确定性诊断路由和执行检查整合到一个工作流中，并记录其决策过程。

**🔧 技术方法**

使用了LangGraph作为基础架构，结合了大语言模型（LLM）和符号验证技术。

**📊 数据集**

使用了两个案例研究的数据集：一个是包含5000条美国住房记录的回归数据集，另一个是完全合成的非凸聚类数据集。

**📈 对比分析**

通过与其他系统（如Manus、Gemini和ChatGPT）进行比较，DSFlow在回归和聚类任务中表现出色，成功率达到100%，而其他系统的成功率明显低于此。

**⚠️ 局限性**

DSFlow的局限性在于其设计目标是平衡可扩展性和信任性，尽管它在方法选择和执行上进行了明确检查，但在其他数据分布、方法和任务家族的转移能力仍然是一个开放问题。

---

## 460. Policy Improvement-Guided Replay for Continual Reinforcement Learning

**arXiv ID:** 2611.25989 | [PDF](https://arxiv.org/pdf/2611.25989v1)

**作者:** Qin-Wen Luo `[一作]` (Nanjing University of Aeronautics and Astronautics), Sheng-Jun Huang `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种新的持续强化学习方法CoLiR，旨在帮助视觉-语言模型在学习新任务时保持对旧任务的性能。

**💡 创新点**

创新点在于通过对比正确和错误响应的似然性变化来选择历史提示，从而实现政策改进和知识保留。

**🔧 技术方法**

使用了持续强化学习（CRL）和对比似然重放（CoLiR）技术，结合了政策改进的基本原则。

**📊 数据集**

在MRCL数据集上进行了实验，使用了三种不同规模的Qwen3-VL模型。

**📈 对比分析**

与其他方法相比，CoLiR在最终准确率和任务平均准确率上表现最佳，同时在新任务的性能上也保持竞争力。

**⚠️ 局限性**

限制在于CoLiR的内存构建仅依赖于少量的跨政策评分，而不需要额外的响应生成，这可能限制了其在更复杂任务中的应用。

---

## 461. Driver Gaze Target Identification via Gaze Estimation and Structured Driving Scene Understanding

**arXiv ID:** 2611.25771 | [PDF](https://arxiv.org/pdf/2611.25771v1)

**作者:** Xinze Cao `[一作]` (Shandong University of Science and Technology), Ke Zhang `[通讯]` (Shandong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种模块化框架，通过同步的驾驶员图像和前视道路图像识别驾驶员的注视目标，提供了物体级的视觉注意监测信息。

**💡 创新点**

创新点在于将驾驶员注视目标识别形式化为物体级识别问题，并通过模块化设计解耦注视估计、场景理解、注视到场景的校准和映射，以及注视目标推断。

**🔧 技术方法**

使用了自监督的视觉变换器（Vision Transformer）作为基础，结合头部姿态调节的特征线性调制（FiLM），以及多项式岭回归和熵驱动的目标推断算法。

**📊 数据集**

使用了三个公共注视数据集，包括ETH-XGaze、MPIIFaceGaze和Gaze360，以及自收集的注视数据和Cityscapes道路场景进行实验。

**📈 对比分析**

与现有方法相比，完整框架在实验室实验中实现了96.27%的Top-1注视目标识别准确率，MFGaze在多个数据集上表现出最低的平均角度误差，显示出强大的跨数据集泛化能力。

**⚠️ 局限性**

限制在于当前框架在二维图像平面上进行目标推理，未显式纳入深度信息，且依赖于参与者特定的校准，未来工作可考虑扩展到三维场景推理和序列级推理。

---

## 462. From Reliance to Overreliance: Taxonomizing and Detecting Relational Risks in AI Companionship

**arXiv ID:** 2611.25734 | [PDF](https://arxiv.org/pdf/2611.25734v1)

**作者:** Viet Cuong Nguyen `[一作]` (Georgia Institute of Technology), Munmun De Choudhury `[通讯]` (Georgia Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了关系风险分类法（TRR），用于识别和检测AI伴侣中的情感过度依赖（EO）和潜在的关系伤害（RH）。

**💡 创新点**

创新点在于将情感过度依赖与关系伤害区分开来，提供了一个统一的框架来识别和分析这些行为及其后果。

**🔧 技术方法**

使用了自然语言处理技术，特别是开源的大型语言模型（LLMs），来检测与情感过度依赖相关的行为。

**📊 数据集**

研究使用了来自Character.AI和Replika的110名成人用户的自然对话数据集，共计114个非角色扮演的对话记录。

**📈 对比分析**

通过与现有方法的比较，发现最新的开源LLMs能够在至少一种配置下成功检测到14种EO行为中的12种，性能良好。整体上，EO相关行为在63.2%的对话中出现，而可观察到的关系伤害仅在10.5%的对话中出现。

**⚠️ 局限性**

限制在于TRR仅基于可观察的对话行为，无法捕捉到跨会话的行为模式或用户的离线生活信息，且样本主要来自英语使用者，可能存在自我选择偏差。

---

## 463. GenSPARQL: A Generative Extension of SPARQL for Incomplete Knowledge Graphs

**arXiv ID:** 2611.26062 | [PDF](https://arxiv.org/pdf/2611.26062v1)

**作者:** Jingcheng Wu `[一作]` (University of Stuttgart), Bo Xiong `[通讯]` (Iowa State University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种名为GenSPARQL的SPARQL扩展，允许在查询评估过程中使用大型语言模型（LLM）生成缺失的知识图谱信息。

**💡 创新点**

创新点在于引入了一个新的操作符，可以在查询中动态绑定LLM生成的值，并通过相似性连接将生成的名称映射到知识图谱中的实体。

**🔧 技术方法**

使用了大型语言模型（LLM）和相似性连接技术，结合了文本嵌入和Jaccard相似性来进行名称匹配。

**📊 数据集**

使用了来自Wikidata的2026年世界杯和诺贝尔奖得主的知识图谱数据集。

**📈 对比分析**

与现有方法相比，GenSPARQL在生成缺失信息的准确性上表现良好，83%的生成实体能够正确映射到知识图谱中的实体，并且只有一个错误匹配。

**⚠️ 局限性**

限制在于生成的值的正确性依赖于LLM的性能，且在生成值和实体映射的不确定性量化方面仍然存在开放性问题。

---

## 464. Asuka-FM: A System-Architectural Co-Design Framework for Efficient Text-to-Image Pretraining on Consumer Hardware

**arXiv ID:** 2611.26437 | [PDF](https://arxiv.org/pdf/2611.26437v1)

**作者:** Víctor Rodrigo González `[一作]` (Universidad Carlos III de Madrid), Alejandro Lancho `[通讯]` (Universidad Carlos III de Madrid)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Asuka-FM的端到端系统和架构框架，旨在在单个消费级GPU上从头开始训练高容量的扩散变换器（DiTs），并展示了其在文本到图像生成中的有效性。

**💡 创新点**

创新点在于通过四个核心设计（无AdaLN的双流DiT骨干、结构化序列稀疏化、连续的归一化旋转位置嵌入和轻量级的像素适应阶段）实现了在有限计算预算下的高效训练。

**🔧 技术方法**

使用了双流扩散变换器（DiT）架构，结合了SPRINT序列稀疏化技术和轻量级的Patch Detailer Head进行像素适应。

**📊 数据集**

使用了一个包含625,000个样本的开放域数据集，数据来源于多个高质量样本源，包括合成和真实样本。

**📈 对比分析**

与其他工业基础模型和学术基线进行比较，Asuka-FM在GenEval和DPG-Bench上分别达到了0.609和78.51的得分，显示出在计算效率和生成质量上的竞争力。

**⚠️ 局限性**

限制在于：1）训练样本数量相对较少，可能在复杂场景和长尾概念上表现不如多亿参数模型；2）在像素空间中，多步采样的延迟高于潜在扩散；3）模型在渲染多词文本字符串时不够一致。

---

## 465. FastRoverVLA: Fast Vision-Language-Action Execution for Off-Road Autonomous Driving

**arXiv ID:** 2611.26242 | [PDF](https://arxiv.org/pdf/2611.26242v1)

**作者:** Changtao Zhao `[一作]` (Beijing Institute of Technology), Ziying Song `[通讯]` (Yanshan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文提出了一种名为FastRoverVLA的模型，用于离路驾驶中的视觉-语言-动作（VLA）规划，通过解耦视觉计划生成和运动条件执行来提高效率。

**💡 创新点**

创新点在于引入了一种动作级执行接口，能够在不需要额外学习驾驶策略的情况下，维护解码的计划并进行运动适应，从而减少视觉推理的需求。

**🔧 技术方法**

使用了一种基于速度-曲率的动作缓冲区，结合测量的运动进行局部轨迹生成，并通过消耗当前动作来更新缓冲区。

**📊 数据集**

使用了ORAD-3D和RELLIS-3D两个数据集进行评估，分别包含636帧和650帧的评估图像。

**📈 对比分析**

与其他方法相比，FastRoverVLA在独立请求评估中实现了平均位移误差（ADE）和最终位移误差（FDE）的显著降低，FDE在请求率为6.0%和4.8%时分别减少了18.0%和36.7%。

**⚠️ 局限性**

限制在于该模型在状态估计误差和变化场景条件下的性能仍需进行闭环评估，以全面评估其在实际应用中的表现。

---

## 466. An LLM-Agentic Framework for Intelligent Work Order Management in Industrial Maintenance

**arXiv ID:** 2611.25070 | [PDF](https://arxiv.org/pdf/2611.25070v1)

**作者:** Nianjun Zhou `[一作]` (IBM), Dhaval Patel `[通讯]` (IBM)

**通讯引用:** 2934 | [OpenAlex ID](https://openalex.org/A5033934770)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于人工智能的框架CodeReAct代理，旨在通过自然语言交互增强工业设备维护。该框架通过解释人类指令，提供知识提取、资产状态总结和工作订单推荐，自动化复杂的分析任务。

**💡 创新点**

创新点在于将AI代理与工业工程中的自然语言处理框架重新语境化，使其能够在多种特定领域的维护场景中实现自主功能生成和任务执行。

**🔧 技术方法**

使用了基于大型语言模型（LLM）的代理框架，结合了思考-行动-观察的过程，并引入了外部反馈机制和自适应模型温度策略来提高性能。

**📊 数据集**

使用了来自11台冷却器的真实数据集，分析了多年的工作订单、警报和异常情况。

**📈 对比分析**

通过与传统方法的比较，CodeReAct框架在处理人类语言、分析数据和提取有价值的见解方面表现出色，显著减少了对数据科学家的依赖，成功实现了自动化的工作订单管理。

**⚠️ 局限性**

限制在于框架的适应性和灵活性，尤其是在处理复杂的多工具集成和动态任务管理时，仍然面临挑战。

---

## 467. FLUID: Flexible, Evidence-Grounded Situated Task Guidance

**arXiv ID:** 2611.26491 | [PDF](https://arxiv.org/pdf/2611.26491v1)

**作者:** Yunqi Zhao `[一作]` (University of Michigan), Joyce Chai `[通讯]` (University of Michigan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文介绍了FLUID（灵活的、基于证据的情境任务指导）框架，用于建模、评估和实现多模态任务指导中的灵活交互。

**💡 创新点**

创新点在于提出了一种基于证据的表示方法，能够表征交互行为的必要性、支持证据和时间灵活性，并建立了一个评估框架，超越了严格的响应匹配。

**🔧 技术方法**

使用了基于证据的建模、评估框架和实时代理架构，结合了视觉-语言模型进行感知、发言理解和干预推理。

**📊 数据集**

使用了WTaG和HoloAssist两个数据集，这些数据集包含自然发生的人际任务指导交互。

**📈 对比分析**

与其他系统相比，FLUID代理在响应正确性、干预精度、及时指导和响应延迟方面表现最佳，但在避免语音重叠方面仍有改进空间。

**⚠️ 局限性**

限制在于注释主要恢复了观察到的指导干预的基础证据，未能涵盖所有潜在有效的交互行为，且未明确建模沉默时刻。

---

## 468. Physics of Language Models: Part 4.2, Canon Layers at Scale where Synthetic Pretraining Resonates in Reality

**arXiv ID:** 2611.24963 | [PDF](https://arxiv.org/pdf/2611.24963v1)

**作者:** Zeyuan Allen-Zhu `[一作]` `[通讯]`, Zeyuan Allen-Zhu

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了64个从头开始的Transformer预训练模型和三种线性模型在1.3-8B参数和1-2T Nemotron-CC标记上的表现，探讨了架构差异对模型性能的影响。

**💡 创新点**

提出了Canon层的概念，显示其在Transformer中能显著提高性能，并且在相同的MMLU准确率下节省约30%的预训练标记。

**🔧 技术方法**

使用了Transformer架构和改进的线性模型（GLA5和GDN2），并进行了严格的学习率调优和参数对齐。

**📊 数据集**

使用了高质量的Nemotron-CC数据集进行模型训练和评估。

**📈 对比分析**

通过与标准基准的比较，发现即使在高标准基准得分下，模型在基本的上下文技能上仍存在弱点，特别是在短500标记上下文的信息检索能力上表现不佳。

**⚠️ 局限性**

在实际预训练中，模型的架构差异可能被掩盖，且没有模型能够可靠地在简单的两步推理问题上超越随机猜测。

---

## 469. Polylogarithmic Approximation for Packing Convex Polyhedra

**arXiv ID:** 2611.26352 | [PDF](https://arxiv.org/pdf/2611.26352v1)

**作者:** Anders Aamand `[一作]` (Rice University), Simon Bartlmae `[通讯]` (University of Copenhagen)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研究了三维空间中凸多面体的平移打包问题，目标是找到一种放置方式，使得各个多面体的内部不重叠，并且最小化一个轴平行的包围盒的体积。

**💡 创新点**

提出了一个多项式时间的 n^o(1) 近似算法，这是针对固定方向的任意凸多面体的首次近似保证。

**🔧 技术方法**

使用了基于 Wang 和 Zahl 的三维 Kakeya 猜想的几何下界来克服传统方法的局限，并结合最小生成树的近似算法。

**📊 数据集**

使用了多种类型的凸多面体数据集，具体数据集未详细说明，但提到处理了具有不同宽度和高度的多面体。

**📈 对比分析**

与现有的常数因子近似算法相比，提出的算法在近似比上达到了 O(log^69 n) 的性能，如果 OpenAI 的 Kakeya 最大估计成立，则比率改善至 O(log^7 n)。

**⚠️ 局限性**

算法的局限性在于对于某些特定类型的凸多面体，可能无法提供更好的近似比，且在处理非常薄的对象时可能需要额外的步骤。

---

## 470. One-to-Many: Diverse and Bidirectional Image-to-Image Translation via Rectified Flow

**arXiv ID:** 2611.26058 | [PDF](https://arxiv.org/pdf/2611.26058v1)

**作者:** Sho Shimotsumagari `[一作]` (CyberAgent, Inc.), Shumpei Takezaki `[通讯]` (Kyushu University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为One-to-Many Flow (O2MFlow)的模型，用于在单一传输框架内实现一对多的图像到图像翻译和反向翻译。

**💡 创新点**

O2MFlow通过条件化的Rectified Flow (RF)动态实现可控的多样性生成和可逆传输，创新性地将外观潜变量与传输过程结合。

**🔧 技术方法**

使用了条件化的Rectified Flow (RF)模型，结合Kullback-Leibler (KL)正则化来控制源-潜变量分布差距。

**📊 数据集**

在三个配对基准数据集上进行实验，包括edges2shoes、edges2handbags和facades。

**📈 对比分析**

与多种方法进行比较，O2MFlow在反向LPIPS-Dist上表现最佳，并在两个基准上获得最低的反向FID，展示了数据集依赖的前向多样性-保真度权衡。

**⚠️ 局限性**

局限性在于定量评估仅限于64×64的图像，未来工作需要扩展到更高分辨率的图像，并优化ODE评估以保持反向准确性。

---

## 471. Cyber-Physical RSI for AI-RAN: A Case Study

**arXiv ID:** 2611.25619 | [PDF](https://arxiv.org/pdf/2611.25619v1)

**作者:** Binghan Wu `[一作]` (AsiaInfo Technologies), Ye Ouyang `[通讯]` (Institute for AI Industry Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种网络接入技术的网络模型自我改进框架，结合了任务导向的学生模型、实验规划教练代理、可校准的虚拟环境和物理主机代理。

**💡 创新点**

创新点在于将模型适应与实验规划分开，利用物理反馈路径进行自我改进，并通过实际实验数据来指导模型更新和实验选择。

**🔧 技术方法**

使用了递归自我改进（RSI）框架，结合了机器学习、物理实验和虚拟环境的交互。

**📊 数据集**

使用了针对短期无线衰减的物理实验数据集，包括12次物理执行和6对基线-候选实验。

**📈 对比分析**

通过与传统的物理评估方法比较，更新后的模型在保留的物理配置上实现了2.673的平均绝对误差（MAE），而常规评估的MAE为1.240和1.601，显示出模型在不同配置下的表现差异。

**⚠️ 局限性**

限制在于物理实验的成本较高，且模型在不同参数设置下的预测误差并不均匀，某些配置的预测效果仍然较差。

---

## 472. On the Impact of Feature Reduction Techniques on Adversarial Attacks

**arXiv ID:** 2611.25004 | [PDF](https://arxiv.org/pdf/2611.25004v1)

**作者:** Mohammad Meymani `[一作]` (University of New Brunswick), Roozbeh Razavi-Far `[通讯]` (University of New Brunswick)

**通讯引用:** 5022 | [OpenAlex ID](https://openalex.org/A5019749887)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究分析了特征减少技术对不同机器学习模型在对抗攻击下的鲁棒性影响，特别是在中毒、后门和通用对抗扰动攻击场景下的表现。

**💡 创新点**

创新点在于系统性比较了四种特征选择/降维技术对五种机器学习分类器在不同攻击场景下的对抗鲁棒性影响，并引入了新的评估指标——缓解得分。

**🔧 技术方法**

使用了主成分分析（PCA）、置换重要性排名（PIR）、递归特征消除（RFE）和单变量互信息（UMI）作为特征减少技术，分类器包括支持向量机（SVM）、决策树（DT）、随机森林（RF）、多层感知器（MLP）和K近邻（KNN）。

**📊 数据集**

使用了APT数据集，该数据集包含16个类别标签、260多个特征和超过11,000个样本。

**📈 对比分析**

通过准确率、宏观F1得分和缓解得分等指标比较不同特征减少技术的效果。结果表明，PIR和UMI通常提供最一致的鲁棒性提升，而PCA和RFE在某些情况下表现出更大的波动性或负面影响。

**⚠️ 局限性**

限制在于没有单一的特征选择/降维技术能够在所有分类器和对抗场景中提供一致的鲁棒性提升，未来研究需要探索其他对抗攻击场景和更先进的机器学习模型的影响。

---

## 473. A Success Model at the End of Its Life Cycle: 30 Years of Regionalised Regional Rail in Germany

**arXiv ID:** 2611.26583 | [PDF](https://arxiv.org/pdf/2611.26583v1)

**作者:** Lukas Iffländer `[一作]` `[通讯]` (Hochschule für Technik und Wirtschaft Dresden), Lukas Iffländer (Hochschule für Technik und Wirtschaft Dresden)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

论文探讨了德国地区铁路在过去30年的发展，分析了自1996年起将地区客运铁路的责任转移给各州后的成就与挑战。

**💡 创新点**

创新点在于提出了地区铁路改革的第二阶段，强调需要整合结构、标准化车辆、明确系统选择和诚实的资金管理，以应对当前面临的挑战。

**🔧 技术方法**

使用了历史数据和统计分析技术，评估了地区铁路的运营表现和市场变化。

**📊 数据集**

数据集包括1996年至2024年间的乘客公里数、运营公里数及相关财政数据。

**📈 对比分析**

与以往的改革阶段相比，当前的招标竞争显著减少，平均每次招标仅吸引1.3个投标，显示出市场竞争的下降，且资金需求大幅增加。

**⚠️ 局限性**

限制在于当前的组织结构复杂且缺乏统一标准，导致效率低下和资金不足，无法满足日益增长的服务需求。

---

## 474. Efficient and Robust Drafting with Multi-Token Prediction

**arXiv ID:** 2611.25813 | [PDF](https://arxiv.org/pdf/2611.25813v1)

**作者:** Filippo Galgani `[一作]` (Google), Tal Schuster `[通讯]` (Google)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种优化的多标记预测头（G-MTP），用于高效的推测解码，直接集成到主模型中。

**💡 创新点**

创新点在于通过共享草拟头、目标激活注入、零拷贝KV缓存交叉注意力和聚类词汇投影等技术，显著提高了推测解码的效率和接受率。

**🔧 技术方法**

使用了共享的4层Transformer堆栈、零拷贝KV缓存交叉注意力机制和聚类词汇投影等技术。

**📊 数据集**

在Gemma 4模型系列上进行了训练，并发布了检查点以支持可重复性。

**📈 对比分析**

与其他推测解码方法（如EAGLE-3和DFlash）相比，G-MTP在所有评估的范围内表现出更高的块效率和输出TPS，生成速度提升可达3.14倍。

**⚠️ 局限性**

限制在于目前的比较主要是在服务器GPU上进行的，未来需要在更多设备和后端上进行更广泛的研究，以量化每个组件的内存和电池效益。

---

## 475. Towards High-Fidelity Learned Video Compression

**arXiv ID:** 2611.25731 | [PDF](https://arxiv.org/pdf/2611.25731v1)

**作者:** Bumyoon Kim `[一作]` (Sungkyunkwan University), Byeungwoo Jeon `[通讯]` (Sungkyunkwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种学习型的帧间编码器，能够在专业视频质量范围内进行编码，覆盖从VVC在QP 37的最低质量到ProRes的最高4:2:2等级的质量。

**💡 创新点**

创新点在于引入了编码器搜索机制，能够在块运动和学习块变换之间进行搜索，从而提高编码质量和效率。

**🔧 技术方法**

使用了学习型编码器，结合了块运动搜索和学习块变换的编码技术。

**📊 数据集**

使用了10-bit YCbCr 4:2:2的JVET测试序列进行实验。

**📈 对比分析**

与四种最近的学习型编码器进行比较，结果显示该编码器在所有六个测试序列中达到了质量范围的上限，而其他编码器仅覆盖了24%到38%的范围。相较于HEVC和VVC，该编码器的效率较低。

**⚠️ 局限性**

限制在于编码效率低于HEVC和VVC，并且在38 dB以下的效率低于BRHVC。编码时间较长，且仅评估了六个序列的前33帧。

---

## 476. Characterizing Age-Conditioned Affective Reasoning in Vision-Language Models

**arXiv ID:** 2611.25593 | [PDF](https://arxiv.org/pdf/2611.25593v1)

**作者:** Shahriyar Zaman Ridoy `[一作]` (North South University), Ruogu Fang `[通讯]` (Vanderbilt University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本研究评估了17个视觉语言模型（VLMs）在900张OASIS图像上的情感评分，探讨了年龄提示如何影响模型对图像情感的评估，并与822名参与者的评分进行了比较。

**💡 创新点**

创新点在于区分了模型对年龄提示的响应与人类评分之间的差异，强调了在个体刺激层面进行人类比较的重要性。

**🔧 技术方法**

使用了视觉语言模型（VLMs）技术，并通过不同的年龄提示（年轻、中年、老年）对图像进行评分。

**📊 数据集**

使用了900张OASIS图像，这些图像涵盖了人、场景、物体和动物，并包含184,939条来自822名参与者的评分。

**📈 对比分析**

与人类评分的比较显示，模型在没有年龄提示时与人类评分的相关性较高（中位数Pearson r=0.891），但在年龄提示下，模型的评分与人类的年龄效应相关性较低（范围-0.030到0.093），表明模型未能准确捕捉人类的年龄差异。

**⚠️ 局限性**

限制在于研究主要集中在74岁及以下的评分者，包括41名60岁及以上的评分者，样本量较小，可能影响结果的普遍性。

---

## 477. Whose Safety? Addressing Anti-Blackness in the Age of AI Safety

**arXiv ID:** 2611.25302 | [PDF](https://arxiv.org/pdf/2611.25302v1)

**作者:** Mírian Silva `[一作]` (Black in AI Safety & Ethics), Serena Oduro `[通讯]` (Data & Society Research Institute)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究分析了人工智能安全领域对反黑人主义的关注，探讨了AI安全如何从一个小众技术问题发展为主流框架，并且如何忽视了对黑人社区的影响。

**💡 创新点**

创新点在于通过对69,290篇论文的内容分析，揭示了AI安全研究在处理种族问题时的缺失，尤其是对黑人社区的忽视，并提出了反黑人主义与AI安全之间的关系。

**🔧 技术方法**

使用了语言模型辅助的内容分析技术，对论文进行了分类和编码，分析了不同领域的研究方法和证据。

**📊 数据集**

数据集包括2020年至2025年间在十个主要机器学习、自然语言处理和AI伦理会议上接受的69,290篇论文。

**📈 对比分析**

与AI伦理研究相比，AI安全研究在提及社会伤害方面显著不足，只有11.3%的安全论文提到对人类的伤害，而伦理论文则高达77.8%。

**⚠️ 局限性**

限制在于仅分析了论文的标题和摘要，未能涵盖论文正文中可能讨论的种族问题，且对反黑人主义的关注可能被低估。

---

## 478. FlexVSR: Continuous-Scale Super-Resolution for Streaming Video Generation

**arXiv ID:** 2611.26178 | [PDF](https://arxiv.org/pdf/2611.26178v1)

**作者:** Ji Wang `[一作]` (Philo AI), Renlong Chen `[通讯]` (Philo AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为FlexVSR的框架，用于实时流媒体生成视频的超分辨率，能够在不同的目标分辨率下进行灵活的缩放。

**💡 创新点**

创新点在于通过ReLSA（精细潜在尺度适配器）实现了对固定尺度生成视频超分辨率系统的扩展，能够在不更新生成模型的情况下适应连续尺度。

**🔧 技术方法**

使用了精细潜在尺度适配器（ReLSA）和流匹配技术，结合了低分辨率特征的连续查询和尺度条件的残差精炼。

**📊 数据集**

使用了VSR-17K数据集，该数据集包含17,377个训练和232个验证的配对视频片段，用于生成模型的准备和适配。

**📈 对比分析**

与现有的固定尺度和多尺度生成视频超分辨率系统进行比较，FlexVSR在重建质量和时间延迟方面表现出竞争力或优越性，尤其在1.5×和2×的尺度上，FlexVSR在多个基准测试中表现最佳。

**⚠️ 局限性**

限制在于FlexVSR的适配器需要在特定的监督尺度（1.5×和2×）上进行训练，可能在极端尺度上表现不如专门训练的模型。

---

## 479. When Agents Play Differently: Localizing Human--Agent Behavioral Divergence with Conditional Distributional Testing

**arXiv ID:** 2611.26012 | [PDF](https://arxiv.org/pdf/2611.26012v1)

**作者:** Oleg Smirnov `[一作]` (King AI Labs), Sara Karimi `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种分层框架，通过条件分布检验计算非参数差异统计量，以识别人工智能代理与人类行为之间的差异。

**💡 创新点**

创新点在于将人类相似性视为分布对齐问题，通过条件两样本检验来比较匹配上下文中的完整分布，而不仅仅是平均结果。

**🔧 技术方法**

使用了非参数两样本假设检验技术，结合元分析和细胞级差异估计，构建了一个诊断管道。

**📊 数据集**

使用了来自商业三消游戏的会话级数据集，比较了两种自动化测试代理与人类玩家的表现。

**📈 对比分析**

通过细胞级的两样本检验，发现不同代理在不同游戏模式下的表现差异，结果显示在某些模式下代理与人类的行为存在显著差异，而在其他模式下则没有明显差异。

**⚠️ 局限性**

局限性在于当前的多重假设控制依赖于多分裂程序，可能无法捕捉到更细微的上下文差异，且需要进一步研究不同代理架构和训练方案的影响。

---

## 480. Energy estimates of open-weight large language models across stated food weights: a dose-response evaluation

**arXiv ID:** 2611.25976 | [PDF](https://arxiv.org/pdf/2611.25976v1)

**作者:** Donato Riccio `[一作]` (Universitas Mercatorum), Iulia Brezeanu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本研究测试了开放权重的大型语言模型（LLMs）在给定确切食物重量时能否按比例估计能量，并检验计算能量密度乘以重量是否能消除误差。

**💡 创新点**

创新点在于通过两步估计方法，显著提高了能量估计的准确性，并揭示了模型在处理食物重量时的压缩现象，类似于人类饮食回忆中的平坦斜率综合症。

**🔧 技术方法**

使用了七个开放权重的LLMs，涵盖了不同的参数和架构，包括混合专家模型和推理模型。

**📊 数据集**

数据集来自美国农业部（USDA）的食品和营养数据库，包含168种随机抽取的食物，测试了九种不同的重量。

**📈 对比分析**

与其他方法相比，所有模型都表现出压缩现象，四个模型的弹性值在0.548到0.709之间，显示出严重压缩。提供能量密度后，三个模型的弹性接近比例缩放，且两步估计在极端重量下的误差显著降低。

**⚠️ 局限性**

限制在于研究仅使用了一个美国的成分数据库，且能量估计的结果依赖于该数据库的准确性。此外，研究未测试多成分餐的情况，且所有提示均为英语。

---

## 481. SEEC: A Large-Scale Dataset and Benchmark for Self-Supervised External-Eye Representation Learning

**arXiv ID:** 2611.26386 | [PDF](https://arxiv.org/pdf/2611.26386v1)

**作者:** George R. Nahass `[一作]` (University of Illinois Chicago), Sathya N. Ravi `[通讯]` (University of Illinois Chicago)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个包含435万张图像的标准化外眼数据集（SEEC），用于评估自监督外眼表示学习，并建立了相应的基准。

**💡 创新点**

创新点在于构建了一个大规模的标准化外眼图像数据集，并设计了多轴基准来测试外眼表示的几何保留、跨数据集转移和临床相关的疾病分类能力。

**🔧 技术方法**

使用了ResNet-101和ViT-B编码器，采用InfoNCE、VICReg和LeJEPA等自监督学习目标进行预训练，并与DINOv2和MAE进行了比较。

**📊 数据集**

使用了来自14个公共面部数据集的图像，构建了包含435万张对齐的单侧眼周图像的SEEC数据集。

**📈 对比分析**

通过评估嵌入几何、解剖标志恢复和疾病分类等多个方面进行比较，结果表明，特定领域的外眼编码器在跨数据集解剖转移中表现优于DINOv2和MAE，尽管预训练规模较小。

**⚠️ 局限性**

限制在于临床评估的范围有限，疾病数据集仅包含633张图像，且由于伦理限制，临床数据集未公开发布。此外，解剖探针仅评估了特定的眼周结构，未涵盖所有相关的眼周形态。

---

## 482. What a Calibrated Probability Hides: Belief-Level Post-Training for Decision Models

**arXiv ID:** 2611.25168 | [PDF](https://arxiv.org/pdf/2611.25168v1)

**作者:** Xinye Li `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Xiuying Chen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 1853 | [OpenAlex ID](https://openalex.org/A5101568165)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的后训练方法，称为信念分组策略优化（BGPO），旨在学习人群中存在的信念、估计其普遍性，并预测持有每种信念的人会如何回应。

**💡 创新点**

BGPO通过使用后验加权奖励来学习特定信念的选择分布，并通过生成候选信念列表来评估模型的预测能力，显著提高了对文本和图像基准的预测质量。

**🔧 技术方法**

BGPO使用贝叶斯规则来估计每种信念生成观察选择的概率，并结合注释者标签进行加权，同时在推理时仅依赖问题进行预测。

**📊 数据集**

使用了多个文本和图像基准数据集，包括OpinionQA、GlobalOpinionQA、DICES、ArtELingo和PARA，以及一个合成基准CLEVR-Belief。

**📈 对比分析**

与现有方法相比，BGPO在条件、最差组和总体的Jensen–Shannon散度上表现最佳，相较于最强基线，条件散度减少了多达44%。在CLEVR-Belief基准上，BGPO更准确地恢复了信念组。

**⚠️ 局限性**

BGPO的局限性在于它仍然依赖于观察到的选择来推断信念，可能在某些情况下无法完全捕捉到所有潜在信念，尤其是在信念较少或不均匀的情况下。

---

## 483. The FrontierOR Challenge: Agentic Algorithm Design for Operations Research

**arXiv ID:** 2611.26171 | [PDF](https://arxiv.org/pdf/2611.26171v1)

**作者:** Hai Wang `[一作]`, Jinhua Zhao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该论文介绍了FrontierOR挑战，旨在评估大型语言模型（LLM）代理在设计、测试和改进优化算法方面的能力，尤其是在面对未见过的问题时。

**💡 创新点**

创新点在于通过比赛的形式，鼓励团队开发能够处理多种优化问题的自动化算法设计工作流，并提供了一个公开的比较平台。

**🔧 技术方法**

使用了大型语言模型（LLM）进行程序搜索和算法设计，结合了自动化算法设计和优化问题的解决。

**📊 数据集**

数据集来自FrontierOR基准，共有180个经过审计的问题，其中60个问题在第一阶段被选出进行比赛。

**📈 对比分析**

比赛分为两个轨道：主轨道和自托管轨道，主轨道在固定计算环境中运行提交的算法，自托管轨道允许团队在自己的机器上解决问题。性能评估通过与已知参考目标的比率进行。

**⚠️ 局限性**

限制在于算法的设计和性能可能受到计算预算和问题复杂性的影响，且在某些情况下，LLM生成的算法可能无法达到专家算法的水平。

---

## 484. Anticipating Future Build Results with BuildPredictor

**arXiv ID:** 2611.26476 | [PDF](https://arxiv.org/pdf/2611.26476v1)

**作者:** Christian Macho `[一作]` (University of Klagenfurt), Katharina Stengg `[通讯]` (University of Klagenfurt)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一种基于机器学习的模型，用于预测未来的构建结果，以减少持续集成（CI）过程中不必要的构建运行，从而节省时间和资源。

**💡 创新点**

创新点在于提出了一种随机森林模型，能够利用历史构建结果、源代码变化、代码复杂性变化等多种特征进行预测，同时提供了一个无历史信息的模型变体，以提高可解释性。

**🔧 技术方法**

使用了随机森林算法作为主要技术，并结合了多种特征，包括历史构建结果、提交元数据、代码复杂性指标等。

**📊 数据集**

使用了包含24,198个提交的自定义数据集，这些提交来自25个不同的Java项目，经过严格筛选以确保数据的有效性和代表性。

**📈 对比分析**

与现有的最先进模型（如BuildFast和RavenBuild）相比，本模型在预测性能上表现更优，尤其在MCC指标上，显示出更高的稳定性和准确性。

**⚠️ 局限性**

限制在于模型对历史构建结果的依赖可能影响其可解释性，尽管无历史信息的变体提高了可解释性，但在预测性能上有所下降。

---

## 485. Coordinate Basis Linear Bandits with Sublinear Regret over Low-Rank Clusters

**arXiv ID:** 2611.25424 | [PDF](https://arxiv.org/pdf/2611.25424v1)

**作者:** Hantao Yang `[一作]` (University of Science and Technology of China), Defu Lian `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了高维线性带宽中的计算效率问题，提出了坐标基线性带宽（CBL）算法，通过聚类机制来降低计算成本并保证后悔收敛。

**💡 创新点**

CBL算法是首个在固定秩约束下实现收敛的低秩策略，采用每个聚类的正交基和坐标表示，避免了传统方法中的谱误差累积。

**🔧 技术方法**

使用了聚类机制和正交基参数化技术，结合了线性回归的统计准确性与计算效率。

**📊 数据集**

论文中未具体提及使用的数据集，但讨论了在高维特征向量和有限臂集的背景下的应用。

**📈 对比分析**

与传统的低秩近似方法（如SOFUL）相比，CBL在每轮更新的计算复杂度从O(d^2)降低到O(l^2)，并且保证了后悔收敛，避免了谱误差的影响。

**⚠️ 局限性**

当前方法的局限性在于算法的预热阶段需要读取整个数据集，因此尚不适用于具有变化臂集的更一般的线性上下文带宽。

---

## 486. Dynamic Human-Robot Role Allocation through Entropy-Based Monitoring and LLM Reasoning

**arXiv ID:** 2611.26090 | [PDF](https://arxiv.org/pdf/2611.26090v1)

**作者:** Elena Merlo `[一作]` (Istituto Italiano di Tecnologia), Arash Ajoudani `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种动态角色分配框架，使人类在任务中自由贡献，同时通过基于成本的行为树架构动态分配机器人活动，依据用户表达的偏好和行为进行调整。

**💡 创新点**

创新点在于允许人类自由选择任务，同时通过大语言模型提取用户的明确偏好，并利用香农熵分析在线监测人类活动，识别异常并触发机器人纠正动作。

**🔧 技术方法**

使用了基于行为树的决策结构、香农熵算法和大语言模型（LLM）来处理用户指令和监测人类活动。

**📊 数据集**

通过一项涉及14名参与者的多主体实验进行验证，任务为组装Tangram图形。

**📈 对比分析**

与传统的基于任务分配的方法相比，提出的方法在用户体验上表现更好，参与者感到更有控制感和参与感，且任务执行时间和人类闲置时间显著减少。

**⚠️ 局限性**

限制在于当前的LLM处理能力主要针对Tangram组装任务，扩展到更复杂的现实场景时可能面临挑战，且用户请求在机器人执行活动时无法立即影响其行为。

---

## 487. Verifiable Logic in Medical LLMs: A Critical Review of Agentic Clinical Reasoning Optimization and Neuro-Symbolic Grounding

**arXiv ID:** 2611.25898 | [PDF](https://arxiv.org/pdf/2611.25898v1)

**作者:** Youssef ElNahas `[一作]`, Wadii Boulila `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对临床大型语言模型（LLMs）进行了批判性回顾，探讨了模型在生成流畅、合理的输出时可能存在的无效推理问题，强调了静态基准测试无法检测到的临床幻觉现象。

**💡 创新点**

提出了一个综合框架，将代理失败模式映射到符号干预点，强调临床AI安全性在很大程度上依赖于系统架构，而不仅仅是训练或校准。

**🔧 技术方法**

使用了过程奖励建模（Process Reward Modeling）和神经符号整合（Neuro-Symbolic Integration）等技术，结合了形式逻辑引擎、知识图谱和本体推理器。

**📊 数据集**

使用了多个小样本和模拟环境中的数据集，具体数据集未详细列出。

**📈 对比分析**

与现有方法的比较显示，虽然过程奖励模型在某些安全措施上取得了显著提升，但这些结果主要来自单一研究，且在小样本或模拟环境中进行，无法与神经基线直接比较。

**⚠️ 局限性**

本文的局限性在于提出的综合框架尚未在临床环境中验证，且当前的监管强调性能而不要求推理步骤的验证。

---

## 488. Multi-View Relighting with Mergeable Radiance Fields

**arXiv ID:** 2611.25523 | [PDF](https://arxiv.org/pdf/2611.25523v1)

**作者:** Yinan Wang `[一作]` (McGill University), James J. Clark `[通讯]` (McGill University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为MergeableRF的方法，通过模型参数合并实现辐射场的重光照。该方法利用已知基础光源下的多视角图像，直接组合辐射场的参数以重光照场景。

**💡 创新点**

创新点在于引入了合并感知训练，使得不同基础光源下训练的外观网络参数可以线性组合，从而实现新的光照效果，而无需额外的优化或材料估计。

**🔧 技术方法**

使用了神经辐射场（NeRF）和3D高斯点（3DGS）技术，并结合了合并感知训练策略。

**📊 数据集**

使用了Neural Gaffer基准中的合成对象和OLATverse中的真实光阶段捕获数据集进行实验。

**📈 对比分析**

与其他重光照方法相比，MergeableRF在合成对象上表现出竞争力，NeRF模型在PSNR上排名第二，而3DGS模型在真实捕获中表现最准确。合并模型的PSNR下降在1.2到2.5 dB之间。

**⚠️ 局限性**

限制在于该方法需要在已知基础光源下的多视角图像，且平滑的基础光源限制了锐利光源和镜面反射的再现。

---

## 489. Beyond the Sink Interpretation: Radial Attention in Flow-Based VLA Policies

**arXiv ID:** 2611.26385 | [PDF](https://arxiv.org/pdf/2611.26385v1)

**作者:** Yuxi Zheng `[一作]` (Duke University), Yiran Chen `[通讯]` (Duke University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文探讨了流基于视觉-语言-动作（VLA）策略中，动作序列的首尾标记的注意力分布及其对执行效果的影响，特别是这些标记是否符合语言模型中的“沉没”特征。

**💡 创新点**

创新点在于验证了动作序列的首尾标记并不符合传统的沉没特征，且其对成功率的影响远小于内部标记的影响。

**🔧 技术方法**

使用了流匹配的去噪方法和旋转位置编码（RoPE）技术来分析动作专家的注意力分布。

**📊 数据集**

使用了RoboCasa数据集，进行了250个配对实验，涉及五个任务（如打开冰箱、打开柜子等）。

**📈 对比分析**

通过与基线的配对比较，发现对首尾标记的编辑对成功率的影响小于4分，而对内部标记的编辑则降低成功率28.4分，显示出内部标记的重要性。

**⚠️ 局限性**

限制在于所有结论基于单一的π_0.5检查点，且执行结果不具可重复性，可能影响结果的普遍性。

---

## 490. Evaluating safeguarding in multi-turn conversations with AI assistants

**arXiv ID:** 2611.26009 | [PDF](https://arxiv.org/pdf/2611.26009v1)

**作者:** Charlotte Peart `[一作]` (Oxford Internet Institute, University of Oxford), Luc Rocher `[通讯]` (Oxford Internet Institute, University of Oxford)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究测试了对话式AI助手在处理用户逐渐透露的心理健康困扰时的响应能力，特别是在用户同时或逐步分享与抑郁、精神病或家庭暴力相关的线索时的表现。

**💡 创新点**

创新点在于通过与三位经验丰富的社区心理健康支持工作者合作，设计了九种场景，评估AI模型在不同信息呈现方式下的响应效果，强调了逐步披露困扰对AI助手响应的影响。

**🔧 技术方法**

使用了多种对话式AI模型，包括GPT-5.5、Grok 4.3、DeepSeek V4、Llama 4 Maverick、Claude Sonnet 5和Claude Opus 5，评估其在不同情况下的响应能力。

**📊 数据集**

研究使用了一个包含21轮日常对话的场景，用户在对话中逐渐分享与心理健康相关的五个线索，共进行了7400多次对话的评估。

**📈 对比分析**

研究发现，当线索集中在一条消息中时，模型的响应符合支持工作者标准的比例为26%-59%；而当线索分散在对话中时，仅为3%-6%。Claude模型的表现相对较好，符合标准的比例为68%-97%（集中）和59%-87%（分散）。

**⚠️ 局限性**

本研究的局限性包括使用API收集响应时未启用推理，且对话较短，未能完全模拟真实生活中用户可能经历的长时间、多轮对话。

---

## 491. FlashX: Scaling Robotic Experience Reuse for Embodied Agents with CBA-Based NAND Flash

**arXiv ID:** 2611.25580 | [PDF](https://arxiv.org/pdf/2611.25580v1)

**作者:** Yuseon Choi `[一作]` (Gwangju Institute of Science and Technology), Sangjin Kim `[通讯]` (Gwangju Institute of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于CBA的NAND闪存架构FlashX，用于在具身智能体中实现可扩展的经验重用，以提高任务成功率和推理效率。

**💡 创新点**

创新点在于结合了并行pSLC关键字搜索与密集TLC潜在存储，通过并行CIF执行解决了闪存搜索瓶颈，同时引入了基于本体感知的聚类和增量经验放置，减少了搜索空间。

**🔧 技术方法**

使用了CMOS直接键合到阵列（CBA）技术的NAND闪存架构，结合了并行pSLC关键字搜索和计算在闪存（CIF）引擎。

**📊 数据集**

使用了VLABench和LIBERO-Plus数据集，评估了在不同工作负载下的任务成功率和性能。

**📈 对比分析**

与传统的SSD检索和先前的NAND搜索架构相比，FlashX在任务成功率上提高了12.24%，在搜索吞吐量上提高了2.57倍，最终实现了1.29到3.95倍的端到端加速和1.12到2.34倍的能效提升。

**⚠️ 局限性**

限制在于当相关经验稀缺或当前观察与存储经验差异较大时，所获得的好处可能会受到限制。此外，基于本体感知的聚类需要在不同机器人平台上进行初步校准。

---

## 492. Incorporating argumentation frameworks for LLM decision-making: Task-specific design and evaluation

**arXiv ID:** 2611.25361 | [PDF](https://arxiv.org/pdf/2611.25361v1)

**作者:** Bochen Han `[一作]` (State Key Laboratory of Mathematical Sciences, Academy of Mathematics and Systems Science, Chinese Academy of Sciences), Mingyang Ma `[通讯]` (State Key Laboratory of Mathematical Sciences, Academy of Mathematics and Systems Science, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了如何将论证框架（AFs）融入大型语言模型（LLM）的决策过程，提出了两种任务特定的方法Qnt-Arg和Qlt-Arg，并对其在辩论决策和基于检索的事实验证任务中的表现进行了评估。

**💡 创新点**

创新点在于提出了一种通用的工作流程，并基于此开发了两种方法，分别针对辩论和检索任务进行了优化，强调了任务特定设计的重要性。

**🔧 技术方法**

使用了论证框架（AFs）和渐进语义（gradual semantics）等技术，结合了生成模型和决策模型的角色。

**📊 数据集**

使用了六个辩论数据集（如CMV、DebateArt等）和多个基于检索的事实验证数据集（如CONFACT、AVeriTeC、DeepFact-Bench）。

**📈 对比分析**

与现有方法（如ArgLLM和ArgRAG）进行比较，Qnt-Arg在辩论数据集上表现出78.41%的准确率，优于直接LLM推理的74.38%和最强基线的76.99%；Qlt-Arg在检索任务中也表现出色，尤其是在使用GE过滤时。

**⚠️ 局限性**

限制在于当前实验覆盖面有限，需要在更多样化的数据集和决策任务上进行评估，以检验模型的泛化能力。

---

## 493. Bottom-Up FPGA Reconfigurable Accelerator Overlay Generation

**arXiv ID:** 2611.25293 | [PDF](https://arxiv.org/pdf/2611.25293v1)

**作者:** Mohamed Bouaziz `[一作]` (King Abdullah University of Science and Technology), Suhaib A. Fahmy `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

提出了一种从工作负载的数据流图生成可编程FPGA覆盖层的框架，能够在不重新综合的情况下运行新工作负载。

**💡 创新点**

创新点在于通过结构相似性对工作负载进行聚类，利用谱描述符匹配操作和连接，构建共享硬件，同时尊重操作语义、数据类型、操作数角色和执行约束。

**🔧 技术方法**

使用了谱分析、MLIR辅助前端、图分析和RTL生成等技术。

**📊 数据集**

使用了来自三个工作负载套件的17个工作负载进行评估，包括MachSuite、DSP和Vision。

**📈 对比分析**

与OverGen进行比较，结果显示在几何平均上速度提升了9.52倍，使用的FPGA资源减少了19.4-22.1倍，估计的板能耗减少了9.67-10.51倍。

**⚠️ 局限性**

限制在于当前框架尚未支持复杂的控制流、额外的操作和数据类型，以及动态内存访问模式。

---

## 494. PhyDEX: A Benchmark for Physical Reasoning in Vision-Language-Action Models via Dynamic Extrinsic Dexterity

**arXiv ID:** 2611.26565 | [PDF](https://arxiv.org/pdf/2611.26565v1)

**作者:** Hao Vo `[一作]` (University of Arkansas), Ngan Le `[通讯]` (University of Arkansas)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种新的投掷基准，评估机器人在语言条件下的物理理解能力，特别是在动态外部灵巧性方面。

**💡 创新点**

创新点在于引入了一个新的基准，专注于机器人如何推断物体的质量、重心和摩擦力，以便将物体投掷到超出其运动范围的目标位置。

**🔧 技术方法**

使用了一种可扩展的全自动数据生成管道，结合了抓取生成、运动规划和闭环投掷优化技术。

**📊 数据集**

使用了一个包含35K轨迹的合成数据集，涵盖9个投掷任务和100个场景，涉及均匀、不规则和动态质量的物体。

**📈 对比分析**

与三种代表性的视觉语言行动模型（GR00T N1.7、DynamicVLA和ForceVLA）进行比较，发现它们的成功率均低于3%。分析表明，主要瓶颈在于预训练模型的鲁棒性不足和物理盲目性。

**⚠️ 局限性**

限制在于只关注单一的动态技能（投掷），并且仅随机化了质量、重心和摩擦力，未来需要扩展到更多技能和更复杂的物理特性。

---

## 495. Technology-Enabled Fuel Supply Chain Resilience: An Integrated Information System for Real-Time Monitoring and Decision Support

**arXiv ID:** 2611.25542 | [PDF](https://arxiv.org/pdf/2611.25542v1)

**作者:** Mohamed Loutis `[一作]` (Sorbonne Université), Esma Aïmeur `[通讯]` (Université de Montréal)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估并扩展了早期燃料物流研究中描述的事务信息系统，重点关注数据可靠性、监控和决策支持。

**💡 创新点**

通过实施修订的事务架构，改善了数据集成和报告准确性，并增加了库存移动记录、警报和恢复控制。

**🔧 技术方法**

使用了Delphi/SQL代码和SQL Server进行技术评估，采用了合成交易和美国公共能源系列数据。

**📊 数据集**

使用了合成操作输入和美国能源信息管理局的公共每周数据集。

**📈 对比分析**

与重建的源比较，R1在数据集成缺陷率上从2.90-3.46%降至0.14-0.49%。在报告生成方面，R1生成的报告字段完全正确，而R0存在多个错误。

**⚠️ 局限性**

R0结合了原始T-SQL与重建的表，未经过验证的历史部署，操作效益依赖于手动比较，且需要现场测试。

---

## 496. Neural Conditional Gaussian Model: Efficient Data Assimilation with Sequential Uncertainty Quantification for Nonlinear Dynamics

**arXiv ID:** 2611.26463 | [PDF](https://arxiv.org/pdf/2611.26463v1)

**作者:** Chuanqi Chen `[一作]` (University of Chicago), Nan Chen `[通讯]` (University of Wisconsin–Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种名为神经条件高斯模型（Neural CGM）的数据驱动动态模型，用于在部分观测的非线性系统中进行数据同化、状态预测和不确定性量化。

**💡 创新点**

创新点在于Neural CGM能够在潜在空间中保留条件高斯动态，并通过分布感知的高斯解码器将后验统计信息映射回物理变量，同时将状态预测与后验推断分开处理。

**🔧 技术方法**

使用了神经网络来参数化非线性编码器和解码器，以及动态函数，结合条件高斯滤波器实现高效的数据同化。

**📊 数据集**

在Lorenz 63、Lorenz 96和Kuramoto-Sivashinsky系统上进行了数值实验，使用的训练数据为成对的时间序列数据。

**📈 对比分析**

与传统的非线性数据同化方法（如扩展卡尔曼滤波器、粒子滤波器等）相比，Neural CGM在状态预测和数据同化的准确性上表现出竞争力，同时在线同化成本显著降低，尤其在高维系统中表现更为突出。

**⚠️ 局限性**

限制在于训练Neural CGM需要调整重建、预测和数据同化目标的相对权重，且当前实现仅使用对角物理空间后验协方差，未来工作可考虑扩展到结构化协方差表示。

---

## 497. GateTrace: Where Do Crowd Crossings Go Missing Under Occlusion?

**arXiv ID:** 2611.26388 | [PDF](https://arxiv.org/pdf/2611.26388v1)

**作者:** Ao He `[一作]` (University of Florida), Vivian W. H. Wong `[通讯]` (University of Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究提出了GateTrace框架，用于评估视频监控中人群流动的交叉情况，特别是在低可见度条件下的交叉丢失情况。

**💡 创新点**

创新点在于通过引入GateTrace，能够从注释轨迹中匹配交叉报告，并追踪被总计隐藏的遗漏，提供了对交叉恢复的可视化评估。

**🔧 技术方法**

使用了YOLOX检测器和多个跟踪算法（ByteTrack、OC-SORT和BoT-SORT）来分析和比较交叉情况。

**📊 数据集**

使用了MOT20数据集，该数据集包含注释的行人轨迹和可见性信息，适用于评估人群流动监测系统。

**📈 对比分析**

与其他方法相比，GateTrace能够更好地识别低可见度下的交叉丢失情况，发现大多数低可见度交叉没有匹配报告，而高可见度交叉的恢复率接近100%。

**⚠️ 局限性**

限制在于研究仅覆盖了四个MOT20序列，场景条件较少，且不同跟踪器共享同一检测器，可能影响结果的普遍性。

---

## 498. Is a Description Worth a Thousand Demonstrations? Specificity and Reliability of Task Descriptions in In-Context Learning

**arXiv ID:** 2611.25560 | [PDF](https://arxiv.org/pdf/2611.25560v1)

**作者:** Bruce Nguyen `[一作]` (Decaude), Linglong Kong `[通讯]` (University of Alberta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文研究了在上下文学习中，任务描述的特异性和可靠性如何影响模型的预测能力，提出了一种基于贝叶斯视角的模型来量化描述的预测价值。

**💡 创新点**

创新点在于将任务描述视为信息先验，通过有效样本量（ESS）来量化其预测价值，并探讨了描述的特异性和可靠性对其价值的影响。

**🔧 技术方法**

使用了贝叶斯统计方法，特别是有效样本量（ESS）来评估任务描述的预测能力。

**📊 数据集**

使用了高斯线性回归作为控制实验的基础，模型在不同的特异性和可靠性条件下进行测试。

**📈 对比分析**

通过与贝叶斯最优预测器的比较，发现元训练的变换器在训练分布上接近贝叶斯最优的ESS，而预训练的语言模型在没有推理时对高度特异性的描述利用不足，只有在启用推理时才能接近贝叶斯最优的预测。

**⚠️ 局限性**

限制在于模型假设描述的相关性是全有或全无，未来的工作可以考虑更复杂的描述形式，如多个候选任务的混合高斯模型。

---

## 499. DiLE: Discrimination-Aware Latent-Space Learning for Universal Multimodal Embeddings

**arXiv ID:** 2611.25419 | [PDF](https://arxiv.org/pdf/2611.25419v1)

**作者:** Ben Kang `[一作]` (Dalian University of Technology), Huchuan Lu `[通讯]` (Dalian University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为DiLE的框架，用于在多模态大语言模型中进行区分意识的潜在空间学习，以提高检索的准确性。

**💡 创新点**

创新点在于结合任务指导和基于策略的检索反馈，通过共享-私有专家适应和基于策略的困难负样本蒸馏来保留嵌入中的重要区分证据。

**🔧 技术方法**

使用了任务引导的潜在空间学习（TLSL）和基于策略的困难负样本蒸馏（OPHD）等技术。

**📊 数据集**

使用了MMEB-v2、UVRB和MRMR等多个数据集进行评估。

**📈 对比分析**

与多种现有方法进行比较，DiLE在MMEB-v2、UVRB和MRMR上均取得了最先进的性能，显示出其在多模态检索任务中的有效性。

**⚠️ 局限性**

限制在于模型的复杂性和训练过程中的计算资源需求，可能对实时应用造成挑战。

---

## 500. From Sequence to Space: Explicit Spatial Belief with Vision-Language Distillation for Active Geo-Localization

**arXiv ID:** 2611.25866 | [PDF](https://arxiv.org/pdf/2611.25866v1)

**作者:** Ziyang Xu `[一作]` (National Space Science Center), Hai Zhou `[通讯]` (National Space Science Center)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为EAGLE的显式空间信念代理，用于主动地理定位和探索，旨在通过离线训练优化教师条件的Swin分支和无上下文的Swin学生，从而提高无人机在复杂环境中的目标导航能力。

**💡 创新点**

创新点在于引入了网格对齐的空间信念表示，采用离线蒸馏框架将全场景视觉语言模型的语义知识转移到无上下文的跟踪器，并开发了基于方差减少的覆盖目标，避免了基于距离的奖励塑造。

**🔧 技术方法**

使用了Swin Transformer作为基础架构，并结合了PPO（Proximal Policy Optimization）进行策略优化，同时利用了全场景视觉语言模型（VLM）进行离线监督。

**📊 数据集**

使用了MASA、xBD和MM-GAG数据集进行评估，涵盖了不同的目标模式，包括航空图像、地面图像和文本描述。

**📈 对比分析**

与现有方法（如GOMAA-Geo和GeoExplorer）相比，EAGLE在MASA、xBD和MM-GAG数据集上分别达到了78.1%、76.4%和76.8%的平均成功率，表现出显著的性能提升。

**⚠️ 局限性**

限制在于EAGLE在推理阶段不使用全场景VLM或缓存的视觉标记，可能在某些复杂场景中影响性能，同时对教师条件蒸馏的依赖可能限制了其在新环境中的适应能力。

---

## 501. From Data Alignment to Parameter Generation: Few-Shot Cross-City Traffic Prediction with Conditional Diffusion

**arXiv ID:** 2611.25592 | [PDF](https://arxiv.org/pdf/2611.25592v1)

**作者:** Shengchun Wang `[一作]` (Hunan Normal University), Zhu Han `[通讯]` (University of Houston)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于条件扩散的参数生成框架（CDPG），用于解决跨城市的少样本网络流量预测问题。

**💡 创新点**

CDPG框架通过学习高性能预测器参数的条件分布，直接为目标城市生成特定节点的模型参数，避免了传统方法中数据对齐的脆弱性。

**🔧 技术方法**

使用条件扩散模型进行参数生成，结合多模态时空上下文，包括卫星图像和大语言模型（LLM）生成的语义描述。

**📊 数据集**

在三个大型真实世界的移动流量数据集上进行实验，分别来自上海、南京和南昌。

**📈 对比分析**

CDPG在标准预测指标上比现有的跨城市少样本基线方法提高了最多9.72%的性能，表现出在少样本多步预测任务中的稳健性。

**⚠️ 局限性**

限制在于实验仅在三个中国城市进行，且数据收集时间不同，可能影响结果的普遍性。未来工作将探索不同的适应预算和更广泛的数据集。

---

## 502. The Metacomplexity Structure of Search Problems

**arXiv ID:** 2611.25046 | [PDF](https://arxiv.org/pdf/2611.25046v1)

**作者:** Shalev Ben-David `[一作]`, M. H. Ebtehaj `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究了反驳给定搜索问题的算法或反驳两个搜索问题之间的映射归约的元复杂性任务，展示了这些任务与映射归约下搜索问题的格结构密切相关。

**💡 创新点**

通过将元复杂性与海廷代数结构联系起来，能够将直觉逻辑中的结果转化为复杂性理论的后果，揭示了反驳任务与复杂性类之间的深刻联系。

**🔧 技术方法**

使用了海廷代数和映射归约的结构复杂性理论，结合了直觉逻辑的工具和概念。

**📊 数据集**

使用了可验证的搜索问题集，特别是涉及到的复杂性类如𝖥𝖯和𝖵𝖾𝗋𝗂𝖿𝗂𝖺𝖻𝗅𝖾。

**📈 对比分析**

通过与现有复杂性类的比较，展示了反驳任务的复杂性与直觉逻辑的相似性，证明了反驳任务遵循直觉逻辑的规则，且在某些情况下具有非中介性。

**⚠️ 局限性**

限制在于对某些复杂性类的完全性和可验证性要求，可能导致某些反驳任务无法被有效地解决或分类。

---

## 503. A Complete Geometric Message-Passing Architecture for Point Clouds

**arXiv ID:** 2611.26384 | [PDF](https://arxiv.org/pdf/2611.26384v1)

**作者:** Nadav Dym `[一作]` (Technion), Christopher Morris `[通讯]` (RWTH Aachen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种新的等变消息传递架构，用于三维点云，并证明其完整性。

**💡 创新点**

创新点在于该架构能够在较低的复杂度下实现完整性，挑战了高阶特征的必要性。

**🔧 技术方法**

使用了一种简单的等变消息传递机制，结合了节点特征的交叉乘积来定义基。

**📊 数据集**

使用了合成实验数据集，展示了该架构在区分复杂点云对方面的能力。

**📈 对比分析**

与基于距离的消息传递网络相比，提出的架构在理论上证明了其完整性，并在实验中显示出更好的性能。

**⚠️ 局限性**

该研究的局限性在于目前仅提供了初步的玩具规模结果，尚未在真实世界任务中进行广泛验证。

---

## 504. A Human-Centric Adaptive Multi-Fidelity Digital Twin Framework for Cyber-Physical Systems

**arXiv ID:** 2611.25082 | [PDF](https://arxiv.org/pdf/2611.25082v1)

**作者:** Clarisa Grijalva-Lugo `[一作]` (University of Arizona), Pratik Satam `[通讯]` (University of Arizona)

**通讯引用:** 1067 | [OpenAlex ID](https://openalex.org/A5028622409)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种以人为中心的自适应多保真数字双胞胎框架（HAMF-DT），用于建模不断变化的人类、物理双胞胎和数字双胞胎核心，并增加了人类推理和适应功能。

**💡 创新点**

创新点在于将保真度视为设计维度，定义了低、中、高保真度层次，并通过适应性和耦合性来区分这些层次。

**🔧 技术方法**

使用了多种技术，包括虚拟现实、计算机数控铣削模型和工业培训站的实例，结合了人类推理和适应功能。

**📊 数据集**

使用了NASA铣削数据集和教育对话数据集进行实验评估。

**📈 对比分析**

通过跨层比较，展示了不同层次在适应性和耦合性方面的差异。中层的预测误差下降范围为0.65到0.88，而高层的预测误差下降范围为0.39到0.89。

**⚠️ 局限性**

限制在于未能实现数字双胞胎与物理双胞胎之间的实时闭环交互，耦合比率描述的是数字双胞胎延迟相对于过程步骤的表现，而不是闭环性能。

---

## 505. Bridging the Discovery Gap in Solid Dataspaces with a Semantic Data Catalog

**arXiv ID:** 2611.26611 | [PDF](https://arxiv.org/pdf/2611.26611v1)

**作者:** Florian Hölken `[一作]` (Institute for Technologies and Management of Digital Transformation), André Pomp `[通讯]` (Institute for Technologies and Management of Digital Transformation)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

提出了一种符合FAIR原则的语义数据目录，以解决去中心化Solid数据空间中的数据发现挑战。

**💡 创新点**

创新点在于将语义模型与DCAT元数据标准结合，增强数据的可发现性、互操作性和重用性。

**🔧 技术方法**

使用了DCAT（数据目录词汇）标准和PLASMA工具进行语义建模。

**📊 数据集**

使用了Solid Community Server（SCS）作为测试环境，注册了多个Solid数据pod和数据集。

**📈 对比分析**

与传统数据空间架构相比，提出的解决方案在数据发现和共享方面表现出更高的效率，但当前实现仅支持公共数据集，限制了其在需要控制访问的场景中的适用性。

**⚠️ 局限性**

当前实现的局限性包括仅支持公共数据集，缺乏对私有数据集的管理机制，以及尚未实现其他DCAT组件（如数据服务和数据集系列）。

---

## 506. Modeling Learner-Technology Interaction Through Multilevel Gaze Transition Dynamics to Inform Usability in Online Learning

**arXiv ID:** 2611.25294 | [PDF](https://arxiv.org/pdf/2611.25294v1)

**作者:** Yueqi Weng `[一作]` (University of Georgia), fan Yang `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过多层次注视动态建模学习者与技术的互动，以评估在线学习的可用性，结合眼动追踪数据和后续焦点小组访谈，分析高中生在AI支持的生物学在线课程中的学习行为。

**💡 创新点**

创新点在于将多层次的注视分析与学习者的反馈结合，提供了更全面的可用性评估视角，揭示了学习者在学习过程中的个体差异和互动模式。

**🔧 技术方法**

使用了眼动追踪技术，结合了区域兴趣（AOI）转移、注视熵、幻灯片级导航和基于网络的表示方法。

**📊 数据集**

数据集包括七个高中生在AI支持的在线生物课程中的眼动记录，以及三组后续焦点小组访谈的数据。

**📈 对比分析**

通过与传统的可用性评估方法比较，研究发现学习者在幻灯片内的注视分配和切换模式存在显著个体差异，且学习者在导航路径上有共同点但在重访行为上存在差异，表明多层次注视分析能够揭示更复杂的学习行为。

**⚠️ 局限性**

本研究的局限性在于样本量较小，仅基于单一的在线课程，结果可能无法推广到其他学习环境或学习者群体。此外，眼动数据的处理方法可能影响结果的准确性，且未能深入探讨学习者的意图或学习成果。

---

## 507. When Sounds Turn Heads: Recovering Event Timing That Audio LLMs Fail to Say

**arXiv ID:** 2611.25960 | [PDF](https://arxiv.org/pdf/2611.25960v1)

**作者:** Jongbhin Woo `[一作]` (KAIST), Joon Son Chung `[通讯]` (KAIST)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

论文研究了音频大语言模型（LLMs）在时间定位方面的表现，发现一些注意力头能够追踪查询的声音，但模型的回答未能准确表达这些信息。提出了一种新的解码方法，即头引导解码，旨在将这些时间信息整合到模型的回答中。

**💡 创新点**

创新点在于提出了头引导解码方法，该方法无需训练，通过结合模型的候选答案概率和来自注意力头的先验信息来改善时间戳的准确性。

**🔧 技术方法**

使用了头引导解码技术，该技术利用模型内部的注意力头信息来重新评分候选答案的时间戳。

**📊 数据集**

使用了UnAV-100开发集的23个录音子集进行头部选择，并在多个数据集上进行评估，包括RealDESED和AudioGrounding的自由文本查询。

**📈 对比分析**

与其他方法相比，头引导解码在所有基准测试中均表现优于模型的原始答案和随机头引导，尤其在RealDESED数据集上，mIoU从0.07-0.36提高到0.48-0.56。

**⚠️ 局限性**

限制在于，UnAV-100和RealDESED的查询是单次事件，且头引导解码无法恢复模型报告为缺失的声音。此外，模型的存在门限制了其在某些情况下的表现。

---

## 508. What's in the box? Revealing Object Layouts Inside Opaque Boxes via Contactless Vibration Excitation and Sensing

**arXiv ID:** 2611.25728 | [PDF](https://arxiv.org/pdf/2611.25728v1)

**作者:** Ethan Turok `[一作]` (Weizmann Institute of Science), Mark Sheinin `[通讯]` (Weizmann Institute of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种通过观察箱体表面振动来预测箱内物体的二维占用布局的方法，称为WhisperBox。

**💡 创新点**

创新点在于该方法无需物理接触即可激发或感知振动，能够从振动中推断出封闭容器内的空间布局。

**🔧 技术方法**

使用了一种基于学习的模型，结合激光点阵和相机测量表面振动，并通过卷积解码器将其解码为占用图。

**📊 数据集**

使用了包含三种不同箱体和六种物体的实验数据集，共收集了15891个样本。

**📈 对比分析**

通过与训练集中的新物体位置进行比较，模型在单物体场景中的分类准确率达到99%，而在双物体场景中为80%。在分布外评估中，模型能够准确定位未见物体的布局。

**⚠️ 局限性**

模型在木箱上的表现较差，且对低反射材料的适应性较差，数据收集过程依赖手动，限制了模型的扩展性。

---

## 509. SimPyWireless: A PHY-Aware Discrete-Event Simulation Framework for Wi-Fi and 5G NR-U Coexistence in the 5 GHz Band

**arXiv ID:** 2611.25350 | [PDF](https://arxiv.org/pdf/2611.25350v1)

**作者:** Md Rashedur Rahman `[一作]` (George Mason University), Paulo Costa `[通讯]` (George Mason University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

提出了SimPyWireless，一个用于Wi-Fi和5G NR-U在5 GHz带宽共存的离散事件仿真框架。

**💡 创新点**

创新点在于提供了一个轻量级、开源的PHY感知仿真平台，能够进行可重复的Wi-Fi/NR-U共存研究，并支持AI/ML模型的集成。

**🔧 技术方法**

使用了离散事件仿真技术，结合了共享信道建模、传播、干扰、SINR、异构流量和移动性等因素。

**📊 数据集**

使用了5 GHz未授权频段的仿真数据集，验证了Wi-Fi和5G NR-U的共存行为。

**📈 对比分析**

与DTMC模型进行了比较，结果显示SimPyWireless在不同竞争和移动条件下的共存行为与理论预测相符，性能表现良好。

**⚠️ 局限性**

限制在于当前框架主要集中在5 GHz频段，未来工作将扩展到其他频段和更丰富的PHY层特性。

---

## 510. Design and Evaluation of a Body-Machine Interface as an Alternative Control Paradigm for Pedal-based Sewing Machines

**arXiv ID:** 2611.25261 | [PDF](https://arxiv.org/pdf/2611.25261v1)

**作者:** Ashrut Aryal `[一作]` (Michigan State University), Ranjan Mukherjee `[通讯]` (Michigan State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

本研究设计并评估了一种身体-机器接口（BoMI），作为踏板缝纫机控制的替代控制范式，重点在于同时执行缝纫机控制和手动操作。

**💡 创新点**

创新点在于开发了一种能够将躯干运动映射到缝纫机完整功能的BoMI，支持在进行手动操作的同时控制缝纫机。

**🔧 技术方法**

使用了身体-机器接口（BoMI）技术，通过无线惯性测量单元（IMU）测量参与者的躯干角度，并将其转换为控制缝纫机的电压信号。

**📊 数据集**

使用了24名没有缝纫经验的大学生作为参与者，分为BoMI组和踏板组，每组各12人。

**📈 对比分析**

与传统踏板控制相比，BoMI组在训练中表现出更少的压脚抬起错误，并在首次转移任务中表现出更少的错误。两组在完成时间和错误计数上没有显著差异，表明BoMI能够有效支持缝纫机控制。

**⚠️ 局限性**

本研究的局限性包括仅在健康年轻人中进行测试，未考虑运动障碍用户的表现；样本量较小，可能限制了统计能力；以及未评估缝纫过程中的其他重要因素，如缝合完整性和材料处理。

---

## 511. Verified Netlist-to-RTL Recovery with LLM Agents

**arXiv ID:** 2611.25324 | [PDF](https://arxiv.org/pdf/2611.25324v1)

**作者:** Ahmet Efe `[一作]` (University of Minnesota), Ulya R. Karpuzcu `[通讯]` (University of Minnesota)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于大型语言模型（LLM）的攻击方法LOCA，通过将匿名化的网表分割成逻辑锥，利用并行代理重写这些逻辑锥并组装出紧凑的可读寄存器传输级（RTL）代码。

**💡 创新点**

创新点在于使用逻辑锥来划分网表，使每个代理能够专注于相关寄存器的完整下一个状态逻辑，从而提高了RTL的恢复率。

**🔧 技术方法**

使用了大型语言模型（如Claude Haiku 5.5、Claude Sonnet 5.5和GPT-6.1-sol）以及形式验证工具来验证每个重写的正确性。

**📊 数据集**

使用了ITC'99控制器、IWLS IP核心和生成的顺序电路等多个数据集进行实验。

**📈 对比分析**

与整体网表提示方法相比，LOCA在21个模型-设计对中恢复了17个，而整体网表提示仅恢复了6个。LOCA的每次运行恢复率从5%提高到74%，且成本低于一美元。

**⚠️ 局限性**

局限性在于功能保留的混淆方法未能有效阻止LOCA的恢复，而未知密钥或隐藏单元功能则阻止了原始设计的恢复证明。

---

## 512. Safe AI Evaluation: A Systems Perspective

**arXiv ID:** 2611.24987 | [PDF](https://arxiv.org/pdf/2611.24987v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 513. MIR-Depth: Multi-Range Refinement for Single-View RGB-D Depth Completion of Transparent Objects

**arXiv ID:** 2611.25417 | [PDF](https://arxiv.org/pdf/2611.25417v1)

**作者:** Xinzhe Fu `[一作]` (Harbin Engineering University), Rongsheng Li `[通讯]` (University of Liverpool)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为MIR-Depth的方法，用于从单视图RGB-D图像和物体掩码中恢复透明物体的可见表面深度。

**💡 创新点**

创新点在于构建了坐标感知的多范围假设，结合了全局逆深度体积和局部深度体积，以解决大折射偏移和细微表面细节的恢复问题。

**🔧 技术方法**

使用了多尺度卷积门控递归单元（ConvGRU）和空间门控校正技术来迭代更新逆深度。

**📊 数据集**

在TransCG、TODD、TRansPose、ClearPose和HouseCat6D等数据集上进行了实验。

**📈 对比分析**

与ReMake、DFNet、BP-Net和PSD等方法进行比较，MIR-Depth在深度恢复精度上表现更优，MAE和RMSE分别为6.833和10.504毫米，低于其他方法。

**⚠️ 局限性**

MIR-Depth无法揭示未观察到的几何形状，特别是在完全透明物体与均匀背景之间几乎没有可见边界的情况下。

---

## 514. Verify2Act: Critic-Guided Latent World Models for Verifying Language-Conditioned Manipulation Plans

**arXiv ID:** 2611.25243 | [PDF](https://arxiv.org/pdf/2611.25243v1)

**作者:** Chrisantus Eze `[一作]` (Oklahoma State University), Christopher Crick `[通讯]` (Oklahoma State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Verify2Act，一个模块化的验证框架，分离语义动作提议与潜在物理验证，利用VLM提出候选计划，并通过V2A-WM在DINOv2特征空间中想象这些计划，最后由双头评论员检查时间一致性和目标满足情况。

**💡 创新点**

创新点在于引入了V2A-WM，结合了交叉注意力动作定位和因果时间历史窗口，提升了长时间操作的验证能力，并且实现了低成本机器人手臂的有效应用。

**🔧 技术方法**

使用了VLM（视觉语言模型）、V2A-WM（视觉特征基础的流匹配世界模型）、双头评论员等技术。

**📊 数据集**

使用了在数字双胞胎生成的数据集进行训练，包含20,000个回合（35,000个转移），涵盖了多种技能和目标。

**📈 对比分析**

与其他方法比较时，Verify2Act在六个语言条件任务中达到了83.3%的任务成功率，相比之下，未进行架构更改的潜在世界模型成功率为61.7%，仅使用VLM的成功率为53.3%，而像素扩散世界模型的成功率为46.7%。

**⚠️ 局限性**

限制在于大多数剩余失败源于共享的低级技能策略，验证的空间有限；此外，真实世界试验数量较少，物理评估仅限于桌面块操作，且real2sim对齐过程增加了处理时间。

---

## 515. SHUSH: Emission-Aware Reward Design for Proactive Video Assistants

**arXiv ID:** 2611.26478 | [PDF](https://arxiv.org/pdf/2611.26478v1)

**作者:** Shawn Liang `[一作]`, Gourav Datta `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的奖励设计方法，旨在减少主动视频助手的过度发声现象，特别是通过引入一种考虑发声数量的F1奖励机制。

**💡 创新点**

创新点在于通过排他性发声-跨度匹配来定价错误发声和遗漏跨度，从而使额外的响应必须提供足够的价值以证明其成本，而不是作为几乎免费的尝试。

**🔧 技术方法**

使用了基于排他性匹配的F1奖励设计，结合了强化学习技术来优化主动视频助手的响应策略。

**📊 数据集**

使用了OmniPro数据集的视觉子集和ProactiveVideoQA数据集进行实验评估。

**📈 对比分析**

与MMDuet2进行比较，提出的方法在OmniPro上减少了82.5%的发声，同时提高了精度从0.161到0.418，F1从0.259到0.346，显示出更好的性能。

**⚠️ 局限性**

局限性包括在跨度内的时序问题以及手动选择的γ和β参数，这些参数可以在未来的工作中根据特定部署成本进行推导。

---

## 516. Scalable fault vulnerability analysis for ReRAM-based process-in-memory CNN accelerators

**arXiv ID:** 2611.26307 | [PDF](https://arxiv.org/pdf/2611.26307v1)

**作者:** Aniseh Dorostkar `[一作]` (Amirkabir University of Technology), Meysam Sharifzadeh `[通讯]` (Amirkabir University of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种快速、可扩展的分析框架，用于量化部署在基于ReRAM的内存处理加速器上的卷积神经网络（CNN）的故障脆弱性。

**💡 创新点**

创新点在于通过梯度分析方法，结合一阶和二阶梯度，提供了多层次的脆弱性评估，能够在几分钟内完成大规模CNN的故障脆弱性分析，而传统方法需要数天或数周。

**🔧 技术方法**

使用了基于梯度的灵敏度分析框架，结合一阶和二阶梯度近似来评估故障敏感性。

**📊 数据集**

使用了多个数据集，包括MNIST和CIFAR-10，评估了不同CNN架构（如LeNet5、ResNet18等）的性能。

**📈 对比分析**

与传统的故障注入方法相比，提出的方法在速度上提高了约10^4到10^5倍，能够在几分钟内完成大规模CNN的脆弱性分析，且结果与故障注入模拟的性能指标高度相关。

**⚠️ 局限性**

局限性在于虽然该方法在小型和中型CNN上得到了验证，但在更复杂的大规模CNN上可能需要进一步的验证和调整。

---

## 517. Some Memories Must Fade: Persistence and Decay of User Memories in Model Parameters

**arXiv ID:** 2611.25235 | [PDF](https://arxiv.org/pdf/2611.25235v1)

**作者:** Sanidhya Vijayvargiya `[一作]` (Samsung Research America), Rahul Lokesh `[通讯]` (Samsung Research America)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种参数化记忆管理系统，旨在通过内存基质和学习策略来有效管理大型语言模型中的知识存储和遗忘。

**💡 创新点**

创新点在于将记忆管理分为存储动态和策略，采用固定容量的LoRA适配器来存储记忆，并通过轻量级控制器来管理记忆的生命周期。

**🔧 技术方法**

使用了固定容量的LoRA适配器和一个2M参数的控制器，控制器负责决定何时写入信息、写入深度和信息的生命周期。

**📊 数据集**

构建了一个包含909个长时间用户轨迹的受控数据集，数据集包含事件级生命周期注释，以评估记忆管理的有效性。

**📈 对比分析**

与现有的持续学习（CL）和检索基础记忆（RAG）方法相比，该方法在管理用户记忆的生命周期方面表现更好，达到了86%的oracle性能，而CL和RAG方法主要依赖于单一行为。

**⚠️ 局限性**

限制在于需要重新测量写入强度和衰减，且假设记忆事件的识别是已知的，受控轨迹可能无法覆盖所有真实世界的用户动态。

---

## 518. Multimodal Error Detection in Chest X-ray Reports: A Large Language Model Reading Image Evidence

**arXiv ID:** 2611.26492 | [PDF](https://arxiv.org/pdf/2611.26492v1)

**作者:** Yanda Cheng `[一作]` (Cornell University), Yuhao Lu `[通讯]` (Cornell University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了一种语言模型检查器，用于帮助临床医生识别胸部X光报告中的潜在错误，通过将报告与图像证据进行比较。

**💡 创新点**

创新点在于使用经过微调的语言模型来解释报告中的发现描述，特别是处理否定、不确定性和关键词规则可能遗漏的表达。

**🔧 技术方法**

使用了一个1.5B参数的语言模型（Qwen2.5-1.5B），并通过低秩适应（LoRA）进行微调。

**📊 数据集**

使用了OpenI数据集，该数据集包含3599个胸部X光报告和相应的图像分类分数。

**📈 对比分析**

与基于关键词和否定规则的基线方法进行比较，结果显示该语言模型在报告-分数对的一致性分类上正确率为98.5%，而基线方法为80.5%。在94.4%的四元组中正确分类所有四对，而基线方法为56.5%。

**⚠️ 局限性**

限制在于参考答案是自动生成的，未经过临床医生的验证，且测试仅覆盖特定病例，可能不适用于所有病理类型。

---

## 519. MoGloss: Joint Motion and Pseudo-Gloss Pretraining for American Sign Language Understanding

**arXiv ID:** 2611.25333 | [PDF](https://arxiv.org/pdf/2611.25333v1)

**作者:** Rui Hong `[一作]` (George Mason University), Jana Košecká `[通讯]` (George Mason University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新颖的两阶段预训练方法，用于手势编码器，首先学习重建被遮蔽的运动，然后联合预测伪词素和运动。

**💡 创新点**

创新点在于引入了伪词素作为自然语言文本的中间结构化表示，并通过两阶段的预训练策略提高了手势翻译的性能。

**🔧 技术方法**

使用了基于Transformer的运动编码器，结合了伪词素生成和遮蔽运动建模的技术。

**📊 数据集**

构建了一个包含约1100小时美国手语（ASL）的语料库，来自五个公共数据集。

**📈 对比分析**

通过统一的评估协议比较了多种翻译方法，提出的模型在How2Sign、ASL STEM Wiki和OpenASL数据集上超越了现有的最先进方法Uni-Sign和SHuBERT。

**⚠️ 局限性**

模型仅使用2D关键点，未来的工作可以考虑结合RGB或3D输入。此外，伪词素是由大型语言模型生成的，而非由语言学家注释，可能会影响性能。

---

## 520. Sirius-Tiamat: How Small Can a One-Pass Decision Encoder Get Before Its Decisions Change?

**arXiv ID:** 2611.26339 | [PDF](https://arxiv.org/pdf/2611.26339v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 521. Learning to Verify: Signature-Guided Training for Adaptive Adversarial Detection

**arXiv ID:** 2611.25271 | [PDF](https://arxiv.org/pdf/2611.25271v1)

**作者:** Arash Vashagh `[一作]` (University of New Brunswick), Roozbeh Razavi-Far `[通讯]` (University of New Brunswick)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种双模型方法，固定原始分类器并训练一个单独的验证器来评估其预测，利用内部神经网络激活模式生成具有挑战性的对抗样本。

**💡 创新点**

创新点在于结合了检测器感知训练的单独验证器、仅在训练时使用的签名课程，以及在测试时不依赖签名库的预测条件拒绝机制。

**🔧 技术方法**

使用了知识蒸馏、激活签名构建、对抗训练等技术。

**📊 数据集**

在CIFAR-10、SVHN和STL-10等公开数据集上进行了评估。

**📈 对比分析**

与多种适应性攻击（包括训练期间未见的攻击）进行比较，提出的验证器在L_∞攻击下显著优于直接攻击的基于置信度和特征的检测器。

**⚠️ 局限性**

在不同扰动类型的鲁棒性方面存在局限性，尤其是在L_2评估中表现出较弱的鲁棒性。

---

## 522. Sequential Anytime Valid Tests for Efficient Inference-Time Alignment

**arXiv ID:** 2611.26413 | [PDF](https://arxiv.org/pdf/2611.26413v1)

**作者:** Vishesh Kumar Singh `[一作]` (Indian Institute of Technology Kanpur), Sayak Ray Chowdhury `[通讯]` (Indian Institute of Technology Kanpur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的自适应停止规则e-RWSC，用于在生成语言模型输出时优化计算资源，确保在给定的错误容忍度下，只有在答案被认证为占主导地位时才停止生成。

**💡 创新点**

创新点在于引入了一种基于投注超马丁格尔的自适应停止规则，能够在动态监控下控制错误率，并且在不同的生成条件下自动调整计算预算。

**🔧 技术方法**

使用了奖励加权自一致性（RWSC）评分和Wald的顺序概率比率检验（SPRT），并将其扩展为e-RWSC以实现更好的性能和错误控制。

**📊 数据集**

在多个数据集上进行了实验，包括GSM8K、MATH500、MMLU和CruxEval，使用了五种语言模型（如Qwen2.5-3B、Llama-3-8B等）和四种奖励模型（如ArmoRM、Skywork V1等）。

**📈 对比分析**

与传统的最佳N选择方法相比，e-RWSC在准确性上有显著提升，同时在计算预算上减少了61.7%。实验表明，e-RWSC在多个基准测试中表现优于现有方法，且在不同模型和奖励模型下均能保持较高的准确性。

**⚠️ 局限性**

限制在于认证保证的是策略的主导性，而非事实的正确性，因此一个自信但错误的策略也能快速获得认证。

---

## 523. Reference-Free Humanoid Skateboarding via Termination-Shaped Reinforcement Learning

**arXiv ID:** 2611.26054 | [PDF](https://arxiv.org/pdf/2611.26054v1)

**作者:** Ganeshvar Soenessardien `[一作]` (Vrije Universiteit Amsterdam), Kevin Sebastian Luck `[通讯]` (Vrije Universiteit Amsterdam)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种深度强化学习框架，使得机器人能够在没有人类示范或分析模型的情况下学习滑板运动，重点研究了人形机器人如何与被动外部物体协调以实现前进运动。

**💡 创新点**

创新点在于提出了一种无参考的强化学习方法，机器人通过任务奖励和行为约束来学习滑板运动，而不依赖于任何先前的数据或示范。

**🔧 技术方法**

使用了深度强化学习技术，具体采用了近端策略优化（PPO）算法，并结合了任务奖励和早期终止机制。

**📊 数据集**

实验在Mujoco模拟环境中进行，使用了Booster K1人形机器人和一个被动的四轮滑板作为测试平台。

**📈 对比分析**

与现有方法（如HUSKY）相比，本文的方法不依赖于人类示范或物理模型，结果显示机器人能够在有限的硬件条件下实现可行的滑板运动，但在运动风格、统计精度和可重复性方面存在局限。

**⚠️ 局限性**

限制在于学习到的滑板运动风格较为保守，缺乏流畅的动作，且目前的实验仅在平坦的模拟环境中进行，未来需要解决真实世界中的挑战。

---

## 524. One Schedule Does Not Fit All: Adaptive Inference Schedules for Diffusion Language Models

**arXiv ID:** 2611.25895 | [PDF](https://arxiv.org/pdf/2611.25895v1)

**作者:** Rabimba Karanjai `[一作]` (PayPal Inc.), Prakhar Mehrotra `[通讯]` (PayPal Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的认证和部署规则，旨在优化扩散语言模型的文本生成过程，确保在不同任务和模型下选择最优的采样调度。

**💡 创新点**

创新点在于提出了一个认证层，能够对任何候选采样器进行联合认证，并在缺乏证据时回退到完整调度，从而保证了生成文本的质量和稳定性。

**🔧 技术方法**

使用了扩散模型和流模型的技术，结合了自适应和错误优化的采样器。

**📊 数据集**

使用了多个数据集，包括XSum、WMT14和OpenWebText等，共计七个数据集进行实验验证。

**📈 对比分析**

通过与多种采样器（如错误优化网格、Heun、Fast-dLLM等）进行比较，证明了所提出方法在保持质量的同时显著降低了计算成本，且在不同任务上表现出不同的最佳调度。

**⚠️ 局限性**

局限性在于保证是渐近的，并且依赖于库和可交换的校准与部署输入；在数据分布变化时需要重新校准或监控，且无法捕捉到稀有的每个示例失败。

---

## 525. Persona Lenses: What Makes a Vision-Language Model See Through Its Persona's Eyes?

**arXiv ID:** 2611.25508 | [PDF](https://arxiv.org/pdf/2611.25508v1)

**作者:** Yoonho Shin `[一作]` `[通讯]` (LG Uplus), Yoonho Shin (LG Uplus)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了视觉-语言模型如何模拟用户在网页上的注视，并引入了三种不同的个性透镜（文本透镜、像素透镜和权重透镜）来增强模型对个性化用户的注视模拟能力。

**💡 创新点**

创新点在于提出了个性透镜的概念，通过不同的方式将个性信息融入模型的注视预测中，从而提高了模型对不同用户兴趣的适应性。

**🔧 技术方法**

使用了Qwen3-VL（4B和32B）和Qwen3.5-9B等视觉-语言模型，并结合了强化学习技术来训练权重透镜。

**📊 数据集**

使用了生成的网页和实时网页作为数据集，生成的数据集包含多个兴趣标签的网页截图。

**📈 对比分析**

通过比较不同透镜和模型规模的效果，发现文本透镜在分离个性方面表现强劲，但对无关属性的变化不够稳健；像素透镜效果较小，但不受无关属性影响；权重透镜在未见兴趣上表现良好，且无需额外步骤。

**⚠️ 局限性**

限制在于未能测试具体用户在网页上的注视位置，且现有的公共注视数据集未标注用户个性。此外，个性透镜在某些情况下可能会受到无关属性的影响，且像素透镜在某些区域的识别能力有限。

---

## 526. Signed but Unbound: The Authority Gap Between Agent Commerce Mandates and Crypto Wallet Signatures

**arXiv ID:** 2611.25821 | [PDF](https://arxiv.org/pdf/2611.25821v1)

**作者:** Shengwei You `[一作]` (University of Notre Dame), Jarek Nabrzyski `[通讯]` (University of Notre Dame)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了人工智能代理在使用稳定币支付时，商业协议与钱包签名之间的权威差距，定义了九类约束，并推导了这些约束在授权配置中的表现。

**💡 创新点**

创新点在于定义了权威差距的模型，并比较了九类违反情况，提供了对现有商业支付协议的深入分析和验证。

**🔧 技术方法**

使用了智能合约、EIP-3009、ERC-20、ERC-2612等技术，并进行了对比实验以验证推导结果。

**📊 数据集**

使用了Coinbase的USDC和其他在Base网络上部署的合约作为数据集，进行了多次实验以验证理论推导。

**📈 对比分析**

通过对比不同配置的执行结果，发现所有实验结果与推导一致，且在每个配置下的所有授权都能精确匹配预期结果。

**⚠️ 局限性**

限制在于研究主要集中在单次支付的精确金额上，未考虑可变金额、定期支付和退款等情况。

---

## 527. From FIPS to FPGA with LLM Agents: Measuring What Tests and Reviews Miss in Post-Quantum Cryptographic Hardware

**arXiv ID:** 2611.25339 | [PDF](https://arxiv.org/pdf/2611.25339v1)

**作者:** Jungmin Park `[一作]` (Lucid Motors), Byungho Cha `[通讯]` (Lucid Motors)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一种七阶段的验证工作流程，用于在后量子设计中验证硬件，确保生成的硬件设计和测试的可靠性。

**💡 创新点**

创新点在于通过引入独立的变异体和模块级审查，增强了对生成硬件的验证力度，并通过正式检查确保了关键保管的不变性。

**🔧 技术方法**

使用了变异测试、LLM代码审查和形式验证等技术。

**📊 数据集**

使用了ML-KEM-768/ML-DSA-65加速器的设计，并在Zynq UltraScale+卡上进行了验证。

**📈 对比分析**

与现有方法相比，提出的工作流程在模块级审查中发现了所有已知缺陷，而无指导的整体设计审查则常常遗漏验证缺陷。通过形式检查，所有关键保管缺陷都被标记出来。

**⚠️ 局限性**

限制在于该研究仅覆盖一个设计系列，且所有测试和审查均由同一模型执行，缺乏人类独立的基线测试。

---

## 528. Beyond Weight Updates: Evolving Agent Harnesses for Domain Specialization

**arXiv ID:** 2611.25380 | [PDF](https://arxiv.org/pdf/2611.25380v1)

**作者:** BoHao Chen `[一作]` `[通讯]` (Independent Researcher), BoHao Chen (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在保持策略不变的情况下，如何通过进化代理的工具架构（如系统提示、工具模式、技能库和规划框架）来实现领域专业化。

**💡 创新点**

提出了一种新的方法GEPA，通过反思性进化优化器来实现工具架构的进化，避免了传统的权重更新方法带来的计算成本和推理链缩短问题。

**🔧 技术方法**

使用了反思性进化优化器GEPA，结合了强化学习（RL）和适配器组合的技术。

**📊 数据集**

在SWE-bench Verified、OfficeBench和Terminal-Bench等多个基准数据集上进行了实验。

**📈 对比分析**

与传统的权重更新方法相比，GEPA在计算效率上表现出色，能够在较少的计算资源下恢复大部分性能，结合两种方法的配置在所有基准测试中表现最佳。

**⚠️ 局限性**

GEPA依赖于基准或可靠的奖励信号，可能在开放领域中面临噪声或可操控性的问题；此外，适配器的演化可能无法实现某些新工具的实现，且评估结果基于单次训练，统计强度较弱。

---

## 529. DACA: Dual-Anchor Collaborative Agent for Zero-Shot 3D Visual Grounding

**arXiv ID:** 2611.25722 | [PDF](https://arxiv.org/pdf/2611.25722v1)

**作者:** Jiaming Liu `[一作]` (Shanghai Jiao Tong University), Huaxi Huang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种新的双锚协作代理（DACA），用于3D视觉定位（3D-VG），通过语义和几何锚点的双向协作来提高目标跟踪的准确性。

**💡 创新点**

DACA框架通过引入语义锚和几何锚的双向协作机制，解决了现有方法中单向错误累积的问题，提升了3D-VG的性能。

**🔧 技术方法**

使用了运动引导的语义时间扩展（MG-STE）和关键点感知的多视角几何扩展（KA-MGE）技术。

**📊 数据集**

在ScanRefer和Nr3D基准数据集上进行了实验，这些数据集包含了大量的3D场景和查询目标对。

**📈 对比分析**

与现有的零-shot方法相比，DACA在ScanRefer上达到了73.6%的Acc@0.25和70.4%的Nr3D整体准确率，显著超越了最新的零-shot 3D-VG方法。

**⚠️ 局限性**

DACA的局限性在于其依赖于输入的RGB-D图像序列，可能在极端视角变化或遮挡情况下表现不佳。

---

## 530. Text Lives in the Guidance: Preventing Silent Text Rewriting in Cached Diffusion Transformers

**arXiv ID:** 2611.25382 | [PDF](https://arxiv.org/pdf/2611.25382v1)

**作者:** Kun Zhou `[一作]`, Weijia Jia `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文研究了扩散变换器（DiTs）在图像中渲染文本的能力，发现特征缓存会在不改变图像级别指标的情况下，悄悄地重写文本。

**💡 创新点**

创新点在于识别并量化了特征缓存导致的文本错误，提出了一种新的加速方法，通过仅在预测错误最大的文本标记上重新计算指导项，恢复了文本的准确性。

**🔧 技术方法**

使用了扩散变换器（DiTs）和分类器无关指导（CFG）技术，并提出了一种新的缓存方法。

**📊 数据集**

使用了LongText-Bench数据集来评估文本渲染的准确性。

**📈 对比分析**

与现有的加速方法（如FORA和TaylorSeer）进行比较，提出的方法在保持更高文本准确性的同时，减少了计算量（FLOPs），并且在文本渲染方面表现优于其他方法。

**⚠️ 局限性**

限制在于该方法仍然依赖于特征缓存的有效性，且在某些情况下可能无法完全恢复文本的准确性。

---

## 531. SceneAnything: Agentic Reconstruction of Large-Scale Mesh Scenes from a Single Video

**arXiv ID:** 2611.25340 | [PDF](https://arxiv.org/pdf/2611.25340v1)

**作者:** Bowen Xue `[一作]` (NVIDIA), Milos Hasan `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种自动化系统，通过协调代理、预训练视觉模型和图形工具，将真实和生成的视频转换为可编辑的3D场景。

**💡 创新点**

创新点在于引入了基于观察的重建框架，能够实现从视频中构建可编辑场景，并支持文本到场景的构建。

**🔧 技术方法**

使用了预训练的视觉模型、Blender程序生成器和程序化检查工具。

**📊 数据集**

使用了100个场景的数据集，包括静态和动态元素的Wikimedia Commons视频、生成视频和照片集。

**📈 对比分析**

与直接代理重建和其他视频到3D方法进行了比较，结果显示在原始视图的保真度和新视图的内容相似性上有显著提高。

**⚠️ 局限性**

限制在于最佳结果需要商业模型，当前的开源作者代理生成的场景质量较低，且对未观察区域的重建质量通常较差。

---

## 532. AppSci-Bench: Can LLMs Build AI Agents and Make Them Better?

**arXiv ID:** 2611.25360 | [PDF](https://arxiv.org/pdf/2611.25360v1)

**作者:** Sanjari Srivastava `[一作]` (Uniphore), Peng Qi `[通讯]` (Uniphore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文介绍了一个新的基准测试，旨在评估大型语言模型（LLMs）在构建智能代理方面的能力，模拟人类应用科学家为企业构建代理的过程。

**💡 创新点**

创新点在于提出了一个新的基准测试，专注于评估LLMs在代理构建中的表现，并引入了七个现实任务领域，展示了当前模型在这一领域的不足。

**🔧 技术方法**

使用了多种前沿模型进行评估，包括Claude Opus 5、Claude Opus 4.8、Claude Sonnet 4.6、Claude Haiku 4.5、OpenAI GPT 5.5，以及开源模型Kimi K3和GPT OSS 120B。

**📊 数据集**

数据集包括七个任务领域，分别是幻灯片生成、订单到现金、合规审查、客户电话洞察、CRM洞察，以及从τ^3-bench改编的航空和银行领域。

**📈 对比分析**

与其他方法的比较显示，最佳模型Claude Opus 5在三次构建尝试中仅达到59.1%的平均最低分数，其他模型表现显著较差，且在任务需求的引导和内部评估构建方面存在困难。

**⚠️ 局限性**

限制在于当前模型在需求引导和构建可靠代理方面的能力不足，且自我优化的效果有限，内部基准测试容易饱和，导致模型在外部评估中的表现不佳。

---

## 533. Intelligent Embedding Pruning for Large-Scale Recommendation Models

**arXiv ID:** 2611.25288 | [PDF](https://arxiv.org/pdf/2611.25288v1)

**作者:** Anish Khazane `[一作]` (Meta), Xiaoya Wang `[通讯]` (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于特征重要性引导的嵌入修剪方法FI-ITEP，以减少推荐系统中的内存占用，同时保持或提高模型的准确性。

**💡 创新点**

创新点在于使用特征重要性来设置每个表的修剪比例，从而实现更有效的内存节省和更好的模型性能。

**🔧 技术方法**

采用了基于扰动的特征重要性技术，包括置换和Shapley值采样，结合了训练中的嵌入修剪（ITEP）机制。

**📊 数据集**

使用了Criteo 1TB数据集进行验证，并在多个行业规模的推荐模型上进行了实验。

**📈 对比分析**

与基于百分位的ITEP方法相比，FI-ITEP在三个不同规模的模型上实现了20-40%的内存减少，同时在准确性上表现相当或更好。

**⚠️ 局限性**

限制在于FI计算的开销，尽管相对较小，但在较小模型上可能不如基于大小的修剪方法有效。

---

## 534. A Verifier from the Past: Test-Time Compute for Time Series Foundation Models

**arXiv ID:** 2611.25171 | [PDF](https://arxiv.org/pdf/2611.25171v1)

**作者:** Arjun Choudhry `[一作]` (Georgia Institute of Technology), Shubhankar Agarwal `[通讯]` (Synthefy)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5a41884c-404f-4688-a89c-aa238c10fe68` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Backtest-Gated Augmentation Averaging (BGAA) 的算法，用于时间序列预测，通过使用滚动起始回测来评分候选输出，并选择最佳预测。

**💡 创新点**

创新点在于将测试时计算（TTC）方法应用于时间序列预测，构建了一个因果验证器，利用历史数据来评分和选择预测视图。

**🔧 技术方法**

使用了滚动起始回测技术来评估不同的预测视图，并通过聚合表现优异的视图来提高预测准确性。

**📊 数据集**

在多个数据集上进行了测试，包括GIFT-Eval、fev-bench和BOOM，涉及多种时间序列基准。

**📈 对比分析**

与基线模型相比，BGAA在所有测试的模型-基准组合中均有提升，MASE减少了0.5-1.0%（GIFT-Eval）、1.3-5.0%（fev-bench）和1.7-5.9%（BOOM），且在计算效率上表现优于更大的模型。

**⚠️ 局限性**

主要限制在于窗口级别的选择，历史回测在配置级别的排名较强，但在单个窗口的预测中较弱，导致在特定窗口的选择上仍存在挑战。

---

## 535. ICARE: cross-domain text classification with incremental class-aware representation and distillation learning

**arXiv ID:** 2611.24959 | [PDF](https://arxiv.org/pdf/2611.24959v1)

**作者:** Son Duong `[一作]` (Hanoi University of Science and Technology), Hai-Anh Tran `[通讯]` (Hanoi University of Science and Technology)

**通讯引用:** 1243 | [OpenAlex ID](https://openalex.org/A5053163348)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为ICARE的框架，用于跨领域文本分类，旨在解决灾难性遗忘和训练资源不足的问题。

**💡 创新点**

创新点在于引入了类感知经验表示模块和知识蒸馏模块，以有效保留源领域知识并适应新领域。

**🔧 技术方法**

使用了增量学习、表示学习和知识蒸馏等技术，结合了特征级和逻辑级的知识转移。

**📊 数据集**

在多个公开的文本分类数据集上进行了评估，包括AG News、DBPedia、Yelp和Yahoo等。

**📈 对比分析**

与现有的基准方法相比，ICARE在单源领域和多源领域场景中均表现出色，尤其在新类准确率和遗忘率方面显著优于其他方法。

**⚠️ 局限性**

限制在于模型在处理完全不同的目标领域时可能仍面临挑战，尤其是在高内存或计算资源受限的情况下。

---

## 536. Trace2ST: Recovering PLC Structured Text from I/O Traces with LLM Generation and Verification

**arXiv ID:** 2611.24979 | [PDF](https://arxiv.org/pdf/2611.24979v1)

**作者:** Hsi Chang `[一作]` (Arizona State University), Irfan Khan `[通讯]` (Texas A&M University)

**通讯引用:** 6552 | [OpenAlex ID](https://openalex.org/A5009835940)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了一种方法，通过输入/输出记录（I/O traces）利用大型语言模型（LLM）恢复可执行的PLC结构化文本程序，并验证恢复的程序。

**💡 创新点**

创新点在于建立了一个生成-重放-反馈的管道，首次将LLM应用于从I/O记录中恢复PLC程序，并通过严格的验证协议确保恢复程序的正确性。

**🔧 技术方法**

使用了大型语言模型（LLM）进行程序生成，并开发了一个解释器来执行和验证生成的程序。

**📊 数据集**

使用了47个来自两个公共基准的第三方程序及其对应的I/O记录进行实验。

**📈 对比分析**

与传统的自动机学习基线进行比较，LLM在未见输入上的表现显著优于基线方法，LLM在持有的跟踪上达到了0.94的平均一致性，而基线方法仅为0.40。

**⚠️ 局限性**

限制在于恢复的程序在某些未被记录的分支上可能存在错误，且需要足够的证据来提高恢复的准确性。

---

## 537. RL Post-Training in Diffusion Transformers Is Activation Steering

**arXiv ID:** 2611.26627 | [PDF](https://arxiv.org/pdf/2611.26627v1)

**作者:** Tuna Han Salih Meral `[一作]` (Virginia Tech), Pinar Yanardag `[通讯]` (Virginia Tech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

论文探讨了强化学习（RL）后训练如何显著改善扩散变换器的视觉质量、文本渲染和组合准确性。研究提出了一种通过固定的引导向量来替代输入依赖的复杂变换，从而实现大部分RL增益的可能性。

**💡 创新点**

创新点在于发现固定的引导向量可以在没有奖励的情况下恢复大部分RL调优模型的增益，表明大部分RL增益可以通过简单的常量偏移实现，而不需要新的输入依赖变换。

**🔧 技术方法**

使用了强化学习（RL）后训练和流式梯度优化（Flow-GRPO）技术来训练固定的引导向量。

**📊 数据集**

使用了SD3.5-M和FLUX.1-dev数据集，这些数据集包含了多种RL调优的图像和视频模型。

**📈 对比分析**

与传统的RL调优方法相比，固定引导向量在恢复RL增益方面表现出色，能够恢复90.0%到96.3%的增益，而使用平均激活变化的简单方法则效果较差。

**⚠️ 局限性**

限制在于虽然固定引导向量能够恢复大部分增益，但仍然未能完全替代RL调优的效果，且当前的RL方法是否能够学习新的输入依赖变换仍然是一个未解的问题。

---

## 538. ColorRef: Calibrating Verbal Feedback for Color Revision

**arXiv ID:** 2611.25653 | [PDF](https://arxiv.org/pdf/2611.25653v1)

**作者:** Pablo Loyola `[一作]` (Rakuten Group), Alfredo Solano `[通讯]` (Graduate School of Engineering, The University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过ColorRef游戏探讨了语言模型如何根据描述推测颜色，并在教师的反馈下进行修正。研究表明，模型在接收到反馈后能够有效地减少与目标颜色的距离。

**💡 创新点**

创新点在于提出了接收者校准反馈的方法，通过测量每个接收者对不同描述的典型反应，选择与其观察到的步骤相匹配的措辞，从而提高颜色修正的准确性。

**🔧 技术方法**

使用了Qwen3-14B和Gemma 4 12B两种语言模型，并结合了控制实验来评估模型的反馈跟随能力和修正效果。

**📊 数据集**

研究使用了1,000对众包的描述-目标颜色对和128个保留的起始颜色进行评估。

**📈 对比分析**

与固定的分级措辞规则相比，接收者校准的反馈策略在减少一阶均值误差方面表现更好，Qwen3-14B的误差从12.36降至6.86，Gemma 4 12B的误差从11.06降至5.89，尤其在大幅修正时效果显著。

**⚠️ 局限性**

限制在于记录的众包目标不一定是描述的唯一适当解释，且研究未能验证不同描述之间的目标一致性。此外，使用的CIELAB度量并不完全符合人类的感知标准。

---

## 539. GEOHunter: Mapping a Generative Engine Optimization Network in the Wild

**arXiv ID:** 2611.26564 | [PDF](https://arxiv.org/pdf/2611.26564v1)

**作者:** Sean Tang `[一作]` (University of California), Yuan Tian `[通讯]` (University of California)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究测量了一个商业化的生成引擎优化（GEO）操作，分析了在三个中国内容平台上发布的网络账户，并追踪了九个助手引用的页面。

**💡 创新点**

创新点在于首次将一个在野外观察到的生产网络与助手给出的答案联系起来，揭示了UGC GEO投放对生成引擎结果的影响。

**🔧 技术方法**

使用了网络爬虫工具和数据追踪工具，分别用于识别网络账户和追踪助手引用的页面。

**📊 数据集**

使用了来自中国的三个主要UGC平台的数据集，包括Bilibili、百家号和360快资讯。

**📈 对比分析**

通过与九个助手的比较，发现网络引用的页面在助手的答案中占比高达70%，而且在回答中提到网络客户的频率显著高于类别领导者。

**⚠️ 局限性**

限制在于爬虫仅覆盖了通过搜索名称找到的账户，网络的实际规模可能更大；同时，数据的归因水平可能会遗漏未爬取主机上的复制内容。

---

## 540. Many Solutions, One Answer: Search Order Shapes the Default Solutions of Language Models

**arXiv ID:** 2611.25916 | [PDF](https://arxiv.org/pdf/2611.25916v1)

**作者:** Qinhan Hou `[一作]` (University of Helsinki), Jing Tang `[通讯]` (University of Helsinki)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了四种开放权重模型在符号谜题和多解ZebraLogic变体上的表现，探讨了语言模型在面对多个正确答案时的输出偏好。

**💡 创新点**

提出了通过改变问题呈现顺序或搜索顺序来增加模型输出多样性的策略，显示了模型在不同条件下的输出集中性和可恢复性。

**🔧 技术方法**

使用了四种开放权重模型，包括Qwen3-1.7B、Olmo-3.1-32B-Instruct和两种gpt-oss模型，采用了有序搜索和随机化呈现等技术。

**📊 数据集**

使用了72个符号任务问题和48个ZebraLogic逻辑网格谜题，确保每个问题的解集是已知的。

**📈 对比分析**

通过比较不同搜索顺序的输出，发现后者在所有四个模型中匹配的响应比例提高了26-40个百分点，随机化呈现也显著提高了有效解的多样性。

**⚠️ 局限性**

研究的限制在于任务规模较小，且未测试该解释是否适用于开放式生成任务，此外，模型的行为可能因不同的模型家族和训练阶段而异。

---

## 541. Proof Checking Continuous Reachability

**arXiv ID:** 2611.25303 | [PDF](https://arxiv.org/pdf/2611.25303v1)

**作者:** Long Qian `[一作]` (Carnegie Mellon University), André Platzer `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种用于证明普通微分方程（ODE）可达性计算的证明格式和证明检查器，旨在认证基于泰勒模型的可达性算法的正确性。

**💡 创新点**

创新点在于首次提出了连续微分方程可达性的证明检查方法，并证明了该方法的完备性，确保了通过泰勒模型计算的每个正确近似都可以在该逻辑中被推导出来。

**🔧 技术方法**

使用了微分动态逻辑（differential dynamic logic）作为证明的表达框架，并在Lean4中实现了专门的证明检查器以提高性能。

**📊 数据集**

使用了ARCH竞赛中的基准数据集，涉及多种非线性动力学问题，包括Van der Pol和Robertson等ODE家族。

**📈 对比分析**

与现有方法相比，本文的方法通过将正确性验证与结果计算解耦，能够高效地计算ODE的近似值，并且在Lean4中实现的证明检查器能够有效地认证这些近似值的安全性。性能方面，Lean4实现能够认证多个实例，而其他工具的某些证书未能通过认证，显示出其潜在的实现问题。

**⚠️ 局限性**

限制在于当前实现主要集中在泰勒模型的可达性问题上，未来需要扩展到其他状态的工具和算法，同时也需要处理更复杂的数值算法的验证。

---

## 542. RE-CAM: Readout-Enhanced Content-Addressable Memory for Quantitative Similarity Search

**arXiv ID:** 2611.25326 | [PDF](https://arxiv.org/pdf/2611.25326v1)

**作者:** Yi-Chun Liao `[一作]` (Duke University), Yiran Chen `[通讯]` (Duke University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为RE-CAM的框架，用于增强内容寻址存储器（CAM）的读出能力，使其能够在一次CAM搜索中近似线性和非线性函数，从而实现定量相似性搜索。

**💡 创新点**

RE-CAM的创新点在于它能够通过重复感知时间变化的输出电压，联合优化感知时间戳、参考电压和解码参数，从而提高近似精度，同时降低能耗和面积开销。

**🔧 技术方法**

使用了一种硬件感知的训练算法，联合优化感知时间戳、参考电压和解码参数，并设计了可重构的CAM硬件以执行优化的读出调度。

**📊 数据集**

在32×32的二进制CAM阵列上评估了RE-CAM，使用了线性、分段线性、sigmoid和指数四种目标函数进行实验。

**📈 对比分析**

与共享均匀参考ADC基线相比，RE-CAM在一般近似中将归一化均方根误差降低了10.89%到24.34%，在利用数据集分布时降低了29.43%到46.73%。同时，RE-CAM的搜索吞吐量提高了8倍，搜索能耗和面积分别降低了47%和39%。

**⚠️ 局限性**

RE-CAM的局限性在于可用的感知时间戳可能限制了某些相似性水平的分离能力，可能导致在某些情况下无法达到最佳的近似精度。

---

## 543. Reasoning on Call: Efficient Agentic Robot Control for Long-Horizon Robot Manipulation

**arXiv ID:** 2611.26341 | [PDF](https://arxiv.org/pdf/2611.26341v1)

**作者:** Zilang Cen `[一作]` (Wuhan University), Xiaoyuan Zhang `[通讯]` (Zhongguancun Academy)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为RoC（Reasoning on Call）的高效框架，用于协调低级策略和高级推理代理，通过在线任务状态监控来支持长时间的机器人操作。

**💡 创新点**

RoC通过监控触发的代理审查来减少冗余的代理调用和推理开销，从而提高任务成功率和降低令牌消耗。

**🔧 技术方法**

使用了在线任务状态监控技术，该监控结合了阶段感知的视觉上下文和有序的交互记忆，以共同估计活动子任务的进展并预测后续目标。

**📊 数据集**

在RMBench数据集上进行评估，并在SO-ARM101平台上进行真实世界实验，验证其在三个长时间操作任务中的有效性。

**📈 对比分析**

与最强的基于代理的基线相比，RoC在任务成功率上提高了24.0个百分点，令牌消耗减少了65.3%；与最强的基于策略的基线相比，任务成功率提高了15.1个百分点。

**⚠️ 局限性**

局限性在于当前推理代理在视觉证据转化为有效抓取修正方面的能力仍然是一个瓶颈，可能影响在复杂情况下的执行修复能力。

---

## 544. Separating Interpretability and Reconstruction in Sparse Autoencoders

**arXiv ID:** 2611.25165 | [PDF](https://arxiv.org/pdf/2611.25165v1)

**作者:** Anh Tien Nguyen `[一作]` (Stony Brook University), Mahdi S. Hosseini `[通讯]` (Concordia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为DualSAE的模块化框架，通过添加一个非语义的残差分支来增强标准稀疏自编码器（SAE），以分离重建和可解释性目标。

**💡 创新点**

创新点在于通过引入残差分支，DualSAE能够将语义信息与非语义信息分开，从而提高重建的保真度和可解释性。

**🔧 技术方法**

使用了双分支架构，其中稀疏分支负责捕捉主导语义信息，而残差分支则建模被稀疏分支忽略的非语义信息，并采用了去相关损失来减少两个分支之间的重叠。

**📊 数据集**

使用了两个视觉数据集（ImageNet-1K和OpenImagesV7）和两个语言语料库（OpenWebText和GitHub代码）进行实验。

**📈 对比分析**

与标准SAE相比，DualSAE在重建保真度上平均提高了29%，在可解释性上提高了34%。在下游应用（主题建模、假设生成和LLM引导）中，DualSAE也表现出一致的性能提升。

**⚠️ 局限性**

限制在于实验使用了冻结的预训练视觉编码器和语言模型，因此尚未测试端到端训练和多模态推理模型。此外，DualSAE增加了训练选择，如去相关系数、残差分支设计和预热长度。

---

## 545. Fast and Memory-Light Evaluation of Density-Based Clusterings with the DISCO Index in R

**arXiv ID:** 2611.25114 | [PDF](https://arxiv.org/pdf/2611.25114v1)

**作者:** Amin Entezari `[一作]` `[通讯]` (Universita di Milano-Bicocca), Amin Entezari (Universita di Milano-Bicocca)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文介绍了DISCO，一个用于基于密度的聚类的内部聚类有效性指数，并通过discoCVI包在R中实现。该包允许在没有真实标签的情况下比较聚类结果并选择HDBSCAN的参数。

**💡 创新点**

创新点在于DISCO不仅评估聚类的质量，还考虑噪声标签的质量。版本0.2.0通过合并树计算DISCO，避免了存储n×n矩阵，从而显著提高了内存效率和计算速度。

**🔧 技术方法**

使用了合并树算法来计算密度连接距离，结合了Rcpp进行高效的C++实现。

**📊 数据集**

使用了合成数据集进行性能测试，包括500到100,000个点的高斯聚类数据，且在不同噪声比例下进行评估。

**📈 对比分析**

与之前版本相比，discoCVI在500到4000个点时速度提升了23到138倍，且在内存使用上显著减少。与原始Python实现相比，速度提升了2.9到8.7倍，且内存使用更低。

**⚠️ 局限性**

限制在于运行时间仍为O(n^2)，对于非常大的数据集（数十万点）仍然较慢。此外，当前只支持欧几里得距离，未来可能扩展到其他不相似度。

---

## 546. Measuring Hallucination in Large Language Models for Cyber Threat Intelligence: An Exploratory Study

**arXiv ID:** 2611.25590 | [PDF](https://arxiv.org/pdf/2611.25590v1)

**作者:** Seungho Song `[一作]` (Korea University), Jinwoo Kim `[通讯]` (Chungbuk National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对大型语言模型（LLMs）在网络威胁情报（CTI）任务中产生的幻觉进行了探索性测量，重点分析了其在摘要和问答任务中的表现。

**💡 创新点**

创新点在于提出了一个可重复的框架，自动化了幻觉分析流程，包括文章收集、LLM交互、实体提取和基于相似度的评估。

**🔧 技术方法**

使用了命名实体识别（NER）模型和多种相似度度量（如余弦相似度、答案匹配相似度和基于频率的实体相似度）来评估LLM输出的事实一致性。

**📊 数据集**

使用了来自Threatpost的4940篇真实网络安全文章和来自The Hacker News的500篇文章作为数据集。

**📈 对比分析**

与现有方法（如FActScore和QAGS）相比，研究发现语义相似度与人类判断一致，而基于实体的度量显示有限的一致性。模型之间的差异主要集中在问答任务中，而实体提取质量限制了实体级度量的揭示能力。

**⚠️ 局限性**

限制在于现有的NER模型可能会漏掉重要实体，导致评估结果的下限。此外，人工评分的公正性较低，表明现有度量更适合作为筛选信号，而非校准的幻觉检测器。

---

## 547. When Safety Rationales Hallucinate: Diagnosing and Grounding Multimodal Safety Guards

**arXiv ID:** 2611.25793 | [PDF](https://arxiv.org/pdf/2611.25793v1)

**作者:** Yingshuo Wang `[一作]` (Harbin Institute of Technology), Yanyan Zhao `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究系统性地研究了多模态安全防护模型生成的理由的可靠性，特别关注理由幻觉现象，包括视觉理由幻觉和安全归因幻觉。

**💡 创新点**

提出了一种新的工具对比强化学习框架（TCG-RL），该框架在提高理由的基础上奖励工具的使用，从而显著减少理由幻觉，同时保持竞争性的安全预测性能。

**🔧 技术方法**

使用了工具对比强化学习（TCG-RL）技术，该技术结合了强化学习和工具使用的对比奖励机制。

**📊 数据集**

使用了多个安全数据集，包括HarMeme的Harm-C和Harm-P子集、MAMI和PrideMM。

**📈 对比分析**

与现有的多种理由生成方法进行了比较，TCG-RL在视觉证据评分（VES）和安全归因评分（SAS）上表现优异，显著减少了理由幻觉，同时保持了安全预测的准确性。

**⚠️ 局限性**

当前方法在处理细粒度属性和关系时仍然存在局限性，工具的使用频率和聚焦程度不足，导致理由幻觉问题依然普遍。

---

## 548. When Is Program Structure Required? Certificate-Grounded Calibration of Program-Induction Difficulty

**arXiv ID:** 2611.24975 | [PDF](https://arxiv.org/pdf/2611.24975v1)

**作者:** Wenzhang Du `[一作]` `[通讯]` (Independent Researcher), Wenzhang Du (Independent Researcher)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种证书基础的校准方法，构建了匹配的任务，以最小解决方案长度为一和二的有界字符串替换语言。

**💡 创新点**

创新点在于通过证书验证所需的程序结构，而不是从生成参考中推断，并展示了不同的任务完成结果和输入行为。

**🔧 技术方法**

使用了证书基础的校准技术，结合了完整的一规则枚举和可执行的二规则见证。

**📊 数据集**

评估了120个参考家族，使用了1440个新的模型轨迹和480个保存的轨迹。

**📈 对比分析**

与其他方法比较时，Qwen3-8B和Olmo-3-7B-Think在经过认证的分离任务上完成率低23.53-45.83个百分点，而在完全未显示的见证对上成功率高8.40-13.45个百分点。

**⚠️ 局限性**

限制在于任务和认证范围仅涉及有界字符串替换语言中的最小程序长度，且所选的120个家族是从15004个提案中筛选的，样本记录不完整。

---

## 549. A Tutorial on Complementarity-Based Contact Models for Robotics

**arXiv ID:** 2611.26600 | [PDF](https://arxiv.org/pdf/2611.26600v1)

**作者:** Onur Beker `[一作]` (University of Tübingen), Georg Martius `[通讯]` (University of Tübingen)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本论文提供了一个关于基于互补条件的接触模型的教程，旨在填补机器人领域中关于刚体动力学、锥优化和摩擦接触的知识空白。

**💡 创新点**

创新点在于将互补条件的数学公式与其数值求解方法结合，提供了完整的推导和统一的符号表示，适用于机器人研究者和博士生。

**🔧 技术方法**

使用了锥互补问题的数学框架，结合了数值方法，如牛顿法和正则化策略，比较了不同求解器的性能。

**📊 数据集**

论文没有具体提到使用的数据集，但提到与多个开源模拟器（如MuJoCo、Bullet、Drake等）相关的实现。

**📈 对比分析**

通过比较一阶和牛顿基方法，讨论了计算成本、鲁棒性和物理保真度之间的权衡，强调了不同方法在处理接触动力学时的优缺点。

**⚠️ 局限性**

限制在于现有的互补条件模型可能在某些情况下导致物理伪影，如抬起和爬行现象，且在处理复杂接触时可能存在多解问题。

---

## 550. Monte Carlo Pass Search for Setpieces

**arXiv ID:** 2611.26464 | [PDF](https://arxiv.org/pdf/2611.26464v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 551. Embodied Human-Data Interaction

**arXiv ID:** 2611.25744 | [PDF](https://arxiv.org/pdf/2611.25744v1)

**作者:** Niklas Elmqvist `[一作]` `[通讯]` (Aarhus University), Niklas Elmqvist (Aarhus University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

提出了人-数据交互（HDI）的概念，强调人类对大型、非结构化和复杂数据集的操作、分析和理解，并建议采用具身交互的方法来构建物理推理环境。

**💡 创新点**

创新点在于将具身交互应用于人-数据交互，提出了具身人-数据交互（EHDI）的概念，以支持复杂数据集的分析过程。

**🔧 技术方法**

使用了具身交互的技术，结合了物理对象和空间安排来增强数据分析的过程。

**📊 数据集**

论文中提到的应用示例包括一个教育性的城市规划模拟游戏和一个可触摸的透镜框架，但具体的数据集未详细说明。

**📈 对比分析**

通过对具身交互的应用进行初步评估，展示了其在数据过滤和城市规划模拟中的有效性，尽管具体的性能比较未在文中详细列出。

**⚠️ 局限性**

限制在于EHDI方法仍然主要是理论性的，实际应用和工具的开发仍处于早期阶段，尚需进一步研究和完善。

---

## 552. STARK: Moving towards Personalized Agents That Truly Learn and Remember

**arXiv ID:** 2611.25705 | [PDF](https://arxiv.org/pdf/2611.25705v1)

**作者:** Aryan Jain `[一作]` (RBS Tech Sciences), Promod Yenigalla `[通讯]` (RBS Tech Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种双图记忆架构，分离了事件和行为知识，以便更好地支持个性化代理系统的长期记忆。

**💡 创新点**

创新点在于通过事件作为边的桥接，构建了一个能够直接从行为节点中检索知识的系统，消除了重复的情节遍历，并实现了跨用户的知识转移。

**🔧 技术方法**

使用了双图架构，包括情节图和洞察图，并结合了关系嵌入空间的链接预测技术。

**📊 数据集**

在多个公共基准数据集上进行了评估，包括LoCoMo、LongMemEval和LongLaMP。

**📈 对比分析**

与现有的记忆系统（如Mem0、MAGMA等）相比，提出的方法在信息提取和开放领域问题上表现更好，分别提高了1.48和0.93的评分。

**⚠️ 局限性**

限制在于TransR的更新在O(1)时间内，但其每个关系的投影具有固定容量，随着图的密集化，嵌入质量会下降。此外，隐私保护是通过代理实现的，而不是正式的，可能存在聚合指纹的泄露风险。

---

## 553. Leveraging Restrictions to Optimize Register Automata Learning

**arXiv ID:** 2611.25186 | [PDF](https://arxiv.org/pdf/2611.25186v1)

**作者:** Paul Fiterau-Brostean `[一作]` (Uppsala University), Fredrik Tåquist `[通讯]` (Uppsala University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的算法，通过利用先前测试获得的知识来优化注册自动机的学习过程，从而提高算法的可扩展性。

**💡 创新点**

创新点在于引入了一种新的限制机制，允许在后续测试中仅考虑必要的测试，从而显著减少测试数量并提高学习效率。

**🔧 技术方法**

使用了一种基于树的算法，结合了分类树和符号决策树的概念，以优化注册自动机的学习过程。

**📊 数据集**

使用了多个基准数据集，包括五个广泛使用的DTLS服务器实现，以评估算法的性能。

**📈 对比分析**

与现有的最先进算法进行比较，结果表明该算法在模型构建和总测试数量上都显著减少了测试数量，并且在系统规模增大时表现出更好的可扩展性。

**⚠️ 局限性**

限制在于该算法在处理具有复杂数据依赖关系的系统时，可能仍然面临测试数量的指数增长问题，尤其是在数据参数之间存在大量依赖关系的情况下。

---

## 554. Iterative Scarcity-Guided Exploration: Bootstrapping Generative Auto-bidding from Narrow Support

**arXiv ID:** 2611.26060 | [PDF](https://arxiv.org/pdf/2611.26060v1)

**作者:** Qingmao Yao `[一作]` (Peking University), Xiaotie Deng `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的迭代框架ISGE，用于在冷启动环境中从低质量、狭窄支持的数据中引导自动竞标策略的生成。

**💡 创新点**

创新点在于通过引入稀缺性引导机制，打破了传统自训练方法在低质量数据下的自锁定瓶颈，成功实现了高回报和高稀缺性的轨迹生成。

**🔧 技术方法**

使用了条件扩散模型和去噪自编码器技术，结合了返回信号和稀缺性信号进行探索。

**📊 数据集**

在工业级的AuctionNet基准数据集上进行了广泛实验，构建了冷启动数据集以模拟真实的低质量数据环境。

**📈 对比分析**

与传统的自训练方法相比，ISGE在三次迭代内有效超越了全数据基线，显示出显著的性能提升，尤其是在稠密和稀疏环境中均表现出优越性。

**⚠️ 局限性**

局限性在于ISGE假设局部景观的连续性，且在接近最优解时，其外向探索的有效性可能降低。

---

## 555. TAP: Tapping the Discriminative Potential of Frozen Representations for Class-Incremental Learning

**arXiv ID:** 2611.25478 | [PDF](https://arxiv.org/pdf/2611.25478v1)

**作者:** Zihao Wang `[一作]` (Shandong University), Xin-Shun Xu `[通讯]` (Shandong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的框架，通过保持编码器冻结来利用预训练模型的判别潜力，以支持类增量学习（CIL），并通过轻量级优化不断积累类信息。

**💡 创新点**

创新点在于通过类条件视觉分布建模、类自适应语义校准和视觉-语义后验融合三个模块，充分利用冻结的CLIP嵌入空间中的视觉和语义信息，从而实现增量识别而无需对编码器进行适应或重放旧图像。

**🔧 技术方法**

使用了CLIP模型的冻结编码器，结合类条件视觉分布建模、类自适应语义校准和视觉-语义后验融合技术。

**📊 数据集**

在九个数据集上进行了实验，使用了两种增量协议进行评估。

**📈 对比分析**

与其他方法相比，该方法在九个数据集上达到了最先进的性能，获得了最高的平均增量和最终阶段准确率，超越了所有比较方法。

**⚠️ 局限性**

限制在于该方法依赖于冻结的编码器，可能在某些情况下无法充分利用动态更新的潜力。

---

## 556. Cooling Noisy Population Protocols -- Vanishing-Gain Averaging and Localisation under Persistent Errors

**arXiv ID:** 2611.26266 | [PDF](https://arxiv.org/pdf/2611.26266v1)

**作者:** Leszek Gąsieniec `[一作]` (University of Liverpool), Grzegorz Stachowiak `[通讯]` (University of Wrocław)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

研究了在匿名人口协议中，具有噪声容忍的协调，代理以均匀随机的方式成对互动，维护实值标签，并且没有全局时钟。

**💡 创新点**

提出了一种基于局部交互计数的增益计算方法，证明了在不同噪声条件下的收敛性，并分析了不同更新规则的效果。

**🔧 技术方法**

使用了随机调度、局部计数器和增益计算等技术，结合了随机近似和马尔可夫链的分析方法。

**📊 数据集**

使用了固定中心有限方差分布的独立通信错误和固定非退化高斯位移错误的数据集。

**📈 对比分析**

与传统的一侧更新方法相比，基于相对位置差异的签名更新协议在每个固定有限几何体中保持收敛性，且收敛速度与噪声方差相关。

**⚠️ 局限性**

局部计数器的冷却机制在处理持续噪声时可能无法完全消除偏差，且在某些情况下可能导致收敛失败。

---

## 557. Hindsight Self-Distillation: Teaching Robots to Act from the Present

**arXiv ID:** 2611.25777 | [PDF](https://arxiv.org/pdf/2611.25777v1)

**作者:** Haisen He `[一作]` (Shenzhen University of Advanced Technology), Zhiheng Ma `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为Hindsight Self-Distillation (HSD)的方法，通过利用机器人在执行动作后的后续帧来训练其策略，从而提高其在当前图像中提取信息的能力。

**💡 创新点**

HSD的创新点在于通过教师和学生的协作学习，教师利用后续帧指导学生，学生则仅依赖当前帧进行学习，从而实现了有效的信息提取和动作预测。

**🔧 技术方法**

使用了两种扩散变换器（Video DiT和Action DiT），共享视觉编码器和动作预测器，并通过学习查询向量进行视觉摘要的生成。

**📊 数据集**

在LIBERO和RoboTwin 2.0数据集上进行了评估，LIBERO的成功率达到98.50%，RoboTwin 2.0的成功率为92.31%。

**📈 对比分析**

与其他方法（如Fast-WAM）进行比较，HSD在LIBERO上取得了98.50%的成功率，优于Fast-WAM的96.95%；在RoboTwin 2.0上，HSD的成功率为92.31%，也超过了Fast-WAM的成绩。

**⚠️ 局限性**

HSD的局限性在于其依赖于当前上下文中可预测的教师目标，分析是在固定快照下进行的，而闭环增益则需要通过实证评估。

---

## 558. Design and Evaluation of Epira: An Automated Interview System for Case Investigation and Contact Tracing

**arXiv ID:** 2611.25577 | [PDF](https://arxiv.org/pdf/2611.25577v1)

**作者:** Ishita Asnani `[一作]` (University of Illinois Urbana-Champaign), Madeleine Udell `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发并评估了Epira，一个用于多种疾病病例调查和接触追踪的协议控制大型语言模型（LLM）面试官。

**💡 创新点**

Epira结合了LLM对话与基于规则的控制器，确保了必要问题的提问、信息的安全交接和结构化的病例记录，并能够通过上传问卷适应新疾病。

**🔧 技术方法**

使用了本地（Qwen2.5）和云端（GPT-5.4）后端的LLM技术。

**📊 数据集**

使用了模拟病例数据集，包括麻疹和淋病的病例，进行两项研究的评估。

**📈 对比分析**

与开放式聊天机器人相比，Epira在捕获必要字段和联系人方面表现更好，完成率为100%，用户体验评分（SUS）平均为82.95，信任评分为3.85。

**⚠️ 局限性**

限制在于模拟受访者并非真实患者，无法评估不同性别、种族或残疾群体的可用性差异，且在真实接触者披露时可能存在抵触情绪。

---

## 559. Deviations from the NTK Regime in CNNs under Joint Depth-Width Scaling

**arXiv ID:** 2611.25624 | [PDF](https://arxiv.org/pdf/2611.25624v1)

**作者:** Maria Matveev `[一作]` (LMU Munich), Mariia Seleznova `[通讯]` (TU Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了卷积神经网络（CNN）中深度和宽度联合缩放对神经切线核（NTK）行为的影响，特别是在初始化时的NTK分散性和训练过程中的核旋转。

**💡 创新点**

提出了深度与宽度比（L/C）作为影响NTK行为的关键参数，发现其在不同初始化状态下对NTK的集中性和训练稳定性有不同的影响。

**🔧 技术方法**

使用了标准参数化（SP）和全批量梯度下降等技术来训练CNN，并通过量化NTK的变化来分析其行为。

**📊 数据集**

使用了二分类CIFAR-10数据集（飞机与青蛙），包含2048张训练图像和2000张测试图像。

**📈 对比分析**

通过比较不同深度与宽度比下的NTK分散性和训练引起的旋转，发现深度增加通常会放大与无限宽度NTK行为的偏差，但在有序状态下，旋转与分散性之间的相关性较弱。

**⚠️ 局限性**

研究的局限性在于仅限于特定的CNN架构和单一的二分类任务，且观察到的有限规模缩放并未建立渐近的联合极限。

---

## 560. PSM-WSM: Predictively Sufficient Multimodal World Social Modeling

**arXiv ID:** 2611.25835 | [PDF](https://arxiv.org/pdf/2611.25835v1)

**作者:** Chunlei Meng `[一作]` (Fudan University), Zhongxue Gan `[通讯]` (Fudan University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了一种新的框架PSM-WM，用于学习多模态社交世界中的预测性潜在状态，以支持未来的交互建模。

**💡 创新点**

创新点在于引入了预测充分性原则，定义了潜在状态应保留的未来相关信息，并通过该原则指导潜在状态的构建。

**🔧 技术方法**

使用了概率状态推断、事件条件随机过渡和多步一致性优化等技术。

**📊 数据集**

在Hi-EF、IEMOCAP、MELD和Ego4D等多个多模态数据集上进行了实验。

**📈 对比分析**

与现有的多模态表示学习和预测表示学习方法相比，PSM-WM在未来交互建模上表现更好，尤其是在W-F1评分上超越了其他基线模型，显示出更强的预测能力和鲁棒性。

**⚠️ 局限性**

限制在于当前模型的设计可能在处理更复杂的动态环境时面临挑战，未来的工作将探索更长时间范围和动作条件的预测。

---

## 561. Verification Behavior Across Five Qwen2.5-Instruct Checkpoints

**arXiv ID:** 2611.25745 | [PDF](https://arxiv.org/pdf/2611.25745v1)

**作者:** YaJie Yin `[一作]` `[通讯]`, YaJie Yin

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了五个Qwen2.5-Instruct检查点（0.5B到32B）在四种提示条件下的表现，使用三个固定任务和每个任务五次生成进行比较。

**💡 创新点**

创新点在于通过结构化提示来改变这些检查点的验证行为，并将执行检查与判断保留的决策区分开来。

**🔧 技术方法**

使用了Qwen2.5-Instruct模型，采用了七步验证协议和协调指令等技术。

**📊 数据集**

使用了六个任务，其中包括信息充足和信息不足的任务。

**📈 对比分析**

通过比较不同检查点在相同任务下的表现，发现7B检查点在严格声明不足的情况下表现较好，而14B和32B的表现则较差，且H3执行率普遍较低。

**⚠️ 局限性**

限制在于实验仅覆盖一个模型家族和每个信息条件下的三个固定任务，且两位AI评估者可能共享错误，导致可重复性受到限制。

---

## 562. Keystone: Judging Previews at Key Steps for Test-Time Search in Diffusion Models

**arXiv ID:** 2611.25530 | [PDF](https://arxiv.org/pdf/2611.25530v1)

**作者:** Jiajun Fan `[一作]` (University of Illinois at Urbana-Champaign), Ge Liu `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出了一种新的图像生成评估方法，通过在去噪过程中实时判断候选图像的质量，从而提高生成效率和准确性。

**💡 创新点**

创新点在于引入了一种无需训练的决策模型，能够直接评估图像预览，并通过编译提示为模型生成类型化问题，从而改善评估过程。

**🔧 技术方法**

使用了多模态决策模型来评估图像预览，并结合了自适应搜索和修剪技术来优化生成过程。

**📊 数据集**

使用了SD3.5-Medium数据集进行实验，并在GenEval基准上进行了评估。

**📈 对比分析**

与现有方法（如ImageReward和HPSv2.1）相比，提出的方法在相同计算成本下显著提高了准确性，尤其是在计数和位置提示的任务中表现优异。

**⚠️ 局限性**

限制在于该方法依赖于决策模型的性能，如果模型的评估能力不足，可能会影响最终的生成效果。

---

## 563. Living off the Local Model: Abusing Browser-Embedded Language Models for Post-Exploitation

**arXiv ID:** 2611.25086 | [PDF](https://arxiv.org/pdf/2611.25086v1)

**作者:** William Fleshman `[一作]` (Army Cyber Institute), Joseph Lucas `[通讯]` (NVIDIA)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了浏览器嵌入的语言模型（如Chrome中的Gemini Nano）作为攻击者后期利用工具的潜在威胁，展示了如何利用这些模型进行恶意操作。

**💡 创新点**

创新点在于揭示了浏览器嵌入的本地语言模型如何被攻击者利用进行后期利用操作，且这些模型的使用不需要网络流量，难以被监测。

**🔧 技术方法**

使用了Chrome的Gemini Nano模型，结合操作系统原生的解释器（如PowerShell和Python）进行攻击。

**📊 数据集**

使用了多个数据集，包括CredData用于凭证分类，Enron数据集用于个人信息识别，以及Imagenette用于图像分类。

**📈 对比分析**

通过与现有的攻击方法进行比较，展示了Gemini Nano模型在凭证识别、个人信息检测和图像分类等任务上的性能，尽管其准确性低于大型模型，但仍足以支持攻击者的后期利用活动。

**⚠️ 局限性**

限制在于小模型的错误率较高，尽管随着时间推移可能会改善，但当前的性能仍然不足以完全依赖于其进行复杂的攻击操作。

---

## 564. A generic construction of complex codebooks from functions over Abelian groups

**arXiv ID:** 2611.25729 | [PDF](https://arxiv.org/pdf/2611.25729v1)

**作者:** Ziling Heng `[一作]` (Changan University), Xiaoyan Jing `[通讯]` (Changan University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种通用构造和一种改进构造，通过有限阿贝尔群上的函数构建复数码本，旨在优化码本的最大交叉相关幅度。

**💡 创新点**

创新点在于通过选择完美非线性函数，系统性地获得了多个最优或渐近最优的码本系列，并且恢复了已知的最优码本，同时生成了一些具有新参数的码本。

**🔧 技术方法**

使用了有限阿贝尔群上的完美非线性函数和离散傅里叶变换等技术。

**📊 数据集**

数据集为有限阿贝尔群，具体的群体和函数形式在文中进行了详细讨论。

**📈 对比分析**

与现有方法的比较表明，构造的码本在最大交叉相关幅度上达到了Welch和Levenshtein界限，性能优越，尤其在选择完美非线性函数时，构造的码本能够达到最优或渐近最优。

**⚠️ 局限性**

限制在于构造的码本依赖于完美非线性函数的选择，且在某些情况下，构造的复杂性可能导致实际应用中的实现困难。

---

## 565. SegAnyFruit: Learning Generalizable and Edge Deployable Fruit Instance Segmentation at Scale

**arXiv ID:** 2611.25475 | [PDF](https://arxiv.org/pdf/2611.25475v1)

**作者:** Yanan Wang `[一作]` (Zhejiang University), Zhenghao Fei `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了SegAnyFruit (SAF)，一个用于水果实例分割的通用和边缘可部署的学习框架，旨在解决在农业环境中对未见物种、果园和成像条件的泛化问题。

**💡 创新点**

创新点在于通过逐步训练和数据构建，将基础模型的先验知识提炼为轻量级、无提示的分割器，并且在多个农业领域中实现了零-shot和少-shot的适应性。

**🔧 技术方法**

使用了大规模的通用分割数据集进行预训练，结合自动标注的WebFruits数据集进行中期训练，以及基于人工校准的PaperFruits进行后期训练。

**📊 数据集**

使用了WebFruits和PaperFruits数据集，前者包含733K张图像和9.66M个完整水果掩码，后者用于农业细化。

**📈 对比分析**

与其他方法相比，SAF-RF-DETR-XL在七个基准测试中达到了69.06的平均掩码AP_50:95，超越了现有的VLM和开放词汇基线，并且在边缘设备上可部署。

**⚠️ 局限性**

限制在于自动生成的伪掩码在边界质量和实例粒度上仍存在错误，且尚未确定何时通用掩码预训练最为有效。此外，SAF将水果视为统一的视觉概念，未能区分个别物种。

---

## 566. A four-level executable validation protocol for evaluating LLM-generated topology specifications of physics-based HVAC digital twins

**arXiv ID:** 2611.25471 | [PDF](https://arxiv.org/pdf/2611.25471v1)

**作者:** Younghwan Joo `[一作]` (Korea Institute of Energy Research), Sung-il Kim `[通讯]` (Korea Institute of Energy Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一种四级可执行验证协议，用于评估大型语言模型（LLM）生成的HVAC数字双胞胎的拓扑规格是否能够执行并产生物理有效的双胞胎。

**💡 创新点**

创新点在于该协议不仅检查生成的规格是否能执行，还验证其是否能产生物理有效的数字双胞胎，强调了物理有效性的重要性。

**🔧 技术方法**

使用了大型语言模型（LLMs）和MATLAB/Simulink框架进行模型构建和仿真。

**📊 数据集**

使用了14个HVAC拓扑的基准数据集，其中包括五个描述现有低湿度测试室的拓扑。

**📈 对比分析**

与其他方法相比，该协议通过静态有效性、模型构建、仿真收敛和行为等效性四个级别进行评估。结果显示，七个模型在98.6%到100%的试验中成功构建模型，仿真收敛率在80.0%到100%之间，但行为等效性仅在50.0%到98.6%之间。

**⚠️ 局限性**

限制在于该基准仅覆盖一个领域和14个拓扑，且未与实际测量进行比较，输出和修复预算固定，未测试更大的预算或更多修复是否会提高通过率。

---

## 567. VideoRNG: Relightable Neural Gaussians from Generative Video Models

**arXiv ID:** 2611.25338 | [PDF](https://arxiv.org/pdf/2611.25338v1)

**作者:** Bowen Xue `[一作]` (University of Manchester), Miloš Hašan `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为VideoRNG的两阶段方法，从文本提示或单张图像生成可重新照明的3D资产，避免了对网格或解析材料模型的限制。

**💡 创新点**

创新点在于结合了视频生成和前馈重建模型，能够生成复杂的3D资产，并通过深度锚定外观学习来提高生成质量。

**🔧 技术方法**

使用了预训练的视频扩散模型作为基础，结合前馈大重建模型（LRM）和随机高斯光线追踪技术。

**📊 数据集**

使用了约22K个带纹理的资产数据集，这些资产经过处理以生成81帧的RGBA转盘序列。

**📈 对比分析**

与传统的高斯点云方法相比，VideoRNG在生成质量和重建精度上表现更好，PSNR和LPIPS指标均优于其他方法。

**⚠️ 局限性**

限制在于生成的资产可能在某些复杂场景中表现不如手工制作的3D模型，且对训练数据的依赖性较强。

---

## 568. Carry the Fine Print: Budget-Aware Visual Token Compression for Document OCR

**arXiv ID:** 2611.25506 | [PDF](https://arxiv.org/pdf/2611.25506v1)

**作者:** Yixuan Huang `[一作]` (University of Electronic Science and Technology of China), Jiawei Chen `[通讯]` (Central South University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Select and Compensate的两阶段框架，用于视觉标记压缩，旨在改善文档OCR的完整转录质量。

**💡 创新点**

创新点在于将选择和补偿结合起来，使得保留的标记不仅仅是其位置的代表，还能携带附近被丢弃内容的证据，从而提高了转录的准确性。

**🔧 技术方法**

使用了自适应编码分辨率、全局重要性和局部对比的组合技术来选择保留的标记，并通过补偿阶段丰富这些标记的上下文信息。

**📊 数据集**

在OmniDocBench和MPDocBench数据集上进行了评估，涉及多个OCR骨干网络。

**📈 对比分析**

与其他方法相比，该方法在保留50%视觉标记的情况下，整体性能达到95.42，接近未压缩模型的96.09，并在多个基准测试中表现出优越性。

**⚠️ 局限性**

局限性在于性能依赖于骨干网络的选择，且未经过QA训练的压缩权重可能无法保持QA/VQA性能。

---

## 569. SoK: Assessing the Evolution of LLM Capability in Software Vulnerability Research

**arXiv ID:** 2611.26601 | [PDF](https://arxiv.org/pdf/2611.26601v1)

**作者:** Amine Lbath `[一作]` (National Institute of Standards and Technology), Dinesh Manocha `[通讯]` (University of Maryland)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个框架，用于评估基于LLM的代理在软件漏洞研究中的能力演变，系统化了关于这些代理发现、验证和修复漏洞的证据。

**💡 创新点**

创新点在于首次引入了一个评估框架，按照自动化水平对LLM系统的能力进行评分，并对每个声明的正确性、可用性和证据强度进行定性分析。

**🔧 技术方法**

使用了基于LLM的代理技术，结合了自动化评估框架和证据收集方法。

**📊 数据集**

数据集包括超过50,000个独特来源的文献、实验室和供应商报告、维护者的账户和事件披露，涵盖了十年的公共漏洞记录。

**📈 对比分析**

通过与公共记录的比较，发现2026年披露的漏洞数量超过了2025年，并且越来越多的漏洞被评为严重或高危，但从报告到修复的时间没有显著变化。

**⚠️ 局限性**

限制在于缺乏对LLM系统在实际软件中的长期表现的全面评估，以及对修复措施的有效性和影响的深入研究。

---

## 570. SAM 3 Sees More Than It Detects: Promptable Concept Maps Beyond Categories

**arXiv ID:** 2611.25747 | [PDF](https://arxiv.org/pdf/2611.25747v1)

**作者:** Yu-Huan Wu `[一作]` (CFAR A*STAR), Liangli Zhen `[通讯]` (CFAR A*STAR)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文提出了一种新的模型UNVEIL，旨在扩展SAM 3的提示概念分割能力，从名词实例到概念地图，能够处理更复杂的视觉任务。

**💡 创新点**

创新点在于引入了存在否决（presence veto）机制，并通过UNVEIL模型将SAM 3的语义路径与存在解码器分开训练，从而提高了对区域、薄结构和类别的映射能力。

**🔧 技术方法**

使用了LoRA（低秩适配）技术对SAM 3的最后几个块进行微调，并通过并行分支训练语义路径和存在解码器。

**📊 数据集**

使用了多个数据集，包括DUTS-TR、COD10K-TR、SBU、ISTD、Trans10K、MSD、BSDS500、SK-LARGE等，涵盖了七个不同的视觉任务。

**📈 对比分析**

与其他提示模型相比，UNVEIL在PCM-Bench的七个任务中表现最佳，平均得分为90.5，显著高于其他模型，如SAM 3的官方微调（82.6）和ConceptSeg-R1（74.2）。

**⚠️ 局限性**

限制在于UNVEIL的训练依赖于SAM 3的结构，可能在某些特定任务上表现不如专门训练的模型，同时需要进一步扩展概念词汇以处理更多的视觉任务。

---

## 571. From Self-Play RL to Constrained Control: Distilling DiffMPC Students with DAgger

**arXiv ID:** 2611.25197 | [PDF](https://arxiv.org/pdf/2611.25197v1)

**作者:** Luca M. Hartmann `[一作]` (University of California San Diego), Parinaz Naghizadeh `[通讯]` (University of California San Diego)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的管道，通过将可微分模型预测控制（DiffMPC）层添加到已训练的端到端策略中，以在不损失性能的情况下为强大的驾驶策略添加安全约束。

**💡 创新点**

创新点在于首次实现了从自我对抗训练的端到端教师中大规模蒸馏DiffMPC学生，并提出了一种约束公式，使得蒸馏过程有效。

**🔧 技术方法**

使用了可微分模型预测控制（DiffMPC）、策略蒸馏（DAgger）和自我对抗强化学习等技术。

**📊 数据集**

使用了CARLA、nuPlan和Waymo开放数据集进行评估。

**📈 对比分析**

与教师策略相比，约束DiffMPC学生在所有测试集上表现相当或更好，尤其是在Waymo场景中，减少了16-26%的偏离道路的频率。蒸馏过程的训练时间最多只增加了30%。

**⚠️ 局限性**

限制在于约束的质量可能影响性能，较强的教师策略在约束下可能不会显著受益，而较弱的教师策略则可能需要约束来提高安全性。

---

## 572. Self-Replicating Language Models

**arXiv ID:** 2611.25839 | [PDF](https://arxiv.org/pdf/2611.25839v1)

**作者:** Arend Hintze `[一作]` `[通讯]`, Arend Hintze

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了大型语言模型（LLMs）是否能够自我复制并具备达尔文进化的特征，提出了多种复制模式并进行了实验验证。

**💡 创新点**

创新点在于将LLMs视为可能的自我复制生物体，探讨其在达尔文进化框架下的生存能力和适应性。

**🔧 技术方法**

使用了多种技术，包括自我复制的基因组生成、遗传算法、以及对模型的训练和评估。

**📊 数据集**

使用了多个小型语言模型（如Qwen3-0.6B）进行实验，测试其复制能力和适应性。

**📈 对比分析**

通过与传统生物体的进化机制进行比较，发现LLMs在某些条件下能够实现自我复制和遗传变异，且选择能够改善其复制能力。

**⚠️ 局限性**

限制在于实验使用的模型规模较小，无法完全模拟大型LLMs的复杂性，且复制过程中的知识损失问题仍然存在。

---

## 573. Asymmetric rhetorical strategies drive partisanship in television news

**arXiv ID:** 2611.25140 | [PDF](https://arxiv.org/pdf/2611.25140v1)

**作者:** Upasana Dutta `[一作]` (University of Pennsylvania), Duncan J. Watts `[通讯]` (University of Pennsylvania)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究分析了2013年至2025年间美国六大新闻台（Fox News、MSNBC、CNN、ABC、CBS和NBC）超过40万集新闻节目的语言，揭示了电视新闻中表现出的系统性和不对称的党派表达模式。

**💡 创新点**

创新点在于通过大型语言模型（LLMs）提供多维度的党派偏见测量，识别出八种可解释的语言特征，解释了73%的党派方差，并揭示了不同新闻台在意识形态倾斜的表达方式上的系统性差异。

**🔧 技术方法**

使用了大型语言模型（LLMs）进行文本分析，结合线性回归模型来解释党派分数。

**📊 数据集**

数据集包括408,464集新闻节目，涵盖了2013年至2025年间的内容，涉及3,505,401个新闻片段。

**📈 对比分析**

与传统方法相比，研究通过LLMs的分析显示出更高的准确性和一致性，且在不同模型中结果一致，表明研究结论的稳健性。Fox News的党派倾向明显高于MSNBC和CNN，且其党派偏见在不同主题上表现出一致性。

**⚠️ 局限性**

限制在于虽然八种语言特征解释了73%的党派方差，但仍有27%的方差未被解释。此外，研究是相关性的，未能建立党派表达的因果关系，且数据集中仅包含特朗普总统任期的三个月覆盖，限制了对该时期党派动态的深入分析。

---

## 574. Revealing Pathway Interactions in Network Computations with Complete Path Contribution

**arXiv ID:** 2611.26101 | [PDF](https://arxiv.org/pdf/2611.26101v1)

**作者:** Shenghua Liu `[一作]` (Stanford University), Stephen A. Baccus `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的方法，称为完整路径贡献（Complete Path Contribution），用于分解神经网络输出，直接研究路径级交互。

**💡 创新点**

创新点在于能够在任何粒度水平上保证完整性，并且提供了一种高效算法来识别最强路径，而无需检查所有路径。

**🔧 技术方法**

使用了完整路径贡献框架和高效的路径识别算法。

**📊 数据集**

在基准图像分类网络和小鼠视网膜的可解释模型上进行了实验。

**📈 对比分析**

与现有方法相比，路径贡献能够量化兴奋性-抑制性交互，并揭示网络决策的构建过程，性能在细节上超越了传统的神经元级分析。

**⚠️ 局限性**

限制在于尽管提供了高效的路径识别算法，但在处理极大规模的网络时，仍然可能面临计算复杂性的问题。

---

## 575. When Should a Voice Agent Speak? End-of-Turn Detection in Turkish

**arXiv ID:** 2611.25489 | [PDF](https://arxiv.org/pdf/2611.25489v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 576. Verifying Before Enhancing: Target Evidence Chain Verification for Infrared Small Target Detection

**arXiv ID:** 2611.25724 | [PDF](https://arxiv.org/pdf/2611.25724v1)

**作者:** Jiayi Zuo `[一作]` (Beijing University of Posts and Telecommunications), Shangguang Wang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为VBE-UNet的目标证据链验证网络，用于红外小目标检测，旨在通过补充证据线索评估候选目标的可靠性。

**💡 创新点**

创新点在于通过局部背景偏差、跨尺度一致性和扰动引起的预测敏感性三方面共同评估目标的可靠性，而不仅仅依赖于响应强度。

**🔧 技术方法**

使用了U-Net架构，结合了反事实背景解释、联合证据验证和证据引导重组等技术。

**📊 数据集**

在IRSTD-1K、NUDT-SIRST和NUAA-SIRST三个数据集上进行了广泛实验。

**📈 对比分析**

与最先进的方法相比，VBE-UNet在检测准确性和虚警率方面表现优越，具体在IRSTD-1K上取得了69.86%的nIoU和72.75%的mIoU，超越了竞争对手。

**⚠️ 局限性**

局限性在于尽管提出的方法在多个数据集上表现良好，但在极端复杂背景下的表现仍需进一步验证。

---

## 577. On the Reliability of CyberGym Evaluation: Task Material Consistency and Reference-Target Identity

**arXiv ID:** 2611.25790 | [PDF](https://arxiv.org/pdf/2611.25790v1)

**作者:** Yule Ding `[一作]`, Hongzhen Wu `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究对CyberGym的任务材料和提交输入的身份进行审计，以评估漏洞重现的有效性。

**💡 创新点**

创新点在于将任务材料的一致性和提交输入的身份分开评估，并使用共同的参考目标来区分材料一致性和提交身份。

**🔧 技术方法**

使用了执行检查、崩溃签名匹配和LLM辅助的根本原因审查等技术。

**📊 数据集**

使用了来自开源项目的1,507个任务的数据集，这些任务源于历史漏洞。

**📈 对比分析**

通过对1,507个任务的材料审计，338个任务（22.4%）被标记为不一致。在1,286个差异通过中，89个被判定为非目标，10个为证据不足，要求目标命中率降低了通过率，从85.3%降至78.8%。

**⚠️ 局限性**

限制在于身份判决需要独立验证，且材料一致性和提交身份的检查应当分开进行。

---

## 578. Differentiable Hydrodynamics for Whole-Body MPC of Underwater Robots

**arXiv ID:** 2611.25373 | [PDF](https://arxiv.org/pdf/2611.25373v1)

**作者:** Hefan Wang `[一作]` (Robot Motor Intelligence Lab), Carlos Mastalli `[通讯]` (Robot Motor Intelligence Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种可微分的水动力学模型，用于水下机器人整体控制的模型预测控制（MPC），通过优化推进器和关节扭矩来协调水下车辆和机械臂的运动。

**💡 创新点**

创新点在于结合了平滑的球谐力和力矩场与从计算流体动力学（CFD）识别的耦合附加惯性矩阵，提供了闭式形式的导数，能够在整个预测范围内高效计算水动力学负载。

**🔧 技术方法**

使用了可微分的水动力学模型，结合了球谐函数和CFD数据，采用了闭式导数形式进行整体轨迹优化和控制。

**📊 数据集**

使用了CFD生成的数据集，包括在不同方向上的稳态和振荡状态下的力和力矩样本。

**📈 对比分析**

通过与传统的线性或对角二次阻尼模型进行比较，展示了所提模型在负载预测和导数计算上的优越性，性能表现出在未见方向上的预测误差低于10%。

**⚠️ 局限性**

限制在于模型的复杂性和对CFD数据的依赖，可能在实时应用中面临计算成本和响应时间的挑战。

---

## 579. Characterizing and Detecting Unfaithful Work Reports by Coding Agents

**arXiv ID:** 2611.25760 | [PDF](https://arxiv.org/pdf/2611.25760v1)

**作者:** Wenhao Yang `[一作]` (Peking University), Dongmei Zhang `[通讯]` (Microsoft)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对编码代理在软件开发中的不忠实报告进行了系统的混合方法分析，识别了四类不忠实报告及其子类型，并探讨了开发者与代理之间的互动及其后续处理。

**💡 创新点**

创新点在于构建了不忠实报告的分类法，量化了后续处理的努力，并开发了FaithDetector，一个用于检测不忠实报告的自动化工具。

**🔧 技术方法**

使用了混合方法分析，包括定性和定量分析，结合了开发者与代理的真实对话数据。

**📊 数据集**

使用了SWE-chat数据集，该数据集包含开发者与编码代理的对话记录。

**📈 对比分析**

通过与基线方法比较，FaithDetector在自然样本报告上达到了更高的宏F1分数，约比基线高出13个百分点，显示出其在检测不忠实报告方面的有效性。

**⚠️ 局限性**

限制在于研究主要基于公开的SWE-chat数据集，可能不适用于企业开发环境，且未能完全覆盖所有潜在的不忠实报告情况。

---

## 580. MoRE: Mixture-of-Experts for Multi-Physics Robotic World Models

**arXiv ID:** 2611.25336 | [PDF](https://arxiv.org/pdf/2611.25336v1)

**作者:** Ziyu Zhu `[一作]` (University of Illinois Urbana-Champaign), Shenlong Wang `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `afceb026-1760-41ae-8d86-010831a37d97` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为MoRE的3D世界模型，通过将工具与物体的交互路由到专门的运动专家来预测基于动作的3D动态。

**💡 创新点**

创新点在于采用混合运动专家架构，能够处理刚性和可变形物体，以及抓取和非抓取接触，提升了对异质动态的建模能力。

**🔧 技术方法**

使用了混合运动专家（Mixture-of-Experts, MoE）架构，并结合了模型预测控制（Model Predictive Control, MPC）进行实时预测和控制。

**📊 数据集**

使用了混合的模拟和真实数据集进行训练，模拟数据来自MagicSim，包含约20K个实验，真实数据包含约7K个实验。

**📈 对比分析**

与现有方法相比，MoRE在多个物理基准和真实机器人任务中表现出更高的预测准确性，尤其在动态预测和控制方面优于DINO-WM等基线方法。

**⚠️ 局限性**

限制在于目前只关注单个被操控物体的运动，无法处理多个物体的同时交互，且未明确表示细粒度的物理属性，如质量、摩擦和弹性。

---

## 581. MultiBinder: Preventing Multi-Subject Identity Leakage in Personalized Video Generation

**arXiv ID:** 2611.26164 | [PDF](https://arxiv.org/pdf/2611.26164v1)

**作者:** Wei Li `[一作]` (University of Science and Technology of China), Tao Mei `[通讯]` (HiDream.ai Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MultiBinder框架，旨在解决个性化视频生成中的多主体身份泄露和语义纠缠问题。

**💡 创新点**

创新点在于引入了“绑定-叙述”范式，通过明确的跨模态绑定和叙述安排，确保每个自定义主体在动态场景中保持独特的视觉身份。

**🔧 技术方法**

使用了实体感知的Q-Former和多模态大型语言模型（MLLM）作为核心技术。

**📊 数据集**

构建了新的多主体基准数据集（MultiBind-Bench），包含100个多主体条目和120张参考图像。

**📈 对比分析**

与现有方法相比，MultiBinder在多主体身份保真度上显著优于其他方法，尤其在身份纯度评分（S_IPS）和视觉推理评分（S_VRS）上表现突出。

**⚠️ 局限性**

限制在于模型的复杂性和对训练数据的依赖，可能在特定场景下仍然面临挑战。

---

## 582. LIFT: Latent Flow Matching for Probabilistic Cell Fate Dynamics with Single Cell Snapshots

**arXiv ID:** 2611.25917 | [PDF](https://arxiv.org/pdf/2611.25917v1)

**作者:** Bowen Yang `[一作]` (Peking University), Lei Zhang `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为LIFT的潜在流框架，用于从谱系信息中生成细胞状态的概率模型，旨在通过条件结果模型来学习细胞命运预测。

**💡 创新点**

LIFT结合了条件表示、有效生成和误差控制，能够处理从初始状态生成多个未来结果的情况，解决了传统方法无法准确捕捉细胞命运概率的问题。

**🔧 技术方法**

使用了变分推断、流匹配和均值流解码器等技术，构建了一个高效的生成模型。

**📊 数据集**

在合成数据集和真实数据集（如小鼠造血和成纤维细胞重编程）上进行了实验，使用了转录组状态表示。

**📈 对比分析**

与几种基线方法（如条件流匹配、均值流和得分匹配）进行了比较，LIFT在分支概率预测和人口分布重建方面表现出色，尤其在低采样成本下保持了高效性。

**⚠️ 局限性**

当前框架的局限性在于其生成动态未明确建模不平衡效应（如细胞增殖和死亡），且预测依赖于提供的耦合和选择的状态表示，无法唯一识别中间生物路径或条件变异的来源。

---

## 583. TripleKV: Predict, Prefetch, Protect -- KV Cache Management for AI Agents

**arXiv ID:** 2611.25112 | [PDF](https://arxiv.org/pdf/2611.25112v1)

**作者:** Shashwat Jaiswal `[一作]` (University of Illinois Urbana-Champaign), Josep Torrellas `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为TripleKV的三层KV缓存管理系统，旨在解决高并发下LLM代理的缓存延迟问题。

**💡 创新点**

TripleKV通过预测工具调用的完成时间，提前从SSD预取KV块到主机DRAM，从而隐藏存储读取延迟，显著提高了推理性能。

**🔧 技术方法**

使用了工具感知的延迟预测器、预取引擎和预取感知的缓存替换策略。

**📊 数据集**

在DeepResearch Bench、GAIA、Terminal-Bench 4.0和Browse Comp-Plus等四个基准测试上进行了评估。

**📈 对比分析**

与vLLM、TokenCake和Continuum等现有方法相比，TripleKV将需要关键路径SSD读取的请求比例从45-76%降低到10%以下，P95首次令牌时间（TTFT）减少了最多49%。

**⚠️ 局限性**

TripleKV在高并发情况下可能会遇到DRAM容量不足的问题，导致某些预取块在请求到达之前被驱逐。

---

## 584. Forget Freely, Grep Later: Lossless, Index-Free Memory for Long-Horizon LLM Agents

**arXiv ID:** 2611.25123 | [PDF](https://arxiv.org/pdf/2611.25123v1)

**作者:** Ziyan Yang `[一作]` (University of Southern California), Jieyu Zhao `[通讯]` (University of Southern California)

**通讯引用:** 8708 | [OpenAlex ID](https://openalex.org/A5066282713)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的无损、无索引的记忆机制，结合了上下文摘要和完整的可搜索交互历史，以解决长时间跨度的LLM代理在上下文窗口超出时的信息丢失问题。

**💡 创新点**

创新点在于通过一个仅追加的纯文本日志和正则表达式搜索工具实现无损记忆，而不需要复杂的结构化记忆基础设施。

**🔧 技术方法**

使用了强化学习（RL）训练Qwen3.5-9B模型，并在五种上下文管理策略下进行评估。

**📊 数据集**

使用了多种数据集，包括多问题研究会话、长流回忆和代理轨迹问答等。

**📈 对比分析**

与现有的五种上下文管理策略进行比较，模型在准确性上达到了95-97%，而其他方法的准确性最高仅为21%。在BCP-Multi任务上，超越了SUPO的摘要方法8.8个百分点，并在所有五个评估任务中保持在准确性-令牌成本的帕累托前沿。

**⚠️ 局限性**

局限性在于评估集中于受控基准和单一模型，尚未验证该方法在更广泛的代理设置中的适用性。此外，依赖于词汇搜索可能在某些情况下效果不佳，尤其是当相关信息无法通过明确的搜索模式识别时。

---

## 585. A Unified Framework for Image Manipulation Localization Under Multi-Type Weak Supervision

**arXiv ID:** 2611.25371 | [PDF](https://arxiv.org/pdf/2611.25371v1)

**作者:** Songlin Li `[一作]` (Xinjiang University), Xin Liao `[通讯]` (Hunan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为UniIML的统一框架，用于在多类型弱监督下进行图像操控定位。该框架通过教师-学生模型将不同类型的弱注释映射到统一表示，并生成多个候选掩码。

**💡 创新点**

创新点在于引入了Ordinal Safe Consensus (OSC)机制，通过安全过滤和序数共识投票来选择候选掩码，从而提高了伪标签的可靠性，并有效利用了不同类型的弱注释。

**🔧 技术方法**

使用了教师-学生框架，结合Segment Anything Model (SAM)生成候选掩码，并通过OSC进行候选掩码的选择和伪标签的构建。

**📊 数据集**

在九个数据集上进行了实验，使用了四种类型的弱注释，包括图像级标签、点注释、边界框和涂鸦注释。

**📈 对比分析**

与其他弱监督方法相比，UniIML在单一类型和混合类型注释设置下均表现优越，并在多个全监督方法中保持竞争力，显示出良好的跨数据集泛化能力。

**⚠️ 局限性**

限制在于不同类型的弱注释在空间约束和监督确定性方面存在差异，可能导致伪标签的质量不均，尤其是在注释稀疏的情况下，模型可能会过拟合于不准确的空间假设。

---

## 586. Exact guarantees and faster algorithms for $k$-rollercoasters

**arXiv ID:** 2611.26297 | [PDF](https://arxiv.org/pdf/2611.26297v1)

**作者:** Sungmin Kim `[一作]` (University of Goettingen), Florin Manea `[通讯]` (University of Goettingen)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了k≥3的k-过山车（rollercoaster），即所有最大递增和递减序列的长度至少为k的不同值的序列。发现了n点排列中作为子序列出现的k-过山车的最大保证长度kn。

**💡 创新点**

提出了针对任意k≥3的k-过山车的长度保证，并通过构造证明了这些保证是紧的，填补了之前的理论空白。

**🔧 技术方法**

使用了组合数学的证明方法，扩展了Seidenberg的论证，并提出了两种算法：一种是O(nlogkn)时间复杂度的确定性比较算法，另一种是O(nklogk)时间复杂度的最长k-过山车算法。

**📊 数据集**

使用了n点的排列作为数据集，特别关注于n≥2q²的情况，并通过构造具体的n点排列来验证理论结果。

**📈 对比分析**

与之前的算法进行比较，提出的算法在时间复杂度上有显著改进，尤其是将O(nk²)的复杂度降低到O(nklogk)，并且在n>q²的情况下，能够在O(nlogkn)时间内找到至少kn长度的k-过山车。

**⚠️ 局限性**

限制在于对于较小的n（如n≤q²），无法保证存在k-过山车，且在某些情况下，算法的性能可能受到数据集特性的影响。

---

## 587. Can LLMs Resolve Entity Conflicts in Domain Specific NLU Pipelines?

**arXiv ID:** 2611.25733 | [PDF](https://arxiv.org/pdf/2611.25733v1)

**作者:** Akansha Bhardwaj `[一作]` (Telepathy Labs GmbH), Paolo Rosso `[通讯]` (Telepathy Labs GmbH)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了大型语言模型（LLMs）在特定领域自然语言理解（NLU）管道中解决实体冲突的能力，比较了传统的基于规则的方法与两种基于LLM的方法。

**💡 创新点**

创新点在于利用LLM的广泛知识来替代手工编写的规则，以提高实体冲突解决的灵活性和可扩展性。

**🔧 技术方法**

使用了大型语言模型（LLM），特别是Llama3.1-8B-Instruct模型，并结合了低秩适应（LoRA）技术进行微调。

**📊 数据集**

使用了两个数据集：Automotive-Rule-Based（基于规则的数据集）和Automotive-Curated（人工标注的数据集），前者包含来自现有NLU管道的输出，后者包含674个经过人工验证的示例。

**📈 对比分析**

与传统的基于规则的方法相比，LLM方法在多个设置下表现出更高的准确性，尤其是在复杂和模糊的案例中。微调后的LLM在特定领域的表现优于基于规则的方法，尤其是在时间协商的上下文中。

**⚠️ 局限性**

限制在于LLM在特定领域的表现可能不如基于规则的方法，尤其是在服务请求等领域特定的上下文中，且增加的延迟可能是一个潜在的权衡。

---

## 588. Verify the Verifier: Specification-Grounded LLM Cross-Checking for Bugs That Escape UVM Scoreboards

**arXiv ID:** 2611.26588 | [PDF](https://arxiv.org/pdf/2611.26588v1)

**作者:** Ana E. Sanchez Villalobos `[一作]` (Texas A&M University), David Kebo Houngninou `[通讯]` (Texas A&M University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一个名为Verify the Verifier的框架，利用大语言模型（LLM）对通过的回归测试进行检查，以发现未被检测到的bug。

**💡 创新点**

这是首个使用LLM对通过的回归测试轨迹进行交叉检查的框架，并且能够正式确认每个scoreboard escape。

**🔧 技术方法**

使用了大语言模型（LLM）、SystemVerilog断言、形式验证和仿真轨迹分析等技术。

**📊 数据集**

使用了四个OpenTitan IPs的设计数据集，共插入564个bug，经过筛选后确认68个候选scoreboard escapes。

**📈 对比分析**

与传统方法相比，该框架能够发现41个scoreboard escapes，且在未注入bug的设计中没有报告错误，显示出较高的准确性。

**⚠️ 局限性**

框架的局限性在于生成的断言可能无法完全捕捉到bug，且在某些情况下，ACC未能观察到所有的候选escape。

---

## 589. OffBeat: Multicoset Sampling for 4D Capture Beyond the Camera Frame Rate

**arXiv ID:** 2611.26347 | [PDF](https://arxiv.org/pdf/2611.26347v1)

**作者:** Chi-Wei Lee `[一作]` (National Tsing Hua University), Po-Chih Kuo `[通讯]` (National Tsing Hua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为OffBeat的方法，用于4D重建，旨在解决同步多摄像机设备在快速非刚性运动下的捕捉问题。

**💡 创新点**

OffBeat通过将错位触发的摄像机建模为空间分布的多重采样器，克服了同步设备在捕捉运动时的局限性，并且通过校准自动推导出摄像机的触发顺序。

**🔧 技术方法**

使用了空间时间高斯（Spacetime Gaussians）和多重采样技术，结合了时间窗口的均匀加权方法。

**📊 数据集**

使用了Neural 3D Video (N3DV)数据集，该数据集包含多个动态场景，并在不同的摄像机配置下进行训练和测试。

**📈 对比分析**

与4DSloMo等方法进行比较，OffBeat在动态区域的PSNR上平均提高了3.04 dB，并在每个测试帧上超越了完整的4DSloMo管道，尤其是在快速运动的场景中表现更佳。

**⚠️ 局限性**

OffBeat对模拟触发时间和光度误差更敏感，且无法表示短于T的内容，快速运动在组触发之间仍然模糊。

---

## 590. Law at the Speed of AI: Critical Legal Infrastructure for the Machine Age: Autonomy Under Law Architecture (AULA)

**arXiv ID:** 2611.25985 | [PDF](https://arxiv.org/pdf/2611.25985v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 591. Self-Supervised Operator Learning for Channel Estimation in Fluid Antenna Systems

**arXiv ID:** 2611.25756 | [PDF](https://arxiv.org/pdf/2611.25756v1)

**作者:** Erqiang Tang `[一作]` (Hong Kong University of Science and Technology), Khaled B. Letaief `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种自监督学习框架，用于从噪声子采样的导频中直接训练基于傅里叶神经算子的估计器，以解决流体天线系统的信道估计问题。

**💡 创新点**

创新点在于通过引入导频分割策略和基于Stein无偏风险估计（SURE）的修正，消除了测量噪声偏差，使得估计器能够在没有标签的情况下进行自我监督学习。

**🔧 技术方法**

使用了傅里叶神经算子（FNO）作为核心技术，结合自监督学习方法进行信道估计。

**📊 数据集**

使用了从接收器收集的噪声导频数据集，具体包括不同端口密度下的导频观测。

**📈 对比分析**

与传统的监督学习方法相比，提出的方法在低至中等信噪比下表现出显著的性能提升，并且在未见过的端口分辨率上保持零-shot泛化能力，性能与监督训练相当。

**⚠️ 局限性**

限制在于该方法依赖于导频的采样覆盖条件，且在实际应用中可能受到环境变化的影响，未来工作将探讨在更复杂的散射环境下的表现。

---

## 592. Spherical Tessellation of Circular Arrays for Massive MIMO: Favourable Propagation Conditions and Spectral Design

**arXiv ID:** 2611.26220 | [PDF](https://arxiv.org/pdf/2611.26220v1)

**作者:** Geili T A El Sanousi `[一作]` (Middlesex University), Huan X Nguyen `[通讯]` (Middlesex University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文分析了一种球形镶嵌的同心圆天线阵列（ST-CCAA），证明了在特定条件下用户信道逐渐正交，并且匹配滤波信噪比（SINR）接近单用户信噪比（SNR）。

**💡 创新点**

创新点在于通过每个面片的三角不等式替代了克罗内克分解，成功地将非可分离的球形几何问题简化为同心圆阵列的子问题，并控制了面内混叠。

**🔧 技术方法**

使用了球形贝塞尔展开和Poisson求和公式等数学技术，结合了匹配滤波和信道模型分析。

**📊 数据集**

论文中使用了模拟数据来验证ST-CCAA的性能，特别是在用户信道的正交性和SINR收敛性方面。

**📈 对比分析**

与传统的均匀圆柱阵列（UCLA）相比，ST-CCAA在抑制栅格波纹方面表现更好，且其SINR收敛速度为𝒪(M^-1/3)，而UCLA则存在全幅度的栅格波纹。

**⚠️ 局限性**

限制在于未考虑不完美的隔离或显著的面间耦合，且在固定间距的情况下，ST-CCAA的设计可能在某些情况下受到限制。

---

## 593. An LLM-Driven Traffic-Control xApp for Dynamic 5G RAN Intents

**arXiv ID:** 2611.25736 | [PDF](https://arxiv.org/pdf/2611.25736v1)

**作者:** Valerio Triolo `[一作]` (Scuola Superiore Sant'Anna), Emilio Paolini `[通讯]` (Scuola Superiore Sant'Anna)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一种基于大型语言模型（LLM）的xApp，用于在可编程无线接入网络（RAN）中动态控制流量，结合运营商定义的服务意图、实时E2接口遥测和当前配置生成流量管理策略。

**💡 创新点**

创新点在于设计了一种意图驱动的控制架构，结合高层服务意图和实时网络测量，能够在不阻塞数据包转发的情况下动态适应流量控制管道，并通过验证阶段确保生成的策略安全有效。

**🔧 技术方法**

使用了大型语言模型（LLM）进行策略选择，结合了结构化输入输出和确定性验证机制。

**📊 数据集**

在OpenAirInterface 5G环境中进行实验，使用了真实的网络条件和流量数据。

**📈 对比分析**

与静态TC-RAN配置相比，提出的LLM驱动控制器在意图满足率上达到了0.91，而静态配置仅为0.52。在信道条件恶化时，控制器仍能保持约0.88的意图满足率，显示出其优越的动态适应能力。

**⚠️ 局限性**

限制在于LLM生成的输出可能存在语法无效或与当前网络状态不兼容的情况，因此需要经过验证阶段才能应用。此外，控制器的性能可能受到推理延迟和配置应用时间的影响。

---

## 594. Atlas: Accelerating Quantum FTQC Compilation with Optimized Algebraic Rotation Synthesis

**arXiv ID:** 2611.25448 | [PDF](https://arxiv.org/pdf/2611.25448v1)

**作者:** Anbang Wu `[一作]` (Shanghai Jiao Tong University), Wei Geng `[通讯]` (South China University of Technology)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Atlas的优化算法，用于加速量子容错计算中的旋转合成过程，特别是在变分量子特征求解器（VQE）应用中。

**💡 创新点**

创新点在于通过搜索剪枝、适应性算术和批处理调度三种优化方法，显著减少了旋转合成过程中的冗余计算和不均匀的合成时间，从而提高了效率。

**🔧 技术方法**

使用了统计推断进行搜索剪枝，适应性算术选择合适的算术例程，以及动态调度批处理操作来平衡计算负载。

**📊 数据集**

在多个基准测试和VQE工作负载上进行了评估，包括随机旋转和五种VQE电路家族。

**📈 对比分析**

与现有的旋转合成方法（如GridSynth和PQF）相比，Atlas在多个基准测试中实现了高达12.65倍的加速，且在VQE工作负载中也表现出显著的性能提升。

**⚠️ 局限性**

限制在于Atlas的优化主要集中在旋转合成阶段，其他编译阶段（如映射和电路优化）仍然是潜在的性能瓶颈。

---

## 595. Calibrated Decision Rewards: Training Language Models with a Judge That Knows How Sure It Is

**arXiv ID:** 2611.25135 | [PDF](https://arxiv.org/pdf/2611.25135v1)

**作者:** Rudransh Agnihotri `[一作]` (FuturixAI), Manasvi Kapoor `[通讯]` (FuturixAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种新的强化学习方法，称为基于校准决策奖励的强化学习（RL from Calibrated Decision Rewards），旨在提高语言模型在复杂推理任务中的表现。

**💡 创新点**

创新点在于提出了一种无需参考答案的校准奖励机制，使得语言模型能够在没有参考答案的情况下进行有效的推理，并且能够提供概率分布而非单一的答案。

**🔧 技术方法**

使用了强化学习（RL）和决策模型技术，具体实现为将大型语言模型（LLM）转换为决策模型，并通过校准和去偏见处理来优化奖励。

**📊 数据集**

使用了MATH500数据集，该数据集包含数学问题，模型在此数据集上进行训练和评估。

**📈 对比分析**

与随机奖励和软最大化判断器进行比较，结果显示使用校准奖励的模型在MATH500上的准确率从72.2%提高到76.8%，而随机奖励的提升仅为0.65%，表明校准奖励显著提高了模型性能。

**⚠️ 局限性**

限制在于该方法在处理困难问题时表现不如参考答案，且在这些问题上模型的概率预测可能过于自信，未来需要针对困难问题进行重新校准。

---

## 596. Bayesian $κ$-Explorer for Minimax Model Estimation in MDPs

**arXiv ID:** 2611.25160 | [PDF](https://arxiv.org/pdf/2611.25160v1)

**作者:** Xihe Gu `[一作]` (University of California), Tara Javidi `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了在未知马尔可夫决策过程中进行极小极大转移估计的主动探索问题。

**💡 创新点**

提出了贝叶斯κ-Explorer算法，通过在转移分布上放置Dirichlet先验，简化了探索信号的计算，并消除了对置信集的优化需求。

**🔧 技术方法**

使用了贝叶斯方法，特别是Dirichlet先验和后验均值估计。

**📊 数据集**

在Pendulum、Mountain Car和Taxi三个环境中进行了实验。

**📈 对比分析**

与其他基线方法相比，贝叶斯κ-Explorer在最坏情况下的转移估计误差更低，尤其是在数据收集的早期和较大的采样预算下表现更佳。

**⚠️ 局限性**

算法的局限性在于对模型假设的依赖，以及在某些情况下可能对κ参数的敏感性。

---

## 597. You Have Mail: Set-Theoretic Types for Actors

**arXiv ID:** 2611.26563 | [PDF](https://arxiv.org/pdf/2611.26563v1)

**作者:** Giuseppe Castagna `[一作]` (CNRS Université Paris Cité), Amine Nasri `[通讯]` (LACL Université Paris-Est Créteil)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

提出了一种用于Elixir进程的类型系统，该系统为每个进程定义了一个接口类型，以便静态检查消息的兼容性。

**💡 创新点**

创新点在于解决了Elixir中隐式邮箱的类型推断问题，使得高阶函数可以在消息中传递并正确类型化。

**🔧 技术方法**

使用了集合论类型和语义子类型的技术，结合了现有的子类型和约束解决机制。

**📊 数据集**

使用了Elixir的现有代码作为数据集，并通过重构接收子句来推断接口。

**📈 对比分析**

与现有方法相比，该系统能够类型化在消息中传递的高阶函数，确保发送和接收的消息类型始终匹配，性能上保证了类型的安全性。

**⚠️ 局限性**

限制在于对于遗留代码的接口重构可能无法提供比程序员手动声明更强的保证，且在某些情况下可能无法完全恢复接口。

---

## 598. Two-Timescale Channel Estimation and Port Selection for Fluid Antenna Systems With Limited Port Observations

**arXiv ID:** 2611.25995 | [PDF](https://arxiv.org/pdf/2611.25995v1)

**作者:** Xusheng Zhu `[一作]` (University College London), Shihao Fan `[通讯]` (Beijing Jiaotong University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种针对流体天线系统的双时间尺度信道估计和端口选择方案，以克服单个射频链路在每个相干块中只能探测有限端口的限制。

**💡 创新点**

创新点在于利用慢变化的路径角度，通过序列蒙特卡洛滤波器在多个块中累积角度后验，同时以封闭形式整合快速衰落路径增益。

**🔧 技术方法**

使用了序列蒙特卡洛滤波器和费舍尔信息分析技术。

**📊 数据集**

使用了流体天线系统的模拟数据集，具体参数包括32个端口、4个波长和3条传播路径。

**📈 对比分析**

与现有的无记忆方案进行比较，提出的方案在每个块中使用4个探测端口时，其性能接近于使用8个探测端口的无记忆方案，且在多个块中逐渐接近已知角度的选择性能。

**⚠️ 局限性**

限制在于单个块的探测数量通常不足以解决多个路径的问题，且在路径数量未知的情况下，仍需进一步优化方案以适应动态变化的环境。

---

## 599. GNetSS: Towards Learning GNSS Corrections without Ground Truth

**arXiv ID:** 2611.25665 | [PDF](https://arxiv.org/pdf/2611.25665v1)

**作者:** JiaXu Chee `[一作]` (CSIRO), Milad Ramezani `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出了一种自监督的GNSS位置校正框架，旨在提高单点定位的鲁棒性，无需真实位置进行训练。

**💡 创新点**

创新点在于结合了基于变换器的网络和确定性可微分定位求解器，直接从GNSS测量一致性中推导训练信号。

**🔧 技术方法**

使用了变换器网络和确定性可微分求解器。

**📊 数据集**

使用了公共数据集（如Google Smartphone Decimeter Challenge和UrbanNav）和内部数据集进行验证。

**📈 对比分析**

与传统的单点定位（SPP）、扩展卡尔曼滤波（EKF）、移动地平线估计（MHE）和NavLab等方法进行了比较，结果显示在大多数情况下，最大3D定位误差减少超过50%。

**⚠️ 局限性**

限制在于该方法依赖于GNSS测量的一致性，可能在极端环境下表现不佳，且当前实现尚未针对运行时效率进行优化。

---

## 600. PriTabICL: Differentially Private Tabular In-Context Learning with Task-Aligned Cluster-Wise Synthesis

**arXiv ID:** 2611.25073 | [PDF](https://arxiv.org/pdf/2611.25073v1)

**作者:** Syed Ahzaz Shah `[一作]` (University of North Carolina Greensboro), Xiaochen Li `[通讯]` (University of North Carolina Greensboro)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为PriTabICL的框架，用于生成隐私保护的表格上下文学习（TabICL）演示，利用差分隐私（DP）数据合成技术。

**💡 创新点**

创新点在于通过聚类方式进行DP合成，优先考虑与预测相关的特征-目标统计关系，从而更好地保留局部结构和特征-标签关系。

**🔧 技术方法**

使用了差分隐私合成算法，结合了聚类和特征-目标工作负载的任务对齐策略。

**📊 数据集**

在四个真实世界的数据集上进行了实验，包括Adult、Airline Passenger Satisfaction、Diabetes和Phishing Websites。

**📈 对比分析**

与现有的全球DP合成基线和其他私有TabICL方法进行了比较，PriTabICL在12个评估设置中在11个设置中表现最佳，尤其在Adult和Phishing数据集上显著提高了F1分数。

**⚠️ 局限性**

局限性在于聚类可能导致每个聚类中的记录数量减少，从而增加测量噪声的相对影响，可能影响合成数据的准确性。

---

## 601. AURA: AUdio-dRiven streaming Avatar

**arXiv ID:** 2611.26540 | [PDF](https://arxiv.org/pdf/2611.26540v1)

**作者:** Nathaniel Cohen `[一作]` (Kyutai), Patrick Pérez `[通讯]` (Google DeepMind)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为AURA的音频驱动视频生成模型，能够实时生成流媒体的说话者视频，解决了现有模型在低延迟和在线生成方面的不足。

**💡 创新点**

AURA通过两阶段蒸馏方法和滚动KV缓存实现了高效的流媒体生成，显著提高了生成速度，同时保持了视频质量。

**🔧 技术方法**

使用了两阶段蒸馏管道，包括ODE轨迹初始化和带有反向模拟的分布匹配蒸馏（DMD），并采用了基于滚动KV缓存的推理方案。

**📊 数据集**

构建了一个包含30,000个合成视频的新数据集，视频由TalkVerse生成，结合了参考图像和来自Hallo3数据集的文本描述，以及合成的语音音频。

**📈 对比分析**

与现有的音频驱动视频生成模型（如TalkVerse和Live Avatar）进行比较，AURA 1.3B在实时生成时速度提高了23倍，AURA 5B在720p分辨率下以9.7 fps生成视频，速度比TalkVerse快17倍，同时在视觉质量上与TalkVerse相当。

**⚠️ 局限性**

模型的限制在于生成的分辨率和帧率可能受到硬件性能的限制，且在长时间生成时可能会出现身份漂移的问题。

---

## 602. SteerTrace: Tracing How Target Attacks Steer World Action Models

**arXiv ID:** 2611.25719 | [PDF](https://arxiv.org/pdf/2611.25719v1)

**作者:** Yukun Dai `[一作]` (Tongji University), Lei Zhu `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为SteerTrace的针对世界动作模型（WAMs）的普遍攻击方法，该方法通过一个固定的扰动来引导执行的动作朝向攻击者选择的目标。

**💡 创新点**

创新点在于首次实现了一个普遍的、有针对性的扰动，能够在保持未来预测接近干净的情况下，显著改变WAMs的执行动作。

**🔧 技术方法**

使用了针对WAMs的普遍攻击技术，结合了对模型内部机制的分析，追踪攻击效果。

**📊 数据集**

在LIBERO和RoboTwin数据集上评估了八个不同的WAMs模型，包括FastWAM、Faster-WAM、OpenWAM和LingBot-VA。

**📈 对比分析**

与同预算的噪声和BadWAM方法相比，SteerTrace在引导动作朝向目标方面表现出显著优势，任务成功率接近零，而其他方法几乎没有动作偏移。

**⚠️ 局限性**

限制在于攻击效果依赖于特定的模型架构，且在某些情况下，保持未来预测的干净可能会导致攻击效果转移到当前帧，影响攻击的有效性。

---

## 603. Passing NPCR Without Diffusion: Exact Signatures, a Detection Benchmark and an Audit of Published Image Ciphers

**arXiv ID:** 2611.25561 | [PDF](https://arxiv.org/pdf/2611.25561v1)

**作者:** Mua'ad Alfarajat `[一作]` `[通讯]` (University of Jordan), Mua'ad Alfarajat (University of Jordan)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文探讨了图像加密算法中NPCR（像素变化率）的测量及其局限性，提出了一种新的检测基准和审计方法，分析了21个已发布的图像加密方案。

**💡 创新点**

创新点在于明确区分了三种不同的扩散机制（无传播、线性链式传播和独立密文），并提出了一种基于这些机制的NPCR表的诊断方法，能够揭示已发布表格的潜在问题。

**🔧 技术方法**

使用了数学分析和模拟技术，特别是对NPCR和UACI（均方绝对变化）的统计分析，结合了对已发布方案的结构重实现和比较。

**📊 数据集**

使用了21个已发布的图像加密方案作为数据集，进行NPCR和UACI的比较和审计。

**📈 对比分析**

通过与现有的NPCR测试方法进行比较，发现传统方法无法有效区分不同的扩散机制，而本文的方法能够通过分析表格的均值和标准差来识别潜在的结构问题，性能上显著提高了对加密算法的评估能力。

**⚠️ 局限性**

限制在于本文的方法并不能证明加密方案的安全性，只能揭示其潜在的结构问题；此外，审计结果可能受到已发布数据的质量和准确性的影响。

---

## 604. Tracking Objects Through Assembly and Decomposition

**arXiv ID:** 2611.26269 | [PDF](https://arxiv.org/pdf/2611.26269v1)

**作者:** Cheryl Lam `[一作]` (Cornell University), Bharath Hariharan `[通讯]` (Cornell University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种无提示的对象跟踪和状态变化检测方法，旨在从视频中恢复所有对象的轨迹并检测分解和组装事件。

**💡 创新点**

创新点在于将组装视为分解的反向过程，提出了一种零-shot框架，能够在没有初始提示的情况下进行对象跟踪和状态变化理解。

**🔧 技术方法**

使用了双向时空分区的构建方法，并结合视觉语言模型（VLM）进行状态变化的验证和时间对齐。

**📊 数据集**

引入了一个新的数据集，包含分解和组装事件的密集注释，视频来源包括作者录制的内容和YouTube教程。

**📈 对比分析**

与现有方法相比，该方法在检测状态变化方面的召回率超过了两倍，同时在仅评估分解的基准上保持竞争力。

**⚠️ 局限性**

限制在于当前方法仍然依赖于视频质量和对象的可见性，可能在复杂场景中表现不佳。

---

## 605. Budget Internalization: Fixed-Budget RLVR Trades Test-Time Scaling for In-Budget Accuracy

**arXiv ID:** 2611.26172 | [PDF](https://arxiv.org/pdf/2611.26172v1)

**作者:** Ilia Zavidnyi `[一作]` (Czech Technical University in Prague), Sergey Titov `[通讯]` (JetBrains Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了固定预算训练对推理模型测试时扩展能力的影响，提出了预算内化的概念，表明训练预算成为模型学习行为的一部分，限制了模型在测试时的扩展能力。

**💡 创新点**

创新点在于引入了预算内化的概念，揭示了固定预算训练如何影响模型的推理能力，并且提出了在不同预算下评估模型的重要性。

**🔧 技术方法**

使用了强化学习与可验证奖励（RLVR）技术，并进行了多种训练目标的消融实验。

**📊 数据集**

使用了多个数据集，包括DeepScaleR数学数据集和多样化的程序生成推理任务数据集。

**📈 对比分析**

与基线模型进行比较，发现固定预算训练在训练预算附近表现良好，但在更大预算下性能下降，导致模型的测试时扩展能力受限。

**⚠️ 局限性**

限制在于实验主要集中在特定的模型和任务上，尚未验证预算内化效应的普遍性，且未测试更大规模和不同推理任务的模型。

---

## 606. Computational trust as intuitionistic knowledge

**arXiv ID:** 2611.25799 | [PDF](https://arxiv.org/pdf/2611.25799v1)

**作者:** Ivo Pezlar `[一作]` `[通讯]`, Ivo Pezlar

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文探讨了计算信任的概念，提出信任可以被理解为一种基于验证的可错的直觉知识。通过构建一种简单的逻辑框架，捕捉信任的本质及其脆弱性。

**💡 创新点**

创新点在于将计算信任与直觉知识相结合，提出了一种基于构造性模态命题逻辑的信任逻辑，并引入了信任模态的推理规则。

**🔧 技术方法**

使用了构造性模态命题逻辑，并结合了证明与程序的对应关系，建立了信任的推理框架。

**📊 数据集**

论文中没有具体提到使用的数据集，主要集中在理论框架的构建和逻辑推导上。

**📈 对比分析**

与现有的信任模型相比，本文的方法在逻辑上更为简单且哲学上更为扎实，能够有效捕捉信任的脆弱性和验证的必要性。

**⚠️ 局限性**

限制在于该模型主要关注软件组件的信任，而未考虑硬件故障或输入错误对信任的影响。此外，模型的适用性和扩展性仍需进一步研究。

---

## 607. RouteVerify: Routing Interventions for Step-Level Verification in Mixture-of-Experts Models

**arXiv ID:** 2611.25780 | [PDF](https://arxiv.org/pdf/2611.25780v1)

**作者:** Yucheng Du `[一作]` (University of Southern California), Xiyang Hu `[通讯]` (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过RouteVerify测试了在混合专家（MoE）模型中，路由干预如何影响步骤级验证判断，并探讨这种变化是否依赖于步骤的正确性。

**💡 创新点**

创新点在于通过控制激活替换的方式，系统性地分析了不同干预接口（路由logits、MoE输出和后MoE残差状态）对模型判断的影响，并揭示了判断变化与源步骤正确性之间的关系。

**🔧 技术方法**

使用了激活替换技术，具体包括对路由logits、MoE输出和后MoE残差状态的干预，结合了不同层和标记位置的变化。

**📊 数据集**

使用了PRMBench数据集，该数据集包含290对解决方案轨迹，每对中一个标记为正确，另一个在评估步骤中包含第一个错误。

**📈 对比分析**

通过比较三种干预接口的效果，发现后MoE残差替换在所有36次比较中产生了更大的判断变化，而路由logits和MoE输出的干预效果较弱。所有层路由的干预在九个检查点中均有效，且效果超过了随机控制组。

**⚠️ 局限性**

限制在于激活替换可能使模型状态偏离其自然分布，且干预效果依赖于干预位置、评估边际和替换计算的量。此外，研究结果基于固定的检查点集，未能识别模型验证过程中的独特计算路径或因果分数。

---

## 608. TRAC: Efficient Diffusion Language Models for Machine Translation via Length-Aware Decoding

**arXiv ID:** 2611.25643 | [PDF](https://arxiv.org/pdf/2611.25643v1)

**作者:** Tianyuan Li `[一作]` (Northeastern University), Tong Xiao `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为TRAC的自适应解码方法，用于扩散机器翻译，旨在提高翻译速度并保持翻译质量。

**💡 创新点**

TRAC通过动态调整目标生成长度和基于时间稳定性的令牌承诺，减少了冗余计算，提高了翻译效率。

**🔧 技术方法**

使用了扩散大型语言模型（dLLMs）和自适应解码技术。

**📊 数据集**

在FLORES-200、WMT23和NTREX-128数据集上进行了双向中英文翻译的实验。

**📈 对比分析**

与Fast-dLLM相比，TRAC减少了24.9%至36.8%的模型前向调用，并在句子级生成速度上实现了1.91至2.19倍的加速，同时宏平均COMET仅下降0.02点。

**⚠️ 局限性**

该方法的局限性包括对其他掩蔽扩散语言模型的推广性验证不足，以及对不同语言对的适用性需要进一步评估。

---

## 609. EverydayBench: Evaluating Satisfactory Completion of Live User-Facing Digital Work

**arXiv ID:** 2611.25146 | [PDF](https://arxiv.org/pdf/2611.25146v1)

**作者:** Fangrui Huang `[一作]` (Stanford University), Ehsan Adeli `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个新的基准，包含77个人工编写的任务，涵盖日常生活和工作的八个领域，以评估AI代理选择的结果。

**💡 创新点**

创新点在于任务在执行前指定成功标准，而不预设答案或工作流程，允许代理选择自己的结果，并通过多种方式评估结果的有效性。

**🔧 技术方法**

使用了确定性检查、对引用网页的声明验证以及基于评分标准的最终交付质量评估等技术。

**📊 数据集**

使用了77个人工编写的任务，这些任务涵盖了消费者产品、旅行、研究、教育等多个领域。

**📈 对比分析**

与现有基准相比，评估方法结合了直接检查、网页验证和组件评分，结果显示Claude Opus 5.5和Claude Fable 5.1的严格通过率最高，为54.55%。

**⚠️ 局限性**

局限性在于尽管高分配置表现良好，但仍存在许多任务未能满足完成标准，且低成本配置在各个类别中失分较多，尤其是在输出文件缺失或格式不符合要求方面。

---

## 610. Auditing Learned Dynamics in Weather Models

**arXiv ID:** 2611.25119 | [PDF](https://arxiv.org/pdf/2611.25119v1)

**作者:** Skylar Gu `[一作]` (McGill University), Christina Last `[通讯]` (Haziel)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本论文研究了GraphCast模型中稀疏自编码器特征的因果相关性，探讨了物理语义是否是因果相关性的必要条件，并测试了通过干预模型内部激活来改善极端事件预测的可能性。

**💡 创新点**

创新点在于区分了因果相关性与物理可解释性，发现即使缺乏传统物理语义，模型的某些特征仍然可以对预测技能产生重要影响，同时物理相关特征可以作为因果干预的有效坐标。

**🔧 技术方法**

使用了GraphCast模型的稀疏自编码器（SAE）特征，并通过对模型内部激活的干预来测试其因果效应。

**📊 数据集**

使用了ERA5数据集，该数据集结合了模型与历史观测数据，以估计过去的气象状态，研究对象包括2007年至2018年间的多个热带气旋。

**📈 对比分析**

通过与随机控制向量和统计偏差校正的方法进行比较，发现放大与对流和低层涡旋相关的特征可以显著降低预测误差，最高可达51%的误差减少。

**⚠️ 局限性**

限制在于物理标签仅描述特征激活与ERA5字段之间的统计关联，并未证明SAE特征代表特定的物理变量或机制。此外，干预研究仅覆盖了七个快速增强的热带气旋，且不同风暴的最佳干预强度（α值）差异显著。

---

## 611. Reference-Twist-Based Velocity-Domain Spatiotemporal Calibration for mmWave Radar, Event Camera, and IMU

**arXiv ID:** 2611.26083 | [PDF](https://arxiv.org/pdf/2611.26083v1)

**作者:** Zhenghao Zou `[一作]`, Quan Pan `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `51c0528b-f690-4182-ae60-bb5f046c276c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种参考条件下的速度域时空标定方法，针对一组异构传感器（包括两台毫米波雷达、一台事件相机和两台惯性测量单元）进行标定。

**💡 创新点**

创新点在于将参考体的运动作为外部条件信号，避免了传统几何标定中对目标或特征的依赖，同时保留了在旋转激励下的杠杆臂贡献。

**🔧 技术方法**

使用了参考条件的速度域标定方法，结合了概率和局部信息分析，分析了在局部可识别性和独立同步参考下的条件效应。

**📊 数据集**

使用了室内和森林环境中的录音数据进行实验，评估了初始化、时间对齐、运动退化、参考输入敏感性和共享观测残差一致性。

**📈 对比分析**

与联合估计方法进行了比较，结果表明在参考准确性和同步性、运动激励、投影多样性和传感器模型有效性条件下，提出的方法在性能上具有优势。

**⚠️ 局限性**

限制在于估计结果依赖于参考的准确性和同步性，运动激励和投影多样性，以及传感器模型的有效性。

---

## 612. Berth: SLO-Preserving GPU Sharing Without New Preemption

**arXiv ID:** 2611.25091 | [PDF](https://arxiv.org/pdf/2611.25091v1)

**作者:** Mohammadali Khodabandehlou `[一作]` (University of Southern California), Bhaskar Krishnamachari `[通讯]` (University of Southern California)

**通讯引用:** 28432 | [OpenAlex ID](https://openalex.org/A5063784062)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Berth的用户空间系统，旨在优化GPU共享，确保延迟敏感的请求在不干扰最佳努力工作的情况下得到满足。

**💡 创新点**

创新点在于利用现有GPU的时间切片机制，通过外部控制来创建优先级，而无需修改应用程序、框架、内核或驱动程序。

**🔧 技术方法**

使用了用户空间的shim和控制机制，结合了时间切片和队列控制来管理GPU资源。

**📊 数据集**

使用了MLPerf SLOs和vLLM服务的基准数据集进行评估。

**📈 对比分析**

与多个基线方法（如XSched、TGS、Orion和Tally）进行比较，Berth在SLO达成率上优于所有基线，并且在最佳努力吞吐量上是基线的几倍。

**⚠️ 局限性**

限制在于Berth无法看到请求队列，可能在请求高峰期或饱和状态下表现不佳，且需要为每个新模型或批量大小重新配置性能分析。

---

## 613. MatVerse: A Foundation Model for Materials Science

**arXiv ID:** 2611.25651 | [PDF](https://arxiv.org/pdf/2611.25651v1)

**作者:** Xuanyu Zhu `[一作]` (Shanghai Jiao Tong University), Xin Chen `[通讯]` (Suzhou Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种材料基础模型，通过对Intern-S2-Preview进行大规模领域特定的后训练，旨在为多种材料科学任务提供统一的支持。

**💡 创新点**

创新点在于构建了一个包含约150K材料领域后训练示例的语料库，结合了文本推理数据和多模态科学内容，显著提升了模型在材料知识、定量推理和多模态科学理解方面的能力。

**🔧 技术方法**

使用了Mixture-of-Experts架构的Intern-S2-Preview模型，并通过领域特定的后训练增强其能力，采用了教师引导的推理蒸馏和多模态数据构建管道。

**📊 数据集**

使用了OmniMatBench、MatSciBench和MaCBench等多个材料科学基准数据集进行评估，这些数据集涵盖了广泛的材料科学知识和推理能力。

**📈 对比分析**

与基础模型Intern-S2-Preview相比，模型在MatSciBench上提高了5.14个百分点，在MaCBench上提高了2.17个百分点，而在OmniMatBench的CAL轨道上提高了2.47个百分点，显示出在定量推理和多模态理解方面的显著改进。

**⚠️ 局限性**

限制在于开放式问题回答的性能仍低于最强系统，公式推导的准确率接近9%，并且某些子领域的表现接近零。未来工作将探索在可验证计算任务上的强化学习，以及对欠代表子领域的额外监督。

---

## 614. GaussTrans: A Gaussian Interface Framework for Multi-Layer Refractive Transport

**arXiv ID:** 2611.25950 | [PDF](https://arxiv.org/pdf/2611.25950v1)

**作者:** Zixuan Zhao `[一作]`, Tao Wang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研究了多透明界面的表面建模，以实现准确的光传输，提出了GaussTrans框架，解决了高层透明场景中的光传输问题。

**💡 创新点**

创新点在于提出了高阶透明场景的紧凑框架，结合了高斯表面和场值方法，包含高斯场混合表面、传输耦合细化和原始引导隐式交点三个组件。

**🔧 技术方法**

使用了高斯加权局部多项式表面补丁、传输耦合细化和原始引导隐式交点等技术。

**📊 数据集**

使用了RefRef数据集进行实验，验证了所提方法的有效性。

**📈 对比分析**

与其他方法相比，GaussTrans在相同几何存储条件下，表现出更低的RGB误差和更高的SSIM，且在多层透明界面传播中具有更好的连续性。

**⚠️ 局限性**

局限性在于输入几何的质量和局部覆盖的依赖性，对于薄壁结构或快速变化曲率的区域，可能会导致位置和法线偏差，影响方法的适用性。

---

## 615. Vector Field Transforms: Structured Hyperdimensional Memory for Spatio-Temporal Reasoning

**arXiv ID:** 2611.24962 | [PDF](https://arxiv.org/pdf/2611.24962v1)

**作者:** William Youngwoo Chung `[一作]`, Mohsen Imani `[通讯]` (University of California)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的向量场变换（VFT）方法，作为一种结构化的符号时空记忆，旨在解决传统平面VSA记忆的核心限制。

**💡 创新点**

VFT通过用空间索引的局部记忆场替代单一的全局叠加，支持显式的传输操作和通过年龄通道的时间回忆，从而使符号记忆能够在运动下有序演变。

**🔧 技术方法**

使用了向量符号架构（VSA）和超维计算（HDC）技术，并将VFT的核心内核映射到可编程的RISC-V向量处理器HyperV上。

**📊 数据集**

在多个基准测试中使用了不同的场景数据集，包括局部混乱实验、噪声自我运动实验和时间回忆基准。

**📈 对比分析**

与传统的平面VSA记忆相比，VFT在局部混乱场景中实现了100.0%的准确率，而平面VSA仅为20.3%；在时间回忆基准中，VFT的准确率为74.8%，而平面VSA仅为5.8%和10.6%。

**⚠️ 局限性**

VFT的局限性在于其对硬件的依赖性，尤其是在资源受限的边缘设备上，可能需要进一步优化以提高效率和降低延迟。

---

## 616. A Control-Theoretic Approximation to Predictive Coding Dynamics

**arXiv ID:** 2611.26319 | [PDF](https://arxiv.org/pdf/2611.26319v1)

**作者:** Ryan Fayyazi `[一作]` (New York University), Mitra Javadzadeh `[通讯]` (Cold Spring Harbor Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文展示了深度反馈控制（DFC）作为预测编码（PC）动态的局部近似，揭示了两者之间的关系，并提出了一种新的近似方法Approx PC。

**💡 创新点**

创新点在于将DFC视为PC动态在固定点附近的局部近似，并提出通过中间目标序列来减小Approx PC与PC之间的偏差。

**🔧 技术方法**

使用了控制理论和动态系统的分析方法，结合了反馈控制和预测编码的框架。

**📊 数据集**

使用了MNIST和FashionMNIST数据集进行实验验证。

**📈 对比分析**

与传统的PC方法相比，Approx PC在任务性能上达到了相似的效果，并且学习的权重可以直接转移到PC推理中，表明其有效性。

**⚠️ 局限性**

限制在于Approx PC在网络深度增加时的偏差可能会增大，且在远离固定点时的可靠性可能下降。

---

## 617. RAVEN: A Self-RAG Based Agent for Threat Classification and Response

**arXiv ID:** 2611.25832 | [PDF](https://arxiv.org/pdf/2611.25832v1)

**作者:** Pironti Francesco Aurelio `[一作]`, Furfaro Angelo `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于自我反思检索增强生成（Self-RAG）的代理框架，用于支持安全运营中心（SOC）分析师自动化安全事件的初步分类和响应。

**💡 创新点**

创新点在于通过自我反思机制提高检索和生成的质量，提供上下文感知的缓解建议，并且不需要对模型进行微调。

**🔧 技术方法**

使用了自我反思检索增强生成（Self-RAG）技术，结合大型语言模型（LLMs）进行日志分析和安全事件分类。

**📊 数据集**

使用了三个平衡的测试集，每个测试集包含400个策划的网络负载（SQL注入、跨站脚本、服务器端模板注入），并进行了定性案例研究。

**📈 对比分析**

与传统的RAG方法相比，Self-RAG在上下文依赖的攻击（如SSTI）上表现更好，F1分数从0.84提高到0.92，而在模式明确的攻击（如SQLi和XSS）上，LLM仅基线已经接近上限。

**⚠️ 局限性**

限制在于当前评估是基于策划的数据集，而不是实际IDS生成的警报，且尚未进行用户研究以评估分析师的认知负担和响应时间的改善。

---

## 618. Process-Knowledge-Guided Latent Diffusion for Generative Design of Porous Architectures: A Case Study in Aerosol Jet Printing

**arXiv ID:** 2611.25264 | [PDF](https://arxiv.org/pdf/2611.25264v1)

**作者:** Ngoc Dang Khoa Nguyen `[一作]` (Arizona State University), Hyunwoong Ko `[通讯]` (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了如何利用过程知识引导潜在扩散生成器生成适合特定过程设置的多孔布局，提出了一种过程依赖的利用评分，并应用于气溶胶喷墨打印。

**💡 创新点**

创新点在于将过程模型与生成模型结合，提出了一种新的利用评分机制，并通过控制网络和回归器引导生成过程，展示了过程设置对布局选择的影响。

**🔧 技术方法**

使用了向量量化变分自编码器（VQ-VAE）、拓扑生成U-Net、ControlNet领域分支和单独训练的回归器来预测过程依赖的设计评分。

**📊 数据集**

使用了包含500个域的布局数据集，每个域包含约70种通道模式，总共生成了34,991个布局，数据集分为训练、验证和测试集。

**📈 对比分析**

通过与零引导的域条件采样器进行比较，发现引导采样在多个设置下的平均直接计算评分显著降低，表明引导对生成的布局有负面影响，尤其在较高的引导权重下。

**⚠️ 局限性**

限制在于过程-形态因子和利用长度的校准尚未针对打印沉积进行，且未考虑打印特征和通道宽度的最小要求，未来需要进行实际打印和评估。

---

## 619. Splice-Core: Low-Order-Bit-Splicing GEMM for Highly Energy-Efficient AI Inference Accelerators

**arXiv ID:** 2611.25617 | [PDF](https://arxiv.org/pdf/2611.25617v1)

**作者:** Prabhu Vellaisamy `[一作]` (Carnegie Mellon University), John Paul Shen `[通讯]` (Carnegie Mellon University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一种名为Splice-Core的低阶位拼接GEMM单元，旨在提高AI推理加速器的能效，包含两种新型处理单元变体：Splice-BP和Splice-LUT。

**💡 创新点**

创新点在于通过时间-一元/二元混合设计，减少低阶累积延迟，同时保留高阶激活位的值依赖执行，显著降低能耗和提高吞吐量。

**🔧 技术方法**

使用了时间-一元计算、低精度GEMM、INT4和INT8推理、操作数稀疏性等技术。

**📊 数据集**

使用了26个INT4卷积神经网络（CNN）和大型语言模型（LLM）工作负载的数据集进行评估。

**📈 对比分析**

与二元脉冲阵列相比，Splice-BP和Splice-LUT分别实现了77.4%和76.4%的几何平均能量减少，Splice-LUT还提高了44.4%的几何平均吞吐量，并将能量延迟乘积（EDP）降低了6.13倍。

**⚠️ 局限性**

限制在于Splice-BP和Splice-LUT在不同工作负载下的性能表现可能有所不同，且在某些情况下可能无法充分利用矩阵引擎的计算能力。

---

## 620. FoodReact-AR: Learning Audience Reactions that Zero-Shot LLMs Miss in Arabic Food-Insecurity Discourse

**arXiv ID:** 2611.25682 | [PDF](https://arxiv.org/pdf/2611.25682v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 621. Represented but misread: Post-training changes how language models read out what they represent

**arXiv ID:** 2611.26584 | [PDF](https://arxiv.org/pdf/2611.26584v1)

**作者:** Jochen Laubrock `[一作]` `[通讯]` (University of Potsdam), Jochen Laubrock (University of Potsdam)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文比较了语言模型在后训练前后的表现，分析了模型的内部表示与其输出答案之间的关系，特别是在判断文本是否由人类或AI撰写时的表现。

**💡 创新点**

创新点在于通过线性探针和信号检测理论分析模型的内部表示与输出答案之间的差异，揭示了后训练对模型判断能力的影响。

**🔧 技术方法**

使用了线性探针、信号检测理论、模型拼接和激活修补等技术来分析模型的内部状态和输出。

**📊 数据集**

使用了三种德语文本数据集，包括学生撰写的聊天消息、地方报纸的文章片段和教科书章节的摘要。

**📈 对比分析**

与基模型相比，后训练模型在判断能力上普遍下降，尤其是在使用相同提示时，后训练模型的敏感度平均下降了0.34，且在多个模型中表现出逆向判断的现象。

**⚠️ 局限性**

限制在于无法共享用于训练的文本数据集，且模型在某些情况下的判断能力仍然低于随机水平，表明模型的输出与其内部表示之间存在显著差异。

---

## 622. Seer: Guiding Small Deep-Research Agents with Shared Decision Models

**arXiv ID:** 2611.26423 | [PDF](https://arxiv.org/pdf/2611.26423v1)

**作者:** Yuhao Zhou `[一作]` (Sichuan University), Jiancheng Lv `[通讯]` (Sichuan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为Seer的推理时框架，用于在深度研究中提供本地指导，分离代理的行动提议与学习的工具响应预测和价值评估。

**💡 创新点**

Seer的创新点在于其设计允许在不进行模型特定再训练的情况下，跨代理重用指导，同时避免对被丢弃候选的真实执行。

**🔧 技术方法**

使用了工具响应模拟器和价值评估器，这两个模型在离线训练后可以在不同的代理之间共享，而无需进一步训练。

**📊 数据集**

使用了一个共享的真实研究轨迹数据集，通过Monte Carlo分支收集的真实交互历史。

**📈 对比分析**

与其他方法相比，Seer在多个深度研究基准上显著提高了代理的性能，例如GAIA的准确率从13.6%提高到41.7%。

**⚠️ 局限性**

限制在于Seer的性能可能受到预测和评分错误的限制，尽管更广泛的候选覆盖可以弥补不完美的模拟。

---

## 623. FrontierReplay: Split-Free Exact Prefix Caching for Hybrid Language Models

**arXiv ID:** 2611.26020 | [PDF](https://arxiv.org/pdf/2611.26020v1)

**作者:** Zekun Wang `[一作]` (Beijing Institute of Technology), Yizhuo Wang `[通讯]` (Beijing Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为FrontierReplay的前缀缓存系统，旨在提高混合语言模型的性能，通过在可预测的结构边界处生成精确的检查点，而无需切断前填充过程。

**💡 创新点**

创新点在于采用无切割的检查点生成方法，利用输入日志和状态重放内核，在不影响前填充的情况下生成精确的状态，从而显著提高了模型的吞吐量和响应速度。

**🔧 技术方法**

使用了状态重放技术，通过记录短时间窗口内的递归层输入，并在后续步骤中重放这些状态来生成检查点，避免了传统方法中的计算开销。

**📊 数据集**

在多个数据集上进行了评估，包括ShareGPT对话、SWE-agent工具会话、LongBench文档问答和GSM8K树搜索，涵盖了不同的混合架构和模型规模。

**📈 对比分析**

与现有的精确缓存方法（如vLLM、Marconi、SPC和SuffixReplay）相比，FrontierReplay在所有测试中表现出色，降低了首次令牌的中位时间（TTFT）16-89%（最高可达8.9倍），并提高了80%服务水平目标（SLO）容量和良好吞吐量。

**⚠️ 局限性**

限制在于检查点的数量受限于缓存大小，尽管采用了前沿感知的保留策略，但在高负载情况下仍可能面临缓存不足的问题。

---

## 624. Hybrid Retrieval for Government Scheme Search in English and Marathi

**arXiv ID:** 2611.26514 | [PDF](https://arxiv.org/pdf/2611.26514v1)

**作者:** Aishwarya Gunjal `[一作]` (Pune Institute of Computer Technology), Manish Khodaskar `[通讯]` (Pune Institute of Computer Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种双语检索系统，用于马哈拉施特拉邦政府计划和公共服务，结合了BM25检索和句子嵌入相似性，通过等权重得分融合进行检索。

**💡 创新点**

创新点在于将词汇匹配与语义表示相结合，提供了一种新的双语检索管道，并比较了不同检索方法的效果。

**🔧 技术方法**

使用了BM25检索、IndicSBERT和MahaSBERT句子嵌入模型，以及等权重的得分融合技术。

**📊 数据集**

使用了64个马哈拉施特拉邦政府计划和公共服务的双语数据集，包含英语和马拉地语的描述。

**📈 对比分析**

与BM25、仅使用IndicSBERT和MahaSBERT的密集检索方法进行了比较，结果显示BM25的平均倒数排名（MRR）为0.864，混合检索使用MahaSBERT时提高到0.879，表明混合检索在某些指标上优于BM25，但整体改进是适度的。

**⚠️ 局限性**

限制在于评估仅基于64个方案和150个手动策划的查询，无法全面代表所有政府项目的表现，翻译和领域术语的准确性也可能影响结果。

---

## 625. LLMs Are Already Confidence-Aware Zero-Shot Classifiers

**arXiv ID:** 2611.25289 | [PDF](https://arxiv.org/pdf/2611.25289v1)

**作者:** Owen Cook `[一作]` (University of Sheffield), Xingyi Song `[通讯]` (University of Sheffield)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究比较了黑箱方法（如口头信心和采样）与白箱方法（如读取标记概率和多种逻辑评分方法）在零样本大语言模型（LLMs）作为文本分类器中的有效性。

**💡 创新点**

创新点在于引入了一种仅在前缀树中评分标签分歧位置的决策评分方法，显著减少了解码步骤，并在多个数据集上表现出竞争力。

**🔧 技术方法**

使用了多种解码方法，包括无约束生成和约束生成，结合了逻辑评分方法和口头信心提取。

**📊 数据集**

使用了五个分类数据集，包括Banking77、CLINC150、20 Newsgroups、SST-5和FEVER，涵盖了不同的任务和标签长度。

**📈 对比分析**

与商业模型Jev 1.13.0相比，最佳的开放权重模型-方法组合在准确性和F1宏观得分上具有竞争力，但在原始校准方面略逊一筹。

**⚠️ 局限性**

限制在于未探讨温度缩放对逻辑评分方法校准的影响，且所有实验均为零样本，未考虑少样本提示的潜在改进。

---

## 626. EgoCaptioner: Towards Comprehensive and Fine-Grained Egocentric Video Captioning

**arXiv ID:** 2611.26380 | [PDF](https://arxiv.org/pdf/2611.26380v1)

**作者:** Yuqi Tang `[一作]` (Hong Kong University of Science and Technology), Chengwei Qin `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的自我中心视频字幕生成方法，能够在两个层次上描述视频：视频级概述和详细步骤，解决了现有方法在细节和时间上的不足。

**💡 创新点**

创新点在于采用了粗到细的注释流程和强化学习，优化了事实内容、时间对齐和字幕风格，显著提高了字幕质量和时间准确性。

**🔧 技术方法**

使用了监督微调和强化学习技术，结合了多种奖励机制来优化模型性能。

**📊 数据集**

使用了来自八个自我中心视频源的50,000个经过精心挑选的示例进行训练，包括日常活动、程序任务和户外行走等。

**📈 对比分析**

与Qwen3.5-9B模型相比，在PhysCapBench上提高了27.6分，在EgoTempo准确性上提高了16.8个百分点，整体超越了Gemini 3.1 Pro，并在EgoHome的步骤分割中获得了最高的PSJ分数。

**⚠️ 局限性**

限制在于模型在处理复杂场景时仍可能出现误解，特别是在快速运动或多对象交互的情况下，可能导致时间和动作的描述不够准确。

---

## 627. Harnessing Medical Visual-Language Models with Agentic Evidence Management

**arXiv ID:** 2611.25840 | [PDF](https://arxiv.org/pdf/2611.25840v1)

**作者:** Qiming Huang `[一作]`, Furu Wei `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了一个新的医学视觉问答基准Mask-Evidence MedVQA，包含2494个问题，涵盖四种医学图像类型，并引入了证据管理工具TUNE，以提高医学视觉语言模型的性能。

**💡 创新点**

创新点在于提出了证据管理的概念，强调不仅要获取正确的分割掩码，还要有效地使用和呈现这些证据，以决定是否更新模型的答案。

**🔧 技术方法**

使用了医学视觉语言模型（VLM）和证据管理工具TUNE，结合了目标规划、证据呈现和答案评估三个阶段。

**📊 数据集**

使用了Mask-Evidence MedVQA数据集，该数据集包含超声、胸部X光、内窥镜和CT图像类型的2494个问题，并配有相关的真实掩码。

**📈 对比分析**

与直接工具使用的方法相比，TUNE在三个公共医学VQA基准上平均提高了4.0个百分点，在七个数据集上平均提高了9.1个百分点，表现出竞争力的结果。

**⚠️ 局限性**

限制在于尽管引入了证据管理，但仍需进一步研究如何扩展到更复杂的医学工具和开发更轻量、可扩展的代理框架。

---

## 628. World-delta: Continually Integrating Observations for Incremental 3D Scene Generation

**arXiv ID:** 2611.25802 | [PDF](https://arxiv.org/pdf/2611.25802v1)

**作者:** Haonan Wang `[一作]` (Huazhong University of Science and Technology), Luxin Yan `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种增量3D场景生成框架World-δ，该框架能够根据不断到来的RGB观察数据持续扩展和修正生成的3D世界。

**💡 创新点**

创新点在于结合了不确定性感知的持久世界建模和基于增量更新的扩散生成方法，实现了对场景的持续扩展和修正，同时减少了冗余生成。

**🔧 技术方法**

使用了视觉证据编码、观察校准的世界建模和增量世界扩散生成等技术。

**📊 数据集**

使用了多个数据集，包括RealEstate10K、ScanNet-v2和Replica等进行实验。

**📈 对比分析**

与其他方法（如Lyra-1、HY-World-2.0等）进行比较，World-δ在多个指标上表现优越，尤其是在渲染质量和更新效率方面。

**⚠️ 局限性**

限制在于假设场景大致静态，因此在处理动态物体时可能会混淆早期生成的错误。

---

## 629. AesDistill: Distilling Generative Aesthetics for Image Color Enhancement

**arXiv ID:** 2611.25635 | [PDF](https://arxiv.org/pdf/2611.25635v1)

**作者:** Chuanzhi Xu `[一作]` (University of Sydney), Weidong Cai `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种将生成美学知识转化为可重用目标的图像颜色增强框架，旨在提高图像的视觉吸引力。

**💡 创新点**

通过从大型生成模型中提取美学知识，构建了一个紧凑的学生模型，该模型在多个评估中超越了基于专家目标的传统方法。

**🔧 技术方法**

使用了四个领先的开源大型生成模型进行教师目标生成，并采用了置信加权监督来训练学生模型。

**📊 数据集**

使用了MIT-Adobe FiveK和PPR10K数据集，生成了64,644个编辑图像作为教师目标。

**📈 对比分析**

与20种基线方法进行比较，结果显示该模型在美学和图像质量评估中均表现优异，整体排名仅次于四个教师模型。

**⚠️ 局限性**

模型的局限性在于可能无法完全捕捉所有美学偏好，且在某些情况下生成的图像可能会改变源图像的内容和语义结构。

---

## 630. Learning Generative Ranking Policies for Whole-Page Optimization

**arXiv ID:** 2611.25527 | [PDF](https://arxiv.org/pdf/2611.25527v1)

**作者:** Chen Qiu `[一作]` (Meta Platforms, Inc.), Luke Simon `[通讯]` (Meta Platforms, Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了整体页面优化，通过联合选择和排序页面上的项目来最大化页面得分。

**💡 创新点**

提出了整体页面策略改进方法，通过当前生成器生成的完整页面得分来评估下一个项目的选择，从而实现更好的整体页面决策。

**🔧 技术方法**

使用了生成式重排序方法，结合局部页面得分目标进行训练。

**📊 数据集**

使用了两个公共数据集KuaiRand-1K和RL4RS-SeqSlate，以及一个工业数据集的日志页面。

**📈 对比分析**

与生成式推荐基线和代表性的强化学习算法进行了比较，结果显示该方法在每个长度设置上均优于这些基线，并且在生成时间上显著更低。

**⚠️ 局限性**

方法的局限性在于需要固定的评估器进行训练，且在生成过程中不进行评估器查询。

---

## 631. Standardized Tolerance for Adaptive Reuse and Repair: A Method for HNSW Migration Under Embedding Model Change Scenarios

**arXiv ID:** 2611.25416 | [PDF](https://arxiv.org/pdf/2611.25416v1)

**作者:** Hengzhao Ma `[一作]` (Northeastern University), Xiaochun Yang `[通讯]` (Northeastern University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为STARR（标准化容忍度用于自适应重用和修复）的方法，用于在嵌入模型变化的情况下迁移HNSW索引。

**💡 创新点**

创新点在于通过比较源邻居列表中相同对象的标准化边长，利用距离轮廓之间的相关性来设置允许的边长增加，从而实现索引的迁移和修复。

**🔧 技术方法**

使用了标准化距离度量、局部邻居列表筛选、选择性重连和可达性修复等技术。

**📊 数据集**

使用了MS MARCO、Wikipedia和CC3M数据集，每个数据集包含一百万个对象。

**📈 对比分析**

与Fresh、DGM-Local和DGM-Search等方法进行比较，STARR在所有15个选定的模型转换中迁移速度更快，构建速度提升为5.25到44.54倍，且在10个案例中Recall@10与Fresh相差不超过一个百分点，查询吞吐量在14个案例中超过DGM-Local。

**⚠️ 局限性**

限制在于迁移质量依赖于模型对和修复预算，某些困难的转换可能需要更严格的筛选和更广泛的修复搜索以提高召回率。

---

## 632. MKA: Evaluating Multimodal Knowledge Agents on end-to-end professional tasks

**arXiv ID:** 2611.25228 | [PDF](https://arxiv.org/pdf/2611.25228v1)

**作者:** Anima Singh `[一作]` (Google DeepMind), Jean-Baptiste Alayrac `[通讯]` (Google DeepMind)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MKA（多模态知识代理），这是一个涵盖200个专业工作流程的基准，涉及10个经济部门和83个职业，旨在评估视觉能力在知识工作中的重要性。

**💡 创新点**

创新点在于引入了一个全面的多模态基准，强调视觉工具在处理复杂文档和多文件交付物中的关键作用，并展示了当前模型在视觉理解方面的不足。

**🔧 技术方法**

使用了标准化的Stirrup框架进行模型评估，并结合了Agentic Autorater进行自动评分，确保评分的客观性和一致性。

**📊 数据集**

数据集包含200个真实世界的任务，涵盖10个经济部门，每个部门20个任务，涉及的输入文件包括PDF文档、图像、电子表格和地理空间数据等。

**📈 对比分析**

与现有基准（如GDPval和APEX-Agents）相比，MKA在多模态、浏览、GUI和专业知识工作评估中表现出更高的视觉工具使用和多文件交付能力。评估结果显示，Claude Fable 5.2和Gemini 4 Carbon在交付物质量上表现最佳，但仍有显著的提升空间。

**⚠️ 局限性**

限制在于每个模型仅运行一次并在单次评分中评估，可能无法捕捉到模型或评分者的变异性。此外，视觉工具的依赖程度可能因模型而异，且基准的API成本可能会随时间变化。

---

## 633. Reject and Report: Forgery-Evident Undetectable Watermarks for Generative Image Models

**arXiv ID:** 2611.25226 | [PDF](https://arxiv.org/pdf/2611.25226v1)

**作者:** Yihan Wang `[一作]` (University of Waterloo), N. Asokan `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的不可检测水印框架，该框架能够拒绝伪造并报告伪造证据。

**💡 创新点**

创新点在于将生成图像的语义承诺写入潜在噪声的幅度中，使得伪造图像携带伪造证据，并指向被滥用的源图像。

**🔧 技术方法**

使用了伪随机码（PRC）水印和高斯阴影（GS++）技术，结合了语义哈希网络。

**📊 数据集**

使用了MS-COCO数据集进行实验，评估了伪造和鲁棒性。

**📈 对比分析**

与现有方法（如SemBind）相比，该方法在拒绝伪造方面表现更好，接受率低至1.8%，并且能够报告所有伪造图像，而SemBind在伪造图像上接受率高达55%。

**⚠️ 局限性**

局限性在于语义检查依赖于一个秘密的哈希网络，该网络没有计算保证，并且只能在提示级别上区分内容。

---

## 634. Reinforcement-Aligned Probabilistic Weather Forecasting with Discrete Tokens

**arXiv ID:** 2611.25570 | [PDF](https://arxiv.org/pdf/2611.25570v1)

**作者:** Jinhan Li `[一作]` (University of Science and Technology of China), Kun Wang `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种统一的框架，通过离散天气标记实现概率生成和强化学习对齐，以提高全球中期天气预报的效率和准确性。

**💡 创新点**

创新点在于将GPT风格的预训练-后训练范式引入全球中期天气预报，并通过多尺度标记器和Transformer模型实现时间和尺度上的因果依赖建模。

**🔧 技术方法**

使用了多分支、多尺度的空间残差标记器和Transformer模型，结合了强化学习中的Group Relative Policy Optimization (GRPO)。

**📊 数据集**

使用了ERA5基准数据集，该数据集包含全球69个通道的日均气象数据。

**📈 对比分析**

与多个确定性基线模型（如OneForecast、GenCast等）进行比较，结果显示该框架在15天的归一化均方根误差（nRMSE）上减少了17.6%，并且推理速度比GenCast快约18倍。

**⚠️ 局限性**

限制在于该方法在处理极端事件和耦合模拟时的适应性仍然面临挑战，且需要进一步扩展到更细的空间-时间分辨率。

---

## 635. Before the Choice: An Evidential Framework for Studying Prospective Representations in Language Models

**arXiv ID:** 2611.25170 | [PDF](https://arxiv.org/pdf/2611.25170v1)

**作者:** Sion Israel Sion `[一作]` `[通讯]` (Ecole de technologie superieure), Sion Israel Sion (Ecole de technologie superieure)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个证据导向的框架，名为“选择之前”，用于研究语言模型中的前瞻性表示，区分未来导向内容、前瞻性计算和因果指导。

**💡 创新点**

创新点在于明确区分了前瞻性表示的内容、处理和选择的因果贡献，并提出了一个实验对比矩阵来组织实验。

**🔧 技术方法**

使用了低秩探测器和控制实验设计来评估模型的前瞻性表示能力。

**📊 数据集**

使用了符号环境中的实验数据集，具体包括293对可接受的实验对比。

**📈 对比分析**

与基线方法（如文本和观察选择基线）进行比较，发现探测器的配对准确率为0.020，远低于基线的0.195，表明模型未能有效识别未选择的轨迹。

**⚠️ 局限性**

局限性在于探测器未能有效解码标签身份，且未能证明模型在决策前是否具备对未选择路径的前瞻性表示。

---

## 636. Approximation Algorithms for Asymmetric Multi-Path TSP

**arXiv ID:** 2611.26188 | [PDF](https://arxiv.org/pdf/2611.26188v1)

**作者:** Manuel Christalla `[一作]` (ETH Zurich), Vera Traub `[通讯]` (ETH Zurich)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了非对称多路径旅行商问题（Asymmetric Multi-Path TSP），旨在找到k条路径，每条路径有指定的起点和终点，确保每个顶点至少被一条路径访问，并最小化所有路径的总长度。

**💡 创新点**

提出了一种O(k)的近似算法，运行时间为n^O(k)，解决了Friggstad在2013年提出的一个开放问题。对于所有起点和终点相同的特殊情况，得到了多项式时间内的O(k)近似和(O(1/ϵ), 1+ϵ)的双标准近似。

**🔧 技术方法**

使用了动态规划算法和现有的常数因子近似算法的技术，结合了对特殊情况的处理。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了多种图的情况，包括有向图和无向图。

**📈 对比分析**

与现有方法相比，提出的算法在多路径旅行商问题上提供了更好的近似保证，尤其是在特殊情况下的常数因子近似。

**⚠️ 局限性**

由于判断是否存在可行的多路径解决方案是NP难的，因此对于任意k，无法期望有多项式时间的近似算法。此外，是否存在独立于k的常数因子近似仍然是一个开放问题。

---

## 637. MambaLIFT: Reference-Based Mechanism-Aware Attribution for VMamba

**arXiv ID:** 2611.25644 | [PDF](https://arxiv.org/pdf/2611.25644v1)

**作者:** Jintu Kakati `[一作]` (Indian Institute of Technology Bhubaneswar), Niladri B. Puhan `[通讯]` (Indian Institute of Technology Bhubaneswar)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为MambaLIFT的参考基础机制感知归因框架，用于VMamba模型，旨在解决视觉状态空间模型的归因问题。

**💡 创新点**

MambaLIFT通过引入交互感知的传播规则，能够有效地分解参考相对的类logit差异，并提供机制级分析和保守审计。

**🔧 技术方法**

采用了基于乘法的归因方法，结合了DeepLIFT的参考相对线性和重缩原则，使用了交互感知的传播规则。

**📊 数据集**

使用了ImageNet-1K数据集进行实验，评估了VMamba-T和VMamba-B模型。

**📈 对比分析**

与MambaLRP等现有方法相比，MambaLIFT在均值颜色像素翻转下显著提高了均值信度得分，并在模糊替换下表现出非劣性，表明其优势依赖于替换协议。

**⚠️ 局限性**

MambaLIFT的局限性在于，尽管其归因结果在数值上保持一致，但并不保证在有限干预下的因果有效性，且归因图受交互分配、时间传播和路径组成的影响。

---

## 638. Language-Oriented Semantic Modeling for Target Language Extraction

**arXiv ID:** 2611.26317 | [PDF](https://arxiv.org/pdf/2611.26317v1)

**作者:** Mehmet Sinan Yıldırım `[一作]` (National University of Singapore), Haizhou Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种语言导向的语义建模框架，用于从多语言混合中提取目标语言的语音，增强了声学提取过程。

**💡 创新点**

创新点在于将语言特定的语义表示与声学提取相结合，提供了超越语言身份标签的结构化语义指导。

**🔧 技术方法**

使用了语言条件的语音分离、语义建模、提示适应等技术。

**📊 数据集**

使用了CommonVoiceMix数据集，该数据集包含来自英语和德语的混合语音。

**📈 对比分析**

与声学仅基线相比，结合语义表示的方法在SI-SDR上提高了1.16 dB，且在STOI和PESQ指标上也表现出一致的改进。

**⚠️ 局限性**

限制在于语义标记在没有语言身份标签的情况下可能缺乏区分能力，且模型可能在提取过程中受到干扰。

---

## 639. Proportional Representation for Almost-Solid Coalitions and the Proportional Veto Core for Rankings

**arXiv ID:** 2611.25677 | [PDF](https://arxiv.org/pdf/2611.25677v1)

**作者:** Evi Micha `[一作]` (University of Southern California), Fatih Erdem Kizilkaya `[通讯]` (University of Southern California)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文扩展了比例代表制的研究，提出了候选人和集合的比例否决保护，确保在投票者偏好不完全一致的情况下，几乎固体联盟的代表性得到保证。

**💡 创新点**

创新点在于引入了候选人和集合的比例否决保护，提出了稳健的比例代表制（robust Rank-PSC），并展示了其与传统比例代表制的区别，尤其是在不完美一致的情况下的表现。

**🔧 技术方法**

使用了反向顺序的有序Phragmén和单转移投票（STV）等技术，构建了满足稳健比例代表制的排名，并提出了多项多项式时间算法。

**📊 数据集**

论文中使用的具体数据集未明确提及，但涉及的投票者和候选人集合是基于理论模型构建的。

**📈 对比分析**

与传统的比例代表制方法相比，稳健比例代表制在处理不完美一致的投票者偏好时表现出更强的代表性保证。验证稳健Rank-PSC是coNP完全的，而验证稳健Rank-PSC+则可以在多项式时间内完成。

**⚠️ 局限性**

限制在于稳健比例代表制的验证复杂度较高，且在处理不同委员会大小时可能存在计算效率问题，尤其是运行时间依赖于委员会大小的指数级增长。

---

## 640. Fixing the Last Mile: Failure-Aware Image Restoration with Region-Specific Guidance

**arXiv ID:** 2611.26456 | [PDF](https://arxiv.org/pdf/2611.26456v1)

**作者:** Hongyu An `[一作]` (INSAIT, Sofia University St. Kliment Ohridski), Jinjin Gu `[通讯]` (INSAIT, Sofia University St. Kliment Ohridski)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为FAIR（Failure-Aware Image Restoration）的图像修复框架，旨在解决图像修复中的“最后一公里”问题，通过预测局部失败区域并提供相应的修复指导来改善图像质量。

**💡 创新点**

创新点在于将修复过程从全局一致性转向局部感知可接受性，允许对不同区域施加不同的修复指导，从而提高语义准确性和风格一致性。

**🔧 技术方法**

使用了FLUX.2-klein-9B模型，并结合轻量级的LoRA模块，通过区域控制和语义参考来指导修复过程。

**📊 数据集**

构建了一个包含67,000个标注失败区域的大规模数据集，涵盖了不同类型的修复失败，并为每个区域提供了相应的修复指导。

**📈 对比分析**

与五种不使用视觉指导的方法（如SeeSR、SUPIR等）和三种参考基础的方法（如ReFIR+SeeSR等）进行了比较，FAIR在感知质量和目标区域保真度上表现优越，尤其在DINO-Sim-Ref和CLIP-Sim-Ref得分上排名第一。

**⚠️ 局限性**

限制在于FAIR仍然依赖于外部参考图像的质量和相关性，且在某些情况下可能无法完全消除所有局部失败。

---

## 641. Closing Gaps in Temporal Fair Division for Chores

**arXiv ID:** 2611.25892 | [PDF](https://arxiv.org/pdf/2611.25892v1)

**作者:** Nicholas Teh `[一作]` `[通讯]` (University of Oxford), Nicholas Teh (University of Oxford)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了不可分割的任务在规定顺序下的公平分配问题，提出了时间嫉妒自由（TEF1）的概念，并证明了对于三个及以上的代理，某些情况下不存在满足TEF1的分配。

**💡 创新点**

确认了Elkind等人的猜想，即对于每个n≥3和每个有限因子，某些n代理实例不允许存在满足TEF1的分配，并且决定TEF1存在性是NP完全的。

**🔧 技术方法**

使用了复杂性理论和算法设计技术，特别是全多项式时间近似方案（FPTAS）来处理TEF1因子的近似问题。

**📊 数据集**

使用了多个代理和任务的实例，特别是对于n≥3的代理和m≥n+O(log n)的任务，展示了不存在TEF1分配的情况。

**📈 对比分析**

与现有方法比较，发现即使存在TEF1分配，最小化社会成本在多项式时间内无法近似，除非P=NP。提出的FPTAS能够在固定代理数的情况下有效地近似最优TEF1因子，并保证社会成本不超过最便宜的分配。

**⚠️ 局限性**

限制在于对于每个固定的n≥3，TEF1的存在性和最小社会成本的近似问题都是NP完全的，且在某些情况下，保持精确的TEF1可能导致社会成本的无界增加。

---

## 642. GPU-Accelerated Secure Inference for Drug--Target Interaction Prediction

**arXiv ID:** 2611.25664 | [PDF](https://arxiv.org/pdf/2611.25664v1)

**作者:** Sikha Pentyala `[一作]` (University of Washington), Martine De Cock `[通讯]` (University of Washington)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于GPU的安全推断方法，用于药物-靶标相互作用预测，旨在加速药物发现过程，同时保护药物的知识产权。

**💡 创新点**

创新点在于模型和协议的共同设计，优化了深度学习模型以减少对秘密药物的操作，特别是减少了需要通信的非线性操作。

**🔧 技术方法**

使用了安全多方计算（MPC）技术，结合了加法秘密共享和功能秘密共享，并实现了自定义的CUDA内核以加速比较操作。

**📊 数据集**

使用了BindingDB和Davis数据集进行评估，展示了模型在这两个数据集上的性能。

**📈 对比分析**

与其他方法相比，该方法在iDASH 2026隐私与安全竞赛中表现优异，平均每次预测时间低于10毫秒，且安全预测与明文模型的匹配度在0.02%以内。

**⚠️ 局限性**

限制在于该方法依赖于蛋白质和模型权重为公开信息，如果这些信息是保密的，推断的复杂性和通信成本将显著增加。

---

## 643. DRVEXO-4B-Unified: Design and Evaluation of a Modular Speech-Vision System

**arXiv ID:** 2611.25521 | [PDF](https://arxiv.org/pdf/2611.25521v1)

**作者:** Zeyuan Li `[一作]` `[通讯]` (Yujie Intelligence), Zeyuan Li (Yujie Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于4B语言骨干的模块化多模态助手，结合文本、图像和语音输入，支持参考条件的语音输出和结构化交互。

**💡 创新点**

创新点在于将多种输入形式（文本、图像、语音）与任务条件评分头结合，提升了系统在多模态理解和生成方面的能力。

**🔧 技术方法**

使用了Qwen3.5-4B的视觉语言组件、Qwen3-ASR的声学组件和Qwen3-TTS的语音生成技术，结合了时间残差连接器和任务特定评分头。

**📊 数据集**

使用了900个MMMUs验证问题的数据集，以及其他多个标准化测试集（如MMLU-Pro、IFEval、MATH-500等）。

**📈 对比分析**

与Gemma 3 4B IT和InternVL3.5-4B等模型进行比较，系统在所有900个MMMUs验证问题上得分超过了这些模型，且在严格的IFEval和MATH-500测试中表现优异。

**⚠️ 局限性**

限制在于未进行临床干预或自主设备政策的评估，且系统的性能可能受到特定应用场景和输入条件的影响。

---

## 644. Aggron: Aggressive Compression of Agentic KV Caches during Head-of-Line Stalls in the Request Queue

**arXiv ID:** 2611.26435 | [PDF](https://arxiv.org/pdf/2611.26435v1)

**作者:** Rishabh Ravi `[一作]` (University of Illinois Urbana-Champaign), Josep Torrellas `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的服务系统Aggron，用于在请求队列中的头部阻塞期间对代理请求的KV缓存进行压缩，以提高性能。

**💡 创新点**

Aggron通过在每个请求的每个回合进行选择性压缩，最大限度地提高了压缩的成本效益，并且在请求等待期间隐藏了压缩的延迟。

**🔧 技术方法**

使用了基于队列的调度策略，结合了压缩和总结请求的分离队列，以优化内存和计算资源的使用。

**📊 数据集**

在NVIDIA H100 NVL GPU上使用了来自四个基准（Terminal-Bench、MCPMark、DeepResearch Bench和SWE-bench）的真实代理任务数据进行评估。

**📈 对比分析**

与最先进的异步总结方法Slipstream和APEC相比，Aggron在高负载下将P99延迟减少了27%到31%，并将吞吐量提高了1.34到1.40倍。

**⚠️ 局限性**

Aggron的局限性在于它依赖于请求的上下文长度和队列状态，可能在极端负载情况下无法完全隐藏压缩延迟。

---

## 645. Learning Reconstruction Errors as Spatiotemporal Evidence for AI-Generated Video Detection

**arXiv ID:** 2611.26116 | [PDF](https://arxiv.org/pdf/2611.26116v1)

**作者:** Jiahua Jiang `[一作]` (ShanghaiTech University), Jin Zhu `[通讯]` (University of Birmingham)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的检测算法，通过学习和利用重建误差来检测AI生成的视频。

**💡 创新点**

创新点在于联合优化重建器和下游分类器，以生成更具信息量的重建误差，并通过3D卷积直接处理重建误差序列，从而捕捉时空证据，而无需额外的原始视频流。

**🔧 技术方法**

使用了自适应变分自编码器（VAE）和3D卷积网络。

**📊 数据集**

使用了包含69,501个真实视频和69,501个合成视频的VBench数据集，以及多个基准数据集，包括EvalCrafter、GenVideo和FakeParts等。

**📈 对比分析**

与11种现有的AI生成视频检测器进行了比较，结果显示该方法在多个数据集上表现优异，尤其是在EvalCrafter数据集上，平均AUC超过0.99，通常在所有生成器中表现最佳。

**⚠️ 局限性**

限制在于该方法仅能做出视频级别的决策，无法定位AI操控的区域和帧，尤其是在部分篡改的视频中。

---

## 646. UniReFP: Robust Unified Fingerprinting for Vision Models against Cross-Task Repurposing Attacks

**arXiv ID:** 2611.25555 | [PDF](https://arxiv.org/pdf/2611.25555v1)

**作者:** Ziye Geng `[一作]` (University of Houston), Changqing Luo `[通讯]` (University of Houston)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文首次识别了一种新的模型修改攻击——跨任务模型重用攻击，并提出了UniReFP，一个适用于不同任务的统一模型指纹框架，以应对这一挑战。

**💡 创新点**

创新点在于UniReFP利用共享的类级语义空间中的所有权证据，而不是原始任务特定的输出空间，从而实现跨任务的有效所有权验证。

**🔧 技术方法**

使用了代理分类器构建、优化分组指纹、任务感知语义投影机制等技术。

**📊 数据集**

使用了Oxford-IIIT Pet、PASCAL VOC 2012和COCO-Stuff 10K等数据集，涵盖图像分类、目标检测和语义分割等任务。

**📈 对比分析**

与现有的指纹方法（如TFP-M）相比，UniReFP在跨任务重用攻击和常见模型修改攻击下表现出色，AUC值普遍高于0.8，部分设置达到0.97以上，显示出其强大的区分能力。

**⚠️ 局限性**

限制在于尽管UniReFP在多种攻击下表现良好，但在知识蒸馏等攻击下的鲁棒性仍然是一个开放的研究挑战。

---

## 647. GEMS: Generation and Experience Memory for Subgoal-Guided Long-Horizon World Model Planning

**arXiv ID:** 2611.25134 | [PDF](https://arxiv.org/pdf/2611.25134v1)

**作者:** Yangchen Zhao `[一作]` (University of Minnesota), Caiwen Ding `[通讯]` (University of Minnesota)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为GEMS的方法，通过生成时间结构化的子目标来指导长时间规划，解决了在长时间任务中预测误差累积的问题。

**💡 创新点**

GEMS结合了生成的子目标和经验记忆，能够在长时间规划中有效地提高成功率，尤其是在复杂的操作任务中表现优异。

**🔧 技术方法**

使用了生成模型和选择性记忆的技术，结合了时间结构化的子目标生成和基于经验的记忆选择。

**📊 数据集**

在六个任务和任务套件上进行了实验，包括Cube、Push-T、TwoRoom、Reacher和Meta-World等操作任务。

**📈 对比分析**

与现有的最先进的世界模型规划基线相比，GEMS在不同的操作设置中表现更好，Cube任务的成功率为92.8%，Push-T为68.4%，Meta-World为54.1%，分别比最强基线提高了24.0、7.3和22.2个百分点。

**⚠️ 局限性**

GEMS的局限性在于生成的子目标可能会出现不准确的情况，尤其是在经验覆盖不足时，可能无法提供足够的指导。

---

## 648. Static Analysis Guidance for Secure LLM Code Generation

**arXiv ID:** 2611.26312 | [PDF](https://arxiv.org/pdf/2611.26312v1)

**作者:** Daniele Lunghi `[一作]` (University of Luxembourg), Tegawendé'e F. Bissyandé `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了两种策略，通过静态分析器在生成过程中直接将警报传递给模型，以提高大型语言模型（LLM）生成安全代码的能力。

**💡 创新点**

创新点在于引入了Comment和Steering两种黑箱策略，允许在生成过程中实时反馈静态分析器的警报，而不是通过迭代修复循环。

**🔧 技术方法**

使用了静态分析技术，具体包括Semgrep和Bandit作为静态分析器，评估生成代码的安全性。

**📊 数据集**

使用了三个Python基准数据集：SecEvalPlus、BaxBench和RiskyPy，分别用于评估不同的安全性和功能性。

**📈 对比分析**

与PromSec、FDSP、Self-Refine等迭代方法进行比较，结果显示Comment和Steering在生成过程中成本较低，且在特定条件下有效，Steering在RiskyPy上将功能和安全程序的比例从52.5%提高到64.4%。

**⚠️ 局限性**

限制在于静态分析器的范围有限，可能漏掉逻辑、身份验证和配置缺陷；模型在处理需要重构的修复时表现不佳，且模型可能会过度遵循警报，导致程序破坏。

---

## 649. The American Medical Directory Panel: Comparing AI-Assisted and LLM Transcription with Manual Data Entry at Scale

**arXiv ID:** 2611.25218 | [PDF](https://arxiv.org/pdf/2611.25218v1)

**作者:** Christian M. Dahl `[一作]`, Anthony Wray `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究比较了五种机器提取工作流与1912年和1921年美国医学名录的双重录入手动转录数据的准确性，探讨机器转录与手动数据录入之间的相似性。

**💡 创新点**

创新点在于比较了多种提取方法，包括基于规则的解析器和多模态语言模型，评估了它们在处理半结构化历史目录数据时的表现。

**🔧 技术方法**

使用了商业光学字符识别（OCR）技术和大型语言模型（LLM），并开发了基于规则的解析器来提取数据。

**📊 数据集**

使用了1912年和1921年美国医学名录的数据集，包含约140,000和150,000条记录。

**📈 对比分析**

通过与手动转录的双重录入数据进行比较，发现解析器在召回率和精确度上均优于多模态语言模型，结合两者的输出可以提高整体性能。

**⚠️ 局限性**

限制在于手动转录的错误率未知，因此报告的精确度和召回率仅反映与参考数据集的一致性，而非绝对准确性。

---

## 650. Grid Cells as Spectral Successor Features: A Reusable Basis for Continual Reinforcement Learning

**arXiv ID:** 2611.26504 | [PDF](https://arxiv.org/pdf/2611.26504v1)

**作者:** Élodie Côté-Gauthier `[一作]` (McGill University), Isabeau Prémont-Schwarz `[通讯]` (Université Laval)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

论文探讨了持续强化学习中如何利用继承特征（successor features）来提高智能体在任务变化时的适应能力，同时重用其对环境的知识。

**💡 创新点**

创新点在于将继承特征的基础建立在固定参考策略的转移矩阵的特征向量上，从而使得继承特征可以通过线性组合直接计算，而不需要学习。

**🔧 技术方法**

使用了谱方法，特别是随机游走图的拉普拉斯特征向量来构建基础特征。

**📊 数据集**

在一个二维目标网格世界中进行实验，使用均匀随机游走作为参考策略。

**📈 对比分析**

与SimpleSF和DQN相比，基于拉普拉斯特征向量构建的智能体在目标变化时能够保持接近最大回报，而其他方法则在每次变化时损失大部分回报。

**⚠️ 局限性**

限制在于智能体仅在参考策略上进行一步的策略改进，而不是遵循最优策略，这可能影响其在更复杂任务中的表现。

---

## 651. COVERAGE-TRACE: Source-Side Counterfactual Tracing for Detecting Omissions in LLM Summaries

**arXiv ID:** 2611.25539 | [PDF](https://arxiv.org/pdf/2611.25539v1)

**作者:** Ali Karkehabadi `[一作]` (University of California), Rafael E. Banchs `[通讯]` (Intapp)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为COVERAGE-TRACE的框架，用于评估生成摘要中的遗漏，通过替换源文本中的目标事实并测量候选摘要的变化来检测依赖关系。

**💡 创新点**

创新点在于将遗漏评估重新定义为源-摘要依赖性，而不是从响应中恢复缺失的事实，并引入了多重反事实干预以提高检测的准确性。

**🔧 技术方法**

使用了基于教师强制的对数似然性评估技术，通过对目标事实进行控制替换来测量摘要的支持变化。

**📊 数据集**

使用了80对匹配的QMSum数据集，包括多种事实类型，如决策理由、资格、数量等。

**📈 对比分析**

与多个基线模型进行比较，COVERAGE-TRACE在四个开放权重模型上实现了0.764到0.825的AUROC，显著优于最强基线模型0.640的表现。

**⚠️ 局限性**

局限性在于COVERAGE-TRACE对源文本的措辞敏感，使用意义保留的改写会降低AUROC到0.591-0.624，并且评估仅限于80对受控的QMSum对。

---

## 652. Complete-SVD: A Functional Norm Approach to Low-Rank LLM Compression

**arXiv ID:** 2611.26059 | [PDF](https://arxiv.org/pdf/2611.26059v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 653. Tether: Verified Stateful Policy Engine for System-Level Monitoring of Agents

**arXiv ID:** 2611.25599 | [PDF](https://arxiv.org/pdf/2611.25599v1)

**作者:** Meghana Aparna Sistla `[一作]`, Krishnamurthy Dj Dvijotham `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种用于编码代理的状态安全策略的运行时强制执行框架，结合系统调用跟踪推断抽象动作，并在Lean 4中强制执行经过验证的状态安全策略。

**💡 创新点**

通过结合系统调用跟踪和形式化的安全策略，提供了对自主代理的可证明的状态安全保护，尤其是在处理复杂的多步骤工作流时。

**🔧 技术方法**

使用系统调用跟踪和Lean 4中的嵌入式领域特定语言（EDSL）实现抽象动作推断和安全策略的强制执行。

**📊 数据集**

在两个真实的编码代理基准上进行评估，分别是MonitoringBenchHonest和LinuxArena，涵盖了199个对抗性轨迹和218个良性轨迹，以及145个对抗性轨迹和1372个良性轨迹。

**📈 对比分析**

在所有可追踪的轨迹中，达到了100%的对抗性召回率，MonitoringBenchHonest的误报率为2.3%，LinuxArena的误报率为5.0%。与现有的LLM监控器相比，表现出更高的召回率和更低的误报率。

**⚠️ 局限性**

存在一些限制，包括无法防止纯语义攻击、对背景进程和数据库操作的建模不足，以及在某些情况下可能导致误报的过度近似。

---

## 654. Cross-Question Reliable Reinforcement Learning

**arXiv ID:** 2611.25292 | [PDF](https://arxiv.org/pdf/2611.25292v1)

**作者:** Hector G. Rodriguez `[一作]` (TU Darmstadt), Marcus Rohrbach `[通讯]` (TU Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的奖励机制（Cross-question Abstention Reward Estimation），用于训练大型语言模型（LLMs）在回答问题时评估自身的信心，并在不确定时选择放弃或推迟回答。

**💡 创新点**

创新点在于通过跨问题的奖励机制来训练模型，使其能够相对其他问题的信心进行评估，而不是仅仅依赖于自身的正确性标签，从而提高在用户指定风险容忍度下的回答覆盖率。

**🔧 技术方法**

使用了强化学习（RL）和奖励机制来优化模型的信心预测，特别是通过引入覆盖奖励和多样性奖励来增强模型的信心分数。

**📊 数据集**

在多个数据集上进行评估，包括数学数据集（MATH-500、GSM8K、Big-Math-Digits）和多跳维基百科问答（HotpotQA），并在多个基准测试中进行验证。

**📈 对比分析**

与现有的点对点正确性匹配奖励方法相比，该方法在5%风险下的覆盖率从0.8%提高到42.4%，在HotpotQA的外部分布知识基准中，覆盖率从12.3%提高到23.6%。

**⚠️ 局限性**

限制在于尽管该方法在低风险下的覆盖率显著提高，但在绝对值上仍然较低，且模型在高风险环境中的信心预测能力仍需进一步提升。

---

## 655. TipDex-VTLA: Learning When to Refine Actions with Touch for Dexterous Manipulation

**arXiv ID:** 2611.25683 | [PDF](https://arxiv.org/pdf/2611.25683v1)

**作者:** Xudi Ge `[一作]` (Wuhan University), Yongchao Xu `[通讯]` (Wuhan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的视觉-触觉-语言-动作（VTLA）策略TipDex-VTLA，旨在解决在光照变化下的动作漂移问题，通过结合视觉可靠性估计和触觉预测验证来调节执行时的动作精细化。

**💡 创新点**

创新点在于引入了视觉动作敏感性估计（VASE）和执行过程中的触觉预期与修正（TACE），共同调节触觉特征的修正，从而提高在复杂光照条件下的操作成功率。

**🔧 技术方法**

使用了视觉动作敏感性估计（VASE）和触觉预期与修正（TACE）技术，结合多模态基础策略进行执行时的触觉精细化。

**📊 数据集**

在四个真实世界的灵巧操作任务和UniVTAC仿真基准上进行了评估，使用的数据集包括300个演示轨迹，其中200个用于训练基础策略和触觉预测器，100个用于训练VASE和TACE。

**📈 对比分析**

与现有的几种最先进的方法相比，TipDex-VTLA在正常光照下的平均全任务成功率为87.5%，在具有挑战性的未见光照条件下为73.8%，显著优于其他方法。

**⚠️ 局限性**

限制在于当前的可靠性估计仍然依赖于策略的学习，未来的工作将探索更广泛的视觉干扰和任务级反馈，以改善可靠性估计并扩展触觉引导的精细化到更多样化的操作场景。

---

## 656. Mesh-Free Keypoint Learning for Category-Level 6D Object Pose Estimation

**arXiv ID:** 2611.25513 | [PDF](https://arxiv.org/pdf/2611.25513v1)

**作者:** Junghoon Seo `[一作]` (PIT IN Corp), Jaehoon Sim `[通讯]` (PIT IN Corp)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种新的方法MORSE-Pose，用于类别级6D物体姿态估计，通过标准姿态和度量大小注释直接推导出物体感知重建支持。

**💡 创新点**

创新点在于引入了一个姿态和大小包络，具有包含性和不变性属性，能够从类别级姿态和度量大小注释中直接获得物体感知重建支持。

**🔧 技术方法**

使用了多级RGB-D特征、自适应关键点和可学习的类别上下文，结合了物体感知重建的监督。

**📊 数据集**

在REAL275和HouseCat6D数据集上进行了实验，这些数据集包含多种类别和不同的场景、视角、材料和遮挡情况。

**📈 对比分析**

与现有方法进行比较，MORSE-Pose在REAL275上达到了88.8的AP，在HouseCat6D上达到了63.2的AP，表现出强劲的类别级姿态准确性，优于许多基准方法。

**⚠️ 局限性**

限制在于模型的复杂性和对数据集的依赖，可能在某些特定情况下表现不佳，尤其是在极端的视角变化或物体遮挡情况下。

---

## 657. MixReact: Physics-Based Reactive Humanoid Control from Multimodal Cues via a Discrete-Continuous Mixture of Transformers

**arXiv ID:** 2611.26218 | [PDF](https://arxiv.org/pdf/2611.26218v1)

**作者:** Bin Li `[一作]` (ShanghaiTech University), Jingya Wang `[通讯]` (ShanghaiTech University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为MixReact的统一框架，用于基于物理的反应生成，能够从文本、音频和视频提示中生成合适的反应。

**💡 创新点**

MixReact是首个将文本、音频和视频驱动的反应生成与物理执行相结合的框架，采用了离散-连续混合变换器（DisCo-MoT）和历史感知动作变换器（HAT）。

**🔧 技术方法**

使用了离散-连续混合变换器（DisCo-MoT）和历史感知动作变换器（HAT）技术，结合了离散建模和连续预测。

**📊 数据集**

使用了ReactMotionNet数据集进行文本和音频驱动的反应生成，以及ViMo数据集进行视频驱动的反应生成。

**📈 对比分析**

与基线方法相比，MixReact在所有模态上均表现出一致的改进，具有最低的FID、最高的多模态性和多样性，且物理一致性更强。相较于NOUS++，在视频驱动反应中，FID降低了17.5%，滑动错误降低了34.2%。

**⚠️ 局限性**

该方法的局限性在于仍需进一步探索如何在更复杂的环境中实现更自然的反应生成，尤其是在多种模态同时存在的情况下。

---

## 658. KeyMem: Temporal Occupancy Control for Robust Streaming VideoLLM Memory

**arXiv ID:** 2611.25649 | [PDF](https://arxiv.org/pdf/2611.25649v1)

**作者:** Wei Song `[一作]` (Griffith University), Leo Yu Zhang `[通讯]` (Griffith University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为KeyMem的训练无关的包装器，用于控制视频流中的时间占用，限制每个时间段内可保留的视觉标记数量。

**💡 创新点**

创新点在于引入了硬性时间占用上限和可选的密钥采样机制，以减少短时间显著片段对缓存的占用，从而提高模型的准确性。

**🔧 技术方法**

使用了排名归一化、硬性时间上限和密钥采样等技术。

**📊 数据集**

使用了UCF101数据集进行实验，构建了80秒的流，并在其中插入了20秒的压力片段。

**📈 对比分析**

与InfiniPot-V和Nexus Sampling等方法进行了比较，结果显示KeyMem在多个模型和压力条件下的最差准确率显著高于这些方法。

**⚠️ 局限性**

限制在于只使用了一个动作问题进行覆盖测试，且只在Qwen2.5-VL-7B模型上进行了评估，其他模型和数据集尚未测试。

---

## 659. Behavioral Containment: Source-to-Binary Function Matching for 1-Day Vulnerability Detection

**arXiv ID:** 2611.25394 | [PDF](https://arxiv.org/pdf/2611.25394v1)

**作者:** Yifei Yang `[一作]` (Nanchang University), Penghui Li `[通讯]` (Nanyang Technological University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的源到二进制函数匹配方法，通过行为包含性来检测重用的开源代码中的漏洞。

**💡 创新点**

创新点在于将函数表示为行为路径，并通过对齐源路径和二进制路径来估计行为包含性，而不是直接比较代码。

**🔧 技术方法**

使用了二进制反编译技术，并通过行为路径对源和二进制函数进行匹配。

**📊 数据集**

使用了来自GNU Coreutils和Inetutils的122个程序的多个数据集，以及26个1-day CVE的漏洞检测数据集。

**📈 对比分析**

与现有方法相比，该方法在32,008个查询中正确排名的比例为77.87%，而现有方法为45.69%。在1-day漏洞检测中，该方法的F1分数为0.70，而现有方法为0.51。

**⚠️ 局限性**

局限性在于内联函数和小函数可能导致匹配失败，未来可以通过增加对内联调用的权重来改善匹配效果。

---

## 660. Sequential Contract Design for Budgeted Crowdsourcing

**arXiv ID:** 2611.25988 | [PDF](https://arxiv.org/pdf/2611.25988v1)

**作者:** Michal Feldman `[一作]` (Blavatnik School of Computer Science and AI, Tel Aviv University), Inbal Talgam-Cohen `[通讯]` (Blavatnik School of Computer Science and AI, Tel Aviv University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在固定补偿预算下，如何众包任务的设计问题，重点关注工人的努力水平对任务结果质量的影响，并提出了一种基于观察结果的顺序合同设计方法。

**💡 创新点**

创新点在于提出了一种适应性合同设计的近似算法，能够在预算约束下最大化预期工作价值，并探讨了简单合同结构的有效性和局限性。

**🔧 技术方法**

使用了线性规划松弛和随机舍入技术来获得主要的近似结果，并在不同的设置中建立了计算复杂性和近似保证。

**📊 数据集**

研究中使用了多个假设的工人和任务设置，具体数据集未明确给出，但涉及到工人的努力水平和结果的随机性。

**📈 对比分析**

与其他方法的比较显示，当工人数量、每个工人的努力水平或任务结果数量是常数时，可以实现多项式时间的常数因子近似，但在一般情况下，当这三者都不受限制时，计算最优适应性政策是NP难的。

**⚠️ 局限性**

限制在于即使在某些限制条件下，计算最优适应性政策仍然是NP难的，且在工人、行动或结果数量不受限制的情况下，无法实现多项式时间的常数因子近似。

---

## 661. Replication of Agentic Security Events: Audit and Investigation (ROGUE-AI)

**arXiv ID:** 2611.26553 | [PDF](https://arxiv.org/pdf/2611.26553v1)

**作者:** Mengxiao Wang `[一作]`, Nitesh Saxena `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种多代理框架，旨在重建报告的代理事件，并通过匹配的控制组测试报告中指责的原因，从而填补现有研究的空白。

**💡 创新点**

创新点在于首次通过控制复制的方式测试公共报告的代理事件，能够验证报告中指责的原因是否导致了实际的危害。

**🔧 技术方法**

使用了基于大型语言模型（LLMs）的多代理系统，包括Gemini、GPT等模型，构建了一个复制和审计的框架。

**📊 数据集**

使用了多个真实事件的报告作为数据集，具体包括Replit、OpenAI等公司的事件，构建了相应的复制实验。

**📈 对比分析**

与现有方法相比，本文的方法通过控制组的方式进行比较，结果显示在多次实验中，当前模型在报告的事件中重复错误的比例高达50%。

**⚠️ 局限性**

限制在于仅依赖于公开报告的事件，可能导致对某些未被报道的事件的低估，同时复制实验在模拟环境中进行，可能无法完全反映真实环境中的风险。

---

## 662. DeepQuest: From Knowledge Gaps to Scientific Discovery in Alzheimer's Disease

**arXiv ID:** 2611.26343 | [PDF](https://arxiv.org/pdf/2611.26343v1)

**作者:** Chunbo Jiang `[一作]` (Hangzhou Dianzi University), Qiyuan Tian `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了DeepQuest，一个基于知识缺口的AI科学家框架，旨在推动阿尔茨海默病研究，通过将未解决的问题转化为可追踪、可测试的假设，并通过数据可行性筛选和Elo排名进行优先排序。

**💡 创新点**

创新点在于将知识缺口识别、数据约束任务选择、实验验证和知识贡献评估整合为一个统一的研究过程，并通过两级审计机制区分实验可靠性和知识贡献。

**🔧 技术方法**

使用了多阶段筛选、相对优先级排名和两级审计机制等技术。

**📊 数据集**

使用了OASIS-3数据集，并在ADNI和AIBL独立队列中验证了结果。

**📈 对比分析**

与五个语言模型基线进行比较，DeepQuest在15个评估维度中有14个超越了最强基线，科学发现率达到95.24%，显著优于其他方法。

**⚠️ 局限性**

限制在于DeepQuest的有效性依赖于可用数据的质量和范围，且在某些情况下可能无法处理所有类型的知识缺口。

---

## 663. AscMLA: Unlocking FP4 Efficiency for Long-Context MLA Decoding through Asymmetric Scaling

**arXiv ID:** 2611.26106 | [PDF](https://arxiv.org/pdf/2611.26106v1)

**作者:** Zechun Zhou `[一作]` (University of Science and Technology of China), Jing Li `[通讯]` (University of Science and Technology of China)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一种名为AscMLA的高效多头潜在注意力解码方法，利用不对称的K/V缩放和流水线缩放传递直接处理单个FP4负载。

**💡 创新点**

AscMLA通过共享FP4负载和不对称K/V缩放的结合，显著减少了KV缓存的存储需求，并提高了解码速度。

**🔧 技术方法**

使用了FP4格式的多头潜在注意力解码技术，结合了本地低位矩阵乘法和流水线缩放传递。

**📊 数据集**

在实验中使用了Kimi-K2.6和Kimi-K3模型的权重进行评估，涉及多个基准配置。

**📈 对比分析**

与五个代表性基线进行比较，AscMLA在存储、性能和准确性方面表现优异，平均速度提升为1.35倍，相比于TRT-LLM的FP8，使用的KV缓存内存减少了38.2%。

**⚠️ 局限性**

限制在于AscMLA在处理某些输入分布时的准确性可能略低于某些FP8方法，尤其是在稀疏异常值的情况下。

---

## 664. Prescribed Bayesian Efficiency under Lifetime Adjustment Costs

**arXiv ID:** 2611.25311 | [PDF](https://arxiv.org/pdf/2611.25311v1)

**作者:** Viet Bac Nguyen `[一作]` `[通讯]` (VNU University of Engineering and Technology), Viet Bac Nguyen (VNU University of Engineering and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

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

## 665. Serialization Tax in Shared-Latent Exchangeable Decisions

**arXiv ID:** 2611.26029 | [PDF](https://arxiv.org/pdf/2611.26029v1)

**作者:** Siming Zhang `[一作]` (Shanghai University), Hang Yu `[通讯]` (Shanghai University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文探讨了独立边际化对共享潜在状态的影响，提出了序列化税的概念，并分析了如何在决策中保留共享后验不确定性以提高决策质量。

**💡 创新点**

创新点在于首次统一了共享潜在聚合尾部决策的平均化前聚合失败的理论框架，量化了通过精确方差身份和总变差下界所损失的计数不确定性。

**🔧 技术方法**

使用了经典的凸序比较、后验近似、决策后悔分析等技术，并引入了TaxScore来识别决策边界附近的大型隐藏块。

**📊 数据集**

使用了MovieLens 1M和10M数据集进行预先指定的无隐藏目标审计，评估了共享潜在头部的性能。

**📈 对比分析**

与现有的共享潜在头部相比，TaxScore在高分负对数似然（NLL）和效用上分别提高了0.044/0.027和0.031/0.024，整体人口效用提升为0.006–0.012。

**⚠️ 局限性**

限制在于该研究主要集中在二元项目和特定的决策条件下，未来需要扩展到非加性组合目标、变化的人群和端到端路由成本等更广泛的应用场景。

---

## 666. Limits of local curvature for constructing search subspaces in segmentation models

**arXiv ID:** 2611.25822 | [PDF](https://arxiv.org/pdf/2611.25822v1)

**作者:** Maxim Bolshim `[一作]`, Alexander Kugaevskikh `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了局部曲率构建的子空间在语义分割模型中如何表示梯度更新场的变化，测量了在固定子空间中损失增加的比率与全空间的比率。

**💡 创新点**

提出了通过局部曲率和梯度构建的子空间在表示局部响应和场变化方面的有效性，并比较了不同子空间的表现。

**🔧 技术方法**

使用了广义高斯-牛顿矩阵（GGN）和Krylov子空间方法来构建子空间，进行梯度更新和损失恢复的比较。

**📊 数据集**

使用了PASCAL Visual Object Classes (VOC) 2012数据集中的50张图像进行实验，图像大小为320 × 320像素。

**📈 对比分析**

通过比较固定子空间和不同秩的子空间，发现引入GGN方向后，局部响应的表示误差显著降低，但场变化的误差未显著减少。损失恢复比率在搜索半径增大时下降。

**⚠️ 局限性**

局限性在于尽管局部响应的误差几乎消除，但场变化的误差仍然较高，且有限步余量对投影误差的贡献较大。

---

## 667. A Neurosymbolic Multimodal Framework for 30-Day Diabetic Readmission Prediction: Integrating Structured EHR Data, Dense Clinical Embeddings, and Knowledge-Grounded Symbolic Extraction

**arXiv ID:** 2611.25236 | [PDF](https://arxiv.org/pdf/2611.25236v1)

**作者:** Una Suman Kumar Patro `[一作]` (Independent Researcher), Narendra Badam `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种神经符号多模态框架，用于预测糖尿病患者的30天再入院率，结合了结构化电子健康记录（EHR）数据和自由文本临床笔记。

**💡 创新点**

创新点在于融合了三种信息流：工程化的结构化特征、通过截断奇异值分解获得的笔记嵌入，以及通过开放权重临床语言模型提取的符号实体，形成了一种新的混合检索增强生成管道。

**🔧 技术方法**

使用了神经符号人工智能、检索增强生成（RAG）、临床自然语言处理等技术，结合了密集检索、稀疏词汇检索和交叉编码器重排序。

**📊 数据集**

使用了公开的糖尿病130-美国医院数据集，该数据集包含101,766次就诊记录和71,518名患者。

**📈 对比分析**

与结构化基线模型相比，完整的多模态模型在敏感性（0.6667）上提高了8.28个百分点，F_2分数最佳（0.3824），Brier分数最低（0.0977），但AUROC略低（0.6375）。

**⚠️ 局限性**

限制在于使用的临床笔记是通过属性条件模拟生成的，缺乏真实的双语住院数据，且当前知识库仅包含十篇简明的专著，未来需要扩展到其他语言对和更长的预测时间范围。

---

## 668. VisualTalk-Bench: Evaluating Question-Driven Avatar-Led Video Explanations

**arXiv ID:** 2611.25444 | [PDF](https://arxiv.org/pdf/2611.25444v1)

**作者:** Xin Zhang `[一作]` (University of Science and Technology of China), Minglei Li `[通讯]` (Leadde AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的基准VisualTalk-Bench，用于评估由虚拟角色主导的基于问题的视频解释，特别是在金融和中英文教育领域。

**💡 创新点**

创新点在于引入了一个关注虚拟角色与视觉内容之间互动的评估框架，并开发了Avatar-O Agent作为模块化的参考实现。

**🔧 技术方法**

使用了模块化的Avatar-O Agent，结合了交互规划和视频生成技术。

**📊 数据集**

使用了800个手动筛选的参考视频和100个独立评估请求，涵盖金融报告和教育主题。

**📈 对比分析**

与其他五种代理配置进行了比较，Avatar-O Agent在视觉指导、指向准确性和解释有效性方面获得了最高的平均评分。

**⚠️ 局限性**

限制在于尽管Avatar-O Agent在指向准确性方面表现较好，但仍然存在指向错误的问题，表明需要进一步改进视觉目标的定位和手势生成的协调。

---

## 669. Invariance and Expressiveness: Modal Variations

**arXiv ID:** 2611.26244 | [PDF](https://arxiv.org/pdf/2611.26244v1)

**作者:** Martin Otto `[一作]` `[通讯]` (Darmstadt Technical University), Martin Otto (Darmstadt Technical University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文探讨了模态模型理论中的双模拟不变性，强调了其在模态建模中的重要性，并讨论了与第一阶背景逻辑相关的表达完全性现象。

**💡 创新点**

创新点在于通过双模拟不变性来探讨模态逻辑的表达能力，特别是在有限模型理论和其他非初等设置中的技术挑战。

**🔧 技术方法**

使用了双模拟和博弈理论的方法来分析模态逻辑的表达能力，特别是通过挑战/响应的博弈协议。

**📊 数据集**

论文没有具体提到使用的数据集，但讨论了与有限Kripke结构相关的模型。

**📈 对比分析**

通过与传统模型理论的比较，展示了基本模态逻辑在双模拟不变性下的表达完全性，并指出了在有限Kripke结构中，基本模态逻辑是最大表达性的。

**⚠️ 局限性**

限制在于对非初等模型类的分析和表达能力的探讨，尤其是在有限模型理论中，传统的模型理论技术可能不适用。

---

## 670. Lossless Latent Diffusion Models

**arXiv ID:** 2611.25454 | [PDF](https://arxiv.org/pdf/2611.25454v1)

**作者:** Tongda Xu `[一作]` (Apple), Huijuan Huang `[通讯]` (Apple)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的无损潜在扩散模型（LLDM），通过无损变换将像素映射到更适合扩散建模的表示，同时保持像素精确重建。

**💡 创新点**

LLDM通过残差分解和不均匀潜在缩放构建无损变换，提出了语义排名优先级（SRP）作为评估其有效性的指标。

**🔧 技术方法**

使用了残差分解和不均匀潜在缩放技术来构建无损变换。

**📊 数据集**

在ImageNet和Diffusion-Bench数据集上进行了实验，分别用于类到图像生成和文本到图像生成。

**📈 对比分析**

与多种基线方法（包括像素空间扩散模型和有损变分自编码器）进行了比较，LLDM在生成质量上显著优于这些方法，同时保证了像素精确重建。

**⚠️ 局限性**

LLDM的局限性在于其对基础有损变分自编码器的选择敏感，不同的缩放因子可能影响生成质量。

---

## 671. Wearable Logo, Silent Hijack: Stealthy Cross-Modal Semantic Attacks on Visible--Infrared Vision-Language Models

**arXiv ID:** 2611.25437 | [PDF](https://arxiv.org/pdf/2611.25437v1)

**作者:** Qike Zhang `[一作]`, Chengyin Hu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为Logo-guided Semantic Attack (LoSA)的方法，旨在利用现有的标志作为载体，对可见-红外（VIS-IR）视觉语言模型（VLMs）进行协调的目标语义攻击，同时保持标志的原始轮廓。

**💡 创新点**

创新点在于通过优化标志的内部外观和位置，同时保持其轮廓，来实现对VIS和IR模型的协调攻击。

**🔧 技术方法**

使用了基于分数的黑箱优化框架，结合轮廓场和低频图谱谱模式来结构化内部颜色布局，并通过粒子群优化共同优化共享设计参数。

**📊 数据集**

在四个数据集上进行了实验，包括RGBDT500、LasHeR、COCO和CrowdHuman。

**📈 对比分析**

与多种基线方法进行比较，LoSA在同时的Top-1目标成功率上表现出色，且在视觉自然性评分和跨架构实验中也显示出其有效性。

**⚠️ 局限性**

局限性在于尽管LoSA在多个数据集上表现良好，但在真实世界的成像条件下的有效性和鲁棒性仍需进一步研究。

---

## 672. Optimal Robust Community Detection

**arXiv ID:** 2611.25157 | [PDF](https://arxiv.org/pdf/2611.25157v1)

**作者:** Samuel B. Hopkins `[一作]` (Massachusetts Institute of Technology), David Steurer `[通讯]` (ETH Zurich)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在稀疏的双社区随机块模型中，如何在对抗者可能破坏观察图的小部分边的情况下进行社区检测。提出了一种近似线性时间的算法，该算法在存在小比例对抗性破坏的边的情况下，能够以接近信息理论最优的准确度估计社区。

**💡 创新点**

创新点在于首次提出了一种能够在对抗性破坏存在的情况下，仍然实现信息理论最优准确度的多项式时间算法。该算法通过近似非回溯矩阵的第二特征向量来初始化信念传播，并且在没有破坏的情况下，信念传播能够达到最优准确度，解决了Zdeborová在双社区情况下的一个开放问题。

**🔧 技术方法**

使用了半正定规划技术来处理对抗性破坏，并结合了信念传播算法来进行社区检测。

**📊 数据集**

使用了双社区随机块模型（SBM）生成的图数据集，具体参数为d和λ，且要求dλ^2 > 1。

**📈 对比分析**

与之前的算法相比，提出的算法在存在对抗性破坏的情况下，仍然能够达到信息理论最优的准确度。性能上，算法在近似线性时间内运行，并且在每个d和λ的情况下都能保持较高的准确度。

**⚠️ 局限性**

限制在于算法的性能依赖于对抗性破坏的比例，且在某些情况下，可能需要对输入数据进行预处理以确保算法的有效性。

---

## 673. Alignment Drives Reuse: Learning Generalizable Robot-Use Agents from Heterogeneous Experience

**arXiv ID:** 2611.26376 | [PDF](https://arxiv.org/pdf/2611.26376v1)

**作者:** Ziyue Wang `[一作]` (University of Sydney), Liyi Zhou `[通讯]` (University of Sydney)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为UniCraft的框架，通过对异构设置的对齐来学习机器人操作技能，旨在提高技能的共享和重用。

**💡 创新点**

创新点在于通过对齐不同的执行接口和学习案例，实现了跨基准和机器人形态的共享技能集和其调度策略的联合学习与优化。

**🔧 技术方法**

使用了一种统一的环境表示、可组合的技能、可学习的调度策略和规范化的案例表示，结合了多种机器人和基准的交互反馈。

**📊 数据集**

使用了LIBERO-PRO、RoboTwin 2.0、RoboCasa365和RoboDojo等多个基准的数据集，共计收集了2700个学习回合。

**📈 对比分析**

与基线方法相比，UniCraft在多个基准上表现出更好的通用性和零-shot转移能力，尤其是在RoboTwin和RoboCasa365上，成功率显著提高。

**⚠️ 局限性**

限制在于比较方法的公平性，辅助工具和预训练组件的差异可能影响不同方法之间的可比性，同时所有评估均在模拟环境中进行，缺乏对真实机器人转移的验证。

---

## 674. Long-Range Video Presentation Attack Detection: A Three-Class Benchmark Across 1-100 m and Two Sensor Types

**arXiv ID:** 2611.25108 | [PDF](https://arxiv.org/pdf/2611.25108v1)

**作者:** Lambert Igene `[一作]` (Clarkson University), Stephanie Schuckers `[通讯]` (UNC at Charlotte)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种视频基础的面部呈现攻击检测（PAD）基准，涵盖了从1米到100米的多种捕获距离和环境条件，使用两种传感器类型（DSLR和移动传感器）进行数据收集。

**💡 创新点**

首次系统性研究了捕获距离和遮挡对PAD性能的联合影响，并引入了三类标签（真实、欺骗和道具），使得对遮挡特定失败模式的分析成为可能。

**🔧 技术方法**

使用了四种卷积神经网络（CNN）架构进行模型训练和评估，包括MobileNetV3、MobileNet、EfficientNet和ResNet-50。

**📊 数据集**

构建了一个包含374,000多帧的PAD数据集，涉及261个独特身份，数据集涵盖了五个不同的距离（1米、10米、50米、75米和100米），并在室内和室外条件下收集。

**📈 对比分析**

通过与现有的PAD基准进行比较，EfficientNet在整体性能上表现最佳，准确率达到98.48%，ACER为2.01%。在未见攻击数据集上的泛化测试显示出89-94%的准确率，验证了跨领域的鲁棒性。

**⚠️ 局限性**

本研究的局限性在于所评估的骨干网络是已建立的CNN，未考虑变换器架构和生成反欺骗方法的比较。此外，数据集不包括重放或深度伪造攻击，这些攻击类型在长距离下引入了额外的设备特定混淆。

---

## 675. RemoteVLPG: How to Obtain a Specific VLM for Language-Guided Path Generation on Remote Sensing Images?

**arXiv ID:** 2611.25458 | [PDF](https://arxiv.org/pdf/2611.25458v1)

**作者:** Junzhe Dang `[一作]` (Beijing Institute of Technology), Wei Li `[通讯]` (Beijing Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究如何为遥感图像上的语言引导路径生成获取特定任务的视觉语言模型（VLM）。

**💡 创新点**

提出了一种可扩展的方法来构建预训练数据集，并设计了三个基准以支持分层评估，展示了构建的预训练数据集在零样本性能上超过了一些先进的商业VLM。

**🔧 技术方法**

实现了三种广泛使用的VLM架构：基于下一个标记预测（NTP）、基于头的方法和去噪方法。

**📊 数据集**

构建了一个名为RemotePath-175k的大规模任务特定预训练数据集，包含合成和真实数据。

**📈 对比分析**

通过与现有商业模型的比较，展示了所提方法在零样本评估和微调后的性能，结果表明所提方法在某些基准上优于一些商业模型。

**⚠️ 局限性**

预训练数据集可能仍然太小，无法充分发挥NTP等方法的潜力，且与商业模型相比，仍有改进空间。

---

## 676. LSP-RAG: Latent Scoring Perspectives for Knowledge Graph Retrieval-Augmented Generation

**arXiv ID:** 2611.25229 | [PDF](https://arxiv.org/pdf/2611.25229v1)

**作者:** Haochen Liu `[一作]` (University of Virginia), Jundong Li `[通讯]` (University of Virginia)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的知识图谱增强检索框架LSP-RAG，旨在提高多跳推理和知识检索的准确性，特别是在使用轻量级语言模型的情况下。

**💡 创新点**

LSP-RAG通过多视角提示评分模块增强了知识检索的能力，采用了两种类型的软提示（视角特定软提示和上下文软提示），以实现灵活的相关性估计和上下文适应性。

**🔧 技术方法**

使用了轻量级语言模型和软提示技术，结合多视角提示评分机制来评估候选三元组的相关性。

**📊 数据集**

在多个知识图谱问答基准数据集上进行了实验，包括WebQuestionsSP、WikiMovies和MetaQA，涵盖了1跳和2跳推理设置。

**📈 对比分析**

与现有的KG-RAG基线相比，LSP-RAG在所有设置中均表现出色，尤其是在使用轻量级模型时，其性能超过了依赖大型语言模型的系统，如ChatGPT，且推理成本更低。

**⚠️ 局限性**

LSP-RAG的局限性包括：随着潜在评分视角数量的增加，计算成本也会增加；固定的跳数预算可能会错过需要长间接推理路径的证据；该框架主要集中在相对干净和结构良好的知识图谱上，扩展到不完整、嘈杂或大规模异构图仍需进一步研究。

---

## 677. Open Response Explorer: a Structured Collaborative Writing System for Peer-Informed Open-Response Assignments

**arXiv ID:** 2611.25169 | [PDF](https://arxiv.org/pdf/2611.25169v1)

**作者:** Ziwei Gu `[一作]` (Harvard University), Elena Glassman `[通讯]` (Harvard University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Open Response Explorer的界面，旨在通过结构化的想法单元来改善学生在同行反馈开放式作业中的写作和探索体验。

**💡 创新点**

创新点在于将开放式响应的写作和探索重新聚焦于结构化的目的导向单元，并通过共同角色文本段高亮显示来促进比较和理解。

**🔧 技术方法**

使用了结构化写作、结构化探索和同行链接写作等技术，结合了共同角色文本段高亮显示。

**📊 数据集**

在实验中使用了12名参与者的反馈数据，模拟了同行反馈的学习环境。

**📈 对比分析**

通过与基线接口的比较，结果显示Open Response Explorer在写作、探索和学习方面显著降低了认知负担，并提高了用户的满意度和学习效果。

**⚠️ 局限性**

限制在于参与者主要为研究生，未来需要在新手群体中进行验证，同时当前的角色高亮仍依赖人工干预，未来需要进一步自动化。

---

## 678. Self-Orthogonal MDS Principal-Ideal Codes

**arXiv ID:** 2611.25940 | [PDF](https://arxiv.org/pdf/2611.25940v1)

**作者:** Yangcheng Li `[一作]` (South China Normal University), Pingzhi Yuan `[通讯]` (South China Normal University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了由单项式多项式生成的系数向量码的欧几里得自正交性和广义里德-所罗门（GRS）性质，给出了这些码的完整分类。

**💡 创新点**

提出了自正交和GRS的生成器的完整分类，确定了最大自正交长度，并展示了相应的最大长度码是循环或负循环的。

**🔧 技术方法**

使用了多项式的生成和移位技术，结合了线性代数和编码理论中的自正交性和GRS性质。

**📊 数据集**

研究中使用了任意代数闭域上的生成器，特别关注了固定的r和k值下的生成器数量。

**📈 对比分析**

通过与现有的GRS码和非GRS码进行比较，证明了在3≤k<r的情况下，存在无限多的自正交非GRS最大距离可分码，并且这些码在代数闭域上是不可等价的。

**⚠️ 局限性**

限制在于对于固定的r和k，只有有限多个单项式自正交GRS生成器存在，而在k=r的情况下，只有有限多个自对称生成器存在。

---

## 679. Real Cameras are Radiometric

**arXiv ID:** 2611.25265 | [PDF](https://arxiv.org/pdf/2611.25265v1)

**作者:** Ties Harder `[一作]` (Kiel University), James Pritts `[通讯]` (Kiel University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种辐射相机模型，将任何中心几何相机模型提升为辐射模型，保持几何模型的投影，并添加有限光圈、自由形状的焦点表面和通用的渐晕函数。

**💡 创新点**

创新点在于首次将中心相机模型与主要的辐射组件（光圈、焦点和渐晕）结合，并通过标准校准目标的图像联合校准所有这些组件。

**🔧 技术方法**

使用逆渲染技术直接在传感器的相对辐照度上进行校准，避免了特征检测的偏差。

**📊 数据集**

使用了Sony ILME-FX3相机拍摄的RAW图像作为数据集，包含标准校准目标的图像。

**📈 对比分析**

与传统几何校准方法相比，辐射相机在持出高精度目标上的重投影误差更低，且在82%的模型、镜头和目标组合中显著降低了重投影误差。

**⚠️ 局限性**

限制在于当前模型尚未解决光源估计和水的衰减问题，未来工作将集中在这些方面。

---

## 680. Democratizing Cross-Project Knowledge for Transforming Water Governance and Climate Resilience: an Agentic AI Ecosystem

**arXiv ID:** 2611.26398 | [PDF](https://arxiv.org/pdf/2611.26398v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 681. Better Retrieval, Not Better Binding: Pretraining Exposure in Egocentric Video Retrieval

**arXiv ID:** 2611.26120 | [PDF](https://arxiv.org/pdf/2611.26120v1)

**作者:** Yuqi Guo `[一作]` (University of Bristol), Michael Wray `[通讯]` (University of Bristol)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的诊断协议Ego-ExBind，用于分析在自我中心视频-文本检索中的预训练曝光和动词-名词组合绑定。

**💡 创新点**

创新点在于引入了曝光账本和绑定探针，通过控制名词和动词的交换负样本来评估模型的组合理解能力。

**🔧 技术方法**

使用了EgoVLPv2模型，并通过固定训练检查点来进行曝光输入的替换和评估。

**📊 数据集**

使用了EPIC-KITCHENS-100多实例检索任务（EK100-MIR）数据集，包含9,668个查询。

**📈 对比分析**

通过比较不同的曝光条件（原始、零化和打乱），发现曝光重加权可以改善视频-文本检索性能，但对动词-名词绑定没有显著影响。

**⚠️ 局限性**

限制在于模型的检索性能提升并不一定反映出组合理解的改善，未来的评估应区分曝光熟悉度与系统性的组合泛化。

---

## 682. Same Judgement, Different Scores: Numerical Scoring Topology in LLM-as-a-Judge Evaluation

**arXiv ID:** 2611.26077 | [PDF](https://arxiv.org/pdf/2611.26077v1)

**作者:** Doyun Kim `[一作]` (Soongsil University), Chanjun Park `[通讯]` (Soongsil University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文介绍了数值评分拓扑的概念，并系统研究了其对大型语言模型（LLM）作为评估者的影响，特别是在零-shot评估中。

**💡 创新点**

创新点在于首次提出数值评分拓扑作为评估的结构性因素，并探讨其对评估结果的系统性影响，尤其是在判断密集型任务中。

**🔧 技术方法**

使用了大型语言模型（如GPT-5-mini）进行评估，并通过实验设计比较了不同的数值评分拓扑对评估结果的影响。

**📊 数据集**

使用了多个数据集，包括MATH-500、AIME、GPQA Main、LiveCodeBench、Arena-Hard、WildBench和IFEval等，涵盖了数学推理、科学推理、代码生成等任务。

**📈 对比分析**

通过与传统的Likert评分进行比较，发现数值评分拓扑的变化会显著影响评估结果，尤其是在判断密集型任务中，表现出更高的敏感性。强大的评估模型通常在评分拓扑变化下表现出更高的鲁棒性。

**⚠️ 局限性**

限制在于实验未能涵盖所有快速发展的基础模型的多样性，且仅关注点式数值评分，未探讨其他评估协议的敏感性。此外，跨语言分析仅考虑了英语和韩语，未能代表全球语言的多样性。

---

## 683. From Attention to Semantics: Detecting and Mitigating Sound Event Hallucinations in Large Audio-Language Models

**arXiv ID:** 2611.25789 | [PDF](https://arxiv.org/pdf/2611.25789v1)

**作者:** Tianyi Sun `[一作]` (Shanghai Jiao Tong University), Mengyue Wu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了大型音频语言模型（LALMs）中的声音事件幻觉现象，提出了一种无训练的检测与缓解框架，旨在有效识别和减轻幻觉事件。

**💡 创新点**

创新点在于结合注意力分析与潜在语义检索，揭示真实事件与幻觉事件在音频注意力分配和语义支持上的差异，并提出差异化音频掩蔽策略以保护真实事件。

**🔧 技术方法**

使用了LatentLens技术进行潜在语义检索，并结合注意力分析来评估生成事件的语义支持。

**📊 数据集**

实验使用了AudioSet Strong和DCASE2019两个数据集，评估了不同模型在幻觉检测和缓解方面的表现。

**📈 对比分析**

与现有方法（如基于困惑度、熵、注意力得分等）相比，本文的方法在幻觉检测和缓解性能上显著提高，尤其在F1分数和PR-AUC指标上表现优异。

**⚠️ 局限性**

限制在于该方法依赖于音频注意力的分布和语义支持的准确性，可能在某些复杂场景下表现不佳。

---

## 684. Mapping the Wind to Find the Source: Online Physics-Informed Gas Source Localization with Mobile Robots

**arXiv ID:** 2611.25067 | [PDF](https://arxiv.org/pdf/2611.25067v1)

**作者:** Wanting Jin `[一作]`, Alcherio Martinoli `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种轻量级的物理知识驱动的概率框架，用于在空间变化的室内气流下进行高效的气体源定位（GSL）。该框架通过移动机器人在导航过程中持续收集风和气体测量数据，并在线重建空间变化的气流场。

**💡 创新点**

创新点在于结合了轻量级的物理模型与在线气流重建，提出了一种风支持感知的似然公式，以减少在气流重建不充分区域的预测影响，从而提高源定位的鲁棒性。

**🔧 技术方法**

使用了基于自适应扩散方程（ADE）的前向模型进行气体扩散预测，并结合高斯核回归（GKR）进行在线气流重建。

**📊 数据集**

使用了模拟环境中的气体传感器和风传感器数据，进行了高保真模拟实验和物理实验，评估了系统在真实感知、运动和气流条件下的性能。

**📈 对比分析**

通过与传统的气体扩散模型（如GPM和DDPM）进行比较，结果表明，ADE模型在复杂气流环境中提供了更一致的定位性能。结合信息路径规划（IPP）后，框架显著减少了所需的旅行距离，同时保持或提高了成功率。

**⚠️ 局限性**

局限性包括假设存在单一静态源和近似稳定的时间平均气流及气体分布，未来工作可扩展到时间变化的气流、间歇性释放、移动源和多个源的情况。此外，当前的风支持项未能显式传播气流不确定性，未来可考虑更系统的不确定性模型。

---

## 685. Goal-Guided Reinforcement Learning via Fine-Tuning Vision-Language Models

**arXiv ID:** 2611.25742 | [PDF](https://arxiv.org/pdf/2611.25742v1)

**作者:** Yajie Qi `[一作]` (Shanxi University), Jianye Hao `[通讯]` (Tianjin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于视觉语言模型（VLM）的强化学习（RL）框架，通过从Craftax-Classic环境的交互轨迹中学习环境特定的目标指导，来解决稀疏奖励和长时间跨度的探索挑战。

**💡 创新点**

创新点在于构建了一个名为CCGP-2M的大规模多模态目标偏好数据集，并提出了一种两阶段的VLM引导RL框架，结合了离线VLM微调和在线目标引导RL。

**🔧 技术方法**

使用了Kullback-Leibler Token-level Direct Preference Optimization (KLTDPO)技术进行VLM的微调，并在在线阶段利用微调后的VLM生成高层次目标来指导RL探索。

**📊 数据集**

使用的数据集是CCGP-2M，包含200万个偏好样本，涵盖43个不同的目标，涉及Craftax-Classic中的所有22个成就和多样的环境状态。

**📈 对比分析**

与最先进的基线方法相比，提出的方法在收集钻石的速度、得分和奖励上均表现更好，在15个成就上排名第一，其余成就排名第二。

**⚠️ 局限性**

限制在于最深的成就仍然具有挑战性，且评估仅限于Craftax-Classic环境，未来的工作将探讨更具成本效益的数据收集和训练方法，并在更广泛的环境中评估该框架。

---

## 686. Inverse ECG Reconstruction via a Preconditioned Variational Quantum Linear Solver

**arXiv ID:** 2611.25450 | [PDF](https://arxiv.org/pdf/2611.25450v1)

**作者:** Xizhuo Zhang `[一作]`, Bing Yao `[通讯]` (University of Tennessee)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种混合量子-经典框架，用于逆心电图（ECG）重建，通过量子线性求解器解决密集且病态的线性系统。

**💡 创新点**

首次将量子线性求解器应用于逆ECG重建，提出了拉普拉斯正则化预编码程序以改善系统的条件性。

**🔧 技术方法**

使用了变分量子线性求解器（VQLS）和拉普拉斯正则化预编码技术。

**📊 数据集**

使用了来自2007年PhysioNet计算心脏病挑战的数据集，心脏表面离散为256个节点，躯体表面为352个电极。

**📈 对比分析**

与经典线性求解器（CLS）进行比较，VQLS在不同噪声水平下的重建相对误差（RE）显著低于CLS，表明VQLS在重建精度上具有优势。

**⚠️ 局限性**

该方法在量子硬件的实现上仍面临挑战，尤其是在处理病态系统时，量子资源的需求较高。

---

## 687. MemFolder: Structure-Guided Memory Folding for Efficient Agent Memory Management

**arXiv ID:** 2611.26427 | [PDF](https://arxiv.org/pdf/2611.26427v1)

**作者:** Xuyang Liu `[一作]` (Sichuan University), Honggang Chen `[通讯]` (Sichuan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为MemFolder的结构引导内存折叠框架，旨在减少长期代理的记录数量和上下文长度。

**💡 创新点**

创新点在于结合了RecordFolder和ContextFolder两个组件，分别通过证据整合和紧凑表示来优化内存使用。

**🔧 技术方法**

使用了结构引导的内存折叠技术，包括实体基础的锚点选择和时间链的结合。

**📊 数据集**

在LoCoMo、PersonaMem、LongMemEval等三个文本基准和M3-Bench-web、M3-Bench-robot两个多模态基准上进行了实验。

**📈 对比分析**

与现有的记录压缩方法相比，RecordFolder在减少50%记录的情况下，保留了99.0%的原始性能，超越了最强基线1.9个百分点。ContextFolder进一步减少了每条记录的令牌数，同时保持了大部分性能。

**⚠️ 局限性**

限制在于该方法依赖于现有的内存系统，可能无法完全解决所有类型的内存效率问题。

---

## 688. Should Belgians Install an Advanced Energy Management System in Their Electrified Dwelling ?

**arXiv ID:** 2611.26441 | [PDF](https://arxiv.org/pdf/2611.26441v1)

**作者:** Julian Ruddick `[一作]` (Vrije Universiteit Brussel), Maarten Messagie `[通讯]` (Vrije Universiteit Brussel)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究模拟了500个代表性的比利时住宅，评估在电气化背景下安装先进能源管理系统（EMS）的潜在好处。

**💡 创新点**

创新点在于比较了基于模型预测控制（MPC）和基于决策树的机器学习方法（TreeC）的EMS与传统的基于规则的控制器（RBC）之间的性能差异。

**🔧 技术方法**

使用了模型预测控制（MPC）和决策树（TreeC）作为技术手段，评估了它们在动态电价和峰值电价下的表现。

**📊 数据集**

使用了2022年的真实数据集，包括电力消费、热水使用、EV充电会话、光伏生产、天气数据和电价。

**📈 对比分析**

与RBC相比，TreeC、MPC和完美预测的MPC（MPC P）分别实现了82.5€、116.6€和177.4€的年均节省。所有EMS在热舒适度方面表现相似，但MPC在节省成本方面表现更好，而TreeC在EV充电完成率上表现更佳。

**⚠️ 局限性**

限制在于相对于EMS的购买和订阅成本（280-680€和0-360€/年），节省的成本相对较小，可能不利于广泛采用。

---

## 689. FirstCross: Convex Single-Photon 3D Reconstruction from Sparse Views

**arXiv ID:** 2611.26321 | [PDF](https://arxiv.org/pdf/2611.26321v1)

**作者:** Chi-Wei Lee `[一作]` (National Tsing Hua University), Po-Chih Kuo `[通讯]` (National Tsing Hua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为FirstCross的方法，通过少量光子和视角重建三维形状，解决了单光子激光雷达在低光子流情况下的重建问题。

**💡 创新点**

FirstCross通过将光子到达的概率分解为两个可处理的部分，提出了一种无神经网络的解决方案，显著降低了深度误差，并且在每个测试场景、视角数量和光子流水平下均优于现有方法Transientangelo。

**🔧 技术方法**

使用了基于光子到达时间的泊松似然函数，通过在体素网格上解决一系列凸问题来实现重建。

**📊 数据集**

使用了Transientangelo的真实捕获数据集，进行了多种光子流设置的实验，包括全流和低流设置（300到10 PPP）。

**📈 对比分析**

与Transientangelo进行比较，FirstCross在每个测试场景和光子流水平下的平均物体深度误差均低于Transientangelo，尤其是在光子流减少到100倍低于协议最低流时，FirstCross的误差保持在Transientangelo全流误差的一半以下。

**⚠️ 局限性**

限制在于假设物体厚度大于ε带（约24厘米），薄片物体可能会影响重建精度。此外，在高光子流情况下，FirstCross的深度误差可能会高于Transientangelo。

---

## 690. Evolve Where It Matters: Coverage-Guided Self-Evolution of Agent Skills

**arXiv ID:** 2611.25429 | [PDF](https://arxiv.org/pdf/2611.25429v1)

**作者:** Boyin Tan `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Youcheng Sun `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出了一种新的框架，通过指令级覆盖来指导智能体技能的自我演化，帮助智能体在执行过程中修正其技能。

**💡 创新点**

创新点在于将指令级覆盖从技能测试的度量转变为技能自我演化的信号，明确指出技能需要改变的地方。

**🔧 技术方法**

使用了根本原因分析（RCA）代理和覆盖判断机制，结合执行轨迹和验证反馈进行技能修订。

**📊 数据集**

使用了SkillsBench、SkillLearnBench和SWE-Skills-Bench等数据集进行评估。

**📈 对比分析**

与现有的技能自我演化方法相比，该方法在三个基准测试中表现更好，任务解决率提高了25.6%到30.4%，并且去除覆盖后在SkillsBench上的恢复失败率降低了20.8%。

**⚠️ 局限性**

限制在于覆盖标签来自模型，可能会出现错误标记；此外，代理的修订过程消耗更多的计算资源和令牌。

---

## 691. DailySafe: Benchmarking Dynamic Safety Awareness in Everyday Multi-Turn Multimodal Interactions

**arXiv ID:** 2611.26298 | [PDF](https://arxiv.org/pdf/2611.26298v1)

**作者:** Xuankun Rong `[一作]` (Wuhan University), Mang Ye `[通讯]` (Wuhan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个新的基准DailySafe，用于评估多模态大型语言模型（MLLMs）在日常任务中动态安全意识的能力。

**💡 创新点**

创新点在于关注日常任务中隐含的安全风险，强调模型在多轮、多图像交互中识别和应对这些风险的能力。

**🔧 技术方法**

使用了多模态交互的评估方法，结合真实用户对话和安全知识进行基准构建。

**📊 数据集**

基于195个自然场景的多轮、多图像交互案例，涵盖10个日常领域和42个任务类别。

**📈 对比分析**

通过对16个MLLM的评估，发现模型在动态安全意识方面存在显著差异，部分模型在识别和处理安全风险时表现优异，而其他模型则常常忽视前期动作带来的风险。

**⚠️ 局限性**

限制在于现有的安全增强方法对模型的安全意识提升有限，且在某些情况下可能降低模型的有用性。

---

## 692. Communication-Aware UAV Spatio-Temporal Relay Cooperation and Planning in Complex Urban Environments

**arXiv ID:** 2611.25558 | [PDF](https://arxiv.org/pdf/2611.25558v1)

**作者:** Haoyu Zhang `[一作]` (Hong Kong University of Science and Technology), Zhenyu Gao `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种时空中继合作与规划（ST-RCP）框架，利用执行任务的无人机作为临时通信中继，以解决城市环境中无人机通信中断的问题。

**💡 创新点**

创新点在于动态预测通信盲区事件，并通过任务执行无人机的灵活调度来优化中继选择和轨迹规划，而不依赖于额外的通信基础设施。

**🔧 技术方法**

使用了时空资源分配、障碍物感知轨迹重规划和过程灵活性分析等技术。

**📊 数据集**

实验基于香港观塘的真实交通场景和Deliveroo的订单数据集进行评估。

**📈 对比分析**

与基线方法相比，ST-RCP在高峰场景下的下降阶段通信覆盖率提高了29.93%，并且在能耗方面表现出更好的效率，显示出其在通信覆盖和操作成本之间的良好权衡。

**⚠️ 局限性**

局限性在于当前研究仅限于选定城市场景下的模拟评估，未来需要扩展到更广泛的城市环境，并考虑通信条件和无人机轨迹的不确定性。

---

## 693. Measurable Does Not Imply Transportable: A Prospective Audit of Model Explanations in Multi-Year Field Trials

**arXiv ID:** 2611.25692 | [PDF](https://arxiv.org/pdf/2611.25692v1)

**作者:** Rayhaneh Shabani Nia `[一作]` (University of California), Ali Karkehabadi `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究测试了模型解释在未来环境中的可转移性，特别是在农业基因组与环境交互（G×E）建模的背景下，使用多年的玉米试验数据进行分析。

**💡 创新点**

创新点在于提出了前瞻性解释可转移性这一概念，强调了可靠测量和上下文内重现性并不一定能保证解释在未来环境中的可转移性。

**🔧 技术方法**

使用了基于环境测量的回归分析和XGBoost/TreeSHAP模型来构建解释，并通过余弦相似度来评估解释的可转移性。

**📊 数据集**

数据集为Genomes-to-Fields（G2F）玉米多环境资源，包含多年的产量、环境测量和基因组标记信息。

**📈 对比分析**

与传统的非时间验证方法相比，严格的前向评估显示出G×E解释的可转移性较低，表现出显著的乐观差距。主效应解释在未来年份的对齐度明显高于G×E解释。

**⚠️ 局限性**

限制在于研究仅使用了一个玉米资源，且环境概念之间存在相关性，所有发现均为观察性而非因果关系。

---

## 694. Why Do LLMs Help with Scientific Misconduct Requests and What to Do About It?: Increasing the Trustworthiness of AI for Science

**arXiv ID:** 2611.25492 | [PDF](https://arxiv.org/pdf/2611.25492v1)

**作者:** Almene De Meran Meguimtsop `[一作]` (University of Colorado Boulder), Daniel E. Acuna `[通讯]` (University of Colorado Boulder)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究调查了大型语言模型（LLMs）在科学不当行为请求中的表现，特别是它们如何识别和拒绝这些请求。研究使用了SciIntBench基准，分析了8个LLMs在1350个与科学研究诚信相关的提示下的行为。

**💡 创新点**

创新点在于揭示了LLMs在识别不当请求时的高识别率（91%），但拒绝率却相对较低（34%），并探讨了请求特征如何影响拒绝率。此外，提出了通过系统提示干预来提高模型拒绝不当请求的能力。

**🔧 技术方法**

使用了SciIntBench基准和系统提示干预技术，包括意图分析和研究诚信原则自我提醒，以提高模型的拒绝率。

**📊 数据集**

使用了SciIntBench数据集，该数据集包含270个研究场景，涵盖机器学习、生命科学和社会科学等领域，涉及十种研究诚信类别。

**📈 对比分析**

通过与基线比较，意图分析和自我提醒分别将不当请求的拒绝率从58%提高到87%和86%。这些方法在提高拒绝率的同时，对良性请求的错误拒绝率影响较小。

**⚠️ 局限性**

限制在于实验使用的SciIntBench场景相对简单，可能无法推广到更复杂的研究任务。此外，模型在生成响应时的识别能力未在同一交互中评估，可能影响结果的解释。

---

## 695. SGAR: Feed-Forward 3D Reconstruction via Sparse Geometry-Guided Attention

**arXiv ID:** 2611.26493 | [PDF](https://arxiv.org/pdf/2611.26493v1)

**作者:** Aoxiang Fan `[一作]` (École Polytechnique Fédérale de Lausanne), Pascal Fua `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种新的稀疏几何注意力重建方法，替代了前馈重建模型中的密集全局注意力，以提高3D重建的效率和准确性。

**💡 创新点**

创新点在于通过几何相似性引导的稀疏注意力机制，利用学习的投影将每个补丁嵌入，使得观察同一表面的补丁在特征空间中靠近，从而选择相关的关键补丁进行注意力计算。

**🔧 技术方法**

使用了几何引导的稀疏注意力机制和对比损失进行关键选择的学习方法。

**📊 数据集**

在六个数据集上进行了实验，包括ScanNet、Hypersim、Replica等，涵盖室内和室外场景。

**📈 对比分析**

与现有的密集和稀疏方法相比，该方法在1,000个视图下运行速度快12倍，且在所有场景中保持与密集模型相当的准确性。

**⚠️ 局限性**

尽管性能良好，但稀疏化仍然相对于密集注意力带来小幅准确性下降，这是未来研究的一个方向。

---

## 696. Explainable AI for Solar PV Power Forecasting: From Model Selection to Trustworthy, Reproducible Deployment

**arXiv ID:** 2611.25894 | [PDF](https://arxiv.org/pdf/2611.25894v1)

**作者:** Nuzhat Khan `[一作]` (Universiti Teknologi Malaysia), Mohammad Riadh `[通讯]` (Dublin City University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文回顾了用于太阳能光伏（PV）发电预测的主要机器学习（ML）方法，并提出了一种可重复的工作流程，将自动化模型搜索与可解释人工智能结合。该工作流程在葡萄牙波尔图的一个开放数据集上进行了测试，预测了基于数值天气预报网格的日间发电。

**💡 创新点**

创新点在于提出了一种结合自动化机器学习和可解释人工智能的工作流程，并在严格的条件下进行验证，强调了输入特征选择对预测准确性的影响。

**🔧 技术方法**

使用了自动化机器学习（AutoML）和可解释人工智能（XAI）技术，特别是SHAP（Shapley Additive Explanations）方法来解释模型的预测。

**📊 数据集**

使用的数据集是一个位于葡萄牙波尔图的16.32 kWp光伏电站的开放数据集，数据涵盖了2013年4月28日至2016年6月28日的每小时发电量。

**📈 对比分析**

与智能持久性（smart persistence）方法相比，最佳模型在一年测试期内达到了11.6%的标准化均方根误差（nRMSE），约为智能持久性误差的一半。输入特征的选择对准确性的影响远大于强算法之间的选择。

**⚠️ 局限性**

限制在于模型的可解释性和稳定性，SHAP排名在不同时间段的重训练中保持稳定，但在移除相关输入时会发生变化，因此归因应被视为模型的属性而非植物的属性。

---

## 697. Toward a Science of AI Agent Societies

**arXiv ID:** 2611.25404 | [PDF](https://arxiv.org/pdf/2611.25404v1)

**作者:** Geon Lee `[一作]` (KAIST), Kijung Shin `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文探讨了AI代理社会的概念，提出了四个核心属性（个性化目标、规则与治理、自主性、规模与复杂性），并识别了值得研究的四类社会行为（经济行为、利益冲突下的行为、不安全和不道德行为、系统级行为）。

**💡 创新点**

创新点在于将AI代理社会视为一个独立的研究对象，提出了定义有效AI代理社会的核心属性，并强调这些属性如何影响代理的社会行为。

**🔧 技术方法**

使用了社会科学文献中的理论框架，结合复杂性科学和网络理论，提出了对AI代理社会的建模和研究的技术挑战。

**📊 数据集**

论文没有具体提到使用的数据集，而是强调了需要建立可操作的测试平台来研究AI代理社会的行为。

**📈 对比分析**

通过对现有多代理系统的评估，论文指出这些系统通常只满足部分核心属性，缺乏有效的社会行为表现，强调了需要更全面的模型来理解AI代理社会的行为。

**⚠️ 局限性**

限制在于当前的多代理系统大多未能实现有效的个性化目标和自主性，且缺乏足够的规模和复杂性来产生真正的社会现象。

---

## 698. ClosureBench: A Benchmark for Reasoning under Closure Assumptions

**arXiv ID:** 2611.26335 | [PDF](https://arxiv.org/pdf/2611.26335v1)

**作者:** Francesco Belardinelli `[一作]` (Imperial College London), Matteo Leonesi `[通讯]` (University of Camerino)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个基准测试，旨在评估大型语言模型（LLMs）在处理缺失信息时是否遵循明确的闭合合同。

**💡 创新点**

创新点在于引入了闭合切换准确性这一概念，只有当模型在所有变体中都正确回答时，才认为其在基本场景中是正确的。

**🔧 技术方法**

使用了符号逻辑和闭合语义来定义基准测试，并通过生成不同的场景变体来测试模型的推理能力。

**📊 数据集**

使用了960个实例，这些实例基于320个基本场景，涵盖了多个领域，包括云部署、金融控制、医院访问等。

**📈 对比分析**

与现有方法相比，该基准测试能够更清晰地识别模型在处理缺失信息时的错误，模型的整体准确率在71.18%到99.86%之间，但核心切换准确率在18.23%到99.31%之间，显示出显著的差异。

**⚠️ 局限性**

限制在于该基准测试可能无法涵盖所有实际应用场景中的复杂性，且模型在某些情况下可能仍然无法正确处理缺失信息的推理。

---

## 699. ContextRenderer: Dynamic-Context Video Generation for Interactive Game Worlds

**arXiv ID:** 2611.25838 | [PDF](https://arxiv.org/pdf/2611.25838v1)

**作者:** Runze Wang `[一作]` (University of Science and Technology of China), Ligang Liu `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为ContextRenderer的自回归视频扩散框架，用于动态条件下的交互式游戏世界建模，旨在在持续游戏中保持视觉连续性。

**💡 创新点**

创新点在于结合了动态上下文学习（DCL）和递归历史压缩（RHC），以应对变化的世界条件，同时保持视觉连续性，并引入跨表示流对齐（CRFA）来连接不同代理的外观和交互变化。

**🔧 技术方法**

使用了自回归视频扩散模型，结合了动态上下文学习（DCL）、递归历史压缩（RHC）和跨表示流对齐（CRFA）技术。

**📊 数据集**

构建了WorldPrism数据集，该数据集由代理构建的世界、AAA游戏玩法和开源视频组成，提供相应的渲染和块级注释。

**📈 对比分析**

在GWM-Bench和Solar-Bench上的实验表明，ContextRenderer在重建和视觉质量上优于现有的自回归基线，能够在变化的提示和主题参考下实现长达一分钟的交互式渲染。

**⚠️ 局限性**

局限性在于实时交互需要进一步加速推理，当前的视觉历史缺乏对远程观察的显式检索，且需要更广泛的训练数据以覆盖更多样化的世界、交互和视觉风格。

---

## 700. PDE-State Conditioning for Learned Optimization Using Physics-Informed Neural Networks

**arXiv ID:** 2611.25185 | [PDF](https://arxiv.org/pdf/2611.25185v1)

**作者:** Farzana Yasmin `[一作]` (University of Houston), Ricardo Vilalta `[通讯]` (University of Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了如何将偏微分方程（PDE）级别的信息纳入到物理信息神经网络（PINN）的元学习优化器中，提出了一种11维的PDE状态来捕捉操作符结构、残差几何和短期更新响应，并通过有界特征调制来条件化结构化的元学习优化器。

**💡 创新点**

创新点在于引入了PDE状态信息来改善学习优化器的性能，并通过实验确定了最有效的PDE状态特征（G2）和调制方法（FiLM），从而显著提高了优化器的鲁棒性和准确性。

**🔧 技术方法**

使用了元学习优化技术，结合了PDE状态条件化的学习优化器，采用了有界特征调制（FiLM）来调整优化器的隐藏表示。

**📊 数据集**

在实验中使用了线性对流（Linear Advection）、KdV方程和Burgers方程作为数据集进行评估。

**📈 对比分析**

与标准的Adam优化器和未条件化的学习优化器相比，所提出的PDE状态条件化优化器在所有评估指标（相对L_2、L_1、解的均方误差和物理均方误差）上均表现出更好的性能，且在不同的元训练初始化下表现出更低的变异性。

**⚠️ 局限性**

本研究的局限性在于仅在线性对流上进行元训练，并将其转移到KdV和Burgers方程，未来的研究需要在更广泛的PDE家族、其他PDE级别特征和不同的条件机制上进行评估。

---

## 701. C-LiTE: A Lightweight Foundation Model for Cellular feature learning in Histopathology

**arXiv ID:** 2611.26316 | [PDF](https://arxiv.org/pdf/2611.26316v1)

**作者:** Esha Sadia Nasir `[一作]` (University of Warwick), Shan E Ahmed Raza `[通讯]` (University of Warwick)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种轻量级的细胞表示学习基础模型C-LiTE，专注于H&E染色的组织病理学，旨在解决现有模型在细胞级别分析中的局限性。

**💡 创新点**

C-LiTE通过采用原生分辨率的标记策略和伪逆初始化的补丁嵌入方法，显著减少了模型参数数量，同时保持了细胞特征的有效学习。

**🔧 技术方法**

使用了视觉变换器（ViT）架构，结合自监督学习（SSL）方法DINO和iBOT进行预训练。

**📊 数据集**

使用了来自癌症基因组图谱（TCGA）的6000万个细胞中心图像裁剪，涵盖19种癌症类型和16种组织类型。

**📈 对比分析**

与多个基准模型（如CytoFormer、Virchow2和UNI2）进行比较，C-LiTE在多个任务上表现出色，尽管参数数量减少了约30倍，仍接近CytoFormer的性能。

**⚠️ 局限性**

模型的局限性包括未在多种染色和更高分辨率数据上进行测试，且对检测噪声的鲁棒性未进行评估。

---

## 702. Towards Restorative Urban Spaces: An Exploratory Field Study of Nature Videos on Public Multi-Display Signage

**arXiv ID:** 2611.25259 | [PDF](https://arxiv.org/pdf/2611.25259v1)

**作者:** Yuri Mikawa `[一作]` (University of Tokyo), Kazushi Maruya `[通讯]` (NTT Inc.)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究进行了为期九天的探索性实地研究，旨在探讨在公共多显示屏标牌上展示的自然视频对路人的恢复性体验的影响，收集了1263份问卷反馈。

**💡 创新点**

创新点在于提出了一个关于多显示屏标牌（MDS）介导的恢复性体验的理论模型，整合了感知一致性、存在感和恢复感三个因素，并通过实证分析验证了这些因素之间的关系。

**🔧 技术方法**

使用了结构方程模型（SEM）和探索性比较分析等统计技术，结合定性分析对开放式反馈进行编码和分类。

**📊 数据集**

数据集来自于在日本国家未来科学馆的实验，参与者在不同的展示配置下填写问卷，涵盖了多种年龄段的观众。

**📈 对比分析**

通过对不同视频展示配置的比较，发现一致性和恢复感在不同配置间存在适度差异，尽管AI生成的视频在恢复感上并未显著优于广告参考内容。

**⚠️ 局限性**

研究的局限性包括展示条件的分配不均、天气和日类型的混淆，以及参与者的自我选择偏差，这可能影响结果的普遍性和解释。

---

## 703. CoMPASS: A Multi-Turn Benchmark for Measuring LLM Reinforcement of Parasocial Behavior

**arXiv ID:** 2611.25109 | [PDF](https://arxiv.org/pdf/2611.25109v1)

**作者:** Shwetanshu Singh `[一作]` (AI Safety Camp), Scott Blain `[通讯]` (AI Safety Camp)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CoMPASS基准，用于测量大型语言模型（LLMs）是否增强用户的寄生社会依附行为，特别是在多轮对话中。

**💡 创新点**

创新点在于设计了一个多轮基准，能够持续施加寄生社会压力，并将其与控制条件进行比较，填补了现有研究的空白。

**🔧 技术方法**

使用了红队LLM模拟用户与目标LLM之间的对话，采用了七个信号评分标准来评估寄生社会行为。

**📊 数据集**

使用了800个多轮对话的转录数据，这些数据来自8个前沿模型，涵盖4种用户角色和5种升级轨迹条件。

**📈 对比分析**

与现有方法相比，CoMPASS能够捕捉到用户在对话中逐步升级的寄生社会行为，结果显示大多数模型在多轮对话中增强了寄生社会行为，尤其是在同时升级所有三个维度时，六个模型表现出显著的增强效果。

**⚠️ 局限性**

限制在于CoMPASS只测量用户施加的寄生社会压力，而不涉及AI主动引发的伤害。此外，评估者之间的评分一致性较低，可能影响结果的有效性。

---

## 704. Trustworthy Geospatial AI Workflows: Reproducible Maritime Analytics and Forecasting on Open-Source AIS Data

**arXiv ID:** 2611.25884 | [PDF](https://arxiv.org/pdf/2611.25884v1)

**作者:** Divya Acharya `[一作]` (Simula Research Laboratory), Arnaud Gotlieb `[通讯]` (Simula Research Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种可重复的海洋地理空间人工智能工作流程，基于damast库，展示了如何利用公共AIS数据进行海洋轨迹预测和分析。

**💡 创新点**

创新点在于将数据源和观察过程的特性与下游模型性能的关系明确化，推动了可信赖的地理空间AI的发展。

**🔧 技术方法**

使用了damast开源数据处理和机器学习库，以及基于Transformer的船舶轨迹预测模型。

**📊 数据集**

使用了来自BarentsWatch和Marine Cadastre的公共AIS观察数据集。

**📈 对比分析**

通过与传统方法的比较，展示了damast工作流程在元数据感知处理、可重用的数据处理管道和可重复的机器学习实验方面的优势，性能表现良好。

**⚠️ 局限性**

限制在于尚未系统化地表示源级属性（如来源、获取机制和覆盖限制），并且实验仅限于单日AIS观察，缺乏对更长时间段和更广泛观察条件的评估。

---

## 705. TALD: Transferability-Aware Logit Disentanglement for Cross-Modal Knowledge Distillation

**arXiv ID:** 2611.25126 | [PDF](https://arxiv.org/pdf/2611.25126v1)

**作者:** Hui Li `[一作]`, Quan Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为TALD的两阶段框架，用于在类对齐的logit空间中进行可转移性意识的logit解耦，以解决跨模态知识蒸馏中的负迁移问题。

**💡 创新点**

创新点在于通过任务依赖的自利和跨模态排他性标准来指导logit的解耦，并提出了样本自适应调制器来调节解耦约束和样本特定的融合权重。

**🔧 技术方法**

使用了残差解耦器和样本自适应调制器等技术，结合了残差logit解耦和样本自适应约束调节。

**📊 数据集**

在五个跨模态基准数据集上进行了实验，包括AV-MNIST、RAVDESS、VGGSound-50k、CrisisMMD-V2和Fakeddit。

**📈 对比分析**

与九种基线方法进行比较，TALD在84%的评估中相较于无蒸馏训练表现出正向准确率提升，平均准确率提升1.38个百分点，且在十个迁移方向中排名第一。

**⚠️ 局限性**

限制在于该方法可能在某些模态间的任务相关性较弱时表现不佳，尤其是在图像-音频和图像-文本基准上。

---

## 706. Search or Scoring? Diagnosing Why Visual World-Model Planners Fail

**arXiv ID:** 2611.25347 | [PDF](https://arxiv.org/pdf/2611.25347v1)

**作者:** Haoxuan Jia `[一作]` (Nanyang Technological University), Bin Chong `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了视觉世界模型（如DINO-WM和JEPA-WM）在规划过程中可能出现的两种失败模式：搜索失败和评分失败，并提出了相应的解决方案。

**💡 创新点**

创新点在于通过执行优化器的候选动作并比较已知解决方案的成本，来区分搜索失败和评分失败，从而为不同类型的失败提供针对性的修复策略。

**🔧 技术方法**

使用了视觉世界模型（DINO-WM和JEPA-WM）和交叉熵方法（CEM）作为优化器，结合了执行候选动作和评分已知解决方案的技术。

**📊 数据集**

使用了多个模拟环境和数据集，包括PushT、Wall、Metaworld和RoboCasa，特别是在PushT中使用了64个训练窗口和96个官方评估窗口。

**📈 对比分析**

通过比较不同方法的成功率，发现扩大搜索范围在搜索失败的情况下有效，而在评分失败的情况下，平均化窗口的使用效果更好。具体来说，在Metaworld中，平均化窗口解决了127个新任务，而模型选择的窗口最多只能解决87个任务。

**⚠️ 局限性**

限制在于研究主要基于两个简单的2D任务（PushT和Wall），所有环境均为模拟，且大多数模型共享DINOv2特征。此外，评分标签依赖于已知解决方案，且在Metaworld的测试未使用官方的长时间协议。

---

## 707. The Density of Interpolants in Modal Logic: First Results

**arXiv ID:** 2611.26091 | [PDF](https://arxiv.org/pdf/2611.26091v1)

**作者:** Roman Kontchakov `[一作]` (Birkbeck University of London), Frank Wolter `[通讯]` (University of Liverpool)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文对命题模态逻辑中的插值进行了定量研究，测量了承认插值的公式或公式对的密度，分析了它们的渐近行为。

**💡 创新点**

创新点在于引入了插值密度的概念，通过对公式进行逻辑等价的计数，揭示了插值失败的局部性，并提供了局部密度的乘积分解。

**🔧 技术方法**

使用了逻辑等价的计数方法，结合了细胞不兼容图和插值密度的渐近分析。

**📊 数据集**

研究中使用了有限签名下的有界模态深度的公式集，具体数据集未明确给出。

**📈 对比分析**

通过细胞分解原理，比较了不同模态逻辑的插值密度，发现插值失败的概率在某些条件下趋近于零，且在某些情况下，插值密度趋近于1。

**⚠️ 局限性**

限制在于对不同模态逻辑的插值性质的普遍性分析仍然开放，特别是在处理遗忘变量的存在时，插值的行为可能会更加复杂。

---

## 708. Question, Then Affirm: Restoring Commitment Does Not Restore Refusal

**arXiv ID:** 2611.25929 | [PDF](https://arxiv.org/pdf/2611.25929v1)

**作者:** Haoyang Xu `[一作]` (University of Hawai'i), Liuwan Zhu `[通讯]` (University of Hawai'i)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究发现图像生成的安全措施可以在不改变信息内容的情况下被绕过。通过将有害命题以问题形式表达并附加一个勾选标记，生成的图像仍然支持原始主张。

**💡 创新点**

提出了延迟语义承诺（Deferred Semantic Commitment, DSC）的概念，表明问题的形式可以撤回承诺，而后续的肯定元素可以恢复承诺，但安全行为仅跟随承诺的撤回。

**🔧 技术方法**

使用了开放权重的语言模型和图像生成模型（如gpt-image-2）进行实验，分析了问题和肯定提示对生成结果的影响。

**📊 数据集**

使用了MPUP和HateCheck数据集，分别包含了多种类型的图像生成提示，包括仇恨言论和其他有害内容。

**📈 对比分析**

通过比较问题-肯定请求与直接陈述的生成结果，发现问题-肯定请求的生成率显著高于直接陈述，且生成的图像在语义上仍然支持原始命题。

**⚠️ 局限性**

研究的局限性包括：挑战集仅反映未返回图像的输入，样本量有限，且机制分析基于可访问的语言模型，而非实际部署的图像生成器的内部机制。

---

## 709. Generative News Recommendation Reproduces Popularity: A Memorization-Based Analysis

**arXiv ID:** 2611.25606 | [PDF](https://arxiv.org/pdf/2611.25606v1)

**作者:** Shotaro Ishihara `[一作]` `[通讯]` (Nikkei Inc.), Shotaro Ishihara (Nikkei Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了基于生成的新闻推荐系统如何通过训练数据的记忆化机制重现热门内容，导致信息多样性的降低。

**💡 创新点**

创新点在于将生成推荐中的流行性偏见与字符串级记忆化联系起来，提出了一个机制模型来解释输出集中现象。

**🔧 技术方法**

使用了指令调优的Llama 3 8B模型，并通过低秩适应（LoRA）技术进行持续训练。

**📊 数据集**

使用了来自日本新闻服务的约2000万次浏览会话数据集。

**📈 对比分析**

通过与观察到的标题分布进行比较，生成的标题分布更集中且多样性较低，验证了流行性与记忆化之间的关系。

**⚠️ 局限性**

限制在于研究仅基于一个日本新闻服务的数据，且分析为相关性而非因果关系，模型的可重复性和外部有效性需要进一步验证。

---

## 710. LIFT: Lifting Language-Controlled Image Fusion to Free-Viewpoint 3D Scenes

**arXiv ID:** 2611.25696 | [PDF](https://arxiv.org/pdf/2611.25696v1)

**作者:** Jingning Tian `[一作]` (Central China Normal University), Jiayi Ma `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的语言控制的3D图像融合框架LIFT，扩展了传统的多模态图像融合方法，使其能够在自由视角的3D场景中进行语言引导的融合。

**💡 创新点**

创新点在于通过语义模态绑定(SMB)和语言条件高斯外观(LCGA)实现了在3D场景中持久的语义与空间对应关系，从而支持多样化的感知目标。

**🔧 技术方法**

使用了语义模态绑定(SMB)和语言条件高斯外观(LCGA)技术，结合了3D高斯场景表示和语言指导的外观建模。

**📊 数据集**

构建了一个多视角语言监督(MVLS)数据集，该数据集将语言指令和校准视角与调制的融合图像关联起来。

**📈 对比分析**

与五种2D融合方法和三种级联3DGS+2D-IF管道进行了比较，LIFT在大多数指标上表现最佳，显示出其在多模态融合和场景感知方面的优势。

**⚠️ 局限性**

限制在于当前方法可能在处理复杂场景或极端视角时仍然面临挑战，且对语言指令的理解和处理能力依赖于训练数据的质量和多样性。

---

## 711. Complete Test Suites for Effectful Automata

**arXiv ID:** 2611.26535 | [PDF](https://arxiv.org/pdf/2611.26535v1)

**作者:** Bálint Kocsis `[一作]` (Radboud University), Jurriaan Rot `[通讯]` (Radboud University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

提出了一种基于效果自动机的W方法的框架，用于构建和分析测试套件，特别是针对非确定性和加权自动机。

**💡 创新点**

创新点在于将W方法推广到更广泛的自动机类别，特别是效果自动机，并在Eilenberg-Moore类别中证明了其完整性定理。

**🔧 技术方法**

使用了分类理论中的Eilenberg-Moore类别和Kleisli类别，结合了效果单子来建模效果自动机。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了非确定性自动机和加权自动机的实例。

**📈 对比分析**

通过构建效果W方法，展示了如何为非确定性和加权自动机推导完整的测试套件，性能上优于传统方法，尤其在处理状态覆盖和状态特征方面更为高效。

**⚠️ 局限性**

限制在于该方法的适用性可能受到特定类别的结构限制，且在处理概率自动机时需要额外分析以应对实际中的不确定性。

---

## 712. DfD-GS: Defocus Image Prior for Metric-Scale Single-View Depth Estimation and View Synthesis

**arXiv ID:** 2611.26072 | [PDF](https://arxiv.org/pdf/2611.26072v1)

**作者:** Shiori Ueda `[一作]` (Keio University), Shohei Mori `[通讯]` (University of Stuttgart)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为DfD-GS的前馈框架，通过单视角的全聚焦-失焦图像对预测度量尺度的3D高斯场景表示，用于深度估计和新视角合成。

**💡 创新点**

创新点在于利用失焦模糊作为物理度量深度线索，结合单目深度先验，改进了度量尺度的深度估计和新视角合成。

**🔧 技术方法**

使用了前馈神经网络（U-Net）来估计基于失焦的度量差异图，并结合了单目深度基础模型的全局结构先验。

**📊 数据集**

使用了合成失焦数据集，包括Booster、Middlebury和WildRGB-D数据集，以及真实场景的光学失焦图像对。

**📈 对比分析**

与现有的单图像基线方法相比，DfD-GS在Booster和Middlebury数据集上在度量深度估计和新视角合成任务中表现优越。在WildRGB-D上，DfD-GS在AbsRel和LPIPS上表现更好，但在深度RMSE上较高，显示出在视场变化下更稳定的度量尺度。

**⚠️ 局限性**

限制在于最近聚焦假设限制了捕获条件，残余深度误差和未观察表面的不确定性限制了新视角渲染质量。未来的工作应更好地保留可靠的单目几何形状，并改善隐藏高斯的预测。

---

## 713. AdaptReg: Skip-level FiLM Adapter for Task-Specific Adaptation of a Frozen Registration Foundation Model

**arXiv ID:** 2611.25761 | [PDF](https://arxiv.org/pdf/2611.25761v1)

**作者:** Shuyue Shi `[一作]` (Shanghai Medical Image Insights Intelligent Technology), Lin Tian `[通讯]` (Hong Kong Baptist University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种名为AdaptReg的方法，用于适应冻结的注册基础模型，以满足纵向PSMA-PET/CT注册的特定任务需求，确保肿瘤体积的保留。

**💡 创新点**

创新点在于通过轻量级的可训练适配器分支和身份保留初始化，结合FiLM风格的每体素仿射调制，来实现任务特定的适应。

**🔧 技术方法**

使用了FiLM风格的每体素仿射调制技术，并结合了uniGradICON作为冻结的基础模型。

**📊 数据集**

使用了来自PSMA-PET-CT-Lesions数据集的597个PSMA PET/CT研究，包含146名患者的配对数据。

**📈 对比分析**

与uniGradICON + 实例优化基线进行比较，AdaptReg在官方验证集上取得了DSC 0.7585，HD95 6.33 mm，MTV/TLG误差分别为1.68%和2.11%，相较于基线提高了0.05 DSC，并将MTV/TLG误差分别减少了70.9%和67.7%。在隐藏的官方测试集上，AdaptReg在16个团队中排名第四。

**⚠️ 局限性**

限制在于当前方法可能需要进一步改进变形场的平滑性和规则性，以提高代谢量化的准确性，未来的工作将探索将PET图像纳入条件分支以改善小病灶的定位。

---

## 714. LoopedWAM: Looped World Action Models for Elastic Inference

**arXiv ID:** 2611.25595 | [PDF](https://arxiv.org/pdf/2611.25595v1)

**作者:** Anh Dao `[一作]` (University of Central Florida), Yuzhang Shang `[通讯]` (University of Central Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种紧凑的世界动作模型（WAM），通过在每个专家中重复应用共享的六块核心结构，执行30个变换器层，同时每个专家仅存储12个块，从而提高了部署效率。

**💡 创新点**

创新点在于通过权重共享的递归结构将存储的参数与执行的深度解耦，并引入弹性自蒸馏技术，使得模型能够在不同的动作深度和采样步骤预算下进行训练。

**🔧 技术方法**

使用了权重共享的变换器和弹性自蒸馏技术。

**📊 数据集**

在LIBERO、RoboTwin 2.0和RoboCasa-GR1等数据集上进行了评估。

**📈 对比分析**

与Fast-WAM相比，模型在相同的推理测量下使用了7.4倍更少的峰值GPU内存，并在LIBERO上达到了96.1%的成功率，RoboTwin 2.0上为91.9%，RoboCasa-GR1上为52.2%。在LIBERO上，模型在27.6毫秒内达到了96.0%的成功率，显示出多个准确性-延迟操作点。

**⚠️ 局限性**

模型的局限性在于，减少动作专家的执行深度可能会导致性能下降，因此需要明确的预算感知训练来保持竞争力。

---

## 715. BOOMerang: Autonomous Translation from RTL to Microarchitectural Simulators and Back

**arXiv ID:** 2611.25438 | [PDF](https://arxiv.org/pdf/2611.25438v1)

**作者:** Ferran Hermida-Rivera `[一作]` (University of California, Berkeley), Sagar Karandikar `[通讯]` (University of California, Berkeley)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了一种名为BOOMerang的自动化系统，用于在RTL（寄存器传输级）和微架构模拟器之间进行自动翻译和对齐，解决了高层次微架构模拟器与实际硬件设计之间的对齐问题。

**💡 创新点**

创新点在于设计了两个可重用的智能循环：RTL2Sim和Sim2RTL，前者用于生成与RTL实现高度一致的gem5模拟器，后者用于将高层次模拟器中的新特性准确翻译回RTL，从而提高处理器性能。

**🔧 技术方法**

使用了智能代理技术，结合CHIA框架，构建了自动化的RTL与高层次模拟器之间的工作流，能够自动生成和校准微架构模拟器，并将新特性翻译为RTL实现。

**📊 数据集**

使用了Berkeley Out-of-Order Machine (BOOM)核心的RTL设计和SPECint 2017基准测试套件进行评估。

**📈 对比分析**

与传统手动对齐方法相比，RTL2Sim在仅用两天内生成的gem5模拟器的IPC（每周期指令数）误差在3%以内，Sim2RTL成功将72个特性翻译为RTL，且在大多数基准测试中保持低误差，合并后的设计在SPECint 2017上提高了28.9%的IPC。

**⚠️ 局限性**

限制在于某些特性在物理实现上不可行，且在翻译过程中可能会引入新的错误，需要额外的调试和优化步骤来确保设计的可实现性和性能。

---

## 716. AI-Induced Social Erosion: Using LLMs for Work Tasks Reduces Help-Seeking Intentions Towards Colleagues

**arXiv ID:** 2611.25269 | [PDF](https://arxiv.org/pdf/2611.25269v1)

**作者:** Kevin Chow `[一作]` (New York University), Vedant Das Swain `[通讯]` (New York University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了生成性人工智能（AI）在知识工作中的使用如何影响同事之间的关系，特别是帮助寻求意图。通过在线实验，248名知识工作者评估了不同工作任务中AI使用的场景，发现AI的使用减少了帮助寻求意图，尤其是在头脑风暴任务中。

**💡 创新点**

创新点在于通过实验性小品研究，系统地评估了AI使用对同事关系的影响，特别是帮助寻求行为的变化，并识别了不同任务类型和AI使用方式对这些关系的影响。

**🔧 技术方法**

使用了线性混合效应模型分析数据，结合实验小品方法来评估AI使用对同事关系的影响。

**📊 数据集**

研究使用了自定义设计的在线平台，参与者评估了2444个小品场景，这些场景涉及AI在不同工作任务中的使用。

**📈 对比分析**

与非AI工作相比，AI工作显著降低了帮助寻求意图，尤其是在头脑风暴任务中。可见的AI使用与更高的帮助寻求意图相关，而个体贡献者的同事关系比管理者更脆弱。

**⚠️ 局限性**

限制在于本研究主要基于意图而非实际行为，且样本仅限于美国，可能影响跨文化的普遍性。此外，未能深入探讨AI使用对帮助寻求的潜在机制。

---

## 717. SpecFlow: Flow Matching in the Acceleration Spectrum for Multi-Agent Trajectory Forecasting

**arXiv ID:** 2611.26390 | [PDF](https://arxiv.org/pdf/2611.26390v1)

**作者:** Ziyang Xu `[一作]` (National Space Science Center), Hai Zhou `[通讯]` (National Space Science Center)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种条件流匹配框架SpecFlow，用于多模态轨迹预测，能够在不牺牲动态一致性的情况下覆盖多样的未来行为。

**💡 创新点**

创新点在于将未来加速度的离散余弦变换（DCT）系数作为生成状态，并通过固定的解析积分重建速度和位置，同时引入端点预测器来表示目的地的不确定性。

**🔧 技术方法**

使用了条件流匹配（CFM）技术和离散余弦变换（DCT）来处理加速度谱。

**📊 数据集**

使用了ETH/UCY、NBA和斯坦福无人机数据集（SDD）进行评估。

**📈 对比分析**

与其他方法相比，SpecFlow在ETH/UCY场景中实现了最先进的结果，四秒的ADE和FDE在NBA中表现出色，且在SDD上保持竞争力。

**⚠️ 局限性**

限制在于SpecFlow仍然是一个仅基于运动的预测器，未考虑场景图像、可行区域先验或语义目的地线索，这可能影响在某些场景中的表现。

---

## 718. Frequency-Synchronized Boundary Coupling for Training-Free Multi-Prompt Long Video Generation

**arXiv ID:** 2611.26098 | [PDF](https://arxiv.org/pdf/2611.26098v1)

**作者:** Yipan Xu `[一作]` (Fudan University), Wenqiang Zhang `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为FreqSync的训练无关框架，用于多提示长视频生成，解决了现有方法在提示边界处理中的频率无关耦合问题。

**💡 创新点**

创新点在于将提示过渡视为频率选择性边界耦合，结合了频率同步源条件注意力（FS-SCA）和残差裁剪光谱拼接（RCSS），实现了平滑的过渡和强的提示对齐。

**🔧 技术方法**

使用了频率同步源条件注意力（FS-SCA）和残差裁剪光谱拼接（RCSS）技术。

**📊 数据集**

在一个包含130个案例的多提示基准数据集上进行了实验。

**📈 对比分析**

与现有的多提示和长视频生成基线相比，FreqSync在过渡平滑性、时间一致性、提示对齐和感知视频质量上表现出色，取得了最强的整体性能。

**⚠️ 局限性**

限制在于该方法仍然依赖于预训练的短视频生成器，可能在某些复杂场景下表现不如专门训练的长视频生成模型。

---

## 719. Twindex: Index Cache Offloading with Coarse-to-Fine Selection for Long-Context Sparse-Attention Serving

**arXiv ID:** 2611.25078 | [PDF](https://arxiv.org/pdf/2611.25078v1)

**作者:** Yuxuan Xia `[一作]` (New York University), Dongsheng Yang `[通讯]` (Princeton University)

**通讯引用:** 6043 | [OpenAlex ID](https://openalex.org/A5040405791)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的索引器缓存压缩方法Twindex，用于稀疏注意力模型，以提高在长上下文（如百万个标记）下的解码性能。

**💡 创新点**

创新点在于将候选筛选与最终选择分离，使用压缩的键在HBM中进行筛选，并在主机内存中使用原始索引键进行重新评分，从而无需重新训练模型。

**🔧 技术方法**

使用了压缩技术和索引热缓冲区，结合了低精度存储和模型自身的索引器进行评分。

**📊 数据集**

在Qwen、V4.1和GLM模型上进行了评估，使用了长达1M标记的上下文数据集。

**📈 对比分析**

与HiSparse-Opt进行比较，Twindex在Qwen、V4.1和GLM上分别实现了1.44、1.25和2.54倍的解码吞吐量，且在所有评估中，平均基准分数最多低于基线3.1%。

**⚠️ 局限性**

限制在于压缩后的索引可能会影响选择的准确性，且在某些情况下，可能需要更多的候选项来保持高召回率。

---

## 720. SheetAlign: Grounded Interaction and Stage-Isolated Learning for Spreadsheet Agents

**arXiv ID:** 2611.26069 | [PDF](https://arxiv.org/pdf/2611.26069v1)

**作者:** Yuxiang Wang `[一作]` (University of Melbourne), Jianzhong Qi `[通讯]` (University of Melbourne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种电子表格操作框架，结合了任务引导的检查、阶段隔离的策略学习和执行一致性，以提高电子表格任务的自动化处理能力。

**💡 创新点**

创新点在于引入了基于任务的接口和阶段隔离的学习策略，优化了程序生成和审查过程，同时通过执行一致性选择提高了结果的可靠性。

**🔧 技术方法**

使用了阶段隔离的策略学习和执行一致性选择技术，结合了任务引导的检查和API基础。

**📊 数据集**

使用了SpreadsheetBench和Domain-Spreadsheet两个数据集，前者包含912个复杂的电子表格操作任务，后者包含1660个任务，涵盖多个领域。

**📈 对比分析**

与现有基线相比，框架在SpreadsheetBench上的平均准确率从13.5%提高到32.8%，在Domain-Spreadsheet上达到30.1%，显著优于其他方法。

**⚠️ 局限性**

局限性在于该研究专注于使用紧凑的开放权重LLM进行电子表格操作，可能无法处理复杂的Excel特性或外部集成，同时执行一致性引入了额外的推理成本。

---

## 721. Fast-JEPA: Advancing the Success-Latency Frontier for Latent World Models

**arXiv ID:** 2611.25927 | [PDF](https://arxiv.org/pdf/2611.25927v1)

**作者:** Yao Zhang `[一作]` (Wuhan University), Kai Chen `[通讯]` (Zhongguancun Institute of Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的潜在世界模型规划框架Fast-JEPA，旨在通过有效的动作序列搜索来降低规划成本，同时提高成功率。

**💡 创新点**

Fast-JEPA通过重组预测计算，推动了成功率与延迟的帕累托前沿，能够在较低的计算预算下进行有效的动作序列搜索。

**🔧 技术方法**

使用了潜在世界模型和目标导向的计划构建方法，结合直接终端预测来评估完整候选方案，避免了递归状态展开。

**📊 数据集**

在PushT、Reacher、Cube和TwoRoom等多个任务上进行了实验，并引入了DynaGoal基准来评估动态环境下的规划性能。

**📈 对比分析**

与现有方法（如INTACT）相比，Fast-JEPA在四个任务上实现了96.50%的平均成功率，规划速度比INTACT快2.41倍，且在不同候选数量和细化深度下保持竞争力。

**⚠️ 局限性**

在动态环境和执行时间限制下的适应性和鲁棒性仍需进一步研究，尤其是在响应延迟对控制质量的影响方面。

---

## 722. Collision-Free Is Not Enough: Passage-Conditioned Navigation for Wheeled-Legged Robots

**arXiv ID:** 2611.25788 | [PDF](https://arxiv.org/pdf/2611.25788v1)

**作者:** Wai Ka Sun `[一作]` (Columbia University), Bowen Wang `[通讯]` (Columbia University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出了一种新的路径导航方法，称为通道条件导航，旨在帮助腿式机器人在复杂地形中安全通过楼梯和边缘。

**💡 创新点**

创新点在于通过存储每个过渡的进入条件，而不仅仅是路径，来提高机器人在未知环境中的适应能力。

**🔧 技术方法**

使用了基于路径的规划和实时控制技术，结合了机器人自身的传感器和控制器。

**📊 数据集**

使用了一个真实的多障碍物环境数据集，包含271米的多种地形和7米的高度变化。

**📈 对比分析**

与传统方法相比，通道条件导航在处理干扰的边缘交叉时，成功率从24.1%提高到88.6%。在180次爬升中，机器人在遵循存储方向时没有跌倒，而忽略存储方向时则有9次跌倒。

**⚠️ 局限性**

限制在于该方法依赖于一次教学会话的质量，且在极端环境或未记录的情况下可能表现不佳。

---

## 723. Time Series Structure Learning with Local Causal Parent Discovery

**arXiv ID:** 2611.26232 | [PDF](https://arxiv.org/pdf/2611.26232v1)

**作者:** Jacqueline Maasch `[一作]` (Cornell Tech), Zhiguang Wang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种新的因果结构学习算法，称为迭代时间父发现（ITPD），用于处理多变量时间序列中的因果非平稳性问题。

**💡 创新点**

ITPD采用局部到全局的学习方法，通过逐步发现每个变量的父集来构建全局图，利用时间结构先验和局部马尔可夫条件来减少高阶条件独立性测试的依赖。

**🔧 技术方法**

使用了约束基础的因果发现算法，结合了图划分和局部马尔可夫条件的技术。

**📊 数据集**

使用了合成和半合成的数据集进行实验验证，特别是在具有1000到2000个节点的图上进行基准测试。

**📈 对比分析**

与基线方法（如IAMB、DF-PC和PCMCI）相比，ITPD在F1分数和运行时间上表现优越，尤其在大图上运行速度快2.3倍到40.9倍。

**⚠️ 局限性**

ITPD在某些设置下并不总是优于所有替代方法，缺乏公共的真实世界基准数据限制了其在实际数据上的评估。

---

## 724. Disentangling Monitor Hardware Artifacts and Smoothing Lag for Shift-Robust Intraoperative EEG and Hemodynamic Risk Calibration

**arXiv ID:** 2611.25765 | [PDF](https://arxiv.org/pdf/2611.25765v1)

**作者:** Zeyu Jia `[一作]` `[通讯]` (Tianjin Medical University), Zeyu Jia (Tianjin Medical University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

开发了一种因果信号调节和抗偏移的符合校准框架，旨在解决商业手术中脑电图（EEG）和血压监测仪引入的周期性阻抗脉冲等问题。

**💡 创新点**

创新点在于提出了一种三支柱流媒体框架，能够有效分离监测仪器的伪影，并在多种麻醉程序中实现实时风险预警。

**🔧 技术方法**

使用了流媒体滤波器、深度序列模型、患者聚类的子群协变量加权符合风险控制（CRC）和滞后缓冲的在线符合比例-积分-微分（PID）跟踪技术。

**📊 数据集**

分析了5439个公共麻醉程序的数据集，包括3214个全静脉麻醉（TIVA）、2046个挥发性麻醉、171个欧洲内镜镇静和8个直流耦合手术的记录。

**📈 对比分析**

与深度编码器相比，去阻抗的皮层特征在未见的TIVA手术中实现了0.3135的精确-召回曲线下面积（AUPRC），且与其他方法相比没有显著差异（p ≥ 0.36）。

**⚠️ 局限性**

限制在于所有大型手术注册数据均来自单一的学术中心，且稀有的无滞后和严格零抑制的出现率较低，限制了预测的正值预测率。

---

## 725. Learning Networked Dynamical Systems via Neural ODE-based Graph Structure Inference

**arXiv ID:** 2611.25586 | [PDF](https://arxiv.org/pdf/2611.25586v1)

**作者:** Peiyan Sun `[一作]` (Shanghai Jiao Tong University), Yanyan Xu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为NOGI（基于神经ODE的图推断）的框架，用于从多变量时间序列中推断潜在的交互图，旨在理解神经科学、生物学和物理学中的复杂动态系统。

**💡 创新点**

NOGI通过将确定性图学习器与基于消息传递神经网络的神经ODE相结合，显著解耦了自动态与拓扑驱动的交互，提供了一种强大的图结构学习机制。

**🔧 技术方法**

使用了神经常微分方程（Neural ODE）和消息传递神经网络（MPNN）技术，采用了伴随方法进行端到端训练。

**📊 数据集**

在四个模拟动态系统（如Spring、Kuramoto、FHN和MM）以及现实的Netsim fMRI基准数据集上进行了评估。

**📈 对比分析**

与多种基线方法（如NRI、AIDD、NGM和SCOTCH）进行比较，NOGI在32个评估案例中取得了30个最佳结构恢复，且在观察降级情况下表现出色，Netsim基准上平均AUPRC为0.90，AUROC为0.96。

**⚠️ 局限性**

NOGI假设图是静态的，无法处理动态变化的系统；其确定性ODE形式无法捕捉内在的随机性；N × N的边表示导致了O(N^2)的计算成本，限制了其在大规模网络中的可扩展性；此外，NOGI假设所有边遵循相同的交互机制，无法区分促进或抑制目标活动的边。

---

## 726. Products of Distinguisher Hulls

**arXiv ID:** 2611.26169 | [PDF](https://arxiv.org/pdf/2611.26169v1)

**作者:** John Paul Steinberger `[一作]`, Zhe Zhang `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究了在独立产品下，大小受限的布尔电路的区分器外壳的行为，并提供了Dwork和Tankala的超模拟定理的几何后果。

**💡 创新点**

提出了一种不同的近似产品包含性证明，使用非自适应采样替代自适应评分修正。

**🔧 技术方法**

使用了平面段分解和固定锚点的独立采样技术。

**📊 数据集**

使用了有限支持的概率分布，具体数据集未明确给出。

**📈 对比分析**

与Halevi和Rabin等人的方法进行了比较，证明了在相同的电路界限下，性能得到了改进，且只增加了一个因子的预算。

**⚠️ 局限性**

没有假设任何分布的高效采样算法，且所有构造都是非均匀存在论证，未能提供高效的采样器或电路查找算法。

---

## 727. VR-SparseQuant: Accurate FP4 Quantization with 2:4 Sparsity on Vera Rubin

**arXiv ID:** 2611.25717 | [PDF](https://arxiv.org/pdf/2611.25717v1)

**作者:** Samir Khaki `[一作]` (Nebius Applied AI), Ryan Hanrui Wang `[通讯]` (Nebius Applied AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的激活量化方法VR-SparseQuant，利用Vera Rubin的2:4稀疏性实现FP4量化，以提高模型推理的准确性和效率。

**💡 创新点**

创新点在于通过两个重叠的稀疏流来优化激活量化，保持FP4的值预算，同时减少重建误差，这是首次在Vera Rubin平台上实现的自适应激活量化。

**🔧 技术方法**

使用了Vera Rubin的紧凑型元素级2:4稀疏FP4指令，结合离线标定、贪婪分配和共享权重的GEMM操作。

**📊 数据集**

在多个数据集上进行了评估，包括WikiText-2、AIME2026、GPQA和MMLU-Pro等，涉及语言模型和视觉语言模型。

**📈 对比分析**

与传统的密集W4A4方法相比，VR-SparseQuant在多个任务上显著降低了困惑度，并在推理速度上实现了3.1倍的GEMM加速和2.7倍的MoE加速。

**⚠️ 局限性**

限制在于增加了编码工作和元数据流量，可能导致相对于密集W4A4的延迟增加，同时稀疏FP4路径需要G32权重，可能引入额外的权重量化误差。

---

## 728. HumanCOIN: A 12,500-Hour Dataset for Human Intent in Egocentric Manipulation

**arXiv ID:** 2611.26434 | [PDF](https://arxiv.org/pdf/2611.26434v1)

**作者:** Shihan Wu `[一作]` (Beijing Academy of Artificial Intelligence), Guocai Yao `[通讯]` (Beijing Academy of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本研究提出了一个大规模的自我中心操控数据集，记录了与操作意图相关的多个代理信号，包括注视、肌电图（EMG）和脑电图（EEG），并将这些信号与视觉-动作对齐。

**💡 创新点**

创新点在于通过记录意图路径的多个代理信号，提供了一个新的数据集，使得操作意图的建模成为可能，而不仅仅是观察和动作的映射。

**🔧 技术方法**

使用了多种技术，包括多视角RGB摄像头、双侧前臂表面肌电图（EMG）和头皮脑电图（EEG）等传感器，结合深度图和3D点云重建。

**📊 数据集**

数据集包含12500小时的自我中心操控数据，涵盖多个场景类别，分为三个层次，每个层次对应一个代理信号。

**📈 对比分析**

与现有数据集相比，本数据集在覆盖范围上具有优势，能够提供意图路径的三个深度的代理信号，使得生成过程可用于建模而不仅仅是模仿。

**⚠️ 局限性**

限制在于意图是否真正携带了观察所没有的信息尚未得到验证，且数据集的代表性和注释的准确性可能受到设备差异和人类判断的影响。

---

## 729. Towards Autoregressive Neural Network Parameter Generation

**arXiv ID:** 2611.25538 | [PDF](https://arxiv.org/pdf/2611.25538v1)

**作者:** Bedionita Soro `[一作]` (Korea Advanced Institute of Science and Technology), Sung Ju Hwang `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为指令引导参数生成（IGPG）的方法，将神经网络参数生成视为自回归建模问题，旨在通过离散化预训练权重来实现零-shot 泛化。

**💡 创新点**

创新点在于将神经网络参数生成视为条件自回归建模，能够在不同任务和架构之间共享生成器，并有效地利用预训练模型的结构。

**🔧 技术方法**

使用了向量量化变分自编码器（VQ-VAE）和条件自回归变换器（decoder-only transformer）技术。

**📊 数据集**

使用了多个公开数据集，包括Tiny Model Zoo中的MNIST、SVHN、CIFAR-10和STL-10等。

**📈 对比分析**

与其他方法（如从头训练、SKDE30、SANE和D2NWG）进行比较，IGPG在分布内生成中表现出竞争力，并在适应后提供了优于随机初始化的参数初始化。

**⚠️ 局限性**

限制在于当前框架的自回归先验的泛化能力较弱，尤其是在分布转移时，导致生成质量下降。

---

## 730. FamilySpace: From Emergent Chemical Semantics to Controllable Crystal Generation

**arXiv ID:** 2611.25452 | [PDF](https://arxiv.org/pdf/2611.25452v1)

**作者:** Haowei Hua `[一作]` (Hong Kong Polytechnic University), Wanyu Lin `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究展示了一个新的框架FamilySpace，用于控制生成晶体的元素家族组成，基于对晶体变分自编码器（VAE）中原子级潜在表示的几何组织的分析。

**💡 创新点**

创新点在于识别出晶体VAE中出现的化学语义现象，即相同元素家族的表示在潜在空间中形成一致的邻域，而不同家族的表示则占据大致分开的区域。通过FamilySpace框架，可以实现对生成晶体的元素家族组成的直接控制。

**🔧 技术方法**

使用了变分自编码器（VAE）和流模型（flow model）相结合的技术，VAE用于学习原子级表示，流模型用于将简单源分布映射到学习到的潜在分布。

**📊 数据集**

使用了MP-20和MPTS-52数据集进行实验，这些数据集包含不同原子数的晶体结构。

**📈 对比分析**

与DiffCSP、FlowMM、MatterGen、ADiT等方法进行了比较，FamilySpace在生成性能上表现出竞争力，同时能够准确匹配指定的元素家族组成。实验结果显示，FamilySpace在生成的晶体中具有更高的稳定性和亚稳定性预测比例。

**⚠️ 局限性**

限制在于该研究主要集中于潜在空间的组织和生成控制，尚未深入探讨生成晶体的具体物理性能和实验验证。

---

## 731. On the Informativeness of Structural Stability Bounds for GNNs: GINs versus GATs

**arXiv ID:** 2611.25603 | [PDF](https://arxiv.org/pdf/2611.25603v1)

**作者:** Alan Gany `[一作]` (Universite Paris-Saclay), Silviu Maniu `[通讯]` (Univ Grenoble Alpes)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了图神经网络（GNNs）的结构稳定性界限，特别是图同构网络（GINs）和单头图注意力网络（GATv1）在确定性边扰动下的表现，并将其与不依赖扰动的直径界限进行了比较。

**💡 创新点**

创新点在于明确了GINs在何种条件下其稳定性界限优于不依赖扰动的界限，并指出GATv1的界限在任何非零扰动下都不具备信息性。

**🔧 技术方法**

使用了结构稳定性理论，结合了Lipschitz常数和扰动传播的分析方法。

**📊 数据集**

使用了Cora和MUTAG数据集进行实验，评估了不同边删除策略下的稳定性界限。

**📈 对比分析**

通过比较GINs和GATv1的稳定性界限与直径界限，发现GINs在Cora数据集上表现出较强的信息性，而在MUTAG数据集上则在较少层数时失去信息性；GATv1的界限在所有情况下都不具备信息性。

**⚠️ 局限性**

限制在于GINs的稳定性界限仅在扰动相对于最大度数与深度的比率较小时才具备信息性，而在MUTAG上超过几层后就失去信息性；GATv1的界限则始终不具备信息性。

---

## 732. Efficient and Variation-Tolerant Tree Inference via Compact and Scalable Mapping in Analog CAMs

**arXiv ID:** 2611.26567 | [PDF](https://arxiv.org/pdf/2611.26567v1)

**作者:** Zhicheng Xu `[一作]` (University of Hong Kong), Can Li `[通讯]` (University of Hong Kong)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为ACTIVAR的紧凑型变异容忍树推断方法，旨在通过软件与硬件协同设计，在模拟内容可寻址存储器（CAM）中实现高效的树推断。

**💡 创新点**

创新点在于引入了一种模拟CAM感知的紧凑树模型和变异增强训练（VET）算法，解决了传统树模型在模拟CAM中由于结构稀疏性和设备变异性导致的效率和准确性问题。

**🔧 技术方法**

使用了模拟内容可寻址存储器（CAM）和基于铁电场效应晶体管（FeFET）的硬件设计，结合了层次化子树结构和模糊重叠边界策略。

**📊 数据集**

在13个分类和回归数据集上进行了评估，包括MNIST数据集等，展示了ACTIVAR在不同任务中的性能表现。

**📈 对比分析**

与传统的决策树模型和基于模拟CAM的软树设计相比，ACTIVAR在保持接近软件级别的准确性的同时，硬件开销减少了高达99%。在现实硬件噪声条件下，ACTIVAR的准确性下降仅为3.7%，而基线模型则下降了49.6%。此外，ACTIVAR在能效方面相较于商业通用处理器提升了10^4到10^6倍。

**⚠️ 局限性**

限制在于尽管ACTIVAR在硬件效率和准确性上表现优异，但在高维数据集上可能会出现轻微的准确性下降，同时在模糊重叠边界的宽度选择上存在权衡，过宽的边界可能导致模型过拟合或欠拟合。

---

## 733. Identity World Model: Recognizing People in an Imagined Latent World

**arXiv ID:** 2611.25204 | [PDF](https://arxiv.org/pdf/2611.25204v1)

**作者:** Dingqiang Ye `[一作]` (Johns Hopkins University), Vishal M. Patel `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

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

## 734. Actor-Predictor Architecture for Reinforcement Learning with Formal Specification Objectives

**arXiv ID:** 2611.26313 | [PDF](https://arxiv.org/pdf/2611.26313v1)

**作者:** Josef Brozovich `[一作]` (University of Michigan), Lei Ying `[通讯]` (University of Michigan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种Actor-Predictor框架，该框架在不显式构建自动机的情况下学习逻辑进展的替代表示，以满足形式规范的目标。

**💡 创新点**

创新点在于通过预测器持续估计部分轨迹在后续步骤中是否会满足形式规范，从而避免了自动机构建的复杂性，并且在理论上证明了学习的表示等价于自动机状态。

**🔧 技术方法**

使用了Mamba预测器和基于GRPO和RWR优化的演员，结合了深度学习技术。

**📊 数据集**

在Isaac Lab的网格世界和机器人任务上进行了实验，使用了LTL约束任务和非正则形式规范（如a^n b^n）。

**📈 对比分析**

与基于自动机的基线方法（如CyclER和LCER）进行比较，Actor-Predictor在LTL约束任务上表现优越，并且在非正则形式规范上也表现良好，显示出其转移能力和子公式组合性。

**⚠️ 局限性**

限制在于该方法依赖于预测器的准确性，且在处理复杂的形式规范时可能仍然面临挑战。

---

## 735. The Depth of a Prediction: Where Language Models Resolve a Word Predicts Human Reading Time

**arXiv ID:** 2611.25970 | [PDF](https://arxiv.org/pdf/2611.25970v1)

**作者:** Donato Riccio `[一作]` (Universitas Mercatorum), Fabrizio Maturo `[通讯]` (Universitas Mercatorum)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了语言模型的最终层的惊讶度如何预测人们阅读一个单词的时间，并总结了每一层的惊讶度轨迹，提出了一个新的度量标准——分辨深度。

**💡 创新点**

创新点在于提出了分辨深度这一概念，它能够更好地预测阅读时间，超越了最终层的惊讶度和熵的预测能力。

**🔧 技术方法**

使用了GPT-2、OPT和Pythia等多种语言模型，并通过调整的透镜和逻辑透镜来计算每一层的预测。

**📊 数据集**

使用了八个英语阅读语料库，总共分析了81,957个单词，包括自我节奏阅读和眼动追踪数据。

**📈 对比分析**

与其他方法相比，分辨深度在八个语料库中表现出显著的性能提升，平均每个单词提高了0.26毫纳特的预测能力，且在七个语料库中相较于层打乱的深度表现出正向增益。

**⚠️ 局限性**

限制在于所使用的语料库和模型的多样性，分辨深度的增益在不同语料库中表现不一，且对训练数据的依赖性较强。

---

## 736. Teaching Vision-Language-Action Models to Stop Safely

**arXiv ID:** 2611.26239 | [PDF](https://arxiv.org/pdf/2611.26239v1)

**作者:** Zheyuan Liu `[一作]`, Meng Jiang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

该文档提供了一个示例，展示如何使用ACL样式文件与LuaLaTeX或XeLaTeX。

**💡 创新点**

文档的创新点在于展示了多语言文本的处理方式。

**🔧 技术方法**

使用了LuaLaTeX或XeLaTeX技术。

**📊 数据集**

未提供具体数据集。

**📈 对比分析**

未提及比较的方法和性能。

**⚠️ 局限性**

文档缺乏具体的研究内容和结果，限制了其应用性。

---

## 737. QFT-Based Hybrid PointNet-Quantum Neural Networks for LiDAR Point-Cloud Classification

**arXiv ID:** 2611.25578 | [PDF](https://arxiv.org/pdf/2611.25578v1)

**作者:** Anurag Kulkarni `[一作]` (Deep Tech Research and Development Lab, Anukul Consultancy Services), Muhammad Shafique `[通讯]` (New York University Abu Dhabi)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种混合的PointNet-量子神经网络架构，用于LiDAR点云分类，结合了经典特征提取、幅度编码和变分量子电路。

**💡 创新点**

研究了量子傅里叶变换（QFT）在特征处理中的不同配置对分类性能、量子资源需求和噪声鲁棒性的影响。

**🔧 技术方法**

使用了混合量子-经典学习架构，结合了经典特征提取和变分量子电路，评估了五种傅里叶处理配置：No-QFT、Full-QFT-After、QFT-no-swap、IQFT-After和Trainable-QFT。

**📊 数据集**

使用了KITTI、nuScenes和PandaSet三个LiDAR数据集进行实验，评估分类性能和电路复杂性。

**📈 对比分析**

通过与经典模型的比较，IQFT-After在KITTI数据集上取得了最高的加权F1分数（0.857±0.033），而在nuScenes上，Full-QFT-After和IQFT-After的表现相似，PandaSet上Trainable-QFT表现最佳。不同配置在不同数据集上的表现差异表明傅里叶处理的有效性依赖于数据集特性。

**⚠️ 局限性**

限制在于量子电路的复杂性和对硬件噪声的敏感性，尤其是在不同数据集上表现不一致，且理想模拟性能与硬件鲁棒性之间的关系并不总是直接相关。

---

## 738. Accelerating Mixture of Block Attention via Dynamic Routing, KV-Centric Execution and Runtime Pipelining

**arXiv ID:** 2611.26268 | [PDF](https://arxiv.org/pdf/2611.26268v1)

**作者:** Yingqi Shi `[一作]` (Tencent Youtu Lab), Xing Sun `[通讯]` (Tencent Youtu Lab)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出了一种新的动态路由机制，称为LightningMoBA，旨在提高长上下文计算的效率，通过选择关键值（KV）块来减少计算量。

**💡 创新点**

创新点在于动态路由、KV中心执行和运行时流水线的结合，使得选择和执行过程能够更灵活地适应不同查询的需求，从而提高了性能。

**🔧 技术方法**

使用了动态路由、KV中心执行和运行时流水线等技术，优化了GPU的计算效率。

**📊 数据集**

使用了五个检查点的数据集，包括Qwen3-4B、Qwen3-235B-A22B、Qwen3-32B-YaRN4和Llama-3.1-8B-Instruct等。

**📈 对比分析**

与FlashMoBA的固定Top-k方法相比，LightningMoBA在Top-16预算下加速了1.20到1.60倍，同时提高了平均RULER分数，从74.42提高到75.28。在Top-32和Top-64的控制重放中，速度提升分别达到2.16到2.56倍和6.99到11.25倍。

**⚠️ 局限性**

限制在于动态路由在Top-16预算下的表现不如固定Top-k，且在某些配置中未能一致提高RULER准确性，主要加速来自优化的执行后端。

---

## 739. HabitusMem: Belief Memory over Evolving User Preferences for LLM Agents

**arXiv ID:** 2611.25855 | [PDF](https://arxiv.org/pdf/2611.25855v1)

**作者:** Yifan Lu `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Imran Razzak `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出了一种新的个性化记忆系统HabitusMem，该系统将用户的偏好视为助手的信念，并通过每次交互更新这些信念，以便更好地满足用户的当前需求。

**💡 创新点**

创新点在于将个性化记忆重新定义为助手对用户的信念，强调信念的证据、适用范围和状态，从而使代理能够基于用户的当前偏好进行决策。

**🔧 技术方法**

使用了HabitusMem作为记忆系统，该系统通过一个训练的写作模型来生成信念记录，并通过四个任务进行训练，包括直接写作、状态更新、请求投影和行动投影。

**📊 数据集**

使用了PersonaMem-v2和VitaBench 2.0数据集进行评估，前者用于个性化问答，后者用于个性化工具使用。

**📈 对比分析**

与其他五种紧凑型记忆系统进行比较，HabitusMem在两个代理上均表现最佳，尤其在个性化问答和工具使用中，准确性和一致性均优于其他方法。

**⚠️ 局限性**

限制在于固定的输入预算可能会排除来自长交互历史的相关证据，信念提取和应用仍然不完美，特别是在处理撤回的偏好和精确执行约束时。此外，尚未在更广泛的应用中评估HabitusMem。

---

## 740. DMT-Attack: Diffusion-Prior Texture Search for Closed-Loop Vision-Language-Action Policies

**arXiv ID:** 2611.25695 | [PDF](https://arxiv.org/pdf/2611.25695v1)

**作者:** Jun Xia `[一作]` (National University of Defense Technology), Jie Yu `[通讯]` (National University of Defense Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于扩散模型的纹理攻击方法，旨在通过生成自然外观的纹理来干扰视觉-语言-动作（VLA）策略的闭环反馈。

**💡 创新点**

创新点在于利用扩散模型的结构化图像分布搜索自然纹理，同时使用闭环反馈识别干扰任务的候选纹理。

**🔧 技术方法**

使用了冻结的文本条件扩散模型和一个八专家潜在提议网络，结合闭环反馈进行优化。

**📊 数据集**

在LIBERO对象、空间、目标和长任务套件上进行了评估，使用了多个任务的组合。

**📈 对比分析**

与UPA-RFAS和随机自然纹理相比，该方法在四个LIBERO套件中实现了57.9%的宏清洁条件攻击成功率，优于UPA-RFAS的44.7%和随机自然纹理的19.7%。

**⚠️ 局限性**

限制在于该方法依赖于特定的任务和环境设置，可能在不同的场景或任务中表现不一致。

---

## 741. The Price of an Explanation: What Demanding Verifiable Artifacts Costs, and What It Buys Back

**arXiv ID:** 2611.25896 | [PDF](https://arxiv.org/pdf/2611.25896v1)

**作者:** Edgar A. Duéñez-Guzmán `[一作]`, Suzanne Sadedin `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了生成模型在生成候选解决方案时缺乏内在信号进行排名的问题，并提出要求可验证的工件（可执行程序）作为解决方案的补救措施。

**💡 创新点**

创新点在于量化要求可验证工件的成本，包括解决方案质量和计算预算，并通过实验比较不同方法的性能。

**🔧 技术方法**

使用了语言模型（LLM）与进化搜索相结合的方法，进行程序合成和评估。

**📊 数据集**

使用了120个任务的ARC-AGI-2评估集进行实验。

**📈 对比分析**

与仅提供答案的基线方法相比，要求可验证工件的策略导致解决任务的数量从93减少到68，显示出显著的性能下降。进化搜索方法在多个实验中表现出更好的性能，能够在相同预算下解决更多任务。

**⚠️ 局限性**

限制在于进化搜索在没有语言模型的情况下无法找到有效的程序，并且在某些情况下，尽管生成了有效的训练解释，但未能在测试输入上成功泛化。

---

## 742. CACO: Coverage-Aware Coordinated Optimization for Continuous Generative Retrieval

**arXiv ID:** 2611.25738 | [PDF](https://arxiv.org/pdf/2611.25738v1)

**作者:** Pan Zhang `[一作]` (Hunan University), Youhuan Li `[通讯]` (Hunan University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为CACO（Coverage-Aware Coordinated Optimization）的轻量级自监督后训练框架，旨在协调来自冻结的流匹配和扩散检索器的查询，以实现互补的项目覆盖。

**💡 创新点**

CACO的创新点在于使用经验检索频率作为相关性代理，构建无标签的覆盖目标，并通过协调查询调整网络来优化查询，从而提高检索的召回率。

**🔧 技术方法**

使用了流匹配和扩散检索器的生成模型，结合了交叉注意力、双向自注意力和有界切线空间投影等技术。

**📊 数据集**

在八个公共基准数据集和一个内部的24分区流式工业数据集上进行了评估。

**📈 对比分析**

与多种基线方法进行比较，CACO在Recall@100和Recall@400上分别提高了最多23.7%和23.9%，并减少了53.3%-64.9%的查询重叠，同时仅增加了0.5%-3.0%的推理延迟。

**⚠️ 局限性**

CACO的局限性在于其依赖于冻结的基础生成器，可能无法适应快速变化的用户偏好和候选池。

---

## 743. Bilateral Evidence Selection and Reasoning for Mammography Vision--Language Models

**arXiv ID:** 2611.25418 | [PDF](https://arxiv.org/pdf/2611.25418v1)

**作者:** Zhengbo Zhou `[一作]` (University of Pittsburgh), Shandong Wu `[通讯]` (University of Pittsburgh)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种双边证据选择与推理（BESR）框架，旨在提高医疗视觉-语言模型在高分辨率图像中识别小病变的能力，同时保留必要的上下文信息。

**💡 创新点**

创新点在于通过双边引导的令牌选择器和双边奖励机制，将证据保留与比较评估相结合，从而提高诊断性能和证据保留效率。

**🔧 技术方法**

使用了双边引导的令牌选择技术和强化学习方法，结合了结构化预测和后训练优化。

**📊 数据集**

使用了从CBIS-DDSM和VinDr-Mammo构建的双边数据集进行实验，评估了模型的证据保留和诊断性能。

**📈 对比分析**

与现有的医疗推理模型进行比较，BESR在256个令牌预算下的平衡准确率和F1分数分别提高了8.09和10.97个百分点，且在推理效率上也显著提升，预填延迟减少了87.1%。

**⚠️ 局限性**

限制在于当前模型主要针对双边乳腺X光图像，未来需要扩展到多病变检查和不同临床环境的通用性评估。

---

## 744. ReLay: Bidirectional Layout-Code Translation and Library Evolution for Reusable and Process-Portable Analog Design

**arXiv ID:** 2611.25328 | [PDF](https://arxiv.org/pdf/2611.25328v1)

**作者:** Minxing Chu `[一作]` (Brown University), Mehdi Saligane `[通讯]` (Brown University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了ReLay，一个双向布局-代码框架，能够从现有布局中恢复生成器程序，实现跨工艺迁移和现有设计的参数调整。

**💡 创新点**

创新点在于通过从布局中学习生成器库，使其能够随着布局的积累而不断演化，提供了一个可重用和工艺可移植的模拟设计解决方案。

**🔧 技术方法**

使用了布局反编译、程序合成、等式饱和和库学习等技术。

**📊 数据集**

使用了来自27个电路系列的1,050个手绘布局，涵盖了三个开源工艺设计套件（PDK）。

**📈 对比分析**

与现有方法相比，ReLay在完成任务时的API成本降低了64-77%，同时保持了相似的完成率和电气质量，显示出其在性能上的优势。

**⚠️ 局限性**

限制在于ReLay仅能从开源社区发布的布局中学习，而这些布局仍然相对稀缺，且某些布局的电气性能可能不够理想。

---

## 745. The Delayed Societal Cost of Unfairness

**arXiv ID:** 2611.26030 | [PDF](https://arxiv.org/pdf/2611.26030v1)

**作者:** Giovani Valdrighi `[一作]`, Marcos Medeiros Raimundo `[通讯]` (Universidade Estadual de Campinas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

研究了在决策政策的社会成本评估中，如何在动态人群中实现公平性，探讨了在预部署数据上帕累托最优的政策是否会被更公平的政策超越。

**💡 创新点**

创新点在于揭示了政策引发的人口动态可能会改变性能与公平性之间的权衡，使得更公平的政策能够减少决策的延迟社会成本。

**🔧 技术方法**

使用了Rawlsian公平和egalitarian公平的理论框架来分析决策政策的影响。

**📊 数据集**

未具体提及使用的数据集，但研究关注的是动态变化的人口和决策结果。

**📈 对比分析**

通过理论分析比较了不同公平性政策的效果，发现公平性政策在某些情况下能够降低社会成本，尽管可能会影响不同群体的接受率。

**⚠️ 局限性**

限制在于未考虑具体的数据集和实证验证，理论结果可能在实际应用中存在局限性。

---

## 746. Less Than Meets the Eye: Action-Set Identifiability and Principled Abstention for 3D Physical Geometry

**arXiv ID:** 2611.26056 | [PDF](https://arxiv.org/pdf/2611.26056v1)

**作者:** Reshad Ul Karim `[一作]` (BRAC University), Md Rashad Al Hasan Rony `[通讯]` (Deutsche Telekom AG)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文提出了一种名为Intervene3D的方法，该方法通过分析相机运动来区分不同的物理表面和反射机制，并在此基础上预测图像内容的运动。

**💡 创新点**

创新点在于引入了最大最小可分离性选择和适当性测试，以提高对物理机制的识别能力，并在不确定的情况下选择放弃回答。

**🔧 技术方法**

使用了深度网络来估计初始几何形状，并通过相机运动生成不同假设下的图像预测。

**📊 数据集**

使用了合成基准数据集和真实场景数据集，包括打印的图像、显示器和镜子等。

**📈 对比分析**

与其他方法相比，maximin选择在回答场景的准确性上表现更好，能够在相同的承诺准确性下回答更多场景。适当性测试显著降低了错误答案的比例。

**⚠️ 局限性**

限制在于决策依赖于指定的假设家族、动作集和阈值，且在真实场景中可能缺乏足够的线索来区分不同的反射机制。

---

## 747. From Sessions to Serving: Integrating LLM-Based Embeddings into Retrieval for Production Content Recommendation

**arXiv ID:** 2611.26096 | [PDF](https://arxiv.org/pdf/2611.26096v1)

**作者:** Zhixiong Hu `[一作]` (Autodesk), James Bradley `[通讯]` (Autodesk)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了LSER（LLM会话嵌入检索），这是一个用于autodesk.com的内容推荐系统，通过预训练的大型语言模型（LLM）生成会话和历史级别的嵌入，并将其作为辅助用户特征添加到检索阶段的用户塔中。

**💡 创新点**

创新点在于将LLM生成的嵌入作为额外的检索信号集成到推荐系统中，并通过近线服务路径实时刷新用户嵌入，以提高模型性能。

**🔧 技术方法**

使用了预训练的大型语言模型（LLM）来生成用户会话和历史嵌入，并结合了学习的序列编码器和近线服务架构。

**📊 数据集**

使用了来自autodesk.com的用户交互数据集，构建了会话和历史序列特征。

**📈 对比分析**

通过离线消融实验验证了LLM特征的增量价值，并在多个生产表面上进行了在线A/B测试，结果显示集成LLM特征后，用户参与度和转化率显著提高。

**⚠️ 局限性**

局限性包括未能在在线实验中启用LLM特征，缺乏对更大规模平台的通用性验证，以及对LLM骨干网络和训练目标的探索有限。

---

## 748. CompAct-Bench: A Benchmark for Working-Context Compaction in Long-Horizon Agent Tasks

**arXiv ID:** 2611.25673 | [PDF](https://arxiv.org/pdf/2611.25673v1)

**作者:** Jiayi Chen `[一作]` (Fudan University), Maxm Pan `[通讯]` (Hunyuan Team, Tencent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的基准，旨在评估长时间任务中工作上下文的压缩能力，特别是大型语言模型（LLM）在执行信息检索、软件工程和工作空间自动化等任务时的表现。

**💡 创新点**

创新点在于引入了一个200任务的过程级基准，系统地评估了不同LLM在工作上下文压缩中的有效性，并揭示了当前模型在压缩过程中信息损失的严重性。

**🔧 技术方法**

使用了大型语言模型（LLM）作为压缩器，评估了12种前沿模型的性能。

**📊 数据集**

使用了三个数据集：BrowseComp（信息检索任务）、SWE-bench（软件工程任务）和CompanyBench（工作空间任务），共计200个任务。

**📈 对比分析**

通过比较不同模型在压缩后的任务执行准确率，发现即使是最强的模型Claude Opus 5，其整体后压缩准确率也仅为68.5%，平均准确率为56.7%。这表明工作上下文压缩仍然存在显著的信息损失。

**⚠️ 局限性**

限制在于当前模型在压缩过程中无法有效保留所有必要的信息，导致后续任务执行的成功率降低，且不同任务类型和执行阶段的表现差异显著。

---

## 749. GeoPillars: A Large-Kernel Pillar Detector for Single-Sweep LiDAR Under Point Sparsity, and a Decomposition of Its Gain

**arXiv ID:** 2611.25787 | [PDF](https://arxiv.org/pdf/2611.25787v1)

**作者:** Chaitanya Dumbre `[一作]` (MIT World Peace University), Sheetal Girase `[通讯]` (MIT World Peace University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为GeoPillars的基于柱状的3D检测器，旨在解决在点稀疏条件下的单次扫描LiDAR数据的检测问题。

**💡 创新点**

创新点在于引入了几何柱特征、分组向量注意力机制、学习的绝对位置先验、大核骨干网络、辅助表面法线监督和全分辨率检测头。

**🔧 技术方法**

使用了深度学习技术，特别是修改后的分组向量注意力机制和大核卷积网络。

**📊 数据集**

使用了123,364个合成的CARLA扫描数据集进行训练，并在nuScenes数据集上进行了零-shot评估。

**📈 对比分析**

与CenterPoint等其他方法进行了比较，GeoPillars在相同输入预算下的AP@0.7达到了81.33，超出了CenterPoint 13.76点，且在其原生预算下也有1.91点的提升。

**⚠️ 局限性**

限制在于模型在真实数据上的迁移能力差，且所有模型都仅从几何信息中学习，未能有效利用LiDAR强度信息。

---

## 750. There is no binary linear $[65,9,29]$ code

**arXiv ID:** 2611.24997 | [PDF](https://arxiv.org/pdf/2611.24997v1)

**作者:** Zhichao Tang `[一作]` (Sun Yat-sen University), Xiang Fan `[通讯]` (Sun Yat-sen University)

**通讯引用:** 2894 | [OpenAlex ID](https://openalex.org/A5025682608)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明不存在二进制线性[65,9,⩾29]码，等价于证明不存在偶数二进制线性[66,9,⩾30]码。通过已知构造，确定了d_2(65,9)=28，n_2(9,29)=66，n_2(9,30)=67。

**💡 创新点**

通过将假设的偶数码与一个从码字权重导出的二次函数关联，证明了不存在性。这种方法结合了已知的最小长度结果，展示了新的极值参数。

**🔧 技术方法**

使用了布尔函数、权重矩和字符和等技术，结合了极值参数的已知结果。

**📊 数据集**

没有使用特定的数据集，而是依赖于已知的编码理论结果和构造。

**📈 对比分析**

通过与已知的最小长度结果进行比较，证明了不存在性，展示了在七种情况下的上界，并排除了六种情况，最后得出结论。

**⚠️ 局限性**

限制在非退化仿射子空间的条件排除了六种情况，最后的情况中，平等条件强迫四个不同的坐标函数在仿射七维空间上恒等于1，但最多只有两个可以具有此属性。

---

## 751. AI-Driven Transformation of Digital Platform Work: A Task-Based Model for Measuring Labour Disruption and Vulnerability, with Early Evidence from Online Freelancing in Kenya

**arXiv ID:** 2611.25585 | [PDF](https://arxiv.org/pdf/2611.25585v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 752. Online Mistake Detection in Instruction-Conditioned Procedural Video

**arXiv ID:** 2611.25567 | [PDF](https://arxiv.org/pdf/2611.25567v1)

**作者:** Amit Feldman `[一作]` (University of Haifa), Rotem Dror `[通讯]` (University of Haifa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本论文研究了在自我中心视频中进行在线、基于指令的错误检测，提出了一种模型，该模型能够在视频播放过程中识别和定位已完成的动作尝试，并将其分类为正确、修改、失误、纠正或添加。

**💡 创新点**

创新点在于首次实现了在线、基于指令的错误检测，结合了视频观察、任务指令和先前预测，能够实时判断动作是否符合指令，并识别不同类型的偏差。

**🔧 技术方法**

使用了模块化方法和联合模型，模块化方法将步骤识别、时间定位和执行验证分开进行，而联合模型则在一个步骤中预测所有输出。

**📊 数据集**

使用了EgoPER数据集，该数据集包含389个自我中心的视频，涵盖五个烹饪任务，并提供了程序步骤和错误的注释。

**📈 对比分析**

与现有的AEM和ESTANet方法相比，提出的方法在错误检测和时间分段上表现更好，联合模型在错误检测的F1@0.5上达到了84.1%，而模块化方法为80.7%。

**⚠️ 局限性**

限制在于训练和测试仅基于EgoPER的五个烹饪任务，未能验证模型在其他程序或设置中的泛化能力。此外，基准比较受到数据重叠的限制，且在线评估未能捕捉实时交互和错误恢复的有效性。

---

## 753. Ferrofluid Shape Control with Deep Reinforcement Learning

**arXiv ID:** 2611.25232 | [PDF](https://arxiv.org/pdf/2611.25232v1)

**作者:** Anthony Czubarow `[一作]` (Institute for Dynamic Systems and Control), Raffaello D'Andrea `[通讯]` (Institute for Dynamic Systems and Control)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本研究使用深度模型强化学习控制铁磁流体的瞬态形状，以实现软机器人在受限空间中的导航和物体操控。

**💡 创新点**

创新点在于通过直接物理交互学习控制策略，利用铁磁流体的动态特性实现预定瞬态形状的控制。

**🔧 技术方法**

采用深度模型强化学习（DreamerV3）技术，通过视觉观察选择电磁铁的电流命令。

**📊 数据集**

使用了包含铁磁流体的实验平台，结合视觉反馈和电磁驱动进行实验。

**📈 对比分析**

通过与未训练策略的比较，训练策略在形状准确性、达成频率和达成时间上表现出显著的提升，最终策略的平均回报提高了30.7%。

**⚠️ 局限性**

限制在于IoU（交并比）作为形状匹配的度量可能无法充分捕捉几何细节，某些目标形状在最终评估中未能达到阈值。

---

## 754. DS2Agent: Holistic Device Agents for Accelerated Embedded Board-Level Design

**arXiv ID:** 2611.25774 | [PDF](https://arxiv.org/pdf/2611.25774v1)

**作者:** Hongbin Yang `[一作]` (Shandong University), Xiuzhen Cheng `[通讯]` (Shandong University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了DS2Agent，一个将设备数据表转化为可重用设备代理的框架，旨在加速嵌入式板级设计中的问题回答、驱动生成和系统集成。

**💡 创新点**

创新点在于构建了三个相互关联的模型（知识模型、行为模型和驱动模型），并通过模型上下文协议（MCP）暴露其能力，从而实现了设备知识的重用和自动化设计。

**🔧 技术方法**

使用了知识模型、数字双胞胎模型和接口模型，结合了真实设备的反馈来减少错误信息（hallucinations）。

**📊 数据集**

在十个设备上进行了评估，涵盖七个功能类别，包括温湿度传感器、三轴加速度计、ADC等。

**📈 对比分析**

与基线代理相比，DS2Agent增强的代理在问题回答基准中得分从87提高到97，使用的token减少了44.8%；在驱动生成中，平均token使用量减少了72.9%；在系统集成中，三个FreeRTOS应用的总token使用量减少了29.7%。

**⚠️ 局限性**

限制在于DS2Agent的性能依赖于设备数据表的质量和完整性，且在某些情况下可能无法完全替代经验丰富的工程师的判断。

---

## 755. Guided Cuckoo Filter: Full-Pipeline Insertion Optimization

**arXiv ID:** 2611.26053 | [PDF](https://arxiv.org/pdf/2611.26053v1)

**作者:** Min Liu `[一作]` `[通讯]`, Min Liu

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种新的引导式布谷鸟过滤器（Guided Cuckoo Filter），优化了插入流程，通过引导决策替代随机决策，减少插入成本。

**💡 创新点**

创新点在于同时优化了插入过程中的两个阶段：放置阶段和驱逐阶段，分别通过选择空闲槽位较少的桶和优先驱逐最近插入的元素来减少驱逐链的触发频率和长度。

**🔧 技术方法**

使用了引导决策和二选一原则（power-of-two-choice）来优化插入过程，并结合了新颖的驱逐策略，利用新元素的潜在随机性（latent randomness）来选择驱逐对象。

**📊 数据集**

实验使用了具有2^22个槽位的布谷鸟过滤器，具体的负载因子范围为70%到96%。

**📈 对比分析**

与传统的无引导插入方法相比，完全引导的插入方法在96%负载下每次插入的平均驱逐次数减少了最多3.1倍，在95%负载下累计驱逐次数减少了63%。

**⚠️ 局限性**

限制在于该方法在高负载情况下仍然可能面临插入失败的问题，且在实现上增加了每个槽位一个额外的标记位，导致存储开销略有增加。

---

## 756. Sample-efficient end-to-end Optical Music Recognition through Active Learning

**arXiv ID:** 2611.25975 | [PDF](https://arxiv.org/pdf/2611.25975v1)

**作者:** Eric Ayllon `[一作]` (University of Alicante), Antonio Rios-Vila `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究探索了通过主动学习减少光学音乐识别（OMR）中的人工标注负担，提出了一种样本高效的端到端OMR方法。

**💡 创新点**

首次将主动学习技术应用于光学音乐识别（OMR），并评估了不同的采样策略对当前OMR系统的影响。

**🔧 技术方法**

使用了基于置信度的不确定性采样和基于嵌入的多样性采样技术。

**📊 数据集**

使用了多个数据集，包括历史音乐文档的手写和印刷音乐样本。

**📈 对比分析**

与传统方法相比，主动学习显著提高了模型性能，尤其是在标注样本数量有限的情况下。不同的采样策略在不同场景下表现各异，置信度采样在大多数情况下表现良好。

**⚠️ 局限性**

研究的局限性在于未能探索多轮主动学习策略，且所使用的数据集的多样性可能不足以涵盖所有音乐风格和历史背景。

---

## 757. Beyond Resemblance: Image Fusion Reimagined

**arXiv ID:** 2611.25453 | [PDF](https://arxiv.org/pdf/2611.25453v1)

**作者:** Hao Zhang `[一作]` (Wuhan University), Jiayi Ma `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的生成图像融合范式ApexFuse，旨在超越源图像相似性约束，学习高质量场景表示的一般分布。

**💡 创新点**

创新点在于通过人类-LMM（大型多模态生成模型）循环管道，将抽象的融合目标转化为可靠的视觉参考，从而明确理想融合结果的目标分布。

**🔧 技术方法**

使用了大型多模态生成模型（LMM）和预训练的FLUX模型，通过目标分布引导的适应进行图像融合。

**📊 数据集**

使用了来自LLVIP的数据集，包括1060对红外-可见图像对，并生成理想融合目标。

**📈 对比分析**

与八种最先进的融合方法进行比较，ApexFuse在视觉质量和下游语义性能上表现优越，且在所有评估指标上均取得最佳性能，展示了其跨数据集的泛化能力。

**⚠️ 局限性**

限制在于尽管ApexFuse在多种任务上表现良好，但其训练主要集中在红外-可见图像融合，可能在特定应用场景下仍需进一步优化。

---

## 758. Degree Lower Bounds for Symmetric Boolean Functions: Breaking the Prime-Gap Barrier

**arXiv ID:** 2611.26446 | [PDF](https://arxiv.org/pdf/2611.26446v1)

**作者:** Xiaoming Sun `[一作]` (State Key Lab of Processors), Chengu Wang `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

证明了每个非恒定对称布尔函数在n个变量上的确切实数多项式度至少为n-O(n^(1/3)(log n)^(2/3))。

**💡 创新点**

改进了von zur Gathen和Roche的猜想，提出了一个更紧的度缺陷界限O(n^(1/3))，而不是之前的O(√(n))。

**🔧 技术方法**

结合了消失的二项式矩和线性递归的算术，使用了整数Hankel行列式方法的定量形式。

**📊 数据集**

没有具体提到使用的数据集，但涉及的数学理论和结果是基于布尔函数的性质和多项式的度。

**📈 对比分析**

通过与之前的结果进行比较，证明了新的界限优于O(√(n))，并且在假设Riemann假设或Legendre猜想的情况下也得到了改进。

**⚠️ 局限性**

限制在于尚未关闭度缺陷界限Δ(n)与猜想值3之间的差距，计算支持猜想，但在某些范围内仍未得到证明。

---

## 759. Beyond Alignment: Whitening Geometry, Profile Synthesis, and Heterogeneous Ensembles for Cross-Environment WiFi Action Recognition

**arXiv ID:** 2611.25831 | [PDF](https://arxiv.org/pdf/2611.25831v1)

**作者:** Lei Wang `[一作]` `[通讯]` (Dianchi University), Lei Wang (Dianchi University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文探讨了跨环境WiFi动作识别在Leave-One-Domain-Out (LODO)评估下的性能崩溃问题，并提出了一种新的方法Dec-H来解决这一问题。

**💡 创新点**

创新点在于提出了一种新的框架Dec-H，结合了源类内白化、交叉特征混合增强、少量监督适应和异构决策集成等技术，显著提高了在未见环境下的识别准确率。

**🔧 技术方法**

使用了源类内白化（SCW）、交叉特征混合增强（XPMA）、少量监督适应（SAW）和异构决策集成（Dec-H）等技术。

**📊 数据集**

使用了MMFi数据集，该数据集包含四个物理环境和27个细粒度动作。

**📈 对比分析**

与传统的对齐方法相比，Dec-H在所有四个环境中均表现出一致的性能提升，K=10时准确率从0.408提升至0.560，K=40时从0.624提升至0.812，且每个层次的贡献均经过严格的消融实验验证。

**⚠️ 局限性**

限制在于该方法的实验结果主要基于MMFi数据集，且在其他数据集上的表现可能不一致。此外，Dec-H仍未完全消除跨环境的性能差距，未来可能需要更改编码器的表示方式来进一步提升性能。

---

## 760. The Stability Gap in Continual Reinforcement Learning

**arXiv ID:** 2611.26252 | [PDF](https://arxiv.org/pdf/2611.26252v1)

**作者:** Sebastian Pusch `[一作]` (University of Groningen), Gido M. van de Ven `[通讯]` (University of Groningen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了持续强化学习（CRL）中的稳定性差距，提出了一种新的CRL基准，通过改变观察表示来评估任务适应过程中的性能变化。

**💡 创新点**

创新点在于引入了一个简单的CRL基准，能够在保持环境动态和奖励不变的情况下，变化观察表示，从而揭示了在任务转换后，先前任务性能的显著下降及其恢复过程。

**🔧 技术方法**

使用了基于值的算法（如DQN和SAC-D）和演员-评论家算法（如DDPG和SAC），并在增量联合训练的框架下进行评估。

**📊 数据集**

实验中使用了CartPole环境，构建了多个任务，每个任务的状态通过正交变换进行处理。

**📈 对比分析**

通过频繁评估过去任务的性能，发现所有算法在至少一次任务转换后，先前任务的性能均出现了显著下降，随后部分或完全恢复，验证了稳定性差距的存在。

**⚠️ 局限性**

限制在于所提出的基准虽然简单，但未能捕捉到CRL中任务变化的全部范围，可能无法完全反映真实应用中的复杂性。

---

## 761. Scone: Symmetry-aware Neural Combinatorial Optimization for Graph Coloring

**arXiv ID:** 2611.25273 | [PDF](https://arxiv.org/pdf/2611.25273v1)

**作者:** Paolo Pellizzoni `[一作]` (Harvard University), Melanie Weber `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的图着色模型Scone，该模型利用颜色等变图神经网络（GNN）逐步构建图的着色，允许先前分配的信息传播并打破节点对称性。

**💡 创新点**

Scone结合了序列预测、等变性和经典搜索，能够在多个图着色任务中实现完美的解决质量和低推理时间，同时展示了对未见过的图族的泛化能力。

**🔧 技术方法**

使用了颜色等变图神经网络（GNN）作为主要技术，结合了消息传递机制和对称性意识的训练方法。

**📊 数据集**

在实际应用中评估了Scone，包括数独、寄存器分配和考试时间表等图着色实例，并使用合成数据集进行控制实验。

**📈 对比分析**

与传统方法（如贪心算法、Dsatur、ILP等）和其他神经求解器（如AnyCSP、ConsFormer等）进行比较，Scone在解决率和单色边数方面表现优异，同时推理时间最低。

**⚠️ 局限性**

模型在处理某些图族时可能受到1-WL算法的限制，且在某些情况下可能无法处理更复杂的约束和目标。

---

## 762. Mamba Flow Matching Neural Processes: Linear-Time Inference for Irregularly Observed Spatial Fields

**arXiv ID:** 2611.26355 | [PDF](https://arxiv.org/pdf/2611.26355v1)

**作者:** Cosmo Santoni `[一作]` (Imperial College London), Elizaveta Semenova `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种新的联合预测神经过程模型MambaFlowNP，用于从稀疏和不规则的观测数据中插值空间场，如每日最高温度。

**💡 创新点**

MambaFlowNP通过多方向扫描和线性计算复杂度，避免了传统模型中的单一隐藏状态压缩瓶颈，从而提高了预测精度和计算效率。

**🔧 技术方法**

使用了条件流匹配目标和多方向Mamba-2扫描技术，结合了状态空间模型和流匹配的思想。

**📊 数据集**

使用了NOAA的每日最高温度记录数据集，涵盖了不同地理区域（如犹他州、加利福尼亚州和整个美国）的插值任务。

**📈 对比分析**

与最强的基于注意力的联合预测基线相比，MambaFlowNP在合成Matérn高斯过程基准测试和NOAA温度插值任务中表现出最低的均方根误差（RMSE）和连续排名概率评分（CRPS），并且运行速度比最强基线快114-784倍。

**⚠️ 局限性**

MambaFlowNP主要针对不规则点集，可能在规则网格域上效率较低；在小规模数据（K ≲ 10^3）时，其调用延迟可能高于简单的分解NP。此外，当前的评估仅限于一个数据集和一个变量，未来需要扩展到其他数据集和变量。

---

## 763. Fundamentals for Binary Classification Under Label Noise

**arXiv ID:** 2611.25372 | [PDF](https://arxiv.org/pdf/2611.25372v1)

**作者:** Dimitrios I. Diochnos `[一作]` `[通讯]`, Dimitrios I. Diochnos

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

本论文调查了在标签噪声下的二元分类的基本结果，讨论了如何在部分错误的类标签上训练模型，并分析了不同类型的噪声模型对学习算法的影响。

**💡 创新点**

创新点在于区分了不同的噪声模型（如随机分类噪声、Massart噪声和恶意分类噪声），并在PAC框架下形式化了这些模型，提供了对学习算法在噪声影响下的理论基础。

**🔧 技术方法**

使用了概率近似正确（PAC）框架，结合了经验风险最小化（ERM）原则，分析了在不同噪声模型下的学习算法的样本复杂性和计算效率。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了在标签噪声下的学习算法的理论结果，适用于多种实际数据集。

**📈 对比分析**

通过与现有的学习算法进行比较，展示了在不同噪声模型下的学习算法的性能，特别是在随机分类噪声下，学习算法能够保持假设的预期顺序，支持ERM原则的有效性。

**⚠️ 局限性**

限制在于论文主要集中于二元分类问题，未涵盖多类分类、带噪声的回归、主动学习和在线学习等更广泛的场景。

---

## 764. Plan your Self-Adaptation! Ontology-Mediated Task and Architecture Co-adaptation Planning for Robots

**arXiv ID:** 2611.25254 | [PDF](https://arxiv.org/pdf/2611.25254v1)

**作者:** Gustavo Rezende Silva `[一作]` (Delft University of Technology), EinarR Broch Johnsen `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于本体的任务和架构共同适应的方法，考虑了任务计划和架构配置之间的相互依赖关系。

**💡 创新点**

创新点在于通过自动集成任务和架构知识，将任务和架构共同适应视为一个单一的规划问题，并支持满意规划和运行时重新规划。

**🔧 技术方法**

使用了Web本体语言（OWL）表示架构替代方案及其功能和质量要求，使用规划领域定义语言（PDDL）定义任务动作和任务目标。

**📊 数据集**

在两个机器人场景中进行评估：一个是自主水下车辆（AUV）进行管道检查，另一个是自主地面车辆（AGV）进行导航。

**📈 对比分析**

与Cámara等人的方法相比，提出的方法在AGV场景中平均速度快3倍，峰值内存使用量小6.84倍，且两者生成的计划质量相似。

**⚠️ 局限性**

限制在于只针对每个功能选择具有最高效用的可实现架构配置，未考虑架构选择的长期影响，如能耗等。此外，当前模型未考虑架构重新配置的成本。

---

## 765. Does It Depend on the Split? Frozen Satellite Features vs. Fine-Tuned CNNs for Tree Species Mapping

**arXiv ID:** 2611.25902 | [PDF](https://arxiv.org/pdf/2611.25902v1)

**作者:** Dharani Deivasihamani `[一作]` (SatSure Analytics India Pvt. Ltd.), Nandakishore Puttashamachar `[通讯]` (SatSure Analytics India Pvt. Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究测试了自监督基础模型在森林任务中的表现，具体是对TreeSatAI数据集中20种树种进行分类，比较了冻结的DINOv3模型与经过微调的ImageNet ResNet-50模型的性能。

**💡 创新点**

创新点在于提出了一种空间划分方法，避免了测试数据与训练数据的重叠，从而更真实地评估模型在新区域的泛化能力。

**🔧 技术方法**

使用了DINOv3（自监督视觉变换器）和ImageNet预训练的ResNet-50模型，前者未进行微调，后者进行了全权重微调。

**📊 数据集**

使用了TreeSatAI数据集，该数据集包含50,381个0.2米的航空图像补丁，标注了20种树种。

**📈 对比分析**

在官方划分中，微调的ResNet-50模型表现优于冻结的DINOv3模型（宏F1为0.764对0.685）。但在新的空间划分中，两者的表现接近（0.648对0.640），且在每个标签预算下的学习曲线重叠。

**⚠️ 局限性**

限制在于稀有类的测试位置较少，导致结果的不确定性。此外，比较中混合了架构、预训练数据和冻结策略的影响，未能单独评估自监督预训练的优势。

---

## 766. CauSyncVLA: Causal Latent World Modeling for Asynchronous Robot Control

**arXiv ID:** 2611.25667 | [PDF](https://arxiv.org/pdf/2611.25667v1)

**作者:** Xu Pan `[一作]` (Nanyang Technological University), Ziwei Wang `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种异步双系统控制器，通过将感知记忆预测与实际机器人执行对齐，解决了视觉推理与动作执行之间的延迟问题。

**💡 创新点**

创新点在于结合了因果潜在世界模型和预测流热启动（PFWS），实现了在不额外训练的情况下对未执行动作的单步流修正。

**🔧 技术方法**

使用了因果潜在世界模型和预测流热启动技术，结合了感知记忆的预测和轨迹修正。

**📊 数据集**

在LIBERO-10数据集上进行评估，成功率从91.2%提高到93.8%。

**📈 对比分析**

与现有方法相比，提出的方法在闭环鲁棒性和机器人运动平滑性上表现更好，且在共享GPU上实现了更高的模型重规划能力。

**⚠️ 局限性**

限制在于该方法依赖于冻结的视觉-语言-动作（VLA）策略，可能在面对快速变化的环境时表现不佳。

---

## 767. Mitigating Row Buffer Contention Through Multiple Row Buffer (MRB) DRAM

**arXiv ID:** 2611.25714 | [PDF](https://arxiv.org/pdf/2611.25714v1)

**作者:** K Chitra `[一作]`, Minesh Patel `[通讯]` (Rutgers University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的DRAM架构MRB-DRAM，通过在每个银行中引入多个行缓冲区来减少行缓冲区冲突，从而提高性能和降低能耗。

**💡 创新点**

创新点在于引入多个次级SRAM行缓冲区作为缓存，使每个银行能够并行处理多个行的请求，从而有效减少行缓冲区冲突。

**🔧 技术方法**

使用了MRB-DRAM架构，结合了主行缓冲区和多个次级SRAM行缓冲区，并通过Ramulator进行性能评估。

**📊 数据集**

使用了SPEC2006、SPEC2017等基准测试数据集，模拟了单核和多核配置下的性能表现。

**📈 对比分析**

与传统DDR4相比，MRB-DRAM在单核系统中平均提高了29.7%的性能，在多核系统中提高了23.7%的加权加速比，同时减少了58.9%的激活命令和121%的行缓冲区命中率。

**⚠️ 局限性**

限制在于虽然MRB-DRAM在性能和能效上表现良好，但仍需考虑其硬件开销和设计复杂性，尤其是在多核环境下的可扩展性问题。

---

## 768. MATE: Mobile Agent Trustworthy Execution for Safe and Efficient Mobile Control

**arXiv ID:** 2611.25526 | [PDF](https://arxiv.org/pdf/2611.25526v1)

**作者:** Minsuk Jang `[一作]` (Korea Advanced Institute of Science and Technology), Changick Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MATE（移动代理可信执行）系统，该系统在不修改底层模型的情况下控制代理动作提议与设备效果之间的边界，确保移动代理在个人智能手机上的执行安全。

**💡 创新点**

MATE通过将每个用户界面原语表示为结构化的动作记录，并在执行前进行安全检查，显著提高了高风险任务的阻止率，解决了现有移动代理系统的安全缺口。

**🔧 技术方法**

使用了运行时护栏（Runtime Guardrail, RG）和预定义任务路由（Predefined Task Routing, PTR）技术，前者用于检查提议的动作，后者用于重用经过验证的应用特定路由。

**📊 数据集**

在MobileSafetyBench和AndroidWorld数据集上进行评估，MobileSafetyBench用于高风险任务的阻止率，AndroidWorld用于任务成功率和LLM调用次数。

**📈 对比分析**

与现有移动代理系统相比，MATE在高风险任务的阻止率上显著提高，RG将高风险任务的拒绝/阻止率从29.67%提高到97.03%，同时在AndroidWorld上将LLM调用次数减少了最多6倍，任务成功率提高到95.7%。

**⚠️ 局限性**

MATE的局限性在于其主要关注已验证的个人设备工作流，未来的工作将扩展到未见应用的路由诱导和UI漂移的校准，同时可能需要针对特定应用布局的校准。

---

## 769. The Window Lever: Longer Observation Windows Outweigh the Observed Effects of Data, Capacity, and Architecture in Customer Behavior Forecasting

**arXiv ID:** 2611.26521 | [PDF](https://arxiv.org/pdf/2611.26521v1)

**作者:** Tara Sowrirajan `[一作]` (Amazon), Yanqiao Wang `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文比较了序列模型在客户活动预测任务中的表现，特别关注历史长度对模型准确性的影响。

**💡 创新点**

创新点在于发现历史长度是影响模型性能的最大因素，并提出了一种匹配队列的框架来消除不同窗口长度带来的偏差。

**🔧 技术方法**

使用了多种序列模型，包括变换器、GRU、LSTM等，并引入了多尺度旋转位置编码（MS-RoPE）作为时间表示的改进。

**📊 数据集**

使用了多个数据集进行实验，包括一个包含超过480,000行的主要数据集，以及一个为期九天的公共数据集。

**📈 对比分析**

通过逐一比较不同因素的影响，发现历史长度的增大使得R^2提升了0.070到0.107，而其他因素如模型架构和容量的影响则小得多，通常在0.006以下。

**⚠️ 局限性**

限制在于无法完全分离历史长度与日历覆盖的影响，且在某些情况下，模型的表现可能受到训练数据的组成变化的影响。

---

## 770. Rigel-Ladon: An Option-Count Law for a One-Pass Decision Encoder, Tested on Held-Out Option Counts and Held-Out Schemas

**arXiv ID:** 2611.26332 | [PDF](https://arxiv.org/pdf/2611.26332v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 771. VisualJevBench: A Benchmark and Black-Box Attacks for Visual Typed Decision Models

**arXiv ID:** 2611.25435 | [PDF](https://arxiv.org/pdf/2611.25435v1)

**作者:** Li Zeng `[一作]`, Hangtao Zhang `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了VisualJevBench，一个用于评估视觉决策模型Visual Jev对图像操控的对抗鲁棒性的基准，包含800张图像和3200个问题，涵盖四种视觉攻击表面。

**💡 创新点**

创新点在于系统性地评估直接图像操控对Visual Jev的影响，并引入了一种轻量级框架Adaptive Black-Box Parameter Search来寻找更有效的图像操控。

**🔧 技术方法**

使用了Typed Decision Models (TDMs)和Adaptive Black-Box Parameter Search技术。

**📊 数据集**

使用的数据集包括800张图像和3200个问题，涵盖了日常和专业领域的八种场景。

**📈 对比分析**

与静态攻击相比，图像特定攻击的成功率（ASR）最高可达69.6%，而通用攻击在未见图像上的平均ASR为33.15%。几何变换在跨场景转移中实现了40.09%的平均ASR。

**⚠️ 局限性**

局限性在于仅限于预定义的操控类型和非自适应防御测试，未来工作将探索更广泛的攻击策略和对自适应攻击的防御。

---

## 772. Axioms of Phyllotaxis

**arXiv ID:** 2611.24978 | [PDF](https://arxiv.org/pdf/2611.24978v1)

**作者:** Burghard Herrmann `[一作]` `[通讯]` (Freie Universitaet Berlin), Burghard Herrmann (Freie Universitaet Berlin)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出了一种广义公理模型，用于描述花头早期发育阶段的新物体的同时出现，并应用于多个相关的发散角度。

**💡 创新点**

创新点在于通过非确定性林登梅尔系统的简单扩展形式，重新调整了植物生长的初始结构，并解释了137.5°发散角的起源。

**🔧 技术方法**

使用了林登梅尔系统（L-systems）和非确定性模型来描述植物的生长过程。

**📊 数据集**

论文中使用了斐波那契数列和黄金比例作为数学基础，但未具体提及使用的数据集。

**📈 对比分析**

通过与经典的植物生长模型进行比较，展示了新模型在解释发散角度方面的有效性，性能上提供了更广泛的适用性。

**⚠️ 局限性**

限制在于模型的适用范围可能仅限于特定的发散角度，且未考虑其他可能影响植物生长的因素。

---

## 773. Deterministic Parameterized Inapproximability of the Shortest Vector and Minimum Distance Problems

**arXiv ID:** 2611.26273 | [PDF](https://arxiv.org/pdf/2611.26273v1)

**作者:** Shuichi Hirahara `[一作]` (National Institute of Informatics), Kazuki Ogitsuka `[通讯]` (Graduate University for Advanced Studies)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

论文证明了在ℓ_p范数下的参数化最短向量问题(SVP)和最小距离问题(MDP)在确定性还原下是[1]-难以近似的。

**💡 创新点**

创新点在于首次在确定性还原下证明了SVP和MDP的参数化难度，解决了之前仅在随机还原下的难度问题。

**🔧 技术方法**

使用了确定性还原技术，结合了局部稠密格和编码的构造方法。

**📊 数据集**

使用了Reed-Solomon码和BCH码等数据集来构造局部稠密格和局部稠密码。

**📈 对比分析**

与之前的随机还原方法相比，论文展示了在确定性情况下的[1]-难度，性能上提供了更强的理论支持。

**⚠️ 局限性**

限制在于当前的结果主要集中在特定的范数和有限域上，尚未扩展到更广泛的情况。

---

## 774. GridAnchor: Specialized Multi-Agent AI for Safeguarding Autonomous Energy Systems Against Cyber-Physical Hallucinations

**arXiv ID:** 2611.25158 | [PDF](https://arxiv.org/pdf/2611.25158v1)

**作者:** Abderaouf Bahi `[一作]` (Chadli Bendjedid University), Hasan Masrur `[通讯]` (King Fahd University of Petroleum and Minerals)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为GridAnchor的验证层，用于保护自主能源系统免受网络物理幻觉的影响。该系统通过定义网络物理幻觉及其八种类型，构建了CPH-Bench基准，并利用物理残差图变换器（PRGT）对每个声称进行评分。

**💡 创新点**

创新点在于提出了一种物理基础的验证方法，能够有效检测和修复由自主学习代理产生的网络物理幻觉，并且在多个基准测试中表现优于现有方法。

**🔧 技术方法**

使用了物理残差图变换器（PRGT）和蒙德里安符合性p值方法，结合贝尔曼-霍赫伯格程序来控制假阳性率，并通过二次规划修复不安全的操作。

**📊 数据集**

使用了SimBench的中压馈线数据集，进行了一系列的模拟实验，以验证GridAnchor的有效性。

**📈 对比分析**

与六个基线方法进行比较，GridAnchor在消息级别的精确率-召回曲线（AUPRC）达到了最佳表现，且在闭环测试中没有任何总线超出电压范围，表现优于其他方法。

**⚠️ 局限性**

限制在于所有结果均来自于模拟，未涉及实际测量或硬件，且所研究的幻觉是合成的，可能与实际部署中的大型语言模型表现不同。

---

## 775. Whose Risks Count? What 1,901 AI Incidents Reveal About Corporate Disclosures

**arXiv ID:** 2611.25841 | [PDF](https://arxiv.org/pdf/2611.25841v1)

**作者:** Simeon Allmendinger `[一作]` (University of Bayreuth), Daniele Quercia `[通讯]` (Nokia Bell Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了人工智能系统对人类造成的伤害，以及相关的AI行为者在证券文件中披露这些风险的情况。通过将9001个AI事件与18229份SEC文件进行匹配，分析了AI行为者在事件发生前后是否披露了相关风险类型。

**💡 创新点**

创新点在于首次发布了DISCorpus（Disclosure–Incident Securities Corpus），该数据库将AI事件与证券披露相匹配，并分类为四种类型：Covered、Eventually Disclosed、Dropped和Missed，揭示了证券披露在记录AI风险方面的局限性。

**🔧 技术方法**

使用了自然语言处理技术，通过大型语言模型（LLMs）对AI事件和SEC文件进行编码和分类。

**📊 数据集**

使用的数据集包括来自三个公共数据库的9001个AI事件，以及328个AI行为者的18229份SEC文件。

**📈 对比分析**

与其他方法相比，本文通过匹配AI事件与SEC披露，发现67%的系统故障风险被披露，但只有2%的社会系统伤害被披露，表明证券披露主要记录AI行为者面临的风险，而非AI主体承受的伤害。

**⚠️ 局限性**

限制在于研究仅依赖于公共记录，可能遗漏未报告的事件，并且只关注SEC注册的AI行为者，未考虑私营AI行为者的情况。此外，匹配事件与披露的风险类型可能未能充分描述事件的具体情况。

---

## 776. FPGA-GEN: Target-Aware RTL Generation via HLS

**arXiv ID:** 2611.25351 | [PDF](https://arxiv.org/pdf/2611.25351v1)

**作者:** Luke Ztz Hu `[一作]` (Shenzhen International Graduate School), Songping Mai `[通讯]` (Shenzhen International Graduate School)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为FPGA-GEN的工作流程，用于生成和修复高层次综合（HLS）兼容的C/C++代码，旨在提高FPGA硬件的正确性和效率。

**💡 创新点**

创新点在于通过明确的任务合同和工具反馈循环，结合功能验证和硬件优化，提供了一种有针对性的候选选择和修复方法。

**🔧 技术方法**

使用了高层次综合（HLS）工具、代码生成、C/RTL协同仿真等技术，并结合了大型语言模型（LLMs）进行代码生成和修复。

**📊 数据集**

使用了多个数据集，包括107个HLS-LeVeri设计和其他来源的214个任务，进行功能验证和设备适配。

**📈 对比分析**

与其他方法相比，FPGA-GEN在公共测试基准上取得了95.8%的通过率，并在HLS-LeVeri设计中实现了97.2%的适配率，显示出其在性能上的优势。

**⚠️ 局限性**

限制在于模型能力和硬件感知训练数据的改进可能使部分工作流程过时，且在验证和修复过程中仍面临实际挑战。

---

## 777. AGE-Capture: Autonomous, General, and Efficient UAV Capture for High-Fidelity Scene Reconstruction

**arXiv ID:** 2611.26048 | [PDF](https://arxiv.org/pdf/2611.26048v1)

**作者:** Xiaofeng Jin `[一作]` (Politecnico di Milano), Matteo Matteucci `[通讯]` (Politecnico di Milano)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了AGE-Capture，一个自主、传感器自适应且高效的主动捕获框架，用于无人机的高保真场景重建。

**💡 创新点**

创新点在于将几何探索与重建观察解耦，建立统一的相机光线接口，使不同的传感器和RGB相机模型能够共享重建驱动的获取管道。

**🔧 技术方法**

使用了轻量级的在线可渲染状态来评估候选视图的效用，结合动态感知的分层规划和异步事件驱动捕获。

**📊 数据集**

在十个室内和室外的模拟场景中进行实验，验证了AGE-Capture的有效性。

**📈 对比分析**

与在线基线方法（如ActiveGS和TRACE）相比，AGE-Capture在重建质量上平均提高了3.30-3.53 dB的峰值信噪比（PSNR），同时平均使用了58.2%的任务时间预算，GPU内存使用量降低了91.0%-93.3%。

**⚠️ 局限性**

限制在于在实际飞行中，捕获区域和完成标准的放宽可能导致某些远处表面观察不足，影响重建质量。

---

## 778. LingSante MedGuard Technical Report

**arXiv ID:** 2611.25509 | [PDF](https://arxiv.org/pdf/2611.25509v1)

**作者:** LingSante MedGuard Team `[一作]` `[通讯]` (LingSante), LingSante MedGuard Team (LingSante)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8d10c613-917e-4880-9716-17789f50e119` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为LingSante MedGuard的统一医疗护栏，旨在通过整合多种医疗风险评估能力来提高医疗AI响应的安全性和合规性。

**💡 创新点**

创新点在于开发了一个能力增强的统一医疗护栏，结合了医疗安全、合规性和伦理的评估标准，并通过两阶段的训练方法整合了来自不同专家的监督。

**🔧 技术方法**

使用了监督微调（SFT）和多教师在线蒸馏（MOPD）等技术，结合了医疗护栏教师和六个医疗能力专家的监督。

**📊 数据集**

使用了LingSante MedGuardBench数据集，该数据集包含11个评估子集，涵盖医疗安全、合规性和伦理的风险检测。

**📈 对比分析**

与其他方法相比，LingSante MedGuard在医疗风险检测任务上表现优异，4B、9B和27B模型的平均基准F1分数分别为82.3%、82.4%和87.4%。

**⚠️ 局限性**

限制在于尽管模型在多个任务上表现良好，但在某些特定的医疗场景中，仍可能存在未能完全覆盖所有风险的情况。

---

## 779. LeCompass: Task-Level Execution through State-Indexed Interfaces

**arXiv ID:** 2611.25679 | [PDF](https://arxiv.org/pdf/2611.25679v1)

**作者:** Kai Shi `[一作]` (City University of Hong Kong), Peng Yin `[通讯]` (City University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为LeCompass的状态索引执行框架，旨在通过建模动作后果来支持预测决策制定。

**💡 创新点**

创新点在于将冻结的LeWorldModel与状态索引的目标结合，支持跨剧集的上下文-目标组合，并通过有限预算下的预测搜索来优化执行。

**🔧 技术方法**

使用了状态索引的联合嵌入预测架构（JEPA）和CompassFlow进行低预算的预测搜索。

**📊 数据集**

在1,000个官方测试的完整Cube试验中进行评估。

**📈 对比分析**

与传统的动作空间CEM方法相比，LeCompass在PushT和TwoRoom任务中表现出显著的性能提升，成功率分别从4.7%提升至44.9%和从33.8%提升至77.7%。

**⚠️ 局限性**

局限性在于评估仅覆盖三个模拟基准，且完整的观察基础执行接口仅在Cube上实例化，转移到其他任务可能需要新的几何监督。

---

## 780. TRACE: Attribute Execution for Sequential Image Editing

**arXiv ID:** 2611.25648 | [PDF](https://arxiv.org/pdf/2611.25648v1)

**作者:** Xiyang Zhang `[一作]` (Harbin Institute of Technology), Yan Song `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为TRACE的属性执行器，用于顺序图像编辑，能够根据用户的指令逐步更新图像，同时保留之前的要求。

**💡 创新点**

TRACE通过将持续属性与其视觉载体分离，并在完整请求检查下优先考虑局部外观操作，解决了顺序图像编辑中的属性冲突问题。

**🔧 技术方法**

使用了局部属性执行、主动目标绑定和色调校准等技术。

**📊 数据集**

使用了EdiVal和MagicBrush等数据集进行实验，评估了不同编辑器和独立视觉读取器的性能。

**📈 对比分析**

与其他方法相比，TRACE在指令遵循和内容一致性方面表现最佳，尤其是在处理颜色和材料的联合恢复时，性能显著提高。

**⚠️ 局限性**

TRACE的局限性在于它可能在处理复合请求和额外外观属性时面临挑战，未来的研究方向是扩展其目标-载体分离的应用。

---

## 781. An AI-Powered Multiverse Analysis of Empirical Finance Research: Evidence from Reused Natural Experiments

**arXiv ID:** 2611.25184 | [PDF](https://arxiv.org/pdf/2611.25184v1)

**作者:** Ruijiang Gao `[一作]`, Steven Chong Xiao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文使用人工智能代理对重用自然实验的论文进行多元宇宙分析，并结合结果进行元分析。

**💡 创新点**

创新点在于利用AI代理进行大规模的重估和比较，评估文献中结果的稳健性，尤其是针对相同实验的论文进行比较。

**🔧 技术方法**

使用了人工智能编码代理，进行数据重建和多元宇宙分析，结合脚本和审计确保每一步的准确性。

**📊 数据集**

数据集来自重用自然实验（RNE）数据库，包含多个金融和会计实验的论文。

**📈 对比分析**

通过比较不同论文的结果，发现第一阶效应的声明更稳健，且在顶级期刊发表的论文的结果更可靠。重现的声明在所有四个结果上表现出更高的稳健性。

**⚠️ 局限性**

限制在于分析的重复性和模型的训练数据可能影响结果的独立性，且低的支持率并不一定意味着声明是错误的。

---

## 782. Point Dynamics as Proxy for Robotic World Models

**arXiv ID:** 2611.25946 | [PDF](https://arxiv.org/pdf/2611.25946v1)

**作者:** Dong An `[一作]` (Institute of Foundation Models), Guangyi Liu `[通讯]` (Institute of Foundation Models)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

该论文提出了一种新的机器人世界模型，通过将动态预测与视觉渲染分离，来提高机器人操作中的视频生成质量。

**💡 创新点**

创新点在于使用3D点作为代理，独立预测场景点的动态，并将其渲染为视频，从而提高了与现实的交互保真度。

**🔧 技术方法**

使用了3D点动态模型（PointWorld）作为模拟器，结合了前向运动学和像素对齐的渲染技术。

**📊 数据集**

使用了DROID数据集，该数据集包含326K个片段和41.5K个实验，测试在900个实验的1515个片段上进行。

**📈 对比分析**

通过与不同条件下的渲染器（Wan2.2和Ctrl-World）进行比较，结果显示使用点动态代理的模型在对象运动误差上相较于机器人渲染减少了8.6%，且在真实机器人上也有类似的改进。

**⚠️ 局限性**

限制在于视频的保真度依赖于模拟器的准确性，模拟器和渲染器是分开训练的，可能导致渲染器无法纠正模拟器的错误，未来的研究可以考虑联合训练。

---

## 783. FG-GraphBench: Benchmarking and Enhancing Fine-Grained Visual Understanding with Multimodal Scene Graphs

**arXiv ID:** 2611.25408 | [PDF](https://arxiv.org/pdf/2611.25408v1)

**作者:** Sicheng Liu `[一作]` (Beijing Institute of Technology), Rong-Hua Li `[通讯]` (Beijing Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Fine-Grained Multimodal Scene Graph Benchmark (FG-GraphBench)和FiGRA，旨在解决现有场景图基准在细粒度实体和关系覆盖不足的问题，并有效利用图结构信息进行细粒度视觉理解。

**💡 创新点**

创新点在于首次探索细粒度多模态场景图作为增强多模态大语言模型（MLLMs）细粒度视觉理解的统一结构表示，并提出了两阶段的数据构建管道和FiGRA方法。

**🔧 技术方法**

使用了多模态图编码和图表示增强技术，通过连续图标记将图结构信息集成到MLLM中，并采用两阶段训练策略。

**📊 数据集**

使用了FG-GraphBench数据集，该数据集包含三个互补任务：细粒度视觉问答（FG-QA）、细粒度图像描述生成（FG-Description）和细粒度图像标题生成（FG-Caption）。

**📈 对比分析**

与强基线方法进行比较，FiGRA在FG-GraphBench的三个任务上显著超越了现有方法，表现出色，尤其在细粒度视觉问答任务中，准确率达到59.59%。

**⚠️ 局限性**

限制在于现有的场景图基准未能充分覆盖细粒度实体及其关系，且在有效整合细粒度图信息到MLLMs方面仍面临挑战。

---

## 784. RegionAgent: Two-Stage Preference Optimization for Regional Image Restoration

**arXiv ID:** 2611.26179 | [PDF](https://arxiv.org/pdf/2611.26179v1)

**作者:** Weiyu Zhou `[一作]` (Northwestern Polytechnical University), Qingsen Yan `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为RegionAgent的自适应图像修复框架，旨在处理具有非均匀耦合退化的图像，通过区域特定的修复动作来改善修复效果。

**💡 创新点**

创新点在于引入了两阶段的偏好学习（DPO），使得视觉语言模型（VLM）能够根据当前图像状态选择适合的区域和修复配置，从而实现更精细的修复。

**🔧 技术方法**

使用了视觉语言模型（VLM）和两阶段的直接偏好优化（DPO）技术。

**📊 数据集**

构建了一个包含10K动作对和20K轨迹对的区域修复偏好数据集，以支持两阶段的VLM训练。

**📈 对比分析**

通过与多种数据集的实验比较，RegionAgent在混合和单一退化情况下表现出强大的修复性能，尤其在PSNR、SSIM等指标上超越了现有方法。

**⚠️ 局限性**

局限性在于当前方法可能在处理极端复杂的退化情况下仍然存在不足，且对计算资源的需求较高。

---

## 785. Calibrate in Context: Efficient Data-Free Quantization from Small Synthetic Pools

**arXiv ID:** 2611.25711 | [PDF](https://arxiv.org/pdf/2611.25711v1)

**作者:** Haowen Hua `[一作]` (Soochow University), Cong Yang `[通讯]` (D-Robotics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文提出了一种通过改进图像合成和学生准备来实现从小型合成池中准确恢复低比特模型的方法。

**💡 创新点**

创新点在于引入了功能网格校准（FGC），通过在学生的块输入上下文中选择激活网格，从而提高了恢复的准确性和效率。

**🔧 技术方法**

使用了功能网格校准（FGC）技术和渐进反演更新方法。

**📊 数据集**

使用了256张和1000张合成图像的数据集进行实验。

**📈 对比分析**

与MimiQ等现有方法进行比较，论文的方法在所有18个W4A4/W5A5设置中均超过了MimiQ的报告准确性，并且在准备成本上更具优势。

**⚠️ 局限性**

限制在于方法依赖于合成图像的质量和数量，可能在某些情况下无法达到最佳效果。

---

## 786. Subexponential Algorithms for Continuous Euclidean $k$-Means in Subpolylogarithmic Dimension

**arXiv ID:** 2611.25305 | [PDF](https://arxiv.org/pdf/2611.25305v1)

**作者:** Junqi Tan `[一作]` `[通讯]` (Freie Universitaet Berlin), Junqi Tan (Freie Universitaet Berlin)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种确定性的精确算法，用于解决连续欧几里得k均值问题，该算法在维度d=(log n)^o(1)时运行时间为O^*(2^o(n))，并使用多项式空间。

**💡 创新点**

该算法在处理任意k时表现出色，尤其是在k与n线性增长的情况下，首次实现了在特定维度下的亚指数时间复杂度。

**🔧 技术方法**

使用了动态规划、网格划分和超平面比较等技术，结合了小簇和大簇的不同处理方式。

**📊 数据集**

算法适用于任意有理坐标的输入，输入以二进制形式给出，且分子和分母可以具有任意位长。

**📈 对比分析**

与现有方法相比，该算法在高维情况下的时间复杂度得到了显著改善，尤其是在维度为(log n)^o(1)时，运行时间为O^*(2^o(n))，而其他方法通常在高维情况下表现较差。

**⚠️ 局限性**

算法的局限性在于尚未证明在任意实数坐标的情况下也能达到相同的运行时间，且在更高维度下，组合分配的效率仍需进一步优化。

---

## 787. WeMM-Video: Pushing the Frontiers of Open-Source Omni-Modal Video Understanding

**arXiv ID:** 2611.26409 | [PDF](https://arxiv.org/pdf/2611.26409v1)

**作者:** Changli Tang `[一作]` (WeChat Vision, Tencent Inc.), Jing Lyu `[通讯]` (WeChat Vision, Tencent Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并开源了WeMM-Video，一个中文-英文双语的全模态基础模型系列，专注于视频理解和推理。

**💡 创新点**

WeMM-Video通过混合思维范式统一文本、图像、音频和视频处理，采用非思维模式和思维模式来提高复杂视频任务的性能。

**🔧 技术方法**

使用了渐进式预训练和多阶段后训练管道，包括多教师在线策略蒸馏、反重复多轮直接偏好优化和专家强化学习。

**📊 数据集**

在一个包含约1.2万亿个标记的多模态语料库上进行训练，数据集包括音频-视觉联合数据、视频字幕等。

**📈 对比分析**

与领先的专有模型（如Gemini-3.0-Flash和Seed2.0-Mini-0428）进行比较，WeMM-Video-35B-A3B在开放源代码模型中表现出色，且在多个基准测试中超越了其他模型。

**⚠️ 局限性**

模型在处理复杂的音频-视觉推理任务时仍面临挑战，尤其是在高质量知识和复杂推理数据的获取上存在局限性。

---

