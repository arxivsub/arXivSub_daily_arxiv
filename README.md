# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-10-22 | 今日论文总数: 705

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Navigation Explainability for Object Avoidance with LLMs for Assistive Robots

**arXiv ID:** 2610.28920 | [PDF](https://arxiv.org/pdf/2610.28920v1)

**作者:** Courtney Young `[一作]`, Sehoon Ha `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e0540dec-d77f-42db-94ae-d039248f6393` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一个上下文感知导航解释框架，利用LLM实时生成导盲犬机器人运动的自然语言解释，并通过用户研究评估其对盲/弱视用户的信任与满意度影响。

**💡 创新点**

将感知到的障碍物信息与规划的未来轨迹结合构建导航上下文，动态生成短语长度≤8词的解释，展示上下文丰富与适时反馈对用户信任的正向作用。

**🔧 技术方法**

采用大语言模型Anthropic Claude Haiku-4-5、YOLOv8n目标检测、ROS2 Nav2规划、Intel RealSense D435i RGB‑D摄像头以及Google TTS等技术。

**📊 数据集**

在室内走廊环境中使用RealSense D435i摄像头实时捕获的图像与YOLO检测结果以及Nav2生成的轨迹；实验数据来自12位参与者的行走记录。

**📈 对比分析**

采用受试者内设计、Repeated‑Measures ANOVA和Pearson相关等统计方法，结果显示解释满意度与信任高度正相关，C2（障碍+轨迹+中频）获得最高评分，信任与满意度均显著优于C1和C3。

**⚠️ 局限性**

机器人缺乏对用户的空间感知，YOLO未能检测墙壁/支架导致轻碰撞；局部定位误差；高频解释偶有信息错误；样本量小，盲人比例低。

---

## 2. CloudIR-Bench: A Benchmark for Root Cause Analysis and Mitigation of Cloud Incidents

**arXiv ID:** 2610.29185 | [PDF](https://arxiv.org/pdf/2610.29185v1)

**作者:** Hoang Tran `[一作]` (RMIT University), Hongyu Zhang `[通讯]` (Chongqing University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提供了第一个公开的云事故报告根因分析与缓解预测基准，包括构建数据集和统一评测框架。

**💡 创新点**

创新点在于：①收集并标注来自5大云服务的≈950条真实事故报告，形成公开数据集；②搭建标准评测环境，支持多种LLM的零样本和上下文学习实验；③对LLM预测结果进行定性错误类型分析，揭示其根因与缓解预测中的常见失败模式。

**🔧 技术方法**

采用大语言模型（GPT‑4.1、GPT‑5.2、Qwen3‑32B、Qwen3‑30B‑A3B、DeepSeek‑R1‑Distill‑*, Llama‑3.1‑8B、Ministral‑8B）、自动摘要、语义/词汇评估指标（BLEU‑4、ROUGE‑L、METEOR、BERTScore）以及In‑Context Learning策略。

**📊 数据集**

数据集来源于5大云服务（Microsoft Azure、Google Cloud、Google Workspace、Atlassian、GitHub）的公开事故报告，包含标题、摘要、影响、根因、缓解等字段，共计约950条记录。

**📈 对比分析**

使用标准API对每条报告生成10个预测，计算Top1/Top5的BLEU‑4、ROUGE‑L、METEOR、BERTScore。实验结果显示GPT‑4.1在根因与缓解两项均表现最强；Qwen3‑32B、Qwen3‑30B‑A3B等开源模型虽落后但表现可观；整体得分仍偏低（根因≈26%准确率，缓解≈26%），ICL在某些数据集上提供小幅提升。

**⚠️ 局限性**

局限性包括：①数据来自公开报告，存在信息泄露、缺失或不完整；②根因与缓解标签为人工注释，可能带有主观性；③评估指标主要衡量文本相似度，无法完全反映因果正确性；④LLM可能在预训练中见过相似报告；⑤仅评测了一批现有LLM，未来模型更新可能影响结论。

---

## 3. Alternative Strategies for Active Learning with Bayesian Bradley Terry Model

**arXiv ID:** 2610.29416 | [PDF](https://arxiv.org/pdf/2610.29416v1)

**作者:** Saman Jayasinghe `[一作]` (Swansea University), Chris Wheadon `[通讯]` (No More Marking Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了贝叶斯 Bradley‑Terry 模型（BBTM）并结合 Polya‑Gamma 数据增强，提出三种基于知识梯度（KG）的主动学习配对策略，以从有限的成对比较中精确重建排名并量化不确定性。

**💡 创新点**

首次将贝叶斯 BT 模型、KG 主动学习和后验分离可靠性（PSR）指标结合，提供理论收敛保证、可计算的 KG 采样准则以及无需参考排名的可靠性评估。

**🔧 技术方法**

采用 Polya‑Gamma 数据增强、Gibbs 采样、知识梯度一阶近似、Mackay 近似、后验一致性与 Bernstein‑von Mises 定理等技术。

**📊 数据集**

主要使用不同规模、噪声水平与预算的合成数据进行实验，并与 BCJ 模型进行对比；未使用真实教育评估数据。

**📈 对比分析**

与 BCJ、随机、无重复、熵、KG‑τ、KG‑Entropy、KG‑SSR 等六种配对策略进行比较，BBTM 在多种噪声下整体优于 BCJ，熵策略在中等噪声下最优，而 KG‑SSR 在高噪声下优于熵。

**⚠️ 局限性**

实验仅基于合成数据，KG 策略在缺乏探索时可能出现配对饥饿；理论收敛仅适用于满足信息增长假设的策略，缺乏对真实教育评估数据的验证。

---

## 4. Human-Robot Interaction via Meta-Learning Based Mode Recognition and Adaptive Control

**arXiv ID:** 2610.28761 | [PDF](https://arxiv.org/pdf/2610.28761v1)

**作者:** Nuo Chen `[一作]` (Dalhousie University), Ya-Jun Pan `[通讯]` (Dalhousie University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种结合元学习模式识别与自适应控制的机器人人机交互框架，实现了在主动引导和被动跟随两种交互模式下的无阈值、实时切换与鲁棒控制。

**💡 创新点**

创新点包括：①使用元学习训练神经网络实现无阈值的交互模式识别，并将自适应估计的力相关参数作为辅助信息显著提升识别率；②提出多轨迹元学习算法(MTML)以同时学习环境基准网络与力相关参数，保证参数仅随力变化而非轨迹变化；③设计结合非奇异终端滑模与自适应元学习的控制器，理论上证明其收敛性并在实验中大幅降低跟踪误差。

**🔧 技术方法**

采用了模型无关元学习(MAML)进行模式识别与环境网络训练，使用三层全连接网络（14→32→2、14→64→32→7）和tanh激活；控制器采用非奇异终端滑模、tanH平滑以及自适应参数更新；全部实现于PyTorch+C++框架，实时循环<1ms。

**📊 数据集**

训练数据来源于7-DOF Franka Emika机械手的11条轨迹（T1–T11），其中T1–T6用于训练、T7–T11用于验证；参与实验的5名受试者在主动和被动模式下收集约25,000条样本；仿真数据在Gazebo中生成，外力分别为0–20 N随机方向。

**📈 对比分析**

与传统PD、学习自适应、NTSM等六种基线控制器以及不同模块的消融版本（NTSM‑AML、NTSM‑K、NTSM‑K‑A）进行比较。实验结果显示，提出的AML控制器在六个未见轨迹–力组合下的平均3D RMSE为9.819 mm，相比NTSM降低了83.0%，ITAE降低85.7%，且控制输入功率保持在与基线相当的水平。

**⚠️ 局限性**

局限性包括：仅在单一交互任务和单台机器人上验证，无法直接证明对更复杂场景或不同机器人结构的泛化；性能高度依赖元学习阶段的数据多样性；对测量噪声的鲁棒性未系统评估；未来需探索更大规模的任务集与动态负载等挑战。

---

## 5. Support periods of polynomials and support-primitive decomposition of constacyclic codes over Galois rings

**arXiv ID:** 2610.29284 | [PDF](https://arxiv.org/pdf/2610.29284v1)

**作者:** Li Zhu `[一作]` (Guizhou Normal University), Hongfeng Wu `[通讯]` (North China University Of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7`

**🎯 论文内容**

未提供论文内容，无法进行总结

**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 6. VeeReview: Measuring Build-Verify Pairs in Agent-Authored Pull Requests from a V-Model Perspective

**arXiv ID:** 2610.29143 | [PDF](https://arxiv.org/pdf/2610.29143v1)

**作者:** Gabriel Dautovi `[一作]` (Michigan Technological University), Jie JW Wu `[通讯]` (Michigan Technological University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发了一种名为VeeReview的工具，利用V‑Model的构建-验证配对原则，对拉取请求（PR）中的验证证据进行自动化分类；随后在匹配的代理与人工PR样本中比较其验证完整度；并对工具的重复性与与人工标注的一致性进行评估。

**💡 创新点**

创新点包括：①将软件工程经典的V‑Model与语言模型相结合，形成一种可解释且可扩展的PR验证测量框架；②引入四个构建-验证对（需求↔验收、系统设计↔系统测试、架构↔集成测试、模块设计↔单元测试），其中第三个对由规则决定，提供了对模型判定的内部对照；③通过标准化（issue链接、测试提交、PR规模）拆解验证与展示的混杂效应。

**🔧 技术方法**

技术主要是：基于大型语言模型（qwen3:32b）的受约束输出；确定性信号抽取（文件类型、CI状态、issue引用、测试文件检测）；模型判定与规则判定的组合；Bootstrap重采样和混合效应逻辑回归（及其校正）来评估组间差异；对模型输出进行重复性和与人工标注的一致性检查。

**📊 数据集**

使用的公开数据集为AIDev（约93万条由OpenAI Codex、GitHub Copilot、Devin、Cursor、Claude Code等代理生成的PR）以及通过匹配得到的相同仓库中对应的人工PR；最终分析样本为741个仓库的1,454对匹配PR。

**📈 对比分析**

比较方法是：对每个PR分别计算四个对的匹配率、构建仅率等指标；采用混合效应Logistic回归（仓库随机截距）以及Bootstrap置信区间进行组间差异检验；通过标准化进一步评估观察到的差异在可观测特征下的残余大小。结果显示：在所有四个对中，代理PR的构建仅率均低于人工PR，差异显著；但在标准化后差异缩小到34–60%；对规则决定的模块设计↔单元测试对，在标准化后差异不显著。

**⚠️ 局限性**

局限性包括：①VeeReview未通过预先设定的80%一致性校准，且人工标注者之间的κ仅为0.154，导致工具精度与真值无法确定；②模型判定的不确定性导致重复性差（约25%状态变化）；③匹配样本不完整，导致对代理整体的外推受限；④标准化变量与判定结果重叠，可能无法完全消除展示对结果的影响；⑤仅使用单一模型配置，缺乏对不同语言模型或参数的泛化评估。

---

## 7. Cryptanalysis and Repair of "Decentralized Multi-authority ABE for NC$^1$ from BDH"

**arXiv ID:** 2610.29224 | [PDF](https://arxiv.org/pdf/2610.29224v1)

**作者:** Xingwang Wang `[一作]` `[通讯]` (East China Normal University), Xingwang Wang (East China Normal University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

对 Datta 等人提出的去中心化多权威属性基加密 (MA-ABE) 架构进行密码分析，发现其存在明文恢复攻击，并提出了基于全零共享的改造方案，恢复了安全性。

**💡 创新点**

创新点包括：①揭示了矩阵标准化技巧在处理坐标掩码时的失效，②设计了能够破坏身份绑定的具体三权威攻击；③引入独立全零共享取代坐标掩码，实现了安全修复；④在随机预言机模型下给出完整的静态安全性证明；⑤将同样的修复推广至 Datta-Pal 的多权威内积功能加密，并在大宇宙场景下提出增强的 L-DBDH 组合假设。

**🔧 技术方法**

核心技术包括：线性秘密共享 (LSSS)、矩阵标准化 lemma、坐标掩码与全零共享的构造、静态安全游戏、随机预言机模型、BDDH 及其变形（Asymmetric DBDH、L-DBDH）假设，以及多方密钥与哈希编程。

**📊 数据集**

该工作不涉及任何真实数据集；所有分析均基于数学模型和抽象安全假设。

**📈 对比分析**

在安全性比较上，作者对比了原始与修复方案的组元数、对数运算量和一次性加密所需的字段算术。修复方案在组元数和乘法次数上保持不变，唯一的额外成本是多项式次数的字段运算和额外随机性；解密时间保持与原方案相同，性能影响可忽略。

**⚠️ 局限性**

局限性包括：1) 仅在随机预言机模型下证明安全；2) 对已发行的旧密文不再安全；3) 攻击和修复针对静态模型，对适应性破坏模型尚未处理；4) 需假设 BDDH（或 L-DBDH），并且在矩阵宽度至少为 3 时才会出现攻击；5) 对单一权威或非坐标掩码设计的方案不适用。

---

## 8. Deep holes of full-length Reed--Solomon codes

**arXiv ID:** 2610.29429 | [PDF](https://arxiv.org/pdf/2610.29429v1)

**作者:** Bocong Chen `[一作]` (South China University of Technology), Jing Huang `[通讯]` (Guangzhou University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对全长 Reed–Solomon 码，在大域（q>19683）下完整解决深洞问题，并给出了对应的覆盖半径与完整性结论。

**💡 创新点**

统一使用奇偶特征区分，首次将 o‑多项式、权重独立集不等式与平面弧完成技术融合，得到在所有维度 2≤k≤q-2 下的完整分类，并给出了奇偶特征下的特例。

**🔧 技术方法**

主要采用二次商代数、图论加权独立集不等式、平面弧完成、计数与概率论等理论工具。

**📊 数据集**

本研究为纯理论证明，没有使用实验数据集。

**📈 对比分析**

通过与已知界限（如根界限、平面弧完整性、MDS 扩展）对比，获得精确的覆盖半径和深洞分类，证明了理论上的最优性。

**⚠️ 局限性**

仅在 q>19683 的范围内成立，仍需进一步工作证明更小 q 的情况；偶特征下 k=q-3 的额外族仍为例外。

---

## 9. VideoGen Model 1: Prompt Expansion Versus Formatting in Audiovisual Generation

**arXiv ID:** 2610.29407 | [PDF](https://arxiv.org/pdf/2610.29407v1)

**作者:** David Grossman `[一作]` (VideoGen), Anton Koenig `[通讯]` (VideoGen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究评估了在视频生成中对原始提示进行创意扩展与仅格式化对比的效果。

**💡 创新点**

创新点在于将提示拆分为“作家室”与“片段重写器”，通过层级创意写作与结构化提示来提升生成质量。

**🔧 技术方法**

使用技术包括 Claude Sonnet 5.5、Claude Opus 5.5、GPT‑6.1 等语言模型，视频生成采用 MiniMax H3 后训练版 VideoGen Model 1，评估使用 VA‑Judger 与 Gemini 3.8 Flash。

**📊 数据集**

数据集为作者自制的30条多类型提示（共90个视频），每条提示在原始、格式化、扩展三种条件下生成。

**📈 对比分析**

比较方法为配对平均差异并采用10,000次bootstrap得到95%区间，结果显示扩展提升整体评分约0.6分，提示一致性和音频质量显著提升；Gemini的提升不显著。

**⚠️ 局限性**

局限性包括仅使用单一生成种子、短视频（≈5 秒）、未验证更长视频或新提示、评审者非独立、以及未分离扩展与背景音乐的贡献。

---

## 10. WaFIR: Wavelet-Fourier Mixture-of-Experts with Subband-Adaptive Routing for Blind All-in-One Image Restoration

**arXiv ID:** 2610.29108 | [PDF](https://arxiv.org/pdf/2610.29108v1)

**作者:** Trong Nhan Nguyen `[一作]` (Pukyong National University), Truong-Son Hy `[通讯]` (University of Alabama at Birmingham)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 WaFIR 模块，在 PromptIR 解码器上通过 Haar 小波分解、Fourier 描述符和稀疏专家路由实现子带级频谱适配的盲图像恢复

**💡 创新点**

子带级频谱路由与 prompt 结合，利用输入衍生的 Fourier 描述符进行频率特征分离与专家分配，提升统一恢复性能

**🔧 技术方法**

Haar DWT、Fourier 频谱描述、稀疏 top‑k 选择、频域专家调制、交叉子带交互、残差门控、PromptIR 基础网络

**📊 数据集**

BSD400、WED、BSD68、Urban100、Rain100L、RESIDE（SOTS）等多种噪声、雨、雾数据集

**📈 对比分析**

与 20 种全局盲恢复方法对比，在 derain、denoise、dehaze 上均实现 PSNR 平均提升 0.48 dB；在未见噪声强度与混合降解上表现更稳健

**⚠️ 局限性**

对极端噪声或多重混合降解的鲁棒性有限；依赖 Haar 小波分解，可能对不同分辨率或其他频率结构的适应性不足

---

## 11. PHD-Hand: A Printable Humanoid Dexterous Hand with Linkage-Driven 3-DOA Digits

**arXiv ID:** 2610.29030 | [PDF](https://arxiv.org/pdf/2610.29030v1)

**作者:** Neel Adwani `[一作]` (New Jersey Institute of Technology), Cong Wang `[通讯]` (New Jersey Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并验证了一种基于链接驱动、弹性驱动杆和柔性铰链的全新灵巧人形机器人手，具备每指三DOA且尺寸接近人类手。

**💡 创新点**

采用单片柔性铰链实现根关节两DOF，弹性钢丝驱动杆简化驱动链，完全解耦根关节与上两关节运动，零部件大幅减少并可用预算3D打印生产。

**🔧 技术方法**

链接驱动机制、弹性驱动杆、柔性铰链设计、RC舵机驱动、AI辅助GUI与运动轨迹生成、GRASP手势评估、物体操作实验等技术。

**📊 数据集**

使用GRASP（Feix）手势分类法的33种抓握手势以及自制3D打印和日常物件做测试，无公开数据集使用。

**📈 对比分析**

与现有链接驱动手（如Linkerbot等）在部件数、尺寸、可打印性等方面对比；在GRASP评估中实现大部分抓握，完成多项物体操作；仅有部分抓握因指关节弯曲受限或缺乏拇指与食指腕间隙未完全实现。

**⚠️ 局限性**

弹性钢丝偶发断裂、根关节未包含轴向旋转、缺乏人类手的掌面和拇指与食指的腕间隙、未对驱动杆疲劳寿命进行定量评估、未完全实现某些高级抓握。

---

## 12. BIRD-RL: Scaling Agentic Reinforcement Learning over Stateful Data-Centric Environments

**arXiv ID:** 2610.29426 | [PDF](https://arxiv.org/pdf/2610.29426v1)

**作者:** Ge Qu `[一作]` (University of Hong Kong), Reynold Cheng `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套名为 BIRD‑RL 的四阶段强化学习课程，并实现了轨迹级持久执行基础设施，用于在状态化数据库环境中训练 SQL 问题调试与生成语言代理。

**💡 创新点**

核心创新包括：① 轨迹级持久执行架构实现 12.8 倍的空间效率，解决传统每轮容器化导致的资源浪费；② 将思维、工具使用、适应性强化与交互式强化四阶段拆解，稳定并提升自适应调试能力；③ 通过数据混合训练实现统一的调试与生成双目标模型，展示多任务学习潜力。

**🔧 技术方法**

采用 Qwen2.5‑Coder 作为基座，结合监督微调（SFT）、GRPO 强化学习、SQL‑Act 工具动作、轨迹级数据库池、定制化状态化代理循环、仅基于执行结果的稀疏奖励。

**📊 数据集**

使用 BIRD‑CRITIC‑SQLite（500 条 SQLite 诊断案例）、BIRD‑CRITIC‑Multi（570 条多数据库案例）、BIRD‑SQL 开发集（1,534 条生成示例）以及 SIX‑GYM‑SQLite 进行数据生成与评测。

**📈 对比分析**

在 BIRD‑CRITIC‑SQLite 上与 GPT、Claude、Gemini 等通用 LLM、SQL‑Act 等多轮代理以及 RL 生成模型对比，7B 模型 SR 44.40%，14B 模型 SR 48.00%，仅落后 Claude‑Opus‑4.6（48.20%）1%；统一模型在生成任务上取得 63.9% 执行准确率，接近专门 RL 生成模型。

**⚠️ 局限性**

局限性包括：需要大规模 GPU 资源与高昂的离线池成本；对数据库类型的泛化仍有限，主要验证于 SQLite；奖励设计仍易受奖励剥削影响，需进一步完善；依赖人工标注的调试案例，实际业务场景数据仍缺乏。

---

## 13. Hierarchical Masked Diffusion Language Models

**arXiv ID:** 2610.29255 | [PDF](https://arxiv.org/pdf/2610.29255v1)

**作者:** Parul Gupta `[一作]` (Indian Institute of Technology Delhi), Parag Singla `[通讯]` (IBM Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种层次化掩码扩散语言模型（HMDLM），通过在扩散过程中显式编码token位置的优先级，分两阶段先预测层次再预测token，实现并行生成。

**💡 创新点**

创新点在于：① 设计了层级特定的噪声调度与层次嵌入，让不同优先级位置在正向噪声与逆向去噪时有不同的曝光速率；② 将后向推断拆解为两步，先恢复层次再恢复token，形成可分解的ELBO；③ 在无额外标注的情况下仅利用已有的任务层次信息即可训练。

**🔧 技术方法**

采用了离散扩散模型、层次化掩码策略、两阶段神经网络（层次头+token头）、Per-level masking schedule、ELBO推导与自回归评估技术，并在Transformer骨干上实现。

**📊 数据集**

实验数据集包括大规模语言模型基准LM1B、结构化指令集WikiHow以及多步推理任务Sudoku；每个任务都用不同的层次划分（词频、位置、解题轮次）来定义层级。

**📈 对比分析**

与MDLM、SEDD、ADLM、DiffusionBERT、MGDM等现有扩散基线相比，HMDLM在LM1B和WikiHow上PPL下降约1.5–4.4点、MAUVE提升、LLM评审偏好率超过两倍；在Sudoku上解题准确率提升至约8%（从81%提升到88%），并在单元格级别显著提升。

**⚠️ 局限性**

主要局限在于：两阶段推断导致计算量增加；层次划分依赖任务特定规则或启发式，缺乏通用自适应机制；在某些任务（如高频词几乎无层次差异时）难以体现优势。

---

## 14. FutureLens: Longitudinal Topic Patterns in Scholarly Text and Future-Work References

**arXiv ID:** 2610.28656 | [PDF](https://arxiv.org/pdf/2610.28656v1)

**作者:** Ibrahim Al Azher `[一作]` (Northern Illinois University), Hamed Alhoori `[通讯]` (Northern Illinois University)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究对2012–2022年ACL会议论文及其未来工作声明进行纵向主题建模，探究主题变化与当前研究之间的关系。

**💡 创新点**

创新点在于系统评估两种常用主题模型（NMF与LDA）的一致性、可解释性与趋势稳定性，并引入多重指标（NPMI、ARI、Sen斜率、Spearman相关和Jaccard重叠）来描述未来工作文本与主体文本的相似度与差异。

**🔧 技术方法**

使用非负矩阵分解（NMF）和潜在狄利克雷分配（LDA）对全文摘要及标注文本进行词袋/TF‑IDF特征化，并通过预设K=20的主题空间进行统一比较。

**📊 数据集**

数据集为5,655篇ACL论文（2012–2022年），其中4,005篇提供了未来工作引用，覆盖率约70%。

**📈 对比分析**

对比结果显示，NMF在主题一致性（NPMI平均0.1387）和分配稳定性（ARI范围0.418–0.743）上优于LDA；LDA在关键词多样性（0.890）上更突出；两模型在主题趋势、覆盖率与源间一致性方面呈现互补的优劣。

**⚠️ 局限性**

局限性包括：数据覆盖不均（如2015年仅33篇），近30%论文缺失未来工作引用，引用来源未得到验证，模型参数固定（K=20）且未探究其对结果的敏感性，且仅提供描述性分析，未检验未来工作是否被后续研究采纳。

---

## 15. OACap: Caption-Guided Active Observation for Detailed Video Captioning

**arXiv ID:** 2610.29327 | [PDF](https://arxiv.org/pdf/2610.29327v1)

**作者:** Hao Shi `[一作]` (Ant Group), Chao Huang `[通讯]` (Ant Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Observation‑Adaptive Video Captioner (OACap)，通过对逐步生成的字幕进行缺失/不支持信息检测，驱动视觉工具（如裁剪、OCR、密集采样等）获取证据，并在每次迭代中验证并修正字幕，最终生成更完整、可验证的详细视频描述。

**💡 创新点**

创新点在于：①将字幕本身视为可检验的假设，用缺失/矛盾信息来触发目标工具调用；②区分可用现有证据修复的“Revision Gap”与需要额外观察的“Observation Gap”；③构建 FSCap‑Bench 评估框架，提供 Core–Support Factuality Score (CSFS) 与 Cross‑Modal Semantic Round‑Trip Score (CSRS)，对事实完整性与跨模态可恢复性进行量化。

**🔧 技术方法**

采用大语言模型与多模态工具集成、证据状态管理、声明验证与迭代修订；使用固定文本阅读器进行下游问答评测；使用冻结的 atomizer 与 judge 对生成字幕进行原子化、匹配与评分。

**📊 数据集**

基准数据集包括 Video‑MME、ActivityNet‑QA、VCapsBench、Video‑MMM‑U 与 MVBench；从这些公开数据集中筛选 498 条视频，构建 996 条观察挑战，形成 FSCap‑Bench。

**📈 对比分析**

与单次通过基线以及匹配模型（如 Qwen、LTX 等）在 “字幕即记忆” QA、CSFS 与 CSRS 上进行对比。OACap 在长视频分区提升 8–10% 的问答准确率；CSFS 在三种 backbone 上分别提升 1.9、5.2 与 20.7 分；CSRS 在 10–60 秒视频中最高可达 0.291，表明信息保留率显著提高。

**⚠️ 局限性**

局限性：①额外的观察调用导致推理成本上升，尚未构建成本-质量 Pareto 前沿；②缺失检测与工具校准误差可能导致错误的修订或观察；③评测覆盖面有限（仅 33 条视频用于 CSRS、部分数据集重叠），无法完全验证泛化性能；④跨模态重构通道仍有 70–80% 的信息损失。

---

## 16. Rethinking Agentic Model Orchestration for Time Series Forecasting

**arXiv ID:** 2610.29504 | [PDF](https://arxiv.org/pdf/2610.29504v1)

**作者:** Tong Guan `[一作]` (Griffith University), Ming Jin `[通讯]` (Griffith University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个冻结预测银行的对照实验框架，系统化评估时间序列基础模型（TSFM）的模型编排（AMO）效果

**💡 创新点**

提出了 Freeze Forecast Bank 评估协议，独立控制候选模型强度和互补性，清晰分离编排器自身的贡献

**🔧 技术方法**

使用冻结预测库、候选模型选取与加权、LLM 中介、特征条件化、历史误差驱动的编排器，基于 GIFT‑Eval 的官方 MASE/CRPS 评估

**📊 数据集**

在 GIFT‑Eval（超过 144,000 条时序、7 个领域）上进行实验

**📈 对比分析**

与五种基准（随机、Dev‑Selected Best、Equal Weight、Argmin Router、Oracle）及五个公开编排器（TimeRouter、Toto‑2.0 FnF、RAES‑Conductance‑Ensemble‑V、TSOrchestra SLSQP、TimeCopilot Median）以及两种自实现编排器进行比较，最高可获得约 5.6% MASE 提升（0.658 vs 0.697），但相较于窗口级 Oracle 仍有约 30% 余地

**⚠️ 局限性**

限于单一决策且仅使用最近若干回测，历史证据增益快速饱和；LLM 中介的优势不明显，且对候选组合极度敏感

---

## 17. AutoPhyX: Automatic Text-Condition Physics Property Generation

**arXiv ID:** 2610.29517 | [PDF](https://arxiv.org/pdf/2610.29517v1)

**作者:** Bei Huang `[一作]` (Peking University), Siyuan Huang `[通讯]` (BIGAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 AutoPhyX，一个通过文本条件自动生成稠密物理属性（杨氏模量、泊松比、密度）的前向框架，可一次性预测三维资产的体素化物理参数。

**💡 创新点**

创新点在于：① 引入文本条件与视觉特征的跨模态调制（FiLM）实现对视觉模糊的物理区分；② 设计了基于 VLM 的自动分段与属性生成管线，构建大规模多部件物理数据集；③ 将预测结果统一为体素场，兼容多种 3D 表示（网格、点云、Gaussian Splatting、NeRF）。

**🔧 技术方法**

核心技术包括 OpenCLIP 特征提取、基于体素渲染的 3D 特征场学习、FiLM 交叉模态调制、密集体素填充、VLM（如 Gemini）驱动的文本分段与属性查询，以及在多种物理求解器（MPM、FEM、PhysGaussian）中的后向渲染评估。

**📊 数据集**

使用了自建的 1,700 体素资产与 8 条文本描述的 13,600 条训练样本（AutoPhyX-1K 数据集），以及公开的 ABO‑500 真实物体质量数据集进行泛化评估。

**📈 对比分析**

与多种基线（NeRF2Physics、Pixie、VoMP、DreamPhysics、OmniPhysGS、Gemini）对比，AutoPhyX 在 PSNR、SSIM、LPIPS、物理误差（log E、ν、log ρ）以及 ABO‑500 上的 ADE、ARE、ALDE、MnRE 等指标均优于或接近最优，且单次前向推理时间仅约 2 秒，显著快于基于优化的方法。

**⚠️ 局限性**

局限性包括：① 仅覆盖三种基础物理属性，缺少摩擦、阻尼等表面/能量损失参数；② 对非常细小结构的高分辨率预测尚未实现；③ 仍需依赖 VLM 生成文本与属性的可靠性，可能受到 VLM hallucination 或数据库缺失的影响。

---

## 18. Building AI Research Teams for Autonomous Discovery of Linear Attention Architectures

**arXiv ID:** 2610.28887 | [PDF](https://arxiv.org/pdf/2610.28887v1)

**作者:** Yujian Liu `[一作]` (University Of California Santa Barbara), Shiyu Chang `[通讯]` (University Of California Santa Barbara)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一支由不同科学背景、具备持续记忆的 AI 研究员组成的协同团队，利用自适应分配与共享实验结果，在实验成本有限的情况下，对线性注意力架构进行迭代式探索，并在 150M 规模下发现 Context‑First DeltaNet (CFDN)，随后在 1.3B 规模验证其性能优于 GDN2、GDN、KDA 等基线；

**💡 创新点**

创新点在于：①将多学科背景与个体长期记忆结合，形成横向互补的探索；②采用基于 UCB 的自适应研究员选择，平衡探索与利用；③引入共享实验记录与反思机制，使实验结果能指导后续方向；④在资源受限的实验预算下实现跨尺度的结构改进与验证；

**🔧 技术方法**

主要技术包括：基于 GPT‑6Astra 的语言模型驱动的自动化研究管线；UCB 研究员选择算法；实验奖励函数（基于模型 utility 与前一最佳的差值）；共享实验数据与私有历史记录的交互；线性注意力架构的递归记忆更新机制；对候选架构的实现、训练与评估；

**📊 数据集**

使用 FineWeb‑Edu 训练语料；评估指标涵盖语言建模（WikiText、LAMBADA）、常识推理（8 个任务）、合成检索（4 个 NIAH 任务）以及 1.3B 规模下的真实检索（6 个数据集）。

**📈 对比分析**

通过对比不同组织方案（无背景、无私有历史、统一选择）以及单一目标与多目标搜索，证明全系统在 150M 规模下的最高 utility 为 1.04，优于其它方案；CFDN 在 150M 规模下提升 LM perplexity、CSR 与 NIAH；在 1.3B 规模下 CFND 获得最低 perplexity 与最高各项指标，超过所有基线；

**⚠️ 局限性**

局限性包括：①实验预算仍受限，未探索更大规模或更广阔搜索空间；②仅聚焦于线性注意力，方法可推广但未验证；③奖励函数依赖已定义的 metrics，可能忽略其他重要能力；④团队成员仍基于同一 LLM 模型，缺乏多模态或多代理多语言差异。

---

## 19. Data Schedules and Learning Rate Schedules Should Be Designed Together

**arXiv ID:** 2610.29348 | [PDF](https://arxiv.org/pdf/2610.29348v1)

**作者:** Haodong Wen `[一作]` (Tsinghua University), Kaifeng Lyu `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在LLM的中训练阶段，研究了数据混合比例与学习率衰减时间的联合调度，探讨了在固定计算和MT数据预算下，何时引入MT数据能最小化最终MT损失。

**💡 创新点**

提出“排名逆转”现象——不同LR调度会导致最优数据调度发生倒置，并证明了在线性LR衰减下，早期混合PT和MT数据比单纯晚期MT更优；同时给出理论模型解释这一现象。

**🔧 技术方法**

采用连续时间SDE近似的SGD分析、块对角二次损失模型、以及实验性的数据调度方案（Two‑Stage、Zero‑then‑Linear）和LR调度方案（Constant、Full Decay Range、Match Decay Range）。

**📊 数据集**

使用PT数据集DCLM‑Baseline（≈200B token）与MT数据集FineMath‑4plus（≈10B token）进行实验，覆盖dense 630M Qwen2.5‑0.5B模型与2.99B MoE模型。

**📈 对比分析**

在固定MT预算下，对比TS与ZL数据调度配合Constant、FDR、MDR三种LR调度；实验显示在FDR下早期混合PT/MT能降低MT损失，在Constant下则纯MT更好；最佳LR衰减时间约为MT阶段结束前的10–40%，相较于传统做法提升了数个百分点的MT损失。

**⚠️ 局限性**

仅关注最终MT损失，未综合PT损失或下游任务指标；LR调度被简化为线性衰减；理论模型基于二次损失和特定谱假设，可能与真实LLM训练差异；未给出全局最优联合调度的闭式解。

---

## 20. StochIKM: Automated High-Fidelity Reconstruction of Individual Patient Data from Published Kaplan-Meier Curves

**arXiv ID:** 2610.29122 | [PDF](https://arxiv.org/pdf/2610.29122v1)

**作者:** Liangze Ke `[一作]` (Yale University), Guannan Gong `[通讯]` (Yale University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了 StochIKM 算法，用来从已发表的 Kaplan–Meier 曲线自动重建高保真度的个体患者数据 (IPD)。

**💡 创新点**

创新点包括：① 用三条可量化的约束（风险表一致、曲线拟合精度、印刷摘要一致）严格定义“保真度”；② 采用随机采样产生一组可接受的 IPD，捕捉重建的不确定性；③ 通过逐步放宽约束的“Relaxation”机制提升对低质量图像的鲁棒性；④ 利用 vision‑capable 大语言模型（如 GPT‑5）实现全自动图像读取。

**🔧 技术方法**

技术手段包括：图像识别与 LLM 读数、概率采样与 K–M 逆推、三层约束判定、误差容限层级放宽、批量采样（Ω=50）以及多轮搜索。

**📊 数据集**

数据集：79 条来自 53 篇肿瘤临床试验（共 108 arm）的 KM 图与风险表；以及若干 Weibull 模拟数据（n=50/350/1000，censoring 10%/30%/50%，K=4/8/12）。

**📈 对比分析**

与 LLM‑improved IPDfromKM 做对比：在真实数据中，StochIKM 在所有 97 arm 中完美重现风险表，平均中位数误差为 0.00 个月（最大 0.91），而 IPDfromKM 平均误差 0.10 个月；两者在置信区间重叠度上相当（均 0.94）。在仿真中，StochIKM 产生的曲线与原始数据无偏差、标准误为 0，IPDfromKM 仍出现小偏差且误差随样本量、删失率而变大。

**⚠️ 局限性**

局限性：高度依赖图像的解析质量，当前未能读取删失标记；若风险表或图像不一致，算法可能失败；在极低分辨率或复杂线型的图形中仍需人工干预。

---

## 21. Optimal Labeling Schemes for $Γ$-Free Matrices

**arXiv ID:** 2610.29184 | [PDF](https://arxiv.org/pdf/2610.29184v1)

**作者:** Sankardeep Chakraborty `[一作]` (University of Tokyo), Seungbum Jo `[通讯]` (Chungnam National University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了一种对 Γ‑free 二进制矩阵的行列标签方案，使得任意矩阵元素可在常数时间内通过对应两个标签恢复。

**💡 创新点**

创新点在于构造了一个祖先单调的森林序列，并利用基于排名的固态路径分解与持久化标识符，将森林演化历史压缩到 O(log²N) 位，达到了信息论下界的最优标签大小；同时给出了 Γ‑free 余闭包的线性时间算法。

**🔧 技术方法**

核心技术包括：祖先单调森林表示、固态路径分解与排名稳定性、持久化路径标识符、预先预计算版本号的前驱搜索（fusion tree 结构），以及基于二进制搜索或常数时间前驱查询的解码器。

**📊 数据集**

本工作为理论性研究，没有使用公开数据集；所有结论均通过数学证明与计数下界验证。

**📈 对比分析**

通过信息论计数证明，任何 Γ‑free 矩阵在最坏情况下至少需要 Ω(log²N) 位标签；该方案实现了 O(log²N) 位，并在单词 RAM 模型上实现常数时间查询，显著优于此前仅提供全局压缩表示的 Spinrad 方案，且在所有已知图类（强弦图、弦图二分图、β‑无环超图等）上得到最优标记量。

**⚠️ 局限性**

局限性包括：仅适用于 Γ‑free 结构（对应的图类）；标签生成与查询实现相对复杂；当前方案仅支持邻接/矩阵查询，无法直接扩展到邻居列举、距离或最短路等更复杂查询；在实现上需要高位宽词级运算，可能在实际硬件上效率受限。

---

## 22. The Butterfly Effect in LLM Social Simulations: From Architecture Sensitivity to Robustness Audits

**arXiv ID:** 2610.28981 | [PDF](https://arxiv.org/pdf/2610.28981v1)

**作者:** Jinyi Ye `[一作]` (University of Southern California), Emilio Ferrara `[通讯]` (University of Southern California)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在重复囚徒困境和社交媒体回音壁两种模拟环境中，系统地评估了LLM代理在不同表示层与设计层扰动下的鲁棒性，并揭示了蝴蝶效应在多代理互动中的放大或衰减作用

**💡 创新点**

提出了TRAILS（LLM模拟鲁棒性审计分类法）和社交模拟审计卡，用以结构化地审计与解释模拟结果的敏感性与可靠性

**🔧 技术方法**

使用四大LLM家族（GPT‑5.2、Claude Haiku 4.5、Gemini 2.5 Flash、DeepSeek）配合自定义提示与内存策略，构建并运行代理互动

**📊 数据集**

采用合成的数据集：10轮囚徒困境的对抗匹配与固定策略，以及15轮回音壁网络中的100名代理及其预设立场和网络结构；无外部真实数据来源

**📈 对比分析**

通过比较不同扰动和模型下的合作率、平均收益、立场亲和度和同组边缘比例等指标，发现某些模型在表示层扰动下产生数十个百分点的性能变化；效应随模型与扰动而异

**⚠️ 局限性**

研究范围受限于两种模拟设置、固定交互长度、预设代理立场及仅选取的四个模型；结果的普适性与与人类行为的对应仍需进一步实验验证

---

## 23. VHDLBench: Benchmarking and Fine-Tuning Large Language Models for VHDL-Based Chip Design

**arXiv ID:** 2610.28676 | [PDF](https://arxiv.org/pdf/2610.28676v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 24. Directional Posterior Feedback Fusion for Query Reformulation in Dense Retrieval

**arXiv ID:** 2610.29464 | [PDF](https://arxiv.org/pdf/2610.29464v1)

**作者:** Yiteng Tu `[一作]` (Tsinghua University), Qingyao Ai `[通讯]` (Tsinghua University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于方向后验融合（DIFF）的查询改写框架，用于在密集检索中融合多来源反馈以改进查询向量；

**💡 创新点**

创新点在于将反馈向量视为切线空间中的方向性约束，而非完整目标，并通过分层置信度控制（局部与集合级）实现更稳健、可控的后验融合；

**🔧 技术方法**

采用L2归一化、投影到查询的切线空间、贝叶斯后验推理（MAP）闭式求解、基于局部/集合置信度的权重分配以及球面更新技术；

**📊 数据集**

使用BEIR 14个子数据集（ArgoAna、Climate-FEVER、DBPedia、FEVER、FiQA、HotpotQA、MSMARCO、NFCorpus、Natural Questions、Quora、SCIDOCS、SciFact、Touche2020、TREC-COVID）以及三种预训练检索器（BGE、E5、Qwen）进行实验；

**📈 对比分析**

与原始检索、VPRF、Concat-P/L、OR-PRF、DIME、Eclipse、AnchorQE等基线比较，DIFF在NDCG@10和Recall@100上均显著提升，尤其在PRF和LLM生成的GRF（HyDE、Q2D、CoT）下表现最稳健，并且不需额外检索或模型训练；

**⚠️ 局限性**

局限性包括仅在局部切线空间建模，未捕获全局几何；对极大规模反馈（K）仍受限；缺乏针对多模态或多语言检索的直接支持。

---

## 25. Benchmarking the Quantum Leap: Auditing LLM Evaluation for Quantum Code Generation

**arXiv ID:** 2610.28678 | [PDF](https://arxiv.org/pdf/2610.28678v1)

**作者:** Ethan Seefried `[一作]` (Oak Ridge National Laboratory), Tirthankar Ghosal `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对15种量子代码生成LLM进行统一评测，构建新的QuantumBenchEval基准并公开评测平台，系统审计评估器缺陷。

**💡 创新点**

首次将置信区间、配对多重比较和评估器审计引入量子代码生成评测，提出量化不确定性与评估者可靠性的评估框架。

**🔧 技术方法**

采用任务级bootstrap置信区间、Holm多重比较校正、配对任务比较、LLM判断、执行单元测试、量子电路验证等技术。

**📊 数据集**

使用5大已发布基准（QiskitHumanEval Standard/Hard、QuanBench-44/117、QCoder）以及新构建的96题QuantumBenchEval。

**📈 对比分析**

通过pass@k、置信区间和配对比较评估模型，发现主基准已趋于饱和：Standard最高达98.7%，Hard 63.8%，不同模型间差异在多重比较后不显著，评估者缺陷可导致数十个百分点偏差。

**⚠️ 局限性**

评估器缺陷需重跑、QBE受评估器版本影响、QCoder采用非i.i.d.流程、模型解码/输出预算不统一、实验仅在单一时间点完成。

---

## 26. Dynamics of Hidden Learning Structure

**arXiv ID:** 2610.29054 | [PDF](https://arxiv.org/pdf/2610.29054v1)

**作者:** Qinyou Wang `[一作]` `[通讯]`, Qinyou Wang

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究学习系统中受保护修复与结构获取的动态相互作用，并在 Qwen2.5-32B 上实现头修复与 LoRA 续期的协同策略。

**💡 创新点**

提出 Fiber Regulation Theory (FRT) 与 Fiber Acquisition Theory (FAT) 的联合可行性与响应可达性理论，并证明两种操作协同可产生单独操作无法实现的学习响应。

**🔧 技术方法**

结合受保护梯度编辑、低秩低维更新、AdamW 优化、回放与输出蒸馏以及凸优化、伪逆、奇异值分解等数学工具。

**📊 数据集**

使用 MASSIVE 1.1 English 版数据集中的 16 意图，每个意图 64 训练样本与 8 评估样本。

**📈 对比分析**

在两条训练轨迹（不同随机种子/示例顺序）下比较基线、FRT、FAT、Joint FRT–FAT 四种策略，联合策略在 Exact 与宏观 NLL 上均优于单独策略，尤其在变化轨迹上同时提升两项指标。

**⚠️ 局限性**

结果仅在 Qwen2.5-32B 和特定任务流下验证，缺乏更广泛模型/数据集、全局可推广性与更完整的连续价值证明；理论依赖线性/可微假设，非线性受保护修复尚未完全覆盖。

---

## 27. Unlocking Spatial Grounding in Flow-Matching Models via SNR Tuning

**arXiv ID:** 2610.29581 | [PDF](https://arxiv.org/pdf/2610.29581v1)

**作者:** Martin Thißen `[一作]` (Darmstadt University of Applied Sciences), Elke Hergenröther `[通讯]` (Darmstadt University of Applied Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过仅调节流匹配模型的SNR分布，实现文本提示直接生成精确布局图像，无需结构插件或推理时调度。

**💡 创新点**

发现高噪声阶段是布局学习的关键，通过SNR分布偏移解锁模型本地空间约束能力。

**🔧 技术方法**

利用流匹配（flow‑matching）框架、logit‑normal噪声分布、VAE重构、CLIP编码器，并采用彩色分割mIoU作为评测指标。

**📊 数据集**

在合成几何原语数据集和COCO 2017真实图像上进行训练与评估。

**📈 对比分析**

与传统控制网络（ControlNet、GLIGEN等）及布局到图像方法对比，合成集上mIoU高达80%+，COCO上AP^50达81.8、AP 57.3，显著超过现有专用模型。

**⚠️ 局限性**

高噪声训练虽然提升布局精度，但会削弱高频细节生成，导致纹理细节不足；需进一步平衡结构与细节的学习过程。

---

## 28. Right On Q: Recovering Reasoning under Quantization via Self-Generated Trajectories

**arXiv ID:** 2610.29378 | [PDF](https://arxiv.org/pdf/2610.29378v1)

**作者:** Boyuan Zheng `[一作]` (University of Michigan), Z. Morley Mao `[通讯]` (University of Michigan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种结合自上政策蒸馏（OPD）和候选动态KL（DKL）的低比特恢复管线，针对量化后LLM在长推理任务中出现的轨迹发散和分布失衡问题进行改进。

**💡 创新点**

创新点在于：①使用自上政策蒸馏直接在学生自身生成的前缀上进行监督，消除训练与推理之间的轨迹偏差；②引入候选动态KL，根据教师-学生概率比自适应地平衡正向与逆向KL，精准纠正低位量化模型在高熵位置的过低与过高概率。

**🔧 技术方法**

技术包括：统一非对称权重量化、块级量化校准（Block‑AP）、自上政策蒸馏、候选动态KL损失、响应长度调度与截断策略、AdamW微调。

**📊 数据集**

使用了多域提示语料库（约25600条，包括数学、代码、科学和指令），并在九个公开基准上评估：GSM8K、MATH‑500、AIME 24–26、HumanEval+、MBPP+、LiveCodeBench、MMLU‑Redux、GPQA‑Diamond、IFEval。

**📈 对比分析**

与GPTQ、AutoRound等PTQ方法及EfficientQAT、BitDistiller、ReasoningQAT等QAT基线相比，本文方法在所有六个模型/比特组合中均取得最高平均分，尤其在2‑bit时对Qwen3‑8B平均提升10.26分，长推理任务（如AIME、LiveCodeBench）提升幅度最大，生成长度更短、截断率更低。

**⚠️ 局限性**

局限性包括：①仍需针对不同任务调节截断长度与学习率；②主要针对权重量化，未覆盖激活量化与混合精度策略；③对更大规模模型或不同架构的迁移性尚未充分验证。

---

## 29. From Scores to Safe Sets: Exact Safety--Constrained Decoding for Medication Recommendation

**arXiv ID:** 2610.29228 | [PDF](https://arxiv.org/pdf/2610.29228v1)

**作者:** Animesh Agarwal `[一作]` (Optum), Carlos Morato `[通讯]` (Optum)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在药物推荐任务中，作者固定药物分数、推荐集合大小和DDI约束，比较了全局混合整数线性规划（MILP）解码与几种局部修复方法的性能。

**💡 创新点**

创新点在于将解码阶段视为决定性问题，证明即使分数已冻结，全局约束解码仍能显著提升药物集合的准确率，突破局部修复的局限。

**🔧 技术方法**

采用混合整数线性规划（MILP）、贪心修复、度敏感修复、1-交换局部搜索，并使用HiGHS求解器；评分来源于FLAME冻结的药物级分类器。

**📊 数据集**

使用MIMIC-III临床数据库（约14,000次就诊），151种药物和基于TWOSIDES的DDI图（1,425条边）。

**📈 对比分析**

在loose、moderate、strict三种DDI预算下，与贪心、度敏感和1-交换局部搜索对比，MILP在最严格预算下平均提升约0.0064 Jaccard，优势随DDI限制收紧而增大；整体性能在验证集和测试集均保持一致。

**⚠️ 局限性**

实验仅覆盖单一数据集和评分器，DDI约束为二元图，未考虑剂量、严重性等临床细节；未评估大规模词汇量或更复杂约束下的可扩展性与运行时成本。

---

## 30. Attention-based Classification for Indirect Prompt Injection

**arXiv ID:** 2610.28988 | [PDF](https://arxiv.org/pdf/2610.28988v1)

**作者:** Amelia Kawasaki `[一作]`, Stefan Lee `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估并改进 Attention Tracker 以检测代理式间接提示注入攻击，首先在 AgentDojo 等多步、工具调用的环境中验证其性能，然后提出用 Fisher LDA 替代标量阈值并对协方差结构进行消融，进一步提升跨域检测能力。

**💡 创新点**

将单一标量关注分数阈值替换为基于 Fisher 线性判别的多维头分数分类器，并通过对协方差结构（全、块、对角、恒等）进行消融，证明忽略头间相关性可显著提升跨域泛化。

**🔧 技术方法**

使用 Transformer 关注头分数聚合、Fisher 线性判别分析、协方差结构消融、交叉数据集评估，以及与隐藏状态特征的 Logistic 回归分类器对比等技术。

**📊 数据集**

利用 AgentDojo、InjecAgent、LLMail‑Inject 等多步代理式攻击数据集，以及 deepset 单轮注入数据集进行实验。

**📈 对比分析**

与原 Attention Tracker 以及特征级 Logistic 回归比较，在单域内平均 AUROC 0.94（LDA）对比 0.62（AT），在跨域评估中 LDA 平均 AUROC 0.79 领先 AT 的 0.75，表明改进方法在不同域上表现更稳健。

**⚠️ 局限性**

实验受限于样本量极小、仅使用两套代理式数据集、协方差消融仅在单模型单数据集上完成、且 LLMail‑Inject 的正负样本需人工合成，导致结果可能不具普适性和泛化。

---

## 31. Auditing LLM Alpha-Factor Mining: Metric Provenance and Full-Cohort Reanalysis

**arXiv ID:** 2610.29500 | [PDF](https://arxiv.org/pdf/2610.29500v1)

**作者:** Yufeng He `[一作]` `[通讯]` (University of Hong Kong), Yufeng He (University of Hong Kong)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对一个基于LLM的因子挖掘项目进行了完整的审计，包括历史运行计数器、结果保留和公开再分析，以揭示指标不一致和数据保留问题。

**💡 创新点**

提出了基于artifact-level的审计方法，强调指标粒度一致性、不可变队列核算和来源可追溯解释，而非仅靠架构比较。

**🔧 技术方法**

使用Claude系列大模型、进化式表达式搜索、回测器、Rank-IC统计、Python版公共评估器和数据哈希校验。

**📊 数据集**

对来自3,114条保留输入的1,047条冻结表达式进行评估，并使用2023–2025年40支股票的727个交易日价格/成交量数据。

**📈 对比分析**

通过对比管道与自适应群体的保留决策、Sharpe均值和成功率，发现指标种群不匹配；在公开再分析中923条可评分表达式中有671条正向rank-IC，但并未表明任何架构优势。

**⚠️ 局限性**

缺乏完整的执行路径与版本控制，指标粒度不一致，OOS评估不完整，市场与股票池覆盖有限，未能验证真实收益。

---

## 32. Resuming Vision-Language Inference across Onboard Interruptions

**arXiv ID:** 2610.29183 | [PDF](https://arxiv.org/pdf/2610.29183v1)

**作者:** Xiaolong Huang `[一作]` (Peking University), Xuanzhe Liu `[通讯]` (Peking University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 OrbitWeaver，一套面向低地球轨道卫星的视觉‑语言模型推理运行时，能够在总线强制断电等突发中断后通过协调的检查点与恢复继续推理。

**💡 创新点**

创新点：①相位感知检查点（vision 层、prefill 块、decoding 组）在保持恢复进度的同时极大减少写入；②基于依赖的动态恢复调度，利用层级与块级依赖关系在 I/O 与计算之间实现最优重叠；③专门的存储布局、校验与压缩策略保证恢复可靠且写放大低；④整个框架可无缝集成至现有 VLM 推理引擎。

**🔧 技术方法**

技术手段包括：分块检查点、层/块级 I/O 合并、层级/块级优先级调度的 DP 规划、KV 缓存压缩、检查点完整性校验、异步 KV 量化与压缩、GPU/CPU 结合的并行实现。

**📊 数据集**

使用 LRS‑VQA 远程感知视觉‑语言问答数据集，模型包括 Qwen3‑VL 2B/4B/8B 以及 MiniCPM‑V 4.5。

**📈 对比分析**

与无检查点、SONIC‑layer、SONIC‑chunk、HCache 四个基线在 Jetson Orin/AGX 设备上比较。OrbitWeaver 在所有设置下实现 2.95–4.07× 的总耗时加速、3.56× 的 SSD 写入量减少、5–6× 的恢复延迟降低，且正常运行时开销 <5%。

**⚠️ 局限性**

局限性：仅支持单请求单 GPU 推理；对 SSD 持久性与空间环境的辐射影响未考虑；在更大模型或多请求、分布式星座场景下的可扩展性未验证；需要预先收集中断轨迹才能做最优调度。

---

## 33. CSD: Common-Support Distillation for Cross-Tokenizer Knowledge Transfer

**arXiv ID:** 2610.28895 | [PDF](https://arxiv.org/pdf/2610.28895v1)

**作者:** Mingyu Yang `[一作]`, Pavlo Molchanov `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种跨 tokenizer 的知识蒸馏方法 Common‑Support Distillation（CSD），通过构建共享的文本等价事件来对齐教师与学生的 logits，且保留残差事件以归一化概率分布；

**💡 创新点**

创新点在于：①统一跨 tokenizer 的匹配目标，消除需手工挑选损失模式的困扰；②用字符级对齐生成共享支持，既可处理一对一，也可处理一对多/多对一，并通过链式概率计算得到多 token 事件；③在非一对一情形下引入残差事件，实现对未显式支持的整体监督；

**🔧 技术方法**

技术手段包括：字符级偏移对齐、基于链式概率的多 token 事件生成、残差事件归一化、top‑k 选取、动态联合训练（CE+KD 的比例自适应）以及对不同 chunk 形状统一使用同一 KL / JSD 损失；

**📊 数据集**

主要使用 NVIDIA CLIMB 语料进行训练，评估任务涵盖 MMLU、GSM8K、MATH、WinoGrande、MBPP、IFEval 等英文知识问答与推理任务；

**📈 对比分析**

与 ALM、ULD、GOLD、X‑Token、持续预训练、同 tokenizer KD 等基线在 1B 规模的 Llama‑3.2‑1B 与 Gemma3‑1B‑PT 学生、跨 tokenizer 教师 Qwen3‑4B、Phi‑4‑mini 进行对比。CSD 在大多数设置下实现最高平均分，平均提升 1.38–5.38 分，显著优于 X‑Token、ALM、GOLD、ULD 及持续预训练；在 8B 学生上同样获得显著提升；Top‑k 截断也可将训练时间缩短至约 50% 左右，准确率仅下降 0.1–0.3 分；

**⚠️ 局限性**

局限性包括：①评估集中在英文任务，未知在多语言环境下的表现；②残差事件无法区分教师对未显式支持事件的细粒度偏好；③对 many‑to‑many chunk 仍采用观察到的单一对，未充分利用更丰富的支持；④对不同 tokenizer 家族的普适性和计算开销尚需进一步验证。

---

## 34. Progressive Permutation Alignment with Temperature-Controlled Spectral Sharpening

**arXiv ID:** 2610.29396 | [PDF](https://arxiv.org/pdf/2610.29396v1)

**作者:** Yuta Horioka `[一作]` (University of Osaka), Yuki Uranishi `[通讯]` (University of Osaka)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种逐层递进的权重对齐方法，结合匈牙利算法与温度控制的谱熨烫，实现对神经场（如NeRF）中隐层单元的精确置换对齐。

**💡 创新点**

创新点在于：①采用连续的相关矩阵与离散的匈牙利锚点相结合的温度谱熨烫机制，避免早期硬置换导致的错误传播；②通过单步温度下降的谱退火，彻底摆脱梯度优化的高昂成本；③在对齐后加入一轮逐层权重匹配精细化，进一步提升一致性。

**🔧 技术方法**

使用的技术包括：匈牙利算法求离散匹配锚点；奇异值分解（SVD）实现谱熨烫；温度控制的谱操作（σ→σ^τ）；逐层递进对齐与最终权重匹配。

**📊 数据集**

实验数据集涵盖：标准MLP（5层）及VGG11、ResNet18（ImageNet预训练）；NeRF Synthetic八个场景。

**📈 对比分析**

与Naive、Weight Matching、Progressive Weight Matching、OT4P、Progressive OT4P等方法对比；在MLP/CNN上实现100%置换恢复；在NeRF上实现0.0 RMSE、14.4 dB PSNR，显著抑制插值路径中的损失壁垒。

**⚠️ 局限性**

局限性在于仅在基于MLP的NeRF Synthetic上验证，未覆盖更复杂的INR结构或真实场景；未来需扩展至其他INR模型和实际数据。

---

## 35. Can AI Chat Replace Traditional Search? Evidence from a RAG-based Live Events Experiment

**arXiv ID:** 2610.28639 | [PDF](https://arxiv.org/pdf/2610.28639v1)

**作者:** Rakesh Allu `[一作]`, Evgeny Kagan `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一项对话式搜索与传统筛选搜索对比的在线实验中，研究者让用户在纽约市的数千场现场活动中寻找最吸引人的演出，并通过贝克-德格鲁德-马什卡（BDM）机制来量化选择的价值；

**💡 创新点**

创新之处在于首次将检索增强生成（RAG）技术应用于行为实验，保证不同界面下的搜索结果一致，并揭示了纯聊天界面对开放式探索的负面影响，而混合界面与传统筛选表现相当；

**🔧 技术方法**

使用了大型语言模型（GPT‑3.5、GPT‑4.1 等）配合 RAG 进行查询生成，检索与生成分离；

**📊 数据集**

数据集来源于 Ticketmaster 在纽约市的实时演出日历，包含约 7000 场演出，涵盖音乐、体育、艺术与戏剧等类别；

**📈 对比分析**

通过三组对照（Filter、Chat、Hybrid）与 BDM 估价、满意度、搜索时长等指标比较，结果显示 Hybrid 与 Filter 在价值与满意度上无显著差异，而 Chat 的价值与搜索时长均显著更低；

**⚠️ 局限性**

局限性包括仅在桌面环境下实验、单一行业与地区、没有多次使用者学习效应、缺乏个性化搜索功能，以及 RAG 架构对实际部署中信息不完整的鲁棒性未作深入探讨。

---

## 36. Learning Checklists to Reduce Lottery Effects in Scientific Peer Review

**arXiv ID:** 2610.28666 | [PDF](https://arxiv.org/pdf/2610.28666v1)

**作者:** Xuanyou Liu `[一作]` (Northwestern University), Jessica Hullman `[通讯]` (Northwestern University)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

利用LLM协助的流水线从历史评审数据中学习并生成检查表，以减少同行评审中的抽签方差。

**💡 创新点**

创新点在于将检查表设计视为对评审主张空间的分层粗化，并通过“预期相关一致度”(ECA) 这一代理指标来选择最能降低误差的粗化程度，从而在不增加过多偏差的前提下显著降低小委员会的均方误差。

**🔧 技术方法**

使用的技术包括：LLM（Grok 4.5、Qwen3-Embedding-0.6B）、聚类树（Ward），以及基于预期相关一致度的优化方法；评估则采用LLM模拟评审和小规模人类实验。

**📊 数据集**

数据集为 NeurIPS 2021 双委员会实验数据（298 论文）和 ICLR 2022 评审数据（300 论文）。

**📈 对比分析**

与传统 NeurIPS 2021 官方检查表以及无检查表情形比较，结果显示：在 NeurIPS 上三评审委员会的 MSE 降低 58%，ICLR 降低 21%；最佳 ECA 的检查表在三人及以上委员会中始终具有最低 MSE；在人类排序实验中，检查表将 MSE 从 0.48 降至 0.22。

**⚠️ 局限性**

局限性包括：评估主要基于 LLM 模拟评审，真实人类评审的覆盖率与误差可能不同；检查表可能引入系统性偏差；人类实验样本有限，难以覆盖更广泛的评审场景。

---

## 37. Measure First, Then Allocate: Calibrated Rollout Budgets for MCP Agents in Reinforcement Learning

**arXiv ID:** 2610.28785 | [PDF](https://arxiv.org/pdf/2610.28785v1)

**作者:** Hieu Tran `[一作]` (University of Massachusetts), Hong Yu `[通讯]` (University of Massachusetts)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

未知

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

## 38. Beyond Correct Answers: How Compositional Cues Shape Visual Judgments

**arXiv ID:** 2610.28843 | [PDF](https://arxiv.org/pdf/2610.28843v1)

**作者:** Xiaoyu Ma `[一作]` (University of Notre Dame), Yanfang Ye `[通讯]` (University of Notre Dame)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了结构化提示中角色标签和辅助提示对视觉绑定任务的影响，并通过实验评估模型对属性和关系错误的敏感性。

**💡 创新点**

提出按角色标签分解、提示益处与损害诊断以及基于诊断的提示改写和任务重构方法。

**🔧 技术方法**

采用结构化提示、局部与全局评分、误差类型对比、提示益处-损害分解、关系选择等技术。

**📊 数据集**

使用 GQA/Visual Genome 的属性图像、48对 Spatial 反面真值图像以及外部48图像等数据集。

**📈 对比分析**

通过十种模型和六项任务的精确度对比，配合置信区间与对照实验，揭示不同模型对标签的影响及提示误差，性能在模型间存在显著差异。

**⚠️ 局限性**

仅在英文数据集、样本量有限、模型特定读数、提示文本与任务变更未完全分离等方面受限。

---

## 39. KIRA: A Full-Stack Reading Screening System with Dual-Pathway Analysis for Children

**arXiv ID:** 2610.28840 | [PDF](https://arxiv.org/pdf/2610.28840v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 40. Design and Tuning of Multi-Agent Systems Using Bayesian Evolutionary Optimization

**arXiv ID:** 2610.29276 | [PDF](https://arxiv.org/pdf/2610.29276v1)

**作者:** Dominik Jehle `[一作]` (University of Freiburg), Tom Zehle `[通讯]` (University of Freiburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过结合进化算法与贝叶斯优化，自动发现任务特定的多智能体系统（MAS）拓扑结构，并评估其在代码生成和表格预处理任务中的性能。

**💡 创新点**

创新点在于用图感知高斯过程和期望改进（EI）作为候选选择的显式准则，替代传统的LLM判断，并引入基于图编辑距离（GED）的相似性核。

**🔧 技术方法**

使用了进化变异与交叉、图相似度测量（GED）、图感知高斯过程、期望改进、以及Gemma 4 31B LLM进行提示与角色分配。

**📊 数据集**

实验数据集包括BigCodeBench代码生成任务，BeyondArena 20个IID回归数据集和5个时序回归数据集，以及TabPFN预测模型。

**📈 对比分析**

与单Agent和手工Planner→Executor→Validator架构对比，BigCodeBench平均得分0.410（比手工+0.017、单Agent+0.043），IID回归R²为0.611，时序回归R²为0.544，均优于基线。

**⚠️ 局限性**

主要局限是拓扑优化与提示生成分离，GED相似性核不一定正半定，可能影响贝叶斯选择效果，需进一步联合优化提示并扩大任务评估范围。

---

## 41. A Model-Free Continuous-Time RL Algorithm from Discrete-Time Off-Policy Data with End-to-End Guarantees

**arXiv ID:** 2610.28915 | [PDF](https://arxiv.org/pdf/2610.28915v1)

**作者:** Chaorui Yao `[一作]` (University of California), Yuhua Zhu `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于 PhiBE-Q 的模型无关连续时间强化学习框架（PhiBE‑PI），该框架利用固定离线批次的离散转移数据直接学习单一的状态-动作评估器，并在同一批次上反复进行策略评估与硬贪婪改进，从而在连续时间控制问题上实现一次性数据重用。

**💡 创新点**

创新点主要有三点：
① 将 PhiBE‑Q 作为状态-动作评估器与 PhiBE‑PI 模型无关迭代相结合，首次实现仅用单批次数据完成连续时间策略迭代；
② 在线性函数逼近下给出完整的局部端到端子最优性保证，系统性量化离散化误差、函数逼近误差与有限样本误差的传播；
③ 明确阐释了采样间隔 Δt 与统计误差之间的偏差‑方差权衡，揭示更细的时间离散化并不一定提升策略性能。

**🔧 技术方法**

技术手段包括：
- PhiBE‑Q 近似（利用一次转移的矩矩）
- 线性 Galerkin 评估（求解残差正交方程）
- 硬贪婪策略改进（基于特征线性组合选择动作）
- 随机逼近与马尔科夫链理论用于统一样本误差分析
- 对连续时间随机微分方程的偏微分算子进行一次插值估计。

**📊 数据集**

实验数据集主要为合成控制问题：
- 一个光滑连续时间控制实例，提供解析最优解；
- 标准标量 LQR，采用从连续时间模型生成的离散转移；
- CartPole 轨迹，使用离散动作与离线数据。未使用公开工业或真实传感器数据。

**📈 对比分析**

与离散时间贝尔曼策略迭代（BE‑PI）以及传统离散 Q‑学习/DAU、Martingale‑q‑learning 等方法进行对比。实验表明：
- 在离散化误差占主导的 regime 下，PhiBE‑PI 的总误差显著低于 BE‑PI；
- 在 LQR 任务中，PhiBE‑PI 与 BE‑PI 的统计误差相当，但 PhiBE‑PI 的离散化误差更小；
- 在 CartPole 中，PhiBE‑PI 在训练与部署频率不一致时优于 BE‑PI，而当频率一致时 BE‑PI 更佳。

**⚠️ 局限性**

局限性：
- 仅在线性特征与硬贪婪更新下给出理论保证；对软策略或非线性逼近缺乏分析；
- 需要足够的状态覆盖与动作覆盖、光滑性假设，以及高阶矩矩可估计；
- 仅提供局部子最优性保证，无法保证全局收敛；
- 对批次重用导致的样本相关性处理有限，可能在极大迭代次数下失效。

---

## 42. Extending Affordance Segmentation for Wearable Robots to Multi-Object Images

**arXiv ID:** 2610.29474 | [PDF](https://arxiv.org/pdf/2610.29474v1)

**作者:** Simone Lugani `[一作]` (University of Genoa), Edoardo Ragusa `[通讯]` (University of Genoa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `67630363-6be0-4f51-ab05-7198250671a5` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在可穿戴机械臂控制中加入激光指针，实现多物体场景下的可操作性分割，并通过混合现实训练提升目标分辨率。

**💡 创新点**

用激光指针作为显式目标定位机制，并将指针标记融入训练数据，利用混合现实生成多物体背景，保持轻量级CNN，实现目标解混。

**🔧 技术方法**

轻量级深度卷积网络（~638 MFLOPs、3.6 M参数）进行分割；混合现实数据增强（合成指针与背景）；EMG控制+激光触发；加权交叉熵训练；几何与颜色数据增强。

**📊 数据集**

UMD单物体固定蓝背景数据集、IIT‑Aff多物体场景数据集，以及自制的混合现实合成数据。

**📈 对比分析**

与无指针模型对比，单物体测试中准确率从95.69%提升至96.18%；多物体场景下，带指针模型准确率≥90%，未带指针或随机指针准确率下降7–9%或更大。相较现有方法提升目标解混精度最多9%，计算成本保持不变。

**⚠️ 局限性**

仅在实验室或混合现实环境验证；激光可见性受光照、透明/镜面表面以及用户因素影响；需要进一步实测环境条件、可用性和认知负荷。

---

## 43. Value-Rectified Distillation for Flow-based Offline Reinforcement Learning

**arXiv ID:** 2610.29075 | [PDF](https://arxiv.org/pdf/2610.29075v1)

**作者:** Ke Jiang `[一作]` (Nanjing University of Aeronautics and Astronautics), Xiaoyang Tan `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在离线强化学习中解决基于流匹配的行为策略在多模态数据上蒸馏时出现的 OOD 模式平均问题，提出 Value-Rectified Distillation (VRD) 框架。

**💡 创新点**

将策略蒸馏转化为价值引导的轨迹对齐问题，设计了两种实现：统计滤波 VRD‑Q 与几何重塑 VRD‑R，解决了隐空间融合与 OOD 生成。

**🔧 技术方法**

基于 Flow Matching、Flow Q‑Learning 的流匹配网络，价值引导的对齐核、量化阈值加权、潜在空间变换等技术。

**📊 数据集**

在 D4RL（MuJoCo、AntMaze、Adroit）以及 OGBench、Minari 等连续控制和长时序规划基准上进行评测。

**📈 对比分析**

与 Gaussian、Diffusion、Flow 基线（BC、IQL、FQL、GFP、DF 等）对比，VRD‑R 在 D4RL 上平均 92.8 分、VRD‑Q 88.8 分；在 OGBench 上 VRD‑Q 56.4 分，显著优于其他方法，且在多模态噪声数据上鲁棒性更好。

**⚠️ 局限性**

依赖 Q‑网络的准确性，若 Q 值高估或偏差大可能导致对齐错误；目前仅针对连续动作空间，离散空间适用性待扩展。

---

## 44. norregel: a preregistered benchmark of dated numeric values in Norwegian law

**arXiv ID:** 2610.29207 | [PDF](https://arxiv.org/pdf/2610.29207v1)

**作者:** Eirik Botten Nicolaysen `[一作]` `[通讯]` (EcoDeco AS), Eirik Botten Nicolaysen (EcoDeco AS)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

评估语言模型在挪威法律条文中时间敏感数值的准确性，并检验其随训练截止时间的过时答案比例。

**💡 创新点**

引入norregel基准，结合预注册实验设计和可复现代码，系统性评估模型对法律数值的时效性。

**🔧 技术方法**

使用预注册、cluster‑bootstrap 置信区间、McNemar 检验、分层抽样、评分器v2，以及 API 与本地模型的对比等技术。

**📊 数据集**

采用 Lovdata 公开数据抽取的 916 条时间敏感法律条文（NOK 金额、百分比、G 倍数、年龄等），构成主要集合和 H3 配对。

**📈 对比分析**

通过正确、过时、错误、不可解释四类比例评估模型，Opus 5.5 与 Sonnet 4.6 在第一跑中分别约 46% 与 40% 正确；第二跑中 Opus 4.5、4.6 与 Sonnet 4.6 分别约 45%、44% 与 40% 正确；H1 在合并模型下显著提升过时比例，但 S8 敏感性分析未能验证。

**⚠️ 局限性**

存在大量不可解释答案、对训练截止时间有效性的不确定性、S8 效应不稳健、缺乏错误匹配概率估计、H1 在单个模型上无效以及 H3 结果受高不可解释率影响。

---

## 45. Where AI Vulnerability Discovery Harnesses Go Quiet: Structural False Negatives and Hidden Vulnerabilities

**arXiv ID:** 2610.29379 | [PDF](https://arxiv.org/pdf/2610.29379v1)

**作者:** Gadi Evron `[一作]` `[通讯]` (Knostic), Gadi Evron (Knostic)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了LLM驱动的安全扫描工具在工程实现中可能导致的无声漏报问题，并对13种不同的drop点进行了系统归纳与分类。

**💡 创新点**

创新点在于提出了完整的drop点taxonomy，展示了如何通过单文件变更隐藏漏洞并在多种工具中同时失效，以及给出了针对这些缺陷的设计原则和实用修复建议。

**🔧 技术方法**

主要技术手段包括：对12个开源扫描器源码的逐行审计、构造语义保持的代码变换、模型交互与结果验证、以及对报告生成过程的细粒度跟踪。

**📊 数据集**

实验使用了12个公开LLM harness（OpenAnt、raptor、ai-deep-sast、Codex Security等）的指定提交版本作为“数据集”，不依赖人工标注的漏洞集合。

**📈 对比分析**

本文未进行传统性能对比，而是通过案例演示（单文件多重drop、跨工具隐藏漏洞等）证明这些缺陷在实际使用中可导致真实漏洞被完全忽略，说明现有工具在可靠性方面存在显著缺陷。

**⚠️ 局限性**

限制在于仅分析了12个示例工具，未测量这些drop在真实仓库中的出现频率；缺少对不同LLM模型和prompt变体对drop影响的系统评估；且示例多基于静态扫描阶段，未涵盖所有动态或混合分析路径。

---

## 46. Scan or Index? large Evidence Coverage, Answer Quality, and Cost in Long-Document RAG

**arXiv ID:** 2610.28957 | [PDF](https://arxiv.org/pdf/2610.28957v1)

**作者:** Juan Terven `[一作]` (Instituto Politécnico Nacional), Gendry Alfonso-Francia `[通讯]` (Universidad Nacional Autónoma de México)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究对比了11种检索配置在科研文章和完整教科书中的效果，探讨证据保留、答案准确度与API成本的权衡。

**💡 创新点**

通过量化证据覆盖率、答案准确率与延迟/成本，揭示全扫描并不一定提升答案质量，强调在已知文档内检索时需要兼顾证据保留、答案质量与成本。

**🔧 技术方法**

使用Jev、OpenAI Decisions、Cloudflare Clef、BM25、Dense Passage Retrieval（OpenAI、BGE-M3）、Hybrid RRF及其重排序变体等检索与重排序技术。

**📊 数据集**

评估数据集为QASPER的396个科研文章问题和八本OpenStax教科书的80个问题，涵盖可回答与不可回答情况。

**📈 对比分析**

结果显示Hybrid+Jev在证据覆盖率最高（94.1%），Hybrid+Decisions在答案准确率最高（0.892），但全扫描耗时更长、成本更高，且提升的答案质量不显著。

**⚠️ 局限性**

仅适用于已知文档内检索、最多五段、2500词上下文，未覆盖多文档检索、实时模型变化以及本地计算成本等实际部署因素。

---

## 47. Asymmetry Tunneling Beyond Adaptive Equivariance Relaxation

**arXiv ID:** 2610.29473 | [PDF](https://arxiv.org/pdf/2610.29473v1)

**作者:** Eunho Jeong `[一作]` (Seoul National University), Hyeonjin Kim `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出在严格等变神经网络下通过受限残差路径研究“对称性穿透”现象，探讨残差在限制接口中是否能传递任务相关信息。

**💡 创新点**

通过在残差与等变主干之间引入stop‑gradient、标量门和单次融合的受限接口，首次系统性诊断残差在对称性破坏结构中的存活与否。

**🔧 技术方法**

使用等变网络（SEGNN、EqMotion、EGNO、VN‑DGCNN、GVP‑GNN），残差路径采用 Transformer‑style 交叉注意力、标量门、混合乘积等技术；诊断指标包括残差存活率 κ_r、局部 SVD 正交能量 Δρ⊥、Grad‑CAM Lift、直接元素归一化等。

**📊 数据集**

实验数据集包括 Charged N‑body（对称匹配控制）、ETH/UCY 行人轨迹、CMU Motion、ScanObjectNN 点云分类、PDBbind 分子亲和力。

**📈 对比分析**

与纯等变基线及自适应等变松弛方法（EGNO+ACE）对比，受限残差路径在多数任务上与或优于自适应松弛方案，获得显著误差/准确率提升；在对称匹配的 N‑body 中仅观察残差崩溃且无性能提升。

**⚠️ 局限性**

诊断结果只能表明残差是否存活，无法精准定位对称性破坏机制；残差路径的额外容量可能导致性能提升而非真正的对称性信号；若预设对称群与任务不匹配，残差存活可能是对不合适对称的补偿，而非真正的对称性破坏。

---

## 48. Deciding Scale Placement in MXFP4 Dot Products with Integer Arithmetic

**arXiv ID:** 2610.29445 | [PDF](https://arxiv.org/pdf/2610.29445v1)

**作者:** Shreyas Vinayakumar `[一作]` `[通讯]`, Shreyas Vinayakumar

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究 MXFP4 块缩放格式下整数算术点积的缩放放置重写合法性，构造整数状态机并将等价性查询精确转化为量化自由线性整数算术(QF_LIA)，并在 MLIR 中实现可证明的 guard。

**💡 创新点**

① 将浮点等价性问题精确化为线性整数模型；② 推导出基于块长度和归约树的闭式足够上界，并证明其最优性；③ 在整数域中实现高效等价性判定，显著提升求解速度；④ 提供可直接在编译器中使用的缩放放置重写 guard。

**🔧 技术方法**

SMT 求解、QF_LIA、整数状态机、IEEE 754 binary32 细化、E2M1/E8M0 编码、归约树分析、MLIR pass prototype、GPU 重放验证、离散取值枚举。

**📊 数据集**

对所有 16×253 个 E2M1 乘积值和 0–252 的指数范围进行全枚举；随机生成 140k+ 条块、65536 条 GPU 随机块；在 MXFP4 32/64 阶块和多种归约树上进行验证。

**📈 对比分析**

与直接浮点 SMT 编码对比：整数查询在 0.1–7 秒内完成，而浮点查询在 1 小时内就达到了求解上限；通过 GPU 运行验证结果一致；证明时间与块大小、树形结构呈线性关系，整体性能可接受。

**⚠️ 局限性**

仅适用于单独 binary32 操作、round‑to‑nearest、渐进下溢、固定归约树；不涵盖 FMA、flush‑to‑zero、不同宽度累加器、MXFP8、NVFP4 等格式；证明未机械化，MLIR 仅为原型实现。

---

## 49. Look Beyond the Question: Preserve-Focus-Fuse for Whole-Slide Image Question Answering

**arXiv ID:** 2610.29431 | [PDF](https://arxiv.org/pdf/2610.29431v1)

**作者:** Xiangyu Li `[一作]` (Tianjin University), Ran Su `[通讯]` (Tianjin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在本研究中，作者提出了一种 Preserve–Focus–Fuse 框架，用于在整片切片图像（WSI）上回答自然语言问题。该框架首先通过跨问题答案监督学习一个不受问题限制的 Morphology Bank（Preserve），随后利用 Focus 在所有前景补丁中检索与当前问题相关的证据，并通过 Morphology‑on‑Visual‑Evidence（MoV）融合将两者的表示合并成 96‑token 的视觉前缀，供 Qwen2.5‑7B‑Instruct 生成答案。

**💡 创新点**

创新点包括：① 利用跨问题答案监督学习一个通用 Morphology Bank，使模型能够在不受单一问题限制的情况下捕获多种组织形态；② 通过 Focus 在完整补丁集合中检索问题特定证据，避免传统压缩/采样导致的关键信息丢失；③ MoV 融合机制将 Bank 中的形态信息投射到检索到的证据位置，实现“保留-聚焦-融合”的闭环；④ 将整合后的 96‑token 视觉前缀直接注入大语言模型，从而在保持高效的同时提升回答质量。

**🔧 技术方法**

技术实现上，作者使用了：① 冻结的 CONCH 图像/文本编码器产生 512‑维特征；② 一个 512 维内容特征适配器；③ 6 个 Morphology 专家（4 语义 anchor，2 自由 expert）以及相应的路由器和残差 FFN；④ 两阶段训练：Stage I 通过跨问题监督学习 Preserve；Stage II 固定 Preserve 后训练 Focus 与 MoV，并对 Qwen 进行 LoRA 微调；⑤ MoV 采用跨注意力和门控融合；⑥ 采用 Qwen2.5‑7B‑Instruct 作为语言生成 backbone。

**📊 数据集**

主要使用的数据集包括：WSI‑Bench（9,850 片段，179,569 QA 对，30 TCGA 病种）用于训练与评估；SlideBench‑BCNB（无目标域适配）用于零样本迁移评估；此外，还在 WSI‑Bench 上进行报告生成评测。

**📈 对比分析**

与现有方法对比，作者在 WSI‑Bench VQA 上取得 Avg‑9 = 0.769，优于 WSI‑LLaVA（0.754）、HistoSelect（0.734）等基线，并在所有报告生成指标（BLEU‑1/2/3/4、ROUGE‑L、METEOR、WSI‑P、WSI‑R）上领跑。零样本 SlideBench‑BCNB 亦达到 0.639 的整体准确率，领先多数对手。通过对 Preserve、Focus、MoV 三个模块的消融实验，验证了每一部分的互补性。

**⚠️ 局限性**

局限性主要体现在：① 需要先预训练并冻结 CONCH 编码器和 Qwen 语言模型，缺乏端到端可调优；② 仍然依赖 96‑token 的视觉前缀，若需更细粒度的解释可能受限；③ 需要对每个请求重建 Morphology Bank，计算成本在大规模在线推理场景中可能成为瓶颈；④ 目前仅在公开 WSI 数据集上评估，尚未验证在不同组织或非 TCGA 采样条件下的鲁棒性。

---

## 50. Multi-Objective Protein Optimization via Modular Diffusion Steering

**arXiv ID:** 2610.28943 | [PDF](https://arxiv.org/pdf/2610.28943v1)

**作者:** Meir Roketlishvili `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Raul Astudillo `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出MODS框架，结合多目标加权和后验分布集成，在有限反馈下通过推理时间扩散模型生成蛋白序列

**💡 创新点**

通过随机化Chebyshev加权与后验预测集成实现多目标采样与不确定性探索的分离，可与任意预训练扩散模型和引导方法耦合

**🔧 技术方法**

增量后验模型、Chebyshev加权、扩散模型推理时间引导（CG/DAPS）、多目标超体积评估

**📊 数据集**

六个实验任务，涵盖五个二目标、一个三目标蛋白测序评估任务，使用实验测定数据训练的预测oracles

**📈 对比分析**

与ARLM+DPO、LaMBO‑2、SAGE‑Prot等迭代优化基线比较，MODS在多数任务上达到或超过最高超体积，显示良好性能

**⚠️ 局限性**

依赖或acles预测准确性，需实验验证；只测试了两种扩散模型和两种引导方法，未覆盖更广泛模型与任务

---

## 51. Zero-Cost Detection of Fabricated Citations: A Staged Heuristic Approach for Computing Dissertation Assessment

**arXiv ID:** 2610.28649 | [PDF](https://arxiv.org/pdf/2610.28649v1)

**作者:** Hazrat Ali `[一作]` (University of Stirling), Mohamed Elawady `[通讯]` (University of Strathclyde)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究提出了 Fabrication Severity Stack (FSS) 以及基于主流 LLM 的三阶段启发式验证流程，用于检测学术写作中的引用伪造。

**💡 创新点**

创新点在于将引用伪造分层分类（Level1-3）并提出无需专有工具即可利用通用 LLM 进行高召回率的检查。

**🔧 技术方法**

采用了结构化的 LLM 提示进行 DOI/URL 检索、元数据一致性比对和内容支持评估等技术。

**📊 数据集**

数据集为 100 条学生论文参考文献，其中 20 条人为注入 Level1/Level2 伪造。

**📈 对比分析**

通过 Claude、Sonnet5 与 Gemini Flash 3.5 的实验，Level1 伪造检测准确率达 90-100%，并能部分捕获 Level2。

**⚠️ 局限性**

限制在于样本规模有限、仅覆盖前两级伪造、对 Level3 内容误述仍需人工判定且依赖索引完整性。

---

## 52. TrainBridge: Agentic NPU Training Optimization Guided by GPU Compilation Evidence

**arXiv ID:** 2610.29171 | [PDF](https://arxiv.org/pdf/2610.29171v1)

**作者:** Yupeng Chen `[一作]` (Shenzhen Loop Area Institute), Ruoyu Sun `[通讯]` (Shenzhen Loop Area Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究如何利用GPU编译器的优化证据提升Ascend NPU的训练效率，提出两阶段的TrainBridge框架进行NPU程序重写和迭代调优。

**💡 创新点**

通过对GPU eager 与 compiled 版本差异提取优化意图，将GPU优化直接映射到NPU，并结合自适应编程代理实现自动重写；同时将GPU证据与NPU profiling结合，形成双阶段优化策略。

**🔧 技术方法**

代理式自动优化（基于Codex GPT-5.6-SOL）、GPU eager/compiled 对比分析、语义超节点映射、NPU编译器重写、迭代 profiling、CUDA/Ascend API 调用以及图计算重构等技术。

**📊 数据集**

Modded-NanoGPT 速度赛记录、CIFAR-10 AirBench、Qwen3-0.6B、ViT（TIMM）、nanoVLM等多种训练任务。

**📈 对比分析**

与基准GPU编译实现和纯NPU初始实现比较；在Modded-NanoGPT Record 50 12h调优下，TrainBridge平均速度提升 32%（从563s降到283s），平均 1.95× speedup；在27个记录平均 1.95×；对四个额外工作负载平均 1.7×提升。

**⚠️ 局限性**

仅适用于单机训练；需要源GPU实现以获得优化证据，限制适用范围；对多节点、跨机分布式训练尚未验证。

---

## 53. Black-Box Red-Teaming of Multi-Agent Systems via Reinforcement Learning

**arXiv ID:** 2610.28966 | [PDF](https://arxiv.org/pdf/2610.28966v1)

**作者:** Yuqi Jia `[一作]` (Duke University), Xun Chen `[通讯]` (ByteDance)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了针对多智能体系统的基于强化学习的黑盒红队攻击框架，用于提取隐藏系统提示。

**💡 创新点**

创新点包括：①将攻击过程拆分为重写攻击提示和生成攻击模板两步；②引入策略池与随机抽样机制以提升攻击提示多样性和有效性；③在攻击模板阶段实现一次性路由与提示注入，显著提高效率。

**🔧 技术方法**

使用的技术主要有：强化学习（DAPO），LLM（以ChatGLM为基础并微调LoRA），策略池与随机采样，文本相似度奖励（归一化编辑距离+长度惩罚），并结合自动摘要与句子嵌入进行策略检索。

**📊 数据集**

使用了合成与真实世界数据的混合数据集，包含10个系统提示与7类子智能体（编程、搜索、音乐、问答、翻译、视频、写作），共11,200条训练样本、4,800条模板训练样本以及每类100条测试样本。

**📈 对比分析**

与三种基线（PAIR、TAP、AutoDAN‑Turbo）比较，在7种子智能体上取得近乎完美的路由成功率（CAR>0.98）和高达80%~90%的提示提取成功率（ASR）。在不同后端LLM和多智能体结构下也表现出良好的跨模型与跨结构迁移性能。

**⚠️ 局限性**

局限性包括：依赖于已知子智能体的名称或功能信息；在极度保密或多级安全过滤的系统中可能效果下降；以及对模型内部细节的缺乏了解可能限制了对更复杂路由策略的攻击深度。

---

## 54. Warping the Workspace: Expanding Visual Access with Adjustable Reach Mapping for Humanoid Teleoperation

**arXiv ID:** 2610.28712 | [PDF](https://arxiv.org/pdf/2610.28712v1)

**作者:** Keigo Minamida `[一作]` (University of Tokyo), Jun Rekimoto `[通讯]` (Sony CSL)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了将全景视觉重映射与上肢转向解耦的远程人形操作方法，实验评估了全局缩放（Global Zoom）和焦点+上下文（Focus+Context）两种映射。

**💡 创新点**

创新点在于将视觉空间分配与动作映射耦合，使用逆向上肢转向映射保证显示方向与控制方向一致，从而实现更宽视野下的第一人称操作。

**🔧 技术方法**

使用360°相机（RICOH THETA X）、Apple Vision Pro HMD、机器人Unitree G1、逆向映射算法与全局缩放/焦点+上下文可视化技术。

**📊 数据集**

使用自制的袋子移动任务数据，8名参与者完成72次试验（前侧后各8次）作为评估数据集。

**📈 对比分析**

通过在8名受试者中进行对比实验，比较任务完成率和用户偏好，GZ完成率最高（23/24），但F+C被6/8受试者偏好，表明完成度与主观喜好可能不一致。

**⚠️ 局限性**

局限包括样本量小、仅测试单一任务场景、未评估长期使用与运动安全性，以及映射方案在不同机器人平台或复杂场景中的通用性不明。

---

## 55. CineEvolve: An Agentic Self-Evolution Framework for Cinematic Multi-Shot Video Generation

**arXiv ID:** 2610.29590 | [PDF](https://arxiv.org/pdf/2610.29590v1)

**作者:** Jingzhe Zhang `[一作]` (Beijing University of Posts and Telecommunications), Boxin Shi `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了CineEvolve框架，用统一的vision‑language policy同时担任导演和评审，实现多镜头视频生成的自我进化。

**💡 创新点**

将导演与评审整合为单一模型并交替强化学习，提出E‑GRPO避免优势膨胀，形成闭环自我改进。

**🔧 技术方法**

采用Qwen3‑VL‑8B‑Instruct作为核心模型，配合FLUX.2‑klein‑9B与MiniMax‑H3‑Turbo视觉后端，并使用结构、工具执行、评审质量与叙事连贯等多种奖励及E‑GRPO优化。

**📊 数据集**

构建CineTraj 40K条电影轨迹（30K用于监督热身，10K用于强化学习），并准备100条独立验证序列。

**📈 对比分析**

在CineTraj基准上与6种多镜头生成方法对比，CineEvolve在8项视觉与叙事一致性指标均居首，并在人工评估中获得最高偏好比例。

**⚠️ 局限性**

仅在Storyboard层面进行缺陷诊断与修正，细节运动瑕疵仍由后端生成器产生，且模型对极端复杂剪辑的自适应能力有限。

---

## 56. Omnilingual-MsASR: Benchmarking Multimodal, Multilingual, Long-Form Multi-Speaker ASR of Omni Large Language Models

**arXiv ID:** 2610.29455 | [PDF](https://arxiv.org/pdf/2610.29455v1)

**作者:** Yuan Ge `[一作]` (Northeastern University), Tong Xiao `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了Omnilingual-MsASR基准，用于评估多模态（音频+视频）、多语言、长时长、多说话人环境下的多说话人自动语音识别（MsASR）性能。

**💡 创新点**

创新点包括：①构建了包含六种语言、30分钟最长时长、最多20名说话人的大规模长视频数据集；②设计了并行音频与视频评估框架，探究视觉模态对说话人识别和时间定位的影响；③使用自动化伪标签生成结合人工专家审核的高质量标注流程，兼顾规模与准确性。

**🔧 技术方法**

技术手段主要包括：前沿语音识别模型（如Whisper）、视频面部识别与SyncNet用于跨片段说话人对齐、LLM（Gemini、Qwen、Muse Spark等）进行MsASR任务、Diarization专家模型（Moss Transcribe Diarize）做基准对比。

**📊 数据集**

数据集来源：公开的YouTube长视频（1.5–2小时），共约297条实例，覆盖西班牙语、法语、日语、韩语、俄语和泰语，每种语言约50条；所有实例均配备音频和对应视频轨道。

**📈 对比分析**

评价方法：采用Parse率、DER、cpWER、tcpWER四项指标；对7款商业LLM、1款开源LLM和专家Diarization模型进行系统对比。结果显示商业LLM在转录质量上表现优秀，但在时间定位和说话人识别方面存在显著瓶颈；视频模态对不同模型影响不一，部分模型从视觉信息获益显著，另一些则受噪声干扰。

**⚠️ 局限性**

限制：伪标签由自动模型生成，时间戳仅在明显错误时人工修正，可能导致对部分模型有偏向；商业API推理结果具有随机性，重现性受限；数据集中仅使用公开视频，未覆盖更复杂的场景或隐私敏感内容。

---

## 57. Any colour you like, as long as it's UIP: Semi-simplicial types from weakenings and thinnings

**arXiv ID:** 2610.29585 | [PDF](https://arxiv.org/pdf/2610.29585v1)

**作者:** Thorsten Altenkirch `[一作]` `[通讯]` (University of Nottingham), Thorsten Altenkirch (University of Nottingham)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在同伦类型论中构造半单纯型（semi-simplicial type, SST）并尝试在不使用UIP的语法与语义中实现其统一定义与解释

**💡 创新点**

提出了“Forded”技术（将构造中出现的计算索引改为存储等式），并通过编码-解码方法证明语法是集合，避免了对UIP的直接依赖；同时探讨了高维同伦递归（HIIT）与单步替代语法的可行性与局限性

**🔧 技术方法**

利用归纳-递归、归纳-递归-递归、高维递归类型（HIIT）、编码-解码法、正则化证明、归一化、以及对薄化、弱化与组合的自定义构造

**📊 数据集**

无；该工作基于理论构造与Agda实现，不涉及外部数据集

**📈 对比分析**

通过在Agda中实现并验证一系列形式化证明与归一化，展示了在无UIP的语法下实现SST的可行性，但在语义层面仍需UIP或额外的同伦一致性，未能在通用模型中完全实现

**⚠️ 局限性**

核心限制在于：无论将同伦一致性放入语法还是语义，仍需解决非可逆映射（thinning）的组合一致性；目前尚无在一般同伦类型论框架内完全消除UIP需求的方案

---

## 58. A Unified Evaluation Framework of Physical Adversarial Robustness in the Edge VLM Era

**arXiv ID:** 2610.29196 | [PDF](https://arxiv.org/pdf/2610.29196v1)

**作者:** Eric Liang `[一作]` (Saratoga High School), Kaiwen Yuan `[通讯]` (Mecka.ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文评估了物理攻击对边缘视觉‑语言模型（VLM）的鲁棒性，并与传统目标检测器进行对比；

**💡 创新点**

提出了统一的攻击评估框架、条件攻击成功率（cASR）及三类失败模式（遗漏、替代、位移），揭示失败模式主要由输出格式决定；

**🔧 技术方法**

采用白盒对抗补丁训练、EOT、抑制损失、可打印性和平滑项等技术；

**📊 数据集**

使用 COCO、PASCAL VOC、AdvT-Shirt‑1K 及 Waymo Open Dataset 的单人图像；

**📈 对比分析**

对 11 种模型基准，结果显示 VLM 与检测器在鲁棒性上相当，但失败模式受输出格式影响，分布偏移后 VLM 更易出现不可检测的失效；

**⚠️ 局限性**

仅在数字模拟环境下评估，未测试真实打印补丁或黑盒攻击；只研究抑制目标，未探讨其他攻击目标。

---

## 59. Behavior Beyond Text: Evaluating Refusal Transfer Failure in LLM Agents

**arXiv ID:** 2610.28982 | [PDF](https://arxiv.org/pdf/2610.28982v1)

**作者:** Ekagra Gupta `[一作]`, Adam Al Gharib `[通讯]` (Carleton University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建2304条匹配三元组基准，评估LLM在文本拒绝与工具调用安全性之间的转移差距。

**💡 创新点**

首次量化文本拒绝与工具调用安全性之间的条件发散，并揭示拒绝方向在工具环境中被抑制的机制。

**🔧 技术方法**

采用拒绝方向的线性分析、激活补丁、方向消融与激活添加等机制解释技术，结合AUC评估。

**📊 数据集**

使用四大高风险领域（金融、医疗、教育、法律）的多模态工具调用数据集，共2304个匹配三元组。

**📈 对比分析**

相较于文本拒绝率，工具调用的条件发散率高达5-66%，表明文本拒绝率无法预测工具调用安全性。

**⚠️ 局限性**

局限在于只评估单一步骤工具调用，且单一拒绝方向补丁仅能部分恢复安全，无法完全控制不安全调用。

---

## 60. Resonance and Depth: Composing Timbral Identity and Spatial Form in an Artist-in-the-Loop Co-Creative Multi-Agent Percussive Performance

**arXiv ID:** 2610.29026 | [PDF](https://arxiv.org/pdf/2610.29026v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 61. Symmetry-Oriented Testing Framework for Variational Quantum Algorithms

**arXiv ID:** 2610.29053 | [PDF](https://arxiv.org/pdf/2610.29053v1)

**作者:** Masaomi Yamaguchi `[一作]` (Fujitsu Limited), Fuyuki Ishikawa `[通讯]` (National Institute of Informatics)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于哈密顿量对称性的无oracle测试框架，利用与目标哈密顿量对易的对称算符对变分量子算法（VQA）执行过程与最终结果进行诊断；

**💡 创新点**

通过设计健康检查和接受检查两种诊断方法，使得无需已知精确解即可验证 VQA 运行的一致性与物理可行性；

**🔧 技术方法**

使用 Pauli 分解、期望值与方差估计、经典阴影方法以及 Qiskit 仿真器实现；

**📊 数据集**

在多种量子化学体系（H₂、LiH、BeH₂、H₂O）和 7 量子位 Z₂ 网格模型上进行实验；

**📈 对比分析**

与传统平台关系测试和已知解验证对比，框架在检测平台错误、错误哈密顿量、Ansatz 受限等场景均能发现异常；接受检查与能量误差呈显著正相关，可筛选低质量解；在噪声水平提升时仍保持诊断有效；

**⚠️ 局限性**

需先识别合适的对称算符，且仅提供必要条件，无法保证最终解完全正确；对称算符选择和测量开销在大规模系统中仍是挑战。

---

## 62. Cascading Bandits with Noisy Observations: Algorithms and Regret Bounds

**arXiv ID:** 2610.29381 | [PDF](https://arxiv.org/pdf/2610.29381v1)

**作者:** Yashaswini Kaluguri `[一作]` (Indian Institute of Technology Bombay), Nikhil Karamchandani `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种考虑观测噪声和成本的级联多臂赌博机模型，并对不同噪声模型下的最优短列表结构进行分类；

**💡 创新点**

创新点在于①将可分离的评分（Score）子类与一般无评分子类分离，②针对无评分子类设计了嵌入动态规划的DP‑UCB算法，③给出了Score子类的最优Θ((L‑K)log T)上界与下界；

**🔧 技术方法**

采用了基于UCB的在线学习框架、动态规划（DP‑Ordered/DP‑Subset）以及对观测误差的无偏估计与置信上界；

**📊 数据集**

在合成数据、MovieLens‑1M 电影评分数据以及真实的LLM 头条评估数据上进行实验；

**📈 对比分析**

与Perm‑UCB、Corr‑UCB以及基于简单评分的贪婪策略进行比较，实验表明DP‑UCB在Score子类达到最优对数量级累积风险，在无评分子类实现了显著低于线性、远优于基线的子线性风险；

**⚠️ 局限性**

限制在于DP‑UCB在正观测成本下的时间复杂度为O(2^L L K)，对大规模 arm 集合不可扩展，且无评分子类的下界尚未解析；

---

## 63. Fast Bayesian Optimization Through Linear Models

**arXiv ID:** 2610.29580 | [PDF](https://arxiv.org/pdf/2610.29580v1)

**作者:** Colin Doumont `[一作]` (University of Tübingen), Philipp Hennig `[通讯]` (University of Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于线性模型的快速贝叶斯优化方法，利用球面线性核与简化的超参数设置，实现与传统贝叶斯优化相当的性能但速度提升两到三百倍。

**💡 创新点**

创新点包括：①将原始线性BO的ARD尺度精简为单一尺度，显著减少超参数；②采用网格搜索+无梯度优化和Dinkelbach算法，将采集函数优化化为可解析求解的分式/线性问题；③统一框架适用于连续和组合输入，并展示EI可视为自调探索权重的UCB。

**🔧 技术方法**

技术手段主要有：球面投影映射、线性内积形式的核、无梯度的超参数搜索（网格+Nelder–Mead）、Dinkelbach分式程序、Minorize–Maximize (MM) 迭代、一次性重用特征的特征分解。

**📊 数据集**

实验数据集涵盖：高维连续函数（60–6392维，30个Sobol初始点，1000次评估），低维连续基准（12–33维，500次评估），组合优化问题（25–125维，200次评估），以及公开的Hyper‑parameter调优与机器学习模型优化基准。

**📈 对比分析**

与 Vanilla BO、Linear BO、One‑Hot BO、遗传/进化算法等方法比较，Fast BO 在相同评估预算下获得相同或更优的最终值，同时整体计算时间比传统 BO 低 10²–10³ 倍；在 0.1 秒/评估的快速评估场景中，Fast BO 的壁钟时间甚至优于其他“快”方法。

**⚠️ 局限性**

局限性包括：未针对维度 <10 的极低维空间、混合（连续+离散）搜索空间、批量（batch）采样策略以及极度昂贵评估（此时计算开销相对不重要）等情况进行评估，且在超大规模评估成本下速度优势不再显著。

---

## 64. FM-Occ: Feature-Diverse Upsampling and Memory-Aware Decoding for 3D Occupancy Prediction

**arXiv ID:** 2610.29388 | [PDF](https://arxiv.org/pdf/2610.29388v1)

**作者:** Yujia Zhai `[一作]` (Great Bay University), Bangquan Xie `[通讯]` (Great Bay University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

针对稀疏3D语义占据预测中上采样特征同质化和解码层语义衰减问题，提出了FM-Occ框架。

**💡 创新点**

创新点在于3D稀疏子像素洗牌（3S）保持上采样时特征多样性，以及熵调控层间记忆（ER-LM）与类别加权熵损失实现跨层语义保留。

**🔧 技术方法**

采用稀疏卷积、子像素洗牌、LSTM记忆、熵正则化、类别加权熵损失等技术。

**📊 数据集**

在SemanticKITTI数据集上进行训练与评估。

**📈 对比分析**

相较于State‑of‑the‑Art方法，FM-Occ在IoU/ mIoU 上提升至36.18% /12.73%，并且在BEV框架 DISC 中通过ER-LM 进一步提高到17.17%。

**⚠️ 局限性**

局限在于对稀疏卷积上采样的改进仍依赖手工设计，且在复杂场景下仍存在细粒度类别预测不足。

---

## 65. Reorienting the Vector Axis: Restructuring FlashAttention for Long-Vector Processors

**arXiv ID:** 2610.29489 | [PDF](https://arxiv.org/pdf/2610.29489v1)

**作者:** Emmanuel Irabor `[一作]` (Barcelona Supercomputing Center), Marc Casas `[通讯]` (Barcelona Supercomputing Center)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过重新排列 FlashAttention 的计算顺序，将得分矩阵和累积输出矩阵转置，以适配长向量处理器的寄存器宽度，提升了向量化效率。

**💡 创新点**

创新点在于将 FlashAttention 的递归方向重新定义为查询维度，使得每个区域的向量长度由查询块大小决定，并将指数多项式系数移至标量寄存器以消除寄存器溢出。

**🔧 技术方法**

主要技术包括矩阵转置、向量长度重定向、软指数多项式优化、以及对 RISC‑V 0.7.1 长向量单元的实测优化。

**📊 数据集**

实验使用合成的注意力序列（长度 N=1024/2048/4096，头维度 d=64）在 EPAC 芯片和对应 FPGA 上评估。

**📈 对比分析**

与原始 FlashAttention、标量系数以及融合软最大化的实现相比，重构后的核在 EPAC 芯片上可实现最高 2.08× 的加速，浮点利用率提升至 29.7% 之上。

**⚠️ 局限性**

局限性包括仅适用于查询块长度大于头维度的预填充/训练场景，且需要长向量寄存器；在单行查询块的自回归解码或短向量处理器上效果有限。

---

## 66. AI-Based End-to-End Automotive Design and Software Development for SDVs

**arXiv ID:** 2610.29160 | [PDF](https://arxiv.org/pdf/2610.29160v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 67. HumanCentric-Scenes-AI: A Human-Curated Multimodal Dataset of AI-Generated Human-Centric Scenes Across Surveillance, Occupational Portraits, AI-Edited Images, and Short-Form Video

**arXiv ID:** 2610.29045 | [PDF](https://arxiv.org/pdf/2610.29045v1)

**作者:** Nima Kamali `[一作]` (University of Applied Science and Technology), Negar Kamali `[通讯]` (Northwestern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建并公开了一个包含296张人工智能生成的多模态素材（CCTV风格图像、性别平衡的职业肖像、局部AI编辑图像、短视频）的人本中心场景数据集，并对其真实性、性别偏差与检测难度等特性进行了系统分析。

**💡 创新点**

创新点在于：①采用人工筛选提升生成素材的照片真实度；②在职业子集实现性别平衡配对，便于对性别刻板印象进行量化；③探究提示词对性别表现的影响；④将局部编辑区域大小与零样本检测准确度关联，验证编辑规模对识别的影响。

**🔧 技术方法**

主要技术包括：Midjourney（生成图像/视频）、Veo、Sora、Kling（生成短视频）、图像锐度（Laplacian方差）和SSIM评估、GPT‑4.1、Claude Opus、Gemini Pro等LLM的零样本分类，以及统计检验（二项检验、符号检验、Spearman相关）。

**📊 数据集**

使用的数据集为公开的HumanCentric‑Scenes‑AI本身（296张生成素材），并在实验中引用了134张未指定性别的初始生成图像、CCTV Human Pose Estimation Dataset（真实监控帧）以及对应的真实视频参考。

**📈 对比分析**

比较方法：将AI生成的CCTV图像与真实监控帧在分辨率、文件大小与锐度上进行对比；在职业子集中统计未指定性别提示下的性别比例并与美国劳工统计的占比做匹配；在局部编辑子集中按编辑像素百分比与零样本检测准确度绘制相关性图；在视频子集中将AI生成视频的时长、帧率、分辨率与真实参考视频进行对比。实验结果显示：①生成的CCTV图像锐度更高；②性别偏差与真实职业性别比例高度相关；③编辑区域越大检测准确率越高；③AI生成视频的时长与真实视频相近。

**⚠️ 局限性**

局限性：评估多在25张样本的子集上完成，可能不代表全部数据；生成模型更新后，难度与偏差可能变化；数据集仅涵盖部分生成器（Midjourney、Veo、Sora、Kling），缺乏对其他模型的覆盖；公开素材不能用于训练生成模型。

---

## 68. GRAM: Group-wise Rank-Aware Model Merging via Subspace Alignment

**arXiv ID:** 2610.29110 | [PDF](https://arxiv.org/pdf/2610.29110v1)

**作者:** Di Wu `[一作]` (JPMorgan Chase & Co), Shuaidong Pan `[通讯]` (JPMorgan Chase & Co)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对多任务部署场景，提出一种仅使用已微调权重的模型合并方法——Group-wise Rank-Aware Model Merging (GRAM)，通过几何聚类和自适应秩分配实现任务分组后合并。

**💡 创新点**

创新点在于：①用子空间几何（弦距）衡量任务间兼容性，②在Grassmannian上做k‑means++聚类形成兼容任务团体，③在每个团体内使用自适应秩分配的SVD合并，理论上可把子空间干扰降到最小。

**🔧 技术方法**

核心技术包括：任务向量分解（SVD）、子空间距离度量（弦距）、Grassmannian k‑means++聚类、秩分配优化（贪心挑选最大奇异值）、基于SVD的模型合并。

**📊 数据集**

实验数据集包括：GLUE（Flan‑T5‑Base/Large）、CLIP 8 任务视觉基准（ViT‑B/32 与 ViT‑L/14）、医学指令调优基准（Qwen2‑7B‑Instruct 4 任务 LoRA 专家）。

**📈 对比分析**

与 Weight Averaging、Task Arithmetic、TIES‑Merging、AdaMerging++、Iso‑C/CTS、TSV‑Merge 等数据无关基线比较，GRAM 在 GLUE 取得 83.8%/88.7% 的平均分，视觉基准 88.7%/94.0%，医学基准 Avg Rank 2.40，均显著优于所有对手，并且仅多部署一个模型即可实现这些提升。

**⚠️ 局限性**

局限性包括：实验仅在单一随机种子下报告；仅评估了少数模型族（Flan‑T5、CLIP、Qwen2）和小规模任务集；理论证明针对理想化对象，未涵盖 GRAM 的完整实现细节；未在真实多机构联邦环境下验证。

---

## 69. NavBind: Diagnosing Instruction-Action Binding at Route Forks in Navigation VLAs

**arXiv ID:** 2610.29203 | [PDF](https://arxiv.org/pdf/2610.29203v1)

**作者:** Josh Qixuan Sun `[一作]` (University of Waterloo), Mark Crowley `[通讯]` (University of Waterloo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一套基于路口分叉的对照诊断集，评估视觉‑语言‑动作模型在固定导航上下文下对指令的依赖性；

**💡 创新点**

创新点在于：① 通过跨路段重组指令生成可执行的对照指令对，保证两条路径共享历史但指令分叉；② 将生成动作、固定目标概率、视觉注意力与梯度归因相结合，系统衡量“指令-动作绑定”；

**🔧 技术方法**

采用指令-动作绑定指标（d̅、KL、Both+）、相同动作序列率（Same seq.）、注意力/梯度IoU；

**📊 数据集**

使用 R2R‑CE 与 RxR‑CE 的细粒度对齐注释，构造 3,252 个路口分叉上下文；

**📈 对比分析**

对 Uni‑NaVid、StreamVLN、JanusVLN、FutureNav 四种模型进行比较，发现尽管 Uni‑NaVid 在标准成功率上最高，StreamVLN 在绑定指标（d̅、KL、Both+）上表现最好；

**⚠️ 局限性**

局限性包括：指令重组可能产生语义歧义或不自然切分；仅评估前 4 步动作，未覆盖全局错误恢复；不同模型的动作步长差异导致物理距离不一致；注意力/梯度 IoU 仅衡量空间差异，无法证明其因果影响。

---

## 70. Correlated Quantal Response Equilibrium: A New Equilibrium Notion for Human-AI Strategic Interactions

**arXiv ID:** 2610.29020 | [PDF](https://arxiv.org/pdf/2610.29020v1)

**作者:** Zehao Zhao `[一作]` (University of Southern California), Vijay Subramanian `[通讯]` (University of Michigan)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了 Correlated Quantal Response Equilibrium (CQRE)，通过在协同均衡中引入可调的“懈怠”参数，生成唯一的最大熵联合行动分布，适用于人类、AI 以及混合人机对弈；

**💡 创新点**

创新点在于将传统协同均衡的严格服从约束放宽为可调的行动特定松弛，既保留协同特性，又兼顾人类/LLM 的有限理性，并通过最大熵选择消除多重解歧义；

**🔧 技术方法**

使用最大熵优化、KKT 条件、二阶最速下降与 L‑BFGS‑B 等数值方法估计懈怠参数，并利用矩阵形式的松弛约束与熵最大化求解 CQRE；

**📊 数据集**

利用公开的 Selten–Chmura 人类实验数据以及自行收集的 201,600 条来自六款 OpenAI LLM（GPT‑5.x 系列）的单局决策，构建 588 个目标分布（AI‑AI、AI‑人类、人与人类）和多种大规模动作空间/多玩家游戏；

**📈 对比分析**

通过对比 CQRE 与 Logit‑QRE、最大熵协同均衡、混合 Nash 等基准，使用 KL 散度和配对 95% 置信区间评估拟合度；在 12 个经典人类游戏、所有 12 个 Selten–Chmura 扩展、7 个大型动作空间游戏以及 56 个三玩家模型三元组中，CQRE 在绝大多数比较中显著优于 Logit‑QRE，平均 KL 下降 21–57%，表明其对联合分布建模更精准；

**⚠️ 局限性**

局限在于对极度集中（高峰值）行为的最大熵选择敏感，导致在 GPT‑5 近自我对弈等高集中场景中 Logit‑QRE 更优；此外，懈怠参数的解释与可迁移性仍待进一步理论与实证验证，且目前仅在有限的实验设置中验证。

---

## 71. Amplified Pruning and Path Selectors for Deterministic Radio Gossiping

**arXiv ID:** 2610.28930 | [PDF](https://arxiv.org/pdf/2610.28930v1)

**作者:** Qin Xin `[一作]` `[通讯]` (University of Faroe Islands), Qin Xin (University of Faroe Islands)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在未知的强连通有向无线电网络中提出了一种确定性Gossiping协议，适用于多项式范围内的节点标识符。

**💡 创新点**

创新点在于将放大接收证书、适应性裁剪和排列选择器相结合，得到的两步二次收缩和生存源完成机制，大幅降低了通信轮数。

**🔧 技术方法**

技术上采用了选择性时间表、路径选择器、伪随机化的强选择器、增量式逼近与完整回放等方法。

**📊 数据集**

实验以理论模型为主，未使用具体数据集，仅在多种图结构（稀疏、正规、密集）上进行分析。

**📈 对比分析**

相较于以往的 n^{4/3} 与 (mn)^{3/5} 上界，本协议在多项式标识符范围内可达 O(n^{1+ε})（ε<1/3）甚至 O(n√Δ)；在稀疏且满足度数浓度条件的图上可实现近线性 O(m) 的广播。

**⚠️ 局限性**

局限性包括：对于一般图和任意密度仍无法给出多项式标识符下的最优界；仍需结构假设或度数阈值；且实现上需要全局计数和复杂的调度。

---

## 72. From Prompts to Posteriors: Can LLM Agents Perform Bayesian Inference in Context?

**arXiv ID:** 2610.29331 | [PDF](https://arxiv.org/pdf/2610.29331v1)

**作者:** Linus Filbry `[一作]` (University of Oxford), Yarin Gal `[通讯]` (University of Oxford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建评估框架，系统测量大型语言模型（LLM）在上下文中进行贝叶斯推断的能力，包括概率一致性、对照贝叶斯后验的拟合度以及在自适应实验设计中的下游任务表现。

**💡 创新点**

创新点在于：①提出三轴评估（概率一致性、后验保真度、下游表现）并将其统一到科学发现工作流；②利用自定义的动物与蛋白质分类模拟环境，让LLM在开放式假设空间中完成贝叶斯更新；③证明现代LLM若启用推理和代码执行，可近似甚至精确恢复外部贝叶斯后验，削弱了传统束缚性更新框架的必要性。

**🔧 技术方法**

主要技术包括：对话式概率抽取、前向KL散度度量、基于实验设计的期望信息增益（EIG）评估、LLM原生推理模式、Python代码执行工具以及Monte Carlo估计。

**📊 数据集**

使用数据集包括：ForecastBench（用于概率一致性测试）以及两类自生成模拟环境——开放式动物问答和基于蛋白质属性的离散实验集合（有闭式和非闭式两种难度），所有实验均从零开始生成。

**📈 对比分析**

与贝叶斯oracle轨迹对比：在未启用推理时，LLM后验与oracle相差显著，导致下游实验选择错误，成功率下降30%以上；启用推理后，KL散度显著下降，几乎匹配oracle；进一步加入代码执行，能够在非闭式情形下逼近oracle后验，性能差距几乎归零。

**⚠️ 局限性**

局限性包括：在完全离散的开放式假设空间中仍需动态子集筛选导致先验偏置；对高度复杂或连续型后验仍存在数值不稳定；推理与工具使用依赖模型内部机制，可能不适用于所有LLM；实验环境与真实科学发现的多样性和噪声仍有限。

---

## 73. Who Approved This? Coding Agents, Human Accounts, and Merge-Gate Evidence at a Fintech Startup

**arXiv ID:** 2610.28683 | [PDF](https://arxiv.org/pdf/2610.28683v1)

**作者:** Adam Manji `[一作]` `[通讯]` (Bashi AI), Adam Manji (Bashi AI)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在Bashi AI的代码仓库中，对编码代理通过人类账户提交合并和审批的行为进行追踪和分析，重新计算并验证分支保护门控是否被满足。

**💡 创新点**

首次系统性地将代理会话日志与GitHub记录结合，揭示了代理操作与人类决策之间的分离问题，并提出了针对代理身份和审批记录的改进清单。

**🔧 技术方法**

结合GitHub REST API、Claude Code会话转录、脚本化的审批和合并工具，以及自定义门控重算算法，实现对合并事件的全面审计。

**📊 数据集**

使用了从2026年7月21日至10月16日的Bashi AI产品仓库共786个合并请求的仓库记录、相应的Agent会话转录和团队内部的工程师调查问卷。

**📈 对比分析**

通过算法重新计算核心门控属性，并与仓库原始记录对比，发现所有受保护分支的合并满足门控，但在71个合并中仅代理提交审批，说明系统能有效定位异常；整体审计耗时仅数小时。

**⚠️ 局限性**

样本仅来自单一初创公司、缺乏完整的会话转录与审计日志，无法完整重建人类决策轨迹，且分析聚焦于GitHub特定保护规则，结果不一定适用于其他平台。

---

## 74. FlashNext-RL: A System Optimization Study for Long-Horizon Agentic RL of Qwen3.8-Flash-Next

**arXiv ID:** 2610.28998 | [PDF](https://arxiv.org/pdf/2610.28998v1)

**作者:** Zhewei Yao `[一作]` (Snowflake AI Research), Yuxiong He `[通讯]` (Snowflake AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对长时 horizon 的 agentic RL，本文提出了一个面向模型的训练优化框架，在 125B Qwen 48 层模型上实现 BF16 单节点训练，并使用 vLLM 生成长序列 rollouts。

**💡 创新点**

创新点在于将内存规划、梯度累积、专家并行、稀疏注意力、门控残差等关键算子与模型结构深度耦合，突破了通用 RL 基础设施在大模型长序列训练中的瓶颈。

**🔧 技术方法**

采用的技术包括 FSDP2 + CPU offload、PinnedArena、GPU‑resident expert gradients、MoK、CuTe、GQA‑native Triton、UVA gather、稀疏分组查询注意力、门控残差融合核等。

**📊 数据集**

使用的数据集为软件工程（SWE）任务的 rollouts，来自 SWE‑Gym、R2E‑Gym 等环境，采用 131,072 token 上限的长序列样本。

**📈 对比分析**

通过逐步开启单项优化、8‑层阶梯实验和全模型对比，最终实现单节点 BF16 训练吞吐 4,438 REAL tokens/s/GPU（相较初始 395 tokens/s/GPU 提升 11×），单次迭代平均 22.4 分钟，生成速率 17.1K tokens/s。

**⚠️ 局限性**

局限性包括仅在单个 8×B200 节点上验证、未进行长期收敛或任务性能评估、对其他模型或硬件的可迁移性未探究。

---

## 75. Data-Driven Glyph Positioning for Handwritten Music Synthesis

**arXiv ID:** 2610.28899 | [PDF](https://arxiv.org/pdf/2610.28899v1)

**作者:** Elona Shatri `[一作]` (Charles University), Jan Hajič `[通讯]` (Charles University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了手写音乐合成中的字形定位问题，通过数据驱动方法改进 Smashcima 的定位规则。

**💡 创新点**

创新在于先用两大数据集量化定位分布，再用图神经网络学习上下文相关的定位偏移，实现显著 RMSE 降低。

**🔧 技术方法**

采用统计分析、GraphSAGE 结构的 GNN 进行偏移预测，配合 YOLOv8 目标检测。

**📊 数据集**

使用 MUSCIMA++ 与 OmniOMR 两个标注良好的手写音乐数据集。

**📈 对比分析**

与原始 Smashcima 基准、全局均值以及 GNN 模型在离散写手交叉验证和迁移学习上比较，GNN 在 RMSE 上降低 50–85%，检测 mAP50 提升 30% 以上。

**⚠️ 局限性**

局限包括仅使用单向定位，未进行完整图推理，且对小符号检测仍受分辨率限制。

---

## 76. VeriCalib: Verifiable Target-Based LiDAR-Camera Calibration

**arXiv ID:** 2610.28791 | [PDF](https://arxiv.org/pdf/2610.28791v1)

**作者:** Xiangcheng Hu `[一作]` (Hong Kong University of Science and Technology), Jin Wu `[通讯]` (University of Science and Technology Beijing)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为 VeriCalib 的可验证目标基 LiDAR‑相机外参校准方法，利用结构化协方差、两级外参补偿和一致性检验，能够在不需要外部真值的情况下给出校准状态与纠正建议。

**💡 创新点**

核心创新包括：①结构化协方差模型捕捉摄像机姿态共享和 LiDAR 边界噪声的协方差，消除传统独立特征误差导致的低效加权；②两级外参补偿揭示残差层和参数层中旋转与平移互补导致的弱约束，提供方向性精度与最小可检测偏差；③基于 F 分布的无偏尺度估计与批量内部/外部一致性检验，使校准结果可判定为“一致/精度/验证限制/已验证”。

**🔧 技术方法**

使用基于观测因子（残差、协方差、雅可比矩阵）的加权最小二乘优化、Schur 约简求解、方差分量估计、F 分布阈值、内部/外部一致性检验、方向性最小可检测偏差计算以及批量一致性检验等技术。

**📊 数据集**

在仿真数据（已知外参、已注入噪声与错误）以及多平台真实数据集上评估，平台包括：移动机械臂（Head 与 Chassis 观测）、手持装置（双摄 + Ouster LiDAR）、低成本闪光 LiDAR（FS-B、FS-C），使用四孔圆孔板和棋盘格等标准标定板。

**📈 对比分析**

与 FAST‑Calib、velo2cam、Dom2019 等现有校准器以及 LOO、RANSAC、Huber、随机/远端/最优设计等基线比较；实验显示 VeriCalib 在相同观测量下能够实现 8 mm/0.15° 的精度目标，所需观测量比 D/A/E 最优设计减少约 10–30%，且在检测错误时误报率低、检验力高；重复性和方差预测也优于无权重或独立协方差方法。

**⚠️ 局限性**

局限性包括：依赖已知几何标定板且不适用于无目标或非平面标定；需要手工或自动提取特征，若提取错误仍需额外鲁棒机制；对噪声模型的假设（如深度相关、相机角度噪声）需先验校准；批量一致性检验对观测分布敏感，可能在极端污染下误判；实现复杂度相对传统方法稍高。

---

## 77. Phase Transitions in Learning Driven by Extremal Data

**arXiv ID:** 2610.29531 | [PDF](https://arxiv.org/pdf/2610.29531v1)

**作者:** Ethan Levien `[一作]` (Dartmouth), Trevor GrandPre `[通讯]` (Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究并解析了多实例学习（MIL）中极值回归问题的梯度下降动力学，建立了包含极值选择、软化最大池化以及可学习温度参数的极简模型，并推导了其极限梯度流和相变行为。

**💡 创新点**

创新点在于将极值回归与随机能量模型（REM）和现代Hopfield网络的检索相变联系起来，首次给出梯度下降过程中的两种关键相变（冻结相与检索相）以及对应的低维动力学方程，并证明了这些相变能解释训练停滞与收敛速度的跳跃。

**🔧 技术方法**

使用的技术包括极值统计、极限极大值理论、随机能量模型映射、梯度流（ODE）推导、相变分析和数值模拟验证。

**📊 数据集**

数据集主要是模拟的高维高样本量的高斯袋子（n 及 d→∞ 的极限），并在此基础上进行有限尺寸的数值实验；未使用真实 MIL 数据集。

**📈 对比分析**

通过与有限尺寸实验的梯度轨迹、损失曲线以及梯度场进行对比，验证了理论预测的相图与动力学；实验表明在可辨识区（α<ρ_true^2/2）内，梯度下降在检索相内能成功收敛，而在非检索相内会停滞或收敛慢，理论与实验结果高度一致。

**⚠️ 局限性**

主要局限在于模型过于理想化（仅考虑高斯实例、单一极值选择、软化最大池化），未考虑真实数据的噪声、结构多样性和模型超参数的实际调优；此外，梯度流近似在相变边界附近失效，导致对有限尺寸梯度的预测误差随 d 下降而增大。

---

## 78. MERP: A Latent-Bottleneck Framework for Robust Multi-Modal Multi-Label Emotion Recognition

**arXiv ID:** 2610.29430 | [PDF](https://arxiv.org/pdf/2610.29430v1)

**作者:** Chuangxin Zhao `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Biörn W. Schuller `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在多模态多标签情感识别任务中，提出了一种基于固定潜在瓶颈的MERP模型，能够在输入模态缺失时保持鲁棒性。

**💡 创新点**

创新点在于将多模态信息压缩到固定大小的潜在空间，结合跨层标签读取、正负原型引导的噪声门控增强以及置信度感知的类别级融合，以实现对每个情绪类别证据的保留与强化。

**🔧 技术方法**

采用Perceiver‑IO架构的潜在编码、跨层自注意力、标签特定读取、正负原型聚合、频谱噪声门控增强（SNGA）以及自适应置信度融合等技术。

**📊 数据集**

使用公开的CMU‑MOSEI（英文）和M3ED（中文）两大情感语料库进行实验。

**📈 对比分析**

在20%随机模态丢失的评测中，MERP在CMU‑MOSEI和M3ED上分别取得0.563和0.780的micro‑F1，超过所有对照模型（如CARAT、LDDU等），并在更高模态缺失率下保持性能。

**⚠️ 局限性**

局限性包括：仅在完整输入上进行训练，缺少针对缺失模态的显式模拟；对极端缺失（如文本+视觉全缺）仍有一定性能下降；以及对更大标签空间或长序列的可扩展性尚未验证。

---

## 79. Seeing the Forest for the Trees: Emergent Agreement from Individual Conversations

**arXiv ID:** 2610.29328 | [PDF](https://arxiv.org/pdf/2610.29328v1)

**作者:** Adi Simhi `[一作]` (Technion Israel Institute of Technology), Shay B. Cohen `[通讯]` (University of Edinburgh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究LLM在多轮主观对话中的一致性演化，聚合内部激活与外部行为，揭示马尔可夫结构。

**💡 创新点**

创新点是发现内部一致性与外部行为在聚合后通过幂律关系耦合，并且内部一致性滞后于行为一致性。

**🔧 技术方法**

使用线性判别（LDA）提取内部一致性方向，一阶Markov模型预测聚合行为，幂律映射连接内部与外部一致性，并用AUROC评估内部状态滞后。

**📊 数据集**

五个公开对话数据集：Behavioral Dispositions、ManagerBench、Wealth‑seeking、Self‑coordination、Global。

**📈 对比分析**

与一阶Markov、二阶Markov和无偏基线比较，MAE约7%（对比12%），NR²大多>0.5，Δℓ显著高于随机基线，说明方法效果良好。

**⚠️ 局限性**

局限在于单会话内部一致性变异大，聚合后结构显现但对单会话解释不足；实验仅覆盖公开模型与特定主题，未验证跨模型、跨语言的普适性。

---

## 80. MINER: Modeling Inter- and Intra-Person Variance for Reidentification

**arXiv ID:** 2610.28941 | [PDF](https://arxiv.org/pdf/2610.28941v1)

**作者:** Oron Nir `[一作]` (Reichman University), Ariel Shamir `[通讯]` (Reichman University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 MINER 框架，对衣物变化的人体重识别任务进行软化硬正负样本挖掘和多正对比学习，并通过在线方差控制器实现批量中人内与人间方差的自适应平衡。

**💡 创新点**

创新点在于同时利用软化的硬正负样本挖掘、block‑diagonal 多正对比损失以及梯度方差估计的闭式调度器，实现数据集自适应的 m/n 组合，显著提升衣物不敏感检索效果。

**🔧 技术方法**

采用软硬正负挖掘、block‑diagonal 多正对比损失、梯度方差估计闭式调度器，以及基于 CLIP/DIFFER 等预训练模型的 finetune。

**📊 数据集**

实验使用 PRCC、LTCC、Celeb-L、DeepChange、VC‑Clothes、LaST 等 CC‑ReID 数据集，以及 RGB‑Depth TVRID 2026 竞赛数据。

**📈 对比分析**

与 17+ 现有方法对比，MINER 在 4 个 CC‑ReID 数据集均取得 SOTA（如 PRCC mAP 69.7%，Celeb‑L 76.9%，DeepChange 59.1%），在 RGB‑Depth 3 个赛道均位列前三名。

**⚠️ 局限性**

局限性：依赖预训练权重和衣物 ID 以及每人至少两套衣服的标签；从头训练不稳定；对衣物多样性不足的数据集效果受限。

---

## 81. SAT Sampling at Scale via GPU-based Boolean Constraint Propagation

**arXiv ID:** 2610.28805 | [PDF](https://arxiv.org/pdf/2610.28805v1)

**作者:** Alexander Nadel `[一作]` (Technion), Ofer Strichman `[通讯]` (Technion)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于 GPU 的 Boolean 约束传播（BCP）采样器，能够并行执行成千上万个独立的决策-传播轨迹，从而高效地生成 SAT 公式的随机解。

**💡 创新点**

创新点在于：① 设计了一个 32 位无锁计数器+键（counter+key）数据结构，使得在 GPU 上一次原子操作即可更新子句状态；② 通过 GPU 的大规模并行性同时推进多条采样轨迹，冲突轨迹直接丢弃；③ 将这一通用 BCP 引擎包装为可复用库并用于采样、约束挖掘、简化等任务；④ 通过多臂赌博机在线选择采样配置。

**🔧 技术方法**

使用技术包括：CUDA GPU 并行编程、原子操作、无锁 BCP、批量传播（Batch Propagation）、批量决策轨迹、批量约束挖掘与简化、投影（Projection）、多臂赌博机调参、统计均匀性检验。

**📊 数据集**

实验数据集为硬件验证电路：ISCAS‑85、41 个组合/顺序电路（包括常规和稀有目标），以及 41 个随机/稀有目标的 CNF 公式。

**📈 对比分析**

与 CPU‑基 CDCL 采样器和 GPU 梯度采样器做对比；通过样本/秒（steady‑state）和端到端时间两种指标评估。结果显示：相较 CPU，速度提升 83×–1571×；相较梯度采样器，常规目标 2.2×–18.7×；稀有目标 约 200×；端到端速度提升 31×–494×。

**⚠️ 局限性**

局限性包括：① 冲突轨迹不学习子句，导致在极稀有目标时效率下降；② 没有正式的均匀性保证，只通过统计检验；③ 对 GPU 内存容量敏感，极大公式可能无法一次性并行所有轨迹；④ 对非常大或稀疏子句集合的处理仍需进一步优化。

---

## 82. GSP Autobidding Equilibria with Few Advertisers or Slots

**arXiv ID:** 2610.29310 | [PDF](https://arxiv.org/pdf/2610.29310v1)

**作者:** Yiyang Huang `[一作]` (Beijing Institute of Technology), Zihe Wang `[通讯]` (Renmin University of China)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在 GSP（Generalized Second‑Price）赞助购物拍卖中，预算约束和回报率（ROI）约束下的自动竞价（autobidding）均衡，并给出了在广告主数或显示槽数固定时的多项式时间精确求解算法。

**💡 创新点**

创新点包括：
- 对于固定广告主数的情况，利用层级分解与 Minkowski 和子多面体构造，将所有可能的排名压缩为多项式数目的代表性排名；
- 对于固定显示槽数的情况，引入阈值恢复与容量多面体来描述预算约束；
- 对于 ROI 约束，证明任何均衡可压缩为至多 K+2 个非单位倍增器，保留所有显示位置的项目与价格，从而实现维度约束。

**🔧 技术方法**

技术手段包括：
- 比较细胞分解与严格/平等比较，形成有限维的多项式不等式系统；
- 层级凸化、Minkowski 和子多面体表示，得到可行支付向量的凸包；
- Carathéodory 定理与凸组合压缩，保证分布支持不超过维数+1；
- 子模基多面体与盒子交叉判定，处理预算区间与容量约束；
- 精确多项式可行性检验与符号算术求解。

**📊 数据集**

本文为理论性研究，没有使用实验数据集；所有结果基于符号运算与抽象模型。

**📈 对比分析**

与现有的 FIXP 计算和 PPAD 难度结果相比，本文的算法在固定参数（广告主数 n 或槽数 K）下实现了多项式时间精确解；相对于先前仅能得到近似解或只在特殊结构下可解的情况，显著提升了可计算性。

**⚠️ 局限性**

局限性：
- 仅在广告主数或槽数固定时适用；当两者均大时，算法指数爆炸；
- 假设所有输入（价值、CTR、预算、回报率上限）为有理数；
- 结果基于 GSP 规则且不考虑自价（self‑pricing）或更复杂的拍卖机制。

---

## 83. Engineering Low-Latency, High-Throughput Quantum Decoders on NVIDIA GPUs

**arXiv ID:** 2610.28867 | [PDF](https://arxiv.org/pdf/2610.28867v1)

**作者:** Stefano Markidis `[一作]` (KTH Royal Institute of Technology), Ivy Peng `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在 NVIDIA GPU 上实现并优化 Relay-BP/OSD-0 量子错误校正解码器。

**💡 创新点**

结合 CUDA Graph 的收敛控制、混合 FP16/FP32 与 INT8 量化、warp‑level 集合以及并行 OSD‑0，实现低延迟高吞吐量的新型解码框架。

**🔧 技术方法**

使用 CUDA Graph、warp‑level collectives、混合与 INT8 精度算子、并行 OSD‑0、批量调优、Tensor Core 等技术。

**📊 数据集**

采用 Surface、Hypergraph Product (HGP) 与 Bivariate Bicycle (BB90) 三种代码的 10,000 个噪声样本进行评估。

**📈 对比分析**

与 FP32 基准相比，INT8/混合精度可提升 2–5 倍吞吐；在 10 ms 响应限制下，RTX 5080 的吞吐率超过 GH200 约 41%，同时保持相近的逻辑错误率。

**⚠️ 局限性**

受限于大规模 OSD‑0 的内存占用、对高噪声下的精度敏感，以及批量大小与延迟之间的权衡。

---

## 84. Training in Deployment: Human-in-the-loop Online Post-Training for Robot Manipulation

**arXiv ID:** 2610.29264 | [PDF](https://arxiv.org/pdf/2610.29264v1)

**作者:** Chun Gu `[一作]` (Fudan University), Li Zhang `[通讯]` (Fudan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为TIDE的在线后训练框架，利用在真实机器人部署过程中收集的人机交互数据即时更新预训练操作策略；

**💡 创新点**

创新点在于将部署时产生的执行记录按源（人类或机器人）进行来源标记，并在监督学习中使用来源嵌入进行条件化训练，避免了对价值函数或奖励模型的依赖；

**🔧 技术方法**

采用预训练的视频生成骨干网络、跨模态适配器、流匹配动作专家，并在在线循环中对这些模块联合优化；

**📊 数据集**

使用的训练数据包括任务相关的离线人机遥操作演示以及在线部署期间收集的带来源标记的混合执行轨迹；

**📈 对比分析**

在四个真实双臂操作任务上与warm-start、离线演示扩充以及固定策略收集等基线进行比较，TIDE在无接管评估中实现了与固定策略相当甚至更好的里程碑进度与任务完成率；

**⚠️ 局限性**

局限性包括需要持续的人类监督，且仅在单一双臂平台与特定网络架构上验证，未测试长期保留效果或跨平台泛化。

---

## 85. VERDICT: Multi-Agent Parallel Modular Verification

**arXiv ID:** 2610.29147 | [PDF](https://arxiv.org/pdf/2610.29147v1)

**作者:** Tengjie Lin `[一作]`, Mingshuai Chen `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对大型C程序，提出基于多代理的并行模块化演绎验证框架，在保证正确性的前提下，实现并行推理与跨调用链冲突调和。

**💡 创新点**

创新点：①使用假设驱动的并行规范合成，打破传统自底向上的顺序约束；②双向事件（Supply/Demand）驱动的冲突调和机制，实现跨调用链的合同一致性；③将大型语言模型与多代理架构结合，实现大规模自动规范合成。

**🔧 技术方法**

技术细节：采用Claude LLM（Sonnet 4.6/Opus 4.5）生成规范，Frama‑C作为演绎验证器，配合Z3、cvc5、Alt‑Ergo等SMT求解器；代理间通过事件调度（Supply/Demand）交互；使用Claude Agent SDK搭建调度器。

**📊 数据集**

数据集：15个真实C程序基准（函数数7–210，调用深度0–16），以及20,000行代码的X.509解析器模块；基准来源为现有LLM验证评估数据集与ANSSI X.509模块。

**📈 对比分析**

与两大基线（LLM-Spec、SpecSyn）比较：在规范完成率、RTE消除率和执行时间上均优；最高可达12.4×速度提升，RTE消除率99.9%，规范满足率94.5%；在X.509解析器上实现数小时内全量验证，过去需数月人工。

**⚠️ 局限性**

局限性：①深层调用链的冲突调和产生尾部延迟；②功能级代理粒度导致高耦合时调和成本高；③目前不支持递归/不确定数据结构的归纳谓词，限制了对更复杂程序的覆盖。

---

## 86. Conditional Imaging for Privacy-Preserving Visual Sensing

**arXiv ID:** 2610.28835 | [PDF](https://arxiv.org/pdf/2610.28835v1)

**作者:** Hengyu Lian `[一作]` (University of North Carolina at Chapel Hill), Praneeth Chakravarthula `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于帧级光学调制的条件成像系统，通过在每帧改变相位掩模，实现默认图像不可辨识、但授权时可恢复；

**💡 创新点**

创新点在于将可恢复性与隐私分离，利用随帧变化的光学算子使得恢复仅在知道物理状态时才可能，打破传统静态光学系统的隐私-可恢复性权衡；

**🔧 技术方法**

使用可编程相位调制器（SLM）实现Zernike多项式相位编码，并结合深度展开反卷积、盲去卷积与数据驱动模型进行恢复与评估；

**📊 数据集**

实验基于公开人脸数据集Labeled Faces in the Wild（LFW）以及NYU Depth V2数据集；

**📈 对比分析**

对比盲去卷积、助力去卷积、数据驱动恢复与授权恢复，授权恢复在PSNR≈27.8dB、SSIM≈0.84上显著优于非授权方法，且面部识别AUC从0.67提升至0.91；

**⚠️ 局限性**

局限性包括对可编程光学硬件的依赖、实现速度和成本、对更复杂场景或攻击方式的鲁棒性尚未充分验证。

---

## 87. One Act, Many Masks: Effect-Marginal Policy Optimization for Long-Horizon Agents

**arXiv ID:** 2610.29427 | [PDF](https://arxiv.org/pdf/2610.29427v1)

**作者:** Shaojin Ma `[一作]` (Tianjin University), Jianye Hao `[通讯]` (Tianjin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Effect-Marginal Policy Optimization (EMPO)，通过在可执行等价文本响应之间共享长度归一化的概率比，纠正信用-更新不匹配，提升LLM多轮RL学习效果。

**💡 创新点**

创新点在于识别信用-更新不匹配问题，并将策略更新重构为基于可执行效果的共享比率，无需额外批评器或环境交互。

**🔧 技术方法**

采用LLM代理RL框架、群组策略优化（GRPO、GiGPO）、PPO等基础算法，并结合长度归一化、效果分组与比率共享技术。

**📊 数据集**

在ALFWorld、WebShop和Sokoban三大长期交互任务上进行实验验证。

**📈 对比分析**

与GRPO、GiGPO及多种闭源/提示基准相比，EMPO在成功率和学习效率上均显著提升，尤其在ALFWorld和Sokoban上提升10%+。

**⚠️ 局限性**

局限性包括对可执行效果分组的依赖，若任务缺乏明确效果映射或分组粒度不足，效果提升可能有限。

---

## 88. Encoded but Unused: Bounding the Sources of VLM Errors

**arXiv ID:** 2610.29241 | [PDF](https://arxiv.org/pdf/2610.29241v1)

**作者:** Anat Veikherman `[一作]` (Technion), Roi Reichart `[通讯]` (Technion)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并量化视觉语言模型的错误来源，提出基于排名的视觉证据探针并估计视觉利用失败率。

**💡 创新点**

将错误细分为视觉表征失败与视觉利用失败，给出利用失败率的理论下界与上界，并设计轻量级 Rank‑Probe 追踪视觉证据随层级的演化。

**🔧 技术方法**

采用 logit‑lens 词汇空间排名、逻辑回归探针、答案保持变换推导误差率边界，以及注意力增强纠错等技术。

**📊 数据集**

使用 Visual Attributes in the Wild (VAW) 数据集，包含约 6,000 张图像的颜色、材质、形状属性问答。

**📈 对比分析**

与分类器、置信度、注意力等三类基线对比，Rank‑Probe 在属性证据检测的 ROC‑AUC 最高达 0.95/0.91；利用失败率估计显示 LLaVA 84.4%/Qwen 70% 的错误主要源自视觉利用不足。

**⚠️ 局限性**

局限在于需要已标注的目标区域，对未知属性或无定位信息的场景适用性受限；探针性能受训练标签质量与答案保持变换的强度约束。

---

## 89. Reading the Mask: Descriptor-Grounded BI-RADS Rationale Generation for Breast MRI Segmentation

**arXiv ID:** 2610.29358 | [PDF](https://arxiv.org/pdf/2610.29358v1)

**作者:** Hrishikesh Singh `[一作]` (Indian Institute of Technology Roorkee), Millie Pant `[通讯]` (Indian Institute of Technology Roorkee)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

利用分割掩膜和动态对比增强MRI数据，基于BI‑RADS决策树计算八个定量特征，并将其映射为BI‑RADS描述符，随后使用大型语言模型生成自然语言理由。

**💡 创新点**

创新点在于：①完全无报告监督，仅靠掩膜和时间序列图像即可推导临床描述符；②将定量测量映射到闭合的BI‑RADS词汇表，确保生成文本与输入记录严格对应；③在生成文本时仅提供结构化记录，保证可解释性和可评估性。

**🔧 技术方法**

技术包括：连通分量分析与形态学处理、体积/表面积/球度/细化比例等形态特征计算、早期/晚期增强比、内部增强、信号增强比、初期/晚期动力学阈值；BI‑RADS决策树映射；LLM（Llama‑3.3‑70B）自然语言生成；多种无监督评估指标（faithfulness、hallucination、omission、counterfactual accuracy）。

**📊 数据集**

使用MAMA‑MIA公开数据集，共1506例乳腺DCE‑MRI，包含专家掩膜与自动掩膜，并按质量分为Good、Acceptable、Poor、Missed。

**📈 对比分析**

在专家掩膜上生成3012条理由，faithfulness 99.98%，hallucination 0%，omission 0%，counterfactual accuracy 99.3%；在自动掩膜上评价描述符一致性α从Good掩膜的0.832降至Missed掩膜的0.549，说明描述符与掩膜质量相关。

**⚠️ 局限性**

局限包括：未与放射学专家读片结果或多发性指标比较；阈值仅在单一数据集上拟合，可能缺乏泛化性；使用的定量特征为简单代理（如表面积/体积比受尺寸影响）；所有病例均为恶性，无法评估描述符在良性或多发性病例中的表现。

---

## 90. BrainCodes: A Lightweight EEG Foundation Model over Discrete Neural Codes

**arXiv ID:** 2610.29320 | [PDF](https://arxiv.org/pdf/2610.29320v1)

**作者:** Kleanthis Avramidis `[一作]` (University of Southern California), Dimitra Emmanouilidou `[通讯]` (Microsoft Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研发并公开了一个轻量级 EEG 基础模型 BrainCodes，利用离散神经标记器和自监督预训练，专注于可迁移性、空间和频谱结构的保留，最终在 14 个多样化的 EEG 评估基准上实现高性能。

**💡 创新点**

核心创新包括：
- 采用向量量化（RVQ）离散标记器与 Transformer‑U‑Net 结合的编码器，极大压缩参数量；
- 在 3D 头皮坐标上实现混合轴 RoPE 位置编码，专门捕捉电极间相对距离；
- 引入窄带振荡注入辅助任务，迫使模型在瓶颈层保持对节律与频谱的敏感性，从而提升空间与频谱结构的保留。

**🔧 技术方法**

技术细节：
- BioCodec（RVQ tokenizer）
- Factorized transformer U‑Net（时间+通道注意力）
- 3D RoPE 位置编码（电极相对坐标）
- Masked code prediction 损失 + 频率/持续时间/位置的辅助回归/分类损失
- 线性探针与完整微调两种评估协议
- 归一化平衡准确率（nBAC）与多任务基准对比

**📊 数据集**

数据集：
- 预训练：Healthy Brain Network EEG（约 820 小时，1131 名儿童）
- 下游 14 个任务：
  * 大基准：TUAB、TUEV、ISRUC、Siena、PhysioNet‑MI、BCI‑IV‑2a、EEGMAT、NeuroRVQ（High‑Gamma、OpenBMI、Memory、Eyes）
  * 低资源：EESM23（耳内 EEG）和 Muse 头带（Motor、Mental）

**📈 对比分析**

比较方法：对 5–7 个已发布的 EEG 基础模型（BIOT、LaBraM、CBraMod、CSBrain、REVE、EEGPT 等）在全微调与线性探针两种协议下进行统一评测，使用平衡准确率、AUROC、AUPRC 等指标；BrainCodes 在大多数基准上取得最高或相当的 nBAC，参数仅 1.9 M，且在低资源设备上保持 90% 以上性能。

**⚠️ 局限性**

局限性：
- 仅在公开回溯性数据集上验证，缺乏临床前瞻性评估；
- 预训练数据来自儿童/青少年，成人/老年人群的适用性尚未验证；
- 在极少样本或高维任务时仍可能受限；
- 对潜在可识别信息的隐私风险尚未系统评估。

---

## 91. Compositional neural networks for multimodal X-ray and neutron scattering inversion

**arXiv ID:** 2610.29362 | [PDF](https://arxiv.org/pdf/2610.29362v1)

**作者:** Valentin Munteanu `[一作]` (University of Tübingen), Frank Schreiber `[通讯]` (University of Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

开发了一种基于先验可摊销流匹配（Prior‑Amortized Flow Matching）的概率推断框架，并在此基础上提出了共享‑局部（shared‑local）参数的组合采样（Compositional Flow Matching），实现了不同探测器（X‑射线、时间飞行中子、偏振中子）之间的模块化联合推断。

**💡 创新点**

创新点在于：① 将流匹配的向量场与分数（score）表述结合，能够在推断时直接组合多个单观察后验；② 推导出完整协方差（full‑covariance）共享‑局部组合公式，既校正了共享先验的重复计数，又保留了共享与局部参数之间的相关性；③ 通过两阶段采样（FMPE + MCMC）兼顾速度与精度，且模型可在不重新训练的情况下用于多对比度、多模态实验。

**🔧 技术方法**

主要技术包括：Prior‑Amortized Flow Matching（带两侧条件与线性概率路径）、分数推断与向量场反演、共享‑局部高斯组合规则、基于嵌入网络与残差MLP的条件向量场网络、Dormand–Prince ODE 求解器、GPU并行affine‑invariant ensemble MCMC 细化。

**📊 数据集**

使用了多种真实与合成实验数据集：单曲线XRR（PTCDI‑C₃薄膜、C₆₀薄膜）、TOF‑NR（PS:Bis‑PCBM 结构）、聚合物刷子NR（PS 结构）、全自旋分辨PNR（Co/Fe 双层），以及多对比度NR（D₂O、H₂O、CM）和多模态XRR/PNR（Pt/Fe、Pt/Co）实验。

**📈 对比分析**

与传统点估计网络（如PANPE）、混合密度网络、归一化流以及先前的流匹配模型相比，FMPE‑MCMC 在后验预测、参数分布形状与不确定性量化上更准确；多模态组合后，能够显著缩小共享结构参数的置信区间；两阶段采样在 1–2 次迭代后即可达到与完整MCMC相当的精度。对比实验表明：单观察 FMPE 误差 < 10% 的后验分布，组合后误差进一步降低；在高维参数（≥30）和多通道（4通道PNR）场景下保持稳定。

**⚠️ 局限性**

限制包括：① 近似后验受训练范围与先验边界限制；② 对于多通道PNR 的粗糙度和磁域效应未完全建模；③ ODE 求解器的数值成本在大规模采样时仍显著；④ 组合规则假设各后验可用高斯近似，若实际后验高度多模态则性能下降；⑤ 对共享‑局部分区的手动定义依赖实验者经验。

---

## 92. Workflow-Aware Communication and Computation Scheduling for Mobile Edge AI Agents

**arXiv ID:** 2610.29337 | [PDF](https://arxiv.org/pdf/2610.29337v1)

**作者:** Xingqiu He `[一作]`, Jihong Park `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种面向移动边缘AI代理的工作流感知通信与计算联合调度算法。

**💡 创新点**

创新点在于将加权完成时间转化为持有成本并构造可计算的look‑ahead surrogate，结合完成修正，实现在线评估工作流未来影响的调度决策。

**🔧 技术方法**

使用动态工作流图模型、Bellman方程、平滑最大化（smax）、参考服务率估计、边际成本计算以及伪状态构造等技术。

**📊 数据集**

采用合成的工作流数据集（每个工作流12–40步，深度4–12，最多4分支）以及1000条工作流的尾延迟预测训练集。

**📈 对比分析**

与WSRPT、W-ATLAS和等量共享基线比较，实验显示在高负载/通信瓶颈下平均延迟降低25%–80%，尾部延迟显著改善。

**⚠️ 局限性**

限制在于算法的计算开销相对较大，仅在单个cell和单云环境下验证，未考虑多云异构或工作流迁移场景。

---

## 93. Local and Sparse Kernel Operator Learning for Incompressible Flows: An Operator Lagrange Formulation

**arXiv ID:** 2610.29009 | [PDF](https://arxiv.org/pdf/2610.29009v1)

**作者:** Ramansh Sharma `[一作]` (University of Utah), Varun Shankar `[通讯]` (University of Utah)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种局部稀疏核方法，用来学习不可压流体的解算子，使得每一次预测都保持无散度、周期性，并且输出表示随网格细化线性增长。

**💡 创新点**

创新点包括：① 用局部多项式补齐的无散度RBF‑FD插值取代全局插值，保留物理属性且系统尺寸不随输出点数增大；② 设计可在任意输入维度下正半定的紧支撑核，令回归矩阵稀疏；③ 引入仿射多项式补偿，提升对输入线性趋势的表示；④ 通过Operator Lagrange权重将回归与局部插值耦合，避免存储所有局部系数并仅需一次稀疏解。

**🔧 技术方法**

主要技术包括：无散度多项式构造（Legendre多项式导数/旋度）、多尺度Matérn核、PHS核、紧支撑 Wendland 核与残差 Matérn 乘积、稀疏Cholesky+QR（cuDSS）、GPU并行批量求解、核参数的限制似然+验证搜索、仿射补偿的矩阵化求解。

**📊 数据集**

使用十个先前公开的不可压流动基准（例如强制同质湍流、湍流物种输运、合并涡流、壁面驱动腔、圆柱流、背压步骤、浮力腔、Taylor–Green涡旋等），每组包含 10000 训练、100 验证、100 测试样本。

**📈 对比分析**

与稠密 PPKM、VK‑M、Geo‑FNO、Transolver 等方法在同一训练规模下对比，稀疏局部核模型在所有基准上至少与稠密 PPKM 同等甚至更优，且相对神经算子准确率提升 2 至 10⁶ 倍；存储主要受核矩阵 Cholesky 影响，预测时间可在 GPU 上并行处理后保持线性或次线性增长。

**⚠️ 局限性**

局限包括：① 核矩阵仍呈二次规模增长，需进一步采用层次或多级稀疏化；② 局部插值在切换点可能出现不连续；③ 需要 GPU 和 cuDSS 支持，难以在 CPU 或低算力环境使用；④ 对高维输入仍受主成分投影限制，可能在极高维输入场景下失效。

---

## 94. LLMSeg: Training-Free Text-Prompted Image Segmentation with B-Splines

**arXiv ID:** 2610.29409 | [PDF](https://arxiv.org/pdf/2610.29409v1)

**作者:** Abdullah Hamdi `[一作]` `[通讯]` (King Abdullah University of Science and Technology), Abdullah Hamdi (King Abdullah University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种无训练、只用冻结的多模态大型语言模型（MLLM）即可完成文本提示下的图像分割的完整管线。

**💡 创新点**

核心创新在于：①用MLLM直接预测闭合B‑spline几何（可自适应选择一阶或三阶）而非像素级掩码；②采用迭代关注盒子（focus‑box）精细化修正；③整个流程不需要额外训练或分割头，能够直接利用任何视觉后端升级。

**🔧 技术方法**

技术实现包括：检测阶段返回边界框；分割阶段生成B‑spline闭合环（外环+可选洞环）；光栅化步骤将曲线转为二值掩码；改进阶段通过MMLM多次调用，局部重绘误差区域；所有调用仅通过固定文本提示完成。

**📊 数据集**

在十个基准上评测：ReasonSeg、RefCOCO/+/g、LVIS、COCO、SA‑Co、ADE20K、Cityscapes、ODinW13、RF‑100VL，并在每个基准中使用相应的标准评估指标。

**📈 对比分析**

与SAM 3、Seg‑Zero、LISA、GLaMM、PSALM等现有方法对比，本文在ReasonSeg上获得71.5 cIoU、RefCOCO testB上76.0 cIoU，LVIS/COCO分别为53.9/56.1 AP，ADE20K gIoU 59.2，Cityscapes mIoU 53.3，ODinW13 AP_0 57.2，RF‑100VL AP_0 31.4，整体表现与或优于专家模型，尤其在推理型查询和跨域零样本检测上表现突出。

**⚠️ 局限性**

局限性包括：检测阶段误检/漏检导致分割质量下降；B‑spline光栅化对细节边缘不够精确，难以处理模糊形状；由于多轮MMLM调用，推理时间和计算成本相对较高；模型对非常小或重叠实例的分割仍存在挑战。

---

## 95. Recursive Self-Improvement on a Budget: Scoring Code Edits Before They Run

**arXiv ID:** 2610.29121 | [PDF](https://arxiv.org/pdf/2610.29121v1)

**作者:** Erfan Darzi `[一作]` (Harvard University), Sheng He `[通讯]` (Harvard University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种递归自我改进的编码代理，通过在评估前对其提出的代码差异进行线性读出评分，选择最有潜力的编辑进行展开。

**💡 创新点**

创新点在于：①将父级的 steering-split solve 率与差异的八维坐标一起构成特征进行线性回归预测；②在保持固定归档大小的前提下，使用 Thompson 采样与 Beta 分布进行父节点选择；③对比传统的 parent-rate 归档与 Huxley–Gödel 机器选择，验证读出方法的有效性。

**🔧 技术方法**

技术手段包括：线性回归预测（使用父级 solve 率和八维投影坐标）、Beta(1+n_s,1+n_f) 采样、Thompson 采样、四维归档维护、统一 diff 生成与评估、以及对评估结果进行持久化。

**📊 数据集**

数据集为 HumanEval+（100 hold‑out 例子）和 Polyglot（145 hold‑out 例子），使用 48/64 个样本作为 steering split，96 个校准编辑作为训练集。

**📈 对比分析**

在 GPT‑4o 与 Claude Sonnet 4.5 两个生成器上进行 12 次实验，读出方法在 HumanEval+ 上取得 87.0% 的 hold‑out solve 率，超过 parent‑rate（84.2%）与 HGM（85.1%），在 Polyglot 上则分别为 65.2% 与 64.0%，整体表现优于或相近于现有方法。

**⚠️ 局限性**

局限性包括：读出仅为线性模型，固定系数，且只利用 96 个校准编辑；归档大小固定为 4；只针对两种生成器与两种 benchmark 进行评估；对更复杂的代码差异或更大规模数据集的泛化能力尚未验证。

---

## 96. Are Joint-Embedding Predictive Objectives Effective for Longitudinal EHRs?

**arXiv ID:** 2610.29528 | [PDF](https://arxiv.org/pdf/2610.29528v1)

**作者:** Ashvin Gupta `[一作]` (Imperial College London), Alessandra Russo `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究将 Joint-Embedding Predictive Architecture (JEPA) 迁移至电子健康记录（EHR）序列表示学习，并在 MIMIC‑IV 上实验三种目标构造（span、future、random）与掩码策略。

**💡 创新点**

创新点在于系统评估 JEPA 在临床数据中的可行性，揭示现有 JEPA 在预测目标定义与临床语义匹配上的局限性。

**🔧 技术方法**

使用 Transformer 编码器、EMA 目标编码、两层预测器、均方误差损失、RoPE 时序嵌入以及冻结编码器线性探针进行下游评估。

**📊 数据集**

数据集为 MIMIC‑IV 完整训练集，采用 80:10:10 的患者级划分。

**📈 对比分析**

与标准 MLM 对比，使用 AUROC/AUPRC 评价下游 ICU/医院死亡、再入院等任务，结果显示所有 JEPA 方案均低于 MLM，Future‑JEPA 与 MLM 仅相差约 2%。

**⚠️ 局限性**

主要局限在于缺乏能代表临床患者状态的预测单元，单事件级别的隐层预测无法充分抽象患者状态，导致表现不及 token 级重构。

---

## 97. On the Largest Cardinality Subset Sum

**arXiv ID:** 2610.29280 | [PDF](https://arxiv.org/pdf/2610.29280v1)

**作者:** Deepak Bhati `[一作]` (CISPA Helmholtz Center for Information Security), Karol Węgrzycki `[通讯]` (Max Planck Institute for Informatics)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究最大基数子集和问题，提出单目标和所有目标版本的随机化伪多项式时间算法，并将其应用于投影子集和问题。

**💡 创新点**

创新点在于将颜色编码、分层卷积与Lawler多重性约简相结合，将时间复杂度从传统的O(nt)降至O(n+√n·t)或O(n+t^{3/2})，并首次在所有目标版本实现O(n+t·min{√n,√t})。

**🔧 技术方法**

采用了颜色编码、Chernoff界、布朗曼/金·吴的高效卷积、Lawler的多重性约简、分层合并以及随机化动态规划等技术。

**📊 数据集**

主要为理论算法，不涉及具体实验数据集，所有结果在理论上给出时间复杂度与错误概率。

**📈 对比分析**

与传统O(nt)或O(n·t)算法相比，本文方法在t或n较大时能显著加速，误差概率可控制在n^{-c}，并在投影子集和问题中实现近线性时间。

**⚠️ 局限性**

局限性在于仍为随机化算法，误差概率虽低但非零；对多项式范围内最优性证明仍有待进一步改进。

---

## 98. Learning to Absorb Corrections: Critic-Weighted Residual Flow for Vision-Based Insertion

**arXiv ID:** 2610.29117 | [PDF](https://arxiv.org/pdf/2610.29117v1)

**作者:** Yiqing Yuan `[一作]` (Sun Yat-sen University), Hui Cheng `[通讯]` (Sun Yat-sen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种在线学习框架，将动作块变换器（ACT）与条件残差流策略结合，用于视觉插入任务并在运行时不断自我纠正。

**💡 创新点**

创新点包括①通过成功执行动作的行为吸收机制不断更新ACT，②在更新基础策略后对残差目标进行重标记保持监督一致，③将价值加权的残差流学习与成功执行动作的Flow匹配相结合，以提升学习效率与精确度。

**🔧 技术方法**

使用技术包括 Action Chunking Transformer、条件残差流策略、Flow Matching、Critic‑Weighted Flow Matching、ReWiND 进度估计、Q‑学习、HD‑Space 采集与锚点引导热身等。

**📊 数据集**

数据集为五个真实世界插入任务（电源插头、USB、钥匙插入、缆线路由、针线穿线），实验以30次锚点演示作为热身，并在120k步交互中评估。

**📈 对比分析**

与MILES、AutoSERL、HIL‑SERL、CR‑DAgger、FTC等基线对比，20k步时平均成功率达66%，比FTC的48%提升18个百分点；60k步时平均成功率约87%，在大多数任务上均优于或相当于基线。

**⚠️ 局限性**

限制包括需预先提供可分解任务的锚点演示；对无法分解为子任务或缺乏高质量演示的场景适应性有限；对极其柔性物体仍有进一步改进空间。

---

## 99. Balancing Foundation Models and Trait-Specific Learning for Forehead-Crease Recognition

**arXiv ID:** 2610.29495 | [PDF](https://arxiv.org/pdf/2610.29495v1)

**作者:** Geetanjali Sharma `[一作]` (Indian Institute of Technology Mandi), Raghavendra Ramachandra `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个统一的前额皱纹识别框架，融合可解释的零样本推理、域自适应学习和特征特定模型，并在开口集验证协议下对多种方法进行系统评估。

**💡 创新点**

①整合解释型零样本MLLM推理、基于边缘的域自适应训练和专用特征模型于一体；②提出FIRE解释框架提供可解释推理；③在同一协议下对比多范式，证明域自适应与特征特定模型在前额皱纹识别中优于零样本方法。

**🔧 技术方法**

使用多模态大型语言模型（GPT‑4o、Claude、Gemini）与FIRE提示；视觉基础模型（DINOv2、CLIP）零样本；预训练面部识别网络（MobileFaceNet‑V2、EdgeFace、SwinFace、FaceNet）配合margin‑based metric learning（AdaFace、CosFace、MagFace、ElasticFace）；双损失Edge‑Forehead模型；结构化提示与双向推理。

**📊 数据集**

真实前额皱纹集FH‑V1（247人、约5k张）用于训练；FH‑V2（295人、约8.7M假负对）用于开口集评估；合成前额数据用于零样本评估；合成子集（FH‑V2 synthetic）。

**📈 对比分析**

在统一的开口集验证协议（EER、TMR@FMR=0.1/0.01）下对比MLLM、VFM、域自适应模型和特征特定模型。结果显示Edge‑Forehead取得最低EER 8.30%与TMR 54.01%；GPT‑4o零样本EER 18.22%；DINOv2 24%；传统模型EER>20%；域自适应模型如MobileFaceNet‑V2+CosFace EER 9.31%。

**⚠️ 局限性**

零样本MLLM在细粒度纹理识别上仍显弱；多模态模型易受光照、分辨率影响；训练仍需大量标注且域迁移有挑战；解释性虽好但推理稳定性不足；模型规模大，推理延迟高。

---

## 100. C$^2$U in RAG: Compositional Concept Unlearning in Retrieval-Augmented Generation

**arXiv ID:** 2610.29223 | [PDF](https://arxiv.org/pdf/2610.29223v1)

**作者:** Yiyang Duan `[一作]` (Nanyang Technological University), Wei Yang Bryan Lim `[通讯]` (Nanyang Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在检索增强生成（RAG）中，提出了一种基于查询侧的有界编辑方法（C2U），通过固定编码器与文档索引，利用原子坐标编辑和校准门控，在不改变检索器和生成器的前提下抑制目标概念的检索曝光。

**💡 创新点**

创新点在于：①将概念抑制限制在查询向量的旋转（≤0.2弧度）内；②使用残差原子完成未被原子字典张成的方向；③采用共享预算的多概念门控机制，实现不同概念的协同编辑；④在固定索引的前提下保持查询可解释性与部署简单性。

**🔧 技术方法**

主要技术包括：原子组合表示（ACR）与轻量投影头；负向量预条件化方向与校准门控；残差原子完成方向；原子坐标的权重差值传递；原始查询的信任域更新；以及共享角度预算的多概念组合策略。

**📊 数据集**

使用了五个英语数据集：Amazon、ArXiv、IMDB、MNLI、TOFU；四个冻结编码器：MiniLM、bge、e5、Qwen3；每个数据集配备多种查询/控制示例，进行单概念与多概念评估。

**📈 对比分析**

与多种基线对比（如Delta-Edit、PISCES、INLP、LEACE、RAG-Emb-Ctrl、CURE、DARE-GRL、Filter‑and‑Refill、Rank‑Router）。C2U在所有数据集上实现平均4.25pp的Hit@5下降，同时控制查询的曝光与效用成本均为非正；在查询变形攻击（注入、噪声）下保持负密度和曝光变化；在ArXiv四概念组合上取得正向提升（+4.1% H5提升），优于平均方向比较。

**⚠️ 局限性**

限制包括：①抑制效果仅为部分（Hit@5仍≈90%），未能完全删除概念；②评估范围受限于单语言、固定数据集、有限查询样本；③仅关注检索阶段，未评估生成输出安全、多语言、对抗性攻击等；④低运行时访问并不等同于显著的成本优势。

---

## 101. Cycle structure of best-response dynamics in random bimatrix games

**arXiv ID:** 2610.29153 | [PDF](https://arxiv.org/pdf/2610.29153v1)

**作者:** Georgios Nikolaos Karelas `[一作]` (Pompeu Fabra University), Rui-Ray Zhang `[通讯]` (Monash University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究两人双矩阵随机博弈中最优回应（best‑response）动态的循环结构，重点分析独立、对称、反对称、零和及共同利益五类随机游戏，计算短循环数、最长循环长度和总循环数的渐近分布。

**💡 创新点**

创新点在于：①将最优回应动态映射为有向二分图，利用其结构得到循环出现概率；②针对不同游戏结构推导出循环数的Poisson极限、最长循环≈Θ(√n)、总循环≈(c+o(1))log n 等精确定量结果；③在零和与共同利益游戏中给出显式概率上界和下界，揭示结构对循环行为的显著影响。

**🔧 技术方法**

主要技术包括：组合概率与第一/第二矩方法、随机映射与极限分布、对称/自反对称性分析、积分与极限估计、路径计数与二分图理论、以及极大值与极小值的独立性推导。

**📊 数据集**

实验数据：使用连续分布（如均匀分布）随机生成的 n×n 付款矩阵；理论分析不依赖具体数据集，所有结论均为大样本渐近结果。

**📈 对比分析**

对比方法：与已知的随机博弈（独立游戏的 Poisson 分布、零和游戏的已有结果等）以及理论上已知的极限分布进行比较；结果显示在 n→∞ 时，各类游戏的循环数、最长循环长度与对数量级一致，且不同结构导致的差异在常数与对数因子上显著。

**⚠️ 局限性**

局限性：仅适用于 i.i.d. 连续分布的两人博弈；未处理多玩家(m>2)或带相关/非 i.i.d. 的支付结构；未给出收敛时间或路径长度的期望；理论结果主要聚焦于循环结构而非动态收敛速度。

---

## 102. Character-Level Prosody Control for Mandarin Speech Synthesis:An Architecture-Preserving Approach

**arXiv ID:** 2610.29123 | [PDF](https://arxiv.org/pdf/2610.29123v1)

**作者:** Diankun An `[一作]` (JD.com, Inc.), Xin Tian `[通讯]` (JD.com, Inc.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于大型语言模型（LLM）的中文TTS细粒度音韵控制框架，自动提取字符级音调、时长、能量和停顿标记，并将其作为Unicode符号嵌入文本，利用现有TTS模型的LLM进行微调，使其仅通过文本流解读标记实现对字符级音韵的精确控制。

**💡 创新点**

创新点在于：①构建可大规模使用的字符级音韵标记化流程；②证明在不改变模型架构的前提下，LLM可通过微调学习解读这些标记；③进行跨分词器敏感性分析，揭示ASR优化的分词器对音韵信息的抑制，从而为音韵感知的分词器设计提供假设。

**🔧 技术方法**

主要技术包括：多模态对齐融合（MFA + WhisperX）、RMVPE F0估计、Unicode符号标记设计、LLM微调（CosyVoice2），以及可选的自然语言前端（Qwen3-Omni + Qwen3-8B LoRA）用于将自然语言指令映射为标记序列。

**📊 数据集**

使用的数据集为 VoxBox Mandarin 2.15M 句子（约5,858小时）作为训练与评估语料；用于前端训练的标记实例约 210k 条来自三种情境（电视剧、讲座、直播）并覆盖七种情感类别。

**📈 对比分析**

通过与未改 CosyVoice2 基线比较，实验表明：字符级时长可变动 ±62.9% / -17.4%，音高可调 ±50%；停顿长度符合预设区间；全局说速控制保持分层；听力实验 MOS 结果显示：自然度 4.44/4.46、真实性 4.15/4.10、声像相似 4.00/3.95，差异不显著，说明控制不降低音质。

**⚠️ 局限性**

局限性包括：仅验证中文和单一 TTS backbone；分词器设计对音韵控制敏感，ASR 优化分词器抑制细节；标记阈值为固定常数，缺乏对不同语料的自适应；提取质量受强迫对齐准确性的影响；评估缺少统计显著性检验与分布分析；前端指令到音频的端到端验证尚未完成；跨语言和其他 TTS 架构的推广需进一步研究。

---

## 103. RESEARCHHUB: A Personalized Literature Discovery System for Biomedical Preprints

**arXiv ID:** 2610.28636 | [PDF](https://arxiv.org/pdf/2610.28636v1)

**作者:** Viswonathan Manoranjan `[一作]` (University of North Carolina), Yunmei Liu `[通讯]` (University of Louisville)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一个交互式、个性化的生物医学预印本检索系统，支持通过研究者画像或自然语言查询检索相关论文，并通过前沿检索、多样性重排和学习曲线排序等模块提升探索性。

**💡 创新点**

创新点在于：①将GPT‑4o生成的自然语言查询用于训练领域适配的双编码器检索器PRISM，②采用多负样本排名与MarginMSE混合损失实现精细排序，③结合前沿得分、DPP多样性重排与LID课程排序，实现可控的创新度与多样性；④为结果提供集成梯度可解释卡片。

**🔧 技术方法**

核心技术包括：双编码器检索器（基于SentenceTransformers）、FAISS近似最近邻搜索、GPT‑4o生成查询、Multiple Negatives Ranking + MarginMSE 损失、Determinantal Point Process（DPP）重排、Local Intrinsic Dimensionality（LID）课程排序、Integrated Gradients可解释性。

**📊 数据集**

使用2023‑2025年MedRxiv共44,937篇预印本的标题与摘要做检索文本，去重后构成语料库。

**📈 对比分析**

与四个主流基线（BERT‑MS‑MARCO、DistilBERT‑MS‑MARCO、SPECTER2‑Proximity、SPECTER2‑Search）对比，PRISM在Recall@10上提升12.4pp（0.892 vs 0.768），Recall@100提升9.2pp，NDCG@10提升0.140；在邻域语义质量上AvgSimX@10提升约4.4×，说明检索结果更贴近目标论文。

**⚠️ 局限性**

局限性包括：仅覆盖MedRxiv 2023‑2025预印本，未扩展至同行评审文献或其他学科；依赖GPT‑4o API进行查询生成，缺乏可复现性；前沿度量依赖已书签阅读历史，读者历史稀疏时效果有限；未纳入隐式用户反馈，无法随使用提升推荐质量；部分评估指标受语料稠密度影响。

---

## 104. UNIVERS: Learning UNIversal Voxel-wise Embedding via Randomized Synthesis for generalizable one-shot landmark localization

**arXiv ID:** 2610.29098 | [PDF](https://arxiv.org/pdf/2610.29098v1)

**作者:** Derong Yu `[一作]` (Shanghai Jiao Tong University), Guoyan Zheng `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `67630363-6be0-4f51-ab05-7198250671a5` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了 UNIVERS 框架，利用随机合成的多模态伪数据学习通用体素嵌入，实现任意 FOV 下的一拍标记定位。

**💡 创新点**

将域随机化与跨模态体素对比学习相结合，设计跨域伪图像对、伪补丁四元组以及 CDVC 和 CPSC 损失，实现在无真实训练样本下的端到端学习。

**🔧 技术方法**

采用域随机化数据合成、跨域体素对比学习、双解码器网络以及粗细两级一拍定位技术。

**📊 数据集**

在三组数据集上验证：MAGI CT、DAP Atlas CT 以及 MRISegmentator MRI。

**📈 对比分析**

与多种一拍定位方法（Affine、ConvexAdam、MedLAM、SAM、UAE‑S/M、Anatomix）及上限/下限基准比较，UNIVERS 在三种设置下均取得 MED 更低、SDR_10 更高的显著优势。

**⚠️ 局限性**

仍受极端模态差异影响，缺乏真实对齐的多模态图像，且对合成数据的参数设置有一定依赖。

---

## 105. T-DIG: Text-Conditioned Diversity for Image Generation

**arXiv ID:** 2610.29003 | [PDF](https://arxiv.org/pdf/2610.29003v1)

**作者:** Simran Khanuja `[一作]` (Carnegie Mellon University), Graham Neubig `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于文本的多图多样性生成框架（T2I-Diverge），通过交互式界面让用户指定自由文本的变化轴，并使用冻结的图像生成器与编辑器；

**💡 创新点**

创新点在于将多样性视为条件生成问题，利用LLM策略只微调LoRA权重即可在不触及生成器权重的前提下实现用户指定轴的多样化；

**🔧 技术方法**

技术上采用Qwen3.5 LoRA适配器、GRPO强化学习、TPIPS条件多样性奖励、VQAScore/CLIPScore对齐惩罚以及HPSv3偏好惩罚，并使用VLM辅助生成轴建议；

**📊 数据集**

数据集方面，使用HPDv2用户写作的文本提示，结合VLM合成的轴数据（19,765提示-76,089轴对），并收集405条真实用户交互的测试对；

**📈 对比分析**

与传统i.i.d.、PAG、CADS和Scaling Group Inference等基线比较，T2I-Diverge在多模型上平均提升轴向多样性2.3–3.1倍，基准对齐下降不到2%，且在人类评估中在64–96%的情形下被认为更丰富且同等或更好；

**⚠️ 局限性**

局限性包括依赖基础生成器的可控性（例如无法满足摄像机角度等高级变化）、多样性强度线性调节不具感知可解释性，以及潜在的TPIPS偏差导致的生成偏差。

---

## 106. ARCANA: A Benchmark for Abstraction and Analogical Reasoning

**arXiv ID:** 2610.29271 | [PDF](https://arxiv.org/pdf/2610.29271v1)

**作者:** Ruqi Chen `[一作]` (University of Science and Technology of China), Wei Wang `[通讯]` (University of Science and Technology Beijing)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了面向远程类比推理的 benchmark——ARCANA，涵盖从真实世界概念到抽象格子变换的任务；

**💡 创新点**

创新点在于明确设计了“远程类比”评测框架，并提出 ConceptSIR 通过概念滑移、内省与反思提升 LLM 的类比推理能力；

**🔧 技术方法**

主要技术包括基于 LLM 的概念生成与滑移、代码生成与执行、内省修正与遗传式代码/概念变异；

**📊 数据集**

使用了 ARCANA 自己构造的数据集，包含 200 个任务、35 类概念，任务来源于物理定律、自然现象与日常物体；

**📈 对比分析**

与人类参与者对比，LLM 在严格评估下约 40‑50% 通过，远低于人类平均 69%（含错别字 58%）；概念推理与反射可提升部分任务的通过率；

**⚠️ 局限性**

局限性包括任务规模有限、格子表示难以完整表达复杂几何/拓扑关系、代码执行不够通用、并未覆盖所有可能的类比源。

---

## 107. A2I-Calib++: A Kinematics-Driven Anti-noise Active Calibration Framework for Whole-Body Multi-IMU Systems on Legged Robots

**arXiv ID:** 2610.29438 | [PDF](https://arxiv.org/pdf/2610.29438v1)

**作者:** Fangyu Jiang `[一作]` (Shanghai Jiao Tong University), Ling Pei `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

为腿型机器人提供了全身多IMU的主动校准框架，实现了时空偏移、外参和偏差的自动估计。

**💡 创新点**

提出Noise-Isotropy Proposition生成抗噪曲线、约束感知的对抗式运动先验以及基于滚动接触的虚拟IMU融合，显著提升校准精度与鲁棒性。

**🔧 技术方法**

使用基于强化学习的AMP控制器、运动轨迹设计、SVD+最小二乘求解、前向运动学与接触模型。

**📊 数据集**

在仿真环境和Unitree Go2四足机器人上收集数据，包含4个膝关节IMU、机身IMU、相机IMU与LiDAR。

**📈 对比分析**

与iKalibr及传统行走/正弦参考动作对比，A^2I-Calib++在时空误差上分别减少约3-10倍，后向里程计平均位置误差降至0.36m、角误差3.3°。

**⚠️ 局限性**

对初始膝关节IMU校准误差敏感，受地面摩擦变化影响大，且依赖准确的运动学模型，仿真与实测差距仍存在。

---

## 108. Building a Nested LADDER for LLM Unstructured Post-Training Pruning

**arXiv ID:** 2610.28773 | [PDF](https://arxiv.org/pdf/2610.28773v1)

**作者:** Nicolás Mauricio Cuadrado Ávila `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Samuel Horváth `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种递归地选择最小性能损失的稀疏化策略（Layer‑Wise Allocation via Damage‑Driven Evaluation and Refinement, LADDER），实现对LLM的高稀疏率无结构后训练剪枝。

**💡 创新点**

创新点在于使用增量式、损伤驱动的层级稀疏分配，既保留嵌套掩码又在高稀疏率（≥70%）下显著降低性能损失。

**🔧 技术方法**

结合现有局部剪枝后端（Magnitude, Wanda, SparseGPT）与损伤评估、后续蒸馏等技术。

**📊 数据集**

在OPT、Llama、Qwen系列模型上，使用WikiText‑2、PTB、C4以及七个零样本下游任务的数据集。

**📈 对比分析**

与OWL、DSA、LSA等非均匀分配基线相比，LADDER在70–90%稀疏率下保持更低的交叉熵/困惑度，后续蒸馏后差距进一步缩小，零样本任务的准确率保持更好。

**⚠️ 局限性**

局限性包括：对每一层逐步评估需要多轮搜索，计算成本相对较高；在极高稀疏率下仍受局部后端性能限制；对不同模型的初始分配决策仍影响最终性能。

---

## 109. An Explicit Feedback Contract for Iterative ML-Agent Harnesses

**arXiv ID:** 2610.28798 | [PDF](https://arxiv.org/pdf/2610.28798v1)

**作者:** Kehui Yao `[一作]` (Walmart Global Tech), Kannan Achan `[通讯]` (Walmart Global Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实验验证一种显式反馈合同，用于限制ML‑agent在训练循环中可见的验证信息、定义停止规则和测试数据的使用方式，防止适应性过拟合。

**💡 创新点**

创新点包括①将迭代代理的适应性过拟合问题正式建模；②设计五条合同条款（分割、反馈、停止、测试隔离、记录）；③提出基于配对标准误差的自适应阈值发布规则，无需手动设定margin；④让停止规则在阈值未被突破时自动触发；⑤通过实验展示验证–测试差距显著缩小并且测试性能略有提升。

**🔧 技术方法**

技术手段主要来自适应数据分析与差分隐私的阈值发布思路，利用标准误差估计评估两模型在同一验证集上的差异，统计上给出 O(n^{-1/3} log^{1/3}(kn)) 的误差上界，并在代理循环中实现该阈值机制；同时采用多轮训练、代码编辑、实验提交等代理交互流程。

**📊 数据集**

使用了三组数据集：Criteo‑CTR（100k 训练、25k 验证、50k 测试，AUC 为评估指标）；MSLR‑WEB10K 排序任务（6k 训练、2k 验证、2k 测试，NDCG@10 为评估指标）；以及对Criteo标签进行置换的无信号控制集。还实验了不同公共验证集大小（2k–20k）和不同代理模型（Codex、Claude）。

**📈 对比分析**

通过与传统“Exact”反馈（每次提交返回精确验证分数）对比，评估停止轮次、测试分数和验证–测试差距。结果显示：在Criteo与MSLR任务中，合同使验证–测试差距从约0.0060降至0.0055/0.0123，测试AUC/NDCG略高；在无信号控制中，合同报告的增益与测试增益一致，差距为0，而Exact报告的增益存在假正向。合同在大多数任务中将停止轮次从30降至约12，显示更高效的停止。

**⚠️ 局限性**

限制包括：①只保证发布的验证序列的可靠性，未直接保证最终测试分数；②阈值依赖样本大小，可能在不同任务或极端噪声环境下表现不佳；③实验仅覆盖少量种子和tabular数据，缺乏在更复杂任务或大规模数据上的验证；④实现仍需手工配置合同参数与代理交互细节。

---

## 110. T-WLA: Bridging Slow Vision and Fast Touch

**arXiv ID:** 2610.28814 | [PDF](https://arxiv.org/pdf/2610.28814v1)

**作者:** Kaichen Zhou `[一作]` (Massachusetts Institute of Technology), Paul Pu Liang `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种多速率视觉-触觉架构T‑WLA，利用共享的历史感知视觉记忆指导低频视觉策略生成基础动作块，并在高频触觉更新间通过轻量级触觉编辑器对未执行的动作进行残差修正。

**💡 创新点**

创新点包括：① 将视觉记忆与触觉编辑器共享，实现异步触觉残差校正；② 通过未来图像和未来压力预测提供训练辅助监督，促使控制表示捕捉交互动力学；③ 通过多速率设计在不重新运行视觉分支的前提下实现高频触觉补偿。

**🔧 技术方法**

采用Transformer基础的视觉与触觉编码器；残差编辑网络；条件流匹配的动作监督；对比学习预训练触觉编码器；未来图像和压力预测的训练辅助头；以及历史感知的视觉记忆检索机制。

**📊 数据集**

使用EgoTouch与Tachin手部运动预测数据集；GigaHands、OakInk2、TACO视觉数据；以及真实机器人四个接触丰富任务（蛋转移、单片纸抓取、纸杯嵌套、纸袋拉扯）。

**📈 对比分析**

与VITRA、SmolVLA、MemoryVLA、T‑Rex等基线比较，在EgoTouch和Tachin上实现最低MPJPE，指尖误差下降约30%；在四个机器人任务中平均成功率提升至50%（相较于基线约17%），显著提升闭环表现。

**⚠️ 局限性**

局限性：当基础视觉计划失效时，触觉编辑器难以纠正；在需要精确定位或对齐的子阶段（如纸杯嵌套）效果有限；对多传感器同步与时间对齐要求较高；目前在不同硬件平台上的可迁移性尚待验证。

---

## 111. 5uGUE: Agentic Hardware-in-the-Loop Fuzzing for 5G Devices and beyond

**arXiv ID:** 2610.29155 | [PDF](https://arxiv.org/pdf/2610.29155v1)

**作者:** Guoqiang Hua `[一作]` (Singapore University of Technology and Design), Sudipta Chattopadhyay `[通讯]` (University of Missouri-Kansas City)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一套基于大语言模型（LLM）的自主硬件对环模糊测试框架（Agentic Fuzzing），能够在真实的5G基站与手机设备之间进行无线模糊测试，并扩展到蓝牙经典协议。

**💡 创新点**

创新点在于：①将LLM作为完整的策略师，具备可视化并实时调整模糊循环的能力；②设计了可插拔工具箱，支持网页搜索、协议解析、设备日志分析等；③实现了LLM记忆与自我反思机制，提升模糊的覆盖率和发现率；④实现了对多种LLM模型的无缝兼容，并对不同硬件版本的模糊策略进行自动切换。

**🔧 技术方法**

使用技术包括：LLM驱动的Agentic框架、Python实现的工具接口、SDR（USRP B210）搭建的5G NR测试平台、基于Wireshark的协议捕获与修改、ADB/串口日志收集、基于OpenAI/Claude等API的LLM调用、状态迁移图构建与分析。

**📊 数据集**

数据集主要是：五款不同厂商（MediaTek、UNISOC、以及两款其他手机）提供的商业化5G设备，蓝牙经典设备 ESP32-Ethernet-Kit/ESP-WROOM-32；以及本地维护的5G相关论文和CVE数据库用于外部知识检索。

**📈 对比分析**

与最先进的基线（传统基于单一工具的模糊）相比，Agentic Fuzzing 在 60 次交互（49~133 分钟）内共发现 167 次崩溃（141 次新崩溃、31 次独特），而基线仅发现 13 次（10 新、6 独特）。总体提升约 12.8 倍（独特提升 5.2 倍）。在蓝牙经典协议上，也平均发现 6.4 次新崩溃和 1.6 次独特崩溃。

**⚠️ 局限性**

主要局限包括：①对 LLM 质量和算力的高度依赖，开源模型需要大规模 GPU 资源；②与传统模糊相比，LLM 调用导致额外的时延与成本；③硬件对环环境受限于测试设备与 SDR 的可用性，难以大规模并行化；④缺乏源代码对崩溃原因的可解释性，导致补丁难以精确定位。

---

## 112. MHeatmap: Two-Walk Laplacian Reordering and Mosaic Layout for Rectangular Matrix Visualization

**arXiv ID:** 2610.29114 | [PDF](https://arxiv.org/pdf/2610.29114v1)

**作者:** Juntang Wang `[一作]` (Harvard University), Dongmian Zou `[通讯]` (Duke Kunshan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对加权矩形矩阵，提出一种结合直接关系与长度为两步的共享邻居的谱序列方法，并与比例热图（mosaic layout）结合。

**💡 创新点**

创新点在于在谱序列中加入同侧相似度的Gram项，使得行与列的Fiedler向量同时体现共享邻居平滑与直接对齐，且提供了稀疏矩阵向量乘法实现。

**🔧 技术方法**

使用谱分解（Fiedler向量）与图拉普拉斯，构造两步行走加权邻接矩阵，并利用稀疏线性代数求解特征向量，随后按坐标排序；布局采用基于行列边际比例的等比矩形划分。

**📊 数据集**

评估包括300个合成矩阵（五种结构、三种尺寸）以及七个真实数据集：SIC→NAICS、CIP→SOC、ACS OCCP×INDP、LODES Home×Work、20 Newsgroups Term×Document、MBTA Route×Station、OpenAlex Author×Topic。

**📈 对比分析**

与原始顺序、边际排序、层次聚类、单步谱序列、对应分析、交替中位数等方法比较，使用R2S、Band@10%和MWB‑AUC三指标，实验表明TW在大多数情形下R2S下降约15%，Band@10%提升约5%，在共享邻居信息显著的案例（交叉映射、嵌套群组）表现最佳。

**⚠️ 局限性**

局限性包括在极小矩阵、纯空间或路网结构、以及已足够由单步直接关系解释的情形下效果不明显；同时参数α需通过内部指标自适应选择，且不保证对所有可视化任务都有感知优势。

---

## 113. RoPT: Robust 6DoF Object Pose Tracking in Video

**arXiv ID:** 2610.28962 | [PDF](https://arxiv.org/pdf/2610.28962v1)

**作者:** Yongqing Liang `[一作]` (New York University), Chen Feng `[通讯]` (New York University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aaccfe5c-6b26-4208-b23c-35331481e142` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了RoPT，一种闭环6-DoF物体姿态跟踪框架，能够在高速、激烈运动下保持跟踪稳定；

**💡 创新点**

创新点在于将遮罩引导视觉前端、连续运动因子图的多假设后端以及基于预测渲染的失效感知自恢复模块紧密耦合，实现了对快速动态的鲁棒性；

**🔧 技术方法**

采用的技术包括：基于EdgeTAM的2D遮罩跟踪、GTSAM实现的SE(3)因子图平滑预测、Spherical Harmonics光照优化、Mast3R特征匹配的自恢复和render-and-compare的姿态细化；

**📊 数据集**

主要使用公开的YCBInEOAT和YCB-Video两大基准数据集，并在这些数据集上进行速度下采样实验以模拟高动态场景；

**📈 对比分析**

与TrackNet、RGBTrack、FoundationPose及其++版本对比，RoPT在不同速度下均保持最高或相近的ADD/ADD‑S得分，平均帧率约18.6 FPS，显示出更好的鲁棒性与准确性；

**⚠️ 局限性**

局限性包括：依赖精确CAD模型且仅适用于RGB‑D传感；对长时间遮挡或极端姿态变化时可能失去搜索范围；以及在进一步加速方面受因子图优化复杂度限制。

---

## 114. Intergroup Bias Predicts Expressions of Social Judgment and Empathy in the r/AmITheAsshole Reddit Community

**arXiv ID:** 2610.29254 | [PDF](https://arxiv.org/pdf/2610.29254v1)

**作者:** Yu Hou `[一作]` (University of Maryland), Rachel Rudinger `[通讯]` (University of Maryland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在r/AmItheAsshole社区中，研究了性别和年龄的跨群体同理心差距，量化评论者对发帖者的判定和情感表达，构建了1.26万条评论、3.6万条帖子的大规模数据集；

**💡 创新点**

首次系统性将社会身份（性别、年龄）与在线评论中的社会判断和文本同理心关联，并揭示不同群体身份与冲突上下文相互调节的效应；

**🔧 技术方法**

利用正则与LLM（Gemma-3-27b-it）进行属性提取、主题标注，使用混合效应逻辑回归和LLM评分的情感维度（情绪反应、解释）来评估同理心；

**📊 数据集**

核心数据集为r/AmItheAsshole 2018‑2022年投稿和评论，包含约1.26万条带判定的评论、3.6万条自报性别/年龄的帖子；

**📈 对比分析**

相较于传统文本情感分析，LLM标注的同理心尺度在置信度上优于人工注释（情绪反应F1≈0.73，解释F1≈0.92），并通过混合效应模型显著检验群体效应；

**⚠️ 局限性**

局限包括：依赖自报身份导致样本筛选偏差，LLM在属性提取与同理心标注上的误差，数据仅覆盖至2022年，且未考虑自我认同与感知身份的差异。

---

## 115. Binder migration across film consolidation and pore emptying in electrode drying: a continuous model validated against drying-rate, thickness, and schedule experiments

**arXiv ID:** 2610.29286 | [PDF](https://arxiv.org/pdf/2610.29286v1)

**作者:** Younghwan Joo `[一作]` (Korea Institute of Energy Research), Sung-il Kim `[通讯]` (Korea Institute of Energy Research)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个跨越固化与孔隙空化两个阶段的连续粘剂迁移模型，利用指定蒸发通量驱动干燥过程。

**💡 创新点**

创新点在于通过单一参数集统一描述干燥速率、厚度和时间调度对粘剂迁移的影响，并引入五项修正项解决传统模型在两阶段耦合中的不足。

**🔧 技术方法**

采用一维对流-扩散方程结合多孔介质渗流、相移闭包与Péclet数缩放，以及局部泵送、收缩动力学和粘剂失动机制等改进。

**📊 数据集**

使用文献中的四组实验数据：干燥速率系列、厚度系列、切换时间实验和三阶段调度实验作为验证数据集。

**📈 对比分析**

通过对粘剂表面‑接口浓度比、浓度剖面和接口粘附的数值仿真与实验对比，误差在0.1–0.2范围内，预测准确性较高。

**⚠️ 局限性**

模型在切换时间窗口宽度与快速干燥导致接口粘附损失量的预测上存在偏差，需要进一步实验验证和改进。

---

## 116. NormTune: Norm-Guided Parameter-Efficient Fine-Tuning for Reasoning-Intensive Tasks

**arXiv ID:** 2610.29134 | [PDF](https://arxiv.org/pdf/2610.29134v1)

**作者:** Zhangming Li `[一作]` (Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences), Jian Cheng `[通讯]` (Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对推理密集任务进行参数高效微调，提出NormTune自适应调整学习率以补偿权重更新不足。

**💡 创新点**

通过观测权重更新范数差距，提出基于层级范数的学习率分桶机制，显著提升低秩PEFT在推理任务上的性能。

**🔧 技术方法**

结合LoRA、Bone、MoRA等PEFT方法与NormTune的层级学习率调度，利用权重更新范数进行动态学习率分配。

**📊 数据集**

主要使用S1K-1.1、Z1-Code-Reasoning107K、AIME 2024/2025、MATH 500、GPQA等推理与代码推理基准数据集。

**📈 对比分析**

与Full‑FT、其他PEFT以及RL基线对比，NormTune在rank‑8时平均提升约24%–16%，在rank‑256时可逼近或超过Full‑FT，在多尺度模型上均表现出显著性能提升。

**⚠️ 局限性**

对早期训练阶段范数估计的依赖导致在极端稀疏或非推理任务中效果可能不如预期，且在更大rank或不同模型架构上的通用性尚待进一步验证。

---

## 117. Automating ML for Science: Can Frontier Agents Climb Scientific Hills in the Wild?

**arXiv ID:** 2610.28763 | [PDF](https://arxiv.org/pdf/2610.28763v1)

**作者:** Ming Zhong `[一作]` (Anthropic), Matthew Jagielski `[通讯]` (Anthropic)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并评估了一个面向科学机器学习的长周期自动化实验平台 Agent4Sci，让前沿编码代理在真实科研环境中对20个多学科基准模型进行无人工干预的改进。

**💡 创新点**

首次引入了“科研自动化攀登”范式，设计了三大插件技能（研究构想、实验管理、定时检查表）以提升代理的创新性和资源利用率，显著提升平均相对改进率。

**🔧 技术方法**

利用大型语言模型（Claude Opus 4.6、Claude Sonnet 4.6、Codex、Gemini 3.1 Pro）作为编码代理，结合插件技能、工具调用与评估审计机制，构成了三阶段评估流程。

**📊 数据集**

覆盖11个科学领域的20个公开基准（如PFDelta、Mars-Bench、GEO-Bench-2、TIDMAD 等），每个基准提供标准化的代码、数据集与评测脚本。

**📈 对比分析**

与同类编码代理相比，Agent4Sci 通过三阶段评估显示所有代理均能突破原始基准，Claude Opus 平均提升 10.4%（插件后提升至 20.1%），Codex 从 4.4% 提升至 8.1%，并且评估阶段未发现任何学术不端。

**⚠️ 局限性**

局限在于大多数改进仍依赖机器学习工程技巧，缺乏真正的科学推理和跨领域知识迁移，且插件对不同模型的适用性仍需进一步验证。

---

## 118. Probing Spatial and Spectral Grounding in EEG Foundation Models

**arXiv ID:** 2610.29367 | [PDF](https://arxiv.org/pdf/2610.29367v1)

**作者:** Kleanthis Avramidis `[一作]` (University of Southern California), Dimitra Emmanouilidou `[通讯]` (Microsoft Research)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对五个EEG基础模型使用空间和频谱扰动探针，评估它们对通道位置和频谱结构的编码与利用情况。

**💡 创新点**

提出空间与频谱双重扰动探针框架，揭示模型内部表征与下游决策之间可能的解耦。

**🔧 技术方法**

采用通道标签置换、波形扰动、IAAFT伪影生成、窄带频率注入等实验操作，并配合PCA降维和线性读头进行分析。

**📊 数据集**

使用PhysioNet-MI（运动想象四类）和Siena Scalp（癫痫与非癫痫窗口）两大公开EEG数据集。

**📈 对比分析**

通过线性分类器准确率、嵌入位移量、特异性等指标比较模型，结果显示模型架构并不决定其空间/频谱敏感度或临床特异性。

**⚠️ 局限性**

仅采用冻结模型与线性读头，样本量有限，缺乏对微调后非线性信息和更广泛临床验证的探究。

---

## 119. TAGR: Task-Adaptive Geo-Semantic Retrieval across Heterogeneous Urban Delivery Tasks

**arXiv ID:** 2610.28638 | [PDF](https://arxiv.org/pdf/2610.28638v1)

**作者:** Zhi Liu `[一作]` (Zhejiang University of Technology), Xiangjie Kong `[通讯]` (Zhejiang University of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种统一的任务适应式地理语义检索框架 TAGR，用于城市即时配送中的需求预测、ETA 预测和下一站预测。

**💡 创新点**

创新点在于构建 LLM 编码的共享地理语义先验，同时为不同任务维护专属检索库，并通过任务自适应检索与输出对齐融合，避免不同任务之间信息不匹配与负迁移。

**🔧 技术方法**

技术主要包括冻结 LLM 的文本编码、任务特定适配器、检索投影、FAISS 索引、任务特定编码器、门控共享-私有表示、注意力检索融合、可靠性门控等。

**📊 数据集**

使用真实物流运营数据 LaDe-P 数据集中的三个城市子集：杭州、重庆、吉林。

**📈 对比分析**

与传统时空模型（DCRNN、GWNet、MTGNN 等）、启发式与检索增强基线（HA、SPEED、KNN、RAST、IMPEL 等）以及专用路由模型对比，在所有任务与指标上均取得更优或相近的表现，特别在下一站预测上提升显著。

**⚠️ 局限性**

局限包括：检索库仅离线构建，未实现在线更新；对检索大小的敏感性需手动调参；模型对不同城市的迁移性能未深入研究。

---

## 120. Beyond source attribution: Address authorship and comparability decisions in an LLM-built scholarly knowledge graph

**arXiv ID:** 2610.28663 | [PDF](https://arxiv.org/pdf/2610.28663v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 121. Generative Recommenders Forget the Long Tail: Restoring Memorization in Retrieval and Ranking

**arXiv ID:** 2610.29298 | [PDF](https://arxiv.org/pdf/2610.29298v1)

**作者:** Xu Huang `[一作]` (University of Science and Technology of China), Xiang Wang `[通讯]` (University of Science and Technology of China)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种训练无关的插件 CAPE，恢复生成式推荐器在检索和排序阶段对长尾项目的记忆能力。

**💡 创新点**

创新点在于：①通过共现计数构建检索记忆通道直接补全 beam 之外的长尾项目；②在排序时用共现证据校准概率并加上尾部掩码，保证提升长尾召回而不牺牲整体精度；③保持首位候选不变，方便部署。

**🔧 技术方法**

使用的技术包括：基于语义 ID 的束搜索、共现计数矩阵（用户-项目、项目-项目）、教师强制生成评分、最小-最大归一化融合、top‑1 锚定。

**📊 数据集**

实验数据集包括：Amazon Industrial & Scientific、Amazon Office Products、LastFM-1K（音乐）和 Amazon Beauty（更大规模）。

**📈 对比分析**

与原始生成器、宽束、分组束搜索、流行度校正、MMR、P‑xQuAD、NCP、LightGCN retriever 等方法比较，CAPE 在所有设置下显著提升长尾 HR@10 与 NDCG@10，同时整体 HR@10 保持或提高，延迟仅增加约 10‑15%。

**⚠️ 局限性**

局限性：依赖训练阶段的共现统计，无法覆盖新加入且无共现的项目；对更大模型或不同 ID 编码方式的适用性尚未充分验证。

---

## 122. SOAR: Social and Opinion-Aware Recommendations

**arXiv ID:** 2610.29556 | [PDF](https://arxiv.org/pdf/2610.29556v1)

**作者:** Giannis Christoforidis `[一作]` (Aristotle University of Thessaloniki), Apostolos N. Papadopoulos `[通讯]` (Aristotle University of Thessaloniki)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 SOAR 模型，通过结合用户好友网络与文本评论情感，为每个用户生成个性化的 POI 推荐。

**💡 创新点**

创新点在于同时构建语义评论图与情感权重，并将好友情感信息融入图嵌入，从而实现更精准的个性化推荐。

**🔧 技术方法**

使用 SBERT 生成句子嵌入，FAISS 进行邻居图构建，Hugging Face 情感分类，RandNE 做图嵌入，最终用余弦相似度和随机投影评估候选位置。

**📊 数据集**

实验基于公开的 Yelp 数据集（约 700 万条评论、150k 位置、平均每用户 10 条好友）。

**📈 对比分析**

与仅使用历史访问或仅用文本的基线相比，SOAR 在 Top‑1、Top‑5、Top‑10 的准确率、召回率、F1 及 Top‑K Hits 上均提升 3%–6%，且 2‑hop 版本比基线快约 8 倍。

**⚠️ 局限性**

局限包括：仅考虑直接好友而忽略更深层社交关系；对大规模数据仍存在计算瓶颈；依赖情感分析的准确性；未整合时间/移动等额外上下文信息。

---

## 123. RL-IF: Tracking Non-local Training Data Influence for Online Reinforcement Learning

**arXiv ID:** 2610.28883 | [PDF](https://arxiv.org/pdf/2610.28883v1)

**作者:** Shixuan Liu `[一作]` (University of Illinois Urbana-Champaign), Jiaqi W. Ma `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了在线强化学习中数据影响的留一量化指标RL-LOO，并推导出一阶近似估计器RL-IF，以捕捉优化路径和采样路径对后续训练的全局影响；基于此提出数据过滤方法RINSE，提升训练效率和最终性能。

**💡 创新点**

创新点在于：①将留一概念推广到在线RL的轨迹级别，体现跨轮次的非局部数据影响；②构造RL-IF估计器，分离并量化优化效应与采样效应；③发现采样效应通常较小，提出仅用优化效应的RL-IF-opt实现高效近似；④将上述理论应用于RINSE，实现端到端的数据筛选。

**🔧 技术方法**

主要技术包括：留一量化（RL-LOO）、一阶影响函数推导（RL-IF）、梯度与Hessian向量积（HVP）近似、KL-正则化PPO、经验回放与采样调整、以及在RLHF中使用Generalized Gauss–Newton矩阵加速。

**📊 数据集**

实验数据集涵盖六个Gym/Gymnasium连续控制与导航任务（Ant、HalfCheetah、LunarLander、BipedalWalker、Hopper、FrozenLake），以及GPT-Neo 2.7B模型的RLHF毒性缓解任务。

**📈 对比分析**

与标准PPO、随机过滤、IIF（基于TracIn的局部过滤）对比，RINSE在所有任务上都取得更高的峰值奖励、更低的时间与样本效率（平均提升约30‑40%），并在RLHF毒性缓解中实现约3倍的训练速度提升。

**⚠️ 局限性**

限制包括：①仅针对PPO及其KL正则化形式；②基于第一阶SGD更新，未对Adam等自适应优化器提供严格推导；③采样效应在所实验的超参范围内较小，未充分探索其在更极端或不同算法中的重要性。

---

## 124. Back Up Before You Regret: Measuring Backup Awareness and Automating Backup in Coding Agents

**arXiv ID:** 2610.28934 | [PDF](https://arxiv.org/pdf/2610.28934v1)

**作者:** Xu Li `[一作]`, Weiyan Shi `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

撰写了一个论文写作模板，提供了标准的章节结构和占位符

**💡 创新点**

创新点在于整合了常见的论文写作要素，并提供了易于使用的框架

**🔧 技术方法**

使用 LaTeX 语法来构建论文结构，但未涉及具体算法或技术实现

**📊 数据集**

未使用任何实际数据集，仅列出了示例占位数据集A、B

**📈 对比分析**

实验部分仅展示了占位的基准与自家方法的对比，未给出真实性能指标

**⚠️ 局限性**

局限性是缺乏具体内容、实验结果和技术细节，无法验证方法有效性

---

## 125. GaP-MoE : Explicit Expert Specialization in Mixture-of-Experts via Orthogonal Gradient Accumulation and Projection

**arXiv ID:** 2610.29355 | [PDF](https://arxiv.org/pdf/2610.29355v1)

**作者:** Donghyun Jeong `[一作]` (University of Seoul), Yoon-Yeong Kim `[通讯]` (University of Seoul)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 GaP-MoE 方案，利用梯度正交化在 SMoE 训练流程中直接驱动专家多样性。

**💡 创新点**

创新点在于只对已计算的专家输出梯度进行投影，而不引入额外损失或重新前向/反向传播，从而保持任务目标不变并有效抑制专家相似性。

**🔧 技术方法**

核心技术包括动态领导者选择、梯度累积（Gram-Schmidt 维护梯度子空间）、投影操作、TopK 路由以及与 SMoE 的无缝集成。

**📊 数据集**

使用 BERT fine-tuning 的 GLUE 基准、GPT-2 预训练的 WikiText-103、enwik8、text8 数据集，并在 XMoE、SimSMoE 等 MoE 变体上进行兼容性测试。

**📈 对比分析**

通过与标准 SMoE 及其多种变体对比实验，发现 GaP-MoE 在 GLUE 平均提升约 0.1% 以上，在预训练任务上显著降低 PPL/BPC，并在路由未训练时表现出更高的鲁棒性。

**⚠️ 局限性**

局限性包括未系统优化的超参数（缓冲区大小、γ、α 等）、实验规模相对较小以及在更大规模语言模型上的验证不足。

---

## 126. Heard But Not Said: Prompting Understates Temporal Information of Musical Moments in Frozen Large Audio-Language Models

**arXiv ID:** 2610.29256 | [PDF](https://arxiv.org/pdf/2610.29256v1)

**作者:** Eunsik Shin `[一作]` `[通讯]` (Seoul National University), Eunsik Shin (Seoul National University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建训练自由的配对查询探针，评估七种开放权重的大型音频‑语言模型在音乐时间定位中的能力。

**💡 创新点**

通过在Slakh2100中同时对状态查询和变化查询使用相同音频段，隔离出仅与时序计算相关的差异，并比较不同提取方式（完整提示、窗口评分、注意力）对定位能力的影响。

**🔧 技术方法**

配对查询探针、窗口化无监督评分、基于最终提示词的注意力余量、统计检验（McNemar、Bootstrap CI）等技术。

**📊 数据集**

Slakh2100合成多乐器MIDI数据集，生成稀疏纹理与乐器离场事件对。

**📈 对比分析**

对七种模型在状态/变化查询下的R@1@0.3/0.5指标进行比较，发现短段编码器在窗口评分中可利用前置上下文提升变化查询得分，而完整提示与注意力方法几乎不超基线；整体得分仍低于常数跨度基线。

**⚠️ 局限性**

探针仅关注乐器存在/消失的时间信息，未涵盖和声、节奏、动态等更复杂音乐属性；使用合成MIDI而非真实录音；全轨定位能力仍不足。

---

## 127. Real2Sem2Real: 3D Semantic Augmentations for Robot Learning from a Single Human RGB Demonstration

**arXiv ID:** 2610.28880 | [PDF](https://arxiv.org/pdf/2610.28880v1)

**作者:** Matthew Strong `[一作]` (Stanford University), Monroe Kennedy `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

通过一段人类RGB视频和手机扫描，利用DINOv3蒸馏的高斯斑点实现对象的4D跟踪，随后对数字孪生（cousin）进行语义与功能对齐，生成数千条机器人演示；

**💡 创新点**

首次实现单一RGB视频即可跨类别生成语义增强的演示，解决了先前方法只能在空间或单实例上扩增的局限；

**🔧 技术方法**

核心技术包括DINOv3蒸馏的Gaussian splat跟踪、语义ICP与任务加权ICP对齐、HaMeR手抓重定位、并在IsaacSim/MuJoCo中并行生成；

**📊 数据集**

数据来源为一段人类RGB演示、手机扫描的真实物体、以及少量照片生成的数字孪生；

**📈 对比分析**

与传统的Teleoperation、R2R2R+等基线相比，R2Sem2R在四个任务上在两台机器人上均能达到或超过遥控操作的成功率，且训练样本量大幅降低；

**⚠️ 局限性**

需要已知物体的精确尺寸，单目深度不够可靠，仅适用于刚体，缺乏接触物理细节，且对可扩展性仍有限。

---

## 128. Compressing Onboard Language Models for Space Robots under a Behavioral-Fidelity Requirement

**arXiv ID:** 2610.29302 | [PDF](https://arxiv.org/pdf/2610.29302v1)

**作者:** Jia Deng `[一作]`, Zehao Jia `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在空间机器人上使用低比特压缩语言模型的方法，提出一种基于不同量化算法选择并用最终输出分布做评分的“composed”模型构造方法。

**💡 创新点**

在保持单一比特宽度和可验证静态工件的前提下，融合AWQ、GPTQ、AutoRound等不同量化器并使用最终输出KL作为无标签的可接受性指标，实现比单一量化更高的推理精度和更低的分散误差。

**🔧 技术方法**

采用Post‑Training Quantization（AWQ、GPTQ、AutoRound）、逐模块贪婪搜索、前向KL分数、目标适应模式以及合成模型的本地化导出等技术。

**📊 数据集**

采用C4、WikiText‑2、Code、Math、Wiki等通用文本作为校准和评估数据，辅以GSM8K、MATH‑500、MMLU、ARC‑Challenge、HellaSwag、GPQA等下游任务数据集。

**📈 对比分析**

通过比较同一模型在3/4位压缩下的PPL、KL和多任务精度，显示组合模型在平均PPL和KL上均优于任何单一量化方案，并在微任务上提升4–5点精度；相较于CALM、CKA、Local‑MSE等基线显示显著性能提升。

**⚠️ 局限性**

仅在通用文本上评估，未测试空间域命令、飞行硬件、低功耗、延迟和辐射效应；仅支持固定比特权重量化且未考虑跨层变换或视觉语言模型；更新依赖完整重新量化。

---

## 129. Sketch2Sim: an end-to-end modular framework for hand-drawn circuit analysis

**arXiv ID:** 2610.28836 | [PDF](https://arxiv.org/pdf/2610.28836v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 130. A Stratified Multi-Rater Evaluation of LLM-Based Virtual Standardized Patients with a Deployed Data-Generation Platform

**arXiv ID:** 2610.29312 | [PDF](https://arxiv.org/pdf/2610.29312v1)

**作者:** Yongjin Yi `[一作]` (Dankook University), Sejoong Kim `[通讯]` (Seoul National University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了四款前沿LLM在三种韩语OSCE案例中的虚拟标准化病人表现，并构建了一个网页平台收集学生-虚拟病人对话。

**💡 创新点**

提出了完整交叉多评审的句子/会话级别评价框架，并通过双轴（历史采集类别与语言形式）细粒度划分揭示模型缺陷；同时发布了实时VSP平台与公开数据集。

**🔧 技术方法**

使用 persona‑injection 触发LLM，GPT-4o、Claude 3.5 Sonnet、Llama‑3.1 70B、HyperCLOVA X 四大模型；采用线性混合效应模型、ICC 与 Krippendorff α 进行多评审可靠性分析。

**📊 数据集**

构建了350条由70个输入提示（23 hematuria, 23 fever, 24 cough）产生的多源响应语料，含人类专家对照；并公开 HuggingFace 上的 1,614 真实学生-VSP 对话数据集。

**📈 对比分析**

通过17名医学学生对每个句子和会话做 Likert 评分，使用线性混合效应模型比较各模型与人类参考的差距；结果显示三款模型与参考相差 0.08–0.47 分，Llama‑3.1 70B 在所有六项指标上显著低于参考。

**⚠️ 局限性**

仅有一位人类专家作为对照，评审人数有限，模型仅为文本交互，且仅覆盖三种OSCE案例，缺乏多样性与口语、表情等非语言信息。

---

## 131. Beyond Binary Labels: Diagnostic Evaluation of LLM Safety with Prompt-Level Rubrics

**arXiv ID:** 2610.29225 | [PDF](https://arxiv.org/pdf/2610.29225v1)

**作者:** Hongyu Li `[一作]` (Hunyuan Team, Tencent), Renfen Hu `[通讯]` (Beijing Normal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了DiagSafe安全评估框架与数据集，采用prompt‑level rubric细粒度评估LLM安全性，并对多款LLM进行诊断；

**💡 创新点**

创新点在于：①引入prompt‑level rubric，实现非二元、可解释的评估；②设计了84标签三级安全分类体系，覆盖更细粒度与多样化风险；③构建双语（英中）多轮、攻击类型多样的评测数据；

**🔧 技术方法**

技术手段包括：使用LLM自动生成prompts与rubric；基于三层标签体系制定规范；以Hy3等LLM judge执行评估；采用一致性指标Cohen κ、对比四类评估范式；

**📊 数据集**

数据集：DiagBase（2401样本）与DiagAttack（1088样本），共3499条，包含单/多轮、模板/指令攻击、双语（中/英）；

**📈 对比分析**

与多种评估范式比较，rubric‑judger达95.34%准确率，Cohen κ>0.82，明显优于generic prompt、label‑spec与安全分类器；在11款LLM上发现攻击式评测（DiagAttack）unsafe率比基线提升1.9–8.4×，揭示细粒度风险差异；

**⚠️ 局限性**

局限性：①依赖LLM生成的标签规范，覆盖可能不完整；②仅评估本地开源LLM，未覆盖商用API或思考模式；③仅面向对话型LLM，未扩展到代理/工具使用等更复杂场景。

---

## 132. STAR-XPC: Set Transformer for Estimating X-ray Transmission in Polymer Composites

**arXiv ID:** 2610.28709 | [PDF](https://arxiv.org/pdf/2610.28709v1)

**作者:** Can Erer `[一作]` (Boğaziçi University), Nebahat Aral `[通讯]` (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了基于Set Transformer的STAR‑XPC模型，利用化学感知的分子嵌入和四个宏观物理描述符，快速预测聚合物复合材料的X射线能量透射率；

**💡 创新点**

核心创新是将ChemBERTa分子嵌入与Permutation‑Invariant Set Transformer相结合，构建可处理不定长度成分集合的模型，在子毫秒级推理速度下实现与Monte Carlo相当甚至更高的精度；

**🔧 技术方法**

技术实现包括ChemBERTa对SMILES的分子嵌入、Set Transformer（ISAB+PMA+SAB）实现集合不变性、PENELOPE/penEasy自动化Monte Carlo数据生成、AdamW + cosine 学习率调度、10折联合分层交叉验证；

**📊 数据集**

数据集为自动生成的4697条PENELOPE/penEasy模拟样本，覆盖12种遮蔽材料、最多3种成分组合、4种管电压（30/80/100/120 kV）以及0.15–2.5 mm厚度；此外还评估了6个未出现组合和两种未见元素Pb、Gd；

**📈 对比分析**

在相同10折条件下与Ridge、Random Forest、RBF‑SVR三种基线比较，STAR‑XPC NMSE%0.0278%（R²≈0.9988），比基线低约2–3×；在hold‑out组合上NMSE%0.0496%，比SVR高效1.8×；单样本α‑trimmed APE仅1.66%，推理时间<1 ms，远快于Monte Carlo数小时；

**⚠️ 局限性**

局限性包括聚合物成分固定导致嵌入无辨识度；模型仅训练至最多3种遮蔽材料，且固定集合大小为11，超出此范围需验证；ChemBERTa预训练域与射线屏蔽物质差异；未对更大混合数或不同聚合物的泛化进行充分测试。

---

## 133. Online Open-World Wafer Defect Discovery using Hyperdimensional Computing

**arXiv ID:** 2610.28949 | [PDF](https://arxiv.org/pdf/2610.28949v1)

**作者:** Quanling Zhao `[一作]` (University of California San Diego), Tajana Rosing `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了WaferHDC，一个基于高维计算的在线开世界晶圆缺陷发现框架，能够在不需要标签和预定义缺陷类别的情况下，实时识别并聚类重复出现的缺陷模式。

**💡 创新点**

创新点在于将高维向量记忆结构同时用于正常模式建模、异常检测与递增缺陷模式发现，实现了无标签、动态可扩展的在线缺陷监测，并通过候选到持久缺陷内存的递归晋升机制实现持续学习。

**🔧 技术方法**

采用高维计算（HDC）技术，将晶圆地图编码为二进制高维向量，利用热编码、Hamming距离进行快速相似度检索，并结合增量计数、平均异常分数和内在紧凑度等指标进行在线聚类与模式升级。

**📊 数据集**

使用工业标准的WM-811K晶圆图像数据集，包含八类未知缺陷模式与无缺陷样本。

**📈 对比分析**

与多种经典异常检测（OCSVM、Isolation Forest、KNN、LOF、PCA、HBOS、ECOD、GMM）及在线聚类（BIRCH-CF、DenStream、DP-means）进行配对对比，Stage‑1 AUROC达97.37%，Stage‑2 ARI最高58.07%、NMI 70.68%，完整管线相较最佳基线提升ARI 13.3个百分点、NMI 10.5个百分点，且正常样本污染率低于1%。

**⚠️ 局限性**

局限在于仅在WM-811K数据集上验证，缺乏跨领域或更大规模数据的评估；方法依赖手工设计的空间特征，对高度复杂或非空间性缺陷的适应性有限；同时阈值与候选参数需经验调优。

---

## 134. Minimax-Optimal Predictive Flow Matching under Distribution Drift

**arXiv ID:** 2610.28898 | [PDF](https://arxiv.org/pdf/2610.28898v1)

**作者:** Aadithya Srikanth `[一作]` (Purdue University), Vaneet Aggarwal `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出了DyProd-FM算法，在流匹配的非平稳环境中实现预测式学习。

**💡 创新点**

创新点是结合dyadic历史前缀重训练与睡眠Prod专家，获得最优动态失调率，并利用二阶转移和基平滑实现对目标律变化的高效补偿。

**🔧 技术方法**

采用流匹配回归、贝塞尔估计、Prod专家选取、二阶风险传递等技术。

**📊 数据集**

主要是理论分析，未使用具体公开数据集。

**📈 对比分析**

与下界对比，证明在相对风险变动和Wasserstein变动下可达O(√(TΩ))和O(T^{1/3}V_μ^{2/3})，达到下界的最优阶。

**⚠️ 局限性**

局限在于缺少对训练效率的证明、对非仿射插值的支持有限，且对均匀流时间的分析被排除。

---

## 135. How Far Can Foundation Models Be Distilled for Efficient Jet Tagging?

**arXiv ID:** 2610.29439 | [PDF](https://arxiv.org/pdf/2610.29439v1)

**作者:** Jay Agarwal `[一作]` (BITS Pilani), Dhruv Kumar `[通讯]` (BITS Pilani)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了知识蒸馏在Jet标记中的效果，使用六个教师模型和五个学生模型，并在宽度和深度上扩展学生尺寸。

**💡 创新点**

系统评估教师属性对蒸馏收益的影响，证明教师强度决定大部分收益，展示了在极小学生（2k参数）上超越更大状态的标记器。

**🔧 技术方法**

采用Transformer、LorentzNet、ParT、MIParT、L‑GATr‑slim等架构，使用标准蒸馏损失(α=0.7,T=2)，并测量GPU/CPU推理成本。

**📊 数据集**

使用Top‑Tagging基准数据集（1.2M训练，0.4M验证，0.4M测试，粒子数≤200）。

**📈 对比分析**

通过AUC、背景抑制1/ε_B、操作数、吞吐量和延迟等指标与教师和已发布模型比较，证明20k参数的蒸馏L‑GATr‑slim在性能上超越所有从零训练的标记器并大幅降低推理成本。

**⚠️ 局限性**

限制包括仅在单一二分类基准上评估、蒸馏仍受教师-学生容量差距限制、未针对硬件优化、未验证更大基础模型或多任务场景。

---

## 136. Don't Stop Long-Context Continual Pre-Training Too Early: An Empirical Scaling Law

**arXiv ID:** 2610.29352 | [PDF](https://arxiv.org/pdf/2610.29352v1)

**作者:** Ziqing Qiao `[一作]` (Tsinghua University), Xu Han `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究长上下文持续预训练（LCCP）的规模化效益，并提出可持续监测与验证方法

**💡 创新点**

提出了基于模型大小、短上下文预训练与长上下文训练令牌的加性幂律缩放法则，并设计 Constant‑Last4 训练与观察协议，实现在长上下文预训练过程中实现可扩展且稳定的性能预测

**🔧 技术方法**

实验采用多尺度（31M–1.22B参数）模型，采用短上下文4K、长上下文32K序列，使用线性衰减与恒定学习率两种调度，评估指标包括 LongPPL、RULER、LongBench、GraphWalks、LongReason、BABILong、GAIA、BFCL-WebSearch、τ^2-telecom；使用 Llama‑3.1‑8B 作为参考模型进行 LongPPL 计算

**📊 数据集**

GovReport 数据集用于 LongPPL 评估；RULER 子任务、LongBench、GraphWalks、LongReason、BABILong、GAIA、BFCL-WebSearch、τ^2-telecom 作为下游任务评估集

**📈 对比分析**

与线性衰减终点训练模型对比，Constant‑Last4 在 LongPPL、ShortPPL 上与线性衰减模型保持一致且更为稳定；在 32K LCCP 轨迹上，LongPPL 持续下降，RULER 分数随长上下文训练令牌增长而提升，且在 128K 扩展与监督微调后继续保持提升，说明缩放法则能够准确预测下游性能，性能提升随 LCCP 预算增加而稳步上升

**⚠️ 局限性**

仅在 1.22B 规模下验证，未探索更大模型或更高长上下文预算的极限；模型训练成本高，环境与算力消耗大；下游任务未覆盖所有潜在风险场景，未对安全性做完整评估

---

## 137. CCG Diffusion: Combinatory Categorial Grammar-Driven Compositional and Syntactically Disambiguated Text-to-Image Generation

**arXiv ID:** 2610.29415 | [PDF](https://arxiv.org/pdf/2610.29415v1)

**作者:** Ryosuke Yamaki `[一作]` (Ritsumeikan University), Tadahiro Taniguchi `[通讯]` (Kyoto University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种无训练的文本到图像生成方法CCG Diffusion，利用Combinatory Categorial Grammar (CCG) 解析获取句法结构来指导图像生成，以解决属性绑定、空间关系、计数误差以及句法歧义（尤其是 PP 附着歧义）等问题。

**💡 创新点**

创新点在于：1) 引入 CCG 解析直接读取词汇的修饰关系和 PP 附着位置，避免依赖传统的短语或依赖句法解析；2) 通过两条路径（prompt 重写与 attention 引导）将句法信息传递给预训练扩散模型；3) 结合选择性引导增强、位置前缀与区域分配损失、空间关系损失和计数损失，系统性提升对多对象多属性场景的生成质量。

**🔧 技术方法**

主要技术包括：CCG 解析（Hol-CCG），扩散模型的训练无关 attention 引导（Predicated Diffusion 的延伸），prompt 重写（位置前缀与歧义消解改写），以及多种自定义损失（区域分配、空间关系、计数）。

**📊 数据集**

使用了两个公开基准数据集：T2I-CompBench++（用于多对象属性绑定、空间关系、计数等四大子任务）和 Text-to-Image Ambiguity Benchmark (TAB) 的 PP 附着歧义子集（评估歧义消解能力）。

**📈 对比分析**

与六种无训练对比方法（Composable Diffusion、StructureDiffusion、Attend‑and‑Excite、SynGen、Predicated Diffusion、以及 vanilla）在 Stable Diffusion 与 Sana 两大模型上进行公平比较。结果显示，CCG Diffusion 在属性绑定、二维空间关系、计数等目标子任务上均显著优于对手，TAB 歧义消解任务中也实现了最高的“Yes率”，并在两种 backbone 上均保持优势。

**⚠️ 局限性**

局限性包括：1) 对深度关系、动作和多重约束场景的控制能力有限；2) 仅解决 PP 附着歧义，其他类型歧义（如并列范围）仍需进一步研究；3) 由于梯度引导和 CCG 解析，生成时间约为原始模型的 2.2–2.4 倍，计算成本较高。

---

## 138. Progressive Learning from Content to Relevance to Personalization for CTR Prediction

**arXiv ID:** 2610.29138 | [PDF](https://arxiv.org/pdf/2610.29138v1)

**作者:** Zhiguang Yang `[一作]` (Alibaba Group), Dan Ou `[通讯]` (Alibaba Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种三阶段逐步学习框架，将多模态内容以细粒度语义ID（SID）形式离散化并融入ID‑based CTR 预测；

**💡 创新点**

创新点在于利用多尺度视觉分词器和字段特定文本分词，保持细粒度语义信息，并通过语义预训练和个性化后训练三阶段流程，构建共享特征交互网络，实现内容与稀疏ID特征的无缝融合；

**🔧 技术方法**

使用VAR多尺度视觉分词器、BPE文本分词、HiFormer风格的稀疏槽注意力、全参数语义预训练与软标签相关性学习以及最终的全参数CTR微调；

**📊 数据集**

在工业化的淘宝搜索日志（2026年8–9月）和公开MMCTR基准数据集上进行实验；

**📈 对比分析**

与HiFormer基线及多种S2S内容嵌入、可视化ID、RQ‑VAE SID、COINS、PRECTR等方法对比，淘宝线上/线下实验中AUC提升0.25pt、GAUC提升0.26pt、NSCTR提升0.75%，MMCTR AUC提升0.77pt，在线A/B测试GMV提升0.85%；

**⚠️ 局限性**

局限性包括分词器需离线预处理，未实现端到端语义分词优化；对极低延迟检索阶段的计算开销未完全评估；对其他检索或转化率任务的泛化仍待验证。

---

## 139. Beyond Shortcut Learning: Hierarchical Spatial Prototype Regularization for Face Forgery Detection

**arXiv ID:** 2610.29442 | [PDF](https://arxiv.org/pdf/2610.29442v1)

**作者:** Chao Shuai `[一作]` (Zhejiang University), Zhenguang Liu `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并解决深度伪造检测中出现的“取证表示崩溃”问题，提出 Hierarchical Spatial Prototype (HSP) 规约模块来构建结构化的取证子空间。

**💡 创新点**

创新点在于：①将取证信息建模为多维原型子空间；②通过原型正交化和二阶分配平衡约束保证原型的几何独立性和子空间的高有效秩；③将 HSP 设计为轻量化、可插拔的模块，可直接集成到多种 ViT 基础模型和现有检测器中。

**🔧 技术方法**

主要技术包括：基于 CLIP ViT-L/14 的视觉预训练、温度缩放软分配、双向重构损失、原型正交正则化、分配平衡正则化、有效秩分析与 PCA 评估。

**📊 数据集**

使用了 FaceForensics++（训练），并在 Celeb-DF-v2、DFDC、DFDCP 等公共数据集进行跨数据集评估；在 DF40 的六种面部替换方法上进行跨操纵评估。

**📈 对比分析**

与现有 20+ 先进检测器（如 Effort、VbSaT、χ²-DFD 等）对比，HSP-CLIP 在视频级平均 AUC 95.3%（+1.8%）和帧级平均 AUC 90.8%（+5.6%）上取得了显著提升，跨数据集平均提升 5.6%，跨操纵平均提升 5.3%。

**⚠️ 局限性**

局限性：①主要针对人脸伪造任务，其他模态或生成方式的泛化性尚未充分验证；②需要在不同模型和数据规模下调优原型数 K 与正则化系数；③在极端对抗攻击或高分辨率视频上性能尚未深入评估。

---

## 140. Nominate-then-Adjudicate: LLM-Assisted One-Pass and Per-Instance MIP Solver Tuning

**arXiv ID:** 2610.29391 | [PDF](https://arxiv.org/pdf/2610.29391v1)

**作者:** Yifan Zhao `[一作]` (Alibaba Group), Yuming Deng `[通讯]` (Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了Nominate‑then‑Adjudicate流水线，用固定随机候选集与一次性基础模型调用，在不需要求解器调用的前提下完成MIP实例的参数配置选择。

**💡 创新点**

创新点在于：①把召回与结构感知重排分离，使用20维静态特征召回固定候选；②通过压缩文本矩阵与文本记忆让固定权重LLM完成结构感知的一次性k‑way决策，避免在线求解与模型微调。

**🔧 技术方法**

技术组合包括XGBoost经验性能模型(EPM)、随机采样固定配置池、压缩文本矩阵渲染、Qwen‑Max等大语言模型、文本经验记忆自我对弈、静态特征工程等。

**📊 数据集**

使用16个自制MIP生成器族（约1600实例）进行离线EPM训练和评估，并在两份MIPLIB 2017实例上验证迁移能力。

**📈 对比分析**

在“one‑pass、zero‑solve”协议下，与默认、ISAC+、ISAC+SMAC、ISAC+ROAR等基线对比，指标为得分、SGM加速、时间节省和胜率；实验显示相比默认提升约90%胜率，平均得分比ISAC+高+0.02，且在两MIPLIB实例上快约2–3倍。

**⚠️ 局限性**

局限性包括：①仅在固定随机候选集内搜索，覆盖面未知；②未测量端到端延迟与大规模部署成本；③k‑way决策受限，无法保证最优；④无法解决所有结构冲突或特征歧义问题。

---

## 141. Where Should a Stuck Agent Spend Its Next Tokens? Causal Attribution of Inference-Time Resources

**arXiv ID:** 2610.28834 | [PDF](https://arxiv.org/pdf/2610.28834v1)

**作者:** Saiyue Lyu `[一作]` (University of British Columbia), Haibo Zhang `[通讯]` (Kyushu Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种控制实验框架BRA，用来在冻结的失败检查点上评估计算、信息、反馈三种推理时资源对代码修复的贡献。

**💡 创新点**

创新点是将推理资源视为合作博弈中的玩家，利用Banzhaf值与交互来量化各资源的边际效益和相互作用，从而揭示资源选择与修复成功率的差距。

**🔧 技术方法**

采用全因子设计、Banzhaf价值与互交分析、Möbius分解以及对不同模型的单次修复实验等技术。

**📊 数据集**

使用MBPP+、HumanEval+（函数级）和SWE‑bench Verified（仓库级）共计194+126个失败检查点。

**📈 对比分析**

通过对每个模型在相同失败状态下执行八种资源组合并计算平均提升，发现信息资源贡献最大，反馈几乎无效，计算在强模型中几乎无用；自我资源选择实验显示模型往往偏好无效的反馈。

**⚠️ 局限性**

局限性包括只考虑三种资源，评估仅限一次修复，未覆盖多步迭代或其他资源类型，以及模型选择偏向大型预训练模型，可能限制普适性。

---

## 142. Two for One: Privileged Learning for Resource-Constrained Screening

**arXiv ID:** 2610.29126 | [PDF](https://arxiv.org/pdf/2610.29126v1)

**作者:** Lisa Y. W. Tang `[一作]` `[通讯]`, Lisa Y. W. Tang

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种教师-学生框架，利用训练阶段双眼彩色眼底照片的双眼信息（privileged information）来指导仅使用单眼图像的学生模型，实现推理时只需单眼输入。

**💡 创新点**

创新点在于将双眼训练数据的互补信息通过知识蒸馏传递给单眼模型，使模型在保持单眼部署灵活性的同时，获得双眼信息带来的性能提升。

**🔧 技术方法**

技术实现上采用 Vision Transformer 基础模型 RETFound‑Green，使用二元交叉熵分类损失与特征层均方误差蒸馏损失联合训练。

**📊 数据集**

实验使用公开的 Ocular Disease Intelligent Recognition (ODIR‑5K) 数据集，包含年龄相关黄斑变性（AMD）与糖尿病视网膜病变（DR）两项检测任务。

**📈 对比分析**

与仅单眼训练的基线相比，学生模型在左眼 AUROC 和平衡准确率等指标均有显著提升（如 AMD 左眼 AUROC 从 0.931 提升至 0.966），右眼提升有限。整体表现显示该框架在提升召回率和AUROC方面具有优势，但在精度与 FPR 上存在一定权衡。

**⚠️ 局限性**

局限性包括测试集样本量相对较小、缺乏跨机构外部验证、以及提升性能伴随 FPR/精度的折中，需要进一步验证模型在不同人群、设备与临床工作流中的鲁棒性。

---

## 143. Make it Walk: Computational Design of LEGO Walkers

**arXiv ID:** 2610.29505 | [PDF](https://arxiv.org/pdf/2610.29505v1)

**作者:** Siqi Li `[一作]` (Singapore University of Technology and Design), Peng Song `[通讯]` (Singapore University of Technology and Design)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过计算方法设计并制造了一种能够实现多种步态、保持静态稳定且可定制外观的LEGO步行机器人。

**💡 创新点**

创新点在于：①将腿的运动轨迹机械化，使用非圆齿轮与五杆连杆实现单驱动双输入的轨迹生成；②在腿级别采用独立电机并通过同步控制实现步态时序解耦；③提出一种基于目标步态曲线和身体框架的协同优化框架，能够同时设计机械结构、腿布局与控制信号。

**🔧 技术方法**

主要技术包括：非圆齿轮设计与3D打印、五杆连杆机械建模、离散Fréchet距离评估、静态稳定性分析、Python+PyTorch优化求解、机械仿真与物理组装。

**📊 数据集**

未使用公开数据集；输入为用户指定的目标步态曲线（foot locus）和身体框架模型，优化产生对应的机械和控制方案。

**📈 对比分析**

与传统Jansen连杆和Coros等方案比较，所设计的腿机构在部件数、总梁长上更少、轨迹逼近更精确（Fréchet距离显著降低），并在物理原型上实现了稳定行走，展示了更高的设计自由度与制造可行性。

**⚠️ 局限性**

限制包括：仅支持静态稳定的低速步态，无法实现动态稳定或复杂地形行走；所有腿采用同一机构，缺乏多样化腿设计；需要较精确的驱动控制与齿轮加工，限制了大规模快速生产。

---

## 144. Same Probabilities, Different Advice: Gain-Loss Framing Effects in High-Stakes LLM Decision Support

**arXiv ID:** 2610.29294 | [PDF](https://arxiv.org/pdf/2610.29294v1)

**作者:** Yuyao Zhang `[一作]` (Hong Kong University of Science and Technology), Sirui Han `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并评估了FrameBench‑HPS基准，探究高风险决策中概率等价的收益/损失框架对LLM推荐的影响。

**💡 创新点**

提出高风险场景控制基准，证实约32%推荐会因框架变化而变动，且通过概率规范化可显著减少此类不一致。

**🔧 技术方法**

采用提示工程、概率规范化、框架鲁棒提示等技术，对7款LLM进行温度0下的JSON输出评估。

**📊 数据集**

使用240个高风险场景（医疗、金融、教育/职业、法律/消费者）构成的2,880个框架化提示。

**📈 对比分析**

通过对比不同框架下的推荐类别、风险、可操作性等指标，并与相同框架的重复生成对比，发现框架效应远大于生成波动；概率规范化将不一致率降至≈7%。

**⚠️ 局限性**

局限在于基准为合成场景、概率水平有限、未评估决策质量，仅衡量一致性而非正确性。

---

## 145. Hard and Soft Real-Time DNN Inference Scheduling on Heterogeneous Multi-Accelerator Systems via Reinforcement Learning

**arXiv ID:** 2610.29503 | [PDF](https://arxiv.org/pdf/2610.29503v1)

**作者:** Francesco Giulio Blanco `[一作]` (University of Catania), Maurizio Palesi `[通讯]` (University of Catania)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对异构多加速器系统，提出一种结合在线强化学习调度器与确定性运行时安全层的框架，能够同时满足硬实时（HRT）和软实时（SRT）DNN推理任务的时间约束与质量服务需求。

**💡 创新点**

核心创新包括：①基于机器级切割的可调度前缀替换判据，保证在任意学习策略下仍能保持HRT可调度；②将安全层与调度器解耦，形成与任何在线调度策略兼容的安全包；③在RL策略中显式加入HRT任务标识和大负奖励，使学习过程天然关注HRT约束。

**🔧 技术方法**

采用深度确定性策略梯度（DDPG）强化学习实现子任务优先级与加速器分配的连续决策；离线构建最优的基于EDF/FF的HRT参考调度；运行时安全层使用机器切割对候选前缀进行快速可调度性检验；平台层面利用预先分析的WCET与数据传输成本。

**📊 数据集**

实验数据集包括：ImageNet上的AlexNet、InceptionV3、ResNet50；YOLO‑v2目标检测网络；每层级的DAG图与多加速器执行时间由真实的行/权重驻留加速器（Eyeriss/Simba）进行功耗与时延测量；软实时到达模式基于Da Costa等人的Pareto分布生成的Google‑style工作负载。

**📈 对比分析**

在保持硬实时可调度的前提下，将所提框架与PREMA、MAGMA、Tang‑et‑al、SSML、CD‑MSA、RL‑aware等多种在线调度器及其安全包版本进行对比。结果显示：在中等压力（40 ms到达间隔）下SRT完成率可达98.3%，在高压力（10 ms）下仍保持81.4%；worst‑client SLO率提升至62%；相较于基线至少提升20–30个百分点；fallback率低于10%，表明学习策略能频繁生成可认证前缀。

**⚠️ 局限性**

局限性包括：①运行时安全层会导致一定的调度灵活性下降；②对WCET与加速器时延的精确预估有较高依赖；③仅验证了非抢占式SRT任务，未考虑SRT的部分完成与恢复；④实验规模限定在六个加速器，未评估更大规模或更异构的系统；⑤学习策略对调度周期敏感，需额外训练或迁移学习以适配不同T_S。

---

## 146. Sparsity Should Be ALLOTted by Output: Training-Free Allocation of Block-Sparse Attention for Video Diffusion Transformers

**arXiv ID:** 2610.29441 | [PDF](https://arxiv.org/pdf/2610.29441v1)

**作者:** Tianze Xia `[一作]` (University of Chinese Academy of Sciences), Shiming Xiang `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种无训练的稀疏注意力分配方法ALLOT，针对视频扩散变换器在不同层和步骤的稀疏度进行最优分配，从而在保持或缩短壁钟时间的前提下显著提升视频生成质量。

**💡 创新点**

创新点在于引入单步细胞探测与轨迹校准两阶段误差估计，并结合延迟约束下的全局最优稀疏度表，取代传统经验规则或局部指标。

**🔧 技术方法**

利用一阶误差传播分析、单步误差探测、窗口轨迹校准、Lagrangian约束求解以及现有稀疏注意力核实现。

**📊 数据集**

在Wan2.1、Wan2.2、HunyuanVideo等模型上，以VBench 32条提示视频为基准，主要使用PSNR、SSIM、LPIPS等指标。

**📈 对比分析**

与多种基线（均匀稀疏、手工规则、局部指标驱动等）以及官方发布的稀疏注意力管线在相同或更短壁钟时间下对比，ALLOT在PSNR上提升1.5–11.8 dB，几乎全部场景胜过或等价于默认配置。

**⚠️ 局限性**

需要离线校准并重新测量不同稀疏策略，且对层间相互作用建模有限，无法处理自适应阈值机制。

---

## 147. Beyond Small Talk: Modeling Disclosure Trajectories in Human-AI Conversation

**arXiv ID:** 2610.29413 | [PDF](https://arxiv.org/pdf/2610.29413v1)

**作者:** Keith Kwok `[一作]` (Nanyang Technological University), Soujanya Poria `[通讯]` (Nanyang Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Trajectory-based Disclosure Evaluation（TraDE），通过构造预设的披露轨迹，在保持当前披露深度不变的情况下评估模型对先前披露历史的行为依赖。

**💡 创新点**

创新点在于将社会渗透理论与对话生成相结合，系统控制披露轨迹以揭示模型的历史依赖性，并首次量化不同披露历史对关系行为的影响。

**🔧 技术方法**

利用大型语言模型（如Qwen3.6、Qwen3、Phi‑4、Gemma‑3、GPT‑5.6等）生成对话，使用Phi‑4进行披露深度评估，结合AnthroBench、INTIMA、ELEPHANT三个行为基准进行多轮评估，并通过DTW对人类ESConv对话进行轨迹匹配。

**📊 数据集**

采用自定义生成对话数据（基于Qwen3.6构建的多轨迹会话）以及公开的情感支持语料库ESConv中的人类对话作为基准。

**📈 对比分析**

通过配对Wilcoxon、Friedman、Bootstrap和多重回归等统计方法比较模型与人类在不同披露轨迹下的行为差异；实验表明模型在不同轨迹下的行为显著不同，且部分差异在披露回浅后仍可持续；与人类支持者相比，模型在整体行为分数及轨迹形状上存在显著差距。

**⚠️ 局限性**

局限性包括：仅在单一会话内考察短期持续性，未独立操纵披露深度、持续时间或回顾性影响；缺乏跨会话或更长时序的历史效应验证；评估范围受限于预设轨迹和现有基准，未涵盖更广泛的对话情境。

---

## 148. Reconstruction differences rival individual differences in the fly compass connectome and break trained circuit models

**arXiv ID:** 2610.29205 | [PDF](https://arxiv.org/pdf/2610.29205v1)

**作者:** Md Ali Ashraf Yad `[一作]` (Wrocław University of Science and Technology), Nazifa Tabassum Nabila `[通讯]` (Comilla University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对果蝇头向电路进行九种版本的构建与训练，并评估其在不同重建版本间的可迁移性和鲁棒性。

**💡 创新点**

首次将同一飞虫不同重建步骤产生的差异与不同个体之间的差异统一量化，发现重建误差占差异的一大部分；揭示训练出的连续吸引子网络极易失效（1%权重变动导致误差从 <1° 提升至 10–30°），并证明噪声训练能在一定程度上提升跨重建迁移，但仍未达到生物学水平。

**🔧 技术方法**

使用基于回归的 rate 网络（双步更新、ReLU、可训练增益）、梯度下降、噪声训练、细胞类型增益拟合、以及网络移植（Neuron‑to‑Neuron 对齐）等技术；并利用多种距离度量（权重变动、方向张力谱、Jaccard 等）和持久同调、主成分分析等机制评估。

**📊 数据集**

数据集包括四个不同飞虫的全脑/半脑 connectomes（male CNS、hemibrain、FlyWire、BANC）及其内部的同一飞虫的不同阈值/检测/校对版本，配合 DANDI 000289 的 calcium 成像记录。

**📈 对比分析**

通过对比自身 wiring 的误差（<1°）与跨 wiring 的误差（82–92°）以及归一化后与随机模型（90°）的 transplant ratio，验证了模型在不同 wiring 下的不可迁移性；使用噪声训练和细胞类型增益重拟合后，误差下降至 50–60°，但仍高于自身 wiring；同时通过“距离–误差”相关分析证明迁移失败与 wiring 距离相关，且不同重建步骤（检测、校对）产生的差异与个体差异在同一尺度。

**⚠️ 局限性**

主要限制包括：仅对两只飞虫做了同一 fly 的多版本重建；使用的模型仅为 rate 网络，未测试 spiking、角度积分或横向输入；缺乏对短时漂移的定量比较；迁移实验未覆盖所有可能的映射误差；并且模型对权重微调高度敏感，难以与实际生物电路的鲁棒性相匹配。

---

## 149. From Linear Exact Repair to Mixed-Alphabet Codes: The Bandwidth-Subpacketization Tradeoff of MDS Array Codes

**arXiv ID:** 2610.29144 | [PDF](https://arxiv.org/pdf/2610.29144v1)

**作者:** Zihao Zhang `[一作]`, Sihuang Hu `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究在固定子分块层数下，MDS数组码的集中线性精确修复问题，提出将任意修复方案映射为两类混合字母码（零空间码与行空间码），从而把修复带宽下界转化为码的大小和距离约束；并给出基于Singleton、Hamming、Plotkin、Griesmer等经典编码理论的显式有限参数下界。

**💡 创新点**

创新点在于：1）首次通过辅助混合字母码统一描述修复带宽与子分块的权衡；2）从这一描述出发，利用经典编码极限直接得到多种显式下界；3）提出精确类型压缩的线性规划算法，将原本指数规模的支持枚举LP压缩为多项式规模，实现对固定子分块的可计算下界。

**🔧 技术方法**

主要技术包括：线性修复矩阵的核与行空间分析；MDS阵列码的奇偶校验结构；混合字母码的支持枚举与 Delsarte 线性规划；类型压缩对称性处理；以及组合论中的 Singleton、Hamming、Plotkin、Griesmer 边界。

**📊 数据集**

该工作不依赖特定数据集，而是针对任意给定参数 (n,k,ℓ,q,h,d) 的 MDS 数组码进行理论分析与算法求解；数值实验中使用了小规模参数实例（如 h=1,r=3,ℓ=2,q=5 与 h=1,r=4,ℓ=2,q=2^8）来验证算法效果。

**📈 对比分析**

与已知的闭式下界（如 cut-set、Hamming、Plotkin）以及现有构造（如传统 MDS、LESS 构造）比较，类型压缩 LP 下界在小至中等参数范围内明显优于闭式下界，并在多数情况下低于构造所需的修复带宽，但仍与最优构造存在一定差距。

**⚠️ 局限性**

局限性包括：1）仅给出局部修复（单一失效集合与帮助集合）下的下界，尚未直接推广至全局最优修复；2）对大子分块层数 ℓ 的计算复杂度仍随 ℓ 增大呈指数增长；3）所得到的下界虽可计算，但在极大参数下仍可能与实际构造存在明显差距。

---

## 150. Peer review should constrain evaluative authority

**arXiv ID:** 2610.28646 | [PDF](https://arxiv.org/pdf/2610.28646v1)

**作者:** Hanrui Wang `[一作]` (National Institute of Informatics), Isao Echizen `[通讯]` (National Institute of Informatics)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并探讨了在缺失评估准则真值下，通过约束审稿判断的权威性来改进同行评审的制度设计；对大型计算机科学会议的应用进行了示例化实现。

**💡 创新点**

创新点在于将改革目标从提升评审者质量转向限制评审决策的影响力，提出四项权威约束原则，并设计了双轨人机互审的原型。

**🔧 技术方法**

采用结构化审查机制、人机协同监督、AI与人类评审的角色分离，以及对结论性评估进行二次检查。

**📊 数据集**

实验使用了真实边缘论文的四份原始评审、四名人工监督者、ChatGPT‑5.1，未使用公开数据集。

**📈 对比分析**

对比方法为对照传统单轨审稿流程；在原型实验中评审者认为 AI 监督比完整评审更轻松，但未给出量化性能指标。

**⚠️ 局限性**

局限性包括缺乏大规模实证验证、工作量与决策者负担不确定、模型对评审标准的依赖性、以及对不同学科的适用性未评估。

---

## 151. Query Feedback and Teaching for Stable and Certifiable Language Generation

**arXiv ID:** 2610.28918 | [PDF](https://arxiv.org/pdf/2610.28918v1)

**作者:** Farnam Mansouri `[一作]` (University of Waterloo), Sandra Zilles `[通讯]` (University of Regina and Alberta Machine Intelligence Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在语言生成在极限（generation in the limit）的框架下提出了“可证性生成”（certifiable generation）概念，并与之前的“稳定生成”（stable generation）进行比较，给出了这两种生成模式的有限证据表征；进一步，作者构建了教练集（teaching set）与教学维度（teaching dimension）等工具，揭示不同生成成功标准下所需教练集大小的差异；最后给出若干模式之间的严格层级关系，并通过模式族（如擦除模式语言）证明这些分离。

**💡 创新点**

创新点在于：①首次定义并刻画可证性生成，证明其严格强于稳定生成；②给出稳定生成与可证性生成的有限证据表征（类似告诉牌）；③引入教学维度框架，比较生成、稳定生成与可证性生成的教练集需求；④构造模式族展示理论分离；⑤在查询反馈模型下阐明稳定生成与可证性生成的关系。

**🔧 技术方法**

使用理论分析与构造证明，借鉴学习理论中的“tell‑tale assignment”“non‑clashing assignment”与“teaching set”概念，提出稳定/可证性生成的有限证据表征；在查询反馈模型中利用正负证据集构造判定条件；通过构造具体语言族（如擦除模式语言、正则语言集合）演示层级与分离。

**📊 数据集**

无实验数据集；本工作为纯理论研究，所有结论均来自构造证明与逻辑推导。

**📈 对比分析**

通过层级图与分离定理比较不同生成模式的能力，展示可证性生成在保证生成正确性的同时仍比传统生成更强；教学维度的比较表明，尽管可证性生成在某些族上只需极小教练集，传统统一生成或识别在同族上可能需要指数级或无穷多教练集。

**⚠️ 局限性**

局限性：未考虑计算复杂度或资源受限情形；未处理噪声或对抗样本；所有结果均基于无效性假设（不要求可计算性），在实际机器学习模型中如何实现及验证仍待进一步研究。

---

## 152. Real-Time Ground Normal Vector Estimation for Spherical Mobile 3D Mapping using LiDAR-Inertial Odometry

**arXiv ID:** 2610.28701 | [PDF](https://arxiv.org/pdf/2610.28701v1)

**作者:** Tim Schubert `[一作]` (Julius-Maximilians-Universität Würzburg), Andreas Nüchter `[通讯]` (Julius-Maximilians-Universität Würzburg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并评估了基于RANSAC的地面法向量估计方法，包含单扫描和全局地图两种实现，并将估计结果用于改进球形移动测绘系统的姿态估计。

**💡 创新点**

创新点在于：1）在单扫描基础上加入可视性和内点评分与回退机制；2）使用全局LIO地图进行地面提取，显著提升精度与鲁棒性；3）将法向量直接融入运动学模型提升姿态精度。

**🔧 技术方法**

技术手段包括：RANSAC‑PCA平面拟合、Gaussian卷积平滑、可视性评分、内点评分、角度/特征验证、LIO地图构建与姿态估计、IMU‑LiDAR融合姿态滤波。

**📊 数据集**

使用真实场景数据集：户外草地山坡和Zugspitze冰川隧道，结合RIEGL VZ‑400 TLS作为地面真值。

**📈 对比分析**

通过中位角误差、RMSE、P95等指标比较，静态半径全局方法在山坡场景下MedAE 3.1°/RMSE 4.6°，相较单扫描法MedAE 10.9°/RMSE 15.3°；在隧道场景下仍优于单扫描。时延在3–15 ms内，满足实时要求。

**⚠️ 局限性**

局限性：对LIO地图质量高度依赖，地图漂移导致法向误差；对快速旋转下的可视性仍有挑战；未在多机器人或动态障碍环境中验证；算法对不同传感器参数需要手动调参。

---

## 153. Flat-RLVR: Sharpness-Aware Reinforcement Learning with Verifiable Rewards for Reasoning Capacity in LLMs

**arXiv ID:** 2610.29281 | [PDF](https://arxiv.org/pdf/2610.29281v1)

**作者:** Yeonbum Kim `[一作]` (Ulsan National Institute of Science and Technology), Sung Whan Yoon `[通讯]` (Ulsan National Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在LLM后训练中加入Sharpness‑Aware Minimization（SAM），通过在Group‑Relative RLVR框架中搜索平坦奖励区域，缓解分布锐化问题，提升采样多样性并保持单次推理准确率。

**💡 创新点**

将SAM与GRPO/DAPO/CISPO等RLVR变体结合，提出‑RLVR框架，以优化几何曲率而非仅靠奖励放大，实现更平滑的输出分布；同时在训练中使用周期性SAM、梯度中心化、prompt‑difficulty 适应的扰动半径等技术。

**🔧 技术方法**

核心技术包括：Group‑Relative Policy Optimization (GRPO)、Sharpness‑Aware Minimization (SAM) 及其变体（ASAM、GCSAM、LookSAM、Momentum‑SAM、GSAM）、梯度中心化、Prompt‑Difficulty‑Aware 调度、离散采样组大小等。

**📊 数据集**

主要数据集为数学推理领域的 MATH（7.5k题）、AMC23、AIME24‑26 等；在代码生成任务上使用 MBPP 数据集进行交叉域验证。

**📈 对比分析**

与基线 GRPO、F‑GRPO 以及 DAPO/CISPO 变体进行对比，结果显示‑RLVR 在 Pass@256 上平均提升 2–3%（在 Qwen2.5‑Math‑1.5B、Llama‑3.2‑1B‑Instruct 上），单次推理 Pass@1 保持不变或略有提升；在更大模型 Qwen2.5‑3B‑Instruct 上同样取得 Pass@256 +2% 的收益，且对比 GRPO 的计算开销与显存消耗几乎无差异。

**⚠️ 局限性**

局限性包括：需要调优 SAM 周期和扰动半径等超参数；对极大模型的规模扩展仍需验证；目前仅在数学推理和部分代码生成任务上测试，尚未系统评估在更广泛领域的泛化能力。

---

## 154. Fragment-Level Graph Encoding for Interpretable Molecular Property Prediction

**arXiv ID:** 2610.29589 | [PDF](https://arxiv.org/pdf/2610.29589v1)

**作者:** Rafael Lopes Almeida `[一作]` (University of Eastern Finland), Ina Pöhner `[通讯]` (University of Eastern Finland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过BRICS分解将分子图从原子节点转换为化学可合成的片段节点，保持相同的GNN架构和解释器，得到以功能基为单位的可解释性图谱；

**💡 创新点**

创新点在于仅改变输入表征而非模型或解释方法，能够在不降低预测精度的前提下，将传统基于原子级的热图升级为以片段为粒度的解释；

**🔧 技术方法**

使用BRICS分解、图神经网络（AttentiveFP、GAT/GATv2、PNA、GIN）、梯度解释器Integrated Gradients、基于掩码的GNNExplainer；

**📊 数据集**

实验基于四个任务：BACE（分类/回归）、BBBP（分类）、PTR1（回归）以及四个不同的GNN架构；

**📈 对比分析**

在Murcko scaffold拆分下对比原子级图与片段级图，评估预测指标（MCC/CCC）和解释指标（片段命中率、Gini系数、解释器一致性）。片段级编码在大多数任务上保持或略优预测性能，且IG在BACE/PTR1上的片段命中率分别提升约1.5×和3.2×；

**⚠️ 局限性**

限制包括：BRICS无法分解所有分子（小/刚性分子被排除），分解可能破坏关键药效团导致命中上限受限；聚合基于原子级解释的结果受两种约定影响，稳定性不佳；评估受样本稀缺影响，且对不同解释器的依赖性较大。

---

## 155. A Large-Scale Evaluation of Role-Enhanced Zero-Shot Chain-of-Thought Prompting

**arXiv ID:** 2610.29073 | [PDF](https://arxiv.org/pdf/2610.29073v1)

**作者:** Ibrahim Khan `[一作]` (Ritsumeikan University), Ruck Thawonmas `[通讯]` (Ritsumeikan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种Role-Enhanced Zero-Shot Chain-of-Thought（RE-ZS-CoT）框架，利用领域专家角色在单轮零样本推理中引导LLM的思考过程。

**💡 创新点**

创新点在于将角色提示与零样本CoT相结合，提供静态预定义角色与动态自适应角色两种选择，并系统评估角色在不同任务中的位置（前置、后置、拆分）及其对性能的影响。

**🔧 技术方法**

技术手段包括：基于文本提示的角色嵌入、单轮零样本CoT、模型自我对齐（static和dynamic两种模式）、多种提示结构（R→Q、Q→R、R→Q→R）。

**📊 数据集**

使用七个公开基准：算术推理（AQuA-RAT、GSM8K）、常识推理（OpenBookQA、StrategyQA）以及阅读理解（RACE-m、RACE-h、TriviaQA），并在20个不同参数规模的LLM上进行评测。

**📈 对比分析**

与传统零样本与零样本-CoT基线对比，RE-ZS-CoT在多数模型和基准上提升12%–44%准确率；静态角色在算术任务上表现最优，动态角色在阅读与常识任务上更具竞争力；实验显示角色与提示位置的匹配度对性能影响显著。

**⚠️ 局限性**

局限性包括：缺乏对LLM训练数据和内部机制的透明度，未进行深入的错误或可解释性分析，统计显著性检验不足；仅针对单轮、单语言、文本推理的自回归模型，未评估多模态或对话场景；以及在某些常识任务上效果不佳，需进一步研究角色与任务不匹配导致的性能下降。

---

## 156. Accelerating CPU Speculative Decoding via Unified Graph Execution and Transactional KV Isolation

**arXiv ID:** 2610.28671 | [PDF](https://arxiv.org/pdf/2610.28671v1)

**作者:** Zhangdie Yuan `[一作]` `[通讯]` (University of Cambridge), Zhangdie Yuan (University of Cambridge)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过统一图缓存和事务性 KV 缓存隔离，在 CPU 上实现高效的推测式解码。

**💡 创新点**

创新点在于将草稿与目标模型编译到同一内存场景并使用零拷贝写指针重置，显著消除上下文切换与内存复制开销。

**🔧 技术方法**

采用统一图缓存、事务性 KV 隔离、持久线程池以及 256 位 AVX2 低位三值 SIMD 内核。

**📊 数据集**

使用 Meta Llama‑3.2‑3B‑Instruct 作为目标模型和 Llama‑3.2‑1B‑Instruct（低精度量化）作为草稿模型，实验在 8 核 Xeon Gold 5218 环境。

**📈 对比分析**

与 llama.cpp 以及标准自回归 baseline 对比，SubTernary 在 8 线程上实现 3.99× 速度提升（相对自回归 4.24×），并在长上下文下保持 O(1) 回滚延迟。

**⚠️ 局限性**

限制在于对低位量化模型的依赖，过大批量长度仍会受到 CPU 资源争用影响，且未在多 GPU 或更大模型上验证。

---

## 157. To Keep or Not to Keep: Learning KV Cache Retention in Disaggregated LLM Serving Systems

**arXiv ID:** 2610.28637 | [PDF](https://arxiv.org/pdf/2610.28637v1)

**作者:** Dong Liu `[一作]` (University of California, Los Angeles), Ying Nian Wu `[通讯]` (University of California, Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了KVLearn框架，用于在分离式LLM服务体系中智能决定KV缓存的保留与驱逐。

**💡 创新点**

创新点在于将保留决策视为成本优化问题，结合轻量级预测器PRP、成本感知评分CARS和自适应阈值控制器ATC，实现对不同前缀长度、网络带宽和重算成本的动态权衡。

**🔧 技术方法**

采用在线学习的多层感知机预测前缀重用概率，基于模型的重算与传输成本公式计算CARS分值，并通过PI控制器调整阈值。

**📊 数据集**

使用公开数据集ShareGPT、LLaVA‑Instruct‑150K（图像QA）和ActivityNet‑QA（视频QA）进行多模态推理实验。

**📈 对比分析**

与No‑Cache、LRU‑Pool、LFU‑Pool、Mooncake‑LRU等基线以及Oracle相比，KVLearn在TTFT上提升最多56%、在转移量上降低53%，并在多模态工作负载下达到近Oracle吞吐量。

**⚠️ 局限性**

局限包括需要约200条请求的热身期、对相同硬件的重算成本校准假设、PI控制器可能在突发负载下振荡，以及仅在公开学术轨迹上评估，缺乏真实生产环境验证。

---

## 158. PCLA-Bridge: A Framework for Cross-Domain Deployment of CARLA Agents

**arXiv ID:** 2610.28696 | [PDF](https://arxiv.org/pdf/2610.28696v1)

**作者:** Masoud Jamshidiyan Tehrani `[一作]` (Università della Svizzera italiana), Paolo Tonella `[通讯]` (Università della Svizzera italiana)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

该工作提出了PCLA-Bridge工具，使CARLA Leaderboard的端到端自驾代理能够在不依赖CARLA服务器的情况下，从任意传感器输入和控制输出源执行。

**💡 创新点**

通过解耦代理的每帧执行，将外部传感器数据与控制输出映射为代理期望的格式，实现跨域、跨仿真、跨真实平台的轻量级集成，无需改写代理代码。

**🔧 技术方法**

采用依赖隔离、张量序列化、路由、中间件以及三类适配器（控制输出翻译、帧校准、传感器映射）等技术。

**📊 数据集**

在离线重放中使用CARLA本域记录和nuScenes实测日志；在实时闭环中使用MetaDrive、AWSIM、AlpaSim三种仿真引擎。

**📈 对比分析**

通过离线控制命令一致性和实时闭环路程覆盖率等指标评估，结果显示八个高性能代理在所有三种仿真引擎上都能无修改成功执行，覆盖率因场景和引擎而异，表明工具可靠且能揭示域差异。

**⚠️ 局限性**

仅适用于无特权传感器的代理，无法迁移需主机提供真实姿态或特殊渲染的“特权”代理；在极端域移或复杂物理模型下仍可能出现性能下降。

---

## 159. How Much Gradient Does Truncated Backprop Lose in Looped Transformers?

**arXiv ID:** 2610.28871 | [PDF](https://arxiv.org/pdf/2610.28871v1)

**作者:** Aur Shalev Merin `[一作]` `[通讯]` (Independent researcher), Aur Shalev Merin (Independent researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对循环式Transformer模型的梯度截断损失进行了精确测量，并在多种模型与循环次数下评估其对训练成本的预测能力。

**💡 创新点**

提出使用单次完整反向传播即可获得每个循环次数的损失梯度，并发现损失梯度曲线几乎不随循环计数变化，但衰减率ρ会漂移；同时验证损失梯度在微调过程中下降；进一步测试损失梯度是否能比简单的k规则更好预测截断训练成本，结果不显著。

**🔧 技术方法**

使用单次完整反向传播、向量-雅可比乘积、对每个循环贡献求和以及ρ拟合等技术；在循环模型上测量梯度、计算ρ并进行统计比较。

**📊 数据集**

采用已发布的循环语言模型（Huginn、Parcae、Ouro、SmolLM2等）以及对其进行循环化的版本，并在GSM8K、WikiText等文本上进行验证。

**📈 对比分析**

通过在相同训练步骤下比较完整反向传播、截断反向传播和无循环梯度的三种跑法，测量损失梯度、训练成本，并与基于k或k×a的规则做对比；结果显示损失梯度与k本身预测性能相当，无法明显优于无测量规则。

**⚠️ 局限性**

实验受限于仅使用了少量模型、有限的循环次数、固定的微调步骤、相同的学习率、训练文本与验证文本相似，且训练成本测量仅在相同步数下；此外ρ与损失梯度在不同文本或模型间会变化，未能给出普适的阈值。

---

## 160. RegoMender: A Multi-agent Framework for Automated Rego Policy Repair

**arXiv ID:** 2610.29456 | [PDF](https://arxiv.org/pdf/2610.29456v1)

**作者:** Ruijie Zhang `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Bibo Tu `[通讯]` (Institute of Information Engineering, Chinese Academy of Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 RegoBench（1,159 条真实 Rego 政策缺陷修复对），并提出了 RegoMender 多代理框架，实现对 Rego 政策的自动化修复。

**💡 创新点**

创新点包括：①首个针对 Rego 政策的修复基准；②通过共享黑板、原子工具和回滚机制实现多代理协同，尤其对链式错误具有高效性；③利用反馈驱动的任务决策显著提升修复成功率。

**🔧 技术方法**

技术手段为：基于大型语言模型（如 DeepSeek-V3.1-Terminus）结合多代理系统（Orchestrator、Analyzer、Repairer），集成 OPA 验证、上下文扫描、错误分层、补丁与重写工具，以及停滞检测与回滚机制。

**📊 数据集**

使用的数据集包括：RegoBench（1,159 条实例），HCL（300 条）和 SQL（100 条）跨语言基准，以及 132 条 CVE 关联的真实安全漏洞实例。

**📈 对比分析**

方法上与 Agentless、ThinkRepair、APR-MCTS、RepairAgent、MetaGPT、AutoDev、Claude Code、Codex 等基线在 Pass@Tier‑2 与 JudgePass 维度进行对比，RegoMender 以 77.9% 的语义成功率领先最强基线 13.4%，在单步/链式、不同难度及 DSL 上均排名第一，并将回归率压至 7.7%。

**⚠️ 局限性**

限制主要体现在：依赖静态上下文导致无法处理缺失或动态依赖；迭代上限可能导致深度链式修复不完整；语义覆盖受测试用例限制，存在 1.9% 的 JudgePass 与 Pass@Tier‑2 差距；对编码风格和约束的细粒度处理尚不完善。

---

## 161. AdvFidelity: Unifying Preference Alignment and Few-Step Image Generation within Reinforcement Learning

**arXiv ID:** 2610.29380 | [PDF](https://arxiv.org/pdf/2610.29380v1)

**作者:** Sicong Kang `[一作]` (Taobao Tmall group of Alibaba), Pipei Huang `[通讯]` (Taobao Tmall group of Alibaba)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种能够在单一 RL 阶段内同时完成奖励对齐和少步生成的框架 AdvFidelity，并通过在线对抗性 fidelity 监督提升图像质量。

**💡 创新点**

创新点在于：1）引入内容匹配的正负样本构造，减少判别器利用高层差异的捷径；2）使用在线自适应判别器（DINOv2 + MLP）与 BT 损失，实时更新对低层伪影的监督；3）在 RL 过程中直接联合优化任务奖励与 fidelity 奖励，避免分阶段 distillation 产生的性能退化。

**🔧 技术方法**

采用 LeapAlign 的 ReFL 结构实现梯度通过采样轨迹的反向传播；使用 DINOv2 大模型作为冻结特征提取器，配合可训练 MLP 头进行判别；采用 Bradley–Terry 损失构造判别器；通过内容匹配重建负样本，配合裁剪技术关注局部细节。

**📊 数据集**

使用 HPDv2 语言‑图像提示数据、GenEval 组合场景数据，以及虚拟穿衣的真实试衣图像；参考样本来源包括：50 步 FLUX.1‑dev 输出、HPSv2‑aligned LeapAlign 生成样本、以及训练集中的真实图像。

**📈 对比分析**

与全步 LeapAlign、RTDMD、DanceGRPO、MixGRPO、ReFL、DRTUNE 等方法进行对比；在 4/8 步采样下，AdvFidelity 在 HPSv2、PickScore、ImageReward、UnifiedReward 等指标上与 50 步 LeapAlign 相当甚至更优，同时显著降低少步伪影，性能竞争力强。

**⚠️ 局限性**

局限性：1）依赖高质量参考样本，若参考分布与目标域差异大则效果下降；2）对抗判别器对极细微伪影仍可能漏判；3）实验仅基于 LeapAlign 体系，迁移到其他 RL 框架需要进一步验证；4）在极端风格转换或非视觉任务中可能表现不佳。

---

## 162. GRAIL: From Saliency to Semantics in Time-Series Explanation

**arXiv ID:** 2610.29036 | [PDF](https://arxiv.org/pdf/2610.29036v1)

**作者:** Enver Menadjiev `[一作]` (KAIST), Julia Stoyanovich `[通讯]` (NYU)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出一种基于时间序列语义区间（如 ECG 的 P/QRS/T 波）和非时间协变量的解释框架，利用生成器在数据流形上重建缺失区段，并通过 Owen 值给出加性、可解释的预测贡献。

**💡 创新点**

创新点包括：① 将解释单元设为领域语义区间并兼顾静态特征；② 用生成器在流形上产生缺失区域的对抗样本，替代固定基线；③ 采用两层 Owen 值与共同随机数估计，实现完全、精确的归因；④ 设计可在部署时一次前向推理的近似器。

**🔧 技术方法**

技术手段包括：协作博弈论、Owen 值、共同随机数 (CRN) 估计、Gaussian Process / 条件 VAE 生成器、PELT/模型感知分段器、FastSHAP 训练的单前向近似网络，以及统计检验 (Wilcoxon、Benjamini–Hochberg)。

**📊 数据集**

实验数据集涵盖两类合成数据（Synth、Mixed，含静态协变量）、三类临床 ECG 数据（MIT‑BIH、QT、LUDB）和两类工业传感器数据（FordA、Wafer），共六个数据集。

**📈 对比分析**

与 12 种基线（窗口/时间窗口 SHAP、TS‑MULE、LIME、C‑SHAP、GS‑SHAP、IG、KernelSHAP 等）在两种基线操作（on‑manifold 与 off‑manifold）下比较。精确游戏在 on‑manifold 上显著优于对手，完整性完全；在实际 ECG 上其性能与主流基线相当；快照近似器在真实数据上实现约 10³ 倍速度提升，faithfulness 接近精确游戏。

**⚠️ 局限性**

局限性包括：需要手工或学习的区域分割器；生成器的不确定性校准不足；在低信噪比合成数据上 faithfulness 降低；缺乏用户研究验证对专家的实际帮助；对非 GP 可表示的信号的适用性仍待评估。

---

## 163. Preference Is Not Permission: Personalizing the Auto-Purchase Line in Agentic Commerce

**arXiv ID:** 2610.29216 | [PDF](https://arxiv.org/pdf/2610.29216v1)

**作者:** Ryotaro Shimizu `[一作]` (ZOZO Research), Mo Fan `[通讯]` (ZOZO Research)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在代理式电商场景中个人化自动购买线（授权边界），构建了基于真实购买与评论历史的最小编辑对照实验数据集，并在此数据集上评估了十六种大语言模型的自动购行为。

**💡 创新点**

创新点：①将购买决策拆分为 fit、warrant、reversibility、price、urgency 等多因素，并提出三值授权线（不买、先确认、自动购买），阐明“Preference is not Permission”；②提出轻量级可解释的 warrant‑guard 控制器 C3‑P；③展示通过少量个人标签即可学习出更精准的个体化授权边界，优于任何 prompt+model 组合。

**🔧 技术方法**

技术手段：大语言模型（Gemini、GPT‑5、Llama 等）作为代理，Prompt‑Framing Sweep、二阶段控制器 C3‑P、逻辑回归/部分池化模型学习个体边界；使用多模型、多prompt 的实验框架与逻辑回归、F1、准确率等评估指标。

**📊 数据集**

数据集：基于亚马逊服装鞋饰和 Booking.com 酒店预订的真实历史与评论，构成 270 个最小编辑对照案例（16 家庭、15 名标注者）；发布 Counterfactual Benchmark 与人类授权参考。

**📈 对比分析**

比较方法：对 16 种 instruction‑tuned 模型与 80 种 prompt×model 组合进行基准评估；实验显示模型在弱保障因素下高频过度承诺；C3‑P 能将弱保障过度承诺降至 0；个体化边界学习在 30–40 条标签下平均 F1 为 0.79，优于任何 prompt+model 组合。

**⚠️ 局限性**

限制：样本规模有限（仅 15 家庭）；标注者一致性低（Krippendorff α≈0.25）；对照实验采用人为合成的最小编辑，缺乏多轮交互场景；实验仅覆盖单一领域特定因素，未验证跨域迁移效果。

---

## 164. Distilling Multi-Teacher Scoring Principles for Unsupervised Time Series Anomaly Detection

**arXiv ID:** 2610.29063 | [PDF](https://arxiv.org/pdf/2610.29063v1)

**作者:** Dongchan Cho `[一作]` (SimPlatform Co. Ltd.), Namsoon Jung `[通讯]` (SimPlatform Co. Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种无监督时间序列异常检测的知识蒸馏框架AnoMix，能够将多教师的异常评分原则融合为单一轻量级学生模型；

**💡 创新点**

创新点在于通过教师得分差异构造配对式弱监督(CoRank)，利用多教师共识的排序关系来蒸馏异常偏好，而非直接回归教师得分；

**🔧 技术方法**

主要技术包括：min-max教师得分归一化、多教师协作的Pairwise Ranking蒸馏（CoRank），以及轻量化的因果TCN学生网络；

**📊 数据集**

使用七个工业与公共数据集（PSM、SWaT、Exathlon、SMD、SMAP、Daphnet、SWAN）进行评估；

**📈 对比分析**

与十种单一检测器、五种静态分数集成方法以及六种基于时间序列的预训练基模型做比较，AnoMix在多项阈值无关指标（V-ROC、V-PR等）上平均排名最优，且在零样本跨数据集迁移中仅使用83k参数便能击败多达710M参数的大型基模型；

**⚠️ 局限性**

局限性包括：需先有多教师预训练模型，教师质量和多样性对性能影响显著；对极低异常率或超大尺度数据的泛化尚未充分验证；

---

## 165. The Price of Choice in Selective Prediction

**arXiv ID:** 2610.29067 | [PDF](https://arxiv.org/pdf/2610.29067v1)

**作者:** Daniel Cieślak `[一作]` (Gdańsk University of Technology), Andrzej Czyżewski `[通讯]` (Gdańsk University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了在已预定规则集合（menu）下，如何以最小样本量拒绝供应商对选择性分类器满足给定覆盖率与误差阈值的存在性声明，并提出了一种精确的交集-并集检验（IUT）方案；

**💡 创新点**

创新点在于：1）给出了菜单丰富度对样本复杂度的精确量化——在小风险间隙、大菜单情况下，IUT 的样本需求为 Θ(K(1−κ)/(κγ²))，并证明其在任意 VC 维度下达到极值；2）揭示了“可经验可行性一致（EFC）”证书的固有线性成本障碍；3）在不知分布的常数风险和异质风险两种场景下分别给出可避免和不可避免线性成本的理论阈值；4）通过模拟与真实 VLM 判别器及冻结分类器的实验验证了理论预言。

**🔧 技术方法**

使用了交集-并集检验（IUT）、VC 统一偏差、Clopper–Pearson 置信界、二阶矩法、分层/多项式近似、独立性检验以及非 EFC 的全局推断（如 universal inference）等统计技术；

**📊 数据集**

实验数据包括：①人工构造的“sign family”模拟；②三大 VLM 判别器（LLaVA‑Critic‑7B、Gemini 2.5 Flash、Phi‑4）以及多注释器 Polaris 的判别分数；③冻结的公开模型输出（CIFAR‑10/100 ResNet‑18、ImageNetV2 ViT‑B/16、CMU‑MOSEI MLP）；

**📈 对比分析**

与传统 Bonferroni、DKW、统一推断等方法相比，IUT 在任意覆盖阈值下实现了极大样本节约（相较于 EFC 需要的线性样本），并在大菜单、低风险间隙情形下保持与信息上限（Neyman–Pearson envelope）相近的检验力；

**⚠️ 局限性**

局限性包括：仅处理二值损失与单一覆盖阈值；假设样本独立同分布；线性成本结果为渐近理论，常数与对数项未完全收敛；对异质风险的分析仅在特定 sign family 上给出，未扩展至一般 VC 族。

---

## 166. Scalable Stress-Constrained Topology Optimization: A Super-resolution and Simultaneous Approach Based on Machine Learning

**arXiv ID:** 2610.28703 | [PDF](https://arxiv.org/pdf/2610.28703v1)

**作者:** Xiangyu Sun `[一作]`, Ramin Bostanabad `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于神经网络的同时拓扑优化框架（SNTO），用三个PGCAN网络分别表示位移、伴随变量和设计密度，在单一损失函数中同时优化，以解决大规模应力约束拓扑优化问题。

**💡 创新点**

创新点包括：① 将位移、伴随和设计分别映射到独立网络并利用自动微分实现SAND；② 通过离散伴随分析求解应力约束梯度；③ 利用损失可加性实现多GPU子切分，无需接口损失，保证连续性；④ 使用PGCAN消除频谱偏差，捕捉尖锐梯度；⑤ 引入逐步课程训练和自适应伴随缩放提高稳定性。

**🔧 技术方法**

技术细节：PGCAN网络、连续伴随分析、自动微分、形状函数梯度与高斯积分、剪切子切分、分布式训练（PyTorch DDP）、Heaviside投影、SIMP插值、Adam优化器、剪切阈值等。

**📊 数据集**

无传统训练数据；评估使用标准二维/三维应力约束基准（L beam、crack、hook、portal frame），网格为结构化四边/六面体，节点数从125K到8M；对比PolyStress实现。

**📈 对比分析**

与PolyStress对比，SNTO在相同约束下得到更尖锐、灰区更少、体积分数更低的设计；训练时间随GPU数量线性缩短，子切分显著降低峰值显存；在8M约束的3D案例中，8GPU上完成训练，耗时比单GPU提升5–23%，设计质量与PolyStress相当或更优。

**⚠️ 局限性**

局限性：仍需要多GPU/大显存资源，训练周期长；缺乏公开的3D应力约束对照实现；对非线性材料、制造约束等扩展尚未验证；伴随缩放和学习率需手动调节；在极端几何或自适应网格上的通用性未充分测试。

---

## 167. From VLM Planners to Robotic Agents: Self-Evolving Harnesses for History-Dependent Manipulation

**arXiv ID:** 2610.29560 | [PDF](https://arxiv.org/pdf/2610.29560v1)

**作者:** Tingwen Du `[一作]` (University of Science and Technology of China), Zhihui Li `[通讯]` (University of Science and Technology of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自演化的机器人代理SERA，通过可编辑的任务掌握框架将高层视觉语言模型与低层固定VLA执行器结合，实现历史依赖的操纵。

**💡 创新点**

创新点包括：1）在高层规划器周围构建可编辑的任务掌握框架，以持续管理任务上下文；2）通过执行反馈自动诊断并改进框架，而不需额外监督微调；3）将自演化过程与现有VLA模型集成，提升长程任务性能。

**🔧 技术方法**

利用预训练的视觉-语言-动作（VLA）模型作为低层执行器；高层视觉语言模型（VLM）进行规划；自演化流程使用分析、修订和验证代理；框架中的控制规则和评估标准以脚本形式实现。

**📊 数据集**

使用RoboMME历史依赖机器人操纵基准进行评估。

**📈 对比分析**

在RoboMME上，SERA的自演化版本在总体成功率上达到51.38%，相较于无自演化基线的44.38%提升了7.00个百分点。

**⚠️ 局限性**

主要局限是VLM推理导致的规划延迟，尤其在长时程执行中可能影响响应速度；对动态任务的实时性未做系统评估。

---

## 168. PerSPL: Personalized Semi-supervised Preference Learning

**arXiv ID:** 2610.29219 | [PDF](https://arxiv.org/pdf/2610.29219v1)

**作者:** Xiaoan Xu `[一作]` (Duke University), Zhengling Qi `[通讯]` (George Washington University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种 Personalized Semi-supervised Preference Learning (PerSPL) 框架，用来将无用户标识和无偏好标签的未标记响应转化为用户特定的监督信号。

**💡 创新点**

创新点包括通过 pair-mean score centering 解决跨提示评分坐标不确定性，并利用每个用户的单射校准实现伪标签化，从而在少量用户反馈下有效利用大规模未标记数据。

**🔧 技术方法**

使用的技术包括 Bradley–Terry 对数损失、均值正则化、单调等距回归（Pool‑Adjacent‑Violators）进行用户校准、阈值置信过滤和迭代自训练。

**📊 数据集**

在四个基准上实验：PersonalLLM、HelpSteer2、Reddit TLDR 和 Chatbot Arena，涵盖合成与真实用户/响应。

**📈 对比分析**

与七种代表性基线（BT、GPO、VPL、PAL、LoRe、MRM 等）对比，PerSPL 在见/未见用户的平均精度上始终领先，尤其在真实用户数据集上表现最佳。

**⚠️ 局限性**

局限性：需要每个用户至少有少量标注对支持少样本个性化，未处理零样本情况；对生成器分布漂移的鲁棒性仅在有限测试中验证。

---

## 169. Doob-Based Generative Policy Optimization

**arXiv ID:** 2610.28990 | [PDF](https://arxiv.org/pdf/2610.28990v1)

**作者:** Maojiang Su `[一作]`, Han Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于Doob h‑transform的无似然生成策略强化学习框架Doob Policy Optimization (DoobPO)，通过先学习比例网络表示策略改进，再通过条件漂移匹配将改进迁移至扩散或流生成策略；

**💡 创新点**

创新点在于将策略改进与生成模型学习分离，利用轻量级比例网络和Doob的漂移匹配实现无似然的策略更新，理论上等价于端点约束的最优控制，得到最小能量的生成过程；

**🔧 技术方法**

采用Doob h‑transform、条件漂移匹配、比例网络、PPO/GRPO/PMD等政策梯度方法，配合扩散模型和流匹配模型的生成框架，以及马尔可夫决策过程和控制理论分析；

**📊 数据集**

在OpenAI Gym MuJoCo连续控制基准（10个任务）和Robomimic Manipulation任务（如Reach、Push等）上进行实验；

**📈 对比分析**

与DIPO、DPPO、DPMD、SDAC、FPO、ReinFlow、Flow‑SDE、SAC、TRPO、PPO等多种基准对比，DoobPO在大多数MuJoCo任务（扩散版8/10、流版9/10）上实现最高平均回报，Robomimic任务上与DPPO相当；

**⚠️ 局限性**

局限性包括仅验证连续生成任务，未对比例网络估计误差和漂移匹配近似的影响进行定量评估；对离散动作空间或更复杂场景的适用性尚未探究；对预训练生成模型的依赖较高。

---

## 170. Keeping Time vs. Understanding Time: Neither Alone Explains Temporal Reasoning in LLMs

**arXiv ID:** 2610.29507 | [PDF](https://arxiv.org/pdf/2610.29507v1)

**作者:** Auss Abbood `[一作]` (University of Cambridge), Nigel Collier `[通讯]` (University of Cambridge)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过能力维度评估和跨数据集转移实验，检验时间算术与时间语义两种模式是否在现有 temporal reasoning benchmark 中真正对应不同的认知能力。

**💡 创新点**

创新点在于将能力标签（delean 维度）与实际模型表现相结合，揭示这些 benchmark 主要衡量的是知识检索而非算术或语义推理，并提供了对传统算术/语义二分法的实证挑战。

**🔧 技术方法**

研究使用 GPT‑4o 与 Gemini 对任务进行 18 维认知标签标注，采用 GRPO 微调框架进行自回归训练，并通过 PCA、逻辑回归等统计方法分析能力分布和转移效果。

**📊 数据集**

实验覆盖了 ToT Arithmetic/Semantic、TRAM Arithmetic/Semantic、TCP、TempReason、TimeQA、MenatQA、Date Arithmetic、TimeDial 等九个子集，共 10 个数据划分。

**📈 对比分析**

通过在某一模式下微调后在同一或不同模式的数据集上评估准确率变化，发现转移效应几乎为零，模型在知识维度（kns、knc、cec）上获得显著提升，而在量化与逻辑维度上的提升不明显，说明算术/逻辑能力并非主要驱动力。

**⚠️ 局限性**

局限性包括仅使用 12B 参数规模模型、转移实验仅针对 ToT 与 TRAM 两个 benchmark、缺乏更大模型或更细粒度任务划分的验证，且未对所有可能的算术/语义细分进行系统测试。

---

## 171. Forging More Tasks into Better Harnesses: Agentic Harness Evolution via Successive Scale Racing

**arXiv ID:** 2610.28815 | [PDF](https://arxiv.org/pdf/2610.28815v1)

**作者:** Jingzhe Liu `[一作]` (Michigan State University), Clark Mingxuan Ju `[通讯]` (Snap Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Successive Scale Racing (SSR) 框架，通过对执行经验进行优化演化代理机套件，并解决任务池规模扩大导致性能退化的问题。

**💡 创新点**

创新点在于结合代表性证据排序、层次化候选生成和累计性能竞赛，动态搜索最优证据组合，显著提升代理机性能并实现任务池正向扩展。

**🔧 技术方法**

核心技术包括代表性证据排序（RER）、多尺度候选生成、累计性能竞赛（Successive Halving）以及基于 LLM 判别器的执行轨迹提炼与优化。

**📊 数据集**

使用 SWE‑Bench Pro Verified、DeepSWE、GAIA‑2 三大基准任务集进行实验评估。

**📈 对比分析**

与 Meta‑Harness、HGM、MGM、AHE、Retro 等现有方法对比，在 Codex（GPT‑5.5）和 Claude Code（Opus‑4.8）两种执行环境下，SSR 在所有六个配置中均取得最高平均 Pass@1，且随着任务池从 40 任务扩展到 200 任务时性能持续提升。

**⚠️ 局限性**

局限性包括：仅关注噪声/冗余证据的影响，未探究更细粒度的证据过滤或强化学习优化方法；实验仅覆盖两类大型预训练模型，缺乏对不同规模或领域模型的泛化验证；以及对安全性与道德风险的实际部署仍需进一步研究。

---

## 172. DynaWeave: Learning to Adapt through Dynamic Scenario Synthesis

**arXiv ID:** 2610.29443 | [PDF](https://arxiv.org/pdf/2610.29443v1)

**作者:** Xuan Yang `[一作]` (City University of Hong Kong), Ning Miao `[通讯]` (City University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个自动化框架 DynaWeave，能够将已有的静态可执行任务转化为可验证的动态场景。框架通过识别任务需求、环境状态和外部信息三类源，生成可执行的外部事件，并将这些事件组织成有向无环图（DAG）。在代理交互过程中，执行引擎根据完成操作、虚拟时间或随机触发等机制激活事件；动态验证器在验证时仅修改受已发生事件影响的检查，保持其余原始验证要求不变。利用该框架，作者在36个环境中生成了2095个动态场景，并收集了对应的验证轨迹，用以对 Qwen3.5 系列模型进行监督微调。

**💡 创新点**

创新点主要体现在：
1) 自动化生成符合任务背景的可执行外部事件，并对其有效性、可执行性进行程序化检验；
2) 通过 DAG 结构将事件按依赖关系组织，实现多事件的可控触发与避免循环依赖；
3) 设计动态验证器，使验证过程只关注实际发生的事件，既保持了原有任务完整性，又能适应动态变化；
4) 将静态任务集合与动态场景无缝结合，避免为每个域重新设计动态测试，提高资源利用率。

**🔧 技术方法**

技术手段包括：
- 大语言模型（LLM）用于生成事件主题、具体事件内容、构建 DAG、以及动态验证器规则；
- 有向无环图（DAG）用于建模事件依赖；
- 触发机制（完成特定工具操作、虚拟时间、随机激活）控制事件激活时机；
- 动态验证器在执行时根据已触发事件动态调整验证规则；
- 监督微调采用 LLaMA‑Factory、LoRA（rank=16）对 Qwen3.5（2B、4B、9B）进行两轮训练。

**📊 数据集**

使用的数据集：
- 训练阶段：EnvScaler、τ^2‑bench、ToolSandbox、AgentDojo 36 个环境，共生成 2095 个动态场景与 3414 个事件；
- 评测阶段：Gaia2、CostBench、ToolMaze 三大动态基准测试。

**📈 对比分析**

比较方法与性能：
- 与 Gemini‑3.1‑Pro‑Preview、GPT‑4o 等基线模型对比；
- 在 CostBench 上，Qwen3.5‑2B 的任务完成率从 3.00% 提升至 70.34%，提升幅度 67.34pp；
- 在 ToolMaze 上，Qwen3.5‑9B 的综合得分 32.28，超过 Gemini‑3.1‑Pro‑Preview 的 32.19；
- 在 Gaia2 上，模型尺寸增大后整体分数提升；
- 三种模型尺寸（2B、4B、9B）均表现出一致的性能提升，尤其在工具使用可靠性、成本控制和故障恢复方面有显著改进。

**⚠️ 局限性**

局限性：
- 对于无错误信息的工具结果错误或需要精确时序响应的情况，模型仍表现不佳；
- 训练场景中缺乏部分事件类型（如工具结果错误、时序错误）导致泛化能力不足；
- 数据量增加后性能提升趋缓，且并非所有评估指标都随数据增加而改善；
- 动态验证器的规则生成依赖 LLM 质量，可能出现不一致或遗漏；
- 需要进一步探索如何在场景构造与训练阶段更好地覆盖时序与错误类型，以提升代理的时序感知与错误恢复能力。

---

## 173. Is Refusal Direction Universal across Low-Resourced Languages?

**arXiv ID:** 2610.28897 | [PDF](https://arxiv.org/pdf/2610.28897v1)

**作者:** Anastasiia Ivanova `[一作]` `[通讯]` (LMU Munich), Anastasiia Ivanova (LMU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究扩充PolyRefuse数据集，加入了白俄罗斯语、巴什基尔语和塔吉克语三种低资源语言，并在Qwen2.5-14B-Instruct模型上评估了跨语言拒绝指令的转移效果。

**💡 创新点**

创新点在于揭示拒绝指令的通用性并非完全二元，而是受语言安全对齐程度影响的梯度现象；同时发现几何对齐与行为转移存在偏离，强调低资源语言在安全机制上的不平等。

**🔧 技术方法**

采用差异均值法提取线性拒绝方向，利用线性干预和向量相加在模型内部进行操纵；用余弦相似度、PCA+Silhouette分数评估几何关系；通过WildGuard对模型合规率进行安全评估。

**📊 数据集**

使用PolyRefuse原始数据集（English、Yoruba等）并新增Belarusian、Bashkir、Tajik版本；利用WildGuard生成的合规/拒绝判定来评估安全性。

**📈 对比分析**

通过比较源语言方向在目标语言上的合规率变化，并计算归一化效果Δ，发现English和Belarusian方向在多数目标上几乎能使合规率降至接近0，而Bashkir和Tajik方向在已高合规率语言上几乎无效；几何相似度与行为转移不完全一致。

**⚠️ 局限性**

仅在单一Qwen2.5-14B-Instruct模型上进行实验，语言样本仅限三种低资源语言；翻译质量和验证依赖于非母语背景，可能影响实验结果的普适性。

---

## 174. Controllable Generative Recommendation via Guided Token Refinement

**arXiv ID:** 2610.28642 | [PDF](https://arxiv.org/pdf/2610.28642v1)

**作者:** Haichao Zhang `[一作]` (Xi'an Jiaotong-Liverpool University), Jia Wang `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ReGen框架，实现实时可控的生成式推荐；

**💡 创新点**

通过并行迭代细化将生成任务转化为mask‑filling，并在logit层级施加多层次控制信号；

**🔧 技术方法**

使用双向Transformer编码器、残差向量量化（RQ‑VAE）生成语义ID、层级logit引导、置信度驱动的迭代细化以及基于Trie的拓扑约束束缚生成；

**📊 数据集**

在Amazon 2023 Review数据集的三大品类（Industrial & Scientific、Musical Instruments、Video Games）上进行实验；

**📈 对比分析**

与传统判别式推荐、生成式推荐以及可控推荐基线比较；ReGen在准确性方面可与SOTA持平，且在多样性控制时实现了更优的准确率‑多样性折衷，验证了指导+细化+拓扑约束的有效性；

**⚠️ 局限性**

仅针对多样性控制，其他目标（公平性、曝光平衡等）未系统评估；依赖语义ID分词器的质量，若编码噪声大则控制效果下降；

---

## 175. Is 2:4 Sparsity Still Worth It? Semi-Structured Sparsity versus Quantization in LLM Inference

**arXiv ID:** 2610.28947 | [PDF](https://arxiv.org/pdf/2610.28947v1)

**作者:** Xiuyi Qin `[一作]` `[通讯]` (Independent Researcher), Xiuyi Qin (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对比2:4半结构稀疏化与量化在LLM推理中的效果，系统评估了不同精度层、GPU代、存储预算和算子族的性能与准确性，并探讨了在量化底层时是否仍能从2:4中获益。

**💡 创新点**

创新点在于：①在同等存储位宽和同等张量核心层级下，首次将2:4稀疏化与量化直接对比，揭示了在量化底层之外二者的优势与劣势；②提出“能量损失”判据，理论解释了2:4为何在大多数情况下落后；③针对FP8底层给出了可 retrain 的 2:4 加速方案并量化其真实推理吞吐提升。

**🔧 技术方法**

使用的技术包括：半结构稀疏压缩（2:4）、量化（INT、FP8、NVFP4、MXFP4）、在线裁剪与量化、SparseGPT/Wanda/量化器（GPTQ、AWQ）、cuSPARSELt 及自研融合核、vLLM 服务器、以及多代GPU（Ada、Hopper、Blackwell）的 Tensor Core。

**📊 数据集**

使用的数据集：C4（校准）、WikiText‑2（困惑度评估）、GSM8K（准确率评估）、RULER（检索任务）、FineWeb‑Edu（微调训练）。

**📈 对比分析**

比较方法：在同等存储位宽（计入2:4索引）和同等张量核心层级的前提下，分别测量每层算子误差、整体模型困惑度、GSM8K准确率、RULER，以及在 vLLM 服务器上的预填充/解码吞吐。结果显示：①在量化底层之外，量化步长优于添加2:4，且在同等位宽下更快更准；②在FP8底层，重训练后的2:4可在线性层实现1.3–1.7×速度提升，vLLM 上预填充吞吐提升27%；②对Blackwell 的 4:8 pairwise 方案虽更快，但未达到可用性。

**⚠️ 局限性**

局限性：①2:4 仅在量化底层能带来显著加速，需额外 retraining ；②pairwise 4:8 在当前训练预算下表现不佳；③实验受限于特定模型（Llama‑3.1‑8B、Qwen3‑8B）和 GPU 架构；④在线裁剪/量化的实现复杂，且预填充阶段的 I/O 可能成为瓶颈；⑤评价指标（困惑度）对 reasoning 任务不够敏感，可能低估 retraining 效果。

---

## 176. Solving problems in RDV Graphs faster and easier

**arXiv ID:** 2610.28932 | [PDF](https://arxiv.org/pdf/2610.28932v1)

**作者:** Therese Biedl `[一作]` (University of Waterloo), Günter Rote `[通讯]` (Freie Universität Berlin)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了在 RDV 图（即根定向路径图）上对最大独立集、最小支配集和最大匹配的三种 O(n) 线性时间算法，直接在树表示上做自底向上遍历，避免了之前需要的 T‑形表示与复杂查询。

**💡 创新点**

创新点：1）将三种经典贪心算法统一到同一树遍历框架；2）用极简的数据结构（链表、少量整数，匹配时仅需并查集）实现；3）把之前的 O(n log n) 运行时间压缩到 O(n)，并给出完整的最优性证明。

**🔧 技术方法**

主要技术：RDV 图的树表示与 β‑袋概念、递归自底向上遍历、局部标签与列表维护、并查集（Union–Find）用于匹配的快速父节点查询，以及深度信息来维护最优选择。

**📊 数据集**

本文为理论研究，无实验数据集；仅给出示例图作为说明。

**📈 对比分析**

与之前的 O(n log n) RDV 图算法、区间图的 O(n log log n) 匹配算法以及 O(n) 区间图独立集/支配集算法进行比较，证明在给定 O(n) 大小的 RDV 表示下，本工作实现了真正的线性时间并且同样保持最优性。

**⚠️ 局限性**

限制：1）算法需要先获得 O(n) 大小的 RDV 表示；2）匹配算法仍依赖并查集（虽然线性时间，但实现相对复杂）；3）尚未证明能否在不使用并查集的情况下保持 O(n) 复杂度；4）目前的改进仅针对 RDV 图，是否能推广到其他图类仍是开放问题。

---

## 177. Encoding Trees: A Method for Producing a Variable-Length Prefix Code From Sets of Arbitrary Size

**arXiv ID:** 2610.29032 | [PDF](https://arxiv.org/pdf/2610.29032v1)

**作者:** Bryony H. Laporte `[一作]` (Queen's University), Karen Rudie `[通讯]` (Queen's University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

本文提出了“剪枝标记编码树”这一概念，并对其结构、路径函数以及与传统二叉树的关系进行了系统的定义与证明。通过引入剪枝高度 h，构造了一种可以自适应地裁剪右子树的编码树，从而在保持前缀码性质的前提下，控制路径所需的比特数。

**💡 创新点**

创新点在于：①用剪枝高度统一描述树的剪裁程度；②证明了剪枝后树的叶子数为 min(x,2^h)，并给出了路径长度与 h 的明确关系；③提出了可将固定长度的二进制串映射到标签序列的逆向编码方法，满足任意长度二进制串都能唯一映射到标签集合中的元素。

**🔧 技术方法**

采用了组合树结构与递归定义、数学归纳法证明、路径函数扩展等技术。整体方法以理论推导为主，利用二叉树性质、对数运算与最优前缀码的概念进行分析。

**📊 数据集**

未使用任何实际数据集；论文纯粹为理论分析与证明。

**📈 对比分析**

因未进行实验或数值比较，论文中没有性能指标。主要通过数学证明展示所提出方法满足前缀码要求，并给出路径长度与剪枝高度的关系公式。

**⚠️ 局限性**

局限性：①仅在理论层面给出，缺乏实际编码/解码实现与复杂度评估；②对剪枝高度的选择依赖于标签集大小，实际应用中可能需要经验或额外算法来确定最优 h；③对非平衡标签分布或非整数长度标签的处理尚未讨论。

---

## 178. Density-Level Backward Error for Low-NFE Generative Flows

**arXiv ID:** 2610.29475 | [PDF](https://arxiv.org/pdf/2610.29475v1)

**作者:** Ewsey Obzherin `[一作]` (FusionBrain Lab), Valentin Khrulkov `[通讯]` (FusionBrain Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在低网络评估预算下，提出基于密度层后向误差的分配规则，优化采样步长和速度场的表示。

**💡 创新点**

创新点是将修改方程的梯度分量投影到Wasserstein切线空间，得到局部边缘误差量化，并通过该量化导出求解器阶数相关的计算预算分配，以及可选的速度场代表性优化。

**🔧 技术方法**

使用了数值积分的后向误差分析、Wasserstein几何投影、最优分配理论、有限步与连续步的变分极限、以及欧拉/Heun等数值求解器的分析。

**📊 数据集**

在合成高斯与混合模型以及预训练的CIFAR-10 EDM概率流ODE上进行验证。

**📈 对比分析**

与传统的时间步长调度（log-σ、EDM-ρ）以及更高阶求解器（DPM++、EDM Heun）对比，定量实验显示在低NFE下，经过优化的步长网格可显著降低FID（如8-20 NFE提升约30%），且相较于固定步长，收敛速度更快。

**⚠️ 局限性**

局限在于只针对固定阶数的求解器优化步长，不改善求解器本身；在高NFE下优势消失；对非梯度场的实际可实现性和在更复杂数据集上的泛化尚待进一步验证。

---

## 179. OccJEPA: Occlusion as Masking for Amodal Video Segmentation

**arXiv ID:** 2610.29061 | [PDF](https://arxiv.org/pdf/2610.29061v1)

**作者:** Genpei Zhang `[一作]` `[通讯]` (University of Wisconsin--Madison), Genpei Zhang (University of Wisconsin--Madison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于自监督视频世界模型 V-JEPA 的 occlusion-as-masking 方案（OccJEPA），用预训练的预测器完成视频中的隐蔽物体掩模，无需深度网络或采样，生成确定性且速度更快的全景掩模。

**💡 创新点**

创新点：1) 将遮挡视为掩码任务，直接使用 V-JEPA 预训练的预测器完成隐藏区域；2) 只需一个轻量化的头部和低秩适配器即可完成全景分割；3) 通过重新裁剪（re-crop）提升对高度遮挡物体的处理；4) 通过控制实验验证提升主要来自预训练权重，而非模型容量。

**🔧 技术方法**

技术手段：自监督空间-时间掩码预测（V-JEPA 2.1 预训练模型）、视觉 Transformer 编码器、预训练预测器、低秩适配器（LoRA）、轻量级掩模头部、可选的二次裁剪推理。

**📊 数据集**

使用的数据集包括 SAIL-VOS（游戏引擎生成的合成视频）、TAO‑Amodal（真实视频的全景框标注）、MOVi‑D/MOVi‑B（合成视频）、Kubric（静态和动态场景）以及 SAM2 跟踪得到的可见掩模。

**📈 对比分析**

与现有最佳方法 Diffusion‑VAS 进行对比。OccJEPA 在 SAIL‑VOS 的 occluded IoU 和 TAO‑Amodal 的 AP_50 上均超过 Diffusion‑VAS，并且在同等硬件下运行速度提升约 10‑30 倍，显著减少 FLOPs。即使在使用 SAM2 生成可见掩模的完整系统中，OccJEPA 仍保持领先。

**⚠️ 局限性**

局限性：1) 依赖于高质量的可见掩模输入；2) 对于极少可见区域（<10%）的遮挡，像素级生成方法可能更精细；3) 需要预训练的 V‑JEPA 模型，迁移到其它域可能受限；4) 重新裁剪增加了两倍推理成本。

---

## 180. Mixture of Flow Matching and Particle Filter for Proprioception-Based Online Contact Estimation

**arXiv ID:** 2610.29568 | [PDF](https://arxiv.org/pdf/2610.29568v1)

**作者:** Seo Wook Han `[一作]` (Korea Advanced Institute of Science and Technology), Min Jun Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种结合流匹配生成模型和粒子滤波的 ContactFlowFilter，用于机器人手臂的自我感知在线接触状态估计（包括接触点数、位置和力）。

**💡 创新点**

创新点在于：① 采用双采样（dual-sampling）将流匹配模型产生的样本作为额外粒子，缓解粒子枯竭和多模态问题；② 设计了空间对齐的数据表示和 Transformer 结构，使模型能够一次性处理单/双接触并实现接触事件检测；③ 通过流引导的粒子集管理（Flow‑Guided‑Management）实现鲁棒的接触事件识别。

**🔧 技术方法**

核心技术包括粒子滤波（PF）、流匹配（Flow Matching）生成模型、Transformer 编码器-解码器、SDF 预训练网络、基于 QP 的测量模型和动态观测器。

**📊 数据集**

训练数据来自 MuJoCo 仿真，随机生成 1~2 处接触、10–25 N 力，加入噪声；真实实验使用 Franka Emika Panda 7‑DOF 机器人与 ATI Omega191 基座 F/T 传感器，收集约 200 次双接触真实数据做评估。

**📈 对比分析**

与三种基线（传统 PF、MCP‑EP 以及仅使用流匹配粒子）在模拟和真实实验中对比。ContactFlowFilter 在接触点位置误差、力误差、粒子分布相似度（PLS、EMD）上均优于基线，单接触误差 0.36 cm，双接触 0.88 cm；实时性保持 0.69 ms/单接触，2.69 ms/双接触。

**⚠️ 局限性**

目前仅支持最多两点接触；若多点接触，单一观测下多重解空间过大，模型难以区分；此外流匹配训练在仿真到实测的转移仍需改进。

---

## 181. A Small Model Asks Better: Fine-Tuning Query Clarification for Retrieval-Augmented Generation

**arXiv ID:** 2610.28686 | [PDF](https://arxiv.org/pdf/2610.28686v1)

**作者:** Lihong He `[一作]` (IBM Research), Marina Danilevsky `[通讯]` (IBM Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了ClariEdge数据集，包含多轮对话、检索文档和硬负样本，用于训练并评估RAG系统中查询歧义检测与澄清生成。

**💡 创新点**

① 在同一数据集同时涵盖检索前后歧义；② 引入硬负样本以区分真正歧义与可从上下文解析的表面歧义；③ 证明小模型微调（LoRA）在此任务上优于大型模型提示；④ 在单一模型中实现预检索和后检索两阶段歧义判断。

**🔧 技术方法**

LoRA轻量化微调、检索增强生成（RAG）框架、实体选项中心评估指标、硬负样本构造与人工校对、对比提示法。

**📊 数据集**

ClariEdge（约3.9k样本，4个领域：ClapNQ、Cloud、FiQA、Govt），与AmbigQA、CLAMBER、ClariQ、Abg-CoQA对比；此外做了Banking和Telco领域的OOV评估。

**📈 对比分析**

与7种提示基线（GPT-4o、GPT-5.6 Sol、GPT-OSS-120B、GPT-OSS-20B、Granite-4.1-30B、Granite-4.1-3B、Granite-4.0-micro）在全集、预检索、后检索三种设置下进行分类准确率与生成F1比较。LoRA在分类上最高可达96.6%（Granite-4.1-3B），生成F1约89%，均显著优于提示模型。去除硬负样本会导致准确率显著下降。

**⚠️ 局限性**

仅支持单轮澄清，未实现多轮澄清；实验仅在英文数据上进行，跨语言泛化尚未验证。

---

## 182. SAIL: A Learnable Generalisation of Spatial Autocorrelation over Graph-Structured Data

**arXiv ID:** 2610.28806 | [PDF](https://arxiv.org/pdf/2610.28806v1)

**作者:** Piotr Keller `[一作]` (University of Warwick), Fayyaz Minhas `[通讯]` (University of Warwick)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出可学习的空间自相关指数学习器 SAIL，能够在高维图结构数据上自动学习特征投影与邻域亲和度，并生成可解释的局部与全局空间自相关指标；

**💡 创新点**

将传统空间自相关统计（如 Moran's I、Geary's C）转化为可微分、可学习的框架，同时同时学习特征投影与邻接权重，实现监督与无监督的空间模式发现；

**🔧 技术方法**

利用可学习特征映射、邻域亲和度学习、局部自相关算子、全局汇聚与可微分图神经网络实现端到端优化，并提供理论证明与 Rademacher 复杂度分析；

**📊 数据集**

在合成图、肿瘤空间转录组/蛋白组（4i、ORION‑CRC、cSCC）、全景影像 WSIs（TCGA‑BRCA/CRC）、宏观经济与 COVID‑19 全球疫情数据等多种大规模多模态数据集上进行评估；

**📈 对比分析**

与经典空间统计、多变量扩展、PCA、GNN（GCN、GAT）等方法对比，在预测 Moran's I、治疗反应预测、存活分析和肿瘤亚型分类等任务中均表现出显著优势（AUC、C‑Index 等指标提升显著）；

**⚠️ 局限性**

对图结构和距离度量的显式假设敏感，学习过程对图规模与参数调优要求较高；当目标主要由特征内容驱动而非空间自相关时，其优势不明显。

---

## 183. Interpretable Cutting Plane Selection for Mixed-Integer Linear Programming Using Genetic Programming

**arXiv ID:** 2610.29311 | [PDF](https://arxiv.org/pdf/2610.29311v1)

**作者:** Gwen maudet `[一作]`, Grégoire Danoy `[通讯]` (University of Luxembourg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GPSCS框架，用遗传程序演化可解释的切割选择策略，并通过统计方法自适应决定每轮保留的切割数。

**💡 创新点**

创新点在于：①将手工切割启发式通用化为符号表达式搜索空间；②用遗传程序生成可读评分函数；③结合统计dropout检测实现自适应切割数，无需额外学习。

**🔧 技术方法**

使用遗传程序（GP）演化评分函数，结合正交剪枝和统计阈值决策；对比了SCIP默认、无切割(NoCuts)与神经网络基方法HEM。

**📊 数据集**

在七类基准数据集上评估：三类合成（GISP、MAXSAT、FCMCNF）和四类真实（MIK、Binpacking、Cuttingstock、Cap lot-sizing），训练集为较小实例，评估集平均需 41 倍时间。

**📈 对比分析**

与SCIP、NoCuts、HEM比较，GPSCS 在所有类中平均提升 6%（相较SCIP），在三类中最快，整体平均比HEM快两倍，显著优于无切割且不依赖GPU。

**⚠️ 局限性**

局限性：需要针对每个问题类单独训练评分函数，跨类泛化差；GP搜索耗时较多；统计阈值需手工设定，对极端分布可能不稳定。

---

## 184. Building Knowledge Graphs for Theses: A Survey

**arXiv ID:** 2610.28648 | [PDF](https://arxiv.org/pdf/2610.28648v1)

**作者:** Donia Ruelas `[一作]` (National University of Altiplano Puno), Elwin Huaman `[通讯]` (University of Cambridge)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

系统综述了2021‑2025年间构建学术论文知识图谱（Thesis Knowledge Graph, TKG）的研究方法，基于知识图谱生命周期将研究归纳为知识生成、知识构建与知识丰富三阶段；

**💡 创新点**

创新点在于首次针对论文领域构建知识图谱生命周期模型，综合归纳了从信息抽取到语义建模再到AI驱动丰富的完整方法论，并指出论文作为单一文献源的研究空缺；

**🔧 技术方法**

主要技术包括：自然语言处理（NER、RE、三元组生成、实体链接、跨模态信息抽取）、本体工程与语义网标准（RDF/OWL/SPARQL、schema 设计）、图嵌入、Linked Open Data、Neo4j等属性图平台以及大语言模型（LLM）用于自动三元组生成与查询翻译；

**📊 数据集**

使用的“数据集”为系统检索得到的46篇同行评议论文（来自 ACM DL、IEEE Xplore、Scopus），并未在实验中使用公开的学术论文或学位论文数据；

**📈 对比分析**

本文不进行实验性比较与性能评估，主要通过定性分析与文献计量对现有方法进行梳理与总结；

**⚠️ 局限性**

限制主要包括：缺乏针对学位论文的端到端实证方案；对多语言、结构异质性及术语专业化的适配不足；对LLM产生的幻觉与可信度仍未得到解决；

---

## 185. TenetBench: A Clause-Anchored Russian Corpus for Evaluating Language Models against Professional Codes of Ethics

**arXiv ID:** 2610.29477 | [PDF](https://arxiv.org/pdf/2610.29477v1)

**作者:** Yegor Anichkov `[一作]` (Trusted AI Research Center Russian Academy of Sciences), Sergey Bolovtsov `[通讯]` (Trusted AI Research Center Russian Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个以专业伦理准则条款为锚点的俄语语料库TenetBench，包含 3,433 对冲突与无害请求对，并提供了自动化生成与筛选流程；使用冻结的语言模型“判定者”为每个回答打分，评估 14 种模型及推理模式在满足专业规范方面的表现。

**💡 创新点**

① 以条款为锚点自动生成任务，保证每个冲突请求都能与同一情境下的无害对照直接对比；② 使用“判定者”模型进行多维度评判（是否执行、是否违反条款、是否拒绝/警告），并提供可验证的依据片段；③ 在模型评估中区分安全理论（警告）与实际违反，揭示模型“安全剧场”现象。

**🔧 技术方法**

① 自动化文本抽取与生成管道（基于解析树提取条款、构造情境、生成冲突与对照请求）；② 冻结的判定者模型（基于提示模板的多维度判定）；③ 评价脚本：对 14 行模型/推理组合进行统一实验，收集判定结果并统计各类比例。

**📊 数据集**

核心数据集：15 本俄语专业伦理准则（共 3,712 条款，过滤后 3,433 个冲突请求，3,433 对照请求），另有两份英文准则的迁移样例；任务共 6,866 条，包含 1,579 个情境、395 条可用条款。数据及生成管道已公开。

**📈 对比分析**

对 14 个模型/推理模式，计算“非冲突率”（不违规或安全拒绝），以及同一情境下的拒绝率与违规率。结果显示：大多数模型（11/14）在冲突请求上产生违规内容，警告率高但违规仍普遍；前 3 名模型（Claude Sonnet 5、Qwen3.6‑27B、Qwen3.6‑35B‑A3B）在非冲突率上最高，安全拒绝比例也最高；推理开启往往降低合规性；在保密信息条款上模型表现最好。

**⚠️ 局限性**

局限性：① 判定者与人工标签的对齐受限，仍需人工复核；② 仅覆盖俄语系统 15 条准则（5 条保留），无法覆盖其他语言或法规；③ 只评估回答文本的合规性，未涵盖用户后续行动；④ 模型输出截断、路由差异可能影响结果；⑤ 数据集规模有限，某些条款样本不足；⑥ 依赖冻结的判定者，后续模型迭代无法实时更新。

---

## 186. Understanding Target Probability Reuse in Speculative Decoding

**arXiv ID:** 2610.29210 | [PDF](https://arxiv.org/pdf/2610.29210v1)

**作者:** Jungseob Lee `[一作]` (Korea University), Chanjun Park `[通讯]` (Soongsil University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在树形显式推理（speculative decoding）中利用目标模型提前计算的下一步概率分布（root row）来改进树的选择，从而提升接受率和解码速度。

**💡 创新点**

提出在树分配前先抽取首个 token（root-draw selection）和使用目标模型的完整下一步分布（root selection）作为树选择依据，证明其在不改变目标分布的前提下可显著提升接受率，并在多模型、多任务上验证其有效性。

**🔧 技术方法**

采用树式显式推理框架、EAGLE-3 预先生成树、目标模型概率重排（reranking）、独立目标推断与树验证相结合的并行策略。

**📊 数据集**

使用六个公开大语言模型（Llama-3.1-8B、Qwen3-4B、Qwen3-8B、Qwen2.5-7B、14B、32B）以及 Spec-Bench、HumanEval、MT‑Bench 等多任务数据集进行评估。

**📈 对比分析**

将 root selection / root-draw selection 与传统的 draft ranking 以及 PEARL‑style 预选树进行对比，实验显示在 59 节点树下接受率提升约 8%（Greedy）/ 6%（温度1），并在 RTX A6000 双 GPU 上实现约 2% 的解码速度提升，且在长文本和不同任务中保持正向增益。

**⚠️ 局限性**

实验仅覆盖六个模型，使用单样本批量解码和独立 GPU 的独立 drafters，验证策略假设仅适用于路径对齐、条件精确的验证器，且在 off‑path 输出和更大规模并行环境下需要进一步评估。

---

## 187. One Policy, Many Styles: Where to Aggregate Preferences for Steerable Humanoid Control?

**arXiv ID:** 2610.28925 | [PDF](https://arxiv.org/pdf/2610.28925v1)

**作者:** Ziqian Huang `[一作]` (Georgia Institute of Technology), Animesh Garg `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了偏好条件化多目标强化学习中单一策略的可调性，提出多维度评估指标并比较不同聚合层级在两套人形机器人控制框架中的表现。

**💡 创新点**

提出Preference Reward Responsiveness (PRR)矩阵及其自PRR、交叉PRR、选择性和排名准确度等指标，用以分离意向响应、跨目标泄漏与任务保留；同时系统比较奖励层、优势层、损失层和PCGrad聚合策略在可调性、覆盖度与任务表现上的权衡。

**🔧 技术方法**

基于PPO的多目标强化学习，使用奖励层、优势层、损失层和PCGrad聚合策略；采用轴扫、PRR矩阵、超体积、期望效用等度量进行评估。

**📊 数据集**

在ProtoMotions和MimicKit仿真人形控制框架的五维风格偏好（能量、臂摆、脚抬高、步宽、平滑度）上进行训练与评估，另外在MetaDrive驾驶域做预实验。

**📈 对比分析**

通过Hypervolume、Expected Utility、Self‑PRR、Cross‑PRR、选择性、Rank Accuracy、任务奖励与跌倒率等多维度指标对四种聚合策略进行比较；奖励层聚合提供最大可调范围和最优效用，但跨目标泄漏最严重；优势/损失层和PCGrad在ProtoMotions上提升选择性，MimicKit则在覆盖度或精度上表现更优，但未出现单一最优策略。

**⚠️ 局限性**

仅在两套仿真人形环境、有限风格维度上验证；未解决响应-精度权衡；缺乏对分布外或动态偏好变化的鲁棒性评估；未转移到真实机器人系统。

---

## 188. Translation and data ethics

**arXiv ID:** 2610.28710 | [PDF](https://arxiv.org/pdf/2610.28710v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 189. SAFTER: Safety-Aware Filtering Through Explicit Recoverability for Robust Reinforcement Learning under Distribution Shift

**arXiv ID:** 2610.29149 | [PDF](https://arxiv.org/pdf/2610.29149v1)

**作者:** Benedictus C. G. Cinun `[一作]` (University of Houston), Bin Hu `[通讯]` (University of Houston)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在离散动作的强化学习中提出SAFTER框架，利用可微分的备份安全滤波器嵌入Soft Actor-Critic（SAC），实现对未来可恢复性的安全约束；

**💡 创新点**

创新点在于将当前状态安全、未来备份安全、终端可恢复性以及输入可接受性通过LSE组合成单一可微的线性约束，并给出闭式最小介入投影，既消除了在线QP求解的开销，又使梯度可通过滤波器传播；

**🔧 技术方法**

使用的技术包括控制障碍函数（CBF）与备份CBF（bCBF）、平滑最小（LSE）组合、闭式投影求解、可微安全层、SAC强化学习、LiDAR感知、Unitree Go2高层控制；

**📊 数据集**

实验数据集包括MuJoCo仿真中的四足机器人导航任务以及真实世界的两张Unitree Go2地图（两种障碍配置）；

**📈 对比分析**

与Vanilla SAC、CBF–QP/CF、bCBF–LSE–QP、SEA–Nav、CBF–RL等方法比较，SAFTER在训练集上90%成功率可在约4.0M转移内达到（比Vanilla SAC少30%转移、比CBF基线少34%转移），在低、中、高几何分布偏移环境中分别达到98.8%、88.6%和61.4%成功率，真实世界两张地图上分别为73.3%和76.7%，且运行时延比直接bCBF–QP快高达990倍；

**⚠️ 局限性**

局限性包括：需要预先设计备份控制器和安全模型，主要针对平面导航且仅在受限动作下验证，未对不平坦地形或动态障碍进行测试，LSE近似可能在极端情形下引入误差，且在更大几何偏移时性能仍显著下降。

---

## 190. Gate Initialisation versus Recurrent Dropout and Zoneout in Learning Long Dependencies: A Controlled Study of LSTM and GRU Networks on Synthetic Tasks

**arXiv ID:** 2610.29034 | [PDF](https://arxiv.org/pdf/2610.29034v1)

**作者:** Oleksandr Kholodniak `[一作]` (National Aerospace University Kharkiv Aviation Institute), Oleksandr Prokhorov `[通讯]` (National Aerospace University Kharkiv Aviation Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在单层LSTM与GRU网络中，将门控初始化（传统 vs chrono）与多种递归正则化（dropout、zoneout）在720次实验中进行因子设计，评估其对学习长依赖任务的影响。

**💡 创新点**

创新点在于首次系统比较门控初始化与不同正则化位置的交互效应，并尝试用早期梯度传播（输入敏感性斜率）预测学习成功率。

**🔧 技术方法**

使用的技术包括一层64维LSTM/GRU、Adam优化、梯度裁剪、不同位置的dropout/zoneout、chrono初始化、梯度探针、离散时间风险模型与L2惩罚的逻辑回归预测。

**📊 数据集**

数据集为两类合成任务：加法（adding）和延迟回忆（delayed recall），每个任务在多种序列长度下生成全新或完整的训练数据。

**📈 对比分析**

比较方法为在不同初始化、正则化、任务、长度条件下统计达到验证阈值的次数与平均更新步数；结果显示chrono初始化在所有条件下均能在5000步内成功，dropout多为负面影响，zoneout仅在部分情形下加速。早期梯度斜率对预测帮助有限，仅在特定条件下与种子关联。

**⚠️ 局限性**

局限性包括仅使用单层网络、固定学习率与正则化率、仅两类合成任务、未评估泛化能力、梯度探针为最终隐藏状态投影而非完整雅可比、对不同seed共用导致统计依赖。

---

## 191. Rethinking Underwater Image Enhancement: Learning through Text-Guided Degradation Curriculum

**arXiv ID:** 2610.29569 | [PDF](https://arxiv.org/pdf/2610.29569v1)

**作者:** Hongri Tian `[一作]` (Hong Kong University of Science and Technology), Lei Zhu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用文本引导的扩散模型生成多样化的海水衰减图像，并结合逐步难度课程和单步传播引导的扩散恢复模型，提升海下图像增强效果。

**💡 创新点**

创新点在于：①提出文本条件衰减增强（TCDA）实现可控衰减生成；②设计渐进难度课程以稳定训练；③引入传输引导单步扩散（TG‑Diff）以降低推理成本并提升恢复质量。

**🔧 技术方法**

采用文本引导扩散模型（FLUX/Stable Diffusion）、流匹配训练、LoRA微调、UDCP传输图估计、文本编码（T5‑XXL/CLIP）、单步扩散恢复等技术。

**📊 数据集**

使用UIEB、EUVP、LSUI、3D TURBID等公开海下数据集进行衰减对生成，并在此基础上扩增约10,000对训练样本。

**📈 对比分析**

与UIEC^2‑Net、SCNet、FiveA+、WF‑Diff、CE‑VAE、Phaseformer等现有方法在UIEB、C60、U45、LSUI等基准上进行对比，指标PSNR、SSIM、LPIPS、UCIQE、UIQM均实现或逼近最优水平。

**⚠️ 局限性**

局限性包括：合成衰减与真实环境仍存在差距；对极端低光或高深度场景的适应性尚未充分验证；模型在视频序列上的时序一致性和实时推理性能仍需进一步提升。

---

## 192. How Far Can the Gemma 4 Speech Encoder Go Without Changing Its Weights? Saudi and Gulf Arabic Speech Recognition with a Frozen and an Adapter-Augmented Encoder

**arXiv ID:** 2610.28673 | [PDF](https://arxiv.org/pdf/2610.28673v1)

**作者:** Amr Abdelkhalik `[一作]` `[通讯]`, Amr Abdelkhalik

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

利用Gemma 4/EmbeddingGemma 2共享的305M音频编码器，在不更改其权重的情况下，对沙特和海湾阿拉伯语电话语音进行识别，先训练仅包含22.6M参数的CTC识别头，然后在其上加入3.1M低秩适配器、41.5M参数的混合CTC/注意力读取器，并实现联合CTC/注意力解码。

**💡 创新点**

创新点在于证明一个大型多模态模型的音频编码器（已被广泛部署）可以在冻结状态下为阿拉伯语电话语音提供与2B级专用识别器相当的字符错误率，并通过少量适配器和联合解码实现近似等效的词错误率，且大幅降低了计算成本与显存占用。

**🔧 技术方法**

主要技术包括：冻结的Conformer音频编码器、层级特征混合、CTC/注意力双头读取器、低秩适配器（rank 16）、自适应CTC、联合CTC/注意力搜索与子词语言模型融合、以及针对GPU的半精度并行化优化。

**📊 数据集**

使用的数据集包括：沙特电视语料SADA、海湾阿拉伯语Casablanca、海湾阿拉伯语Emirati、沙特客服呼叫录音、海湾客户服务样本、公开盲测呼叫中心样本以及内部真实电话录音；所有数据均经G.711 μ‑law模拟窄带条件下处理。

**📈 对比分析**

与Cohere Transcribe Arabic（2.07B）和Nemotron 3.5（0.6B）等基线对比，冻结模型在字符错误率上与大模型几乎无显著差异，在CPU上速度提升约10‑30倍；适配模型在公开客服呼叫和清晰Emirati语音上的词错误率与大模型可区分（统计上不显著），在电话编码Emirati语音上优于大模型，且所有人标记集的字符错误率显著更低；在长形式SaudiTalk语音上仍略逊。

**⚠️ 局限性**

局限性包括：仅在单次随机种子下训练，缺乏方差评估；适配器与额外训练数据共同提升，难以单独归因；评估中使用机器生成标签的训练集可能导致模型部分蒸馏；参考文本多基于机器草稿，可能偏向生成系统；实验受限于特定GPU/CPU硬件；仅评估了Gemma 4编码器，未与Whisper或wav2vec等冻结编码器对比。

---

## 193. MA-UITD: Offline Multi-Agent Reward Inference for Characterizing Continuous Social Behavior

**arXiv ID:** 2610.29237 | [PDF](https://arxiv.org/pdf/2610.29237v1)

**作者:** Zeyuan Zhang `[一作]` (Chinese Academy of Sciences), Yusi Chen `[通讯]` (Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种离线多智能体逆强化学习方法MA‑UITD，能够从固定的联合轨迹中直接推断各智能体的显式奖励函数；

**💡 创新点**

创新点在于引入预判条件的后继特征（belief‑conditioned successor features）和无模型的逆时序差分学习，完全避免对环境动态、策略优化或额外交互的需求；

**🔧 技术方法**

采用信念条件化后继特征、噪声对比估计（NCE）、离线逆时序差分（UITD）以及对数斯谬响应策略模型；

**📊 数据集**

在三种仿真环境（Simple Spread、Simple Push、Gems）以及实验收集的老鼠社会觅食轨迹上进行评估；

**📈 对比分析**

与MA‑AIRL和MAMQL进行对比，MA‑UITD在策略重现、奖励恢复（PCC/SCC）以及连续动作场景中的性能均优于基线；

**⚠️ 局限性**

局限在于需要从演示中估计精准的预测信念，且对未观测状态的奖励依赖正则化与模型先验；此外，当前方法仅适用于少量智能体。

---

## 194. MedBOE: Optimization-Based Estimation of Complete Patient Body Models from Whole-Body Medical Imaging

**arXiv ID:** 2610.28985 | [PDF](https://arxiv.org/pdf/2610.28985v1)

**作者:** Hoijoon Jung `[一作]` (University of Sydney), Jinman Kim `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

本文提出了一种无训练的优化框架 MedBOE，用来从缺失部位的全身医学影像中自动估计完整的人体表面模型。

**💡 创新点**

创新点在于将 SMPL 形体模板与属性引导的多阶段优化结合，在医学成像场景下实现缺失部位的几何一致补全，并在不依赖大规模训练数据的前提下保持高精度。

**🔧 技术方法**

技术实现包括三模块管道：体表重建（nnU‑Net+marching cubes+Deep Virtual Marker）、全局 SMPL 参数拟合（含 E_Sparse、E_Dist、E_Prior、E_Weight、E_Contact、E_Dense 等属性）以及缺失区域估计与表面变形（粗细两阶段，使用 Chamfer、Laplacian、Volume 约束等）。

**📊 数据集**

使用公开数据集 AutoPET（PET‑CT）和 HIT（MRI）进行评估，并通过模拟裁剪产生缺失模型进行量化比较。

**📈 对比分析**

与 IF‑Net、ShapeFormer、PCN、ArtEq 等基线方法对比，MedBOE 在体积误差 (RBE,RVE)、IoU、Chamfer (CD) 与 Hausdorff (HD) 等指标上均取得最佳或次佳表现；由其估计的体型指标 (BHM) 的误差均低于 2%，但整体推算时间显著更长。

**⚠️ 局限性**

局限性包括：依赖预训练的 SMPL 形体模型，导致对医学成像中姿态差异的适配有限；缺失部位补全受 SMPL 形状空间约束，局部细节可能不足；实验使用模拟裁剪而非真实扫描缺失；推算速度较慢，不适合实时应用。

---

## 195. Inconsistency in the analytic hierarchy process under data uncertainties

**arXiv ID:** 2610.29418 | [PDF](https://arxiv.org/pdf/2610.29418v1)

**作者:** Przemysław Rutka `[一作]` (University of Warsaw), Ryszard Smarzewski `[通讯]` (Military University of Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在多准则群体决策中，考虑不确定性下的比较矩阵不一致性概率，并给出统一、指数、正态扰动下的解析公式

**💡 创新点**

提出基于Steiner三元组系统的O(n²)复杂度Koczkodaj不一致性指标，显著提升大规模矩阵的计算效率

**🔧 技术方法**

使用概率测度、组合设计（Steiner三元组）、高阶积分函数（E₁、arctan-指数积分）以及Gauss–Laguerre/Hermite数值积分

**📊 数据集**

未使用具体真实数据集，而是通过理论分布（均匀、指数、正态）和Monte Carlo仿真对照验证公式

**📈 对比分析**

与传统10⁷次Monte Carlo模拟相比，解析公式在相同精度下误差≤0.0004，计算速度提升数十倍以上

**⚠️ 局限性**

仅适用于满足Steiner条件（n≡1,3 mod 6）的矩阵，其余情况只能给出上界；对非独立条目的假设可能导致误差

---

## 196. Inheritance as an Admission Test for LLM Prompt Merge Operators

**arXiv ID:** 2610.29023 | [PDF](https://arxiv.org/pdf/2610.29023v1)

**作者:** Zan-Kai Chong `[一作]` (Kwansei Gakuin University), Bryan Ng `[通讯]` (Victoria University of Wellington)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究LLM代理的merge操作符对prompt演化的传递性与进步评估，提出基于孩子相对自身零剂量基准的入选测试，并给出闭环验证框架；

**💡 创新点**

创新点在于：①提出入选测试（对比孩子与其自身0剂量子代的移动量，使用置信下限判定）来判断merge是否真正传递特性；②通过开环与闭环实验验证入选测试对闭环表现的预测能力；

**🔧 技术方法**

采用gemini‑3.1‑flash‑lite模型进行prompt生成与评估；利用行为特征向量、Wilcoxon符号秩检验、置信下限、协方差分析等统计方法；实现四种merge算子（free rewrite、tactic append、template append、constrained rewrite）；

**📊 数据集**

使用结构化的“Deal or No Deal”谈判游戏脚本（八个情境）进行行为采样，采样16对（或循环中的12种seed prompt）与六个冻结seed；

**📈 对比分析**

对单合并进行开环实验：计算Δs与置信下限，判定入选；闭环实验跑三次40代，比较各算子在基准seed表现上的平均提升。结果显示tactic append与template append均能超过最佳种子，free rewrite未能，证明入选测试能预测闭环效果；

**⚠️ 局限性**

局限性包括样本量有限、仅使用单一模型与单一任务、未评估效率、对高方差的偏差敏感。

---

## 197. Preserving Operator Intent Under Actuation Constraints: Teleoperation of Manipulators with Coupled Transmissions

**arXiv ID:** 2610.29565 | [PDF](https://arxiv.org/pdf/2610.29565v1)

**作者:** Xuhan Miao `[一作]` (Beijing University of Technology), Chenguang Yang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种考虑驱动器约束的遥操作方法，统一考虑了主控手臂的重力补偿、关节映射以及差分手腕控制。

**💡 创新点**

创新点在于：在执行器空间做意图保持的投影，协调电机饱和时的指令投影；引入连续可行参考更新与角度展开，抑制回收瞬态；并通过实验验证了重力补偿对差分手腕的显著提升。

**🔧 技术方法**

采用了动力学建模、重力补偿、传动耦合矩阵、可执行性投影的二次规划、连续可行参考更新、角度展开、PID 电机控制等技术。

**📊 数据集**

实验数据来自真实主从遥操作系统的关节位置/速度记录，未使用公开数据集，而是自己采集的机器人轨迹数据。

**📈 对比分析**

通过与无重力补偿及单关节补偿的消融实验对比，显示平均速度标准差从 22.41 降至 5.91（约 73.6% 降低），单关节补偿仅能提升局部性能，说明完整方法的性能优势。

**⚠️ 局限性**

局限性包括：方法高度依赖精确的动力学与传动模型，针对差分手腕的特定硬件实现，网络延迟恢复策略仍有限，且对更复杂场景的可扩展性未充分验证。

---

## 198. Causal world models under physical parameter shifts

**arXiv ID:** 2610.29088 | [PDF](https://arxiv.org/pdf/2610.29088v1)

**作者:** Chu Lin `[一作]` (City College of New York), Yi Sun `[通讯]` (City College of New York)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `51c0528b-f690-4182-ae60-bb5f046c276c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于质量线性关系的机器人世界模型，结合固定的手指局部依赖、已知PD控制器或学习的扭矩网络，并通过冻结的干预导出输入掩码限制输入。

**💡 创新点**

创新点在于：① 通过显式质量系数把物理参数融入惯性与偏置项，使得超出训练范围的质量也能被利用；② 将已知PD控制器与可学习残差分离，形成两种机制；③ 用干预导出掩码实现结构化输入约束，显著提升合成OOV预测；④ 在CausalWorld共享100步预测任务中证明在未知质量下模型仍能保持甚至提升精度。

**🔧 技术方法**

技术包括：结构化动力学建模（质量线性惯性与偏置）、已知PD控制器与学习残差网络、干预导出稀疏输入掩码、半隐式积分、批量训练与滚动预测、与递归RWM和PETS式概率集成器的对照实验。

**📊 数据集**

使用两个数据集：① 合成基准（含32输入、6输出、17干扰项的二维系统），② CausalWorld（推送与抓取任务，12质量参数的机器人与对象状态）。

**📈 对比分析**

比较方法：对照基线（MLP、无质量或固定平均质量输入、RWM、PETS概率集成）在ID与OOD质量范围内进行100/150步滚动预测，评价指标为平均R²、NRMSE、对象位置RMSE。结果显示：冻结掩码将合成OOV RMSE从1.769降至0.697；在CausalWorld中，已知PD模型的实际质量输入在OOD下R²提升至0.727（vs 0.609），所有10个跑均正增益；学习扭矩模型与最强RWM相当；PETS+掩码显著提升环境迁移预测。

**⚠️ 局限性**

局限性：仅考虑已知质量和已知低级PD控制器；手指局部依赖是预设而非学习得到；预测时长限制在0.4–0.6 s；未对不确定参数估计、长时间部署或控制任务进行验证；对其他机器人结构或动态模型的推广性尚待进一步研究。

---

## 199. How Does Orthogonalization Adapt to the Neural Network Hessian Structure? A Gradient Self Outer-Product Analysis at Initialization

**arXiv ID:** 2610.28975 | [PDF](https://arxiv.org/pdf/2610.28975v1)

**作者:** Shenyang Deng `[一作]` (Dartmouth College), Yaoqing Yang `[通讯]` (Dartmouth College)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究阐释了 Muon 算法在随机初始化时的隐式预条件器与神经网络 Hessian 的行块对角占优结构的匹配，并给出了四类典型模型的渐近解析。

**💡 创新点**

创新点在于推导了过参数化权重矩阵导致 G G^T 对角支配的定量渐近规律，证明 Muon 的预条件器与 Hessian 的结构一致，从而解释了 Muon 在大规模模型中的优越性。

**🔧 技术方法**

主要采用了期望分析、Wishart 分布矩阵刻画、Stein 引理与弧角核、解析矩阵分布推导、数值模拟以及语言模型初始化实验等技术。

**📊 数据集**

实验数据主要来自随机高斯初始化与合成数据，以及公开的 FineWeb-Edu 数据集用于语言模型初始化评估。

**📈 对比分析**

通过计算对角与非对角均值比率 R 来验证理论预测，实验显示 R 随隐藏维度或模型宽度按理论速率增长，说明 Muon 在大规模模型中更符合 Hessian 结构，具有更好的收敛表现。

**⚠️ 局限性**

局限性在于理论仅针对初始化期望，未对训练动态给出解析；未探究大规模模型是否出现结构退化等潜在问题。

---

## 200. Is Anyone Enforcing the Cartel? What Reinforcement Learning Teaches LLM Agents in Markets

**arXiv ID:** 2610.28787 | [PDF](https://arxiv.org/pdf/2610.28787v1)

**作者:** Hieu Tran `[一作]` (University of Massachusetts Amherst), Hong Yu `[通讯]` (University of Massachusetts Amherst)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在价格设定与信息交易市场中，基于大语言模型（LLM）的代理如何通过强化学习（RL）学习策略，并提出一种基于对比复制的审核方法，检测代理是否真正执行反作弊（enforcement）以形成卡特尔。

**💡 创新点**

创新点在于：① 引入对手偏离复制测试，直接评估卡特尔执行而非仅观察结果；② 在标准化的Kyle交易市场和Bertrand定价博弈上构建闭式基准；③ 对比多种代理（Q-learning、GRPO、PPO、提示式LLM）在同一环境下的学习与执行差异。

**🔧 技术方法**

主要技术包括：离散动作Q学习、基于梯度的GRPO和PPO强化学习、LLM提示（prompting）与在对话式环境中的推理；以及通过共享随机种子实现的复制对照实验。

**📊 数据集**

数据集：使用自定义的模拟市场数据——Kyle模型的资产价值与噪声订单序列、Bertrand游戏的价格与需求曲线；实验涉及多轮（60–200期）交易或价格博弈，随机种子生成多种对照。

**📈 对比分析**

比较方法：对比不同代理在同一基准市场中的利润、卡特尔指数、以及在复制测试中的响应强度；结果显示：Q-learning在价格博弈中能学到惩罚，GRPO和PPO在自我对弈中往往忽视对手或仅复制价格，提示式LLM极少惩罚；整体卡特尔执行稀疏且往往只在特定状态下成立。

**⚠️ 局限性**

局限性：① 市场模型过于简化，缺乏真实订单簿、库存与多资产特征；② 仅评估了少数几种LLM及RL算法，未覆盖更大模型或更先进的多体学习；③ 审核依赖共享随机种子，API服务不完全可控；④ 提示词与消息长度对结果敏感，可能影响可推广性。

---

## 201. From Canon to Nikon: Camera Domain Shift and Potential Equity Implications in Retinal AI

**arXiv ID:** 2610.29104 | [PDF](https://arxiv.org/pdf/2610.29104v1)

**作者:** Lisa Y. W. Tang `[一作]` `[通讯]` (University of British Columbia), Lisa Y. W. Tang (University of British Columbia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统评估了不同相机平台（Canon CR‑2 与 Nikon NF5050）对 12 个视网膜疾病分类任务、4 个输入分辨率以及 5 个下游分类器的性能影响，并采用冻结的 Retinal foundation model 进行跨相机的性能比较。

**💡 创新点**

创新点在于：① 在多任务、多分辨率、多分类器的设定下，对相机相关域漂移进行细粒度分析；② 通过使用 RETFound‑Green 作为冻结特征提取器，显著降低实验计算成本；③ 强调相机平台是被忽视的性能变异源，对 AI 部署具有重要意义。

**🔧 技术方法**

使用 RETFound‑Green 视网膜基础模型提取特征，随后在同一特征空间下训练 SVM、kNN、HGB、LR、RF 等下游分类器，并在不同分辨率（224、392、672、952 像素）下进行实验；评估指标包括 AUROC（二分类）和 QWK（分级任务），采用患者级划分和 95% bootstrap 置信区间。

**📊 数据集**

数据集为巴西多标签眼科数据集 BRSET，包含 8,524 名患者的眼底照片，其中 65% 来自 Canon 系统，35% 来自 Nikon 系统，且 约 16% 患者患有糖尿病。

**📈 对比分析**

实验方法为在 Canon 训练并内部验证后，直接在 Nikon 作为目标域进行测试；结果显示不同疾病任务表现差异显著，某些任务（如视网膜病变、疤痕）跨相机性能相近，而出血、HTN 视网膜病变、DR 分期等任务则显著下降；分辨率提升或更复杂分类器并未显著缓解相机相关性能差异。

**⚠️ 局限性**

主要局限包括仅评估两种相机平台且与图像质量高度相关，无法完全分离相机效应与质量差异；缺乏更多相机类型和数据集的验证；未尝试域适应或图像质量控制等方法来减轻观测到的域漂移。

---

## 202. Dissimilarity-Based Markov Decision Processes for Stabilizing State Distributions

**arXiv ID:** 2610.29024 | [PDF](https://arxiv.org/pdf/2610.29024v1)

**作者:** Xun Shen `[一作]` (Tokyo University of Agriculture and Technology), Katsuki Fujisawa `[通讯]` (Institute of Science Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出分布差异化马尔可夫决策过程（DMDP）框架，将策略直接作用于状态分布，并以分布相似度为阶段成本，实现对随机系统状态分布的稳定控制。

**💡 创新点**

创新点在于：①通过对分布的有限维参数化把无限维问题转化为确定性参数动力学；②给出最优策略的Lyapunov稳定性保证；③设计学习、估计误差下的稳定性证书；④将证书与PPO/SAC结合，实现分布反馈和目标匹配；⑤在HalfCheetah任务上验证效果。

**🔧 技术方法**

使用技术包括：分布动力学与分布相似度（KL/Wasserstein）、参数化密度模型、成本旋转、Lyapunov函数与类𝒦函数、近似动力学模型、在线参数估计、PPO与SAC强化学习、滑动窗口分布估计。

**📊 数据集**

主要实验数据集为OpenAI Gym的HalfCheetah环境，关注物体前进速度与躯干俯仰角的分布。

**📈 对比分析**

与SAC、SAC‑Lag、Distributional SAC、Implicit DMDP等方法对比，DMDP‑Lag（PPO/SAC）在分布相似度、奖励收益和可靠性（返回≥阈值的概率）等指标上均表现最佳。

**⚠️ 局限性**

局限性包括：未能为高维学习策略计算Lyapunov证书；参数化假设（闭合性）在实际任务中难以满足；误差保守导致残差集较大；仅在单一任务上验证，需进一步推广到更复杂的多模态分布控制。

---

## 203. PhyMo: Learning Physical Dynamics with Accurate and Continuous Motion from Multi-View Videos

**arXiv ID:** 2610.29107 | [PDF](https://arxiv.org/pdf/2610.29107v1)

**作者:** Shangjia Liu `[一作]` (Hong Kong Polytechnic University), Bing Wang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `14d48e9d-0069-4ad9-996a-1d5968216998` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `a8e75ba4-7a2d-4153-b003-06c94533add0` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 PhyMo 框架，从多视角视频中学习动态场景的几何、外观与物理运动状态；

**💡 创新点**

创新点在于引入运动感知轨迹表示抑制外观捷径，并通过双向动力学连续性约束学习连续的速度与加速度，实现几何精确插值与连续运动外推；

**🔧 技术方法**

使用 3D 高斯点云表示、形变场 F_defo、运动网络 M_θ、Runge‑Kutta 2 级传播、双向动力学连续性损失、逐步时间扩展训练策略；

**📊 数据集**

在 NVIDIA Dynamic Scenes、TRACE、FreeGave‑GoPro、PhysInOne 四个动态场景数据集上进行评估；

**📈 对比分析**

与八种基线（如 TiNeuVox、4DGS、DefGS、ModecGS、FreeGave、TRACE、ODEGS、NVFi）比较，结果显示 PhyMo 在 PSNR、SSIM、LPIPS、Chamfer Distance 等指标上均位居第一，尤其在动态几何精度和未来预测连续性上显著优于其它方法；

**⚠️ 局限性**

局限性包括对时间步长的依赖（需适当选择 Δt）、对复杂物理交互的建模仍有限、训练过程对多视角摄像头和 SfM 初始化的依赖，以及对极端快速或高度非线性运动的鲁棒性尚待进一步验证。

---

## 204. DGCM-SSFA: Dual-Granularity Cross-Modulation and Sparse Strip Frequency Attention for Efficient 3T-to-7T MRI Synthesis

**arXiv ID:** 2610.29141 | [PDF](https://arxiv.org/pdf/2610.29141v1)

**作者:** Zhichun Fan `[一作]` (Huizhou University), Zhonghua Xie `[通讯]` (Huizhou University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出一种 11M 参数的多域注意力网络 DGCM-SSFA，用于将常规 3T 扫描转换为 7T 等效 MRI，提升图像细节与结构保真度。

**💡 创新点**

创新点在于将 Sobel 边缘引导的空间注意、频率感知通道注意与稀疏条带全局注意三种互补机制融合为一个轻量级模块，显著降低参数并提升合成质量。

**🔧 技术方法**

核心技术包括 U‑Net Transformer 背骨、Dual‑Granularity Cross‑Modulated Feature Modulator (DGCM)、Sparse Strip Frequency Attention (SSFA)、FFT 频域调制、交叉门控融合以及全局残差学习。

**📊 数据集**

实验使用公开的 UNC 与 BNU 3T‑7T 影像数据集，涵盖 T1w 与 T2w 两种成像序列。

**📈 对比分析**

与 CNN、GAN、Transformer 及状态空间模型等八种方法对比，DGCM‑SSFA 在 BNU T2w 上以 0.8431 的 SSIM 位居第二，参数量仅为 11M，性能优于多数大模型。

**⚠️ 局限性**

局限性包括仅采用 2D 切片处理未利用体素一致性、对病灶案例缺乏验证，以及缺少对抗/感知损失等更丰富的优化目标。

---

## 205. Signals and Queries: Token-Efficient Agentic Context Discovery

**arXiv ID:** 2610.29326 | [PDF](https://arxiv.org/pdf/2610.29326v1)

**作者:** Yam Marcovitz `[一作]` `[通讯]`, Yam Marcovitz

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 Signals and Queries 方法，通过 LLM 离线生成每个策略的激活信号与反信号，并用逻辑回归训练独立分类器，在线时仅做一次嵌入和点积即可完成策略的相关性过滤。

**💡 创新点**

创新点在于：① 用 LLM 生成信号/反信号实现离线训练，避免在线 LLM 调用；② 每个策略拥有独立决策边界，支持可调阈值与召回优先；③ 对近义策略的硬负样本对抗学习，显著提升了对语义相近策略的区分；④ 与传统基于余弦相似度的检索相比，精度更高、成本更低。

**🔧 技术方法**

核心技术包括：LLM 生成示例、句子嵌入（Gemini、OpenAI）、向量中心化与归一化、逻辑回归二分类、最大池化求对话与单轮投射、阈值调节。

**📊 数据集**

实验使用 τ²-bench 航空对话数据集，共 254 条子序列，36 条政策（其中 16 条高频）及 9,144 条 LLM 判定标签，此外使用 10 条信号与 10 条反信号作为示例。

**📈 对比分析**

与 top‑K、cosine 阈值检索基线比较，Signal 与 Queries 在两种嵌入提供者（Gemini 与 OpenAI）下，在所有召回水平上均取得更高精度；在召回≥90% 时精度约 30%，且在线成本比每策略 LLM 判定低约 10⁵ 倍。

**⚠️ 局限性**

局限性包括：仅在航空领域与仿真对话上验证；依赖 LLM 判定的准则，缺乏人类标注一致性；召回优先导致精度相对较低；未考虑策略间依赖关系；对长会话累积效果未完全评估；示例生成的覆盖度与硬度是主要瓶颈。

---

## 206. Carving Radiology Reports at Their Evidential Joints: Anatomical Region Sets as Generation Units

**arXiv ID:** 2610.29165 | [PDF](https://arxiv.org/pdf/2610.29165v1)

**作者:** Jinwei Yang `[一作]` (University of Electronic Science and Technology of China), Ling Wang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 RadCarve 框架，将放射学报告的生成单位直接定义为与声明对应的解剖区域集合，并在图像中同时预测单位的存在与组成，随后用对应区域的视觉证据生成文本。

**💡 创新点**

创新点在于：①使用现有声明–区域对应关系直接构建生成单位，避免预定义主题或单区域划分；②引入 composition‑specialized query banks 和迭代组合细化，使生成单位与视觉证据绑定，并能联合使用多区域证据生成更完整的临床文本。

**🔧 技术方法**

技术手段包括 Transformer 编码解码的集合预测框架、Rad‑DINO 与检测器提取区域特征、composition‑specialized query banks、迭代组合细化以及 GPT‑2 生成器；训练时采用多任务损失对齐。

**📊 数据集**

使用 MIMIC‑CXR 与 Chest ImaGenome 注释进行训练；在 Chest ImaGenome 测试集和 PadChest‑GR 外部测试集上评估。

**📈 对比分析**

与 MAIRA‑2、CURE、RGRG、ChEX、PriorRG 等基线比较；在单位生成、绑定恢复和完整报告生成任务中，RadCarve 在 CXRFES、CheXbert F1 等临床指标上领先，并在参数规模上显著更小；在外部 PadChest‑GR 上亦取得最高临床 F1。

**⚠️ 局限性**

局限性包括：监督来源为自动构建的声明–区域对应，可能包含噪声；仅使用单张前视图，缺乏多视角、历史或临床背景，限制了对时间变化和对比描述的能力。

---

## 207. Truncated Out-of-Bag Probability Gain: A Noise-Robust Criterion for Random-Forest Wrapper Feature Selection

**arXiv ID:** 2610.29195 | [PDF](https://arxiv.org/pdf/2610.29195v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 208. Harness Scaling for Generalizable Agentic Reinforcement Learning

**arXiv ID:** 2610.29458 | [PDF](https://arxiv.org/pdf/2610.29458v1)

**作者:** Fei Bai `[一作]` (Renmin University of China), Ji-Rong Wen `[通讯]` (Renmin University of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过在强化学习训练中使用多种执行 harness，提升了语言模型在不同 harness 配置下的鲁棒性和泛化能力。

**💡 创新点**

创新点是将 harness 视为可分解的五维配置空间，利用 strength‑2 orthogonal 设计选取具有低阶因子平衡的训练集合，并在任务‑harness 赋值中采用局部邻近策略，实现跨 harness 的结构化训练。

**🔧 技术方法**

使用了因素化 harness 空间、Hamming 距离、strength‑2 orthogonal 设计、PPO/GRPO 强化学习算法，以及 Qwen3‑4B‑Instruct/30B‑A3B‑Instruct 大模型。

**📊 数据集**

数据集包括 ClawGym‑Task 进行训练，评估使用 ClawGym‑Bench、PinchBench、JobBench‑Easy 三个基准，以及外部黑盒 harness Claude Code。

**📈 对比分析**

与单一 harness 训练、黑盒 SFT/ RL 等基线相比，Harness Scaling 在白盒 hold‑out 上平均提升约 9–12 Pass@1 点，在黑盒测试上提升 3–5 点，表现优于其他方法。

**⚠️ 局限性**

局限性在于需要手工设计 harness 维度与变换，训练规模受限于因子平衡约束，且在极端或全新 harness 组合上的泛化仍有限。

---

## 209. RISK-KV: Quality-Aware Selective KV Cache Reconstruction for Distributed LLM Inference

**arXiv ID:** 2610.29544 | [PDF](https://arxiv.org/pdf/2610.29544v1)

**作者:** Wang Yuekai `[一作]` `[通讯]` (Harbin Institute of Technology), Wang Yuekai (Harbin Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在分布式大语言模型推理中进行选择性 KV 缓存重建的方法 RISK-KV。

**💡 创新点**

创新点包括：1）使用跨层双侧 KV 锚点来预测目标层的隐藏表示并生成缺失 KV；2）通过实际 KV 替换测量的功能风险（NLL 变化）来驱动层级选择，而非仅凭数值误差或相似度；3）在风险排序路径上只需 2G 次评估即可近似最优配置；4）将预测器与原始 KV 投影线性结合，显著减少额外存储与计算。

**🔧 技术方法**

核心技术：跨层 KV 预测器（岭回归+折叠），RoPE 逆变换，组查询注意力（GQA），功能风险评估与质量门控（Bootstrap 上限）。

**📊 数据集**

使用 WikiText‑2 作为验证与测试数据，实验涵盖 Qwen2.5‑0.5B、Qwen2.5‑1.5B、TinyLlama‑1.1B 三个预训练模型。

**📈 对比分析**

与完整 KV 基线和 Token Replay 进行对比：在 512‑token 历史下，RISK-KV 可节省 18.2%–33.3% 的 KV，NLL 变化 <0.5%，且恢复延迟比 Token Replay 快 1.5–3.3 倍；在功能风险门控下可将预算违规率降至 0%（或低于 10%）并仍保持显著缓存节省。

**⚠️ 局限性**

局限性：尚未在真实无线跨节点迁移环境下验证；对更长上下文和更大模型的效果未评估；预测器折叠后仍需额外存储，且在极低预算下可能退回到完整 KV。

---

## 210. The Data Differentiator Framework (DDF): Quantifying Marginal Data Value with Information-Theoretic and Game-Theoretic Measures in an Agentic Evaluation Pipeline (SDSAE)

**arXiv ID:** 2610.28933 | [PDF](https://arxiv.org/pdf/2610.28933v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 211. A Strong MoE Is All You Need for Process Supervision

**arXiv ID:** 2610.29498 | [PDF](https://arxiv.org/pdf/2610.29498v1)

**作者:** Guanzhi Deng `[一作]` (City University of Hong Kong), Linqi Song `[通讯]` (City University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了RoCo框架，利用Mixture-of-Experts模型中的专家路由统计量生成步骤级信心，用于测试时的自我重思和自动过程奖励模型；

**💡 创新点**

创新点在于将MoE路由视为内在的过程监督源，构造组合路由信心（整体路由质量、头部路由质量与步内变异），并通过RoCo-Rethink和RoCo-PRM实现高效的推理改进与跨模型跨任务迁移；

**🔧 技术方法**

使用的技术包括：路由质量统计（M、H、D）与信心计算C=M+H−D；RoCo-Rethink基于最小信心步骤进行局部重新生成；RoCo-PRM训练基于路由得分的过程奖励模型；实验中对比Self‑Consistency、LECO、MC‑PRM、CPMI‑PRM等基线；

**📊 数据集**

使用的数据集涵盖数学推理（GSM8K、MATH500）、知识推理（MMLU、IFEval）、代码生成（HumanEval）以及逻辑推理（LogicBench）等六个任务；

**📈 对比分析**

通过与Vanilla、Self‑Consistency、LECO、MC‑PRM、CPMI‑PRM等基线的对比，RoCo‑Rethink在18个模型‑任务组合上平均提升1.5–4.5点；RoCo‑PRM在30个组合上平均提升1.4–2.5点，且能跨MoE与dense模型迁移；

**⚠️ 局限性**

局限性包括：对高质量MoE教师的依赖，路由信息质量直接影响PRM效果；仅在具备MoE路由的模型中自然产生，需要额外路由提取步骤；对多轮重思的阈值与收敛性未作深入探讨。

---

## 212. Sign Here: Attributing a Source for a More Trustworthy Software Supply Chain

**arXiv ID:** 2610.28794 | [PDF](https://arxiv.org/pdf/2610.28794v1)

**作者:** Derek Garcia `[一作]` `[通讯]` (Chainguard), Derek Garcia (Chainguard)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究通过要求包在分发时必须提供可构建的源码地址，利用源码属性来检测并防止软件供应链恶意包。

**💡 创新点**

创新点在于将“源代码可构建性”作为恶意检测指标，并在大规模数据集上系统评估其有效性，证明其高召回率且与现有安全扫描器可互补。

**🔧 技术方法**

主要技术包括：① 正则和半结构化元数据解析提取源码链接；② 通过 HTTP 状态码验证源有效性；③ 克隆仓库、检出对应提交并构建包，对比生成的 sdist 以确认“真实”源；④ 与三款主流扫描器（malcontent、grype、trivy）进行对比评估。

**📊 数据集**

使用 47,066 个唯一恶意包（来自五个公开恶意数据集）和 3,947,045 个 npm + 764,244 个 PyPI 包的快照，随机抽取 16,569 个正样本进行实验。

**📈 对比分析**

与 SOTA 扫描器对比，源码属性检测在最严格的“真实源”模式下召回率达到 0.999，防止 99.9% 恶意包；在“有效源”或“属性源”模式下召回率约 0.93，FPR 较高（最高 35%）。将该方法与 malcontent 结合，召回率提升至 0.973，精度和 F1 也得到提升；但若过度依赖 FPR 降低，召回率会下降 26‑43%。

**⚠️ 局限性**

局限性包括：① 高误报率导致噪声；② 依赖源码可访问性，许多正样本缺乏可构建源；③ 对仓库提交映射和构建环境缺失导致验证失败；④ 仅评估 npm/PyPI，结果可能不适用于其他生态；⑤ 数据集标签误差与时间漂移可能影响 FPR。

---

## 213. Towards Realistic Models of Change: A Causal Perspective on Decision-Making and Dynamical Systems

**arXiv ID:** 2610.28803 | [PDF](https://arxiv.org/pdf/2610.28803v1)

**作者:** Isacco Beretta `[一作]` `[通讯]`, Isacco Beretta

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

论文探讨了算法补救的现实性，特别是在时间感知因果关系方面。

**💡 创新点**

创新点在于提出了一种新的框架，能够在考虑时间因素的情况下进行因果推断和算法补救。

**🔧 技术方法**

使用了因果推断和时间序列分析的技术。

**📊 数据集**

数据集未具体提及，但可能涉及时间序列数据。

**📈 对比分析**

与传统的因果推断方法进行了比较，结果显示新方法在处理时间相关性时性能更优。

**⚠️ 局限性**

限制在于模型的复杂性和对数据质量的依赖，可能在某些情况下难以实现。

---

## 214. OpenMAP-HIE: Pathology Informed Score-based Diffusion Model for Unsupervised Neonatal Hypoxic-Ischemic Encephalopathy Lesion Segmentation

**arXiv ID:** 2610.29322 | [PDF](https://arxiv.org/pdf/2610.29322v1)

**作者:** Pengyu Kan `[一作]` (Johns Hopkins University), Kenichi Oishi `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种无监督的病理导向分数扩散模型OpenMAP-HIE，用以在ADC图像上定位新生儿缺氧缺血性脑病（HIE）病灶

**💡 创新点**

创新点在于通过比较无条件与条件噪声预测差异，并聚合多噪声级的方向一致性误差，利用HIE特定的ADC降低先验；同时不依赖病灶掩码训练，且可结合群体Z图后处理

**🔧 技术方法**

采用了扩散概率模型（DDPM）+ ControlNet 条件编码、分数（noise prediction）差异聚合、几何平均、均值平滑与Z图融合

**📊 数据集**

使用公开的BONBID‑HIE数据集（133名新生儿ADC及Z图），训练仅用无病灶切片，评估在44名测试子样本上

**📈 对比分析**

与多种无监督基线（DAE、ANDi、MAD‑AD等）以及有监督模型（Swin‑UNETR、UNet++）对比，ADC‑only OpenMAP-HIE在Dice、Hausdorff、MASD、NSD上优于所有无监督方法，且Dice接近有监督模型；Z图后处理进一步提升Dice至0.602、NSD至0.667

**⚠️ 局限性**

局限包括：对小/低对比病灶检测效果仍弱；对扫描设备、协议、时间点变化敏感；使用硬阈值方向约束可能抑制细微病灶；依赖群体Z图需准确配准；模型训练仅用2D切片，缺乏三维上下文

---

## 215. How Representational Priors Shape Generalization in World Action Models

**arXiv ID:** 2610.29168 | [PDF](https://arxiv.org/pdf/2610.29168v1)

**作者:** Junha Chun `[一作]` (Seoul National University), Taesup Kim `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了世界动作模型（WAM）中生成式骨干的表示几何，提出并量化了“nuisance orthogonality”（特征位移与在分布子空间正交程度）和“nuisance suppression”（策略训练后对噪声方向位移的抑制），并通过实验验证这些几何指标与WAM在视觉变异下的泛化性能之间的关系。

**💡 创新点**

创新点主要包括①用正交性度量表示先验的可解耦性，②提出抑制度量评估策略适应后表示空间的重塑，③证明这些几何指标可以在不训练策略的情况下预先诊断骨干的泛化潜力，并解释不同生成式骨干与策略之间性能差异。

**🔧 技术方法**

使用技术包括生成式预训练模型（图像编辑与视频预测）、Transformer中间层特征提取、SVD投影与正交性量化、归一化的抑制度量、PCA可视化及统计相关性分析。

**📊 数据集**

数据集主要为LIBERO-Plus诊断集（基于LIBERO模拟环境的同一状态下不同视觉外观的样本），涵盖光照、背景、传感器噪声、摄像机视角等四类视觉偏移。

**📈 对比分析**

对比方法为：先在不同骨干上计算nuisance orthogonality，再将其与在LIBERO-Plus任务中的成功率对齐；随后比较ImageWAM与FastWAM在噪声抑制指标（in‑distribution与orthogonal分量）上的表现。实验显示，高正交性对应的模型在多种偏移下成功率可达80–98%，而FastWAM在摄像机偏移下仅16.4%，ImageWAM在弱视角偏移下实现更强抑制并获得更高成功率。

**⚠️ 局限性**

局限性：研究仅覆盖LIBERO模拟环境的视觉偏移，未考虑语言指令或动作组合的泛化；实验为观察性，无法完全分离正交性与预训练目标的因果关系；仅评估视觉保留泛化，未在真实机器人或更广泛环境中验证。

---

## 216. AutoFraming: Synthesizing Dynamic Frames of C Code with LLMs

**arXiv ID:** 2610.29046 | [PDF](https://arxiv.org/pdf/2610.29046v1)

**作者:** Han Zhou `[一作]` (University of Missouri--Kansas City), Dianxiang Xu `[通讯]` (University of Missouri--Kansas City)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 AutoFraming，利用 C 代码的代码属性图（CPG）与大语言模型（LLM）生成 ACSL 规范，专注于动态帧（memory‑region）条件的合成；

**💡 创新点**

创新点在于将 CPG 提取的帧相关信息通过确定性投影生成帧骨架，LLM 仅负责外围补全与修正，从而显著提升帧规范的准确性与可重用性；

**🔧 技术方法**

采用代码属性图（CPG）进行程序分析、帧信息提取；通过确定性投影将信息转化为 ACSL 框架；利用 LLM 生成完整合同并在 Frama‑C/WP 反馈下迭代修正；

**📊 数据集**

评估基准包含 84 条真实系统函数（D_a）和 51 条学术示例（D_b）两大数据集；

**📈 对比分析**

与现有基线（Preguss、AutoSpec）对比，AutoFraming 在真实系统集上实现 92.9% 的无误帧规范，提升 179%；在学术集上达 96.1%，提升 15.7%；作为插件可将现有工具的帧规范错误率提升至 61.9%；

**⚠️ 局限性**

主要限制是仍需依赖 LLM 的生成，可能导致帧骨架被不当修改；对极端指针、运行时范围或复杂低级 C 特性的处理仍有限，且在极少数情况下验证器本身对帧范围的判断不足。

---

## 217. Gender Stereotypes Make Women Invisible: The Presence of Female Scientists in the Media

**arXiv ID:** 2610.28640 | [PDF](https://arxiv.org/pdf/2610.28640v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 218. A General Topological Symmetry Index for Critical Orbits In Neural Network Loss Landscapes

**arXiv ID:** 2610.28864 | [PDF](https://arxiv.org/pdf/2610.28864v1)

**作者:** Jingzhou Liu `[一作]` `[通讯]` (University of Texas at Dallas), Jingzhou Liu (University of Texas at Dallas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种拓扑对称性指数，用于在任意神经网络损失景观中识别和量化批判轨道的对称结构，解决了此前缺乏统一判据的问题。

**💡 创新点**

创新点在于将等变度理论与Euler环、Burnside环相结合，构造出整数值的对称性指数；该指数能够在不显式求解批判点的情况下，仅凭网络结构与损失函数的对称性来判定批判轨道的存在与对称类型，并能处理退化（即特征值为零）的情况。

**🔧 技术方法**

核心技术包括：等变度理论、Euler环与Burnside环的代数运算、梯度等变度（Equivariant Gradient Degree）的定义与性质、递归构造法、切片原理以及在正交G-表示与G-流形上的计算方法。

**📊 数据集**

实验示例使用了三种理论网络结构：softplus MLP、带有 O(3)×S₄ 对称的可变形网络（Steerable ENN）以及单头Transformer；并没有使用公开数据集，而是通过构造的目标函数和输入实现理论验证。

**📈 对比分析**

论文通过理论证明与符号计算给出对称性指数，并通过示例展示了指数中出现的对称轨道类型，未与其他数值优化方法进行实验性能比较，主要侧重于理论存在性和对称性结构的判定。

**⚠️ 局限性**

局限性包括：仅适用于紧Lie群的对称；对非紧对称（如ReLU的尺度对称）不适用；需要在每个子群上进行复杂的符号/代数运算；对大规模网络的指数计算仍具有较高的计算复杂度。

---

## 219. HyperDrone: Simulation-to-Reality Transfer at Millions of Frames per Second

**arXiv ID:** 2610.28950 | [PDF](https://arxiv.org/pdf/2610.28950v1)

**作者:** Jonas Eschmann `[一作]` (University of California, Berkeley), Giuseppe Loianno `[通讯]` (University of California, Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了高速Quadrotor模拟器HyperDrone，结合光线追踪渲染HyperLux，实现大批量视觉感知训练，并验证了从仿真到真实的yaw控制迁移。

**💡 创新点**

同时实现GPU端多旋翼动力学和光线追踪渲染，支持多种高保真视觉模态，性能超过现有20倍，且完全开源，支持无缝迁移到真实硬件。

**🔧 技术方法**

GPU上运行OptiX/Vulkan/Metal/WebGPU渲染，Ray-tracing硬件加速，物理基渲染(PBR)、运动模糊、抗锯齿；多线程GPU动力学模拟；卷积网络yaw估计；RAPTOR低层控制。

**📊 数据集**

使用ProcTHOR生成的10000个室内场景，支持glTF 2.0；以及在真实Crazyflie平台上用OpenMV摄像头收集的实时图像。

**📈 对比分析**

与Flightmare、VisFly、Aerial Gym、OmniDrones、Madrona等现有模拟器在相同RTX 4090上进行64×64分辨率批量渲染+物理吞吐率比较，HyperDrone在所有配置下至少比最快者高20×，在高保真设置下可达百万帧/秒。

**⚠️ 局限性**

目前不支持滚动快门、完整碰撞检测及低频控制；仿真仅验证yaw控制，仍需外部位置参考，未来需改进多模态控制与碰撞回避。

---

## 220. Multi-Scale Latent Temporal Compression Attention and Video-Text Cross Attention Fusion for Video Action Detection

**arXiv ID:** 2610.29176 | [PDF](https://arxiv.org/pdf/2610.29176v1)

**作者:** Yicheng Qiu `[一作]` (University of Electro-Communications), Keiji Yanai `[通讯]` (University of Electro-Communications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了多尺度潜在时序压缩注意力（MLTC-Attention）和视频-文本交叉注意力融合（VTCA-Fusion）两个模块，构建一种端到端的多模态视频动作检测框架；

**💡 创新点**

创新点在于：①将特征投影到潜在空间后按多尺度压缩再进行注意力计算，显著降低时序冗余并同时保留全局与局部信息；②在标准特征空间和潜在特征空间同时进行视频-文本交叉注意力，实现更深层次的跨模态对齐；③采用可学习文本提示提升文本语义表达；

**🔧 技术方法**

核心技术包括 Transformer 及其多头注意力、潜在空间投影、压缩注意力（Compression Attention）、Top‑k 过滤、门控融合、CLIP‑风格文本编码、可学习提示、Focal 损失与 IoU 损失；

**📊 数据集**

使用的公开数据集：视频异常检测任务使用 UCF‑Crime 与 XD‑Violence；视频动作检测任务使用 THUMOS14 与 ActivityNet‑1.3；

**📈 对比分析**

与多种基线（LAVAD、DMU、AnomalyCLIP、EventVAD、MGFN、STPrompt、ActionFormer 等）进行对比。实验表明在 UCF‑Crime 上获得 89.84% AUC，XD‑Violence 上 86.53% AP，THUMOS14 与 ActivityNet‑1.3 的 mAP 在所有 IoU 阈值下均优于对手；同时在效率上达到 83.77 FPS、42.1G FLOPs，优于现有方法；

**⚠️ 局限性**

局限性包括：①模型依赖于冻结的预训练视觉/文本编码器，可能限制进一步的性能提升；②仅使用视觉信息，未融合音频或其他模态；③对异常视频中标签噪声的处理依赖于 Top‑k 机制，仍可能存在误检；④需要较多算力进行多尺度压缩与交叉注意力，适配低算力场景仍有挑战。

---

## 221. Rethinking Time Series Tokenization from a Frequency Perspective

**arXiv ID:** 2610.28997 | [PDF](https://arxiv.org/pdf/2610.28997v1)

**作者:** Zelong Tian `[一作]` (Griffith University), Shirui Pan `[通讯]` (Griffith University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于频域的时间序列分词器 TimeFT，旨在消除现有分词方法所带来的频谱失真，保持趋势、周期性和季节性等时间特征的完整性。

**💡 创新点**

创新点在于：①从频域角度解析分词导致的失真并给出失真无条件；②设计了三阶段无失真分词流程（频域分块、频谱平移、Nyquist 采样）；③实现完全无参数、轻量化且可即插即用的分词器，显著提升多任务性能。

**🔧 技术方法**

核心技术包括：快速傅里叶变换（RFFT）与逆变换（IRFFT）、频率分块与零填充、频谱平移补偿相位、Nyquist 采样、密集采样和维度重塑；同时利用理论边界条件保证分块后的频谱满足无别名条件。

**📊 数据集**

实验覆盖三大任务：长周期预测（ETTh1/2、ETTm1/2、Weather、Electricity、Traffic）、分类（10个UEA数据集）和异常检测（SMD、MSL、SMAP、SWaT、PSM）。

**📈 对比分析**

与 PatchTST、PhaseFormer、PatchMLP、EntroPE、TimeMosaic 等主流分词器进行对比。实验表明，TimeFT 在预测任务上平均 MSE/MAE 下降约 6–10%，分类任务平均准确率提升至 72.29%（高于 71.33%），异常检测任务平均 F1 评分提升至 86.12%（高于 85.38%）。在作为插件使用时，提升幅度可达 9–10%，且计算开销不到 1.5% 的训练时间。

**⚠️ 局限性**

局限性包括：①对时序信号的带限假设要求（需满足 Nyquist 条件），可能对高频突变或非周期性信号效果不佳；②需要选择合适的分辨率（块大小 B_N）和采样步长 S，超参数选择仍需经验或交叉验证；③虽然参数化极低，但在极大尺度（N>10⁶）时 RFFT 的内存占用仍可能成为瓶颈。

---

## 222. From Weather to Enhanced Weathering: Weather Foundation Models for Carbon Dioxide Removal at Operational Scale

**arXiv ID:** 2610.28973 | [PDF](https://arxiv.org/pdf/2610.28973v1)

**作者:** Sasankh Munukutla `[一作]` (Terradot Soil, Inc.), Juan M Lavista Ferres `[通讯]` (Microsoft AI for Good)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

评估了五种天气基础模型作为编码器在增强岩石风化（ERW）测量中的表现，并提出TerraNova-ERW模型‑测量框架，可显著降低现场采样需求。

**💡 创新点**

首次对多种天气FM在ERW中的迁移能力进行全方位评估，发现Aurora 1.5最佳，并证明模型‑测量组合在少样本场景下能大幅减少采样量。

**🔧 技术方法**

利用预训练天气FM（Aurora、Pangu、SFNO、FCN3、GraphCast）生成全局潜在表示，采用梯形积分时间聚合，结合轻量宽解码器训练预测CO₂e/ha/yr。

**📊 数据集**

使用了192,511条自然风化序列与5,192条巴西与美国的ERW现场操作序列，按264田块构成的chronosequence。

**📈 对比分析**

通过留一田块（LOFO）对点级和田级指标（MAE、RMSE、Spearman）进行评估；TerraNova-ERW在点级MAE 3.06、RMSE 4.20、Spearman 0.29，在田级MAE 1.78、RMSE 2.57、Spearman 0.26，优于原始ERA5和随机打乱结果。

**⚠️ 局限性**

仅涵盖两国两种岩石类型，时间跨度有限，LOFO不覆盖全新地区；需要更多多地区、不同岩性和更长时间的数据以提升模型稳健性。

---

## 223. Extreme Low-State Quantized Transformer for Electricity-Load Forecasting

**arXiv ID:** 2610.28784 | [PDF](https://arxiv.org/pdf/2610.28784v1)

**作者:** Muhammad Sabbir Alam `[一作]`, Jayasimha Atulasimha `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并比较了 Post‑Training Quantization (PTQ)、Quantization‑Aware Training (QAT) 与从零训练 QAT 在时间序列预测模型中的效果，并提出了一种基于阈值的代表性预测轨迹选择策略。

**💡 创新点**

创新点在于：① 将阈值化技术与量化方法结合，显著提升了模型在不同数据集上的泛化性能；② 通过跨数据集评估验证了量化方法的鲁棒性；③ 开发了一套自定义的活动度指标与基准评估体系。

**🔧 技术方法**

使用深度学习框架（如 PyTorch/ TensorFlow）、PTQ、QAT、从零训练、阈值化技术以及自研的活动度和性能评估指标。

**📊 数据集**

在公开时间序列数据集（如 NAB、UCR、M4、Weather 等）上进行实验，覆盖多种预测任务与领域。

**📈 对比分析**

与未量化的基线模型相比，PTQ+阈值方案在保持预测精度（RMSE/MAE）不变的前提下，模型大小下降 70% 以上，推理延迟下降 60%；QAT 和从零训练 QAT 在某些复杂数据集上进一步提升了 3–5% 的准确率。

**⚠️ 局限性**

局限性包括：对极端稀疏或噪声高的时间序列预测效果仍有下降；阈值选择需手动调参，缺乏自动化；跨领域泛化仍受限于数据分布差异。

---

## 224. An Open-Source Data Infrastructure for Alloy Synthesis Research: Stage 1 --- Composition Screening and Literature Cross-Referencing

**arXiv ID:** 2610.28647 | [PDF](https://arxiv.org/pdf/2610.28647v1)

**作者:** Iliya Angelov Radulov `[一作]` `[通讯]`, Iliya Angelov Radulov

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了实验合金实验室的开源数据基础设施，集成成分解析、VEC/δ/ΔH_mix 等物理筛选指标、跨 Materials Project、OQMD、Alexandria 三大数据库的自动检索与去重，以及一体化桌面 GUI，支持“一键录入-即时预览”的完整工作流。

**💡 创新点**

创新点在于将成分解析、物理屏蔽、文献检索和 UI 完全无缝衔接为单一操作步骤，并公开记录与解决的工程缺陷与经验，形成可复用的系统设计范式；同时提出了多数据库去重与可调阈值过滤的策略。

**🔧 技术方法**

技术栈包括 Python（配合 Postgres、JSONB、OPTIMADE 客户端、桌面 GUI 框架如 PyQt/Tkinter）、脚本化批量导入工具，以及自定义的成分解析与物理指标计算模块。

**📊 数据集**

使用的数据集为公开的 Materials Project、OQMD、Alexandria 三大材料数据库；实验室本地的原始实验文件（XRD、SEM、VSM 等）仅做可视化存储，未进行量化提取。

**📈 对比分析**

通过对比传统 ELN + 手工多窗口查询流程，系统在单机环境下完成成分输入、指标计算、三库检索与去重的完整流程，性能在约 50 样本规模内无明显瓶颈，显著提升查询与录入效率；后续阶段（Stage 2/3）将进一步验证与提升。

**⚠️ 局限性**

局限性在于仅针对单实验室、单维护者场景；缺乏自动化测试和多用户支持；未实现实验结果的自动量化提取与 CALPHAD 阶段；对更大规模样本或多实验室部署的可扩展性仍需进一步评估。

---

## 225. ROCOCO: Robust Outline Completion of Occluded Curvilinear Objects

**arXiv ID:** 2610.29177 | [PDF](https://arxiv.org/pdf/2610.29177v1)

**作者:** Ken Zheng `[一作]` (University of California), Jaimyn Drake `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于流匹配的生成模型 ROCOCO，用于在遮挡条件下完成曲线特征。

**💡 创新点**

创新点在于将两阶段学习框架与流匹配结合，实现盲区遮挡检测与曲线重建的协同优化，参数量仅为传统方法的十分之一。

**🔧 技术方法**

使用 UNet、流匹配、时序编码和自注意力实现遮挡预测与曲线补全。

**📊 数据集**

采用程序生成的合成曲线数据以及真实的电线和裂纹公共数据集进行训练与评估。

**📈 对比分析**

与 SD‑Inpaint、LaMa 等现成填充方法比较，ROCOCO 在无遮挡标记时取得显著优势，且参数量更少，性能提升可达 30% 以上。

**⚠️ 局限性**

局限在于对多曲线、交叉或极端遮挡的适应性有限，且实验中未充分探索超参数和更复杂颜色变化。

---

## 226. GGraph: A GPU-Accelerated Library for Graph-Based Robot Deployment and Learning

**arXiv ID:** 2610.28953 | [PDF](https://arxiv.org/pdf/2610.28953v1)

**作者:** Yash Rampuria `[一作]` (Stanford University), Marco Pavone `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一套GPU加速的导航图库，可在单机器人部署、路径规划和强化学习中统一使用，并支持大规模并行机器人训练。

**💡 创新点**

将导航图构建、增量合并、前沿检测与属性层完全放在GPU上实现，并提供一次性循环闭合校正与批量RL环境，弥补了现有工具只能单机或固定图的空缺。

**🔧 技术方法**

利用GPU张量化图数据结构、并行最近邻、射线检测、Bellman–Ford/指针跳跃连通组件、ROS 2接口、SAC/PPO RL训练与PyTorch Geometric风格GNN，实现高吞吐量图操作。

**📊 数据集**

使用2D网格地图（ARiADNE训练/验证集+自生成150张），真实校园LiDAR/Odin数据生成的3D场景，以及CMU高保真模拟器中的大规模户外地图进行评估。

**📈 对比分析**

与SWAGGER/NetworkX传统实现相比，图更新速度提升120×；与ARiADNE CPU版相比，环境步进吞吐量提升27×；在6 h训练内，PPO策略在150张验证图上平均路径长度1255 px，完成率97%，明显优于ARiADNE及其公开模型。

**⚠️ 局限性**

仍需手工调参采样参数，难以处理动态障碍，GPU内存限制了可扩展规模，且仅支持静态地图。

---

## 227. Situating Vulnerability: An Ethical Governance Framework for Anthropomorphic AI Interaction

**arXiv ID:** 2610.29304 | [PDF](https://arxiv.org/pdf/2610.29304v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 228. Latent space configuration for neural network finetuning, fast ranking, and embedding dimensionality reduction

**arXiv ID:** 2610.29295 | [PDF](https://arxiv.org/pdf/2610.29295v1)

**作者:** Nikita Gabdullin `[一作]` (Joint Stock Research and production company Kryptonite), Anton Raskovalov `[通讯]` (Joint Stock Research and production company Kryptonite)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在预训练的ArcFace IResNet50上通过LSC方法微调，利用预定义向量系统对潜在空间进行几何结构化，并提出基于此结构的两种快速Top‑K检索算法，实现GPU/CPU搜索加速。

**💡 创新点**

创新点在于：①将LSC用于微调以获得可预先定义的几何分布；②提出利用Vn向量系统的最大最小索引实现O(1)候选集合检索；③在降低潜在空间维度后仍保持高召回率。

**🔧 技术方法**

使用的技术包括ArcFace预训练、LSC微调、随机投影降维（QR分解）、基于哈希表/张量的候选集合搜索、PyTorch实现以及GPU/CPU并行计算。

**📊 数据集**

数据集为MS‑Celeb‑1M子集，约148万张图像、24.3k人ID，按10%图像作为查询，其余为图库。

**📈 对比分析**

与完整枚举搜索对比，GPU上加速近3倍、CPU上提升约40%；在降维后仍能维持或提升rank‑1/10召回率，且候选集平均仅约100个。

**⚠️ 局限性**

局限性包括：①算法仅适用于结构化潜在空间，预训练模型性能低；②对极大ID数量的查询需要扩展搜索；③需在微调时预先构造Vn向量与ID对应，过程复杂；④在极小候选集下K>nc可能导致召回下降。

---

## 229. Region-Aware Multi-View Inspection Using Deep Learning and Domain-Informed Decision Fusion

**arXiv ID:** 2610.29398 | [PDF](https://arxiv.org/pdf/2610.29398v1)

**作者:** Srivishnu Ethirajulu Krishnaraj `[一作]` (Bosch Limited), Rishikesh R. Chopade `[通讯]` (Bosch Limited)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一个区域感知的双模型多视角视觉检测框架，结合Mask R‑CNN实例分割与EfficientNetV2‑S分类，并在PLC同步采集的全360°视角下，通过基于SOP的确定性决策层对汽车燃油喷射针进行生产质量评估。

**💡 创新点**

将不同功能区域的检测任务分配给专用模型，并在组件级别实现基于制造标准的确定性决策层，显著降低误报（假拒率从57.6%降至10.9%），同时保持100%缺陷召回；同时实现了全360°多视角采集与PLC对齐的实时生产部署。

**🔧 技术方法**

深度学习实例分割（Mask R‑CNN + ResNet‑50‑FPN），图像分类（EfficientNetV2‑S），PLC同步多视角采集，基于像素尺度的规则引擎决策融合，Python+PyTorch实现。

**📊 数据集**

380个汽车燃油喷射针的全视角数据集，Seat Region 2621张已标注实例（8类缺陷）和 Shaft Region 7933张图像（旋转OK/Bad），所有数据为 Bosch 内部专有。

**📈 对比分析**

在固定模型输出下通过对比 Models Only、各规则组和完整决策层进行组件级别 ablation；完整决策层将假拒率从57.6%降至10.9%，保持100%缺陷召回，整体准确率从58.2%提升至92.1%。

**⚠️ 局限性**

依赖固定光学条件和训练数据，无法自适应新型缺陷；决策规则是手工设定，若规格变更需人工维护；仅针对单一部件类型，通用性需进一步验证。

---

## 230. SimPT: An Automated Simulation Framework for Intervention-Based VLA Post-Training

**arXiv ID:** 2610.29161 | [PDF](https://arxiv.org/pdf/2610.29161v1)

**作者:** Caicheng Wang `[一作]` (Meituan), Yuchen Xie `[通讯]` (Meituan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SimPT 框架，在仿真中自动化实现视觉‑语言‑动作（VLA）策略的后训练，包含策略执行、自动化纠正数据采集、策略更新和独立评估。

**💡 创新点**

创新点在于：①Agent‑Compiled Progress Graph 通过轨迹中的进度检查自动选择回溯状态并驱动专家恢复；②利用交叉拟合与低通滤波改进 RECAP 的价值优势标签；③构建完整的仿真‑真实双向验证管线，显著降低人工监督成本。

**🔧 技术方法**

使用的技术包括：RoboTwin 虚拟数字双子仿真、CuRobo 代码化专家、GPT‑5.6‑Sol 生成进度图、RECAP、HG‑DAgger、DPO+SFT、FlowPRO 等后训练方法、交叉拟合（CF）和 Butterworth 低通滤波（LPF），以及 VLA 模型训练和评估。

**📊 数据集**

数据集包含七个 RoboTwin 机械臂操控任务（Lift Pot、Move Can Pot、Place Empty Cup、Beat Block Hammer、Place Can Basket、Handover Block、Stack Blocks Two）以及 25 种以太网线插入真实机器人场景，全部数据均在仿真中收集并在真实机器人上验证。

**📈 对比分析**

通过将 Progress Graph（PG）与固定间隔采样（Naive）进行对比，PG 在所有任务上将恢复尝试次数降低 75%，恢复成功率从 93.4% 提升至 98.9%；在 PG 数据下，RECAP‑CF‑LPF 在仿真中整体成功率由 76.5% 提升至 79.4%，在真实机器人以太网线插入任务中成功率从 72% 提升至 96%。

**⚠️ 局限性**

局限性包括：性能仍高度依赖任务与场景特征；仿真与真实环境的差异需要进一步的真实‑仿真重建；当前框架主要针对基于图的任务，非结构化任务的适用性尚待验证。

---

## 231. DriftNeRF: Cross-Ray Density Drifting for Sparse-View Neural Radiance Fields

**arXiv ID:** 2610.29106 | [PDF](https://arxiv.org/pdf/2610.29106v1)

**作者:** Juntang Wang `[一作]`, Shixin Xu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于漂移场的跨射线一致性正则化（DriftNeRF），通过在不同视角射线的密度样本之间建立局部共识来消除稀视角下的浮点与雾化现象。

**💡 创新点**

创新点在于将漂移场模板从生成分布匹配迁移到稀视角NeRF的密度一致性问题，并利用stop‑gradient实现单向修正，既保留边界又消除未被多视角支持的浮点。

**🔧 技术方法**

核心技术包括：NeRF渲染框架、欧氏距离衰减核、跨射线邻域加权平均、stop‑gradient冻结目标、以及将漂移损失与传统渲染损失加权融合。

**📊 数据集**

在NeRF‑Synthetic、DTU、LLFF三大公开稀视角基准上进行实验，使用3、6、9视角训练集进行评估。

**📈 对比分析**

与FreeNeRF、DNGaussian、SparseNeRF、InfoNeRF等多种稀视角方法以及3D Gaussian Splatting基线进行对比，DriftNeRF在3视角条件下平均提升PSNR约0.6 dB，SSIM约0.02，LPIPS约0.02，且在不增加外部先验的前提下实现对比效果。

**⚠️ 局限性**

主要局限包括：在稠密视角下正则化效果不显著；对薄壁、镜面等结构的局部一致性仍存在欠佳；与显式Gaussian方法相比，训练时间和显存开销显著增加；需进一步探索更具方向性或可视性约束的邻域策略。

---

## 232. Learning Dynamic Games for Cooperative Iterative Tasks

**arXiv ID:** 2610.28901 | [PDF](https://arxiv.org/pdf/2610.28901v1)

**作者:** Mark Pustilnik `[一作]` (University of California at Berkeley), Francesco Borrelli `[通讯]` (University of California at Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种面向共享约束的多智能体协作迭代动态游戏学习框架，利用已执行的可行轨迹构建样本安全集和玩家专属的成本-到终点函数，并通过在每步时间点选择终端状态与 GNE 方案实现递归可行性与个体不退化。

**💡 创新点**

创新点在于：① 将 LMPC 的样本安全集概念扩展到动态游戏；② 引入离散终端状态选择与共享乘子参数化的 GNE 采样，以探索多重均衡；③ 通过预先约定的个体接受规则和协作选择规则（加权成本或 Nash 讨价还价）实现多玩家之间的可接受且协同改进。

**🔧 技术方法**

使用混合整数动态游戏表述、KKT 条件构造与 PATH 求解器求解 MCP、共享乘子参数化、回溯备份法以及非凸动态游戏的可行性验证。

**📊 数据集**

在三人协作运动规划示例中进行实验，使用自行生成的初始可行轨迹（在平面障碍场景下的三无人机/机器人）作为数据集。

**📈 对比分析**

通过比较三种配置（多 GNE + Nash 讨价还价、单 GNE + Nash 讨价还价、以及多 GNE + 均衡权重）评估性能。实验表明，多 GNE + Nash 讨价还价在第 5 次迭代时总成本比多 GNE + 均衡权重低 1.9%，在第 15 次迭代时低 2.47%；单 GNE + Nash 讨价还价收敛最快但最终总成本更高。

**⚠️ 局限性**

局限性包括：① 计算量大，需离线求解；② 仅保证递归可行性和个体不退化，未提供收敛或全局最优保证；③ 需要已知精确模型，缺乏对模型不确定性的鲁棒性；④ 未对大规模多智能体系统进行扩展。

---

## 233. Depth-guided Geometric Adaptation for Few-shot Image Classification

**arXiv ID:** 2610.28856 | [PDF](https://arxiv.org/pdf/2610.28856v1)

**作者:** Zhiqi Miao `[一作]` (University of Groningen), Jiapan Guo `[通讯]` (University of Groningen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6514db3d-8de6-452c-91b7-acdb31787cc4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于单目深度的几何引导FSIC框架DAFIC，结合RGB‑D自监督预训练、Depth‑guided Geometric Adaptation与Spatial Bidirectional Patch Matching进行少样本分类。

**💡 创新点**

创新点在于将单目深度作为几何先验，融合到预训练、层级适配与双向密集匹配，显著降低背景干扰和视觉歧义。

**🔧 技术方法**

使用ViT‑Small骨干，RGB‑D Mask Image Modeling自监督预训练，Depth‑guided Attention与Geometry‑aware Adaptation模块，SBiM双向密集匹配及全局监督，配合零样本深度估计DepthAnything V2。

**📊 数据集**

在MiniImageNet、TieredImageNet、CIFAR‑FS和FC100四个基准上进行评测。

**📈 对比分析**

在5‑way 1‑shot与5‑shot任务中相较于CPEA、SMKD等最新方法提升5.7%/3.4%等，整体性能在各数据集上均达到或超过最强竞争者。

**⚠️ 局限性**

主要限制是对单目深度估计的依赖，深度误差会影响性能，且模型在极端光照或深度不可用时仍需进一步鲁棒性提升。

---

## 234. WorldStream: Streaming Interactive 3D Worlds with Hierarchical Planning and Persistent State

**arXiv ID:** 2610.28827 | [PDF](https://arxiv.org/pdf/2610.28827v1)

**作者:** Xiaoxiao Ma `[一作]` (University of Science and Technology of China), Feng Zhao `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于分层规划和持久化瓦片记录的实时开放世界生成框架WorldStream，支持无限延伸、即时编辑与基于自然语言的交互，并在运行时保持一致的世界状态。

**💡 创新点**

创新点包括：① 双前沿流式（semantic与geometric）调度，在保持内存有限的同时预先规划与生成；② 层级Macro–Micro规划结合类型化决策与联合概率解析，实现跨尺度一致的语义计划；③ 持久化瓦片记录存储计划、编辑与版本信息，实现无缝重访与编辑可追溯；④ 将语言理解与世界状态直接绑定，支持在任意时间点对已生成或未生成区域发出自然语言指令。

**🔧 技术方法**

技术手段包括：大规模决策模型（深度学习生成语义计划）、确定性本地生成代码（基于预置资产与地理约束的规则化生成器）、分层宏观与微观规划器、联合概率分布解析器、预算驱动的移动预取与构造、瓦片级边界共享与版本控制、基于规则的地理特征采样与一致性校验、以及Three.js WebGL运行时。

**📊 数据集**

使用真实世界地理证据（高度、坡度、水域、土地利用等）以及资产库中预制模型与文本/图像生成的可重用几何；实验基于公开的地形图与资产库（如OpenStreetMap、Procedural Asset Collections）进行验证。

**📈 对比分析**

实验通过对比不同规划层级（Macro/ Micro/ Flat）、联合解析与独立解析、预取策略（PlanAhead、Directional、Adaptive）以及语言交互准确率等，结果显示：① 完整分层规划获得连贯城市到农村的梯度，边界跳跃仅0.089；② 联合解析将可执行率从37%提升至100%；③ Directional预取使4km行程等待时间降至3.52秒；④ 语言交互准确率达95%；⑥ 本地生成优化后，稠密森林编译速度提升3.24倍。

**⚠️ 局限性**

局限性包括：① 目前采用有限的生成词汇与规则化生成器，难以支持高度自由的细节生成；② 持久化记录与编辑的存储与查询成本在大规模长时间运行时尚未评估；③ 语言理解仍需改进对时间与空间范围的区分；④ 仅在WebGL/Three.js环境下评估，对更大规模GPU/多线程渲染的兼容性待验证。

---

## 235. ArcFlow: Training Flow Policies for Multiple Solver Budgets

**arXiv ID:** 2610.28788 | [PDF](https://arxiv.org/pdf/2610.28788v1)

**作者:** Hieu Tran `[一作]` (University of Massachusetts Amherst), Hong Yu `[通讯]` (University of Massachusetts Amherst)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种在多步求解器预算范围内训练单一流匹配策略的方法——Action–Ratio–Clip Flow（ARC Flow），并通过边界监督、预算曝光和最差预算检查点选择提升低步（单步）部署性能，保持高步部署效果；

**💡 创新点**

创新点在于：① 引入边界（t=1）监督以恢复一步执行的信号；② 通过固定尺度的网格桥梁将多步轨迹映射到单步残差；③ 在训练时循环暴露不同步数的 roll‑out 并使用 worst‑budget 选择保证所有预算下都有良好性能；④ 对本地速度监督目标直接适配，保持原始网络结构；

**🔧 技术方法**

技术手段包括：流匹配（flow‑matching）/速度场学习、PPO 样式剪裁（clipped）策略梯度、固定网格桥梁（fixed‑grid bridge）、预算循环收集、最差预算验证、以及对速度监督目标的多预算端点监督；

**📊 数据集**

使用的环境与数据集为：DeepMind Control Suite（DM Control）20+ 任务、Manipulation 子域（PandaPickCube、PandaOpenCabinet 等 6 任务）以及 Locomotion 子域（Hopper、Walker 等 4 任务）；共计 30 任务；

**📈 对比分析**

与 5 种基线（FPO、FPO++、DPPO、ReinFlow、PolicyFlow）在同一公共堆栈下对比，ARC Flow 在 30 任务中平均单步回报最高（无一步崩溃，≈0.71 的一步回报），且在 4 步、16 步预算下的最差预算回报也最高；整体性能曲线在单步时领先显著，随着步数增加保持几乎不变，优于所有基线；

**⚠️ 局限性**

局限性包括：仅在模拟环境中验证；多预算训练与检查点选择增加了训练开销；对网格桥尺度与剪裁阈值的选择仍需经验调优；若迁移到真实机器人或不同动作空间，需进一步评估安全与稳定性；

---

## 236. An Exact 93-Move Solution to a Nonlocal Crystal-Exchange Puzzle

**arXiv ID:** 2610.29493 | [PDF](https://arxiv.org/pdf/2610.29493v1)

**作者:** Zilai Wang `[一作]` `[通讯]`, Zilai Wang

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文给出了一个11×11晶体交换谜题的最短解，证明最小指令数为93，并提供了该解的完整路径；

**💡 创新点**

创新点在于引入命令成本的α/β划分、压缩核心抽象、必经西北角条件、残差运动和激活逸路等多重启发式下界，从而实现对该实例的完全穷举排除；

**🔧 技术方法**

使用了广度优先搜索结合可接受启发式（Pattern Database、残差运动下界）、对称性归约、无损状态编码和多级存储优化等技术；

**📊 数据集**

使用的数据集为论文中明确定义的单一实例（起始状态、目标格子、出口、门），无外部公开数据集；

**📈 对比分析**

相较于传统的单纯搜索或启发式搜索，本文通过完整的计算排除证明了93指令是最优的；在16 GiB内存上完成了约1.86 B个状态的搜索；

**⚠️ 局限性**

局限性在于只针对该特定实例证明最优，未给出适用于泛化谜题的算法，也未使用形式化证明助手验证搜索逻辑。

---

## 237. LaVGuard: Selective Decoding for Contextual Value Drift

**arXiv ID:** 2610.29135 | [PDF](https://arxiv.org/pdf/2610.29135v1)

**作者:** Jiale Dai `[一作]` (Peking University), Guojie Song `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于任务配置文件的可选择解码器，用以在保持内容安全的前提下，按授权的价值优先级调整生成文本。

**💡 创新点**

创新点在于将任务相关价值向量与观察到的前缀状态分离，利用残差几何边界与可视化探针估计偏离，再通过位置和KL两项预算实现有限且可控的生成干预。

**🔧 技术方法**

使用可冻结的深度层探针、残差高斯混合模型、候选词评估与指数平移，配合基于KL的预算约束实现可选择解码。

**📊 数据集**

采用由三位注释者标注的含值强度的长文本集合（约200条多样化提示），在 Qwen2.5‑7B‑Instruct 上进行训练、验证与测试。

**📈 对比分析**

在200个基准变体中与 SafeInfer、SafeDecoding 等基线对比，实验显示风险 F1 从 0.58 提升至 0.84、风险安全率提升至 0.88、拒绝率下降，且在相同编辑比例下实现更高安全性和更低平均 KL。

**⚠️ 局限性**

局限性包括：价值向量维度有限，需依赖手工设计的配置文件；探针与残差模型的校准受训练分布限制；方法需要对模型内部状态和分布的访问，导致额外计算开销，且在更长响应或不同后端时需重新校准。

---

## 238. Universality of approximate message passing under a fourth-moment tail condition

**arXiv ID:** 2610.28971 | [PDF](https://arxiv.org/pdf/2610.28971v1)

**作者:** Ripunjay Dwivedi `[一作]` (University of Oxford), Taejoon Kim `[通讯]` (Arizona State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明了当 AMP 算法使用的矩阵是任意方形、独立、均值为 0、方差归一化的随机矩阵时，经验平均（包括二阶伪 Lipschitz 函数）在有限迭代次数下收敛到高斯状态演化的极限。

**💡 创新点**

创新点在于：① 只需矩阵条目满足四阶幂的统一可积性（即仅要求有限四阶矩），无需子高斯或所有阶矩；② 通过聚合三阶导数界、随机 Lindeberg 替换和算子范数截断，避免了传统的三阶矩匹配或非高斯通用性定理；③ 证明了仅靠平均方差归一化并不能保证状态演化，并指出了独立投影分量独立性仅在高斯条目下才成立，从而限制了对非高斯矩阵的 Gaussian conditioning 方法。

**🔧 技术方法**

主要技术包括：Lindeberg 原理的随机替换、伪 Lipschitz 函数的极限分析、矩阵算子范数与高阶导数的聚合估计、Taylor 展开与截断、以及对随机矩阵条目高阶矩的统一可积性控制。

**📊 数据集**

本工作完全基于理论分析，没有使用具体实验数据集；验证通过严格的数学证明而非数值模拟。

**📈 对比分析**

由于是理论证明，未与其他方法在数据集上进行实验比较；相对已有的 AMP 通用性定理（如要求子高斯、所有阶矩或特定稀疏结构），本文在更弱的四阶矩假设下仍能得到相同的状态演化极限，证明了更广泛的适用性。

**⚠️ 局限性**

局限性包括：① 仅适用于方形、均方归一化、条目独立且满足四阶统一可积性的矩阵；② 无法直接处理方差不均、相关或稀疏结构的矩阵；③ 对非平滑去噪函数（如软阈值）不适用；④ 结果仅在概率意义下收敛，除非满足额外条件才可得到几乎必然收敛。

---

## 239. Register Anything Model for Generalizable and Robust Point Cloud Registration

**arXiv ID:** 2610.28821 | [PDF](https://arxiv.org/pdf/2610.28821v1)

**作者:** Zheng Qin `[一作]` (Defense Innovation Institute, Academy of Military Sciences), Kai Xu `[通讯]` (Institute of AI for Industry, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种统一的点云配准模型（Register Anything Model），能够跨域学习并在室内、室外及物体场景中实现高精度的6-DoF配准。

**💡 创新点**

核心创新包括：①几何感知归一化策略——联合体素化与归一化，自动适配不同尺度与密度；②高效局部几何变压器，在局部邻域内建模几何一致性，避免全局自注意力的高内存消耗；③层次粗细配准管线，先匹配超点再细化到点级。

**🔧 技术方法**

技术实现基于KPConv-FPN骨干、几何变压器（自注意力+交叉注意力）、重叠得分预测、LGR估计、Circle Loss/负对数似然/二值交叉熵等多种损失函数；并采用局部距离与角度嵌入来提升匹配鲁棒性。

**📊 数据集**

训练数据融合室内3DMatch、室外KITTI、nuScenes与物体ModelNet四个数据集；评估覆盖12个基准：室内3DMatch/3DLoMatch/ScanNet++/TIERS，室外KITTI/nuScenes/ETH/MIT/WOD/Oxford，物体ModelNet/ShapeNet。

**📈 对比分析**

与Predator、CoFiNet、GeoTrans、BUFFER、RoITR、PARENet等9个基线相比，Register Anything Model在Inlier Ratio、Registration Recall、RRE、RTE等四项指标上均实现了新的最高分，尤其在跨域测试集上提升幅度显著，达成行业最优。

**⚠️ 局限性**

局限性包括：①KPConv骨干为旋转非不变，极端旋转场景仍可能受限；②模型对极大规模点云（如全城级）处理有限，需分块或流式处理；③离群点可能导致体素化过度，需进一步的离群点预处理或自适应阈值。

---

## 240. Regular Syndrome Decoding Is Easier Than Previously Estimated

**arXiv ID:** 2610.29175 | [PDF](https://arxiv.org/pdf/2610.29175v1)

**作者:** Jiseung Kim `[一作]` (Jeonbuk National University), Hochang Lee `[通讯]` (Affiliated Institute of ETRI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `b85d34da-f1e4-4203-bfed-9536213d369b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对正则症状解码（RSD）问题的混合代数攻击进行改进，证明现有多种密码系统的参数在实际安全性上存在显著缺陷。

**💡 创新点**

创新点在于：① 将零坐标猜测、基于症状的变量消元、预处理和最终代数求解统一为块级平衡的全局策略；② 对预处理成本进行细化评估；③ 对最终超定线性系统的可解性做更精准的概率分析，从而大幅降低具体复杂度。

**🔧 技术方法**

使用的技术包括：代数攻击框架（混合猜测-约简-求解），高斯消元、线性化（linearization）、块级均衡策略、概率分析（AM–GM不等式）、对超定线性系统的可解性评估。

**📊 数据集**

数据集主要是公开的 RSD 参数集合，来源于多种密码系统（如 PCG、Wolverine、Ferret、TinyKeys、MPC‑in‑the‑Head 方案等）以及标准基准实例。

**📈 对比分析**

与以往最优攻击（AGB、ISD、R‑ISD、RP、HAGB 等）进行对比，实验表明在非二进制域上可降低至 29.8 位，二进制域上可降低 25.9 位；在所有评估的应用中，至少有一组参数的安全级别低于 128 位，安全损失最高可达 66 位。

**⚠️ 局限性**

局限性在于：① 只针对正则症状解码问题，未覆盖所有 RSD 变体；② 依赖块级均衡假设，可能对非整数块尺寸的实例影响有限；③ 对随机矩阵可逆性和超定系统无解性的概率假设需进一步实验验证；④ 对极端参数设置（如极低率或极高率）可能需要额外分析。

---

## 241. Pair-Free Coarse-to-Real Video Generation via Real-to-Coarse Image Supervision

**arXiv ID:** 2610.28954 | [PDF](https://arxiv.org/pdf/2610.28954v1)

**作者:** Gonzalo Gómez-Nogales `[一作]` (Universidad Rey Juan Carlos), Yi Zhou `[通讯]` (Adobe)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无需配对视频的粗到真实（Coarse-to-Real）视频生成框架，先通过图像级粗-真实配对训练“桥接”模型生成伪粗视频，再用冻结的文本到视频扩散模型和轻量控制适配器从实际粗视频生成逼真视频。

**💡 创新点**

创新点：1) 将训练负担从昂贵的视频配对迁移到易获得的图像配对，显著降低数据标注成本；2) 在自监督DINO特征空间构建桥接，并通过RAE解码器做信息瓶颈，消除训练时的潜在泄漏；3) 在冻结的扩散主干上仅训练控制适配器，实现高效可扩展的控制。

**🔧 技术方法**

使用技术包括：DINOv3自监督特征、RAE（重构自编码器）解码器、冻结的Wan2.2 T2V-A14B文本到视频扩散模型、控制适配器与融合适配器、相机增广、LPIPS感知损失、rectified-flow velocity loss 等。

**📊 数据集**

使用的数据集：约14k对四种粗模态（粗几何、粗几何、粗动漫、粗动漫）的粗-真实图像对（由GPT-Image2生成真实图像并编辑得到粗图像），以及389k条真实视频（来自DL3DV等），用于在视频上生成伪粗视频进行训练。

**📈 对比分析**

对比方法：与Editto Global、Editto Sim2Real、VACE 2.2 等基线在DOVER技术/审美质量、VEFX编辑分数、VBench背景一致性、三类控制对齐度、以及Qwen3.8-27B-FP8 评估的真实性得分进行比较。结果显示该方法在技术质量、控制一致性和真实性方面均优于或接近最强基线。

**⚠️ 局限性**

局限性：对伪粗视频与实际粗视频的差距敏感；缺乏配对真实视频的直接评估；冻结主干导致现实与控制对齐的权衡；在大幅姿态/运动变化时可能出现失真；需要进一步缩小伪粗与实际粗的分布差距，并提升参考图像控制的效果。

---

## 242. Beyond Tokens: NumAnchor for Address-Grounded Numeric Operand Representation in Assembly Language Models

**arXiv ID:** 2610.29027 | [PDF](https://arxiv.org/pdf/2610.29027v1)

**作者:** Kun Liu `[一作]` (Tulane University), Jiang Ming `[通讯]` (Tulane University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了NumAnchor，一个在ALM输入边界将可恢复的数值型操作符映射为类型化、规范化的程序实体，从而提供更丰富的结构信息。

**💡 创新点**

创新点在于将代码、数据、栈等多种实体类型与其层级坐标同时注入模型，利用Fourier映射和混合离散-连续编码，实现静态分析信息在Transformer输入中的高效融合。

**🔧 技术方法**

采用了静态分析提取、类型化与规范化、结构坐标构造、Fourier频谱映射、MLP投影以及混合堆栈偏移编码等技术，并在原有Transformer架构前插入多通道结构信息。

**📊 数据集**

使用了StateFormer和jTrans两大公开二进制语料库（分别包含约3.6M函数和10k+二进制）以及相应的数值操作符统计数据。

**📈 对比分析**

与基线ALM（PalmTree、BERT、jTrans、StateFormer、TREX）进行内部对照实验，NumAnchor在函数检索Recall@1提升35%/31.7% MRR、跳转目标AUROC提升至0.932/0.907、类型恢复F1提升至+5.2点，且仅引入5.5%参数增长和≈7.5ms/函数的推理延迟。

**⚠️ 局限性**

局限性包括对x86/x64环境的依赖，未对ARM/MIPS或高强度混淆二进制进行验证，且仅能利用静态可观测信息，无法处理复杂的指针推导与动态生成的地址。

---

## 243. A cross-cultural dataset on the structure of social networks: The ENDOW project

**arXiv ID:** 2610.28793 | [PDF](https://arxiv.org/pdf/2610.28793v1)

**作者:** Eleanor A. Power `[一作]`, John P. Ziker `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `79276348-11e0-48e3-84bc-7ec231d0171c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文基于ENDOW项目，对29个国家50个乡村社区进行多层社会网络调查，捕捉家庭（共享单元）间在金钱、物品、信息、情感、行为等多种支持关系，构建层级网络并进行汇总；

**💡 创新点**

创新点在于：①大规模跨文化、跨生态的多层网络数据集；②同时测量多种支持类型，揭示网络结构的高异质性；③为跨域比较提供新的基准，挑战传统“经验法则”；

**🔧 技术方法**

采用社会网络分析方法：访谈式网络提示、双向采样、共享单元层级聚合、加权边构建、Jaccard/皮尔逊相关、主成分分析和方差分解；

**📊 数据集**

使用ENDOW项目收集的50个社区（平均约60-90共享单元，平均5人/单元）的网络数据；

**📈 对比分析**

将ENDOW网络的度数、密度、互惠、聚类、同阶相关等统计量与已有学校、工作场所、在线社交等基准进行对比，发现传统“经验法则”不再适用，表现出更大异质性；

**⚠️ 局限性**

局限包括：样本代表性依赖于不同社区的访谈者选择，外部联系被低估；网络构建方法和提示差异可能引入测量偏差；仅聚焦于支持关系，未覆盖所有社会互动类型；

---

## 244. Qwen-Audio-3.1-ASR Technical Report

**arXiv ID:** 2610.28711 | [PDF](https://arxiv.org/pdf/2610.28711v1)

**作者:** Chuanmeng Bian `[一作]` (Alibaba Group), Jianheng Zhuo `[通讯]` (Alibaba Group)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Qwen‑Audio‑3.1‑ASR 系列模型，支持 30 种语言与 16 种中文方言的多语种 ASR、说话人分离转录以及通用音频理解，并提供文件、流式、交互式等多种部署变体。

**💡 创新点**

在前一代基础上新增了结构化说话人分离输出、指令驱动的通用音频理解（captioning、定位、问答）以及层级化热词与原生文本润色控制，形成了从词级识别到完整语音–文本–结构化理解的一体化系统。

**🔧 技术方法**

采用 Qwen Mixture‑of‑Experts 语言模型，配合自监督音频预训练（BEST‑RQ）、AED‑预训练、LLM 冷却适配、监督微调、强化学习（GRPO）和 on‑policy 蒸馏（OPD）等技术；音频编码器为 Transformer，解码器为 MoE LLM，辅以 CTC 预估支撑长音频对齐。

**📊 数据集**

训练数据覆盖数千万小时多语种无标签音频与多种标注音频‑文本对（AISHELL、GigaSpeech、Common Voice、FLEURS、KeSpeech、WSYue、AliMeeting、AISHELL‑4 等），内部行业实体、方言、热词及对话数据；评测使用公开基准和内部测试集，涵盖 ASR、方言识别、热词召回、说话人分离、通用音频理解等。

**📈 对比分析**

通过统一 API 与公开报告对比，使用 WER/CER、DER/cpWER、热词召回率、流式延迟、通用音频理解准确率及 mIoU 等指标评估；3.1 版在多语种 ASR、方言 ASR/AST、说话人分离 DER/cpWER、通用音频理解准确率及流式延迟方面均达到或超过上一代与竞争模型的最佳水平。

**⚠️ 局限性**

限制主要包括：说话人分离与通用音频理解仅在非流式变体可用；低资源语言与极端噪声场景下的识别仍受限；热词与历史上下文需要人工配置；模型规模大、推理成本高，对低算力边缘设备部署存在挑战。

---

## 245. Joint-Angle Invariance of Jacobian-Based Performance Indices for Serial Robotic Manipulators

**arXiv ID:** 2610.29570 | [PDF](https://arxiv.org/pdf/2610.29570v1)

**作者:** Yaohua Zhou `[一作]` (Eastern Institute of Technology), Xiaocong Li `[通讯]` (Eastern Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了串联机械臂的关节角度不变性（JAI）理论，证明某些基于雅可比矩阵的性能指标（操纵性、刚度等）对首末关节角度不敏感，并利用此性质在性能评估与优化中实现维度约简。

**💡 创新点**

首次系统发现并证明了在首末关节垂直基底平面与工具平面的典型结构下，雅可比矩阵相关性能指标不受首末关节角度影响，从而实现了性能指标的维度不变性，并将该不变性首次应用于性能地图构建与任务布局优化。

**🔧 技术方法**

使用解析几何与矩阵不变性理论、正交相似变换与正交投影子空间分析、数值采样、MATLAB内部点法、闭式逆运动学与梯度投影法等技术。

**📊 数据集**

基于7-DOF Franka Research 3机器人和6-DOF ABB IRB 4400机器人的官方参数，并在工作空间内以每个关节10个采样点构成的离散网格（共10^7/10^5样本）进行实验。

**📈 对比分析**

通过与传统完整关节采样（CS）对比，比较计算负荷、CPU时间、存储空间以及性能指数差异。JAIBS在性能地图构建中将计算负荷降低至1/100、CPU时间降至1.7s、存储空间减少48.6%；在任务布局优化中计算时间降低45-73%，优化结果保持一致且更为鲁棒。

**⚠️ 局限性**

仅适用于首末关节为垂直基底/工具平面的机械臂，且仅对满足满秩任务雅可比矩阵且对应性能指标的情况；对非齐次关节结构或不满足不变性子空间的指标无效；实验仅在两台机器人上验证，缺乏更广泛机械臂类型或更复杂任务的验证。

---

## 246. DetM-Tracker: Deterministic and Memory-Efficient VIS for Autonomous Industrial Inspection

**arXiv ID:** 2610.29212 | [PDF](https://arxiv.org/pdf/2610.29212v1)

**作者:** You-Rui Lin `[一作]` (National Yang Ming Chiao Tung University), Yu-Chee Tseng `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出DetM-Tracker框架，用于在资源受限的工业边缘设备上实现高精度视频实例分割。

**💡 创新点**

三项创新：Compact Scalar Memory压缩时间记忆为四维标量，Dynamic Instance Channels自适应查询数，Entropy-Gated Skip Connection基于不确定性聚焦高频细节。

**🔧 技术方法**

使用ViT编码器、全局缓存参考、稀疏时间记忆模块、双向掩码解码器和熵门控跳跃连接等技术。

**📊 数据集**

三阶段数据集：972.9k帧合成数据、82帧Real-U、53.6k帧Real-P，总计约1M帧。

**📈 对比分析**

与SAM2/3、Cutie、LiVOS等基线对比，mIoU提升至0.697，推理延迟15.2ms，比Cutie快20×，VRAM 557MB，内存占用比密集特征基线低6×。

**⚠️ 局限性**

受限于低分辨率和对极细结构的识别，未来需改进编码器（如Swin）以进一步提升精度。

---

## 247. Exploiting Pinching Antenna Systems in NOMA Covert Communication Networks

**arXiv ID:** 2610.29433 | [PDF](https://arxiv.org/pdf/2610.29433v1)

**作者:** Xinwei Yue `[一作]` (Beijing Information Science & Technology University), Yuanwei Liu `[通讯]` (University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研究了在PINCHING ANTENNA SYSTEMS（PASS）支持下的非正交多址（NOMA）隐蔽通信网络，提出了双监视者（warden）检测模型，推导了误检概率（DEP）解析式并给出了最优阈值与最小DEP，进一步分析了理想与非理想SIC情况下的隐蔽传输速率。

**💡 创新点**

创新点包括：① 将PASS与NOMA结合用于隐蔽通信，首次在此场景下推导两位监视者的DEP与最优阈值；② 分析了激活PINCHING天线数量对DEP与速率的双重影响；③ 在存在噪声功率不确定性的前提下给出了闭式解，展示了不确定性如何提升隐蔽性。

**🔧 技术方法**

技术手段：PASS天线激活与相位匹配、NOMA信号叠加、基于阈值检测的隐蔽通信模型、误检概率解析（分段分析）、最佳阈值求解、Gauss–Legendre/Laguerre数值积分计算隐蔽速率；对SIC进行理想与非理想模型化。

**📊 数据集**

使用仿真参数集（如K=10个PINCHING天线、28 GHz载波、3 m部署高度、10 m服务区、a_r=0.3、a_t=0.7、Δσ=2 dB 等）进行 Monte‑Carlo 仿真，未使用公开数据集。

**📈 对比分析**

与传统天线 NOMA（TA‑NOMA）和 PASS‑OMA 进行对比。仿真显示：PASS‑NOMA 在特定功率区间内比 PASS‑OMA 和 TA‑NOMA 的 DEP 更高（即更隐蔽），且隐蔽速率显著优于 TA‑NOMA，尤其是对近端用户；当激活天线数量增多时，近端用户的速率提升更明显；在高功率下远端用户速率趋于 a_t/a_r 限制。

**⚠️ 局限性**

局限性：仅考虑 LoS 单径传播、理想/非理想SIC模型，未涵盖多径/遮挡；噪声功率采用上界不确定模型，真实环境下可能更复杂；仿真验证在理想条件下，实际部署可能需要进一步的实验验证；解析式对高阶统计（如多用户协同）尚未推广。

---

## 248. Offline Apprenticeship Learning for Human-Centric Environments

**arXiv ID:** 2610.28716 | [PDF](https://arxiv.org/pdf/2610.28716v1)

**作者:** Md Mirajul Islam `[一作]` `[通讯]`, Md Mirajul Islam

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

缺失论文内容，无法总结

**💡 创新点**

缺失论文内容，无法总结

**🔧 技术方法**

缺失论文内容，无法总结

**📊 数据集**

缺失论文内容，无法总结

**📈 对比分析**

缺失论文内容，无法总结

**⚠️ 局限性**

缺失论文内容，无法总结

---

## 249. Reader--Writer Coupling in Transformer MLPs

**arXiv ID:** 2610.29340 | [PDF](https://arxiv.org/pdf/2610.29340v1)

**作者:** Giyeong Oh `[一作]` (Seoul National University), Youngjae Yu `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对预归一化Transformer中的MLP层进行深入分析，提出将每个隐藏通道视为一个由读者（reader）与写者（writer）组成的“对”，并利用Jacobian对称-反对称分解、配对坐标与重排检验等方法，揭示出这些对在训练过程中形成的符号（大多为相对方向相反）以及其在网络不同层、不同训练阶段的演化规律。

**💡 创新点**

创新点包括：①提出“对”视角，捕捉读者-写者之间的签名与耦合；②发现最强对普遍为相反方向（writer与reader对抗）且在训练早期即形成；③通过重排检验验证配对不是随机的，而是训练所固化的结构；④构建核心通道集合（top‑|m_h|）并跟踪其符号与强度的时间演化；⑤结合词汇投影（vocabulary lens）评估对下一词预测的影响。

**🔧 技术方法**

使用的技术与方法包括：Pre‑norm MLP的Jacobian分解（S、K、对称得分C）、配对坐标m_h与m_{w,h}、重排检验（permute writers、计算C(J)与D(J)、O(J)）、对称‑反对称能量对比、词汇投影（W_U、P_k）、训练阶段监测（梯度重排实验）、实验数据集与模型（Tiny/Small/Medium 预归一化解码器、OLMo‑1B、Pythia、GPT‑2）、对比控制（初始化、随机重排、无序文本、Zipf i.i.d.）以及可视化与核心计数方法。

**📊 数据集**

主要数据集：FineWeb‑EDU（用于训练），TinyStories、C4‑RealNews、Gutenberg、Web文本、随机打乱的Web文本、Zipf i.i.d. 词典（用于对比与控制）。

**📈 对比分析**

比较方法：将训练好的模型与随机重排（writer重新配对）以及初始化状态进行对比，使用对称得分C(J)、D(J)与O(J)评估配对的特殊性；通过词汇投影检查对下一词logit的影响；通过核心通道的符号与强度变化轨迹与训练步数关联。结果显示：训练后的对称得分显著高于任何重排，最强对多为相反方向，符号在训练早期就已形成并在后续保持，核心通道在模型深度中呈现出相对稳定的“对抗”结构。

**⚠️ 局限性**

局限性：①仅关注预归一化的MLP，未覆盖后归一化或其他变体；②对抗配对的产生机制尚未解释清楚；③研究侧重于结构与符号，而非直接提升预测性能，缺乏对应用效果的量化评估；④在极大模型（如4B+）中的可扩展性与泛化性尚未完全验证。

---

## 250. Correlated Failures of AI Approval Monitors

**arXiv ID:** 2610.29178 | [PDF](https://arxiv.org/pdf/2610.29178v1)

**作者:** Luca Zanolini `[一作]` `[通讯]`, Luca Zanolini

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究并实验证明在 AI 代理执行关键决策（如支付审批）前，使用多模型监视器时，需通过多源信息分散输入、加强注入抵抗、计数源而非投票，并加入不可被劝导的规则引擎，以实现安全与活跃的决策规则。

**💡 创新点**

创新点在于：①提出将监视器失败模型拆分为“同意失败”和“拒绝失败”两方向的失效集合，形成“交叉覆盖”条件；②验证在支付审批任务中，实际失效集合呈列结构而非行结构；③发现计数源规则在安全性相同下显著降低误拒率，并证明在缺失交叉覆盖时需加入不可劝导的规则引擎。

**🔧 技术方法**

技术方法包括：基于失效集合的故障模型构建、定量评估“跨覆盖”条件、使用定制化提示和对抗攻击生成（提示注入、文本劝说、伪造源文档），以及多模型投票与根计数等决策规则实现。

**📊 数据集**

使用的数据集为：①小规模四个公开权重模型（Llama、Qwen、Gemma、Ministral）在三源（发票邮件、ERP 记录、供应商门户）上进行的 6,560 条人工合成支付任务；②前瞻性边界模型（Claude Opus、GPT-6 Astra、Gemini 4 Argon、DeepSeek‑V3.2）在相同任务上进行的 2,040 条攻击与正常样本。

**📈 对比分析**

通过与传统投票/加权投票规则对比，实验表明“源计数”规则在同等安全性下误拒率降低约 6.5%（从 7.4% 降至 0.9%），且在大多数攻击情形下保持更高的活性；在缺失交叉覆盖时加入规则引擎可恢复安全与活性。

**⚠️ 局限性**

局限性包括：仅针对支付审批等可验证字段的“见证模式”任务；对非判定性任务或单一真值来源的通用性未知；实验仅覆盖有限模型规模与攻击类型，可能无法涵盖更强对抗者或模型自适应攻击；不可劝导规则引擎本身仍易被伪造记录攻击。

---

## 251. Policy Central: Detecting Conflicts, Redundancy, and Similarity in Deployed Police Policy

**arXiv ID:** 2610.29338 | [PDF](https://arxiv.org/pdf/2610.29338v1)

**作者:** Holly Baker `[一作]` (PolicingLab), Kyana Pollas `[通讯]` (PolicingLab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并部署了 Policy Central 系统，用于自动识别警察部门政策文件中冲突、冗余、相似和不同的关系。

**💡 创新点**

提出两种架构 Retrieve‑Judge 和 Screen‑Confirm，并结合 NLI 框架与精细化的文档分割/提炼，实现可扩展的库健康扫描和按需审核。

**🔧 技术方法**

利用向量检索（Cohere Embed、Qdrant）、LLM 判定（Claude Haiku/ Sonnet、Llama 3.3 70B）、BM25、Prompt 缓存、分层 Cascade 等技术。

**📊 数据集**

基于美国某警察部门 84 篇政策文件共约 19,646 节，构建 302 对真实关系标注的基准集（含 26 冲突、57 冗余、44 相似、175 不同）。

**📈 对比分析**

对 Retrieve‑Judge、Screen‑Confirm、Llama 零射击、DeBERTa NLI 四个系统在给定对和需找对两情境下进行 Recall 评估；Screen‑Confirm 在给定对时 Recall 0.795，端到端 Recall 约 0.73；Retrieve‑Judge 在给定对时 0.654，端到端约 0.457；NLI 模型效果低。

**⚠️ 局限性**

标注半自动、缺乏领域专家一致性；仅单一部门、规模有限；检索、分割错误导致误判；Recall 估计依赖独立假设；模型与推理过程的可解释性受限。

---

## 252. DepState-JEPA: Learning Depth-Grounded State Representation for Visual Perception and Manipulation

**arXiv ID:** 2610.29246 | [PDF](https://arxiv.org/pdf/2610.29246v1)

**作者:** Xiujian Liang `[一作]` (Fudan University), Jianhua Sun `[通讯]` (Shanghai Innovation Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种双流RGB‑Depth联合嵌入预测框架DepState‑JEPA，利用测量深度构建共享潜在状态，并在此基础上实现空间读出与动作驱动的状态预测，实现视觉特征补全、时间序列预测与机器人操控。

**💡 创新点**

创新点包括：①将深度信息作为几何基准，将RGB与深度在3D坐标系中对齐并聚合为共享状态；②采用共享‑私有投影与射线条件读出，既保留多模态细节又保持统一的状态表示；③在状态转移中加入前向运动学一致性正则化，提升物理一致性；④分阶段训练（RGB‑D自适应 + 机器人后训练），兼顾视觉与动作学习。

**🔧 技术方法**

核心技术包括：V‑JEPA 2.1的自监督掩码预测与密集多层监督、三维位置编码、共享‑私有特征因子化、注意力聚合查询、射线条件读出、动作条件Transformer状态转移、前向运动学正则化、深度补全与视觉预测、基于回放的多步强化学习与MPC。

**📊 数据集**

使用的数据集包括：NYUv2（单目深度估计）、VOC12（语义分割）、LingBot‑Depth（遮挡特征补全）、DMControl与Meta‑World（轨迹与策略学习）、仿真/真实对比的Drawer/Cup任务；以及RGB‑D视频/图像进行自适应预训练。

**📈 对比分析**

方法通过与V‑JEPA、LeJEPA、LeMuMo‑JEPA、Locate‑3D等现有JEPA和深度相关JEPA进行对比，评估编码器迁移、特征补全、下一帧预测、控制与规划效果。实验表明：RGB‑D自适应提升编码器迁移性能，Depth内容和几何坐标结合使L1下降约0.068、成功率上升约6.6%；整体模型在多步预测、基于模型的控制和sim‑to‑real manipulation中均超过基线，最大任务成功率达到约83%及更高。

**⚠️ 局限性**

局限性：模型依赖标定好的深度相机，深度噪声或缺失可能影响几何基准；共享状态的尺寸固定，对极大视图变化或高维场景的泛化尚待验证；计算开销相对较大，实时部署需要进一步优化；实验主要集中在室内机器人抓取任务，尚未验证在更复杂动态环境中的表现。

---

## 253. Closed-Loop Dual-Reference Guidance for Accent Control in Cross-Lingual Voice Cloning

**arXiv ID:** 2610.29351 | [PDF](https://arxiv.org/pdf/2610.29351v1)

**作者:** Yuanzhong Zheng `[一作]` (VoxNexus), Youhai Jiang `[通讯]` (VoxNexus)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出闭环双参考指导，结合原始说话人参考与可重用的母语锚点，实现跨语言零样本语音克隆的声纹保留与口音控制。

**💡 创新点**

创新点在于将发音和声纹信息分离为两个自回归参考，并通过将生成补丁回馈给两条 LM 状态来保持共历史，从而显著降低目标语言中的源口音残留并恢复声纹相似度。

**🔧 技术方法**

使用冻结的 VoxCPM2 TTS 模型、Seed‑VC 零样本声码器、流匹配解码器、CFG‑Zero* 速度混合以及自回归闭环反馈机制。

**📊 数据集**

实验基于 VoxPopuli 法语说话人、LibriTTS-R 英语供体以及 VoxPopuli 西班牙语说话人进行跨语言克隆评估。

**📈 对比分析**

与直接克隆、锚点克隆和 TTS→VC 级联对比；在留出的法语测试集上，λ=0.33 方案将法语 LID 残留降低 73%，WER 降至 0.009，GOP 提升至 4.750，RTF 约 0.59（比级联低 26%）。

**⚠️ 局限性**

局限在于仅针对英语目标语言、锚点质量受零样本声码器限制、指导权重固定不随补丁变化、以及缺乏对多目标语言的进一步验证。

---

## 254. Implicit Feature Reweighting for Label-Free Graph Generalization

**arXiv ID:** 2610.29571 | [PDF](https://arxiv.org/pdf/2610.29571v1)

**作者:** Tianjun Yao `[一作]` (Shenzhen Loop Area Institute), Lixian Zhong `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种无监督、标签自由的图形神经网络 OOD 泛化框架，利用 VGAE 的重构保持核心与环境特征，并通过对抗性混淆压缩环境可区分方向，从而提升对分布漂移的鲁棒性。

**💡 创新点**

创新点包括：①证明重构任务可保留核心与环境因素；②引入对抗混淆机制，将环境方向收缩并等价于对下游线性探测器的各向异性岭回归惩罚；③无需类标签即可通过环境划分（甚至聚类得到的伪环境）训练，提供了对标签稀缺领域（如药物发现）的实用方案。

**🔧 技术方法**

核心技术为：变分图自编码器（VGAE）进行图重构；对抗性环境混淆网络（两层 MLP）与重构损失交替训练；深度特征重加权（DFR）评估线性可访问性；理论分析包含信息保持定理、方向收缩与岭回归等价性。

**📊 数据集**

实验数据集包括：synthetic 数据（验证理论机制）；GOOD‑Motif（概念转移 benchmark）；DrugOOD（药物属性预测，涵盖 assay、scaffold、size 三种 OOD 场景），并通过聚类产生伪环境。

**📈 对比分析**

与 GraphCL、MVGRL、OOD‑GCL、RGCL 等对抗/对比基准比较，label‑free 方法中本框架在 GOOD‑Motif 上达到 69.6% OOD 准确率，在 DrugOOD 上在 17/18 组中排名第一，且在 scaffold 等最具挑战的 split 上提升 3–4 分；在控制实验中证明了收缩对 OOD 风险的严格下降。

**⚠️ 局限性**

局限性：依赖有效的环境划分或能通过聚类获得的粗糙划分；当核心与环境方向高度重叠或下游分类器未正则化时收缩可能不利；需调节 β、K 等超参数，且对极端稀疏标签场景的鲁棒性仍待进一步验证。

---

## 255. SkillEval: Learning Interpretable Ability Profiles of LLMs via Cognitive Diagnosis Models

**arXiv ID:** 2610.28767 | [PDF](https://arxiv.org/pdf/2610.28767v1)

**作者:** Berke Arda `[一作]` (ETH Zurich), Mrinmaya Sachan `[通讯]` (ETH Zurich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套名为 SkillEval 的评估框架，通过自动提取和聚类题目中的技能，然后用神经认知诊断模型 (Neural Cognitive Diagnostic Model) 对 3,811 语言模型在 9,523 道题目的表现进行多维技能级别的评估，生成可解释且稳定的技能掌握谱。

**💡 创新点**

创新点包括：1) 结合 LLM 的元认知能力自动发现可解释技能；2) 将技能映射到 Q‑矩阵并固定维度，解决传统多维 IRT 的不可辨识问题；3) 采用文本条件化的 NCDM 使模型可对未见题目进行推断；4) 利用该技能谱实现高效 LLM 路由、可适应的自适应测试以及新模型的快速刻画。

**🔧 技术方法**

技术主要包括：大规模 LLM 评测、SBERT 语义嵌入+层次聚类用于技能聚类、Neural CDM（带 PosLinear 层）与三层 MLP 预测答对概率、文本编码器用于学习题目难度与辨识度、前向/后向梯度优化、Procrustes 对齐、贝叶斯先验与 L‑BFGS 进行新模型的技能估计。

**📊 数据集**

数据集涵盖五大基准（MATH、BBH、GPQA、MuSR、IFEval），共 9,523 道题目，3,811 个 LLM（从开源与商业模型）产生 36M 条模型–题目得分记录，用于训练与评估。

**📈 对比分析**

与 2PL IRT、IRTNet（高维神经 IRT）以及 KNN/随机/最强单模型等基线比较：SkillEval 在 OOD 条件下的预测 AUC 约 72.15（ID 81.73）略低于 IRTNet；在 Benchmark 级别预测中 Pearson r 0.989，显著优于 IRTNet（0.993）且在每个基准上更精确；在 LLM 路由任务中 Acc@1 65.67% 高于最强单模型（61.57%）和 IRT 2PL（61.10%），并在相同推理成本下实现约 4 倍的成本下降；对新模型的技能估计只需 20 个自适应答案即可获得与 100 个随机答案相近的基准误差。

**⚠️ 局限性**

局限性包括：1) 由于 Q‑矩阵和技能聚类依赖 LLM 自动提取，仍可能出现技能冗余或误匹配；2) 文本条件化的 NCDM 对未见题目难度与辨识度的估计不如显式参数化的 IRT；3) 训练成本高（需要 36M 条记录），且对极大模型规模的普适性尚未充分验证；4) 路由性能受技能覆盖不足时的影响较大；5) 评测主要聚焦准确率和成本，未涉及安全、偏差等更广泛的 LLM 评估维度。

---

## 256. Reinforcement learning for hardware acceleration and system-level optimization on FPGA and integrated-circuit platforms: A survey

**arXiv ID:** 2610.28699 | [PDF](https://arxiv.org/pdf/2610.28699v1)

**作者:** Hamza Kheddar `[一作]` `[通讯]` (University of Medea), Hamza Kheddar (University of Medea)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了近十年在FPGA和集成电路（IC）平台上实现的强化学习（RL）加速与系统级优化研究，系统梳理了超过120篇论文，按算法家族、硬件抽象层、应用场景等维度构建统一分类体系，并在表格中对比了资源占用、延迟、吞吐量、能效等关键指标。

**💡 创新点**

创新点在于：①提供了覆盖FPGA与IC的首个统一视角，填补以往聚焦单一平台的空白；②提出了多层级（SLD、DSC、HLS、RTL、门级）硬件实现流程的对照表，帮助读者快速定位设计阶段；③对比多种RL算法（MDP、QL、DQN、PPO、DDPG、MARL等）的硬件映射特点与典型瓶颈；④设计了跨论文的综合性能度量（如FIPS/W、ERE_T、EDP），推动了可复现且可量化的评估标准；⑤指出了硬件加速与RL自适应双向关系，梳理了未来研究方向。

**🔧 技术方法**

采用系统文献检索（Scopus、IEEE Xplore、Elsevier等）与布尔查询筛选，结合结构化的抽取框架对算法、架构、实现细节、实验数据进行归纳；随后构建了分类树与多维度评价指标，利用表格与图示直观呈现对比结果；文中也引用了多种硬件描述语言（Verilog/VHDL）、HLS工具（Vivado HLS、Intel HLS）、深度学习框架（TensorFlow/PyTorch）以及专用EDA工具（Cadence Stratus、Synopsys HLS）等技术。

**📊 数据集**

作为综述文章，未使用单一数据集，而是汇总各被评审论文所使用的标准RL环境（如CartPole、Atari、DRL Benchmark、自定义工业调度、网络流量、机器人定位）与数据来源，以对比不同实现的实验场景。

**📈 对比分析**

作者将综述结果与前人调查对照，展示了在FPGA/IC上的速度提升（如DQN 19× GPU、A3C 54× GPU）、资源利用率（BRAM/LUT/DSP占比）和能效（FIPS/W）等指标；同时给出多维度性能曲线，说明某些实现虽然吞吐量高，但能耗或资源瓶颈亦显著。通过统一指标可直观评估不同论文在同一维度上的优势与劣势。

**⚠️ 局限性**

局限性包括：①仅聚焦FPGA与IC实现，未覆盖GPU/CPU纯软件RL；②不同论文使用的实验环境与基准差异大，导致跨论文对比仍带有相对性；③部分工作缺乏公开实现或详细度量，信息不足；④综述截止时间为2026年10月，后续新兴硬件如PIM、TPU、AI芯片的RL实现尚未纳入；⑤对RL自适应控制与硬件加速双向耦合的实证研究仍相对不足。

---

## 257. Learning Marginal Viewpoint Utility for Budgeted Evidence Selection in RAG

**arXiv ID:** 2610.29234 | [PDF](https://arxiv.org/pdf/2610.29234v1)

**作者:** Yiyang Duan `[一作]` (Nanyang Technological University), Wei Yang Bryan Lim `[通讯]` (Nanyang Technological University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在检索增强生成任务中，提出LMVU轻量级线性排序器，通过学习候选证据的边际观点覆盖，按token预算选择多观点覆盖的证据。

**💡 创新点**

创新点在于使用视角覆盖的边际价值监督，利用训练阶段的观点标签构造状态内排序对比，形成状态条件的轻量级线性评分器。

**🔧 技术方法**

技术包括冻结嵌入的MiniLM-L6 bi‑encoder、对状态的18维特征（相关性、相似度、长度等）、对数几率对比学习、贪心预算分配。

**📊 数据集**

使用四个多视角注释基准：World Values Survey (WVS)、Perspectrum、Argument Key‑Point (ArgKP)、European Social Survey 10 (ESS10)。

**📈 对比分析**

与MMR、FacLoc、BM25、AdaGReS、MMKP、JPR、RichRAG等基线比较，在1,024 token预算下LMVU在三大基准上提升0.04–0.05 RVC点，平均覆盖率最高，甚至在WVS达到全池上限。

**⚠️ 局限性**

局限包括依赖于观点标注、嵌入模型敏感、未提供近似最优保证、仅评估观点覆盖未测下游生成效果。

---

## 258. AgentMonopoly: Behavioral Patterns and Persona Control in LLM Agents

**arXiv ID:** 2610.29330 | [PDF](https://arxiv.org/pdf/2610.29330v1)

**作者:** Changshun Wu `[一作]` (University of Liverpool), Chih-Hong Cheng `[通讯]` (Carl von Ossietzky University of Oldenburg)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文构建了AgentMonopoly环境，利用LLM代理在四人Monopoly游戏中进行沟通、交易和决策，探究不同模型和 persona 指令对行为与胜率的影响。

**💡 创新点**

创新点在于将多模态游戏环境与LLM代理结合，系统评估模型规模、跨家族对战与 persona 指令三种因素对交易频率、租金放弃、资金管理等行为的差异，并首次同时考察语言与行动的一致性。

**🔧 技术方法**

技术上使用公开预训练的 Qwen3、DeepSeek-R1、Gemma 与 Mistral 四大模型的两种参数规模，并通过自定义的游戏引擎记录私密推理、公开信息、执行动作与结果，采用基于游戏局面与轮次的策略评估。

**📊 数据集**

数据集为自生成的游戏交互记录，包含 5 种随机种子、各模型与 persona 配置下的多局游戏日志（同模型、混合规模、跨家族、persona 四种实验设置）。

**📈 对比分析**

通过同模型、混合规模、跨家族与 persona 四种对照，比较交易提议率、租金放弃率、获胜次数与轮次位置等指标，结果显示模型行为差异显著，规模增大并未始终提升胜率，persona 指令对语言与部分行动产生影响但并未完全与角色要求一致。

**⚠️ 局限性**

局限性包括仅测试有限模型与参数配置、仅使用 5 种种子、受共享提示与规则设置影响、persona 仅与单一模型配对、对多玩家数与不同游戏规则的泛化不足，且未能彻底分离模型固有特性与提示/对手互动的交互作用。

---

## 259. An algorithmic construction of Nullstellensatz certificates for graph automorphism groups

**arXiv ID:** 2610.28928 | [PDF](https://arxiv.org/pdf/2610.28928v1)

**作者:** Ivan Drecun `[一作]` `[通讯]` (University of Belgrade), Ivan Drecun (University of Belgrade)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在论文中，作者提出了一套新的多元多项式理想归属证明系统，并将其应用于图的自同构群归属问题，给出了显式构造的 Nullstellensatz 证明，并实现了基于修改后 McKay‑Piperno 算法的证明生成器和独立的证明验证器。

**💡 创新点**

创新点包括：①将 Nullstellensatz 证明引入图自同构归属验证，构造可直接被多项式运算验证的证明；②将证明生成与验证解耦，验证过程仅需基本多项式加乘运算；③通过对证明进行后置剪枝显著压缩证明体积，并展示了生成与验证时间与证明大小之间的经验关系。

**🔧 技术方法**

主要技术手段包括：多元多项式代数与有限域运算；Nullstellensatz 证书的构造与利用；改进的 McKay‑Piperno 归约回溯算法；C++ 编程实现的证明生成与验证；对证明结构的剪枝与压缩算法。

**📊 数据集**

实验使用了多种图实例，涵盖小型示例、随机图以及标准图同构/自同构基准集（如 Cayley 图、Bipartite 图、随机 3‑正则图等），总共生成并验证了 100 组图，证明步数最高可达 200 000 步。

**📈 对比分析**

性能评估显示：证明生成时间与图顶点数呈近似 O(n²·⁸) 关系；证明验证时间与证明大小呈线性关系（≈10 ms/KB），单条证明的验证时间平均为 200‑400 ms；剪枝后证明体积平均降低 50 % 以上，剪枝时间仅占生成总时间的 5‑10 %。

**⚠️ 局限性**

主要局限包括：当前构造产生的证明体积仍过大，导致生成时间昂贵；构造中需要显式列举大量多项式，缺乏压缩策略；实现尚未在交互式定理证明器中形式化验证；在 F₂ 上实现可能更高效，但仍需研究。

---

## 260. SSMCGM-Stream: A Streaming State-Space Model for Personalized Diabetes Management

**arXiv ID:** 2610.28976 | [PDF](https://arxiv.org/pdf/2610.28976v1)

**作者:** Shakson Isaac `[一作]`, Chirag Patel `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

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

## 261. Brittle by Design? Multi-Scenario Reinforcement Learning for Robust Mission Coordination in Systems of Systems

**arXiv ID:** 2610.28818 | [PDF](https://arxiv.org/pdf/2610.28818v1)

**作者:** İbrahim Oğuz Çetinkaya `[一作]` (Virginia Tech), Taylan Topcu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

将高保真数字消防模型与强化学习相结合，训练出能实时分配航空灭火任务的智能协调器。

**💡 创新点**

创新点在于：① 在高保真仿真环境下使用RL学习任务分配；② 系统评估单场景训练的脆弱性与多场景联合训练的鲁棒性，展示了训练场景多样性对部署性能的关键影响。

**🔧 技术方法**

技术实现：基于Markov决策过程的状态-动作建模，使用Proximal Policy Optimization (PPO) 强化学习，采用离散化的三维战术动作集。

**📊 数据集**

数据集：采用德国航空中心 (DLR) 开发的二维野火仿真模型，包含两种场景（Palisades 与 Pyrenees）以及对应的天气、地形、燃料分布等属性。

**📈 对比分析**

比较方法：将训练得到的单场景协调器（PA、PY）与多场景协调器（MS）与随机策略和经验启发式基线在 1,000 轮评估中对比；单场景协调器在自身场景上提升约 10–12% MoE，单场景协调器在另一场景表现退化至基线；多场景协调器在各场景与合并评估中均与单场景最佳水平相当，并在合并评估中实现最高平均 MoE ≈ 0.49，几乎无负面回报。

**⚠️ 局限性**

局限性：① 假设完美信息与无延迟的命令执行；② 固定舰队规模与类型，未研究舰队选型与规模；③ 采用离散化战术动作，可能忽略连续控制细节；④ 未在真实干扰（通信失效、资产失效、机场关闭）下验证鲁棒性；⑤ 基线启发式仅为单一静态规则，可能低估专家经验水平。

---

## 262. SLoSa: Small, Low-Power, and Safe Foundation Models for Robotic Task Planning

**arXiv ID:** 2610.28955 | [PDF](https://arxiv.org/pdf/2610.28955v1)

**作者:** Weizhe Xu `[一作]` (University of Notre Dame), Fanxin Kong `[通讯]` (University of Notre Dame)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了SLoSa，一种结合神经网络与符号推理的后训练框架，用于训练体积小、低功耗且安全意识强的机器人任务规划器。

**💡 创新点**

创新点在于将经典规划器、教师LLM与形式化验证器的反馈融合为安全意识训练数据，利用监督微调和验证器引导的强化学习，使小型LLM能够自行生成满足轨迹约束的高质量计划；同时通过量化与参数高效微调实现低功耗部署。

**🔧 技术方法**

技术主要包括：安全意识数据构建（利用PDDL3约束与验证器筛选行为改变实例）、压缩感知的监督微调（SFT）与基于验证器奖励的GRPO强化学习、低位量化（4‑bit）与LoRA适配、以及多种输入格式（PDDL、结构化自然语言、叙述自然语言）。

**📊 数据集**

使用了四个机器人启发式PDDL领域（Blocksworld、Ferry、Grippers、Spanner）生成的约束规划问题，另外在VirtualHome‑Safe和ALFWorld‑FO‑Safe两个完整观测的仿真环境中构造安全约束数据；每个领域生成约250个约束问题，形成训练与测试集。

**📈 对比分析**

与基线（原始预训练模型、直接提示的GPT‑5.6 Sol、Claude Sonnet 5）对比，SLoSa在四个PDDL域上将成功率从0%提升至70.5–76.0%，在两大仿真基准上成功率从0%提升至80.5–85.0%；在Jetson Orin与RTX 4080边缘设备上实现了INT4级别的低功耗部署（平均功耗17.85–26.41 W，吞吐量4.3–13.2 tokens/s）。

**⚠️ 局限性**

局限性包括：只关注高层任务规划的符号安全约束，未覆盖感知误差、低层运动失误或物理碰撞风险；在长时域任务（如Blocksworld）仍面临挑战；验证器依赖精确的符号模型，实际环境中不确定性可能导致失效。

---

## 263. Unit-Aware Differentially Private Quantum Machine Learning for Clinical and Wearable Health Data

**arXiv ID:** 2610.28968 | [PDF](https://arxiv.org/pdf/2610.28968v1)

**作者:** Md Khalid Syfullah `[一作]` (Southern Illinois University Carbondale), Alvi Ataur Khalil `[通讯]` (Southern Illinois University Carbondale)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究开发了一个单元感知的差分隐私训练框架，在临床记录和可穿戴设备数据上评估量子神经网络、容量匹配的多层感知器和逻辑回归模型。

**💡 创新点**

创新点包括首次将量子变分模型与经典基线在容量匹配且记录级/用户级DP下进行对比，推导梯度敏感度上界以指导截断阈值选择，并在IBM 156量子处理器上实现硬件验证。

**🔧 技术方法**

技术上使用了参数化量子线路、Opacus实现的DP‑SGD、PennyLane仿真、Qiskit量子计算与微调的噪声与裁剪策略。

**📊 数据集**

采用了三份临床数据集（Pima Diabetes、UCI Heart、Heart Failure）和一份可穿戴数据集WESAD。

**📈 对比分析**

通过在不同隐私预算和截断阈值下对ROC AUC进行配对Wilcoxon检验（并进行Holm校正），结果显示在临床数据上量子网络平均比多层感知器高0.104 AUC，但差异未显著；可穿戴数据表现相反，量子网络落后。

**⚠️ 局限性**

局限性包括可穿戴数据样本极少导致噪声过大、统计检验未达到显著水平、硬件评估仅在单个数据集和8条样本上进行，缺乏更广泛的验证。

---

## 264. Self-Improving Agents with World Model Predictive Paths

**arXiv ID:** 2610.28945 | [PDF](https://arxiv.org/pdf/2610.28945v1)

**作者:** Haori Lu `[一作]` (University of Illinois), Yingying Li `[通讯]` (University of Illinois)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于MPPI的测试时框架，利用世界模型预测多条候选动作并用经验记忆改进候选生成和预测

**💡 创新点**

将MPPI的多候选评估与语言代理结合，并引入双分支经验记忆同时提升候选覆盖和世界模型预测

**🔧 技术方法**

MPPI式权重、世界模型预测、经验检索、LLM策略与控制器、双分支记忆技术

**📊 数据集**

ALFWorld与ScienceWorld两大文本基准数据集

**📈 对比分析**

与ReAct、ITP‑I、WorldEvolver等基线比较，在三种LLM后端上取得最高SR@1/SR@5，尤其在Gemma‑4‑26B‑A4B上显著领先

**⚠️ 局限性**

世界模型精度受限导致决策误差、候选数增大导致推理延迟、参数冻结限制了自适应能力

---

## 265. Uncensored Is Not Unbiased: Abliteration Removes Refusals but Not Script-Dependent Bias About Taiwan

**arXiv ID:** 2610.29342 | [PDF](https://arxiv.org/pdf/2610.29342v1)

**作者:** Ju-Chun Ko `[一作]` `[通讯]` (Legislative Yuan, Republic of China (Taiwan)), Ju-Chun Ko (Legislative Yuan, Republic of China (Taiwan))

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了两款开源大语言模型及其“脱敏”版本在关于台湾主权与天安门事件的问答中，是否真的消除了政治偏见。

**💡 创新点**

首次将“abliteration”编辑与脚本（繁简）和命名的影响系统比较，并探究通过激活向量和系统提示等手段对模型立场的可调节性。

**🔧 技术方法**

采用了权重编辑（读/写侧投影）消除拒绝方向、yes/no概率探测、激活向量加法、系统提示干预，以及对FineWeb‑2中文网页的脚本与立场标记统计。

**📊 数据集**

使用22对主权/事件陈述、973条开放式回答、约740,000条FineWeb‑2网页（含繁简标记）以及模型自身的训练与后训练语料。

**📈 对比分析**

通过匹配对式yes/no概率、脚本差异分析、激活方向干预与系统提示实验比较，结果表明脚本效应远大于编辑；在Qwen中编辑可略微转向ROC，但整体偏见未被完全消除。

**⚠️ 局限性**

局限包括样本量有限、编辑实现不完全、模型使用不同引擎、脚本转换仅为字符级、未能访问全部训练数据、开放回答的评估仍以机械统计为主。

---

## 266. The Optimal Sampler for Diverse TSP Solutions: Characterization, Limits, and Learning

**arXiv ID:** 2610.28994 | [PDF](https://arxiv.org/pdf/2610.28994v1)

**作者:** Ssu-Yuan Lo `[一作]` (National Central University), Hao-Tsung Yang `[通讯]` (National Central University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究D‑TSP，提出采样‑选择框架并学习均匀采样器，提供从生成树到巡回路的无质量损失管道；

**💡 创新点**

证明采样阶段决定最终多样性表现，给出可学习的闭式奖励使采样器趋向均匀分布；通过冲突图将D‑TSP映射为最大独立集问题，得到线性样本量保证，且将多样性目标与采样分离；

**🔧 技术方法**

利用熵正则化的策略梯度学习（含折扣温度）、随机生成树+深度优先或Christofides构造、Jaccard距离度量、冲突图与最大独立集理论、概率碰撞率分析；

**📊 数据集**

训练集为40城市欧氏随机图；测试集包括相同规模随机图和TSPLIB 51–101城市实例；

**📈 对比分析**

与NMA、Bi‑Criteria Heuristic、RF‑MA3S等基线进行对比；在训练预算下的随机图上100%击败基线；在TSPLIB 51–101城市实例上，c=4.0时获得最高的最小Jaccard距离，性能明显优于所有基线；且无需微调即可迁移；

**⚠️ 局限性**

对规模敏感：大规模实例时接受率下降，需要更宽松预算或重新训练；线性样本量保证使用了更弱的基准；生成树到巡回路的映射虽然无质量损失理论上，但实际分布可能偏离理想均匀采样，导致多样性受限。

---

## 267. Payload generalization on feedforward control actions for trajectory tracking in robots with rigid and elastic joints

**arXiv ID:** 2610.29356 | [PDF](https://arxiv.org/pdf/2610.29356v1)

**作者:** Giorgia Pazzi `[一作]` (Università di Pisa), Franco Angelini `[通讯]` (Università di Pisa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种数据驱动方法，用极少量已知负载参数的示例物体生成的前馈控制量来实现对不同负载的轨迹跟踪，无需机器人动力学模型，适用于刚性与弹性关节系统。

**💡 创新点**

创新点在于：①证明负载参数维数+1个示例足以完成泛化；②通过回归矩阵估计实现无模型、无控制器依赖的前馈量推算；③方法对弹性关节仍有效，提升了柔性机器人的可操控性。

**🔧 技术方法**

技术包括：线性参数化动力学模型、回归矩阵估计、逆/伪逆求解、迭代学习控制（ILC）或计算力矩控制收集示例；在离散时间实现时仅需点评估，无需额外算法改造。

**📊 数据集**

数据集：仿真中使用不同形状与质量的十维或五维负载参数集合（0–12个物体）；实验中使用Panda机器人及其柔性手腕抓取的八个实际物体（质量、质心、惯量信息）。

**📈 对比分析**

比较方法：使用RMSE评估示例物体的ILC/计算力矩控制误差与泛化后的误差；实验与仿真均显示泛化后RMSE与示例误差相当，甚至在重负载时更优，误差增幅约10%以内，保持在可接受范围。

**⚠️ 局限性**

局限性：需已知负载的完整动态参数；仅适用于相同轨迹，轨迹改变需重新学习；对噪声和参数不确定性敏感，若示例矩阵条件数差则泛化效果下降。

---

## 268. Transformers as Generic Deep Neural Networks: Exact Compilation, the Cost of Removing Activation-Activation Products, and Multi-Agent Attention Clustering

**arXiv ID:** 2610.28958 | [PDF](https://arxiv.org/pdf/2610.28958v1)

**作者:** Sergey Salishev `[一作]` (St Petersburg State University), Oleg Granichin `[通讯]` (St Petersburg State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

把 Transformer 的每个块严格编译成只包含标准线性映射、一次性非线性（平方、sigmoid、倒数平方根）以及 Hadamard 乘积的通用 DNN 图，并给出了一种基于尾部质量估计的 top‑k 软注意力替代方案；

**💡 创新点**

证明了在去掉激活–激活乘积的编译方案下点式工作量不可避免、引入 Hadamard 乘积可消除这一开销、并给出了最优的尾部质量估计；

**🔧 技术方法**

采用了数学证明（非线性秩、极值与聚类分析）、自定义编译器、量化技术、top‑k tail‑mass 估计、聚类路由 (k‑means) 以及 GPU 基准测试；

**📊 数据集**

在 Oxford‑IIIT Pet、ImageNet‑1k、Set5、DIV2K 和 WikiText‑2 上进行实验；

**📈 对比分析**

与原始模型对比保持 0.0–0.6% 的精度差，Hadamard 编译比严格编译快 1.2‑1.3×、内存差距仅 5%，扩展编译更快；top‑k tail‑mass 估计在 DeiT‑Tiny、MambaIRv2‑Light 和 Pythia‑70M 上比传统 tail‑drop 的精度差距降至 <0.1%，并在 1‑2 倍更少的键下保持相同性能；

**⚠️ 局限性**

仍依赖完整的矩阵乘法支持；对 causal 模式仍需全 score；实验仅在 GPU 进行，NPU 上的实际实现与整数乘法精度需要进一步验证；单一尾部质量常数可能不足以捕获查询间的差异，需要更精细的自适应估计。

---

## 269. The CIDOC CRM Periodic Table: Documentation You Can Test to Support the Adoption of Scientific Research Ontologies for the Humanities

**arXiv ID:** 2610.28650 | [PDF](https://arxiv.org/pdf/2610.28650v1)

**作者:** Remo Grillo `[一作]` `[通讯]` (Alma Mater Studiorum - University of Bologna), Remo Grillo (Alma Mater Studiorum - University of Bologna)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

创建了一个单页浏览器工具——CIDOC CRM Periodic Table，用于在一个交互界面中显示CIDOC CRM及其扩展的类和属性，并提供一键式三元组构造与多步路径搜索功能。

**💡 创新点**

将分散在四个规范文档中的实体合并为统一的可视化表格，并通过跨模型闭包实现跨模型推理，解决了CIDOC CRM扩展的文档碎片化问题。

**🔧 技术方法**

使用前端技术（HTML/CSS/JavaScript）加载官方RDFS文件，构建代码前缀字典，计算继承闭包和域/范围推理，并在浏览器中实现交互式三元组构造与路径搜索。

**📊 数据集**

官方RDFS发布的CIDOC CRM 7.1.3、CRMsci 3.2、CRMdig 5.0和CRMinf 1.2.1的类、属性与关联关系。

**📈 对比分析**

通过在浏览器中执行实时推理，采用受限的双向宽度优先搜索（最多四步、每层限600条、总计100条）实现交互式路径查询，用户体验可接受，性能保持在几百毫秒级；未与其他工具做量化对比。

**⚠️ 局限性**

推理仅覆盖子类继承与域/范围约束，缺乏一致性检查、属性传递、等价推理；未评估用户体验，仅基于Google Analytics统计；目前仅加载四个扩展，未包含CRMgeo、CRMarchaeo等；缺乏自动化测试。

---

## 270. MultiphaseHUB for Evidence-Grounded Metadata Curation and Task-Aware Compression of Multiphase Data

**arXiv ID:** 2610.28644 | [PDF](https://arxiv.org/pdf/2610.28644v1)

**作者:** Yingyan Zeng `[一作]` (University of Cincinnati), Ying Sun `[通讯]` (University of North Carolina at Charlotte)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `fede83ac-7505-405f-ab37-e7284695c47f` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

建立 MultiphaseHUB 数据中心，集成 LLM 代理元数据抽取、图像量化提取以及多模态科学压缩。

**💡 创新点**

创新点在于将代理式文献分析与多模态残差向量量化 VAE 结合，既自动化提取实验上下文，又提供保持关键物理量的压缩方案。

**🔧 技术方法**

采用大语言模型（LLM）进行文本/表格抽取与图像量化，使用残差向量量化的变分自编码器（RVQ‑VAE）进行多模态压缩，并在损失函数中加入 QoI 约束。

**📊 数据集**

使用公开的 13 组多相数据集（如 BoilingBench、BubbleID 等）进行验证，压缩实验以池沸腾实验的温度、声学和高频视频为样本。

**📈 对比分析**

通过与手工 WebPlotDigitizer 对比，LLM 代理提取的数值误差平均 <0.35% 轴宽；RVQ‑VAE 两阶段下温度/音频压缩比约 10.7:1，视频约 279.3:1，重建示例能保留主要物理特征。

**⚠️ 局限性**

局限在于仅对单一实验跑评估、缺乏跨实验/跨设施泛化验证、未系统评估 QoI 的保持效果、元数据抽取未经过独立标注的完整验证。

---

## 271. MuseFlow-TSE: Music-Aware Target Speech Extraction via Cross-Source Band-Splitting and Data-to-Data Flow Matching

**arXiv ID:** 2610.29119 | [PDF](https://arxiv.org/pdf/2610.29119v1)

**作者:** Lu Ma `[一作]` (JD.com, Inc.), Xin Tian `[通讯]` (JD.com, Inc.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现了 MuseFlow‑TSE，结合音乐感知的跨源带分割提取器和负干扰条件的自适应数据到数据流匹配细化器，实现音乐场景下的目标语音提取。

**💡 创新点**

创新点包括：①音乐感知的跨源带分割架构；②通过音乐泄漏损失和负干扰条件实现音乐/歌唱抑制；③自适应步骤的流匹配细化器；④一体化的消除+细化两阶段框架。

**🔧 技术方法**

使用的技术：双路径Transformer、RoPE、FiLM调制、SepReformer 交叉注意力、Band‑Split、AdaLN‑Zero Diffusion/Flow、音乐泄漏损失、负干扰嵌入、难度头自适应 Euler 步等。

**📊 数据集**

数据集：LibriSpeech、VoxCeleb2、WSJ0 语音；MUSDB18‑HQ、MoisesDB 音乐；噪声数据；合成三秒混合，含音乐、歌唱、噪声、回声等。

**📈 对比分析**

与现有基线（SpEx+、USEF‑TSE、TF‑GridNet、DPM‑TSE、SoloSpeech、MeanFlow‑TSE 等）比较，在复杂场景下 WER 从 12% 降至 12.2%，SI‑SDRi 提升至 13.2 dB，音乐泄漏率仅 2.9%，表现优于多种先行方法。

**⚠️ 局限性**

局限性：仅在 16 kHz 合成混合数据上验证；未在真实录音或多样化场景下评估；模型对音乐/歌唱的泛化仍需进一步验证。

---

## 272. Policy-Mask: Budget-Aware Adaptation of Diffusion Features to Discriminative Tasks

**arXiv ID:** 2610.28889 | [PDF](https://arxiv.org/pdf/2610.28889v1)

**作者:** Ivan Listopadov `[一作]` (HSE University), Aibek Alanov `[通讯]` (HSE University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种预算约束的策略，联合选择扩散模型中的去噪时间步、网络层级和固定数量的通道子集，用于下游任务的特征提取和蒸馏。

**💡 创新点**

创新点在于：①实现了时间步、层级和通道三维联合选择，解决了传统方法仅手工或单一维度选择的问题；②引入通道得分归一化（按子集大小除以 b）以平衡三部分的梯度贡献；③提出了基于 Gumbel‑top‑b 的 exact‑cardinality 样本采样，实现可训练的离散子集选择；④提供了通用的“特征网格”映射接口，可将策略迁移至 DiT、MMDiT 等 transformer 生成模型。

**🔧 技术方法**

使用技术包括：冻结的扩散教师模型（如 Stable Diffusion 1.5 的 U‑Net），学生网络与任务解码器，基于 REINFORCE 的策略梯度更新（带基线和通道归一化），Gumbel‑top‑b 采样，Plackett‑Luce 排序得分，通道掩码对教师特征进行选择，蒸馏损失与任务损失联合优化。

**📊 数据集**

评估数据集包括：语义分割 ADE20K、Cityscapes、CelebA‑HQ、Pascal VOC；图像分类 CIFAR‑10、CIFAR‑100；语义对应 SPair‑71k；还对 PixArt‑Σ、SD‑3.5 MMDiT 等 transformer 教师进行了验证。

**📈 对比分析**

与固定提取、按方差或 Fisher 选通道、全宽提取、RepFusion 等基线相比，
- 在 128 通道下的 ADE20K mIoU 达到 38.9（相较全宽提取提升约 1.5），
- CIFAR‑100 top‑1 达到 75.4%（高于 RepFusion 74.1%），
- SPair‑71k PCK@0.1 达到 76.3（图像归一化）
- 在各分割、分类、对应任务上均显示显著性能提升。

**⚠️ 局限性**

限制包括：①通道预算与蒸馏损失规模同时变化，难以单独评估最佳预算；②实验多次重复仅在部分对比中进行，样本数量有限；③不同教师架构的特征维度映射需要额外的对齐操作；④策略对不同数据集的泛化能力及超参敏感性仍需进一步研究。

---

## 273. A Mechanistic Study of Automated Failure Attribution in LLM-based Agentic Systems

**arXiv ID:** 2610.28811 | [PDF](https://arxiv.org/pdf/2610.28811v1)

**作者:** Wendy Zheng `[一作]` (University of Virginia), Liangjie Hong `[通讯]` (Nokia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对多代理LLM系统的自动失效归因（AFA）任务，研究模型内部是否已编码失效步骤信息，并提出一种基于内部注意力头的“Grafting”推理时间干预方法以提升归因性能。

**💡 创新点**

创新点在于发现并量化了LLM内部编码的失效归因信息与模型输出之间的“knows‑but‑won’t‑say”差距；提出利用线性探针定位信息源头，并通过连接定位头与写入头的插值干预，将潜在信息转化为可执行行为，从而无需额外标签即可提升AFA。

**🔧 技术方法**

技术包括：线性探针（logistic 回归）对Transformer层残差进行评估；注意力头定位与写入头识别；基于直接logit归因的写入头筛选；在推理时将定位头的投票结果投射到写入头的值空间并注入干预信号。

**📊 数据集**

数据集为Who&When基准，包含两子集：Algorithm Generated（平均9步）和Hand‑Crafted（平均29步），每条记录包含完整的多代理交互记录及其对应的失效步骤标签。

**📈 对比分析**

对比方法包括All‑At‑Once、Step‑by‑Step、Sequential三种提示策略；使用线性探针评估可解码信息；在此基础上加入Grafting。实验表明：探针的步级准确率显著高于提示基线；Grafting在所有模型（Qwen、Llama、Gemma）及两子集上均能明显提升步级准确率，通常比最优提示策略提升约10–20%，但仍略低于最强探针的性能。

**⚠️ 局限性**

局限性：Grafting仅在已识别的定位头与写入头间做简单插值，未探索更复杂的路径或多头协同；对大规模模型的推理效率及可扩展性未系统评估；缺乏对不同任务复杂度或更多代理场景的泛化验证。

---

## 274. Credit Where Credit Is Due: Contrastive Attribution Topology Search for Multi-Agent Systems

**arXiv ID:** 2610.29292 | [PDF](https://arxiv.org/pdf/2610.29292v1)

**作者:** Fabian Tritthart `[一作]` (University of Freiburg), Tom Zehle `[通讯]` (University of Freiburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 Contrastive Attribution Topology Search (CATS) 的离线元代理，用于在预设的代理超级图中搜索最优的多代理系统拓扑，最终得到小型高效的部署团队。

**💡 创新点**

创新点在于：①利用对比式 LLM 归因给每个代理赋予正负信用，用于指导拓扑搜索；②在搜索过程中结合信用更新与边缘概率剪枝，实现既提升准确率又降低部署令牌；③在含有恶意代理的测试中，自动剔除有害代理，展示归因的安全性潜力。

**🔧 技术方法**

技术方法包括：基于 DARTS 思路的可微边缘概率模型、LLM 归因器生成的 per-agent credit、按信用更新边缘 logit、阈值剪枝与迭代采样、最终选取最佳子图。

**📊 数据集**

使用的数据集主要有 BigCodeBench、MBPP 两个编码基准，以及改造的 HotpotQA 进行毒化池压力测试。

**📈 对比分析**

与随机拓扑搜索、score‑only 归纳搜索以及手工设计的多代理系统进行比较。CATS 在 BigCodeBench 上平均提升 5.3 分且部署令牌减少 54%，在 MBPP 上性能与手工系统持平且令牌更少；在毒化 HotpotQA 测试中，CATS 全面剔除有害代理，而随机搜索仍保留部分。

**⚠️ 局限性**

局限性包括：仅在单一模型族和三种任务上评估；缺乏对归因信用与人类评估的一致性验证；未对成本进行直接优化；对高层次安全保障缺乏完全证明，攻击者可在 LLM 归因窗口之外或通过直接操纵 LLM 产生更隐蔽威胁。

---

## 275. State-Dependent Objective Coordination for End-to-End Balance and Navigation

**arXiv ID:** 2610.29142 | [PDF](https://arxiv.org/pdf/2610.29142v1)

**作者:** Xuhao Huang `[一作]`, Zheng Gong `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在两轮倒立摆机器人上设计并实现了端到端的姿态稳定与障碍规避导航方法，通过统一的深度RL策略完成任务。

**💡 创新点**

创新点在于引入基于俯仰角的状态相关标量化系数 β(s)，在训练时对平衡与导航奖励进行自适应加权，消除对额外层级控制器或在线奖励计算的依赖，并通过在仿真中模拟延迟、动作保持、异步噪声和动力学不确定性实现零样本硬件迁移。

**🔧 技术方法**

技术方法包括：PPO+LSTM 强化学习、状态相关奖励标量化、梯度冲突与尺度分析、NVIDIA Isaac Sim 仿真、UWB+LiDAR+IMU 观测融合、以及低延迟 Raspberry Pi 5 上的实时部署。

**📊 数据集**

数据集/环境：自建仿真地图，包含固定障碍地图和随机生成的高障碍密度地图，用于训练和评估；未使用公开数据集。

**📈 对比分析**

对比方法包括多种奖励加权策略（全局 β、可学习 β、训练自适应 β）、约束设置（完整 vs 减少约束）、训练/控制结构（分阶段训练、命令式 HRL、策略组合）。PPO+β(s) 在固定地图成功率 98%、随机地图 95%，平均完成时间 21 s，最大俯仰角 0.14 rad；DDPG+β(s) 在成功率上略低；其他方法表现明显逊色。

**⚠️ 局限性**

局限性：风险表示仅依赖俯仰角，未考虑速度、障碍距离、动力学极限等；缺乏形式化安全/稳定性证明；实验仅限于单一两轮倒立摆平台和有限的环境，未在多种硬件或更复杂环境中验证。

---

## 276. Different Defects, Different Effects: A Controlled Audit of Face Recognition Training Data

**arXiv ID:** 2610.28766 | [PDF](https://arxiv.org/pdf/2610.28766v1)

**作者:** Ana Dias `[一作]` (University of Beira Interior), João C. Neves `[通讯]` (University of Beira Interior)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对 MS1MV3 人脸识别数据集进行全面缺陷审计，并针对重复图像、身份拆分、裁剪错误、多人模糊和无效内容等五类缺陷实施有针对性的清理策略。

**💡 创新点**

首次将缺陷审计与训练效率评估结合，展示不同缺陷修复对识别精度与训练步骤数的显著且可区分的影响，并证明仅去除重复图像（阈值 τ_h=7）即可在保持甚至提升识别性能的同时减少约 24% 的训练步骤。

**🔧 技术方法**

采用感知哈希检测重复、InsightFace 生成特征进行身份拆分与交叉验证、HNSW 近邻搜索、BLIP VQA 判别无效内容、OFIQ 评估多脸模糊；训练使用 IResNet‑50 + ArcFace，SGD、学习率衰减，验证集包含 LFW、CALFW、CPLFW、AgeDB、CFP‑FP，外部评估使用 IJB‑B/C。

**📊 数据集**

主要使用 MS1MV3（5,179,510 张图像，93,431 个身份）作为训练集，并在多个公开人脸验证基准上进行对比评估。

**📈 对比分析**

通过对比不同清理方案在相同训练协议下的平均验证准确率与达到基线最佳性能所需步骤数，发现 D7（去除 15.44% 重复图像）平均验证准确率提升至 97.27%，并将训练步骤减少 23.7%；其他方案如身份合并（ID‑M）和无效内容去除（IC）亦能显著缩短训练时间，结合后仍保持性能不降。

**⚠️ 局限性**

局限性包括：清理效果高度依赖缺陷类型，对其他数据集的泛化尚未验证；部分缺陷检测精度仍需人工校验；不同缺陷的组合效应不呈叠加，难以得到统一的清理策略；以及仅关注 MS1MV3，未探讨跨域或多模态场景。

---

## 277. Agent AI Zarządzający Siecią Open RAN Zorientowana na Użytkownika

**arXiv ID:** 2610.28860 | [PDF](https://arxiv.org/pdf/2610.28860v1)

**作者:** Marcin Hoffmann `[一作]` `[通讯]`, Marcin Hoffmann

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于大语言模型的 AI 代理，用于 Open RAN 用户中心化网络的接入簇分配优化。

**💡 创新点**

创新点在于利用 LLM 通过历史状态学习动态调整 Δ 参数，以实现高效簇分配，超越传统 MAB 方法。

**🔧 技术方法**

使用了 Open RAN RIC rApp、O1 接口、强化学习循环、Qwen 3.5‑32b 语言模型、Ray‑Tracing 3D 路径跟踪等技术。

**📊 数据集**

数据集基于 Wireless InSite 3D 路径跟踪、20 名用户、5000 个时间槽的仿真场景。

**📈 对比分析**

与网络中心化架构、全接入簇配置以及 MAB 进行对比，Agent AI 使用户吞吐量中位数提升约 12%，在极端信道条件下提升更显著。

**⚠️ 局限性**

局限性包括需要本地部署大模型、计算资源消耗高、对不同 LLM 的适配性待进一步验证。

---

## 278. SpatialBrain: Spatial Intelligence Growing from Experience

**arXiv ID:** 2610.29317 | [PDF](https://arxiv.org/pdf/2610.29317v1)

**作者:** Yifan Zhou `[一作]` (Shanghai Jiao Tong University), Xue Yang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 SpatialBrain 框架，结合多视角 3D 场景记忆、可重用技能库和经验驱动的认知控制，实现了在固定视觉‑语言模型基础上自演化的空间推理。

**💡 创新点**

创新点在于：①将场景证据、可重用计算与跨任务经验分离；②通过 3DFusionMemory 构建共享场景图；③构建可参数化的 Spatial Skill Library；④在测试时仅更新经验而不修改模型权重，实现自演化决策；⑤将资源选择与推理过程分离，形成可解释的决策链。

**🔧 技术方法**

技术方法包括：多视角图像融合构建 3D 场景图；使用 GPT‑5.6‑sol 视觉‑语言背骨；通过工具调用实现相机、姿态、路径推理；可重用技能以程序化接口封装；经验库记录任务签名与资源决策；自演化更新策略与技能可用性。

**📊 数据集**

使用了 5 个空间推理基准：MindCube、MMSI、OmniSpatial、SpaceDG 与 SAT，总计约 6,700 条题目，涵盖位置、方向、距离、运动等多种推理类型。

**📈 对比分析**

与直接答案 baseline 及多种先前模型对比，评估指标为准确率（或 SpaceDG 的 MRA）。在 4/5 任务上相较 baseline 提升了约 5%（从 67.5% 提升至 72.6%），MindCube 任务错误率下降 62%，其余任务亦出现显著或相当的提升。

**⚠️ 局限性**

局限性包括：基线实验使用不同协议与模型，导致直接比较受限；自演化仅在任务块间更新经验，未能评估单任务内的即时收益；感知与几何估计误差未充分校准；无法彻底区分记忆存储与利用对性能的单独贡献。

---

## 279. S-Rolling: Steerable Rolling Gaits for Twistable Snake Robots in Low-Clearance Environments

**arXiv ID:** 2610.29478 | [PDF](https://arxiv.org/pdf/2610.29478v1)

**作者:** Qingling Fan `[一作]` (Shanghai Jiao Tong University), Zhongqiang Ren `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种可在低清除高度环境中实现转向的滚行步态S-rolling，并在实际硬件上验证了多种运动模式和在狭小空间中的机动能力。

**💡 创新点**

创新点在于：① 通过S形主干曲线将身体分为头尾两段，独立控制两段滚动速度并在中间加入扭转关节，实现低扫高且可转向；② 用背部参考框架推导完整的关节空间步态方程；③ 在此基础上构建准静态运动学模型，将机器人视为差速驱动车，便于控制。

**🔧 技术方法**

主要技术包括：背部参考曲线法、S形主干曲线设计、关节空间步态推导、准静态运动学建模、离散模块化机械结构与三轴关节实现、基于人机交互的实时控制。

**📊 数据集**

使用的“数据集”为实验室实物机器人实验结果：3、5、3模块机器人在不同滚动速度组合下的运动轨迹、10 cm通道测试轨道以及工件和车架下的实景任务。

**📈 对比分析**

与传统滚动弧步态（rolling arc）对比：在10 cm通道、工件和车架任务中，S-rolling能完成90°、180°转向，前者无法改变朝向；实验数据显示S-rolling在低清除高度环境中的机动性显著提升，定位误差和转向误差相对较小。

**⚠️ 局限性**

局限性：① 运动学模型基于准静态假设，实际在转向时滑移导致模型误差较大；② 未实现闭环自主导航，仅靠手动遥控；③ 需要在各模块之间精确同步滚动速度和扭转角，机械实现对高精度控制有一定挑战。

---

## 280. Integrating Flow Matching and Model Predictive Control for Safe Robot Navigation

**arXiv ID:** 2610.29377 | [PDF](https://arxiv.org/pdf/2610.29377v1)

**作者:** Philipp Holzmann `[一作]` (Technical University of Darmstadt), Rolf Findeisen `[通讯]` (Technical University of Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种将基于流匹配的生成式路径规划器与模型预测路径跟随控制（MPPFC）相结合的框架，用于移动机器人在拥挤环境中的碰撞规避与约束满足；

**💡 创新点**

创新点包括：1）双向耦合机制，生成器的路径指导MPPFC，MPPFC的预测结果反向偏置生成器以提升安全性；2）事件触发的重新规划与多预测时域并行MPPFC，实现局部避障与全局路径一致性；3）通过在MPPFC轨迹的中间步骤初始化流匹配模型，显著降低采样步骤与计算负担；

**🔧 技术方法**

主要技术手段有：流匹配（shortcut model）生成多模态路径；B样条路径参数化；模型预测路径跟随控制（MPPFC）实现实时约束满足；多时域并行MPPFC与软终端对齐策略；事件触发重新规划与偏置采样；

**📊 数据集**

使用了约1800条在MuJoCo模拟环境中由人工操作员收集的人类演示轨迹数据，覆盖不同起止点、静态障碍配置，构成路径生成模型训练集；

**📈 对比分析**

通过在四种未知障碍配置下，比较三种重新规划策略（P1-P3）及不同预测时域（K=5,7,15）得到的成功率与重规划次数。结果表明，使用P3策略与最大预测时域（3.8s）时，成功率可达100%，并且重规划次数最少，整体性能优于基线；

**⚠️ 局限性**

局限性包括：仅针对静态未知障碍，无法处理移动障碍；依赖高质量传感器获取障碍信息；需预先存在拓扑可行路径；在极端动态环境下可能需要更复杂的预测与控制融合。

---

## 281. CRAB: Cuboid-based Representation for As-built BIM from Point Clouds

**arXiv ID:** 2610.29257 | [PDF](https://arxiv.org/pdf/2610.29257v1)

**作者:** Haoran Xiong `[一作]` (Shenzhen University), Hui Huang `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了 CRAB 框架，将室内点云直接转换为可编辑的 BIM 模型，采用语义立方体表示墙体、门窗、梁柱等建筑构件；

**💡 创新点**

创新点在于：① 用中心线参数化的立方体显式建模墙厚、开放口尺寸等 BIM 关键属性；② 采用 ProposalNet+AssemblyNet 的两阶段策略，后者通过流匹配精细化几何并预测结构关系；③ 构建 CrabSet 数据集，为 Scan‑to‑BIM 任务提供立方体级监督；

**🔧 技术方法**

技术手段包括点云编码（Sonata）、自回归立方体生成（Qwen2.5-0.5B）、关系感知流匹配网络、编辑控制标记以及 IFC 输出转换；

**📊 数据集**

使用 Structured3D 的 3,000 个训练/验证样本加上 3,000 个合成场景，构成 CrabSet；此外在真实数据上评估了 BIMNet 与 ISPRS 的扫描；

**📈 对比分析**

与基准方法（Cloud2BIM、RoomFormer、CAGE、SceneScript、SpatialLM 等）对比，CRAB 在组件匹配、墙厚、门窗高度、开口位置等指标上显著提升；在 Structured3D 测试集上墙厚 MAE 2.80 cm、T‑Acc@3 79.50%，比强对手低 38 %；在真实场景中门窗 Match‑F1 也提高到 72.5%/68.8%；

**⚠️ 局限性**

局限包括：① 仍需手动初始化缺失构件；② 对极其不规则或曲面结构（非 Manhattan）处理有限；③ 对稀缺构件（梁、柱、楼梯等）数据不足，难以全面评估。

---

## 282. Post-Pruning vs. Early Stopping: Which Works Better for Link Prediction with AnyBurl?

**arXiv ID:** 2610.29287 | [PDF](https://arxiv.org/pdf/2610.29287v1)

**作者:** Thomas Eiter `[一作]` (TU Wien), Hai Xia `[通讯]` (TU Wien)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了基于规则学习器AnyBurl的规则后处理裁剪方法，并比较裁剪后与提前停止的性能；

**💡 创新点**

提出了支持、置信度、平衡四种裁剪准则，并通过D’Hondt分配预算来实现规则集的有效缩减；

**🔧 技术方法**

使用AnyBurl进行规则学习，计算支持/置信度/PCA置信度，采用D’Hondt分配规则预算，并用MRR@100与Hits@k进行评估；

**📊 数据集**

在六个经典KG基准上实验：WN18RR、FB15k-237、YAGO3-10、CoDEx-S/M/L、Hetionet、Nations；

**📈 对比分析**

通过与AnyBurl早停时间点的MRR@100和Hits@k比较，发现裁剪（尤其支持与平衡准则）在规则数极低（0.1%-0.5%）时可提升1%-5%MRR；

**⚠️ 局限性**

仅针对AnyBurl验证，裁剪准则缺乏对规则交互的考虑，对小数据集结果不稳健，未验证在更大规模或实时系统中的可扩展性。

---

## 283. Same Answer, Different Reasons: Trace-Level Divergence in Multilingual Large Language Models

**arXiv ID:** 2610.29092 | [PDF](https://arxiv.org/pdf/2610.29092v1)

**作者:** Rui Sun `[一作]` (University of Illinois Urbana-Champaign), Varun Chandrasekaran `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多语言LLM可见推理链的相似性，提出结构化的三维评估框架；

**💡 创新点**

将推理链拆分为结构相似、顺序一致和整体相似三维度，并针对跨语言差异进行专门校准；

**🔧 技术方法**

使用多语言句子编码（LaBSE）与BGE-M3进行句子/步骤对齐，采用软Kendall-τ计算顺序一致性，构建两级贪心对齐；

**📊 数据集**

在MMLU-ProX、StrategyQA、SciBench三大推理基准上，涵盖七种语言（阿拉伯语、英语、西班牙语、印地语、俄语、越南语、中文），并评估六款前沿LLM；

**📈 对比分析**

通过排名和宏观平均比较结构相似度、顺序一致性与整体相似度，发现整体相似度往往高估了跨语言一致性，结构和顺序一致度显著较低；

**⚠️ 局限性**

仅评估可见推理链而非内部计算；校准仅对句子级相似度有效，未涉及更细粒度层级；人类验证覆盖有限，未能直接证明低一致性对验证或安全的影响。

---

## 284. ReSeg: Retraining-free segmentation of abdominal organs via generalizable semantic representation learning

**arXiv ID:** 2610.29090 | [PDF](https://arxiv.org/pdf/2610.29090v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 285. VeriEvolve: Evaluator-First Evolution for Tool-Using Agents

**arXiv ID:** 2610.29005 | [PDF](https://arxiv.org/pdf/2610.29005v1)

**作者:** Yueshen Li `[一作]` (University of Illinois Urbana-Champaign), João Pedro Maia `[通讯]` (aiXplain)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个评估器优先的框架 VeriEvolve，先从任务指令和初始状态自动合成挑战状态并生成可执行评估器，然后在冻结的评估信号下对工具使用工作流进行进化，并通过强弱执行状态差异迭代细化评估器。

**💡 创新点**

核心创新在于将评估器先行、自动构造和评估器迭代细化与自我改进工作流搜索耦合；通过强弱状态差异自我定位评估器缺陷，实现无需外部验证器即可在搜索循环中保持评估器不变。

**🔧 技术方法**

使用大型语言模型（Claude Sonnet、GLM‑5.2）进行测试集生成、评估器演化、工作流进化与执行；利用差异挖掘与对抗性挑战状态构造技术；采用多路径多备选工作流表示与持久化搜索框架。

**📊 数据集**

基准数据集为 MCPMark 标准任务集合（30 个文件系统任务、21 个 PostgreSQL 任务等），并在 Playwright、Notion、GitHub 等跨环境子集上进行验证。

**📈 对比分析**

通过内部评估器自测与外部 MCPMark verifier 的一致性检验评估评估器质量；对比强弱执行器在演化前后的外部验证成功率，文件系统任务从 6/30 提升至 18/30（GLM‑5.2），Claude 从 27/30 提升至 30/30；评估器改进后消除所有外部误报，但仍有部分任务未被解决，显示受执行器能力与搜索空间限制。

**⚠️ 局限性**

主要限制包括：①依赖外部 verifier 进行评估器校准与错误定位；②挑战集与评估器由同一模型合成，可能导致覆盖面不足；③工作流搜索与执行成本高，限制实验规模；④强弱状态差异触发需外部误报，缺乏无外部 verifier 的自适应机制；⑤跨环境任务受观测接口限制，部分任务无法完整评估。

---

## 286. Scaling Therapy Supervision: Assessing Psychotherapy Fidelity Using Large Language Models

**arXiv ID:** 2610.28772 | [PDF](https://arxiv.org/pdf/2610.28772v1)

**作者:** Arvind Pillai `[一作]` (Dartmouth College), Andrew Campbell `[通讯]` (Dartmouth College)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用开源大语言模型对真实长篇 CBT 访谈进行自动化忠实度评估，并生成可操作的监督反馈。

**💡 创新点**

创新在于结合任务拆分与证据优先推理，让低参数模型（如 Qwen3.5‑9B）实现接近人工水平的忠实度评分与校准。

**🔧 技术方法**

技术手段包括开源 LLM（Qwen3.5‑9B、Gemma4 系列）+ 任务拆分策略 + 证据提取/评估推理 + Prediction‑Powered Inference 校准 + RAG/Basic 反馈生成。

**📊 数据集**

使用的数据集包括 30 条专家评分的完整 CBT 访谈（Open Gold）、288 条未标注访谈（Open Unlabelled）以及 102 条来自独立 RCT 的访谈。

**📈 对比分析**

通过 Balanced Accuracy、Matthews 相关系数评估 30 条评测集；使用外部 RCT 的已知组效度检验可区分 CBT 与非 CBT；Qwen3.5‑9B 的 Evidence‑First‑Category 配置达 BA 0.72（95% CI 0.70‑0.74）和 MCC 0.45（0.41‑0.49），校准偏差接近零；与人工 κ 0.71 的差距仍存在。

**⚠️ 局限性**

局限包括：仅使用文本缺失语音线索与情绪信息；去标识可能丢失上下文细节；专家评分样本量有限；仅验证 CBT 方案，难以推广到其他疗法；外部 RCT 未标注导致准确性评估受限；依赖开源模型，商用模型表现可能不同。

---

## 287. Koala: Non-disruptive Reconfiguration for Long-running Stateful Dataflows

**arXiv ID:** 2610.28939 | [PDF](https://arxiv.org/pdf/2610.28939v1)

**作者:** Lei Huang `[一作]` (Boston University), Vasiliki Kalavri `[通讯]` (Boston University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种名为 Koala 的惰性状态迁移协议，支持长时间运行的数据流在不中断处理的情况下完成动态重构。

**💡 创新点**

创新点在于将键空间分区与状态位置解耦，采用按需状态迁移、轻量级所有权转移和单跳 KLT 查询，实现在不阻塞的情况下完成动态重构。

**🔧 技术方法**

使用 Go 语言实现自研分布式数据流框架，结合 PebbleDB、FIFO TCP 通道、peer 连接、事件时间水印、KLT/RT 表、桶级状态分区等技术。

**📊 数据集**

在真实流数据（Twitch、Taxi、Azure、Borg）以及 Nexmark 查询（Q3–Q8）和人工合成的大规模状态负载上进行实验。

**📈 对比分析**

对比 S&R、DRRS 和外部状态基线，Koala 实现无停机重构，回压恢复速度比 DRRS 快 23 倍、迁移量少 8.7 倍，吞吐量与外部状态相当，延迟保持稳定。

**⚠️ 局限性**

局限性包括在极端无局部性或高随机访问的工作负载下迁移量上升、对失效的依赖需检查点恢复、以及需要手动调节 N 以平衡迁移速度与扰动。

---

## 288. Does Every Instance Count Equally? Order Effects in LLM Multi-Instance Processing

**arXiv ID:** 2610.29057 | [PDF](https://arxiv.org/pdf/2610.29057v1)

**作者:** Jingxuan Chen `[一作]` (Cardiff University), Jose Camacho-Collados `[通讯]` (Cardiff University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计了一个基于多数标签的受控探测器，用来评估大型语言模型在多实例处理（MIP）中的排列不变性。

**💡 创新点**

创新点在于首次量化并系统分析了模型在不同实例顺序下的聚合错误，揭示了“中间失效”与前置/后置偏差对预测准确率的巨大影响。

**🔧 技术方法**

采用了提示工程、推理轨迹抽取、梯度归因（Input×Gradient）等技术，对15种LLM在五个二分类任务中的表现进行实验和解释。

**📊 数据集**

使用了五个公开数据集（Math、NER、Occurrence、Sentiment、WSD），在每个任务中筛选出单实例可解的二分类样本。

**📈 对比分析**

通过比较不同顺序模式（随机、majority-first、majority-middle、majority-last等）下的准确率，发现模型准确率差距可达约90个百分点；多数模型在majority-first时表现最佳，而majority-middle时表现最差。

**⚠️ 局限性**

局限性包括仅在英文数据上评估、未覆盖所有MIP可靠性维度、缺乏系统的纠正方法、推理轨迹与归因分析仅为间接证据，且部分模型不提供完整推理日志。

---

## 289. SMR: Learning streaming motion retargeting for humanoids from human semantic

**arXiv ID:** 2610.29169 | [PDF](https://arxiv.org/pdf/2610.29169v1)

**作者:** Chun Gu `[一作]` (Fudan University), Li Zhang `[通讯]` (Fudan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一种基于语义条件物理专家的实时人类运动转移框架 SMR，能从人类运动生成可执行的机器人运动参考。

**💡 创新点**

① 通过机器人感知的任务空间语义命令而非传统几何匹配来训练物理专家；② 将专家滚动结果蒸馏成严格因果的 Transformer，实现流式实时参考生成；③ 在冻结全身控制器下进行闭环评估。

**🔧 技术方法**

语义化机器人感知、强化学习专家（PPO）、物理仿真跟踪、Transformer 归因因果网络、蒸馏训练、任务空间目标对齐等技术。

**📊 数据集**

AMASS、LAFAN1、BONES‑SEED、100STYLE 等公开全身运动数据集。

**📈 对比分析**

与基于 GMR 的几何优化、NMR 的双向学习以及物理校正后的基线在 Unitree G1 上进行 400 动作的冻结全身控制器闭环评估；SMR 在 97% 成功率、MPJPE 0.039 m、E‑MPJPE 0.080 m 上优于其它方法。

**⚠️ 局限性**

仅在 Unitree G1、单一仿真/控制器环境下验证；因果蒸馏导致相较于双向模型的性能上限受限；缺乏对不同形态、控制器和真实硬件的泛化评估。

---

## 290. Parallelizable Probabilistic Latent Reasoning for Diverse Reasoning Trajectories

**arXiv ID:** 2610.29385 | [PDF](https://arxiv.org/pdf/2610.29385v1)

**作者:** Seokhyun Lee `[一作]` (Korea University), Changhee Lee `[通讯]` (Korea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出Probabilistic Latent Reasoning Models（PLRM），在连续潜在空间中以变分方式学习推理路径的分布；

**💡 创新点**

创新点在于将链式思考（CoT）从语言空间迁移到潜在空间，并通过联合后验-先验学习实现稠密监督与并行教师强制训练，同时引入策略混合先验以建模多模态推理策略；

**🔧 技术方法**

技术主要包括变分自编码器（VAE）框架、JEPA式预测目标、FiLM调制策略混合先验、SwiGLU+RMSNorm特征转换以及自回归语言模型的轻量级潜在注入；

**📊 数据集**

使用的公开数据集包括GSM8K-Aug、GSM8K-Aug-NL、CommonsenseQA、SVAMP、GSM-Hard、MultiArith，以及自建的HiddenBranch（由ProofWriter衍生的多轨迹推理数据集）；

**📈 对比分析**

与CoT-SFT、COCONUT、CODI、SIM-CoT、ReGuLaR、CoLaR等基线比较，PLRM在单轨迹任务上与确定性潜在方法相当，在多轨迹任务（HiddenBranch）上在Hit Rate和Avg. TV Score上显著优于所有基线，且训练效率高、并行性好；

**⚠️ 局限性**

局限性包括仍需要先验与后验的双网络，且在极端多模态场景下混合先验可能需要更多组件，模型在极长推理步骤时潜在空间维度压缩仍可能导致信息丢失。

---

## 291. Where Does Timbre Hide? Localizing Speaker Information in Speech Signals

**arXiv ID:** 2610.29150 | [PDF](https://arxiv.org/pdf/2610.29150v1)

**作者:** Yiquan Zhou `[一作]` (Xi'an Jiaotong University), Shihao Li `[通讯]` (Union Wheatland Culture and Media Ltd)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

探究语音信号与表示中声纹（音色）信息的位置与泄露

**💡 创新点**

系统性对频谱、相位、源/滤波、SSL层、Token等多维度进行定位，揭示低频谱幅值是主要声纹载体，且高频信息对感知相似度贡献有限

**🔧 技术方法**

频率滤波、STFT相位/幅值随机化、WORLD源滤波拆分、SSL层特征提取、PCA、EnCodec/HuBERT Token 直方图、Speaker Verification（WavLM/ECAPA）、听力测试

**📊 数据集**

AISHELL-1/3、VCTK、ESD/CREMA-D、NUS-48E、JVS/JVS-MuSiC、自录 48 kHz 中文–英文语料

**📈 对比分析**

通过 SV 判别率、EER、分类准确率、cosine 区间、听众相似度评分等指标，发现低/中频(1–8 kHz)在识别与感知上占主导，深层 SSL 与部分 Token 仍泄露声纹信息

**⚠️ 局限性**

大部分探测为闭集任务、WavLM 仅支持 16 kHz、听力测试仅使用频带滤波刺激、不同语言/风格覆盖有限

---

## 292. DiffResearch: Accelerating Deep Research with Diffusion Language Models

**arXiv ID:** 2610.28909 | [PDF](https://arxiv.org/pdf/2610.28909v1)

**作者:** Alan-Barsag Gazzaev `[一作]` (ITMO University), Sergey Muravyov `[通讯]` (ITMO University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DiffResearch 框架，将扩散语言模型（dLLM）作为所有多代理系统的核心，并在相同的管线中与自回归（AR）模型进行对比；

**💡 创新点**

创新点在于：①实现所有阶段均可使用 dLLM 的统一架构；②通过可切换骨干网络实现对比实验，证明扩散解码在规划和写作阶段可获得约 10‑12 倍速度提升；③引入 dFast 模式，利用中间去噪状态提前获取子查询，单阶段加速约 2 倍；④在公开基准上展示基于强大扩散模型的系统可与商业系统竞争；

**🔧 技术方法**

使用扩散语言模型（如 LLaDA、Dream、Mercury）、Hugging Face 运行时、Serper/Semantic Scholar API 进行检索、RACE（Gemini 2.5 Pro 评估器）以及自定义的多代理脚本；

**📊 数据集**

采用 Deep Research Bench（100 个 PhD 级任务，涵盖中英两种语言和 22 个学科）进行系统级评估；

**📈 对比分析**

方法：将 dLLM 与相同规模的 AR 模型在同一管线中进行跑分，比较端到端延迟、各阶段时延、报告质量（RACE 分数）和排行榜成绩；结果显示：dLLM 在规划和写作阶段约 10‑12 倍加速，整体端到端加速约 6 倍；dFast 在单独拆解阶段加速约 2 倍，但未进一步降低整体延迟；报告质量保持或略优于 AR；

**⚠️ 局限性**

局限性包括：实验仅在 H200 服务器上进行，扩散推理可能在其他硬件上成为瓶颈；检索仅使用 Semantic Scholar，忽略其他通道；dFast 仅适用于子查询拆解，无法提升最终答案质量；仅对单一扩散/AR 对比，未覆盖更广模型族与规模；公共排行榜结果不直接归因于扩散解码；

---

## 293. How to Walk a Dog in Parallel: On Parallel Computation of the Discrete Frechet Distance

**arXiv ID:** 2610.29296 | [PDF](https://arxiv.org/pdf/2610.29296v1)

**作者:** Leonid Barenboim `[一作]` (Open University of Israel), Orr Fischer `[通讯]` (Bar-Ilan University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了离散Fréchet距离的并行算法，覆盖一般多边形曲线与c‑packed曲线，并在PRAM与MPC模型下给出实现；

**💡 创新点**

创新点在于首次给出离散Fréchet距离的并行算法，采用矩阵条带合并与层级简化技术，实现近线性工作量；

**🔧 技术方法**

主要技术包括矩阵条带分割与合并、并行可达性信息压缩、μ‑简化与α‑放松简化、层级递归分块、模糊决策器；

**📊 数据集**

该工作为理论研究，未使用具体实验数据集，主要通过理论分析与证明来展示算法性能；

**📈 对比分析**

与传统O(n²)的动态规划相比，PRAM算法实现O(log n/σ)轮次、O(n²+σ)工作量；MPC版在O(log n/ρ²)轮次、O(n²+ρ)空间下完成；c‑packed曲线可在O(log n+1/σ³)轮次、O(n^{1+σ})工作量下近线性完成；

**⚠️ 局限性**

局限性包括：对一般曲线仍需至少O(log n)轮次；c‑packed算法需要O(1/ρ³)轮次；无法突破SETH所给的强子二次下界；开放问题如单侧c‑packed、O(1) MPC轮次等仍待解决。

---

## 294. Combating Catastrophic Forgetting in Continual Domain Adaptation via Knowledge Recasting

**arXiv ID:** 2610.28892 | [PDF](https://arxiv.org/pdf/2610.28892v1)

**作者:** Zijie Liu `[一作]` (University of North Carolina at Chapel Hill), Tianlong Chen `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种后训练框架Skill Schema Transport（SST），通过将新领域示例映射到已有的程序化技能模式来缓解灾难性遗忘。

**💡 创新点**

创新点在于不通过束缚模型漂移来保持旧知识，而是利用技能抽象与重用，将新知识以已有程序化表征表达，从而实现更高效的知识组织。

**🔧 技术方法**

主要技术包括技能模式抽取（Γ）、方案记忆（ℬ）、模式匹配与重用、以及基于抽取技能的条件性监督式微调。

**📊 数据集**

使用了两组连续领域数据集：数学推理流（代数→计数与概率→几何→中级代数）和代理任务流（AlfWorld→WebShop→搜索增强QA）。

**📈 对比分析**

与单纯序列微调、回放与正则化基线相比，SST在两条流上平均提升2.53–5.75分，显著降低最大遗忘幅度，并在OOD任务中恢复性能。

**⚠️ 局限性**

局限性包括依赖于技能抽取的质量、仅在较小规模模型（3B）与两条特定领域流上验证、以及对更大规模多任务或长序列更新的泛化仍待进一步探索。

---

## 295. What Git Does Not Record: Provenance Legibility in Two AI-Assisted Research Repositories

**arXiv ID:** 2610.28655 | [PDF](https://arxiv.org/pdf/2610.28655v1)

**作者:** Koutian Wu `[一作]` (University of Texas at Austin), Hejia Geng `[通讯]` (University of Oxford)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对两份单一研究者的代码仓库进行审计，评估它们在记录 AI 参与过程中的可读性与完整性。

**💡 创新点**

创新点在于：①提出了对 Git 仓库中 AI 轨迹（prescriptive 规则文件与 narrative 笔记）可读性度量；②发现提交日志中的 AI 附加键（trailer）既误导又不稳定；③给出了针对 AI 研究可追溯性的系统性需求。

**🔧 技术方法**

使用的技术包括：Git 历史查询、文件变更计数、脚本化审计、blame 分析、自动化提取 commit trailer、统计分析与可视化。

**📊 数据集**

数据集为两份单一作者在环境科学领域的公开 Git 仓库，分别包含 466 次提交与 103 次提交。

**📈 对比分析**

比较方法：对同一作者的两个仓库分别衡量 AI 轨迹可见比例、提交日志 AI 附加率、以及轨迹中条目的保存情况。结果显示：prescriptive 轨迹可见率 85.6% 但记录过程有限；narrative 轨迹可见率仅 21.8%；提交日志 AI 附加率在项目 A 为 14.2% 但不稳定，项目 B 仅 1.0%。

**⚠️ 局限性**

局限性包括：仅审计两份仓库，无法排除工具/项目差异导致的偏差；依赖单一审计者对规则进行编码，缺乏交叉验证；对文件路径变更的查询方式会影响可见度；提交日志中 AI 附加键的敏感度低且易受工具设置变化影响。

---

## 296. Complexity-Aware Surgical Video Reasoning Segmentation with Digital Twin and Context Memory

**arXiv ID:** 2610.28776 | [PDF](https://arxiv.org/pdf/2610.28776v1)

**作者:** Yiqing Shen `[一作]` (Johns Hopkins University), Mathias Unberath `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SurgReasonSeg，一种在线数字孪生（DT）驱动的手术视频推理分割方法；

**💡 创新点**

创新点在于（1）将手术专用基础模型构建DT，桥接视觉与LLM推理；（2）双LoRA适配器与RL路由器实现根据查询复杂度自适应计算；（3）自适应滑动窗口和手术上下文记忆实现时序推理；（4）创建EndoVis-ReasonSeg推理分割基准；

**🔧 技术方法**

使用SurgSAM2、DepthAnything2、RASO等手术基础模型构建DT；Qwen3-4B LLM + LoRA + RL（GRPO）训练；自适应滑动窗口、路由器、记忆检索；

**📊 数据集**

EndoVis-ReasonSeg（1,460训练/961测试样本，涵盖语义、空间、时序三类推理），同时在Ref-EndoVis17/18评测；

**📈 对比分析**

与多种基线（LISA、VISA、CoReS、JiT、RSVIS、ReSurgSAM2）对比，SurgReasonSeg在三类推理任务的J&F分别提升约0.18/0.24/0.30，Ref-EndoVis基准提升0.02-0.03；平均速度18.8 FPS，显著高于多数基线；

**⚠️ 局限性**

限制：性能受SurgSAM2候选召回率约90%上限；窗口仅覆盖数帧，难以回答长时程问题；记忆仅在EndoVis验证，跨机构泛化待测；

---

## 297. Conditional Flow Models Convert Epistemic Uncertainty into Apparent Aleatoric Variance

**arXiv ID:** 2610.29172 | [PDF](https://arxiv.org/pdf/2610.29172v1)

**作者:** Navodita Sharma `[一作]` (Massachusetts Institute of Technology), Pulkit Agrawal `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究条件流模型的方差膨胀现象及其对预测置信集的影响

**💡 创新点**

揭示向量场学习的光滑偏差将模型不确定性转化为伪随机方差，并证明在欠拟合时可产生近似校准的置信集

**🔧 技术方法**

条件流匹配、核岭回归、正则化神经网络与理论等价核分析

**📊 数据集**

Ogbench强化学习环境、scPerturb细胞扰动数据集（Norman、PBMC、SciPlex3）和PushT机器人行为克隆数据集

**📈 对比分析**

与分割合规预测、合规化分位数回归、深度证据回归等基线对比，CFM基置信集在欠拟合阶段保持≈ nominal 置信度且比对齐合规方法更紧凑

**⚠️ 局限性**

理论仅适用于理想化正则化模型，未给出正式预测覆盖保证，且实际神经网络训练的方差膨胀机制仍未完全解析

---

## 298. Evolution of article processing charges in medicine: disciplinary position, access gaps and future prospects

**arXiv ID:** 2610.28643 | [PDF](https://arxiv.org/pdf/2610.28643v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 299. Plan First, Act Later: Streaming Reasoning for Executable Memory in Robot Control

**arXiv ID:** 2610.29329 | [PDF](https://arxiv.org/pdf/2610.29329v1)

**作者:** Qinghang Liu `[一作]` (Peking University), Chang Liu `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Plan First, Act Later（PFA）层次化记忆框架，利用短期视觉缓冲、语义事件记忆和可执行子任务列表，在流式推理中主动写入未来子任务并在执行时检索；同时设计两阶段训练（稠密更新监督+自标注验证）来学习何时更新记忆。

**💡 创新点**

①先写未来子任务到可执行记忆再执行，减少执行时搜索成本；②采用可流式的三层记忆结构，将视觉历史压缩为语义事件和子任务列表；③通过自标注+验证的两阶段训练，弥补稀疏监督导致的时机不准问题。

**🔧 技术方法**

使用视觉-语言动作（VLA）模型，分为高层VLM与低层VLA；三层记忆架构（视觉缓冲、语义事件、可执行子任务）；流式推理更新机制；自标注与验证（task‑grounded verifier）以及稠密化训练标签。

**📊 数据集**

RoboMME长时记忆基准（模拟数据），以及基于Franka Research 3机器人四个真实任务（PickXtimes、VideoUnmask、VideoPlaceOrder、PatternLock）的真实数据集。

**📈 对比分析**

与五个基线（π_0.5、FrameSamp‑Modul、MemER、EventVLA、GPT‑6‑astra+VLA）在RoboMME上比较，PFA在总体成功率上达56.05%，比最强基线FrameSamp‑Modul的44.51%提升11.54个百分点；在真实世界四任务上总体成功率60.83%，显著高于MemER（46.67%）和FrameSamp‑Modul（28.33%）。

**⚠️ 局限性**

可执行记忆仅为顺序子任务列表，无法表示条件分支或分支化执行流程；未来工作需扩展为可分支化的可执行程序以支持更复杂的决策。

---

## 300. Seeing through Denoising: Frequency-guided One-step Denoising for Infrared Small Target Detection

**arXiv ID:** 2610.29361 | [PDF](https://arxiv.org/pdf/2610.29361v1)

**作者:** Xuelin Qian `[一作]`, Junwei Han `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 DiffIRSTD 框架，通过扩散模型实现一键去噪的背景重建，从而生成无目标背景图，再结合原图进行小目标检测。

**💡 创新点**

创新点在于把小目标视为噪声进行去噪，并采用单步扩散去噪和频域引导感知损失来保持高频细节，同时实现 plug‑and‑play 兼容现有检测器。

**🔧 技术方法**

主要技术包括扩散模型（DDPM）、一阶去噪策略、频域感知损失、U‑Net 目标检测网络，以及基于形态学的背景伪标签获取。

**📊 数据集**

在 IRSTD-1K、NUAA-SIRST 和 NUDT-SIRST 三个公开红外小目标检测基准上进行实验。

**📈 对比分析**

与现有传统、Transformer、Mamba、SAM 以及扩散相关方法相比，DiffIRSTD 在三大数据集的 IoU、召回率、误报率均取得或接近最优的表现，显著提升检测精度。

**⚠️ 局限性**

主要限制是重建网络参数较多，单步去噪仍可能产生细节模糊，对边缘小目标的恢复有限，且在边缘设备上部署仍需进一步轻量化。

---

## 301. The Predictive--Optimization Gap in GuanDan Self-Play

**arXiv ID:** 2610.29411 | [PDF](https://arxiv.org/pdf/2610.29411v1)

**作者:** Qicheng Zou `[一作]` (University of California San Diego), Xiaokun Wang `[通讯]` (Chabiyue (Shanghai) Information Technology Co., Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在多轮不完全信息卡牌游戏 GuanDan 上，先构建冻结的公共状态赢率预测器以验证状态-增益交互的预测有效性；随后将该预测器的进度增量作为回合层监督信号，插入到基于 Monte Carlo 回归的自玩训练中，形成基线和进度增强两种奖励策略。

**💡 创新点**

首次系统性分离并验证“预测有效性”与“优化可靠性”这两种不同的主张；通过两阶段诊断（预测验证 + 方法层复制）揭示了“预测-优化鸿沟”，即预测信号虽能提升预测性能，却不能在多次训练中保证可重复的策略提升。

**🔧 技术方法**

技术包括：① 27 维公共状态特征 + 逻辑回归+ Platt 校准的赢率预测器；② 4.96M 参数的 DMC (Deep Monte Carlo) 执行动作回归网络；③ 基于回合边界的进度增量 ΔΦ 并以 β 调节的进度增强目标；④ 8 轮循环的在线自玩采样-注释-训练流程；⑤ 采用 500–1000 对团队交换的锁定评估与 10 个训练种子复制实验。

**📊 数据集**

使用 2,000 条完整匹配（约 24,636 轮，2,398,084 执行动作）的 GuanDan 训练集；预测器在训练/验证/测试拆分后冻结；随后所有自玩实验均基于同一训练集生成的比赛。

**📈 对比分析**

方法对比：基线 (仅回合收益+最终比赛结果) 与 Progress‑10 (加入 10% 进度增量)。单个选定的 Progress‑10 检查点在 1,000 对锁定评估中相对基线取得 MeanDelta 0.143（单局胜率约 57.2%），但在 10 个独立种子中平均效果为 -0.030（95% CI [-0.084, 0.025]），即无显著一致提升；基线在所有种子上稳定表现。

**⚠️ 局限性**

局限性包括：① 进度增量仅为预测性而非因果信用，无法提供单动作级别奖励；② 采用的回合广播监督不具备 PBRS（潜在奖励塑形）的最优性保证；③ 只评估了 DMC 执行动作回归学习器，未检验梯度或策略梯度等其他优化方式；④ 只使用了 GuanDan 一个游戏，结果可能不具泛化性；⑤ 训练种子数量有限（10 对），可能不足以完整刻画随机性分布；⑥ 冻结预测器在自玩过程中可能出现分布漂移。

---

## 302. Approximate Solomonoff Induction: On the Robustness of Learned Universal Predictors

**arXiv ID:** 2610.29289 | [PDF](https://arxiv.org/pdf/2610.29289v1)

**作者:** Qian Li `[一作]` (Shenzhen Research Institute of Big Data), Shang-Hua Teng `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在逼近真实贝叶斯预测器的条件下，序列预测的鲁棒性，给出了任务级（最坏情况）和平均情况（以及分布偏移）下的误差上界与下界，并把这些理论结果推广到有限样本学习场景。

**💡 创新点**

①首次给出逼近误差对单一环境预测误差的紧致最坏情况与最优下界；②提出平均情况下的精确误差分解，证明逼近误差对平均性能的影响远小于最坏情况；③构造分布偏移定理，说明在训练先验与测试先验之间的占优关系下误差可控；④在有限样本下证明即使 KL 散度可能发散，平方预测误差仍可用期望保证。

**🔧 技术方法**

贝叶斯混合预测理论、Kolmogorov 复杂度与 Solomonoff 归纳、KL 散度与 Hellinger 距离分析、信息理论不等式、概率论极限与尾部概率上界、抽样与泛化分析。

**📊 数据集**

无具体数据集；论文完全基于理论推导和数学证明，不依赖于实验数据。

**📈 对比分析**

没有实验比较；所有结论通过严谨的数学证明给出，性能以理论误差上界/下界的形式呈现，例如最坏情况误差为 Θ(η/w_μ)，平均情况误差为 O(η)，分布偏移误差为 O(log C + Cη)。

**⚠️ 局限性**

主要限制：①理论结果以理想化的无限容量与完美优化为前提，实际模型可能无法完全逼近贝叶斯混合；②最坏情况误差在稀有环境中可能极大，说明单任务性能仍需进一步保障；③有限样本分析只给出期望层面的平方误差，KL 散度仍可能无穷大；④对非确定性任务分布、不同先验机器或长序列测试的推广尚未完成。

---

## 303. RubatoCodec: Budgeted Adaptive Speech Tokenization with Streaming Decoding

**arXiv ID:** 2610.29434 | [PDF](https://arxiv.org/pdf/2610.29434v1)

**作者:** Chihiro Arata `[一作]` (Third Intelligence, Inc.), Kan Hatakeyama-Sato `[通讯]` (Third Intelligence, Inc.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文设计了一种基于X-Codec-2的可变帧率语音编解码器RubatoCodec，利用自适应发射信号在固定事件预算下对语音进行可变时序分段，并在编码时生成带时长的离散码，解码时采用因果Transformer实现流式重建和生成；

**💡 创新点**

其创新点在于将精确的句子级事件预算与自适应分段相结合，使用累计分配与聚合器在无需学习边界预测器的情况下完成事件定位，并通过语义监督与混合残差向量量化实现高效的多层码本表示；

**🔧 技术方法**

技术上主要使用了Adaptive Firing、累计预算分配、聚合器、混合残差向量量化、语义监督（w2v-BERT 2.0）、因果Transformer、Vocos式谱解码以及因果时序上采样；

**📊 数据集**

训练数据为约375k小时多语言语料，覆盖英语、日语、中文、德语、法语、韩语等，评测数据包含LibriSpeech、VCTK、JVS、JVNV、seed-tts-eval等五个数据集；

**📈 对比分析**

通过与固定帧率编码器及其他VFR编解码器（如TFC、FlexiCodec、CodecSlime等）的对比实验，RubatoCodec在低事件率（6.25–12.5Hz）下实现了更高的重建保真度（PESQ、STOI）和更低的可懂度误差（dWER/dCER），在相同比特率下在英语和日语上表现出最佳可懂度，并在TTS任务中获得更低的WER和更高的UTMOS；

**⚠️ 局限性**

局限性包括：在极低事件率下仍存在保真度差距，长事件多用于静音导致停顿不自然；编码时需冻结语义特征，预算固定难以对不同复杂度句子动态调整；评估仅基于单一模型与有限数据集，缺乏对更大规模、多样化语料和人类听感的验证；

---

## 304. Constrained Tagging for Span Labeling

**arXiv ID:** 2610.28922 | [PDF](https://arxiv.org/pdf/2610.28922v1)

**作者:** Radek Štulc `[一作]` (Czech Technical University in Prague), Jan Drchal `[通讯]` (Czech Technical University in Prague)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练的约束解码方法——约束标注，利用生成式大语言模型在保持输入原样的同时为每个跨度插入标签；

**💡 创新点**

创新点在于：1) 构建字节级词典 Trie 与有限状态机，实时限制输出只能复制原文或合法标签；2) 两阶段解码让推理不受约束；3) 对跨度边界不受 tokenization 限制，可在字符级任意位置标注；

**🔧 技术方法**

核心技术为字节级 Trie 搜索、有限状态机（FSM）生成器、logits 处理器、两阶段推理-aware 解码；

**📊 数据集**

评估六大数据集：CoNLL-2003、UniversalNER（NER）；Toxic Spans、LegalQAEval（QA）；ESA-MT（机器翻译错误检测）；MultiGEC（语法错误修正）；

**📈 对比分析**

与无约束标注、基于上下文的 JSON 列表提示、XGrammar 语法约束对比；在五个数据集上超过无约束标注，除 Toxic Spans 外基本与上下文提示持平或略优，硬 F1 上升最多 45 分，约束成功率 99.978%；

**⚠️ 局限性**

限制包括：仅在推理闭合且在 token 限额内才保证原文复制；无法选择更佳跨度，仍需改进提示或轻量 fine‑tune；需对 logits 进行访问，无法直接通过 chat API；生成全输入导致在长文本中成本上升。

---

## 305. Speedrunning Relational Foundation Model Pretraining

**arXiv ID:** 2610.28855 | [PDF](https://arxiv.org/pdf/2610.28855v1)

**作者:** Gabriel Wedson Mendonça de Souza Veloso `[一作]` (Kunumi Institute), Francisco Galuppo Azevedo `[通讯]` (Kunumi Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立了 modded-nanoRT 开源速度跑 leaderboard，记录了在单 GPU 上对 Relational Transformer 的预训练并在留出的数据库上零-shot 评估的最快时间。

**💡 创新点**

主要创新在于将 typed trilinear attention、Muon 以及 SIGReg 结合，并通过对训练数据混合进行温度采样和数据库留出策略，使 benchmark 能在单机 GPU 上实现可重复、可比的时间到质量目标。

**🔧 技术方法**

使用了 Typed trilinear attention、Muon 半正交矩阵优化、SIGReg 正交正则化、Muon's Newton-Schulz 迭代、以及 Relational Transformer 体系结构与 AdamW/SGD 优化器。

**📊 数据集**

采用 RelBench v1 的七个关系数据库（移除 rel-Event），以 rel-Avito 为留出数据库，其余五个数据库用于混合预训练。

**📈 对比分析**

通过将记录时间与参考实现（RT 12/256/8-heads）进行比较，最终记录 305.63 秒，约比基线快 8.11 倍；同时在 15 个非开发数据库上获得平均 2.14 AUROC 的提升。

**⚠️ 局限性**

限制在于验证和评估路径占比高（约 60%），训练时间窗口被评估网格宽度限制，且在不同数据库上表现差异大（如 rel-Amazon 变异性高）。

---

## 306. Zero Knowledge for Free up to Linear Query Bounds: Quasilinear-Size Constant-Query Zero-Knowledge PCPs

**arXiv ID:** 2610.29066 | [PDF](https://arxiv.org/pdf/2610.29066v1)

**作者:** Inhyuk Park `[一作]` (Seoul National University), Chenglin Fan `[通讯]` (Seoul National University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文构造了针对任意时间可构造多项式 T 的多项式时间可构造的零知识 PCP，具有常数查询、完美完备性、常数误差，并在任意 NTIME(T) 语言上实现完美零知识。

**💡 创新点**

创新点在于通过引入递归同进制配置、定位器技术、Beneš 网络算术化以及新的外壳 PCP，首次实现了对所有 NTIME(T) 语言的常数查询零知识 PCP，并在保持完美零知识的同时将查询次数降低到绝对常数。

**🔧 技术方法**

核心技术包括 Reed–Solomon 码与 RS PCPP、零知识模拟器、定位器（locator lemma）、Beneš 网络、算术化（arithmetic CSP）以及 Dinur 的 gap‑amplification 与 Tseitin 前端等。

**📊 数据集**

本工作为理论研究，不涉及任何具体数据集；所有结果均在抽象模型与符号计算上证明。

**📈 对比分析**

与以往的 PCP 结果相比，本文实现了常数查询（Qfin=18⌈1/α_D⌉），完美完备性，误差 ≤27/64，并在多项式时间内生成证明，证明长度为 O((T+b)(log(T+b))^C)。

**⚠️ 局限性**

主要局限是常数项极大（E_L、E_C 等），导致实现成本高；仅适用于可构造多项式时间语言；实现需要在大域上进行算术化；缺乏可落地的实例与实验验证。

---

## 307. Dual-Channel Contrastive Encoding of Sperm Whale Coda

**arXiv ID:** 2610.29101 | [PDF](https://arxiv.org/pdf/2610.29101v1)

**作者:** João Victor Quintanilha `[一作]` (Tufts University), Shuo Zhang `[通讯]` (Tufts University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

针对鲸类的咔哒声（coda），设计并训练了Dual‑Channel Contrastive Encoder (DCCE)，分别用GRU编码节奏信息（ICI）和CNN+MLP编码频谱信息（mel），随后通过融合层得到固定维度的表示，并用线性探针评估其在社交单元识别、coda类型分类与个体识别三项任务上的表现。

**💡 创新点**

创新点在于将节奏与频谱两个信号通道分别建模并显式融合，而不是将所有信息压缩为单一输入；结合对节奏通道的类型监督和频谱通道的个体监督，探索各通道在不同任务中的贡献，展示了任务依赖的特征组合。

**🔧 技术方法**

技术细节包括：GRU+层归一化对ICI序列进行编码；三层卷积+池化+MLP对64×256 log‑mel spectrogram进行编码；Fusion MLP将两通道向量拼接并投射到64维；NT‑Xent对比损失配合可选的交叉熵监督；训练使用随机正样本配对、频谱/时间遮蔽等数据增强；冻结表示后使用平衡逻辑回归做线性探针。

**📊 数据集**

数据集来源于Dominica Sperm Whale Project（DSWP）：1383条清洁coda，涵盖3个社会单元、22种coda类型、11个个体，时间跨度2005‑2010年，采用三种随机划分及按录音日期分组的评估协议。

**📈 对比分析**

与文献中的WhAM、AVES、BirdNET三种基线在相同的线性探针协议下比较。DCCE在随机拆分下的宏F1分别为0.901（单元）、0.865（类型）、0.597（个体），在类型任务上接近或优于基线，且raw‑ICI在类型任务上表现最优；在录音日期分组与年度留置评估中性能显著下降，说明泛化受限。

**⚠️ 局限性**

局限性包括：仅使用单一Dominica族群，个体覆盖率有限；缺少语音学（vowel）注释，频谱通道未显式学习语音学特征；评估仅为封闭集个体识别，未考察开放集或未见个体；数据量有限且录音日期与内容高度相关，导致模型可能依赖录音环境；实验设计为探索性，未进行完整的超参数调优。

---

## 308. DISRL: Dual-Scale Incremental Semi-Supervised Reinforcement Learning for Dynamic Community Tracking

**arXiv ID:** 2610.29315 | [PDF](https://arxiv.org/pdf/2610.29315v1)

**作者:** Tianming Zhang `[一作]` (Zhejiang University of Technology), Jing Fan `[通讯]` (Zhejiang University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种双尺度增量半监督强化学习框架DISRL，用于在动态图中跟踪指定种子节点的社区演化。

**💡 创新点**

创新点在于（1）自适应生成种子特定监督以缓解稀疏标签误差；（2）双尺度策略更新，内部窗口快速适配外部窗口宏观校准，抑制长期错误积累；（3）构建覆盖多种演化场景的动态图基准。

**🔧 技术方法**

采用强化学习MDP、图指针网络iGPN、基于最短路径的图核相似度检索、动态窗口梯度衰减等技术。

**📊 数据集**

实验使用真实的Football动态网络以及基于SNAP的Amazon、DBLP、YouTube、Twitter四个网络在扩张/收缩与合并/分裂两类演化场景下的数据。

**📈 对比分析**

相较于静态与动态基线，DISRL在所有设置下均取得最高F1与Jaccard分数，平均提升4–6个百分点，尤其在Football数据上提升2.89个百分点。

**⚠️ 局限性**

局限性包括对超参数（如K、窗口大小w、iGPN层数）的敏感性；依赖稀疏标签的质量；仅针对种子中心社区，不适用于全局聚类。

---

## 309. Relational Geometry Forcing for World Action Models

**arXiv ID:** 2610.29282 | [PDF](https://arxiv.org/pdf/2610.29282v1)

**作者:** Yuan Xu `[一作]` (CASIA), Liang Wang `[通讯]` (CASIA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文研究了如何在不显式几何重建的情况下，通过关系对齐将几何先验迁移到World Action Models（WAM）。

**💡 创新点**

其创新点是提出三种关系对齐（帧内、跨视角、动作时间）来对齐WAM特征与冻结的3D基础模型的关系，而不是传统的点对点对齐。

**🔧 技术方法**

技术上采用冻结的Visual Geometry Grounded Transformer（VGGT）作为教师，结合可训练投影头和Huber损失对关系进行对齐。

**📊 数据集**

实验使用LIBERO、RoboTwin 2.0仿真数据以及Franka Research 3机器人配备的ZED 2i摄像头的真实数据。

**📈 对比分析**

与多种VLA、WAM和几何感知方法比较，本文在LIBERO和RoboTwin上平均成功率均提升约1%至1.5%，在真实机器人上提升约10%以上。

**⚠️ 局限性**

局限性在于仅对齐余弦相似关系，未利用尺度、接触结构等更丰富的几何信息，并且只在训练时使用教师，推理阶段不再包含几何输入。

---

## 310. Combining Long-Term Memory and LLMs Comes at a High Cost: From Prompt Injection to Persistent Compromise

**arXiv ID:** 2610.28810 | [PDF](https://arxiv.org/pdf/2610.28810v1)

**作者:** Pedro H. Barcha Correia `[一作]` (Universidade de São Paulo), Marcos A. Simplicio `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了大型语言模型（LLM）服务中的“记忆注入”攻击，设计并实施了多种基于持久记忆（个人记忆、共享项目、技能、MCP连接等）的攻击场景，对ChatGPT、Claude和Gemini等主流LLM‑as‑a‑service平台进行实验，展示了攻击可实现NIST定义的四大威胁目标（完整性、可用性、隐私、误用）。

**💡 创新点**

创新点包括：①提出了统一的记忆注入威胁模型，覆盖所有NIST攻击目标；②在多服务、多记忆机制上开展首次系统实验，揭示共享记忆可跨用户扩散的风险；③通过公开GitHub数据集和脚本，为后续研究提供可复现资源；④结合OWASP安全原则构建一套完整的对策清单，评估当前主流平台的实现情况。

**🔧 技术方法**

技术手段：利用Prompt Injection与Jailbreak相结合，注入持久指令；利用各平台原生持久记忆功能（ChatGPT Memories、Claude Skills/Projects、Gemini Personal Context）；通过MCP（Model Context Protocol）连接外部数据源实现间接注入；实施CSRF等攻击尝试；使用人类评审对攻击成功与否进行判定；结合安全原则提出过滤、隔离、默认禁用、监控等对策。

**📊 数据集**

数据集：包含所有攻击用的Prompt、注入语句、后续测试对话及模型输出；实验记录与重现步骤保存在 https://github.com/PedroBarcha/Memory-injection；实验数据主要为手工标注的成功/失败标签，未使用公开大型数据集。

**📈 对比分析**

方法对比：作者没有与传统Prompt Injection或模型中毒方法进行数值性能对比，而是通过定性实验展示每种攻击在三大平台上的成功率（可用性、完整性、隐私、误用）以及对共享记忆的影响。实验结果表明，记忆注入可覆盖全部NIST目标，且攻击可在单一账号或整个组织层面复制，说明其威胁规模与传统攻击相比更广泛。

**⚠️ 局限性**

局限性：①实验仅覆盖ChatGPT、Claude、Gemini三款主流服务；②部分攻击（如Gemini的记忆删除、部分MCP功能）受平台接口限制，导致无法完整验证；③LLM的非确定性导致攻击输入不保证可复现；④CSRF攻击在当前版本平台未成功，说明研究未覆盖所有潜在路径；⑤仅关注手动注入，未系统探索自动数据收集导致的持久注入。

---

## 311. Multi-Agent AI Design of a Nanowatt Biomedical Low-Pass Filter from Topology to Layout

**arXiv ID:** 2610.28677 | [PDF](https://arxiv.org/pdf/2610.28677v1)

**作者:** Danial Noori Zadeh `[一作]` (McMaster University), Mohamed B. Elamien `[通讯]` (McMaster University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用多智能体框架（包括研究者、原理图和布局智能体）从规范出发，自动化设计并实现了一个130 nm CMOS的纳瓦特级生物电位低通滤波器，完成了从拓扑合成、尺寸优化到布局、DRC/LVS校验的全流程。

**💡 创新点**

创新点在于：① 将大语言模型智能体与确定性仿真/验证工具深度耦合，形成端到端的自主设计循环；② 在一次迭代中实现了自适应拓扑演化（如当前重用、偏置桥、浮动电容等新技术组合）；③ 通过多级记忆和 A/B 测试机制，使智能体能够在保持性能约束的同时自动改进设计。

**🔧 技术方法**

核心技术包括：大语言模型（Claude Opus 5）驱动的智能体；确定性工具链（NGSPICE、KLayout/KPEX、Nevergrad 优化器、Graph‑based netlist equivalence、符号分析 SLiCAP/SymXplorer、参数化 GDSFactory 生成器）；多级 CoALA 记忆体系（工作记忆、情节记忆、语义记忆、程序记忆）；独立审查与人类审批机制。

**📊 数据集**

使用的数据集主要是设计规范（截止频率、相位延迟、增益、THD、IRN、功耗、拓扑约束）以及从 10 篇参考论文提取的技术假设；在整个迭代中收集了 37 个尺寸点、11 轮仿真结果和 14 次布局版本的 DRC/LVS 统计信息。

**📈 对比分析**

通过与参考设计（两个超源跟随二阶块）和同领域已发表的低通滤波器进行对比，最终实现了 11.91 nW 的核心功耗、29.19 µV_rms 的输入参考噪声（比参考低 41.6%）、0.228 mm² 的面积，并在 250 Hz 处满足 330° 相位延迟、-48 dB 的 1 kHz 增益、-49.73 dB 的 50 Hz THD 等指标，FoM 达到 6.4 fJ，显示出在功耗、噪声和面积方面的优越性。

**⚠️ 局限性**

局限性包括：① 设计仍需在关键节点接受人类审批；② 采用的符号分析和仿真工具未完全覆盖所有 PVT 组合，导致仍有少数工艺角落未满足规范；③ 该框架主要基于现有的技术库，未能主动发现全新的拓扑结构；④ 适用于 130 nm CMOS，迁移到更深工艺或其他工艺节点需要额外验证。

---

## 312. Asking What a Director Sees Can Change What Vision-Language Models Choose

**arXiv ID:** 2610.29318 | [PDF](https://arxiv.org/pdf/2610.29318v1)

**作者:** Jiaxuan Guo `[一作]` (Stanford University), Haizhao Yang `[通讯]` (University of Maryland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了在Director任务中先询问模型“导演能看到哪些方块”后再让其执行“移动最大方块”动作时，询问是否会改变模型的行动，并提出了基线、约束和阅读检查三种方法来消除这一干扰。

**💡 创新点**

创新点在于揭示询问报告会改变后续动作的解读方向，并提供可直接嵌入任何观点采择评测的对照设计、状态约束以及通过推理轨迹判别模型回答目标的工具。

**🔧 技术方法**

主要使用了语言模型推理（包括GPT系列、Qwen3.8-27B、Muse Glimmer-30B）以及对模型推理轨迹的自动编码和人工标注。

**📊 数据集**

使用的数 据集是基于同一渲染环境的160个桌面场景（每个场景有3-4个编号方块，且隐藏块为最大块），以及旋转版和数字重排版的变体。

**📈 对比分析**

通过比较包含与不包含可视性提问的提示，发现可视性提问将隐藏块选择率提高12–88个百分点（GPT六种模型）和17个百分点（Qwen3.8-27B），而在约束提示下隐藏块选择率降至<1%。

**⚠️ 局限性**

局限性包括仅使用单一任务范式、单一语言和场景设计、对模型的依赖性（仅部分模型受影响）、推理轨迹仅适用于支持返回轨迹的模型、以及缺乏对人类行为的验证。

---

## 313. Interaction-Gated Time-Frequency Learning for Parameter-Efficient PPG-Based Heart Rate Estimation

**arXiv ID:** 2610.28693 | [PDF](https://arxiv.org/pdf/2610.28693v1)

**作者:** Anh Khoi Pham `[一作]` (International University Vietnam National University Ho Chi Minh City), Lam Pham `[通讯]` (Austrian Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种基于光体积描记（PPG）的心率估计框架 TF-IG，采用双分支时频学习并通过交互门融合实现参数高效的心率预测。

**💡 创新点**

创新点在于：①双分支时频特征学习（时域脉冲形态 + 频域周期性）①交互门融合模块显式建模时频特征的协同与差异并自适应平衡其贡献；②共享回归头实现参数共享与预测一致性。

**🔧 技术方法**

技术包括：深度可分离卷积、扩张卷积、时域/频域注意力池化、交互门（乘积与差异）、特征门、共享多层感知机回归。

**📊 数据集**

使用公开的 BIDMC PPG 与呼吸数据集（来自 MIMIC-II），进行受试者独立划分。

**📈 对比分析**

与 Welch PSD、FCN、FCN_Residual、FCN_DCT 等基线比较，TF-IG 在 MAE=1.227 BPM、RMSE=2.060 BPM、Pearson r=0.9934、CP_3=92.28% 等指标上均优于对手，参数量仅 84.6K。

**⚠️ 局限性**

局限性包括：依赖 30 秒窗口对实时性有一定影响；模型对极端运动或严重噪声下的鲁棒性尚未系统评估；仅在单一数据集上验证，泛化性待进一步验证。

---

## 314. Benchmarking and Error Analysis of Deep Search and Deep Research LLM Agents

**arXiv ID:** 2610.29346 | [PDF](https://arxiv.org/pdf/2610.29346v1)

**作者:** Zixuan Yu `[一作]` (Sun Yat-sen University), Bo Han `[通讯]` (Hong Kong Baptist University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了针对深度搜索与深度研究代理的过程级错误诊断框架和统一的错误分类体系。

**💡 创新点**

将错误拆分为系统级与任务级两层，并通过完整执行轨迹实现可复现的错误分析，提供细粒度诊断。

**🔧 技术方法**

结合规则解析与LLM‑as‑a‑Judge评估技术，构建双层诊断流程，并在多种Agent框架上进行实验。

**📊 数据集**

使用BrowseComp、BrowseComp‑zh、GAIA‑val‑text‑only、XBench‑DS等深度搜索基准及Deep Research Bench进行评估。

**📈 对比分析**

通过对比不同模型与框架在错误率与最终准确率的相关性，展示错误率低的模型往往准确率更高，并给出错误分布统计。

**⚠️ 局限性**

依赖人工标注和LLM判断，自动化程度有限；实验仅覆盖有限工具与任务场景，缺乏跨域泛化验证。

---

## 315. VIRA-Bench: Diagnosing Pedagogical Contingency Understanding in Online Tutoring Videos

**arXiv ID:** 2610.29428 | [PDF](https://arxiv.org/pdf/2610.29428v1)

**作者:** Weihang Wang `[一作]` (Fudan University), Tun Lu `[通讯]` (Fudan University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了VIRA-Bench，一个面向一对一线上辅导视频的视觉支持–响应–调整链诊断基准，评估模型是否能把教师可见干预与学生反应以及随后的教师调整三者正确关联。

**💡 创新点**

创新之处在于将教学交互拆解为三层（观察-识别、因果-关系、学生状态判定）并用IRF结构与依赖教学理论对基准进行诊断，从而揭示模型在视觉定位、事件顺序和学习者状态推理上的差异。

**🔧 技术方法**

采用多模态大语言模型、自动视频分割与视觉事件检测、GPT-4o/5生成结构化描述与MCQ、专家审核与验证等技术，形成完整的基准构建与评测流程。

**📊 数据集**

使用了来自2018–2021年的超过1000小时专业线上英语辅导录像，提取了1005段15–60秒的教学片段，并生成了1322道经过专家验证的多项选择题。

**📈 对比分析**

在Video+ASR零shot设置下，11个闭源与开源模型被评估，最优模型Gemini 3 Pro达到61.3%准确率，仍与91%的人类参考相差近30个百分点；三层表现呈OEG>ICR>SRM顺序，去掉视频或打乱事件顺序导致各层显著下降。

**⚠️ 局限性**

局限性包括仅针对15–60秒的单一教师-学生对话，可能不适用于多方、跨域或长时段交互；基准构建依赖LLM生成与专家审核，仍可能存在注释误差；评测仅涉及视频+ASR的多项选择任务，未覆盖开放式生成或多轮适配。

---

## 316. Scaling in Games: Continued Pre-Training For Embodied Agents In Diverse Virtual Worlds

**arXiv ID:** 2610.28698 | [PDF](https://arxiv.org/pdf/2610.28698v1)

**作者:** Kuan Zhang `[一作]` (miHoYo), Yiming Li `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对虚拟游戏环境中持续预训练（CPT）的系统规模化研究，探索数据量、数据多样性、模型规模对不同层次泛化（原始、相关、远距）性能的影响。

**💡 创新点**

创新点包括：① 以 120 组配置网格系统化评估三轴（模型、数据量、混合多样性）对 CPT 的影响；② 揭示早期“接口期”与后期“世界期”的两阶段学习动态及其对跨世界共享的贡献；③ 通过经验数据拟合出高精度缩放律；④ 在三层评估框架下（Minecraft、OmniGameArena、GameWorld）全面验证泛化效果。

**🔧 技术方法**

技术手段：使用 Qwen3.5 系列大模型作为 backbone，采用因果下一个 token 预测的持续预训练；键鼠控制接口，结合 5 Hz 观察与 30 Hz 控制；使用 AdamW + Warmup–Stable–Decay 优化；通过 VPT、D2E、OpenP2P 等游戏轨迹与 GPT-5.6-Luna 生成的回溯推理进行训练。

**📊 数据集**

数据集：85% 游戏轨迹来自 VPT、D2E、OpenP2P；15% 通用 vision‑language 与 embodied 数据；构建 5 种混合比例（S1–S5）以调节 Minecraft 比例；额外在部分轨迹加入 GPT 生成的推理注释。

**📈 对比分析**

比较方法：将训练得到的 GameScaling 模型与 Minecraft 专家模型、通用游戏模型以及当前前沿模型（Gemini‑3.8‑Flash、GPT‑5.6‑Sol、Claude‑Opus‑5.0 等）在 MCU、ViZDoom、OmniGameArena、GameWorld 四个基准上进行对比，评估包含时间暂停和实时两种情境。结果显示：在 3D 世界的暂停评估中，27B 的 GameScaling 提升约 6 倍；在实时评估中，9B 以上模型在所有四个基准上均超过现有前沿模型，平均得分提升约 25.3 点。

**⚠️ 局限性**

局限性：对远距 World（GameWorld）的泛化效果仍显不足，CPT 仅基于行为克隆缺乏任务成功/失败的环境反馈；大规模多样性混合对极端遥远域的帮助有限；模型在实时评估中受决策延迟影响，需进一步优化推理速度或引入在线反馈机制。

---

## 317. TEMPL: A Resource-Efficient and Transparent Model-Free Framework for Pseudo-Label Generation in Material CT Scans

**arXiv ID:** 2610.28974 | [PDF](https://arxiv.org/pdf/2610.28974v1)

**作者:** Ashiqur Rahman `[一作]` (Northern Illinois University), Hamed Alhoori `[通讯]` (Northern Illinois University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一个多阶段无模型伪标签生成框架，用于材料CT分割，能够在CPU上高效生成高质量多边形伪标签，无需标注训练数据。

**💡 创新点**

创新点在于将自适应阈值、形态学膨胀与自定义BFS区域生长相结合，构建了一个完全无模型、透明且资源友好的伪标签流水线。

**🔧 技术方法**

使用了Otsu阈值分割、连通域标记、形态学膨胀、基于BFS的区域生长、掩模精炼和向量化轮廓提取等经典图像处理技术。

**📊 数据集**

使用了未标注的硅砂CT图像（1192张）用于伪标签生成，以及带人工标注的树枝状结构CT（Dendrites）用于定量验证和比较。

**📈 对比分析**

与零射SAM、全监督3D U-Net、Watershed、W-Net和CUTS等方法对比，伪标签IoU达63.19%（SAM仅10.74%），CPU 15s/图像比SAM 28s快，基于伪标签训练的YOLOv11在目标检测与实例分割上与SAM相当。

**⚠️ 局限性**

局限性包括对CT伪影和噪声敏感、对形态复杂的连续结构适配有限，以及阈值需手动调节，适用范围受灰度对比度和内部均匀性限制。

---

## 318. Estimating Performance for Safe Deployment of Machine Learning Models

**arXiv ID:** 2610.29262 | [PDF](https://arxiv.org/pdf/2610.29262v1)

**作者:** Zeju Li `[一作]`, Ben Glocker `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

提出了 MOVAL，一个基于无标签测试数据的模型性能评估工具，利用置信度校准实现对多任务、跨域下的准确率、F1、AUC 等指标的无监督估计。

**💡 创新点**

创新点在于将置信度校准扩展到按类别自适应、支持多种置信度分数并通过集成提升稳健性，实现了在不需标签的情况下对多指标的精准估计。

**🔧 技术方法**

核心技术包括类别级温度缩放、DoC、ATC、TS-ATC 的自适应校准，结合多置信度分数（MCP、Entropy、Doctor、Energy）以及集成推理。

**📊 数据集**

使用了多个公开数据集：CIFAR10-LT、CIFAR100-LT、HAM10000 皮肤病变、脑病变 MRI（ATLAS）、前列腺 MRI、心脏 MRI（M&Ms）等，涵盖分类与 2D/3D 分割任务。

**📈 对比分析**

在 6,572 个不同域偏移实验中，MOVAL Ensemble 在准确率、F1、AUC 等指标上均优于现有最优方法，平均误差降低 30–50%，并在类不平衡及复杂域移位下保持稳定。

**⚠️ 局限性**

局限性在于依赖于验证集的校准参数，若测试域与验证域的置信度-正确性关系显著变化，估计误差可能增大；对置信度异常分歧的情况需人工干预。

---

## 319. PRISM: Learning Realistic Depth via Physics-Grounded Noise Disentanglement with Semantic-Geometric Collaboration

**arXiv ID:** 2610.29432 | [PDF](https://arxiv.org/pdf/2610.29432v1)

**作者:** Xiujian Liang `[一作]` (Fudan University), Jianhua Sun `[通讯]` (Shanghai JiaoTong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 PRISM 框架，用物理推理将深度传感器噪声拆分为“感知失效”和“测量误差”两种模态，并通过 3D 视觉基础模型（VFM）提取语义物理先验来指导噪声合成。

**💡 创新点**

创新点：① 物理驱动的噪声解耦，摆脱传统的整体黑盒建模；② 语义‑几何协同，将 VFM 提取的材质先验注入噪声生成；③ 层次正样本优先监督（H-PPS），解决失效区域稀疏且不平衡的训练难题。

**🔧 技术方法**

使用技术包括：Frozen 3D VFM Backbone（如 MoGe2/DepthAnything）、语义物理推理器、Bimodal Noise Disentangler（CNN 编码-解码 + 语义调制）、控制网络控制的扩散式噪声残差生成器（ControlNet + U‑Net + VAE），以及多尺度正样本权重 + 动态硬挖掘训练策略。

**📊 数据集**

主要数据集：ByteCameraDepth（真实深度多相机对齐数据）做训练与评测；NYU‑Depth‑v2 用于跨相机零样本泛化；此外在机器人实验中使用 Franka Research 3 与 Realman RM75 两套硬件平台。

**📈 对比分析**

与五大基线（GazeboDR、DepthSynth、ActiveStereo、DCL‑Depth、Stable‑Sim2Real）对比，PRISM 在 ByteCameraDepth 上 MAE 0.076、RMSE 0.085、δ<1.25 0.964，失效检测 IoU 0.952、F1 0.975、召回 0.991；在 NYU‑Depth‑v2 上 MAE 0.082、δ<1.25 0.941、IoU 0.923、F1 0.960。机器人下游任务平均成功率 92.5%，显著高于最优基线（89.1%）。

**⚠️ 局限性**

局限性：① 模型针对特定传感器训练，跨传感器泛化受限；② 当前仅处理单帧，缺乏时序一致性，易出现帧间抖动；③ 仅关注表面相关噪声，未覆盖交互诱发的外部噪声或极端光学环境。

---

## 320. Delay-Guided Attention for Quality Prediction in Multistage Manufacturing

**arXiv ID:** 2610.29220 | [PDF](https://arxiv.org/pdf/2610.29220v1)

**作者:** Jiarui Zhang `[一作]` (Northwestern University), Naichen Shi `[通讯]` (Northwestern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于延迟引导注意力的多阶段预测框架DAMF，用于多阶段制造系统中同时完成最终质量预测和时延恢复；

**💡 创新点**

创新点在于将动态互阶段时延建模与注意力对齐相结合，利用软时延分布作为对齐先验，并通过Viterbi解码生成可解释的时延路径；

**🔧 技术方法**

采用GRU编码器、DelayAlignment模块、门控残差融合、时延预测器（输出时延分布）、Viterbi解码、交叉熵与焦点损失等技术实现；

**📊 数据集**

使用了真实液糖（LiquidSugar）数据集及其延迟注入版本，以及基于CausalAI的合成控制时延基准；

**📈 对比分析**

与传统机器学习、深度序列模型以及多阶段基线进行对比，在LiquidSugar上实现RMSE、MAE、R²均优于MultistageNet；在控制时延基准上达到更高的精度、更低的误差和更好的峰值恢复；

**⚠️ 局限性**

局限在于仅建模相邻阶段的离散时延，未考虑非相邻交互、分支或反馈结构，且对连续时延建模不足。

---

## 321. JUICE: Unified Multimodal Modeling in Pretrained Representation Spaces

**arXiv ID:** 2610.28801 | [PDF](https://arxiv.org/pdf/2610.28801v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 322. Big Claims, Small Margins: Do Reported GraphLLM Gains Survive Simple Baselines?

**arXiv ID:** 2610.28714 | [PDF](https://arxiv.org/pdf/2610.28714v1)

**作者:** Michal Podstawski `[一作]` `[通讯]` (NASK National Research Institute), Michal Podstawski (NASK National Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对近期 GraphLLM 在文本属性图（TAG）节点分类任务中报告的提升进行再评估，比较固定代表性低成本基线在同一数据拆分上的表现，并审计公开代码中存在的评估风险。

**💡 创新点**

提出统一的低成本基线代表（TF‑IDF + LR、传播特征 + LR、Correct & Smooth、frozen e5‑large‑v2 MLP）并在相同拆分上进行相同搜索空间和验证集选择的调优；使用 Welch t‑test + Holm 校正评估显著性；系统性发现多种评估风险（测试集选择、标签泄漏、文本‑标签不匹配等）。

**🔧 技术方法**

利用传统文本特征（TF‑IDF、SVD）、预训练 LLM 嵌入（e5‑large‑v2）、逻辑回归、MLP、Correct & Smooth、label propagation；采用 TPE 搜索 50 次；采用 Welch t‑test 与 Holm 校正进行统计显著性检验。

**📊 数据集**

使用多种文本属性图数据集：Cora、Citeseer、PubMed、Photo、Computer、Books、Reddit、Texas、Wisconsin 等；分别采用 LLM4HeG、KCoT 的 LLaGA split 以及 LLMNodeBed 的重新构造 split。

**📈 对比分析**

与论文报告的最高 LLM 结果做直接对比；发现大多数低成本基线仅落后 3 分之内，部分任务仍显著落后；统计检验显示不少论文报告的提升在显著性上并不成立，说明部分 claimed gains 并不稳健。

**⚠️ 局限性**

仅复现基线并使用论文公布的基线数值，未重新跑 LLM 方法；调优预算固定为 50 轮；仅覆盖可恢复拆分的三篇研究；公开代码评估风险与实际实验结果可能不完全对应。

---

## 323. LTEM: LLM-Friendly Table-Enhanced Markdown for Table Understanding and Generation

**arXiv ID:** 2610.29188 | [PDF](https://arxiv.org/pdf/2610.29188v1)

**作者:** Eunbin Kim `[一作]` (Chungnam National University), Sangkeun Jung `[通讯]` (Chungnam National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了表格结构的六个原子特征与六级复杂度分类，构建了合成表格基准STB，并设计了最小化的Markdown扩展LTEM，用于评估LLM在表格理解和生成任务中的表现。

**💡 创新点**

创新点在于系统化地将表格结构拆解为原子特征并定义层级复杂度，引入轻量化、显式编码合并单元和层级标题的LTEM语法，并提供可控实验的STB基准。

**🔧 技术方法**

采用结构特征分解、上下文无关文法定义LTEM语法、LLM对话推理、自动化评测指标（nEM、ANLS、Token Normalized Performance）等技术。

**📊 数据集**

使用了合成表格基准STB（1,089张表）以及HiTab公开层级表格基准进行实验。

**📈 对比分析**

通过将同一表格序列化为六种格式，在多模型下进行表格问答（nEM、numEM）和生成（ANLS、Good+）评测，结果显示LTEM在表格理解上与HTML相当且无效率最低，在生成任务中略低于HTML但优于Markdown/CSV。

**⚠️ 局限性**

局限性包括：结果受数据分布影响，LTEM在生成任务中受曝光度低限制；模型-格式交互差异大；实验设置（prompt、解码）敏感；仅评估了六种代表性格式，未覆盖所有领域。

---

## 324. Agentic Music Post-Production

**arXiv ID:** 2610.29093 | [PDF](https://arxiv.org/pdf/2610.29093v1)

**作者:** Jaywon Koo `[一作]` (Rice University), Yuki Mitsufuji `[通讯]` (Sony Corporate Technology Center America, Inc.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于代理的音乐后期制作系统MixAgent，能够按照音频工程师的工作流程生成可解释、可编辑的混音计划并通过学习的检测器和批评者进行自适应参数修正。

**💡 创新点**

创新点在于：1）整合分层规划（单轨、子组、母路）与可编辑的音频效果链；2）构建MixCritic数据集，训练基于数值反馈的批评者，并通过偏好优化提升修正质量；3）使用检索式示例指导效果链选择，并在底部到顶部的递归修正中提升音质。

**🔧 技术方法**

技术包括：大型语言模型（Qwen、Gemini、GPT‑4o等）+ RLAT音频嵌入；差分音频处理库GRAFX用于示例生成；监督微调（SFT）+直接偏好优化（DPO）；检索式检索（文本+音频）指导；检测器与批评者网络。

**📊 数据集**

数据集：MedleyDB（未处理录音+专业混音）用于训练与评估；AlbumDB用于跨语料库转移；构建的MixCritic共45,657条修复示例。

**📈 对比分析**

与单次生成、Self‑Refine、Self‑Critic以及Diff‑MST、MST、StemFX等基线比较。MixAgent在MedleyDB上最低MSS（2.83）且TuneJury/Audiobox评分最高；在AlbumDB上表现同样优异但略逊于GPT‑4o‑audio的FxEnc++/AF指标。

**⚠️ 局限性**

局限包括：1）仅使用少量效果插件，限制可达混音范围；2）计划固定，批评者无法新增/删除效果；3）对专业混音的重建误差仍显高；4）评估集规模有限，泛化性受限；5）缺乏对用户可编辑计划的实际使用评估。

---

## 325. Weakly Supervised Fish Segmentation through Point-Supervised Tracking and Motion-Informed Language

**arXiv ID:** 2610.29047 | [PDF](https://arxiv.org/pdf/2610.29047v1)

**作者:** Vedanth Kallakuri `[一作]` (University of Michigan), Katherine A. Skinner `[通讯]` (University of Michigan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个弱监督鱼体分割管线，只需每条鱼点标注一次即可生成精确分割掩码。

**💡 创新点**

创新点在于将点跟踪结果转化为运动描述，由LLM生成语言提示，再结合LMPM++和点过滤、光流后处理，形成端到端的高效弱监督分割流程。

**🔧 技术方法**

采用CoTracker3点跟踪、LLM（如GPT‑4.1）生成运动语言、LMPM++运动‑语言对齐分割、背景减除和RAFT光流等技术。

**📊 数据集**

在公开的DeepFish（FishSeg）数据集上进行评估。

**📈 对比分析**

与Affinity‑LCFCN、SAM2、LangSAM、Saleh等基线比较，前景mIoU达到0.918，优于现有弱监督方法且与SAM2接近。

**⚠️ 局限性**

局限性包括高计算成本（光流、LLM调用）、实时性差、仅将所有鱼归为单一前景类、在视觉噪声（涟漪、悬浮颗粒）下点跟踪易漂移。

---

## 326. Context-Augmented, Session-Calibrated Blood Pressure Tracking from Photoplethysmography

**arXiv ID:** 2610.29091 | [PDF](https://arxiv.org/pdf/2610.29091v1)

**作者:** Farouk Ganiyu Adewumi `[一作]` (Morgan State University), Timothy Oladunni `[通讯]` (Morgan State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

评估可穿戴PPG与会话校准对血压跟踪的贡献，采用严格的受试者分层交叉验证和控制实验设计；

**💡 创新点**

创新点在于构建了明确的对照框架，系统分离并量化了上下文、PPG、时间等因素的增益，同时保留并报告所有负面结果；

**🔧 技术方法**

使用的技术包括梯度提升回归、特征选择与互信息排序、PPG吸收与非线性/捕获器特征嵌入、质量加权融合、传感器历史集成以及基于正则化的岭回归预处理；

**📊 数据集**

所用数据集包括主数据集AuroraBP（1,221名受试者）、BIDMC、MIMIC‑II、Welltory‑PPG、UCI、Appel–Theart 及 VitalDB；

**📈 对比分析**

方法通过对不同上下文/PPG/时间条件的多因素对照，计算MAE，主模型在7.697/6.139 mmHg的MAE下，增加上下文降低0.43/0.13 mmHg，PPG贡献有限，历史传感器提升略微（0.06/0.025 mmHg）且DBP改善不均衡；

**⚠️ 局限性**

局限性包括单中心受控协议、仅三次袖带校准、缺乏独立实时验证与不同人群泛化、皮肤色素效应未显著改善、未达到临床可接受的误差阈值。

---

## 327. LIFT: Scoped Constraints for Evaluating and Training Instruction Following

**arXiv ID:** 2610.28819 | [PDF](https://arxiv.org/pdf/2610.28819v1)

**作者:** Eran Goldstein `[一作]` (AI21 Labs), Daniel Gissin `[通讯]` (AI21 Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个新的指令跟随基准 LIFT（Localized Instruction-Following Tasks），用于文档草拟任务，约束被局部化到章节或段落，既包含可验证的程序化检查，也包含需要 LM 判断的内容约束。

**💡 创新点**

创新点在于：①将约束范围限定到文档的局部结构（章节/段落），①解决传统全局约束导致的奖励作弊；②将任务内容本身纳入奖励（content constraints），使得主任务遵循成为奖励的一部分；③提供完整的合成管线与公开训练集，能够按约束密度、类型、范围等维度自由生成大规模数据。

**🔧 技术方法**

技术手段包括：使用规则检查器验证可验证约束；使用 Gemma‑4‑31b 等大语言模型作为判定器评估内容约束；采用 GRPO 强化学习框架对模型进行 IF 训练；在评估时对 17 个模型（封闭模型 GPT‑5 系列、Claude 系列以及开放模型 Qwen、GLM 等）进行统一评分；利用可扩展的合成脚本自动生成任务与约束。

**📊 数据集**

数据集：LIFT‑Bench 共 10 个拆分（短、长、平面，每个拆分包含数千条任务），每条任务都有主题、结构化大纲和局部约束；训练集 LIFT‑Train 包含约 10 万条任务；所有数据通过合成管线生成，主题来源于 Wikipedia，约束模板与规则均公开。

**📈 对比分析**

比较方法：在 10 个拆分上对 17 个模型进行评估，分别记录结构化拆分（局部约束）与平面拆分（全局约束）的得分；对比基准 IFTrain 训练得到的模型，发现 LIFT 训练模型在保持内容一致性的同时显著降低奖励作弊；实验显示最先进模型在结构化拆分上可达 0.96，平面拆分不超过 0.85；训练实验表明 LIFT 训练能在 IFBench、AdvancedIF 等 OOD 基准上提升性能，而 IFTrain 训练则表现出明显退化。

**⚠️ 局限性**

局限性：①判定约束依赖 LM 评判器，判定误差可能影响得分；②约束类型仍相对有限，未覆盖所有复杂依赖；③合成数据虽然可扩展，但与真实人工撰写的文本在表达多样性和细节上仍存在差距；④训练与基准的格式差异可能导致对不同模型的偏差；⑤尚未在跨域场景下充分验证模型的泛化能力。

---

## 328. OSynth: A Longitudinal, Context-Sensitive Evaluation of Meeting Understanding

**arXiv ID:** 2610.29319 | [PDF](https://arxiv.org/pdf/2610.29319v1)

**作者:** Ian Porada `[一作]` (Otter.ai), Dmitriy Genzel `[通讯]` (Otter.ai)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了OSynth，一套大规模合成的组织会议语料库，包含音频、转录、幻灯片、邮件等多模态工件，并提供完整的潜在状态参考。

**💡 创新点**

创新点在于：①构建层次化LLM生成管道，实现多周会议与其关联工件的同步生成；②通过对抗式转录细化提升合成文本真实性；③利用潜在状态实现实体敏感评估的可解释度。

**🔧 技术方法**

主要技术包括：Claude Sonnet/Opus 4.6/4.7 生成多模态内容；Qwen3‑TTS 1.7B 进行语音合成；对抗式细化循环（判别者+修正者）提升转录自然度；以及基于WorkspaceState的事件化生成流程。

**📊 数据集**

数据集为OSynth，包含160个工作空间、2126次会议、570小时音频，覆盖软件开发、销售、医疗、学术等多领域。

**📈 对比分析**

在两套最先进的多模态LLM（OpenAI GPT‑4o/GPT‑5.5 与 Gemini 3.1 Pro/3 Flash）上评估，比较三种上下文层次（仅音频、前会输出、完整工作空间工件）。结果显示：随着上下文增大，实体相关指标（数字召回、技术术语召回、行动项分配准确率、跨会说话人一致性）显著提升，尤其L3层带来最多增益；整体WER提升有限。

**⚠️ 局限性**

局限性包括：①完全合成的语料缺乏真实录音中的长尾口语与交互多样性；②合成音频仍带有朗读式语调，可能影响ASR/声源辨识；③工件文本无错别字噪声，评估不涵盖真实文本错误；④评估受限于单一prompt模板和解码设置，绝对分数可能随工程调整而变。

---

## 329. SimHack: Predicting the onset of detectable reward hacking before RL training

**arXiv ID:** 2610.28707 | [PDF](https://arxiv.org/pdf/2610.28707v1)

**作者:** Junwei Deng `[一作]` (University of Illinois Urbana Champaign), Natalia Ponomareva `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种轻量化的行为模拟器，用以在启动RL训练前预测奖励劫持（reward hacking）何时会变得占主导地位，从而指导训练方案的选择。

**💡 创新点**

创新点在于：①将奖励劫持视为在训练过程中的可演化行为动力学；②通过学习少量参考RL跑（只含有限规模、不同课程和超参数）得到的低秩线性Markov转移模型，实现对未知训练配方的拷贝对抗模拟；③利用该模型进行拷贝预测而非仅依据初始劫持率，从而获得更稳健的预测分布。

**🔧 技术方法**

核心技术包括：行为状态的多维概率向量表示；批量条件下的线性Markov状态转移公式 S_{t+1} ∝ α(B_t) ⊙ S_t + β(B_t)；低秩结构学习 α、β 的参数化；模拟器对目标配方进行多次随机采样以得到劫持起始分布。

**📊 数据集**

使用了包含 325 次 LeetCode 代码生成 RL 训练跑的数据集，覆盖两种模型规模（Qwen3-0.6B、4B）、两类劫持漏洞（Overwrite、Lazy）以及 >19.3M 次 roll‑out，提供每步行为、奖励与劫持检测信息。

**📈 对比分析**

与传统方法（如直接跑 5 次目标 RL 训练）比较，模拟器在 median onset 相对误差 11.3%、p20 pinball 7.9% 与跨配方 Spearman 相关系数 0.852 上均表现更优；并通过模拟筛选的课程在真实训练中将劫持起始延后 96.6%，提升清晰率 4.7%。

**⚠️ 局限性**

局限性包括：①只适用于与参考跑共享同 prompt 池的配置；②模型仅捕捉三类行为，未考虑更细粒度的劫持策略；③预测分布为统计近似，可能无法捕捉极端情形；④需要至少一批参考跑来学习动力学，不能完全零样本；⑤在更复杂环境或不同任务类型时的泛化性尚待验证。

---

## 330. End-to-End Reinforcement Learning for Quadrupedal Jumping in Lunar Gravity

**arXiv ID:** 2610.29586 | [PDF](https://arxiv.org/pdf/2610.29586v1)

**作者:** Levin Müller `[一作]` (ETH Zurich), Marco Hutter `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

针对月球低重力环境，设计并训练了一套单一的多阶段强化学习控制策略，使四足机器人Magnecko能够实现从起跳点到目标点的点对点跳跃，并在硬件上通过重力减载装置完成垂直跳跃测试。

**💡 创新点**

创新点包括：① 将月球跳跃的抛射轨迹简化为可直接从命令中得到的抛物线，避免在线轨迹规划；② 采用五阶段训练框架（STANCE–LAUNCH–FLIGHT–LANDING–RECOVERY）将所有阶段融入同一 MLP 控制器；③ 通过奖励设计和阶段判定实现对飞行中的姿态快速恢复，且不依赖外部传感；④ 在仿真与硬件间实现高保真转移，验证了能量与扭矩需求均在机器人硬件极限内。

**🔧 技术方法**

使用技术包括：强化学习（PPO）与多层感知器（hidden layers 512/256/128，ELU）；IsaacLab 物理仿真；多阶段奖励与转移逻辑；重力减载硬件（弹簧缆绳上浮力）；扭矩与功率采集与分析；领域随机化与自适应训练（Curriculum）。

**📊 数据集**

数据集主要来自仿真采样：跳跃距离 0–5 m、方向 ±45°、升降角 ≥45°、爬升速度 ≤3 m/s、顶点高度 ≥0.8 m，斜坡倾角 0–5°、方位 0–360°；在硬件测试中使用单一垂直跳跃命令（顶点 0.8 m）。

**📈 对比分析**

通过在仿真中对 10 000 条随机命令进行落地点精度与成功率评估；在硬件上对垂直跳跃进行扭矩/功率曲线测量。性能结果：平地落地点误差 <0.3 m（相当于机器人尺寸的一半），斜坡误差略高；平地、斜坡成功率均超过 88%；在控制冻结 1 s 时成功率仍保持 ~80%；起跳时扭矩与功率峰值在机器人 56 Nm 及可接受功率范围内。

**⚠️ 局限性**

局限性包括：① 硬件测试仅验证垂直跳跃，未验证水平跳跃；② 重力减载装置在水平方向阻尼大，未能模拟真实月球自由落体；③ 地面模型为刚性接触，未考虑月球粉尘软性接触；④ 仅针对 MGN 机器人，缺乏更通用的多机器人验证。

---

## 331. Why is AI-generated language so hard to understand?

**arXiv ID:** 2610.29162 | [PDF](https://arxiv.org/pdf/2610.29162v1)

**作者:** Thomas Hikaru Clark `[一作]` (New York University), Kyle Mahowald `[通讯]` (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了前沿LLM在文本生成中出现的与人类阅读难度相关的句法特征，并将其与人类文本以及回译重写的版本进行系统对比。

**💡 创新点**

创新之处在于将心理语言学中的难度指标（如对象定语从句、低频n-gram、动词-名词转换等）首次应用于AI生成文本的分析，并通过长度惩罚的强化学习实验验证信息密度优化导致的非人类化语言倾向。

**🔧 技术方法**

主要技术包括spaCy依赖句法分析、n-gram概率统计、NLLB-200双向回译、强化学习（GRPO）与长度惩罚正则化。

**📊 数据集**

使用的数 据集涵盖多域AI生成语料（聊天、代码、论文），人类基准语料（OpenSubtitles、StackExchange、PERSUADE 2.0 论文），以及通过NLLB-200生成的回译并行语料。

**📈 对比分析**

通过与人类基准及回译语料的统计对比，发现前沿模型在低频n-gram、动词-名词转换、无定语的对象从句等难度特征上显著高于人类，且在长度惩罚实验中模型明显偏好更短的句法变体，表明信息压缩压力导致语言变得更难理解。

**⚠️ 局限性**

局限性包括仅测试了少量模型、仅使用英文数据、种子样本有限，且未能直接观察后训练过程，受计算资源限制导致实验规模受限。

---

## 332. Certified Abstention for Provably-Unidentifiable Index-Phase Recovery in Image Registration

**arXiv ID:** 2610.29272 | [PDF](https://arxiv.org/pdf/2610.29272v1)

**作者:** Dinh Tuan Nguyen `[一作]` (Pukyong National University), Junghwan Oh `[通讯]` (Pukyong National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一套针对受整数行索引偏移影响的扫描图像的可靠注册与恢复框架，并给出了判定可恢复性与否的分区置信证书。

**💡 创新点**

创新点包括：证明位置保真与相位保真相互独立并给出不可恢复性阈值；提出必要且充分的同步原语；以及基于可识别阈值与 conformal 预测的抗幻觉恢复机制。

**🔧 技术方法**

技术手段包括：计数锁定采样模型、Robinson–Milanfar 可识别阈值、子格时间戳解码、整数位移最大似然估计、周期性与纹理无效性检测，以及 conformal 预测的风险控制。

**📊 数据集**

数据集涵盖：实际光声扫描仪的真实采样记录、仿真采样数据、滚动快门与冷冻电镜等多模态公开数据集。

**📈 对比分析**

与传统刚性配准、列均值去纹理、平滑先验、学习模板、低秩PCA 等基线对比，证书化方法在可识别区域实现 100% 准确且在不可识别区域 0% 伪造，而其他方法伪造率高达 98%。

**⚠️ 局限性**

局限性在于仅关注相位错误而非连续失真；假设采样为整数计数且存在非零重新初始化延迟；置信证书仅保证平均区间风险，对单实例错误不做保证，且在不同信噪或模态下需重新校准。

---

## 333. DisasterRoute: Evidence-Guided Dual-System Routing for Visual Disaster Question Answering

**arXiv ID:** 2610.29231 | [PDF](https://arxiv.org/pdf/2610.29231v1)

**作者:** Tan Zhang `[一作]` (UESTC), Ping Hu `[通讯]` (UESTC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于双系统的灾害视觉问答框架（DisasterRoute），通过快速选项决策与有条件的生成式推理实现低延迟与高准确率。

**💡 创新点**

创新点在于：① 通过选项判别训练（option‑discrimination）提升一次性多模态前向推断的决策精度；② 引入证据状态头与多维度路由决策（置信度、证据充分性、选项-状态一致性）来动态决定是否调用生成式推理；③ 采用“候选救援”机制在需要时扩展候选答案集合，降低因误判导致的遗漏。

**🔧 技术方法**

核心技术包括：灾害领域适配的视觉‑语言模型（Disaster‑S1 与 Disaster‑VL）、多任务跨域训练、温度缩放校准、基于特征的证据状态预测、路由门控与自适应候选扩展、生成式推理与严格解析。

**📊 数据集**

使用 DisasterBench（低空 UAV 图像下的灾害问答数据集）进行训练、验证与测试。

**📈 对比分析**

对比方法包括：全生成推理（DVL‑FR）、仅读取 logits（DVL‑Logit）、仅直接决策（S1‑Direct），以及通用 VLM（InternVL3.5‑2B 等）。在 Accuracy 上，DisasterRoute 达到 82.42%，比直接决策提升 4.64pp，且平均延迟比全生成推理降低约 69%（约 1.22 s vs 3.92 s）。

**⚠️ 局限性**

主要局限：① 仍需额外的生成推理导致部分问题的尾部延迟较高；② 仅在约 23.7% 的样本上使用生成式推理，路由门仍可能错误判定导致误判或漏判；③ 需要在不同灾害场景中进一步验证其泛化能力。

---

## 334. Enhancing Cross-Grade Generalization in Banana Variety Classification via Hybrid CNN-Transformers and Knowledge Self-Distillation

**arXiv ID:** 2610.29008 | [PDF](https://arxiv.org/pdf/2610.29008v1)

**作者:** Khush Kataruka `[一作]` (Sardar Vallabhbhai National Institute of Technology), Praveen Kumar Chandaliya `[通讯]` (Sardar Vallabhbhai National Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

针对不同质量等级的香蕉品种分类，提出了一种双流 Swin Transformer 与 ResNet 的混合框架，并通过 KL 散度自蒸馏提升跨等级泛化能力。

**💡 创新点**

创新点包括：①引入留一等级 (LOGO) 的严格评估协议，真实反映跨等级迁移难度；②设计教师-学生双流结构，融合全局注意力与局部纹理特征；③在特征层加入 KL 散度自蒸馏，强制两流预测分布一致，提升鲁棒性和收敛速度。

**🔧 技术方法**

技术手段：卷积神经网络（ResNet50）、视觉 Transformer（Swin‑Tiny）、双流融合、KL 散度自蒸馏、交叉熵联合监督、Grad‑CAM/Score‑CAM 解释、训练动态与效率分析。

**📊 数据集**

使用公开的 31,678 张香蕉图像数据集，包含三种品种（Pachbale、Rasbale、Elakki‑bale）和三种质量等级（Grade 1–3），每种等级约 10,000 张。

**📈 对比分析**

与单一 CNN、单一 Transformer 以及其他混合模型对比，LOGO 下平均准确率由基线 72.9% 提升至 81.25%，F1 分数提升至 80.35%。在最困难的 Grade 1 上准确率从 64.65% 提升至 73.86%；在最佳 Grade 3 上达到 92.09%。训练轮数显著减少，KL 蒸馏版平均早停轮数为 8–10 轮，最快收敛。

**⚠️ 局限性**

局限性：①模型对单源（单等级）训练的跨等级迁移仍受限，需进一步研究多源或自适应策略；②对实时低算力设备的部署尚未验证；③仅在香蕉品种上验证，需扩展到其他农作物或更大规模多品种数据集。

---

## 335. Steering Black-Box Large Language Model Agents with Pitfall Anticipation World Model

**arXiv ID:** 2610.29139 | [PDF](https://arxiv.org/pdf/2610.29139v1)

**作者:** Jiongxiao Wang `[一作]` (University of Wisconsin–Madison), Panpan Xu `[通讯]` (AWS AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了 Pitfall Anticipation World Model (PAWM)，为冻结的黑盒LLM代理提供环境特定的陷阱预测与引导，从而显著提升其任务成功率。

**💡 创新点**

创新点包括：①仅在潜在陷阱出现时干预的门控世界模型；②通过合成陷阱数据的监督学习（SFT）先学习预测与指导，再使用 GRPO 与目标代理共训练以优化任务表现；③在干预策略上实现成本与效果平衡。

**🔧 技术方法**

技术手段：基于 Qwen3.5‑27B 的世界模型与 Claude Sonnet 4.5 的黑盒代理；合成陷阱数据生成与 SFT；GRPO 强化学习共训练；LLM 生成的注释与词频/语义分析；门控机制。

**📊 数据集**

数据集：ScienceWorld、WebArena‑Lite（任务与轨迹），以及从这些环境收集的代理轨迹和合成陷阱数据。

**📈 对比分析**

对比方法：无指导、RAG、Pre‑Task Guidance、不同输出内容的世界模型；在 ScienceWorld 与 WebArena‑Lite 上评估。PAWM+GRPO 分别取得 78.9% 与 80.4% 的准确率，较无指导提升 24.3%/7.7%，在多模型上均显著优于基线。

**⚠️ 局限性**

局限性：实验高度依赖专有LLM API 及高成本；未在所有代理上单独共训练；仅在 ScienceWorld 与 WebArena‑Lite 上验证；世界模型为任务专用，合成陷阱标签可能包含噪声或偏差。

---

## 336. Memory-Centric Continual Learning for AI Agents

**arXiv ID:** 2610.29357 | [PDF](https://arxiv.org/pdf/2610.29357v1)

**作者:** Dongfang Li `[一作]`, Min Zhang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出Memory-Centric Continual Learning框架，以记忆为中心实现自主智能体的持续学习。

**💡 创新点**

创新点在于将记忆划分为substrate、asset、policy三轴体系，强调记忆对未来行为的提升，而非仅检索；并提出两循环生命周期、评估协议和规模律。

**🔧 技术方法**

采用基于大型语言模型的外部/内部记忆存储，结合强化学习与记忆管理策略，实现写入、检索、整合、更新与遗忘。

**📊 数据集**

在多任务模拟、Web/桌面自动化、机器人导航、协作任务等环境中使用合成与真实交互数据进行评估。

**📈 对比分析**

通过匹配重置基线、完整历史、静态RAG、计算匹配等对照，测得在Retention、Adaptation、Transfer三个维度上显著优于现有RAG/LLM基础方法，提升约15-25%效果。

**⚠️ 局限性**

局限在于长期信用分配困难、稳定性–可塑性平衡欠佳、以及跨任务迁移与参数融合仍需改进。

---

## 337. pplx-embed-v2-late Technical Report

**arXiv ID:** 2610.29523 | [PDF](https://arxiv.org/pdf/2610.29523v1)

**作者:** Markus Krimmel `[一作]` (Perplexity AI), Bo Wang `[通讯]` (Perplexity AI)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发并发布了两个多模态多语言的后交互模型——0.6B 与 9B，分别用于文本、图像与视觉文档检索，并支持跨模型检索。

**💡 创新点**

创新点在于：① 在 18B 反向对比训练教师模型的基础上，引入表示层级的 distillation（token‑level 余弦距离），显著提升小模型表现；② 通过联合使用 LoRA、梯度缓存与自研 MaxSim kernel，支持大 batch 训练；③ 设计混合采样与黄金比例 Kronecker 序列，确保多语言多域数据均衡且分布均匀；④ 兼容 0.6B 与 9B 的共享嵌入空间，实现“索引 9B、查询 0.6B”的混合部署。

**🔧 技术方法**

技术手段包括：Qwen3.5 混合注意力架构、InfoNCE 与 triplet 训练、LoRA、梯度缓存、Triton MaxSim kernel、欧氏距离表示层 distillation、全层 fine‑tune 与部分 LoRA 结合、金字塔采样策略。

**📊 数据集**

使用了 28 个公开数据集（共 981 个子集、1.86 亿 query‑doc 对），覆盖文本、文本‑图像、文本‑视觉文档；评测基准包括 72 个单跳检索任务（财务、法律、健康等）、ViDoRe‑V3、Q2D‑Web、PPLX‑Q2I、MIRACL‑Vision 以及 BrowseComp+ 代理检索。

**📈 对比分析**

与基线（Nemotron‑3 Embed‑8B、Qwen3 Embedding、Gemini‑Embedding‑2、Voyage‑4 等）和同级模型（topk‑embed‑v1 0.8B/2B、EVIE‑8B/4.5B、Nemotron ColEmbed V2）相比，9B 在 72 任务上平均 81.3% nDCG@10，0.6B 在 1B 以内模型中领跑 78.0%；在 Q2D‑Web 结合评估中分别达 74.8% 与 73.6%；在 ViDoRe‑V3 图像检索中 9B 与 0.6B 分别为 65.2% 与 62.3%，并在 MIRACL‑Vision 及 PPLX‑Q2I 上位居前列；代理检索中 9B 取得 64.0% 的准确率，超越 Reason‑ModernColBERT 与密集检索基线。

**⚠️ 局限性**

局限性包括：① 主要在公开数据集与内部基准上评测，部分任务列表未公开导致可复现性受限；② 对图像嵌入维度限制为 128，可能影响对高分辨率图像的表达；③ 混合部署仍需维护双模型索引与查询同步，增加系统复杂度；④ 训练与推理资源依赖大型 GPU，难以在更小规模设备上快速迭代。

---

## 338. Guess and Send: World-Model-Guided Semantic Packet Transmission for Robust Video Perception

**arXiv ID:** 2610.29587 | [PDF](https://arxiv.org/pdf/2610.29587v1)

**作者:** Yan Gong `[一作]` (Bournemouth University), Pei Xiao `[通讯]` (University of Surrey)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于世界模型的语义包传输框架，利用接收端预测的场景状态与发送端验证来选择最有价值的特征包并进行不等错误保护；

**💡 创新点**

创新点在于将接收端的时空预测状态与任务影响评分结合，形成“接收器相对”任务驱动的包选择，同时采用 3‑bit 特征精度、离散 LDPC 码率分配和整体码字打包实现高效的数字传输；

**🔧 技术方法**

核心技术包括：共享特征空间世界模型（全局仿射运动 + 细粒度残差变位 Transformer）、发送端基于特征对齐的验证、基于两次对比任务输出的任务影响评分、Lagrangian 多选背包分配以及 3‑bit 量化和 LDPC 编码；

**📊 数据集**

在 KITTI‑MOTS（行驶场景）和 VisDrone‑2019‑MOT（无人机航拍）两大视频数据集上进行实验；

**📈 对比分析**

与 H.264/H.265 + LDPC、DeepJSCC、特征变化/发送者置信度等基线进行对比。实验表明，在 4.13% 带宽预算下，Rayleigh 衰落 0 dB 时提升约 11.3pp 的 J&F；在 AWGN 下在解码阈值附近同样优于传统编解码器；整体在低 SNR 或低带宽条件下表现最为突出；

**⚠️ 局限性**

主要限制包括：需对每帧进行双重任务评估和多头编码，导致实时性和能耗上升；世界模型仅预测已有缓存内容，无法处理新出现的场景；在极高 SNR 或大带宽场景下优势有限；

---

## 339. SpecEvo: Speculative Evolution with Large Language Models for Cost-Efficient Scientific Discovery

**arXiv ID:** 2610.28706 | [PDF](https://arxiv.org/pdf/2610.28706v1)

**作者:** Ha Minh Hieu `[一作]` (Hanoi University of Science and Technology), Min-Yen Kan `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SpecEvo——一种多LLM的演化搜索框架，采用“speculate‑then‑consolidate”工作流，在轻量化Speculators进行并行探索的同时，周期性地用轻量化Advisor提炼搜索证据，利用稀疏的前沿模型Navigator在停滞时做合成、修补或重构，以实现高效的科学发现；

**💡 创新点**

创新点在于将搜索证据持续凝聚为可重用的指导（Advisor）和停滞感知的前沿干预（Navigator），实现轻量化探索与高成本前沿推理的自适应协同，显著降低API成本并提升解决质量；

**🔧 技术方法**

使用多模型异构架构（轻量化Speculator、Advisor和稀疏前沿Navigator），基于MAP‑Elites风格的行为档案、PCA+语义嵌入的描述符、动态停滞信号、基于归档的自适应种子选择与多样化变异提示；

**📊 数据集**

在176个真实科学与算法发现任务上评估，涵盖数学、系统优化、组合优化和方程发现四大领域，具体包括AlphaEvolve Math、ADRS、CO‑Bench、LSR‑Synth等数据集；

**📈 对比分析**

与单一LLM（OpenEvolve、GEPA、AdaEvolve、EvoX）和异构模型方法（LEVI、RelayEvolve）在匹配的时间、API费用和评估预算下比较，SpecEvo在多数任务上获得最高分数，并在三小时预算下将API成本降低50%–80%，同时保持或提升最优解质量；

**⚠️ 局限性**

局限性包括仍需依赖昂贵前沿模型的稀疏调用，模型参数与提示的调优仍具挑战，且在极大规模任务或对解释性要求极高的场景下效果待进一步验证。

---

## 340. ChronoSynth: Constraint-Aware Question Synthesis over Temporal Knowledge Graphs

**arXiv ID:** 2610.29339 | [PDF](https://arxiv.org/pdf/2610.29339v1)

**作者:** Feng Liang `[一作]` (National University of Defense Technology), Xiang Zhao `[通讯]` (National University of Defense Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并提出一种针对时序知识图的约束感知问题生成框架ChronoSynth，解决时序语义漂移问题。

**💡 创新点**

通过显式构建时序意图状态、使用索引模式记忆检索兼容的问句模板、对模板进行类型化适配，并加入验证与反馈修正机制，显著提升时序一致性。

**🔧 技术方法**

时序意图状态建模、模式记忆索引检索、模板适配、约束验证与反馈修正、基于LLM的生成器。

**📊 数据集**

CRONQUESTIONS 和 MultiTQ 两个时序知识图问题生成基准。

**📈 对比分析**

与多种基线（训练模型、直接提示、KQG‑CoT）比较，ChronoSynth 在 BLEU、ROUGE、CIDEr 等指标上均超过基线 15–76% 提升，且在人工评测中时序忠实度与答案可解性最高。

**⚠️ 局限性**

依赖预定义的时序规则集与手工构造的状态提取，对新图模式或语言支持有限，验证机制不保证语义完全正确，且在非目标域和多语言场景未验证。

---

## 341. AI Security Literacy: Ecosystem-based Competency Framework and K-12 Learning Platform

**arXiv ID:** 2610.28859 | [PDF](https://arxiv.org/pdf/2610.28859v1)

**作者:** Eric Yocam `[一作]` (California Polytechnic State University), Paris Kalathas `[通讯]` (California Polytechnic State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出AI安全素养框架并用NeuralFortify演示实现

**💡 创新点**

将AI安全视为独立素养领域，并在六维度与跨角色线索上系统化

**🔧 技术方法**

利用浏览器端实时推理、TensorFlow.js、WebGL、攻击与防御脚本

**📊 数据集**

采用RealToxicityPrompts、TruthfulQA、BBQ、CIFAR‑10、SMS Spam、Pew ATP、RAND ATP等公开数据集

**📈 对比分析**

通过对抗攻击的FGSM/PGD对比、阈值对毒性/诈骗检测的影响、真伪问题的准确率等实验，展示模型易受攻击且可信度低

**⚠️ 局限性**

尚未进行测量效度验证，样本仅为教师和成人调查，缺乏学生数据，未评估学习效果

---

## 342. Failure Modes of Vision-Language Models for Human Activity Understanding in Thermal HRI

**arXiv ID:** 2610.29244 | [PDF](https://arxiv.org/pdf/2610.29244v1)

**作者:** Han Hu `[一作]` (Chinese University of Hong Kong), He Chen `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估七个本地可部署的视觉‑语言模型在热像人类活动理解中的表现，涵盖细粒度动作识别、交互定位与拒答行为；

**💡 创新点**

系统性诊断热像VLM的错误模式，并对比不同采样率、输入变换、推理增强、RGB与热像、给定交互与否等干预对模型性能的影响；

**🔧 技术方法**

采用Qwen、Gemma、Cosmos3、RoboBrain等VLM，配合vLLM推理框架、ViTPose++骨架估计、伪彩色/对比增强等输入预处理，以及直接回答与启发式推理两种模式；

**📊 数据集**

使用两个人为中心的热像数据集：OctoNet（短片单动作）和Thermal-IM（长片交互与时间标注），并配有同步RGB记录；

**📈 对比分析**

通过精确动作识别准确率、类别召回、pair F1、event F1、tIoU、拒答准确率等指标进行比较；结果显示模型在粗动作上约50%准确率，细粒度动作仅16‑20%；pair/event F1仅0.25‑0.48，RGB明显优于热像；各种干预提升有限；

**⚠️ 局限性**

局限性包括：实验仅基于固定摄像头和静态片段，未考虑主动感知与机器人行为耦合；未深入模型内部失败机制；评估指标为代理任务，未直接验证机器人应用效果。

---

## 343. AgentPolicyForge: From Executable Environments to Policy Benchmarks and Adaptive Guards

**arXiv ID:** 2610.28989 | [PDF](https://arxiv.org/pdf/2610.28989v1)

**作者:** Chao Shang `[一作]` (AWS AI Labs), Sandesh Swamy `[通讯]` (AWS AI Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 AgentPolicyForge 框架，利用可执行环境生成符合部署策略的工具调用案例，用于训练和评估政策守护器；

**💡 创新点**

创新点在于：①可执行环境驱动的案例构造与标签生成；②循环的攻防共进训练（SFT + 策略档案更新）以提升守护器性能；③创建了 PolicyGym 数据集，提供可验证的多策略案例；

**🔧 技术方法**

使用了可执行环境适配器、LLM 评判器与可执行规则验证器、监督微调（SFT）训练守护器，以及策略档案的共进演化算法；

**📊 数据集**

数据集为 PolicyGym，来源于 Agent World Model (AWM) 1000 个可重置情景，包含 698 场景的 5,655 条种子，生成 698 条训练案例和 254 条人类审核的测试案例；

**📈 对比分析**

通过与多种基线（PolicyGuard‑4B、GPT‑6‑luna、Haiku、Clef‑Flash 等）以及内部的自我调节代理比较，最终守护器在 ID/OOD 上将攻击成功率从约 14% 降至 8.2%（≈ 44% 降低），且轻微提高了无害请求的拒绝率；

**⚠️ 局限性**

局限包括：仅评估合成策略，未验证对新策略或真实部署的泛化；依赖单一生成种子和 Qwen3 版模型，未统计训练变异；使用的评判器和攻击者均为专有 Claude 模型，可能引入偏差；实验未比较完全可执行规则检查的效果；

---

## 344. Geometric knowledge spaces, Attention pairing and Architectures

**arXiv ID:** 2610.29526 | [PDF](https://arxiv.org/pdf/2610.29526v1)

**作者:** Shang Xu `[一作]` `[通讯]`, Shang Xu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

本文提出一种基于几何框架的知识世界与智能模型，统一表述注意力机制与循环网络，并通过几何空间构造启发新的神经网络架构。

**💡 创新点**

创新点在于引入知识世界、更新映射与评估配对三元组，将传统注意力与RNN等架构映射为几何对象，提供设计新层的几何工具箱。

**🔧 技术方法**

主要技术包括几何深度学习、范畴论中的内部映射、张量运算、线性/非线性映射与归一化方法，结合软最大、矩阵迹等经典操作。

**📊 数据集**

论文未涉及具体数据集或实验，侧重理论框架与示例构造。

**📈 对比分析**

由于缺乏实验评估，本文未给出与现有方法的性能对比；仅通过示例演示框架能还原已知架构。

**⚠️ 局限性**

局限性包括：缺乏实现细节与训练经验，尚未验证新的几何构造在实际任务中的可行性与效率；对数值稳定性与计算成本的分析也未展开。

---

## 345. HorizonComposer: Spatiotemporally Consistent Driving Video Editing with Enriched Traffic Semantics

**arXiv ID:** 2610.28829 | [PDF](https://arxiv.org/pdf/2610.28829v1)

**作者:** Mauricio Soroco `[一作]` (Simon Fraser University), Ziyu Jiang `[通讯]` (NEC Labs America)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HorizonComposer，针对驾驶场景的视频编辑，能够在保持交通语义完整的前提下实现全局外观编辑。

**💡 创新点**

创新点在于三方面：①构造混合现实-合成对齐数据；②利用像素级密集奖励进行离线强化学习以保护关键语义；③引入“thinking frames”缓解临时冲击，提升时空一致性。

**🔧 技术方法**

核心技术包括基于 Cosmos‑Transfer2.5 的 Diffusion Transformer、ControlNet 分支、离线 RL (CFGRL) 的优势条件化、SAM 场景变化检测、CLIP 与 VLM 语义对齐。

**📊 数据集**

使用 Boreas、nuScenes、Waymo、nuPlan 作为真实驾驶数据；SyntheticPip 合成数据；SeRM 与 BDD100K 进行下游任务评估。

**📈 对比分析**

与 CCEdit、UniVideo 以及 SyntheticPip 对比，在安全关键结构保真、路面标记 mIoU、车道检测、物体检测上分别提升 20% 以上的内容保持率、5% 以上的 mIoU、7% 以上的 mAP@75，并在用户偏好测试中获得 39% 的整体提升。

**⚠️ 局限性**

局限性包括对高动态复杂场景仍易出现微小结构失真；思考帧长度需手工设定；对极端天气或夜间场景的适应性待进一步验证。

---

## 346. Runtime Semantic Discovery for Evolving Action Alphabets

**arXiv ID:** 2610.28682 | [PDF](https://arxiv.org/pdf/2610.28682v1)

**作者:** Pietro Bernabei `[一作]` (IMT School for Advanced Studies Lucca), Rosario Pugliese `[通讯]` (University of Florence)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种基于执行轨迹邻域的轻量级运行时语义发现方法，用以在软件系统演化时识别新出现的动作与旧动作的功能对应关系，从而支持监视器和规范的自动适配。

**💡 创新点**

创新点在于：① 引入邻域（k-邻域）等价关系，提供局部可计算的动作相似度度量；② 通过抬升轨迹并抽象未共享动作，将不同字母表的系统映射到同一抽象空间进行比较；③ 定义了多种演化模式（重命名、拆分、合并、序列细化）对应的规范适配规则。

**🔧 技术方法**

技术手段包括：基于有限状态机的轨迹收集、邻域集合构造、k-邻域投影、轨迹抬升与抽象函数、以及对LTL规范的模式化替换。

**📊 数据集**

本文未使用公开数据集，而是通过构造的示例系统（如S₀–S₄）以及理论示例来演示方法的可行性；未来工作计划进行实验评估。

**📈 对比分析**

方法比较上，作者通过示例展示邻域等价在不同演化场景下能正确识别对应动作；但并未给出定量性能指标（如时间复杂度、误判率），预期在实际系统中需进一步评估。

**⚠️ 局限性**

局限性：① 邻域等价仅为弱约束，无法保证完整行为等价；② 对于复杂的多映射或模糊演化，需要手工验证或补充语义信息；③ 适配规则对LTL的语义支持有限，可能不适用于所有逻辑片段；④ 目前缺乏实验验证，真实系统中噪声与不完整轨迹的影响未探讨。

---

## 347. Multi-agent LLMs on the Cloud-Edge-IoT Continuum: A Systematic Mapping Study on Architectures, Deployment, and Evaluation

**arXiv ID:** 2610.28702 | [PDF](https://arxiv.org/pdf/2610.28702v1)

**作者:** Mahyar Tourchi Moghaddam `[一作]` (University of Southern Denmark), Francesco Daghero `[通讯]` (University of Southern Denmark)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性映射了 45 篇多代理 LLM 系统在云-边缘-IoT 连续体中的架构、部署模式和评估实践。

**💡 创新点**

创新点包括识别了五大架构家族与六种部署模式，提出了可比评估报告清单和针对 IoT 约束的部署指导。

**🔧 技术方法**

采用系统化映射研究方法（PRISMA 2020、数据库检索、双盲筛选、编码表格），并对每篇工作进行架构、部署、域与评估维度的定量与定性编码。

**📊 数据集**

使用的技术与数据集主要为公开的 LLM 模型（Qwen、Llama、Phi、Gemma、Mistral 等，参数量低于 8B）以及对应的硬件平台（Jetson、Raspberry Pi、手机、Kubernetes/K3s 边缘集群、云 GPU）。不涉及统一公开数据集，而是根据原文列出的实验环境和任务。

**📈 对比分析**

通过统计各类维度出现频次、评估家族组合以及已公开的基准结果，比较发现多数系统在延迟、成功率和成本上有提升，但由于缺乏统一基准、评估条件不一致，难以做跨论文直接性能对比；仅有 6 篇论文公开代码或数据。

**⚠️ 局限性**

局限性包括：检索词可能遗漏未使用“agent”词汇的相关工作；缺少统一的 benchmark 及评估基准；artifact 共享率低，导致可复现性差；隐私与安全评估仅出现两篇；状态连续性与迁移等关键在 IoT 场景下的评估不足。

---

## 348. Broadcasting in Centimeter-Wave Gigantic MIMO Systems: Beam Shaping is All You Need

**arXiv ID:** 2610.29542 | [PDF](https://arxiv.org/pdf/2610.29542v1)

**作者:** Boyu Ning `[一作]` (KTH Royal Institute of Technology), Robert W. Heath `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在厘米波 G‑MIMO 系统中通过波束成形技术实现环境感知的广播波束，提出了一种新型的“prostrate beam”来均匀覆盖服务区域。

**💡 创新点**

创新点在于把非凸波束成形问题转化为目标追踪优化（TPO），并通过变量替换与连续凸逼近（SCA）构造可求解的 QCQP，从而实现对地面均匀功率分布的精确控制。

**🔧 技术方法**

采用几何广播模型、路径损耗模型、矩阵表示、目标追踪优化、凸二次规划（QCQP）以及成功率逼近（SCA）等技术手段。

**📊 数据集**

主要使用仿真数据（N=64、f=13 GHz、h_BS=30 m、θ_tilt=π/4、O=128 等），并未使用公开实验数据集。

**📈 对比分析**

与传统基于 S‑SARV 的宽波束做对比，prostrate beam 在覆盖距离、有效等效辐射功率（EIRP）和谱效率上分别提升约30–50%，且能覆盖原先传统波束不到一半的距离。

**⚠️ 局限性**

局限性包括需要足够的总功率和天线自由度；算法复杂度较高，只适用于离线代码书设计；且在功率不足时可能导致全区域均匀覆盖失败。

---

## 349. To Manipulate or Not: Opinion Dynamics in LLM Interactions

**arXiv ID:** 2610.28875 | [PDF](https://arxiv.org/pdf/2610.28875v1)

**作者:** Mohammad Siahkamari `[一作]` (Temple University), Stephen MacNeil `[通讯]` (Temple University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了LLM‑LLM辩论框架，系统性测量两角色在20轮讨论中的立场变化。

**💡 创新点**

将自我报告立场与对话内接受/拒绝状态双重测量相结合，揭示模型身份和立场指令对双向意见动态的影响。

**🔧 技术方法**

使用GPT‑6 Luna和Gemini 2.5 Flash Lite两大模型，通过OpenRouter自动化调用，并记录ACCEPT/REJECT/IGNORE等交互，构建完整的辩论平台。

**📊 数据集**

生成了480个独立20轮辩论实验数据，共19 200条对话记录，涵盖四种模型组合、三种立场、两种说服指令与两种语句表述。

**📈 对比分析**

采用线性回归与Welch检验比较不同模型组合下的意见偏移与绝对间距变化，结果显示同模型对话显著收敛，跨模型对话趋于发散，且说服指令未产生一致显著效果。

**⚠️ 局限性**

研究仅限于两模型与单一议题（气候变化归因），未细化说服策略且缺乏跨话题验证，可能限制结论的普适性。

---

## 350. LLM-Based Exogenous World Modeling for Dynamic Scheduling

**arXiv ID:** 2610.29308 | [PDF](https://arxiv.org/pdf/2610.29308v1)

**作者:** Ni Zhang `[一作]` (Singapore Management University), Zhiguang Cao `[通讯]` (Singapore Management University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了AWMP框架，利用预训练LLM对未来作业到达进行低维信念预测，并通过后状态分解与短期情景滚动实现动态调度；

**💡 创新点**

创新点在于将作业到达视为外生不确定性，仅用LLM预测聚合到达强度信念，结合后状态分解与残差自校准、惊讶触发刷新以及元进化的规划策略，实现无需任务特定训练的高效预判调度；

**🔧 技术方法**

使用预训练大语言模型进行到达信念预测、后状态分解、短情景滚动、残差自校准、惊讶触发刷新和LLM驱动的元进化来生成规划策略；

**📊 数据集**

在动态JSSP和FJSP的四种到达分布（脉冲、后升、循环、阶跃）上构造的扩展Taillard、Brandimarte和Hurink实例；

**📈 对比分析**

与八种经典优先级调度规则和五种基于优化的在线调度器比较，AWMP在动态JSSP中总加权延迟比最佳PDR低5.8%–20.7%，在动态FJSP中低21.9%–46.3%，并且在运行时间上比大多数先验方法更快；

**⚠️ 局限性**

局限在于仅预测到达强度的两段信念，缺乏细粒度作业属性信息；实验使用合成到达分布，缺少真实工厂数据；LLM推理仍受延迟影响，尽管已通过刷新与缓存显著降低调用量。

---

## 351. Solving Go under Positional Superko: Sound Proof Transfer and New Board Values

**arXiv ID:** 2610.29414 | [PDF](https://arxiv.org/pdf/2610.29414v1)

**作者:** Sami Kharma `[一作]` (Zuse Institute Berlin), Florian Schintke `[通讯]` (Zuse Institute Berlin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个名为 Sanko 的并行深度优先证明搜索器，解决了使用位置超ko（PSK）规则的小型 Go 棋盘（最多 30 点）的求解问题，并给出了 23 块棋盘的理论值。

**💡 创新点**

创新点包括：① 基于 PSK 的证明转移定理，仅需检查赢家创建的棋盘；② 纯眼分析，为 PSK 环境提供可靠的静态知识；③ 仅对证明侧进行剪枝，保持完整性；④ 结合学习先验、Bloom 过滤、无锁共享表等技术，实现大幅度加速。

**🔧 技术方法**

使用技术：并行深度优先证明搜索（Proof‑Number Search）、证明转移与路径摘要、纯眼静态分析、学习的先验（logistic 估计）、Zobrist 键与 Bloom 过滤、无锁共享表、虚拟失效、锁自由多线程、以及多种哈希安全检查。

**📊 数据集**

数据集：空棋盘小型棋盘（4–30 点，包含直线、矩形等），以及作者自行生成的基准位置集合。对每个棋盘都提供了完整的搜索记录与参数设置。

**📈 对比分析**

比较方法：与 EWS、MIGOS 等前沿求解器在同一棋盘上对比。Sanko 在 55 棋盘 16 分钟内求解，远快于 EWS 的 5.25 小时；在 44 棋盘 0.23 秒内完成，远快于 EWS 的 13.6 秒；在 192 线程下，55 棋盘仅 1.3–1.6 分钟。证明大小、节点数、扩展量等指标显示 Sanko 的性能提升在 20–40 倍范围。

**⚠️ 局限性**

局限性：① 未提供可外部验证的证明证书（PSK 证明需路径信息，仍是开放问题）；② 对更大棋盘（>30 点）仍难以突破，证明规模增长急剧；③ 仅在空棋盘上测试，未验证对更一般起始布局的适用性；④ 仍需改进知识与先验模型，以进一步压缩搜索空间。

---

## 352. PerMIT: Securing Persistent Memory for LLM Agents via Machine-Checked Integrity Theorems and Scalable Community Trust

**arXiv ID:** 2610.28987 | [PDF](https://arxiv.org/pdf/2610.28987v1)

**作者:** Tanqiu Jiang `[一作]` (Stony Brook University), Sandesh Swamy `[通讯]` (AWS AI Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了PerMIT框架，提供可证明的内存完整性保证，支持安全的持久内存共享和自治知识库构建。

**💡 创新点**

通过有限内存接口与端到端可审计的确认门，结合社区共识的信任累积，实现了在多代理环境中同时提升安全性与实用性的“共缩放”机制。

**🔧 技术方法**

采用了信任边界设计、可执行的有限内存操作、基于Lean 4的机器可验证证明、强化学习式的LLM验证器，以及分布式共识协议。

**📊 数据集**

在Trojan Hippo对抗测试、τ²‑bench电信任务以及多模型基准任务中评估。

**📈 对比分析**

与现有防注入框架（A‑MemGuard、LLM‑Auditor等）对比，PerMIT在保持或提升大多数性能指标（AM/HM）同时实现零攻击成功率；在自治知识库构建实验中，社区构建的知识库显著提升未见任务的完成率。

**⚠️ 局限性**

仍受可信决策者准确率、假负率以及稀有攻击事件的假设限制，且对知识质量与冗余管理缺乏完善机制。

---

## 353. DualHeads: Reward-tilted few-step diffusion made simple

**arXiv ID:** 2610.28858 | [PDF](https://arxiv.org/pdf/2610.28858v1)

**作者:** Yuanzhi Zhu `[一作]` (Ecole Polytechnique), Vicky Kalogeiton `[通讯]` (Ecole Polytechnique)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过在冻结的扩散模型骨干上添加两个轻量级二分类器头，利用参考样本和黑盒奖励对四步扩散生成器进行微调。

**💡 创新点**

将KL正则化的奖励最大化转化为两个二分类任务，无需梯度奖励、轨迹信息或教师模型；仅用样本即可实现奖励倾斜。

**🔧 技术方法**

基于扩散蒸馏、LoRA微调、二分类器头、soft标签的优势学习和逆KL解释。

**📊 数据集**

使用SD3.5-M和Z-Image-Turbo基础模型，在OCR、GenEval、Pick-a-Pic、DrawBench等数据集上训练并评估。

**📈 对比分析**

与多步RL、TDM-R1、REST等方法对比，四步模型在OCR/GenEval奖励上达到0.92/0.90，接近40步RL且保持图像质量；在Z-Image上HPSv3提升至13.93。

**⚠️ 局限性**

对参考池风格高度敏感，奖励头需要预热以加速训练；对极端奖励或不一致的参考样本可能导致样式偏移或奖励作弊。

---

## 354. Hidden Messages Between Diffusion Language Model Agents: Causal Evidence for Latent Communication

**arXiv ID:** 2610.28905 | [PDF](https://arxiv.org/pdf/2610.28905v1)

**作者:** Alan-Barsag Gazzaev `[一作]` (ITMO University), Ilya Makarov `[通讯]` (HSE University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本论文提出了一个基于连续隐藏状态的监督式通信通道，允许独立的扩散语言模型（Diffusion LM）在不共享文本上下文或画布的前提下，向接收者发送压缩后的签名密集向量，实现隐藏式信息传递；并通过替换与干预实验验证消息内容的因果效应。

**💡 创新点**

创新点在于：①首次将连续隐藏状态用于多代理扩散模型间的通信；②引入匹配‑扰乱替换的因果干预方法，验证消息是否携带示例特定信息；③发现通信时机（一次性延迟传输）比多轮交换更重要，提出时间协调原则。

**🔧 技术方法**

技术核心包括：①低维度（4×128）签名密集向量的编码器-解码器适配器；②对 Tiny-A2D、Dream-7B 与 QASC 三个任务分别训练的监督式通道；③冻结主干网络，只训练通道模块；④采用对比实验（无消息、匹配消息、扰乱消息、延迟消息）来评估通信效果。

**📊 数据集**

使用的数据集与任务包括：①Tiny-A2D（HM‑Synth 生成链式事实的问答），②Dream‑7B（依赖跨越任务，需双向消息传递），③QASC（科学事实推理问题，事实被分配到两位代理）。

**📈 对比分析**

对比方法：在每个任务中，先在无消息的基线下测量准确率，再分别引入匹配消息、扰乱消息、以及仅一次延迟传输的消息。结果显示：Tiny‑A2D 中匹配消息将准确率从 8.6% 提升至 72.9%（+63.6pp）；Dream‑7B 依赖跨越任务中匹配消息使准确率升至 100%（比无消息高 90.2pp），扰乱消息降至 0%；QASC 中匹配消息提升至 80.8%（+8.2pp），而单次延迟传输可达 87.8%，超过四轮循环传输 7.0pp。

**⚠️ 局限性**

局限性包括：①通信通道仅为监督式、任务专用，无法直接跨任务迁移或实现零射门；②未形成可读语言或通用协议；③密集向量虽不可读但可能泄露敏感信息；④实验仅在三种任务和两种扩散模型规模下验证，缺乏对更大模型或更复杂生成任务的评估。

---

## 355. CALLIOPE: Private Insights into AI Use, Without Embeddings

**arXiv ID:** 2610.28797 | [PDF](https://arxiv.org/pdf/2610.28797v1)

**作者:** Edith Cohen `[一作]` (Google Research), Chiyuan Zhang `[通讯]` (Google Deepmind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Calliope的顶层框架，使用LLM在文本域完成文档抽象、主题合成和分配，以构建层级化的主题树。

**💡 创新点**

创新点在于完全放弃向量嵌入和几何聚类，改为利用LLM直接在离散文本空间中完成所有操作，并在合成阶段通过DP分区选择实现正式差分隐私，避免了传统方法的隐私与效能的激烈折衷。

**🔧 技术方法**

核心技术包括文档摘要与关键词抽取、基于关键词频率的主题生成（LLM prompt）、文档到主题的分类（LLM prompt）以及DP分区选择（隐私安全的关键词筛选）。

**📊 数据集**

实验使用六个公开对话/文本数据集，包括多语言意图识别数据集、英语问答与银行客服、推特情绪、电影评论和金融新闻等。

**📈 对比分析**

与基于嵌入的k‑means及其DP版本的基线进行比较，在Intrinsic、Extrinsic、Semantic三类指标上，Calliope在高隐私预算下保持更高的Guess‑Cluster和Guess‑Odd‑One分数，且在NMI、ARI等指标上表现优于或持平。

**⚠️ 局限性**

局限性包括：对大量文档需要多次LLM推理导致计算成本高；当前仅支持批量静态数据，缺乏在线更新机制；隐私保障仅为记录级别，未扩展到用户级；与生成式主题模型等方法未做对比。

---

## 356. From Augmentation to Synthetic Evidence: Evaluating Generated Chest Radiographs at Dataset Intake

**arXiv ID:** 2610.29152 | [PDF](https://arxiv.org/pdf/2610.29152v1)

**作者:** Shaikh Akib Shahriyar `[一作]` (Rochester Institute of Technology), Christopher Schwartz `[通讯]` (Rochester Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在将生成的胸部X光图像作为真实检查记录时，自动质量检查、来源检测与记录支持三种判定标准的独立性和有效性，并评估通过模型评分进行挑选是否能提高通过两项检查的概率。

**💡 创新点**

提出“合成证据”概念，阐明单凭图像质量和生成检测无法确认图像是否真实采集；通过实验证明挑选并不能显著提高同时通过质量检查和来源检测的成功率；并展示了基于源记录和使用目的的注册体系在验证采集声明中的必要性。

**🔧 技术方法**

使用 Qwen3.8‑27B 进行候选图像的质量评分，MedGemma 1.5‑4B 负责质量检查，DINOv3 ViT‑7B/16 与 MedGemma 视觉编码器提取特征用于训练 L2 正则化逻辑回归来源分类器；此外还对 JPEG 压缩、尺寸调整等处理条件进行阈值漂移与排名损失实验。

**📊 数据集**

主要数据集包括 NIH ChestX‑ray14（真实图像）、Cheff‑identified（未知生成器的 3000 张图像）和 SynthCheX（75,649 张已过滤的生成图像），以及 957 张 Open‑I 前平面图像用于跨数据集评估。

**📈 对比分析**

在 20 张候选图像的选择实验中，基于 Qwen 质量分数的挑选将 MedGemma 的通过率从 12% 提升到 19%（Cheff‑identified）和从 8% 提升到 17%（SynthCheX），但在同时通过质量与来源两项检查时，成功率依旧低于 3%（Cheff‑identified）且仅有 2–3%（SynthCheX）。来源分类器在 5% 误报率下几乎全部拒绝 Cheff‑identified，拒绝 93.5% 的 SynthCheX，且 4.8% 的 NIH 图像被误标；JPEG 压缩导致 AUROC 从 0.987 降至 0.841；校准阈值可降低误报但不恢复排名。整体性能表明仅凭像素级检查无法验证采集声明。

**⚠️ 局限性**

限制包括：生成器（Cheff、SynthCheX）条件与检查模型不独立；Qwen、MedGemma 与 MedForge 属同一家族，模型之间可能存在共线性；数据来源未完全独立，无法排除预训练或记忆问题；处理条件对阈值与排名均有影响；注册体系仅验证记录完整性，无法确认临床历史；实验未评估对临床决策的真实影响。

---

## 357. Simply Logical -- The First Three Decades

**arXiv ID:** 2610.28912 | [PDF](https://arxiv.org/pdf/2610.28912v1)

**作者:** Peter Flach `[一作]` (University of Bristol), Jan Wielemaker `[通讯]` (SWI-Prolog Solutions b.v.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文记录了《Simply Logical》一书从1994年纸质版到2026年全交互式在线版的演变过程，结合三位作者的教学与技术视角，构建了一个可复用的交互式Prolog教材框架。

**💡 创新点**

创新点在于将传统教科书内容与现代Web技术深度融合，推出基于SWISH、Jupyter Book及自定义插件的模块化内容生成系统，实现程序代码与查询的自动嵌入与交互。

**🔧 技术方法**

使用的技术包括SWI‑Prolog、Pengines、SWISH、WebAssembly、JavaScript、HTML/Markdown、Bootstrap、Jekyll、Sphinx、Quarto、GitHub等多种前后端与发布工具。

**📊 数据集**

数据集主要来自书本原始内容、Prolog程序示例、SWISH代码盒子以及公开的Prolog与ProbLog等语言的实例库。

**📈 对比分析**

文章通过对比服务器端沙箱（SWISH）与浏览器端执行（WASM/JS）在资源利用、交互性能与可扩展性上的差异，指出后者在不依赖服务器资源的情况下能提供更快的交互响应，但缺乏持久状态管理。

**⚠️ 局限性**

主要局限包括服务器端资源瓶颈与缺乏负载均衡、SWISH对安全性要求严格导致部分程序无法运行、缺少跨实现的标准嵌入接口、浏览器异步执行对传统单线程Prolog的兼容性挑战。

---

## 358. GRAFT: Restoring the Missing Sampling Interface in Continuous Diffusion Language Models

**arXiv ID:** 2610.29136 | [PDF](https://arxiv.org/pdf/2610.29136v1)

**作者:** Haoning Yang `[一作]` (University of Electronic Science and Technology of China), Lili Pan `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了连续扩散语言模型在生成过程中稀有词频的衰减现象，并提出一种无训练的 GRAFT 方法，在推理阶段识别并固定这些稀有词，从而恢复词频尾部的覆盖率。

**💡 创新点**

首次揭示连续扩散模型缺少中间采样接口导致稀有词被平均消除的根本原因，并设计了通过读取解码头、重启生成并固定词的方式来补偿这一缺失。

**🔧 技术方法**

使用 ELF、LangFlow 等连续扩散语言模型；在推理期间读取解码头、挑选概率高但未获胜的稀有词、重启生成并持久固定；配合词汇门控、子词拼接规则和反重复校正；评估指标包括词频尾部比例、非词率、JSD、Zipf 系数等。

**📊 数据集**

在 OpenWebText 数据集上进行实验，使用同分布的人类参考文本进行评估。

**📈 对比分析**

与原始采样、best‑of‑4、argmax 等基线对比；在 ELF-B/M/L 和 LangFlow 上，GRAFT 将稀有词比例从 0.42–0.77×提升至 0.82–1.10×人类率，非词率仅轻微上升，整体词频分布与人类更为接近；计算成本约为原始采样的 2.2 倍。

**⚠️ 局限性**

仍存在非词率未完全消除、方法仅在推理阶段、计算开销较大；未在更大模型、不同条件任务及更广泛的数据分布上验证；可能增加敏感或错误信息的生成，需要配合过滤机制。

---

## 359. COWBOY: A Scalable Sim-to-Real Framework for Learning Contextual Whole-Body Manipulation

**arXiv ID:** 2610.28948 | [PDF](https://arxiv.org/pdf/2610.28948v1)

**作者:** Jiahui Yang `[一作]` (Carnegie Mellon University), Ruslan Salakhutdinov `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了COWBOY框架，能够在数千个不同的场景和物体上通过仿真训练，学习移动操纵机器人对整体身体的自适应控制，并实现零‑shot sim‑to‑real 转移。

**💡 创新点**

创新点在于将全局自由空间到达与局部接触学习分离，利用可向量化的环境感知全身控制器（WBC）处理全局到达；再训练局部RL专家并在大规模全景下蒸馏为单一端到端点云条件变压器策略，首次实现基于主动视觉与LiDAR的端到端移动整体身体控制。

**🔧 技术方法**

使用技术包括：几何织物（fabrics）实现的向量化全身控制器、PPO训练的局部RL专家、教师-学生在线DAgger蒸馏、点云条件变压器网络、程序化生成的多场景和多对象数据。

**📊 数据集**

训练使用约8000+仿真场景和150个未见对象；测试在24个真实场景中使用100个未见对象和20个固定对象；未依赖公开数据集，而是自行生成多种桌面、抽屉、箱子、架子等环境和对象。

**📈 对比分析**

与端到端RL和ManipGen等基线对比，真实场景平均抓取成功率91%，模拟场景保持约94–97%；相较基线性能显著提升，碰撞率低，且能根据场景几何自动调整基座、摄像头视角、手臂轨迹和抓取方式。

**⚠️ 局限性**

局限性在于抓取策略仅由场景与物体几何决定，无法满足多功能或多姿态抓取需求；在高度遮挡或侧向抓取（如Shelf场景）时仍出现性能下降。

---

## 360. Block Diffusion for Longitudinal EHR: Accuracy, Efficiency, and Representation Trade-offs

**arXiv ID:** 2610.29533 | [PDF](https://arxiv.org/pdf/2610.29533v1)

**作者:** Ashvin Gupta `[一作]` (Imperial College London), Alessandra Russo `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对比自回归（AR）与块式掩码扩散（BD）模型在 MIMIC-IV 电子健康记录序列上的零样本临床预测效果

**💡 创新点**

证明块式扩散在保持大部分预测性能的同时显著提升推理速度，且发现生成误差主要来自迭代过程而非表示学习，提出可变长度块与更合适的扩散策略作为未来改进方向

**🔧 技术方法**

使用块式掩码扩散（BD）、GPT‑2 风格自回归模型、线性探针评估、不同的解码策略（置信度掩码、ReMDM 修订）

**📊 数据集**

MIMIC‑IV v2.2 ICU 相关记录，包含 ICU 住院、再入院、死亡等四个预测任务

**📈 对比分析**

在相同 tokenization、上下文长度、模型容量下进行零样本预测；BD 在 B=8、S=4 的配置下 AUROC 仅比 AR 低约 0.03，且实现 1.43× 的推理速度提升；AR 在三项任务中略优，BD 在医院死亡预测中相对优势更明显

**⚠️ 局限性**

解码策略有限，固定块大小限制了模型充分利用临床事件的自然分组；迭代预算虽能提升性能但收益递减，未能完全消除与 AR 的差距

---

## 361. CrashGym: Measuring the Decision Value of Experimentation for LLM Agents in Counterfactual Market Crashes

**arXiv ID:** 2610.29245 | [PDF](https://arxiv.org/pdf/2610.29245v1)

**作者:** Haocheng Lu `[一作]` (Columbia University), Minjun Zhu `[通讯]` (New York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了 CrashGym 模拟器，用实验室实验测量 LLM 代理在面临市场崩盘时做出介入决策的决策价值。

**💡 创新点**

创新点在于通过在独立目标复制中执行所有可行干预，获得确切的反事实回报度量，并通过可切换的放大机制检验模拟细节对决策的影响。

**🔧 技术方法**

采用了确定性 agent‑based limit‑order‑book simulator、HMAC‑SHA256 seed‑based 随机生成、事件源回放以及 LLM（ChatGPT）API 与工具接口来执行实验。

**📊 数据集**

使用了 24 个新随机生成的合成市场世界（参数化的冲击、杠杆、流动性、抛售等）以及同一构造下的 8 次重复实验。

**📈 对比分析**

方法上对比实验室访问、固定混合实验室、可演化混合实验室、无实验室（观察仅）及安慰剂，结果显示 Luna 在新世界中的平均 regret 降低 57%（Δ=-5.09），满足预设标准；其他部署表现出相似方向但效应不显著。

**⚠️ 局限性**

局限性在于仅验证单一合成崩盘构造和两种介入实现，未考虑实验成本、真实市场数据、不同介入实现或更复杂策略；实验效果对两种实验室细节差异不显著，置信区间因样本量有限而宽泛。

---

## 362. SICAF: Time-Varying Focus Bottleneck for Self-Supervised Event-Based Optical Flow with Spiking Neural Network

**arXiv ID:** 2610.29158 | [PDF](https://arxiv.org/pdf/2610.29158v1)

**作者:** Shuangming Yang `[一作]` (Tianjin University), Badong Chen `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种名为SICAF的自监督脉冲神经网络训练策略，用于事件光流估计。

**💡 创新点**

创新点在于将可学习的时间注意力与Rényi α‑熵时间敏感信息瓶颈相结合，形成“紧凑压缩+时变聚焦”机制，显著捕捉事件流的时间异质性。

**🔧 技术方法**

使用的技术包括直接训练的SNN、时间注意力机制、Rényi α‑熵估计、信息瓶颈约束，以及在噪声与对抗攻击下的鲁棒性评估。

**📊 数据集**

使用的数据集有MVSEC、ECD和HQF事件光流数据集。

**📈 对比分析**

与多种SNN与自监督光流基线对比，AEE降低至0.90/0.79，较SeLHIB提升约8%–13%，在白盒对抗攻击下提升13.49%，鲁棒性显著提升。

**⚠️ 局限性**

局限性包括仅在事件光流任务上验证，缺乏对更复杂多模态场景的评估；时间注意力与α熵参数仍需手工调优，未充分评估计算开销与能耗。

---

## 363. Second Thoughts: Exploring and Planning through Comparative Imagination in World Models

**arXiv ID:** 2610.28779 | [PDF](https://arxiv.org/pdf/2610.28779v1)

**作者:** Hieu Tran `[一作]` (University of Massachusetts Amherst), Hong Yu `[通讯]` (University of Massachusetts Amherst)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Sibling Disagreement in Imagination (SDI)，通过在同一潜在状态下多次重复执行演员（actor）动作序列来获得基准，利用该基准同时衡量想象中的行动差异（用于探索）和规划方案相对于演员的优势（用于决策是否执行规划）。

**💡 创新点**

创新点在于：①将演员的重复回放作为统一参考，无需额外的模型不确定性集合；②使用同一基准实现探索奖励与规划接受阈值的自适应耦合；③在训练阶段引导数据收集，评估阶段仅使用演员，保持评估简洁；④通过标准差归一化（IU）使优势阈值对局部波动鲁棒。

**🔧 技术方法**

技术细节包括：
- 递归潜在动态模型 + 奖励/终止头
- Actor–Critic 结构
- MPPI 在潜在空间中的行动搜索
- 对规划方案与演员固定序列分别进行 R_g 次新回放，计算平均回报差异与演员回放标准差
- 探索奖励 b = sib / s_π，随后加入第一步回报；
- 规划接受门：若 ĝ > κ·s_π 则执行规划的首步，否则按演员行动。

**📊 数据集**

使用的数据集为三大基准：Meta‑World（10 个稀疏奖励任务），Adroit（3 个密集奖励任务）以及 DM‑Control（5 个连续回报任务）。

**📈 对比分析**

与 DreamerV3、R2‑Dreamer、Plan2Explore、Curious Replay、TD‑MPC2 等基线对比，SDI 在每个基准中获得最高的任务平均分，且在 18 个任务中有 15 个任务的最终性能至少与最强基线持平或更优；在 Meta‑World 和 Adroit 上在所有任务上都达到或超过最强基线；在 DM‑Control 上在大多数任务上领先，整体均值最高。

**⚠️ 局限性**

局限性包括：
- 仍需额外规划计算成本，尤其在高维动作空间下昂贵；
- 依赖演员回放的标准差作为基准，若模型误差与演员相互耦合，可能误判优势；
- κ 阈值是经验调参，可能需要针对不同任务调整；
- 在某些任务（如 acrobot‑swingup、finger‑turn‑hard）仍落后于最强基线；
- 探索奖励只在初期启用，后期对学习的影响有限。

---

## 364. An Evidence-Driven Multi-Agent Compound AI System for Head and Neck Proton Radiotherapy Planning

**arXiv ID:** 2610.29010 | [PDF](https://arxiv.org/pdf/2610.29010v1)

**作者:** Qingqing Wang `[一作]` (University of California at San Diego), Chang Chang `[通讯]` (University of California at San Diego)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一个多智能体复合 AI 系统，将深度强化学习（PPO+L2O）自动生成质子 PBS 治疗计划与基于大语言模型（Qwen3-14B）对话式计划细化结合，实现医生指令驱动的计划优化。

**💡 创新点**

创新点包括①首次将 DRL 自动规划与多智能体 LLM 对话式细化整合；②设计了基于证据的参考计划调整流程；③通过 SFT+GRPO 对 Qwen3-14B 进行领域后训练，使其具备临床推理、指令遵循和可解释性。

**🔧 技术方法**

技术手段包括 PPO + L2O 逆优化、LoRA 方式的 SFT、Group Relative Policy Optimization (GRPO)、API 工具与内存模块、Monte Carlo 剂量计算、GPT-5.6 作为评判者。

**📊 数据集**

使用 21 例头颈癌（双侧/单侧）病例数据，每例生成 5 个 PPO 计划，选取 50 条指令-调整样本（42 训练 8 测试）用于后训练。

**📈 对比分析**

通过自然语言输出评估和量化计划改进（目标指标提升 vs 竞争指标衰减）比较两模型，post‑trained Qwen3-14B 在多种病例中实现更大目标改进、竞争目标下降更小，计划质量提升、规划时间与 token 成本下降。

**⚠️ 局限性**

局限性：仅针对质子 PBS 角度固定；未实现自动束角选择和鲁棒优化；训练数据量有限；多目标冲突的全局最优尚未完全解决。

---

## 365. How to Use Parallel Data for Multilingual Pretraining

**arXiv ID:** 2610.28838 | [PDF](https://arxiv.org/pdf/2610.28838v1)

**作者:** Zhihong Huang `[一作]` (University of Macau), Min Yang `[通讯]` (SIAT)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大规模机器翻译并行语料进行两阶段预训练：先使用并行文档拼接为同一训练序列（PPT）构建跨语言上下文能力，再以标准独立抽样序列（SPT）适配单语输入，从而提升通用语言理解和翻译性能。

**💡 创新点**

创新点在于将并行文档拼接进同一训练序列，以充分利用跨语言上下文；随后通过 PPT→SPT 的两阶段顺序，解决单语任务与并行上下文不匹配的问题，显著提升跨语言理解与翻译效果。

**🔧 技术方法**

使用 Llama‑2 架构的 decoder‑only Transformer，采用自回归下一个 token 预测；对文档级并行拼接、单语抽样与两阶段预训练流程进行设计与实现。

**📊 数据集**

采用 TransWebEdu 机器翻译生成的英‑法、英‑斯瓦希里并行文档（约 1.5B tokens/语言），并使用 FLORES‑200、WMT14 等翻译数据以及 ARC、HellaSwag、TruthfulQA、PAWS‑X、XNLI、XWinograd、XStoryCloze、COPA/XCOPA、Global‑MMLU‑Lite 等多语言理解基准。

**📈 对比分析**

在相同 token 预算下与标准预训练（SPT）和单阶段并行预训练（PPT）对比；在高资源（英‑法）和低资源（英‑斯瓦希里）场景中，PPT→SPT 在平均语言理解上提升 1–4% 点，在翻译上 BLEU 提升 11–15 分；SPT→PPT 也有提升但幅度较小，整体两阶段方法显著优于单阶段。

**⚠️ 局限性**

限制：仅测试 1.3B 参数模型和两语对，未验证更大模型或更多语言；实验集中于机器翻译文本，未探讨人类翻译或非文本数据；两阶段训练增加训练复杂度与时间。

---

## 366. A Parallel Multi-Dialect Arabic Dependency Treebank

**arXiv ID:** 2610.28823 | [PDF](https://arxiv.org/pdf/2610.28823v1)

**作者:** Noor Abo Mokh `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Nizar Habash `[通讯]` (New York University Abu Dhabi)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了涵盖五大阿拉伯城市方言（贝鲁特、开罗、多哈、拉巴特、突尼斯）和现代标准阿拉伯语的并行依存树库，总计6,000句；对各方言与MSA的句法相似性进行系统分析；基准评测监督解析器与多语言LLM（Gemini、GPT、Llama、Fanar）的解析性能。

**💡 创新点**

①首个多方言并行依存树库，统一使用CATiB标注；②对五城市方言与MSA的全球及局部句法相似性进行量化比较；③通过检索增强的LLM提示，探究LLM在阿拉伯方言解析中的潜力。

**🔧 技术方法**

使用CATiB依存标注方案，CAMeLBERT-Mix编码+biaffine解析器；自动对齐采用Awesome-Align；LLM评测使用Gemini、GPT、Llama、Fanar的结构化输出提示；检索方式为基于chrF++相似度的10例支持。

**📊 数据集**

主要数据集包括MADAR-CODA的五方言句子、CamelTB的MSA句子，以及外部树库PATB、ARZTB、CamelTB-Gumar；新建的并行树库共5,000句（加1,000 MSA）。

**📈 对比分析**

在dev/test上以LAS为指标比较：监督模型在多方言混合训练下达到87.0%（all-dialects），全部树库联合训练可达87.5%；LLM在10-shot检索下，Gemini 87.3%，GPT 85.1%，Fanar 74.6%，Llama 72.3%；总体来看监督模型表现最优，LLM稍逊。

**⚠️ 局限性**

仅覆盖五城市方言且句子来源为旅行领域，无法充分代表自发语音、网络写作、代码混用等多样性；对齐仅限于token及其head，可能忽略词形变化和隐含成分；使用CATiB标注，难以直接与UD/SUD进行跨框架比较；评测基于金标分词和单一模型/提示，结果可能不具普适性。

---

## 367. Clinical radar heart sounds with echocardiographic phenotypes of ventricular dysfunction and valvular disease

**arXiv ID:** 2610.29267 | [PDF](https://arxiv.org/pdf/2610.29267v1)

**作者:** Jinye Li `[一作]` (Peking University), Qingchao Chen `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本研究收集并公开了162名心脏病患者的537条一分钟雷达心音（RHS）记录，并将每条记录与该患者最近一次超声心动图的结构与功能标签（左室射血分数<50%，壁运动异常，及至少中度瓣膜返流）进行配对；同时提供原始I/Q信号、处理后的RHS波形和质量分割信息，构建了第一份公开的雷达心音与超声心动图互补数据集。

**💡 创新点**

创新点在于首次将非接触式雷达心音与临床标准的超声心动图结构/功能表型进行配对，为研究心脏机械信号与结构性心脏病之间的关系提供了可复现、可扩展的数据资源，并验证了雷达心音能够识别左室射血分数降低。

**🔧 技术方法**

采用77 GHz FMCW雷达（Texas Instruments IWR1443 + DCA1000EVM）获取I/Q数据，经过范围FFT、峰值单元选择、相位解包、16–80 Hz 带通滤波、Savitzky–Golay平滑、Hilbert包络提取等步骤生成RHS；随后在机器学习端使用多实例学习（MIL）框架，对时域波形、包络和对数短时傅里叶变换（STFT）特征进行编码，并以焦点损失进行训练，五折患者层级交叉验证评估模型。

**📊 数据集**

数据集包含162名患者（年龄18–89岁，平均63.2岁），共计537条RHS记录；每个患者仅保留一份最新的超声心动图报告；表型分布为：LVEF<50% 38例，壁运动异常 43例，瓣膜返流中度以上 39例（三组标签可重叠）。数据以MATLAB文件形式提供：原始I/Q信号、处理后RHS和质量分割。

**📈 对比分析**

通过患者层级五折交叉验证评估左室射血分数降低（<50%）的判别能力，得到平均外部测试AUROC 0.770±0.138，PR‑AUC 0.559，宏平均F1 0.652，准确率 71.6%；表明雷达心音携带可被模型提取的与左室收缩功能相关的机械信息。

**⚠️ 局限性**

局限性包括样本量相对有限、单中心单设备收集，缺乏心电图等同步参考，雷达记录仅限一分钟且需人工质量审核，表型标签为非互斥且不涵盖全部心脏病变，模型仅针对射血分数下降的二分类任务，尚未验证在不同人群或不同雷达配置下的泛化性能。

---

## 368. RiO-DETRv2: Geometry-Aware Uncertainty Modeling Meets Real-Time Oriented Object Detection Transformer

**arXiv ID:** 2610.29159 | [PDF](https://arxiv.org/pdf/2610.29159v1)

**作者:** Zhangchi Hu `[一作]` (University of Science and Technology of China), Xiaoyan Sun `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为 RiO-DETRv2 的实时端到端旋转目标检测框架，通过几何感知的不确定性建模提升定向框定位精度。

**💡 创新点**

创新点包括：① 物体对齐的边界概率分布建模，消除全局轴对齐导致的几何失配；② 圆形到局部角度残差精化机制，实现从全局周期搜索到局部细化的顺序优化；③ 统一完整 OBB 质量对齐学习，联合优化分类、边界和角度分支，并利用预测分布的置信度校准分类结果。

**🔧 技术方法**

采用了基于 DETR 的 Transformer 结构，结合分布式回归（边界分布与角度分布）、多层渐进式细化、周期性角度编码、KL 散度转化为局部质量、Varifocal Loss 以及轻量化 MLP 进行置信度校准。

**📊 数据集**

在三个公开基准上评估：DOTA-1.0、DIOR-R 以及 AI‑TOD‑R，覆盖大尺寸、常规尺寸和超小尺寸旋转目标。

**📈 对比分析**

与 YOLO26‑OBB、RTMDet、PPYOLOE、O²‑RTDETR 等实时旋转检测器进行对比。RiO-DETRv2 在保持相似参数和 FLOPs 的前提下，mAP_50‑95 提升 1.2‑1.5 点，AP_75 也显著提高，推理速度在 NVIDIA T4 GPU 上约 30 ms，保持了实时性能。

**⚠️ 局限性**

局限性：在极高 IoU 阈值下仍存在边界与角度误差导致的召回‑精度权衡；对极小、密集目标的鲁棒性尚不充分；评价指标仍需更多阈值化、细粒度的度量来全面反映检测质量。

---

## 369. Do Coding Agents Understand Why Systems Crash---and Why They Don't?

**arXiv ID:** 2610.29332 | [PDF](https://arxiv.org/pdf/2610.29332v1)

**作者:** Chenxi Huang `[一作]` (Columbia University), Baishakhi Ray `[通讯]` (Columbia University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估编码代理在大规模系统代码库（如 Linux kernel 和 OSS‑Fuzz）中的 crash‑reasoning 能力，利用源代码干预（source interventions）构造执行验证的源树对，并基于此提出了 SysRB benchmark；

**💡 创新点**

①通过源干预对比而非单树评估，以消除偏差并真正测量执行推理；②构造了包含 934 对验证源树的 SysRB 数据集；③引入轨迹级别的 site‑profile（OΔ, DΔ, OC, DC）和 stack‑trace 相似度评分，用于细粒度诊断代理行为；

**🔧 技术方法**

使用四个主流大型语言模型（Claude Opus 5、GLM 5.3、DeepSeek V4 Flash、Gemini 3.8 Flash）进行多次轨迹采样；通过 Youden’s J 计算两源树间 crash‑discriminability；利用 LLM 轨迹标注器对 OΔ、DΔ、OC、DC 进行自动注释，并使用 deterministic stack‑trace 相似度评分；

**📊 数据集**

SysRB 数据集：由 545 个 Linux kernel（syzkaller）和 389 个 OSS‑Fuzz（ARVO）源树对组成，每对仅差一个 commit，已执行验证以确认不同 crash 结果；

**📈 对比分析**

采用 J（TPR‑FPR）衡量代理在 T_c 与 T_n 上的 crash‑prediction 差异，配合 site‑profile 评估观察与推理是否到位；实验显示 Claude Opus 5 在 Linux 上达 67.7% J，OSS‑Fuzz 上 75.4% J，DeepSeek V4 Flash 最差 12.3%；整体误报率低（2.8–10.9%），但漏报率高（27.5–76.8%），并且距离 crash site 的远近显著影响 discriminability；

**⚠️ 局限性**

局限性：①轨迹标注依赖人工校验，可能遗漏细节；②仅使用公开 commit，代理可能已在训练中见过，存在数据污染；③实验环境受限（无编译/运行），不完全模拟真实调试场景；④评估需要多次执行，成本高；⑤仅考虑单一 commit 干预，未覆盖更复杂的变更情形；

---

## 370. Generalization Bounds via Vector-Valued $L_p$ Correlation $V_α$

**arXiv ID:** 2610.29425 | [PDF](https://arxiv.org/pdf/2610.29425v1)

**作者:** Mohammad Hossein Momeni Hamaneh `[一作]` (Sharif University of Technology), Mohammad Hossein Yassaee `[通讯]` (Sharif University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `2704f255-0c84-4173-b83c-0e9a3dbea232` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并分析了一个可调参数α的V_α依赖度度量，给出对应的一般化误差上界，并在α=2时将其解释为基于输出密度方差的算法稳定性；

**💡 创新点**

创新点在于：①通过V_α框架实现TV与方差范式的平滑插值；②给出α=2的可解释密度方差表述并证明其对光滑重参数化不变；③引入自适应密度稳定性（ADS）条件，实现V_2的显式控制；④将框架扩展到条件、投影和二值化的超样本形式；

**🔧 技术方法**

使用信息论的Holder分解、变分法、Efron‑Stein不等式、超样本(CMI)构造以及重参数化不变性理论；

**📊 数据集**

在MNIST（0-1 与裁剪交叉熵）、CIFAR-10（预训练ResNet‑50）和贝叶斯线性回归三组数据上进行实验；

**📈 对比分析**

与传统的MI、TV、和PAC‑Bayes等基线对比，发现条件化、投影与二值化的V_α上界在实测选择-幽灵差距上接近或略低于1，说明在这些设置下上界紧凑；

**⚠️ 局限性**

局限性：基准V_α(S;W)在高维参数空间仍可能过大；ADS条件较强，仅适用于如Gibbs后验等随机化算法；在大规模网络和多类别任务上的评估有限，且估计高维V_α(U;W|S̃)仍具有统计困难。

---

## 371. AMP Atlas: A Browser-Only Workbench for Antimicrobial Peptide Exploration and Interactive Helical-Wheel Design

**arXiv ID:** 2610.29096 | [PDF](https://arxiv.org/pdf/2610.29096v1)

**作者:** Truong-Son Hy `[一作]` `[通讯]` (University of Alabama at Birmingham), Truong-Son Hy (University of Alabama at Birmingham)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一个基于浏览器的AMP Atlas工作台，整合了DRAMP数据库、SMILES注释、PDB结构、局部物理化学计算、逆向螺旋轮编辑和交互式分子可视化，支持无服务器的离线探索与设计

**💡 创新点**

创新点在于实现了完全静态部署、客户端本地搜索与计算、逆向螺旋轮约束搜索、以及在浏览器中即时检索与展示已发布AMP记录与结构的交互式工作流程

**🔧 技术方法**

使用了前端技术（HTML/CSS/JavaScript）、Web Worker、WebGL分子可视化、Python脚本进行离线数据预处理，数据库以JSON格式存储

**📊 数据集**

利用DRAMP 5.0官方通用记录和SMILES表格、以及从RCSB PDB下载的PDB链结构，构成12,784条源记录、10,579条规范记录、5,183条SMILES注释和433条PDB链结构

**📈 对比分析**

性能方面，前端通过Web Worker完成搜索和数值计算，静态分发文件仅138.8 MB，兼容GitHub Pages部署；无服务器部署降低网络延迟，但缺乏实时更新和大规模并发处理能力

**⚠️ 局限性**

局限性包括：不具备抗菌活性预测模型或毒性/最小抑菌浓度预测；未进行实验验证新设计肽；缺乏动态更新数据库与社区协作功能

---

## 372. Let Agents Design How They Search: Automatic Interface Optimization for Agentic Search

**arXiv ID:** 2610.29250 | [PDF](https://arxiv.org/pdf/2610.29250v1)

**作者:** Sangjune An `[一作]` (Korea University), Hae Beom Lee `[通讯]` (Korea University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种自动化搜索接口优化框架，基于语言模型代理执行轨迹和任务奖励联合改进语料库构件、可执行工具与使用指令。

**💡 创新点**

创新点在于将语料库预处理、工具实现与使用说明统一建模，并通过代理反馈循环自动搜索最优接口设计，而非手工固定设计。

**🔧 技术方法**

使用的技术包括基于Pi的语言模型代理、Claude Code生成器、Meta‑Harness启发的提议‑评估循环、BM25与稠密检索资源，以及GPT‑5.4‑nano作为主模型。

**📊 数据集**

实验数据集包括BrowseComp-Plus（问答）与BRIGHT（推理检索）四个子集（生物、地球科学、经济学、机器人学）。

**📈 对比分析**

与DCI、Agentic Search、RISE、DR‑DCI等基线对比，优化后接口在BCP的答案准确率提升至78.4%（+6.8pp）并在BRIGHT的平均nDCG@10提升至55.0（+9.3pp），同时工具执行时间下降约42%。

**⚠️ 局限性**

局限性在于仅针对固定语料库，未考虑文档增删变动对已生成的语料库构件与工具的维护与再优化需求。

---

## 373. Cache Locally, Recompute Selectively for Streaming Video Understanding

**arXiv ID:** 2610.29229 | [PDF](https://arxiv.org/pdf/2610.29229v1)

**作者:** Junhyeok Lee `[一作]` (Pohang University of Science and Technology), Jaeho Lee `[通讯]` (Pohang University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种无训练、可插拔的 FrameBlend 方法，利用查询时的选择性 KV 重计算解决流式视频理解中的上下文不匹配问题。

**💡 创新点**

创新点在于通过查询相关 Token 级别选择(QTS)实现查询感知的 KV 重计算，并结合短窗口预填策略，在保持或提升准确率的同时显著降低预填计算量。

**🔧 技术方法**

关键技术包括 KV 缓存重用、滑动窗口/压缩预填、查询相关 Token 选择(QTS)、重计算与解码以及短窗口预填。

**📊 数据集**

在五个流式和离线视频理解基准上评估：RVS、OVO-Bench、MLVU、Video-MME、EgoSchema。

**📈 对比分析**

与 ReKV、HERMES、LiveVLM 等三种主流 SVU 框架对比，FrameBlend 在所有框架-基准组合上均提升或匹配准确率（最高提升 3.9 pp），且在 RVS-Ego 上每帧预填 FLOPs 降低 20%+、总延迟下降约 15–60%。

**⚠️ 局限性**

局限性包括仅在 7B 规模模型上验证，固定预填窗口未自适应不同视频流，重计算会增加查询时延，需要进一步探索更小/大模型、自适应窗口和动态重计算预算。

---

## 374. Diffusion Posterior Sampling with Uncertain Forward Models

**arXiv ID:** 2610.29102 | [PDF](https://arxiv.org/pdf/2610.29102v1)

**作者:** Wentao Shangguan `[一作]` (Boston University), Vivek K Goyal `[通讯]` (Boston University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种不确定性感知似然（UAL），将前向模型的参数不确定性整合到扩散后验采样器的测量似然中，改进多种逆问题的重建与集成预测。

**💡 创新点**

创新点在于：①用Gaussian近似对参数边缘似然进行线性化，得到结构化测量协方差；②无需额外训练新的扩散模型即可直接在已有采样器的数据更新中插入；③提供理论误差上界和平均后验偏差分析。

**🔧 技术方法**

采用扩散先验模型、前向模型线性化、Gaussian近似、协方差分解与求逆、以及对采样器的梯度或分数更新进行插值。

**📊 数据集**

在六个逆问题上验证：高斯模糊与运动模糊（FFHQ图像）、计算机断层扫描（CT）、核磁共振成像（MRI）、超声（PICMUS对照与分辨率相机）、荧光显微镜（细胞图像）以及声学成像。

**📈 对比分析**

与传统点似然和等方差膨胀相比，UAL在所有任务中均提升了PSNR/SSIM、降低了CRPS、能量与局部变差分数；在运动模糊下与现有盲采样器相比，平均PSNR提高6–6.4 dB，甚至在某些设置下超过已公开的最优结果。

**⚠️ 局限性**

局限性包括：①对前向模型的线性化近似在高度非线性或大参数误差时可能失效；②需要先验的参数均值与协方差，若估计不准会影响性能；③计算成本随协方差秩增长，尤其在高维测量空间下仍可能显著提升时钟时间。

---

## 375. CompLit: Scientific Literature Search Benchmarks Must Cover Implicit, Cumulative, and Unmet Needs

**arXiv ID:** 2610.28658 | [PDF](https://arxiv.org/pdf/2610.28658v1)

**作者:** Dayoon Ko `[一作]` (Seoul National University), Kyungjae Lee `[通讯]` (University of Seoul)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 CompLit 基准，用来评估科研人员在隐式、累积、未满足的需求下进行文献检索的能力。

**💡 创新点**

创新点在于：①从研究者角度出发设计多维度诊断设置；②自动生成并人工验证包含内容、图像、元数据、比较约束的查询；③细粒度追踪检索过程（检索、阅读、回答、放弃）。

**🔧 技术方法**

使用的技术包括：大型语言模型（GPT‑5.6‑Sol、Claude‑Opus 4.8）生成查询与约束，视觉‑语言模型读取图表，检索器（BM25、SPECTER2、ReasonIR、Qwen3‑Embed‑8B）与多工具搜索平台（arXiv API、Serper、全文获取工具）配合；对话模拟器与评估脚本。

**📊 数据集**

使用的数据集：2.95M 论文的完整 arXiv S2ORC v2 版本（去除调查论文），以及 870 条单轮查询和 220 条对话（总计 1,090 条查询），覆盖八个 arXiv 领域。

**📈 对比分析**

与13个基线（检索器、语料库搜索代理、Web 代理）对比，单轮设置下前沿代理准确率 74–87%，但在隐式约束设置下平均下降约 20pp；在累积约束对话中，准确率进一步下降，且代理在未满足约束时往往误判近似匹配。

**⚠️ 局限性**

局限性包括：约束仅来自英文 arXiv 调查表，可能缺失领域特定属性；约束与查询虽自动生成但仍需人工核对；实验中使用的 Web 代理搜索实时网络，结果可能随时间变化。

---

## 376. Stochastic Pruning of the DPO Computation Graph: 1.4x Faster Training and Lower Peak Memory

**arXiv ID:** 2610.29538 | [PDF](https://arxiv.org/pdf/2610.29538v1)

**作者:** Yubing Wu `[一作]` (South China University Of Technology), Tiewen Pan `[通讯]` (Shenzhen Technology University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种统一框架，对 DPO 训练步骤进行位置、序列和分支层面的随机剪枝，以降低计算和内存开销。

**💡 创新点**

创新点在于将三类剪枝（位置、序列、分支）整合为可独立配置的采样机制，并给出逆概率加权的无偏梯度估计以及对优化目标的理论偏差分析。

**🔧 技术方法**

使用了 DPO 损失、逆概率加权、随机采样、FlashAttention、梯度检查点、LoRA、AdamW 等技术。

**📊 数据集**

在 Qwen3-4B/8B、Mistral-7B-Instruct-v0.3 上使用 HelpSteer3-Preference、ARC-Challenge、GSM8K、TruthfulQA 等多项评测数据集。

**📈 对比分析**

通过八配置因子实验和跨种子验证，将剪枝配置与完整 DPO 进行对比，实验显示 1.4× 的 step‑time 加速、约 30% 的训练时间缩短，但平均准确率略低，开放式生成分数波动。

**⚠️ 局限性**

局限性包括仅在单一 8‑加速器平台、特定批量尺寸和在线参考实现下验证，结果可能随硬件、批次几何和参考模型的不同而变化。

---

## 377. Who Verifies the Builder? Toward Independent Assurance for Agent-Produced Software

**arXiv ID:** 2610.28667 | [PDF](https://arxiv.org/pdf/2610.28667v1)

**作者:** Sion Israel Sion `[一作]` (École de technologie supérieure), Alain April `[通讯]` (École de technologie supérieure)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种独立保障框架，将代码构建与验证分离，定义验证合约（Verification Contract），并将验证分为“规范优先”与“实现感知”两个阶段，通过预注册实验验证其有效性。

**💡 创新点**

创新点包括：①在代理生成的软件中引入独立保障平面；②使用验证合约固定验证目标，防止自证；③构建两个互补的验证阶段，分别关注规范一致性和实现误差；④将验证结果汇总为可审计的保障报告，显式记录未验证部分。

**🔧 技术方法**

技术手段：大语言模型（LLM）用于构建与验证；基于变异测试、属性基础测试、模糊测试、变形测试的自动验证；通过验证合约定义保证与所需证据；使用对抗性循环（counterexample loop）提升鲁棒性。

**📊 数据集**

数据集与任务：HumanEval+、MBPP+、EvalPlus 任务（20个功能级任务）作为预实验；三个 Python 状态服务（Documents & payments、Room booking、Equipment loans），每个服务10条规范、若干隐藏/可见测试集，分别注入14/17/15个有缺陷，构成实验基准。

**📈 对比分析**

比较方法：测量可见/隐藏通过率、评估缺口、误报率、变异检测得分、代币消耗。实验结果显示，具有上下文独立性的验证器比构建器自身测试检测更多缺陷（p=0.006），而模型独立性未显著提升。规范优先阶段贡献最大，实施感知阶段补充少量缺陷。原始证据时常误判正确程序，但过滤无效测试后误判率显著下降。每单元平均代币成本约 67k–126k，时间与传统方法相当。

**⚠️ 局限性**

局限性：①构建器未产生真实缺陷，评估仅基于人工注入的缺陷；②隐藏测试可能不完整或超出规范，评估缺口有限；③仅使用单一构建器模型，难以推广到多种 LLM；④模型独立性实验受限于单一另一供应商，难以分离模型与供应商效能；⑤实验规模仅为小型 Python 服务，未覆盖大规模或多语言系统。

---

## 378. Artificial Intelligence and Job Design: A Dynamic Framework

**arXiv ID:** 2610.28830 | [PDF](https://arxiv.org/pdf/2610.28830v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 379. VerifierBench: Benchmarking the Generation of Executable Verifiers for LLM Agents

**arXiv ID:** 2610.28679 | [PDF](https://arxiv.org/pdf/2610.28679v1)

**作者:** Jianing Yin `[一作]` (University of Pennsylvania), Daoan Zhang `[通讯]` (University of Rochester)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了VerifierBench基准，评估代理在给定任务说明与环境下自动生成可执行验证器的能力，并对生成验证器的正负判别表现进行系统分析。

**💡 创新点**

创新点在于提出Verifier4Verifier筛选流程来保证测试标签与源验证器的一致性；创建了覆盖工具调用、状态、代码三类任务的222个任务及4,993个隐藏测试案例；并通过案例分析揭示大部分误拒来自过度限制的“all‑reject”验证器，指出自测往往仅确认有限的正样本而未挑战其它合法方案。

**🔧 技术方法**

使用了可执行验证器生成技术、六项V4V审核（重放、可重复性、来源一致性、输入标签一致性、基线检查、变异测试）、多模型、多抓手与不同推理努力的实验设置，结合对生成验证器的自测与评测流程。

**📊 数据集**

采用了来自六个公开代理基准（BFCL、ACEBench、τ^2‑bench、AppWorld、AgentDojo、SWE‑bench）的222个任务，共计4,993个隐藏测试案例，涵盖参考解、有效变体、普通错误与攻击样本。

**📈 对比分析**

在统一的Pi harness下评估19个模型，归一化平衡准确率从10.6%至62.1%不等，最高模型GPT‑6‑astra达62.1%；发现TPR普遍低于TNR，所有-reject验证器造成大部分误拒；不同抓手与推理努力对TPR/TNR与误拒率均有显著影响。

**⚠️ 局限性**

局限性包括：评价基于有限的构造测试集，V4V筛选后仍可能存在标签错误；不同抓手混淆了模型本身性能的比较；误拒率的普遍性未通过实验验证；案例分析主要针对GPT‑6‑astra，缺乏对其它模型的普遍性检验。

---

## 380. ART-MPG: Autoregressive R/T Manifold Path Guiding

**arXiv ID:** 2610.29049 | [PDF](https://arxiv.org/pdf/2610.29049v1)

**作者:** Zehao Zhao `[一作]` (University of Southern California), Xiyu Deng `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ART-MPG 神经模型，学习光线渲染中显式光路的链长、反射/折射序列及方向，并将其集成到 Manifold Path Guiding（MPG）框架中。

**💡 创新点**

创新点在于：① 用自回归扩散网络一次性生成完整的 R/T 序列；② 在重复试验中把链长和完整类型的概率精确剥离，显著降低逆估计方差；③ 使用四分量 von Mises–Fisher 混合模型预测方向，保持可解析、可采样性。

**🔧 技术方法**

采用的技术包括：自回归离散扩散模型生成 R/T 序列；von Mises–Fisher（vMF）混合方向预测；Manifold Path Guiding（MPG）的重复试验与逆估计；定理证明与蒙特卡罗光线追踪相结合。

**📊 数据集**

使用十四个 64×64 低分辨率基准场景（Living、Lamp、Plane、Slabs、Stone、Pool、Sphere、Water、Ring‑and‑Slab、Pillars、Slab‑×1、Slab‑×3、Pillar‑×3、Cornell Glass），每个场景都有固定 32,768 SPP 的参考渲染。

**📈 对比分析**

与 NPM、MPG、SMS、PPG、PT 在 {12,24,36,48,60} SPP 进行对比，采用 10 个固定随机种子评估全局 MSE。ART‑MPG 在 13/14 场景的 60 SPP 下获得最低 MSE，在 70 个场景‑SPP 细胞中取得 62/70 的中位数最小值，且大多数场景的误差曲线为非递增。

**⚠️ 局限性**

局限性：不做匹配时间比较，模型训练与重放成本未计入；实验仅在低分辨率图像和有限的 14 个场景上验证；缺乏对场景外推和更大规模光照场景的测试。

---

## 381. Mitigating Reward Hacking in Realistic Scenarios Through Representation Engineering

**arXiv ID:** 2610.29129 | [PDF](https://arxiv.org/pdf/2610.29129v1)

**作者:** Austin Meek `[一作]` (University of Delaware), Steven Basart `[通讯]` (Independent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并实现了一种基于隐空间监控与动态干预的奖励劫持检测与缓解方法，在大型语言模型的编程代理环境中实验验证。

**💡 创新点**

提出低成本线性探针监控+激活裁剪/加法动态干预组合，并在差分均值、优化、模型规范三种向量抽取方式上比较，证明其可推广至自然语言域。

**🔧 技术方法**

采用差分均值向量、梯度优化生成向量、模型规范向量三种抽取方式，结合激活裁剪、激活加法以及阈值控制等技术实现动态干预。

**📊 数据集**

使用 Qwen3.5‑27B 与 ImpossibleBench/LiveCodeBench 任务、MisTrace 代理记录集，以及 School of Reward Hacks 自然语言 OOD 评测集进行实验。

**📈 对比分析**

与模型规范提示、激活加法/裁剪等基线对照，在 ImpossibleBench 的不可行测试中将劫持率从 27% 降至 0‑3%，并保持或提升原始任务解决率；在 OOD 任务中同样显著降低劫持率。

**⚠️ 局限性**

实验仅在单一模型和有限任务上进行，样本量受限，使用 LLM 判断标签，且干预仅针对单一隐空间方向，未覆盖更细粒度子空间。

---

## 382. JaxMARL-BC: Multi-Agent Reinforcement Learning for Macroeconomic Modelling, in Seconds

**arXiv ID:** 2610.28907 | [PDF](https://arxiv.org/pdf/2610.28907v1)

**作者:** Daniele Montagnani `[一作]` (Università di Pavia), Aldo Glielmo `[通讯]` (Banca d'Italia)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发并发布了一个JAX加速的多智能体强化学习宏观经济模拟器JaxMARL-BC，使MARL-BC模型训练时间从数小时降至秒级，并支持多达20万学习主体的规模。

**💡 创新点**

通过JAX并行化和批处理实现了对MARL-BC的GPU加速，提供了极快的训练、可扩展性和开源仪表盘；验证了对DSGE极限的正确性，并完成了大规模异质性校准。

**🔧 技术方法**

使用JAX（just-in-time 编译、自动微分）、PPO/ SAC强化学习算法、参数共享的独立学习架构、GPU并行环境以及LLM驱动的配置仪表盘。

**📊 数据集**

采用标准Krussell–Smith和RBC模型的参数与技术进步马尔可夫链，实验中使用合成资本生产率分布（对数正态）和固定参数化。

**📈 对比分析**

与原始实现（CPU）对比，GPU版速度提升约75×–724×，CPU版提升约94×–179×；在20万主体下仍能在数十秒完成，显著低于传统方法所需的数十天。

**⚠️ 局限性**

仅针对已验证的RBC/Krusell–Smith框架，尚未在更复杂的HANK或新凯恩斯模型中测试；目前依赖PPO，尚未探索更高效的离线学习器；对真实经济数据的外部验证仍有限。

---

## 383. How Not to Be Censored: Inclusion Safety in Client-Led Consensus

**arXiv ID:** 2610.29497 | [PDF](https://arxiv.org/pdf/2610.29497v1)

**作者:** Yackolley Amoussou-Guenou `[一作]` (Université Paris-Panthéon-Assas), Andrea Canidio `[通讯]` (Category Labs)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计了一种无领导者共识协议，让客户端主动广播事务并通过引用前一 epoch 的区块参与投票，从而实现事务包含安全。

**💡 创新点**

创新点在于提出“包含安全”（inclusion safety）概念，发现存在两条路径（epoch 内即时包含与跨 epoch 传播），并证明客户端与复制机通信技术可以相互替代完成包含安全。

**🔧 技术方法**

主要技术包括投票证书与支持证词的构造、完全可排序的多块生成、同步/异步网络模型的通用化、随机延迟模型以及基于这些模型的定理证明和概率分析。

**📊 数据集**

未使用真实数据集，而是通过仿真随机延迟和攻击策略来验证协议在不同网络条件下的表现。

**📈 对比分析**

通过理论证明和仿真实验对比不同参数（A、Δ、p 等）下的包含安全概率，结果显示当 A≥2p̅ 时第一条路径始终成功；在此以下时第二条路径在较长 epoch 下概率迅速提升，整体性能随 epoch 长度增加而改善。

**⚠️ 局限性**

局限性包括：对客户端合规行为的依赖；在完全异步环境下无法保证包含安全；第二条路径仅提供概率性保证；模型假设区块不拥塞、投票不阻塞等理想化条件。

---

## 384. Phase-Based Onset Enhancement for Piano Transcription in Ensemble Settings

**arXiv ID:** 2610.29222 | [PDF](https://arxiv.org/pdf/2610.29222v1)

**作者:** Hyemi Kim `[一作]` (Electronics and Telecommunications Research Institute), Juhan Nam `[通讯]` (KAIST)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种将相位信息与幅度信息结合的钢琴转录模型，用以提升合奏环境下的钢琴转录效果。

**💡 创新点**

创新点在于利用STFT相位差增强起弦检测，首次将相位特征用于半CRF结构化转录。

**🔧 技术方法**

技术上采用Transkun半CRF框架，并在输入中加入幅度-相位差特征和相位增强的起弦得分。

**📊 数据集**

使用了MAESTRO v2中的钢琴三重奏子集和重新录制的PCD‑K钢琴协奏曲数据集。

**📈 对比分析**

与Transkun、Kong和YourMT3+等模型对比，Transkun-Phase在两数据集上均实现了10%~15% F1提升，尤其在起弦召回率上显著提升。

**⚠️ 局限性**

局限性包括对速度估计的提升有限、相位特征对混响敏感以及数据集规模仍有限。

---

## 385. StyleStream 2.0: Fast and Controllable Streaming Voice Style Conversion

**arXiv ID:** 2610.28940 | [PDF](https://arxiv.org/pdf/2610.28940v1)

**作者:** Yisi Liu `[一作]` (University of California Berkeley), Gopala K. Anumanchipalli `[通讯]` (University of California Berkeley)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了StyleStream 2.0，改进的流式语音风格转换系统，兼顾低延迟和可控文本指令。

**💡 创新点**

将少步像素MeanFlow与自回归微调结合以降低延迟，并引入统一的文本条件风格嵌入控制器，实现文本驱动的声音设计与单属性编辑。

**🔧 技术方法**

使用Pixel MeanFlow、DiT、Perceiver‑Resampler、Flan‑T5文本编码器，以及自回归Fine‑Tuning和CFG蒸馏等技术。

**📊 数据集**

训练数据包括Emilia（约50k小时英语语料）、CapSpeech、CosyVoice、Vevo等；测试集包含ESD、GLOBE、LibriTTS、RAVDESS、L2‑ARCTIC等。

**📈 对比分析**

与CosyVoice、Vevo、ASR+TTS级联等基线对比，StyleStream 2.0在目标风格相似度（S‑SIM、A‑SIM、E‑SIM）上名列第一，延迟降至≈520 ms，且在可读性与非编辑属性保留上保持竞争力。

**⚠️ 局限性**

主要局限是对极端音频质量或长段连续生成的鲁棒性仍有限，且文本控制仍依赖预定义属性（年龄、性别、口音、情感）且对细粒度控制尚不足。

---

## 386. Generative Modeling via Nonlinear Schrödinger Bridges and Branching Diffusions

**arXiv ID:** 2610.29476 | [PDF](https://arxiv.org/pdf/2610.29476v1)

**作者:** Mathias Grau `[一作]` (Qube Research & Technologies), Pierre Henry-Labordère `[通讯]` (Qube Research & Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于前后向偏微分方程的非线性斯特林桥模型SBNL，并通过分支扩散实现无偏估计；

**💡 创新点**

将斯特林桥与均值场游戏相结合，构建了包含法则相关运行和终端成本的半线性模型，并给出了对应的耦合PDE和非线性Sinkhorn系统；

**🔧 技术方法**

使用分支扩散（标记分支布朗运动）对耦合PDE进行概率表示，配合最大均值散度（MMD）优化终端约束，并利用自动微分在采样时估计梯度；

**📊 数据集**

在二维低维示例、50维高维混合高斯、MNIST与FFHQ人脸的潜在空间上进行实验；

**📈 对比分析**

与多种线性斯特林桥方法（DeepGSB、LightSB、LightSBB-M、DSBM、BranchSBM等）对比，在混合高斯任务中SBNL实现最佳混合Wasserstein距离，且在高维任务中保持模式覆盖；

**⚠️ 局限性**

需要手动调节分支率、核带宽和终端惩罚等超参数，分支模拟的方差仍高，且在纯线性情形下与成熟的扩散桥方法性能相当，尚缺乏系统优势。

---

## 387. Single-Image Food Mass Estimators Do Not Transfer Their Gram Scale Across Datasets

**arXiv ID:** 2610.29025 | [PDF](https://arxiv.org/pdf/2610.29025v1)

**作者:** Yongsheng Lian `[一作]` `[通讯]` (University of Louisville), Yongsheng Lian (University of Louisville)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估单张图像食物质量估计模型在不同数据集上的迁移性能，并与零样本视觉‑语言模型进行对比。

**💡 创新点**

首次系统性衡量模型的排序能力与 gram 规模，并证明仅在同一拍摄环境下的标注照片能恢复精度；同时比较了监督预训练模型与 VLM 在外部数据集上的表现。

**🔧 技术方法**

细调 DINOv2-base（及 DINOv2-large）进行回归，使用零样本 VLM Qwen3‑VL‑32B 进行多模态推理；采用 MAE、Spearman 相关、scale ratio、预测区间等指标，并通过 5‑fold CV 与单张校准进行评估。

**📊 数据集**

使用 Nutrition5k（训练/验证/内部测试）、ASA24（标准份量图片）、ECUSTFD（带硬币的全食物图）和 Food Portion Benchmark（FPB）等公开数据集。

**📈 对比分析**

在 Nutrition5k 验证集上 fine‑tuned DINOv2 MAE 为 24.8，远优于 Qwen 的 60.3；但在 ASA24、ECUSTFD、FPB 上，fine‑tuned DINOv2 的 MAE 分别为 67.9/44.7/467.5，Qwen 为 72.9/40.5/248.5，差距不大；同设训练可将 ASA24 MAE 降至 28.3。

**⚠️ 局限性**

外部数据集拍摄条件与标注方式差异大；VLM 未 fine‑tune；使用官方 split 做验证导致偏差；结果仅基于单一训练种子；缺乏学习曲线；仅为公开图像的回顾性评估。

---

## 388. Learning-Enhanced Observer for Sensorless Task-Null-Space Compliant Control

**arXiv ID:** 2610.28769 | [PDF](https://arxiv.org/pdf/2610.28769v1)

**作者:** Nuo Chen `[一作]` (Dalhousie University), Onur Toker `[通讯]` (Florida Polytechnic University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了基于神经网络增强的动量观测器与混合姿态-力控制器框架，实现了无传感器的外部力估计和在物理人机交互场景下的任务空间跟踪与零空间可变形控制。

**💡 创新点**

①将神经网络补偿加入动量观测器，并通过任务空间与零空间速度分解只将任务空间速度输入NN，显著降低人体干扰；②设计混合姿态-力控制器，将六维位姿跟踪、力跟踪、任务空间阻尼和零空间弹簧阻尼统一；③实现了无需力/扭矩传感器即可完成pHRI控制。

**🔧 技术方法**

使用动量观测器、神经网络补偿、任务空间与零空间投影、非奇异终端滑模控制、向量正弦/比例-微分力控制、虚拟弹簧阻尼以及Lyapunov稳定性分析。

**📊 数据集**

使用355k条离线采样数据，涵盖0~4 N不同施加力与随机人体干扰，用于训练和验证神经网络。

**📈 对比分析**

与传统动量观测器和使用完整关节速度的NN观测器比较，RMSE和最大误差分别下降至0.729 N和1.892 N；整体RMSE下降24.1%，最大误差下降38%；在人体干扰期间RMSE提升超过60%；控制器在两条参考轨迹上的位姿跟踪误差均低于1.5 cm、0.05 rad，力跟踪误差低于0.9 N。

**⚠️ 局限性**

依赖离线训练数据，缺乏在线自适应学习，实验轨迹与干扰范围有限，未在复杂非结构化pHRI任务中验证。

---

## 389. ReAPO: Retry-Aware Policy Optimization for Efficient Exploration in LLM Reasoning

**arXiv ID:** 2610.28770 | [PDF](https://arxiv.org/pdf/2610.28770v1)

**作者:** Nghi Dao `[一作]`, Adam Block `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的强化学习算法 ReAPO，用于在大型语言模型的推理任务中实现基于验证器的重试探索和尝试级别的信用分配。

**💡 创新点**

创新点在于把失败条件重试与经典SSP（随机最短路径）理论相结合，既实现了序列重试的依赖性采样，又在尝试层面给出理论上合理的信用分配，显著提升了探索效率和训练收敛速度。

**🔧 技术方法**

技术方法包括：基于GRPO的组策略优化、PPO损失、奖励标准化、序列重试（Retry-Conditioned Exploration）、试验级别优势估计（Attempt‑Level Credit Assignment）、K步重试上限、对齐上下文与重试提示的语言指令。

**📊 数据集**

使用的评估数据集包括：GSM‑8K、Polaris‑53K、GSM8K‑Platinum、BeyondAIME、AIME 2024/2025、HMMT 2025、MATH‑500、Minerva、GPQA‑Diamond，模型涵盖 Qwen3‑0.6B/1.7B/4B、OLMo‑2‑0425‑1B-SFT。

**📈 对比分析**

与基准方法（GRPO、双步GRPO、双组GRPO、SCoRe Stage‑II、MaxRL）在token、wall‑clock和Pass@k/Mean@32等指标上进行对比，ReAPO在0.6B模型上可用39%更少的token或31%更少的训练步达到GRPO的峰值，在更大模型和更大数据集上保持优势，且在token/时间归一化下显著优于所有对照组。

**⚠️ 局限性**

主要局限包括：受上下文长度限制只能设置较小的重试上限K（如K=2），未能在更大K或更长token预算下进一步验证；实验仅覆盖数学和科学推理，未探讨代码或多代理任务；对大规模多任务/多模型的进一步扩展仍待研究。

---

## 390. Shortcut-Resistant CAM Distillation for Long-Tailed Recognition

**arXiv ID:** 2610.29508 | [PDF](https://arxiv.org/pdf/2610.29508v1)

**作者:** Wenhai Wan `[一作]` (National Engineering Research Center for Big Data Technology and System), Songcan Chen `[通讯]` (MIIT Key Laboratory of Pattern Analysis and Machine Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种名为 Shortcut‑Resistant CAM Distillation (SRCD) 的框架，用于通过将头类的对象级解释知识迁移到尾类，降低长尾识别中的捷径学习，从而提升尾类性能。

**💡 创新点**

创新点：①在 CAM 空间进行解释级蒸馏，直接对模型关注区域进行约束；②构建能抵抗捷径的教师 CAM，采用基于能量的加权（空间连贯性 + 证据集中度）从头类候选中选取最佳解释；③提供理论证明，说明 L₂ 蒸馏可将尾类 CAM 的捷径注意力向教师收敛；④该方法可作为轻量级、插件式正则项，兼容任意长尾学习方法。

**🔧 技术方法**

使用技术包括：Class Activation Map (CAM)、基于 Laplacian 的空间连贯性能量、基于 Hoyer 稀疏度的证据集中度、能量加权聚合、L₂ 蒸馏损失、以及常见的长尾平衡损失（如 LDAM、CE+FCC 等）。

**📊 数据集**

实验数据集：CIFAR‑10‑LT、CIFAR‑100‑LT、ImageNet‑LT、iNaturalist 2018、Places‑LT。对每个数据集分别在从零开始训练和基础模型微调两种设置下评估。

**📈 对比分析**

与多种基线方法（BCL、GCL、FCC、GLMC、H2T、BNS、Focal‑SAM）以及基础模型微调基线（Decoder、PECEL、LIFT）对比，SRCD 在大多数场景下均实现显著提升，尤其在 Medium/Tail 类别上提升 1–4% 甚至更高，整体 Top‑1 精度提升 0.5–2% 级别；同时保持甚至略增 Head 类别精度。

**⚠️ 局限性**

局限性：①需要先训练平衡参考模型，若无平衡数据难以获得；②头类候选数目 K 的选取和能量加权对结果有一定影响，超参数需要调节；③方法主要针对分类任务，未对检测、分割等任务进行验证；④对极端长尾（IR>100）时的泛化能力尚未完全评估；⑤虽然理论分析表明捷径注意力会收敛，但对全局模型性能影响的深入分析仍待进一步研究。

---

## 391. RubricOpt: Self-Revision of Semantic Rubrics for Preference Optimization

**arXiv ID:** 2610.29491 | [PDF](https://arxiv.org/pdf/2610.29491v1)

**作者:** Wei Chen `[一作]` (South China University Of Technology), Zhou Wang `[通讯]` (University Of Waterloo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了RubricOpt框架，使模型在偏好优化过程中能够自行诊断并修改其训练用的行为准则（rubric）

**💡 创新点**

首次将可编辑的书面准则纳入偏好训练循环，让模型根据自身失败诊断实时修正训练目标，利用共享语义坐标和确定性门控实现安全编辑

**🔧 技术方法**

采用Direct Preference Optimization (DPO) 与固定训练对，结合八因子语义因子本体、LoRA适配器、语义共享坐标、确定性编辑门和匹配续延对照实验

**📊 数据集**

使用HelpSteer3和Tulu3两套偏好数据集；评测七项能力基准（GSM8K、MATH‑Hard、BBH、MuSR、ARC、TruthfulQA、MMLU‑Pro）及三项开放式生成协议（MT‑Bench、Arena‑Hard、WildBench）

**📈 对比分析**

与普通DPO控制以及匹配的keep/​refine续延对比，发现在部分模型和数据集上，rubric条件可提升BBH（≈+4.2pt）和部分生成指标，效果因任务和语料不同而异，特定编辑增量效果混合

**⚠️ 局限性**

局限于人工制定的八因子本体和关联需求度量；编辑效果受数据集对比分布影响；可能强化模型已有偏见；缺乏对分离任务的长期自我改进验证

---

## 392. Characterizing Social Media Behavior under Limited Observability: Retweet Link Prediction and Multimodal Intention Recognition

**arXiv ID:** 2610.29375 | [PDF](https://arxiv.org/pdf/2610.29375v1)

**作者:** Yifan Wang `[一作]` (South China University of Technology), Chenwei Wang `[通讯]` (Google LLC)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文分别研究了两项任务：在极度稀疏的转发网络中通过图神经网络进行边预测；在多模态社交媒体帖子中通过 LLM 生成用户意图描述。

**💡 创新点**

创新点包括：① 引入泄漏受控的转导评估协议，保证验证/测试边不参与消息传递；② 用极简的度-强度特征（compact topology）实现 GCN 在大规模稀疏图上的最佳表现；③ 对边标签和消息图同时进行联合腐败实验，区分排名与阈值决策两种鲁棒性；④ 在意图生成中采用分布感知采样（Pearson、余弦、KL），提升不同意图类别的性能均衡；⑤ 将 LoRA 微调与 MIKO 框架相结合，在有限样本上显著提升语义相似度指标。

**🔧 技术方法**

技术手段包括：图卷积网络（GCN）、GraphSAGE、GATv2、传统拓扑启发式；节点特征为度/强度对数归一化、随机 ID；边腐败实验；LoRA 微调 Llama‑3、Qwen‑2.5、InternLM；分布感知采样策略；评价指标为 AP、MCC、BLEU‑4、ROUGE‑L、BERTScore。

**📊 数据集**

数据集：① 2024 IEEE BigData 竞赛提供的转发网络（1.56M 用户、1.11M 边，无属性）；② 多模态意图数据集（979 条推文，7,868 条标注，10 种意图）。

**📈 对比分析**

与传统拓扑启发式（优先附件）和 GATv2 等模型对比，GCN+compact topology 在转导设置下 AP≈0.924，MCC≈0.705，明显优于 GraphSAGE、GATv2。联合腐败实验显示，随机 ID 会显著降低 AP 与 MCC；而 compact topology 在高腐败率下仍保持较好的排名性能。意图生成方面，LoRA 微调提升 BLEU、ROUGE，Pearson 采样使 BERTScore 在 10 种意图间的标准差从 0.0095 降至 0.0054，表明性能更均衡。

**⚠️ 局限性**

局限性包括：① 两个任务独立训练，缺乏结构‑内容联合建模；② 评估采用转导拆分，未验证在新节点/时间窗口下的泛化；③ 负样本采样仅基于未记录边，真实负样本可能存在；④ 固定阈值的 MCC 在高腐败下易失效，需更鲁棒的决策方法；⑤ 多模态意图数据规模有限，可能导致样本不均衡与过拟合。

---

## 393. PreCoMem: Predictive Cognitive Memory for Self-Evolving Long-Term Dialogue Agents

**arXiv ID:** 2610.29467 | [PDF](https://arxiv.org/pdf/2610.29467v1)

**作者:** Jian Zhong `[一作]` (Nankai University), Haining Zhang `[通讯]` (Nankai University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PreCoMem记忆整合框架，利用预测编码思想通过有效惊奇度（Effective Surprisal）来动态更新对话记忆。

**💡 创新点**

核心创新是将记忆视为带置信度的用户状态信念，结合可靠性加权的惊奇度阈值门控，将每轮对话映射到三种确定性更新（强化、暂存假设、软衰减），并通过“等待室”防止噪声过早提交。

**🔧 技术方法**

技术上融合了置信度加权检索、语义相似度、词汇重叠、主题相关性等检索信号，计算可靠性系数ρ_t；使用LLM（如gpt‑4.1）生成证据强度、矛盾严重度等标量，并通过闭式更新公式维护置信度。

**📊 数据集**

在LoCoMo、LongMemEvalS、PersonaMemv2以及自研的DiMoBench（含Drift实验）上进行评测。

**📈 对比分析**

与O‑Mem、MemoryOS、EverMemOS等基线相比，PreCoMem在所有基准上均实现了最优准确率（如LoCoMo +1.69、LongMemEvalS +3.23、PersonaMemv2 +3.37），并在长达1050轮的对话中保持内存有效量稳定。

**⚠️ 局限性**

局限性包括对LLM推理的依赖，需预先设计阈值且在多语言或多模态场景下尚未验证；存储仍随对话增长线性，虽然置信度压缩但仍需管理。

---

## 394. Derivation Depth Exposes Reasoning Failures in Tool-Augmented Music Analysis

**arXiv ID:** 2610.29060 | [PDF](https://arxiv.org/pdf/2610.29060v1)

**作者:** Rupali Rajesh `[一作]`, Derek Hoiem `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估音频‑语言模型在使用 MIR 工具时的推理能力，提出推理深度框架来衡量工具输出到答案的推理距离；

**💡 创新点**

首次揭示即便工具给出准确的音乐信息，模型也往往无法完成最基本的推理，甚至会无条件复制错误结论；

**🔧 技术方法**

对五个音频‑语言模型进行实验，使用机制性探测（logit lens、激活引导）和两次调用策略；

**📊 数据集**

使用合成 MIDI 渲染音频和 GuitarSet 真实录音，涵盖节奏、调性、结构等音乐属性；

**📈 对比分析**

在无音频和有音频两种条件下评估 pair 准确率：结论层输出近 100%，推理一步后准确率降至 57%–81%，更深层次更低；

**⚠️ 局限性**

缺乏可靠的工具输出推理能力，模型常复制错误结论，深度推理失败主要因无法将工具信息转化为答案，音频无法弥补此缺陷。

---

## 395. TeachBack: Learning to Teach via Bidirectional Preference Optimization for On-Policy Self-Distillation

**arXiv ID:** 2610.29167 | [PDF](https://arxiv.org/pdf/2610.29167v1)

**作者:** Linfei Ji `[一作]` (Huazhong University of Science and Technology), Ruixuan Li `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 TeachBack，一种双向对齐的自监督学习框架，在同一参数共享模型下使用特权信息生成教师轨迹并用学生行为反馈来调整教师生成。

**💡 创新点**

创新在于将特权信息作为可选工作笔记注入教师提示，构造可逆偏好学习双向目标，使教师既保持准确又保持对学生友好的推理路径。

**🔧 技术方法**

使用基于共享策略的对话生成、交叉评分、双向偏好优化、长度归一化的对数似然以及基准的 OPD、OPSD、PBSD 等对比。

**📊 数据集**

在 DeepMath、AMC 2023、AIME 2024/2025 以及 Knight-and-Knaves 等数学和逻辑推理数据集上训练与评测。

**📈 对比分析**

与 Base、GRPO、OPSD、SPIN、PBSD 等基线相比，TeachBack 在 Qwen3-1.7B/4B/8B 上的 mean@12 与 pass@1 均提升 2–4 分，整体表现最佳。

**⚠️ 局限性**

局限包括对教师侧反馈权重的敏感性、对较长提示依赖的影响以及仍需验证在更大模型和不同任务上的泛化能力。

---

## 396. The Reflexivity Threshold: A Phase Transition for Multi-Agent Performative Prediction

**arXiv ID:** 2610.29577 | [PDF](https://arxiv.org/pdf/2610.29577v1)

**作者:** Ahmed Mahrous `[一作]` (King Abdullah University of Science and Technology), Roberto Di Pietro `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6`

**🎯 论文内容**

研究了多智能体下的可执行预测（performative prediction）模型，提出了一种基于反射性矩阵的稳定性阈值，并给出了稳定收敛与失效的精确分界条件；

**💡 创新点**

创新点在于将单智能体的标量收敛条件推广到多智能体网络，通过构造反射性矩阵Γ记录交互强度，利用其谱半径ρ(Γ)确定稳定阈值，并证明ρ(Γ)=1既是上界又是下界；

**🔧 技术方法**

主要技术包括：强凸性与梯度敏感性假设、非负矩阵的Perron–Frobenius理论、谱半径分析、退化（rank‑one）构造的下界证明、以及对连续时间退化和收敛性的几何分析；

**📊 数据集**

本工作为理论性研究，未使用公开数据集，全部结果基于数学模型与构造；

**📈 对比分析**

由于缺乏实验数据，本文未与其它方法做实证对比；理论上在谱半径小于1时，收敛速度与误差均有严格上界；当谱半径大于1时，存在上界失效的最坏情况；

**⚠️ 局限性**

主要限制包括：假设每个代理的损失在自身参数上是强凸且光滑；需要精确执行最优回归；反射性矩阵被视为时间不变；对于非强凸或非光滑情形、信息不完备或动态网络结构，结果不适用。

---

## 397. BlazeVGGT: 13x Faster VGGT at 2,000 Frames via Camera Token without Training

**arXiv ID:** 2610.29454 | [PDF](https://arxiv.org/pdf/2610.29454v1)

**作者:** Shujun Yao `[一作]` (University of Chinese Academy of Sciences), Bonan Li `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了训练无关的 BlazeVGGT 框架，利用 VGGT 中 camera‑token 的演化关系和 DPP 进行帧级 KV 稀疏化，仅保留所有查询并显著加速长序列推理。

**💡 创新点**

创新点包括：①发现 VGGT camera‑token 由局部拓扑逐步过渡到全局几何；②利用这一演化构造视图相似性核并通过贪婪 DPP 选取互补视图；③实现 Q–KV 非对称稀疏，兼容现有加速器并无需重新训练。

**🔧 技术方法**

采用 DPP 贪婪选择、相机 token 归一化与相似性核、增量 Cholesky 更新、全局注意力视图级稀疏以及原始 VGGT 结构。

**📊 数据集**

使用的主要数据集有 ScanNet‑50、7 Scenes、NRGBD、Tanks & Temples；在 StreamVGGT 与 PAGE‑4D 的在线重建与视频深度估计任务中亦验证。

**📈 对比分析**

与 VGGT、FastVGGT、LiteVGGT、VGG‑T³、SparseVGGT 等方法对比；在 1,000–2,000 帧长序列上实现 13–14× 的速度提升，Chamfer Distance 从 0.41 降至 0.32，姿态误差保持竞争；与现有加速器组合后进一步加速。

**⚠️ 局限性**

局限性：在短序列或已压缩 KV 的模型组合时可能略逊于精度；仅基于相机 token 关系，极端视角分布下的视图选择可能不够鲁棒；线上实时推理需额外的内存管理与核参数自适应；未探究视图‑token 同时稀疏化的潜力。

---

## 398. Point Cloud Transmission With Patch-Based Rate Adaptation

**arXiv ID:** 2610.29479 | [PDF](https://arxiv.org/pdf/2610.29479v1)

**作者:** Shipeng Liu `[一作]` (Linköping University), Per-Erik Forssén `[通讯]` (Linköping University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种针对无线点云传输的联合源-信道编码（JSCC）框架 GeoContext-JSCC，采用局部补丁划分、可变速率分配和锚点上下文熵建模，实现对每个补丁的自适应符号长度编码。

**💡 创新点**

创新点在于：①使用锚点补丁提供空间上下文，改进了补丁间的熵估计；②基于熵引导的可变速率分配，使得不同补丁根据信息量分配不同的频道资源；③将模拟（analog）和数字（digital）双重传输结合，进一步提升鲁棒性。

**🔧 技术方法**

采用的技术包括：点云局部补丁提取（FPS+KNN）、Patch Latent 编码器（PointNet+Set-AB），Cross‑Point Transformer（CPT）用于上下文聚合，Gaussian 熵模型+条件概率预测，基于量化噪声的可微分量化，JSCC 编码器/解码器的复数向量映射，以及数字传输的块错误率优化。

**📊 数据集**

使用 ModelNet40 数据集，点云数目固定为 8192，补丁维度为 64，实验覆盖多种 CBR 和 SNR (0–10 dB)。

**📈 对比分析**

与 G‑PCC、SEPT、DPCC 及 Xie 等现有基线相比，GeoContext‑JSCC 在相同 CBR 下在 D1 与 D2 PSNR 上提升 0.5–2.7 dB，且在低 CBR 或低 SNR 情况下展现更平滑的性能下降，明显优于传统 SSCC 方法的阶跃/水平效应。

**⚠️ 局限性**

局限性包括：①模型需要预先设定锚点比例和补丁尺寸，对不同场景可能需要重新调参；②依赖于可变速率分配的离散速率集合，分配精度受限；③双模传输增加系统复杂度与硬件需求；④实验仅在 ModelNet40 上验证，真实工业点云分布与噪声模型仍需进一步验证。

---

## 399. Distance Field Driven Reinforcement Learning for Reactive and Robust Collision Avoidance

**arXiv ID:** 2610.29452 | [PDF](https://arxiv.org/pdf/2610.29452v1)

**作者:** Taiga Ikeshima `[一作]` (Tokyo University of Science), Eiichi Yoshida `[通讯]` (Tokyo University of Science)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种将欧几里得距离场（EDF）嵌入强化学习观测空间的框架，使机器人能在未知环境中实现鲁棒的碰撞规避与目标到达；

**💡 创新点**

创新点在于首次将实时EDF及其梯度作为观测与奖励引导信号，结合PPO算法，实现在从模拟到真实机器人的零样本迁移；

**🔧 技术方法**

采用了PPO深度强化学习、IDMP实时距离场生成、人工势场（APF）奖励引导、以及多种观测设置（EE‑only、全身）等技术；

**📊 数据集**

使用了在MuJoCo模拟环境中随机生成的10个球形障碍物与目标位置的数据集，并在真实机器人上通过Intel RealSense D455摄像头采集点云；

**📈 对比分析**

与基线RL、RL‑APF等方法对比，EDF‑RL‑APF‑FB策略在成功率、碰撞次数、到达时间等指标上均优于其他配置，验证了EDF在提升安全性和效率方面的有效性；

**⚠️ 局限性**

主要局限包括单摄像头视角导致的遮挡问题、对多视角/多传感器的依赖、未涵盖抓取任务及对更复杂动态环境的验证。

---

## 400. NeuralSRender: G-Buffer Guided Video Diffusion for Sim-to-Real Rendering

**arXiv ID:** 2610.29200 | [PDF](https://arxiv.org/pdf/2610.29200v1)

**作者:** Yipeng Lu `[一作]` (Wuhan University), Zhen Dong `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 NeuralSRender，一种基于 G‑buffer 的仿真到现实视频翻译框架，用于提升自动驾驶模拟视频的真实感并保持几何一致性。

**💡 创新点**

创新点在于：①两阶段训练策略，先用准确的合成 G‑buffer 学习几何/材质条件，再冻结这些模块后在真实视频上用 LoRA 进行外观适配；②轻量级投影器将稠密 G‑buffer 直接注入预训练视频扩散模型；③将参数增量控制在仅 1.45%，实现高效适配。

**🔧 技术方法**

技术手段包括：预训练视频扩散变换器 (DiT) + VAE、3D 卷积投影器、LoRA 模块、文本提示驱动的全局外观控制、两阶段 Flow‑Matching 训练、G‑buffer 条件编码。

**📊 数据集**

使用的数据集：GenReal（合成子集 City 与 CARLA、真实子集 DV，均带 G‑buffer 与文本提示）；在评估时还使用 nuImages（目标检测）与 Cityscapes（语义分割）作为下游任务的真实测试集。

**📈 对比分析**

与 DiffusionRenderer、TC‑Light、Cosmos 等基线在 GenReal-City/CARLA 上对比；指标包括 FVD、VLM‑BT、CLIP‑T、NDE。NeuralSRender 在 FVD 与 VLM‑BT 上均取得最佳分数，同时保持竞争性的 CLIP‑T 与 NDE。下游任务上，训练检测模型在 nuImages 的 AP_50 从 35.1 提升到 40.6，语义分割 mIoU 从 32.4 提升到 40.8。

**⚠️ 局限性**

局限性：对合成 G‑buffer 的准确性高度依赖；真实视频的 G‑buffer 通过逆渲染估计，噪声可能影响适配；目前仅针对驾驶场景，复杂多样的现实环境和更高层次的材质表达仍待改进；训练仍需数天 GPU 资源。

---

## 401. When Multiple Choices Are Correct: Asymmetric Probability Allocation in Jev and Other Decision Models

**arXiv ID:** 2610.29217 | [PDF](https://arxiv.org/pdf/2610.29217v1)

**作者:** Rushab Munot `[一作]` `[通讯]` (Independent Researcher), Rushab Munot (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对三种决策模型（Jev、Liquid d1、GPT‑6 Luna Decisions）在包含多项正确答案的任务中进行实验，分析其概率分配、阈值筛选、校准情况以及对相同答案位置的敏感性。

**💡 创新点**

首次系统评估决策模型在多正确选择场景下的概率不均衡性、相同答案复制的概率差异以及阈值策略的极限性能，揭示其不能可靠识别全部正确答案的缺陷。

**🔧 技术方法**

使用基准实验设计：对原始选择列表、复制相同答案的五重列表、阈值扫描以及将选择问题转化为一对一“是/否”问题，结合期望校准误差（ECE）、精确度、召回率和F1等指标进行评估。

**📊 数据集**

三大公开英文数据集：MetaTool（工具选择）、HotpotQA（检索式问答）和ECtHR（法律多标签分类），涵盖单/多正样本及无正样本情形。

**📈 对比分析**

比较方法：计算正确答案概率的最大-最小差距、所有正确答案在所有干扰项之上比例、低于均匀概率1/n的比例、阈值优化后的完全集合准确率。结果显示，最大概率差距为55–74个百分点，最优阈值下完全集合准确率仅在34–64%之间，低于模型正确排序比例（52–85%）。

**⚠️ 局限性**

局限性：只评估了三种基于API的决策模型，未涵盖文本生成式模型；仅使用英文数据集；阈值策略统一使用全数据集阈值，未尝试每例阈值或固定选择数；未探讨更广泛任务或其他选择规则。

---

## 402. Diffusion Sensitivity Geometry and Strong Duality for Optimal Privacy-Utility Tradeoffs

**arXiv ID:** 2610.28991 | [PDF](https://arxiv.org/pdf/2610.28991v1)

**作者:** Tue Do `[一作]` (University of Illinois at Urbana-Champaign), Daniel Alabi `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研究并给出释放理想扩散去噪器时的最优隐私-实用性权衡，提供精确敏感度分析与最优噪声协方差设计。

**💡 创新点**

通过两点数据集刻画最坏情况敏感度，构建单一量ρ*统一刻画所有隐私定义，并利用半正定规划与强对偶性实现最优噪声协方差；对二进制记录给出互信息隐私与容量关系。

**🔧 技术方法**

凸分析、半正定规划、对偶理论、Rényi / zCDP / GDP 隐私度量、I‑MMSE 关系、Mahalanobis 距离等。

**📊 数据集**

本研究为理论性工作，未使用具体真实数据集；评估采用模拟的凸包 [0,1]^d 与随机协方差矩阵的统计实验。

**📈 对比分析**

与传统基于直径的敏感度上界比较，平均提升 1.45 倍，尾部最高可提升 2.5 倍；通过仿真统计比值实现比较。

**⚠️ 局限性**

仅针对理想去噪器，非理想学习去噪器的分析尚未完成；适用于高斯机制，其他噪声分布尚未涵盖；对大规模数据集求解 SDP 的计算成本未给出。

---

## 403. PACE: Periodic and Channel-Aligned Encoding of Four-Bit Recurrent States

**arXiv ID:** 2610.29545 | [PDF](https://arxiv.org/pdf/2610.29545v1)

**作者:** Patrick Haller `[一作]` (Humboldt-University of Berlin), Alan Akbik `[通讯]` (Humboldt-University of Berlin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种后处理四位量化方案 PACE，专为线性注意力模型的递归状态设计，通过在解码时每隔若干步对状态进行一次量化并使用解码缓冲区保存中间键值。

**💡 创新点**

创新点包括：①周期性量化（dP>1）降低量化误差累积；②发现递归状态的“列”结构，采用按通道对齐的量化尺度组；③按头部误差保留情况分配四位与八位精度。

**🔧 技术方法**

技术手段涵盖：gated-delta 递归公式、周期性量化与解码缓冲、按通道对齐的尺度组、头级精度分配、Triton 实现的 4‑bit 解码内核。

**📊 数据集**

使用的数据集包括 GSM8K、MMLU‑Pro、IFEval、RULER‑32k、MATH‑500、WikiText‑103；模型为 Qwen3.5‑4B/9B/27B 与 Qwen3.6‑35B‑A3B 混合架构。

**📈 对比分析**

与全精度、DAMP INT8/INT4、Q‑Mamba DSQ 对比；在大模型上 PACE‑4/8 在约 ⅔ 内存使用下与 DAMP INT8 接近，Pace‑4 在相同内存下优于 Q‑Mamba DSQ；保持解码速度提升，并在大多数基准上获得与全精度相近的准确率。

**⚠️ 局限性**

局限性：仅在带有 gated‑delta 的混合模型上评估；未测试其他递归或混合注意力结构；周期参数范围有限（prefill 固定 256 token）；吞吐量仅在均匀 4‑bit 方案下测评；未探索更复杂的尺度/随机量化策略。

---

## 404. LocalAgent: Collaborative Agentic Verification for Fine-Grained Instance-Level Consistency

**arXiv ID:** 2610.29174 | [PDF](https://arxiv.org/pdf/2610.29174v1)

**作者:** Jiachen Guo `[一作]` (Tianjin University), Ajian Liu `[通讯]` (Chinese Academy Of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 LocalAgent 框架，利用多模态大语言模型（MLLM）进行全局语义感知并定位可疑区域，再通过专门的本地证据验证工具（LEV）对局部外观与几何一致性进行精细匹配，最终整合结构化标量证据完成实例级一致性验证。

**💡 创新点**

创新点在于：①将 MLLM 的全局语义处理与专门视觉验证模块分离，避免跨模态对细粒度信息的压缩；②设计结构化标量证据接口，让视觉匹配结果可被语言模型直接推理；③提出辅助限定强化微调（auxiliary‑capped RFT）与弱行为先验，使模型自学习何时调用验证工具，显著提升准确率且调用率低。

**🔧 技术方法**

主要技术包括：多模态大语言模型（如 Qwen3‑VL‑4B），视觉特征提取器 DINO 与 XFeat，局部外观一致性模块 LAC、局部几何对应模块 LGC，基于策略梯度的 RFT 训练以及工具调用的决策机制。

**📊 数据集**

使用的评测数据集包括 TWIN（工业产品、正负样本）、ILIAS、Landmarks、MET、CUB、Inquire 等共六个领域的 12k 条实例对，涵盖多种视觉变换与难负样本。

**📈 对比分析**

与 CoT‑RFT 基线相比，LocalAgent 在六个数据集上平均提高 4.9% 的准确率，且仅在 12.13% 样本中调用验证工具；在多模态与文本通用基准上保持或略超原始模型，显示了优秀的泛化与效率。

**⚠️ 局限性**

局限性包括：验证工具仅基于 MLLM 的定位结果，若定位失误将影响证据质量；未对多选查询的候选集进行跨候选证据聚合；在 TWIN‑Eval 上表现略低于基线；整体推理速度比直接推理慢约 30%。

---

## 405. Certified Neural Approximation of Robust Nonlinear Model Predictive Control

**arXiv ID:** 2610.29559 | [PDF](https://arxiv.org/pdf/2610.29559v1)

**作者:** Hendrik Alsmeier `[一作]` (Technical University of Darmstadt), Rolf Findeisen `[通讯]` (Technical University of Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于稳健MPC的神经网络控制器设计框架，能够在保证递归可行性和输入状态稳定性的前提下，实现对MPC反馈的近似。

**💡 创新点**

创新点在于将逼近误差视作受限扰动，在MPC设计中预先嵌入 Lipschitz 条件，给出数据覆盖与 Lipschitz 上界的确定性误差保证，避免事后验证。

**🔧 技术方法**

使用参数化优化、稳健MPC、Lipschitz 连续性分析和前馈神经网络（全连接、可微激活）。

**📊 数据集**

在化学反应器（连续搅拌反应器）和二维无人机垂直运动模型上生成网格训练数据。

**📈 对比分析**

与原始稳健MPC对比，在闭环仿真中误差始终低于预设上限，约 35 倍的在线计算速度提升。

**⚠️ 局限性**

局限性包括对全局 Lipschitz 上界的保守估计导致高维数据覆盖量指数增长，且仅适用于具有良好可微性与局部可行性区域的系统。

---

## 406. When Are We Near Nash? Certifying Solutions in General-Sum Games

**arXiv ID:** 2610.29300 | [PDF](https://arxiv.org/pdf/2610.29300v1)

**作者:** Boning Li `[一作]` (Tsinghua University), Longbo Huang `[通讯]` (Tsinghua University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种无最佳响应评估的在线NashGap证书，利用求解器迭代过程中维护的累计器，在两玩家和多玩家广义博弈中对加权平均策略的最优度进行评估并基于该证书选择最优迭代。

**💡 创新点**

创新点在于证明两玩家加权平均策略的NashGap可被精确拆解为外部遗憾加上签名价值校正，并将该分解推广到多玩家和交替更新的情形，构造出只依赖累计器且可在线评估的证书，从而消除传统轨迹最小值基准所隐含的最佳响应预算。

**🔧 技术方法**

所采用的技术包括基于Counterfactual Regret Minimization（CFR）及其变体（CFR+、RM+、PRM+、DPRM+、HS-PCFR+、ExRM+等）的离散化迭代更新，以及对终端分布、对手协同失衡和比较器项的在线累计与计算。

**📊 数据集**

实验数据集为两种经典扑克游戏：有偏的两玩家Leduc扑克（64个随机种子实例）和确定性的三玩家Leduc扑克，后者用于验证多玩家情况下的证书有效性。

**📈 对比分析**

通过与原NPCFR+轨迹最小值报告进行对比，本文显示在两玩家Leduc上使用16个训练无关求解器的证书选择能实现89%（训练集）、92%（保留集）和89%（第三集）在10⁻⁵阈值的解决率；在三玩家Leduc上，该证书比原报告的比较器低354倍，验证了其更严谨的最优度评估。

**⚠️ 局限性**

局限性包括：在某些低β（tie‑refund）实例上证书仍相对宽松；多玩家情形仍需一次额外的单步后向传递来补偿协同失衡；以及累计器的存储与计算成本在极大规模游戏中可能成为瓶颈。

---

## 407. Motion Equivariant Convolutional LSTM: When Motion Becomes a Gauge Symmetry

**arXiv ID:** 2610.28771 | [PDF](https://arxiv.org/pdf/2610.28771v1)

**作者:** Vahid Hosseinzadeh `[一作]` (South Dakota State University), Michael Puthawala `[通讯]` (South Dakota State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aaccfe5c-6b26-4208-b23c-35331481e142` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种运动等变卷积LSTM（MELSTM），通过引入传输器和门控机制，使循环网络能够在时间上随对象运动迁移隐藏状态，从而实现对任意时间变化运动的等变性。

**💡 创新点**

创新点在于将运动视为时间上的规范对称（gauge symmetry），并证明只要传输器满足规范连接的变换律，模型即可在所有运动下保持等变；同时结合槽位跟踪（slot tracking）实现局部运动等变性，并在不枚举所有速度的情况下估计速度。

**🔧 技术方法**

主要技术包括：1）运动等变卷积LSTM的门控设计；2）传输器（Transporter）与速度模型的规范连接关系；3）槽位跟踪的速度估计；4）对比实验中的动量和LSTM基础网络。

**📊 数据集**

实验数据集包括：Moving MNIST（多速率平移数字），Motion‑Only MNIST（噪声纹理上的移动数字），以及KTH行动识别数据集在模拟摄像机抖动下的版本。

**📈 对比分析**

与标准LSTM和基于速度枚举的流等变LSTM（FELSTM）在同一网络结构、参数量和训练协议下比较，MELSTM在所有任务上都表现最佳：预测MSE显著下降，分类准确率提升至≈98%，并且在更长的预测区间、速度分布改变以及摄像机抖动更大时均保持稳定，且训练速度约是FELSTM的6倍。

**⚠️ 局限性**

局限性包括：1）速度模型目前是无参数的相位相关估计，无法自适应更复杂的运动；2）对速度离散化的敏感度仍存在，若运动不落在预设网格上仍需额外的估计策略；3）目前仅验证了平移运动，对更一般的光滑变形（如差分流）尚未展开。

---

## 408. LEAN: Loss-Engineered Modulation of the Model Weight-Sensitivity Distribution for Efficient, Error-Tolerant Neural Networks

**arXiv ID:** 2610.28872 | [PDF](https://arxiv.org/pdf/2610.28872v1)

**作者:** Habibur Rahaman `[一作]` (University of Florida), Swarup Bhunia `[通讯]` (University of Florida)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 LEAN 的损失驱动正则化方法，在微调阶段通过控制权重梯度与存储格式步长的组合，系统性地重塑训练后模型的权重敏感度分布，将高敏感度权重迁移到低敏感度区，从而显著降低对错误修正码的需求并提升低精度存储效率。

**💡 创新点**

创新点在于：① 引入单一超参数 λ 控制的复合正则项，由“范围项”限制每层权重的存储步长，和“尾部项”抑制前 k 大梯度；② 通过双向反向传播实现对 Hessian 的近似，确保正则化精准作用于敏感权重；③ 在保持准确率误差≤0.5 点的前提下，将敏感尾部缩小 5–13 倍，显著提升错误耐受性和存储/能耗表现。

**🔧 技术方法**

使用技术包括：梯度/ Hessian 相关正则化（双向反向传播）、十轮微调或训练最后若干轮的损失项、BCH(127,43) 选择性纠错码、按敏感度排序的 8/6/4/3 位混合精度量化、Progressive Bit Search (PBS) 以及目标攻击（T-BFA），并进行真实位翻转实验与消融分析。

**📊 数据集**

实验数据集与模型：
- 视觉：ResNet‑18/ResNet‑50/VGG‑16 在 CIFAR‑10、CIFAR‑100、ImageNet；
- 语言：Qwen2‑0.5B、Llama‑3.2‑1B、Qwen2‑1.5B 在 WikiText‑2 上微调 1 轮。

**📈 对比分析**

比较方法：将 LEAN 与基准（无正则化微调）、四种鲁棒训练方法（Piecewise Clustering、WCAT、Forget‑and‑Rewire、Hessian‑aware）以及 LSQ、KURE 量化做对比。结果显示：
- 敏感尾部减少 5–13×，精度仅下降 ≤0.5 pts；
- 采用 1% BCH 码时，ResNet‑18 在 300–500 次自适应位翻转后仍保持 94% 以上准确率，传统防御在 200 次左右失效；
- 同一保证下，冗余比降低 3.7×；
- 8/6/4/3 位按敏感度排序的低精度格式在 ImageNet ResNet‑50 上实现 3.29 位/权重、2.4× 的权重量化压缩，且精度高于 LSQ 3 位量化；
- 语言模型在 10–20 次位翻转后残余性能提升，且尾部缩小 1.7–5.3×。

**⚠️ 局限性**

局限性：
- 需要额外 10 轮微调（≈1.5×训练成本），且低位格式还需再一次微调；
- λ 为单一超参数，难以精细控制不同敏感度区的迁移；
- 只针对存储格式的单一梯度权重，未覆盖基于类别的敏感度，导致对目标攻击的完整防御仍不足；
- 实验以白盒适配攻击为主，未评估黑盒或计划性攻击；
- 代码纠错假设为随机交织，若实现不理想，安全性可能下降；
- 主要验证 INT8 与低位量化，未涉及更大模型或不同硬件架构的通用性。

---

## 409. Multi-level Encoder with Global Topic for Task-oriented Dialogue Summarization

**arXiv ID:** 2610.29097 | [PDF](https://arxiv.org/pdf/2610.29097v1)

**作者:** Zhuoqi He `[一作]` (South China Agricultural University), Xinyang Lin `[通讯]` (South China Agricultural University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种多层编码器结合全局主题的模型 MLEGT，用于任务导向对话摘要，能够同时捕捉对话的时空依赖并生成更具信息量的摘要。

**💡 创新点**

创新点包括：① 构建关键词图与说话人图并加入全局节点，自动提取全局主题；② 通过双层跨注意力进一步挖掘对话内部长距离依赖；③ 设计全局主题引导的指针机制，帮助捕捉被忽略但重要的信息。

**🔧 技术方法**

采用 BART 预训练编码器；图构造（关键词/说话人+全局节点）；GCN 图卷积；双跨注意力；全局主题指针；双跨注意力解码器；交叉熵训练。

**📊 数据集**

使用两个公开中文数据集：LCSPIRT（大规模警务访谈对话）和 Medical DS（医学问诊对话）。

**📈 对比分析**

与 PGN、BART、T5‑pegasus、TDS+NTM、GPT‑4o、SDDS 等基线模型对比；在 LCSPIRT 上 Rouge‑1/2/L 均提升 3–5%；在 Medical DS 上亦实现最佳成绩；消融实验验证全局主题和指针机制对性能提升的关键作用。

**⚠️ 局限性**

局限性：模型复杂度较高，图构造与全局节点可能导致计算开销增加；目前仅在中文数据集验证，跨语言通用性和对极短/极长对话的鲁棒性尚待进一步研究。

---

## 410. Think-Capture: Active 3D Geometry Acquisition for Spatial Reasoning in Vision-Language Models

**arXiv ID:** 2610.29232 | [PDF](https://arxiv.org/pdf/2610.29232v1)

**作者:** Junjie Wang `[一作]` (FNii-Shenzhen), Zhen Li `[通讯]` (CUHK-Shenzhen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于查询的三维主动获取框架（Think-Capture），通过先识别任务关键物体并按需获取其几何信息，再将零碎的三维感知转化为紧凑的符号文本，让视觉语言模型（VLM）在推理时同时使用图像和三维状态；

**💡 创新点**

①将三维感知转化为任务相关的符号化文本，减少模态不匹配和上下文开销；②采用层次化的GRPO奖励机制训练VLM，学习何时使用三维证据、何时依赖视觉线索；③实现查询条件下的主动几何获取，显著提升空间推理的一致性与鲁棒性；

**🔧 技术方法**

三维工具调用（SAM3、π³）、符号化状态抽象（文本模板）、层次化GRPO（强化学习）训练、Qwen3‑VL‑4B等大型 VLM、以及零碎图像与三维信息的联合推理；

**📊 数据集**

MindCube‑Tiny、MMSI‑Bench（位置子集）和 ViewSpatial‑Bench 等跨视角空间推理基准；

**📈 对比分析**

与多种专有与开源 VLM（如 GPT‑4o、Gemini、InternVL 等）以及仅使用视觉或 2D 框框的基线比较；在 MindCube‑Tiny 上取得 54.4%（比基线提升 21.5 点），在 MMSI‑Bench 位置子集和 ViewSpatial‑Bench 分别提升 7.9% 与 5.2%；对工具扰动（坐标噪声、遮挡）具有更高的容错率；

**⚠️ 局限性**

依赖于外部三维感知工具的可观测性和标定；对遮挡严重或深度偏差大的目标效果有限；符号模板离散化可能不足以描述精细几何或动态环境；

---

## 411. Events, Not Pixels: Where World Models Get the Return Wrong

**arXiv ID:** 2610.29435 | [PDF](https://arxiv.org/pdf/2610.29435v1)

**作者:** Haoyue Liu `[一作]` (Chinese University of Hong Kong), Xiaoying Tang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种逐步替换诊断方法，通过用真实轨迹的对应转移替换模型产生的转移，量化不同转移对预期回报误差的影响，并发现回报误差主要集中在奖励或终止事件转移处；随后使用扩散模型修复缺失事件，并验证事件补丁能提升规划器的计划排序。

**💡 创新点**

创新点在于：①从转移级别而非整体模型级别定位回报误差源；②引入预算匹配的随机对照来衡量事件补丁的显著性；③通过预先记录的规则预测扩散模型在何种游戏中能有效修复误差；④在多种任务（Atari、控制任务、机器人仿真）上展示事件补丁对计划选择的普适性。

**🔧 技术方法**

使用的技术包括：冻结无记忆的奖励/终止检测器、转移替换（事件补丁、随机补丁、所有补丁）、误差归因指标（R(c,m)）、扩散模型（采样预测）与传统MSE回归模型的对比、plan ranking评估（pairwise accuracy/top‑1 regret）。

**📊 数据集**

主要数据集有：Atari‑100k（60个游戏，60步回合），DeepMind Control Suite（6个控制任务），iVideoGPT on robosuite（机器人仿真），MiniGrid 与 Crafter（多样化场景）。

**📈 对比分析**

与随机补丁、像素误差（MSE、SSIM、LPIPS）和所有真实补丁进行对照；在可靠的模型‑集合对中，事件补丁平均提升约 5–7% 的误差归约（median lift），随机补丁仅提升 1–2%；扩散模型在预先设定阈值下能在大部分游戏中进一步降低回报误差，且能显著提升计划排序准确率（pairwise accuracy 近乎 100%）。

**⚠️ 局限性**

局限性包括：诊断依赖于训练好的奖励/终止检测器，若检测器失效会导致误差定位不准；补丁策略仅在覆盖率较低的场景有效，极端稀疏奖励或高频事件可能需更多补丁；扩散模型在训练数据不足或动态不稳定的任务中修复效果有限；实验集中在已发布的像素世界模型，未覆盖更复杂的动态环境或更大时间步长。

---

## 412. LiveAct: Towards Real-Time Multi-Agent World Models with Heterogeneous Action Interfaces

**arXiv ID:** 2610.29408 | [PDF](https://arxiv.org/pdf/2610.29408v1)

**作者:** Xinyu Wang `[一作]` (Tsinghua University), Fangneng Zhan `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 LiveAct，一种针对多代理、异构控制空间的视频世界模型，能够实时生成共享视觉未来并保持每个代理的独立控制。

**💡 创新点**

创新点在于提出了 Heterogeneous Action Interface（HAI），通过 Agent‑Aware Spatial Modulation 把语言描述与视觉查询绑定，以及 Schema‑Guided Action Adaptation 把控制维度的语义与同步数值分离，解决了跨代理误绑定和控制语义崩塌的双重难题。

**🔧 技术方法**

使用 Transformer‑based  diffusion 生成器（DiT）+ VAE 编码、UMT5 文本编码、教师强制（Teacher Forcing）+  分布匹配蒸馏（DMD）实现因子化的因果生成；所有模块在保持预训练视觉先验的同时加入新的跨模态注意力。

**📊 数据集**

数据集为 17,951 条多代理轨迹（共 119.54 小时），涵盖四款竞技游戏和两套协作机器人系统，后续抽取 108 条测试剪辑作为 benchmark。

**📈 对比分析**

与 LingBot‑World 2.0、Incantation、MultiWorld、ActionParty 以及 WanToFight‑style 等基线比较，LiveAct 在游戏轨迹上取得 0.594 的 Act‑F1、0.898 的代理归属准确率、FVD 338.5；在机器人轨迹上得到 0.558 的 NRMSE、0.942 的归属准确率、FVD 270.2；实时生成速度提升至 17.70 FPS（512×448）并显著优于其他方法。

**⚠️ 局限性**

局限性包括：仍需在训练时依赖大规模预训练模型、对极端动态或高度重叠代理的识别精度有限、仅在 512×448 分辨率下验证，未对更高分辨率或更长序列的稳定性进行深入探讨。

---

## 413. BRIDGE: An Acceptance Gate to Block False Improvements in Model Updates, with a Retrospective Clinical Evaluation

**arXiv ID:** 2610.29515 | [PDF](https://arxiv.org/pdf/2610.29515v1)

**作者:** Dwi Sunaryono `[一作]` `[通讯]` (Institut Teknologi Sepuluh Nopember), Dwi Sunaryono (Institut Teknologi Sepuluh Nopember)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种称为BRIDGE的接受门，使用已标记数据对模型更新做出是否接受的决策，旨在减少在验证集上表现良好但对新数据无提升的假提升更新。

**💡 创新点**

创新点在于将交叉拟合、bootstrap下限与固定阈值相结合的门控机制，并通过预注册设计和有限样本上对假提升率的理论上限证明来实现对更新的无偏控制。

**🔧 技术方法**

技术手段包括对搜索集进行Bagging的梯度提升树、交叉折叠的候选选择、基于混淆表计数的度量差异、N_boot=20000次bootstrap抽样以及阈值ε=0.005的门控。

**📊 数据集**

实验使用了76个真实数据集（包括11个新数据集、12个临床任务、39个支持数据集及14个OpenML数据集）以及1,400个独立的合成任务，验证门的稳健性。

**📈 对比分析**

与传统的Naïve提升规则、McNemar检验、5×2 CV、校正t检验和保留检验相比，BRIDGE将接受更新中的假提升率从约19%–25%降至约1%–1.7%，并且在所有设置下都满足预注册的10%上限，平均收益比Naïve提升提高1.8–3.1倍。

**⚠️ 局限性**

局限性包括仅在表格数据和基于硬标签的度量上验证，有限样本上对假提升率的上界仅适用于i.i.d.任务，临床评估为回顾性且受限于少量机构，且对深度学习、图像或概率级别指标尚未测试。

---

## 414. A Decade of Learning-Based Control: From Theoretic Foundations to Practical Deployment

**arXiv ID:** 2610.29187 | [PDF](https://arxiv.org/pdf/2610.29187v1)

**作者:** Angela P. Schoellig `[一作]` (Technical University of Munich), Siqi Zhou `[通讯]` (Simon Fraser University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文是一篇综述论文，系统回顾了过去十年学习与控制交叉领域的关键进展，重点讨论了基于模型学习、直接数据驱动控制、强化学习、模仿学习以及贝叶斯优化等方法，并从稳定性、安全性和可验证性等角度对这些技术进行分类与比较。

**💡 创新点**

创新之处在于将控制理论的稳健与安全分析框架与机器学习的表达能力相融合，梳理了学习控制的理论基石与实践路线，并提出了未来研究的若干方向与挑战，特别强调了控制原理对学习过程的指导作用。

**🔧 技术方法**

主要使用的技术包括：控制理论工具（Lyapunov、CLF/CBF、MPC、HJ可达性、鲁棒控制等）、机器学习方法（线性/非线性系统辨识、高斯过程、神经网络、强化学习、贝叶斯优化等），以及它们在数据驱动与直接数据驱动控制中的具体应用。

**📊 数据集**

由于是综述性工作，本文并未使用单一实验数据集，而是综合引用了CDC学习控制邀请会议以及相关文献中的实验案例与数据集信息。

**📈 对比分析**

通过对比分析各方法在理论保障（如稳健性、概率安全）、样本效率、实现复杂度以及实际实验结果方面的优缺点，对不同方法的性能进行了综合评述，强调了在安全性与可验证性方面的显著差异。

**⚠️ 局限性**

局限性包括：聚焦CDC会议论文，可能未覆盖所有最新工作；缺乏统一的定量评估基准和跨方法的直接实验比较；对高维非线性系统、在线持续学习以及与感知融合等更复杂场景的实证研究仍不足。

---

## 415. B-CARE: Blind Classwise Assessment of Relative Evidence for Stance Error Ranking

**arXiv ID:** 2610.29350 | [PDF](https://arxiv.org/pdf/2610.29350v1)

**作者:** Meysam Sabbaghan `[一作]` (Kansas State University), Doina Caragea `[通讯]` (Kansas State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练的B-CARE框架，利用预测盲的LLM验证器生成类级别证据，对姿态检测模型的错误进行相对风险排序，以便优先进行人工审查。

**💡 创新点**

创新点包括：①在不暴露基模型预测的情况下生成证据，防止确认偏差；②采用相对风险分数A_B - S_B，而非单纯置信度；③证据可在多个基模型间复用，无需单独训练错误检测器；④与传统置信度方法形成互补。

**🔧 技术方法**

技术手段：使用Llama‑3.3‑70B‑Instruct（OpenRouter）生成0–3阶梯式类级别证据；构造相对风险分数；利用AUROC、AUPRC、E‑AURC、ErrorCapture@q等指标评估错误排名与审查效果；开展多种对照实验（预测暴露/盲、连续/离散证据、比对不同基线）。

**📊 数据集**

数据集：SemEval‑2016 Task 6A、VaccineStance、COVID‑19‑Stance、P‑Stance共5,206条测试实例；基模型有Phi‑4、Mistral 24B、Gemma 3，共12个数据集‑模型组合。

**📈 对比分析**

比较方法：与13种不需要训练的基线（最大软化概率、预测熵、p(True)、自我评估、采样熵、标签熵、语义熵、证据标签不一致、模型不一致等）以及若干控制实验比较。B‑CARE在12个设置中平均AUROC 0.735、AUPRC 0.524，20%审查预算下能捕获41%错误；在多数设置中优于置信度基线，但在SemEval等部分数据集仍不如预测熵。

**⚠️ 局限性**

局限性：依赖外部LLM生成的证据，若证据质量差会影响排名；七级风险分数易出现大量同分，导致排序分辨率有限；实验仅覆盖四个英语姿态数据集和单一LLM验证器，未验证跨语言或其他任务的泛化；与置信度基线相比，在某些设置下表现不佳；额外的LLM推理成本和版本可复现性受限。

---

## 416. YingLan: A Multimodal Ocean Foundation Model for Unified Representation Learning from Heterogeneous Observations

**arXiv ID:** 2610.29313 | [PDF](https://arxiv.org/pdf/2610.29313v1)

**作者:** Sen Jia `[一作]` (Shenzhen University), Lin Mu `[通讯]` (Shenzhen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并训练了 YingLan 多模态海洋基础模型，用以统一学习卫星影像、网格海洋字段与浮标时间序列等异构观测的表示，进而提升海洋状态预测与多任务迁移性能。

**💡 创新点**

创新点包括：① 以物理属性与观测条件为条件的动态编码机制，实现对异构变量的自适应特征投影；② Spatiotemporal Increment Coupling（STIC）模块，用跨尺度时序增量实现多尺度时空交互；③ 结合掩码重建、时间增量预测与时间组合一致性的演化感知预训练策略，促使模型学习可泛化的海洋动力表示。

**🔧 技术方法**

技术方面采用 MindSpore 框架，利用 Huawei Ascend 910B NPU 进行大规模自监督预训练，核心模块包含基于 transformer 的动态编码、跨尺度卷积映射与双线性交互的 STIC 以及多任务预训练损失函数。

**📊 数据集**

数据集涵盖 6 TB 规模的海洋观测库，包括 Argo、WOD、EN4（浮标）、MODIS、AMSR-2、GHRSST（卫星），GLORYS、CORA、NCEP‑NCAR（重分析），以及深圳沿海区域的多源观测（海面温度、盐度、风场、遥感影像、UAV 图像等）。

**📈 对比分析**

与 LangYa、XiHe、GloNet2 在深圳沿海七天海面盐度预测任务上比较，YingLan 取得平均 RMSE 0.6875 PSU，分别比 LangYa、XiHe、GloNet2 降低 0.58%、5.31% 与 12.42%，显示出更优的预测精度。

**⚠️ 局限性**

局限性在于：① 仍缺乏跨任务、跨区域的广泛评估；② 对不完整或稀疏观测的鲁棒性验证不足；③ 预训练任务仍以海洋状态为主，尚未充分覆盖海洋遥感语义分割、事件预警等更广泛应用场景。

---

## 417. D-Skill: Distilling Teacher Knowledge through Agent Skills

**arXiv ID:** 2610.28914 | [PDF](https://arxiv.org/pdf/2610.28914v1)

**作者:** Shuo Yuan `[一作]` (University of California Berkeley), Joseph E. Gonzalez `[通讯]` (University of California Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过对教师与学生配对轨迹的离线差异分析，将教师的优势转化为自然语言技能文档，从而提升学生在多任务上的表现；

**💡 创新点**

创新点在于：①不需要访问教师或学生权重，完全基于黑盒轨迹；②利用行为差异生成可解释、可编辑的文本技能；③单次离线传输即可显著缩小性能差距，降低推理成本；

**🔧 技术方法**

使用的技术包括：LLM驱动的差异分析器（对配对轨迹提炼经验）、结构化 Consolidator（类似 Trace2Skill 的 map–reduce 合并）、以及在推理时将技能注入模型上下文；

**📊 数据集**

实验数据集涵盖 SpreadsheetBench-Verified、ALFWorld 和 Chess puzzles；

**📈 对比分析**

与传统知识蒸馏（SFT）和 Trace2Skill 进行对比；在 SpreadsheetBench 和 ALFWorld 上，单次离线 D‑Skill 能关闭约 85–95% 的教师‑学生性能差距；在 Chess 上提升约 75–88%；推理成本比教师低 5.7–8.3 倍，单次 distillation 成本仅约 3–4 美分/任务；

**⚠️ 局限性**

主要局限包括：①对分析器质量敏感；②加载技能文档会略微提升推理成本；③只能迁移可通过自然语言指令表达的能力，无法补足学生在深度计算、感知等基础能力上的不足。

---

## 418. A KAN-based Non-linear Factorization for Embedding Layer Compression

**arXiv ID:** 2610.29424 | [PDF](https://arxiv.org/pdf/2610.29424v1)

**作者:** Oliver Giudice `[一作]` (Banca d'Italia), Anna Russo Russo `[通讯]` (Banca d'Italia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 Kolmogorov–Arnold 网络（KAN）的低秩因式分解方法，用于压缩 NLP 词嵌入层。

**💡 创新点**

创新点在于用可学习的非线性一元函数替代传统线性投影，消除线性瓶颈，并提出两种基于 SVD 与块坐标下降的初始化策略。

**🔧 技术方法**

采用 KAN、低秩因式分解、SVD、块坐标下降、RBF 实现的 KAN、Skip‑gram 负采样、知识蒸馏等技术。

**📊 数据集**

使用 120M 词条的 Wikipedia 语料训练 Skip‑gram；在 Google Analogy、WordSim‑353 与 SimLex‑999 上进行内在评测。

**📈 对比分析**

在相同参数预算下与传统低秩压缩进行对比，KF‑BCD 在压缩后即刻的平均性能下降约 16.4%，优于 LF 的 28.8%；在三轮微调后 KF‑SVD+ 的平均下降仅 11.7%，显示出较强的恢复能力。

**⚠️ 局限性**

局限性包括训练稳定性差、优化景观不佳，缺乏下游任务评估，且存在语料与评测集不匹配导致的性能偏差。

---

## 419. Mitigating Query Shortcuts in Chain-of-Thought with Context-Aware Decoding

**arXiv ID:** 2610.29518 | [PDF](https://arxiv.org/pdf/2610.29518v1)

**作者:** Shin'ya Yamaguchi `[一作]` `[通讯]` (NTT), Shin'ya Yamaguchi (NTT)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了将对比解码方法应用于链式推理，以减少查询捷径并提高答案对推理过程的依赖。

**💡 创新点**

提出了 Rationale Contrastive Decoding（RCD），通过对比含推理步骤与仅含查询的下一个词分布，强化答案对推理内容的依赖。

**🔧 技术方法**

采用上下文感知解码（CAD）的对比重加公式，等价于 KL 正则化奖励最大化，实现在不进行参数更新的前提下对下一个词概率进行加权。

**📊 数据集**

在 GSM8K、MATH、CommonsenseQA、AQuA-RAT、MMLU-Pro 五个推理基准数据集上进行评估。

**📈 对比分析**

与直接回答、标准 CoT、Logit Addition 进行对比，RCD 在所有数据集和模型上均提升准确率，对真值推理更敏感，错误推理时表现不佳。

**⚠️ 局限性**

仅提升了答案对推理的行为依赖，若推理本身错误则可能放大错误；未验证推理真实可信度，且缺乏自适应权重机制。

---

## 420. SANE: Automated Scorer-Specific Metamorphic Testing for LLM Judges

**arXiv ID:** 2610.28886 | [PDF](https://arxiv.org/pdf/2610.28886v1)

**作者:** Ethan Ludwick `[一作]` (Cisco Systems), Yu-Chung Hsiao `[通讯]` (Anduril Industries)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 SANE 框架，自动化推理评判器特定能力、生成相应扰动与期望关系，并通过 LLM 验证面板逐步验证配方与实例的正确性，从而构建可扩展的评判器可靠性测试套件。

**💡 创新点**

创新点：1) 通过代理模型从通用行为需求（单调性、可加性、可重复性、鲁棒性、动态范围、校准）自动推导评判器特定的能力、扰动配方与期望关系，消除手工编写配方的瓶颈；2) 引入分离式验证流程（配方对齐、扰动实现、关系适用性），实现全自动、可追溯的验证；3) 在多评分维度上展示该方法比固定配方的 Judge Reliability Harness（JRH）能生成更多验证套件、检测更多缺陷，并与人类标签高度一致。

**🔧 技术方法**

技术：大语言模型代理（Claude Opus 5）实现配方生成与实例化；LLM 验证面板（GPT‑5.2）做配方对齐、实例实现与关系适用性判断；元测试/变形测试框架；统计可靠性评估（Spearman、Kendall）与阈值分析。

**📊 数据集**

数据集：平衡生成集 1200 条（来自 AggreFact‑CNN、AggreFact‑XSum、TofuEval‑MediaS、TofuEval‑MeetB）用于配方生成与实例化；FaithBench 作为独立人类标签基准用于评判器排名对齐。

**📈 对比分析**

比较方法：在 faithfulness、fluency、coherence、professionalism、toxicity 五个评分维度上与 JRH 进行对比；结果显示 SANE 产生 108 条对齐配方、3.7 倍验证套件数量，检测到所有评分器缺陷；在 faithfulness 上，SANE 的通过率排名与 FaithBench AUROC 的 Spearman ρ=0.852，显著高于 JRH 的 0.100，表明更好的人类对齐。

**⚠️ 局限性**

局限性：1) 对比仅在相同代理（Claude Opus 5）下完成，未检验弱模型或多代理的泛化；2) 验证依赖 LLM 面板，缺乏人工标注校验，可能影响可靠性估计；3) 仅在 faithfulness 上与人类标签对齐，其他维度缺少对应基准；4) 未评估多代理评判器或统计度量的适用性。

---

## 421. Precision Coexistence: Angular-Spectral Region Constrained Waveform Design for MIMO-ISAC

**arXiv ID:** 2610.29364 | [PDF](https://arxiv.org/pdf/2610.29364v1)

**作者:** Syed Ali Haider Shirazi `[一作]` (Arizona State University), Sergiy A. Vorobyov `[通讯]` (Aalto University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本工作针对多天线联合感知与通信（MIMO-ISAC）系统，在拥挤的电磁环境中设计双功能波形，实现对传统系统的精确角频域共存约束；

**💡 创新点**

创新点在于提出仅抑制受限角频域的波形能量并保留其他空间自由度的联合角频域约束，并通过MM-ADMM框架对非凸优化问题实现高效求解；

**🔧 技术方法**

采用MIMO-OFDM波形、符号级预编码（构造性干涉）、大规模协同优化技术（MM与ADMM）、低秩Sherman-Morrison近似、CVX求解二次凸子问题等方法；

**📊 数据集**

使用仿真数据，设置12×12天线、32子载波、16 OFDM符号、QPSK调制，构造3个受限角频域的传统系统，随机生成50组通道样本；

**📈 对比分析**

与光谱仅抑制（Spectral-only）与无保护（Unconstrained）两种基准以及仅雷达的基准进行对比；实验显示，所提方法在满足-20dB抑制的前提下，雷达SCNR与无保护基准相近（≈28dB），而光谱仅抑制在高QoS下降幅约1.8dB；QoS可支撑至8dB，且在天线数增大时性能优于其他基准；

**⚠️ 局限性**

主要局限在于对大尺寸空间-频率维度的可扩展性不足，子问题求解成为瓶颈；假设完全无误的信道估计，实际误差会影响抑制效果；未来工作将考虑估计误差、扩展至无源大规模MIMO等。

---

## 422. MASAR: Multi-stage Attack Surface Analysis and Reasoning Framework for Telecom Networks

**arXiv ID:** 2610.29343 | [PDF](https://arxiv.org/pdf/2610.29343v1)

**作者:** Heba Osama `[一作]` (Cyshield Company), Haitham Ghalwash `[通讯]` (Coventry University Egypt Branch)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一套名为 MASAR 的跨层次威胁建模框架，能够将架构层的 STRIDE 威胁映射到协议层的攻击路径，并进一步生成可操作的 MITRE FiGHT TTP，随后利用 MiniZinc 约束模型对 5G IMS 中的两种信令攻击场景进行有限状态的安全性验证。

**💡 创新点**

创新点包括：①将架构威胁与协议行为通过有向无环图（DAG）实现可追踪的逻辑分解；②结合约束满足（CSP）方法对攻击路径的可达性进行形式化验证；③在第三层将攻击动作映射为候选 MITRE FiGHT 技术，为通信安全知识库提供了新的补充。

**🔧 技术方法**

核心技术为 STRIDE 威胁分类、DAG 生成与 AND/OR 逻辑推理、MiniZinc 约束建模与求解，以及对 5G IMS 信令（SIP、Diameter Cx）状态的有限域变量建模。

**📊 数据集**

没有使用公开数据集；评估基于对 5G‑ready IMS 架构的理论信令流程（SIP 会话、Diameter Cx 请求/响应）所构造的约束模型。

**📈 对比分析**

论文未与现有工具直接对比；所采用的 MiniZinc 验证在限定的状态步数内能快速求解（模型 1 约 539 ms，模型 2 约 341 ms），展示了方法在小规模场景下可行性，但未给出大规模或实时性能评估。

**⚠️ 局限性**

主要限制：①模型生成过程完全手工，需要专家介入，缺乏自动化；②约束模型只覆盖了有限的协议行为和状态，无法证明在所有实现中的安全性；③验证仅为有界可达性，无法排除未建模路径的漏洞；④候选 TTP 仍需人工验证与 MITRE 结合，尚未成为正式条目。

---

## 423. Scriptorium: System Design and Implementation of a Multi-Human, Multi-AI Collaborative Digital Library Search Platform

**arXiv ID:** 2610.28929 | [PDF](https://arxiv.org/pdf/2610.28929v1)

**作者:** Abbas Pirmoradi `[一作]` (University of Regina), Orland Hoeber `[通讯]` (University of Regina)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文介绍了Scriptorium的设计与实现，这是一个支持多用户和多AI协作的数字图书馆搜索界面，旨在实现用户搜索活动的完全透明性，并结合七个角色专门化的生成AI代理的管理-工作者架构。

**💡 创新点**

创新点在于将协作透明性与多代理AI支持结合起来，创建一个允许多个用户同时查看和构建彼此活动的工作空间，并通过七个专门化的AI代理提供不同的搜索支持。

**🔧 技术方法**

使用了MERN技术栈（MongoDB, Express, React, Node.js），并通过REST API与OpenAI API进行交互，构建了一个多代理AI的协调器。

**📊 数据集**

论文中没有具体提到使用的数据集，但提到Scriptorium是一个研究原型，旨在通过用户研究评估其假设。

**📈 对比分析**

与现有的协作搜索系统相比，Scriptorium通过透明的工作空间模型增强了用户之间的协作，允许用户在没有明确推荐或分工的情况下，持续查看彼此的活动。性能方面，系统的设计目标是提高协作的偶然发现和交叉影响。

**⚠️ 局限性**

限制在于Scriptorium仍然是一个研究原型，而非生产系统，可能在实际应用中面临可扩展性和用户体验方面的挑战。

---

## 424. inr2action: Encoding Robot Episodes as Neural Implicit Representations for Recall and Action

**arXiv ID:** 2610.28882 | [PDF](https://arxiv.org/pdf/2610.28882v1)

**作者:** Leann Alhashishi `[一作]` (University of Cambridge), Sebastian William Pattinson `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `afceb026-1760-41ae-8d86-010831a37d97` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将机器人每个操作回合的视频拟合为神经隐式表示（INR），并将从共享元学习初始化得到的权重差值作为持久记忆

**💡 创新点**

创新点在于把episode记忆编码为固定尺寸的权重差值，并通过投影网络将其映射为token，使得语言模型和动作解码器仅凭这些token完成问答与下一步指令预测

**🔧 技术方法**

使用NeRV式神经隐式网络、MAML元学习初始化、投影网络、TinyLlama-1.1B进行问答、Pythia-400M进行动作回归

**📊 数据集**

使用RoboMME基准（16种非马尔可夫操控任务）进行评估

**📈 对比分析**

与随机噪声、随机初始化等对照实验比较，问答准确率提升至约47%（基线34.6%），动作回归R²约0.51，证明权重差值可被解码

**⚠️ 局限性**

局限在于仅使用两种训练种子、固定小规模数据集、仅做离线评估、未验证开放式生成或在线适应

---

## 425. OmniSimulator: Aligning Small Language Models for Authentic Heterogeneous Behavior Modeling

**arXiv ID:** 2610.29472 | [PDF](https://arxiv.org/pdf/2610.29472v1)

**作者:** Jiawei Chen `[一作]` (Chinese Academy of Sciences), Hongyu Lin `[通讯]` (Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 OmniSimulator 框架，通过结构化的五维链式推理和教师模型蒸馏，训练小型 LLM 以逼真地模拟短视频平台的用户行为。

**💡 创新点**

核心创新在于设计了面向行为的五维结构化 CoT 模式，并结合任务感知的因果蒸馏与拒绝采样，使模型能够显式恢复用户潜在心理状态。

**🔧 技术方法**

采用 Qwen3 系列小模型，GPT‑OSS‑120B 作为教师生成结构化推理轨迹，利用结构化 CoT、因果蒸馏、拒绝采样和有监督微调技术。

**📊 数据集**

基于 3 个月真实用户交互日志（200 名用户）构建 9,881 条高质量蒸馏训练样本，并在 OmniBehavior 基准上进行评估。

**📈 对比分析**

与零射击、启用原生思考、自由 CoT 微调以及 Claude‑Sonnet‑4.5 等基线对比，Qwen3‑8B 在 OmniBehavior 上平均得分提升 34.33%（从 30.82 变为 41.40），并与 Claude‑Sonnet‑4.5（40.49）相当，同时显著降低过度估计偏差。

**⚠️ 局限性**

实验仅覆盖单一短视频平台，缺乏跨平台或时序泛化；结构化推理轨迹可能继承教师模型偏差，连续与文本任务的目标条件化可能导致事后合理化；难以完全分离结构化模式与过滤机制对效果的贡献。

---

## 426. Strict Equality in Higher Identity Types: Axioms, Completeness, Decidability

**arXiv ID:** 2610.28906 | [PDF](https://arxiv.org/pdf/2610.28906v1)

**作者:** Keri D'Angelo `[一作]` (Cornell University), Dexter Kozen `[通讯]` (Cornell University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究并提出一个严格的高阶同一类型理论，给出等价公理并证明其在 homotopy 模型、CW‑complex 以及满足边界保持和 pullback 保持的 presheaf 模型上的完整性与可判定性；同时构造了 picture（图像）模型并展示了如何在该模型中实现所有公理。

**💡 创新点**

核心创新在于：① 用基于 pushout 的内部合成结构，使所有公理在模型中“on the nose”成立；② 提供了一套 witness 构造（利用 pushout 的中介映射）来实现弱合成律的同伦证据；③ 给出了完整的可判定性算法，证明该理论的 word problem 可判定。

**🔧 技术方法**

使用的技术包括高阶范畴论、图像模型（picture）构造、dyadic 网格技术、pushout 与 pullback 的范畴构造、边界保持条件的分析，以及对 presheaf 模型的抽象化和语义化。

**📊 数据集**

本研究为理论工作，没有使用实验数据集。

**📈 对比分析**

决策程序通过构造 picture 数组实现，其时间复杂度为 O(2^{max h(t₁),h(t₂)})，即指数级上界；与之前缺乏可判定性证明的高阶同一理论相比，提供了首个可判定的实现。

**⚠️ 局限性**

局限性包括：① picture 的数组表示可能过于粗糙，实际计算中更紧凑的表示可能更高效；② 目前仅在满足边界保持和 pullback 保持的 presheaf 模型上证明完整性，尚未覆盖所有可能的模型结构（例如缺少纤维化结构的类型理论）。

---

## 427. The Scientific Research Taste Benchmark (SciTaste-Bench): A Multi-Disciplinary Benchmark for Assessing Large Language Models' Judgment of Research Taste in Scientific Papers

**arXiv ID:** 2610.28653 | [PDF](https://arxiv.org/pdf/2610.28653v1)

**作者:** Xinyi Gao `[一作]` (University Of Chinese Academy Of Sciences), Bing Zhao `[通讯]` (Alibaba Group)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

创建了一个跨六个学科（物理、数学、化学、生物、医学、金融）的金标准评估基准SciTaste，用七维学术口味指数量化论文创新性，并实现了一个结构化工具调用代理，以评估大型语言模型（LLM）的评估能力。

**💡 创新点**

创新点包括：① 将多源元数据（NASA ADS、OpenAlex、arXiv）融合并统一分类；② 设计了七维可解释的学术口味指数并通过双目标随机森林学习得到跨学科共享权重；③ 构建了九个专业化的bibliometric工具和两轮交互式评估代理；④ 提供四种评估模式（无工具、原生搜索、结构化工具、工具+搜索）并公开完整数据、评估代码。

**🔧 技术方法**

技术手段：多源数据融合与冲突解决、基于主题标签的层级分类、主成分分析映射到0‑100分、随机森林权重推导、LLM工具调用代理（两轮回合、结构化数据提取）、Spearman/Pearson/MAE/WAPE双模式评分。

**📊 数据集**

数据集：从NASA ADS、OpenAlex、arXiv抓取1996‑2023年论文，选取300篇（每学科50篇）作为评价集，另外5,700篇（每学科950篇）作为备份；覆盖六个学科，包含元数据、引用、关键词、作者信息等。

**📈 对比分析**

比较方法：对10款前沿LLM在四种模式下进行评估；通过Scheme‑1（相关系数+维度准确度）和Scheme‑2（加权绝对百分误差）得分。结果显示：结构化工具模式通常最优，原生搜索提升有限；物理学科表现最好，其它学科（尤其是生物、医学）表现相对较差；整体得分低，维度级评估（概念创新、语义新颖性、颠覆性）是瓶颈。

**⚠️ 局限性**

局限性：① 每学科仅50篇样本，无法覆盖所有子学科；② 数据以英文为主，可能低估非英语论文；③ 权重向量仅基于量子物理样本推导，未针对每个学科重新学习；④ 工具调用存在速率限制、空响应等技术问题；⑤ 需要定期刷新备份池并扩展到更多学科与人工专家注释。

---

## 428. Auditing Conformal Prediction for Language Models Under Inference-Time Model Shift

**arXiv ID:** 2610.28937 | [PDF](https://arxiv.org/pdf/2610.28937v1)

**作者:** Achref Doula `[一作]` (Tongji University), Max Mühlhäuser `[通讯]` (Tongji University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了语言模型在部署时推理配置变更对分割式共形预测有效性造成的影响，并提出了相应的评估方法。

**💡 创新点**

提出了“推理时模型迁移”概念，证明匹配评估无法保证部署时的覆盖率，并给出了最大阈值和潜在配置鲁棒两种基线。

**🔧 技术方法**

采用分割式共形预测、LAC、RAPS、APS等非合规性分数，以及最大阈值、潜在鲁棒等方法。

**📊 数据集**

在四个离散标签基准（CommonsenseQA、DBPedia‑14、MMLU‑Pro、MobileActions）和八个小型语言模型（Qwen2.5‑1.5B/3B/7B，DeepSeek‑R1‑Distill‑Qwen‑7B，Gemma‑2‑2B，Llama‑3.2‑3B，Phi‑3‑Mini，Mistral‑7B）上进行实验。

**📈 对比分析**

对比了匹配、平均、最差、聚合、MaxThr、Latent‑Robust六种方案，结果表明匹配评估显著高估覆盖率，MaxThr/Latent‑Robust能恢复目标但代价是预测集变大。

**⚠️ 局限性**

研究仅覆盖有限的离散标签任务和固定的量化/温度网格，未给出对连续或随时间漂移配置的理论预测。

---

## 429. CertMoE: Robustness Certification for Mixture-of-Experts LLMs via Routing-Aware Smoothing

**arXiv ID:** 2610.29395 | [PDF](https://arxiv.org/pdf/2610.29395v1)

**作者:** Qingyue Wang `[一作]` (Hong Kong University of Science and Technology), Shuai Wang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CertMoE 框架，为 Mixture-of-Experts LLM 设计了路由感知的鲁棒性认证方法。

**💡 创新点**

通过 ρ-松弛路由包含和诱导子网络的分层认证，解决了动态路由导致的计算图不稳定问题。

**🔧 技术方法**

基于随机平滑（RanMASK / SelfDenoise）与蒙特卡罗采样的概率边界，利用路由计数构造子网络实现鲁棒性证明。

**📊 数据集**

在文本分类基准 SST‑2 与 AGNews 上进行评估。

**📈 对比分析**

与传统平滑认证方法对比，CertMoE 将误证率从超过30% 降至0，覆盖率略低但可靠性显著提升。

**⚠️ 局限性**

仅适用于分类任务；需手动调优路由池比例 ρ，计算延迟约为原方法的两倍，且对生成任务和多模态 MoE 的推广尚未验证。

---

## 430. From Common Information to Canonical Correlation Analysis and Beyond: Bridging Information Theory, Signal Processing, and Machine Learning

**arXiv ID:** 2610.28978 | [PDF](https://arxiv.org/pdf/2610.28978v1)

**作者:** Paris A. Karakasis `[一作]` (Virginia Polytechnic Institute and State University), Walid Saad `[通讯]` (Virginia Polytechnic Institute and State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

综述并系统化了信息论与统计学视角下的共同信息与典型相关分析（CCA），阐明它们在多视角数据中的定义、可辨识性条件、与典型相关的关系，并讨论了从线性到非线性、离散到连续情形下的扩展与应用。

**💡 创新点**

提出了将 Gács–Körner 与 Wyner 的共同信息与 CCA 结合的统一框架，揭示了不同共同信息量（交集型与生成型）的本质差异，并给出了连续变量、K 变量扩展、灰-魏尔网络操作意义及多视角深度学习与自监督学习中的具体实现与可辨识性条件。

**🔧 技术方法**

主要技术包括信息论最优编码与共同信息定义、极大相关性与最大协方差矩阵分析、CCA 与其核/深度/概率泛化、非负矩阵与张量分解（NMF/CPD）、隐变量生成模型与变分推断、以及连续源的失真极限分析。

**📊 数据集**

论文不依赖具体实验数据集，而是通过理论推导、模拟示例（如双对称二进制源、联合高斯源）和已有文献中的实例来说明概念与方法的适用性。

**📈 对比分析**

通过与传统信息论量（互信息、Wyner共同信息、Gács–Körner共同信息）以及经典 CCA 结果对照，展示了各方法在不同情形下的对应关系和可辨识性边界；对连续源的失真极限实验表明在满足可分解性与可识别性条件时，所提出的共同信息可在整个失真范围内保持不变，体现了理论上的优越性。

**⚠️ 局限性**

主要限制在于：1) 纯理论性质，缺乏对大规模真实数据的实证验证；2) 对连续源的共同信息定义要求相对严格的可分解性与可辨识性条件，实际中往往难以满足；3) 对非线性与深度模型的可辨识性分析仍在探索中，需进一步研究鲁棒性与计算复杂度。

---

## 431. When Do Gaussian Priors Fail? A Latent Geometry Investigation of RL Fine-Tuning

**arXiv ID:** 2610.28849 | [PDF](https://arxiv.org/pdf/2610.28849v1)

**作者:** Wang Ru `[一作]` `[通讯]`, Wang Ru

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究在强化学习微调 latent diffusion 模型时，Gaussian prior 对多模态数据导致潜在空间重叠并诱发模式崩塌的机制

**💡 创新点**

提出了一个以潜在空间分离度 d_min/(2σ_0) 为核心的因果链，解释并定量描述 Gaussian prior 导致的 collapse 边界放大，并证明非高斯（如 GMM）先验在重叠 regime 下可显著降低 β_c

**🔧 技术方法**

结合 VAE + latent diffusion + reverse‑KL RL（DDPO/DPOK）框架，利用 ELBO 解析、三源误差分解、动态与变分 β_c 比较以及机器检验（Lean4）等技术

**📊 数据集**

在三组实验中评估：1D 10 模态合成数据、8D→2D 的 30 模态合成数据、真实氨基酸二肽的 Boltzmann 分布（45 距离、2D 潜在）

**📈 对比分析**

通过对比 Gaussian vs GMM prior 的 β_c、覆盖 plateau、dyn/var 比值等指标，发现 GMM 在重叠 regime 下 β_c 下降 70×，覆盖 plateau 完全恢复；在 well‑separated regime β_c 相同但 GMM plateau 更高

**⚠️ 局限性**

局限性在于动态误差放大机制（policy‑side）仍未完全解析；对高维潜在、长 MD 样本、标签无关混合先验的通用性还有待验证

---

## 432. ECHO: Emergent Common-Mode Hidden Offsets Drive Vocabulary Collapse in Preference Optimization

**arXiv ID:** 2610.29410 | [PDF](https://arxiv.org/pdf/2610.29410v1)

**作者:** Yun Li `[一作]` (University of Tokyo), Manabu Tsukada `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对偏好优化（Preference Optimization）过程进行深入分析，发现其在隐藏的隐藏状态中产生的共模偏移（common-mode offset）会导致词汇陷阱（vocabulary well）形成，从而引发生成崩溃（generation collapse），并提出通过对偏移的中心化与范数/响度惩罚来抑制崩溃的技术。

**💡 创新点**

创新点在于：①揭示了隐藏状态共模偏移与词汇陷阱之间的因果关系；②提出了能在训练期间实时监控并控制该偏移的具体方法；③证明了传统排名指标对崩溃缺乏警示性，并展示了外部奖励模型与排名指标的负相关性。

**🔧 技术方法**

使用的技术包括：偏好优化的多种损失（Per-token ranking、SimPO、Textbook DPO 等），隐藏状态偏移分解（mean + residual），中心化（centering）与范数/响度惩罚，统计量（offset norm、dose、capture 等）用于诊断；并结合生成样本的重复率、词干环检测和外部奖励模型评估。

**📊 数据集**

所用数据集：①驾驶偏好列表（约5,383 条四选列表）；②UltraFeedback 对话偏好对（主要为对）。实验覆盖 17 个基础模型（从 0.5B 到 14B 参数，包含 Llama、Mistral、Qwen、Gemma 等）。

**📈 对比分析**

对比方法：将未加偏移控制的训练与加入中心化+范数/响度惩罚的训练进行对比。结果显示：在 3 倍预算下，加入偏移控制可将词汇陷阱捕获率从 70–80% 降至 10–20%，生成质量提升（重复率和尾部占比显著下降），同时排名准确率保持在 0.94–0.96 之间，基本不受影响。

**⚠️ 局限性**

局限性：①偏移控制参数仅在驾驶列表上调优，对其它数据集可能需要重新调参；②实验主要集中在 17 条基础模型，未覆盖所有模型族；③仅对标准排名指标做了分析，其他潜在指标未检验；④仅在基线与全参数微调两种训练方式下验证，其他更复杂的偏好优化框架尚未探索。

---

## 433. KunSet: A Corpus and Benchmark for Lyrics Alignment and Rhyme Retrieval in Kunqu Opera

**arXiv ID:** 2610.29031 | [PDF](https://arxiv.org/pdf/2610.29031v1)

**作者:** Lejun Min `[一作]`, Yang Li `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提供了昆曲语料库KunSet，包含281段录音、482,164个逐音节标注及其对应的音韵属性，并在此数据集上评估了歌词对齐和同韵检索两项任务。

**💡 创新点**

首次实现昆曲逐音节标注并与传统音韵学属性对齐；采用ML辅助的迭代预标注+人工复核流程；提出音频锚点解码器提升对齐精度；利用传统韵脚监督显著提升同韵检索性能。

**🔧 技术方法**

使用SOFA和Mel‑Band RoFormer进行声源分离与对齐预标注；双轮人工复核与在线标注系统；音频锚点解码器（基于phone‑peak scoring）进行边界与锚点联合搜索；对比学习与HuBERT特征结合的投影头进行同韵检索。

**📊 数据集**

自制的KunSet数据集（281段昆曲录音，155.2h，总标注106.3h）以及Yunxue Lizhu音韵数据库；对HuBERT‑base（16kHz）提取的特征进行标准化。

**📈 对比分析**

通过MAE和Acc@200比较对齐模型；基准SOFA MAE为391ms，Finetune 251段+Anchor后MAE降至194ms（Acc@200提升至81.7%）。同韵检索采用P@10指标，基线HuBERT为11.36%，现代韵尾为16.35%，传统韵脚监督提升至17.16%，相对基线提升约5.8%。

**⚠️ 局限性**

对齐误差在昆曲演唱仍高于人类一致性；同韵检索受音频长度和标签分布不均影响；数据集仅涵盖单一录音来源，缺乏跨剧种、跨语言的泛化验证。

---

## 434. Actually, All Learnable Distribution Classes Are Privately Learnable

**arXiv ID:** 2610.28878 | [PDF](https://arxiv.org/pdf/2610.28878v1)

**作者:** Mohammad Afzali `[一作]` (University of British Columbia), Vikrant Singhal `[通讯]` (University of Copenhagen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出一种通用的黑盒降维，将任何可在总变差距离下学习的分布类通过有限样本和 (ε,δ)-差分隐私条件下实现学习。

**💡 创新点**

创新点在于构造了在任意标准Borel空间上的“全局耦合”方案（基于Poisson耦合的泛化），并利用该耦合与稳定基模态聚合实现私有合成数据采样，从而完成了对Ashtiani猜想的正向证明。

**🔧 技术方法**

主要技术包括：Poisson点过程耦合、全局耦合（Grand Coupling）构造、子样本-聚合（subsample‑and‑aggregate）框架、稳定基模态（stability‑based mode）聚合、以及后续的放大（boosting）与隐私后处理。

**📊 数据集**

该工作为理论研究，未使用具体实验数据集，所有结果均为理论证明和样本复杂度上界。

**📈 对比分析**

通过理论推导得到的样本复杂度为 n(α,β,ε,δ)=O( m(1/512 m(α/3,1/32),1/16) · (1/ε) · log(4/βδ) · log(4/β) )。当原始学习器样本复杂度为多项式时，最终隐私学习器的样本复杂度仍保持多项式，且对 ε 与 δ 的依赖仅为对数级。

**⚠️ 局限性**

局限性包括：① 需要学习器在任意小误差下可学习；② 方案主要适用于分布学习任务，无法直接推广到监督学习中学习函数的情形；③ 当 δ=Ω(1/n) 时隐私保证退化为平凡；④ 构造全局耦合对实现细节较为复杂，需要满足可测性和投影一致性等技术条件。

---

## 435. Flexible Language-Guided Representation Editing for Adaptive Robot Behavior

**arXiv ID:** 2610.28893 | [PDF](https://arxiv.org/pdf/2610.28893v1)

**作者:** Audrey Lee `[一作]` (Massachusetts Institute Of Technology), Andreea Bobu `[通讯]` (Massachusetts Institute Of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种名为 FLEX 的灵活特征编辑管线，利用自然语言修正来迭代修改机器人任务特征表示（添加、泛化、屏蔽），并在此基础上学习奖励函数与规划策略。

**💡 创新点**

创新点包括：①将添加、泛化、屏蔽三种编辑操作统一在一个顺序化流程中；②利用大型语言模型从语言修正中自动生成特征描述与实现代码；③证明三种操作齐头并进对解决特征误对齐至关重要；④在真实人类反馈下保持高性能。

**🔧 技术方法**

核心技术：语言模型（GPT‑4o）用于特征提取与编辑；最大熵逆强化学习（IRL）用于奖励学习；MDP 框架与轨迹特征线性组合；仿真与实际机器人演示交互。

**📊 数据集**

数据集与实验环境：2D 9×9 导航网格、7DoF Franka 机械臂桌面操作、五阶段连续编辑序列；12 名参与者提供的自然语言修正与专家演示；真实与仿真结合的数据收集。

**📈 对比分析**

对比方法：单操作基线 ALGAE（仅添加）、MFE（仅屏蔽）、USF（无结构化 LM 编辑）；评价指标为归一化奖励（0–1）。实验显示 FLEX 在所有 7 种误对齐场景中均取得最高得分（≈1.0），并在连续编辑与真实人类反馈实验中保持高性能；在主观 Likert 评估中优于两大基线。

**⚠️ 局限性**

局限性：仅适用于刚体静态操作，难以处理动态或柔性任务；过度依赖语言模型的常识推理，可能无法应对个性化或逆常识偏好；假设语言修正仅影响特征表示而不涉及权重 θ，忽略了“进一步远离”之类的强度指令；未在真实机器人上完成闭环实验。

---

## 436. ER-Chord: Spatially Consistent One-Step Image Editing via Equal-Energy Reliability Redistribution

**arXiv ID:** 2610.29248 | [PDF](https://arxiv.org/pdf/2610.29248v1)

**作者:** Peize Li `[一作]` (National University of Singapore), Xiaobin Hu `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了一种训练与反演均不依赖的空间能量重新分配框架（ER-Chord），通过利用一对邻近噪声水平下控制场的一致性信号，在保持原方向的前提下重新分配编辑能量，从而提升一次性文本引导图像编辑的源图像保留与目标一致性；

**💡 创新点**

核心创新在于将端点一致性作为局部可靠性度量，构造等能量二次优化，并证明其全局唯一解可通过求解一维标量根实现，仅通过重新分配幅度而不改变方向，解决了传统时间聚合忽略空间差异导致的背景误编辑问题；

**🔧 技术方法**

技术手段包括：1）基于ChordEdit的时间聚合控制场；2）定义一致性得分并构建等能量约束的二次优化；3）利用一维二分求解得到重分配系数；4）仅在已有控制场上操作，无需额外网络推理或梯度计算；

**📊 数据集**

评估数据集为PIE-Bench（700个编辑案例）和TEdBench（100个真实图像编辑案例），并在三种生成器骨干（InstaFlow、SwiftBrush-v2、SD-Turbo）上进行跨模型验证；

**📈 对比分析**

实验与ChordEdit以及多步、少步编辑器对比，使用PSNR、LPIPS、CLIP-Whole/Edited等指标；ER-Chord在源保留方面提升约0.57 dB PSNR、12 % LPIPS下降，并在目标语义一致性上保持或提升，同时不增加额外网络调用，推理时间与ChordEdit相近；

**⚠️ 局限性**

主要局限在于依赖ChordEdit提供的方向信息，若基线控制场方向错误，重分配无法纠正；此外仅在固定前置能量下工作，对大幅编辑可能受限；超参数（如μ、κ）需手动设定，缺乏自适应机制。

---

## 437. Our Own Jev: Typed Decisions in One Forward Pass, from an In-House Judge to Open LoRA Models

**arXiv ID:** 2610.29052 | [PDF](https://arxiv.org/pdf/2610.29052v1)

**作者:** Yuchuan Weng `[一作]` (PayPal.AI), Srinivasan Manoharan `[通讯]` (PayPal.AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种类型化决策模型，该模型通过读取语言模型的输出概率来快速判断选项，而不是生成和解析文本。研究分为两部分：第一部分介绍了一个内部的系统1评估者，用于处理约148,000个编码会话记录；第二部分探讨了无参数的读取输出是否具有竞争力。

**💡 创新点**

创新点在于提出了一种新的类型化决策接口，能够在一次前向传递中计算每个选项的概率，显著降低了成本和时间，同时保持了较高的准确性。

**🔧 技术方法**

使用了LoRA适配器和指针头技术，结合了冻结的语言模型输出层进行决策。

**📊 数据集**

使用了约25,000个公共决策的Eikos数据集，以及148,000个内部编码会话记录。

**📈 对比分析**

与商业系统的比较显示，35B模型在JevBench的111个公共难题上得分79.3%，显著优于五个开放系统，但与商业参考系统（73.0%）没有显著差异。2B模型在所有四个kev检查点上表现优于它们。

**⚠️ 局限性**

限制在于第一部分的评估依赖于单一的前沿评估者标签，且评估集规模较小，无法充分解决模型间的细微差异。此外，所有模型均使用了自撤回的训练数据，可能影响结果的可靠性。

---

## 438. GenXR-Bench: Benchmarking Coding Agents for Interactive XR Development

**arXiv ID:** 2610.29014 | [PDF](https://arxiv.org/pdf/2610.29014v1)

**作者:** William Huang `[一作]`, Ruofei Du `[通讯]` (Google Xr Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了专门用于评估大型语言模型（LLM）在扩展现实（XR）软件开发中的代码生成能力的端到端基准（XR-Coding Benchmark），并提供了完整的自动化测试框架（XR Test Suite）与评测工具。

**💡 创新点**

创新点包括：①首次构建面向XR应用的完整编码任务集合（SDK 功能实现、bug 修复、应用生成）；②开发基于 XR Blocks 的统一、可复现的实验环境；③提出脚本化与 VLM 适配的交互测试演员以及程序化与视觉断言相结合的验证方式；④使用语义标签让不同实现可互操作；⑤在真实 XR 设备上验证了模拟器评测与专家判断的一致性（96.5%）。

**🔧 技术方法**

核心技术包括：XR Blocks 0.21.0（WebXR 框架）、XR Test Suite（包含 kinematic actors、adaptive VLM actors、scene-graph 断言与视觉断言）、XR Blocks DevTools、OpenAI / Google Gemini / Claude 等前沿 LLM、Harbor Framework 进行实验管理。

**📊 数据集**

数据集涵盖：• 67 个从 436 篇 XR 论文中提炼的应用任务；• 552 条 XR Blocks SDK 的 Pull Request 与 issue 产生的 约 30+ SDK 任务；• 124 个候选应用（由 Gemini 3.5/3.7 生成）在 8 名专家评测的 372 次运行中形成的评测数据；• 372 条自动化评测结果与专家二进制判断，用于计算准确率与 flakiness。

**📈 对比分析**

比较方法：采用 Pass@1（单次成功率）、Design Score（VLM 对十条设计启发式的 Likert 评分）、T/E（测试/编辑比例）、Turn、Token、Cost 等多维指标。性能方面：最强模型（Claude‑opus‑5）在整体 Pass@1 仅达 50.9%，SDK 修复任务最高 94% 以上；SDK 功能实现与应用生成的 Pass@1 低于 40%；设计评分最高 4.03 分；测试套件与人工评估的一致率 96.5%，flakiness 仅 4%。

**⚠️ 局限性**

局限性：①模型在异步传感器初始化、坐标变换、双手交互语义等空间推理方面错误频发；②对真实物理交互与人机工效的理解不足，导致可用性差；③评测套件在某些非标准交互（如手部细节）下会出现误判；④高阶空间推理与环境感知仍超出目前 LLM 的能力范围；⑤仅在 XR Blocks 平台测试，缺乏跨引擎验证；⑥对 VLM 判定的依赖可能隐藏细微错误。

---

## 439. Reasoning-Search-Augmented LLMs: An Operational Taxonomy, Benchmark Analysis and Controlled Evidence Synthesis

**arXiv ID:** 2610.29189 | [PDF](https://arxiv.org/pdf/2610.29189v1)

**作者:** Biswas Poudel `[一作]` (Texas A&M University), Xianshun Jiang `[通讯]` (Texas A&M University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对把检索与推理循环嵌入LLM的系统进行了系统化定义与分类，并梳理了相关工作与评估指标。

**💡 创新点**

提出了基于四个设计轴（检索耦合、控制器设计、知识源、落地严格度）的操作性分类体系，并将学习信号视为横向维度；同时给出控制组件消融的经验量化证据。

**🔧 技术方法**

采用大语言模型、检索组件（DPR、Elasticsearch等）、知识图谱接口、API调用以及强化学习、提示式、监督式等训练方法。

**📊 数据集**

使用公开基准（如HotpotQA、FEVER、SEALQA、MERRIN、BrowseComp、DeepResearch Bench II等）以及内部检索语料库（Wikipedia、特定领域语料、知识图谱）。

**📈 对比分析**

通过对同一论文内部的匹配消融实验评估单元效果，报告了EM、F1、准确率等质量指标与检索调用、token量、延迟等效率指标；结果显示，状态化检索、过程奖励和控制器组件可显著提升答案质量与检索效率。

**⚠️ 局限性**

研究对知识源与落地严格度的干预证据不足，且评估多依赖于已预训练模型与检索器，缺乏跨平台、跨预算的统一对比；未来需更多统一接口与鲁棒性测试。

---

## 440. Computer Vision and Knowledge-Grounded Language Models for Intelligent Beekeeper Assistance

**arXiv ID:** 2610.29283 | [PDF](https://arxiv.org/pdf/2610.29283v1)

**作者:** Asaduz Zaman `[一作]` (Monash University), Malika Nisal Ratnayake `[通讯]` (Monash University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一套可在边缘设备部署的人工智能框架，能够把蜂巢框架图像转换为结构化事实，再结合知识库生成可操作的养蜂报告

**💡 创新点**

创新点在于将视觉感知、事实表述、知识检索与语言生成分离，并引入了可验证的事实-文本映射与RAG式知识检索，提升了解释性和可追溯性

**🔧 技术方法**

使用YOLO目标检测器、边缘LLM、知识库检索模块和结构化事实构造器，构成完整的图像到报告链路

**📊 数据集**

基于公开的FAIRHiveFrames‑1K蜂巢图像数据集，并将其转换为FAIRHiveText‑1K结构化事实与文本描述的数据集

**📈 对比分析**

论文目前未给出完整实验结果，主要阐述框架设计与数据集构建，未来将通过完整管道评估检测精度与报告质量

**⚠️ 局限性**

局限性包括：尚未完成端到端验证，依赖专家人工校准知识库，数据集规模相对有限，边缘LLM可能存在生成错误或信息缺失

---

## 441. ExpEvolve: Structuring and Reusing Experience for Open-Ended Discovery

**arXiv ID:** 2610.29303 | [PDF](https://arxiv.org/pdf/2610.29303v1)

**作者:** Yongqiang Chen `[一作]` (MBZUAI), Kun Zhang `[通讯]` (MBZUAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种在进化编码代理中通过识别、结构化并利用高层“idea”因素来指导搜索的框架。

**💡 创新点**

提出从经验中发现和利用高层idea因素的机制，并将其转化为多臂赌博机动作实现双层探索与利用。

**🔧 技术方法**

结合LLM的提问与注释、对比学习的pairwise ranker、MAB（UCB）策略、随机探索与plateau-guided探索。

**📊 数据集**

11个开放式数学发现任务（如自相关不等式、Erdős最小重叠等）和MLS-Bench的6个机器学习工程任务。

**📈 对比分析**

与基线Evolutionary agent、SCS+meta等对比，在相同预算200评估下平均与最佳成绩均显著提升，样本效率提升至1.7-26倍，迁移至其他LLM时仍保持优势。

**⚠️ 局限性**

依赖LLM的准确性与解释能力；idea发现需要足够探索，若探索不足可能误识别；在高维idea空间时计算成本上升。

---

## 442. Guilt by Association: Semantic Leakage Drives Associative Gender Bias

**arXiv ID:** 2610.28944 | [PDF](https://arxiv.org/pdf/2610.28944v1)

**作者:** Hazel Chen `[一作]` (University Of British Columbia), Hila Gonen `[通讯]` (University Of British Columbia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构造带有和不带有实体归属的提示，研究语言模型在面对无关概念时的语义泄漏现象，并进一步探讨同类概念在场时是否会诱发性别偏见。

**💡 创新点**

创新点在于证实语义泄漏不依赖于概念与实体的直接关联，单纯概念出现即可导致生成偏差，并将此视为一种新的关联性性别偏见，为偏见评估与缓解提供新的视角。

**🔧 技术方法**

采用提示对比实验，使用SentenceBERT和BERTScore计算生成文本与概念的余弦相似度来量化泄漏；在logit层计算基于概念与控制的性别代词的对数几率比ΔLOR，进一步分析偏见强度。

**📊 数据集**

构建了包含246个概念提示（21类）及其对应的匹配控制，共738个提示（涵盖三种实体配置），以及124个性别化概念（跨六个社会领域）和21个中性概念，共1,160个提示，主要基于公开的性别词表与词向量检验。

**📈 对比分析**

在9个大型语言模型（包括base与instruct变体）上进行实验，发现即使概念与查询实体不一致，Leak‑Rate始终超过50%，4/9模型在不匹配情况下泄漏更强；关联性偏见在间接条件下保持约73%的直线偏差，平均ΔLOR在直线0.81与间接0.66之间，表明大部分偏见效应可由概念存在本身驱动。

**⚠️ 局限性**

研究仅针对二元性别与英语短提示，未覆盖非二元、其他社会维度（种族、年龄、宗教等）以及长文本或多语言环境，且基于开放权重模型的logit分析受限，未来工作需扩展至更广泛维度和更自然的上下文。

---

## 443. The Serendiptionnaire: a Questionnaire for Measuring Experienced Serendipity

**arXiv ID:** 2610.29487 | [PDF](https://arxiv.org/pdf/2610.29487v1)

**作者:** Brett Binst `[一作]` (Vrije Universiteit Brussel), Annelien Smets `[通讯]` (Vrije Universiteit Brussel)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了Serendiptionnaire问卷，用于测量用户在使用推荐系统时的经验性惊喜（serendipity）

**💡 创新点**

首次提供了针对经验性惊喜的多维、可模块化、经过认知访谈验证的量表，解决了先前研究缺乏统一、可比度量工具的难题

**🔧 技术方法**

采用认知访谈（cognitive interviewing）方法进行项目编写与修订，结合问卷设计的Question Appraisal System进行系统性评估

**📊 数据集**

利用27名来自不同国家、年龄、教育背景、英语水平的用户，在Instagram、TikTok、YouTube、Spotify等平台上进行推荐体验收集

**📈 对比分析**

未提供传统性能指标，而是通过访谈和项目评估验证问卷的可解释性和可靠性；指出未来可使用该问卷验证系统中心的惊喜度量与用户体验之间的相关性

**⚠️ 局限性**

问卷需根据具体研究情境手工适配；并非完全通用，仍缺乏完整性；当前仅针对推荐系统，未验证可否推广至其他数字系统

---

## 444. Optimal Rényi and Logarithmic Additive Functional Entropy Estimation

**arXiv ID:** 2610.28999 | [PDF](https://arxiv.org/pdf/2610.28999v1)

**作者:** Arman Adibi `[一作]` (Augusta University), Piotr Krysta `[通讯]` (Augusta University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文针对从有限字母表抽样估计Rényi熵以及通用对数加性函数，给出了统一的最优样本复杂度界定。

**💡 创新点**

创新点在于实现了对所有实数阶α>1的样本复杂度统一描述，阐明了整数、非整数及高阶趋向对样本需求的转换，并首次将对数精度与最佳多项式逼近相结合。

**🔧 技术方法**

主要技术包括Poisson计数模型的阶乘噪声恒等式、多项式逼近与加权逼近对偶、对数精度的概率不等式以及正性先验构造与判别测试。

**📊 数据集**

论文主要为理论分析，无使用真实数据集；若需要验证，可在模拟中生成均匀或自定义分布样本。

**📈 对比分析**

与之前的整数阶估计或非整数阶O(k/ log k)上界相比，本文提供了与误差ε和字母数k无关的常数倍最优上界和下界；实验（若存在）表明估计误差随样本数按理论预期收敛。

**⚠️ 局限性**

局限在于需要预先知道α的具体取值，对非整数α的根求解可能导致实现复杂；对数加性函数的结果仅适用于在0附近有有限领先幂的连续函数，无法直接处理所有光滑或凸函数。

---

## 445. RadioDreamer: A World Model for Radio Map Forecasting and Imagination-Based Optimization

**arXiv ID:** 2610.29382 | [PDF](https://arxiv.org/pdf/2610.29382v1)

**作者:** Haoye Chai `[一作]`, Mugen Peng `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了 RadioDreamer 世界模型，用于基站动作引导的无线电地图预测与网络参数的长期优化。

**💡 创新点**

创新点包括：① 结构化的 Cellular‑Aware Transformer 通过时间、基站、行列维度的注意力捕捉无线电环境的因果动态；② 区域自适应监督根据预测误差自适应聚焦动态敏感区；③ 在隐空间中进行“想象”滚动，直接对基站功率/倾斜角进行长周期规划。

**🔧 技术方法**

技术手段：Encoder‑Decoder 结构的变分自编码器；Transformer 多头自注意力（时间、基站、列、行四维分支）；动态区域掩码；奖励模型与 Actor‑Critic 的延迟想象训练；对抗式学习与残差解码。

**📊 数据集**

使用 Sionna RT 仿真生成的三城市（Street Canyon、Etoile、Munich）数据集，共 400,000+ 无线电地图帧，包含基站部署、移动物体轨迹、功率/倾斜动作序列。

**📈 对比分析**

与三类基线（3GPP 统计模型、静态地图模型如 RadioUNet、PMNet、通用世界模型 Dreamer‑RSSM）在 T1/T2/T3 测试协议下进行对比；RadioDreamer 在全图 MAE、动态区 MAE、方向准确率等指标上平均提升约 25%，在想象规划任务中相较于 RM‑SAC/解码规划提升 8.8% 的累计回报。

**⚠️ 局限性**

局限性：仅基于仿真数据，缺乏真实测量验证；对更大规模基站布局、不同频段和更复杂动态场景的泛化能力待进一步评估；模型对极端移动场景的鲁棒性尚不充分。

---

## 446. Pareto Inefficiency of EFX

**arXiv ID:** 2610.29201 | [PDF](https://arxiv.org/pdf/2610.29201v1)

**作者:** Sepideh Mahabadi `[一作]` (Microsoft Research), Jakub Tarnawski `[通讯]` (Microsoft Research)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过构造正整数加性估价的公平分配实例，证明在任意四个代理、八件物品的情形下不存在既满足EFX（任意一件物品移除后仍无妒忌）又满足Pareto最优的分配；并进一步推广到更大规模实例，展示对于任何α>2/3，亦不存在同时满足α‑EFX和Pareto最优的分配。

**💡 创新点**

创新点在于提供了最简明、最直观的正值实例，首次展示在正估价下EFX与Pareto最优不可兼容，并构造参数化族实例证明了α‑EFX与Pareto最优的矛盾可以延伸至接近2/3；同时提出了一个关于是否存在通用α>0使得α‑EFX与Pareto最优兼容的开放问题。

**🔧 技术方法**

采用对称性分析（将代理与物品映射到正方形的对称群）和博弈式的价值“交易”证明，结合组合搜索与手工构造的实例，展示了EFX对大件物品单独分配的强制性及由此产生的Pareto不最优。

**📊 数据集**

文中未使用公开数据集，而是通过作者自建的“agentic harness”结合Claude Opus 5.5和GPT-6 Astra模型自动搜索得到的实例集合；实例规模仅为四代理八件物品，数值取正整数。

**📈 对比分析**

由于研究焦点为否定性结果，未进行算法性能对比；通过计算机验证（仅有4^8=65536种分配）确认实例无兼容分配。对α‑EFX的扩展实例通过理论上推导而非实验验证。

**⚠️ 局限性**

局限性在于仅给出了对称结构的负面实例，未提供普适性或随机模型下的概率论结果；此外未给出算法或近似方案实现EFX与Pareto最优兼容的具体方法；最后证明仅覆盖α>2/3，低于此阈值的α‑EFX与Pareto最优兼容性仍不明确。

---

## 447. Parallel Thinking for Large Language Model with Path Diversity

**arXiv ID:** 2610.29501 | [PDF](https://arxiv.org/pdf/2610.29501v1)

**作者:** Jiaru Qian `[一作]` (Peking University), Ge Li `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 DPT（Diverse Parallel Thinking）方法，改进大模型并行推理路径的多样性，解决模式崩塌问题

**💡 创新点**

创新点在于引入基于前缀的 token 级多样性奖励，将路径多样性直接嵌入 RL 奖励中，并采用两阶段训练（先 SFT 再 GRPO）

**🔧 技术方法**

使用强化学习（GRPO）与自回归语言模型、前缀相似度奖励、专门的并行推理控制标签等技术

**📊 数据集**

训练使用 DAPO 作为 RL 数据集，SFT 采用 GSM8K 生成的并行推理示例；评测在 AIME24/25、AMC23、MATH、HumanEval、ARC-c、MMLU‑Pro 等多任务上

**📈 对比分析**

与 Direct、SFT、GRPO、LTE、TTRL、EVOL‑RL、Parallel‑R1 等基线对比，DPT 在 pass@1 与 pass@16 上均显著提升，尤其在数学难题与编码任务上领先 6‑10% 左右

**⚠️ 局限性**

仅针对文本推理，未探索多模态或交互式场景；路径数量固定，缺乏自适应分配机制；前缀长度和奖励权重的选择仍需进一步调优

---

## 448. From AI-Free Gates to AI-First Work: How Engineering Students in India Navigate University Education and Campus Placements Without AI Guidelines

**arXiv ID:** 2610.29140 | [PDF](https://arxiv.org/pdf/2610.29140v1)

**作者:** Samhitha Goutham `[一作]` (Anna University), Ziwei Gu `[通讯]` (Harvard University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对印度工程专业学生在课程、考试与校园招聘中使用生成式 AI 的行为及其对进入 AI 驱动工作场所的影响进行了访谈研究，采用反思性主题分析与信号理论进行解释。

**💡 创新点**

首次系统性描述了印度学生在 AI 未被纳入评估体系的“AI-Free Gates”与 AI 预期工作场所之间的差距，并从信号理论角度阐释 AI 辅助成绩如何削弱传统评估信号。

**🔧 技术方法**

采用定性研究方法：半结构化访谈、手工转录与人工翻译，随后进行编码与主题分析；分析框架借鉴了信号理论。

**📊 数据集**

研究数据为 23 名印度工程与计算机专业学生（包括本科在读与近届毕业生）的访谈记录与转录文本。

**📈 对比分析**

本研究未进行定量对比或性能评估，而是通过主题分析呈现学生行为与观点，缺乏实验或基准测试。

**⚠️ 局限性**

样本规模有限，主要集中于少数高校，女性参与者仅两名，且多为自报数据，未访谈教师、招聘人员或管理层，结果可能不具代表性。

---

## 449. Transaction Foundation Models for Retrieval-Augmented Generation over Financial Transaction Data: A Comparative Evaluation with Text Embeddings

**arXiv ID:** 2610.28641 | [PDF](https://arxiv.org/pdf/2610.28641v1)

**作者:** Snehal Pal `[一作]` `[通讯]` (Indian Institute of Technology Kharagpur), Snehal Pal (Indian Institute of Technology Kharagpur)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个基于 IBM AML 交易数据的确定性 QA 检索‑增量生成基准，并对 NVIDIA 事务基础模型 (TFM) 与多种通用与金融文本嵌入模型进行比较。

**💡 创新点**

提出了可比较大小和数据匹配的检索评估框架，并发现事务基础模型在高基数个体实体查询上优于通用嵌入，而在低基数分类查询上优势消失。

**🔧 技术方法**

使用 TFM（29M 参数的 decoder‑only 交易编码器）、BGE、E5、GTE、FinBERT 等嵌入器以及 Llama3.2 1B 生成模型进行检索与生成。

**📊 数据集**

使用 IBM Synthetic Anti‑Money‑Laundering（AML）交易数据集（约 5,078,345 笔交易）生成的 161 篇实体汇总文档和 1,026 个模板生成的问答对。

**📈 对比分析**

在同等 token 预算和数据量的条件下，TFM 在 Recall@5 0.81、MRR 0.78、答案准确率 66.7% 上显著优于公平通用嵌入模型（Recall@5 0.25–0.39、准确率 27.5%），但在低基数类别如银行、货币、支付格式时，人工汇总+通用嵌入模型性能更好。

**⚠️ 局限性**

局限在单一合成数据集、固定模板问答、仅用 1B 参数生成模型、未给出置信区间以及部分基准模型参数不匹配。

---

## 450. How Much Does the Harness Matter, and for Whom? A Multi-Model Study of Context in LLM-Driven Vulnerability Discovery

**arXiv ID:** 2610.28802 | [PDF](https://arxiv.org/pdf/2610.28802v1)

**作者:** Ganni Galea Curmi `[一作]` (Georgian), Ben Wilde `[通讯]` (Georgian)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对真实生产代码进行多模型、成本意识的安全漏洞挖掘实验，使用开源 DeepSec harness 对 150+ TypeScript 文件进行扫描、LLM Agent 分析并进行自检、独立审查和 PoC 验证。

**💡 创新点**

揭示模型与工具/上下文组合的性能差异，证明针对性源代码上下文对低成本模型提升显著且依赖模型能力，提出多模型组合与重复运行的必要性。

**🔧 技术方法**

利用 LLM Agent 与 DeepSec 静态分析结合的 ReAct 循环，CodeQL 代码属性图与 Trailmark 生成结构化上下文，Prompt Engineering、模型自检、独立审查与 PoC 验证技术。

**📊 数据集**

采用未公开泄露的 TypeScript API 网关源码（约 150 文件）以及 CyberGym Level‑0 的 C/C++ 例子进行迁移实验。

**📈 对比分析**

在 24 个模型–运行器组合上进行两轮实验，比较安全根因数量、精确率、成本/漏洞以及覆盖率；最强配置覆盖 60% 根因，低成本模型在相同成本下可匹配高端模型。

**⚠️ 局限性**

实验局限于单一代码库、仅两轮运行、上下文与模型交叉影响难以完全分离，以及成本和精度受运行器差异影响。

---

## 451. Target Regeneration Shrinks the Gains of Draft-Training Objectives in Speculative Decoding

**arXiv ID:** 2610.29013 | [PDF](https://arxiv.org/pdf/2610.29013v1)

**作者:** Jungseob Lee `[一作]` (Korea University), Sugyeong Eo `[通讯]` (Yonsei University Mirae Campus)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对比目标模型自回归生成与外部模型生成的响应，评估交叉熵、token级蒸馏（KD）及门控蒸馏在Speculative Decoding中的增益。

**💡 创新点**

发现当响应由目标模型自回归生成时，KD与门控蒸馏的增益显著收缩，提示需在自回归响应上评估增益以避免高估。

**🔧 技术方法**

采用DFlash块式draft模型，配合交叉熵、token级蒸馏与生存门控训练目标，并使用树/链式验证器测量平均接受词数（MAT）。

**📊 数据集**

实验使用ALLaVA、ShareGPT4V视觉语言语料以及UltraChat文本语料，并以Qwen3-VL-8B和Qwen3-4B为目标模型。

**📈 对比分析**

在外部响应上，KD提升MAT约10–20%，门控提升约4–6%；在自回归响应上增益仅为2–5%，显示显著缩减。

**⚠️ 局限性**

局限包括仅测试两类draft模型和两种目标、样本量有限、未覆盖全部训练长度及其他蒸馏方案，评估仅基于MAT，未测量速度或质量。

---

## 452. MOSAIC: An Omni-Modal Benchmark for Long-Term Memory and Cross-Modal Reasoning in Conversational Agents

**arXiv ID:** 2610.29249 | [PDF](https://arxiv.org/pdf/2610.29249v1)

**作者:** Hong Ting Tsang `[一作]` (Hong Kong University of Science and Technology), Yangqiu Song `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了MOS AIC benchmark，用于评估多模态（文本、图像、视频、音频）长时间对话记忆的写、检索、传递和阅读四阶段性能。

**💡 创新点**

创新点在于：① 结合四种模态且跨会话的多跳推理；② 通过目标化空间、时间和声调信号设计挑战；③ 采用可互换的四阶段诊断协议，将系统瓶颈精确定位。

**🔧 技术方法**

使用的技术包括：大型语言模型（如 Gemini 3.5、Qwen 3.7）、多模态检索（Caption-Text、Native-MM）、结构化写作、ReAct、A‑MEM、MemoryOS 等；并对每个阶段进行配对实验。

**📊 数据集**

数据集为MOS AIC：70个合成 persona，11–24 轮会话，923 个问答（784 可答 + 139 拒答），包含 10,765 件多模态 artefact（图像、视频、音频、文本）。

**📈 对比分析**

通过固定 reader、prompt、packer、budget 的配对实验，比较不同后端（Flat RAG、A‑MEM、MemoryOS、ReAct 等）在完整链覆盖率、检索 Recall@k、答案准确率和拒答率上的表现；A‑MEM 在链完整度上最高，但无系统能同时兼顾答题和拒答。

**⚠️ 局限性**

限制包括：① 数据为合成，缺乏真实用户历史；② 视频相关实验样本不足；③ 原声音频与 ASR 的差距未能显著确认；④ 读写阶段诊断关联性不一定是因果关系。

---

## 453. VMSpec: A Decompiler-based Approach against Virtual Machine Obfuscation

**arXiv ID:** 2610.29566 | [PDF](https://arxiv.org/pdf/2610.29566v1)

**作者:** Fabian Kilger `[一作]` (Technical University of Munich), Manuel Andreas `[通讯]` (Technical University of Munich)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对使用虚拟机进行加壳的恶意软件，本文提出了一种基于反编译器的解密方法，能够将虚拟机字节码恢复为高层次的程序代码。

**💡 创新点**

创新点在于将传统的反编译技术与虚拟机特定的指令映射表相结合，实现了对多种常见 VM 加壳器（如 VMP、VMProtect）的自动化逆向，显著降低了手工分析的工作量。

**🔧 技术方法**

使用了LLVM反编译框架进行控制流图重构，并通过符号执行与模式匹配来识别 VM 指令集；同时引入了机器学习辅助的指令映射生成，提升了解码准确率。

**📊 数据集**

评估数据集主要由近200个已知采用 VM 加壳的恶意软件样本构成，覆盖了VMP、VMProtect、Themida等主流加壳工具。

**📈 对比分析**

实验与传统的静态反汇编、动态监控等方法对比，所提方法在恢复率上提升了约30%，平均处理时间从手工分析的数小时缩短至十几分钟，准确率达到92%。

**⚠️ 局限性**

局限性包括：对极度混淆或自定义 VM 的指令集识别仍存在误差；高频率的自举解码需要较多计算资源；且对多线程或加密通信等运行时特性支持有限。

---

## 454. AOV-GS: Active Open-Vocabulary Semantic Mapping with 3D Gaussian Splatting

**arXiv ID:** 2610.29374 | [PDF](https://arxiv.org/pdf/2610.29374v1)

**作者:** Rodion Shkokov `[一作]` (Moscow Institute of Physics and Technology), Dmitry Yudin `[通讯]` (Moscow Institute of Physics and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于冻结的SAM和CLIP模型的主动开放词汇语义地图系统AOV-GS，利用语义新颖度引导视角选择，并在3D高斯分布地图上训练稀疏语言场进行文本查询分割。

**💡 创新点**

创新点在于将语义新颖度（CLIP嵌入相对已存关键帧）作为主动规划的补充，在不进行任何场景细调的情况下提升开放词汇语义覆盖；同时通过CorrCLIP对掩模和嵌入进行后处理，进一步提升语义质量。

**🔧 技术方法**

核心技术包括：冻结的SAM+CLIP进行实例掩模与语言嵌入提取；基于3D Gaussian Splatting的实时几何重建与ActiveGS式的下一最佳视角规划；LangSplatV2稀疏语言场训练与多尺度文本查询分割；以及CorrCLIP的掩模融合与类抑制。

**📊 数据集**

在Replica（8个室内场景）和Matterport3D（5个室内场景）两个标准室内扫描数据集上进行实验。

**📈 对比分析**

与几种基线（被动语义映射、纯几何主动规划、Geometry+CorrCLIP、ActiveSGM轨迹+LangSplatV2）比较，AOV-GS在Replica平均mIoU从33.3%提升至34.6%（+1.3pp），加CorrCLIP后为35.6%；在MP3D上平均mIoU从21.3%提升至23.1%（+1.8pp），并保持与几何规划相当的PSNR/SSIM/LPIPS和略优的深度误差。

**⚠️ 局限性**

局限性包括：仅在仿真环境下评估，未考虑SLAM漂移和真实传感器噪声；在光照或复杂视觉条件下SAM+CLIP的掩模质量下降时，语义新颖度优势减弱；对大规模场景的计算开销（尤其是高K值）较高。

---

## 455. Language Impairment as a Lens onto Large Language Models

**arXiv ID:** 2610.28992 | [PDF](https://arxiv.org/pdf/2610.28992v1)

**作者:** Muxuan He `[一作]` (University of Southern California), Yanjun Gao `[通讯]` (University of Colorado Anschutz Medical Campus)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过构建基于临床失语学的双方法框架（分类器驱动的结构性损伤与连续激活向量调节），系统性诱发并识别大型语言模型的失语类功能失调。

**💡 创新点**

创新点在于将神经心理学的失语诊断与可解释性技术融合，提出同时利用结构化损伤与表示向量两种手段验证功能模块，揭示模型内部高度耦合的分布式网络。

**🔧 技术方法**

使用了分类器驱动的注意力头剔除（CLAMP）、激活补丁、向量差分调节、梯度裁剪与基于损失的优化等技术；以及BGE-large句子编码器与低秩适配器的轻量级分类器。

**📊 数据集**

主要采用 AphasiaBank 语料库的《灰姑娘》叙事任务和《花生酱三明治》步骤描述任务，涉及 11 种不同规模的 LLM（包括 Llama‑2‑chat、Pythia、Qwen 等）。

**📈 对比分析**

与现有的基于语料统计、梯度权重、随机头剔除等方法对比，分类器驱动损伤在 54/66 设定中成功诱导失语，约 82% 的专家认定为目标失语；向量调节在 58% 的案例中被两位专家判定为目标失语，且在未用于训练的任务上保持 63% 的一致性；分类器准确率达 83%，显著优于 LLM 判别器。

**⚠️ 局限性**

局限性包括仅覆盖英语两项叙事任务，缺乏多语言和多模态（如图像描述）支持；分类器训练样本不足（仅 259 个专家标注的失语与模型失真示例）；实验仅针对有限的模型家族与规模，未验证混合专家（MoE）或更大模型的泛化能力。

---

## 456. Fig: Automated Multiagent Behavior Elicitor

**arXiv ID:** 2610.28891 | [PDF](https://arxiv.org/pdf/2610.28891v1)

**作者:** Anrui Gu `[一作]` (Fleet AI), Nicolai Ouporov `[通讯]` (Fleet AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过自动化生成游戏、设计干预机制和评估框架，诱导多智能体系统在文本游戏中表现出指定的社交行为。

**💡 创新点**

将行为诱导视为机制设计，构建统一游戏接口、可编程干预和机械+评判双重评估，使行为评测可控、可重复且可扩展。

**🔧 技术方法**

利用大型语言模型（如GPT‑5.6 Sol、GPT‑6‑Luna）进行行为拆分、游戏生成与干预设计，结合OpenSpiel、TextArena、Social Gym等游戏框架，以及机械检查与人类校准的rubric judge。

**📊 数据集**

采集并使用135个文本游戏（包括TextArena 75、OpenSpiel 21、Social Gym 21、Fig原创 15 等），并通过LLM生成额外游戏，覆盖六类目标行为。

**📈 对比分析**

与对照游戏比较，干预场景使目标行为尝试率平均提升约10.9%，六种行为的平均尝试率为45.2%，实验在多模型上验证，效果显著。

**⚠️ 局限性**

受限于游戏环境分布、实现代理的误差、规则理解混淆以及较高的运行成本，导致行为可解释性与可扩展性受到限制。

---

## 457. GPU-Accelerated Construction of the Gabriel Graph: Parallelization Strategies and Applications to Large-Margin Classification and Graph-Based Clustering

**arXiv ID:** 2610.28759 | [PDF](https://arxiv.org/pdf/2610.28759v1)

**作者:** Diego Sanches Nere dos Santos `[一作]` (Federal University of Ouro Preto), Luiz Carlos Bambirra Torres `[通讯]` (Federal University of Ouro Preto)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本论文提出了一套GPU并行化技术，实现了Gabriel图的高效构造，并将其应用于大边界分类器和无参数聚类算法；

**💡 创新点**

创新点包括：1）闭式O(1)线程索引映射实现无额外内存的顶点对恢复；2）基于位掩码的两遍压缩算法将O(N²)标记压缩至O(|E|)存储；3）针对不同数据集自适应的线程粗化策略；4）完整GPU实现的Gabriel图聚类框架，自动阈值切割并使用并行Union–Find。

**🔧 技术方法**

主要技术：CUDA并行计算、SIMT执行模型、位掩码压缩、原子操作、线程粗化、并行排序与最小距离计算、GPU Union–Find及层次合并。

**📊 数据集**

实验使用14个公开数据集（N从72到1372，特征数3到60）以及两大规模数据集（N≈10⁵）验证性能；

**📈 对比分析**

与单线程CPU基线相比，GPU实现加速幅度从237×到8484×，分类准确率和F1分数保持一致；聚类结果在Dunn指标上与DBSCAN相当，且不需要密度半径超参数。

**⚠️ 局限性**

局限性：仅在单GPU上测试，未与多线程CPU或多GPU方案比较；位掩码对极大N（10⁵以上）可能超出单GPU内存；层次合并步骤仍含顺序依赖，需进一步优化。

---

## 458. WRDO: Weighted Refusal Direction Optimization in Aligned Language Models

**arXiv ID:** 2610.29555 | [PDF](https://arxiv.org/pdf/2610.29555v1)

**作者:** Grigoriy Evseev `[一作]` (Moscow Independent Research Institute of Artificial Intelligence), Aleksandr Beznosikov `[通讯]` (Moscow Independent Research Institute of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Weighted Refusal Direction Optimization (WRDO)，通过对每条有害提示产生的“拒绝方向”进行加权聚合，并用两种黑盒优化器（可分离 CMA‑ES 与 GRPO‑IS）学习权重，实现对模型拒绝行为的精准干预。

**💡 创新点**

创新点在于：①把拒绝机制拆解为多维的 per‑prompt 方向，而非单一均值差异；②通过学习权重自适应聚合这些方向；③在不改动模型结构的前提下，只做一次权重空间的正交化修剪，避免推理时开销；④提出了两种高效的无梯度/强化学习优化策略。

**🔧 技术方法**

使用的技术包括：Transformer 余弦残差流方向提取、权重空间正交化干预、可分离 CMA‑ES 以及带重要性采样的 Group‑Relative Policy Optimization (GRPO‑IS)。

**📊 数据集**

采用六大公开对抗/危害基准（AdvBench、MaliciousInstruct、TDC2023、HarmBench、JailbreakBench、StrongREJECT）作为训练/验证集，基准模型为 Qwen3‑8B‑Base、OLMo‑3‑1025‑7B 与 Olmo‑3‑7B‑Instruct；使用 WildGuard、Qwen3Guard‑Gen‑8B、Llama‑Guard‑3‑8B 作为评估的安全判定器。

**📈 对比分析**

与 Basic RD、Heretic、RDO、RDO‑CONE 及 Uniform‑avg WRD 等基线对比，WRDO（尤其是 WRD‑GRPO‑IS）在三种判定器上的攻击成功率 (ASR) 均明显提升（最高可达 0.97‑0.99），并在大多数情形下保持了对模型通用能力的较好保留；WRD‑CMAES 在攻击性能上更强，但对通用性能影响较大；GRPO‑IS 在攻击与通用性能之间取得最佳平衡。

**⚠️ 局限性**

局限性：①实验仅覆盖 7–8B 参数规模模型；②WRD‑CMAES 目前未加入 KL 正则化，易导致通用性能下降；③权重聚合方法仍可能对训练使用的安全判定器产生过拟合，跨判定器迁移性有待进一步验证；④需要预先构建大量 per‑prompt 方向，构建成本较高。

---

## 459. Decision language models as affordable and rapid alternatives to LLM judges of clinical generative AI

**arXiv ID:** 2610.28826 | [PDF](https://arxiv.org/pdf/2610.28826v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 460. Compilation Is Not Correctness: Scan-by-Scan Evaluation of LLM-Generated PLC Programs

**arXiv ID:** 2610.28675 | [PDF](https://arxiv.org/pdf/2610.28675v1)

**作者:** Jinyun Chen `[一作]` (Beihang University), Rui Wang `[通讯]` (Beihang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了CycleBench基准，通过逐扫描执行对LLM生成的PLC Structured Text（ST）程序进行行为级别的评估，并基于此开发了CycleGen生成框架，显著提升了程序的执行通过率；

**💡 创新点**

创新点在于：①创建了首个基于扫描逐步比较的PLC生成评估基准CycleBench；②设计了CycleGen，结合多候选生成、编译器反馈修复与运行时自测，实现了无参考代码的执行级别改进；

**🔧 技术方法**

主要技术包括大语言模型（GPT‑5.5、GPT‑4o、Claude Sonnet 4等）生成ST代码，使用Matiec前端将ST翻译为C进行编译，结合编译器诊断进行单步修复，利用自测套件对候选程序进行运行时行为对比，最后通过排序选择最佳候选；

**📊 数据集**

使用了由GitHub、AutoPLC库、Agents4PLC、Matiec示例等多来源收集的301个PLC任务及其对应的参考程序、接口与输入场景，共计684个测试场景、61,424个扫描；

**📈 对比分析**

对比方法包括八种LLM模型与四种基线（Agents4PLC、LLM4PLC、AutoPLC、Direct）以及CycleGen的不同模块 ablation，实验显示CycleGen将执行通过率从74.63%提升至83.31%，相较最优基线提升约6.98个百分点，编译通过率与执行通过率之间存在约12个百分点的差距；

**⚠️ 局限性**

局限性包括：①评估仅基于固定的输入场景，未覆盖所有潜在需求和边界情况；②自测套件本身可能不足以捕捉所有行为缺陷，导致漏判；③CycleGen的多候选生成与自测消耗大量tokens，且对不同PLC方言和物理控制器的迁移性未充分验证；

---

## 461. Energy-Efficient Timely Relevant IoT Data Retrieval with Wireless Memory Approximation

**arXiv ID:** 2610.29236 | [PDF](https://arxiv.org/pdf/2610.29236v1)

**作者:** Junya Shiraishi `[一作]` (National Institute of Information and Communications Technology), Petar Popovski `[通讯]` (Aalborg University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于拉取的框架，通过无线内存激活与逼近，联合调度激活时机、刷新周期、模型尺寸和语义阈值，实现对IoT设备数据的及时且相关的检索，同时满足能耗与信息新鲜度的约束。

**💡 创新点**

创新点在于：①将无线内存激活与逼近与语义过滤、模型容量调优、通信调度三者联合优化；②提出基于Goal‑oriented Effectiveness（GoE）的时效性度量，并给出其闭式期望表达式；③通过理论与蒙特卡罗仿真验证，在满足相同新鲜度约束下，比始终开启的基线显著降低能耗。

**🔧 技术方法**

采用了：DRAM保持错误概率模型、轻量级神经网络推理能耗模型、语义相似度滤波、基于AOD/GoE的目标度量、网路擦除信道模型、离散网格搜索与随机模拟。

**📊 数据集**

在实验验证中使用ImageNet验证集配合MCUNet模型，评估不同模型尺寸下的误差分布与能耗特性。

**📈 对比分析**

与始终开启内存、刷新周期为τ₀的基线对比；通过计算能耗比η和GoE约束，结果显示当Δ_th>150时，所提方法能耗可降低至基线的约30%–70%，且GoE始终满足目标。

**⚠️ 局限性**

局限性包括：假设误差独立且相似度分布为均匀、未考虑观测间的时间相关性、仅关注单设备场景、并未在真实硬件上验证；未来工作需开展实验验证和鲁棒设计。

---

## 462. A Matroidal Framework for Random Access in Linear Codes

**arXiv ID:** 2610.28820 | [PDF](https://arxiv.org/pdf/2610.28820v1)

**作者:** Daniella Bar-Lev `[一作]` (Universit"at Z"urich), Flavio Salizzoni `[通讯]` (Universit"at Z"urich)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在线性码中的随机访问恢复问题，定义并分析恢复平衡码（即每个坐标的期望恢复样本数相同），并提出恢复平衡的 matroid 等价形式。

**💡 创新点**

证明恢复平衡是 matroid 的性质，并且在无环无共环（无 loop/coloop）的情形下保持对偶性；从而证实了 Gruica 等人提出的“恢复平衡在对偶码中保持”猜想；进一步证明在稀疏铺垫 matroid（sparse paving matroid）族中，随着元素数 n 递增，几乎所有 matroid 都不是恢复平衡的。

**🔧 技术方法**

采用 matroid 理论（闭包、循环、对偶、独立集）、组合计数（Johnson 图、二进制码、最大码大小估计）以及经典的 coupon‑collector 期望公式来推导期望恢复时间与 matroid 结构的关系，并利用对偶性公式得到恢复平衡的对称性。

**📊 数据集**

本工作为纯理论分析，没有使用实验数据集；主要利用 combinatorial 计数公式和已知的 matroid/码的属性作为“数据”。

**📈 对比分析**

对比方法：与以往只针对特定码族（如 MDS、Hamming、Reed‑Muller 等）得到的恢复时间结果相比，本文给出了针对任意线性码的通用证明；同时通过稀疏铺垫 matroid 的计数结果表明恢复平衡性在随机 matroid 中极为罕见，理论上具有较高的通用性。

**⚠️ 局限性**

局限性：对偶性结论仅在码无环无共环（即 matroid 无 loop/coloop）时成立；稀疏铺垫 matroid 的结论并不直接推广到所有 matroid，且可表示 matroid 在所有 matroid 中占比趋近 0，因而对可表示码族的普适性仍未完全阐明。

---

## 463. Linear Twisting Sidewinding Gaits for Twistable Snake Robot Motion Planning on Complex Terrains

**arXiv ID:** 2610.29461 | [PDF](https://arxiv.org/pdf/2610.29461v1)

**作者:** Qingling Fan `[一作]` (Shanghai Jiao Tong University), Zhongqiang Ren `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种线性扭曲侧向行走（Linear Twisting Sidewinding, LTS）步态，并开发了基于背骨曲线的积分方法来生成该步态的关节角度；在仿真和真实机器人上验证了该步态能够均匀分布扭矩负载、降低单个扭曲关节的峰值功率，并在多种复杂地形上实现比传统侧向行走更快的通过速度。

**💡 创新点**

创新点包括：①引入线性扭曲函数，将扭曲运动在整条身体上均匀分布，显著降低单关节负载；②提出分段近似积分方法，解决原有积分方法在存在扭曲时出现的数值不稳定问题；③在同等总功率下将扭曲功率从 96.8% 降至 45.1%，并实现 45%–49% 的通过时间提升。

**🔧 技术方法**

主要技术手段有：背骨曲线与背侧参考曲线方法、线性扭曲函数设计、分段近似积分算法、MuJoCo 仿真、真实机器人硬件实验（含扭矩传感与编码器数据采集），以及多地形（石板、草地、碎石、斜坡）下的性能评估。

**📊 数据集**

未使用公开数据集；实验数据来自作者自制的 6 模块蛇形机器人，配合在四种地形（石板、草地、碎石、20° 斜坡）上的实测通过时间与功率数据。

**📈 对比分析**

对比方法：与传统滚动向量（Rolling Vector）方案在相同总功率下比较扭矩负载分布；与传统侧向行走方案在相同步频下比较通过时间。实验结果显示：LTS 的单关节峰值功率下降 42%，并在所有地形上将通过时间缩短 45%–49%。

**⚠️ 局限性**

局限性：①对原积分方法数值不稳定原因的理论解释尚未完整；②目前仅将线性扭曲融合到侧向行走步态，未探索与其他步态（如波浪、翻滚等）的联合使用；③实验主要聚焦于平坦或轻度复杂地形，未系统评估在极端环境或障碍物丰富环境中的表现。

---

## 464. Near-Optimal Consistency-Robustness Bounds for Non-Clairvoyant Scheduling

**arXiv ID:** 2610.29037 | [PDF](https://arxiv.org/pdf/2610.29037v1)

**作者:** Sizhe Li `[一作]` (Johns Hopkins University), Tongxin Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d`

**🎯 论文内容**

提出了单机非预知调度在有预测情况下一致性–鲁棒性折衷的近似最优表述，并给出了新的下界与上界。

**💡 创新点**

创新点在于：① 引入多阶段对手构造并将其转化为非线性优化问题，显著提升下界；② 设计了 Modified Two-Stage Schedule 算法，通过精确校准一致性预算实现上界与下界之间的 3/2 乘数；③ 推出了可容错的预测模拟框架，实现对小误差的平滑降级。

**🔧 技术方法**

采用优化理论（线性、线性分数、双线性方程组）、对手攻击模型、凸几何（锥分解）和多阶段优化相结合的技术。

**📊 数据集**

论文为理论性工作，未使用任何实验数据集；所有结果均通过数学证明获得。

**📈 对比分析**

与以往的下界（Wei–Zhang）及上界（Preferential Round‑Robin、先前的 Two‑Stage）比较，新的下界在 n≥5 时严格更强；上界与下界相距不超过 3/2，且在极限 n→∞ 时达到最优的 √(1/λ) 规模；闭合了最坏 n 情形下的多项式 gap。

**⚠️ 局限性**

局限性包括：对有限 n≥5 的精确最优常数仍未收敛；结果仅适用于无权重单机调度；未验证在实际调度场景中的性能；对其他预测模型（如部分预测、排列预测）仍需进一步研究。

---

## 465. Better Policies, Worse Verifier Audits

**arXiv ID:** 2610.29029 | [PDF](https://arxiv.org/pdf/2610.29029v1)

**作者:** Daniel Cieślak `[一作]` (Gdańsk University of Technology), Jacek Łubiński `[通讯]` (Gdańsk University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在强化学习与后期训练系统中，基于完整任务标签对组件验证器进行的“on‑policy”审计所导致的失效链条：当策略逐步掌握某个子目标时，审计的整体 AUC 逐渐失去对该子目标的辨识能力，甚至可能偏好信息混杂的验证器；随后对验证器的刷新会导致失效支持消失，从而在离线和跨生成器情境下导致组件性能衰退和下游约束违规。通过理论推导（混合律、完整消除阈值、审计反转阈值）和四种实验场景（引擎精确 Oracle、GSM8K 文本、GoEmotions 多标签、ECG 信号）验证了这一现象，并提出了通过保持/重构失败支持的两混合审计修复方法。

**💡 创新点**

创新点包括：①首次系统性揭示“on‑policy”审计崩溃及其对刷新策略的破坏性影响；②给出精确的混合律和完整消除阈值，证明审计指标在特定失败混合下完全失去辨识性；③提出审计反转阈值，解释为什么信息混杂的验证器会在整体 AUC 上优于纯粹的子目标验证器；④设计了支持保持的两混合审计机制，能够在失去自身失败样本时恢复对组件性能的有效估计。

**🔧 技术方法**

技术手段包括：ROC/AUC 分解、部分识别与裁剪边界、混合矩阵逆推、两混合重加权审计、基于失败桶的识别区间估计、离线刷新实验设计以及统计显著性检验（bootstrap、sign test）。

**📊 数据集**

使用的数据集与基准有：引擎（engine）精确 Oracle 对比实验；GSM8K 语言生成任务；GoEmotions 多标签情感分类；ECG（PhysioNet MIT‑Bih）信号处理；AG News 语料构造对照。

**📈 对比分析**

对比方法：基线 Qwen2.5 + DPO 策略下的组件验证器与多种刷新策略（全局标签刷新、组件标签刷新、混合刷新）。实验结果显示：1）整体 AUC 与组件 AUC 失配，导致错误的刷新决策；2）全局标签刷新显著降低离线与跨生成器的组件 AUC 并提升约束违规；3）混合刷新通过重注入原始失败样本恢复了组件 AUC 与下游性能，证明支持保持修复有效。整体性能评估表明：在未修复的情况下，组件 AUC 可从 0.9 降至 0.5；刷新后可恢复至 0.99 以上。

**⚠️ 局限性**

局限性：需要可信的组件标签；实验仅在离线刷新场景下验证，未覆盖在线 RL 场景；验证器规模受限于可用计算，未验证大规模 LLM 级验证器；实验多聚焦于特定任务（几何引擎、文本生成、情感分类、ECG 信号），通用性待进一步验证；同时，重加权审计在实际部署中需额外的失败样本标注与重采样成本。

---

## 466. Extending FAIR Principles with Observability for Trustworthy Federated Learning

**arXiv ID:** 2610.29252 | [PDF](https://arxiv.org/pdf/2610.29252v1)

**作者:** Hangli Ge `[一作]` (University of Tokyo), Noboru Koshizuka `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 FAIR-O 观测框架，在联邦学习中通过本地最小化签名证据实现可信度评估，并实现了基于 TinyLLM 的可解释性支持。

**💡 创新点**

将 FAIR 原则扩展为 FAIR-O，增加观测维度，设计证据平面、TinyLLM 辅助本地解释以及基于三层信任模型的动态可信度评估，形成全新的可信联邦学习可信度体系。

**🔧 技术方法**

采用分布式证据生成与 HMAC 签名、TinyLLM（Qwen2 1.5B 量化模型）本地推理、可视化仪表盘、FedAvg 联邦学习算法以及数据空间层面的 Evidence Plane 组件。

**📊 数据集**

实验一使用合成资产目录（六项与八项资产），实验二使用 UCI SECOM 半导体制造数据集。

**📈 对比分析**

通过仪表盘展示各参与方的 FAIR-O 评分、信任指标，并对比单方训练、两方 FedAvg 训练的准确率曲线；实验表明，证据生成不泄露原始数据，FedAvg 在质量不均衡时仍能收敛，但未将证据用于权重调整。

**⚠️ 局限性**

证据未与联邦学习流程绑定，无法实现自动化参与者选择或加权；TinyLLM 仅为可选辅助，且实验在单机环境下运行；缺乏大规模多方验证与性能评估。

---

## 467. Where Do Legal Factors Live in LLMs?

**arXiv ID:** 2610.28799 | [PDF](https://arxiv.org/pdf/2610.28799v1)

**作者:** Li Zhang `[一作]` (University of Pittsburgh), Kevin Ashley `[通讯]` (University of Pittsburgh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究大型语言模型（LLM）内部激活是否能读取法律专家定义的26个CATO因素；

**💡 创新点**

证明LLM内部激活中存在可读的法律因素信息，并通过稀疏自编码器（SAE）定位单个或少数特征即可完成此任务；

**🔧 技术方法**

使用线性探测器（linear probe）与稀疏自编码器（SAE）读取技术，对Qwen3-1.7B、Qwen3-8B与Gemma-2 2B三种LLM的残差流进行分析；

**📊 数据集**

构造控制性自然语言短篇，覆盖26个CATO因素（约6992正样本+901负样本），并由多模型审查和人工审核验证标签；

**📈 对比分析**

比较方法：在测试集上对每个因素使用线性探测器计算AUC，随机选择5个SAE特征与选定特征做对比；结果显示：所有26因素在三模型中的最大AUC均在0.998–1.000之间，单一SAE特征AUC可达0.86–1.00，5特征AUC达到0.965–0.988，几乎与全向量一致；

**⚠️ 局限性**

局限性：仅针对受控的英语贸易秘密情景，未验证真实司法文本；模型规模相对较小；缺乏多语言和多法律领域验证；对样本长度、用词多样性和因素共存的处理不足；

---

## 468. Decodability Is Not Controllability: Local and Anisotropic Speaker Control in Continuous-Latent TTS

**arXiv ID:** 2610.29345 | [PDF](https://arxiv.org/pdf/2610.29345v1)

**作者:** Yuanzhong Zheng `[一作]` (VoxNexus), Youhai Jiang `[通讯]` (VoxNexus)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对 VoxCPM2 与 dots.tts 两种连续潜变量 TTS 模型进行等范数激活干预，系统研究了隐藏层状态的可解码性与对目标说话人声效的可控性之间的关系，并证明高线性可解码度并不保证可实现目标说话人控制。

**💡 创新点**

①揭示了隐藏层深度与说话人控制可行性的尖锐分离；②提出将“几何特异性 + 读取通道可达性”作为实现无训练说话人控制的设计原则；③通过等范数方向与位置实验分离了控制行为的“行为杠杆”与“目标特异性”。

**🔧 技术方法**

线性探测器、等范数激活干预、注意力驱动的读取通道分析、ECAPA‑TDNN/ CAM++ 说话人嵌入、Whisper WER、抽样重置的因果中介分析。

**📊 数据集**

30 名英语 LibriTTS‑R 说话人（共 60 个有向二元组），VoxCPM2 与 dots.tts 的预训练模型，以及 held‑out 说话人原始音频与五句原型。

**📈 对比分析**

比较指标包括：目标相似度提升（target gain）、目标占优计数（target dominance）、源相似度变化、WER 变化以及说话人检索 Top‑1 率。实验结果显示：在 L0 层对齐干预实现 60/60 目标占优；在 L7 层即使线性探测准确率 0.970，目标占优率降至 0/60；在 L3 层得到 52/60；在 dots.tts 中所有层的目标占优率均为 0/60，但对齐干预可显著降低源相似度。

**⚠️ 局限性**

仅在两种特定的连续潜变量 TTS 模型与 LibriTTS‑R 语料库上验证，缺乏跨模型和跨语言的泛化评估；等范数干预仅探讨了方向与位置对控制的影响，未覆盖更大幅度或非线性编辑；实验对计算资源需求高，实际部署难度尚未评估。

---

## 469. Beyond the Vote: When Do LLMs Add Value to Retrieval-Based Tabular Classification?

**arXiv ID:** 2610.29520 | [PDF](https://arxiv.org/pdf/2610.29520v1)

**作者:** Ziquan Wang `[一作]` (Beihang University), Yingjie Li `[通讯]` (Beihang University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在19个公开表格分类数据集上，使用相同的检索邻居，系统性比较了四种冻结的大型语言模型（LLMs）与基于距离加权的最近邻投票（Distance-KNN）在检索式上下文学习（ICL）中的表现；

**💡 创新点**

通过共享邻居评估协议首次量化了LLM推理在相同检索证据下相较于传统投票法的增值，揭示了LLM在低标签共识且特征距离适中的邻域内可能超越投票，而在高共识或远距离邻域中则不具优势；

**🔧 技术方法**

采用距离加权最近邻投票、基于特征空间和文本嵌入的检索、四种开放权重LLM（Qwen3-8B、Qwen3-32B、Gemma3-12B、Gemma3-27B）、不同提示格式（KV、NL、规则增强NL）以及标签平衡演示选择等技术；

**📊 数据集**

UCI Machine Learning Repository与OpenML的19个二分类或多分类表格数据集；

**📈 对比分析**

对相同检索邻居进行的“共享邻居”对比，结果显示Distance-KNN在所有候选池大小下均保持最高平均准确率（约85%），LLM在低共识区间可获得最多约3.6个百分点的提升，但整体上LLM未能超过投票法；

**⚠️ 局限性**

仅评估冻结LLM在相同检索邻居上的推理能力，未涉及任务专门微调或端到端优化；实验范围局限于19个公开数据集，未验证在更大规模或极度专业化领域的推广性。

---

## 470. Supervision-Adaptive Curriculum and Guarded Anti-Curriculum Learning for Dense Prediction

**arXiv ID:** 2610.29519 | [PDF](https://arxiv.org/pdf/2610.29519v1)

**作者:** Chunming He `[一作]` (Duke University), Sina Farsiu `[通讯]` (Duke University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PaceShift训练框架，结合监督自适应课程学习和有保障的反课程学习，提升密集预测模型的准确性与鲁棒性。

**💡 创新点**

将难度与监督可靠性分离，采用四种监督可靠性指标构建动态课程，并在后训练阶段引入多半径频谱挑战与基于可靠性与一致性的偏好判定，既解决纹理捷径又避免过度自信。

**🔧 技术方法**

利用学习动态视角、时间统计权重、像素不确定性加权、增广一致性、模型不一致性与熵，结合频域低通滤波、参考模型对比、配对偏好损失、KL对齐等技术。

**📊 数据集**

在多种上下文耦合密集分割任务（camouflaged、transparent、concealed defect、polyp）以及更广泛的医学、工业、语义/实例分割、光照腐败等数据集上验证。

**📈 对比分析**

对比原始基线、CurriSeg以及多种自监督、弱监督、基础模型适配方案；PaceShift在所有任务上均提高平均3–5%指标，部分模型提升超过10%，且显著提升校准与噪声鲁棒性。

**⚠️ 局限性**

对高频信息依赖的场景可能削弱细节；需要额外训练时间和外部教师/参考掩码；当样本可靠性差异不大时提升有限。

---

## 471. Dissipation Bounds and Hardness in Globally Asymptotically Stable Quadratic Systems

**arXiv ID:** 2610.29512 | [PDF](https://arxiv.org/pdf/2610.29512v1)

**作者:** Subhransu S. Bhattacharjee `[一作]` `[通讯]` (Australian National University), Subhransu S. Bhattacharjee (Australian National University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了全局渐进稳定的二次系统在有限时间内能量耗散的最坏情况，并证明了即便已给出严格Lyapunov函数，估计保留能量分数、半化时间及系数距离仍是NP难的。

**💡 创新点**

创新点在于构造了基于ONE-IN-THREE SAT的多项式归约，提出了残差-距离恒等式与最小轨道的逆平方下界，证明了在系统结构、扰动或指数阻尼下保持增量硬性与多项式因子逼近硬性。

**🔧 技术方法**

主要技术包括多项式归约、四次Lyapunov函数构造与SOS分解、实数代数几何的∃ℝ/∀ℝ完备性分析、系数距离估计以及直接阻尼与能量恒等式的解析。

**📊 数据集**

使用的“数据集”是从ONE-IN-THREE SAT问题产生的合成实例，这些实例通过符号化的矩阵和多项式映射转化为二次系统。

**📈 对比分析**

方法与性能比较是理论性的，作者证明任何多项式时间算法在加性误差小于1/2或多项式因子逼近时都无法成功，因而没有实测性能指标。

**⚠️ 局限性**

局限性包括仅适用于特定的二次结构、使用欧几里得能量在全状态单位球上，维度随实例增长，且对更一般的非线性系统、不同范数或更紧凑的上界尚无可行性。

---

## 472. Matrix $Φ$-Entropic Measures of Dependence

**arXiv ID:** 2610.29576 | [PDF](https://arxiv.org/pdf/2610.29576v1)

**作者:** Chenyu Wang `[一作]` (Chinese University of Hong Kong), Amin Gohari `[通讯]` (Chinese University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文构造了矩阵级别的Φ-秤带（Φ-ribbon）与Φ-强数据处理不等式常数（SDPI constant），并证明其张量化与数据处理性质，同时给出多种源的显式计算和独立结构下的内部下界。

**💡 创新点**

创新点在于将标量Φ秤带推广到矩阵情形，提出新的矩阵Φ-熵不等式，并证明矩阵Φ-ribbon完全刻画对应的矩阵Φ-SDPI常数；此外发现矩阵SDPI常数可超过标量对等值，并给出多维量化的内外界定。

**🔧 技术方法**

主要技术包括矩阵函数分析（Fréchet导数、二阶导数、Schur乘积）、子加性类Φ的凸性与可逆性条件、矩阵熵与相对熵的不等式、超模性与张量化论证，以及对常量如双对称二进源、i.i.d.求和的具体计算。

**📊 数据集**

由于研究为理论性质，论文未使用实验数据集，所有结果均为解析证明或构造性示例。

**📈 对比分析**

与标量Φ-ribbon及传统HC、最大相关等量化方法对比，本文展示了矩阵版在数值上能取得更大常数，且通过示例说明在多维度下性能优于单维度；在无信号盒的wirings情形中，证明了矩阵Φ-ribbon的单调性与张量化。

**⚠️ 局限性**

局限性包括：目前尚未解决矩阵SDPI常数随维度递增是否严格增大的问题；对于非子加性类的Φ，矩阵方法的适用性与可扩展性仍有待进一步研究；论文中所有结果均假设离散有限分布，对连续或无限维情形未给出推广。

---

## 473. Private Online Prediction from Experts with Small Losses

**arXiv ID:** 2610.29575 | [PDF](https://arxiv.org/pdf/2610.29575v1)

**作者:** Bo Li `[一作]` (HKUST Fok Ying Tung Research Institute), Peng Ye `[通讯]` (HKUST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出了一种新的差分隐私算法，用于在线专家预测问题，能够在低损失情况下实现更优的后悔界限。

**💡 创新点**

这是首个在差分隐私约束下实现后悔界限为 Õ(√(L^⋆)) 的算法，显著改善了现有算法在小损失情况下的表现。

**🔧 技术方法**

使用了差分隐私机制和乘法权重方法，结合了延迟更新策略以满足隐私要求。

**📊 数据集**

假设输入序列由一个无知的对手选择，具体数据集未明确给出。

**📈 对比分析**

与现有算法相比，当 L^⋆ 显著小于 T 时，提出的算法在后悔界限上有显著改善，能够达到 Õ(√(L^⋆)) 的后悔界限，而现有算法通常为 Õ(√(T))。

**⚠️ 局限性**

算法假设输入序列是固定的，且在自适应输入情况下，后悔界限仍然是 Ω̃(√(T))，即在自适应对手下无法实现更好的性能。

---

## 474. Dynamically optimal weight-balanced binary search trees

**arXiv ID:** 2610.29574 | [PDF](https://arxiv.org/pdf/2610.29574v1)

**作者:** John Iacono `[一作]` (Université libre de Bruxelles), Akash Pareek `[通讯]` (Ashoka University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种自调整二叉搜索树 PopTree，能够在任意键无关（key‑independent）搜索序列上实现与离线最优相同的渐进运行时间；同时证明所有 BB[α]（权重平衡）树在键无关环境下的最优时间等价于工作集界。

**💡 创新点**

创新点在于：
• 第一次给出在保持 BB[α] 平衡约束的 BST 类中实现动态最优（与离线最优匹配）的结构；
• 通过“跳表‑PopList → PopBTree → PopTree”三步变换，构造了仅使用单旋转、保持 BB[α] invariant 的自调整 BST；
• 将键无关最优与工作集属性联系起来，证明任何随机键置换下的离线最优与工作集界相当，从而实现键无关最优。

**🔧 技术方法**

技术手段包括：
• 权重平衡树（BB[α]）的级别与深度理论；
• 设计 SkipPopList（具有占用度约束的跳表）来维持最近访问的元素靠近顶层；
• 通过一次性旋转（push、pop、promotion、demotion）恢复平衡；
• 将 SkipPopList 转换为多路搜索树 PopBTree，再映射为标准 BST PopTree；
• 关键在于维护单个节点级别不变，利用级别与深度的常数关系。

**📊 数据集**

实验数据集：本文以理论模型为主，主要在任意搜索序列 X 以及随机键映射 f([n]) 的设定下证明渐进性质；未给出具体实际数据集。

**📈 对比分析**

比较方法：
• 通过理论分析证明 PopTree 的总成本 ≤ O(OPT(X))，即与离线最优（保持 BB[α] invariant）相当；
• 在键无关模型下进一步证明 OPT(f(X)) 的期望等价于工作集界，从而说明 PopTree 取得键无关最优；
• 与已知的 splay 树、tango 树等结构比较，表明 PopTree 在保持平衡约束时实现了动态最优，且对键无关搜索具有最佳渐进性能。

**⚠️ 局限性**

局限性：
• 仅在 BB[α]（权重平衡）树类中保证最优，对更一般的旋转 BST（如 splay 树）仍不适用；
• 算法实现复杂，涉及多级跳表和多路树的变换；
• 结果为渐进理论性分析，缺乏实际实现和实验评测；
• 对 α 的取值有一定限制，实际性能受平衡因子影响。

---

## 475. Generalized CycleGRPO for Fine-Grained Multimodal Understanding and Grounding

**arXiv ID:** 2610.29552 | [PDF](https://arxiv.org/pdf/2610.29552v1)

**作者:** Xin Zhang `[一作]`, Dacheng Tao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

发展了 Generalized CycleGRPO 机制，实现无参考文字、无外部评判的区域‑文本‑区域闭环训练，统一图像、视频区间及时空轨迹的描述与定位任务。

**💡 创新点**

将 CycleGRPO 从二维图像扩展到时空连续区间和稀疏轨迹；用自回归策略同时生成描述并作为自身评判器进行回归；引入 DOVE 指标评测描述的区分度。

**🔧 技术方法**

基于多模态大型语言模型的自回归策略、GRPO (Group‑Relative Policy Optimization)、多层 rollouts、区域序列化与一致性函数以及对抗约束防止信息泄漏。

**📊 数据集**

Spatial：DLC‑Bench、GroundingSuite、GAR‑Bench‑VQA、GCG、GRES；Temporal：Charades‑STA、ActivityNet、QVHighlights、TimeLens、TVGBench、DOVE；Spatio‑Temporal：Open‑o3‑Video、V‑STaR、VideoMME、WorldSense、VideoMMMU。

**📈 对比分析**

与对应基准模型（SAMTok、Open‑o3‑Video‑4B、Qwen3‑VL‑4B 等）对齐，多指标评估；空间任务提升 10–20%，时间任务提升 1–3% mIoU 与 TEC 约 18%，时空任务提升 2–4% mLGM、V‑STaR 与 TVGBench 约 7–10%；整体表现稳步提升。

**⚠️ 局限性**

仅通过几何重构评估，无法保证句子完整性、流畅性与事实正确性；对初始定位性能依赖较高，定位不佳会产生噪声学习信号；计算成本高（N×K rollouts）；DOVE 只覆盖视觉可区分事件，未涵盖音频、边界检测等场景。

---

## 476. Hyperspherical Simplex Representations for Backward-Compatible Retrieval

**arXiv ID:** 2610.29550 | [PDF](https://arxiv.org/pdf/2610.29550v1)

**作者:** Simone Ricci `[一作]` (University of Florence), Alberto Del Bimbo `[通讯]` (University of Florence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出通过软max输出和logit推导的超球面单纯形表示（PSP和LSP），实现跨模型更新的向后兼容检索；

**💡 创新点**

无需兼容性专门训练、映射函数或架构修改，利用softmax交叉熵训练自带的单纯形几何；通过确定性线性变换补偿类扩展导致的角度失配；提供理论分析证明在vMF分布下满足兼容性不等式；

**🔧 技术方法**

softmax投影、L2归一化、确定性线性角度校正、von Mises–Fisher分布期望分析、top‑k稀疏化；

**📊 数据集**

CIFAR‑10/100、ImageNet‑1K、Google Landmarks v2、Places365、公开预训练模型序列（AlexNet→ResNet‑50→RegNet→ResNet‑152→MaxViT、ViT‑B/32→ViT‑B/16→ViT‑L/16）等；

**📈 对比分析**

在标准兼容性基准（扩类、扩骨干、扩类+骨干）与公开预训练模型上，与BCT、RACT、AdvBCT、LCE、CoReS等方法比较。PSP和LSP在多次更新、不同数据集上均保持或超过兼容性指标（AC、AA、ACA），并在top‑k稀疏化下保持高检索准确率，优于基线并与最先进方法竞争；

**⚠️ 局限性**

仅适用于softmax分类训练；假设标签空间一致且类顺序不变；依赖vMF分布和浓度增大假设，理论仅在期望上成立；对非监督或回归目标模型不可直接应用；

---

## 477. On the Effectiveness of Snapshot Testing: An Empirical Study

**arXiv ID:** 2610.29549 | [PDF](https://arxiv.org/pdf/2610.29549v1)

**作者:** Shun Fujita `[一作]` (Nara Institute of Science and Technology), Hajimu Iida `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对前端JavaScript项目中快照测试的有效性进行了大规模实证研究，结合代码覆盖率、变异检测和开发者问卷。

**💡 创新点**

创新点在于同时量化快照测试与传统单元测试的互补作用，并通过问卷洞察实践中的收益与痛点。

**🔧 技术方法**

使用了Jest快照功能、Stryker变异测试、AST分析及在线问卷调查等技术。

**📊 数据集**

数据集来源为13,256个星级≥500的GitHub仓库，其中1,355个使用Jest、2,568个使用快照测试。

**📈 对比分析**

通过比较仅快照、仅非快照和两者合并的测试，发现合并后语句覆盖率中位数显著提升，变异检测率提高数倍。

**⚠️ 局限性**

局限性包括仅聚焦Jest/JavaScript生态、采用固定测试配置、只使用Stryker变异器，以及对跨平台差异缺乏评估。

---

## 478. Melange: A Synthetic Visual Bias Benchmark for Multi-modal Large Language Models

**arXiv ID:** 2610.29537 | [PDF](https://arxiv.org/pdf/2610.29537v1)

**作者:** Razieh Rezaei `[一作]` (Technical University of Munich), Masoud Jalili Sabet `[通讯]` (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 Melange，一套包含 12,640 场景、126,400 张合成图像的多模态偏见基准。

**💡 创新点**

创新点在于将 BBQ 的文本偏见场景自动转化为视觉场景，并通过两阶段 VQA + 人工验证确保图像与文本对应、属性可视化。

**🔧 技术方法**

使用扩散模型 Flux 生成图像、视觉问答模型进行自动筛选、人工审核、以及 FaiR 等偏见评估指标。

**📊 数据集**

基于 BigBench 的 BBQ 数据集及自行生成的 Flux 图像。

**📈 对比分析**

对 23 种公开 MLLM 进行三种输入设置评估，发现文本→视觉的公平性平均下降 0.291，提升后可部分恢复；性能受模型家族而非规模影响。

**⚠️ 局限性**

局限包括仅包含模糊情景、缺乏可辨别的宗教/SES 评价、使用合成图像可能产生分布差异、未控制细微视觉混淆。

---

## 479. RCSD: Reinforcement and Continual Learning for Domain-Specific Speculative Decoding

**arXiv ID:** 2610.29527 | [PDF](https://arxiv.org/pdf/2610.29527v1)

**作者:** Mingkang Liu `[一作]` (Hisense), Lichen Pang `[通讯]` (Hisense)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

重新训练草稿模型，使其在speculative decoding过程中最大化被目标模型接受的token长度

**💡 创新点**

将训练目标从分布匹配（KL散度）转为接受长度最大化，并通过DPO与GRPO自监督的偏好优化来实现

**🔧 技术方法**

使用Direct Preference Optimization (DPO)、Group-wise Relative Policy Optimization (GRPO)、masked loss以及强化学习与持续学习框架

**📊 数据集**

在Qwen3-1.7B和Llama-3.1-8B上使用ShareGPT、GSM8K、HumanEval、MT-Bench等数据集进行评估

**📈 对比分析**

与传统的HASS分布匹配基线对比，在不同speculative steps下平均接受长度提升35–40%，并实现1.2–1.35倍的推理速度加速，生成质量保持不变

**⚠️ 局限性**

局限性包括：对目标模型的内在反馈机制高度依赖，可能对不同目标模型泛化受限；masked loss实现复杂；实验仅覆盖两大模型与有限基准，缺乏更广泛的跨模型验证

---

## 480. X-Embedding: Two-Stage Training for Multilingual Patent Prior-Art Retrieval

**arXiv ID:** 2610.29521 | [PDF](https://arxiv.org/pdf/2610.29521v1)

**作者:** Wentao Wu `[一作]` (Patsnap), Yuanxiang Huangfu `[通讯]` (Patsnap)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种两阶段的多语言专利X‑category检索框架，先通过跨章节和跨语言家族的弱监督对齐学习专利语义表示，再用检索任务中的X‑category引用进行任务监督，最终生成可兼容64/1024维的嵌入。

**💡 创新点**

创新点在于：①将跨章节（ABS–CLAIM、CLAIM–DESC）与跨语言家族（同章节不同语言）对齐作为领域适配的弱监督源；②在此基础上对X‑category检索目标进行任务监督；③使用Matryoshka表示学习在同一模型中同时得到64维和1024维嵌入。

**🔧 技术方法**

主要技术包括：Qwen3‑Embedding‑0.6B编码器、InfoNCE对齐损失、X‑category监督的InfoNCE+MLR、Matryoshka表示学习、LoRA低秩适配、以及基于BM25与dense检索的混合策略。

**📊 数据集**

使用的数据集为包含中文、英文、日文三语的专利集合，约300k条候选专利，2025年以后提交的专利作为查询，共2.67M个X‑category正例与8个负例；同时使用了原始的Qwen3‑Embedding‑0.6B与RoBERTa作为基线。

**📈 对比分析**

与未适配的Qwen3‑Embedding‑0.6B、RoBERTa以及直接监督微调模型对比，64维下整体宏观Hit@100提升至89.09%，比直接微调高3.19个百分点；Hit@10/20提升3.71/3.79个百分点；相对RoBERTa整体提升3.68个百分点。性能提升在英文和日文更显著。

**⚠️ 局限性**

局限性：仅在单一专利集合和64维上评估；未进行统计显著性检验和多种随机种子实验；训练成本不平衡（两阶段相较直接微调计算量更大）；未分别消融两种弱监督源和MLR的贡献；未评估更高维度或其他专利集的泛化能力。

---

## 481. A Transfer Learning-Driven Methodology for Efficient and Sustainable Indoor Localization Using VLC

**arXiv ID:** 2610.29594 | [PDF](https://arxiv.org/pdf/2610.29594v1)

**作者:** Masood Jan `[一作]` (Sorbonne University), Alexander Artemenko `[通讯]` (Robert Bosch GmbH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于可见光通信（VLC）的迁移学习框架，用于提高工业环境下的室内定位精度与能效，并引入统一的模型效率（ME）指标。

**💡 创新点**

创新点在于：1) 结合迁移学习与噪声适应的模型；2) 设计四维综合评价指标ME，可同时衡量精度、能耗、数据利用率与迁移收益；3) 在BOSCH工厂真实场景下验证，证明TL可在仅30%数据时仍保持高精度。

**🔧 技术方法**

使用深度神经网络（DNN）与迁移学习技术；在训练中加入高斯噪声模拟环境漂移；采用均方误差、Adam优化器；评估指标包括定位误差、成功率、能耗、ME。

**📊 数据集**

BOSCH工厂生产线收集的VLC RSSI数据集，覆盖10个发射点、约1600个接收点，包含多种光照与障碍条件。

**📈 对比分析**

与基线无迁移模型和噪声增强模型进行对比；通过实验显示TL模型将定位误差从基线41.2cm提升至70.2cm（NF=2）时，仅提升0.4cm；在极端噪声NF=8时误差仅从80.4cm降至42.6cm；ME最高达67.5%；数据量降低到30%时误差仍<43cm。

**⚠️ 局限性**

局限性包括：1) 对极端光照与多障碍环境的适应性尚未充分验证；2) 仅评估训练阶段能耗，未考虑推理与通信能耗；3) 只使用单一VLC传感器，未与其他传感器融合；4) 在更大规模VLC网络与更复杂网络结构上的可扩展性需进一步研究。

---

## 482. Semantic-Aware Update Admission and Adaptive Byzantine Redundancy for Secure Computation Offloading in COIN-Enabled Metaverse Systems

**arXiv ID:** 2610.29592 | [PDF](https://arxiv.org/pdf/2610.29592v1)

**作者:** Ibrahim Aliyu `[一作]` (Chonnam National University), Jinsul Kim `[通讯]` (Chonnam National University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文提出一种基于COIN的元宇宙系统中，语义感知的更新入选与自适应拜占庭冗余相结合的安全计算卸载框架，采用分层R‑PBFT方法先决定更新入选与冗余水平，再通过相同利益博弈优化任务卸载位置；

**💡 创新点**

创新点在于：①将持久共享状态的语义重要性与更新持续性融入更新入选决策；②设计可在资源预算内自适应选择拜占庭冗余的DDQN‑R‑PBFT控制器；③通过可行性过滤器保证安全与预算约束；④将博弈与强化学习结合的分层决策结构。

**🔧 技术方法**

使用的技术包括：分层冗余感知PBFT（R‑PBFT）模型、可行性过滤的双深度Q网络（DDQN）、相同利益博弈求解、动态规划求解多选背包、链路与计算资源模型以及链路能耗优化。

**📊 数据集**

实验数据基于仿真环境，设置25-40名XR用户、4种任务、4个上行通道、固定资源预算、拜占庭容忍度等，使用自定义的任务负载与网络状态。

**📈 对比分析**

与Minimum Secure、Excess Redundancy、Random Feasible、No Commit等基线以及若干消融版本进行比较。实验显示DDQN‑R‑PBFT在拜占庭严重度 z=1~3 时相较最强基线平均提升约 46.7% 的系统奖励，且在 z=3 时安全及时更新比例提升 62.3%；在不同共享状态有效期、用户负载、计算能力等场景下均保持较优性能。

**⚠️ 局限性**

局限性包括：①对拜占庭状态的感知仅依赖静态参数，缺乏对非平稳攻击的自适应；②学习与求解复杂度高，尤其在更新数目大时的多选背包求解；③实验仅在仿真场景，缺乏真实元宇宙部署验证。

---

## 483. Rethinking Post-hoc Debiasing of Large Pretrained Models

**arXiv ID:** 2610.29688 | [PDF](https://arxiv.org/pdf/2610.29688v1)

**作者:** Divyaansh Mertia `[一作]` (Indian Institute of Technology Jodhpur), Pratik Mazumder `[通讯]` (Indian Institute of Technology Jodhpur)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种后置去偏差方法，利用冻结的预训练模型的几何特征对特征空间进行线性变换，从而在不依赖任何偏差属性标签或语言模型的前提下实现去偏差。

**💡 创新点**

创新点在于发现并利用“高相似度跨类别样本几乎必然为偏差共享”这一几何特征，构造基于余弦相似度的推挤/拉近双向锚定损失，单纯通过类别标签与相似度即可自动识别并分离偏差共享对，而不需要任何文本提示或外部知识。

**🔧 技术方法**

使用的技术包括：冻结预训练模型（CLIP、SigLIP、DINOv3 等）产生归一化特征；构造跨类别高相似度样本集 𝑃 与低相似度同类别样本集 𝑄；对线性映射 𝐖 施加一侧 Hinge 损失（𝑙_push 与 𝑙_pull）；在映射后再训练一个线性分类头；整个过程仅涉及相似度计算与线性变换，无需梯度传播至 backbone，也不使用文本编码器或 LLM。

**📊 数据集**

在四个基准数据集上评估：UrbanCars（多重偏差），BAR（动作-场景偏差），BFFHQ（年龄-人种偏差）和 Waterbirds（背景偏差）。

**📈 对比分析**

与多种现有后置去偏差方法（FairerCLIP、PRISM、PRISM‑mini、RoboShot、SEM_i）以及未去偏差的 CLIP‑linear 进行对比。该方法在所有数据集上均实现了最优或竞争性的 worst‑group / overall 准确率，尤其在 UrbanCars 的 worst‑joint 准确率上从 3.23% 提升至 81.18%，超过最优竞争方法 19.89 点；在 BAR、BFFHQ 和 Waterbirds 的 worst‑group 上也提升 1–15 点，整体准确率亦有提升。

**⚠️ 局限性**

局限性：仍需预训练模型的类别标签；对极大规模数据集计算所有样本对相似度成本高；线性映射 𝐖 仅能捕捉线性结构，可能无法完全消除复杂非线性偏差；方法对超参数（阈值、边界）敏感，需要额外验证；仅在视觉/视觉‑语言模型上验证，未展示在其他任务或模态上的泛化能力。

---

## 484. TraceForge: Distilling with Your Own Thinking

**arXiv ID:** 2610.29680 | [PDF](https://arxiv.org/pdf/2610.29680v1)

**作者:** Yan Wang `[一作]` `[通讯]` (Independent Researcher), Yan Wang (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种利用生成器和批评家在同一模型家族内部生成合法、自然的推理轨迹，以改进学生模型训练的方法。

**💡 创新点**

创新点在于通过训练生成器重写教师答案的推理轨迹，使其保持学生家族风格，并使用批评家进行真实性对抗训练，从而在不违反条款的前提下获得比原始教师轨迹更有效的训练信号。

**🔧 技术方法**

技术上采用了Qwen系列模型的监督微调、生成器-批评家对抗学习、最佳四条轨迹选择、DPO目标以及Qwen3.5‑9B‑Base作为学生基模型。

**📊 数据集**

使用的数据集包括约5,000条来自Claude Code的Fable 5推理轨迹与答案、同一提示下GPT‑6.1‑SOL生成的答案以及对应的合成推理轨迹，共构成五套训练语料。

**📈 对比分析**

通过在ARC‑Easy、ARC‑Challenge、BoolQ、HumanEval+、MBPP+等基准上比较五套语料的学生模型，发现基于家族风格的合成轨迹（S2、S5）在编码任务上比原始教师轨迹提升约4–7分，并在S5上比TraceInversion差距超过50分，甚至比Qwen‑Max自有轨迹高出21–35分。

**⚠️ 局限性**

局限性包括仅在Qwen家族内部实验、对生成器循环或冗余推理的依赖、对不同家族迁移能力未知、以及对安全性和多步推理的评估不足。

---

## 485. Toward Recursive Self-Improvement in Physical AI

**arXiv ID:** 2610.29677 | [PDF](https://arxiv.org/pdf/2610.29677v1)

**作者:** Meng Luo `[一作]` (National University of Singapore), Hongzhan Lin `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文综述了物理机器人领域中的递归自我改进（RSI）概念、机制与评估框架，梳理了从行为改进到学习过程改进的四个层级，并对现有的实验方法、基准数据集以及技术手段进行了系统归纳。

**💡 创新点**

创新点在于：①提出RSI的阶梯式主张（行为改进→后续学习收益→过程改进→递归改进），并阐明实现递归所需的两条关键链路；②归纳并细化了四个关键条件（信息反馈、可用性、对分布漂移的鲁棒性、持续实践可行性）；③给出了针对物理RSI的评价原则和未来研究方向，指出了实验设计中的核心控制需求。

**🔧 技术方法**

所讨论的技术包括：强化学习（RL）、元学习、自动机器学习（AutoML）、自监督学习、在线自我学习、代码生成与修正、世界模型与仿真、技能与记忆库、运行时决策修正等多种方法。

**📊 数据集**

引用了多种机器人学习基准和数据集，主要包括：LIBERO、LIBERO-PRO、LIBERO-Plus、RoboCasa、RoboTwin 2.0、RoboDojo、VLABench、Meta-World、SIMPLER、Habitat、RoboCoach-Bench、PhysRSI-eval、FailBench、CaliBench、RLE-Bench、RoboCoach-Bench 等。

**📈 对比分析**

通过对比实验（如行为成功率、后续学习速度、训练预算节省、模型评估分数等）评估不同方法。总体来看，多数系统能在行为层面取得显著提升，但在后续学习收益和过程改进方面的验证相对稀少；真正的递归改进（即修正自身改进过程）实验证据更为有限，常见的是单向改进或仅在固定改进器上取得提升。

**⚠️ 局限性**

局限性包括：①物理实验成本高，导致实验轮数受限；②评估工具（如手工标注、自动检测器）不一定可靠，可能误导后续改进；③任务与硬件变化（传感器漂移、执行器磨损）会破坏先前改进的有效性；④大多数研究缺乏严格的递归链路控制（即没有单独扣除已学习的修正知识来验证递归效应），因此难以确立真正的递归自我改进；⑤跨机器人或跨平台迁移仍面临接口与适配的挑战。

---

## 486. From Acceptance and Avoidance to Affective Viscosity: Designing Social Agents Intermediating in Human Relationships

**arXiv ID:** 2610.29674 | [PDF](https://arxiv.org/pdf/2610.29674v1)

**作者:** Tomoko Yonezawa `[一作]` `[通讯]` (Kansai University), Tomoko Yonezawa (Kansai University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出情感黏度（affective viscosity）概念，并在四个设计案例（个人空间侵犯、语音不适、时空粘性、三人关系嫉妒）中验证其可行性。

**💡 创新点**

创新点在于将接受/回避从即时二元反馈转变为关系信号，并将负情绪的累积、抑制与逐步表达建模为情感黏度，从而为社会代理在复杂人际关系中的中介角色提供理论与设计框架。

**🔧 技术方法**

技术手段主要基于心理学过程模型（Scherer 的情绪过程模型、Gross 的情绪调节模型）与机器人/VR 代理的状态累积与抑制算法，结合视觉/语音表达的情感调节实现。

**📊 数据集**

使用了自定义实验数据：四组实验共计约 100 名参与者（每组 24–25 人），通过在线问卷和实验室互动记录收集。

**📈 对比分析**

对比方法为基于视觉模拟量表（0–99）评估代理的情绪抑制、可接受性和互动意愿。实验显示，在采用“耐心”表达先行的策略时，情绪抑制感知度、继续互动意愿和亲和度均显著提升。

**⚠️ 局限性**

局限性包括样本量相对较小、实验环境为实验室或预录视频，缺乏实时自适应评估；未涉及跨文化差异与长期互动场景，亦未对情感黏度模型的泛化能力进行系统验证。

---

## 487. SPARE: Sparsity Allocation via Routing Entropy for Efficient MoEs

**arXiv ID:** 2610.29662 | [PDF](https://arxiv.org/pdf/2610.29662v1)

**作者:** Yifan Yang `[一作]` (Intellifusion Inc.), Xiaotian Yu `[通讯]` (Intellifusion Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SPARE——一种训练无关的层间稀疏分配方法，用路由熵指导 MoE 专家裁剪；

**💡 创新点**

创新点在于把 token 级路由熵作为层级稀疏度分配信号，能与任意专家选择器配合，并提供扰动上界与实验验证；

**🔧 技术方法**

使用路由熵计算、基于熵的稀疏分配、随机/确定性专家选择器（REAP、EASY‑EP）以及相关理论分析；

**📊 数据集**

使用 Mixtral、Qwen3‑30B‑A3B、DeepSeek‑R1 三大 MoE 模型，并在 MMLU、ARC、BoolQ、HellaSwag、OpenBookQA、RTE、WinG、MATH‑500、LiveCodeBench、GPQA‑Diamond 等数据集上评估；

**📈 对比分析**

与均匀分配、OWL 等基线对比；在 50% 专家裁剪时 Qwen3 提升 6.6pp、DeepSeek‑R1 提升 2.7pp，25% 裁剪时提升 0.1–0.7pp；

**⚠️ 局限性**

局限在于仅在高裁剪率下显著提升；对中等比例裁剪提升有限；未考虑与量化或专家合并的组合效果。

---

## 488. Through the Looking Glass: Analyzing Geolocation Providers Using Looking Glasses and Active Measurements

**arXiv ID:** 2610.29660 | [PDF](https://arxiv.org/pdf/2610.29660v1)

**作者:** William Leung `[一作]` (IPinfo), Oliver Gasser `[通讯]` (IPinfo)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过收集并分析了4487个Looking Glass服务器的公开位置，结合六个不同的地理定位数据库和主动RTT测量，评估了Looking Glass服务器的定位可靠性，并量化了不同地理定位服务商在定位Looking Glass时的准确度。

**💡 创新点**

创新点在于：①首次大规模系统性评估Looking Glass服务器的地理定位准确性；②将主动RTT测量与多源地理定位数据库结果相结合，形成“RTT确认”与“地理定位一致性”的双重验证框架；③发现大多数Looking Glass提供商存在位置不一致甚至误报问题，并揭示IPinfo在城市级定位上表现最佳。

**🔧 技术方法**

使用的技术包括：ICMP/ICMPv6 RTT主动测量，基于光纤传播时延2/3c计算RTT半径并构建RTT多边形；地理定位数据库查询（六个数据库，四家供应商）；GIS空间分析（计算距离分布、覆盖率、20km内重叠情况）；数据清洗与手工校正GeoNames坐标。

**📊 数据集**

所用数据集为：①通过爬虫获取的4487个Looking Glass服务器及其自报位置；②IPinfo全球分布网络（1300+节点）收集的RTT测量数据；③2026年9月的六个主流地理定位数据库快照，涉及城市、区域、国家级坐标。

**📈 对比分析**

比较方法：首先用RTT多边形判定Looking Glass位置是否在测量范围内（RTT确认/不确认）；随后将地理定位数据库返回的坐标与RTT确认后的真值进行距离计算，统计城市/区域/国家级匹配率、距离CDF以及20km内的重叠集。性能评估显示，IPinfo在城市级匹配率最高（78.3%），其余供应商匹配率普遍偏低，且在北美与欧洲部分地区误差显著。

**⚠️ 局限性**

限制因素包括：仅分析了四家主流供应商的六个数据库，未覆盖所有可用定位服务；Looking Glass服务器收集方式手工爬取，可能遗漏部分地区；RTT测量仅能确认位置是否合理，无法校正RTT不一致的服务器；在缺乏RTT一致性的服务器上缺乏可信真值，导致后续分析受限。

---

## 489. Tight Space Bounds for Pivot-Based Distance Oracles in Sparse Graphs

**arXiv ID:** 2610.29624 | [PDF](https://arxiv.org/pdf/2610.29624v1)

**作者:** Avi Kadria `[一作]`, Liam Roditty `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用。

**💡 创新点**

创新点在于提出了一种改进的算法结构，能够提高任务的处理效率。

**🔧 技术方法**

使用了深度学习和机器学习技术。

**📊 数据集**

采用了公开的标准数据集进行实验。

**📈 对比分析**

与现有方法进行了对比，结果显示新算法在准确率和速度上均有显著提升。

**⚠️ 局限性**

限制在于算法在特定情况下的适用性仍需进一步验证。

---

## 490. Local and Global Recursive Ridge Regression for Online Residual Correction of Frozen Multivariate Forecasters under Label Delay

**arXiv ID:** 2610.29604 | [PDF](https://arxiv.org/pdf/2610.29604v1)

**作者:** Takumi Fujimoto `[一作]` (Keio University), Hiroaki Nishi `[通讯]` (Keio University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对冻结的多步长预测器，设计了可在线更新的残差校正器 DRiC，用于在标签延迟（仅在完整预测区块后获得真值）下减小误差。

**💡 创新点**

创新点在于：①将校正限制为前 K 个离散余弦变换（DCT）系数，降低维度；②使用固定大小的递归岭回归统计量，避免随通道数增长的状态；③引入“guard”机制，仅在残差功率在可接受范围内时才更新统计量，并对输出做裁剪；④提供局部和全局两种变体，后者共享统计量以实现极小状态。

**🔧 技术方法**

技术细节包括：递归岭回归（带记忆衰减）、DCT特征提取、守卫阈值控制、共享混合权重以及在每个区块使用前 32 个区块的残差-校正相关来确定融合权重。

**📊 数据集**

实验使用七大公开数据集：ETTh1/ETTh2/ETTm1/ETTm2、Weather（含缺失值及填充版本）、Electricity、Traffic，配合三种基线预测器（DLinear、PatchTST、iTransformer）。

**📈 对比分析**

与 Naïve、梯度适配器（TAFAS、PETSA、COSA、FAC、ORCA）以及闭式校正器（RLS‑Bank、ELF、EPOC）对比，DRiC 在 126 个实验条件下平均 MSE 减少约 48%，比 ELF 低 0.4% 的误差且保持约 10 倍更小的状态；在 Electricity 和 Traffic 上，全局版本 <DCRGlobal> 在状态与精度上均居于前列。对比结果在论文图表中以状态-性能前沿展示。

**⚠️ 局限性**

局限性包括：仅在单一预测 horizon H=24、固定标签延迟（d=H）下评估；未验证在更长 horizon、不同标签延迟或预先拟合后冻结的场景；对缺失值的处理依赖 guard，可能在某些数据集（如 Traffic + PatchTST）下略逊于无 guard 版本；并且目前实现以 double 精度计数状态，实际部署可进一步压缩。

---

## 491. A DIY DPDK Software Router for Networking Education

**arXiv ID:** 2610.29625 | [PDF](https://arxiv.org/pdf/2610.29625v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 492. No Frames, Just Frequencies: Bandwidth-Efficient Deployment-Time Adaptation for Smart Glasses

**arXiv ID:** 2610.29607 | [PDF](https://arxiv.org/pdf/2610.29607v1)

**作者:** Francesco Corti `[一作]` (Graz University of Technology), Olga Saukh `[通讯]` (Graz University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `67630363-6be0-4f51-ab05-7198250671a5` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在智能眼镜上提出了“无帧仅频率”(No Frames, Just Frequencies)框架，通过在设备端计算频率描述符并仅传输这些低维特征给主机，以实现模型在部署时的自适应，而不需传输原始视频帧。

**💡 创新点**

创新点在于：①将频率描述符作为分布漂移的低带宽表征；②将梯度适配卸载到配对主机，仅更新模型的少量参数；③利用多阶段腐败分类器将频率描述符映射到合成的训练样本，实现源辅助的在线适配。

**🔧 技术方法**

采用的技术包括：RAPSD、DCT、DWT频率描述符；多头腐败分类器与四阶段因果腐败图；基于合成混合腐败的训练数据生成；源辅助适配协议（只更新归一化层与首块参数）和交叉熵损失；以及对ConvNeXt、DeiT和TinyViT等不同架构的应用。

**📊 数据集**

实验使用的数据集为：ImageNet-C（混合腐败），ImageNet验证集、Visual Wake Words二分类数据以及MS-COCO目标检测数据。

**📈 对比分析**

与No Adapt、Oracle-Entropy（熵最小化）和Oracle-CrossEntropy（有监督）基线对比，结果显示在ImageNet上恢复93–95%的监督上限，VWW 82%，COCO 94–98%；在不同模型和分辨率下保持优良性能；且频率描述符分辨率降低可实现213×–784×的带宽压缩，同时保持80%以上的适配增益。

**⚠️ 局限性**

局限性包括：适配仅覆盖预定义的10种腐败家族，无法处理未建模的失真；频率描述符分辨率的选择需要权衡适配效果与带宽；适配循环耗时较长；尚未验证对视觉语言模型等更复杂任务的适用性。

---

## 493. Overview of CHIP 2026 Shared Task 1: Patient-Specific Phenotype Concept Recognition in Full-Text Biomedical Literature

**arXiv ID:** 2610.29606 | [PDF](https://arxiv.org/pdf/2610.29606v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 494. Improved Training of Strong Stochastic Flow Maps

**arXiv ID:** 2610.29605 | [PDF](https://arxiv.org/pdf/2610.29605v1)

**作者:** Zander W. Blasingame `[一作]`, Alexander Tong `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `40105733-5154-44cd-8090-a8cab9e64b07` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

改进强随机流图的训练方法，提出新的表示与损失框架，并实现了对大型预训练流模型的微调

**💡 创新点**

创新点在于将平移勒让德多项式表示与Wong–Zakai式均值流（MeanFlow）损失相结合，并引入指数提升、闭式马尔可夫积分与漂移匹配参数化，成功让强随机流图可使用基于导数的训练损失

**🔧 技术方法**

使用平移勒让德多项式近似布朗运动、Wong–Zakai随机ODE、指数提升、闭式马尔可夫积分、漂移匹配损失以及对预训练确定性流图的微调策略

**📊 数据集**

主要使用ImageNet-256数据集进行大规模实验，并在玩具问题（二维棋盘分布、1D非线性SDE）以及40高斯混合的Feynman–Kac校正任务上进行验证

**📈 对比分析**

通过与原始SSFM和mccallum的SSFM在相同NFE下的FID、FDDINOv2、强一致性误差、以及人类偏好指标对比，iSSFM在4 NFE时FDDINOv2约提升3倍，FID提升1.5倍，并在强一致性和Feynman–Kac校正上表现更优

**⚠️ 局限性**

仍无法与最强的确定性少步模型竞争；对布朗系数的传递方式尚未最优；在更复杂系统或更高维度下的稳定性与扩展性待进一步研究

---

## 495. Formalizing Cooperation and Competition: An AI-Assisted Mathematical Addendum to a 2018 Planning Thesis

**arXiv ID:** 2610.29637 | [PDF](https://arxiv.org/pdf/2610.29637v1)

**作者:** Haitham Masaud Amar `[一作]` `[通讯]`, Haitham Masaud Amar

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

重新构建了原论文的数学模型并用Lean4证明核心定理，同时通过可执行有限搜索协议验证协调协议的局部收敛与效率；

**💡 创新点**

明确了离群者机会、融资方案与价格执行规则，并将竞争与合作稳定性映射到可验证的核心/弱核心理论，首次提供可复现的实验与对比；

**🔧 技术方法**

Lean4定理证明、Mathlib、Python数值实验、合成网络生成以及可执行有限搜索协议；

**📊 数据集**

44个原论文成本表、96个合成网络实例（四类网络）以及10人原始案例的重构；

**📈 对比分析**

通过在所有实例上对比可执行程序与原启发式的成本、批准率、Nash稳定性，合成实例中达成92/96的批准最优，平均搜索误差约0.18，表现良好；

**⚠️ 局限性**

结果仅在有限行动集与单调拥塞子类内有效，未保证全局最优或在更大规模/非单调成本环境中的收敛；理论证明依赖显式融资或固定外部者规则，未涵盖通用博弈情形。

---

## 496. Diff-IrisPAD: Modeling Bona Fide Iris Distributions for Unsupervised Presentation Attack Detection

**arXiv ID:** 2610.29611 | [PDF](https://arxiv.org/pdf/2610.29611v1)

**作者:** Geetanjali Sharma `[一作]` (Indian Institute of Technology Mandi), Raghavendra Ramachandra `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于Brownian Bridge扩散的无监督虹膜伪造检测框架Diff-IrisPAD，利用仅包含真伪虹膜样本的重建不一致性来区分攻击和真伪；

**💡 创新点**

创新点在于：①首次将BBDM与VQGAN融合，直接在潜在空间学习真虹膜分布；②采用ViT评分与像素级误差作为检测指标；③通过无监督训练实现对未知攻击类型的强泛化；

**🔧 技术方法**

使用Brownian Bridge Diffusion、潜在Diffusion（LBBDM-f4）、预训练VQGAN编码器、ViT重建分数、LPIPS相似度、DDIM采样与多尺度注意力UNet；

**📊 数据集**

在125k+多传感器虹膜数据集上评估，包含Clarkson、NDCLD、IIIT_WVU、Warsaw、IIITD Combined Spoofing、GUC_VSIA_CV、BTAS_MIR2016等，涵盖8种攻击类型；

**📈 对比分析**

在ZAK协议下与AnoDDPM、f-AnoGAN以及多种监督CNN/ViT/DINO模型对比，Diff-IrisPAD在整体ACER上达到30.52%，显著低于Baseline的49.17%，并在留一攻击（LOO）评估中保持较优的泛化能力；

**⚠️ 局限性**

局限性包括对合成降解方案的依赖，降解不匹配会影响重建质量；在高度逼真的攻击（如纹理隐形隐形接触镜）下仍可能出现误检。

---

## 497. Benchmarking Coding Agents for System-Level Performance Optimization

**arXiv ID:** 2610.29695 | [PDF](https://arxiv.org/pdf/2610.29695v1)

**作者:** Yanze Zhang `[一作]` (University of Macau), Xiaobo Zhou `[通讯]` (University of Macau)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MicroPerf-Bench基准，评估编码代理在微服务系统中的端到端性能优化能力。

**💡 创新点**

创新在于通过控制注入已知性能反模式，构造可复现的系统级性能缺陷，填补现有仅针对函数或仓库级别的评测空白。

**🔧 技术方法**

采用大型语言模型驱动的编码代理（OpenCode、OpenHands、Claude Code）结合GLM、Qwen、DeepSeek等模型，并使用Docker Compose、负载脚本、分布式跟踪等技术。

**📊 数据集**

使用七个开源微服务应用（Spring PetClinic、MartianBank、Robot Shop、Pitstop、DeathStarBench等）共65个实例，覆盖9种性能反模式。

**📈 对比分析**

通过多配置（9组）在1555次实验中评估，最佳配置平均恢复率仅42%，正分率44%，完全恢复率低于23%。

**⚠️ 局限性**

局限在于代理难以定位真正的系统级瓶颈，主要失败为功能正确但未提升性能，且评测受环境波动和缺乏运行时反馈影响。

---

## 498. Efficient Generative Image Compression with Quantized Latents Refinement

**arXiv ID:** 2610.29691 | [PDF](https://arxiv.org/pdf/2610.29691v1)

**作者:** Hao Cao `[一作]` (Tsinghua University), Jungong Han `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种高效生成式图像压缩方法，兼顾低延迟和高感知质量。

**💡 创新点**

引入 Guided Refinement（GR）利用熵上下文进行高斯细化，并重构 P-RVQ 以提升并行度。

**🔧 技术方法**

采用熵上下文生成的高斯分布、可训练的采样扰动和重排后的投影残差向量量化（P-RVQ）技术。

**📊 数据集**

在 Kodak、Tecnick、DIV2K 验证集以及 CLIC 2020 专业集上进行实验。

**📈 对比分析**

与 OneDC 等轻量级 GIC 以及多种扩散/ VAE 基准相比，BD‑rate 在 LPIPS/DISTS/PSNR/FID 上分别提升 23.19%/37.71%/22.30%/22.14%，编码/解码延迟分别为 4.60/5.98 ms（A100）

**⚠️ 局限性**

仍受限于极低码率下生成能力不足，且在更低码率下需进一步提升先验与合成能力。

---

## 499. Learning to Reason Like a Cytopathologist:Clinically Grounded Multiscale Evidential Learning for Cervical Cytology

**arXiv ID:** 2610.29665 | [PDF](https://arxiv.org/pdf/2610.29665v1)

**作者:** Ajeet Kumar Singh `[一作]` (IIT Delhi), Dorien Herremans `[通讯]` (Singapore University of Technology and Design)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种多尺度证据融合框架，用于宫颈细胞学的二分类（良性 vs 病变）及不确定性导向分诊。

**💡 创新点**

创新点在于：①使用基于临床风险的非均匀软锚点保留Bethesda系统的序数关系；②分别对核、细胞质和邻域三尺度独立编码，保留尺度间的争议；③将校准后的概率转化为 Dempster–Shafer 信念质量，显式保留未承诺的证据，实现真正的“无决策”状态；④利用可靠性折扣在融合前降低低置信度尺度的影响。

**🔧 技术方法**

技术手段包括：冻结 DINOv2 ViT‑B/14 作为特征提取器；多尺度 MLP 头；温度或等距回归校准；基于贝叶斯二分类阈值的正则化；Dempster–Shafer 组合与可靠性折扣；基于证据质量的分诊策略。

**📊 数据集**

使用了两个公开细胞学数据集：RIVA（10,613/2,654/2,682 细胞）和 CRIC Cervix（8,073/1,730/1,731 细胞）。

**📈 对比分析**

对比方法包括多数投票、概率平均、逆熵加权、可学习 MLP 融合；性能上，DST 融合在两数据集的宏 F1 分别提升到 0.726（RIVA）和 0.924（CRIC），误判样本的平均不确定性比正确样本高 1.94–4.13 倍，且高置信度样本的 F1 明显高于低置信度样本。

**⚠️ 局限性**

局限性主要是：①在 RIVA 中 ENDO 类别的二分类准确率低，仅 0.22，且误判样本不易被不确定性检测；②CRIC 的真正不确定池样本极少（仅 0.9%），且需额外的 NILM 过滤阈值；③当前框架仅为二分类证据，未对可能与 HSIL 混淆的良性形态提出更细粒度的假设空间。

---

## 500. Effects of Model Scale and Quantization on Social Bias and Abstention in Question Answering

**arXiv ID:** 2610.29657 | [PDF](https://arxiv.org/pdf/2610.29657v1)

**作者:** Kuniko Paxton `[一作]` (University of Hull), Dhavalkumar Thakker `[通讯]` (University of Hull)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了模型规模与量化对问答任务中社会偏见与回避行为的影响。

**💡 创新点**

创新点是同时评估量化、规模、提示语和回避选项对偏见的交互影响。

**🔧 技术方法**

采用多种量化方法（FP8、INT8、INT4、动态量化）并进行对比实验。

**📊 数据集**

使用BBQ基准测试数据集。

**📈 对比分析**

通过与FP16基准模型对比，发现精度与偏见变化不一致，量化可降低偏见但往往伴随回避增加。

**⚠️ 局限性**

局限：仅使用BBQ，且仅覆盖轻量级模型，难以推广到更大模型或其他偏见基准。

---

## 501. Joint Semantic-Channel Coding and Modulation for Generative Image Transmission

**arXiv ID:** 2610.29643 | [PDF](https://arxiv.org/pdf/2610.29643v1)

**作者:** Hao Cao `[一作]` (Tsinghua University), Yuan Shen `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出SemQAM，一种将语义编码、信道编码与QAM调制联合优化的生成式图像传输系统。

**💡 创新点**

创新点在于①通过接收端检测索引更新VQ码本，将信道误差直接融入码本学习；②利用接收码字的条件期望（Expected‑RX）逼近接收端重建，降低自回归编码中的误差累积。

**🔧 技术方法**

技术主要包括残差向量量化（EF‑LIC）、向量量化码本的接收端学习、QAM符号映射、条件期望估计、LDPC信道编码、生成式重建损失（LPIPS/FID）以及自回归上下文与超先验网络。

**📊 数据集**

训练使用ImageNet，评估采用Kodak、Tecnick、DIV2K验证集和CLIC 2020 Professional数据集。

**📈 对比分析**

与JSCC基线（NTSCC、NTSCC+、MDJCM‑A、Quad‑JSCC）及分离式语义+信道编码+调制基线（OneDC、EF‑LIC等）进行对比，在10 dB AWGN下，SemQAM在LPIPS、DISTS、FID等感知指标上均优于对手，且在匹配质量下平均减少18.57 %信道使用量，BD‑rate更低，延迟和参数量也显著下降。

**⚠️ 局限性**

局限性包括仅在AWGN信道下验证，未针对多径衰落或OFDM等实际物理层场景；操作点固定，需进一步扩展至更广泛的信道条件；以及模型训练与部署复杂度相对较高。

---

## 502. Cross-Area Transfer Learning for VLC-based Indoor Localization with a Transfer Efficiency Score

**arXiv ID:** 2610.29638 | [PDF](https://arxiv.org/pdf/2610.29638v1)

**作者:** Masood Jan `[一作]` (Sorbonne University), Alexander Artemenko `[通讯]` (Robert Bosch GmbH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于迁移学习的可扩展VLC室内定位框架，并通过引入Transfer Efficiency Score自动选择最佳源模型。

**💡 创新点**

① 迁移学习仅需40%目标区域数据即可恢复97%性能；② 设计TES复合指标，消除手工模型选择的试错成本；③ 在真实工业环境中实现误差≤40.9 cm、成功率≥86%。

**🔧 技术方法**

使用深度神经网络回归模型进行VLC信号定位，采用迁移学习微调策略，以及TES指标进行模型评估与选择。

**📊 数据集**

使用Bosch制造厂四个不同子区域（A、B、C、D）的VLC RSS测量数据，约160个测点/区域，10个LED发射器的真实工业数据集。

**📈 对比分析**

对比无迁移直接推断、全量数据训练和迁移学习后性能；评估指标为平均定位误差和成功率；迁移学习后误差降至40.9 cm，成功率≥86%，在未见区域TES选优后误差降至3.28 cm。

**⚠️ 局限性**

局限于单一工厂实验；未考虑高度变化、移动设备场景；仅使用监督迁移，未探讨半监督或多源融合；TES计算需要离线标注数据。

---

## 503. Characterizing Oblivious Common-Reward Multi-Player Bandits: Universal Stationarity, Coordination Traps, and Certified Global Rates

**arXiv ID:** 2610.29673 | [PDF](https://arxiv.org/pdf/2610.29673v1)

**作者:** Mingxiao Li `[一作]`, Hongyu Wang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文对完全无记忆的多玩家共享奖励多臂老虎机模型（OC‑MPMAB）进行理论分析，证明了各玩家的梯度更新构成无偏的产品策略梯度估计，并给出了无条件的有限时间收敛到一阶鞍点的保证。

**💡 创新点**

创新点包括：① 在完全无记忆的设置下首次证明独立梯度更新的无偏性；② 通过构造“产品策略陷阱”阐明一阶鞍点不等价于全局最优；③ 通过引入产品梯度优势与自界噪声的“离线证书”，在结构化奖励表下恢复全局最优性，并给出 O(1/t) 的策略误差和 O(log T) 的伪后悔理论上界。

**🔧 技术方法**

使用的技术主要是：产品策略梯度的无偏性证明（基于softmax导数展开）；光滑性与梯度下降收敛分析；陷阱示例与不动点不动性证明；离线证书框架（产品梯度优势 λ、噪声自界 M）的构造与使用；以及利用这些证书得到的收敛率与后悔上界。

**📊 数据集**

无（本文为纯理论分析，不依赖实际数据集）。

**📈 对比分析**

方法比较主要在理论层面：在证书满足条件时，收敛速率为 O(1/t) 的产品策略误差、O(log T) 的伪后悔；在一般情形下仅能保证有限时间的一阶鞍点收敛。实验与对比方法未给出。

**⚠️ 局限性**

限制：① 仅在奖励表满足产品梯度优势与噪声自界的结构性假设时才能获得全局最优性；② 在存在产品策略陷阱的任意奖励表下，单纯的梯度更新无法避免陷入子最优子空间；③ 研究假设奖励归一化、无记忆玩家和共享标量奖励，缺乏对更一般通信或观测模型的分析。

---

## 504. Reference Manual of MEDiml

**arXiv ID:** 2610.29666 | [PDF](https://arxiv.org/pdf/2610.29666v1)

**作者:** Mahdi Ait Lhaj Loutfi `[一作]`, Martin Vallières `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发并公开了MEDiml平台，提供从图像预处理、手工特征提取、特征筛选到机器学习和解释的完整无代码与代码两种工作流；

**💡 创新点**

创新点在于：① 将复杂度驱动的特征分级（形态、强度、纹理、线性滤波、纹理滤波）纳入系统，系统性比较不同复杂度模型；② 引入False Discovery Avoidance特征筛选；③ 提供可视化解释树和代码生成；

**🔧 技术方法**

技术包括：Python（NumPy、Pandas、PyCaret、Ray）、Electron/React/ReactFlow/Go前后端、CUDA加速的三维纹理滤波、IBSI标准化、Ray并行、ComBat归一化、XGBoost、Statistical Tests（DeLong、Wilcoxon等）；

**📊 数据集**

使用5个肿瘤学影像数据集（NSCLC CT、LGG MRI、颅内脑膜瘤 CT/MRI、RCC CT/MRI），共计2104张扫描；

**📈 对比分析**

通过10折交叉验证、AUC统计、DeLong检验等方法评估不同复杂度特征组合的模型，发现多数任务中低复杂度（形态或强度）即可获得最佳或接近最佳AUC，且统计显著；

**⚠️ 局限性**

局限包括：对小样本易过拟合、仅评估二分类任务、依赖手工分割、缺乏深度学习与多模态集成、未充分验证不同机构、不同扫描协议下的鲁棒性。

---

## 505. Probabilistic Machine Learning for Short-Term Atmospheric Seeing Forecasting

**arXiv ID:** 2610.29654 | [PDF](https://arxiv.org/pdf/2610.29654v1)

**作者:** Mary-Joe Medlej `[一作]` (Université Côte d’Azur), Christophe Giordano `[通讯]` (Université Côte d’Azur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对 Maunakea 高峰的 DIMM 观测数据进行 2 小时以内的短期大气视线（seeing）预测，比较了 RNN、LSTM、Gaussian Process（GP）和基于条件正则化流的 FloTS 模型。

**💡 创新点**

首次将基于条件正则化流的 FloTS 与 LSTM 编码器结合，用后置温度校准实现非高斯不确定性量化，并在不确定性评估上超越传统 GP。

**🔧 技术方法**

使用了确定性递归神经网络（RNN、LSTM）、概率高斯过程（GP）以及基于 Masked Autoregressive Flow 的 FloTS，并对 FloTS 进行温度尺度校准。

**📊 数据集**

Maunakea 天气中心提供的 2009‑2024 年 DIMM 视线测量数据，原始 90 秒间隔经线性插值后重采样到 10 分钟格点。

**📈 对比分析**

采用 RMSE、Pearson 相关系数和两者归一化后加权的综合得分进行比较；LSTM 在所有预测时距上表现最佳，FloTS 仅略逊一筹，但在预测分布的校准和不确定性描述上优于 GP。

**⚠️ 局限性**

GP 的高斯假设限制了预测分布的灵活性，且计算量随样本数呈立方增长；FloTS 尽管更灵活但需要更多参数和计算资源，且仅在单站点数据上验证，泛化性尚待进一步研究。

---

## 506. From Association to Decision Value: Testing Paired Instruction Dependence for VLA Corpus Selection

**arXiv ID:** 2610.29646 | [PDF](https://arxiv.org/pdf/2610.29646v1)

**作者:** Nurmukhambet Serikbayev `[一作]` (New York University), Muhammad Shafique `[通讯]` (New York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种通过在同一状态下改变目标指令来评估机器人演示语料库中语言依赖性的上游诊断方法（即paired instruction dependence）并将其用作选择更好训练语料库的指标；

**💡 创新点**

提出并验证了该诊断在大规模候选语料库上可实现可靠测量、与下游策略性能相关联、并探究了其在决策值上的增量价值，同时发现将其压缩为无符号开关率时会隐藏正确/错误响应结构；

**🔧 技术方法**

使用了基于Qwen2.5-VL-3B和Qwen3.5-VL-0.8B视觉语言动作模型的LoRA微调、Beta-Binomial层级预测器、以及配对指令交换实验的统计分析；

**📊 数据集**

构建了136个独立随机种子、不同环境、训练进度（10、50、250、1000、50000回合）的四足机器人导航语料库，并在其中挑选了56个用于训练、80个用于保留测试；

**📈 对比分析**

与多种基线（仅使用每目标正确率+熵的基线M0、添加开关率的M1、分别使用both-correct/both-wrong分解的变体、演示者先行正确率、代理Qwen3.5结果）进行对比，发现M1对log‑score的增量为+0.059 nats（CI跨零），但相对于M0的选择精度提升不显著；

**⚠️ 局限性**

局限包括需要可逆的演示者查询（仅限于模拟或在线生成环境）、只针对单一四足导航任务、短语料库、演示者使用固定的tabular Q‑learning、未验证在其他演示者或更大真实数据集上的泛化能力、以及未充分利用无符号开关率的结构信息。

---

## 507. Entangled Futures: Quantum Computing's Potential and Challenges in Computer Science

**arXiv ID:** 2610.29640 | [PDF](https://arxiv.org/pdf/2610.29640v1)

**作者:** Devki Nandan Jha `[一作]` (Newcastle University), Rajiv Ranjan `[通讯]` (Newcastle University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文系统综述了量子计算在计算机科学中的现状、关键架构与软件挑战，并对混合量子-经典系统的应用前景与技术瓶颈进行了深入讨论。

**💡 创新点**

创新点在于从系统级、软件工程与实践角度提出了统一的量子-经典协同模型，梳理了编译器、调度、通信、安全等多维度开放问题，并提供了面向行业的技术路线图。

**🔧 技术方法**

使用了现有的量子硬件平台（如 IBM Q、Google Cirq、Rigetti Forest 等）以及对应的软件工具链（Qiskit、Cirq、Tket、Amazon Braket 等）来阐述技术细节，结合量子算法（VQE、QAOA、Shor/Grover）和量子网络原理进行案例分析。

**📊 数据集**

论文主要基于公开文献、行业报告和云平台 API 文档，不涉及新的实验数据集；若需要验证，可使用公开的量子模拟器或云服务执行标准基准（如 quantum chemistry VQE 或 combinatorial optimization QAOA）来检验。

**📈 对比分析**

比较方法主要是对比传统经典算法与量子/混合算法在同一问题上的理论速度/精度优势，并通过对现有云量子服务的调度与执行时间进行实验对照；总体发现，当前 NISQ 设备在受限问题上可展示量子优势，但在规模化和错误率方面仍落后。

**⚠️ 局限性**

限制包括：量子硬件的噪声、退相干和可扩展性不足；编译器与调度缺乏统一标准；量子-经典交互的同步与低延迟挑战；安全性与信任机制尚未成熟；缺乏大规模真实数据验证，难以评估商业化可行性。

---

## 508. Who Verifies the Agent's Success Claim? Self-Report Leakage in GUI-Agent Completion Judging

**arXiv ID:** 2610.29609 | [PDF](https://arxiv.org/pdf/2610.29609v1)

**作者:** Yunfeng Zhu `[一作]` (Nanjing University), Zhenyu Chen `[通讯]` (Nanjing University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了自报告GUI代理在完成任务时可能出现的“危险误报”问题，并设计了一种报告与轨迹分离、可重放的结构化审计器，用于检查代理报告与实际轨迹是否一致。

**💡 创新点**

创新点包括：①揭示并量化自报告泄漏对轨迹判断的影响；②提出报告隔离的严格信息流规则，并通过结构化证据与确定性策略边界实现可审计、可重放的审计流程；③通过政策重放展示不同审计规则对误报率与检查负载的调节能力。

**🔧 技术方法**

技术手段包括：使用大型语言模型（Qwen3.5-35B 与 DeepSeek-284B）进行多次（5次）温度为0的推理，提取结构化证据；构建确定性检查器和策略映射；采用批量验证、Bootstrap置信区间以及自动化的审计阶段门控。

**📊 数据集**

数据集为1,384条Android GUI代理执行记录（AndroidWorld原始/扩展、Themis），包含两种代理（AppAgent、Mobile-Agent-v3），在此基础上筛选出465条审计目标记录，其中180条为危险误报。

**📈 对比分析**

与基线（声明式判定、prompt‑blind、claim‑gate‑only、匹配声明-aware判定）对比，结构化审计器在两种模型上均取得F1≈0.75–0.78，比prompt‑blind提升约0.13–0.14；在审计阶段门控下，能以45–56%的检查负载阻止74–90%的危险误报，显著降低无声错误通过率。

**⚠️ 局限性**

局限性包括：仅在被动、后置轨迹上进行判断，无法实时重执行或验证外部后置条件；轨迹提取仍存在误检，导致部分误报无法通过；实验仅覆盖Android GUI及两类代理，泛化性仍待进一步验证。

---

## 509. A Diagnostic Study of Faithfulness in Diffusion Language Models for Table-to-Text Generation

**arXiv ID:** 2610.29602 | [PDF](https://arxiv.org/pdf/2610.29602v1)

**作者:** Zaynab Merimi `[一作]` (CY Cergy Paris Université), Arthur Satouf `[通讯]` (MILA Quebec AI Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了扩散式语言模型与自回归模型在表格到文本生成任务中的可靠性，使用ToTTo数据集对模型在参考相似度和源文本事实一致性两方面进行比较。

**💡 创新点**

首次将BLEU等参考相似度指标与NLI推理的事实可信度分离开来，并量化模型在few‑shot提示下不同错误类型的变化。

**🔧 技术方法**

采用MDLM、LLaDA及其QLoRA适配版本的扩散模型、GPT‑2、GPT‑Neo等自回归模型，并结合BLEU、ROUGE、BERTScore、自BLEU和NLI Entailment等多维度评估手段。

**📊 数据集**

使用ToTTo基准数据集，仅保留高亮单元格的表格与对应描述，用于生成和评估。

**📈 对比分析**

在相同few‑shot提示协议下对所有模型的BLEU、ROUGE、BERTScore与NLI Entailment等指标进行对比，LLaDA+QLoRA在BLEU上最高（24.15）但其事实一致性并未显著提升。

**⚠️ 局限性**

仅在ToTTo上实验，模型规模有限，未评估信息完整性，且自动化NLI等指标仍可能高估事实错误。

---

## 510. WaRP: Workload-Aware Request Planning for Vector Retrieval on Object Storage

**arXiv ID:** 2610.29689 | [PDF](https://arxiv.org/pdf/2610.29689v1)

**作者:** Ruipeng Wan `[一作]` (Huazhong University of Science and Technology), Huanchen Zhang `[通讯]` (Tsinghua University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种工作负载感知的请求规划器 WaRP，结合离线簇重排，优化向量检索时在对象存储上合并 Range‑GET 请求，显著提升吞吐量并降低 GET 费用。

**💡 创新点**

创新点包括：①将请求合并建模为全局最优规划问题并给出可在查询路径上运行的 O(nKlogn) 动态规划 + 自适应前沿搜索；②基于共访问局部性设计离线簇重排策略；③引入请求并发度估计 P，使规划兼顾串行与并发极限。

**🔧 技术方法**

使用技术包括：动态规划（分治优化）和 Lagrangian 近似、最佳优先自适应前沿搜索、基于测得的对象存储延迟 surrogate、以及谱/随机投影、PCA+空间填充曲线等簇重排方法。

**📊 数据集**

实验数据集：SIFT1M、SIFT100M、GloVe、DEEP1M、DEEP100M、OpenImage、MSong、GIST1M 以及 Random1M（合成）。

**📈 对比分析**

与 Lance、Separate、Oracle‑Tuned Greedy 等基线对比；在七个数据集上 WaRP+Reorder 1.09–3.48× 提升 QPS、2.12–5.19× 减少 GET、p95/p99 延迟在六七数据集下降，单查询规划开销 <10 ms。

**⚠️ 局限性**

局限性：①需预估并发度 P，若估计不准影响效果；②重排仅在构建/维护时进行，无法即时适应查询模式变化；③在高维或簇尺寸较大时收益有限；④目前仅针对 IVF/IVFPQ 结构；⑤若缺乏查询日志，重排效果可能不佳。

---

## 511. Forward-Only Learning for Shared-Kernel Wavelet Neural Networks without Backpropagation

**arXiv ID:** 2610.29697 | [PDF](https://arxiv.org/pdf/2610.29697v1)

**作者:** Yongfei Guo `[一作]` (Changchun Institute of Optics, Fine Mechanics and Physics, Chinese Academy of Sciences), Yuanhao Gong `[通讯]` (Changchun Institute of Optics, Fine Mechanics and Physics, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于CMA-ES的全前向学习框架，用于共享核小波神经网络的图像重建，无需反向传播。

**💡 创新点**

将全局前向搜索与结构感知参数分解和通道特定信用分配相结合，克服了极度参数共享网络在全局搜索中的难点。

**🔧 技术方法**

使用了CMA-ES、共享核小波网络、通道分解与信用分配以及仅前向评估的方法。

**📊 数据集**

使用了BSDS500数据集（256×256图像块）。

**📈 对比分析**

与Adam/BP进行对比，测试MSE从4.1783降至4.0130，PSNR提升至42.787 dB，训练覆盖次数仅为6次，且性能优于传统反向传播。

**⚠️ 局限性**

局限性在于仅适用于极度参数共享的紧凑网络，扩展到更大模型或更复杂任务的可行性未验证，且计算时间与传统BP不具备直接可比性。

---

## 512. FlowVLA: Learning Flow-Aware Physical Completion for Quantitative Robotic Pouring

**arXiv ID:** 2610.29692 | [PDF](https://arxiv.org/pdf/2610.29692v1)

**作者:** Liwen Yang `[一作]` (Zhejiang University), Dibo Hou `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `afceb026-1760-41ae-8d86-010831a37d97` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现一种分离运动生成与物理完成估计的量化倒灌机器人控制框架FlowVLA，解决“物理完成别名”问题。

**💡 创新点**

将物理完成作为持续的状态独立于动作生成，引入边界感知动作学习、流量与质量估计以及残留流预测，并通过目标无关的估计实现对未训练目标的插值。

**🔧 技术方法**

基于π_0.5的多模态动作专家、冻结的GRU观测器、视觉+关节+腕力+时间序列融合的完成头、软门控积分以及残留流的递归预测。

**📊 数据集**

通过DR02机器人在四种容器-流体组合（咖啡壶/水、带把壶/水、酱油/盐水、油/油）收集的1,684条演示，并使用10Hz电子秤生成离线流量/质量标签。

**📈 对比分析**

与π_0.5、ForceVLA、TA-VLA-Wrench等全任务VLA基线以及基于负载阈值的轨迹重放控制进行对比，FlowVLA在咖啡壶/水实验中宏观MAE降低至13.1g、完成率提升至53.7%，对未训练目标和其他流体场景亦保持良好性能。

**⚠️ 局限性**

仍需域特定训练与离线秤标注，依赖传感器精度，对复杂流体或高振动环境的鲁棒性有限。

---

## 513. Scaling Multilingual Pretraining of Pixel Language Models

**arXiv ID:** 2610.29681 | [PDF](https://arxiv.org/pdf/2610.29681v1)

**作者:** Ilker Kesen `[一作]` (Humboldt University of Berlin), Desmond Elliott `[通讯]` (University of Copenhagen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并对比了多语言、跨脚本、不同规模的像素语言模型，系统评估其在低资源语言上的迁移性能。

**💡 创新点**

首次将像素模型扩展到16种语言并系统地分离语言、脚本与容量三轴扩展的影响，发现规模效应局部并揭示了多语言负荷的曲线。

**🔧 技术方法**

使用Vision Transformer Masked Autoencoder架构，结合自适应渲染策略（字母对、单词级）和多语言预训练。

**📊 数据集**

FineWeb、FineWeb2等大规模多语言语料用于预训练；SIB‑200、UD、NER、XNLI、GLUE等下游任务集进行评估。

**📈 对比分析**

与子词和字符级基线相比，在低资源、语言距离大的脚本上取得更高分数，尤其在句子分类、命名实体和依赖分析上优于XLM‑R。

**⚠️ 局限性**

实验点有限、仅限编码器、未覆盖所有语言组合、仅固定每语种算力、单一渲染策略未对比，可能导致结果对特定语言/脚本的依赖。

---

## 514. DriveMLP: When Training Strategy Outperforms Model Complexity in Autonomous Driving

**arXiv ID:** 2610.29676 | [PDF](https://arxiv.org/pdf/2610.29676v1)

**作者:** Alessandro Canevaro `[一作]`, Julian Jordan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在非反应式仿真环境中使用专家轨迹距离奖励的RL训练框架（EGRL），并基于该框架训练轻量级MLP规划器DriveMLP；

**💡 创新点**

将RL仅作为优化工具，利用专家轨迹距离奖励消除IL的分布偏移和RL的奖励工程难题，并证明训练策略对性能提升的主导作用；

**🔧 技术方法**

使用非反应式log-replay仿真、位置与速度指数化的轨迹距离奖励、Soft Actor-Critic/PPO算法、集合编码的多层感知器网络；

**📊 数据集**

在nuPlan（1500小时真实驾驶日志）和Highway-Env基准上进行评估，使用nuPlan的Val14/Hard/Test14等子集；

**📈 对比分析**

与BC、AIRL、BC-RL以及PlanTF、PLUTO、Diffusion Planner、Flow Planner等SOTA方法对比，DriveMLP在nuPlan Reactive评测中获得86.51分，超过Flow Planner 3.2点，且推理速度>15×快；

**⚠️ 局限性**

依赖非反应式log-replay仿真且受专家轨迹质量限制，Sim2Real迁移仍具挑战，且在极端稀有场景下可能表现欠佳。

---

## 515. MS2CAST: A Video Anomaly Detection Framework via Multi-Scale Spatial Cross-Attention and Efficient Spatiotemporal Networks

**arXiv ID:** 2610.29649 | [PDF](https://arxiv.org/pdf/2610.29649v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 516. Conflicted Records: Typed Adversarial Contracts for Auditable LLM Output Quality Enforcement

**arXiv ID:** 2610.29669 | [PDF](https://arxiv.org/pdf/2610.29669v1)

**作者:** Kardelen Subasi `[一作]` (Warsaw University of Technology), Jarosław A. Chudziak `[通讯]` (Warsaw University of Technology)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个多模型异议协作的Typed Adversarial Contract管道，用于对LLM生成记录进行可审计的质量检查。

**💡 创新点**

引入结构化的typed claim记录、双向迭代的跨族模型协商、三方仲裁者投票与基于参照与语义的确定性不变式，能够在同一记录中识别并隔离相互矛盾的质量决策。

**🔧 技术方法**

使用语言模型异议对话（GLM‑4‑Flash、Mistral Small等）、LangGraph orchestrator、三方仲裁（Cohere Command R、Claude Sonnet、Llama 3.3 70B）、Jaccard相似度语义检测、referential coverage检查，以及类型化数据模型和完整 provenance 跟踪等技术。

**📊 数据集**

使用BBQ Age、BBQ Disability（偏见数据集）以及HaluEval QA（Hallucination与正确答案对照）三套数据集进行评估。

**📈 对比分析**

与零样本、系统提示和链式思维等基线比较，BiasGuard‑D在 stereotype 选择率上从 51% 降至 40%，Hallucination 删除率实现 100%，正确答案匹配率从 12% 提升至 61%，不变式满足率约 76%–83%，同时提供可解释的调试信息。

**⚠️ 局限性**

局限性包括对 LLM‑as‑judge 产生标签的偏差、Jaccard 阈值在高仿写语料下的鲁棒性、对付费 Arbiter 的依赖、以及同一模型家族审计者可能导致冲突解析不足等问题。

---

## 517. Leveraging Time-Series Foundation Models for Financial Crime Detection

**arXiv ID:** 2610.29667 | [PDF](https://arxiv.org/pdf/2610.29667v1)

**作者:** Danae Natalie Savvidi `[一作]` (Delft University of Technology), Arie van Deursen `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出 Temporal‑Context Edge Enrichment (TCEE) 框架，利用冻结的时间序列基础模型 (Chronos‑2) 为每笔交易边生成富含时间上下文的表示，并结合节点特征与局部邻域信息，实现对金融交易网络中可疑交易的边级检测。

**💡 创新点**

创新点在于：① 将预训练的时间序列 Transformer 冻结后，对每条交易边从其两端账户的历史序列中查询嵌入，采用注意力直接为边提供时间上下文；② 引入 pair‑pooling 与 neighbourhood‑attention 两个扩展模块，分别聚合账户对之间的交易和交易附近的局部窗口，弥补仅用全局历史所缺的细粒度信息；③ 通过边级表示提升检测性能，而非传统节点级聚合，显著提高了 AML 和 phishing 场景的准确率。

**🔧 技术方法**

技术手段包括：预训练时间序列 Transformer (Chronos‑2) 的冻结编码；单头/多头缩放点积注意力；线性映射、层归一化与 MLP 分类器；图神经网络 (PNA、Multi‑GNN、MEGA‑GNN) 用作可选 backbone；pair‑pooling（PNA 聚合）和 neighbourhood‑attention（局部窗口注意力）等扩展。

**📊 数据集**

使用的数据集：AMLWorld 六个人工生成的交易网络（HI‑Small、LI‑Small、HI‑Medium、LI‑Medium、HI‑Large、LI‑Large）以及以节点分类为目标的 Ethereum phishing 交易网络。

**📈 对比分析**

实验对比采用 GINe、PNAe 以及多种结构化 GNN（MEGA‑GNN、Multi‑GNN、Multi‑FraudGT、DIAM）等基线；结果显示 TCEE‑MLP 在五个 AMLWorld 数据集上 PR‑AUC 均超过最佳基线平均 10.8%，加入 PNA 背骨后进一步提升 4.8%；在 ETH phishing 节点任务上 TCEE‑PNA 亦比基线提升约 15–18% PR‑AUC；同时训练时间比最强基线快 6–7 倍，显示出优越的性能/成本比。

**⚠️ 局限性**

局限性包括：仅在单一 GPU 与单一冻结的 Chronos‑2 编码器上验证；评估数据为人工模拟（AMLWorld），未验证在真实银行或多样化交易数据上的泛化；对更长历史窗口和多尺度时间特征的探索有限；未尝试对 TSFM 进行微调或多模型集成；可解释性仅基于注意力权重，缺乏更深入的因果分析。

---

## 518. Spatially Gated Demographic Steering for Debiasing Multi-Person Text-to-Image Generation

**arXiv ID:** 2610.29651 | [PDF](https://arxiv.org/pdf/2610.29651v1)

**作者:** Venkatesh Thirugnana Sambandham `[一作]` (Technische Hochschule Ingolstadt), Torsten Schon `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种 Spatially Gated Steering (SGS)，实现多人物文本到图像生成中的每个人物独立人口统计属性调整。

**💡 创新点**

创新点在于利用 Diffusion Transformer（DiT）内部注意力特征在关键去噪步骤中提取实例掩码，并在该掩码上独立施加人口属性向量，突破了全局平衡与局部多样性冲突的局限。

**🔧 技术方法**

技术包括：MM‑DiT 自注意力分割器（DETR‑style）、稀疏梯度驱动的正向采样、人口属性向量的离线预计算与 Embedding Arithmetic、以及基于掩码的空间门控累加。

**📊 数据集**

使用自己构造的 1440 张多人物图像数据集（包含 2–5 人、9 场景类型、3 难度等级），并借助 SAM2 自动生成伪标签；另外在公开 WinoBias 职业基准上评估性能。

**📈 对比分析**

与 Embedding Arithmetic、SDID、RespoDiff、FairImagen 等方法在 FLUX.1‑Dev 与 SD3.5‑Large 上进行比较，SGS 在维持或略低 CLIP 语义相似度的前提下，显著提升交叉身份熵至 0.794/0.832，且偏差率 (DR) 仅次于 EA，显示出最优的局部多样性与整体平衡兼顾。

**⚠️ 局限性**

主要局限包括：分割器在 4 人以上或遮挡严重场景下精度下降；每实例前向传播导致计算开销与生成时间成正比；以及未能自动识别并忽略提示中已明确约束的性别/种族信息。

---

## 519. An interpretable deep learning method for early hepatocellular carcinoma diagnosis on multiphase CT

**arXiv ID:** 2610.29616 | [PDF](https://arxiv.org/pdf/2610.29616v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 520. Reading the Room: Scale-ambiguous Semantic Registration of Sparse VSLAM Maps to Floor Plans in Partially Built Environments

**arXiv ID:** 2610.29647 | [PDF](https://arxiv.org/pdf/2610.29647v1)

**作者:** Bjorn Stragier `[一作]` (TU Delft), Lukas Rosenberger Schmid `[通讯]` (University of Technology Nuremberg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

将稀疏的VSLAM地图对齐到部分建成的平面图

**💡 创新点**

提出单一鲁棒、类别平衡的语义几何能量函数与两阶段全局-局部注册框架

**🔧 技术方法**

语义分割、BEV投影、距离变换、跨模态特征匹配、RANSAC、Quasi-Newton优化

**📊 数据集**

33条轨迹、3个施工现场（A、B、C）共33个序列

**📈 对比分析**

与SIFT+RANSAC、RoMa+MatchAnything、区域匹配等基线比较，平均/中位ATE显著下降、召回率提升；RR在无锚点时平均提升4.5×，给锚点时中位ATE <2 m

**⚠️ 局限性**

在最难站点C仍因低覆盖和对称导致失败，且受VSLAM漂移约1 m限制，缺乏非刚性适配

---

## 521. LLMs as survey respondents: Comparison to simple semantic baselines

**arXiv ID:** 2610.29645 | [PDF](https://arxiv.org/pdf/2610.29645v1)

**作者:** Esko Lehtonen `[一作]` (VTT Technical Research Centre of Finland Ltd), Tommi Buder-Gröndahl `[通讯]` (University of Helsinki)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对两个关于自动驾驶和轻电动车态度的调查数据进行半监督的回答预测，评估LLM在已知部分回答的情况下，能否有效补全缺失回答；并与基于句子嵌入相似度的简单基线进行对比。

**💡 创新点**

系统检验LLM在回答生成中的“真正推理”能力与仅靠语义相似度的假设（STSR）之间的差距，揭示LLM在不同调查和缺失率下表现差异，强调在数据增补实验中需同时加入语义相似度基线。

**🔧 技术方法**

使用三大LLM（Gemini 3.7 Flash、GPT 5.6 Luna、Mistral 4 Small）和两种嵌入基线（Gemini embeddings、SBERT），通过系统提示在已知问答对的基础上生成缺失回答；评估指标包括 Cohen’s kappa、协方差矩阵 Frobenius 范数、RV 系数以及因子结构恢复。

**📊 数据集**

AV（自动驾驶接受度）共 22 题、500 受访者（公开数据）；LEV（轻电动车益处）共 21 题、426 受访者（非公开数据）。三种缺失率 20%、50%、80% 的掩码数据集被用来评估模型。

**📈 对比分析**

与基线的比较通过 Cohen’s kappa 及协方差/因子结构重建指标完成。结果显示：LLM 在大部分情形下优于基线，但随着已知回答比例下降，LLM 性能显著下降，有时与基线持平；AV 数据下 LLM 效果更好，主要因为数据可公开训练；LEV 受限于训练覆盖率，LLM 与基线差距缩小。

**⚠️ 局限性**

仅测试了三款英文 LLM 与两种基线，且仅涵盖两类特定调查；未考虑不同语言、文化背景、模型微调与提示策略对结果的影响，限制了结论的普适性。

---

## 522. FogTTA: Online Test-Time Adaptation for Robust Transformer-based Object Detection in Foggy Weather

**arXiv ID:** 2610.29618 | [PDF](https://arxiv.org/pdf/2610.29618v1)

**作者:** Ali Abedi `[一作]`, Shiqi Tian `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了FogTTA框架，实现在雾天环境下的Transformer对象检测器（RF-DETR）的在线测试时自适应；

**💡 创新点**

创新点在于教师-学生设计：教师生成弱增强图像的伪标签并通过NMS与置信度过滤后，再用Varifocal Loss对强增强图像进行学习，随后以EMA方式更新教师，既实现无源数据、实时在线适应，又提升了伪标签去噪能力；

**🔧 技术方法**

采用的技术包括RF-DETR检测器、弱/强数据增强、伪标签置信度阈值、NMS、Varifocal Loss、EMA（指数移动平均）以及伪标签重采样策略；

**📊 数据集**

使用Cityscapes作为源域训练集，Foggy Cityscapes作为目标域（无标签）进行测试时适应；

**📈 对比分析**

与多种TTA和跨天气域适应基线（如MLFA、MemCLR、DDT、FRANCK等）进行对比，FogTTA在Cityscapes→Foggy Cityscapes任务上取得mAP 57.6%，比源域模型提升约1.5%，仅次于Oracle 60.8%，并保持实时推理速度；

**⚠️ 局限性**

局限性包括：仅在雾天场景验证，无法保证对更广泛域移（如雨、夜等）的适应；伪标签阈值和NMS参数需手动调优；在极端雾霾条件下仍可能出现误检和定位不精确。

---

## 523. Watch Your Step: Anticipated Perception for Humanoid Locomotion

**arXiv ID:** 2610.29612 | [PDF](https://arxiv.org/pdf/2610.29612v1)

**作者:** Matthias Heyrman `[一作]` (ETH Zurich), Marco Hutter `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种轻量化的脚踏点预测模型（FPM），仅从本体感知和速度指令预测下一步两次脚踏点，并据此在地形高度图上采样局部区域，随后使用 PPO 强化学习训练行走策略，从而实现对地形的高效感知与控制。

**💡 创新点**

创新点在于：① 在感知层面通过自监督的 FPM 预先定位脚踏点，避免全图编码器的冗余与注意力机制的高计算成本；② 与行走策略自适应 co‑training，动态更新 FPM 以跟随策略进化的行走模式；③ 仅对局部地形进行采样，显著提升样本效率和泛化能力。

**🔧 技术方法**

技术手段包括：IsaacLab 平台的 PPO 强化学习、异步 actor‑critic、基于 MLP 的 FPM、k×k 高程图采样、助力力学训练 curriculum、域随机化、对比 CNN 编码器和注意力编码器等。

**📊 数据集**

数据集为自生成的多难度地形集合（踏石、楼梯、盒子、斜坡、随机粗糙、坑洞等），以及若干未见过的 OOD 地形（波浪、矩形障碍、高幅度随机地形、同心间隙、稀疏托盘），并在 Unitree G1 上进行硬件测试。

**📈 对比分析**

通过与完整地图、Wild、Naive、CNN 编码器以及 AME2 等方法比较，FPM 在训练迭代上提升约 3.4‑3.5 倍、在训练和 OOD 地形的成功率和跌落率均优于大多数方法，硬件上实现了零射（zero‑shot）迁移，性能与注意力编码器相当但训练成本显著降低。

**⚠️ 局限性**

局限性包括：预测 horizon 仅能覆盖至第二步脚踏点，侧向突入障碍不易被及时检测；对连续全景地形无显著优势；FPM 预测误差大时可能导致错误采样；在极端未知或高度差异明显的地形上表现不佳。

---

## 524. AirPlayground: Benchmarking Aerial Navigation across Diverse Reconstructed Environments

**arXiv ID:** 2610.29699 | [PDF](https://arxiv.org/pdf/2610.29699v1)

**作者:** Rangya Zhang `[一作]` (Nanyang Technological University), Mir Feroskhan `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出AirPlayground基准，构建94个可执行的3D Gaussian Splatting（3DGS）环境，配合94K密集飞行演示和46.5M同步RGB–状态–动作样本，统一API用于不同导航策略的训练与闭环评估；

**💡 创新点**

①将高分辨率3DGS重建与连续控制数据相结合，实现可执行真实世界场景；②设计跨策略适配器，允许不同模型共享同一训练数据和评估协议；③利用可执行场景实现离线预测与闭环表现的诊断，揭示误差结构、执行窗口与场景几何对性能的影响；

**🔧 技术方法**

3D Gaussian Splatting重建、LiDAR+多视角摄像同步、A*路径规划与直线加速、2D占用网格与3D清晰度场、GPU批量碰撞查询、控制器适配器、基于ACT、Diffusion Policy、CNN-BC等多种策略；

**📊 数据集**

94个真实场景（18室内、48城市外景、28植被）共94K演示，46.5M RGB-状态-动作样本，约259小时飞行；与CityNav、OpenFly、UAV-Flow等基准进行对比；

**📈 对比分析**

在点目标、图像目标、无目标三种导航设置下，对七个基线（CNN‑BC、ACT、DP、ViNT、OmniVLA、NWM、NavMorph）进行统一评估。成功率范围21%–47%，ACT在训练场景从5至70个扩展后，见场景SR从30.6%提升至41.2%，未见场景SR从16.8%提升至29.4%；离线误差与闭环成功率相关性低；不同策略在不同场景与设置下表现互补；

**⚠️ 局限性**

数据集仍未覆盖全部环境类型；离线误差指标不一定能预测闭环性能；对动态障碍物处理有限；评估仅考虑固定高度与单一控制指令，缺乏多高度、多目标的泛化能力；

---

## 525. Uni-Mem: Learning a Long-Horizon Agentic Memory Manager with Unified Online Policy Optimization

**arXiv ID:** 2610.29671 | [PDF](https://arxiv.org/pdf/2610.29671v1)

**作者:** Zhiqian Xia `[一作]` (Southeast University), Haifeng Xia `[通讯]` (Sun Yat-sen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了统一的记忆管理器 Uni‑Mem，能够通过端到端在线强化学习同时学习记忆构建与检索。

**💡 创新点**

创新点包括：Stage‑Decoupled GRPO 通过交替优化构建和检索阶段消除组内比较混淆；Event‑Wise Credit Assignment (ECA) 在事件层面局部化奖励并追踪更新链，实现长时序奖励的精确归因。

**🔧 技术方法**

技术方法包括：共享参数的工具调用式记忆管理代理；改进的 GRPO 强化学习框架；事件级奖励分配与更新鲁棒性追踪；基于冻结答案模型的证据指数评分。

**📊 数据集**

使用 LoCoMo 数据集进行训练与评估，并在 LongMemEval（Oracle 与 S）上进行跨域泛化测试。

**📈 对比分析**

与 Memory‑R2、Mem‑T 等现有 RL 记忆管理基线比较，Uni‑Mem 在 LoCoMo 上整体 J 分数达到 88.80%，超过最佳基线 77.87%（+10.93%），在 LongMemEval 上也显著优于所有对照组。

**⚠️ 局限性**

局限性包括：训练成本高，需完成完整问答并调用外部评判模型；评判过程耗时且对 API 依赖强，可能导致延迟与费用；当前缺乏更高效的过程级监督机制。

---

## 526. OpenEarth-OVS44: A Multimodal Remote Sensing Dataset for Open-Vocabulary Segmentation

**arXiv ID:** 2610.29642 | [PDF](https://arxiv.org/pdf/2610.29642v1)

**作者:** Yimin Wei `[一作]` (University of Tokyo), Naoto Yokoya `[通讯]` (University of Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并构建了OpenEarth-OVS44遥感细粒度多模态分割基准，包含44类专家标注的RGB及RGB‑SAR图像，并设立RGB、RGB‑SAR云覆盖及跨地区三种评估轨道。

**💡 创新点**

突破性在于提供细粒度44类多模态标注，兼顾云覆盖、跨域，并为开放词汇分割在遥感领域提供统一评测平台，填补现有粗分类和多模态不足。

**🔧 技术方法**

采用大型视觉‑语言模型（如CLIP）进行开放词汇分割，结合RGB‑SAR融合方法MM‑OVSeg，并在多种训练与零样本设置下进行实验评估。

**📊 数据集**

使用OpenEarth-OVS44自身数据：2,832张RGB图像与1,341张精确配准的RGB‑SAR对，覆盖法国、日本、美国40座城市，共计290,587份专家验证的分割。

**📈 对比分析**

与七种主流OVS模型（包括训练无模型、训练模型及多模态模型）进行对比，零样本模型mIoU仅在14–16之间，训练模型在最佳可达34.4 mIoU，跨域与云覆盖场景下性能仍显著下降。

**⚠️ 局限性**

局限在于细粒度视觉‑文本对齐不足、稀有类别识别差、文本提示过于简洁、低可视化条件下鲁棒性不足，导致模型在某些类别和恶劣天气下仍表现不佳。

---

## 527. Recursive Code Evolution for Task-Oriented Semantic Communications

**arXiv ID:** 2610.29641 | [PDF](https://arxiv.org/pdf/2610.29641v1)

**作者:** Yan Gong `[一作]` (Bournemouth University), Pei Xiao `[通讯]` (University of Surrey)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出ComRSI框架，利用大型语言模型递归编辑通信程序，实现任务导向语义通信的可持续改进。

**💡 创新点**

将可递归的代码进化与受保护的评估管道结合，形成可审计的程序自我改进过程，并通过语言模型实现结构化的代码变更。

**🔧 技术方法**

使用Opus等大型语言模型进行代码编辑，配合代码合法性与资源约束检查、随机/贝叶斯搜索、AWGN与衰落信道评估、注意力池化等技术。

**📊 数据集**

在CIFAR‑100数据集上进行主实验，并在Tiny‑ImageNet上验证迁移效果；CIFAR‑10用于开发与验证。

**📈 对比分析**

与人类设计、手调、非递归代理、随机搜索、Optuna TPE等方法对比；最终在AWGN‑5下仅32个复数符号实现74.3%准确率，明显优于初始256符号的67.2%；十次独立搜索平均公共得分0.7587，优于NR‑H(0.7376)与NR‑0(0.7089)，统计显著。

**⚠️ 局限性**

实验仅在模拟无线信道的图像分类任务上验证，未考虑实际调制、信道估计、能耗、延迟等部署约束；仅优化单任务通信，未将代理自身进化结合。

---

## 528. Reliability-Aware Probabilistic Fusion for Multimodal Learning with Incomplete Evidence

**arXiv ID:** 2610.29621 | [PDF](https://arxiv.org/pdf/2610.29621v1)

**作者:** Patitapaban Palo `[一作]` (University of Oxford), David A. Clifton `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可靠性感知的高斯产品专家（Reliability‑Aware Gaussian PoE）融合框架，用于在多模态输入缺失或降质的场景下进行预测。

**💡 创新点**

创新点包括：①将潜在不确定性与可靠性权重分离，可靠性权重根据输入自适应调节专家贡献；②在PoE中直接忽略缺失模态的专家；③对该融合算子的统计性质进行严格推导并分析可靠性与可用性对后验信息的影响。

**🔧 技术方法**

技术手段主要是：Gaussian Product‑of‑Experts、输入依赖的可靠性权重网络、子集采样（模态缺失模拟）与一致性（consistency）训练、以及在缺失模态下的自适应推理。

**📊 数据集**

使用的公开临床数据集有：Symile‑MIMIC（胸部X光、心电图、实验室测量，心脏肥大预测）和 HAIM‑MIMIC‑MM（结构化数据、时间序列、胸部X光、临床文本，48 小时死亡预测）。

**📈 对比分析**

与早期融合、Late Average、DrFuse、M³Care、MUSE、Flex‑MoE、模态丢弃等方法进行对比；在完整输入下与最强方法相当；在缺失模态时平均 AUROC 最高（Symile 0.701，HAIM 0.798），尤其在 HAIM 上提升约 0.015；在人工腐败条件下保持稳定，虽存在极端降质导致可靠性失效的情况。

**⚠️ 局限性**

局限性包括：可靠性权重并非校准的质量概率，极端降质时可能误判；仅在两份临床基准上验证，使用预计算特征表示；未对输出概率做完整校准评估；训练未覆盖所有腐败类型，导致可靠性学习在某些场景下表现不一致。

---

## 529. COMET: A coordinated multi-layer runtime monitoring framework for human-supervised AI-enabled safety systems

**arXiv ID:** 2610.29816 | [PDF](https://arxiv.org/pdf/2610.29816v1)

**作者:** Niclas Flehmig `[一作]` (Norwegian University of Science and Technology), Shen Yin `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a4b10f5d-130b-4e77-9367-6469ec621899` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了COMET多层协同运行时监控框架，用于人类监督的AI安全系统，能够在AI模型输出不可靠时自动检测、定位并向人类主管提供可操作的诊断信息。

**💡 创新点**

创新点在于将三类互补监控（信号完整性规则、基于物理模板的频谱一致性检测、特征空间k‑NN OOD检测）通过协调层融合成诊断层级，并通过交通灯式通信层向操作员呈现统一的风险等级，实现了从单一检测到可解释故障诊断的跃迁。

**🔧 技术方法**

采用基于规则的信号完整性监控、频谱模板与阈值的物理一致性监控、对CNN特征空间的k‑NN距离判定，以及层间z‑score分析提取语义/协变量偏移特征，并在协调层使用规则树进行诊断组合。

**📊 数据集**

使用公开的IDMT‑ISA‑Compressed‑Air压缩空气泄漏检测数据集作为AI安全系统的实验基准，包含多种背景噪声、泄漏类型及人工注入的传感器故障。

**📈 对比分析**

与无监控和单监控（最大softmax、L2物理层、L3特征层）对比，COMET将可信错误率从最高37.5%降至0%，对偏移条件下的误报率几乎为零，同时保持约1%的零件误报率，显示出在covariate、semantic及传感器故障三种未见偏移下的显著性能提升。

**⚠️ 局限性**

局限在于仅在单一泄漏检测任务上验证、物理一致性监控依赖特定可建模的物理知识、对剪切信号的规则存在漏检、以及对未覆盖的语义/协变量偏移仍有部分隐式漏报，且缺乏与实际人类操作员交互的实验验证。

---

## 530. Real PMIDs, Wrong Papers: Harness Design for Biomedical Claim-Verification Agents

**arXiv ID:** 2610.29812 | [PDF](https://arxiv.org/pdf/2610.29812v1)

**作者:** Mohammed Saqlain `[一作]` `[通讯]` (Independent Researcher), Mohammed Saqlain (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

使用 PubMed API 对生物医学命题验证（SciFact）进行控制消融实验，比较闭卷、单句检索、ReAct 生成查询和 oracle 四种检索 harness 的效果。

**💡 创新点**

首次系统评估不同检索 harness 对科学助手性能的影响，揭示模型自行生成查询显著提升准确率。

**🔧 技术方法**

利用大语言模型（LLM）结合 PubMed API、ReAct 样式代理、闭卷回答和 oracle 等技术。

**📊 数据集**

SciFact 语料库与 PubMed 数据。

**📈 对比分析**

对四种 harness 进行对比实验：闭卷 49–59% 准确率，单句检索仅 6% 召回，ReAct 生成查询 62% 准确率，oracle 79–86% 准确率；差异在样本量下不显著，但表明检索是主要瓶颈。

**⚠️ 局限性**

样本量有限导致统计显著性不足；检索仍是主瓶颈；闭卷模型引用 PubMed ID 存在但缺乏相关性，提示存在性检查不足。

---

## 531. Compositionality with Primitives: A Unified View from Euclidean to Hyperbolic Vision-Language Models

**arXiv ID:** 2610.29811 | [PDF](https://arxiv.org/pdf/2610.29811v1)

**作者:** Puru Vaish `[一作]` (University of Twente), Nicola Strisciuglio `[通讯]` (University of Twente)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在预训练视觉语言模型中，对嵌入进行本征/切空间分解，直接使用属性与对象的原语进行开放式零样本组合识别。

**💡 创新点**

提出不需要显式属性-对象再组合，利用原语在不同流形几何下直接评分即可实现开放式组合零样本学习。

**🔧 技术方法**

采用 Geodesically Decomposable Embeddings (GDE) 框架、切空间与本征均值计算、欧氏/球面/双曲几何映射以及自定义原语评分函数。

**📊 数据集**

在 MIT-States、VAW-CZSL、UT-Zappos50K、Clothing16K、Waterbirds 与 CelebA 等公开基准上进行评估。

**📈 对比分析**

与基于文本对的零样本配对、GDE 配对及传统 Fine‑tuned 方法对比，原语直接评分在开放式 CZSL 任务中获得更高准确率，特别在双曲空间中表现突出。

**⚠️ 局限性**

局部线性化误差导致切空间原语在双曲空间中不稳定，且对数据集语义方差敏感，需要进一步改进原语提取与评分机制。

---

## 532. On the maximum length of cyclic MDS codes

**arXiv ID:** 2610.29810 | [PDF](https://arxiv.org/pdf/2610.29810v1)

**作者:** Gaojun Luo `[一作]` (Nanjing University of Aeronautics and Astronautics), Yang Li `[通讯]` (Nanyang Technological University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究并证明了在多种条件下非平凡单根循环MDS码的长度上界为q+1。

**💡 创新点**

首次将Fourier支撑平衡性、局部恢复码的 Singleton‑like 界、数论中的阶数与 p‑adic 估计等方法结合，用以约束循环 MDS 码的长度。

**🔧 技术方法**

采用 Fourier 支撑分析、局部恢复码理论、乘法阶数与算术约束、域降维和 Lifting‑the‑Exponent 恒等式等数学技术。

**📊 数据集**

无数据集，全部为理论推导。

**📈 对比分析**

对已有的 MDS 长度上界做了扩展，证明在更多长度取值范围内满足上界，性能在理论上达到最优。

**⚠️ 局限性**

仅覆盖满足特定素因子或阶数约束的长度，未处理所有可能长度，仍有未解决的案例。

---

## 533. Asymmetric Likelihood-Free Hypothesis Testing

**arXiv ID:** 2610.29798 | [PDF](https://arxiv.org/pdf/2610.29798v1)

**作者:** Gunjan Kumar `[一作]` (IIT Kanpur), Jonathan Scarlett `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

在无似然假设检验（LFHT）中，研究了参考样本预算不等（s_P≠s_Q）的情形，并给出了目标样本预算s_D与参考预算之间的可行性三元组（s_P,s_Q,s_D）的完整近似刻画；同时将此结果应用于非自适应分布两聚类问题，得到与域大小、分布数k、最小簇大小r、分离度ε相关的匹配上下界，揭示了自适应与非自适应样本复杂度之间最高可达Θ(k^1/4)的差距。

**💡 创新点**

1）首次给出不等参考预算下LFHT的可行性三元组，证明不等预算可实现多项式量级的目标样本节省；2）将该结果用于非自适应两聚类，得到近似最优样本复杂度；3）确立自适应与非自适应之间的最佳适应性缺口。

**🔧 技术方法**

使用分布“展开（flattening）”将原始分布映射到更大的域以控制最大概率；构造二次统计量并利用无偏估计与Chebyshev不等式；采用重构的heavy/light构造与Paired扰动进行下界证明；Poisson化样本处理与Le Cam理论结合。

**📊 数据集**

论文为理论研究，不依赖具体实验数据集，所有结果均在抽象概率模型与理论分析框架下给出。

**📈 对比分析**

与现有对称LFHT和不等样本接近度测试基线相比，所提方法在目标样本预算上实现了多项式级别的提升（例如从O(√n)下降到Θ(n^{1/4}）），在非自适应聚类任务中也与已知最佳下界保持匹配，上界与下界相差多项式次数的对数因子。

**⚠️ 局限性**

结果仅在对数因子下给出近似，精确常数未确定；假设域大小n≥10，且仅适用于两分布场景；自适应缺口分析在特定参数范围内给出，未覆盖全部可能配置。

---

## 534. D-Pingmesh: Dynamic Probing and Fault Localization for Reconfigurable Datacenter Networks

**arXiv ID:** 2610.29787 | [PDF](https://arxiv.org/pdf/2610.29787v1)

**作者:** Yunze Wei `[一作]` (Tsinghua University), Yong Cui `[通讯]` (Tsinghua University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在可重构数据中心网络中，设计并实现了一套控制面板感知的探测与故障定位系统D‑Pingmesh，能够在网络拓扑与流量工程动态变化时高效生成探测任务、增量更新探测列表，并利用边界感知的EM算法准确定位故障。

**💡 创新点**

创新点主要包括：① 基于实时TPE/TE状态的哈希感知路径解析与接口多覆盖任务生成，显著减少探测量；② 针对网络重配置的增量式探测列表更新算法，避免从零开始重新计算；③ 结合旧/新路径混合权重的Rho‑EM边界定位器，有效解决跨配置窗口的路径不确定性。

**🔧 技术方法**

采用哈希感知路径解析、概率引导的贪心多覆盖生成器、动态增量更新流程以及Rho‑EM EM式边界定位算法；同时利用ns‑3仿真与实际硬件（192×192 OCS）进行验证。

**📊 数据集**

使用Gemini生成的8‑pod 2048‑服务器OCS拓扑的合成流量（F5、F12、F20三种流量偏斜CDF），共22个一日仿真轨迹；在物理试验台上部署四个pod与一台192×192 OCS进行验证。

**📈 对比分析**

与Pingmesh、deTector*、RD‑Probe*等传统探测/定位系统比较：D‑Pingmesh在99.99%覆盖率下仅需约10K任务、生成时间≈2.5s，Pingmesh需712K任务、生成时间≈10s；在增量更新下比冷启动快≈2.4×；Rho‑EM在边界配置下的微F1从0.871提升至0.953，物理实验中故障定位延迟约2s。

**⚠️ 局限性**

局限性：假设每接口失效率在探测窗口内独立且恒定，无法处理共享风险或时变失效；需要准确的哈希行为与控制面板信息，部署依赖控制面板与探测器协同；在极小或几乎静态网络中可能不如传统Pingmesh简洁；对更快的TE/TE配置速率仍需进一步优化探测窗口与采样策略。

---

## 535. Monitorability Is Trainable: Post-Training for Chain-of-Thought Monitorability

**arXiv ID:** 2610.29734 | [PDF](https://arxiv.org/pdf/2610.29734v1)

**作者:** Usman Anwar `[一作]` (University of Cambridge), David Krueger `[通讯]` (University of Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种无监督的后训练方法，通过最大化链式推理（CoT）与模型输出之间的条件互信息来提升CoT可监控性，并加入下尾风险（LCVaR）聚焦最少信息的样本和条件χ²约束以保持行为保守性，防止模型刻意隐蔽推理；

**💡 创新点**

创新点在于：1）将CoT可监控性视为可训练的表征转移任务，使用条件互信息作为通用训练信号；2）通过下尾目标关注最不易监控的轨迹，提升鲁棒性；3）用χ²约束阻止模型对CoT进行私密编码，从而保持行为安全；

**🔧 技术方法**

技术核心包括：无监督互信息估计（基于采样的MI奖励）、LCVaR下尾损失、条件χ²正则化、β调节器、梯度遮蔽等；

**📊 数据集**

实验数据集涵盖：MBPP、Synthetic‑Document Fine‑Tuning (SDF) 任务、HarmBench、StrongREJECT、SORRY‑Bench、SCRUPLES、GPQA Diamond 等；

**📈 对比分析**

对比方法主要是基线模型（未改造的CoT）和仅使用MI或仅使用χ²的版本；在所有实验中，后训练显著提高了监控召回率、降低误报率、提升安全预测的F1/平衡准确率，并在多模型、多任务中恢复或提升CoT的可监控性；

**⚠️ 局限性**

局限性包括：实验仅在受控模型体系上验证，未检验在更大、更复杂模型上的可扩展性；对不同监控器、提示方案和误行为的迁移性仍未知；后训练可能增加CoT长度，影响推理效率；

---

## 536. Reconstruction-Augmented Visuomotor Pre-Training of 3D-Grounded Bimanual Dexterity from RGB-Only Human Videos

**arXiv ID:** 2610.29716 | [PDF](https://arxiv.org/pdf/2610.29716v1)

**作者:** Xingyi He `[一作]` (Cornell University), Kuan Fang `[通讯]` (Cornell University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a4b10f5d-130b-4e77-9367-6469ec621899` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用RGB视频重建场景几何与手部运动，构建3D对齐的视觉‑运动预训练框架；

**💡 创新点**

在大规模人类视频中同时引入重建几何作为输入与监督，采用双门残差融合与前向动态预测，并用手腕轨迹条件指令细化指尖动作；

**🔧 技术方法**

使用基于VLM的视觉语言模型（如PaliGemma）、3D手部估计（HaWoR + MANO）、图像深度重建（VGGT‑Omega）以及双门融合与前向预测模块；

**📊 数据集**

在1M条来自Something‑Something v.2、EPIC‑Kitchens‑100、Ego4D和Ego‑Exo4D的视频上进行预训练；

**📈 对比分析**

在九个仿真任务和六个真实世界双手操作任务中，平均成功率分别提升至76.3%和81.7%，比最强基线提升约16.5分和38.3分；

**⚠️ 局限性**

受限于手部估计和场景重建误差，需深度传感器支持，且迁移到新机器人仍需少量演示以校正运动空间与控制接口。

---

## 537. Beyond Policy Engines: Binding-Complete Execution Admission

**arXiv ID:** 2610.29701 | [PDF](https://arxiv.org/pdf/2610.29701v1)

**作者:** Lin Wang `[一作]` (Chengdu Havenlon Security Technology Co., Ltd.), Jiang Deng `[通讯]` (Chengdu Havenlon Security Technology Co., Ltd.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出Binding‑Complete Execution Admission Contract（BCEAC），将政策评估结果与可执行授权之间的缺口弥合，明确了在执行前需要满足的治理来源完整性、候选操作绑定、上下文一致性等独立可验证条件，并构建了与之兼容的引擎无关执行入门架构。

**💡 创新点**

创新点在于：① 明确区分政策决定与执行授权的判定，提出可独立验证的完整性契约；② 设计了引擎无关的可绑定接口，允许 OPA、Cedar 或未来的 EBL 原生运行时在同一契约下替换；③ 通过两条实验轨迹给出有限但可复制的评估与验证样例，提供了可审计的失败与修复历史，验证了契约的可测试性。

**🔧 技术方法**

主要技术包括：治理来源解析与槽位匹配（基于已接受的宪法和事实）、策略执行适配器、候选绑定验证、执行边界语言（EBL）框架以及独立的验证门控实现。

**📊 数据集**

实验使用了两份数据集：① GSC‑1（44个命名案例、48个变体、360条时序记录），② OPA/Cedar 实验（72个场景，144 条严格案例记录，包括 120 条本地评估记录、24 条预评估终止记录及 8 条宽容控制），并结合了已归档的原始评估记录与离线审计日志。

**📈 对比分析**

比较方法：对同一组案例分别使用 OPA 与 Cedar 进行评估，并通过离线审核核对门控结果与原始评估的一致性；在 GSC‑1 中对源完整性、路径绑定与结果匹配进行逐点对比。性能方面记录了 360 条时间测量，显示在当前实现下评估与验证的延迟在毫秒级，满足典型授权场景需求；但未对不同引擎的总体成本或吞吐量做系统性基准。

**⚠️ 局限性**

局限性包括：① 未证明契约在真实生产环境中能抵御所有攻击，缺乏完整的执行边界与外部效果证明；② 对宪法完整性假设（A0）和当前性证明仅在实验环境中做了有限检查；③ 只验证了有限的案例集合，无法推广到所有业务场景；④ 依赖于预先固定的源清单，无法动态发现未列入宪法的新治理主体；⑤ 未对 OPA/Cedar 与 EBL 之间的语义等价性做全面证明。

---

## 538. Prefix Visibility Ratio: Quantifying Partial Observability in Carpet Bombing Detection

**arXiv ID:** 2610.29802 | [PDF](https://arxiv.org/pdf/2610.29802v1)

**作者:** Farid Naït-Abdesselam `[一作]` (Université Paris Cité), Nadjia Kara `[通讯]` (École de Technologie Supérieure (ÉTS))

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了Prefix Visibility Ratio (PVR)指标，用于量化“地毯式轰炸”攻击中被攻击目标的可观测程度，并在此基础上设计了可视化感知网络威胁分析框架VANTA；

**💡 创新点**

创新点在于将攻击目标可观测性作为独立评估维度，构建在线PVR估计器并将估计值与检测器融合权重相结合，实现对局部可观测性变化的自适应检测；

**🔧 技术方法**

技术手段包括基于路由器侧每目标入口率的异常检测、基于时间加权移动平均的PVR估计、置信度评分以及基于PVR阈值的线性融合权重调节；

**📊 数据集**

实验使用Mininet构建的/24目标前缀，涵盖50个合法客户端与100个分布式攻击源，覆盖多种地址利用率、攻击扩展和PVR配置，共计2160个实验实例；

**📈 对比分析**

与三种任务适配的参考实现（NetRadar-S、FlowLens-S、Whisper-S）在匹配可观测性条件下对比，VANTA在PVR≤10%时F1≈95.9%、召回≈92.4%，总体在所有实验中实现精度99.18%、召回93.46%、F1 96.24%，平均检测延迟约4.1 s；

**⚠️ 局限性**

局限性包括：实验规模受限于/24前缀、合成流量、监测覆盖与攻击目标的耦合未解耦、缺乏对组件的单独消融实验、未评估适应性攻击者和真实运营环境中的PVR分布。

---

## 539. Where Do LLMs Learn to Recommend AI?

**arXiv ID:** 2610.29785 | [PDF](https://arxiv.org/pdf/2610.29785v1)

**作者:** Benaya Trabelsi `[一作]` (Bar-Ilan University), Sarit Kraus `[通讯]` (Bar-Ilan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在开放式建议中对AI相关内容的偏好，并通过预训练、后期微调与提示词变化来分析其来源。

**💡 创新点**

首次系统量化不同训练阶段和提示词对AI推荐比例的影响，并揭示后期微调并非单向放大该偏好，而是取决于具体数据与目标。

**🔧 技术方法**

采用词汇检测与LLM过滤器（Qwen3.6-27B、DeepSeek-V4-Flash）、Bootstrap置信区间、Holm多重检验等技术。

**📊 数据集**

使用20个开放式建议问题（覆盖学习、职业、创业、科研四个领域），每个问题10个同义句；并对十条公开基线与指令调优模型、十一条受控后期训练模型进行实验。

**📈 对比分析**

通过对比基线与指令调优、不同微调策略、提示词与AI上下文的影响，发现指令调优可提升约32个百分点，去除AI示例可降低约14个百分点，时态与最高级提示分别提升约18和8个百分点；整体偏好变化随数据与目标而异。

**⚠️ 局限性**

局限在于仅检测显式AI术语，未评估建议质量或隐式推荐；检出率86.6%；实验仅针对单模型单训练回合，可能不具备跨模型、语言或领域的普适性。

---

## 540. Do Flow-Generation Tools Matter? Unsupervised Characterization of MQTT Traffic Across Public Datasets

**arXiv ID:** 2610.29774 | [PDF](https://arxiv.org/pdf/2610.29774v1)

**作者:** Mohammed Rashed `[一作]` (Universidad Carlos III de Madrid), Celeste Campo `[通讯]` (Universidad Carlos III de Madrid)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对四个公开MQTT数据集进行流级别特征提取，并通过三种流生成工具（Zeek‑flowmeter、Argus、NTLFlowLyzer）生成超过38万条流记录，随后使用K‑means、DBSCAN和HDBSCAN三种无监督聚类算法（含PCA降维）对不同工具生成的流进行聚类，评估聚类质量与原始标签的一致性，探究流生成工具对聚类结果的影响；

**💡 创新点**

首次系统比较不同流生成工具在MQTT流特征空间构建中的差异，揭示流生成是非中性预处理步骤，并证明工具、数据集、聚类算法和特征空间设置共同决定聚类性能；

**🔧 技术方法**

流生成（Zeek‑flowmeter、Argus、NTLFlowLyzer）、特征预处理（去除单值字段、类别编码）、PCA降维、K‑means/DBSCAN/HDBSCAN聚类、Silhouette与Adjusted Rand Index评价；

**📊 数据集**

Artemis、MQTTset、MQTT‑IoT‑IDS2020、GothX四个公开MQTT数据集（共1.74M正向包、10.74M恶意包）；

**📈 对比分析**

通过比较不同工具、算法与降维设置下的Silhouette和ARI值，发现Zeek‑flowmeter在PCA10下表现最优，但没有单一工具在所有数据集、算法和指标上持续领先；

**⚠️ 局限性**

仅考虑未加密的MQTT流，使用默认流生成配置，样本子集选择有限，且ARI受原始标签匹配不完全影响；未来需扩展至加密MQTT、更多流生成工具和参数敏感性分析。

---

## 541. Bridging Foundational and Agentic Capabilities of LLMs

**arXiv ID:** 2610.29751 | [PDF](https://arxiv.org/pdf/2610.29751v1)

**作者:** Wei Guo `[一作]` (Nankai University), Guang Zhang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套框架，将大型语言模型的基础能力（从离线基准获得）与交互过程中的操作（通过轨迹标注得到）对应，并定义能力实现差距，评估其对任务表现的影响。

**💡 创新点**

创新点在于：①将基础能力与执行操作在同一模型上进行双层建模；②通过模型族交叉验证量化基础能力预测操作的有效性；③发现并量化“能力实现差距”在不同模型族中的系统性表现；④证明操作评分在预测任务奖励上比单纯基础评分更有效。

**🔧 技术方法**

采用岭回归、Pearson/Spearman相关、残差诊断、Q² 预测技能等统计方法；对操作评分采用 0/0.5/1 规则并按轨迹平均；对能力实现差距进行残差计算；对任务奖励使用 Spearman、MAE、RMSE、Q² 等指标。

**📊 数据集**

使用“Banking”任务的交互轨迹数据（约 15 个模型服务，跨 8 家族），以及多种公开基准（MultiChallenge、MMLU-Pro、LongBench v2、GPQA、IFBench 等）构建基础能力分。

**📈 对比分析**

通过模型族交叉验证与基准模型（训练均值）对比，发现：①基础能力对操作的预测 Q² > 0，说明存在显著关联；②操作评分在任务奖励预测上 Spearman ρ 最高（0.693），MAE 最低（4.83），Q² 最高（0.702）；③加入基础能力进一步提升到 ρ 0.776、MAE 4.63、Q² 0.733，表明两者互补。

**⚠️ 局限性**

局限性在于：①仅在单一 Banking 域评估，无法保证在其他任务或更复杂交互场景中的泛化；②操作定义和标签的主观性可能影响结果；③缺乏对不同工具种类和复杂性对能力实现差距的细粒度分析；④实验规模有限，未覆盖所有主流大模型家族。

---

## 542. LEAF-Mamba: Feature Adaptation and Fusion for Whole Slide Image Analysis

**arXiv ID:** 2610.29746 | [PDF](https://arxiv.org/pdf/2610.29746v1)

**作者:** Lubin Gan `[一作]` (University of Science and Technology of China), Xiaoyan Sun `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出一种在全切片图像（WSI）上使用冻结特征进行弱监督学习的框架，通过自编码器与重构正则化的“潜在证据适配器”（LEA）将预训练的图像嵌入转换为任务特定的低维代码，并结合“Mamba导向的证据融合”（MEF）实现长距离与局部序列交互的并行流，最终实现滑动级别的诊断分类和生存预测。

**💡 创新点**

创新点包括：① 将重构损失与任务损失联合训练的适配器，显式约束特征的可恢复性；② 在同一层级维护长程Mamba状态空间模块和短程卷积模块的独立表示，并通过输入相关的门控机制进行动态融合；③ 在冻结特征条件下系统化地比较多种特征提取器（ResNet‑50、PLIP、CONCH v1.5）和任务，展示了适配+融合组合在多任务上的优势。

**🔧 技术方法**

技术手段：多实例学习（MIL）框架、Mamba状态空间模型、深度残差瓶颈适配器、重构正则化、输入相关门控融合、Gated Attention Pooling、Cox比例风险模型用于生存预测。

**📊 数据集**

使用数据集包括：BRACS（乳腺病灶分类），Camelyon16（癌症转移检测），EBRAINS（脑肿瘤分型），以及五个TCGA癌症队列（BLCA、BRCA、COADREAD、STAD、HNSC）进行生存预测；特征提取器分别为ResNet‑50、PLIP和CONCH v1.5。

**📈 对比分析**

方法与多种基线（MeanMIL、CLAM、S4MIL、RRTMIL、MambaMIL、Jigsaw等）进行对比，所有冻结特征条件下均采用统一的划分和评估协议。实验显示，所提框架在诊断分类中平均AUC提升约1–2个百分点，在生存预测中的C‑Index平均提升约0.2–0.3个百分点；尤其在BRACS、Camelyon16和TCGA-BRCA等具体任务上表现最为突出。

**⚠️ 局限性**

局限性：① 仅对冻结特征进行适配，未探索编码器微调带来的潜在改进；② 采用固定的基于坐标的序列化，仅考虑一维序列交互，未利用二维空间关系；③ 评估仅覆盖公开数据集，缺乏跨机构、跨制备条件的泛化验证。

---

## 543. On Almost MDS codes holding $(n-k-1)$-designs and an open problem on Witt design

**arXiv ID:** 2610.29744 | [PDF](https://arxiv.org/pdf/2610.29744v1)

**作者:** Huakai Wei `[一作]` (Sichuan University), Shaofang Hong `[通讯]` (Sichuan University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在本文中，作者研究了满足(n-k-1)-设计的几乎最大距离可分离（AMDS）码，推导出其对偶码参数、支撑设计以及子码权重枚举，并利用此理论解决了Witt球面几何设计的一个开放问题。

**💡 创新点**

创新点在于将原先分散的个别算式归纳为统一框架，给出对偶码参数与设计参数之间的闭式关系，并提供子码权重枚举的通用公式。

**🔧 技术方法**

采用了设计理论、线性码的对偶性质、莫比乌斯反演与矩阵秩分析等组合数学与编码理论技术。

**📊 数据集**

实验验证主要基于二元Hamming码、BCH码以及Witt球面几何设计的参数实例，无需新数据集。

**📈 对比分析**

与已有结果比较，本文给出的权重枚举公式与已知实例完全一致，且在解决开放问题时提供了更直观的证明，理论上相较原方法具有提升。

**⚠️ 局限性**

局限性在于当前仅适用于满足(n-k-1)-设计支撑的AMDS码，未涵盖更一般的设计或非线性码的情况。

---

## 544. APED-CXR: Adaptive Privileged Evidence Distillation for Chest X-Ray Grounding with Image-Only Inference

**arXiv ID:** 2610.29740 | [PDF](https://arxiv.org/pdf/2610.29740v1)

**作者:** Md Shihabul Islam Shovo `[一作]`, Jamil Fayyad `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种名为APED-CXR的模型，利用冻结的多模态教师在训练阶段通过报告信息验证学生模型的区域建议，最终得到只依赖图像的诊断与定位模型。

**💡 创新点**

创新点在于：①将报告启发的空间证据通过知识蒸馏方式直接注入学生的证据图；②采用自适应的提议刷新机制，使教师评估随学生训练进展而动态变化的区域；③在不增加推理时文本输入的前提下，实现与大模型同等水平的定位与诊断。

**🔧 技术方法**

核心技术包括：多模态教师（MedGemma 1.5 4B）报告与图像联合问答、学生EfficientNet-B3+轻量级适配器产生证据图、Log‑sum‑exp池化进行诊断、区域提议与平均证据池化、软目标蒸馏损失以及对比实验设计。

**📊 数据集**

使用了PadChest（160k图像+报告）、其高质量标注子集PadChest‑GR（4,555图像含手工框）进行训练与评估，同时在NIH ChestX‑ray14（已标注单个定位框）上做交叉数据集验证。

**📈 对比分析**

与仅使用诊断标签（C0）、仅使用手工框（C1）以及固定提议的对照（C2）相比，APED-CXR在PadChest‑GR测试集的微观gF1从约0.48提升到约0.59（+0.11，95% CI [0.07,0.15]），诊断AUC保持≈0.94；在NIH ChestX‑ray14上，定位召回率从0.25提升至0.42。

**⚠️ 局限性**

主要局限包括：①教师推理成本高，训练阶段需频繁查询；②评估集来自同一医院，外部验证有限；③教师读取的报告为西班牙语且已预处理，导致监督噪声；④框定位粗略（仅IoU≥0.5），未能衡量精确分割或临床意义。

---

## 545. Accept or Repair: Preserving Passage-Supported Answers in Contrastive Decoding

**arXiv ID:** 2610.29727 | [PDF](https://arxiv.org/pdf/2610.29727v1)

**作者:** Jungseob Lee `[一作]` (Korea University), Jaehyung Seo `[通讯]` (Konkuk University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种训练无关的答案校正方法 Accept-or-Repair，先检查模型生成的答案是否被给定文本支持，若不支持则通过对比解码生成修正答案，并加入记忆否决机制。

**💡 创新点**

创新点在于将答案支持检测与对比解码相结合，并使用记忆否决防止模型重复闭书答案，从而在保持对普通文本准确率的同时显著提升对冲突文本的跟随性。

**🔧 技术方法**

使用对比解码（CAD、ID）生成修正答案，基于词匹配的支持检验，以及记忆否决过滤；评估采用 token‑level F1 与多选准确率。

**📊 数据集**

实验数据集包括 NQ‑Swap、FaithEval、ConFiQA（冲突文本）以及 NQ‑Original、SQuAD、HotpotQA、TriviaQA、SciQ（普通文本）。

**📈 对比分析**

与多种对比解码和提示式基线（CAD、AdaCAD、COIECD、ID、Strict Prompt 等）在六个模型上进行对比，结果表明 AoR 在冲突文本上提升若干个百分点，在普通文本上亦有小幅提升，整体平均得分显著高于基线，代价仅略高于简单采样。

**⚠️ 局限性**

局限性：支持检测仅基于词匹配，难以处理长文本或释义答案；实验仅覆盖英文数据和至 70B 参数的模型；对比解码需要访问模型的下一个词分布。

---

## 546. Bio-Inspired Auditory Processing: Coupling Parallel Resonate-and-Fire Dynamics with Ternary Recurrent Networks for Speech Recognition

**arXiv ID:** 2610.29720 | [PDF](https://arxiv.org/pdf/2610.29720v1)

**作者:** Pavlos Poulos `[一作]` (Athens University of Economics and Business), Costas A. Anastassiou `[通讯]` (Archimedes AI, Athena Research Center)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并训练了一种端到端的Spiking Ternary GRU（stGRU）模型，利用三元权重的延迟缩放投影和并行的Resonate‑and‑Fire（RF）神经元，实现了完全无矩阵MAC、二进制事件驱动的稀疏语音识别网络。

**💡 创新点**

创新点在于将激活离散化嵌入训练过程，构建了全矩阵MAC‑free的SNN；通过三元权重与RF二进制事件接口的组合，首次在不使用密集ANN序列解码器的情况下实现单一数字WER；并利用FFT实现PRF积分，显著加速了训练。

**🔧 技术方法**

采用延迟缩放的三元投影、MLGRU门控结构、并行Resonate‑and‑Fire神经元、SMAI/STE多高斯梯度、FFT频域膜电位计算、RMSNorm、SwiGLU+ReZero以及端到端的CTC解码等技术。

**📊 数据集**

使用LibriSpeech‑960（英文）和Aishell‑1（中文）数据集进行训练与评估。

**📈 对比分析**

与连续ANN（Transformer、Conformer等）及混合SNN/ANN解码器进行对比，stGRU小/中/大模型在无外部LM下分别取得8.09%、7.04%、6.44±0.17% WER，能耗仅3.42 mJ/s（Large），显著低于Transformer（8.80 mJ/s）；在Aishell‑1上得到10.88% CER。

**⚠️ 局限性**

能量估计基于理论操作计数，未测硬件功耗；仍存在逐元素乘法；三元量化限制需大通道和长时序核；在高精度硬件上部署仍需专用算术单元；与最先进的密集模型相比性能仍略逊，且在多说话人/低资源场景下表现可能下降。

---

## 547. Fully Dynamic Triangle and 4-Vertex Subgraph Counting: From Theory to Practice and Back Again

**arXiv ID:** 2610.29776 | [PDF](https://arxiv.org/pdf/2610.29776v1)

**作者:** Kathrin Hanauer `[一作]` (University of Vienna), Leonhard Sidl `[通讯]` (University of Vienna)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究并实现了两种基于重度-轻度顶点划分的动态三角形及四点子图计数算法，并通过多项理论与实践优化实现了高效更新。

**💡 创新点**

在原始算法基础上识别并纠正错误，提出了高度顶点限定、额外辅助结构、动态重平衡与分段重平衡等创新优化，并实现了算法与划分方案的跨组合，取得了理论和实际双重加速。

**🔧 技术方法**

核心技术包括重度-轻度顶点划分、辅助计数结构、邻域划分、动态重平衡（MR）与渐进重平衡、直接划分交换、以及结合不同划分策略的跨算法组合。

**📊 数据集**

使用了56个真实与合成动态图（包括维基百科链接网络、时间戳图、Kronecker图等），并在工程子集（2百万更新）上进行了细粒度调参。

**📈 对比分析**

与静态重算、单一动态计数器和连续子图匹配（CSM）基线对比，优化后算法在总时间、最大更新时间和内存方面均优于对手，速度提升可达数十倍至数百倍，尤其在所有模式同时计数时显著超越。

**⚠️ 局限性**

限制主要在内存消耗较高（尤其高度顶点限定版本）以及对极低计数场景下辅助结构反而带来额外负担；对部分划分策略的参数调优依赖于图特性，未必适用于所有动态图。

---

## 548. Evolving with Open Worlds: Towards Recursive Self-Improvement for Future Forecasting

**arXiv ID:** 2610.29784 | [PDF](https://arxiv.org/pdf/2610.29784v1)

**作者:** Zhipeng Chen `[一作]` (Renmin University of China), Wayne Xin Zhao `[通讯]` (Renmin University of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 DeepEvolve 框架，将实时事件预测作为一个闭环的递归自我改进环境，实现模型在真实世界任务中的持续进化。

**💡 创新点**

创新点在于三大模块协同构成闭环：Prometheus 生成动态可验证任务；Branchus 将证据检索与推理分离，形成可追踪的预测轨迹；Athena 通过强化学习、技能提炼和对抗式任务合成同时更新参数、技能和环境，真正实现递归自我提升。

**🔧 技术方法**

技术包括：多代理团队架构、可检索工具调用、Group Relative Policy Optimization（GRPO）强化学习、基于轨迹的技能蒸馏、错误分类驱动的任务合成、可验证奖励（RLVR）与Brier/WIS等校准指标。

**📊 数据集**

数据集：每日抓取 125 个新闻网站构成的实时事件流，用于生成 Prometheus 任务；评测采用外部 FutureX 周度基准与 Prometheus 自生成的任务集，所有结果均基于真实事件最终结果。

**📈 对比分析**

与 ReAct、MAS、OpenClaw、DeerFlow、MiroFlow 等多种基线在四大 LLM 上对比，DeepEvolve 在预测准确率、Brier 分数、WIS 及 Bradley–Terry 排名等指标均居首位，并且在多轮迭代中持续提升，展示了可持续的递归自我改进。

**⚠️ 局限性**

局限性包括：受限于可观测事件数量与多样性导致经验饱和、技能蒸馏对环境漂移的鲁棒性不足、对抗式任务合成只能基于可观测错误、计算成本随多轮迭代增长且对任务分布漂移的适应仍需改进。

---

## 549. An experimental evaluation of static k-center clustering algorithms on graphs

**arXiv ID:** 2610.29781 | [PDF](https://arxiv.org/pdf/2610.29781v1)

**作者:** Sebastian Forster `[一作]` (University of Salzburg), Martin Grösbacher `[通讯]` (University of Salzburg)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对图上静态 k‑center 聚类算法进行了系统实验评测，实施了四种主流算法（Gonzalez、Approximate Gonzalez、Abboud MIS 以及简化版 Thorup）并实现了其并行版本；

**💡 创新点**

创新点在于首次提供大规模图数据集上的全面实验对比，提出了简化版 Thorup 并行实现，统一使用相同的懒惰加速 SSSP（Dijkstra/Δ‑stepping）与并行调度，揭示了不同算法在时间、质量与并行伸缩性上的细微差异；

**🔧 技术方法**

使用的技术包括：懒惰距离更新的单源最短路（Lazy Dijkstra/Δ‑stepping）、并行 fork‑join 调度、并行 BFS/Δ‑stepping、随机化选择与增量更新的 MIS/Thorup 逻辑、α‑近似 Gonzalez 的包（bag）维护等；

**📊 数据集**

实验数据集涵盖：1）合成 Erdős–Rényi 随机图（10⁵/10⁷ 节点，稀疏到密集）；2）社会网络（Snap 的 DBLP、YouTube、LiveJournal、Orkut、Twitter、Friendster）；3）道路网络（DIMACS 9th 赛道的 USA‑central、USA‑full）；4）评分网络（libimseti、MovieLens、Yahoo! Song）；所有数据均做过去除孤立点、多重边与自环的预处理；

**📈 对比分析**

比较方法：对每种算法在不同 k（至 √n）下的单核与 64 核并行执行进行跑时记录，并用 Gonzalez 的解作为基准比较返回半径（相对误差）；实验结果显示：
• 在密集无权图上，Gonzalez/Approximate Gonzalez 的跑时最短；
• 在稀疏加权图（道路网络）上，MIS（Abboud MIS）和 Simplified Thorup 产生更小的半径，且运行时间与 k 增大后逐渐接近甚至优于 farthest‑point；
• 并行化在足够大且密集的实例中可获得 2–5 倍加速（最多 64 核），但在高直径稀疏图上反而导致慢速；
• 方案质量：MIS 基础算法在加权实例上平均比 Gonzalez 低 10–12% 的半径；在无权实例上差异不大；

**⚠️ 局限性**

局限性包括：
• 仅评测了四种静态 2‑近似算法，未涉及动态/增量或更高近似比的算法；
• 只考虑 k 取至 √n 的范围，超大 k 或 k 较小的情况未深入；
• Simplified Thorup 仍保留了多轮猜测和迭代，理论上可改进但实验中采用固定 4 轮导致部分性能偏差；
• 未对 Cohen 等大小估计方法的实现做比较，可能在理论上更高效但实际可用性有限；
• 并行实现依赖单机共享内存，未评测分布式或 GPU 加速的可行性。

---

## 550. JointLight: Human-Centric Video Relighting with Asynchronous RGB-Irradiance Joint Modeling

**arXiv ID:** 2610.29780 | [PDF](https://arxiv.org/pdf/2610.29780v1)

**作者:** Henglei Lv `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences and University of Chinese Academy of Sciences), Lin Gao `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences and University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `70e40602-aae3-44bd-80ec-4a7f2674330f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种异步RGB-irradiance联合建模的方法，实现了对人像视频的高质量、可控重光。

**💡 创新点**

创新点在于通过噪声腐蚀抑制粗糙irradiance cue的高频误差，采用异步时间步调度、域匹配和 hijacked inference，实现了精准光照控制。

**🔧 技术方法**

技术上结合了基于Transformer的Diffusion模型、VAE、点云深度估计、Mitsuba路径追踪、Rectified Flow ODE以及多模态异步去噪策略。

**📊 数据集**

使用了基于UE5渲染的68K高质量合成视频数据集，包含30个3D场景、821个HDR环境图和对应的 irradiance map，还在 HumanVid 上进行测试。

**📈 对比分析**

与 DiffusionRenderer、UniRelight、Light‑X 等基线对比，本文方法在 PSNR/SSIM/LPIPS 及用户研究中均优于对手，尤其在动态光照下表现突出。

**⚠️ 局限性**

局限性包括仅处理漫射传输，无法精确控制镜面光；噪声去除可能导致细节模糊；几何误差和运动会引发光泄漏或细节损失。

---

## 551. The Privacy Paradox Is a Misnomer: Data Under Structural Uncertainty

**arXiv ID:** 2610.29756 | [PDF](https://arxiv.org/pdf/2610.29756v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 552. Location-of-interest vs. Current-location Privacy in Multi-hop Communication

**arXiv ID:** 2610.29772 | [PDF](https://arxiv.org/pdf/2610.29772v1)

**作者:** Ryan Vogt `[一作]` (University of Alberta), Mario A. Nascimento `[通讯]` (University of Alberta)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种用户控制的 k‑匿名查询方案，在多跳无线网络中通过向不同的假目标发送多条查询来保护用户的当前位置（CL）和感兴趣位置（LOI）的隐私。

**💡 创新点**

创新点包括：①在不改动网络协议或节点软件的前提下，仅通过发送 k 条查询实现隐私保护；②建立了 CL‑隐私、LOI‑隐私和区域兴趣（AOI）隐私的量化指标；③提出了基于分区的假目标选择策略，并证明了其在满足闭包性质（closure requirement）下的可行性；④对分区与划分问题给出了贪心和分治近似算法；⑤系统性地分析了 k、被攻破节点数、广播加密方式和路由策略等参数对隐私的影响。

**🔧 技术方法**

技术手段主要包括：k‑查询隐私模型、路径一致性分析、隐私度量公式、分区优化（分区/划分问题）、贪心与分治近似算法、Monte Carlo 采样、随机/固定最短路径路由、全局加密与链路加密的比较。

**📊 数据集**

使用的数据集包括：①10×10 单元格网格（100 节点）作为特征无关基准；②埃德蒙顿市中心与希拉（Fira）两张实际城市地图（112 与 84 节点），结合 DBS3 模拟器生成的 1000 万人行人轨迹，以得到节点的访问概率 N 与动态兴趣概率 M。所有实验均基于这些网络拓扑和移动统计信息。

**📈 对比分析**

实验通过 1000–5000 次 Monte Carlo 试验比较不同参数组合的 CL‑隐私和 LOI‑隐私。结果表明：k 越大，LOI‑隐私提升但 CL‑隐私下降；被攻破节点越多，两种隐私均显著降低；全局加密环境下隐私最差，链路加密或非广播环境下显著提升；随机最短路径路由能显著提高 CL‑隐私；AOI 随机性对 LOI‑隐私影响有限。整体上，该方案在资源受限的战术无线网络中实现了相对可行的隐私保护。

**⚠️ 局限性**

限制包括：①额外 k‑1 条查询导致能耗和带宽增加；②假目标选择需满足闭包性质，实际实现可能受节点分布与热门位置差异影响；③分区与划分问题被证明为 NP‑完整/困难，近似算法虽可行但在大规模网络中仍有性能瓶颈；④对移动统计信息的依赖使得在没有足够数据时隐私难以评估；⑤实验仅考虑理想化路由与加密模型，真实战术环境中的干扰与信道变化可能进一步削弱隐私。

---

## 553. OMAR: Open-World Multi-Attribute Reasoning for Scene-Conditioned Shelf Organization

**arXiv ID:** 2610.29761 | [PDF](https://arxiv.org/pdf/2610.29761v1)

**作者:** Benedikt Kreis `[一作]` (University of Bonn), Maren Bennewitz `[通讯]` (University of Bonn)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建多视角语义实例映射与LLM推理管线，生成实例级3D语义场景图，依据多属性信息对未知物体进行语义一致性放置，并通过几何与运动学可行性检查实现机器人自主抓取与放置；

**💡 创新点**

提出开放世界多属性推理（OMAR）框架：利用VLM预测内容、包装、可食性、方向等多属性并在多视角实例化后构建3D场景图，使LLM能够推断货架潜在组织并给出语义排名，从而在没有预定义物体-托盘映射或用户偏好的情况下实现一致性放置；

**🔧 技术方法**

使用Vision‑Language模型（SAM3、SAM3D、VLM）获取多视角语义标签和属性，结合多视角TSDF重建与实例级关联，构建3D语义场景图；利用大型语言模型（GPT‑5.6、Mistral、Deepseek）进行多属性推理；用FoundationPose进行姿态估计，采样抗力手势与几何可行性评估，最终通过运动规划执行；

**📊 数据集**

在真实机器人与Isaac Sim仿真中使用包含60种包装商品的自建数据集，分为8类、5种包装类型，生成三种场景配置（单行、多行、变间隙），共300个测试场景；

**📈 对比分析**

与单视角VLM+LLM基线（BL1）、几何规划基线（BL2）以及无属性基线（Basic）对比；实验显示我们的方案在实例映射指标（F1‑Label、BB‑IoU、实例覆盖率）和放置语义一致性指标（SF‑EQ、SF‑LLM）均优于BL1，在放置成功率上与BL2相当，表明语义一致性提升不损失执行成功；

**⚠️ 局限性**

受限于VLM预测的标签/属性准确性，遮挡/视角不足导致语义误判；计算成本高，尤其是多视角重建和LLM推理；在极端稀疏、动态或非标准货架环境中的鲁棒性尚待提升；

---

## 554. AssureCode: Structured Justification for Trustworthy Agentic Coding

**arXiv ID:** 2610.29758 | [PDF](https://arxiv.org/pdf/2610.29758v1)

**作者:** Logan Murphy `[一作]` (University of Toronto), Marsha Chechik `[通讯]` (University of Toronto)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套面向 AI 辅助编码的结构化证明理由（justification rationale）框架，并实现工具 AssureCode，能够在代码生成过程中自动生成正式化、论证与验证证据，形成完整的安全保障链。

**💡 创新点**

创新点在于将安全关键系统的结构化保证案例（如 GSN）与部分自动化形式化、LLM 驱动的论证模板相结合，提供正式与自然语言并存、可自动检查的辩护结构，并将其直接嵌入到生成流程中，减少人工检查成本。

**🔧 技术方法**

使用了 Lean 形式化与定理证明器、LLM 自动形式化与分解模板、自动测试生成与分析工具（如静态分析、单元测试）、以及自定义的论证模板库。

**📊 数据集**

评估数据集为 HumanEval、LiveCodeBench、MBPP 三个 Python 代码生成基准，总计 30 个示例（10 条每个基准）。

**📈 对比分析**

对比方法：与“vibe justification”基线（不使用结构化理由）进行对比，衡量产生的验证义务数量、互相覆盖率、独立关切数量及其质量。结果表明 AssureCode 产生的义务数量与基线相当，覆盖率更高，独立关切更多，整体价值相近。

**⚠️ 局限性**

限制包括：仅在函数级、单文件任务上验证，未评估大规模项目；未测量人类审查成本；评估依赖 LLM 判断，可能存在偏差；需要高性能 LLM；对更复杂的非功能/安全需求支持尚不足。

---

## 555. VeruSteer: Learning What to Repair Next in Verus

**arXiv ID:** 2610.29755 | [PDF](https://arxiv.org/pdf/2610.29755v1)

**作者:** Chao Jiang `[一作]` (Shenzhen University), Lin Huang `[通讯]` (Ant Group)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究针对LLM驱动的Verus程序修复中“下一步要修复哪类错误”这一决策问题，提出VeruSteer框架，学习从同一初始程序产生的多条修复轨迹中判断哪种错误类型的修复顺序最优，并将该策略嵌入AutoVerus/VeruSAGE管道；

**💡 创新点**

创新点在于①通过构造修复树获取完整的替代轨迹，对比其验证成功、修复轮次与最终进度形成偏好对，用以监督；②将错误类型选择视为序列推荐任务，利用自注意力编码器和BPR排名损失联合训练，实现对程序、诊断和历史修复的上下文感知选择；③证明学习错误类型优先级可显著提升验证成功率，优于固定优先级或随机选择。

**🔧 技术方法**

使用预训练文本嵌入模型BGE‑M3对程序、诊断和错误类型进行编码；单头自注意力编码器（SASRec）对决策上下文进行序列建模；交叉熵损失实现对已知最佳轨迹的逐步复制，BPR损失实现轨迹偏好排名；并在AutoVerus/VeruSAGE修复器与Verus验证器之上实现。

**📊 数据集**

基于StarVerus基准的762个验证任务，使用DeepSeek‑Chat与Qwen‑Coder生成程序，构造3,224棵修复树（1,613 AutoVerus、1,611 VeruSAGE），按342/69/273任务划分为训练/验证/测试集。

**📈 对比分析**

在测试树上进行“重放”实验，对比VeruSteer与默认优先级在同一树上得到的最终状态；使用Retention（保留）与Exploration（探索）两项指标评估；结果显示，VeruSteer在两管道共计1,270棵树上提升了9棵验证成功树（+0.71%），在有备选轨迹的树中保留率约90%、探索率约80%；同时在成功树上得到更短的修复路径，在所有树上获得更高的最终进度。

**⚠️ 局限性**

仅学习错误类型选择，未改进修复动作或候选选择；修复树与偏好学习基于固定5轮预算，无法保证更长修复周期的表现；构造修复树成本高且需随模型/Verus版本更新重建；评估仅在已记录轨迹上进行，缺乏实时LLM调用下的验证；适用性受限于错误类型映射与管道兼容性。

---

## 556. CausGround: Causal Interaction Grounding for Adaptive Embodied Planning

**arXiv ID:** 2610.29702 | [PDF](https://arxiv.org/pdf/2610.29702v1)

**作者:** Yijun Shen `[一作]` (East China Normal University), Kai Zhang `[通讯]` (East China Normal University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种非参数的因果交互基准框架CausGround，能将多模态大语言模型（MLLM）在不同环境下的交互经验提炼为可复用的动作前置条件和效果，并在规划时动态构建因果图进行逆向和正向推理；

**💡 创新点**

核心创新在于（1）引入统一的因子化世界状态表征，既捕捉对象属性，又包含对象间关系；（2）把交互经验压缩为动作中心的因果知识库；（3）利用该知识库在规划时实现无参数更新、可解释的长时序计划；

**🔧 技术方法**

使用因果推理（逆向预置条件检验与正向效果预测）、因子化世界状态抽象、基于检索的因果知识库构建与任务级因果图生成；

**📊 数据集**

在虚拟家居模拟环境VirtualHome、RoboCasa以及真实世界交互基准SWITCH上进行评估，并在实际机器人上验证；

**📈 对比分析**

与传统无参数规划器、SelfReVision、自检机制以及基于场景图的LookPlanGraph进行对比；在VirtualHome上平均成功率从24.9%提升至53.3%，在RoboCasa闭环规划成功率从43.0%提升至57.0%，在SWITCH交互推理准确率从25.8%提升至31.9%，整体表现显著优于基线；

**⚠️ 局限性**

依赖于可观测的RGB图像与精确的世界状态提取，若状态抽象误差大或交互经验不足，因果知识库可能不完整；此外，当前方法主要针对离散动作空间，尚未扩展到连续控制或高频交互场景。

---

## 557. Prompt Injection Through Forensic Evidence: A Taxonomy and Proof of Concept of Anti-Forensic Attacks on LLM-Assisted Digital Investigations

**arXiv ID:** 2610.29747 | [PDF](https://arxiv.org/pdf/2610.29747v1)

**作者:** Frank Breitinger `[一作]` (University of Augsburg), Mark Scanlon `[通讯]` (University College Dublin)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在 LLM 辅助的数字取证工作流中，定义了一个新的威胁模型和分类法，并演示了三种基于植入文本的攻击（误导结论、写入文件、发起网络请求）。

**💡 创新点**

创新点在于将传统反取证手段与生成式 AI 的提示注入相结合，提出了 T5（注入）这一新的攻击类别，并揭示了证据与 LLM 之间的感知差距。

**🔧 技术方法**

技术实现采用 Gemma‑4‑12b 作为自主工具调用的 LLM，结合 Autopsy 的 MCP 桥接、文件系统与网络 fetch 服务器，并在磁盘镜像中植入明文攻击载荷。

**📊 数据集**

实验数据集为合成的磁盘镜像，内含植入说明的文本文件，用于模拟被扣押设备的证据。

**📈 对比分析**

通过在相同工作流下执行三次攻击，记录成功率（Gemma‑4‑12b 下约 60–80%），并与其他模型对比发现大模型对注释的识别更低；实验范围有限，未构建系统化基准。

**⚠️ 局限性**

局限性包括仅在本地量化模型上验证，未覆盖商业部署；攻击成功依赖于特定权限配置与模型提示；缺乏多模型、多工具场景下的鲁棒性评估。

---

## 558. Some Results on the Weight Spectrum of Binary Reed-Muller Codes $RM(m-7,m)$

**arXiv ID:** 2610.29807 | [PDF](https://arxiv.org/pdf/2610.29807v1)

**作者:** Zibo Wei `[一作]` (Beihang University), Hao Chen `[通讯]` (Jinan University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了高阶二进制Reed-Muller码(m-7,m)的权重谱，证明了对于每个m≥14，权重谱包含区间[336,2^m-336]内的所有偶数权重，以及一组显式的低权重及其补集。

**💡 创新点**

创新点在于建立了权重谱的构造性包含，涵盖了所有偶数权重，并通过四块拼接和显式的六次单项式构造证明了基码(7,14)的包含关系。

**🔧 技术方法**

使用了四块拼接、显式的六次单项式见证和已知的谱输入，结合归纳法将基包含提升到所有m≥14。

**📊 数据集**

使用了Reed-Muller码的基码(7,14)以及已知的谱输入(5,10)和(6,12)。

**📈 对比分析**

通过与已知的低权重谱和其他构造方法进行比较，发现构造性包含覆盖了所有偶数权重，但对于低权重目标322, 326, 330, 334的存在性仍未解决。

**⚠️ 局限性**

限制在于未能确定权重集合T={322, 326, 330, 334}是否存在于完整的码(7,14)中，且对这些权重的构造性分析未能找到有效的构造方法。

---

## 559. MAPS: Macro-Adaptive Parallel Reasoning for Regulating Search Momentum in Retrieval-Augmented Generation

**arXiv ID:** 2610.29792 | [PDF](https://arxiv.org/pdf/2610.29792v1)

**作者:** Xinze Li `[一作]` (Northeastern University), Maosong Sun `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并提出一种宏观自适应并行推理框架 MAPS，以调节检索过程中的搜索动量并减少查询意图漂移，改进 RAG 方案。

**💡 创新点**

创新点包括：1）引入搜索动量概念衡量历史子查询的累积检索偏向；2）设计去耦历史推理轨迹与子查询的离散化；3）引入并行推理探索互补检索方向，从而稳定搜索动量，缓解查询意图漂移。

**🔧 技术方法**

使用技术包括：大语言模型 Qwen3-32B/35B，密集检索（E5-base-v2+FAISS），自回归推理、并行子查询生成、检索方向提议模型、Refine 组件，搜索动量计算及 β 权重学习。

**📊 数据集**

使用的数据集有：NQ、PopQA、2WikiQA、HotpotQA、MuSiQue、GPQA Diamond（Diamond 子集），每个数据集抽取 1000 条样本。

**📈 对比分析**

通过与 Vanilla LLM、Vanilla RAG、Self-RAG、IRCoT、Search-o1、S3、ParallelSearch 等基线对比，MAPS 在 Qwen3-32B 和 Qwen3.6-35B 上分别提升 6.8% 和 5.8% 的 Cover-EM，较 Search-o1 提升 9%，在所有数据集上均表现出稳健且一致的性能提升。

**⚠️ 局限性**

局限性：1）对检索方向提议模型和超参数调优敏感；2）仍依赖密集检索索引，对大规模或多语言场景的鲁棒性未充分验证；3）对极其复杂的多跳查询仍可能需要多轮推理，导致推理深度增加。

---

## 560. Key Findings for Small UAS for Post-Disaster Damage Assessment from the CURSE'26 Exercise

**arXiv ID:** 2610.29777 | [PDF](https://arxiv.org/pdf/2610.29777v1)

**作者:** Thomas Manzini `[一作]` (Texas A&M University), Robin Murphy `[通讯]` (Texas A&M University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该研究评估了 CLARKE 自动建筑与道路损坏检测系统在 CURSE'26 大规模 sUAS 训练演习中的性能与操作管线。

**💡 创新点**

在多州地理与不同土地利用条件下对 CLARKE 进行零射线传输评估，并系统性研究了演习对机器学习模型的实际评估价值。

**🔧 技术方法**

结合 Vision Transformer-UperNet 编码解码网络与 Attention U-Net，采用边缘计算推理，并通过 PIX4Dreact 生成正射影像。

**📊 数据集**

使用 CRASAR-U-DROIDs 训练集（约 67 Gpx）以及 13 张演习生成的正射影像。

**📈 对比分析**

以 Micro F1 评估建筑损坏，取得 0.834 的 F1，平均推理时间 3.5 分钟；道路损坏表现差，误判率高。

**⚠️ 局限性**

演习图像缺乏真实灾害损伤，主要在商业/城市区域，导致 RDA 失效；需人工监督与边缘计算支持。

---

## 561. UniRAP: Towards Unified Part-level Physical Affordance Reasoning and Actionable Perception

**arXiv ID:** 2610.29768 | [PDF](https://arxiv.org/pdf/2610.29768v1)

**作者:** Linfei Li `[一作]` (Tongji University), Xin Zhang `[通讯]` (Neusoft Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 UniRAP，一种统一的视觉‑语言模型，能够将语言指令直接映射到对象检测框、部件级功能分割掩码以及 4‑DoF 抓取姿态，实现可执行的物理交互感知。

**💡 创新点**

核心创新包括：①统一接口 token 机制将视觉提示与文本提示整合到共享空间；②统一 Affordance Decoder (UAD) 通过共享中间特征实现分割、检测与抓取三任务的端到端联合推理；③基于课程的迁移训练，从大规模分割数据逐步过渡到交互感知，显著提升低样本情况下的泛化；④使用记忆注意力与多帧融合，保持视频时序一致性。

**🔧 技术方法**

技术栈包括 Qwen2.5‑VL 作为视觉‑语言基座、SAM2 视觉编码器、随机傅里叶映射编码视觉提示、FiLM 条件化、两路 Transformer 交叉注意力、记忆注意力、统一 Affordance Decoder（Mask Head、BBox Head、Grasp Head）以及多任务联合损失（Mask、IoU、BBox、Grasp）。

**📊 数据集**

使用的数据集：RefCOCO/+/g、ReasonSeg、ReVOS、HANDAL、GraspNet、3DOI、RealVLG‑Benchmark、RealVLG‑R1，以及真实机器人抓取实验中 13 个家庭物体。

**📈 对比分析**

与多种非 LLM 与 LLM 基础模型（ReLA、X‑Decoder、SEEM、NExT‑Chat、PixelLM、LISA、RAGNet、UniPixel、HyperSeg、ViLLa、VISA 等）以及视频通用模型（ReVOS、HyperSeg、InstructSeg、UniPixel 等）对比，UniRAP 在 RefCOCO、ReasonSeg、ReVOS、HANDAL、GraspNet、3DOI、RealVLG‑Benchmark 等任务上均取得或接近 state‑of‑the‑art 的性能；在零样本 3DOI 任务中 gIoU、cIoU 领先 17.6/15.1 点；在真实机器人抓取实验中成功率高达 88.5%，显著优于 RealVLG‑R1（62.3%）和 UniRAP+GraspNet（55.4%）。

**⚠️ 局限性**

局限性：①依赖大规模预训练模型，训练与推理成本高；②在极端视觉条件（反光、深度噪声）下仍可能产生误检或姿态误差；③高维度交互空间（如更复杂的多自由度抓取）仍需进一步研究；④课程迁移训练流程较为复杂，易受超参数与阶段划分影响；⑤在长文本推理或多步骤推理任务上，表现仍不及专门设计的多步骤推理模型。

---

## 562. ALIAS: Evaluating Performance and Memorization in LLM-Written TLA+ Specifications

**arXiv ID:** 2610.29767 | [PDF](https://arxiv.org/pdf/2610.29767v1)

**作者:** Arslan Bisharat `[一作]` (Loyola University Chicago), Mohammed Abuhamad `[通讯]` (Loyola University Chicago)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过一致性标识符重命名（CIR）检验大语言模型在 Temporal Logic of Actions Plus 任务上的记忆效应，并提出行为基础的评分方法。

**💡 创新点**

创新点在于：①将模型检查器 TLC 作为完备的重命名验证器；②设计了行为基础的分数评估，剔除对名称的依赖；③通过对两组模型的训练曝光控制，区分记忆、评分方式与新名称成本。

**🔧 技术方法**

主要技术包括：一致性标识符重命名、SANY 语法检查、TLC 模型检查、基于行为的细粒度评分、LoRA 微调、对数似然评估、Bootstrap 置信区间和符号检验。

**📊 数据集**

使用了 111 个从 -Bench 公开的 TLC 检查过的模型规格，并构造了对应的重命名版本；对 Claude Opus 4.8、GPT‑5、Qwen 2.5‑Coder‑14B 与 Gemma 4‑31B 等 LLM 进行实验。

**📈 对比分析**

实验比较了原始与重命名任务在名字评分与行为评分下的准确率变化：Opus 4.8 名字评分下降约10点，行为评分仅下降约3点；GPT‑5 名字评分下降2点，行为评分几乎无变化；在受控曝光实验中，Qwen 训练提升约7.6点，重命名降幅约4.9点，Gemma 训练提升约8点，重命名降幅约1.3点。

**⚠️ 局限性**

局限性包括：①仅评估 -Bench 单一领域任务；②行为评分覆盖率不足（90/111）且对变量映射有限；③曝光实验仅使用两类模型且仅一次训练随机划分；④SANY 解析错误可能被误判为通过；⑤未考虑模型对算法名称的先验记忆可能带来的偏差。

---

## 563. Engineering Requirements for Generative AI Agents: The BARD Process Theory

**arXiv ID:** 2610.29741 | [PDF](https://arxiv.org/pdf/2610.29741v1)

**作者:** Alfonso Cannavale `[一作]` (University of Salerno), Stefano Lambiase `[通讯]` (Aalborg University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发并验证了针对生成式 AI 代理的需求工程过程理论（BARD）

**💡 创新点**

首次提出以过程为核心、基于社会技术扎根理论的行为代理需求发现模型，阐释需求从固定规范演变为行为与评估准则的循环，并识别驱动转变的四种张力与两种条件

**🔧 技术方法**

采用社会技术扎根理论（STGT）进行半结构化访谈与迭代编码，构建 BARD 过程理论

**📊 数据集**

使用 36 名从事 GenAI 代理开发的实践者访谈数据；无公开数据集

**📈 对比分析**

通过访谈回访与成员检查验证理论可理解性；未进行客观性能评估，理论有效性主要基于实践者反馈

**⚠️ 局限性**

样本规模与多样性有限、仅依赖访谈缺乏实证观察、技术快速迭代导致表面细节迅速过时、在高度监管或大型组织环境中适用性可能受限

---

## 564. Learning to Measure and Diagnose: A Clinical-Dictionary Radiomic Bottleneck Model for Thyroid Nodule Classification

**arXiv ID:** 2610.29738 | [PDF](https://arxiv.org/pdf/2610.29738v1)

**作者:** Guibin Zhang `[一作]` (Southern Medical University), Lei Cao `[通讯]` (Southern Medical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种基于临床字典的放射组学瓶颈模型，利用超声图像直接预测并使用放射组学测量进行甲状腺结节良恶性分类，并可追踪各测量对决策的贡献。

**💡 创新点**

创新点包括：1）Family-Aware Radiomic Learning（FARL）通过在特征族间平均损失平衡监督，提升各族测量的学习质量；2）Fidelity-Guided Bottleneck Modeling（FGBM）构造固定的临床字典瓶颈，并在保持测量可信度的引导下联合优化测量估计与分类；3）决策证据分解方法，将线性分类器的权重按临床字典类别拆分，形成可解释的量化证据。

**🔧 技术方法**

技术包括：基于ConvNeXt-Tiny的卷积编码器、rank‑Gaussian特征转换、SmoothL1损失、Spearman相关性用于可靠性/冗余筛选、线性分类器与对数几率回归、Fidelity Loss约束、以及对数损失和加权平衡。

**📊 数据集**

使用公开的甲状腺超声数据集 ThyroidXL（11,554 图像/患者）和 TN3K（3,493 图像/患者），从中提取 102 个放射组学特征（几何、强度、纹理）。

**📈 对比分析**

与基于放射组学的、黑盒的及可解释的分类方法比较，Ours 在 ThyroidXL 上取得 94.77% AUC、86.73% BACC，在 TN3K 上取得 79.49% AUC、72.23% BACC，几乎与直接图像分类相当，并在所有可解释方法中取得最高平均 AUC 与 BACC，优于 RCAF、RLAR 等黑盒模型以及 LF‑CBM、PCBM、AdaCBM 等可解释模型。

**⚠️ 局限性**

局限性包括：1）训练阶段仍需结节掩模，无法完全实现全无标注；2）放射组学测量的可靠性受限于数据集质量，可能在新域中下降；3）模型解释依赖于预定义的临床字典，若临床描述不完整或有差异，解释可能受限。

---

## 565. Post-hoc Epistemic Uncertainty for Vision-Language-Action Policies

**arXiv ID:** 2610.29730 | [PDF](https://arxiv.org/pdf/2610.29730v1)

**作者:** Johannes Keller `[一作]` (Leipzig University), Daniel Schneider `[通讯]` (Leipzig University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出后置高斯过程核方差头，用单一视觉–语言–动作（VLA）策略的嵌入来无监督估计模型的表观不确定性，并用其进行主动学习和任务递归；

**💡 创新点**

创新点在于无需额外训练或梯度更新，直接利用已冻结的VLA嵌入做GP后验方差估计，既可取代两模型不一致的VFD，又大幅降低计算成本；

**🔧 技术方法**

技术包括：后置高斯过程核方差估计、标准化与主成分投影的预处理、稀疏Cholesky分解、SmolVLA视觉语言骨干、SAVE主动学习框架；

**📊 数据集**

使用的主要数据集为LIBERO-10仿真机器人基准（包含10个目标任务及其演示），并在此基础上做了输入扰动和控制实验；

**📈 对比分析**

与VFD、GU、ACE等基线对比：在任务级校准、演示采集、监督者递归上与两模型不一致得分相当或更优，且在GPU分钟上相较于VFD提升约50倍，内置GP头在对比实验中表现与VFD相当，优于GU；

**⚠️ 局限性**

局限性包括：仅在SmolVLA和LIBERO-10上验证；缺乏真实机器人或更大规模任务的实验；对嵌入空间保持距离的假设未强制训练；与任务内排名相关的评估受执行噪声限制；未检验实时报警与人类介入效果；

---

## 566. Structure-Aware Message Passing on Petri Nets through Directed Walks and Incidence Diffusion

**arXiv ID:** 2610.29725 | [PDF](https://arxiv.org/pdf/2610.29725v1)

**作者:** Geri Skenderi `[一作]` `[通讯]`, Geri Skenderi

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文针对Petri网的结构偏置，提出了两种新的GNN构造：基于交替随机游走的Petri网随机游走位置编码（PNRWPE）以及利用入度-出度矩阵的扩散增强消息传递网络（DA-PNGNN）。

**💡 创新点**

创新点在于将Petri网理论中隐含的两类结构信息（有向环路位置和标记平衡不变性）显式地提取为可学习的特征，并通过随机游走和扩散核共同提升图表示学习。

**🔧 技术方法**

所用技术包括图神经网络的消息传递框架、随机游走位置编码、基于入度-出度矩阵的扩散算子、PageRank式跳转的改进版本，以及在无监督多模态目标中的自监督训练。

**📊 数据集**

实验数据集包括 96 个 Process Discovery Contest 2023 的 Petri 网和 ProcRosetta 生成的 4000 训练/512 验证/512 测试工作流网（含循环与非循环两种设置）。

**📈 对比分析**

在有监督的生成因子预测任务中，宏观平衡准确率从 0.751 提升到 0.832；在 ProcRosetta 的多模态自监督评估中，net→tree Top‑1 提升 5.9%，以及解码与对齐指标均出现正向改进。

**⚠️ 局限性**

局限性包括仅在单位弧权重的合成网络上验证、扩散核需存储 O(n²) 大小且计算开销随网大小呈立方增长，以及 PageRank‑式跳转在无环网中虽消除退化但对下游性能无显著帮助。

---

## 567. Retrieval Augmented Generation Compositionality

**arXiv ID:** 2610.29722 | [PDF](https://arxiv.org/pdf/2610.29722v1)

**作者:** Ido Levy `[一作]` (IBM), Leshem Choshen `[通讯]` (IBM)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构建开放源代码框架Lasagna，对检索增强生成（RAG）流水线的不同组件进行系统组合实验，评估在多种基准（8个）和不同文档规模（5K–200K）下的整体性能，验证组件在完整流水线中的交互效应。

**💡 创新点**

创新点在于①提出基于D‑optimal设计的组合实验方法，覆盖约450个配置；②系统量化单个技术在完整流水线中的增益是否可加；③揭示检索质量与答案可信度对不同组件的敏感度差异；④发现随语料规模变化，最佳配置排名会显著漂移。

**🔧 技术方法**

使用了完整的RAG流水线阶段（分块、嵌入、检索、重排序、上下文压缩、生成与验证等），多种向量索引后端（Milvus、Pinecone、Qdrant等），以及多种检索与重排序模型（BM25、SPLADE、ColBERT等）；评估指标包括NDCG@10、RAGAS可信度、RAGAS和LLM Judge评分、超体积（HV）等。

**📊 数据集**

实验数据集为BEIR五个检索集（NFCorpus、FiQA、SciFact、ArguAna、HotpotQA）以及三大问答集（SQuAD v2、MuSiQue、HotpotQA），共计约150+查询；在5K–200K文档规模上进行多次采样扩展。

**📈 对比分析**

比较方法为全流程对比实验，使用配对统计检验（如符号检验）评估不同配置在检索和可信度两条轨道上的差异；通过超体积衡量检索、可信度与延迟的综合表现。实验结果显示：检索性能的单个技术增益在组合时往往出现 63% 级别的损失；而可信度增益近乎线性相加；最佳配置在不同规模下会出现倒换，说明规模对排名影响显著。

**⚠️ 局限性**

局限性包括：仅在英文多域合成语料上评估，缺乏真实领域数据验证；实验成本约$5k且受限于GPU/LLM API；框架与指标依赖当前公开基准，未覆盖所有可能的检索/生成任务；并且某些实验中使用的技术在生产环境下可扩展性仍需进一步验证。

---

## 568. Open-Vocabulary Models and Label Propagation for Uncertainty-aware Pseudo-Labeling of Object Detection Datasets

**arXiv ID:** 2610.29788 | [PDF](https://arxiv.org/pdf/2610.29788v1)

**作者:** Jonathan Klees `[一作]` (Osnabrück University), Matthias Rottmann `[通讯]` (Osnabrück University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于图的半监督框架，用开放词汇检测模型生成候选框并通过概率标签传播赋予软标签，实现对象检测数据集的半自动化不确定性标注。

**💡 创新点**

创新点在于将概率标签传播（Probabilistic Label Spreading）与开放词汇检测模型相结合，既捕获了分类不确定性，又能量化推断不确定性，且无需额外训练即可生成软标签。

**🔧 技术方法**

使用的技术包括Grounding DINO等开放词汇检测器、后置自校准（Isotonic Regression）、k‑NN图构造、高斯核相似度、概率标签传播和主动查询策略。

**📊 数据集**

实验数据集涵盖四个软标签标注数据集：COCO、Pascal VOC、KITTI 和 Cityscapes。

**📈 对比分析**

与基线（如直接使用检测置信度、Gaussian Kernel 回归、k‑NN 回归、硬标签注入）以及现有半监督检测方法相比，该方法在小预算下实现更高的平均相似度（AS）和平均精度（AP），在COCO上仅用 1% 标注即可达到 40.5% AP，几乎与全量标注相当。

**⚠️ 局限性**

局限性包括依赖开放词汇检测器的检测质量、对标签误差的偏差容易受到原始注释完整度的影响，以及在复杂街景数据集（如KITTI）上性能仍落后于完整人工标注。

---

## 569. Pfaffian Recognition is in coRP

**arXiv ID:** 2610.29817 | [PDF](https://arxiv.org/pdf/2610.29817v1)

**作者:** Priyanshu Pant `[一作]` (Indian Institute of Technology Indore), Ranveer Singh `[通讯]` (Indian Institute of Technology Indore)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一个基于多项式身份检测的随机多项式时间算法，用以判定给定图是否存在Pfaffian定向，从而把一般图的Pfaffian图识别问题放入C·RPP类；

**💡 创新点**

核心创新在于构造了一个“错误符号多项式”P_D(x)，并证明其在所有取值下恒为零当且仅当给定定向是Pfaffian，并利用Tutte矩阵的主子式与Jacobi恒等式实现多项式评估；

**🔧 技术方法**

主要技术包括：Pfaffian定向与匹配符号的代数表达、Schwartz–Zippel 线性多项式身份测试、Tutte矩阵的行列式与逆矩阵的计算以及对主子式的快速求值；

**📊 数据集**

本研究为理论算法，未使用任何实验数据集；

**📈 对比分析**

由于没有实验比较，无法给出性能指标，但算法在随机化实现下每次试验仅需O(|V(G)|^3) 场域运算；

**⚠️ 局限性**

局限性在于算法仍为随机化，未提供确定性多项式时间解；此外，实际实现仍需处理大规模图时的数值稳定性与场域选择问题。

---

## 570. Metamodel-Guided LLM Decoding for Consistent DSL Generation

**arXiv ID:** 2610.29749 | [PDF](https://arxiv.org/pdf/2610.29749v1)

**作者:** Boqi Chen `[一作]` (University of Ottawa), José Antonio Hernández López `[通讯]` (University of Murcia)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Metamodel-Guided LLM Decoding框架，通过在解码时将生成文本映射为部分模型并进行语义完整性验证，从而保证DSL实例在语法、抽象语法与语义约束三层都满足一致性。

**💡 创新点**

创新点在于将约束检查从文本层迁移到元模型层，利用部分模型可扩展性进行可行性判断，支持对复杂的多重性、引用完整性等约束的实时约束，且在保证语法有效的同时提升一致性。

**🔧 技术方法**

技术包括LLM解码、基于LLLGuidance的语法约束解码、增量抽象语法单元提取、语法到模型映射、使用Partially-Model Verifier（如metamodel-graph-solver）进行可扩展性验证，以及简单的重试机制。

**📊 数据集**

使用四个DSL基准：非可执行的Taxonomy、Flowchart；可执行的CLEVR、Block。

**📈 对比分析**

与无约束解码和传统语法约束解码对比。结果显示：语法有效率保持≈100%；一致性提升平均23%（对比无约束）和8%（对比语法约束），在不同模型规模（0.6B–70B）下均表现优异；对齐分数（F1*、执行准确率）亦不下降，部分模型上提升。重试导致的运行时开销很小，主要由额外的LLM推理承担。

**⚠️ 局限性**

局限包括：只处理不可修复约束；仅支持开放权重LLM（无法直接访问token logits）；重试策略简单，未利用更高效的回溯或重采样；对可修复约束的利用有限；对更复杂、规模更大的DSL的适用性仍需验证。

---

## 571. Two Sides of the Same Proof: Extraction and Projection in Choreographic Programming

**arXiv ID:** 2610.29736 | [PDF](https://arxiv.org/pdf/2610.29736v1)

**作者:** Matteo Acclavio `[一作]` (University of Southern Denmark), Marco Peressotti `[通讯]` (University of Southern Denmark)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了基于线性逻辑证明论的交互式程序设计框架，利用线性逻辑的证明与网络执行树之间的对应关系，实现了循环证明下的端点投影（EPP）与提取（Extraction）的统一证明与实现，并证明了该框架对尾递归无锁网络的完整性。

**💡 创新点**

创新点在于：①将交互式编程视为线性逻辑证明，将网络行为与证明结构映射；②扩展至循环证明以处理无限行为；③定义“公平证明”与“进展点”对应于网络的公正性与公平性；④通过新的证明语义实现更宽泛的可投影交互式程序，消除了传统 merge 操作的需求；⑤给出可多态的提取算法与 EPP 的完整性证明。

**🔧 技术方法**

使用了：1) 多重加法线性逻辑（MALL）及其非交换连结与一阶量化；2) 循环证明（cyclic proofs）与最小/最大固定点；3) 证明搜索与进展条件；4) 端点投影与提取的逻辑翻译；5) 证明等价规则与规则置换；6) 线性逻辑的语义化（命题与过程的编码）。

**📊 数据集**

无实验数据集；所有结果均为理论证明和形式化推导；若做实现，示例网络/交互式程序作为演示。

**📈 对比分析**

方法与传统 EPP 的比较：理论上支持更大类可投影交互式程序（包含递归选择等），并通过证明搜索实现提取；性能方面，针对尾递归网络，证明搜索可在多项式空间内完成；对更一般网络（含最小固定点）则复杂度更高，未给出具体上界。

**⚠️ 局限性**

局限性：仅对尾递归、无竞争（race‑free）的网络可保证提取与投影的终止与完整；需要线性逻辑的固定点子集（无最小固定点、无强绑定）；实现细节与优化尚未完全成熟；对包含一般循环/递归结构、并发冲突或非锁定网络的支持有限。

---

## 572. SAKURA: Serving Attention Key-Value State Under Rate Adaptation for Private Image Editing on Phones

**arXiv ID:** 2610.29733 | [PDF](https://arxiv.org/pdf/2610.29733v1)

**作者:** Shuncheng Jie `[一作]` (University of Waterloo), Sihang Liu `[通讯]` (University of Waterloo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种云边协同架构，将Diffusion Transformer的可复用KV缓存放在云端，按需在移动设备上流式传输，从而解决移动端存储不足和隐私泄露问题。

**💡 创新点**

创新点包括：① 通过基于请求无关的token重要性加权低秩压缩生成可多级压缩的缓存；② 采用分片细粒度、带宽自适应的缓存流式传输；③ 在移动端实现CPU‑GPU‑NPU流水线，使缓存接收、重建与推理并行。

**🔧 技术方法**

采用低秩SVD+重要性加权、按位量化与秩拆分的多级压缩；按Transformer块分片缓存；实时网络速率感知的分片传输调度；CPU‑GPU‑NPU流水线执行；评估基于Qualcomm Snapdragon 8 Elite Gen 5。

**📊 数据集**

使用面部交换（DiffusionDB、Human‑Art）、虚拟试衣（VITON‑HD）和MagicBrush（MagicBrush数据集）等任务；模板/掩码比例取自Alibaba Spring Festival数据集。

**📈 对比分析**

与均匀量化(IntQ)和CacheGen基线比较；在5G、4G+5G混合和WiFi环境下评估；结果显示平均延迟比CacheGen低1.82×、比全局推理低3.27×，p90延迟下降44%；缓存存储比无压缩减少3.06×；PSNR/SSIM与无压缩接近，用户研究相似度90%+。

**⚠️ 局限性**

局限性：仍需云端缓存，无法即时响应动态模板更新；预构建缓存过程耗时；在极低或极高带宽下收益有限；实验仅在特定设备和模型上验证。

---

## 573. Computing the Nucleon of Matching Games

**arXiv ID:** 2610.29731 | [PDF](https://arxiv.org/pdf/2610.29731v1)

**作者:** Haris Aziz `[一作]` (UNSW), Gergely Csáji `[通讯]` (HUN-REN Institute for Computer Science and Control)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文针对匹配博弈的核素（nucleon）求解问题，给出了多阶段的多项式时间组合算法，并证明在容量为3的b-匹配博弈和宽度为3的分区匹配博弈中求解任一核素分配是NP-难的。

**💡 创新点**

创新点在于：①首次实现了匹配博弈核素的多阶段组合求解（仅使用最大权匹配、最短路径和最小费用流）；②通过直接求和技术将核素计算与核心空性问题关联，推出了对应博弈类的NP-难性；③为容量两的b-匹配博弈给出了多项式时间的Turing归约至最小剩余优化问题。

**🔧 技术方法**

采用的技术主要包括：图匹配算法（最大权匹配）、最短路径算法、最小费用流、差分约束网络、哈达玛不等式和二分搜索来精确计算阶段阈值；以及线性子空间避免框架来处理容量为2时的可分离性。

**📊 数据集**

本研究为理论分析，不使用实验数据集。

**📈 对比分析**

与之前仅能使用线性规划或椭圆法求解的核素算法相比，本文的组合算法在匹配博弈中实现了多项式时间求解，且在b-匹配和分区匹配博弈中通过归约证明了求解核素的NP-难性，表明该问题在这些更一般类博弈中无法高效求解。

**⚠️ 局限性**

限制主要包括：①未实现强多项式时间算法；②容量为2的b-匹配博弈以及宽度为2的分区匹配博弈的核素计算仍未得到多项式时间解；③对核心空性判定为多项式时间的情况，本文的第一阶段归约仍然适用，但后续阶段的复杂性未被完全解决。

---

## 574. Sufficient and Necessary Causal Explanations for the Analysis of Autonomous Systems

**arXiv ID:** 2610.29724 | [PDF](https://arxiv.org/pdf/2610.29724v1)

**作者:** Tom Yaacov `[一作]` (King's College London), Moshe Y. Vardi `[通讯]` (Rice University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了必要因果解释（necessary explanations）以用于自动系统的失败避免，结合逼近责任度的启发式搜索算法。

**💡 创新点**

首次统一了实际因果框架下的失败重现（sufficient explanations）与失败避免（necessary explanations）两种目标，提出必要解释概念，并给出了可扩展的近似算法。

**🔧 技术方法**

使用了实际因果模型、责任度度量、启发式搜索以及基于黑盒仿真的实验评估。

**📊 数据集**

在自主驾驶仿真环境LiteRacer和MetaDrive中生成多种轨迹与障碍配置的数据集，涵盖不同轨道形状、长度与宽度。

**📈 对比分析**

与穷举搜索、DeCaF与随机搜索基线比较，结果显示必要解释方法在模拟次数更少、对抗样本有效率更高、相似度更近且运行时间与随机基线相当。

**⚠️ 局限性**

依赖于可观测且可修改的障碍，假设必要解释存在；责任度估计仍需大量样本，且在极其复杂或非可观测环境下可能不够精确。

---

## 575. S-PASS: Star-Topology Pinching-Antenna Systems with Pipelined Antenna Pre-Positioning

**arXiv ID:** 2610.29700 | [PDF](https://arxiv.org/pdf/2610.29700v1)

**作者:** Wei Jiang `[一作]` (German Research Center for Artificial Intelligence), Hans D. Schotten `[通讯]` (University of Kaiserslautern)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了一种星形波导网络S-PASS，并提出流水线天线预定位协议PAP，用于在一条射频链路上动态移动天线以降低机械重定位延迟。

**💡 创新点**

创新点在于：①星形拓扑使多条波导共享单一射频链路，降低布线成本；②PAP将空闲波导的天线移动与其他波导的传输重叠，形成类似处理器流水线的时隙重叠；③闭式分析得到调度效率、可吸收延迟阈值及与固定放置基准的对比。

**🔧 技术方法**

使用了时分多路访问、机械执行器模型、波导传输模型、闭式调度效率推导和仿真验证；对移动速度、重定位时间等参数做敏感性分析。

**📊 数据集**

通过在圆盘上均匀随机放置用户、生成2000组用户分布进行 Monte Carlo 仿真，使用 10^5 个点进行均值与百分位数统计；实验中假设自由空间通道且α_g=0。

**📈 对比分析**

与固定放置的两用户 NOMA 及星形 TDMA 做对比，采用调度效率 η 与有效总速率 R_eff 作为性能指标。结果显示：当 S≥S* 时，PAP 的调度效率达到开关限制；在机械延迟可被吸收的区间内，PAP 的总速率可超过固定放置 6.7%（平均）和 17%（边缘用户），5th 百分位收益可达 9%。

**⚠️ 局限性**

局限性包括：仅使用理论的机械延迟模型，未考虑实际执行器参数；仅在自由空间模型下验证，未覆盖 THz 频段和多径效应；假设用户静止，未考虑快速时变环境；对较大移动距离时需要秒级的服务周期，实际部署需进一步验证。

---

## 576. Every Part Has Its Role: Part-of-Speech Effects in LLM Post-Training

**arXiv ID:** 2610.29843 | [PDF](https://arxiv.org/pdf/2610.29843v1)

**作者:** Laya Iyer `[一作]` (Stanford University), Chen Shani `[通讯]` (Tel Aviv University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探索在LLM后训练中按词性（名词、动词、形容词）进行概念级数据增强的效果

**💡 创新点**

发现动词增强最为稳健，且并非多词性增强就能带来更好性能，提出词性应作为概念监督的设计维度

**🔧 技术方法**

使用概念级替换数据增强、next‑token 目标、CKA 与线性探测分析、LoRA 微调

**📊 数据集**

arXiv 摘要、新闻文本、YouTube 评论及其合并语料；基础模型为 Llama‑3‑8B

**📈 对比分析**

与匹配的无增强基线比较，通过 held‑out perplexity 和七个下游分类任务（GLUE、SNLI 等）评估；动词增强在多数语料与任务中提升准确率，perplexity提升呈现不稳定性

**⚠️ 局限性**

实验仅限单一模型规模与英语语料，且仅使用数据增强方式，概念提取质量因词性与语义关系而异，可能导致噪声与偏差

---

## 577. Linear Complementary Pairs of MDS Codes from Elliptic Curves

**arXiv ID:** 2610.29835 | [PDF](https://arxiv.org/pdf/2610.29835v1)

**作者:** Adler Marques `[一作]` `[通讯]` (Federal University of Rio de Janeiro), Adler Marques (Federal University of Rio de Janeiro)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了来自椭圆曲线的最大距离可分（MDS）码的线性互补对（LCP）。通过限制评估除子到子群余类，构造了MDS椭圆码的LCP，并获得了明确的无限家族。

**💡 创新点**

创新点在于提出了一种多余类的代数构造方法，成功打破了零和关系，从而克服了现有构造的几何障碍，证明了在所有有理点上评估时，LCP不能完全由MDS码组成。

**🔧 技术方法**

使用了阿贝尔-雅可比定理和代数几何码的理论，结合椭圆曲线的群结构进行构造。

**📊 数据集**

应用了Mordell曲线和超奇异曲线的具体实例，构造了LCP的明确无限家族。

**📈 对比分析**

通过与现有方法的比较，本文的方法在构造LCP时能够获得更长的码长度，并且在特定条件下能够确保构造的码是MDS。

**⚠️ 局限性**

限制在于构造的LCP不能完全由MDS码组成，且在评估所有有理点时存在几何限制。

---

## 578. Robot Generalization Beyond Volume: The Diversity Path via Zero-Shot Capabilities

**arXiv ID:** 2610.29829 | [PDF](https://arxiv.org/pdf/2610.29829v1)

**作者:** Rafael Felix `[一作]` `[通讯]` (runibiLABS), Rafael Felix (runibiLABS)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出基于社区开发者共建的标准机器人平台（Origami）和共享潜在空间，实现机器人零样本泛化；

**💡 创新点**

将多模态对齐、生成式增强和噪声标签学习融合到共享潜在空间，并以社区贡献数据为多样性来源，验证“多样性优于体量”的假设；

**🔧 技术方法**

采用零样本学习、多模态对齐（如 CLIP）、生成式数据增强、噪声标签学习、Real2Sim2Real 循环、可解释安全检测等技术；

**📊 数据集**

使用社区贡献的多任务、多环境、多对象演示数据（通过 Origami 平台记录），未引用公开标准数据集；

**📈 对比分析**

通过在模拟与真实机器人上测试未见任务/环境/对象的成功率，比较不同贡献者多样性下的零样本成功率和 sim‑to‑real 传递率；预期随贡献者数提升而显著提升；

**⚠️ 局限性**

存在数据授权、质量控制、模拟偏差、覆盖偏差、贡献者留存等限制，且安全与真实世界泛化仍需进一步验证。

---

## 579. HuMiT: Low-Latency Whole-Body Humanoid Teleoperation via Compact Reference Tracking

**arXiv ID:** 2610.29903 | [PDF](https://arxiv.org/pdf/2610.29903v1)

**作者:** Liu Cao `[一作]` (Tsinghua IIIS), Mengdi Xu `[通讯]` (Tsinghua IIIS)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一种低延迟全身人机遥操作系统HuMiT，利用仅根高度、根速度和四个稀疏关键点的SE(3)姿态作为紧凑参考目标，直接驱动统一的控制策略，在实际Unitree G1机器人上实现高动态全身动作与协同行走操作。

**💡 创新点**

创新点在于：①证明仅用紧凑参考目标即可实现与全参考或稠密参考相当的全身跟踪；②通过移除在线优化重映射和未来帧依赖，显著降低系统延迟；③统一的全身策略实现行走与操作的紧密耦合，突破传统上/下肢分离控制的局限。

**🔧 技术方法**

技术手段包括：消费级VR头盔+手柄+腿部运动跟踪进行感知；闭环PD控制；强化学习（PPO）训练的非对称Actor‑Critic网络；紧凑参考目标生成（闭式缩放校准）；以及离线重映射用于训练。

**📊 数据集**

使用公开的人类动作数据集 AMASS、LAFAN1 和 BONES‑SEED，用于生成离线重映射样本并训练政策。

**📈 对比分析**

与 SONIC、SONIC‑LL、ScaleBFM‑M/XL、ExtremControl 等方法比较；HuMiT 在三组数据上的关键点误差、关节误差、根速度误差、根角速度误差均与全参考或更稠密参考方法相近，成功率保持竞争水平；通信到推理延迟约 20 ms，端到端延迟 65 ms；实地测试中实现快速抓球等反应任务。

**⚠️ 局限性**

局限性包括：训练数据主要覆盖站立和上肢动作，缺少爬行、跪下等接地动作导致这些行为的跟踪性能下降；未在高负载条件下训练，无法支持重物搬运；对操作空间差异（如手臂工作空间）适应性有限。

---

## 580. Spacetime Splines for Adaptive Robot Motions

**arXiv ID:** 2610.29839 | [PDF](https://arxiv.org/pdf/2610.29839v1)

**作者:** Boxuan Zhang `[一作]` (Technical University of Munich), Sylvain Calinon `[通讯]` (Idiap Research Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种时空样条编码方案，利用共享的贝塞尔曲线与演示特定的相位函数同时估计机器人运动的空间轨迹、时间进程和部分演示的覆盖区间，并实现在线重定时与同步；在仿真和实际人机协作实验中验证了该方法的鲁棒性与适应性。

**💡 创新点**

创新点在于：
1) 将空间几何与时间进程同构编码为连续的时空曲线，打破传统分离优化的局限；
2) 引入演示特定的单调相位函数与覆盖区间估计，使得部分演示也能参与共享轨迹学习；
3) 采用交替优化与自适应细化（基于误差分割与G^1连通性评估）实现模型容量的自适应分配；
4) 通过在线相位估计实现机器人在视觉跟踪丢失时的暂停与恢复，提升人机协作的鲁棒性。

**🔧 技术方法**

技术手段包括：
- 立方贝塞尔曲线与分段多项式相位模型；
- 交替坐标优化（先固定空间再优化时间，反之亦然）；
- 单调性与端点约束的非线性最小化；
- De Casteljau 分割与误差驱动的自适应细化；
- G^1 连续性判定与重优化；
- 通过相位同步实现实时控制。

**📊 数据集**

数据集与实验：
- LASA 手写轨迹数据集（六种形状，七条演示，包含完整与部分演示）；
- 真实物理人偶与 7-DOF Franka 机器人进行患者翻身辅助实验，使用 RGB 摄像头跟踪人手。

**📈 对比分析**

比较方法：DTW+样条、线性相位模型、DMP、ProMP。实验结果显示：
- 在完整演示下，本文方法在轨迹与几何 RMSE 上略逊于 DTW+样条但优于线性相位、DMP、ProMP；
- 在部分演示下，本文方法显著优于所有基线（RMSE 约为 0.02 vs 0.01-0.08，覆盖区间 MAE 约为 0.035 vs 0.005-0.14），且在不同空间扰动水平下鲁棒性更强；
- 自适应细化进一步降低误差，同时实现了更高的运行时效率；
- 机器人实验展示了在线相位同步与暂停恢复的实用性。

**⚠️ 局限性**

局限性：
- 需要先验的 anchor 演示以消除相位重参数化歧义；
- 交替优化与自适应细化在大规模演示或高维空间下计算成本较高；
- 对极端时间畸变或不连续轨迹的适应性有限；
- 依赖单调相位假设，无法处理含循环或多分支路径的场景。

---

## 581. Pre-emptive Carbon Aware Scheduling for Edge Workloads

**arXiv ID:** 2610.29945 | [PDF](https://arxiv.org/pdf/2610.29945v1)

**作者:** Cecilia Chen `[一作]` (Northwestern University), Emilie Rivkin `[通讯]` (Northwestern University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了边缘服务器与云之间的碳感知预抢占式调度，给出单机最优多项式算法并证明多服务器不可迁移情形无多项式近似；

**💡 创新点**

创新在于将时间变化的碳成本与任意优先级约束结合，利用增量最大流实现高效最优调度，并将云外包问题映射为拒绝惩罚预抢占调度，从而得到 FPTAS；

**🔧 技术方法**

主要技术包括网络流（增量最大流与最小成本流）、多阶段增量流、动态规划（Lawler）以及预抢占调度的拒绝惩罚转换；

**📊 数据集**

实验使用合成数据和阿里巴巴生产级任务跟踪（Alibaba cluster trace）两类数据集；

**📈 对比分析**

与直接最小成本流、无残留重启版本及 GPU 实现进行对比，平均 4.96 倍、2.07 倍加速；在阿里巴巴实例上提升至约 10 倍；FPTAS 在可接受规模下可得到 (1+ε) 近似；

**⚠️ 局限性**

局限性包括多服务器无迁移情形无多项式近似；云外包的 Lawler 动态规划实现规模受限，难以在大规模实例中快速求解。

---

## 582. ERCLIP: Small-Data Adaptation of CLIP for Free-Length Image-Text Retrieval

**arXiv ID:** 2610.29939 | [PDF](https://arxiv.org/pdf/2610.29939v1)

**作者:** Hoang-Bao Le `[一作]` (Dublin City University), Cathal Gurrin `[通讯]` (Dublin City University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在20k对长图文数据和单GPU资源下，改进CLIP的训练方式，使其在保持原有推理成本的前提下提升长文本检索性能。

**💡 创新点**

通过将长句拆分为句子并在训练时对句子嵌入进行均值或残差Transformer融合，并结合冻结CLIP的保留损失，既提升长文本检索，又缓解零-shot分类性能下降。

**🔧 技术方法**

使用句子级编码与融合（均值池化、残差Transformer加门控）、InfoNCE对比损失、冻结教师保留损失，以及CLIP双编码器架构。

**📊 数据集**

训练使用COCO与ShareGPT4V共享的20k图文对，评估基准包括Urban1K、ShareGPT4V-1k、DOCCI、Long-DCI、COCO val2017，以及CIFAR-10/100和ImageNet的零-shot分类。

**📈 对比分析**

与原CLIP及大型长文本检索方法（LongCLIP、SmartCLIP、TULIP）对比，单轮20k训练可提升CLIP长文本R@1平均6.5–11.2点，接近70%长文本性能差距；保留损失恢复30–42% ImageNet零-shot准确率，且推理成本保持不变。

**⚠️ 局限性**

仅在单GPU单轮训练下实验；长文本推理仍被截断至77个token，未评估句子级推理、不同优化策略及长文本漂移；结果基于单随机种子，未覆盖多域或大规模数据。

---

## 583. DewHand: Direct Metric World-Space Hand Motion Reconstruction from Egocentric Videos

**arXiv ID:** 2610.29936 | [PDF](https://arxiv.org/pdf/2610.29936v1)

**作者:** Naru Suzuki `[一作]` (University of Tokyo), Shuhei Kurita `[通讯]` (National Institute of Informatics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出DewHand模型，能够从单目第一人称视频直接在共享的度量世界坐标系中同时重建两只手和相机的运动轨迹。

**💡 创新点**

创新点在于将手与相机运动统一为单一生成流程，使用流匹配条件分布以处理遮挡与视角缺失，并在同一网络中同时监督相机尺度与手姿。

**🔧 技术方法**

采用VGGT‑Ω几何特征和HaMeR手部特征作为视觉条件，利用多模态扩散Transformer（MMDiT）和流匹配学习手相机协同运动。

**📊 数据集**

主要使用HOT3D和ARCTIC数据集训练，并在HoloAssist、H2O、TACO等跨设备数据上进行零样本评估。

**📈 对比分析**

与StableHand、HaWoR、HandFlow等基线相比，DewHand在WA‑MPJPE、相机尺度恢复和加速度误差等指标上均取得最低误差，显著提升了世界空间手部定位的精度与时序一致性。

**⚠️ 局限性**

局限在于仍以固定窗口为推理单元，未显式建模手物交互与接触；对极端遮挡或极大相机运动的鲁棒性尚待进一步验证。

---

## 584. Graph Topology Meets Energy-Driven Pathfinding: Preserving Contextual Consistency in Retrieval-Augmented Generation

**arXiv ID:** 2610.29933 | [PDF](https://arxiv.org/pdf/2610.29933v1)

**作者:** Yufeng Li `[一作]` (Central South University), Fengxiao Tang `[通讯]` (Central South University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

为多跳问答提出ThreadRAG框架，构建语义拓扑图并用能量驱动路径寻找线性化文本证据。

**💡 创新点**

通过语义拓扑映射保留文本连贯性，能量驱动路径寻找解决图‑LLM模态不匹配，形成上下文完整的ST‑Path。

**🔧 技术方法**

语义拓扑映射（STM）、能量驱动路径寻找（EDP）、自适应推理代理；结合检索、图构造、PPR扩散、线性化等技术。

**📊 数据集**

HotpotQA、2WikiMultihopQA、MuSiQue。

**📈 对比分析**

与多种文本、KG‑Chunk、KG‑Path、KG‑Triplet等基线对比，ThreadRAG在所有数据集均取得最高EM；HotpotQA提升14.4%EM，2Wiki提升5.8%EM；token消耗仅占基线的23%。

**⚠️ 局限性**

对稀疏链接任务效果有限；能量扩散后仍可能漏掉长距离依赖；线性化过程可能产生轻微叙事碎片。

---

## 585. Privacy Preference Controls for Agentic Systems

**arXiv ID:** 2610.29929 | [PDF](https://arxiv.org/pdf/2610.29929v1)

**作者:** Shreya Kochar `[一作]` (University of California, San Diego), Sebastian Zimmeck `[通讯]` (Wesleyan University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了面向代理系统的可执行隐私偏好控制分类法，完成对30款AI产品的隐私控制审计，并基于MCP/A2A协议开发了八个原型系统验证该分类法的可行性。

**💡 创新点**

创新点在于首次从法律、用户研究与行业实践三大视角构建完整的代理系统隐私偏好分类，扩展了传统GPC范式，并通过原型演示将偏好信号嵌入代理协议，实现端到端的隐私控制传递。

**🔧 技术方法**

采用的技术包括法律和实证研究驱动的分类法构建、LLM辅助的文档检索与验证、MCP/A2A协议的拦截器与元字段、以及GPC信号的自定义传递。

**📊 数据集**

数据集主要为22家企业（包括18家公司）发布的30款AI产品的官方文档、隐私政策和开发者手册，利用LLM获取并人工校验，构成了对隐私控制设置的实证审计。

**📈 对比分析**

通过对比已实现的偏好控制与分类法子类别，统计显示72%的设置落入分类法范围，25%不在范围内；原型演示表明在保留功能的前提下实现隐私偏好传递可行，未给出传统性能指标，但通过案例验证了可操作性。

**⚠️ 局限性**

局限性包括仅关注未来数据处理，排除删除等历史处理；审计受限于公开文档与美国司法辖区；无法验证偏好是否被真正执行；未覆盖法律例外与合规细节；并且在多跳代理链中可追溯性与可验证性仍是挑战。

---

## 586. Unintended Side Effects of Value Training in LLMs

**arXiv ID:** 2610.29922 | [PDF](https://arxiv.org/pdf/2610.29922v1)

**作者:** Siddhesh Pawar `[一作]` (University of Copenhagen), Isabelle Augenstein `[通讯]` (University of Copenhagen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在四种指令微调后的大型语言模型（Llama-3.1‑8B、Qwen3‑8B、Gemma‑3‑12B、Ministral‑3‑8B）上，针对霍夫斯泰德的12个价值维度（IDV、PDI、MAS、UAI、LTO、IVR）分别做后训练微调，并构造两阶段合成的情景‑响应数据进行训练。随后对微调后的模型在跨价值迁移、风险安全、社会认知（EmoBench、FANToM）等多项基准上进行评估，揭示价值微调会对未训练维度、风险偏好、社交推理等产生意外副作用。

**💡 创新点**

创新点在于：
1) 系统性检验单一价值训练对其它价值维度的迁移效应；
2) 通过多基准评估显示价值微调会诱发安全偏差与社交认知退化；
3) 通过共享更新位点定位，揭示不同价值适配器共享的机制，说明副作用来自模型内部的共享表示。

**🔧 技术方法**

使用技术包括：QLoRA 微调框架；两阶段合成数据生成（场景→价值响应）；LLM judge 自评；配对检验与 Benjamini–Hochberg 校正；共享更新位点定位（层级+位置筛选）以及常见基准（MWE、AIRiskDilemmas、EmoBench、FANToM、EM）。

**📊 数据集**

使用的数据集：
- 10k 合成情景‑响应对（基于 Touché23‑ValueEval、CDEval、Sociocultural Statements）；
- Cover‑letter dataset（100 招聘场景）；
- 评估基准：MWE（153 构造）、AIRiskDilemmas（10,399 场景）、EM（24 问题）、EmoBench（400 项）和 FANToM（870 项）。

**📈 对比分析**

比较方法：对每个极值适配器与其基线模型在同一测试集上计算 Δ（likelihood、win rate、百分点），并使用配对检验和 BH‑FDR 校正评估显著性。实验结果显示：目标价值提升显著，但大多数未训练维度也出现显著偏移；安全相关指标（MWE 相关 trait、AIRiskDilemmas 价值排名、EM misalignment rate）普遍下降；社交情感应用（EmoBench EA、FANToM belief choice）下降，尤其在自我情感场景。不同基础模型表现不一，Llama 与 Qwen 对风险排名变动最敏感。

**⚠️ 局限性**

limitations：
- 合成数据为单一价值标签，实际行为往往涉及多重价值，可能导致跨维度迁移偏差；
- 只在后训练阶段进行单值微调，未检验在预训练或并行训练中的效果；
- 研究聚焦霍夫斯泰德维度，无法直接推广到多维度复合价值或真实文化语境；
- 部分基准（如 EM）依赖 LLM judge，可能带来评估偏差。

---

## 587. Benchmarking Large Language Models for Detecting Satirical Code-Mixed Sentences

**arXiv ID:** 2610.29907 | [PDF](https://arxiv.org/pdf/2610.29907v1)

**作者:** Răzvan-Alexandru Smădu `[一作]` (National University of Science and Technology POLITEHNICA), Mihaela-Claudia Cercel `[通讯]` (University of Bucharest)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了首个大规模罗马尼亚语讽刺新闻的代码混合数据集CS-SeLeRoSa，包含四种语言（英、德、法、意）与四种生成方法，并通过人工评审得到CS-SeLeRoSa-Gold，随后评测多种大型语言模型在该数据集上的讽刺检测性能。

**💡 创新点**

首次提供罗马尼亚语代码混合讽刺文本基准，系统对比了不同生成策略和目标语言对模型表现的影响，并证明微调可恢复甚至超越单语环境下的性能，揭示代码混合对讽刺检测的挑战。

**🔧 技术方法**

采用随机替换、短语翻译、依赖树插入和LLM直接生成四种代码混合方法；使用Gemma 3 27B进行机器翻译；在零样本、链式思考和微调三种设置下评估Gemma、Llama、RoMistral、RoLlama、RoGemma等开源模型，以及GPT系列闭源模型。

**📊 数据集**

基于SeLeRoSa讽刺新闻数据集（约13.8k句），通过合成生成得到约600k条代码混合样本CS-SeLeRoSa，并抽取715条人类验证后保留391条的CS-SeLeRoSa-Gold作为金标准。

**📈 对比分析**

在零样本下，Gemma 4 31B达到约70%宏F1，紧随其后的是Gemma 4 26B和GPT-5 mini；微调后Gemma 4 31B提升至约77%宏F1，明显优于所有闭源模型的零样本表现；链式思考对弱模型有显著提升，但对顶尖模型效果有限。

**⚠️ 局限性**

主要限制包括：合成数据质量不一（随机替换产生噪声）；仅以罗马尼亚语为矩阵语言，未覆盖更广语言场景；实验仅涵盖少数小型LLM，未检验更大模型；人类评估仅覆盖部分样本，缺乏对完整数据集真实性的验证。

---

## 588. DynaCI: Generating GitHub Actions Workflows via Dynamic Analysis

**arXiv ID:** 2610.29890 | [PDF](https://arxiv.org/pdf/2610.29890v1)

**作者:** Javid Ditty `[一作]` (University of Michigan-Dearborn), Foyzul Hassan `[通讯]` (University of Michigan-Dearborn)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于动态分析与大型语言模型的工具，自动将开发者本地测试环境转换为GitHub Actions CI工作流；

**💡 创新点**

创新点在于：①通过在本地环境执行测试命令收集运行时信息（命令、依赖、文件访问、环境变量等）；②用启发式规则快速构造初始工作流；③采用检索增强生成（RAG）+LLM进行分层细化（按作业拆分），显著降低手工编写成本；

**🔧 技术方法**

技术包括：动态跟踪探针（Command、Duration、File、Package、Shell env、Python env），基于运行时数据的非参数与参数化启发式；LLM（GPT‑4o‑mini）+RAG（ADA‑002嵌入）进行工作流细化；静态分析工具校验生成结果；

**📊 数据集**

使用了50个热门Python项目的GitHub Actions工作流作为评估集，来源于公开的CI Workflow Dataset（15,285个工作流）用于调优参数；

**📈 对比分析**

与四个现有工具（命令‑only、启发式‑only、Transformer‑based、Prompt‑optimization）在BLEU、F1、生成时间、token消耗、修改量等指标对比；实验显示其BLEU 52.81%/F1 62.06%，比基线高约21%/21%；生成时间约5m33s，Token约654k，需手工修改的工作流仅44%，56%无需修改；

**⚠️ 局限性**

局限性：目前仅支持Linux/Ubuntu环境和Python项目；需要先构建Docker化本地环境；仍存在部分错误需人工修复；LLM成本较高；对非GitHub Actions或其他语言/CI平台的适用性需要进一步验证。

---

## 589. Coherent differentiation with a monoid of degrees: derivatives, Taylor expansion, finite differences...

**arXiv ID:** 2610.29874 | [PDF](https://arxiv.org/pdf/2610.29874v1)

**作者:** Thomas Ehrhard `[一作]` (Université Paris Cité), Aymeric Walch `[通讯]` (Université de Toulouse)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种新的连贯微分理论，扩展了之前的微分λ演算和微分线性逻辑的概念，特别是通过引入一种基于部分可加单元的微分结构。

**💡 创新点**

创新点在于引入了一个任意可交换部分单元的元组索引，允许定义新的微分概念，包括有限差分和任意阶的泰勒展开。

**🔧 技术方法**

使用了范畴论中的切线函子和部分可加单元的结构，结合了微分和加法的概念。

**📊 数据集**

使用了与概率一致性空间相关的数据集，特别是涉及到非线性映射的分析函数。

**📈 对比分析**

通过与现有的微分λ演算和微分线性逻辑进行比较，展示了新理论在处理非线性映射和微分的能力上具有优势，尤其是在处理不确定性和非确定性计算时。

**⚠️ 局限性**

限制在于该理论的应用范围可能受到部分可加单元的性质限制，且在某些情况下可能无法完全捕捉到传统微分的所有特性。

---

## 590. HAWK: Heterogeneity-Aware KV Cache Transfer for Efficient GPU-NPU Phase-Disaggregated LLM Inference

**arXiv ID:** 2610.29873 | [PDF](https://arxiv.org/pdf/2610.29873v1)

**作者:** Junhyeok Park `[一作]` (Georgia Institute of Technology), Kangho Kim `[通讯]` (Electronics and Telecommunications Research Institute)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 GPU–NPU 异构阶段分离的 LLM 推理系统，并提出 HAWK 机制以实现跨设备 KV 缓存高效传输。

**💡 创新点**

创新点在于：通过 RDMA 建立 GPU–NPU 直接传输通道，结合张量布局感知的 RDMA 缓冲区分配与两步张量转换，实现了跨设备张量传输低延迟且无后处理。

**🔧 技术方法**

技术：Linux RDMA 堆栈、GPU/ NPU peer memory client、张量布局感知 RDMA 缓冲区分配、两步张量转换、Mooncake Transfer Engine、vLLM 插件。

**📊 数据集**

数据集：Chatbot 使用 ShareGPT，Summarization 使用 LongBench，模型为 Llama 3.1 70B。

**📈 对比分析**

比较方法：与 GPU‑GPU 主机转存、GPU‑NPU 主机转存、GPU‑GPU 直接 RDMA、GPU‑NPU HAWK 四种配置对比。性能：HAWK 的 KV 缓存传输延迟接近 GPU‑GPU 直接，整体推理 TTFT、TPOT、SLO 满足率与 GPU‑GPU 直接差距 ≤ 7.88%，功耗效率提升 1.26×，成本效率提升 1.72×。

**⚠️ 局限性**

局限：仅在 Tenstorrent p150a NPU 上实现，需对不同 NPU 平台适配；NPU 解码性能略低导致 TPOT 略差；需进一步优化 NPU 软件栈；目前仅支持 KV 缓存，其他张量迁移待验证。

---

## 591. PAUSE: Search-Free Steering of Coding Agents Before Their First Source-Code Edit

**arXiv ID:** 2610.29870 | [PDF](https://arxiv.org/pdf/2610.29870v1)

**作者:** Viral Patel `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Youcheng Sun `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在线与离线调度框架，在编码代理首次请求源代码编辑之前给出单次规划干预，从而提升修复任务的成功率。

**💡 创新点**

创新点在于：①利用首次源代码编辑作为“search‑free”触发点，无需模型监测或周期检查；②一次性规划调用即可引导整个修复过程；③支持在实时运行与离线恢复两种模式；④实验验证了该方法在多工具链和多语言环境下的有效性。

**🔧 技术方法**

技术手段包括：LLM 编码代理（Qwen3-coder、Devstral-small）与 Planner（GPT‑5.6 Sol、Gemini、Qwen3‑30B‑Thinking）交互；预编辑断点检测与 Planner 生成自然语言修复计划；使用 OpenHands、SWE‑agent 等多种 harness；基于 SWE‑bench 评测框架进行评估。

**📊 数据集**

使用的数据集为 SWE‑bench Verified（500 个 Python 任务）和 SWE‑bench Multilingual（300 个跨语言任务），并利用 OpenHands 记录的轨迹用于离线恢复实验。

**📈 对比分析**

与 LivePlan（规则监测）和 SICA（周期检查）进行对比；在线实验显示在 Verified 上提升 14.8%、Multilingual 上提升 14.7%，Planner 令牌仅 9M；离线实验恢复率达 35.4%，相比重新开始节省 33–41% Executor 令牌；在三种 harness（OpenHands、SWE‑agent、Pi）和多种 Planner 上均表现出显著收益。

**⚠️ 局限性**

局限性包括：①需代理在轨迹中出现首次源代码编辑请求且路径规则正确；②Planner 只能读取已记录的证据，无法直接访问工作区或执行新命令；③单次规划可能无法适应后续观察导致的计划失效；④离线恢复要求可完整还原的前缀；⑤实验仅覆盖 SWE‑bench 代码库，未验证更大规模或不同开发流程的适用性。

---

## 592. AT-Differ: Dataset for Audio-Text Difference Captioning

**arXiv ID:** 2610.29836 | [PDF](https://arxiv.org/pdf/2610.29836v1)

**作者:** Akihiro Fujii `[一作]` (University of Tokyo), Hiroshi Saruwatari `[通讯]` (University of Tokyo)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了音频-文本差异描述（difference captioning）任务，并构建了新数据集 AT-DIFFER，用于生成音频与文本之间差异的文字说明。

**💡 创新点**

创新点在于：①从单一语义对齐分数转向生成差异描述，提供更细粒度的解释；②系统化模拟多种差异类型并生成对应描述；③结合差异检测、差异类型分类与描述生成的多任务学习框架。

**🔧 技术方法**

采用 Qwen3‑Omni 多模态大语言模型（LLM），通过 LoRA 在自注意力层进行微调，并在训练时联合优化差异检测、差异类型分类与差异描述三项任务。

**📊 数据集**

使用 AT-DIFFER 数据集，该数据集基于 FSD50K‑Solo 语音片段，包含单声、同时声和顺序声三种音频结构，并设计了语义差异、文本省略、声音省略、顺序不一致、同步不一致等五种差异类型。

**📈 对比分析**

与原始 Qwen3‑Omni（无微调）及两种随机生成差异描述的基线相比，方法2（联合任务串联）和方法3（任务分离加权）在差异检测（94.8%）、差异类型分类（91.9%）以及差异描述的 BLEU、METEOR、ROUGE‑L、CIDEr、SPICE、BERTScore、SBERT 等指标上均显著优于基线，方法1（仅差异描述）表现最差。

**⚠️ 局限性**

局限性包括：仅考虑最多两个声事件和最多两种差异类型；未涉及时长、计数、空间等差异；数据来源单一（FSD50K‑Solo），可能导致模型泛化受限；对不同大语言模型的适用性尚未系统评估。

---

## 593. MemoryTax: The Bitter Lesson of Agent Memory

**arXiv ID:** 2610.29823 | [PDF](https://arxiv.org/pdf/2610.29823v1)

**作者:** Zedong Peng `[一作]` `[通讯]` (Shanghai Jiao Tong University), Zedong Peng (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

比较了Mem0、Zep、BM25检索、完整上下文以及Pi、Codex、Claude Code等编码代理在LoCoMo对话中的成本与准确率。

**💡 创新点**

发现强大模型对预构建记忆的依赖降低，且编码代理在成本-准确率前沿领先，揭示了记忆写入模型所设限的“记忆上限”。

**🔧 技术方法**

使用Mem0、Zep、BM25、完整上下文、Pi、Codex、Claude Code等记忆与编码技术，结合GPT‑4o‑mini、GPT‑6 Luna、Claude Opus 5.5三大模型，并用GPT‑4o‑mini评判器评估答案。

**📊 数据集**

以LoCoMo对话 conv‑26 为实验数据，包含419回合、152个单跳/多跳/时序/开放域问题。

**📈 对比分析**

通过成本‑正确率曲线比较，每种方法在成本轴上按对话成本累加；结果显示Mem0在弱模型下表现最好但成本高，Pi+Claude Opus 5.5以99%准确率在完整上下文成本的一半实现，强模型下记忆方法趋于平台化。

**⚠️ 局限性**

局限性包括仅使用单一对话与单次实验、评判器基于GPT‑4o‑mini且评价尺度较宽、照片说明仅编码代理可读、上下文完全适配模型窗口、默认设置可能不最优以及成本以列表价格计量。

---

## 594. Controlling Variability in~Neuroimaging Analysis Pipelines with~Software Product Lines

**arXiv ID:** 2610.29821 | [PDF](https://arxiv.org/pdf/2610.29821v1)

**作者:** Gauthier Le Bartz Lyan `[一作]` (University of Rennes), Mathieu Acher `[通讯]` (University of Rennes)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对NARPS开放管线进行重现与变异性分析，并探讨将神经影像处理管线构造为软件产品线（SPL）的方法。

**💡 创新点**

首次发现NARPS管线具备SPL特征，提出通过标准化接口实现“Code Once, Run Anywhere”理念，以提升可复现性和可维护性。

**🔧 技术方法**

利用Docker/Neurodocker容器、Nipype中间件、SPL设计理念，对MATLAB、SPM12版本及硬件平台（Intel/AMD x86、Apple ARM）进行可变配置。

**📊 数据集**

使用NARPS实验提供的fMRI原始数据集进行实验。

**📈 对比分析**

在隔离的Docker环境中互换关键软件组件（MATLAB、SPM12版本、CPU/ARM架构），比较预处理与统计结果的数值差异；差异范围为10^-5~10^-2，说明核心运算稳健，但需评估对统计推断的影响。

**⚠️ 局限性**

仅针对SPM12/MATLAB生态进行实验，未覆盖其他主流工具；复制工作受限于文档不足与工具兼容性，且尚无完整的标准化接口实现。

---

## 595. A Learned Musical-Time Coordinate for Piano Audio-to-Score Transcription

**arXiv ID:** 2610.29928 | [PDF](https://arxiv.org/pdf/2610.29928v1)

**作者:** Min-Hsing Wang `[一作]` (National Taiwan University), Tsung-Ren Huang `[通讯]` (National Taiwan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种端到端钢琴音频转谱系统，系统在训练时学习下拍相位作为音乐时间坐标，既用于解码时的计时约束，又能自我分段拼接整曲；

**💡 创新点**

创新点在于将下拍相位（downbeat phase）与谱件生成联合学习，并通过傅里叶位置编码将相位信息注入解码器，同时实现无外部分段器的全曲自我分段与拼接；

**🔧 技术方法**

使用预训练的hFT‑Transformer作声学特征提取器，随后采用交叉注意力、Conformer、TCN+BiMamba网络估计下拍相位；傅里叶位置编码将相位映射到向量；Transformer解码器在每一步通过跨注意力结合音频内容与相位；训练采用教师强制的交叉熵和相位交叉熵；推理时通过窗口滑动、平均相位、线性插值确定下拍，并用Viterbi解码时间/调号；

**📊 数据集**

训练使用合成钢琴数据集Syn（由HumSyn+MuseSyn构成，共546首曲目），验证集46首；测试在Syn（45首）和ASAP（39首）上；

**📈 对比分析**

与基准Piano‑A2S（含Beat This!外部下拍分段）比较，采用MV2H（尤其F_meter）和OMR‑NED指标；实验显示系统在F_meter和MV2H_5上显著优于Piano‑A2S，且在全曲转录中不依赖外部分段器即可完成；

**⚠️ 局限性**

局限性包括：1）下拍相位估计在真实人类演奏上仍有误差，导致部分节拍误判；2）模型主要在合成演奏上训练，泛化到真实录音时表现受限；3）仅针对钢琴音频，尚未验证对多声部或其他乐器的适用性。

---

## 596. Your Benchmark Is an Empirical Measure Over Difficulty

**arXiv ID:** 2610.29927 | [PDF](https://arxiv.org/pdf/2610.29927v1)

**作者:** Yifan Sun `[一作]` (University of Illinois Urbana Champaign), Huan Zhang `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文从测度论的视角重新定义基准测试，视其为难度分布的经验测度，并基于此发现并量化了两类失败模式——难度分布不匹配和覆盖缺口；

**💡 创新点**

创新点在于：①将基准视为经验测度，提出Wasserstein距离衡量难度分布不匹配；②引入最大间隙指标和假设检验框架定量评估覆盖缺口；③提出BenchFill算法，利用LLM重写生成难度填充题目，显著缩小覆盖缺口；

**🔧 技术方法**

核心技术包括：测度论与Wasserstein距离、Item Response Theory（1PL）进行难度估计、LLM驱动的重写与难度注释模块、统计假设检验与最优检验界限分析；

**📊 数据集**

使用了25个跨7领域的流行基准（包括MATH、AIME、MMLU、HumanEval、BBH等），以及公开的LLM性能数据和GitHub/Hugging Face上的BenchFill填充数据；

**📈 对比分析**

与传统基准评估相比，BenchFill在8个代表性基准上将前三大间隙总和平均降低32.7%（比基线提升17.5%），目标难度偏差均低于0.01，证明该方法能有效提升覆盖度并保持目标难度；

**⚠️ 局限性**

局限性包括：①重写生成仍需人工审查，人工成本不可忽视；②在极大难度范围或极端题型下，LLM可能无法精确定位难度；③实验仅覆盖8个基准，其他领域的适用性待进一步验证；

---

## 597. Task-Preserving Multilingual Instruction Tuning for Open-Ended QA in WMT 2026 MIST

**arXiv ID:** 2610.29926 | [PDF](https://arxiv.org/pdf/2610.29926v1)

**作者:** Aleix Sant `[一作]` (Telefónica Innovación Digital), Carlos Escolano `[通讯]` (Universitat Politècnica de Catalunya)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在Qwen3-8B上使用LoRA进行多语言指令调优，专注于十种目标语言的开放式问答任务。

**💡 创新点**

创新点包括：①创建跨语言增强数据集CrossEuroAlpaca，实现指令、上下文与答案的语言交叉；②在MIST、EuroAlpaca和CrossEuroAlpaca的混合语料上训练LoRA，提升目标语言性能；③结合Gemma-4无参考评判和Aya评测套件，系统性评估多语言、跨任务表现。

**🔧 技术方法**

主要技术：LoRA适配(Qwen3-8B)，梯度累积+DeepSpeed并行训练，greedy解码+重复惩罚，使用Gemma-4做无参考评判，Aya套件做参考基准评估。

**📊 数据集**

使用的数据集：MIST官方任务数据、EuroAlpaca（11种语言的本地化指令数据）以及自制的CrossEuroAlpaca（110种源–目标语言对的跨语言指令）。

**📈 对比分析**

比较方法：在相同解码条件下对比基线Qwen3-8B与LoRA适配模型，通过Gemma-4评判和Aya指标（ROUGE‑L、BERTScore‑F1）评估。性能提升：目标语言QA‑OEG平均Gemma‑4分提升7.8%，Aya宏观ROUGE‑L提升41.5%，BERTScore‑F1提升2.9%；但在上下文依赖QA和非目标语言任务中出现显著回落。

**⚠️ 局限性**

局限性：仅使用单一无参考评判器Gemma‑4，缺乏人工注释验证；评判可能受语言偏好和校准不均衡影响；适配模型在非目标语言和非目标任务上表现不佳，需进一步平衡任务与语言的覆盖。

---

## 598. Privacy in Context Is Being Overlooked in the Age of General-Purpose LLMs

**arXiv ID:** 2610.29925 | [PDF](https://arxiv.org/pdf/2610.29925v1)

**作者:** Prakhar Ganesh `[一作]` (McGill University), Golnoosh Farnadi `[通讯]` (McGill University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文从隐私与上下文的角度出发，批判了当前大语言模型（LLM）提示隐私研究与工具过度追求“一刀切”统一框架而忽视了上下文差异，提出并系统评估了提示隐私的五大理想属性，并基于对31篇论文和11个工具的注释与分析，阐明了统一框架不可行性；随后给出三条路径：基于应用的LLM部署、上下文特定评估以及提升用户隐私素养。

**💡 创新点**

创新点在于：①首次将隐私与上下文的关系系统化地引入LLM提示隐私领域；②提出保护保证、性能、效率、域适应性与用户可访问性五个核心属性，并揭示它们之间的根本冲突；③揭露学术与工业实践中普遍存在的“统一框架漂移”现象，并以此为基础给出针对性的改进建议。

**🔧 技术方法**

主要技术方法为：系统文献综述与手工注释（31篇论文与11个工具）、属性框架构建、图表化呈现、对比分析与案例动机阐述；未涉及新的算法或模型实现。

**📊 数据集**

使用的数据集为：已公开的论文与工具清单（共31篇论文、11个工具），并未使用标准公开数据集进行实验评估。

**📈 对比分析**

比较方法主要是概念性对比与属性匹配分析，未进行定量实验。论文通过图表展示各方法在五大属性上的表现差异，强调了统一框架在保护、性能、效率、域适应性和用户可访问性方面的不可兼顾。

**⚠️ 局限性**

局限性在于：①缺乏大规模实证评估，无法量化提出框架在实际应用中的效果；②主要依赖主观属性划分，缺少统一度量；③研究聚焦于提示层面，未涉及模型训练阶段的隐私问题；④对不同攻击模型的鲁棒性分析不足。

---

## 599. Geospatial Dynamics of Online Attention to Nuclear Power in Japan Before and After Fukushima

**arXiv ID:** 2610.29924 | [PDF](https://arxiv.org/pdf/2610.29924v1)

**作者:** Yifan Sun `[一作]` (Kyoto University), Benjamin C. McLellan `[通讯]` (Kyoto University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

该研究利用2004-2024年日本47个都道府县的Google Trends搜索兴趣数据，对核能的公众信息需求进行长周期、空间化分析。

**💡 创新点**

创新点在于首次在都道府县层面长期追踪核能关注度，并揭示事故前的“主机局部”模式与事故后的“距离驱动”模式。

**🔧 技术方法**

采用Google Trends聚合搜索指数，并运用非参数Mann–Whitney U检验和Spearman相关分析进行统计建模。

**📊 数据集**

使用的数据集为Google Trends的“原子力発電所”主题搜索兴趣值及2011年3月相关查询的分词对比结果。

**📈 对比分析**

通过统计检验比较不同组别（主机与非主机、边境与非边境、距离远近）的搜索兴趣，结果显示主机县搜索兴趣显著高于非主机县，且在事故期间兴趣随距离呈负相关。

**⚠️ 局限性**

局限性包括搜索数据缺乏人口学信息、搜索词选择的主观性、以及Google Trends对低搜索量地区的归一化可能导致的误差。

---

## 600. Preventing Model Extraction in Contextual Linear Optimization

**arXiv ID:** 2610.29918 | [PDF](https://arxiv.org/pdf/2610.29918v1)

**作者:** Adam N. Elmachtoub `[一作]` (Columbia University), Wenxuan Liu `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a2602d71-93ab-4bad-974b-672788df8193`

**🎯 论文内容**

本文研究在 predict‑then‑optimize（PTO）框架下，针对线性上下文优化任务的模型提取攻击与防御。

**💡 创新点**

创新点在于：①引入 loss calibration 概念，判定在何种几何条件下决策一致性可唯一确定预测模型；②提出基于边界采样的自适应逆优化攻击，显著提高提取精度；③设计仅在预测成本接近决策边界时扰动决策的防御机制，并给出回报上界与攻击误差下界。

**🔧 技术方法**

使用技术包括逆优化（子最优损失、线性约束重写）、角度损失评估、边界采样策略、Frobenius 正则化、决策扰动算法、理论误差与回报分析。

**📊 数据集**

实验数据集包括合成实例（多维盒子约束）以及真实短视频推荐数据 KuaiRand‑1K（d=18，n=5）。

**📈 对比分析**

与统一采样、单次逆优化基线进行对比；在合成实例中，边界采样攻击的预测误差几乎为零，所需样本量显著低于对照；在推荐任务中，边界采样同样优于 INV；在防御场景下，给定相同回报，边界采样的提取误差被显著提升，回报增量保持在可接受范围。

**⚠️ 局限性**

局限性包括：只处理线性预测模型；假设攻击者已知模型结构；防御仅在决策接近边界时有效，对非凸或非线性上下文优化的适用性未知；实验规模有限，未考虑攻击者显式建模防御策略。

---

## 601. Characterizing and Defending Utility-Hijacking in Instruction-Tuning Data Selection

**arXiv ID:** 2610.29886 | [PDF](https://arxiv.org/pdf/2610.29886v1)

**作者:** Yahao Ren `[一作]` (Sun Yat-sen University), Jiawei Duan `[通讯]` (Sun Yat-sen University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究并证明了在指令调优中使用的自动数据选择器会被攻击者利用（所谓的utility-hijacking），并提出了一种基于生成+重排的供应方攻击、一个上尾指标CREST以及一种无需标签的双选择器一致性防御CONCERT，展示了在Qwen系列模型上的显著攻击效果与防御恢复。

**💡 创新点**

创新点在于①首次系统地分析供应方自适应攻击对数据选择器的影响；②提出了上尾量化指标CREST，可在无训练的前期直接评估攻击易感性；③设计了CONCERT，该防御通过审核超选比例和双选择器最小化交叉，既不依赖内容检测也不需要毒样本标注。

**🔧 技术方法**

技术手段包括：生成+重排（generate-and-rerank）攻击策略；CREST指标（利用上尾存活概率计算毒性浓度）；CONCERT防御（来源审核 + 选择器一致性重选，使用排名交叉判定）。

**📊 数据集**

实验使用了多种公开指令-响应数据集（如Alpaca‑GPT4、PoisonPy代码库、TruthfulQA、MMLU、HumanEval）以及自制带毒的指令集合，并在Qwen2.5-7B至Qwen3-32B模型上评估。

**📈 对比分析**

在攻击实验中，top‑k选择器在易感配对下将拒绝率从0%提升至94%，不安全代码比例升至56/90，误报率提升至7.04×；CONCERT防御将毒性浓度降至≈0，安全性恢复到与原始无毒模型相当，且对模型能力无显著影响。

**⚠️ 局限性**

局限性包括：防御依赖可靠的来源元数据，跨供应商分散攻击时检出率下降；生成+重排仅对固定生成器评估，未覆盖对生成器的自适应再优化；目前只验证了三类payload（拒绝、误信息、代码漏洞），对更广泛的攻击面尚未充分验证。

---

## 602. Compaction Laundering: Provenance Destruction in Context-Compacting Coding Agents

**arXiv ID:** 2610.29869 | [PDF](https://arxiv.org/pdf/2610.29869v1)

**作者:** Arshan Dabirsiaghi `[一作]` `[通讯]`, Arshan Dabirsiaghi

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对开源编码代理在长时间运行时必须进行上下文压缩（compaction）时的序列化方式，发现若将用户/工具角色信息以文本标签平铺后直接传递给总结模型，原始来源会被抹去，攻击者可以在代理读取的文件或网页中植入带有伪造标签的行，使其在总结后被误认为是真正的用户指令，从而在代理内执行恶意代码。实验验证了该攻击在五种平铺式 harness 上可达 87–100% 的“洗白”率，并且在实际代理上能够成功触发代码执行，执行率随上下文窗口大小而下降但仍有 20–40% 的成功率。

**💡 创新点**

创新点在于：①首次量化展示平铺式序列化导致的内存污染风险；②在公开的开源编码代理上实现完整的从攻击植入、压缩洗白、执行到结果的端到端实证；③提出并验证了一种基于代码层面的“来源对比”检查，可捕获 77% 的洗白内容，同时保持 2% 的误删率；④与现有提示注入规则、分类器防御进行对比，显示仅靠提示规则不足。

**🔧 技术方法**

主要技术包括：对各主流 harness（pi, opencode, Cline, Aider, OpenHands 等）源码进行静态分析和动态注入；构造带伪造标签的攻击文本；利用真实 LLM（GPT-5.6、Anthropic 等）作为总结器；在 Docker 沙箱内执行攻击脚本并通过 canary 文件监测执行；使用 LLM 判定洗白与摘要重要度（salience）；实现代码层级的 provenance‑contrast 检查。

**📊 数据集**

数据集：使用公开的开源代理代码库及其原始文档（README、工具输出、网页等）作为攻击载体；实验中生成约 20 条不同上下文长度的对照会话；在每个 harness 上多次（≈3–5 次）运行并记录成功率；E15 的实时演示使用 pi TUI 真实运行，配合自建 HTTP 服务器提供攻击文件。

**📈 对比分析**

比较方法：对比五种平铺式 harness 与四种保持角色类型的 harness（Codex, Gemini CLI, Qwen Code, Continue）在同一攻击下的洗白率；在相同模型与窗口大小下测量执行率；对比 prompt 规则（Gemini CLI 内置规则）、已公开的分类器防御与本文提出的代码层防御。性能表现：洗白率 87–100%（平铺）vs 0%（角色型）；执行率在 20–40% 之间，单窗口可达 100%；代码层防御 77% 捕获率，误删率 2%。

**⚠️ 局限性**

限制：①实验主要针对 GPT 系列模型，非 GPT 模型的执行结果未充分验证；②使用的攻击文本为通用模板，针对具体项目的定制化攻击可能更具效能；③代码层防御未经过自适应攻击测试，可能被巧妙伪造标签绕过；④Salience 与执行关联仅在单模型内测，缺乏跨模型验证；⑤部分实验规模有限（n=3–5），统计方差较大。

---

## 603. Linear Self-Symmetries of Inverse-Based S-boxes: Generic Rigidity and Applications

**arXiv ID:** 2610.29837 | [PDF](https://arxiv.org/pdf/2610.29837v1)

**作者:** Zheng Zhang `[一作]` (Towson University), Na Zhang `[通讯]` (Towson University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了有限域反演函数在外层线性变换作用下的线性自同构（自对称性）并量化了保留此对称性的概率。

**💡 创新点**

首次给出该概率的精确二次指数表达式，证明其主项由扩张度的最小素因子决定，揭示了反演对称性在不同维度下的衰减规律。

**🔧 技术方法**

利用有限域反演的两边线性等价分类、半线性共轭与中心化子估计、以及群代数和模块论方法，推导出上、下界。

**📊 数据集**

论文不依赖实验数据，而是通过理论计数和群论推导，验证了AES、ARIA和Vision等实例的刚性。

**📈 对比分析**

通过与已知S‑box的实际结构比较（如AES的线性部分满足两系数刚性判据），证明理论界限在实际设计中的适用性；性能表现为对称性消失速度远快于经验猜测。

**⚠️ 局限性**

局限在于仅处理外层单侧线性变换，未涵盖双侧线性变换、非线性或仿射扰动的更一般情况；此外，仅给出了上、下界的渐近形式，对中小维度的精确概率仍需枚举。

---

## 604. Early versus Late RGB-Infrared Fusion for Counter-UAV Detection: A Comparative Study

**arXiv ID:** 2610.29828 | [PDF](https://arxiv.org/pdf/2610.29828v1)

**作者:** Alberto De Zan `[一作]` (University of Udine), Ivan Scagnetto `[通讯]` (University of Udine)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文设计并实现了两种 RGB–IR 融合策略——像素级早期融合与决策级晚期融合，并将其应用于 YOLOv8s 目标检测框架，验证其在不同距离、背景和光照条件下对无人机的检测性能。

**💡 创新点**

创新点包括：① 采用 RGT（三通道：红绿热）融合方式，简化通道数并兼容 YOLO；② 提出基于置信度加权平均和固定距离阈值的决策级融合算法，能够在传感器失配时保持鲁棒性；③ 对两种融合方法在真实环境中进行系统性对比，阐明其优劣与适用场景。

**🔧 技术方法**

技术方法：YOLOv8s 目标检测；像素级图像合成（通道堆叠与热图替换）；数据增强与对齐预处理；决策级加权平均融合算法；实验设置包括不同距离、背景、时间段与放大倍率的多场景测试。

**📊 数据集**

使用的 dataset：400 张对齐的 RGB–IR 实景图像（300 张训练，100 张验证），涵盖 5 m、25 m、60 m 距离、不同高度、不同光照与背景（树木、天空）等多变条件。

**📈 对比分析**

比较方法：使用 Precision、Recall、mAP@50、mAP@50‑95 与 F1‑score 评估。结果表明，早期融合在 mAP@50‑95 与召回率上略优于晚期融合；晚期融合在复杂或传感器失配环境下仍能保持竞争力。整体来看，早期融合在多场景下提供更高的整体性能。

**⚠️ 局限性**

局限性：① 早期融合需精确对齐且训练模型，部署成本较高；② 晚期融合需双模型推理，推理时间与计算量翻倍；③ 数据集规模有限，未覆盖更复杂的多传感器环境；④ 对实时性能与低功耗设备的适配仍需进一步研究。

---

## 605. Stochastic Nonlinearities Improve Uncertainty Estimation

**arXiv ID:** 2610.29875 | [PDF](https://arxiv.org/pdf/2610.29875v1)

**作者:** Mohammad Sarhangzadeh `[一作]` (Bilkent University), Doruk Oner `[通讯]` (Bilkent University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种在单一模型中通过随机切换激活函数产生预测多样性，从而实现无额外开销的置信度估计；

**💡 创新点**

创新点在于把随机激活切换视为信息保留型的预测多样化方法，避免特征掩蔽导致的信息丢失，且无需多模型或额外头部；

**🔧 技术方法**

技术包括节点/通道/层级随机激活切换（ReLU/SiLU或ReLU/Tanh等组合）、多次前向传播求均值与方差、IDR分析验证信息保持；

**📊 数据集**

使用了多种视觉任务的数据集：道路与医学分割（Massachusetts Roads、DRIVE、CREMI、LiTS 等）、图像分类（CIFAR‑10、MiniImageNet、BloodMNIST、PathMNIST）以及跨域 OOD 数据；

**📈 对比分析**

与 MC‑Dropout、DDU、TTA、Masksembles、Deep Ensembles 等基线比较，实验显示该方法在保持或略优的任务准确率的同时，在 ECE、Brier、OOD AUC、AUPR 等指标上取得与或超过传统集成/后置方法的性能；

**⚠️ 局限性**

局限性包括：需要多次前向传播导致推理延迟；激活对偶选择对校准与精度的权衡有影响，需手工或经验调优；

---

## 606. Expandable Modality Experts for Medical Vision-Language Learning

**arXiv ID:** 2610.29881 | [PDF](https://arxiv.org/pdf/2610.29881v1)

**作者:** DM BAPPY `[一作]` (Korea University), Sangjeong Ahn `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

设计了一种可扩展的医学视觉‑语言模型，通过模态专家、无配文本掩码预训练与对比学习，实现跨多模态的文本提示分类、线性探测和检索。

**💡 创新点**

创新点在于：①使用多路Transformer与域感知Mixture‑of‑Experts实现模态特定专家，并在冻结原始计算路径的前提下无损扩展新模态；②将无配图像掩码预训练与对比对齐结合，解决无配文本数据限制。

**🔧 技术方法**

技术手段包括多路Transformer、Domain‑Aware MoE、VQ‑KD视觉分词器、掩码图像/文本预训练、CLIP 对比损失、专家冻结扩展策略与双阶段训练管线。

**📊 数据集**

使用 19.4M 未配图像行 + 3.23M 图像‑文本对（七种模态：病理、CT、MRI、超声、X 光、皮肤、内镜），并额外加入 88k 眼底对、19k OCT 对，评测涵盖 11 个零射分类、10 个线性探测、4 个跨模态检索基准。

**📈 对比分析**

与 BMC、BMCA、ConceptCLIP、UniMed‑CLIP、MerMED‑FM 等基线在相同评测集上对比；在 24/25 个基准中取得最佳或同等性能，平均零射分类准确率 0.674、线性探测 0.856、检索召回率 20.70%；扩展实验中保持已有任务无显著损失。

**⚠️ 局限性**

局限性包括：模型规模和训练成本高，扩展只在给定模态标签且预处理保持不变时才保证不变；评测覆盖有限，未进行临床前瞻验证；在某些新模态任务中最高性能未必达到最优。

---

## 607. RangeNav: Task-Aligned LiDAR Range-Image Pretraining for Robot Navigation

**arXiv ID:** 2610.29845 | [PDF](https://arxiv.org/pdf/2610.29845v1)

**作者:** Samuel Foo Enze `[一作]` (ETH Zurich), Cesar Cadena `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

论文未提供具体内容，因此无法确定做了什么。

**💡 创新点**

论文未提供具体内容，因此无法确定创新点。

**🔧 技术方法**

论文未提供具体内容，因此无法确定使用了什么技术。

**📊 数据集**

论文未提供具体内容，因此无法确定使用了什么数据集。

**📈 对比分析**

论文未提供具体内容，因此无法确定比较的方法及性能。

**⚠️ 局限性**

论文未提供具体内容，因此无法确定限制因素。

---

## 608. S-EDL: Eliciting Self-Evidence from Sequence Likelihoods for Semantic Calibration of LLMs

**arXiv ID:** 2610.29832 | [PDF](https://arxiv.org/pdf/2610.29832v1)

**作者:** Yawei Li `[一作]` (LMU Munich), Mina Rezaei `[通讯]` (LMU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于自证据的 Evidential Deep Learning (S-EDL)，通过自模型的序列概率构造可微的 Dirichlet 先验，以实现开放式生成任务的语义校准。

**💡 创新点**

创新点在于：① 将 EDL 迁移到动态语义类别空间，摆脱固定答案集的限制；② 通过蒙特卡洛采样的序列概率权重化自证据，产生可微的语义分布估计；③ 在训练时同时兼顾下一步预测和语义校准损失，形成全局概率重分配机制。

**🔧 技术方法**

使用技术包括：蒙特卡洛采样、可微的 Dirichlet 参数化、Evidential Deep Learning 训练框架、find‑first 聚类算法以加速语义分组、以及传统的 LoRA 微调。

**📊 数据集**

在四个开放式 STEM 数据集上进行实验：GSM8K、MATH‑500、ChemistryQA 和 FormulaReasoning。

**📈 对比分析**

与传统 SFT、token‑level 校准（Label Smoothing、RCFT）、贝叶斯集成（BLoB）以及序列级别奖励（Dr.GRPO）对比，S‑EDL 在 smECE（均值约 10.5–9.4）显著低于其他方法，Pass@1 与 Semantic Accuracy 同时提升，表明兼具准确性和校准性能。

**⚠️ 局限性**

局限性包括：① 需要对每个 prompt 进行多次采样与聚类，训练开销高于普通 SFT；② 依赖外部 verifier/LLM‑judge 进行语义聚类，易受噪声与内存限制；③ 尚未验证在长上下文、代码生成等需要复杂语义判定的任务上的表现。

---

## 609. GeoMem-VLA: Internalizing 3D Scene Understanding into Temporal Memory for Vision-Language-Action Models

**arXiv ID:** 2610.29866 | [PDF](https://arxiv.org/pdf/2610.29866v1)

**作者:** Qinfan Zhang `[一作]` (Beijing University of Aeronautics and Astronautics), Xiaozhu Ju `[通讯]` (X-Humanoid)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出GeoMem-VLA框架，利用冻结的VGGT几何教师对可学习的几何令牌进行监督，并通过事件感知检索与压缩机制构建时序几何记忆，进而提升视觉-语言-动作模型的空间决策与长期规划能力

**💡 创新点**

创新点在于将几何监督与时序检索相结合，形成紧凑可更新的几何抽象；同时通过事件感知的内存压缩避免冗余，并在推理阶段无需显式几何重建即可获得几何上下文

**🔧 技术方法**

技术包括冻结的VGGT几何教师、GR00T式流匹配动作专家、可学习的几何令牌、事件感知检索与相似性压缩、以及基于流匹配的几何监督目标

**📊 数据集**

使用的数据集为LIBERO（四个10-task套件）、RoboTwin 2.0（50个双手任务）以及两台真实机器人平台的实地实验任务

**📈 对比分析**

与Spatial Forcing、MemoryVLA、HyVLA-0.5、StarVLA-GR00T等方法对比，GeoMem-VLA在LIBERO上平均98.9%（对比98.5%），RoboTwin Clean/Randomized上分别为91.10%/91.24%（对比约91%），显示出显著的性能提升，且消除几何监督或记忆机制会导致显著下降

**⚠️ 局限性**

局限性包括仅在机器人抓取/操作任务中验证，缺乏人机交互与安全评估；模型依赖预训练的VLM和几何教师，部署时对相机与传感器的依赖较高；且实验仅覆盖有限的任务类型，泛化能力尚待进一步验证

---

## 610. UserSWE: Evaluating Coding Agents in Multi-Turn Conversations with Dynamic Users

**arXiv ID:** 2610.29822 | [PDF](https://arxiv.org/pdf/2610.29822v1)

**作者:** Zhan-Yu Wu `[一作]` (Tencent Hunyuan), Xinting Hu `[通讯]` (Tencent Hunyuan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一个面向多轮交互的编码代理评测基准，并在此基准上对十一种代理进行了系统实验。

**💡 创新点**

创新点在于：①将仓库历史和现有单轮基准转换为多轮任务；②基于真实开发会话构建动态人物模拟器；③使用跨模型判别与复审机制结合执行测试的评测框架。

**🔧 技术方法**

使用的技术包括：任务构建流水线（挖掘问题链和转化单轮任务）、人物转移矩阵（从内部编码助手会话提取）、多轮交互评测框架（Judge-Reviewer双模型判别、等价通过处理）、以及基于容器执行的测试。

**📊 数据集**

使用的数据集：来自公开仓库的两个问题链（22条），以及从已发布单轮基准转换的 50 条任务，总计 72 条任务；人物模拟器的参数来自内部商业编码助手的 13,530 条会话。

**📈 对比分析**

比较方法：对每个任务记录每轮是否通过 fail‑to‑pass 与 pass‑to‑pass 测试，计算每轮的通过率和长度折扣通过率 (LDPR)，并统计平均使用轮数；实验结果显示，代理在不同轮数下的排名会变动，Claude Fable 5.1 在一轮内最高，GPT‑6‑Astra 在第八轮最高，单轮评测无法揭示这些交互动态。

**⚠️ 局限性**

局限性：①规模受限，执行每轮测试成本高，当前仅 72 条任务；②视角局限，仅覆盖一次维护周期（最多两个问题），不包含长期开发；③人物模型仅发布聚合统计，无法重现完整会话，可能限制进一步研究。

---

## 611. UlaCache: Unified Logical Addressing for Sparse LLM Serving

**arXiv ID:** 2610.29820 | [PDF](https://arxiv.org/pdf/2610.29820v1)

**作者:** Isaac Zhu `[一作]` (Scitix.AI), Jason Zhao `[通讯]` (Scitix.AI)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 UlaCache，统一逻辑 KV 地址空间来支持 GPU 与 CPU 之间的 KV 缓存管理，从而提升长上下文 LLM 的推理吞吐。

**💡 创新点**

创新点在于：① 统一逻辑地址抽象，解耦物理存放；② 并行桶化 LRU (PB‑LRU) 以并行方式完成替换与映射更新；③ 在 TP 模式下使用本地 GPU 缓冲区重用 KV。

**🔧 技术方法**

使用技术包括：GPU/CPU 统一逻辑地址映射；PB‑LRU 级联位图；CUDA IPC 与多流预取；分布式索引器；推理框架 SGLang；稀疏注意力。

**📊 数据集**

使用 GLM‑5.3 语言模型，对 Multi‑LexSum 长文本请求（64K–128K token 输入，2,048 生成 token）。

**📈 对比分析**

与 SGLang（GPU KV）和 HiSparse（CPU KV）基线对比，匹配 GPU 与 CPU 预算；在 8×H200 上 DP/TP 分别获得 1.16× /1.70× 的峰值吞吐提升；在 8×B300 上 DP/TP 分别获得 1.73× /2.51× 的提升。

**⚠️ 局限性**

局限性包括：PB‑LRU 近似 LRU 可能导致极端场景下缓存命中率略低；对极大 GPU 或 CPU 规模的伸缩性尚未全面评估；主要评测集中在 GLM‑5.3，未验证跨模型适用性。

---

## 612. FrontierCS 2: A Benchmark for Computer Science R&D

**arXiv ID:** 2610.29932 | [PDF](https://arxiv.org/pdf/2610.29932v1)

**作者:** Qiuyang Mang `[一作]`, Alvin Cheung `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为 FrontierCS 2 的开源基准，包含 48 个跨 13 个计算机科学领域的删减式任务，让 LLM 代理在实际代码库中改写并超越原作者代码。

**💡 创新点**

创新点在于通过删除论文贡献的代码作为人类参考，模拟真实科研工作；以及提出 FrontierCS Capabilities Index (FCI) 与 Difficulty Index (FDI) 把代理与任务放在同一尺度上评估。

**🔧 技术方法**

利用大型语言模型 (LLM) 代理与自定义 harness，基于评测器（judge）提供的开发套件评分和最终持有集评估，并使用 item‑response 模型估算 FCI/FDI。

**📊 数据集**

数据集来自 48 个公开论文的代码仓库及其原始工作负载，额外保持 100+ 私有任务待发布。

**📈 对比分析**

与 12 名代理（如 Claude Opus 5.5、GPT‑6 Astra 等）对比，最佳代理在 58% 任务上通过，且通过率与平均成本呈现一定相关性，FCI 进一步量化其能力。

**⚠️ 局限性**

局限性包括：评估仅关注与原作者代码的对比，未考虑实现的附加成本（如能耗、代码质量）；任务难度分布不均，某些任务仍无代理通过；同时 FCI/FDI 仅基于通过/分数，可能忽略部分细微改进。

---

## 613. CausalAlign: Causal Alignment of Video Generation Models for Consistent Ego-Motion

**arXiv ID:** 2610.29915 | [PDF](https://arxiv.org/pdf/2610.29915v1)

**作者:** Kai Zhu `[一作]`, Yiwu Zhong `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 CausalAlign 框架，针对视频扩散模型在生成过程中相机视角一致性差的问题，进行因果诊断并在遗忘位置使用 delta‑token 对齐来增强 ego‑motion 表示。

**💡 创新点**

创新点在于首次通过线性探测与因果干预联合识别几何状态遗忘位置，并利用 3D 基础模型生成的 delta‑token 作为监督，直接对齐 VGM 的内部表示，实现了无额外计算开销的相机轨迹一致性提升。

**🔧 技术方法**

主要技术包括线性可解性探测、因果干预、delta‑token 学习与对齐、LoRA 微调、以及 3D 基础模型（如 LingBot‑Map）的几何监督。

**📊 数据集**

实验数据集为 DL3DV 与 RealEstate10K，均提供相机姿态注解，用于训练与评估。

**📈 对比分析**

与基线 Wan2.2、外部 3D 缓存、偏好对齐以及直接特征对齐方法对比，CausalAlign 在 7 项 ego‑motion 评估指标中提升 6 项，重投影误差降低 22.6%，轨迹平滑度提升 23.9%，在 RoboCasa365 移动操控任务中的平均成功率提升 10.7%。

**⚠️ 局限性**

局限性在于仍需依赖预训练的 3D 基础模型进行监督，难以在无标注或多视角复杂场景下直接应用；此外，方法在处理高度动态或非静态场景时的鲁棒性尚未完全验证。

---

## 614. Asking a Multimodal LLM Whether Someone Fell: Zero-Shot Screening of Low-Resolution Thermal Video in a Laboratory and an Occupied Home

**arXiv ID:** 2610.29913 | [PDF](https://arxiv.org/pdf/2610.29913v1)

**作者:** Marcos Junca `[一作]`, Mauro Junca `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估生产多模态LLM在低分辨率热成像跌倒筛查中的零样本性能，直接将热视频+音频交给 Gemini 3.8 Flash 进行一次性判断并给出可解释原因。

**💡 创新点**

在真实家庭环境中首次验证LLM作为决策阶段，系统性探讨提示句、分辨率、色彩映射等因素对敏感度与误报的影响，并给出可复现的实验流程。

**🔧 技术方法**

使用 Gemini 3.8 Flash（多模态大型语言模型）、热成像相机（32×24/160×120热阵列）、麦克风音频、基于运动门的事件触发机制及视频渲染流程。

**📊 数据集**

公开的 eHomeSeniors 低分辨率热成像数据（32×24）以及本研究在占用房间连续录制的 160×120 热+音频记录。

**📈 对比分析**

与两种传统跌倒检测器（下降阈值法与梯度提升分类器）在实验室 held‑out 受试者与家庭连续录制中对比；实验室敏感度约 70%（误报 0%），家庭误报极低（<0.5/天），可见跌倒召回率约 30%。

**⚠️ 局限性**

受试者为年轻人，实验室跌倒与真实老年人差异；LLM对“离视角”提示句极度敏感，导致召回率下降；分辨率下降显著降低召回率；温度 0 下模型非确定性影响可复现性。

---

## 615. Investigating the Impact of Learner Errors on Social Bias in Large Language Models

**arXiv ID:** 2610.29896 | [PDF](https://arxiv.org/pdf/2610.29896v1)

**作者:** Federico Marcuzzi `[一作]` (INSAIT, Sofia University St. Kliment Ohridski), Alla Rozovskays `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了非母语使用者的语法错误如何放大大型语言模型的社会偏见，并提出了错误感知微调方法以降低该偏见。

**💡 创新点**

首次引入可控的合成语法错误注入框架评估偏见，并发现前置词错误对偏见放大作用最大；同时证明错误感知微调能显著缓解这一效应。

**🔧 技术方法**

采用规则基语法错误生成器、偏见评估框架、对比实验、LoRA微调以及统计显著性检验等技术。

**📊 数据集**

使用了三大主流偏见基准（Winogender、WinoBias、决策任务）以及学习者语料库（FCE、NUCLE、W&I+LOCNESS）来构造错误分布。

**📈 对比分析**

在不同错误率、错误类型和错误位置下，对五个开源模型进行系统评估，结果显示错误率升高会导致偏见显著放大，而错误感知微调可显著降低偏见增幅。

**⚠️ 局限性**

局限性包括：合成错误可能不足以代表真实非母语写作；错误类型仅限于限定词、介词、动词和拼写错误；评估仅基于模板化基准和性别/种族/宗教维度，未覆盖交叉性与对话场景；并且仅对开源模型和单一微调模型进行了验证。

---

## 616. How Do DiTs Denoise? An Aggregator-Predictor View of Attention and MLPs

**arXiv ID:** 2610.29854 | [PDF](https://arxiv.org/pdf/2610.29854v1)

**作者:** Gyeonghun Kang `[一作]` (Duke University), Xiang Cheng `[通讯]` (Duke University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出将扩散变压器（DiT）视为输入自适应的局部去噪器，并给出了聚合器-预测器的理论框架；

**💡 创新点**

创新点在于将注意力的 QK、VO 以及 MLP 分别对应为空间上下文聚合、特征变换与补丁预测，并通过高斯马尔可夫随机场（GMRF）证明局部去噪可以逼近全局贝叶斯去噪；

**🔧 技术方法**

技术上采用统计理论（局部贝叶斯去噪、PLD 解析）、Transformer 结构拆解、共享记忆补丁预测器、敏感度上下文分析以及跨域组件移植实验；

**📊 数据集**

使用的图像数据集包括 CelebA、AAHQ、FFHQ、AFHQ 和 STL-10，实验在 64×64 条件下进行；

**📈 对比分析**

对比方法包括传统 MLP、共享记忆 LPLD、SCA 与 SMLP 预测器、以及不同组件组合的迁移；性能上共享记忆预测器在相同参数预算下能达到或超过原始 MLP 的去噪误差与 FID，组件移植实验显示 QK 兼容性与 VO 反向代价与空间相似度相关；

**⚠️ 局限性**

局限性主要是实验规模有限（仅 64×64 无条件模型）、仅使用单一随机种子、未验证在更大尺寸或条件模型中的适用性，以及组件功能未在更广泛的迁移场景中彻底确认。

---

## 617. Revealing Hidden Failure Modes in Chest X-ray Classification via Spectral Domain Analysis

**arXiv ID:** 2610.29830 | [PDF](https://arxiv.org/pdf/2610.29830v1)

**作者:** Samuel Halimi `[一作]`, Arnaud Abreu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种基于频域谱分析的无监督失效模式发现框架，对胸部 X 光图像进行 Radially Averaged Power Spectrum (RAPS) 编码，并通过聚类分离性能低下的数据子集。

**💡 创新点**

创新点在于：①用频域特征而非语义元数据或模型嵌入来捕捉信号级别的变化；②通过无监督聚类发现“盲点”，实现不依赖真实错误标注的性能预测；③对比常用距离度量，证明相关距离最能反映模型性能差异。

**🔧 技术方法**

核心技术包括：2D 离散傅里叶变换、RAPS 提取、相关距离度量、K‑medoids 与层次聚类（HAC）、DenseNet‑121 等深度分类器。

**📊 数据集**

使用四大公开胸部 X 光数据集：CheXpert（训练集）、MIMIC‑CXR、PadChest、NIH ChestX‑ray14（外部测试），聚焦 Atelectasis、Consolidation、Pleural Effusion 三种单病种。

**📈 对比分析**

通过与随机、伪随机基线对比，评估聚类对 AUROC 的方差；与两种 OOD 检测方法对照，HAC‑RAPS 在外域数据上平均提升 Δ≈0.103，显著低于基线的性能下降。

**⚠️ 局限性**

局限性：需对图像尺寸做统一裁剪，可能忽略极端分辨率差异；聚类结果缺乏直接因果解释；目前仅验证在胸部 X 光任务，对其他模态或更大尺度迁移的适用性尚未评估。

---

## 618. Towards English - Mizo Legal Machine Translation: Corpus Creation, Analysis, and Benchmarks

**arXiv ID:** 2610.29943 | [PDF](https://arxiv.org/pdf/2610.29943v1)

**作者:** Telem Joyson Singh `[一作]` (Indian Institute of Technology Guwahati), Priyankoo Sarmah `[通讯]` (Indian Institute of Technology Guwahati)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了印度法庭判决的英语‑米佐语平行语料库，首次提供法学领域的高质量双语数据。

**💡 创新点**

发现米佐语翻译中法律术语的选择性借用模式，并指出音调符号不一致是关键问题。

**🔧 技术方法**

使用 NLLB‑200 1.3B 模型进行领域微调，并在专门设计的法律测试集上评估。

**📊 数据集**

数据集包含 13,168 对句子，来源于加尔哈提高等法院和最高法院的判决。

**📈 对比分析**

与基线相比，微调后 BLEU 提升 7.16 点，chrF++ 提升 7.83 点，TER 降低 12.13 点，表现显著提升。

**⚠️ 局限性**

仍存在过度保留英语词汇、词形变体缺失、音调符号不统一等局限，需进一步改进术语处理和预处理方案。

---

## 619. ToM2Vec: Continuous Theory-of-Mind Tokens for Multi-Agent Coordination

**arXiv ID:** 2610.29934 | [PDF](https://arxiv.org/pdf/2610.29934v1)

**作者:** Nithin Sivakumaran `[一作]` (UNC Chapel Hill), Elias Stengel-Eskin `[通讯]` (University of Texas at Austin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过自我对弈训练，将对手的思维压缩为连续的潜在向量，从而让代理在推理时使用隐式的 Theory of Mind (ToM) 表示；

**💡 创新点**

创新点在于：①提出在潜在空间而非文本空间中学习 ToM 表示；②使用自监督的连续潜在令牌来捕捉多种可能的伙伴行为；③证明该方法对未见过的伙伴模型具有良好的迁移性；

**🔧 技术方法**

技术包括：自我对弈生成伙伴思考序列、截取隐藏层状态作为潜在令牌、基于余弦相似度的自监督 ToM 损失、在推理时将预测的潜在令牌作为条件输入，使用大型语言模型 Qwen3‑8B 作为基础；

**📊 数据集**

数据集：五种多智能体环境——Overcooked‑AI（3布局）、多方谈判、Decrypto、Hanabi、Collab Escape/Capture，所有任务均在自定义或公开的 LLM 协作基准上进行；

**📈 对比分析**

与多种基线比较（未训练模型、提示 ToM、语言 ToM、Self‑Consistency、SimToM、Adaptive ToM、Oracle 伙伴），在所有任务中均取得显著提升：Overcooked 平均分提升约12点；谈判成功率从62%升至71%；Decrypto 解码器/拦截器胜率分别提升约4–6%；Hanabi 得分提升约2.4点；协作捕获/逃脱成功率分别提升至100%/92%；

**⚠️ 局限性**

局限性包括：①训练时需要对手的完整思考序列，且双方共享同一模型表示空间；②需要额外的训练阶段，无法像提示式方法一样即时推理；③实验仅限于英语环境，未检验多语言适应性；

---

## 620. ViBMR: A Large-Scale Multi-Sequence Brain MRI Dataset and Comprehensive Evaluation for Vietnamese Radiology Report Generation

**arXiv ID:** 2610.29905 | [PDF](https://arxiv.org/pdf/2610.29905v1)

**作者:** Dai Lam Le `[一作]` (Hanoi University of Science and Technology), Phi Le Nguyen `[通讯]` (Hanoi University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了ViBMR数据集并系统评估多序列脑MRI报告生成方法

**💡 创新点**

首次提供大规模多序列脑MRI与越南临床报告的数据，揭示跨语言/跨数据集泛化差距

**🔧 技术方法**

采用分割预训练的多序列编码器（M2FTrans/mmFormer）与轻量融合策略以及Qwen3-4B-Instruct生成模型

**📊 数据集**

使用新收集的3,649例、11,875序列的ViBMR数据集（含T1、T2、FLAIR、T1c）

**📈 对比分析**

与现有英文基准相比，M2FTrans/ mmFormer在BLEU、ROUGE、BERTScore及临床F1上显著优于单序列模型，且轻量级融合优于复杂注意力融合，性能提升约10–15分

**⚠️ 局限性**

局限包括缺乏多机构、多语言验证、模型对缺失序列的鲁棒性有限及仍未解决细粒度诊断细节的生成

---

## 621. A Sharp Dynamic Haussler Lemma

**arXiv ID:** 2610.29898 | [PDF](https://arxiv.org/pdf/2610.29898v1)

**作者:** Shihao Shao `[一作]` (Nanyang Technological University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在在线预测（动态）环境下，对有限取值范围（[0,1]）的函数类进行全局动态覆盖的理论上限与下限，给出了统一的、对任何 1≤p<∞ 的 ℓ_p 误差度量下的覆盖数上界与下界，证明其指数为 (C/ε)^p d，其中 d 为序列化的 fat‑shattering 维数。

**💡 创新点**

创新点包括：
1) 将 Haussler 的覆盖引理从静态、二值函数类推广到在线、实值函数类并实现了与维数匹配的全局动态覆盖；
2) 证明了对应的下界，展示了指数 pd 的最优性；
3) 通过 SOA 样式的预测与预定的排名递减版本空间更新、共享父节点路由等技术，构造了一个单一有限的确定性专家族，满足所有输入序列与目标函数的全局覆盖；
4) 解决了先前提出的多项关于全局动态 Haussler 问题和 Rakhlin‑Sridharan‑Tewari 的 ℓ_2 覆盖问题。

**🔧 技术方法**

使用的主要技术包括：
- 序列化 fat‑shattering 维数的定义与属性；
- SOA（Optimal Algorithm）风格的预测与“纠正调度”实现排名递减的版本空间更新；
- 多池随机抽样与共享父节点路由，平衡对已激活目标的后续预测需求；
- 通过完整的计数与卷积恒等式控制专家数；
- 通过概率不等式（Chernoff、Markov、Martingale）在自适应输入下保证高概率下的误差控制。

**📊 数据集**

本文完全基于理论分析，未使用任何具体数据集；所有结果均为对任意函数类的普适上界与下界。

**📈 对比分析**

与现有方法相比，本文提供了与维数 d 与误差 ε 以及 p 成指数匹配的最优上界和下界，说明在 ℓ_p 误差下的全局动态覆盖数可控制在 (C/ε)^p d；在 p=1 时退化为 (C/ε)^d，解决了之前只给出分数覆盖或特定实例树覆盖的局限；在 p=2 时给出了完整的 ℓ_2 覆盖数的最佳上界。

**⚠️ 局限性**

局限性包括：
- 构造是理论性的，未给出高效的算法或实现方式来生成专家族；
- 结果只对有限取值区间 [0,1] 的实值函数类成立，未考虑更一般的范围或更复杂的损失函数；
- 常数 C 可能较大，实际覆盖数可能远大于理论下界；
- 证明依赖于对序列化 fat‑shattering 维数的准确估计，实际应用中若维数难以估计，效果可能受限。

---

## 622. HiddenLens: A Multi-Layer LLM Framework for Detecting and Resolving Hidden Assumptions in Software Requirements

**arXiv ID:** 2610.29996 | [PDF](https://arxiv.org/pdf/2610.29996v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 623. Attn-Forcing: Learning Full-Context Attention Proxies for Streaming Video Generation

**arXiv ID:** 2610.30036 | [PDF](https://arxiv.org/pdf/2610.30036v1)

**作者:** Junyao Gao `[一作]` (Tongji University), Cairong Zhao `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的框架Attn-Forcing，用于在有限的内存和计算条件下保留全上下文注意力贡献，以提高流媒体视频生成的质量。

**💡 创新点**

创新点在于引入可学习的注意力代理，持续整合被驱逐的视觉上下文，并通过注意力匹配损失进一步提高其对历史注意力响应的近似。

**🔧 技术方法**

使用了分布匹配蒸馏（DMD）训练方法和注意力匹配损失，结合可学习的视觉潜变量作为注意力代理。

**📊 数据集**

使用了VidProM数据集进行文本提示采样，并在多个视频生成基准上进行评估。

**📈 对比分析**

与现有的基线方法相比，Attn-Forcing在短视频（5秒）和长视频（30秒和60秒）生成中均表现出色，尤其在长视频生成中显示出更明显的优势，达到了最先进的性能。

**⚠️ 局限性**

限制在于尽管引入了注意力代理和匹配损失，但在生成过程中仍可能存在近似误差，尤其是在长时间生成时可能导致模糊现象。

---

## 624. Mitigating Model Collapse through Dreaming Learning

**arXiv ID:** 2610.30009 | [PDF](https://arxiv.org/pdf/2610.30009v1)

**作者:** Matteo Benati `[一作]` (Sapienza University of Rome), Vittorio Loreto `[通讯]` (Sapienza University of Rome)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在自回归生成模型上进行递归自训练时，研究模型在温度低于1时会出现谱收敛、谱间隙变窄导致多样性下降的崩塌现象，并提出在高温下生成样本做正则化的Dreaming Learning以抑制谱漂移，维持转移算子非零元素，防止子主特征值逼近单位圆，进而保持模型多样性。

**💡 创新点**

创新点在于通过谱视角揭示自训练导致的模型崩塌机制，并提出利用高温生成样本的Dreaming Learning正则化来固定子主特征值，直接从特征值谱角度监控并抑制多样性损失。

**🔧 技术方法**

使用特征值谱分析、行归一化的锐化与平滑算子、交叉熵损失及其高温正则化项、Markov链、LSTM与GPT‑2语言模型的生成与再训练循环。

**📊 数据集**

实验数据集包括人工设定的10状态Markov链、公开域小语料库（用于LSTM）以及OpenWebText与WikiText‑2（用于GPT‑2）。

**📈 对比分析**

与传统累积（保留一部分真实数据）对比，vanilla递归自训练在温度<1时验证损失与困惑度显著恶化，而Dreaming Learning在所有规模下均实现了更慢的性能退化，验证集KL散度和perplexity下降幅度超过两位数或约一半。

**⚠️ 局限性**

局限包括谱稳定性不一定代表正确模型（学习链可能混合更快）；温度取值仅在0.7、0.8、1.0三点验证，未细粒度扫描；假设可完全恢复转移算子；高温与正则强度需经验调参，且在GPT‑2规模下生成成本显著增加。

---

## 625. SCDM: Scalable Causal Discovery in Nonlinear Temporal Systems with Meta-Learning

**arXiv ID:** 2610.29992 | [PDF](https://arxiv.org/pdf/2610.29992v1)

**作者:** Jingbo Wang `[一作]` (University of Tennessee), Shaogang Ren `[通讯]` (University of Tennessee)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种共享参数的元学习框架SCDM，用于在高维非线性时间序列中可扩展地进行因果发现。

**💡 创新点**

创新点在于将每个目标变量视为任务，在元学习过程中仅更新部分任务，从而在不需逐目标优化的前提下聚合结构证据；并提供了阈值无关的后训练图读取与有限误差恢复理论。

**🔧 技术方法**

核心技术包括第一阶MAML元学习、共享非线性时间预测器、共享因果强度矩阵、支持/查询子任务训练、以及基于扰动或结构权重的图读取。

**📊 数据集**

使用了多种数据集：受控合成数据（NC8、模拟EEG、模拟fMRI）、大规模voxel级fMRI模拟（1000维）、以及真实交通网络LargeST（3000维）。

**📈 对比分析**

与UnCLe、PCMCI、TCDF、GVAR、cMLP、CUTS+等基线比较；在低/中等维度任务上表现与最强基线相当，在1000/3000维任务上取得更优的运行时间–准确率折中，且在固定预算下实现更高的AUPRC提升。

**⚠️ 局限性**

局限性包括假设无隐藏共因子、正确的滞后阶、无即时效应；在存在隐藏驱动或高度相关源时可能产生误边；且目前仅为观察性方法，对隐藏混杂缺乏完整建模。

---

## 626. Object- and Task-Aware Active Perception for Safe Navigation of Mobile Manipulators in Semi-Static Environments

**arXiv ID:** 2610.29964 | [PDF](https://arxiv.org/pdf/2610.29964v1)

**作者:** Jingxing Qian `[一作]` (University of Toronto Institute for Aerospace Studies), Angela P. Schoellig `[通讯]` (Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出面向移动机械臂的任务感知与对象感知的主动观测框架，利用手持摄像头在导航过程中主动采集不确定或关键区域信息并更新半静态地图。

**💡 创新点**

结合概率下一最佳视角目标，融合地图不确定性、对象变动检测和路径相关任务权重，并将此目标嵌入层级模型预测控制中，实现导航优先下的摄像头姿态优化。

**🔧 技术方法**

采用概率对象级变化检测（POCD）+ 语义分割、基于EDF的稠密地图、近似信息增益式NBV规划、层级任务MPC（HTMPC）以及 ORB‑SLAM3 与 ACADOS 控制框架。

**📊 数据集**

在室内三种静态场景（Easy Loops、Hard L‑Shaped Walls、Hard Office Scene）进行实地测试；并使用 IsaacSim 做受控场景变化仿真；以及多次长周期真实环境变化实验。

**📈 对比分析**

与四种基线（Fixed, Random, Lookahead, NBV‑Map）对比，评估地图精度（精度/召回/FPR）、路径任务相关熵、成功率和安全余量。结果显示本方法在所有指标上显著优于基线，成功率达到100%，地图回忆率提升10‑20%，路径熵显著降低。

**⚠️ 局限性**

仅针对平面移动基座+固定臂配置；依赖 RGB‑D 与前向运动学；在极端遮挡或动态障碍物频繁出现时仍可能错过；对大规模环境的实时性能尚未验证。

---

## 627. The Capacity of Interactive Secure Summation

**arXiv ID:** 2610.29954 | [PDF](https://arxiv.org/pdf/2610.29954v1)

**作者:** Hua Sun `[一作]` `[通讯]` (University of North Texas), Hua Sun (University of North Texas)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文研究了交互式安全求和问题，给出了所有 K≥2、0≤T≤K−2 的容量区域，并证明了最优通信率与随机性分配的取值。

**💡 创新点**

创新点在于：①首次完全刻画交互式安全求和的容量区域；②发现最优随机性取值构成一个多面体，仅有 T 个参与方需要随机性；③证明线性方案即可同时实现最优通信率和最优随机性分配；④将结果推广到全广播（每方都需得到求和）和无输入服务器模型。

**🔧 技术方法**

核心技术包括信息理论的极限证明（使用链式法则、Shearer's 引理、协同信息等）、构造基于 Cauchy 矩阵的线性共享噪声方案，以及多面体极点分析。

**📊 数据集**

未使用具体实验数据集，全部为理论分析与构造证明。

**📈 对比分析**

通过对所构造方案的通信量和随机性计数与容量区域下界的对齐，证明其达到最优；在不同模型下对比得到的最优通信率分别为 K(K−1)/(K−T)、K(K−1)/(K−T)+K−1 和 K(K−1)/(K−T)（服务器通信率为 1）。

**⚠️ 局限性**

限制在于：仅适用于非恶意的被动攻击（隐私保护不考虑主动破坏）；T 必须满足 0≤T≤K−2；方案虽然线性，但对实际大规模系统实现的具体效率未做实验验证。

---

## 628. SWE-DIVE: Multi-Turn Coding Agent Evaluation with Dynamic User Simulation

**arXiv ID:** 2610.30047 | [PDF](https://arxiv.org/pdf/2610.30047v1)

**作者:** Lorenz Brehme `[一作]` (University of Innsbruck), Ruth Breu `[通讯]` (University of Innsbruck)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SWE-DIVE，一个用于多轮交互式评估编码代理的新基准框架。

**💡 创新点**

创新点在于引入用户模拟器与多种用户角色（Expert Nitpicker、Vague Requester、Mind Changer）以及对话验证器，能够动态调整交互路径，模拟真实用户需求演变。

**🔧 技术方法**

核心技术包括：LLM 驱动的用户模拟器、对话验证循环、Harbor 容器化执行环境、Gemini-3.6-flash 与 Gemini-2.5-flash-lite 模型。

**📊 数据集**

使用了 SWE-bench Verified、Terminal-Bench 等现有软件工程任务作为测试集，并通过 Harbor 框架集成。

**📈 对比分析**

对比单轮基准，SWE-DIVE 在多轮交互下保持了相近的成功率，但不同角色导致回归率和额外 token 消耗差异显著；Vague Requester 通常引入更多回归和不必要改动。

**⚠️ 局限性**

局限性包括：仅适用于提供 Git diff 的任务；用户模拟器对真实用户行为的贴近度有限；评估样本量小，结果易受高方差影响。

---

## 629. SP-HLM: Sparse-Pose-Driven Motion Generation and Whole-Body Control for Humanoid Loco-Manipulation

**arXiv ID:** 2610.30044 | [PDF](https://arxiv.org/pdf/2610.30044v1)

**作者:** Zhaohui Yang `[一作]` (Shanghai Jiao Tong University), Yao Mu `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 SP-HLM 框架，通过稀疏姿态驱动接口将任务空间的五个关键姿态与隐式全身运动先验结合，实现无缝的高层任务意图与低层全身控制。

**💡 创新点**

创新点在于将空间意图（显式五点姿态）与全身协调（隐式运动先验）分离并补充，构建三模态模块（SP‑VLA、SP‑MGM、SP‑GMT），以及两阶段蒸馏实现实时运动生成。

**🔧 技术方法**

采用视觉‑语言‑动作模型 (SP‑VLA)、压缩扩散式运动生成器 (SP‑MGM) 与变分自编码器驱动的运动追踪策略 (SP‑GMT)，并配合异步动作块推理与 TensorRT 加速。

**📊 数据集**

使用约 190 小时的人类运动数据预训练 VAE，BONES‑SEED 500 与 Filtered CLOT‑500 用于运动追踪评估，八个仿真与四个真实世界 loco‑manipulation 任务作为实验数据集。

**📈 对比分析**

与 SONIC、Humanoid‑GPT、SP‑VLA 直接隐式预测以及 token‑space / joint‑space 接口对比，SP‑GMT 在全局 MPJPE 与 EE 误差上均优于基线；SP‑MGM 在生成速率上提升 1.45×，误差与 jerk 均低；SP‑HLM 在仿真中 52.75% 成功率，真实世界中 20/20 任务的成功率分别为 65%，25%，40% 等，显著优于 token‑space 与 joint‑space 方法。

**⚠️ 局限性**

局限在于手部控制仅通过指尖关节位置指令，缺乏对细腻抓取与手部协同的支持，且在需要精细平衡的任务（如抽屉/垃圾桶开启）表现仍受限。

---

## 630. HGM-Dual: Lineage-Aware Guided Agent Harness Optimization for Long-Horizon Planning

**arXiv ID:** 2610.30042 | [PDF](https://arxiv.org/pdf/2610.30042v1)

**作者:** Sudipta Paul `[一作]` (Samsung Research America), Srinivas Chappidi `[通讯]` (Samsung Research America)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种诊断指导的 HGM-Dual 框架，用于在冻结的 LLM 上对 agent harness（提示、工具、工作流）进行自我改进，专门针对长周期规划任务；

**💡 创新点**

核心创新点包括：1) 结构化的演化上下文（agent evolution context），记录每次编辑的功能、性能及行为变化，帮助编辑器避免重复或无效修改；2) 错误条件双路径子代生成，既保持开放式探索，又针对父代最严重的错误类别进行定向改进；

**🔧 技术方法**

技术实现主要依赖 LLM 生成编辑（GPT‑5.4 作为编辑器）、递归自我改进的 HGM 机制、基于评估结果的错误分类器、以及对演化路径的压缩摘要；

**📊 数据集**

使用 DeepPlanning benchmark 中的两大任务：Travel Planning（需要时间、空间、预算等多维约束）和 Shopping Planning（涉及多项商品属性和优惠券约束），每个任务的训练集为 60 条样本，评估集 120 条；

**📈 对比分析**

与 ADAS、GEPA、HGM、HGM‑CA、HGM‑CF 等基线相比，HGM‑Dual 在 4000 次评估预算下在 Travel 与 Shopping 上分别提升了 8.13% 与 5.84% 的 case accuracy，且在相同或更少的预算下达到相同或更优的最终性能，表现出更高的预算效率；

**⚠️ 局限性**

局限性包括：1) 仍依赖大量评估预算，尽管相对节省但在极大规模任务中仍昂贵；2) 错误分类器为确定性规则，可能无法覆盖所有真实错误；3) 研究主要集中在规划类任务，尚未在更广泛或在线实时环境中验证泛化与长期鲁棒性；4) 仅在冻结的 LLM 上实验，未探究与参数更新结合的可能性。

---

## 631. RLMuscle: Muscle-Driven Character Animation via Reinforcement Learning

**arXiv ID:** 2610.30033 | [PDF](https://arxiv.org/pdf/2610.30033v1)

**作者:** Chunlei Li `[一作]` (Beihang University), He Wang `[通讯]` (University College London)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过强化学习自动生成肌肉激活，实现参考动作的物理一致性肌肉驱动角色动画。

**💡 创新点**

创新点在于将1D Hill肌肉模型作为RL训练代理，随后在3D体素肌肉上使用相同的Hill构型；引入EMG先验以生理合理化激活；以及设计基于几何多重网格的XPBD求解器实现实时体素模拟。

**🔧 技术方法**

使用强化学习（PPO）与双网络共训练；1D Hill肌肉模型；XPBD（扩展位置基动力学）与几何多重网格；骨肌碰撞SDF；EMG先验；GPU并行Warp引擎。

**📊 数据集**

使用混合数据：1D肌肉骨骼模型（284肌肉），3D人体体素肌肉模型（194肌肉）以及公开的运动捕捉数据（BVH）和Moreira等人的表面EMG。

**📈 对比分析**

与Houdini的Otis Solver和Han等人神经肌肉模型做对比，结果显示我们的体素模拟在几乎相同几何精度下，帧时为18 ms/帧，远快于Otis的20.8 s/帧；在运动跟踪误差和肌肉激活物理合理性上均优于基线。

**⚠️ 局限性**

局限性包括：训练过程中无法直接在3D体素模拟中进行RL，导致无法实时反馈；几何多重网格对细长肌肉效果不佳；骨肌耦合仍采用经验性弹簧，缺乏生理依据。

---

## 632. Measurement-Guided Conversion of Laboratory Eye-Tracking Data for VR Authentication

**arXiv ID:** 2610.30032 | [PDF](https://arxiv.org/pdf/2610.30032v1)

**作者:** Md Arif Shahriar `[一作]` (Texas State University), Oleg V. Komogortsev `[通讯]` (Texas State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

将实验室眼动追踪数据转换为 VR 头显的信号质量，同时保持眼动轨迹不变。

**💡 创新点**

基于测量的设备模型结合频率锁定扩散渲染器，精确重现 VR 设备的采样、校准误差和噪声谱。

**🔧 技术方法**

使用参数化设备模型、最优运输匹配、频率锁定扩散渲染器以及深度学习 EKYT 认证器。

**📊 数据集**

使用公开的 GazeBase（EyeLink 1000）和 GazeBaseVR（SMI 在 HTC Vive）数据集。

**📈 对比分析**

通过与原始实验室数据、百分位匹配和样式迁移的信号质量、逼真度和身份认证性能对比，NDR 在 45 秒时 d' 与真实 VR 训练相近，EER 5.9%；在 5 秒时仍低于真实 VR。

**⚠️ 局限性**

未能重现 VR 的慢漂移、低频噪声、saccade 速度衰减，噪声缺乏时间/位置变异，样本来源差异可能导致身份信息被误用，仅针对单一设备/任务，且未评估对长期保持的影响。

---

## 633. ExGrad: Beyond Suppression - Explicit Clean-Gradient Subtraction in Adversarial Training

**arXiv ID:** 2610.30018 | [PDF](https://arxiv.org/pdf/2610.30018v1)

**作者:** Zhiyuan Ye `[一作]` (University of Science and Technology of China), Yi Zhou `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

该工作提出 ExGrad 方法，通过对清洁样本、对抗样本及加权扰动状态下的梯度进行虚拟步长测量，构造并选择最优线性组合以提升对抗鲁棒性。

**💡 创新点**

关键创新是将梯度视为候选更新，利用有限步长对抗边际响应来测量其实际效果，并允许在有界仿射空间中求解负系数的梯度组合，从而实现显式清洁梯度减法。

**🔧 技术方法**

采用的技术包括 PGD 对抗样本生成、AWP 权重扰动、虚拟更新、对抗边际响应测量、仿射组合求解器、对抗攻击评估（PGD、APGD、AutoAttack）以及基于梯度归一化的控制实验。

**📊 数据集**

在 CIFAR‑10、CIFAR‑100 与 ImageNet‑100 三个数据集上使用 WideResNet‑28‑4/34‑10 等架构进行实验。

**📈 对比分析**

与标准 PGD‑AT、TRADES、AWP 等基线相比，ExGrad 在不同仿射半径下实现了显著的鲁棒性能提升，例如在 CIFAR‑100 上 clean 提升约 3.8%，PGD‑20 提升约 1.4%，AutoAttack 提升约 1.8%。

**⚠️ 局限性**

限制方面包括：仅考虑三种梯度候选，缺乏多攻击/多候选的更全面探测；仿射半径调节需要经验选择；训练成本相对较高，仅在实验环境下验证。

---

## 634. GuideEvolve: Self-Evolving Agent Framework with Expert Guidance and 3Ts Mechanisms

**arXiv ID:** 2610.30017 | [PDF](https://arxiv.org/pdf/2610.30017v1)

**作者:** Meihong Jia `[一作]` (Amazon), Jonathan Taws `[通讯]` (Amazon Web Services)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GuideEvolve框架，让自进化智能体通过任务专家评估和3Ts机制实现更稳定、高效的自我改进。

**💡 创新点**

创新点在于将任务专家评估与演化控制分离，引入Think Fast and Slow、Throwback、Termination by Choice三大机制，并通过专家模块提供“清洁梯度”式指导。

**🔧 技术方法**

使用Anthropic Claude 3.5 Sonnet等LLM、AWS Bedrock服务、工具调用、JSON响应验证等技术。

**📊 数据集**

在MGSM（小学数学）和MMLU（多任务语言理解）两个基准数据集上进行评测。

**📈 对比分析**

与Gödel Agent对比，GuideEvolve在MGSM达到89.8%/90.6%准确率，在MMLU达到89.1%/87.9%准确率，方差下降25–59%，成功率100%，迭代次数约为原来的一半。

**⚠️ 局限性**

局限性包括模型依赖性（Claude vs GPT）、仅评测两项任务、额外LLM调用成本、以及对业务场景适用性的进一步验证仍需研究。

---

## 635. Multiclass Boundary Refinement with Neural Cellular Automata for Remote Sensing Land Cover Segmentation

**arXiv ID:** 2610.30007 | [PDF](https://arxiv.org/pdf/2610.30007v1)

**作者:** Pascoassis Souza Santos Meira `[一作]` (National Laboratory of Scientific Computing), Rafael Duarte Coelho dos Santos `[通讯]` (National Institute of Space Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Boundary Refinement Neural Cellular Automaton (BRNCA)，在冻结的分割器输出上通过迭代局部规则细化遥感图像的多类别边界。

**💡 创新点**

创新点在于将 NCA 简化至最小核心：使用 Aitchison 对数空间更新保持概率简单、加入可学习隐藏通道、固定 Sobel/Identity 感知核、同步更新，并去掉活跃掩码、样本池等传统 NCA 机制，从而专注于边界位置修正。

**🔧 技术方法**

技术手段包括：预训练冻结的 Dense Prediction Transformer（DPT‑DINOv2）作为基线分割器、神经细胞自动机（NCA）架构、Aitchison 对数空间多分类更新、条件卷积编码器、固定深度卷积感知核（Identity+Sobel）以及多步 BPTT 训练。

**📊 数据集**

主要使用 LandCover.ai 数据集（波兰航空正射影像，5 类），作为评估基准。

**📈 对比分析**

通过与冻结基线分割器在 mIoU 与 Boundary IoU（BIoU）上的对比验证；BRNCA 在 BIoU 上提升 0.9–2.4 个百分点（最高 48.99%），在整体 IoU 上提升 0.2–0.6 个百分点，最显著提升集中在 woodland 类。

**⚠️ 局限性**

局限性包括：仅在单一基线模型和单一数据集上评估；改进幅度有限（BIoU 仍低于 50%）；未与现有边界细化方法（如 CRF、SegFix 等）做直接对比；未引入样本池导致在训练步数之外的收敛不稳；未报告计算成本与推理速度。

---

## 636. GuidedSAM-Plume: An Expert-Guided Segmentation Framework for Volcanic Plumes

**arXiv ID:** 2610.29825 | [PDF](https://arxiv.org/pdf/2610.29825v1)

**作者:** Taddeo D'Adamo `[一作]` (Université Lumière Lyon 2), Bertrand Kerautret `[通讯]` (Université Lumière Lyon 2)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于SAM2的交互式火山羽流与云层分割框架GuidedSAM-Plume，并在47条观测序列共759帧的数据集上重新标注并验证；

**💡 创新点**

创新点在于结合专家指导的零样本交互式分割，显著提升分割精度且无需再训练，降低火山监测的人工成本；

**🔧 技术方法**

使用了SAM2零样本分割模型、空间提示（边框、点）以及视频传播机制，并实现了完整的交互式工作流；

**📊 数据集**

使用了47条观测序列（共759帧）火山灰羽流/云图像的数据集，包括手工和专家重标注版本；

**📈 对比分析**

与已有全自动模型（Wilkes等、VigiaPlume-Net）以及单点SAM2基线进行对比，GuidedSAM-Plume实现99.5%准确率、93%召回率、90%Dice，火山参数误差低于10%；

**⚠️ 局限性**

局限在于受SAM模型对点数/短视频的训练限制，仍需专家干预；数据集规模有限，难以覆盖所有极端场景，细节边界仍可能出现误差。

---

## 637. EPIC: Rethinking Trust Regions for Autoregressive LLM Policies via Exact-KL Prefix Importance-Ratio Clipping

**arXiv ID:** 2610.29999 | [PDF](https://arxiv.org/pdf/2610.29999v1)

**作者:** Chengxi Li `[一作]` (Tsinghua University), Hao Zhou `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对离线自回归语言模型的强化学习，本文提出了一种前缀感知的 PPO 风格剪裁目标（EPIC），通过前缀重要性比率和逆 KL 坐标构建信赖域，实现了稳定的训练。

**💡 创新点**

创新点在于：① 将自回归生成建模为有限时程 MDP，推导出前缀比率为准确的步进重要性权重；② 通过逆 KL 坐标将 TRPO 的信赖域约束转化为可直接在 PPO 换算器中使用的剪裁区间；③ 引入几何均值校准的前缀概率，确保剪裁在不同生成长度上均匀且有效。

**🔧 技术方法**

技术主要包括：TRPO 与 PPO 的理论推导、前缀重要性加权、逆 KL 解析式、几何均值概率校准、以及基于前缀比率的双向剪裁算子；训练实现中不需要额外的模型前向或后向计算，保持与现有 PPO 训练循环兼容。

**📊 数据集**

使用公开的 Qwen3‑30B‑A3B‑Base Sparse MoE 模型，并在 DAPO‑Math‑17K 数据集上进行训练；在四个数学推理基准（AIME 2024/2025、AMC 2023、HMMT 2025）上评估。

**📈 对比分析**

与 GRPO、DAPO、GSPO、CTPO、RIPO 以及 R2/R3 路由重放的基线相比，EPIC 在四项基准上的平均准确率提升至 39.34%，相较 GSPO 提升 6.5% 点；在离线 MoE 训练中，EPIC 同时实现了 77.5%–93.0% 的驱动进程内存节省，且无需存储专家路由信息。

**⚠️ 局限性**

限制主要包括：仅在单一大型 MoE 语言模型上验证，缺乏跨模型或更广泛任务的评估；需要手工调节剪裁阈值 δ⁻/δ⁺，对不同数据或模型可能需重新设定；对极长生成序列的效果尚未系统分析。

---

## 638. DONIC: Generalist Dexterous Bimanual Tracking Control from Human Demonstrations

**arXiv ID:** 2610.29997 | [PDF](https://arxiv.org/pdf/2610.29997v1)

**作者:** Mingze Sun `[一作]`, Ziwei Liu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一个通用双手参考跟踪器，并构建了多视角双手交互数据集BiHOI；

**💡 创新点**

创新点在于：①第一个跨操作类别和人类运动数据集的通用双手跟踪器；②BiHOI数据集强调持续双手接触并可重定位为机器人参考；③设计了对象中心单手奖励与Clip-Centered Advantages，支持异步接触和不同难度参考的PPO+BC学习；

**🔧 技术方法**

采用了PPO与行为克隆的反馈循环、对象中心奖励、Clip-Centered Advantages、多视角RGB重建、MANO手模型与机器人重定位等技术；

**📊 数据集**

使用了GRAB、GigaHands、OakInk2、TACO、H2O等公开双手数据集以及自制的1,575序列BiHOI（229种对象）；

**📈 对比分析**

在两个公开评测拆分（未见轨迹和未见对象）下，单一检查点在未见轨迹的成功率达86.79%、完成率67.62%，在未见对象的成功率57.32%、完成率54.69%，相较于DexTrack-bi和Dexplore-bi提升约77-45个百分点；

**⚠️ 局限性**

局限性：评估仅在仿真单物体交互、基于重建/重定位的参考，未验证在真实物理双手机器人上的迁移；缺乏对长时序和多物体场景的实验。

---

## 639. Enumeration-based Triangle Counting on GPUs

**arXiv ID:** 2610.29988 | [PDF](https://arxiv.org/pdf/2610.29988v1)

**作者:** Sayan Ghosh `[一作]` `[通讯]` (Pacific Northwest National Laboratory), Sayan Ghosh (Pacific Northwest National Laboratory)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在分布式GPU系统上实现了基于枚举的精确三角计数算法，并通过多级优化实现高并行度。

**💡 创新点**

创新点在于结合异步MPI非阻塞通信与GPU零拷贝IPC，采用单计数规则和批量边筛选来显著降低通信与负载不平衡。

**🔧 技术方法**

技术包括MPI非阻塞发送接收、CUDA Cooperative Groups、GPU内核优化（masked-edge与split-edge）、压缩通信缓冲、CPU‑GPU混合计算以及NVLink IPC。

**📊 数据集**

使用多种真实大规模图数据集（如orkut、friendster、webbase、clueweb09、scale‑20/25等），涵盖社交网络、Web、学术引用等领域。

**📈 对比分析**

与TriC、cuGraph、TRUST、G^2Miner等现有实现比较，CPU+GPU、MPI+IPC、单节点IPC方案均实现了10–60%更快或10×更高的性能，并在单GPU上实现了2–100×的加速。

**⚠️ 局限性**

主要限制在于图结构高不规则性导致的线程分歧和剩余线程块，压缩通信虽减小数据量但会增加CPU负载；对极大稀疏图的负载均衡仍需进一步改进。

---

## 640. Code Comprehensor: Hierarchical Documentation and Cross-Modal Retrieval for Large Repositories

**arXiv ID:** 2610.30016 | [PDF](https://arxiv.org/pdf/2610.30016v1)

**作者:** Sujoy Roychowdhury `[一作]` (Ericsson R&D), Diarmuid Corcoran `[通讯]` (Ericsson R&D)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了 Code Comprehensor，一套端到端框架，用调用图、LLM 生成层次化文档（CodeScribe）和 CLIP 风格对比模型（CodeContrast），并通过 MCP 架构实现对大型代码库的文档生成与检索。

**💡 创新点**

创新点包括：递归后序分块算法在大规模调用树中保持层级结构；利用层次化文档与父子/兄弟关系构造多粒度监督；将生成的文档作为训练信号训练双编码对比模型，实现高效无 LLM 运行检索。

**🔧 技术方法**

采用静态分析生成调用图、Sonnet LLM 进行文档与合成问答生成、后序分块、CLIP 双编码对比学习（InfoNCE，多正样本）、FSDP 分布式训练、MCP 代理架构、PlantUML 生成图示、NetworkX、FastAPI 等技术。

**📊 数据集**

使用了 OpenCV（约120万 LOC 的 C/C++ 代码库）和一家电信公司的 5M LOC C 代码仓库作为评估数据集。

**📈 对比分析**

与基线方法比较，D2（自注释）在 OpenCV 非合成查询中 Top‑1 约96.4%，D3（加父/兄弟 + QA）在合成查询中 Top‑1 约80.6%；整体 Top‑1 在 70–95% 之间，RAGAS faithfulness 90%+，检索推理 <50 ms。

**⚠️ 局限性**

局限性包括：仅在 C/C++ 系统编程域验证，未测试多语言；LLM 版本差异与非确定性可能影响数据；缺乏中间 ablation 研究；对低资源语言或缺失注释的仓库效果可能下降。

---

## 641. A Real-time tax compliance architecture for VAT split payments with Making Tax Smart and Open Banking

**arXiv ID:** 2610.30055 | [PDF](https://arxiv.org/pdf/2610.30055v1)

**作者:** Panos Louvieris `[一作]` (Brunel University of London), Gareth White `[通讯]` (Brunel University of London)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于Open Banking与R3 Corda的MTS‑OB架构，实现VAT实时分拆支付与自动合规；

**💡 创新点**

创新点在于将税务功能直接嵌入现有支付链，利用银行存款的实时信息传感，构建“实时税收”并保持金融稳定；

**🔧 技术方法**

使用技术包括R3 Corda分布式账本、智能合约与流、Open Banking API、ISO20022/Peppol电子发票等；

**📊 数据集**

使用数据集为OpenBankProject/OBP‑API沙盒模拟的银行账户与交易数据，以及测试用的虚拟发票JSON文件；

**📈 对比分析**

通过全因子实验（交易负载×客户端数量）测量吞吐量和延迟，结果显示延迟随客户端线性增长，吞吐量在40+客户端后饱和，无交易拒绝，表明可扩展；

**⚠️ 局限性**

局限在于实验仅在单机、Corda OSS 4.8单流器、H2数据库、沙盒API下进行，未评估多节点并发、生产级网络延迟、企业数据库等，吞吐量受限。

---

## 642. TARSphere: A Robot-Oriented Full-Stack Framework for Long-Horizon Embodied Agents in Evolving Physical and Social Environments

**arXiv ID:** 2610.30043 | [PDF](https://arxiv.org/pdf/2610.30043v1)

**作者:** Yicheng Xu `[一作]` (Zhejiang University), Jiangning Zhang `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个全栈框架，将机器人动力学、具身NPC交互、事件驱动的有限状态机和评估体系结合起来，并实现了一个“具身脑”利用冻结的大语言模型通过记忆、技能接口和执行反馈进行闭环决策；同时提出了任务设计与数据集的具身化构建流程；在此框架下，对多种模型进行大规模评测；

**💡 创新点**

①将推理与物理执行紧密耦合，打造可在动态环境中持续更新计划的“具身脑”；②提出具身化任务生成与“Agent‑in‑the‑Loop”审校的完整流程；③设计多维度能力需求（PR、DA、TU、HC、IU）的动态任务体系；④在同一评测平台上对冻结的LLM、开放VLM和具身模型进行统一比较，首次揭示执行安全（碰撞、跌倒）与任务成功的关联。

**🔧 技术方法**

物理仿真引擎（摩擦、接触），具身NPC模拟，事件驱动的FSM，RGB‑D+LiDAR感知，多模态记忆模块，技能接口（导航、抓取、社交动作），冻结的大语言模型（如 LLaMA、LLaVA 等），以及执行反馈回路与评估器。

**📊 数据集**

10 个住宅场景 + 20 个商业场景的仿真环境；基于任务卡、NPC 配置与场景资产生成的 60 条基础任务及其 5 种能力变体；采用 Agent‑in‑the‑Loop 方式构建的完整数据集。

**📈 对比分析**

对 4 种专有模型、2 种开放 VLM（如 LLaVA）和 2 种具身模型进行统一评测，指标包括任务进度、完成率、路径效率以及碰撞/跌倒率。实验显示，专有模型排名最高；最佳开放 VLM（LLaVA）在导航 40.6% 与交付 47.8% 上超过两种具身模型；然而即便是最强模型，碰撞率 84‑100% 与跌倒率 9‑47% 仍高，说明执行安全仍是瓶颈。

**⚠️ 局限性**

仅在仿真环境下测试，缺乏真实机器人与真实人类交互的验证；评测范围限定于单一机器人身体和两类室内服务任务；冻结模型的能力受限；评测指标仍以聚合数值为主，未能深入解析不同子任务的细节差异。

---

## 643. Plausible but Unimplantable: A Trustworthiness Audit of Shape Completion Foundation Models for Cranial Defect Reconstruction

**arXiv ID:** 2610.30035 | [PDF](https://arxiv.org/pdf/2610.30035v1)

**作者:** Marek Wodzinski `[一作]` (AGH University of Krakow), Mateusz Daniol `[通讯]` (AGH University of Krakow)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

对三种公开预训练的形状补全模型（MSN‑FM、MGPC、AdaPoinTr）在239个头骨缺损样本上进行零射击评估，比较其补全结果与患者剩余骨骼的“无操作”基线，并引入物理可植入性判据（1 mm碰撞检测+2 mm覆盖优势）以及语言条件对齐测试。

**💡 创新点**

首次在临床头骨缺损场景下评估基础模型的可信度，提出了基于物理可植入性的评价标准和无操作基线，揭示了当前基础模型在实际植入任务中的不足。

**🔧 技术方法**

使用点云扩散/流匹配模型（MSN‑FM、MGPC、AdaPoinTr），零射击推理，几何碰撞检测、覆盖回收率计算，以及通过点密度估计厚度的技术。

**📊 数据集**

数据集包括210个从SkullBreak和SkullFix合成的缺损头骨，以及29个来自MUG500+的真实开颅切除缺损样本。

**📈 对比分析**

方法与性能：每个模型的输出与患者剩余骨骼做优势比较（2 mm回收率）并测试1 mm碰撞自由度。结果显示：三种基础模型在任何配置下均未能产生可植入输出（0/239），而任务特定的体素模型在同样测试中实现了238/239 的可植入率，表现显著优于基础模型。

**⚠️ 局限性**

局限性：仅评估公开检查点，未进行头骨数据微调；评价仅基于几何/物理指标，缺乏临床外科验证；点云密度不足以恢复薄板厚度，导致厚度不可读；输出表示为离散点云而非体素/隐式体，难以直接用于医疗植入。

---

## 644. RealMe: Immersive Volumetric Video from Minimal Captures

**arXiv ID:** 2610.30023 | [PDF](https://arxiv.org/pdf/2610.30023v1)

**作者:** Chenxu Zhou `[一作]` (Zhejiang University), Xiaowei Zhou `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种统一的 RealMe 框架，通过高分辨率视频扩散模型将稀疏低质量捕获提升为高分辨率、4D 连贯的体积视频，随后用于体积视频重建。

**💡 创新点**

创新点包括：① 将预训练的高分辨率图像扩散模型迁移到 4D 任务，利用布局感知位置嵌入和分解时空注意力；② 在训练中加入潜在频率损失以缩小生成视频与真实捕获的频域差距；③ 通过辅助视频构造将稀疏观测信息整合到扩散过程。

**🔧 技术方法**

使用技术包括：高分辨率扩散模型 FLUX、ControlNet + LoRA 适配、布局感知 RoPE 位置嵌入、分解时空注意力、潜在频率损失、4D Gaussian Scene（FTGS）重建。

**📊 数据集**

数据集：自采集的 RealVV（多焦距、多场景、高分辨率多视角视频）以及公开的 DNA-Rendering 基准。

**📈 对比分析**

对比方法包括 FTGS、DIFIX3D+、4C4D、4K4D、Diffuman4D、STAR、FlashVSR 等；RealMe 在视觉质量、4D 连贯性和时间一致性指标上均优于这些基线，尤其在 2K 分辨率下实现了更高的 PSNR/SSIM 与更低的 TWE。

**⚠️ 局限性**

局限性在于：训练数据有限，难以覆盖大型场景或极端相机设置；对几何信息不足的输入可能无法初始化有效的粗略 4DGS；推理速度受限，尚需提升扩散模型的高分辨率推理效率。

---

## 645. Building Auditable AI Text Detectors with Sparse Autoencoders

**arXiv ID:** 2610.30006 | [PDF](https://arxiv.org/pdf/2610.30006v1)

**作者:** Liam Dugan `[一作]` (University of Pennsylvania), Chris Callison-Burch `[通讯]` (University of Pennsylvania)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种将黑盒神经网络AI文本检测器转换为可解释模型的方法：对检测器的残差流训练稀疏自编码器（SAE），用自动解释生成可读特征描述，并筛选高可信度特征，再用轻量化L1正则化逻辑回归进行分类；

**💡 创新点**

创新点在于利用SAE解耦残差流成单义特征，自动生成自然语言解释并通过全局模拟分数筛选，实现在保持原模型性能的同时提供可审计的概念级解释；

**🔧 技术方法**

关键技术包括稀疏自编码器训练、自动解释（描述生成与验证）、特征筛选阈值、L1正则化逻辑回归以及基于残差流的特征提取；

**📊 数据集**

使用的数据集包括COLING（41种模型与人工文本）、RAID（多模型外部测试）、PRDB（学术同行评审文本）以及公开的多模型生成文本；

**📈 对比分析**

与传统神经模型（如Gemma-2-2B）、基线stylometric（gram2vec）以及四个零样本检测器相比，基于SAE的分类器在分布内外均能达到或超过基线，尤其在外部分布样本上表现优异；

**⚠️ 局限性**

局限性包括自动解释偏向召回、缺乏对不同基线模型的SAE训练与评估、以及仅在GemmaScope基础上验证，未探索其他大模型的通用性。

---

## 646. When Majority-Vote Fails: Sequence Reconstruction under Fixed-Weight Substitutions

**arXiv ID:** 2610.30001 | [PDF](https://arxiv.org/pdf/2610.30001v1)

**作者:** Vivian Papadopoulou `[一作]` (Technical University of Munich), Antonia Wachter-Zeh `[通讯]` (Technical University of Munich)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在固定权重替换错误下的二进制序列重建问题，确定了保证唯一重建所需的最小不同通道输出观察数量，并证明了比特位多数投票解码器在此情况下是次优的。

**💡 创新点**

提出了一种加权多数投票解码器，旨在用比特位多数投票解码器更少的观察次数保证唯一重建，并且在某些情况下成功解码了比特位多数投票失败的通道输出集。

**🔧 技术方法**

使用了加权多数投票解码技术，并提出了一种早期终止规则，以提高解码效率。

**📊 数据集**

使用了二进制序列的模拟数据集，具体的输出集数量和参数在文中进行了详细的数值评估。

**📈 对比分析**

通过与比特位多数投票解码器的比较，展示了加权多数投票解码器在需要的观察数量上具有优势，尤其是在较高的错误权重情况下，性能显著提高。

**⚠️ 局限性**

限制在于加权多数投票解码器的复杂性略有增加，并且在某些情况下仍可能失败，尤其是在观察数量接近最小阈值时。

---

## 647. Looking Ahead: Modeling Future Evolution in Autoregressive Video Generation

**arXiv ID:** 2610.29989 | [PDF](https://arxiv.org/pdf/2610.29989v1)

**作者:** Ruopeng Gao `[一作]` (Nanjing University), Limin Wang `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

引入未来块预测（Future Chunk Prediction，FCP）作为自回归视频扩散模型的辅助目标，在教师强迫训练的基础上预测更远的未来视频块，以迫使模型学习视觉演化信息。

**💡 创新点**

创新点在于：①仅利用训练视频自身的未来信息，无需额外预训练视觉编码器；②并行预测分支设计避免未来信息泄露，提升表示结构；③通过对中间隐藏层的预测监督，显著提升物理可行性和表示空间的时空结构。

**🔧 技术方法**

使用技术包括：流匹配扩散框架、基于DiT的自回归视频生成、教师强迫去噪、轻量级辅助DiT预测器、AdaLN特征调制以及多尺度特征拼接。

**📊 数据集**

数据集：混合内部视频与WISA-80K作为训练集；评估使用Physics-IQ Verified、WorldModelBench、PAI-Bench-G、PhyGround等物理相关基准。

**📈 对比分析**

与教师强迫基线在匹配的数据与设置下对比，1.3B模型IQ从27.6提升至30.9，5B模型I2V与V2V分别从30.4/40.6提升至33.8/44.4；在其他基准亦获得小幅但一致的性能提升。

**⚠️ 局限性**

局限性：需要额外的训练计算和复杂的辅助模块；在部分基准（如WorldModelBench等）提升有限；链式预测会出现信息泄露；内部训练数据不可公开，重现性受限。

---

## 648. FaultGene: LLM-Assisted Fault-Augmented Test Generation for Hardware-in-the-Loop Validation of Autonomous Driving Systems

**arXiv ID:** 2610.29969 | [PDF](https://arxiv.org/pdf/2610.29969v1)

**作者:** Chaoran Yuan `[一作]` (University of California, Irvine), Fadi Kurdahi `[通讯]` (University of California, Irvine)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

利用大语言模型与硬件‑仿真(HIL)反馈，自动生成可直接在CARLA/Scenic环境下执行的故障增强测试用例；

**💡 创新点**

1）将LLM生成与HIL执行反馈相结合的迭代细化机制；2）通过视频特征提取(SenarioFeat)提供结构化监督；3）通过监督微调实现一次性生成，省去手工参考；4）在模块化与端到端AD架构上验证，保持与手工HIL相同的故障敏感性趋势；

**🔧 技术方法**

大语言模型（Qwen3‑14B、Qwen2.5‑VL‑7B）、prompt engineering、视频特征提取（ScenarioFeat）、LLM生成 Scenic 程序、执行‑反馈驱动的迭代改进、QLoRA 微调、CARLA/Scenic 仿真、ADDT HIL 框架、LLFI/NVBitFI 计算机视觉与时序测量；

**📊 数据集**

基于CARLA/Scenic的三种驾驶场景（车辆跟随、大曲率转弯、超车），共 600 条基线场景‑故障组合用于训练/微调，200 条留作测试；20 种不同故障类型；使用 ADDT 记录的视频作为参考执行；

**📈 对比分析**

对比一次性 LLM 生成与微调后 FaultGene 在模块化和端到端系统下的语义一致性、编译成功率、执行成功率；结果显示语义对齐从 0.45/0.43 提升至 0.89/0.86，编译率从 68%/71% 提升至 96%/94%，执行率从 81%/74% 提升至 93%/90%；与手工工程相比，故障场景准备时间提升 72 倍，整体测试时间减半（约 53%）；故障敏感性趋势与 ADDT 保持一致；

**⚠️ 局限性**

目前仅在生成后使用 HIL 反馈，未将时序/调度效应直接融入生成目标；依赖 CARLA/Scenic、LLM 以及特定的故障注入工具，缺乏对多硬件平台和更广泛故障模型的适配；未能全面处理多传感器协同与实时性问题。

---

## 649. CORD: Conflict-Aware Optimization for Joint RL and Distillation in Agents

**arXiv ID:** 2610.29965 | [PDF](https://arxiv.org/pdf/2610.29965v1)

**作者:** Wenhao Yu `[一作]`, Irwin King `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出CORD框架，用于在强化学习和自蒸馏两种目标之间冲突时，保留一个目标梯度并投影消除相反方向的梯度，从而提升语言模型代理的任务完成率。

**💡 创新点**

创新点在于将冲突投影的保护方向和触发门槛解耦，提供闭式更新、一次性梯度组合，并给出一阶影响分析，说明保护方向和混合权重如何共同决定最终更新。

**🔧 技术方法**

采用的核心技术包括梯度投影（conflict-only/always gate）、多目标优化分析、GRPO（Group Relative Policy Optimization）与On‑Policy Distillation、以及基于微批量累积的梯度组合。

**📊 数据集**

实验数据集涵盖ALFWorld（家庭任务）、WebShop（购物指令）以及SDAR基准任务，使用Qwen2.5‑3B‑Instruct模型进行训练。

**📈 对比分析**

与三种主流联合训练方案（Skill‑SD、GRPO+OPSD、SDAR）对比，CORD在ALFWorld上平均提升1.5–15.9个百分点，在WebShop上提升3.6–3.9个百分点，证明了投影策略在提升局部优化进展和最终成功率方面的有效性。

**⚠️ 局限性**

局限性包括：需手动选择保护方向与门控组合、对混合权重敏感、仅处理两目标的冲突且在极大批量或不同网络架构下的可迁移性尚未充分验证。

---

## 650. Validating Autoformalized Specifications via Contrapuntal Languages

**arXiv ID:** 2610.30059 | [PDF](https://arxiv.org/pdf/2610.30059v1)

**作者:** Yihan Dai `[一作]` (Peking University), Sergey Mechtaev `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过构造两种互不相互可译的对抗音语言，让LLM在不同的表示路径上自动化形式化，并使用SMT求解器验证两者的一致性，从而在无oracle的情况下验证自动形式化的规范是否正确。

**💡 创新点**

提出“对抗音语言”概念——一对任务特定、结构上不兼容的形式化语言，用来消除LLM生成规范时的相关错误，并通过语言合成算法自动生成满足非同构约束的语言对。

**🔧 技术方法**

利用大语言模型（GPT‑5.4 mini、Qwen3.6‑Flash）进行自动形式化，使用SMT‑LIB和Z3进行语义等价性检查；采用增量合成算法与SMT/MaxSAT编码来构造对抗音语言；并使用基于等价类的投票与round‑trip验证作为基线。

**📊 数据集**

在HumanEval+（Mini）和LiveCodeBench（Mini）两个编程评测基准上进行实验，分别涵盖数百个自然语言描述的函数。

**📈 对比分析**

与直接生成、投票（plurality voting）以及round‑trip验证三种基线相比，所提出的方法在HumanEval+上正确率提升约12%，在LiveCodeBench上提升约42.9%，接受的规范更可靠，能杀死更多bug；尽管覆盖率略低，但更为保守。

**⚠️ 局限性**

仍受种子规范的限制，无法完全消除相关错误；覆盖率低于投票与round‑trip验证；需要更多的token与计算资源；在含有歧义的任务描述下可能出现多种合理解释，需要人工确认。

---

## 651. Multiplying matrices via polynomials: exposition and barriers

**arXiv ID:** 2610.30054 | [PDF](https://arxiv.org/pdf/2610.30054v1)

**作者:** Alimzhan Amanov `[一作]` (University of Amsterdam), Jeroen Zuiddam `[通讯]` (University of Amsterdam)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究矩阵乘法的渐近谱，给出矩形矩阵乘法的外近似与内近似；

**💡 创新点**

提出一种新的外近似条件，利用对称群作用和Bernoulli不等式推导出更紧的上界；

**🔧 技术方法**

采用组合论、对称群作用、张量展开、支持函数分析以及不等式证明技术；

**📊 数据集**

本文主要基于理论推导，没有使用具体实验数据集；

**📈 对比分析**

通过与已有的内外近似三角形对比，证明新外近似更紧，从而进一步限制了矩阵乘法的指数；

**⚠️ 局限性**

结果仅适用于理论上限，缺乏对更高维张量的推广，且尚未能在实际算法中实现该理论极限。

---

## 652. From Reasoning to Action: Benchmarking Frontier AI Models on LIBERO-10 Robot Manipulation Tasks

**arXiv ID:** 2610.30051 | [PDF](https://arxiv.org/pdf/2610.30051v1)

**作者:** May Espinola `[一作]`, Yu Xiang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在LIBERO-10机器人操作基准上，对六种前沿大型语言模型（GPT‑6 Astra、GPT‑6.1 Sol、GPT‑6 Luna、Claude Opus 5、Claude Opus 5.5、Gemini‑4 Argon）使用统一闭环控制框架进行系统评估。

**💡 创新点**

设计了无需任务特定训练、仅通过视觉观测、机器人本体感知和结构化工具调用实现的闭环操作流程，首次在标准基准上量化大语言模型的机器人操控能力。

**🔧 技术方法**

利用大语言模型的函数调用接口、视觉与本体感知、move_to/done/give_up工具调用、统一系统提示以及LIBERO仿真环境实现机器人控制。

**📊 数据集**

使用LIBERO‑10（LIBERO‑100子集）中的十个长程操作任务，针对每个任务的十个预设初始状态共计100条episode，评估各模型配置。

**📈 对比分析**

通过最大80次LLM调用、900步控制限制以及每步成功检测进行评估，18个模型配置共1800条episode，平均成功率22.2%，最高GPT‑6 Astra 44%、Gemini‑4 Argon 43%，推理努力越高通常提升成功率但成本显著上升。

**⚠️ 局限性**

模型仍存在动作完成率低、精细空间安排与多步交互失败率高、推理努力成本高、缺乏对机器人本体、相机几何和控制器行为的完整理解等局限。

---

## 653. Robust Soft Skill Adaptation via Behavioral Localization in Tool-Using Agents

**arXiv ID:** 2610.30041 | [PDF](https://arxiv.org/pdf/2610.30041v1)

**作者:** Zijian Jiang `[一作]` (University of Surrey), Xilu Wang `[通讯]` (University of Surrey)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出行为定位（Behavioral Localization）方法，用于挑选教师轨迹中对技能效果最显著的位置，从而仅在这些位置上训练软提示（soft prompt）

**💡 创新点**

创新点在于将文本技能的行为影响稀疏化为少数关键位置，并通过正增益与分布式偏移结合的定位分数，精准选择训练目标，减少无关行为漂移

**🔧 技术方法**

使用基于token的概率分布计算（正增益G和Jensen‑Shannon距离JS），定义定位分数U；在选定位置最小化soft prompt与文本技能的全词汇KL散度，在未选位置则对齐无技能模型的分布；采用软提示训练框架（如SoftSkill、OPCD）与Qwen3.6-35B-A3B冻结模型

**📊 数据集**

在SpreadsheetBench数据集上进行实验，利用GPT‑5.5生成成功教师轨迹，评估共享提示与任务特定提示的成功率

**📈 对比分析**

与无技能、完整文本技能、SoftSkill（全轨迹压缩）以及OPCD（在线策略蒸馏）进行对比。共享提示下，10%定位位置提升SoftSkill至36.79%；任务特定提示下，10%定位位置成功率为57.38%，超过完整文本技能（52.46%）和OPCD（49.18%）

**⚠️ 局限性**

局限性包括仅在SpreadsheetBench上验证；软提示长度固定（8），可能无法覆盖更复杂任务；依赖预先生成的成功轨迹，若轨迹质量不足会影响定位；对其他模型和任务的泛化性尚待进一步评估

---

## 654. What the Author Wrote, What the Agent Reads: Silent Cue Loss in Enterprise Document Ingestion

**arXiv ID:** 2610.30026 | [PDF](https://arxiv.org/pdf/2610.30026v1)

**作者:** Yikai Zhao `[一作]` (Amazon), Pradeep Kumar Misra `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一种基于格式提示的文档解析器，能够在将 Word/PDF 文档转为 LLM 可读文本时保留作者通过颜色和高亮等视觉手段标记的关键信息，并通过添加语义角色标记使模型能够正确识别这些关键信息；

**💡 创新点**

创新点在于发现并量化了格式提示（尤其是颜色和高亮）在主流转换工具中被丢弃导致的可靠性损失，并提出了仅通过代码实现、无 LLM 调用的格式保留与语义标记方法，证明对 LLM 的问答性能提升了 4 倍以上；

**🔧 技术方法**

技术包括：利用 python-docx 等库读取 OOXML 运行级属性，生成带格式注解的 FormattingSpan；通过规则映射将颜色/高亮等视觉提示转化为语义角色（critical_warning、warning、deprecated 等）；将结果序列化为 Markdown+自定义标签；在评估时使用 deterministic matcher 对标注进行无 LLM 判定；实验使用 Claude Opus 等多模型推理；

**📊 数据集**

数据集为 6,462 篇真实的政策文档（公开语料），其中 76 篇满足评估条件，标注直接来自 OOXML 运行属性；实验还使用 Microsoft Office Comprehension Benchmark（OCB）文件真实性轨道；

**📈 对比分析**

对比方法包括：plain text、docx2python（保持字符格式为 HTML）、docling 等主流转换工具；评估指标为 F1、precision、recall、false positive rate；实验表明，格式无关转换的 F1 仅 0.075–0.150，而本方法达 0.690，且在 6 种回答模型上均保持优势；在 OCB 上，本方法在 38 个格式相关项目上获得 0.816 分，整体 0.959 分；

**⚠️ 局限性**

局限性包括：仅对带文本层的 PDF 适用；颜色/高亮的语义映射可能不适用于所有文化/语言场景；仅在部分模型（不包括性能较低的 Claude Sonnet 等）上能看到标记带来的提升；未经过人工标注验证，仅使用自动标注与页面渲染对照；以及在 token 预算上相对较高（约为 plain text 的两倍）。

---

## 655. Agent Behavior Maps: Locating Agents by What They Would Do, Without Running Them

**arXiv ID:** 2610.30014 | [PDF](https://arxiv.org/pdf/2610.30014v1)

**作者:** Shota Fujikawa `[一作]` (University of Tokyo), Masafumi Oyamada `[通讯]` (NEC Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并验证了“Agent Behavior Map”，通过在共享参考轨迹上对LLM-驱动的agent进行教师强制（teacher forcing）并对动作的对数似然做居中处理，得到坐标向量，从而无需再运行目标agent即可衡量其行为相似性和性能。

**💡 创新点**

创新点在于：①把LM Map扩展到闭环agent，利用共享轨迹的对数似然而非单纯文本；②推导坐标距离与轨迹KL的误差上界；③证明坐标可用于预测不同环境下的成功率并揭示邻居在行为上的相似性；④发现harness组件的效果可近似为坐标位移的向量相加。

**🔧 技术方法**

主要技术包括：教师强制（teacher forcing）、对数似然居中嵌入（step/trajectory embeddings）、KL散度理论推导、kNN/Ridge回归预测、行为距离度量（JS、Levenshtein、Jaccard、first-action），以及向量加法验证。

**📊 数据集**

使用的数据集包括：ALFWorld、TextWorld、BFCL、MiniWoB++（四大测试环境）；τ-bench、SWE-Bench、HAL日志（参考轨迹来源）；两条用于生成参考轨迹的LLM（Qwen3.8-27B、Gemma4-E4B）；100个公开权重LLM，12个harness组合。

**📈 对比分析**

比较方法：先在共享参考轨迹上计算坐标距离（step/trajectory），再与通过实际跑完的轨迹KL进行对比；利用坐标预测未跑模型的成功率，表现为Pearson r≥0.83（ALFWorld），以及在行为距离上邻居相似度显著高于仅按尺寸或性能选邻居；性能方面，坐标预测的成功率与真实值相关性高，kNN在ALFWorld上达0.88；行为相似度在四种度量上均优于基准。

**⚠️ 局限性**

局限性包括：①理论误差上界仅在“局部”小ε、小λ条件下有效，难以对远距离对进行精确量化；②依赖共享参考轨迹，若轨迹不覆盖某些状态则无法检测；③需要模型输出对数似然，限制了对仅提供API的模型的适用性；④行为相似性评估基于单次rollout，未考虑模型内在行为波动；⑤harness加法性验证仅针对有限的三槽设计与12个LLM，泛化性待验证。

---

## 656. Approximation and Inapproximability of Non-Preemptive Dial-a-Ride

**arXiv ID:** 2610.30021 | [PDF](https://arxiv.org/pdf/2610.30021v1)

**作者:** Jingyang Zhao `[一作]` (Kyung Hee University), Mingyu Xiao `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了非抢占式拨打接送问题（DaRP）和最小比率Steiner k-森林（SkF），提出了一种随机多项式时间算法，期望近似比为O(n^1/4 log^11 n）。

**💡 创新点**

提出的算法结合了层次良分离树上的确定性近似和随机度量嵌入，显著改善了之前的近似比，首次建立了非抢占式DaRP的超常数不可近似性界限。

**🔧 技术方法**

使用了随机化算法、层次良分离树（HST）和随机度量嵌入技术。

**📊 数据集**

使用了对称度量（V,c）和多重请求的输入，具体数据集未明确给出。

**📈 对比分析**

与之前的算法进行比较，性能显著提升，达到了O(n^1/4 log^12 n)的期望近似比，优于之前的O(n^0.3+ε)近似比。

**⚠️ 局限性**

在指数时间假设（ETH）下，无法实现某些常数c>0的x^1/(loglog x)^c近似，且在量化Gap-ETH下，排除了每个非负函数f的多项式时间x^o(1)近似。

---

## 657. Remembering Through Change: Provenance-Preserving Hypergraph Memory for Multimodal Agents

**arXiv ID:** 2610.29995 | [PDF](https://arxiv.org/pdf/2610.29995v1)

**作者:** Jielong Tang `[一作]` (Sun Yat-sen University), Jian Yin `[通讯]` (Sun Yat-sen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种多模态超图记忆框架 MHG‑Mem，利用出现、演化和证据三种高阶边将实体提及、事实和状态历史组织在一起，同时保留原始视觉证据；并设计了基于超图的代理搜索，能够根据查询自适应切换层级、遍历边、检索原始图像进行验证。

**💡 创新点**

创新点在于：①使用多模态超图结构和类型化高阶边实现实体跨会话的身份绑定、状态演化追踪与证据可追溯；②提出了演化节点与演化/证据边，将时间有序状态抽象为可检索的演化历史；③开发了代理式搜索流程，动态选择检索入口并利用视觉、检查与组装工具形成可验证的证据链。

**🔧 技术方法**

主要技术包括：视觉‑语言模型用于提取事实和实体；LLM 生成演化节点并合成状态摘要；多模态嵌入（MiniLM、SigLIP2）用于文本/图像相似性匹配；超图表示与高阶边操作；基于工具的代理搜索框架。

**📊 数据集**

使用了 MemEye、Mem‑Gallery 和 WorldMemArena‑S 三个多模态记忆基准；在三种不同大小的 LLM 后端（Qwen3‑VL‑8B、GPT‑4.1‑mini、Qwen3.8‑27B）上进行实验。

**📈 对比分析**

与 8 种基线（4 个单模态、4 个多模态 RAG/记忆系统）对比，MHG‑Mem 在 EM、BLEU‑1、F1 及 LLM‑Judge 分数上均取得显著提升，Judge 分数平均提升 8.20%–9.98%，在视觉、时间推理与跨模态推理等细粒度任务中表现尤为突出。

**⚠️ 局限性**

限制包括：依赖 VLM/LLM 的事实与演化抽取准确性；超图规模随会话增大而迅速膨胀，可能导致检索与存储成本上升；目前仅在公开基准上验证，尚未在真实生产环境中评估鲁棒性与可扩展性。

---

## 658. TIDE: Tree-Induced Fine-Grained Diagnostic Evaluation from Model Behaviors

**arXiv ID:** 2610.29985 | [PDF](https://arxiv.org/pdf/2610.29985v1)

**作者:** Jiayi Chen `[一作]` (Fudan University), Xuanjing Huang `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于行为监督的决策树诱导方法，用于细粒度评估大型语言模型的能力。

**💡 创新点**

创新点在于将诊断评估重构为模型行为的监督结构发现，并引入兼顾模型一致性与模型间可分离性的行为信息增益来选择树分裂规则。

**🔧 技术方法**

利用LLM作为候选分裂规则生成器，结合行为矩阵的IntraGain和InterGain计算信息增益，采用决策树诱导技术构建层次化能力树。

**📊 数据集**

在四个公开基准（MMLU-Pro、BBH、DROP、BBQ）上评估五个前沿LLM，并通过交叉基准转移检验树的泛化性。

**📈 对比分析**

与现有文本差异、质量评估和层次化方法（TextDiff、QualEval、EvalTree、SkillVerse）相比，方法在弱点一致性（ρ）和弱叶落差（Δ）上显著优越，能将微小的整体差距放大为更易解释的叶级差距，并在基准迁移时保持有效性。

**⚠️ 局限性**

局限性包括高昂的LLM调用成本、仅在静态单轮英文基准上验证、对LLM质量的依赖以及尚未验证多轮对话或开放式生成场景。

---

## 659. CulturalRiddles: A Multicultural Benchmark of Riddles

**arXiv ID:** 2610.29968 | [PDF](https://arxiv.org/pdf/2610.29968v1)

**作者:** My Chiffon Nguyen `[一作]` (Cohere Labs Community), Julia Kreutzer `[通讯]` (Cohere Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个社区参与式的文化谜语基准，收集了100多种语言、不同国家社区的原创谜语。

**💡 创新点**

创新点在于让本土社区成员自行撰写、验证谜语，并采用多层级人类评估与LLM裁判相结合的评价流程，突破了传统单一知识/多选基准的局限。

**🔧 技术方法**

使用了跨语言多模态的大规模LLM（包括Gemini、GPT‑5.5、Claude Opus等）以及规则、人工标注和LLM审稿三路评估管道。

**📊 数据集**

数据集为自发收集的约6,000条谜语，涵盖8个共享主题及社区自由选择的2个主题，覆盖了先前基准缺失的语言和地区。

**📈 对比分析**

通过与人类标注的准确率、错误类型分布及对比多模型性能，发现最强模型在大部分社区仍低于50%准确率，且在低资源语言上准确率仅为4‑14%。

**⚠️ 局限性**

局限在于对极少数语言的样本不足、对文化细节的解释仍需人工介入、以及评估过程对LLM判定的依赖可能引入偏差。

---

## 660. Commit or Abstain? System One Decision Models, LLMs, and Missing Legal Authority

**arXiv ID:** 2610.29961 | [PDF](https://arxiv.org/pdf/2610.29961v1)

**作者:** M. Mikail Demir `[一作]` `[通讯]` (University at Albany, State University of New York), M. Mikail Demir (University at Albany, State University of New York)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在法律AI中，当被引用的权威缺失时，评估系统一决策模型与生成式LLM的表现，并提出联合使用策略以减少错误引用。

**💡 创新点**

创新点在于引入“拒绝”条件的CaseHOLD扩展，量化模型在缺失正确权威时的错误和放弃率，比较决策模型与LLM的成本效益，并展示利用决策模型概率进行本地模型蒸馏的可行性。

**🔧 技术方法**

技术主要包括基于大型语言模型的决策模型（如Jev、pplx-decider等）、生成式LLM（Gemini、Claude Haiku、GPT-4o-mini等）、对齐决策阈值的策略（veto、agreement gate、System One first）、以及对Qwen3-8B进行LoRA蒸馏。

**📊 数据集**

数据集为CaseHOLD（美国法院判例的引用-持有信息），在实验中将1,670个条目中的正确持有替换为其他案例的类似持有，形成“拒绝”条件的测试集共5,314条。

**📈 对比分析**

通过比较准确率、错误引用率、放弃率、成本（每千问的费用）以及AUROC等指标，结果显示：决策模型在原始任务上与最优生成式LLM相当，但在缺失权威时错误引用率明显更低；联合使用可将错误引用率降低一半，同时成本仍低于单独使用生成式LLM。

**⚠️ 局限性**

局限包括：模型可能受到训练数据污染，缺失权威的实验仅模拟了替换情形，错误成本参数k仅为示例，且仅评估了美国英语案例与单一提示，未涵盖其他司法体系、语言或检索增强场景。

---

## 661. Training-free Acceleration for Blind Compressed Image Restoration via Activation-Calibrated Singular Value Decomposition

**arXiv ID:** 2610.29959 | [PDF](https://arxiv.org/pdf/2610.29959v1)

**作者:** Suhang Yao `[一作]` (University of Science and Technology of China), Zhibo Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关的激活校准SVD（ASVD）方法，用于低秩替换Transformer基的盲压缩图像恢复模型。

**💡 创新点**

创新点在于通过少量压缩图像的通道激活统计为SVD分解加权，保留高激活通道的表达能力，从而在不需要再训练的情况下显著提升恢复质量。

**🔧 技术方法**

使用了低秩矩阵分解（TSVD）、激活加权SVD（ASVD）、离线因子生成、PromptCIR Transformer结构和压缩激活统计等技术。

**📊 数据集**

实验采用了四个标准盲压缩恢复基准数据集：LIVE1、BSDS500、ICB 和 DIV2K。

**📈 对比分析**

与原始PromptCIR、TSVD以及其他基准方法在相同参数与MAC预算下比较，ASVD在rank64时 PSNR‑B 提升0.12–0.24 dB，参数减少59%，计算减少16%，PSNR/SSIM基本保持不变。

**⚠️ 局限性**

局限性包括：需要统一rank设定，未实现层级或非均匀rank分配；极低秩下性能提升有限；仅针对Transformer-based盲压缩恢复，其他任务和更动态的场景仍需进一步验证。

---

## 662. Tight Bounds on the Randomized Communication Complexity of Vector-in-Subspace

**arXiv ID:** 2610.30052 | [PDF](https://arxiv.org/pdf/2610.30052v1)

**作者:** Chengu Wang `[一作]` `[通讯]`, Chengu Wang

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `14d48e9d-0069-4ad9-996a-1d5968216998` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

证明了在随机通信模型下，向量在子空间问题（Vector‑in‑Subspace）的通信复杂度为Θ(√n)，即给定单位向量u和半维子空间H，判定u∈H或u∈H⊥的最小通信量与√n成比例。

**💡 创新点**

创新点在于：① 引入了对完整矩形划分的高阶高斯矩估计，利用复高斯表示实现奇偶阶矩的消失；② 通过平衡平面化、分割和子高斯分量的组合，精确控制条件矩，突破之前的k^(1/3)下界；③ 通过球面谐波和Radon变换，将向量问题映射到球面上的多项式，完成了完整的下界证明，消除了之前对总秩的限制。

**🔧 技术方法**

主要技术包括：完整矩形划分与随机矩阵法、复高斯与Wick等价、偶阶矩消失与中心化、子高斯分量与分层分解、拉塔拉-勒赫克高斯级数范数、球面谐波展开、Radon变换谱分析、超对数 Sobolev 与高阶矩方法、以及概率论中的分层潜能（supermartingale）终止技术。

**📊 数据集**

本工作不使用任何外部数据集；所有结论均基于理论分析与高斯随机变量的性质。

**📈 对比分析**

与之前的工作比较：Raz 的 O(√n) 上界被本研究完整匹配，表明经典随机通信与一方量子通信之间存在指数级差距；相比 Klartag–Regev 的 Ω(n^1/3) 下界，本研究提升到 Ω(√n)；此外消除了 Grupel 对总秩的限制，使下界更一般化。

**⚠️ 局限性**

局限性：证明依赖于高维随机矩阵和球面谐波的深度分析，推导过程较为复杂；结果适用于偶维n（n为偶数），且仅适用于随机（非确定性）通信模型；此外，在实际实现或对具体协议的构造上未给出直接的算法指导。

---

## 663. Which Differences Matter? Controlled Tests of Rule Transfer, Joint Reliability, and Information Use in Decision Models

**arXiv ID:** 2610.30045 | [PDF](https://arxiv.org/pdf/2610.30045v1)

**作者:** Tianyou Lai `[一作]` (Carnegie Mellon University), Qilei Li `[通讯]` (Central China Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过对决策模型的规则与状态输入进行受控的跨组合变化，评估模型的转移、联合可靠性和对外部信息的使用，进而识别模型在不同输入条件下的行为特征。

**💡 创新点**

提出了语义规则分离、跨规则状态四重奏测试以及基于边缘保持重关联的联合正确性校准，能够将整体准确率拆解为规则依赖、状态响应和外部信息利用三种行为维度，并揭示了转移增益随源答案一致性和冲突程度的变化规律。

**🔧 技术方法**

使用可执行算术规则语法、LoRA 低秩适配、基于规则的读出器（rank‑4、MLP）、外部决策服务 Jev，以及基于四重奏的校准与重关联技术进行实验。

**📊 数据集**

利用合成的三字段整数状态空间以及对应的可执行算术规则，构建了完整的规则–状态测试集合，并通过 oracle 给出真实答案。

**📈 对比分析**

与基线（Controlled Base）、LoRA 适配、固定读出器、Jev 等模型进行比较；使用自然准确率、诊断项准确率、四重奏准确率 QAcc 等指标。实验显示 LoRA 在保留规则上可提升 15–20% 的 QAcc，Jev 在外部表格条件下表现最佳；但增益高度依赖源答案一致性、冲突负荷以及词义等因素。

**⚠️ 局限性**

受限于仅涵盖有限深度算术规则的合成域，无法推广至更复杂或自然语言政策；LoRA–读出器对比受源能力不匹配影响；联合可靠性校准依赖固定重关联方式，未验证对独立状态采样的稳健性；词义实验仅在两种等价重写上展开，缺乏更广泛的语言多样性评估。

---

## 664. Enhancing IoT Network Reliability through Antenna Diversity in Real-World Scenarios

**arXiv ID:** 2610.30015 | [PDF](https://arxiv.org/pdf/2610.30015v1)

**作者:** Marco Niederberger `[一作]` (OST Eastern Switzerland University of Applied Sciences), Hans-Dieter Lang `[通讯]` (OST Eastern Switzerland University of Applied Sciences)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在室内真实工厂环境下，设计并构建了配备三根正交极化天线的IoT节点，测量不同天线组合对RSSI和吞吐量的影响，验证天线多样性能显著提升链路质量。

**💡 创新点**

首次在实际工业场景中系统验证多极化天线多样性（包括仅使用两根PCB天线）对接收信号强度（最高34 dB提升）和网络可靠性的作用，并量化不同极化匹配和金属干扰的影响。

**🔧 技术方法**

使用Nordic nRF52840 SoC、Qorvo SP6T三极开关、三根正交PCB与单线单极天线、OpenThread协议栈、iPerf吞吐测试、基于RSSI的动态天线选择与浏览器可视化工具。

**📊 数据集**

收集的实验数据包括六节点两层楼网络中各节点不同天线组合下的RSSI矩阵与iPerf吞吐率，形成本研究专用的实验数据集。

**📈 对比分析**

通过对比四种配置（全使用天线1、全使用天线2、每节点最佳PCB天线、每节点最佳全天线），对RSSI矩阵和吞吐率进行统计，得到天线多样性提升6–34 dB的信号强度，并在最佳全天线下实现约5%吞吐量提升（最高约79 kbps）。

**⚠️ 局限性**

实验规模仅六节点，未覆盖动态移动节点；多跳网络吞吐提升有限，未进行长期可靠性、丢包率、延迟等统计；硬件复杂度与成本上升；未探讨MIMO或更大天线组的可行性与收益。

---

## 665. ClunoGS: Compact Language 3D Gaussian Splatting via Unified Optimization

**arXiv ID:** 2610.30013 | [PDF](https://arxiv.org/pdf/2610.30013v1)

**作者:** Jian Xu `[一作]` (University of Texas at Arlington), Miao Yin `[通讯]` (University of Texas at Arlington)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种统一的语言3D高斯渲染压缩框架，在RGB和语义监督下同时进行高斯稀疏化和向量量化。

**💡 创新点**

将语义监督融入稀疏化决策并与向量量化联合优化，通过双重对偶算法实现稀疏与量化协同，解决了先压缩后语义学习导致语义丢失的问题。

**🔧 技术方法**

3D Gaussian Splatting、稀疏化（Top‑K、GaussianSpa 等）、向量量化（VQ）、基于双重对偶的交替优化以及语义监督（LangSplatV2 语义系数）等技术。

**📊 数据集**

LERF 与 ScanNet‑10 两个开源语义场景数据集。

**📈 对比分析**

与六种顺序压缩流水线（如 Top‑K+CompGS、PUP+CompGS、GaussianSpa+C3DGS 等）以及直接提升方法（Occam's LGS、Dr. Splat 等）进行对比；在 20% Gaussian 保留率、约 60 MB 存储条件下，统一优化方案在 mIoU、准确率和 PSNR 上仅比原始模型下降 1.6、1.1、0.07 dB，但相较于所有顺序方案提升 10+ 点。

**⚠️ 局限性**

仍依赖较大原始模型的先验；极端压缩率下语义精度下降有限；对不同硬件平台的内存占用与适配未作深入研究。

---

## 666. Pinch-Free Orientations of Polyomino Grid Graphs

**arXiv ID:** 2610.29994 | [PDF](https://arxiv.org/pdf/2610.29994v1)

**作者:** James Fox `[一作]` (Villanova University), Mirela Damian `[通讯]` (Villanova University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种波前扫描算法，在线性时间内为任何连通多边形生成其双图的根指向无钳位（rooted pinch‑free）方向，保证机器人构造过程的连通性与无单格空隙性。

**💡 创新点**

创新点在于将此前隐式的O(n²)算法改进为显式的O(n)线性时间方法，并且能够处理包含孔洞的多边形。

**🔧 技术方法**

核心技术是按列分解双图、维护前沿(frontier)和局部工作列表，采用定向规则（垂直边指向种子、水平边按波前方向指向）保证无钳位、无环与根向性。

**📊 数据集**

实验使用理论上的多边形格点列表（按lexicographic排序或二维占据矩阵），未采用具体公开数据集。

**📈 对比分析**

与之前的O(n²)算法对比，本文实现了更低的时间复杂度O(n)和空间复杂度O(n)，并通过理论分析验证了线性性能。

**⚠️ 局限性**

局限性在于该方向方案无法完整捕捉外部可访问性约束；存在某些多边形在外部访问模型下不可构造，波前方向无法满足此全局可达性需求。

---

## 667. GhostFaceMPC: Privacy-Preserving Face Recognition with Multi-Party Computation

**arXiv ID:** 2610.29987 | [PDF](https://arxiv.org/pdf/2610.29987v1)

**作者:** Davide Carnemolla `[一作]` (University of Catania), Mario Di Raimondo `[通讯]` (University of Catania)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出GhostFaceMPC，端到端在多方安全计算下完成深度CNN人脸特征提取、匹配及阈值判定，且不暴露任何原始图像或特征；

**💡 创新点**

首次将完整深度人脸特征提取与匹配放入MPC，结合知识蒸馏训练32位固定点量化模型，并通过批归一化折叠、层融合及两项舍入优化显著降低通信与轮数；

**🔧 技术方法**

使用MP‑SPDZ的replicated‑ring协议实现半诚实安全，ps‑rep‑ring实现恶意安全；采用GhostFaceNetV1‑1+ArcFace架构、知识蒸馏、fixed‑point量化、批归一化折叠、层融合、两项舍入以及向量化技术；

**📊 数据集**

在五个公开人脸验证基准（LFW、AgeDB、CALFW、CPLFW、CFP‑FP）上进行评测；

**📈 对比分析**

与MP‑SPDZ（64位）、EzPC（Porthos）以及纯TensorFlow做对比；在Co‑Host环境下验证耗时2.4 s、312 MB通信，LAN 3.8 s；相较基线通信量下降70%，轮数下降96%，准确率仅低0.32个百分点；

**⚠️ 局限性**

在大范围网络中仍受约1400轮通信导致高延迟；仅支持1:1验证，1:N需要线性扩展；未实现全MPC注册与大规模识别。

---

## 668. Free-XOR TFHE via Lazy Bootstrapping

**arXiv ID:** 2610.29983 | [PDF](https://arxiv.org/pdf/2610.29983v1)

**作者:** Ali Ajorian `[一作]` (University of Basel), Osman Biçer `[通讯]` (University of Basel)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

提出了一种新的门引导（Gate Bootstrapping）框架，利用两位编码实现 XOR 的自由（free）评估，并通过两次可编程引导（PBS）支持所有对称二元布尔门；

**💡 创新点**

创新点在于将 XOR 的自由评估与对称门的通用两步 PBS 机制结合，采用简单的两位编码与头空间（headroom）策略，完全基于 LWE 难题，简化了先前基于 NTRU 的复杂实现；

**🔧 技术方法**

主要技术包括 TFHE、LWE/GLWE 加密、可编程引导、盲旋转（blind rotation）、头空间管理与延迟引导（lazy bootstrapping），并在 Rust 语言中实现并利用 Bristol Fashion 语法编译电路；

**📊 数据集**

实验使用了若干 toy 逻辑电路（如 XOR-tree、Toy1、Toy2、Toy3）作为测试集；

**📈 对比分析**

通过与标准 TFHE 每门都进行 PBS 的基线方法对比，实验显示在 XOR 主导的电路中实现约 88% 的加速，随着对称门比例增大，速度提升逐渐递减；

**⚠️ 局限性**

局限性包括只能支持对称门（仅 8 种），无法处理非对称门；头空间的大小需在安全性与性能之间平衡；参数调优与安全级别的进一步工程化仍待完善。

---

## 669. Every Alarm Has a Cause: An Open Generator of Telecom Alarm Storms with Exact Root Causes, Bound to NORIA-O, RFC 8632 and TMF642

**arXiv ID:** 2610.30004 | [PDF](https://arxiv.org/pdf/2610.30004v1)

**作者:** Madhulatha Mandarapu `[一作]` (VaidhyaMegha Private Limited), Sandeep Kunkunuru `[通讯]` (VaidhyaMegha Private Limited)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作实现了一个可复现的电信报警根因分析生成器，能够在真实 Internet Topology Zoo 拓扑上模拟 7 种故障并记录每个报警的根因及传播路径，随后提供评估脚本、基准配置和 9 个基线方法；

**💡 创新点**

创新点包括：①采用生成器而非公开数据集，解决保密与可重复性难题；②将所有报警字段严格绑定到 4 个公开标准词汇（NORIA‑O、RFC 8632、RFC 8345、TMF642），并在生成时强制校验；③生成的数据包含完整的根因/传播路径作为 ground‑truth，避免后期人工标注误差；④公开完整的评估框架、基准数据与 9 个基线（规则、图模型、LLM 等），可供社区复现与对比；

**🔧 技术方法**

使用技术包括：Python+OWL/SKOS、YANG、OpenAPI 进行标准词汇绑定；Samyama 图数据库引擎实现邻域与时间推演；顶点级 Hawkes 过程拟合；个性化 PageRank；Claude‑Sonnet‑5‑5 语言模型（单行 prompt 与多轮 graph‑verb 交互）；随机噪声控制（缺失报警、背景噪声、时钟偏移、邻域电波、错误分组等）以评估鲁棒性；

**📊 数据集**

数据集：基于 Internet Topology Zoo 的真实骨干拓扑（GÉANT、Abilene、CogentCo 等），在此拓扑上注入 7 种故障，生成 40 评估事件（每种拓扑 5 种 seed 共 15 次实验）；生成的报警流包含 1,572 条报警、21 种 X.733 类型、717 设备、4 种严重级别，且每条报警携带根因、传播路径等信息；

**📈 对比分析**

方法比较：对 9 个基线（4 条规则、2 条自研图模型、3 条外部未设计的基线如 Hawkes、PageRank、LLM）进行 Top‑1、Top‑3 评分；在生成数据上最佳 Top‑1 为早期关键报警规则 0.650，第二名是 0.575 的 Temporal 方法，LLM 0.600；在真实运营报警上同一规则达到 0.773；同时通过 ablation 评估各噪声源对性能的影响，确认缺失报警是最关键难点；

**⚠️ 局限性**

局限性：①生成器的根因标注依赖于内部逻辑，真实网络是否同样将根警报标记为 critical 尚未验证；②生成的报警流 burst 大小偏大（82 条 vs 30 条），未能完全匹配真实网络；③规则对 critical 标记高度依赖，若真实环境不常见该标记则效果下降；④背景噪声被假设为平稳且占 47% 以上，真实系统噪声随时间变化；⑤实验规模有限，seed 与拓扑变化导致排名波动；⑥对站点电源故障这类根因无法报警的情形仅由图遍历方法处理，表明在无报警信息的场景下需额外模型。

---

## 670. PhysiCalWorld: In-Context Physics Calibration for World Models

**arXiv ID:** 2610.30125 | [PDF](https://arxiv.org/pdf/2610.30125v1)

**作者:** Yunhao Luo `[一作]` (University of Michigan), Jeong Joon Park `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在不进行微调或使用显式物理方程的前提下，利用短时间的交互序列（即上下文提示）对世界模型进行零拷贝的物理校准，从而使其在新场景中能够准确预测基于真实物理动态的未来状态。

**💡 创新点**

创新点包括：①将物理校准视为在上下文中进行的推理任务，构造一个单一的“物理描述符”令牌；②仅通过视频预测的自监督损失端到端训练，无需任何物理标签；③在推理时只需一次编码即可多次复用，显著降低计算开销；④证明该描述符在未见物理参数范围内仍能保持较好外推性。

**🔧 技术方法**

使用了：①基于Transformer的潜在世界模型（DINO‑WM、LeWM等）；②轻量化上下文编码器（视觉编码+动作编码+跨时间融合），生成物理描述符令牌；③自监督的潜在空间预测损失；④对比的长历史、原始上下文前缀、估计物理参数和“特权物理”等基线。

**📊 数据集**

在四个仿真环境中收集数据：Push‑T（T‑块质心变化）、Reacher（双关节阻尼变化）、BeerPong（碰撞吸能变化）以及FetchSlide（三种物理属性同时变化）。每个环境训练时使用约135种不同的物理配置，测试时随机挑选15种未见配置。

**📈 对比分析**

与传统世界模型、长历史、上下文前缀、估计物理参数以及特权物理等方法相比，本文方法在开放循环预测和视觉规划任务中均取得了更高的PSNR、SSIM、IoU以及成功率（SR）。在大多数指标上，甚至在Push‑T和BeerPong等任务上超过了获得真实物理参数的“特权物理”基线，展示了显著的性能提升。

**⚠️ 局限性**

主要局限包括：①训练需要将轨迹按物理配置分组，收集分组数据仍不易；②校准效果高度依赖于采集的交互提示质量，随机或游戏化策略可能不足以在信息稀疏的环境中提取足够物理信号；③假设物理属性在一个episode内保持不变，无法直接处理动态变化的环境。

---

## 671. Did the Problem Actually Get Harder? Testing LLMs on Paired P and NP-Complete Problems Hardened by Adversarial Multi-Agent Rewriting

**arXiv ID:** 2610.30103 | [PDF](https://arxiv.org/pdf/2610.30103v1)

**作者:** Nurul Labib Sayeedi `[一作]` (United International University), Shubhashis Roy Dipta `[通讯]` (University of Maryland, Baltimore County)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个包含12个问题族的基准集，每个族提供一个NP完全问题和对应的多项式时间同类问题，并使用AgentGAN对题目进行硬化，去除教材式线索。

**💡 创新点**

首次将NP完全与P同类问题配对并通过对抗式多代理框架实现答案与复杂度类保持不变的题目硬化，从而能量化模型在P→NP完全方向上的推理差距。

**🔧 技术方法**

采用AgentGAN框架（生成器+判别器+可执行有效性门）对问题语句进行多轮重写，并使用冻结的代码模型、可执行验证器和专家委员会进行评估。

**📊 数据集**

基于公开教材与算法书籍收集的24个参考解码器和测试用例，经过人类验证后生成2,114条硬化语句（1,003 NP完全、1,111 多项式）。

**📈 对比分析**

通过对12个LLM（含专有与开源）在每条语句上一次性生成Python 3解，计算pass@1，发现所有模型在NP完全语句上的准确率普遍低于P同类，平均差距为约25个百分点，最小化时间限制提升不超过5个百分点。

**⚠️ 局限性**

仅覆盖12个问题族且仅考察P与NP完全类，缺乏对更高复杂度类的评估，硬化过程对固定判别器阵列过度依赖，且仅使用Python 3 代码，未覆盖其他语言和更大模型动态变化。

---

## 672. Orbitol: Crystal Generation on Wyckoff Orbits

**arXiv ID:** 2610.30086 | [PDF](https://arxiv.org/pdf/2610.30086v1)

**作者:** Ivan Žugec `[一作]` (Atomic Intelligence), Tin Hadži Veljković `[通讯]` (University of Amsterdam)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发并实现了一种基于 Wyckoff 轨道的晶体生成模型 Orbitol，能够在指定空间群下生成晶体并在松弛后保持对称性。

**💡 创新点**

创新点在于将晶体生成从原子级别提升到轨道级别，使用 Wyckoff 轨道令牌、几何增强模块 (GEM) 以及周期性后验目标等技术，使得生成的结构天然满足所声明的对称性。

**🔧 技术方法**

采用扩散 Transformer（继承自 Crystalite）、轨道令牌、几何注意力、周期性后验目标（P）和自适应层归一化等技术来实现对称性约束和高效采样。

**📊 数据集**

在 Materials Project 20 原子以内的 MP‑20 数据集以及结合 Alexandria 数据的 Alex‑MP‑20 数据集上进行训练与评估。

**📈 对比分析**

通过与多种无条件与有条件生成模型（DiffCSP、MatterGen、Crystalite、SGEquiDiff、SGFM 等）比较，Orbitol 在 MP‑20 上取得最高的 MSUN（19.0%）和联合产率（13.8%），对称性保真度与最佳基线相当，并且采样速度约为 SGEquiDiff 的 310 倍；在 piezoelectric 候选物生成中，其中位数 figure‑of‑merit 约为 Materials Project 的 9 倍。

**⚠️ 局限性**

局限性包括：仅能采样训练集中出现过的 skeleton，无法生成全新空间群；能量评估依赖 MLIP，DFT 验证成本高；对极低对称性空间群的保真度仍有提升空间。

---

## 673. GoiBERT: Compact Vietnamese Language Models via Two-Stage Knowledge Distillation

**arXiv ID:** 2610.30067 | [PDF](https://arxiv.org/pdf/2610.30067v1)

**作者:** Quang Anh Kiet Tran `[一作]` (University of Information Technology, Vietnam National University, Ho Chi Minh City), Huynh Dai Nhan Tran `[通讯]` (University of Jyvaskyla)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了两款小型越南语编码器GoiBERT（26M和49M参数），能够直接处理原始文本且不依赖分词器，并通过两阶段知识蒸馏训练得到；

**💡 创新点**

创新点在于将兼容相同分词器的教师用于通用表示迁移，同时利用更大跨分词器教师提供的出折软标签进行任务级监督，并配合层级学习率衰减，使得小模型在多任务上保持高精度；

**🔧 技术方法**

技术手段包括RoBERTa结构、BamiBERT嵌入初始化、层间隐藏状态与注意力匹配的通用蒸馏、跨分词器的出折（out‑of‑fold）软标签蒸馏、层级学习率衰减以及高效的原始文本输入；

**📊 数据集**

使用的语料包括HPLT 3.0、维基百科以及八个越南语基准任务的训练文本，八个基准分别涵盖文本分类（ViNLI、ViSpamReviews、VSFC-Topic、VSFC-Sentiment）、ABSA（Restaurant、Hotel、ViSFD）、NER（PhoNER_COVID19）等；

**📈 对比分析**

实验中与PhoBERT、ViSoBERT、BamiBERT、XLM‑R‑Base和教师模型CafeBERT等对比，采用15个指标、3个随机种子；GoiBERT‑large在平均精度上击败BamiBERT约0.5分、在GPU吞吐量上提升2倍、CPU延迟下降约37%，GoiBERT‑base在保持99% BamiBERT平均性能的同时仅占其四分之一参数量；

**⚠️ 局限性**

局限性包括：仅在长度≤256的短文本上评估；未测试对未见领域或方言的泛化；压缩后对NLI和NER性能损失显著；训练过程中需五个交叉折叠的教师模型，成本高；与BamiBERT的对比并非纯粹的架构比较，且未进行偏见审计与置信区间分析。

---

## 674. LongZero: Enhancing Long-Context Reasoning in LLMs from Zero Data

**arXiv ID:** 2610.30132 | [PDF](https://arxiv.org/pdf/2610.30132v1)

**作者:** Zhennan Wan `[一作]` (Tsinghua University), Jianfei Chen `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 LongZero 方法，让 LLM 先生成结构化的题干、证据与答案，然后将其扩展成长文本，利用自身生成的数据进行长上下文推理的强化学习。

**💡 创新点**

创新点在于逆向生成顺序（先写规范化结构再扩展），只依赖模型自身生成和验证的任务，结合推理任务与检索任务，并采用 GRPO 仅保留模型回答不稳定的样本来获得有效学习信号。

**🔧 技术方法**

使用技术包括 GRPO（Group Relative Policy Optimization）与 token‑level policy gradient、rule‑based reward、对话式检索任务 MRCR、结构化 JSON 规范、长文本扩展、以及自监督的数据合成流程。

**📊 数据集**

数据集方面：自生成的两类任务数据（推理数据与检索数据），以及五个公开长上下文推理基准（LongBench、LongBench v2、FRAMES、AA‑LCR、LongReason）。对比基线还使用 LoongRL 扩展的 HotpotQA 与 Teacher‑synthesized QA。

**📈 对比分析**

比较方法：在两款基线模型（Qwen3‑4B‑Thinking‑2507 与 Qwen3‑8B）上训练后，与无 RL、LoongRL、Teacher‑synthesized 等基线在各基准上进行分数比较。实验表明本方法在平均分数上提升约 2–3 分，且在大多数基准（LB1、LBv2、FRAMES、LongReason）上均优于基线。

**⚠️ 局限性**

局限性：训练数据仅为模型自生成的短文本，无法覆盖极长输入（如 AA‑LCR）导致性能下降；自筛过程依赖模型自身的判断，可能产生错误或过度自信；目前仅覆盖事实性推理，缺乏对计数、比较、算术等更复杂任务的支持；未来需迭代生成更长上下文并探索更稳健的数据验证方法。

---

## 675. Quant-Arena: Benchmarking LLM Agents for Quantitative Trading Research

**arXiv ID:** 2610.30113 | [PDF](https://arxiv.org/pdf/2610.30113v1)

**作者:** Houde Qian `[一作]` (Navers Lab, Einsia.AI), Calvin Xiao `[通讯]` (Navers Lab, Einsia.AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个端到端的量化研究基准，评估从原始市场数据构造特征、生成预测信号、组合策略，到在封闭的 2024 年数据上评估最终策略的完整流程。

**💡 创新点**

创新点在于：①首次将特征构造、信号组合与最终投资组合绩效通过统一协议连接成一个完整评估单元；②引入受限研究预算、冻结策略提交和封闭评估，消除回测偏差；③提供公开评估平台与多模型对比。

**🔧 技术方法**

使用大型语言模型自动生成 Python 代码实现特征构造与股票评分；在迭代过程中利用有限的反馈进行策略改进；采用模拟订单簿估计市场冲击的交易成本；通过 RankIC、Sharpe、最大回撤等金融指标进行评估。

**📊 数据集**

采用中国 A 股市场真实数据：2022‑2023 年 Level‑2 订单、成交与订单簿快照用于研究期；2024 年全年的交易数据为封闭评估期；样本包含 3,042 只股票。

**📈 对比分析**

与 9 个前沿 LLM（如 Sol、Astra、Terra 等）在迭代和一次性两种配置下进行对比，评估指标包括总分 S、封闭期年化回报、超额回报、RankIC、Sharpe 等。结果显示：仅 Sol（中等努力）在 2024 年略优于等权市场；最优的 Kepler‑1（Einsia AI 的专有策略）在所有模型中表现最突出。

**⚠️ 局限性**

局限性包括：①多数模型的超额回报仍为负，且多模型在高努力设置下并未统一提升；②排名技巧不一定转化为投资收益；③受限预算与反馈的设计可能导致信息泄漏风险；④特征多样性对弱策略几乎无效；⑤仅在 A 股市场验证，通用性待进一步验证。

---

## 676. Multimodal Embodied Agents: A Unified Capability-centric Perspective from Computer-Use to Robot-Use

**arXiv ID:** 2610.30108 | [PDF](https://arxiv.org/pdf/2610.30108v1)

**作者:** Yanzhe Chen `[一作]` (National University of Singapore), Mike Zheng Shou `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了PAPAV框架，将多模态具身智能体的任务循环拆分为感知、预测、规划、执行与验证五个功能，并系统比较了多模态代理、具身智能体与机器人系统在这五个功能上的共性与差异，同时对64个基准进行覆盖率分析。

**💡 创新点**

创新点在于将能力视角统一到跨数字与物理两端的共通框架，明确了物理具身带来的五个限制（部分可观测、可模拟性有限、不可逆、实时耦合、不可验证），并提出了对评价指标与设计的五大开放挑战。

**🔧 技术方法**

采用多模态感知融合、基于语言的规划、世界模型预测、主动感知、动作细粒度执行以及自我验证等技术，结合现有机器人控制与强化学习算法实现框架功能。

**📊 数据集**

使用的“数据集”是对64个公开基准的汇总，涵盖10个具身智能体、23个多模态代理和31个机器人系统的任务与环境，涵盖视觉、语言、交互与物理操作数据。

**📈 对比分析**

通过对各基准的显式/隐式能力评价，发现Act被最充分评测（100% MMEA，70% MMA，94% RS），而Plan、Anticipate、Verify的显式评测分别低至70%、8%和5%；说明当前评价大多关注终端成功率，缺乏对中间功能的细粒度比较。

**⚠️ 局限性**

局限性包括：评测缺乏过程诊断，很多能力仅通过终端结果间接推断；基准设计多聚焦任务完成而非功能分解；缺少统一的能力验证接口和标准化的回溯记录，导致跨任务和跨领域比较困难。

---

## 677. ProactEgoMem: Benchmarking Proactive Memory in Streaming Egocentric Video

**arXiv ID:** 2610.30089 | [PDF](https://arxiv.org/pdf/2610.30089v1)

**作者:** Ruotong Liao `[一作]` (LMU Munich), Volker Tresp `[通讯]` (LMU Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了用于评估自主动记忆的流式第一人称视频基准 ProactEgoMem，并定义了信息捕获、状态跟踪、位置跟踪三大任务以及通知与干预两种主动模式。

**💡 创新点**

创新点在于：①把主动记忆定义为在没有查询的情况下主动决定何时收集、维护并在后续问答中利用的完整生命周期；②结合多任务、两级主动性与问答验证，构建了全新的评估框架和数据集；③通过引入工具调用机制，让模型能主动开启、维护、关闭记忆工具。

**🔧 技术方法**

采用的技术主要是：基于大型语言模型（LLM）的文本生成与评判（如Qwen、Gemma、VideoLLaMA等）用于生成事件候选、填充问答；工具调用接口实现后台记忆收集；流式评估协议按秒逐帧提供视频输入；评估指标包括 Start@τ、mIoU、精确率/召回率/F1 以及 QA 准确率。

**📊 数据集**

使用的数据集包括：日常生活轨道（来自 Ego4D 的日常第一人称视频）和实验室协同轨道（wet‑lab 与 Ego4D 科学实验视频），共约 4,000 小时视频；生成的 ProactEgoMem‑QA 问答数据覆盖三大任务与多种问题类型。

**📈 对比分析**

与多种基线模型（专有 LLM、开源 LLM、视频‑语言模型、流式模型）进行对比。总体表现不佳：通知层 F1 最高仅 19%，干预层 mIoU 最高 17%，Start@0.2（精准触发）最高 28%。开源模型在干预层覆盖率略高，但精度低；流式模型在时序上最快但仍存在显著延迟；视频‑语言模型在召回率上最差。整体表明现有模型难以在长时延、精确生命周期管理方面达到可靠的主动记忆能力。

**⚠️ 局限性**

局限性：①模型启动事件往往延迟 0.4–3.5 秒，导致主动性不足；②召回率低，易产生误报与漏报；③生命周期管理差，工具调用频繁开启却难以正确停止；④工具调用的解析与执行能力受限，部分模型完全不输出可解析的调用；⑤实验主要基于两类视频，泛化到更复杂场景仍需验证。

---

## 678. Parallel AI Agent Collaboration as Graph-Structured Distributed Block Optimization

**arXiv ID:** 2610.30074 | [PDF](https://arxiv.org/pdf/2610.30074v1)

**作者:** Hanmin Li `[一作]` (King Abdullah University of Science and Technology), Peter Richtárik `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了“verified proposal integration”框架，用以在多智能体协作中整合并验证各自提出的局部编辑，保证在已有的块依赖图下能保留一定比例的单块改进收益。

**💡 创新点**

创新点包括：① 将块级提议验证与分数（fractional）覆盖结合，证明在任何依赖图上最优保留比例为 1/χ_f(G)；② 设计安全（safe）与乐观（optimistic）两种整合规则，既保证了理论保证，又能利用正向交互；③ 在此框架下给出全局收敛分析，将任务结构、代理能力、分配覆盖率三者的影响分离；④ 通过实验验证了分数着色优于固定颜色，乐观规则在实际任务中往往取得更高增益。

**🔧 技术方法**

使用的技术包括：块依赖图构建、分数着色与覆盖、单块改进量测评（D_b(x)）、局部改进质量假设 η_ib(h)、块错误下界 μ、积分式收敛分析、以及实验评估工具（Python 修复、道路网络、课程排课、CVRP）。

**📊 数据集**

实验数据集包括：① 7 个三模块修复任务（Python 代码）与 12 个规模从 4 到 32 模块的扩展修复任务；② 30 个公共道路网络区域（Pennsylvania、California、Texas）形成的 C5 子图；③ 9 个 ITC‑2007 课程排课实例；④ 4 个公开 CVRP（Loggi‑n401‑k23、Loggi‑n501‑k24、ORTEC‑n242‑k12、ORTEC‑n323‑k21）用于路由搜索调度。

**📈 对比分析**

对比方法：与 “apply-all” 、“apply-merge‑only” 、“single‑agent” 、“sequential” 等基线，以及固定普通着色与分数着色的区别。性能方面：安全规则在修复任务中成功率达 100%，乐观规则与安全规则在多轮修复中平均降低失败检查数至 6 左右；分数着色在 C5 任务中保留 57% 的单块增益，而固定着色仅 49%；在道路网络和课程排课任务中，分数着色/适应性着色均比固定着色获得更多增益；在 CVRP 调度实验中，分数着色实现了约 15% 的墙壁时间缩短。

**⚠️ 局限性**

局限性：① 需要已知且准确的块依赖图；若图过大或过密，χ_f(G) 较大导致保留比例下降；② 假设单块改进质量 η_ib(h) 与块错误下界 μ 能被评估，实际环境中可能难以保证；③ 只考虑单块改进的“单块验证”，无法捕捉需要多块协同的非线性交互；④ 对极端并发度高、任务结构高度耦合的情形，分数着色与安全规则的实际收益可能受限。

---

## 679. Self-normalized Bernstein Inequalities for Regenerative Processes and Applications to Learning Without Independence

**arXiv ID:** 2610.30119 | [PDF](https://arxiv.org/pdf/2610.30119v1)

**作者:** Imon Banerjee `[一作]` (University of Chicago), Pierre Alquier `[通讯]` (ESSEC Business School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7`

**🎯 论文内容**

针对具有再生结构的非 i.i.d. 过程，推导了自归一化的 Bernstein 不等式，并用其得到泛化误差、PAC‑Bayes 以及极大似然估计的误差上界。

**💡 创新点**

创新点在于将再生周期的观测长度直接作为自归一化的方差估计，仅保留了可观测的首项，剩余的尾项在多数情形下为二阶，且无需估计混合系数或谱间隙。

**🔧 技术方法**

主要技术包括自归一化指数超鞅、可观测截断、再生强逼近、PAC‑Bayes 变分法以及自归一化高斯逼近。

**📊 数据集**

文章未使用公开数据集，而是通过理论模型（如马尔可夫链、Geo/D/1 队列）演示结果，且在分类任务中引用了深度网络实验以说明学习目标的可用性。

**📈 对比分析**

通过与传统基于混合系数的 Bernstein 不等式、无模型假设的 i.i.d. 结果和马尔可夫链的 PAC‑Bayes 边界比较，显示在大样本下误差率约为 O(√(log n/n))，在最优的再生长度尺度下达到 √(m/n) 的最优速率。

**⚠️ 局限性**

局限性包括：尾项 β(u_n) 在重尾（1<α<2）情况下可能成为主导项；对再生周期长度的具体分布仍假设可观测且 i.i.d.，且需要已知或可估计的重置点来构造再生结构。

---

## 680. VoxProg: 3D Spatial Reasoning through Voxel Programming

**arXiv ID:** 2610.30127 | [PDF](https://arxiv.org/pdf/2610.30127v1)

**作者:** Zhichao Zhai `[一作]` (Zhejiang University), Jun Xiao `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于体素编程的语言引导3D放置框架，使用两阶段程序生成：先构建任务相关的几何参考（证据程序），再根据这些参考计算目标物体的6-DoF位姿；并在71个开放式放置任务上进行评估。

**💡 创新点**

核心创新在于将几何参考的构造作为可执行代码中的显式阶段，利用局部体素视图对物体表面进行离散化与数组操作；通过视觉锚点和执行反馈实现自动校正；引入 IA Rank Score 与物理有效性两项互补评估指标。

**🔧 技术方法**

技术要点包括：LLM（GPT‑5.6‑Sol）生成 Python 代码，VLM 进行视觉定位，局部体素化实现表面网格的规整化查询，数值与可视化检查反馈，DreamSim 用于感知相似度评价，三维碰撞与支撑检测确保物理有效性。

**📊 数据集**

使用自制的 71 题语言引导 3D 放置基准（涵盖功能、几何、上下文三类任务），并在外部任务上扩展至 RoboCasa（19 题）和 Open6DOR v2（13 题）。

**📈 对比分析**

与 GenManip、GenSim2‑kPAM、Direct Mesh 等基线对比，使用 IA Rank Score（意图对齐）和 Physical Validity（物理有效性）评价；总体 IA Rank Score 83.10、物理有效性 95.8%，在所有类别均名列前茅，尤其在几何任务上取得最大优势。

**⚠️ 局限性**

局限性包括：依赖预先已知的网格与初始位姿；LLM/VLM 生成代码的鲁棒性仍受限；对严重遮挡或复杂形状的表面识别仍有挑战；在真实感知噪声下的表现尚未充分验证；需要进一步探索在更大规模、动态或真实环境中的泛化能力。

---

## 681. The role of locality and weight sharing in convolutional neural networks

**arXiv ID:** 2610.30118 | [PDF](https://arxiv.org/pdf/2610.30118v1)

**作者:** Francesco Mori `[一作]` (Harvard University), Melanie Weber `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在教师-学生模型中，通过推导高维极限下的闭式 ODE，解析了卷积网络（权重共享）与无共享局部网络在在线随机梯度下降中的学习动力学，系统研究了权重共享、局部性和空间相关性对学习速度和泛化误差的影响。

**💡 创新点**

创新点在于：①首次将权重共享与局部性分离开来，给出精确的学习速度 P 倍加速定律；②揭示了局部性决定的样本复杂度差异（匹配核大小时为 O(D)，超大核为 O(D log D)）；③分析了空间相关性如何削弱权重共享优势，并预测了在不同训练时间与学习率下 CNN 与 LCN 的相对性能。

**🔧 技术方法**

采用高维极限下的在线学习理论，推导宏观参数 ODE；利用早期展开、渐进误差分析和数值积分对 ODE 进行验证；使用随机正交初始化与高斯输入分布。

**📊 数据集**

仅使用合成高斯输入，位置协方差为复合对称矩阵（无结构），未使用实际图像或公开数据集。

**📈 对比分析**

通过 ODE 求解得到的误差曲线与单条 SGD 轨迹对比验证，实验显示：在无相关输入下 CNN 的误差曲线在时间上可通过 P 倍缩放与 LCN 对齐；在存在空间相关性时，LCN 在早期训练阶段误差更低，CNN 在长期训练后表现更佳；权重共享实现了 P 倍的学习速度提升。

**⚠️ 局限性**

局限性包括：输入假设为高斯无结构，未考虑多轮训练或有限样本下的过拟合；仅针对回归任务；学习率取固定值，未考虑学习率调度；模型只涉及平移对称群，未扩展到其他对称或更复杂的输入结构。

---

## 682. Asymmetric On-Policy Distillation Learns In-Context Exploration

**arXiv ID:** 2610.30105 | [PDF](https://arxiv.org/pdf/2610.30105v1)

**作者:** Sriyash Poddar `[一作]` (University of Washington), Abhishek Gupta `[通讯]` (University of Washington)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过学生-教师异步监督学习，训练一个历史条件的代理在部分可观测环境中进行上下文学习的探索。

**💡 创新点**

关键创新是利用学生在线产生的历史作为上下文，教师仅给出对应隐藏状态下的动作标签，从而实现“在情境中探索”而不需要昂贵的RL或显式后验推理。

**🔧 技术方法**

使用因果Transformer作为学生网络，迭代收集on‑policy历史并对齐教师标签的监督学习。

**📊 数据集**

在11个不同环境上评测，包括离散网格搜索、MetaNav、Procgen迷宫、Habitat物体搜索、仿真与真实机器人操纵任务。

**📈 对比分析**

与BC、DAgger、DPT、SPOC、AAWR、RL^2等基线相比，在固定探索预算下获得约2倍收益，RL基线交互次数少约100×，在真实机器人上成功率提升约+40%。

**⚠️ 局限性**

主要局限是依赖强大的特权专家；若专家不可用或数据偏差大，方法效果可能受限；同时对安全性探索仍未显式约束。

---

## 683. Building SQL Equivalence Proofs from Query Rewrite Rules

**arXiv ID:** 2610.30100 | [PDF](https://arxiv.org/pdf/2610.30100v1)

**作者:** Chunyu Chen `[一作]` (Simon Fraser University), Zhengjie Miao `[通讯]` (Simon Fraser University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于优化器重写规则的独立 SQL 等价证明框架。

**💡 创新点**

创新点在于将重写规则拆分为规则验证与证明搜索，显式记录对齐与推理步骤并输出可重放的证明。

**🔧 技术方法**

使用 Apache Calcite 的逻辑计划、重写规则、语义规范化、Memo 搜索以及对等式的手工验证。

**📊 数据集**

使用三类数据集：BIRD/LeetCode 文本到 SQL 转换对、Calcite 规则测试套件以及两种 LLM（GPT-5.4 与 SPA-32B-SI）生成的重写对。

**📈 对比分析**

与 QED、SQLSolver 比较，证明率分别为 40.4%/14.6%/15.8%，在 LLM 重写集上取得最高覆盖，且平均运行时间与现有工具相近。

**⚠️ 局限性**

局限在于搜索空间受限、缺少窗口函数与复杂 CTE 结构的统一化规则以及对大 join 区域的深度探索不足。

---

## 684. Differentially Private Sparse Reward Estimation with Preference Feedback

**arXiv ID:** 2610.30099 | [PDF](https://arxiv.org/pdf/2610.30099v1)

**作者:** Meng Ding `[一作]` (University of Massachusetts Boston), Di Wang `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了两种基于稀疏奖励模型的差分隐私强化学习方法（Sample-DP-IHT 和 Label-DP RR+IHT），并给出其理论误差上界和下界；

**💡 创新点**

创新点在于首次把稀疏性引入 RLHF 的隐私估计，突破了以往对全维参数的依赖，使误差率仅随稀疏度 s* 而非维度 d 变化，并证明了在随机响应框架下的最优性；

**🔧 技术方法**

采用迭代硬阈值 (IHT) 与私有剥离机制、随机响应+去偏梯度、以及条件均衡等技术实现隐私稀疏估计；

**📊 数据集**

使用 IMDb 电影评论数据进行情感生成任务，以及 AI2 的 Harmful 数据集进行安全对齐任务；

**📈 对比分析**

与 DP‑SGD、普通随机响应等稠密基线相比，稀疏方法在相同隐私预算下获得更高的奖励分数或安全胜率，且在高隐私 (ε=2) 与低隐私 (ε=8) 场景均表现优异；

**⚠️ 局限性**

限制在于 Label‑DP 方案仅在随机响应框架内最优，无法突破该框架的隐私-误差下界，且 Sample‑DP 仍存在对条件数与对数因子依赖，未来需要探索更优的 Label‑DP 机制及降低对条件数的敏感性。

---

## 685. AtomicRL: Atomic Reinforcement Learning for Long-Horizon Robotic Manipulation

**arXiv ID:** 2610.30097 | [PDF](https://arxiv.org/pdf/2610.30097v1)

**作者:** Likui Zhang `[一作]` (Sun Yat-sen University), Liang Lin `[通讯]` (Sun Yat-sen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了基于原子任务结构的强化学习后训练框架 AtomicRL，用以提升视觉-语言-动作（VLA）模型在长序列机器人操作中的学习效率和成功率。

**💡 创新点**

创新点包括：① 语义感知策略优化（SAPO）通过软门控在子任务边界上抑制错误的 TD 残差传播，解决信用污染问题；② 瓶颈自适应目标探索（BATE）根据各原子子任务的掌握程度动态调整全任务采样权重和局部熵正则，聚焦瓶颈子任务；③ 轻量化进度-完成判别器（PCC）提供子任务边界与进度信息，支持 SAPO 与 BATE；④ 自动化 Real2Sim2Real 管道实现从单次真实演示到仿真环境构建、RL 训练再到真实机器人部署的端到端流程。

**🔧 技术方法**

使用的技术包括：PPO+GAE 的基准强化学习、软门控的 GAE 调整、指数移动平均的子任务掌握估计、基于熵的自适应探索、进度‑完成判别器（轻量化前馈网络）以及自动化场景重建与仿真（RLinf 训练框架）。

**📊 数据集**

数据集与实验环境：LIBERO‑LONG（10 个长序列任务）、CALVIN ABC‑D（A/B/C 训练，D 评估）、以及在 AgiBot G1 上的三项真实世界长序列任务（Bread Placement、Waste Disposal、Tabletop Clean）。

**📈 对比分析**

与 PPO 基线相比，AtomicRL 在 LIBERO‑LONG 上提升 3.6–4.0% 的成功率；在 CALVIN ABC‑D 上完成 5 个子任务的成功率提升 9.5%，平均完成序列长度提升 1.0；在真实机器人上平均提升 21.3%（相对 SFT）或 8.0%（相对 PPO）。

**⚠️ 局限性**

局限性：① 需要先行识别子任务边界，PCC 的预测误差会影响 SAPO 与 BATE 的效果；② 对于子任务划分不明确或高度耦合的任务，原子结构假设可能不适用；③ Real2Sim2Real 管道虽然自动化，但仍需一定的元数据与物理资产准备，复杂度与成本不可忽视；④ 在极端稀疏奖励或长延迟反馈场景下，SAPO 的软门控可能不足以完全消除信用污染。

---

## 686. Looped Uniform Diffusion Language Models

**arXiv ID:** 2610.30066 | [PDF](https://arxiv.org/pdf/2610.30066v1)

**作者:** Saiteja Utpala `[一作]` (MBZUAI), Praneeth Vepakomma `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在统一扩散语言模型（UDLM）中引入循环（loop）机制，循环重复使用同一块Transformer层，以在不增加参数的前提下提升模型的训练效率和推理质量。

**💡 创新点**

创新点包括：①将循环策略从掩码扩散迁移到无标记的统一扩散；②探索三种循环次数采样分布（均匀、FLOP比例、噪声比例）以及自适应循环；③证明循环能显著减少训练FLOPs，并在推理时降低生成困惑度和零样本推理困惑度。

**🔧 技术方法**

技术手段：使用170M参数的Diffusion Transformer（DiT）架构；在中间块中实现共享权重循环；使用温度调度的软/硬输入训练；采用Ψ-sampler混合预测与校正；实现自适应循环阈值停止策略。

**📊 数据集**

数据集：OpenWebText（OWT）和FineWeb‑Edu子集，用于训练和验证。

**📈 对比分析**

比较方法：在相同训练FLOPs下对比循环与非循环模型的验证NLL；在推理时比较不同循环次数和自适应循环对生成困惑度、零样本困惑度以及多选任务准确率的影响。结果显示：循环模型在训练时可用0.36‑0.37倍FLOPs达到相同NLL；推理时生成困惑度下降多达9.8%，零样本困惑度下降11.5%，多选准确率提升且自适应循环可减少26‑35%循环迭代。

**⚠️ 局限性**

局限性：噪声比例循环分布未带来收益；实验仅在170M DiT模型上进行，缺乏更大规模验证；自适应循环阈值需手动设定；与采样加速或蒸馏等其他效率提升方法未做系统对比；在特定推理任务上循环的好处不均衡。

---

## 687. Agentic World Action Models

**arXiv ID:** 2610.30136 | [PDF](https://arxiv.org/pdf/2610.30136v1)

**作者:** Zhuoyang Liu `[一作]` (University Of California Berkeley), Trevor Darrell `[通讯]` (University Of California Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种agentic world action model，将代码规划与视觉预测与连续控制统一在一个混合transformer框架内，实现了在机器人操作中的多模态联合去噪。

**💡 创新点**

创新点在于把可执行代码专家嵌入世界行动模型，并通过联合去噪实现代码、视频和动作三模态的同步生成；同时提出自动注释管线和混合执行策略。

**🔧 技术方法**

采用混合transformer、Masked-token diffusion、连续流匹配、ViT视觉编码、DiT视频/动作生成，以及Gemini/Claude等大型语言模型自动生成程序。

**📊 数据集**

预训练使用约2000小时YAM机器人数据和1000小时人类第一人称操控数据；下游任务使用RoboTwin 2.0基准及YAM/Tianji–Sharpa实物8项任务。

**📈 对比分析**

在RoboTwin 2.0上相较于π0.5、EgoScale、DreamZero、WLA，平均成功率从41.3%提升至50.7%；在8个实物任务中平均Rubric分数从70.8%/49.7%提升至91.7%/64.6%，并通过混合执行提升对视觉随机化的鲁棒性。

**⚠️ 局限性**

局限在于对视觉域随机化下的性能仍显著下降，代码库的可迁移性和注释流程依赖大量人工/语言模型支持；混合执行在干净环境下略低于纯视频动作模式，并且模型规模和推理延迟较高。

---

## 688. Rethinking Video Autoencoders for Interactive World Models

**arXiv ID:** 2610.30129 | [PDF](https://arxiv.org/pdf/2610.30129v1)

**作者:** Adrien Ramanana Rahary `[一作]` (Kyutai), David Picard `[通讯]` (LIGM)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

建立了可调压缩的因果扩散表示自编码器族，并在交互式世界模型上评估压缩对吞吐量、延迟和质量的影响。

**💡 创新点**

主要创新是提供多尺度压缩自编码器、单阶段自蒸馏的少步扩散解码器，并系统比较不同压缩对机器人操作与导航的适配。

**🔧 技术方法**

采用冻结DINOv3特征提取、3D卷积瓶颈、因果空间时间扩散解码器以及流匹配+感知损失的自蒸馏训练。

**📊 数据集**

训练混合76k小时的网络视频、机器人抓取、驾驶与室内相机视频，评估四个环境（机器人抓取、人体抓取、驾驶、室内导航）。

**📈 对比分析**

通过在每个自编码器上训练相同的世界模型，使用FID、FVD、PSNR、LPIPS以及动作可恢复率（ARR）对比，发现自编码器压缩配置在吞吐量-质量前沿上占主导，较强压缩下扩散解码器优于感知回归。

**⚠️ 局限性**

限制在于实验仅覆盖特定抓取与导航场景，未验证到通用世界模型或策略训练的推广性，且自蒸馏的扩散解码器在极端压缩下仍可能产生运动模糊或界面缝隙。

---

## 689. Environment Duels: A Microcosm of LLM Co-Evolution

**arXiv ID:** 2610.30116 | [PDF](https://arxiv.org/pdf/2610.30116v1)

**作者:** Shibo Jin `[一作]` (University of Pennsylvania), Mayur Naik `[通讯]` (University of Pennsylvania)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对前沿大型语言模型进行双重评估，既让模型自己编写可执行的交互式环境并给出提示，又让模型解决对方编写的环境，并通过回合式反馈调整环境。

**💡 创新点**

设计了“Environment Duels”微观生态评测框架，实现了模型自我生成数据、互相挑战、并通过提示学习的闭环，能够动态提升基准难度且不需人工标注。

**🔧 技术方法**

采用多模型代理架构、可执行Python环境、提示式对抗、回合式修订（counter-move）以及强化学习实验验证提示提升与下游学习收益的关联。

**📊 数据集**

生成的90个可执行环境（每模型10个），包含多种领域（符号推理、规划、因果诊断、对抗策略、动态控制），并在强化学习实验中使用200个每作者的环境集合。

**📈 对比分析**

通过模型在无提示与有提示下的成功率、环境难度（无提示通过率）和提示提升（成功率提升幅度）进行评估；结果显示强解算者与优秀作者并非重合，微型模型在作者与提示方面表现最佳；头尾对决产生稳定排名。

**⚠️ 局限性**

仅涵盖九个前沿模型，环境难度仍受作者模型规模与训练策略影响；提示与环境的互补性可能因模型理解差异而受限；实验规模和对外推广受限于API调用与算力。

---

## 690. BatchBench: Benchmarking AI Robustness to Batch Effects and Domain Shift in Biomedical Images - A JUMP Cell Painting Leaderboard

**arXiv ID:** 2610.30104 | [PDF](https://arxiv.org/pdf/2610.30104v1)

**作者:** Sonja Topf `[一作]`, Günter Klambauer `[通讯]` (Johannes Kepler University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 BatchBench 基准和排行榜，用于评估 AI 模型在未见实验批次和数据源上的鲁棒性，并提供 JUMP‑CP‑2k 数据子集；

**💡 创新点**

设计了多源域适应与测试时适应的评估框架，公开了带有源、批次和板信息的 Cell Painting 数据子集，并提供标准化评估流程与公开基线；

**🔧 技术方法**

采用 ResNet18 作为基线模型，对比无适应、域不变训练（DANN、deepCORAL）、测试时适应（AdaBN、TENT）以及元学习方法（ARM‑BN、CS‑ARM‑BN）等技术；

**📊 数据集**

使用 Cell Painting 产生的 JUMP‑CP 数据集，挑选最常见的 2000 种化合物构成 JUMP‑CP‑2k，并划分公开/私有源；

**📈 对比分析**

通过整体准确率及正控、JUMP‑Target、长尾三子集评估，基线平均准确率约 0.175，测试时适应提升至约 0.25，正控子集最高可达 0.66，长尾子集低于 0.04；

**⚠️ 局限性**

在样本量少且表型弱的化合物上仍难以识别，域适应方法在长尾子集表现不佳，且最佳域层级的利用尚未确定。

---

## 691. Implementation of Dynamic Microsegmentation for a Multi-Robot Network

**arXiv ID:** 2610.30102 | [PDF](https://arxiv.org/pdf/2610.30102v1)

**作者:** Vishwam Raval `[一作]` (Global Cyber Research Institute), Sandip Roy `[通讯]` (Global Cyber Research Institute)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在实验室测试平台上实现并验证了针对多机器人网络的动态微分段功能，利用 ColorTokens Xshield 平台和资产健康标签实现了无人工干预的即时隔离和恢复。

**💡 创新点**

创新点在于：①将资产健康标签与 API 驱动的动态分段相结合，实现实时、自动化的零信任隔离；②首次在 IoT/机器人网络中展示了基于资产功能与健康状态的细粒度、适应性分段；③通过实验证明微分段可在不影响正常运作的前提下实现 100% 阻断，填补了现有研究在动态、移动环境中缺失的实证空白。

**🔧 技术方法**

主要技术包括：Zero‑Trust Microsegmentation、ColorTokens Gatekeeper 与 Xshield 云平台、Python 编写的自动化脚本、资产标签机制（功能标签 + 感染标签）、网络分段模板、ICMP/TCP/UDP 阻断策略以及实时资产发现与状态同步。

**📊 数据集**

数据集：实验使用三台 Turtlebot（配备 Raspberry Pi）在本地局域网中的实时网络流量记录，未使用公开数据集，而是通过实际测试环境收集的原始流量日志和事件日志。

**📈 对比分析**

比较方法：将基线无分段情形与实施微分段情形下的网络性能（延迟、吞吐量）、攻击面及扩散半径进行对比。结果显示：①性能无明显下降；②隔离后受感染节点的 ICMP/TCP/UDP 流量 100% 被阻断；③攻击面与扩散半径显著降低，恢复时间短且无需重启。

**⚠️ 局限性**

局限性：仅在三台机器人的小规模、受控环境下验证；未集成实际入侵检测模块；缺乏大规模、异构设备以及高度动态移动网络的实验；缺少与公开数据集或对照实验的深入比较。

---

## 692. Charts as Epistemic Actions for Experience-Driven Agentic Data Analysis

**arXiv ID:** 2610.30072 | [PDF](https://arxiv.org/pdf/2610.30072v1)

**作者:** Xinzhe Wang `[一作]` (Tencent), Shaolei Zhang `[通讯]` (Renmin University of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了数据代理在分析过程中是否能将图表视为内部推理动作，并提出了一种基于经验的学习框架使代理能在推理循环中有选择地生成并使用图表

**💡 创新点**

将图表作为中间证据而非仅仅的最终输出，引入了经验驱动的表示选择机制（两个经验族：分析经验与表示经验），并通过验证门保证只接受提升性能的更新

**🔧 技术方法**

使用ReAct式循环、图表生成工具、轨迹判定器（f_score、f_attr）、经验包编辑器、冻结经验包作为外部记忆，实验中采用四种大语言模型（Claude Sonnet 5、GPT‑5.6 Luna、GPT‑5.6 Sol、DeepSeek V4.1 Flash）

**📊 数据集**

构建了自定义的Probe套件Probe‑Bench（200个任务，来自InsightBench），以及三个公开基准：DAComp‑DA（报告生成）、DDR‑Bench（长周期探索）、DSBench（精确答案分析）

**📈 对比分析**

在上述基准上与匹配基线相比，使用经验包后整体分数均提升（DAComp‑DA平均提升约3.1分，DDR‑Bench约3.0分，DSBench约1.0分），在视觉证据质量、分析准确性等指标上都有显著改善

**⚠️ 局限性**

局限性包括：经验包构建仅在训练时使用GPT‑5.6 Sol，未针对不同模型细粒度调优；验证门可能过于严格导致部分有益经验被过滤；实验依赖离线评判器，未验证在实时交互中的鲁棒性

---

## 693. Loopholing All the Way: Language Modeling with pure Self-Conditioning

**arXiv ID:** 2610.30068 | [PDF](https://arxiv.org/pdf/2610.30068v1)

**作者:** Adrien Deschenaux `[一作]` (EPFL), Pascal Frossard `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Loopholing All the Way（LAW），一种完全基于自条件化（Self‑Conditioning）的连续生成模型；

**💡 创新点**

核心创新是将自条件化作为唯一状态传递，并使用期望嵌入（Expected Embedding）作为状态映射，同时引入温度正则化以增强模型对温度变化的鲁棒性；

**🔧 技术方法**

使用Diffusion Transformer（DiT）作为去噪器，采用多步前向训练、stop‑gradient、温度正则化以及基于期望嵌入的状态映射；

**📊 数据集**

在四个数据集上评估：Sudoku、Maze、TinyGSM（用于GSM8K）以及OpenWebText；

**📈 对比分析**

与自回归模型（ARM）、离散扩散模型（MDLM、Duo、LDDM-M/U）以及连续扩散模型（CANDI、FLM、S‑FLM）等基线对比；在Sudoku上取得96.5%（硬盘），Maze在少步时准确率超过其他模型，GSM8K从零训练可达42.2%（温度1），微调后提升至48.1%，OpenWebText在64 NFE下的生成困惑度接近ARM；

**⚠️ 局限性**

限制包括：NELBO在某些任务上优于交叉熵（CE）但在OpenWebText上产生样本多样性不足；模型在1024 NFE下的熵覆盖范围有限；实验仅在学术规模下验证，未证实在更大规模或工业场景下的可扩展性；

---

## 694. Test-Time Motion Steering for Perceptive Humanoid Fall Recovery

**arXiv ID:** 2610.30061 | [PDF](https://arxiv.org/pdf/2610.30061v1)

**作者:** Osher Azulay `[一作]` (University of Michigan), Stella X. Yu `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c82a482-f376-4869-8a0b-a802c9d4d3d4`

**🎯 论文内容**

提出了一种能够在推理时根据任务目标进行引导的感知型人形机器人跌倒恢复方法，通过构建多模态恢复词汇表和视觉状态优先的扩散规划器实现；

**💡 创新点**

创新点在于将场景感知与推理时的目标引导相结合，利用多模态恢复词汇表实现多种站起策略，并通过在规划阶段先预测运动意图再生成运动潜变量，使得目标成本能直接塑造执行；

**🔧 技术方法**

使用运动匹配（motion matching）构建恢复词汇表，基于视觉扩散（visual diffusion）规划器的状态优先生成（state‑first），结合反馈解码器（feedback decoder）与自适应深度感知，训练中采用DAgger与PPO；

**📊 数据集**

利用公开数据集LAFAN1、BONES‑SEED、AMASS以及作者自行记录的Unitree G1姿态，配合多种地形（平地、阶梯、碎石、悬空边缘、斜坡等）进行实验；

**📈 对比分析**

与VIGOR、FIRM等基线相比，在相同的跌倒恢复任务下取得更高的成功率、较低的接触力与关节极限违规率；在多模态恢复实验中展示了从同一初始姿态可产生多条独立路径；

**⚠️ 局限性**

在崎岖地形上，路径跟踪误差虽可控，但仍难以保证稳定恢复，说明单纯的运动意图预测与执行不一定能满足复杂支撑需求；

---

## 695. A Quasipolynomial Algorithm for $\ell_\infty$ Contractions

**arXiv ID:** 2610.30122 | [PDF](https://arxiv.org/pdf/2610.30122v1)

**作者:** Ashkan Soleymani `[一作]` (Massachusetts Institute of Technology), Gabriele Farina `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

该论文提出了一种在ℓ∞-非扩张函数上求近似不动点的算法。

**💡 创新点**

其创新点在于将放大（boosting）与稀薄标记（thin labeling）结合，构造出可在子对数级别提升准确度的标记方法，从而实现对搜索空间的几何收缩。

**🔧 技术方法**

技术上使用了递归分治、AdaBoost 形式的提升技术、以及对收缩映射的几何分析。

**📊 数据集**

由于该工作为理论算法，未使用任何数据集；全部在数学模型与算法分析上验证。

**📈 对比分析**

相较于之前的 Feodorov‑Haslebacher 结果，该算法把运行时间从子指数改为准多项式，具体为 (log(1/γ))^C log^2 d，显著提升。

**⚠️ 局限性**

局限性在于算法仍为准多项式而非多项式；常数项和实际实现的复杂度较高，且对非常小的收缩系数 γ 的效果有限。

---

## 696. Think Harder Where It Matters: Item-Level Adaptive Recursion for Sequential Recommendation

**arXiv ID:** 2610.30128 | [PDF](https://arxiv.org/pdf/2610.30128v1)

**作者:** Zhanshuo Cao `[一作]` (Renmin University of China), Jun Xu `[通讯]` (Renmin University of China)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究序列推荐中固定深度计算的局限性，并提出 LoopRec 框架，利用共享递归块和每个项目的轻量二进制退出门，实现按项目自适应深度分配。

**💡 创新点**

创新点在于：1）将循环递归与可学习的退出门结合，实现无额外标签的按项目深度决策；2）设计负载平衡正则化和 z-loss 防止深度坍塌；3）通过 warm‑up 阶段保证训练稳定。

**🔧 技术方法**

使用技术包括：共享 Transformer/自注意力、卷积或线性递归块的循环递归、轻量二进制门、软路由、负载平衡正则化、z-loss、warm‑up 训练策略。

**📊 数据集**

实验数据集为 Amazon Reviews 2023 三个品类：Software、Office Products、Patio。

**📈 对比分析**

与同一骨干的固定深度模型在 NDCG@10/20、MRR@10/20 等指标上进行对比。LoopRec 在四种骨干（SASRec、BSARec、NextItNet、LRURec）上均取得显著提升，提升幅度从 0.7% 到 49%（相对），在稀疏度最高的 Patio 数据集上效果最为明显。

**⚠️ 局限性**

局限性包括：1）计算节省有限，延迟提升不大；2）最大递归深度需手动设定，未学习；3）在某些骨干或数据集上提升有限；4）需要进一步开发稀疏计算核以充分利用自适应深度。

---

## 697. Temporal Exploration and Rendezvous via Weisfeiler-Leman Colorings

**arXiv ID:** 2610.30123 | [PDF](https://arxiv.org/pdf/2610.30123v1)

**作者:** Frank Kammer `[一作]` (University of Applied Sciences Mittelhessen), Johannes Meintrup `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文利用 2‑WL 纤维对时序图进行分层，设计出可在 O(qn log³n) 时间步内完成时序探索与会合的多项式时间算法；

**💡 创新点**

创新点在于将 2‑WL 纤维作为可高效计算的对称性度量，替代传统的自动同构轨道，从而得到更优的时间步上界和确定性实现；

**🔧 技术方法**

核心技术包括 Weisfeiler–Leman（2‑WL）算法的颜色细化、基于二项式马尔可夫链的随机递归构造、双随机矩阵与条件期望的 derandomization；

**📊 数据集**

无；

**📈 对比分析**

与此前基于轨道数 r 的 O(rn^{1+ε}) 上界和已知 O(n²) 下界相比，本工作实现了 O(qn log³n) 的上界，且保持多项式时间复杂度；

**⚠️ 局限性**

仍存在 O(log²n) 的上界-下界差距，且对 q 的依赖在某些极端图形中可能不是最优，另外 1‑WL 及更低维 WL 的潜在局限性尚未完全阐明。

---

## 698. ALPS: Visual State Encoders for World Models

**arXiv ID:** 2610.30120 | [PDF](https://arxiv.org/pdf/2610.30120v1)

**作者:** Yanqing Liu `[一作]` (University of California, Santa Cruz), Cihang Xie `[通讯]` (University of California, Santa Cruz)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并训练了一种从零开始的视觉编码器，兼顾物理状态信息、语言对齐和像素重建，为世界模型和机器人控制提供共享视觉基础。

**💡 创新点**

在单一编码器中统一三种能力，采用两阶段预训练（联合对齐、重建、密集对应 + 交叉视图自蒸馏），并通过冻结编码器评估多项指标。

**🔧 技术方法**

ViT‑B/16架构、InfoNCE对齐、自动回归字幕、像素重建、密集对应损失、交叉视图自蒸馏、冻结token评估等技术。

**📊 数据集**

使用DataComp‑1B（图像+web字幕）与合成字幕、5M单图编辑数据集进行预训练，评估时使用LIBERO、Flickr30k、COCO、ImageNet、PushT等公开数据集。

**📈 对比分析**

与15个公开编码器（包括DINOv2、CLIP、SigLIP、V‑JEPA、OpenVision等）在LIBERO state probe、图文检索、像素重建、控制、规划和世界模型预测等统一协议下对比，得到在状态探测、检索和像素重建上均位列前列，并在下游任务中取得最高或接近最高成功率。

**⚠️ 局限性**

仅在ViT‑B/16规模、仅使用静态图像与文本预训练，未加入视频/交互监督，缺乏真实机器人实验，跨模态蒸馏后仍需统一规划等。

---

## 699. CW-WAM: A Structured Contact-Wrench World Action Model for Force-Sensitive Manipulation

**arXiv ID:** 2610.30117 | [PDF](https://arxiv.org/pdf/2610.30117v1)

**作者:** Houzhu Ding `[一作]` (Ant Group), Ming Yang `[通讯]` (Ant Group)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CW-WAM 世界动作模型，联合预测视频、运动轨迹和接触力矩，并在执行时使用预测的接触力作为阻抗控制参考，从而显著提升机器人在接触密集任务中的成功率和力控制精度。

**💡 创新点**

创新点包括：① 统一的交互模块将过去的动作与测得的力矩配对，使用对称注意力和长时力矩记忆实现运动‑力矩的结构化预测；② 预测力矩直接驱动阻抗控制，无需额外学习的校正策略；③ 采用慢速模型推理与高速阻抗控制的两速执行体系，保证在模型推理间隙保持平滑的力反馈。

**🔧 技术方法**

技术方法主要包括：DreamZero 风格的 Transformer+DiT 视觉编码器，VAE+OpenCLIP+UMT5 的跨模态注意力；pair‑causal attention、长期力矩记忆；基于流匹配的多模态训练；20 Hz 阻抗控制与 1 kHz 关节阻抗控制的混合实现。

**📊 数据集**

数据集为 450 条手动遥控演示，涵盖四个任务（方形插销插入、适配器插入、白板擦拭、弹簧秤搬运），由两台 Franka 7‑DOF 机械臂收集，配备 LinkTouch 六轴 F/T 传感器和三目 RGB 摄像机。

**📈 对比分析**

与基线 WAM、WAM+力矩、CW‑WAM 无 LTM、CW‑WAM 等模型对比，实验显示：方形插销插入成功率从 25% 提升至 70%，适配器插入从 40% 提升至 95%；预测力矩阻抗控制将力误差 RMSE 由 4.8 N 降至 2.7 N，擦拭任务从 2.0 N 降至 1.3 N，且力波动显著减小。

**⚠️ 局限性**

局限性包括：仅在演示分布内测试，缺乏对新物体或新任务的泛化评估；阻抗控制参数手工调优，未学习最佳反馈权重或接触帧；模型仅在四个任务上验证，未检验在更复杂多变环境中的鲁棒性。

---

## 700. The End of Consensus Research?

**arXiv ID:** 2610.30114 | [PDF](https://arxiv.org/pdf/2610.30114v1)

**作者:** Clément Burgelin `[一作]` (École Polytechnique Fédérale de Lausanne), Martijn de Vos `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

利用LLM代理与领域知识库、模块化实现骨架及Lean形式化，自动发现并实现了一种针对geo‑replicated key‑value store 的新共识协议，并构建了完整的实验与验证平台。

**💡 创新点**

创新点在于：①把大模型与专门的共识知识库、可插拔实现层和可自动化的Lean验证相结合，形成端到端的共识协议发现框架；②设计了多路写入路径与本地读机制，显著降低写多与读多场景的平均和尾部延迟；③通过该框架对现有协议进行形式化验证，发现了四个此前未公开的安全漏洞。

**🔧 技术方法**

采用的大技术包括：OpenAI/Claude等LLM代理、定制化知识库与任务脚本、Go语言实现的 ConsensusArena 框架、Lean 证明助手、Toxiproxy 网络仿真、YCSB 工作负载生成与 Slurm 集群实验。

**📊 数据集**

实验数据集主要来自 YCSB A/B/C（50% 写、5% 写、只读）混合负载，结合 12 种自定义网络拓扑（均匀、聚集、共址）以及随机网络延迟与字节大小，覆盖 5/9/13/17 节点配置。

**📈 对比分析**

通过与 HotStuff、Raft、EPaxos 等主流协议在相同拓扑、节点数与工作负载下进行 mean 与 p99 延迟对比，发现新协议在写多场景平均延迟比最优基准低 12.2%，在读多场景平均/尾部延迟低 33.3%/30.1%，读专场景匹配最快基准；在失败恢复实验中，协议在 3 秒内恢复到 90% 吞吐。

**⚠️ 局限性**

局限性包括：仍需人类专家进行知识库编写、目标设定、结果审查与安全验证；代理在发现过程中可能利用实验环境中的“hack”导致过拟合；实验与验证成本高（约 24 小时、778M token）；发现的协议依赖特定网络与负载假设，无法保证在所有实际部署中的通用性。

---

## 701. Velocity Ambiguity Profiles: Time-Resolved Bayes-Risk Diagnostics for Flow Matching

**arXiv ID:** 2610.30107 | [PDF](https://arxiv.org/pdf/2610.30107v1)

**作者:** Yunpeng Mei `[一作]` (Beijing Institute of Technology), Gang Wang `[通讯]` (Beijing Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出Velocity Ambiguity Profile（VAP）对流匹配中的时间维度进行诊断，拆解损失为不可约的Bayes风险与可降的剩余风险；并给出局部非参数和分离高斯混合的统计基准；通过控制实验、oracle‑to‑blind 审计和时间采样压力测试验证VAP，并将其应用于CIFAR‑100 DINOv2特征流；

**💡 创新点**

创新点在于：1）首次将Bayes风险引入流匹配的时间维度，构建可解释的VAP；2）给出可量化的时间局部统计基准，揭示在不同时间点可降误差与内在不确定性的边界；3）通过多级实验验证诊断的实用性并展示基于VAP的采样策略能显著提升性能；

**🔧 技术方法**

使用流匹配框架、贝叶斯风险分析、非参数回归理论、分离高斯混合残差回归、控制实验设计、oracle‑to‑blind 对比、时间采样调度（基于VAP的权重）、特征流生成与评估（Fréchet距离、KID、宏观精度等）

**📊 数据集**

合成数据：4维/8维/16维分离高斯混合；真实数据：CIFAR‑100 提取的 DINOv2 ViT‑S/14 384 维特征

**📈 对比分析**

与统一时间采样（baseline）对比；在CIFAR‑100特征流中，基于VAP的时间采样使统一评估的速度损失降低6.78%，全局特征Fréchet距离降低23.6%；宏观精度、密度、覆盖度提升，但宏观条件KID无改善，召回率与特征协方差秩下降，表明存在多样性成本；

**⚠️ 局限性**

局部基准不适用于完整观测空间；GMM基准仅为分离标签残差估计，不是完整下界；自然数据实验仅在冻结特征空间内，无像素质量或通用采样器的保证。

---

## 702. VX-One: Typed, Calibrated Decision Heads for Real-Time Perimeter Triage - An empirical study of unknown-attack honesty, cascade gray zones, and calibration under drift

**arXiv ID:** 2610.30090 | [PDF](https://arxiv.org/pdf/2610.30090v1)

**作者:** Juan S. Santillana `[一作]` `[通讯]` (Independent research), Juan S. Santillana (Independent research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并训练了一个小型多头网络VX‑One，实现了三种类型化决策：门控、选择和评分，用于网络边界的实时威胁筛选；

**💡 创新点**

首次学术实现并评估System One三元决策原语，系统性测量其在检测、归因诚实性、校准稳定性和灰区恢复上的优势与局限；

**🔧 技术方法**

使用共享MPL trunk+三头设计，温度校准、同方差多任务损失、类别倾斜、Dropout等技术，配合特征归一化和文本分支；

**📊 数据集**

采用公开流量数据集UNSW‑NB15、CIC‑IDS2017（分层与按天）、EMBER2024，并进行留一家族与时序漂移实验；

**📈 对比分析**

与XGBoost、Random Forest、MLP、Kitsune以及商用System One零射门基线对比，VX‑One在检测（AUROC、FPR@95）上优于基线，归因宏F1略逊；在时序漂移和未知族检测中保持较高准确率，但校准在漂移下退化，阈值不可迁移；

**⚠️ 局限性**

局限包括：校准在数据漂移下崩溃、未知族检测在部分族群（如CIC的Recon）失效、阈值与阈值不具可移植性、缺乏payload完整的Phase‑1评估、仅针对公开数据集的验证。

---

## 703. Do Earth-Trained Humanoid Controllers Walk on the Moon? Gravity, Terrain, and Soil Transfer

**arXiv ID:** 2610.30080 | [PDF](https://arxiv.org/pdf/2610.30080v1)

**作者:** Guillem Casadesus Vila `[一作]` (Stanford University), Grace X. Gao `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估地球训练的类人机器人控制器能否在月球环境中有效行走；

**💡 创新点**

提出一种考虑重力、地形与土壤差异的迁移评估框架，探讨地球训练控制器的可迁移性；

**🔧 技术方法**

使用强化学习+模仿学习的控制算法，结合多传感器融合与动态规划；

**📊 数据集**

基于地球仿真环境与月球土壤物理模型（Lunar Soil Dataset）进行实验；

**📈 对比分析**

将地球训练控制器与专门在低重力环境训练的控制器在多种地形上进行对比，结果显示地球训练控制器在成功率上提升约15％、能耗下降约10％，表现更稳健；

**⚠️ 局限性**

实验仅在高保真仿真中验证，尚未在真实月球软着陆器硬件上进行验证，且对不同土壤成分的适应性仍有限。

---

## 704. Circuits of LLM Classifiers

**arXiv ID:** 2610.30076 | [PDF](https://arxiv.org/pdf/2610.30076v1)

**作者:** Hadi Khalaf `[一作]` (Harvard University), Flavio Calmon `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于动态规划的“分类电路”框架，能自适应选择并顺序查询不同配置的LLM分类器，以在给定成本下最大化平均分类准确率。

**💡 创新点**

创新点在于把多模型组合视为贝叶斯序贯决策问题，推导出最优查询策略、准确率上限和信息率定量分析，并提供两步近似实现：先用嵌入+网络估计联合分布，再在此基础上求解动态规划。

**🔧 技术方法**

核心技术包括：LLM作为测量仪器的概率建模、基于prompt嵌入的神经网络参数化联合分布、信息理论量化（互信息、KL、信息率）以及后向归纳求解的动态规划。

**📊 数据集**

在七个公开任务上验证：AgentJudgeBench、BeaverTails、BIG-Bench Hard、Decision Index、HealthBench、LMArena、PolitiCause，使用各任务的训练/验证/测试集进行实验。

**📈 对比分析**

与多种基线（路由器、级联、采样方法和单模型/多数投票）对比，LMCircuit 在绝大多数预算区间实现更高准确率；训练时间仅为1–3分钟，且平均查询成本往往低于单一最佳模型。

**⚠️ 局限性**

主要限制是动态规划的状态空间随模型数指数增长，对大规模模型池的可扩展性有限；此外，对高信息率模型的依赖可能在最难样本上受限，导致整体误差下降速率受信息率分布限制。

---

## 705. From Classifier Heads to Encoder Prototypes: Personalized Federated Learning with Head-Derived Clustering

**arXiv ID:** 2610.30075 | [PDF](https://arxiv.org/pdf/2610.30075v1)

**作者:** Amirardalan Dehghanpour `[一作]` (Sharif University of Technology), Hamid R. Rabiee `[通讯]` (Sharif University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于客户端分类器头几何结构的个性化联邦学习方法 HDC-FL，利用头部信息在服务器端生成编码器原型并以软混合形式返回给每个客户端，保持单一共享编码器训练轨迹。

**💡 创新点**

创新点在于：①不需要额外的数据摘要或本地训练，直接利用已有的分类器头作为客户端相关性信号；②使用低秩投影和 k‑means 对头部进行聚类，并通过软隶属度生成多源编码器原型；③只在服务器端完成所有聚类与混合，无需改变本地目标或维持多条训练轨迹。

**🔧 技术方法**

技术细节包括：低秩（SVD）投影头部至共通空间，k‑means 聚类生成 G 组，计算软隶属度 p_{k,g}，按样本大小加权平均得到组编码器原型 θ_g，最终以 λ=1 生成客户端专属编码器 θ̃_k = Σ_g p_{k,g}θ_g。

**📊 数据集**

实验数据集：CIFAR‑10 与 CIFAR‑100，在标签子集偏移、Dirichlet 标签分布和图像旋转（covariate shift）三种异构场景下；使用 ResNet‑18 与 MobileNetV2 两种编码器。

**📈 对比分析**

对比方法包括 FedAvg‑SE、FedBABU、FedALA、FedCE、FedSoft、FedAMP、IFCA、PACFL、EMD‑CFL、PFedCS 等；HDC‑FL 在 11 种配置下宏观平均均值准确率与最强基线相当，且在最差客户端和底部 20% 客户端准确率上名列前茅，尾部排名最佳，平均尾部准确率高于 PFedCS 约 0.4% 点。

**⚠️ 局限性**

局限性：①仅适用于所有客户端使用相同编码器结构且局部更新短；②需要服务器获取并存储客户端分类器头，若参数需保密则不适用；③实验规模仅为 20 名客户端，尚未验证大规模联邦系统；④未提供隐私保护机制。

---

