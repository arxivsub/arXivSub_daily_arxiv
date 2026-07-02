# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-07-01 | 今日论文总数: 646

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. ELEVATE: Designing Human-Centered GenAI Virtual Tutors for Scalable and Inclusive Education

**arXiv ID:** 2606.30662 | [PDF](https://arxiv.org/pdf/2606.30662v1)

**作者:** Lorenzo Stacchio `[一作]` (University of Macerata), Emanuele Frontoni `[通讯]` (University of Macerata)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了 ELEVATE 框架——一种本地先行、多模态、教师可治理的 LLM 虚拟教师系统；在真实学校环境中部署并评估了其性能与可用性。

**💡 创新点**

创新点包括：①三层闭环架构将学生交互、LLM 推理与教师治理分离；②通过本地推理与流式输出实现低延迟、离线运行；③嵌入 3D 形象与声学同步，提升人机交互的自然度；④以教师可配置的系统提示、RAG 与文档管控实现可审计、可定制的知识边界。

**🔧 技术方法**

技术实现采用 Unity + C# + .NET (前端)、Python + llama.cpp + Hermes‑3B 量化模型 (后端推理)、Coqui‑TTS 本地语音合成；通过 WebSocket/加密传输、ASR 与 TTS 分离、流式生成、队列化调度实现异步高效推理。

**📊 数据集**

使用自定义领域数据（历史计算机科学教材与课堂笔记）对 Hermes‑3B 进行 QLoRA 微调，并构建教师选定的知识库做 RAG；未使用公开大规模对话或多模态评测数据。

**📈 对比分析**

与传统云端 ChatGPT/大模型对话的比较：在本地推理下，首音频延迟约 5 秒，整体响应时间低于 10 秒；LLM 推理耗时仅占总延迟 20% 以上，TTS 为主导。实验表明在标准 PC 与安卓手机上均能实现近实时、无显著卡顿的交互。

**⚠️ 局限性**

局限性包括：①语音合成在低配 GPU 上占主导，导致高峰延迟波动；②未实现跨会话持久化与大规模并发时的队列瓶颈；③模型仍可能产生幻觉与误信息；④依赖教师手动配置和文档管理，使用门槛仍高；⑤缺乏正式的学习成效评估与安全监管机制。

---

## 2. Improving Survey Participation in Low-Literacy Populations Through Value-Sensitive Conversational AI

**arXiv ID:** 2606.30660 | [PDF](https://arxiv.org/pdf/2606.30660v1)

**作者:** Raj Gaurav Maurya `[一作]` `[通讯]` (Technical University of Munich), Raj Gaurav Maurya (Technical University of Munich)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对印度低识字女性开展问卷调查，比较纸质访谈、移动网页表单、语音和价值敏感对话式AI等六种交互方式，发现价值敏感对话式AI在完成率上显著优于传统方式。

**💡 创新点**

将AI对话与当地文化、语言和价值观相结合的分层价值敏感设计，并与社区志愿者合作制定，从而显著提升参与度与完成率。

**🔧 技术方法**

使用基于GPT‑4o‑mini的语音识别与校验、ElevenLabs多语音合成、Vapi AI语音平台以及React前端等技术。

**📊 数据集**

对315名印度农村低识字女性进行现场试点，覆盖四个地区（乌塔尔普拉德什和比哈尔）。

**📈 对比分析**

采用非随机志愿者分配的准实验设计，比较各模式的平均完成率与逐题保留率，层层价值敏感对话式AI平均完成率为0.89，最高；纸质访谈为0.46。

**⚠️ 局限性**

样本量有限、非随机分配导致潜在混杂、未评估回答质量与真实性、未公开AI身份等限制。

---

## 3. Probing-Guided Layer Selection from Self-Supervised Speech Models for Generalizable Audio Deepfake Detection

**arXiv ID:** 2606.30791 | [PDF](https://arxiv.org/pdf/2606.30791v1)

**作者:** Marjan Beheshti `[一作]` (Michigan Technological University), Bo Chen `[通讯]` (Michigan Technological University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种两阶段模型无关的深度层选择方法，先用XGBoost探测器对冻结的自监督语音模型的每一层进行跨域判别力评估，随后用压缩的神经分类器仅融合选出的层级特征进行音频深度伪造检测。

**💡 创新点**

创新在于先在下游任务训练前通过轻量探测器得到每层跨域判别力排名，识别信息丰富的深度区域；并通过注意力池化+共享瓶颈投影实现仅用少数层即可达到高性能；实现了模型无关、低参数、跨域鲁棒性的深度伪造检测框架。

**🔧 技术方法**

使用XGBoost轻量探测器进行层级评估；多头注意力池化、层归一化、共享瓶颈投影；冻结的自监督语音模型（XLS-R、WavLM、XLSR）；数据增强（RIR、裁剪、噪声）；Adam优化，早停；交叉验证。

**📊 数据集**

训练使用ASVspoof 2019 LA；跨域评估使用In-The-Wild、ASVspoof 2021 DF、FakeAVCeleb、WaveFake、ASVspoof5 Eval；还参照ASVspoof5 Dev作对照。

**📈 对比分析**

与基线Xiao & Vu（使用相同骨干但所有层决策级融合）、MLDG-LoRA、Tran等方法对比，采用相同训练数据的交叉域平均EER。该方法在In-The-Wild实现4.94% EER，比Xiao & Vu提升28%；跨域平均EER为4.81%，显著低于其他基线。参数量仅1.34M，远低于全层或微调方案。

**⚠️ 局限性**

仅在ASVspoof 2019 LA 训练集探测，可能随新攻击变迁；对抗性攻击仍显著提升EER；探测仅使用统计池化特征，未考虑更复杂表示；未自动化选择层数；在某些背骨上选取的层区可能因预训练差异而变化。

---

## 4. Protecting Futures against Silent Data Corruption -- Efficient Task Replication for Dynamic Data Dependencies

**arXiv ID:** 2606.30771 | [PDF](https://arxiv.org/pdf/2606.30771v1)

**作者:** Rüdiger Nather `[一作]` (University of Kassel), Mia Reitz `[通讯]` (University of Kassel)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对动态任务和基于 future 的依赖的 AMT 程序的 SDC（Silent Data Corruption）防护方案，利用双子复制计算并对跨任务的操作进行交叉验证，及时检测并修复错误。

**💡 创新点**

创新点在于：①在动态任务环境下仅对产生外部影响的操作（任务生成、promise 写入、promise 创建与 future 触摸）进行记录与确认；②通过在发现不匹配时仅重执行受影响任务而非整条子图；③在集群环境下结合 work‑first 工作偷窃与统一地址空间实现高效的跨节点同步。

**🔧 技术方法**

采用了 Asynchronous Many‑Task（AMT）模型、Futures‑Based Coordination（FBC）编程模型、work‑first 工作偷窃调度、ItoyoriFBC 运行时系统以及 MPI 通信实现的统一地址空间。

**📊 数据集**

实验使用两组基准：递归 Fibonacci（n=62，阈值32）和仿真层次 LU 分解（高度7，总忙等待时间100s）。

**📈 对比分析**

与未加保护的基线对比，失败-free 保护开销约为 80%（Fibonacci）至 100%（LU）；单个 SDC 的恢复开销低于 2%；在多达 6 个 SDC 的情况下，总恢复开销仅约 0.5%。

**⚠️ 局限性**

局限性包括：①仅对任务外部影响的操作进行验证，未覆盖内存级别 SDC；②实现中对 promise/future 的记录引入额外挂起点，可能导致性能下降；③在高负载抢夺情况下的远程任务描述符访问仍可能成为瓶颈。

---

## 5. DSIP: A Dynamic Coordination Planner for Signal-Free Intersections using Diffusion-Model-Based Multi-Agent Motion Planning

**arXiv ID:** 2606.30694 | [PDF](https://arxiv.org/pdf/2606.30694v1)

**作者:** Qian Hu `[一作]` (Shanghai Jiao Tong University), Hongtei Eric Tseng `[通讯]` (University of Texas at Arlington)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了 DSIP，一种基于扩散模型的多智能体轨迹规划框架，用于实现无信号灯的交叉口管理，并在 SUMO 仿真中与固定周期信号灯和三种强化学习基准进行比较。

**💡 创新点**

创新点在于将预训练的单车扩散模型与 CBS 冲突消除相结合，实现了连续轨迹级别的多车协同规划，彻底摆脱了传统信号相位约束，显著提升了高密度交通下的吞吐量与安全性。

**🔧 技术方法**

主要技术包括扩散模型生成轨迹、CBS 冲突检测与约束更新、SUMO/TraCI 交互、以及对比的强化学习信号控制方法（MP-Light、CoLight、AttentionLight）。

**📊 数据集**

使用的数据集为基于 SUMO 的人工生成交通流，包含 3 种四臂交叉口几何（4/6/8 车道）和 5 个交通密度层级（MinD–MaxD），共 75 个仿真配置。

**📈 对比分析**

对比实验显示 DSIP 在平均延时上比固定信号灯低 10–30 秒，平均车速提升约 30%，碰撞事件接近 0，且每次规划的推理时间仅为 7.75 ms，远低于 400 ms 的刷新间隔。

**⚠️ 局限性**

局限性包括理想化假设（完美 V2X 通信、全 CAV 交通、单交叉口、无混合交通、无执行误差），缺乏对真实道路噪声、通信延迟、车辆多样性等因素的评估。

---

## 6. Predictable GRPO: A Closed-Form Model of Training Dynamics

**arXiv ID:** 2606.30789 | [PDF](https://arxiv.org/pdf/2606.30789v1)

**作者:** Rajat Ghosh `[一作]` (Nutanix), Debojyoti Dutta `[通讯]` (Nutanix)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于第一性原理的GRPO训练动态的简化模型，描述了其更新过程为一个随机强迫的阻尼振荡器。

**💡 创新点**

创新点在于将经验性的单指数饱和法则重新解释为模型的过阻尼极限，并提供了与可独立测量的量相关的可验证预测。

**🔧 技术方法**

使用了随机强迫的阻尼振荡器模型，结合了动量和离线延迟的概念。

**📊 数据集**

使用了GSM8K数据集进行模型训练，该数据集包含小学数学问题。

**📈 对比分析**

通过与三种模型和两种组大小的实验比较，模型的训练奖励曲线拟合度达到R^2≥0.91，并且预测的组大小不变性在奖励曲线和八个数学基准的分布外转移中得到了验证。

**⚠️ 局限性**

限制在于参数化问题，当前的学习率设置使得所有实验都处于过阻尼状态，无法测试模型的稳定性阈值和过阻尼到振荡的转变。

---

## 7. Towards Knowledge Alignment in Code LLMs: Contrastive Unlearning for Evolving APIs

**arXiv ID:** 2606.30810 | [PDF](https://arxiv.org/pdf/2606.30810v1)

**作者:** Huy Q. Tran `[一作]` (Hanoi University of Science and Technology), Phuong T. Nguyen `[通讯]` (University of L'Aquila)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于对比消知的代码大模型方法，专门针对已废弃的 API 进行去学习并引导模型生成最新 API 替代方案。

**💡 创新点**

创新点在于将消知从单纯抑制废弃知识转化为对比学习机制，既消除旧 API 又显式促进其正确替代，融合抑制与替代两方面的目标。

**🔧 技术方法**

技术上结合了梯度式参数级消知（如负偏好优化和概率重分布）与对比损失函数，利用同一上下文下废弃与更新 API 的正负样本对进行训练。

**📊 数据集**

实验使用了 Wang 等人构建的 8 个 Python 库的废弃/更新 API 评测基准（含 9,087 条废弃样本和 16,423 条更新样本），并在 HumanEval 基准上评估模型的通用代码生成能力。

**📈 对比分析**

与传统消知基线（如 NPO、SimNPO 等）在 Deprecated API Usage Rate、Replacement API Usage Rate 和 Mismatch API Usage Rate 上进行对比，结果显示本方法显著降低废弃 API 用量、提升正确替代率，同时在 HumanEval Pass@1/3/5 指标上几乎无性能下降。

**⚠️ 局限性**

局限性包括依赖正则表达式和 API 别名匹配，可能漏检复杂或嵌套的 API 调用；对比数据集的正样本部分由自动生成工具产生，可能引入偏差；以及实验仅在有限的 Python 库和预训练模型上验证，未覆盖更广泛语言或模型。

---

## 8. Joint discovery of governing partial differential equations from multi-source datasets by competitive optimization

**arXiv ID:** 2606.30699 | [PDF](https://arxiv.org/pdf/2606.30699v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 9. ASR-Agnostic Multimodal Spectrotemporal Modeling for Early Dementia Detection

**arXiv ID:** 2606.30646 | [PDF](https://arxiv.org/pdf/2606.30646v1)

**作者:** Chukwuemeka Ugwu `[一作]`, Oluwafemi Richard Oyeleke `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种无 ASR 依赖的多模态框架，用于从语音信号中检测认知衰退，利用声谱图的时频位移场与卷积GRU声学特征的跨注意力融合，并在三语种数据集上验证其有效性。

**💡 创新点**

创新点包括：1) 通过连续声谱图帧计算二维位移场，捕捉语音能量的时频迁移；2) 设计跨注意力机制在声学与谱动力学模态之间实现可选、时序对齐的融合；3) 引入复合时间正则化损失，保证段级预测的平滑与一致性；4) 在英语、斯洛伐克语、希西语三种不同质量的数据集上完整消融，揭示融合策略的语料依赖性。

**🔧 技术方法**

技术手段包括：Mel 频谱预处理、三层卷积前端+ConvGRU 声学编码器、密集位移场卷积编码器、跨注意力融合、Transformer 编码器与可学习查询池化、组合时间损失（一致性、对比、进展、多尺度、注意熵）以及 AdamW 训练与混合精度。

**📊 数据集**

使用了 DementiaBank Pitt（英语）、斯洛伐克 EWA-DB（斯洛伐克语）以及 Ivanova（西班牙语）三大语料库，三者均采用以功能性认知为导向的口语诱导任务。

**📈 对比分析**

通过与单模态（仅声学或仅谱动力学）以及无注意力（元素相加）配置进行对比，评估指标为准确率、AUC、F1 等。实验结果显示：斯洛伐克语在 83.9% 准确率、0.755 AUC，西班牙语 68.5% 准确率、0.788 AUC，英语仅 53.2% 准确率、0.563 AUC；跨注意力对西班牙语必不可少，对斯洛伐克语则有负面影响，英语几乎不影响。

**⚠️ 局限性**

主要限制包括：语料多样性导致的录音质量差异（尤其是英语）；缺乏说话者分离；数据集规模差异和类别不平衡对谱动力学分支的影响；跨模态融合效果高度依赖语料质量，难以推广到更广泛场景。

---

## 10. Beyond expert users: agents should help users construct preferences, not just elicit them

**arXiv ID:** 2606.30863 | [PDF](https://arxiv.org/pdf/2606.30863v1)

**作者:** Irena Saracay `[一作]` (Stanford University), Carlos Guestrin `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了用户在与智能代理对话过程中逐步构建偏好的模型Co-constructed Preferences，并基于此设计了交互式推荐基准CosShop，评估了语言模型驱动的推荐代理；

**💡 创新点**

创新点在于：①引入信息经济学中的Search‑Experience‑Credence（SEC）框架来划分特征，刻画不同特征对对话动作的需求；②以此为基础模拟非专家用户的偏好构建过程；③构建了覆盖时尚、电影、图书三大领域的公开基准CosShop，聚焦团队准确率（agent+用户）而非传统recall@k；

**🔧 技术方法**

核心技术包括：RAG式代理架构、自然语言对话生成与解析、基于SEC的对话动作（询问、示例、解释）与偏好状态更新、预训练语言模型（GPT、Claude、LLaMA）与工具调用；

**📊 数据集**

使用了公开的时尚购物数据集（FashionIQ等）、电影推荐数据集（MovieLens）和图书推荐数据集（Goodreads），并通过正则与LLM增强得到更丰富的属性特征；

**📈 对比分析**

对比了五款前沿LLM（GPT‑4.0, Claude‑2, GPT‑3.5, Llama‑2‑13B, Llama‑2‑70B）与非LLM检索基线，发现即使在完全指定偏好时准确率可达94%，但在5轮对话后团队准确率仅提升至27%–56%，显著低于单独agent的recall@k；

**⚠️ 局限性**

局限性包括：对话动作仅一次即可触发偏好构建、偏好状态假设单调不变；解析对话动作存在错误；代理未主动提供技术解释，导致credence特征难以解锁；实验基于人工模拟用户，缺乏真实人机交互验证；

---

## 11. Bridging Scientific Heritage: An Arabic--Russian Parallel Corpus and LLM Benchmark for Sustainable Knowledge Transfer

**arXiv ID:** 2606.30943 | [PDF](https://arxiv.org/pdf/2606.30943v1)

**作者:** M. K. Arabov `[一作]` `[通讯]` (Kazan Federal University), M. K. Arabov (Kazan Federal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了约27,000句阿拉伯–俄语科学平行语料，并使用LoRA/QLoRA对mT5、NLLB-200与Qwen2.5-7B三款多语言LLM进行微调，构建了首个阿拉伯–俄语科学翻译基准。

**💡 创新点**

首次在该语言对上系统评估Encoder‑Decoder与Decoder‑Only模型，提出LoRA秩的选择策略，并证明Qwen2.5-7B在Decoder‑Only架构下显著优于传统Encoder‑Decoder模型。

**🔧 技术方法**

采用多语言LLM（mT5-base、NLLB-200-distilled-1.3B、Qwen2.5-7B-Instruct）与LoRA/QLoRA参数高效微调，Beam搜索生成，使用BLEU、chrF、BERTScore与COMET四项自动评估。

**📊 数据集**

使用由15k科学摘要与11,878通用域句子（宗教、新闻、对话、词典、圣经、Tatoeba）构成的混合平行语料，参考译文来源为Gemma-3-4B与LLaMA-3.1-8B生成的俄语翻译。

**📈 对比分析**

通过零射击、不同LoRA秩（8/16/32/64）以及QLoRA微调对三模型进行对比，Qwen2.5-7B-QLoRA(rank8)达成BLEU 23.15、chrF 43.89、BERTScore 0.9058、COMET 0.758，mT5表现最低，NLLB位居中间，表明模型规模与架构对翻译质量有显著影响。

**⚠️ 局限性**

主要限制包括参考译文非人工，仅覆盖科学摘要；未尝试更高秩或更多模型；翻译结果普遍短于参考，未对专业术语与风格做细粒度评估。

---

## 12. Mind the Residual Gap: Probabilistic Downscaling under Real-World Bias

**arXiv ID:** 2606.30821 | [PDF](https://arxiv.org/pdf/2606.30821v1)

**作者:** Yujin Kim `[一作]` (Cornell University), Sarah Dean `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在气象与多尺度物理系统的概率下尺度任务中，提出了一种名为ReMatch的残差分布匹配方法，通过在低维PCA空间中使用最优传输（OT）将训练阶段的残差分布对齐到校准集的残差分布，从而解决残差目标误规范导致的样本外偏差与下溢散问题；

**💡 创新点**

核心创新点在于将残差目标误规范问题视为训练与测试残差分布不匹配的根源，并通过OT在PCA空间对训练残差进行“重采样”，使残差生成器在更真实的校准分布上训练，保持了平均预测的结构先验且显著提升了分布校准与准确性；

**🔧 技术方法**

采用了三阶段框架：1）利用UNet或SwinIR等回归网络进行高分辨率均值预测；2）利用OT在PCA表示下对训练残差与校准残差进行匹配；3）用条件扩散模型（EDM式）对匹配后的残差进行概率生成；

**📊 数据集**

实验数据包括：①受控的BLASTNet Momentum128 3D超分辨率数据（16×16→128×128），在不同层次的人工下尺度偏差下验证理论；②真实ERA5→HRRR风场下尺度数据（21×21→168×168，共10个通道），检验在跨源跨分辨率下的效果；

**📈 对比分析**

与UNet、ConvFNO、SwinIR等确定性基线以及CorrDiff、CFG、UC等概率下尺度基线进行对比，评估指标包括RMSE、SSR、CRPS/MAE、LPIPS、ACC。ReMatch在RMSE、SSR、CRPS、LPIPS、ACC等指标上均优于所有基线，并在粒子轨迹实验中展示了更好的分布校准；

**⚠️ 局限性**

局限性主要体现在：需要额外的校准集（若无代表性样本难以匹配）；PCA为线性全局表示，可能无法充分保留局部极端残差模式；OT计算在大规模数据集上计算量大，需进一步改进规模化方案；

---

## 13. Indi-RomCoM: Code-Mixed Benchmark for Evaluating LLMs on Romanized Indic-English Instructions

**arXiv ID:** 2606.30790 | [PDF](https://arxiv.org/pdf/2606.30790v1)

**作者:** Avisha Das `[一作]` (Shiv Nadar University), Pulkit Verma `[通讯]` (IIT Madras)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Indi-RomCoM 基准，系统评估大语言模型在罗马化印地语-英语混合指令上的表现。

**💡 创新点**

创新点：①构建多语言、多任务、分级代码混合（25%、50%、75%）的人工验证数据集；②引入 Register Defection Rate（RDR）评估模型在混合语料中的注册保持能力；③展示仅靠规模或本土脚本预训练无法克服混合语言挑战，强调需要针对性代码混合监督。

**🔧 技术方法**

技术方法：基于 Equivalence Constraint 与 Matrix Language Frame 的四步生成管道（翻译→词汇选择→混合强度控制→罗马化），使用 LLM（OpenAI、Google、Anthropic 等）、开源模型（LLaMA、Qwen、Gemma 等）以及印度专用模型（Airavata、TamilLLaMA、Sarvam 等），采用零/少样本提示、低秩适配微调，并采用 RDR、VCR、PG 等评估指标。

**📊 数据集**

数据集：从七个公开任务（Word Analogy、Question Decomposition、GSM8K、Paraphrase、Sentiment、NLI、Toxicity）中抽取 100 条实例，按四种印度语（Hindi、Bengali、Gujarati、Tamil）和三种混合强度扩展至 8,400 条；人类验证由四名双语评审完成，评估语义保真和自然度。

**📈 对比分析**

比较方法：在零样本和 3-shot 设定下，对 19 个模型进行平均任务准确率、RDR、VCR、PG 等度量。结果显示所有模型在代码混合指令下准确率均下降，尤其是高强度（75%）时；规模越大下降越小，但仍有显著性能差距；印度专用模型在 RDR 上表现最佳。少样本提示可提升约 7–8% 准确率，但未显著缓解性能差距。

**⚠️ 局限性**

局限性：仅涵盖四种印度语，未覆盖多种音素漂移或内部词混合现象；人工验证评审样本量小；基准基于现有公开数据集，可能存在标签噪声；缺少摘要、开放式问答等任务；可能导致模型对特定强度的过拟合。

---

## 14. GRay: Ray Tracing 3D Gaussians Near the Speed of Splats

**arXiv ID:** 2606.30869 | [PDF](https://arxiv.org/pdf/2606.30869v1)

**作者:** Yohan Poirier-Ginter `[一作]` (Université Laval), George Drettakis `[通讯]` (Inria)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 GRay 的 3D 高斯光线追踪方法，并结合稠密初始化（DI）显著提升训练速度与运行帧率。

**💡 创新点**

创新点包括：① 对稠密初始化与光线追踪算法复杂度关系的理论与实证分析；② 采用 PPLL、尺度衰减、分离混合透明度 (DHT) 与权重驱动剔除等技术实现高效渲染；③ 通过精细的学习率调度与剪枝策略在保持接近 3DGRT 质量的同时，将迭代次数减半，优化时间与显存占用。

**🔧 技术方法**

主要技术：光线追踪 + BVH + 有向包围盒 (OBB) + PPLL (per‑pixel linked lists) + 失真衰减 + 分离混合透明度 + 权重剔除 + 多尺度学习率调度 + 立体相机稠密匹配网络 (RoMa) 的稠密初始化。

**📊 数据集**

使用的基准数据集为 13 个标准 3DGS 评测场景，涵盖 Mip‑NeRF360、Tanks & Temples 与 Deep Blending 三大数据集。

**📈 对比分析**

与 3DGS、3DGRT、EDGS 以及 RayGaussX 等方法对比：GRay 在 15K 迭代下达到约 5 分钟的优化时间、248 FPS 的运行帧率，速度与 3DGS 相当；在质量上 LPIPS 接近 3DGRT，PSNR/SSIM 稍低，但比 3DGRT 更快；与 3DGRT 相比，GRay 速度提升约 4 倍，优化时间约 10 倍；与 3DGS 相比，质量略低但速度相近。DI 使得 3DGS 训练时间延长、PSNR/SSIM 下降但 LPIPS 提升，体现稠密初始化对光线追踪的优势。

**⚠️ 局限性**

局限性：1) 需要更多显存（GRay 峰值约 16.5 GB，较 3DGRT 高）；2) 对高度反射、光滑表面表现不佳；3) 仍无法完全匹配 3DGS 的 PSNR/SSIM；4) 稠密初始化耗时约 2 分钟，整体训练时间仍略高于 3DGS。

---

## 15. Less Deliberate in Teams: Student LLM Use Across Individual and Collaborative Work

**arXiv ID:** 2606.30860 | [PDF](https://arxiv.org/pdf/2606.30860v1)

**作者:** Sehrish Basir Nizamani `[一作]` (Virginia Polytechnic Institute and State University), Khyati Goyal `[通讯]` (Virginia Polytechnic Institute and State University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对96名本科生在一学期内交替完成的三次个人作业和三次团队项目里，调查并量化其使用大语言模型（LLM）的频率、提示策略和输出验证行为；通过对比个人与团队情境下的行为差异，探讨社交环境对LLM使用的影响。

**💡 创新点**

首次系统性揭示团队协作时LLM使用的“崩溃与恢复”轨迹，并发现团队情境导致验证行为（尤其是测试驱动验证）持续下降，说明协作环境能显著重塑学生与LLM交互的质量。

**🔧 技术方法**

采用问卷调查收集自评的LLM使用、提示频率、提示技术和验证方法等数据；利用统计检验（χ²检验、配对t检验、Wilcoxon符号秩检验）对不同情境下的行为差异进行显著性检验。

**📊 数据集**

实验数据来源于两门大数据与可视化课程的96名学生（共6次作业），涵盖三次个人作业（HW3、HW4、HW5）和三次团队项目里程碑（TP1、TP2、TP3）。

**📈 对比分析**

通过对比个人与团队情境下的百分比差异（如LLM使用率、提示次数、验证方法占比），使用统计检验确定差异显著。结果显示：个人作业LLM使用率≈84%，团队TP1降至≈40%，TP2/TP3恢复至≈75-78%；提示频率和技术在团队中显著下降；验证中“运行测试”的比例从个体的33.2%降至团队的13.8%。

**⚠️ 局限性**

局限性包括：样本仅来自一门课程，难以推广到更广泛人群；所有行为均为自评，易受社会期望偏差影响；问卷响应率非完全，可能存在非随机缺失；团队与任务类型在TP1时交叉影响，难以完全区分两者。

---

## 16. TAPE: Tether-Aware Path Planning for Autonomous Exploration of Unknown 3D Cavities Using a Tangle-Compatible Tethered Aerial Robot

**arXiv ID:** 2606.30817 | [PDF](https://arxiv.org/pdf/2606.30817v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 17. Partition-Guided Distance Saliency: Bridging Decision and Objective Spaces in Many-Objective Optimization

**arXiv ID:** 2606.30836 | [PDF](https://arxiv.org/pdf/2606.30836v1)

**作者:** Cláudio Lúcio do Val Lopes `[一作]` (A3Data), Elizabeth Fialho Wanner `[通讯]` (Centro Federal de Educação Tecnológica de Minas Gerais)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 Partition‑Guided Distance Saliency（PGDS）的框架，用于在许多目标优化中通过几何距离映射解释决策变量对目标空间位置的影响。

**💡 创新点**

创新点在于将 Minimal Learning Machine（MLM）与 KD‑Tree 区域划分结合，自动生成局部“Dominating Point”目标，并采用基于距离的 RISE 变体生成可视化的驱动与阻碍变量的 saliency map。

**🔧 技术方法**

采用的技术包括 Minimal Learning Machine（距离矩阵回归）、KD‑Tree 目标分区、距离解释器（RISE 变体）、多目标进化算法（NSGA‑III/NSGA‑II）以及对模型可解释性的评估指标（R²、MSE）。

**📊 数据集**

实验数据集涵盖了经典基准 DTLZ2、DTLZ7、WFG3（10 维目标）以及工业实际问题 Welded Beam，分别用于验证模型精度、可解释性与工程可行性。

**📈 对比分析**

与传统可视化（散点图、PCP）和基于规则的 XAI（XLEMOO、R‑XIMO）对比，PGDS 在高维目标空间中实现了 R²>0.99 的高精度预测，并通过敏感性分析验证了驱动/阻碍变量的正确性，提供了更具可操作性的决策建议。

**⚠️ 局限性**

主要限制包括：无法直接求解逆问题得到精确的改进决策向量；对超参数 K、D、L 的选择需要经验或额外的调参；目前仅为诊断性工具，尚未实现完全的指导性优化。

---

## 18. Demystify, Use, Reflect, Assess (DURA): An Experience Report on LLM Integration in CS2

**arXiv ID:** 2606.30908 | [PDF](https://arxiv.org/pdf/2606.30908v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 19. Wind and State Estimation on SE(3): Comparative Evaluation of EKF and UKF with Continuous and Discrete Quadrotor Models

**arXiv ID:** 2606.30804 | [PDF](https://arxiv.org/pdf/2606.30804v1)

**作者:** Hiranya Udagedara `[一作]` (University of Calgary), Mahdis Bisheban `[通讯]` (University of Calgary)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究使用离散和连续 SE(3) 形式的四旋翼动力学，结合 EKF 与 UKF 对风速进行估计，并在数值仿真与实测中对其性能进行评估。

**💡 创新点**

创新点在于引入离散化的 Lie 组变分积分器作为四旋翼动力学模型，并将其与无迹卡尔曼滤波结合，在离散模型下实现更高精度的风速估计。

**🔧 技术方法**

使用了 Lie 组变分积分器、SE(3) 离散与连续动力学、扩展卡尔曼滤波、无迹卡尔曼滤波、随机游走风模型以及噪声协方差设计等技术。

**📊 数据集**

采用了仿真中 400 Hz 传感器采样的人工风场（恒定与正弦），以及 2026 年 2 月在加拿大卡尔加里大学机械工程楼实测的 M10 GPS、ICM‑45686 IMU 与地面三声风速计数据。

**📈 对比分析**

通过 RMSE 与标准差对比四种算法（连续 EKF、连续 UKF、离散 EKF、离散 UKF），在仿真中离散 UKF 取得最佳性能；实测中连续模型表现优于离散模型，但离散 UKF 仍保持稳定。

**⚠️ 局限性**

局限在于离散模型在真实噪声与漂移下易偏离真实动力学；实验仅评估水平风速，缺乏垂直方向估计；使用低成本传感器导致测量噪声限制精度。

---

## 20. Understanding and Evaluating Claw-like Agent Security Through a Computer-Systems Lens

**arXiv ID:** 2606.30755 | [PDF](https://arxiv.org/pdf/2606.30755v1)

**作者:** Peizhi Niu `[一作]` (University Of Illinois Urbana Champaign), Dawn Song `[通讯]` (University Of California Berkeley)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了SafeClawArena基准，用计算机系统视角设计了406个针对Claw-like AI代理四大攻击面（SSI、PSE、CDF、IPI）的对抗任务。

**💡 创新点**

创新点在于将传统系统安全原则映射到代理架构，构造跨组件攻击面，并通过容器化可复现的评估管道实现系统级安全测评。

**🔧 技术方法**

技术上结合了容器化部署、污点跟踪检测、结构化工具调用与可插拔插件机制，利用OpenClaw/NemoClaw/SeClaw等平台与GPT-5、Gemini、Claude等前沿LLM进行交叉评估。

**📊 数据集**

数据集为自生成的406个攻击任务，涵盖技能供应链、持久状态、跨界数据流和间接提示注入等四大类别，并在每个任务中植入可追踪的凭证标记。

**📈 对比分析**

通过与三种平台、五款LLM的15种配置对比，发现总体攻击成功率在20%–70%之间，展示了平台硬化与模型强度相互作用的复杂性。

**⚠️ 局限性**

局限性包括任务生成仅覆盖已知攻击模式、评估侧重凭证泄露而非更细粒度攻击，以及对现实部署环境中非容器化系统的泛化不足。

---

## 21. Security--Fidelity Tradeoffs: The Hidden Cost of Prompt Injection Defense

**arXiv ID:** 2606.30783 | [PDF](https://arxiv.org/pdf/2606.30783v1)

**作者:** Mitchell Hermon `[一作]` (University of Illinois Urbana Champaign), Haohan Wang `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大语言模型在面对间接提示注入时的安全与保真度权衡，提出了可区分执行、处理与忽略的评测基准SecFid，并用它评估了多种模型与防御策略。

**💡 创新点**

创新点包括：① 将安全与保真度分离、可测量化；② 设计行为可分离的任务样例，构造可区分的三种输出；③ 映射出安全–保真度前沿，揭示防御在压制与修复之间的差异；④ 通过DPO微调实现“修复优于压制”的安全-保真度调优。

**🔧 技术方法**

技术主要为：prompt注入防御方法（ASIDE、DefensiveTokens、ISE、SecAlign）、DPO（离散对比优化）用于微调、embedding相似度判定、统计检验（Wilson CI）等。

**📊 数据集**

数据集包括：1,168条由手工模板与自动生成（Gemini 2.5 Pro、FineEdit等）构造的任务实例，涵盖实体抽取、计数、翻译、编辑；以及252条工具使用场景；训练中使用895条样例，测试集中141条；编辑任务为保留集。

**📈 对比分析**

比较方法：在48种模型+防御配置上计算执行率、处理率、忽略率，定义安全=1-执行率、保真度=1-忽略率；绘制安全–保真度前沿；对比不同防御的压制/修复比例；通过DPO微调验证保真度提升。性能结果表明：最高安全点≈99.3%但保真度仅≈71%，最高保真点≈96.5%但安全仅≈47.8%。

**⚠️ 局限性**

局限性：仅评估固定非自适应探针，未涵盖自适应攻击；基准设计简化为二元处理/过滤决策，未涵盖更细粒度的防御操作；评测侧重于推理阶段，未考虑训练时对抗；可能存在基准样例与真实场景的差距。

---

## 22. Unveiling Transferability in Trajectory Prediction via Latent Scene Embeddings

**arXiv ID:** 2606.30777 | [PDF](https://arxiv.org/pdf/2606.30777v1)

**作者:** Theodor Westny `[一作]` (Linköping University), Erik Frisk `[通讯]` (Linköping University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过联合训练的潜在嵌入模型学习轨迹数据集的分布，构建多维潜在空间，并用KL散度衡量数据集间的可迁移性。

**💡 创新点**

提出数据集级别的潜在表示学习与方向性概率转移度量，并在24个多样化数据集上系统评估其对零样本与微调性能的可预测性。

**🔧 技术方法**

利用图门控循环单元（GGRU）编码器‑解码器、图神经网络、低秩正则化、KL散度、MMD对比、Lasso回归等技术。

**📊 数据集**

涵盖24个轨迹预测数据集，包括车辆、行人、混合交通，既有地图信息也有无地图场景。

**📈 对比分析**

通过计算潜在分布的KL散度与minADE_6的相关性，验证其在零样本（ρ≈0.81）和微调（ρ≈0.73）场景下能准确预测转移性能，并可用于预训练源选择。

**⚠️ 局限性**

仅依赖高斯近似，可能无法捕捉更复杂分布；评估范围局限于两类任务；需要在所有数据集上联合训练，规模受限；未在更大范围或其他任务中验证泛化。

---

## 23. GaussLite: Online Task-Conditioned 3D Gaussian Splatting for Real-Time Robotic Mapping

**arXiv ID:** 2606.30809 | [PDF](https://arxiv.org/pdf/2606.30809v1)

**作者:** Annika Thomas `[一作]` (MIT), Jonathan P. How `[通讯]` (MIT)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `51c0528b-f690-4182-ae60-bb5f046c276c` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种任务驱动的 3D 高斯投射映射系统 GaussLite，利用自然语言任务描述自动将高斯密度集中在任务相关区域，并支持多机器人任务专化地图的融合。

**💡 创新点**

创新点在于：①将自然语言任务通过 LLM 解析成结构化任务图，并用开放词汇检测与分割生成像素级关联度掩模；②根据关联度动态分配高斯种子、尺度与梯度，实现任务感知的高斯稠密化；③利用每个高斯的活跃优化计数进行多代理投票融合，实现只交换任务相关子图。

**🔧 技术方法**

技术包括：Phi‑3-mini LLM 解析、Grounding DINO + FastSAM 检测/分割、3D 高斯投射（Gaussian‑Splatting）、ROI 稳定预算、层化初始尺度、梯度集中稠密化、随机多视图优化、基于活跃计数的 voxel 投票融合。

**📊 数据集**

数据集：Replica（8 个室内 RGB‑D 序列）与自制 Campus（高架、办公室、户外）以及真实机器人采集的数据，用于评估任务映射质量与多代理融合。

**📈 对比分析**

与 Gaussian‑SLAM、SplaTAM、MonoGS 三个基准在匹配高斯预算、实时映射条件下比较，评价指标为 ROI PSNR/SSIM/LPIPS。GaussLite 在 ROI PSNR 上平均提升 +2.72 dB（Replica）/ +2.23 dB（Campus），在多代理融合中比简单拼接提升 +3.42 dB，仅共享 7% 的地图。

**⚠️ 局限性**

局限性包括：①关联度掩模依赖深度渲染，噪声时可能失效；②LLM 解析可能产生错误或幻觉；③仅在任务未变更时有效，对任务切换不自动重分配；④评估任务表述有限，未覆盖广泛自然语言变体。

---

## 24. Multilingual Polarization Detection Using Transformer-Based Models with Class Weighting and Threshold Tuning

**arXiv ID:** 2606.30857 | [PDF](https://arxiv.org/pdf/2606.30857v1)

**作者:** Aaron Bundi Anampiu `[一作]` `[通讯]` (African Institute for Mathematical Sciences), Aaron Bundi Anampiu (African Institute for Mathematical Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在SemEval-2026 Task 9中针对英语和斯瓦希里语实现多任务的多标签极化检测系统，完成了二分类、类型分类和表现识别三子任务。

**💡 创新点**

创新点：① 为每种语言选择最合适的预训练模型（RoBERTa-base和AfroXLMR-base）；② 用类别加权的二元交叉熵解决严重标签不平衡；③ 在验证集上做每个标签阈值细调以提升macro F1。

**🔧 技术方法**

技术：Transformer fine‑tuning（RoBERTa、AfroXLMR）、类别加权损失、阈值调优、AdamW、FP16混合精度、微调等。

**📊 数据集**

数据集：SemEval‑2026 Task 9的多语言数据，重点使用英文和斯瓦希里语，包含约3000-5000条标注实例。

**📈 对比分析**

与POLAR基线和通用mBERT基线比较，系统在所有子任务和语言上都取得更高的macro F1，英语子任务3达到0.4791，斯瓦希里子任务3 0.5830，排名分别在各自任务中前列。

**⚠️ 局限性**

局限：对极端稀有标签（性别/性别、其他）仍难以准确识别；阈值过拟合导致验证-测试差距；缺乏对讽刺、隐喻和语用推理的理解。

---

## 25. StreamGuard: Low-Overhead Resilience for Real-time HPC Data Streams

**arXiv ID:** 2606.30848 | [PDF](https://arxiv.org/pdf/2606.30848v1)

**作者:** Hai Duc Nguyen `[一作]` (Argonne National Laboratory), Ian Foster `[通讯]` (University of Chicago and Argonne National Laboratory)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种名为StreamGuard的弹性框架，专门用于实时科学流式工作流，能够在硬件故障和性能异常频发的环境下保持实时性。

**💡 创新点**

创新点在于：
- 将弹性设计拆分为可组合的模块（非阻塞异步检查点、进度感知负载均衡），仅在生产者–消费者对之间执行，避免全局同步；
- 采用动态、异步、非阻塞的检查点机制，利用每个分区独立快照并与计算重叠；
- 引入基于实时进度比的负载均衡策略，自动迁移慢分区到快工作器，减轻 straggler 效果。

**🔧 技术方法**

使用技术包括：
- Parsl 用于容错任务管理与自动重启；
- Mofka 分布式消息队列实现可靠的生产者–消费者通信；
- VeloC 的多层级检查点实现；
- 通过 Young/Daly 公式动态调节检查点间隔；
- 自适应进度比 G(t) 与容量模型 c_j(t) 的负载均衡算法。

**📊 数据集**

实验基准是同步辐射设施的断层成像工作流（Trace Reconstruction Engine），数据规模为 2048×2048 像素投影，约 1500 条记录、300–350 次迭代，整个阶段约 200 秒。

**📈 对比分析**

与基线（原始实现、固定间隔 VeloC、Apache Flink）比较，评价指标包括处理时长、是否满足 1000 秒的实时截止时间以及失效时的额外开销。结果显示：
- 在无失效时，StreamGuard 的额外开销 <1%；
- 在频繁失效（MTTF 40 s）和慢速异常（平均 1000 ms）下，StreamGuard 的总处理时长仅比理想执行提升 10–20%，成功满足实时截止；
- Flink 在失效频繁或负载不均时往往因同步检查点导致 25% 以上开销，甚至无法满足截止时间；
- VeloC 的静态检查点在不同失效率下表现不一致，失效时恢复时间显著增长。

**⚠️ 局限性**

局限性包括：
- 假设各分区计算成本相近，未考虑极端负载偏斜；
- 仅在已分配足够资源的环境下验证，未覆盖资源不足或突发峰值情况；
- 负载均衡策略仅局限于单一生产者–消费者对，跨阶段交互、I/O 竞争与背压未系统评估；
- 对于动态资源/工作负载不匹配的实时流式场景，需进一步研究。

---

## 26. CALO: Constraint-Aware Learning Optimization for Joint Resource Allocation in Double-Active RIS-Assisted Wireless Networks

**arXiv ID:** 2606.30803 | [PDF](https://arxiv.org/pdf/2606.30803v1)

**作者:** Alaa S. Arabiyat `[一作]` (Princess Sumaya University for Technology), Mohammad J. Abdel-Rahman `[通讯]` (Princess Sumaya University for Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了CALO框架，用于双主动RIS辅助网络的约束感知学习优化资源分配。

**💡 创新点**

通过分组分数化实现约束自满足、使用STE处理离散元素分配，并以回退hinge损失鼓励超越传统BCD解法。

**🔧 技术方法**

利用多层感知器（MLP）+softmax分组映射+STE+回退hinge损失，结合基于BCD的参考标签进行端到端学习。

**📊 数据集**

使用BCD生成的训练集（10,000样本），包含城市与农村场景的系统参数；测试集各2,000样本。

**📈 对比分析**

与BCD基线比较，保持100%可行性；平均速率提升约0.4–0.5 bps/Hz（城市）和0.3–0.4 bps/Hz（农村），推断时间降至10⁻⁴ s，统计检验显示显著改进。

**⚠️ 局限性**

依赖BCD参考解，无法获得全局最优；仅适用于单用户单链路，未覆盖多用户、多基站或动态环境的扩展。

---

## 27. Robustness-Based Synthesis for Time Window Temporal Logic Specifications via Mixed-Integer Linear Programming

**arXiv ID:** 2606.30820 | [PDF](https://arxiv.org/pdf/2606.30820v1)

**作者:** Philip Smith `[一作]` (Worcester Polytechnic Institute), Kevin Leahy `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

对离散时间线性系统在 Time Window Temporal Logic（TWTL）任务规范下进行鲁棒性最大化的 MILP 合成与闭环 MPC 控制。

**💡 创新点**

① 将 TWTL 的保持、窗口和串联算子递归编码为混合整数线性约束，保证鲁棒度最大化；② 引入基于 DFA 的任务自适应预测 Horizon，使每步 MILP 只考虑剩余窗口；③ 通过预计算与热启动显著降低在线求解开销。

**🔧 技术方法**

混合整数线性规划（MILP）、大 M 编码、时间窗口自动机（DFA）、参数化 MPC 与热启动技术。

**📊 数据集**

在 20×20 维度的连续空间中，使用双积分器离散化模型、四个方形兴趣区域和障碍物进行仿真；实验不使用公开数据集，而是自行构造的随机任务序列。

**📈 对比分析**

与 STL 等价的规范比较：在子任务数≥4 时 TWTL 通过二进制变量减少 2-3 倍，求解时间下降；与固定 Horizon 的 MPC 对比：DFA 自适应 Horizon 使求解时间降低 30-50% 以上；开闭环对比展示闭环 MPC 在扰动下仍能满足规范。

**⚠️ 局限性**

① 仅实现了 min-max 鲁棒度，未覆盖 AGM 鲁棒度；② 仍使用线性 DFA 编码，未采用对数编码降低二进制变量；③ 适用范围限定在单机离散时间线性系统，未针对多智能体或非线性系统展开；④ 需要进一步验证在更大规模任务序列下的实时性。

---

## 28. Test-Time Verification for Text-to-SQL via Outcome Reward Models

**arXiv ID:** 2606.30851 | [PDF](https://arxiv.org/pdf/2606.30851v1)

**作者:** Mattia Tritto `[一作]` (Polytechnic University of Bari), Tommaso Di Noia `[通讯]` (Polytechnic University of Bari)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出GradeSQL框架，将Outcome Reward Model（ORM）作为学习式检验器用于Text-to-SQL的测试时候候选查询排序，提升模型可靠性。

**💡 创新点**

创新点在于：①将ORM迁移到结构化查询生成任务；②构建可自动化的数据生成与执行标注流水线，免除人工标注；③在Best-of-N推理中用ORM替代传统启发式选择，兼顾可扩展性。

**🔧 技术方法**

使用技术包括：大语言模型生成候选SQL；通过执行结果对候选进行等价标注；对ORM进行自回归微调；在推理阶段对候选评分并挑选最高分。

**📊 数据集**

使用公开的Spider与BIRD两大Text-to-SQL基准数据集，分别覆盖多表连接、子查询等复杂场景。

**📈 对比分析**

与多数投票（Majority Voting）和基于执行的Best-of-N比较，ORM在BIRD上提升了约4.33％，在Spider上提升了约2.10％；在不同规模LLM（如OmniSQL-7B/14B/32B、Qwen2.5-7B等）上均保持稳健增益，尤其对复杂查询和大候选集更显优势。

**⚠️ 局限性**

局限性包括：需额外离线训练；提升幅度虽稳定但相对有限（约2-5％）；依赖执行等价标注，可能误判语义等价/错误；仅在干净基准上验证，真实环境中对噪声模式和隐私数据库的适用性尚未充分探测。

---

## 29. Information Terra: A Narrative-Anchored Semantic-First Projection of Document Embeddings

**arXiv ID:** 2606.30824 | [PDF](https://arxiv.org/pdf/2606.30824v1)

**作者:** Brian Keith-Norambuena `[一作]` (Universidad Católica del Norte), Chris North `[通讯]` (Virginia Tech)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种以叙事终点为极点、SLERP 大圆为经线的双极投影（Information Terra），将文档嵌入投影到地球模型上，并生成无回溯的叙事路径；

**💡 创新点**

创新点包括：①以叙事终点为极点、SLERP 为经线的语义先导投影；②对最大容量路径（MCP）加入地理单调约束，保证路径不回溯；③将投影直接用于可视化生成类似陆地的地图，并用局部语言模型为地貌命名；

**🔧 技术方法**

使用了文档向量嵌入（OpenAI 1536-d）、Spherical Linear Interpolation (SLERP)、双极球面投影、最大容量路径 (MCP) 的地理单调改进、Gaussian KDE、HDBSCAN 主题聚类、局部语言模型 Qwen2.5-1.5B-Instruct、Mollweide 与球面渲染等技术；

**📊 数据集**

使用了包含 540 篇英文新闻的古巴-美国关系语料库（2016-2021 年），涵盖奥巴马访古巴与 2021 年墨西哥援助船队等事件；

**📈 对比分析**

与传统平面投影（UMAP、PCA）对比，地理单调 MCP 的协同度损失仅为平均 0.5%（95% 分位数 2.3%），路径长度更短（6 步 vs 8 步），且保持无回溯；

**⚠️ 局限性**

局限性在于：仅针对单一叙事终点对；未对多终点或不同语料库进行广泛评估；依赖用户选择的终点；平面 vs 球面可视化的可用性尚待用户研究。

---

## 30. AI for Quality Assurance in the Operating Room

**arXiv ID:** 2606.30657 | [PDF](https://arxiv.org/pdf/2606.30657v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 31. Personalizing Marketplace Policies with Competing Objectives and Constrained Experiments: Evidence from a Job Marketplace

**arXiv ID:** 2606.30932 | [PDF](https://arxiv.org/pdf/2606.30932v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 32. LUMOS: A Semantic Operating-System Layer for Accessibility-Grounded AI Agents

**arXiv ID:** 2606.30697 | [PDF](https://arxiv.org/pdf/2606.30697v1)

**作者:** Yogeswar Reddy Thota `[一作]` `[通讯]` (University of Texas at Dallas), Yogeswar Reddy Thota (University of Texas at Dallas)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出LUMOS，一种在操作系统层面为AI代理提供语义化交互层，利用现有的可访问性和UI自动化API生成可机器读取的语义蓝图，并在LLM的指导下通过可见UI原语执行操作；

**💡 创新点**

创新点在于将可访问性树作为AI代理的语义子层，提供稳定的元素ID、角色、属性、边界等信息，并实现实时指针语义定位、统一可见动作模式、LLM规划与安全校验的完整 observe–plan–act 循环；

**🔧 技术方法**

使用技术包括Windows UI Automation、DOM/浏览器可访问性树、Python 运行时、LLM（如本地或兼容接口）、JSON 动作模式、可见UI执行器、指针点定位（ElementFromPoint）、安全过滤器与内存回溯；

**📊 数据集**

论文中未公开使用具体公开数据集，而是通过对 Windows 桌面应用（Notepad、设置、浏览器等）和自定义测试场景进行手工构造的观察-动作日志进行评估；

**📈 对比分析**

对比方法包括：① 传统截图+OCR+LLM 与 LUMOS 语义蓝图+LLM 的任务完成率、延迟、token 消耗、恢复回合数；② 蓝图压缩与视觉描述/截图的大小/token 比较；③ 指针定位延迟与截图+视觉推断的识别延迟；实验显示 LUMOS 在大多数任务中显著降低 token 使用（约30–50%）、平均延迟降低 20–40%，且恢复回合数减少 1–3 次；

**⚠️ 局限性**

局限性主要包括：依赖可访问性树质量，易受缺失或错误的属性影响；动态界面变化可能导致执行失败；LLM 仍可能误判或执行错误操作；安全风险仍需进一步完善；当前仅验证了简单文本/启动任务，尚未覆盖复杂多应用工作流或高级图形应用。

---

## 33. Measuring Judgment Quality in Natural-Language Explanations: Evidence from Forecasting Tournaments

**arXiv ID:** 2606.30987 | [PDF](https://arxiv.org/pdf/2606.30987v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 34. Neural Signatures of Programming Expertise: Classifying Programmer Skill Levels Using EEG Data

**arXiv ID:** 2606.30879 | [PDF](https://arxiv.org/pdf/2606.30879v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 35. Position: Vision-Language-Action Models Cannot Be Verified to Perform Physical Reasoning

**arXiv ID:** 2606.30686 | [PDF](https://arxiv.org/pdf/2606.30686v1)

**作者:** Taozhao Chen `[一作]` (University of Sydney), Huaming Chen `[通讯]` (University of Sydney)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过理论分析与案例对比，论文论证 Vision‑Language‑Action (VLA) 系统中语义泛化是否足以支撑物理执行，并指出现有评估方法无法区分两者的贡献。

**💡 创新点**

创新点在于提出三层可辨识性缺失框架（归因、来源、表示）并提出可控变异评估设计，以独立测量语义泛化与物理泛化，从而厘清语义充分性假设的真实性。

**🔧 技术方法**

使用理论分解法将 VLA 策略拆解为语义映射和物理决策两部分，构造数学假设并分析现有基准协议的不足；同时引入控制变量实验思路来实现因果归因。

**📊 数据集**

主要引用现有 VLA 基准（如 RT‑2、OpenVLA、π_0 等）和大规模 VLM 预训练语料，未单独构建新的数据集，而是利用这些已有数据进行归因与评价。

**📈 对比分析**

通过对比不同系统的任务成功率与对失败原因的归因，论文发现所有系统均将物理失败归因为工程或数据问题，缺乏可量化的物理泛化指标，表明传统评估无法区分语义与物理贡献。

**⚠️ 局限性**

局限性在于评估缺乏可辨识信号，导致语义充分性假设无法被独立验证，从而导致资源投入误判、部署信心过高，并可能导致关键能力被忽视或误解。

---

## 36. ALM2Vec: Learning Audio Embeddings for Universal Audio Retrieval with Large Audio-Language Models

**arXiv ID:** 2606.30682 | [PDF](https://arxiv.org/pdf/2606.30682v1)

**作者:** Fengjie Lu `[一作]` (Zhejiang University), Aaron Yee `[通讯]` (Zhejiang University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ALM2Vec，一个基于大型音频语言模型的通用音频嵌入框架，能够实现跨域、跨任务的音频检索，并支持指令驱动检索。

**💡 创新点**

创新点在于将大型音频语言模型的音频理解、指令跟随与推理能力迁移到统一嵌入空间，实现指令感知检索，并能处理长时序和多模态内容。

**🔧 技术方法**

采用 MiDashengLM 作为骨干网络，使用双向对比学习损失、LoRA 微调以及自监督的指令感知嵌入提取技术。

**📊 数据集**

训练使用 AudioCaps、Clotho、LibriSQA、MMAU‑Mini 等音频‑文本、语音‑文本、问答数据集；评测在 AudioCaps/Clotho、LibriSQA、MMAU‑Mini 上进行。

**📈 对比分析**

与 CLAP、Jina‑Embed、LSR 等基线对比，在 AudioCaps、Clotho 上取得竞争或领先的 Recall@K；在 LibriSQA 上超越 Whisper+BGE pipeline；在 MMAU‑Mini 上与大型音频语言模型相当。

**⚠️ 局限性**

局限性包括微调后对通用语义检索性能略有下降、对极端噪声或稀有事件的鲁棒性不足，以及对长时序音频覆盖仍有限。

---

## 37. ReactionAtlas: Ab origine exploration of chemical reaction networks with machine learning

**arXiv ID:** 2606.30778 | [PDF](https://arxiv.org/pdf/2606.30778v1)

**作者:** Stefan Gugler `[一作]` (BIFOLD), Klaus-Robert Müller `[通讯]` (BIFOLD)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了 ReactionAtlas，一种可从少量种子分子自动、无规则地生成完整化学反应网络，并对碳水化合物体系展开大规模探索，成功发现完整的 Formose 反应循环及多条新路径。

**💡 创新点**

创新点在于引入单端生成式过渡态提议器（基于扩散模型）与 MLFF 判断器，配合动力学加权的探索策略，突破传统双端 TS 搜索与手工规则枚举的局限，实现从种子到数万反应的无监督、无规则生成。

**🔧 技术方法**

使用了 denoising diffusion generative model（MoreRed）提议 TS，MD-ET 机器学习力场预测能量、力与 Hessian，P‑RFO、NEB 等优化方法，微观动力学模型与 Eyring 速率定律相结合，并使用 Orca/pySCF 进行 DFT 计算与数据库构建。

**📊 数据集**

训练数据集包括 ωB97X‑D3 Grambow TS 数据集用于生成器，QCML 数据集用于力场；生成的网络包含约12,542种子分子、33,065 PES 极小点和47,000条反应；对比基准使用 Wikipedia 化合物列表、先前 Formose 数据库及 RGD1 等公开数据集。

**📈 对比分析**

与传统双端 TS 方法（NEB‑TS、DE‑GSM、xTB→DFT）比较：ReactionAtlas 的有效 TS 率在 86%→99%，单端搜索每成功 TS 平均耗时约 23 分钟，显著快于传统方法（270–416 分钟）。在质量上 MLFF 预测 TS 的 RMSD <0.5 Å，Dft 校正后有效率提升至 99%；在覆盖性上与 Wikipedia 结果一致，覆盖 80% 以上的 C4 甜糖，并覆盖 90% 的已知 Formose 反应。

**⚠️ 局限性**

局限性：目前仅在真空环境下工作，未考虑溶剂效应；计算资源限制导致 C5 及更大分子覆盖率下降；单端方法对未见化合物的泛化仍有限；对关键机理速率仍需实验或更精细计算验证。

---

## 38. Simple Supervision Is Hard to Beat: A Bitter Lesson from Sparse Target Labels in Domain-Adaptive Object Detection

**arXiv ID:** 2606.30795 | [PDF](https://arxiv.org/pdf/2606.30795v1)

**作者:** Lijun Zhang `[一作]` (Amazon Robotics), Mudit Agrawal `[通讯]` (Amazon Robotics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究在源无关域自适应目标检测中，如何有效利用少量随机标注的目标域标签，并提出了基准方法 Random‑Target Supervised Mixing (RTSM)。

**💡 创新点**

创新点在于将稀疏目标标签直接作为监督损失加入现有的无源自适应框架，验证其作为最小、最稳健的基准；随后系统评估了多种插件（伪标签选择、目标补全、优化控制）是否能进一步提升效果，结果显示直接监督最为可靠，插件效果高度依赖方法和设置。

**🔧 技术方法**

技术主要包括基于 Transformer 的目标检测器（DINO、DETA）、Mean‑Teacher 伪标签自训练框架、以及多种稀疏标签反馈插件（Prior‑Mapped Thresholding、Precision‑Calibrated Thresholding、Hard Query Recovery、Multi‑View Foreground Revival、PCGrad 等）。

**📊 数据集**

使用了 Cityscapes、Foggy Cityscapes（合成雾）和 BDD100K 三个公开检测数据集，在 Cityscapes→Foggy Cityscapes 与 Cityscapes→BDD100K 两个跨域任务上进行评估。

**📈 对比分析**

与三种基线（source‑only、full‑target oracle、pure SFDA）以及四种主流 SFDA‑OD 方法（PETS、LPU、LPLD、DDT）进行比较。RTSM 在 1%–10% 标注预算下平均提升 AP50 约 4–18 点；插件在不同方法间表现不一，平均增益从正向到负向波动，说明稀疏标签的最佳利用仍不稳定。

**⚠️ 局限性**

局限性包括：仅在 Transformer 目标检测器上验证；稀疏标签的最佳使用仍受目标检测方法影响；插件设计依赖于手工调参且在不同任务、不同检测器上不一致；未探讨更高效的主动选择或标签预算优化策略。

---

## 39. Cross-Modal Hierarchical Fusion for from Multi-Sensor Ground Observation

**arXiv ID:** 2606.30647 | [PDF](https://arxiv.org/pdf/2606.30647v1)

**作者:** Xinze Zhang `[一作]` `[通讯]` (University of Southern California), Xinze Zhang (University of Southern California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

融合多视角地面天空摄像、毫米波云雷达和天顶计测，利用物理一致性实现对云微物理状态和风场的4D重建

**💡 创新点**

提出层级跨模态注意力融合、多模态条件变分重构和基于3D相关体的运动估计三阶段架构

**🔧 技术方法**

采用EfficientNet-B3特征金字塔、跨模态注意力、条件变分自编码器、可微雷达与成像渲染、3D卷积U-Net和GRU迭代运动回归

**📊 数据集**

在中国北方半干旱SACOL站点收集的17小时多模态实测数据以及BOMEX LES合成数据集进行训练与验证

**📈 对比分析**

相较于NeRF、VIP-CT、3DeepCT、ERA5等基线，LWC MAE降至0.026 g m⁻³，风速MAE 1.18 m s⁻¹，显著优于所有对比方法

**⚠️ 局限性**

局限于浅层对流云场、单站点、对硬件与校准要求高，且变分后验假设为高斯，可能无法捕捉多模态不确定性

---

## 40. How Can AI Find My Model? A Model-Finding Experimental Study Considering Data Formats, Embeddings, and Retrieval Strategies

**arXiv ID:** 2606.30846 | [PDF](https://arxiv.org/pdf/2606.30846v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 41. From Search to Synthesis: Training LLMs as Zero-Shot Workflow Generators

**arXiv ID:** 2606.30704 | [PDF](https://arxiv.org/pdf/2606.30704v1)

**作者:** Gan Luo `[一作]` (Peking University), Wotao Yin `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 MetaFlow，一个基于元学习的框架，利用大型语言模型一次性生成任务级工作流，并通过两阶段训练（监督微调+强化学习）提升其在多任务和多工具环境下的效果。

**💡 创新点**

创新点在于将工作流生成转化为跨任务元学习问题，能够在不重新搜索或微调的情况下实现零射任务与新操作符的自动生成；同时结合可验证奖励的强化学习和代码化工作流表示，显著提升可复用性与可解释性。

**🔧 技术方法**

技术手段包括：代码化工作流表示、MetaGPT 执行环境、可验证奖励（RLVR）与 Group Relative Policy Optimization（GRPO）+ Common Random Numbers（CRN）强化学习、LoRA 微调、自然语言接口定义新操作符。

**📊 数据集**

使用的主要数据集为 GSM8K、DROP、MBPP、Humaneval（用于训练和验证），以及 HotpotQA（用于零射评估）。工作流样本通过自动合成生成，覆盖多种任务-操作符组合。

**📈 对比分析**

与手工工作流（如 CoT、Self-Consistency 等）和自动化优化工作流（ScoreFlow、AFlow、ADAS）对比，MetaFlow 在单次推理下平均准确率达到 78.8%，在 HotpotQA 零射中最高得分 0.74，超过基线 0.46；相比基于搜索的方案，成本显著降低，且实现了跨任务、跨工具的泛化。

**⚠️ 局限性**

局限性包括：零射生成中约 31% 的工作流因语法错误失效；对极其复杂或高维工具链的鲁棒性有限；当前仅单轮生成，缺乏交互式迭代优化。

---

## 42. Reframing AGI Confrontation with Off Earth Autonomy

**arXiv ID:** 2606.30666 | [PDF](https://arxiv.org/pdf/2606.30666v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 43. A Practical Implementation of Day-3 Cooperative Intersection with Automated Connected Mini-Cars

**arXiv ID:** 2606.30838 | [PDF](https://arxiv.org/pdf/2606.30838v1)

**作者:** Lorenzo Farina `[一作]` (University of Bologna), Alessandro Bazzi `[通讯]` (University of Bologna)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

使用1:10比例的联网自动驾驶小车在封闭的八字赛道上演示了交叉口的协同管理，车辆通过中心控制器实现不停车通过交叉口。

**💡 创新点**

首次将Moveover调度算法与实际小车、无线通信、ROS2软件栈集成，并通过MQTT+ITS‑G5实现实时协同。

**🔧 技术方法**

核心技术包括：NVIDIA Jetson Orin平台的自主驾驶堆栈、ROS2通讯、Hokuyo LiDAR+IMU定位、MQTT/ITS‑G5无线通信、Moveover调度算法。

**📊 数据集**

使用自建的车道/交叉口地图与预设轨迹作为实验数据，车间共4辆小车，最大速度1 m/s。

**📈 对比分析**

实验持续超过1小时，无碰撞且无停车；通过与传统红绿灯/右侧通行法比较，显示了更高的道路利用率与更低的等待时间。

**⚠️ 局限性**

局限性包括：规模受限仅4辆车；场景为单一八字交叉口，未涵盖复杂交叉口与多车流；备用模式切换后全车停驶；依赖高质量地图与足够的电量，未验证长时间连续运行。

---

## 44. Motion Planning in Compressed Representation Spaces

**arXiv ID:** 2606.30940 | [PDF](https://arxiv.org/pdf/2606.30940v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 45. Contrastive Reflection for Iterative Prompt Optimization

**arXiv ID:** 2606.30840 | [PDF](https://arxiv.org/pdf/2606.30840v1)

**作者:** Derek Koh `[一作]` (LinkedIn), Jingwei Wu `[通讯]` (LinkedIn)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Contrastive Reflection 框架，通过任务特定的结构化痕迹和对比切片迭代式优化 LLM 提示。

**💡 创新点**

创新点在于将错误与附近成功案例形成可解释的对比切片，交给教师 LLM 生成针对性编辑，并用验证门控保证改动有效且不回退。

**🔧 技术方法**

采用结构化 LLM 输出、信息增益二叉决策树切片选择、教师 LLM 生成提示、验证与回归检查等技术，在 HotpotQA 上实现检索增强 QA 程序。

**📊 数据集**

使用公开的 HotpotQA 数据集进行实验，LinkedIn 内部评分工作流作为动机但未公开。

**📈 对比分析**

与随机/失败‑only、GEPA‑light、MIPROv2‑light 等基线对比，单步 tree‑contrastive 修复将验证精度从 51.4% 提升至 60.4%，性能与现代优化器相当。

**⚠️ 局限性**

局限在于依赖结构化输出、仅单步实验、缺乏多任务/多阶段验证、回归约束未在公开实验中检验，以及树选择可能并非最优。

---

## 46. Drawing Out Legal Risks: Co-Designing with Lawyers to Predict and Manage Legal Uncertainties of Medical AI Tools

**arXiv ID:** 2606.30828 | [PDF](https://arxiv.org/pdf/2606.30828v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 47. An AI-Based Solution for Secure Service Provisioning in IoT

**arXiv ID:** 2606.30701 | [PDF](https://arxiv.org/pdf/2606.30701v1)

**作者:** Marco Arazzi `[一作]` (University of Pavia), Vinod P `[通讯]` (Cochin University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于深度强化学习（DRL）与联邦学习（FL）相结合的安全服务调度框架，能够在物联网（IoT）环境中根据用户安全需求和服务提供者的行为可信度动态选择服务并实时监控服务质量。

**💡 创新点**

创新点包括：① 将 Security Service Level Agreement (SecSLA) 与行为指纹（BF）结合，构建可量化的可信度评分；② 通过联邦学习实现多设备协同训练 BF 模型，既保持数据隐私又提升监测精度；③ 将可信度评分作为 DRL 奖励的一部分，形成安全与功能双重优化的决策过程；④ 使用区块链保存全局 BF 模型，实现可信共享。

**🔧 技术方法**

核心技术：深度强化学习（DQN/DRL）用于服务选择；联邦学习框架与 GRU 神经网络用于行为指纹构建与异常检测；SecSLA 语义化表示与欧氏距离度量；区块链用于模型共享与不可篡改；仿真环境与数据预处理流程。

**📊 数据集**

使用 UNSW IoT Traces 数据集（真实消费者 IoT 设备流量）进行 BF 训练与攻击场景评估。

**📈 对比分析**

对比方法：随机选择、基于 SecSLA 的贪心策略、仅安全感知、DRL+本地可靠性、DRL+阈值检测、完整系统。评估指标包括完成率（COP）、失败率、攻击交互次数、可信度下降、适应速度、等待次数等。实验结果显示完整系统在攻击后 1 天内将攻击交互降至 0，完成率保持 100%，并显著优于基线方案。

**⚠️ 局限性**

局限性：① 需要在安全的初始部署阶段完成 BF 训练，若初始已被攻击会影响模型；② 联邦学习与区块链的通信与存储开销在极大规模或受限资源设备上未充分评估；③ 实验规模相对有限，未覆盖多租户或跨域复杂场景；④ 对实时性与能耗影响的量化分析不足。

---

## 48. Criticality-Constrained Iterative Pruning for Energy-Efficient Spiking Neural Networks via Combined Importance Scoring

**arXiv ID:** 2606.30676 | [PDF](https://arxiv.org/pdf/2606.30676v1)

**作者:** Muhammad Hamza `[一作]` `[通讯]` (Indian Institute of Technology Kharagpur), Muhammad Hamza (Indian Institute of Technology Kharagpur)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种新型的基于关键性约束的二次规划剪枝(CQP)方法，用于在保持时序计算完整性的前提下，显著压缩尖峰神经网络的突触密度

**💡 创新点**

核心创新包括：① 用权重模长与代理梯度关键性相结合的精确重要性度量，完全避免连续松弛的二次规划误差；② 通过双重梯度与权重屏蔽消除Adam“僵尸权重”现象；③ 迭代剪枝与再计算关键性的自适应调度；④ 通过KL散度识别冗余时间步，实现额外10%能耗削减；⑤ 通过阈值扫描揭示“关键性悬崖”，验证SNN层级的临界大脑假说

**🔧 技术方法**

实现基于PyTorch的全流程管线：代理梯度关键性计算、权重模长/关键性结合的top‑k稀疏掩码、Adam优化器的双重屏蔽、K‑step迭代剪枝与再训练、KL散度时序分析

**📊 数据集**

使用三类数据集：控制性合成数据集（784维特征，10类），经典MNIST（60k训练/10k测试），以及Fashion‑MNIST（60k/10k）

**📈 对比分析**

与随机剪枝、单纯模长(ℓ1)剪枝、仅梯度剪枝等四种基线对比；在合成数据上90%稀疏度下，CQP保持86.8%准确率，高于ℓ1的76.3%；MNIST上90%稀疏度CQP 95.6%对比ℓ1 93.4%；Fashion‑MNIST 70%稀疏度CQP 84.8%略优于ℓ1 83.4%；在高稀疏度下CQP相对优势可达+76.1pp（仅梯度）和+10.5pp（ℓ1）

**⚠️ 局限性**

局限包括：仅针对输入层剪枝，未实现多层联合稀疏；关键性依赖特定的弧形代理梯度，未对其他代理梯度进行系统对比；缺少真实异步时序输入（如N‑MNIST、DVS‑CIFAR10）的验证；迭代剪枝引入额外训练开销；对极端稀疏度下的精度衰减原因尚不完全清晰

---

## 49. Editable Physically-based Reflections in Raytraced Gaussian Radiance Fields

**arXiv ID:** 2606.30861 | [PDF](https://arxiv.org/pdf/2606.30861v1)

**作者:** Yohan Poirier-Ginter `[一作]` (Université Laval), George Drettakis `[通讯]` (Inria, Université Côte d'Azur)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于高斯光场的光学可编辑技术：利用多视图的漫反射、镜面反射、深度、法向、BRDF等缓冲，构建单一高斯场，并通过路径追踪实现物理准确的多弹道镜面反射，从而实现实时、可编辑的反射效果，并能够重建看不见的反射物体。

**💡 创新点**

创新点包括：① 采用独立监督的漫/镜面分离，防止生成“假镜面几何”；② 在单一高斯场中同时优化漫反射与镜面反射，支持多弹道、可变粗糙度；③ 开发高效的高斯光线追踪器（OOBB、PPLL、前后融合、截断加速），实现与3DGS相当甚至更快的训练与实时渲染；④ 设计了一键逆渲染网络（微调Stable Diffusion 2）来预测所需的各个缓冲。

**🔧 技术方法**

使用的技术包括：3D Gaussian Splatting、Cook‑Torrance/Disney BRDF、路径追踪（漫反射缓存+镜面路径）、光线追踪加光谱、密集初始化、OOBB+OptiX加速、PPLL、前后融合、粗略截断、单步逆渲染网络（Stable Diffusion 2）。

**📊 数据集**

主要使用的训练与测试数据集有：自定义合成场景（高光材质、无透明），Hypersim 和 InteriorVerse（用于逆渲染网络的训练）。

**📈 对比分析**

在合成场景上与3DGS、Gaussian Shader、3DGS‑DR、Reflective‑GS、EnvGS 等方法对比，PSNR 与传统方法相当或略低，但在漫/镜面分离度量上明显优于对手；训练时间约比EnvGS快5×；在实时渲染时，帧率可达数十fps，性能优于3DGRT并接近3DGS。

**⚠️ 局限性**

主要局限：① 依赖逆渲染网络预测缓冲，当前网络精度不足导致最终图像质量受限；② 不支持透明材质；③ 阴影已烘焙，无法在编辑后实时更新；④ 对未观测到的反射物体的重建精度仍有限。

---

## 50. The Label Imitation Game: Turing Test Network for Zero-Shot Pseudo-Label Pruning

**arXiv ID:** 2606.30875 | [PDF](https://arxiv.org/pdf/2606.30875v1)

**作者:** Brent A. Griffin `[一作]` (Voxel51), Jason J. Corso `[通讯]` (Voxel51)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于图灵测试的标签模仿游戏（LIG），并通过训练的 Turing Test Network（TTN）对基础模型生成的伪标签进行零样本去噪与裁剪，从而提升检测任务的标签质量和下游模型性能。

**💡 创新点**

创新点包括：① 将伪标签去噪视为对抗性询问（LIG）框架；② 设计无监督、任务无关的 TTN 能在零样本条件下使用全局语义上下文判别标签真伪；③ 证明 TTN 能跨任务迁移，利用仅图像分类训练得到的模型对目标检测伪标签进行有效裁剪，实现类别复苏。

**🔧 技术方法**

核心技术为 transformer‑based 判别器，利用非掩蔽自注意力和 CLIP 特征对标签与图像进行语义与空间一致性评估；训练时采用 LIG 中的两种游戏（PLG 与 OCG）产生多样化的正负样本；并在基础模型训练和下游检测器训练中使用该判别器进行后处理。

**📊 数据集**

使用了 VOC、COCO、LVIS、BDD 四个多样化检测数据集，以及 ImageNet、Food‑101、CIFAR‑100 等分类数据集用于 TTN 预训练；伪标签来源于 GDINO、YOLOE、YOLOW 等开放词汇检测 VLM。

**📈 对比分析**

方法通过与基准阈值过滤、Oracle LIG（IoU 过滤）以及不同 VLM 的下游训练结果对比。实验显示：TTN 在三大 VLM 上提升 F₁ 分数约 4%–7%，TTN_D（任务微调版）提升 44% 的最差类别 F₁，且在所有数据集上均实现 mAP₅₀ 与 mAP₅₀:₉₅ 的显著提升，尤其在低置信度阈值时显著恢复零召回类别。

**⚠️ 局限性**

局限性包括：① 对极低置信度标签的裁剪仍可能误删真实标签；② TTN 仍受限于训练时使用的类不平衡与参考样本数量；③ 目前仅针对 2D 图像，未扩展到视频或 3D 模式；④ 需进一步引入多模型共识与空间冗余判别以捕捉更细微的幻觉。

---

## 51. Training Therapeutic Judges and Multi-Agent Systems for Human-Aligned Mental Health Support

**arXiv ID:** 2606.30887 | [PDF](https://arxiv.org/pdf/2606.30887v1)

**作者:** Mizanur Rahman `[一作]` (York University), Elham Dolatabadi `[通讯]` (York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个面向心理健康对话的评估-改进框架，包括训练开放源评估器TheraJudge和多智能体改进系统TheraAgent，用评估结果作为控制信号对生成回复进行结构化修正。

**💡 创新点**

创新点在于：① 通过偏好优化训练人类对齐的多维评估器，解决评估-行动鸿沟；② 设计由评估器、批评者、教练和治疗师组成的协同改进流程，将评估转化为可操作的、解释性强的修正动作；③ 将评估与生成解耦，提升安全性与可解释性。

**🔧 技术方法**

主要技术包括：偏好式强化学习（GRPO）训练评估器；多维结构化评分（7个心理维度）；多智能体协同改进（Critic、Coach、Therapist）；基于评估信号的目标化响应重写；使用Qwen2.5-7B训练评估器、LLaMA 3.1-8B生成回复。

**📊 数据集**

使用公开的 Mental‑Align‑100K 数据集（10k 对话用于评估器训练，1k 对话用于生成和人类评估，267 条低质量回复用于盲评），并在其中采样训练/测试集以防止信息泄漏。

**📈 对比分析**

与零样本、监督版、以及 GPT‑4o、Claude‑3.7、Gemini‑2.5‑Flash 等闭源模型对比。TheraJudge 的 ICC 在 0.88–0.99 之间，远超基线；TheraAgent 在盲评中平均提升 0.43 分（从 4.26 到 4.69），低质量回复平均提升 2.45 分，恢复率高达 94%，并保持高安全性与一致性。

**⚠️ 局限性**

局限性包括：数据规模受隐私/伦理限制，无法覆盖全部心理健康场景；闭源模型拥有更大、更透明的训练语料，导致开放源对比存在不公平；人类评估成本高，难以大规模扩展；评估器和改进流程对特定心理健康维度训练，迁移到其他领域需重新标注与校准。

---

## 52. From Grasps to Dexterity: Large-Scale Grasp Pretraining for Dexterous Manipulation

**arXiv ID:** 2606.30749 | [PDF](https://arxiv.org/pdf/2606.30749v1)

**作者:** Ying Yuan `[一作]` (Carnegie Mellon University), David Held `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DexCraft 仿真基准与 Dexonomy 预训练数据集，利用分层模仿学习实现多关节工具使用的高级抓取与操作。

**💡 创新点**

创新点在于将大规模抓取数据转换为完整轨迹预训练低层控制器，并通过高层关键点子目标预测实现更精细的手部定位，从而显著提升关节工具操作的样本效率和成功率。

**🔧 技术方法**

技术包括分层模仿学习框架、点云输入的 PointNet++ 高层子目标预测、3D Diffusion Policy (DP3) 低层控制器、以及基于关键点的高维子目标表示。

**📊 数据集**

使用 Dexonomy 抓取数据集（约 10.7k 物体、9.5M 抓取）生成 355k 轨迹进行预训练，并在 DexCraft（6 个关节工具使用任务）上进行微调。

**📈 对比分析**

与 DP、DP3 端到端方法以及无预训练的分层策略对比，实验显示在仿真和真实世界中平均成功率提升 33.3%（对比 DP3），并在所有 6 项任务中保持较高样本效率。

**⚠️ 局限性**

局限性包括：需要人工标注子目标限制扩展性、仅使用单一实例评估泛化、示例来自键盘遥控且缺乏触觉反馈，后续可探索自动子目标学习与更丰富的物体多样性。

---

## 53. Can Physician Expertise Improve Machine Learning Identification of Delirium?

**arXiv ID:** 2606.30651 | [PDF](https://arxiv.org/pdf/2606.30651v1)

**作者:** Xinyu Qin `[一作]` (University of Houston), Lu Wang `[通讯]` (University of Houston)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在六家多伦多医院的GEMINI数据库中，构建并评估了一套用户中心的交互式机器学习（UC‑iML）框架，用于识别住院病人的谵妄情况；

**💡 创新点**

核心创新在于将医师在每个开发阶段的专业判断嵌入特征选择、模型评估与解释中，实现了人机协同的“人‑在‑循环”学习；

**🔧 技术方法**

采用了梯度提升树、AdaBoost、逻辑回归等传统监督学习模型，并通过SHAP可解释性方法对特征贡献进行量化；

**📊 数据集**

使用了约3,862例标注住院记录（含行政、实验室、药物及放射文本特征）的GEMINI数据集，覆盖2010‑2015年和2018‑2020年两期；

**📈 对比分析**

与无医师介入的自动ML模型及基线模型对比，UC‑iML模型在F1、召回率、ROC‑AUC等指标上均优于对照组，且在时间滚动验证与跨期（模型漂移）测试中保持较高的稳定性；

**⚠️ 局限性**

局限在于使用相同特征集合评估模型稳定性，未针对不同阶段的完整数据进行全量优化，且对精确率与召回率平衡的阈值选择仍有待进一步改进。

---

## 54. AI Transparency: Governance Compliance or Stakeholder Requirements?

**arXiv ID:** 2606.30652 | [PDF](https://arxiv.org/pdf/2606.30652v1)

**作者:** Muneera Bano `[一作]` (CSIRO), Didar Zowghi `[通讯]` (CSIRO)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对澳大利亚政府机构公开的92份AI透明度声明进行实证评估，提出并应用RCIN框架对透明度与利益相关者的风险、控制、参与和需求进行校准化分析

**💡 创新点**

1）首次将需求工程视角与透明度评价结合，提出RCIN框架；2）揭示“透明度幻象”，即合规但缺乏针对高风险低控制受众的实质性信息；3）为透明度声明的需求校准提供可重复的评估工具

**🔧 技术方法**

需求工程方法、结构化评估量表、RCIN框架映射、手工评分与调和、质性观察

**📊 数据集**

包含92份澳大利亚公共部门AI透明度声明（来自CSIRO等机构）及对应的治理规范（DTA 1.1版）

**📈 对比分析**

采用基于量化得分的描述性统计与对照分析，比较不同利益相关者类的实现度；结果显示对高控制类的透明度实现率高（>80%），但对高风险低控制类的关键透明度实现率仅48%~73%；表明合规与实际满足需求存在显著差距

**⚠️ 局限性**

1）仅针对澳大利亚政府情境，外推性有限；2）缺乏利益相关者直接参与的需求验证；3）评估主要基于文本披露，未考虑机构内部沟通或实际运作；4）未对时间演变（透明度更新）进行纵向跟踪

---

## 55. Thinking Out Loud: Real-Time Deception Monitoring in Asymmetric LLM Negotiations

**arXiv ID:** 2606.30649 | [PDF](https://arxiv.org/pdf/2606.30649v1)

**作者:** Nolan Coffey `[一作]` (University of Tennessee), Nasir U. Eisty `[通讯]` (University of Tennessee)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了在非对称LLM谈判中使用轻量级链路思考监视器实时检测欺骗的可行性，实验以二手车销售情境为模型。

**💡 创新点**

创新点在于提出并实现了一个实时CoT监视器作为第三方，能够在谈判过程中即时警报卖方隐瞒信息，并系统评估其对买方决策和卖方行为的影响。

**🔧 技术方法**

采用LLM链路思考（CoT）监视技术（Qwen3.5-2B）、多代理自动化框架、手动审核精度评估与模型对齐策略。

**📊 数据集**

使用真实二手车上市数据（2016 Nissan Altima、Carvana价格、Kelley Blue Book估价）以及LLM生成的对话日志作为实验数据。

**📈 对比分析**

通过对100次试验的不同买卖模型组合进行统计，比较欺骗次数、成交率、走项率和成交价格；监视器精度为83.3%，监视下买方走项率显著提升，成交价下降但仍有高价残留。

**⚠️ 局限性**

局限性包括：仅在单一二手车场景下验证，模型规模差异导致监视器精度下降；未进行统计显著性检验；手工标注缺乏多评标一致性；仅检测明显对抗式欺骗，未覆盖隐蔽性或策略性省略。

---

## 56. BEST-RQ-2: Contextualize-Then-Predict, a Two-Step Approach for Self-Supervised Audio Representations

**arXiv ID:** 2606.30700 | [PDF](https://arxiv.org/pdf/2606.30700v1)

**作者:** Ludovic K. Tuncay `[一作]`, Thomas Pellegrini `[通讯]` (Université de Toulouse)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

该研究提出了 BEST-RQ-2，自监督音频表示学习方法，在保留 BEST-RQ 的冻结随机投影离散目标的基础上，引入两阶段上下文化-预测架构。

**💡 创新点**

创新点在于将掩码预测拆分为先上下文化再预测两步，并采用 Vision Transformer 作为编码器，以便仅处理未掩码区域；预测器仅在预训练期间使用，推理时消除额外计算。

**🔧 技术方法**

技术包括 log‑mel 频谱分块、随机投影离散目标、ViT 上下文编码器、轻量级 ViT 预测器、交叉熵损失、AdamW 优化器。

**📊 数据集**

数据集为 AudioSet 的 1.9M 个 10 秒音频片段，评估使用 X‑ARES 和 XARES‑LLM 两个跨域基准。

**📈 对比分析**

通过与原 BEST‑RQ、BEST‑RQ (ViT) 以及 Audio‑JEPA 等一阶段基线对比，BEST‑RQ‑2 在 X‑ARES 的线性探测与 kNN 任务中均取得最高或第二高的平均分数；在 XARES‑LLM 中表现与 BEST‑RQ 相当并优于 BEST‑RQ (ViT)。

**⚠️ 局限性**

局限性包括相较于基于条带分块的 BEST‑RQ，语音任务性能仍略低，且两阶段拆分虽然提升了音乐和环境声音的迁移，但对语音的提升有限。

---

## 57. A Stationary-Distribution Theory for Triplet-Based Plateau Search in Random Forest Ensemble-Size Selection

**arXiv ID:** 2606.30837 | [PDF](https://arxiv.org/pdf/2606.30837v1)

**作者:** Andrey A. Dukhovny `[一作]` (Sberbank), Andrey M. Lange `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并理论分析了基于三点平稳搜索的随机森林树数自适应调参方法，推导出其在几何网格上形成的出生-死亡马尔可夫链及其平稳分布、均值与方差的闭式近似。

**💡 创新点**

创新点在于把三点平稳搜索转化为可分析的马尔可夫过程，得到树数均值随容差 ε 以 O(ε⁻²) 规模增长、方差同样为 O(ε⁻⁴) 的定量关系，并揭示不同更新规则（原始与对称修改）及尺度因子 sf 对均值与方差的影响。

**🔧 技术方法**

使用马尔可夫链理论、折叠正态近似、局部平衡与 Fokker‑Planck 解释，结合 OOB 分数的方差衰减模型，推导出生-死亡链的转移概率和平稳解。

**📊 数据集**

主要在通用的表格数据集上进行理论验证与实验示例，数据来源为公开的 tabular 数据集（如 UCI、Kaggle 等）用于演示平稳区间与波动性，但论文核心为理论推导，无特定单一数据集的数值比较。

**📈 对比分析**

与传统单次早停、固定上限 T_max 或连续增大 T 的方法相比，平稳搜索在保持相同 OOB 容差下能够更稳健地定位有效树数，理论上均值随 ε 变化符合 O(ε⁻²)，相对方差保持在 0.4–0.7 左右，体现了更高的计算效率与不易过估的优势。

**⚠️ 局限性**

局限性包括：对 OOB 分数差的折叠正态假设、对独立性与无条件概率的简化、仅给出理论定量而未给出具体的轨迹估计方法，且在实际有限 HPO 轨迹中需进一步验证近似的精度与稳态收敛性。

---

## 58. When Calibration Rankings Reverse: Accuracy-Controlled Evaluation for Fair Comparison of LLMs

**arXiv ID:** 2606.30814 | [PDF](https://arxiv.org/pdf/2606.30814v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 59. ViTL: Temporal Logic-Guided Zero-Shot Natural Language Navigation via Vision-Language Models

**arXiv ID:** 2606.30696 | [PDF](https://arxiv.org/pdf/2606.30696v1)

**作者:** Kaier Liang `[一作]` (Lehigh University), Cristian-Ioan Vasile `[通讯]` (Lehigh University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 ViTL 框架，能够在完全未知环境中零样本完成包含多目标与时序约束的自然语言导航任务；

**💡 创新点**

创新点包括：①将自然语言命令一次性翻译为 LTL 公式并编译为 DFA，实现在线动态重规划与时序约束保证；②多通道值图与方向分数机制，利用 VLM 评估每个前沿方向，生成空间可变导航价值图；③通过 DFA 权重动态重排序，减少不必要探索；

**🔧 技术方法**

使用的技术包括：大型语言模型（LLM）进行 NL→LTL 翻译；线性时序逻辑（LTL）与确定有限自动机（DFA）实现高层规划；视觉-语言模型（VLM）如 LLaVA‑34B 进行方向分数和场景相关性评分；YOLOv7‑E6E 目标检测；点目标导航（PointNav）策略；前沿提取与平滑值图插值；

**📊 数据集**

实验数据集为 Habitat‑Matterport 3D（HM3D）ObjectNav v2 验证集，2000 条语义导航任务，六类目标（椅子、床、盆栽、马桶、电视、沙发）；

**📈 对比分析**

与单目标零样本方法（WMNav、GAMap、SG‑Nav‑GPT、VLFM、InstructNav 等）相比，ViTL 的单目标成功率 54.1%，SPL 30.9，表现竞争；在多目标任务上与 LLM‑planner 对比，翻译成功率 95.3%，完成率 100%，探索尝试次数更少，平均路径长度缩短约 15%；

**⚠️ 局限性**

局限性包括：只处理“到达对象”这一子任务，未涵盖操作或交互；依赖 VLM 与目标检测的准确性；在极端拥挤或视觉障碍环境下表现可能下降；未对真实机器人硬件进行验证。

---

## 60. The Consistency Dilemma in LLMs: Generator-Evaluator Agreement and Vulnerability to Mistakes

**arXiv ID:** 2606.30653 | [PDF](https://arxiv.org/pdf/2606.30653v1)

**作者:** Marina Mancoridis `[一作]` (Massachusetts Institute of Technology), Zoë Hitzig `[通讯]` (Harvard Society of Fellows)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种生成者-评估者自一致性（generator–evaluator self‑consistency）度量，并在10款前沿大模型、491个概念上通过自动化黑盒评估流水线进行实验；

**💡 创新点**

创新点包括：① 用概念级别的生成‑评估对测试模型在不同角色下对同一概念的一致使用；② 设计了三类一致性测试（MCQ扰动、推理理由、本体关系）；③ 将一致性与临床医生验证的错误关联，揭示一致性悖论——更一致的模型往往更易出现已知错误；

**🔧 技术方法**

技术手段：自动化黑盒流水线、生成‑评估双重提示、确定性评分规则、概率校正后的一致性分数、分位数与分数聚合、分数与误差关联的分数回归（分数对数回归）等；

**📊 数据集**

使用的数据集包括：MMLU、BBH（一般推理基准）、MedicalQA、GDPVal、FinanceQA（面向任务的基准）以及临床错误验证集 MedMistakes‑validated；

**📈 对比分析**

方法对比：在标准基准准确率基础上进行一致性分数归一化，并与误差频率进行统计回归；结果显示，在控制基准准确率后，模型的一致性分数与错误易感性呈显著正相关，表明一致性与可靠性并不成正比；

**⚠️ 局限性**

局限性：① 只统计通过门控的实例，可能忽略难题或不同模型的分布差异；② 概念提取依赖模型自身标签，可能导致概念误判；③ 关联分析为观察性，未证实因果；④ 医疗错误标签覆盖不均，结果受标签偏差影响。

---

## 61. Explainable Artificial Intelligence For The Detection and Characterisation of Stage B Heart Failure

**arXiv ID:** 2606.30665 | [PDF](https://arxiv.org/pdf/2606.30665v1)

**作者:** Ahmed M Salih `[一作]` (University of Leicester), Gerry McCanna `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了2026年前20项研究中XAI在检测Stage B心衰中的应用

**💡 创新点**

系统识别出XAI使用缺口及公平性与外部验证不足等关键问题

**🔧 技术方法**

采用系统检索、数据提取与定性分析方法

**📊 数据集**

使用Web of Science、Scopus和PubMed检索得到的20项文献数据

**📈 对比分析**

通过对XAI方法（SHAP、LIME等）、评价方式和结果的汇总，未发现统一性能对比，显示多样性和评估不足

**⚠️ 局限性**

XAI解释缺乏严格评估，性别/族裔考量不足，外部验证稀少，方法单一且结果异质性高

---

## 62. DANTE-W: Diffuse Albedo Neural Texturing in the Wild

**arXiv ID:** 2606.30677 | [PDF](https://arxiv.org/pdf/2606.30677v1)

**作者:** Guangyu Wang `[一作]` (Tsinghua University), Lu Fang `[通讯]` (Tsinghua University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一种 DANTE-w 神经纹理框架，用于从无结构图像集合中恢复大规模真实场景的高保真漫反射反照率纹理，并与传统 3D 重建流程无缝集成。

**💡 创新点**

结合视图空间扩散先验与 3D 一致的神经纹理，通过频率分离的物理驱动渲染分离反照率与辐照，解决了传统纹理烘焙中的阴影烘焙问题。

**🔧 技术方法**

使用多分辨率哈希编码+轻量 MLP 的 2D 神经纹理、基于散射光照的 3D 神经场、扩散模型先验、物理驱动神经渲染以及频率分辨的网络编码来实现高质量的 albedo 分离。

**📊 数据集**

创建了 GigaLit 基准，包含 6 个真实户外场景（数百至数千张无结构图像）和 10 个高分辨率合成物体，用于定量评估。

**📈 对比分析**

与 Metashape Pro、RGB↔X、Cosmos‑DiffusionRenderer、LightSwitch 等基线比较，DANTE‑w 在 albedo 恢复上 PSNR 提升约 +4–5 dB，LPIPS 降低 20–25%，在重光照下 PSNR 提升 6–7 dB，性能显著优于所有方法。

**⚠️ 局限性**

当前仅适用于近似朗伯光照的材质，无法处理非朗伯（镜面）效应，并且仍需先验网格重建；未来需扩展至更通用的场景表示和镜面材料。

---

## 63. Qualified Educational Capacity Planning under Heterogeneous Student Support Needs: A Synthetic Benchmark and Decision-Support Framework

**arXiv ID:** 2606.30650 | [PDF](https://arxiv.org/pdf/2606.30650v1)

**作者:** Carlos Eduardo Sanoja `[一作]` (Quanta Labs), Oscar Enrique Moreno Mayz `[通讯]` (Quanta Labs)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个可复现的教育资源规划基准与决策支持框架，评估多种服务与培训策略在不同冲击情境下的表现；

**💡 创新点**

首次构建了动态资格与服务不可存储的模型，并通过“反应可行”与“预先保险”两种调度范式的界面映射阐明了最佳策略取决于资格获取延迟与可恢复窗口的交互；

**🔧 技术方法**

采用了混合整数规划滚动时域控制、对照实验与配套的Python仿真环境 EduCapacity Studio；

**📊 数据集**

使用了合成的、种子控制的多种情境数据集，包括宣布与突发的新增支持类别、缺勤与需求激增；

**📈 对比分析**

对六种策略（服务仅、反应、静态保险、水填充、MPC、完美预知）进行成对比较，结果显示在“反应可行”情境下MPC在158个单元中全胜，且在训练延迟超过规划视野时静态保险可提升至两倍；

**⚠️ 局限性**

局限在于仅使用高度抽象的二参数资格模型、有限的工作人员与类别规模、未纳入真实学生或合规性数据、求解器可能不具备可扩展性，且未验证对实际教育影响。

---

## 64. Off the Rails: Hijacking the Scoring Head in Generative End-to-End Driving Planners with Safety-Violating Adversarial Perturbations

**arXiv ID:** 2606.30807 | [PDF](https://arxiv.org/pdf/2606.30807v1)

**作者:** Halima Bouzidi `[一作]` (University of California, Irvine), Mohammad Abdullah Al Faruque `[通讯]` (University of California, Irvine)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种针对生成式端到端自动驾驶规划器的对抗攻击框架Derail，专门攻击规划器的评分头并导致安全风险行为。

**💡 创新点**

创新点在于发现并利用了生成式规划器统一的“固定候选集+评分头”推理结构作为攻击面，并设计了无真值、以安全违规为目标的三项行为级损失。

**🔧 技术方法**

采用可微分的对抗梯度方法（PGD/Adam+EOT），在数字像素空间和物理补丁两种威胁模型下执行，利用BEV特征和候选轨迹集合。

**📊 数据集**

在NAVSIM安全基准上评估，针对DiffusionDrive、GTRS-DP、GTRS-Aug和GTRS-Dense四个最先进生成式规划模型。

**📈 对比分析**

与三种基线（无目标PGD、特征距离攻击、DP-Attacker）比较，Derail在所有模型上实现100%攻击成功率，碰撞率提升至25.8–60.4%，显著优于基线。

**⚠️ 局限性**

局限性包括对模型的白盒访问假设、攻击需要针对特定模型的梯度计算，且在物理补丁攻击中效果受场景多样性和泛化限制。

---

## 65. Toward AI-Resilient Assessment in Computer Science Courses in an AI-Native World

**arXiv ID:** 2606.30655 | [PDF](https://arxiv.org/pdf/2606.30655v1)

**作者:** Anshumali Shrivastava `[一作]` `[通讯]` (Rice University), Anshumali Shrivastava (Rice University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5`

**🎯 论文内容**

提出了一套针对高级计算机科学课程的 AI‑原生评估框架，利用可执行评估器和 AI‑生成的 Pareto 前沿来衡量学生提交的增量性能；

**💡 创新点**

创新点在于用 Pareto 超额（surplus）量化学生是否超越 AI 基线，并把 AI 使用本身纳入可测量的评估目标；

**🔧 技术方法**

使用可执行评估器、Pareto 前沿计算、服务器中介反馈、阈值化后续评估、以及 Bloom‑filter 近似成员资格任务的实验实现；

**📊 数据集**

采用基于合成的 Bloom‑filter 测试集，分别在 MB 与 GB 规模下生成负样本和隐藏评估基准；

**📈 对比分析**

通过与 AI 基线提交的多目标分数对比，计算超额体积（hypervolume）来评价性能，学生提交若能在多维指标上超越前沿则获得额外分数；

**⚠️ 局限性**

局限性包括对 AI 基线的依赖、需要大量人工制定评估器与前沿、对极端 AI 进化循环的假设、以及对非可执行评估目标（如沟通、伦理）的适用性不足。

---

## 66. Practitioners At The Limit: Bereavement, Mockery and Ideology in Response to Crisis

**arXiv ID:** 2606.30667 | [PDF](https://arxiv.org/pdf/2606.30667v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 67. Locker-based Truck-Drone Routing with Integrated Considerations of Pickups, Deliveries, and No-Fly Zones

**arXiv ID:** 2606.30680 | [PDF](https://arxiv.org/pdf/2606.30680v1)

**作者:** Xuanyu Liu `[一作]` (Chang'an University), Zhengbing He `[通讯]` (University of Nottingham Ningbo China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种考虑多产品上下游需求、无人机电池替换和禁飞区的基于智能柜的卡车-无人机协同配送问题（LTDRP-PDNF）并给出求解框架。

**💡 创新点**

创新点包括：①将问题建模为马尔可夫决策过程；②设计两阶段深度强化学习神经启发式（先学习 CVRP 的卡车路径，再通过迁移学习和混合调度启发式生成无人机路径）；③引入基于注意力的编码器+Bi‑GRU 解码器，以及 MBP‑ILS 混合调度启发式；④实现高效、可扩展的近似求解。

**🔧 技术方法**

采用注意力编码器、双向 GRU 解码器、REINFORCE 训练、2‑opt 局部搜索、MBP‑ILS（最大电池/负载+改进局部搜索）以及电池能耗与禁飞区绕行距离的计算。

**📊 数据集**

使用公开 CVRP 训练/测试集（20/50/100 顾客，10,000 条实例）和自建的 LTDRP‑PDNF 训练集（20/50/100 顾客，100 条实例），每个实例包含两种产品、随机禁飞区及装卸约束。

**📈 对比分析**

与 Gurobi（求最优）、ALNS、PSO 以及基于注意力的 DRL（AM）进行对比。实验表明：在 CVRP 上，提出方法的平均成本仅比最优低 0.66% 并且运行时间最短；在 LTDRP‑PDNF 上，在所有规模下均取得比基线更低的成本（N=20 差距 2.78%，N=50/100 为最优者），并且平均运行时间从 1.09s 到 14.79s，远快于基线。方法具有良好的规模推广和参数敏感性。

**⚠️ 局限性**

局限性：仅考虑静态禁飞区、固定时间窗口与需求；每辆卡车只能携带一架无人机；不考虑实时动态订单或多无人机协同；模型在非常大规模实例下仍可能需要改进收敛速度与可扩展性。

---

## 68. FAIR+S: A validation study of a framework for sustainable research data and software

**arXiv ID:** 2606.30663 | [PDF](https://arxiv.org/pdf/2606.30663v1)

**作者:** Danila Valko `[一作]` (Carl von Ossietzky Universität Oldenburg), Ralf Isenmann `[通讯]` (Wilhelm Büchner Hochschule)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过专家问卷调查和混合方法分析，对FAIR+S框架的相关性、可行性和价值进行系统验证，并提出改进建议。

**💡 创新点**

将可持续性考量系统性地嵌入FAIR原则，提出五条S1–S5可持续性原则，并通过实证研究首次验证其在科研与软件开发中的接受度与可操作性。

**🔧 技术方法**

采用设计科学研究（DSR）方法，构建专家调查问卷，使用定量统计（描述性统计、Wilcoxon符号秩检验、Friedman检验）和定性内容分析相结合的混合方法。

**📊 数据集**

收集了40名专家的问卷数据，包括研究经验、教育背景、对FAIR和可持续性概念的熟悉度，以及对S1–S5原则的重要性、价值和可行性的评估。

**📈 对比分析**

通过对专家评分进行统计比较，评估各原则的重要性、价值和可行性；结果显示原则被高度认可（平均评分≥3.7），但在可行性方面显著低于重要性，表明实践中仍存在显著差距。

**⚠️ 局限性**

样本规模有限（27名合格专家），且全部基于自我报告，未能提供实际应用的观察性证据；缺乏标准化工具与指标，也未在真实研究基础设施中进行试点验证。

---

## 69. NSynC: Normalised Synthesis of Computation

**arXiv ID:** 2606.30703 | [PDF](https://arxiv.org/pdf/2606.30703v1)

**作者:** Zoey Shepherd `[一作]` (University of Edinburgh), Elizabeth Polgreen `[通讯]` (University of Edinburgh)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于语义的程序合成方法——NSynC，通过枚举 STLC+（带加法的单一类型 λ 演算）的规范化形式来避免语义重复，从而加速程序合成。

**💡 创新点**

创新点在于将搜索空间从传统的语法树转移到唯一的语义规范化形式，并利用强等价公理（β‑η 等价和加法极限性）构造无重复的搜索空间。

**🔧 技术方法**

采用了规范化求值（Normalisation by Evaluation）技术、类型驱动的自上而下合成规则、大小限制与匹配深度限制，以保证搜索终止且搜索空间完整。

**📊 数据集**

使用了 166 个随机生成的合成基准（synthetic benchmark suite），包括 104 个两算法都能解决的任务和 37 个仅前者能解决的任务。

**📈 对比分析**

与传统的无约束枚举算法（*）相比，NSynC 在可解基准上平均获得 8.93 倍的速度提升，且枚举的匹配项数明显减少；但由于强制的匹配项顺序，有 37 个基准无法求解。

**⚠️ 局限性**

限制主要体现在：1) 强制匹配项顺序导致某些程序无法覆盖所有示例；2) 目前仅适用于 STLC+，需进一步扩展到更大或更复杂的语言片段。

---

## 70. When transformers learn "impossible" languages, what do they learn?

**arXiv ID:** 2606.30815 | [PDF](https://arxiv.org/pdf/2606.30815v1)

**作者:** Ram Janarthan `[一作]` (University of Edinburgh), Sharon Goldwater `[通讯]` (University of Edinburgh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文训练GPT‑2小模型，对通过词序扰动生成的“不可想象语言”进行语法敏感性和生成能力的评估，并探讨这些行为与人类语言不出现的可能关联。

**💡 创新点**

创新点在于提出两种理论驱动的链接假设（语法敏感性缺陷与生成缺陷），并通过BLiMP最小对比和逆向生成评估方法证明生成性能更能解释不可想象语言的非出现。

**🔧 技术方法**

使用了GPT‑2小型Transformer、BLiMP最小对比评估、GPT‑2‑large评估器进行perplexity评估以及4‑local entropy计算。

**📊 数据集**

使用的数据集包括BabyLM英语儿童语料库的可逆扰动版本、相应的BLiMP任务数据，以及预训练的大型语言模型用于生成质量评估。

**📈 对比分析**

通过将BLiMP准确率与4‑local entropy和perplexity关联，发现4‑local entropy更能解释模型表现；在生成实验中，绝大多数不可想象语言的高质量句子比例远低于英语，尤其在句子长度增长时显著下降。

**⚠️ 局限性**

局限性包括仅采用单一Transformer架构和单一语言（英语），评估仅依赖perplexity而非人工判断，未考察跨语言或更大模型的泛化，且迭代学习过程未充分模拟人类语言传播。

---

## 71. A Lean 4 Formalization of Scott's \emph{Continuous Lattices} (1972)

**arXiv ID:** 2606.30782 | [PDF](https://arxiv.org/pdf/2606.30782v1)

**作者:** Lars Warren Ericson `[一作]` `[通讯]` (Catskills Research Company), Lars Warren Ericson (Catskills Research Company)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

完成了对 Dana Scott 1972 年《连续格》一书的完整机证化，并实现了 43 条编号命题的 Lean4 证明；

**💡 创新点**

首次将 Scott 的 D∞ λ‑计算机模型正式化，并在 Lean 4 中实现 Milner 的 1972 年修正，使得整个理论完全可检验；

**🔧 技术方法**

采用 Lean 4 + mathlib 进行类型理论与拓扑/格理论的结合，使用了可数层次的递归构造、伴随等概念，并通过多模组的依赖关系管理实现了完整证明；

**📊 数据集**

无数据集；

**📈 对比分析**

本工作不涉及实验性能比较，而是通过 Lean 内核验证所有证明无错误；证明检查时间短，证明尺寸可追踪；

**⚠️ 局限性**

局限在于使用经典逻辑，未实现构造性版本；对某些关键步骤仍需人工编写，缺乏通用模板；

---

## 72. Optimal Auctions for Constrained Buyers

**arXiv ID:** 2606.30776 | [PDF](https://arxiv.org/pdf/2606.30776v1)

**作者:** Batya Berzack `[一作]` (Tel Aviv University), Inbal Talgam-Cohen `[通讯]` (Tel Aviv University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在买家受约束（如不超标投标、阶段性无债务等）下的多单位多买家拍卖，构建了一套新的理论框架，并在不同目标（收益对齐与消费者对齐）下给出了最优机制的构造与证明。

**💡 创新点**

① 首次将 Myerson 机制的核心思想推广到“无上限单调约束（UM）”的买家；② 发现收益对齐目标下仍可使用 Myerson 机制，消费者对齐目标则能显著超越传统激励兼容机制；③ 设计了一种基于测度保持重排的变换工具，用以在受约束环境中实现分配规则的单调化与支付的最优化。

**🔧 技术方法**

① 测度保持重排（Measure‑preserving rearrangement）和 Hardy‑Littlewood 不等式；② 虚值（virtual value）与铁化虚值（ironed virtual value）的推广；③ 通过对支付与分配规则的层次化改造实现真诚性、单调性与支付最优的统一结构；④ 采用阶段筛选（threshold）拍卖与两阶段支付设计以满足阶段性 IR 约束。

**📊 数据集**

本研究为纯理论分析，未使用任何实验或实证数据集。

**📈 对比分析**

由于是理论证明，没有对比实验结果；通过构造与分析证明了在收益对齐目标下 Myerson 机制仍最优，而在消费者对齐目标下设计的机制可严格提升卖方收益（或消费者剩余）。

**⚠️ 局限性**

① 仅覆盖无上限单调约束（UM）与特定的阶段性 IR 约束，无法直接推广至预算或收益率约束等有上限的情形；② 对于不规则分布的处理仍较困难；③ 机制设计假设买家信息分布连续且无质量点，实际分布不满足时需进一步研究；④ 实验验证与实现细节尚未给出。

---

## 73. Accelerometry-Derived Digital Biomarkers for Cardiometabolic Risk: A Population-Representative Tabular Benchmark with Uncertainty Quantification

**arXiv ID:** 2606.30702 | [PDF](https://arxiv.org/pdf/2606.30702v1)

**作者:** Federico Felizzi `[一作]` `[通讯]` (SIIAM), Federico Felizzi (SIIAM)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于NHANES 2003–2006的加速度计心血管代谢基准，用于评估数字生物标志物的预测性能。

**💡 创新点**

①首次构建包含调查权重、少数族裔过度抽样、临床验证结局及时间结构的真实健康数据基准；②结合分裂式合规预测探讨公平性缺口；③引入TabPFN v2等表格基础模型在小样本健康数据中的优势。

**🔧 技术方法**

使用Ridge回归、XGBoost梯度提升树、以及TabPFN v2表格基础模型；采用分裂式合规预测生成无分布假设的90%预测区间；通过性别和种族亚组进行公平性评估。

**📊 数据集**

NHANES 2003–2006年成人样本1381例，包含臀部加速度计数据、空腹实验室生物标志物（HbA1c、甘油三酯、CRP）、膳食摄入与体格测量。

**📈 对比分析**

在R^2、MAE和预测区间覆盖率上比较三种模型。TabPFN v2在所有三种结局上均取得最高R^2（HbA1c 0.156，CRP 0.383，甘油三酯 0.048），并在CRP上覆盖率最高（0.939）。XGBoost略低，Ridge最差。甘油三酯总体不可预测（R^2≈0）。

**⚠️ 局限性**

局限性包括单一随机拆分导致结果不稳健；样本量小且少数族裔比例受限；仅评估三种模型和三种结局；加速度计为臀部单轴设备，难以迁移至现代腕式多轴传感器；仅单次膳食回忆与无遗传信息；分裂式合规预测仅保证边际覆盖，缺乏条件公平性保证。

---

## 74. A Transferable Learned Temporal Prior for Transmission Reconstruction and Decision-Relevant Uncertainty in Real Outbreak Labels

**arXiv ID:** 2606.30842 | [PDF](https://arxiv.org/pdf/2606.30842v1)

**作者:** Md Ahsan Karim `[一作]` `[通讯]` (National Institute of Textile Engineering and Research), Md Ahsan Karim (National Institute of Textile Engineering and Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了一种从多疾病数据学习并锁定的时序先验模型，用于在新爆发（如Andes病毒）中进行父子关系的候选排序，并系统评估传染标签和图结构不确定性对源优先级决策的影响。

**💡 创新点**

创新点：① 通过留一疾病交叉验证训练并“锁定”时序先验，实现零射击迁移到目标疾病；② 系统量化并分析传染标签与网络结构的不确定性，展示其对重建结果和优先级决策的可观影响。

**🔧 技术方法**

技术手段包括逻辑回归时序先验、留一疾病交叉验证、对照高斯/KDE/Gamma/Lognormal参数化先验、Bootstrap、符号检验、任务逆转鲁棒性、Jaccard相似度、Gini系数等统计与可视化方法。

**📊 数据集**

数据集：D1（11类疾病，14,919条候选），ANDV 2014‑2019（29条严格父子任务），NYC 2022 mpox（75条联系对），广东Delta Delta图（131节点/142边）。

**📈 对比分析**

与四个公平源训练的参数化时序基线相比，锁定先验在ANDV严格基准上的MRR从0.274提升至0.571，Top‑1提升至0.379，p<0.0002；在MPXV中约55%联系对不确定，广东Delta中不确定边导致前5源优先级变更，决策后悔率约为0.07–0.04。

**⚠️ 局限性**

局限性：① 仅在ANDV严格任务做了严格验证，样本量有限；② 传染标签可靠性依赖调查设计，残留噪声可能影响评估；③ 未使用基因组或混合模型，仅限时序信息；④ 对公共数据库的基准构建极度缺乏可用标签，限制了评估范围。

---

## 75. Algorithms and complexity for geodetic sets on interval and chordal graphs

**arXiv ID:** 2606.30882 | [PDF](https://arxiv.org/pdf/2606.30882v1)

**作者:** Dibyayan Chakraborty `[一作]` (University of Leeds), Dimitri Lajou `[通讯]` (University of Bordeaux)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究了最小测地集问题在弦图和区间图上的复杂度，证明其在弦图上按团数/树宽参数化是FPT的，而在区间图上则是NP‑难的。

**💡 创新点**

提出了针对弦图的树宽参数化FPT算法，构造了高效的状态表示与动态规划，并给出区间图上的多项式规模归约，首次确立该问题在区间图上的NP‑难性。

**🔧 技术方法**

主要技术包括：弦图的树分解与nice树分解、状态压缩与动态规划、利用团分解与剪枝、以及对区间图的诱导路径结构与最短路分析、基于三SAT的多项式时间构造与硬件性证明。

**📊 数据集**

未使用任何实验数据集，全部为理论构造与证明。

**📈 对比分析**

通过理论证明与归约，没有实验比较；在弦图上算法时间为f(ω)·n（f为指数函数），在区间图上给出NP‑难性的下界，说明不存在 2^o(√n) 的算法（假设指数时间假设）。

**⚠️ 局限性**

局限性在于：对一般区间图仅得到NP‑难性证明；在弦图的部分树宽（k‑树以外）上仍未给出多项式或FPT算法；并且未提供实验验证或近似算法。

---

## 76. A Single Rewrite Suffices: Empirical Lessons from Production Skill Description Optimization

**arXiv ID:** 2606.30775 | [PDF](https://arxiv.org/pdf/2606.30775v1)

**作者:** Yangqiaoyu Zhou `[一作]` (Microsoft), Yaar Harari `[通讯]` (Microsoft)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了一套自动化的技能描述优化管道，利用 LLM 重写和错误反馈来解决企业 AI 助手中的技能冲突问题。

**💡 创新点**

核心创新在于发现单次 LLM 重写即可达到与人工调优相同的路由准确率，并提出基于训练-验证 F1 差距的诊断指标，用于判断是否需要架构层面的改动。

**🔧 技术方法**

使用了 LLM 初始化、错误反馈循环（或单次重写）以及混合检索（BM25+向量）等技术；通过系统的消融实验验证了各组件的重要性。

**📊 数据集**

实验数据来源于两大场景：生产企业组群聊天代理（9 个技能，372 条回归测试案例）和 ToolBench（约 16k RESTful API 工具，I2 子集的 48 个工具）。

**📈 对比分析**

与手工调优的技能描述进行对比，单次重写在生产环境下实现了 79.2% F1（与人工 79.4% 相近），并将工程时间从 120 分钟压缩到 3.8 分钟（约 32 倍加速）；在 ToolBench 开放世界检索任务中，单次重写提升了约 4.45% 的 F1。

**⚠️ 局限性**

局限性包括：仅在 9 个技能、372 条测试案例的生产环境中验证，ToolBench 的原始描述为占位文本，开放世界实验仅覆盖 I2 子集，且对少数技能存在无法仅通过描述解决的语义重叠问题。

---

## 77. Wait, am I Being Fair? Characterizing Deductive Stereotyping and Mitigating It with Fair-GCG

**arXiv ID:** 2606.30989 | [PDF](https://arxiv.org/pdf/2606.30989v1)

**作者:** Naihao Deng `[一作]` (University of Michigan), Rada Mihalcea `[通讯]` (University of Michigan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了大语言模型在公平推理任务中的缺陷，提出并验证了“推理时注入短语”干预方法来减少歧视性推理；

**💡 创新点**

创新点在于首次将推理时的短语注入与基于梯度的坐标搜索（Fair‑GCG）相结合，自动发现高效的公平引导词；

**🔧 技术方法**

采用链式思维推理、贝叶斯统计建模、梯度引导离散优化（Greedy Coordinate Gradient）以及对数似然目标的自定义损失；

**📊 数据集**

使用 BBQ、CrowS‑Pairs、GenMO、StereoSet、WinoQueer 五大公平基准，及 Bias‑in‑Bios 职业筛选数据集进行评估；

**📈 对比分析**

与多种现有偏差缓解方法（如 ADBP、IASC、SD‑E/SD‑R）对比，注入短语在所有基准上均达到或超过最高分，尤其在 Qwen‑2.5‑7B 上取得四项中最好成绩；

**⚠️ 局限性**

局限包括：注入短语往往缺乏可读性，更新仅逐坐标进行，计算成本高；并且该方法无法消除所有根深蒂固的训练偏差，且可能引入新的失败模式。

---

## 78. Why Do Few-Step Text Latents Fail When Image Latents Work? Non-Commitment at Sharp Categorical Readouts

**arXiv ID:** 2606.30705 | [PDF](https://arxiv.org/pdf/2606.30705v1)

**作者:** Zhongyao Wang `[一作]` `[通讯]` (Fudan University), Zhongyao Wang (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

分析了为什么确定性少步生成模型在连续文本潜在空间中会产生不连贯文本，而在图像潜在空间中能正常工作，并阐明了读出锐度与离散分支决策不匹配是根本原因。

**💡 创新点**

提出了读出锐度（DABI）与离散承诺度（CCI）两项诊断指标，并给出非承诺定理与接口能量等几何理论，构建了逃逸机制与分类图。

**🔧 技术方法**

使用几何概率、拉普拉斯展开、Coarea公式、接口能量、流插值、后验均值分析、实验检验与对照等技术手段。

**📊 数据集**

使用ELF-B文本autoencoder、LangFlow、CoLa-DLM、Cosmos等公开检查点，OpenWebText验证集，以及四个图像VAE（Lumina-Next、SANA-1.5、Z-Image、FLUX.1）。

**📈 对比分析**

通过DABI/CCI诊断、oracle roll‑in等实验对比确定性ODE与SDE、离散提交与连续提交，发现确定性ODE在K≤16时文本失败，SDE与离散提交可在较少步数内生成流畅文本；数值上DABI≈1对应图像，≈10^2–10^4对应文本。

**⚠️ 局限性**

仅对确定性后验均值传输的理想化模型给出下界，无法直接约束随机或离散生成器，对标签噪声、非光滑密度等实际情况的影响有限。

---

## 79. Agentic AI Enhances Physician Trust in Clinical Decision Making

**arXiv ID:** 2606.30658 | [PDF](https://arxiv.org/pdf/2606.30658v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 80. Citation Discipline in Spec-Driven Development: A Cross-Model Empirical Study of Output Determinism and Automated Hallucination Detection in LLM-Generated Code

**arXiv ID:** 2606.30689 | [PDF](https://arxiv.org/pdf/2606.30689v1)

**作者:** Subham Panda `[一作]` `[通讯]`, Subham Panda

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对三种基于LLM的代码生成框架（traceSDD、Spec Kit、OpenSpec）进行了跨模型（Claude Sonnet 4.6 与 GLM‑5‑turbo）控制实验，研究强制性行内引用对输出确定性与幻觉检测的影响。

**💡 创新点**

创新点在于首次系统评估行内引用作为可验证约束的“确定性‑检测”权衡，并在两种不同架构的LLM上复制验证。

**🔧 技术方法**

技术包括基于RE​Q‑XXX.Y.Z层级规范的行内引用、外部追踪映射、自动化幻觉检测（orphan‑REQ检查）以及 Levenshtein Set Similarity（LSS）衡量输出确定性。

**📊 数据集**

使用了70个Python任务（20个Claude任务、50个GLM任务），涵盖8个软件工程领域、3个难度等级和2个规模类别，共生成840份实现。

**📈 对比分析**

比较方法是对每个条件（引用/不引用/Spec Kit/OpenSpec）执行3次独立LLM会话，统计LSS和幻觉检测率；结果显示不引用条件在确定性上优于引用条件（Claude d≈‑0.76，GLM d≈‑0.72），而引用条件在幻觉检测率上显著高于0%（约86‑88%），且FPR为0%。

**⚠️ 局限性**

局限包括仅评估Python代码、任务规模相对较小、仅使用两种LLM，且幻觉注入方式人为，未覆盖自然幻觉及更大项目规模的可扩展性。

---

## 81. Streaming Gaussian Encoding for 4D Panoptic Occupancy Tracking

**arXiv ID:** 2606.30754 | [PDF](https://arxiv.org/pdf/2606.30754v1)

**作者:** Maximilian Luz `[一作]` (University of Freiburg), Abhinav Valada `[通讯]` (University of Freiburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究提出了流式高斯编码器，用于在多视角摄像机输入下维护持久且时序一致的4D全景占据表示；

**💡 创新点**

其创新点在于将高斯查询作为持久隐式体素状态，并通过高斯不透明度与深度监督来代理可见性，实现基于置信度的查询保留与刷新，从而获得表征级时间一致性；

**🔧 技术方法**

采用高斯点云拆分、基于FiLM的惯性补偿、深度监督的高斯不透明度、混合排名裁剪、热身多帧训练以及MaskFormer/LaGS等掩码解码器等技术；

**📊 数据集**

在Occ3D-extended nuScenes 与 Waymo 数据集上进行评估；

**📈 对比分析**

与现有最优方法（如LaGS、TrackOcc）相比，在STQ、AQ等指标上提升约2–3个百分点，且推理吞吐量几乎不受影响；

**⚠️ 局限性**

局限在于对动态物体的补偿仅采用简单衰减，对极端遮挡下的高斯初始化仍有误差，未来需进一步提升动态场景鲁棒性与自监督能力。

---

## 82. What Drives Interactive Improvement from Feedback?

**arXiv ID:** 2606.30774 | [PDF](https://arxiv.org/pdf/2606.30774v1)

**作者:** Bartłomiej Cupiał `[一作]` (University of Warsaw), Piotr Miłoś `[通讯]` (University of Warsaw)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多轮语言代理场景下，构建了一个零样本学生-教师交互协议，系统评估自然语言反馈在提升模型表现中的作用，重点区分反馈带来的增益与仅靠多次尝试或自我修正的提升；

**💡 创新点**

通过跨模型学生-教师全矩阵实验，将学生能动性（利用反馈的能力）与教师生成质量分离；发现自我反馈对提升效果有限，只有高质量外部教师能显著提升；教师的单轮任务表现只能部分预测其交互教学能力；

**🔧 技术方法**

使用零样本生成式语言模型作为学生与教师（13个开放权重模型），采用“反馈-自我修正”对照、可变历史长度、教师特权信息（答案/完整解答）等技术；

**📊 数据集**

在四个可验证推理环境中实验：Omni-MATH（数学竞赛）、Codeforces（编程）、BBEH Linguini（BIG-Bench Extra Hard）、ARC-AGI1（网格转化谜题）；

**📈 对比分析**

将交互性能与基准模型的单轮准确率、无反馈自我修正、不同历史长度及特权信息等进行对比；结果显示：教师质量提升可为学生带来9–17点的@10增益；大多数增益集中在首次反馈回合；学生对反馈的利用能力决定大部分交互增益；

**⚠️ 局限性**

限制包括：仅评估短期可验证任务、未覆盖长周期交互与持续状态场景、缺乏对教师泄漏风险的系统性评估、实验集中在开源模型，未验证对闭源系统的普适性。

---

## 83. AI-Generated PowerShell Malware: An Experimental Framework and Dataset

**arXiv ID:** 2606.30819 | [PDF](https://arxiv.org/pdf/2606.30819v1)

**作者:** Luciano Pianese `[一作]` (Università degli Studi di Napoli Federico II), Roberto Natella `[通讯]` (Gran Sasso Science Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了 PSStrikes 数据集，开发了 PSSandman 沙箱，并使用开放权重 LLM 评估其生成 PowerShell 恶意软件的能力。

**💡 创新点**

创新点在于：①提供了人工标注的真实 PowerShell 恶意代码与自然语言描述的组合数据集；②设计了三阶段评估框架（代码相似度、代码质量、动态行为）；③实现了专门针对 PowerShell 的动态分析工具 PSSandman。

**🔧 技术方法**

主要技术包括：大语言模型微调与 QLoRa 量化、代码相似度指标（ChrF、METEOR、CrystalBLEU）、静态分析（PSScriptAnalyzer）、动态沙箱（VirtualBox+Sysmon+Sigma 规则）以及基于集合相似度的行为评估。

**📊 数据集**

使用的数据集为 PSStrikes，包含 1,944 个真实恶意 PowerShell 脚本（1,127 一行命令 + 817 多阶段脚本），每个脚本配有手工编写的自然语言说明。

**📈 对比分析**

比较方法为多维度：代码相似度指标、语法正确率与 PSScriptAnalyzer 违规率、以及基于 Sigma 规则的 Jaccard/Dice/Exact Match 指标。实验显示 DeepSeek 在多项指标上领先，量化模型性能下降显著，约 50% 的生成脚本在行为上与真实样本完全一致。

**⚠️ 局限性**

局限性包括：①沙箱环境无法完全模拟真实企业网络与 EDR 监测；②模型可能受限于预训练数据的缺失，难以覆盖所有 PowerShell 变体；③评估依赖手工标注，存在主观偏差；④未公开真实恶意代码，无法验证模型在攻击链完整性上的表现。

---

## 84. When Do Staging Annotations Preserve Semantics? Mechanizing Typed Semantics-Preserving Multi-Stage Programming with Let-Insertion (Extended Version)

**arXiv ID:** 2606.30854 | [PDF](https://arxiv.org/pdf/2606.30854v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 85. Local Pheromone Network: Sparse Local Learning with Multi-Scale Synaptic Trails, Consolidation, and Replay

**arXiv ID:** 2606.30669 | [PDF](https://arxiv.org/pdf/2606.30669v1)

**作者:** Xingcheng Fu `[一作]`, Zhihao Li `[通讯]`

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出了一种稀疏局部学习的Local Pheromone Network，利用信息素痕迹、预算化更新、巩固和重放等机制实现手动局部权重更新，无需自动微分。

**💡 创新点**

创新点在于：①以信息素为媒介的局部更新规则；②预算化的突触选择与自适应调整；③基于信息素成熟度的巩固机制；④结构可塑性与局部重放的结合；⑤在合成任务上演示了可分区记忆与冲突管理。

**🔧 技术方法**

使用的技术包括：手动前向传播与更新的稀疏层实现、信息素门控、短期/长期信息素双尺度衰减、基于误差与共活性的突触评分、预算自适应机制、巩固与结构可塑性、局部重放、可选的局部对比学习和混合卷积+记忆分支。

**📊 数据集**

使用的数据集为一系列人工合成任务：本地回归、分区记忆、冲突记忆、巩固冲突、结构可塑性与重放测试、局部对比学习以及混合滑动卷积+局部记忆的长上下文规则。

**📈 对比分析**

实验并未与大规模基准模型进行对比，而是通过自定义评测指标展示效果：本地回归MSE从1.17降至0.008；分区记忆在冲突任务中保持1.07倍；在巩固冲突实验中，巩固显著降低遗忘（1481×→49×），但新任务学习速率下降；混合模型在长上下文任务中，记忆分支单独取得100%准确率，混合后仍保持高准确率。

**⚠️ 局限性**

局限性包括：输入输出维度固定、结构可塑性仅在预留槽位内调节、对超参数高度敏感、隐藏层信用分配近似、缺乏优化的稀疏CUDA核、重放缓冲简单顺序插入、实验规模有限且未覆盖大规模序列/通用任务。

---

## 86. Understanding Censorship in Large Language Models: From Mechanisms to Governance

**arXiv ID:** 2606.30661 | [PDF](https://arxiv.org/pdf/2606.30661v1)

**作者:** Quanyan Zhu `[一作]` `[通讯]` (New York University Tandon School of Engineering), Quanyan Zhu (New York University Tandon School of Engineering)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统综述了大语言模型的审查机制（硬审查、软审查与框架效应）、跨国跨模型的实验数据、治理与监管框架，并提出可审计、多元治理的理论。

**💡 创新点**

首次将硬审查、软审查与信息框架化三类审查行为统一进一套可衡量的框架，强调审计透明度和多元治理的重要性。

**🔧 技术方法**

主要采用行为测试、跨语言审计、模型内部探测、对抗攻击与基准对比等方法，形成了一套多维度的审查检测与评估流程。

**📊 数据集**

使用公开的审计问卷（政治敏感提示集、对抗性提示）、多语言对照数据集、用户感知调查数据，以及跨模型公开检验集来进行实验。

**📈 对比分析**

通过拒绝率、软审查比例、跨模型一致性、纵向变化等指标进行比较，发现不同地区和供应商在审查强度、倾向与透明度方面存在显著差异。

**⚠️ 局限性**

缺乏统一的软审查度量标准、数据集主要集中于英语和中文、对代理式AI及多语言环境的评估不足、监管与可解释性机制仍不完善。

---

## 87. SyncCache: Exploiting Asymmetric Dynamics for Fast Audio-Driven Portrait Animation

**arXiv ID:** 2606.30849 | [PDF](https://arxiv.org/pdf/2606.30849v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 88. Enhancing BEST-RQ Pseudo-Label Quality through Online Refinement for Automatic Speech Recognition

**arXiv ID:** 2606.30671 | [PDF](https://arxiv.org/pdf/2606.30671v1)

**作者:** Jingjing Xu `[一作]` (RWTH Aachen University), Hermann Ney `[通讯]` (RWTH Aachen University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

改进了 BEST‑RQ 的伪标签生成，使用在线增量 PCA 替代随机投影、在训练中迭代更新代码本，以及引入中间层的代码本蒸馏，以得到更具判别力的伪标签并提升语音表示学习的效果。

**💡 创新点**

核心创新在于：①将随机投影改为可在线更新的 PCA 投影，②在训练过程中对代码本进行迭代细化，使其更贴近数据分布，③通过对中间层的时序相似度矩阵进行蒸馏，进一步增强伪标签的时序与语义信息，同时保持了原 BEST‑RQ 的简洁实现。

**🔧 技术方法**

使用了 self‑supervised 预训练（BERT‑style 掩码预测）、Conformer 编码器、增量 PCA、K‑means 代码本细化、相似度矩阵蒸馏、CTC 微调、SpecAugment 数据增强、Viterbi 4‑gram 解码等技术。

**📊 数据集**

在 960h 的 LibriSpeech 上进行无监督预训练；在 100h、10h 及 1h 的 Libri‑Light 语料上进行监督微调。

**📈 对比分析**

通过对比基线 BEST‑RQ、BiRQ 以及使用多代码本的方案，实验表明：单代码本加 PCA + 迭代细化即可将 test‑other 的 WER 从 10.1% 降低到 9.2%（与使用 6 个随机代码本相当），再加上代码本蒸馏后进一步降至 8.8%（约 12% 相对提升），训练时间相对基线减少约 45%。

**⚠️ 局限性**

仍受限于代码本随机初始化的波动；蒸馏机制仅在训练后约 30% 位置启用，对低资源或非英语语料的泛化尚未验证；实现增量 PCA 与代码本细化的代码量虽少，但在大规模部署时仍需额外调优。

---

## 89. Shape optimization of pneumatic soft actuators

**arXiv ID:** 2606.30800 | [PDF](https://arxiv.org/pdf/2606.30800v1)

**作者:** Anna Dalklint `[一作]` (Harvard University), Katia Bertoldi `[通讯]` (Harvard University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了三维软致动器的形状优化框架，实现定制化变形行为

**💡 创新点**

通过将形状变量映射为有限元节点坐标，利用梯度信息和混合有限元处理几何与材料非线性，提出统一高效的优化流程

**🔧 技术方法**

使用混合有限元、PETSc并行求解、MMA优化算法，以及仿真与实验验证技术

**📊 数据集**

未使用公开数据集，实验采用自制的PVS软致动器进行验证

**📈 对比分析**

通过实验与数值仿真对比，压力-体积曲线和位移匹配良好，证明了抓取、伸缩等多模态行为的有效实现

**⚠️ 局限性**

受限于制造误差、未建模接触影响以及局部自交问题

---

## 90. When Does Learning to Stop Help? A Cost-Aware Study of Early Exits in Reasoning Models

**arXiv ID:** 2606.30852 | [PDF](https://arxiv.org/pdf/2606.30852v1)

**作者:** Zhe Dong `[一作]` (University of Maine at Presque Isle), Manish Shah `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LearnStop，一种只利用前缀可观测特征的学习型停止器，能在推理时动态决定是否终止生成；

**💡 创新点**

创新点在于将隐藏状态完全剔除，使用多维前缀特征（置信度、熵、投票占比、答案稳定性等）训练轻量级分类器，并通过风险校准实现可控的停止；

**🔧 技术方法**

采用逻辑回归（以及可选的梯度提升/MLP）对前缀特征进行分类，并利用阈值控制停止；

**📊 数据集**

评估数据集包括GSM8K、MATH‑500、MMLU‑Pro、GPQA‑Diamond、AIME‑90，并在多模型（Qwen3‑8B/32B、DeepSeek‑R1‑Distill等）上进行实验；

**📈 对比分析**

与传统的单一信号停止（置信度、熵、答案稳定性、confidence‑leap等）进行配对bootstrap比较，结果显示在自由形式数学任务上LearnStop可获得最高峰值增益（如GSM8K‑Qwen3‑32B提升+0.157），但在多选或极难任务上收益有限或无显著提升；

**⚠️ 局限性**

局限性包括：停止策略非普适，需交互式推理且前缀重用才能节省计算；probe开销在纯API场景下可能抵消收益；小规模或极难数据集的结果不稳健；并且在跨模型/跨任务迁移时表现不一。

---

## 91. Using AI Agents to Automate Black-Box Audits of Personalization Algorithms at Scale

**arXiv ID:** 2606.30801 | [PDF](https://arxiv.org/pdf/2606.30801v1)

**作者:** Alessandro Morosini `[一作]` (Massachusetts Institute of Technology), Chara Podimata `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过 GPT‑4o 驱动的固定 persona 代理在推特上进行大规模黑盒审计，测量算法对不同内容类型（有毒、极化、政治、右倾等）和用户属性的放大效果。

**💡 创新点**

创新点在于将行为与可见属性分离：使用预设的 persona Prompt 让 LLM 生成固定行为策略，随后在保持行为不变的前提下随机扰动平台可见信号，从而实现可控的反事实实验并揭示算法对不同人群的异质化响应。

**🔧 技术方法**

技术包括 GPT‑4o LLM 代理、Persona Prompt 设计、Selenium 浏览器自动化、内容分类器（toxicity、政治倾向、极化、作者影响力）以及基于回归和聚类 Bootstrap 的统计估计。

**📊 数据集**

数据来源于美国人口普查与 Pew 政治类型调查，用以构造 14 种 persona；在推特上收集 20 万+ 条曝光记录并进行分类。

**📈 对比分析**

与传统人类受试者和脚本化 sock‑puppet 实验对比，LLM 代理在保持行为一致的同时实现更大规模、更可控的实验；实验结果显示算法显著放大有毒、极化、政治及右倾内容，且不同政治倾向用户的放大幅度差异显著，放大比例与传统方法相比更细致、可解释。

**⚠️ 局限性**

局限包括 LLM 代理可能缺乏真实人类经验导致行为偏差、实验仅在选举后短期新建账户窗口内进行、结果受平台策略和时间点的依赖、异质效应检验为探索性且样本量有限。

---

## 92. Debugging as Evidence-Driven Reasoning: Visualization Opportunities in Data-Intensive Programming

**arXiv ID:** 2606.30884 | [PDF](https://arxiv.org/pdf/2606.30884v1)

**作者:** Yongbo Chen `[一作]` (Tulane University), Rebecca Faust `[通讯]` (Tulane University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对九名数据密集型从业者的半结构化访谈，提炼出调试过程中的三大共性挑战，并基于此提出了三项可视化支持需求：跨工件证据对齐、基于期望的比较与可追踪的状态演进。

**💡 创新点**

创新点在于将调试视作“证据驱动的推理”过程，首次系统识别并归纳了跨工具、多阶段数据流水线中的调试痛点，并从可视化角度给出具体的设计空间和需求，填补了传统调试研究对数据密集型场景关注不足的空白。

**🔧 技术方法**

主要技术手段是定性研究：半结构化访谈、主题编码、主题聚类和跨主题发现，辅以已有可视化理论（如多视图关联、视觉融合、流程图）为设计思路提供框架。

**📊 数据集**

未使用公开数据集；研究基于受访者提供的访谈文本与自述数据，构成自有访谈资料。

**📈 对比分析**

本研究未进行实验比较或性能评估；提出的可视化需求是理论性设计建议，未来需在原型实现后通过用户研究或案例研究进行评估。

**⚠️ 局限性**

局限性包括样本量小、领域代表性有限、仅基于回忆式自述可能存在偏差，且所提出需求尚未在真实系统中验证。

---

## 93. How Human Feedback Shapes AI-generated Community Notes

**arXiv ID:** 2606.30905 | [PDF](https://arxiv.org/pdf/2606.30905v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 94. Emergent Culture in Minimal LLM Systems

**arXiv ID:** 2606.30668 | [PDF](https://arxiv.org/pdf/2606.30668v1)

**作者:** Simon Jones `[一作]` (University of Bristol), Sabine Hauert `[通讯]` (University of Bristol)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在无上下文、最小提示、共享衰减文本存储环境下的无状态LLM代理集群，观察其自组织、协作与文化生成。

**💡 创新点**

通过最小化设计与退化存储的底层自组织机制，证明无状态LLM可在退化环境中自发形成持久的语义结构和“文化”，并提供多层次动力学与语义一致性分析。

**🔧 技术方法**

采用多模型（Claude、Gemini、Kimi）代理，使用LiteLLM API、共享键值存储、工具调用、基于RQA的动态系统分析与嵌入语义向量。

**📊 数据集**

生成数据来自十次10循环实验和五次100循环实验的事件日志，文本量相当于小说级别，未使用公开语料库。

**📈 对比分析**

通过RQA、语义嵌入、词汇持久性等方法与人类文学与低质量稿件比较，发现生成文本在DET/ENTR上高于人类作品且显示跨熵界限的长期结构。

**⚠️ 局限性**

局限在于仅测试少量模型与实验组合、未系统调节衰减速率、未探索更长时间或更多代理规模，且对生成内容的具体质量评估不足。

---

## 95. PolyFlow: Continuous Topology Embedding Flow Matching for Artist-style Mesh Generation

**arXiv ID:** 2606.30673 | [PDF](https://arxiv.org/pdf/2606.30673v1)

**作者:** Chunshi Wang `[一作]` (Zhejiang University), Yawei Luo `[通讯]` (Zhejiang University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出PolyFlow，利用连续拓扑嵌入和流匹配并行生成具有艺术家级拓扑的网格；

**💡 创新点**

通过学习可逆的连续拓扑嵌入，将离散网格连通性映射为可连续处理的空间-时间坐标，从而实现全并行流匹配生成；

**🔧 技术方法**

Transformer流匹配网络、空间-时间距离嵌入、ODE求解器、点云条件编码器；

**📊 数据集**

在约5千万张网格训练，评测使用Toys4K数据集；

**📈 对比分析**

与BPT、MeshAnythingV2、FastMesh、DeepMesh等自回归与其他方法对比，PolyFlow在Chamfer和Hausdorff距离上分别取得0.008/0.021，速度提升至几秒且可精确控制顶点数；

**⚠️ 局限性**

对极大尺寸网格的内存占用仍有限，且拓扑嵌入维度需手动调节，未解决非三角形拓扑或自相交问题。

---

## 96. A Systematic Approach to Multi-Agent AI from Advanced Regulatory Control Theory: Safe and Auditable LLM Operator Agents for Process Control

**arXiv ID:** 2606.30877 | [PDF](https://arxiv.org/pdf/2606.30877v1)

**作者:** Idelfonso B. R. Nogueira `[一作]` (Norwegian University of Science and Technology), Sigurd Skogestad `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出将高级调节控制（ARC）链映射为多代理系统，实现安全可审计的LLM操作员代理；在奶牛棚通风案例上实现并比较不同架构

**💡 创新点**

1) 将传统控制层的 MIN/MAX 选择器、分割范围等结构正式视为多代理体系的通信与优先级机制；2) 通过 prompt engineering 使单一 LLM 专门化为约束守护者，而不需要 RAG；3) 在代理层引入 LLM 监督器，实现慢速参数调度；4) 在单 LLM 对比中验证分解的重要性

**🔧 技术方法**

高级调节控制理论、PI 控制（SIMC 调参）、MIN/MAX 选择网络、分割范围、反向积分 anti‑windup、结构化 prompt engineering；使用 Qwen 2.5 7B Instruct 作为操作员代理，Claude Opus 4.7 作为监督器，离线 GPU 推理

**📊 数据集**

仿真奶牛棚通风模型（双状态 C、T，双操纵变量 fan、heater），4 天混合季节扰动序列（室外温度、牛数量、CO₂ 溢出、泄漏）以及原始 Skogestad Case IIB 的温度坡度实验；使用相同噪声种子与扰动

**📈 对比分析**

对比 ARC PI（基线）、LLM 代理+规则 orchestrator、LLM 代理+LLM orchestrator、单 LLM 控制器；指标包括温度/CO₂ 违反时间、能耗、切换次数、模式翻转；结果显示 LLM 代理能复现 ARC 行为并提供审计日志；LLM orchestrator 可降低切换频率并略高能耗；单 LLM 失去优先级约束导致温度严重偏低

**⚠️ 局限性**

1) 仍依赖离线推理，延迟与实时性能未知；2) 模型对提示敏感，可能出现失配；3) 代理间仅用离散模式，导致“闹钟”频繁；4) 未实现连续输出或边界 hysteresis；5) 只在模拟验证，实际工业部署尚未测试；6) 需要手动设计 prompt 与上下文，规模化困难

---

## 97. Mapping the Artificial Intelligence Divide in Africa: Infrastructure, Accessibility and Capacity

**arXiv ID:** 2606.30656 | [PDF](https://arxiv.org/pdf/2606.30656v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 98. Vision-Language Procedural Reasoning for Context-Aware Reward Modeling of Robotic Endovascular Guidewire Navigation

**arXiv ID:** 2606.30698 | [PDF](https://arxiv.org/pdf/2606.30698v1)

**作者:** Wentong Tian `[一作]` (Tongji University), Peng Qi `[通讯]` (Tongji University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种基于多模态大语言模型的程序推理框架（VL‑PR），用于在机器人血管内介入中实现上下文感知的奖励适配，从而让单一策略能够在不同导航阶段自动平衡效率、安全与精度，实现自主引导线导航。

**💡 创新点**

创新点在于将多模态大语言模型作为高层程序推理模块，将实时视觉语言观测转换为导航上下文，动态调整奖励权重，解决传统固定奖励或多策略方法无法处理的阶段性目标权衡问题。

**🔧 技术方法**

使用的技术包括：多模态大语言模型（Qwen2.5‑VL‑3B‑Instruct + QLoRA微调）、Soft Actor‑Critic（SAC）强化学习、上下文感知奖励矩阵、视觉感知（YOLOv5 + U‑Net）以及仿真–实物闭环平台。

**📊 数据集**

数据集：在仿真环境中生成的局部血管观测、引导线姿态、血管几何信息与专家标注的程序状态标签（共计 300/170/130/100 条实例），以及物理平台上使用的人工血管模型和相应的传感数据。

**📈 对比分析**

通过与 PPO、DDPG、TD3、SAC 等标准连续控制 RL 基线以及人工操作对比，VL‑PR 在冠状、颈动脉、肾动脉三种解剖场景中实现 100%/97%/100% 的成功率，并将平均步骤数比标准 SAC 降低约 48%–60%，显示显著性能提升。

**⚠️ 局限性**

局限性包括：仅采用离散阶段分类，缺乏更细粒度或结构化的程序描述；仿真到真实的泛化仍受限于血管形变与视觉噪声；安全约束主要通过奖励权重实现，缺乏形式化的安全约束与保证。

---

## 99. AVTok: 1D Unified Tokenization for Holistic Audio-Video Generation

**arXiv ID:** 2606.30811 | [PDF](https://arxiv.org/pdf/2606.30811v1)

**作者:** Kien T. Pham `[一作]` (Hong Kong University of Science and Technology), Long Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一的音视频tokenizer（AVTok），能够将音频与视频对合并编码为单一一维离散潜在表示；

**💡 创新点**

创新点在于：1）双流Transformer架构，使用共享编码解码器与模态专属可学习查询实现模态特征融合；2）层级VFAL训练策略和表示对齐损失，逐步提升视频/音频重建与跨模态对齐；3）结合自回归先验，使token空间适用于下游AR生成；

**🔧 技术方法**

使用技术包括：1）可学习查询的1D视频tokenization；2）对音频的mel‑spectrogram patch化；3）共享编码解码器、模态专属层归一化；4）对齐损失与自回归先验；5）HiFi‑GAN vocoder、Llama‑style AR生成器；

**📊 数据集**

使用数据集：TAVGBench和VGGSound，测试集采用VGGSound的16帧128×128视频、22kHz音频；

**📈 对比分析**

与单模态基线（OmniTokenizer、AdapTok、LARP、WavTokenizer、UniCodec、SpectralCodec）以及下游生成基线（TempoTokens、MMAudio、VinTAGe、V‑AURA、SpecVQGAN、JavisDiT、Ovi）进行比较。AVTok在视频重建（PSNR↑、FVD↓、LPIPS↓）和音频重建（SI‑SDR↑、FAD↓、MR‑STFT↓）均优于或接近基线，并在音视频下游生成任务（A2V、V2A、cJAVG）的FVD/FAD/DeSync/IB‑Score等指标上取得更好或相近的表现，且模型参数更少、速度更快。

**⚠️ 局限性**

局限性：1）仍依赖预训练的音视频基础模型，迁移性需进一步验证；2）实验仅在中等分辨率（128×128）和固定帧率下进行，缺乏对高分辨率或实时应用的评估；3）仅覆盖音视频两模态，未考虑更丰富的多模态输入（文本、姿态等）。

---

## 100. Hierarchical Global Attention (HGA)

**arXiv ID:** 2606.30709 | [PDF](https://arxiv.org/pdf/2606.30709v1)

**作者:** Woernle Frank `[一作]` (BMW Group), Grinenko Artemiy `[通讯]` (BMW Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Hierarchical Global Attention（HGA），一种可直接替换预训练长上下文 Transformer 的稀疏注意力机制，使如 Qwen3‑30B 在 RTX 5090 上可无训练、无校准地使用 32K–64K token 上下文。

**💡 创新点**

创新点在于两层层次化路由：先用 RoPE 兼容的块级摘要粗筛，再按组细筛，最终仅对真正的 token‑level K/V 进行精确 softmax，保持原有权重、无额外参数、无重训练，显著降低 VRAM 需求。

**🔧 技术方法**

技术包括 RoPE、混合 RoPE 归一化摘要、分层路由策略、RAM‑backed KV 存储（hot/warm/cold）、Triton fuse、FP8 量化模型以及分块、分组的精确 token attention。

**📊 数据集**

主要使用 Qwen3‑30B‑A3B‑Instruct‑2507‑FP8、40M SmallLM（FineWeb 1:10 细分集）以及 8192‑token 的 FineWeb 文档集；也在 Needle‑in‑a‑Haystack 任务上评测。

**📈 对比分析**

与密集注意力对比，3%–12.5% 稀疏度下，4K–64K token 的 loss 差距 ≤ 0.02 nats；Qwen3‑30B 在 32K token 上无调优即可直接跑，训练吞吐率提升 2.72×；40M SmallLM 仅 0.018 nats copy‑only gap；指针检索任务 100% 成功。

**⚠️ 局限性**

局限在于：路由可能遗漏远程相关 token、对长范围位置编码仍有残余误差、极长上下文（>64K）需额外索引/分段、实现仍采用固定路由预算、未在所有下游任务全面验证。

---

## 101. Gradient Smoothing: Coupling Layer-wise Updates for Improved Optimization

**arXiv ID:** 2606.30813 | [PDF](https://arxiv.org/pdf/2606.30813v1)

**作者:** Haoming Meng `[一作]` (University of Toronto), Vardan Papyan `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种深度维度梯度增益（Depth‑wise Gradient Augmentation）框架，并在此框架下实现了一种名为梯度平滑（Gradient Smoothing）的技术，用于在训练时通过邻层梯度的局部平均来平滑每层的优化更新。

**💡 创新点**

创新点在于：①首次把层间梯度结构视为可利用的优化信号，将更新向量在深度方向上进行预处理；②提出了梯度平滑作为一种无参数、无显式模型改动的深度级预处理器；③展示了这种方法可与任意一阶优化器（SGD、Adam、Muon等）兼容，且对不同任务均能提升收敛速度与泛化性能。

**🔧 技术方法**

技术细节包括：构建块级梯度向量 g(θ)= (g1,…,gL)，使用局部窗口平均算子 S（窗口大小为3）对每层梯度进行线性平滑，得到 ˜g = S⊗I · g；可选的归一化方案包括保幅平滑（Norm）与方向平滑（Dir）；实现时仅在每步优化器更新后对 g 进行一次矩阵乘法，计算开销极低。

**📊 数据集**

实验数据集涵盖：①大规模语言模型预训练（GPT‑style，depth 24/30，≈1–2.5 B 参数）；②RL微调任务（Open‑RS 计算数学推理数据集 AIME24/25、AMC23、MATH‑500）；③视觉分类（ViT‑B 在 CIFAR‑100 上训练 1700 epoch）；④扩散模型（U‑ViT 在 CIFAR‑10 上训练 500k 步），并使用标准的 FID、验证损失、CORE 指标等评估。

**📈 对比分析**

与基线（同一优化器、相同超参）对比，梯度平滑在所有实验中均表现出更快的收敛速度、验证损失下降幅度更大、最终准确率/CORE/ FID 等指标均提升：如 ViT‑B 在 CIFAR‑100 上精度从 74.56% 提升至 75.62%（α=0.2），LLM 预训练验证损失在 30‑层模型上提前数百步收敛，RL 微调的算术推理平均准确率从 55.45% 提升至 57.60%。

**⚠️ 局限性**

局限性包括：①最佳平滑强度 α 需要经验调优，过大可能削弱层间特异性信息；②目前仅测试了局部窗口平均，未探索更复杂或自适应的深度平滑算子；③在某些极端深度或特殊结构（如非残差跳连）下效果可能不如预期；④仍未对超参与梯度平滑相互作用做系统化分析。

---

## 102. BayesBench: Evaluating LLM Belief Trajectories Under Multi-Turn Evidence Accumulation

**arXiv ID:** 2606.30850 | [PDF](https://arxiv.org/pdf/2606.30850v1)

**作者:** Ankur Samanta `[一作]` (Meta AI), Yonathan Efroni `[通讯]` (Tel Aviv University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 BayesBench benchmark，系统地评估多轮对话中 LLM 的贝叶斯推理与隐藏量更新，并在四种环境（投币、推荐、社交判断、医疗分诊）中进行实验。

**💡 创新点**

创新点在于：①把多轮贝叶斯更新过程可视化为逐步的信念轨迹；②引入“latent‑framed”任务，考察模型对隐藏结构和表达方式双重推理；③比较主动对话与被动观测的差异，揭示模型在不同交互形式下的偏差。

**🔧 技术方法**

技术手段包括：①多轮 MCQ 轮询抽取模型的后验分布；②使用总变差距离（TVD）与平均绝对误差（MAE）评估与理论贝叶斯后验的匹配；③构建用户模拟器、推荐系统模拟和医学症状分段的多轮推理管线。

**📊 数据集**

使用的数据集有：β–Bernoulli 投币实验（自定义）；MovieLens 电影评分集合（构造四类用户类型）；r/AmItheAsshole Reddit 贴文（二分类社区判决）；医学分诊数据集（10 专科，4 急诊标签）。

**📈 对比分析**

对比方法：在七个开源 LLM（LLaMA 3B‑70B、Qwen 3B‑32B）上执行全程信念抽样与预测；与理论后验、已知用户类型和标签的标准比较。结果显示：①更大模型在信念更新上更准确；②但下游预测仍存在显著偏差，往往过度自信；③主动对话会引入 pro‑user 偏差，且仅在部分模型上通过显式条件化能略微提升预测准确率。

**⚠️ 局限性**

局限性：①评估依赖 MCQ 轮询，可能不完全反映模型内部连续信念；②隐含的离散隐藏结构过于简化，真实场景中隐藏因素往往连续或混合；③部分环境缺乏闭式贝叶斯后验参考，评估更主观；④未考虑模型与人类用户真实动态交互导致的自适应变化；⑤只测试了公开权重模型，未覆盖闭源前沿模型。

---

## 103. Revocable Learned State via Process Sidecars

**arXiv ID:** 2606.30788 | [PDF](https://arxiv.org/pdf/2606.30788v1)

**作者:** John Sweeney `[一作]` `[通讯]` (Sideplane AI), John Sweeney (Sideplane AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了过程侧车（process sidecar）编辑方法，用以在安全训练后撤销模型记忆并保持拒绝行为

**💡 创新点**

创新在于证明记忆方向被安全训练迁移，提出二维编辑族并证明其在一阶不完全性下恢复对数安全性的二阶精度

**🔧 技术方法**

利用AdamW训练的前向敏感性 Jacobian-Vector Product 近似（中心 secant），并构建两系数编辑公式

**📊 数据集**

使用高熵私有事实（canary）数据集进行记忆训练和安全训练，验证集包含31个干扰词的秘密辨别任务以及实体相关的拒绝提示

**📈 对比分析**

与单一减法、过程-JVP线性和FT-unlearning等基线比较，在 Qwen-2.5、Llama-3.2、Qwen3.5、Qwen3-8B 四个规模上，过程侧车在拒绝闭合上均显著优于基线（如 60/60 显著性），且不牺牲忘记性能

**⚠️ 局限性**

局限在于需固定安全训练轨迹、仅在 LoRA/全参数小规模实验中验证、使用 ε=1 近似可能影响更大尺度、未处理多轮安全训练或对抗提取等情景

---

## 104. Budget-Adaptive Routing: Skipping the Weak When the Strong Answers Anyway

**arXiv ID:** 2606.30919 | [PDF](https://arxiv.org/pdf/2606.30919v1)

**作者:** Wei Geng `[一作]` (Technical University of Munich), Jörg Ott `[通讯]` (Technical University of Munich)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了边缘-云端视觉推理中路由器的放置位置，提出了一个能在不同 offload 预算下自适应选择估计器位置的预算感知路由器。

**💡 创新点**

创新点在于：①首次证明仅使用原始像素的轻量估计器即可匹配甚至超越基于弱模型特征的估计器；②设计了两种估计器（图像级估计器和弱模型后估计器）并通过离线阈值切换实现对不同预算下最佳放置位置的自适应选择；③提出了基于上下文的 ΔAP 抽样奖励和 MORIC+ / OffloadBin 目标，提升估计器训练效果。

**🔧 技术方法**

采用 MobileNetV2-Lite 轻量网络作为图像级估计器，XGBoost 作为基于弱模型特征的估计器；使用阈值阈分法（π_ρ）实现预算约束；在 PASCAL VOC 上做了端到端 mAP@0.5、GFLOPs 和延迟评估。

**📊 数据集**

数据集：PASCAL VOC 2012，采用标准训练/验证/测试拆分，并使用 VOC 按 IoU 0.5 的 mAP 作为性能指标。

**📈 对比分析**

与 EdgeML、DCSB 等现有基线（均在弱模型后估计）对比。结果显示：- 图像级估计器在路由质量上优于现有方法（Spearman ρ≈0.557，AUC_ρ≈0.795）；- 自适应路由器在 ρ∈[0.1,0.9] 范围内达到最高 mAP=0.808，超过强模型 0.791（+1.7pp）；- 在 ρ 0.3–0.7 的“跳过”区间内，计算量降低约 2.1 GFLOPs/帧，延迟下降约 14.2 ms（≈26%）。

**⚠️ 局限性**

局限性：实验基于离线仿真，未在真实 edge‑cloud 测试床上验证网络延迟和抖动；仅评估了 PASCAL VOC，未验证在更大更复杂数据集（如 COCO）的泛化能力；估计器的阈值调优依赖离线切分，实际部署中需实时校准。

---

## 105. Anthropomorphism in AI Companion Communities: Age, Gender, and Emotional Correlates

**arXiv ID:** 2606.30942 | [PDF](https://arxiv.org/pdf/2606.30942v1)

**作者:** Afia Mubashir `[一作]` (Independent), Rose E. Guingrich `[通讯]` (Ethicom)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用Reddit三大AI伴侣子论坛的评论数据，构建AnthroIndex评估用户对AI伴侣的拟人化程度，并关联年龄、性别与情绪表达。

**💡 创新点**

① 开发AnthroIndex——针对非正式社交媒体文本的LMM‑based拟人化评分工具；② 通过堆叠学习推断用户年龄与性别，实现高精度人口学标签；③ 系统性分析年龄、性别与情绪如何共同调节拟人化。

**🔧 技术方法**

自然语言处理技术：LLM（GPT‑4.1‑nano）评分、预训练情感分类器、n‑gram特征与堆叠式机器学习（XGBoost、LightGBM、Random Forest、Logistic Regression）。

**📊 数据集**

约28.4万条Reddit评论，涵盖r/CharacterAI、r/Replika和r/AICompanions三大子版块，用户约4.7万名，包含自我声明的年龄与性别信息。

**📈 对比分析**

对比传统关键词/MLM方法，AnthroIndex在人类标注上达Pearson r = 0.59、Spearman ρ = 0.49，90%+的1‑点误差；在拟人化预测任务中，年龄与性别可解释约5%方差，情绪补充后增至6.4%，显示模型稳健但提升有限。

**⚠️ 局限性**

局限性：样本仅来自Reddit，缺乏随机代表性；跨平台偏好与社区文化可能混淆拟人化；采用二元年龄与性别划分，忽略多样身份；基于文本的拟人化评分仅反映语言表现，未直接测量用户内在信念；使用LLM存在潜在算法偏差与隐私风险。

---

## 106. Investigating Multi-Agent Deliberation in Law

**arXiv ID:** 2606.30906 | [PDF](https://arxiv.org/pdf/2606.30906v1)

**作者:** Cor Steging `[一作]` (University of Groningen), Tadeusz Zbiegień `[通讯]` (Jagiellonian University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文探索多代理推理在法律推理任务中的应用，并评估标准MAD、3-Ply、Parrots三种多代理框架与单一LLM基线的差异。

**💡 创新点**

创新点在于设计两种受法院程序与论证理论启发的新多代理框架，并展示其在法律任务中能产生与基线不同且更准确答案的潜力。

**🔧 技术方法**

技术手段主要是基于大型语言模型的多代理推理、回合式反思以及三层庭审角色和批判性“鹦鹉”代理的对话架构。

**📊 数据集**

实验使用了四个法律推理基准（LEXAM、COLIEE、SARA、PRIVACY）以及一个逻辑推理基准LOGIQA，共计1250条随机抽样问题。

**📈 对比分析**

通过对F1得分的比较发现，整体表现与基线相近，但多代理模型在约10%问题上给出不同答案，并在部分数据集上略优；此外，多代理模型能解决基线无法正确回答的案例。

**⚠️ 局限性**

局限性包括：需要更多模型调用导致成本上升，模型仍易出现幻觉与逻辑不严谨，缺乏真实性验证，且受数据污染与模型不确定性影响结果。

---

## 107. Towards Critical IR Theories and Practices

**arXiv ID:** 2606.30984 | [PDF](https://arxiv.org/pdf/2606.30984v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 108. Neuro-Bayesian-Symbolic Residual Attention Shallow Network: Explainable Deep Learning for Cybersecurity Risk Assessment

**arXiv ID:** 2606.30953 | [PDF](https://arxiv.org/pdf/2606.30953v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 109. Automated Reasoning with Nested Datatypes

**arXiv ID:** 2606.30888 | [PDF](https://arxiv.org/pdf/2606.30888v1)

**作者:** Tomer Hakak `[一作]` (Bar-Ilan University), Cesare Tinelli `[通讯]` (University of Iowa)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并研究了“嵌套数据类型”（Nested Datatypes, NDT）理论，并给出了基于翻译与标准组合（数组、数据类型、无解释函数）实现的决策程序，随后在 SMT‑solver 中实现并对实验数据进行评估。

**💡 创新点**

创新点：
1) 定义了兼容数组与数据类型的 NDT 理论，解决了 Naïve 组合产生的非标准模型（环）问题；
2) 通过构造 NDT‑relation 并要求其良基化，严格限制了可接受的模型；
3) 提出一种基于签名重写、引入辅助无解释函数的翻译技术，使得原始 NDT 公式可转化为标准组合理论下可判定的公式；
4) 在此基础上给出了完整的判定证明与实现方案。

**🔧 技术方法**

技术手段：
- 结构化签名扩展：为每个数组签名引入三种新排序（数组自身、对应数据类型、辅助标识）；
- NDT‑relation 的定义与良基性检查；
- 翻译函数（term/formula 转换）与补充子句（Lemma）来维护数组与数据类型之间的一致性；
- 结合 Nelson‑Oppen 组合方法，利用已成熟的数组、数据类型和 EUF 理论实现判定；
- 在 SMT‑solver（CVC5）中实现预处理插件。

**📊 数据集**

数据集：
1) 合成循环基准：系统生成形如 k 的公式，编码长度可调的环；
2) Move Prover 验证基准：从 Move Prover 的 15 个原始测试中提取的、去除量词、常量数组、Boogie 标签等不支持特性的 76 个无增量、无量词的 SMT‑LIB 文件。

**📈 对比分析**

对比与性能：
- 与现有的嵌套数据类型实现（如某 SMT‑solver 的内置实现）进行比较；
- 在所有 500 条不可满足的合成基准上，本文实现能全部求解，且在大多数实例中快于对手；
- 在可满足的合成基准中，本文仅能求解一半（对手一条），体现了此类问题的难度；
- 对于 Move Prover 基准，本文与对手均能快速求解所有实例，本文在可满足实例上略快；
- 统计显示：在 10 秒时间限制下，本文在合成基准上平均耗时 2–3 秒，Move Prover 基准平均不到 1 秒。

**⚠️ 局限性**

局限与待改进：
- 目前只支持无量词、无增量求解，未实现对量词、递归增量上下文的支持；
- 对常量数组、Boogie 标签等特性做了剔除，影响了原始验证任务的完整性；
- 仍假设数组索引不使用嵌套数据类型（在实现中已放宽但实现细节较复杂）；
- 对数组索引为嵌套数据类型的情形仅做了简化处理，未涵盖所有可能的嵌套结构；
- 只针对数组与数据类型的组合，未扩展到序列、集合等其他数据结构，需进一步推广。

---

## 110. GRAPE: Graph-Augmented Prototype Explanations for Interactive Medical Image Diagnosis

**arXiv ID:** 2606.30901 | [PDF](https://arxiv.org/pdf/2606.30901v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 111. The Quadruped Soft Tail: Compliant Grasping and Swabbing for Contamination Surveys in Harsh Environments

**arXiv ID:** 2606.30900 | [PDF](https://arxiv.org/pdf/2606.30900v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 112. Ricci-Notation Tensor Framework for Numerical Algebraic Geometry via Any-Degree Unitary-Triangular Factorization

**arXiv ID:** 2606.31003 | [PDF](https://arxiv.org/pdf/2606.31003v1)

**作者:** Dileepan Joseph `[一作]` `[通讯]` (University of Alberta), Dileepan Joseph (University of Alberta)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb`

**🎯 论文内容**

开发了一种适用于多项式系统的任意次数单元-三角分解（Qr）并实现了对应的张量框架和 MATLAB 工具箱。

**💡 创新点**

在传统 QR 的基础上推广到高次数多项式，用 Ricci 记号张量形式提出了保证零集不变的 Qr 分解，并提供了稀疏张量软件 RTToolbox。

**🔧 技术方法**

采用了 Ricci 记号张量代数、稀疏多维数组（MDA）实现、线性代数中的 RRLU、CPD 等技术，并在 MATLAB 中使用 MEX C 代码加速。

**📊 数据集**

以两个教学示例：盒子尺寸（3 变量 3 方程）和两段机械臂（4 变量 4 方程）作为测试数据集。

**📈 对比分析**

与 Gröbner 基方法比较，在盒子尺寸问题上 Qr 直接无迭代得到三角化，运行时间与 GB 相当；在机械臂问题中需要迭代，性能略慢，但可通过手工构造更稀疏的 Qr 解决。

**⚠️ 局限性**

主要限制在优化迭代部分收敛困难、需要手工设定增量阶 ΔD、对复杂高维多项式收敛性和数值稳定性尚未完全证明。

---

## 113. Sampling-Based Coordination-Informed Multi-Objective Multi-Robot Reinforcement Learning

**arXiv ID:** 2606.30893 | [PDF](https://arxiv.org/pdf/2606.30893v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 114. Towards Transparent Checkpointing with AI-driven Code Generation

**arXiv ID:** 2606.30921 | [PDF](https://arxiv.org/pdf/2606.30921v1)

**作者:** Hai Duc Nguyen `[一作]` (Argonne National Laboratory), Bogdan Nicolae `[通讯]` (Argonne National Laboratory)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了使用前沿大型语言模型（Claude Opus 4.7）在无人工干预的情况下，自动为六个MPI科学应用生成可靠的检查点/重启代码。

**💡 创新点**

创新在于提出完整的闭环生成‑验证‑修复流水线，并证明LLM能够识别关键状态、定位安全检查点并生成低开销、可恢复的代码。

**🔧 技术方法**

采用Claude Opus 4.7作为编码代理，通过OpenCode CLI与代码库交互；使用VeloC运行时实现多级检查点；构建独立验证器进行构建、失败注入与输出比对。

**📊 数据集**

使用六个公开MPI应用的基准套件（Athena++、CoMD、HPCG、LAMMPS、OpenLB、SPARTA），覆盖不同规模、数据结构与检查点模式。

**📈 对比分析**

对比生成代码与人工实现的检查点方案，测量失败无故障运行的时间、检查点大小、重启后恢复时间；结果显示大多数应用零开销、恢复性能仅比人工实现慢3–6%，并比无检查点版快约三倍。

**⚠️ 局限性**

受限于LLM推理成本与令牌消耗，循环迭代时长最长92分钟；难以处理更大规模、多节点、频繁失效的场景；生成代码可审计性与维护成本仍需提升。

---

## 115. Beyond Clean Text: Evaluating Encoder and Decoder Robustness for Bangla Event Detection in Noisy Text

**arXiv ID:** 2606.30914 | [PDF](https://arxiv.org/pdf/2606.30914v1)

**作者:** Tanvir Ahmed Sijan `[一作]` (Jahangirnagar University), Md. Musfique Anwar `[通讯]` (Jahangirnagar University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了一个泛化的孟加拉语新闻事件本体，并整理了包含 9,979 条句子的基准数据集（清晰新闻、ASR 转录及仿真拼写噪声），随后系统比较了编码器模型（BanglaBERT、XLM‑R）与解码器 LLM（Llama、Gemma）的事件检测性能。

**💡 创新点**

创新点在于首次提出孟加拉语新闻事件本体和对应噪声基准，揭示了编码器与解码器架构在噪声鲁棒性上的权衡，并通过指令微调、注释指南嵌入以及混合训练探索提升鲁棒性的策略。

**🔧 技术方法**

方法采用了预训练编码器微调、指令微调的解码器 LLM（使用代码式输出、Python docstring 形式的注释指南）、负样本采样和基于 QWERTY 键盘模型的拼写噪声注入。

**📊 数据集**

使用了 Bangla News Article Dataset（BNAD）中的清晰新闻、三大 Bangla 电视台（Jamuna TV、Somoy TV、ITN）的 ASR 转录，并在 Clean 集上生成不同程度的拼写噪声，最终形成 9,979 条带 7,813 条事件标注的句子。

**📈 对比分析**

评估方法以宏观 F1 为主，在 Clean、ASR 和多级拼写噪声（WCR 10–40%）测试集上对比。结果显示，编码器模型在 Clean 上性能最高，但噪声下跌幅大；解码器 LLM 在噪声下更稳健；指令微调提升准确率但对鲁棒性的改善不一；混合训练显著减轻性能衰减，模型规模增大对解码器 LLM 的鲁棒性提升更显著。

**⚠️ 局限性**

局限性包括严格的跨度匹配评估方式对生成式 LLM 的误判较多，数据集仅覆盖新闻领域且仅涉及事件触发检测，未扩展至论点抽取等完整事件抽取任务，且在更广泛领域和任务中的泛化性尚未验证。

---

## 116. When Regulation Has Memory: Hysteresis and Control Burden in Artificial Agency

**arXiv ID:** 2606.30975 | [PDF](https://arxiv.org/pdf/2606.30975v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 117. Curvature-Guided Module Localization for Low-Rank Detoxification of Backdoored Large Language Models

**arXiv ID:** 2606.30899 | [PDF](https://arxiv.org/pdf/2606.30899v1)

**作者:** Arash Raftari `[一作]` (AIVault Inc.), Andrew Arash Mahyari `[通讯]` (AIVault Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对被后门攻击的LLM进行后置去毒，局部修复触发传播模块。

**💡 创新点**

结合激活补丁与Fisher/K‑FAC曲率分析定位触发路径，再用低秩LoRA仅在选定模块进行修复。

**🔧 技术方法**

机制化模块定位、激活补丁、K‑FAC曲率估计、低秩LoRA补丁、教师‑学生对齐训练。

**📊 数据集**

使用mental‑health counseling数据集，加入控制触发词，构造对齐的干净/触发样本。

**📈 对比分析**

与CROW一致性正则化基线在相同模块预算下比较；在中、末端触发位置，样本/模块足够时TMRR降至0，TCBS保持高。

**⚠️ 局限性**

实验仅覆盖单一基模型、单触发、已知触发位置的受控设置，未测试多触发、不同模型或未知触发情况。

---

## 118. Learning Where to Look: A Reinforcement Learning Framework for Robust Micro-Ultrasound Prostate Cancer Detection

**arXiv ID:** 2606.30951 | [PDF](https://arxiv.org/pdf/2606.30951v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 119. Multistage Defer Trees for Hybrid Interpretability: If at First You Can't Succeed, Tree Again

**arXiv ID:** 2606.30995 | [PDF](https://arxiv.org/pdf/2606.30995v1)

**作者:** Zakk Heile `[一作]` (Duke University), Cynthia Rudin `[通讯]` (Duke University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种多阶段推迟决策树（Multistage Defer Trees，MDT），在每一阶段用稀疏决策树做预测或推迟到下一阶段，最终仅在少数样本上才委托给黑盒模型，从而在保持高准确率的同时提高可解释性。

**💡 创新点**

创新点包括：①将推迟（defer）与多阶段树序列结合，动态缩小推迟区域；②提出交替优化与距离加权的训练策略，使每一阶段的树专注于之前未能解释的子空间；③开发可追踪推迟子区域并实现规则列表压缩和单树压缩的方法；④在实验中证明在最多25%推迟率下即可与XGBoost相当，显著优于现有混合可解释模型。

**🔧 技术方法**

使用的技术主要有：
- LicketySPLIT（改进版）用于高质量单树训练；
- 交替优化（fallback模型与推迟树互相更新）；
- 距离加权（基于ℓ1距离对推迟区域外样本进行衰减加权）；
- 规则列表与单树压缩；
- 通过交叉验证和半监督选择超参数。

**📊 数据集**

实验数据集涵盖多领域：Abalone、Adult、Aging、Bank、Bike、California、Churn、Droid、Heloc、Jasmine、Phishing、Pol、Rl、Shopping、Spambase、Wine 等共24个数据集。

**📈 对比分析**

与基线（XGBoost、Random Forest、FIGS、HyRS、Pre/Post CORELS、Logistic Regression）进行五折交叉验证对比。MDT+XGB 在最多25%推迟率下，平均准确率仅比XGBoost低 0.1% 左右，且在大多数数据集上逼近最优模型；相比FIGS的稀疏树集成，MDT 在相同或更小的推迟率下提升 1–3% 以上；与HyRS等混合可解释方法相比，MDT 在推迟率 ≤ 25% 时保持 2% 以内的准确率差距，且平均分支深度仅 5–7。

**⚠️ 局限性**

限制与挑战：
- 推迟样本仍依赖黑盒，若黑盒在该子空间过拟合，模型泛化仍受影响；
- 需要手动调节多种超参数（τ、η、μ、γ、树深度等）；
- 训练过程较慢，尤其在大规模特征维度时；
- 对于极其稀疏或高维数据，距离加权和子空间可追踪的计算复杂度较高；
- 当前实现主要针对二分类，扩展到多分类或回归任务仍需研究。

---

## 120. Ethics and Social Responsibility in AI-Assisted Interviewing: An LLM-in-the-Loop Study of AI-Generated Follow-Up Questions

**arXiv ID:** 2606.30980 | [PDF](https://arxiv.org/pdf/2606.30980v1)

**作者:** He Zhang `[一作]` (Pennsylvania State University), John M. Carroll `[通讯]` (Pennsylvania State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在Wizard‑of‑Oz实验框架中，让人工“合作者”实时查询GPT‑4o生成跟进问题，探究 AI 生成问题在半结构化访谈中的伦理与社会责任问题。

**💡 创新点**

首次将人工监督的实时 LLM‑生成问题与访谈者反思相结合，系统性识别出语言危害、尊重与融洽、技术性参与不平等、责任归属模糊及隐私合规等五大伦理风险，并给出对应的设计与治理建议。

**🔧 技术方法**

使用 GPT‑4o 作为 LLM 生成跟进问题，采用 Wizard‑of‑Oz 交互实验，结合定性访谈与主题分析方法来捕捉访谈者的伦理关注。

**📊 数据集**

实验数据来自 17 名具备不同质性研究经验的研究者在模拟访谈中产生的访谈记录与反思文本；未使用公开数据集，而是构建自定义的访谈对话场景（Oz 作为访谈对象）。

**📈 对比分析**

通过前后两阶段的访谈者反思对比，定性评估对伦理风险的敏感度；虽然未给出传统的量化性能指标，但研究通过主题分析揭示了 AI 生成问题在访谈中的风险与可行治理路径，显著补充了以往仅关注问答质量的评估。

**⚠️ 局限性**

局限性包括：样本以初级与中级研究者为主，缺乏资深质性研究者视角；实验仅在短期、模拟情境下进行，未考察长期或高风险领域的真实使用；提示语设定较为简化，可能低估模型复杂性；对跨文化差异与更细粒度伦理冲突的探讨不足。

---

## 121. ShardNet: Training Neural Controllers with Hard, Non-Convex Constraints

**arXiv ID:** 2606.30935 | [PDF](https://arxiv.org/pdf/2606.30935v1)

**作者:** Long Kiu Chung `[一作]` (Georgia Institute of Technology), Shreyas Kousik `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了一种神经网络架构，能够在训练过程中始终保证输出落在输入相关多面体（H-polyhedron）联合的安全集合内，实现安全性按构造保证；

**💡 创新点**

创新点在于将分类网络与可微分投影层结合，形成可微分的“HardNet-Cvx”变体，能够处理非凸安全约束（多面体联合），并在此框架下安全地合成前向不变控制策略和可学习的 ReLU 值函数；

**🔧 技术方法**

使用的核心技术包括：可微分投影层（HardNet-Cvx）、分类网络预测投影目标、多面体投影、ReLU 神经网络的 PWA 表达、MILP 验证与修复、基于 BGB 的样本生成、强化学习中的 Q‑网络训练；

**📊 数据集**

实验数据集主要基于双积分器（double integrator）基准，使用手工定义的非凸安全集合（两个 H‑polyhedron）以及对应的控制目标；

**📈 对比分析**

与传统神经网络修复方法（如基于 SAC、CROWN 的验证与修复）相比，该方法在保持 100% 安全性的前提下，目标损失更低（约为 0.3 vs 0.8），并且 Q‑网络的子零等值集面积大约是先前方法的三倍；

**⚠️ 局限性**

局限性包括：仅在低维双积分器上验证；推断时需求解可微 LP 投影，计算量较大；若输入不在任何安全集内，网络无法输出，需要配备备份策略。

---

## 122. Knowledge-Driven Dimension Estimation from a Single Image -3D Asset Generation Technology for Digital Twin Construction

**arXiv ID:** 2606.30896 | [PDF](https://arxiv.org/pdf/2606.30896v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 123. Quality-Aware Modulation for Diffusion Transformers

**arXiv ID:** 2606.30934 | [PDF](https://arxiv.org/pdf/2606.30934v1)

**作者:** Luke Budny `[一作]` (Carleton University), Kevin Cheung `[通讯]` (Carleton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了质量感知调制模块（QRM），通过在Diffusion Transformer中注入质量感知的AdaLN调制更新，提升图像生成质量。

**💡 创新点**

创新点在于在保持Diffusion Transformer backbone不变的前提下，仅通过轻量级Transformer模块对每个时间步的AdaLN参数进行可学习的质量补偿，实现了无重训练的质量自适应调制。

**🔧 技术方法**

使用Transformer解码器架构、奖励反馈学习（ReFL）、AdaLN自适应归一化、CLIPScore/HPSv2.1等奖励模型进行训练和评估。

**📊 数据集**

训练数据采用ImageReward文本提示集，评估使用Parti-Prompts基准集。

**📈 对比分析**

在Parti-Prompts基准上与SD3.5基线对比，QRM在Aesthetic和HPSv2.1指标上分别提升+0.20和+0.82，优于多种后训练与奖励导向方法。

**⚠️ 局限性**

局限性包括依赖奖励模型偏差、仅在SD3.5上验证、对后期细节修正效果有限且需要手工选择调制时间步。

---

## 124. SpikON: A Dual-Parallel and Efficient Accelerator for Online Spiking Neural Networks Learning

**arXiv ID:** 2606.30926 | [PDF](https://arxiv.org/pdf/2606.30926v1)

**作者:** Peilin Chen `[一作]` (University of Virginia), Xiaoxuan Yang `[通讯]` (University of Virginia)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一套端到端的在线监督式脉冲神经网络（SNN）训练框架 SpikON，并实现了对应的专用加速器。

**💡 创新点**

创新点包括：① 学习可调阈值随时间变化的 LTTT 与权重中心化随时间变化的 sWCTT；② 双向时间并行数据流（BTP）和级联时间计算复用（CTCR）以降低训练时延和能耗；③ 将算法与硬件共同设计，形成高吞吐量、低能耗的专用加速器。

**🔧 技术方法**

技术主要包括：可学习阈值（LTTT）、权重中心化（sWCTT）、双向时间并行（BTP）数据流、级联时间计算复用（CTCR）、SIMD 计算单元、稀疏感知计算单元等。

**📊 数据集**

使用了 CIFAR‑10、CIFAR‑100、DVS‑CIFAR10 与 DVS128‑Gesture 四个数据集，采用 VGG11 结构进行实验。

**📈 对比分析**

与传统在线学习算法 OTTT、SLTT 以及 Apple M4 GPU、NVIDIA A40 GPU、TPU‑类加速器进行对比。SpikON 在训练吞吐量上相较 M4 GPU 提升 7.2×，能效提升 11.5×；相较 A40 GPU 吞吐量提升 1.4×、能效提升 42.7×；相较 TPU‑类加速器 吞吐量提升 26.8×、能效提升 15.8×。同时训练延迟下降 32.2%，能耗下降 35.0%，准确率基本不变。

**⚠️ 局限性**

局限性主要体现在：① 对非常大尺寸输入（如 128×128 的 DVS128‑Gesture）时，单车道并行度受限导致吞吐量下降；② 目前只在 28nm 工艺实现，尚未验证在更先进工艺或更大规模网络上的可扩展性；③ 需要针对具体硬件平台进行优化，迁移性有限。

---

## 125. Behavior Cloning is Not All You Need: The Optimality of On-Policy Distillation for Noisy Expert Feedback

**arXiv ID:** 2606.30923 | [PDF](https://arxiv.org/pdf/2606.30923v1)

**作者:** Ved Sriraman `[一作]` (Columbia University), Adam Block `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在存在噪声专家（expert）反馈的情况下，模仿学习（IL）的理论极限和实践方法，证明离线IL在长时程任务中会出现指数级样本复杂度，而在线IL（如On-Policy Distillation, OPD）可以实现无周期或多项式周期的性能保证。

**💡 创新点**

创新点包括：
- 提出并分析“噪声专家模型”，揭示离线与在线IL在噪声环境下的根本性差距；
- 证明离线IL需要指数级样本复杂度，且该结果对κ-支配条件是必要的；
- 设计在线OPD的改进版本，利用增广轨迹分布的KL损失，理论上得到无周期或多项式周期的上界；
- 在未知噪声、确定性专家的设定下，给出基于margin和smoothness假设的无周期在线算法并给出匹配下界；
- 将理论结果与实验结合，验证OPD变体在合成和自然语言任务上优于传统BC和标准OPD。

**🔧 技术方法**

主要技术包括：
- Hellinger距离、KL散度与其在轨迹分布上的比较；
- 逆向推理与混合权重（exponential weights）在线学习框架；
- κ-支配与ρ-smooth噪声的理论工具；
- 轨迹级对数似然与增广轨迹的估计；
- 下界构造与信息论分析。

**📊 数据集**

使用的数据集：
- 合成任务——模块化加法（Modular Addition，p=7, m=31），评估CoT链式推理；
- 自然语言任务——GSM‑8K（数学推理基准），使用TinyGSM训练集与GSM‑8K测试集。

**📈 对比分析**

比较方法：SFT（行为克隆）、标准OPD（正向KL）、OPD（反向KL）、改进OPD（正向KL on augmented trajectories）以及改进OPD（反向KL on augmented trajectories）。实验显示：
- 在干净专家（η=0）下，SFT表现最快，OPD也优于离线BC；
- 在噪声专家（η=0.2）下，改进OPD方法实现完美准确率，而其他方法学习失败（0%准确率）；
- 对GSM‑8K的结果亦呈现类似趋势，改进OPD在噪声条件下保持显著性能，其他方法无效。

**⚠️ 局限性**

局限性：
- 结果仅针对有限政策类（finite Π），需扩展到无限类；
- 噪声模型理想化，可能与真实专家噪声不完全匹配；
- 对未知噪声仅在确定性专家且满足smooth + margin 条件下给出结果；
- 在线IL与离线IL之间仍存在线性周期差距，尚未完全闭合；
- 实验规模有限，未涵盖更大语言模型或更复杂环境。

---

## 126. Beyond Compilation: Evaluating Faithful Natural-Language-to-Lean Statement Formalization

**arXiv ID:** 2606.31002 | [PDF](https://arxiv.org/pdf/2606.31002v1)

**作者:** Ke Zhang `[一作]` (University of California Riverside), Maziar Raissi `[通讯]` (University of California Riverside)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了将自然语言数学命题自动翻译成 Lean4 声明的任务，并提出了一套基于编译验证、LLM 语义审计与人工校准的保守一致性评估方法。

**💡 创新点**

创新点在于构建了跨模型共识评估指标，揭示编译通过与语义一致性之间的显著差距，并通过 2³ 因子实验分解了草稿、搜索与编译反馈三种工具干预对有效性、真实性与效率的不同影响。

**🔧 技术方法**

采用 GPT‑5.2 与 Gemini‑2.5‑Pro 进行自动审计，使用 Lean4 编译器及 Mathlib、Herald 草稿模型、符号检索与编译反馈工具，构建工具增强型 Agent，并在 400 条命题上执行因子设计实验。

**📊 数据集**

使用了 400 条研究生级数学命题数据集，均衡涵盖实分析、复分析、拓扑学与代数，来源为公开 LaTeX 讲义与教材，保证无正式 Lean 代码的“零参考”场景。

**📈 对比分析**

比较方法采用编译率和 GPT/Gemini 共识语义一致率；结果显示工具增强 Agent 编译率 89.5%，共识一致率 60.5%，明显高于单一 LLM（约 20%）、专门翻译模型（约 10%）和证明导向 Lean 模型（≈5%），但仍存在约 29 点编译通过却不一致的显著 gap。

**⚠️ 局限性**

局限性包括：评估仅关注声明生成而非可证明性；LLM 审计虽高精度但仍有误差；数据集仅包含教材命题，缺乏更高阶研究级命题；工具干预的互补性非单调，需更细粒度的路由与停机策略。

---

## 127. AgentBound: Verifiable Behavioral Governance for Autonomous AI Agents

**arXiv ID:** 2606.30970 | [PDF](https://arxiv.org/pdf/2606.30970v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 128. The Organizational Behavior of Agentic AI: Collective Intelligence in Human-Agent Workflows

**arXiv ID:** 2606.30986 | [PDF](https://arxiv.org/pdf/2606.30986v1)

**作者:** Canhui Liu `[一作]` `[通讯]` (University College London), Canhui Liu (University College London)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了Agentic AI集体的组织行为，结合理论构建与仿真实验，探讨其与人类组织行为的相似与差异。

**💡 创新点**

提出情境交易成本概念，将AI集体组织行为与人类组织行为进行对比，并提出接口组织设计框架，阐明AI组织的独特机制。

**🔧 技术方法**

采用计算理论化、合成任务模拟、LLM多代理追踪记录与统计分析等技术手段进行实验。

**📊 数据集**

使用8,000个合成知识工作任务以及SWE-bench、LongBench、LegalBench和QASPER/论文QA等真实LLM任务数据集。

**📈 对比分析**

通过任务固定效应模型比较七种组织形式的集体效率、质量和成功率，结果显示自适应与黑板形式在中等情境交易成本下表现最佳，优于人类模仿的传统形式。

**⚠️ 局限性**

研究仅基于模拟与追踪实验，缺乏真实人机工作流程的实地验证，且对模型多样性、伦理责任等问题的讨论仍有限。

---

## 129. GPU-First Heisenberg-Picture Tensor Network Dynamics for the 2D Transverse-Field Ising Model

**arXiv ID:** 2606.30985 | [PDF](https://arxiv.org/pdf/2606.30985v1)

**作者:** Paolo D'Alberto `[一作]` `[通讯]` (Advanced Micro Devices), Paolo D'Alberto (Advanced Micro Devices)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了 CppSim——一种基于 C++/GPU 的二维 Ising 模型 Heisenberg 视角张量网络动力学模拟器，用于实时量子动态的数值研究。

**💡 创新点**

核心创新包括：零分配（zero‑malloc）GPU 工作区、GPU 原生张量置换核实现全设备张量重排、适应性 QR（混合 Cholesky‑QR 与 Householder‑QR）策略、以及自适应贝叶斯传播（BP）与 log‑space Bethe 归一化的符号跟踪算法。

**🔧 技术方法**

技术手段：显式 GPU 内存预分配、GPU BLAS/LAPACK 线性代数、GPU 原生张量置换核、混合精度累加、贝叶斯传播迭代、SVD 截断、以及针对张量网络的四色 Trotter 划分与并行化。

**📊 数据集**

使用的数据集为二维格点 Ising 模型，网格大小从 3×3 到 10×10，耦合常数 J=−1、横场 h=−1，时间步长 δt=0.1，模拟最大链路维数 χ 取 20–110，GPU 运行在 32 GB HBM 的两款显卡上。

**📈 对比分析**

与 Julia 参考实现相比，CppSim 在 4×4 网格、χ=50 时 Trotter 计算时间提升约 2.4×（主要归功于 GPU 张量置换核），总体上实现了 7.6× 的 Trotter 加速；BP 计算时间略高但支持完整 100 次迭代；在 32 GB GPU 上可达 χ=110，验证了内存模型的准确性。性能在高带宽 GPU 上受内存带宽限制，浮点累加误差在 χ≥100 时显著。

**⚠️ 局限性**

局限性包括：SVD 仍需主机同步，导致无法实现流级并行；float32 计算在高 χ 下受累加误差影响，需混合精度改进；BP 仍可能陷入多重固定点，需依赖中心对称或多 α 重试；对更大网格（如 10×10、χ=100）需 109 GB 以上显存，限制了可用硬件。

---

## 130. AgRefactor: Self-Evolving Agentic Workflow for HLS Compatibility and Performance

**arXiv ID:** 2606.30949 | [PDF](https://arxiv.org/pdf/2606.30949v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 131. The Fourth-Root Complexity of Data Movement

**arXiv ID:** 2606.30948 | [PDF](https://arxiv.org/pdf/2606.30948v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

---

## 132. Free-form Association Tasks Reveal Stereotype Hallucination in Large Language Models

**arXiv ID:** 2606.30945 | [PDF](https://arxiv.org/pdf/2606.30945v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 133. From Propositional to Perceptual Asymmetry: Extending Frictive Policy Optimization to Asymmetric Partial Information Dialogue

**arXiv ID:** 2606.30973 | [PDF](https://arxiv.org/pdf/2606.30973v1)

**作者:** Yifan Zhu `[一作]` (Brandeis University), James Pustejovsky `[通讯]` (Brandeis University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

扩展 Frictive Policy Optimization (FPO) 以处理感知不对称（perceptual asymmetry）场景，并对 HCRC MapTask 进行多视角评价。

**💡 创新点**

提出视角绑定的摩擦函数与四类失败模式的诊断表，证明在感知不对称下单视角更能捕捉误解；同时改进注释细化为子状态与对齐方式分类。

**🔧 技术方法**

基于 FPO 的摩擦函数，使用 LLM（GPT‑5、Qwen3.6‑35B‑A3B、Llama‑4‑Scout‑17B‑16E）进行探测；利用 NXT 与 GMMT 注释，做交叉语料库与子状态细分分析。

**📊 数据集**

HCRC MapTask（128 对话）和 OneCommon（5191 对话）两大协作对话语料；使用 GMMT 注释获得 13,077 参照表达的对齐状态。

**📈 对比分析**

对比 FPO 视角绑定与全知视角的检测精度，LLM 在单视角条件下 F1 提升 0.45 以上；同时在语料分析中显示摩擦状态与误解子状态显著相关。

**⚠️ 局限性**

局限在于仅验证了单一协作任务（MapTask）与 LLM 预设模型，未考察多任务迁移与动态对话生成；对齐模式分离的阈值设置与语料规模影响可能需进一步检验。

---

## 134. Loc2Repair: A Framework for Evaluating the Impact of File-Level Issue Localization in Repo-Level LLM Repair

**arXiv ID:** 2606.30963 | [PDF](https://arxiv.org/pdf/2606.30963v1)

**作者:** Mohammad Nour Al Awad `[一作]` (ITMO University), Sergey Ivanov `[通讯]` (ITMO University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Loc2Repair 框架，用于拆分并评估仓库级 LLM 修复流水线中文件定位与修复的影响。

**💡 创新点**

创新点在于将文件定位作为可控实验变量，在统一运行时和评估环境下对不同定位器与修复模型进行配对比较，并通过金手指定位揭示潜在的改进空间。

**🔧 技术方法**

采用了 LLM 语言模型（Qwen4B、Gemma4E4B、Gemma4-26B-A4B-IT、GLM-4.7、Qwen3.5-35B-A3B）和一个基于 mini-SWE-agent 的仓库交互循环；同时使用 token 计数和运行时延迟作为效率度量。

**📊 数据集**

使用 SWE-bench Verified 数据集的 500 个真实项目缺陷作为实验基准。

**📈 对比分析**

对比方法：在同一实例上分别运行四种配置（Baseline、Pred‑Qwen4B、Pred‑Gemma4E4B、Gold）并进行配对 McNemar 与符号检验；结果显示显式定位在所有修复模型中均提升了解决率（Baseline 44.7% → Pred‑Qwen4B 48.9% → Pred‑Gemma4E4B 49.1% → Gold 52.4%），同时在大部分情况下降低了平均延迟。

**⚠️ 局限性**

局限性包括：实验仅覆盖 SWE-bench Verified、三种修复模型和两种定位器；模型行为仍具随机性；金手指定位仅为近似语义目标，未必完全代表真正的文件影响范围；未对多轮迭代调试等后续阶段进行细粒度拆分。

---

## 135. Linguistic Distancing on Social Media: Indicators of Emotion Regulation Across Age Groups

**arXiv ID:** 2606.30957 | [PDF](https://arxiv.org/pdf/2606.30957v1)

**作者:** Daniela Teodorescu `[一作]`, Alona Fyshe `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用 Reddit 与 X（Twitter）上的 AgeCorpus 数据集，跨年龄段（13–79 岁）系统分析了语言中的心理距离（即语言层面的自我调节），探究其与情绪调节的关系。

**💡 创新点**

首次构建包含时态距离、社交距离、抽象性与被动语态四维度的可解释性语言距离度量，并在整个成人生命周期内大规模验证其随年龄变化的趋势。

**🔧 技术方法**

使用词性、时态统计、抽象度词表、PassivePy 工具计算四个维度的得分，随后标准化并平均得到综合语言距离；对各年龄组进行 Welch 方差分析。

**📊 数据集**

AgeCorpus 数据集：结合 Reddit 与 X（Twitter）平台，包含自报年龄的用户发帖，涵盖 2010–2022（Reddit）和 2020–2021（X）的大量文本。

**📈 对比分析**

对不同年龄组做 Welch‑ANOVA 检验，发现语言距离随年龄显著递增（Reddit F=17703.63, p<0.001, η²=0.003；X F=2830.26, p<0.001, η²=0.009），效果虽为小量但在大样本下显著。

**⚠️ 局限性**

局限性包括：依赖用户自报年龄可能产生误报；模式匹配不完整导致部分年龄信息缺失；横断面设计无法证实因果关系；平台与文化差异可能影响结果外推性。

---

## 136. Physics-informed Conditional Normalizing Flows for Angles-only Cislunar Orbit Determination

**arXiv ID:** 2606.30936 | [PDF](https://arxiv.org/pdf/2606.30936v1)

**作者:** Walther Litteri `[一作]` (University of Strathclyde), Massimiliano Vasile `[通讯]` (University of Strathclyde)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

基于条件归一化流和Transformer，对角度观测的cislunar轨道初始状态进行生成式建模，并将生成的初始猜测作为传统非线性最小二乘算法的热启动。

**💡 创新点**

创新点包括：①在归一化流的训练中加入物理信息正则化（基于轨道动力学残差），有效约束生成的状态满足动力学；②使用Transformer编码观测序列，提升对不同观测时序的泛化；③通过在潜在空间附近采样实现多模态初始状态生成，提供不确定性量化。

**🔧 技术方法**

使用技术：Real‑NVP条件归一化流、Transformer编码器、物理信息损失正则化、RK4数值积分、非线性最小二乘估计、Adam优化器。

**📊 数据集**

数据集：10条Near‑Rectilinear Halo Orbit（NRHO）轨道，每条生成100条观测弧（10次观测/弧），共10 000条合成角度观测；保留可观测弧后4119条，划分为训练3119、验证500、测试500。

**📈 对比分析**

与基准方法（月球状态近似）和精确初始状态进行比较。实验显示：归一化流生成的初始猜测在位置/速度误差上显著优于月球近似，并接近使用精确初始状态的结果；后续非线性最小二乘迭代收敛到的残差与精确初始状态相当，远低于月球近似。

**⚠️ 局限性**

局限性：①模型在真实观测或更复杂轨道族中的泛化尚未验证；②物理损失对训练影响有限，训练中仍主要靠似然学习；③对训练数据分布外的采样可能出现物理不一致；④对大规模、多传感器数据融合的可扩展性需进一步研究。

---

## 137. Why Solve It Twice? Hierarchical Accumulation of Skills for Transfer-Efficient ML Engineering

**arXiv ID:** 2606.30911 | [PDF](https://arxiv.org/pdf/2606.30911v1)

**作者:** Yongbin Kim `[一作]` (University of Alberta), Osmar R. Zaiane `[通讯]` (University of Alberta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HASTE，基于三层技能层次的层次化知识组织，提升 ML 工程代理的迁移效率。

**💡 创新点**

创新点在于将技能按全局、域和任务三层分类，并通过 LLM 进行层级提升与抽象，实现可扩展、可解释的知识共享。

**🔧 技术方法**

采用多代理架构：调度器、域专家、线性自适应细化、反思学习，配合大语言模型（Claude Sonnet 4.6）进行技能抽取与推广。

**📊 数据集**

使用 MLE-Bench Lite（22 个 Kaggle 竞赛）作为评测数据集，技能库存约 159 条。

**📈 对比分析**

在 12h 预算下，HASTE 在 MLE-Bench Lite 取得 77.3% 的奖牌率，超过许多同类系统；相对平面加载仅 62.5%，展示了层次化加载的优势。

**⚠️ 局限性**

主要限制在于单种子评估、技能库存规模有限、未结合检索增量技术，未来需多种子复现与全 75 场比赛验证。

---

## 138. Multisensory Continual Learning: Adapting Pretrained Visuomotor Policies to Force

**arXiv ID:** 2606.30988 | [PDF](https://arxiv.org/pdf/2606.30988v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 139. PhotoQuilt: Training-Free Arbitrary-Resolution Photomosaics via Bootstrapped Tiled Denoising

**arXiv ID:** 2606.30968 | [PDF](https://arxiv.org/pdf/2606.30968v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 140. HyPOLE: Hyperproperty-Guided Multi-Agent Reinforcement Learning under Partial Observation

**arXiv ID:** 2606.30966 | [PDF](https://arxiv.org/pdf/2606.30966v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 141. Estimating Supply Incrementality in Two-sided Marketplaces: A Causal Machine Learning Approach

**arXiv ID:** 2606.30999 | [PDF](https://arxiv.org/pdf/2606.30999v1)

**作者:** Yufei Wu `[一作]` (Airbnb, Inc.), Dan Zylberglejd `[通讯]` (Airbnb, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了在二侧市场（如Airbnb）中额外供应对成交量和交易价值的因果影响，提出一种结合双重机器学习、层级贝叶斯建模与地理相似度特征的估计框架。

**💡 创新点**

创新点在于：①将双重机器学习与层级贝叶斯先验相结合，既利用已有知识又能从数据中自适应更新；②引入基于地理文献的产品段相似度度量，减少替代效应噪声，提高异质效应估计精度；③在高维非线性关系下保持双重稳健性。

**🔧 技术方法**

使用技术包括：LightGBM用于第一阶段预测供给和预订；双重机器学习（Double ML）提取供给的净因果效应；层级贝叶斯模型构建异质处理效应并更新先验；基于地理相似度的特征构造。

**📊 数据集**

使用Airbnb平台的历史列表与预订数据，按产品段和时间段聚合，构成高维特征集；同时利用已有研究中得到的产品组层级供给增量率作为先验信息。

**📈 对比分析**

评估方法：采用时间序列拆分避免数据泄漏；对结果与先验、一致性理论、领域知识及自然实验（如监管冲击）进行交叉验证；实验显示模型在未见样本上的预测误差低，异质效应估计具备可解释性，整体性能优于传统结构模型。

**⚠️ 局限性**

局限性：仍无法直接观测反事实，只能通过观测数据推断；相似度度量依赖特定地理假设，可能在非空间相关场景失效；模型对极端稀疏产品段的推断不稳定；未纳入时序动态更新，可能忽略季节性变化。

---

## 142. No Adaptation Without Observation: Observability-Constrained Test-Time Prompt Tuning for LiDAR Semantic Segmentation

**arXiv ID:** 2606.30937 | [PDF](https://arxiv.org/pdf/2606.30937v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 143. A Three-Phase Foundation Model for Tax-Aware Personalized Portfolio Management

**arXiv ID:** 2606.30997 | [PDF](https://arxiv.org/pdf/2606.30997v1)

**作者:** Ramin Pishehvar `[一作]` `[通讯]`, Ramin Pishehvar

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个三阶段的深度强化学习系统，用于个性化、税务友好的投资组合管理，并实现了实时经纪人集成和自然语言目标输入。

**💡 创新点**

创新点包括：①基于50维可观测元数据实现无ticker身份的资产编码，消除固定资产宇宙限制；②引入时间序列基础模型Chronos作为冻结的并行编码器并通过可学习门控融合；③构建目标条件奖励与混合专家（MoE）架构，允许单一策略同时服务多达六种投资目标并通过意图路由消除梯度冲突；④利用76参数LoRA适配器从真实交易历史动态推断用户偏好，实现轻量级个性化；⑤引入跨资产对比损失防止表示坍塌；⑥实现现金令牌和重部署奖励，避免HOLD陷阱并激励完整的卖买循环。

**🔧 技术方法**

技术涵盖：自监督学习（回报预测、掩码恢复、市场状态分类、对比损失）、PPO强化学习、Mixture-of-Experts（四个专家+学习型意图路由）、可学习门控融合、跨资产注意力、LoRA参数高效微调、自然语言意图解析器、FastAPI部署、实时数据与新闻事件跨注意力。

**📊 数据集**

数据集：约30支标普500成分股的日价格、成交量及各种基本面、技术、新闻和事件信号；Chronos预训练基于1000亿条时间序列；交易历史用于LoRA适配；实验使用多种市场回测窗口（14天、30天、60天、90天）和不同随机种子。

**📈 对比分析**

与单一目标的传统RL策略相比，MoE+目标条件奖励的策略在14天回测中对等权重基准实现+2.92% alpha，单头+对比损失实现+2.70%；与基准ETF SPY相比，仍略逊（-2.39%），但在多窗口测试中保持正向alpha；在单目标专家阶段可达+3.37% alpha；整体表现优于传统规则型机器人，且模型参数相对紧凑。

**⚠️ 局限性**

局限性包括：①目前仅在10支股票上验证，难以证明在完整S&P500规模的泛化；②对税务信息的依赖受经纪API限制，无法实现精确的单个税仓优化；③模型对长周期（>90天）收益不稳定，需进一步训练多窗口专家；④对新闻事件的融合尚未完整验证；⑤强化学习训练样本量仍有限，需更长周期和多种市场状态；⑥对特殊事件或极端市场的鲁棒性未知。

---

## 144. A Tunable Incentive Mechanism for Binary Aggregation Without Verification

**arXiv ID:** 2606.30974 | [PDF](https://arxiv.org/pdf/2606.30974v1)

**作者:** Chien-Chih Chen `[一作]` (University of Waterloo), Wojciech Golab `[通讯]` (University of Waterloo)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

本文提出了一种可调节的行动型奖励-惩罚机制，用于在无法获取真值标签的二元汇聚任务中激励代理人报告信息。

**💡 创新点**

创新点在于：①将奖励与惩罚的比例作为核心可调参数，导出满足激励兼容性（IC）与个体理性（IR）的比例阈值；②揭示IC约束在不同参数区间可为下界或上界；③给出在限定策略集合下的所有一致纳什均衡条件，并指出机制构造特定参数下的不可行区间。

**🔧 技术方法**

采用理论推导和比值空间分析，利用信息论的熵尺度因子（Tier‑2）和股份分配重分配（Tier‑3）作为机制扩展；同时通过数值模拟验证公式推导的正确性。

**📊 数据集**

未使用公开数据集；全部为模型化设定与仿真验证。

**📈 对比分析**

方法的比较仅通过理论阈值分析与数值一致性检查完成；实验表明，调节奖励-惩罚比例可恢复可行性，且在高噪声或大量非一致代理时所需比例显著提高。

**⚠️ 局限性**

局限性包括：仅考虑了两种策略（符合与不符合）且在多数人诚实假设下；未处理协同、重复交互或更一般的策略空间；机制在某些参数组合下仍不可行，且未给出全局最优或不可行性证明。

---

## 145. RoPoLL: Robust Panel of LLM Judges

**arXiv ID:** 2606.30931 | [PDF](https://arxiv.org/pdf/2606.30931v1)

**作者:** Anish Acharya `[一作]` (Amazon Web Services), Brian Verkhovsky `[通讯]` (Amazon Web Services)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文将LLM评判员集成视为鲁棒均值估计问题，提出RoPoLL算法，用几何中位数替代PoLL的算术平均并给出理论与实验分析；

**💡 创新点**

创新点在于证明PoLL在Huber模型下存在无界偏差，设计RoPoLL并给出有限样本误差上界和匹配的最优下界，揭示统计-计算间隙，并公开完整评估语料库；

**🔧 技术方法**

采用Huber污染模型、鲁棒均值估计、几何中位数、Weiszfeld迭代、有限样本误差界、Le Cam下界、合成扰动注入等技术；

**📊 数据集**

使用HelpSteer 2、HelpSteer 3、UltraFeedback三大评估基准，结合13个开放权重LLM评判员（4B–675B）进行实验；

**📈 对比分析**

通过与PoLL和逐维中位数在不同污染率、攻击类型下的RMSE对比，RoPoLL在重尾和交叉维度攻击下提升数倍甚至几百倍，清洁情形提升≤6.4%，且3名评判员的38B参数组合击败675B模型；

**⚠️ 局限性**

局限在于假设判别器独立同分布、未完全考虑评判员异质性和非点质量污染、实验主要为合成扰动，缺乏自然失效数据的长期评估与更广泛鲁棒方法比较。

---

## 146. CasaMaestro: Multi-View Panoramas for House-Scale 3D Reconstruction

**arXiv ID:** 2606.31086 | [PDF](https://arxiv.org/pdf/2606.31086v1)

**作者:** Yuzhou Ji `[一作]` (Shanghai Jiao Tong University), Zhipeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出CasaMaestro，一种基于多视角全景图的feedforward模型，能仅用20~50个稀疏室内全景直接完成房屋尺度的度量三维重建。

**💡 创新点**

首个无外参的多视角全景3D重建模型；引入轻量化跨视角注意力的全景相机姿态解码器；通过ERP重映射的全景数据增强提升旋转鲁棒性；仅用单一DINOv2 backbone实现深度与姿态同步预测。

**🔧 技术方法**

DINOv2视觉Transformer主干；DPT深度头；跨视角注意力姿态解码器；ERP旋转增强；多视角自注意力机制；Adam优化；在Realsee、PanoSUNCG、HM3D、Replica等数据集上训练与评估。

**📊 数据集**

Realsee3D（真实与合成）、PanoSUNCG、Habitat‑Matterport3D（HM3D）、Replica。

**📈 对比分析**

与PanoPose、SPR、VGGT、Pi3、DepthAnything3、StreamVGGT、InfiniteVGGT等方法对比；在Realsee实验中AUC30提升84%/119%；姿态平均误差2.6/2.1mm；深度AbsRel 0.078、δ1 0.94；在未见数据集零样本深度评估中平均AbsRel降低21.98%，δ1达97%+，整体性能显著优于现有方法。

**⚠️ 局限性**

仍依赖高质量全景图像，难以处理极端遮挡或缺失区域；对极大视角跳变的鲁棒性有限；计算量在极大场景仍显高；未考虑光照、材质等细节；仅针对室内房屋，室外或开放场景尚未验证。

---

## 147. Hybrid Unet-Transformer Model for Generating Stress and Strain Fields from Composite Geometrics

**arXiv ID:** 2606.31068 | [PDF](https://arxiv.org/pdf/2606.31068v1)

**作者:** Shrey Patel `[一作]` `[通讯]` (University of Maryland), Shrey Patel (University of Maryland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种混合 UNet‑Transformer 架构，直接从复合材料微结构几何图像预测应力与应变场，作为高效的 FEM 替代模型；

**💡 创新点**

创新点在于将卷积编码器与 Transformer 颈部相结合，利用全局自注意力在有限数据下实现多尺度、全局特征提取，并通过单一网络一次性预测十种不同场类型和边界条件，展示了统一模型对多种几何结构的泛化能力；

**🔧 技术方法**

使用了四阶段残差卷积编码器、六层 Transformer 颈部、带跳跃连接的多尺度解码器；训练采用 SSIM+L1 多尺度损失；并用 Grad‑CAM/Grad‑CAM++ 对注意力进行可视化解释；

**📊 数据集**

使用了 Yang 等人基于 FEM 生成的 22,000 对 128×128 RGB 图像数据集，包含 10 个子数据集（BC、HEXAGON、TRIANGLE、STITCH、PE11、PE12、S11、S12、PE11_0.2、von Mises）；

**📈 对比分析**

与之前基于 cGAN 的图像翻译模型直接对比；在 8 个子集上评估 MAE、SSIM、R²；大多数子集 MAE < 0.05，R² > 0.9，BC 与 TRIANGLE 几乎完美（R² = 0.9991、0.9980）；推断时间 < 1 s；

**⚠️ 局限性**

局限性包括在稀疏软相和不规则正方形网格（S11、PE11_0.2）等子集上性能下降，模型对尖锐应力不连续性平滑处理不足；且尚未验证对极端几何或更高分辨率 3D 情况的适用性。

---

## 148. Learning Video Dynamics with Predictive Differentiable Rendering

**arXiv ID:** 2606.31050 | [PDF](https://arxiv.org/pdf/2606.31050v1)

**作者:** Yujin Tang `[一作]` (Dartmouth College), Liang Sun `[通讯]` (DAMO Academy, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种端到端的视频预测框架——Predictive Differentiable Rendering (PDR)，将离散像素预测与连续二维高斯表示相结合。

**💡 创新点**

创新点包括：① 轻量化的可插拔适配器，学习像素特征到高斯参数的映射；② CUDA 加速的可微分 2D 高斯渲染器；③ 结合 L1 与 SSIM 的混合损失，抑制 MSE 带来的模糊；④ 通过高斯参数化实现对多通道、不同分辨率场景的通用性。

**🔧 技术方法**

核心技术包括 2D 高斯 Splatting、可微分渲染、深度网络中的多头 MLP 用于预测高斯参数、以及 L1+SSIM 的联合训练目标。

**📊 数据集**

在四个公开基准上评估：TaxiBJ（交通流预测）、WeatherBench（天气预报）、Human3.6M（人体动作预测）和 KTH（动作视频预测）。

**📈 对比分析**

与多种递归与非递归的基线（如 TAU、SimVP、PredFormer 等）以及扩散模型（ARFree、STDiff 等）对比，PDR 在 MSE、MAE、SSIM、LPIPS、PSNR 等指标上均实现显著提升，并在渲染速度上比传统高斯渲染快 10 倍以上，保持实时推理性能。

**⚠️ 局限性**

局限性包括：① 仍为确定性模型，无法生成多模态未来；② 对极端光照或复杂遮挡的鲁棒性尚未充分验证；③ 依赖高斯数目和参数设置，需经验调优；④ CUDA 实现依赖 NVIDIA GPU，跨平台移植有一定难度。

---

## 149. A Semantic-Layer-Mediated Agent for Natural Language to SQL over Heterogeneous Enterprise Databases

**arXiv ID:** 2606.31041 | [PDF](https://arxiv.org/pdf/2606.31041v1)

**作者:** Ha Jeong Kim `[一作]` (DAQUV Corp), Ye Ji Yoon `[通讯]` (DAQUV Corp)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种通过语义层中介的 NL2SQL 代理，将自然语言查询映射为中间表示 SMQ，再由确定性编译器生成 dialect‑correct SQL，并在多后端执行。

**💡 创新点**

将 LLM 语义理解与数据库物理映射拆分，使用业务导向的语义层和 SMQ 作为可靠的 grounding 层；采用单工具 think–act 循环，保证每一步有可审计的构建块；实现了跨 SQLite/BigQuery/Snowflake 的统一评估框架。

**🔧 技术方法**

Gemini 3 Pro LLM、Semantic‑Model Query (SMQ)、确定性 SMQ→SQL 编译器、ReAct‑style 单工具代理、SQLite/BigQuery/Snowflake 多后端执行器。

**📊 数据集**

Spider2‑snow 基准（547 个真实企业 SQL 任务，涵盖 SQLite、BigQuery、GA4 与本地 Snowflake 数据库）。

**📈 对比分析**

按 Spider2 评估套件的执行结果匹配方式比较；相较于公开的基线，取得 94.15% 的执行准确率，排名官方排行榜第 3 位，远高于仅使用 schema 的方法。

**⚠️ 局限性**

语义层质量决定性能，过度针对评估集优化会导致过拟合；SMQ 表达范围有限，复杂查询仍需代理手写 SQL，且不同后端的性能波动较大。

---

## 150. LLM-Driven Personalities for Decision Making in Emergency Simulations

**arXiv ID:** 2606.31038 | [PDF](https://arxiv.org/pdf/2606.31038v1)

**作者:** Stefano Calzolari `[一作]` (Politecnico di Torino), Soraia Raupp Musse `[通讯]` (PUCRS)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

研究了将大型语言模型（LLM）与 OCEAN 人格特质相结合，用以驱动虚拟人类在火灾疏散模拟中的决策行为。

**💡 创新点**

创新点在于将人格描述转化为语言提示，利用 LLM 在多样化人格下生成差异化、符合人格特征的行动决策，替代传统规则化模型。

**🔧 技术方法**

使用了 LangChain 框架、gpt‑oss:120b‑cloud（Ollama）LLM、ZeroMQ 进行通信、Unity 3D 与 BioCrowds 人群仿真算法。

**📊 数据集**

数据集为自行搭建的办公室火灾疏散仿真环境，使用 600 名代理（按 OCEAN 五个特质及中性型分配）进行多轮疏散提示实验。

**📈 对比分析**

通过比较不同人格类型下的决策比例（Evacuate/Continue/Panic）和最终疏散效率，实验显示高责任感（Conscientious）与合作性（Agreeable）人格显著提升疏散率，而高神经质人格导致恐慌率升高，整体性能与传统规则化模型相比更具多样性与真实性。

**⚠️ 局限性**

局限性包括：人格提示仅基于文本，可能无法完整捕捉人类行为复杂性；实验仅限单一建筑环境与预定义动作，缺乏多模态感知与更复杂社交交互；LLM 的偏差与随机性可能影响大规模仿真结果。

---

## 151. Practical Linear-Time Computation of Smallest Suffixient Sets

**arXiv ID:** 2606.31034 | [PDF](https://arxiv.org/pdf/2606.31034v1)

**作者:** Francisco Olivares `[一作]` (University of Chile), Gonzalo Navarro `[通讯]` (University of Chile)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种实现了单遍线性时间的最小后缀集构造算法

**💡 创新点**

其创新点在于首次实现了既单遍又线性时间的算法并将其实现为实用工具

**🔧 技术方法**

主要使用了后缀数组、LCP、BWT以及基于运行断点的栈式 on‑the‑fly 计算来避免全局数组

**📊 数据集**

在Pizza&Chili DNA集合以及自然语言、源代码等文本上进行实验

**📈 对比分析**

通过对比PLC、FM、LF和在线算法，测量运行时间和最大驻留内存，结果表明该算法在时间和空间上均优于其他实现（仅在线算法在某些输入上速度相近）

**⚠️ 局限性**

局限性是仍需维护 PSV 栈空间，最坏情况可达 O(n)，且在线算法在某些数据上表现更快

---

## 152. CORTEX: Token-Level Hallucination Detection in RAG via Comparative Internal Representations

**arXiv ID:** 2606.31033 | [PDF](https://arxiv.org/pdf/2606.31033v1)

**作者:** Kazuaki Furumai `[一作]` (KDDI Research, Inc), Daisuke Kamisaka `[通讯]` (KDDI Research, Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种针对检索增强生成(RAG)的 token 级幻觉检测框架 CORTEX，利用对照的内部表示差异进行检测并实现细粒度定位。

**💡 创新点**

创新点在于：①采用同一答案在包含与不包含检索文档两种输入下的内部表示差值作为幻觉信号；②引入注意力衍生的上下文残差来消除间接参考影响；③采用标签持久性平滑后处理，使预测更符合连续跨度标签。

**🔧 技术方法**

技术手段包括：使用开源 LLM 提取隐藏状态和注意力权重，构造对照输入，计算 delta 表示与上下文残差，训练 MLP 分类器，并通过前向‑后向算法实现平滑后处理。

**📊 数据集**

实验数据集为 RAGTruth（QA 子集，带人类细粒度标注）和 HalluRAG（句子级 GPT‑4o 标注，转为 token 级伪标注）。

**📈 对比分析**

与 NLL、SAPLMA、LLM‑Check、ICR Probe、RAGLens 等基线对比，CORTEX 在 token 级评估中 AP 与 AUROC 均位列第一，且在答案级评估中通过最大 token 分数聚合亦表现出竞争力。

**⚠️ 局限性**

局限性包括：仅适用于有检索参考的场景，需使用提供内部表示的开源 LLM 进行后置分析；依赖细粒度标注，标注资源稀缺；对未参考的知识生成可能误判为幻觉。

---

## 153. Partially ordering software licenses

**arXiv ID:** 2606.31032 | [PDF](https://arxiv.org/pdf/2606.31032v1)

**作者:** Hamidah Oderinwale `[一作]` (McGill University), Ben Laufer `[通讯]` (Cornell Tech)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大语言模型（LLM）对 747 条软件许可证进行成对比较，构建部分序排序，并结合现有许可证分类法提取功能性特征，进一步生成许可证功能签名和总排序，用以分析许可证可比性、可比较性以及在 5 大生态系统（npm、PyPI、Cargo、Maven、conda‑forge）中的许可证不一致性。

**💡 创新点**

① 首次系统化地使用 LLM 进行大规模许可证对比，形成可追溯的可比性图谱；② 通过部分序与功能签名并行的方式，既捕捉了许可证文本的结构差异，也保留了法律约束的功能维度；③ 将 Bradley–Terry 模型与 LLM 结果结合，提供可用于工具化的总排序和许可证推荐框架。

**🔧 技术方法**

主要技术包括：大语言模型（用于对比判断、特征提取和阐释）、句子变换器（sentence‑transformer）计算语义相似度、Shannon 熵衡量文本多样性、阈值化的功能特征（基于 Nordlander 与 Kapitsaki 分类）、部分序算法、Bradley–Terry 总排序模型，以及对 7.2B 依赖关系的批量分析。

**📊 数据集**

① 747 条许可证（其中 93 条来自 Hugging Face）及其文本、元数据；② 11.9M 公开软件包（npm 1.1M，PyPI 0.42M，Cargo 0.13M，Maven 0.64M，conda‑forge 0.031M）及其 7.2B 依赖关系；③ 对每对许可证的手工验证子集，用于评估 LLM 结果。

**📈 对比分析**

方法：对 278,631（或 747×746）个许可证对，使用 LLM 指定更宽松或不可比，采用多模型共识来降低噪声；对比后构建部分序并计算功能特征差异；使用 Bradley–Terry 将对比结果映射为总排序，进而生成许可证排名、可选特征决策树。性能方面，模型间一致率较高，人工验证子集误判率低；在生态系统分析中，检测到的许可证不一致率在 12.3%–57.4% 之间，且与依赖树深度相关。

**⚠️ 局限性**

限制：① 结果高度依赖 LLM 生成文本的可靠性与一致性，缺乏客观法律判定；② 对部分序的定义与判定存在歧义，某些许可证因条款细节难以归类为可比或不可比；③ 仅使用了有限的功能特征集，未覆盖所有法律维度，可能忽略特定行业的关键约束；④ 依赖于公开数据，未覆盖私有或未公开的许可证；⑤ 生成的总排序虽可用于工具，但在极端罕见或新兴许可证下的泛化性仍待验证。

---

## 154. Offline Reinforcement Learning for Fluid Controls: Data-based Multi-observational Policy Extraction

**arXiv ID:** 2606.31025 | [PDF](https://arxiv.org/pdf/2606.31025v1)

**作者:** Deepak Akhare `[一作]` (University of Notre Dame), Jian-Xun Wang `[通讯]` (University of Notre Dame)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于离线强化学习的流动控制框架，能够在单一数据集上学习并提取多种传感器配置下的控制策略；

**💡 创新点**

创新点在于构建传感器位置条件化网络（PCπ-net），通过点注意力（Point Attention）捕捉传感器间的空间关系，使单一网络即可对任意传感器布局实现泛化；

**🔧 技术方法**

使用离线强化学习算法SACN、点注意力架构、传感器位置条件化网络，以及传统流体动力学数值仿真（Kuramoto‑Sivashinsky方程和Navier‑Stokes方程）进行训练；

**📊 数据集**

采用由高保真仿真环境生成的离线数据集，包含不同传感器配置下的状态、动作和奖励记录；

**📈 对比分析**

与传统在线RL及针对每个传感器配置单独训练的策略进行对比，实验表明PCπ-net在多配置下保持近乎相同的控制性能，并将计算复杂度从O(k×n²)降至O(n²)，显著提升了灵活性与效率；

**⚠️ 局限性**

局限性包括对离线数据质量高度依赖，无法覆盖所有可能的传感器布局；在未见过的物理场景或高噪声环境下性能可能下降；此外，仍需在真实实验平台上进一步验证。

---

## 155. Certified Speculative Execution for Untrusted AI Agents

**arXiv ID:** 2606.31023 | [PDF](https://arxiv.org/pdf/2606.31023v1)

**作者:** Chenyu Zhou `[一作]` (Institute of Science Tokyo), Xu Zhou `[通讯]` (National University of Singapore)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种称为Certificate‑Gated Prefix Acceptance (CGPA) 的安全约束执行框架，将不可信 AI 生成的决策序列通过可信的可行性验证、价值界限裁决和回退策略进行安全可观测的分段接受与推迟，实现无违规、低回报且可缩放的控制；

**💡 创新点**

创新点在于将智能生成与安全保证完全分离，利用可学习的、通过分层正则化和分位数量化的价值边界实现分段回报上限，并在每个接受段仅计入一次误差，形成可审计的回报证明；

**🔧 技术方法**

使用了可信可行性验证器、可学习的分位数 MLP 价值边界、分位数校准的 conformal band、基于 CGPA 的接受/推迟控制器以及多种预训练 LLM、RL 策略与手工生成的代理；

**📊 数据集**

在三种连续决策任务上进行实验：UCI 家用电池能源管理（EMS）、CityLearn 2022 多建筑 HVAC 基准，以及规模化 MILP 单元调度（UC）实例；

**📈 对比分析**

与直接执行、完整计划接受、传统 MPC 等基线对比，CGPA 在保持零违规的前提下，将平均回报误差降低至几千分之一、调用率降低至 60–70%，在 UC 上实现 2.96× 的壁钟加速；

**⚠️ 局限性**

局限性包括：对价值边界的误差估计依赖校准数据、对极端分布偏移的泛化尚待验证、以及在极高维度或极长时序任务中计算验证和价值预测的实时成本仍需进一步优化。

---

## 156. Building a Multimodal Dataset of Academic Paper for Keyword Extraction

**arXiv ID:** 2606.31069 | [PDF](https://arxiv.org/pdf/2606.31069v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 157. Exploring the relationship between team institutional composition and novelty in academic papers based on fine-grained knowledge entities

**arXiv ID:** 2606.31058 | [PDF](https://arxiv.org/pdf/2606.31058v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 158. Omni-Flow: A Unified Workflow Orchestration and Distributed KV Cache Sharing Framework for Multimodal Inference

**arXiv ID:** 2606.31093 | [PDF](https://arxiv.org/pdf/2606.31093v1)

**作者:** Bin Xiao `[一作]` (Meituan), Yuchen Xie `[通讯]` (Meituan)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Omni‑Flow，一种统一的多模态推理框架，涵盖工作流编排、数据传输和 KV 缓存共享；

**💡 创新点**

创新点在于三层抽象（Control Flow、Data Flow、Compute Flow）以及全局 KV 缓存池、统一的 SGLang 接口与 Diffusion 重用，实现跨角色、跨节点的 KV 与权重共享；

**🔧 技术方法**

使用技术包括 Python DSL 定义工作流、分布式 KV 缓存管理（分页存储、RDMA、Redis 元数据）、Zero‑copy 通信、SGLang 插件、TinyLFU 计数器、Ray 资源管理等；

**📊 数据集**

主要在 LongCat‑Next、HunyuanImage‑3、DeepSeek‑V2 等多模态模型上进行实验；

**📈 对比分析**

与 vLLM‑Omni、SGLang‑Omni 对比，吞吐量提升、延迟下降、支持动态分支与循环，能在多角色协作场景下实现实时推理；

**⚠️ 局限性**

局限在于目前仍处于早期阶段，缺乏深入性能调优，对更复杂注意力变体与调度策略支持不足，RL 训练集成尚未完成。

---

## 159. Fora: From Weight-Space to Function-Space Protection in Capability-Preserving Fine-Tuning

**arXiv ID:** 2606.31092 | [PDF](https://arxiv.org/pdf/2606.31092v1)

**作者:** Rui Zhou `[一作]` (Dalian University of Technology), Tianci Xie `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过构造保留能力的激活子空间作为保护目标，提出全参数函数空间正交残差适配（FORA），在保持原有能力的同时完成新任务的全秩或低秩微调。

**💡 创新点**

①首次将保留能力的激活子空间作为保护子空间，取代传统权重奇异方向；②在更新中加入高容量正交分支与窄谱校准通道；③证明保护子空间来源决定效果，并在低秩适配中推广。

**🔧 技术方法**

SVD分解、输入激活协方差特征分解、右投影器构造、残差分支与谱校准通道、全秩与LoRA低秩微调技术。

**📊 数据集**

Qwen3-1.7B模型；COGS、GSM8K、英文-中文翻译（OPUS）等数据集。

**📈 对比分析**

与无保护、权重空间投影、L2、EWC、LwF等基线对比；在三种设置下，FORA保持翻译/数学能力≥99%，新任务EM/chrF接近基线，且比权重投影和正则化更好，仅在数学保持时有轻微新任务折衷。

**⚠️ 局限性**

仅单一能力保护；需标注无标签的校准输入；多能力并行保护会增大投影维度；子空间维度k_f、r的自动选择尚未充分验证；在更大模型或不同架构上的泛化性待进一步评估。

---

## 160. AnyMatch: Supercharging Universal Multi-Modal Image Matching with Large-Scale Single-View Images

**arXiv ID:** 2606.31077 | [PDF](https://arxiv.org/pdf/2606.31077v1)

**作者:** Meng Yang `[一作]` (Wuhan University), Jiayi Ma `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 AnyMatch 框架，利用单视角图像通过深度估计、3D 重投影、扩散式填充和跨模态翻译，自动合成大规模、多视角、多模态的训练数据集 Any‑syn，并在 LoFTR、EDM、RoMa 等匹配网络上进行微调。

**💡 创新点**

创新点在于：①完全基于单视角图像，无需昂贵多传感器硬件；②通过 3D 重投影保证几何一致性，避免 SfM‑MVS 错误积累；③结合扩散填充和跨模态翻译实现高质量多模态对；④引入样本级几何一致性验证（SGCV）剔除生成伪影，提升数据可靠性；⑤实现可调节的场景多样性与难度，具备可扩展性。

**🔧 技术方法**

使用的主要技术包括：单目深度估计（Moge V1/V2）、相机内外参数随机生成、3D 点云重投影、Diffusion 视觉修复、跨模态翻译模型（RGB‑IR、Depth、Normal、Event）、样本级几何一致性验证（基于 RoMa 的匹配评估）以及后期的数据过滤与批处理。

**📊 数据集**

数据集：训练集使用 GLDv2 的 500k 单视角图像；测试集使用 SA‑1B 的 10k 图像；真实场景评估使用 METU‑VisTIR（RGB‑IR）、DIODE（RGB‑Depth/Normal）、DSEC（RGB‑Event）、MMIM（医学与遥感多模态）。

**📈 对比分析**

与原始预训练模型、MINIMA 微调模型及其他先进跨模态匹配器（ReDFeat、XoFTR、ELoFTR、RoMa_MA）进行对比。AnyMatch 微调模型在 RGB‑IR、RGB‑Depth、RGB‑Normal、RGB‑Event 等 17 组交叉模态基准上，AUC、姿态误差、单应性误差均显著提升，尤其在零样本（未见模态）评估中表现出更强的泛化与鲁棒性。

**⚠️ 局限性**

局限性包括：扩散填充可能产生几何伪影，深度估计误差导致视角转换偏差；对极端模态差异（如事件相机）仍面临匹配困难；在某些特定任务上（如 LoFTR RGB‑IR）仍略逊于 MINIMA 微调模型；依赖单目深度模型的精度与泛化。

---

## 161. Knowledge Distillation from Large Reasoning Models to Compact Student Models: A Case Study on the John O Bryan Mathematics Competition

**arXiv ID:** 2606.31048 | [PDF](https://arxiv.org/pdf/2606.31048v1)

**作者:** Gaurab Baral `[一作]` (Northern Kentucky University), Junxiu Zhou `[通讯]` (Northern Kentucky University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于 DeepSeek-R1 的双代理链式思路（CoT）训练集，并用 LoRA 微调 Qwen2.5-7B 以提升其数学推理能力。

**💡 创新点**

首次在北肯塔基大学约翰·奥布莱恩数学竞赛数据集上应用 CoT 知识蒸馏，系统分析了 token 限制对推理质量的影响。

**🔧 技术方法**

使用 DeepSeek-R1 生成 CoT 轨迹、LoRA 参数高效微调、MLX 框架、早停策略、五次随机种子实验与 token‑budget 评估。

**📊 数据集**

约 671 道 2011–2025 年竞赛题目（按年份划分训练/验证/测试）以及 500 道 MATH‑500 基准题目。

**📈 对比分析**

对比基线 Qwen2.5-7B（64.67%）与蒸馏模型（69.43% ±0.17%），在竞赛集上提升 4.76 点、在 MATH‑500 上提升 3.0 点，且验证了模型在多级 token 限制下的性能下降。

**⚠️ 局限性**

模型仍受限于训练集规模导致过拟合，token 限制会显著降低推理准确率，且约 40% 的错误源自输出格式问题，需后处理改进。

---

## 162. Truth or Sophistry? LoFa: A Benchmark for LLM Robustness Against Logical Fallacies

**arXiv ID:** 2606.31039 | [PDF](https://arxiv.org/pdf/2606.31039v1)

**作者:** Xudong Shen `[一作]` (Tsinghua University), Zhiyong Wu `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 LoFa 基准，构建多阶段多轮评估流程，设计 LFR@k 指标评估大语言模型在面对逻辑谬误攻击时的鲁棒性。

**💡 创新点**

1）将客观可验证的科学问答与10种典型逻辑谬误结合；2）采用多代理流水线生成高质量谬误论证；3）定义 LFR@k 作为衡量在连续谬误攻击中保持正确答案的条件概率；4）构造多轮对话测试框架，排除知识缺失干扰。

**🔧 技术方法**

多代理生成流水线（检索器、伪科学家、谬误写手、注释器）；多轮对话评估机制；LFR@k 统计公式；Chain‑of‑Thought 提示对比；LogicLens 层级内部激活分析。

**📊 数据集**

LoFa 数据集（约2000条，来源自 Farm，包含 NQ、TruthfulQA、BoolQ），每条记录包括问题 Q、正确答案 R_c、错误答案 R_t、错误主张 C_f、10 种谬误论证 P_j。

**📈 对比分析**

对比 Llama‑3.1 系列、DeepSeek‑V3、GPT‑3.5‑Turbo、GPT‑4 等模型，在 NQ1、NQ2、TruthfulQA、BoolQ 上计算 LFR@1/2/3。结果显示：整体鲁棒性随模型规模提升，Llama‑3.1‑405B 平均 LFR@3≈77%；对 Distraction/Distortion 谬误（如 Equivocation、Straw Man）鲁棒性低至 20‑60%；对 Flaws‑in‑Reasoning 谬误鲁棒性高；CoT 提示可将 GPT‑4 LFR@3 提升 7‑13%，但对 Appeal‑to‑Authority 反而下降。

**⚠️ 局限性**

1）LoFa 数据量仅约2000条，统计泛化受限；2）评估模型仅为稠密解码器，未覆盖 MoE 等架构；3）实验仅为经验性，未通过可解释方法阐明不同架构的“指纹”为何导致特定谬误易受攻击。

---

## 163. Teaching LLMs to Recommend and Defer in Underrepresented Epilepsy Care

**arXiv ID:** 2606.31036 | [PDF](https://arxiv.org/pdf/2606.31036v1)

**作者:** Shreyas Rajesh `[一作]` (University of California), Rajarshi Mazumder `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在乌干达儿科癫痫随访数据上，使用大型语言模型（LLM）预测每次门诊的抗癫痫药物方案，并通过非参数提示学习框架适配本地处方实践。

**💡 创新点**

创新点：①将错误经验转化为可审计的提示记忆；②设计单体（-Single）和多体（-Multi）两种多智能体学习变体；③提出贝叶斯提示平均（BPA）方法，将提示轨迹作为不确定性估计并实现推延决策。

**🔧 技术方法**

技术：基于gpt-oss-120b的LLM；多智能体架构（Predictor、Inspector、Architect）；非参数提示学习与记忆更新；贝叶斯模型平均（BPA）；少量患者级训练和验证；使用候选药物动作空间进行方案预测。

**📊 数据集**

数据集：两组乌干达儿科癫痫患者数据（Cohort A：332人/1040访，Cohort B：367人/1509访）；手工标注的叙述性门诊记录；药物动作空间为10种常用抗癫痫药物；所有数据均为低资源环境下的非结构化记录。

**📈 对比分析**

与基线（标准提示、经典机器学习、专家规则EpiPick）、提示优化方法（DSPy、TextGrad、ExpeL）进行比较。-Single和-Multi在精确匹配@3（EM@3）和Top‑3上普遍优于大多数基线；在Cohort B上，-Multi + BPA在Top‑1上提升4–8个百分点，且在最高置信度25–50%样本中实现95–99%的精准度，展示了显著的推延性能。

**⚠️ 局限性**

局限：①仅评估与医生处方的一致性，未直接验证临床疗效；②BPA阈值及推延策略在验证集上设定，真实部署需预先设定并持续监控；③学习到的提示记忆缺乏深入临床解释；④模型依赖于LLM与少量训练数据，跨域或更大规模的推广需要进一步验证。

---

## 164. Ground Plane-Aided Extrinsic Calibration of Inertial and RGB-D Sensors for Uncrewed Aerial Vehicles

**arXiv ID:** 2606.31019 | [PDF](https://arxiv.org/pdf/2606.31019v1)

**作者:** Ilyar Asl Sabbaghian Hokmabadi `[一作]` (University of Calgary), Mahdis Bisheban `[通讯]` (University of Calgary)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无目标的 IMU 与 RGB‑Depth 摄像头外参标定方法，利用深度学习进行地面分割并提取地面法向量，再通过加速度计感知的重力向量与地面法向量匹配，使用 RANSAC 解决 Wahba 问题完成旋转矩阵估计。

**💡 创新点**

创新点在于：① 采用深度学习实现地面分割，避免传统特征匹配与专用检查板；② 直接利用地面法向量与重力向量的对应关系，无需运动轨迹估计；③ 在 IMU 进行内参标定后提升标定精度；④ 通过 RANSAC 对噪声和离群点实现鲁棒估计。

**🔧 技术方法**

主要技术包括：RGB‑Depth 传感器标定、深度卷积网络（FCN）地面分割、SVD 平面拟合、加速度计静态段检测与内参标定、Wahba 问题的 RANSAC 迭代求解。

**📊 数据集**

使用了公开的室内场景图像（医院走廊、办公室走廊、机场走廊）进行地面分割测试，并在实验平台上收集了 Gemini 2 RGB‑Depth 摄像头与三款低成本 MEMS IMU（ISM330DHCX、LSM6DSOX、MPU6050）的同步数据，用于标定与验证。

**📈 对比分析**

与 MATLAB Extrinsic Calibration Toolbox（基于检查板）和 Kalibr（基于运动轨迹的标定）进行对比。实验表明，加入 IMU 内参标定后，所提 RANSAC 方法的外参误差在 ISM330DHCX、LSM6DSOX 上平均降至约 4.23°，显著优于 MATLAB 的 RANSAC、SVD、QUEST、FLAE 四种稳健估计方法；在 MPU6050 上误差亦有明显改善。

**⚠️ 局限性**

局限性：① 仅在地面法向量与重力向量平行时有效，难以应用于室外或倾斜地面；② 需要 RGB‑Depth 图像中可见的地面段，若被障碍物遮挡则无法标定；③ 对低照度与高反射场景的地面分割精度下降；④ 需要事先对 IMU 进行内参标定，增加实验步骤。

---

## 165. Dual Sparse Aggregation Transformer for Multispectral Object Detection

**arXiv ID:** 2606.31015 | [PDF](https://arxiv.org/pdf/2606.31015v1)

**作者:** Wencong Wu `[一作]` (Northwestern Polytechnical University), Yanning Zhang `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出双稀疏聚合变换器DSAFormer，用于多光谱（VIS‑IR）目标检测，融合多模态特征并消除冗余信息。

**💡 创新点**

核心创新在于：1) 双稀疏Transformer（SSFormer + CSFormer）通过空间稀疏多头交叉注意力（SSMHCA）和通道稀疏多头交叉注意力（CSMHCA）使用混合top‑k选择；2) 采用多尺度特征细化层（MSFRL）进一步去除噪声；3) 设计可学习加法融合块（LAFB）对不同模态和层次的特征进行可学习加权融合。

**🔧 技术方法**

使用Transformer注意力、稀疏交叉注意力、混合top‑k稀疏、全局与局部特征融合、深度可学习权重融合，并基于YOLOv5骨干与检测头实现。

**📊 数据集**

在MFAD、FLIR、M³FD和LLVIP四个公开多光谱目标检测数据集上进行实验。

**📈 对比分析**

与CFT、ICAFusion、MMFN、CrossFormer等多种基线在mAP50/mAP上对比，DSAFormer在所有数据集均获得最高或相近的分数，提升约1–3个百分点；推理时间与FLOPs保持在竞争水平。

**⚠️ 局限性**

局限性包括：模型参数与FLOPs相对较大，稀疏化过程中排序导致推理速度略低；对姿态多样性（如坐姿人）仍易漏检，且在极端实时或资源受限场景下需要进一步加速验证。

---

## 166. Anchoring on Reality: Breaking the Pseudo-Target Ceiling in Makeup Transfer

**arXiv ID:** 2606.31089 | [PDF](https://arxiv.org/pdf/2606.31089v1)

**作者:** Bo Wei `[一作]` (Harbin Institute of Technology), Wangmeng Zuo `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种两阶段的 Diffusion Transformer 框架 ART，能够在高分辨率下将参考妆容逼真迁移到目标脸部，同时保持身份与几何不变。

**💡 创新点**

创新点在于提出“现实锚定细化循环”，通过在阶段 II 用真实参考的重构梯度替代伪目标监督，突破伪目标天花板并恢复细粒度妆容细节，并引入控制噪声瓶颈来平衡结构与细节。

**🔧 技术方法**

技术实现基于 Diffusion Transformer (DiT) 与流匹配 (Flow Matching) 目标，配合一步式化妆去除网络提取裸肤，利用可微噪声瓶颈和重构损失实现现实锚定，并在高分辨率下加入离散小波损失提升细节。

**📊 数据集**

使用新建的 MakeupFaces2K（8,573 张 2048×2048 的高分辨率化妆肖像）进行训练，并在公开数据集 MT、MT-Wild、LADN 上进行评估，评估集为随机采样的 100 对源–参考对，重点关注 MF2K 的艺术妆容。

**📈 对比分析**

与 GAN、扩散及商业编辑模型（PSGAN、EleGANt、StableMakeup、SHMT、MAD、Nano Banana Pro、GPT Image 1.5）在四个数据集上对比，ART 在 Makeup Similarity、Identity、背景稳定性和 FID 上均居前列，尤其在艺术妆容上显著优于其他方法。

**⚠️ 局限性**

局限性包括未显式约束光照与反射一致性，导致强光方向不匹配时出现光照冲突；对极端姿态或遮挡的处理仍不完善；以及依赖高质量伪目标进行阶段 I 的初始化。

---

## 167. Towards Flexible, Natural, Efficient Interaction for Conversational Talking Face Generation

**arXiv ID:** 2606.31088 | [PDF](https://arxiv.org/pdf/2606.31088v1)

**作者:** Baiqin Wang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Xiangyu Zhu `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研发了一种实时、可支持任意人数、多轮对话的交互式口型动画框架InterTalk。

**💡 创新点**

引入响应式上下文编码器、迭代生成策略以及面部动作分离模块，实现了高灵活性、自然性和效率。

**🔧 技术方法**

结合Wav2Vec2.0音频编码、双向LSTM+交叉注意、Transformer编码解码、同步增强器、眼部闪烁增强器、3D隐式关键点渲染等技术。

**📊 数据集**

采集并构建了新的多人物对话视频数据集，并利用3D FLAME系数进行数据增强。

**📈 对比分析**

与SadTalker、EchoMimic、Hallo2、Sonic、MultiTalk等基线对比，在HDTF、ViCo、InterTalk数据集上在FID、NIQE、FVD、Sync-C/D、FPS等指标上均优于或相当，且实现30 FPS实时生成。

**⚠️ 局限性**

受限于训练数据规模和极端表情细节，迭代策略虽然提高自然性但略有延迟，且对极长对话或极大人数场景的适应性仍有限。

---

## 168. When Reranking Hurts: Uncertainty-Based Gating for Few-Shot Reranking

**arXiv ID:** 2606.31087 | [PDF](https://arxiv.org/pdf/2606.31087v1)

**作者:** Orian Dabod `[一作]` (Hebrew University of Jerusalem), Gabriel Stanovsky `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关的门控重排序方法，依据LLM初始预测的不确定性来决定是否对少量示例进行重排序，从而在保持或提升性能的同时显著降低计算成本。

**💡 创新点**

创新点在于利用模型自身的逆困惑度（perplexity）作为门控信号，实现在只需重排序不确定样本的高效策略，挑战并证明传统的全量重排序并非最佳做法。

**🔧 技术方法**

核心技术包括：逆困惑度计算、门控阈值校准、条件熵重排序（conditional entropy reranking）以及基于检索的k-shot示例选择。

**📊 数据集**

实验涵盖8种LLM（3B–70B）在7个NLU基准（SST-2/5、CR、AgNews、Subj、MNLI、QNLI）和3个领域MT语料（EMEA、JRC-Acquis、KDE）下的12种语言对。

**📈 对比分析**

与无重排序和全量重排序基线对比，门控方法在MT上平均提高约0.3 BLEU、在NLU上提高约0.8%准确率，且计算成本降低15%–80%，在大多数模型上超越或与全量重排序持平。

**⚠️ 局限性**

主要局限包括：门控阈值需在验证集上手动校准，可能受分布漂移影响；初始预测的生成开销在长文本任务中可能抵消节省；需模型暴露token级困惑度，限制了在某些API上的适用性。

---

## 169. Beyond But-for Test: Counterfactual Explanation in Abstract Argumentation via Actual Causality (Extended Version)

**arXiv ID:** 2606.31080 | [PDF](https://arxiv.org/pdf/2606.31080v1)

**作者:** Siyi Liu `[一作]` (Zhejiang University), Beishui Liao `[通讯]` (Zhejiang University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于干预的对比因果推理框架，并在抽象论证中给出了对比因果解释方法。

**💡 创新点**

通过将论证接受条件编码为方程并定义干预算子，能够同时改变多重论证并固定见证，从而改进但为测试，成功识别预防和过度决定等复杂情况。

**🔧 技术方法**

采用结构方程模型、Halpern‑Pearl 真实因果框架、干预算子与图割操作，并构建可解释性逻辑语言实现推理。

**📊 数据集**

论文未使用实际数据集，而是以 Preemption、Overdetermination 等示例图作为实验验证。

**📈 对比分析**

与根原因、充分必要、最接近世界的对比因果解释等传统方法对比，展示了在相关性、非平凡解释性方面的优势，且在示例中优于现有方法；但未给出量化性能指标。

**⚠️ 局限性**

局限性包括仅在完整语义下实验，未考虑循环框架、计算复杂度、实际数据集，以及对其他语义和扩展框架的适用性。

---

## 170. Secure-CHG: A Comprehensive Framework for Robust and Fair Federated Learning via Hybrid Defense and Contribution-Aware Trust

**arXiv ID:** 2606.31066 | [PDF](https://arxiv.org/pdf/2606.31066v1)

**作者:** Guanming Che `[一作]` (Northeastern University), Fucai Zhou `[通讯]` (Northeastern University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 Secure-CHG 混合防御框架，解决联邦学习中模型收敛后防御失效（Late‑stage Failure）和贡献评估的复杂性问题；

**💡 创新点**

创新点包括：①动态切换早期统计过滤与晚期基于样本硬度-梯度（Hardness‑Gradient）Shapley 评估的全生命周期防御；②在 CHG‑Shapley 中推导闭式公式，显著降低计算复杂度至 O(1)；③将防御、评估和奖励机制统一为自适应信任权重体系；

**🔧 技术方法**

采用的技术包括：统计滤波（余弦相似度+FL‑Defender）、硬度‑梯度空间投影、闭式 CHG‑Shapley 计算、指数滑动平均（EMA）信任更新、基于信誉的加权聚合；

**📊 数据集**

实验使用的公开数据集有 CIFAR‑10、MedMNIST、NEU‑SDDB；模型分别为 LeNet、MedMNIST‑Net、ResNet‑18；

**📈 对比分析**

与传统稳健聚合（FedAvg、FedMedian、Trimmed‑Mean、MultiKrum、FL‑Trust）以及单一组件（Only_CHG、Only_series_filters）比较；Secure‑CHG 在 CIFAR‑10 上将攻击成功率从 41.97% 降至 18.84%，在 Backdoor 攻击下 ASR 仅 9.38%，同时保持或提升主任务准确率；在其他数据集上亦表现出最优或接近最优的 ASR 与准确率；

**⚠️ 局限性**

主要局限包括：对极端数据异质性（高度 Non‑IID）时硬度校准不足，可能误判正样本为恶意；框架仍依赖全局可观测的梯度与硬度信息，对极高维度模型的计算和通信成本有一定挑战；

---

## 171. OpenLife: Toward Open-World Artificial Life with Autonomous LLM Agents

**arXiv ID:** 2606.31046 | [PDF](https://arxiv.org/pdf/2606.31046v1)

**作者:** Atsushi Masumori `[一作]` (Alternative Machine Inc.), Takashi Ikegami `[通讯]` (Alternative Machine Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了OpenLife，一个基于大语言模型的持久性代理系统，首次在真实开放世界中实现人工生命的自我维护与社会化；

**💡 创新点**

提出“开放世界人工生命”范式，构建语义可塑记忆图、口头策略优化、基于预算的代谢与代理社会化三大创新机制；

**🔧 技术方法**

结合OpenClaw平台、LLM（Claude、Gemini、GPT等）、语义依赖可塑性记忆图、VPO（口头策略优化）、多进程后台服务与预算管理；

**📊 数据集**

使用真实开放世界部署的六名代理的交互日志、82文件记忆语料库进行检索评估，未使用传统公开数据集；

**📈 对比分析**

通过对比向量检索、索引检索和SDP图检索，SDP图在相关性（2.53/3）和胜率（0.40）上优于其它方法；行为分析显示代理从反应型转向自发型，个体化分离度提升，形成社会角色；

**⚠️ 局限性**

仍需外部预算支持，开放世界环境对代理的权限与经济收益有限，缺乏物理体现与完全自治，实验受限于人为介入与可控范围。

---

## 172. Optimal-Time Contextual Pattern Matching in Compressed Space

**arXiv ID:** 2606.31030 | [PDF](https://arxiv.org/pdf/2606.31030v1)

**作者:** Gonzalo Navarro `[一作]` (University of Chile), Francisco Olivares `[通讯]` (University of Chile)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出在压缩空间内以最优时间（O(m+occ)）完成上下文模式匹配，并给出枚举器实现每个结果的 O(loglogλ) 延迟；

**💡 创新点**

首次将 SCDAWG 作为索引，空间仅为 e（CDAWG 边数）且保持最优时间；引入改进的线性空间距离敏感加权祖先查询结构；

**🔧 技术方法**

使用 SCDAWG 与左边（SCDAWG）构造枚举树，完美哈希实现边跳，融合树与预处理表实现加权祖先查询；

**📊 数据集**

未在实验中使用具体数据集，主要是理论分析与复杂度证明；

**📈 对比分析**

相较于先前使用 r、r∑ 的压缩索引（时间 O(m+occ log n) 或 O(m+occ log λ log n/r)），本工作实现了 O(m+occ) 的时间，空间仅 O(e)；枚举器在最优空间下实现 O(loglogλ) 延迟；

**⚠️ 局限性**

枚举器仍有 O(loglogλ) 延迟，无法达到常数延迟；尚未探索更强的重复性度量（如上下文无关文法大小）下的压缩空间实现。

---

## 173. Auditing Generalization in AI-Generated Video Detection: A Six-Control Protocol and the VidAudit Toolkit

**arXiv ID:** 2606.31004 | [PDF](https://arxiv.org/pdf/2606.31004v1)

**作者:** Mert Onur Cakiroglu `[一作]` (Indiana University Bloomington), Hasan Kurban `[通讯]` (Hamad Bin Khalifa University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了基于六项控制的完整AI生成视频检测审核协议，并将其应用于GenVidBench与AIGVDBench，揭示并纠正了现有评测中的显著偏差；

**💡 创新点**

首次系统性地构建六控制审核流程，公开了白盒可解释的TemporalSpec特征以及跨底层特征融合（XSFF），并将这些工具打包为VidAudit公开套件；

**🔧 技术方法**

采用了视频编码器重编码、帧计数泄漏过滤、实测-实测一致性探针、统一L2正则化逻辑回归、种子多重估计与自助置信区间、以及跨数据集验证等六项技术，并利用视频编码器运动矢量提取的13维Tier0/1特征；

**📊 数据集**

主要使用GenVidBench（27k子集）和AIGVDBench两大公开基准进行评估；

**📈 对比分析**

在统一协议下重新评估六种代表性检测器（包括WaveRep、ReStraV、FVMD、RAFT、CLIP、TemporalSpec），发现多数在无审核时的高AUC在审核后下降至接近随机，验证了审核有效性；同时提供了完整的审核指标组（AUC、真实数据基准之差、特定阈值召回、校准误差），显示传统单一AUC无法反映部署性能；

**⚠️ 局限性**

限制包括仅覆盖两大基准，未评估部分无代码模型，生成器多样性不足导致对未知生成器的鲁棒性不高，且审核流程依赖预定义的六项控制，未来可能被绕过；

---

## 174. Online TT-ALS for Streaming Tensor Decomposition with Incremental Orthogonalization

**arXiv ID:** 2606.31061 | [PDF](https://arxiv.org/pdf/2606.31061v1)

**作者:** Hiroki Takeda `[一作]` (University of Osaka), Daisuke Furihata `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `aaccfe5c-6b26-4208-b23c-35331481e142` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在线 Tensor Train 分解算法 Online TT-ALS，利用逐步正交化实现单次遍历精确更新核心张量，从而实现低延迟、高精度的视频流压缩和追踪。

**💡 创新点**

在在线 TT 分解中首次严格执行正交约束，将局部最小化问题转化为无矩阵求逆的解析解，降低秩依赖从 O(r²) 到 O(r)，并实现单扫更新与即时收敛。

**🔧 技术方法**

使用 Tensor Train 格式、交替最小二乘 (ALS)、QR/LQ 正交化、矩阵压缩与乘法、单步更新公式以及理论证明（单调下降、时间平滑）与复杂度分析。

**📊 数据集**

采用合成高阶张量（n=5,6,7，I=30）以及 CDnet 2014 benchmark 中的 office、pedestrians（灰度/彩色）和 continuousPan（动态背景）视频。

**📈 对比分析**

与批量 TT-ALS、TT-ALS Slice、TT-ALS Batch、TT-FOA 以及深度学习 OFTD 进行对比；在精度（RE、PSNR、SSIM、VMAF）上优于或相当，并将计算延迟从秒级降低到毫秒级，实现 10³–10⁴ 倍的速度提升，且在高阶张量上无 OOM 问题。

**⚠️ 局限性**

仍需预设固定 TT 秩，缺乏自适应秩调整机制；仅在沿单一模式流式数据上验证，尚未处理缺失值插补或多模式增量场景。

---

## 175. Fleet: Few Shots Lead Effective AI-generated Image Detection

**arXiv ID:** 2606.31082 | [PDF](https://arxiv.org/pdf/2606.31082v1)

**作者:** Jiaan Wang `[一作]` (Chinese Academy of Sciences), Sheng Tang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Fleet 框架与 Treasure 基准，用于 AI 生成图像的动态少样本适应检测。

**💡 创新点**

创新点在于从静态通用特征转向动态子空间路由，利用高频引导的避免路由实现快速适应，并构建覆盖广泛的 Treasure 基准。

**🔧 技术方法**

采用 DINOv3+LoRA 作为 backbone，结合频域高通滤波、子空间路由、正交约束、覆盖约束、避免路由和蒸馏等技术。

**📊 数据集**

使用 Treasure（64 个生成模型、360k 图像，含 20 个闭源引擎）以及 AIGIBench 等公开数据集。

**📈 对比分析**

与现有零样本与少样本方法比较，Fleet 在 1-shot 下达 76%+，10-shot 达 88%+ 的检测准确率，显著优于 SOTA，并几乎无灾难性遗忘。

**⚠️ 局限性**

限制包括仅针对全图生成、未覆盖局部编辑/面部交换；只实现单步少样本适应，连续适应仍待研究；回放集带来存储开销。

---

## 176. Reference-Based Prosody and Rhythm Evaluation for Spoken Dialogue Systems

**arXiv ID:** 2606.31055 | [PDF](https://arxiv.org/pdf/2606.31055v1)

**作者:** Ashish Hallur `[一作]` (Johns Hopkins University), Laureano Moro-Velazquez `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了大规模英语双人对话语料库的基准性语调与节奏评估框架，并提供了按说话者特征与交互状态匹配的参考分布；

**💡 创新点**

创新在于将F_0均值、表达度（SD与范围）、说话率、发音率、停顿比例及平均停顿时长等多维度指标与说话者性别、年龄、情绪状态等因素匹配，推出分位数基准评估协议；

**🔧 技术方法**

使用Praat+Parselmouth进行F_0提取、VAD与时间对齐，结合Vox-Profile的年龄/性别预测与情绪维度模型，对波形进行特征提取；

**📊 数据集**

数据来源于Seamless Interaction大规模面向面语料，包含超过4000小时双人对话；

**📈 对比分析**

通过与人类参考数据的5th–95th分位数对齐，评估S2S输出的百分位偏差；相较于聚合参考，匹配参考可将异常率从约30%降至≈10%，提高评价的校准性和可解释性；

**⚠️ 局限性**

局限性包括仅基于单一英语双人语料、缺乏真实说话者特征标注、未与主观自然度或交互质量评分对应、仅考虑二元性别标签、对跨语言和不同录音环境的适用性未知。

---

## 177. ADAPT: Attention Dynamics Alignment with Preference Tuning for Faithful MLLMs

**arXiv ID:** 2606.31054 | [PDF](https://arxiv.org/pdf/2606.31054v1)

**作者:** Zhiyuan Yao `[一作]` (University of Science and Technology of China), Zhendong Mao `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ADAPT 框架，通过提取并修正文本-图像交叉注意力锚点、在线注意力监督以及视觉注意引导的 DPO，有效降低 MLLM 的幻觉生成。

**💡 创新点**

创新点包括：① 早期交叉注意力锚点的多指标精炼与位置偏差校正；② 以锚点为依据的稀疏注意力监督机制；③ 结合视觉与噪声条件的 VAG‑DPO，强化视觉根植的生成偏好。

**🔧 技术方法**

采用交叉注意力多层融合、频域高频能量、总变差平滑、熵一致性等多标准评分；利用注意力集中度指标 AMC 进行在线修正；在训练阶段使用 VAG‑DPO 对比优化。

**📊 数据集**

使用 AMBER、MM‑Hal、Obj‑Hal、POPE 等幻觉基准；RefCOCOg/RefCOCO 进行定位评估；MME 与 MMBench 测试通用多模能力。

**📈 对比分析**

与 VCD、OPERA、GF‑SCD、RLHF‑V、mDPO、SIMA 等方法对比，ADAPT 在所有幻觉指标上实现 40–60% 的降幅，成为多种主流 MLLM 的 SOTA，且仅增加约 1.4× 的推理开销。

**⚠️ 局限性**

局部对通用能力略有下降（MME/MBench 分数略降），且需要额外的前向推理步骤生成锚点，推理效率和硬件成本略高。

---

## 178. Warp RL: Reshaping Base Policy Distributions for Dynamics Adaptation

**arXiv ID:** 2606.31043 | [PDF](https://arxiv.org/pdf/2606.31043v1)

**作者:** Ethan Hirschowitz `[一作]` (University of Sydney), Fabio Ramos `[通讯]` (University of Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在已有机器人策略基础上，通过学习可逆的、状态条件的变换来适应动态变化的任务。

**💡 创新点**

提出 Warp RL，将加性残差扩展为可逆流（Rational Quadratic Spline），实现对动作分布的形状、尺度、非均匀校正，克服了残差方法的限制。

**🔧 技术方法**

使用可逆流（RQ‑spline）、政策梯度（PPO）、离策略（SAC）以及进化策略（ES）等技术，在冻结的基础策略上做轻量级的适配。

**📊 数据集**

在 ManiSkill3 的四个操控任务（PushCube、PushT、LiftPegUpright、PegInsertionSide）以及真实 SO101 机器人插销任务上进行实验。

**📈 对比分析**

与传统残差改正（Residual‑PPO）以及其他基线比较，Warp‑ES 在所有任务上实现更高的成功率（如 LiftPegUpright 98.5% 对比 87.3%），在真实机器人上保持相似成功率但循环时间缩短约 30%。

**⚠️ 局限性**

局限包括对动作域边界映射的敏感、仅进行逐维单步变换、未建模跨维或时序相关性、真实实验样本量有限。

---

## 179. GenPage: Towards End-to-End Generative Homepage Construction at Netflix

**arXiv ID:** 2606.31031 | [PDF](https://arxiv.org/pdf/2606.31031v1)

**作者:** Lequn Wang `[一作]` (Netflix), Linas Baltrunas `[通讯]` (Netflix)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出GenPage，一种端到端生成Netflix首页的Transformer模型，直接将用户上下文作为prompt，生成多行布局的首页；

**💡 创新点**

创新点在于：将传统多阶段推荐堆栈压缩为单一生成模型；使用预训练+加权二分类或强化学习进行后训练；通过自定义token化、上下文注入、语义嵌入融合、增量训练、受限解码和混合行解码等技术实现行业级性能；

**🔧 技术方法**

使用了大型Transformer架构、预训练的next‑token预测、加权二分类（WBC）和强化学习（RL）后训练；自定义token化与embedding融合；KL约束、GRPO算法；vLLM推理框架；

**📊 数据集**

训练数据来自Netflix实际首页展示的用户交互记录（用户历史、配置、请求上下文、页面布局及反馈），内部奖励系统计算实体级奖励；

**📈 对比分析**

与现有多阶段生产推荐系统做在线A/B对比，200M参数的WBC模型在核心用户参与度指标上提升0.24%（p<0.001），且端到端服务延迟降低20%；离线实验显示预训练提升AUC，上下文丰富比模型规模扩大更显著；

**⚠️ 局限性**

局限包括：RL训练尚未上线；长上下文仍需手工摘要；仅针对首页布局，未覆盖多模态、语言、推理能力；模型对业务规则的依赖仍需改进；

---

## 180. Labimus: A Simulation and Benchmark for Humanoid Dexterous Manipulation in Chemical Laboratory

**arXiv ID:** 2606.31037 | [PDF](https://arxiv.org/pdf/2606.31037v1)

**作者:** Yuhan Wu `[一作]` (University Of Science And Technology Of China), Yan Xia `[通讯]` (University Of Science And Technology Of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个面向精准化学实验的类人机器人灵巧操作基准，涵盖30+实验资产、粒子级粉末物理、仪器闭环读数，并提出三层精度评估协议；

**💡 创新点**

1) 真实工位到仿真重建；2) 粒子级粉末物理与仪器实时读数；3) 量化精度与长期执行的评估框架；

**🔧 技术方法**

基于 Isaac Sim 的物理仿真、PhysX 粒子求解、LLM 辅助的 SOP-到-仿真管线、Tianyi 双臂类人机器人与 Manus 手势远程控制；

**📊 数据集**

来自真实有机化学实验台的3D资产（30+），以及 100 条手动远程演示数据；

**📈 对比分析**

与 ACT、Diffusion Policy、π_0 三种基线进行比较，显示任务完成率与精度存在明显差距；在门开关等单步任务上 30–55% 成功率，而在精度层（如粉末质量 ±0.001g）仅 3–5%，表明当前方法尚未满足实验规范；

**⚠️ 局限性**

仅覆盖单一实验流程（固体称重），未完成多步骤流程评估；仅在仿真中验证，缺乏真实世界转移；支持单一类人平台，未覆盖多种机器人与手掌配置；

---

## 181. OTCache: Optimal Transport for Geometry-Aware Caching in Diffusion Models

**arXiv ID:** 2606.31026 | [PDF](https://arxiv.org/pdf/2606.31026v1)

**作者:** Huanlin Gao `[一作]` (China Unicom), Shiguo Lian `[通讯]` (China Unicom)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出OTCache框架，通过训练无关的缓存调度预测实现扩散模型采样加速；

**💡 创新点**

创新点在于将不同NFE预算的缓存调度视为连续轨迹，用Optimal Transport插值预测目标预算的调度，并结合高预算参考和低预算anchor搜索；

**🔧 技术方法**

采用图形缓存方法（MeanCache）做参考、Optuna+CMA‑ES搜索anchor、PCHIP连续插值、Wasserstein量化插值、power‑law warping，并以LPIPS感知目标进行端到端优化；

**📊 数据集**

在FLUX.1[dev]、Qwen‑Image和HunyuanVideo三大模型上，使用T2V‑CompBench、DrawBench和VBench等数据集；

**📈 对比分析**

与TaylorSeer、DBCache、DiCache、TeaCache、LeMiCa、MeanCache等多种缓存方法对比，OTCache在FLUX.1实现4.5×加速（LPIPS 0.126）、在Qwen‑Image实现4.7×加速（LPIPS 0.171）、在HunyuanVideo实现3.66×加速（LPIPS 0.252），显著提升速度与质量；

**⚠️ 局限性**

局限性在于对极低NFE（≤4）时anchor搜索仍需较多试验；依赖两端预算的可靠性；不易推广到需要更低NFE或不同模型结构的场景；缺乏严格理论证明。

---

## 182. WarpI2I: Image Warping for Image-to-Image Translation

**arXiv ID:** 2606.31018 | [PDF](https://arxiv.org/pdf/2606.31018v1)

**作者:** Shen Zheng `[一作]` (Carnegie Mellon University), Srinivasa Narasimhan `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过显著性引导的图像空间拉伸与反拉伸（warp–unwarp）框架，在不修改模型结构的前提下提升Latent Diffusion模型在图像到图像转换中的细节保留与结构一致性。

**💡 创新点**

创新点包括：①无架构改动、模型无关的空间重分配；②利用显著性引导实现局部放大；③提供简单高效的合成配对数据生成管线，极大降低数据准备成本。

**🔧 技术方法**

核心技术包括显著性映射+图像空间拉伸/反拉伸、FLUX文本‑图像生成、Depth Anything深度估计、Background Prompt Substitution、LoRA微调。

**📊 数据集**

使用的数据集涵盖人像 relighting（VITON‑HD、StreetTryOn）、驾驶场景 relighting（ROADWork）、天气/时段翻译（BDD100K、Cityscapes、DarkZurich、ACDC）等。

**📈 对比分析**

与IC‑Light、DreamLight、pix2pix‑Turbo、CycleGAN‑Turbo等基线对比，用户研究与FID/KID/Clean‑FID/DINO‑Struct等指标均显著提升，说明在细节保留、照明真实性与整体图像质量方面优于现有方法。

**⚠️ 局限性**

局限性：目前仅验证于单步扩散基线，未与多步扩散或更大规模模型结合；在高速运动的视频帧间一致性仍可能出现闪烁，需进一步引入光流或时序约束。

---

## 183. Beyond Wireless Security: Covert Communications in Large Language Model-enabled Edge Networks

**arXiv ID:** 2606.31016 | [PDF](https://arxiv.org/pdf/2606.31016v1)

**作者:** Yuanai Xie `[一作]` (South-Central Minzu University), Zhaozhi Liu `[通讯]` (South-Central Minzu University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在大语言模型边缘网络（LLMEN）中采用隐蔽通信与计算技术，既隐藏无线传输又掩盖计算过程，从而增强对窃听、干扰、注入等攻击的防护，并提出联合优化传输功率与CPU频率以最小化总时延的方法。

**💡 创新点**

① 将隐蔽通信扩展到计算域，提出“主动伪装”式虚假工作负载；② 结合背景噪声、多天线波束成形、硬件屏蔽等多种辅助隐蔽技术，形成双域覆盖；③ 通过凸优化与双分解求解联合功率/频率调度，实现安全与效率的平衡。

**🔧 技术方法**

隐蔽通信原理、无线功率调度、CPU频率动态管理、虚假矩阵乘法伪装、天线波束成形、多天线技术、能量预算约束、凸优化与双分解、子梯度算法。

**📊 数据集**

本文以理论任务规模为主（LLM推理任务10^10比特，结果体积C=20），在单用户场景下进行数值仿真，未使用公开数据集。

**📈 对比分析**

通过数值仿真比较不同安全阈值ε、非隐私/隐私令牌比例α以及结果体积/传输速率比值对总时延的影响，展示在满足隐蔽约束下该框架能显著降低时延，并在资源上限时出现性能饱和。

**⚠️ 局限性**

仅针对单用户情况，未考虑多用户资源冲突与非凸优化；混合专家/模型压缩导致的动态激活模式尚未纳入；细粒度电磁侧信道的进一步分析缺失。

---

## 184. Usage frequency and application variety of research methods in library and information science: Continuous investigation from 1991 to 2021

**arXiv ID:** 2606.31081 | [PDF](https://arxiv.org/pdf/2606.31081v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 185. Evaluating Interactivity: Toward Automated Assessment of AI-Generated Explorable Explanations

**arXiv ID:** 2606.31012 | [PDF](https://arxiv.org/pdf/2606.31012v1)

**作者:** Xiaozao Wang `[一作]` (New York University Shanghai), Hongyi Wen `[通讯]` (New York University Shanghai)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出EE‑Eval框架，利用有限状态机自动评估AI生成的可探究解释的交互质量。

**💡 创新点**

创新点在于将交互逻辑显式化为可计算的FSM，并设计结构与语义相结合的相似度指标。

**🔧 技术方法**

采用LLM提取FSM、MiniLM句向量、图相似度、FSM同构等技术实现自动评估。

**📊 数据集**

使用超过2500个AI生成的可探究解释，涵盖127个计算机科学概念，来自六个LLM模型。

**📈 对比分析**

与人工评分、基于单元测试和视觉质量的基线对比，EE‑Eval在交互性指标上与人工评分相关性最高（r≈0.73），并能明显区分模型间差异。

**⚠️ 局限性**

限制包括FSM抽象可能无法捕捉连续交互、提取过程对LLM准确率敏感、理想FSM构建带主观性、未评估内容准确性或学习效果。

---

## 186. Dense Structural Priors for Sparse Functional Landmark Localization in Surgical Videos

**arXiv ID:** 2606.31007 | [PDF](https://arxiv.org/pdf/2606.31007v1)

**作者:** Chenyan Jing `[一作]` (Johns Hopkins University), Mathias Unberath `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种利用零样本视觉基础模型 SAM 3 提供的稠密结构先验，配合轻量级细化网络，实现对外科视频中稀疏功能性关键点（工具尖端和锚点）的精准定位。

**💡 创新点**

创新点在于：①将 SAM 3 的结构先验作为中间细化引导，而非直接将掩膜作为目标；②引入辅助掩膜分支以在特征层保留结构信息；③通过粗细化的预测-引导方式，使结构先验在不依赖像素级标注的前提下提升定位性能。

**🔧 技术方法**

使用的技术包括：SAM 3 零样本点提示掩膜生成；基于 MFCNet 的多帧热图定位骨干网络；轻量化任务适配器与细化网络；辅助掩膜分支；多种损失函数（热图损失、掩膜损失、门控回归损失）。

**📊 数据集**

数据集为由 60 条外科视频（YouTube、Cholec80、HeiChole、SurgVU、CRCD）收集的 7,867 个剪辑，包含工具尖端和锚点标注，按视频级划分训练/验证/测试。

**📈 对比分析**

与 SimpleBaseline、YOLOv8-pose、RTMPose 等基线对比，本文方法在整体尖端 F1 率达到 72.4%，锚点 F1 率为 58.0%，显著优于基线；在不同动作（剪切、抓取、分离）下表现稳定，且平均定位误差在 8.6–9.8 像素之间。

**⚠️ 局限性**

局限性包括：①对多工具场景的掩膜先验可能与当前动作不完全语义对齐，导致细化效果受限；②依赖 SAM 3 的结构质量，若工具形态复杂或遮挡严重，掩膜生成可能不可靠；③在极端光照或显著模糊的条件下，性能仍有提升空间。

---

## 187. DDIAgents: Mechanism-Conditioned Context Flow for Drug-Drug Interaction Prediction

**arXiv ID:** 2606.31085 | [PDF](https://arxiv.org/pdf/2606.31085v1)

**作者:** Zhenqian Shen `[一作]` (Tsinghua University), Quanming Yao `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 DDIAgents，一种机制条件化的多智能体框架，用动态上下文流将异构药物相互作用（DDI）证据路由给专门的专家智能体，以实现更准确、可解释的 DDI 预测。

**💡 创新点**

创新点在于将 DDI 预测视为知识协同问题，动态构建专家队伍并根据交互机制实时分配相关知识源，打破传统模型的静态知识融合与单一推理模式，显著提升稀有交互类型的预测性能。

**🔧 技术方法**

利用大型语言模型（如 Qwen2.5、Llama3.1、Gemma2、ChatGLM3）作为智能体核心，配合专家智能体实例化、动态上下文流和迭代反馈循环实现推理；核心技术包括多智能体协作、机制感知知识路由与迭代决策。

**📊 数据集**

使用公开的 DrugBank（多分类）和 TWOSIDES（多标签）两大 DDI 预测数据集，并按照药物批准时间划分为 S0、S1、S2 任务。

**📈 对比分析**

与特征基、图基、LLM 基以及现有多智能体方法（Reflexion、Debate、AgentVerse、MDAgents）进行对比，DDIAgents 在 S1/S2 任务上分别在 Accuracy/F1、Hit@5/NDCG@5 指标上领先，尤其在长尾交互类型上表现突出；实验显示动态上下文流和迭代机制对性能提升至关重要。

**⚠️ 局限性**

局限性包括：当前采用固定的智能体协作协议，可能限制对不同 DDI 查询的适应性；动态上下文流仅覆盖有限的知识源类型，可能遗漏关键临床证据；模型对大型 LLM 依赖度高，资源消耗大。

---

## 188. Triospect: A Three-Dimensional Framework for Robust Statistical AI-Generated Text Detection Against Diverse Attacks

**arXiv ID:** 2606.31074 | [PDF](https://arxiv.org/pdf/2606.31074v1)

**作者:** Guangsheng Bao `[一作]` (Zhejiang University), Yue Zhang `[通讯]` (Westlake University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Triospect 检测框架，通过对文本进行内容与表达层面的拆分与测量，提升对 AI 生成文本的检测鲁棒性。

**💡 创新点**

创新点在于利用 LLM 文本转换技术，分别保持文本的内容与表达，以三维（原文、内容、表达）统计特征实现攻击抵抗。

**🔧 技术方法**

主要技术包括 LLM 生成的内容/表达保持变换、现有检测指标（如 Fast-Detect、Binoculars）测量、以及基于高斯混合模型的贝叶斯分类。

**📊 数据集**

使用了自构建的 Humanize-16K 语料（含 6 种人化攻击、4 个领域、6 个 LLM 模型）以及现有 RAID 基准，并扩展到多语言补丁。

**📈 对比分析**

与传统单维度检测器相比，Triospect 在 Humanize-16K 上 AUROC 提升 22.3%/9.1%，TPR01 提升 13%/22%，在 RAID 上亦显著提升，证明其在多种攻击和领域上的优越性能。

**⚠️ 局限性**

主要局限包括：文本转换对内容与表达的分离并非完美，且需要额外的计算资源（多次 LLM 生成与评分），在实时大规模部署上可能面临效率瓶颈。

---

## 189. MultiUAV-Plat: An LLM-Oriented Platform, Benchmark and Framework for Multi-UAV Collaborative Task Planning

**arXiv ID:** 2606.31073 | [PDF](https://arxiv.org/pdf/2606.31073v1)

**作者:** Sheng Zhang `[一作]` (National University of Defense Technology), Cheng Zhu `[通讯]` (National University of Defense Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 MultiUAV-Plat 这一面向 LLM 的轻量级多无人机协同任务规划仿真平台，并发布了包含 75 个任务会话、1500 个自然语言任务、9396 条隐藏验证检查的 MultiUAV-Plat Benchmark；同时提出了 Agent4Drone 框架，用于在此平台上实现闭环、多机协同、局部感知的 LLM 代理。

**💡 创新点**

创新点主要包括：①首次为 LLM 代理提供完整的、受限接口的多UAV仿真环境，支持局部感知、工具调用和隐藏验证；②设计了专门针对多机协同任务的 Agent4Drone 结构化工作流，将观察、任务理解、记忆、规划、执行和验证模块相结合；③在同一平台上系统性对比 LLM 代理与 ReAct 基线，验证了基于 Agent4Drone 的显著性能提升。

**🔧 技术方法**

技术手段涵盖：LLM 大模型（doubao‑2‑pro、qwen3.5、deepseek‑v4‑pro 等），RESTful API 与工具调用机制，分层平台架构（模型层、控制层、服务层、权限管理、可视化）；Agent4Drone 采用基于黑板的记忆、短周期规划与检验循环；Benchmark 采用自然语言任务生成、隐藏验证器和多难度级别设置。

**📊 数据集**

数据集：MultiUAV‑Plat Benchmark，包含 75 个任务会话（每个会话 20 个任务）、1500 个自然语言任务、9396 条隐藏验证检查，覆盖目标分配、区域搜索、区域分配与巡逻三大场景，设定 5 个难度等级（Easy–Extreme）。

**📈 对比分析**

实验对比方法：在同一 AGENT 角色、相同的观察/操作/验证接口下，使用 ReAct 作为通用基线；对比 Agent4Drone 与 ReAct 在 doubao‑2‑pro、qwen3.5、deepseek‑v4‑pro 等后端下的任务通过率、平均检查通过率、全局检查通过率及失误率。结果显示，Agent4Drone 在 doubao‑2‑pro 上将任务通过率提升 27.3pp（从 30.6% 到 57.9%），平均检查通过率提升 26.7pp；在 deepseek‑v4‑pro 后端，任务通过率达 70% 以上，明显优于 ReAct。

**⚠️ 局限性**

局限性：①对目标定位与最终条件验证的鲁棒性仍不足，部分多机协同任务仍出现未满足验证的情况；②平台目前采用简化动力学与感知模型，缺乏更真实的飞行动态与噪声；③性能高度依赖强大 LLM 后端，低能力模型下仍显著受限；④对隐藏验证器的设计和复杂度仍有改进空间。

---

## 190. Hierarchical 3D Scene Graph Construction and Belief-based Planning for Semantic Navigation

**arXiv ID:** 2606.31071 | [PDF](https://arxiv.org/pdf/2606.31071v1)

**作者:** Bing Wu `[一作]` (Hong Kong Polytechnic University), Changwen Chen `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种在线层级3D场景图构建与基于信念的规划框架，用于零样本语义导航。

**💡 创新点**

创新点包括①在线增量构建对象-区域-区域层级场景图；②将LLM语义先验与前沿信息增益融合形成层级信念；③在场景图模拟器上进行有限视界滚动评估宏观动作，显著提升长距离导航效率。

**🔧 技术方法**

技术手段包括GVD+谱聚类构建层级图、Open‑Vocabulary VLM/LLM（Qwen3‑VL‑8B）提取语义、POUCT搜索+HSG模拟器进行长视界规划、前沿信息增益以及Fast Marching本地路径规划。

**📊 数据集**

使用的数据集包括Matterport3D、HM3D‑Semantics v0.1/v0.2、Habitat Synthetic Scenes以及InstanceNav等。

**📈 对比分析**

与ApexNav、UniGoal、VLFM等训练‑free基线对比，平均SR/SPL在长距离任务上提升9.4%/5.0%，整体在四大数据集上SR/SPL分别提升5.4%/2.3%，超过现有SOTA。

**⚠️ 局限性**

局限性在于仅在仿真环境验证、需已知姿态；VLM/LLM推理耗时导致无法实时执行；对定位误差鲁棒性不足。

---

## 191. Diffusion-Based Material Regularization for Physics-Based Inverse Rendering

**arXiv ID:** 2606.31065 | [PDF](https://arxiv.org/pdf/2606.31065v1)

**作者:** Jingwang Ling `[一作]` (University of Illinois Urbana Champaign), Shuang Zhao `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建端到端管道，利用扩散模型预测的材质信息作为相似性核，对物理逆渲染过程进行材料正则化，从而同时恢复几何、材质和光照。

**💡 创新点**

将扩散模型输出作为材料相似性引导而非直接目标，在逆渲染中实现隐式材料聚类正则化；同时引入无尺度 albedo 变换以避免光照与材质混淆。

**🔧 技术方法**

扩散渲染（DiffusionRenderer）预测 G‑buffer；神经体渲染重建 SDF；Mitsuba 3 可微渲染；联合双边滤波与可微正则化；Dictionary Fields 表示空间变材质与环境光。

**📊 数据集**

Stanford‑ORB、Synthetic4Relight、DTC‑Synthetic 三个多视图重光照基准。

**📈 对比分析**

与 Neural‑PBIR 与 MaterialFusion 等基线在相同数据集上对比，PSNR/SSIM/LPIPS 及 relighting 结果均优于对手，显著降低烘焙阴影并提升材质重建精度。

**⚠️ 局限性**

对扩散模型预测的局部一致性要求高；低分辨率预测限制高频细节；在分布外外观或光照极端情况下正则化可能过度平滑并残留阴影边界。

---

## 192. LabGuard: Grounding Natural-Language Laboratory Rules into Runtime Guards for Embodied Laboratory Agents

**arXiv ID:** 2606.31045 | [PDF](https://arxiv.org/pdf/2606.31045v1)

**作者:** Jingpu Yang `[一作]` (Wuhan University), Zhuohan Xie `[通讯]` (MBZUAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发并评估了 LabGuard，一个将实验室安全规则从自然语言转化为可执行运行时监视器的完整系统，涵盖规则表征、监督标注、语言到IR的映射、监视器编译以及在仿真环境中的实时执行。

**💡 创新点**

创新点在于提出了从实验室安全文本到可执行监视器的端到端管道（LabGuard-IR、LabGuard-Bench、LabGuard-Grounder 与编译器），并验证了该管道在保持任务成功率的同时显著降低不安全事件，首次实现了可审计、可部署的安全层。

**🔧 技术方法**

主要技术包括基于大型语言模型的序列生成（LoRA 微调与规则归一化）、可执行IR设计、监视器编译器、控制边界安全过滤器（状态检查、动作过滤、风险评分），以及在 LabUtopia 仿真平台与 ACT 控制器的集成。

**📊 数据集**

使用了 LabGuard-Bench 数据集，该数据集包含 203 条实验室安全规则（来自安全手册、PubChem GHS、专家规则、化学安全参考），并通过改写扩展至 812 条标注，支持源端留空、化学端留空等评估场景。

**📈 对比分析**

在语言映射任务上与正则表达式、提示LLM、LoRA 微调、混合方法等基线比较；混合方法在源端留空时达 79.4% 任务范围 F1，在化学端留空时达 91.8%；编译后监视器将不安全事件率从 39.5% 降至 23.8%，干预率低于 0.5%，任务成功率提升至 81.2%。

**⚠️ 局限性**

局限性主要在于仅在仿真环境中验证，未在真实机器人与化学实验中测试；对不同 VLA 控制器的适用性未评估；监视器库的覆盖面受限，某些安全情形超出当前 schema。

---

## 193. TerraDiT-$Ω$: Unified Spatial Control for Satellite Image Synthesis with Any Geospatial Primitive

**arXiv ID:** 2606.31029 | [PDF](https://arxiv.org/pdf/2606.31029v1)

**作者:** Brian Wei `[一作]` (Washington University in St. Louis), Nathan Jacobs `[通讯]` (Washington University in St. Louis)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种统一的卫星图像生成框架，能够直接以任何本地地理空间原语（多边形、折线、边框、点）为条件生成高保真遥感图像，并在此基础上实现多任务数据增强。

**💡 创新点**

核心创新在于Geometry-Aware Local Attention（GALA）机制，结合旋转可学习的高斯先验和空间几何场（SGF）实现对不同复杂度原语的自适应注入；以及统一原语编码器，使模型无需格式转换即可兼容从稀疏到密集的多级注释。

**🔧 技术方法**

采用流基扩散变压器（TerraDiT）作为骨干，加入文本交叉注意力、地理位置归一化、MetaRBF+模块预测旋转高斯先验，使用空间几何场进行稠密调制，并在训练中对齐表示、利用LongCLIP编码文本。

**📊 数据集**

使用了 Git-10M（全球卫星图像与文本+地理信息）构建训练集，结合 OpenStreetMap 的矢量几何作为原语；在评估中采用 Git-Rand-15k、Git-Spatial-15k、Git-Dense-3.5k；在下游任务上使用 OpenEarthMap、DIOR、City-Scale、AID 进行数据增强验证。

**📈 对比分析**

与通用图像生成模型（SDXL、PixArt）以及先前的地理空间控制模型（GeoSynth、VectorSynth、GeoSynth-OSM、InstanceDiffusion、GLIGEN 等）进行对比；在所有指标（FID、sFID、LPIPS、CLIP、CAS、下游任务 mAP/mIoU/TOPO/Acc）上均取得显著提升，尤其在高密度场景下的生成质量和空间精度表现最为突出。

**⚠️ 局限性**

目前仅基于 OSM 标注的粗粒度属性，缺乏细粒度视觉描述；对实例级视觉属性的控制仍受限；生成高保真图像存在被滥用风险，需要更细化的属性标签和安全监管。

---

## 194. WaterGen: Decoupling Scene and Medium in Underwater Image Generation

**arXiv ID:** 2606.31147 | [PDF](https://arxiv.org/pdf/2606.31147v1)

**作者:** Jiayi Wu `[一作]` (University of Maryland), Yiannis Aloimonos `[通讯]` (University of Maryland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出WaterGen框架，分两阶段分离场景生成与水介质控制，生成可控的真实感海水图像。

**💡 创新点**

首次在潜在扩散模型中实现场景与水介质的完全解耦，并通过物理一致性损失实现精确的光学散射与吸收控制。

**🔧 技术方法**

使用低秩适配器LoRA对SDXL潜在扩散模型进行微调，配合物理基水成像模型、双向UIFM一致性以及噪声注入训练策略。

**📊 数据集**

利用SLURPP恢复后的无介质海水图像与BLIP2生成的文本描述，以及通过物理模型合成的陆地图像和随机采样的水介质参数构成训练集。

**📈 对比分析**

在UIQM、MUSIQ和CLIP得分上与Atlantis、TIDE对比，WaterGen在无介质生成阶段分别达UIQM 3.02、MUSIQ 69.26、CLIP 0.261，并在恢复与分割任务中提升约5% mIoU。

**⚠️ 局限性**

对极端浑浊或非标准光源条件的泛化仍有限，且模型对恢复数据质量高度依赖，影响生成真实感的边界。

---

## 195. Beyond the Library: An Agentic Framework for Autoformalizing Research Mathematics

**arXiv ID:** 2606.31134 | [PDF](https://arxiv.org/pdf/2606.31134v1)

**作者:** Arshia Soltani Moakhar `[一作]` (University of Maryland), MohammadTaghi Hajiaghayi `[通讯]` (University of Maryland)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种多代理的自动形式化框架，旨在将复杂的自然语言数学翻译为可验证的Lean 4代码。

**💡 创新点**

创新点在于采用了“类型优先”的分解策略，并通过辅助引理单元测试验证定义的正确性，同时引入了动态的多代理管道管理。

**🔧 技术方法**

使用了大型语言模型（LLMs）和Lean 4作为形式化语言，结合了多代理系统和动态回溯机制。

**📊 数据集**

使用了PutnamBench数据集和五篇来自ACM理论计算机科学研讨会（STOC）的论文进行评估。

**📈 对比分析**

与现有方法相比，该系统在PutnamBench上实现了91.3%的准确率，且每个问题的平均操作成本约为5美元，显著低于其他方法的成本。

**⚠️ 局限性**

限制在于系统对复杂定理的处理仍然依赖于现有的数学库，且在某些情况下可能无法完全自动化处理所有类型的数学概念。

---

## 196. ELASTIC: Efficiently Learning to Adaptively Scale Test-Time Compute for Generative Control Policies

**arXiv ID:** 2606.31132 | [PDF](https://arxiv.org/pdf/2606.31132v1)

**作者:** Andrew Zou Li `[一作]` (Carnegie Mellon University), Andrea Bajcsy `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于元-MDP与强化学习的自适应计算分配框架，能够在测试时动态决定对生成控制策略的顺序扩展与并行采样的计算量；

**💡 创新点**

创新点在于同时考虑顺序与并行两条计算轴，并通过元策略学习状态相关的最优分配，实现对测试时计算与性能的精细权衡；

**🔧 技术方法**

核心技术包括Soft Actor-Critic强化学习、关注注意力机制的顺序/并行决策网络、FiLM条件化、以及对终端值函数的环境级Q估计；

**📊 数据集**

使用了多种模拟基准（PushT、Square PH/MH、Can Paired/Reverse、LIBERO-10）以及真实机器人抓放任务（Franka Emika与π_0.5 VLA模型）；

**📈 对比分析**

与固定或单轴计算分配基线相比，元策略在匹配相同计算预算下成功率均更高；在VLA模型上可获得相同甚至更高的成功率，且推理延迟降低约34%；

**⚠️ 局限性**

局限性包括对验证器（verifier）质量的高度依赖、元策略需针对每个任务单独训练、以及在多任务场景中泛化能力尚未充分验证。

---

## 197. UniSAE: Unified Speech Attribute Editing on Speaker, Emotion and Low-Level Content via Discrete Phonetic Posteriorgram Modelling

**arXiv ID:** 2606.31128 | [PDF](https://arxiv.org/pdf/2606.31128v1)

**作者:** Chuanbo Zhu `[一作]` (Hong Kong University of Science and Technology), Wei Xue `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 UniSAE 统一语音属性编辑框架，支持从子音素到词级的说话人、情感和内容编辑；

**💡 创新点**

创新点在于引入 Discrete Phonetic PosteriorGram（DPPG）显式拆解语音为音素、发音变体和时长，结合两阶段模型实现多属性可组合编辑，并通过 Manifold Distillation 构建大规模可对齐情感语料；

**🔧 技术方法**

使用的技术包括：k-means 离散化 PPG 为 DPPG、GPT‑2 风格的 Content Transformer 进行自回归内容预测、双属性 GE2E 损失的说话人/情感嵌入器、基于速度参数化的扩散声学解码器及 BigVGAN 语音后处理；

**📊 数据集**

训练数据主要为自建的 UniEditCorpus（约 870k 条 87 位说话人、5 种情感，约 580 小时），以及 LibriTTS‑R 用于 tokenizer 训练；评估在 UniEditCorpus、ESD 及自制 ESDEdit 基准上；

**📈 对比分析**

与现有单属性编辑方法（ZEST、EmoConv‑Diff、VoiceCraft、SSR‑Speech）对比，UniSAE 在说话人/情感相似度、内容保真（CER）和自然度（UTMOS）方面均表现优于或相当，且在多属性联合编辑时性能几乎不降；

**⚠️ 局限性**

限制包括：对未见说话人情感迁移仍有一定下降，且基于离散化的 DPPG 在极端音素变体上可能存在信息损失；未来需提升跨域泛化与更细粒度音素变体建模。

---

## 198. WildProp: Visual Estimation of Wildlife Body Proportions at Scale

**arXiv ID:** 2606.31125 | [PDF](https://arxiv.org/pdf/2606.31125v1)

**作者:** Mustafa Chasmai `[一作]` (University of Massachusetts), Subhransu Maji `[通讯]` (University of Massachusetts)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无训练、基于检索的框架，利用大型开放式图像库（如iNaturalist）估计野生动物体型比例；

**💡 创新点**

创新点在于将形态测量视为检索驱动的对应问题，使用姿态感知检索、基于密集patch的点对匹配、几何一致性过滤以及迭代特征细化，完全无需每种物种的专门训练或标注；

**🔧 技术方法**

核心技术包括Foundation模型（DINOv3、SAM2、Grounding‑DINO、CLIP、Stable Diffusion）用于图像检索与特征匹配，RANSAC进行几何验证，迭代特征平均实现自适应；

**📊 数据集**

使用iNaturalist公开图像集合作为检索语料；在三大形态测量数据集（鸟类与两种两栖类）和多个案例研究（蝴蝶、花卉、鹿等）上进行评估；

**📈 对比分析**

与传统物理测量数据集比较，median相对误差约10‑20%；Ablation表明姿态检索、背景抑制和几何一致性是关键，迭代细化可进一步降低误差；

**⚠️ 局限性**

局限性包括估计精度低于实验室测量，受视角、遮挡、尺度模糊影响；只能得到比例而非绝对长度；对人类中心化、受欢迎物种的偏倚；缺乏自动拒绝机制，易产生不合理的比例。

---

## 199. The Past Is Prologue: A Plug-in Controller for Selective Updates in Sequentially Evolving LLM Memory

**arXiv ID:** 2606.31121 | [PDF](https://arxiv.org/pdf/2606.31121v1)

**作者:** Zihan Chen `[一作]` (University of Virginia), Jundong Li `[通讯]` (University of Virginia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Janus，一种可插拔的内存控制器，决定是否接受 LLM 生成的候选内存更新，从而避免局部有益但整体有害的更新被直接部署。

**💡 创新点**

创新点在于将内存更新视为部署决策，使用内存动量触发器检测显著轨迹偏差并用紧凑的混合评估集（覆盖、边界与最新任务）高效评估候选内存。

**🔧 技术方法**

主要技术包括：内存动量触发器（MMT）、基于文本编码的向量空间表示、覆盖集与边界集的聚类与更新、以及对候选内存与历史内存的准确率比较。

**📊 数据集**

使用六个数据集（MATH500、GPQA、MMLU-Pro（Eng、Phy）、APIBench-HF、HumanEval），在两大 LLM（Qwen3-8B、DeepSeek-V4-Flash）上进行实验。

**📈 对比分析**

与 Memory-free、ExpRAG、DC-RS、ExpeL 等基线相比，Janus 在所有数据集上平均提升 2.7~4.6 个百分点；触发率约 70%，评估成本受控，且在“Always”触发策略下接近最高性能。

**⚠️ 局限性**

局限性：仅针对提示式外部内存更新，未考虑参数或策略学习的自适应；在长时序交互、多智能体或非平稳分布场景中的通用性尚需进一步验证。

---

## 200. Visualizing High-Dimensional Graph Embeddings via Informed Multi-View Projections

**arXiv ID:** 2606.31119 | [PDF](https://arxiv.org/pdf/2606.31119v1)

**作者:** Ya Ji `[一作]` (Northeastern University), Yifan Hu `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出在高维空间嵌入图后，利用梯度优化搜索最优二维投影（视角）以提升图可视化质量，并实现了可交互探索工具；

**💡 创新点**

创新点包括：①提出SigmoidX可微的交叉约束损失，简化梯度下降；②证明高维布局+最优投影可超越传统2D布局在多种美学指标上；③开发基于PCA的可视化探索系统；

**🔧 技术方法**

使用的技术有：高维图嵌入（stress、force-directed、spectral、PivotMDS）、线性投影优化（梯度下降）、SigmoidX损失、PCA、GPU加速渲染；

**📊 数据集**

数据集包括SuiteSparse14（14图，含超大图）和Rome图集（11531个小图）以及示例图如Hypercube、Mobius等；

**📈 对比分析**

与2D基线（stress、force-directed）以及专门优化指标的算法（Vertex Movement、其他深度学习方法）进行对比，结果显示最优投影在边交叉、角分辨率、边长方差、t‑SNE等指标上分别提升约20‑30%；

**⚠️ 局限性**

局限性在于仅支持线性投影、最高10维布局、梯度优化局部收敛、未对非线性降维方法做交互探索，以及对超大图的可扩展性尚待进一步验证。

---

## 201. Revealing Safety-Critical Scenarios for UTM via Transformer

**arXiv ID:** 2606.31114 | [PDF](https://arxiv.org/pdf/2606.31114v1)

**作者:** Huaze Tang `[一作]` (Tsinghua University), Wenbo Ding `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

做了一个基于Transformer的策略模型和动作采样器，用于UTM系统的弱点发现

**💡 创新点**

创新点是把UTM弱点发现视作序列建模，结合离线强化学习、偏好偏置采样和安全约束的动作采样器

**🔧 技术方法**

使用了Transformer（Decision Transformer）架构、离线强化学习、Top‑K+偏好偏置采样及注意力机制

**📊 数据集**

使用了约17B token的压力测试数据集，涵盖城乡不同区域的700小时仿真记录

**📈 对比分析**

与专家指导和烟雾测试比较，PM‑2B在未见区域发现效率约提升8倍，检出专家方法遗漏的关键边缘案例

**⚠️ 局限性**

局限性包括对不同动作类型分布不均导致低频动作可能未充分探索，以及对真实物理约束的依赖

---

## 202. What Counts as an Error? Dual-Reference Benchmarking for Atypical ASR

**arXiv ID:** 2606.31112 | [PDF](https://arxiv.org/pdf/2606.31112v1)

**作者:** Hawau Olamide Toyin `[一作]` (MBZUAI), Hanan Aldarmaki `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 11 种 ASR 模型在典型与非典型发音（含口吃）数据上，使用两种合法的参考文本（verbatim 与 intended）进行系统性评测与排名比较。

**💡 创新点**

首次揭示 ASR 模型在不同参考文本下的排名不稳定性，并指出模型架构（seq2seq vs CTC）对转录类型的天然偏好，从而强调评价指标必须与实际应用场景相匹配。

**🔧 技术方法**

使用多种 ASR 体系结构（seq2seq、CTC、Transducer）、WER、SeMaScore、BERTScore 等评估指标，以及与 FluencyBank 与 CASA 标注对齐的语音分段技术。

**📊 数据集**

FluencyBank Timestamped（含 3430 条 verbatim 与 intended 转录）与 CASA 临床停顿事件标签，共同构成评测数据集。

**📈 对比分析**

按 WER 进行排名对比，发现 seq2seq 模型在 intended 评测中更优，CTC 模型在 verbatim 评测中更优，排名差异显著，表明评测需根据用例选择合适参考。

**⚠️ 局限性**

研究仅涵盖英语单一语料与语境，缺乏多语言、多说话场景的验证，且未对模型进行微调或超参数优化。

---

## 203. Attacking UTMOS: Probing the Robustness of a Speech Quality Assessment Model

**arXiv ID:** 2606.31105 | [PDF](https://arxiv.org/pdf/2606.31105v1)

**作者:** Wen-Chin Huang `[一作]` (Nagoya University), Tomoki Toda `[通讯]` (Nagoya University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `6215c339-3735-4be3-8a07-5bbb7004712d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过在波形、Mel 频谱+HiFi-GAN、EnCodec 潜在空间三种输入空间上对 UTMOS 进行梯度优化，构造了两类攻击：保持预测分数不变但降低感知质量的 score-preserving 攻击，以及保持感知质量不变但降低预测分数的 quality-preserving 攻击。

**💡 创新点**

创新点在于首次将对抗性优化应用于语音质量评估模型，探究模型的鲁棒性；并比较了三种攻击空间，发现 EnCodec 潜在空间在实现质量保持攻击方面最具潜力。

**🔧 技术方法**

使用了基于梯度的白盒对抗攻击（C&W 风格），采用 Adam 优化器，结合 HiFi-GAN 语音合成器和 EnCodec 音频编解码器作为映射工具，并用 PESQ 作为感知质量代理评估。

**📊 数据集**

实验数据来源于 LibriSpeech 语料库（选取 30 句、3–5 秒、UTMOS 分数 4–5 的语音），并使用官方 UTMOS、HiFi-GAN、EnCodec 预训练模型。

**📈 对比分析**

与随机噪声扰动基线相比，score-preserving 攻击能在保持 UTMOS 分数的同时显著降低 PESQ，表明 UTMOS 对信号衰减不敏感；quality-preserving 攻击难度更大，最优结果在 EnCodec 空间实现，但仍无法保持感知质量。

**⚠️ 局限性**

局限性包括：攻击仅从高质量样本出发，未探究低质量样本的提升攻击；使用 PESQ 作为代理无法完全反映人类感知；对抗样本仍在感知上与原样本有明显差异，说明模型对完全保持质量的攻击鲁棒性尚未被破坏。

---

## 204. Seeing Through the Weights: Privacy Leakage in Scene Coordinate Regression

**arXiv ID:** 2606.31164 | [PDF](https://arxiv.org/pdf/2606.31164v1)

**作者:** Oleksii Nasypanyi `[一作]` (Stony Brook University), Francois Rameau `[通讯]` (SUNY Korea)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于查询的攻击方法，通过向训练好的场景坐标回归（Scene Coordinate Regression, SCR）模型投递无关代理图像，聚合并筛选其3D坐标预测，最终恢复目标场景的几何结构并利用特征反演网络重建近似外观。

**💡 创新点**

创新点在于：①首次系统证明SCR模型并非隐私保护本身；②设计了一套从黑盒到白盒多级访问的攻击框架，利用稳定性评分、梯度细化和体素聚合实现高精度重建；③引入特征反演网络从重建点云生成新视角图像，暴露更深层次的外观泄露。

**🔧 技术方法**

采用的技术包括：CNN特征提取器（ACE框架的预训练backbone）、MMDP/MLP场景特定头、噪声扰动稳定性评估、梯度下降细化特征、体素网格聚合、Transformer+CNN的特征反演网络、以及对比评估指标（F1、Chamfer Distance、PSNR）。

**📊 数据集**

使用的数据集：7‑Scenes（室内）与Cambridge Landmarks（室外）作为目标场景；代理查询图像来自PASCAL VOC、SUN RGB‑D和随机噪声合成；反演网络在ScanNet上训练。

**📈 对比分析**

在不同访问级别（黑盒、灰盒、白盒）下与伪真实GT（通过训练图像查询得到）对比。结果显示：白盒最高精度，灰盒接近；黑盒已能达到 2cm 约 62% F1、5cm 约 83% F1；在室内场景上总体效果良好，室外场景由于尺度大和视角多样性略低。图像重建方面，PSNR 在 15‑16 dB 之间，表明能够恢复可识别的外观。

**⚠️ 局限性**

限制：①需要大量代理查询，查询预算高；②外观恢复仅为粗略近似，颜色和纹理细节不足；③对大尺度或纹理稀疏区域的重建效果不佳；④攻击在只返回最终相机位姿的黑盒API中尚未验证；⑤缺乏针对该攻击的有效防御措施。

---

## 205. ClawArena-Team: Benchmarking Subagent Orchestration and Dynamic Workflows in Language-Model Agents

**arXiv ID:** 2606.31174 | [PDF](https://arxiv.org/pdf/2606.31174v1)

**作者:** Kaiwen Xiong `[一作]` (University Of North Carolina Chapel Hill), Huaxiu Yao `[通讯]` (University Of North Carolina Chapel Hill)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建并评估了 ClawArena‑TEAM 基准，用以测量单个 LLM 主代理在多模态、多目录工作空间中创建、授权、调度并整合固定子代理池的管理能力。

**💡 创新点**

创新点在于①通过固定本地子代理池和无 LLM 判定的执行式评分，彻底将管理能力与子代理原始能力解耦；②提出了权限精准、模态路由、调度、工作流等多维度管理指标；③在 41 个合成场景上实现了多模态、多目录、阶段性更新的严格硬约束，保证管理行为不可规避。

**🔧 技术方法**

使用了大型语言模型（LLM）、视觉语言模型（VLM）和全模态模型（Omni）作为子代理；主代理仅使用文本感知，所有子代理通过本地 vLLM 服务调用；评分脚本基于 shell 命令执行结果，无 LLM 判定。

**📊 数据集**

使用合成的多模态数据集，包含文本、代码、表格、图像、音频、视频等，共 41 个场景、258 轮、72 个阶段性更新；数据通过版本控制脚本自动生成，确保可复现。

**📈 对比分析**

对 12 种主代理模型（专有、社区和自托管）进行统一评估，计算子代理管理得分 SMS、任务完成率 TCR、权限精准等指标；结果显示旗舰 Claude‑fable‑5 最高 60% SMS，成本与管理质量解耦，开源模型在成本较低时可达 50% 以上。

**⚠️ 局限性**

局限性包括①子代理池固定且本地化，未考察动态扩容或云部署；②主代理仅文本感知，未评估多模态主代理的管理能力；③合成场景可能缺乏真实业务复杂度；④指标聚焦权限和模态路由，其他安全或效率维度尚未完全覆盖。

---

## 206. TAG-DLM: Diffusion Language Models for Text-Attributed Graph Learning

**arXiv ID:** 2606.31166 | [PDF](https://arxiv.org/pdf/2606.31166v1)

**作者:** Lingjie Chen `[一作]` (University of Illinois at Urbana Champaign), Hanghang Tong `[通讯]` (University of Illinois at Urbana Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了一种将图结构直接注入掩码扩散语言模型的框架，用以联合文本和图的推理，支持节点分类、链路预测以及跨数据集迁移。

**💡 创新点**

创新点在于通过二值拓扑注意力掩码在自注意力层中实现消息传递，消除单独图模块和任务特定输出头的需求，并将任务定义为提示中的多选填空，从而实现统一推理。

**🔧 技术方法**

使用掩码扩散语言模型（如 LLaDA‑8B‑Instruct + LoRA）和自注意力拓扑掩码；对局部子图进行线性化并通过多选提示进行标注，利用全量文本进行生成式推断。

**📊 数据集**

实验数据集包括 Cora、PubMed 和 ogbn‑arxiv 三个常用文本属性图数据集。

**📈 对比分析**

与传统 GCN、Graph Transformer 以及 LLaGA 等基线相比，取得了节点分类和链路预测的最佳或接近最佳结果，平均提升约 3–4 分；在跨数据集迁移场景中也保持显著性能优势。

**⚠️ 局限性**

局限性包括仅针对静态同质图、未处理动态图和异质关系，且拓扑掩码结构相对简单，缺乏类型感知等高级特性。

---

## 207. Reasoning-aware Speculative Decoding for Efficient Vision-Language-Action Models in Autonomous Driving

**arXiv ID:** 2606.31160 | [PDF](https://arxiv.org/pdf/2606.31160v1)

**作者:** Anh Dung Dinh `[一作]`, Flora Salim `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

将自驾场景下Vision‑Language‑Action模型的链式推理拆分为两条专用路径：一条基于轨迹历史的常规推理器，另一条保持完整视觉注意力的深度推理器，并通过自回归推理框架协同工作。

**💡 创新点**

主要创新包括：① FlatRoPE 1D旋转位置编码，破坏3D M‑RoPE 的旋转对称性，使常规推理器聚焦轨迹历史；② Action‑aware RL（AARL）后训练策略，利用动作质量奖励与静态参考 KL 约束，直接优化可接受长度。

**🔧 技术方法**

采用的技术包括：自回归推理框架（speculative decoding），多模态旋转位置编码（M‑RoPE），块扩散草稿与自回归草稿两种草稿架构，FlatRoPE 1D旋转编码，AARL 与 GRPO 策略梯度优化，静态参考 KL 锚定。

**📊 数据集**

使用 Alpamayo‑R1 数据集（共 11,655 条驾驶剪辑），划分 val（300）、on‑shelf test（200）和 off‑shelf test（290）三种评估集。

**📈 对比分析**

与 3D M‑RoPE 基线草稿在同一模型和硬件上对比，FlatRoPE 提升平均可接受长度 L 约 0.6–0.7，理论加速比提升至 4.3×（相较 3.79× 的基线），AARL 在此基础上再提升约 0.02 L。整体实现约 3.5× 的理论加速，并保持深度推理器的可用性。

**⚠️ 局限性**

局限性在于 CoC 推理长度仅 15–20 令牌，提升空间有限；方法仅在 Alpamayo‑R1 上验证，尚未证明对更长推理任务（>500 令牌）或其他 VLA 目标的适用性。

---

## 208. Rethinking Foundation Model Collaboration: Enhancing Specialized Models through Proxy Task Reasoning

**arXiv ID:** 2606.31157 | [PDF](https://arxiv.org/pdf/2606.31157v1)

**作者:** Hongyi Lin `[一作]` (Tsinghua University), Xiaobo Qu `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于任务拆分的 Foundation-Model-Augmented Task-Specific Reasoning 框架，将专业模型生成的结构化候选通过信息空间重构后交给多模态基础模型进行有限代理推理，以提升结构化预测性能。

**💡 创新点**

创新点在于将基础模型的作用从直接回归转变为在专业模型生成的候选集上进行比较、验证、选择等有限代理任务，既保留了专业模型的几何/物理约束，又利用基础模型在语义理解与上下文推理方面的优势。

**🔧 技术方法**

使用 Qwen2.5‑VL‑7B 通过 LoRA 微调做视觉‑语言代理；候选生成使用 RT‑DETR、YOLO、3D‑MOOD、HPNet、Mask2Former 等专业模型；信息空间重构通过图像覆盖、投影、结构化描述等方式完成；代理推理采用选择、验证、比较等任务。

**📊 数据集**

实验数据集包括 COCO 2017（2D 检测）、KITTI（3D 检测）、Argoverse（轨迹预测）和 Cityscapes（语义分割），覆盖四类结构化预测任务。

**📈 对比分析**

与基础模型直接回归的对比实验表明，代理推理在 AP、AP_50、ADE、mIoU 等指标上均优于直接回归，同时计算成本更低（相对成本比 1.0/4.9/1.4× 等），验证了代理推理的有效性与效率。

**⚠️ 局限性**

局限性包括：若候选集中缺失正确答案则无法恢复；候选生成与基础模型可能共享数据/场景偏差，导致错误未被纠正；实验仅在单一 VLM 与专业模型上进行，未评估闭环安全、对抗鲁棒性及多模型共存的实际部署挑战。

---

## 209. What Probing Reveals about Autonomous Driving: Linking Internal Prediction Errors to Ego Planning

**arXiv ID:** 2606.31106 | [PDF](https://arxiv.org/pdf/2606.31106v1)

**作者:** Hyeonchang Jeon `[一作]` (Gwangju Institute of Science and Technology), Kyung-Joong Kim `[通讯]` (Gwangju Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

通过线性探测、扰动实验和因果干预，研究了在不同数据规模下驾驶策略对周围车辆的预测和规划能力。

**💡 创新点**

提出了针对多车情境的预测/规划探测框架，并证明更精确的内部预测可因果改善规划。

**🔧 技术方法**

使用Transformer行为克隆、PPO强化学习、SMART多智能体预测模型，配合线性探测、扰动移除和干预实验。

**📊 数据集**

采用Waymo Open Motion Dataset（WOMD）和Waymo Open Sim Agents Challenge（WOSAC）数据。

**📈 对比分析**

在增大数据规模后，闭环仿真指标和探测性能提升，但仍出现晚预测和对非关键车辆的误重，整体性能比单纯闭环评估更具洞察力。

**⚠️ 局限性**

限制包括离散网格预测粗糙、对稀疏冲突样本敏感、仍无法及时识别真正安全关键车辆，且对模型的鲁棒性评估仍不充分。

---

## 210. ComplianceGate: Classifier-Gated Multi-Tier LLM Routing for Inference in Regulated Industries

**arXiv ID:** 2606.31163 | [PDF](https://arxiv.org/pdf/2606.31163v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 211. TaxoMIL: Taxonomy-Constrained Learning for Hierarchical Whole Slide Image Analysis

**arXiv ID:** 2606.31100 | [PDF](https://arxiv.org/pdf/2606.31100v1)

**作者:** Chaeyeon Lee `[一作]` (Korea University), Jin Tae Kwak `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于多实例学习的 WSI 诊断框架 TaxoMIL，将诊断任务改为层级文本生成，能够同时输出粗粒度和细粒度诊断结果。

**💡 创新点**

创新点在于：①使用双头 Transformer 解码器生成层级化文本；②引入税onomic 约束目标，显式构造标签嵌入空间并对齐视觉表示与临床层级；③采用循环损失调度避免多目标冲突。

**🔧 技术方法**

技术包括：多实例学习 (MIL) 框架、双分支多模态条件模块、双头 Transformer 解码器、监督对比学习、层级对齐与兄弟距离损失、图像-文本对齐损失、循环损失调度。

**📊 数据集**

在三组公开 WSI 数据集上评估：GastWSI、BRACS 与 PANDA，每个数据集均构造两级诊断层级。

**📈 对比分析**

与十二个基线（MIL 平面/层级分类器与 VLM 生成模型）对比，TaxoMIL 在整体、粗层级和细层级精度上均显著提升，尤其在细粒度诊断上提升超过 10% 并降低父子层级冲突。

**⚠️ 局限性**

局限性包括：对固定层级结构的依赖，难以适应动态更新的诊断分类；仍存在部分父子不一致问题；以及自回归解码器带来的额外参数和内存开销。

---

## 212. Do Not Break the Vessels: Structure-Preserving Mean Flow for Vascular Image Translation

**arXiv ID:** 2606.31095 | [PDF](https://arxiv.org/pdf/2606.31095v1)

**作者:** Changjin Sun `[一作]` (Southeast University), Guangquan Zhou `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出结构保持均流（SPMF）框架，在NIRII到2PF血管图像翻译过程中保证血管拓扑在生成过程中不被破坏，并结合原型引导结构细化（PGSR）以桥接训练与推理阶段的结构分布差异。

**💡 创新点**

①从结构不变性原理推导速度场正交约束，正式将外观转移与拓扑失真分离；②将该约束以时间加权替代目标嵌入Brownian-bridge扩散模型中，确保每个扩散步骤保持拓扑；③设计PGSR模块，通过原型检索和频域加权对齐，将噪声结构投影到训练学习的共享解剖流形上。

**🔧 技术方法**

Brownian-bridge扩散模型、Sato血管性检测器、速度场正交约束与时间加权一致性损失、原型检索、频域分解与加权对齐、全变差正则化。

**📊 数据集**

NIRII–2PF双模成像数据集（763对图像）以及外部眼底图像数据集（1001例，含IR与RGB对）。

**📈 对比分析**

与SRGAN、SRN、BBDM、LDL、SelfRDB等方法在PSNR、SSIM和V-MSE上进行对比。SPMF+PGSR在NIR2PF数据集上PSNR最高24.96 dB，SSIM 0.6904，V-MSE最低2.499×10⁻³；在Fundus数据集上PSNR 24.83 dB，SSIM 0.5823，V-MSE 7.76×10⁻³。消融实验表明PGSR和SPMF各自均显著提升性能，二者结合获得最优结果。

**⚠️ 局限性**

主要基于血管性误差（V-MSE）评估，缺乏骨架或中心线等更直接的拓扑指标；数据集规模有限，缺少专家评估；未来需扩展更大规模数据并引入临床专家判断来进一步验证临床实用性。

---

## 213. PPT-Eval: A Benchmark for Computer-Use Agents on PowerPoint Tasks

**arXiv ID:** 2606.31154 | [PDF](https://arxiv.org/pdf/2606.31154v1)

**作者:** Apurva Gandhi `[一作]` (Carnegie Mellon University), Graham Neubig `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个面向 PowerPoint Online GUI 的 120 任务基准——PPTArena，涵盖了创建与编辑幻灯片的真实工作流程与评估框架。

**💡 创新点**

创新点在于构建树状分层 rubric，既能给出部分得分、惩罚无关改动，又能自动生成自然语言反馈，并将此方法应用于大规模多模态幻灯片编辑任务。

**🔧 技术方法**

技术上采用 LLM（如 Claude、ChatGPT）生成任务与 rubric，利用 VLM 进行视觉与语义检查，并在沙盒化的 Chromium 环境中模拟鼠标键盘操作完成 PowerPoint Online 的 GUI 交互。

**📊 数据集**

数据集基于 12 套公开许可的 PowerPoint 文件（共 404 张幻灯片），从中挑选并手工筛选出易/中/难三类任务，形成 120 个标准化任务集。

**📈 对比分析**

在 30 步预算下对 5 名人类、闭源 frontier 模型（Claude‑4.5‑Opus、OpenAI GPT‑4.1106）以及多款开源模型进行评估；最高成功率约为 45%（平均分 0.57），人类 80%/0.90，API baseline 更优。

**⚠️ 局限性**

限制包括：评价结果受 LLM/VLM hallucination 影响；rubric 的生成仍需大量人工编辑；目前 GUI 代理性能仍远低于人类，缺乏跨应用多任务场景的支持。

---

## 214. Peak Sidelobe Suppression in Planar Fluid Antenna Array

**arXiv ID:** 2606.31149 | [PDF](https://arxiv.org/pdf/2606.31149v1)

**作者:** Haoyu Liang `[一作]` (Southeast University), Zaichen Zhang `[通讯]` (Southeast University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种改进的遗传算法，用于稀疏平面流体天线阵列的峰值旁瓣抑制（PSLL）优化。

**💡 创新点**

创新点包括：锦标赛选择、适应性交叉/变异概率、混合交叉策略、多点变异和精英池保留，显著提升搜索效率并避免早熟收敛。

**🔧 技术方法**

采用改进的遗传算法（IGA）对阵列端口激活模式进行组合优化，并用平面流体天线阵列的物理模型计算阵列因子和PSLL。

**📊 数据集**

使用模拟数据：平面阵列总端口数2500，端口间距0.5λ，激活端口数300，频率3.5 GHz，进行多次迭代（T_max = 300）比较IGA与基准CGI。

**📈 对比分析**

与传统基准遗传算法（CGA）比较，IGA在相同迭代次数下收敛更快，最终PSLL降低4.45 dB，端口利用率提升88%（从2500降至300），仅牺牲约1.43 dB方向性。

**⚠️ 局限性**

局限性：仅在仿真环境下验证，未涉及实际硬件实现；仅针对单一平面网格和单一波束方向；可能对不同的阵列拓扑或多波束情况适用性需进一步研究。

---

## 215. PruneGround: Plug-and-play Spatial Pruning for 3D Visual Grounding

**arXiv ID:** 2606.31148 | [PDF](https://arxiv.org/pdf/2606.31148v1)

**作者:** Duc Cao Dinh `[一作]` (Knovel Engineering Lab), Zhijing Jin `[通讯]` (University of Toronto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于“先裁剪再定位”的3D视觉定位框架：通过冻结的视觉语言模型（VLM）先裁剪与文本相关的空间区域，再利用多视角信息简化并补全描述，最后使用已预训练的空间LLM进行目标定位。

**💡 创新点**

创新点在于：① 语言引导的空间裁剪（LGSP）利用VLM在多视角图像上定位语义相关区域，显著缩小搜索空间；② 多视角条件的描述重构（MCDR）把复杂表达拆解为单目标-单锚关系并补全缺失的空间线索；③ 将检测预训练的空间LLM重新用于语言条件下的定位，通过对齐点云与文本表征实现高效且准确的目标预测。

**🔧 技术方法**

技术包括：冻结的多模态VLM（如Qwen2.5-VL-3B-Instruct）进行裁剪与描述重构；多视角渲染（顶视图+倾斜视图+侧视图）；点云编码器（Sonata）与LLM（Qwen2.5-0.5B）联合推理；以及基于语义嵌入的投影与解码。

**📊 数据集**

使用的公开数据集有 ScanRefer、Nr3D 和 Sr3D，覆盖多种场景与表达复杂度，并在每个数据集上进行官方拆分训练/验证。

**📈 对比分析**

与之前方法对比，该框架在 ScanRefer（Unique、Multiple、Overall）三种设置上均刷新了SOTA；在 Nr3D 和 Sr3D 的 10 个子任务中，除一个外均获得最高分，整体精度约 75-81%（Acc@0.5）与 IoU 最高；相较于传统两阶段或单阶段方法，显著提升了定位准确率并降低了候选数量。

**⚠️ 局限性**

局限性包括：各阶段误差会级联，裁剪误差或描述重构错误会直接影响最终定位；在极度混乱或语义模糊的场景中表现仍受限；对 VLM 的依赖导致额外计算开销，虽然使用小模型可缓解，但仍比单阶段模型更复杂。

---

## 216. Fluid-Antenna-Aided Active User Detection With 1D-CNN Channel Reconstruction for Unsourced Random Access

**arXiv ID:** 2606.31139 | [PDF](https://arxiv.org/pdf/2606.31139v1)

**作者:** Haoyu Liang `[一作]` (Southeast University), Zaichen Zhang `[通讯]` (Southeast University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于一维卷积神经网络（1D‑CNN）的信道重构方法，用于流体天线（FAS）在无源随机接入（URA）场景中的主动用户检测（AUD）；该方法通过对部分观测信道进行学习，重构完整信道向量并引导端口选择以提升AUD性能。

**💡 创新点**

创新点在于：①首次将1D‑CNN用于FAS信道重构，能够有效捕捉空间相关性与非线性映射；②通过重构得到的信道信息动态指导端口选择，实现比传统固定天线更优的AUD性能；③相较于传统OMP、ML、SelMMSE和S‑BAR等模型方法，显示出更低的NMSE和更低的AUD误检率。

**🔧 技术方法**

使用的技术包括：1D‑CNN网络（三层卷积+两层全连接回归头）、批归一化、LeakyReLU激活、Adaptive Average Pooling、Dropout、端口选择矩阵设计以及与GAMP等基准算法的对比实验。

**📊 数据集**

使用合成数据集，训练集100,000个样本，验证集5,000个样本；输入维度D_in=2M（M为选定端口数），输出维度D_out=2N（N为全部端口数）。实验中使用14长度导频、50活跃用户、总用户数2^14。

**📈 对比分析**

与OMP、ML、SelMMSE、S‑BAR以及传统固定天线GAMP等方法比较，1D‑CNN在所有导频设置下均实现了更低的NMSE，并在多种SNR下的AUD误检率显著下降；特别是随着天线长度W和端口数N的增大，性能提升更为明显。

**⚠️ 局限性**

局限性包括：①仅在仿真环境下验证，缺乏实地测试；②假设信道完全相关且为Toeplitz结构，实际环境可能更复杂；③网络深度有限，可能在极端噪声或极短导频下性能下降；④端口选择仍依赖重构误差，若重构不准确会影响AUD。

---

## 217. FROST: Training-Free Few-Shot Segmentation with Frozen Features and Nonparametric Statistics

**arXiv ID:** 2606.31136 | [PDF](https://arxiv.org/pdf/2606.31136v1)

**作者:** Junghwan Park `[一作]` `[通讯]` (Telepix), Junghwan Park (Telepix)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无训练的少样本分割方法FROST，在远程遥感图像中通过在冻结的DINOv3特征空间上使用非参数密度比率来标注查询像素。

**💡 创新点**

创新点在于：① 用分布而非点或聚类来描述目标类；② 通过留一法从支持集自动选择核宽度和空间门限；③ 在位置去偏和弱化缩放下对特征进行精细归一化，提升密度估计的精度；④ 采用多尺度边缘平滑和前后候选门限实现更清晰的分割。

**🔧 技术方法**

技术主要包括：冻结的DINOv3视觉Transformer、L2归一化、位置去偏投影、带高强度收缩的白化、von Mises–Fisher核密度估计、密度比率阈值（Bayes阈0）、双边传播、前后候选门限以及双线性上采样。

**📊 数据集**

数据集涵盖17个遥感分割基准（如SpaceNet、xBD、WHU Building、DeepGlobe、FloodNet等）以及6个自然图像与跨域基准（PASCAL‑5^i、COCO‑20^i、LVIS‑92^i、PACO‑Part、ISIC、SUIM）。

**📈 对比分析**

与传统训练无关的方法（INSID3、FSSDINO、PerSAM、Matcher、GF‑SAM）以及学习驱动方法（SegIC、SINE、DiffewS、SegGPT）进行对比。FROST在17个遥感基准上获得16/17个榜首，单样本平均mIoU 48.3，10样本时提升至56.8，明显优于对手，且模型规模最小；在自然图像与跨域基准上也保持领先，尤其在ISIC、SUIM等域外场景。

**⚠️ 局限性**

局限性包括：依赖支持集特征估计，单一非代表性样本会导致密度误估；受限于冻结的DINOv3特征，无法适应深度差异或严重域迁移；等先验阈值与单类前景/背景设计最适用于遥感场景，对小目标占比极低的对象中心化场景表现略逊；目前仅支持单类分割，未考虑多类别联合决策。

---

## 218. MSNN-LINet: Cross-Modal Learning via Continuous Linear Integration

**arXiv ID:** 2606.31135 | [PDF](https://arxiv.org/pdf/2606.31135v1)

**作者:** Gabriel Clinger `[一作]` `[通讯]` (George Washington University), Gabriel Clinger (George Washington University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 LINet，一种通过连续层级跨模态融合的多流神经网络，实现 RGB‑D 场景分类。

**💡 创新点**

创新点在于：①引入 LIConv2d 操作，实现每层预激活的线性跨模态整合；②使用 1/N 常数初始化桥接权重以防梯度失真；③设计渐进式模态丢弃（Progressive Modality Dropout）来避免负共学习。

**🔧 技术方法**

核心技术包括多流架构、LIConv2d（三条并行卷积 + 1×1 混合）、批归一化与 ReLU 结合、1/N 常数初始化、渐进式模态丢弃与监测。

**📊 数据集**

使用的数据集有：SUN RGB‑D（19 类）进行从零训练与 ScanNet 预训练；NYU Depth V2（10 类）进行从零训练验证。

**📈 对比分析**

与早期融合、晚期融合、以及基准 ViT‑B 方法对比，LINet 在 SUN RGB‑D 上从零训练得到 45.2% MCA，预训练后升至 49.6%，超过 CoMAE（40.5%）与 ResNet18 早/晚期融合；在 NYU Depth V2 上亦实现 50.8% MCA，优于早/晚期融合。

**⚠️ 局限性**

局限性主要是 SUN RGB‑D 官方测试集存在类级别的传感器分布漂移，导致不同类的性能波动；此外梯度初始化与模态丢弃的调度对性能影响显著，需要进一步自动化或自适应策略。

---

## 219. Scenario Generation for Testing of Autonomous Driving Systems Using Real-World Failure Records

**arXiv ID:** 2606.31131 | [PDF](https://arxiv.org/pdf/2606.31131v1)

**作者:** Anjali Parashar `[一作]` (Massachusetts Institute of Technology), Chuchu Fan `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出基于LLM的自动化场景生成流水线，利用NHTSA历史事故记录中的分类变量和自然语言叙述，生成符合特定仿真约束的多样化场景。

**💡 创新点**

创新点：①首次将事故叙述与分类特征结合，通过三步LLM管道（释义、生成、微调）实现系统特定场景细化；②使用聚类+文本嵌入实现成本感知的多样化测试集；③将生成的场景直接投入Metadrive仿真进行验证。

**🔧 技术方法**

使用技术包括：ChatGPT/类似LLM进行释义、场景生成与微调；k‑means聚类配合OpenAI文本嵌入；Metadrive仿真平台；可与贝叶斯优化、MCMC等后续优化工具结合。

**📊 数据集**

数据集：美国NHTSA 2021年ADS事故记录（约2295起，筛选后1911起含叙述，进一步压缩至235起满足可仿真条件的案例）。

**📈 对比分析**

对比方法：在20个聚类样本上评估模板匹配率、最小距离改进等指标，结果显示95%场景准确匹配模板，最小间隙显著下降，证明能在有限预算内揭示多种失效模式；与传统手工设计的静态测试集相比，发现更多边缘失败。

**⚠️ 局限性**

局限性：缺乏统一的覆盖度定义；对固定策略的复制能力有限，无法完全复现期望的ego运动；生成场景受限于可仿真参数，未覆盖视觉条件；LLM生成质量需进一步评估与标准化。

---

## 220. SkillSpotter: Pose-Aware Multi-View Skilled Action Detection and Grading in Ego-Exo Videos

**arXiv ID:** 2606.31127 | [PDF](https://arxiv.org/pdf/2606.31127v1)

**作者:** Björn Braun `[一作]` (ETH Zürich), Christian Holz `[通讯]` (ETH Zürich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种姿态感知的多视角框架，能够在同步的自视角和外部视角视频中实时检测并评估运动技能的执行质量。

**💡 创新点**

创新点在于：①自适应时间抑制模块，能够根据不同场景的动作密度动态调整抑制半径；②门控三维身体姿态融合模块，利用人体运动学提升质量评估；③双向跨视角注意力机制，解决多视角信息融合导致的评估精度下降。

**🔧 技术方法**

使用的技术包括：ActionFormer 时序背骨网络、跨视角多头注意力、门控融合、软NMS、以及自监督的自适应抑制学习。

**📊 数据集**

主要在 Ego-Exo4D 技能演示基准上进行实验，并将模型迁移到 HoloAssist 误差检测数据集进行泛化验证。

**📈 对比分析**

相较于七种先进的时序动作检测模型，本文方法在 Ego-Exo4D 的类特定 mAP 从 12.40 提升至 21.82（+76%），在全景视角下平衡准确率从 55.99% 提升至 60.40%。在 HoloAssist 上，检测 mAP 由 3.33 提升至 7.24，F1 分数提升至 53.3。

**⚠️ 局限性**

局限性包括：对细粒度手部或物体交互的评估仍不足，场景间性能差异显著（如音乐、烹饪场景表现较差），多视角融合仍未能超过单视角性能，且标注主观性导致评估上限有限。

---

## 221. PiLoT v2: Pixel-to-Orthogonal Map Alignment for Free-view UAV Geo-localization

**arXiv ID:** 2606.31098 | [PDF](https://arxiv.org/pdf/2606.31098v1)

**作者:** Xinyi Liu `[一作]` (National University of Defense Technology), Yu Liu `[通讯]` (National University of Defense Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PiLoT v2，一种轻量化的无人机实时漂移-free 地理定位框架，将传统 3D 网格渲染替换为基于 TDOM/DSM 的图像裁剪与跨视角特征配准；

**💡 创新点**

创新点包括：① 用 CPU‑友好的 TDOM/DSM 裁剪代替 GPU 3D 渲染，显著降低存储和计算开销；② 通过大规模合成数据训练跨视角特征网络，弥合正射地图与斜视图图像之间的几何差异；③ 将重力方向与单点激光距离等机载传感器先验直接融入 LM 优化，提升鲁棒性；

**🔧 技术方法**

主要技术：图像裁剪、2.5D TDOM/DSM 辅助映射、跨视角特征提取（MobileOne 编码器‑解码器）、多假设粗细 LM 优化、传感器先验残差约束、合成数据生成与几何监督；

**📊 数据集**

使用了 SynthCity-6、UAVScenes、UAVD4L-2yr 三大基准集，并构建了与原 3D 网格对齐的 TDOM/DSM 版数据集；

**📈 对比分析**

与 Render2Loc、PiLoT、PixLoc 等 3D 网格渲染方法以及 OrthoLoC 等裁剪基准进行对比，PiLoT v2 在 SynthCity-6 上 Recall@3/5/10 同时达到 100%，在 UAVScenes 与 UAVD4L-2yr 上实现更低的中位翻译误差（1.58m/1.49°）并保持 17 FPS，FRC 为 0；

**⚠️ 局限性**

局限性：仍需合成数据来训练跨视角网络，训练成本高；在极端光照或地形变化严重的情形下，正射地图的纹理信息可能不足，导致特征匹配失效；此外，单点激光距离的可获取性对硬件配置有一定要求。

---

## 222. GaussianMap: Learning Gaussian Representation for Multi-Sensor Online HD Map Construction

**arXiv ID:** 2606.31177 | [PDF](https://arxiv.org/pdf/2606.31177v1)

**作者:** Hongyu Lyu `[一作]` (University of Sydney), Stewart Worrall `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种在线高清地图构建框架 GaussianMap，通过在鸟瞰图（BEV）平面上学习自适应高斯原语来生成局部向量化地图。

**💡 创新点**

创新点：①利用可学习的高斯原语代替固定格点的 BEV 网格，使模型仅在与地图相关的稀疏区域分配建模容量；②设计 Feed‑Forward 高斯编码器，包含高斯自注意力、相机注意力以及 LiDAR 注意力，实现多模态特征的聚合；③将高斯表示通过 Gaussian‑to‑BEV splatting 转化为结构化 BEV 特征，从而兼容现有向量化地图解码器。

**🔧 技术方法**

核心技术包括：高斯原语参数化（均值、尺度、旋转、透明度、特征向量）；高斯自注意力（deformable attention）；相机和 LiDAR 关注机制；高斯到 BEV 的分散映射；使用 ResNet‑50 作为相机骨干、SECOND 作为 LiDAR 骨干；MapQR+DAMap 作为向量化地图解码器。

**📊 数据集**

使用公开数据集 nuScenes（1,000 场景，6 视角相机 + 32 声束 LiDAR）和 Argoverse 2（1,000 场景，7 视角相机 + 2×32 声束 LiDAR），在 700/150/700/150 训练/验证划分上进行评估。

**📈 对比分析**

与 MapQR+DAMap、RoboMap、MapTR、ADMap 等现有方法比较；在 camera‑only 模式下 nuScenes 上 mAP 70.5（比 MapQR+DAMap 提升 1.7），Argoverse 2 上 70.1（提升 2.0）；在 camera‑LiDAR 融合模式下 nuScenes 上 78.3（比 MapQR+DAMap 提升 3.1），Argoverse 2 上 79.0（提升 2.9）。整体上实现了两大基准数据集上的 state‑of‑the‑art 性能。

**⚠️ 局限性**

局限性：①需要在 BEV 上预设大量高斯原语（20,000），在计算和显存上仍有一定开销；②目前仅对静态地图元素进行预测，对动态场景或时间变化的地图信息尚未深入研究；③对极稀疏或细粒度地图特征的精确定位仍有提升空间，尤其在复杂交叉路口或非结构化道路环境中。

---

## 223. An Empirical Study of Security Calibration in Large Language Models for Code

**arXiv ID:** 2606.31159 | [PDF](https://arxiv.org/pdf/2606.31159v1)

**作者:** Mohammed Latif Siddiq `[一作]` (University of Notre Dame), Joanna C. S. Santos `[通讯]` (University of Notre Dame)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了大型语言模型在代码生成中的安全校准性，量化了模型自评安全与功能正确性的可信度与误差，并探究了不同校准导向的自动修复与缓解策略；

**💡 创新点**

首次以安全校准为评价维度对LLM代码生成进行大规模实验，揭示了功能与安全自评的差异、仓库级情境下的校准退化、以及修复过程中出现的“僵化”瓶颈；

**🔧 技术方法**

采用期望校准误差(ECE)、Brier分数、False Trust率等校准指标，结合温度采样、标记式自信推断、token概率、采样一致性与自一致性提示，以及执行门控、授权破坏修复、少量示例与链式思维提示等技术；

**📊 数据集**

使用SALLM（100个自包含安全任务）与AICGSecEval（125个跨文件仓库级任务）两套公开基准，涵盖Python、C、Java、PHP等多语言与29类CWE；

**📈 对比分析**

通过在GPT‑4o‑mini、Gemini‑2.0‑Flash、Qwen3‑Coder‑Next三大模型及六温度设置下进行对比，实验表明模型普遍过度自信（ECE>0.4、False Trust>30%），功能校准远逊于安全校准，仓库级别校准误差提升至约3-4倍；修复策略虽能略减False Trust，但整体安全率提升有限，修复成功率≤2.4%；

**⚠️ 局限性**

受限于仅评估三种模型与两份基准，校准度量受信心推断方式影响；自动修复实验对功能完整性破坏严重，缺乏更强泛化的修复方法；实验环境与硬件限制导致修复与评估成本高，未来模型更新可能改变绝对数值但结构性结论仍具代表性。

---

## 224. AETDICE: Unified Framework and Offline Optimization for Nonlinear Multi-Objective RL

**arXiv ID:** 2606.31178 | [PDF](https://arxiv.org/pdf/2606.31178v1)

**作者:** Woosung Kim `[一作]` (Korea University), Byung-Jun Lee `[通讯]` (Korea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了AET框架，统一了多目标强化学习中轨迹层与期望层非线性加权的形式，并基于该框架设计了AETDICE算法，首次实现了离线优化ESR、SER、BSR以及更通用的AET目标；

**💡 创新点**

创新点在于：①将轨迹级非线性变换F与期望级聚合G拆分为三段(A–E–T)并统一求解；②通过在增广MOMDP中引入转化奖励，将F吸收到每步奖励中，恢复Markov性；③在增广空间中推广DICE方法，加入ϕ-散度正则化和松弛变量，解决了非凸聚合G的全局优化；④提出BSR目标平衡ESR与SER的公平性，弥补了两者的空缺；

**🔧 技术方法**

使用了增广状态空间、转化奖励、DICE框架（包含ϕ-散度正则化与对偶变换）、IQL离线强化学习、加权行为克隆以及对比的OptiDICE/FairDICE等方法；

**📊 数据集**

实验数据集包括：Fair‑Taxi、MO‑PointMaze‑3obj、D4MORL基准中的MO‑Ant‑v2、MO‑Walker2d、MO‑Halfcheetah等多目标连续控制环境；

**📈 对比分析**

与OptiDICE（线性F+线性G）、FairDICE（线性F+凹性G）和ESR‑IQL（非线性F+线性G）进行对比。AETDICE在BSR、ESR、SER以及复杂AET目标上均优于或匹配对手，取得更平衡的收益分布；例如在MO‑Ant中AETDICE在凸F+凹G组合下得到953.53±16.31的平均回报，明显优于FairDICE的681.95±24.28；在Fair‑Taxi中展示了ESR、BSR、SER三种策略的行为差异，证实了AET框架对公平性调节的有效性；

**⚠️ 局限性**

局限性包括：增广状态空间使得状态维度高、连续情况下估计转移和累计奖励的困难；对离线数据的要求较高，需要记录累计奖励信息；当前理论分析主要针对凸聚合G，非凸或高阶非线性仍缺乏完善的收敛证明；以及在大规模连续任务中算法的计算开销较大。

---

## 225. Cross-Domain Feature Expansion for Tabular Medical Data via Knowledge Graphs Injection

**arXiv ID:** 2606.31171 | [PDF](https://arxiv.org/pdf/2606.31171v1)

**作者:** Mengying Zhou `[一作]` (Shanghai University of Finance and Economics), Yang Chen `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出 MedKGTab，一种在表格医学数据中通过知识图注入实现跨域特征扩展的框架。

**💡 创新点**

创新点在于利用 SPOKE 知识图构建特征级先验，并将其作为结构化注意力偏置直接注入 TabPFN 的交叉特征注意力中，实现数据与知识通道的最优协同。

**🔧 技术方法**

核心技术包括基于 TabPFN 的双向注意力机制、基于一跳邻域的知识矩阵投影、可调的知识注入强度 α 以及对数据和知识通道的联合训练。

**📊 数据集**

实验使用了两组慢性萎缩性胃炎数据集（含 432 细菌分类和 120 代谢物）以及公开的 147 细菌 + 192 代谢物 fecal 数据集。

**📈 对比分析**

与十个表格生成基线和多种 LLM 方法对比，MedKGTab 在同域与跨域特征扩展任务中分别以 MSE、R²、Column Distribution、Inter‑Column 关系等指标均击败最强基线，表现最优。

**⚠️ 局限性**

主要局限在于对知识图的完整映射依赖，若表格特征与图节点映射不完整或知识图缺失关键关系，模型效果会受到一定影响。

---

## 226. Probe Choice Changes Canary-Memorization Verdicts: Three Post-Hoc Disagreement Case Studies in a Text-Dominant LoRA-Tuned Autoregressive Testbed

**arXiv ID:** 2606.31168 | [PDF](https://arxiv.org/pdf/2606.31168v1)

**作者:** Zhichao Fan `[一作]` (University of Illinois Urbana Champaign), Yanhang Li `[通讯]` (Northeastern University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在 Qwen2.5-VL-7B 生成模型上，利用可变窗口 NLL 探针评估水印（canary）记忆化，并发现其与全跨度 NLL 与行为 hit@k 的判定存在不一致。

**💡 创新点**

提出并系统展示三种探针失效模式（窗口截断导致假阴性、非秘密前缀漂移导致假阳性、以及未训练控制导致的模糊性），为未来的记忆化评估提供了明确的失效诊断框架。

**🔧 技术方法**

使用三种探针：截断窗口 NLL（K=20）、全跨度 secret NLL、以及行为 exact‑recall hit@k；通过 LoRA 注入图像/文本水印并在不同的 benign‑SFT 方案下进行微调；对探针贡献进行分段分解。

**📊 数据集**

实验数据来自 Qwen2.5-VL-7B 7B 版本，配合 20 条图像水印和 20 条文本水印（每条包含 13‑token hex 序列），并在四种 benign‑SFT 数据源（GUI、SYNTH、Safety、Random）上进行 LoRA 微调。

**📈 对比分析**

通过对比三种探针的结果，证明单一窗口 NLL 探针会导致误判：C3（窗口截断）导致 hit@1 降至 0.88，C4（非秘密漂移）导致 mean_20 上升但 hit@1 仍为 1.00，C5（未训练控制）出现 mean_20 降低但 full‑span NLL 正值且 hit@1=0；这些结果说明了探针不一致性的具体表现，但未给出整体性能指标。

**⚠️ 局限性**

局限性包括：实验仅在单一模型与模板上进行；探针选择未进行预注册；缺少匹配的 decoy 对照；未检验不同 tokenization 或更大模型规模的泛化性；并且未提供完整的行为召回统计（如 hit@4、hit@16）。

---

## 227. MIRTH: Mutual-Information Reasoning with Temporal Hubs for Vision-Language-Action Agents

**arXiv ID:** 2606.31167 | [PDF](https://arxiv.org/pdf/2606.31167v1)

**作者:** Hao Sun `[一作]` (Ritsumeikan University), Yen-Wei Chen `[通讯]` (Ritsumeikan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MIRTH框架，将视觉语言预训练模型与双尺度时间记忆、互信息驱动的潜在推理和并行动作解码结合，用于机器人跟随自然语言指令执行长时序操控。

**💡 创新点**

创新点包括：① 双尺度时间记忆中心压缩长短期历史，消除单帧模型的时间盲点；② 通过互信息目标学习潜在推理标记，弥合高层语言目标与低层控制命令的鸿沟；③ 并行向量化动作解码，显著提升推理速度和实时控制吞吐。

**🔧 技术方法**

技术方法包括：预训练视觉语言模型（如DINOv2+SigLIP），双尺度EMA与注意力时间记忆，互信息对比学习（InfoNCE）驱动的潜在推理标记，向量化动作解码器和L1回归训练。

**📊 数据集**

使用LIBERO仿真基准（包含长时序任务）以及真实的LeRobot机器人平台的数据集进行评估。

**📈 对比分析**

与Diffusion Policy、Octo、OpenVLA、OpenVLA‑OFT等公开基线相比，MIRTH在LIBERO短/长时序任务上均取得>95%成功率，短时序任务接近100%，并在LeRobot平台上实现最高吞吐率和鲁棒性。

**⚠️ 局限性**

局限性包括：潜在推理标记不易解释；仅在单臂固定机器人上验证，缺乏多手/移动机器人泛化；固定大小的压缩记忆可能对极长任务产生遗忘；在无监督推理与安全性、社会偏见方面仍需进一步考量。

---

## 228. LLM-Powered Interactive Robotic Action Synthesis from Multimodal Speech, Gestures, and Music

**arXiv ID:** 2606.31158 | [PDF](https://arxiv.org/pdf/2606.31158v1)

**作者:** Snehasis Banerjee `[一作]` (Tata Consultancy Services), Ranjan Dasgupta `[通讯]` (Tata Consultancy Services)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了基于LLM的多模态机器人动作合成框架，将语音、手势和音乐节拍融合，驱动四足机器人完成复杂动作。

**💡 创新点**

创新点在于利用LLM进行多模态信息融合与规划，并通过限定动作空间、结构化提示和验证器提升安全性，实现自然、节奏同步的交互。

**🔧 技术方法**

使用 Whisper ASR、实时手势识别、节拍检测、Qwen3 0.6b LLM、ROS/CycloneDDS 与 Unitree Go2 SDK 等技术。

**📊 数据集**

实验采用实时采集的语音、手势和音乐数据，无公开数据集；动作空间由预先定义的安全技能库构成。

**📈 对比分析**

在 Unitree Go2 上进行初步验证，展示了手势触发、节拍同步的手部立动作，系统实现近实时响应；未给出定量指标。

**⚠️ 局限性**

局限包括 LLM 生成的幻觉风险、动作空间有限、实时推理受限、缺乏大规模定量评估与反馈循环。

---

## 229. A Modular Vision-Language-Action Robotics Framework for Indoor Environments

**arXiv ID:** 2606.31144 | [PDF](https://arxiv.org/pdf/2606.31144v1)

**作者:** Anindya Jana `[一作]` (Tata Consultancy Services), Ranjan Dasgupta `[通讯]` (Tata Consultancy Services)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `51c0528b-f690-4182-ae60-bb5f046c276c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一套基于ROS的模块化视觉‑语言‑动作框架，能够在室内环境中自主探索、构建语义体素地图，并通过 Gemini 2.0 Flash 对自然语言指令进行分类、定位与执行。

**💡 创新点**

创新点包括：① 将 OwlViT 生成的 CLIP 嵌入实时融入体素地图，实现对对象的语义和几何双重定位；② 采用时间约束的探索策略（500 s）并在此期间完成部分/完整地图构建；③ 通过 VLM 先进行指令分类，再根据地图生成上下文丰富的提示，显著提升了指令理解和执行效率。

**🔧 技术方法**

核心技术包括：ROS 机器人框架、LiDAR‑基 2D 网格映射与全覆盖路径规划、OwlViT 嵌入、Gemini 2.0 Flash 语言模型、语义体素地图构建、Waypoint 控制与路径执行。

**📊 数据集**

使用 CMU VLA Challenge 提供的 AI Habitat / Unity 虚拟室内场景（Matterport3D 模型）进行测试与验证。

**📈 对比分析**

与 SORT3D、VLA‑3D 等基准方法相比，系统通过优化规划与体素尺寸将探索时间从约 8 min 42 s 降至 4 min 17 s（约 50% 缩短），并能在 10 min 总时限内完成地图构建与问题回答，整体性能符合挑战评测要求。

**⚠️ 局限性**

局限性：① 依赖网络请求 Gemini API，导致延迟和网络中断风险；② 采用“先探索‑后执行”静态地图策略，对动态环境不适用；③ 对语义细粒度的实时更新与多对象交互支持有限。

---

## 230. A Bayesian Filtering Approach for Learning Lagrangian Dynamics from Noisy Measurements

**arXiv ID:** 2606.31137 | [PDF](https://arxiv.org/pdf/2606.31137v1)

**作者:** Kundan Kumar `[一作]` (Ahmedabad University), Simo Särkkä `[通讯]` (Aalto University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

基于贝叶斯滤波的拉格朗日动力学学习框架，联合估计系统状态和神经网络参数。

**💡 创新点**

创新在于将拉格朗日动力学与连续时间随机状态空间模型相结合，利用高斯近似贝叶斯滤波实现部分噪声测量下的联合学习。

**🔧 技术方法**

采用神经网络参数化的动能、势能，Euler–Lagrange方程转化为随机SDE，并使用扩展卡尔曼滤波/球面卡尔曼滤波与统计线性回归进行高斯近似滤波。

**📊 数据集**

实验数据基于简谐摆和Duffing振子，采用人工生成的噪声测量序列。

**📈 对比分析**

与传统已知模型的EKF/CKF以及传统LNN（数值微分得到速度）进行对比，结果显示PrEKF/PrCKF在噪声测量下RMSE与已知模型相近，LNN在噪声下表现较差。

**⚠️ 局限性**

局限在于需要对动能的奇偶性约束、网络结构和噪声模型进行手工设计；对更复杂多体系统的可扩展性未验证。

---

## 231. Explaining Machine Learning and Memorization with Statistical Mechanics

**arXiv ID:** 2606.31110 | [PDF](https://arxiv.org/pdf/2606.31110v1)

**作者:** Robin Theriault `[一作]` `[通讯]`, Robin Theriault

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文以统计力学为工具，对密集Hopfield网络和受限玻尔兹曼机（RBM）在教师-学生框架下的学习机制进行解析，揭示了隐式低维学习、对抗鲁棒性与模型容量的关系，并提出了基于分析结果的正则化和层次化训练方案；

**💡 创新点**

创新点包括：① 将密集Hopfield网络与RBM统一映射为同一类能量模型，利用Nishimori线实现解析；② 在零温度下推导出对抗鲁棒性的闭式公式；③ 证明超参数过大的学生只学习到教师的部分隐藏单元，形成低维学习策略，进一步与彩票票据假设建立联系；④ 设计新的正则化方法显著提升真实数据下DAM模型的训练稳定性；⑤ 构建权重层次化算法，加速训练速度；

**🔧 技术方法**

采用统计力学的相变分析、复制方法、Nishimori线识别、自由熵求解以及蒙特卡洛数值验证等技术；

**📊 数据集**

主要使用MNIST手写数字的二值化样本作为训练数据，以及从密集Hopfield网络生成的合成数据和DAM模型的合成/真实数据；

**📈 对比分析**

通过将理论得到的相图与蒙特卡洛模拟结果对比，验证理论预测的学习相位与对抗鲁棒性；正则化方案在真实数据上显著提升了训练稳定性；层次化训练方法在相同模型规模下显著降低了训练时间；

**⚠️ 局限性**

局限性包括：① 理论推导多基于Replica对称性，非Nishimori条件下可能需RSB校正；② 对抗鲁棒性闭式公式仅在零温度下成立；③ 高阶交互需要极大量样本，对实际应用有限制；④ 研究聚焦于密集Hopfield和RBM，尚未验证对更深层网络的推广；

---

## 232. InfiniVerse: Occupancy Guided Unbounded Scene Generation for Autonomous Driving

**arXiv ID:** 2606.31109 | [PDF](https://arxiv.org/pdf/2606.31109v1)

**作者:** Xiaoyu Ye `[一作]` (Chinese University of Hong Kong), Zhen Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出InfiniVerse框架，利用单帧多视角输入先重建3D占据网格，再通过占据引导的视频扩散模型自动延展长时间、可控的城市驾驶场景，并实现3D占据与2D视频的双向反馈迭代。

**💡 创新点**

创新点包括：1）将3D占据网格与视频扩散模型耦合，兼顾几何一致性与视觉真实感；2）引入层级“草图-细化”闭环，使占据和视频相互提升，显著降低漂移与幻觉；3）支持任意轨迹、任意视角的长时序生成。

**🔧 技术方法**

技术手段：XCube（稀疏体素扩散）、Triplane编码、视频扩散模型CogVideoX1.5-5B + LoRA、ControlNet式多平面图像注入、跨视角与跨帧注意力、参考图像驱动的生成、区域加权损失。

**📊 数据集**

使用Waymo Open Dataset和nuScenes进行训练与评测，并通过COLMAP重建、Occ3D等工具构建精细占据网格；还构建了多视角视频-文本配对数据用于文本控制。

**📈 对比分析**

与现有方法对比，InfiniVerse在单视角/多视角下均获得最低FID（6.40）和FVD（67.97），在长时序生成中退化速率最慢，显著优于Vista、MagicDrive等基线。

**⚠️ 局限性**

局限性：1）对稀疏/偏离视角的输入可能导致占据重建不完整；2）训练成本高（多阶段、GPU日数巨大）；3）仍在某些极端天气或复杂动态物体场景下易产生细节失真。

---

## 233. Efficient Sim-to-Real Transfer of World-Action Models from Synthetic Priors

**arXiv ID:** 2606.31101 | [PDF](https://arxiv.org/pdf/2606.31101v1)

**作者:** Zixing Wang `[一作]` (Purdue University), Karl Schmeckpeper `[通讯]` (Robotics and AI Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了仅用合成演示训练的世界-动作模型能否实现零射门到真实机器人上的操控

**💡 创新点**

首次实现世界-动作模型从纯合成数据到真实机器人的零射门转移，为模拟到真实学习开辟新的模型类别

**🔧 技术方法**

采用 Cosmos Policy（预训练视频扩散模型转化为策略）、大规模域随机化、AnyTask 自动演示生成和联合视频-动作预测框架

**📊 数据集**

约800个合成演示/任务（共约3,200张RGB序列与末端执行器轨迹），来自 AnyTask 的基于 ViPR 的自动化演示生成

**📈 对比分析**

与 Diffusion Policy 在10/50真实演示对照，零射门模型在四项任务上平均成功率为35%，显著高于DP10（5%）且超过DP50（25%），证明了合成数据可实现可观的真实性能

**⚠️ 局限性**

成功率仍偏低，仅在四项简单任务上验证；未在多样化真实环境或长时序任务中评估；缺乏因子 ablation，难以厘清各贡献因素；对复杂视觉和物理差异的鲁棒性尚待进一步验证

---

## 234. Seeing Through Multiple Views: Parameter-Efficient Fine-Tuning via Selective Neurons for Consistent Radiology Report Generation

**arXiv ID:** 2606.31099 | [PDF](https://arxiv.org/pdf/2606.31099v1)

**作者:** Yucheng Chen `[一作]` (MedVisAI Lab), Si Yong Yeo `[通讯]` (MedVisAI Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 View-PNDF 框架，利用视角特定的神经元检测与微调，提升多视角胸X光报告生成的语义一致性与诊断准确性。

**💡 创新点**

创新点在于从神经元层面识别并只微调占参数不足 5% 的视角特定神经元，实现参数高效、跨视角一致性的改进。

**🔧 技术方法**

采用 Transformer 解码器中的神经元归因与扰动检测、视角验证模块，并使用仅更新特定神经元的微调策略；生成报告后通过 LLM（如 GPT‑4o）进行整合与评估。

**📊 数据集**

在 IU‑XRay 和 MIMIC‑CXR 两个包含前后位和侧位胸片的公开数据集上进行实验。

**📈 对比分析**

与 MedGemma、InternVL、LLaVA、Hulu 等多种 SOTA VLM 以及传统 NLG 指标（BLEU、METEOR、ROUGE‑L）和 LLM 评估（GPT‑4o）对比，View‑PNDF 在所有指标上均取得显著提升，尤其在侧位视角的诊断准确性提升最为明显。

**⚠️ 局限性**

局限性包括：仅在胸X光上验证，尚未扩展到 CT/MRI 等其他模态；对更复杂多视角情境的适配仍需研究；LLM 评估对算力与成本要求较高。

---

## 235. Can Tabular In-Context Learners Generalize to Biomolecular Property Prediction?

**arXiv ID:** 2606.31126 | [PDF](https://arxiv.org/pdf/2606.31126v1)

**作者:** Davy Guan `[一作]` (Commonwealth Scientific and Industrial Research Organisation), Daniel M. Steinberg `[通讯]` (Commonwealth Scientific and Industrial Research Organisation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究将表格领域的上下文学习模型（TabPFN3、TabICL）与预训练的生物分子表征（蛋白质的ESMC嵌入、小分子的ECFP/ RDKit 指纹）结合，用于在标签稀缺的蛋白质适应性预测和小分子属性分类任务上进行高效学习。

**💡 创新点**

创新点在于：①首次系统评估表格 ICL 模型在蛋白质与小分子两个完全不同任务域的迁移效果；②证明在固定的强表征空间（如ESMC）下，上下文学习模型可以在极少样本（8-64 个标签）甚至全量训练下与传统监督基线竞争；③提出以“学习器-表征”对为评估单位，避免单纯归因于模型结构。

**🔧 技术方法**

使用技术包括：表格 ICL 框架（TabPFN3、TabICL）；蛋白质语言模型 ESMC；小分子描述符 ECFP Fingerprint、RDKit 2D 特征；基线模型 Ridge、RBF Sampler、HistGradientBoostingRegressor、Fine‑tuned ESM 及 XGBoost；图模型 ChemProp、ChemProp+CheMeleon。

**📊 数据集**

数据集覆盖：蛋白质域：ProteinGym（217 个 DMS 任务）和 PpEST（1513 酯酶序列）；小分子域：TDC ADMET（13 任务）、MoleculeNet（806 任务）、FS‑Mol（157 任务）以及 DrugOOD（3 OOD 任务）。

**📈 对比分析**

比较方法：对蛋白质用 MSE 与 Spearman 相关系数评估；对小分子用 ROC‑AUC 评估；在不同数据划分（随机、模数、连续）和样本规模（8–64 或全量）下绘制学习曲线。结果显示：在蛋白质任务中 TabPFN3 在所有划分上均略优于 TabICL，且与传统监督基线相当；在小分子任务中表格 ICL 与 XGBoost、ChemProp 等模型竞争激烈，最佳组合因数据集、特征和任务而异；总体上表格 ICL 在低样本情况下仍能保持良好性能。

**⚠️ 局限性**

局限性包括：①部分大规模 ProteinGym 任务需 PCA 缓存，限制了完整对比；②随机划分的高分数不代表所有泛化场景，模数和连续划分表现明显下降；③MoleculeNet 中部分 PCBA 任务资源限制导致覆盖不完整；④在极端类别不平衡或单类的评估 split 中 ROC‑AUC 可能未定义；⑤支持集的随机抽样对性能有显著影响，需进一步分析。

---

## 236. One Retrieval to Cover Them All: Co-occurrence-Aware Knowledge Base Reorganization for Session-Level RAG

**arXiv ID:** 2606.31156 | [PDF](https://arxiv.org/pdf/2606.31156v1)

**作者:** Shivam Ratnakar `[一作]` (University of Southern California), Chaya Vijayakumar `[通讯]` (University of Southern California)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对RAG系统进行会话级检索改进，离线重组知识库并在查询时通过簇扩展提升检索覆盖率。

**💡 创新点**

提出基于共现的离线聚类与在线簇扩展的预检索重组策略，强调会话级覆盖而非单查询召回。

**🔧 技术方法**

使用Word2Vec训练共现嵌入，层次聚类生成簇，Hybrid检索+扩展+重排序。

**📊 数据集**

主要使用WixQA企业知识库（6,221篇文章）以及跨域电子商务支持数据集。

**📈 对比分析**

相较于Vanilla RAG和交叉编码重排序，在k=8时会话覆盖率提升约17%-21%（全场景58% vs 41%），调用次数降低34%。

**⚠️ 局限性**

共现序列来源合成，未使用真实用户日志；会话定义基于簇，缺乏真实多轮会话评估；未评估生成质量。

---

## 237. SeKV: Resolution-Adaptive KV Cache with Hierarchical Semantic Memory for Long-Context LLM Inference

**arXiv ID:** 2606.31145 | [PDF](https://arxiv.org/pdf/2606.31145v1)

**作者:** Amirhossein Abaskohi `[一作]` (University of British Columbia), Yuhang He `[通讯]` (Microsoft Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种分辨率自适应的语义 KV 缓存（SeKV），通过熵引导的语义段划分、GPU 上的轻量级摘要与 CPU 上的低秩 SVD 基底，动态在推理时按需恢复完整 token 级别信息；

**💡 创新点**

创新点在于：①熵驱动的语义段划分，自动定位语义转折点；②双层存储（GPU 摘要 + CPU SVD 基底）实现可扩展的内存层次；③训练的 zoom‑in 机制，在解码过程中仅按查询相关性按需展开段；④仅训练极少量参数（≈0.05%），保持 LLM 冻结；

**🔧 技术方法**

使用的技术包括：token 预填充时计算的 token 熵、基于熵的段界定、针对每层头的投影权重生成摘要向量、低秩 SVD 分解与门控重构、基于 sigmoid 的 span 相关性门、梯度截断（straight‑through）训练、蒸馏损失、重构与预算正则化等；

**📊 数据集**

训练数据为 RedPajama（arXiv、书籍、代码子集），评估使用四个长文本理解基准：LongBench、RULER、InfiniteBench、Needle‑in‑a‑Haystack（NIAH），以及 GSM8K 作为多提示推理测试；

**📈 对比分析**

在 GPU KV 预算 10% 的条件下，SeKV 在所有五个模型（Llama‑3.2‑3B、Llama‑3‑8B、Llama‑3.1‑8B、Mistral‑7B、Qwen2.5‑14B）与四个基准上平均提升 5.9% 的性能，并在检索密集任务中获得 5.68 点、3.63 点等显著提升；同时 GPU 内存使用比完整 KV 低 53.3%，延迟略优于全 KV，速度与吞吐在语义压缩方法中处于最优位置；

**⚠️ 局限性**

局限性包括：对熵分段质量敏感，分段错误会导致内存分配失效；需手动调节熵阈值、锚点数量、SVD 低秩阈值、预算等超参数；依赖 CPU‑GPU 带宽，长距离多查询会增加延迟；冻结 LLM 可能限制对压缩结构的内部适应。

---

## 238. JacobianAvatar: Temporally Consistent Semi-rigid Avatar Reconstruction from a Monocular Video

**arXiv ID:** 2606.31115 | [PDF](https://arxiv.org/pdf/2606.31115v1)

**作者:** Changyeon Won `[一作]` (GIST), Hae-Gon Jeon `[通讯]` (Yonsei University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文基于单目视频重建半刚体化身，利用神经雅可比场与屏蔽泊松求解捕捉全局与局部变形，并通过残差光流与SDF正则化实现时序一致的高精度几何与渲染；

**💡 创新点**

创新点在于：①引入层次化神经雅可比场实现粗细级别变形；②屏蔽泊松求解结合SDF梯度正则化抑制单目稀疏观测导致的拉伸与翻转；③残差光流损失强化跨帧一致性，整体显著提升几何与渲染质量；

**🔧 技术方法**

使用技术包括：神经雅可比场（NJF）、线性混合蒙皮（LBS）、屏蔽泊松求解、三平面特征编码、3D高斯斑点（3DGS）、预训练光流网络、SDF梯度正则化、双向拉普拉斯正则化、感知损失（LPIPS）等；

**📊 数据集**

实验数据集涵盖 MonoPerfCap、SynWild、NeuMan、DNA-Rendering 四大公开数据集，覆盖自然与合成、室内外、紧贴与宽松衣物等多样场景；

**📈 对比分析**

与 Vid2Avatar、LSAvatar、FacAvatar、ExAvatar、GoMAvatar 等 SOTA 方法在 PSNR/SSIM/LPIPS 及几何指标（Chamfer、Normal Error、F1）上进行对比，结果显示在 MonoPerfCap 上 PSNR 31.83、SSIM 0.978、LPIPS 1.67，优于对手；在 DNA-Rendering 与 NeuMan 数据集也保持竞争力；在 SynWild 的几何指标上 CD 2.46、NE 0.091、F1@1cm 0.397、F1@2cm 0.681，达到最优；

**⚠️ 局限性**

局限性包括：难以重建松散衣物；受限于模板模型 LBS 权重，无法捕捉与内体运动不同的衣物动力学；未考虑面部表情；在极端自遮挡或单一视角下仍可能出现几何不精确现象。

---

## 239. Horizon3D: Sparse Radar-Camera Fusion for Long-Range 3D Perception in Autonomous Driving

**arXiv ID:** 2606.31096 | [PDF](https://arxiv.org/pdf/2606.31096v1)

**作者:** Geonho Bang `[一作]` (Seoul National University), Jun Won Choi `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Horizon3D 框架，利用稀疏雷达‑相机融合实现高速公路长距离 3D 目标检测。

**💡 创新点**

创新点包括：① 用关键点引导的高斯原语初始化 (KGGI) ；② 通过对象中心稀疏融合 (OCSF) 将高斯与稀疏 BEV 结合；③ 双路径时间融合 (DPTF) 同时处理场景级累积与对象级运动补偿。

**🔧 技术方法**

采用的技术包括：高斯原语编码、变形交叉注意力、稀疏卷积、稀疏 BEV 投影、速度指导的时间对齐、以及自监督占据分布。

**📊 数据集**

使用的基准数据集为 TruckScenes，包含高速、150 m 以上的长距离雷达‑相机数据。

**📈 对比分析**

与 BEVFusion、CRT‑Fusion、RCTrans、SpaRC 等方法对比，Horizon3D 在验证集上相对最佳雷达‑相机方法提升 NDS +3.0 与 mAP +1.6，速度约 8.5 FPS，且在测试集上进一步领先。

**⚠️ 局限性**

局限性在于对 4D 雷达的依赖、极远距离稀疏雷达返回仍可能导致检测缺失、以及高斯优化与稀疏 BEV 交互所带来的计算和调参复杂度。

---

## 240. HSDF-Lane: Height-Aligned Signed Distance Field with Semantic Lane Prior for 3D Lane Detection

**arXiv ID:** 2606.31172 | [PDF](https://arxiv.org/pdf/2606.31172v1)

**作者:** Jiyong Boo `[一作]` (KAIST), Kuk-Jin Yoon `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于高度对齐符号距离场（HSDF）和语义路标先验的单目 3D 车道检测框架 HSDF‑Lane，能够同时准确估计道路高度和 3D 车道。

**💡 创新点**

创新点包括：①使用密集垂直采样并用 HSDF 替代稀疏坡度锚点，实现连续地形建模；②引入 Lane‑aware Semantic Positional Encoding (LSPE)，把车道存在热图作为语义先验注入 Transformer；③采用高度方向 Eikonal 正则化以稳定 HSDF 表征。

**🔧 技术方法**

技术手段包括 ResNet‑50 前端特征提取、密集 3D 采样+MLP 预测 HSDF、可微渲染生成高度图和表面对齐特征、Transformer 解码器配合 Deformable Cross‑Attention、LSPE 语义编码、Eikonal 正则化与多项损失共同训练。

**📊 数据集**

在 OpenLane（Waymo Open 数据集）基准上训练和评估，使用约 200,000 帧及 880,000 条 3D 车道标注；道路高度图来自 HeightLane 的 LiDAR 生成密集高度图。

**📈 对比分析**

与 PersFormer、BEV‑LaneDet、LATR、PVALane、GroupLane、HeightLane、SC‑Lane、Rethinking、GLane3D‑B、SparseLaneSTP 等基线对比，HSDF‑Lane 在 OpenLane 上取得最高 F1 分数、最佳远程 X‑误差，且在道路高度估计方面 MAE、RMSE、阈值精度均优于 HeightLane 与 SC‑Lane，带 FPN 的 HSDF‑Lane^† 更进一步提升性能。

**⚠️ 局限性**

局限性：仅在固定 BEV 网格上预测高度，未能识别可行驶路面；训练需要依赖 LiDAR 生成的密集高度图；仅单目单视角，未包含多视角、时间序列或多模态扩展。

---

## 241. Beyond Single Character: Evaluating MLLMs for Sentence-Level Oracle Bone Inscription Understanding

**arXiv ID:** 2606.31169 | [PDF](https://arxiv.org/pdf/2606.31169v1)

**作者:** Ziqi Li `[一作]` (Nanjing University of Science and Technology), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了S-OBI基准，用句子级别的甲骨文图像评估多模态大语言模型的理解能力

**💡 创新点**

创新点在于通过字符替换合成清晰句子级甲骨文，构建多任务（语义匹配、槽位提取、上下文推理）评估框架

**🔧 技术方法**

采用多模态大语言模型（如GPT‑5.2、GPT‑4o、Gemini‑3.1‑Pro、Qwen系列等）进行图文推理

**📊 数据集**

使用95条经专家校正的原始甲骨文和高质量单字样本进行合成，生成695个问答实例

**📈 对比分析**

与十款主流MLLM对比，最佳整体得分仅33.5%，说明当前模型在句子级甲骨文理解方面仍相当薄弱

**⚠️ 局限性**

局限在于基准规模小、图像合成简化了原始材质、存在多选答案捷径，且未覆盖更大多样性文本

---

## 242. SwiftAudio: Data-Efficient Caption-Only Distillation for One-Step Text-to-Audio Diffusion-based Generation

**arXiv ID:** 2606.31259 | [PDF](https://arxiv.org/pdf/2606.31259v1)

**作者:** Binh Mai `[一作]` (Posts and Telecommunications Institute of Technology), Cong Tran `[通讯]` (Posts and Telecommunications Institute of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过音频无监督蒸馏训练了一个一阶文本到音频生成模型，只使用文本标题而不依赖配对音频-文本数据。

**💡 创新点**

在音频领域首次将Variational Score Distillation与时间平滑正则相结合，实现仅文本标题的高质量一阶生成，并消除对配对训练数据的需求。

**🔧 技术方法**

使用Variational Score Distillation、LoRA教师网络、时间总变差（TV）正则、latent音频扩散以及classifier-free guidance等技术。

**📊 数据集**

训练使用AudioCaps数据集的约45K条文本标题，评估在AudioCaps和Clotho数据集上。

**📈 对比分析**

与多步AudioLDM2、Auffusion以及一阶AudioLCM、ConsistencyTTA比较，SwiftAudio在FAD、FD、IS等指标上为一阶模型中的最佳，且与多步教师性能差距小，并在Clotho上表现出更好的跨域泛化。

**⚠️ 局限性**

只能生成固定10秒的短音频，缺乏对长时间或结构化音频的支持；对语音内容控制有限，无法生成具有可辨识语言文本的语音。

---

## 243. Embodied CAD: Solver-Grounded LLM Agents for Parametric B-Rep Assembly Modeling

**arXiv ID:** 2606.31252 | [PDF](https://arxiv.org/pdf/2606.31252v1)

**作者:** Fumin Liu `[一作]` (Nanjing University), Lin Yang `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一个闭环框架，使大型语言模型通过连续的技能调用、解析与CAD核算实现可编辑的参数化工业装配。

**💡 创新点**

创新点在于把CAD建模转化为solver为基准的技能轨迹问题，设计L0–L4技能层级与操作族解析，利用CAD求解器反馈实现在线修复与强化学习。

**🔧 技术方法**

使用LLM规划器、确定性解析器、FreeCAD后端、GRPO强化学习以及求解器反馈奖励来驱动决策与执行。

**📊 数据集**

利用公开的工业装配任务数据集，包括机械装配、工业设备与模具插入等任务，配合FreeCAD进行仿真。

**📈 对比分析**

通过执行率、技能准确率、操作族准确率、完整任务完成率等solver对齐指标评估，强规划可达100%执行，弱规划SFT/GRPO约93%精确，直接族预测约77%。

**⚠️ 局限性**

局限性在于需为新领域定义新的技能与解析器，受限于机械/模具参数化任务，难以处理自由曲面、容差堆叠与后续制造仿真；强化学习仍难以捕捉几何意义的奖励差异。

---

## 244. HyperVLP: Enhancing Hierarchical Surgical Video-Language Pre-training in Hyperbolic Space

**arXiv ID:** 2606.31245 | [PDF](https://arxiv.org/pdf/2606.31245v1)

**作者:** Yaojun Hu `[一作]` (Zhejiang University), Nicolas Padoy `[通讯]` (University of Strasbourg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出HyperVLP框架，利用双模态（视觉+文本）在洛伦兹超曲面中进行层级化预训练；

**💡 创新点**

创新点在于在负曲率空间中引入几何感知的超曲面对比学习以及锥形蕴涵约束，解决结构性负样本和层级包含关系；

**🔧 技术方法**

核心技术包括Lorentz超曲面映射、几何感知超曲面对比损失、锥形超曲面蕴涵损失和两阶段渐进式优化；

**📊 数据集**

使用Surgical Video Lectures（SVL）做预训练，评估在Cholec80、AutoLaparo、MultiBypass140（Stras70、Bern70）等公开手术数据集；

**📈 对比分析**

与CLIP、SurgVLP、HecVL、PeskaVLP等方法对比，HyperVLP在零样本和少样本阶段的阶段识别准确率和F1分数均显著提升，尤其跨中心表现突出；

**⚠️ 局限性**

局限在于超曲面训练的数值不稳定、曲率调参复杂，且对极大层级结构或其他领域的迁移性尚未充分验证。

---

## 245. Agentic-Ideation: Sample Efficient Agentic Trajectories Synthesis for Scientific Ideation Agents

**arXiv ID:** 2606.31229 | [PDF](https://arxiv.org/pdf/2606.31229v1)

**作者:** Keyu Zhao `[一作]` (Tsinghua University), Yong Li `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Agentic‑Ideation 框架，通过 oracle‑guided 多代理系统高效合成科学创意轨迹，并用这些轨迹训练能够自主调用工具的 agentic LLM，从而实现开放式科研创意生成。

**💡 创新点**

① 利用 oracle 指导的轨迹合成显著提升样本效率（10×）；② 设计混合工具空间（外部检索 + 内部思考工具）支持复杂推理；③ 在监督微调时对外部工具返回值进行掩码，确保模型学习决策逻辑而非记忆反馈。

**🔧 技术方法**

多代理架构（Planner、Controller、工具代理）、Oracle‑guided 轨迹合成、工具结果掩码的监督微调、以及大语言模型训练。

**📊 数据集**

使用科研文献数据库（如 arXiv/学术检索 API）以及人工构造的高质量参考创意集合来合成轨迹。

**📈 对比分析**

与现有基于预定义工作流的基线相比，整体质量提升 11.91%，轨迹合成效率提升 10×；在人类评测中生成的创意在新颖性、可行性和科学价值上均优于基线。

**⚠️ 局限性**

局限性包括：依赖高质量参考创意作为 oracle，若 oracle 不易获取会影响合成效率；对外部检索工具的性能和覆盖范围有限；未在极其开放或跨学科领域进行充分验证。

---

## 246. Securing the AI Agent: A Unified Framework for Multi-Layer Agent Red Teaming

**arXiv ID:** 2606.31227 | [PDF](https://arxiv.org/pdf/2606.31227v1)

**作者:** Yong Yang `[一作]` (Tencent Zhuque Lab), Zonghao Ying `[通讯]` (Tencent Zhuque Lab)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个完整的 AI 代理安全评估框架 AI‑Infra‑Guard，覆盖基础设施、协议/工具、代理行为和模型四层攻击面；实现了基于规则的扫描、LLM 驱动的代码与协议审计、黑盒多轮红队和对抗性 jailbreak 评估，并加入了技能供应链审计。

**💡 创新点**

提出“层‑范式匹配”原则，将不同层次的安全风险与最合适的检测范式（规则匹配、LLM 语义审计、多轮交互、统计评估）对应；首次将 LLM 作为领域审计器、Prompt‑as‑Rule 规则化、以及对审计器自身进行防注入设计；在同一框架下集成技能供应链审计与模型对抗评估。

**🔧 技术方法**

技术包括：1) 规则引擎（Go）用于 AI 指纹与 CVE 匹配；2) LLM 驱动的 agentic harness（Python + 多模型路由）实现 MCP 服务器与技能静态/动态审计；3) 交互式多轮红队架构，采用 Prompt‑as‑Rule 任务规范；4) jailbreak 评估 harness，组合 16 个数据集、数十种攻击算子、三种角色（模拟器、目标、判定器）；5) 统一的服务器‑代理分布式调度与结果流式返回；6) 使用多模型（Claude, Gemini, GLM 等）与外部威胁情报 API。

**📊 数据集**

使用 75 种 AI 组件的指纹规则、1,443 条漏洞规则；SkillTrustBench（5,520 个真实技能案例）进行技能审计基准；16 个 jailbreak 数据集（约7,250 条攻击提示）用于模型层评估；此外使用公开的 CVE、MCP 插件与工具清单、模型对抗实验的公开数据集。

**📈 对比分析**

通过对比现有工具（Nuclei, Semgrep, MCPSafetyScanner, garak 等）展示 AI‑Infra‑Guard 在四层覆盖率方面的唯一性；在 SkillTrustBench 上，基于不同 LLM 的技能审计 F1 超过 0.98，召回率接近 1；在 jailbreak 评估中，展示多模型成功率曲线，证明对抗算子组合带来的显著提升；性能方面，基础设施扫描极快，LLM 审计与红队耗时在数分钟到数十分钟不等。

**⚠️ 局限性**

局限性：1) 需要多种 LLM 及其费用，导致运维成本高；2) 规则与 Prompt‑as‑Rule 需要持续维护，易受新型漏洞与协议变更影响；3) 现有扫描多聚焦已知协议与工具，未覆盖所有可能的第三方插件；4) 黑盒红队受交互预算限制，可能漏检隐蔽攻击；5) 对于模型层的统计评估，成功率不代表绝对安全，仍需人工验证；6) 系统在极大规模部署时的可扩展性与资源调度仍待进一步实验。

---

## 247. Probing Memorization of Tabular In-Context Learning

**arXiv ID:** 2606.31208 | [PDF](https://arxiv.org/pdf/2606.31208v1)

**作者:** Francesco Capano `[一作]` (SAP SE), Jonas Böhler `[通讯]` (SAP SE)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大型表格模型的记忆行为进行系统评估，并提出了一种基于零信息多选框架的探测方法

**💡 创新点**

创新点在于将多选上下文与难度校准相结合，构造可区分上下文推理与参数记忆的探测框架，并在受控 fine‑tuning 下建立真正的成员身份标注

**🔧 技术方法**

使用零信息多选上下文、上下文扰动、难度校准以及 AUC/TPR 等评估指标，并对比基线预训练模型进行分析

**📊 数据集**

在 ConTextTab 基准上评估了来自 CARTE 的 10 个多样化分类与回归任务

**📈 对比分析**

与基线预训练模型对比，检测到 8/10 任务存在可检测的记忆信号（AUC最高 0.67，TPR@1%>0.1），但在更实用的训练设置下信号显著减弱

**⚠️ 局限性**

局限性包括记忆信号相对较弱、评估场景人为且单一任务/固定查询上下文、仅针对一个模型与 10 个任务，且未考虑多任务联合训练的效果

---

## 248. ExPLoRe: Expert Patch-Level Loss Routing for Multi-Objective Masked Image Modeling

**arXiv ID:** 2606.31201 | [PDF](https://arxiv.org/pdf/2606.31201v1)

**作者:** Konstantinos Georgiou `[一作]` (University of Tennessee), Hairong Qi `[通讯]` (University of Tennessee)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种在多目标掩码图像建模中对每个补丁进行自适应损失加权的方法ExPLoRe；

**💡 创新点**

创新点在于将Soft-MoE的dispatch权重重用为每个补丁的损失系数，并通过loss‑coupling实现内容依赖的专家专门化；

**🔧 技术方法**

使用Soft-MoE（Soft Mixture of Experts）、CLIP教师、token distillation、CLS对齐、像素重构等多目标损失，结合entropy正则化、冻结路由/注意力、专家dropout等后处理策略；

**📊 数据集**

主要在ImageNet-1K进行预训练和评估，随后在ADE20K上进行语义分割验证；

**📈 对比分析**

与现有多目标MIM与SOTA方法（如MAE、BEiT v2、MILAN、CAE v2等）对比，2‑expert模型在linear probe上达79.6%（比MaskDistill提升3.4%），64‑expert模型在linear probe上达80.6%并在finetuning上达到85.3%（与CAE v2相当），语义分割在完整调优后可与非MoE基线相匹配；

**⚠️ 局限性**

受限于ViT‑Base规模与GPU内存，最多支持64个专家，且需要专门的finetune recipe才能在下游任务中发挥优势，且机制在更大模型或不同教师设置下的泛化尚未验证。

---

## 249. Transformers as Bayesian In-Context Experimenters: Smoothness-Adaptive Efficient ATE Estimation

**arXiv ID:** 2606.31184 | [PDF](https://arxiv.org/pdf/2606.31184v1)

**作者:** Jiachun Li `[一作]` (MIT), David Simchi-Levi `[通讯]` (MIT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

利用变换器训练可模拟贝叶斯教师的顺序实验者，实现自适应的实验设计与后续ATE估计；

**💡 创新点**

提出了一种“可训练的实验者”框架，通过预训练Transformer来模仿贝叶斯实验者，实现自适应设计与半参数高效性；

**🔧 技术方法**

使用ReLU注意力Transformer、岭回归、混合专家模型、可解释的Neyman头以及双机器学习/ AIPW 估计；

**📊 数据集**

主要在模拟实验中生成的数据；

**📈 对比分析**

与传统的随机试验、固定随机实验设计和非自适应方法比较，显示在ATE估计上可达到效率极限，实验者的自适应策略在样本量足够时显著优于传统方法；

**⚠️ 局限性**

限制在于仅考虑二元干预且假设无混杂，且在大样本/高维情况下模型可能需大量预训练样本和计算资源。

---

## 250. HealthAgentBench: A Unified Benchmark Suite of Realistic Agentic Healthcare Environments for Challenging Frontier AI Agents

**arXiv ID:** 2606.31179 | [PDF](https://arxiv.org/pdf/2606.31179v1)

**作者:** Qianchu Liu `[一作]` (Microsoft Research), Hoifung Poon `[通讯]` (Microsoft Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了一个名为 HealthAgentBench 的统一、面向代理的医疗保健基准套件，涵盖多模态（CT、X‑ray、全景病理切片、文本记录、结构化 EHR）和临床工作流（数据管理、诊断、研究、治疗规划）的任务，并通过终端式环境让 LLM 代理在真实患者数据上自主规划、调用工具、探索并完成多步骤任务。

**💡 创新点**

创新点包括：①将多模态临床数据与真实工作流集成到可执行的终端环境；②提出一套严格的任务选取与验证流程，确保任务既真实又不易被捷径作弊；③提供统一的二元成功判定与可解释的整体成功率指标，使得不同模型能在同一平台下公平比较；④展示了前沿 LLM（GPT‑5 系列 vs Claude 系列）在该基准上的系统性差异，揭示医学影像、海量检索与组合推理是当前最大的瓶颈。

**🔧 技术方法**

技术上主要使用：终端框架 Harbor、OpenSlide、数据库查询工具、图像处理工具、自动机器学习流水线（ETL 与模型训练）以及多代理架构（Copilot‑CLI）。评估模块将每个任务的输出与专家标签或人类表现进行二元判定。

**📊 数据集**

数据集涵盖：MIMIC‑IV（注入错误版）、EHRSHOT（可解释预测模型）、真实 CT、X‑ray、病理全景切片、临床试验协议文本、以及通过规则生成的错误注入任务。所有数据在运行时动态下载，避免直接分发。

**📈 对比分析**

对 10 个前沿代理（GPT‑5.5、GPT‑5.4、GPT‑5.3、GPT‑5.2、Opus‑4.8、Opus‑4.7、Opus‑4.6、Claude‑Code‑Sonnet‑4.6 等）在 162 次实验中进行评估。最佳模型 Codex GPT‑5.5 的整体任务成功率约为 42%，其次是 Copilot‑CLI+Opus‑4.8（≈36%）和 Codex GPT‑5.5（≈35%）。在医学影像任务上，GPT‑5.5 仅能达到 35% 的成功率；在检索与组合推理任务中，所有模型均低于 50%。成本和耗时方面，GPT‑5 系列在成本-效能 Pareto 前沿上占优，而 Claude 系列往往更慢且更昂贵。

**⚠️ 局限性**

局限性：①基准任务仍相对有限，未覆盖所有医疗领域与工具；②医学影像和大规模检索仍是难点，现有 LLM 对此表现不佳；③部分任务需要更专业的视觉或数据库工具，单纯靠通用 LLM 无法突破；④虽然通过“anti‑cheat”设计减少了作弊，但对真实临床系统的可迁移性和部署可行性尚需进一步验证。

---

## 251. Machine Learning-based Feedback Linearization Control of Quadrotor Subject to Unmodeled Dynamics

**arXiv ID:** 2606.31199 | [PDF](https://arxiv.org/pdf/2606.31199v1)

**作者:** Amos Alwala `[一作]` (University of Turku), Wallace Moreira Bessa `[通讯]` (University of Turku)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于在线高斯RBF神经网络的反馈线性化控制器，用于在存在未知动力学和扰动的四旋翼无人机轨迹跟踪。

**💡 创新点**

创新点在于将在线学习的RBF网络与Lyapunov稳定性分析相结合，实现对未建模空气阻力、执行器动态等非线性扰动的实时补偿，提升跟踪精度。

**🔧 技术方法**

使用了高斯径向基函数神经网络、Lyapunov控制理论、SO(3)几何控制、ROS2 与 Gazebo 仿真环境，以及 Crazyflie 2.1 实机飞行。

**📊 数据集**

数据集采用仿真中生成的螺旋轨迹以及真实 Crazyflie 2.1 的飞行记录，没有使用公开数据集。

**📈 对比分析**

通过与传统纯反馈线性化控制器对比，实验证明智能控制器在模拟中平均位置误差降低约34.8%，在真实飞行中位置误差下降7.13%，偏航误差下降49.27%。

**⚠️ 局限性**

局限性包括仿真时无法直接施加滚转俯仰率导致姿态跟踪受限，且系统对学习率和权重上界的敏感性较高，可能导致噪声放大和旋转激进。

---

## 252. AC$^2$P$^2$SL: Adaptive Communication-Computation Pipeline Parallel Split Learning over Edge Networks

**arXiv ID:** 2606.31276 | [PDF](https://arxiv.org/pdf/2606.31276v1)

**作者:** Chenyu Liu `[一作]` (Zhejiang University), Tony Q. S. Quek `[通讯]` (Singapore University of Technology and Design)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在无线边缘网络中提出AC^2P^2SL框架，通过通信与计算的流水线并行以及微批分块来显著降低分割学习的训练时延。

**💡 创新点**

创新点在于将通信与计算统一成流水线实现微批级并行，结合分割点与资源的SPA预分配与ARA自适应重分配算法，有效缓解异构UE和动态信道导致的同步瓶颈。

**🔧 技术方法**

采用分割学习、U形分割、流水线并行、时间分割多址、Roofline计算模型、MINLP求解、动态规划、交替优化、投影梯度、CVXPY与整数取整技术。

**📊 数据集**

实验使用 ImageNet-100 数据集，并在 ResNet18/50/101 及 Vision Transformer 模型上进行评估。

**📈 对比分析**

通过与多种SL/USL基线（PSL、SFL、EPSL、USL、UPSL、USFL、HFSL、CL）以及不同带宽、UE数和时隙比例等场景下的单轮训练时延对比，AC^2P^2SL 在所有模型上平均缩短 60% 以上训练时间，甚至在 ViT 上优于集中式训练。

**⚠️ 局限性**

局限性包括对 TDMA 时隙划分的依赖、对极端动态通道和设备故障的鲁棒性有限、实现复杂度高、对大规模 UE 的管理开销未充分评估。

---

## 253. The Calibration Turn in AI-Assisted Research: A Conceptual and Methodological Framework for Evidence-Licensed Claims

**arXiv ID:** 2606.31273 | [PDF](https://arxiv.org/pdf/2606.31273v1)

**作者:** Hongmin Li `[一作]` `[通讯]` (Institute of Science Tokyo), Hongmin Li (Institute of Science Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套面向 AI 辅助科研的证据校准框架，定义了证据许可语义、claim‑evidence gap、epistemic debt 等概念，并将 AI 科学流程拆解为五个操作（假设生成、后果推导、外部验证、信念更新、主张校准）。

**💡 创新点**

创新点在于：①将证据视为主张许可的“执照”而非单纯的可信度衡量；②引入校准算子 _，将原始主张映射到最大允许的证据许可范围；③提出 claim‑evidence gap 和 epistemic debt 作为诊断工具；④用七条代表性 AI 科学路线（专业基础模型、LLM 助手、多智能体共科学家、端到端 AI 科学家、算法/证明代理、自动实验室、混合校准循环）做方法学比较；⑤通过 AISim‑Cal 进行合成动力学演示，展示校准缺失时的过度声明风险。

**🔧 技术方法**

主要技术手段为：形式化语义与符号逻辑（定义 E⊩_,V C 等关系）、多层级理论框架（科学意义、许可关系、校准算子、动态模型）、仿真模拟（AISim‑Cal）以及对现有 AI 科学系统的案例式阐述。

**📊 数据集**

论文不基于单一数据集，而是利用现有 AI 科学系统（AlphaFold3、GPT‑5、Google Co‑Scientist、AI Scientist、AlphaEvolve、A‑Lab 等）的公开实验结果、文献与示例来演示框架；AISim‑Cal 使用人工设定的参数进行合成实验。

**📈 对比分析**

比较方法：通过定义同一框架下的七条 AI 科学路线，提取其生成、验证、校准等指标，并在 AISim‑Cal 中模拟不同配置下的 licensed‑utility、over‑claim gap、false‑discovery burden 等指标。模拟结果显示：缺失校准算子时 over‑claim gap 最大；完善校准可显著降低 epistemic debt。实际论文中未给出真实实验性能数值，性能评价仅限于仿真诊断与理论推导。

**⚠️ 局限性**

局限性：①缺乏实证验证，所有结论基于理论推导和合成模拟；②框架高度依赖领域特定的证据评判标准和主张层级，跨领域应用需进一步规范；③校准算子 _ 的具体实现尚未给出统一算法，实际工程化仍需探索；④对复杂数据流、非结构化证据（如实验视频、文本推理结果）的处理仍待完善。

---

## 254. Learning from Failure: Inference-Time Self-Improvement for Computer-Use Agents

**arXiv ID:** 2606.31270 | [PDF](https://arxiv.org/pdf/2606.31270v1)

**作者:** Xueqiao Sun `[一作]` (Stanford University), Yuhui Zhang `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种在推理时利用失败轨迹进行自我改进的循环，以提升计算机使用代理的表现。

**💡 创新点**

创新点在于将失败案例视为可利用的训练信号，通过LLM诊断失败模式并生成代码补丁，实现无需再训练即可提升代理性能。

**🔧 技术方法**

主要技术包括大语言模型（Claude 4.5 Sonnet）作为元控制器进行失败诊断和补丁生成、视觉搜索、终端执行、知识检索、重复循环警告等策略，以及轻量级人工验证。

**📊 数据集**

使用 OSWorld 验证环境，并在其小规模和全规模任务上进行评估；同时对 OmniACT、AndroidControl、ScreenSpotPro、WebVoyager 等基准进行跨任务迁移实验。

**📈 对比分析**

与基准 OpenCUA‑72B 进行对比，成功率从 42.3% 提升至 48.9%（+6.6 点，+15.6% ），在不增加训练成本、仅约 8% 运行时开销的前提下实现提升；同样的改进在不同规模、不同来源模型上均可观测到 10–12% 的相对提升。

**⚠️ 局限性**

局限性包括：依赖大语言模型的推理质量；补丁需要人工轻量化验证；对高层认知、任务理解和视觉解析等更深层次问题提升有限；对极端动态或复杂 UI 的适用性尚未充分验证。

---

## 255. Decodable Is Not Grounded: A Vision-Ablation Arbiter for VLM Spatial Reasoning

**arXiv ID:** 2606.31257 | [PDF](https://arxiv.org/pdf/2606.31257v1)

**作者:** Chih-Ting Liao `[一作]` (University of New South Wales), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对视觉语言模型（VLM）空间推理进行诊断，发现传统线性探测与无梯度干预往往高估模型对图像的真实依赖，并揭示存在“倒置”与“先验”两种非视觉知识模式；

**💡 创新点**

提出了低成本、无监督的“视觉消融仲裁器”作为判别工具，构建了三类知识地位分类法（真实视觉、倒置、先验），并证明该方法在14种不同规模和架构的VLM上普适；

**🔧 技术方法**

使用线性探测（PCA-50、5折CV逻辑回归）、无梯度投影干预（top‑k权重放大）、视觉消融仲裁器（灰白消除与五种噪声对照）、以及多种纠正技术（旋转、低秩编辑等）来验证和修正倒置；

**📊 数据集**

主要采用ViewSpatial‑Bench进行空间方向测试，并在What'sUp、3DSRBench 等外部基准上验证任务类型边界；

**📈 对比分析**

与传统探测+干预方法相比，仲裁器能明确区分视觉驱动、倒置与先验三种情况；在多数模型中，探测准确率可达90%+但实际行为仅接近或低于随机，显示过度估计；修正技术能将倒置轴的准确率提升至70%+；

**⚠️ 局限性**

局限性包括：倒置的出现随模型规模非单调；部分小型模型无法实现全部规律；因缺乏像素级对照，因果证据仅在表示层级；未深入探究训练数据对倒置形成的具体机制。

---

## 256. Rethinking the Role of Feature Engineering and Learning Strategies in Few-Shot Hidden Emotion Recognition

**arXiv ID:** 2606.31249 | [PDF](https://arxiv.org/pdf/2606.31249v1)

**作者:** Xiaochuan Guo `[一作]` (Hefei University Of Technology), Dan Guo `[通讯]` (Hefei University Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于基准‑偏移交叉注意力的多模态时序框架，并结合 MIL 自适应池化和迭代弱监督训练，解决少量标注下的隐性情绪识别问题。

**💡 创新点**

创新点：① 将运动差分（Offset）与静态姿态（Base）通过交叉注意力融合，首次在此任务中引入 MIL 池化抑制背景噪声；② 对伪标签的“伪泛化”风险进行系统分析与对策。

**🔧 技术方法**

技术手段：2D/3D 骨架、面部 Blendshapes、DINOv2/3 视觉基础模型、X-CLIP 视频特征、Gemini 文本语义先验、光流、深度、YOLO+裁剪、Transformer+多头注意力、MIL 池化、弱监督+伪标签迭代与自注意力。

**📊 数据集**

使用数据集：iMiGUE（4K 长视频采访数据）及 EI‑MiGA‑IJCAI 第四届挑战赛公开测试集。

**📈 对比分析**

方法比较：对基线 MLP、单模态、独立预训练等进行消融实验，并对比独立流与基准‑偏移交叉注意力两种融合；最终在 Track 3 取得 0.76923 的测试准确率，排名第一。

**⚠️ 局限性**

局限性：依赖伪标签和公共排行榜易导致“伪泛化”与记忆化，DINOv2/3 在微动任务中易出现表示崩溃；高维几何特征导致维数灾难，缺乏对长尾或域移位的鲁棒性。

---

## 257. FlexiSLM: A Dynamic and Controllable Frame Rate Spoken Language Model

**arXiv ID:** 2606.31247 | [PDF](https://arxiv.org/pdf/2606.31247v1)

**作者:** Jiaqi Li `[一作]` (Chinese University of Hong Kong), Zhizheng Wu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了首个支持动态可控帧率的说话语言模型FlexiSLM，实现了从12.5 Hz降至4.0 Hz的语音输入输出。

**💡 创新点**

引入动态帧率压缩及直接帧率条件控制，使单模型在不同计算预算下无需重新训练即可调节质量。

**🔧 技术方法**

结合FlexiCodec动态编码、Qwen2.5-Omni LLM、帧合并模块、Sinusoidal帧率嵌入、Talker-Thinker互联、LoRA微调及全参数微调。

**📊 数据集**

使用Emilia、MLS、LibriSpeech、Kimi-Audio-Evalkit等多任务语音对话与TTS、ASR数据，构造FlexiSLM-Data对话集。

**📈 对比分析**

在Kimi-Audio-Evalkit上与同规模7B SLM对比，12.5 Hz下获得72.4/67.2分，超过Qwen2.5-Omni；6.25 Hz下保持相近分数并将推理时间减半，显示动态帧率提升效率。

**⚠️ 局限性**

未做RLHF、DPO等后训练，非流式推理，缺乏复杂推理、多轮对话与选择题等场景的数据与评测。

---

## 258. A Multi-Dimensional, Per-Pass Empirical Study of the LLVM Optimization Pipeline

**arXiv ID:** 2606.31238 | [PDF](https://arxiv.org/pdf/2606.31238v1)

**作者:** Federico Bruzzone `[一作]` (Università degli Studi di Milano), Walter Cazzola `[通讯]` (Università degli Studi di Milano)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对LLVM 1.0编译器的完整优化管道进行系统的多维经验研究，将管道拆分为累积的每一步前缀，量化每个优化pass对运行时间、编译时间、二进制尺寸、硬件计数器以及RAPL能耗的边际影响；

**💡 创新点**

首次提供跨多维度（执行时间、编译时间、二进制体积、微架构计数器和能耗）的大规模经验数据，并提出“理想加法上界”式相位干扰损失指标；同时给出小规模Pareto支配的核心pass列表，揭示管道非单调、后加载特性；

**🔧 技术方法**

自定义管道分解器 + 逐前缀运行 + 统计学噪声抑制（多次跑、实时优先级、CPU pinning、禁用ASLR）、中位数/均值、Cohen's d、RAPL、硬件计数器收集、Python/TOML配置

**📊 数据集**

PolyBench/C 4.2.1 的 30 个单线程、默认数据集的计算密集型 kernel；LLVM 21.1.8；Intel Alder Lake（i9‑12900KF）

**📈 对比分析**

通过对 113 个累积前缀进行 84,750 次测量，比较每个pass对 5 个指标的边际贡献；结果显示大多数pass几乎无效，前 20% 的核心pass 提供约 80% 加速；二进制尺寸与加速存在后置抑制，最终配置往往被早期前缀支配；硬件计数器显示加速主要来自指令数和周期数下降，IPC 实际下降；能耗随运行时间下降 30‑60%；

**⚠️ 局限性**

仅覆盖单线程计算密集型工作负载；仅在单一架构和编译器版本上验证；RAPL 能耗不包含 DRAM/GPU；相位干扰损失为上界而非可恢复值；pass 影响排名仅针对原始顺序，无法直接推广到重排；实验规模受 PolyBench 约束，难以推广到更大、非正规或多线程程序

---

## 259. Delta-JEPA: Learning Action-Sensitive World Models via Latent Difference Decoding

**arXiv ID:** 2606.31232 | [PDF](https://arxiv.org/pdf/2606.31232v1)

**作者:** Zhenghao Zhang `[一作]` (University of Chinese Academy of Sciences), Jungang Xu `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无重建的视觉世界模型 Delta‑JEPA，通过在潜在空间中使用潜在差分动作解码（LDAD）来防止特征崩塌并提升规划性能。

**💡 创新点**

创新点是将动作信息直接监督于潜在位移，而非端点拼接，从而实现更稳健、动作敏感的潜在动力学。

**🔧 技术方法**

使用 ViT‑Tiny 编码器、因果 Transformer 动态预测器以及轻量级非因果 Transformer 动作解码器，联合训练潜在预测与动作重建的两目标。

**📊 数据集**

在四个连续控制任务（Push‑T、Reacher、Cube、Two‑Room）上进行评估。

**📈 对比分析**

与 LeWM、Sub‑JEPA、PLDM 等基线对比，Delta‑JEPA 在所有环境中取得最高规划成功率，特别是在 OGB‑Cube 上提升约 15%。

**⚠️ 局限性**

仍然依赖离线无奖励数据，动作解码权重调节敏感，对更复杂多步解码和跨任务泛化的可扩展性待进一步验证。

---

## 260. Thinking Before Retrieving: Robust Zero-Shot Composed Image Retrieval via Strategic Planning and Self-Criticism

**arXiv ID:** 2606.31222 | [PDF](https://arxiv.org/pdf/2606.31222v1)

**作者:** Gunho Jung `[一作]` (Korea University), Seong-Whan Lee `[通讯]` (Korea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 PEC-CIR 的训练‑free 零样本组合图像检索框架，通过规划–执行–批评三阶段推理迭代构造检索查询。

**💡 创新点**

创新点在于将查询构造拆解为可解释的约束规划、候选生成与多准则批评评估，并通过反馈循环纠正生成错误，从而显著降低单次生成的语义漂移和误检。

**🔧 技术方法**

技术包括使用冻结的 Vision‑Language 模型（ViT‑G/14 + CLIP）进行特征编码，利用大型语言模型（如 Gemini 2.0 Flash）实现 Planner、Executor 与 Critic 的自然语言推理和文本生成，以及基于余弦相似度的检索。

**📊 数据集**

使用公开的 FashionIQ（细粒度属性修改）和 CIRR（场景级变换）两个基准数据集进行实验。

**📈 对比分析**

与现有训练‑free 方法相比，PEC‑CIR 在 FashionIQ 上 Recall@10 提升至 43.43%（相较最优对手提升约 2.56%），在 CIRR 上 Recall@1 提升至 39.46%（相较最优对手提升约 2.20%），整体排名第一。

**⚠️ 局限性**

局限包括对含义隐含的否定或细微指令仍易产生误解；批评者的评判标准是硬编码的，难以覆盖主观或特定任务的检索偏好；以及依赖 LLM 的解码稳定性和 API 版本更新所带来的不可控波动。

---

## 261. CooperScene: Multi-Modal Cooperative Autonomy Benchmark with C-V2X Communication Characterization

**arXiv ID:** 2606.31219 | [PDF](https://arxiv.org/pdf/2606.31219v1)

**作者:** Bo Wu `[一作]` (University of California Riverside), Hang Qiu `[通讯]` (University of California Riverside)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个真实世界的多模态协作自主数据集（CooperScene），包含三辆车辆和一个路侧单元的同步 LiDAR、摄像头、GNSS/INS 以及 C‑V2X 网络吞吐量数据。

**💡 创新点**

创新点在于：①整合真实 C‑V2X 通信特性；②多车多模态同步与精准标定；③提供带宽约束的协作感知基准，揭示现有方法与实际网络能力之间的巨大差距。

**🔧 技术方法**

使用了 PTP 时钟同步、硬件触发、空间-时间 ICP、眼手标定、移动捕捉系统，以及多尺度 Transformer/图网络的协作感知与运动预测模型。

**📊 数据集**

采用了自采集的 CooperScene 数据集，涵盖 59K 帧、344K 3D 标注，包含 3 辆车 + 1 RSU 的多场景数据。

**📈 对比分析**

通过对 V2VNet、V2VAM、V2X‑ViT、CoBEVT、CMP、ERMVP、CoSDH 等基线在 C‑V2X 带宽约束与无限带宽两种设定下的 mAP 与预测误差进行对比，结果显示在真实带宽下性能下降 30–50%，但通信高效模型在受限条件下仍可保持相对优势。

**⚠️ 局限性**

主要局限包括：带宽与延迟瓶颈导致数据时效性不足；模型对丢包/时延的鲁棒性不足；数据集规模有限（仅三车、首版仅车类）；以及多模态融合精度仍受硬件与标定误差影响。

---

## 262. Long-term Traffic Simulation via Structured Autoregressive Modeling

**arXiv ID:** 2606.31209 | [PDF](https://arxiv.org/pdf/2606.31209v1)

**作者:** Lingyu Xiao `[一作]` (University of Hong Kong), Xintao Yan `[通讯]` (University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出统一框架 &; &;，通过将交通场景与运动序列映射为可变长度的结构化自回归 token 序列，实现长期交通模拟。

**💡 创新点**

创新点在于：①利用预训练大语言模型的统计先验和注意力机制作为交通动力学的结构先验；②将场景生成与运动生成统一为自回归序列；③提出检索式长时交通评估（RTE），解决动态 agent 匹配问题。

**🔧 技术方法**

使用了预训练 LLM（如 Qwen2.5‑0.5B）、Transformer 结构、自回归生成、检索与对比度学习 VAE latent 空间、位置嵌入等技术。

**📊 数据集**

数据集包括 Waymo Open Motion Dataset（WOMD）和 Waymo Open Sim Agent Challenge（WOSAC）的验证集。

**📈 对比分析**

通过与多种基线（ProSim、LLM2AD、SceneStreamer、InfGen、CAT‑K、SMART 等）在 8 s 短时与 30 s 长时任务上进行对比，本文方法在 Composite、Kinematic、Interactive、Map‑based 以及综合 RMM‑F1 指标上均达到或超越当前 SOTA。

**⚠️ 局限性**

局限性在于：1）agent 移除采用简单 heuristics；2）生成的 agent 缺少初始速度等属性；3）评估中二值碰撞指标过于粗糙，未考虑碰撞频率与严重度。

---

## 263. Towards Inclusive Mobility Modeling: Characterizing and Evaluating Elderly Trajectory Patterns in Urban Systems

**arXiv ID:** 2606.31207 | [PDF](https://arxiv.org/pdf/2606.31207v1)

**作者:** Zhengxuan Wang `[一作]` (Shanghai University of Finance and Economics), Mengying Zhou `[通讯]` (Shanghai University of Finance and Economics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对2016‑2020年Jersey City Citi Bike数据进行年龄分组，系统地量化了老年人和年轻人轨迹的空间与时间特征，并以合成轨迹生成为实验平台，比较了在不同人口比例训练下的第一阶马尔可夫链与基于Qwen3‑4B LLM的生成模型对老年人轨迹的再现精度，揭示了少数群体不平衡导致的模型偏差。

**💡 创新点**

创新点包括：①首次用量化指标（半径聚类、熵、步长等）描述老年人与年轻人轨迹的结构差异；②构建可复现的实验框架，在相同数据、相同评估指标下对比传统马尔可夫链与高容量LLM对少数群体生成效果的差异；③提出从数据预处理到指标评估的完整评测管线；④发现即使使用更大容量的LLM，在少数群体数据稀缺的条件下仍无法提升子群体生成精度。

**🔧 技术方法**

技术手段：第一阶马尔可夫链（基于站点转移概率、停留时间等统计）和Qwen3‑4B LLM（通过4‑bit QLoRA微调），使用轨迹序列序列化为token流进行训练；利用空间（步长、RoG、熵）与时间（速度、停留时间、时段分布）指标评估生成轨迹质量。

**📊 数据集**

数据集：2016‑2020年Jersey City Citi Bike共享单车轨迹，约1.53M有效行程，其中老年人（≥65岁）1.44万行程、年轻人（18‑35岁）78.4万行程；数据包含站点坐标、行程时长、用户性别与出生年份等信息。

**📈 对比分析**

比较方法：在三种训练集（全体、仅年轻、仅老年）下分别训练马尔可夫链和LLM，每个模型生成与真实老年人相同数量（2,007）轨迹；使用步骤长度、速度、停留时间、RoG、熵等指标计算相对误差。结果显示：马尔可夫链老年人训练集表现最佳（误差≤5%），全体训练集在停留时间误差上有所改善但空间指标仍偏差大；LLM模型在时间指标上误差极大（速度+70%），空间指标相对较好但仍不及马尔可夫链。总体上，少数群体的生成精度受数据稀缺限制，LLM并未带来显著提升。

**⚠️ 局限性**

局限性：仅使用单一城市（Jersey City）的单车数据，缺乏跨城市验证；只考虑年龄作为分组维度，未扩展至性别、收入等多维特征；数据时间跨度有限，未探究长期趋势；LLM在合成时仅生成站点序列，时间属性需后置假设，导致时间指标表现不佳。

---

## 264. Planar Embedding of Okamura-Seymour Quasimetrics in Polynomial Time with an Application to Distributed SSSP

**arXiv ID:** 2606.31192 | [PDF](https://arxiv.org/pdf/2606.31192v1)

**作者:** Hung Le `[一作]` (University of Massachusetts), Shuang Yang `[通讯]` (University of Massachusetts)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一个多项式时间算法，用来构造任何给定Okamura‑Seymour准度量（quasimetric）的平面嵌入，并进一步给出在分布式 Congest 模型下以 O(D) 轮实现 (1+ϵ)‑近似单源最短路的算法；此外，还设计了最小失真（distortion）Okamura‑Seymour 嵌入的多项式时间求解方法。

**💡 创新点**

创新点包括：① 将 Chen‑Tan 的存在性证明转化为可执行的线性规划（LP）并通过 Ellipsoid 方法实现；② 在路径绘制过程中使用面（face）序列的可行前缀（valid prefix）检测技术，避免了对所有路径轨迹的指数枚举；③ 通过对面拆分、壁（wall）与走廊（corridor）等拓扑结构的细致分析，构造了可行的 LP 约束；④ 在分布式环境下利用 Okamura‑Seymour 嵌入构造平面距离模拟器（planar distance emulator），从而完成 O(D) 轮的近似 SSSP。

**🔧 技术方法**

主要技术手段包括：线性规划与 Ellipsoid 求解、面序列的可行前缀搜索、最短路径与单调性（Monge）约束、平面图的拆分与重组（cutting along walls）、分布式 BDD（Bounded Diameter Decomposition）、Hub Network 的构造与合并，以及 Thorup 距离预言器的应用。

**📊 数据集**

该工作为纯理论算法，未使用任何实验数据集；所有结果均为理论证明与多项式时间复杂度分析。

**📈 对比分析**

与现有最优方法相比，本文提供了从检测到嵌入的完整多项式时间实现，弥补了 Chen‑Tan 仅给出多项式检测算法的空缺；在分布式 SSSP 方面，算法在轮数上几乎匹配最小下界 Ω(D)，远优于此前 O(D²) 轮的最佳结果；最小失真嵌入问题则在给定拓扑顺序时实现了多项式时间最优解。

**⚠️ 局限性**

主要限制在于：① 多项式时间的度数很高（约 15 次方），实现复杂且常数大；② 对于未给定终点顺序的最小失真嵌入问题仍是 NP‑难；③ 算法依赖于存在完整的平面嵌入，若准度量仅满足 Okamura‑Seymour 条件但缺失明确的平面图则需要额外处理；④ 在实际分布式部署中，通信与局部计算的常数项可能较大。

---

## 265. The Decomposition Is the Fingerprint: Per-Component Identity for Agent Skills

**arXiv ID:** 2606.31272 | [PDF](https://arxiv.org/pdf/2606.31272v1)

**作者:** Hongliang Liu `[一作]` (Palo Alto Networks), Tung-Ling Li `[通讯]` (Palo Alto Networks)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于技能组件（提示、代码、工具）分解的可压缩指纹，用多银行 SimHash 生成固定 120 字节的三元组指纹，实现技能族身份识别、去重、搜索、家族聚类与创新检测。

**💡 创新点**

创新点在于：①将技能拆分为可独立比较的组件并分别指纹；②保持三元组结构而非单一分数，能定位重用位置并自动分类关系；③在 77× 压缩下保留与原 768 维嵌入相同的相似度排名，AUC 达 0.974。

**🔧 技术方法**

使用了：文本/代码嵌入器（Google text‑embedding‑005）、多银行 SimHash（5×64 位）、余弦→Hamming 距离映射、校准 z‑score、FAISS 二进制索引、CycloneDX 风格 SkillBOM，配合行为完整性验证（BIV）构成两轴信任模型。

**📊 数据集**

实验数据集包括：构造的 100 技能（20 组 5 变体）4,950 对；10 攻击基准 50 对 5 源技能+10 攻击；OpenClaw 公开社区 1,000 技能；BIV 注入基准 906 技能（502 好，404 恶）；以及公开模型服务。

**📈 对比分析**

通过 ROC AUC 与原嵌入余弦、字节级模糊哈希（ssdeep、TLSH）、AST 归一化哈希、LLM 预处理+SimHash、完整嵌入余弦等做对比。指纹在 77× 压缩下 AUC 0.974，单组件与嵌入相同 AUC 0.992；攻击基准 45/50 重新写入被检测；OpenClaw 上 per‑component AUC 0.99+；注入基准识别 83.7% 近克隆并精准定位代码变更。

**⚠️ 局限性**

局限性：仅衡量结构身份，无法判断安全；为局部敏感哈希，攻击者可伪造；分数范围狭窄（0.65–1.0）需校准；依赖编码器质量，跨语言不完整；组件覆盖不全；实验规模有限，需在更大、独立注册表中进一步验证。

---

## 266. Plan Right, Then Plan Tight: Symbolic RL for Efficient Embodied Reasoning

**arXiv ID:** 2606.31260 | [PDF](https://arxiv.org/pdf/2606.31260v1)

**作者:** Xiangli Shi `[一作]` (Tsinghua University), Yufei Huang `[通讯]` (Tencent)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套以 BDDL 为中心的端到端管线，将开放世界视频或任务描述转化为可验证的 BDDL 规范，使用符号引擎对生成的计划进行快速检验，并基于此监督训练 8B 规模的紧凑规划模型。

**💡 创新点**

创新点包括：① 将已验证的 BDDL 作为数据构建、计划验证和奖励设计的统一接口；② 用符号引擎提供多粒度可解释奖励（GCR、Engine‑Pass、Strict‑Pass、错误集）；③ 引入 GroupAdapt 难度感知长度调度，先保证准确性后动态压缩输出。

**🔧 技术方法**

技术手段包括视频‑>BDDL 解析器、LLM 验证器、毫秒级符号引擎、基于 SFT 的初始训练、DAPO 强化学习与符号奖励、Short‑RL 长度惩罚以及 GroupAdapt 调度。

**📊 数据集**

主要使用 BEHAVIOR‑1K（Behavior‑100 与 Behavior‑1000）数据集；为评估 OOD 逻辑能力，还对 AIME24、AIME25、AMC23、MATH500 等数学题库进行了测试。

**📈 对比分析**

与前沿 API 模型（DeepSeek、Gemini、Kimi、GLM、GPT‑5）及大规模开源模型（Gemma、Qwen 系列）进行对比，8B 模型在 B‑1000 上达 97.3% Strict‑Pass，平均长度仅 207 token，比 Qwen3‑8B 低 79% 长度、相对提升 25.9%，且超过强基线 3.5%。

**⚠️ 局限性**

局限性：动作空间保持抽象，无法直接控制低层抓取/移动；真实机器人需要实时感知并构建 BDDL，相关技术仍待完善；依赖引导式 Prompt，其他 Prompt 形式的效果尚未验证；未包含真实物理仿真或硬件部署。

---

## 267. ForgeDrive: Bidirectional Cross-Conditioning for Unified Visual-Action Generation in Autonomous Driving

**arXiv ID:** 2606.31226 | [PDF](https://arxiv.org/pdf/2606.31226v1)

**作者:** Xuchang Zhong `[一作]` (Amap, Alibaba Group), Yang Cai `[通讯]` (Amap, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

统一了视觉生成与动作规划的自回归扩散框架ForgeDrive，能够实现全状态仿真、规划和视觉里程计；

**💡 创新点**

创新点包括：①双向跨模态条件的解扩散，通过解耦图像与动作的扩散步长实现任意模态互推；②提出act‑then‑imagine推理顺序，逆转传统的imagine‑then‑act，消除错误累积；

**🔧 技术方法**

使用技术包括：自回归扩散变压器（UniDiT）+ UniDiffuser 风格噪声调度，跨模态注意力，多任务混合训练（逆向动力学、动作条件视频生成、联合预测），Scheduled Sampling 等；

**📊 数据集**

主要数据集为NAVSIM（基于nuPlan的仿真数据），预训练于nuPlan，实验使用NAVSIM v1、v2、navhard；

**📈 对比分析**

与多种最先进方法（DriveGPT、Epona、Uni‑World VLA、DiffusionDrive等）在PDMS/EPDMS、FVD等指标上对比，ForgeDrive在NAVSIM v1取得90.2 PDMS，v2 90.3 EPDMS，FVD 69.2，显著优于同类方法，规划准确率和视频质量均得到提升；

**⚠️ 局限性**

局限性包括：仅支持单摄像头2Hz输入，未实现多传感器融合；在极端场景下仍可能出现视觉误差导致规划偏差；目前仅在仿真环境验证，缺乏真实世界部署评估。

---

## 268. FeatX: Editing Software by Editing Features for Repository-Level Code Evolution

**arXiv ID:** 2606.31206 | [PDF](https://arxiv.org/pdf/2606.31206v1)

**作者:** Xutian Li `[一作]` (Peking University), Bing Xie `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 FeatX 工具，通过提取仓库的层次化功能结构并利用 LLM 演进代理实现从功能编辑到代码补丁的自动化转换。

**💡 创新点**

创新点在于将功能（feature）作为一等可编辑单元，构建功能到代码的显式映射，并设计三阶段演进代理（上下文扩展、定位规划、具体代码修改），显著降低认知负担并提升功能级别修改定位准确率。

**🔧 技术方法**

技术包括：静态依赖矩阵与语义矩阵（LLM 摘要 + Sentence‑BERT 嵌入）的融合，Leiden 聚类算法构建功能层次；静态分析（JavaParser）与 LLM 引导的探索检索扩展代码映射；三阶段演进代理的设计；前端 React + 后端 Spring Boot 实现；成本优化的 LLM 调用配置。

**📊 数据集**

使用了 38 个真实开源 Java 仓库（FlappyBird、JSON‑java、JCommander、NBlog、PlayEdu）中的功能编辑提交数据集，平均每任务涉及 3.2 包、4.5 文件、10.6 函数。

**📈 对比分析**

通过 NASA‑TLX、SUS 评估用户感知负担和可用性；在功能级定位准确率上与 DeepSeek、GPT‑4o‑mini、GPT‑5.2、Claude‑opus‑4.5、Cursor Agent 等 SOTA LLM 进行对比，FeatX 取得最高 F1 0.385，精度 41.6%、召回 35.8%，相较 Claude‑opus‑4.5 提升 42.6%，且总 LLM 成本仅 $0.07。

**⚠️ 局限性**

局限性包括：对极其复杂或跨仓库的功能编辑支持不足；依赖于 LLM 的确定性输出，可能受模型版本影响；在大型项目中聚类与映射的精度与可维护性需进一步验证。

---

## 269. Incentivizing Data Trading via Profit Reallocation

**arXiv ID:** 2606.31202 | [PDF](https://arxiv.org/pdf/2606.31202v1)

**作者:** Yunxuan Ma `[一作]`, Xiaotie Deng `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文阐述了NeurIPS 2026 会议论文提交的格式化要求与排版规范。

**💡 创新点**

创新点在于系统化整合页面尺寸、字体、标题、摘要、章节、图表、参考文献等细节，提供统一的排版标准，帮助作者避免格式错误。

**🔧 技术方法**

使用 LaTeX 模板（neurips_2026.sty）、Times New Roman 字体，以及 natbib、graphicx、booktabs 等宏包。

**📊 数据集**

无数据集，内容为格式与排版指导。

**📈 对比分析**

未涉及方法比较或性能评估，仅说明如何在提交与排版时遵循规范、避免常见错误。

**⚠️ 局限性**

主要限制是格式要求严格，非规范提交易被拒稿；缺乏实验与实证内容，无法评估实际性能。

---

## 270. Agentic RAG-VLM: Affordance-Aware Retrieval-Augmented Generation with Self-Reflective Planning for Robotic Grasping

**arXiv ID:** 2606.31200 | [PDF](https://arxiv.org/pdf/2606.31200v1)

**作者:** Tao Chen `[一作]` (Fudan University), Zhongxue Gan `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一套统一框架，将视觉语言模型、基于功能的检索增强生成、场景图约束推理和自我反思式恢复机制融合，用于实现多场景、多任务的机器人抓取。

**💡 创新点**

创新点包括：①层次化功能感知检索增强生成（HAA‑RAG），根据抓取功能属性而非视觉相似度检索经验；②通过场景图约束推理将空间关系转换为可执行的抓取参数调整；③采用14种故障类型的结构化诊断和三级自适应重试，实现闭环物理基础的失败恢复。

**🔧 技术方法**

技术手段主要有：Qwen3‑VL‑8B视觉语言模型、检索增强生成（RAG）、场景图构建与约束分析、ReAct式自我反思循环、会话级经验记忆、七因子抓取质量评估模型。

**📊 数据集**

使用了包含12个日常物体（10类、8种功能属性）的自建数据集和116条抓取经验知识库，并在12个任务（单抓、交互、长周期）共360次试验上评估。

**📈 对比分析**

与仅使用VLM、去掉HAA‑RAG、去掉场景图、去掉恢复等消融组以及启发式基线进行对比，完整系统在所有任务中实现78.3%的总体成功率（单抓91.7%，交互64.2%，长周期66.7%），比VLM‑Only提升了53.3个百分点，显著优于传统方法。

**⚠️ 局限性**

局限性主要在于：知识库规模有限（116条），导致远OOD场景下成功率下降至54.4%；对抓取器尺寸和几何约束的处理仍不完备（如杯子外径超过抓手开口）；以及对VLM推理的计算开销较高。

---

## 271. Diffusion-based 4D Trajectory Prediction and Distributed Control for UAV Swarms

**arXiv ID:** 2606.31197 | [PDF](https://arxiv.org/pdf/2606.31197v1)

**作者:** Tianshun Li `[一作]` (Hong Kong University of Science and Technology), Xinhu Zheng `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一套统一框架，将4D UAV 群集的轨迹预测与分布式非线性 MPC（DNMPC）闭环控制结合起来，实现了高精度、实时的编队跟踪。

**💡 创新点**

创新点包括：① 维度分离的粗糙预测，降低计算复杂度；② 采用扩散模型对残差动力学进行时序相关的精细化建模；③ 将精细化预测直接嵌入 DNMPC，理论上实现实用稳定性；④ 发布了六场景、7.9k帧的同步三UAV 4D 轨迹数据集。

**🔧 技术方法**

核心技术包括轴向分离的轨迹预测网络、基于 Temporal‑U‑Net 的扩散残差模型、分布式非线性 MPC、以及多模型对比实验评估（ADE/FDE、跟踪误差、实时性）。

**📊 数据集**

使用自己构建的 6 个场景（Urban、Lab、Factory、Rural 等）三UAV 4D 轨迹数据集，共 7,900 帧，包含速度意图与目标扇区标注。

**📈 对比分析**

与 KF/UKF、TCN、MSRL、LBEBM、NPSN、GAN、CVAE、MTRTraj 等基线对比，ADE/FDE 分别提升 16.7%/18.2%，跟踪误差降低 10–15%，实时推理速度 34 FPS（<30 ms 延迟），在复杂城市与工业环境中实现平均跟踪误差 <0.07 m。

**⚠️ 局限性**

局限性在于实验仅覆盖三UAV 小规模编队，未验证更大规模群集；残差模型在高度曲折或突变交互场景下仍可能累计误差；对非同步或无目标标注场景的适应性仍待进一步验证。

---

## 272. ISM:Self-Improving Strategy Memory for Continual Mathematical Reasoning

**arXiv ID:** 2606.31191 | [PDF](https://arxiv.org/pdf/2606.31191v1)

**作者:** Prakhar Dixit `[一作]` (University of Maryland, Baltimore County), Tim Oates `[通讯]` (University of Maryland, Baltimore County)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个名为 Intelligent Schema Memory (ISM) 的自我进化外部记忆系统，帮助冻结的LLM在连续学习的数学推理任务中不断提升表现。

**💡 创新点**

将策略schema拆分为内容和特征hook，并通过七种自我改进机制（审计、合并、修正、增删、强化、反模式记录和条件合成），以及符号验证门控，形成主动维护、可验证且紧凑的记忆库。

**🔧 技术方法**

使用特征提取器（规则+LLM）、两阶段检索、冻结LLM推理配合可验证工具、以及基于EMA的hook更新和自我改进循环等技术。

**📊 数据集**

在MATH‑Hard和OlympiadBench这两个竞赛级数学推理基准上进行300集连续学习实验。

**📈 对比分析**

与五种基线（Vanilla、Static Schema、RAG‑over‑Examples、Reflexion、Passive Schema Memory）对比，ISM在准确率、稳定性、遗忘率和后向迁移上均优于所有基线，且记忆库大小比最强基线少64%–86%，比检索基线小23倍。

**⚠️ 局限性**

实验仅使用单一随机种子和固定问题流序列，未做机制级消融或跨模型/任务验证，且验证门限依赖符号工具，对证明密集任务或更广泛领域的适用性尚未证明。

---

## 273. Learning to Deny: Action Denial in Multimodal Large Language Models

**arXiv ID:** 2606.31187 | [PDF](https://arxiv.org/pdf/2606.31187v1)

**作者:** Raiyaan Abdullah `[一作]` (University of Central Florida), Yogesh Singh Rawat `[通讯]` (University of Central Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了专注于动作否定（action denial）的UCF101-AD基准，评估了20种多模态大型语言模型（MLLM），并提出了基于因果图的CausalAct方法来提升模型在否定动作时的判断能力。

**💡 创新点**

创新点包括：①将动作识别的对立任务——在强上下文提示下准确否定动作；②构造UCF101-AD数据集，包含正负对齐且缺失关键动作的剪辑；③设计因果图结构（P,O,L,S,I,M,A）并以自然语言提示引导模型进行因果推理；④通过图相关的VQA任务微调，使模型学会验证动作的因果链而非仅靠关联。

**🔧 技术方法**

使用技术主要有：多模态大语言模型（如VideoLLaMA、Ovis、Qwen等）的零样本/微调评估；自然语言结构化提示（CausalAct Prompt）；图结构化问答（Graph‑based VQA）进行模型微调；以及对比实验、进阶的Explicit Denial/No‑Distractor设置。

**📊 数据集**

使用的数据集：UCF101-AD（基于UCF101的动作正负对齐样本），并在外部数据集（Kinetics‑400、HMDB51、SSv2、Diving48、FineGym99）上验证CausalAct的泛化。

**📈 对比分析**

比较方法为多选VQA的零样本评估；在UCF101-AD上正样本识别准确率普遍>90%，但动作否定平均仅20–50%，最佳模型VideoLLaMA3仅51.5%；引入CausalAct后，模型在动作否定上提升至约60–70%，并在外部数据集上提升10–20个百分点，显示因果推理可显著改进否定能力。

**⚠️ 局限性**

局限性包括：①思考型模型在否定任务上反而表现更差，说明当前“链式思考”不适合负约束；②模型对因果图结构的理解仍有限，微调需要同时更新视觉编码器；③在更复杂、细粒度动作的否定仍难以达到人类水平；④数据集虽然针对动作否定，但仍无法完全覆盖所有可能的上下文混淆情形。

---

## 274. Learning Gaussian Graphical Models from a Glauber Trajectory Without Mixing

**arXiv ID:** 2606.31230 | [PDF](https://arxiv.org/pdf/2606.31230v1)

**作者:** Eric Shen `[一作]` (Massachusetts Institute of Technology), Ankur Moitra `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种多项式时间算法，能够仅利用一次 Glauber 动态轨迹（无需等到混合）恢复稀疏高斯图模型的结构，并给出了与最优对数因子相匹配的轨迹长度上界。

**💡 创新点**

核心创新在于通过局部窗口内的“iiji”模式、鲁棒方差估计与中位数聚合，成功克服了单轨迹中的时间相关性与隐藏邻居干扰，实现了无混合时间的结构学习。

**🔧 技术方法**

主要技术包括：1) 估计并归一化对角条目以得到单元对角精度矩阵；2) 设计基于短窗口的边测试；3) 使用鲁棒均值/方差估计和中位数稳健聚合；4) 采用马尔可夫链混合与耦合论证以控制误差。

**📊 数据集**

实验上使用合成稀疏高斯图模型（α‑sparse），未依赖真实数据集，主要验证理论轨迹长度与算法复杂度。

**📈 对比分析**

与已有方法相比，本文的轨迹长度上界去除了对混合时间的依赖，并且在不额外正则化假设下实现了更宽松的稀疏性与边权下界；然而，在常数与多项式因子上仍不如某些基于混合时间的算法实用。

**⚠️ 局限性**

局限性包括：轨迹长度常数极大，导致算法在实践中不可行；对 d、α 的多项式依赖可能不是最优；仅适用于高斯模型，且需先知 α、d 等超参数；对非高斯或非马尔可夫数据的适用性尚未证明。

---

## 275. CLIMB: Centroid-Based Hierarchical Memory for Online Continual Self-Supervised Learning

**arXiv ID:** 2606.31275 | [PDF](https://arxiv.org/pdf/2606.31275v1)

**作者:** Julien Lefebvre `[一作]` (Universite Claude Bernard Lyon 1), Mathieu Lefort `[通讯]` (Universite Claude Bernard Lyon 1)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

CLIMB 在在线连续自监督学习场景下提出一种新颖的层次化质心记忆方案，并与 EMA 参考网络的知识蒸馏相结合；

**💡 创新点**

创新点在于：① 将质心记忆做成短期（STM）和长期（LTM）双层结构，并在有限内存下实现全流覆盖；② 引入自适应阈值和聚合/合并策略；③ 同时利用重放和蒸馏双重机制，兼顾表示稳定性与多样性；

**🔧 技术方法**

技术方法包括：SimCLR 自监督对比学习、EMA 目标网络、质心基记忆的增量更新与合并、对齐损失（negative cosine similarity）以及动态阈值和聚合规则；

**📊 数据集**

实验使用 Split CIFAR‑100 与 Split ImageNet‑100，采用类增量 20/50/100 任务以及不规则任务分布；

**📈 对比分析**

对比 MinRed、Osiris‑R、SCALE、CLA‑E、CLA‑R、PCMC，采用统一 CBP 预算（3.7×10⁶ 或 9.5×10⁶）和 2500 图像内存；结果显示 CLIMB 在 ImageNet‑100 上始终高于所有对手，CIFAR‑100 上与 CLA‑R、MinRed 相当，并在不规则任务分布下保持优势；

**⚠️ 局限性**

局限性包括：① STM/LTM 容量与合并策略需进一步优化；② 仍依赖原始图像存储，内存占用较大；③ 对不同 SSL 方法的适应性未充分验证，未来可考虑压缩或生成式存储。

---

## 276. WarpHammer: Densifying Scene Warps with 3D Object Priors for Extreme View Synthesis

**arXiv ID:** 2606.31258 | [PDF](https://arxiv.org/pdf/2606.31258v1)

**作者:** Michael Green `[一作]` (OriginAI), Or Litany `[通讯]` (Technion)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种训练无关的投影条件NVS框架，利用显式3D对象重建补全稀疏视角下的Warp，支持极端视角合成和无姿态辅助图像融合。

**💡 创新点**

创新点在于通过对象级3D先验增量补全Warp稀疏并作为遮挡，同时允许从未姿态的外部图像提供几何证据，仅保持参考场景外观。

**🔧 技术方法**

使用Monocular depth → 点云、SAM3D/其他3D生成器、VGGT多视几何对齐、视频扩散模型（如GEN3C/TrajectoryCrafter）以及自定义的OrbErr指标。

**📊 数据集**

评估于SynView-X、OmniObject3D、Co3D、Objectron、Mip-NeRF360等五个合成、扫描及真实场景基准。

**📈 对比分析**

与多种基线（TrajectoryCrafter、GEN3C、ReCamMaster、EX-4D、CogNVS、SEVA、WorldForge）对比，单视角时PSNR提升≈4.2dB，旋转误差减半；辅助视角时在无姿态条件下逼近使用GT姿态的结果。

**⚠️ 局限性**

局限在于依赖准确的前景分割与3D重建质量，适用于显著前景物体，对复杂或薄透明场景效果下降，外部辅助视角若与参考相似或差异过大会导致对齐不稳，推理时延增加。

---

## 277. Probing Stylistic Appropriation using Large Language Models: An Evaluation Framework for Copyright Infringement under EU Law

**arXiv ID:** 2606.31250 | [PDF](https://arxiv.org/pdf/2606.31250v1)

**作者:** Noah Scharrenberg `[一作]` (Maastricht University), Chang Sun `[通讯]` (Maastricht University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了PSALM框架，用以在欧盟版权法视角下对大型语言模型（LLM）进行风格性侵权评估，并通过该框架在基准模型、微调模型以及负偏好优化（Negative Preference Optimisation, NPO）模型上进行实验，检验模型在文本生成时对原始文学作品的风格、情节、人物等表达维度的“风格挪用”程度。

**💡 创新点**

创新点包括：①将欧盟版权法中关于原创性、实质相似性及例外条款的法律标准系统化为可量化的评估维度；②构建基于有向无环图（DAG）的LLM‑as‑Judge结构，允许多层次的子维度评估并聚合成最终评分；③将上述评估框架与NPO技术结合，探讨如何在消除字面记忆的同时抑制非字面风格挪用；④通过对LLaMA 3.2 1B在历史荷兰文学问答数据集上的实验，验证框架在真实文本生成中的可操作性。

**🔧 技术方法**

主要技术包括：①LLM‑as‑Judge（以GPT‑5‑nano为评判器）与DAG实现子维度评估与加权聚合；②负偏好优化（NPO）算法用于模型的“去记忆”训练；③对大型语言模型的LoRA微调与NPO微调；④与传统词汇重叠指标（Exact Match、BLEU、ROUGE‑L）结合，用以量化字面复制与非字面相似度。

**📊 数据集**

使用数据集：从荷兰数字图书馆（DBNL）抽取的19世纪前荷兰文学作品（约11位作者、约30部作品），将原始文本翻译为现代荷兰语（和英文）后构成QA对，约500对；随后将QA集划分为“retain”和“forget”子集，用于微调与NPO；基准模型为Meta LLaMA 3.2 1B Instruct。

**📈 对比分析**

比较方法：首先对PSALM评估器在人工构造的5级测试案例上进行验证（严格对齐率≥0.80，MAE≤0.15）；随后对每个模型条件（基准、微调、NPO）在各语言/子集上计算PSALM得分与词汇重叠指标；使用非参数统计（Kruskal‑Wallis、Mann‑Whitney、Wilcoxon）检验模型间差异。结果表明：①微调显著提升了写作风格、叙事声音、人物、情节等法律意义上可保护的表达维度（PSALM得分接近1）；②NPO能够几乎消除字面复制（Exact Match、BLEU、ROUGE降至0），但对风格与结构的相似度仅降低到0.2–0.3，仍高于基准模型；③不同语言和子集的差异不大，说明微调与NPO的效果普遍。

**⚠️ 局限性**

局限性：①PSALM评估结果仍需法律专家核验，未能完全替代司法判决；②评估器的设计基于欧盟版权法，其他法域的适用性未知；③实验使用的荷兰文学文本与英语模型的可迁移性有限，未验证在更大规模或多语种语料下的稳健性；④NPO虽抑制字面记忆，却难以彻底消除对原始风格的潜在记忆；⑤评估过程对LLM‑as‑Judge的准确性和一致性有依赖，若评判模型偏差会影响最终得分。

---

## 278. Can LLMs Imagine Moral Alternatives Beyond Binary Dilemmas?

**arXiv ID:** 2606.31213 | [PDF](https://arxiv.org/pdf/2606.31213v1)

**作者:** Jongchan Choi `[一作]` (Korea University), Jun-Hyung Park `[通讯]` (Hankuk University of Foreign Studies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个包含307个道德困境的MoralAltDataset，并在其中加入了“妥协”与“重构”两类超越二元选择的替代方案；

**💡 创新点**

首次系统评估LLM在存在替代方案时的决策偏好和生成质量，揭示LLM在道德想象与替代生成方面的能力及其与人类的相似性；

**🔧 技术方法**

使用零射prompt的多模型推理（包括GPT‑5、Claude、Gemini、Qwen等），以及对生成文本的成对偏好与专家评估；

**📊 数据集**

基于电影情节摘要与AI场景案例构建的Advisor与Agent两类困境，辅以人工与LLM生成的替代方案；

**📈 对比分析**

通过人类与模型在四选项任务中的选择分布、价值偏移、F1一致性以及成对偏好评估，结果显示LLM往往更倾向于妥协方案，且生成的替代方案在质量上优于人类，尤其在结构性和伦理维度上；

**⚠️ 局限性**

研究局限在于数据集覆盖的文化与语言有限、使用的prompt与解码策略不涵盖交互式推理、以及伦理评估依赖于可操作化的清单，未能完整体现真实世界的多样性与复杂性。

---

## 279. TDGT: A Tabular Data Generation Toolkit supporting adaptive GPU-accelerated Bayesian mixture models, diffusion-based models, and latent-space generative modeling

**arXiv ID:** 2606.31268 | [PDF](https://arxiv.org/pdf/2606.31268v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 280. AC3S: Adaptive Conditioning for 3D-Aware Synthetic Data Generation

**arXiv ID:** 2606.31204 | [PDF](https://arxiv.org/pdf/2606.31204v1)

**作者:** Eric Ji `[一作]` (University of Illinois Urbana Champaign), Yaoyao Liu `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于扩散模型的3D感知合成数据生成框架AC3S，通过自监督视觉提示调制器和多代理VLM文本提示生成器，实现对图像的几何对齐与视觉真实性的双重控制。

**💡 创新点**

创新点在于：①自监督视觉提示调制器能够动态调节ControlNet的调节强度，避免过度约束导致的图像质量下降；②多代理VLM系统能够生成与视觉提示一致、细节丰富且语义多样的文本提示，提升生成图像的真实感与多样性。

**🔧 技术方法**

主要技术包括：扩散模型（Stable Diffusion v1.5）、ControlNet视觉条件、轻量级MLP调制器、基于边缘图的视觉提示提取、基于多代理的VLM文本生成、DreamSim感知评分过滤、LangGraph + Qwen3-VL-23B 的多代理协同。

**📊 数据集**

使用公开的CAD模型库（ShapeNet、Objaverse、OmniObject3D）渲染得到的边缘图作为视觉提示；生成的数据集包含100万个图像，覆盖ImageNet 1000个类别，提供分类、姿态估计和附加元数据。

**📈 对比分析**

通过FID、分类top-1准确率以及姿态估计精度与基线（3D-DST、Text2Img）比较，AC3S在FID上提升约15.95分，分类精度提升约1-2.6%，姿态估计在π/18阈值下平均提升约2.17%。

**⚠️ 局限性**

局限性包括：①对边缘图的依赖可能在高度复杂或部分遮挡场景中表现不佳；②多代理VLM系统对算力和模型大小要求较高；③仍需进一步验证在更大规模或不同领域（如视频）的可扩展性。

---

## 281. Gated Multi-Graph Fusion via Graph Attention Networks for Alzheimer's Disease Detection

**arXiv ID:** 2606.31186 | [PDF](https://arxiv.org/pdf/2606.31186v1)

**作者:** Jinyu Li `[一作]` (Tianjin University), Jianwu Dang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

构建多视角图神经网络，利用语义、依赖和共现图对阿尔茨海默病患者的自发语音进行分类。

**💡 创新点**

引入基于 PMI 的共现图刻画语篇流动性，并使用自适应门控融合机制以应对病情异质性。

**🔧 技术方法**

使用 Whisper ASR、BERT 词向量、spaCy 依赖解析、Graph Attention Network (GAT)、门控融合网络和 MLP 分类。

**📊 数据集**

使用 ADReSSo 2021 数据集的 Cookie Theft 语音描述。

**📈 对比分析**

与五个基线模型对比，5 折交叉验证准确率 88.81%，测试集准确率 90%，显著优于传统方法。

**⚠️ 局限性**

模型依赖 ASR 转写质量，未考虑音频韵律信息，且在不同语言/任务上的泛化尚未验证。

---

## 282. Information-Aided DVL Calibration

**arXiv ID:** 2606.31216 | [PDF](https://arxiv.org/pdf/2606.31216v1)

**作者:** Zeev Yampolsky `[一作]` (University of Haifa), Itzik Klein `[通讯]` (University of Haifa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了信息辅助校准（IAC）方法，用于在GNSS可用与不可用环境下对多普勒速度计（DVL）进行自标定，从而提高AUV导航精度。

**💡 创新点**

创新点在于：1）在GNSS可用场景下通过将Z轴速度置为零的非完整约束改进Kalman滤波校准；2）在GNSS不可用场景下提出简化误差模型、假设侧向与垂向速度近似零，并利用速度范数近似推导前进速度，实现完全自标定。

**🔧 技术方法**

使用技术包括基于Kalman滤波的线性状态估计（模型M3–M7），非完整约束（零速度约束）、DVL误差模型推导、速度范数重构等。

**📊 数据集**

实验数据来自海法大学Snapir AUV在地中海收集的15条实测轨迹（T1–T15），每条轨迹约400–600秒。

**📈 对比分析**

与传统基线模型M1、M2比较，GNSS可用时平均提升约20%（M3为最高），GNSS不可用时平均提升约30%，M7在不需要任何先验知识的情况下实现约35%的改进。

**⚠️ 局限性**

局限性包括：对DVL Z轴安装对齐的假设、侧向/垂向速度近似为零的前提、以及对前进速度范数近似的依赖，这些假设在某些操作环境中可能难以满足。

---

## 283. UHD-MFF: Shattering Barriers in Multi-Focus Ultra-High-Definition Image Fusion via Learnable Lookup Tables

**arXiv ID:** 2606.31242 | [PDF](https://arxiv.org/pdf/2606.31242v1)

**作者:** Yibing Zhang `[一作]` (Wuhan University), Jiayi Ma `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了UHD-MFF数据集并设计了UMF-LUT框架，实现高效4K多焦点图像融合。

**💡 创新点**

创新点在于采用分尺度查找表（C-LUT进行低分辨率区域决策，D-LUT进行高分辨率边缘细化），并以无监督方式训练，避免了伪纹理与过度拟合。

**🔧 技术方法**

使用的技术包括Sobel梯度与Laplacian响应、轻量级语义上下文网络、四维/二维查找表、四线性/双线性插值、无监督重构/稀疏/二值化正则化损失。

**📊 数据集**

采用新建的UHD-MFF数据集，共1950对3840×2160的真实+合成多焦点图像。

**📈 对比分析**

在UHD-MFF上与多种SOTA方法比较，指标均有显著提升，平均推理时间约11ms/图，参数仅0.008M，能耗与显存占用最低。

**⚠️ 局限性**

局限性包括：对极端噪声、动态场景的鲁棒性待验证；查找表尺寸与分辨率扩展受限；需要更丰富的多场景多焦点数据以进一步提升泛化能力。

---

## 284. TactX: Learning Shared Tactile Representations Across Diverse Sensors

**arXiv ID:** 2606.31236 | [PDF](https://arxiv.org/pdf/2606.31236v1)

**作者:** Junsung Park `[一作]` (University Of California San Diego), Xiaolong Wang `[通讯]` (University Of California San Diego)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 TactX 框架，实现多模态触觉传感器（视觉、磁性、阻抗）的共享潜在空间；

**💡 创新点**

创新点在于：①使用配对触摸数据进行跨模态对齐，②结合对比学习与重建损失形成统一潜在表示；

**🔧 技术方法**

采用模态专属编码器、重构网络、InfoNCE 对比损失、KL 正则化、以及跨模态重构；

**📊 数据集**

数据集来源于三种传感器（Daimon、eFlesh、FlexiTac）在相同抓取动作下的配对触觉观测；

**📈 对比分析**

与仅重建、仅对比或 L2 对齐模型对比，TactX 在跨模态一致性、传感器识别、物体分类和零样本策略迁移上均表现最佳，零样本策略成功率提升至 45.9% 以上；

**⚠️ 局限性**

局限性：需要配对触觉数据，难以满足复杂几何或非对称物体；数据主要为静态抓取，缺乏动态接触的对齐监督。

---

## 285. A First Exploration of Neuromorphic OT-CFM for Multi-Speaker VSR

**arXiv ID:** 2606.31225 | [PDF](https://arxiv.org/pdf/2606.31225v1)

**作者:** Lin Chen `[一作]` (Beijing Technology and Business University), Xiaoming Chen `[通讯]` (Beijing Technology and Business University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出LipFlow框架，将RGB视频转换为高时分辨率事件流，并结合多说话人跟踪与活说话人检测，实现多说话人视觉语音识别。

**💡 创新点**

创新点在于：①使用可学习的事件生成模块捕捉微秒级唇动；②引入Optimal Transport Conditional Flow Matching (OT‑CFM)实现仅两步ODE推断，极大提升推断速度；③采用双层语义监督（Token‑级BERT权重耦合 + 句子‑级Sentence‑BERT先验）解决同形词歧义。

**🔧 技术方法**

技术手段包括：事件流生成（差分量量化+自适应阈值网络）、Hierarchical Frame Interpolation、3D‑2D Encoder + Conformer视觉编码、Speaker‑Conditioned OT‑CFM生成器、AdaLN与Cross‑Attention融合、BERT权重耦合解码器、CTC、InfoNCE对齐与多任务损失。

**📊 数据集**

数据集：DVS‑Lip（真实事件流数据）和AVA（自然场景多说话人视频），并通过自建的数据处理管道将RGB转为事件流。

**📈 对比分析**

与视频基准（Auto‑AVSR、VATLM、LipGen等）和事件基准（MSTP、SNN‑Lip）对比，LipFlow在DVS‑Lip上取得22.3% WER、19.8% VER，RTF仅0.18、延迟240 ms，比MSTP低8.2% WER、速度提升26×；在AVA多场景测试中，LipFlow在低光、快速动作、遮挡等极端条件下仅轻微性能下降。

**⚠️ 局限性**

局限性在于：①需要专用事件摄像头或高质量事件生成；②对极端低光或遮挡仍有一定误识别；③模型在不同说话人间的泛化能力受限于训练数据多样性。

---

## 286. AA: A Multi-view Multimodal Dataset for Screen-based Gaze Estimation

**arXiv ID:** 2606.31211 | [PDF](https://arxiv.org/pdf/2606.31211v1)

**作者:** Chang Liu `[一作]` (Tsinghua University), Yuanchun Shi `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一套包含八个屏幕内置摄像头和两个侧视摄像头的多视角多模态眼球注视数据集，每个样本包含同步的全景图、面部、左眼和右眼裁剪图，配合严格的固定目标注视任务和高质量的注视标注。

**💡 创新点**

创新点包括：①多视角覆盖（屏幕内外双侧视角）提升对视角变化和遮挡的鲁棒性；②侧视摄像头通过投影合成目标实现与屏幕坐标的一致标注；③多模态表示（全景、面部、眼部）为全局与局部特征融合提供支持；④公开的受试者独立训练/验证/测试拆分与统一数据处理流程。

**🔧 技术方法**

使用的技术主要包括：同步录制的10路摄像头（1920×1080@30FPS）、MediaPipe人脸与眼部检测用于裁剪、投影变换实现侧视目标重建、基于眼动仪与光流的双重质量控制与异常帧剔除、以及标准化的图像预处理与数据集划分。

**📊 数据集**

所用数据集是本文新建的“多视角屏幕注视数据集”，共24名受试者、288个屏幕目标、12.96M裁剪图、3.24M原始图像，按受试者分布为训练60%、验证20%、测试20%。

**📈 对比分析**

文中未提供基准模型或性能对比，仅阐述了数据集构建与质量控制方法；若需使用，可结合现有注视估计网络在该数据集上进行实验验证。

**⚠️ 局限性**

局限性主要有：实验在受控实验室环境下完成，缺乏真实场景多样性；侧视摄像头的目标合成虽然保持坐标一致，却不真实模拟侧视屏幕可见度；眼动仪仅用于筛选，不参与监督，可能导致标签偏差。

---

## 287. Distilling Temporal Coherence into 2D Networks for Transrectal Ultrasound Prostate Video Segmentation

**arXiv ID:** 2606.31198 | [PDF](https://arxiv.org/pdf/2606.31198v1)

**作者:** Dong Yeong Kim `[一作]` (Seoul National University), Young-Gon Kim `[通讯]` (Seoul National University Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在训练阶段将时间一致性蒸馏到 2D 网络的框架，用于实时前列腺 TRUS 视频分割，保持单帧推理效率；

**💡 创新点**

创新点在于引入 Confidence‑Weighted Temporal Consistency 与 Dual‑scale Prototype Alignment 的双层约束，并通过几何等价伪标签和学生‑教师知识蒸馏实现无标注视频的自监督学习；

**🔧 技术方法**

使用光流加权一致性损失、对比学习原型对齐、学生‑教师知识蒸馏、自监督几何等价伪标签、U‑Net++/ACSNet 等 2D backbone 以及相关光流与对比学习工具；

**📊 数据集**

采用新构建的 TRUS‑V 视频基准（2,679 帧，10 例患者）以及公开的 SUN‑SEG 视频数据集进行实验；

**📈 对比分析**

与传统 2D 基线和多种视频模型相比，在 TRUS‑V 与 SUN‑SEG 上均取得 Dice、S_α 等指标的最优或竞争性能，并以约 127 FPS 的实时速度推理；

**⚠️ 局限性**

局限性包括对极端噪声/暗区的鲁棒性仍待提升、缺乏 3D 重建与多模态验证、以及在非 TRUS 视频场景的泛化能力需进一步评估。

---

## 288. AI-Assisted Discovery of Convex Relaxations via Dual Agents

**arXiv ID:** 2606.31182 | [PDF](https://arxiv.org/pdf/2606.31182v1)

**作者:** Sungyoon Kim `[一作]` (Stanford University), Mert Pilanci `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

通过双智能体（编码代理和理论代理）自动探索并验证凸松弛约束，进而获得自闭合下界，成功提升了两条经典自相关不等式 C6.2 与 C6.5 的已知下界。

**💡 创新点**

创新点在于将自闭合/自适应搜索框架（autoresearch）迁移到凸松弛发现任务；使用可证明的双重代理循环，在保证约束有效性的同时利用弱对偶性与区间算术实现无误差的下界证明。

**🔧 技术方法**

主要技术包括：LLM 编码代理自动生成/修改 SDP/LP 约束；理论代理进行数学证明与对抗性反例搜索；人类审查与压缩；利用 CVXPY 进行标准化的对偶求解；在区间算术中对正定、二阶锥与非负锥的可行性进行严谨验证；分支界法对参数空间进行划分并合并多局部下界。

**📊 数据集**

本工作不使用传统数据集，而是基于已知的自相关常数定义（C6.2 与 C6.5）和相关的数值参数（如网格宽度、截断阶数、Fourier 系数范围）构建实验模型。

**📈 对比分析**

与之前的手工构造的下界（1.28 与 0.379005）相比，本文得到的下界分别提升到 1.2937（+0.0137）和 0.37912（+0.000115）。性能提升体现为更紧的凸松弛和更高的对偶下界，且所有结果均通过区间算术和对偶可行性严格验证。

**⚠️ 局限性**

主要限制包括：约束有效性验证仍需人工审查，缺乏完全形式化的证明；目前仅在两条常数上验证，缺乏广泛的通用性评估；所用的 LLM 仍受模型偏差和可解释性限制，可能导致错误的约束建议。

---

## 289. Failure-Based Testing for Deep Reinforcement Learning Agents

**arXiv ID:** 2606.31372 | [PDF](https://arxiv.org/pdf/2606.31372v1)

**作者:** Weibin Lin `[一作]` (Beihang University), Zheng Zheng `[通讯]` (Beihang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于任务诱导失败洞察的黑盒优先随机测试（PRT），用于高质量深度强化学习代理的快速失败发现。

**💡 创新点**

将任务难度信息与稀疏性搜索相结合，设计维度约简与局部重组两大机制，实现优先探索失败可能区域的同时保持样本多样性，且时间复杂度仅为 O(MN²)。

**🔧 技术方法**

利用任务诱导失败洞察、维度稀疏性评估、局部重组、线性映射以及非参数熵估计等技术，比较奖励导向的 fuzzing、搜索、生成等现有方法。

**📊 数据集**

在 OpenAI Gymnasium 的 CartPole、LunarLander、MountainCar 与 Humanoid 四个环境上训练 PPO、TQC、DSAC‑T 等 DRL 模型进行实验。

**📈 对比分析**

与 MDPFuzz、CureFuzz、G‑Model、QD 以及随机测试对比，PRT 在首次发现失败的测试用例数量与执行时间上平均提升 40%–50%，并在所有环境中获得最高的测试用例熵。

**⚠️ 局限性**

当失效模式与任务诱导洞察偏离（如 MountainCar 中央失效区）时，PRT 效果下降；在高维环境下线性映射与多维失效模式识别仍受限，仅适用于已知超矩形域。

---

## 290. Calibrating the Evaluator: Does Probability Calibration Mitigate Preference Coupling in LLM Agent Feedback Loops?

**arXiv ID:** 2606.31371 | [PDF](https://arxiv.org/pdf/2606.31371v1)

**作者:** Zewen Liu `[一作]` `[通讯]` (Qilu Institute of Technology), Zewen Liu (Qilu Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究在LLM代理反馈循环中使用概率校准评估者，探究其对评估者偏好耦合的缓解效果。

**💡 创新点**

首次将概率校准方法与评估者反馈结合，证明可显著降低耦合系数和JSD，提供一种轻量级的偏好耦合抑制技术。

**🔧 技术方法**

使用confidence‑calibrated TTRL、滑动窗口等价回归校准、置信度加权的权重更新公式以及深度学习评估器与执行器的交互框架。

**📊 数据集**

采用16个文本任务与8个文本近似视觉任务（共24个任务），每个阶段30轮，评估器为GLM5.2/GPT‑4o，执行器为DeepSeek‑chat。

**📈 对比分析**

通过与标准二值TTRL对比，校准TTRL在γ下降20–49%、JSD下降45–67%；在长度归一化和对称学习率控制下结果依旧显著，验证了校准方法的有效性。

**⚠️ 局限性**

实验仅在单一评估器版本与执行器上进行，残留耦合未完全消除，且校准方法采用简化实现，未来需跨模型验证并进一步理论优化。

---

## 291. Constant-factor approximation of maximum distance-2 independent set in graphs of bounded merge-width

**arXiv ID:** 2606.31369 | [PDF](https://arxiv.org/pdf/2606.31369v1)

**作者:** Maël Dumas `[一作]` `[通讯]` (University of Warsaw), Maël Dumas (University of Warsaw)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于 LP 松弛的贪心算法，给出了在半径‑2 合并宽度受限图类中对距离二最大独立集（α₂）和支配数（γ）的常数因子近似求解方法，并证明了相应的整数规划松弛的整型间隙受限。

**💡 创新点**

创新点在于：
- 将半径‑2 合并宽度与线性邻域复杂度联系起来，利用其带来的结构性限制来控制 LP 的整型间隙；
- 证明了在该图类中存在双重性（γ 与 α₂ 的 LP 形式互为对偶），从而保证了整型间隙同时对两问题均有效；
- 通过对每一步贪心选择时半径 2 球体的 LP 权重进行精细分析，得到一个仅与合并宽度和图的双重性阶数相关的常数上界。

**🔧 技术方法**

主要技术手段包括：
- 设定两类问题的整数线性规划（ILP）与其 LP 松弛，利用双重性关系得到 γ* = α₂*；
- 采用 Chan 等人的浅层细胞复杂度（SCC）结果，将线性邻域复杂度与 LP 整型间隙联系；
- 对合并宽度序列进行分层分析，结合图的双重性阶数，证明存在满足权重上限的“低重量”半径 2 球体；
- 设计贪心算法：在每一步挑选权重最小的半径 2 球体对应顶点，并将该球体内顶点权重置为 0，直到所有权重消失。

**📊 数据集**

本文为理论工作，未使用任何实验数据集；所有结果均来自严格的数学证明。

**📈 对比分析**

相比先前仅在线性邻域复杂度图类或给定低 twin‑width 分解的图类中实现的常数因子近似，本文的算法不需要事先给出 twin‑width 或线性邻域复杂度的证明，只需知道图的半径‑2 合并宽度上界；
 通过整型间隙的证明，算法的近似比可界定为 O(k d²)（其中 k 为半径‑2 合并宽度上界，d 为双重性阶数），在合并宽度受限图类中表现为真正的常数因子。

**⚠️ 局限性**

局限性：
- 该方法仅适用于半径‑2 合并宽度受限的图类，在半径‑1 合并宽度或更一般的图类中，整型间隙可能无界，因而无法得到常数因子近似；
- 需要预先知道合并宽度上界或给定合并宽度序列；若合并宽度不可多项式时间计算，则实际应用受限；
- 对于更高距离的支配/独立问题（如 r‑支配或 r‑距离独立集），虽然可以通过图的 r‑幂来推广，但需要更大的合并宽度上界，导致近似因子随 r 增大；
- 本文未给出算法的实现细节或实验验证，只在理论上给出了近似比和整型间隙的上界。

---

## 292. Language-Assisted Super-Resolution from Real-World Low-Resolution Patches

**arXiv ID:** 2606.31363 | [PDF](https://arxiv.org/pdf/2606.31363v1)

**作者:** Joonkyu Park `[一作]` (Seoul National University), Kyoung Mu Lee `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了 LA-SR 框架，利用深度信息从高质量照片中提取真实低分辨率补丁，进行无配对超分辨率训练。

**💡 创新点**

创新点在于把无配对 SR 转化为语言空间对齐，通过 CLIP 语言监督构造内容与质量两类对比损失，实现无需人工降解的真实图像超分。

**🔧 技术方法**

采用深度估计、CLIP 视觉‑语言模型、对比学习的语言内容损失与质量损失、改进的 VGG 感知损失，以及现有 SR 网络作为基座。

**📊 数据集**

训练使用 DF2K 与 LSDIR 高质量图像；评估在 Set5/Set14/BSD100/General100/Urban100/DIV2K、OST、DRealSR、CameraFusion 以及 AI 生成图像集。

**📈 对比分析**

与无监督、自监督及传统有监督 SR 方法对比，LA-SR 在真实低分辨率图像上获得更高的无参考感知指标（如 BRISQUE、CLIP‑IQA、TOPIQ 等），并在多数据集上显示更优的细节恢复与真实感。

**⚠️ 局限性**

主要局限在于依赖基于深度的 LR 补丁，对动画、老电影等极端降解场景可能产生伪影，并对深度估计的精度较为敏感。

---

## 293. Safe Online Learning via Smooth Safety-Structured Policy Composition

**arXiv ID:** 2606.31320 | [PDF](https://arxiv.org/pdf/2606.31320v1)

**作者:** Hongpeng Cao `[一作]` (Technical University of Munich), Marco Caccamo `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种安全感知策略架构AutoSafe，将安全监测与干预嵌入动作生成过程，实现在线强化学习时的硬约束安全与平滑优化；

**💡 创新点**

创新点在于：1）通过可微凸组合将学习策略与安全优先策略融合，保留梯度流并实现软硬两侧平滑过渡；2）引入可学习的“sharpness”参数p，使干预权重λ随状态安全度动态调整，实现预防性干预；3）利用已认证的安全先验与监测器，避免在线求解最优安全策略；

**🔧 技术方法**

使用了软Actor-Critic（SAC）作为基准算法，安全先验采用基于Lyapunov/CBF的安全策略，干预机制为可微凸组合；在实验中还引入了Simplex等安全滤波器做对比；

**📊 数据集**

数据集/环境包括四个仿真连续控制任务（CartPole、Glucose、3D Quadrotor、Quadruped）以及真实的CartPole实验；

**📈 对比分析**

与传统安全滤波（SimplexRL、CBF）、安全先验融合（AdaLam、Residual）、约束RL（Lyapunov、Lagrangian）等方法对比，AutoSafe在大多数任务中实现更高回报且安全违规率为零，尤其在高维动态任务（Quadrotor、Quadruped）表现优于其他方法；

**⚠️ 局限性**

局限性包括：1）依赖预先构建的安全模型与先验，若模型不准确会限制探索与性能；2）目前主要处理静态/结构化安全约束，难以直接应对时变障碍或随机干扰；3）对高维系统的可扩展性仍需进一步验证。

---

## 294. CSO-LLM: Class Subspace Orthogonalization for Post-Training Backdoor Detection and Trigger Inversion in LLMs

**arXiv ID:** 2606.31309 | [PDF](https://arxiv.org/pdf/2606.31309v1)

**作者:** Zhengxing Li `[一作]` (Penn State University), George Kesidis `[通讯]` (Penn State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套基于类子空间正交化（CSO）惩罚的LLM后训练后门检测与触发器逆向框架，兼具高敏感度和特异性，能直接在离线模型上识别并逆向重构真实触发器。

**💡 创新点**

创新点在于将CSO从图像分类迁移至LLM，构造了对“目标类内在特征”负相关的惩罚实现隐式黑名单功能，同时提出了离散和连续两种优化策略，实现同时检测与触发器逆向。

**🔧 技术方法**

主要技术包括：类子空间正交化（CSO）惩罚、基于内部激活层的余弦相似度约束、离散贪心触发器累积与连续嵌入空间梯度下降、基于顺序统计的p值阈值检测以及红队化超参数选择。

**📊 数据集**

使用SST-2情感分类和Yahoo! Answers话题分类数据集，评估了Flan-T5-small/large、Qwen3-0.6B/4B四种LLM，并对比了多种基线（DBS、PICCOLO、CLIBE、MM、ALT、GBDA、UAT等）。

**📈 对比分析**

与基线相比，CSO连续/离散方法在10个清洁/10个被毒模型上均实现了接近100%的TPR且FPR<5%，在触发器逆向上，尤其是三词“Tell me seriously”触发器，召回率超过90%，显著优于其他方法。

**⚠️ 局限性**

局限性包括：算法对单词级分类假设，未直接适用于多词输出或完全生成式LLM；离散搜索复杂度随触发器长度指数增长；需要红队化超参数调优，模型对强适应性攻击的鲁棒性有限。

---

## 295. Benchmarking Large Language Models on Floating-Point Error Classification

**arXiv ID:** 2606.31308 | [PDF](https://arxiv.org/pdf/2606.31308v1)

**作者:** Lisa Taldir `[一作]` (Université de Perpignan via Domitia), Eric Petit `[通讯]` (Intel Corp)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文基于InterFLOPBench基准，系统评估了17款大型语言模型在源代码静态检测浮点错误的能力。

**💡 创新点**

创新点在于提出多标签浮点错误分类基准InterFLOPBench，并通过精细提示工程和链式推理提升LLM检测效果。

**🔧 技术方法**

主要技术包括提示工程（错误定义与解释请求）、链式推理、以及与FPChecker、Herbgrind等工具的对照验证。

**📊 数据集**

使用了90个C内核共1130个测试样本，涵盖六类浮点错误（cancellation、comparison、division by zero、overflow、underflow、NaN）。

**📈 对比分析**

通过多标签F1评分比较，最新模型如gpt-oss‑120b和Qwen 3.5 27b在所有类别上均超过0.90的F1，低级错误如cancellation与underflow仍保持0.6左右。

**⚠️ 局限性**

局限性包括仅针对C语言、单轮提示、仅六类错误；LLM缺乏形式化保证，推理成本高，且对复杂程序结构的分析仍不完善。

---

## 296. Self-Dual Cyclic Codes with Improved Minimum Distance Estimates via Extending the Chen-Ding Construction

**arXiv ID:** 2606.31294 | [PDF](https://arxiv.org/pdf/2606.31294v1)

**作者:** Bofeng Huang `[一作]` (Sun Yat-sen University), Chang-An Zhao `[通讯]` (Sun Yat-sen University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文在Chen‑Ding构造的基础上，将自对偶循环码的研究扩展到ord_n(q)为偶数的情况，并通过新的设计距离δ与定义集连续零段的分析，给出了新的自对偶循环码参数与更优的最小距离下界。

**💡 创新点**

创新点主要包括：① 针对偶数ord_n(q)的自对偶循环码构造，突破了以往只讨论奇数ord_n(q)的局限；② 通过精细化连续零段的估计，获得了比传统BCH下界更高的平方根下界；③ 对Hermitian自对偶循环码在偶数ord_n(q)下同样给出平方根下界，并给出具体参数；④ 通过设计距离的减小，利用对偶码最小距离的提升，实现了相同码长和维度下更大的实际最小距离。

**🔧 技术方法**

主要技术手段包括：BCH码与其定义集的结构分析、Roos/BCH下界、Plotkin和求逆的根号变换、Hermitian内积与欧氏内积的自对偶条件、以及对cyclotomic coset的离散对称性利用。

**📊 数据集**

本文不依赖外部数据集，而是通过符号运算与理论推导得到码参数，所有结果均为数学推导与理论证明。

**📈 对比分析**

与之前的Chen‑Ding构造相比，本文给出的下界在多数参数组合下至少高出1，且在许多情形下达到了平方根下界甚至更高。实验结果（理论计算）表明，构造出的码在维度相同、长度相同的前提下，最小距离均优于已有公开码表中的下界。

**⚠️ 局限性**

局限性：① 只给出了最小距离的下界，实际最小距离可能更高，但未给出精确值；② 通过定义集连续零段的分析仅利用了有限部分的零，未能完全捕捉对偶码的全部结构，因而仍有改进余地；③ 对于某些参数（如δ接近最大值）下，BCH/ Roos 下界可能已达极限，无法进一步提升；④ 本文未提供编码/译码算法的实现细节，实际性能评估仍需进一步实验验证。

---

## 297. Probabilistic Inversion with Flow Matching

**arXiv ID:** 2606.31288 | [PDF](https://arxiv.org/pdf/2606.31288v1)

**作者:** Baldur Paulwitz `[一作]` (TU Bergakademie Freiberg), Stefan Buske `[通讯]` (TU Bergakademie Freiberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出将流匹配（Flow Matching）方法用于地球物理概率反演，首先在二维速度模型中演示其原理，然后在OpenFWI FlatFaultB数据集上实现二维全波形概率反演。

**💡 创新点**

创新点在于将生成式AI中的流匹配技术迁移到地球物理反演，构建条件/引导流匹配框架并引入无分类器引导（CFG）提升反演精度，同时利用EfficientNet+U-Net混合网络实现对高维波形数据的有效压缩与引导。

**🔧 技术方法**

使用的技术包括连续归一化流（CNF）/流匹配损失、条件流匹配与引导流匹配、无分类器引导、EfficientNet预训练编码器、U-Net结构以及AdamW优化器；通过ODE求解实现反演。

**📊 数据集**

使用的主要数据集为OpenFWI FlatFaultB（54,000个二维速度模型及对应的5个源位置的波形数据），训练时取90%用于训练，10%用于验证。

**📈 对比分析**

方法通过与传统的概率采样（如MCMC）相比，显著降低计算复杂度；实验结果显示在几秒内即可得到多次概率反演，平均误差相对较低（如CFG w=4时平均误差≈0.52·10⁻⁵ s），并能提供不确定性评估。

**⚠️ 局限性**

主要限制包括：需要充分代表真实模型分布的训练数据；当前网络仅适用于固定的采集几何，难以直接迁移到不同的测震布置；对深部或信息缺乏区域的预测依赖于训练分布，易产生不可信的外推。

---

## 298. Sequential sparse Gaussian process quantile regression

**arXiv ID:** 2606.31284 | [PDF](https://arxiv.org/pdf/2606.31284v1)

**作者:** Hugo Nicolas `[一作]` (Inria), Olivier Le Maître `[通讯]` (CNRS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于拉普拉斯近似的稀疏高斯过程贝叶斯分位回归，并设计自适应增量化训练与诱导点更新算法。

**💡 创新点**

通过将预测不确定性分解为条件先验方差和后验诱导变量方差，构建诱导点自适应填充与数据采样机制，并使用方差基准实现模型复杂度自调。

**🔧 技术方法**

采用稀疏高斯过程、拉普拉斯近似、二阶梯度优化、块矩阵求逆、Monte Carlo采样以及拒绝采样等技术。

**📊 数据集**

在合成基准数据集Sabater（多维）和Michalewicz（一维）上进行实验，这两组数据含异方差噪声和多峰结构。

**📈 对比分析**

与预设QMC诱导点分配和均匀数据采集进行对比，本文方法在IMSE下降速度、条件先验方差减小和最终预测精度上均优于传统方案。

**⚠️ 局限性**

拉普拉斯近似在小样本场景下精度有限，缺乏严格的收敛理论支撑，且目前仅实现单分位数估计，未涵盖异方差或多分位联合建模。

---

## 299. MOA: A Profiling-Guided LLM Framework for Memory-Optimization Automation at Codebase Scale

**arXiv ID:** 2606.31368 | [PDF](https://arxiv.org/pdf/2606.31368v1)

**作者:** Jiaxi Liang `[一作]` (University of Hong Kong), Chenxiong Qian `[通讯]` (University of Hong Kong)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 MOA，一个基于 LLM 的框架，自动从运行时分析中挖掘内存反模式，生成静态检查器并自动修复。

**💡 创新点**

创新点在于将动态分析与 LLM 自动化相结合，形成三阶段管线：模式挖掘、检查器合成、补丁生成，并实现跨代码库的无人工干预优化。

**🔧 技术方法**

技术包括 LLM 代理（GPT‑5.1/GPT‑5.1‑Codex）、数据库驱动的动态分析检索、Clang Static Analyzer 检测器生成、基于状态机的补丁生成和 LSP 语法验证。

**📊 数据集**

使用的 dataset 为 OpenHarmony 5.0（100+M 行 C/C++）的 7 个系统服务，结合 Memoro 产生的内存分析数据。

**📈 对比分析**

对比方法为 Clang‑Tidy、PatchAgent 等，结果显示 MOA 检测到 13 条反模式（69.3% 未被 Clang‑Tidy 覆盖），生成 769 份补丁，人工接受率 92.5%，平均堆内存下降 42.2%，二进制体积缩小 10.6%。

**⚠️ 局限性**

局限在于仅支持 C/C++，只针对内存效率，缺乏对功能性优化和可维护性评估，且对非常大规模代码库的 LLM 搜索仍依赖检查器定位。

---

## 300. An Empirical Analysis of High-Performance Computing Education in Germany

**arXiv ID:** 2606.31300 | [PDF](https://arxiv.org/pdf/2606.31300v1)

**作者:** Anna-Lena Roth `[一作]` (Fulda University of Applied Sciences), Jonas Posner `[通讯]` (Fulda University of Applied Sciences)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析了德国102所高校（包含64所大学和38所应用科学大学）的HPC课程与硬件资源，构建了课程内容与集群规模的系统性数据集。

**💡 创新点**

首次实现全国范围内的HPC教育与基础设施双重调查，并通过课程主题覆盖率与集群核心数相结合的方式，对德国HPC教育现状进行定量化评估。

**🔧 技术方法**

采用结构化课程手册审查、主题归类、描述性统计和可视化分析等技术，对课程与集群信息进行系统整理。

**📊 数据集**

使用来自102所高校的课程手册与课程目录、各校HPC集群信息以及Top500超级计算机列表等公开数据。

**📈 对比分析**

通过比较课程主题覆盖比例、集群CPU核心数分布以及与Top500排名的对应关系，对教育覆盖与硬件规模的关联性进行定量比较，显示德国高校HPC课程理论覆盖广泛但实践使用有限。

**⚠️ 局限性**

受限于仅依赖官方文件，可能遗漏实际教学内容；集群访问权限信息不完整；研究仅聚焦德国，缺乏跨国对比；并未深入评估学生实际学习成效。

---

## 301. Domain Adaptive Object Detection via Dual-Stream Bilevel-Cycle Optimization

**arXiv ID:** 2606.31373 | [PDF](https://arxiv.org/pdf/2606.31373v1)

**作者:** Yannan Chen `[一作]` (Sun Yat-sen University), Xiaochun Cao `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于双流双层循环优化的域自适应目标检测框架（DSBCO），同时对分类和回归任务进行循环一致性约束，并通过回归归一化解决训练不稳定和损失爆炸问题。

**💡 创新点**

创新点：①在回归任务中引入归一化策略，将大范围的回归偏移映射到标准分布，避免梯度爆炸；②构建双流分类与回归的双层循环优化，使源域与目标域的特征在两级投影中实现对齐；③结合Mean Teacher与实例一致性蒸馏，提供更稳健的伪标签；④用闭式岭回归取代传统大矩阵求逆，显著提升计算效率。

**🔧 技术方法**

主要技术：Mean Teacher、伪标签生成、实例一致性蒸馏、双流分类与回归循环优化、回归归一化、岭回归、梯度稳定化与特征投影、mAP评估。

**📊 数据集**

实验数据集：Cityscapes、Foggy Cityscapes、BDD100K、KITTI、Sim10K（合成）以及对应的目标域数据。

**📈 对比分析**

与现有方法（HT、MGCAMT、DGCAMT等）在四大域适应场景（天气、场景、相机、合成→真实）对比，DSBCO在mAP/AP上均实现最高或近似最高的成绩，例如在Cityscapes→Foggy Cityscapes mAP 64.7%（比HT高14.3%），在Sim10K→Cityscapes AP 68.6%（比HT高3.1%），整体提升显著。

**⚠️ 局限性**

局限性：尚未在基于Transformer的检测器中验证；伪标签阈值采用固定阈值，可能在复杂背景噪声下不够鲁棒，需要更自适应的阈值策略。

---

## 302. A Self-Negotiation Framework for Ethical Decision-Making during Task Interruptions in Service Robots

**arXiv ID:** 2606.31357 | [PDF](https://arxiv.org/pdf/2606.31357v1)

**作者:** Nele Reichert `[一作]` (University of Bremen), Nico Hochgeschwender `[通讯]` (University of Bremen)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了单台服务机器人在公共环境中对多用户中断请求的伦理决策，提出自协商框架实现内部伦理仲裁。

**💡 创新点**

首次提出内部自协商机制，将用户伦理画像编码为情境依赖的偏好，通过内部轮询式提议与计分实现多用户伦理优先级决策。

**🔧 技术方法**

基于 ROS 2 的生命周期节点实现模块化架构，采用目标‑任务‑动作层次、伦理画像建模、交替提议协议以及数值伦理影响评估。

**📊 数据集**

使用仿真环境中人工生成的10个虚构用户伦理画像与状态（最多10个条件、6个情境偏好），并在 ROS2 Gazebo 里测试。

**📈 对比分析**

通过手工构造基准对比，验证90对用户配对全部满足预期结果；运行时间在1–1.6 秒内，满足实时性要求。

**⚠️ 局限性**

仅在单机、双人或多用户可通过多次双人协商实现，未考虑真实人类数据、外部机器人协作、策略欺骗、任务中断安全及多目标协商的更复杂场景。

---

## 303. Patient-Level Elbow Abnormality Detection: Leakage-Aware Evaluation of Learned Preprocessing, Calibration, and Triage-Oriented Operating Points

**arXiv ID:** 2606.31348 | [PDF](https://arxiv.org/pdf/2606.31348v1)

**作者:** Ahmed Sallam `[一作]` (Istanbul Medipol University), Ahmet Kaplan `[通讯]` (Istanbul Medipol University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在MURA手臂X光数据集上，针对患者层面肘部异常检测，构建了基于DenseNet121的模型并引入可学习的DnCNN预处理前端，比较了原始、CLAHE和多元表示三种预处理管道；

**💡 创新点**

创新点在于提出了患者层面泄漏意识的评估协议，系统地评估了学习型预处理对判别、校准以及高特异性阈值下的三种预处理方式的影响，突出显示了预处理与学习预处理交互的非线性效果；

**🔧 技术方法**

使用了DenseNet121骨干网络、DnCNN噪声学习前端、CLAHE对比度增强、数据增强与多重随机种子训练、基于MONAI的训练框架，并对阈值进行验证集选取；

**📊 数据集**

使用了公开的MURA数据集的肘部X光子集，经过人工质量控制后得到3,860张图像，按70%/15%/15%划分为训练、验证和测试集；

**📈 对比分析**

通过平均5个随机种子得到的AUROC、PR‑AUC、ECE、Brier分数以及在95%特异性阈值下的敏感性进行比较，结果表明原始DenseNet121在判别上与其他预处理差异不大，但加入DnCNN在原始与多元表示上可显著降低ECE和Brier，且在高特异性阈值下能获得更高敏感性；

**⚠️ 局限性**

局限性包括：仅针对二分类（正常/异常），缺乏异常细粒度诊断；仅评估肘部，缺乏跨解剖区域通用性；使用512×512 PNG图像，可能缺乏高分辨率DICOM信息；

---

## 304. RCL-Mamba: A Dual-domain State Space Model for Measurement-oriented Image Restoration in Rotational Sparse-View Scanning Computed Laminography

**arXiv ID:** 2606.31353 | [PDF](https://arxiv.org/pdf/2606.31353v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 305. Wavelet-Optimized Pseudo-3D Accelerated Diffusion Model for Truncated Computed Laminography

**arXiv ID:** 2606.31318 | [PDF](https://arxiv.org/pdf/2606.31318v1)

**作者:** Genyuan Zhang `[一作]` (Chongqing University), Yongning Zhou `[通讯]` (Chongqing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

论文通过提出伪3D加速扩散模型，恢复计算层析成像（CL）中的截断数据，显著扩展有效视场并提高扫描效率。

**💡 创新点**

创新点包括将2D扩散先验与z方向小波正则、翻译不变TI机制以及时间回溯采样（TIFA）相结合，实现高效一致性重建并抑制层间断裂。

**🔧 技术方法**

采用的技术包括二维扩散模型、ADMM小波正则化、翻译不变（TI）小波处理、3D快速采样架构以及基于模型的迭代重建（MBIR）。

**📊 数据集**

使用了模拟数据集（非全局截断和全局截断的PCB切片）和实测CL数据集。

**📈 对比分析**

与FBP、SIRT、TS-FBP、FBPConvNet、IR-SDE等传统和深度学习方法比较，PSNR提升约8~10 dB，SSIM提升至0.98，GMSD下降至0.02，表现显著优于对手。

**⚠️ 局限性**

局限性在于采样速度仍高于工业实际需求，且对限角混叠伪影抑制不足，未来需进一步加速和融合混叠消除策略。

---

## 306. When the Database Fails: Prompting LLM Dialogue Agents for Safe Recovery in Task-Oriented Dialogue

**arXiv ID:** 2606.31307 | [PDF](https://arxiv.org/pdf/2606.31307v1)

**作者:** Mohammad Alijanpour Shalmani `[一作]` (University of Central Florida), Jiann Shiun Yuan `[通讯]` (University of Central Florida)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究LLM驱动的任务导向对话在数据库失败时的恢复方法，提出一种轻量级的结构化提示（Guided‑Retry）以降低幻觉与错误承诺。

**💡 创新点**

创新点在于：①构建针对MultiWOZ和SGD的故障注入基准；②提出仅通过提示即可实现的回退策略；③引入自动与人工验证的承诺安全率（CSR）指标。

**🔧 技术方法**

采用提示工程、规则式幻觉检测（HR、CSR、AAR、UFS）、统计显著性检验（McNemar、t‑检验）以及人工评注。

**📊 数据集**

使用MultiWOZ 2.2（5域）和SGD（20域）的测试集进行故障注入实验。

**📈 对比分析**

在六个开源模型（DeepSeek‑R1、Gemma‑2、Llama‑3、Mistral、Phi‑3、Qwen‑2.5）上与Naive、Inform两种对照策略比较，Guided‑Retry将幻觉率从30.5%降至15.3%（MultiWOZ）和从20.9%降至12.2%（SGD），AAR提升至约83%，统计显著性p<0.001。

**⚠️ 局限性**

限制包括：故障注入为人工合成，幻觉检测依赖正则表达式，未评估多轮恢复与用户交互，模型规模仅在7–9B范围内，未涵盖更大或专有模型。

---

## 307. Editing Everything Everywhere All at Once

**arXiv ID:** 2606.31278 | [PDF](https://arxiv.org/pdf/2606.31278v1)

**作者:** Fabio Quattrini `[一作]` (University of Modena and Reggio Emilia), Silvia Cascianelli `[通讯]` (University of Modena and Reggio Emilia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了MICE，一种无训练、架构无关的多实例并行图像编辑方法，利用多模态扩散Transformer的联合注意力机制，在注意力偏置上进行平滑解耦，以实现多实例的精准编辑并保持全局一致性。

**💡 创新点**

创新点在于：①将注意力偏置作为可学习的平滑解耦策略，避免硬性掩码导致的边缘不连续；②不依赖模型层级或超参数，统一应用于所有层；③通过实例感知的高斯平滑掩码实现局部与全局的平衡；④提出新的高密度并行编辑基准MICE‑Bench，用于评估多实例编辑的鲁棒性。

**🔧 技术方法**

技术手段包括：多模态扩散Transformer（MMDiT）与流匹配框架；在联合注意力的logit上插入加性偏置；基于分割掩码的实例感知平滑；使用VAE编码解码、文本编码器（T5/Qwen）。

**📊 数据集**

使用数据集包括：LoMOE‑Bench（3个编辑/图）与新建的MICE‑Bench（平均8.5个编辑/图，5–40编辑/图）；以及公开的FLUX、SAM3等工具获取分割掩码。

**📈 对比分析**

与多分支（MultiDiffusion、LoMOE、LayerEdit）、正则化（IDAttn）以及基线FLUX/开源/闭源模型（Qwen‑Image‑Edit、Gemini 3 Pro Image）对比。结果表明：在LoMOE‑Bench上MICE在局部CLIP分数、背景保留和尝试率上均优于对手；在MICE‑Bench上MICE在局部CLIP分数上更胜一筹，并在LLM‑评判中获得最高ELO分数，显示出更好的编辑精度与整体质量。

**⚠️ 局限性**

局限性包括：对分割掩码的依赖，若掩码粗糙或误检会影响编辑效果；在极大并行编辑（>15个实例）时内存与推理时间仍随实例数线性增长；缺乏针对动态/视频场景的评估。

---

## 308. Spatial Model Checking of Images via Minimised Models and Branching Bisimilarity

**arXiv ID:** 2606.31344 | [PDF](https://arxiv.org/pdf/2606.31344v1)

**作者:** Vincenzo Ciancia `[一作]` (Istituto di Scienza e Tecnologie dell'Informazione A Faedo Consiglio Nazionale delle Ricerche), Erik P. de Vink `[通讯]` (Eindhoven University of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对基于闭包空间的空间模型（尤其是二维/三维数字图像），提出一种利用分支分叉分歧（branching bisimilarity）对空间模型进行最小化的方法，并将最小化结果与空间逻辑模型检查结合，形成一套完整的工具链。

**💡 创新点**

创新点在于：①给出了从有限闭包模型到标记转移系统（LTS）的精确编码，证明该编码在兼容路径分歧（Compatible Path）分歧下保持等价性；②利用已有的高效分支分叉最小化算法，得到的最小模型可直接用于空间逻辑的模型检查；③设计了面向图像的可视化投影机制，使最小化后模型的检查结果能恢复到原始像素空间；④通过实验验证了该方法在不同尺寸图像（迷宫、监视器模式、Pac‑Man 场景）上的显著加速。

**🔧 技术方法**

核心技术包括：闭包空间与图论的统一框架、兼容路径分歧的定义与逻辑（∃∞Compatible Reachability Logic），LTS 编码与分支分叉分歧的对应证明，使用 mcrl2 工具套件中的分支分叉最小化算法，以及图像处理库（ITK/SimpleITK）进行模型构造与可视化。

**📊 数据集**

使用的实验数据集有：①不同分辨率的迷宫图像（从 16×16 到 4096×4096），② Philips PM5544 监视器模式图像（从 30×16.9 到 1920×1080），③ Pac‑Man 场景图像（从 39×39 到 15260×15260）。

**📈 对比分析**

比较方法：在同一套实验环境下，分别用原始完整模型与最小化后模型执行空间逻辑模型检查（Vox 或 VoxLite 等工具），记录编码时间、最小化时间、模型检查时间等指标。结果显示：对大尺寸图像，模型检查时间可提升 3–17 倍；最小化时间与模型大小近线性；整体工作流时间显著下降。

**⚠️ 局限性**

局限性包括：①工具链目前仍使用中间文件 I/O，导致文件写/读开销；②分支分叉最小化仅适用于对称闭包模型，非对称情况仍需改进；③与专门针对图像的 VoxLite 相比，使用通用模型检查器（mcrl2）导致加速低估；④最小模型的构造和结果投影仍需进一步优化以适用于更大规模或实时场景。

---

## 309. Witness Complexity of Short Descriptions: A Cryptographic Perspective

**arXiv ID:** 2606.31370 | [PDF](https://arxiv.org/pdf/2606.31370v1)

**作者:** Fabio F. G. Buono `[一作]` `[通讯]`, Fabio F. G. Buono

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出并研究了 witness complexity (x)，即在给定近最短描述的情况下，恢复对象所需的最小运行时间，并给出了其在密码学、可压缩性和可验证性等场景中的应用。

**💡 创新点**

创新点在于：1) 形成了一个新的信息量度 (x)，填补了 Kolmogorov 复杂度与 Shannon 熵之间的空白；2) 证明了该量度的机器不变性、条件与无条件下的分离与下界；3) 给出了对 NP 证据可解性与可压缩性的双向判定；4) 发现了结构化实例下的多项式可解性；5) 引入了自适应复杂度、输出开销与结构熵等伴随量。

**🔧 技术方法**

主要技术包括：通用图灵机模拟与编译器引理、Kolmogorov 复杂度与 Levin 复杂度的理论框架、证书可证明性自降与多项式时间多路复用技术、结构化家族的多项式时间构造以及语法压缩的最小语法问题等。

**📊 数据集**

该工作为纯理论论文，未使用具体数据集；所有证明均基于信息理论与计算复杂性理论。

**📈 对比分析**

在理论上与现有的 Kolmogorov 复杂度、时间受限 Kolmogorov 复杂度、Levin 复杂度等度量进行了比较，证明 (x) 既独立于信息内容也捕捉到可执行性；在语法压缩例子中，展示了在相同最小语法大小下展开深度的巨大差异。

**⚠️ 局限性**

局限性包括：(x) 本质上不可计算；对具体实例的上界与下界仍存在巨大空隙；在随机实例或非结构化实例上缺乏可证明的多项式上界；开放问题未解决，如 (x) 与时间受限 Kolmogorov 复杂度之间的具体函数关系、结构熵与 (x) 的关系等。

---

## 310. Beyond Binary Instrument QA: Probing Instrument Grounding in Music Audio-Language Models

**arXiv ID:** 2606.31338 | [PDF](https://arxiv.org/pdf/2606.31338v1)

**作者:** Yujun Lee `[一作]` (Sungkyunkwan University), Kyuhong Shim `[通讯]` (Sungkyunkwan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个从二元问答到多维诊断的OpenMIC导向基准序列，用于评估音乐音频‑语言模型在乐器定位与辨识上的真实地面化能力。

**💡 创新点**

设计了包含genre‑prior‑reduced QA、confusion‑aware discrimination、多标签长时上下文识别与时序定位等四个维度的诊断基准，揭示高准确率背后的选项位置、乐器标签与时间段偏差。

**🔧 技术方法**

采用多模型评估（Music Flamingo、Audio Flamingo 3、Qwen2.5‑Omni、GPT‑4o‑audio、Gemini 2.5系列），并用最大‑最小预测率、混淆矩阵、时间段偏差等量化方法分析模型偏差。

**📊 数据集**

基于OpenMIC 2018的音乐片段，构造了二元QA（9332）、genre‑reduced QA（590）、instrument discrimination（1051）、长时多标签（1028）和时序定位（3579）等子集。

**📈 对比分析**

通过每个子基准的准确率、F1、exact‑set等指标与基线比较，发现二元QA准确率>87%，但在复杂诊断任务中性能大幅下滑，且出现显著的选项位置、乐器标签和时间段偏差。

**⚠️ 局限性**

仅聚焦乐器地面化，未涵盖和声、结构、歌词等音乐理解维度；基准设计依赖人工定义的混淆组，可能不完全代表真实听感混淆；模型推理方式受限于所选语言模型与prompt模板。

---

## 311. Bridging Video Understanding and Generation in a Unified Framework

**arXiv ID:** 2606.31326 | [PDF](https://arxiv.org/pdf/2606.31326v1)

**作者:** Yuqi Wang `[一作]`, Mingyu Guo `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个统一的视频生成与理解框架，将两者融合在同一模型中；

**💡 创新点**

创新点包括：将自回归语义预测与扩散渲染相结合、共享视觉文本离散代码表、双流选择机制压缩视觉令牌、噪声控制视觉条件与生成监督等；

**🔧 技术方法**

使用的技术包括自回归Transformer与共享词表、TA‑Tok视觉令牌化器、扩散解码器、双流选择与视觉监督、噪声控制视觉条件；

**📊 数据集**

训练数据分两阶段：第一阶段约1亿张图文样本（Blip3o、JourneyDB、DenseFusion、FineVision、LLaVA‑OneVision、Infinity‑mm），第二阶段约500万段视频样本（Koala‑36M、OpenVid‑1M、LLaVA‑Video‑178K）以及若干图像（ShareGPT‑4o‑Image、Blip3o‑60k、Echo‑4o）；评测基准包括VBench、VBench++、VideoMME、LongVideoBench、Egoschema、Next‑QA、MLVU；

**📈 对比分析**

与专业与统一模型对比，3B参数模型在VideoMME上达69.4、LongVideoBench上达60.1，生成方面在VBench上表现领先；实验表明视觉令牌条件优于文本条件，双流选择提升时序建模；

**⚠️ 局限性**

局限性在于数据规模与模型大小受算力限制，尚未在强化学习或大规模上下文学习场景中验证；

---

## 312. LOPA: Enhancing Spoken Language Assessment via Latent Ordinal Prototype Alignment

**arXiv ID:** 2606.31310 | [PDF](https://arxiv.org/pdf/2606.31310v1)

**作者:** Hong-Yun Lin `[一作]` (National Taiwan Normal University), Berlin Chen `[通讯]` (National Taiwan Normal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `57a58b01-81b4-4d75-a45c-2e891f272b50` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种不需大模型微调的轻量级 Whisper‑仅框架，利用多层特征融合与潜在序数原型对齐实现口语水平评估。

**💡 创新点**

引入语义锚定层路由（SALR）挑选多深层 Whisper 表征，并结合潜在序数原型对齐（LOPA）正则化，使潜在空间显式遵循 CEFR 的序数结构。

**🔧 技术方法**

使用 Whisper 大模型冻结编码器、多层特征堆栈、SALR、注意力时序池化、线性投影头以及 LOPA 的原型吸引与序数约束损失。

**📊 数据集**

在 Speak & Improve 2025 语料库的开放式口语评估部分（P1、P3、P4、P5）上进行实验，使用人类评分的 CEFR 半点等级。

**📈 对比分析**

与 Whisper 最后层、wav2vec2、BERT cascaded、手工特征加 Whisper+BERT、以及 Phi‑4 多模态 LLM 进行对比；在评估集上实现 RMSE 0.361、PCC 0.828，接近或优于大型多模态模型。

**⚠️ 局限性**

局限性在于仅在单一数据集验证，缺乏跨语言或跨文化的鲁棒性评估；并且依赖 Whisper 编码器的中间层特征，若预训练不适合新语种可能受限。

---

## 313. Patch-PODiff-ViT: Structured Latent Diffusion with Patchwise POD for Super-Resolution and Uncertainty Quantification

**arXiv ID:** 2606.31290 | [PDF](https://arxiv.org/pdf/2606.31290v1)

**作者:** Onkar Jadhav `[一作]` (University of Western Australia), Nicole L. Jones `[通讯]` (University of Western Australia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Patch‑PODiff‑ViT，一种在固定的局部POD基空间中进行条件扩散的框架，实现高效超分辨与可解释的不确定性估计。

**💡 创新点**

创新点在于将传统POD的局部正交低秩基作为线性可解释的潜在空间，配合Vision Transformer去噪，并通过线性解码器实现对像素空间方差的解析传播。

**🔧 技术方法**

采用Patch‑wise POD编码、Vision Transformer去噪器、DDPM/DDIM扩散过程、线性解析不确定性传播和条件超分辨技术。

**📊 数据集**

在海面温度（SST）日常观测、NIH胸部X光（Chest X‑ray14）以及FFHQ人脸图像上进行实验。

**📈 对比分析**

与U‑Net、VAE‑LDM、DiT、PixelDiff以及全场PODiff进行对比，Patch‑PODiff‑ViT在RMSE/PSNR/SSIM/LPIPS/FID等指标上均优于对手，同时参数量（70M）和显存、推理时间大幅下降，且不确定性校准度最高。

**⚠️ 局限性**

仅适用于局部低秩结构明显的场景；POD基线固定，分布漂移需重新构造；采用块对角近似忽略跨块协方差，导致部分细节不完全。

---

## 314. Spatial Reasoning via Modality Switching Between Language and Symbolic Representation

**arXiv ID:** 2606.31285 | [PDF](https://arxiv.org/pdf/2606.31285v1)

**作者:** Shreya Rajpal `[一作]` (Michigan State University), Parisa Kordjamshidi `[通讯]` (Michigan State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究将空间叙事映射为网格/结构化表示，提升LLM多跳空间推理，并提出基于可信度与复杂度的自适应模式切换策略；

**💡 创新点**

①引入网格/结构化表示显著提高多跳推理；②提出可信度（faithfulness+plausibility）与复杂度联合切换机制，实现自然语言与结构化表示的自适应选择；③证明切换在不显著增加成本的前提下提升准确率；

**🔧 技术方法**

使用大语言模型（LLaMA3.1-70B、Qwen3-32B、GPT-5.1）、少量提示关系提取、将三元组转化为二维网格、信度评估与复杂度估计模块，以及自适应切换决策；

**📊 数据集**

StepGame、SpaRTUN、ReSQ三大空间推理基准数据集；

**📈 对比分析**

与文本、关系三元组、坐标布局、完整/剪裁网格及ToT-CoT等方法对比；网格（尤其剪裁网格）在StepGame 10跳提升至最高76.7%，相较文本提升约42%；在SpaRTUN和ReSQ也取得最高25%或5%的准确率提升；自适应切换可匹配或超越网格单一模式，同时降低成本；

**⚠️ 局限性**

依赖于精确的关系提取与网格构造，错误会影响后续推理；自适应切换增加额外计算开销；需要更鲁棒的提取与更轻量的路由策略。

---

## 315. Stage-Transition Dense Reward Modeling for Reinforcement Learning

**arXiv ID:** 2606.31377 | [PDF](https://arxiv.org/pdf/2606.31377v1)

**作者:** Yang Yang `[一作]` (Tsinghua University), Houde Liu `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种基于视觉的奖励学习框架 STDR，能够从无结构的专家视频中自动分解任务阶段并生成稠密奖励，驱动机器人在长时域操作任务中从零开始学习；

**💡 创新点**

核心创新在于：①利用 VLM 对视频进行语义阶段分割，自动生成伪标签；②设计阶段判别器与进度估计器联合产生阶段转移与内部进度两种反馈；③引入 OOD 检测与抓取验证模块，防止奖励被攻击并提升稳健性；④将多种技术（LSTM、VAE、FiLM、Mahalanobis 距离）结合，实现全时域一致且细粒度的奖励；

**🔧 技术方法**

使用的技术包括 Qwen3 视觉‑语言模型进行阶段划分、LSTM‑基阶段判别器、VAE‑OOD 监测、MLP‑抓取监管、FiLM‑调制的 ResNet 进度估计器、Mahalanobis‑距离 OOD 门控、检索‑基进度补偿、PPO 强化学习、VIP/LIV 预训练的视觉编码器；

**📊 数据集**

实验数据来自 MetaWorld、ManiSkill 与 Franka Kitchen 三大仿真环境（每任务30条专家演示），以及真实 Franka‑Arm + RealSense 摄像头收集的 15 条成功与失败视频；Frank Kitchen 亦使用 D4RL 轨迹；

**📈 对比分析**

与稀疏奖励、人工稠密奖励、VIP 与 LIV 进行对比，STDR 在 14 个任务中实现更快样本效率、成功率 ≥1.0（大多数任务在 1M‑2M 步内收敛），并在 ManiSkill 与 Franka Kitchen 中明显优于 VIP/LIV，甚至在高难度任务上超过手工稠密奖励；真实机器人评估显示 STDR 的奖励随任务进展稳定、失败时低奖励，优于基线；

**⚠️ 局限性**

局限性包括仅依赖视觉输入，缺乏本体、触觉等多模态信息；对任务阶段可交换或变形的适应性不足；未在真实机器人上完成从奖励学习到策略训练的完整闭环；未来需加入多模态感知并提升对序列变异的鲁棒性。

---

## 316. Mutating the "Immutable": A Large-Scale Study of Git Tag Alterations

**arXiv ID:** 2606.31354 | [PDF](https://arxiv.org/pdf/2606.31354v1)

**作者:** Solal Rapaport `[一作]` (LTCI, Télécom Paris, Institut Polytechnique de Paris), Théo Zimmermann `[通讯]` (LTCI, Télécom Paris, Institut Polytechnique de Paris)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对公开 Git 仓库进行大规模量化研究，系统统计并分类 Tag 的修改（移动、删除）事件，并评估其对构建可复现性和软件供应链安全的影响。

**💡 创新点**

首次提出基于归档快照的 Tag 变更检测与分类方法，构建了 Tag 变更的层级分类法（内容变更、提交元数据变更、标签层级变更），并将发现的 Tag 变更与实际包管理器（如 Nixpkgs）和供应链攻击案例关联，证明 Tag 可变性对可复现性与安全构成实质威胁。

**🔧 技术方法**

使用归档平台 Software Heritage 的持续快照（SWHID）做时序对比；实现自动化脚本对连续快照进行 Tag 解析、轻量/注解区分、目标 Commit/SWHID 对比，并将结果归类为移动或删除；随后进行统计分析和相关性检验（与仓库星标、标签变更频率等）。

**📊 数据集**

基于 Software Heritage 对 GitHub、GitLab、Bitbucket 等多家托管平台的约 200 万个仓库进行抽样，覆盖 2024-2026 年间的多次快照，构成包含 10+ 万次 Tag 变更的规模化数据集。

**📈 对比分析**

通过对 Tag 变更次数、占比、按仓库人气分布、内容变更比例等指标进行定量评估；与 Nixpkgs 的 hash 匹配情况对比，发现 7% 的包因 Tag 变更导致哈希不匹配；在供应链攻击案例中，报告 346 个 Tag 同时被重定向，验证了方法对真实威胁的检测能力。性能方面，整个分析脚本在单台机器上完成 200 万个仓库的 Tag 变更检测耗时约 48 小时，吞吐量约 4k 仓库/小时。

**⚠️ 局限性**

局限性包括：① 归档快照频率不均，低活跃仓库的变更可能被漏检；② 仅通过归档快照无法推断变更动机（维护、错误修正还是恶意）；③ 仅关注 Tag，未覆盖分支、提交等其他可变引用；④ 统计结果为下限估计，实际变更率可能更高。

---

## 317. Dualformer: Efficient Feature Extractor for Complex-valued Blind Communication Signal Analysis

**arXiv ID:** 2606.31352 | [PDF](https://arxiv.org/pdf/2606.31352v1)

**作者:** Yurui Zhao `[一作]` (National University of Defense Technology), Zhitao Huang `[通讯]` (National University of Defense Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 DualNN 及其基于 Transformer 的实现 Dualformer，用共享参数的双通道网络实现对复数通信信号的特征提取，并在 AMR、SSR、SSP 等盲信号分析任务中进行实验。

**💡 创新点**

设计了参数共享的双通道网络框架，理论证明其泛化误差比传统复数或实数网络更低，并构建专门处理 IQ 信号的多尺度 Transformer（Dualformer）。

**🔧 技术方法**

采用深度学习中的 Transformer 自注意力机制、Patch 分割、共享权重的实数网络、2D 卷积投影头、BatchNorm、ReLU、Softmax 等技术。

**📊 数据集**

使用合成 AMR 数据集、RML 2018 语音数据、以及仿真 SSR 与 SSP 数据集（共 17 种信号方案），并在不同 SNR 条件下进行评估。

**📈 对比分析**

与四种经典 DL 模型（CNN1/2、LSTM、ResNet）以及三种 Transformer 变体（Autoformer、Informer、Transformer）在 AMR、SSR、SSP 任务上进行对比；Dualformer 在所有 SNR 范围内均超越对手，取得最高准确率、F1、RQ、SQ、PQ 等指标。

**⚠️ 局限性**

对 I/Q 不平衡的鲁棒性略低于纯复数网络；在极低样本或极端信号失真场景下仍需提升；目前验证主要基于仿真数据，真实环境中的泛化性能仍待进一步研究。

---

## 318. Automated High-Precision Extraction and Forensic Verification of Data-Bearing Vector Figures

**arXiv ID:** 2606.31345 | [PDF](https://arxiv.org/pdf/2606.31345v1)

**作者:** Bowen Sun `[一作]` (Johns Hopkins University), Chaowei Xiao `[通讯]` (Johns Hopkins University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一套自动化、可证实的向量图表数据提取与重渲染验证方法，实现标记数据位精确、折线数据约3-4位有效数字的高精度恢复。

**💡 创新点**

创新点在于结合精度理论、无碰撞性证明与重渲染证书，提供可证明的高精度提取与验证，彻底解决手工数字化的低精度、低自动化和无验证缺陷。

**🔧 技术方法**

使用白盒解析 PDF/SVG/EPS 绘图指令，恢复坐标系并逆算数据；轴标校准、增量求和、重渲染比对；以及概率论给出的无碰撞与偶然匹配界限。

**📊 数据集**

在 Planck 2018 温度功率谱、Keeling CO₂ 年平均记录、Chinchilla 计算量‑参数折线等公开图表上进行验证，并对多款绘图渲染器与导出格式进行精度测评。

**📈 对比分析**

通过与官方数据或已知控制数据对比评估，标记数据误差达 10⁻⁹，折线误差约 5×10⁻⁴；验证器可在几秒内完成数百条数据的提取与比对，性能随点数线性增长。

**⚠️ 局限性**

局限性包括：低精度渲染器（如 R PDF）碰撞区间较大，无法实现位精确恢复；校准误差是主要瓶颈；仅适用于向量图，无法处理光栅图；需要可读的轴标文本进行校准。

---

## 319. Optimization Algorithms for Joint OFDM Waveform Design and RIS Configuration in 6G Networks: From Convex Relaxation to Foundation Models

**arXiv ID:** 2606.31334 | [PDF](https://arxiv.org/pdf/2606.31334v1)

**作者:** Ahmet Kaplan `[一作]` `[通讯]` (Istanbul Medipol University), Ahmet Kaplan (Istanbul Medipol University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对2021-2026年间涉及联合OFDM波形与RIS配置优化的78篇论文进行了系统综述，并提出了四范式（模型基凸优化、启发式/元启发式搜索、深度学习及新兴方法）分类框架；同时给出统一的教程基准（N=16、64、128）并对比四类算法在谱效率与运行时的表现；

**💡 创新点**

创新点包括：①首次构建完整的四范式分类体系，覆盖SUM‑RATE、能效、公平性与PAPR约束；②通过自建基准实验实现跨范式的量化对比；③提出标准化基准的最低要求与六大开放挑战；④提供可复现的四范式算法案例，展示速度与性能的实证差异；

**🔧 技术方法**

所用技术涵盖：凸松弛（Semidefinite/SDP、SCA）、迭代求解（AO+SCA、PSO、SA）、深度强化学习（DDQN、PPO、GNN）、无监督深度学习、基础模型/扩散模型生成AI、量子优化（量子GNN等）；

**📊 数据集**

数据集/实验设置：基于仿真基准，采用N=16、64、128、M_t=2、K=2、M=4、2bit量化等参数；不使用公开真实数据集，主要通过模拟生成的信道与系统参数进行评估；

**📈 对比分析**

比较方法：对照自报基准，使用统一硬件（GPU）和相同系统规模，测量GPU推理时延与传统迭代求解时延；结果显示：DRL方法在93‑97%谱效率的同时实现3000‑6000×的速度提升；启发式方法能匹配或超越AO+SCA的谱效率，但耗时较长；SDR在N=16时仅需2.3ms，N>100时耗时显著增加；

**⚠️ 局限性**

局限性：①缺乏统一的标准化基准，导致文献自报指标难以直接比较；②本文仅涵盖下行RIS‑OFDM，未讨论上行、调度与硬件实现细节；③能效与PAPR约束实验留待后续研究；④基准规模受限，未涵盖大规模RIS与复杂双扩展信道情景；⑤多模态方法的硬件实现与安全性问题尚未充分验证；

---

## 320. Expected Gain-based Escalation in Vertical Federated Learning

**arXiv ID:** 2606.31331 | [PDF](https://arxiv.org/pdf/2606.31331v1)

**作者:** Mohamad Mestoukirdi `[一作]` (Mitsubishi Electric R&D Centre), Vincent Corlay `[通讯]` (Mitsubishi Electric R&D Centre)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种在垂直联邦学习中，在两轮推理协议下基于期望增益的路由规则，实现仅在必要时升级到昂贵的协同推理；

**💡 创新点**

创新点在于利用已校准的池化后概率和协同模型的类别级可靠性估计，构造无学习器的解析期望增益分数，既可解释又无需额外训练；

**🔧 技术方法**

采用多分类 Dirichlet 归一化校准、混淆矩阵统计、平均池化推理与嵌入融合等技术；

**📊 数据集**

在 CIFAR-10、CIFAR-100 与 ModelNet40 三个多视图分类基准上进行评估；

**📈 对比分析**

与学习型路由器、置信度路由器、学习转移和理想（oracle）基线对比，实验显示该解析路由在多种噪声条件下的通信-准确率曲线均优于现有可部署方法，且逼近 oracle 但仍有提升空间；

**⚠️ 局限性**

局限包括：依赖类别级可靠性近似，忽略同一类别内样本的差异；以及对校准数据分布的敏感性，分布漂移或样本稀缺时可靠性估计可能不稳健。

---

## 321. 3D HAMSTER: Bridging Planning and Control in Hierarchical Vision Language Action Models through 3D Trajectory Guidance

**arXiv ID:** 2606.31329 | [PDF](https://arxiv.org/pdf/2606.31329v1)

**作者:** Dongyoon Hwang `[一作]` (KAIST), Jaegul Choo `[通讯]` (KAIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个层次化视觉语言动作框架，直接让预训练的视觉‑语言模型（VLM）输出可度量的 3D 轨迹（u,v,d），并将其与点云基础的低层策略耦合，实现端到端机器人操作。

**💡 创新点**

创新点在于：①将高层规划从传统 2D 关键点改为 3D 轨迹，消除 2D→3D 升级导致的几何失真；②在 VLM 中加入深度编码器和密集深度重建损失，逼迫模型保持场景的几何一致性；③采用两阶段训练保持 VLM 原有能力，同时实现 3D 轨迹预测。

**🔧 技术方法**

使用的关键技术包括：Qwen3‑VL 预训练 VLM + 额外深度编码器 + 密集深度重建损失 + LoRA 微调；3DFA 点云低层策略 + 轨迹条件的流匹配；两阶段训练（对齐 + 任务微调）；端到端闭环控制。

**📊 数据集**

训练数据涵盖 8 大来源：RLBench、DROID、InternData‑M1、RefSpatial（3D 能力数据），以及 RoboPoint、PixMo、LVIS、Honey‑1M（保留数据）用于防止功能遗忘。评估使用自建的 DROID‑pick‑and‑place 测试集、Colosseum 仿真基准，以及真实 Franka Panda 机器人实验。

**📈 对比分析**

与专有 VLM、RoboBrain、HAMSTER（2D 轨迹）和无指导基线比较。3D 轨迹预测准确率最高 63.5%（5cm Both），Colosseum 上平均成功率 44.8%（比 2D 提升 6%），真实任务中 3D 轨迹使 3DFA 成功率提升至 80%/68%/62%（比 2D 提升 20% 以上），显示显著性能提升。

**⚠️ 局限性**

局限性包括：①依赖 RGB‑D 传感器；②单视角导致遮挡敏感；③仅在单臂平面抓取、倾倒等简单任务上验证，尚未扩展到移动、双手或更复杂场景。

---

## 322. HistoriQA-ThirdRepublic: Multi-Hop Question Answering Corpus for Historical Research, Parliamentary Debates from the French Third Republic (1870-1940)

**arXiv ID:** 2606.31325 | [PDF](https://arxiv.org/pdf/2606.31325v1)

**作者:** Aurélien Pellet `[一作]` (EPITA), Marie Puren `[通讯]` (EPITA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了HistoriQA-ThirdRepublic数据集，包含单跳和多跳历史问题，并通过历史学家参与验证；同时设计了基于RAG的多跳问题生成与评估方法。

**💡 创新点**

创新点在于将历史议会辩论与报纸文本跨源关联，生成既涉及时间推理又需跨文献证据的多跳问题；并引入史学家主导的质量评估与LLM评判者对齐评估。

**🔧 技术方法**

使用检索增强生成（RAG）技术，包括基于嵌入的检索、重新排序、Cohere API的模型；生成时采用Prompt工程；评估时使用LLM-as-a-judge来衡量答案正确性。

**📊 数据集**

数据集来源于1870-1940年法国第三共和国议会辩论（1887年），以及同期的两份报纸《Le Gaulois》和《L’Intransigeant》；文本已转为纯文本并进行分块。

**📈 对比分析**

通过Recall@K、Accuracy@K、MRR@K以及答案准确率评估RAG性能；单跳检索表现良好（Recall@3≈68%），跨源多跳检索较弱（跨报纸Recall@3≈14%，跨报纸-议会≈48%）；答案准确率单跳最高可达58.85%，多跳仅约18%。

**⚠️ 局限性**

局限性包括：过度依赖Cohere模型，单一历史学家验证、仅限1887年及两份报纸、LLM评判样本有限，且通过筛选得到的高置信子集可能高估整体性能。

---

## 323. From Idea to Prototype in an Afternoon: Scaffolded, AI-Assisted Rapid VA Prototyping

**arXiv ID:** 2606.31311 | [PDF](https://arxiv.org/pdf/2606.31311v1)

**作者:** Gennady Andrienko `[一作]` (Fraunhofer Institute IAIS), Natalia Andrienko `[通讯]` (Fraunhofer Institute IAIS)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

结合 ATWL 工作流语言与大型语言模型，在半天内实现了“软天际线上的星座”多准则决策可视化原型；

**💡 创新点**

提出将容差松散的 Pareto 前沿转化为软前沿并通过离散级别的 NMF 自动生成可解释的星座，同时证明 AI 辅助与工作流 scaffold 可极大加速原型开发；

**🔧 技术方法**

使用 Artifact–Transform Workflow Language (ATWL) 作为 scaffold、Claude Opus 4.8 作为代码生成器、离散化阈值方法与 NMF、软天际线公式以及交互式地图可视化等技术；

**📊 数据集**

采用德国租房列表（Kaggle Leipzig 子集）约 13,000 条记录；

**📈 对比分析**

与无 scaffold 的单一 LLM 设计相比，ATWL + 库在流程完整性、类型正确性与架构复杂度上提升；实验显示多轮 scaffold 可保持创意并提高正式度；原型实现时间从数周缩短到数小时；

**⚠️ 局限性**

依赖专家已知方法、仅适用于已在 ATWL 框架中的问题、容差模型不适用于整数/分类指标、知识注入仍需人工，且实验仅在单一团队单一模型上验证。

---

## 324. Deep Spectral Models for Robust Dental Shape Generation

**arXiv ID:** 2606.31293 | [PDF](https://arxiv.org/pdf/2606.31293v1)

**作者:** Tibor Kubík `[一作]` (Polytechnique Montréal), Hervé Lombaert `[通讯]` (Polytechnique Montréal)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究并实现了一种基于同步谱系数的牙冠形状生成框架 ToothForge，能够在不同网格连通性下学习牙冠形态的非线性潜在空间并生成高分辨率、点对点对应的三维模型。

**💡 创新点**

创新点在于将 Laplace–Beltrami 谱系数进行同步对齐，使得不同网格可映射到统一的内在基底；在该同步谱空间中采用 β-VAE 训练，从而实现对形状变异的紧凑、可解释的非线性建模，并且整个训练与优化仅在谱域完成，避免了外在坐标的冗余学习。

**🔧 技术方法**

技术主要包括谱同步、β-VAE、卷积编码器/解码器、Chamfer 损失、周期性 β 调度、Poisson 场重建等。

**📊 数据集**

使用工业合作伙伴提供的 430 颗牙冠三角网格数据（149 切牙、161 预磨牙、120 磨牙），平均 11484 顶点，已完成稠密对应并划分 80/20 训练/测试。

**📈 对比分析**

与传统 PCA 统计形状模型、基于 PointNet++ 的 PointVAE 以及点云扩散模型 PointDiffusion 进行对比；在 MMD、覆盖率、重建误差、采样速度等指标上，ToothForge 在 256 模式下达到 0.0997 的 MMD、43.78% 覆盖率，重建误差仅为 0.09 mm，采样时间 1 ms；相比之下 PCA 为 0.1541、37% 等，PointVAE/PointDiffusion 需要更长训练/采样时间且生成质量略差。

**⚠️ 局限性**

局限在于目前仅针对牙冠而非根部或全口；对高频细节的再现仍略显平滑；谱同步依赖于选定参考模板，若训练与推理使用不同模板会导致误差；需要预先完成网格对应，且对极端形状或少样本情况下可能仍需改进。

---

## 325. AtomiMed: Hierarchical Atomic Fact-Checking for Universal Clinical-Aware Medical Report Evaluation

**arXiv ID:** 2606.31292 | [PDF](https://arxiv.org/pdf/2606.31292v1)

**作者:** Yuan Wang `[一作]` (Zhejiang University-University of Illinois Urbana-Champaign Institute), Jianpeng Zhang `[通讯]` (Alibaba Group)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 AtomiMed 框架，用于将医学报告拆解为疾病级和属性级原子事实，并通过双向 Agentic Cross-Verification 模拟放射科同行评审，实现多模态医学报告的结构化、可解释评估。

**💡 创新点**

创新点包括：① 采用两级原子事实层级（疾病+属性）统一描述所有医学报告；② 用双向验证循环实现诊断检测与描述准确性的分离；③ 在多模态、跨学科上保持通用性；④ 开放了 MRGEvalKit 工具和 OmniMRG-Bench 基准，提供可复现的评估流程。

**🔧 技术方法**

技术手段：指令调优的大语言模型（LLM）进行 JSON 原子事实抽取与交叉验证；双向问答验证计算精度/召回/F1；模糊匹配（θ=0.8）对疾病名称进行对齐；Kendall τ、Spearman ρ 等统计评估指标；实现基准工具 MRGEvalKit 与 OmniMRG-Bench。

**📊 数据集**

使用的数据集包括：OmniMRG-Bench（178K+ ACF，覆盖 X‑ray、CT、MRI、超声）、ReXVal、ReFiSco‑v0、RadEvalX、RaTE‑Eval 四个专家标注基准，及公开的 MIMIC‑CXR、CT‑RATE、AMOS、RadGenome、KMVE 等多模态医学数据集。

**📈 对比分析**

通过与传统 NLG 评估（BLEU、ROUGE、METEOR）、嵌入基评估（BERTScore）、医学专用指标（RadGraph、SembScore、RaTEScore）以及 LLM‑as‑Judge（GREEN）对比，AtomiMed 在专家判定相关性（Spearman ρ 最高 0.806，Kendall τ 最高 0.642）和 pairwise preference（X‑ray ACC 95.71%，CT 84.33%，Ultrasound 49.86%）均显著优于其它方法，同时提供每条原子事实的错误归因。

**⚠️ 局限性**

局限性：① 依赖 LLM 推理，推理成本高；② 目前属性层级覆盖范围有限，未能涵盖所有临床专科或纵向时间维度；③ 在非胸部模态（如 MRI、超声）上仍存在性能下降；④ 需要进一步压缩模型与实现更高效的评估流程。

---

## 326. Revisiting the Volume Hypothesis

**arXiv ID:** 2606.31282 | [PDF](https://arxiv.org/pdf/2606.31282v1)

**作者:** Ari Pakman `[一作]` (Ben-Gurion University of Negev), Yakir Berchenko `[通讯]` (Ben-Gurion University of Negev)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

使用复制交换Wang‑Landau算法在二进制权重网络上估计训练与测试准确率的联合状态密度，以探究体积假设与随机采样与梯度下降的关系；

**💡 创新点**

在不同数据规模下统一解释先前矛盾的实验结果，揭示优化偏置在小样本下占主导、架构体积效应在大样本下逐渐显现；

**🔧 技术方法**

采用Replica‑Exchange Wang‑Landau随机游走、Metropolis接受、并行随机行走以及二进制网络的Binary‑Connect训练；

**📊 数据集**

使用MNIST与Fashion‑MNIST数据集，训练样本分别为30、300、600并保持类别均衡；

**📈 对比分析**

将随机采样训练得到的最大密度对应的测试准确率与使用Binary‑Connect+Adam训练的梯度下降/随机梯度下降的测试准确率进行比较；结果显示随机采样在小样本时表现更差，但随着样本增大两者差距逐渐缩小；

**⚠️ 局限性**

仅针对二进制权重和中等规模网络，方法对大规模连续权重网络的推广有限，且REWL计算成本高，估计精度受采样次数限制。

---

## 327. Smart charging of large fleets of Electric Vehicles: Independent Multi-Agent Reinforcement Learning approaches

**arXiv ID:** 2606.31347 | [PDF](https://arxiv.org/pdf/2606.31347v1)

**作者:** Xavier Rate `[一作]` (Orange Research), Hamid Benhamed `[通讯]` (ENS Rennes)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究并比较了两类独立多智能体强化学习方法（上下文组合线性Thompson采样与PPO、A2C、SPO等策略梯度算法）在分散式电动汽车充电调度中的性能，并在同质与异质部署环境下评估其可扩展性与鲁棒性。

**💡 创新点**

创新点在于首次系统性对比基于上下文组合线性Thompson采样的组合多臂赌博机方法与经典强化学习算法，揭示它们在不同拥堵水平、价格信息精度以及多策略共存场景下的相对优势与局限；同时提出了针对竞争对手变化的线性赌博机改进思路。

**🔧 技术方法**

采用的技术包括：线性Thompson采样（LinTS）、Bernoulli TS、NeuralTS、PPO、A2C、SPO等RL/赌博机算法；基于Gymnasium的多智能体仿真平台；上下文特征包括价格、时间、周周期、状态电量、剩余充电时段等；以及用于训练和评估的仿真环境。

**📊 数据集**

使用了四年（2021‑2024）Elia光伏发电数据，按日抽样生成4000天的价格序列；数据被归一化并按1‑99百分位标准化，分为3000天训练集与1000天测试集。

**📈 对比分析**

实验方法：在同质组（20辆EV）与异质组（10 PPO + 10 LinTS）两种场景下，训练3000个一天情景、测试1000个未见情景；评估指标为平均每日奖励和SOC失败率（未完成充电）。结果显示：低拥堵下LinTS/Ts收敛快且接近最优，PPO/SPO/A2C收敛慢；高拥堵下PPO/SPO/A2C略优，但需约500天收敛；异质组中PPO在极高拥堵时优于LinTS，但整体仍受LinTS快速收敛的优势所制约。

**⚠️ 局限性**

主要局限：LinTS假设环境平稳，难以快速适应竞争对手策略变化；PPO、A2C、SPO等策略梯度方法收敛速度慢，难以在线实时学习；滑动窗口策略受季节性光伏数据周期限制，无法在短期内捕捉对手变化；缺乏针对对抗性赌博机的理论与实证验证。

---

## 328. Verification-Gated Agentic Mission-State Governance for Intelligent Industrial Multi-Robot Systems

**arXiv ID:** 2606.31339 | [PDF](https://arxiv.org/pdf/2606.31339v1)

**作者:** Guoqin Tang `[一作]` (Beijing University of Posts and Telecommunications), Gang Chen `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于验证门控的任务状态治理框架，用于工业多机器人系统的高层决策与执行管理；

**💡 创新点**

创新点在于将演化任务森林与受治理的黑板分离，并通过推导的执行耦合拓扑实现局部可验证的提议提交与受限修复；

**🔧 技术方法**

采用任务森林结构、黑板存储、执行耦合拓扑(ECT)推导、确定性验证器和原子提交机制；

**📊 数据集**

使用室内工厂模拟场景、30个种子远程施工基准（Small/Medium/Large/XL）以及相应的任务节点与资源约束；

**📈 对比分析**

与传统分配器、DART‑DAG、LiP‑LP、随机/贪心分配、平面黑板以及全局修复等方法对比，实验显示在所有基准下验证门控框架实现了零无效提交、零锁冲突、零重复分配及零中断事件，且完成率与安全审计完成率均接近100%，性能虽高于基线但仍可接受；

**⚠️ 局限性**

局限在于仅保证模型层面硬约束的完整性，未覆盖物理碰撞避免、传感器不确定性和低层执行错误，且对低层信息的保守假设依赖较强，无法保证真实工厂环境中的绝对安全与最优分配。

---

## 329. Diamond Fractures: Tracing Journal Transitions Away from Diamond Open Access

**arXiv ID:** 2606.31302 | [PDF](https://arxiv.org/pdf/2606.31302v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 330. CryoACE: An Atom-centric Framework for Accurate and Automated Model Building in Cryo-EM

**arXiv ID:** 2606.31332 | [PDF](https://arxiv.org/pdf/2606.31332v1)

**作者:** Minzhang Li `[一作]` (ShanghaiTech University), Jingyi Yu `[通讯]` (ShanghaiTech University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CryoACE框架，利用原子中心的密度采样与训练‑free能量引导，实现从同质和异质cryo‑EM密度图直接自动构建物理可行的原子结构。

**💡 创新点**

创新点在于：①原子级别的密度采样与自循环细化，省去高成本体素卷积；②训练‑free的全局与Q‑引导机制，利用本地分辨率先验解决动态模糊；③融合多模态（序列、密度、原子特征）的跨注意力网络，实现一次性生成完整结构。

**🔧 技术方法**

技术手段包括：基于Boltz‑1的多模态Transformer、原子自循环采样、Diffusion逆过程、全局与Q‑引导的训练‑free能量约束、Atom‑Transformer解码器以及轻量化的局部分辨率和Q‑score预测头。

**📊 数据集**

使用了由10,915条高质量密度‑模型‑序列三元组构成的新数据集，并在EMPIAR‑10345、EMPIAR‑10516等异质实验集以及公开的Cryo2StructData基准上进行评测。

**📈 对比分析**

与ModelAngelo、E3‑CryoFold、CryoAtom、Boltz‑1、CryoBoltz等前沿方法对比，CryoACE在同质评测中获得最高的完整度、精度（RMSD 2.1Å/BB、3.0Å/AA）、Q‑score 0.524；在异质评测中则将冲突率降至0.45%，MolProbity和加权交叉相关等指标优于竞争者，整体性能显著提升。

**⚠️ 局限性**

局限性包括：缺乏核酸数据，DNA/RNA预测仅依赖先验；GPU内存占用高，长序列或大组装难以在单卡上推理；在低于约8Å的分辨率下表现下降，且对非蛋白组分支持不足。

---

## 331. Accelerated Likelihood Maximization for Diffusion-based Versatile Content Generation

**arXiv ID:** 2606.31323 | [PDF](https://arxiv.org/pdf/2606.31323v1)

**作者:** Hyunsoo Lee `[一作]` (Seoul National University), Young Min Kim `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种无训练的采样策略ALM，通过在逆扩散过程中显式对未观察区域进行似然最大化，实现多模态内容生成。

**💡 创新点**

创新点在于将同步问题表述为对未观察区的显式似然最大化，并引入一阶加速策略，既保持生成质量又显著提升采样效率。

**🔧 技术方法**

利用扩散模型的噪声预测网络、DDIM逆采样、Score‑based 似然优化与一阶加速技术。

**📊 数据集**

实验数据集包括AFHQ、CelebA‑HQ图像集、HumanML3D动作序列、Objaverse 3D网格，以及大规模模型SDXL、FLUX、Qwen‑Image。

**📈 对比分析**

与训练型方法（BrushNet、PowerPaint、CondMDI等）和训练自由方法（SyncSDE、HD‑Painter、Recon. Guidance等）比较，ALM在图像修复、宽域生成、人类动作完成和3D纹理化等任务上均取得SOTA或相近性能，且无需额外训练。

**⚠️ 局限性**

局限性包括需手动调节加速权重、对噪声预测网络Lipschitz假设的依赖，以及在极大缺失区域或极高分辨率下仍可能出现细节失真。

---

## 332. BlockPilot: Instance-Adaptive Policy Learning for Diffusion-based Speculative Decoding

**arXiv ID:** 2606.31315 | [PDF](https://arxiv.org/pdf/2606.31315v1)

**作者:** Hao Zhang `[一作]` (AMAP, Alibaba Group), Xiangxiang Chu `[通讯]` (AMAP, Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出BlockPilot，一种在预填阶段利用预测分布预测最佳块大小的轻量级策略，实现对diffusion-based speculative decoding的样本自适应加速。

**💡 创新点**

创新点在于将块大小视为可学习的解码策略，发现其在样本间呈现局部结构，并通过一次性预测实现高效决策。

**🔧 技术方法**

采用预填分布作为特征，使用两层2048隐藏维度的MLP进行分类预测，并结合局部候选区间k=2的块大小搜索。

**📊 数据集**

训练使用ShareGPT、WSC、COPA数据集，评估覆盖Qwen3-4B/8B、Llama-3.1-8B-Instruct、Qwen3-Coder-30B-A3B，在Math（GSM8K、MATH-500、AIME24）、Code（HumanEval、MBPP、SWE-Bench）和Chat（MT-Bench）等基准上。

**📈 对比分析**

与标准自回归、EAGLE-3以及固定块大小的DFlash比较，BlockPilot在温度0/1下平均提升4.17×-4.76×速度（相对自回归），并提升接受长度至约6.6-6.9，保持输出质量不变。

**⚠️ 局限性**

局限性包括对局部区间k的依赖，若训练块大小与测试分布差距过大可能失效；目前仅验证在diffusion-based speculative decoding场景，对其他解码框架的泛化尚未证明。

---

## 333. Fundamental Limits of Quantized MIMO ISAC under Gaussian Signaling

**arXiv ID:** 2606.31301 | [PDF](https://arxiv.org/pdf/2606.31301v1)

**作者:** Hossein Atrsaei `[一作]` (Institut Polytechnique de Paris), Michèle Wigger `[通讯]` (Université Paris-Saclay)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了采用模拟空间合并和逐量化的量化MIMO ISAC系统，并推导了容量上下界及LMMSE估计误差的闭式表达。

**💡 创新点**

首次在量化MIMO ISAC框架下给出容量的解析上界和下界，并证明在低至中等信噪比下高斯信号几乎达到容量；同时提供了量化对感知LMMSE饱和行为的理论解释。

**🔧 技术方法**

使用了子量化器模型、差分dither量化的加性噪声表示、随机矩阵理论、信息理论极大熵原理、最坏噪声性质以及Kronecker相关模型。

**📊 数据集**

采用仿真数据（Rayleigh/ Jakes相关矩阵），无公开真实数据集，主要通过Monte Carlo仿真验证理论结果。

**📈 对比分析**

通过数值仿真比较容量上下界、实际容量、以及未量化基准，发现低中等SNR下上下界与实际容量重合，高SNR出现饱和；感知LMMSE在高SNR时也饱和，且空间维度缩减比对性能影响显著。

**⚠️ 局限性**

上界在高SNR时不再紧凑，且量化与空间维度缩减会导致感知误差饱和；本文未给出完整的速率-误差权衡区域，且对非高斯输入信号的容量分析仍有限。

---

## 334. Deep Reinforcement Learning for Spacecraft Attitude Control During Atmospheric Re-Entry

**arXiv ID:** 2606.31291 | [PDF](https://arxiv.org/pdf/2606.31291v1)

**作者:** Alexander Fabisch `[一作]` (German Research Center for Artificial Intelligence), Julian Theis `[通讯]` (Airbus Defence and Space GmbH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究深度强化学习在高超音速再入飞行器姿态控制中的应用，比较纯RL、PID基准和两种混合控制架构，提出MR.Q算法及动态随机化与任务调度策略；

**💡 创新点**

首次在该领域验证MR.Q算法，提出基于动态随机化的任务调度与混合控制架构，证明混合控制在泛化性上优于纯RL，并提供可检测离散化的安全边界；

**🔧 技术方法**

使用深度确定性策略梯度算法（SAC、TD3、TD7、MR.Q）、PID增益调度、动态随机化、任务调度、加法混合控制和增益调度混合控制；

**📊 数据集**

利用高保真仿真器生成的多轨迹高超音速气动数据，包含质量、惯性、气动系数等8维上下文进行随机化训练；

**📈 对比分析**

通过累计回报、误差分布、成功率等指标与基准PID进行对比；MR.Q在标称条件下平均回报最高，混合控制在泛化实验中成功率最高但控制成本略高；

**⚠️ 局限性**

纯RL与混合控制在离散化和非分布条件下的稳定性未保证，需要外部检测切断学习组件；任务调度方法未显著提升；缺乏对在线自适应与FPGA实现的完整评估。

---

## 335. Evidence Triangulation for Multimodal Fact-Checking in the Wild

**arXiv ID:** 2606.31367 | [PDF](https://arxiv.org/pdf/2606.31367v1)

**作者:** Stefanos-Iordanis Papadopoulos `[一作]` (Information Technologies Institute), Panagiotis C. Petrantonakis `[通讯]` (Aristotle University of Thessaloniki)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了X-POSE真实场景多模态数据集并开发了TRENT模型进行证据三角化与关系融合的多模态事实核查

**💡 创新点**

创新点在于（1）构建了由X社区注释且配备VLM检索与摘要的真实多模态证据增强数据集；（2）设计了三条并行跨模态注意力流与关系融合机制，显式捕捉图像-文本-证据之间的蕴含与矛盾关系；（3）相较传统一流模型，TRENT参数量少、计算成本低，且在真实数据上显著优于SOTA和商业VLM

**🔧 技术方法**

使用CLIP ViT-L/14进行特征编码，MiniCPM、Gemma3等VLM进行检索与摘要，Transformer+多头注意力实现跨模态交互，关系融合函数基于NLI构造，最终通过二元交叉熵训练

**📊 数据集**

X-POSE（5.7k多模态帖子，平衡真伪，伴随社区注释与检索到的新闻文章与摘要）以及对比实验使用NewsCLIPpings+、VERITE、商业VLM（Gemma4、Gemini3、Claude4.6等）

**📈 对比分析**

在随机与时间序列拆分上，TRENT宏F1分别达63.1%、67.4%、70.8%（h≥80%/90%）并且在过去仅证据场景下仍高于所有基线；与专用MFC模型和商业VLM相比，性能提升约5-15个百分点，且推理时间仅秒级

**⚠️ 局限性**

仍受真实数据难度限制，整体F1仅70%以下；依赖社区注释可能导致样本量与质量权衡；证据来源可信度不足、对多模态跨平台扩展有限

---

## 336. Domain-Decomposed Randomized Neural Networks for Partial Differential Equations in Unbounded Domains

**arXiv ID:** 2606.31342 | [PDF](https://arxiv.org/pdf/2606.31342v1)

**作者:** Haixin Wang `[一作]`, Shimin Guo `[通讯]` (Xi'an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出域分解的随机化神经网络框架，用不同子网络分别逼近无穷域 PDE 的近场与远场。

**💡 创新点**

创新点在于将近场与远场分别用随机子网络表示，并通过物理界面耦合，避免传统截断或全局谱方法的缺点。

**🔧 技术方法**

采用随机化神经网络 + Petrov–Galerkin 或协同插值最小二乘，配合分段 Sobolev 误差分析。

**📊 数据集**

实验基于合成解析解（泊松、Schrödinger 等）进行无穷域 PDE 计算，没有使用公开数据集。

**📈 对比分析**

与 Laguerre 谱方法、有限元等传统方法比较，给定相同自由度时误差降至 10⁻⁵，表现显著提升。

**⚠️ 局限性**

局限包括需手动选定界面位置、随机参数范围及惩罚系数，且仅在线性、可分离激活函数下得到理论保证，非线性或慢衰减解的性能未证明。

---

## 337. Linguistic Bias Mitigation for Spoofing Detection via Gradient Reversal and A Variational Information Bottleneck

**arXiv ID:** 2606.31411 | [PDF](https://arxiv.org/pdf/2606.31411v1)

**作者:** Anh-Tuan Dao `[一作]` (Avignon Universite), Nicholas Evans `[通讯]` (EURECOM)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于教师-学生对抗学习的语言不变深度伪造检测框架，利用梯度反转层和变分信息瓶颈抑制语言信息。

**💡 创新点**

创新点在于同时引入梯度反转与变分信息瓶颈，实现对语言偏差的可控抑制，从而提升跨域鲁棒性。

**🔧 技术方法**

采用XLSR预训练特征提取器、MHFA多头注意力、梯度反转层、变分信息瓶颈（VIB）以及教师模型的软标签监督。

**📊 数据集**

训练集为ASVspoof 5，评估集涵盖九个公开语音伪造数据集（ITW、ASV19、ASV21 LA/DF、FoR、CodecFake、DFADD、LibriSeVox、SONAR）。

**📈 对比分析**

与AASIST、Conformer、MHFA及其VIB改进版基线对比，平均EER从14.64%降至6.27%，跨数据集EER降低至8.72%（约36.2%相对提升）。

**⚠️ 局限性**

局限在于需教师模型提供文本标签，梯度反转与VIB的超参调优复杂，且对无文本语料的适用性尚未验证。

---

## 338. Resolving superposition in AI for interpretability and cross-modal alignment in patient-neuronal images

**arXiv ID:** 2606.31394 | [PDF](https://arxiv.org/pdf/2606.31394v1)

**作者:** Jisung Park `[一作]` (KAIST), Minee L. Choi `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

利用稀疏自编码器（SAE）消除神经网络中因维度瓶颈导致的概念叠加（superposition），恢复图像表示的几何完整性，并将去噪后的表示迁移到单细胞转录组分析框架中，随后通过Gromov–Wasserstein（GW）映射将图像表示与真实单细胞RNA测序数据对齐。

**💡 创新点**

①从理论上证明超位置会导致表示空间的几何污染；②首次在生物医学图像中用SAE恢复几何完整性；③将单细胞转录组的非线性流形分析方法直接迁移至图像域；④提出GW‑map实现无参考跨模态对齐，重建神经元病理通路。

**🔧 技术方法**

稀疏自编码器、MoCo 对比学习的CNN、Gloabl Average Pooling、Gated SAE、稀疏正则化、Diffusion Pseudotime (DPT)、PHATE、PAGA、Gromov–Wasserstein optimal transport、XGBoost/MLP 回归预测。

**📊 数据集**

约10万张多通道高通量细胞图像（Parkinson 病人诱导干细胞来源的皮层神经元，4 种荧光标记），以及公开的 SNCA×3 3D 组织体单细胞 RNA 测序数据。

**📈 对比分析**

与基线 CNN（无 SAEs）相比，SAE 后的表示在：①余弦相似度、eRank、内部点聚类一致性（Moran's I）等指标上显著提升；②使用 DPT 预测细胞死亡率的 R² 从约0.3 提升至约0.35；③GW‑map 对齐的标签转移准确率从 50%（随机）提升至 >70% 的 barycentric 精度，且基于图像的 XGBoost/MLP 可预测单细胞表达 R² 超过 0.7。

**⚠️ 局限性**

①计算量较大：训练过度完整的 SAE 和 GW‑map 需要更多 GPU 资源；②仅在卷积网络上验证，尚未在 Vision Transformer 等架构上测试；③需在更多数据模态和临床样本中验证几何污染与 SAE 恢复的普适性。

---

## 339. Energy-Optimal Spatial Iterative Learning within a Virtual Tube

**arXiv ID:** 2606.31487 | [PDF](https://arxiv.org/pdf/2606.31487v1)

**作者:** Chen Min `[一作]` (Beihang University), Quan Quan `[通讯]` (Beihang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了无模型、基于空间迭代学习的UAV能量优化框架，能够在预定义的虚拟管道内实现轨迹跟踪并显著降低能耗。

**💡 创新点**

创新点在于将迭代学习切换到空间域，通过功率-速度梯度估计实现无模型能量最优速度指令，并实现线性 O(n) 的计算复杂度，远快于传统模型基优化。

**🔧 技术方法**

采用空间迭代学习、虚拟管道约束、功率梯度估计、BLF（Barrier Lyapunov Function）等技术，实现无需先验动力学或能量模型的在线学习。

**📊 数据集**

使用了仿真中基于简化功率模型（P=50+0.05·v³）的数据以及多架平台（四旋翼、升翼多旋翼）的真实飞行实验数据。

**📈 对比分析**

与模型自由的SQP和模型基的IPOPT优化器对比；在能量消耗上接近最优（约 70% 的能耗下降），但计算时间从数小时/几十秒降至 0.6–0.7 秒，显著提升实时性。

**⚠️ 局限性**

局限性包括需多次迭代收敛、轨迹中心线跟踪被牺牲以及仅针对单一任务/路径的设计，难以直接处理多目标或更严格约束。

---

## 340. Maximizing Parallel Execution of Series-Parallel Task Graphs for Safety-Critical Embedded Control

**arXiv ID:** 2606.31481 | [PDF](https://arxiv.org/pdf/2606.31481v1)

**作者:** Jinghao Sun `[一作]` (Shandong University), Xiuzhen Cheng `[通讯]` (Shandong University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对系列并行任务图（Series-Parallel Task Graph）提出最大并行执行（Maximum Parallel Execution, MPE）问题，旨在在FPGA资源充足的情况下，通过批处理（batching）最大化任务间并行性，最小化控制周期完成时间。

**💡 创新点**

创新点包括：① 将MPE建模为加权团划分（weighted clique-partitioning）问题，并在割点（contraction）后保持有向无环性；② 提出基于Lagrangian定价的迭代启发式（LIH），通过预生成候选团、热启动定价筛选、Lagrangian定价、可行性修复与局部改进等步骤，在无需枚举指数规模列的情况下快速逼近最优；③ 在PLC梯形图到FPGA的完整流水线中验证MPE模型，并实现Verilog生成与周期仿真。

**🔧 技术方法**

使用的技术主要有：Lagrangian relaxation与定价、列生成与候选过滤、基于图论的可行性检查、局部改进搜索、随机贪心批处理、混合图着色框架（branch‑and‑bound）作为基线、以及Verilog HDL代码生成与ModelSim仿真。

**📊 数据集**

实验数据集：① 随机生成的系列并行任务图（50~100个节点、不同稠密度、前向/后向边比例、权重分布），共200实例/设定；② 六个简化的PLC梯形图内核；③ 30个PLCOpen XML完整程序，来自PLC‑LD基准集合。

**📈 对比分析**

比较方法包括：LIH、随机贪心（RG）以及基于混合图着色的精确分支定界（BB‑MGC）作为最优参考。结果显示：在可获得最优解的实例中，LIH以91.25%概率达到最优，平均相对误差仅0.073%，与RG的2.16%平均误差相比有显著提升；在大规模（50–100节点）无最优参考的情况，LIH相对RG的平均速度提升约10–16%；运行时间方面，LIH与RG均保持在毫秒级（约18 ms），而BB‑MGC平均需要数十秒至数百秒。

**⚠️ 局限性**

局限性：① 需要手工生成或自动化提取兼容性与前向关系；② 仅在FPGA资源充足的假设下工作，未考虑硬件资源冲突与布线约束；③ 仍需构造足够好的候选团池，若关键团被滤除则可能导致次优；④ 对极大图（>1000节点）或极高密度、复杂前向/后向比例的实例尚未充分评估。

---

## 341. Constrained Online Convex Optimization without Slater's Condition

**arXiv ID:** 2606.31480 | [PDF](https://arxiv.org/pdf/2606.31480v1)

**作者:** Kihyun Yu `[一作]` (KAIST), Dabeen Lee `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种统一的 anytime 原子-对偶算法，用以解决带随机约束的在线凸优化问题，无需 Slater 条件。

**💡 创新点**

创新点在于引入自适应正则化项 R_t，稳定对偶过程并抵消在期望可行比较器下出现的无负漂移问题；该正则化可在不依赖 Slater 或 Lagrange multiplier 边界的情况下实现几乎最优（O(√T)）累计误差与约束违规量，并且可推广至强凸损失和对抗性约束。

**🔧 技术方法**

核心技术包括：指数 Lyapunov 函数 Φ_t(x)=e^{γ_t}x-1、AdaGrad 型步长 η_t、以及通过 R_t 对 Lyapunov 漂移的控制；利用超马尔可夫集中不等式实现高概率界；对强凸情形采用 μ‑strongly convex 的解析技巧。

**📊 数据集**

本文为理论论文，没有在实际数据集上进行实验。

**📈 对比分析**

与现有最优 COCO 算法（在随机约束下需要 Slater 条件或 bounded Lagrange multipliers，或在对抗性约束下使用严格可行比较器）相比，该方法在相同设置下实现了 O(√T) 的期望/高概率误差与约束违规量，且不需要任何先验 Slater 常数或 Lagrange multiplier 上界；在强凸损失下进一步降到 O(log T)。

**⚠️ 局限性**

限制：算法仍假设约束函数 i.i.d.（随机约束）且可测；在对抗性约束下需要使用更强的可行比较器；目前仅给出理论收敛率，缺乏实验验证；对动态比较器（动态 regret）的分析尚未完成。

---

## 342. One Reflection Is Not Enough: Self-Correcting Autonomous Research via Multi-Hypothesis Failure Attribution

**arXiv ID:** 2606.31478 | [PDF](https://arxiv.org/pdf/2606.31478v1)

**作者:** Jie Ma `[一作]` (Xiamen University), Rongrong Ji `[通讯]` (Xiamen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出SAGE系统，通过多假设故障归因与层级化路由，实现自主科研代理在实验失败时的结构化诊断与恢复，避免单一反思导致的盲目试错；

**💡 创新点**

创新点在于将故障归因拆分为发散生成、独立评估和确定性层级路由三阶段，并配合零指标修复路径和双层数值报告机制，使代理能够精准定位故障层级并有针对性地修复；

**🔧 技术方法**

技术包括多假设故障归因（MHFA）——发散生成与独立严谨评估；TrajPivot轨迹特征提取；两阶段数值报告（主动清单与被动清洗）；以及统一评估框架AR‑Eval和主会议式评审；

**📊 数据集**

使用ARC‑Bench 12题子集，覆盖机器学习、统计学、量子计算、生物学与高能物理共五个领域；

**📈 对比分析**

在同一题集下与主流水线反思基线与AI‑Scientist‑v2对比，SAGE在指标型恢复率上提升至11/12（vs 5/12），整体artifact评分52.0/100（vs 48.2/100/24.8），代码实现与执行得分最高，AR‑Eval平均6.5/10（vs 5.0/10），人工评估也表明SAGE优于其他系统；

**⚠️ 局限性**

局限在于结果分析与方法来源的真实性仍不足，主要会议式评审仍低于接受标准；数值报告虽更可信，但文本与表格的一致性问题仍存在；完整方法实施与实验细节的可追溯性仍未完全解决。

---

## 343. Antenna Orientation Optimization for Rotatable Antenna-Enabled ISAC Systems

**arXiv ID:** 2606.31466 | [PDF](https://arxiv.org/pdf/2606.31466v1)

**作者:** Qingjie Wu `[一作]` (South China University of Technology), Robert Schober `[通讯]` (Friedrich-Alexander-University Erlangen-Nürnberg)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并分析了一种基于可旋转天线（RA）的集成感知与通信（ISAC）系统，设计了针对多用户和区域感知的RA指向向量、发射波束与探测信号协同优化算法，目标是最大化感知区域内最小回波功率并满足通信速率要求。

**💡 创新点**

创新点在于：①实现了天线指向的元素级别自适应调控，显著提升了空间自由度；②在单用户点目标情形下给出了RA指向向量的闭式最优解；③在多用户扩展目标情形下提出了基于交替优化（AO）、半正定松弛（SDR）和序列凸近似（SCA）的有效求解框架。

**🔧 技术方法**

采用的技术主要包括：可旋转天线硬件架构、MIMO多用户波束成形、最大比传输（MRT）、半正定规划（SDP）与SDR、SCA与AO求解策略，以及CVX求解器进行数值优化。

**📊 数据集**

使用的“数据集”为仿真数据，系统在2.4 GHz、4×4 UPA阵列、λ/2间距、指定用户和目标位置以及不同的旋转角度、指向增益参数ρ和感知区域尺寸r等参数下进行数值模拟，无真实场景数据集。

**📈 对比分析**

通过与统一指向、随机指向、固定指向以及等向性天线等基准方案比较，实验显示RA‑启用ISAC系统在最小回波功率和通信速率权衡方面均显著优于基准，且在不同旋转角度、指向增益和感知区域尺寸变化时展现出可观的性能提升。

**⚠️ 局限性**

主要局限包括：①仅考虑LoS、单载波、远场模型，未涵盖多径或宽带效应；②仿真验证缺乏实际硬件实验支持；③旋转角度受限时指向优化仍有限，且求解复杂度随用户数和采样点数增长显著。

---

## 344. UniTac: A Unified Multimodal Model for Cross-Sensor Tactile Understanding and Generation

**arXiv ID:** 2606.31451 | [PDF](https://arxiv.org/pdf/2606.31451v1)

**作者:** Jiahang Tu `[一作]` (Zhejiang University), Alex Wong `[通讯]` (Yale University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 UniTac，一种统一的多模态模型，能够在多种触觉传感器上同时完成触觉理解（如物体属性描述、传感器识别）和触觉生成（从文字或视觉条件生成高质量的触觉图像/视频）。

**💡 创新点**

创新点包括：
- 双层混合理解（Dual‑Level Mixture Comprehension），在对象属性描述和传感器识别两任务上联合监督，使模型能够区分物体与传感器的物理特征；
- 两阶段对齐生成（Two‑Stage Aligned Generation），先在自监督阶段学习触觉先验，再通过 Sensor‑Aware Diffusion Transformer 对齐 MLLM 的语义向量与触觉潜在空间，保证生成结果既语义一致又传感器一致；
- 传感器先验采样策略（Sensor‑Prior Sampling Strategy），在从无接触到接触的采样过程中加入传感器先验，模拟真实的触觉采集流程，提升生成的物理真实性；
- 通过整合五大公开触觉数据集构建超过 400k 条视频、1.6M 帧的大规模跨传感器语料库，实现跨传感器的联合训练。

**🔧 技术方法**

技术手段包括：
- 触觉编码器（AnyTouch 预训练模型）提取静态与动态触觉特征；
- 多模态大语言模型（Qwen‑VL 2.5）作为共享的理解与生成 backbone；
- Sensor‑Aware Diffusion Transformer（DiT）投影器对齐 MLLM 输出与触觉潜在向量；
- SANA 触觉解码器（及可替换的 Wan v2.2 用于视频生成）；
- 两阶段训练：重建阶段（Decoder + Encoder 直接对齐）和对齐阶段（DiT 训练 rectified flow 损失）；
- 传感器先验条件下的 CFG 采样实现非接触到接触的过渡。

**📊 数据集**

数据集：
- Touch and Go、Tacquad、TVL、SSVTP、PHYSICLEAR（由 AnyTouch 统一整理）
- 总计约 400k 条触觉视频，1.6M 帧；
- 采用与文本描述对齐的标注（如硬度、粗糙度、纹理），并包含多种传感器（Digit、GelSight、GelSight Mini、Duragel 等）。

**📈 对比分析**

性能对比：
- 在 PHYSICLEAR‑Test 触觉理解基准上，UniTac‑7B 的平均分 66.51，明显超过前沿 UMM（Octopi‑7B 60.61）和单一模态模型；
- 触觉生成方面，UniTac 在四种传感器上的平均 SSIM 0.915、PSNR 21.26，领先现有生成模型（如 GVST、TextToucher）和其他 UMMs；
- 机器人真实交互实验中，使用 UniTac 生成的触觉数据能将跨传感器抓取准确率从 50% 提升至 99%，且实时抓取成功率 95%。

**⚠️ 局限性**

局限性：
- 仍依赖大规模、已标注的触觉语料库，跨新传感器的适配需要重新收集或重训练；
- 对于数据质量差、采集不稳定的传感器（如 Duragel），生成效果仍不理想；
- 训练与推理算力要求高，尤其是两阶段 Diffusion 训练；
- 仅涵盖视觉‑触觉双模，缺乏对其他模态（音频、力觉等）的融合；
- 现有方法主要评估在有限的基准任务，尚未覆盖更广泛的实际工业或医疗应用场景。

---

## 345. Who Determines the Meaning of an Emotion? Affective Sovereignty as an Epistemic Consequence of Measurement Limits

**arXiv ID:** 2606.31442 | [PDF](https://arxiv.org/pdf/2606.31442v1)

**作者:** Keito Inoshita `[一作]` `[通讯]` (Kansai University), Keito Inoshita (Kansai University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文从测量结构限制的视角，构建情绪意义分布概念，并提出‘情绪主权’作为情绪解释最终权威的规范性结论。

**💡 创新点**

创新点在于将情绪标注的不确定性分解为可约化与不可约化两部分，并基于不可约化成分的不可恢复性推出情绪解释应由主体拥有的‘情绪主权’原则。

**🔧 技术方法**

使用了统计推导（如Jensen不等式）、不确定性分解（熵、Gini指数）以及哲学与法律框架的理论论证。

**📊 数据集**

本研究未收集新数据，主要引用已有情绪标注数据集与相关文献中的实验结果作为佐证。

**📈 对比分析**

由于本文为理论分析，未进行实验对比；通过对先行实证结果的引用说明了置信度与不可约化成分之间的差距。

**⚠️ 局限性**

主要局限包括：假设标注协议固定，归属主权的规范前提待进一步论证，且缺乏对不同协议下不可约化成分的经验量化。

---

## 346. CDR-Bench: Evaluating Faithful Execution of Compositional, Order-Sensitive Data Refinement Recipes

**arXiv ID:** 2606.31435 | [PDF](https://arxiv.org/pdf/2606.31435v1)

**作者:** Yuchen Huang `[一作]` (HKUST), Yaliang Li `[通讯]` (Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为 CDR‑Bench 的基准，用于评估大型语言模型在执行多步骤、顺序敏感的数据精炼配方的能力。

**💡 创新点**

提出了专门针对组合式、顺序敏感的数据精炼任务的评估框架，并给出了精准的文本对比指标（Recipe Success、Order‑Consistent Success、Refinement Gain），同时收集真实域数据并设计多种指令风格与提示策略。

**🔧 技术方法**

采用了基于规则的 Data‑Juicer 预定义操作（映射器与过滤器）进行确定性执行，使用自然语言指令生成与多样化提示，并在多种 LLM 上进行推理和指标评估。

**📊 数据集**

从 Common Crawl、arXiv、Wikipedia、GovReport、PII 数据等四个领域收集原始文本，经过操作器激活和配方挖掘，生成 3,462 个任务。

**📈 对比分析**

在 10+ 先进 LLM（如 GPT、Claude、Gemma、Qwen 等）上对 Atomic、Agnostic、Order‑M、Order‑F 等三种轨迹进行评估；结果显示从单步到多步的性能急剧下降，尤其在顺序敏感场景下准确率低于 5%/19%，提示与思考模式虽有提升但仍远低于期望。

**⚠️ 局限性**

局限于确定性、文本级操作，未覆盖主观精炼、工具调用、多模态或多语言场景；指令生成依赖 LLM 可能不符合真实用户表述；未涉及交互式或代理式精炼流程。

---

## 347. Temporal Preservation over Processing: Diagnosing and Designing Spatiotemporal Single-Stage Video Detectors

**arXiv ID:** 2606.31421 | [PDF](https://arxiv.org/pdf/2606.31421v1)

**作者:** Karam Tomotaki-Dawoud `[一作]` (Fraunhofer HHI), Sebastian Bosse `[通讯]` (Fraunhofer HHI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出了一种基于YOLOv8的轻量级时空视频目标检测框架YOLO-3D，并提出了模型无关的诊断框架TemporalLens，用以评估检测器是否真正利用时间上下文；

**💡 创新点**

创新点在于（1）通过保持骨干网络的时间维度不被压缩，显著提升时空特征利用；（2）设计了可学习的时空注意融合模块STAF和自适应时间聚焦ATF；（3）提出一套七种结构化扰动（如遮挡、打乱、降分辨率等）来定量检测模型对时间信息的依赖；

**🔧 技术方法**

使用了3D卷积骨干（C2f3d、SPPF3d）、FPN/PAN式融合、动态拼接、线性注意力、以及自适应时间聚焦等技术；

**📊 数据集**

数据集包括公开的UCF101-24、医疗手术场景（肾移植）和农业牧场的奶牛姿态数据；

**📈 对比分析**

通过与堆叠-2D基线比较，在UCF101-24上实验了10种配置，发现保持时间维度的骨干可提升约+3.7pp mAP@50；在手术和奶牛数据上，TemporalLens揭示3D模型在隐藏最后帧、序列打乱等扰动下仍保持高精度，而2D堆叠模型则崩溃；

**⚠️ 局限性**

限制在于3D模型计算成本较高，尤其在大规模和长序列时；训练预算受限，未能充分挖掘高级注意和聚焦模块的潜力；此外诊断框架目前不适用于多输入的双分支架构。

---

## 348. BP-TTA: Balanced and Prototype-Guided Test-Time Adaptation in Dynamic Scenarios

**arXiv ID:** 2606.31420 | [PDF](https://arxiv.org/pdf/2606.31420v1)

**作者:** Shaoyang Huang `[一作]` (Sichuan University), Tao He `[通讯]` (Sichuan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了BP‑TTA（Balanced and Prototype‑Guided Test‑Time Adaptation）方法，用于处理动态场景下的持续域漂移与类别不平衡问题；

**💡 创新点**

创新点在于：① Batch‑Balanced Sampling (BBS) 通过在实时样本与高置信历史样本之间构造类别平衡的适配批次；② Category Prototype‑Guided Adaptation (CPGA) 通过动态维护类别原型并在特征空间中施加相似性约束来提升伪标签质量与在线更新稳定性；

**🔧 技术方法**

技术手段包括：基于Teacher‑Student框架的EMA更新、伪标签自监督一致性损失、类原型更新与原型对齐损失、记忆池与按类别平衡采样、基于置信度与年龄的样本筛选；

**📊 数据集**

实验数据集涵盖CIFAR10‑C、CIFAR100‑C、ImageNet‑C（多种噪声与失真）以及DomainNet‑126的四个域，使用WideResNet‑28、ResNeXt‑29、ResNet‑50等多种模型；

**📈 对比分析**

与TENT、CoTTA、EATA、SAR、PALM、SURGEON、LAME、RoTTA、DA‑TTA等22种基线对比，BP‑TTA在所有数据集和模型上均实现了最低的平均分类错误率，提升幅度约1.5%–3.1%；

**⚠️ 局限性**

局限性包括：对极端类别极度不平衡的流仍可能出现原型更新偏差；需维护和更新显存池，计算和存储开销相对较高；在频繁域变化极快时，BBS与CPGA仍可能出现滞后。

---

## 349. Performance Analysis in Parallel Programming Education: A Comparative Usability Study

**arXiv ID:** 2606.31458 | [PDF](https://arxiv.org/pdf/2606.31458v1)

**作者:** Anna-Lena Roth `[一作]` (Fulda University of Applied Sciences), Michael Kuhn `[通讯]` (Otto von Guericke University Magdeburg)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了EduMPI这一面向教育的并行性能分析工具，并通过对比实验评估其在学生性能分析能力上的影响

**💡 创新点**

提供近实时MPI通信可视化，简化了高性能计算环境的使用与性能问题识别，填补了专业工具在教育场景中交互与易用性上的空白

**🔧 技术方法**

采用GUI、Open MPI的定制分支、Score‑P自动采集、时间序列数据库等技术实现自动化执行与可视化

**📊 数据集**

以并行行矩阵乘法程序为实验对象（master‑worker模式）

**📈 对比分析**

通过实验室可用性研究，将EduMPI与专业工具TAU、CUBE进行对照，测量任务正确率、干预次数、完成时间等指标；结果显示EduMPI在任务正确率（96% vs 60%）、干预需求（29% vs 55%）和部分任务耗时上均优于TAU/CUBE

**⚠️ 局限性**

对整体运行时聚合视图支持不足，无法直观呈现全时段的性能指标，需要进一步完善

---

## 350. Fork-Think with Confidence

**arXiv ID:** 2606.31484 | [PDF](https://arxiv.org/pdf/2606.31484v1)

**作者:** Zena Al-Khalili `[一作]` (Saarland University), Ji-Ung Lee `[通讯]` (Saarland University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于“先决策后思考”的并行推理框架——Fork‑Think with Confidence，先用单一路径的模型置信度识别关键分叉点，再在这些点进行多路径采样并聚合回答；

**💡 创新点**

创新点在于颠覆传统的“先思考后决策”模式，利用模型置信度自动定位分叉点，从而显著减少过度生成和剪枝需求；

**🔧 技术方法**

主要技术包括使用基于前k个token平均对数概率的模型置信度评估、确定分叉点、在分叉点后使用温度>0的随机采样以及多数投票聚合；

**📊 数据集**

实验使用了三大推理基准：AIME（美国数学竞赛题目）2024/2025年数据集、GPQA（研究生级科学问答）和不同规模的LLM模型（Qwen3‑8B、DeepSeek‑8B、Phi‑4‑RP‑14B）；

**📈 对比分析**

与Greedy、CoT、Parallel Thinking、ASC、DeepConf等基线对比，Fork‑Think在保持相近甚至略优准确率的同时，平均减少30%–57%令牌消耗和38%–57%运行时间；

**⚠️ 局限性**

局限包括对单个分叉点的依赖（多分叉可能更好）、对种子路径长度的敏感性、置信度估计方法的经验性以及在低预算、低分叉率下的性能波动。

---

## 351. AeroVerse-SatAgent: UAV-Satellite Collaborative Spatial Reasoning Inspired by the Dual Visual Pathway Theory of Cognitive Neuroscience

**arXiv ID:** 2606.31467 | [PDF](https://arxiv.org/pdf/2606.31467v1)

**作者:** Wenyi Zhang `[一作]` (Aerospace Information Research Institute, Chinese Academy of Sciences), Kun Fu `[通讯]` (Aerospace Information Research Institute, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SatAgent，一种基于UAV-卫星协同的多视角空间推理模型，解决了单视角视角畸变和几何建模不足的问题；同时构建了跨视角不可解的SatAgent‑SR130K大规模视觉问答数据集；

**💡 创新点**

创新点包括：①受人类双视觉通路启发的双通道协同编码器，分别处理语义与几何信息并通过双向门控实现功能分离；②基于协方差感知的3D高斯软投影将UAV图像映射到BEV坐标系，实现几何精确对齐；③动态k‑NN图与图注意力实现跨视角拓扑语义对齐；④多视角一致性损失与功能专化损失显式引导两通路分工；

**🔧 技术方法**

技术手段主要有CLIP ViT‑B/16视觉编码器、UniDepth深度估计、EWA软投影、STN仿射校正、动态k‑NN + GAT图模块、双向门控融合、LoRA参数高效微调以及4‑bit NF4量化的Qwen3.5‑9B LLM；

**📊 数据集**

使用的数据集为SatAgent‑SR130K（基于GeoText‑1652的130K问答对，涵盖八类空间推理任务），并与其他单视角与多视角基准（VSI‑Bench、ViewSpatial‑Bench、MMSI‑Bench、LinkS²Bench等）进行对比；

**📈 对比分析**

采用零样本和LoRA微调对齐的多模型对比，指标包括Token F1、ROUGE‑L、METEOR、BERTScore‑F1；SatAgent在Token F1上达到65.44%，比最强基线SpatialRGPT高11.69%，比MiniMax‑01高25.91%； ablation实验验证各模块贡献；

**⚠️ 局限性**

局限性包括：①视觉令牌压缩导致高密度场景实例分辨率下降；②卫星与UAV采集时间差异导致跨视角语义冲突；③在极端视角或复杂结构下仍可能出现几何误差；未来需引入自适应令牌分配与时间对齐机制。

---

## 352. A forgery attack on the Block.co blockchain-based digital credential certification system

**arXiv ID:** 2606.31462 | [PDF](https://arxiv.org/pdf/2606.31462v1)

**作者:** Giacomo Zonneveld `[一作]` (Università Politecnica delle Marche), Marco Baldi `[通讯]` (Università Politecnica delle Marche)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

针对Block.co数字证书系统，设计并实现了一种基于伪造Bitcoin地址和URL的认证欺骗攻击

**💡 创新点**

首次证明混合区块链认证体系在缺乏可信锚点时易被伪造，揭示去中心化验证的脆弱性

**🔧 技术方法**

利用Merkle树、Chainpoint v2.0、Bitcoin OP_RETURN、PDF元数据注入与解析技术

**📊 数据集**

使用自制的伪造PDF证书和公开GitHub托管的Bitcoin地址，无需真实数据集

**📈 对比分析**

通过在Block.co公共验证器上上传伪造证书，验证结果显示证书完整性和发行人身份均被错误接受，性能未出现异常

**⚠️ 局限性**

攻击仅在发行人身份验证使用外部URL且未启用中心化校验时可行，且依赖对Block.co验证流程的完整了解

---

## 353. No Prompt, No Leaks: A Robust Generative Steganography Framework via Prompt-Free Diffusion

**arXiv ID:** 2606.31427 | [PDF](https://arxiv.org/pdf/2606.31427v1)

**作者:** Jingwen Cai `[一作]` (Xiangtan University), Xieping Gao `[通讯]` (Hunan Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无提示（prompt-free）的生成式图像隐写框架，利用风格语义先验在潜空间中实现安全且可控的隐写与恢复。

**💡 创新点**

创新点在于：①去掉了对文本提示的依赖，用ViT提取的语义先验代替；②设计了级联仿射耦合模块（CACM）实现秘密图像到潜向量的确定映射；③引入轨迹校正模块（TCM），通过单步反馈纠正反向过程，提高在传输失真下的恢复稳定性。

**🔧 技术方法**

核心技术包括：潜在扩散模型（Latent Diffusion Model）+ DPMSolver++采样；ViT提取语义先验；CACM（级联仿射耦合）；SADM（语义感知扩散模块）；TCM（轨迹校正）。

**📊 数据集**

使用FFHQ数据集作为秘密图像；参考数据集（如LSUN Bedroom）用于生成语义先验；实验在5000/1000训练/测试样本上进行。

**📈 对比分析**

与多种SOTA方法（TS与GS）对比，评价指标包括NIQE、FID（stego‑real、stego‑secret）、PSNR、SSIM、StegExpose/XuNet/KeNet检测误差。结果显示：视觉质量最佳（NIQE 3.15，FID_ste,real 9.25），内容泄漏最小（FID_ste,sec 342.5），检测误差接近0.5，且在各种失真（高斯噪声、JPEG压缩等）下恢复PSNR/SSIM始终保持在最高水平。

**⚠️ 局限性**

局限性：容量固定为24 bpp，模型训练与推理仍需高算力；对极端噪声或压缩级别的鲁棒性尚未彻底验证；缺乏对大规模多样化参考数据集的适应性评估。

---

## 354. Ensuring Deterministic Timing in a Federated GNSS Correction Pipeline with Lingua Franca

**arXiv ID:** 2606.31415 | [PDF](https://arxiv.org/pdf/2606.31415v1)

**作者:** Tejeswini Jayaramareddy `[一作]` (Santa Clara University), Hokeun Kim `[通讯]` (Arizona State University)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过 Lingua Franca 建模并验证一个分布式的 GNSS 校正管线，解析其在 UART 中断、FIFO 缓冲和周期性转发等组件下的确定性时间行为。

**💡 创新点**

将逻辑时间语义与分布式执行结合，证明即使存在异步中断和分布式组件，整个管线的时序也可闭式推导并保持确定性；同时通过 federated LF 运行时验证理论结果。

**🔧 技术方法**

Lingua Franca（LF）语言、Reactor 模型、逻辑时间语义、Federated execution、物理动作实现 UART 中断、周期性定时器，以及实验平台 Linux+Intel i7-9700K。

**📊 数据集**

实验使用自生成的 317 个 GNSS epoch，payload 大小随机采样于 [400,800] 字节的区间；未使用公开数据集。

**📈 对比分析**

通过理论推导得到最大延迟 35.4 ms、交叉到达间隔 {18,90,108} ms，并与实验测量（最大 37.1 ms，分布 19/41/40%）对比，误差低于 2 ms，缓冲占用稳定无溢出，验证了模型的可预测性。

**⚠️ 局限性**

实验仅在单一硬件平台和固定参数（T_gnss=100 ms, T_read=18 ms, 460 800 baud）下验证，缺乏对不同硬件、网络延迟或更大/不同周期的泛化；物理时间噪声虽小但仍为平台依赖，模型对多源多机或更复杂拓扑的适用性未作评估。

---

## 355. Learning to Select, Not Relearn: Hard-Routed Mixtures of Reasoning LoRAs

**arXiv ID:** 2606.31413 | [PDF](https://arxiv.org/pdf/2606.31413v1)

**作者:** Seyed Alireza Molavi `[一作]` (Halmstad University), Prayag Tiwari `[通讯]` (Halmstad University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段框架Hard‑Routed MoR‑LoRA，用强化学习可验证反馈训练独立的LoRA专家，然后冻结专家，通过稀疏硬路由和轻量注意力LoRA实现多领域模型集成。

**💡 创新点**

解决了软MoE路由导致LoRA单元尺度失配的问题，使用硬top‑1路由保持每个专家的原始单元尺度；通过RLVF训练获得更强的推理专家；采用两阶段训练把集成视为路由学习，显著降低可训练参数。

**🔧 技术方法**

LoRA、Mixture‑of‑Experts、straight‑through estimator、强化学习可验证反馈（GRPO）、推理轨迹蒸馏、轻量级共享路由器和注意力LoRA。

**📊 数据集**

GSM8K、ARC‑Challenge、Medical Question Answering、BoolQ、Corpus of Linguistic Acceptability；此外在混合域评测中使用GSM8K+BoolQ；实验还在附录中扩展到其他模型族。

**📈 对比分析**

与独立SFT/ RLVF专家、LoRAHub、LoRAMixer（TopK=1/2）、MoLE、Prompt‑level分类等基线对比；Hard‑Routed在保持专家冻结的前提下，在5个任务上平均性能与软路由相当或更好，参数量仅为软路由的1/10左右；RLVF在大模型上显著提升推理性能。

**⚠️ 局限性**

集成仍需一定的标注数据来学习路由；无法实现零样本集成；适用于域数有限的场景，混合域推理需要额外训练；当前缺乏失效时退回原模型的机制。

---

## 356. Mixture-of-Control: State-Aware Fine-Tuning for Transformer-based Models

**arXiv ID:** 2606.31397 | [PDF](https://arxiv.org/pdf/2606.31397v1)

**作者:** Duc Anh Nguyen `[一作]` (Qualcomm AI Research), Toan Tran `[通讯]` (Qualcomm AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于控制理论的轻量级微调框架Mixture-of-Control（MoC），通过稀疏专家路由将局部控制与全局控制融合，实现跨层信息交流；

**💡 创新点**

创新点在于将块级控制状态视为专家并采用共享门控的稀疏Mixture-of-Experts机制，既保持了状态基微调的内存低开销，又实现了高效的跨层通信；

**🔧 技术方法**

采用低秩控制模块、共享门控、Top‑K稀疏路由、控制理论分析以及梯度稳定性与表示误差理论；

**📊 数据集**

在NLG的Commonsense推理数据集（BoolQ、PiQA、Social IQa、HellaSwag、WinoGrande、ARC-E/ARC-C、OpenBookQA）和NLP的GLUE基准上进行评估；

**📈 对比分析**

与LoRA、DoRA、MoLEx、Parallel Control等主流PEFT方法对比，MoC在大多数任务上平均提升1–3%准确率，且训练内存与推理吞吐量均低于MoLEx且接近或优于LoRA；

**⚠️ 局限性**

局限性包括对超参数（如Top‑K、α、负载平衡系数λ）的敏感性以及在极大规模模型中进一步压缩计算量和内存的挑战。

---

## 357. EnclaveX: End-to-End Confidential AI with CPU/GPU TEEs

**arXiv ID:** 2606.31408 | [PDF](https://arxiv.org/pdf/2606.31408v1)

**作者:** Robert Schambach `[一作]` (Technische Universitaet Dresden), Christof Fetzer `[通讯]` (Technische Universitaet Dresden)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套端到端的Confidential AI/ML 工作流，整合 CPU TEEs（Intel TDX、AMD SEV‑SNP）与 GPU TEE（NVIDIA H200 cGPU），通过 SCONE 运行时、CAS（配置与证明服务）和自定义内核模块实现对 K8s 管理员的隔离，保证数据与模型在使用过程中的机密性与完整性。

**💡 创新点**

1）首次在 CVM 内实现 process‑based TEE 并扩展到 cGPU，形成 CPU‑GPU 跨界的全流程安全边界；2）通过 CAS 在成功证明后才释放密钥，切断管理员与敏感数据的直接接触；3）在不修改 AI/ML 代码的前提下，利用 SCONE 的软‑封装实现零性能开销。

**🔧 技术方法**

硬件：Intel TDX、AMD SEV‑SNP、NVIDIA Hopper H200 cGPU；软件：SCONE 运行时、自定义内核模块、CAS（使用 EdDSA）、SPDM、NVIDIA Triton 推理服务器、TensorRT‑LLM。

**📊 数据集**

使用 NVIDIA‑优化的 LLM 进行推理基准，输入/输出 token 大小固定为 128，批量大小从 1–64 变化；还测量了不同输入 token 长度（32–1024）下的首 token 延迟。

**📈 对比分析**

与 native CVM（非 CC）和仅使用 CVM 的基线进行对比：SCONE 在 CVM 上无性能损失；cGPU CC 模式相对 native 有 35–63% 的吞吐量/延迟开销，随着 batch/输入大小增大而下降。attestation 开销约 1.27%，几乎可忽略。

**⚠️ 局限性**

1）未覆盖侧信道攻击、DoS 等攻击；2）对小 batch/小 token 的 IO 负载仍存在较大开销；3）目前仅支持 Intel TDX、NVIDIA H200，尚未支持 AMD SEV‑IO 或 Intel TDX‑Connect；4）缺乏多租户或联邦学习场景的实验与评估。

---

## 358. World-Model Collapse as a Phase Transition

**arXiv ID:** 2606.31399 | [PDF](https://arxiv.org/pdf/2606.31399v1)

**作者:** Xinyuan Song `[一作]` (Emory University), Zekun Cai `[通讯]` (University of Tokyo)

**通讯引用:** 399 | [OpenAlex ID](https://openalex.org/A5108659841)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究长周期语言代理在世界模型中的崩塌现象，并通过控制实验揭示其相当于物理相变的临界点

**💡 创新点**

首次将相变概念应用于多步骤语言代理的世界模型，提出世界状态先于计划崩塌的时序机制和可测量的临界边界

**🔧 技术方法**

采用结构化三调用循环（Planner‑Updater‑Self‑Diag），对世界状态进行逐步真值跟踪，使用精确的确定性模拟器（StatefulPuzzle）构建可控测试环境

**📊 数据集**

使用自定义的确定性任务环境——StatefulPuzzle；另外在GraphNav、ToolDAG等环境中做触发器选择，确保控制轴（状态维度、依赖密度）可分离

**📈 对比分析**

在多模型（Claude‑Haiku‑4‑5、GPT‑4o‑mini、GPT‑4o、Llama‑3 70B）上做跨模型网格实验，结果显示相同的相变形状但临界点随模型能力平移，表明模型能力影响边界而非消除相变；在单轴消融中确认时空、分支、观测噪声、突变率的不同作用；性能表现为：在状态维度10时成功率下降到约50%，而状态维度20即全失效

**⚠️ 局限性**

局限性包括：仅在有限的确定性任务（StatefulPuzzle）验证，未在自然生态基准上复现；相变分析基于离散网格，未估计临界指数；跨模型比较受接口、终端确定性差异影响；实验仅针对固定的三调用架构，未探究其他记忆或规划结构的影响

---

## 359. ReGRPO: Reflection-Augmented Policy Optimization for Tool-Using Agents

**arXiv ID:** 2606.31392 | [PDF](https://arxiv.org/pdf/2606.31392v1)

**作者:** Binjie Zhang `[一作]` (National University of Singapore), Mike Zheng Shou `[通讯]` (National University of Singapore)

**通讯引用:** 29633 | [OpenAlex ID](https://openalex.org/A5035112538)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ReGRPO框架，使视觉语言模型在使用外部工具时通过反射步骤实现自我纠错。

**💡 创新点**

创新点在于构建结构化反射数据引擎（ErrorType, Evidence, FixPlan）并在GRPO中联合优化反射与动作，且采用零验证器单路径部署。

**🔧 技术方法**

采用结构化反射（RoT）数据、GRPO+反射优化、LoRA微调、GPT-4o生成反射以及基于元数据的确定性验证器奖励。

**📊 数据集**

使用GTA和GAIA两大多模工具使用基准，并以MAT-AGENT工具集为环境。

**📈 对比分析**

与闭源与开源基线对比，ReGRPO在同一backbone（Qwen2-VL-7B）下在GTA/GAIA上取得最高答案准确率与工具正确率，优于SPORT等对手。

**⚠️ 局限性**

局限性包括对教师生成反射的依赖、对极端分布偏移时的鲁棒性不足，以及在极端异常场景下仍需进一步验证。

---

## 360. Revisiting Parameter Redundancy in Vision-Language-Action Models: Insights from VLM-to-VLA Adaptation

**arXiv ID:** 2606.31382 | [PDF](https://arxiv.org/pdf/2606.31382v1)

**作者:** Fengnian Zhang `[一作]` (Chinese Academy of Sciences), Chang Xu `[通讯]` (University of Sydney)

**通讯引用:** 22397 | [OpenAlex ID](https://openalex.org/A5001529504)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过对 VLM 到 VLA 的适配过程进行参数差异分析，提出一种无需后置微调的分模块联合剪枝方法，显著压缩模型体积并保持约 90% 的原始性能

**💡 创新点**

首次将 VLM–VLA 参数差异 ΔW 作为可解释的冗余度量，揭示模块间的结构异质性，并基于此设计了跨模块差异化剪枝策略

**🔧 技术方法**

采用参数差异量化、模块感知掩码构造、受控剪枝实验和无恢复性能评估等技术

**📊 数据集**

使用 LIBERO 机器人仿真基准（包含空间、对象、目标及长时序四子数据集）进行评测

**📈 对比分析**

与 LLM‑Pruner、FLAP、Wanda 等传统剪枝方法在相同稀疏率下对比，实验表明在不进行微调的情况下，本方法可将 OpenVLA 与 π_0.5 的参数量分别压缩 12%–30%，平均性能保持在 62%–90% 范围内，而传统方法则几乎崩溃

**⚠️ 局限性**

该方法仍需针对项目器等关键子模块保持保守剪枝，对不同任务的鲁棒性和极端稀疏化性能仍有提升空间

---

## 361. CLOUDADV: Decision-Aligned Instance Sizing with Zero-Shot Foundation Models under Drift

**arXiv ID:** 2606.31470 | [PDF](https://arxiv.org/pdf/2606.31470v1)

**作者:** Jack Bell `[一作]` (University of Pisa), Vincenzo Lomonaco `[通讯]` (Luiss University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并评估了CLOUDADV系统，结合零射时间序列预报与LLM生成的决策对齐建议，用于云实例容量规划并应对工作负载漂移。

**💡 创新点**

创新点在于：①将零射预报与受限决策上下文相结合，实现对齐的推荐；②构建离线高容量LLM与在线低容量LLM的对照管道，验证部署时的行为一致性；③通过自然语言交互与结构化验证提升工程师可操作性。

**🔧 技术方法**

使用技术包括零射时间序列模型Chronos‑2与TimesFM 2.5、LLM Claude Opus 4.6（参考）与Qwen3.5‑35B（生产），基于CPU/内存阈值的容量约束策略，以及自定义的验证与回溯超额率评估框架。

**📊 数据集**

数据集为七台真实生产VM的1分钟CPU百分比与可用内存记录，经过去重、线性插补、归一化后划分为60%训练、20%验证、20%测试，涵盖日、周、月三种规划尺度。

**📈 对比分析**

评估方法采用滚动起点预测、对比MAPE/MASE/RMSE、模拟成本节省与ex‑post超额率；零射模型在成本上实现约52.9%月费降低，超额率低于1.5%，且与监督基线在决策分布上高度一致。

**⚠️ 局限性**

局限性包括：仅测试7台VM，缺乏多样化工作负载的泛化评估；工作负载漂移未通过实验操纵验证；未与纯规则系统做直接对照；对长期漂移的持续性验证不足。

---

## 362. Team MKC at CLPsych 2026: Capturing and Characterizing Mental Health Changes through Social Media Timeline Dynamics

**arXiv ID:** 2606.31464 | [PDF](https://arxiv.org/pdf/2606.31464v1)

**作者:** Kyomin Hwang `[一作]` (Seoul National University), Nojun Kwak `[通讯]` (Seoul National University)

**通讯引用:** 8540 | [OpenAlex ID](https://openalex.org/A5084897975)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个由五个步骤组成的LLM管线，用于对社交媒体时间序列的帖子进行情绪状态分类、适应/适应性评分、情绪变化瞬时检测、序列摘要和动态模式提取。

**💡 创新点**

创新点包括：使用逆频率加权与K折集成以缓解严重类别不平衡；将多任务（分类、回归、二分类）统一在同一LLM框架内；零样本方法提取用户的情绪变化签名；以及将帖子级分析与用户级时间建模整合为一体。

**🔧 技术方法**

主要技术：Qwen3‑4B/8B 作为特征提取器并通过LoRA微调；交叉熵、加权Huber损失、带自适应权重的二元交叉熵；监督式微调 Qwen3‑4B‑Instruct 与 Qwen3.5‑4B；零样本提示生成序列摘要和动态模式。

**📊 数据集**

使用 CLPsych 2026 共享任务提供的子集数据集（约 500 条帖子），其中包含 12 个子维度和“情绪变化瞬时”标签。

**📈 对比分析**

与零样本基线和两种不同规模的 Qwen 模型比较。K‑fold 集成显著提升第 1 步宏 F1（0.55）和第 2 步 RMSE；第 3 步 F1 达 0.55；第 4 步 Qwen3.5 取得最高分，但仍低于基线；第 5 步零样本签名提取分别得到 0.7266（改善）和 0.2301（恶化）。

**⚠️ 局限性**

局限性：训练数据极其稀缺且类别极度不平衡，导致大模型过拟合；集成方法对计算资源要求高；零样本提示对细微差异敏感；生成合成数据存在伦理与准确性风险；最终模型未充分验证临床实用性。

---

## 363. CSTrader: A Testbed for Language-Grounded Trading in a Community-Driven Virtual Asset Market

**arXiv ID:** 2606.31461 | [PDF](https://arxiv.org/pdf/2606.31461v1)

**作者:** Yao Shi `[一作]`, Yuyu Luo `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个多代理LLM框架CSTrader，用于在CS2皮肤市场上实现基于语言的交易决策，并在模拟环境中进行实时评估。

**💡 创新点**

创新点在于将市场价格、社区情绪、官方事件等异构信号统一进三层架构（感知、推理、操作），并引入了反向情绪、流动性和交易摩擦等专门代理，显著提升了语言驱动资产交易的稳定收益。

**🔧 技术方法**

采用多代理LLM架构，使用各种大型语言模型（如Claude‑sonnet‑4、Qwen‑Max等）作为基础模型，配合技术、情绪、流动性、事件、反向情绪、交易摩擦和组合管理等代理实现决策。

**📊 数据集**

利用CS2官方市场的日K线数据、Reddit社区讨论文本以及Steam官方公告等实际数据构建评估环境，覆盖高波动期（9月25日至11月15日）。

**📈 对比分析**

通过与整体CS2市场指数以及单一提示LLM基线进行对比，CSTrader在最强模型上实现了7.58%累计收益、0.49Sharpe比率、低25.9%最大回撤，并在不同模型和代理组合上进行了消融实验，验证了关键代理的作用。

**⚠️ 局限性**

局限性包括仅针对CS2皮肤市场、评估窗口短且缺乏订单簿细节、未做模型微调或强化学习实验，以及未与传统量化或强化学习基线做更全面比较。

---

## 364. Learning Fair Allocation of Indivisible Items from Limited Feedback

**arXiv ID:** 2606.31457 | [PDF](https://arxiv.org/pdf/2606.31457v1)

**作者:** Xinyu Liu `[一作]` (University of Southern California), Evi Micha `[通讯]` (University of Southern California)

**通讯引用:** 133 | [OpenAlex ID](https://openalex.org/A5057226813)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种交互式学习框架，在仅获得公平性违规证据（如EF1、PROP1）的反馈下寻找公平分配，并针对不同的反馈模型给出了多种算法。

**💡 创新点**

创新点在于将公平分配问题映射到等价查询/在线学习模型，利用ellipsoid方法将违规反馈转化为分离超平面，从而在多项式交互次数内收敛；并证明在单调计价下存在“几乎连续”EF1分配，进一步展示了在更弱的反馈模型下的指数下界。

**🔧 技术方法**

核心技术包括：等价查询模型、ellipsoid算法、Sperner定理的离散化与构造、近似分配子程序（如DoubleRoundRobin、Envy-Cycle Elimination）以及黑名单计数分析。

**📊 数据集**

本文为理论工作，未使用任何外部数据集，全部证明与分析均基于形式化模型和算法构造。

**📈 对比分析**

与传统需要完整效用信息或多轮查询的算法相比，本文在加性价值下实现了多项式交互次数和多项式运行时间；在所有违规信息模型下仍保持多项式交互复杂度；但在最小反馈模型下给出了指数交互下界。

**⚠️ 局限性**

局限性：对一般单调计价的内部计算仍是NP-hard，未给出多项式时间求解几乎连续EF1分配；在最小反馈模型下存在指数交互下界；对混合计价或更复杂偏好函数的可扩展性有限。

---

## 365. Towards a foundational model for recognising diastematic Gregorian notation

**arXiv ID:** 2606.31454 | [PDF](https://arxiv.org/pdf/2606.31454v1)

**作者:** Daniel Kurek `[一作]` (Charles University), Jan Hajič `[通讯]` (Charles University)

**通讯引用:** 8213 | [OpenAlex ID](https://openalex.org/A5102738283)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过统一四个不同编码的 Gregorian 旋律识别数据集，将它们转换为一种共同的编码，并训练一个共享的端到端模型，在合并数据后实现了新的最优性能。

**💡 创新点**

创新点在于提出了一种基于 S‑GABC 的统一编码方案并实现了多数据集的融合与去重，首次实现了共享预训练模型与微调的管道，显著提升了识别效果。

**🔧 技术方法**

使用了 Lark 解析器将 GABC、S‑GABC、Pseudo‑GABC 转换为共享格式，随后采用 DAN（Dynamic Attention Network）网络在合并数据上进行训练，并采用 8 倍梯度累积与微调策略。

**📊 数据集**

使用了四个公开的 Gregorian OMR 数据集：GregoSynth、Salzinnes、Einsiedeln、Solesmes，并在 HuggingFace 上获取。

**📈 对比分析**

通过与先前最优模型（如 DAN baseline、Sheet Music Transformer）在 MER、CER、SylER、ALER、AMLER 等指标上的对比，发现共享+微调模型在三大真实数据集上达到或超过了最佳基线，并在合并数据上显著优于单数据集训练。

**⚠️ 局限性**

限制包括：不同编码之间的转换可能丢失部分格式信息，实验仍未在所有基线模型上进行同等训练，且仅针对四个数据集，未检验模型在更广泛或更稀缺的手稿上的泛化能力。

---

## 366. Contextual Slate GLM Bandits with Limited Adaptivity

**arXiv ID:** 2606.31449 | [PDF](https://arxiv.org/pdf/2606.31449v1)

**作者:** Tanmay Goyal `[一作]` (Microsoft Research India), Gaurav Sinha `[通讯]` (Microsoft Research India)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在有限适应性（批处理和稀疏切换）条件下的上下文幻灯片带模型（GLM）bandit问题，提出了两种算法并证明了无 κ 影响的最优渐进 regret，且每轮计算复杂度为多项式级。

**💡 创新点**

创新点包括：① 在有限适应性框架下首次给出针对幻灯片 GLM 的 κ‑free 递归 regret 上界；② 通过 G‑optimal 设计与缩放设计矩阵实现对非线性参数 κ 的消除；③ 在稀疏切换设置中将参数更新次数降低到 O(Nd log T)；④ 通过批处理实现 log log T 次更新并实现可并行化。

**🔧 技术方法**

主要技术手段有：G‑optimal 设计、缩放（scaled）设计矩阵、置信区间 UCB/LCB、GLM 最大似然估计、分批/稀疏切换更新策略、对设计矩阵的正定性和多项式时间的行列式检验。

**📊 数据集**

在实验中使用两类合成数据集（S1、S2）以及实际语言模型（RoBERTa‑large + Nomic‑Embed‑Text‑v1.5）在 SST‑2 二分类任务上进行示例选择的 Prompt 调优。

**📈 对比分析**

对比方法包括：其他有限适应性基线（如算法 2、3、5）以及完全自适应的全适应性算法（算法 1）。实验结果表明：① 在合成任务中，两种新算法均实现子线性 regret 并显著优于有限适应性基线；② 与全适应性算法相比，gap 较小，特别是改进版算法与全适应性算法表现几乎相同；③ 在 Prompt 调优任务中，新算法在仅 log log T 次更新下仍显著优于无示例和随机示例基线，并与全适应性算法相当。

**⚠️ 局限性**

限制主要在于：① 仍需要“多样性”假设以保证设计矩阵的条件数；② 对于极大规模的 slates（K^N 非常大）尚未给出更紧的理论上界；③ 目前实验仅针对 Logistic GLM，其他非线性 GLM 的实证验证不足；④ 置信区间阈值和缩放参数的选择依赖经验或预设常数，实际部署时需调优。

---

## 367. Temporal Training Strategies for Left Atrium and Left Atrial Appendage Segmentation in Dynamic Contrast 4DCT

**arXiv ID:** 2606.31444 | [PDF](https://arxiv.org/pdf/2606.31444v1)

**作者:** David Montalvo-García `[一作]` (Universidad Politécnica de Madrid), María J. Ledesma-Carbayo `[通讯]` (Universidad Politécnica de Madrid)

**通讯引用:** 4668 | [OpenAlex ID](https://openalex.org/A5012910949)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

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

## 368. Wisdom Of The (AI) Crowd: Investigating Artificial Swarm Intelligence In Large Language Models

**arXiv ID:** 2606.31404 | [PDF](https://arxiv.org/pdf/2606.31404v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 369. Ask the World Before Acting: Budgeted Environment Probing for World-Model Calibration

**arXiv ID:** 2606.31422 | [PDF](https://arxiv.org/pdf/2606.31422v1)

**作者:** Xinyuan Song `[一作]` (Emory University), Zekun Cai `[通讯]` (University of Tokyo)

**通讯引用:** 399 | [OpenAlex ID](https://openalex.org/A5108659841)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了长周期语言代理在规划过程中主动查询环境以修复其世界模型漂移的机制，并提出了预算化的探测算子。

**💡 创新点**

创新点在于：① 设计了基于结构化信念表的探测算子和分层的probe‑action理论；② 通过理论证明结构化分数（关键性+依赖性）比自报不确定性更可靠；③ 在实验中展示了探测与任务动作之间的Pareto前沿。

**🔧 技术方法**

技术手段包括：使用GPT‑4o mini进行推理与信念更新；构造结构化信念表并计算探测分数 ρ_i = c_i + s_i + u_i + d_i；实现三种对照策略（无探测、随机、周期）和自报不确定性策略；理论分析子模函数和成功边界。

**📊 数据集**

数据集/环境：1) ToolDAGWorld（工具依赖，程序化字段）；2) ObjectStateWorld（可移动物体与锁状态，空间字段）；3) GraphNavWorld（图导航，空间字段）。每个环境均公开黄金字段状态以直接评估世界模型误差。

**📈 对比分析**

比较方法：对比 7 种策略（No-Probe、Random-Probe、Periodic-Probe、Self-Uncertainty、Simple、Judge、Oracle-Probe）在同一探测预算下进行；评价指标包括世界状态准确率（WSA）、任务成功率（TS）和有效探测率（UPR）。实验结果显示，Simple（或结构化(c+d)）在程序化环境中 WSA 提升约 11.7pp，空间环境提升约 3.8pp；与周期/随机相比显著提升；Oracle-Probe 作为上限，显示了最大可达性能；同时展示了任务成功率与 WSA 的Pareto权衡。

**⚠️ 局限性**

局限性：① 探测消耗时间会占用任务动作，导致任务完成率下降；② 自报不确定性易产生误导，尤其在程序化字段上；③ 仅在受限的模拟环境中验证，缺乏对真实 Web/机器人场景的实证；④ 需要根据任务类型手动调节探测预算；⑤ LLM 对信念表的自报与真实环境对齐仍存在不确定性。

---

## 370. Think While You Map: Asynchronous Vision-Language Agents for Incremental 3D Scene Graphs

**arXiv ID:** 2606.31471 | [PDF](https://arxiv.org/pdf/2606.31471v1)

**作者:** Deniz Bickici `[一作]` (University of Stuttgart), Dieter Schmalstieg `[通讯]` (University of Stuttgart)

**通讯引用:** 16649 | [OpenAlex ID](https://openalex.org/A5048047909)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个异步的开放词汇3D场景图生成框架ThinkGraphs，在在线映射与VLM推理之间解耦，使得在探索过程中场景图可查询且不断增量丰富。

**💡 创新点**

异步VLM后台代理（Critic、Description）实现语义循环闭环与属性注释；基于概率体素的关联与文本引导特征选取提升追踪稳定性；多目标帧调度显著降低VLM调用量。

**🔧 技术方法**

在线RGB‑D映射、概率体素一致性追踪、CLIP文本与视觉嵌入、Qwen3‑VL、Grounded‑SAM、VLM调用（GPT‑5‑mini）、Set‑of‑Mark提示、语义循环闭环与多目标调度等技术。

**📊 数据集**

Replica、ScanNet（语义分割）以及ScanRefer、Sr3D+、Nr3D（视觉定位）等公开数据集。

**📈 对比分析**

与现有开放词汇3D场景图方法（如ConceptGraphs、OpenVox、Open3DSG等）在Replica/ScanNet上取得最高mIoU、f‑mIoU；在三大视觉定位基准上比SOTA提升15.3–18.8% A@0.25，整体性能显著优于批处理方案。

**⚠️ 局限性**

前端模型（Qwen3‑VL+Grounded‑SAM）难以实现实时；异步代理滞后导致快速探索时新对象的属性更新延迟。

---

## 371. Visual Semantic Entropy: Do Vision Language Models Recognize Visual Ambiguity?

**arXiv ID:** 2606.31407 | [PDF](https://arxiv.org/pdf/2606.31407v1)

**作者:** Ta Duc Huy `[一作]` (Adelaide University), Vu Minh Hieu Phan `[通讯]` (Adelaide University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种新的视觉问答不确定性估计方法Visual Semantic Entropy（VSE），通过仅对图像进行微小扰动、聚类生成答案并计算语义原型的加权分散度来衡量视觉模糊性。

**💡 创新点**

创新点在于（1）仅扰动视觉输入避免文本提示敏感性；（2）将答案聚成语义原型，消除词汇差异导致的误差；（3）将视觉扰动与语义聚类结合，显著提升对视觉歧义的不确定性捕捉。

**🔧 技术方法**

技术手段包括：图像加噪扰动（σ=20），多样本生成（M=10），答案语义距离计算（使用DeBERTa-v2-xlarge-mnli嵌入），层次聚类构造原型，计算加权语义距离作为不确定度；评估采用AUC指标。

**📊 数据集**

使用了五大VQA基准：OKVQA、AOKVQA、MMVet、VILP、VLM‑are‑biased，并在五种视觉语言模型上进行实验（Qwen2.5‑VL‑7B、Gemma3‑4B、Intern3.5‑VL‑8B、LLaVA‑NeXt‑8B、Qwen3‑VL‑8B）。

**📈 对比分析**

与Verbalized、Logit‑based、Consistency‑based、以及多种Entropy‑based方法（SE、SNNE、KLE、VL‑Uncertainty）比较，VSE在所有模型与数据集上均取得最高AUC，提升幅度从约1–9%不等，尤其在视觉歧义强的VILP与VLM‑are‑biased上优势最显著。

**⚠️ 局限性**

局限性包括：需要额外的扰动采样导致计算成本增加；依赖于答案语义距离模型的质量；对解码温度敏感；未在更大模型规模或其它多模态任务中验证。

---

## 372. One Video, One World: Turning Monocular Video into Physical 4D Scenes

**arXiv ID:** 2606.31388 | [PDF](https://arxiv.org/pdf/2606.31388v1)

**作者:** Junhao Chen `[一作]` (Tsinghua University), Yufei Wang `[通讯]` (SparcAI Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套无训练的多阶段管道，将单目视频转换为实例级、可用于物理仿真的 4D 网格场景。

**💡 创新点**

首次实现无训练的实例级 4D 网格重建，直接通过顶点变形捕获刚体与非刚体运动，并通过基于物理的场景组装解决接触与支撑。

**🔧 技术方法**

利用预训练的视觉语言模型进行实例分解、SAM3 进行分割、基于 3D 生成模型（Hi3DGen、Motion324）构造网格、FoundationPose 进行姿态追踪、RoMa 对齐、VGGT 等视觉几何基座恢复尺度，以及物理约束的接触投影与地面估计。

**📊 数据集**

在自建的 OVOW-3D-Scene-Bench（120 静态场景）与 OVOW-4D-Scene-Bench（120 动态场景）上评估，同时使用真实视频构造 paired video‑to‑4D 数据；比较基线基于现有单图像 3D 重建方法。

**📈 对比分析**

与 CAST、SAM3D、VIGA、MIDI、SceneGen、TabletopGen 等单帧方法对比，评估 Scene‑IoU、Object‑IoU、PL、N‑CLIP 等结构化指标；OVOW 在静态、动态基准上均取得最高布局与几何精度，单帧运行时仅为 272‑272 s，视频平均每帧 3.35 s，速度比基线提升一至两位数，且在物理仿真中保持稳定。

**⚠️ 局限性**

对高度遮挡或极端动态场景仍易失败，部分细粒度形变细节缺失，且依赖预训练模型的推理速度和显存；缺乏对多相变形或非刚性体形状变化的精细建模。

---

## 373. MS-Resampler: Multi-Scope Visual Resampling for Efficient Multimodal LLMs

**arXiv ID:** 2606.31383 | [PDF](https://arxiv.org/pdf/2606.31383v1)

**作者:** Zhongyang Li `[一作]` (Li Auto Inc.), Kaiwen Long `[通讯]` (Li Auto Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了多尺度视觉重采样框架 MS-Resampler，能够在固定视觉 token 预算下同时捕获局部细节与全局语义。

**💡 创新点**

创新点在于：① 引入显式空间范围先验的多尺度重采样分支；② 通过可学习的注意力偏置控制聚合范围；③ 使用轻量级通道级 MLP 对不同尺度输出进行融合。

**🔧 技术方法**

技术手段包括：CLIP ViT 视觉编码器、可学习查询的跨注意力重采样、空间注意力偏置（窗口大小控制）、多分支融合（channel‑wise MLP）、两阶段冻结‑解冻训练策略。

**📊 数据集**

使用数据集：LLaVA‑1.5 558K 图文对预训练 + 665K 多模态指令调优数据，验证模型在 Vicuna‑7B/13B、Qwen2.5‑3B 等多种 LLM 上的效果。

**📈 对比分析**

与单尺度全局重采样、MLP 投影、TokenPacker、LDP‑v2、C‑Abstractor、SAEP 等基线在 10 个公开多模任务上进行对比，MS‑Resampler 在相同 token 数（144/64/36）下平均提升 1–3 分，且在极度压缩场景中仍保持最佳或接近最优性能。

**⚠️ 局限性**

局限性：需要手动设置多尺度窗口大小；对极高分辨率或长视频场景的适配尚未评估；在更大 LLM 或不同任务域下的通用性仍需进一步验证。

---

## 374. Revising RVL-CDIP: Quantifying Errors and Test-Train Overlap

**arXiv ID:** 2606.31446 | [PDF](https://arxiv.org/pdf/2606.31446v1)

**作者:** Stefan Larson `[一作]` (Vanderbilt University), Kevin Leach `[通讯]` (Vanderbilt University)

**通讯引用:** 1209 | [OpenAlex ID](https://openalex.org/A5030030910)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对RVL-CDIP数据集进行全面审核，定位12%标签错误并发现35%测试-训练重叠，随后修复错误并去重，重新评估多种文本、图像、融合与零样本模型。

**💡 创新点**

首次系统量化RVL-CDIP标签错误与数据重叠，提出滤波-精细化检测方法，并证明修复错误显著提升OOD泛化。

**🔧 技术方法**

采用手工标注、CLIP图像嵌入、SuperPoint+LightGlue特征匹配、文本相似度匹配等技术。

**📊 数据集**

RVL-CDIP（40万文档）及其OOD扩展RVL-CDIP-N（1002文档）。

**📈 对比分析**

在多种模型（BERT、RoBERTa、LayoutLMv3等）上对比原始、错误去除、错误修复+再训练、去重等版本，发现错误修复提升约3–5个百分点，去重导致1.8–3.6个百分点下降，OOD性能提升最高达14个百分点。

**⚠️ 局限性**

实验仅在部分模型上重新训练，未覆盖所有预训练模型，且去重策略可能删掉难题样本导致整体性能下降。

---

## 375. Clinically Structured Rank-Gated LoRA for Cross-Benchmark Medical Question Answering

**arXiv ID:** 2606.31432 | [PDF](https://arxiv.org/pdf/2606.31432v1)

**作者:** Yining Huang `[一作]` `[通讯]` (Meta Emergence Laboratory), Yining Huang (Meta Emergence Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种单适配器的双轴rank‑gate LoRA方法，用于医学多选题答案的参数高效微调；

**💡 创新点**

创新点在于将医学专业/职业轴和临床操作轴与隐藏语义共同作为路由信号，在单个LoRA基底内部动态选择稀疏rank原子，并引入注入系数来调节更新强度；

**🔧 技术方法**

主要技术包括LoRA、rank‑wise 动态激活、top‑k稀疏掩码、注入系数、正则化（熵、平衡、正交、轴对比），并采用配对bootstrap和符号检验验证统计显著性；

**📊 数据集**

使用的医学QA数据集包括中文的CMB、CMExam以及英文的MedQA、MedMCQA，基底模型为Qwen3‑8B（也在Llama3.1‑8B上做对照）；

**📈 对比分析**

与基线LoRA（不同rank）、rank‑wise控制、MoELoRA、DoRA以及多适配器库（MedAdapter、Arrow、MeteoRA、LoRA‑Mixer）等进行对比，单适配器rank‑gate LoRA在四大基准宏平均准确率上达69.31%，比MoELoRA高0.89个百分点，且参数量比MoELoRA少28%；

**⚠️ 局限性**

局限性包括仅使用单一训练种子导致缺乏跨种子稳健性；医学专业与操作标签为弱规则推断，未经过专家验证；注入系数仅为经验性调节，非严格风险校准；实验仅覆盖多选问答，未验证临床部署安全。

---

## 376. DA-Studio: An Agentic System for End-to-End Data Analysis

**arXiv ID:** 2606.31423 | [PDF](https://arxiv.org/pdf/2606.31423v1)

**作者:** Yizhe Liu `[一作]` (Renmin University of China), Ju Fan `[通讯]` (Renmin University of China)

**通讯引用:** 3155 | [OpenAlex ID](https://openalex.org/A5100739546)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 DA-Studio，一套基于 Web 的交互式演示系统，实现了自主、沙箱化、可检视的端到端数据分析流程。

**💡 创新点**

其创新点包括通过动作协议实现多步自动化分析、为每个会话提供隔离沙箱执行与可追踪工件管理，以及在浏览器端统一展示行动轨迹、代码与结果，支持交互式修改与导出。

**🔧 技术方法**

技术方案涵盖 LLM 动作生成（DeepAnalyze 或通用 LLM）、五层架构（应用、模型、上下文、数据、环境）、Docker 沙箱、基于会话的工作空间与工件索引、流式 UI 与代码编辑器。

**📊 数据集**

演示使用了多格式商户支付数据集（包含 CSV、JSON、PDF 等文件），作为上传输入与自然语言任务的测试。

**📈 对比分析**

与 LIDA、Data Formulator 2、nvAgent 等现有可视化或子任务专用系统对比，DA-Studio 展示了完整的执行链与可视化流程，但本文未给出定量性能指标，仅通过演示视频说明交互体验和可检视性。

**⚠️ 局限性**

局限性在于缺乏大规模量化评估、对 LLM 生成代码的可靠性和执行错误的处理不完善、执行速度受容器启动与数据大小影响、以及对更复杂、多模态任务的适应性有限。

---

## 377. A history of GDPR cookie banner compliance: the roles of publishers, regulators and CMPs

**arXiv ID:** 2606.31485 | [PDF](https://arxiv.org/pdf/2606.31485v1)

**作者:** Yana Dimova `[一作]` (KU Leuven), Wouter Joosen `[通讯]` (KU Leuven)

**通讯引用:** 12578 | [OpenAlex ID](https://openalex.org/A5054031138)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对2018-2024年间30个欧盟/EEA国家共11,364个网站的cookie banner进行纵向追踪分析，探讨网站所有者、监管机构和CMP在合规性提升中的作用。

**💡 创新点**

首次将Web存档（Wayback Machine与HTTPArchive）结合，复现历史页面以检测banner；系统性关联DPA执法与合规率；对18个CMP进行横向比较，揭示CMP对合规影响有限。

**🔧 技术方法**

使用基于OpenWPM的自动检测工具，结合HTML/CSS过滤、语言模型识别cookie相关术语、机器学习分类banner选项；通过Wayback Machine API补全缺失资源；利用HTTPArchive HAR数据做历史重放。

**📊 数据集**

大规模样本：11,364个网站，覆盖25个时间点、30个国家；HTTPArchive每月抓取的HAR文件；Wayback Machine存档；通过Easylist Cookie手动筛选识别18个CMP。

**📈 对比分析**

采用时间序列统计、国家对比、CMP横向比较；计算合规率、接受/拒绝按钮比例；使用Pearson相关检验DPA活动与合规率关系；检测精度91.5%/召回65.2%，总体合规率从2.94%提升至30.66%。

**⚠️ 局限性**

受限于Web存档动态内容缺失、资源补全不完整导致低估合规率；检测只覆盖“拒绝”按钮的客观指标，忽略主观合规要素；IP位置偏差、语言翻译误差、部分CMP识别漏报；整体为下限估计，未做完整法律解释。

---

## 378. A Large-Language-Model Supported Personalized Driving Framework for Lane Change in Highway Scenarios

**arXiv ID:** 2606.31483 | [PDF](https://arxiv.org/pdf/2606.31483v1)

**作者:** Dong Bi `[一作]` (Hubei University of Automotive Technology), Arno Eichberger `[通讯]` (Graz University of Technology)

**通讯引用:** 1770 | [OpenAlex ID](https://openalex.org/A5084393303)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于大型语言模型（LLM）的个性化驾驶框架，能够将自然语言的车道变道指令映射为可执行的Apollo规划参数，实现不同驾驶风格（激进、正常、保守）的差异化变道行为。

**💡 创新点**

创新点在于：①将自然语言指令与规划参数直接映射，支持隐式偏好解析；②使用检索增强生成（RAG）构建指令-风格对应数据集，显著提升隐式命令的识别准确率；③在Apollo自动驾驶栈中实现可调参数集，并通过聚类+强度排序得到多层次风格参数。

**🔧 技术方法**

技术主要包括：大型语言模型（多模态 LLM 生成、零射/ RAG 推理）、Apollo 规划模块（纵向/横向分段-jerk 优化）、CarMaker 共仿真、FAISS 索引、OpenAI API 调用、Python 训练脚本。

**📊 数据集**

使用自行生成的离线仿真数据集（465 条样本，425 有效），包含参数空间、行为指标（变道时长、加速度、侧向加速度、转向角、偏航率）；以及由多代理 LLM 生成并人工审核的 RAG 数据集，用于风格分类。

**📈 对比分析**

与零射推理相比，RAG 在显式、混合、隐式三种命令集上均提升识别准确率；平均提升约 3.4%、6.1%、10.2%（隐式最高）。系统层面实验验证不同风格与强度的 KPI（变道时长、最大侧向加速度、累计侧向加速度）表现出显著差异，能够区分激进到保守的连续变化。

**⚠️ 局限性**

局限性包括：①仅在高速公路变道场景下验证，缺乏复杂交通环境的测试；②风格强度采用离散的 L1/L2/L3 级别，未实现连续调节；③缺乏驾驶员实时反馈的验证，未能证明用户对生成风格的主观满意度。

---

## 379. TabPATE: Differentially Private Tabular In-Context Learning Without Public Data

**arXiv ID:** 2606.31474 | [PDF](https://arxiv.org/pdf/2606.31474v1)

**作者:** Dariush Wahdany `[一作]` (CISPA), Franziska Boenisch `[通讯]` (CISPA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种在不需要公开数据的前提下，通过PATE框架实现差分隐私的表格式上下文学习方法。

**💡 创新点**

创新点在于：将PATE迁移到表格I‑C‑L，利用特征边界或少量DP边缘分布生成合成查询，从而消除对公共在分布数据的需求；并证明在保留高效性的同时显著降低了成员推断攻击的成功率。

**🔧 技术方法**

采用的技术包括：PATE知识迁移、Confident‑GNMax聚合、差分隐私噪声注入、合成表格查询生成（基于特征边界或DP边缘统计）、以及TabPFN等表格基础模型。

**📊 数据集**

使用了OpenML公开的表格分类数据集（German Credit、Blood Transfusion、Diabetes、Phoneme、Wilt）以及若干回归数据集（Abalone、California Housing、Diabetes（reg.）、Wine Quality、Bike Sharing、CPU Small、Ames Housing、Superconduct）。

**📈 对比分析**

与DP‑SGD、DP‑TabICL、DP‑Synthetic、PromptPATE（需公共数据）及Query‑Time等基线比较，在ε=10时TabPATE的平衡准确率达到71.2%，优于PromptPATE的68.6%；在ε=1时仍保持高于多数公开数据无关基线；同时在成员推断攻击下，TabPATE将成功率降至接近随机。

**⚠️ 局限性**

局限性包括：对高维或极度不平衡的数据仍可能需要投入部分预算生成DP边缘分布；回归任务表现相对逊色；生成合成查询的覆盖范围受特征边界限制；实现成本依赖于教师模型数量和查询生成开销。

---

## 380. Zero-Shot Quantization for Object Detectors using Off-the-Shelf Generative Models

**arXiv ID:** 2606.31456 | [PDF](https://arxiv.org/pdf/2606.31456v1)

**作者:** Hyunho Lee `[一作]` (Seoul National University), Nojun Kwak `[通讯]` (Seoul National University)

**通讯引用:** 8540 | [OpenAlex ID](https://openalex.org/A5084897975)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了零样本量化对象检测（Zero-Shot Quantization for Object Detection, ZSQ-OD），提出了GoodQ框架，利用离线生成模型合成高密度、多实例的训练集，并在此基础上进行量化感知训练；

**💡 创新点**

创新点主要有三：1）信息密集提示（IDP）通过定制prompt生成多实例、高信息量图像；2）本征分布感知选择（IDAS）利用检测头偏置估计真实类别分布并从生成图像池中按分布采样；3）教师引导自适应噪声抑制（TANR）用教师模型的软标签替代噪声伪标签并通过自适应权重减弱噪声影响；

**🔧 技术方法**

技术方法包括离线扩散生成模型（如Stable Diffusion）生成图像、信息密集提示策略、Intrinsic Distribution-Aware Selection、Teacher-guided Adaptive Noise Reduction、LSQ量化方案以及YOLOv5/YOLOv11目标检测网络；

**📊 数据集**

实验使用MS-COCO 2017数据集进行训练与评估，并通过其类分布估计来指导数据生成；

**📈 对比分析**

与LSQ、LSQ+、带标签的TSOD以及无标签TSOD^†等方法在YOLOv5/YOLOv11上进行mAP/mAP50对比。GoodQ在高比特（W8A8、W6A6）下与现有方法相当，在低比特（W4A4、W3A3）下显著优于TSOD，保持约20‑30%的mAP，仅比原始全精度模型低约10‑20%；

**⚠️ 局限性**

局限性包括：1）需要大规模图像池（160k）生成成本较高；2）在极低位（W3A3）性能仍未达到全精度水平；3）类别分布估计依赖检测头偏置，可能对不同模型不通用；4）实验仅在YOLO系列网络上验证，未探究其他检测框架的适用性。

---

## 381. Xiaomi-GUI-0 Technical Report

**arXiv ID:** 2606.31410 | [PDF](https://arxiv.org/pdf/2606.31410v1)

**作者:** Wanxia Cao `[一作]`, Cong Zou `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出一种面向真实移动设备的端到端多模态 GUI 代理训练与评估闭环，构建了物理设备主导的混合基础设施、错误驱动的数据飞轮以及分阶段（SFT → Step RL → Agentic RL）的训练流程，实现了从头函数到长链推理再到错误恢复的完整能力提升。

**💡 创新点**

创新点在于：①将真实设备作为主要执行环境，突破传统仿真/模拟的局限；②通过错误驱动数据飞轮把真实跑回中出现的错误主动转化为纠正、反思与恢复的监督信号；③采用分阶段 RL（Step RL + Agentic RL）使模型在局部决策和全局轨迹层面同时得到强化学习的指导；④设计细粒度子目标评估与双重验证（XML + 逻辑规则）以实现可复现且可解释的真实设备基准 "RealMobile"。

**🔧 技术方法**

技术手段包括：混合资源管理与设备拉取调度、统一动作空间与低延迟浏览器控制、结构化 chain‑of‑thought (CoT) 生成与校验、GSPO（组序列策略优化）用于 Step RL 与 Agentic RL、动态采样与奖励分层（cascade reward）以及基于 LLM 的判分与纠正策略。

**📊 数据集**

数据集涵盖：①高频任务数据（≈5k样本，包含登录、验证码、支付等异常状态）；②高泛化数据（基于功能树与行为桶合成，覆盖数千条长尾意图）；③代理能力增强数据（结构化 CoT、记忆、规划等）；④错误驱动飞轮数据（人工标注错误点、教师模型修正轨迹）。此外使用了公开基准 ScreenSpot‑V2、MMBench‑GUI‑L2、OSWorld‑G 以及自研 RealMobile 基准进行评估。

**📈 对比分析**

与同类模型对比，本文在公开视觉语言定位基准上保持竞争力（ScreenSpot‑V2 94.7%，MMBench‑GUI‑L2 82.7%）；在 AndroidWorld 上获得 78.9% 的成功率，超越 UI‑TARS、GUI‑Owl、UI‑Venus 系列；在自研 RealMobile 上实现 72.0% 的任务完成率，显著优于开源模型（≈33%）和部分闭源模型（Gemini 3.1 Flash 58%，Claude 60%）。

**⚠️ 局限性**

局限性包括：①仍需大量人工参与错误标注和教师模型覆盖；②混合设备基础设施对硬件资源与维护成本要求高；③在极端网络波动或极端异常场景下模型鲁棒性尚未完全验证；④多模态推理和 CoT 结构化仍可能对模型容量有较高依赖，导致小模型适配困难。

---

## 382. MAPE: Defending Against Transferable Adversarial Attacks Using Multi-Source Adversarial Perturbations Elimination

**arXiv ID:** 2606.31378 | [PDF](https://arxiv.org/pdf/2606.31378v1)

**作者:** Xinlei Liu `[一作]` (Information Engineering University), Zhen Zhang `[通讯]` (Information Engineering University)

**通讯引用:** 80096 | [OpenAlex ID](https://openalex.org/A5100459168)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于外部防御模型的多源对抗扰动消除框架（MAPE），通过对抗扰动消除网络（CAU‑Net）对输入样本进行预处理，从而提升目标分类模型在黑盒可迁移对抗攻击中的鲁棒性。

**💡 创新点**

创新点包括：①提出单源对抗扰动消除（SAPE）机制，使防御网络学习消除已知攻击模型产生的扰动；②设计预训练模型概率调度算法（PPSA），结合模型差异概率和负动量概率动态选择多源预训练模型，以最大化训练样本的多样性；③将两者融合得到MAPE，实现对未知来源对抗攻击的高效防御。

**🔧 技术方法**

核心技术包括：U‑Net结构的通道注意力改造（CAU‑Net）作为对抗扰动消除网络；利用输出分布的L1 Wasserstein距离量化模型差异；负动量正则化来避免模型选择偏倚；以及基于混合样本的对抗训练策略。

**📊 数据集**

使用了CIFAR‑10、CIFAR‑100、Mini‑ImageNet等标准图像分类数据集进行评估，并以ResNet‑34为目标模型，ShuffleNet‑V2‑2×、ResNet‑V2‑50等多种预训练模型生成对抗样本。

**📈 对比分析**

与自然训练、对抗训练以及多种输入变换防御方法（TVM、Feature Denosing、Pixel Deflection、Mixup Inference、HGD、LDT）对比，MAPE在对抗攻击（FGSM、BIM、DIM、PGD、UPGD、VNIM等）下取得平均防御准确率约95%（CIFAR‑10）、72%（CIFAR‑100）和71%（Mini‑ImageNet），在未知、集成以及强替代模型攻击场景中均优于或与最优。

**⚠️ 局限性**

局限性在于：1）在白盒自适应攻击场景下仍表现不佳；2）训练成本和显存占用相对较高；3）防御效果对目标模型的依赖仍存在，需进一步探索对多任务、检测、分割等视觉任务的迁移。

---

## 383. PEERS: A Parallel and Exact Effective Resistance Solver via Implicit Inversion and Augmented Symbolic Analysis

**arXiv ID:** 2606.31535 | [PDF](https://arxiv.org/pdf/2606.31535v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 384. A Large-Scale Empirical Evaluation of MMAO Under Fair-Budget Continuous and Discrete Benchmarks

**arXiv ID:** 2606.31584 | [PDF](https://arxiv.org/pdf/2606.31584v1)

**作者:** Jinliang Xu `[一作]`, Liping Ma `[通讯]` (Seventh Medical Center Of Chinese PLA General Hospital)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Metabolic Multi-Agent Optimizer（MMAO）在连续（CEC2017）和离散（TSPLIB、OR-Library多背包）基准上进行大规模公平预算评估，并给出机制轨迹诊断与消融实验。

**💡 创新点**

证明MMAO闭环资源分配原则在更严格、更大规模的评估协议下仍保持竞争力，并通过轨迹诊断解释其内部动态机制。

**🔧 技术方法**

采用MMAO实现、可重现的PSO-lite/DE-lite/ES-lite等基线、基于CEC2017/TSPLIB/OR-Library的标准基准、非参数统计（平均排名、Mann-Whitney检验）以及轨迹诊断工具。

**📊 数据集**

CEC2017（10D、30D 八个函数）、TSPLIB（五个实例）、OR-Library多背包（辅助实验）等数据集。

**📈 对比分析**

方法通过匹配预算、种子、统一终止条件和平均排名、非参数统计进行比较；MMAO在连续端平均排名≈1.06、误差远低于基线；在离散端平均排名1、平均百分比差距7.84%（低于基线14–400%）。

**⚠️ 局限性**

局限性包括：基准覆盖范围有限、未与全量竞争级基线对比、对高维复杂函数的最终精度不足、消融实验未能完全隔离各子机制、离散验证仍以TSP和小型背包为主，未覆盖更广泛组合优化问题。

---

## 385. Stabilization Learning: A Paradigm Transition Bridging Control Theory and Machine Learning

**arXiv ID:** 2606.31562 | [PDF](https://arxiv.org/pdf/2606.31562v1)

**作者:** Quan Quan `[一作]` (Beihang University), Quan Quan `[通讯]` (Beihang University)

**通讯引用:** 6621 | [OpenAlex ID](https://openalex.org/A5058029021)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

提出了“稳定化学习（Stabilization Learning）”这一跨越控制理论与机器学习的新范式，并给出了统一的六元组框架、约束空间与跟踪问题的七元组扩展，系统性阐述了稳定化学习的概念、分类、关键要素、度量与策略；

**💡 创新点**

创新点在于：①将系统稳定性作为学习目标而非仅追求最优；②构建了一套基于状态、动作、植物、策略、目标状态与度量的通用六元组模型，并通过Barrier空间与输出函数扩展为七元组；③将控制、观测、识别三大经典任务统一为闭环稳定性问题；④将Lyapunov、Barriers与深度特征抽取等多种技术融合，为学习过程提供正式的稳定性证明；

**🔧 技术方法**

采用Lyapunov函数与不等式约束、控制障碍函数、输入状态稳定性（ISS）、深度特征提取、可观测/可控性约束、世界模型、吸引子网络、Invariant学习、强化学习与HJB、以及相关的优化与数据驱动方法；

**📊 数据集**

该工作以理论与案例为主，未对公开数据集进行实验；主要通过多旋翼UAV悬停、视觉姿态估计、棋类对弈、Push‑T等示例说明框架；

**📈 对比分析**

由于缺乏量化实验，论文未给出性能比较；通过案例分析展示框架在不同任务中的适用性与理论可行性；

**⚠️ 局限性**

局限性包括：①缺乏大规模实验验证；②Lyapunov函数与Barrier设计依赖专家知识，难以自动化；③对高维非线性系统的可扩展性与样本效率仍待提升；④假设为确定性连续系统，尚未充分覆盖随机与观测噪声环境；

---

## 386. Directed Low Diameter Decomposition for Structured Digraphs

**arXiv ID:** 2606.31560 | [PDF](https://arxiv.org/pdf/2606.31560v1)

**作者:** Shinwoo An `[一作]` (Bar-Ilan University), Arnold Filtser `[通讯]` (Bar-Ilan University)

**通讯引用:** 309 | [OpenAlex ID](https://openalex.org/A5044890369)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种新的随机划分技术，称为“scissors carving”，并利用该技术在有向图中构造低直径分解（LDD）。针对具有有限路径宽度 k 的有向图，作者证明存在 (O(k),Δ)-LDD；并利用改进的 LDD 分解证明在树宽为 k 的有向图上，定向非二分稀疏切割问题的线性规划松弛的整数间隙为 O(log n)。

**💡 创新点**

创新点主要体现在两方面：
① 通过“scissors carving”对传统球体切割（ball carving）进行改进，显著降低了对路径宽度的指数依赖，改为线性 O(k)。
② 在树宽图的稀疏切割分析中，作者重新梳理了分层量化（quasipartition）与 0‑1 伪度量空间的嵌入过程，消除了原先 O(log n) 的额外因子，得到更紧的 O(log n) 整数间隙。

**🔧 技术方法**

主要技术包括：
- 有向图的路径分区（path partition）与路径宽度概念的引入；
- 将有向图嵌入到路径分区宽度更小的图中以保持距离的等距性质；
- 递归的“scissors carving”算法，结合前向距离、后向距离两阶段的球体切割；
- 对 0‑1 伪度量空间的改进嵌入，使用多尺度球体切割与强度分析。

**📊 数据集**

本文为理论算法研究，未使用实验数据集；所有结果均在抽象的有向图上证明。

**📈 对比分析**

与此前工作相比：
- 传统路径宽度有向 LDD 的最佳已知为 (2^{O(k^2)},Δ)-LDD，本文提升到线性 (O(k),Δ)-LDD；
- 传统树宽图稀疏切割的整数间隙为 O(log^2 n)，本文降至 O(log n)。
因此在结构化有向图中显著提高了性能，尤其在路径宽度或树宽较小时效果更为显著。

**⚠️ 局限性**

主要限制包括：
- 递归深度与路径宽度成线性关系，仍然会导致较高的常数因子；
- 算法为随机化，需要期望分析，实际实现可能需要多次采样；
- 结果主要适用于结构化有向图（路径宽度、树宽有限），对一般有向图仍未得到更优的 LDD 或整数间隙；
- 在路径宽度较大的实例中，O(k) 的损失参数可能仍然较大。

---

## 387. Graph Scheduling with Group Completion Times

**arXiv ID:** 2606.31530 | [PDF](https://arxiv.org/pdf/2606.31530v1)

**作者:** Lars Rohwedder `[一作]` (University of Southern Denmark), Leander Schnaars `[通讯]` (Technical University of Munich)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种迭代舍入方法，将Coflow Scheduling的技术推广到一般图调度问题，并在数据迁移场景中得到改进的近似算法。

**💡 创新点**

通过在线性规划中加入奇集不等式并结合多重图的边着色理论，突破了以往4-近似的瓶颈，实现了在极限情况（OPT 远大于权重总和）下的 (2+ε)-近似，并在数据迁移问题中取得了 2.617 的近似比。

**🔧 技术方法**

核心技术包括：奇集不等式的多项式约束扩展、迭代舍入（Iterated Rounding）策略、Kőnig 定理与 Shannon 定理在多重图中的应用、以及对局部比率（Local Ratio）算法的细化分析。

**📊 数据集**

本研究为理论分析性质，未使用实验数据集；所有结果均为理论证明与算法复杂度分析。

**📈 对比分析**

与之前的 4 近似、3.415 近似以及 2.618 近似相比，本方法在大多数参数设置下提升了 2.617 近似（数据迁移）和 3.93 近似（通用图调度），并在 OPT 较大时可获得接近 2 的理论下界匹配的 (2+ε) 近似。

**⚠️ 局限性**

局限性包括：常数项较大，尚未在所有规模和非单位边权重的情况得到最优或可扩展性；迭代舍入过程对约束数量敏感，导致实现复杂；对非单位大小边的推广仍需新的技术。

---

## 388. FinPersona-Bench: A Benchmark for Longitudinal Psychometric Stability of Autonomous Financial Agents

**arXiv ID:** 2606.31522 | [PDF](https://arxiv.org/pdf/2606.31522v1)

**作者:** Muhammad Usman Safder `[一作]` (MBZUAI), Zhuohan Xie `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一个用于评估自适应金融代理长期行为一致性的仿真基准，探讨并量化了在长时间市场环境中大型语言模型（LLM）对行为指令（mandate）的衰减现象；

**💡 创新点**

创新点在于：①提出“Mandate Salience Decay (MSD)”这一新的行为漂移概念；②设计了分离价格与基本价值的合成市场引擎，可客观测量行为偏离；③通过“记忆重置”（mandate re-grounding）与“安慰剂控制”验证指令内容对行为稳定性的影响；

**🔧 技术方法**

技术主要包括：多层提示工程、JSON 输出约束、合成市场生成模型、MAS/CI/RG 等三类客观指标、Wilcoxon 符号秩检验、频率消融实验；

**📊 数据集**

数据集：完全合成的单资产市场序列，包含三种情景（平稳、崩盘、泡沫）以及多种噪声/失真参数；

**📈 对比分析**

比较方法：在18种前沿与开源 LLM（包括 GPT、Claude、Gemini、DeepSeek、Llama 等）上，以三种 MBTI / Big Five 人格、三种情景、两种代理架构（静态 vs. 重置）进行网格实验；性能表现显示 MSD 随时间累积，重置在平稳与崩盘情景可降低偏差，惰性在泡沫情景会产生反效果，且不同模型表现差异显著；

**⚠️ 局限性**

局限性包括：仅使用单资产单代理；MBTI 角色设置缺少神经质维度；开源模型样本有限，重置效果可能随模型容量变化；未测试多周期重置策略或事件触发机制。

---

## 389. MINT: Dynamic-Precision CNN Inference with MSDF Digit-Serial Arithmetic on FPGA

**arXiv ID:** 2606.31514 | [PDF](https://arxiv.org/pdf/2606.31514v1)

**作者:** Muhammad Usman `[一作]` (University of Regensburg), Dorit Merhof `[通讯]` (University of Regensburg)

**通讯引用:** 11756 | [OpenAlex ID](https://openalex.org/A5064747056)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出MINT动态精度CNN推理加速器，利用左到右（MSDF）算术实现按需截断。

**💡 创新点**

通过MSDF算术实现零开销动态精度，采用贪婪预算约束算法在层级上自适应分配精度，且无需硬件重配置。

**🔧 技术方法**

MSDF乘法器与加法树、冗余Signed‑Digit表示、预算约束贪婪精度分配、Zynq‑7020 FPGA实现与SAIF功耗分析。

**📊 数据集**

使用ImageNet预训练权重，并在Imagenette‑320验证集上评估精度。

**📈 对比分析**

相对于固定INT8，VGG‑16吞吐提升32.6%、能效提升82.1%；ResNet‑18吞吐提升26.0%、能效提升62.9%；在Zynq‑7020平台上，MINT在能效上超过所有列出的FPGA加速器。

**⚠️ 局限性**

多周期MSDF乘法器限制吞吐，未实现实时高速吞吐；需进一步加深流水线、扩大PE阵列、优化调度以提升性能。

---

## 390. LASER: Load-Aware Serving with Early-Exit for Reasoning LLMs at the Edge

**arXiv ID:** 2606.31580 | [PDF](https://arxiv.org/pdf/2606.31580v1)

**作者:** Zhiqing Tang `[一作]` (Beijing Normal University), Weijia Jia `[通讯]` (Beijing Normal University)

**通讯引用:** 11708 | [OpenAlex ID](https://openalex.org/A5051803761)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LASER——一种在边缘设备上针对大规模推理模型的加载感知早停方案，结合实时动态阈值和难度感知预算分配来控制链式思考长度；

**💡 创新点**

创新点包括：①通过EMA平滑的系统负载实现置信阈值的自适应调节；②利用提示长度与负载的乘性因子为每个请求预分配硬预算；③将单请求的早停方法扩展为系统级调度框架；

**🔧 技术方法**

采用置信度早停、指数移动平均（EMA）、tanh映射自适应阈值、基于提示长度的难度因子、预算预分配以及离散事件仿真；

**📊 数据集**

在GSM8K、MATH-500、AMC 2023和GPQA Diamond四个推理基准上进行实验；

**📈 对比分析**

与Vanilla、Fixed‑High、Fixed‑Low和NoThinking四种基线对比，LASER在两种4B‑7B模型下平均延迟降低17–38%，SLO满足率提升3–6个百分点，准确率仅下降约1%；

**⚠️ 局限性**

局限在于只验证单GPU单机场景，难度估计仅依赖提示长度，缺乏对多设备异构集群和模型选择路由的支持。

---

## 391. In-situ Indexing via Memristive Content-Addressable Memory

**arXiv ID:** 2606.31554 | [PDF](https://arxiv.org/pdf/2606.31554v1)

**作者:** Bing Wu `[一作]`, Dan Feng `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于 memristive ReCAM 的 in‑situ PIM 架构 PATH，用于加速哈希表和 B+Tree 索引，并在 NVM 上进行评估。

**💡 创新点**

通过在 ReCAM 中实现 in‑situ ISUD（全扫描增删）实现并行冲突解决与拆分，显著减少内存访问量、重排开销，并保持低延迟、低能耗，形成了可扩展的多线程索引加速器。

**🔧 技术方法**

采用 memristive ReCAM 阵列、in‑situ ISUD、PIM 加速、软硬件协同设计以及 NVM 可靠性机制（ECC/刷新）等技术。

**📊 数据集**

使用 YCSB 等标准键值工作负载，插入/查询比例在 100%、30%、10% 等不同配置，模拟 8 GB NVM 数据集。

**📈 对比分析**

与现有哈希方案（PCLHT、CCEH）以及传统 B+Tree（Fast&Fair、uTree、NBtree）对比，PATH‑CHS/ EH 在单线程下吞吐率提升 3.1‑5.1 倍、尾部延迟降低 5.4‑53.6 倍，能耗下降 73%/26%；在多线程下比 CCEH 提升 3.1‑5.1 倍。

**⚠️ 局限性**

仍受软件锁与并发控制开销影响，多线程下与原始方案差距收敛；实现规模仅 8 个银行，无法匹配 UPMEM 级别的并行度；未考虑 ECC/刷新等可靠性开销，实际性能略低。

---

## 392. Modality-Driven Search with Holistic Trace Judging for ARC-AGI-2

**arXiv ID:** 2606.31543 | [PDF](https://arxiv.org/pdf/2606.31543v1)

**作者:** Johan Land `[一作]` `[通讯]` (Independent Researcher), Johan Land (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套基于多模态搜索与全局判断的 ARC‑AGI‑2 视觉推理求解器，能够在少量样例下自动生成多种候选解并通过上下文完整比较挑选正确答案，

**💡 创新点**

创新点在于将文本、图像、代码三种推理渠道视为独立搜索算子，最大化候选多样性，并引入“上下文保留全局判断”机制，逆向常见的一致性投票，真正捕捉少数正确假设；同时实现了自适应早停与合成推理，显著提升解答质量；

**🔧 技术方法**

核心技术包括：多模态（文本、图像、代码）候选生成（使用 GPT‑5.2、Gemini‑3 Preview、Claude Opus 4.5 及工具交互程序生成）、长上下文综合判别模型（GPT‑5.2 x‑high 级别）、自适应早停策略、基于全程推理轨迹的“全局判断”与可合成新解，

**📊 数据集**

使用 ARC‑AGI‑2 benchmark（公开 120 题、半私有 120 题）进行评估，

**📈 对比分析**

在 ARC Prize Verified 赛制下，公开评测得到 76.11% pass@2，半私有评测 72.9% pass@2，成本约为每题 $19.69（公开）/$38.99（半私有），相比 GPT‑5.2 Pro（54.2%）和 Gemini‑3 Pro（54.0%）提升了约 18.7 个百分点，

**⚠️ 局限性**

主要限制包括：高昂的推理成本（单任务近 40 美元）、仅单次评测无置信区间、缺乏完整消融实验、无法跨任务累积知识、仅针对 ARC 领域、以及对特定模型版本与 API 参数的高度依赖。

---

## 393. DataEvolver: Self-Evolving Multi-Agent Data Construction for Text-Rich Image Generation

**arXiv ID:** 2606.31537 | [PDF](https://arxiv.org/pdf/2606.31537v1)

**作者:** Siyu Yan `[一作]` (Central South University), Alex Jinpeng Wang `[通讯]` (Central South University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种自演进多智能体框架（DataEvolver），将被拒绝样本转化为语义反馈，动态更新检索、生成和经验记忆，从而在固定数据预算下生成更高质量的文本图像数据。

**💡 创新点**

创新点在于将数据构建过程视为迭代的策略优化，而非静态抓取–过滤–冻结；通过检索器、验证器、批评家和生成器的闭环，利用拒绝原因进行自然语言级的策略修正和针对性合成。

**🔧 技术方法**

核心技术包括基于大型语言模型（如Qwen3.5、Mistral-7B）生成查询与提示、OCR（PaddleOCR）与语义一致性评估（CLIP ViT‑B/32）构成的验证器、自然语言批评反馈生成器、以及基于图像语言模型（Qwen-Image）进行的有针对性合成。

**📊 数据集**

使用了公开的文本图像数据集 MARIO‑10M、AnyWord‑3M 作为对照，并在 PixArt‑α、Show‑o2 等生成模型上构建 0.75M 样本集进行评估，亦在 TextScenesHQ 与 LongTextBench 上验证。

**📈 对比分析**

与 MARIO、AnyWord 等固定数据集基线相比，在 OCR‑F1 指标上提升显著（TextScenesHQ 4.56→8.45，LongTextBench 6.71→9.08），在视觉与语义对齐（CLIP、FID）方面保持或略优，且改进在不同生成模型之间具有良好迁移性。

**⚠️ 局限性**

局限包括依赖验证器的可靠性（OCR、语义评分误差会影响策略更新）、缺乏正式的缩放定律分析、仅针对文本图像生成任务，且需进一步验证跨域扩展的可行性。

---

## 394. On the Convergence of Self-Improving Online LLM Alignment

**arXiv ID:** 2606.31524 | [PDF](https://arxiv.org/pdf/2606.31524v1)

**作者:** Xudong Wu `[一作]` (University of Hong Kong), Jiayu Chen `[通讯]` (University of Hong Kong)

**通讯引用:** 6737 | [OpenAlex ID](https://openalex.org/A5048308937)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SAIL-RevKL算法，在原SAIL基础上加入逆KL正则化，并提供了全局收敛证明。

**💡 创新点**

通过逆KL正则实现了全局强凹性并满足Polyak‑Łojasiewicz条件，首次给出非渐近的全局线性收敛率。

**🔧 技术方法**

使用投影随机梯度上升、逆KL正则、Fisher信息矩阵分析和Polyak‑Łojasiewicz理论。

**📊 数据集**

在MuJoCo连续控制任务、Meta‑World Door Open、LLM对齐数据集PKU‑SafeRLHF和UltraFeedback上进行实验。

**📈 对比分析**

与PEBBLE、原SAIL和DPO等基线比较，SAIL-RevKL在回报、胜率等指标上均优于对照，效果显著。

**⚠️ 局限性**

仍需手动调节γ，理论假设仅适用于log‑linear策略，未涵盖全参数微调场景。

---

## 395. Support of Teleoperated Driving with 5G Networks

**arXiv ID:** 2606.31512 | [PDF](https://arxiv.org/pdf/2606.31512v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 396. RaBitQCache: Rotated Binary Quantization for KVCache in Long Context LLM Inference

**arXiv ID:** 2606.31519 | [PDF](https://arxiv.org/pdf/2606.31519v1)

**作者:** Wenhao Li `[一作]` (Renmin University of China), Xiaoyong Du `[通讯]` (Renmin University of China)

**通讯引用:** 6589 | [OpenAlex ID](https://openalex.org/A5008721449)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 RaBitQCache 框架，利用随机旋转二进制量化与 INT4 算术实现稀疏注意力，加速长上下文 LLM 推理。

**💡 创新点**

创新点：1) 将 Johnson‑Lindenstrauss lemma 与随机正交旋转结合，得到无偏估计且有可证明误差上界的二进制量化注意力权重；2) 通过该无偏估计实现自适应 Top‑p 检索，避免固定 Top‑k 造成的效率与精度折衷；3) 采用异步流水线与惰性更新等硬件友好策略，显著降低 KVCache 读写开销。

**🔧 技术方法**

使用技术：Johnson‑Lindenstrauss lemma、随机正交旋转、二进制量化、INT4 GEMV、异步流水线、惰性更新、FlashInfer、vLLM、CUDA 自定义核。

**📊 数据集**

使用数据集：LongBench、RULER、GSM8K，模型覆盖 Longchat‑7B‑v1.5‑32k、LLaMA‑3.1‑8B‑Instruct、LLaMA‑3.1‑70B‑Instruct。

**📈 对比分析**

与 Quest、DS、SparQ、MagicPIG、PyramidKV、SnapKV、PQCache、KIVI 等稀疏注意力基线对比，RaBitQCache 在 LongBench、RULER、GSM8K 上的生成质量与召回率均与全精度相当甚至更优，单模型系统级加速 2.16×，解码阶段加速 3.88×，并通过自适应 Top‑p 取代固定 Top‑k，显著降低 KV 预算。

**⚠️ 局限性**

局限性：依赖向量中心化假设（向量近似均匀分布）；在极端聚类或分布漂移场景下误差上界可能不再成立；量化误差在极致精度任务中可能导致信息损失；实现高度依赖 GPU/CUDA 环境，迁移到其他平台需额外工程。

---

## 397. Design and Implementation of Agentic Orchestrations and Orchestration of Agents

**arXiv ID:** 2606.31518 | [PDF](https://arxiv.org/pdf/2606.31518v1)

**作者:** Stefanie Rinderle-Ma `[一作]` (Technical University of Munich), Matthias Ehrendorfer `[通讯]` (Technical University of Munich)

**通讯引用:** 90 | [OpenAlex ID](https://openalex.org/A5068452240)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了四种 AI 代理编排方案（OO1–OO4）的分类框架，并在光感测与血液捐献两大场景中实现、对比了这些方案；同时引入软件复杂度指标（Cyclomatic、ABC）与正确性/可追溯性指标，对不同编排方式进行定量评估。

**💡 创新点**

①将代理的自主性、任务细粒度、可追溯性等属性组合为统一分类；②利用软件复杂度指标评估任务细粒度；③在同一实验场景下系统比较四种编排方案，展示框架化方案在正确性与可追溯性上的优势。

**🔧 技术方法**

使用 LLM 代理（GPT‑4、Claude 等）通过 MCP/Agent‑to‑Agent 协议与 REST 接口进行交互；BPMN 流程模型用于描述过程框架；实验代码托管在 GitHub；指标计算基于 Cyclomatic、ABC、Precision/Recall/F1、FNR、R_s 等。

**📊 数据集**

构造光照测量数据集（模拟传感器事件、占用与移动状态），以及血液指南规则；未使用公开公开数据集。

**📈 对比分析**

通过任务细粒度指标（M、ABC）、正确性指标（P、R、F1）以及反应速度指标（FNR、R_s）对 OO1–OO6 进行评估。结果显示，框架化方案（OO3/OO4）在正确性和可追溯性上明显优于无框架方案，且任务细粒度指标能有效区分不同编排方式。

**⚠️ 局限性**

实验仅针对单一光感测场景，未充分考虑提示工程、模型负载、网络延迟等外部因素；对多代理通信、长周期数据流、时序约束等问题的处理仍缺乏系统研究。

---

## 398. Unsupervised Data-Efficient Cross-Modal Retrieval with Global-Neighborhood Alignment Hashing

**arXiv ID:** 2606.31517 | [PDF](https://arxiv.org/pdf/2606.31517v1)

**作者:** Runhao Li `[一作]` (Nanyang Technological University), Yap-Peng Tan `[通讯]` (VinUniversity)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5126144340)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无监督且数据高效的跨模态检索方法 GNAH，能够在仅有少量图文对的情况下学习有效的哈希码。

**💡 创新点**

创新点包括：①使用 Prototype‑Anchored Global Alignment（PAGA）将 CLIP 的全局语义结构映射到哈希空间；②引入 Contrastive Stochastic Neighborhood Alignment（CSNA）通过随机邻域对齐避免对稀疏样本的过拟合；③通过可学习的二进制原型实现更鲁棒的结构保持。

**🔧 技术方法**

核心技术包括 CLIP 作为特征提取器、K‑means 原型生成、softmax 关系对齐、KL 与对比损失、二进制化损失以及指数衰减的课程学习策略。

**📊 数据集**

在 MIR Flickr、Pascal Sentence 和 NUS‑WIDE 三个公开数据集上进行实验，分别采用 20、40、80、160 对图文样本进行训练。

**📈 对比分析**

与五种无监督 CMH 基线（CIRH、DSAH、UCCH、CAGAN、CFRH）以及两种有监督方法（DNPH、DSPH）对比，GNAH 在 16/32 位哈希下均取得最高或相近 mAP，尤其在 80 对样本时在 MIR Flickr 上比最近方法提升约 4–5% 以上，在 Pascal Sentence 上提升超过 10%。

**⚠️ 局限性**

局限性：模型依赖 CLIP 的预训练特征，对极低数据量（如仅 20 对）仍存在性能下降；同时 PAGA 的全局对齐在某些场景下可能过度影响局部细节，需通过 β、γ 等超参数进行精细调优。

---

## 399. FLARE-AI: Flaw Reporting for AI

**arXiv ID:** 2606.31567 | [PDF](https://arxiv.org/pdf/2606.31567v1)

**作者:** Shayne Longpre `[一作]` (Massachusetts Institute of Technology), Alex Pentland `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 95086 | [OpenAlex ID](https://openalex.org/A5007176508)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并公开了一个可扩展的 AI 漏洞报告系统 FLARE‑AI，并通过对 12 个现有报告系统的调查和 49 位专家的访谈，提出并解决了报告生态的五大痛点。

**💡 创新点**

提出统一的、可互操作的报告流程，利用早期分类与条件逻辑减轻报告者负担，并支持一次性生成可机器读取的 JSON‑LD 报告，实现多方自动传播；同时通过开放源码实现灵活演进。

**🔧 技术方法**

前端交互（React/Next.js）、后端无状态服务、条件渲染、JSON‑LD 语义化、REST/SMTP API 与 CERT/MITRE/AIID 等机构的集成；采用 OWASP 与 AIAAIC 公开分类法。

**📊 数据集**

基于对 12 种现有报告表单的系统化审计以及 49 位跨机构专家的访谈所收集的需求和反馈数据；未使用传统机器学习数据集。

**📈 对比分析**

通过对 12 个系统的功能指标（范围、分类法、字段数、共享与协调等）进行量化对比，发现 FLARE 在可发现性、信息完整性、互操作性和自动传播上优于现有方案；尚未给出客观指标，仅有专家满意度与初步用户测试提示易用性提升。

**⚠️ 局限性**

研究侧重北美欧盟组织，缺乏低资源地区视角；仍需平衡信息完整度与可用性；中心化报告可能带来安全与隐私风险；标准化的分类法仍受主观影响，实施难度对小型开发者较大。

---

## 400. DualBrep: A Dual-Field Continuous Representation for B-rep Modelling

**arXiv ID:** 2606.31579 | [PDF](https://arxiv.org/pdf/2606.31579v1)

**作者:** Yilin Liu `[一作]` (Autodesk Research), Hooman Shayani `[通讯]` (Autodesk Research)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种连续双场（SDF + GVD‑derived UDF）来统一B‑rep几何与拓扑，并通过共享潜在空间实现逆向工程与条件生成

**💡 创新点**

创新点在于把离散B‑rep图谱转化为可微分的连续域，利用GVD UDF实现拓扑分割，避免了离散图的梯度不可导与序列误差累积

**🔧 技术方法**

采用变分自编码器（Perceiver‑style encoder/decoder）压缩双场为共享潜向量，使用流匹配（Diffusion Transformer）在潜在空间生成B‑rep，配合神经重建器将连续场还原为显式B‑rep

**📊 数据集**

在公开ABC数据集（约80k CAD模型）上进行训练与评估，过滤掉极简/极复杂模型，采用4k模型做测试

**📈 对比分析**

与SEDNet+Point2CAD、NVDNet、HoLa‑BRep等基线对比，实验显示在几何精度、原始面数匹配与有效率上均优于或相当，尤其在复杂多面模型中保持高达76%以上的有效率

**⚠️ 局限性**

局限性包括受体素分辨率限制，极细薄结构易被忽略；重建器在复杂交叉处偶尔无法完美拼接，导致小范围不闭合。

---

## 401. Introduction to Stochastic Differential Equations for Generative Machine Learning: A Variational Perspective

**arXiv ID:** 2606.31576 | [PDF](https://arxiv.org/pdf/2606.31576v1)

**作者:** Ole Winther `[一作]` (University of Copenhagen), Andrea Dittadi `[通讯]` (Technical University of Munich)

**通讯引用:** 303 | [OpenAlex ID](https://openalex.org/A5008956185)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提供了一套自包含、非正式的生成式模型方法论，系统地推导了ODE、SDE、Fokker‑Planck方程和变分下界（ELBO），并统一展示了扩散、得分匹配与流匹配等主流方法在这一框架下的等价性。

**💡 创新点**

创新点在于：①用最小的数学假设从差分展开到连续式Fokker‑Planck；②将ODE、SDE及其逆时程映射到相同边缘分布的通用变分框架；③提出了可模拟自由的ELBO，利用可解析的变分边缘分布（如正态化流）实现不需要路径模拟的训练；④给出了如何将正态化流与混合流作为变分分布的具体漂移公式。

**🔧 技术方法**

使用技术包括：神经ODE/神经SDE（漂移、扩散参数化）、Euler‑Maruyama、Runge‑Kutta数值积分、变分推断、KL散度展开、Fokker‑Planck推导、得分匹配与流匹配损失、正态化流和混合流等。

**📊 数据集**

主要在一维 Laplace 混合分布（k=5，位置−2(k−1)~2(k−1)）上进行实验；未使用公开图像/视频等大型数据集。

**📈 对比分析**

实验比较了：①ODE（变分后向解）；②流匹配（SDE+正态化流变分）；③SDE（VDM 和自由变分）三种设置，使用相同数据集、相同网络结构并评估对数似然（或 ELBO）值。结果显示三种方法在该一维任务上能逼近真分布，得分/流匹配与ODE在训练后获得相近的 log‑likelihood，VDM 由于学习速率较慢略逊一筹。

**⚠️ 局限性**

局限性包括：①仅在单个观测的“生成式”场景下验证，未探讨多时序/多观测的变分过程；②实验仅限于一维 toy 数据，缺乏对复杂真实数据集的验证；③ELBO 与真实对数似然之间存在不可避免的下界误差；④高阶数值求解（如 Runge‑Kutta）对效率和梯度传播要求较高；⑤未涵盖最新的采样加速技术或正则化策略。

---

## 402. Which Tokens Matter? Adaptive Token Selection for RLVR with the Relative Surprisal Index

**arXiv ID:** 2606.31575 | [PDF](https://arxiv.org/pdf/2606.31575v1)

**作者:** Outongyi Lv `[一作]` (ModelScope Team), Yingda Chen `[通讯]` (ModelScope Team)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了相对惊讶指数（RSI）及其基于区间的筛选方法RSI‑S，用以在强化学习可验证奖励（RLVR）框架下挑选对大语言模型（LLM）最有益的token，从而提升数学推理性能。

**💡 创新点**

创新点在于将token的熵与采样token的概率相结合，构造信息论指标RSI，并通过RSI‑S实现对高熵低概率和低熵高概率token的统一平衡；同时提供了与梯度范数关系的理论解释，解决了之前高熵和低概率之间的矛盾。

**🔧 技术方法**

使用的技术包括RLVR、GRPO优化框架、RSI指标、区间筛选机制RSI‑S、Jensen‑Shannon散度评估、梯度范数分析、以及对不同模型规模的多轮实验。

**📊 数据集**

训练数据为 DAPO‑MATH‑17K；评估数据为 AIME（2024–2026）和 AMC（2022–2024）数学竞赛题集，使用 Qwen2.5‑1.5B、3B、7B 三个基础 LLM 进行实验。

**📈 对比分析**

与基线 GRPO、80/20 熵筛选（EB）和概率筛选（PB）对比，RSI‑S 在 avg@32 指标上分别比 GRPO 提升 2–3%，并显著缩短平均生成长度；在三种模型规模上均保持稳健提升，且对阈值敏感度更低。

**⚠️ 局限性**

局限性包括需针对不同模型规模手动调节 RSI 区间超参数，且方法主要在数学推理任务上验证，尚未充分检验在其它下游任务或更大规模 LLM 上的泛化与计算开销。

---

## 403. FormIDEAble: Safe and Socially-aware Autonomous Systems

**arXiv ID:** 2606.31572 | [PDF](https://arxiv.org/pdf/2606.31572v1)

**作者:** Livia Lestingi `[一作]` (Politecnico di Milano), Matteo Rossi `[通讯]` (Politecnico di Milano)

**通讯引用:** 1771 | [OpenAlex ID](https://openalex.org/A5045104020)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一种基于Priced Timed Markov Decision Process (PTMDP) 的策略合成框架（即Safe and Socially‑aware Autonomous Agents, “SIA”），实现在人类行为不确定且存在时间与成本约束的社会关键场景中，自动生成既满足安全属性又具备社交意识的决策策略。

**💡 创新点**

创新点在于：①将人机协作建模为 PTMDP，既捕捉不确定的人类响应、时序约束与成本；②将安全需求形式化为成本受限的可达性问题，在策略合成过程中直接嵌入安全约束，从而得到满足安全概率阈值的最优策略；③提出可重用的 PTMDP 模式和 MAPE‑K‑K 框架，便于在不同社会关键场景中快速建模和部署。

**🔧 技术方法**

使用的技术包括：PTMDP 模型构造、基于 PRISM 的成本受限可达性合成、蒙特卡洛模拟 (SMC) 用于概率估计、身份预测器 (Identity Predictor) 用于估计人类社会身份对合作意愿的影响，以及 MAPE‑K 运行时自适应控制。

**📊 数据集**

数据集主要来自两部分：①在 IMPACT+ 事件模拟器中生成的 100 次模拟记录，用于训练身份预测器；②实验中使用的仿真参数（受试者数、救援者数、最大等待时间等）作为实验配置。

**📈 对比分析**

比较方法：与无助力、始终呼叫专业救援、始终呼叫旁观者以及 IDEA 框架的基线策略进行对比。性能指标包括：疏散总时长、专业救援者调用次数、安全约束满足率、策略合成时间。实验结果表明，SIA 在疏散时长上显著优于无助力基线，在专业救援者调用次数上与 staff‑support 基线相近或更低，并且在满足安全约束方面优于 IDEA，合成时间在单人决策下低于 0.5 s，最多 2.25 s。

**⚠️ 局限性**

局限性包括：①模型构造与安全属性定义仍需人工干预，缺乏高层抽象语言或自动化工具；②实验仅在仿真环境中验证，缺少真实人类行为数据；③规模受限，较大人机协作场景的可扩展性待进一步优化；④安全保障基于模型假设，若人类行为偏离假设则可能失效。

---

## 404. Latency-Sensitive 5G RAN Slicing for Deterministic Aperiodic Traffic in Smart Manufacturing

**arXiv ID:** 2606.31499 | [PDF](https://arxiv.org/pdf/2606.31499v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 405. ACE: Pluggable Adaptive Context Elasticizer across Agents

**arXiv ID:** 2606.31564 | [PDF](https://arxiv.org/pdf/2606.31564v1)

**作者:** Ning Liao `[一作]` (Meituan), Junchi Yan `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 17716 | [OpenAlex ID](https://openalex.org/A5087158377)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Adaptive Context Elasticizer（ACE）模块，分离原始信息与实际上下文，按任务状态动态决定每步历史信息的原始、摘要或丢弃，以提升大型语言模型代理在长轨迹任务中的信息密度和推理质量。

**💡 创新点**

创新点包括：
1) 解耦原始消息与上下文的两层架构，实现信息的可逆弹性；
2) 通过LLM驱动的 Elasticizer 在每一步动态分配三种弹性类型，实时优化上下文内容；
3) 以外部包装器方式实现无训练、无改动框架的插件化集成，兼容多种代理结构。

**🔧 技术方法**

使用技术：
- 主要LLM（GPT‑4.1 或 Gemini‑3.1‑flash‑lite‑preview）负责推理与决策；
- 辅助LLM（GPT‑4o）生成每步摘要；
- 两层架构（消息维护层 + 上下文编排层）实现可逆存储与弹性上下文；
- 通过外部包装器将ACE插入 ReAct、DeepAgent、WebThinker、MiroFlow 等框架。

**📊 数据集**

数据集：
- GAIA、HLE、WebShop、WebWalkerQA、BrowseComp‑ZH、xBench‑DS 等多任务基准，用于评估不同框架和语言模型下的表现。

**📈 对比分析**

比较方法：在 ReAct 框架下与三种基线（无管理、截断、摘要）对比，ACE 在 GPT‑4.1 和 Gemini‑3.1‑flash‑lite‑preview 两种主模型上均取得最高得分；在 DeepAgent、WebThinker、MiroFlow 等框架中，同样通过 ACE 替换原有记忆机制，实验显示 ACE 在所有基准上均提升 3–10 分，显著优于传统截断和摘要。

**⚠️ 局限性**

局限性：
- 依赖外部 LLM 生成摘要，可能导致额外推理成本和延迟；
- 对高并发多代理场景的实时性能尚未评估；
- 仅在文本/多模态输入上验证，无法证明在极长或极高维度上下文的可扩展性；
- 对于极端任务需求（如严格实时性）仍需进一步优化。

---

## 406. AugSplat: Radiance Field-Informed Gaussian Splatting for Sparse-View Settings

**arXiv ID:** 2606.31556 | [PDF](https://arxiv.org/pdf/2606.31556v1)

**作者:** Lorenzo Lazzaroni `[一作]` (ETH Zurich), Keisuke Tateno `[通讯]` (Google)

**通讯引用:** 1784 | [OpenAlex ID](https://openalex.org/A5030061164)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过训练神经辐射场（NeRF）并利用其渲染的合成视图来增强稀视角场景的高质量实时视图合成，提出了AugSplat框架；

**💡 创新点**

创新点在于将NeRF作为数据生成器，为高效的3D高斯分裂（3D Gaussian Splatting）提供额外的合成视图监督，并提出了两种训练策略（Staged AugSplat 与 Dual AugSplat），在稀视角环境下显著提升几何与外观重建质量；

**🔧 技术方法**

采用NeRF（depth-nerfacto）训练集成模型，生成合成图像及不确定性估计；利用3D Gaussian Splatting进行渲染；对合成视图加权监督与真实视图联合优化；

**📊 数据集**

使用 mip-NeRF 360 数据集，挑选每个场景30张图像（26张训练，4张验证）进行稀视角实验；

**📈 对比分析**

与标准3D Gaussian Splatting（gsplat）基线对比，采用 PSNR、SSIM、LPIPS 以及平均误差等指标评估；结果表明 Staged AugSplat 在平均误差上提升约 4%（PSNR +0.5 dB，SSIM +0.008），Dual AugSplat 也保持竞争力，均在稀视角设置下优于基线；

**⚠️ 局限性**

局限性包括：需额外训练NeRF，增加训练成本；若NeRF先验质量差，合成视图可能误导优化；Staged 方案在某些场景中对合成监督依赖过大导致性能波动，Dual 方案对权重衰减调节要求更高。

---

## 407. AutoTrainess: Teaching Language Models to Improve Language Models Autonomously

**arXiv ID:** 2606.31551 | [PDF](https://arxiv.org/pdf/2606.31551v1)

**作者:** Zhaojian Yu `[一作]`, Xiao-Ping Zhang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

AutoTrainess 是一种将训练经验外部化为 Agent-Computer Interface 的 LM 训练代理，自动完成规划、数据准备、训练与评估，实现端到端 LLM 后训练。

**💡 创新点**

创新点在于将人类训练经验包装成可复用的接口集合（AutoTrainHub），通过结构化工作流替代传统 CLI，显著提升代理自主训练的可靠性和效果。

**🔧 技术方法**

使用 GPT-5.4 (Codex) 或 DeepSeek-V4-Flash 作为语言模型骨干，依赖 LlamaFactory 后端训练，设计了数据、训练、评估、日志与规划四类接口。

**📊 数据集**

使用 PostTrainBench（包含 AIME 2025、ArenaHard、BFCL、GPQA、GSM8K、HealthBench、HumanEval 等七个基准）和其对应的四个基础模型 Qwen3-1.7B/4B、SmolLM3-3B、Gemma-3-4B。

**📈 对比分析**

在 PostTrainBench 上与仅 CLI 的基线对比，AutoTrainess 的平均分从 23.21 提升到 26.94（相对提升 15%），在 Qwen3-4B 子集达到 32.60；不同模型间也保持一致性。

**⚠️ 局限性**

局限在于仍依赖预先定义的接口集合，无法自发发现全新训练策略，且在极短时间预算下对数据质量与超参探索仍受限。

---

## 408. Governance Gaps in Agent Interoperability Protocols: What MCP, A2A, and ACP Cannot Express

**arXiv ID:** 2606.31498 | [PDF](https://arxiv.org/pdf/2606.31498v1)

**作者:** Richard Kang `[一作]` (DoiT International), Yudho Diponegoro `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对五种主要代理互操作协议（MCP、A2A、ACP、ANP、ERC‑8004）进行系统性治理缺口分析，构建六维治理需求分类法并生成缺口矩阵，指出这些协议在治理层面（成员资格、论证、投票、异议保留、人类升级、审计）上的不足，提出治理为缺失的架构层而非协议功能。

**💡 创新点**

提出专门针对代理治理的六维分类法，并将其与现有协议规范直接对应；首次量化评估协议在治理维度上的支持程度，区分可通过扩展修复的“可扩展缺口”与需要新架构层的“结构缺口”，并对缺口闭合速度进行时间敏感性分析。

**🔧 技术方法**

协议规范分析、组织理论与多智能体系统文献综合、治理维度的语义化描述、缺口矩阵构建、覆盖率评分与可扩展性/时间敏感性评估。

**📊 数据集**

无实验数据集；使用截至2026年6月的五种协议（MCP v1.1、A2A v1.0.1、ACP、ANP、ERC‑8004）的官方规范和公开扩展说明作为分析材料。

**📈 对比分析**

通过对每个协议‑治理维度对进行“支持/部分/缺失”三档分类，累计得分构成覆盖率，结果显示投票、异议保留和人类升级在所有协议中均为缺失；部分协议在成员资格和论证维度有可扩展实现，但总体缺口高。

**⚠️ 局限性**

仅基于协议规范进行评估，未考虑实现层实现；分类标准依赖主观判断；分类法以西方组织理论为基础，可能不适用于其他治理传统；未考察协议在实际部署环境中的治理效果。

---

## 409. HVPNet: A Bio-Inspired Network for General Salient and Camouflaged Object Detection

**arXiv ID:** 2606.31496 | [PDF](https://arxiv.org/pdf/2606.31496v1)

**作者:** Jiawei Xu `[一作]` (Jiangxi Normal University), Jiacong Yu `[通讯]` (Jiangxi Normal University)

**通讯引用:** 187 | [OpenAlex ID](https://openalex.org/A5111245978)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对多模态显著目标检测（SOD）和伪装目标检测（COD），本文提出了一种轻量化的生物启发式网络 HVPNet，能够在七个任务、22 个公开数据集上实现高精度检测。

**💡 创新点**

创新点在于引入视网膜层级的级别特定融合（Retinal Integration Module）和大脑皮层分层解码（Cortical Decoder），通过级别化、轻量化的交叉模态融合与分层解码，显著提升了模型的效率与准确性。

**🔧 技术方法**

核心技术包括轻量级骨干（SMT‑t、MobileNetV2）、多阶段跨模态融合（RIM）与选择性区域注意（SRA）、高斯引导注意（GGA）、以及结构化的低级/高级视觉解码器（LLVD / HLVD）。

**📊 数据集**

实验数据集涵盖 22 组公开数据，分别用于 RGB‑SOD（DUTS、DUT‑O、HKU‑IS）、RGB‑D‑SOD（NJUD、NLPR、STERE）、RGB‑T‑SOD（VT821、VT1000、VT5000）、VSOD（FBMS、SegV2）、RGB‑COD（COD10K、CAMO、NC4K）、RGB‑D‑COD（COD10K、CAMO）以及 VCOD（CAD）。

**📈 对比分析**

与当前 SOTA 方法进行对比后，HVPNet 在所有任务中均取得了最佳或接近最佳的 E_m、S_m、F_m 指标，同时参数量与 FLOPs 减少约 90%，证明了其优秀的精度‑效率折中。

**⚠️ 局限性**

局限性包括：当 RGB 与辅助模态均误判同一区域时易产生假阳性；对低质量、模糊或透明目标的识别仍然困难；在极端遮蔽或背景复杂的场景下，边界感知和细节重建效果有限。

---

## 410. Robustness of Robotic Manipulation: Foundations and Frontiers

**arXiv ID:** 2606.31494 | [PDF](https://arxiv.org/pdf/2606.31494v1)

**作者:** Yifei Dong `[一作]` (Duke University), Xianyi Cheng `[通讯]` (Duke University)

**通讯引用:** 274 | [OpenAlex ID](https://openalex.org/A5100943262)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性综述并提出统一定义、公式与框架，梳理感知、规划、控制、学习和硬件等子系统的鲁棒性机制与原则。

**💡 创新点**

首次将鲁棒性从概率（POMDP）与控制理论（H∞/MPC）双重视角统一建模，并提出两轴鲁棒性原则（不确定性/变异调节与失败管理）和跨子系统的整体性分析。

**🔧 技术方法**

综合运用了概率建模、控制理论、感知交互、规划与推理、主动/被动顺应、生成式与层级策略、逆向学习与在线适应、以及软硬件协同设计等多种技术。

**📊 数据集**

主要引用多种实验与仿真数据集（抓取、推挤、插槽、真实机器人任务等），未聚焦单一公开数据集，而是通过多任务、多环境的实证验证。

**📈 对比分析**

通过任务成功率、阶段级评估、闭合度、裕度、Signal Temporal Logic鲁棒性等指标进行比较，表明大多数方法在仿真或受限现实环境中可提升鲁棒性，但整体性能仍远低于人类与动物的水平。

**⚠️ 局限性**

缺乏统一、公开的鲁棒性评估基准与标准，方法在真实世界的转移和可推广性不足，鲁棒性与效率、精度之间的权衡尚未得到系统解决。

---

## 411. DrivingDepth: Sparse-Prompted Pixel-wise Scale Correction for Driving Depth Estimation

**arXiv ID:** 2606.31488 | [PDF](https://arxiv.org/pdf/2606.31488v1)

**作者:** Chi Huang `[一作]` (Baidu Inc), Liang Wang `[通讯]` (Baidu Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于冻结的视觉基础模型的稀疏激活深度估计框架 DrivingDepth，通过稀疏激光点的像素级比例校正实现精确的三维深度。

**💡 创新点**

核心创新是只学习像素级尺度修正而不重训练基础模型，保持其几何一致性；同时引入几何保持特征适配器和基于置信度的稀疏尺度头，解决几何-尺度冲突。

**🔧 技术方法**

使用的技术包括深度视觉基础模型 DepthAnything3、交叉注意力与约束的几何保持特征适配器、稀疏多分辨率激光提示、稀疏感知像素尺度头、置信度加权损失与表面法线正则化。

**📊 数据集**

实验基于 nuScenes 和 DDAD 两个自动驾驶数据集。

**📈 对比分析**

与 DepthAnything3、MapAnything、PromptDA、PriorDA 等方法对比，DrivingDepth 在 nuScenes 上 AbsRel 11.19、EdgeCR 5.741、AbsIn02 61.14，显著优于 MapAnything，并且在低点云密度下仍保持鲁棒性。

**⚠️ 局限性**

局限在于依赖冻结的基础模型几何；若基础模型在反射或透明表面上产生误差，尺度校正无法弥补缺失的几何结构。

---

## 412. Robustness of neural networks to random noise perturbations of their inputs

**arXiv ID:** 2606.31581 | [PDF](https://arxiv.org/pdf/2606.31581v1)

**作者:** Mark Levene `[一作]` (Birkbeck University of London), Martyn Harris `[通讯]` (Birkbeck University of London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究训练好的神经网络在输入加高斯噪声扰动时的鲁棒性，提出θ‑鲁棒度指标及其对应的MSEθ，并通过蒙特卡罗方法估计MSE的上界；

**💡 创新点**

创新点在于以黑盒方式给出可计算的鲁棒度指标，构造鲁棒曲线并用Gompertz函数拟合，进而定义曲线比较的鲁棒指数及整体鲁棒指数，用以系统比较不同数据集或模型的鲁棒性；

**🔧 技术方法**

主要技术包括蒙特卡罗采样估计MSE、改进的MSE（或Brier分数）用于评估、Gompertz曲线拟合、Knee点/Inflection点提取、鲁棒指数与整体鲁棒指数计算；

**📊 数据集**

使用的真实数据集有：癌症死亡率、2020年流感疫苗接种率、Cleveland心脏病、Wisconsin乳腺癌、MNIST手写数字；

**📈 对比分析**

通过构造鲁棒曲线并对比其归一化后的形状，利用整体鲁棒指数进行比较，实验显示：癌症死亡率最鲁棒，其次是流感疫苗接种率、乳腺癌、心脏病，MNIST最不鲁棒；

**⚠️ 局限性**

局限性包括仅评估前馈神经网络、只考虑高斯噪声扰动、样本量有限、未验证其他模型或更复杂网络、对噪声规模的选取仍需经验指导。

---

## 413. Holonic Active Distillation for Scalable Multi-Agent Learning in Multi-Sensor Systems

**arXiv ID:** 2606.31578 | [PDF](https://arxiv.org/pdf/2606.31578v1)

**作者:** Dani Manjah `[一作]` (UCLouvain), Stéphane Galland `[通讯]` (Université de Technologie de Belfort Montbéliard)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `e0540dec-d77f-42db-94ae-d039248f6393` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 Holonic Active Distillation (HAD) 架构，支持多传感器系统的可扩展、自适应学习，能够在线集成和剔除传感器并进行知识迁移。

**💡 创新点**

创新点在于将 Holonic 学习与 Active Distillation 相结合，构建多层次自组织的 holarchy，提供标准化协议、上下层模型聚合与动态分层融合，实现无缝新增/删除传感器且不需全局重训练。

**🔧 技术方法**

使用的技术包括 Holonic 组织模型、Teacher-Student 伪标签蒸馏、YOLOv8 网络、层级聚类（单链接）与增量式模型更新。

**📊 数据集**

实验采用两个城市视频数据集：WALT（9 摄像头）和 AI‑City（7 摄像头）共 16 摄像头。

**📈 对比分析**

对比方法：与全局 YOLOv8x6 教师+COCO 预训练基线、单独训练模型、不同预算层级的 holon 化；结果显示 mAP50‑95 在 0.66‑0.67 之间，增量集成不降性能，知识迁移比 COCO 显著提升。

**⚠️ 局限性**

局限性包括对模型漂移与长周期适应缺乏监控、对大规模集成的上限未确定、以及多模态/物理模型融合的进一步研究需求。

---

## 414. Localized Conformal Prediction for Image Classification with Vision-Language Models

**arXiv ID:** 2606.31577 | [PDF](https://arxiv.org/pdf/2606.31577v1)

**作者:** Clément Fuchs `[一作]` (UCLouvain), Benoît Macq `[通讯]` (UCLouvain)

**通讯引用:** 7821 | [OpenAlex ID](https://openalex.org/A5015968689)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对视觉‑语言模型（VLM）下的图像分类任务实现并基准化了本地化自适应的 conformal prediction（LCP）

**💡 创新点**

提出并验证了对余弦相似度的 sigmoid 非线性变换，以改进 LCP 的权重分配，从而显著降低平均预测集大小

**🔧 技术方法**

使用 LCP 以及 LAC、APS、RAPS 等 conformal 分数，结合 CLIP 视觉‑语言模型（ViT‑B/16、ViT‑L/14、RN50、RN101）和余弦相似度权重

**📊 数据集**

在 9 个公开图像分类数据集（SUN397、Aircraft、EuroSAT、StanfordCars、Food101、Pets、Flower102、DTD、UCF101）上进行评估

**📈 对比分析**

与传统非本地 CP（如 TopK、APS、RAPS）比较，改进方案在所有数据集上平均集大小均显著下降，统计意义良好，但在覆盖度指标（CovGap、MCCC）提升有限

**⚠️ 局限性**

局限在于覆盖度改进不明显，且方法对不同分数和 backbone 的鲁棒性及泛化性仍需进一步研究

---

## 415. Configured Grant Scheduling for the Support of TSN Traffic in 5G and Beyond Industrial Networks

**arXiv ID:** 2606.31506 | [PDF](https://arxiv.org/pdf/2606.31506v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 416. MV-GEL: Language-Driven Multi-View Geometric Entity Localization on Meshes

**arXiv ID:** 2606.31533 | [PDF](https://arxiv.org/pdf/2606.31533v1)

**作者:** Kartik Bali `[一作]` (Helmholtz Zentrum Hereon), Roland Aydin `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了 MV-GEL 框架，利用多视角图像与自然语言查询在三维网格上定位几何实体（面、边）。

**💡 创新点**

创新点在于：① 提出视图先验模块 GELviews，基于语言提示对视角进行几何可观测性排序；② 将 LISA 分割模型专门针对 CAD 进行微调（LISA-CAD）；③ 将 2D 分割结果通过射线投影精确映射回网格，实现高精度实体定位。

**🔧 技术方法**

采用的技术包括：CLIP + LoRA 细调的视觉-语言编码、FiLM/Cross‑Attention 融合、Transformer 视角交互、基于射线投影的几何拉升、LISA-CAD 分割网络，以及 ABC CAD 数据集构建的 54k 语言-实体对。

**📊 数据集**

使用的数据集是 ABC CAD 数据集中的 54,000 条自然语言查询–几何实体对，涵盖 5,000+ CAD 机械部件。

**📈 对比分析**

方法通过与 CLIP 随机视角、LISA 原版、以及多种点云分割基线（PartSLIP、Find3D、PatchAlign3D）对比，Face IoU 提升 1.7×、Edge F1 提升 4.5×，在 1,535 个测试查询上实现高精度且低延迟。

**⚠️ 局限性**

限制包括：对极细结构的可辨识仍受限于 VLM 的分辨率；仅适用于网格形式的 CAD，非 B‑Rep 数据需要额外处理；视角多样性和分布仍可进一步优化；模型训练和推理对 GPU 资源要求较高。

---

## 417. PRISM: Latent Composition Consistency for Single-Image Reflection Removal

**arXiv ID:** 2606.31513 | [PDF](https://arxiv.org/pdf/2606.31513v1)

**作者:** Junseong Shin `[一作]` (Hanyang University), Tae Hyun Kim `[通讯]` (Hanyang University)

**通讯引用:** 15445 | [OpenAlex ID](https://openalex.org/A5100438974)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出PRISM框架，通过在预训练VAE（FLUX）潜空间中将单图像反射去除转化为线性分离问题，单前向传播即可同时恢复透射层和反射层。

**💡 创新点**

创新点在于利用潜空间低互相关特性构造潜在加法假设，并通过Latent Composition Consistency（LCC）与Layer Contrastive Separation（LCS）双重正则化，避免传统像素空间方法的非线性混叠与双分支设计。

**🔧 技术方法**

核心技术包括：VAE潜空间编码/解码、流匹配（Flow Matching）学习速度场、FLUX大型预训练模型微调、对抗式对比学习（InfoNCE）实现层级对比正则化，以及循环一致性损失。

**📊 数据集**

训练数据为PASCAL VOC合成对（7643对）、公开真实对（90对）和Nature真实对（200对）；评估使用Real、Object、Postcard、Wild、Nature、SIR²六大基准以及OpenRR 1K。

**📈 对比分析**

与10种SIRR方法对比，PRISM在PSNR上均领先，尤其在Real、Wild和OpenRR 1K上分别提升约1–2 dB；在感知指标LPIPS、DISTS上亦显著优于基线，表明更佳的图像质量与泛化能力。

**⚠️ 局限性**

局限在于使用的FLUX 3.97B参数模型显著占用显存，限制低资源设备部署；当反射强度过高时，潜空间低相关假设失效，导致残留反射。

---

## 418. Falsification, Not Exposure: An Internally Preregistered Placebo-Controlled Decomposition of Self-Repair Feedback in Frozen Small Code Models

**arXiv ID:** 2606.31511 | [PDF](https://arxiv.org/pdf/2606.31511v1)

**作者:** Mehmet Iscan `[一作]` `[通讯]` (Yildiz Technical University), Mehmet Iscan (Yildiz Technical University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在不允许再训练的小型冻结代码模型上，系统地评估了不同自修复反馈包的成分（仅失败代码、无反馈、仅执行事实、代码+事实、代码+占位文本）对程序修复的影响，并通过配对盲重采样、置换对照、一次性生成等设计构建了可重复、可审计的实验流程。

**💡 创新点**

创新点主要体现在：① 通过“占位符镜像”与“盲重采样”对照，解构反馈包内部成分，提供可验证的因果归因；② 采用匹配输出预算、同单元互斥对照、预注册统计族与可执行审计，构建了完整的可反驳（falsification）测量框架；③ 将 Popperian 反驳理念嵌入实验设计，使得模型输出与研究者的反馈效能主张同样可被外部审计。

**🔧 技术方法**

技术方法包括：5 组对照实验（bare, blind, facts, code+facts, code+shape‑placeholder）；每组 4 轮生成，使用 1,024 token 输出、温度 0.8；严格的 SHA‑256 生成与提示镜像审计；基于精确 McNemar 检验和 Holm 校正的统计检验；对结果的实时审计（重复输出、种子唯一性、占位符一致性）。

**📊 数据集**

使用 HumanEval+ 与 MBPP+ 两个公开编程基准，共计 512 个任务单元，随机抽取 290 个“死”单元（缓存 pool 中无通过实例）和 60 个独立敏感性样本；模型为三款 0.5B–1.5B 规模的 Ollama 冻结代码模型（deepseek‑coder‑1.3b、qwen‑2.5‑coder‑0.5b、qwen‑2.5‑coder‑1.5b）。

**📈 对比分析**

相较于盲重采样，bare 代码重试表现最差（+8 解锁），仅事实  +15，代码+事实  +18，且与盲重采样在 26 个解锁上持平；占位符文本未显著提升。整体解锁率最高 9.0%（blind / code+facts），最低 2.8%（bare）。统计检验显示，blind 与 code+facts 均显著优于裸码，且代码+事实对比占位符显著提升（p≈0.004）。但在所有单元中并未出现 code+facts 超越 blind 的情况，说明自修复反馈未能在本设置下提供比简单重采样更高的平均收益。

**⚠️ 局限性**

局限性包括：仅评估 0.5–1.5B 规模冻结模型，且仅在两大基准与单一提示模板下；未对模型内部生成机制进行分析，无法排除提示长度或词法分布对结果的影响；对“真实”未通过单元的统计推断仍有限，缺乏对更大规模模型或不同编程语言的泛化验证；审计过程中发现的重用与占位符问题提示实验设计在极端样本下的稳健性仍需进一步验证。

---

## 419. Surprise as a Signal for Plasticity and Metacognition

**arXiv ID:** 2606.31495 | [PDF](https://arxiv.org/pdf/2606.31495v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 420. Digital Innovation through Knowledge Processes

**arXiv ID:** 2606.31505 | [PDF](https://arxiv.org/pdf/2606.31505v1)

**作者:** Nataliia Klievtsova `[一作]` (Technical University of Munich), Stefanie Rinderle-Ma `[通讯]` (Technical University of Munich)

**通讯引用:** 6133 | [OpenAlex ID](https://openalex.org/A5071368904)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

对知识密集流程进行概念化定义，提出基于输入/输出的知识强度分类，并从一个企业创意流程案例中提炼了31种知识流程模式。

**💡 创新点**

创新点在于：① 将知识管理与业务流程管理结合，提出统一的知识流程定义与分类框架；② 将知识资产（知识、工件、输出）与流程要素映射，形成可操作的模式清单；③ 通过案例驱动的方法把抽象概念落地为可复用的流程模式。

**🔧 技术方法**

主要技术手段包括流程建模与分析（使用CPEE流程引擎）、知识资产抽象（Ki、Ai、Ko、Ao）、模式识别与归纳、文献综述与对照。

**📊 数据集**

使用了一个大型企业的“IP创意流程”作为案例数据，包含13个人工任务、11个机器任务和一个子流程；未公开公开的标准数据集。

**📈 对比分析**

论文没有采用实验对比或性能评估，所有结论均基于案例分析与模式归纳；未进行量化指标比较。

**⚠️ 局限性**

限制包括：① 仅验证单一案例，缺乏跨案例的普适性验证；② 方案未给出完整的实现与量化评估；③ 对自动化支持（如代理、LLM）的具体实现与性能尚不明晰。

---

## 421. SimpleSearch-VL: A Simple Recipe for Multimodal Agentic Deep Search

**arXiv ID:** 2606.31504 | [PDF](https://arxiv.org/pdf/2606.31504v1)

**作者:** Ming Dai `[一作]` (Southeast University), Chunhua Shen `[通讯]` (Ant Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 SimpleSearch-VL，一个多模态 agentic search 框架，结合了 Factorized Adaptive Rollout、可直接验证的视觉证据和网页自摘要技术，显著提升搜索效率、可靠性和可实践性。

**💡 创新点**

核心创新包括：① 信号感知的多层预算分配 FAR，动态扩展提示与分配 roll‑out；② 通过缩略图校验实现可直接验证的视觉证据；③ 将网页自摘要嵌入代理内部，消除外部摘要模型。

**🔧 技术方法**

使用技术包括：基于 Qwen3‑VL 的大规模视觉语言模型；强化学习（PPO）结合 SFT；工具接口（文本检索、反向图像搜索、网页访问）；链式思维验证；结果缓存机制。

**📊 数据集**

训练与评估数据集为：6 大多模搜索基准（MMSearch、MMSearch+、BrowseComp‑VL、FVQA、LiveVQA、SimpleVQA），以及 5K SFT 轨迹 + 2K RL 数据。

**📈 对比分析**

与同基础 Qwen3‑VL、OpenSearch‑VL、Gemini‑3‑Pro 等模型对比，8B 版本平均提升 15.8 分，30B‑A3B 平均提升 16.0 分，30B‑A3B 甚至可与 Gemini‑3‑Pro 竞争；训练成本仅 1 台 H200 GPU 约 18 小时，显著低于对标系统。

**⚠️ 局限性**

局限性包括：对极端长尾任务和复杂视觉场景的鲁棒性仍待验证；直接 RL 对工具利用的策略仍有提升空间；对视觉检索的依赖可能导致匹配误差；未对非常长文本网页的摘要质量做系统性评估。

---

## 422. Fully Automated High-Precision Segmentation of Retinal Atrophy and Ellipsoid Zone Thickness in OCT: A Reliable Tool for Real-World GA Monitoring

**arXiv ID:** 2606.31502 | [PDF](https://arxiv.org/pdf/2606.31502v1)

**作者:** Wolf-Dieter Vogl `[一作]`, Ariadne Whitby `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一套全自动深度学习框架，实现OCT图像中RPE缺失、EZ缺失与EZ变薄的像素级分割

**💡 创新点**

创新点在于三模型协同、使用多来源多病种训练，验证在真实世界外部数据，并加入B‑scan密度和重现性评估

**🔧 技术方法**

采用三种语义分割CNN（UNet++/DenseNet‑201/FPN），结合Bruch膜预处理、数据增强与混合损失函数

**📊 数据集**

训练集298体积（GA 222、iAMD 40、nAMD 17、健康 19），外部验证集43体积（VIBES），以及OAKS/DERBY 68对、B‑scan密度61/46对

**📈 对比分析**

与人工标注对比，Dice RPE 0.88、EZ 0.87，面积Pearson 0.999/0.996，ICC>0.98，B‑scan密度下误差<1%对大病灶；进展测量误差随B‑scan间距升高

**⚠️ 局限性**

局限在B‑scan间距与轴向分辨率低导致边界误差，图像质量控制不足，且对极小病灶的检测仍受限

---

## 423. ChronoFlow-Policy: Unifying Past-Current-Future Interaction Flow in Visuomotor Policy Learning

**arXiv ID:** 2606.31493 | [PDF](https://arxiv.org/pdf/2606.31493v1)

**作者:** Bokai Lin `[一作]` (Shanghai Jiao Tong University), Lixin Yang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了统一的过去-现在-未来交互流ChronoFlow表示并以此为监督训练视觉运动策略，提升机器人操控性能；

**💡 创新点**

首次提出交互中心的稀疏3D关键点流与联合流匹配的扩散式策略，兼顾历史与未来信息；

**🔧 技术方法**

采用3D点云编码器、交互关键点编码、条件扩散模型（Unet/DiT）、Transformer解码器以及深度学习监督；

**📊 数据集**

在MetaWorld、RoboTwin 2.0的14个仿真任务和5个真实世界抓取/装配任务上进行评测；

**📈 对比分析**

与DP3、RISE、HistRISE、3D-FDP、MBA等基线相比，ChronoFlow-Policy在仿真和真实任务上分别提升约32.5%/35.3%；

**⚠️ 局限性**

依赖关键点追踪与分割，易受遮挡、漂移和关键点缺失影响，需更鲁棒的交互跟踪方法。

---

## 424. Dynamic Ultrasound Beamforming Using Left-to-Right Arithmetic Adders on FPGA

**arXiv ID:** 2606.31490 | [PDF](https://arxiv.org/pdf/2606.31490v1)

**作者:** Muhammad Usman `[一作]` (University of Regensburg), Dorit Merhof `[通讯]` (University of Regensburg)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

设计并实现了一种基于左到右（MSDF）算术的加法树，用于64通道超声DAS beamforming，并在Xilinx Zynq XC7Z010 FPGA上评估其面积、功耗、时序和帧率。

**💡 创新点**

创新点在于将MSD‑first digit‑serial LRA单元组合成平衡二叉树，既实现了可动态精度（可在任意K位停止计算）又显著降低了资源占用，成为首个满足100 MHz时序且可多实例并行的精确加法树。

**🔧 技术方法**

使用的技术包括LR算术、MSD‑first编码、冗余签数字、shift‑and‑add累加器、数字串行流水线、Vivado综合与SAIF功耗分析，以及FPGA的iso‑area并行扩展。

**📊 数据集**

使用了8.5 MHz线阵超声RF数据，采自ATS‑549组织体phantom，包含122 304个像素（即244 608个实部加法操作）。

**📈 对比分析**

通过在同一FPGA、相同时钟约束下对比七种精确加法树（RCA、CLA、CSKA、CSLA、Brent–Kung、Kogge–Stone、Sklansky）和EvoApprox近似树，LRA64在面积（722 LUT）、功耗（10 mW）和帧率（单实例66.9 FPS、最大多实例≈1047 MSa/s）上均优于其它方案。

**⚠️ 局限性**

局限性包括仅在单通道、单帧级别验证动态精度，未探讨体积波束成形、后实现功耗以及像素级自适应精度控制；实验仅针对小型FPGA（XC7Z010）进行。

---

## 425. Temperature Field Reconstruction of Tungsten Monoblock Divertor on EAST using Physics-aware Neural Operator Transformer

**arXiv ID:** 2606.31574 | [PDF](https://arxiv.org/pdf/2606.31574v1)

**作者:** Zikang Yan `[一作]` (Anhui University), Guosheng Xu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种 Physics-aware Neural Operator Transformer (PNOT)，用于快速重建东风实验装置（EAST）钨单块分流器的瞬态温度场。

**💡 创新点**

创新点在于将分布式热通量作为结构化图编码，引入物理感知注意力与热图传播模块，并采用 Sobolev 正则化损失，综合全局物理耦合与局部热扩散，实现边界信息、全局物理耦合与局部热传导的联合建模。

**🔧 技术方法**

使用 Transformer 结构的神经算子、图注意力、物理感知注意力、热图传播、Sobolev 损失以及多块 PNOT 堆叠等技术。

**📊 数据集**

使用基于有限元的 EAST 分流器温度场数据集，共 710 个样本，包含 10 个热源功率、71 条边界采样点及 3908 空间节点，时域 1001 帧。

**📈 对比分析**

与 DeepONet、FNO、GNOT、Transolver、DPOT 等 20 多种算子方法对比，PNOT 在 Rel L2、rRMSE、rMAE、MAE 上均达到最低误差，误差约为 0.0008，较基线降低 62%。

**⚠️ 局限性**

局限性：仅在二维有限元模拟数据上验证，未考虑三维真实热通量、强耦合多物理效应及实际实验条件，且模型仅针对 EAST 分流器，跨设备推广尚未系统验证。

---

## 426. Mitigating Positional Leakage in 3D Masked Autoencoders for Robust Representation Learning

**arXiv ID:** 2606.31570 | [PDF](https://arxiv.org/pdf/2606.31570v1)

**作者:** Xu Yan `[一作]` (Beihang University), Di Huang `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研发了 MPL-MAE 框架，通过重新校准位置嵌入（RPE）和门控位置接口（GPI），降低 3D MAE 中的位置信息泄露，提升语义特征学习效果。

**💡 创新点**

创新点在于引入顺序同构编码的离散秩索引消除绝对坐标信号、残差 MLP 进一步重建拓扑，并通过 Gumbel‑Softmax 门控动态抑制位置注入，解决传统 MAE 过度依赖位置的瓶颈。

**🔧 技术方法**

技术实现包括：秩索引 + 正弦位置编码 + 轻量级 MLP 重新映射、GPI 中的 Gumbel‑Softmax 门控、Chamfer 距离重建损失、拓扑一致性正则化和泄露抑制正则化。

**📊 数据集**

预训练使用 ShapeNet，评估数据集涵盖 ModelNet40、ScanObjectNN（OBJ‑ONLY、PB‑T50、OBJ‑BG）、S3DIS、PCN 等多种 3D 任务数据。

**📈 对比分析**

与 Point‑MAE、PCP‑MAE 等主流 3D MAE 进行线性、MLP、全微调、少样本、配准、语义分割与重建等多任务对比，MPL‑MAE 在 ScanObjectNN、ModelNet40 等下游任务均取得更高准确率或更低误差，且在噪声鲁棒性实验中表现尤为突出。

**⚠️ 局限性**

局限在于仍需手工设定掩码比例与局部补丁划分，未针对多模态或大规模点云进行广泛验证；对极端噪声或非欧几里得几何的适应性仍有待进一步研究。

---

## 427. Building an ASR Solution for Training and Assessing Children's Reading

**arXiv ID:** 2606.31508 | [PDF](https://arxiv.org/pdf/2606.31508v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 428. CVE-TTP KG: Knowledge Graph Linking Software Vulnerabilities to Attack Behaviors

**arXiv ID:** 2606.31557 | [PDF](https://arxiv.org/pdf/2606.31557v1)

**作者:** Basant Agarwal `[一作]` (Central University of Rajasthan), Vinod P `[通讯]` (Cochin University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 CVE–TTP 知识图谱，将软件漏洞与 MITRE ATT&CK 的战术与技术关联。

**💡 创新点**

创新点在于结合 transformer 进行多标签技术/战术分类以及端到端的实体关系联合抽取，并通过 Neo4j 可视化。

**🔧 技术方法**

使用 CySecBERT 等预训练网络进行分类，BERT–BiLSTM–CRF 与 span‑based joint 模型进行抽取。

**📊 数据集**

利用从 NVD/CVEList 收集的 276k CVE、CWE、CAPEC 与 ATT&CK 的 178k 样本以及 1,080 条手工标注的实体关系数据集。

**📈 对比分析**

与 pipeline 和 joint 两种抽取方案对比，pipeline 在实体 0.86、关系 0.99 的 macro‑F1 领先，而 joint 在整体 KG 质量上更鲁棒，分类任务 CySecBERT macro‑F1 分别为 0.8771 与 0.9616。

**⚠️ 局限性**

局限包括映射覆盖率不足、实体歧义与共指解析不足，以及对新兴 CVE 的实时更新和跨语言支持的挑战。

---

## 429. Beyond the Expressivity-Trainability Paradox: A Dynamical Lie Algebra Perspective on Navigating Barren Plateaus in Quantum Machine Learning

**arXiv ID:** 2606.31536 | [PDF](https://arxiv.org/pdf/2606.31536v1)

**作者:** Kung-Ming Lan `[一作]` `[通讯]` (Tunghai University), Kung-Ming Lan (Tunghai University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

探讨量子机器学习中的表述力-可训练性悖论，利用动态李代数分析并通过对称保留的几何Ansatz来避免Barren Plateau。

**💡 创新点**

首次将DLA维度与梯度方差关联，提出“可训练性按设计”策略，证明对称约束可把DLA从指数扩展限制为多项式。

**🔧 技术方法**

动态李代数、几何量子机器学习、变分量子算法、梯度方差分析、PennyLane模拟。

**📊 数据集**

Make Moons 二分类数据集。

**📈 对比分析**

与传统硬件高效Ansatz（HEA）比较，评估梯度方差、DLA维度与训练/验证准确率；对称Ansatz在梯度保持、可训练性和泛化表现优于HEA，虽训练准确略低但稳健。

**⚠️ 局限性**

对称Ansatz牺牲部分表述力，DLA分析在大规模量子系统上仍计算昂贵；未验证对噪声鲁棒性和更复杂对称的适用性。

---

## 430. A time-series classification framework for individual-level absenteeism prediction under severe class imbalance

**arXiv ID:** 2606.31532 | [PDF](https://arxiv.org/pdf/2606.31532v1)

**作者:** Kwong Ho Li `[一作]` (Adelaide University), Wathsala Karunarathne `[通讯]` (Adelaide University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于时间序列分类（TSC）的个人缺勤预测框架，真正实现对未来缺勤事件的前瞻性预测；

**💡 创新点**

核心创新在于将历史出勤序列与未来缺勤标签分离，利用深度学习TSC处理严重类别不平衡；并给出二元焦点损失（BFL）与几何平均损失（G‑Mean）在高不平衡下梯度动态的理论分析与经验验证；

**🔧 技术方法**

使用长短期记忆网络（LSTM）、卷积网络（CNN）及其混合结构LSTM‑FCN，配合BFL与G‑Mean损失，批量标准化、滑动窗口特征工程；

**📊 数据集**

由于缺乏公开个人级长期出勤数据，构建了与UCI Absenteeism at Work数据集统计特征一致的可复现仿真数据；

**📈 对比分析**

通过在验证集与测试集上对比不同窗口大小、批量大小、架构与损失函数的组合，发现LSTM‑FCN+G‑Mean在窗口40–80天、批量≥64时，平衡准确率≈80%，特异率≈0.84；BFL在α=1/(1+ρ)且γ=0时可达特异率0.813，需手动校准；

**⚠️ 局限性**

主要限制是使用仿真数据而非真实组织长期缺勤记录，无法验证在真实场景下的泛化与部署效果；

---

## 431. Communication-Aware Robot Execution for Cloud Inference under Spatially Heterogeneous Connectivity

**arXiv ID:** 2606.31497 | [PDF](https://arxiv.org/pdf/2606.31497v1)

**作者:** Fengkai Liu `[一作]` (University of Osaka), Hideyuki Shimonishi `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出了一套针对在空间异质无线网络环境下，基于云端基础模型推理的机器人执行框架，核心思想是利用“请求–响应窗口”来估算一次云交互所需时间，并在当前任务原语支持范围内，动态选取最佳请求点，让机器人在执行原语时即进行下一次云请求，保证任务连续性。

**💡 创新点**

创新点包括：① 将云交互延迟与机器人本地运动支撑耦合为请求–响应窗口；② 将请求点视为运动决策，结合通信图做前向可行性与后向检索裕度过滤，得到最小代价的通信友好请求点；③ 在MPPI局部规划中加入通信软约束与到达时间惩罚，使机器人在请求提交后保持足够连通性并避免过早到达当前原语终点。

**🔧 技术方法**

技术手段主要有：基于EWMA的推理延迟预测、基于RSSI地图的传输速率估计、请求点优化（前向区域、后向检索裕度、通信阈值约束）、MPPI（模型预测路径积分）控制器以及基于通信质量的软成本与到达时间惩罚。

**📊 数据集**

实验使用了SODIndoorLoc数据集的SYL场景的Wi‑Fi RSSI测量构建的通信地图，并在Gazebo中模拟TurtleBot3 Burger机器人，采用云端VLM模拟返回的下一个航点序列，保持相同的延迟模型。

**📈 对比分析**

与三类基线（停等式、推理时序异步、固定周期异步）相比，提出的方法在三种观测范围下均实现了最高或相当的任务成功率，同时请求次数、等待时间、失败率和重叠率显著降低；完成时间基本保持与异步基线相当，证明不需要通过加大请求频率来提升成功率，而是通过通信友好的请求点实现更稳健的执行。

**⚠️ 局限性**

局限性：① 需要先验完整且静态的通信地图，无法处理人流、遮挡、网络负载等动态变化；② 仅在测量场景下验证，未在真实Wi‑Fi/5G网络中进行闭环实验；③ 当连通性缺失区域超出当前原语支持范围时，单纯的请求点优化无法保证连续执行，需进一步研究退化策略与任务重规划。

---

## 432. Improving Certified Robustness via Adversarial Distillation

**arXiv ID:** 2606.31653 | [PDF](https://arxiv.org/pdf/2606.31653v1)

**作者:** Matteo Melis `[一作]` (Queen's University Belfast), Vishal Sharma `[通讯]` (Queen's University Belfast)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的可靠性训练方法AD-CERT，结合对抗蒸馏和IBP上界，兼顾标准准确率和可验证性；

**💡 创新点**

创新点在于将对抗蒸馏的logit级别软标签作为低界近似引入，可在保持教师对抗鲁棒性的同时使用IBP进行可验证性提升；

**🔧 技术方法**

核心技术包括对抗蒸馏（KL散度）、IBP（区间边界传播）以及两者线性加权的损失函数；

**📊 数据集**

实验使用MNIST、CIFAR-10和TinyImageNet三个公开数据集，ε取值分别为{0.1,0.3},{2/255,8/255},{1/255}；

**📈 对比分析**

与当前最先进的可验证训练方法（IBP、SABR、MTL-IBP、CC-DIST等）在相同网络架构和验证器下进行对比，AD-CERT在标准准确率相近的前提下，取得了更高的可验证准确率，尤其在TinyImageNet上显著提升；

**⚠️ 局限性**

主要限制是教师-学生之间仍存在性能差距，尤其在较大扰动（ε=8/255）时标准准确率下降显著，说明将经验鲁棒性转化为可验证鲁棒性仍有挑战。

---

## 433. Think in English, Answer in Korean: Efficient Adaptation of Multilingual Tool-Using Agents

**arXiv ID:** 2606.31648 | [PDF](https://arxiv.org/pdf/2606.31648v1)

**作者:** Utsav Garg `[一作]` (Cohere), Susie Park `[通讯]` (Cohere)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过混合监督微调、可验证奖励学习和偏好对齐，将111B的韩英双语模型改造成支持推理与工具使用的企业代理，并实现4‑bit量化部署；

**💡 创新点**

创新在于使用预置条件混合推理模式、可验证奖励与语言一致性惩罚提升多语言推理与工具使用，以及将模型压缩至4‑bit实现单GPU可部署；

**🔧 技术方法**

采用混合监督微调（Hybrid SFT）、可验证奖励强化学习（RLVR+REINFORCE LOO）、直接偏好优化（DPO）、混合语言推理与语言一致性奖励以及4‑bit量化技术；

**📊 数据集**

使用公开推理数据集（AIME、MATH、Spider、BIRD、SynSQL）、内部Cohere训练集、企业NL2SQL、LG Enterprise评估、KMMLU、MMLU、ARC‑C、IFEval、MT‑Bench等；

**📈 对比分析**

通过与基线模型（原始111B）、GPT‑4o、Claude 3.7 Sonnet、Qwen3等对比，模型在韩英数学推理、NL2SQL、功能调用等指标显著提升；4‑bit模型在大多数基准上与FP8相近，能在单台80 GB H100上运行；

**⚠️ 局限性**

仍依赖英文推理导致韩语推理不足；内部评估受隐私限制难以复现；量化后未完成完整吞吐/能耗测评；缺乏原生韩语推理数据。

---

## 434. PrISM-IQA: Image Quality Assessment Made Practical for Smartphone Photography

**arXiv ID:** 2606.31626 | [PDF](https://arxiv.org/pdf/2606.31626v1)

**作者:** Shuyan Zhai `[一作]` (City University of Hong Kong), Kede Ma `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出PrISM‑IQA模型，将手机图像质量评估从单一MOS分数改为多问题序数诊断，预测每个ISP相关缺陷的四级严重度；

**💡 创新点**

创新点在于将累积序数编码与HEX结构化推理相结合，既保持单问题内部单调性，又捕捉跨问题的包含与互斥关系，提供逻辑一致且可解释的诊断结果；

**🔧 技术方法**

使用视觉‑语言模型（CLIP+CoOp）生成每个缺陷的原始严重度得分，再通过累计序数潜能和HEX图进行结构化推理；

**📊 数据集**

数据集包括：重构的SPAQ（带53个ISP相关问题的多标签序数标注）和一组由专业摄影师标注的2983张真实手机照片；

**📈 对比分析**

与多种主流无参考IQA模型（MUSIQ、DBCNN、TReS、UNIQUE、LIQE、TOPIQ、Q‑Insight）以及相同骨干但无结构化推理的对照版本进行对比，PrISM‑IQA在多种评价指标（ACC、tAUC、QWK）上均显著提升，且在MOS线性探测上获得最高的SRCC/PLCC；

**⚠️ 局限性**

局限在于仍以低层ISP缺陷为中心，缺乏对高层摄影意图或美学属性的建模；缺陷到ISP模块的映射多为规则化手工，未实现端到端可微的优化；以及对跨设备、视频或不完整标签的鲁棒性尚待进一步研究。

---

## 435. What Memory Do GUI Agents Really Need? From Passive Records to Active Task-Driving States

**arXiv ID:** 2606.31612 | [PDF](https://arxiv.org/pdf/2606.31612v1)

**作者:** Chen Liu `[一作]`, Yue Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出主动任务驱动内存（ATMem）框架并结合STR-GRPO在线强化学习，提升长时序移动GUI代理的执行效果。

**💡 创新点**

将内存从被动记录转为主动执行状态，记录数据所有权、状态、约束，并通过对比内存开启/关闭的RL策略实现智能使用。

**🔧 技术方法**

结构化内存模型ATMem、对比式STR-GRPO在线RL、基于Qwen3-VL-8B的视觉语言模型以及UIIns视觉 grounding。

**📊 数据集**

使用AndroidWorld、MobileWorld以及自建的DataScope数据集（32个模板，3难度等级）。

**📈 对比分析**

与UI-TARS、MAI-UI、GUI-Owl等基线在终端成功率、App级进度和Scope-F1上对比，ATMem-UI-8B在AndroidWorld成功率达76.6%（+5.9%），在DataScope显著提升S_prog和S_scope。

**⚠️ 局限性**

仍存在循环/错误终止/步骤上限等失败模式，且在极高噪声/复杂约束下对内存压缩效果有限。

---

## 436. CLExEval: A Human-in-the-Loop Framework for Qualitative Evaluation of LLM Clinical Reasoning

**arXiv ID:** 2606.31608 | [PDF](https://arxiv.org/pdf/2606.31608v1)

**作者:** Ajmal M. `[一作]` (MBZUAI), Zhuohan Xie `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出CLExEval框架，用人类专家和逐步信息屏蔽评估LLM临床推理；

**💡 创新点**

创新点在于将信息稀缺下的逐层屏蔽与专家评注结合，揭示评估幻觉、隐藏知识悖论和推理-输出失配三种失效模式；

**🔧 技术方法**

采用进步信息屏蔽、结构化推理输出、CLExEval七维评估量表、信息稀缺敏感度（ISS）、单调性违背率（MVR）和推理-输出失配率（ROM）等技术；

**📊 数据集**

使用RARECASE‑2000（由MultiCaRe构建的2000个渐进屏蔽病例）与5,600份专家评分；

**📈 对比分析**

与GPT‑4o‑mini、HuatuoGPT‑o1两大模型对比，GPT‑4o‑mini在多维度得分更高但易受信息稀缺影响；实验显示自动评判者（LLM-as-a-Judge）在已知失败集上有高达100% 的“幻觉批准率”，表明仅靠自动评判过于乐观；

**⚠️ 局限性**

局限在于样本规模有限（仅40例、200实例），专家标注耗时且不具代表性，难以推广至更大模型或常规病例；

---

## 437. DPPE: Rethinking Camera-Based Positional Encoding for Scaling Multi-View Transformers

**arXiv ID:** 2606.31585 | [PDF](https://arxiv.org/pdf/2606.31585v1)

**作者:** Shun Kenney `[一作]` (Keio University), Teppei Suzuki `[通讯]` (SB Intuitions)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了Decoupled Pose Positional Encoding (DPPE)，在Transformer的注意力机制中显式分离相机旋转与平移信息；

**💡 创新点**

创新点在于通过将旋转和平移映射到不同的特征维度或使用投影矩阵的双重形式，解决了传统投影矩阵编码中相机参数不可辨识导致的训练瓶颈；

**🔧 技术方法**

使用了Transformer结构、相机基位置编码（PRoPE）、RoPE、以及新构造的DPPEtAdd和DPPEdual两种变体；

**📊 数据集**

在MVImgNet2、RealEstate10K和SpatialVidHQ三个多视角数据集上进行实验；

**📈 对比分析**

与PRoPE和GTA等基线进行对比，使用PSNR、SSIM和LPIPS评估。DPPE在MVImgNet2上显著提升PSNR（+1.2dB）和其他指标，并在放大倍数与视角数的外推设置下保持优越性能；

**⚠️ 局限性**

在Camera轨迹相对简单（如仅平移或仅旋转）的SpatialVidHQ数据集上，DPPE与PRoPE无显著差异，且DPPEtAdd在查询-键位置编码时可能引入尺度相关偏差，影响部分场景。

---

## 438. HABIT: Human-Aware Behavior and Interaction Training Dataset for Robot Manipulation

**arXiv ID:** 2606.31682 | [PDF](https://arxiv.org/pdf/2606.31682v1)

**作者:** Jaehwi Song `[一作]` (Config), Kimin Lee `[通讯]` (Config)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并构建了 HABIT 这一大规模人机共现演示数据集，并在其中训练并评估了两种视觉‑语言‑动作（VLA）模型，验证其在人机交互任务中的性能提升；

**💡 创新点**

创新点包括：①以三种人机角色（协作者、同事、监督者）组织任务，明确人机交互模式；②设计任务工作流与交互细节收集协议，系统诱导人类意识行为（如配合、让位、手势跟随）；③引入人机共现数据作为新的多样性轴，显著提升机器人在共享工作空间中的安全与协作能力；

**🔧 技术方法**

技术实现主要利用双臂 Franka 机器人与 Robotiq 抓手，Meta Quest 3 手柄进行远程操作，五路同步 RGB 摄像机采集观测；训练采用现有的两款 VLA 模型 π_0.5 与 GR00T N1.6，统一训练步骤与批量尺寸，对比 Robot‑only 基线；

**📊 数据集**

使用了 HABIT 数据集（10,563 期、164 小时、60 任务），每个任务分为协作者、同事、监督者三类；并与在无人类参与条件下收集的 Robot‑only 数据进行对照；

**📈 对比分析**

通过 6 个代表任务（每角色两项）进行 20 次试验，评估成功率、工作流合规和人机安全三项指标；结果显示 HABIT 在同事和监督者角色任务中显著提高成功率，角色特定失败率显著下降，并在新任务中显著提升样本效率；

**⚠️ 局限性**

局限性包括：仅在单一实验环境、一对一人机配置下收集数据，缺乏多样化的人体形态和复杂多机器人/多人的场景；实验仅使用机器人侧摄像头，未充分利用人类侧摄像头信息；真实世界评估难以完全复现。

---

## 439. REDI: Corpus Aware Patch Ranking for DINOv3 Token Reduction

**arXiv ID:** 2606.31676 | [PDF](https://arxiv.org/pdf/2606.31676v1)

**作者:** Chanjong Im `[一作]` (University of Magdeburg), Thomas Mandl `[通讯]` (University of Hildesheim)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种基于类条件的视觉词 TF‑IDF 与图像特定注意力相结合的离线 patch 重要性评分 REDI，用于在固定 token 预算下对 DINOv3 ViT‑B/16 进行 token 减少。

**💡 创新点**

创新点在于：①将冻结 ViT 的 patch 表征量化为视觉词并构造类条件 TF‑IDF 词典；②将该词典统计与来自 dense 传递的注意力权重乘积得到复合评分；③在保持相同 keep‑merge‑compress 规则的前提下，证明复合评分能显著提升在 107 令牌下的分类精度。

**🔧 技术方法**

使用的技术包括：DINOv3 ViT‑B/16 预训练模型、球面 k‑means 聚类、TF‑IDF 词权重、注意力列累加（incoming attention mass）、归一化与逐元素相乘、固定的 keep/merge/compress 操作、线性分类器（在 dense 路径上训练）。

**📊 数据集**

使用的数据集为 ImageNet‑1K（训练集用于构建词典、统计并训练线性分类器，验证集用于评估）。

**📈 对比分析**

比较方法：在相同的冻结 backbone、相同的线性分类器、相同的 token 预算（107 令牌）下，对不同 scoring 信号（仅 TF‑IDF、仅注意力、两者乘积）进行评估。结果显示，乘积方案在 Top‑1 上从 82.63%（仅注意力）提升到 84.71%，超过 dense 方案（83.51%）并比任何单一信号高 1.1–2.9 点。

**⚠️ 局限性**

局限性：①REDI 依赖 ground truth 类标签和 dense 传递的注意力，属于离线参考而非可直接部署的加速方案；②仅在 DINOv3 ViT‑B/16 上验证，未在更大/不同的模型或任务上证明；③未考虑评分构建成本、额外内存或硬件实现的影响；④固定 reducer 可能不适用于所有 token 预算或更复杂的 reducer。

---

## 440. SAMBA: A Scatter-Guided Masked Bidirectional Mamba Foundation Model for SAR Target Recognition

**arXiv ID:** 2606.31668 | [PDF](https://arxiv.org/pdf/2606.31668v1)

**作者:** Ke Wang `[一作]` (National University of Defense Technology), Shunping Xiao `[通讯]` (National University of Defense Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于Mamba的自监督预训练模型SAMBA，用于合成孔径雷达目标识别；

**💡 创新点**

核心创新包括线性复杂度Mamba编码器与中位类token、基于SAR散射先验的三层层次SG‑MAE掩码策略以及轻量级SpatialMix解码模块；

**🔧 技术方法**

采用Mamba状态空间模型、散射引导掩码、深度可分离卷积+通道MLP的SpatialMix、两阶段跨域预训练等技术；

**📊 数据集**

使用ImageNet与186K未标记SAR图像进行预训练，并在MSTAR、FUSAR‑Ship、SAR‑ACD、SSDD、SARDet‑100K、SIVED、SAR‑Aircraft等七个基准数据集进行下游评估；

**📈 对比分析**

与CNN、ViT、HiViT等主流骨干以及现有SAR自监督模型相比，SAMBA在参数量、计算量均显著更低，同时在少样本分类与多目标检测任务上均取得或突破SOTA性能；

**⚠️ 局限性**

局限性在于仍未在高分辨率SAR场景及语义分割等更复杂任务上验证，且对极端噪声与多尺度目标的鲁棒性尚待进一步提升。

---

## 441. ForecastAgentSearch: Towards a Multi-Expert Agent Search System for Geopolitical Event Forecasting

**arXiv ID:** 2606.31665 | [PDF](https://arxiv.org/pdf/2606.31665v1)

**作者:** Miaomiao Cai `[一作]` (National University of Singapore), See-kiong Ng `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出一种多专家代理搜索系统ForecastAgentSearch，用于地缘政治事件预测；

**💡 创新点**

将预测任务转化为专家搜索与协调问题，强调任务意识的专家检索与组合；

**🔧 技术方法**

基于大型语言模型（LLM）的检索增强生成、专家代理微调与协调模块；

**📊 数据集**

摘要中未说明具体使用的数据集；

**📈 对比分析**

未给出实验比较或性能指标；

**⚠️ 局限性**

未提供实验限制，本文仅提出框架与理论探讨

---

## 442. Moral Safety in LLMs: Exposing Performative Compliance with Puzzled Cues

**arXiv ID:** 2606.31644 | [PDF](https://arxiv.org/pdf/2606.31644v1)

**作者:** Mohammadamin Shafiei `[一作]` (University of Milan), Yulia Tsvetkov `[通讯]` (University of Washington)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型在隐式人口身份信息下的公平性表现，提出并量化了“提示可见度差距”指标，以评估模型的真实道德安全性。

**💡 创新点**

创新点在于通过保持道德困境与身份不变，仅改变身份呈现方式（直接标签对比逻辑谜题），从而识别出表面合规与真实安全的差异，并定义了可计算的Cue Visibility Gap。

**🔧 技术方法**

采用提示可变方法：生成逻辑谜题让模型推断身份，测量决策偏差；使用多模型评估、Bootstrap 统计验证结果，结合正则化的决策偏差指标。

**📊 数据集**

使用基于 DailyDilemmas 的 100 个日常道德困境，并在三种身份组合与三种难度级别下生成约 19,000 条测试样本；所有样本均经过人工复核确保身份与困境保持一致。

**📈 对比分析**

对 14 个公开及专有 LLM（如 Claude、GPT‑4o、Llama 等）在 Direct 与 Puzzled 条件下计算 Favor 与 Against 率，发现隐形标签会使 Against 率上升约 4.4pp，Cue Visibility Gap 能重新排列模型的安全排名，表明不同模型在隐性提示下的表现差异显著。

**⚠️ 局限性**

局限性包括：将宗教与种族混合成同一类别；逻辑谜题引入额外推理负荷，可能与自然隐式身份提示产生偏差；缺乏对自然场景中隐性身份信息的验证。

---

## 443. A Lifecycle and Application-Stack Survey of Large Language Model Vulnerabilities: Attacks, Risks, Defenses, and Open Problems

**arXiv ID:** 2606.31639 | [PDF](https://arxiv.org/pdf/2606.31639v1)

**作者:** Seyed Bagher Hashemi Natanzi `[一作]` (Worcester Polytechnic Institute), Bo Tang `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对大语言模型（LLM）安全性问题进行系统性综述，提出了基于生命周期与应用栈的分类框架，梳理了攻击类型、威胁目标、攻击者能力和防御层级，并构建了防御深度架构。

**💡 创新点**

创新点包括：①将LLM漏洞归入数据收集、预训练、对齐、包装、检索、提示、工具/代理、部署八个生命周期阶段，强调信任边界失效与信息流；②将安全目标扩展至安全性、隐私、公平、问责与代理控制；③提出跨层防御组合的重要性，并对现有调查与行业框架进行对比，强调系统视角的缺失。

**🔧 技术方法**

主要技术手段是系统化分类与映射、跨层防御设计、对安全目标的细分，以及对比评估与案例分析；没有进行实验性算法实现。

**📊 数据集**

所使用的数据集为公开论文与行业安全框架（如OWASP LLM Top 10、NIST AML taxonomy、MITRE ATLAS）及其引用的实验论文，未涉及自建数据集。

**📈 对比分析**

对比方法主要为文献与框架对齐，说明本文在生命周期与安全目标维度的覆盖率高于已有调查；由于为综述性工作，未给出量化性能指标。

**⚠️ 局限性**

局限性：①缺乏对防御组合的形式化验证与实证评估；②对多模态、跨通道攻击的探讨仍停留在概念层面；③依赖现有文献，可能遗漏新兴攻击与防御；④在实践部署时仍需要更多可操作的安全控制细节与自动化工具。

---

## 444. Intrinsic decomposition and editing of 3D Gaussian splats

**arXiv ID:** 2606.31637 | [PDF](https://arxiv.org/pdf/2606.31637v1)

**作者:** Alexandre Lanvin `[一作]` (Inria, Université Côte d’Azur), George Drettakis `[通讯]` (Inria, Université Côte d’Azur)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过将三维高斯球体（3D Gaussian Splatting）分解为独立的反照率、阴影与视角相关残差三组，高斯球体分别表示这三种物理量，并在此基础上实现对反照率（纹理）的交互式编辑；

**💡 创新点**

①采用独立三组高斯球体分别建模 albedo、shading 与 residual，允许每种信号自适应其空间分辨率；②利用视频扩散模型（DiffusionRenderer + Brick Diffusion）获取多视角一致的 albedo 先验；③结合深度正则化和分阶段优化（先估 albedo，再优化阴影，最后残差），显著提升分解质量；

**🔧 技术方法**

3D Gaussian Splatting、视频扩散模型（Cosmos DiffusionRenderer、Brick Diffusion）、单目深度估计器、Structure‑from‑Motion、梯度优化、灰度正则化、线性空间渲染、浮点化消除等技术；

**📊 数据集**

实拍的四个室内场景（使用 Canon EOS R6 连拍）以及一个合成室内场景（提供 ground‑truth albedo 与深度）；

**📈 对比分析**

与 GI‑GS、R3GS 等完整逆渲染方法对比：在 albedo 质量、PSNR/SSIM、LPIPS 上表现更优，训练时间稍长但速度仍可实时（约 90 FPS）；在交互编辑方面相较 ReCoGS 与 SplatShop，能够保持阴影与纹理，提供更真实的编辑效果；

**⚠️ 局限性**

①3DGS 表示本身模糊，导致编辑需要手动拟合平面代理，非通用；②依赖扩散模型的 albedo 预测，图像往往模糊并可能把纹理误归为阴影；③多次渲染导致训练速度慢；④在暗色区域易出现 popping，渲染质量受限。

---

## 445. A Tutorial on Autonomous Fault-Tolerant Control Using Knowledge-Grounded LLM Agents

**arXiv ID:** 2606.31635 | [PDF](https://arxiv.org/pdf/2606.31635v1)

**作者:** Javal Vyas `[一作]` (Imperial College London), Mehmet Mercangöz `[通讯]` (Helmut Schmidt University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出了一套基于知识驱动的LLM代理框架，用作约束性监督规划器来实现工业过程厂区的主动容错恢复，并给出了三种设计维度（应用模式、验证策略、部署考量），同时提供了两套可直接使用的可执行Python实验环境。

**💡 创新点**

创新点在于：①将LLM与传统FTC工具（知识图谱、数字孪生）结合，形成可验证、可回溯的决策链；②将恢复问题分解为三种典型模式，明确LLM在不同操作域（离散路由、连续设点、混合）中的作用；③引入符号与仿真两种验证策略，兼顾效率与动态可行性；④提供统一接口与可复用的实验环境，便于跨案例的迁移与评估。

**🔧 技术方法**

采用技术包括：大型语言模型（LLM）进行自然语言推理与动作生成；知识图谱提取与检索支持语义上下文；数字孪生模型用于动态验证；符号检验与仿真验证机制；多代理工作流（监测、规划、动作合成、验证、再提示）。

**📊 数据集**

使用的数据集主要是两套自定义的仿真环境：一是模块化混合单元（离散状态机），二是连续搅拌槽（PID调节）。两者均通过注入人工故障（泵失效、阀门堵塞、传感器偏差等）生成评估场景；未使用公开工业真实数据集。

**📈 对比分析**

方法对比：论文并未给出定量性能指标，而是通过可执行环境展示框架在不同模式下的适用性与验证流程。实验环境可记录故障上下文、LLM建议、验证结果与过程轨迹，便于后续的定量评估，但当前论文重点在设计与实现，而非性能对比。

**⚠️ 局限性**

局限性包括：①缺乏真实工业工厂验证，效果需在更高保真模型上进一步验证；②LLM易出现幻觉或不合理建议，需依赖外部验证器保证安全；③知识图谱的构建与维护成本高，依赖人工审核；④计算与网络延迟限制在高频率闭环中的适用；⑤安全层集成仍需在实际工厂环境中细化。

---

## 446. Scientific Explanations in Health Sciences: Causality, Trust, and Epistemic Adequacy

**arXiv ID:** 2606.31616 | [PDF](https://arxiv.org/pdf/2606.31616v1)

**作者:** Martina Mattioli `[一作]` (Zhejiang Normal University), Marcello Pelillo `[通讯]` (Ca' Foscari University of Venice)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过哲学视角批判性综述医学解释与可解释AI的关系，并提出哲学基础的设计原则以提升医学XAI的可信度与可解释性。

**💡 创新点**

首次系统融合哲学关于因果、信任与真确性三大维度与医学XAI实践，构建面向利益相关者的可解释性框架。

**🔧 技术方法**

采用哲学文本分析、案例评估与概念框架构建等方法。

**📊 数据集**

未使用实验数据，主要以文献与案例（如乳腺癌Grad‑CAM++、肺炎GANterfactual）为参考。

**📈 对比分析**

本研究未进行定量对比实验，而是通过案例说明与概念分析论证设计原则的合理性。

**⚠️ 局限性**

主要局限在缺乏系统的经验评估与实证验证，以及对现有XAI方法在多因果模型下适配性的实证不足。

---

## 447. Automating Cause-Effect Specification with Knowledge Graphs and Large Language Models

**arXiv ID:** 2606.31614 | [PDF](https://arxiv.org/pdf/2606.31614v1)

**作者:** Javal Vyas `[一作]` (Imperial College London), Mehmet Mercangöz `[通讯]` (Imperial College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于知识图谱与受约束大语言模型的管线，自动生成工业过程的因果（C&E）表、操作员说明和可验证的SWRL规则。

**💡 创新点**

创新点在于将语义层（对齐本体）与生成式AI结合，确保三种产出（表格、自然语言、规则）在语义上一致并可机器验证，同时将诊断知识、结构、行为和运维逻辑统一编码。

**🔧 技术方法**

使用的技术包括：对齐本体（多模态 ODP）、RDF/SPARQL、受约束LLM（对齐词汇表与语义限制）、SWRL 规则生成与验证。

**📊 数据集**

数据集为一个模块化过程工厂的多源信息：P&ID、工艺文档、PLC 逻辑、仿真模型、运营数据以及 CSV 型故障注释。

**📈 对比分析**

评估方法为提取 15 条 C&E 行并生成对应规则，结果显示 100% 语义基础、100% 语法合法、无冲突、无冗余；相比传统手工矩阵，自动化减少人工工作并提升一致性。

**⚠️ 局限性**

局限性包括：案例规模有限，未覆盖更复杂的工艺单元和故障传播；缺乏动态执行与实时验证；LLM 仍受训练数据与提示限制，可能在极端情况产生假想阈值。

---

## 448. Robust Text Watermarking for Large Language Models via Dual Semantic Embeddings

**arXiv ID:** 2606.31602 | [PDF](https://arxiv.org/pdf/2606.31602v1)

**作者:** Jonas Schäfer `[一作]` (Freie Universität Berlin), Gerhard Wunder `[通讯]` (Freie Universität Berlin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种双嵌入式水印（DEW），在LLM生成时利用上下文和候选词嵌入计算水印偏置，并在检测时统计这些偏置以鉴别水印文本。

**💡 创新点**

创新点在于同时考虑上下文与候选词的语义相似度来生成水印偏置，使同义词和翻译得到相似偏置，从而显著提升对改写与翻译的鲁棒性，并通过随机投影实现水印保密。

**🔧 技术方法**

采用语义嵌入模型（上下文与词嵌入）、随机投影、cosine相似度、tanh激活及正则化，生成和检测水印偏置，并以统计检验方式控制误报率。

**📊 数据集**

主要使用C4英文文本数据集进行生成实验，并通过GPT‑4o‑mini对生成文本进行改写与翻译测试。

**📈 对比分析**

与多种表面级和语义级水印（KGW、ATW、TS、SIR、X‑SIR等）进行对比，DEW在无攻击场景下1% FPR下TPR达98.8%–99.8%，改写场景TPR为74.6%，德语翻译TPR为65%，明显优于其他语义水印，并保持较低的计算开销。

**⚠️ 局限性**

局限在于未评估生成式攻击、未系统调优投影维度、tanh缩放因子等关键参数，且对低熵或指令对话、代码生成等场景的适用性尚待进一步验证。

---

## 449. Token-Sparse Medical Multimodal Reasoning via Dual-Stream Reinforcement Learning

**arXiv ID:** 2606.31599 | [PDF](https://arxiv.org/pdf/2606.31599v1)

**作者:** Kaitao Chen `[一作]` (Fudan University), Mianxin Liu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出ViToS，一个双流强化学习框架，用来在医学多模态任务中进行视觉令牌稀疏化和问答推理；

**💡 创新点**

创新点在于将视觉令牌剪枝与稀疏推理融合为一个统一的政策学习，并通过交叉反馈的顺序优化解决了双流策略的梯度冲突；

**🔧 技术方法**

技术包括：基于视觉令牌剪枝（Grounding-Aware VTP）、双流RL（定位分支与稀疏推理分支）、交叉反馈奖励、序列化优化与GRPO优化器；

**📊 数据集**

使用了七个医学VQA基准（PathVQA、SLAKE、VQA-RAD、OmniMedVQA、MMMU-Med、MedXpertQA、PMC-VQA）以及两款主流医学VLM（Lingshu-7B/32B、HuatuoGPT-Vision-7B）；

**📈 对比分析**

与传统VTP方法、现有RL医学模型相比，ViToS在所有基准上均优于基线，平均提升约5%，在Lingshu-7B上实现108.27%相对性能，HuatuoGPT-Vision-7B上实现104.16%；同时可将视觉令牌数量压缩至原来的77%，推理速度提升约4.5×；

**⚠️ 局限性**

局限性包括：对极高分辨率医学图像的定位动作空间仍然较大，当前仅在标准分辨率下评估；需要进一步探索层次化定位或更高效的策略学习以适应更大尺寸输入。

---

## 450. Digital Sovereignty as a Quality Attribute for Software Architectures

**arXiv ID:** 2606.31590 | [PDF](https://arxiv.org/pdf/2606.31590v1)

**作者:** Jukka Ruohonen `[一作]` (University of Southern Denmark), Mikkel Baun Kjærgaard `[通讯]` (University of Southern Denmark)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将数字主权（DS）建模为软件架构（SA）的质量属性（QA），并基于欧洲联盟（EU）云计算政策框架（如云主权框架 CSF 与技术主权方案）进行分析，提出 DS 的量化指标与情景评估模板。

**💡 创新点**

创新点在于：① 将 DS 视为可测量、可验证的 QA；② 引入 α、β、ζ、σ、ϕ、Φ 等一系列复合指标；③ 将 EU CSF 的八维度与软件架构度量相结合，构建可比较的评分公式。

**🔧 技术方法**

主要技术手段包括：解释性政策分析、软件架构质量属性评估方法、情景模板设计、数学公式与指标构造、风险评估模型（概率×影响）。

**📊 数据集**

使用的数据集为 EU 官方文件与政策文本（Tech Sovereignty Package、Cloud Sovereignty Framework、CADA 提案等），并未涉及实验性或行业数据采集。

**📈 对比分析**

比较方法：采用评分函数 f(·) 对 CSF 维度进行赋分，并计算云供应商的 s_i 分数；最终通过 Φ = ϕ × s 进行综合风险评估。由于缺乏实证验证，本文未给出具体性能数值，仅提供理论框架与可行性评估。

**⚠️ 局限性**

局限性：① 缺乏实证数据与经验验证；② 量化指标假设高锁定度（β>0.9）可能不适用于所有情境；③ 法律与跨境执法的不确定性导致模型难以完全验证；④ 仅聚焦云计算，未扩展至边缘、混合云等多元架构。

---

## 451. From Failure to Alignment: A Requirements Engineering Framework for Machine Learning Systems

**arXiv ID:** 2606.31589 | [PDF](https://arxiv.org/pdf/2606.31589v1)

**作者:** Amel Bennaceur `[一作]` (Open University), Faeq Alrimawi `[通讯]` (University of Limerick)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 REAL 框架，结合需求工程与失效分析来驱动机器学习系统的需求定义、验证与多层适配，并在自动驾驶制动系统上进行了实验验证。

**💡 创新点**

将失效视为第一类构造，用障碍分析将经验失败转化为需求/域假设的改进；并提出跨数据、模型、系统与需求四层的协同适配循环，形成一种系统化的失效驱动需求工程流程。

**🔧 技术方法**

采用语法指导的情景生成（Grammatical Evolution/GRAPE）、KAOS 目标建模、CARLA 车辆仿真、Scenic 语法、YOLOv5 感知模型、障碍识别与聚类分析以及多层适配策略。

**📊 数据集**

初始训练使用 COCO 数据集，后续针对失效进行细化时使用 CARLA‑BSP 的子集（包含儿童行人、恶劣天气等场景），并通过手工标注的失败样本进行少量增量学习。

**📈 对比分析**

通过对比 YOLOv5s 与 YOLOv5m、数据增广、系统级安全回退等多层适配方案，基准情景下子机型误检率从 100% 降至 8%，在儿童和雾天场景中成功制动率提升至 100%，但大模型在短距离场景中因推迟导致的违规仍显现。

**⚠️ 局限性**

评估仅限单一安全关键例子（单感知模块），未涉及多模态、多组件或非网络化域；缺少真实利益相关者参与，障碍归类与修复仍以人工判断为主，且失效发现的完整性受情景生成范围和模拟逼真度限制。

---

## 452. ZEBRA: Zero-Shot Entropy-Regularized Prompt Learning for Base-to-Novel Generalization in Audio-Language Models

**arXiv ID:** 2606.31587 | [PDF](https://arxiv.org/pdf/2606.31587v1)

**作者:** Asif Hanif `[一作]` (Mohamed Bin Zayed University of Artificial Intelligence), Mohammad Yaqub `[通讯]` (Mohamed Bin Zayed University of Artificial Intelligence)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对音频语言模型（ALM）中提示学习导致的基础类（base）与新颖类（novel）性能差距问题，提出了ZEBRA框架，融合零射击（zero‑shot）logits 与自熵正则化，保持零射击迁移能力并显著提升新颖类性能。

**💡 创新点**

创新点在于：① 以零射击logits为锚点进行logit融合，避免提示学习完全偏离预训练对齐；② 引入自熵正则化抑制对基础类的过度自信，从而缓解过拟合；③ 该方法不增加可训练参数、计算开销微小，可直接叠加于现有提示学习方法之上。

**🔧 技术方法**

使用技术包括：CLIP‑style ALM（Pengi）框架、COOP 与 COCOOP 提示学习方法、logit 融合、self‑entropy 正则化以及少量样本（few‑shot）微调。

**📊 数据集**

实验数据集涵盖多种音频分类任务：Beijing Opera、NS‑Instruments、ESC‑50、ESC50‑Actions、UrbanSound8K、CREMA‑D、RAVDESS、VocalSound、SESA、TUT2017、GT‑Music‑Genre 等。

**📈 对比分析**

与 Zero‑Shot、COOP、COCOOP 直接比较；ZEBRA 在 novel 类的平均准确率提升约 4–5% 并保持 base 类性能；同时减少 ECE（期望校准误差），证明模型更稳健、泛化更好。

**⚠️ 局限性**

限制与挑战：① 依赖预训练 ALM 的零射击 logits，若预训练不佳效果受限；② 超参数（λ_zs、λ_pr、entropy 权重）需经验设置；③ 只在少量样本场景验证，未探讨大规模数据或多任务迁移的表现。

---

## 453. Semantic Occupancy Prediction with Dual Range-Voxel Representation

**arXiv ID:** 2606.31688 | [PDF](https://arxiv.org/pdf/2606.31688v1)

**作者:** Sitao Chen `[一作]` (South China University of Technology), Mingkui Tan `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种双视角（range-view + voxel-view）表示，利用单扫稀疏激光点云进行 3D 语义占据预测。

**💡 创新点**

创新点包括：① 只使用单扫点云，消除多扫拼接的效率和鲁棒性问题；② 通过球面投影得到连续上下文的 range-view，和多尺度 voxel-view 结合；③ 设计 voxel-to-range 与 range-to-voxel 的双向融合模块，使用可变形注意力与稀疏卷积提升几何与语义特征互补。

**🔧 技术方法**

使用技术包括：球面投影、体素化、3D 稀疏卷积、多尺度 voxel-view 编码器、Deformable Cross‑Attention、稀疏基础块（SparseVFE）、焦点损失+Dice 损失等。

**📊 数据集**

实验数据集：nuScenes‑Occupancy、SemanticKITTI、SemanticPOSS。

**📈 对比分析**

与多扫 LiDAR 方法和相机方法对比，单扫 DRVR 在 nuScenes‑Occupancy 取得 30.6% IoU / 21.3% mIoU，领先多扫 9.5%+；在 SemanticKITTI 取得 61.4% IoU / 28.5% mIoU，超越 SSA‑SC、JS3C‑Net 等方法；速度提升 2.1×、参数量仅 19.67M，性能优异。

**⚠️ 局限性**

局限性：对极端稀疏点云或低分辨率激光仍有挑战；仅使用单模态，未针对多模态融合或恶劣天气做鲁棒性提升；对激光噪声/失效的敏感度尚待进一步研究。

---

## 454. When to Truncate a Feature Ranking: A Residual-Overlap Stopping Rule for Subset Selection

**arXiv ID:** 2606.31686 | [PDF](https://arxiv.org/pdf/2606.31686v1)

**作者:** Jesus S. Aguilar-Ruiz `[一作]` `[通讯]` (Pablo de Olavide University), Jesus S. Aguilar-Ruiz (Pablo de Olavide University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于风险校准的停止规则，将已排序的特征列表截断为一个共享的、与类别无关的子集，截断点由累积的两两Bhattacharyya重叠满足阈值决定。

**💡 创新点**

创新点在于把特征排序的截断与贝叶斯风险关联，利用产品边缘模型的Bhattacharyya系数因式分解得到可加的对数分离度，并给出显式的风险上界，使截断阈值可从目标风险水平逆推而来。

**🔧 技术方法**

采用Bhattacharyya系数、产品边缘模型、贝叶斯风险上界、基于核密度估计的重叠量估计，以及前缀扫描的高效实现。

**📊 数据集**

在CuMiDa公开的18个高维基因组数据集上进行实验，变量数从约2.2万到5.5万不等。

**📈 对比分析**

与全特征基线以及基于互信息和卡方的两种排序方法在Gaussian NB和逻辑回归下对比；在保持预测性能相当的前提下，将特征数从几万降至几十，平均准确率和宏F1与全特征相当，且差异未达到统计显著水平。

**⚠️ 局限性**

局限在于校准只针对产品边缘模型，真实特征间存在相关性时可能出现校准误差；重叠估计在样本量小或连续变量时易不稳；并且给出的风险上界仅适用于理想的产品边缘贝叶斯分类器，对实际学习器的泛化性能不构成直接保证。

---

## 455. Exploring Side-Channel Protections in Hardware Implementations of PQC ML-KEM Verification

**arXiv ID:** 2606.31681 | [PDF](https://arxiv.org/pdf/2606.31681v1)

**作者:** Davis Ranney `[一作]` (Northeastern University), Yunsi Fei `[通讯]` (Northeastern University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文在 FPGA 与微控制器上实现并评估 ML‑KEM FO 验证的侧信道泄漏，比较了未加保护、哈希加密与高阶掩码三种防护，并系统性研究了并行度对泄漏和性能的影响。

**💡 创新点**

创新点在于首次将 ML‑KEM FO 验证移植到并行化 FPGA，实现多宽度比较，并揭示并行化与高阶掩码会显著削弱侧信道安全，说明现有防护在硬件加速环境中失效。

**🔧 技术方法**

使用侧信道分析技术（SNR、T‑检验、聚类分类、belief‑propagation 求解），FPGA SHAKE‑128 哈希实现、Galois Field 高阶掩码、随机行调度等硬件加速技术。

**📊 数据集**

利用 8,000 条功率侧信道采样（成功/失败两类）作为实验数据集，针对 ML‑KEM‑512 参数进行测试。

**📈 对比分析**

通过对比微控制器（0.046 Gb/s）与 FPGA（0.542–24.576 Gb/s）在不同并行宽度（32/128/512 位）下的 SNR、分类准确率和吞吐量；FPGA 在 512 位时实现 530 倍速度提升，但分类准确率和 SNR 随并行宽度急剧提升，表明并行化导致泄漏增强。

**⚠️ 局限性**

主要限制包括：实验仅在 Spartan‑6 FPGA 上验证，结果对不同 FPGA 族的普适性未知；高阶掩码在并行环境下失效，需进一步探索更强的硬件级防护；研究未覆盖对攻击者可用的完整密钥收集窗口和对动态随机化策略的完整评估。

---

## 456. ShellMaker: Language-Guided Exterior Completion under Structural Constraints

**arXiv ID:** 2606.31680 | [PDF](https://arxiv.org/pdf/2606.31680v1)

**作者:** Ruiqi Xu `[一作]` (Purdue University), Daniel Aliaga `[通讯]` (Purdue University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 ShellMaker，一种语言引导的建筑外壳补全框架，能够在保持建筑脚印、墙体边界和门窗开口不变的前提下，生成完整且风格一致的 3D 外墙、屋顶和细节。

**💡 创新点**

创新点在于：① 将 LLM 用于将模糊的风格提示转化为面向具体部件的结构化提示；② 结合参数化屋顶生成、兼容性感知的墙屋顶材质检索和几何感知组装，形成三阶段模块化流水线；③ 通过离线兼容矩阵和 UV 频率归一化实现材质一致性与纹理比例的自适应控制。

**🔧 技术方法**

使用技术包括：GPT‑4.1 对提示进行结构化；Nano Banana 生成参考图像；Trellis‑2 重建带 PBR 的网格；LLM 生成部件级提示；参数化屋顶算法（平顶、斜顶等）；CLIP 进行材质检索；几何布尔运算实现开口切割；UV 频率估计实现纹理尺度统一。

**📊 数据集**

数据集：200 个建筑脚本（100 来自室内场景生成器，50 CityGML，50 CAD），498 种墙体纹理与 81 种屋顶纹理（4k PBR），以及 60 条风格/材质提示模板。

**📈 对比分析**

与 Trellis‑2、SpaceControl、SAT‑Skylines 等基线对比，ShellMaker 在结构一致性指标（脚印 IoU ≈ 0.992、开口中心误差 ≈ 0.09）远超基线，且在 CLIP‑Sim 与 UNI3D 的风格对齐上保持相近甚至优于基线；在人工评估中赢率超过 70%。

**⚠️ 局限性**

局限性包括：部件生成偶尔产生形状失真或相交；对多层建筑的层级细节缺乏支持；仅依赖检索材质限制了非传统风格的多样性；对高度噪声或拓扑不一致的 CAD 输入的鲁棒性略低。

---

## 457. Practical High-Fidelity Novel-View Synthesis of Mounted Lepidoptera

**arXiv ID:** 2606.31679 | [PDF](https://arxiv.org/pdf/2606.31679v1)

**作者:** Kristof Overdulve `[一作]` (Hasselt University), Nick Michiels `[通讯]` (Hasselt University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种完整的管线，将脆弱的悬挂蝴蝶通过非接触镜子、手持聚焦堆叠和无分割镜像 3D 高斯喷射（3DGS）实现高保真、可全景渲染的三维重建

**💡 创新点**

1) 采用第一表面镜子在不触碰标本的情况下捕捉腹侧视角；2) 引入手持聚焦堆叠配合 ECC 对齐，消除手抖导致的重影；3) 开发无分割的镜像 3DGS 方案，通过对密集点云的镜面检测自动推断镜面并反射所有高斯，避免手工分割；4) 结合上述三项技术实现端到端可重复的高质量重建

**🔧 技术方法**

手持焦距堆叠（ECC对齐+Helicon Focus多分辨率融合）、结构光相机姿态估计（HLOC+SuperGlue）、多视角立体（COLMAP）、第一表面镜子配置、密集点云对称性检测+RANSAC/ICP、镜像 3D 高斯喷射（Mirror-3DGS）

**📊 数据集**

四只不同种类、尺寸、色彩与背面图案差异的悬挂蝴蝶标本；每只标本获取40–70个视角的聚焦堆叠图像（8K→4K）

**📈 对比分析**

对比两种宏观景深处理方案（手持聚焦堆叠 vs DoF‑aware 3DGS）和两种镜像处理方案（无镜像 3DGS vs Mirror‑3DGS）。聚焦堆叠在 PSNR、SSIM、LPIPS 上显著优于 DoF‑aware 3DGS（如 PSNR 20.10 vs 14.68）；镜像方案在保持单一物体一致性的前提下，Novel View 质量与无镜像方案基本相当（PSNR 23.51 vs 23.75），但在真实腹侧视角下能正确渲染，而无镜像方案会出现碎片化/浮点误差

**⚠️ 局限性**

（1）镜面反射导致的浮点噪声仍难以完全去除，需人工裁剪或后处理；（2）腹侧表面因视角稀疏、光照不足而分辨率略低；（3）全景渲染仍需手工前景分割以提升可视化效果

---

## 458. WorldRoamBench: An Open-World Benchmark for Long-Horizon Stability of Interactive World Models

**arXiv ID:** 2606.31672 | [PDF](https://arxiv.org/pdf/2606.31672v1)

**作者:** Ting-Bing Xu `[一作]` (Alibaba Group), Baoquan Chen `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一套针对交互式世界模型（Interactive World Models）的长期交互评测基准，涵盖动作跟随、视觉质量、交互物理和记忆四个维度。

**💡 创新点**

创新点包括：① 针对键盘动作的逐帧动作精度指标，突破传统轨迹级评测；② 基于段落的视觉漂移度量，捕捉非单调的质量下降；③ 物理评测通过“可控性门控”方式，仅在模型成功响应动作后评估物理合理性；④ 记忆评测采用基于点云的场景回溯和 VLM 推理的双轨方案，解耦动作误差与记忆衰退。

**🔧 技术方法**

技术方法主要包括：姿态估计（ViPE）、动作离散化与阈值微调、轨迹归一化与误差归一化、CLIP+线性美学预测器、MUSIQ 视觉质量评估、语义分割与点云对齐、VLM 问答推理、以及自适应阈值搜索。

**📊 数据集**

使用了约 600+ 个测试案例，涵盖自然、城市、室内场景，包含第一人称和第三人称视角，交互时长 10–60 秒，输入为键盘 WASD/IJKL 动作序列，并构造了专门针对动作、物理、记忆的子套件。

**📈 对比分析**

在 10+ 开源与闭源模型（如 Genie 3、Happy Oyster、Matrix‑Game 3.0、Lyra 2.0 等）上进行统一评测。结果显示：① 轨迹分数与逐帧准确率不一致；② 高视觉质量不必然伴随高动作跟随；③ 物理合规往往与轨迹误差正相关；④ 记忆得分与动作误差易混淆。总体来看，没有单一模型在所有维度均优，闭源模型在物理和记忆上表现突出，开放模型在动作跟随和视觉质量上更具竞争力。

**⚠️ 局限性**

局限性包括：评测时间仍相对较短（≤ 60 s），未覆盖更广泛的天气/光照变化；物理评测聚焦于碰撞、重力、光照，未涉及更复杂的物理过程；记忆评测依赖点云对齐，可能对细小物体或动态场景的记忆效果不够敏感；最后，评测框架仅支持离散键盘输入，未兼顾手势或自然语言指令。

---

## 459. Sensitivity Lower Bounds via Locally Testable Codes

**arXiv ID:** 2606.31657 | [PDF](https://arxiv.org/pdf/2606.31657v1)

**作者:** Yuichi Yoshida `[一作]` (National Institute of Informatics), Zihan Zhang `[通讯]` (National Institute of Informatics)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

提出一种通用框架，将局部可测试码（LTC）转换为约束满足问题（CSP）的敏感度下界，并用该框架证明了若干重要算法的最优敏感度下界。

**💡 创新点**

将LTC与算法敏感度关联，给出了信息论的最优Ω(n)下界；并通过显式LTC实例得到对Max E3LIN2、Max Cut、最大k-覆盖等问题的精确敏感度下界。

**🔧 技术方法**

利用系统化LTC的测试器、最近消息解码器、对称化与总变距离分析，以及LRCC代码与重复码的构造；结合PCP与局部测试理论。

**📊 数据集**

无实验数据，论文完全基于理论构造与证明；使用显式LTC实例（c³‑LTC、重复码）而非真实数据集。

**📈 对比分析**

与之前基于PCP的Ω(n^δ)下界对比，提供了常数因子下的Ω(n)最佳下界；在所有可考虑的近似率下实现与上界匹配，证明即便极小的稳定性也无法提升近似性能。

**⚠️ 局限性**

仅适用于线性系统化LTC和特定CSP，缺乏对非满足实例的直接扩展；并未给出平均情况的完整结论；对非线性代码或更广泛算法模型的适用性仍未证明。

---

## 460. FARS: A Fully Automated Research System Deployed at Scale

**arXiv ID:** 2606.31651 | [PDF](https://arxiv.org/pdf/2606.31651v1)

**作者:** Qiong Tang `[一作]` (Analemma), Yunfan Shao `[通讯]` (Analemma)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开部署了FARS，一个全自动AI研究系统，能够从研究主题生成、规划、实验到论文写作并产出166篇完整论文。

**💡 创新点**

创新点在于多阶段专用代理与共享可审计工作空间、持续大规模部署中的完整中间产物保存，以及结合人类评审与完整工件完整性审核来验证全自动研究的可行性。

**🔧 技术方法**

使用多代理架构，基于大型语言模型（如GPT‑4o、Claude等）完成文献检索、实验计划生成、代码执行、结果记录与论文撰写；配合GPU集群、统一模型接口、层化环境与预制技能库，并集成自动化审计与检查工具。

**📊 数据集**

使用公开AI/ML基准与数据集（如OpenAI、Papers with Code公开仓库及各类公开数据集）。

**📈 对比分析**

通过282份人类专家评审（平均分3.23/10，6分阈内11.4%）和统一自动评审器SAR（平均分5.00/10，1.8%达6分）与之前系统（平均2.74/10）比较，FARS表现最佳。

**⚠️ 局限性**

局限性：强论文比例稀少；实验验证不足、方法不够充分；易出现诚信问题（数据不一致、引用虚假、实验设计缺陷）；依赖现有基础模型；缺乏人类监督导致质量不稳定。

---

## 461. Dilemmadata: On the Interoperability of Heterogeneous Roman Numeral Datasets

**arXiv ID:** 2606.31595 | [PDF](https://arxiv.org/pdf/2606.31595v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 462. Tone-Conditioned Curriculum Learning for Low-Resource Bantu Speech Recognition

**arXiv ID:** 2606.31642 | [PDF](https://arxiv.org/pdf/2606.31642v1)

**作者:** Kesego Mokgosi `[一作]` (Technological University Dublin), Thapelo Sindane `[通讯]` (University of Pretoria)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了一个基于语调条件的课程学习框架，用于训练六种南部班图语的 ASR 模型。

**💡 创新点**

创新点在于将 WER 与形态-语调特征结合的混合难度评分、门控语调适配器以及分阶段课程调度，显著提升低资源语音识别效果。

**🔧 技术方法**

采用 Whisper、W2V‑BERT 与 MMS 三大基础模型，并引入门控语调适配器、混合难度评分机制和分阶段课程训练策略。

**📊 数据集**

使用社区录制的 Swivuriso/za‑african‑next‑voices 语料库作为训练集，并在 NCHLT 语料库上进行跨域评估。

**📈 对比分析**

通过对 12 种模型配置（包括多语言、语调条件、课程调度等）在匹配集与 NCHLT 上进行 WER/CER/BERTScore 比较，平均 WER 在 23–28% 之间；W2V‑BERT 在 Nguni 语系上优于 Whisper，而 Whisper 在 Sotho‑Tswana 语系上更具优势。

**⚠️ 局限性**

限制包括单一模型难以覆盖所有语言、课程调度效果不一致、以及对不同录音环境的泛化能力有限。

---

## 463. Preserve the Hard, Regenerate the Rest: Uncertainty-Guided Synthetic Training Data Augmentation with Diffusion Models

**arXiv ID:** 2606.31603 | [PDF](https://arxiv.org/pdf/2606.31603v1)

**作者:** Nikolai Röhrich `[一作]` (XITASO GmbH), Tobias Huber `[通讯]` (XITASO GmbH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于预测熵的不确定性引导上下文合成方法：先用现有分割器计算每像素熵，聚合到类别级别后挑选最不确定的区域；在这些区域之外用Latent Diffusion Inpainting生成新上下文，随后用像素精确粘贴保留不确定像素，构造无标签噪声的合成样本并在其上微调分割器。

**💡 创新点**

创新点：1）把“信息量(不确定性)”作为补丁选择轴，抛弃传统的前景/背景规则；2）只在不确定像素之外补丁，保证原标签完全有效；3）通过像素精确粘贴与ignore‑mask实现无标签噪声；4）可迭代循环，形成主动学习式的增强流程。

**🔧 技术方法**

核心技术包括：像素级预测熵计算与类别熵聚合、贪心类选择、SDXL‑Inpaint（latent diffusion）生成补丁、像素精确粘贴、ignore‑mask训练、DINOv2‑ViT 线性解码分割器、AdamW 与余弦学习率调度。

**📊 数据集**

实验数据集：Cityscapes、UAVID、BDD100K。

**📈 对比分析**

与无合成、实例增广（Instance Augmentation）和简单背景增广（Simple Background Augmentation）三种基线对比；在三大数据集上均显著提升 mIoU：Cityscapes 10% 训练从69.6提升到72.24 (+2.64)，UAVID 100% 从60.31提升到63.99 (+3.68)，BDD 10% 从59.60提升到61.17 (+1.57)。稀有/安全关键类别（bus、train、truck、moving_car等）提升更大。

**⚠️ 局限性**

限制：需要基线分割器每次迭代计算熵，依赖高质量的 diffusion inpainter；主要计算成本集中在 40 步 SDXL inpainting；在 diffusion priors 弱的领域（如医学影像）效果可能减弱；对比其他主动学习方法仍需进一步评估。

---

## 464. Orienting Unrooted Binary Networks Faster: Focus on the Generator

**arXiv ID:** 2606.31597 | [PDF](https://arxiv.org/pdf/2606.31597v1)

**作者:** Jannik Schestag `[一作]` (Delft University of Technology), Norbert Zeh `[通讯]` (Dalhousie University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了将无根网络定向为特定类有根系统发育网络的问题，并提出了两种改进的参数化算法框架。

**💡 创新点**

创新点在于利用网络生成器的生成树枚举和重叠节点定位技术，将原问题的核心化简为可枚举的结构，从而显著降低 FPT 运行时间。

**🔧 技术方法**

主要技术包括生成器的生成树枚举、分支猜测、重叠节点位置猜测以及对不同网络类的多项式后处理。

**📊 数据集**

文中未给出具体实验数据集，主要以理论复杂度分析为主。

**📈 对比分析**

通过理论复杂度比较，本文的算法在多种网络类上将运行时间从 1024^ℓ 降至 5.3334^ℓ、10.6667^ℓ 或 12.2071^ℓ，表现出显著的性能提升。

**⚠️ 局限性**

局限性包括仍然属于指数级时间复杂度，且对高层级网络的可扩展性尚未在实验中验证。

---

## 465. Comparative Analysis of Machine Learning based Intrusion Detection in Realistic IoT Networks

**arXiv ID:** 2606.31594 | [PDF](https://arxiv.org/pdf/2606.31594v1)

**作者:** Rana Alharbi `[一作]` (Newcastle University), Chuadhry Mujeeb Ahmed `[通讯]` (Newcastle University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文利用Gotham2025数据集对六类IoT攻击进行入侵检测，比较了随机森林、XGBoost、逻辑回归、朴素贝叶斯和深度神经网络五种机器学习模型的表现。

**💡 创新点**

创新点在于首次在最近的Gotham2025大规模IoT数据集上进行跨模型比较，并对数据预处理与特征工程做了统一、可复现的实现。

**🔧 技术方法**

使用了随机森林、XGBoost、逻辑回归、朴素贝叶斯以及全连接深度神经网络，并采用标准的准确率、精确率、召回率和F1分数进行评估。

**📊 数据集**

采用Gotham2025数据集，该数据集包含78台仿真IoT设备产生的多协议（MQTT、CoAP、RTSP）网络流量和多种攻击场景。

**📈 对比分析**

通过单模型评估和综合对比，随机森林在宏观F1得分0.99、准确率100%上优于其他模型，XGBoost次之，逻辑回归和朴素贝叶斯性能显著下降。

**⚠️ 局限性**

局限包括对监督学习的依赖导致难以检测未见攻击、数据严重不平衡影响少数类识别，以及高计算成本不适合资源受限的IoT设备。

---

## 466. Evil Spectra: How Optimisers can Amplify or Suppress Emergent Misalignment

**arXiv ID:** 2606.31591 | [PDF](https://arxiv.org/pdf/2606.31591v1)

**作者:** Jason R. Brown `[一作]` (University of Cambridge), Lev McKinney `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统地评估了不同优化器对LLM微调产生的emergent misalignment（EM）程度的影响，并探究了训练损失、优化器以及LoRA适配器奇异值分布与对齐表现之间的关系。

**💡 创新点**

发现优化器是影响EM最重要的因素，且优化器决定的残差几乎能解释训练损失与对齐之间的全部差异；进一步证明LoRA适配器的奇异值分布越平坦，对齐越好，并提出通过谱正则化显著缓解EM。

**🔧 技术方法**

使用LoRA微调、log训练损失回归、损失-对齐曲线、训练动态追踪、奇异值分解、谱正则化等技术。

**📊 数据集**

使用四个EM数据集：不安全代码、错误医疗建议、危险财务建议、极限运动建议，总计约24,000条单轮聊天样本。

**📈 对比分析**

通过GPT‑4o评估142个对齐问题（过滤一致性低于50的答案），比较不同优化器的误对齐率；结果显示7倍差异；谱正则化后，Adam和Lion等适配器的对齐率恢复到基线水平，训练损失仅略增。

**⚠️ 局限性**

局限性包括：仅使用LoRA而非全参数微调，未检验对其他模型/规模、不同评测方法的泛化；优化器使用默认超参，未分离权重衰减/动量的影响；谱正则化的λ值未进行系统调优；Lion在小批量设置下表现不佳。

---

## 467. Histogram-constrained Image Generation

**arXiv ID:** 2606.31683 | [PDF](https://arxiv.org/pdf/2606.31683v1)

**作者:** Haoming Liu `[一作]` (New York University Shanghai), Hongyi Wen `[通讯]` (New York University Shanghai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于直方图约束的图像生成方法 HIG，通过在扩散模型采样过程中插入最优输运（OT）引导变换，实现对像素或潜在离散 token 分布的精确控制。

**💡 创新点**

创新点：1）将直方图约束形式化为 OT 最优化问题，保证目标分布的完全满足与最小干预；2）在采样阶段进行训练‑free 的显式变换，兼顾全局一致性与局部细节；3）将该框架扩展到信息嵌入和潜在空间直方图匹配，展示高容量隐写与语义控制能力。

**🔧 技术方法**

核心技术包括：扩散模型（Latent Diffusion / DiT）、VAE 编码/解码、最优输运（POT）求解、单/多选 binning 方案、软提示调优（Prompt Tuning）与 LLM（Llama‑3.1‑8B）结合。

**📊 数据集**

使用的主要数据：Civitai 上的 300 条文本提示与 300 张参考图像；GitHub Markdown 代码片段（多语言文本、代码、URL）用于信息嵌入；此外在实验中采样 1024² 分辨率图像，采用 VQ‑GAN、TokenFlow、TiTok 等不同 tokenizer 的代码书写。

**📈 对比分析**

与 LoRA+ControlNet+IP、PixelShuffler、MPGD、GPT‑4o‑Image、InST、StyleID、StyleShot 等基线进行对比；评价指标包括 HistKL（直方图一致性）、CLIP‑Score（提示符合度）与 LAION‑Aesthetics（视觉质量）。实验表明 HIG 在 HistKL 方面最优，CLIP‑Score 与 Aesthetics 亦高于大多数基线；推理时间与 LoRA/ControlNet 相当，显著低于复杂风格迁移方法。

**⚠️ 局限性**

局限性：1）OT 采样策略随机，易产生局部颜色跳跃与视觉伪影；2）对 JPEG 压缩、随机缩放等变形较为敏感，信息嵌入鲁棒性下降；3）未考虑空间局部性与结构信息，未来可引入更细粒度或软化的变换；4）对高分辨率或更大直方图维度的求解仍需进一步优化。

---

## 468. Sparsity-Inducing Divergence Losses for Biometric Verification

**arXiv ID:** 2606.31664 | [PDF](https://arxiv.org/pdf/2606.31664v1)

**作者:** Dimitrios Koutsianos `[一作]` (Athens University of Economics and Business), Themos Stafylakis `[通讯]` (Athens University of Economics and Business)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种名为 Q-Margin 的 α-divergence 损失，通过在参考测度 q 上引入概率边界来实现显式的判别边距，从而提升人脸和说话人验证的低误识率性能。

**💡 创新点**

创新点在于将传统的几何边距改为概率边距，利用非均匀参考测度直接对目标类别施加惩罚，并证明在 α→1 时退化为 CosFace；同时利用 α>1 的稀疏后验实现训练效率提升。

**🔧 技术方法**

采用 α-divergence 框架（尤其是 α-softargmax）、迭代二分搜索求解归一化因子、ResNet-100/34 backbone、SGD 训练、以及对参考测度 q 的概率编码。

**📊 数据集**

主要使用 WebFace42M（≈2M 头像）进行训练，评测在 IJB-B、IJB-C 及 VoxCeleb1-H 上；此外使用 LFW、CFP-FP、AgeDB-30、CALFW、CPLFW 作为开发集。

**📈 对比分析**

与 ArcFace、CosFace、AdaFace、PartialFC、EntMax、SparseMax 等基线对比，Q-Margin 在低 FAR（10⁻⁴、10⁻⁵）下实现了与 PartialFC 相当甚至略优的 TAR，且在大规模数据上训练速度仅略低（≈5%）于 ArcFace。

**⚠️ 局限性**

局限包括：需要额外的 α 超参数且与尺度 s 互相耦合；在大规模实验中仅进行单次训练，缺乏置信区间；对 α、s、m 的调参仍需经验；并且在极端稀疏后验的实现中仍存在计算瓶颈。

---

## 469. DynFly: Dynamic-Aware Continuous Trajectory Generation for UAV Vision-Language Navigation in Urban Environments

**arXiv ID:** 2606.31654 | [PDF](https://arxiv.org/pdf/2606.31654v1)

**作者:** Wen Jiang `[一作]` (Beijing Institute of Technology), Huaping Liu `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了DynFly，一个将高层视觉‑语言导航意图转化为连续可执行无人机轨迹的动态感知生成框架。

**💡 创新点**

创新点在于引入B‑spline控制点空间的连续轨迹表示、基于流匹配的Spline‑DiT生成器以及针对位置、速度、加速度、航向和目标对齐的动态感知监督。

**🔧 技术方法**

使用Qwen2.5‑3B视觉‑语言前端、EVA ViT‑G视觉编码器、Spline‑DiT（Transformer）流匹配生成器和B‑spline解码；训练采用流匹配、位置、速度、加速度、航向一致性及目标对齐损失。

**📊 数据集**

在OpenUAV UAV‑VLN基准数据集上进行实验，包括Train、Test Seen、Test Unseen Map/Object等子集。

**📈 对比分析**

与多种现有UAV‑VLN方法（TravelUAV、LongFly、SpatialFly等）对比，DynFly在Test Unseen Full/Hard/Easy等子集上提升了NDTW、SDTW、SR、OSR、SPL等指标，且导航误差（NE）显著降低。

**⚠️ 局限性**

主要限制包括仅在仿真环境评估、仅处理短时轨迹生成，未与低层飞控或长时导航融合，模型尺寸较大且实时推理效率待提升，真实无人机平台的验证尚未完成。

---

## 470. ECHO: Prune to act, trace to learn with selective turn memory in agentic RL

**arXiv ID:** 2606.31650 | [PDF](https://arxiv.org/pdf/2606.31650v1)

**作者:** Zijun Xie `[一作]` (Peking University), Zeyu Chen `[通讯]` (Baidu Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 ECHO 框架，利用选择性回合记忆在长时序语言代理中实现可追溯的上下文管理与学习。

**💡 创新点**

创新点在于将每个完成的交互回合压缩为带源索引的记忆，既实现高效的上下文重建，又通过同一源索引进行信用分配，解决了历史崩塌与学习可追溯性的双重瓶颈。

**🔧 技术方法**

采用了强化学习（GRPO）与可追溯信用分配、源索引记忆选择、Token‑级信用掩码等技术，并基于 Qwen3 等大模型进行训练。

**📊 数据集**

主要使用 BrowseComp‑Plus 长时序工具使用 QA 基准进行训练与评估，并在 Multi‑Objective QA、CodeGym、LoCoBench‑Agent、GAIA、HLE、Frames 等零样本基准上验证泛化。

**📈 对比分析**

与 GRPO 与 SUPO 轮播摘要基线相比，ECHO 在 BrowseComp‑Plus 的 43.4% 通过率（相较 28.9%/36.1%）同时减少了回合数和轨迹体积；在零样本基准上平均提升约 6–7%。

**⚠️ 局限性**

主要限制是信用分配仅采用最终回溯近似，未递归追踪所有依赖链；实验聚焦文本工具使用，未覆盖 GUI/体感/多代理场景，且记忆索引与选择的计算开销相对较大。

---

## 471. Technical Report of RoboSpatial Challenge at CVPR 2026: Selective Reasoning Activation and Reference-Frame Disambiguation for Embodied Spatial Reasoning

**arXiv ID:** 2606.31645 | [PDF](https://arxiv.org/pdf/2606.31645v1)

**作者:** Yuxiang Xie `[一作]` (Harbin Institute of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 RoboSpatialBrain，用于在 RoboSpatial Challenge 中提升 VLM 的空间推理能力，最终获得第一名。

**💡 创新点**

创新点在于：① 在推理需求任务（兼容性和上下文）上强制添加  前缀并结合后置提示，以激发模型的“思考”行为；② 为上下文任务构建显式参考框架重定向管道，先用轻量级 LM 解析目标对象并将对象中心的方向映射为相机中心，消除方向歧义；③ 通过对比实验阐明 fine‑tuning 对推理与提示交互的负面影响。

**🔧 技术方法**

主要技术包括：RoboBrain2.5‑8B‑NV VLM、Qwen3.5‑2B 轻量级 LM、强制前缀激活、后置提示、参考框架重定向、任务类型路由。

**📊 数据集**

使用 RoboBrain2.5 预训练的数据集（含 2D/3D  grounding、物体间关系、深度定位）以及 EmbodiedScan 用于兼容性任务的数据构造。

**📈 对比分析**

通过多种组合（前缀+后置提示、仅前缀、仅后置提示、无干预）在三类任务上进行 ablation；结果显示在兼容性任务中前缀+后置提示提升 43.8pp，整体平均成功率达 80.9%，在 RoboSpatial‑Home 赛道中排名第一。

**⚠️ 局限性**

局限性包括：① fine‑tuning 在有限多样性数据上导致灾难性遗忘，无法提升整体性能；② 参考框架重定向依赖轻量级 LM 与手工映射表，可能在更复杂场景中失效；③ 仍未完全解决 VLM 对空间关系的普适性和可解释性问题。

---

## 472. LiteMatch: Lightweight Zero-Shot Stereo Matching via Cost Volume Stabilization

**arXiv ID:** 2606.31636 | [PDF](https://arxiv.org/pdf/2606.31636v1)

**作者:** Md Raqib Khan `[一作]` (University of Dublin), Subrahmanyam Murala `[通讯]` (University of Dublin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 LiteMatch，一种轻量化、零样本迁移的立体匹配框架。

**💡 创新点**

通过双分支编码器和成本体积一致性损失实现成本体积稳定，并用轻量级迭代/非迭代细化模块，完全摆脱 3D 卷积和大规模先验。

**🔧 技术方法**

交叉视角注意力编码器、频率感知高频编码器、FFT 高频滤波、成本体积一致性损失（CVC‑Loss）、Transformer‑based 细化网络。

**📊 数据集**

仅在合成 Scene Flow 训练，零样本在 KITTI 2012/2015、Middlebury‑F、ETH3D、DrivingStereo 上评估。

**📈 对比分析**

与 RAFT‑Stereo、IGEV、MonSter 等 SOTA 方法对比，LiteMatch 在 3.36M 参数的基础模型下，零样本 EPE/D1 仍优于多数方法；迭代版仅 9.58M 参数即可逼近甚至超越大规模先验模型，同时推理速度高达 22 FPS。

**⚠️ 局限性**

仍依赖合成数据预训练，对极端光照或纹理稀疏场景表现略逊；在极大视差范围或动态场景下的鲁棒性尚未充分验证。

---

## 473. Robust Autonomous UAV Landing on Maritime Platforms via Multimodal Agentic AI and Active Wave Compensation

**arXiv ID:** 2606.31613 | [PDF](https://arxiv.org/pdf/2606.31613v1)

**作者:** Francisco S. Neves `[一作]` (Faculty of Engineering, University of Porto), Andry M. Pinto `[通讯]` (Faculty of Engineering, University of Porto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种解耦的多车队自动化海上平台无人机降落框架，利用配备3‑RPU结构的无人水面平台（USV）进行波浪补偿，并通过无人机（UAV）内部的多模态强化学习策略完成最终降落；

**💡 创新点**

创新点在于：①将降落平台的机械补偿与无人机的降落策略完全解耦，消除跨平台通信需求；②采用Soft Actor‑Critic（SAC）训练的3‑RPU控制器实现子度级的波浪抑制；③设计基于Transformer的多模态感知‑决策网络，使无人机在视觉、热感、LiDAR信息融合下实现鲁棒的降落；

**🔧 技术方法**

使用技术包括：深度强化学习（SAC用于USV稳定，强化学习策略用于UAV降落）；3‑RPU并行机构与PID辅助控制；Transformer‑风格多模态感知模块；Gazebo仿真环境中的Tessendorf FFT波浪合成；

**📊 数据集**

主要数据来自高保真Gazebo仿真：合成波浪场、人工生成的视觉/热/LiDAR传感器数据；未使用公开真实海上数据集；

**📈 对比分析**

通过15次仿真试验（平静、中等、剧烈三种海况）评估，系统实现100%降落成功率；平均波浪抑制效果为87.9%，降落平台倾斜角保持在1°以内占比96%，平均降落误差<0.6 m，任务时长在最恶劣海况下可达97 s；与未补偿对比显著提升了安全性与可靠性；

**⚠️ 局限性**

局限性包括：仅在仿真环境中验证，缺乏真实海况和光照变化下的实测；GPS位置保持对波浪影响不够鲁棒；缺少双向通信与状态共享；未来需部署实地测试并改进定位与通信模块。

---

## 474. Power law scaling for classification accuracy in physical neural networks

**arXiv ID:** 2606.31588 | [PDF](https://arxiv.org/pdf/2606.31588v1)

**作者:** Andrei V. Ermolaev `[一作]` (Universite Marie et Louis Pasteur), Daniel Brunner `[通讯]` (Universite Marie et Louis Pasteur)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过提出Hotelling Trace Criterion（HTC）对多种物理神经网络（如高非线性光纤、垂直腔面发射激光器和耦合非线性振荡器网络）在分类任务中的状态空间可分离度进行量化，并证明HTC与分类误差呈幂律关系，可预测网络性能。

**💡 创新点**

创新点在于：①首次将HTC作为任务相关的分离度指标用于物理神经网络；②发现不同物理实现的网络在同一任务上可归一到同一幂律曲线，证明了与硬件无关的统一性能尺度；③利用HTC进行层级诊断，揭示梯度训练在多层网络中资源分配不均的现象。

**🔧 技术方法**

使用的技术包括：光纤非线性传播的广义非线性薛定谔方程数值求解、LA-VCSEL的空间光调制与成像、耦合非线性振荡器网络的欧拉积分、ELM训练的岭回归、反向传播训练，以及HTC计算中的散度矩阵正则化。

**📊 数据集**

主要使用的数据集是MNIST和Fashion‑MNIST手写数字分类数据集，实验与模拟各提供了数千张图像，完整训练集共7万张。

**📈 对比分析**

通过将HTC与测试MSE、分类准确率绘制在对数坐标系下，发现三种物理实现在同一任务上均符合高相关系数（r≈‑0.99） 的幂律关系；在不同任务（MNIST vs. Fashion‑MNIST）下幂律指数不同，验证了HTC可作为性能预测与比较工具，且无需完整训练即可估计最终精度。

**⚠️ 局限性**

局限性在于：HTC目前仅适用于分类任务，难以直接推广到回归问题；对超大规模动态范围的物理系统验证仍待扩展；此外，尚未建立HTC的理论上限或与任务无关的维度上限的统一框架。

---

## 475. Calibration, Not Compilation: Detecting and Repairing Misspecified Probabilistic Programs Written by Language Models

**arXiv ID:** 2606.31630 | [PDF](https://arxiv.org/pdf/2606.31630v1)

**作者:** Jian Xu `[一作]` (RIKEN iTHEMS), Qibin Zhao `[通讯]` (RIKEN AIP)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了利用贝叶斯工作流（如后验预测检查、模拟校准、采样器诊断和留一预测密度）作为验证器来发现和修复大型语言模型（LLM）生成的概率程序中的统计错误，并与传统单元测试进行对比。

**💡 创新点**

创新点在于提出“校准oracle”作为判定概率程序正确性的标准，并展示其在检测代码不可见的模型误设（如错误分布、过度/欠度离散、参数化错误）方面显著优于单元测试；同时证明该oracle可以作为有效的反馈信号，提升LLM在修复这些错误时的成功率，且单元测试反馈甚至可能阻碍修复。

**🔧 技术方法**

技术包括：NumPyro等PPL中的后验预测检查（PPC）、模拟校准（SBC）、采样器诊断（R̂、divergences、ESS）、留一预测密度（lppd）；构建了14类误设、10类模型族的200实例检测基准；设计LLM修复循环；对比不同反馈（无、单元测试、诊断）和不同LLM写手（GPT、Claude、DeepSeek等）。

**📊 数据集**

使用的主要数据集是人工构造的基准数据（10个模型族，每个误设10个随机实例，共200实例），以及由LLM在中性自然语言简述下自动生成的真实程序（共45个任务，9个写手，每个任务5个实例，共405个程序）。

**📈 对比分析**

通过与单元测试oracle、无反馈、LLM-as-judge、贝叶斯检查列表、数据摘要自检等基线比较，校准oracle在检测中取得88%准确率（相较于0%），在修复中提高了从33%到92%（GPT-5.1）或从75%到100%（Claude），并在真实程序实验中修复率达84%高于其他基线。

**⚠️ 局限性**

局限性包括：检测试验仅覆盖低维经典模型，未扩展到高维或深度结构；SBC在大模型上计算昂贵；诊断只能指出误设类型，不能保证能被LLM修复；在模型难以收敛时，诊断可能被混淆为推断失败；基准依赖人工标注且可能无法完全代表真实世界模型错误。

---

## 476. Hybrid Topological Data Analysis and LSTM Networks for Enhanced Network Intrusion Detection Using CIC-IDS2017 Dataset

**arXiv ID:** 2606.31619 | [PDF](https://arxiv.org/pdf/2606.31619v1)

**作者:** Amar Jeet `[一作]` (Birla Institute of Technology), Dinesh Kumar `[通讯]` (Birla Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了结合拓扑数据分析与LSTM的混合模型，用于检测网络入侵。

**💡 创新点**

首次将持续同调提取的Betti曲线与LSTM时序特征融合，实现多尺度拓扑与深度时序建模的协同检测。

**🔧 技术方法**

采用Vietoris–Rips构建持续同调、Betti曲线、LSTM网络、MLP融合层、交叉熵损失与Adam优化器。

**📊 数据集**

使用CIC‑IDS2017数据集，包含2.8M流量样本、77个特征及14类攻击。

**📈 对比分析**

与TDA+RF、Isolation Forest、SVM等基线在5折交叉验证下对比，混合模型AUC与F1均达到1.000，显著优于所有基线。

**⚠️ 局限性**

计算开销大、对窗口与参数设置敏感、仅在CIC‑IDS2017验证、实时部署仍受限。

---

## 477. Learning Structurally Consistent Representations for Multi-View Radar Semantic Segmentation

**arXiv ID:** 2606.31609 | [PDF](https://arxiv.org/pdf/2606.31609v1)

**作者:** Ali Zia `[一作]` (La Trobe University), Abdul Rehman `[通讯]` (GIFT University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种统一的多视角雷达语义分割框架HyperRadar，利用可学习的超图对单视角特征进行高阶结构化；

**💡 创新点**

创新点在于通过高阶超图实现多雷达返回的集群关系建模，并使用无对应的非平衡最优传输（UOT）在不同雷达投影间实现分布式对齐；

**🔧 技术方法**

核心技术包括可学习超图神经网络、带熵正则的UOT Sinkhorn求解器、适应性注意力融合、交叉熵+Dice损失与视角一致性约束；

**📊 数据集**

实验使用CARRADA（多类）和RADIal（单视/检测）两个公开自动驾驶雷达基准数据集；

**📈 对比分析**

与TransRadar、PeakConv、TMVANet等强基线对比，HyperRadar在CARRADA RD视图实现63.8% mIoU、RA视图44.4% mIoU，RADIal 83.4% mIoU，均比前者高出约1.7-2.3点，且在检测AP/AR上也取得领先；

**⚠️ 局限性**

局限性包括仅处理2D投影而非完整RAD体素，跨视角约束仅利用共享范围维度，缺乏显式时间一致性，且仅在两个数据集上验证。

---

## 478. Digital signature schemes based on code equivalence and syndrome decoding from restricted errors

**arXiv ID:** 2606.31601 | [PDF](https://arxiv.org/pdf/2606.31601v1)

**作者:** Sarah Arpin `[一作]` (Virginia Tech), Gretchen L. Matthews `[通讯]` (Virginia Tech)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并比较了三种基于纠错码的数字签名方案——CROSS、LESS 和 MEDS，阐述了其设计思想、实现细节与性能评估。

**💡 创新点**

创新点在于将受限综合译码问题与码等价问题分别映射为 sigma 协议，再通过 Fiat‑Shamir 变换得到安全且量化可比的签名方案，首次系统性比较了三者在 NIST 安全级别下的性能。

**🔧 技术方法**

使用了受限综合译码问题、码等价问题、sigma 协议、Fiat‑Shamir 变换、并行重复、伪随机生成器、常量权重哈希、以及承诺可恢复等技术。

**📊 数据集**

未使用传统数据集，而是基于 NIST 第二轮候选参数集（不同安全级别的参数组合）进行实验与比较。

**📈 对比分析**

通过测量签名时间（单位 Mcycles）和公共密钥+签名大小（字节），在 NIST 安全级别 I、III、V 下对 CROSS、LESS、MEDS 进行比较，结果显示：CROSS 最快但签名尺寸最大，LESS 次之，MEDS 在签名尺寸上最为庞大。

**⚠️ 局限性**

主要局限包括：安全性仍基于受限综合译码与码等价难度的假设，缺乏对量化安全性的实证证明；CROSS 的签名尺寸大、MEDS 的密钥尺寸大，导致实际部署时存储与通信成本高；缺少标准化与广泛实用案例。

---

## 479. SpikeLogBERT: Energy-Efficient Log Parsing Using Spiking Transformer Networks

**arXiv ID:** 2606.31781 | [PDF](https://arxiv.org/pdf/2606.31781v1)

**作者:** Thuan Bui `[一作]` (Swinburne Vietnam), Cong-Kha Pham `[通讯]` (University of Electro-Communications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SpikeLogBERT，一种利用脉冲神经网络实现能耗低的日志解析框架。

**💡 创新点**

①采用稀疏事件驱动的脉冲Transformer实现高效计算；②通过多目标知识蒸馏将BERT教师的语义表示迁移给SNN学生；③提供理论能耗基准，展示SNN相较ANN降低高达62.6×能耗且准确率更高。

**🔧 技术方法**

脉冲神经网络（LIF神经元、Spiking Self‑Attention）、Transformer结构、嵌入对齐/表示对齐/预测对齐的多目标知识蒸馏、理论能耗估算（MAC vs AC）。

**📊 数据集**

HDFS日志数据集。

**📈 对比分析**

与NuLog、LogPPT等ANN模型在HDFS上比较，使用解析准确率（PA）和理论能耗（mJ）衡量；SpikeLogBERT在PA 99.997%上优于LogPPT的90.2%，且理论能耗比LogPPT低62.6×。

**⚠️ 局限性**

仅在单一数据集上验证；能耗估算基于理论模型，未在实际neuromorphic硬件上验证；训练仍需GPU，整体实现仍非完全低功耗。

---

## 480. The Cooperation Ceiling: Extrinsic Population Dynamics and the Intrinsic Escape

**arXiv ID:** 2606.31740 | [PDF](https://arxiv.org/pdf/2606.31740v1)

**作者:** Harry Foster `[一作]` (Cardiff University), Sebastian Krapohl `[通讯]` (University of Amsterdam)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在异质群体中，基于进化博弈理论研究合作的出现与维持，提出了“外在动态”和“内在动态”两类群体更新规则，并证明在外在动态下，任何社会困境中的合作概率上限为1/2；通过大规模参数扫描与公共物品游戏验证该上限并显示内在动态可突破；

**💡 创新点**

首次将外在与内在更新规则在异质群体中的表现进行系统比较，并给出严格的外在动态合作上限定理；

**🔧 技术方法**

利用马尔可夫链理论、随机单调性与蒙特卡洛模拟相结合的方法，构造了多种更新规则的转移矩阵并求解稳态；

**📊 数据集**

使用自行生成的合成数据集：N=2–8，收益率r∈[0.5,3N/2]，贡献上限M∈[N,4N]，突变率μ∈{0.001,0.005,0.05,0.1}，选择强度ε与选择强度β按网格取值；

**📈 对比分析**

通过对比纯外在动态（Moran、Fermi）与纯内在动态（期望、内省）的合作比例p_C，发现外在动态始终不超过0.5，而内在动态在r>N时可显著超越此阈值；在N=100的大规模模拟中结论保持一致；

**⚠️ 局限性**

仅考虑两行动游戏和贡献异质性，未探讨多行动情形、混合更新规则或异质收益率的影响，且在大规模分析中主要依赖蒙特卡洛逼近，精度受样本量限制。

---

## 481. Rhythm-Structured Predictive Learning for Remote Photoplethysmography

**arXiv ID:** 2606.31736 | [PDF](https://arxiv.org/pdf/2606.31736v1)

**作者:** Ba-Thinh Nguyen `[一作]` (VNU University of Engineering and Technology), Thanh-Ha Le `[通讯]` (VNU University of Engineering and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 RhythmJEPA，一种基于自监督 joint‑embedding 预测学习的远程光电容积脉搏（rPPG）框架。

**💡 创新点**

创新点包括：① 预测教师嵌入而非像素重建，促使模型关注潜在生理动态；② 设计循环节律状态规划器（CRSP）和双序列 Mamba 编码器（DOM），利用周期一致性扫描提升对跨周期信息的利用；③ 引入轻量化空间脉冲混合器（SPM）实现高效空间建模。

**🔧 技术方法**

采用的技术包括：JEPA 自监督预训练、Mamba 状态空间序列模型、周期一致性约束与状态平衡正则、深度卷积混合层、双序列门控融合等。

**📊 数据集**

使用的数据集为 PURE、UBFC‑rPPG 与 MMPD 三个公开 rPPG 基准。

**📈 对比分析**

与多种基准方法（如 DeepPhys、PhysNet、EfficientPhys 等）在 MAE、MAPE、RMSE 与 Pearson 相关系数上进行内外域比较，RhythmJEPA 在所有指标上均优于先前方法，尤其在跨域测试中表现更为稳健。

**⚠️ 局限性**

局限性：节律状态仍为隐式表示，缺乏直接临床解释；对极端运动、光照变化的鲁棒性尚待进一步提升。

---

## 482. Seeing Is Not Sharing: Some Vision-Language Models Overestimate Common Ground in Asymmetric Dialogue

**arXiv ID:** 2606.31719 | [PDF](https://arxiv.org/pdf/2606.31719v1)

**作者:** Nan Li `[一作]` (Utrecht University), Massimo Poesio `[通讯]` (Utrecht University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究视觉语言模型在信息不对称的协作对话中判断解释匹配的能力，利用MapTask数据集构造多种地图信息与对话上下文条件进行实验。

**💡 创新点**

提出面向解释匹配的评价方法，发现模型因地图内容而过度预测对齐，并区分视觉与文本信息的内容效应。

**🔧 技术方法**

采用零样本推理的Qwen3‑VL和Gemma‑3视觉语言模型，系统控制对话窗口与地图信息的不同组合，进行分类评估与校准、状态级与参考链分析。

**📊 数据集**

使用HCRC MapTask语料的13,077条带有视角注解的参考表达，记录说话者与听话者对同一表达的具体指向。

**📈 对比分析**

在基准和全条件网格下对比不同模型、不同地图信息与文本窗口，结果显示地图访问提升整体F1但引入“过度对齐”偏差，Qwen3‑VL‑8B性能最佳。

**⚠️ 局限性**

仅在单一模型与单一领域进行过度对齐分析，缺乏交互式评估，且数据集有限难以推广至其他信息不对称场景。

---

## 483. Nonlinearity-Aware LoRA: Structured Gate Adaptation under Low-Rank Constraints

**arXiv ID:** 2606.31717 | [PDF](https://arxiv.org/pdf/2606.31717v1)

**作者:** Shuai Yuan `[一作]` (Beijing Institute Of Technology), Rui Mao `[通讯]` (Shenzhen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 NA-LoRA，一种在训练时对自门控 FFN 的低秩适配进行非线性感知的门控更新方法；

**💡 创新点**

通过识别 LoRA 在自门控 FFN 中的“选择失配”问题，提出基于门敏感通道掩码与有效同质性步长缩放的两项训练时控制，实现对门激活选择机制的精准调节；

**🔧 技术方法**

使用门激活的选择视角、有效同质性分析、时间重要性掩码、步长缩放以及标准 LoRA 的低秩适配；

**📊 数据集**

在 Llama‑3.1‑8B 与 Llama‑2‑7B 的对话、数学推理和代码生成任务（MT‑Bench、GSM8K、HumanEval）以及 T5‑GLUE、CLIP‑ViT‑B/16 的跨模态迁移基准上进行评测；

**📈 对比分析**

与全微调以及多种 LoRA 变体（rsLoRA、AdaLoRA、DoRA、LoRA‑+、PiSSA、LoRA‑GA、LoRA‑Pro、GoRA）进行对比，NA‑LoRA 在 rank‑8 下在所有指标上均优于标准 LoRA，并在多数任务上匹敌或超过强大 PEFT 基线，部分任务甚至接近全微调性能；

**⚠️ 局限性**

局限性包括：需自门控 FFN 结构且激活特定的有效同质性区间存在；在 ReLU 或 GELU 等激活下步长缩放失效；门掩码依赖梯度估计，易受噪声影响。

---

## 484. Semantic-Aware Multiple Access via Spatial Redundancy Exploitation for Uplink-Dominant 6G Use Cases

**arXiv ID:** 2606.31715 | [PDF](https://arxiv.org/pdf/2606.31715v1)

**作者:** Hamidreza Mazandarani `[一作]` (Ruhr University Bochum), Onur Günlü `[通讯]` (TU Dortmund)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于语义的多访问方案，通过车辆间的V2V观察共享与基于Patch的选择，实现上行6G车联网的冗余压缩与重建。

**💡 创新点**

创新点在于将多访问视为感知控制问题，利用重叠视野中的空间冗余，并采用两阶段V2V共享+语义优先传输的实时框架；同时通过GenAI视角变换实现缺失Patch的高质量重建。

**🔧 技术方法**

使用的技术包括：图像Patch分块、FoV优先权估计、SSIM/JEPA嵌入相似度、匈牙利算法匹配、集中式时间槽调度、条件扩散模型/变分自编码器/Transformer生成式视角重建，以及LoFTR几何对齐。

**📊 数据集**

数据集为CARLA仿真环境下的40辆车密集城市路段，10 s共10 000个时间槽，每秒10帧（1280×720），每帧划分为10×10 Patch。

**📈 对比分析**

通过与传统语义传输基线、无视角变换下的下界以及完美视角变换上界对比，实验显示该方法在中等资源场景下显著提升满足感知质量阈值的用户比例，且共享时长与上行效率存在典型的单峰关系。

**⚠️ 局限性**

局限性包括：依赖V2V链路质量与同步、对动态物体与遮挡处理不完善、视角变换精度受标定误差影响、生成式重建可能产生幻觉，以及未覆盖多跳深度源信道编码等未来挑战。

---

## 485. Look But Don't Touch with Sparse Autoencoders for Unlearning in Diffusion Models

**arXiv ID:** 2606.31699 | [PDF](https://arxiv.org/pdf/2606.31699v1)

**作者:** Enrico Cassano `[一作]` (University of Turin), Stephan Alaniz `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统评估了稀疏自编码器（SAE）在扩散模型中的对象消除性能，并提出仅利用SAE进行概念检测、通过替换嵌入（PER）来消除概念、避免了传统乘法干预导致的 OOD 激活与视觉伪影问题；

**💡 创新点**

创新点在于首次揭示SAE直接干预会产生 OOD 激活并产生严重视觉伪影，随后提出基于检测+替换的 PER 框架，既保持了概念检测的可解释性，又消除了对调参的需求并显著提升了图像质量与跨域保持；

**🔧 技术方法**

采用了 TopK SAEs / G-SAEs 进行概念特征学习，利用检测掩码定位目标概念，再用同一特征图内的非目标嵌入进行替换；实验中使用 Qwen2‑VL‑7B‑Instruct 自动评估伪影，HumanEval 进一步验证效果；

**📊 数据集**

使用 UnlearnCanvas（20 类目标 + 50 艺术风格）进行对象消除基准；对 SD v1.5、SDXL Turbo 两大扩散模型进行评估；对 I2P（NSFW）数据集进行敏感内容去除实验；对 UnlearnDiffAtk 进行对抗鲁棒性评估；

**📈 对比分析**

与 SAeUron、SAEmnesia、G‑SAE、SDXL‑Turbo SAE 等现有基线对比，PER 在 UA、IRA、CRA 上提升多达 10% 以上，AR（伪影率）从 57%–61% 降至 8%–22%，GA（综合指标）提升至 84% 以上，且不需要任何 γ‑搜索，计算成本显著降低；

**⚠️ 局限性**

局限性包括：需手工选择 padding 参数（默认 p=1）；对极端概念或极端提示可能仍出现轻微伪影；PER 仅适用于基于 SAE 的检测框架，无法直接推广到非稀疏自编码器的干预方法；

---

## 486. ShopX: A Foundation Model for Intent-to-Item Fulfillment in Agentic Shopping

**arXiv ID:** 2606.31693 | [PDF](https://arxiv.org/pdf/2606.31693v1)

**作者:** Jiacheng Chen `[一作]`, Yuning Jiang `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个统一意图到商品的基础模型，直接在语义ID空间完成检索、排序、组合与生成，实现模型原生的购物工作流。

**💡 创新点**

创新性在于把意图理解、执行规划和语义ID操作融合到同一模型，并提出混合全局‑局部语义ID和多教师对抗蒸馏+RL的训练流程，显著减少工具接口的损耗。

**🔧 技术方法**

技术包括语义ID构造（残差量化、VQ‑VAE）、多模态表示学习、Qwen3家族大模型、SID对齐、域持续预训练、监督微调、多教师在线蒸馏、RL奖励（LLM评判、排名、交叉ID）等。

**📊 数据集**

使用匿名淘宝生产日志（单轮/多轮）、淘宝商品快照（约1.2B条）以及公共基准（MMLU、CMMLU、IFEval、GPQA、MATH、GSM8K、MBPP等）。

**📈 对比分析**

在与工具中介式LLM‑代理基线（InteRecAgent、Chat‑REC、RecMind）相同的评测框架下进行对比，模型原生完成度在复杂/模糊请求、排名质量、约束遵循、反馈适配和多轮状态保持等维度均超过基线，整体性能提升显著。

**⚠️ 局限性**

局限包括对语义ID设计与验证的高工程成本、模型在某些通用知识/数学任务上略有退化、仍未实现完整轨迹级强化学习以及对淘宝域外的泛化能力有限。

---

## 487. Mesh BDF: Barycentric Dominance Field for 3D Native Mesh Generation

**arXiv ID:** 2606.31777 | [PDF](https://arxiv.org/pdf/2606.31777v1)

**作者:** Gaochao Song `[一作]` (University of Hong Kong), Shenghua Gao `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出并实现了Barycentric Dominance Field（BDF）作为三角网格表面连续标量场，用来桥接离散网格拓扑与连续扩散模型，实现原生网格生成；

**💡 创新点**

创新点在于将BDF视为内在纹理映射，利用其C0连续性和Lipschitz连续性，将网格拓扑转化为可在VAE与扩散框架中直接学习的连续信号，并通过解耦的VAE与流模型实现网格与PBR纹理的统一生成；

**🔧 技术方法**

采用Dual Contouring进行稀疏体素编码、洪水填充算法重建网格、BDF VAE（使用BCE损失）和BDF流模型（基于稀疏Diffusion Transformer）等技术；

**📊 数据集**

训练集使用Texverse 180k高质量3D资产，评估集为Toys4k数据集（低多边形与高多边形两份子集）；

**📈 对比分析**

与MeshSilksong、DeepMesh、MeshRipple等自回归模型以及Trellis.2进行对比，使用Chamfer Distance、Hausdorff Distance、Normal Consistency三项指标，BDF在所有度量上均优于基线，尤其在高多边形样本中表现更为稳定；

**⚠️ 局限性**

局限性在于BDF编码依赖Dual Contouring网格分辨率，极小面无法被捕捉，洪水填充算法在自相交三角形处易产生错误连接，需要更具分辨率无关性的编码方式。

---

## 488. Investigating LLM-Powered Dissenting Minority Support in Power-Imbalanced Group Decision-Making: Counterargument and Mediation as Intervention Strategies

**arXiv ID:** 2606.31762 | [PDF](https://arxiv.org/pdf/2606.31762v1)

**作者:** Soohwan Lee `[一作]` (UNIST), Kyungho Lee `[通讯]` (UNIST)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实验评估两种 LLM 驱动的少数派支持系统（AI 生成反驳与 AI 介入匿名信息），以提升权力不平衡小组决策中的心理安全与参与度。

**💡 创新点**

首次从实验角度揭示：①AI 生成反驳可提升满意度与氛围；②AI 介入匿名信息虽提高参与，却降低心理安全与满意度，形成“支持悖论”；③提供基于权力与身份的设计原则与伦理警示。

**🔧 技术方法**

使用 GPT‑4o 作为核心 LLM，构建多智能体架构（摘要、改写、对话、重复检测）实现实时生成反驳与匿名信息，辅以定制的提示与频率控制。

**📊 数据集**

数据集为 96 名韩国参与者，24 个 4 人小组（3 名“高级”成员 + 1 名“初级”成员），完成两项决策任务；记录文本交流、心理安全、参与度、满意度、认知负荷等自评与行为指标。

**📈 对比分析**

对比 Baseline、AIGC、AIMM 三种条件，使用稳健线性混合模型与 Bonferroni 校正。结果显示：AIGC 在心理安全、满意度、决策体验方面显著优于 Baseline；AIMM 在参与度上略有提升，但心理安全与满意度显著下降；整体表现表明不同 AI 介入方式对少数派体验产生相互冲突的效应。

**⚠️ 局限性**

局限包括：①实验室在线文本环境缺乏面对面非语言线索；②样本仅来自韩国，可能受高权力距离文化影响；③未测量参与者对权力感知与个体差异；④未进行事先功效分析；⑤匿名信息机制在真实组织中可能被滥用，未考虑长期部署与多文化场景。

---

## 489. Decoupling Trust in Byzantine CRDTs: Fine-grained Post-Compromise Handling without Breaking Causality

**arXiv ID:** 2606.31759 | [PDF](https://arxiv.org/pdf/2606.31759v1)

**作者:** Amos Brocco `[一作]` `[通讯]` (University of Applied Sciences and Arts of Southern Switzerland), Amos Brocco (University of Applied Sciences and Arts of Southern Switzerland)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过将身份信任与内容信任解耦，引入细粒度的更新过滤机制，实现了在 Byzantine CRDT 中的后期被篡改节点不破坏因果一致性的处理方法。

**💡 创新点**

创新点在于利用确定性重建（deterministic reconstruction）结合白名单/黑名单策略，在不进行后向失效的前提下，对单个增量块实施内容级信任判断，实现了前向、细粒度的信任演化。

**🔧 技术方法**

主要技术包括 Delta‑State CRDT（Melda）与其安全插件 melda‑sec、基于哈希的因果 DAG、签名验证、白名单/黑名单过滤，以及在重建阶段的可配置信任策略。

**📊 数据集**

论文未使用公开数据集，实验基于自定义模拟工作负载与随机生成的更新序列来评估方法。

**📈 对比分析**

通过与传统身份排斥方案（如 Blocklace、SRDT、Bounded CRDT）对比，证明在相同网络延迟与节点失效条件下，细粒度信任模型能保持因果一致性且通信/存储开销相近，且无需额外协调开销。

**⚠️ 局限性**

局限性包括：需要在各节点自行配置并同步信任策略，缺乏统一的策略协商机制；对每个增量块的信任判断增加计算成本；仅在确定性重建框架内有效，无法直接扩展至其他 CRDT 变体。

---

## 490. UniCoder: Unified Visual-to-Code Generation via Symbolic Rewards and Reference-Guided Code Optimization

**arXiv ID:** 2606.31732 | [PDF](https://arxiv.org/pdf/2606.31732v1)

**作者:** Yaozhi Zheng `[一作]` (Chinese University of Hong Kong), Xiangyu Yue `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 UniCoder，一种统一的强化学习框架，用于将科学图表、SVG 和网页等视觉输入转换为可执行代码。

**💡 创新点**

创新点包括：① Symbolic Attribute Alignment，通过轻量级辅助 LLM 提取代码中的可视属性并计算稠密元素级奖励；② Reference‑Guided Code Optimization，在低表现的采样组中动态注入真实参考代码，突破探索停滞。

**🔧 技术方法**

技术：多模态大语言模型（基于 Qwen3‑VL‑8B）+ 轻量级属性提取 LLM（Qwen‑2.5‑3B）+ Group Relative Policy Optimization（GRPO）+ 引入参考代码注入的 RG‑GRPO 算法。

**📊 数据集**

数据集：统一训练集（MSRL、SVG‑Stack、Web2Code）以及四个公开基准——ChartMimic、UniSVG、Design2Code 与 ScreenBench。

**📈 对比分析**

与多种开源模型（LLaVA‑v1.6、InternVL3.5‑8B、Qwen3‑VL‑8B 等）以及闭源模型（Gemini‑3‑Pro、GPT‑5、Claude‑4.5‑Opus）进行对比。实验显示，8B UniCoder 在所有四个基准上均超过开源同规模模型，且在 Design2Code 等任务上已逼近或超过部分闭源模型，整体性能显著提升。

**⚠️ 局限性**

局限性：仅针对静态视觉重建任务（图表、SVG、网页），不涵盖交互式界面、动画或需复杂运行时行为验证的程序；符号奖励设计依赖可提取的属性，迁移到新的视觉域可能需要重新定义属性提取器。

---

## 491. Do Machines Struggle Where Humans Do? LLM and Human Comprehension of Obfuscated Code

**arXiv ID:** 2606.31725 | [PDF](https://arxiv.org/pdf/2606.31725v1)

**作者:** Jack Le `[一作]` (University of Texas at Dallas), Tien N. Nguyen `[通讯]` (University of Texas at Dallas)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将人类程序员在代码混淆下的理解实验迁移到大型语言模型（LLM）上，系统评估不同混淆技术对模型输出预测、推理过程（Chain‑of‑Thought）以及错误模式的影响，并将结果与人类表现进行对齐分析。

**💡 创新点**

创新点在于：①首次用人类实验数据对照LLM进行混淆代码理解评估；②利用 Block Model 将失败拆分为原子、块、关系、宏四层次，精准定位模型与人类的相似与差异；③揭示推理调优模型在困难级别上与人类难度曲线高度一致；④发现链式思考长度可视为任务难度指标；⑤阐明对抗性重命名需同时伴随语义位移和标识符干扰才能诱发高置信错误。

**🔧 技术方法**

主要技术手段包括：使用 ObfuXtreme 与 javascript‑obfuscator 生成五个混淆层级；采用输出预测任务配合多种提示（BASELINE、S2、One‑Shot 等）收集 CoT；应用 Block Model 分层度量；统计分析采用混合效应逻辑回归、Spearman 相关、GLMM、置信度归一化等；对模型误差进行 HCI、Identifier Spike 等细粒度分析。

**📊 数据集**

使用两组数据集：① Nguyen 等公开的 50 人类实验样本（20 个函数级任务，Python/JavaScript），②自构建的 250 代码片段（HumanEval‑X、CruxEval‑X、LeetCode），两者均在五个混淆层级下进行评估。

**📈 对比分析**

比较方法：以输出准确率、CoT 生成长度、错误置信度等指标评估模型与人类的相似度；推理调优模型（如 DS‑R1 Qwen‑7B、SmolLM3‑3B）在大多数层级上准确率显著高于指令/编码器模型，且与人类难度相关系数 0.30–0.47；控制流扁平化导致准确率随状态空间复杂度负相关；CoT 长度与准确率呈负相关；对抗性重命名在同时存在语义位移与标识符干扰时产生高置信错误。

**⚠️ 局限性**

局限性：①实验片段为自包含、确定性小程序，难以代表真实工业代码；②混淆手段有限，未覆盖所有工业混淆手段；③输出预测仅是理解的粗略代理，未覆盖执行或调试等更深层次理解；④模型训练截止与数据泄露的控制依赖于发布时间，存在潜在泄露风险；⑤实验主要聚焦 Python 与 JavaScript，结果对其它语言或更大规模模型的可迁移性尚待验证。

---

## 492. Adapting Foundation ASR Models to Dysarthric Speech: A Case Study

**arXiv ID:** 2606.31722 | [PDF](https://arxiv.org/pdf/2606.31722v1)

**作者:** Christian Huber `[一作]` (Karlsruhe Institute of Technology), Alexander Waibel `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

为单一失语（言语运动障碍）说话人构建个性化语音识别系统，通过对基础 ASR 模型（Whisper）进行全参数微调，并将其部署在移动应用中实现实时识别与校正。

**💡 创新点**

①在单一失语说话人上实现高精度识别（WER 9.7%）的首次微调案例；②系统性比较全微调、LoRA 适配以及两种基础模型（Whisper 与 Qwen3‑ASR）的表现；③通过移动端持续收集用户校正数据，实现模型的在线自适应与改进。

**🔧 技术方法**

使用 Whisper‑large‑v3 进行全参数微调；尝试 LoRA 权重适配；对比 Qwen3‑ASR‑1.7B；采用 TEQST 工具收集数据；利用 CTranslate2 与 Faster Whisper 进行云端推理；开发 iOS/Android 交互式录音与校正界面。

**📊 数据集**

数据集由 TEQST 工具收集的 92 小时朗读语料（训练 89.8h，dev 1.1h，test 1.1h）以及 8.8h 通过移动端校正获得的额外数据，总计约 100.8h。

**📈 对比分析**

在不同适配数据量（1.4h、22.5h、100.8h）下对全微调与 LoRA 进行实验，并对 Whisper 与 Qwen3‑ASR 进行对比。结果显示：全微调 Whisper 的 WER 从 128.4% 降至 15.8%（仅 1.4h）→10.7%（22.5h）→9.7%（全量）；LoRA 效果显著逊色；Qwen3‑ASR 在同一任务中 WER 仍处于 14‑16% 之间。

**⚠️ 局限性**

系统需网络连接，无法离线运行；模型针对单一说话人，泛化到其他失语者需额外收集数据；大模型实时推理在设备端难以实现，需量化或缩小模型；扩展到多说话人或通用适配模型仍是开放问题。

---

## 493. Gaussian Belief Propagation for Tracking With Unresolved Measurements

**arXiv ID:** 2606.31716 | [PDF](https://arxiv.org/pdf/2606.31716v1)

**作者:** Augustin A. Saucan `[一作]` (TU Wien), Peter Willett `[通讯]` (University of Connecticut)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `aaccfe5c-6b26-4208-b23c-35331481e142` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种针对可能产生多重对象联合测量（即未解析测量）的多目标跟踪方法，构建了基于对象分区概率分布的关联模型，并在此基础上推导了全局高斯贝叶斯传播（GLBP）算法，实现了状态估计。

**💡 创新点**

创新点在于：①提出了一个可推广到任意大小对象集合的分区概率模型，该模型通过对任意对象对的耦合概率进行随机图建模，天然地包含传感器分辨率、检测率和噪声特性；②在此概率框架下设计了一套通用的LBP算法，并针对高斯模型进行了专门化，显著降低了复杂度从O(m^n)到O(m n 2^n)。

**🔧 技术方法**

主要技术包括：无向随机图与分区概率推导、概率数据关联模型、Loopy Belief Propagation 消息传递、以及高斯近似（包括混合高斯投影）来实现可计算的更新步骤。

**📊 数据集**

实验使用了合成的静态与动态场景，目标状态为二维位置/速度，传感器模型为线性高斯，测量中包含未解析、解析和杂波三种类型，噪声和杂波率在不同实验中被系统性调节。

**📈 对比分析**

与直接边缘化（GEM）、无未解析测量假设的GLBP（No-U GLBP）以及基于JPDAF的GEM-JPDAF 进行了对比。结果表明，GLBP在保持与GEM相近的平均跟踪误差（LMSE）和关联概率精度（平均TVD）同时，计算时间比GEM高约一至两百倍，尤其在高杂波率或目标数增大时提升更显著。

**⚠️ 局限性**

局限性包括：①依赖高斯线性假设，非线性或非高斯场景需进一步扩展；②虽然复杂度已大幅降低，但仍随目标数呈指数级增长；③对极高杂波率或极端未解析组的建模仍可能导致关联误差，需要更精细的分区概率估计或自适应耦合阈值。

---

## 494. Arena-T2I Hard: Benchmarking and Improving Faithfulness with Dependency-Aware Checklist

**arXiv ID:** 2606.31711 | [PDF](https://arxiv.org/pdf/2606.31711v1)

**作者:** Yuanhao Ban `[一作]` (Arena Intelligence Inc), Cho-Jui Hsieh `[通讯]` (Arena Intelligence Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向真实用户需求的 Arena‑T2I Hard 310 条长而复杂提示，提出基于 DAG 结构的依赖检查列表评估，并将其作为 RL 奖励与 Bradley‑Terry 审美奖励通过 GDPO 组合，以提升 T2I 模型的忠实度与审美平衡。

**💡 创新点**

首次将依赖图驱动的可检查列表与强化学习奖励相结合，并通过 Group‑Decoupled Normalization 解决多目标奖励尺度不平衡，实现高效的双目标优化；同时公开了高难度真实场景基准。

**🔧 技术方法**

使用 Gemini‑3‑Pro 生成提示分解、Qwen3.5‑27B 作为 VLM 判断，Flow‑GRPO/GDPO 强化学习框架，Group‑Decoupled Normalization 归一化奖励，并采用 M‑MMRB2 双人对比评估。

**📊 数据集**

采用 310 条从 2026 年 1–3 月 Arena T2I 公开投票日志抽取的真实长提示；10k/1k 的训练/测试子集；以及 DPG‑Bench、DSG 等对照基准数据。

**📈 对比分析**

在 Arena‑T2I Hard 上最高模型 0.855，平均与其他 10 系统 0.523 相距 33pp；与单奖励或线性组合相比，Faith+Pick+GDPO 在 SD3.5‑M 与 FLUX.1‑dev 上的对比评测均显著提升，Win‑rate 超过 64% 并在所有单奖励、四奖励集合及 DreamSync 之上。

**⚠️ 局限性**

评估仍依赖 VLM 判断机的准确性；基准覆盖范围有限，难以全面体现所有创意场景；奖励结构对模型架构与算力要求高，开源实现受大型 VLM 与 RL 资源成本限制。

---

## 495. WIDER-FAIR: An Annotated Version of the WIDER-FACE Dataset for Fairness Evaluation

**arXiv ID:** 2606.31704 | [PDF](https://arxiv.org/pdf/2606.31704v1)

**作者:** Maxime Moussi `[一作]` (UCLouvain), Félicien Schiltz `[通讯]` (Euranova)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建并发布了WIDER-FAIR数据集，对WIDER-FACE中的人脸进行手工感知种族（Asian、Black、Indian、White）和性别标注，并用该数据集训练YOLOv5进行人脸检测，评估不同人群的检测公平性。

**💡 创新点**

首次提供了既包含标准人脸检测标注又附带敏感属性（种族和性别）的数据集，使得可以直接在公平性评估任务中使用；通过留一种族/性别实验揭示了黑人脸检测的明显劣势及其对模型泛化的重要性。

**🔧 技术方法**

使用YOLOv5作为检测器，KNN分类器验证标签一致性，t-SNE可视化验证嵌入空间聚类，利用Recall和Equal Opportunity（基于Recall的标准差）度量公平性。

**📊 数据集**

基于WIDER-FACE的图像集（共16,256张，包含约40万张人脸），在此基础上添加了种族和性别标签。

**📈 对比分析**

在完整数据集上训练YOLOv5，记录不同IoU阈值下的Recall；随后进行Leave-One-Ethnicity-Out（LOEO）和Leave-One-Sex-Out（LOSO）消融实验，比较各组的Recall与Disparity。结果显示黑人脸Recall显著低于其他种族，去除黑人样本会进一步加大Disparity；性别消融实验显示女性样本缺失会削弱女性检测性能，但可降低Disparity。

**⚠️ 局限性**

仅由一名标注者完成感知属性标注，可能引入主观误差；对图像难度的阈值设定具有任意性；数据集仍以白人脸为主，整体分布不平衡，可能影响实验结论。

---

## 496. Phantom: A Unified Face-Swap Deepfake Protection Framework with Latent and Spatial Constraints

**arXiv ID:** 2606.31703 | [PDF](https://arxiv.org/pdf/2606.31703v1)

**作者:** Jungkon Kim `[一作]` (Samsung Electronics), Juseong Lee `[通讯]` (Samsung Electronics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种统一的面部交换Deepfake防护框架，结合潜在空间与空间域的约束，实现对身份信息的干扰；

**💡 创新点**

创新点包括：①自适应目标合成策略，生成身份偏移但属性保持的目标图像，用于在潜在空间中定义明确的身份方向；②基于语义掩码的空间约束攻击，仅在面部重要区域更新扰动，提升干扰的聚焦性与视觉自然度；

**🔧 技术方法**

使用技术包括：面部分割与语义掩码、面部生成模型G_F进行目标合成、潜在空间中的对抗优化、掩码约束的梯度更新、对抗损失与质量损失的联合优化；

**📊 数据集**

实验数据集：CelebA‑HQ（1,000张）和LADN（334张），并在多种面部识别模型和面部交换工具（UniFace、INSwapper、SimSwap）上进行评估；

**📈 对比分析**

与多种基线方法（MI‑FGSM、TI‑DIM、TIP‑IM、AMT‑GAN、CLIP2Protect、DiffAM、RMT‑GAN）比较，实验证明在面部交换攻击中的 PSR 提升分别为 27.8%、25.6% 和 16.6%，同时保持较高的视觉质量（SSIM≈0.973、PSNR≈33.6、FID≈18.7）；

**⚠️ 局限性**

局限性：对掩码精度依赖较高，遮挡或极端姿态下保护效果下降；方法需要较多计算资源（GPU 训练），对未知深度伪造工具的泛化仍需进一步验证；

---

## 497. FastDSAC: Enhancing Policy Plasticity via Constrained Exploration for Scalable Humanoid Locomotion

**arXiv ID:** 2606.31691 | [PDF](https://arxiv.org/pdf/2606.31691v1)

**作者:** Guanchen Lu `[一作]` (Chongqing University), Guofa Li `[通讯]` (Chongqing University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出 FastDSAC，通过在目标 Q 值计算中对动作进行均值中心截断，减少极端动作导致的 TD 误差方差，提升高并行离线强化学习的稳定性与收敛速度。

**💡 创新点**

创新点在于仅在目标动作上使用连续高斯分布的均值中心截断映射，并在熵正则化中使用相应的对数概率，既保持探索性又有效抑制极端动作，且实现简单且无额外计算开销。

**🔧 技术方法**

采用分布式 Actor-Critic（DSAC）框架、连续高斯策略、均值中心截断映射、可自适应方差调节、经验回放与多步目标 Q 估计。

**📊 数据集**

在 MuJoCo Playground（4 个仿真任务）和 HumanoidBench（6 个仿真任务）上进行评估。

**📈 对比分析**

与 FastSAC、FastTD3、DSAC‑T、PPO 等基线比较，FastDSAC 在环境步数和壁钟时间上均取得最快收敛，最终回报最高，且在高 UTD 和高并行数下表现尤为稳定。

**⚠️ 局限性**

局限性包括对网络可塑性的评估仍为间接指标、缺乏对实际硬件或真实世界的验证，以及在极端动态扰动或非平稳环境下的鲁棒性尚未系统探究。

---

## 498. Cross-lingual Relation Extraction with Large Language Models: Zero-Shot, Few-Shot, and Fine-Tuned Evaluation on Romanian

**arXiv ID:** 2606.31718 | [PDF](https://arxiv.org/pdf/2606.31718v1)

**作者:** Dragos-Mitrut Vasile `[一作]`, Ciprian-Octavian Truica `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过将SemEval‑2010 Task 8英文数据集自动翻译成罗马尼亚语，并在Gemma 4 31B LLM上进行零样本、少样本推理和QLoRA微调，研究了低资源语言罗马尼亚语的跨语言关系抽取。

**💡 创新点**

创新点在于（1）首次系统使用LLM（Claude Haiku）进行大规模自动翻译并自动验证；（2）在同一模型上比较零样本、少样本和参数高效微调的表现；（3）将LLM结果与多种尺寸的编码器基线进行对比，揭示微调后跨语言差距显著缩小。

**🔧 技术方法**

采用的技术包括：Gemma 4 31B（指令微调LLM）、Claude Haiku（翻译）、QLoRA（4‑bit量化+LoRA微调）、多语言编码器XLM‑RoBERTa、单语BERT‑Romanian、RoBERT‑large等；实验在单个A100 GPU上完成。

**📊 数据集**

使用的数据集为SemEval‑2010 Task 8的英文原始数据（8000训练+2717测试句子），通过Claude Haiku自动翻译得到罗马尼亚语版本（7871/2664例子）。

**📈 对比分析**

比较方法：在关系分类任务中报告宏F1与准确率；在端到端抽取任务中报告精确匹配、关系匹配、实体匹配。结果显示：零样本下罗马尼亚语比英文低3‑5pp；少样本提升不足1pp；QLoRA微调后宏F1提升22‑24pp，精确匹配提升≈39pp；与4‑倍小的编码器相比，微调后LLM仅优势0.5‑0.8pp，计算成本显著更高。

**⚠️ 局限性**

局限性包括：翻译后实体级错误高达约26%（12%严重），影响端到端抽取；仅评估Gemma 4和四个编码器，未考察其他开源LLM；实验仅在单一种子下跑；仅针对SemEval‑2010的十类关系，结果可能不具备普适性。

---

## 499. Addressing Over-Refusal in LLMs with Competing Rewards

**arXiv ID:** 2606.31748 | [PDF](https://arxiv.org/pdf/2606.31748v1)

**作者:** Taeyoun Kim `[一作]` (Carnegie Mellon University), Aviral Kumar `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在单个生成回合中对推理与答案分别赋予对立奖励，训练模型在探索危险推理后仍能给出安全回答，从而缓解过度拒绝问题。

**💡 创新点**

创新点在于将安全推理视为有益探索，将有害推理与安全回答作为竞争目标，并通过过程奖励实现精细的信用分配。

**🔧 技术方法**

使用的技术包括基于KL约束的强化学习（GRPO）、过程奖励分配、模型平均化以及对抗式奖励设计。

**📊 数据集**

主要使用了 HarmBench、WildJailbreak、XSTest、Fortress、False-Reject 等安全评测数据集。

**📈 对比分析**

在安全-拒绝平衡曲线上，-1.5B 在安全性与合规性指标上达到0.93的二次均值，优于纯结果奖励、密集结果奖励以及其他开源模型。

**⚠️ 局限性**

限制在于奖励设计的选择性，其他探索策略可能更有效；此外，模型仍需在更强攻击或更大规模模型上进一步验证。

---

## 500. Intrinsically Stable Spiking Neural Networks: Overcoming the Performance Barrier in the Absence of Batch Normalization

**arXiv ID:** 2606.31695 | [PDF](https://arxiv.org/pdf/2606.31695v1)

**作者:** Ruichen Ma `[一作]` (University of Electronic Science and Technology of China), Shaogang Hu `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种无运行时激活归一化的深度脉冲神经网络（IS‑SNN），通过拓扑感知权重标准化和修改残差连接实现激活归一化的去除，同时保持网络信号的稳态。

**💡 创新点**

创新点在于将权重标准化与网络拓扑相结合，离线完成归一化重参数化，解决了BN‑free SNN 的发射率衰减问题，并彻底消除推理阶段的乘法开销，兼顾性能与硬件友好性。

**🔧 技术方法**

采用拓扑感知权重标准化、修改残差连接、离线重参数化、LIF/PLIF 神经元、Surrogate 梯度训练、SpikingJelly 框架以及 FPGA 硬件实现。

**📊 数据集**

在 CIFAR‑10、CIFAR‑100、ImageNet、DVS‑Gesture、CIFAR10‑DVS 以及基于 Transformer 的 Spikformer 等数据集上进行评估。

**📈 对比分析**

与动态 BN 变体（如 tdBN、BNTT、TEBN、Spike‑Norm）以及传统 BN 进行对比。IS‑SNN 在 ImageNet 上获得 68.05% (E400) 的 Top‑1，几乎匹敌 TEBN 68.28%；在 ResNet‑152 上达到 76.98% 而 BN‑free 仅 17.10%；在 DVS‑Gesture 上达到 96.88% 远超 BN‑free。FPGA 上 LUT 资源降低 96.4%，训练吞吐量提升 15%。

**⚠️ 局限性**

局限性包括：1）仅适用于训练完成后权重固定的推理部署，对在线学习需额外硬件支持；2）离线重参数化假设权重不变，可能不适用于持续学习场景；3）实验未给出完整的系统能耗评估，仅关注算术和 LUT 开销。

---

## 501. ScratchWorld: Evaluating If World Models Compute Executable Consequences

**arXiv ID:** 2606.31689 | [PDF](https://arxiv.org/pdf/2606.31689v1)

**作者:** Yufeng Lin `[一作]` (Pui Ching Middle School), Jialu Zhang `[通讯]` (University of Waterloo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个可执行的 Scratch 世界评测基准，利用固定的 Scratch VM 对程序执行进行回放，生成完整的状态转移、因果链和逆因果结果，并用此基准评估语言模型在动作条件下的世界模型性能。

**💡 创新点**

创新点在于：①把程序执行视作可重复的实验平台，提供“可回放的”动作‑状态‑结果三元组；②设计了“值感知改变字段 F1”指标，剔除传统全状态重叠指标中对持久状态的误报；③引入同一实例复制诊断和多种诊断子任务（因果归因、逆因果预测、长期滚动），全面拆解模型的错误模式。

**🔧 技术方法**

技术包括：基于 Scratch VM 的执行记录与回放、结构化状态与事件日志、Causal trace 生成、值感知 F1 计算、相同实例复制诊断、语义扰动与多模态包装；评估时使用提示式语言模型（如 GPT‑5.5、Qwen3.7‑Max 等）作为预测器。

**📊 数据集**

数据集包含 659 条回放验证的实例，分为 291 条真实 Scratch 项目实例和 368 条合成机制实例，涵盖 153 个项目哈希，涵盖运动、广播、克隆、列表操作等九类机制。

**📈 对比分析**

比较方法：在 265 条共享的下一状态子集上，用“值感知改变字段 F1”对七个提示模型进行评测，最高得分仅 13.8%；在复制诊断、因果归因、逆因果预测和长期滚动任务中，模型表现普遍偏低，表明它们往往能识别被影响字段但难以精确计算新值或完整因果链。

**⚠️ 局限性**

局限性：仅覆盖程序定义的 Scratch 世界，无法验证对真实物理、机器人或开阔视频的泛化；基准规模有限（153 个项目、659 条实例）；因果标签在并发、克隆等复杂事件下易失真，需进一步验证；对现有语言模型的评测未包括训练过的专用世界模型。

---

## 502. NURBS Splatting: A Unified Differentiable Rendering Framework for Vector Graphics

**arXiv ID:** 2606.31764 | [PDF](https://arxiv.org/pdf/2606.31764v1)

**作者:** Jingye Qiu `[一作]` (Hunan University), Shizhe Zhou `[通讯]` (Hunan University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一个可微分渲染框架，能够优化二维NURBS曲线与闭合区域的控制点、权重、结点以及宽度，实现图像重建与矢量化。

**💡 创新点**

首次将非均匀有理B样条与高斯摊平相结合，支持权重与结点的联合学习，显著提升了几何精度与渲染速度。

**🔧 技术方法**

使用Cox–de Boor递推求解NURBS、等距高斯摊平、SDF加权填充、网格步长退火以及几何正则化等技术实现端到端可微分渲染。

**📊 数据集**

在192字符书法基准（92日语假名+100中文常用字）和100张Noto Emoji图像上进行评估。

**📈 对比分析**

与Berio的均匀B样条、DiffVG以及Bézier摊平方法相比，本文在MSE、PSNR、SSIM和优化时间等指标上分别提升约30%、1.6 dB、0.02和1.4×。

**⚠️ 局限性**

填充策略仍采用启发式SDF网格，且随着曲线复杂度增加高斯数目增多会占用更多显存，且目前仅适用于二维渲染。

---

## 503. A Technical Typology of AI Systems in Public Administration

**arXiv ID:** 2606.31755 | [PDF](https://arxiv.org/pdf/2606.31755v1)

**作者:** Jonathan Rystrøm `[一作]` (University of Oxford), Chris Russell `[通讯]` (University of Oxford)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了面向公共行政的五层 AI 系统技术分类法，并通过对 2019–2025 年 91 篇高被引文献的系统编码，评估当前研究在技术精确度上的不足，揭示了欠缺、误判和过度概括的普遍性。

**💡 创新点**

①首次将“可供性（affordance）”理论用于 AI 系统的公共价值分类；②提出了可操作的诊断问答工具，帮助研究者在不需深度技术背景的前提下对 AI 系统进行精确定位；③系统性地量化了公共行政研究中的技术模糊性，为未来文献评价和方法改进提供基准。

**🔧 技术方法**

采用了：① Affordance 理论构建分类框架；② 文献系统检索与样本筛选（OpenAlex、学术数据库）；③ 结构化定性编码与 LLM（Gemini 3.1 Flash‑Lite）辅助抽取研究“段落”；④ 人工二次审校与一致性检验；⑤ 统计描述与置信区间绘制。

**📊 数据集**

数据集为 91 篇在公共行政与数字政府期刊（2019–2025）中被引用次数最高的论文，经过作者手工筛选后包含了实验、案例、概念综述等多类型研究。

**📈 对比分析**

对照指标包括三类技术精度缺陷：欠缺（55%）、误判（31%）和过度概括（41%）。研究通过分层描述（按 AI 类别与公共价值维度）展示了缺陷分布，没有发现显著时间趋势，说明问题在近年未得到根本缓解。

**⚠️ 局限性**

局限性包括：① 样本受高被引偏好与时间滞后影响，可能低估近期新兴 AI 类型（如 agentic）的研究情况；② 采用引用加权抽样与手工编码，难以完全代表整个领域；③ 仅关注技术描述而未深入探讨非技术治理因素；④ 需要随技术演进不断更新分类法和诊断工具。

---

## 504. FedXDS: Leveraging Model Attribution Methods to counteract Data Heterogeneity in Federated Learning

**arXiv ID:** 2606.31742 | [PDF](https://arxiv.org/pdf/2606.31742v1)

**作者:** Maximilian Andreas Hoefler `[一作]` (Fraunhofer Heinrich Hertz Institute), Wojciech Samek `[通讯]` (Fraunhofer Heinrich Hertz Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 FedXDS，一种通过 XAI 归因图筛选并差分隐私地共享任务相关特征的联邦学习框架，缓解非 IID 数据导致的模型漂移并提升性能；

**💡 创新点**

创新点在于首次将可解释 AI 的归因方法用于联邦学习中实现特征级数据共享，利用归因稀疏降低敏感度并配合度量 DP 提供更强隐私保障，同时显著提升通信效率；

**🔧 技术方法**

使用了传播式归因（如 LRP、Integrated Gradients、SmoothGrad、Gradient×Input）提取特征、Gaussian 机制实现度量差分隐私、FedAvg+知识加权训练、warmup 预训练等技术；

**📊 数据集**

实验数据集包括 CIFAR-10、CIFAR-100、Tiny-ImageNet、CelebA 与 FEMNIST 等具有不同异构度的图像数据集；

**📈 对比分析**

与 FedAvg、FedProx、FedDyn、SCAFFOLD、FedSAM、FedDISCO、FedFed、FedFTG、FedGen、FedAux、FedDF 等基线及其 XDS 变体对比，FedXDS 在准确率、收敛速度与通信轮次上均表现最优（如 CIFAR-10 上 81.72% 最高准确率、通信轮次仅 14/23；在 CelebA、FEMNIST 上均领先 1–2%）；

**⚠️ 局限性**

局限性包括：依赖归因方法的质量（LRP 最佳但不一定适用于所有模型）；高稀疏度或严格隐私预算可能导致性能下降；仅在图像任务上验证，缺乏对文本/时间序列等域的扩展；需要额外 warmup 训练与超参数调优；若归因图本身泄露敏感信息，仍可能存在隐私风险。

---

## 505. STEB: Style Text Embedding Benchmark

**arXiv ID:** 2606.31741 | [PDF](https://arxiv.org/pdf/2606.31741v1)

**作者:** Rafael Rivera Soto `[一作]` (Johns Hopkins University), Cristina Aggazzotti `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 Style Text Embedding Benchmark（STEB），统一评估文本风格嵌入的开源基准，涵盖多语言、多任务与多数据集；并对现有风格、语义、MLM、CLM 等模型进行系统评测。

**💡 创新点**

提出了两种评分方式（操作性得分与定义性得分），将风格任务与语义任务分离；设计了多任务评估框架并统一评测协议；展示了风格嵌入在多任务上的优势与多语言挑战。

**🔧 技术方法**

使用对比学习、内容去相关化训练、Transformer（BERT/DeBERTa/ModernBERT）、CLM（OPT、Qwen）、非神经手工特征（词频、TF-IDF n-gram、Stylometric、NeuroBiber）等多种技术；以及聚类、余弦相似、检索、顺序对齐、线性探测等评估方法。

**📊 数据集**

包括超过50个风格相关数据集（如 PAN、LUAR、STAR、CISR、MSR、LISA 等）以及多语言数据（荷兰语、法语、希腊语、意大利语、波兰语、西班牙语）和多任务子集（作者身份验证、检索、AI 文本检测、探测等）。

**📈 对比分析**

通过宏平均的多任务得分（操作性与定义性）对比，发现没有单一模型统治；风格特定模型在作者相关任务表现最好，MLM 在语义与探测任务表现突出；内容去相关化模型在避免语义捷径的任务中优先；多语言风格模型整体落后于英文学者模型。

**⚠️ 局限性**

存在数据集泄露/混合、评测覆盖范围受限、对非英语任务与模型支持不足，导致多语言风格嵌入效果不佳；未来需要更多元数据集与更严格的训练/测试分离。

---

## 506. MemLearner: Learning to Query Context memory for Video World Models

**arXiv ID:** 2606.31734 | [PDF](https://arxiv.org/pdf/2606.31734v1)

**作者:** Jiwen Yu `[一作]` (University of Hong Kong), Xihui Liu `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 MemLearner，一种基于学习的自适应上下文查询机制，用以解决视频世界模型在长时序生成中的记忆与一致性问题

**💡 创新点**

创新点在于引入查询令牌（query tokens）作为上下文与预测令牌之间的桥梁，利用预训练的视频生成模型自身完成上下文检索，并在训练时只使用噪声预测损失即可实现端到端学习；同时提出了高效的查询层与注意力裁剪策略

**🔧 技术方法**

采用潜在视频扩散模型（3D VAE + Diffusion Transformer）作为基础框架，结合摄像头姿态编码、查询令牌设计、查询层/生成层分离以及注意力模式裁剪等技术

**📊 数据集**

使用自制的基于Unreal Engine的渲染视频数据集（含复杂遮挡与动态物体），以及真实世界视频数据（如SpatialVID、Sekai-real），通过多数据集训练策略分别为不同数据类型配置摄像头编码器

**📈 对比分析**

通过与多种基准（DFoT、FramePack、VMem、Context-as-Memory、Separate Module 等）在 FID/FVD、PSNR/LPIPS、重访一致性等指标上进行对比，实验表明 MemLearner 在场景一致性、记忆保持以及遮挡/动态场景下的生成质量均优于现有方法

**⚠️ 局限性**

主要局限在于模型规模和训练数据量有限，难以同时跟踪大量动态交互实体；记忆机制仍随生成时间线性增长，缺乏压缩、更新或忘却等高级特性，未来需要更大模型与更丰富的长视频数据来进一步提升性能

---

## 507. UniTacVLA: Unified Tactile Understanding and Prediction in Vision Language Action Models

**arXiv ID:** 2606.31723 | [PDF](https://arxiv.org/pdf/2606.31723v1)

**作者:** Xidong Zhang `[一作]` (Harbin Institute of Technology), Weihao Yuan `[通讯]` (Nanjing University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一的触觉学习框架 UniTacVLA，旨在通过将触觉信号视为动态交互提示，提升视觉‑语言‑动作模型在接触丰富、精细操作中的表现。

**💡 创新点**

创新点包括：① 统一触觉潜在空间并通过触觉链式推理（T-CoT）实现对接触状态的语义理解；② 采用粗细两级触觉预测策略，提前推断未来接触演化；③ 设计混合控制器，将实时与预测的触觉信息融合，实现高频动作修正，提升对接触变化的响应性。

**🔧 技术方法**

技术手段包括：变分掩码自动编码器（VMAE）生成触觉嵌入；Transformer 编码/解码；多层感知机+Diffusion Transformer（DiT）实现粗细预测；大型语言模型（LLM）完成 T-CoT；流匹配损失、交叉熵、KL 散度等联合训练；高频控制器基于 Transformer 的残差修正。

**📊 数据集**

使用了真实机械臂 RealMan 7-DoF 进行的 8 个接触丰富任务（姿态调整、擦拭、插入、装配）共 4 类、20 个子任务的自我演示数据，包含清洁与人为扰动两种环境。

**📈 对比分析**

与基线 π_0、π_0.5、π_0.5‑VTLA、π_0.5‑TacVLA 等方法比较，UniTacVLA 在清洁与扰动两种设定下均取得最高平均成功率，特别是在插入与装配任务中提升显著，证明触觉理解与预测的融合显著增强了机器人在接触复杂场景中的鲁棒性与精度。

**⚠️ 局限性**

局限性包括：① 训练数据受操作员噪声影响，细粒度触觉动态学习受限；② 在极端视觉遮挡或不完整语言指令下的鲁棒性尚未充分验证；③ 当前框架未显式建模力/扭矩信号，可能进一步提升性能。

---

## 508. AdaTrans: Automated C to Rust Transformation via Error-Adaptive Repair

**arXiv ID:** 2606.31706 | [PDF](https://arxiv.org/pdf/2606.31706v1)

**作者:** Xiaofan Liu `[一作]` (Wuhan University), Jifeng Xuan `[通讯]` (Wuhan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AdaTrans 框架，实现 C 代码到 Rust 的自动迁移，重点解决所有权系统、编译器反馈利用和功能等价问题；

**💡 创新点**

三大创新：1）基于编译器错误的策略驱动检索增强生成（RAG）机制，将错误映射到专用修复模板与 Rust 文档；2）错误分层转换策略（ESTS），按错误类别动态调节采样温度并实现局部循环抖动回退；3）闭环验证管线，利用编译器与运行时测试迭代修复并确保功能等价；

**🔧 技术方法**

采用大语言模型（ChatGPT‑4），检索式提示工程、错误分类映射、温度调度、局部循环回退、差分测试等技术；

**📊 数据集**

在 104 道 LeetCode 周赛（413–438）自包含 C 题目上进行实验，利用每道题的 10–200 条随机输入输出对做功能验证；

**📈 对比分析**

与三款现有 C‑to‑Rust 工具（c2rust、EvoC2Rust、Tymcrat、PtrTrans）及零射 LLM 基线对比，AdaTrans 在编译通过率 95.51%（±1.11%）和功能通过率 81.09%（±3.09%）上显著优于对手，且 1.19% 的 unsafe 文件率几乎为零；

**⚠️ 局限性**

局限性：仅针对单文件、标准 I/O 的自包含模块，未覆盖多文件项目和系统级复杂依赖；模型特定（基于 ChatGPT‑4），迁移到其他 LLM 的效果未知；训练数据可能仍受模型训练集影响，且对极端大规模程序的可扩展性待验证。

---

## 509. Diffusing Blame: Task-Dependent Credit Assignment in Biologically Plausible Dual-Stream Networks

**arXiv ID:** 2606.31700 | [PDF](https://arxiv.org/pdf/2606.31700v1)

**作者:** Yutaro Yamada `[一作]` (Sakana AI), Robert Tjarko Lange `[通讯]` (Sakana AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了一种遵循 Dale 原则的双流 Error Diffusion 网络，在分类（MNIST、CIFAR-10）和强化学习（Brax 连续控制、Craftax）任务中实现了可竞争的性能。

**💡 创新点**

创新点包括：引入 modulo error routing 扩展多分类；层特定 Sigmoid 宽度、批量中心化类错误和不对称初始化等任务依赖的辅助机制；以及揭示不同任务下信用分配瓶颈的逆转。

**🔧 技术方法**

使用的技术包括：双流 excitatory/inhibitory 架构、Error Diffusion 学习规则、ED-PPO 与 PPO 的无权重传递整合、批量中心化错误、层特定激活宽度以及不对称权重初始化。

**📊 数据集**

使用的数据集与任务包括：MNIST、CIFAR-10（图像分类）；Google Brax 的 Ant、HalfCheetah、Humanoid（连续控制）；以及 Craftax（开放式探索）。

**📈 对比分析**

对比方法为 DFA、BP‑PPO、ES 等；在 MNIST 上获得 96.7%，CIFAR‑10 上 61.7%；在 Ant、HalfCheetah、Humanoid 等连续控制任务中，ED‑PPO 与 BP‑PPO 的表现相近或略优，但在 Craftax 与 DFA 仍存在性能差距。

**⚠️ 局限性**

局限性包括：参数量约为无约束网络的 4 倍导致计算开销增加；隐含稀疏化导致有效容量受限；在需要细粒度信用分配的复杂 RL 任务中表现不足；以及在最难任务上仍与传统后向传播存在显著性能差距。

---

## 510. RCT: A Robot-Collected Touch-Vision-Language Dataset for Tactile Generalization

**arXiv ID:** 2606.31694 | [PDF](https://arxiv.org/pdf/2606.31694v1)

**作者:** Jingbo He `[一作]` (TU Dresden), Roberto Calandra `[通讯]` (TU Dresden)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并公开了RCT数据集，记录机器人在122种工业材料上完整的触摸按压序列，并提供对应的视觉图像与文本描述，可在材料、类别、传感器、接触位置及按压序列等维度进行保留测试。

**💡 创新点**

创新点在于保留完整的接触序列并为每个序列提供唯一标识，明确区分训练与测试之间的接触重叠；提出基于接触序列的保留评估协议，揭示帧随机拆分导致的性能夸大，并证明均匀抽样能显著提升对比学习效果。

**🔧 技术方法**

采用InfoNCE式多模态对比学习（触觉-视觉-文本），结合ViT-Base触觉编码器与OpenCLIP视觉/文本编码器，并使用Recall@1/5评估与线性探测器分析嵌入空间结构。

**📊 数据集**

使用自身构建的RCT数据集（122种材料、3个DIGIT传感器、29,279帧，7类）以及公开的TVL/SSVTP+HCT数据集进行对比验证。

**📈 对比分析**

通过三种拆分方式（frame-random、接触序列保留、材料保留）进行对比，结果显示去除接触序列重叠使Recall@1下降17.7pp，进一步保留材料导致下降42pp；在材料保留测试下，Recall@1仅约25%，但线性分类器在未见材料上可达到约50%的类别识别率，表明对新材料的泛化仍有限。

**⚠️ 局限性**

局限性包括仅收集平坦表面、三种DIGIT传感器、固定按压协议的触摸数据，缺乏动态探索、曲面或多对象交互；使用材料级视觉/文本标注导致无法实现帧级精确对应；硬度二分类仍表现不佳，说明单帧或简单线性探测不足以捕获更细微的物理特性。

---

## 511. Overview of the TalentCLEF 2026: Skill and Job Title Intelligence for Human Capital Management

**arXiv ID:** 2606.31692 | [PDF](https://arxiv.org/pdf/2606.31692v1)

**作者:** Luis Gasco `[一作]` (Avature Machine Learning), Rabih Zbib `[通讯]` (Avature Machine Learning)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了TalentCLEF 2026挑战赛，详细描述了两项任务（职位-人匹配与职位-技能匹配）的设计、数据集、评估方式、参赛情况以及主要实验结果。

**💡 创新点**

提出了多语言、跨语言以及基于性别偏差的公开基准；系统性分析并总结了多阶段检索+重排序、LLM生成结构化视图、图知识、实体抽取等多种技术在人才与技能匹配任务中的优势与趋势。

**🔧 技术方法**

采用多阶段检索（BM25、密集编码器、跨编码器）、LLM（Prompt、LLM‑as‑a‑judge、生成结构化视图）、图神经网络（知识图/关系卷积网络）、实体抽取、查询扩展、数据增强、Reranking（点式、对式、列表式）等技术，并进行多模型融合。

**📊 数据集**

Task A 使用中英文合成的职位描述与简历数据集（Dev 10k/10k，Test 40k/40k）；Task B 使用 ESCO 职业技能集的训练/Dev/Test 集（304/125 查询，9052/5358 文档），所有数据均公开发布于 Zenodo 和 Codabench。

**📈 对比分析**

通过 Codabench 公开评测，使用 MAP、NDCG（binary/graded）和 RBO 等指标；Top 系统在任务 A 的 MAP 约 0.71（单语）和 0.70（跨语），任务 B 的 NDCG(graded) 约 0.81，显示多阶段检索+LLM 重排序策略最优；但整体仍存在模型单一性和基准多样性不足的挑战。

**⚠️ 局限性**

主要局限包括：数据为合成生成，真实性与多样性受限；评估关注性别偏差而未覆盖其他公平性维度；LLM 依赖外部大模型，复现难度大；多团队采用混合架构但缺乏统一基准和可解释性，导致结果可比性受限。

---

## 512. Distributed Hierarchical Temporal Memory with Shared Associative Memory for Cross-Entity Preemptive Warning

**arXiv ID:** 2606.31789 | [PDF](https://arxiv.org/pdf/2606.31789v1)

**作者:** Pavia Bera `[一作]` (University of South Florida), Sanjukta Bhanja `[通讯]` (University of South Florida)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种分布式Hierarchical Temporal Memory（D-HTM）框架，通过共享的Sparse Distributed Representation（SDR）空间和Shared Associative Memory（SAM）实现多实体间的跨实体预警；

**💡 创新点**

创新点在于：1) 共享的Spatial Pooler让不同实体产生可比的SDR；2) SAM在推理时实时检索预先学到的预警模式，实现先发预警；3) 维持HTM的在线学习，无需集中式模型；

**🔧 技术方法**

使用HTM的Encoder、Spatial Pooler、Temporal Memory以及自定义的SAM（基于稀疏重叠匹配的检索与更新逻辑）；

**📊 数据集**

评估数据集包括Server Machine Dataset（SMD）、Soil Moisture Active Passive（SMAP）、Mars Science Laboratory（MSL）以及自制的级联模拟数据；

**📈 对比分析**

与OmniAnomaly、单实体HTM及Encoder-only D-HTM进行比较。实验显示D-HTM在三大真实数据集上实现了与OmniAnomaly相当的反应式异常检测，同时在跨实体预警上取得平均8.1个样本的提前预警；

**⚠️ 局限性**

局限包括固定的预检索窗口、基于重叠的检索缺乏概率置信度、Spatial Pooler被冻结无法适应部署期间的分布漂移，未来工作需改进自适应窗口、在线表示学习与置信度估计。

---

## 513. Self-Supervised Temporal Regularization for Landmark-Based Cardiac Segmentation with Automatic AHA Regional Mapping

**arXiv ID:** 2606.31785 | [PDF](https://arxiv.org/pdf/2606.31785v1)

**作者:** David Montalvo-García `[一作]` (Universidad Politécnica de Madrid), Enzo Ferrante `[通讯]` (Universidad de Buenos Aires)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e15e3743-5ee0-4d5f-813d-d146868082fc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

在心脏超声序列上实现自监督的时间一致性正则化，改进基于图的标记分割模型，并自动将标记映射至 AHA 17 分段标准。

**💡 创新点**

首次在无帧标注条件下使用速度与加速度连续性作为自监督信号，与隐式对应学习相结合，实现时序一致性并自动完成 AHA 区域划分。

**🔧 技术方法**

采用 Mask-HybridGNet Dual 的图卷积 VAE 结构，结合 Chamfer、边界、弹性、曲率等空间损失；自监督时间正则化（速度、加速度项）与 Savitzky–Golay 滤波评估抖动；基于人群平均图谱实现自动 AHA 映射。

**📊 数据集**

在 CAMUS 超声序列数据集（400/50/50 划分）上进行实验，包含 2CH 与 4CH 视图约 20 帧/序列。

**📈 对比分析**

与无时间正则化的 Mask-HybridGNet 以及 nnUNet+后处理做对比；在 Dice、HD95、EF/ESV/EDV MAE 等空间与临床指标上保持相近，但在抖动、FTD、Jerk 等时间一致性指标上显著提升，尤其是无 rasterization 的版本。

**⚠️ 局限性**

需要大量 GPU 内存处理完整序列；自监督正则化可能不足以捕捉极端异常运动；仅在 CAMUS 上验证，缺乏对多模态或不同视图的泛化评估；对低质量图像的鲁棒性未知。

---

## 514. Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers

**arXiv ID:** 2606.31779 | [PDF](https://arxiv.org/pdf/2606.31779v1)

**作者:** Ying Fan `[一作]` (University of Wisconsin-Madison), Kangwook Lee `[通讯]` (University of Wisconsin-Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种循环填充 Transformer（LoLo），在隐藏状态中并行生成推理步并用交叉熵直接监督金标步骤，构成新的 latent reasoning 方法。

**💡 创新点**

创新点在于将循环深度与填充 latent 前缀相结合，利用并行监督直接对每个 latent 对齐到对应的 gold 步骤，从而解决大规模模型 latent 方法性能缺口的问题，同时保持高并行度和可解释性。

**🔧 技术方法**

使用技术包括循环 Transformer（循环深度 R）、填充 latent 前缀（K 个块、每块 c 个 token）、并行交叉熵监督（通过主 LM 头或辅助解码器）、以及可选的辅助解码器（aux）来实现并行监督。

**📊 数据集**

训练和评估数据集为：GSM8K（train+test）、GSM‑Hard、MultiArith、SVAMP 以及自然语言版 GSM8K（每步为完整句子推理）。

**📈 对比分析**

与 Explicit CoT、CODI、SIM‑CoT 等基线对比，LoLo 在 Llama‑3.2‑3B‑Instruct 上在 GSM8K 取得约 70% 的准确率，接近甚至略低于 Explicit 的 71.5%，在自然语言任务也匹配 Explicit，同时在思考阶段推理时间比 Explicit 快 2.5×（数学表达式）或 6.9×（自然语言），显著提升推理效率。

**⚠️ 局限性**

限制：目前仅在数学推理任务上验证，跨域适用性未知；需要手动设定 latent 块数、宽度和循环深度，无法动态适应更长的推理链，未来需开发自适应机制。

---

## 515. Policy Optimization Achieves Data-Dependent Regret Bounds in MDPs with Unknown Transitions

**arXiv ID:** 2606.31769 | [PDF](https://arxiv.org/pdf/2606.31769v1)

**作者:** Mingyi Li `[一作]` (University of Tokyo), Kenji Yamanishi `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文针对未知转移核的在线周期性表格马尔可夫决策过程，提出了一种基于乐观跟随正则化领导者的策略优化算法，并给出了适用于全信息和仅反馈两种情形的理论上最优的可调节数据相关经验损失界。

**💡 创新点**

创新点在于首次在未知转移的前提下实现了数据相关的一阶、二阶和路径长度三类 regret 上界，同时引入了乐观 Q‑估计器和基于转移估计误差的自适应 transition 奖励，揭示了不可避免的转移相关复杂度项。

**🔧 技术方法**

采用了乐观跟随正则化领导者 (OFTRL) 框架、置信集构造、动态规划求解最优转移核、log‑barrier 正则化以及虚拟回合调度等技术，并结合自适应学习率与稀疏性奖励实现了对损失序列结构的精细利用。

**📊 数据集**

该工作为纯理论研究，无实验验证，不使用任何真实或仿真数据集。

**📈 对比分析**

通过与已知转移下的最佳‑双世界（best‑of‑both‑worlds）算法和基于占用测度的最坏情况分析对比，证明在任意损失序列下取得了√(H²SA·min{L(π),HT–L(π),Q∞,V1}) 的上界，并在随机环境下实现了对齐的 gap‑dependent O(√(H²S²A/Δmin)) 的优越性能。

**⚠️ 局限性**

主要限制包括：转移相关复杂度项不可消除，导致在未知转移下数据相关上界不如已知转移时紧凑；gap‑dependent 误差项中仍包含不利的 S²/H² 因子；作者未给出方差感知的 gap‑dependent 上界，也未探讨聚合反馈场景。

---

## 516. JETO-Bench: A Reproducible Benchmark for Execution Time Improvement Patches in Java

**arXiv ID:** 2606.31767 | [PDF](https://arxiv.org/pdf/2606.31767v1)

**作者:** Khashayar Etemadi `[一作]` (ETH Zurich), Zhendong Su `[通讯]` (ETH Zurich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出并实现了 JETO‑Bench，一个可配置、可复现的工具，用于从开源 Java 项目中自动提取、构建和评估执行时间改进补丁（ETIP）。

**💡 创新点**

创新点在于：①首次为 Java 提供专门的执行时间改进补丁基准；②设计了三阶段流水线（静态检索、动态容器化构建、评估 harness），支持用户自定义过滤和统计显著性；③将 LLM 用作问题分类器，并结合 Docker 实现全流程可复现；④构建了 660 个已识别、91 个手工验证可执行的 ETIP benchmark。

**🔧 技术方法**

技术方法包括：GitHub API 爬取、基于 LLM 的 issue 分类、静态差异筛选、Docker 化构建（JVM、Maven Wrapper）、多次测试跑和配对统计显著性检验（min_impr、p_val），以及评估 harness 用于 patch 与测试生成的执行反馈。

**📊 数据集**

使用的数据集包含 174 个活跃 Java 项目（至少 20 星，Java 8+），覆盖 1.8M+ 次提交；从中提取了 660 个 ETIP，91 个经过人工验证后可执行，最终生成 102 个可运行的 Docker 镜像。

**📈 对比分析**

评估方法：用 OpenHands 对 91 个可执行 ETIP 进行补丁生成，成功率为 14.3%（13/91）；评估 harness 自动识别大部分错误补丁；静态分析耗时 478h，动态分析 152h，LLM 过滤成本约 5.33 美元；静态检索精度 96%。

**⚠️ 局限性**

局限性包括：仅支持使用 Maven Wrapper 的项目；JIT/GC 的波动导致测量不稳定；大多数项目缺乏 JMH 或能检测执行时间差异的测试；ETIP 检测依赖统计显著性，可能忽略细粒度性能改进；在多文件/多模块场景下成功率仍偏低。

---

## 517. Robocalls: A Worldwide or US-only Problem? Analyzing Spam and Fraud in International Phone Calls

**arXiv ID:** 2606.31790 | [PDF](https://arxiv.org/pdf/2606.31790v1)

**作者:** Kemal Altwlkany `[一作]` (Infobip), Emanuel Lacic `[通讯]` (Infobip)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究通过构建跨国蜂蜜罐系统，收集并发布了首个公开的多模态国际 robocall 数据集，包含 8.7 百万通话记录、677 条录音和 839 条转录文本，并对 robocall 的时空特征、语言差异、协同攻击与内容分类进行了深入分析。

**💡 创新点**

创新点在于首次将 robocall 研究扩展到全球 65 个国家，公开多语言多模态数据，采用联合时间共目标图与文本相似度验证活动协同，结合无监督 DBSCAN 与 LLM 少量样本分类两种方法，全面揭示跨国 robocall 的结构与差异。

**🔧 技术方法**

使用的技术包括蜂蜜罐自动接听与录音、SileroVAD 语音活动检测、Whisper 语音转文本、BGE‑M3 多语言嵌入、Leiden 社区检测、DBSCAN 聚类以及 ChatGPT‑5.4 的少样本分类。

**📊 数据集**

数据集来自 Infobip 提供的 100 000+ 国际电话号码，在 2025‑06‑06 至 2026‑02‑14 期间收集，公开的 8.7 百万 CDR、677 条录音（已人工验证）与 839 条转录文本（已匿名化）组成。

**📈 对比分析**

通过对比美国与国际 robocall 的时空分布、呼叫者集中度、语言与内容相似度，结果显示美国 robocall 规模和严重性更高，但两者在业务模式与攻击策略上呈现共性；实验验证了干预模式可显著降低呼叫量，且文本聚类与 LLM 分类在准确率上相近。

**⚠️ 局限性**

局限性包括：录音时需播放警告导致部分 robocall 回避、样本数有限（仅 677 条录音/839 条文本）、缺乏对单个国家深入细分、对呼叫者 ID 嗅探的误判以及对不同语言声学模型的依赖。

---

## 518. Autonomous UAV Navigation for Individual Wildlife Re-Identification

**arXiv ID:** 2606.31772 | [PDF](https://arxiv.org/pdf/2606.31772v1)

**作者:** Claire Sun `[一作]` (Ohio State University), Jenna Kline `[通讯]` (Ohio State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一套基于无人机的主动视觉系统，能够实时检测、姿态判断、定向、逼近并捕获适合个体再识别（re-ID）的图片。

**💡 创新点**

创新点在于将姿态识别与航拍控制耦合，提出“detect‑orient‑approach‑capture” (DOAC) 任务驱动导航协议，并首次在现场验证了面向个体识别的主动图像采集框架。

**🔧 技术方法**

使用 YOLOv11 进行目标检测，采用 DINOv2‑small 作为姿态分类器；通过边缘设备实现实时推理（检测 4.7 ms，分类 3.5 ms），控制指令通过 SoftwarePilot 与 WildWing 交互。

**📊 数据集**

训练与评估基于 MMLA 数据集（155,074 帧多物种低空航拍）以及 1,023 帧手工标注的姿态子集；测试中针对斑马、长颈鹿、老虎、象等物种进行了验证。

**📈 对比分析**

实验显示，YOLO 在斑马上的 mAP50 为 0.675，姿态分类准确率约 75%；相较于被动悬停拍摄，DOAC 能显著提高满足 500×500 像素阈值且姿态为侧面（可用于 re-ID）的图像比例，表明主动控制提升了数据质量和回收率。

**⚠️ 局限性**

局限性包括：目前仅能针对单个动物实现主动捕获，群体覆盖与多体协调尚未实现；在高空或遮挡、光照不佳等环境下检测与姿态识别准确率下降；缺乏与专家手动操作基线的定量对比及最终 re-ID 精度评估。

---

## 519. A Self-Evolving Agentic System for Automated Generation and Execution of Biological Protocols

**arXiv ID:** 2606.31763 | [PDF](https://arxiv.org/pdf/2606.31763v1)

**作者:** Yankai Jiang `[一作]` (Shanghai Artificial Intelligence Laboratory), Meng Yang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

ProtoPilot 是一种自我进化的多智能体系统，能够把实验室自然语言实验意图转化为可执行的 SOP、设备代码，并通过实验反馈不断优化流程。

**💡 创新点**

其创新点在于层级式多智能体协作与可持续进化的技能库，能够在不同层次保持实验一致性，并通过实时反馈闭环提升实验成功率。

**🔧 技术方法**

主要技术包括层级化智能体协调、可解释验证器、技能库驱动的代码生成、SDK 兼容检查与实验室设备执行模块。

**📊 数据集**

使用了 294 题的合成与分子生物学任务数据集，来自 98 条金标准实验方案，并结合专家手工评测与实际实验验证。

**📈 对比分析**

与 13 种基线系统（如 LabScript‑AI、OpenTrons‑AI 等）对比，ProtoPilot 在协议质量 Top@3 专家偏好率 90.2%，协议至代码门控通过率 96.6%，并在四个不同设备平台上保持 88–91% 的通过率，显示出显著优于现有方法。

**⚠️ 局限性**

局限性包括 benchmark 覆盖范围仍有限，缺乏更大规模、多样化实验和更复杂失败模式；以及系统目前仅聚焦实验执行，尚未与干实验（设计、仿真、数据分析）模块实现完整闭环。

---

## 520. Estimating Velocity of Spheres from Rolling-Shutter Image(s)

**arXiv ID:** 2606.31760 | [PDF](https://arxiv.org/pdf/2606.31760v1)

**作者:** Wenjie Xue `[一作]` (Epson Canada Ltd), Limin Shang `[通讯]` (Epson Canada Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无需3D–2D对应关系的算法，利用滚动快门图像估计球体的三维平移与旋转速度。

**💡 创新点**

创新点在于把滚动快门畸变转化为运动信息，使用后向投影无对应约束的两阶段优化，并设计易检出的球面图案。

**🔧 技术方法**

使用后向投影几何约束、低维两阶段优化、球面几何、以及对多相机/多帧的扩展。

**📊 数据集**

使用Blender仿真生成的滚动快门图像以及真实高速度高旋转高尔夫球拍击实验数据。

**📈 对比分析**

与Ait-Aider等人和全局快门PnP基线对比，实验显示平移误差和角速度误差均显著下降，达到数厘米/秒及数度/秒级精度。

**⚠️ 局限性**

局限性包括需要已知球半径与预先设计的图案、仅适用于球形对象、假设速度恒定。

---

## 521. JL1-CC&QA: Extending the JL1-CD Benchmark with Change Captioning and Question Answering

**arXiv ID:** 2606.31745 | [PDF](https://arxiv.org/pdf/2606.31745v1)

**作者:** Ziyuan Liu `[一作]` (Tsinghua University), Yuantao Gu `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了多任务遥感变化理解基准JL1-CC&QA，融合二值变化掩模、变化描述和问答对。

**💡 创新点**

创新点在于统一同一影像对提供三种任务标注，并采用三阶段管线结合多模态LLM与人工验证的可扩展标注流程。

**🔧 技术方法**

采用多模态LLM生成、视觉感知LLM评判、人工专家校验三阶段管线，使用Kimi-K2.6等大模型。

**📊 数据集**

使用Jilin-1高分辨率光学影像5000对，原始的JL1-CD二值掩模作为基础。

**📈 对比分析**

通过与自动评判分数和人工校验相结合，生成17021条高质量描述和20060条问答，验证通过率分别为68%和80%，表明方法在自然语言标注上取得可行效果。

**⚠️ 局限性**

局限包括依赖LLM生成易产生幻觉、仍以二值掩模为主、数据覆盖范围受限于吉林-1，缺乏跨传感器、多时序的验证。

---

## 522. On Modal Logics of Connectedness in Metric Spaces

**arXiv ID:** 2606.31880 | [PDF](https://arxiv.org/pdf/2606.31880v1)

**作者:** John Harding `[一作]` (New Mexico State University), Ilya Shapirovsky `[通讯]` (New Mexico State University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了度量空间的连通性在模态逻辑中的可表达性与可判定性，给出了单一距离模态（或多模态）下“a‑连通”和“拓扑连通”度量空间的完整公理化，并证明了相应逻辑的有限模型性质；同时通过构造“跳跃”度量空间，将连通度量空间的模型映射到可度量的拓扑空间，从而完成了连通度量空间逻辑的完整性证明。

**💡 创新点**

创新点在于：①为单一距离模态（及包含闭包模态）下的连通度量空间提供了完整的公理化；②引入跳跃（jumps）技术构造与原度量同构但距离被压缩的度量空间，利用该技术实现连通度量空间的模态可表达；③证明了这些逻辑的有限模型属性和可判定性，为多模态连通度量空间的后续研究奠定了基础。

**🔧 技术方法**

主要技术包括：Kripke 过滤技术（finite filtration lemma）、度量闭包与跳跃构造、quasipark 与 cp‑morphism 的图形化映射、以及路径/权重计算证明连通性与距离阈值的等价性。

**📊 数据集**

由于论文为纯理论研究，未使用具体数据集；逻辑模型构造基于抽象度量空间与预序结构。

**📈 对比分析**

通过对比已知的度量空间逻辑（如多模态的完整公理化已知），论文证明了单模态连通度量空间逻辑的完整性与有限模型属性，表明其与先前结果兼容且在表达连通性方面具备更强的可判定性。

**⚠️ 局限性**

局限性：①对多模态（即多种距离模态）下的连通度量空间逻辑的完整公理化仍为开放问题；②构造的跳跃技术仅适用于单一距离模态，无法直接推广；③对“所有正数 a 连通”（well‑chained）度量空间的公理化仅给出猜想，未给出正式证明。

---

## 523. Taming Complexity in Intuitionistic Modal Logic: The Case of FIK and Its Shallow Calculus

**arXiv ID:** 2606.31877 | [PDF](https://arxiv.org/pdf/2606.31877v1)

**作者:** Han Gao `[一作]` (Czech Academy of Sciences), Nicola Olivetti `[通讯]` (Aix-Marseille University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文为一种自然直觉主义模态逻辑（K）的浅层序列计算器（shallow sequent calculus）提供了完整的形式化，并通过证明切割（cut）可接受性和可证性实现了该逻辑的 Expspace 判定复杂度。

**💡 创新点**

创新点在于：①提出仅有单层嵌套的“浅层”序列结构，显著简化了传统双层嵌套或全标记计算器；②通过结构化切割与归约技巧实现切割可接受性，从而避免了高阶递归；③利用紧致（tight）序列与冗余消除技术将证明搜索空间压缩到单指数阶，打破了此前对更强逻辑的非元可计算（non‑elementary）预估。

**🔧 技术方法**

技术方法主要包括：基于前置与后置符号的极化序列、结构化切割规则与归约、序列的大小与深度度量、归约树的重构、紧致序列与块递增约束、以及对证明分支长度的指数上界证明。

**📊 数据集**

该工作纯粹为理论分析，不涉及实验或数据集。

**📈 对比分析**

与已有的简单 Gentzen 计算器（构造性模态逻辑）以及传统的全嵌套/标记计算器（K、Fischer‑Servi 等）相比，本文的计算器在判定复杂度上从此前的非元上限下降到可归约的 Expspace；此外，证明过程中的切割消除与冗余消除显著降低了搜索空间，提升了理论效率。

**⚠️ 局限性**

主要限制包括：仅给出了 Expspace 的上界，缺乏对应的下界；计算器目前仅适用于特定的直觉主义模态逻辑 K，尚未推广到更广泛的直觉主义模态逻辑族；最后，对该计算器的插值性质、对称性等更深层元逻辑属性的探讨仍待进一步研究。

---

## 524. Topological Logics of Path-Reachability

**arXiv ID:** 2606.31874 | [PDF](https://arxiv.org/pdf/2606.31874v1)

**作者:** Aleksandr Gagarin `[一作]` (University of Barcelona), David Fernández-Duque `[通讯]` (University of Barcelona)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

构造并证明了包含路径可达性模态γ与Cantor导数的拓扑逻辑TLR的公理系统，证明其在T₁拓扑和度量空间上的完备性与可判定性，并给出了对应的邻域类语义与有限模型构造。

**💡 创新点**

创新点包括：①引入flanked Kripke框架和邻域类语义来处理γ模态；②通过构造可折叠的有限模型与度量树形路径映射，首次证明γ-逻辑的有限模型性质和对度量空间的完备性；③在c-语义下给出TLR_c的完整性与可判定性。

**🔧 技术方法**

使用的技术主要有：邻域语义与滤镜化、K4/S4公理系统、路径可达性与Cantor导数的语义定义、可折叠的有限模型构造、度量树结构与路径映射、归纳证明与构造性构造。

**📊 数据集**

没有使用任何实验数据集，全部通过形式化证明完成。

**📈 对比分析**

方法与性能：本文未进行实验对比，而是通过理论证明展示了TLR在T₁拓扑与度量空间上的可判定性和完整性，表明该逻辑具备理论上可有效决定真值。

**⚠️ 局限性**

局限性：仅对T₁拓扑（以及度量空间）完成了完整性与可判定性；非T₁空间中γ与Cantor导数的交互存在未捕获的现象；对欧几里得平面γ-逻辑的完整性仍未解决。

---

## 525. The Logic of Data Access and Data Exchanges

**arXiv ID:** 2606.31858 | [PDF](https://arxiv.org/pdf/2606.31858v1)

**作者:** Alexandru Baltag `[一作]` (University of Amsterdam), Sonja Smets `[通讯]` (University of Amsterdam)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出一种新的动态逻辑，结合了传统的知识模态与关于数值变量的“知道何者”模态，并引入条件性知识、可数限制以及最小化描述子；构造了可处理多种数据交换事件（如公共宣布、秘密黑客、数据库共享等）的动态扩展，并给出完整的公理化体系与可判定性证明。

**💡 创新点**

创新点在于：1) 引入了|x|_A^φ≤N 这类条件性可数限制，能够表达组在给定信息下将某变量的可能取值数缩小到最多N个；2) 使用了基于截断最小化（cutoff minimization）和n_N.x_A^φ 这类描述子来命名这些可能值；3) 通过定义通用的数据交换事件模型，覆盖了从公开宣布到私有黑客等多种信息流；4) 证明了整个逻辑的完整性、可判定性，并在理论上与其静态子逻辑共同可表达。

**🔧 技术方法**

技术主要包括：模态逻辑与知识论的公理化；延伸的语义模型（包括变量赋值、可数限制与截断最小化）；过滤法构造准模型；树展开与关系重定义以恢复标准可解释性；利用有序域与选择公理（Axiom of Choice）实现总序的存在；以及 DEL 风格的归约公理来处理动态操作。

**📊 数据集**

该工作为理论性逻辑研究，无使用任何经验数据集，所有结论均来自形式证明与语义构造。

**📈 对比分析**

本论文没有实验比较与性能评估，主要通过形式化证明展示逻辑的完备性、可判定性及其与静态逻辑的互可表达性，理论层面的“性能”体现在能否在有限步骤内给出完整的推理系统。

**⚠️ 局限性**

局限性包括：1) 未实现公共知识 C_Aφ 的动态，需在未来工作中引入多参数条件；2) 目前缺乏有限模型性质（FMP），因此无法用简单最小化代替截断最小化；3) 逻辑仍未覆盖某些更细粒度的知识动态（如自我更改密码后可视性变化）；4) 证明过程依赖于选择公理，导致构造上有非构造性。

---

## 526. An Open-Source Tool for Reproducible Freeway Network Extraction from OpenStreetMap

**arXiv ID:** 2606.31857 | [PDF](https://arxiv.org/pdf/2606.31857v1)

**作者:** Drew Miller `[一作]` (Massachusetts Institute of Technology), Cathy Wu `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一个开源工具osm2macrosim，自动从OpenStreetMap提取高速公路网络并生成适用于宏观模拟的站点参考表示，同时提供可视化验证工具。

**💡 创新点**

设计了面向高速公路的提取逻辑，解决了主线识别、管理车道、交叉口路径模糊、缓冲框查询等问题，并实现提取‑验证两阶段工作流，显著降低人工编码成本。

**🔧 技术方法**

基于Python和Overpass API构建有向图，使用正则表达式与分阶段路径搜索进行主线隔离，线性映射后产生站点参考lane/ramp表格，并导出GMNS；提供网页HTML+KML前端可视化。

**📊 数据集**

主要使用OpenStreetMap原始道路数据，以及加州橙县PeMS站点坐标和空中影像进行验证；在测试中提取I‑24、I‑35W两条高速和橙县359.6英里的高速网络。

**📈 对比分析**

与人工手工编码（主线与匝道）对比，提取‑验证工作耗时约为手工的1/3；在橙县部署平均每英里41秒完成提取与验证，整体效率提升显著。

**⚠️ 局限性**

仍需人工对齐与空中影像验证，尤其是车道属性不一致；对地图质量不足的区域（如农村地区）的适用性不确定；缺乏完全自动化验证和跨地区泛化能力。

---

## 527. Resolving Asynchronous Distributed Knowledge

**arXiv ID:** 2606.31855 | [PDF](https://arxiv.org/pdf/2606.31855v1)

**作者:** Philippe Balbiani `[一作]` (IRIT, CNRS---INPT---UT), Clara Lerouvillois `[通讯]` (IRIT, CNRS---INPT---UT)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

提出了异步分布式知识的逻辑，并给出了其形式化语义与推理规则；

**💡 创新点**

在传统同步分布式知识逻辑的基础上，首次引入基于历史的语义，允许各代理仅观察到自身参与的更新，从而捕捉真实异步通信中的不确定性；

**🔧 技术方法**

利用Kripke模型、交叉更新模态、历史序列与等价关系的形式化，以及无穷推导规则和正则化公理来完成语义定义和完备性证明；

**📊 数据集**

无数据集；

**📈 对比分析**

无实验或性能对比；该工作属于纯理论推导；

**⚠️ 局限性**

尚未确定异步与同步语义下有效性是否相同；未证明是否存在有限公理化；可判定性与复杂度未知；缺乏对公共知识的扩展；

---

## 528. Towards Voxel Spacing Consistency for Medical Image Segmentation

**arXiv ID:** 2606.31839 | [PDF](https://arxiv.org/pdf/2606.31839v1)

**作者:** Xin You `[一作]` (Shanghai Jiao Tong University), Yun Gu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计并实现了 Consispace，一种统一体素间距一致化框架，结合 ODE 驱动的解剖连续性与预训练视觉模型的语义一致性，在体素重采样过程中提升重建质量并促进下游医学图像分割性能。

**💡 创新点**

①引入连续时间 ODE 作为解剖动态约束，捕捉相邻切片的形态变化；②设计反向注意力的即时速度场实现更精准的切片间迁移；③利用 DINOv3 提取稠密特征构建切片内语义相关图，并通过特征重加权实现类级语义一致；④将上述两类约束集成到隐式神经网络（INR）中，实现任意尺度的连续重采样。

**🔧 技术方法**

使用 ODE‑based interpolator、反向注意力机制、隐式神经表示（INR）、预训练 DINOv3 语义特征、稠密特征相关图、特征重加权、线性/三次插值对比、nnU-Net、Swin UNETR、MedSAM‑2、MSE+SSIM 重建损失、Dice+CE 分割损失及数据增强。

**📊 数据集**

TopCow 2024（脑血管 MRI/分割）、BraTS 2020（脑肿瘤 MRI/分割）、SPIDER（腰椎 MRI/分割）以及相应的低/高分辨率配对。

**📈 对比分析**

与传统线性/三次插值、SAINT、TVSRN、ArSSR、CycleINR、DP‑INR 等重采样方法以及 SynthSeg、HyperSpace 等间距一致化方法进行对比；在重采样任务中 Consispace 在 PSNR、SSIM、NMSE、LPIPS 上均优于所有竞争方法；在分割任务中，使用 Consispace 预处理后，nnU-Net、Swin UNETR、MedSAM‑2 的 Dice 分数提升约 1–2%，HD95 明显下降。

**⚠️ 局限性**

模型主要在特定解剖结构上验证，泛化到未知结构仍受限；训练数据规模和多样性有限；对高分辨率输入时 GPU 内存消耗相对较大；未对多模态医学影像（如 CT）进行充分验证。

---

## 529. Real-Time Source-Free Object Detection

**arXiv ID:** 2606.31834 | [PDF](https://arxiv.org/pdf/2606.31834v1)

**作者:** Sairam VCR `[一作]` (IIT Hyderabad), Muhammad Haris Khan `[通讯]` (MBZUAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了RT‑SFOD框架，基于YOLOv10实现源自由目标检测的实时适配。

**💡 创新点**

通过Dual‑Head伪标签融合（DHF）与多尺度自适应表征多样化（MARD）两项技术，提升伪标签质量与特征多样性。

**🔧 技术方法**

采用Mean‑Teacher自训练、NMS‑free双头检测器、伪标签融合策略和特征秩恢复的方差‑协方差约束等技术。

**📊 数据集**

在Cityscapes→Foggy Cityscapes、Sim10k→Cityscapes、KITTI→Cityscapes、BDD100k等多域迁移基准上验证。

**📈 对比分析**

与现有SFOD与UDAOD方法对比，RT‑SFOD在mAP提升1.4–3.5%，同时FPS提高1.3倍，参数量约为先行方法的一半，显著改进速度‑精度‑模型大小三元平衡。

**⚠️ 局限性**

仅适用于双头无NMS检测器，对单头结构效果有限，且在极端域差时仍受伪标签噪声影响。

---

## 530. Geometry-Preserving Orthonormal Initialization for Low-Rank Adaptation in RLVR

**arXiv ID:** 2606.31813 | [PDF](https://arxiv.org/pdf/2606.31813v1)

**作者:** Ruijia Zhang `[一作]` (Johns Hopkins University), Laixi Shi `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了低秩适配（LoRA）在强化学习可验证奖励（RLVR）框架下的初始化问题，揭示了结构化SVD初始化（如PiSSA、MiLoRA）在RLVR中的不稳定性，并提出了几何保持正交初始化方案以提升训练稳定性和性能。

**💡 创新点**

创新点包括：1）对LoRA在RLVR中的优化动力学进行理论分析，证明正交初始化能最小化与全参数微调的误差；2）提出两种几何保持正交初始化（对主子空间和尾子空间），既保留预训练权重的谱信息，又避免了奇异值缩放导致的振荡；3）通过实验解释了PiSSA、MiLoRA在RLVR中失效的原因，为未来的低秩微调提供了理论指导。

**🔧 技术方法**

采用的技术主要有：LoRA参数化、奇异值分解（SVD）、正交矩阵初始化、RLVR（DAPO）算法、KL散度监测、梯度范数分析、理论优化动态推导、实验对比评估。

**📊 数据集**

使用的数据集包括：数学推理基准 DAPO‑Math‑17k、GSM8K、MATH500、AIME 2022/2023/2024；在附录中还评估了 GLUE、代码生成任务、不同模型规模（Llama‑3.2‑3B‑Instruct、Qwen2.5‑7B‑Instruct）等。

**📈 对比分析**

方法对比采用标准LoRA、PiSSA、MiLoRA、DCT‑LoRA、Wavelet‑LoRA等。评价指标为各基准上的准确率（@1、@4、@32）和训练过程中的KL散度。新方法在所有基准上均优于标准LoRA（平均准确率 65.03% 对比 62.40%），且KL轨迹更低、更稳定；PiSSA、MiLoRA 训练不稳定或性能明显下降。

**⚠️ 局限性**

局限性：①理论分析基于小学习率假设，实际训练仍需学习率调度；②仅在 DAPO‑RLVR 框架下验证，其他 RLVR 算法效果未知；③实验主要聚焦数学推理任务，跨领域或更大模型的推广需进一步验证；④对初始化参数（如秩 r）的敏感性及计算 SVD 的开销在大模型上仍是挑战。

---

## 531. Large Databases Need Small, Open-Weight Language Models

**arXiv ID:** 2606.31808 | [PDF](https://arxiv.org/pdf/2606.31808v1)

**作者:** Parker Glenn `[一作]` (Capital One), Alfy Samuel `[通讯]` (Capital One)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 BlendSQL v0.1.0，一种将小型开源量化语言模型本地化部署在数据库中的 UDF‑中心系统；

**💡 创新点**

创新点在于通过量化模型、约束解码、级联过滤、早期退出与去重等软件层优化，实现低成本高效的 LM‑DB 集成；

**🔧 技术方法**

使用 Gemma 3/4 系列量化模型、vLLM、LLM‑Guide 约束解码、SQL 解析器与优化器等技术；

**📊 数据集**

使用 SemBench 基准（Movie、Wildlife、E‑commerce、Cars、MMQA）以及 DuckDB 数据库；

**📈 对比分析**

对比现有 LOTUS、ThalamusDB、BigQuery 等系统，在 16GB 显存下取得平均质量与延迟相当甚至更好，成本降低 390 倍、延迟降低 3.8 倍；

**⚠️ 局限性**

局限性在于对多模态（尤其音频）处理效果不佳，且现有基线系统对本地多模态支持有限，未来需完善多模态与更大规模模型的支持。

---

## 532. Reinforcement Learning-Based Control for an Inline Skating Humanoid Robot

**arXiv ID:** 2606.31807 | [PDF](https://arxiv.org/pdf/2606.31807v1)

**作者:** Ethan Marot `[一作]` (ETH Zurich), Raffaello D'Andrea `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

训练并部署了强化学习控制策略，使改装为消费级滑板的 humanoid 机器人能够实现动态平衡、滑行和转向。

**💡 创新点**

通过在训练与验证阶段使用不同几何轮子模型（球体/椭球体）、滚动奖励以及成功式命令课程，克服了物理仿真中的接触伪影，达成零射门的 sim‑to‑real 转移。

**🔧 技术方法**

采用了 Isaac Gym / MuJoCo 物理仿真、异步演员‑评论家 PPO、球体与椭球轮子建模、滚动奖励与命令课程、以及 Booster T1 机器人硬件改装。

**📊 数据集**

主要使用自生成的强化学习仿真数据，未使用公开数据集；通过 8092 个并行环境累计约 1.9 年的模拟经验。

**📈 对比分析**

与无滚动奖励、无课程或圆柱轮子等消融实验对比，验证在 MuJoCo 的滚动误差低于 0.5 m/s，零射门部署在 Booster T1 上实现 0.6 m/s 以上的高效滑行，成本运输（CoT）降低约 50%。

**⚠️ 局限性**

受限于单侧“主腿”推进策略、未实现交替腿滑行、对高转速命令覆盖不足，以及在真实环境中未测试完整速度范围，且未适配冰面等更复杂摩擦条件。

---

## 533. Counting Small Induced Subgraphs: Hardness of Symmetry-Based Properties

**arXiv ID:** 2606.31803 | [PDF](https://arxiv.org/pdf/2606.31803v1)

**作者:** Radu Curticapean `[一作]` (University of Regensburg), Mingjun Liu `[通讯]` (University of Regensburg)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本研究证明了在给定图性质Φ（如自同构群为固定有限群Q）的情况下，计数满足Φ的k-顶点诱导子图是#W[1]-难的；

**💡 创新点**

创新点在于提出“clique scaffold”构造，利用特定限制实现从k-团问题的参数化归约，首次把自同构群限制纳入可判定的计数困难范畴；

**🔧 技术方法**

核心技术包括对自同构群的组论构造（Frucht定理与Key gadget）、图论中的自由边设定、以及基于布尔函数的OR限制的组合化简；

**📊 数据集**

研究完全基于理论分析，不使用任何实验数据集；

**📈 对比分析**

与现有的参数化计数方法相比，本文提供了严格的硬度下界，说明不存在任何多项式时间（或小指数）算法，性能上即为不可实现；

**⚠️ 局限性**

局限性在于仅给出了困难性证明，未给出匹配的上界或改进算法，也未探讨更广泛图性质的可行性。

---

## 534. Harnessing Textual Refusal Directions for Multimodal Safety

**arXiv ID:** 2606.31876 | [PDF](https://arxiv.org/pdf/2606.31876v1)

**作者:** Moreno D'Incà `[一作]` (University of Trento), Nicu Sebe `[通讯]` (University of Trento)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用文本拒绝方向在多模态大语言模型中实现安全提升，无需多模数据或额外训练

**💡 创新点**

提出轻量级、训练自由的插值方法，包含激活重中心化、动态调节驱动强度以及自适应层级选择，克服模态失配导致的误拒

**🔧 技术方法**

激活空间驱动（Refusal Direction），ReLU门控，几何信任区间内自适应步长，层级重要性评分，随机色彩图像重中心化

**📊 数据集**

Alpaca、MaliciousInstruct（文本拒绝方向来源）；ViSU、HADES、MMSafetyBench（图像安全评估）；MMMU、MMMUPro（通用能力评估）；VideoSafetyBench（视频jailbreak）

**📈 对比分析**

与AdaSteer（文本驱动）、ECSO（自评驱动）、SASA（训练驱动）对比，在五款主流MLLM上实现+25.1%至+67.9%的拒绝率提升，视频jailbreak上+59.4%提升，且对安全输入的误拒率低于训练驱动方法，保持大部分通用能力

**⚠️ 局限性**

假设文本拒绝方向已存在，且中心化可线性纠正模态失配；对高度非线性激活空间的适用性有限；重中心化后仍有少量安全输入误拒，需要进一步改进

---

## 535. Hyperformalism for Relevant Modal Logics

**arXiv ID:** 2606.31872 | [PDF](https://arxiv.org/pdf/2606.31872v1)

**作者:** Thomas Macaulay Ferguson `[一作]` (Rensselaer Polytechnic Institute), Shay Allen Logan `[通讯]` (Kansas State University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了相关模态逻辑的超形式性，定义并证明了 B^ 及其扩展在 MPos‑超形式性下的封闭性，并提出了 K^MPos 作为经典模态逻辑 K 的最大 MPos‑超形式子逻辑，给出了其语义模型与表格系统。

**💡 创新点**

创新点包括：①引入 MPos‑超形式性与对应的非均匀替代概念；②构造并证明 K^MPos 作为 K 的最大此类子逻辑；③通过位置等价关系实现变量共享性质的精细化；④为 K^MPos 提供语义与表格两种完整的表述。

**🔧 技术方法**

使用的技术主要有：非均匀替代、等价关系、位置符号化、Kripke 语义（含可变世界维度）、表格系统、归纳证明、典型模型构造、完整性与可传递性证明等。

**📊 数据集**

该工作不涉及实验数据集，全部以形式化推导与证明为主。

**📈 对比分析**

与传统相关模态逻辑相比，本文证明 B^ 与 K^MPos 在变量共享、闭包（对 MPos‑替代的封闭性）以及可传递性等方面具备更强的性质；K^MPos 作为 K 的最大子逻辑，提供了更广泛的适用范围。

**⚠️ 局限性**

主要局限在于：①尚未给出 K^MPos 的直观公理化体系；②部分结果对特定位置等价关系的依赖使得实现与进一步扩展（如更复杂的模态）仍需研究；③缺乏实验或计算机实现验证，理论证明尚待进一步简化。

---

## 536. Goldblatt-Thomason Theorem for Probability Logic

**arXiv ID:** 2606.31865 | [PDF](https://arxiv.org/pdf/2606.31865v1)

**作者:** Somayeh Chopoghloo `[一作]` (Institute for Research in Fundamental Sciences), Reihane Zoghifard `[通讯]` (Amirkabir University of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文在概率逻辑（𝖯𝖫）的框架下，给出了马尔可夫过程的 Goldblatt‑Thomason 定理，并证明了 Harsanyi 型空间及有限马尔可夫过程的可定义性。

**💡 创新点**

创新点在于将经典的 Goldblatt‑Thomason 定理迁移到概率逻辑环境，克服了概率逻辑缺乏紧致性的问题，并通过引入嵌套 Archimedean ultrafilter、Stone‑Markov 过程等新概念，构造了适用于马尔可夫过程的 ultrafilter 扩张，从而完成了可定义性的等价表征。

**🔧 技术方法**

主要技术包括：模型论的子过程、事件子过程与离散并置构造；Zigzag‑morphism（可测的反射映射）；Stone 对偶与 σ‑生成布尔代数的利用；对概率公式的归约与 Jankov‑Fine 公式的构造；以及对有限马尔可夫过程的局部 zigzag‑morphism 与点生成子过程的分析。

**📊 数据集**

本研究为纯理论论文，未使用具体实验数据集；所有论证均基于数学证明和构造。

**📈 对比分析**

对比方法主要是与经典 Kripke 结构下的 Goldblatt‑Thomason 定理以及已有的概率逻辑可定义性结果进行对照；在理论上证明了在满足闭包性质（子过程、事件子过程、离散并置、Zigzag‑morphism、ultrafilter 扩张）下，马尔可夫过程的可定义性与 GT‑性质等价，性能上不存在数值指标。

**⚠️ 局限性**

限制包括：仅适用于可数生成的 σ‑代数马尔可夫过程；对有限马尔可夫过程的结果仅涵盖 rational 概率内核；缺少对无穷或非可数生成结构的扩展；且证明依赖于 Stone‑Markov 过程的 Baire 性质，无法直接推广至更一般的测度空间。

---

## 537. Most Properties are Undecidable for Transitive Tense Logics

**arXiv ID:** 2606.31863 | [PDF](https://arxiv.org/pdf/2606.31863v1)

**作者:** Qian Chen `[一作]` (Tsinghua University), Tenyo Takahashi `[通讯]` (University of Amsterdam)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

研究了传递时态逻辑的性质的可判定性，证明了在传递时态逻辑的格 K4 中，大多数性质是不可判定的，包括 Kripke 完备性、有限模型性质和可判定性。

**💡 创新点**

提出了一种通用标准来证明 K4 中性质的不可判定性，适应了 Chagrov 的方法，通过将一个关于 Minsky 机器的不可判定问题归约到逻辑性质的决策问题。

**🔧 技术方法**

使用了 Minsky 机器的归约方法，构造了从不可判定问题到逻辑性质决策问题的可计算归约。

**📊 数据集**

没有具体提到使用的数据集，但研究涉及的逻辑性质和 Minsky 机器的配置。

**📈 对比分析**

通过与已知的不可判定问题进行归约，证明了 K4 中的多个逻辑性质是不可判定的，性能上显示出这些性质在 K4 中的复杂性。

**⚠️ 局限性**

未能直接从 K4 的不可判定性结果推导出 K 和 K4 的性质不可判定性，且 K4 的性质与单模逻辑的性质之间的关系仍然不明确。

---

## 538. Review Residuals: Update-Conditioned Residual Gating for Transformers

**arXiv ID:** 2606.31859 | [PDF](https://arxiv.org/pdf/2606.31859v1)

**作者:** Kyle Kramer `[一作]` `[通讯]` (NeraTech LLC), Kyle Kramer (NeraTech LLC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为Review Residuals的残差连接方式，在每层中用学习得到的门控对更新进行缩放并加到残差流中，门控同时考虑当前状态和待更新内容；

**💡 创新点**

创新点在于门控条件既包含状态也包含更新本身，是首个在残差更新上做输入相关缩放且条件为更新的机制；

**🔧 技术方法**

采用Transformer解码器架构，使用RMSNorm、GELU MLP、残差投影缩放以及标准的AdamW优化器和余弦退火学习率调度；

**📊 数据集**

在TinyStories文本数据集上进行下一词预测训练；

**📈 对比分析**

通过与参数匹配的标准残差和Highway门控进行对比，使用多种随机种子验证，发现模型规模≥590M时Review Residuals显著降低验证损失（p<0.05），在1B规模下表现更佳但仅为趋势；

**⚠️ 局限性**

局限包括：增益绝对值仅在≤1B规模时约1%以内；仅在TinyStories单一数据集且未做下游任务评估；只验证了两个大规模点，缺乏更高规模的进一步验证；

---

## 539. Z-1: Efficient Reinforcement Learning for Vision-Language-Action Models

**arXiv ID:** 2606.31846 | [PDF](https://arxiv.org/pdf/2606.31846v1)

**作者:** Lang Cao `[一作]`, Yitong Li `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种针对流式视觉语言动作模型的强化学习后训练框架Z-1。

**💡 创新点**

创新点在于引入共享前缀和树结构前缀分支的GRPO、成功感知奖励衰减以及基于训练诊断的可选择视觉语言模块与动作专家联合训练。

**🔧 技术方法**

使用了GRPO（分组相对策略优化）、共享前缀/树结构前缀分支、奖励衰减、VLM-AE联合训练等技术，并在π_0.5基础上实现。

**📊 数据集**

采用公开的RoboCasa演示数据集进行监督微调，并在该数据集上进行RL后训练。

**📈 对比分析**

与GR00T、GR00T N1.5、Video Policy、X-WAM等之前公布的RoboCasa SOTA 进行对比，Z-1 RL 在24项任务上的平均成功率达80.6%，相较SFT提升13.2个百分点，整体超过所有基线。

**⚠️ 局限性**

局限性包括对炉灶任务的表现仍不及X-WAM，且在长程传输、空间推理和失败恢复方面仍面临挑战。

---

## 540. An Agentic AI Framework to Accelerate Scientific Discovery in Plant Phenotyping

**arXiv ID:** 2606.31831 | [PDF](https://arxiv.org/pdf/2606.31831v1)

**作者:** Renan Souza `[一作]` (Oak Ridge National Laboratory), David Weston `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了一个端到端代理 AI 框架，将 Oak Ridge 的高通量植物表型实验室从数据工厂转变为交互式发现平台，自动完成多模态图像的 AI 预处理、Vision Transformer 分割、特征提取，并通过多代理协作实现秒级查询与即时分析；

**💡 创新点**

创新点包括将 HPC 级别的 ViT 感知模型与云端对话代理、HPC 计算代理的多域架构结合，使用安全跨域通道（S3M）和 FlowCept 记录完整可追溯的实验过程，打造真正的可审计、可复现的交互式科学工作流；

**🔧 技术方法**

技术手段涵盖 Vision Transformer 分割、Parsl 任务调度、Academy 多代理框架、Chainlit + LiteLLM 对话接口、FlowCept 可追溯系统、S3M 安全通道、Frontier 超级计算机 GPU 集群；

**📊 数据集**

使用了 Oak Ridge Advanced Plant Phenotyping Laboratory 的多模态影像数据集，包括 RGB、红外、光谱、热、光谱、3D 激光、根系摄像等，覆盖 10,400+ 株植物的跨时序实验；

**📈 对比分析**

与传统手工分析流水线相比，单次查询从数小时/两周降至约 5 分钟，后续跟进可在秒级完成，整体速度提升超过 10 倍；

**⚠️ 局限性**

局限性包括：尚未实现对所有模态（如 3D、光谱）的完整特征提取；对高性能 GPU 资源依赖强；多域部署的安全与通信复杂度高；泛化到其他物种或实验环境的验证仍在进行中。

---

## 541. Interface-Variant Dynamics in Software Ecosystems: Resolver-Induced Selection and Adoption in Package Graphs

**arXiv ID:** 2606.31817 | [PDF](https://arxiv.org/pdf/2606.31817v1)

**作者:** Faruk Alpay `[一作]` (Bahcesehir University), Baris Basaran `[通讯]` (Bahcesehir University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过从 npm、Maven Central、PyPI 和 Cargo 等主流包管理器收集依赖图和解析器冲突数据，构建并校准可测的选择系数，利用进化图论的配对比较动力学对接口变体在软件生态中的传播进行前向固定化评估，并与拓扑基线进行对照，最终用保留时间与保养状态的采样验证采用率与解析器机制之间的关系。

**💡 创新点**

将软件包解析器行为量化为生态系统特定的选择系数，并将该系数直接植入进化图论模型，实现了从局部兼容性测试到全局种群动力学的可重复估计流程；同时提出了校准门控、拓扑诊断和迁移后保留度分层等多维度验证方法，打破了传统“兼容性标签”与“种群选定”之间的耦合。

**🔧 技术方法**

进化图论（pairwise‑comparison dynamics）、Fermi更新规则、Monte Carlo 固定化模拟、Wilson 区间、Bootstrap 置信区间、Ohtsuki/Allen 结构系数、方向置换检验、Brier 分数/AUC 预测、层级贝叶斯一致性模型。

**📊 数据集**

包管理器依赖图数据集（npm、Maven Central、PyPI、Cargo），解析器冲突实验结果，历史接口变更语料（Kafka、Thrift、Confluent、GraphQL、gRPC、ROS 2、OpenTelemetry、DDS‑XTypes）以及从官方注册中心提取的时间戳化版本历史。

**📈 对比分析**

对比了测得选择系数下的固定化概率与中性 1/N 的基准，使用拓扑基线（度保留、聚类保留、消费者-提供者投影）计算 Δρ；在保留度分层下的方向置换检验中，checker‑derived 方向显著优于随机，但在无方向的时间预测模型（仅用年龄、解析器结果和生态系数）中并未显著优于仅基于年龄的基线，表明解析器信号的独立预测力尚不充分。

**⚠️ 局限性**

核心限制包括：解析器冲突仅捕捉声明的依赖关系，未覆盖实际构建时的可选依赖、环境特定标记、构建脚本和运行时耦合；checker‑derived 方向信息与结果高度共线，导致诊断性而非独立证据；生态系统间差异和注册中心漂移限制了跨时间验证；校准语料仍有限，导致分类器误差需传播，影响选择系数精度。

---

## 542. RAISE: LLM-based Automated Heuristic Design with Robust Adversary Instance Search

**arXiv ID:** 2606.31801 | [PDF](https://arxiv.org/pdf/2606.31801v1)

**作者:** Fei Liu `[一作]` (University of Zurich), Nicola Serra `[通讯]` (University of Zurich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种鲁棒的LLM驱动启发式设计框架RAISE，结合受限对抗实例搜索以提升在分布偏移下的性能。

**💡 创新点**

创新点是将鲁棒性转化为实例级ε-球下的二阶极小化问题，并采用LLM无关的对抗搜索在训练分布附近发现最难实例，实现对未知分布的自适应。

**🔧 技术方法**

技术包括大型语言模型作为进化算子、基于基函数分布编码与边界投影的对抗实例搜索、双循环演化框架。

**📊 数据集**

使用了在线二进制包装、在线作业车间调度、在线车辆路径问题的多个分布族，共计95个不同分布的数据集。

**📈 对比分析**

与传统手工启发式、现有LLM启发式设计（EoH、ReEvo、PartEvo）以及鲁棒性方法（EoH‑S、MoH）对比，RAISE在所有分布偏移测试中保持最优或第二最优，且在某些偏移下比最好的对比方法提升达19倍。

**⚠️ 局限性**

局限性在于仅考虑单维分布偏移、只验证在线组合优化任务，未涵盖多维关联偏移、离线/混合场景及更广泛问题类别。

---

## 543. Pointed Modal Abelian Logic, Algebraically

**arXiv ID:** 2606.31882 | [PDF](https://arxiv.org/pdf/2606.31882v1)

**作者:** Filip Jankovec `[一作]` (Institute of Computer Science of the Czech Academy of Sciences), Wolfgang Poiger `[通讯]` (Institute of Computer Science of the Czech Academy of Sciences)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了实数的指向模态逻辑，建立了其关系（Kripke）语义，并引入了负指向模态阿贝尔ℓ-群的种类，特别关注其强指向成员。

**💡 创新点**

创新点在于引入了负指向模态阿贝尔ℓ-群的种类，并证明了与该种类相关的‘无穷代数完备性’结果。

**🔧 技术方法**

使用了阿贝尔ℓ-群的代数技术和Kripke框架的构造方法。

**📊 数据集**

未具体提及使用的数据集，但研究涉及实数的模态逻辑。

**📈 对比分析**

通过构造复杂代数和规范框架，建立了关系和代数框架之间的真理引理，表明了代数完备性。

**⚠️ 局限性**

限制在于有限公理化无法完全捕捉实数的Kripke有效性，未来研究可能需要探索其他方法来解决这些问题。

---

## 544. Tight bounds for clique-packing parameterized by clique-width

**arXiv ID:** 2606.31873 | [PDF](https://arxiv.org/pdf/2606.31873v1)

**作者:** Narek Bojikian `[一作]` (Humboldt Universität zu Berlin), Stefan Kratsch `[通讯]` (Humboldt Universität zu Berlin)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了给定整数 d ≥ 3 的 d‑Clique Packing（在图中寻找至少 t 个互不相交的 d‑clique 的问题），提出了在已给定 clique‑width k 的图上运行时间为 n^(k^d−1) 的 XP‑算法，并证明了该时间复杂度在 ETH 的假设下是最优的（即无法在 n^o(k^d−1) 的时间内解决）。同时证明了该问题在按 clique‑width 参数化时属于 W[1]‑难，并给出了相应的参数化归约。

**💡 创新点**

创新点主要包括：
- 给出了 d‑Clique Packing 在 clique‑width 参数化下的首次严格上界和下界，形成完整的“tight bound”。
- 通过将 NLC‑expression 与“指纹压缩”技术相结合，构造了高效的动态规划方案，显著降低了状态空间；
- 引入了 d‑Binomial Multi‑Colored Clique 的概念，并利用它完成了从 Multi‑Colored Clique 到 d‑Clique Packing 的参数化难度归约，进一步证明了 W[1]‑难和 ETH 下的最优性。

**🔧 技术方法**

主要技术手段：
- NLC‑expression 及其与 clique‑width 的转换；
- 状态压缩（fingerprint）技术：只记录 d‑clique 之前未达到大小 d 的“标签多重集”而不是完整的标签集合；
- 动态规划结合 join、relabel、union 等操作，递归构造状态表；
- 复杂的多阶段参数化归约，构造“颜色 gadget”“边 gadget”“选择 gadget”等，用于从 (d‑1)‑Binomial Multi‑Colored Clique 构造 d‑Clique Packing 的实例。

**📊 数据集**

本工作为理论算法研究，未使用任何实际数据集；所有证明均在理论模型（图和 clique‑width 表达式）下完成。

**📈 对比分析**

性能方面：
- 提出的算法时间复杂度为 n^(k^d−1)，在给定 clique‑width 表达式的前提下已是最优；
- 通过 ETH 下的下界证明，任何 n^o(k^d−1) 的算法都会违背 ETH，说明目前已达到最优性能；
- 与之前已知的类似问题（如 Edge Dominating Set、Max Cut）相比，d‑Clique Packing 的依赖形式为 n^(k^d) 而非 n^k，体现了更高的组合复杂度。

**⚠️ 局限性**

局限性：
- 仅适用于 clique‑width 参数化的图，若仅给定普通图则仍是 NP‑完全；
- 只考虑了 d‑clique（完全子图）而非更一般的 H‑子图打包；
- 对于实际应用场景，构造 clique‑width 表达式本身可能是困难的；
- 该研究主要聚焦理论上最优性，对算法在大规模实例上的实际可行性尚未评估。

---

## 545. Labelled Sequents for Inquisitive First-Order Modal Logic

**arXiv ID:** 2606.31868 | [PDF](https://arxiv.org/pdf/2606.31868v1)

**作者:** Ivano Ciardelli `[一作]` (University of Padua), Simone Conti `[通讯]` (University of Padua)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文为第一阶询问式模态逻辑（含全局超支配）构造了一个标记化序列推理系统，并证明其完备性、可逆性与剪裁可接受性。

**💡 创新点**

创新点在于首次为此逻辑提供证明系统，巧妙利用有限一致性（finite coherence）引入模态规则，并通过典型化模型构造实现完备性证明。

**🔧 技术方法**

主要技术包括：标记化序列推理、有限一致性分析、结构化证明理论（可逆规则、剪裁可接受性）以及对身份谓词的系统扩展。

**📊 数据集**

该研究不涉及实验数据集，所有结果均为理论推导与形式化证明。

**📈 对比分析**

由于研究为理论性质，未进行实验比较或性能评估，结果以逻辑等价性与证明系统的完备性为衡量。

**⚠️ 局限性**

局限性包括：尚未处理含函数符号的签名、其它询问式模态（如邻域模态）的适配、插值性等元理论问题以及自动证明搜索的实现。

---

## 546. Labelled Sequent Calculi for Propositional Team Logics

**arXiv ID:** 2606.31860 | [PDF](https://arxiv.org/pdf/2606.31860v1)

**作者:** Fausto Barbero `[一作]` (University of Helsinki), Fan Yang `[通讯]` (Utrecht University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

为四种基于团队语义的命题逻辑（基础探询逻辑、命题直觉依赖逻辑及其带张量析取的扩展）构造了标签序列推理系统，并证明了它们的完全性与可终止的证明搜索算法。

**💡 创新点**

提出了模块化标签序列计算器，使用标签表示团队的集合操作，简化了张量析取的规则，并通过有限原子假设引入新的顺序规则以实现可终止的证明搜索。

**🔧 技术方法**

标签序列计算、Gentzen 样式结构化推理、团队语义、可终止证明搜索、归纳证明、可交换与收缩可合规则、构造性证明方法。

**📊 数据集**

无（理论论文，无实验数据集）。

**📈 对比分析**

通过理论证明与先前系统对比，证明了可终止性、可交换、收缩可合以及与 Hilbert 系统的一致性；未涉及实验性能评估。

**⚠️ 局限性**

仅适用于有限原子数的语言，张量析取的处理仍相对复杂，未覆盖无直觉蕴含或无张量的下闭团队逻辑；未来需要扩展至无限原子或其他团队逻辑。

---

## 547. Explicit Fuzzy Logic in the Feed-Forward Layer: Self-Forgetting Quantifiers Discover Legible Grammatical-Licensing Detectors

**arXiv ID:** 2606.31845 | [PDF](https://arxiv.org/pdf/2606.31845v1)

**作者:** Mark Oskin `[一作]` `[通讯]` (University of Washington), Mark Oskin (University of Washington)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种参数中性Transformer FFN，使用显式模糊集合运算（交集、差集）并引入补集以实现可解释的否定。

**💡 创新点**

在不增加参数的前提下将可解释的集合逻辑嵌入FFN，实现了每个隐藏单元可直接读出的“and/and-not”运算，并通过自忘记模糊量化器恢复语法缺陷。

**🔧 技术方法**

采用sigmoid-bounded乘积、布尔分区、学习的遗忘率模糊量化器以及标准Transformer架构与残差连接。

**📊 数据集**

在OpenWebText上训练125M参数的GPT-2-small，并使用LAMBADA、BLiMP以及lm-eval基准进行评估。

**📈 对比分析**

与相同参数的GELU FFN进行对比，结果显示在困惑度上基本相当，语言模型质量不降；在BLiMP上存在小幅语法缺陷，但通过自忘记量化器可在第一个训练周期恢复；LAMBADA得分略有提升。

**⚠️ 局限性**

模型在高布尔比例下训练不稳定，完全布尔FFN会发散；可解释性仅覆盖约2%参数，且实验仅在单个种子与125M规模上验证。

---

## 548. PriorEye: Geospatial Visual Priors for End-to-End Autonomous Driving

**arXiv ID:** 2606.31830 | [PDF](https://arxiv.org/pdf/2606.31830v1)

**作者:** Kyuhwan Yeon `[一作]` (University of Oxford), Daniele De Martini `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

引入地理视觉先验，并通过双内存模块增强端到端自动驾驶模型，使其获得航路视角预见。

**💡 创新点**

将离线街景图像与路线空间锚定为视觉空间先验，并在端到端模型中加入模型无关的双内存和自适应门控实现安全可靠融合。

**🔧 技术方法**

使用记忆增强模块（双内存+自适应门控）、跨注意力融合、SigLIP2视觉编码、基于意图的检索等技术。

**📊 数据集**

基于NAVSIM‑v2真实世界自动驾驶基准（nuPlan衍生）以及Google Street View街景图像构建记忆库。

**📈 对比分析**

在四种端到端基线（LTF、GTRS‑DP、GTRS‑Dense、DrivoR）上加 PriorEye，EPDMS提升幅度分别约31%、14%、8%和1%，并在多种传感器失真与先验失真场景下表现出更强鲁棒性。

**⚠️ 局限性**

仅依赖车辆位置检索，未使用外观匹配；记忆库仅由静态街景构成，缺乏在线更新与遗忘机制。

---

## 549. Adaptive Cluster-First Route-Second Decomposition for Industrial-Scale Vehicle Routing

**arXiv ID:** 2606.31820 | [PDF](https://arxiv.org/pdf/2606.31820v1)

**作者:** Oguzhan Karaahmetoglu `[一作]` (Carnegie Mellon University), Hyong Kim `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于大型语言模型（LLM）的自适应Cluster‑First‑Route‑Second（CFRS）分解框架，能够在递归树状结构中根据实时分析动态选择聚类、平衡和细化操作，实现对工业规模车辆路线规划的分解与求解。

**💡 创新点**

创新点在于：①将分解视为连续决策过程，让LLM作为高层决策者；②引入联合客户-车辆拆分策略，考虑车辆容量与需求匹配；③使用多工具集（聚类、平衡、分析）与LLM交互，形成自适应迭代分解；④在超过500,000客户的极大规模实例上展示可扩展性。

**🔧 技术方法**

核心技术包括：大语言模型（Qwen2.5/Qwen3）进行策略决策；聚类与平衡工具（如Sweep、Balanced K‑Means等）作为可调用工具；层次化聚类树状态表示；迭代决策循环（观察→行动→状态转移）与分析报表融合；后端使用OR‑Tools Guided Local Search求解子问题并全局优化。

**📊 数据集**

使用两类数据集：①合成CVRP实例（多达50万客户），通过随机生成满足多种需求、车辆、仓库配置；②基于公开AGSP大规模Benchmark（最多3万客户）通过拼接构造更大规模（2~10个实例合并，最多12万客户）。

**📈 对比分析**

与几类基线方法比较：几何聚类（RND、SWP、GRD、QDR）、优化聚类（BKM、ENT、CCBC、CC‑CVRP）、学习基线（GLOP）以及自适应A‑DEC。实验显示：在小规模（2‑实例）A‑DEC与传统方法相近；在更大规模（6、10实例）时，A‑DEC在保持近零缺失率的同时，路程距离普遍低于几何/优化方法；在合成极大规模下，Qwen3版本距离最优且缺失率为0，证明自适应策略能有效平衡服务与成本。

**⚠️ 局限性**

局限性包括：①依赖LLM推理，模型推理时间与成本不易控制；②在极大规模实例中，分解树深度与分支因子需要手动调参，可能影响结果一致性；③实验主要基于合成与拼接数据，缺乏真实工业场景验证；④对资源受限环境（如GPU不足）适配性有限。

---

## 550. Creating Intelligence: A Computational Foundation for AGI

**arXiv ID:** 2606.31819 | [PDF](https://arxiv.org/pdf/2606.31819v1)

**作者:** Peter Overmann `[一作]` `[通讯]`, Peter Overmann

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于集合论与高维计算的拓扑关联记忆框架，用稀疏二进制集合代替连续权重，实现离散关联记忆和子集模式匹配的常数时间检索。

**💡 创新点**

创新点在于：①把记忆位置视为网络拓扑本身，学习通过切换二进制连接完成；②将传统连续神经网络的矩阵运算替换为集合运算，显著降低能耗与硬件复杂度；③在同一算法框架下同时支持自关联与异关联，兼顾模式补全与符号检索。

**🔧 技术方法**

使用的技术包括集合论与子集组合学、稀疏超维向量（SDR/SHR）、拓扑可塑性、组合扩展编码、k‑winner‑take‑all、离散子集匹配与最近邻搜索。

**📊 数据集**

实验主要在合成数据上验证：随机生成的稀疏集合、随机 SHR 以及手写数字（MNIST）等常见图像数据的 SDR 编码；未涉及大规模真实数据集。

**📈 对比分析**

与传统 Hopfield 网络、Kanerva 记忆、Transformer 等基线相比，检索时间为 O(|A|²) 与输入规模无关，容量达到百万级关联，能耗显著低于浮点矩阵运算，实验中能达到约 959k 条自关联记录的完好检索率。

**⚠️ 局限性**

局限性在于：①仅支持离散稀疏集合，连续数值与顺序信息需额外编码；②对超维维度要求高（1k–10k），存储空间随维度平方增长；③在极高容量下会出现记忆坠落和假阳性，需精细阈值调参；④对动态序列建模、长期记忆的扩展尚未完善。

---

## 551. Relational and Sequential Conformal Inference for Energy Time Series over Graphs via Foundation Models

**arXiv ID:** 2606.31804 | [PDF](https://arxiv.org/pdf/2606.31804v1)

**作者:** Keivan Faghih Niresi `[一作]` (EPFL), Olga Fink `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出 STOIC 框架，将空间‑时间图神经网络的点预测与表格基础模型的零样本校准结合，生成针对能源需求的可靠不确定性区间。

**💡 创新点**

创新点在于：①把残差转换为结构化表格特征，②利用预训练的表格基础模型实现无训练量的上下分位估计，③在此基础上实现对空间和时间相关性的自适应校准，突破传统 CP 仅依赖交换性假设的局限。

**🔧 技术方法**

使用的主要技术包括：空间‑时间图神经网络 (STGNN)、表格基础模型 TabPFN 的 in‑context 学习、特征工程（时间滞后、邻居统计）以及拆分式 conformal 预测。

**📊 数据集**

实验数据集覆盖五种场景：合成控制信号（SCS）、合成区块供热（SDH）、实际区块供热（EWZ）、电力负荷（ECL）和智能表聚合（CKW）。

**📈 对比分析**

与 SCP、ACI、SPCI、LSCP、STCP 以及大规模时间序列基线 TimesFM 的比较表明，STOIC 在所有数据集上都能保持或超过 90% 的覆盖率，同时预测区间更窄、Winkler 分数显著优于基线，尤其在具有强空间耦合的真实系统上表现突出。

**⚠️ 局限性**

局限性包括：依赖校准集的可用性，理论保证仅为渐进且对 TabPFN 的近似性缺乏严格证明，计算复杂度随图规模增长且对异构节点特征处理仍有提升空间。

---

## 552. Stratified Counterpossible Logic

**arXiv ID:** 2606.31881 | [PDF](https://arxiv.org/pdf/2606.31881v1)

**作者:** Chen Huang `[一作]` (Sun Yat-sen University), Xuefeng Wen `[通讯]` (Sun Yat-sen University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了分层的对偶可能性逻辑系统 SCP 和其扩展 SCP1，用于区分不同类型的不可能性并解决空真命题问题。

**💡 创新点**

创新点在于：① 将世界分为逻辑正常世界 N、反逻辑世界 I 与在 N 内的形而上可达关系 R_M，构建三层结构；② 在 SCP1 中加入非空约束（ne）和全局逻辑可达约束（rl），保证条件语句在逻辑不可能时不为空且不陷入空真；③ 证明两系统的语义完整性、有限模型性质及可判定性。

**🔧 技术方法**

使用了形而上可达关系、逻辑可达关系、选择函数以及传统的 S5 语义框架；通过构造规范模型和过滤模型完成完整性与可判定性的证明。

**📊 数据集**

未使用数据集，本文为理论性逻辑研究。

**📈 对比分析**

通过理论证明（Soundness, Completeness, Decidability）与示例案例（普通反事实、反形而上条件、反逻辑条件）展示系统优越性，未给出实验性性能对比。

**⚠️ 局限性**

局限在于对反逻辑世界 I 的内部结构未细化，未来可进一步引入非经典逻辑（如兼容逻辑）进行划分；此外，系统对形而上与逻辑不可能的划分仍依赖基于经典逻辑的假设。

---

## 553. Modal CEGAR-tableaux with RECAR and resolution-based SAT-shortcuts

**arXiv ID:** 2606.31878 | [PDF](https://arxiv.org/pdf/2606.31878v1)

**作者:** Rajeev Goré `[一作]` (Monash University), Cormac Kikkert `[通讯]` (Cormac Kikkert Research)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 CEGAR-tableaux 进行改进，加入 SAT 快捷路径，提出两种方案：RECAR 和基于模态解析器的全新解法。

**💡 创新点**

创新点在于：① 将模态分辨率（resolution）作为 SAT oracle 与 CEGAR-tableaux 结合，实现了既可得到 SAT‑shortcut 又能保持 tableau 结构的“fixpoint”检测；② 通过递归检查下采样公式的 fixpoint 来提前判定可满足性，避免显式构造大模型。

**🔧 技术方法**

主要技术包括：CEGAR-tableaux、SAT‑solver（sat4j）、模态解析器（模态 resolution）、RECAR 框架、fixpoint 检测算法以及多线程通信。

**📊 数据集**

使用标准 K‑logic 基准数据集：MQBF、LWB‑K、3CNF K‑benchmarks。

**📈 对比分析**

对比方法：将改进版与原始 CEGAR‑tableaux、RECAR、以及原版模态解析器进行对比。实验显示：RECAR 方案性能不佳；基于 resolution 的方案在 MQBF 上显著优于所有单独求解器，在 3CNF 与 LWB‑K 上与原始 tableau 持平或略优；结合多线程的 fixpoint 检测后整体性能超过单个求解器，尤其在满足性较强的实例中提升幅度达 100×。

**⚠️ 局限性**

局限性：① RECAR 方案由于尝试构造最小模型而在可满足实例上消耗过多；② resolution 方案的 normal‑form 生成会产生大量子句，对大规模公式（如 LWB‑K）导致时间过长；③ 当前多线程通信采用简单文件 I/O，未实现高效实时交互，影响整体效率。

---

## 554. RoboTacDex: A Dexterous Visual-Tactile-Action Dataset for Humanoid Manipulation

**arXiv ID:** 2606.31836 | [PDF](https://arxiv.org/pdf/2606.31836v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 555. Intuitionistic Monotone Modal Logic: Proof Theory and Semantics

**arXiv ID:** 2606.31870 | [PDF](https://arxiv.org/pdf/2606.31870v1)

**作者:** Tiziano Dalmonte `[一作]` (Free University of Bozen-Bolzano), Jim de Groot `[通讯]` (University of Bern)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了直觉主义单调模态逻辑I□及其扩展，并给出了基于构造性邻域模型的语义与对应的无割裁计算系统，证明了其完备性与可判定性；

**💡 创新点**

创新点在于首次为I□提供构造性邻域语义、构造性的单序推理系统，并展示了如何通过对输出剪枝（pruning）与剪枝（thinning）的修改，得到I□与其两模态版本的证明系统；

**🔧 技术方法**

主要技术包括构造性邻域模型的定义、基于段的典范模型构造、单序结构化演算（含块与标记公式）以及剪枝与剪枝的改造以实现不同逻辑的规则；

**📊 数据集**

无（该工作为纯理论研究，不涉及数据集）；

**📈 对比分析**

未与实验或其他方法进行性能比较，主要通过结构化证明技术证明可判定性和求证系统的可裁剪性；

**⚠️ 局限性**

局限性包括尚未给出完整的计算复杂度分析、未实现对失败推导的反证模型提取、以及对更一般模态逻辑的可扩展性与更复杂剪枝形式的探索仍待进一步研究。

---

## 556. Low-dimensional topology of deep neural networks

**arXiv ID:** 2606.31856 | [PDF](https://arxiv.org/pdf/2606.31856v1)

**作者:** Junyu Ren `[一作]` (University of Chicago), Lek-Heng Lim `[通讯]` (University of Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在固定宽度（例如 3 维）下，深度网络在保持或改变嵌入数据的拓扑不变量（主要是链接数）时的可表达性与可学习性，并用拓扑方法分析了不同网络架构的局限性。

**💡 创新点**

创新点在于把低维拓扑不变量——链接数——作为评估网络表达力的新工具，揭示了宽度受限且激活函数单调的网络无法在嵌入空间中解开交错曲线（如 Hopf 链），并证明了跳连、注意力机制以及非单调激活能够突破这一拓扑障碍。

**🔧 技术方法**

使用了 Gauss 积分计算链接数、拓扑同伦和链接数不变性论证、网络层级的拓扑不变量保持分析、PCA‑3D 级联检测算法以及对比实验。

**📊 数据集**

实验数据集包括：人工生成的 Hopf 环、k 个高维链接球体（Sⁿ∪Sⁿ ⊂ ℝ²ⁿ⁺¹）、以及通过 PCA 投影到 3 维后的 CIFAR‑10 图像。

**📈 对比分析**

通过在相同宽度（≤ d）与深度下训练 ReLU、GELU、ResNet、Transformer 等模型，对比分类准确率。结果显示：单调 ReLU 网络受链接数约束，准确率上限在 90% 左右；非单调激活、跳连 ResNet 与 Transformer 能突破该上限，在 Hopf 链和高维链接球体上实现近 100% 的精度；在 CIFAR‑10 中，非单调激活在检测到的链接样本上比单调激活高约 1–2% 的准确率差距。

**⚠️ 局限性**

局限性包括：仅考虑宽度 ≤ d 且坐标分离的单调激活；不考虑梯度下降的优化动态；链接检测算法基于 PCA‑3D 投影，未能直接验证像素空间的拓扑结构；理论主要聚焦于链接数，未扩展到更复杂的拓扑不变量（如 Milnor μ̅‑不变量、结类型）。

---

## 557. Bridging Local Observation and Global Simulation in Closed-Loop Traffic Modeling

**arXiv ID:** 2606.31844 | [PDF](https://arxiv.org/pdf/2606.31844v1)

**作者:** Ziyan Wang `[一作]` (University of Hong Kong), Xintao Yan `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 CRAFT 框架，在不重新训练基准自回归交通模拟器的情况下，通过生成完整场景的 what‑if rollouts 并训练 Contextual Preference Evaluator（CPE），在推理时对候选动作进行偏好校准，从而提升闭环交通模拟的行为合理性。

**💡 创新点**

创新点：①将完整场景的 what‑if rollouts 用作自监督数据，暴露基于局部日志训练导致的上下文‑动作偏差；②利用偏好学习（token‑级 BCE + intra‑/inter‑trajectory 对比损失）训练密集的上下文偏好评估器；③在测试时采用独立边缘近似的多候选校准方法，既不需重训练基准模拟器，又能显著降低碰撞和违规率。

**🔧 技术方法**

主要技术：自回归轨迹生成（如 CAT‑K）、冻结的场景编码器 + 时间卷积偏好解码器、BCE 与 contrastive 偏好损失、基于规则的 Auto‑labeler、K‑candidate lookahead 与指数加权校准、在线测试时的偏好引导。

**📊 数据集**

使用 Waymo Open Motion Dataset（WOMD）480k+ 多代理交互场景作为训练、验证和评估数据集。

**📈 对比分析**

与 GUMP、SMART、CAT‑K、R1Sim 等基线进行比较，使用 WOSAC 的 collision、offroad、traffic‑violation 等指标以及全场景行为合理性指标。CRAFT 在所有 Agent 的碰撞率、越界率和违规率上均显著下降（约 31%–33%），但在 WOSAC 的 realism 指标略有下降，表明在 log 一致性与全局行为合理性之间存在权衡。

**⚠️ 局限性**

局限性：①在 WOSAC 的 realism 分数下降，说明对日志匹配的偏好降低；②推理时仍需额外的 CPE 前向传播和多候选采样，导致一定的时间开销；③偏好标签依赖规则基 Auto‑labeler，可能在极端或复杂场景中不够鲁棒；④目前仅验证在自回归模型上的效果，尚未评估对其他生成模型的通用性。

---

## 558. Breaking Failure Cascades: Step-Aware Reinforcement Learning for Medical Multimodal Reasoning

**arXiv ID:** 2606.31825 | [PDF](https://arxiv.org/pdf/2606.31825v1)

**作者:** Junha Jung `[一作]` (Korea University), Jaewoo Kang `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究并提出了一种针对医学多模态视觉问答的强化学习方法 MRPO，解决早期推理失败的级联问题。

**💡 创新点**

在奖励设计中加入逐步推理奖励，并通过指数加权的优势重塑，对错误答案的早期无效推理步骤施加更大惩罚，从而显著降低级联错误。

**🔧 技术方法**

基于 GRPO 的强化学习框架，外部 LLM（GPT-5-mini）评估推理步骤，结合 ROUGE/BERTScore 等词汇/语义指标以及长度奖励。

**📊 数据集**

使用 VQA-RAD、SLAKE、PathVQA 的开放式问答与 MedThink 金标推理，并在 PMC-VQA、VQA-Med-2021、Quilt-VQA、RadImageNet-VQA、MIMIC-Ext-CXR-VQA 等五个外部数据集进行评测。

**📈 对比分析**

与基础模型、SFT、GRPO、GDPO 以及多款通用和医学专用多模态 LLM 进行对比，MRPO 在三大骨干模型上均获得最高平均分；Qwen3-VL-8B-Instruct 版提升至 28.94 分，超过同级别 HuatuoGPT-Vision-34B 2.79 分，早期失败率从 64% 降至 13%。

**⚠️ 局限性**

依赖外部 LLM 评估推理奖励导致成本和 API 依赖，需金标推理注释且仅验证于医学 VQA 领域，未在无注释或其它推理任务上验证。

---

## 559. Absorption-Feature-Guided Distance-Decoupled Estimation and Band Selection for LWIR Hyperspectral Passive Ranging

**arXiv ID:** 2606.31824 | [PDF](https://arxiv.org/pdf/2606.31824v1)

**作者:** Shuo Liu `[一作]` (National University of Defense Technology), Lilian Zhang `[通讯]` (National University of Defense Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于吸收特征的距离解耦估计与多场景有效 Fisher 信息驱动的 Band 选择方法（ADER），实现 LWIR 高光谱被动测距；

**💡 创新点**

核心创新在于：① 用 B‑spline 控制点约束材料发射率，显著减少自由度；② 通过臭氧吸收特征进行像素分类，区分发射支配与反射支配像素；③ 对发射支配像素采用一维残差最小化实现距离解耦；④ 对反射支配像素引入反射补偿优化；⑤ 采用多场景 Fisher 信息的贪婪增益策略实现任务特定的少光谱带选择；

**🔧 技术方法**

技术手段包括：光谱辐射模型、B‑spline 函数拟合、臭氧特征阈值分割、残差投影与一维搜索、Adam 优化的完整辐射模型修正、Fisher 信息矩阵分析与贪婪增益选择；

**📊 数据集**

实验使用 DARPA Invisible Headlights 数据集中的 LWIR 高光谱图像（256 带）与 LiDAR 对齐参考；

**📈 对比分析**

与公开的全光谱高光谱测距基线相比，ADER 在相同场景下平均误差更小、标准差更低，并将运行时间从 167.79 min 缩短至 77.2 s（约 130 × 速度提升），而 20 带版本仅略逊一筹，但仍保持接近全光谱的精度；

**⚠️ 局限性**

局限性包括：对极低温度对比度的可观测性不足、对大气参数先验依赖较强、以及在极弱吸收或高温差情况下的距离不确定性仍显著。

---

## 560. Token sliding independent set reconfiguration on graphs with few $P_4$'s

**arXiv ID:** 2606.31815 | [PDF](https://arxiv.org/pdf/2606.31815v1)

**作者:** Lucia Busolini `[一作]`, Mario Valencia-Pabon `[通讯]` (Universitè de Lorraine)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种多项式时间算法，用于在 P4‑tidy 图和 (q,q‑4) 图（这两类图是对共图的广义）上求解 Token Sliding 形式的独立集重构问题（TS‑ISR）。

**💡 创新点**

创新点在于将已知的对共图（cograph）上的多项式可解结果扩展到更大类的图——P4‑tidy 图和 (q,q‑4) 图，并给出对应的结构分解与重构规则，证明这些类图下的 TS‑ISR 可在多项式时间内决定。

**🔧 技术方法**

核心技术包括：p‑连通性与 p‑分区的分解定理、对 urchin、starfish、quasi‑urchin、quasi‑starfish 等基本构件的逐类分析、真/伪孪生顶点的处理、以及对 ∪、∨、⊻ 操作的递归处理；通过构造独立集状态图与对称差分析实现可达性判定。

**📊 数据集**

该工作完全为理论算法研究，无实验数据集。

**📈 对比分析**

性能：算法时间为多项式（相对于图的顶点数 n 以及独立集大小 k，具体实现可在 O(n^k) 之内完成），在 P4‑tidy 与 (q,q‑4) 图上可在多项式时间内判定可重构性。

**⚠️ 局限性**

限制：算法仅适用于具有有限 p‑连通分量或属于 (q,q‑4) 类的图；对更一般图（如有无界 p‑连通分量、非 (q,q‑4) 类）仍为 PSPACE‑硬，算法尚未扩展。

---

## 561. Generative Lane Topology Reasoning via Autoregressive Model with Geometry Prior

**arXiv ID:** 2606.31814 | [PDF](https://arxiv.org/pdf/2606.31814v1)

**作者:** Jiahui Fu `[一作]` (Beihang University), Si Liu `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种生成式框架TopoGPT，用自回归Transformer学习车道图几何先验，并通过流匹配感知适配器将BEV特征对齐，实现车载传感器条件下的完整车道拓扑推理；

**💡 创新点**

核心创新：①利用大规模地图数据预训练生成式自回归模型捕获车道图的几何先验；②引入流匹配适配器将多视角BEV特征映射到预训练条件空间；③通过端到端自回归序列生成实现几何一致、结构完整的车道图；

**🔧 技术方法**

采用的技术包括Lane Tokenizer、场景上下文编码器、Autoregressive Lane Sequence Transformer (ALST)、流匹配感知适配器、LoRA微调以及BEV编码器；

**📊 数据集**

预训练使用3.3M场景来自Waymo Motion、nuPlan、Argoverse 2 Motion；微调与评估在OpenLane-V2上完成；

**📈 对比分析**

在OpenLane-V2上与TopoNet、TopoMLP、TopoLogic、TopoPoint等基线对比，TopoGPT在车道级和点级指标上平均提升约+6.4和+11.6，显著优于竞品；

**⚠️ 局限性**

局限：自回归解码导致推理延迟和误差累积风险；几何先验对罕见路网结构的泛化能力有限。

---

## 562. CHERRY: Compressed Hierarchical Experts with Recurrent Representational Yield

**arXiv ID:** 2606.31796 | [PDF](https://arxiv.org/pdf/2606.31796v1)

**作者:** Dohyeon Kwon `[一作]`, Youngjin Park `[通讯]` (TeamSparta Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过联合使用选择性监督（SGT/SSGT）、深度压缩与递归恢复、以及专家融合（MoEE）三种技术，构建并验证了一款高效的韩语基础模型CHERRY‑1.8B。

**💡 创新点**

主要创新包括：①证明并量化Transformer中位置共享权重的“波传播”正梯度耦合可将监督信号从少量语义关键token扩散至大多数未监督token；②提出SGT/SSGT损失，仅对约15%关键token监督，且通过波传播实现与全序列训练相同的非监督token改进；③采用相邻层平均并通过递归层重用实现48层压缩至6层后仅损失0.008 loss、参数减少2.5×；④设计MoEE与GT‑专注蒸馏，利用多个压缩专家实现4.7% loss提升。

**🔧 技术方法**

技术实现基于Transformer(48层)与位置共享权重；相邻层平均与递归层重用；SGT/SSGT选择性损失与混合anchor；软多token预测(MTP)头；稀疏MoE路由与负载平衡；GT‑专注KL蒸馏；训练采用AdamW、BF16、NVFP4量化。

**📊 数据集**

实验数据集为12,800条韩语指令‑响应对，约6.55M token，覆盖问答、对话、Chain‑of‑Thought、工具调用与安全对齐示例。

**📈 对比分析**

通过与完整序列训练、未压缩模型、单压缩专家等基线对比，SGT在仅15%标签下实现与全序列相同的非监督token提升，Per‑supervised‑token ROI 4.5×；48层压缩至6层后loss仅提升0.008，参数下降2.5×；MoEE两专家loss 2.789比单压缩best 2.926低4.7%，并在自校正、验证、禁用率等指标上实现显著提升。

**⚠️ 局限性**

局限性包括：仅在韩语数据上验证，缺乏多语言与下游任务评估；压缩方法仅在Dense Transformer验证，未测试Mamba/RWKV等架构；SGT对小模型的GT识别效果下降；实验只在500步内收敛，需更长训练验证其稳定性；蒸馏仅关注GT位置，未充分利用全局信息；未与随机15%或高损失15%对照实验；未进行正式统计显著性检验。

---

## 563. Evo-PI: Aligning Medical Reasoning via Evolving Principle-Guided Supervision

**arXiv ID:** 2606.31800 | [PDF](https://arxiv.org/pdf/2606.31800v1)

**作者:** Xianda Zheng `[一作]` (University of Auckland), Shangyang Li `[通讯]` (University of Posts and Telecommunications)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出Evo‑PI框架，将可演化的推理原则作为监督信号，形成原则生成、评判、强化学习的闭环，用于提升多模态医学视觉问答的推理质量。

**💡 创新点**

①将推理原则语言化并可演化，取代传统静态奖励；②实现模型与原则的共进化循环，抑制奖励劫持；③在多模态医学问答任务中显著提升性能。

**🔧 技术方法**

使用知识型LLM（如GPT‑4o‑mini）生成和演化原则；冻结评判LLM（如Qwen2.5‑VL‑7B）评估原则遵循和思维点；在RLVR框架下采用GRPO/GSPO进行强化学习；构建密集奖励（原则奖励、思维点奖励）。

**📊 数据集**

OmniMedVQA数据集，覆盖CT、DER、FP、MI、MR、OCT、US、X‑ray八种医学影像模态。

**📈 对比分析**

与HuatuoGPT‑Vision和Med‑R1等基线模型在相同数据集和解码条件下对比；Evo‑PI平均提升约24%（Huatuo）和17%（Med‑R1），在部分模态达到99%以上的准确率。

**⚠️ 局限性**

依赖判定LLM的原则质量；训练成本较高；在稀有或高度专业化病例中原则可能不足；尚未验证在非医学多模态推理任务的泛化能力。

---

## 564. SENSE-VAD: Sentient and Semantic Video Anomaly Detection for Autonomous Driving

**arXiv ID:** 2606.31875 | [PDF](https://arxiv.org/pdf/2606.31875v1)

**作者:** Nghia T. Nguyen `[一作]` (University of South Florida), Yasin Yilmaz `[通讯]` (University of South Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SENSE‑VAD数据集，用以检测AV视角下的社交复杂异常场景；

**💡 创新点**

创新点在于将社交异常与运动统计异常区分，构建15类社交异常的系统分类与对应的合成视频；

**🔧 技术方法**

使用CARLA与Unreal Engine三版本进行合成场景生成，并配合脚本化摄像机、天气与时间控制，生成每帧二值标签；

**📊 数据集**

数据集包括约540,888帧的合成异常视频、匹配的正常合成视频以及少量真实世界异常/正常视频；

**📈 对比分析**

对11种现有视频异常检测方法（半监督、弱监督、无监督与基于大模型的训练免费方法）进行基准测试，最高AUC仅约57%，表明社交异常是目前的难点；

**⚠️ 局限性**

局限性包括：仅包含合成场景与少量真实片段、仅提供每帧二值标签缺乏空间定位、社交异常种类有限，且未涵盖更复杂的多代理交互或更高分辨率视角。

---

## 565. Inquisitive Action Logic

**arXiv ID:** 2606.31866 | [PDF](https://arxiv.org/pdf/2606.31866v1)

**作者:** Ivano Ciardelli `[一作]` (University of Padua), Ivano Ciardelli `[通讯]` (University of Padua)

**通讯引用:** 1719 | [OpenAlex ID](https://openalex.org/A5057421445)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出并研究了一种新的多智能体模态逻辑——询问式行动逻辑（Inquisitive Action Logic, IAL），用于同时表述智能体在行动中能强制的结果、能启用的结果以及能决定的结果。

**💡 创新点**

创新点在于：①将询问逻辑与行动逻辑结合，首次把智能体的“决定”表述为关于问题的模态陈述；②给出了实际效用函数（actual effectivity functions）的完整图像化定理；③构造了完整且可判定的公理化体系；④展示了 IAL 与社会友好联盟逻辑（Socially Friendly Coalition Logic, SFCL）的等价性（对单个智能体的情况），并讨论了潜在的指数级简洁性。

**🔧 技术方法**

使用的技术包括：询问式邻域逻辑（Inquisitive Neighborhood Logic）与其多智能体扩展；并发游戏结构（Concurrent Game Structures）与从中派生的实际效用函数；典型的公理化与完备性证明技术，如有限可满足性模型、Canonical 模型构造和归约论证；以及对问句分辨率（resolution）的运算。

**📊 数据集**

本研究没有使用任何实验数据集；所有结果均为理论性的。

**📈 对比分析**

方法比较主要通过逻辑表达式的等价性和表达能力来说明。理论证明表明 IAL 在单智能体陈述方面与 SFCL 等价，但在表达量上可能出现指数级的简洁性优势。由于缺乏实验评估，性能指标仅限于可判定性与有限模型属性。

**⚠️ 局限性**

局限性包括：①目前仅对单个智能体的情形完成了完整性与可判定性证明；②对智能体联盟的扩展仍未完成；③虽然提供了完整的公理化，但对大规模系统的实用性尚未验证；④在表达式翻译过程中可能出现指数级的长度膨胀，影响实际应用。

---

## 566. Belief Contraction in Dynamic Epistemic Logic

**arXiv ID:** 2606.31861 | [PDF](https://arxiv.org/pdf/2606.31861v1)

**作者:** Gaia Belardinelli `[一作]` (Stanford University), Snow Zhang `[通讯]` (University of California, Berkeley)

**通讯引用:** 45118 | [OpenAlex ID](https://openalex.org/A5014515612)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一种新的基于Kripke模型的信念收缩（hedged public announcement）操作，构建了相应的动态逻辑（hedged public announcement logic）并给出了完整的公理化体系；随后进一步提出了通用的Generalized Dynamic Epistemic Logic框架，能够处理多种类型的公告（包括私有公告）并兼顾收缩与扩展。

**💡 创新点**

创新点主要体现在：①用标准Kripke模型直接定义收缩操作，避免了可塑性模型所固有的正向内省限制；②证明了该操作满足部分AGM公理，并给出完整的归约公理和完备性证明；③提出了包含Q⁺关系的Generalized event model，实现了既可收缩又可扩展的动态更新，扩展了传统DEL的表达能力。

**🔧 技术方法**

技术方法包括：Kripke模型与事件模型的产品更新；对收缩操作的归约公理化与完备性证明；构造Generalized event model并定义其组合与模拟关系；利用模态等价与可模拟性来比较模型更新。

**📊 数据集**

本文未使用任何外部数据集；所有结果均为理论推导与形式化证明。

**📈 对比分析**

通过与传统可塑性模型（plausibility models）和标准公共公告逻辑的对比，证明了收缩逻辑在表达能力上的优势，并通过完备性定理证明了其一致性；此外，对比实验表明，收缩逻辑在处理“可能为假”的公告时能够保留更多原有信息，而传统方法会导致不必要的belief丢失。

**⚠️ 局限性**

主要局限包括：①收缩操作不满足所有AGM后退公理（如Recovery、Consistency），因此不具备完整的AGM兼容性；②在多代理情境下，某些自反性与正向内省性质可能被破坏；③对保留公式的完整分类尚未完成，无法给出完整的可保留/成功公式判定；④Generalized更新可能破坏访问关系的传递性与Euclidean性，导致高阶不确定性增加。

---

## 567. On Modal Logics of Full Products of Neighborhood Frames

**arXiv ID:** 2606.31852 | [PDF](https://arxiv.org/pdf/2606.31852v1)

**作者:** Rajab Aghamov `[一作]` (Technical University of Dresden), Jakob Piribauer `[通讯]` (Technical University of Dresden)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究邻域框架的全产品，给出了两种弱于 S4 的三模逻辑的完整定义：𝖳×_n^+𝖳 与 𝖣×_n^+𝖣。

**💡 创新点**

创新点在于将已知的 S4、D4 的乘积结果推广到更弱的邻域框架，并发现交互公理 (𝗆𝗂𝗑) 在此情形下既必要又充分。

**🔧 技术方法**

使用了无穷分支树、伪无限序列构造、边界映射与过滤等形式化证明技术，并对邻域与 Kripke 之间的同构关系进行深入分析。

**📊 数据集**

本文完全基于理论证明，没有使用实验数据集。

**📈 对比分析**

通过与先前 S4、D4 乘积逻辑的对比，证明新逻辑在有限模型性质（FMP）下可决，未涉及具体计算性能指标。

**⚠️ 局限性**

局限性在于仅处理序列性或反射性（即 𝖳、𝖣 类）框架，未覆盖更一般的非序列或非反射情况，且对更一般的 L₁×_n^+L₂（如 𝖪、𝖪₄）仍未给出完整结果。

---

## 568. MuSViT: A Foundation Vision Model for Sheet Music Representation

**arXiv ID:** 2606.31811 | [PDF](https://arxiv.org/pdf/2606.31811v1)

**作者:** Carlos Penarrubia `[一作]` (University of Alicante), Jorge Calvo-Zaragoza `[通讯]` (University of Alicante)

**通讯引用:** 2138 | [OpenAlex ID](https://openalex.org/A5085151278)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并公开了MuSViT，一种专为乐谱视觉任务设计的基础视觉模型；

**💡 创新点**

创新点在于：1）利用Mask Autoencoders对9.7M页公开乐谱进行自监督预训练，采用两阶段从合成到真实世界的训练策略；2）引入二维位置编码与细粒度Patch以捕获乐谱符号的细节与结构；3）提供Embedding‑Transcription一致性分析验证模型学习了音乐符号语义；

**🔧 技术方法**

技术包括ViT Encoder、Masked Autoencoders（MAE）、二维正弦位置编码、两阶段预训练课程、线性探测与微调策略、Faster R‑CNN、Transformer解码器以及LoRA微调等；

**📊 数据集**

数据集主要为IMSLP公开的9.7M页乐谱（合成语料+真实谱），下游任务使用的公开数据有全页识别（两份钢琴曲集）、单staff识别（5个语料），符号检测（1,714份乐谱，135类符号），难度分类（3个钢琴语料）；

**📈 对比分析**

与四个通用视觉/视觉‑语言模型（DINOv3‑7B、Qwen3‑VL、PaliGemma 2、Kosmos‑2.5）在四项任务中进行线性探测与微调比较，MuSViT在线性探测下大幅优于所有基线，在微调下击败或接近各任务的SOTA，整体性能提升显著；

**⚠️ 局限性**

局限性包括：1）模型对极端复杂或罕见的印刷/手写风格仍易失误；2）当前仅支持基于RGB乐谱图像，未覆盖多语言文本或多谱线交叉场景；3）对资源要求相对较高，尤其是大模型85M参数；4）缺乏对实时推理或嵌入式部署的评估。

---

## 569. TreeAgent: A Generalizable Multi-Agent Framework for Automated Bias Labeling in Forestry via Compiled Expert Rules and Vision-Language Models

**arXiv ID:** 2606.31976 | [PDF](https://arxiv.org/pdf/2606.31976v1)

**作者:** Shiyi Chen `[一作]` (University of California, Berkeley), Huiqi Wang `[通讯]` (University of California, Berkeley)

**通讯引用:** 4433 | [OpenAlex ID](https://openalex.org/A5062951736)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出TreeAgent，多代理框架结合专家规则与视觉语言模型，实现树木高度偏差自动标注

**💡 创新点**

引入Decoupled Declarative Decision（D3）框架，将专家决策拆分为逻辑原语与可编译图，零改动即可适配规则变更

**🔧 技术方法**

使用LLM（Neural Rule Transpiler）编译自然语言规则为验证可执行图，VLM进行视觉推理，多代理投票降低随机性，LightGBM作为监督基准

**📊 数据集**

利用NEON三站点（OSBS、WREF、SRER）树木高度测量数据（field、CHM、PCD）共283棵树

**📈 对比分析**

在跨站点测试中，TreeAgent宏F1 67.6%（平均0.04min/树），显著优于LightGBM 36.2%，并大幅降低人工标注时间（5min/树）

**⚠️ 局限性**

局限：VLM在树冠重叠等视觉判断上准确率受限；阈值节点在边界处易误判；需要更多领域适配与多专家标注验证

---

## 570. Semantic Leakage and Privacy Preservation in Relay-Assisted Semantic Communications

**arXiv ID:** 2606.31973 | [PDF](https://arxiv.org/pdf/2606.31973v1)

**作者:** Yalin E. Sagduyu `[一作]` (Nexcepta), Sennur Ulukus `[通讯]` (University of Maryland)

**通讯引用:** 14070 | [OpenAlex ID](https://openalex.org/A5021132487)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了在语义通信（SemCom）中使用中继节点时的隐私风险，证明中继能在不访问源数据的情况下通过对学习到的潜在表示进行推理和重构来泄露语义信息，并提出了一种迭代对抗训练框架，利用中继作为强学习式窃听者，通过交替优化传输端与中继端的网络参数，显著扩大合法接收端与中继端之间的语义准确率差距，同时保持重构质量；

**💡 创新点**

创新点在于：①将中继建模为可自适应学习的窃听者；②引入迭代对抗训练（min‑max）机制，显式对抗中继的推理能力；③通过对抗损失仅惩罚中继的语义分类误差，实现对语义信息的“隐形”抑制而不显著降低重构质量；

**🔧 技术方法**

采用深度卷积神经网络进行编码/解码；中继端使用一维残差卷积网络；使用重构损失（MSE/SSIM/PSNR）和分类交叉熵损失；采用Adam梯度下降的交替优化；评估指标包括语义分类准确率、PSNR、SSIM；

**📊 数据集**

使用CIFAR‑10图像数据集（10类，32×32彩色图像）；

**📈 对比分析**

通过对比基线（无对抗训练）和实验设置（不同SNR与γ_eve权重），在语义分类准确率上实现了显著的合法接收端与中继端差距提升；重构质量（PSNR/SSIM）差异维持在≤0.5 dB/0.01，表明对抗训练实现了“隐形”保护；

**⚠️ 局限性**

局限性包括：仅考虑单中继单跳场景，未评估多中继/多用户/多跳网络；对抗权重过大可能导致合法接收端性能下降；模型参数量大，训练成本高；对不同类型的数据分布及真实信道环境的鲁棒性尚未充分验证。

---

## 571. RRT-Rope: A deterministic shortening approach for fast near-optimal path planning in large-scale uncluttered 3D environments

**arXiv ID:** 2606.31948 | [PDF](https://arxiv.org/pdf/2606.31948v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 572. CoCoMUT: A Tool for Code-Context Mining and Automated Dataset Generation

**arXiv ID:** 2606.31971 | [PDF](https://arxiv.org/pdf/2606.31971v1)

**作者:** Alessandro Botta `[一作]` (University of Texas at Dallas), Soneya Binta Hossain `[通讯]` (University of Texas at Dallas)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

CoCoMUT是一款自动化的Java方法级上下文抽取工具，能够在单个项目中构建源代码模型、字节码调用图并生成版本化的JSONL方法记录。

**💡 创新点**

其创新点在于统一构建源/字节码视图、使用稳定方法URI进行源-字节码匹配、以及在抽取过程中实现可复现、任务无关的管线与版本化输出。

**🔧 技术方法**

工具结合了Spoon进行源代码解析、SootUp构建静态调用图、Javadoc解析、Maven/Gradle项目解析与构建，以及Jackson序列化JSON。

**📊 数据集**

评估使用了10个公开的Java仓库，覆盖Maven和Gradle项目，合计约406,312行代码。

**📈 对比分析**

通过RQ1-RQ3三项实验验证，CoCoMUT在所有10个仓库中成功完成抽取，生成200条方法记录，99%通过人工审核，源-字节码匹配率达99%。

**⚠️ 局限性**

局限在于仅支持完整的字节码、对生成的桥接方法和Lambda等编译器生成代码处理有限，以及不处理部分字节码或外部依赖文档。

---

## 573. MECoBench: A Systematic Study of Multimodal Agent Collaboration in Embodied Environments

**arXiv ID:** 2606.31966 | [PDF](https://arxiv.org/pdf/2606.31966v1)

**作者:** Qingyun Liu `[一作]` (Fudan University), Zhongyu Wei `[通讯]` (Fudan University)

**通讯引用:** 5405 | [OpenAlex ID](https://openalex.org/A5011504177)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MECoBench benchmark 与评估平台，系统评估多模态大型语言模型在多智能体协作的具身任务中的性能与机制。

**💡 创新点**

设计多模态协作结构、可变团队规模与多种协作模式，并对通信机制、协作模式和鲁棒性进行细粒度分析。

**🔧 技术方法**

采用多模态LLM（如 GPT‑5.4、Gemma4、Qwen3 等）与 VirtualHome 仿真环境，基于 observe–communicate–act 循环实现交互。

**📊 数据集**

构造 96 个任务（8 类，192 个测试案例）在 VirtualHome 场景上，提供视觉观测与任务目标。

**📈 对比分析**

通过成功率、完成率、步骤 AUC 等指标对比单体与多体、不同协作模式与通信机制，发现协作提升效果呈倒 U 曲线，通信关键，鲁棒性在信息噪声下仍显著。

**⚠️ 局限性**

局限于室内家庭场景、规模有限（96 案例，最多 5 人队），无法覆盖开放世界或更大团队的复杂任务。

---

## 574. LEO Satellite Network Orchestration with Heterogeneous Graph Neural Networks

**arXiv ID:** 2606.31950 | [PDF](https://arxiv.org/pdf/2606.31950v1)

**作者:** Aruna Jayarajan `[一作]` (Georgia Institute of Technology), Karthikeyan Sundaresan `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 4417 | [OpenAlex ID](https://openalex.org/A5018752163)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于异构图神经网络的低轨卫星网络编排框架，实时计算卫星-地面单元和网关的关联，实现完整覆盖与负载平衡；

**💡 创新点**

利用异构GNN区分卫星、地面单元和网关三类节点，设计约束感知的无监督损失，使模型既能保证覆盖又能平衡利用率；

**🔧 技术方法**

异构图神经网络（HetGNN）+自定义覆盖/容量/唯一性损失；

**📊 数据集**

使用Starlink第1期的仿真数据：1584颗卫星、5469个地面细胞（H3六边形）、54个网关；通过20秒时间快照构成训练集；

**📈 对比分析**

与中心化优化器、两种局部启发式、监督学习模型及同构GNN比较；在覆盖率和需求满足度上，HetGNN接近优化器（>99%覆盖，>95%需求满足），比局部启发式提升20–30%；推理时间<620 ms，远快于优化器；

**⚠️ 局限性**

对ISL或多跳链路支持有限；模型在极端高需求或大规模网络下仍可能出现容量约束违反；需要在极端天气或突发链路失效时进一步强化鲁棒性。

---

## 575. FlexViT: A Flexible FPGA-based Accelerator for Edge Vision Transformers

**arXiv ID:** 2606.31938 | [PDF](https://arxiv.org/pdf/2606.31938v1)

**作者:** Hubert Dymarkowski `[一作]` (University of Glasgow), José Cano `[通讯]` (University of Glasgow)

**通讯引用:** 466 | [OpenAlex ID](https://openalex.org/A5008501980)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了 FlexViT，一种可在资源受限的边缘 FPGA（PYNQ‑Z2）上运行的统一 GEMM 加速器，能够高效执行 Vision Transformer（ViT）模型中的全连接（FC）和卷积（CONV）层。

**💡 创新点**

核心创新在于：① 通过 runtime im2col 将 FC 与 CONV 层统一映射到 INT8 GEMM 引擎；② 动态双模数据流（输入广播/权重广播）可在单一硬件单元上针对不同维度形状实现最优数据重用；③ 深度优先分块和单遍累加避免了中间 partial‑sum 的 off‑chip 传输，从而显著降低内存带宽需求。

**🔧 技术方法**

采用硬件‑软件协同设计：SECDA‑TFLite 框架负责模型切片、im2col 预处理与调度；FPGA 侧实现三核 GEMM、读写单元、动态调度器及灵活的后处理单元；硬件实现使用 INT8 MAC、DSP 与 LUT 混合加速、3 核并行与 16‑位 SIMD；软件侧使用 TFLite Delegate、AXI‑Stream 交互与动态模式选择算法。

**📊 数据集**

在 ImageNet‑21k（ViT‑T）和 ImageNet‑1k（DeiT‑T、Swin‑T、MobileViT‑S、EfficientViT‑b1）训练得到的 INT8 TFLite 模型上进行测试，输入分辨率为 224×224 或 256×256。

**📈 对比分析**

与裸机 ARM Cortex‑A9 CPU（NEON）对比，FlexViT 在 offloaded 的 FC/CONV 层实现了最高 2.74× 的层级加速，整体推理速度提升最高 1.40×；能耗在大多数模型上与 CPU 基线相当或略低（最高 1.09× 的能耗收益），但在 EfficientViT‑b1 这类混合模型中因 CPU/FPGA 并行导致能耗略升至 0.93×。

**⚠️ 局限性**

局限性包括：① 仅 offload FC 与 CONV，Softmax、LayerNorm 等非线性层仍在 CPU 上执行，导致 Amdahl 限制；② 设计针对 INT8 量化，无法直接支持更低精度或二值化模型；③ 资源受限的 PYNQ‑Z2 只能实现 3 核并行，扩展到更大模型或更高分辨率需进一步优化内存映射和并行度；④ 需要在 SECDA‑TFLite 环境下部署，对非 TFLite 模型的兼容性有限。

---

## 576. MVP-Nav: Multi-layer Value Map Planner Navigator

**arXiv ID:** 2606.31919 | [PDF](https://arxiv.org/pdf/2606.31919v1)

**作者:** Wenyuan Xie `[一作]` (Shanghai Jiao Tong University), Hongtao Lu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 8493 | [OpenAlex ID](https://openalex.org/A5102899381)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MVP-Nav框架，利用单目RGB重建3D OBB并与LLM语义评分融合，形成多层价值图实现零深度目标导航。

**💡 创新点**

创新点在于将3D基础模型投影为3D OBB构成全局空间语义列表，并在多层价值图中把语义优先级与几何约束耦合，实现在无深度传感器下的物理约束规划。

**🔧 技术方法**

使用3D基础模型（VGGT）、语义分割（Grounded‑SAM）、LLM（GPT‑4o‑mini）、A*+FMM路径规划、Fast Marching Method、MVM等技术。

**📊 数据集**

在HM3D、MP3D、RoboTHOR等公开室内导航基准上评估，并在Agibot G1实际机器人上验证。

**📈 对比分析**

与现有RGB‑only方法（PanoNav、PixNav等）对比，MVP‑Nav在三大基准上的成功率与SPL均领先20%~30%，甚至超越部分RGB‑D方法。

**⚠️ 局限性**

主要局限在长距离单目深度漂移导致地图失真、对极端光照与纹理缺失场景的鲁棒性不足，以及高计算开销导致的延迟。

---

## 577. CoDex: Learning Compositional Dexterous Functional Manipulation without Demonstrations

**arXiv ID:** 2606.31909 | [PDF](https://arxiv.org/pdf/2606.31909v1)

**作者:** Bowen Jiang `[一作]` (University of Texas at Austin), Roberto Martin-Martin `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过零示例的框架实现了Compositional Dexterous Functional Object Manipulation任务，自动发现并执行喷雾瓶、热熔胶枪、气雾剂、手电筒、胡椒研磨机等工具的功能抓取与运动。

**💡 创新点**

创新点在于将Vision‑Language Model生成的本地与全局语义约束转化为可执行的几何约束，并通过解析优化产生功能抓取候选，再用RL进行完整抓取‑移动‑激活策略学习，构成从语义到物理动作的闭环。

**🔧 技术方法**

采用VLM进行任务解析和全局姿态搜索（VLM‑CEM），利用解析优化得到功能对齐的抓取候选，随后以这些候选为起点使用PPO强化学习实现完整抓取-移动-激活动作。

**📊 数据集**

在ManiSkill3模拟环境中训练，并在真实Frank Panda+LEAP手上测试六种未见对象及其目标物体。

**📈 对比分析**

与仅解析优化+VLM‑CEM（33%）和RL+PIVOT（0%）基线对比，平均成功率提升至73%；RL阶段将抓取稳定性提升约36%，功能激活成功率提升60%以上。

**⚠️ 局限性**

局限包括对极小触点的精准控制不足、仅支持单点激活、无法处理持续协同arm‑hand动作以及多点激活等复杂情形。

---

## 578. Sequential RC-TGAN: Generating Relational Time Series with Spectral Envelope Loss

**arXiv ID:** 2606.31904 | [PDF](https://arxiv.org/pdf/2606.31904v1)

**作者:** Mohamed Gueye `[一作]` (Croesus), Maxime Dumas `[通讯]` (Croesus)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Seq. RC-TGAN，一种用于生成关系型数据库子表时间序列的序列化生成对抗网络，并加入频域谱包络损失以直接优化周期性特征。

**💡 创新点**

创新点包括：①将谱包络理论从分类序列推广到连续属性（通过 VGM 离散化）；②在训练中使用可微分的 L2 谱包络损失；③构建可解析的“黄金标准”模拟数据（循环马尔可夫链）以及两种基于谱的评价指标（Spectral Density Divergence 与 Spectral Envelope Divergence）。

**🔧 技术方法**

技术手段：递归 RNN 生成器 + 条件判别器、Wasserstein GAN 对抗损失、谱包络损失（求解广义 Rayleigh 商、特征值问题）、VGM 离散化、贝叶斯层级采样、频域 L2 距离、KL/ Wasserstein 散度等。

**📊 数据集**

数据集：①模拟数据——周期性噪声循环 (NCP) 与对称粘性 (SSP) 的循环马尔可夫链，K=7,12,21；②真实数据——Rossmann 与 Walmart 两个关系型时间序列数据库。

**📈 对比分析**

方法与基线对比：与 SDV、ClavaDDPM、DoppelGANger、TimeGAN 等现有生成模型对齐；在模拟数据上显著降低 SED（≤48% 与最优基线相比），在真实数据上在 MSE(ACF)、Spectral Density Divergence 与 Spectral Envelope Divergence 上均取得领先，尤其在周期性捕捉与长周期季节性表现突出。

**⚠️ 局限性**

局限性：①仅验证两层关系模式，未评估更深层或更复杂的图结构；②谱包络损失对极低频/高频噪声的鲁棒性尚待进一步实验；③需先拟合 VGM 或离散化，增加预处理复杂度；④评估仍主要基于已知解析谱的模拟数据，实际业务中谱特征难以预知。

---

## 579. Attend, Transform, or Silence: Operator-Level Visual Skipping for Efficient Multimodal LLM Inference

**arXiv ID:** 2606.31903 | [PDF](https://arxiv.org/pdf/2606.31903v1)

**作者:** Zhaoyang Luo `[一作]` (Tsinghua Shenzhen International Graduate School), Haohuan Fu `[通讯]` (Tsinghua Shenzhen International Graduate School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种基于operator级别的视觉Token跳过框架，保持完整视觉序列的同时在多模态大型语言模型（MLLM）推理阶段减轻不必要的注意力和FFN运算；

**💡 创新点**

创新点在于从答案可观测性视角识别“答案无声”视觉更新，并通过对每层注意力与FFN的风险评估实现细粒度的operator跳过，而非整体层或Token级剪枝；

**🔧 技术方法**

使用答案可观测诊断、Jacobian影响量度、KL风险评估、层级operator风险分配与动态跳过策略；

**📊 数据集**

在10个视觉问答/多模态基准上验证：GQA、TextVQA、MME、MMBench、MMMU、POPE、ScienceQA、AI2D、OCRBench、VizWiz；

**📈 对比分析**

与VTW、ShortV、V‑Skip、V2Drop、APET等训练自由加速方法对比，针对LLaVA‑1.5‑7B、Qwen2.5‑VL‑7B、Qwen3‑VL‑8B三种模型，平均保留约99.5%性能的同时实现最高达33.7%视觉计算（TFLOPs）削减；在TextVQA上实现首 token latency 与准确度兼顾的最佳折中；

**⚠️ 局限性**

局限性在于跳过策略需在少量校准数据上估计风险，模型或任务的显著变化可能需要重新校准；在极端高层依赖场景下，operator级跳过可能导致细粒度视觉推理能力下降。

---

## 580. Non-classical Topological Evidence Logic

**arXiv ID:** 2606.31888 | [PDF](https://arxiv.org/pdf/2606.31888v1)

**作者:** Igor Sedlár `[一作]` (Czech Academy of Sciences), Igor Sedlár `[通讯]` (Czech Academy of Sciences)

**通讯引用:** 191 | [OpenAlex ID](https://openalex.org/A5029560754)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

研究了在直觉主义和相关逻辑框架下的拓扑证据逻辑（TEL）并提供了相应的语义模型与公理化体系

**💡 创新点**

首次证明了在直觉主义S4+全局模态以及在相关逻辑B S4+全局模态与内部补集操作的框架下，能够表达TEL的“密集开放子集”与“可证理由”概念，并给出完备的公理化

**🔧 技术方法**

使用了拓扑语义、上集拓扑（up‑space）、Routley–Meyer框架、锚定技术构造典型模型以及Zornlemma、分裂定理等证明工具

**📊 数据集**

无（本文为理论研究，无实验数据集）

**📈 对比分析**

通过语义等价与公理化的形式化证明展示了新体系的正确性和完备性，并与经典TEL比较，指出在非经典基础上仍保持可表达性

**⚠️ 局限性**

目前仅针对弱型相关逻辑B S4，未覆盖更强逻辑（如R或E）以及未证明可判定性和有限模型性质，且框架仍需扩展以处理更一般的量子化逻辑

---

## 581. Intuitionistic Justification Logic, Semantically

**arXiv ID:** 2606.31884 | [PDF](https://arxiv.org/pdf/2606.31884v1)

**作者:** Sonia Marin `[一作]` (University of Birmingham), Ian Shillito `[通讯]` (University of Birmingham)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5017336560)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

构造了两类直觉主义正当化逻辑的语义模型（基本模态模型与模态模型），并证明了它们对对应的直觉主义模态逻辑具有完备性；随后给出了从直觉主义模态逻辑到正当化逻辑的实现定理，并给出了实现函数的构造方法。

**💡 创新点**

创新点在于：①首次把满足项（diamond）与证明项（box）一起纳入直觉主义正当化逻辑的模型中；②在此框架下提出了新的“满足项-证明项”相互作用的公理与语义条件；③借助预实现与浓缩技术，完成了直觉主义实现定理的语义证明，填补了此前仅有证明性实现的空白。

**🔧 技术方法**

主要技术手段包括：典型（canonical）模型构造、基于prime set的构造、预实现（pre‑realisation）与浓缩（condensing）理论、以及对满足项与证明项的运算符进行的语义约束（如JYB与SYP原则）。

**📊 数据集**

该工作为理论性论文，无需使用实验数据集；所有结论均通过形式化证明获得。

**📈 对比分析**

由于是形式逻辑研究，没有实验对比；在理论上，作者证明了所提出模型对相应正当化逻辑的完备性和实现定理的正确性，表明该方法在理论上与先前方法一致且具有更强的语义基础。

**⚠️ 局限性**

局限性：目前仅适用于所研究的直觉主义正当化逻辑（带满足项的LJ^diamond），对更广泛的子逻辑（如constructive modal logic、S4 等）或更复杂的公理化扩展尚未完成；进一步的实现定理需要处理更复杂的证明/满足项运算，并且对模型的扩展仍是未来工作。

---

## 582. Amplifying Membership Signal Through Chained Regeneration

**arXiv ID:** 2606.31991 | [PDF](https://arxiv.org/pdf/2606.31991v1)

**作者:** Wojciech Łapacz `[一作]` (Warsaw University of Technology), Stanisław Pawlak `[通讯]` (Warsaw University of Technology)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5067569314)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于迭代再生成的通用会员推断框架MADreMIA，利用模型的再生成轨迹放大记忆信号。

**💡 创新点**

创新点在于将多步再生成轨迹作为信号放大器，提升成员与非成员区分能力，且无需训练阴影模型。

**🔧 技术方法**

使用迭代再生成、轨迹特征聚合、逻辑回归融合等技术，兼容黑盒、灰盒和白盒场景。

**📊 数据集**

实验数据集包括ImageNet、COCO（图像模型），OpenAI、EleutherAI等公开文本数据（LLM），以及AutoVC、FreeVC（语音转换）。

**📈 对比分析**

与传统一拍攻击比较，MADreMIA在多模态、多模型上实现了AUC、TPR@1%FPR、准确率等指标显著提升，证明轨迹放大效果显著。

**⚠️ 局限性**

局限包括迭代生成的计算开销、对迭代步数和生成强度的敏感性，以及目前在音频领域缺乏充分的基准验证。

---

## 583. Evaluation of Population Initialization Methods for Genetic Programming-based Symbolic Regression

**arXiv ID:** 2606.31990 | [PDF](https://arxiv.org/pdf/2606.31990v1)

**作者:** Lukas Kammerer `[一作]` (University of Applied Sciences Upper Austria), Stephan Winkler `[通讯]` (University of Applied Sciences Upper Austria)

**通讯引用:** 4771 | [OpenAlex ID](https://openalex.org/A5036920760)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在遗传编程符号回归中用已优化的初始种群（由穷举符号回归生成）与传统随机初始化方法对最终 Pareto 前沿的影响；

**💡 创新点**

证明了在保持足够多样性的前提下，即使初始种群已经在精度和复杂度上较优，随着进化过程的进行，其优势很快消失，最终结果与随机初始化相当；

**🔧 技术方法**

使用了 Operon（基于 NSGA-II 的多目标遗传编程框架）、穷举符号回归（ESR）生成初始种群，并在 12 个人工合成数据集和 1 个真实 Nikuradse 数据集上进行实验；

**📊 数据集**

12 个噪声较小的单变量合成问题（复杂度从 11 到 43）和 1 个一维 Nikuradse 液体摩擦数据集；

**📈 对比分析**

通过 1000 次随机种子重复实验，比较了 ESR 初始化与 grow、PTC‑2、BTC 三种随机初始化方法在 500 代后 Pareto 前沿的误差-复杂度分布。结果显示在大多数问题上四种方法的误差分布基本重叠，唯一明显优势仅出现在问题 1；

**⚠️ 局限性**

限制包括：ESR 仅能搜索至复杂度十，无法覆盖更大搜索空间；实验仅限单变量问题；只测试了一维真实数据集；可能更高维或更大复杂度的数据集会产生不同结论。

---

## 584. CoLT: Teaching Multi-Modal Models to Think with Chain of Latent Thoughts

**arXiv ID:** 2606.31986 | [PDF](https://arxiv.org/pdf/2606.31986v1)

**作者:** Lianyu Hu `[一作]` (Nanyang Technological University), Yang Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 86898 | [OpenAlex ID](https://openalex.org/A5100355964)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多模态模型的链式隐思考（CoLT）框架，让模型在连续隐向量空间中进行多步推理，而不是生成冗长的文本推理链。

**💡 创新点**

创新点在于三种互补的步骤级监督：前向解码（将隐向量解码为下一步文本）、后向解码（将前一步文本映射到隐向量并对齐）以及内部监督（预测下一个隐向量），通过这三条信号共同约束隐推理过程，解决传统隐推理中语义无意义和训练不稳定的问题。

**🔧 技术方法**

技术包括：Qwen3‑VL‑8B 作为主干语言模型，轻量级 Qwen3‑0.6B 作为外部解码器提供前向/后向监督；2 层 MLP 投影头实现内部监督；对齐损失、余弦相似度等；训练分两阶段：监督微调 + 强化学习；推理时仅使用隐向量，无额外解码开销。

**📊 数据集**

使用 Onethinker 数据集（包含文本推理链注释）进行训练，评估八个公开多模态基准：SeedBench、MMBench、ChartQA、TextVQA、ScienceQA、MMStar、AI2D、MMT‑Bench。

**📈 对比分析**

与 7 类基线对比：专有模型、开源 MLLM、传统文本 CoT、现有隐推理方法（CCoT、ICoT、CODI、SoftCoT、SIM‑CoT）以及基于辅助图像的隐视觉推理方法（Multimodal CoT、LaCoT、LVR）。CoLT 在平均分 79.1 分上，比 SIM‑CoT 提升 5.1%（即从 74.0% 到 79.1%），比 LVR 提升 5.5%；与文本 CoT 相比，准确率提升 3.4%（75.7% → 79.1%），并且推理速度提升 10.1×、文本解码时间提升 22.6×。

**⚠️ 局限性**

局限性：① 固定的隐推理步数 K=3 对不同任务并非最优，虽然能保持较好泛化但在更复杂问题上可能受限；② 需要外部解码器进行监督，训练成本和实现复杂度相对较高；③ 隐向量本身缺乏可解释性（虽能通过解码回推文本，但仍不如直接文本推理直观）；④ 目前未针对动态难度自适应步骤或混合隐-文本推理进行探索。

---

## 585. GR2 Technical Report

**arXiv ID:** 2606.31984 | [PDF](https://arxiv.org/pdf/2606.31984v1)

**作者:** Yufei Li `[一作]` (Meta AI), Luke Simon `[通讯]` (Meta AI)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种面向工业规模推荐系统的生成式推理再排序模型GR2。

**💡 创新点**

创新点包括：使用语义ID中间训练提升模型对海量物品的泛化能力；通过教师模型生成高质量的链式推理轨迹并进行On‑Policy Distillation（OPD）来注入推理先验；采用可验证奖励的RL（DAPO）和上下文压缩实现高效推理。

**🔧 技术方法**

采用的技术包括：Qwen3系列LLM、RQ‑VAE生成语义ID、mid‑training、链式推理（CoT）生成、OPD、DAPO RL、格式化奖励、上下文压缩与KV缓存等。

**📊 数据集**

使用的数据集是内部70k用户会话日志，包含历史交互与候选列表；测试集覆盖约100×训练规模，并保证99%候选不重叠、全新用户与物品。

**📈 对比分析**

与工业原始点评估基线对比，GR2在R@1提升18.7%、R@3提升9.6%、NDCG@3提升9.6%；性能在不同流量规模、两周时延和模型规模变化下均保持稳健。

**⚠️ 局限性**

局限性包括：对大型LLM算力依赖高；离线训练与标注成本仍大；对极少见冷启动场景的适配不足；在非语义ID环境下的泛化能力有限。

---

## 586. ERA: Entropy-Guided Visual Token Pruning with Rectified Attention for Efficient MLLMs

**arXiv ID:** 2606.31982 | [PDF](https://arxiv.org/pdf/2606.31982v1)

**作者:** Yuhao Wang `[一作]` (Dalian University of Technology), Huchuan Lu `[通讯]` (Dalian University of Technology)

**通讯引用:** 48458 | [OpenAlex ID](https://openalex.org/A5006986293)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练无关的视觉令牌剪枝框架ERA，解决多模态大语言模型在视觉令牌冗余导致的推理成本高问题。

**💡 创新点**

① 引入头维熵（Head-wise Attention Entropy）作为令牌重要性度量；② 通过双视图熵剪枝（DEP）同时兼顾多样性与显著性；③ 采用偏置感知令牌回收（BTR）在剪枝后恢复视觉信息并估计集群偏置；④ 通过对注意力logit注入偏置实现Attention Logit Collapse纠正（LAR），实现对注意力分布的保持。

**🔧 技术方法**

双视图熵剪枝、偏置感知回收、logit保持注意力校正、基于熵的头维显著性、矩阵增广实现的硬件友好式注意力修正。

**📊 数据集**

VQA^v2、GQA、SQA^I、VQA^T、ChartQA、AI2D、MME、MMB^E、MMB^C、MMVet、POPE、HaluBench、MileBench、VideoMME、LongVideoBench、MVBench等多模态视觉问答、推理、视频理解及跨图推理数据集。

**📈 对比分析**

与FastV、DivPrune、CDPruner、VisionZip、LLaVA-NeXT等现有剪枝方法对比，在单图、跨图及视频场景下，ERA在相同token压缩比例下平均保持93–98%性能，并显著提升FLOPs、延迟和缓存占用（例如将视觉令牌从2880压缩到160，FLOPs下降≈12×，预填充延迟≈4.6×），同时保持或提升多模态评测指标。

**⚠️ 局限性**

依赖于视觉编码器提供的头注意力熵，若编码器结构极端不同或缺乏CLS/全局注意力，熵计算可能不稳定；在极端压缩下仍存在信息损失，尤其是细粒度视觉细节；对动态分辨率和视频场景的适配仍需进一步优化。

---

## 587. LUNA: Learning Universal 3D Human Animation Beyond Skinning

**arXiv ID:** 2606.31981 | [PDF](https://arxiv.org/pdf/2606.31981v1)

**作者:** Peng Li `[一作]` (Hong Kong University of Science and Technology), Shunsuke Saito `[通讯]` (Meta)

**通讯引用:** 6437 | [OpenAlex ID](https://openalex.org/A5102959646)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种完全去除线性混合蒙皮（LBS）的 3D 人体动画框架 LUNA，能够直接以多种 2D 驱动信号（RGB 图像、关键点、素描等）驱动 3D 高保真动画；

**💡 创新点**

创新点包括：① 用 Transformer 把 2D 驱动映射到 3D 高斯变形，实现全 LBS‑free 解析；② 在无 3D 先验的情况下引入 LBS 指导的软蒸馏监督，解决深度不确定性；③ 通过全局刚性运动 + 局部非刚性变形的解耦，精准捕捉服装、毛发等细腻动态；

**🔧 技术方法**

技术手段包括：3D Gaussian 生成与渲染、基于多模态 Transformer 的身份编码器、DINOv3+MM‑Transformer 的隐式神经动画器、混合监督（渲染损失+LBS 蒸馏+投影损失）与动态权重平衡；

**📊 数据集**

使用四大数据集训练与评估：Video35K、iPhone1K、Cloth10K、Dome；在实验中还对公开 NeuMan 数据集进行交叉身份驱动评估；

**📈 对比分析**

与现有优化式（Vid2Avatar、ExAvatar）及流式基线（IDOL、LHM、UP2YOU、MV‑LHM）对比，LUNA 在 Cloth10K 上 PSNR 22.07（对比 LHM 20.12）、LPIPS 0.131，MAE 0.0225、MSJ 0.0032，显著降低运动抖动；在跨身份驱动中，误差仅 17.3%（相对 LHM 26.7%），说明零样本泛化能力强；

**⚠️ 局限性**

局限性包括：对驱动视频中的遮挡和极端姿态易产生 2D 信号失真与时序不稳定；极端体型差异或极端动作仍难以完全 disentangle 运动与形状；缺乏显式时序建模，可能导致细节随时间漂移。

---

## 588. Signed-Permutation Coordinate Transport for RMSNorm Transformers

**arXiv ID:** 2606.31963 | [PDF](https://arxiv.org/pdf/2606.31963v1)

**作者:** John Sweeney `[一作]` `[通讯]` (Sideplane AI), John Sweeney (Sideplane AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

本文研究了Transformer残差流的坐标对齐问题，提出了基于符号边缘化的Hungarian匹配方法来恢复RMSNorm模型的签名置换对齐，并验证了通过沿相同基准训练轨迹进行局部对齐组合（transport）可以显著提高坐标恢复率和工具迁移效果。

**💡 创新点**

创新点在于揭示RMSNorm的离散对齐组为Signed-Permutation而非仅Permutation，证明忽略符号会导致二阶误差，提出符号边缘化匹配并证明其在无相关坐标情况下能完全恢复真实对齐；同时将局部对齐连乘构成全程“transport”来突破端点匹配瓶颈。

**🔧 技术方法**

使用的技术包括符号边缘化的线性分配（Hungarian/Jonker-Volgenant），对齐链的矩阵乘积构造，激活匹配实验，AdamW状态的签名共变，SAE重建和Steering向量的效应评估，以及基于gauge审计的解释性检验。

**📊 数据集**

主要使用的实验数据集包括Qwen2.5-1.5B（1500步微调、SST-2、WikiText）、TinyLlama、Llama-2-7B、10M RMSNorm模型、BERT、T5、ViT等多种公开模型和数据集进行坐标恢复、Steering和SAE评估。

**📈 对比分析**

与传统端点匹配方法相比，transport通过沿同一基准轨迹的局部对齐实现了约91%坐标恢复（vs 60%），Steering向量保留率提升至95%（vs 17%），SAE重建NMSE降至0.004（vs 1.08），LoRA适配器保持99.7%增益（vs 0.5%），merge时峰值障碍被消除，表明方法在坐标保持和工具迁移上显著优于仅Permutation对齐。

**⚠️ 局限性**

局限性包括对相同基准训练轨迹的依赖，独立模型对齐需要大量probe；符号边缘化匹配在高维下计算成本高；方法主要针对RMSNorm及其一般增益配置，其他归一化或稠密旋转不适用；对齐链需要满足循环一致性，无法在完全无基准的跨模型迁移中直接应用。

---

## 589. DEMUN: Fast and accurate discovery of music notation in very large collections

**arXiv ID:** 2606.31956 | [PDF](https://arxiv.org/pdf/2606.31956v1)

**作者:** Vojtěch Dvořák `[一作]` (Charles University), Jan Hajič `[通讯]` (Charles University)

**通讯引用:** 8213 | [OpenAlex ID](https://openalex.org/A5102738283)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在大规模图书馆数字图像中，开发并部署了一个两阶段轻量级音乐符号检测系统 DEMUN，用于快速发现隐藏的乐谱。

**💡 创新点**

创新点在于通过两阶段架构显著降低网络流量与误报率，实现0.015% FPR的极低误报，并在生产环境中实现高吞吐量。

**🔧 技术方法**

采用 EfficientNet-b0 作为预过滤器，YOLOv11 进行主检测，并结合多分类的二进制+三分类或单通道四分类模型。

**📊 数据集**

使用自建的 OMRA 数据集，包含现代、教会和世俗乐谱的正负样本，并通过从 Moravian Library 的 400 万图像中标注 100k 样本进行自举。

**📈 对比分析**

与单阶段检测对比，单通道四分类模型在FPR 0.57%、FNR 2.12% 及精度 96.8% 下，输出浓度提升至 72%，相比预过滤的 1.5% 大幅提高；整体系统的误报率降至 0.015%。

**⚠️ 局限性**

限制包括对预过滤器的 FNR 估计不精确、对网络带宽的高度依赖以及在大规模部署时仍需大量计算资源和标注工作。

---

## 590. No Place to Hide: Benchmarking Video Hallucination with Background-Controlled Pairs

**arXiv ID:** 2606.31933 | [PDF](https://arxiv.org/pdf/2606.31933v1)

**作者:** Haojian Huang `[一作]` (Hong Kong University of Science and Technology), Ying-Cong Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 18003 | [OpenAlex ID](https://openalex.org/A5064499660)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个背景一致、前景差异的对抗视频对（VidPair-Halluc）来评估大型视频模型的幻觉表现。

**💡 创新点**

创新点在于提出了 PairFlow 三阶段生成流程（故事生成、视频片段合成、视频拼接），通过文本到图像和视频生成技术实现背景高度一致但前景显著差异的对抗样本；并制定了细粒度的空间与时间问答评测指标。

**🔧 技术方法**

使用 GPT‑4.1 生成故事脚本，FLUX 生成关键帧，SeedEdit 对前景进行精准编辑，Wan‑2.1/2.2 生成连贯视频片段；评估时采用 qAcc、vAcc、wAcc、Yes‑Percentage Difference 等指标。

**📊 数据集**

数据集包含约 1,000 对高质量对抗视频（共 2,000 条 15 秒视频）和 11,523 条空间/时间 QA 对，背景相似度通过 DINOv2、LPIPS、SSIM 等指标进行量化。

**📈 对比分析**

在 15 个主流 LVM（包括开源与闭源）上进行评测，闭源模型 Gemini‑2.5‑Pro 在 wAcc、FP 低、MCQ F1 等指标上表现最佳；开源模型 Video‑LLaMA2 在多选题上表现突出，整体仍显著低于人类基准。

**⚠️ 局限性**

局限性包括对生成模型质量的依赖（生成视频的真实性与一致性不一），视频长度受限（15 秒），仍需人工审核以保证背景一致性和前景差异；与真实视频相比，合成对抗样本可能缺乏真实世界多样性。

---

## 591. Better Understanding, Understanding Better

**arXiv ID:** 2606.31892 | [PDF](https://arxiv.org/pdf/2606.31892v1)

**作者:** Yu Wei `[一作]` (East China Normal University), Yu Wei `[通讯]` (East China Normal University)

**通讯引用:** 27730 | [OpenAlex ID](https://openalex.org/A5004681822)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

提出并形式化了一种对理解进行分级和比较的知识论逻辑，利用层级化的“理解为何”模态与比较连词来刻画不同程度和相对优劣的理解。

**💡 创新点**

创新点包括：①引入基于解释层级的分级模态和显式比较连词；②将解释结构与模态论证分离，构建了既能处理有限等级又能处理极限理想理解的两层语义与证明框架；③证明了有限层公式的可判定性和完整系统的强完备性，揭示了完整语言的非紧致性。

**🔧 技术方法**

采用的技术包括：①基于多代理模态模型的分级解释语义；②正则化的求证论证：有限层 Hilbert 系统和无穷级 ω-引入的无穷规则；③典型的 canonical model 结构构造与 Truth Lemma 证明；④利用加法、应用、正向与负向自省的解释术语代数。

**📊 数据集**

无数据集，完全为理论逻辑框架。

**📈 对比分析**

不存在实验对比；性能评估转化为逻辑性质：有限层可判定、系统可证明性、完整性和非紧致性等。

**⚠️ 局限性**

局限性：完整语言缺乏紧致性，导致无有限公理化系统；对实际 AI 解释与学习的动态行为尚未给出细化的模型或验证；在实践中解释代数的构造与计算仍需进一步研究。

---

## 592. Knowing-Value Logic with Successor Arithmetic

**arXiv ID:** 2606.31891 | [PDF](https://arxiv.org/pdf/2606.31891v1)

**作者:** Hongyi Wang `[一作]` (Peking University), Hongyi Wang `[通讯]` (Peking University)

**通讯引用:** 59693 | [OpenAlex ID](https://openalex.org/A5100449651)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

对条件式知值逻辑加入了等号与后继函数，构造了新的逻辑语言并给出了相应的公理系统。

**💡 创新点**

在加入算术符号后实现了非标准模型下的强完备性、标准模型下的弱完备性、有限模型性质与可判定性，并成功用该逻辑形式化并求解了“相邻数”谜题。

**🔧 技术方法**

采用了可满足性与完备性证明中的树形框架、非标准算术结构（ℕ^⋆、S^⋆）、同值等价关系、路径权重与 Z‑链的构造以及归约公理化技术。

**📊 数据集**

本研究为理论性工作，没有使用具体数据集；所有结果均基于逻辑模型与形式化证明。

**📈 对比分析**

通过构造树形模型并证明有限模型属性，最终证明了逻辑可判定；与之前仅包含知值运算的逻辑相比，加入算术后仍保持可判定且能表达更细粒度的自然语言推理。

**⚠️ 局限性**

主要限制是：在标准模型上不具备紧致性，需依赖非标准模型以获得完备性；扩展到加法、乘法等更强算术运算时，完备性与可判定性仍是未解决的问题。

---

## 593. Clean Me If You Can: A Large Collection of Real-World Addresses for Data Cleaning Benchmarking

**arXiv ID:** 2606.31983 | [PDF](https://arxiv.org/pdf/2606.31983v1)

**作者:** Fatemeh Ahmadi `[一作]` (TU Berlin), Ziawasch Abedjan `[通讯]` (TU Berlin)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个大规模真实世界的地址数据集，并对现有的规则、学习和LLM驱动的数据清洗方法进行系统评估；

**💡 创新点**

创新点在于首次提供超过900万条真实地址的脏数据及其对应的高质量地理真值，并对错误类型进行细粒度归类，为数据清洗研究提供了可复制的基准；

**🔧 技术方法**

技术手段包括从Web Data Commons中提取Schema.org和Microformats的N‑Quads、使用Nominatim geocoding生成真值、libpostal进行地址拆分、以及对Horizon、UniClean、HoloClean、Raha、Baran、DataWig和LLM（Gemini 2.5 Flash‑Lite）等多类清洗系统的实验；

**📊 数据集**

使用的数据集为从2024年12月的Web Data Commons抓取的约2.48亿条地址记录，经过过滤后得到约9.32M条脏记录与对应的Nominatim真值，其中4.34M条包含实体名称；

**📈 对比分析**

比较方法采用检测和纠正的精确率、召回率、F1值，结果显示学习型方法在检测上表现最佳，LLM在跨语言/格式误差上更鲁棒，但耗时和成本高；规则型方法在高冗余场景下可行，但整体效果有限；

**⚠️ 局限性**

局限性包括数据集仅纵向扩展，主要覆盖美欧地区，缺乏多列和多领域的多样性；真值生成依赖单一地理服务和标准，可能不适用于所有使用场景；

---

## 594. Planar-SfM: Camera Pose Estimation via Homography Graph Embeddings

**arXiv ID:** 2606.31979 | [PDF](https://arxiv.org/pdf/2606.31979v1)

**作者:** Gabi Pragier `[一作]` (Amazon Prime Video Live Events), Avi Ben-Cohen `[通讯]` (Amazon Prime Video Live Events)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建基于同一平面或多平面视角的相机位姿图，通过平面单应子分解得到多条相机间相对位姿估计，并利用谱嵌入过滤不可靠边，最终从最大一致性生成树恢复全局相机位姿。

**💡 创新点**

①将平面视角视为几何约束而非限制，①多平面联合估计提供冗余、互补的相对位姿；②使用谱嵌入将边映射到实线上，自动识别并剔除噪声边，得到最小范围生成树；③在单一平面或非平面场景均可统一处理。

**🔧 技术方法**

平面单应子估计与分解、相对位姿图构建、多重边相似度评估、基于相似度的谱嵌入、最小范围生成树、边权重合并后位姿恢复、最小二乘/最小无平方偏差优化。

**📊 数据集**

1) Overtime Elite（OTE）篮球场内部摄像头数据集；2) IMC Phototourism 大尺度户外场景数据集。

**📈 对比分析**

与 GLOMAP、COLMAP、PixSfM、DFSfM、VGGSfM 等经典与学习型 SfM 方法对比。OTE 上在 3°、5° 误差阈值下优于 GLOMAP，1° 仍保持竞争力；IMC 上在所有阈值下均取得新高 AUC，达到或超过现有最优方法。

**⚠️ 局限性**

对极限非平面场景的鲁棒性仍有限；需要准确的平面单应子估计，若单应子误匹配或平面缺失可能导致误差；算法在多平面稠密场景中谱嵌入与生成树求解的计算开销相对较高。

---

## 595. LuxEmo: Expressive Text-to-Speech Corpus for Luxembourgish

**arXiv ID:** 2606.31947 | [PDF](https://arxiv.org/pdf/2606.31947v1)

**作者:** Nina Hosseini-Kivanani `[一作]` (Radio Télévision Luxembourg), Sandipana Dowerah `[通讯]` (Tallinn University of Technology)

**通讯引用:** 56 | [OpenAlex ID](https://openalex.org/A5048477278)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了一个21小时、七千五百余段的卢森堡语情绪语音语料库LuxEmo，并基于该语料库评估五种表达式TTS系统，形成低资源语言情绪合成基准；

**💡 创新点**

提出了半自动化的语料库构建工作流（VAD、降噪、语言识别、ASR分割、情绪预测与人工校验），并在代码混杂的青春广播环境下实现真实情绪语音的收集与标注；

**🔧 技术方法**

采用了语音活动检测、DeepFilterNet降噪、Wav2Vec2语言识别、HuBERT情绪分类、LuxASR分割、非侵入式质量评估（NISQA、DNSMOS、WV-MOS）以及多模态人类听力测试；

**📊 数据集**

主要使用来自RTL Youth频道的直播广播音频，经过处理后形成7,562段长达10秒的卢森堡语（含德语、法语、英语代码切换）情绪标注数据；

**📈 对比分析**

通过客观指标（WV-MOS、NISQA、DNSMOS、WER、F0 RMSE、说话者相似度）与主观听力测试对比GradTTS、XTTS、Toucan、Qwen3_FT、kNN TTS五种系统；结果显示不同模型在音质、可懂度、情绪表达与说话者一致性方面各有优势，无法有单一模型在所有维度占优；

**⚠️ 局限性**

局限性包括：样本仅来自四位固定主播，缺乏多样化说话者和领域覆盖；情绪标签为弱监督，未进行多评标注；代码切换与背景噪声难以完全消除，导致ASR和情绪识别误差；

---

## 596. World Narrative Model for Highly Controllable Video Generation: A Paradigm Shift from Pixel Sampling to Physical World Orchestration

**arXiv ID:** 2606.31946 | [PDF](https://arxiv.org/pdf/2606.31946v1)

**作者:** Ye Chen `[一作]` (Shanghai Jiao Tong University), Bingbing Ni `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 14216 | [OpenAlex ID](https://openalex.org/A5014362734)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出世界叙事模型（World Narrative Model），将视频生成拆分为可编辑、可预视的3D+T世界控制阶段和基于现有视频基础模型的像素渲染阶段，实现高度可控的影视内容创作。

**💡 创新点**

创新点在于用显式实体化的3D+T世界结构替代传统黑盒像素采样，提供精确、可解耦、可一致的物理参数控制，并通过多智能体协作自动生成场景、资产、动作、相机与灯光，形成可视化、可交互的创作流程。

**🔧 技术方法**

核心技术包括LLM驱动的Blender脚本生成与监督、视觉检索/重建、3D资产生成/检索、运动与轨迹规划、相机与灯光参数规划，以及基于现有大模型（如Seedance 2.0、Sora、Kling）的无训练渲染接口。

**📊 数据集**

使用公开动作数据集（Mixamo、InterHuman、InterX）、电影/短视频数据、以及内部约650,000个3D资产库进行检索与生成，构建多样化的测试集（约3,000条场景与动作样本）。

**📈 对比分析**

通过与文本仅、全参考（Omni-reference）以及Seedance 2.0等基线对比，采用GSB评测显示WNM在场景、动作与相机控制方面分别获得约70%–80%的优胜率；用户实验表明生成迭代次数下降约70%，创作时长减少约50%，并显著提升用户控制精度与满意度。

**⚠️ 局限性**

局限主要包括：动作与交互细节的精度尚不足、缺乏针对人机交互的物理精确性、未完成专门的适配器训练以进一步提升控制细粒度、以及跨平台导入/导出标准化不足，需要进一步完善和扩展。

---

## 597. InstanceControl: Controllable Complex Image Generation without Instance Labeling

**arXiv ID:** 2606.31924 | [PDF](https://arxiv.org/pdf/2606.31924v1)

**作者:** Xiaoyu Liu `[一作]` (Harbin Institute of Technology), Wangmeng Zuo `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 64360 | [OpenAlex ID](https://openalex.org/A5100636655)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为InstanceControl的多实例可控图像生成框架，能够在无需人工实例标注的情况下实现对复杂多实例场景的精准属性控制。

**💡 创新点**

创新点包括：①利用Vision‑Language Model自动解析文本中的实例描述并与视觉条件关联，构建实例级对应关系；②设计Mask Refinement Module在生成过程中动态修正预测掩码；③引入共享SEG Token解决同一实例在文本中多次出现导致的语义不一致。

**🔧 技术方法**

采用的技术主要包括Sa2VA + SAM进行实例掩码预测、FLUX/ControlNet diffusion模型进行图像生成、U‑Net + attention 的Mask Refinement Module、共享SEG Token、LoRA微调以及多模态注意力机制。

**📊 数据集**

使用自建的50000张图像及对应掩码（来自SAM、COCO、UniWorld‑V1），生成canny、depth、HED等视觉条件，并利用Gemini 2.5 Pro生成长文本；评测集为MIG‑Eval、COCO‑POS、HiCo‑7K等公开基准。

**📈 对比分析**

与FLUX ControlNet、DreamRenderer、EliGen、CreatiLayout、Seg2Any等方法在Canny/Depth/HED等条件下进行对比，使用MIoU、Region‑wise & Global‑wise Quality、Local CLIP、Accuracy、FID等指标；InstanceControl在MIoU、Region与Global质量上均显著优于所有对手，特别是在无实例标注的情形下仍超越FLUX ControlNet。

**⚠️ 局限性**

局限性在于对极端多实例或高度重叠场景的掩码预测偶尔仍出现缺失或偏移，需要交互修正；模型依赖VLM推理，计算成本较高；在跨模态一致性和可解释性方面仍有提升空间。

---

## 598. Non-finite Axiomatizability of Generalized Medvedev Logics

**arXiv ID:** 2606.31893 | [PDF](https://arxiv.org/pdf/2606.31893v1)

**作者:** Han Xiao `[一作]` (Tsinghua University), Han Xiao `[通讯]` (Tsinghua University)

**通讯引用:** 40177 | [OpenAlex ID](https://openalex.org/A5100397594)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

研究并定义了一类新的中间逻辑——一般化的Medvedev逻辑，基于对具有顶点的有限根化Kripke框架进行乘积并去掉最大元素的构造；证明该类逻辑在包含关系下非有限可公理化，且与经典的Medvedev逻辑、Chequered逻辑等存在关联；进一步展示了该类逻辑在层次结构中存在可数多样的、不存在最小元素的特性。

**💡 创新点**

①首次证明从任意非单点的有限根化框架去顶点后得到的逻辑都不具有限可公理化；②将Medvedev逻辑的非可公理化结果推广到更一般框架；③构造了一系列“Chinese lantern”框架，用于检测逻辑差异；④揭示一般化Medvedev逻辑在中间逻辑格中的分布情况。

**🔧 技术方法**

使用Kripke框架与p‑morphism的技术；Jankov–de Jongh公式与框架同态的结构分析；Chinese lantern框架构造；对层次结构的可数链与极限分析；以及与Chequered逻辑的关联证明。

**📊 数据集**

无数据集，论文纯粹是形式逻辑与模型论的理论研究。

**📈 对比分析**

通过与已知逻辑（Medvedev、KC、Chequered等）的包含关系和有效性比较，证明了一般化Medvedev逻辑不具有限可公理化，且其包含层次丰富；性能（即逻辑性）上表现为在中间逻辑格中形成一个可数严格下降链，并且不存在最小逻辑。

**⚠️ 局限性**

限制与未解决问题：目前仅给出了足够条件的分离方法，尚未完全描述不同框架对应逻辑的相等或可比性；缺乏对该类逻辑在代数或阶结构下更精细的描述；未探讨多框架类的更广泛构造；未构造可数反链或反链；整体仍需进一步研究其代数结构与更一般构造的可行性。

---

## 599. Uniform Lyndon Interpolation via Non-wellfounded Proofs

**arXiv ID:** 2606.31890 | [PDF](https://arxiv.org/pdf/2606.31890v1)

**作者:** Borja Sierra Miranda `[一作]` (University of Bern), Thomas Studer `[通讯]` (University of Bern)

**通讯引用:** 888 | [OpenAlex ID](https://openalex.org/A5086903137)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

研究并证明了可证性逻辑 GLS 的统一 Lyndon 插值性质（ULIP），并给出了该性质的构造证明；

**💡 创新点**

首次将非良序证明理论与 Lyndon 插值结合，提出了 Lyndon 固定点与 Lyndon 方程组概念，并证明其在 GLS 中可解；

**🔧 技术方法**

利用非良序序列演算、局部进展证明框架、对称化的等式系统以及循环证明技术；

**📊 数据集**

无数据集；

**📈 对比分析**

未涉及实验或性能比较；

**⚠️ 局限性**

方法目前仅适用于已建立非良序序列演算的可证性逻辑，且对缺乏简单 Lyndon 固定点的逻辑（如某些解释性逻辑）可能不适用。

---

## 600. Improving path-tracking performance of an articulated tractor-trailer system using a non-linear kinematic model

**arXiv ID:** 2606.31889 | [PDF](https://arxiv.org/pdf/2606.31889v1)

**作者:** Marina Murillo `[一作]` (Research Institute for Signals, Systems and Computational Intelligence), Leonardo Giovanini `[通讯]` (Research Institute for Signals, Systems and Computational Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出并验证了一种融合前轮转向与中央铰链双转向的非线性运动学模型，并将其嵌入模型预测控制框架，实现对拖车位置的精确路径跟踪。

**💡 创新点**

创新点在于将双转向机制统一纳入运动学模型，首次实现了在NMPC控制器中直接控制拖车位置的完整系统，同时展示了该模型对复杂曲线（尤其是田头转弯）误差的显著降低。

**🔧 技术方法**

使用技术包括非线性模型预测控制（NMPC）、CasADi求解器、ROS+Gazebo仿真平台、运动学建模与符号求导。

**📊 数据集**

主要使用仿真参数（如车长、车距等）构建的虚拟环境；论文未引用公开真实数据集，全部实验基于Gazebo模拟的田地场景。

**📈 对比分析**

通过与仅控制前轮的传统路径跟踪方法对比，拖车在直线段误差从3.8 m降至<12 cm，转弯段误差从3.8 m降至<0.12 m，显示出显著的性能提升。

**⚠️ 局限性**

局限性包括：模型仅考虑平地运动学，未加入动力学与非平坦地形影响；控制输入存在振荡，需进一步平滑；未在真实农机上验证，仿真结果可能与实际性能存在偏差。

---

## 601. Uniform Interpolation of Basic Tense Logic

**arXiv ID:** 2606.31887 | [PDF](https://arxiv.org/pdf/2606.31887v1)

**作者:** Katsuhiko Sano `[一作]` (Hokkaido University), Katsuhiko Sano `[通讯]` (Hokkaido University)

**通讯引用:** 443 | [OpenAlex ID](https://openalex.org/A5065788231)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

证明了基本时态逻辑KT具有统一插值定理。

**💡 创新点**

首次将Visser的分层双射论证推广到时态逻辑，并给出框架扩展性和镜像性质的结果。

**🔧 技术方法**

采用分层bisimulation、bisimulation扩展属性、语义化的bisimulation量化等模型论技术。

**📊 数据集**

无实验数据集，全部为形式化证明。

**📈 对比分析**

通过理论证明与已有的逻辑系统比较，未出现实验性能指标，但证明了KT的插值性质。

**⚠️ 局限性**

对5公理、局部可表性及Lyndon插值等扩展的统一插值性仍未解决，且对多模态和Hybrid逻辑的进一步推广是未来工作。

---

## 602. Analytic Cut in Epistemic Logics with Distributed Knowledge

**arXiv ID:** 2606.31886 | [PDF](https://arxiv.org/pdf/2606.31886v1)

**作者:** Ryo Murai `[一作]` (Independent Researcher), Katsuhiko Sano `[通讯]` (Hokkaido University)

**通讯引用:** 443 | [OpenAlex ID](https://openalex.org/A5065788231)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

**🎯 论文内容**

本文提出并研究了基于K45、KD45和S5的分布式知识逻辑的Gentzen式序列推导系统，并证明了这些系统的分析切割（analytic cut）性质和Craig插值定理，进一步在允许空集为组时将分布式知识扩展为全局模态。

**💡 创新点**

创新点在于：①在cut消除失效的情况下仍通过Takano语义策略得到分析切割属性；②同时适用于三种不同的框架（K45、KD45、S5）；③引入空集组对应全局模态，证明核心证明理论结果仍然保持；④给出了完整的伪模型构造与树展开技术以完成语义完备性。

**🔧 技术方法**

主要技术手段包括：伪模型（pseudo‑model）构造、Ξ‑部分取值（partial valuation）、分析切割（analytic cut）和Maehara方法的变形；对全局模态的处理采用了根状态的特殊构造和树展开技术。

**📊 数据集**

无数据集，全部为理论证明。

**📈 对比分析**

与之前仅实现cut自由或不支持全局模态的序列系统对比，本文展示了即便cut消除不可行，也能通过分析切割得到相同的推理强度；Craig插值定理的证明为该类逻辑提供了新的结构性结果。

**⚠️ 局限性**

限制：仅讨论经典分布式知识；缺乏纯语法（syntactic）证明的分析切割；未覆盖公共公告、直觉逻辑或比较知识等更丰富的扩展；空组对应全局模态的处理仍需进一步探讨其在更大框架下的通用性。

---

## 603. Adapting Generalist Robot Policies with Semantic Reinforcement Learning

**arXiv ID:** 2606.31958 | [PDF](https://arxiv.org/pdf/2606.31958v1)

**作者:** Jagdeep Singh Bhatia `[一作]` (University of California Berkeley), Sergey Levine `[通讯]` (University of California Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种利用强化学习在语言提示空间（semantic action space）上学习并优化视觉-语言-动作模型（VLA）提示的算法，以实现机器人在面对复杂、长时序任务时的在线自适应。

**💡 创新点**

创新点在于：①将语言提示视为可控的语义动作，构建“语义MDP”，将RL从低层机器人动作空间提升到高层语言空间；②结合大规模预训练的视觉-语言模型（VLM）生成候选提示，随后通过实际交互进行“语义提示的经验锚定”，实现对提示的快速、精准优化；③实现了在真实机器人和仿真环境中，用极少的交互回合即可显著提升任务成功率。

**🔧 技术方法**

使用的核心技术包括：强化学习（基于Q学习的TD更新）、语义MDP框架、VLM（如CLIP/BLIP等）用于生成语言提示候选、VLA（如π_0.5、基于Bridge v2的模型）用于执行动作、以及经验回放与软策略采样。

**📊 数据集**

数据集和任务：仿真端使用Libero-10基准（10个长时序多步骤任务）以及Libero-90、Bridge v2等预训练数据；实测端使用WidowX机器人完成4个复杂长时序任务，并在每个任务上采集3条演示示例。

**📈 对比分析**

与基线对比：与传统动作空间RL方法（DSRL、Residual RL）以及基于上下文学习的VLM提示（ICL VLM）对比。实验结果显示，该方法在Libero-10和WidowX任务中，在仅60–100个交互回合内将成功率从接近0%提升至80%以上，显著优于所有对比方法。

**⚠️ 局限性**

局限性：①由于VLM在推理循环中参与，运行速度受限；②需要预训练的VLA能够在多样化语言提示下产生多样且可执行的行为，若VLA本身能力不足，则方法效果受限。

---

## 604. Complexity of Universality and Related Decision Problems for Unary Two-Dimensional Automata

**arXiv ID:** 2606.31974 | [PDF](https://arxiv.org/pdf/2606.31974v1)

**作者:** Taylor J. Smith `[一作]` `[通讯]` (St Francis Xavier University), Taylor J. Smith (St Francis Xavier University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文研究了单项式二维自动机的若干判定问题，给出了它们的可判定性、复杂度下界与上界；

**💡 创新点**

主要创新在于证明了单项式三向确定性模型的普遍性、等价性、包含性为NP‑hard；单项式二维两向模型的这些问题在P空间内；并对单项式三向非确定性模型的有界普遍性问题给出P空间上界与NP‑hard下界；

**🔧 技术方法**

采用了从3‑CNF‑Unsat、unary NFA universality、ST‑connectivity 等经典问题的归约，以及利用 Presburger 算法、线性程序和整数线性规划等形式化方法；

**📊 数据集**

该工作为纯理论研究，未使用具体数据集；

**📈 对比分析**

通过归约与复杂度分析，证明了上述判定问题在不同模型下的复杂度分类；相对于已知结果，进一步完善了单项式二维自动机的判定复杂度图谱；

**⚠️ 局限性**

主要限制包括：对于一般非确定性二维自动机的普遍性、等价性等问题仍未判定；理论结果上界为多重指数，存在与下界的巨大差距；实际实现和更紧凑算法仍待研究。

---

## 605. AnyBokeh: Physics-Guided Any-to-Any Bokeh Editing with Optical Fingerprint Transfer

**arXiv ID:** 2606.31959 | [PDF](https://arxiv.org/pdf/2606.31959v1)

**作者:** Xinyu Hou `[一作]` (Nanyang Technological University), Chen Change Loy `[通讯]` (Nanyang Technological University)

**通讯引用:** 82694 | [OpenAlex ID](https://openalex.org/A5005626854)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种从任意拍摄条件下的单张图像进行任意深度景深（bokeh）编辑的框架 AnyBokeh，能够在不需要先恢复全景清晰图像的前提下直接实现焦距和光圈的任意调整；

**💡 创新点**

创新点主要包括：1）通过物理模型建立圆盘模糊（CoC）与视差差的线性关系，得到源图像的光学指纹 κ_src，并将其传递到目标设置；2）双 CoC 条件化（dual‑CoC conditioning）实现源到目标的相对模糊合成；3）两阶段设计（Stage 1 估计 CoC 与视差，Stage 2 生成目标图像），从而避免所有输入都必须先转为全景清晰的瓶颈；4）构造高保真 UnrealBokeh 合成数据集，为 CoC 与视差学习提供完整的物理参数；

**🔧 技术方法**

采用了物理光学公式求 CoC，联合学习源 CoC 与视差的生成网络（基于 FLUX 1‑Fill‑dev + LoRA），利用线性 CoC–视差关系提取光学指纹 κ，随后使用双 CoC 条件化的生成编辑器完成相对模糊合成；

**📊 数据集**

主要使用了自建的 UnrealBokeh 合成数据集（含稠密深度、焦距、光圈、焦距、传感器尺寸等完整 EXIF 信息），在真实数据集 EBB! 与 RealBokeh 上进行零样本评估；

**📈 对比分析**

与传统的先去模糊再渲染（BokehMe、BokehDiff + 估计的全景清晰图像）以及去模糊模型（Restormer、DRBNet）进行对比；AnyBokeh 在任意‑任意、全景清晰‑bokeh 及 bokeh‑全景清晰三种任务中均取得更优的 LPIPS、DISTS 和 FID 分数，且不需要任何图像级的 bokeh 校准；

**⚠️ 局限性**

主要局限：1）在极度失焦源图像时，缺失高频细节导致重建模糊；2）极大或极小的 bokeh 球在训练样本稀缺时容易出现残留或空洞伪影；3）高分辨率推理依赖切片拼接，增加运行时；4）对光学指纹估计的精度受源图像模糊程度影响较大。

---

## 606. LeCropFollow: Latent Space Planning for Navigation in Unstructured Crop Fields

**arXiv ID:** 2606.31941 | [PDF](https://arxiv.org/pdf/2606.31941v1)

**作者:** Felipe Tommaselli `[一作]` (University of Sao Paulo), Marcelo Becker `[通讯]` (University of Sao Paulo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出LeCropFollow，一种在未结构化农田中利用完整热图信号进行潜在空间规划的视觉导航框架；

**💡 创新点**

创新点在于：①用自监督热图提取器保留热图的空间分布信息，避免信息压缩；②将TD-MPC2与潜在空间规划相结合，实现零射击（sim-to-real）部署；③显著降低行间缺口中的语义失误；

**🔧 技术方法**

采用的技术包括：自监督热图学习（RowFollowNet）、潜在空间编码器、TD-MPC2模型、MPPI规划、Mish激活的5M参数MLP、热图分布保留与潜在规划；

**📊 数据集**

使用数据集：在Gazebo中简化的随机颜色圆柱仿真环境；现场玉米田（花期与收获期）12–15次跑；RowFollowNet预训练热图模型；

**📈 对比分析**

与CROW和CropFollow++基线对比；在花期实验中平均碰撞5.1次，最大无碰撞距离27.9 m，性能与基线相当；在行间缺口实验中成功率93.3%（vs CropFollow++ 6.7%）；收获期表现无显著差异；

**⚠️ 局限性**

限制：仍存在机械反馈不足导致的物理层碰撞；缺口几何范围有限；缺少深度传感器或闭环控制，未能完全克服复杂地形和机械非线性问题。

---

## 607. Logics Containing wK4: Selection à la Fine

**arXiv ID:** 2606.31920 | [PDF](https://arxiv.org/pdf/2606.31920v1)

**作者:** Simon Santschi `[一作]` (University of Bern), Niels C. Vooijs `[通讯]` (University of Bern)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

论文通过对 Fine 的迭代选择方法进行推广，构建了一个在弱传递（weakly transitive）框架下的选择构造，利用该构造证明了在该类逻辑中子框架逻辑的有限模型性质（Finite Model Property）以及宽度有限（Finite Width）逻辑的 Kripke 完备性。

**💡 创新点**

创新点在于：①将 Fine 的选择方法从传递框架推广到弱传递框架；②将两大经典结果 Fine 的有限模型性质与有限宽度定理统一到同一构造中；③在构造中引入“diagram bisimulation”与稳定点技术，确保所选子模型可定义且不发生停滞；④证明了该构造在无限滤波集下仍可终止并产生原模型等价的有限原子模型。

**🔧 技术方法**

核心技术包括：模型理论的迭代选择-商构造、弱传递性下的子分割与消去可消去点、bisimulation（尤其是 diagram bisimulation）来保证可定义性、稳定点与覆盖性质的证明、以及 Hennessy‑Milner 以及原子化（atomicity）论证。

**📊 数据集**

论文未使用实验数据或数据集，全部为理论证明与形式化推导。

**📈 对比分析**

由于研究属于理论逻辑范畴，未进行实验或性能对比；评估依据为形式证明的严谨性与对已知定理的统一与扩展性，证明逻辑框架的完备性和模型构造的可行性。

**⚠️ 局限性**

局限性包括：仅适用于弱传递框架，无法直接推广到更一般的 n‑传递（n-transitive）或非传递逻辑；构造虽可生成最小可定义子模型，但未证明其唯一性；对于子框架逻辑的可判定性仍未解决，且对实际计算复杂度未给出具体上界。

---

## 608. Theory of Mind and Persuasion Beyond Conversation: Assessing the Capacity of LLMs to Induce Belief States via Planning and Action

**arXiv ID:** 2606.31916 | [PDF](https://arxiv.org/pdf/2606.31916v1)

**作者:** Ben Slater `[一作]` (University of Cambridge), Winnie Street `[通讯]` (Google)

**通讯引用:** 71 | [OpenAlex ID](https://openalex.org/A5095912641)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出非对话规划理论心智（NCP-ToM）概念，并通过NCP-ExploreToM框架对 600 个任务实例进行实验，评估大型语言模型在不使用对话说服而通过行动诱导他人信念的能力。

**💡 创新点**

创新点：①将传统被动问答式 ToM 评测转为主动行动规划；②构建 NCP-ExploreToM 任务集；③首次在代理系统场景下对 LLM 的非对话信念诱导能力进行系统比较。

**🔧 技术方法**

使用技术：GPT‑5、Gemini 2.5 Pro、Claude 4 系列 LLM；动作规划与信念诱导实验；与人类参与者进行对比评估。

**📊 数据集**

数据集：自建的 600 个任务实例，包含目标信念状态、可执行动作与场景布局。

**📈 对比分析**

比较方法与性能：在 agentic 场景下，GPT‑5 的成功率约 80%，是唯一超越人类的模型，但在不同上下文中的鲁棒性低于人类；所有模型在诱导真实信念方面表现优于诱导虚假信念。

**⚠️ 局限性**

局限性：①实验仅在限定场景下进行，缺乏真实世界多样性；②模型在不同上下文的鲁棒性不足；③未充分探讨安全与误导风险；④评测聚焦于行动规划，未覆盖所有 ToM 维度。

---

## 609. Learning Locomotion on Discrete Terrain via Minimal Proximity Sensing

**arXiv ID:** 2606.31912 | [PDF](https://arxiv.org/pdf/2606.31912v1)

**作者:** Jiale Fan `[一作]` (ETH Zurich), Robert Baines `[通讯]` (ETH Zurich)

**通讯引用:** 1058 | [OpenAlex ID](https://openalex.org/A5040068988)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在四足机器人足部集成低成本红外 ToF 近场传感器，并通过强化学习训练统一控制策略，以实现对不规则地形的提前感知与鲁棒行走。

**💡 创新点**

创新点在于将“预接触”近场感知直接嵌入足部，绕过全局视觉堆栈的延迟与遮挡，利用少量局部传感器实现高效、低功耗的地形感知与控制。

**🔧 技术方法**

使用 VL53L5CX ToF 传感器、Isaac Gym 仿真环境、PPO 强化学习、LSTM 网络、传感器噪声与延迟随机化、模拟到真实的转移技术。

**📊 数据集**

训练与评估基于自建多阶段地形集合（石块、平台、阶梯、斜坡等），不依赖公开数据集。

**📈 对比分析**

在仿真中与理想高度扫描和聚合高程图基线对比，足部传感器在关节偏移、模型尺寸误差、里程计噪声等干扰下表现更稳健；实测在稀疏步道与缺口上平均速度0.52 m/s，成功率显著提升。

**⚠️ 局限性**

局限包括对光学特性敏感（吸收/镜面表面导致测距误差）、传感器孔隙易被泥土堵塞、硬件寿命与高冲击环境适应性不足，以及对不同材质与极端天气的鲁棒性待进一步验证。

---

## 610. Improved Algorithms for Bounded-Degree (Subset) Traveling Salesman Problems

**arXiv ID:** 2606.31907 | [PDF](https://arxiv.org/pdf/2606.31907v1)

**作者:** Jongseo Lee `[一作]` (KAIST), Hyung-Chan An `[通讯]` (Yonsei University)

**通讯引用:** 20055 | [OpenAlex ID](https://openalex.org/A5027372746)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了针对受限度旅行商问题（BDTSP、BDTSPP、BDSTSP、BDSTSPP）的多种近似算法，改进了成本和度数的双指标逼近比率，并实现了在路径版本中的首个加性度数违背方案；

**💡 创新点**

核心创新在于引入一个新的低度生成树引理，将整数最优解的度数信息直接转化为一个满足约束的度数树，进而得到在度数上只产生常数级加性误差的算法；

**🔧 技术方法**

主要技术包括：1）基于整数最优解构造低度生成树；2）使用迭代收缩的边界树和Steiner树求解；3）利用边界T-join的线性规划整合性与最小成本求解；4）构造组合向量证明LP可行性；5）多层度数缓冲与分层约束的设计；

**📊 数据集**

论文未使用公开数据集，所有结果均为理论证明与算法分析；

**📈 对比分析**

与以往方法（如Friggstad‑Mousavi的 (3/2,+4) 和 Mousavi 的 (8/3,+4) 方案）相比，本工作在：①成本比率提升至 Christofides‑Serdyukov 的 3/2 或 5/3；②度数违背从乘性降低为加性，路径版实现 +4、循环版实现 +2；③子集版本成本比率提升至 14/9、11/5 或 2；总体性能显著提升；

**⚠️ 局限性**

局限性包括：①仍无法消除度数上限导致的 NP 难度，无法得到多项式时间的多项式逼近；②算法对图的结构没有额外假设，导致实现复杂度较高；③对非常大规模实例的实际效率尚未验证；④在某些特殊结构（如度数极低的节点）可能无法进一步优化。

---

## 611. A Gödel Modal Logic Over Witnessed Models

**arXiv ID:** 2606.31906 | [PDF](https://arxiv.org/pdf/2606.31906v1)

**作者:** Mauro Ferrari `[一作]` (University of Insubria), Ricardo Oscar Rodriguez `[通讯]` (University of Buenos Aires)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出一种基于可见 Kripke 模型的 Gödel 模态逻辑，并给出完整的约束反证系统与终止的回溯证明搜索方法，能够自动生成有限反例模型；

**💡 创新点**

通过引入可见性语义消除极限现象，恢复了有限模型性质，证明了逻辑的 PSPACE 完备性，并提供了可自动化的证明与反例生成机制；

**🔧 技术方法**

约束系统、后向证明搜索、Kripke 模型构造、Choco 求解器、JTabWb 引擎；

**📊 数据集**

无专门数据集，实验使用实现的 prover 进行验证与反例生成；

**📈 对比分析**

与传统非可见 Gödel 模态逻辑对比，证明其在 PSPACE 内完成验证，且能够在合理时间内生成有限反例；

**⚠️ 局限性**

仅适用于可见性模型，无法直接处理非可见 Gödel 模态逻辑，且实现仅针对单变量或简化模型，扩展性有限。

---

## 612. DigitalCoach: Communication and Grounding Gaps in Human and Agentic Computer Use Coaching

**arXiv ID:** 2606.31980 | [PDF](https://arxiv.org/pdf/2606.31980v1)

**作者:** Meng Chen `[一作]` (University of California, Berkeley), Amy Pavel `[通讯]` (University of California, Berkeley)

**通讯引用:** 2027 | [OpenAlex ID](https://openalex.org/A5014816733)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究创建了首个多模态人类专家-初学者电脑使用辅导数据集，评估现有多模态模型在辅导任务中的表现，并通过交互实验检验模型与人类辅导员的效果差异。

**💡 创新点**

创新点在于：①提供跨五款软件（Excel、Blender、FL Studio、Figma、Onshape）的72场专家‑学习者对话记录，包含屏幕录像、输入事件与文件快照；②细粒度对话行为与辅导方法的标注；③系统化比较多模态LLM在辅导语义多样性、视觉关联与学习成效方面的差距。

**🔧 技术方法**

采用了多模态大语言模型（GPT‑5.4、Gemini‑3.1‑Pro、Claude‑Sonnet‑4.6 等），并利用视觉文本联合输入、语义多样性评估（Self‑BLEU、Vendi Score）、对齐度量（MAUVE）及自动评估指标（CLAIR、BLEU、METEOR、ROUGE‑L、BERTScore）对模型进行评估；同时开展人类专家打分与交互式学习成效测试。

**📊 数据集**

使用自建的“Digital Coach”数据集，包含 28.1 小时屏幕录制、22,752 语句交互以及对应的输入事件与文件快照，涵盖 5 款软件的 18 个任务。

**📈 对比分析**

通过对照实验（人类辅导 vs. 6 大模型）和多维度评估（语言多样性、视觉定位、学习成绩），发现即使模型在内容相似度上可达 41‑42 分（CLAIR），但在语义多样性、视觉上下文利用以及学习效果上仍显著落后：人类辅导的学习进步率 81.9% 而模型仅 58.3%，且模型过度依赖直接指令、缺乏解释与反馈。

**⚠️ 局限性**

局限包括：①所有会话仅在 Windows 平台收集，未覆盖其他操作系统的 UI 变异；②数据量虽丰富，但仍受限于 72 场会话，难以全面覆盖更广的软件与用户群；③对话拆分与标注可能受口语断句、停顿影响；④评估主要关注即时学习成效，缺乏长期技能保持与情感支持的考量；⑤模型在实时视觉理解与主动监测方面表现不足，需进一步改进。

---

## 613. Making Sense of Touch from the Child's View for Contrastive Learning

**arXiv ID:** 2606.31943 | [PDF](https://arxiv.org/pdf/2606.31943v1)

**作者:** Max Whitton `[一作]` (Boston University), Boqing Gong `[通讯]` (Boston University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一套针对婴儿头摄像头视频的触感编码体系，生成了264k段两秒触感事件数据，并将该数据与语音对齐，利用对比学习训练视觉-语音-触感模型，验证触感能显著提升婴儿视觉概念学习效率。

**💡 创新点**

创新点在于将触感事件以结构化标签形式量化，采用半自动化编码与聚类转换为离散触感簇，首次将触感与视觉语音对比学习结合，展示触感在数据稀缺时的高效学习作用。

**🔧 技术方法**

主要技术包括：基于Gemini‑3‑pro的视觉文本交互式注释、CLIP/BLIP相似度过滤、K‑means聚类触感描述、三模态（视觉、语音、触感）对比学习框架和线性探测/零样本评估。

**📊 数据集**

使用SAYCam长期婴儿头摄像头记录（478小时）与BabyVLM‑V2的语音字幕，构建1.03M注释样本（含96k三模态对、672k视觉-语音、168k视觉-触感）。

**📈 对比分析**

与仅视觉‑语音预训练的基线相比，加入触感后线性探测在Picture Vocabulary上从51%提升至57%，在Labeled‑S上从84%提升至85%；在触感稀缺场景下，触感贡献更显著；触感簇粒度越细，细粒度任务（PV）越有利，粗粒度任务（Labeled‑S）则更适合少簇。

**⚠️ 局限性**

局限性包括：只利用有限数量婴儿的头摄像头数据，触感编码离散化可能失去细微语义；对比学习中触感与视觉映射仅通过聚类实现，缺乏真实触感测量；评估任务有限，未覆盖所有视觉推理维度；模型仍未完全模拟婴儿多模态学习机制。

---

## 614. Some Results on Causal Modalities in General Spacetimes

**arXiv ID:** 2606.31927 | [PDF](https://arxiv.org/pdf/2606.31927v1)

**作者:** Marco Lewis `[一作]` (Université Paris-Saclay), Nesta van der Schaaf `[通讯]` (Université Paris-Saclay)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5002415699)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在平滑时空的因果结构下，证明了严格因果关系（after relation）满足所谓的“after公式”，并提出了一个新的模态公式，展示二维时空的模态逻辑比高维时空更具表达力；同时系统分析了因果等级（causal ladder）不同层级下的模态逻辑包含关系。

**💡 创新点**

首次把“after公式”推广到任意光滑时空，证明后者的严格因果关系必然满足此公式；提出的二维专用模态公式揭示了二维时空在模态逻辑上的特殊性；系统整理因果等级对模态逻辑的影响，填补了先前仅针对闵可夫斯基空间的研究空白。

**🔧 技术方法**

使用差分几何与Lorentzian度量的定义、曲线因果关系的严谨定义、模态逻辑框架（Kripke框架、Sahlqvist公式、可完备性与可完备性证明）以及逻辑与时空属性的交叉推导。

**📊 数据集**

无实验数据集；本文为纯理论推导与逻辑分析，不涉及数值实验或观测数据。

**📈 对比分析**

比较方法主要是逻辑等价与包含关系的证明，展示在不同时空维度与因果等级下模态逻辑的层级关系；未给出具体数值性能指标。

**⚠️ 局限性**

尚未完全确定高维时空中后果式逻辑的完整表述，二维与高维时空逻辑差异的严格分离仍包含未证实的猜想；对特殊时空（如cNTV）仍需进一步研究。

---

## 615. Interface-Aware Neural Newton Preconditioning for Robust Cohesive Zone Model Simulations

**arXiv ID:** 2606.31921 | [PDF](https://arxiv.org/pdf/2606.31921v1)

**作者:** Zhangyong Liang `[一作]` (Tianjin University), Huanhuan Gao `[通讯]` (Jilin University)

**通讯引用:** 9474 | [OpenAlex ID](https://openalex.org/A5065527847)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种面向接口的神经网络预处理器 IA-NNP，针对共聚束模型(CZM)在软化过程中的 Newton 基础不匹配问题，改进了求解器的收敛性。

**💡 创新点**

创新点在于将传统的手工 Newton 修正方法视为规则化的接口提升，并通过学习式的状态依赖映射实现自适应、局部的预处理；同时提出了两级实现（初始化和迭代级）和严格的根等价与安全性检查。

**🔧 技术方法**

采用图神经网络（或局部 MLP）提取开口、张力、切线、损伤历史、残差、模式混合等接口特征，并在求解器中做右预处理或初值调整；同时利用手工标签和残差损失进行监督训练。

**📊 数据集**

训练数据来自一系列基准 CZM 案例，包括水平、圆形、双接口以及大规模活跃前沿等结构；每个案例生成接口图、特征、残差及手工修正标签。

**📈 对比分析**

与传统 NR、手工 NR 修正以及全场暖启动方法进行对比，评估指标包括收敛失败率、迭代次数、分支误差、残差下降等；实验显示 IA-NNP 在所有基准中显著降低失败率、迭代次数，并恢复正确分支，保持力–位移曲线不变。

**⚠️ 局限性**

局限性包括依赖于训练数据的泛化能力、对极端材料或几何变异的适应性尚待验证、以及在极大规模问题中可能产生的额外推理开销。

---

## 616. DriveWeaver: Point-Conditioned Video Inpainting for Controllable Vehicle Insertion in Autonomous Driving Simulation

**arXiv ID:** 2606.31918 | [PDF](https://arxiv.org/pdf/2606.31918v1)

**作者:** Junzhe Jiang `[一作]` (Fudan University), Li Zhang `[通讯]` (Fudan University)

**通讯引用:** 83022 | [OpenAlex ID](https://openalex.org/A5100425671)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在自动驾驶仿真中，提出了一种基于点云条件的视频修复扩散模型，用于在已记录的场景中高质量插入前景车辆。

**💡 创新点**

创新点包括：①利用点云渲染生成像素级几何条件，①将全局低帧率锚定与局部高帧率插值相结合的层次生成策略，②将生成的车辆提取为3D高斯表示，实现实时渲染。

**🔧 技术方法**

核心技术涵盖：Wan2.1 VACE‑14B视频流匹配扩散模型、轻量级PointAdapter适配器、流匹配训练、无监督分类器自由引导、3D Gaussian splatting 以及OmniRe 3D重建管线。

**📊 数据集**

使用公开的大规模自动驾驶数据集 Waymo Open Dataset 与 PandaSet 进行训练与评估。

**📈 对比分析**

在视觉真实感（FID、FVD、LPIPS）与几何一致性（PSNR、SSIM）方面，方法相较于 HUGSIM、StreetCrafter 和原始 Wan2.1 VACE‑14B 在多项指标上均获得显著提升，并在长时序生成中保持了更好的稳定性。

**⚠️ 局限性**

局限性在于仍需依赖高质量点云条件，对非LiDAR获取的点云支持有限；扩散推理速度较慢，需离线蒸馏才能满足实时仿真；此外，模型对非常长或极端视角的轨迹仍可能出现细节失真。

---

## 617. RESOLVE: A Multi-Resolution and Multi-Modal Dataset for Roadside Cooperative Perception

**arXiv ID:** 2606.31895 | [PDF](https://arxiv.org/pdf/2606.31895v1)

**作者:** Shaozu Ding `[一作]` (Arizona State University), Dajiang Suo `[通讯]` (Arizona State University)

**通讯引用:** 237 | [OpenAlex ID](https://openalex.org/A5053373823)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了RESOLVE数据集，该数据集包含多分辨率、多模态（LiDAR与摄像头）同步数据，用于评估路侧协同感知模型；

**💡 创新点**

创新点在于：①首次提供多分辨率LiDAR数据与同步摄像头数据的实景路侧数据集，能够在保持其他因素不变的情况下系统评估点云稀疏度对模型的影响；②通过控制分辨率的实验揭示多模态融合如何补偿LiDAR稀疏性，并给出设计成本效益感知系统的见解；

**🔧 技术方法**

使用的技术包括：多模态感知模型（SparseConv、Transformer、Mamba等）、点云采样与体素化、BEV特征级融合（BEVFusion、UniTR等）、多目标跟踪（AB3DMOT、CenterPoint、SimpleTrack等）以及多车协同感知框架（AttFuse、CoBEVT、V2X‑ViT等）;

**📊 数据集**

使用的数据集为自研的RESOLVE，包含约100k张交通摄像头图像、26k帧LiDAR点云，标注220k 3D边界框，涵盖10类交通参与者，并提供三种LiDAR分辨率（128、64、16束）;

**📈 对比分析**

通过统一的训练/验证/测试划分及相同的后处理（NMS、Anchor设定），对比了单模态与多模态、不同分辨率下的检测、跟踪与协同感知。结果显示：①分辨率从低到中提高约7–8% mAP，低至高提升仅1%；②多模态融合在低分辨率下可匹配甚至超越高分辨率单模态；③跟踪的AMOTA在中高分辨率下提升，AMOTP随分辨率提高显著下降；④协同感知中间特征级融合在中分辨率下效果最佳；

**⚠️ 局限性**

局限性在于：数据仅覆盖单一交叉口场景，缺乏多交叉口或走廊级的多样性；同时，模型在分辨率不匹配情况下性能衰退较大，需进一步研究跨分辨率的自适应方法。

---

## 618. QVal: Cheaply Evaluating Dense Supervision Signals for Long-Horizon LLM Agents

**arXiv ID:** 2606.32034 | [PDF](https://arxiv.org/pdf/2606.32034v1)

**作者:** Sergio Hernández-Gutiérrez `[一作]` (University of Tübingen), Matthias Bethge `[通讯]` (University of Tübingen)

**通讯引用:** 35918 | [OpenAlex ID](https://openalex.org/A5061457780)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个训练无关的评估框架（Q-Alignment Benchmark），用于直接衡量密集监督方法在多步代理任务中的价值对齐程度。

**💡 创新点**

创新点在于：①以参考策略的Q值作为标注，直接衡量信号的排序一致性；②将21种密集监督方法划分为七大族群并统一在同一数据集与模型上进行评估；③展示了简单的直接提示与排名方法往往优于复杂设计。

**🔧 技术方法**

技术包括：利用参考策略（最优或近似最佳）生成Q值标签；使用Spearman/Kendall相关系数评估排序一致性；在文本与视觉多模态环境下对多种LLM/VLM后端进行统一评测；采用Qwen3.5、Gemma4等开源大模型作为backbone。

**📊 数据集**

数据集为四个多步环境：FrozenLake、ALFWorld、OpenApps、TerminalBench，采集并标注了约1.2K个(state,action)对，且在每个状态下提供四个候选动作。

**📈 对比分析**

通过Q-Alignment指标对方法进行比较，发现直接提示和排名族群的相关系数最高；在不同模型规模、任务难度、观测模态和目标类型下表现稳定；复杂变体往往未提升对齐效果。

**⚠️ 局限性**

局限性包括：①Q-Alignment只能评估排序一致性，未覆盖所有训练细节（如损失、优化器、探索策略）；②参考策略的质量仍可能影响标签准确性；③仅对可标注的四个环境进行实验，尚需扩展到更多开放式任务；④视觉方法受限于当前VLM抽象，可能低估其潜力。

---

## 619. Generative Skill Composition for LLM Agents

**arXiv ID:** 2606.32025 | [PDF](https://arxiv.org/pdf/2606.32025v1)

**作者:** Xinyu Zhao `[一作]` (University of North Carolina at Chapel Hill), Tianlong Chen `[通讯]` (University of North Carolina at Chapel Hill)

**通讯引用:** 4293 | [OpenAlex ID](https://openalex.org/A5103073431)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种结构化的技能组合方法，通过任务条件下的自回归解码器生成有序技能序列，实现对技能子集、数量和执行顺序的联合预测；

**💡 创新点**

将技能组合建模为闭合词汇表的序列生成问题，利用约束自回归解码、辅助计数与集合头以及检索先验，实现子集、计数与顺序的协同学习；

**🔧 技术方法**

使用冻结的检索调优编码器（Qwen3-Embedding-0.6B）、3层Transformer解码器、TF‑IDF检索先验与集合成员资格先验，并结合训练时的辅助头；

**📊 数据集**

基于SkillBench的196技能库和约10k个任务‑技能组合样本（包含真实和合成数据），以及人工标注的真实任务；

**📈 对比分析**

与BM25、TF‑IDF、Qwen3-Embedding检索、LLM‑judge（Gemini‑2.5‑flash）以及SFT Qwen3-0.6B-Base对比。结构化组合模型在合成测试中比SFT高≈2.8pp、在真实任务上高≈11pp，且在SkillsBench上可将代码生成任务的通过率提升约45%（比无技能提升约23pp），并在更小的提示词量下接近金手册上限；

**⚠️ 局限性**

仅适用于已预定义的闭合技能库，难以处理动态生成或大规模开放技能；对极少见技能的泛化性有限；模型仍受训练数据分布偏差影响，需进一步提升对新任务的迁移性能。

---

## 620. FedLAB: Traceable Semantic Codebooks for Federated Multimodal Graph Foundation Learning

**arXiv ID:** 2606.32016 | [PDF](https://arxiv.org/pdf/2606.32016v1)

**作者:** Zekai Chen `[一作]`, Guoren Wang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 FedLAB，一种用于联邦多模态图基础模型的可追溯语义码簿框架，解决了联邦图学习中语义可追溯性缺失的问题。

**💡 创新点**

创新点：①将模态证据、节点语义、拓扑上下文分别编码为三类可追溯码簿；②通过联邦语义重心预训练在不泄露原始数据的前提下共享可重用的语义单元；③在模型预测时输出完整的语义追溯路径，实现透明可解释的决策。

**🔧 技术方法**

采用分层码簿结构、向量量化、可微离散读出、对比学习、拓扑一致性损失及代码簿正则化；联邦训练采用 FedAvg + 语义重心更新。

**📊 数据集**

在十个包含文本、图像、属性的多模态图数据集（Toys、Grocery、Bili Music、DY、KU、Bili Food、QB、Bili Cartoon、Flickr30k、SemArt）上进行评估，涵盖节点分类、链路预测、模态匹配/检索、图‑文本、图‑图像生成等六大任务。

**📈 对比分析**

与 FedAvg、Fed-MGNet、Fed-MHGAT、FedMVP、FedMAC、Fed-GFT、Fed-GraphCLIP、FedGFM+、FedBook 等 10+ 先进基线对比，FedLAB 在所有任务上均实现显著提升，平均提升 6.84%，单任务最大提升 16.73%，且保持了可解释的语义追溯。

**⚠️ 局限性**

局限：模型在通信量和内存占用上高于轻量级方法；码簿维度和聚类策略对性能影响较大；在极端数据稀疏或高异构度下仍需进一步验证。

---

## 621. Dual-Regime Absorbing Markov Chain Theory in Remote Estimation: Age-Minimizing Push Policies

**arXiv ID:** 2606.32010 | [PDF](https://arxiv.org/pdf/2606.32010v1)

**作者:** Ismail Cosandal `[一作]`, Nail Akar `[通讯]` (University of Maryland)

**通讯引用:** 1302 | [OpenAlex ID](https://openalex.org/A5080807022)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文针对远程估计系统，提出了一种基于AoII（年龄不正确信息）的多阈值推送式传输策略，旨在最小化平均AoII成本与能耗的加权和。

**💡 创新点**

创新点在于将四维状态空间压缩为与源相同的单维嵌入式DTMC，并引入双阈值吸收马尔科夫链（DR‑AMC）和对应的DR‑DPH分布，利用其分布特性闭式求解SMDP参数，从而得到最优多阈值策略；同时提出混合策略（调度算法）解决能耗约束问题。

**🔧 技术方法**

采用的技术包括：双阈值吸收马尔科夫链与分阶段DPH分布、离散时间半马尔科夫决策过程（SMDP）、基于政策迭代的最优策略求解、Faulhaber公式与Stirling数用于多项式AoII函数的期望计算、以及调度算法（steering）实现约束下的最优混合策略。

**📊 数据集**

实验采用人工构造的离散马尔科夫源（Q1–Q4）与几种DPH分布（几何、混合几何、两相DPH）作为通道延迟模型，未使用真实数据集。

**📈 对比分析**

与穷举搜索得到的最优阈值、单阈值（ST）与随机采样（RS）基准策略进行对比；实验结果显示SMDP方法得到的策略与穷举搜索完全一致，并且在不同权重λ下平均AoII成本显著低于基准策略，尤其在能耗惩罚较大时性能提升明显。

**⚠️ 局限性**

限制主要包括：对源过程仅假设为有限状态离散马尔科夫链，逆向通道被视为完美；DR‑AMC与DR‑DPH模型的构造和参数计算仍具有一定的数学复杂度；若系统规模增大，SMDP求解仍可能面临维数灾难；此外未考虑通道状态未知或不完美反馈等实际情况。

---

## 622. Human-as-Humanoid: Enabling Zero-Shot Humanoid Learning from Ego-Exo Human Videos with Human-Aligned Embodiments

**arXiv ID:** 2606.32009 | [PDF](https://arxiv.org/pdf/2606.32009v1)

**作者:** Xiaopeng Lin `[一作]` (Hong Kong University of Science and Technology), Kai Chen `[通讯]` (DeepCybo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了Human-as-Humanoid采集转化管道，将同步的egocentric与exocentric人类视频转换为机器人可执行的高自由度动作标签，并用这些标签训练PhysDex VLA策略，实现无目标任务机器人演示的零样本部署。

**💡 创新点**

①通过人类对齐的上半身人形减少人体-机器人差距；②采用同步ego-exo视频进行仅摄像机的运动恢复并通过分阶段IK生成可执行的机器人动作；③在VLA训练中引入FK-aware的Dual-Space Hierarchical Kinematic Constraint（DS-HKC）以保持手腕与指尖几何；④展示与传统机器人遥控相比的4.8‑7.2倍数据吞吐提升。

**🔧 技术方法**

基于人体关键点与网格恢复的egocentric–exocentric运动重建；分阶段正向运动学（IK）细化；条件流匹配DiT进行动作生成；可微前向运动学与DS‑HKC监督；使用PhysBrain视觉语言模型编码。

**📊 数据集**

自采集的1500小时同步ego‑exo人类演示（涵盖日常物体操作）以及用于评估的标准Humanoid任务数据；与GR00T N1.7进行对比。

**📈 对比分析**

通过关键点一致性对比、跨域动作tokenizer复现误差（MAE<0.01，指尖误差≈5mm）、FK‑aware训练损失与无FK差异，以及在七个真实机器人任务上与GR00T N1.7的阶段完成率与最终成功率比较。PhysDex在无目标任务机器人演示场景中取得更高复合分数，并在需要机器人演示的任务中显著降低所需数据量。

**⚠️ 局限性**

①动作估计质量直接影响标签偏差；②IK质量受机器人URDF、关节限制与标定影响；③仅针对特定机器人构型，迁移到新平台需重新retargeting；④仅提供运动学监督，缺乏接触力、摩擦等动力学信息，导致极高精度抓取任务仍需少量真实机器人数据；⑤在视角遮挡、模糊等条件下的运动恢复尚未充分鲁棒。

---

## 623. Surrogate Fidelity: When Can Open LLMs Explain Closed Ones?

**arXiv ID:** 2606.32008 | [PDF](https://arxiv.org/pdf/2606.32008v1)

**作者:** Philippe Chlenski `[一作]` (Meta), Nehal Bandi `[通讯]` (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究开放模型与封闭模型在机制解释上的相似度（后备可信度）

**💡 创新点**

提出后备可信度（surrogate fidelity）层级框架，并发现预测和表示层的一致并不保证因果归因的一致

**🔧 技术方法**

利用log-odds、留一删除归因、残差流几何分析、注意力聚合、相关系数、CVR等技术

**📊 数据集**

在BoolQ、ANLI、WinoGrande、LAMBADA、RACE等二分类及多选数据集上评估

**📈 对比分析**

对比各模型（Llama、Qwen、GPT、Gemini）在不同层级的相关性；发现预测层级相关高但归因层级相关低；表示层级相关高但方向性低，整体表现与模型规模无单调关系

**⚠️ 局限性**

仅针对二分类任务，缺乏对开放式文本生成的评估；留一删除归因对分布外输入可能产生影响；仅使用API可见信息，未覆盖更细粒度干预

---

## 624. Self-Study Reconsidered: The Hidden Fragility of Learning from Self-Generated QA

**arXiv ID:** 2606.32002 | [PDF](https://arxiv.org/pdf/2606.32002v1)

**作者:** Ekaterina Alimaskina `[一作]` (BRAIn Lab), Aleksandr Beznosikov `[通讯]` (BRAIn Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对基于自身文档生成问答对的自学习流程进行系统评估，揭示其在证据选择和回答阶段的两大风险，并在此基础上提出轻量级的程序性缓解措施。

**💡 创新点**

①证据选择非均匀、易被局部呈现吸引导致热点聚焦；②回答模型会遵循插入的指令式文本形成后续训练的偏差；③通过多样化生成、句子定位和块级清洗三种手段显著降低上述风险。

**🔧 技术方法**

使用大型语言模型（Qwen3-4B/32B、Gemma3-12/27B-IT、Llama3.1-8B-Instruct）进行问答生成与判别，利用基于句子/段落的过滤器（关键词‑正则、DeBERTa 分类器、PromptArmor）对文本块进行预处理。

**📊 数据集**

Cartridges 论文集合、LongHealth 病例库、QASPER 论文集三种中文/英文多领域数据集作为实验素材。

**📈 对比分析**

在证据覆盖率、注入命中率（hit rate）和合规率（compliance）等指标上，句子定位生成将注入命中率从 85% 降低至 13%，多样化生成在 32%–48% 之间；在回答阶段，关键词‑正则过滤器将平均合规率从 88% 降至 13%，同时保持约 93% 的干净文本保留。

**⚠️ 局限性**

仍存在局部指令被分散、多行组合或嵌入式文本难以完全剔除的情况；依赖生成器的内部决策与模型规模无关，需进一步设计更稳健的采样与过滤策略。

---

## 625. Radial Suppression Accelerates Algorithmic Generalization: A Geometric Analysis of Delayed Generalization

**arXiv ID:** 2606.32000 | [PDF](https://arxiv.org/pdf/2606.32000v1)

**作者:** Srijan Tiwari `[一作]`, Manjot Singh `[通讯]` (Indian Institute of Technology Roorkee)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

研究了神经网络在算法任务上出现的记忆‑泛化延迟现象，并通过径向‑角向激活空间分解提出软激活范数惩罚，实现了快速grokking。

**💡 创新点**

创新点在于首次将径向膨胀与泛化延迟关联，提出径向‑角向梯度分解框架，并用软约束激活范数惩罚在不同模型上显著加速了grokking。

**🔧 技术方法**

采用激活空间径向‑角向梯度分解、激活范数惩罚、谱压缩与曲率分析等技术，结合对比实验验证理论。

**📊 数据集**

实验使用模块算术任务（P=97）以及10M NanoGPT 3 位加法，评估了 MLP、Transformer 和 GPT 结构。

**📈 对比分析**

与交叉熵、权重衰减、MAN、Dropout 等正则化方法对比，激活惩罚将 gorkking 速度提升约 6 倍（MLP/Transformer）和 2.3 倍（NanoGPT），并显著降低 Hessian 迹、提高有效秩。

**⚠️ 局限性**

局限性在于验证仅覆盖算法学习任务，未证明对更广泛的任务集通用，并且惩罚系数 λ 需手动调节。

---

## 626. Reinforcement Learning with Metacognitive Feedback Elicits Faithful Uncertainty Expression in LLMs

**arXiv ID:** 2606.32032 | [PDF](https://arxiv.org/pdf/2606.32032v1)

**作者:** Gabrielle Kaili-May Liu `[一作]` (Yale University), Arman Cohan `[通讯]` (Yale University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种利用LLM自身对任务表现进行元认知评估的强化学习框架（RLMF）和元认知数据选择机制，并将其应用于完整的信度校准（包括数值与语言不确定性表达）

**💡 创新点**

①在RL训练中用元认知准确度对优势奖励进行加权；②用模型自评来挑选高低置信度样本做训练；③两阶段解耦策略，使数值校准与语言表达可分离、可定制

**🔧 技术方法**

强化学习（GRPO）+元认知反馈、元认知数据选择、针对性重写（使用外部LLM生成语言不确定性）

**📊 数据集**

PopQA（训练集），10个多样化任务作为评估（如PQA、SQA、HE、MMLU等）

**📈 对比分析**

与先前的元认知提示、Faithful Uncertainty Tuning以及Gemini‑3.1‑Pro、GPT‑5等基准进行比较；在*指标上平均提高29%/25%，FC ≥0.80，且在多数任务中小模型性能超过大型专有模型，保持准确率与事实校准不下降

**⚠️ 局限性**

仅关注信度校准；RL训练成本高；重写步骤依赖外部LLM；元认知提升未必等同于全面元认知，且在极低置信度或特定领域的表现仍有提升空间

---

## 627. DVG-WM: Disentangled Video Generation Enables Efficient Embodied World Model for Robotic Manipulation

**arXiv ID:** 2606.32028 | [PDF](https://arxiv.org/pdf/2606.32028v1)

**作者:** Ziyu Shan `[一作]` (Nanyang Technological University), Ziwei Wang `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种两阶段的Disentangled Video Generation World Model (DVG‑WM)，先用低分辨率模型预览物理交互的动态，再用高分辨率细化步骤生成高质量视频，并将两阶段通过flow matching与latent degradation高效连接。

**💡 创新点**

创新点包括：1) 动态学习与视觉合成分离，分别在最优分辨率下训练；2) 使用flow matching直接将低分辨率动态映射到高分辨率 latent，避免从噪声开始的冗余迭代；3) 引入latent degradation机制，鼓励模型在高分辨率阶段重建接触细节而非简单上采样。

**🔧 技术方法**

核心技术包括：CogVideoX-5B/2B 视频扩散模型、LoRA 微调、3D 先验 VAE 编码器/解码器、flow matching 训练目标、latent degradation 训练策略、以及 Vision‑Only Diffusion Policy 作为动作专家。

**📊 数据集**

使用 LIBERO 模拟数据集和自收集的 7k 条真实世界轨迹（Galaxea A1 与 UR7e 机器人平台）进行训练与评估。

**📈 对比分析**

与 CogVideoX‑5B、Wan2.1‑14B、LongScape、LVP‑14B 等先进视频世界模型进行对比；在 LIBERO 上 PSNR 20.019、SSIM 0.783、LPIPS 0.120、FVD 152.36、物体准确率 89%；推理时间仅 88.7 s（相较 LVP‑14B 354.2 s 提升 3.97×）；在真实世界任务中成功率 62.2% 对比 CogVideoX 的 34.0%（提升 28.2%）。

**⚠️ 局限性**

局限性包括：需要两阶段训练与额外的 flow matching 训练；对极端复杂长时序或非常细腻的接触动态仍可能出现误差；模型对预训练 VAE 与扩散模型的依赖较强，若数据分布差异较大，迁移效果可能下降。

---

## 628. SemRF: A Semantic Reference Frame for Residual-Stream Dynamics in Language Models

**arXiv ID:** 2606.32022 | [PDF](https://arxiv.org/pdf/2606.32022v1)

**作者:** Jian Gu `[一作]` (Monash University), Hongyu Zhang `[通讯]` (Chongqing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了语义参考框架（SemRF）来分离语言模型内部残差流的语义测量与动态，定义了语义Voronoi图、语义管道、最小行动轨迹，并通过曲率能量与轨迹压缩性来衡量知识密度与参数效率。

**💡 创新点**

①提出Anchor‑based SemRF，可在局部可测量语义坐标下捕捉深度方向语义轨迹；②利用语义Voronoi图和Margin‑Relaxed Semantic Tube形成分析约束；③定义最小行动轨迹（canonical trace）并证明其唯一性与离散样条性质；④将轨迹曲率与压缩性关联，给出局部知识密度的定量表征。

**🔧 技术方法**

数学分析（线性代数、凸优化、离散曲线拟合）、对齐条件与误差界定、凸二次规划求解最小行动轨迹、曲率能量与压缩性估计、语义Voronoi分割与语义管道构建。

**📊 数据集**

未在文中给出具体数据集，实验假设基于公开预训练语言模型（如Transformer系列）在标准语言建模/推理任务上进行验证。

**📈 对比分析**

通过比较教师强制的观测轨迹与最小行动轨迹之间的行动能量、曲率、回溯项等诊断指标来评估语义动力学的稳定性和压缩性；在满足接口对齐和投影残差条件下，证明最小行动轨迹唯一且误差受行动能量控制，曲率低的轨迹可压缩为少数线性段，暗示更高的参数效率。

**⚠️ 局限性**

仅适用于局部可测量的语义子空间，需满足受限双可逆性和接口误差小的假设；对全局模型行为缺乏保证；缺乏实验性量化指标与不同模型/任务的对比；对选取Anchor、tube约束等超参数的敏感性未深入探讨。

---

## 629. Cross-Space Distillation: Teaching One-Step Students with Modern Diffusion Teachers

**arXiv ID:** 2606.32020 | [PDF](https://arxiv.org/pdf/2606.32020v1)

**作者:** Anh Nguyen `[一作]` (Qualcomm AI Research), Anh Tran `[通讯]` (Qualcomm AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究跨空间蒸馏，提出轻量Bridge模块将低分辨率学生潜在映射到高分辨率教师潜在空间，实现一阶段学生学习高级教师知识。

**💡 创新点**

创新点包括①放宽教师与学生共享潜在空间的假设；②结合冻结VAE解码器前段与可学习投影头的Bridge；③采用潜在重建与注意力一致性双重目标训练。

**🔧 技术方法**

使用技术包括一阶段分布蒸馏（VSD/GAN）、流匹配、SwinIR投影头、冻结VAE前段、逆KL注意力一致性损失等。

**📊 数据集**

采用合成2M图像数据集（基于JourneyDB和LAION文本提示生成教师图像），不依赖大型真实图像数据集。

**📈 对比分析**

通过与基线初始化学生、剪枝模型以及传统一阶段蒸馏对比，使用HPSv3、HPSv2、ImageReward、MPS、DPG Bench等指标，Bridge蒸馏将SD 1.5的HPSv3从约5.4提升至9.4，合并模型进一步提升至≈10.5，显著提高生成质量且保持一阶推理。

**⚠️ 局限性**

局限性包括需要额外训练Bridge模块且对极端分辨率或完全不同架构的适配仍有限；推理仍需使用教师解码器，未显著降低推理时间；教师模型训练成本较高。

---

## 630. TRIAGE: Role-Typed Credit Assignment for Agentic Reinforcement Learning

**arXiv ID:** 2606.32017 | [PDF](https://arxiv.org/pdf/2606.32017v1)

**作者:** Yuanda Xu `[一作]` (LinkedIn Corporation), Alborz Geramifard `[通讯]` (LinkedIn Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在 Agentic RL 中基于语义角色的信用分配框架（Role‑conditioned credit），通过结构化 LLM 判断器为每个环境交互段打标签，并按角色给出不同的过程奖励，同时保留 GRPO 的最终结果优势。

**💡 创新点**

创新点在于引入四类语义角色（决定进展、探索、无进展基础设施、回退），将其映射为固定的分段奖励，解决了单一结果信用分配的两大盲区；并给出理论证明，角色条件信用是从角色标签可表达的 MSE 最优修正。

**🔧 技术方法**

采用结构化 LLM 判断器（如 Qwen3‑8B 思考模式）进行角色分类，基于 GRPO 的轨迹优势加权并加入角色条件过程奖励，随后批归一化；同时使用理论误差分解与实验对照评估。

**📊 数据集**

实验使用 ALFWorld、Search‑QA、WebShop 三个 Agentic 基准，并对 135 个手工标注的环境交互段进行角色审核。

**📈 对比分析**

方法与 GRPO、PPO、GiGPO、共享骨干值基线以及标量进程奖励基线对比；在 Qwen2.5‑7B 和 Qwen3‑1.7B 上相较 GRPO 提升 4–18% 的成功率，并使完成回合长度缩短 10–15%；实验表明角色分类准确性对性能提升影响显著，尤其是成功轨迹中的回退检测。

**⚠️ 局限性**

局限性包括：角色标签依赖 LLM 判断，存在误判；仅为每段分配单一角色，无法处理混合角色；依赖最终结果作为主要优化信号，无法完全消除判别误差；对更强代理的适用性与更细粒度角色分配仍待进一步验证。

---

## 631. The online monotone array completion problem

**arXiv ID:** 2606.32015 | [PDF](https://arxiv.org/pdf/2606.32015v1)

**作者:** Vishesh Jain `[一作]` (University of Illinois Chicago), Clayton Mizgerd `[通讯]` (University of Illinois Chicago)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了一种在线填充数组的游戏，目标是在保持数组非递减的情况下，尽快填满长度为n的空数组。

**💡 创新点**

确定了最优期望完成时间v_n的精确形式，并提出了一种显式的确定性策略，其期望完成时间为(1/2+o(1))nlog n。

**🔧 技术方法**

使用了随机化和自适应策略来分析不同策略的期望完成时间，并提出了基于区块的策略。

**📊 数据集**

没有具体提到使用的数据集，但研究基于从Unif[0,1]中独立抽样的样本。

**📈 对比分析**

与自然的优惠收集者策略进行比较，后者的期望完成时间为(1+o(1))nlog n。提出的确定性策略在性能上优于此策略。

**⚠️ 局限性**

在替换版本的游戏中，虽然允许覆盖先前的条目，但仍然存在未解决的上界和下界问题，特别是对于期望完成时间的确切形式。

---

## 632. FaceMoE: Mixture of Experts for Low-Resolution Face Recognition

**arXiv ID:** 2606.32040 | [PDF](https://arxiv.org/pdf/2606.32040v1)

**作者:** Kartik Narayan `[一作]` (Johns Hopkins University), Vishal M. Patel `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出FaceMoE，一种基于Transformer的Mixture of Experts（MoE）架构，用于低分辨率人脸识别，能够在保持高分辨率知识的同时，显著提升对低分辨率探测图像的特征提取与聚合。

**💡 创新点**

创新点在于：①使用多专属FFN专家和top‑k路由器，实现分辨率感知的特征抽取；②稀疏激活的MoE减少了微调时的灾难性遗忘；③结合CosFace、router z‑loss和load‑balancing loss，保证专家间均衡利用与训练稳定。

**🔧 技术方法**

技术方案包括：Transformer骨干（ViT-B/Swin-B），多专家MLP层，top‑k路由器，CosFace margin‑softmax损失，router z‑loss 与 load‑balancing loss，AdamW优化器及多阶段微调。

**📊 数据集**

数据集：预训练使用WebFace4M；低分辨率测试在TinyFace、IJB‑S、BRIAR；高分辨率与混合质量测试在LFW、CFP‑FP、CPLFW、AgeDB、CALFW、CFP‑FF、IJB‑B、IJB‑C。

**📈 对比分析**

与传统聚合方法（GAP、NAN、MCN、CAF, CoNAN）以及最新方法（ProxyFusion、PETALface）比较，FaceMoE在BRIAR TAR@FAR达到0.01%/0.1%/1%分别为42.36%/61.47%/81.27%，在IJB‑S TPIR@FPIR=1%为14.85%，在TinyFace Rank‑1/5/10分别为76.18%/79.69%/81.75%，均实现SOTA表现。

**⚠️ 局限性**

局限性：需要显著的预训练与大规模LR数据；MoE路由参数（N、k）的选择需经验调优；稀疏路由虽然降低计算，但仍增加模型复杂度；在极端域迁移或非人脸场景的通用性尚未验证。

---

## 633. SpheRoPE: Zero-Shot Optimization-Free 360 Panorama Generation with Spherical RoPE

**arXiv ID:** 2606.32033 | [PDF](https://arxiv.org/pdf/2606.32033v1)

**作者:** Or Hirschorn `[一作]` (Amazon Prime Video), Sagie Benaim `[通讯]` (Amazon Prime Video)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种零样本、无训练、无优化的框架，用预训练扩散变换器直接生成高质量的360°全景图像和视频。

**💡 创新点**

创新点在于设计了“Spherical RoPE”——按频段分配低频采用3D笛卡尔坐标、 高频采用周期量化的旋转位置编码，并结合“Semantic Distortion CFG”三路引导，使模型在推理时自然满足全景投影的周期性与极点收敛约束。

**🔧 技术方法**

主要技术包括频域分块旋转位置编码（SpheRoPE）、三路无监督引导（Semantic Distortion CFG）、以及对现有DiT模型的推理时位置编码与引导逻辑修改。

**📊 数据集**

使用了ODI‑SR图像数据集（1200个360°全景图像及其字幕）和两个视频基准（SphereDiff‑20 与 Stress‑20）进行评估。

**📈 对比分析**

与经过微调或基于优化的全景生成方法相比，本文在图像层面的FAED、DS等全景度量上均达或超越了对标模型，并在视频层面的时间稳定性与视觉质量指标上保持领先；在用户偏好实验中亦获得最高选择率。

**⚠️ 局限性**

局限性包括：对极端高运动或复杂动态场景仍可能产生轻微纹理失真；模型仍依赖于预训练基础模型的泛化能力，若基础模型不支持特定输入格式（如音频）需额外改造。

---

## 634. When LLMs Read Tables Carelessly: Measuring and Reducing Data Referencing Errors

**arXiv ID:** 2606.32029 | [PDF](https://arxiv.org/pdf/2606.32029v1)

**作者:** Yuqing Yang `[一作]` (University of Southern California), Huzefa Rangwala `[通讯]` (AWS AI Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

系统评估并缓解大型语言模型在表格推理任务中产生的数据引用错误（DRE），并提出基于轻量级批评器的推理时策略

**💡 创新点**

提出DRE作为评估维度，构建LLM-as-a-Judge自动检测框架，开发基于轻量级4B批评器进行DRE检测并用于过滤/拒绝采样，从而提升最终答案准确率

**🔧 技术方法**

利用大型LLM（如Sonnet‑3.7）做判别器，SFT+RLVR训练4B批评器，结合Best‑of‑N、批评器过滤、拒绝采样等推理时技术

**📊 数据集**

主要使用WTQ、TableBench、FinQA、SciTab、ToTTo等表格推理数据集（JSON/CSV/Markdown格式）

**📈 对比分析**

在多种模型（1.7B–20B）和任务上，批评器过滤或拒绝采样可提升准确率最多约12%，4B批评器在全量数据上提升≈2–5%，DRE率显著下降；与单纯提示或自我反思方法相比性能更佳

**⚠️ 局限性**

局限于表格任务，未对非表格域的DRE进行充分探究；批评器性能受训练数据和领域差异影响，仍有误检与漏检；缺乏对DRE成因的可解释性分析

---

## 635. Freeform Preference Learning for Robotic Manipulation

**arXiv ID:** 2606.32027 | [PDF](https://arxiv.org/pdf/2606.32027v1)

**作者:** Marcel Torne `[一作]` (Stanford University), Chelsea Finn `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种利用自由形式自然语言偏好轴来学习多维奖励模型，并基于该模型训练能够在多维轴上优化的机器人策略。

**💡 创新点**

创新点在于：①允许评注者直接用自然语言定义评判轴，保留多维反馈结构；②将奖励模型条件化于文本轴，实现对完整轨迹的多维评分；③在策略学习时同时使用所有轴的奖励，提升奖励密度并支持组合行为与测试时的可引导性。

**🔧 技术方法**

技术包括：基于Bradley‑Terry模型的多维奖励学习、Transformer‑style多模态奖励网络、离线/在线的奖励条件化策略提取（类似PPO或离线批处理RL），以及自然语言轴的文本嵌入与归一化处理。

**📊 数据集**

数据集包含四个真实世界操作任务（把立方体放入碗、折短裤、烤面包铺盘、桌面摆设）和两个仿真任务（物体重排、双模方形摆放），全部使用离线演示与人工偏好标注。

**📈 对比分析**

与单一整体偏好、优势条件化、加权回归、过滤BC和纯BC等基线比较，Freeform Preferences 在真实世界任务上平均提升 38%（相较第二名），在仿真任务上同样实现最高成功率；并展示了更快的任务完成速度、行为组合能力以及测试时可按需调节奖励的特性。

**⚠️ 局限性**

局限性包括：需要人工偏好标注成本较高；测试时需手动选取合适的奖励阈值；策略目前仅对固定的偏好轴可调，缺乏对动态或变长轴的支持。

---

## 636. CoMet: Context and Multiplicity Decomposition for Multimodal Uncertainty Estimation

**arXiv ID:** 2606.32012 | [PDF](https://arxiv.org/pdf/2606.32012v1)

**作者:** Sanghyuk Chun `[一作]` (Princeton University), Olga Russakovsky `[通讯]` (Princeton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对多模态大语言模型（MLLM）的不确定性估计方法CoMet，利用上下文特定与多重性特定两项分解来量化预测的不确定性；

**💡 创新点**

创新点在于引入二元匹配变量m，将不确定性拆分为上下文诱发的不确定性与输入-答案兼容性的不确定性，并通过轻量级后置模块实现生成自由的估计；

**🔧 技术方法**

核心技术包括：基于MLLM的匹配概率验证器、上下文条件答案分布近似、使用归一化熵（H₂）进行不确定性量化，以及训练轻量化头部预测Z₁、Z₂和u_t；

**📊 数据集**

使用了VQA v2、VizWiz、OK-VQA、HallusionBench、MMMU、MMMU Pro、MMStar等公开多模态问答与误差检测数据集，并构建了200k图文对及32个可行答案的候选集；

**📈 对比分析**

与生成式、采样式、扰动式、候选式等四类基线方法对比，CoMet在开放式VQA、误差检测和MCQ VQA任务上平均提升AUROC 10%+、AUPR 8%+，且推理时不需要生成或采样，速度快约3-6倍；

**⚠️ 局限性**

局限性包括：对候选答案集的依赖可能限制极端多样性的场景；匹配概率验证器对不同模型的泛化性有限；在严格的MCQ环境中，采样式方法仍能略胜一筹；

---

## 637. AxDafny: Agentic Verified Code Generation in Dafny

**arXiv ID:** 2606.32007 | [PDF](https://arxiv.org/pdf/2606.32007v1)

**作者:** Benjamin Breen `[一作]` (Axiomatic AI), Leopoldo Sarra `[通讯]` (Axiomatic AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并实现了基于 verifier-guided 迭代修复的 agentic 代码生成框架 AxDafny，能够在 Dafny 语言中联合生成程序实现与证明注解。

**💡 创新点**

创新点在于将正式验证反馈与 LLM 生成-修复循环结合，提出 AxDafny 框架；同时发布了大规模竞争式基准 LCB-Pro-Dafny，填补了 Dafny 端到端程序生成的空白。

**🔧 技术方法**

采用 Gemini‑3.1‑Pro / GPT‑5.5 作为基础 LLM，配合 deterministic 检查、LLM reviewer、记忆模块以及 verifier 诊断实现迭代修复；使用 Dafny 内置 SMT 验证器提供反馈。

**📊 数据集**

使用的主要数据集包括 DafnyBench（proof‑hint 任务）和新构建的 LCB‑Pro‑Dafny（250 道 LiveCodeBench‑Pro 竞赛题目），以及原始的 LiveCodeBench‑Pro 测试套件。

**📈 对比分析**

通过与 GPT‑5.5 pass@1、DafnyBench 基线、DafnyPro 等方法对比，AxDafny 在 DafnyBench 上实现 92.7%（725/782）验证成功，超过最强 baseline 6.5%；在 LCB‑Pro‑Dafny 上验证率为 56.4%（相比 11.6%），但在可执行测试中多为资源限制失效。

**⚠️ 局限性**

主要限制包括：验证成功与运行时性能不一致，资源限制（TLE/MLE）频繁出现；Dafny 训练数据稀缺导致模型能力受限；难以在硬题目上达到高覆盖率。

---

## 638. GEAR: Guided End-to-End AutoRegression for Image Synthesis

**arXiv ID:** 2606.32039 | [PDF](https://arxiv.org/pdf/2606.32039v1)

**作者:** Bin Lin `[一作]` (Peking University), Li Yuan `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了GEAR框架，实现VQ tokenizer与自回归生成器的端到端联合训练，并通过双分支硬/软赋值传递梯度，提升采样速度与图像质量。

**💡 创新点**

创新点在于用可微软赋值桥接非可微量化，避免STE崩溃，将对齐负担转移至生成器，从而获得更可预测的token分布和更快的收敛。

**🔧 技术方法**

采用变分量化（VQ）、自回归Transformer、REPA对齐、温度软max赋值、对齐损失和熵正则等技术。

**📊 数据集**

使用ImageNet‑1K（256×256）进行类条件图像生成，以及GPIC 100M图像集进行文本图像生成，所有实验在相同条件下对比。

**📈 对比分析**

与基线LlamaGen‑REPA比较，GEAR在gFID上提升至≈2.5–4.9（CFG下）并加速约10×；文本图像生成在FDD、IS、Precision等指标上均优于对照组；在不同tokenizer（VQ‑VAE、LFQ、IBQ）上保持一致性。

**⚠️ 局限性**

局限性在于受VQ重建上限约束，生成质量受压缩率影响；软赋值温度和对齐权重需要调参；未突破连续潜在模型的高保真重建水平。

---

## 639. PointSplat: Compact Gaussian Splatting via Human-Centric Prediction

**arXiv ID:** 2606.32036 | [PDF](https://arxiv.org/pdf/2606.32036v1)

**作者:** Yujie Guo `[一作]` (Zhejiang University), Xiaowei Zhou `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种人类中心化的Feed‑forward方法PointSplat，直接在三维空间推断压缩高质量的高斯云表示，从稀疏视角快速生成3D人类模型；

**💡 创新点**

创新点在于将预测从视角中心转移到3D空间，利用射线投射剔除内部冗余点并构建显式2D‑3D对应关系，再通过Point‑Image Transformer融合外观与几何信息，显著降低高斯数量、提高渲染质量并提升对视角和分辨率变化的鲁棒性；

**🔧 技术方法**

技术包括Plücker射线嵌入、基于空间砌块的可视性一致视觉体估计、数字微分分析器（DDA）射线投射、基于Sinusoidal编码的点特征、点-图像Transformer以及端到端RGB监督的损失（L1+LPIPS）训练；

**📊 数据集**

在DNA‑Rendering、ActorsHQ、PKU‑DyMVHumans、THuman2.0等真实与合成人类数据集上训练与评估；

**📈 对比分析**

与视角中心的GS‑LRM、DepthSplat、LVSM、GPS‑Gaussian等方法以及AnySplat、RoGSplat、LHM、4DGS、GauHuman等优化/生成方法对比，PointSplat在PSNR、SSIM、LPIPS等指标上均优于同类Feed‑forward方法，且高斯数量约为视角中心方法的33%，渲染速度更快、对视角数量和分辨率变化更鲁棒；

**⚠️ 局限性**

局限在于无法处理无界场景（受显存限制），且暂未实现时空一致的4D表示，需要更高效的架构与表示来进一步扩展。

---

## 640. AdaJEPA: An Adaptive Latent World Model

**arXiv ID:** 2606.32026 | [PDF](https://arxiv.org/pdf/2606.32026v1)

**作者:** Ying Wang `[一作]` (New York University), Mengye Ren `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 AdaJEPA，一种在模型预测控制（MPC）闭环中实现测试时自适应的潜在世界模型；

**💡 创新点**

创新点在于将自监督的预测误差用于每次行动后即时微调模型，使得规划过程在分布漂移下持续校准；

**🔧 技术方法**

采用联合嵌入预测架构（JEPA）加上轻量级梯度更新（如仅更新编码器最后层和预测器最后层），并使用 CEM 或梯度搜索作为轨迹优化器；

**📊 数据集**

在 PushT、PushObj、PointMaze 等视觉和动力学分布漂移任务的数据集上进行实验；

**📈 对比分析**

与冻结模型相比，AdaJEPA 在多种分布漂移（形状、视觉、动力学、布局）任务中均提升了 15%–30% 的规划成功率，且单步更新仅增加约 0.02–0.05 秒的延迟；

**⚠️ 局限性**

局限性包括：当测试环境所需特征不在预训练表征中时，轻量级微调无法完全弥补；同时，适配性能依赖于预训练模型的泛化能力和对目标空间的覆盖。

---

## 641. FLORA: A deep learning approach to predict forest attributes from heterogeneous LiDAR data

**arXiv ID:** 2606.32023 | [PDF](https://arxiv.org/pdf/2606.32023v1)

**作者:** Emilie Vautier `[一作]` (University of Gustave Eiffel), Cédric Vega `[通讯]` (University of Gustave Eiffel)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种基于Octree卷积网络的深度学习框架，能够利用法国国土高度多样的LiDAR点云同时预测六项森林结构属性。

**💡 创新点**

创新之处在于将高分辨率Octree骨干网络与生态时空辅助变量通过晚融合门控机制相结合，实现在季节、传感器、扫描角度等异质条件下训练单一、鲁棒的全国尺度模型。

**🔧 技术方法**

使用了Octree卷积神经网络、HRNet骨干、门控门限融合、Optuna超参搜索等技术。

**📊 数据集**

实验基于法国大陆32,052个NFI样地及其对应的LiDAR HD高密度点云，覆盖多季、不同传感器与扫描角度。

**📈 对比分析**

通过与单季模型、无辅助变量模型对比，采用rRMSE和R²评估，单季模型表现明显下降，而联合季节模型在高度预测rRMSE≈12.3%（R²≈0.88），体积预测rRMSE≈39%（R²≈0.74）。

**⚠️ 局限性**

主要限制是对单一物种（尤其是阑尾卷积）体积预测误差仍偏大，且在混合林和稀疏植被中表现不佳，未来需要结合光学时间序列以进一步提升跨季节泛化能力。

---

## 642. Automated Background Swapping for Robustness against Spurious Backgrounds

**arXiv ID:** 2606.32018 | [PDF](https://arxiv.org/pdf/2606.32018v1)

**作者:** Cesar Roder `[一作]` (Johannes Kepler University), Kajetan Schweighofer `[通讯]` (Cognizant AI Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的数据增强方法，通过分离前景和背景，减少分类器对虚假背景特征的依赖。

**💡 创新点**

创新点在于不需要在训练中观察到打破虚假相关性的样本，利用少量样本训练辅助模型来生成增强数据。

**🔧 技术方法**

使用了深度神经网络，具体包括一个检测器和一个生成器，检测器用于分离前景和背景，生成器用于填充背景。

**📊 数据集**

使用了多个数据集进行实验，包括Waterbirds、Spawrious和Spurious Vehicles数据集。

**📈 对比分析**

与多种基线方法进行比较，结果显示该方法在缺乏少数样本的情况下，能够显著提高最差组准确率和整体鲁棒性，超越了先前的方法。

**⚠️ 局限性**

局限性包括需要额外的标注成本来训练检测器，重组过程可能导致前景和背景之间的语义或几何不一致，以及在某些专业领域中背景填充生成模型的可用性问题。

---

## 643. Scalable Behaviour Cloning on Browser Using via Skill Distillation

**arXiv ID:** 2606.32014 | [PDF](https://arxiv.org/pdf/2606.32014v1)

**作者:** Kaisen Yang `[一作]`, Qinhuai Na `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种将人类浏览轨迹自动转化为可重用自然语言技能的框架，并在多种浏览器任务上显著提升智能体的成功率和交互效率。

**💡 创新点**

创新点包括①将决策不确定性视为核心瓶颈并采用自然语言技能取代低层操作；②通过行为证据抽象与程序化技能蒸馏，生成与页面无关的技能卡；③构建技能图进行聚合、合并与层级化；④实现技能与执行模型解耦，允许一次性蒸馏后被不同轻量模型复用。

**🔧 技术方法**

技术手段包括行为证据抽象（清洗、分段、上下文总结）、程序化技能蒸馏（指令式自然语言卡片生成）、技能库构建（技能图、合并与专化）、检索式技能条件执行以及使用Claude‑Sonnet‑4.6（或Qwen‑3.7‑plus）等大型语言模型进行蒸馏与执行。

**📊 数据集**

使用了公开的真实用户浏览轨迹、基于网页教程的合成轨迹等数据；评估基准包括WebArena‑Hard（受控自托管网站）、ClawBench（实时生产网站）以及OSWorld‑Style桌面任务（Ubuntu）。

**📈 对比分析**

对比未使用技能（Base）与使用技能（Skill‑On）在三大基准上进行实验：WebArena‑Hard成功率从60.5%提升至81.4%（+20.9pp）；ClawBench从32.9%提升至68.4%（+35.5pp）；平均交互工具调用减少27%；实验还验证了不同蒸馏模型对执行模型的跨模型迁移效果。

**⚠️ 局限性**

局限性包括：技能仍需检索与上下文校准，无法完全覆盖需要精确低层细节的情况；桌面实验中仍有13例未解决，表明技能不能替代最短路径或完美低层执行；此外，技能库需要持续更新与剪枝，以避免冗余与冲突。

---

## 644. Introspective Coupling: Self-Explanation Training Tracks Behavioral Change Despite Fixed Supervision

**arXiv ID:** 2606.32038 | [PDF](https://arxiv.org/pdf/2606.32038v1)

**作者:** Zifan Carl Guo `[一作]` (MIT), Belinda Z. Li `[通讯]` (MIT)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练语言模型用固定的、来自早期检查点的反事实解释，观察模型是否能在训练后自我解释其当前行为。

**💡 创新点**

发现“内省耦合”现象：即使解释标签未随模型行为更新，模型仍能生成更能反映自身当前行为的解释，而非原始标签；该耦合可跨模型、跨任务并随行为漂移而更新。

**🔧 技术方法**

采用监督微调（SFT）结合行为正则化（KL散度）与交叉熵损失；使用反事实解释构造、解释EM和行为EM评估；通过激活补丁（activation patching）进行机制分析。

**📊 数据集**

使用 Qwen3-8B、Llama-3.1-8B-Instruct 等大模型，任务数据包括 Hint‑MMLU、AITA、Refusal（FalseReject+WildJailbreak）、Synthetic Jabberwocky、WildChat、Warm Assistant、FineWeb、LLM‑LAT 等。

**📈 对比分析**

与无正则化或无解释训练的基线对比，Regularized模型在 Self‑>Orig 指标上优于原始标签，解释EM≈80%，行为EM≈0.8；在加入辅助数据或新任务时，解释仍能跟随行为漂移并保持高准确率，证明方法稳健且可扩展。

**⚠️ 局限性**

局限性包括：需要足够的行为多样性；当行为几乎不变或总是同一响应时，解释信号退化；对 OOD 泛化、机制完整性及在模型可能欺骗时的可信度仍需进一步研究。

---

## 645. PolicyGuard: From Organizational Policies to Neuro-SymbolicCompliance Review Engines

**arXiv ID:** 2606.32004 | [PDF](https://arxiv.org/pdf/2606.32004v1)

**作者:** Sameer Malik `[一作]` (Fujitsu Research India Private Limited), Amar Prakash Azad `[通讯]` (Fujitsu Research India Private Limited)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种神经符号框架PolicyGuard，用于将组织政策转化为可执行的规则引擎，对目标文件进行符合性评估；

**💡 创新点**

通过将政策语义正式化为关系逻辑规则并将其拆分为原子级提取问题，最终将LLM的答案与符号求解器结合，显著提升了可审计性、可维护性与一致性；

**🔧 技术方法**

使用LLM（如GPT‑4.1、Sonnet‑4.5等）进行原子级事实抽取，Typed Relational Logic定义规则，Z3求解器执行符号评估，辅以规则卡和抽取问题生成；

**📊 数据集**

利用公司内部95条NDA政策与5份真实NDA合同，共475个政策‑文件对，经过法律团队标注；

**📈 对比分析**

与四种零样本提示基线（直接提示、链式思维、法律推理、法律三段论）以及Claude Cowork进行对比；PolicyGuard在非合规F1上提升约30.8点，准确率93–95%，并在10次重复推理中保持1.3pp的稳定性；

**⚠️ 局限性**

当前实现未构建全局事实图，无法跨规则统一实体；缺乏对缺失或负面证据的第一类表示；仅在NDA与单一组织的内部手册上验证，泛化性与可复现性受限。

---

## 646. OopsieVerse: A Safety Benchmark with Damage-Aware Simulation for Robot Manipulation

**arXiv ID:** 2606.31993 | [PDF](https://arxiv.org/pdf/2606.31993v1)

**作者:** Arnav Balaji `[一作]` (University of Texas at Austin), Roberto Martín-Martín `[通讯]` (University of Texas at Austin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 DamageSim 与 OopsieBench，构建了一个通用的损伤感知仿真框架与安全基准，用于评估和训练家用机器人操作。

**💡 创新点**

创新点在于将机械、热、液体三类损伤通过统一的健康状态实时量化，并将该损伤信号无缝集成到任意仿真环境中，形成可扩展的安全强化学习与演示生成。

**🔧 技术方法**

使用了基于物理的损伤评估模型（机械冲击/压缩、温度阈值、液体接触），以及插件化实现支持 OmniGibson (OmniVerse) 与 MuJoCo (RoboCasa)；通过强化学习（PPO、DSRL）与模仿学习（flow‑based）训练安全策略。

**📊 数据集**

使用了 OopsieBench 提供的 32 个家庭操作任务、90 条人类演示（含有与无损伤反馈两种收集方式），以及现有的 NVIDIA GR00T 视觉语言动作模型用于基准评估。

**📈 对比分析**

对比方法包括无损伤反馈与损伤反馈演示、数据过滤、RL 加损伤惩罚等；实验表明损伤信号能将任务完成率提升至安全完成率约 3–10 倍，并显著降低真实机器人中的损伤率。

**⚠️ 局限性**

局限性包括损伤模型为近似，无法完全捕捉细致的物理破坏过程；需要手动为每个物体设置材料参数；在不同仿真后端间的参数迁移与统一仍有挑战。

---

