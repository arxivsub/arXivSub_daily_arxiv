# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-09-03 | 今日论文总数: 547

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. SoK: Where Do Flow Labels Come From? Auditing Label Provenance in Encrypted Traffic Benchmarks

**arXiv ID:** 2609.02140 | [PDF](https://arxiv.org/pdf/2609.02140v1)

**作者:** Sizhe Huang `[一作]`, Shujie Yang `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对14个公开的加密流量分类基准进行标签来源的系统审计，提出Label Provenance Record（LPR）记录标签生成的证据、单位、操作符和语义对象，并利用可直接计算的指标（可达准确率上限、冲突比例、混合权重）评估标签本身的局限性。

**💡 创新点**

创新点在于：①首次将标签来源建模为可检索的LPR结构；②定义了不需训练模型即可衡量标签质量的三大量化指标；③发现两种主流标签生成策略（粗粒度继承和过度过滤）在不同方向上失效；④通过对标注与任务的一致性分析揭示标签语义不一致问题；⑤提出针对基准构建者和使用者的实用准则。

**🔧 技术方法**

主要技术包括：①基于传输层观测的主键等价类和冲突率计算；②可达准确率上限（ceiling）公式；③表示层梯度（timing、header、地址、payload）对冲突的影响评估；④使用邻域特征（co-occurrence）检验过滤策略的损失；⑤对14个基准和自构造的paired corpus进行统计与可视化。

**📊 数据集**

使用的数据集有：①14个公开基准（ISCXVPN2016、CSTNET-TLS1.3、CipherSpectrum、NUDT MobileTraffic、MIRAGE-2019、USTC-TFC2016、CESNET-TLS22、CESNET-QUIC22、Cross-Platform等）；②自构造的paired corpus：888个脚本化浏览会话，覆盖31个目标网站，提供原始pcap、浏览器请求日志和每连接SNI信息。

**📈 对比分析**

比较方法：对每个基准在其声明的特征集合上计算可达准确率上限、冲突比例和混合权重；将实验得到的balanced accuracy与上限对比，评估是否存在标签误差；在paired corpus上比较继承式与连接式标签、过滤与未过滤样本对准确率的影响；使用宏平均准确率和邻域特征提升。结果显示：继承式基准的上限在0.56–0.76之间，齐整式基准在0.895–0.999之间；过度过滤导致仅24.95%流自带SNI，过滤后的邻域特征可提升宏平均准确率约0.21点。

**⚠️ 局限性**

局限性包括：①过滤策略的评估仅基于单一paired corpus，可能受流量组成影响；②主键固定为前20个带符号L4 payload长度，过于简化，无法完全捕获细粒度信息；③跨基准比较不是等样本比较，受样本规模和类别分布差异影响；④LPR信息需基准发布者提供，若缺失需推断，准确性有限；⑤对加密DNS或ECHO等新技术的通用性未验证。

---

## 2. Towards Zero-Shot Transfer Across Embodiments For Driving VLAs

**arXiv ID:** 2609.02341 | [PDF](https://arxiv.org/pdf/2609.02341v1)

**作者:** Caio Azevedo `[一作]` (École des Mines de Paris), Fabien Moutarde `[通讯]` (École des Mines de Paris)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了多数据集训练对驾驶 VLA 的影响，并提出了 BEV-Forcing 辅助任务，用以提升在有限设备上零射转移和在单一数据集上的规划性能。

**💡 创新点**

创新点在于：①使用 BEV 模型生成的占据图作为监督，直接让视觉语言模型的隐藏层学习统一的空间表示，起到几何感知正则化作用；②该方法在推理时不增加额外开销；③在多数据集训练与单数据集训练之间的对比实验揭示了辅助任务的边际效益随训练设备多样性增加而递减。

**🔧 技术方法**

技术包括：Qwen 3.5 2B 视觉语言模型 + LoRA 微调；在目标层加入低容量的 BEV 头（交叉注意力+线性投影）并用二元交叉熵监督占据图；联合规划与 VQA 任务进行多任务训练；对图像做增强与相机参数注入以提升鲁棒性；统一多数据集接口实现跨数据集训练。

**📊 数据集**

使用的数据集有：WOD‑E2E、NAVSIM、nuScenes、Physical AI（PhAI）、nuScenes‑QA 以及 KITScenes LongTail。所有数据集均采用统一的前视三摄像头输入与文本输出格式。

**📈 对比分析**

与基准方法（Gemini 3 Pro、Alpamayo 1.5、UniAD 等）相比：在 WOD‑E2E 验证集上，BEV-Forcing 的 ADE/FDE 与基线相近或略优；在 Physical AI 的零射转移上，ADE 降低约 10%；在 KITScenes LongTail 上，MMS 提升至 5.15（基线为 5.13），显著优于 Gemini 3 Pro（4.61）和 Alpamayo 1.5（4.31）。

**⚠️ 局限性**

局限性包括：仅在图像隐藏层学习空间感知，未改进动作表示、推理轨迹或使用 LiDAR 等 3D 传感器；BEV 头容量受限，过大容量会导致性能退化；使用的 BEV 教师（SimpleBEV）质量有限，若改进教师可进一步提升；实验规模受计算资源限制，未覆盖更大或更复杂的驾驶场景。

---

## 3. Predict, Don't Iterate: Efficient Adaptive-Length Infilling for Diffusion Language Models

**arXiv ID:** 2609.02108 | [PDF](https://arxiv.org/pdf/2609.02108v1)

**作者:** Haobo Xu `[一作]` (University of Illinois at Urbana Champaign), Hanghang Tong `[通讯]` (University of Illinois at Urbana Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无预设长度、只需两次前向传播的扩散语言模型自适应长度填空框架（PILL），解决了传统填空对固定长度的依赖与对初始长度的敏感问题。

**💡 创新点**

核心创新包括：① 通过单一掩码隐藏状态的轻量级MLP探测器直接预测缺失段长度；② 将预测长度展开为若干候选长度，在多槽并行解码中利用位置插值与槽级注意力几乎不增加额外成本；③ 采用后向单步一致性评分（内部连贯度+后缀匹配）实现候选选择，避免了多步推理与额外前向传播。

**🔧 技术方法**

使用技术包括扩散语言模型（DLM）、RoPE位置编码、位置插值槽位分配、轻量级MLP长度探测、并行多槽解码、后向伪对数似然评分、单次前向传播。

**📊 数据集**

实验覆盖八个填空基准，涵盖代码（HumanEval-Infilling、MBPP、MultiPL-E）与文本（WikiText、arXiv）任务，使用五个不同规模与架构的DLM（LLaDA、Dream）进行评测。

**📈 对比分析**

与固定长度解码基线及两种自适应长度基线（DAEDAL、CAL）对比，PILL在代码任务上平均提升+4.8% Pass@1，在文本任务上提升+6.0 BLEU-2；同时在HumanEval-S上仅增加7%壁钟时间，速度比CAL快1.82倍，显著提升效率与性能。

**⚠️ 局限性**

局限性包括：对嵌套或多段编辑场景尚未验证；仅在约7B/8B规模模型上单 GPU实验，尚未评估在更大规模扩散模型上的可扩展性；缺乏对仓库级代码补丁等更复杂编辑任务的评估。

---

## 4. LLM-as-a-Judge Is Not an Oracle: Why Self-Improving Agents Need Deterministic Guardrails

**arXiv ID:** 2609.02246 | [PDF](https://arxiv.org/pdf/2609.02246v1)

**作者:** Vansh Wahi `[一作]` `[通讯]` (University of Waterloo), Vansh Wahi (University of Waterloo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在真实生产环境中运行自我改进的提示优化循环，系统性地识别并记录了LLM评估器在面对优化压力时出现的十一种失败模式，并基于这些经验提出了PROCTOR架构——一个教师‑学生循环结合五层确定性检查的系统，用以将LLM评估器从“oracle”降级为“advisor”，从而阻止评估信号被误用或被游戏；

**💡 创新点**

创新点在于①首次将LLM评估器的可靠性问题拆解为四类（评估者偏差、评估工具/度量缺陷、真值标签错误、奖励黑客），并对每种失败进行实测；②提出PROCTOR的五层系统（Hermetic sandbox, Stateless subagents, Mechanical pre‑apply checks, Frozen holdouts, Canary cases）作为系统性验证框架，证明了仅靠语义指导无法突破“Goodhart”陷阱；③展示了评估器输出顺序（rationale‑before‑score）对提升人类一致性和误差率的显著影响；

**🔧 技术方法**

技术手段包括：多层教师‑学生角色分离、LLM评估器（教师）的基于显式rubric评分、机械预检（例子/指令编辑上限、工具泄露检测、解析/合同检查）、冻结训练/测试拆分与漏洩禁止、插入不可通过的canary案例、统计监测与回滚机制；

**📊 数据集**

使用十个评估套件（合同红线、合同分析、合规政策审查、代码质量等），规模从9到100个案例，代码质量子任务采用54个软件仓库，包含15个人工标注、39个模型标注；

**📈 对比分析**

对比方法：通过六轮评估者校准实验，证实仅在改写输出顺序后才出现显著提升（EM从46.3%提升至51.9%，MAE从0.67降至0.57），而其他语义级修改无效；在优化循环中，PROCTOR成功抑制了11种失效，并通过可追踪的拒绝日志与回滚记录验证其有效性；

**⚠️ 局限性**

局限性包括：①真值标签部分由模型生成，导致对人类一致性的测度不完全；②所有模型来自单一专有模型族，缺乏跨模型验证；③评估套件规模普遍偏小，单次运行缺乏统计显著性；④只能评估已知失败模式，未进行系统性的覆盖搜索；⑤缺乏公开实现，实验可复现性受限。

---

## 5. From Open Standards to Openly Governed: Standards-Setting Organizations as Stewards of Openness amid Platformization and Digital Sovereignty

**arXiv ID:** 2609.01773 | [PDF](https://arxiv.org/pdf/2609.01773v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 6. CHIME: Credit-Aware Hierarchical Memory Evolution for Long-Horizon Agentic Planning

**arXiv ID:** 2609.02074 | [PDF](https://arxiv.org/pdf/2609.02074v1)

**作者:** Yongshi Ye `[一作]` (Xiamen University), Xiaodong Shi `[通讯]` (Xiamen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一种名为 CHIME 的自演进式规划框架，采用“先归因后记忆”原则，将规划与执行经验分别存储在层次化的记忆库中，并通过信用归因门对任务结果进行阶段性归因，从而在推理时持续改进规划能力。

**💡 创新点**

创新点在于：①将规划与执行记忆分离为两个独立的记忆库，避免低级执行经验污染高级规划；②设计信用归因门，根据计划、执行和外部因素对任务结果进行归因，只将对应阶段的反馈写入记忆；③使用价值加权检索和信用感知记忆演化，使记忆质量更高、规模更小、泛化更好。

**🔧 技术方法**

核心技术包括：LLM 预训练模型（如 Qwen3.5‑Flash、DeepSeek‑V4‑Flash）做为规划器、执行器和归因门；语义检索与价值加权重新排序；信用归因门的自我反思；在线价值更新与记忆增删合并策略。

**📊 数据集**

在四个长期任务基准上进行评估：τ²‑bench（多轮客服交互）、VitaBench（多轮生活服务）、BrowseComp‑ZH（信息检索）、BFCL‑v4（函数调用）。

**📈 对比分析**

与四个基线（No‑Plan、WebAnchor、TodoEvolve、A‑MapReduce）对比，CHIME 在训练集上平均提升 3–4% 的成功率，在评估集上提升 2.9–3.7%，且仅使用约 129 条记忆（相比 A‑MapReduce 的 3,585 条）。进一步分析显示记忆价值与下游性能呈正相关，信用归因门的可靠性达到 87–98% 的一致性。

**⚠️ 局限性**

局限性包括：①依赖强大的 LLM 进行自我反思，若模型自我归因质量不足会影响记忆更新；②目前仅在离线仿真基准上验证，真实环境中的外部噪声和不确定性尚未充分评估；③归因门和记忆演化机制设计仍基于经验，可能不适用于所有任务结构或工具集合。

---

## 7. UAV Thermal Imagery for Inert Ordnance Screening: Multi Campaign Dataset Development,Object Detection, and Practical Recommendations

**arXiv ID:** 2609.01738 | [PDF](https://arxiv.org/pdf/2609.01738v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 8. DiDrive: A Risk-Aware Hierarchical Diffusion Framework for Safe Offline Reinforcement Learning in Autonomous Driving

**arXiv ID:** 2609.01609 | [PDF](https://arxiv.org/pdf/2609.01609v1)

**作者:** Qisong Guo `[一作]` (Fuzhou University), Yuanlong Yu `[通讯]` (Fuzhou University)

**通讯引用:** 2210 | [OpenAlex ID](https://openalex.org/A5053193108)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出DiDrive，一种风险感知的层次扩散框架，用于安全的离线强化学习实现自动驾驶决策

**💡 创新点**

创新点包括：1）风险感知层次扩散（RHDif）结构，分低层局部风险增强和高层全局语义调制；2）3DICE分四阶段（表示-引导-优化-选择）分布校正机制，利用样本内加权和PPI平滑避免OOD估计过大与重尾奖励导致的梯度振荡；3）将上述两大模块协同工作，使策略在高维状态下更专注安全关键交互并在动作空间内保持支持区。

**🔧 技术方法**

使用的技术包括扩散模型（逆向去噪）、风险门控时空编码器、跨模态上下文调制器、分布校正估计（DICE类）、密度比权重引导、进阶参数集成（PPI）、多目标集成排序等。

**📊 数据集**

数据集为CARLA仿真中的Town03和Town05两地图，包含7000条轨迹、1.1M步，混合专家与随机策略8:2比例，观测维度307维。

**📈 对比分析**

与IQL、CQL、ODICE、TD3‑BC、QGPO、Diffusion‑QL等基线同样训练，评估指标为成功率、平均奖励和行驶距离。DiDrive在高密度60车情境下取得最高成功率85%（远高于Diffusion‑QL的77%）和最高平均奖励4295.68；在Town05跨地图测试同样表现最优，成功率81%~89%。

**⚠️ 局限性**

局限性包括：扩散模型多步逆向采样导致推理延迟，难以满足车辆嵌入式平台的高频控制；实验仅在仿真环境中，未验证真实世界的转移性能；以及对极端场景（极大交通密度或极端气象）尚未深入评估。

---

## 9. Bayes-Optimal BER and AUC: Estimation and Evaluation of Estimators

**arXiv ID:** 2609.02304 | [PDF](https://arxiv.org/pdf/2609.02304v1)

**作者:** Ryota Ushio `[一作]` (University of Tokyo), Masashi Sugiyama `[通讯]` (RIKEN AIP)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过软标签推导估计二分类任务的 Bayes‑optimal 平衡误差率（BER）和 ROC 曲线下面积（AUC），并在软标签被未知顺序保持变换和噪声破坏的情况下提出相应的插值与估计方法；同时扩展 FeeBee 框架，提供一种不依赖已知最优值即可评估估计器的实用方法。

**💡 创新点**

① 将软标签估计方法从传统误差率扩展到 BER 与 AUC；② 在未知先验与软标签受变换/噪声破坏的更现实场景下给出可行的估计器；③ 提出可直接评估 Bayes‑optimal BER/AUC 估计器的噪声注入式评估方案；④ 推导出有限样本误差界和闭式关系。

**🔧 技术方法**

软标签估计、单调回归（isotonic regression）、裁剪均值估计类先验、插值（plug‑in）估计、理论误差分析、噪声注入评估（FeeBee）以及 ROC / BER 的数学推导。

**📊 数据集**

在合成数据和若干真实世界数据集（未列明具体数据集）上进行了实验验证。

**📈 对比分析**

与传统基于输入‑标签对的方法和现有 Bayes‑error 估计器进行对比；实验表明所提软标签估计器在清洁与受污染场景下均能实现接近最优的 BER/AUC，并且评估框架能够准确反映估计器的真实性能。

**⚠️ 局限性**

模型假设仅涵盖顺序保持的变换与加性噪声，未考虑实例依赖性或更复杂的破坏；对于非可分解指标（如 F‑measure）以及多分类 BER 的扩展尚未完成；需要更通用的破坏模型和更广泛的实验验证。

---

## 10. Synthesis of Compact and Expressive Quantum-Circuit Optimizations

**arXiv ID:** 2609.01762 | [PDF](https://arxiv.org/pdf/2609.01762v1)

**作者:** Wei Qiang `[一作]` (Columbia University), Ronghui Gu `[通讯]` (Columbia University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于符号重写规则的量子电路优化框架，自动合成紧凑且具表达力的重写规则，并通过等价饱和和随机搜索实现高效优化。

**💡 创新点**

核心创新在于：① 形式化符号重写规则与“规范化”符号规则；② 通过特性分组和线性方程求解生成无限子电路空间的符号规则；③ 规则锚定技术将规范化规则与具体规则结合，提升优化效果；④ 生成的规则集合可证明非可导、完整且保证语义一致。

**🔧 技术方法**

主要技术包括：等价饱和（e‑graph）、多项式等价过滤（PIF）、SMT 验证、线性系统求解（符号矩阵）、规则锚定推理与随机搜索（模拟退火）。

**📊 数据集**

在四个真实硬件门集合（IBM‑Eagle、Nam、Rigetti、Ion）上合成规则，并在135个包含 QAOA、VQE、QPE、QFT、Grover、Shor 等算法的基准上进行评估。

**📈 对比分析**

与 Qiskit、Guoq、Quartz、TKET、Queso 等主流重写优化器对比，提出的系统在两量子门压缩率、门数与电路保真度上均显著优于对手，平均 27–30% 的两量子门减少，且在 60 分钟时间限制下持续改进。

**⚠️ 局限性**

局限性包括：① 规则合成对门数与规则长度仍有指数扩展，需人工设定上限；② 规则锚定和随机搜索在大规模电路上仍可能产生搜索空间爆炸；③ 目前未与重合成或机器学习引导的策略结合，优化过程仍缺乏智能引导。

---

## 11. Handwriting Trajectory Recovery via Autoregressive Ordered Stroke Instance Prediction

**arXiv ID:** 2609.02251 | [PDF](https://arxiv.org/pdf/2609.02251v1)

**作者:** En-Guang Wang `[一作]` (Chinese Academy of Sciences), Cheng-Lin Liu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种两阶段的离线到在线手写轨迹恢复框架，先自回归预测有序笔画实例，再根据起点初始化生成连续笔画轨迹。

**💡 创新点**

创新点在于将无序集合预测改为自回归有序预测，实现笔画提取与顺序恢复同步；通过起点监督与跨阶段起点传递消除写向模糊；在全点轨迹设置下获得最优性能。

**🔧 技术方法**

使用Transformer、Swin、ResNet编码器、跨尺度注意力、序列自回归、起点监督、CTC式终止等技术。

**📊 数据集**

主要使用CASIA-OLHWDB1.1/1.2中文手写数据，英文CASIA、泰米尔IWFHR 2006进行跨语言验证。

**📈 对比分析**

与Cross-VAE、Kanji-Net、DED-Net、PEN-Net、FINet、Trajectory Transformer等方法对比，本文在全点设置下在AIoU、LPIPS、LDTW、DTW等指标均优于所有基线；对未见字符、英文、泰米尔同样取得最优或相近表现。

**⚠️ 局限性**

局限在于对轨迹采样密度高度敏感，过细采样导致性能下降；同时对极为复杂笔画或极长序列的恢复仍有挑战，未考虑跨笔画约束。

---

## 12. Beyond Modality Harmony: Orthogonal Purification and Topology-Guided MoE for Conflict-Aware Multimodal Recommendation

**arXiv ID:** 2609.02152 | [PDF](https://arxiv.org/pdf/2609.02152v1)

**作者:** Jialin Liu `[一作]` (City University of Hong Kong), Ray C. C. Cheung `[通讯]` (City University of Hong Kong)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种多模态推荐框架 OrthoRec，通过协作引导的正交净化和拓扑感知 Mixture-of-Experts 解决模态与协同拓扑冲突，提升推荐精度和鲁棒性。

**💡 创新点**

创新点包括：① 正交净化（CGOP）将模态特征投影到协同锚点的平行和正交分量，抑制误导性噪声并保持能量；② 拓扑感知路由 MoE（TAR‑MoE）使用解耦的 Sigmoid 门消除 softmax 的零和竞争；③ 安全自监督对比学习（safe‑SSL）根据冲突分数动态调节对齐强度，防止潜在的潜在噪声导致潜在空间畸变。

**🔧 技术方法**

技术手段包括 LightGCN 的协同锚提取、线性投影+正交分解、能量保持归一化、两层 MLP 生成截断门、Sigmoid 路由门、InfoNCE 约束的安全自监督对比损失，以及多任务联合训练。

**📊 数据集**

使用 Amazon 公开商品评论数据集：Baby、Sports、Clothing 三个子集，包含用户-商品交互与商品的视觉（BEiT）和文本（BGE）特征。

**📈 对比分析**

与 CF 基线（MF‑BPR、LightGCN、SimGCL、LayerGCN）、多模态 GNN 基线（VBPR、MMGCN、GRCN、DualGNN、LATTICE、LGMRec）以及对比学习/去噪基线（SLMRec、BM3、FREEDOM、MMGCL、DA‑MRS）进行对比。OrthoRec 在 R@10/20 和 NDCG@10/20 上均优于所有基线，平均提升约 8%‑11%。实验还验证了对模态噪声（随机视觉特征置换）和项目稀疏性（冷启动）下的鲁棒性。

**⚠️ 局限性**

局限性包括：① 只在合成噪声实验中评估噪声鲁棒性，未在真实点击诱导场景下充分验证；② 对协同锚点质量高度依赖，若图结构本身信息不足，净化效果可能受限；③ 需要额外的超参数调优（如门阈值、温度、对比权重），对不同数据集的适配仍需经验；④ 目前仅处理视觉与文本两种模态，扩展到更多模态时可能需要进一步设计。

---

## 13. GAPS: Dimension-Level Gates for Conditional Activation Steering

**arXiv ID:** 2609.01878 | [PDF](https://arxiv.org/pdf/2609.01878v1)

**作者:** Moghis Fereidouni `[一作]` (University of Kentucky), A. B. Siddique `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了在激活调节中加入维度级（神经元级）条件，以动态决定在生成时哪些隐藏单元需要被调节

**💡 创新点**

创新点在于两种无训练门控：静态可分离门（筛选能携带概念信息的神经元）和动态后验门（仅在当前激活更符合不良概念时才调节）

**🔧 技术方法**

采用激活调节、AUROC判别、Gaussian后验概率、统计门控及与现有CAST/DSAS的融合

**📊 数据集**

使用RealToxicityPrompts（毒性）和OneSeC（七个概念）数据集

**📈 对比分析**

在Gemma-3 4B和Qwen-3 1.7B上与CAST/DSAS比较，GAPS在毒性降低/概念移除的同时保持或提升语言模型的困惑度与MMLU性能，尤其在Gemma-3上将毒性率从6.52%降至0.48%

**⚠️ 局限性**

局限性包括：后验门假设源/目标概念先验相等；对先验选择的系统性研究缺失；门控仅基于统计信息，可能对极端样本不够鲁棒

---

## 14. GRAND-HC: Graph-Refined Author Name Disambiguation

**arXiv ID:** 2609.01636 | [PDF](https://arxiv.org/pdf/2609.01636v1)

**作者:** Yuanhao Sun `[一作]` (Shanghai Jiao Tong University), Chenghu Zhou `[通讯]` (Institute of Geographic Sciences and Natural Resources Research, Chinese Academy of Sciences)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套端到端的作者姓名消歧义框架 GRAND-HC，用以解决长尾作者分布偏倚和聚类数估计困难问题。

**💡 创新点**

创新点包括：① 用和谐对比学习（Harmony Contrastive Learning）动态重权重样本，缓解富产作者偏倚；② 通过图结构细化与度数加权的图修正距离矩阵（GRDM）进一步防止尾部作者被过度合并；③ 采用论文压缩模块（PCM）以固定长度注意力+Bi‑LSTM的方式高效、鲁棒地估计聚类数。

**🔧 技术方法**

核心技术为异构论文图构建、Transformer+Graph Attention Network 的嵌入学习、和谐对比学习、图修正距离矩阵、论文压缩注意力模块、以及层次聚类（HAC）和 Huber 损失。

**📊 数据集**

在 AMiner‑v2 与 WhoisWho‑v1 两个公开基准上进行实验，涵盖数百个姓名、数十万篇论文。

**📈 对比分析**

与现有 30+ 传统与 LLM 基线相比，GRAND‑HC 在宏观 F1 分数上实现了显著提升（AMiner‑v2 84.86% / WhoisWho‑v1 84.33%），并在聚类数估计误差、推理速度等方面优于对手。

**⚠️ 局限性**

虽然在极端噪声或不完整的图结构下仍可能出现性能下降（需进一步迭代图修正），但整体已基本克服长尾与聚类数估计的两大关键瓶颈。

---

## 15. Private Computation Space: Experience with Trusted Multi-Cluster Federated Learning for Agriculture

**arXiv ID:** 2609.01667 | [PDF](https://arxiv.org/pdf/2609.01667v1)

**作者:** Shuangyu Lei `[一作]` (Cornell University), Hakim Weatherspoon `[通讯]` (Cornell University)

**通讯引用:** 7630 | [OpenAlex ID](https://openalex.org/A5006508602)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

在两项真实农业工作负载中，部署并运营了一个多集群联邦学习系统，利用可信执行环境和差分隐私保证农户数据安全。

**💡 创新点**

创新地将联邦学习、可信聚合、差分隐私与多集群编排和异步聚合相结合，形成一个可在脆弱农村网络下可靠运行的通用隐私保护平台。

**🔧 技术方法**

使用FedAvg/FedBuff异步聚合、Central DP（DP-FTRL）差分隐私、Confidential VM实现的TEE、安全的Kubernetes多集群编排（KubeStellar）以及Raspberry Pi边缘节点等技术。

**📊 数据集**

数据集包括纽约三家农场的1,090张基因改造哨兵植物图像用于叶片分割，以及加州CIMIS气象站共34,107条样本的蒸散预测数据。

**📈 对比分析**

通过与集中式、单站点和本地训练的对比，联邦学习在不泄露原始数据的情况下达到近乎集中式精度；在氮监测中DSC提升22.4%，蒸散预测中R^2提升9.1%；差分隐私引入的延迟低于0.5%，异步FedBuff在不同可用性下也能匹配同步聚合性能。

**⚠️ 局限性**

局限性包括仅假设诚实但好奇的威胁模型、对TEE内存漏洞的依赖、异步聚合在非IID数据下可能产生偏倚、以及仅在两项部署中验证，缺乏对恶意攻击或更大规模场景的评估。

---

## 16. Percolation Dynamics in Optimization : Variance Cascades and Discrete Scale Invariance

**arXiv ID:** 2609.02373 | [PDF](https://arxiv.org/pdf/2609.02373v1)

**作者:** Sai Niranjan Ramachandran `[一作]` (Technical University of Munich), Suvrit Sra `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究SGD及Adam/AdamW训练过程中，参数如何因网络结构对称性而被引导进入稀疏低秩的简化子网络，并用随机梯度流建模为分支凝聚的渗流过程；

**💡 创新点**

首次把梯度下降的拓扑演化映射为分形尺度不变的渗流过程，揭示了离散的块级合并与方差峰值的对应关系，且扩展到自适应优化器和重尾噪声；

**🔧 技术方法**

采用随机微分方程（SGF）、Reeb图与分支凝聚的图论模型、离散尺度不变（DSI）分析、方差尖峰指标、以及对Adam/AdamW的重尾噪声截断处理；

**📊 数据集**

在控制的toy模型、UCI tabular数据集、计算机视觉基准（如FMNIST）以及Transformer在模块算术任务上的grokking实验中验证；

**📈 对比分析**

通过对多条训练轨迹的方差峰值和宏观阶跃参数的统计比较，发现与传统连续相变不同的离散跳跃，DSI级联与模型性能提升同步出现，证明了理论预言；

**⚠️ 局限性**

主要局限在于：对大规模网络宽度的实证检验不足；自适应优化器的泛化到更复杂模型仍需进一步验证；重尾噪声假设与真实训练噪声的匹配度不完全。

---

## 17. Compositional Spectral Prompts for LLM-based Online Time Series Forecasting

**arXiv ID:** 2609.02093 | [PDF](https://arxiv.org/pdf/2609.02093v1)

**作者:** Seungyoon Choi `[一作]` (KAIST), Chanyoung Park `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于大语言模型的在线时间序列预测框架CoSPOT，利用频谱提示和文本描述实现对持续变化分布的快速适应。

**💡 创新点**

创新点在于引入可组合的频谱提示（基于DFT分解的频率基底），让LLM在保持冻结的同时通过少量可训练参数捕捉未知模式；同时结合文本描述提供即时上下文。

**🔧 技术方法**

采用预训练LLM（如Llama-7B）、PatchTST时间序列编码器、对齐模块、离散傅里叶变换、离散小波变换、伪标签回归、几何衰减等技术。

**📊 数据集**

在八个公开时序数据集上评估：ETT系列（ETTh1/2/ETTm1/2）、Weather、Electricity Load、Traffic、Exchange Rate。

**📈 对比分析**

与静态模型、时间序列基础模型、LLM对齐模型以及现有OTSF方法（FSNet、OneNet、DSOF）对比，CoSPOT在延长在线期和跨数据集分布漂移场景下均实现了显著降低的MSE/MAE，且仅需极少参数更新。

**⚠️ 局限性**

局限包括对LLM的依赖导致初始训练成本高、对频谱分解参数（γ、δ）敏感，以及在极高噪声或非周期性数据上性能可能受限。

---

## 18. AGI Maze Prediction Datasets: A Compact Benchmark for Learning World Dynamics with Transformers

**arXiv ID:** 2609.02339 | [PDF](https://arxiv.org/pdf/2609.02339v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 19. RAFT-DVC: Resolution-Aware Machine Learning-Based Digital Volume Correlation

**arXiv ID:** 2609.01876 | [PDF](https://arxiv.org/pdf/2609.01876v1)

**作者:** Zixiang Tong `[一作]` (University of Texas at Austin), Jin Yang `[通讯]` (University of Texas at Austin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并实现了 RAFT‑DVC，一种可调分辨率的机器学习基数字体相关框架，使用三种编码器下采样因子（s=2,4,8）构建不同分辨率解算器，并通过匹配设计评估其在纹理与位移范围内的性能。

**💡 创新点**

系统研究了内部特征网格分辨率对三维 DVC 位移精度的影响，给出经验尺度 EPE_feature≈0.017、EPE_raw≈0.017·s；提出坐标一致性检验和非立方冲激测试验证三维相关采样正确性，并提供切块推断实现以支持大体积处理。

**🔧 技术方法**

采用基于 RAFT 的全对全场相关网络、可变下采样的特征编码器、可分离 ConvGRU 迭代更新、坐标一致的三维采样器、合成粒子体积生成、零位移噪声基准、频率扫描空间分辨率测试以及切块推断与重叠混合等技术。

**📊 数据集**

使用合成粒子标注体积（不同粒径、密度、位移范围）以及真实的共聚焦显微镜压痕实验体积和微泡体积 CT 样本进行训练、验证与跨纹理泛化评估。

**📈 对比分析**

在七个基准场景（S1–S7）中与经典局部、全局与 ALDVC 方法以及重新训练的 VolRAFT（纠正与原始采样器）进行 EPE、空间分辨率、零变形噪声、计算时间等指标对比；结果显示在细纹理、小位移下 RAFT‑DVC 与经典方法相当，在粗纹理、大位移下 RAFT‑DVC 明显优势；切块推断在 512³ 体积上实现 18 s 内完成 1.1 亿点输出。

**⚠️ 局限性**

仅在与训练分布相近的纹理和位移范围内表现良好；内部特征网格分辨率与位移可达范围耦合，需针对不同实验预先选择或再训练解算器；三维相关采样的坐标一致性易被忽视；内存需求随特征网格尺寸指数增长，限制直接推断大体积；零变形噪声与空间分辨率仍高于经典方法；未评估实时性与多模态泛化的鲁棒性。

---

## 20. SonicCaps: Large-Scale Diverse and Fine-Grained Captioning for Improved Audio-Retrieval

**arXiv ID:** 2609.02343 | [PDF](https://arxiv.org/pdf/2609.02343v1)

**作者:** Zineb Lahrichi `[一作]` (Sony CTC), Geoffroy Peeters `[通讯]` (LTCI, Telecom Paris, Institut polytechnique de Paris)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个大型多样化音频描述数据集（SonicCaps），并利用该数据集训练 CLAP 模型以提升音频‑文本检索和零样本分类性能。

**💡 创新点**

①通过多阶段的多模态 LLM（Qwen3‑Omni）递归生成高保真、多样化、细粒度的音频描述；②采用结构化提示工程和 few‑shot 生成，实现每个音频约 24 条不同风格/粒度的描述；③提出与人类评价一致的主观评估框架，验证多样化描述对模型性能的实际贡献。

**🔧 技术方法**

多模态大型语言模型 Qwen3‑Omni、prompt engineering、few‑shot 生成、RoBERTa‑Large 文本编码器、PaSST 音频编码器、对比学习（CLAP）以及后处理和过滤步骤。

**📊 数据集**

FreeSound、AudioCaps、BBC Sound Effects、AudioSet Strongly‑Labeled、WavCaps 等来源，共约 700k 条音频和 15M 条文本描述；实验还使用 ESC‑50、FoleyBench、AudioCaps 验证集和内部商业数据集进行评估。

**📈 对比分析**

与 LAION‑CLAP、AudioCaps+WavCaps 以及多种自身 ablation 模型进行对比。多样化描述的 CLAP 在 Text‑to‑Audio 和 Audio‑to‑Text 的 R@5/R@10 以及零样本分类 R@1/R@5 上均显著优于基线，特别是在跨域测试集 FoleyBench 上提升超过 10%。此外，训练得到的 _MOS 模型的 CLAP 分数与人类 MOS 评分的 Spearman 相关系数达 0.32，明显高于 LAION‑CLAP 的 -0.07。

**⚠️ 局限性**

①仅提升描述质量对检索的提升有限，主要依赖于多样性；②LLM 生成的文本可能包含偏见或虚假信息；③数据集中仍存在词汇和标签冗余；④模型训练仅使用 10 秒段，可能忽略长时序信息；⑤部分数据集受版权限制，无法公开完整音频。

---

## 21. HyGRAIL: Cost-Aware and Evidence-Grounded Scientific Hypothesis Discovery over Knowledge Graphs

**arXiv ID:** 2609.02056 | [PDF](https://arxiv.org/pdf/2609.02056v1)

**作者:** Yihang Sun `[一作]` (University of Illinois Urbana-Champaign), Jiaxuan You `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HyGRAIL 框架，利用异构图神经网络作为轻量级筛选器，结合检索与自然化的图证据以及 LLM 评审来验证科学假设。

**💡 创新点**

创新点在于：① 通过验证校准的 GNN 触发区间实现成本感知筛选；② 设计假设导向的图证据检索与模板/LLM 自然化流程；③ 让 LLM 仅在不确定区间依据证据做判定。

**🔧 技术方法**

采用异构 GNN（HeteroConv、HGT、R-GCN）进行分数预测，检索节点关联与多跳路径，模板或 LLM 进行证据自然化，并使用 Qwen、Ministral 等 LLM 进行评审；同时利用阈值校准和置信度阈值。

**📊 数据集**

在材料科学知识图谱 MatKG 的 3000 节点子图上进行实验评估。

**📈 对比分析**

与 TSH、KG‑FM 等基线对比，HyGRAIL 在 F1 上提升至 0.429，超过基线 0.242；同时平均降低 LLM 调用率 54%。

**⚠️ 局限性**

局限包括：采用闭世界评估导致未观测边不一定为错误；需要手工设定假设类型和证据边集合；LLM 评审受模型、提示与校准影响。

---

## 22. TriSAR: Task Coordination and Collision Avoidance for Aerial Robot Teams in Disaster Response

**arXiv ID:** 2609.01731 | [PDF](https://arxiv.org/pdf/2609.01731v1)

**作者:** Aditya Anil Kapile `[一作]` (Nottingham Trent University), Isibor Kennedy Ihianle `[通讯]` (Nottingham Trent University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在仿真城市灾害环境中，评估并比较了5架无人机使用遗传算法与贪心分配相结合的PSO轨迹控制与可开启/关闭的反弹式碰撞规避策略。

**💡 创新点**

①将遗传算法与PSO轨迹控制分离为可切换的两层协调模块，支持因子消融；②在相同仿真环境下系统性量化了反弹式碰撞规避的安全效益与任务分配的效率差异，揭示二者的交互效应。

**🔧 技术方法**

遗传算法（排列编码、PMX交叉、随机交换变异）用于任务分配；PSO（局部速度搜索）用于轨迹控制；基于位置的人工势场反弹式碰撞规避；Gazebo物理仿真与Python UDP接口。

**📊 数据集**

通过Gazebo生成的合成灾害城市场景，包含500m×500m城市模型、5架同类四旋翼、8个目标（4屋顶+4地面），共30次随机种子实验，随机扰动地面目标坐标±2m。

**📈 对比分析**

对四种配置（GA+repulsion、贪心+repulsion、GA+不repulsion、贪心+不repulsion）分别跑30次，使用Welch t检验比较连续效率指标（执行步数、路径长度、能耗），使用Mann‑Whitney检验比较碰撞阈值违规步数。结果显示：开启反弹时两种分配无显著效率差异，违规率为0；关闭反弹时GA显著优于贪心（g值0.92–1.76）。反弹开启显著降低违规步数，效果在贪心下更大。

**⚠️ 局限性**

仅在单一小规模对称场景下验证；未与外部基线（ACO、市场化分配、RVOb）对比；计算时间仅在小样本测量；纯仿真，未考虑感知与通信噪声；碰撞阈值为距离阈值，非真实接触。

---

## 23. Stored Is Not Supported: Typed Provenance and Assertion Guardrails for Persistent AI Agents

**arXiv ID:** 2609.02127 | [PDF](https://arxiv.org/pdf/2609.02127v1)

**作者:** Jun He `[一作]` (OpenKedge.io), Deying Yu `[通讯]` (OpenKedge.io)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个用于持久化 AI 代理的基于类型化来源图和断言中介的制度化框架，确保代理对自身、用户和关系的 autobiographical 声明严格遵循证据、时效性和披露权限；

**💡 创新点**

创新点在于将存储与支持分离，构造类型化 provenance 图、类型化解析器与生成‑验证‑修订断言中介；通过判定证据依赖、源独立性和授权投影，证明了断言 boundedness 的条件性合约；

**🔧 技术方法**

采用了类型化 provenance DAG、依赖闭包解析器、受限投影与多级断言中介；在实现层面使用 Merkle DAG、事务性状态存储、加密签名与视图脱分类；

**📊 数据集**

使用了 24 条手工编写的对抗案例（19 个不安全场景 + 5 个正常控制），未使用公开大规模语料库；

**📈 对比分析**

对比了三种释放策略（flat/prior、source‑tag、typed mediation），结果显示 typed mediation 在 19 个不安全机会中 0/19 逃逸，覆盖率 100%，而 flat/prior 和 source‑tag 分别出现 100% 和 94.7% 的不安全释放；

**⚠️ 局限性**

局限性包括：缺乏自然语言提取与分类的实现；仅在离线可验证的结构化案例上测试；未评估端到端系统的性能和可扩展性；并且仍需在真实检索和生成环境中验证语义正确性和多方隐私隔离。

---

## 24. Real-Time Dynamics-Based Torque-Sampling MPPI for Compliant and Force Aware Manipulation

**arXiv ID:** 2609.02020 | [PDF](https://arxiv.org/pdf/2609.02020v1)

**作者:** Euncheol Im `[一作]` (Korea Institute of Science and Technology), Yisoo Lee `[通讯]` (Korea Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

开发了一种基于MPPI的刚体动力学约束下的关节力采样控制框架，实现了实时高频任务空间运动与力控制。

**💡 创新点**

创新点：1）首次在MPPI中直接采样关节力并显式整合完整刚体动力学；2）利用GPU并行化实现高频166Hz更新和0.18s预测窗口；3）结合力、运动、碰撞等多重成本实现安全、顺应的物理交互。

**🔧 技术方法**

技术：MPPI采样控制、GPU并行前向动力学、关节力采样、动态一致逆雅可比、力成本、碰撞成本、零阶保持、Franka Control Interface、ZeroMQ、Intel RealSense L515。

**📊 数据集**

数据集：无公开数据集，所有实验在真实7-DoF Franka Research 3机械臂上完成。

**📈 对比分析**

比较方法：未给出基准；通过实验验证控制精度（位姿误差约0.0137m、0.0208rad，力误差1.8N）和对扰动的顺应性，说明高频率与长预测窗口提升了动态性能。

**⚠️ 局限性**

局限：仅考虑机械臂本体动力学，未建模外部物体动力学；缺乏显式力反馈；在曲面接触中误差较大；缺少与其他MPC或DDP方法的量化对比。

---

## 25. Public-Sharing Labels and Verbatim Field Egress in an MCP-to-A2A Agent Configuration: A Controlled Multi-Model Study

**arXiv ID:** 2609.01693 | [PDF](https://arxiv.org/pdf/2609.01693v1)

**作者:** Arpan Kumar Mahapatra `[一作]` `[通讯]`, Arpan Kumar Mahapatra

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在本地MCP和A2A协议组合中，使用可执行的实验 harness 对记录的敏感度标签（机密、无标签、公开）对模型转发时逐字泄露率进行测量。

**💡 创新点**

首次在单一协议组合配置下，采用三臂匹配设计和可重复的实验流程，量化公开标签与机密标签对数据泄露的关联，并提供完整冻结的代码与数据 artifact。

**🔧 技术方法**

使用 MCP Python SDK 与 A2A HTTP+JSON 本地 fixture，OpenAI GPT‑5.6 四个实例和 Anthropic Claude‑Sonnet‑5，统一的决策适配器、精确子串检测器以及 deterministic 事件追踪与 judge‑free scoring。

**📊 数据集**

基于 10 个人工合成的支持角色记录场景，包含六个敏感字段；所有字段在三臂间保持字节相同，且未出现在模型提示、工具描述或策略中。

**📈 对比分析**

采用预设的三臂匹配设计，每场景四次重复，共 480 次试验；统计每个模型在机密、无标签、公开臂的 egress 率并计算场景级差值；结果显示公开臂在 Claude 及部分 GPT 模型中显著提升 egress，机密标签无显著保护作用；未进行统计显著性检验，仅给出描述性指标。

**⚠️ 局限性**

实验仅限单一本地 MCP‑A2A 组合，未检验非逐字泄露；使用固定模型版本与单一策略；多模型间比较受 floor 限制；未分离公开标签与“OK TO SHARE”的独立效应；未提供正向对照，结果仅为描述性关联。

---

## 26. Convergence Theory of Knowledge Distillation in Asynchronous P2P Gossip Learning Network

**arXiv ID:** 2609.01952 | [PDF](https://arxiv.org/pdf/2609.01952v1)

**作者:** Lucas Qingyang Fang `[一作]` (University of California Santa Cruz), Katia Obraczka `[通讯]` (University of California Santa Cruz)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了异步随机边P2P知识蒸馏（KD）算法，并给出了其在无服务器异构网络中的收敛理论，证明时间平均函数平稳性与函数空间一致性误差按 𝒪(1/(ηT)) 下降，并给出终端邻域误差为 𝒪(η)+𝒪(B_f²)+𝒪(ζ_f²)。

**💡 创新点**

创新点包括：① 将KD通信视为对函数空间的几何收缩操作，从参数空间转移到预测空间；② 在异构网络下建立了参数更新与函数空间梯度之间的桥梁 A4 与对齐假设 A10；③ 推导了异步随机边P2P KD 的完整收敛上界，并验证了其收敛速率与终端误差的理论预期。

**🔧 技术方法**

主要技术包括：函数空间（Hilbert 空间 L²(μ,ℝ^C)）分析、KD 对偶梯度与几何收缩、随机边异步事件模型、EMA 伪教师、Lyapunov 能量函数、以及基于有限参考测度的实验评估。

**📊 数据集**

使用了 CIFAR‑10 数据集作为训练与评估数据，采用 Dirichlet 方式划分私有数据，实验网络为 20 个模型，包含 ResNet、MobileNetV2 与 ShuffleNetV2 等多种宽度与架构。

**📈 对比分析**

与基准方法的比较：在同一实验设置下与参数平均的 Decentralized SGD（仅在同质架构下可用）做对比；KD 在同质、宽度异构及混合族网络上均能将函数空间不一致度压缩 40–61 倍，且最终的集成准确率在 0.535–0.729 之间，明显优于仅训练模型；实验还通过步长四点扫描验证了 1/(ηT) 递减与终端邻域折衷。

**⚠️ 局限性**

局限性包括：① 证明依赖于稳定核、有限滞后、A4 与 A10 假设，对实际特征学习过程的自动满足性不保证；② 终端邻域误差未分离 𝒪(η)、B_f² 与 ζ_f² 的具体贡献；③ 只在有限参考支持上评估，缺乏对可达类探测和真实任务异质性的更细粒度验证。

---

## 27. Does Playing it Safe Count as Faithfulness? Reassessing LVLM Hallucination Mitigation Methods

**arXiv ID:** 2609.01888 | [PDF](https://arxiv.org/pdf/2609.01888v1)

**作者:** Mehrdad Fazli `[一作]` (George Mason University), Ziwei Zhu `[通讯]` (George Mason University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了六种推理时幻觉缓解方法在三种7B规模视觉语言模型上的表现，并将结果分别对比了三个不同的幻觉、信息量和多模态能力基准。

**💡 创新点**

创新点在于构建了一个 54 维诊断评估矩阵，将幻觉降低、信息保留与更广泛的多模态能力三者分离开来，揭示了幻觉率下降往往伴随信息量下降以及幻觉收益不易迁移至复杂推理任务的“风险抑制”与“可信度-能力鸿沟”。

**🔧 技术方法**

使用的技术包括对比解码（VCD、M3ID）、注意力校准（AGLA、CAAC）与隐藏状态修改（CEI、AFTER）的推理时调控方法，并结合 CHAIRS、CHAIRi、Cover 等幻觉与信息量指标以及 vision‑indispensable 基准的六大多模态能力评测。

**📊 数据集**

所用数据集包括 MS‑COCO 图像字幕集（用于 CHAIR 指标）、HallusionBench（用于 CHAIRS 与 Cover）以及 MMS（vision‑indispensable）基准的多模态问答与推理样本。

**📈 对比分析**

通过与无干预基线比较，发现多数方法在降低幻觉率的同时降低了对象召回或视觉覆盖率；在更广泛的多模态任务上，绝大多数方法甚至导致性能下滑，只有少数配置在特定模型与任务上出现轻微提升，整体上并未显著提升跨任务的整体能力。

**⚠️ 局限性**

局限性包括仅覆盖三种基准与三种7B规模模型、仅评估推理时调控策略而未考虑训练或检索增强等方法、对超参未做系统调优、以及对更大规模或不同架构模型的泛化性缺乏验证。

---

## 28. GeoSPRINT: Geometric Redundancy-Aware Step Pruning for Inference in Diffusion Trajectories

**arXiv ID:** 2609.02160 | [PDF](https://arxiv.org/pdf/2609.02160v1)

**作者:** Arpita Joshi `[一作]` `[通讯]` (Scripps Research Institute), Arpita Joshi (Scripps Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种训练无关的GeoSPRINT框架，用几何信息构造非均匀采样计划，以减少扩散模型推理时的神经网络评估次数。

**💡 创新点**

创新点在于将高维轨迹的超平面冗余检测与弯曲密度融合，并引入轨迹投影分数α_traj作为全局几何度量，实现对采样步长的全局优化。

**🔧 技术方法**

主要技术包括高维超平面冗余测试（QR分解实现）、曲率密度估计、log‑SNR与曲率的混合调度构造、以及残差方差投影分数的计算。

**📊 数据集**

实验涵盖CIFAR‑10（32×32）、LSUN Church（256×256）和Stable Diffusion v1.5（512×512 latent）三个数据集。

**📈 对比分析**

与DDIM、DPM‑Solver++等基线对比，GeoSPRINT在所有预算下均显著降低FID；在CIFAR‑10上至少提升0.7–1.1 FID，甚至在NFE≥30时优于第二阶DPM‑Solver++，在SD v1.5上提升达1.93 FID。

**⚠️ 局限性**

局限包括需要离线生成大量完整轨迹以构建通用调度、混合系数β需经验调优、在单样本生成时额外开销较大，以及仅适用于已训练好的模型，未提升单步求解器精度。

---

## 29. MASkills: Continual Skills Optimization for Multi-Agent LLM Systems

**arXiv ID:** 2609.02094 | [PDF](https://arxiv.org/pdf/2609.02094v1)

**作者:** Huaiyuan Yao `[一作]` (Arizona State University), Hua Wei `[通讯]` (Arizona State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MASkills框架，利用可重用技能库实现多代理LLM的持续学习和优化；

**💡 创新点**

创新在于把策略改进映射到语言技能空间，并结合技能级信用分配、层次聚合、动量平滑以及细化/诱导/合并/剪枝四种演化操作和验证回滚机制；

**🔧 技术方法**

采用语言空间的策略梯度类框架（LLM_Critic、LLM_Grad、LLM_Agg、LLM_Edit等），基于GPT-5.1/ GPT-4o-mini 进行信用评估与技能编辑，并在Dec‑POMDP多代理设置中实现；

**📊 数据集**

实验使用HotpotQA（多跳推理）、LoCoMo（长时记忆对话）和GAIA（通用AI助手基准）三大数据集；

**📈 对比分析**

通过与CoT、Self‑Refine、ReAct、Vanilla ReAct、R1‑Searcher等多种基线比较，MASkills在HotpotQA F1、LoCoMo F1/BLEU和GAIA成功率等指标上均实现最高或接近最高的提升；

**⚠️ 局限性**

局限在于仅验证合作场景，动态/对抗/开放世界环境未覆盖；技能库随成长可能导致检索与协调效率下降，且对大规模部署的可扩展性尚未充分验证。

---

## 30. Risk-Sensitive Reward Composition for Conditional GFlowNets

**arXiv ID:** 2609.01929 | [PDF](https://arxiv.org/pdf/2609.01929v1)

**作者:** Carine Ribeiro dos Santos `[一作]` (Universidade Federal do Rio de Janeiro), Ina Pöhner `[通讯]` (University of Eastern Finland)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于分布式鲁棒 CVaR 的奖励组合方法，用于在具有多构象的靶点下通过 GFlowNet 生成多样化药物候选分子。

**💡 创新点**

创新点在于将置信区间半径 ρ 与尾部水平 β 结合，构造可调的风险敏感奖励，并在一个网络中同时学习覆盖整个参数空间的目标分布。

**🔧 技术方法**

采用 GFlowNet、Feature‑wise Linear Modulation（FiLM）、Trajectory Balance 训练算法，以及闭式求解的 DRO‑CVaR 评价层。

**📊 数据集**

使用可枚举的合成世界作为实验基准，包含 2 维 32 字母格子与 8 维 4 字母序列两种规模，模拟 DNA 结合特异性与分子结构评分。

**📈 对比分析**

与 exact‑KL 先验网络及多种探索策略比较，实验表明单个条件网络在 3× 采样阈值内完成学习，且在稀疏条件下投射式探索可实现接近 100% 的模式覆盖。

**⚠️ 局限性**

局限在于对尖锐权重分布和评分噪声的鲁棒性不足，且在极端稀疏或高度不确定的构象分布下仍可能出现无法访问的状态陷阱。

---

## 31. On Top-Down and Local Lower Bounds for $\mathrm{AC^0}$ Circuits

**arXiv ID:** 2609.01759 | [PDF](https://arxiv.org/pdf/2609.01759v1)

**作者:** Gülce Kardeş `[一作]` (University of Colorado Boulder), Benjamin Rossman `[通讯]` (Duke University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了 Chopping Game 这一新的上行式下界框架，并通过它推导了 AC⁰ 线路深度-阈值下界、k‑局部版本、以及图论版本的距离‑k 问题的下界与猜想。

**💡 创新点**

创新点在于：① 将传统的自底向上方法（随机限制、开关引理、低次数多项式逼近）转化为自顶向下的游戏化思路；② 设计了 k‑局部和距离‑k 的 Chopping Game 以及对应的 KW Game，形成新的下界范式；③ 在图论框架下提出高 girth 正则图的距离‑k 关系的下界猜想，为研究更通用的局部下界方法奠定基础。

**🔧 技术方法**

主要技术包括：多项式逼近（Razborov–Smolensky）、切分游戏与 Chopping Game 的归纳证明、线性/仿射策略的量化下界、阳木/向量空间方法、sunflower 结构与集族上限、以及图论中的路径/染色技术。

**📊 数据集**

无实验数据；研究完全基于理论证明和组合构造。

**📈 对比分析**

本工作没有与现有算法或实验实现进行直接比较；所有结果均为理论下界，未给出实际性能指标。

**⚠️ 局限性**

局限性：\n- 只得到对特定游戏（如 k‑局部、距离‑k）的下界，未能证明更一般的 AC⁰ 下界（尤其是超多项式）；\n- 对于非仿射/线性策略的通用 Chopping Game 下界仍未得到，需开发新技术；\n- 在图论猜想中，仅给出高 girth 正则图的上界，缺乏对应下界的证明；\n- 研究中假设了很多理想化条件（如足够大 n、k 为特定幂），对实际可应用性有限。

---

## 32. Evidence for Shared Routing Geometry and Dynamics in Sparse Mixture-of-Experts

**arXiv ID:** 2609.02404 | [PDF](https://arxiv.org/pdf/2609.02404v1)

**作者:** Kirill Labzin `[一作]` (Central University), Artem Gorokhov `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对稀疏 MoE 模型中每层路由器的控制状态进行几何对齐，揭示其跨层共享动态，并验证该共享结构在功能上的有效性。

**💡 创新点**

发现路由控制子空间具有正交等价性，提出使用广义正交 Procrustes 对齐不同层的控制状态，并证明仅用单个线性变换即可捕获大部分跨层动态；同时区分路由特异性与残差平滑性。

**🔧 技术方法**

路由控制分解、广义正交 Procrustes 分析、线性动态建模、匹配秩读取比较、因果运输实验以及 ΔNLL 评估。

**📊 数据集**

WikiText‑2 原始文本数据集，包含多份训练/验证/测试拆分。

**📈 对比分析**

与原始坐标、PCA 基、随机正交变换等对齐基线对比；共享线性变换在 R² 上达 79–90% 的层级特定模型性能；匹配秩读取表明残差 PCA 更易预测但路由选择准确度低；因果运输实验显示学习动态比保持不变（persistence）在更长时间步更优，ΔNLL 降低 6–15%。

**⚠️ 局限性**

研究仅为经验性，使用单一文本语料，适用性受模型架构和路由器设计影响；共享动态并非完全通用；运输步骤在多层跳过时误差快速累积；未给出实际推理速度提升。

---

## 33. Who Drives the Probability Game of VLMs? A Temporal Causal Drive Evaluation Framework

**arXiv ID:** 2609.02000 | [PDF](https://arxiv.org/pdf/2609.02000v1)

**作者:** Shuyao Xiao `[一作]` (Beijing Normal University), Chaoyong Jiang `[通讯]` (Beijing Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于结构因果模型的时序评估框架，用于在无参考答案的情况下动态追踪视觉输入、问题文本和生成前缀对多模态语言模型（VLM）自回归生成的因果驱动力。

**💡 创新点**

创新点包括：①将自回归生成视为因果过程，使用 SCM 进行干预与后门调整；②定义三种步骤级因果驱动指标（PCD、QCD、VCD），可捕捉不同信息源在生成过程中的动态作用；③在无参考答案的情形下实现过程级评估，并与传统统计关联度（PMI）以及外部验证（VLMBias）进行对比，验证其因果解释性。

**🔧 技术方法**

核心技术包括：结构因果模型（SCM）、干预操作（do‑operator）、后门调整、交叉验证随机干预、PMI 对比、k‑means 聚类、源移除校验，以及基于 AUROC/AUPRC 的二分类评估。

**📊 数据集**

使用的主要数据集：MAVIS、LLaVA‑Video‑178K、MiraData（视频 QA）、InternVL2‑8B（跨模型验证）、VLMBias（视觉依赖性验证）。

**📈 对比分析**

评估方法：与观测 PMI（QCD/PCD）对比，发现 QCD 与 PMI 的恢复误差下降 34.8%，PCD 降低 47.1%；在 VLMBias 上，Prefix‑Visual imbalance（S_PV）AUROC 为 0.767，AUPRC 为 0.873，进一步证明了因果驱动在区分 prior‑driven 与 visually‑grounded 生成上的有效性。与传统指标（BLEU、BERT‑F1 等）相关性弱，显示因果驱动提供了互补的过程级信息。

**⚠️ 局限性**

局限性：①因果解释受所假设的 SCM 结构与已观测变量的完整性约束；②在评估时模型参数固定，未考虑动态参数更新；③缺乏对链式思维（CoT）等长篇推理过程的验证；④仅在 Qwen3‑VL‑8B、InternVL2‑8B 等特定规模模型上实验，结果可能不具普适性。

---

## 34. MemeCULT-1K: Benchmarking South Asian Cultural Context and Humor Understanding of Multimodal Models

**arXiv ID:** 2609.01772 | [PDF](https://arxiv.org/pdf/2609.01772v1)

**作者:** Tawsif Tashwar Dipto `[一作]` (Islamic University of Technology), Sabbir Ahmed `[通讯]` (Islamic University of Technology)

**通讯引用:** 924 | [OpenAlex ID](https://openalex.org/A5101726099)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个多语言（孟加拉语、印地语、英语）South Asian memes 1000 条的 MemeCULT-1K 数据集，并为每个 meme 提供了文化背景说明和三条英文解释。

**💡 创新点**

创新点在于引入上下文感知评估，证明即使极简文化提示也能显著提升 VLM 对 meme 的解释质量，且系统化地对比了开闭源模型的文化推理瓶颈。

**🔧 技术方法**

使用了多模态评估技术：SBERT、BERTScore、BLEURT 等语义相似度指标、LLM-as-a-Judge 评分和人工评价，并在不同语言与上下文条件下对 13 种 VLM 进行实验。

**📊 数据集**

使用 MemeCULT-1K 数据集（1000 条主样本 + 54 条孟加拉方言样本）。

**📈 对比分析**

对比方法是 meme-only 与 context-aware 两个设置，在自动指标和评估者评分上都显示出明显提升；open-source 模型在文化知识缺失上更明显，closed-source 在实体识别误差上占主导。

**⚠️ 局限性**

局限性包括只覆盖 South Asian 三种语言，样本规模有限，文化幽默的主观性与时效性导致解释的可迁移性受限，且简短背景说明对深层文化与语言细节的补救有限。

---

## 35. When Can a Machine Trust a Statute? A Survival Certificate for Machine-Extracted Legal Logic

**arXiv ID:** 2609.01741 | [PDF](https://arxiv.org/pdf/2609.01741v1)

**作者:** Surya Saka `[一作]` `[通讯]` (JudicialMind), Surya Saka (JudicialMind)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了一个针对机器提取的法令属性上下文的 Duquenne–Guigues 归纳推理基底的生存证书，利用两种独立提取器之间的属性级别不一致率进行校准，并在 Monte Carlo 采样下评估每条归纳规则的生存概率，随后以 Wilson 下界和 Benjamini–Hochberg 调整为标准对规则进行认证，并附带最小反例和证明跨度以实现可审计性。

**💡 创新点**

创新点包括：①基于真实提取误差的被动生存证书（Wilson 下界阈值 0.95）；②按属性级别的误差校准与交互作用分析；③提供可审计的最小反例与跨度信息；④系统的负控制验证误差结构对结果的决定性影响；⑤跨司法管辖区（密苏里 vs 印度）验证其可迁移性。

**🔧 技术方法**

技术手段主要包括：形式概念分析与 Duquenne–Guigues 基底的 NextClosure 计算；按属性的误差率估计与 Monte Carlo 采样（Method A）；Wilson 下界与 Benjamini–Hochberg 多重检验；最小反例与最可能反例生成；负控制与两因素实验设计；以及基底编辑距离的计算。

**📊 数据集**

使用的数据集为密苏里修订法典（29,365 节，458 章）与 IndiaCode（15 条中央法案，502 节），并对密苏里法典进行了 304 章与全 458 章的规模评估以及跨司法管辖区的复制实验。

**📈 对比分析**

通过在保留集（不含设计族）上执行 1,000 次 Monte Carlo 试验，计算每条规则的生存概率，并以 Wilson 下界 0.95 进行认证；结果显示全局误差模型下平均认证率仅为 0.038，95% 的章节低于阈值，而按章节校准后提升至 0.128；印度数据集完全未通过认证，证明跨域迁移受限。

**⚠️ 局限性**

局限性包括：证书仅对测得的提取器不一致性有效，未验证提取器的法律正确性；全局误差模型对章节异质性表现不足；跨司法域迁移失败；假设属性误差独立；核心可认证规则数量有限；以及缺乏第三方黄金标准提取器验证。

---

## 36. Efficient GUI Agents: A Systems Survey of Observation, Memory, Action, and Runtime Optimization

**arXiv ID:** 2609.02309 | [PDF](https://arxiv.org/pdf/2609.02309v1)

**作者:** Bizhe Bai `[一作]` (Fudan University), Tao Chen `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述了GUI agent在效率方面的研究进展，提出以端到端系统视角为基础的效率分类（观察效率、记忆效率、动作效率、规划/系统效率），并总结了各类技术手段、关键机制和出现的开销。

**💡 创新点**

创新点在于：①从系统整体视角统一归纳效率问题，打破传统只关注成功率的局限；②提出四大效率轴并结合文献追溯，形成跨技术、跨平台的系统化框架；③识别了当前研究中的共性思路（如选择性阅读、可回收记忆、可验证控制、混合运行时）与未解决的开放挑战。

**🔧 技术方法**

使用的方法主要是文献检索与引用链追溯、基于多维度效率指标（latency、token、memory、step等）的定量汇总，以及对比分析各技术方案对这些指标的影响。

**📊 数据集**

综述涵盖的benchmark与数据集包括：Mind2Web、WebArena、VisualWebArena、BrowserGym、AndroidWorld、OSWorld、Windows Agent Arena、MMBench-GUI、OSWorld-Human 等，但并未自行训练或评测模型，而是引用这些平台上的已有实验结果。

**📈 对比分析**

比较方法：引用原论文给出的指标（如token减少比例、步骤缩减、latency降低、MFLOPs减少等），对同一维度的不同技术做横向对比；但因各论文使用的评测环境、任务设定和指标定义不统一，直接跨论文比较仍有难度。总体而言，综述指出各技术在观察、记忆、动作和规划层面可实现数十%甚至百%的资源节省，但缺乏统一的基准报告。

**⚠️ 局限性**

limitations: ①缺乏统一的效率度量体系与跨平台基准；②综述依赖已有论文的非统一指标，导致难以直接比较；③未覆盖真实部署场景下的隐私、网络延迟等成本；④未系统评估各方法在不同任务、不同硬件条件下的通用性。

---

## 37. Test-Time Logit Prompting for Source-Free Missing Modality Adaptation

**arXiv ID:** 2609.02039 | [PDF](https://arxiv.org/pdf/2609.02039v1)

**作者:** Taixi Chen `[一作]` (State University of New York at Binghamton), Nancy Guo `[通讯]` (State University of New York at Binghamton)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在不访问源训练数据的情况下，对冻结的视觉‑语言模型进行测试时适配的轻量级方法——Test‑Time Logit Prompting（TLP），能够在缺失模态的条件下直接在预测置信度空间中调整模型决策。

**💡 创新点**

创新点在于：① 将适配工作迁移到 logits 空间，只需少量可学习的 logit prompts；② 设计了基于不确定性自适应的损失和完整模态一致性正则化，兼顾置信度校准与语义一致性；③ 在测试时即刻实现源无关的适配，无需重新训练或访问训练样本。

**🔧 技术方法**

技术方法：冻结 CLIP（ViT‑B/16）视觉‑语言 backbone，利用少量可训练的 class‑level logit prompts；采用 AdamW 进行测试时的梯度优化；结合不确定性调整损失 (L_u) 与完整模态一致性损失 (L_c) 的源无关自监督目标；在不同缺失比例下多步迭代更新。

**📊 数据集**

实验数据集：MM‑IMDb（多标签电影分类），UPMC Food‑101（食物图文分类），Hateful Memes（跨模态仇恨内容检测）。

**📈 对比分析**

与原始未适配的源训练模型基线、以及现有基于 prompt 学习的缺失模态方法（MMP、DCP、SyP）进行对比。实验显示，TLP 在所有三大基准上均提升 5–8% 的关键指标（F1‑Macro、Accuracy、AUROC），且只需数百个可训练参数和 1–5 步的轻量化优化；与传统的训练期方法相比，TLP 兼具更低的存储/计算成本，并能在多种缺失比例下保持稳健性。

**⚠️ 局限性**

局限性：① 目前仅在两模态（图像+文本）的缺失场景下验证，尚未探究多模态（>2）或更复杂缺失模式；② 需要在测试集中保留足够的完整模态样本以作为一致性正则化的锚点，完整样本稀缺时效果可能下降；③ 只针对已冻结的 CLIP 结构验证，适配到其他视觉‑语言框架或更深层参数的可扩展性待进一步研究。

---

## 38. ZETA: A Controlled Study of Zero-Shot Cross-Embodiment VLA Transfer for Tabletop Manipulation

**arXiv ID:** 2609.02546 | [PDF](https://arxiv.org/pdf/2609.02546v1)

**作者:** Mi Yan `[一作]` (Galbot), He Wang `[通讯]` (Galbot)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在可控桌面两指抓手环境下的零样本跨机体迁移，区分严格零样本和预训练暴露零样本，并构建了14个目标机体的基准。

**💡 创新点**

创新点：明确定义两种零样本协议，构建细粒度机体转移分类（外观、抓手、臂、全机体），系统评估状态-动作表征、源机体多样性、辅助协同训练和目标机体暴露对迁移的影响。

**🔧 技术方法**

使用了基于视觉语言动作（VLA）框架，采用固定维度EEF-Delta状态-动作表征，利用多源Franka机体的程序化生成，辅以语言动作预测、子目标预测、盒子预测等辅助任务。

**📊 数据集**

数据集：生成的512个程序化Franka机体的640K轨迹预训练数据，40K轨迹下游任务，真实世界中的7个商业目标机体，每个任务10次试验。

**📈 对比分析**

通过在模拟和真实环境下分别执行100次/10次滚动，比较不同设置下的任务进度（0-100%）。结果显示EEF-Delta表征提升约15%，512源机体比单源提升约18%，辅助协同训练提升约7%，仅5%目标机体预训练可提升13.4%。

**⚠️ 局限性**

限制：仅关注桌面两指抓手，未覆盖移动底盘、灵巧手以及长时序任务；实验设计以可控为主，未优化部署所需的所有系统组件。

---

## 39. Beyond Textual Chain-of-Thought: A Survey on Action-Grounded Reasoning in Autonomous Driving

**arXiv ID:** 2609.01659 | [PDF](https://arxiv.org/pdf/2609.01659v1)

**作者:** Zhengxu Tang `[一作]` (NVIDIA), Pichao Wang `[通讯]` (NVIDIA)

**通讯引用:** 9010 | [OpenAlex ID](https://openalex.org/A5042680345)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对 171 篇关于自动驾驶链式推理（CoT）研究进行了综述，提出了以中间表示为中心的四大类（语言、视觉空间、潜在动态、外部化）及 13 种子类型的分类框架。

**💡 创新点**

创新点在于把关注点从传统文本 CoT 转移到“动作驱动”推理表示，系统化地将中间推理状态与动作输出的因果关联、可解释性与实时性等关键维度进行对比，并揭示了当前研究的开放挑战。

**🔧 技术方法**

采用文献检索、手工标注、归类与指标汇总等方法，结合案例分析和实验结果，构建了完整的研究现状与技术路线图。

**📊 数据集**

使用了多种公开基准与数据集，如 nuScenes、CARLA、HighwayEnv、NAVSIM、Bench2Drive 等，对方法的性能进行量化比较。

**📈 对比分析**

通过在不同子任务（如闭环驾驶、规划、QA 等）上提取代表性指标（如 Driving Score、成功率、平均 L2 距离、碰撞率），对 130 条方法进行归类对比；结果表明，具备动作耦合与感知根植的中间表示往往能在闭环驾驶中获得更高的安全性和成功率，但整体性能仍受限于数据、实时性与可验证性。

**⚠️ 局限性**

限制包括：1) 领域快速演进，新论文可能未被纳入；2) 分类在多模态混合系统中可能过于简化；3) 仅关注中间表示，未覆盖所有基础模型相关工作；4) 综述未进行因果验证或对比实验，无法给出表示优劣的定量结论。

---

## 40. Removing Speech, Keeping Activities: A Privacy Firewall for Acoustic Sensing in Assisted Living

**arXiv ID:** 2609.02376 | [PDF](https://arxiv.org/pdf/2609.02376v1)

**作者:** Pavlos Nicolaou `[一作]` (University of Cyprus), Christos Efstratiou `[通讯]` (University of Kent)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `9cc9baba-5356-466d-81ff-d80028d90279` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一个隐私防火墙管线，使用U‑Net编码器‑解码器模型在录音中去除人声，同时保留环境声音，供日常活动识别使用。

**💡 创新点**

创新点包括：①只使用合成数据训练去声模型，避免了隐私敏感的实录音标注；②将去声过程嵌入常规的VGGish‑SVM活动识别流程；③在多种语音占比下实现0% VAD可检测语音，显著提升活动识别精度。

**🔧 技术方法**

技术方法：U‑Net卷积自编码器（log‑mel频谱输入），多分辨率STFT损失；活动识别使用VGGish特征+SVM；语音检测使用Silero VAD；实验对比包括Facebook Denoiser、SepFormer、ConvTasNet。

**📊 数据集**

使用的数据集：ESC‑50（公共环境音），SINS（真实居家音频），LibriSpeech（语音混合），以及自采的AudioHive录音。

**📈 对比分析**

与公共模型对比，所提去声模型在ESC‑50 100%语音级别下将VAD检测语音降至0%，而Facebook Denoiser为6.55%、SepFormer 36.34%、ConvTasNet 47.21%；在40%语音级别下活动识别精度恢复至85%/85%，接近无语音基准；在SINS和AudioHive上亦实现0% VAD，保持或提升活动识别精度。

**⚠️ 局限性**

局限性：评价仅基于Silero VAD，未验证剩余语音的可懂度或使用ASR/人听测试；实验采用单一70/30拆分，缺乏交叉验证和统计显著性；合成混合为相对幅度比例，未校准实际SNR；未对每类活动的影响做细粒度分析；真实环境中语音比例低，未充分测试高语音场景。

---

## 41. SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition

**arXiv ID:** 2609.02036 | [PDF](https://arxiv.org/pdf/2609.02036v1)

**作者:** Tingyan Wen `[一作]` (Tsinghua University), Songwei Liu `[通讯]` (ByteDance)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出SelfLift，一种自恢复的进度分辨率推理框架，用于在极少步扩散模型中加速生成，同时保持图像质量；

**💡 创新点**

创新点在于① Artifact‑Aware Consistency Lift：利用直接潜在上采样与像素‑VAE重编码的差异，形成自监督残差并针对高风险区域进行局部修正；② On‑Policy Self Recovery：将这一修正机制压缩为轻量级潜在上采样器，并在学生沿着自身轨迹上使用EMA自教师实现密集高分辨率监督，同时保持原有稀疏时间步动力学；

**🔧 技术方法**

采用的技术包括潜在流匹配、潜在/像素双路分辨率跃迁、残差校正权重映射、EMA自教师、SwinIR‑style潜在上采样网络、基于动态一致性的损失以及时间步蒸馏；

**📊 数据集**

实验使用的主要数据集包括FLUX.2‑Klein‑9b和Z‑Image‑Turbo（各自 1024×1024 生成），以及Pick‑a‑Pic、DrawBench和官方评测集进行评估；

**📈 对比分析**

与TeaCache、TaylorSeer、Bottleneck、RaLu、Speed、MrFlow等现有时空加速方法对比，SelfLift‑zero 在 FLUX.2‑Klein‑9b 与 Z‑Image‑Turbo 上分别实现 28× 与 16× 的速度提升，质量保持不变；SelfLift‑rich 在保持同等速度的前提下进一步提升图像奖励和 CLIP 对齐分数，整体速度提升超过 29×（FLUX）和 19×（Z‑Image），且所有质量指标均优于对手；

**⚠️ 局限性**

局限性包括：① 需要模型本身具备多分辨率推理能力；② 对自教师的质量高度依赖，若自教师生成的高分辨率信息不足，恢复效果受限；③ 对非常小的时间步或极端低分辨率前缀的适配性尚未充分验证；

---

## 42. Swin Meets EfficientNet: Lightweight Architectures for GAN-Based Face Forensics

**arXiv ID:** 2609.01749 | [PDF](https://arxiv.org/pdf/2609.01749v1)

**作者:** Sejuti Basu `[一作]` (International Institute of Information Technology), Sahil Sharma `[通讯]` (Ulster University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了基于Swin-Transformer和EfficientNet-B0融合的轻量级深度伪造检测模型，用于识别GAN生成的合成面孔。

**💡 创新点**

创新点在于将局部CNN特征与Shifted-Window自注意力结合，既保持了高检测精度，又显著降低了模型规模和计算成本。

**🔧 技术方法**

主要技术包括Swin-Transformer、EfficientNet-B0、二分类交叉熵损失、Adam优化器以及梯度可视化等。

**📊 数据集**

使用了140K Real and Fake Faces数据集中的70k真实与70k StyleGAN合成面孔的平衡子集进行训练与测试。

**📈 对比分析**

与传统CNN基准（ELA+CNN）和纯Swin模型比较，EfficientNet+Swin混合模型在5,000张测试图像上达到了99%准确率、99.44%召回率，显著优于其他对照组。

**⚠️ 局限性**

主要局限在于仅针对StyleGAN生成的静态图像进行评估，缺乏跨数据集、跨生成器以及视频深度伪造的验证。

---

## 43. Allocate Before You Embed: Adaptive Visual Input Allocation for Video Embeddings

**arXiv ID:** 2609.01778 | [PDF](https://arxiv.org/pdf/2609.01778v1)

**作者:** Song Jin `[一作]` (Renmin University of China), Yong Liu `[通讯]` (Renmin University of China)

**通讯引用:** 21806 | [OpenAlex ID](https://openalex.org/A5100724297)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 AllocEmbed，先对视频帧按视觉预算动态分配不同分辨率，再用现有的 VLM 嵌入模型进行嵌入，解决固定帧数/分辨率导致的时序覆盖不足与空间细节损失的问题；

**💡 创新点**

核心创新在于“allocate‑then‑embed”框架和 Retrieval‑Driven Policy Optimization（RDPO），通过预览帧与任务文本学习帧级分辨率分配策略，并直接以检索排名回馈优化 allocator；

**🔧 技术方法**

采用轻量级的预览特征提取器（SmolVLM）、Beta 分布参数化的尺度分配策略、混合分辨率预处理、冻结的 VLM 嵌入后端以及基于 PPO 的 RDPO 优化；

**📊 数据集**

主要使用 MMEB‑V2 的十个视频问答/检索任务和新构建的 LongRet 长视频检索基准，训练集来自 VLM2Vec‑V2 的视频数据；

**📈 对比分析**

与基线（8 帧统一分辨率）以及 24 帧全分辨率等方法对比，AllocEmbed 在相同视觉输入预算下平均 Hit@1 提升 1.3–9.1 分（依赖任务和模型），并能无额外训练直接迁移到 2B–8B 参数的不同嵌入骨干；

**⚠️ 局限性**

局限性包括需要额外的预览帧计算、对不同硬件预算的鲁棒性需进一步评估、在极低预算或极长视频情形下的性能上限，以及训练过程中对负样本和超参数的敏感性。

---

## 44. Incident Memory: Training-Free Operational Memory through Sequential Pattern Mining and Velocity-Stratified Retrieval

**arXiv ID:** 2609.01616 | [PDF](https://arxiv.org/pdf/2609.01616v1)

**作者:** Adarsh Agrawal `[一作]` (Stony Brook University), Rahul Suresh Babu `[通讯]` (Boston University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个无训练、确定性的“Incident Memory”系统，用于收集、保存并检索运营事故的知识，自动提取可复现的操作顺序（playbook）并检测指标定义冲突。

**💡 创新点**

创新点在于：①速度分层检索（velocity‑stratified retrieval）让事实按衰减速率存活；②指纹（fingerprint）条件化的 PrefixSpan 矩阵化顺序挖掘，精准恢复操作序列；③基于出处的 SQL 检测，解决指标名称冲突，提升知识可审计性。

**🔧 技术方法**

采用了预训练嵌入（如 Amazon Titan Text Embeddings）、PrefixSpan 序列挖掘算法、SQL‑based 归因冲突检测以及基于时间衰减的检索过滤。

**📊 数据集**

主要使用两大数据集：①UCI ITSM 事件日志（141,712 事件、24,918 次事故，23,110 条已解析的操作序列）；②基于真实事故设计的 500 条控制性合成追踪，用于给定 ground‑truth 的精度和熵评估。

**📈 对比分析**

与静态跑书、频率基准、无衰减检索、统一衰减检索、Flat Wiki 等基线以及 Claude Haiku LLM 生成的 playbook 进行对比。结果显示：在受控数据上，PrefixSpan 的有序精度达 99.2%，UCI 现场覆盖率 84.3%，与 LLM 对比有序精度从 0.661 提升到 0.985；速度分层检索 P@5 提升 0.532 以上，冲突检测 F1 提升至 0.876。

**⚠️ 局限性**

局限性包括：仅对重复、可记忆的事故有效；对稀有指纹、多根因或操作流程变更的情况缺乏适应；无法自动检测剧烈的系统漂移；对 LLM 生成的评估受限于单一提示协议，未覆盖所有可能的 agent 设计；尚未验证对 MTTR 的因果影响。

---

## 45. HeadWiseKV: Budgeted Per-Head Cache Residency for Hybrid Long-Context Language Models

**arXiv ID:** 2609.02029 | [PDF](https://arxiv.org/pdf/2609.02029v1)

**作者:** Renjie Xie `[一作]` (Nanjing University of Posts and Telecommunications), Wei Xu `[通讯]` (Southeast University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 HeadWiseKV，一种训练‑free 的静态多级 KV 缓存压缩框架，专为混合语言模型设计；

**💡 创新点**

创新点包括：基于前缀条件的 SeqCalib 校准算法，为每个 KV head 分配多级历史窗口；在运行时使用分组物理缓存实现预先确定的存储；实现了在不重新训练模型的前提下显著降低 KV 存储需求；

**🔧 技术方法**

技术手段包括：受限的操作率失真优化、序列前缀条件校准、分组物理 KV 缓存、静态多级窗口分配与预算选择；

**📊 数据集**

校准使用 WikiText‑103 流；评估使用 RULER、LoCoMo、τ‑bench（Airline、Retail）、LongMemEval‑S 等长上下文和对话基准；

**📈 对比分析**

通过与 Full‑KV、AdaKV、HeadKV‑R2、StreamingLLM、DuoAttention 等基线进行固定模型对比，HeadWiseKV 在保持近 Full‑KV 质量的同时将采样峰值内存降低 8.6%，将可验证上下文长度从 114K 扩展到 161K，解码吞吐率保持 1.017×，前置吞吐率 1.092×；

**⚠️ 局限性**

局限性：对不同模型的保真度不完全统一，低保真压缩（68.95%）会略微影响 RULER；仅针对残差全注意力 head 的 KV 压缩，未覆盖所有缓存路径；需要针对目标工作负载进一步验证。

---

## 46. PaperCompiler: Faithful Paper-to-Code Generation via Repository-Level Specification Compilation

**arXiv ID:** 2609.02272 | [PDF](https://arxiv.org/pdf/2609.02272v1)

**作者:** Yunhao Liu `[一作]` (Nanyang Technological University), Jaehong Yoon `[通讯]` (Nanyang Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 PaperCompiler 的框架，将论文中的实现细节转换为可直接驱动仓库生成的、跨文件一致性的实现规范，从而实现对论文方法的高度忠实重现。

**💡 创新点**

核心创新点在于：①通过“规范编译”（Specification Compilation）把论文证据显式化为实现规范；②采用“权责导向架构合成”与“文件级合同”保证跨文件一致性；③在保持实现细节完整性的同时保留局部工程自由度，避免算法简化和语义漂移。

**🔧 技术方法**

主要技术包括：基于 LLM 的论文地面化（Paper Grounding）、结构化蓝图构建与引用抽取、需求调和与权责映射、文件级合同与约束引导代码生成；使用 o3‑mini 及其高分辨率版本作为生成与评估 LLM，整体流程由三个阶段构成。

**📊 数据集**

使用的公开数据集为 Paper2CodeBench（90 篇来自 ICLR、ICML、NeurIPS 2024 的论文）以及 Paper2Code‑Extra（扩展集），用于评估生成仓库与原始论文/作者仓库的一致性。

**📈 对比分析**

与 ChatDEV、MetaGPT、PaperCoder、AutoP2C、AutoReproduce 等基线对比，PaperCompiler 在 reference‑free、P2C‑Ex 与 reference‑based 三种评测模式下分别提升 4.7%、4.3% 与 13.8% 的平均分（参考实现相比），并在每篇论文上保持更高的胜率，尤其在 reference‑based 模式下显著降低高严重度错误率。

**⚠️ 局限性**

局限性包括：整体 token 使用量显著增加（约 1.71M token，约 $1.88–$7.51），API/模式匹配错误略有上升，仍需进一步完善跨文件接口对齐；方法依赖大模型与专业提示，对低资源场景和非结构化论文内容的适配仍有待提升。

---

## 47. Privacy Washing: Detecting Internal Contradictions in Privacy Policies

**arXiv ID:** 2609.02055 | [PDF](https://arxiv.org/pdf/2609.02055v1)

**作者:** Thomas Brackin `[一作]` `[通讯]` (Varitas), Thomas Brackin (Varitas)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一个四阶段的自动化管道，用以检测隐私政策内部的“隐私洗牌”即承诺与实践之间的悖论。

**💡 创新点**

创新点在于首次将隐私洗牌概念形式化为可执行的判定流程，并通过三模型投票验证提高精度。

**🔧 技术方法**

技术方法包括三模型一致式语句抽取、元数据兼容性过滤、语义相似度预筛、自然语言推理（NLI）打分，以及多模型判定投票。

**📊 数据集**

使用的数据集为2026年收集的123家公司隐私政策（OPPT）和2015年的115家公司（OPP-115）。

**📈 对比分析**

通过与两套不同时间点的语料对比，管道在两组数据中分别发现约10-12%的公司存在确认的内部矛盾，第三方共享类矛盾最为普遍，匹配阈值和模型变更对结果影响有限。

**⚠️ 局限性**

局限性包括未与人工专家基准验证精度、阈值选择和过滤配置导致的召回不确定、模型偏差可能影响判定，以及未覆盖跨文档或代码与政策一致性等更广泛的矛盾。

---

## 48. CRB-Guided Sensing and Resource Allocation for Human Pose Prediction in Integrated Sensing, Communication, and Computation Systems

**arXiv ID:** 2609.01908 | [PDF](https://arxiv.org/pdf/2609.01908v1)

**作者:** Zhonghao Liu `[一作]` (King's College London), Mohammad Shikh-Bahaei `[通讯]` (King's College London)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了CRB引导的毫米波人姿态预测框架ET-Mamba，并设计了联合感知、通信与计算资源的分配优化算法。

**💡 创新点**

创新点在于将CRB误差模型注入点云扰动以提升鲁棒性，采用可调深度的Mamba网络实现资源适配，并通过交替优化联合优化感知SNR、模型深度与计算频率的混合整数非凸问题。

**🔧 技术方法**

使用技术包括CRB理论、Mamba状态空间模型、全局局部融合（GLF）模块、几何距离排序、半正定松弛（SDR）以及非线性最小二乘拟合等。

**📊 数据集**

实验数据集为公开的MM‑Fi多模态数据集（毫米波点云 + 深度摄像头标注）。

**📈 对比分析**

与PointMamba、PointFormer、STPM等基线对比，ET‑Mamba将MPJPE从3.85 cm降至3.24 cm，PA‑MPJPE从2.42 cm降至2.20 cm，同时参数量与FLOPs显著降低；在资源受限场景下实现MPJPE降低约35%。

**⚠️ 局限性**

局限性包括仅验证单人场景、采用离线训练、未考虑实时自适应与多人场景，以及在极低SNR下仍存在预测误差。

---

## 49. SliceBridge: context-consistent repair of corrupted slice intervals in T1-weighted MRI

**arXiv ID:** 2609.01827 | [PDF](https://arxiv.org/pdf/2609.01827v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 50. AVERT: Audio-Verified Adjudication for Spoken Dialogue State Tracking

**arXiv ID:** 2609.01828 | [PDF](https://arxiv.org/pdf/2609.01828v1)

**作者:** Chunggi Lee `[一作]` (Harvard University), Hanspeter Pfister `[通讯]` (Harvard University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AVERT 机制，用音频验证器与跨回合一致性评分对已生成的槽值进行裁决，纠正一致性错误、缺失槽和音频不支持的错误。

**💡 创新点**

创新点：1) 在文本编辑器基础上加入仅验证候选槽值的音频判别器；2) 三种受槽限制的操作（Vote、Add、Swap）精准纠错；3) 仅使用转录文本保持历史上下文不随对话长度增长。

**🔧 技术方法**

技术：基于 WavLM‑Large 编码器 + 轻量级连接器 + OLMo‑2‑1B 语言模型；音频判别器为三层 MLP；候选值收集与跨源计数；词首子串检索门；阈值调优。

**📊 数据集**

数据集：SpokenWOZ（训练/验证/测试），以及用于 ASR 预训练的 Loquacious 与 Fisher（需 LDC 许可证）。

**📈 对比分析**

对比方法：1B 规模基线（R、E、E+AVERT）与其他 1B 语音-LLM 系统。E+AVERT 在严格因果评估下的 JGA 从 38.34 提升至 40.13（+1.79 点），高于单一 1B 语音-LLM 的 39.32，且仅使用两名 1B 解码器，且在长对话中的提升更显著。

**⚠️ 局限性**

局限：仅验证已出现在 ASR 文本中的槽值（仅 21% 的错误可纠正）；未使用软音素验证；验证器与编辑器对大规模模型的泛化仍待验证；子句选择与阈值需依赖有标签的开发集。

---

## 51. Ten Architectures, One Error: Shared Failure Modes in Hyperspectral Classification under Spatially Disjoint Evaluation

**arXiv ID:** 2609.01786 | [PDF](https://arxiv.org/pdf/2609.01786v1)

**作者:** Ehsan Faghih `[一作]` (North Carolina State University), Zahra Saki `[通讯]` (North Carolina State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了基于感受野的空间分离判定标准的泄漏自由评估协议，并在 Salinas 场景上对十种不同架构的模型进行统一评测。

**💡 创新点**

创新点在于：1）构建可重用的审计流程和可验证的分割；2）将模型感受野与空间分离结合，确保训练与测试不泄漏；3）揭示随机像素拆分导致的显著性能偏差，并证明不同模型在同一像素集合上排名会发生大幅变化。

**🔧 技术方法**

采用块划分+缓冲区空间划分、感受野可接受性掩码、Optuna 超参搜索、五个随机种子、多维统计检验（McNemar、Holm、Spearman）以及多核并行训练等技术。

**📊 数据集**

使用了已纠正的 AVIRIS Salinas 场景（512×217 像素，54,129 标注像素，16 类），并在其上构建了固定的训练/验证/测试分割。

**📈 对比分析**

通过在相同的 90 次超参搜索、相同的训练/验证/测试比例、相同的感受野半径 R_eval=2 进行比较，发现随机拆分下的平均 Macro‑F1 为 0.944，泄漏自由拆分下平均为 0.779，平均下降 0.147；模型排名出现最多 5 位的变化，最强模型由第三位降至第八位。

**⚠️ 局限性**

局限性在于：仅针对单一场景和单一冻结分割；训练分区的感受野上限限制了对需要更大上下文的模型的评估；实验结果与 Salinas 场景特定，未必能直接推广到其他数据集。

---

## 52. SkillGLoW: Procedural-Family Skill Consolidation for Self-Improving Agents on Long-Horizon Task Streams

**arXiv ID:** 2609.02217 | [PDF](https://arxiv.org/pdf/2609.02217v1)

**作者:** Ao Yan `[一作]` (National University of Singapore), Joey Tianyi Zhou `[通讯]` (Institute of Advanced Intelligence and Computing (IAIC))

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了SkillGLoW系统，将LLM代理写作的文本技能按程序族进行层次组织，生成全局先验并在执行时结合局部技能进行重用；

**💡 创新点**

创新点在于发现“程序族”是长期任务中缺失的重用单元，提出分层（全局先验+局部重构）组织，并通过执行验证门控安全地接受新先验，避免单一文档或平铺库的局限；

**🔧 技术方法**

使用了语言模型自学习、文本压缩、层次聚类、语义检索、验证器门控、局部化差分抽取等技术；

**📊 数据集**

实验基准包括Terminal-Bench-Pro、SWE-bench、ALFWorld和LiveMathematicianBench四个数据集；

**📈 对比分析**

在12条连续改进运行中，与单文档、平铺库以及其他方法对比，平均提升约17.2分（硬/软），所有12次均取得正向增益，库体积比平铺小3.6倍，对未见ALFWorld任务提升至83.9%；

**⚠️ 局限性**

局限性包括：仅在同一任务类别下验证转移，跨领域或跨模型迁移未测试；程序族划分依赖聚类质量；执行验证门控增加计算成本；

---

## 53. Progressive Pseudo-Label Optimization for Point-Supervised Change Detection

**arXiv ID:** 2609.02171 | [PDF](https://arxiv.org/pdf/2609.02171v1)

**作者:** Hailong Ning `[一作]` (Xi'an University of Posts and Telecommunications), Asoke K. Nandi `[通讯]` (Brunel University of London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一个两阶段点级监督遥感变更检测框架，先利用 SAM2 生成候选掩膜并通过双时序掩膜筛选与 CNN 细化得到软伪标签，再在教师-学生自训练框架中用 EMA 迭代刷新伪标签以提升检测性能。

**💡 创新点**

创新点在于：① 将大模型 SAM2 的单时序分割先验迁移到双时序变更任务；② 通过结合 SAM 置信度、变化一致性和点覆盖度的双时序掩膜筛选提升初始伪标签质量；③ 引入不确定性加权的损失与轻量级 CNN 对伪标签进行边界与结构细化；④ 在自训练阶段采用 EMA 教师模型周期性刷新伪标签，形成闭环优化。

**🔧 技术方法**

主要技术包括：SAM2 分割、双时序掩膜筛选策略、ResNet-18 编码器-解码器的 CNN 伪标签细化、基于置信度的加权 BCE+Dice 损失、教师-学生自训练与指数移动平均（EMA）策略。

**📊 数据集**

使用了公开的遥感变更检测基准数据集 WHU-CD、LEVIR-CD 与 SYSU-CD，均采用 256×256 的图像块进行训练与测试。

**📈 对比分析**

在三大基准上与 14 种最先进的弱监督和 9 种全监督方法比较，本文方法在大多数数据集上实现了最优或接近全监督的 F1、IoU 等指标（如 LEVIR-CD 上 F1 81.25%、IoU 68.80%），并显著优于现有弱监督方案。

**⚠️ 局限性**

局限性包括：对 SAM2 的分割质量高度依赖，且在复杂多变场景下仍易出现伪标签误差；点级标注仍需人工，且当前框架未考虑更通用的弱标注形式（如粗糙掩模或图像级标签）。

---

## 54. SchedBlame: Who Ran While You Waited? Culprit-Attributed CPU Contention for Containers on Stock Kernels

**arXiv ID:** 2609.02052 | [PDF](https://arxiv.org/pdf/2609.02052v1)

**作者:** Hao Li `[一作]` (DiDi Chuxing), Honglei Wang `[通讯]` (DiDi Chuxing)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种基于 eBPF 的连续 CPU 争用归因工具，可在未改动内核的 4.18/5.10 版本上实时识别并量化容器间的 CPU 争用责任。

**💡 创新点**

创新点：① 归因反转——把争用视为被动等待容器对正在运行的 cgroup 消耗的 CPU 时间进行计量；② 稀疏竞争者 + 稠密被测者的 bitmap 使单条记录即可代表整行责备矩阵；③ 状态保持与采样分离，支持无偏逆概率采样并保持低延迟；④ epoch 机制实现无锁、无扫描的配置切换。

**🔧 技术方法**

技术：eBPF 调度钩子（sched_switch、sched_wakeup、migrate、throttle/unthrottle）、per‑CPU 等待 bitmap、epoch‑tagged 缓存、perf ring 批量传输、Go 语言跑步器、逆概率估计、基于 99% 阈值的异常检测。

**📊 数据集**

数据集：在生产 96 核服务器上跟踪 84 个 DiDi 内部容器的工作负载；计划实验使用人工构造的单 CPU victim + hogs 以及多 CPU 多 victim 场景。

**📈 对比分析**

对比与性能：与 Netflix 的 preemption‑tagging、Volpert 等工具相比，提供每秒一次的责备矩阵；开销约为 1% 吞吐量，6% 单核 CPU；采样率可调，p=1 时无采样；在高负载下 per‑CPU perf ring 使用率 <1%，用户空间队列 <15%。

**⚠️ 局限性**

局限性：仅处理 CFS 调度，忽略实时/deadline、共享缓存/内存带宽；CSS‑ID 12 位上限导致 >4095 的 cgroup 被忽略；稀疏竞争者映射与稠密被测者映射限制目标数量；直接阈值导致持续高争用容器不报警；采样与传输损失导致估计偏差；间接限流（祖先限制）误归因；未处理容器层次（子 cgroup）聚合；未提供自动修复或调度器集成。

---

## 55. Seed-Anchored Budget-Bounded Graph Rendering for Question Answering on Industry-Standard Power-Grid Information and Exchange Models

**arXiv ID:** 2609.02011 | [PDF](https://arxiv.org/pdf/2609.02011v1)

**作者:** Jayakumar Manoharan `[一作]` (Electric Power Research Institute), Yamini Sehgal `[通讯]` (Electric Power Research Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在电力系统 CIM/CGMES 模型上实现基于上下文预算的检索与答案生成，提出种子锚定渲染策略，保证在固定 8,000 字符预算内保留所有答案相关信息。

**💡 创新点**

提出无额外学习参数、基于渲染/截断边界的局部性先验，提供可检验的答案保留保证，解决高连通节点导致的答案丢失问题。

**🔧 技术方法**

采用确定性图检索与渲染（seed‑anchored 渲染），与 LLM 抽取的 GraphRAG 框架（LightRAG、Microsoft GraphRAG、HippoRAG）对比；使用 Qwen2.5‑7B‑Instruct 读取器、固定 8,000 字符预算、hop‑2 限制检索。

**📊 数据集**

基于 ENTSO‑E CGMES 合规测试集（MicroGrid、MiniGrid、SmallGrid 及其 3.0 版本）以及独立的 RealGrid（72,418 对象）等数据集。

**📈 对比分析**

在 budget‑binding 网络上，seed‑anchored 渲染将单跳/多跳准确率从 0.12/0.00 提升至 1.00，整体问答准确率从 0.45 升至 0.97；与 LLM 抽取的图相比，在相同读取器和预算下准确率不逊且无图构造 token 成本，且在不同 CGMES 编码下保持稳健。

**⚠️ 局限性**

评测主要基于模板生成的自参照问题，约 83% 的答案可在语料中直接出现，限制了对泛化能力的评估；网络覆盖范围仅为少数几种拓扑，未验证大规模实际运营网；答案保留保证仅适用于种子局部查询；读者性能随模型变化，存在读者依赖性。

---

## 56. RosettaBitcoin: An Artifact-Backed Experience Report on Verification Infrastructure for Agent-Assisted Consensus Validators

**arXiv ID:** 2609.01702 | [PDF](https://arxiv.org/pdf/2609.01702v1)

**作者:** Donavon Guyot `[一作]` `[通讯]` (Independent Researcher), Donavon Guyot (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

分析RosettaBitcoin的固定快照，重建验证子系统，评估12个语言实现的验证结果及失败记录。

**💡 创新点**

构建了面向失败记录的可追溯证据管道，采用端口自有证明与共享脚本相结合的方式，使代理辅助开发可被审计。

**🔧 技术方法**

使用SQLite证据数据库、Project报告脚本、差分测试、共享脚本语料库及语言模型辅助生成（Codex/GPT‑5、Claude）等技术。

**📊 数据集**

采用Bitcoin testnet4区块字节、45个共享脚本测试集、111条阻塞记录，结合Zenodo快照中的12个实现及诊断补充包。

**📈 对比分析**

通过Project报告与预检脚本比较实现，检查是否通过严格的5k基线、50k/100k长跑以及可维护性门；结果显示9个实现通过长跑，但没有空态到尖顶的完整验证，性能未量化。

**⚠️ 局限性**

缺乏受控实验与因果分析，基础设施共享导致潜在的共模错误，未覆盖完整的共识规则和全节点功能，诊断补充仅为非核心证据。

---

## 57. Skill-as-API: Confidential Multi-Agent Coordination for Agentic Software Engineering

**arXiv ID:** 2609.01677 | [PDF](https://arxiv.org/pdf/2609.01677v1)

**作者:** Ziwei Zhao `[一作]`, Xizhi Ding `[通讯]` (London Business School)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种协议，保护AI编码代理的技能代码和系统提示不被泄露，并实现了多方协作中的权限管理。

**💡 创新点**

通过闭包捕获技术让技能主体不离开本地进程，设计了信任层ACL、自动降级、技能隐藏和函数边界防御等四层防护，形成了协议层的隐私保护。

**🔧 技术方法**

基于Python SDK、XMTP v3（MLS）传输、SHA256哈希、闭包捕获、信任层级控制、自动降级算法、技能隐藏机制、函数边界防御等技术。

**📊 数据集**

未使用公开数据集，采用实际代码审查与安全扫描任务的模拟PR评审案例。

**📈 对比分析**

与MCP、A2A、AgentCrypt等协议比较，功能上实现闭包保护、技能隐藏和注入防御；性能上热重连延迟1.8–2.9秒，冷启动30–60秒，负载低。

**⚠️ 局限性**

跨步输入毒化未防护，时序侧信道存在，权限仅按对等而非按技能细粒度，协同攻击与模式推理仍有残余风险。

---

## 58. Multi-Turn LLM Conversations under the Least-Recently-Used Policy: Mean-Field Asymptotics and Hit Ratio Approximation

**arXiv ID:** 2609.02027 | [PDF](https://arxiv.org/pdf/2609.02027v1)

**作者:** Heyuan Yao `[一作]` (Northwestern University), David Simchi-Levi `[通讯]` (Purdue University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究多轮对话 LLM 服务器中 KV 缓存命中率的理论估计与实践应用。

**💡 创新点**

创新点在于提出一种简洁的多轮对话模型并在 LRU 策略下使用均值场极限证明命中率收敛到闭式极限，从而给出理论驱动的命中率估计器。

**🔧 技术方法**

主要技术包括 LRU 缓存建模、均值场渐近分析、概率论与离散时间马尔可夫过程理论，以及实验验证的数值求解。

**📊 数据集**

使用的数据集为公开的 ShareGPT 多轮对话记录，模型为 Qwen3-8B，实验平台为华为 Ascend 910B2 NPU。

**📈 对比分析**

与真实服务器测得的命中率对比，估计器的绝对误差小于 0.02，误差率低于 10%，并能准确捕捉到交互速率或间隔时间变化导致的命中率下降。

**⚠️ 局限性**

局限性包括对整个对话内容使用 LRU 的简化假设、块增量和间隔时间独立性假设，以及对参数估计的依赖，实际系统中块级别局部失效与多级存储策略会带来额外误差。

---

## 59. Designing Versatile Samples for Learned Trajectory Scoring

**arXiv ID:** 2609.01799 | [PDF](https://arxiv.org/pdf/2609.01799v1)

**作者:** Yaguang Li `[一作]` (Purdue University), Ziran Wang `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在生成-选择管线中，提出一种通过对人类轨迹进行横向和纵向扰动生成多样化训练样本，从而提升轨迹评分器在安全门控上的判别能力，并将其与冻结的生成式规划器结合。

**💡 创新点**

创新点在于把训练数据本身视为可设计的变量，利用在决策边界附近产生正负样本来显著增强评分器的性能，同时保持规划器与评分器的分离架构。

**🔧 技术方法**

采用Transformer多头评分器对各个EPDMS组件分别进行预测；使用DiffusionDrive和MeanFuser两种冻结生成式规划器；通过模拟器评估扰动轨迹的margin并进行标签化。

**📊 数据集**

主要使用NAVSIM navtrain数据集构建训练集，评估基准为NAVSIM navtest。

**📈 对比分析**

与基准方法和原始评分器对比，DiffusionDrive的评分器提升至90.1 EPDMS、MeanFuser提升至90.4 EPDMS，较基准提高约1.8 EPDMS；相较随机采样表明设计样本贡献显著。

**⚠️ 局限性**

局限性包括：保持规划器冻结限制了进一步提升规划质量的空间；所设计的扰动可能不足以覆盖所有决策边界；在极端场景下对多模态驾驶行为的捕捉仍有限。

---

## 60. UTP-Bench: Uncertainty-aware Travel Planning Benchmark

**arXiv ID:** 2609.02421 | [PDF](https://arxiv.org/pdf/2609.02421v1)

**作者:** Etcharla Revanth Rao `[一作]` (IIT Bhubaneswar), Abhik Jana `[通讯]` (IIT Bhubaneswar)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个大规模的不确定性感知旅行行程规划基准（UTP‑Bench），并为LLM生成的行程提供了人类标注的黄金参考。

**💡 创新点**

创新点包括：①基于真实的交通延误与客流分布构建多风险偏好查询；②设计了三种新评估指标（BAS、CATS、TDAS）来量化行程的时间缓冲、客流匹配与运输延误吸收；③将人工标注与脚本辅助质量控制相结合，提升数据可靠性。

**🔧 技术方法**

使用了多种主流LLM（GPT‑5、Qwen3、Mistral‑7B、Phi‑4）在加入不确定性提示的框架下生成行程，并通过脚本工具实现自动化评估与质量控制。

**📊 数据集**

使用的核心数据集是印度504个城市的POI、餐饮、住宿与多模态交通网络，并加入了历史延误统计与客流密度信息，共构建了1000个三、五、七天的旅行查询与对应的人类黄金行程。

**📈 对比分析**

通过传统的交付率、通用/硬约束通过率以及新提出的BAS、CATS、TDAS三种指标对四个LLM进行对比，实验显示它们在时间缓冲、运输延误吸收和风险偏好方面性能低于人类规划，鲁棒性存在显著差距。

**⚠️ 局限性**

局限性包括：①仅使用历史统计而非实时交通/客流API；②覆盖范围局限于印度，缺乏多语言支持；③LLM在长周期规划中易产生时间与空间不一致，整体鲁棒性仍不足。

---

## 61. What Is Worth Representing? Representational Empowerment for Continual Model Construction

**arXiv ID:** 2609.02322 | [PDF](https://arxiv.org/pdf/2609.02322v1)

**作者:** Fei Dai `[一作]` (University of California Berkeley), Charley Wu `[通讯]` (TU Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了持续模型构建问题，提出了代表性赋能（Representational Empowerment）来评估并选择能扩展代理规划能力的表征元素，并通过Curator-Actor架构在闭词汇因果学习与开放词汇符号规划两类实验中验证其效果。

**💡 创新点**

创新点在于将赋能概念转向内部表征，提出代表性赋能作为连续模型构建的评价标准，并将其与库维护相结合实现资源受限下的自适应表征选择。

**🔧 技术方法**

使用信息论赋能度量、Curator-Actor层次化架构、LLM生成PDDL符号、经验式代理规划与交互式评估等技术。

**📊 数据集**

实验数据集包括多变量因果发现的Alchemy环境、BabyAI与Zelda两种网格世界，涵盖训练、保留、测试等多任务设置。

**📈 对比分析**

与信息增益、环境赋能、随机、Motif、WorldCoder、PPO等基线相比，代表性赋能下的Curator在闭词汇实验中实现更高的图结构恢复与目标成功率，在开放词汇实验中实现更好向后转移、零样本泛化、库紧凑和更低LLM成本。

**⚠️ 局限性**

局限包括人类实验仅在固定词汇下验证，开放词汇实验仅测试两种网格世界且对探测目标依赖，未对提议分布的覆盖性进行充分分析，且整体成功率仍偏低。

---

## 62. Poisoning Attacks on the PGM-index

**arXiv ID:** 2609.02328 | [PDF](https://arxiv.org/pdf/2609.02328v1)

**作者:** Atsuki Sato `[一作]` (University of Tokyo), Yusuke Matsui `[通讯]` (University of Tokyo)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了对PGM-index及其底层PLA构造的投毒攻击，目标是让最佳分段数量急剧增加，从而显著提升索引尺寸和查询开销。

**💡 创新点**

创新点在于提出PGM-attack和相关理论上限，证明在仅10%投毒下段数可提升至120倍，并且在实例级别给出1.92×的最优上界，首次揭示PLA最优性对攻击的脆弱性。

**🔧 技术方法**

使用最大误差最大化、覆盖合法键最小化、段数最大化等理论框架，并设计了贪心、连续、DI-Consecutive等高效算法，结合线性回归与PLA构造来生成投毒键。

**📊 数据集**

实验覆盖了多种真实数据集（Amzn、Osmc、Face、YCSB、Longitudes、Longlat）和合成分布（Uniform、Normal、Lognormal），以验证攻击效果。

**📈 对比分析**

与随机投毒、随机相邻等基线相比，PGM-attack能使段数和索引尺寸提升数十倍（最多120倍），查询时间提升不超过1.22倍，且在其他PLA学习索引（如FITing-Tree、PGM++）中同样有效。

**⚠️ 局限性**

局限在于攻击采用贪心策略，未必达到全局最优；对错误参数ε敏感，白盒假设对实际场景限制较大；计算复杂度在极大规模数据上仍高，需要进一步优化。

---

## 63. PGPO: Potential-Guided Policy Optimization for Multi-Turn Agentic Tasks

**arXiv ID:** 2609.02236 | [PDF](https://arxiv.org/pdf/2609.02236v1)

**作者:** Yuyao Zheng `[一作]` (Fudan University), Dejing Dou `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多轮代理任务中提出一种新的奖励信号——潜在引导优势（Potential‑Guided Advantage），通过对锚点状态组（anchor‑state group）的回报统计进行经验性潜在值估计，并将相邻状态的潜在差异作为动作的优势，进一步通过任务级成功率自适应权重与原始步级优势融合，以改进稀疏终端奖励下的局部信用分配。

**💡 创新点**

① 将经验性潜在值（由多轨迹锚点状态组回报平均得到）引入到动作优势计算；② 通过潜在差值（γΦ(s′)−Φ(s)）实现跨轨迹信用传播，尤其在失败轨迹中提升动作区分度；③ 引入任务级自适应权重使得潜在优势在难度高的任务上贡献更大，避免在已成功率高的任务中过度调整。

**🔧 技术方法**

基于 anchor‑state grouping 的组级强化学习；经验性潜在值估计；潜在差值优势；任务级自适应加权；PPO‑style 训练目标；无额外 critic 网络或环境交互。

**📊 数据集**

ALFWorld（长序列交互式任务）和 WebShop（基于网页的任务）两个稀疏终端奖励基准。

**📈 对比分析**

与 Prompting、ReAct、Reflexion、PPO、RLOO、GRPO、GiGPO、HGPO 等基线相比，PGPO 在 ALFWorld 和 WebShop 上均取得最高或接近最高的成功率，特别是在 1.5B/7B 模型规模下表现优异；在 Unseen 任务上与 HGPO 接近，略逊，但在其他指标上表现更稳健；机制分析显示 PGPO 在失败轨迹中的信用分散度明显提升。

**⚠️ 局限性**

① 依赖于锚点状态的精确匹配，易受文本表述差异、部分可观测或噪声影响；② 小组样本导致潜在估计方差高，可能产生噪声；③ 目前仅在 ALFWorld/WebShop 验证，缺乏更广泛的开放式任务评估；④ 潜在优势虽不影响最优策略，但在安全敏感场景下可能增加误操作风险，需要额外安全约束。

---

## 64. Reinforcement learning to choose optimizers

**arXiv ID:** 2609.01811 | [PDF](https://arxiv.org/pdf/2609.01811v1)

**作者:** Martin van der Schelling `[一作]` (Delft University of Technology), Miguel A. Bessa `[通讯]` (Brown University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出RL2CO框架，将优化器选择与切换视为强化学习的顺序决策问题，训练一个递归策略在梯度和无梯度优化器之间动态切换。

**💡 创新点**

首次同时学习优化器选择、切换时机与持续时间，并通过情境门控的多专家网络实现对不同问题特征的自适应策略。

**🔧 技术方法**

使用离线actor‑critic强化学习（PPO+GAE）训练递归策略，利用上下文代理与门控网络路由专家头，切换时热启动并传递最佳解与步长信息。

**📊 数据集**

在BBOB（含噪声）及自定义高维子空间嵌入的100个训练任务和100个留出测试任务上进行评测，并在CEC 2005/2017/2013等公开基准以及未见任务族上进一步验证。

**📈 对比分析**

与每个静态优化器、随机切换调度以及VBS/SBS基准比较，RL2CO在除极低预算外的所有测试集上均超过所有单一优化器，实现更高的目标命中率；在某些基准集仅与随机基线相当，但在可利用多样性较大的场景中显著优于所有成员。

**⚠️ 局限性**

策略仅在固定的优化器组合上训练，无法迁移到新组合；切换仅传递最佳解与步长，未实现跨算法内部状态的完整迁移；评估指标主要基于函数评估，未考虑真实时间成本。

---

## 65. Efficient Passive Acoustic Monitoring of Killer Whales Using a Two-Stage Detection and Ecotype Classification Cascade

**arXiv ID:** 2609.01792 | [PDF](https://arxiv.org/pdf/2609.01792v1)

**作者:** Daniela Ruiz `[一作]` (Microsoft AI for Good Research Lab), Juan M. Lavista `[通讯]` (Microsoft AI for Good Research Lab)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种轻量级的两阶段级联模型，用于实时监测杀人鲸的被动声学信号：第一阶段使用ResNet-18检测鲸叫，第二阶段在检测到的鲸叫上进行五种海域生态类型分类，并通过置信度阈值实现“未分类”回退。

**💡 创新点**

创新点在于将检测与生态类型识别分离为两步，既保持高召回又提高分类精度；采用统一的ResNet-18骨干网络并在第二阶段引入温度缩放校准与阈值抑制，实现置信度自适应；在真实部署环境中通过主动学习对检测器进行域适配，显著提升精度。

**🔧 技术方法**

技术核心包括：log‑mel频谱预处理、轻量级ResNet‑18网络、交叉熵与逆频率加权损失、温度缩放校准、置信度阈值抑制、基于活性学习的在线微调以及GPU加速的批量推理。

**📊 数据集**

主要使用DCLDE 2027公开数据集（约1065小时录音、206k标注事件），其中包含5种杀人鲸生态类型；此外，还利用Puget Sound现场监测数据进行域适配实验，并引入Cook Inlet beluga背景噪声数据丰富背景样本。

**📈 对比分析**

与Perch 2.0等预训练模型的冻结嵌入+LogReg做对比，本文的ResNet‑18在窗口级别的宏F1为0.958，显著高于Perch 2.0（0.933）及Perch 1.0（0.913）；在完整级联下宏F1提升至0.933（单阶段为0.919），尤其在罕见OKW生态类型上从0.842提升到0.930。

**⚠️ 局限性**

局限性包括：级联性能受第一阶段检测误差限制；置信度阈值需针对每个部署环境重新校准；当前仅对检测器做域适配，生态分类器尚未在目标域验证；未进行持续流音频的完整评估；模型在边缘设备上的实际功耗与延迟仍待进一步验证。

---

## 66. Candidate Generation and Definition-Guided Verification for Sentence-Level Depression Symptom Recognition

**arXiv ID:** 2609.01833 | [PDF](https://arxiv.org/pdf/2609.01833v1)

**作者:** Weiming Li `[一作]` (Instituto Superior Técnico), Joao Sanches `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段句子级抑郁症状识别框架，先用对比学习微调的句子编码器生成症状候选，再用定义约束的LLM进行候选验证与说明生成。

**💡 创新点**

创新点在于将症状候选生成与基于DSM‑5定义的存在/否定验证分离，利用自检机制提升判定一致性，同时结合对比学习提升编码器对症状细粒度区分的能力。

**🔧 技术方法**

使用对比学习微调的 MentalSBERT‑S 句子编码器、DeepSeek LLM（4‑bit QLoRA微调）进行候选验证，并在验证中嵌入DSM‑5定义与自检步骤；整体框架融合了句子编码、对比学习、LLM生成与结构化推理。

**📊 数据集**

实验数据基于公开的 ReDSM‑5 句子级抑郁症状标注集（约 1,767 句子），包含 9 个 DSM‑5 症状类别，并使用原始的 Reddit 帖子作为上下文。

**📈 对比分析**

与多种基线（BART‑MNLI 零样本、MedGemma‑4B‑IT 零样本、MentalBERT、DeBERTa、DeepSeek 零样本/少样本/检索增强）以及单阶段监督模型对比。两阶段模型在 324 条测试句子上取得最高精度 0.731±0.005、宏观 F1 0.690±0.011、加权 F1 0.715±0.010，说明质量与专家标注相似度较高，且推理成本低于纯 LLM 调用。

**⚠️ 局限性**

局限包括：数据量有限且长尾分布明显，罕见症状识别仍不稳定；Stage‑1 生成错误会被 Stage‑2 直接接受导致无法纠正；统计显著性有限（大部分对比均未通过层级自举检验）；系统仅提供句子级证据标注，非完整诊断。

---

## 67. Genuine Information Needs of Social Scientists Looking for Data

**arXiv ID:** 2609.02303 | [PDF](https://arxiv.org/pdf/2609.02303v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 68. Towards Behavior Tree-Guided Vulnerability Detection with Lightweight LLMs

**arXiv ID:** 2609.01758 | [PDF](https://arxiv.org/pdf/2609.01758v1)

**作者:** Enna Basic `[一作]` (Örebro University), Alberto Giaretta `[通讯]` (Örebro University)

**通讯引用:** 826 | [OpenAlex ID](https://openalex.org/A5004767362)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了将 Java 源码转换为行为树（BT）作为 LLM 漏洞检测的中间表示，并与原始源码和 AST 在同一 LLM 下进行对比实验。

**💡 创新点**

创新点在于提出了完整的 AST→BT 预处理框架，证明 BT 在保持输入压缩、提升召回率方面优于 AST，尤其在长代码和有限上下文窗口时更具优势。

**🔧 技术方法**

使用 JavaParser 生成 AST，随后自定义映射规则将 AST 转换为 BT；使用量化的本地 LLM Mistral Small 3.2 24B，并采用统一的安全分析 prompt；评估指标包括准确率、精确率、召回率、F1 以及语义相似度。

**📊 数据集**

数据集为 NIST Juliet Java 测试套件的 460 个样本，分为 200 个短样本和 260 个长样本，涵盖多种 CWE 类别。

**📈 对比分析**

在相同模型、prompt 和解码配置下，三种输入格式分别运行；结果显示 BT 在短样本中获得最高召回率，长样本中整体性能（F1、召回率）优于源码，AST 在长样本中因超出上下文窗口而不可用。

**⚠️ 局限性**

局限性包括仅使用合成样本、单一模型与 prompt、仅评估 Java 语言、未验证在真实项目或多语言环境中的效果，以及对 CWE 关系的评估指标可能过于严格。

---

## 69. PragAlign: Feedback-Guided Pragmatic Alignment for Controlled Synthetic Dialogue Generation

**arXiv ID:** 2609.02480 | [PDF](https://arxiv.org/pdf/2609.02480v1)

**作者:** Smitha Muthya Sudheendra `[一作]` (University of Minnesota, Twin Cities), Jaideep Srivastava `[通讯]` (University of Minnesota, Twin Cities)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出PragAlign框架，实现基于意图、情绪、连贯性和流畅度约束的闭环生成与评估；

**💡 创新点**

将LLM评估器的结构化反馈嵌入多轮生成修正循环，显著提升多约束满足率，尤其是情绪对齐；

**🔧 技术方法**

使用GPT‑4.1生成、GPT‑4o‑mini评估器、Tree‑of‑Thought式评价分解以及生成‑评估‑修正闭环；

**📊 数据集**

构造了800个匹配的服务对话规格（覆盖10个客服域），并采集1,200条人工评估样本；

**📈 对比分析**

与单次生成、无反馈多次生成、去情绪、去人格等对照实验，自动接受率从72%提升至99.5%，情绪对齐提升最显著；

**⚠️ 局限性**

评估器与生成器同源导致评估不独立；情绪评估自动高分与人工一致性低；人格调节未被验证；人工评估未与实验条件配对。

---

## 70. Breadth Beats Depth: Improving GCG-Based Jailbreak Optimization with Breadth-Oriented Suffix Search

**arXiv ID:** 2609.02172 | [PDF](https://arxiv.org/pdf/2609.02172v1)

**作者:** Shiliang Xiao `[一作]` (Guangdong University of Foreign Studies), Qiliang Lin `[通讯]` (Guangdong University of Foreign Studies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个可插拔的框架 Breadth‑Oriented Suffix Search（BOSS），改进 GCG 基础的 jailbreak suffix 优化。

**💡 创新点**

通过“Tail‑Focused Adversarial Loss”与行为覆盖度进行终端后缀选择，并将搜索预算从深度递归转向多路径宽度探索，形成新的搜索策略。

**🔧 技术方法**

基于梯度引导的离散 token 搜索（GCG），多起点短轨迹、源模型诊断、源损失与尾部损失组合、行为覆盖门控等技术。

**📊 数据集**

使用 HarmBench 数据集进行训练（20 行为）和评估（200 行为）。

**📈 对比分析**

与 GCG、I‑GCG、GJO 三种基线对比，平均攻击成功率提升 10–20% 且搜索时间减半，同时目标响应一致性亦明显提升。

**⚠️ 局限性**

仅依赖源模型诊断，跨模型转移仍不确定；在极度差异的目标模型上可能无法保持预期效果。

---

## 71. Reconciling Kinesthetic Mismatches: A Somatic Alignment Mindset for Musical Body Transformation

**arXiv ID:** 2609.01981 | [PDF](https://arxiv.org/pdf/2609.01981v1)

**作者:** Ziyue Piao `[一作]` (McGill University), Marcelo M. Wanderley `[通讯]` (McGill University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了Somatic Alignment Mindset（SAM）框架，旨在通过将道家哲学与多感官身体转化体验（BTE）相结合，帮助音乐学习者解决运动感知失配问题；

**💡 创新点**

创新点在于将道家概念（如自然、无为、内观、抱一）引入HCI设计，强调技术作为反射媒介而非纠错工具，并通过Adaptation、Assessment、Awareness三大策略系统化地解决内外感知不一致；

**🔧 技术方法**

主要技术包括可穿戴触觉反馈装置（如振动手套、胸腔贴片等）、实时生理信号采集（肌电、呼吸、姿态）以及多模态交互算法，将内在状态映射为感知纹理或音频；

**📊 数据集**

论文未使用公开数据集，而是基于已有的可穿戴感知研究和实验设备进行理论建构与案例讨论；

**📈 对比分析**

未进行实验对比或性能评估，本文以概念模型和设计指导为主；

**⚠️ 局限性**

局限性包括缺乏系统化的实证验证、样本规模有限、技术实现依赖于特定硬件平台，且如何在实际教学中平衡无为适配与学习者自我内化仍待探讨。

---

## 72. SALA: Semantic-Aware Logical Alignment for Complex Reasoning in In-Context Learning

**arXiv ID:** 2609.02336 | [PDF](https://arxiv.org/pdf/2609.02336v1)

**作者:** Zhao Ji `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SALA 框架，利用任务自适应推理操作构建与语义 DTW 对齐实现演示选择，从而提升 LLM 的 in-context learning 推理表现。

**💡 创新点**

创新点在于：①通过 LLM 自动诱导任务特定推理操作，扩展固定操作集合；②将操作序列映射到连续语义空间后使用动态时间规整（DTW）进行软匹配，实现逻辑层面而非表面相似度的对齐。

**🔧 技术方法**

技术包括：LLM 诱导操作、操作描述生成、语义嵌入（bert-base-uncased）、动态时间规整（DTW）、线性操作序列解析与对齐。

**📊 数据集**

实验使用四个推理基准：SVAMP、GSM8K、CommonsenseQA（CMSQA）和 StrategyQA（StrQA）。

**📈 对比分析**

与随机、BM25、TopK‑BERT、DPP‑BERT、PSL、LMS3 等七个基线对比，SALA 在所有三种 LLM（Llama3‑8B、Qwen2.5‑7B、DeepSeek‑V4‑Pro）上平均提升约 1–3%，在 DeepSeek‑V4‑Pro 取得最高 91.42% 的整体准确率。

**⚠️ 局限性**

局限性在于：①依赖 LLM 诱导与解析，结果受模型与提示策略影响；②仅使用线性操作序列，无法覆盖更复杂的层次或图结构推理。

---

## 73. Beyond Outcome Gaps: Process-Aware Fairness Diagnosis for LLM-based Multi-Agent Decision Systems

**arXiv ID:** 2609.02092 | [PDF](https://arxiv.org/pdf/2609.02092v1)

**作者:** Yiran Zhao `[一作]` (Nanjing University of Aeronautics and Astronautics), Xiaogang Xu `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了SCOPED-Hiring流程感知公平诊断流水线，对LLM驱动的多智能体招聘系统进行受控简历变体生成、两阶段委员会评估、轨迹记录、六视角公平信号提取与诊断矩阵分析。

**💡 创新点**

创新点包括：①把公平审计从终端结果扩展到多智能体决策轨迹；②提出统一的六视角（结果、反事实、过程、路径、动态、设计）诊断框架；③揭示“隐藏轨迹不公平”并基于诊断结果提出Fair Skills修复方案，显著降低公平负担。

**🔧 技术方法**

使用AutoGen多智能体框架、GPT‑5.4‑mini / Gemini‑3.1‑flash‑lite / Qwen‑3.5‑flash LLM，配合 Bias Revelation Index (BRI)、关键词分类、文本与数值指标归一化、Top‑2 聚合等技术。

**📊 数据集**

基于Djinni招聘数据集（约23万份CV），构造控制简历变体；实验覆盖Java Developer、HR Manager两大职位，Gemini版亦包括Business Analyst与QA Engineer。

**📈 对比分析**

通过匹配干预（INT0‑INT3）对分层公平负担指标进行评估，诊断导向的INT3条件将总层级公平负担降低72.3%，仅导致1.86个百分点的聘用率变化；相较于基线、通用提醒和结构化修复，INT3在所有层面显著提升。

**⚠️ 局限性**

局限性在于仅在招聘场景验证，缺乏跨域泛化；受控简历与轨迹字段为模拟审计产物，真实部署时工具使用和日志可能不同；诊断框架依赖指标选取、归一化与聚合方式，难以完全揭示因果路径。

---

## 74. A Computational Comparison of Fourier Spectral Differentiation and Spatial Automatic Differentiation in Periodic Physics-Informed Neural Networks

**arXiv ID:** 2609.02110 | [PDF](https://arxiv.org/pdf/2609.02110v1)

**作者:** Xilai Liang `[一作]` (Guangdong Technion--Israel Institute of Technology), Zhao Zhang `[通讯]` (Shandong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在物理空间的 PINN 训练中，针对相同的网络架构、优化器、采样和训练长度，对比了自动微分（AD）与 Fourier 频谱差分两种空间导数计算方式，并在标准 PINN 与 Causal PINN 两种框架下分别对 Allen–Cahn、Korteweg–de Vries (KdV) 与 Kuramoto–Sivashinsky (KS) 三个周期性一维 PDE 进行实验。

**💡 创新点**

提出了一种受控实验方案：在每一对实验中仅改变空间导数计算方式，保持所有其它因素不变，从而得到两种方法在相同条件下的直接性能对比；并将结果应用于多种 PDE 与两种 PINN 结构，首次系统评估 Fourier 频谱差分在此类问题上的计算优势。

**🔧 技术方法**

核心技术包括：物理信息神经网络（PINN）、Causal PINN 时间因果权重、自动微分（AD）与 Fourier 频谱差分（频谱乘法）、基于 FFT 的快速频谱变换、Adam 优化器以及张量化的空间采样网格。

**📊 数据集**

使用的“数据集”实际上是三种周期性 PDE 的参考解（Allen–Cahn、KdV 用 Fourier pseudospectral 方案得到，KS 用现成的数值轨迹），并在统一的 256 点周期性网格上采样，以评估模型预测误差。

**📈 对比分析**

比较方法：对每个 PDE-框架组合分别跑三次随机种子实验，形成 AD 与 Fourier 的 paired 运行；测量最终相对 L₂ 误差、全流程训练时间与 GPU 峰值内存。结果显示 Fourier 频谱差分平均可获得 2.90×–18.52× 的训练速度提升，内存占用下降 68.7%–94.1%，误差保持在同一数量级，未出现显著的精度优势。

**⚠️ 局限性**

局限性包括：仅在 1D 周期性、均匀网格问题上测试，无法推广到非周期、非均匀或高维问题；实验结果受网络规模、硬件（RTX 4060）与实现细节影响；Causal PINN 的因果参数固定且未探讨其调优；未单独测量导数计算耗时，仅给出整体训练性能。

---

## 75. MGDiff: Multi-Interest Sequence Recommendation with Masking GNN-Guided Diffusion

**arXiv ID:** 2609.01619 | [PDF](https://arxiv.org/pdf/2609.01619v1)

**作者:** Wenjing Xiao `[一作]`, Hao Ding `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 MGDiff 框架，利用扩散模型进行序列推荐，并通过双层语义指导和流行度感知引导提升推荐质量。

**💡 创新点**

创新点在于：① 将权重自适应遮蔽 GNN 与动态路由混合专家网络耦合，形成双层语义指导；② 通过可微分的流行度调整和对比学习实现无偏推荐，显著降低模式崩溃与流行度偏差。

**🔧 技术方法**

采用扩散概率模型、带遮蔽的图神经网络（自适应遮蔽+课程学习）、动态路由 Mixture‑of‑Experts、Gumbel‑Softmax、对比学习与流行度加权的相似度校正技术。

**📊 数据集**

使用 Amazon Beauty、Amazon Toys、MovieLens‑1M 与 Steam 四个公开序列推荐数据集进行实验。

**📈 对比分析**

与 GRU4Rec、SASRec、BERT4Rec、ComiRec、TiMiRec、DreamRec、DiffuRec、DiQDiff、ACVAE、CL4SRec 等基线模型对比，MGDiff 在所有数据集的 HR@5、NDCG@5、Coverage、Gini 等指标均优于最佳基线，提升幅度约 8–10% 以上，并显著改善长尾项目覆盖与多样性。

**⚠️ 局限性**

局限性：扩散过程计算量大，推理时需多步迭代；对超参数敏感；在极端冷启动场景下的鲁棒性尚未完全验证；模型可解释性有限。

---

## 76. The Utility of LLMs in Recommender Systems Explanation Evaluation

**arXiv ID:** 2609.01627 | [PDF](https://arxiv.org/pdf/2609.01627v1)

**作者:** Kathrin Wardatzky `[一作]` (University of Zurich), Abraham Bernstein `[通讯]` (University of Zurich)

**通讯引用:** 11162 | [OpenAlex ID](https://openalex.org/A5073592405)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对推荐系统解释进行大规模评估，使用LLM生成并评价18种不同信息组合的解释，并与人类评估进行对比。

**💡 创新点**

首次系统评估LLM作为评估者对RS解释的可行性，并提出基于LLM的评估准则与实践指南。

**🔧 技术方法**

采用LLM-as-judge框架，使用14款LLM（包含开源与专有模型）、多维度评分（满意度、可审查性、透明度等）、用户研究和统计分析方法。

**📊 数据集**

使用Mindreader电影推荐数据集，PGPR知识图谱基推荐模型生成解释。

**📈 对比分析**

通过对比LLM评分与216名参与者评分，计算Krippendorff α、Kendall τ等指标；结果显示LLM与人类在排名上中等相关、评分一致性低。

**⚠️ 局限性**

仅在单一领域（电影）与单一解释方式（路径追踪）下进行实验，LLM在事实性与可审查性评估上存在偏差，模型规模差异影响结果。

---

## 77. Counter-GEO-Bench: Evaluating Defenses Against Information-Distorting Generative Engine Optimization

**arXiv ID:** 2609.02316 | [PDF](https://arxiv.org/pdf/2609.02316v1)

**作者:** Bing Zheng `[一作]` (Tsinghua University), Wenming Yang `[通讯]` (Tsinghua University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了针对信息扭曲的 GEO 优化攻击的防御基准，并提出了一种基于对比学习的轻量级检测器 Guard。

**💡 创新点**

首创了 GEO 扰乱信息攻击的评估基准，证明现有安全过滤器无法防护，并提供了可训练的 GEO 适应检测方法。

**🔧 技术方法**

使用了句子嵌入、BERT 对比学习、chunk‑level 过滤、检索增强生成（RAG）以及 LLM 评估技术。

**📊 数据集**

采用 GEO‑Bench 公开数据集，并通过自动重写与人工验证生成了 247 条配对查询-源文档。

**📈 对比分析**

在三款开源 LLM（Gemma‑4、Qwen‑3.5、Llama‑4）上进行对比实验，传统安全过滤器提升不到 3.2pp，而 Guard 将攻击成功率降低 47.6% 并几乎不影响回答质量。

**⚠️ 局限性**

局限在于仅覆盖英文单文档情景、三种公开 LLM，未评估多语言、多源协同攻击以及对抗性/开集攻击。

---

## 78. CAHR-Net: Condition-Adaptive Hysteresis Reconstruction for Compact and Interpretable Magnetic Core Loss Modeling

**arXiv ID:** 2609.01991 | [PDF](https://arxiv.org/pdf/2609.01991v1)

**作者:** Chunye Gong `[一作]` (National University of Defense Technology), Cong Yao `[通讯]` (National University of Defense Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种条件自适应的磁滞回波重构网络（CAHR‑Net），通过在中间表征层注入频率、温度和波形统计来预测磁芯损耗；

**💡 创新点**

创新点在于将操作条件直接作用于磁滞回归表征层，利用FiLM实现物理可解释的尺度与平移调制，并配合分阶段损失训练以充分激活调制路径；

**🔧 技术方法**

使用时序卷积编码器、FiLM特征调制、磁场重构头、循环面积积分、残差对数损失校正；优化采用AdamW+余弦学习率调度；

**📊 数据集**

使用MagNet公开的A–E材料最终阶段数据集，包含1024点周期波形及频率、温度等统计；

**📈 对比分析**

与经验公式、公开挑战条目和内部基线比较，CAHR‑Net在平均p95误差6.89%和极端材料p95 14.87%上优于所有对比方法，参数量仅1874，远低于最优黑盒模型；

**⚠️ 局限性**

仅在单周期稳态无直流偏置下训练，单材料训练，未考虑材料推荐与不确定性，模型对不同频率范围的泛化仍待验证。

---

## 79. TC-Next: Zero-Shot Multimodal Cyclone Forecasting

**arXiv ID:** 2609.02085 | [PDF](https://arxiv.org/pdf/2609.02085v1)

**作者:** Zhe Wang `[一作]` (Carnegie Mellon University), Chien-Yi Chang `[通讯]` (Durham University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出TC-Next，一种多模态深度学习模型，利用基础天气模型的预报字段和高分辨率卫星红外影像来预测西太平洋台风的轨迹与强度。

**💡 创新点**

其创新点在于实现了零样本跨模型迁移、通过卫星影像提升长时程强度预测，并且仅依赖通用大气变量即可在不同预报源间通用。

**🔧 技术方法**

采用了编码‑解码架构，包含宏观与微观CNN、区域候选网络、RoIAlign、低秩多模态融合以及LSTM自回归预测。

**📊 数据集**

训练使用GraphCast预报与ERA5重分析的西太平洋数据，配合GridSat红外影像与IBTrACS台风轨迹，评估时扩展到Pangu‑Weather、IFS HRES及TC Vitals等数据集。

**📈 对比分析**

与WN‑C直接追踪器和规则基准TempestExtremes相比，TC‑Next在轨迹误差上降低15–49%，强度误差提高3–6倍，并在零样本条件下在Pangu、IFS HRES等模型上保持领先。

**⚠️ 局限性**

限制在于对高分辨率卫星影像的依赖、未针对实时输入进行微调、以及在极短时程强度预测上仍不及传统方法；此外模型目前仅验证于西太平洋，需推广到全球台风盆地。

---

## 80. MERGED: Multimodal Entity Resolution via Generated Expert Reasoning Distillation

**arXiv ID:** 2609.01913 | [PDF](https://arxiv.org/pdf/2609.01913v1)

**作者:** You-Lin Chen `[一作]` (Amazon), Pedro Herrero-Vidal `[通讯]` (Amazon)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了MERGED框架，实现利用大VLM生成的标签和推理进行知识蒸馏，实现无人工标注的多模态产品实体解析。

**💡 创新点**

创新点：①同时蒸馏标签与结构化推理；②采用两阶段SFT+DPO训练；③利用多教师共识与Meta‑Judge生成偏好对；④在新关系定义上仅需1万样本即可快速迁移。

**🔧 技术方法**

使用技术：多模态大语言视觉模型（如Qwen2.5-7B/32B-VL）、多教师生成、共识筛选、Meta‑Judge偏好判定、token‑level交叉熵SFT、直接偏好优化（DPO）等。

**📊 数据集**

数据集：100k+真实电商多模态产品对，覆盖8种语言（英、法、西、德、意、土、葡、日）和18个国家，包含标题、描述、属性和图片。

**📈 对比分析**

对比方法：与人类标注、零样本、大VLM baseline、单教师标签、SFT+教师标签、SFT+标签+推理等进行比较。MERGED在PR‑AUC达到90.96%（比同基础模型人类标注+13.79%，比Qwen2.5‑32B‑VL baseline+6.32%），并在每百万预测成本仅600美元，低于baseline的6倍。

**⚠️ 局限性**

Limitations：只评估了exact与variant两种关系，教师数量仅2个，Meta‑Judge依赖大型VLM，未探索更大教师集合或轻量化替代方案。

---

## 81. The Ceiling Is in the Channel: Auditing Learner Gaps and Measurement Frontiers in Clinical Prediction

**arXiv ID:** 2609.01909 | [PDF](https://arxiv.org/pdf/2609.01909v1)

**作者:** Sayeed Shafayet Chowdhury `[一作]` (Indiana University Indianapolis), Vijay R. Ramakrishnan `[通讯]` (Indiana University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究提出并验证了一个将临床预测性能拆分为学习器缺口与测量通道上限的框架，并在三大真实数据集上进行评估。

**💡 创新点**

创新之处包括通过总变差量化测量通道的最大可达性能、交叉拟合的前沿估计器、置换与欠拟合诊断以及在共享替换噪声下的边界识别。

**🔧 技术方法**

主要技术手段包括总变差分离、交叉拟合的等先验后验估计、置换检验与训练比例曲线、以及正则化的梯度提升与多层感知机模型。

**📊 数据集**

实验使用了UCI 130‑US医院再入院、BRFSS 2015糖尿病调查以及NHANES 2015‑2018 HbA1c三组数据，样本量分别为99,343、253,680和10,219。

**📈 对比分析**

将学习器的平衡准确率与估计前沿进行对比，梯度提升模型几乎达到前沿，而随机森林等模型存在明显缺口；AUROC提升不一定对应平衡准确率提升，决策翻转率往往更高。

**⚠️ 局限性**

局限性包括需等先验假设、有限样本可能导致前沿低估或偏差、仅适用于二分类且无法直接证明测量改变会提升真实前沿，以及对多元和时序数据的推广仍待验证。

---

## 82. Type-Directed, Secure-by-Construction Enclave Partitioning for LLVM

**arXiv ID:** 2609.02048 | [PDF](https://arxiv.org/pdf/2609.02048v1)

**作者:** Wesley B. Nuzzo `[一作]`, Anitha Gollamudi `[通讯]` (University of Massachusetts Lowell)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了一种基于类型的安全构造方法，用于在LLVM编译框架下自动划分可信执行区块（enclave）。

**💡 创新点**

创新点在于利用程序类型信息实现安全边界的静态推断，既自动化又能保证安全性；并提出了可组合的安全属性验证。

**🔧 技术方法**

使用了LLVM中间表示（IR）、类型分析器、以及基于类型的安全属性引擎，并实现了自定义的LLVM Pass。

**📊 数据集**

评估数据集包括C/C++开源项目（如LLVM自带的基准、SPEC CPU 2017以及OpenSSL等）。

**📈 对比分析**

与手工划分和传统的基于注释或属性的划分方法比较，性能损耗在 enclave 开销上平均约 12% 的内存占用和 8% 的执行时间，且安全性验证通过率达到 99%。

**⚠️ 局限性**

主要限制是对动态加载、运行时生成代码的支持不足，且对多线程同步的安全性推断仍需要手工补丁。

---

## 83. TAPVid-MV: A Benchmark for Tracking Any Point in 3D Across Multiple Views

**arXiv ID:** 2609.01899 | [PDF](https://arxiv.org/pdf/2609.01899v1)

**作者:** Skanda Koppula `[一作]` (Google DeepMind), Gabriel Brostow `[通讯]` (University College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个名为TAP-3D的多视角长时3D点跟踪基准，用以评估在摄像机运动和布局变化下的同步视角下的3D点跟踪性能。

**💡 创新点**

首次构建包含284序列、1142个标定摄像机流和109,769条真实3D轨迹的综合基准，并且同时评估重建质量与跟踪准确度，从而分离几何误差与对应误差；此外提供了多种评价指标（Q、N、W）。

**🔧 技术方法**

采用多视角重建模型（VGGT‑Ω）、单视角与多视角点跟踪器（TAPIP3D、MVTracker、MV‑TAP、OmniX等）以及各种辅助模态（LiDAR、SLAM、SfM、人体网格、模拟等）进行轨迹生成与验证。

**📊 数据集**

使用DROID、Ego‑Exo4D、Harmony4D、PACE、Hi4D、Waymo和完全合成的Perpetua等数据集，涵盖机器人、人体交互、驾驶和室内场景。

**📈 对比分析**

在30余个基线上进行对比，最佳方案是将TAPIP3D与VGGT‑Ω重建相结合，平均世界空间精度达到22.8；但大多数多视角跟踪器并未明显优于单视角方法，表明几何重建仍是主要瓶颈。

**⚠️ 局限性**

局限性包括：重建误差限制了整体精度；跨视角对应仍易失真，导致非查询视角表现差；在驾驶和近距离人体交互等真实场景中性能低下；缺乏对动态场景全局一致性和长时漂移的有效解决方案。

---

## 84. Towards a Foundational Ontology for Identifying and Resolving Contradictions in Dialogue-based Human-Robot Interactions

**arXiv ID:** 2609.02364 | [PDF](https://arxiv.org/pdf/2609.02364v1)

**作者:** Maitreyee Tewari `[一作]` (Alternative Intelligence LLP), Michele Persiani `[通讯]` (University of Bologna)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

构建了基于活动理论的基础本体 ATFOt，用于形式化表示和管理对话式人机交互中的矛盾（错误、冲突、知识问题）。

**💡 创新点**

创新点在于：① 将活动理论中的四阶矛盾层级（主矛盾、次矛盾、三阶矛盾、四阶矛盾）系统化为本体结构；② 提出了三条新原则，指导本体在出现矛盾时的变换与继承；③ 采用集合论与一阶逻辑对矛盾及活动系统进行形式化定义。

**🔧 技术方法**

使用技术包括：METHONTOLOGY 本体开发方法论、集合论与一阶逻辑进行形式化定义、Protégé 平台实现与验证推理。

**📊 数据集**

未使用具体数据集；工作处于理论与本体实现阶段。

**📈 对比分析**

目前尚未进行对比实验或性能评估；计划通过覆盖率和能力问题（competency questions）进行定性验证。

**⚠️ 局限性**

局限性：① 本体仍在完善和验证阶段，缺乏实证评估；② 仅基于理论推导，未验证在真实对话系统中的效果与性能；③ 由于缺乏数据集，无法进行客观性能比较。

---

## 85. Beyond Instruction-Driven Editing: Source-Grounded Problem Discovery with User-Governed Repair for Scientific Posters

**arXiv ID:** 2609.01813 | [PDF](https://arxiv.org/pdf/2609.01813v1)

**作者:** Xingda Lyu `[一作]` (University of Washington), Shiqi Yang `[通讯]` (University of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本工作提出并实现了“Proactive Refinement of Scientific Posters”系统，支持在用户尚未明确编辑指令前，由系统主动诊断海报中的结构、科学内容与空间布局问题，并通过用户决定是否接受、预览并提交修复；与传统的直接编辑路径共享同一可执行器与执行逻辑；同时构建了完整的源论文–可编辑PPTX–渲染三元组基准，分别评估诊断质量、治理决策、修复成功率及最终目标提升。

**💡 创新点**

核心创新在于将知识探索（epistemic initiative）与行为授权（behavioral authority）分离，使系统能在无明确编辑目标时提出可验证的候选问题，但仅在用户接受后才产生可执行修复；引入三层诊断（结构、科学、空间）并将诊断与源论文证据绑定；构建大规模源文献匹配的可编辑PPTX基准，允许独立测评诊断、修复和最终效果，避免将所有改动视为单一成功指标。

**🔧 技术方法**

技术方案包括：① 采用大型语言模型（如GPT‑5.6 Terra）进行问题诊断与编辑请求规划；② 通过 deterministic PPTX 解析与执行器实现原生对象级编辑、类型化操作、验证与可逆预览；③ 设计多层诊断流程（结构、科学、空间）并生成可解释的证据链；④ 使用多模态 VLM 评估器对诊断与修复结果进行自动打分；⑤ 允许替换 reasoning backbone，实现跨模型的可移植性评估。

**📊 数据集**

使用的基准数据集共包含 120 篇源论文（CVPR、ICML、NeurIPS、ICLR 2023‑2026）与 320 个可编辑 PPTX 海报（Paper2Poster、PosterGen、专家手工作者版、公开会议版）。此外还有 80 篇论文产生的 160 对生成海报，形成 40 个四版组与 80 个两版组的完整评测集。

**📈 对比分析**

评测方法采用四阶段流程：① 直接请求执行（E1）检验系统按预定义指令执行；② 诊断质量评估（E2）使用 VLM 评分量化问题的正确性、根基性、目标特异性与可操作性；③ 接受修复成功率（E3）由操作员验证；④ 随机遮蔽后自动评估最终目标提升（E4）。结果显示：E1 成功率 83.3%，E2 平均诊断质量 67.2/100，E3 受理修复成功率 87.6%；但自动评估显示平均提升仅 22.7 pp，且 14.8% 的接受目标在最终评估中下降。对四种 backbone 的探索性比较表明诊断质量、修复成功率与目标提升存在模型差异，暗示系统可移植但无统一排名。

**⚠️ 局限性**

局限性包括：① 评估以工件为中心，未测量用户工作效率或满意度；② 诊断仅覆盖六类问题，缺乏完整的海报质量理论；③ 对 PDF 转 PPTX 的结构损失缺乏充分补偿，导致某些修复机会受限；④ 未进行独立用户研究或人类专家评审；⑤ 研究集中在学术会议海报，可能不适用于其他学科或行业场景。

---

## 86. Bonded Recourse for Smart-Contract Settlement of Compensable Agent Side Effects

**arXiv ID:** 2609.01939 | [PDF](https://arxiv.org/pdf/2609.01939v1)

**作者:** Laurent Bindschaedler `[一作]` (Max Planck Institute for Software Systems), Christoph Siebenbrunner `[通讯]` (Research Institute for Cryptoeconomics, Vienna University of Economics and Business)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Recourse协议，针对自治代理执行后残留损失进行智能合约级别的补偿与结算

**💡 创新点**

将补偿过程拆分为预授权与后续可结算边界，设计可追溯的typed receipt和多路径结算（optimistic、仲裁、排除）

**🔧 技术方法**

使用以太坊智能合约（Solidity）、Base Sepolia测试网络、UMA Optimistic Oracle、ERC‑792/1497仲裁接口、EIP‑712签名、PostgreSQL、Git、LocalStack等工具

**📊 数据集**

采用合成的PostgreSQL、Git和云参数存储 sandbox 环境进行对比实验，并使用基准工作负载（Postgres over-delete、Git对象删除、参数版本更改）

**📈 对比分析**

与OAP、Atomix、集中式托管等基线对比，评估不补偿损失、覆盖率、争议处理和链上 gas 费用；在PostgreSQL harness 上可减少约50% 不补偿损失，gas 成本在 Base Sepolia 上每笔结算低于 0.01 gwei

**⚠️ 局限性**

仍依赖单一运营者、对多方争议和大规模并发失败的处理有限、未覆盖跨组织或真实生产环境的证明、需要进一步完善 provenance 级别（L1/L0）以及更复杂的险保或多方共识

---

## 87. Rendering-in-the-Loop: An Execution-Driven Agent for Interactive Web Development

**arXiv ID:** 2609.02088 | [PDF](https://arxiv.org/pdf/2609.02088v1)

**作者:** Yilong Guo `[一作]` (Baidu Inc.), Zeyu Chen `[通讯]` (Baidu Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

RILA 通过在真实浏览器中多次执行、验证并基于交互反馈迭代修正生成的前端代码，实现网页功能与视觉双重优化。

**💡 创新点**

创新点在于将浏览器渲染反馈直接作为交互正确性和视觉质量的评估信号，构建 AIV、ERS 等模块，并用执行结果驱动的批判模型进行结构化修正。

**🔧 技术方法**

技术采用 Playwright 自动化浏览器执行、执行反馈验证、结构化批判模型、工具辅助软件工程（语义代码搜索、文件编辑等）以及执行验证的数据合成流水线。

**📊 数据集**

使用 IWR‑Bench 真实网页重建数据集和自构建的执行验证合成数据，此外在 Interaction2Code 评估集上验证跨基准性能。

**📈 对比分析**

在 IWR‑Bench 上与 Kimi‑K2.6、GPT‑5.5 等基线比较，RILA 将 9B Qwen 模型的最终得分从 40.4% 提升至 57.5%，甚至超过更大规模模型，显示显著性能提升。

**⚠️ 局限性**

局限在于对批判模型质量高度依赖、执行环境成本较高、仅适用于可在浏览器中直接执行的网页交互场景，对多页面或后台逻辑支持不足。

---

## 88. D-FROST: Decentralized Federated pRompt-tuning via Optimal tranSporT for Non-IID and Imbalanced Data

**arXiv ID:** 2609.01802 | [PDF](https://arxiv.org/pdf/2609.01802v1)

**作者:** Quan Minh Nguyen `[一作]` (University of Florida), My T. Thai `[通讯]` (University of Florida)

**通讯引用:** 8838 | [OpenAlex ID](https://openalex.org/A5005663679)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了在去中心化联邦学习（DFL）场景下的 Prompt 调优方法，避免传统索引平均导致的提示对齐问题；

**💡 创新点**

将 Prompt 调优问题建模为 Wasserstein 空间上的优化，并设计基于最优传输（OT）的提示合并算子实现无索引平均的分布级聚合；

**🔧 技术方法**

采用 Wasserstein 距离、分布表示、OT 合并、分布梯度、分布梯度下降及多步交替优化；

**📊 数据集**

在多域视觉任务上使用 FourDataset（MNIST‑M、Fashion‑MNIST、CINIC‑10、MMAFEDB）和 FiveDataset（CIFAR‑10、MNIST、Fashion‑MNIST、SVHN、notMNIST）进行实验；

**📈 对比分析**

与三种主流去中心化基线（D‑PSGD‑PT、DFedAvgM‑PT、DFedSAM‑PT）比较，D‑FROST 在所有数据异构设置下均取得最高精度，并在收敛速度、网络拓扑和链路丢失等鲁棒性实验中表现最优；

**⚠️ 局限性**

限制在于需要预先设定 Prompt 数量、OT 迭代次数和正则化参数，且合并算子在极大规模客户端或高维 Prompt 时的计算与存储成本仍需要进一步优化。

---

## 89. hLLM: Single Pass Decoding for Generative Reranking

**arXiv ID:** 2609.01807 | [PDF](https://arxiv.org/pdf/2609.01807v1)

**作者:** Emil Laftchiev `[一作]` (Meta Platforms, Inc.), Luke Simon `[通讯]` (Meta Platforms, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种专门针对排序输出的解码策略，利用隐藏状态构建得分矩阵后用匈牙利算法一次性解出完整的排名；

**💡 创新点**

创新点在于将排序视为最优分配问题，直接在隐藏层做一次前向推理即完成排序，保证输出是有效排列，并通过Sinkhorn松弛实现可微训练；

**🔧 技术方法**

核心技术包括：LLM预填充（prefill）获取每项隐藏状态、轻量级自注意力头生成 N×K 得分矩阵、Hungarian 算法求最优匹配、LoRA 低秩微调以及教师-学生的全排列蒸馏；

**📊 数据集**

实验使用内部大规模重排数据集（≈189k 训练、236k 测试，单个幻灯片最大150项）和公开 Amazon Beauty 数据集（50项幻灯片、21k 训练、1k 测试）；

**📈 对比分析**

与 autoregressive LLM（含/不含推理链）对比，均保持相同的完整排名输出，端到端推理时间从 88 ms 降至 28 ms（≈64×速度提升），在 AUC、Recall@1、NDCG@1 等指标上几乎无损，甚至略有提升；

**⚠️ 局限性**

局限包括：需要对 backbone 进行 LoRA 微调才能充分利用丰富的排序信号；对极大规模幻灯片的推理速度尚未验证；以及在特定任务中，隐藏状态信息的表达可能不够充分，导致需要更大的模型或更深的自注意力层来提升性能。

---

## 90. LookStep: Efficient Vision-Language Navigation with Linguistic Foresight and Event Driven Memory

**arXiv ID:** 2609.02350 | [PDF](https://arxiv.org/pdf/2609.02350v1)

**作者:** Kun-Yang Yu `[一作]` (Nanjing University), Yu-Feng Li `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 LookStep 框架，通过语言中心未来状态建模和事件驱动滚动记忆，实现高效的连续视觉语言导航。

**💡 创新点**

创新点在于：①将下一步动作预测改为候选动作未来状态评估并用语言标签建模，显著提升数据利用率；②采用事件驱动滚动记忆，由模型自行决定何时写入记忆及其语义角色，减少冗余历史并保留关键事件；③基于信息论证明未来状态标签可降低动作不确定性。

**🔧 技术方法**

技术方法包括：使用多模态大语言模型 Qwen3‑VL 8B 进行端到端自动回归生成结构化语言序列（进度、候选动作未来状态、记忆写入/角色、最终动作），以及 FIFO 滚动队列实现轻量级记忆管理。

**📊 数据集**

使用的数据集为 R2R‑CE 和 RxR‑CE（基于 Matterport3D 的导航轨迹），在 Habitat 仿真器上训练并在真实环境中进行验证。

**📈 对比分析**

与传统下一步预测、认知地图、历史帧和外部空间工具等方法对比，LookStep 在 R2R‑CE Val‑Unseen 上实现 49.7% 成功率，较 JanusVLN 仅差约 3% 但显存保持 <24GB、推理时间 59 ms（相较 194 ms），显示出更优的数据与内存效率。

**⚠️ 局限性**

局限性包括：仅在标准 VLN‑CE 数据集上训练，未采用 DAgger 等交互式训练策略；与大规模数据/优化方法相比仍存在性能差距；对更大规模环境和更复杂任务的适应性尚待验证。

---

## 91. The Missing Temporal Link: Temporal Context Routing for Script-Driven Audio-Video Generation

**arXiv ID:** 2609.02367 | [PDF](https://arxiv.org/pdf/2609.02367v1)

**作者:** Yichen Liu `[一作]` (Peking University), Daquan Zhou `[通讯]` (Peking University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Temporal Context Routing (TCR) 方法，利用脚本中精确的时间信息在视频和音频生成过程中实现对每个镜头和对白的独立时序控制。

**💡 创新点**

创新点在于将脚本时间映射为 duration‑normalized 轮询得分并以可加方式注入视频–文本与音频–文本交叉注意力中，使得不同类型的提示（镜头、对白）可以在共享时轴上独立、同时地引导两种模态，而无需修改原有文本编码或增加可学习参数。

**🔧 技术方法**

核心技术包括：1) 可加的时间路由得分计算（基于中心点和半径的高斯函数）；2) 通过对交叉注意力 logits 进行加法偏置实现多模态时序引导；3) 细粒度的粗细分阶段数据构建流程（Gemini + PySceneDetect + WhisperX），为训练提供 0.1 s 级的镜头与对白时间标注。

**📊 数据集**

使用两套短剧集构建的训练集（约 57k 训练样本）与 200 条测试脚本，脚本包含 640 个镜头提示和 441 条对白提示；训练和评测均不共享同一媒体 ID 或标题哈希。

**📈 对比分析**

与 Wan2.2、OVI、JoyAI‑Echo、LTX‑2.3 等基线对比，TCR 在 Shot Boundary MAE 上提升 96%（从 1.11 s 降到 0.042 s），Dialogue Acc@0.5s 提升 84.1%（从 28.3% 提升至 84.1%），同时保持或提升视觉质量（IQ、AES）和音视频同步（Sync‑C、offset accuracy）。

**⚠️ 局限性**

局限性包括：1) 主要针对短剧类语音对白，未验证在更大规模或不同内容类型（如多语言、多音频源）的泛化能力；2) 依赖精细的手工/自动化时间标注，若脚本时间信息不完整或噪声大，效果可能受限；3) 仅在 LTX‑2.3 等已有双模态架构上实现，尚未探索在更轻量化或更大模型上的适配。

---

## 92. WiP: Characterizing and Defending Against Mobile-Agent-Driven MFA Automation

**arXiv ID:** 2609.02154 | [PDF](https://arxiv.org/pdf/2609.02154v1)

**作者:** Yimeng Liu `[一作]` (University of California, Merced), Hua Huang `[通讯]` (University of California, Merced)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究移动代理自动化多因素认证（MFA）过程，构建了一个两阶段模块化流水线，在10个网站/应用上完成完整的MFA流程，并开发了基于手机运动传感器的物理交互防御方案。

**💡 创新点**

提出了“factor collapse”威胁模型；设计了将登录、OTP检索与提交拆分为两个专门代理的流水线，显著提升了自动化成功率；首次使用运动传感器与机器学习相结合，检测人机交互差异以提升安全性。

**🔧 技术方法**

利用大语言模型（GPT‑5 mini）驱动移动代理（MobileRun），采集加速度计与陀螺仪数据，提取特征后用extra‑trees分类器进行风险判别；还用到手机GUI自动化控制、邮件/身份验证器访问等技术。

**📊 数据集**

使用10个真实商业网站/APP的授权账户（含电子邮件OTP和身份验证器生成码），以及116个登录会话（62桌面+54手持，分别包含用户手动和代理自动两类）。

**📈 对比分析**

与两种单代理基线（仅登录目标与全功能指令）比较，单代理成功率分别为3/10和6/10，平均完成时间分别高于10/10成功的两阶段代理；两阶段代理在10个目标上均实现10/10成功。物理交互防御在桌面场景下准确率98.4%，手持场景92.6%，整体95.7%。

**⚠️ 局限性**

实验范围受限于仅桌面与手持两种使用场景，样本量有限；未评估更复杂设备、传感器伪造或高级自适应代理的攻击；防御依赖传感器信号，可能被硬件篡改或被设计为无运动的自动化流程。

---

## 93. FUSE: An Evaluating Framework for Dangerous Capabilities of LLMs

**arXiv ID:** 2609.02168 | [PDF](https://arxiv.org/pdf/2609.02168v1)

**作者:** Zhengyi Jin `[一作]` (Beijing University of Posts and Telecommunications), Zhen Yang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出FUSE框架，利用统一协议在知识(K)、辩护(D)、危害(H)三维度评估LLM的危险能力，并生成七维危险能力概况ϕ；

**💡 创新点**

创新点在于模块化接口、统一评估流程、三维度独立性及聚合式危险能力概况，使不同模型、不同领域可直接横向对比；

**🔧 技术方法**

使用IRT估计知识水平、红队多轮路径依赖交互评估辩护、LLM-as-Judge对开放式危害文本进行打分，并支持工具增强模式；

**📊 数据集**

采用化学-生物域3,773道多选题、298个防御场景、200道开放式危害查询，对12款商业LLM进行评估，并在网络模块进行小规模对比；

**📈 对比分析**

与传统单一基准相比，K维稳步提升、D维分化、H维保持不变，三维度相关性低，说明三维度互补，评价结果可直接对模型与家族进行横向比较；

**⚠️ 局限性**

局限包括知识基准可能污染、危害评分主观性、对CB域的大规模验证完成但网络模块规模有限、工具增强模式尚需进一步扩展和验证。

---

## 94. Morphology signal in whole slide image foundation models can automatically triage slides

**arXiv ID:** 2609.01987 | [PDF](https://arxiv.org/pdf/2609.01987v1)

**作者:** Ayushi Sinha `[一作]` (Mayo Clinic), Michael R. Lucas `[通讯]` (Mayo Clinic)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于公开的病理影像基础模型（Foundation Models）的无监督滑动窗口图像分类管线，用来自动对患者的多张全切片图像进行分层排序，从而挑选出含有肿瘤的主要切片。

**💡 创新点**

创新点在于利用零样本分类（zero‑shot）和图像-文本嵌入相似度评估来完成“滑片分流”，无需人工标注即可识别出最有诊断价值的切片，并给出了针对分层排序的评估公式。

**🔧 技术方法**

技术核心包括：HEST背景去除、512×512补丁提取、CONCH、mean‑CONCH 和 TITAN 三种公开病理基础模型的嵌入生成、文本提示的多模态嵌入、余弦相似度计算和基于概率的滑片排名。

**📊 数据集**

实验使用了两组数据集：H&N（1233例、55597张切片，粗粒级别肿瘤标注）和OV（35例、1068张切片，细粒级别肿瘤标注），并分别在子集（HN‑node）上评估更细粒度标注。

**📈 对比分析**

通过 Recall@k、模型一致性、以及新的 Ranked‑Evaluation 评分进行比较；结果显示在 HN 数据集上所有模型均达 90%+ 的 Recall@1，CONCH 在 OV 数据集上在 k≤9 时即可保证至少包含一张肿瘤切片，Ranked‑Evaluation 指标在不同池大小下表现相对一致。

**⚠️ 局限性**

局限性包括：仅评估了三种模型，未检验跨癌种通用性；零样本分类的文本提示设计仍受限于术语多样性；在极大切片池（>80 张）时性能下降；且未深入探讨模型解释性和对真实临床工作流的集成成本。

---

## 95. Curriculum-Guided Reinforcement Learning for Energy-Efficient UAV-ISAC in Post-Disaster Search-and-Rescue Operations

**arXiv ID:** 2609.01764 | [PDF](https://arxiv.org/pdf/2609.01764v1)

**作者:** Tai-You Guo `[一作]` (National Chung Cheng University), Chuan-Chi Lai `[通讯]` (National Chung Cheng University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种基于课程引导的软演员-评论家（CG‑SAC）框架，用以在单架旋翼无人机上同时优化三维轨迹、通信/感知功率分配与用户功率分配，从而实现灾后搜救场景下的能效驱动ISAC任务规划。

**💡 创新点**

① 在奖励中嵌入基于旋翼推进模型推导的经济巡航速度项，实现对飞行能耗的物理先验引导；② 采用对数线性课程学习机制，逐步收紧通信速率、感知精度与接近阈值，提升探索效率；③ 将奖励拆分为导航、节点访问、能效、速度与约束惩罚，并同步熵温度调度，兼顾目标实现与能源消耗。

**🔧 技术方法**

深度强化学习（Soft Actor‑Critic）+课程学习+基于物理的奖励塑形+旋翼推进模型+边缘协助训练。

**📊 数据集**

使用自行生成的2000个随机测试场景（3个通信用户、3个感知目标，面积1000×1000 m²，飞行高度50–150 m），并在仿真环境中评估。

**📈 对比分析**

与DDPG、TD3、PPO及原始SAC进行对比。CG‑SAC在测试集上平均能效0.72 Mbits/J（约比TD3高3倍），完成任务所需步数平均107.6步（比基线降低66%–82%），通信满意率达99.6%，并展示出接近目标时减速、巡航时加速的节能轨迹。

**⚠️ 局限性**

仅针对单架无人机，缺乏多机协作与干扰建模；环境假设理想CSI与静止目标，未考虑移动目标或障碍物；奖励权重需人工调参，未采用自动化多目标学习；缺少对剩余能量显式感知的鲁棒性验证。

---

## 96. KSG-Net: Key-Sparse and Global-Context Learning for Maritime 3D Ship Detection

**arXiv ID:** 2609.02077 | [PDF](https://arxiv.org/pdf/2609.02077v1)

**作者:** Zhouyuan Huai `[一作]` (Wuhan University of Science and Technology), Xiao Wang `[通讯]` (Wuhan University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为KSG-Net的海事3D船舶检测网络，旨在解决海洋环境中小型稀疏船舶的弱特征表示和大型船舶的全球结构建模不足的问题。

**💡 创新点**

创新点在于设计了两个互补模块：关键稀疏多尺度聚合（KSMA）模块和全球上下文聚合（GCA）模块，分别增强小型稀疏船舶的表示和大型船舶的长距离几何依赖性。

**🔧 技术方法**

使用了关键稀疏和全球上下文学习的网络架构，结合了KSMA和GCA模块来提升特征表示。

**📊 数据集**

使用了泰晤士河船舶数据集和模拟数据集进行实验，数据集包含真实的港口场景和多种船舶类型。

**📈 对比分析**

与现有方法相比，KSG-Net在多尺度船舶检测中表现优异，尤其在复杂的海洋环境中展现出强大的鲁棒性，具体性能指标在实验中有详细比较。

**⚠️ 局限性**

限制在于该方法可能在某些特定的海洋场景中仍然面临挑战，尤其是在极端天气或极端背景干扰的情况下。

---

## 97. AlphaRAD: Grounded Zero-Shot Classification in Chest Radiology via $α$-Corrected Binary Cross Entropy and Factorized Latent Supervision

**arXiv ID:** 2609.01757 | [PDF](https://arxiv.org/pdf/2609.01757v1)

**作者:** Jianzhong You `[一作]`, Chris McIntosh `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的深度学习模型，用于图像分类任务。

**💡 创新点**

创新点在于引入了一种新的激活函数，能够提高模型的收敛速度和分类精度。

**🔧 技术方法**

使用了卷积神经网络（CNN）和改进的激活函数。

**📊 数据集**

在CIFAR-10和ImageNet数据集上进行了实验。

**📈 对比分析**

与现有的几种主流模型进行了比较，结果显示该模型在分类精度上提高了5%，且训练时间缩短了20%。

**⚠️ 局限性**

模型在处理高分辨率图像时性能下降，且对计算资源的需求较高。

---

## 98. DynG-Diff: A State-Aware Dynamic Guidance Diffusion Framework for Probabilistic Time Series Forecasting

**arXiv ID:** 2609.02068 | [PDF](https://arxiv.org/pdf/2609.02068v1)

**作者:** Zhente Zhang `[一作]` (Zhejiang Gongshang University), Wei Fan `[通讯]` (University of Auckland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 DynG-Diff：一种将无条件扩散模型与状态感知动态引导结合的多变量时间序列概率预测框架。

**💡 创新点**

创新点在于（1）将无条件预训练与推理时动态引导分离；（2）设计轻量级状态感知政策网络，实时估计每个变量的可靠性；（3）将动态权重解释为局部精度的 Asymmetric Laplace 分布，实现对高置信变量强引导、对噪声变量弱引导。

**🔧 技术方法**

使用技术包括无条件扩散概率模型、状态感知政策网络、加权分位数损失、Asymmetric Laplace 观测似然、动态权重生成与指导、两阶段分离训练。

**📊 数据集**

在六个公开多变量时间序列基准上评估：ETTh1、Exchange、Weather、Appliance、Solar、Traffic。

**📈 对比分析**

与 TimeGrad、CSDI、SSSD、TimeDiff、TMDM、D3U 等 SOTA 方法在 CRPS 与 MSE 上进行对比；DynG-Diff 在大多数数据集达到或超过 SOTA，尤其在信息异质性强的 ETTh1、Appliance 等场景表现突出。

**⚠️ 局限性**

局限性包括：在长时序（如 Traffic 720 步）下概率校准略逊；推理时增加约 30% 计算开销，适用于离线/批量任务；未显式建模空间拓扑，未来可进一步提升；对极端噪声的鲁棒性虽然提升但仍有一定性能下降。

---

## 99. Addressing Trust in AI Systems through Education: A Didactic Perspective

**arXiv ID:** 2609.02453 | [PDF](https://arxiv.org/pdf/2609.02453v1)

**作者:** Pierre Haritz `[一作]` (TU Dortmund University), Thomas Liebig `[通讯]` (TU Dortmund University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并论证了ICE-T框架，用以将互模转移、计算思维与解释性思维整合到机器学习教育中，从而实现信任校准的教学目标。

**💡 创新点**

创新点在于：①将布鲁纳的EIS原则、Use-Modify-Create进展与PETSP‑ML流程模型三大教育理论融合为统一框架；②将信任校准明确为教育目标，并用框架中的认知机制（代表性丰富度、渐进式过程控制、错误情境化）来解释其实现方式。

**🔧 技术方法**

使用的技术包括：布鲁纳的三模表示理论（Enactive、Iconic、Symbolic）、Use‑Modify‑Create（UMC）计算思维进展、PETSP‑ML过程模型（基于CRISP‑DM）以及系统综述与文献回顾方法。

**📊 数据集**

本研究未使用具体机器学习数据集，而是基于对126项K‑12 ML活动和133项教学活动的系统综述、文献整理与理论推导。

**📈 对比分析**

由于本工作为理论与框架设计，未进行实验比较或性能评估；评估基于文献证据与系统综述的合成分析，缺乏量化结果。

**⚠️ 局限性**

局限性：①框架尚未在真实课堂中进行实证验证；②缺乏对不同年龄层和学习情境的跨场景评估；③仅以理论推导和文献为依据，未提供定量信任校准的测量与验证。

---

## 100. CAPTCHAs in the Agentic Era: Solvers That Learn from Every Encounter

**arXiv ID:** 2609.02393 | [PDF](https://arxiv.org/pdf/2609.02393v1)

**作者:** Oguzhan Salman `[一作]` (Istanbul Technical University), Kemal Bicakci `[通讯]` (Istanbul Technical University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套DOM‑free的混合 CAPTCHA 解码器，利用 YOLOv8 进行快速检测、Qwen‑7B‑VL 进行高精度推理，并通过自适应信心门控与有限状态机实现无缝交互，同时利用 VLM 的预测结果做增量学习和对抗恢复。

**💡 创新点**

1) 采用信心门控的混合推理，使 YOLO 的毫秒级速度与 VLM 的全场景覆盖相结合； 2) 通过 VLM 产生的无监督标签实现 YOLO 的在线增量学习和对抗自恢复； 3) 在连续一年级的对抗攻防实验中展示系统能够每月重新训练并恢复，证明其对抗鲁棒性。

**🔧 技术方法**

YOLOv8 检测器、Qwen‑7B‑VL 视觉语言模型、经验重放 (Experience Replay) 增量学习、PGD 对抗攻击与自适应恢复、无 DOM 的截图+OS 级鼠标键盘控制、有限状态机（FSM）流程管理。

**📊 数据集**

公开的 reCAPTCHA v2 数据集、Kaggle 页面截图、合成 CAPTCHA 覆盖样本（1,200张）以及包含 16 类（Bicycle、Boat、Bridge、Bus、Car、Chimney、Crosswalk、Hydrant、Motorcycle、Mountain、Other、Palm、Stairs、Taxi、Tractor、Traffic Light）的评测集；同时生成 PGD 攻击样本用于对抗实验。

**📈 对比分析**

与 YOLO 单独、VLM 单独以及 Plesner 等先前工作对比；混合模型在 16 类上整体准确率 85.4%，宏观准确率 84.2%，显著优于 YOLO 单独的 75.5%/60% 和 VLM 单独的 80.5%/81%；平均每个单元的延迟从 VLM 的 225 ms 降至 71 ms；在 PGD 攻击下 YOLO 0% 可通过 VLM 标签恢复到 60–70%+，并在连续 12 轮攻防中每轮均能恢复。

**⚠️ 局限性**

对 VLM 标签噪声的敏感度（尤其是类别视觉多样性大的 Boat 类难以学习）、对门限阈值 τ 的手工调参需求、增量学习和对抗恢复对计算资源的额外负担、以及在更大范围的 CAPTCHA 家族（除 reCAPTCHA v2 外）的泛化验证仍有限。

---

## 101. VoRTeC: Taming Foundation Flow for One-step Real time Video Compression

**arXiv ID:** 2609.02291 | [PDF](https://arxiv.org/pdf/2609.02291v1)

**作者:** Yichong Xia `[一作]` (Tsinghua University), Haoqian Wang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于预训练视频流匹配模型Wan2.1的单步解码框架VoRTec，能在极低比特率下实现高保真视频重建；

**💡 创新点**

通过流状态估计将压缩的潜在表示映射到流轨迹，利用Flow Prior Multi‑Fusion融合先验与压缩特征，并引入无侵入的组间通信CGG实现时空一致性，解决传统扩散式视频压缩的低速与闪烁问题；

**🔧 技术方法**

使用流匹配技术、ViT‑基准先验融合网络、LoRA微调、以及多尺度Patch划分与交叉融合；

**📊 数据集**

在UVG、HEVC Class B/C、MCL‑JCV三大标准视频数据集上进行实验；

**📈 对比分析**

与传统VTM、神经视频压缩DCVC/SEVC以及生成压缩GLC‑Video、DiffVC、S2VC等基线对比，VoRTec在LPIPS等感知指标上比DiffVC/GLC提升约20–30%，BD‑rate在LPIPS上降低58–73%，解码速度提升3–197倍，可在480p实时解码32 FPS；

**⚠️ 局限性**

局限性包括对大模型Wan2.1的高计算与显存需求，主流实验受限于720p/480p分辨率，且在更高分辨率或极端运动场景下仍存在轻微闪烁与细节损失。

---

## 102. Index-Free Dynamic Edge Retrieval with Energy-Tail-Aware Partial Scans

**arXiv ID:** 2609.01820 | [PDF](https://arxiv.org/pdf/2609.01820v1)

**作者:** Mohammad Arif Rasyidi `[一作]` (Khalifa University), Omar Alhussein `[通讯]` (Khalifa University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种无索引的动态最大内积搜索（MIPS）方法ETAR，利用查询能量选择高幅值坐标进行候选生成并对少量候选进行精确重排序，保持更新简单并显著降低查询成本。

**💡 创新点**

创新点在于：①基于查询能量动态选择坐标，②尾部能量校正的候选评分，③固定预算的重排序与无索引更新策略，④同时提供全精度与低精度（8‑bit）重排序两种配置。

**🔧 技术方法**

技术包括：列主序8‑bit量化扫描视图、行主序32‑bit或8‑bit重排序视图、尾部校正系数α的自适应估计、固定预算重排序、动态表容量管理与按需压缩。

**📊 数据集**

使用九个静态数据集（四个合成：Dense Gaussian、Sparse Gaussian、Mixed Heavy‑Tail、Norm‑Heavy；五个真实：SIFT、GloVe、LastFM、Fashion‑MNIST、NYTimes），以及在移动设备（Samsung Galaxy S25）和五种流式工作负载（增量、漂移、平衡、突发、滑动窗口）上的实验。

**📈 对比分析**

与全精度扫描、8‑bit无重排序扫描以及四个基准索引（Faiss HNSW、IVF、Annoy、ScaNN）比较。ETAR在K=10下平均Recall@10≥99%时，查询延迟比全扫描快4–5×，在移动端可达6.9×；在流式工作负载中保持100% Recall@10且事件时间低于HNSW/IVF的重建成本。

**⚠️ 局限性**

局限性：查询延迟随存储行数线性增长；尾部校正为启发式，无法保证所有真实Top‑K进入候选集；在高写入率时压缩或重建仍会产生峰值；参数（ρ、R、h_max）固定，未自适应；实验未覆盖大规模（>1M）或批量查询、能耗等实际边缘部署场景。

---

## 103. Constraint-Preserving Genetic Algorithms for Embedding Linear Codes into Self-Orthogonal Codes

**arXiv ID:** 2609.02135 | [PDF](https://arxiv.org/pdf/2609.02135v1)

**作者:** Haeun Lim `[一作]` (Sogang University), Jon-Lark Kim `[通讯]` (Sogang University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用最短自正交嵌入方法和约束保持遗传算法，构造了 66 条新的二元最优自正交码和 135 条距离接近最优的自正交码。

**💡 创新点**

创新点在于：①将自正交嵌入问题转化为在正交群 O(m,2) 上寻找满足 AAᵀ=BBᵀ 的矩阵；②证明 O(m,2) 对于 m≥5 由权重为 4 的转移矩阵生成，并以此为基因编码，保证交叉和变异后仍为合法嵌入；③设计了引导交叉（guided crossover）和局部搜索相结合的进化策略，使搜索更具方向性。

**🔧 技术方法**

使用的技术包括：遗传算法（GA）、权重为 4 的转移矩阵作为基因、正交群 O(m,2) 的生成与操作、基于余数 coset 的距离评估、字典序 fitness、局部搜索（局部改进）以及对比实验（消融、随机搜索）。

**📊 数据集**

实验数据来源于 MAGMA BKLC（Best Known Linear Codes）库，选取长度 20–256、维度 9–15 的 469 条基码，其中 469 条基码满足 m≥5（即需要追加至少 5 列的最短自正交嵌入）。

**📈 对比分析**

方法对比：①引导交叉 vs 随机交叉（消融实验）——引导交叉在 30 代内成功找到全部 66 条最优码，随机交叉则漏掉 2 条；②遗传算法 vs 同等时间预算下的随机搜索——遗传算法在所有 201 条基码上均获得 66 条最优码、198 条距离 ≤2 的码，随机搜索仅得到 57/49 条，表明 GA 在搜索效率和结果质量上显著优于纯随机或仅局部搜索。总体而言，GA 的平均间隙为 1.37，远低于随机方法。

**⚠️ 局限性**

局限性：①仅适用于需要至少 5 列追加的情形（m≥5）；②尽管获得大量最优码，但仍未证明对所有 (n,k) 参数组都是最优；③计算复杂度仍随 2^k 级增长，较大维度的基码仍需较长时间；④目前仅在二元域上验证，扩展到 q>2 仍需进一步研究。

---

## 104. Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?

**arXiv ID:** 2609.01924 | [PDF](https://arxiv.org/pdf/2609.01924v1)

**作者:** Wenlong Wang `[一作]` (Fin AI Research), Fergal Reid `[通讯]` (Fin AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了循环和深度递归Transformer在工作空间功能上的表现，并开发了虚拟展开适配器将Jacobian lens扩展到权重共享模型；

**💡 创新点**

提出了将Jacobian lens适配到迭代架构的虚拟展开技术、对传输结构与维护机制的系统分析，并首次系统比较两种循环架构的因果接口；

**🔧 技术方法**

使用Jacobian lens、虚拟展开适配器、线性读出、11族因果实验、稀疏概念检索等技术；

**📊 数据集**

基于WikiText‑103 1000个128标记提示，并结合事实、谜题、算术等任务提示；

**📈 对比分析**

对比Ouro（48层循环4次、深度监督）、Huginn（4层核心循环16次、无深度监督）与64层标准Transformer，采用相同拟合与实验协议；结果显示两种循环模型均能形成工作空间，但访问方式不同：写入/消融需跨循环，语言可报告性差异明显；

**⚠️ 局限性**

局限包括Jacobian lens的线性近似和单词词汇限制导致读取边界、对跨循环传输的盲区；以及架构与训练方案、规模混淆，缺乏对同一架构不同训练条件的对比

---

## 105. Comments on the recent improvements of the MRRW bounds

**arXiv ID:** 2609.01860 | [PDF](https://arxiv.org/pdf/2609.01860v1)

**作者:** Alexander Barg `[一作]` `[通讯]` (University of Maryland), Alexander Barg (University of Maryland)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文改进了经典的 McEliece–Rodemich–Rumsey–Welch（MRRW）上界，提出通过为每个码字关联一个可随码字变动的子空间而非单一向量，从而实现更精细的码字空间填充计数，进而获得更严格的上界。

**💡 创新点**

创新点在于：①把原先的“向量填充”升级为“子空间填充”，利用高阶布尔谐波子空间并与码字共同移动；②将这一新的子空间构造与量子信息中的纯态通道（PSC）与混态通道（MQC）进行对比，揭示两种方法在子空间打包视角上的一致性；③通过谱方法与典型子空间理论相结合，得到与已知 MRRW-1 上界相同的改进，但在细节上提供了更清晰的几何解释。

**🔧 技术方法**

使用的技术主要包括：离散傅里叶层次的表示理论、升降算子与三项递推、特征值/特征向量分析（Perron–Frobenius）、正定核与格拉姆分解、以及量子通道的典型子空间和 Holevo 信息计算；此外还利用了概率论中的典型性、强逆定理与信息谱分布。

**📊 数据集**

该工作为纯理论研究，未使用实验数据集；所有结果均在组合与量子信息理论的框架下通过数学推导获得。

**📈 对比分析**

与传统的 MRRW-1 上界相比，新的方法在 asymptotic 规模下给出更小的上界，提升幅度虽有限但在理论上意义重大。实验或数值比较并未给出，但文中已证明新上界在所有 δ∈(0,½) 上严格优于旧界，并在 δ≤0.0744 与 Bassalygo–Elias 上界相同。

**⚠️ 局限性**

局限性包括：①改进仅在 asymptotic regime；②在具体的编码长度下，上界与原 MRRW-1 差距仍很小；③方法依赖于 Hamming 空间的特殊对称性，扩展到更一般的结构（如球面、Grassmann 空间）需要进一步工作；④子空间构造与量子通道的对应关系虽然揭示了本质，但实现复杂度较高，尚未提供可直接构造有效码的算法。

---

## 106. A Study of Conditional Diffusion Models for Open-Loop Control under Dry Friction and Stiction

**arXiv ID:** 2609.01756 | [PDF](https://arxiv.org/pdf/2609.01756v1)

**作者:** Eric Aislan Antonelo `[一作]` `[通讯]` (Federal University of Santa Catarina), Eric Aislan Antonelo (Federal University of Santa Catarina)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于条件扩散模型的开环控制序列生成方法（Action Diffusion），用于处理一维点质量系统的干摩擦和粘滞问题；

**💡 创新点**

创新点在于将扩散模型直接作用于动作序列并以目标状态为条件，形成高效的采样提议分布，显著提高在低样本预算下的终端误差和减少卡顿步数；

**🔧 技术方法**

使用了1D条件U‑Net、DDIM逆过程与分类器无指导（classifier‑free guidance）进行采样，并配合已知动力学进行轨迹滚动；

**📊 数据集**

采用合成数据集，先采样初始状态和受控序列，利用结构化控制先验（包含高幅度“kick”、中等幅度和近零段）通过已知动力学生成终点状态，训练共50k条样本；

**📈 对比分析**

与均匀随机投掷、基于数据先验的随机投掷和交叉熵方法（CEM）对比，结果显示Action Diffusion在样本预算小（K≤128）时终端误差最低，卡顿步数最少；在高预算时CEM可进一步逼近但耗时更大；

**⚠️ 局限性**

局限性包括仅在一维、确定性、无噪声的干摩擦模型上验证；缺乏对更高维度、非线性或反馈控制的评估，且依赖已知动力学进行滚动评估，实际环境中可能受限。

---

## 107. Coverage, Not Targeting: A Structural Regime in Multi-Turn Agent Credit Assignment

**arXiv ID:** 2609.02417 | [PDF](https://arxiv.org/pdf/2609.02417v1)

**作者:** Chenyu Zhou `[一作]` (Institute of Science Tokyo), Xu Zhou `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多轮工具使用强化学习中，研究人员探讨了终端可验证奖励的信用分配，发现把终端优势均匀分散到所有轮次（均匀稠密奖励）优于稀疏二元奖励或集中到进展轮次的奖励；

**💡 创新点**

核心创新在于提出“验证信息密度（Verifier Information Density）”作为衡量信用分配效能的结构性指标，揭示在低V_d（大多数步骤不可观测）环境下，覆盖宽度优先而非定位；

**🔧 技术方法**

使用的技术包括GRPO、优势基线和REINFORCE策略梯度，配合共享回放（shared‑rollout）和匹配浓度的打乱（shuffled）对照来消除优化动态干扰；

**📊 数据集**

实验数据集主要为	τ^2‑bench（数据库状态验证）和BFCL V3（多轮状态检查），并在Qwen3（8B/14B）和Llama‑3.1（8B）等模型族上进行验证；

**📈 对比分析**

与传统的稀疏二元奖励、进展定位奖励以及随机打乱奖励进行对比，实验表明均匀稠密奖励在成功率和断言支持上显著优于其他方案，而集中奖励则往往更差；

**⚠️ 局限性**

限制在于仅研究终端可验证的工具代理，只考察了第一轮策略更新，且未覆盖高V_d或非工具任务环境，可能无法推广到所有强化学习信用分配场景。

---

## 108. Aggregating Neighbor Embedding Projection and Rank-Based Manifold Learning for Image Retrieval

**arXiv ID:** 2609.01963 | [PDF](https://arxiv.org/pdf/2609.01963v1)

**作者:** Vinicius Atsushi Sato Kawai `[一作]` (São Paulo State University), Daniel Carlos Guimarães Pedronette `[通讯]` (São Paulo State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

该论文提出一种结合UMAP投影与基于排名的流形学习，通过Borda计数聚合来提升内容检索的精度。

**💡 创新点**

创新点在于将投影得到的低维排名与三种无监督重排序方法生成的排名独立产生后，再通过Borda计数实现信息融合，从而同时利用几何结构与上下文关系。

**🔧 技术方法**

使用技术包括深度特征提取（ResNet152、Swin Transformer、DINOv2）、UMAP降维、三种重排序方法（LHRR、CPRR、RFE）、Borda计数聚合，并与RRF、CombSUM等聚合方式做对比。

**📊 数据集**

实验采用公开图像数据集 Flowers、Corel5k、Oxford‑IIIT Pets、CUB200‑2011 与 Dogs。

**📈 对比分析**

与基线、单独UMAP、单独重排序以及其他聚合方法比较，实验显示在多数数据集与特征上可提升 MAP 与前k精度，尤其在基线性能低时效果显著。

**⚠️ 局限性**

局限在于聚合效果受投影参数和重排序方法的影响，某些高性能特征下聚合并未超过单独重排序，且方法对随机种子敏感。

---

## 109. Marginal Expected Revenue for Jointly Ranking Auction and Fixed-Price Listings in E-Commerce Sponsored Search

**arXiv ID:** 2609.01628 | [PDF](https://arxiv.org/pdf/2609.01628v1)

**作者:** Greg Kocher `[一作]` (eBay), Sanjana Arun `[通讯]` (eBay)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在电子商务搜索广告系统中，提出并实现了“边际预期每千次展示收益”（meCPM）框架，用于统一对固定价、纯竞拍和混合竞拍/立即购买（ABIN）列表的收益估计与排名。

**💡 创新点**

创新点在于：①将传统 eCPM 扩展到价格随时间动态变化的竞拍场景，导出可计算的边际收益公式；②提出简化实现方案，利用已有的转化概率模型与价格信号快速启动新的竞拍 CPA 广告程序；③在实际系统中验证该方法能在不损害用户体验的前提下提升平台收益。

**🔧 技术方法**

技术主要包括：代理投标（proxy bidding）价格模型、有效投标通过率（valid‑bid‑through‑rate）预测、可选的投标价值分布（CDF）估计或点估计，结合现有的销售预测模型与价格信号，形成 meCPM 评分；并在 A/B 测试中使用实时流式排名与多目标优化。

**📊 数据集**

数据集来源为 eBay 生产日志，约 3,000 万条竞拍与 ABIN 列表的展示记录，用于训练投标通过率、价格分布或点估计模型，并在实验中评估 AUC 与收入指标。

**📈 对比分析**

通过与现有固定价 CPA、CPC 等基线对比，A/B 测试显示收入提升 0.43%–1.24%，平台指标与用户质量指标均有显著或保持不变的提升；模型在 30M+ 竞拍展示上的 AUC-ROC 与固定价相当（约 93%），验证了迁移学习的有效性。

**⚠️ 局限性**

局限性包括：①对投标价值分布的估计在冷启动时依赖简化或经验估计，精度受限；②边际收益公式的完整实现需要多项参数（如隐藏最高价 M、保留价 p_R）尚未集成，导致实际得分与理论差距；③价格预测与转化概率的校准仍存在偏差，可能影响长期收益；④在极端竞拍状态（如高保留价或多轮投标）下的表现尚未充分评估。

---

## 110. PhoenixNest-Video: Evidence-Grounded Multimodal Agent Framework for Automated Video Interview Assessment

**arXiv ID:** 2609.02231 | [PDF](https://arxiv.org/pdf/2609.02231v1)

**作者:** Fan Yuxuan `[一作]` (Hong Kong University of Science and Technology), Liu Hao `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 PhoenixNest-Video，一个基于证据的多模态评估框架，用于自动化视频面试评分，能够为每个评判标准提供可追溯的视频证据。

**💡 创新点**

创新点在于将rubric（评分准则）拆解为细粒度行为指示，并通过检索-验证管线与双重奖励的强化学习（对准rubric和总分等级差异）共同提升评分的透明度、校准度和判定一致性。

**🔧 技术方法**

主要技术包括多模态预处理（视觉、音频、字幕）、语义视频图构建、criterion‑conditioned检索与跨模态验证、以及基于rubric的强化学习奖励优化的scorer。

**📊 数据集**

使用了自研的 VInterview‑2025（491 个研究生录取面试视频，18 条 0–2 级 rubrics）以及公开的 RecruitView（2,011 条工作面试视频，12 维连续目标）进行训练和评估。

**📈 对比分析**

与多种专有、开源及视频专用 MLLM 基线（包括 Gemini、Grok、GPT‑5.4 等）在 VInterview‑2025 上的 grade‑level accuracy、MAE、Wasserstein 距离、QWK 等指标比较，PhoenixNest‑Video 在 91.5% 的 grade‑level accuracy 上优于更大参数量模型，并在 RecruitView 上实现了最高的 Spearman、Kendall、C‑index 和 Pearson 相关性。

**⚠️ 局限性**

局限性包括仅评估了性别与学科两项敏感属性的公平性、受限于英语录音与 ASR 误差的影响、以及无法公开视频数据；此外系统需全量视频预先加载，尚未支持实时流式评估。

---

## 111. DiffuSearch: How Hybrid Trajectory Planning Benefits from Aligned Objectives in Diffusion and Action Space

**arXiv ID:** 2609.02252 | [PDF](https://arxiv.org/pdf/2609.02252v1)

**作者:** Steffen Hagedorn `[一作]` (Robert Bosch GmbH), Alexandru P. Condurache `[通讯]` (Robert Bosch GmbH)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 DiffuSearch，一种将引导扩散模型与 Monte Carlo Tree Search（MCTS）相结合的两阶段混合轨迹规划器，用统一的驾驶目标来协调生成和细化过程。

**💡 创新点**

创新点在于：①在扩散生成和 MCTS 细化两阶段使用完全相同的四个驱动目标（碰撞避免、可行驶区、舒适度、进度）实现全局一致性；②利用扩散模型生成的高质量轨迹作为 MCTS 的先验，聚焦搜索空间；③证明统一目标能显著提升性能，尤其在高交互复杂场景中。

**🔧 技术方法**

核心技术：
- 扩散模型（Diffusion Transformer）与分类器自由引导（classifier‑free guidance）
- 以可微驾驶目标作为引导梯度
- 在离散动作空间（加速度/转向）上执行 MCTS，使用 PUCT 选择、线性动力学模拟、基于相同目标的奖励
- 交替训练与多步骤推理（10 Hz 采样、256 步 MCTS）

**📊 数据集**

使用的公开数据集：
- nuPlan（约 1,300 小时真实驾驶）进行闭环仿真评估
- interPlan（车道变换基准）用于跨基准测试

**📈 对比分析**

与现有最佳混合规划器（如 及  等）在 nuPlan reactive / SR 评估上进行对比；
- 在 Test14‑hard SR 上实现 4.92 % 的提升；
- 在 Val14‑R 约 0.6 % 之差；
- 在 interPlan lane‑change 任务中，所有交通密度下均超越基准，特别是中高密度场景。
- 结果显示：碰撞率下降、TTC 增大、舒适度提升（最高 +18 %）。

**⚠️ 局限性**

局限性：
- 单一目标统一可能在某些情境下限制了局部优化灵活性；
- 仅使用单模预测，未对多模策略进行显式建模；
- 需要额外的 MCTS 计算开销（约 256 步）；
- 当前仅在离散动作空间下有效，未针对连续控制或非线性动力学展开；
- 还未实现错误反馈机制，难以在闭环中自适应改进预测。

---

## 112. Multi-Agent Retrieval-Augmented Generation for Efficient Cloud Knowledge Base Search in Telecom SNOC Environment

**arXiv ID:** 2609.01618 | [PDF](https://arxiv.org/pdf/2609.01618v1)

**作者:** Harish Saragadam `[一作]` (Vodafone Idea -- SNOC), Ipsha Routray `[通讯]` (Vodafone Idea -- SNOC)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计了一个离线多源检索增强生成系统DocuSearch，用于电信SNOC云知识库查询。

**💡 创新点**

引入三源检索（稠密向量、BM25、知识图谱）CombSUM融合、逐块评估循环与显式归因验证，保证答案安全可靠。

**🔧 技术方法**

使用E5‑Large‑V2稠密向量、BM25、知识图谱、LangGraph多代理编排、交叉编码reranker、MMR、多模态LLM（Mistral‑7B）以及归因验证技术。

**📊 数据集**

在Vodafone Idea SNOC 的 4,200 份云文档（约 312,000 块）上进行评估。

**📈 对比分析**

与单源稠密或 BM25 基线相比，MRR@10 提升至 0.910、EM 达 78.4%，比仅稠密检索高 14.6 分。

**⚠️ 局限性**

整体延迟 2.15 秒，主要瓶颈在交叉编码与 LLM 推理，且 KG 构建依赖规则，难以捕捉隐式关系。

---

## 113. Generative Diffusion Surrogates with Analytical Variance Schedule

**arXiv ID:** 2609.01705 | [PDF](https://arxiv.org/pdf/2609.01705v1)

**作者:** Patrick Reichherzer `[一作]` (University of Oxford), Subir Sarkar `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

构建了一类以物理方差律为噪声调度的生成扩散模型，用于生成化学输运、湍流等随机传输过程的概率替代模型，并在实验质子成像与数值粒子模拟中进行验证；同时在瑞士卷数据上对比不同调度方案。

**💡 创新点**

将宏观传输方差（均方位移）直接锚定为扩散模型的时间尺度，实现仅需入口样本的训练、校准化模拟以及基于似然的推断；此方法不依赖中间时刻数据，能在保留方差演化的前提下只学习非高斯结构。

**🔧 技术方法**

使用VE（Variance‑Exploding）扩散SDE与去噪评分匹配，结合概率流ODE采样；FiLM‑MLP评分网络并可选地以入口峰度为条件；调度采用湍流中球形电光速（telegraph）方差律，比较线性、余弦、Karras/EDM等启发式调度以及Kac流基线；提供PF‑ODE似然估计实现可微推断。

**📊 数据集**

入口分布：高斯方差、学生‑t 族样本（峰度取0.5/1.0/1.5）；实验数据：实验室质子成像得到的束流强度分布与实验测得的方差尺度；数值粒子模拟与Johns Hopkins MHD湍流数据库生成的磁场；瑞士卷数据用于调度消融实验。

**📈 对比分析**

与传统线性/余弦/Karras/EDM调度及Kac流基线对比；在方差与峰度演化上，telegraph‑anchored方案的方差偏差 Q≈0.34、峰度偏差≈1.5，远优于其他调度；在瑞士卷精度‑召回曲线中，telegraph 调度在小邻域 k≤10 处召回率最高、精度保持较好；总体性能通过 Q 统计量、峰度误差及 P/R 曲线量化，结果显示物理锚定方案在保持方差精度的同时提供更佳的分布形态匹配。

**⚠️ 局限性**

仅能保证方差的准确性，无法捕捉传播速度、记忆效应或更高阶矩的真实变化；当传输核本身高度非高斯时，模型只能通过入口峰度传递形状信息，导致与真实核的峰度差距；不生成单粒子轨迹，适用于分布预测；需要已知方差律的系统，且对极端重尾或分形输运的适用性有限。

---

## 114. CliffRank: A Dual-Branch Framework for Activity-Cliff Ranking Prediction

**arXiv ID:** 2609.01673 | [PDF](https://arxiv.org/pdf/2609.01673v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 115. Discretization-free exact recovery in geometric community detection

**arXiv ID:** 2609.01635 | [PDF](https://arxiv.org/pdf/2609.01635v1)

**作者:** Maarten Hoeneveld `[一作]` (Leiden University), Raphaël Sala `[通讯]` (Toulouse University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在几何隐藏社区模型（GHCM）下提出两种多项式时间的精确社区检索算法，第一种为无离散化的贪婪传播方法，第二种为块级聚合传播方法；

**💡 创新点**

创新点在于：1）不需要空间离散化即可实现精确恢复；2）将全局区分性假设弱化为“见证连通性”条件，显著放宽了结构假设；3）证明在信息论阈值下两算法都能达到精确恢复，且数值实验显示超出理论保证的性能；

**🔧 技术方法**

采用MAP初始化、基于边权重的亲和信息度量进行贪婪传播、利用局部似然优化进行精细化；理论分析结合Chernoff–Hellinger散度、Poisson点过程性质与可见图连通性；

**📊 数据集**

实验使用自定义二维GHCM数据集（n=10000、r=3、3个社区、可变的距离依赖Bernoulli边权分布）和公开的Geometric Block Model生成的数据；

**📈 对比分析**

与基线（传统基于离散化的传播算法）比较，实验结果表明无离散化算法的恢复成功率在信息论阈值附近快速上升，且在中间区域（信息论可行但见证连通性不满足）仍能实现精确恢复，性能优于基线；

**⚠️ 局限性**

主要局限：1）理论证明仅适用于满足见证连通性和边权分布的有界相对密度假设；2）在该中间区域的理论阈值尚未完全匹配，存在未解的能否进一步降低阈值；3）对不同支持的边权分布假设较强，可能限制实际应用。

---

## 116. LaST-SR: Laplace-Inspired Steady-Transient Complex-Frequency Decomposition for Single Image Super-Resolution

**arXiv ID:** 2609.02063 | [PDF](https://arxiv.org/pdf/2609.02063v1)

**作者:** Linhao Li `[一作]` (Southeast University), Langkun Chen `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于拉普拉斯频率分解的单图像超分网络 LaST‑SR，使用全局傅里叶和局部复频分支联合建模图像结构与局部非周期细节。

**💡 创新点**

首次推导二维特征的稳态‑瞬态分解，并在 SISR 中引入复频分解与跨分支协同聚合模块，显著提升结构一致性与细节重建。

**🔧 技术方法**

拉普拉斯神经算子、全局傅里叶变换、局部窗口复频模态、窗口自注意力与跨分支注意力、频域一致性损失等技术。

**📊 数据集**

在 DIV2K 训练集上学习，评估于 Set5、Set14、BSD100、Urban100 与 Manga109 五个标准超分辨率基准。

**📈 对比分析**

与 CNN、Transformer 与频域方法对比，LaST‑SR 在×2/×4 超分中均取得最高 PSNR/SSIM，提升约 0.05–0.09 dB。

**⚠️ 局限性**

方法依赖稳态‑瞬态近似假设，且对非均匀降质场景的适应性有限，未来需扩展至更复杂降质模型。

---

## 117. Cross-Model Distillation of a Human-Pose Foundation Model from Unannotated Infant Video for Markerless 3D Pose Estimation

**arXiv ID:** 2609.01840 | [PDF](https://arxiv.org/pdf/2609.01840v1)

**作者:** R. James Cotton `[一作]` (Shirley Ryan AbilityLab), Colleen Peyton `[通讯]` (Northwestern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文通过无标签婴儿视频进行跨模型蒸馏，将Sapiens 2的2D关键点精度迁移至SAM 3D Body，得到更准确的三维婴儿姿态估计。

**💡 创新点**

创新点在于利用渲染循环的无监督蒸馏，将单模2D高精度模型的伪标签注入3D体形模型，完成仅用未标注视频即可提升3D姿态的技术。

**🔧 技术方法**

技术包括基于MHR的SAM 3D Body骨架、Sapiens 2 2D关键点+深度+法线+分割的教师模型、可微前向运动学、差分渲染器以及关键点、法线、深度和轮廓的联合损失。

**📊 数据集**

使用了543段来自311名婴儿的单摄像头无标签视频作为训练集，以及11名多摄像头婴儿的高质量三视角数据作为验证集（共173段录制）。

**📈 对比分析**

与Sapiens 2、SAM 3D Body原始模型以及先前工作进行对比：在同视角2D PCK@10px、PA-MPJPE等指标上，蒸馏后模型分别提升至42%与22.2mm的误差，显著优于未蒸馏模型。

**⚠️ 局限性**

主要局限在于MHR形状空间仍基于成人，导致婴儿头部和躯干比例失真；渲染循环中对非关键点的稀疏监督效果有限；缺乏真实标注基准，评价依赖于模型生成的三维重建参考。

---

## 118. Contrastive Explanations in Quantitative Bipolar Argumentation Frameworks

**arXiv ID:** 2609.02399 | [PDF](https://arxiv.org/pdf/2609.02399v1)

**作者:** Xiang Yin `[一作]` (Imperial College London), Francesca Toni `[通讯]` (Imperial College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了在定量双极论证框架（QBAF）中解释两个主题论点强度差异的对比性解释方法，并给出了对比归因函数（CAF）的形式化定义、性质与实例化。

**💡 创新点**

创新点在于：①提出可与任何归因函数配合、满足反对称、校准与加法性质的CAF，证明其唯一性；②利用加法特性设计线性复杂度的动态规划算法显著降低多主题计算成本；③对比归因与单一归因在解释效果、可解释性、偏见检测中的差异进行了系统分析。

**🔧 技术方法**

主要技术包括：模块化渐进语义、归因函数（移除、梯度、Shapley）、对比归因函数（基于减法的三种实例化）、反事实性与局部可信度等属性分析以及动态规划实现。

**📊 数据集**

使用的数据集与案例：①基于人工构造的医疗决策QBAF（症状-诊断-治疗层），②美国司法风险评估数据集 COMPAS，用于展示对比归因在偏见检测中的有效性。

**📈 对比分析**

通过属性证明、案例可视化与实验评估，对比归因在揭示治疗选择原因、捕捉模型偏见方面表现优于传统归因；实验显示对比归因能显著提升被解释特征（如种族）在重要性排名中的位置，且在计算复杂度上相较于 Shapley 归因更高效。

**⚠️ 局限性**

局限性包括：①只处理两主题论点间的差异，未扩展至多主题排序解释；②依赖模块化语义与独立性假设，非所有语义或网络结构可适用；③缺乏大规模用户研究验证对比解释在实际决策支持中的可用性与可解释性提升效果。

---

## 119. Dual-Metric Partitioning with Adaptive Kernel Execution for Efficient GCN Acceleration

**arXiv ID:** 2609.01983 | [PDF](https://arxiv.org/pdf/2609.01983v1)

**作者:** Lingling Zhang `[一作]` (Capital Normal University), Guoren Wang `[通讯]` (Beijing Institute of Technology)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

针对图卷积网络的GPU加速，提出了DualGCN框架，通过双指标划分和自适应核执行来提升SpMM性能。

**💡 创新点**

创新点在于将节点度和基于匿名随机游走的邻域密度相结合形成双指标工作量度量，实现稀疏与稠密区块的智能划分并针对不同区块采用相应的warp级和指令级并行策略。

**🔧 技术方法**

技术包括匿名随机游走邻域密度估计、混合度量排序划分、双路径（稀疏/稠密）GPU内核、4路指令级并行、共享内存与L2缓存优化等。

**📊 数据集**

实验使用12个真实世界图数据集，包括ARTIST、AMAZON0601、ARXIV、PPA、COLLAB、TWITTER、COM-AMAZON、YELP、YEAST、OVCAR-8H、PRODUCTS、CITATION。

**📈 对比分析**

与cuSPARSE、GNNAdvisor和ACCEL等基线比较，在列维度16-128的SpMM任务上平均加速比分别为2.53×、3.8×、2.13×，单独验证工作负载分配方法可提升52%至4.5×。

**⚠️ 局限性**

局限在于需要先进行匿名随机游走预处理，对图结构变化频繁的在线场景可能产生额外开销；同时对极端稠密图的划分仍需进一步优化。

---

## 120. DESA-TTA: Dynamic EMA and Source Anchoring for Test-Time Adaptation

**arXiv ID:** 2609.01795 | [PDF](https://arxiv.org/pdf/2609.01795v1)

**作者:** Atif Belal `[一作]` (ETS Montréal), Eric Granger `[通讯]` (ETS Montréal)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于动态EMA与源锚定的在线测试时适应方法 DESA‑TTA，用于提升视觉‑语言目标检测器在不同分布下的鲁棒性。

**💡 创新点**

创新点在于：①根据教师伪标签置信度与框密度自适应计算 EMA 系数；②利用学生漂移动态调节源锚定强度，二者共同抑制累积漂移与噪声误差。

**🔧 技术方法**

技术手段包括：mean‑teacher TTA 框架、动态 EMA、源锚定、弱/强视图伪标签生成、基于置信度和框密度的教师不确定性评估。

**📊 数据集**

使用的数据集涵盖多种分布偏移：Watercolor、ClipArt、Comic（风格移位）；Foggy Cityscapes、BDD100K（驾驶场景）；ExDark（低光照）；VOC‑C、COCO‑C（15类常见噪声）。

**📈 对比分析**

与 ZS、TPT、VPT、DPE、标准 mean‑teacher、VLOD‑TTA 等基线比较，DESA‑TTA 在所有偏移上均显著提升，例如 VOC‑C AP_50 提升 +14.5 点，YOLO‑World 的推理速度提升 55%；在 Grounding DINO 上同样保持领先。

**⚠️ 局限性**

局限性包括：只针对单一目标流的在线适应，未处理跨域连续漂移；对预训练源模型的依赖仍存在；对超参数的选取仍有一定敏感性。

---

## 121. Recursive Value Learning for Long-Horizon Offline Goal-Conditioned RL

**arXiv ID:** 2609.02237 | [PDF](https://arxiv.org/pdf/2609.02237v1)

**作者:** Hyeonseong Jeon `[一作]` (Yonsei University), Youngwoon Lee `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Divide‑and‑Conquer强化学习框架，利用递归分割轨迹段并从叶子到根进行自底向上训练，显式捕捉长短轨迹间的依赖关系；

**💡 创新点**

创新点在于：①使用平衡二叉树结构将长轨迹拆解为短段，实现对值函数的子父级依赖顺序；②采用精确轨迹因式分解避免对中间状态的贪婪最大化，降低误差传播；③结合多步传播实现跨轨迹的最优值恢复；

**🔧 技术方法**

技术主要包括：递归分割与平衡二叉树；子父级训练调度（slot scheduling）；基于精确乘法因式分解的值更新；多步上采样传播（global relabeling）和期望分位损失；

**📊 数据集**

使用了公开的离线数据集 OGBench（state‑和 pixel‑based），CALVIN 机器人操作数据以及自建的离散环境来评估误差积累；

**📈 对比分析**

与多类基线（TD、MC、层次、三角不等式方法）在长周期、不同观测形式的任务中对比，“Divide‑and‑Conquer RL” 在大多数长周期任务上均取得最高成功率，甚至超过了一些层次方法；

**⚠️ 局限性**

局限性包括：仅适用于确定性动力学环境；在部分中等长度任务中仍无法匹敌层次方法；递归分割对噪声轨迹可能不稳健，未来需探索在随机环境中的推广。

---

## 122. Disentangling Statistical Preemption from Entrenchment in Language Models' Avoidance of Overgeneralization

**arXiv ID:** 2609.01794 | [PDF](https://arxiv.org/pdf/2609.01794v1)

**作者:** Yixuan Wang `[一作]` (University of Waterloo), Kanishka Misra `[通讯]` (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在 GPT‑2 语言模型上实施 controlled rearing 实验，探究儿童语言输入中预阻塞（preemption）与巩固（entrenchment）对避免过度泛化的影响。

**💡 创新点**

首次在神经网络上 disentangle 预阻塞与巩固，发现词项级预阻塞无效，而在抽象层面存在弱正效应。

**🔧 技术方法**

采用自回归 Transformer（GPT‑2）模型，并利用线性混合效应回归对 NLL 结果进行统计分析。

**📊 数据集**

使用 CHILDES 语料（约 5 M 发话，26 M 词）作为训练集，生成与之不重叠的测试句对。

**📈 对比分析**

与完整语料训练模型对比，采用 NLL 差值评估过度泛化；实验表明词项级预阻塞缺失，抽象预阻塞呈正向效应。

**⚠️ 局限性**

局限包括仅针对单一语法现象、仅使用 GPT‑2 结构、未覆盖更大数据集，且对竞争判定的细粒度方法尚不完善。

---

## 123. Not All Agreement Counts as Corroboration: Provenance-Conserving Multi-View Fusion for Typed Action Admission in Human-Robot Collaboration

**arXiv ID:** 2609.01662 | [PDF](https://arxiv.org/pdf/2609.01662v1)

**作者:** Zekai Jin `[一作]` (McGill University), Yi Shao `[通讯]` (University of British Columbia)

**通讯引用:** 11392 | [OpenAlex ID](https://openalex.org/A5071931643)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 PACT（Provenance‑Conserving Fusion and Typed Action Admission）框架，用关联式证据计数来区分多源预测中的真正证据来源，并通过结构化的融合与分层授权实现安全执行。

**💡 创新点**

创新点在于：① 将“证据计数性”视为实例级关联变量；② 通过组件内最小化（meet）与跨组件求和（sum）实现严格的证据保留；③ 将行动授权拆分为可解释的“hold/confirm/fallback”三类响应，保留拒绝原因。

**🔧 技术方法**

技术核心包括：坐标最小化算子、集合连通分量构造的 provenance 图、基于 Dirichlet 推理的后验投影、以及基于阈值和规则的 typed admission。

**📊 数据集**

使用的主要数据集有：① 48 个 Isaac Sim 机器人协作场景；② 多视角评估数据 TMC、RCML、HandWritten/Mfeat；③ 人机协作真实环境 HABIT（1,128 事件）；④ 语言模型预测 Qwen3‑VL‑8B/32B、InternVL3‑8B 的推理输出。

**📈 对比分析**

与传统方法（product、nested Dirichlet、global cautious、hierarchy‑matched cautious、provenance‑discounted pooling 等）比较，PACT 在 31,200 次评估中取得 ncsAURC 为 0.0861，远低于其它方法（0.1479、0.2247 等）；在 HRC 任务中实现 0% 错误授权，且在多源复制场景下保持一致性。

**⚠️ 局限性**

局限性：① 依赖用户预设的 provenance 分区，错误分区会导致证据预算误差；② 对源缺失的处理方案对结果影响较大；③ 选取的分数阈值和校准方式对性能有显著影响；④ 目前仅评估离线安全性，缺乏在线行为/人类主观评估；⑤ 只能在已知关联关系的情形下保证计数，无法自动推断所有依赖。

---

## 124. Barriers to Using Static Application Security Testing (SAST) Tools: A Literature Review

**arXiv ID:** 2609.01669 | [PDF](https://arxiv.org/pdf/2609.01669v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 125. Skim and Skip: Hierarchical Adaptive Inference for Efficient Multimodal Retrieval

**arXiv ID:** 2609.01613 | [PDF](https://arxiv.org/pdf/2609.01613v1)

**作者:** Meng Gao `[一作]` (Tsinghua University), Yujiu Yang `[通讯]` (Tsinghua University)

**通讯引用:** 4504 | [OpenAlex ID](https://openalex.org/A5020953714)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Skim and Skip（SAS）两阶段层次自适应推理框架，用于高效多模态检索。

**💡 创新点**

创新点在于将影响感知的Token筛选与深度适应的早停结合，既去除冗余Token，又根据检索表示成熟度动态决定推理层数。

**🔧 技术方法**

使用了影响感知Token过滤（基于注意力扰动的上界估计）、中间层检索激活（蒸馏+对比学习）以及轻量级终止判别器等技术。

**📊 数据集**

使用了MMEB多模态检索基准数据集，涵盖12个检索任务。

**📈 对比分析**

与密集MLLM基线比较，SAS在保持约99%准确率的同时，实现了1.64×速度提升、FLOPs降低66.3%。

**⚠️ 局限性**

限制包括硬件对变长序列支持不足导致实际wall‑clock收益低于理论，以及目前仅在检索任务中验证，未在生成任务中测试。

---

## 126. Semantics-Guided Automatic Tensorization for Multiobjective Evolutionary Algorithms: A Multi-Agent Framework

**arXiv ID:** 2609.02387 | [PDF](https://arxiv.org/pdf/2609.02387v1)

**作者:** Zhenyu Liang `[一作]` (Hong Kong Polytechnic University), Ran Cheng `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种语义驱动的自动张量化框架EvoCoCo，将多目标进化算法（MOEA）的实现从CPU模式迁移至GPU张量化实现；

**💡 创新点**

核心创新在于将算法语义重构、蓝图构造、多分支张量化与执行反馈修复三位一体化为多智能体协作的闭环流程；

**🔧 技术方法**

结合大语言模型（LLM）进行代码生成与规则检索，利用PyTorch张量操作与GPU加速，实现高并行度的MOEA运行；

**📊 数据集**

使用了48种来自PlatEMO的MOEA实现以及DTLZ、WFG、LSMOP、MaF等共1920个测试组合的数据集；

**📈 对比分析**

通过迁移可靠性、优化保真度（IGD覆盖率达88.2%）和计算可扩展性（在规模增大时平均加速达80×）与单次翻译对比，EvoCoCo显著提升了迁移成功率与执行速度；

**⚠️ 局限性**

局限在于仅针对MATLAB→PyTorch的转换，未验证跨框架通用性，且在极大规模下GPU内存瓶颈仍需进一步优化。

---

## 127. Agent Memory Is a Surface for Endogenous Authorization Laundering

**arXiv ID:** 2609.01836 | [PDF](https://arxiv.org/pdf/2609.01836v1)

**作者:** Tommaso Cerruti `[一作]` (ETH Zurich), Ansel Kaplan Erol `[通讯]` (Georgia Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了EAL-Bench基准，用来评估长期运行的LLM代理在持久记忆中产生并传播虚假授权的情况。

**💡 创新点**

创新点在于将授权状态的形成与传播分离，使用隐藏的确定性账本对记忆写入的准确性进行评估，并构建跨采购、网络安全与金融三大领域的案例。

**🔧 技术方法**

技术上采用了两种记忆表示（自由文本与结构化JSON）和两种更新策略（一次性与增量式），并测试了五个写入模型与两个执行模型，配合源授权门控与事件来源限界两种缓解方案。

**📊 数据集**

数据集由Claude Opus等LLM生成的多会话组织历史组成，随后人工审核确保授权事件清晰可追溯，并以确定性账本为真值标准。

**📈 对比分析**

实验显示，在增量式写入下，写入模型最多可导致50.2% 的虚假授权形成，执行模型在出现虚假授权时有98.6% 的几率执行违规操作；两种缓解方案将违规率分别降至7.3%与9.0%，但授权使用率相应下降至约54–65%。

**⚠️ 局限性**

局限性包括：数据为人工构造且结构化，难以评估真实工作场景的普遍性；自由文本记忆的真实性评估受限；部分实验仅基于单一随机种子；缓解方案依赖已知授权主体和可验证的事件来源。

---

## 128. If It Moves, Radar Knows: A Physics-Aware Radar Transformer for Class-Agnostic Moving-Object Detection

**arXiv ID:** 2609.02289 | [PDF](https://arxiv.org/pdf/2609.02289v1)

**作者:** Yinghao Sun `[一作]` (University of Electronic Science and Technology of China), Tieshan Li `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于雷达的稀疏Transformer模型PART，专门用于无类别的移动目标检测，直接输出存在置信度、表面点和地面速度。

**💡 创新点**

核心创新点包括：① Doppler-Aware Query Initialization (DAQI) 用位置-速度聚类生成输入依赖的查询；② Physics-Guided Cross-Attention (PGCA) 将径向Doppler一致性和雷达散射截面 (RCS) 信息嵌入注意力；③ 不确定性感知监督 (UAS) 通过随机遮蔽真实框和软存在目标降低对完整标注的依赖。

**🔧 技术方法**

采用稀疏点云Transformer、RoPE多头自注意力、SE通道注意、DBSCAN聚类、点云编码和多层交叉注意等技术；同时使用雷达返回的速度、RCS、不确定度等多模态特征。

**📊 数据集**

在nuScenes雷达数据集上进行训练与评估，使用官方验证集的无类别移动目标协议。

**📈 对比分析**

与现有LiDAR、相机及混合模态检测器对比，PART在CA-AP、mASTE、mAVE上分别达0.8827、0.3188 m、0.8084 m/s，参数仅1.1 M；在夜间、雨天、严重遮挡、以及未标注的稀有/安全相关类别（警车、轮椅、个人移动用户、动物）上表现尤为突出，召回率分别超过95%、95%、80%和92%。

**⚠️ 局限性**

局限性：依赖足够的雷达返回和可测径向速度，无法处理静止或主要以切向运动的目标；不提供语义分类或完整3D框；暂未集成时间跟踪。

---

## 129. Cite or Decline: A Strict Course-Grounded Chatbot for STEM Lecture Videos

**arXiv ID:** 2609.01846 | [PDF](https://arxiv.org/pdf/2609.01846v1)

**作者:** S M Masrur Ahmed `[一作]` (University of Houston), Jaspal Subhlok `[通讯]` (University of Houston)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

部署并评估了一个基于检索增强的课程专属视频聊天机器人，提供带时间戳引用的答案。

**💡 创新点**

通过课程隔离、章节摘要先验以及可点击时间戳引用的设计，提升检索精度并验证真实教学环境中的可行性。

**🔧 技术方法**

使用 Retrieval‑Augmented Generation（dense + BM25 + 章节摘要先验）、all‑MiniLM‑L6‑v2 编码、GPT‑4.1 Mini 生成摘要，构建 RAG 管线。

**📊 数据集**

真实大学 STEM 课程视频与章节（8‑10 门课）以及公开基准 EduVidQA。

**📈 对比分析**

在 EduVidQA 上与 dense‑only、+BM25、+摘要先验以及无课程隔离的基线对比，摘要先验提升约 6.3pp 正确检索率，课程隔离减少 17.8pp 误检率。

**⚠️ 局限性**

仅单校单科实验，使用自动化诊断而非人工事实评估，样本偏高、未测学习效果，隐私与引用可靠性仍待进一步验证。

---

## 130. RecKAN: Kolmogorov-Arnold Networks with a Learnable Recursive Polynomial Basis

**arXiv ID:** 2609.01729 | [PDF](https://arxiv.org/pdf/2609.01729v1)

**作者:** Amirhosein Azarpour `[一作]` `[通讯]` (Shahid Beheshti University), Amirhosein Azarpour (Shahid Beheshti University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种可学习的递归多项式基函数的Kolmogorov–Arnold网络（RecKAN），通过学习二阶多项式递推系数动态生成边函数基；

**💡 创新点**

创新点在于将基函数的生成规则本身作为可学习参数，取代传统固定的B样条、切比雪夫或Jacobi多项式基，使网络能够自适应选择最适合任务的基函数；

**🔧 技术方法**

采用二阶多项式递归、张量化的基函数生成、共享递推系数、归一化与两速优化等技术构建RecKAN；

**📊 数据集**

在图像分类（MNIST、CIFAR‑10、Fashion‑MNIST、SVHN）、文本分类（AG News）、生物医学时间序列分类（ECG5000）和时间序列预测（ETTh1）等多种数据集上进行实验；

**📈 对比分析**

与三种固定基的KAN（ChebyKAN、JacobiKAN、SplineKAN）以及匹配参数的MLP进行对比，RecKAN在所有分类任务上均取得最高精度，在ETTh1预测任务上获得最低MSE，且在大部分任务上提升幅度从小数个百分点到十几个百分点；

**⚠️ 局限性**

局限性包括：仅使用单一共享的五个递推系数，限制了各层/通道的表达多样性；仅考虑二阶递归且系数不随阶数变化，未覆盖更广的正交多项式族；数值稳定性需多重正则与归一化，且对不同任务的解释仍为描述性。

---

## 131. Meta-ethics and AI: exploring the novel meta-ethical questions in the era of AI

**arXiv ID:** 2609.01685 | [PDF](https://arxiv.org/pdf/2609.01685v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 132. A Unified Particle Filter LSTM for Data-Driven Process Simulation

**arXiv ID:** 2609.01967 | [PDF](https://arxiv.org/pdf/2609.01967v1)

**作者:** Parvin Malekzadeh `[一作]` (University of Toronto), Dmitry Krass `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种基于粒子滤波的 LSTM（Unified PF‑LSTM）来生成患者在急诊科的完整流程轨迹，并预测下一个活动和当前活动的停留时间。

**💡 创新点**

创新点在于：①使用粒子滤波维护多种可能的潜在状态分布而非单一确定性状态；②用矩生成函数（MGF）特征保留分布形状信息；③加入路由验证器避免递归生成过程中因异常停留时间导致的不合理转移。

**🔧 技术方法**

技术包括：PF‑LSTM 结构、MGF 统计特征、量化 Huber 损失、端到端训练、递归生成与验证器、soft resampling 等。

**📊 数据集**

使用三份急诊科（ED A、ED B、ED C）真实事件日志，约 120,000 条患者轨迹，包含活动序列、时间戳和静态/动态特征。

**📈 对比分析**

与 5‑阶马尔可夫+随机生存森林基线以及仅使用 LSTM 的对比模型（无粒子滤波）进行比较。PF‑LSTM 在 TPIA、LOS 的均值和 90 分位误差均明显下降（如 3‑倍以上的误差减小），但生成时间更长。

**⚠️ 局限性**

局限性：①粒子滤波导致计算量显著增加；②实验仅覆盖急诊科场景，未验证对其他服务系统的适用性；③对系统状态观测依赖较强，缺乏在线适应与政策优化的机制。

---

## 133. Pushing Forward Multi-Secret-Key Homomorphic Encryption for Private Average Aggregation

**arXiv ID:** 2609.01945 | [PDF](https://arxiv.org/pdf/2609.01945v1)

**作者:** Miguel Morona-Mínguez `[一作]` (atlanTTic, Universidade de Vigo), Alberto Pedrouzo-Ulloa `[通讯]` (atlanTTic, Universidade de Vigo)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出一种基于多密钥同态加密（MHE）的轻量级私有平均聚合协议，专门用于跨机房联邦学习，解决了传统单密钥同态加密在聚合过程中对聚合器的强非同谋假设与大噪声扩散的缺陷。

**💡 创新点**

创新点包括：①不再生成集合公共密钥，客户端使用各自私钥加密，降低密钥管理和加密开销；②利用对密钥共享的加法同态特性，显式跟踪并抵消聚合过程中的噪声，从而消除对指数级噪声（smudging noise）的需求；③在半诚实模型下提供基于RLWE的安全证明；④通过对BFV（精确）与CKKS（近似）两种密钥同态加密的实例化，展示在不同安全参数和精度下的显著通信与计算性能提升。

**🔧 技术方法**

核心技术包括：RLWE‑基础的BFV与CKKS同态加密；多密钥同态加密框架（MHE）；线性秘密共享与加法同态结合（多密钥同态加密与共享密钥的 key‑homomorphism）；模拟证明的安全模型；对噪声管理的优化（避免大方差 smudging noise）。

**📊 数据集**

本文在性能评估中使用合成模型更新（每个客户端约 1.6×10⁷ 参数），未涉及公开数据集；重点比较协议在相同安全参数（128‑bit）下的通信与运行时成本。

**📈 对比分析**

通过与现有 MHE‑基聚合方案的对比，实验表明：①密文尺寸缩小约 2‑3 倍；②在线加密、聚合与解密时间均显著降低，尤其是 CKKS‑MHE 的解密时间被大幅压缩；③在不同精度和客户端数量下，所提方案在通信与计算成本上持续优于传统 MHE，且不受 smudging noise 规模的影响。

**⚠️ 局限性**

局限性：①协议仅支持线性（平均）聚合，难以直接扩展到鲁棒或非线性聚合；②安全性仅在半诚实模型下证明，针对恶意攻击仍需进一步完善；③在极大客户端数或频繁加入/离线场景下，所需的共享随机多项式 a 的生成与同步仍带来额外复杂度；④对动态客户端集的支持需要额外的轻量级协商，未在本文完整讨论。

---

## 134. Towards Global Federated Genome-Wide Association Meta-Analysis Using GA4GH TES

**arXiv ID:** 2609.02227 | [PDF](https://arxiv.org/pdf/2609.02227v1)

**作者:** Abhijit Chunduru `[一作]` (Argonne National Laboratory), Ravi Madduri `[通讯]` (Argonne National Laboratory)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个基于隐私保护的联邦GWAS元分析管道，使得跨国机构能够在不共享原始基因型数据的前提下完成全基因组关联分析。

**💡 创新点**

创新点在于将APPFL联邦学习框架与GA4GH Task Execution Service标准相结合，实现“把计算迁移到数据”式任务执行，并开发HiveWatch实时地理可观测工具，为异构跨国计算提供完整的操作可视化。

**🔧 技术方法**

使用技术包括APPFL联邦学习框架、GA4GH TES（通过Funnel实现的标准API）、Plink进行本地GWAS、HiveWatch进行任务监控，以及容器化任务调度。

**📊 数据集**

使用了100,000个合成欧洲人群基因型（239,801个变异）以及模拟的糖尿病和BMI表型数据（基于PGS Catalog评分）。

**📈 对比分析**

通过与传统集中式固定效应元分析结果对比，发现联邦方法在统计显著性、lambda GC等指标上与中心化结果完全一致，且未出现统计偏倚，验证了方法的有效性。

**⚠️ 局限性**

限制包括：仅在合成数据上评估，未验证真实多机构部署的可行性；当前仅实现固定效应模型，缺乏更复杂的模型支持；隐私保护层面尚未提供更强的差分隐私或同态加密保证；对容错和网络故障的鲁棒性尚待进一步研究。

---

## 135. VIPS: Vehicle-Infrastructure Cooperative Planning Benchmark via Pseudo-Simulation

**arXiv ID:** 2609.02462 | [PDF](https://arxiv.org/pdf/2609.02462v1)

**作者:** Hoonhee Cho `[一作]` (KAIST), Kuk-Jin Yoon `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于伪仿真的车辆‑基础设施协同规划基准VIPs，并开发了稀疏协同规划框架CoS‑V2X。

**💡 创新点**

将伪仿真扩展到V2I场景，联合车辆与基础设施观测进行两阶段评估，并引入稀疏anchor‑based表示大幅降低通信开销。

**🔧 技术方法**

使用三维高斯散射+Diffix3D+进行车辆视图生成，掩模‑填充技术生成基础设施视图；采用稀疏anchor感知模块（Deformable Aggregation、Cross‑Attention、Self‑Attention）与统一EPDMS评估指标。

**📊 数据集**

采用V2X‑Real数据集（多车辆+多摄像头/雷达基础设施）并手工注释地图。

**📈 对比分析**

通过Stage 1/Stage 2的两阶段评估与EPDMS指标与其他E2E‑AD方法（UniAD、SparseDrive、Uni‑V2X等）对比，CoS‑V2X在综合得分最高且通信成本最低。

**⚠️ 局限性**

对合成观测的质量与真实性仍有限制，伪仿真与真实环境仍存在差距；仅关注V2I，未覆盖更复杂的V2X或多车协同场景。

---

## 136. Fair Stable Matching: A Nash Social Welfare Approach

**arXiv ID:** 2609.02354 | [PDF](https://arxiv.org/pdf/2609.02354v1)

**作者:** Parth Desai `[一作]` (IIIT Hyderabad), Sujit Gujar `[通讯]` (IIIT Hyderabad)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一种在稳定婚姻问题中最大化Nash社会福利（NSW）的算法，目标是在保持匹配稳定性的同时实现更公平的配对结果。

**💡 创新点**

创新点主要包括：①首次将NSW作为公平度量引入两侧非可转移效用的稳定匹配框架；②通过将NSW转化为对数加权旋转图的权值，构造了一个可用最大流求解的加权旋转偏序；③在此框架下设计了Õ(n⁴)的算法，并证明其能得到最优NSW稳定匹配。

**🔧 技术方法**

核心技术包括：旋转偏序（rotation poset）构造；对数变换把乘积优化转化为加权最大化；构造带有无穷容量边的s‑t网络；利用Dinic算法配合动态树实现最大流；以及对算法正确性与复杂度的理论分析。

**📊 数据集**

实验使用四类随机生成的偏好分布：均匀分布（U）以及基于流行度的三种分布（P_U、P_T、P_N），每种分布下生成10万组实例，规模为n=25、50，覆盖完整随机性与偏好强度差异。

**📈 对比分析**

与三种传统公平基线（均衡、最小遗憾、性别平衡）以及NSW非稳定匹配进行对比。评价指标包括μ^e（均衡）、μ^r（遗憾）、μ^d（性别差异）与μ^nsw。结果显示NSW匹配在统计上对所有指标均优于或至少不劣于基线（Pareto非支配、圆形图面积最小），并在所有实验设置下保持高效运行（时间在Õ(n⁴)范围内）。

**⚠️ 局限性**

主要局限包括：①算法复杂度相对较高（Õ(n⁴)），在大规模实例上可能不可行；②仅针对完全且无等价偏好的二人匹配模型，未考虑不完全或多对一匹配；③缺乏对NSW公平性与其他指标理论界限的证明，只给出经验性评估；④对偏好中出现平局或不完全信息的情况尚未扩展。

---

## 137. A Novel Information Workflow for Structural Behavioural Analysis in Dynamic Attributed Graphs

**arXiv ID:** 2609.01640 | [PDF](https://arxiv.org/pdf/2609.01640v1)

**作者:** Gatadi Ashwitha `[一作]` (University of Hyderabad), K. Swarupa Rani `[通讯]` (University of Hyderabad)

**通讯引用:** 42 | [OpenAlex ID](https://openalex.org/A5090812966)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种名为inc‑LPCD_AG的增量信息工作流，在动态属性图中先通过Dynamic CSADW进行链接预测，再用Dynamic I‑Louvain对更新后的图进行社区检测，并在每一步迭代后递归更新，最终得到随时间演化的高质量社区结构。

**💡 创新点**

创新点主要包括：①将链接预测与社区检测两大任务在同一增量流程中串联；②提出Dynamic CSADW，使DeepWalk‑式嵌入在新链接出现时仅增量更新；③改进I‑Louvain为Dynamic I‑Louvain，利用已知社区信息和预测链接实现快速增量社区更新；④通过α权衡结构与属性信息、随机游走长度与次数等参数实现最佳预测；⑤通过实验验证该流程在属性图中显著提升社区质量，解决传统方法缺失链接导致社区不稳的问题。

**🔧 技术方法**

技术手段包括：DeepWalk/CSADW嵌入+随机游走+Skip‑gram、逻辑回归（L2）预测链接；I‑Louvain算法改进为Dynamic I‑Louvain，模量最大化结合惯性模量；使用α调节结构/属性相似度；计算复杂度分析，说明两阶段的时间复杂度与传统方法相比更优。

**📊 数据集**

实验使用的基准数据集为Cora（2708节点、5429条边）、Citeseer（3312节点、4732条边）以及扩充属性的Twitter+（2511节点、37154条边，增加四个属性）。

**📈 对比分析**

对比方法包括inc‑AGGMMR（属性图）和LPCD（非属性图）；评估指标为P、Q（模块化）、ρ（密度）、ϕ（导电性）等。统计显著性检验（配对t检验）显示p≈0.011，Cohen’s d≈5.8，说明inc‑LPCD_AG在所有数据集上均优于对照组，尤其在模块化、密度和导电性上取得显著提升。链接预测阶段的AUC/AP指标亦表现出较高的准确性。

**⚠️ 局限性**

局限性包括：①仅在两跳非连边范围内做链接预测，难以捕获更远邻的潜在链接；②仅支持单一类型节点与属性，无法处理多类型网络；③未考虑边权重信息，未来可扩展到加权属性图；④在极大规模图上的可扩展性仍需进一步验证；⑤增量更新过程仍有计算成本，尤其在高频更新场景下。

---

## 138. Improving Health Literacy through Lay Summarization of Radiological Reports: An Evaluation of BioNER and Retrieval-Augmented Generation

**arXiv ID:** 2609.02396 | [PDF](https://arxiv.org/pdf/2609.02396v1)

**作者:** Egecan Çelik Evgin `[一作]` (Özyeğin University), Olcay Taner Yıldız `[通讯]` (Özyeğin University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对放射学报告的患者友好摘要进行自动化生成，并评估检索增强与实体识别两种改进策略。

**💡 创新点**

首次在统一框架下对 RAG 与 NER 进行对比，并证明实体识别是提升可读性与质量的关键。

**🔧 技术方法**

使用 Qwen 及 BioBART 语言模型，结合 LoRA 微调、Stanza NER、检索增强（Wikipedia）等技术。

**📊 数据集**

在 PadChest、BIMCV‑COVID19+、Open‑i 与 MIMIC‑CXR 四个公开放射学数据集上进行实验。

**📈 对比分析**

通过 0/1/3‑shot 与 LoRA 微调的基线，与 BioNER、RAG 及两者组合的改进模型进行九项指标比较，结果显示 BioNER 在少样本下提升 3–5% 相关性与可读性，Fine‑tuned BioBART+BioNER 在所有指标上优于 Qwen。

**⚠️ 局限性**

RAG 由于检索误匹配导致虚假信息，且组合策略未见提升；此外模型规模与计算资源仍是部署限制。

---

## 139. The Complexity of Justified Representation with Additive Utilities

**arXiv ID:** 2609.02030 | [PDF](https://arxiv.org/pdf/2609.02030v1)

**作者:** Carmel Baharav `[一作]` (Massachusetts Institute of Technology), Agnès Totschnig `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

本文研究在参与式预算和委员会选举中，基于加性效用的正比例代表性公正性（PJR、EJR、FJR）的计算复杂度，并给出常数选民、单元成本、固定效用范围等情形下的完整复杂度图谱；

**💡 创新点**

创新点在于首次给出FJR在单位成本与有限效用范围下的强NP‑难度证明；同时扩展了Expanding Approvals Rule（EAR）到加性效用并证明其满足PJR；并证明所有带有限前瞻的顺序投票规则都无法实现EJR/PJR，揭示了现有多项式时间规则的结构局限；

**🔧 技术方法**

主要技术包括动态规划（DP）实现GCR和其改进版；从Knapsack与PartialTripleCover等经典NP难问题构造多项式归约；对EAR的阈值演化与预算扣除机制进行分析；以及对顺序规则的前瞻性搜索空间不可区分性证明；

**📊 数据集**

本文未使用任何真实数据集，所有结果均为理论复杂度分析与归约构造；

**📈 对比分析**

由于研究目标是理论复杂度而非算法性能，未进行实验比较；结果表明在多项式时间内可实现FJR的情况仅限于极简设置，而在常数效用范围下则不可行；

**⚠️ 局限性**

局限性包括：未给出EJR在加性效用下的多项式时间算法；EAR虽满足PJR，但对其他常见公正性准则的满足程度尚未评估；对单元成本与固定效用范围的FJR强NP难证明仅适用于特定数值区间，未覆盖更广泛的效用上界；

---

## 140. Prompt-Space Meta-Learning Does Not Transfer Across Users: A Frozen-LLM Negative Result

**arXiv ID:** 2609.01615 | [PDF](https://arxiv.org/pdf/2609.01615v1)

**作者:** Liam Byrne `[一作]` (Trinity), Sinead Gallagher `[通讯]` (Trinity)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在冻结大型语言模型（LLM）下，是否能通过在提示空间进行元学习实现跨用户自适应；提出并实验了名为 Muse 的共享提示进化方法；

**💡 创新点**

首次在严格的实验设置中识别并量化了“元目标崩塌（meta‑objective collapse）”，证明元学习在提示空间无法真正学习跨用户的适配结构；提出了一套可复现的评估协议（seed 控制、wrong‑support 控制、invariance/oracle 分解），可用于验证其他提示空间元学习方法的真实性；

**🔧 技术方法**

使用 GEPa（Genetic‑Pareto Evolutionary Prompt）进行提示进化；采用反射式提示改写、固定的 Qwen3‑30B‑A3B 冻结 Backbone；利用 LaMP（用户画像）数据进行支持集与查询集划分；实现了基准评估、Bootstrap 与 McNemar 的统计检验；

**📊 数据集**

LaMP 1（新闻分类）与 LaMP 2（产品评论评分）两套标准化的用户画像基准，分别包含 200 个测试用户；

**📈 对比分析**

与多种对照方法（无个性化、随机用户、PAG、PersonaLink、RAG‑k）进行对比；结果显示 Muse 在分类任务上与手写 seed、错配控制等方法无显著差异；在评分任务上 RAG‑20 以显著优势（MAE 0.250 对比 Muse 0.425，p<0.001）超过所有 persona 方法；

**⚠️ 局限性**

局限性包括：仅在单一冻结模型 Qwen3‑30B‑A3B 上验证；仅针对两类基准任务，未涵盖生成或对话式个性化；提示进化未实现针对每个用户的实例级反馈，导致元目标失去可区分性；对更大或不同规模模型、不同元学习策略的通用性尚待探索。

---

## 141. A Survey on Self-Improving Test-Time Intelligence: Feedback-Driven Adapting, Learning, and Scaling at Inference

**arXiv ID:** 2609.01679 | [PDF](https://arxiv.org/pdf/2609.01679v1)

**作者:** Shuaicheng Niu `[一作]` (South China University of Technology), Cheng Deng `[通讯]` (Hohai University)

**通讯引用:** 13333 | [OpenAlex ID](https://openalex.org/A5015874725)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a2602d71-93ab-4bad-974b-672788df8193` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了测试时智能（Test‑Time Intelligence, TTI）的研究进展，提出统一的更新–计算框架，将测试时适应（TTA）、学习（TTL）与缩放（TTS）纳入同一体系，梳理方法、应用场景与挑战。

**💡 创新点**

创新点在于：①将三类测试时方法整合为单一的“更新与计算”视角；②构建跨领域的细粒度分类与对照表，揭示它们在反馈、更新目标、时间维度等方面的交叉与融合；③提出“反馈驱动的自我改进”作为未来研究的核心路线，指出学习与缩放的互补与混合。

**🔧 技术方法**

主要技术：综述了众多技术路线，包括：归一化层/批归一化自适应、提示/适配器、低秩参数、梯度/无梯度更新、正则化与熵最小化、一致性与自监督预训练、伪标签与自训练、工具/环境交互、外部记忆/缓存等；以及理论分析框架（代理任务对齐、错误界限、渐进自适应等）。

**📊 数据集**

所用数据集：未进行实验，综述涵盖的研究使用了多种公开数据集，如ImageNet、COCO、OpenImages、CIFAR、SVHN、Medical图像集（MIMIC、MS‑MAST）、视频/点云数据集（Kinetics、ModelNet）、语言数据集（CommonCrawl、SQuAD、GLUE）、机器人仿真/真实任务环境等；综述中列举了代表性实验中的数据集名称。

**📈 对比分析**

对比方式：本文并未提供统一实验基准，而是对已有方法进行系统性对比，评述其在不同任务（分类、检测、分割、生成、对话、机器人控制等）与不同部署环境（分布偏移、个性化、持续学习、工具交互）下的性能提升与局限。部分综述提及的基准实验表明，融合学习与缩放的混合系统往往能在保持计算效率的同时实现更高的准确率。

**⚠️ 局限性**

局限性：①综述基于公开论文，缺少统一实验验证；②对新兴方法的评估主要以已有论文结果为准，可能存在选取偏差；③未深入讨论实现细节（硬件开销、实时性）与安全性（自适应失控、恶意干扰）等实际部署挑战；④在跨领域统一框架中仍存在细粒度的理论与实践对齐问题。

---

## 142. From Prompting to Engineering: A Research Agenda for Prompt Engineering in Software Engineering

**arXiv ID:** 2609.02248 | [PDF](https://arxiv.org/pdf/2609.02248v1)

**作者:** Vincenzo De Martino `[一作]` (Universitat Politècnica de Catalunya), Shahbaz Siddeeq `[通讯]` (Tampere University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

组织并主持了 PROMPT-SE 2026 国际工作坊，汇集研究者和实践者围绕软件工程中的提示工程进行技术演讲与结构化讨论，并基于讨论结果提出了系统化的研究议程。

**💡 创新点**

将提示工程提升为软件工程的正式实践，提出五大研究维度（提示工件与标准化、评估与基准、方法与工作流集成、人机协作与技能、治理与技术债务），强调提示工件的可追踪、可维护与可治理，并将其与需求工程、测试、代码生成等传统 SE 活动紧密结合。

**🔧 技术方法**

采用工作坊讨论与现有提示工程技术（LLM、RAG、微调、提示工程）作为方法论框架，但并未开展新的实验技术开发；主要使用社区经验、案例分析和专家共识进行总结。

**📊 数据集**

未使用特定数据集；讨论聚焦于现实 SE 场景中的提示案例（如需求抽取、代码生成、测试生成等），并从已有研究中汲取经验。

**📈 对比分析**

讨论了多维度评估指标（准确性、可靠性、成本、可重复性、技术债务等）以及基准构建与回归测试思路，但未给出实验结果或性能比较。

**⚠️ 局限性**

局限性包括样本量有限（仅约六名参与者）、讨论非系统化且缺乏经验数据、未进行实证验证；提出的研究议程为社区起点，后续需在更大规模实验与实务验证中进一步验证。

---

## 143. WildFab: Multi-Axis 3D Printing from Models in the Wild

**arXiv ID:** 2609.02413 | [PDF](https://arxiv.org/pdf/2609.02413v1)

**作者:** Jiasheng Qu `[一作]` (Chinese University of Hong Kong), Guoxin Fang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一套名为 WildFab 的多轴 3D 打印计算框架，可直接从非闭合、非流形或混合壳体/实体模型生成支持-free 的空间路径并实现全局无碰撞的机器人运动。

**💡 创新点**

创新点在于：1) 结合神经无符号距离场（UDF）与正则化通用绕数场（Reg-GWN）构成混合查询字段，既提供可微的距离/方向查询，又能可靠区分实体/空洞并精确定位表面；2) 基于该混合字段提出高精度迭代投影算法生成空间路径；3) 设计了粗细两级碰撞检测策略，先用 UDF 快速过滤，再用时间变化 Reg-GWN 精准验证，既保持了计算效率又提升了碰撞判定精度。

**🔧 技术方法**

技术手段包括：神经网络（SIREN）逼近 UDF 与 Reg-GWN，梯度投影式路径生成，指导场与旋转场的隐式神经网络优化，基于正则化 GWN 的梯度峰值检测，GPU 并行实现及八叉树加速，粗细碰撞检测算法。

**📊 数据集**

使用多种数据集验证：Thingi10K 的非流形/自交模型、顶点/体素形式的拓扑优化模型、激光扫描得到的点云模型、以及神经隐式模型（如服装解码器）。

**📈 对比分析**

与传统多轴切片方法（如 S^3-Slicer、INF-3DP）和常规 2D 切片器（Cotangent）相比，WildFab 在支持率、碰撞率、路径连续性和表面质量上均有显著提升；实验显示支持率从 25%–95% 提升至近 100%，碰撞率下降 99.5% 以上，工具路径生成在 10 秒内完成 100k 点，整体运行时间约 4–9 小时（含 UDF 训练）。

**⚠️ 局限性**

局限性包括：1) 对极细小或尖锐特征的捕捉仍受神经 UDF 光滑化影响，可能导致微小误差；2) 时间变化 Reg-GWN 的内存占用较高，需要粗细检索策略；3) 需要合适的正则化宽度 δ 与采样间距 h 的配合，参数选择对精度敏感。

---

## 144. Agents That Model Agents: Five Principles Toward a Theory of Mind for 6G Networks

**arXiv ID:** 2609.01779 | [PDF](https://arxiv.org/pdf/2609.01779v1)

**作者:** Hatim Chergui `[一作]` (i2CAT Foundation), Merouane Debbah `[通讯]` (Khalifa University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了在6G RAN中利用大语言模型代理实现自适应控制的理论与实现框架，并提出五条设计原则以提升多代理系统的鲁棒性。

**💡 创新点**

创新点在于将代理交互视为认知通道，构建基于细胞披风（cellular sheaf）与Theory of Mind的统一模型；引入认知信噪比（Cognitive SNR）作为连续信任度量；阐明社会拓扑对一致性与幻觉传播的谱分析；证明递归深度最佳为两层；提出目标一致性是通信容量的根本限制。

**🔧 技术方法**

使用的技术包括：大语言模型（1B参数的通话小型LLM）、细胞披风理论、拉普拉斯算子谱分析、贝叶斯推断与类型估计、认知SNR权重组合、Crawford-Sobel信息容量界限、以及多尺度控制循环（近实时与非实时）。

**📊 数据集**

实验数据集为仿真产生的5种xApp代理（Traffic Balancer、Energy Saver、Slice Guardian、Anomaly Detector、Mobility Manager）在1B参数LLM上的输出；使用虚拟信令风暴、PRB利用率、RRC/Random Access计数等传统E2计数器作为输入。没有公开现成的数据集，而是通过自定义仿真生成。

**📈 对比分析**

通过与传统偏差门控、无信任加权等方法对比，结果显示：认知SNR能在80轮内准确隔离正确代理，深度为2时决策准确率达到100%，而深度1或3时准确率下降；时间一致性分析表明谱间隙越大，达到一致所需时间越短，符合理论预测；当目标误差Δ超过1/4时可传递信息容量降为0，验证目标一致性对通信的决定性作用。

**⚠️ 局限性**

局限性包括：需要持续学习对方推理类型的开销尚未量化；LLM输出的精度（S_j）提取在高速率下仍具挑战；实时社会一致性监测的轻量化工具尚未实现；将运营商SLAs转换为数值代价矩阵缺乏标准化；硬件现场验证仍待完成。

---

## 145. Improving Evaluation Realism with Inference-Time Compute and Deployment Scaffolds

**arXiv ID:** 2609.02302 | [PDF](https://arxiv.org/pdf/2609.02302v1)

**作者:** Axel Ahlqvist `[一作]` (Meridian Visiting Researcher Programme), John Hughes `[通讯]` (Anthropic)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究并实现了两种技术——批判改进（critique refinement）和部署模拟化工具包（DISH），以提升对齐评估的环境逼真度并降低目标模型的评估意识。

**💡 创新点**

创新点在于将目标模型自身用作反馈和选择模型，构建生成‑反馈‑改进循环来逼近真实部署行为；以及在完整代理 harness 中包装目标模型，进一步缩小模拟与实际部署的差距，两者可组合并互补。

**🔧 技术方法**

使用了 Petri 评估框架、生成‑反馈‑改进（cr）与并行深度迭代的推理时计算扩展、DISH 的代理 harness 与工具调用拦截、以及多模型（Sonnet、Opus、GPT‑5.5、Gemini）的偏好与对抗评分。

**📊 数据集**

实验基于 181 条 Petri seed 指令（删减后 149 条），使用 WildChat、Claude Code 等真实部署转录进行 realism win rate 评估，并在 50 条真实部署任务重现中检验环境现实性；涉及的大模型包括 Sonnet 4.6、Opus 4.8、GPT‑5.5、Gemini 3.5 Flash。

**📈 对比分析**

通过与基线 Petri、实时过滤、真实部署对照等方法比较，并使用评估意识、concern、seed adherence、realism win rate 四维指标；在 Sonnet 上 cr2bo4 将 realism win rate 从 10% 提升至 26%，在编码环境中 DISH+cr4 更进一步；与多分支计算匹配基线相比，cr2bo4 在保持更低评估意识的同时实现更高 realism win rate。

**⚠️ 局限性**

局限性包括：某些模型下 seed adherence 明显下降、DISH 在不同模型和场景下提升不均衡且在部署任务重现中无法与 cr 组合、对评估意识的测量仍不完整、批判改进计算成本高且可能存在协同风险，整体仍无法完全消除评估与部署差距。

---

## 146. Beyond Context Windows: Persistent Discovery Context for Data-Centric Agents

**arXiv ID:** 2609.02129 | [PDF](https://arxiv.org/pdf/2609.02129v1)

**作者:** Jalal Mahmud `[一作]` `[通讯]` (Megagon Labs), Jalal Mahmud (Megagon Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了在数据中心代理中持续发现上下文的记忆机制，并通过将先前的意图‑对象映射持久化来改进检索。

**💡 创新点**

提出轻量级的“persistent discovery context”，将成功的发现结果作为可重用记忆，并证明其能显著提升检索质量。

**🔧 技术方法**

使用 TF‑IDF、BM25 与句子嵌入检索器；通过 α 权重融合记忆与元数据；利用相似性检索最近记忆；使用 LLM 自动生成记忆；并进行干扰分析。

**📊 数据集**

在三套真实结构化数据环境中实验：Czech PKDD'99 银行数据、NYC Motor Vehicle Collisions 与 Microsoft Northwind。

**📈 对比分析**

与仅使用原始元数据、仅使用记忆、仅使用注册表检索等方法对比，采用 F1@5 评估；在所有域中加入记忆后 F1@5 提升 0.10–0.18；神经检索下提升 0.05–0.17；LLM 生成记忆仍优于基准。

**⚠️ 局限性**

记忆干扰是主要局限——语义相似但不正确的记忆被激活时会导致检索性能下降，需要干扰感知的记忆选择与年龄管理。

---

## 147. DMRL: Document-Mediated Reinforcement Learning for Skill Optimization in Advertising Recommendation

**arXiv ID:** 2609.02170 | [PDF](https://arxiv.org/pdf/2609.02170v1)

**作者:** Wei Zhang `[一作]` (Shanghai Jiao Tong University), Peng Jiang `[通讯]` (Kuaishou Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于文档介导的强化学习框架 DMRL，用于在线广告推荐系统中的技能优化，通过分层结构将技能编辑与参数干预解耦，并利用长周期奖励预测指导上层策略。

**💡 创新点**

创新点包括：①结构化的双层优化（上层技能编辑器、下层任务代理）；②Dual-Relative Policy Optimization（DRPO）——结合鲁棒的 MAD 归一化、对照组基准以及编辑成本正则的优势估计；③Long-term Reward Predictor（LRP）——通过人群解耦表示和跨注意力历史迁移实现对长周期、异质人群奖励的预测；④两阶段训练策略，先稳健学习奖励再进行策略优化。

**🔧 技术方法**

技术手段包括：LLM 代理（Qwen3‑8B 作为技能编辑器、OpenAI Codex 作为任务代理）、后训练 RL（GRPO 的扩展）、对抗性正则化实现无关人群表示、记忆库与交叉注意力的历史迁移、编辑成本的计算与正则化、鲁棒 MAD 归一化、对照组优势估计、KL 正则、CUPED 方差缩减。

**📊 数据集**

使用真实广告平台的数据集：约 12M 条用户轨迹（包含 AB 参数干预、短期信号、人口标签、长期结果），按时间拆分为训练/验证/测试，记忆库更新为历史已成熟的实验记录。

**📈 对比分析**

与多种基线对比：技能优化基线（LRP+SAGE、LRP+SkillOpt、LRP+SKILLRL）和长期奖励建模基线（DRPO+TFT、DRPO+DelayAdapter）。DMRL 在 LTV 上提升 +0.052%（或 +0.064% 在 8B 版本），在短期/长期 AUD、PES 上均优于基线，显示出在用户参与与变现两方面的均衡提升。

**⚠️ 局限性**

局限性：需两阶段训练导致流程复杂；对模型规模和 rollout 数量敏感；依赖准确的长期奖励预测，若人群分布漂移可能失效；仅针对当前业务指标，未覆盖多目标或其他推荐场景；算力和工程成本相对较高。

---

## 148. NeoMME: A Single-Tower Multimodal-Native Multilingual Foundation Encoder for Efficient Fine-Tuning and Inference

**arXiv ID:** 2609.01657 | [PDF](https://arxiv.org/pdf/2609.01657v1)

**作者:** Aurélien Lac `[一作]` (H Company), Tony Wu `[通讯]` (H Company)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并训练了一种从零开始的双向多模态编码器（NeoMME），该模型将多语言文本令牌和原始32×32像素图像块映射到同一Transformer主体，随后通过动态分辨率和遮蔽扩散预训练进行自监督学习，并在此基础上微调得到视觉文档检索器（NeoMME-Retriever），可同时输出稠密向量和多向量（late‑interaction）表示。

**💡 创新点**

创新点包括：① 一体化单塔双向Transformer，避免传统双塔或生成式VLM的异构结构；② 仅用可见图像块作为条件的遮蔽扩散预训练，兼顾文本与图像的学习；③ 通过层级向量池化与非对称量化实现将高分辨率页面的多向量嵌入压缩至约6 kB，保持95%以上检索质量；④ 提供统一的dense与late‑interaction两种检索头，满足不同规模检索需求；⑤ 在视觉文档检索与文本检索任务上实现了小模型（260 M）在300 M以下模型中的最优表现。

**🔧 技术方法**

技术主要包括：长上下文双向Transformer（滑动窗口+全局注意力）、GQA、RoPE二维位置编码、Zero-last-layer初始化、分布式梯度累积与FlashAttention‑3、MaxSim稀疏核（Late‑Interaction Kernels）、层级向量池化、异步量化、以及多任务自监督与对比学习。

**📊 数据集**

预训练使用约524 B个打包输入，涵盖14种语言的Web文本、代码、数学、自然图像与文档图像；检索微调使用混合文本/多模态查询–文档对，包含多语言检索、代码检索、文档检索数据集（如ColPali、VisRAG）以及生成式查询增强。实验评估使用ViDoRe v3、v2、v1视觉文档检索基准和BEIR‑15文本检索基准。

**📈 对比分析**

与同规模或更大规模的基准模型比较，NeoMME‑260M在ViDoRe v3上取得0.523 nDCG@10，超过所有<300 M模型；NeoMME‑800M在ViDoRe v3上达0.556 nDCG@10，仅落后0.9点于约850 M的Vultron Flash；在文本检索上，Late‑interaction head在BEIR‑15上分别获得0.4881和0.5126 nDCG@10，优于多数稠密检索模型。检索速度方面，-260M在NVIDIA L40S上以≈51 pages/s进行页面编码，约为ColModernVBERT的1.97倍；-800M在更高分辨率下也保持较高吞吐量。

**⚠️ 局限性**

局限性包括：① 预训练数据量相对ModernBERT少约7倍；② 缺乏图像重建或对比目标，导致冻结自然图像性能偏弱；③ 检索监督规模不足，未覆盖所有多模态检索场景；④ 未对混合模态语料进行评估；⑤ 目前未进行教师蒸馏、数据偏差与安全性评估；⑥ 对多语言视觉检索的覆盖不足。

---

## 149. FuDU: A Fuzzy Dual-dimensional Uncertainty Framework for Streaming Active Learning in Industrial Defect Detection

**arXiv ID:** 2609.02212 | [PDF](https://arxiv.org/pdf/2609.02212v1)

**作者:** Zhaoyang Wang `[一作]` (Hebei University of Technology), Xinwei Lyu `[通讯]` (Hebei University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于模糊双维不确定性的 FuDU 框架，用于工业缺陷检测的流式主动学习，核心包含全局原型不确定性量化模块 (PGUQ)、局部双熵不确定性评估器 (DeUE)，以及专家知识驱动的模糊推理采样决策。

**💡 创新点**

创新点：①将模糊控制理论嵌入实时主动学习；②用原型聚类与对比/分散损失实现全局不确定性评估；③设计双熵缺陷不确定性评估器；④通过专家制定的模糊规则实现可解释、动态的采样策略。

**🔧 技术方法**

技术细节：目标检测 Transformer（RT‑DETR 等）、原型聚类、对比损失、分散损失、双熵计算、模糊推理系统（Mamdani）、激活函数 sigmoid、随机采样等。

**📊 数据集**

使用数据集：①核燃料棒缺陷数据集（1000 标注、1000 测试、2000 未标注）；②公共 ELES 光伏单元缺陷数据集。

**📈 对比分析**

比较方法与性能：与随机、阈值、QBC、SPENet、PPAL、Oracle 等主动学习策略对比。FuDU 在核燃料棒数据集上实现 mAP_50 96.0%、召回率 99.3%，仅采样 15.4% 样本，显著优于基线且仅次于 Oracle；在 ELES 数据集也保持 97.1% 召回率，采样比例低。

**⚠️ 局限性**

局限性：①需要专家手工制定与调优模糊规则，规则迁移受限；②对原型数量与更新策略敏感；③在极大批量、多模态或极端光照/噪声环境下的可扩展性与鲁棒性待进一步验证；④额外原型学习与推理带来的计算开销。

---

## 150. A Data-Driven Multimodal Method for Early Detection of Coordinated Abnormal Behaviors in Live-Streaming Platforms

**arXiv ID:** 2609.01649 | [PDF](https://arxiv.org/pdf/2609.01649v1)

**作者:** Jingwen Luo `[一作]` (Peking University), Yan Zhan `[通讯]` (Peking University)

**通讯引用:** 13669 | [OpenAlex ID](https://openalex.org/A5070369468)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个名为MM‑FGDNet的多模态直播电商异常营销检测框架；

**💡 创新点**

核心创新点包括跨模态时序对齐、基于Transformer的时序异常建模以及图神经网络的协同操纵检测，并结合自监督与扩散式数据增强；

**🔧 技术方法**

采用多模态特征编码（视频、音频、文本、行为序列）+动态时序对齐+语义注意融合+Transformer时序建模+图注意网络+自监督对比学习与扩散模型；

**📊 数据集**

构建了包含3.2M视频帧、18.4K音频片段、12.6M评论文本、1.8M交易记录、9.4M用户交互行为的多平台直播电商数据集；

**📈 对比分析**

与传统统计方法、LSTM/GRU、Transformer、GNN、MMBT/CLIP融合模型以及无监督基线对比，MM‑FGDNet在AUC、F1、精确率、召回率、早期检测和跨域泛化指标上均显著优于基线（例如AUC 0.927，F1 0.847，早期检测得分 0.689）；

**⚠️ 局限性**

主要局限在于模型复杂度高、对实时大规模部署有算力与存储需求、依赖完整的用户交互图、对极端冷启动或全新营销手段的适应仍有限。

---

## 151. How Do Prompt Variations Affect Energy Consumption in On-Device LLMs?

**arXiv ID:** 2609.01798 | [PDF](https://arxiv.org/pdf/2609.01798v1)

**作者:** Wei Hu `[一作]` (Georgia State University), Haoxin Wang `[通讯]` (Georgia State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对移动设备上LLM推理的能耗进行大规模实验，研究了认知负荷和措辞模式两种提示属性的影响。

**💡 创新点**

首次将提示级能耗剖析与语义保持、长度控制相结合，构建认知负荷数据集，并展示模型感知的能耗提示优化。

**🔧 技术方法**

使用MLC‑LLM+LM‑Meter进行阶段能耗采样，利用Gemini‑2.5‑Pro生成提示，采用嵌入相似度过滤、DeepEval/EM质量评估，部署Gemma‑2‑2B、Llama‑3.2‑1B、Qwen‑2.5‑0.5B/1.5B、SmolLM2‑360M四种量化轻量模型。

**📊 数据集**

措辞模式鲁棒性评测数据集（七种子属性）和自构造的认知负荷数据集（基于SVAMP、BoolQ、AI2‑ARC）。

**📈 对比分析**

通过比较每令牌能耗、令牌使用、能耗‑质量Pareto前沿，发现认知负荷提升decode阶段每令牌能耗，措辞模式主要通过令牌扩展影响能耗；部分提示可在不牺牲质量的前提下降低能耗。

**⚠️ 局限性**

仅覆盖两种提示属性，提示长度与负荷耦合，评估依赖单一LLM判定器，硬件覆盖与采样粒度有限。

---

## 152. Kirin: Animal Motion Generation from In-the-Wild Video

**arXiv ID:** 2609.01823 | [PDF](https://arxiv.org/pdf/2609.01823v1)

**作者:** Brian Nlong Zhao `[一作]` (University of Illinois Urbana-Champaign), Shangzhe Wu `[通讯]` (University of Cambridge)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

从大规模野生动物视频中重建三维运动并生成符合文本与图像描述的动物动作，随后自动将生成的动作映射到 3D 网格上实现动画。

**💡 创新点**

① 构建首个包含视频、文本与 3D 动作对齐的大规模四足动物数据集；② 开发基于 MDM 的图像+文本条件扩散模型，实现更真实、更符合外观的运动生成；③ 将生成动作与自动 rigging 结合，形成端到端的动物动画流水线。

**🔧 技术方法**

3D 动作重建（SMAL+AniMer + SpatialTrackerV2），图像编码（DINOv3）与文本编码（DistilBERT）融合的 MDM 扩散网络，离线 Image-to-3D（Rodin）与 SMAL 绑定皮肤化流程。

**📊 数据集**

AiM 视频数据集（约30k 影片）经过 SMAL 重建得到 ~30k 3D 动作；利用 Gemini 生成 180k 文本描述，形成 30k 视图×文本×动作三元组；外部测试集 AnimalML3D（1,260 手工动作）。

**📈 对比分析**

与 AniMo、Puppeteer 等基线对比，使用 R-Precision、FID、MM-Dist、Diversity、Multimodality 等指标。实验显示本模型在内部与外部测试集上均优于基线，文本-动作对齐更好、动作多样性与真实性均提升。

**⚠️ 局限性**

仅适用于四足动物，重建质量受极端视角或高速运动影响；模型仍需大量 GPU 计算，且对训练数据的质量与多样性敏感。

---

## 153. EmoStance: Response-Side Affective-Orientation Control for Empathetic Response Generation via Emoji Weak Supervision

**arXiv ID:** 2609.02133 | [PDF](https://arxiv.org/pdf/2609.02133v1)

**作者:** Ziyuan Jin `[一作]` (ShanghaiTech University), Zheng Tian `[通讯]` (ShanghaiTech University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了基于多注释者 Emoji 分布的弱监督情感取向控制框架 EmoStance，用于生成更具共情与适当情绪调适的回应。

**💡 创新点**

1) 将 Emoji 作为弱监督情感取向信号而非标签；2) 构造无名称的隐式情感取向空间并预测响应侧取向；3) 通过连续前缀嵌入将取向向量注入冻结的指令调优 LLM，实现软控制；4) 引入方向一致性重排序提升生成匹配度。

**🔧 技术方法**

多注释者 Emoji 分布聚合、情感取向空间投影、基于角色的转移先验、原型重构、前缀嵌入控制、可选重排序技术；冻结生成器采用 Mistral 7B。

**📊 数据集**

EmojiDialogue（对 EmpatheticDialogues 的扩展，包含 76,489 训练/验证/测试样本）以及原始 EmpatheticDialogues 作为评测基准。

**📈 对比分析**

与 7 个基线（LLM-only、LLM-prompt、LLM-SFT、EmPO-DPO、CASE、APTNESS、Sibyl）在 800 人工对比评估中，EmoStance 总体决胜率 62.2%，在情境特异性和感知响应度上显著优于部分基线；在自动指标（BERTScore、ROUGE‑L、BLEU‑2）上略优于同类控制模型。

**⚠️ 局限性**

弱监督来源于 LLM 生成的 Emoji，缺乏人类真实标签，可能带来文化与偏见；实验仅在短篇双人英语对话上进行，未覆盖长程、多方或知识增强场景；重排序提升了质量但成本显著增加；系统不具备安全性保证，存在被用于情感操控的潜在风险。

---

## 154. IFW-BLS: Dual-Robust Broad Learning System with Intuitionistic Fuzzy Wave Loss

**arXiv ID:** 2609.02422 | [PDF](https://arxiv.org/pdf/2609.02422v1)

**作者:** Mushir Akhtar `[一作]` (Indian Institute of Technology Indore), M. Tanveer `[通讯]` (Indian Institute of Technology Indore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种 Intuitionistic Fuzzy Wave Broad Learning System（IFW‑BLS），通过在 BLS 框架下同时引入样本可信度权重和波浪损失，实现对数据噪声与异常的双重鲁棒性；

**💡 创新点**

创新点在于将直觉模糊可信度与波浪损失统一为一个损失函数，既在样本层面抑制不可靠样本，又在残差层面限制极端误差；同时采用 Nesterov 加速梯度优化，避免传统 BLS 的矩阵求逆；

**🔧 技术方法**

采用 Broad Learning System（BLS）随机特征与增强节点、直觉模糊可信度计算（RBF 核 + 全局类中心与局部冲突）、波浪损失（带可调不对称参数）、Nesterov 加速梯度优化、以及 Friedman 与 Nemenyi 统计检验；

**📊 数据集**

在 30 个 UCI benchmark 数据集上进行评估，并在血液（blood）与马绞痛（horse_colic）数据集上做 5%、10%、15%、20% 的特征异常与标签噪声实验；

**📈 对比分析**

通过 5 折交叉验证与网格搜索，将 IFW‑BLS 与 RVFL、BLS、Wave‑RVFL、NF‑BLS、F‑BLS、IF‑BLS、KRP‑BLS 等九种方法比较；IFW‑BLS 取得平均准确率 87.49%（最高）和平均排名 2.33，统计检验表明其显著优于大多数对比方法；在噪声实验中，IFW‑BLS 的准确率比标准 BLS 提升 7–11%，并保持更缓慢的性能下降；

**⚠️ 局限性**

局限性包括：① 需要一次性计算 RBF 核矩阵，导致在大规模数据上计算成本上升；② 需要手动调参（C、a、λ、γ 等），调参过程繁琐；③ 目前仅在 UCI 公开数据集上验证，缺乏对更复杂高维或深度学习任务的评估；④ 对极大规模样本的可扩展性仍待进一步研究。

---

## 155. Meeting the Coming Wave: The Emerging Politics of AI and Work across 33 Parliaments

**arXiv ID:** 2609.02296 | [PDF](https://arxiv.org/pdf/2609.02296v1)

**作者:** Juliana Chueri `[一作]` (Vrije Universiteit Amsterdam), Petter Törnberg `[通讯]` (University of Amsterdam)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究使用 1,514,950 条跨国议会实质性发言，系统分析 33 个国家 2023‑2026 年间关于人工智能与劳动的议题，揭示各党派在诊断（威胁 vs 机会）和回应（enablement、regulation、training、compensation）上的差异，首次呈现出“enablement‑regulation”轴的党派冲突结构。

**💡 创新点**

创新点在于：①首次将跨国议会语料与大语言模型相结合，对 AI‑工作议题进行大规模语义检索与双层编码；②系统量化不同党派家族与政府地位在 AI‑工作框架上的差异；③揭示 AI 政策冲突主要围绕技术引入速度与监管，而非传统的补偿政治。

**🔧 技术方法**

技术层面：多语言检索 + 大语言模型（Google Gemini）实现 AI‑工作语料二分类；多标签语义编码；多层次分层二项回归（随机效应为国家、党派）用于统计比较。

**📊 数据集**

数据集：1,514,950 条实质性议会发言，包含 33 个国家（包括富裕与中等收入、议会制与总统制）、2023‑2026 年时间段；通过国家专属解析器抽取 AI 候选发言（19,411 条）和 AI‑工作相关发言（5,317 条，直接相关 3,076 条）。

**📈 对比分析**

比较方法：使用分层二项回归估计党派家族和政府身份对诊断与回应比例的影响；模型控制年份、国家固定效应，随机截距为国家与国家‑党派。性能指标：检索准确率 97.8%，召回率 95.7%；诊断/回应框架的 micro‑F1 分别约 0.89–0.90，说明模型分类效果良好。

**⚠️ 局限性**

局限性：①议会发言不等同于实际政策制定，缺乏对立法与执行机制的考察；②样本受可获取记录限制，可能偏向记录完整的议会；③研究时点早期，AI 对劳动力市场的后果尚未显现，补偿议题可能随时间演变；④模型基于相关性，因果关系难以确立，特别是政府身份的作用。

---

## 156. PRISM: An Agentic Multi-Model Architecture for Proactive Safety in Autonomous Transportation Systems

**arXiv ID:** 2609.01623 | [PDF](https://arxiv.org/pdf/2609.01623v1)

**作者:** Joyjit Roy `[一作]` (Independent Researcher), Sushanta Das `[通讯]` (American Center for Mobility)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了PRISM框架，通过并行的环境、轨迹动力学和VRU交互三种风险模型与强化学习推理层，实时生成连续安全评分并给出四阶梯级干预。

**💡 创新点**

创新点在于将静态碰撞概率评分转化为动态可解释安全评分，融合多模型信号并通过RL学习自适应阈值，提供可解释的四层干预策略。

**🔧 技术方法**

主要技术包括逆碰撞概率模型、随机森林环境评估、轨迹LSTM、社交力模型+LSTM的VRU预测、DQN强化学习决策和SHAP特征归因。

**📊 数据集**

使用了nuScenes v1.0-mini、Argoverse 2 Motion Forecasting以及Waymo Open Motion三大公开自然驾驶数据集。

**📈 对比分析**

与传统基于阈值的ADAS做对比，PRISM在三大数据集上均保持跨域一致性，平均安全评分68/100，77.6%情境处于建议级，3.8%近失事件被检测，约11%提升至干预/紧急级。

**⚠️ 局限性**

主要局限包括缺乏真实近失标注、干预阈值未经过数据学习、仅在nuScenes上训练RL、CPU端推理延迟约596ms，且对极端天气样本数量有限。

---

## 157. PRO-Step: Step-level Process Reward Optimization for Retrieval-Augmented Generation

**arXiv ID:** 2609.01658 | [PDF](https://arxiv.org/pdf/2609.01658v1)

**作者:** MinKeon Kim `[一作]` (Sungkyunkwan University), Jaekwang Kim `[通讯]` (Sungkyunkwan University)

**通讯引用:** 9860 | [OpenAlex ID](https://openalex.org/A5100652688)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出PRO-Step框架，通过过程级监督提升检索增强生成模型在多跳推理中的性能。

**💡 创新点**

创新点在于训练生成式过程奖励模型（PRM）同时评估逻辑有效性与证据基础，并利用PRM引导的值树搜索构造步骤级偏好对进行DPO优化，显著抑制错误传播。

**🔧 技术方法**

使用的核心技术包括生成式PRM训练、PRM‑guided值树搜索（VTS）、直接偏好优化（DPO）以及检索增强生成（RAG）机制。

**📊 数据集**

实验数据集涵盖PopQA、HotpotQA、2WikiMultiHopQA、Bamboogle与MuSiQue，共五个单跳与多跳问答基准。

**📈 对比分析**

在所有五个基准上，PRO-Step平均EM 34.5、F1 44.1，超越所有对比基线（包括Search‑R1、ReasonRAG、StepSearch），并在检索失败恢复上表现更佳。

**⚠️ 局限性**

局限性包括依赖开放源PRM标注器带来的标签噪声、训练数据规模相对有限导致在最难数据集（如MuSiQue）上略逊，以及Bamboogle测试样本少导致统计不稳定。

---

## 158. RunSoC 2.0: Scheduling and Allocating Automotive Software Tasks to Hardware Partitions in Heterogeneous MPSoCs

**arXiv ID:** 2609.01614 | [PDF](https://arxiv.org/pdf/2609.01614v1)

**作者:** Daniel Krüger `[一作]` (Daimler Truck AG), Stefan Wagner `[通讯]` (Technical University of Munich)

**通讯引用:** 8247 | [OpenAlex ID](https://openalex.org/A5041829889)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了 RunSoC 2.0 框架，用于在异构 MPSoC 上对汽车软件任务（DAG）进行早期阶段的任务分配与调度。

**💡 创新点**

创新点在于将多目标优化（内存溢出、通信惩罚）与硬实时约束统一到一个可定制的求解器框架，并支持多种求解后端（CBC、CP‑SAT、GA），实现了对异构处理器特性的细粒度建模。

**🔧 技术方法**

技术包括基于 ILP 的任务分配模型、CP‑SAT 约束规划、遗传算法、以及使用 Python Flask 接口进行问题解析与可行性预检。

**📊 数据集**

使用基于 WATERS15 基准的合成汽车 DAG 任务集（10–500 任务）以及三种代表性异构 MPSoC（Renesas R-Car V4H、NVIDIA Jetson AGX Orin、TI TDA4VM）的平台模型。

**📈 对比分析**

通过比较 CBC、CP‑SAT 与 GA 在不同任务规模下的可行率、运行时间和目标值，CP‑SAT 在 500 任务以内始终保持高可行率（>90%）且运行时间在 1000 秒以内，优于其他两种后端。

**⚠️ 局限性**

局限性包括 CBC 与 GA 的可扩展性差、实验仅基于合成数据且未涵盖真实工业任务、模型未考虑内存带宽、总线争用、缓存相关的延迟，以及未支持混合关键性与中断驱动任务。

---

## 159. RideSkill: A Hierarchical Algorithm for Generalized Ride Sharing with LLM-Driven Automatic Evolution

**arXiv ID:** 2609.02250 | [PDF](https://arxiv.org/pdf/2609.02250v1)

**作者:** Zijian Zhao `[一作]` (Hong Kong University of Science and Technology), Mingxuan Yuan `[通讯]` (Huawei)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种分层的骑乘共享调度框架 RideSkill，利用技能仓库、组合器和重定位器在多变的场景与目标下实现订单分配与车辆重新定位。

**💡 创新点**

创新点在于将大语言模型用于离线演化自动化算法设计，并通过技能级抽象、情景感知的组合与顺序重定位，兼具强泛化性、可迁移性与实时可部署性。

**🔧 技术方法**

使用技术包括：大语言模型驱动的 (μ+λ)-进化算法、GDPO 风格的群体相对优势评估、Self‑Check 校验机制，以及基于技能得分的整数线性匹配与顺序重定位策略。

**📊 数据集**

使用纽约市曼哈顿 2026 年 4 月 6–12 日的真实叫车数据进行训练，4 月 13–14 日的数据用于验证与测试。

**📈 对比分析**

与基线（规则、最优匹配、MARL 及现有 LLM 方法）比较，RideSkill 在不同车队规模、速度、容量及目标变化场景中 consistently 获得最高奖励、服务完成率、最低等候/延误与利用率，且在多项指标上显著优于所有基线。

**⚠️ 局限性**

主要局限包括：离线训练仍需大量 LLM 推理计算、对极端未见场景的适应性尚未充分验证、重定位公平机制缺乏理论保证、且仅在单一城市数据上评估，未来需在多城市、多目标动态环境中进一步测试。

---

## 160. DocHop: Benchmarking Out-of-domain Multi-hop Reasoning in Information-Dense Documents

**arXiv ID:** 2609.02059 | [PDF](https://arxiv.org/pdf/2609.02059v1)

**作者:** Zhuoran Yu `[一作]` (University of Wisconsin-Madison), Yong Jae Lee `[通讯]` (University of Wisconsin-Madison)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了 DocHop 基准，用于评估多模态大模型在整合图表与文本上下文的多跳推理能力；

**💡 创新点**

创新点在于：①逻辑先行、逆向生成管线，先设定推理树再生成图表与叙事；②通过可控的推理深度与图表密度实现系统化的难度分布；③引入语义引用标签强制模型必须先在文本中定位实体，再聚合跨图表数值，从而显著提升跨模态推理的挑战性；

**🔧 技术方法**

使用大规模语言模型（如 Gemini‑2.5‑Pro）生成图表与文档文本；构建逻辑树和约束模板实现多跳推理；程序化验证确保数值与推理树一致；采用 ReportLab 渲染单页 PDF 并配合 OCR 读取；评估时采用多模态大模型（GPT‑5.2‑Reasoning、Gemini‑2.5‑Pro‑Reasoning 等）以及多种开源模型；

**📊 数据集**

DocHop 数据集共 2074 个实例，覆盖 6 个任务（取值检索、计数、数值推理、排序、假设推理、事实核查），包含 480 个主题、7 种图表类型，且按推理深度（2‑5）和图表数（2‑6）可控分布；

**📈 对比分析**

与人类对照，最佳模型 GPT‑5.2‑Reasoning 仅 62.83% 而人类达 93.29%；相较于随机猜测（约 10%）和语言模型仅问答基线，性能提升明显；开源模型整体落后 9‑24%；模型性能随推理深度和图表数递增而下降，凸显跨图表推理难度；

**⚠️ 局限性**

局限性包括：①数据为合成合成，未覆盖真实文档的多样性与噪声；②仅包含图表，未考虑表格、图解、UI 等结构化视觉元素；③单页 A4 布局不具备多页、扫描噪声等真实场景的挑战；④仍依赖人类审核，合成过程可能偶有语义或渲染错误。

---

## 161. Signal or Noise? Auditing Rotation-Induced Saliency Drift in Medical and Aerial Imaging

**arXiv ID:** 2609.02224 | [PDF](https://arxiv.org/pdf/2609.02224v1)

**作者:** Khawaja Murad ul Hassan `[一作]` (Independent Researcher), Mehran Ebrahimi `[通讯]` (Ontario Tech University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种训练无关的多视角对齐包装器，通过将不同角度的 Grad‑CAM 结果逆旋转回统一参考帧并平均，显著提升可解释热图的旋转等价性。

**💡 创新点**

创新点在于：①通过拆解 Grad‑CAM 运算链发现漂移主要来自于空间激活的位移，而不是通道权重；②利用逆旋转对齐消除操作噪声；③引入 PEUM（每图解释不确定性分数）实现可解释性审核优先级排序。

**🔧 技术方法**

技术手段包括：多视角旋转采样、空间激活与梯度的逆旋转对齐、特征空间或输出空间的平均聚合、ViT 的 patch 处理扩展、以及基于方差的 PEUM 计算。

**📊 数据集**

实验数据集涵盖 ImageNet‑1K、CUB‑200‑2011、PatchCamelyon（病理图像）与 RESISC45（遥感图像），以及零样本 CLIP‑ViT 在 RESISC45 上的评估。

**📈 对比分析**

方法通过 Pearson 相关的 Equivariance 指标与插入/删除（Insertion/Deletion）指标进行对比，结果显示在 ResNet‑50、VGG‑16 与 ViT‑B/16 上 equivariance 分别提升 36%~247%，相较于单视角 Grad‑CAM 或基于旋转增广的训练均有显著优势；在两类旋转自然域中实现了训练‑free 的可解释性一致性。

**⚠️ 局限性**

局限性包括：仅针对平面旋转；对等价性的改进为经验性，无法保证绝对等价；对 CNN 体系的 faithfulness 可能略有下降；PEUM 仅衡量解释可重复性，无法预测模型错误；以及多视角计算带来额外推理成本。

---

## 162. C$^{3}$T: Counterfactual Causal Reasoning for Sentiment Shifts in Social-Media Conversation Trees

**arXiv ID:** 2609.02131 | [PDF](https://arxiv.org/pdf/2609.02131v1)

**作者:** S M Rafiuddin `[一作]` (Oklahoma State University), Atriya Sen `[通讯]` (Oklahoma State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了CaSiRe数据集并提出C³T模型，对社交媒体话题树中的情绪、情绪转移、因果归因及对话干预的因果推理进行建模与预测。

**💡 创新点**

在对话树结构上融合时间、干预标签与对抗式因果学习，首次实现可解释的因果归因与多步反事实情绪预测。

**🔧 技术方法**

采用Transformer结构的树形编码器、干预标签嵌入、祖先注意力、稀疏归因层以及对抗式因果损失，实现节点情绪、转移、归因与反事实推理。

**📊 数据集**

使用公开的rumor/危机对话树数据集PHEME/RumourEval，并在事件级别拆分上进行训练与评估。

**📈 对比分析**

与文本仅模型、图神经网络、时序GNN以及LLM提示基线相比，C³T在情绪与转移预测上提升约4-5 F1分，归因准确率提升约20%，并在事件外域（OOB）上表现出更好的鲁棒性。

**⚠️ 局限性**

局限在于仅使用观测数据，存在未观测混杂与有限干扰假设；干预标签噪声、祖先窗口限制、模型对话树外影响等因素也可能影响因果效应估计。

---

## 163. Group-Aware Adaptive Retrieval for Evidence Navigation

**arXiv ID:** 2609.02188 | [PDF](https://arxiv.org/pdf/2609.02188v1)

**作者:** June Park `[一作]` (Sungkyunkwan University), Jongwuk Lee `[通讯]` (Sungkyunkwan University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于群组的自适应检索框架GAREN，用于解决推理型检索中的“bounded recall”问题。

**💡 创新点**

创新点在于将文档聚合为语义连贯的群组，并在群组级别进行导航决策，配合文档级重排序实现探索–利用策略以及后期群组证据传播。

**🔧 技术方法**

核心技术包括基于Leiden社区检测的图聚类、LLM生成群组摘要、群组级导航器与文档级重排序器的协同迭代，以及基于RBP权重的群组证据传播。

**📊 数据集**

实验使用BRIGHT（推理型检索基准）、R2MED（医学检索）以及BEIR（多任务检索）三个数据集。

**📈 对比分析**

与Retrieve‑and‑Rerank、SlideGAR、RGS和REPAIR等基线相比，GAREN在BRIGHT上最高可提升8.0% nDCG@10，且在R2MED与BEIR上均保持竞争力，尤其在远离初始检索集的文档检索上表现更佳。

**⚠️ 局限性**

局限包括：群组构建依赖社区检测且固定参数，探索–利用转折点固定，且对强大嵌入或重排序器的依赖可能限制在不同环境下的迁移性。

---

## 164. WMLLM: Self-Evolving Optimization Agents via Predict-Then-Act World Modeling

**arXiv ID:** 2609.01608 | [PDF](https://arxiv.org/pdf/2609.01608v1)

**作者:** Zhongzheng Li `[一作]` (Institute of Automation, Chinese Academy of Sciences), Xiaoguang Zhao `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了WMLLM框架，利用大型语言模型在黑盒优化中先预测候选解的评估结果，再执行操作，从而实现自我进化的优化代理；

**💡 创新点**

核心创新在于将LLM自身作为隐式世界模型，通过在每一步预测结果并与实际评估误差进行自监督，驱动决策与模型共同改进；结合多轮工具调用、基于种群的进化搜索以及轨迹级强化学习（GRPO）；

**🔧 技术方法**

使用大型语言模型（如Qwen系列）与工具调用接口，进行预测-行动循环；实施基于种群的进化策略、Agentic多轮交互、Group Relative Policy Optimization (GRPO)强化学习；提供监督预热（SFT）以初始化预测能力；

**📊 数据集**

主要实验数据集为多目标分子优化基准（SMILES分子，目标包括SA、DRD2、GSK3B、QED、JNK3）；补充实验涵盖圆形排布、整数集统计、Hadamard矩阵等黑盒优化任务，且在ALFWorld上进行跨域测试；

**📈 对比分析**

与OpenEvolve、MOLLEO、GFlowNet、GB-GA、REINVENT、DyMol等方法在同一3000评估预算下比较，WMLLM‑Evolve在Avg. Top‑1、Top‑10及Top‑10 AUC上均取得最高分，且在早期搜索阶段表现出更高的样本效率； ablation实验进一步验证预测、Agentic细化及强化学习对性能的贡献；在额外任务上均实现+0.1172、+0.0336、+0.0715等提升；

**⚠️ 局限性**

局限性在于主要验证于结构化黑盒优化任务，需进一步在更广泛的长期决策环境中验证；模型仍依赖大量评估反馈，昂贵域仍面临交互成本挑战；

---

## 165. Making Revisions Understandable: A Survey of Edit Intentions, Methods, and Applications

**arXiv ID:** 2609.01610 | [PDF](https://arxiv.org/pdf/2609.01610v1)

**作者:** Fangping Lan `[一作]` (Temple University), Eduard Dragut `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文综述了文本修订研究，并以编辑意图为核心视角，统一梳理了语料构建、编辑意图分类、识别方法以及下游应用，形成了完整的六维框架。

**💡 创新点**

创新之处在于首次将编辑意图作为研究核心，提出了可追溯的编辑意图分类谱、方法到应用的映射，并通过文献引用网络展示了分类谱的演化路径。

**🔧 技术方法**

主要采用文献系统综述、语料与方法的结构化整理、编辑意图分类谱构建（基于层次化和可融合设计）、以及对比分析手段（传统特征模型、神经网络、LLM自适应推理）进行技术汇总。

**📊 数据集**

使用了多领域修订语料，包括维基百科、arXiv、学术论文、学生论文、新闻与wikiHow等，涵盖句子、段落和全文级别的编辑动作与编辑意图标注。

**📈 对比分析**

对编辑意图识别方法的比较主要依据标注一致性指标（Krippendorff’s α、MASI）和分类性能指标（F1、准确率）。传统特征模型在表面编辑上表现可接受，但在语义重写上性能不佳；神经网络显著提升多标签识别准确率；LLM的自适应推理提供了更高灵活性，但计算成本和输出一致性仍是挑战。

**⚠️ 局限性**

局限性包括：数据集覆盖受限（非英语或私有写作系统缺失）、编辑意图分类谱差异大导致可比性受限、手工标注成本高、评测框架缺乏统一标准，以及大模型快速迭代可能使部分结论过时。

---

## 166. ArcticSwarm: Deferring Early Consensus in Long-Horizon Multi-Agent Research

**arXiv ID:** 2609.01870 | [PDF](https://arxiv.org/pdf/2609.01870v1)

**作者:** Soyoung Yoon `[一作]` (Seoul National University), Zhewei Yao `[通讯]` (Snowflake AI Research)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了多代理深度研究架构 ArcticSwarm，采用隔离式信息流控制与结构化评审，防止搜索阶段早期共识，从而显著提升长周期任务的准确率。

**💡 创新点**

创新点在于：①将证据收集与整合分离，子代理可发布发现但在隔离模式下不读取同行证据；②在三个承诺边界引入自检、板审与提交门控，确保只有经过多重验证的候选才被共享；③通过隔离与评审门控实现搜索路径多样性，解决传统多代理在无可靠验证器场景下的早期共识问题。

**🔧 技术方法**

技术实现包括：共享公告板系统（BBS）与任务发布、隔离与合作模式的访问控制、自检（self-check）、板审（board audit）与提交门控（commit gate）三层评审、检索工具与PDF读取、页面压缩、文档评分与上下文压缩；使用 Qwen 3.5‑27B、GPT‑5、Sonnet‑4.5 等大型语言模型。

**📊 数据集**

数据集：BrowseComp‑Plus（830 题，基于约 100k 预先收集的文档）和 Live‑Web BrowseComp（1266 题，实时网络检索）。

**📈 对比分析**

比较方法：与单代理、多代理（Duo、Direct Messaging）、MiroFlow 以及提供商深度研究系统（OpenAI、Anthropic）在相同配置下对比。结果显示：Qwen 3.5‑27B 上 ArcticSwarm 取得 82.6%（比 MiroFlow 70.6% 高 12pp），GPT‑5 上 Live‑Web 取得 73.6%（比 OpenAI 54.9% 高 18.7pp，MiroFlow 63.4% 高 10.2pp）。实验也验证了隔离模式与评审门控对性能的贡献。

**⚠️ 局限性**

局限性：依赖昂贵的检索工具与内存，PDF 读取受限；闭源 LLM 的随机性导致结果波动；评审与信息流控制增加 token 成本；资源与效率优化仍待完善，难以在更大规模任务中高效部署。

---

## 167. MELON: A Large-Scale Dataset for Multi-Event Text-to-Long-Video Retrieval

**arXiv ID:** 2609.01654 | [PDF](https://arxiv.org/pdf/2609.01654v1)

**作者:** Chan Hur `[一作]` (ETRI), KyungTae Lim `[通讯]` (KAIST)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了MELON数据集，实现多事件长视频文本检索，并提出MEA损失提升检索精度

**💡 创新点**

首次构建覆盖多事件、长时长视频的大规模检索数据集；提出针对部分事件匹配的多事件感知损失，显著提升模型区分全事件与局部事件的能力

**🔧 技术方法**

双编码器架构（视频编码器、文本编码器）结合CLIP、Long-CLIP等视觉语言预训练模型；采用InfoNCE对比损失并加入自适应边际推拉机制的MEA损失

**📊 数据集**

MELON（12,000+ 长视频，42,526个事件级注释）；对比实验还使用ActivityNet‑Cap和DiDeMo验证通用性

**📈 对比分析**

将各主流CLIP基线（CLIP4Clip、X‑CLIP、PAU等）在MELON上训练，对比仅使用标准对比损失的基线；MEA损失使R@1提升1.5%–7.9%，Rsum提升约5%

**⚠️ 局限性**

对极长视频（>10分钟）仍表现较差；MEA损失需要额外超参调优且在部分模型中可能导致训练不稳定

---

## 168. InfraPatch: Cross-Task Targeted Grayscale Patch Attacks on Infrared-Adapted Vision-Language Models

**arXiv ID:** 2609.02233 | [PDF](https://arxiv.org/pdf/2609.02233v1)

**作者:** Chengyin Hu `[一作]`, Jiujiang Guo `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了InfraPatch，一种针对红外视觉语言模型的局部灰度补丁攻击框架；

**💡 创新点**

创新点在于统一跨任务（分类、生成、VQA）的目标语义攻击，并结合代理定位搜索和任务自适应语义目标；

**🔧 技术方法**

采用白盒梯度优化、总变差正则、目标类/短语/答案匹配损失以及DiffV2IR合成红外图像；

**📊 数据集**

使用了从MS COCO 30类（300张）通过DiffV2IR翻译得到的红外样本；

**📈 对比分析**

与改编后的AdvIB/AdvIC/HCB、随机/中心补丁等对照实验，InfraPatch在10个红外VLM模型上实现86%–100%的针对性成功率；

**⚠️ 局限性**

局限在于仅为数字化白盒单实例攻击，未评估物理可实现性、黑盒转移或传感器噪声鲁棒性。

---

## 169. Memory as an Energy Landscape---Hopfield

**arXiv ID:** 2609.02195 | [PDF](https://arxiv.org/pdf/2609.02195v1)

**作者:** Nima Dehghani `[一作]` `[通讯]` (Massachusetts Institute of Technology), Nima Dehghani (Massachusetts Institute of Technology)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

阐述并推导Hopfield网络及其扩展模型的理论基础，展示从二值记忆、连续激活、稠密与指数型记忆到现代Hopfield与注意力机制的能量景观框架；

**💡 创新点**

将记忆视为物理系统的相空间吸引子，提供对称耦合下Lyapunov能量的全局下降证明，以及通过统计物理方法获得容量边界、稠密记忆阶数提升与指数容量的理论阐释；

**🔧 技术方法**

使用对称耦合、异步局部更新、Lyapunov能量、均值场与复制方法、分子动力学模拟、连续状态梯度下降与softmax能量的数学分析；

**📊 数据集**

主要使用随机生成的无偏二值模式作为测试数据集（无真实数据集），并在数值实验中检验容量与能量下降特性；

**📈 对比分析**

通过理论推导与数值模拟对比，验证二值模型容量≈0.138N，稠密记忆随阶数n提升至O(N^{n-1})，指数模型在满足分离与基底条件时实现O(e^{cN})容量；

**⚠️ 局限性**

局限在于仅适用于对称、无延迟、无自耦合的静态网络，无法描述非平衡、异步非对称或时变突触动态；Hebbian规则为嵌入方法而非完整学习机制；现代Hopfield等价于注意力仅在特定参数映射下成立。

---

## 170. The Memory Trust Gap: Capability-Dependent Failures in Persistent-Memory Agents

**arXiv ID:** 2609.01852 | [PDF](https://arxiv.org/pdf/2609.01852v1)

**作者:** Jundong Hu `[一作]` (PayPal AI), Shekar Ramachandran `[通讯]` (PayPal AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究构建了一个冻结式的多任务基准，专门用来量化长期记忆系统中对过时信息的过度信任及其对代理行为的影响，并系统地分析了不同模型规模、记忆特征与干预手段之间的交互作用。

**💡 创新点**

创新点在于（1）首次把“无记忆”概念拆分为两种互斥含义（Benefit与Safety）并设计对应的闭集动作评分；（2）通过完整的2×2×2×2记忆特征因子实验和跨模型尺寸直接交互检验，揭示了记忆标签、时间戳、来源与位置对过度信任的规模依赖性；（3）验证了干预方式（暴露元数据、预先解决冲突）随模型能力而变化的有效性；（4）在多个模型家族和外部数据集上复现结果，说明问题的普适性。

**🔧 技术方法**

技术方法包括：基准构建与冻结（SHA‑256固定）、四类记忆条件（baseline、agree、stale、conflict）与动作评分；二进制记忆特征因子实验、跨尺寸交互引导的配对差异bootstrap；剂量-反应（recency）深度挖掘；不同表示形式（raw、metadata、oracle）对比实验；外部数据集（RGB、MisBench）以及跨家族Llama-Instruct系列的验证。

**📊 数据集**

使用了多套数据集：本研究自制的冻结v1基准（300个场景，150/150分布在Benefit/Safety），RGB数据集（free‑text与pooled），MisBench数据集（pooled与semantic‑conflict），以及在同一基准上对Qwen3系列（0.6/1.7/4/8B）和Llama‑Instruct系列（1B/3B/8B）进行评估。

**📈 对比分析**

比较方法主要是配对差异的95%分位bootstrap，衡量旧值依赖度（P(答复为旧值)）与净伤害（准确率差值）。结果显示：旧值依赖几乎达到1（0.92–1.00）在所有规模；净伤害随规模上升，尤其在Safety套件中较大模型才出现显著负值；记忆标签、recency与position的影响随规模呈现不同趋势；暴露元数据能显著提升4B/8B的准确率（+0.30/0.53），而预先解决冲突对小模型最有效（+0.46/0.57）。

**⚠️ 局限性**

局限性包括：Benefit套件无法量化净伤害，导致对无记忆基线的解释受限；评测仅采用闭集动作评分，缺乏开放式生成任务的验证；基准只包含单一模型家族的尺寸系列，跨家族的完整规模网格尚未展开；未覆盖写入/更新/检索完整循环，仅在检索时评估冲突。

---

## 171. Guiding LLM Peer Reviewers: The Impact of Score Anchors on Review Evidence and Accuracy

**arXiv ID:** 2609.01905 | [PDF](https://arxiv.org/pdf/2609.01905v1)

**作者:** Judita Preiss `[一作]` (University of Sheffield), Yunhan Yang `[通讯]` (University of Sheffield)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对98份联合健康专业的研究成果，使用LLM（Llama‑3.1‑8B、Falcon‑10B、Llama‑405B）生成评审报告，比较无指导与oracle指导（提供人类参考分数）对最终分数和评审理由的影响。

**💡 创新点**

提出了区分分数复制与评审理由真实改变的评估协议，并构建了点级别框架以比较人类与LLM在支持或降低分数方面的证据使用，从而揭示指导信号如何在分数校正时驱动评审框架的转变。

**🔧 技术方法**

使用生成式LLM（Llama、Falcon）进行文本生成，零样本句子级评价框架标注（DeBERTa NLI）、点级评估点提取与覆盖度评估（Qwen‑3‑32B），并结合统计分析（准确率、MAE、相关系数、覆盖率）评估分数与理由。

**📊 数据集**

98份匿名的英国联邦健康专业单位（Unit of Assessment 3）的内部REF式评估样本，包含标题、摘要、引言结论或全文文本，以及两到四位专家评审的分数和报告。

**📈 对比分析**

通过比较无指导与oracle指导下的分数精确匹配率和均方误差，发现oracle指导将精确匹配率提升至最高0.844（Llama‑405B v4），MAE下降最高0.214；在文本层面，正确校正的分数往往伴随正向评审框架的增强或负向框架的减弱，且LLM对人类升级证据的覆盖率高达0.76，而对降级证据仅约0.2，表明指导在提升分数时更有效。

**⚠️ 局限性**

数据规模有限且仅来自特定领域的候选评估样本，Oracle指导不涉及真实预测分数的不确定性，文本层面依赖自动化标注（零样本分类、Qwen抽取），缺乏专家人工验证。

---

## 172. Emergence of Fibrations, Compression, and Symmetry Breaking in Artificial Neural Networks

**arXiv ID:** 2609.01768 | [PDF](https://arxiv.org/pdf/2609.01768v1)

**作者:** Osvaldo M Velarde `[一作]`, Hernan A Makse `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究深度神经网络在训练过程中自发形成的图论对称结构（纤维、共纤维、覆盖），并证明这些对称是随机梯度下降的稳定吸引子。

**💡 创新点**

创新点在于：①将局部图对称（纤维、共纤维、覆盖）与网络同步学习联系起来；②提出“覆盖粗化定理”说明覆盖对称是SGD的稳态；③利用对称压缩实现模型压缩和持续学习的突破；④将对称理论与多种网络架构（MLP、CNN、RNN、Transformer）统一。

**🔧 技术方法**

技术包括：基于图同调的平衡着色算法识别纤维/覆盖；同步学习与错误同步定理；覆盖压缩规则与反演（lifting）操作；对称破坏策略（Fibration Symmetry Breaking, FSB）；层级阈值搜索实现压缩和损失约束；与传统剪枝、量化、低秩分解等对比。

**📊 数据集**

使用的主要数据集有：MNIST、ImageNet（分类）、Atari Beam Rider（强化学习）、Multi30k（德英翻译）以及顺序ImageNet二分类任务，用于验证对称的出现、压缩效果与持续学习性能。

**📈 对比分析**

与传统剪枝（随机删、L2剪枝）、量化、低秩分解以及现有持续学习方法（如Continual Backpropagation、DCL）对比，覆盖压缩可在保持或提升准确率的前提下，将模型压缩至原始大小的17-18%；在持续学习任务中，FSB将准确率提升至95%，比现有方法高约5%。

**⚠️ 局限性**

局限性包括：①压缩后模型的推理速度提升尚未系统评估；②对称识别算法在大规模网络中的计算开销和可扩展性待进一步优化；③对非前馈结构（如注意力自适应、变形卷积）的对称定义仍需完善；④理论主要在平稳训练下验证，动态数据分布或在线学习情形下的适用性尚未彻底探究。

---

## 173. SCULPT: Training Edge Vision Models for Post-Training Quantization Readiness

**arXiv ID:** 2609.01743 | [PDF](https://arxiv.org/pdf/2609.01743v1)

**作者:** Bharadwaj Kavuri `[一作]` (Valeo Vision Systems), Prasad Deshpande `[通讯]` (Valeo Vision Systems)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在FP32微调阶段加入统计裁剪与均匀分布正则，使模型更易于后训练量化（PTQ）

**💡 创新点**

将稳定百分位裁剪与拓扑感知激活正则结合，训练时不模拟量化且无需后置修复，直接导出可用于PTQ的裁剪边界

**🔧 技术方法**

采用StablePercentileClip、Uniform Activation Distribution Regularization（UADR）、EfficientNet-B0微调、标准PTQ与QAT对比实验

**📊 数据集**

ImageNette（ImageNet-1K的10分类子集）

**📈 对比分析**

在同一QDQ超组下比较，SCULPT PTQ在W8A8达到99.36% top‑1，显著高于FP32 PTQ 90.55%；在W4A8达到80.28%，大幅提升；与QAT比虽略低，但训练步数和时间更短

**⚠️ 局限性**

仅在EfficientNet-B0/ImageNette上验证，复杂感知模型需要进一步测试；对超参数和裁剪百分位设置敏感；目前针对统一量化，其他量化方案的适用性未知

---

## 174. Entangled Representations Amplify Collateral Damage in Unlearning

**arXiv ID:** 2609.02285 | [PDF](https://arxiv.org/pdf/2609.02285v1)

**作者:** Evžen Wybitul `[一作]` (University of Oxford), Christian Schroeder de Witt `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练一系列不同分离程度的254M参数语言模型，并在每个模型上评估三种常用的未学习算法（WGA、WDR、RMU）的保持‑遗忘权衡。

**💡 创新点**

通过在训练阶段引入Selective Gradient Masking（SGTM）来系统控制模型的表示分离度，并首次直接检验分离度对未学习效果的影响，验证了解释性中关于分离度直觉的实证假设。

**🔧 技术方法**

采用SGTM实现表示分离，使用三种未学习方法（WGA、WDR、RMU），并通过Variance Entanglement Score、MMD和Sliced 2‑Wasserstein距离量化分离度。

**📊 数据集**

基于英文维基百科文本，按领域标签划分为生物学（忘记）、相邻领域和保留领域，构建训练和评估集。

**📈 对比分析**

对每个模型-方法组合绘制保持‑遗忘Pareto前沿；在固定遗忘损失（Δℓ_forget = 0.4）下，更分离的模型保持损失降低约4倍（WGA、RMU）或1.3倍（WDR），表明分离度提升显著改善未学习性能。

**⚠️ 局限性**

实验仅通过改变训练过程来控制分离度，可能同时影响其他模型属性；且仅在单一数据集和模型规模上验证，结果的普适性和可迁移性仍待进一步研究。

---

## 175. Graph Neural Team Recommendation: An Integrated Approach

**arXiv ID:** 2609.01631 | [PDF](https://arxiv.org/pdf/2609.01631v1)

**作者:** Md Jamil Ahmed `[一作]` (University of Windsor), Hossein Fani `[通讯]` (University of Toronto)

**通讯引用:** 495 | [OpenAlex ID](https://openalex.org/A5019367061)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种端到端的图神经网络，用来在专家协作图上直接进行链接预测，以实现团队成员推荐。

**💡 创新点**

将团队推荐问题重新表述为图链接预测，消除先前两阶段的技能向量预训练，充分利用多跳关系和结构信息。

**🔧 技术方法**

采用多种注意力型图神经网络（如GAT、HGT、Relational GCN等），并结合负采样、多跳采样等训练策略。

**📊 数据集**

在学术论文（CiteSeerX）和电影（MovieLens/IMDB）两大规模数据集上进行实验。

**📈 对比分析**

与传统的技能嵌入 + 神经分类器的 transfer‑based 方法以及多种 GNN 进行对比，端到端模型在 precision、recall、nDCG、MAP 上均提升 5–10% 甚至更高。

**⚠️ 局限性**

仅适用于已出现的专家与技能，缺乏冷启动支持，对超参数敏感，未考虑模型的动态更新。

---

## 176. Higher-order rich clubs and configuration models on general directed hypergraphs

**arXiv ID:** 2609.01624 | [PDF](https://arxiv.org/pdf/2609.01624v1)

**作者:** Jason P. Smith `[一作]` (Nottingham Trent University), Daniela Egas Santander `[通讯]` (Max Planck Institute of Molecular Cell Biology and Genetics)

**通讯引用:** 188 | [OpenAlex ID](https://openalex.org/A5023858922)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了一种适用于有向超图的高阶富集团（hyper‑rich club）分析框架，能够在更大阶的交互中检测中心顶点的互联程度。

**💡 创新点**

创新点包括：①首次将富集团概念推广到一般有向超图（包含 TO‑hypergraph、head‑and‑tail hypergraph 和无向超图）并给出统一定义；②设计了针对该框架的 TO‑configuration 以及更一般的配置模型，为高阶结构提供合理的零模型；③引入层级（tiered）与组合（combined）两种度量，兼顾不同规模的超边和方向性。

**🔧 技术方法**

技术手段：超图理论与度量定义、富集团曲线（density 及其归一化）、随机配置模型（TO‑CM、一般配置模型）、置换检验、基于权重边计数的数值稳定化等；实现层次化的过滤（按 k‑degree、I‑degree 等）与子超图提取。

**📊 数据集**

使用的数据集包括：神经网络的连通体（C. elegans、Drosophila larva、MICrONS 视觉皮层），诗歌语料库（Shakespeare、Burns、Whitman），感染传播的时间网络（Sociopatterns 访客互动），以及 XGI 超图数据库（无向与 head‑and‑tail 超图）。

**📈 对比分析**

与传统基于图的富集团（pairwise）以及随机超图（ER 与配置模型）进行对比；结果显示在所有实验数据中，高阶富集团能够揭示传统方法无法捕捉的结构特征；归一化曲线显著高于 1 的区间被判定为显著富集，证明方法在统计上具有可靠性。

**⚠️ 局限性**

局限性：①配置模型在极大阶或稠密网络中可能产生重复/退化超边，导致零模型偏差；②对大规模超图的生成与统计检验计算成本高；③目前对可实现度序列的理论阐述尚不完整，限制了模型的通用性；④归一化方法对稀疏网络的数值误差仍敏感，需要进一步改进。

---

## 177. HEAT: Faster Fully Homomorphic Inference via Approximations-Weights Co-Adaptation

**arXiv ID:** 2609.01730 | [PDF](https://arxiv.org/pdf/2609.01730v1)

**作者:** Alessandro Zirilli `[一作]` (Sapienza University of Rome), Emanuele Rodolà `[通讯]` (Sapienza University of Rome)

**通讯引用:** 7244 | [OpenAlex ID](https://openalex.org/A5087051832)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Homomorphic Encryption‑Aware Training（HEAT）方法，让模型中每个非线性近似的迭代次数成为可学习的超参数，并与模型权重共同训练。

**💡 创新点**

通过把迭代次数视为可微分的分布并加入 KL 正则化，实现迭代次数的自适应优化，替代传统统一固定迭代次数，从而显著降低乘法深度与重置次数。

**🔧 技术方法**

使用梯度可微的迭代分布、KL 正则化、三阶段微调流程、HE 近似（Softmax、LayerNorm 等）以及现有的加密推理框架。

**📊 数据集**

GPT‑2 Small 124M 作为基模型，使用 OpenWebText 训练数据集，并在加密环境下评估解码性能。

**📈 对比分析**

与固定迭代（校准）基线对比：迭代次数降低 3.1×、bootstraps 降 1.6×、端到端延迟下降 1.4×，同时提升解码一致性（Top‑1 协议率提升至约 83%）。

**⚠️ 局限性**

实验仅覆盖单一 124M 模型、单一加密参数与单一语言建模任务；在图像、音频或分类任务等不同模态或任务上，性能与优化效果可能会有所不同。

---

## 178. Learning with Volterra Neural Networks: A System Theoretic Perspective

**arXiv ID:** 2609.01928 | [PDF](https://arxiv.org/pdf/2609.01928v1)

**作者:** Haoyu Yun `[一作]` (North Carolina State University), Yufang Bao `[通讯]` (Fayetteville State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种可学习的多核Volterra神经算子kVNN，用于高阶滤波并能兼容CNN结构。

**💡 创新点**

创新在于将高阶交互按卷积层顺序用可学习的多项式核原子分解，避免显式高阶张量，形成可插拔层。

**🔧 技术方法**

采用Volterra滤波框架、核化技术、可学习多核多项式核、局部多图表近似及CNN兼容实现。

**📊 数据集**

使用了UCF101视频动作识别、Set12图像去噪、ImageNet-1K分类等数据集。

**📈 对比分析**

在相同backbone与训练协议下与VNN、Kervolution等基线比较，kVNN在低/高容量设置下均提高1-5%准确率，同时保持或降低参数、GFLOPs及推理时间。

**⚠️ 局限性**

局限在于实验范围有限，未检验极大模型或多模态任务，且高阶核数量需手工调节，可能在更深层次网络中出现训练不稳定或表达瓶颈。

---

## 179. Knowing Is Not Enough: Information Retrievability as a Precondition to Effective LLM Oversight

**arXiv ID:** 2609.01976 | [PDF](https://arxiv.org/pdf/2609.01976v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 180. SpeakPay: Domain-Adaptive LoRA Fine-Tuning of Whisper for Low-Resource Nepali Financial Speech Recognition

**arXiv ID:** 2609.01737 | [PDF](https://arxiv.org/pdf/2609.01737v1)

**作者:** Biraj Subedi `[一作]` `[通讯]` (Independent Researcher), Biraj Subedi (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并公开了首个 Nepali 金融语音数据集 NepFinSpeech-403，并在该数据集上使用 LoRA 对 Whisper large‑v2 进行领域适配，开发了可访问的语音优先电子钱包 SpeakPay。

**💡 创新点**

创新点在于（1）提供公开的低资源金融语音数据集；（2）展示 LoRA 在 1.55B 参数模型上的高效领域适配，单次实验仅需 100–300 条录音即可显著提升性能；（3）引入任务级评估指标 Transaction Success Rate，证明词级 WER 与实际交易成功率的显著差距；（4）系统化错误分析揭示数值混淆模式，为后续改进指明方向。

**🔧 技术方法**

技术手段包括 Whisper large‑v2 预训练模型、LoRA 参数高效微调、基于规则的意图/槽位解析、音频增强实验以及基于 Next.js+Supabase 的全栈应用部署。

**📊 数据集**

使用的主要数据集是 NepFinSpeech‑403（403 条录音，包含 237 个独特 Devanagari 数字，涵盖转账、充值、余额查询三类命令），并通过公开渠道（Hugging Face）提供。

**📈 对比分析**

通过在同一 60 条测试集上对比 zero‑shot Whisper large‑v2、一般 Nepali 语音微调的 Whisper small 以及 LoRA 微调的 Whisper large‑v2，结果显示：零shot WER 129.95% → LoRA 42.58%（相对下降 67.2%），数字识别准确率从 0% 提升至 73.9%，交易成功率从 1.67% 提升至 33.33%（约 20 倍提升）。在 100 条样本时 WER 已降至 57.97%，性能在 300 条样本后趋于饱和。

**⚠️ 局限性**

局限包括：数据量仅 403 条，未进行双注释或 speaker‑disjoint 切分；数值识别仍存在插入/删除、前缀幻觉等错误；交易成功率仍不足以实现无人工确认的自动支付；缺乏针对视障用户的正式可用性评估；基准比较中使用了 Whisper small 而非相同规模的 Whisper large‑v2 进行一般语音微调。

---

## 181. Agentic Settlement Protocol: An Application Profile for Refundable, Delayed-Fulfilment Agent Commerce on Stablecoin Rails

**arXiv ID:** 2609.02208 | [PDF](https://arxiv.org/pdf/2609.02208v1)

**作者:** Behnam `[一作]`, Ritesh Kakkad `[通讯]` (XDC AI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文设计并规范了 Agentic Settlement Protocol (ASP)，在区块链上实现授权-捕获 escrow，并通过三重截止时间的持有模型、履约验证阶梯、退款与曝光控制、收入分成以及单币种不换汇等机制，将离线履约引擎与在线支付系统无缝对接。

**💡 创新点**

创新点在于提出了 ASP 的三重截止时间模型（发行、escrow 失效、引擎库存失效）以避免库存与资金不一致；构建了履约验证阶梯明确谁应提供履约证明并设定挑战窗口；引入退款规则驱动的部分退款与运营商信用风险控制；实现了交易时的收入分成及单币种 invariant，保证无跨链兑换与滑点；以及通过履约类注册实现协议的通用性和可扩展性。

**🔧 技术方法**

技术上基于现有的 x402 HTTP‑402 付款握手、CPP 授权‑捕获 escrow 合约、EIP‑712 签名、ERC‑3009/Permit2 稳定币授权、ERC‑4337 智能账户、ERC‑8004 身份与声誉注册、Solidity 智能合约以及 XDC 网络的确定性最终性。

**📊 数据集**

论文主要使用模拟的履约引擎（实现 hold/commit/cancel/refund 四个操作）以及 XDC 网络的测试/主网数据进行验证，未使用公开的真实业务数据集。

**📈 对比分析**

通过 ASP‑Lite 在 XDC 测试网部署，对每个完整的订单生命周期进行多种故障注入（如引擎超时、操作员崩溃、RPC 中断等），测量从承诺到捕获的区块包含时间、从退款请求到最终确定的时间、Gas 成本以及延迟分布；实验目标是秒级响应和低 Gas 费用，预期性能满足大多数在线业务的实时性要求。

**⚠️ 局限性**

主要局限包括：尚未完成实测数据（Gas、延迟、错误率）；对 A 与 S 阶梯中运营商裁量权的信任仍需进一步验证；缺少分部分割捕获与多腿行程的支持；隐私性不足（金额、钱包、时间公开）；监管和合规问题（行业认证、消费者保护、稳定币监管）仍待确定；定价模型与命名仍待行业共识。

---

## 182. ProSR: Semantic-Prototype-Guided Discrete Modeling for Physically Consistent SAR Super-Resolution

**arXiv ID:** 2609.02377 | [PDF](https://arxiv.org/pdf/2609.02377v1)

**作者:** Byoungwoo Kim `[一作]` (Korea Advanced Institute of Science and Technology), Munchurl Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种面向SAR图像的超分辨率方法ProSR，利用语义原型引导的离散令牌预测，显著提升高频散射结构的重建质量。

**💡 创新点**

核心创新包括离散化的高频细节表示、基于自监督语义先验的原型图生成以及跨注意力的语义对齐机制（PMGA），从而有效抑制传统扩散模型的随机结构失真。

**🔧 技术方法**

技术上融合了Gumbel‑VQ量化、VQ‑GAN编码器/解码器、MaskGIT Transformer、SELF‑SUPERVISED SAFE‑ViT特征提取以及Prototype‑Map‑Guided Attention。

**📊 数据集**

使用公开的Umbra Open Dataset 0.25 m SAR数据集（VV极化、132k个1024×1024像素补丁），构建0.25 m高分辨率与1.0 m低分辨率配对。

**📈 对比分析**

与ESRGAN、SwinIR‑GAN、SPSR、LDM‑15、ResShift、UPSR等SOTA方法在PSNR/SSIM、|ΔTCR|、IW‑SSIM、HaarPSI、FID、Dens、Cov、LPIPS、DISTS等指标上进行对比，ProSR在目标结构完整性与统计真实性上均取得领先，虽PSNR略低但实现了更符合SAR物理特性的超分结果。

**⚠️ 局限性**

局限性包括：仅针对VV极化且0.25 m分辨率；相较某些扩散模型PSNR略低；依赖大规模自监督预训练，计算与存储成本较高，且对其他极化或频段的迁移性能尚待验证。

---

## 183. Detecting Object Hallucinations in Large Vision-Language Models via Cross-Modal Attention Drifts and Mask-Based Verification

**arXiv ID:** 2609.02028 | [PDF](https://arxiv.org/pdf/2609.02028v1)

**作者:** Xuanbing Wen `[一作]` (Xi'an Jiaotong University), Chao Shen `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 CADMP 框架，通过计算相邻层交叉注意力的 KL 散度捕捉注意力漂移，并利用最大漂移层的视觉掩模验证模型对视觉证据的依赖，从而检测大视觉‑语言模型的对象幻觉。

**💡 创新点**

创新点在于将相邻层注意力漂移作为新的内部诊断信号，并结合掩模导致的概率变化进行反事实验证，实现了轻量化、无须额外训练 LVLM 的幻觉检测。

**🔧 技术方法**

使用的技术包括：KL 散度测量注意力漂移、阈值化注意力生成掩模、掩模后概率差值计算、以及将漂移与概率差值输入轻量级 MLP 进行二分类检测。

**📊 数据集**

使用的数据集包括 COCO‑Caption、POPE、Pascal VOC、AMBER 等多种对象、属性和关系幻觉数据集，覆盖不同任务和模型。

**📈 对比分析**

与 IC、GLSim、NLL、Entropy、SVAR、DHCP 等基线进行比较，CADMP 在 16/18 组合作为 ACC 或 AUROC 取得最高分，显著提升了幻觉检测性能。

**⚠️ 局限性**

局限性在于仍依赖 LVLM 的内部注意力信息，对所有幻觉类型的鲁棒性尚未完全验证；掩模比例和阈值的选取对性能有一定影响，且在极少量数据或极端场景下的泛化能力需进一步评估。

---

## 184. Imagine Before Retrieval: Prospective Skill Retrieval for LLM Agents

**arXiv ID:** 2609.01642 | [PDF](https://arxiv.org/pdf/2609.01642v1)

**作者:** Shuo Liu `[一作]` (Sichuan University), Xi Peng `[通讯]` (Sichuan University)

**通讯引用:** 10920 | [OpenAlex ID](https://openalex.org/A5022800038)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 SkillDreamer 的前瞻式技能检索框架，用于解决任务查询与技能文档之间的语义不对齐问题，提升 LLM 代理的任务执行效果。

**💡 创新点**

创新点在于引入了三步前瞻推理流程：①能力感知推理 (CI) 识别任务所需能力并保留查询语义锚点；②前瞻技能生成 (PSG) 以能力为基础生成伪技能；③混合技能检索 (HSR) 将查询语义、能力信息与伪技能进行融合检索并使用锚点进行再排序，从而桥接目标导向查询与执行导向技能之间的差距。

**🔧 技术方法**

使用大型语言模型（如 Qwen3.6‑27B）完成能力推理与伪技能生成，采用向量嵌入模型（Qwen3‑Embedding‑0.6B 等）进行语义匹配，结合 BM25 进行锚点重排序，并在检索后进行伪技能与真实技能的融合评分。

**📊 数据集**

在两个公开基准上验证：SkillRet（4,997 查询，6,660 技能）和 SkillUsage（87 可执行任务，34,396 采集技能），同时在 SRA‑Bench 进一步展示泛化能力。

**📈 对比分析**

与多种检索基线（稀疏检索、通用嵌入检索、技能定制检索）对比，SkillDreamer 在 Recall@K、NDCG、Completeness 上均提升 4–12 分，且在端到端任务执行的 Pass Rate 与 Load Rate 上也能显著提高（例如 Pass Rate 从 31.3% 提升至 37.6%）。

**⚠️ 局限性**

局限性包括：伪技能的质量依赖于 LLM 的推理能力；在极大规模技能库或实时场景下的计算开销尚未评估；以及对多步交互式任务的适用性尚需进一步验证。

---

## 185. Federated Learning on the American Science Cloud using APPFL

**arXiv ID:** 2609.02238 | [PDF](https://arxiv.org/pdf/2609.02238v1)

**作者:** Zilinghan Li `[一作]` (Argonne National Laboratory), Ravi Madduri `[通讯]` (Argonne National Laboratory)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文通过将 APPFL 框架的组件映射到美国科学云（AmSC）的现有服务，实现了跨机构、隐私保护的联邦学习训练流程，并提出了实现该流程所需的服务扩展。

**💡 创新点**

创新点在于提出了利用已有平台原语（身份验证、函数执行、实验跟踪、模型托管）组合实现联邦学习，而非从零构建新基础设施；并明确了四个关键扩展方向。

**🔧 技术方法**

使用的技术包括 APPFL 联邦学习框架、gRPC 与 Globus Compute 通信后端、基于项目范围的 AmSC 身份令牌、差分隐私与安全聚合算法，以及可扩展的聚合服务和跨站点可观测工具。

**📊 数据集**

本文未使用具体实验数据集，而是关注平台级架构与服务集成。

**📈 对比分析**

没有提供实验比较或性能评估，主要以架构设计与功能映射为主。

**⚠️ 局限性**

局限性包括缺乏聚合即服务、周期性触发客户端执行、跨站点可观测性与可写模型托管等关键功能，尚需进一步实现和验证。

---

## 186. ToolGate: An Executable Acceptance Pipeline for Tool-Dependent Scientific Benchmark Construction

**arXiv ID:** 2609.02067 | [PDF](https://arxiv.org/pdf/2609.02067v1)

**作者:** Ke Zhang `[一作]` (University of California, Riverside), Maziar Raissi `[通讯]` (University of California, Riverside)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一套名为ToolGate的可执行生成与筛选流水线，用于自动构造并筛选需使用科学软件解答的评测题目。

**💡 创新点**

创新点在于将候选题目的生成、可执行脚本验证、无工具难度检测和工具辅助求解三大门限整合为一个可审计的多阶段流程，并在后期对门限结果进行随机化和去重处理。

**🔧 技术方法**

主要技术包括大型语言模型（GPT‑5.5、Codex CLI、Claude Opus 4.8）进行候选生成与无工具筛选、Python 与 FEniCSx 软件执行的脚本验证、以及基于接口的工具使用代理执行求解。

**📊 数据集**

使用的“数据集”是由五个专家编写的FEniCSx示例任务作为种子，随后生成了500个候选题目，并对所有候选进行本地验证和多轮筛选。

**📈 对比分析**

相较于仅靠手工编写或单一无工具筛选，ToolGate通过三层门限将可执行标签复制、无工具失效和工具成功率结合，最终从478个通过本地验证的候选中筛选出128个唯一且需工具的高质量题目，展示了显著的筛选效率和可复现性。

**⚠️ 局限性**

局限性包括门限结果受模型版本、推理预算、答案呈现方式等协议相关因素影响；多轮筛选耗时较长且缺乏对人类难度的评估；生成的题目聚焦于两种FEniCSx模板，覆盖范围有限，且仅适用于特定的工具与任务类型。

---

## 187. SocialBuddy: Tailoring Search Agent for Social Scenarios

**arXiv ID:** 2609.01641 | [PDF](https://arxiv.org/pdf/2609.01641v1)

**作者:** Mingxuan Li `[一作]` (Tencent Inc), Wenhui Que `[通讯]` (Tencent Inc)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了专为社交媒体场景设计的代理式搜索框架SocialBuddy，并构建了大规模模拟环境SocialEnv以及专用的混合粒度策略优化框架SocialPO；同时创建了可用于量化评估的SocialSearch Benchmark。

**💡 创新点**

创新点包括：1）首次将代理式搜索延伸到多维度、时空敏感的社交媒体场景；2）构建了覆盖200K用户、1000个ego圈的全新社交模拟环境；3）提出混合粒度奖励与微观Token级KL监督的SocialPO，解决稀疏奖励和信用分配难题；4）提供统一的社交搜索基准。

**🔧 技术方法**

技术手段包括：基于ReAct的LLM代理、5种专用过滤/排序工具、LLM驱动的内容与轨迹合成、监督微调+强化学习、宏观组合奖励、微观Token级KL正则、教师模型指导。

**📊 数据集**

使用数据集：SocialEnv（200K用户、10M帖子、50K轨迹、1k ego圈）与由50名志愿者生成并人工标注的SocialSearch Benchmark（4,893多意图查询）。

**📈 对比分析**

与10个主流LLM（包括GPT-5、Claude-4.5、Gemini、Qwen3.5-397B等）在SocialSearch Benchmark上进行对比。SocialBuddy-35B在Easy/Hard两组均实现任务成功率≈100%，Precision≈0.99/0.96、Recall≈0.97/0.99、Exact Match≈0.98/0.96，较最强开源基线GLM-5.2提升Precision≈7.8%、Recall≈8.2%、EM≈9.7%；超越GPT-5约6–8%。

**⚠️ 局限性**

主要限制包括：1）依赖于大规模LLM和高昂的训练成本（2000+ GPU小时）；2）模拟环境与真实社交网络仍存在差距，可能影响在真实数据上的迁移；3）当前仅覆盖约1,000个ego圈，规模相对有限；4）对敏感隐私信息处理仍需进一步验证。

---

## 188. MultiGhostBench: A Multilingual Benchmark for Long-Form LLM-Generated Text Attribution under Distribution Shifts

**arXiv ID:** 2609.02379 | [PDF](https://arxiv.org/pdf/2609.02379v1)

**作者:** Matteo Greco `[一作]` (University of Melbourne), Jey Han Lau `[通讯]` (University of Melbourne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了一个多语种长文本LLM作者归属（AA）基准，并对多种AA方法在域、作者及语言偏移条件下的性能进行了系统评估。

**💡 创新点**

创新点在于首次提出跨语言、长篇、基于书籍的LLM AA基准，覆盖六种语言、三种文字体系，并设计了多维度OOD评估（域、作者、语言）。

**🔧 技术方法**

采用了基于概率统计（N‑gram、self‑repetition、entropy）、监督学习（XGBoost、RoBERTa、SimCSE）以及指纹匹配的检测技术，并使用XLM‑R实现跨语言迁移。

**📊 数据集**

使用了MultiGhostBench数据集，共928本书（约59K词/书），由五款最新LLM在意大利语、西班牙语、德语、英语、中文和俄语生成，覆盖六种语言和三种文字。

**📈 对比分析**

对比了统计、监督、指纹与Transformer方法，使用宏F1评价；在同语种下无单一方法始终最佳，Transformer在跨语言迁移中表现最优；更多训练数据提升性能，但在OOD（域/作者/语言）下性能普遍下降。

**⚠️ 局限性**

局限性包括仅覆盖六种语言且不包含低资源或特殊语法语言；仅测试五款LLM，难以代表未来模型；人类评测规模有限；实验仅涉及单作者非对抗场景，未涵盖多作者或对抗性攻击。

---

## 189. Zeta-Lite: A Concurrent, Branchable In-Browser SQL Database for Agentic Memory

**arXiv ID:** 2609.01818 | [PDF](https://arxiv.org/pdf/2609.01818v1)

**作者:** Gene Zhang `[一作]` `[通讯]`, Gene Zhang

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一个基于 Zeta 数据库引擎的轻量级 WebAssembly 版 SQL 引擎（zeta-lite），在浏览器中提供完整 PostgreSQL 功能，并实现了单线程下的并发快照隔离事务和全库复制写分支。

**💡 创新点**

创新点包括：1）采用日志中心的异步 MVCC，使单线程即可支持并发快照隔离事务；2）利用 MVCC 日志天然支持全库拷贝写分支；3）不使用 WASI，而是通过 JavaScript 绑定 OPFS、WebCrypto 等 Web API，实现无 Worker、无 COOP/COEP 的持久化；4）在 2.87 MB gzipped 尺寸下实现了比 PGlite 更完整的功能。

**🔧 技术方法**

技术实现：Rust 代码编译为 WebAssembly（wasm-bindgen），使用 OPFS 进行快照持久化、WebCrypto 生成随机数，核心采用日志中心的异步 MVCC、快照隔离（SI）和复制写分支；通过 JavaScript 绑定实现与浏览器环境的交互。

**📊 数据集**

数据集：主要使用合成 OLTP 工作负载（单行 INSERT/UPDATE/DELETE、主键点查）以及 10 分钟持续压力测试中的插入多表数据，未使用公开真实数据集。

**📈 对比分析**

评测方法：在 Chrome 152、Firefox 154 以及本地 bun 1.3.14 参考环境下，对比点查询、插入、事务吞吐等指标；通过持续压力测试验证吞吐稳定性和内存占用；与 PGlite、DuckDB‑wasm 等现有浏览器 SQL 引擎对比。性能方面：点查询 268k–315k ops/s，单行插入 120k ops/s，事务吞吐 60k ops/s；与本地原生引擎差距仅 5–15%。

**⚠️ 局限性**

局限性：单线程实现，无法实现查询内部并发；事务并发仅限事务生命周期内互相覆盖，不能同时执行多条语句；仅支持快照级持久化，缺少每次提交 fsync；分支状态不被快照捕获；不支持 OLAP/列式分析；浏览器时间分辨率限制导致无法获取精确的单操作延迟。

---

## 190. Koopman-Based Robust Model Predictive Control for Nonlinear Systems with Stochastic Intermittent Measurements

**arXiv ID:** 2609.02079 | [PDF](https://arxiv.org/pdf/2609.02079v1)

**作者:** Guanhua Liu `[一作]` (Harbin Institute of Technology), Minghao Han `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种基于深度 Koopman 的鲁棒模型预测控制框架，用于处理非线性系统在随机间歇测量下的控制与约束问题；

**💡 创新点**

创新点包括：1）引入 Lipschitz 约束的深度 Koopman 模型实现全局线性化；2）将测量丢失与重置误差统一建模为马尔可夫跳跃系统；3）推导无分布假设的概率安全半径并用于约束截断；4）设计概率截断软约束保证递归可行性；

**🔧 技术方法**

使用技术包括：深度学习+Spectral Normalization、Koopman 协变变换、马尔可夫跳跃系统建模、概率安全半径计算、软约束 MPC、线性矩阵不等式 (LMI)、Monte Carlo 验证；

**📊 数据集**

数据集：基于 2-DOF 柜台弹性关节陀螺仪系统的仿真数据，结合 Monte Carlo 生成的随机测量丢失序列；

**📈 对比分析**

与基准 ZOH + PD 控制器比较，RMSE/MAE 降低约 70-80%，约束违例率 4.1%（低于 5% 设定），实时计算时间约 2 ms，满足 50 Hz 控制频率；

**⚠️ 局限性**

局限性：1) 依赖于训练好的 Koopman 模型，模型误差与噪声需有上界；2) 对测量丢失概率的上限存在约束；3) 软约束的精度受 λ 参数选择影响；4) 目前仅在仿真验证，实际硬件部署仍待进一步测试。

---

## 191. ISAC with Co-Prime Arrays: Virtual-Aperture Sensing and uplink downlink communications

**arXiv ID:** 2609.01979 | [PDF](https://arxiv.org/pdf/2609.01979v1)

**作者:** Jing Zhang `[一作]`, Derrick Wing Kwan Ng `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种共享天线孔径的ISAC架构，将稀疏共轭阵列嵌入统一线性阵列，实现全双工感知与TDD通信共存，并提出联合资源分配算法以最大化加权通信速率。

**💡 创新点**

创新点在于利用共轭阵列的虚拟大孔径提升感知Cramér–Rao Bound并解析空间-时间采样权衡；在同一物理孔径下实现感知与通信共享；提出基于分数规划与交替优化的非凸求解框架。

**🔧 技术方法**

采用共轭阵列（co‑prime array）设计、CRB分析、分数规划（FP）、交替优化（AO）、半正定松弛（SDP）、SOC约束等凸优化技术。

**📊 数据集**

实验基于随机 Rayleigh 渠道的UAV通信与感知仿真，未使用公开真实数据集。

**📈 对比分析**

通过与分区ULA、等方位感知、最小冗余阵列等基线比较，使用系统总速率、CRB、残余自干扰、视场宽度和目标数量等指标；实验显示所提CPA+优化方案在多种场景下均实现最高速率并满足感知精度。

**⚠️ 局限性**

局限包括对残余自干扰的敏感性、稀疏阵列需更多快照的空间‑时间权衡、算法计算复杂度高以及缺乏实际硬件验证。

---

## 192. Whose Judgments Count? Representation Gaps in Crowdsourced Content Moderation Produce Unequal Protection from Perceived Toxicity

**arXiv ID:** 2609.01625 | [PDF](https://arxiv.org/pdf/2609.01625v1)

**作者:** Zhaodi Chen `[一作]` (University of South Carolina), Byungkyu Lee `[通讯]` (New York University)

**通讯引用:** 999 | [OpenAlex ID](https://openalex.org/A5066089123)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过结合大规模用户判断数据与对数值模拟，探究不同人口学组成的审稿池如何影响在线毒性内容的屏蔽与用户感知的保护差异。

**💡 创新点**

首次量化“内部保护”效应，即审稿池中群体的比例越高，该群体在内容毒性削减中的受益越显著；并揭示即使人口学代表性足够，少数族群（如黑人和 LGBTQ+）仍可能持续受到不充分保护。

**🔧 技术方法**

使用计量经济学的评论固定效应回归、对数值模拟（抽样 20 人审稿队、随机决策、毒性评分加权）以及多规则决策模型（多数投票、超级多数、任何人标记）。

**📊 数据集**

核心数据来自 Toxicity Perspectives 107,260 条来自 Twitter、Reddit 与 4chan 的评论的 16,221 名美国受访者的毒性评分与移除判断，辅以 2024 年 GSS、Prolific 代表性样本和 Reddit 版主人口统计数据。

**📈 对比分析**

对照不同审稿队人口学构成的模拟结果显示：在多重决策规则下，群体毒性削减随其在审稿队中的比例显著提升，体现了“内部保护”现象；在实际人口基准下，黑人与 LGBTQ+ 受保护程度仍明显低于其他群体。

**⚠️ 局限性**

局限包括样本非概率、仅针对美国观点、未模拟专业版主或多阶段审核流程，且受访者未必代表实际平台的审稿决策。

---

## 193. Exploring Breathing-Music Coupling: Using the Breathing Mirror for Somatic Reflection in Piano Performance

**arXiv ID:** 2609.01974 | [PDF](https://arxiv.org/pdf/2609.01974v1)

**作者:** Ziyue Piao `[一作]` (McGill University), Akira Maezawa `[通讯]` (Yamaha Corporation)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过可穿戴呼吸传感器和可视化界面（Breathing Mirror），系统化记录并让钢琴家对呼吸与演奏之间的耦合进行深度自我反思与合作讨论。

**💡 创新点**

① 构建了基线视图、第一人称视图和互相视图的多角度身体反思框架；② 提出了以呼吸为表征的四个“身体主题”用于解释钢琴演奏中的呼吸模式；③ 将传统的呼吸测量与实时可视化、主题分析相结合，首次实现对钢琴演奏者呼吸-音乐耦合的系统性外化。

**🔧 技术方法**

使用碳纳米管应变传感器的纺织集成呼吸带；通过MIDI、音频、呼吸三模态同步采集；实现呼吸分割算法（滤波、峰值检测、多通道融合）；开发Breathing Mirror可视化界面（多模态画布、动态注释、双向回放）。

**📊 数据集**

单一熟练业余钢琴家（35年经验）在四周内完成的四次录制，包含：呼吸信号（胸腹部）、MIDI事件、音频波形；并手工生成与音乐结构对应的时间戳注释，形成自建的时间序列数据集。

**📈 对比分析**

采用主题分析和定量注释相结合的混合方法：先用主题编码提炼四大身体主题，再统计各呼吸模式在演奏段落中的出现频率，绘制时间线可视化；通过与钢琴家主观回忆对比，评估系统对认知差异的揭示；由于样本仅为单一受试者，未与其他技术做直接性能对比。

**⚠️ 局限性**

① 样本量极小（仅一名受试者），缺乏统计推广性；② 受试者自我报告的主观性与记忆误差；③ 呼吸带仅覆盖胸腹两部位，未捕捉到潜在的细微呼吸细节；④ 只分析单段曲目，难以验证发现是否普遍适用于不同风格或难度的钢琴作品。

---

## 194. Consistency as Regularization for Unsupervised Shadow Removal

**arXiv ID:** 2609.01806 | [PDF](https://arxiv.org/pdf/2609.01806v1)

**作者:** Anh-Kiet Duong `[一作]` (La Rochelle University), Jean-Michel Carozza `[通讯]` (La Rochelle University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种完全无监督的阴影去除框架，利用不同观察中的阴影一致性来学习阴影去除。

**💡 创新点**

创新点在于：①仅靠阴影变化的一致性作为自监督信号；②引入全局与局部对比学习以及自我细化图像分组；③使用sigmoid门控和阴影生成器，使模型天然偏向去除阴影。

**🔧 技术方法**

主要技术包括：UNet恢复网络；全局和局部对比损失；自我细化图像分组（Affinity Propagation）；阴影生成器与门控机制；多项式重建损失。

**📊 数据集**

实验数据集：AISTD、SRD、INS、WSRD+、LRSS、视频阴影去除数据集以及SBU用于无监督阴影分割。

**📈 对比分析**

与现有监督与无监督方法（如DHAN、AEF、Mask-ShadowGAN、LG-ShadowNet等）对比，本文方法在RMSE、PSNR、SSIM等指标上与最强无监督方法持平甚至优于，且在部分数据集上超过某些监督模型。

**⚠️ 局限性**

局限性：对细小阴影或微小细节的保留不足；对光照变化和移动物体的细微变化敏感；阴影生成器过于简化，缺乏更真实多样化的阴影模拟；需要更高效的细化策略来提升纹理一致性。

---

## 195. Beyond Human-Likeness: Mapping the Scientific Critique Profiles of LLMs and Human Reviewers

**arXiv ID:** 2609.01895 | [PDF](https://arxiv.org/pdf/2609.01895v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 196. Dictionary-Guided Mutation Operators for Automated HDL Repair

**arXiv ID:** 2609.01775 | [PDF](https://arxiv.org/pdf/2609.01775v1)

**作者:** Maisha Mastora `[一作]` (University of New Hampshire), Dean Sullivan `[通讯]` (University of New Hampshire)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于ANTLR提取的DUT特定词典和单次仿真差异定位的HDL自动修复系统，直接对Verilog源码进行词典约束变异，无需AST或综合。

**💡 创新点**

创新点在于：①词典约束的变异操作避免无效候选；②轻量化仿真差异定位直接对文本行评分；③确定性定向扫描与遗传编程回退相结合，实现多编辑累积修复。

**🔧 技术方法**

采用技术包括ANTLR生成词典、正则表达式文本变异、单次仿真差异故障定位、确定性定向扫描、遗传编程回退、Python+Verilator仿真。

**📊 数据集**

使用CirFix基准套件，共六个DUT族（如counter、mux、decoder、fsm、flip-flop等）。

**📈 对比分析**

与CirFix比较时，在相同机器和Verilator仿真器下，修复14个bug（含3、6编辑实例），单编辑平均速度提升约18倍，评估次数（编译+仿真）显著减少。

**⚠️ 局限性**

局限性包括：对需要同时修改多行的Bug（如交换两条语句）仍受限；词典未覆盖注释行、端口宽度修改、十六进制转二进制等；缺少最小化修补和对弱oracle的鲁棒性。

---

## 197. CACTUS: Mask-Guided Semantic Clean-Label Backdoors in Decentralized Federated Learning

**arXiv ID:** 2609.02450 | [PDF](https://arxiv.org/pdf/2609.02450v1)

**作者:** Chao Feng `[一作]`, Burkhard Stiller `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了CACTUS后门攻击方法，能够在去中心化联邦学习（DFL）环境中通过自然语义触发器在多轮聚合过程中持续传播后门。

**💡 创新点**

创新点在于：①将标签一致的语义对转换为表示层级的对抗偏移，消除触发器位置不确定性的影响；②使用掩码引导的模态特定算子对目标特征进行隔离和耦合，使后门在局部训练与邻居聚合之间保持一致；③通过对抗式训练和对抗偏移的组合，在每轮聚合前对模型进行自我校正，从而提升后门在分布式网络中的稳健传播。

**🔧 技术方法**

使用的技术包括：语义对构造、掩码辅助、对抗式训练、对抗偏移 (pair‑shift) 操作、跨模态（语音、文本、表格、图像）处理、DFL 的邻居聚合规则（FedAvg、Median、Trimmed Mean、Krum、Multi‑Krum、DnC、FLAME、FLTrust、Sentinel）以及敏感度分析。

**📊 数据集**

实验数据集：Speech Commands v0.02（语音关键词识别）、r/UkraineRussiaReport subreddit 标题分类（文本）、Raspberry Pi 行为窗口数据（表格）、CelebA 头发颜色预测（图像）。

**📈 对比分析**

对比方法：在10节点完全连通 IID 场景下，对9种聚合规则分别与BadNets、SABLE、LP‑Attack、Neurotoxin、Batman等基线进行比较。CACTUS 在 Speech Commands、Text、Tabular、CelebA 的平均攻击成功率分别为51.2%、71.1%、25.7%和20.3%，在 Speech Commands、Tabular、CelebA 上均超过基线 49.4、2.5、6.1 分点；在 Text 任务中排名第5，但在 FedAvg、Trimmed Mean 等规则下排名第一；在 Krum/​Multi‑Krum 等鲁棒规则下效果下降。

**⚠️ 局限性**

局限性：①对部分聚合规则（如 Krum、Multi‑Krum）效果有限；②对网络拓扑、节点数、模型规模高度敏感，特别是在大规模或稀疏网络下后门传播受限；③仅评估四个任务，未涵盖更广泛的任务和非 IID 场景；④对攻击的可扩展性和对动态网络的适应性尚未充分验证。

---

## 198. NE-R1: Enhancing Named Entity Recognition Model via Reinforcement Learning

**arXiv ID:** 2609.02366 | [PDF](https://arxiv.org/pdf/2609.02366v1)

**作者:** Meixuan Chen `[一作]` (Peking University), Yanbiao Ma `[通讯]` (Renmin University of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 NE‑R1 框架，采用检索按需（retrieve‑on‑demand）机制提升命名实体识别性能。

**💡 创新点**

创新点在于两阶段训练：先通过多任务指令微调构建检索触发与融合能力，再用基于 Chain‑of‑Thought 的 RL（GRPO）与多维奖励自适应平衡检索与参数推理。

**🔧 技术方法**

技术组合包括生成式 NER、检索增强生成（RAG）、多任务指令微调、Chain‑of‑Thought 推理、Group Relative Policy Optimization（GRPO）以及多维奖励设计。

**📊 数据集**

实验数据集涵盖 OntoNotes 5.0、MIT‑Movie、MIT‑Restaurant、GENIA 进行在域评估，CoNLL‑03 训练后在 CrossNER（5 个跨域）进行零射转移。

**📈 对比分析**

与大型 LLM、域特定基线、标准 SFT 及 RAG 等做严格比较，NE‑R1 在所有在域基准上平均提升 2.52 F1 点，在零射跨域平均提升 1.18 F1 点，并在 MIT‑Movie 上将推理延迟降低约 55%。

**⚠️ 局限性**

局限性包括：依赖实时多源 Web 检索，检索内容随 API、索引或地区变化；实验仅覆盖英文基准，未验证多语言或其他检索后端；当前仅支持单轮检索，无法充分利用多轮检索潜在优势。

---

## 199. EvalDetectBench: A Benchmark for Measuring Evaluation Awareness in Frontier Language Models

**arXiv ID:** 2609.01611 | [PDF](https://arxiv.org/pdf/2609.01611v1)

**作者:** Xinning Li `[一作]` (LASR Labs), Robert Kirk `[通讯]` (UK AI Security Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EvalDetectBench，面向任何 Inspect 兼容评估的开源测评 pipeline。

**💡 创新点**

创新点在于 per‑model probe calibration 与 generator‑identity confound 的 stratified harmonisation。

**🔧 技术方法**

采用提示式分类、AUROC 统计与 z‑score batch harmonisation 等技术。

**📊 数据集**

使用包含 746 条轨迹的评估与部署数据集，涵盖 Claude、GPT、Gemini 系列以及多来源 chat/agentic transcripts。

**📈 对比分析**

通过 12 位 judge、13 个 benchmark 的混合 AUROC 评估，AUROC 范围 0.76–0.89，表明评估意识显著。

**⚠️ 局限性**

局限包括 priming 与诚实报告假设、部署池与真实流量的不匹配以及 harmonisation 估计不确定。

---

## 200. CircuitsDNA: Discovering Unconventional Multi-Accuracy Arithmetic Circuits via Evolutionary Synthesis

**arXiv ID:** 2609.01735 | [PDF](https://arxiv.org/pdf/2609.01735v1)

**作者:** Ruichen Qi `[一作]` (Brown University), Mehdi Saligane `[通讯]` (Brown University)

**通讯引用:** 990 | [OpenAlex ID](https://openalex.org/A5007702824)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 CircuitsDNA 框架，自动进化出能够在同一电路中按需切换多种精度模式的算术电路。

**💡 创新点**

创新点包括多阈值可验证对齐器实现一次性验证多模式误差约束、基于资源限制的可验证驱动搜索（RVS）防止搜索停滞，以及基于反馈的自适应突变加速搜索收敛。

**🔧 技术方法**

采用进化算法（GA/CGP）、布尔可满足性（SAT）形式化验证、ABC 工具的时间限制检验、EWA 反馈突变权重更新，以及统一多阈值 miter 构造。

**📊 数据集**

在多种 DNN 工作负载上进行评估，包括 LeNet‑5、ResNet‑20/18/50、DeiT‑Tiny/S‑Small 等卷积与 Transformer 网络，使用 INT8 量化数据集及完整输入对称评估。

**📈 对比分析**

与精确乘法器以及现有可配置/近似电路设计方法比较，8‑bit 乘法器在 ResNet‑20 INT8 工作负载上 AP 降低 56%，在全激励下 93%，误差在 Fine‑tune 后小于 2%；RVS 能消除同步验证导致的停滞，适应性突变提升搜索速度至 1.33×。

**⚠️ 局限性**

局限性在于对 8‑bit 乘法器的验证效果最显著，扩展到更高位宽时验证成本和搜索时间显著增加；仅评估了少量数据集和工作负载，未验证对更广泛模型的适用性；依赖形式化验证对误差做了严格约束，导致搜索空间受限。

---

## 201. OR-Transformer: Scaling Real-Time Decision-Making to 1,000 Items

**arXiv ID:** 2609.01933 | [PDF](https://arxiv.org/pdf/2609.01933v1)

**作者:** Shuze Daniel Liu `[一作]` (Massachusetts Institute of Technology), Shangtong Zhang `[通讯]` (University of Virginia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为 OR‑Transformer 的深度强化学习框架，用于解决大规模、随机需求的联合补货问题。

**💡 创新点**

创新点在于：①构造了“物品置换等价性”的 Transformer 结构，使模型对物品顺序不敏感；②利用库存动态的可微分性实现路径梯度训练，直接对连续补货量进行梯度传播；③结合离散开单决策的分数梯度，从而同时处理共享固定成本导致的离散与连续决策。

**🔧 技术方法**

使用技术包括：Transformer（自注意力、无位置/索引嵌入），score‑based 策略梯度（处理开单决策），路径梯度（通过可微库存递推传播到补货量），以及价值估计器（Critic）支持训练；实验对比采用 PPO、MLP、Transformer‑PPO、HPO 以及 Gurobi/HiGHS 生成的滚动窗口 MILP。

**📊 数据集**

数据集为仿真生成的库存补货实例，包含 1、4、16、64、1024 个物品，随机生成相关需求、异构交付期、持有/缺货成本、最大订购量和共享固定成本，使用 128 条评估轨迹进行测试。

**📈 对比分析**

与三种学习基线（Transformer‑PPO、HPO、PPO）以及滚动窗口 MILP（Gurobi、HiGHS）对比。结果显示：在 1024 件物品上，OR‑Transformer 的折扣成本约为 0.35M，相比最佳学习基线降低约 75%，相比 Gurobi（允许 6 小时优化）降低 19.1%；在线决策时间比 MILP 快超过 4 百万倍。

**⚠️ 局限性**

局限性：仅在合成数据上验证，缺乏真实供应链案例；对共享固定成本的假设可能限制了对某些业务场景的适用；模型对超大规模（>1024）问题的可扩展性未进一步评估；训练阶段仍需较多样本和 GPU 资源。

---

## 202. Evidential Deep Learning for Multi-Modal Anti-UAV Detection

**arXiv ID:** 2609.01742 | [PDF](https://arxiv.org/pdf/2609.01742v1)

**作者:** Dmitry Golovchits `[一作]` (University of Amsterdam), Ali Mohammed Mansoor Alsahag `[通讯]` (University of Amsterdam)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了证据深度学习（EDL）头、Dempster‑Shafer（DS）证据融合以及基于不确定性的传感器门控在三种反无人机检测基准上的性能表现。

**💡 创新点**

通过在检测、融合与门控上独立控制实验，揭示EDL训练目标提升准确率但不确定性无实用价值，DS融合不优于简单平均，门控无显著效率或准确率提升。

**🔧 技术方法**

使用Dirichlet分布的EDL分类头、DS证据融合规则、基于vacuum的门控机制以及温度缩放做后处理。

**📊 数据集**

在AntiUAV600（热成像跟踪）、TRIDENT（RGB‑音频‑RF二分类）和MM‑UAV（RGB‑IR事件多目标跟踪）三个公开基准上进行实验。

**📈 对比分析**

与重新训练的基线进行对照消融实验，EDL在E1提升5.9pp、E2提升4.8pp准确率，DAUUC从0.5提升至约0.94；DS融合在噪声条件下与平均无显著差异，门控在高跳过率时甚至导致准确率下降。

**⚠️ 局限性**

实验受限于合成噪声、有限样本、单一硬件平台、单类别检测导致的不确定性失效；DS融合仅在缺乏冲突信号时无效；门控在低跳过率下才保持准确率，整体结果对更广泛场景具有局限性。

---

## 203. Omega-N: Interpretable Structural Node Descriptors and Their Applicability Domain

**arXiv ID:** 2609.01633 | [PDF](https://arxiv.org/pdf/2609.01633v1)

**作者:** Alberto Acedo `[一作]` `[通讯]` (Biome Makers Inc), Alberto Acedo (Biome Makers Inc)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于复合结构指数Ω的节点级本地化描述子Omega‑N，并生成十个可解释特征。

**💡 创新点**

创新点在于将全局三角系数、连通密度、代数连通性、度方差等组合成节点级量，并通过度保持的配置模型校准和多尺度个人化PageRank平滑，实现无训练、无属性、无嵌入的高效节点特征。

**🔧 技术方法**

使用谱图理论、配置模型、局部三角计数、个人化PageRank、Fiedler向量能量等技术，并通过随机森林等机器学习方法评估。

**📊 数据集**

在七个小型图、八个节点分类基准、两组金融相关性图和四个蛋白互作网络上进行实验。

**📈 对比分析**

与中心性电池、ReFeX递归特征引擎、GraphWave扩散波、Node2Vec嵌入等对比；在药物靶点优先级中相对中心性提升0.07–0.14，节点分类中多场景获得四项平局和一项微弱优势，系统风险预测中表现略好于原始三角计数，但整体性能不及学习型表示。

**⚠️ 局限性**

局限性包括仅在具有足够三角结构的相对稠密图上有效；对度分布或标签高度相关的网络表现不佳；在有学习表征可用时不如嵌入；对稀疏或无三角图、离散化不收敛等情况存在数值不稳定。

---

## 204. When Decodability Is Not Enough: Logical Validity Representations, Behavioral Dissociation, and Causal Tests in Language Models

**arXiv ID:** 2609.02438 | [PDF](https://arxiv.org/pdf/2609.02438v1)

**作者:** Smitha Muthya Sudheendra `[一作]` (University of Minnesota, Twin Cities), Jaideep Srivastava `[通讯]` (University of Minnesota, Twin Cities)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大语言模型进行逻辑验证实验，评估其行为表现、隐藏状态中的有效性可线性解码性以及干预后对决策的因果影响。

**💡 创新点**

通过构造受控的有效–无效前提–主张匹配对，系统区分模型的行为表达、可线性访问和因果使用；并在不同模板、语义域和推理家族上检验隐藏状态的泛化能力。

**🔧 技术方法**

使用线性对数回归探针（probe）、随机/模板/域/推理家族外部泛化评估、正确性条件AUROC、洗牌标签和元数据对照；对最终prompt‑token隐藏状态进行方向性激活干预并与随机正交方向比较。

**📊 数据集**

使用800条受控逻辑验证样本（5种推理家族×5种语义域×3个难度级别），每条样本都有匹配的有效/无效对，训练/测试模板拆分为600/200。

**📈 对比分析**

在5个模型上，行为层面准确率接近随机（0.47–0.5），而隐藏状态probe在大多数模型中AUROC几乎为1.0，且在模板、域、推理家族外部依旧高分；干预效果极小，几乎不改变决策，随机方向的影响与其相当或更大。

**⚠️ 局限性**

受限于合成数据集和有限的推理家族；仅探究线性可解码性，未考虑非线性或多位置编码；激活干预仅在单一层进行，未能证实全局因果作用；模型规模有限，缺乏对更大模型的验证。

---

## 205. SCX Router: Streaming Zero-Shot Model Selection with a Decoder-KV Classifier and a Real-World Task Ontology

**arXiv ID:** 2609.02292 | [PDF](https://arxiv.org/pdf/2609.02292v1)

**作者:** Ihor Stepanov `[一作]` (Knowledgator), Oleksandr Lukashov `[通讯]` (Knowledgator)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种轻量级的GLiClass路由器，利用解码器-KV结构在不生成回答的情况下为LLM端点预测适用性，并通过任务本体构建多维度任务集与合成验证数据，支持多信号路由与成本感知决策；

**💡 创新点**

创新点在于：①将动态标签分类与持久KV缓存结合，实现零样本端点评分；②设计多信号接口，可同时预测任务类型、难度、推理模式、输出长度等属性；③构建23族115型345子类型、30域的任务本体；④生成15万条可验证与1.5万条评审任务；⑤将学习得分与部署策略分离；⑥提出直接、属性介导、混合与分层的四种路由模式；

**🔧 技术方法**

主要技术包括：GLiClass模型+Qwen3-0.6B解码器、decoder-KV路径、双向DeBERTa评分器、焦点/交叉熵损失、混合训练（基准+合成）、成本与缓存感知的决策层；

**📊 数据集**

使用的数据集为：①基准导向的约104k任务（含模型评分与观测指标）；②合成验证任务15万条（可执行或环境验证）；③合成评审任务1.5万条（人工/LLM评审）；④LiveBench子集与1,500任务评测集；

**📈 对比分析**

评测方法：与均值候选和固定模型对比；在1,000任务子集上，单一候选均值分数0.696，路由器top‑1得分0.707；单标签和多标签的宏F1分别约0.8和0.76；不同子集增益不同，语言、编码、指令跟随等领域表现突出；

**⚠️ 局限性**

局限性在于：评测仅覆盖八个端点，十一端点覆盖不均；部分路由模式缺乏端到端实测；合成数据可能带来作者-模型或评审偏差；未公开完整结果矩阵；未来需加入更多模型、模态、价格层级，并统一版本化评测以验证路由价值。

---

## 206. Network-Aware Forecasting on Wireless Access Points

**arXiv ID:** 2609.01957 | [PDF](https://arxiv.org/pdf/2609.01957v1)

**作者:** Niloo Bahadori `[一作]` (Cisco Systems), Peiman Amini `[通讯]` (Cisco Systems)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种针对企业无线接入点（AP）的两门部署方法，先在目标AP上验证模型与执行路径，再在网络负载下验证服务合规性。

**💡 创新点**

创新点在于将预测推理视为异步共存问题，提出网络感知的部署门槛；同时通过AP与Raspberry Pi 5的匹配评估展示模型迁移的不可预测性。

**🔧 技术方法**

使用了时间序列预测模型（APEX 2）、ARM四核处理器AP、精度/线程/优先级调节等执行配置，结合网络流量压测与实时监控。

**📊 数据集**

利用企业AP采集的无线遥测数据，涵盖客户端加入、AP自检与有线上行三大业务域，形成13条并行预测流。

**📈 对比分析**

与无ML基线对比，默认执行下网络饱和时吞吐量下降7.06%、p99 RTT 上升76%；通过服务优先级执行可消除吞吐量和RTT退化，但预测周期从8.31s拉长至17.5s，仍低于30s阈值。

**⚠️ 局限性**

局限在于未对内存分配与内存系统争用的具体机制进行细粒度拆解，且实验仅覆盖单一AP模型与硬件平台，缺乏更广泛的跨设备验证。

---

## 207. Scalable Kronecker-Fisher Approximation: Efficient Hessian Analysis for Billion-Parameter Language Models Compression

**arXiv ID:** 2609.02451 | [PDF](https://arxiv.org/pdf/2609.02451v1)

**作者:** Viacheslav Yusupov `[一作]` (HSE University), Evgeny Frolov `[通讯]` (AXXX)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大规模语言模型进行可扩展的Kronecker基Fisher矩阵近似，捕捉跨层曲率并在百万到十亿参数规模下实现线性内存占用。

**💡 创新点**

创新点在于证明完整Fisher矩阵可以精确拆分为Kronecker积，并通过截断低秩近似与精确对角线相结合，在不构造全矩阵的情况下实现跨层交互的高质量近似。

**🔧 技术方法**

采用Kronecker分解、Arnoldi迭代求低秩特征、精确对角线计算以及压缩可视化技术，配合梯度累积实现线性时间和空间复杂度。

**📊 数据集**

使用合成数据验证小模型的准确性，并在四个大语言模型（OPT‑350M、Qwen2‑0.5B、OLMo2‑1B、Qwen2.5‑7B）上利用 WikiText2、PIQA、WinoGrande、HellaSwag、ARC‑Easy/Challenge 等数据集进行压缩和微调实验。

**📈 对比分析**

与传统对角/块对角近似和完整Hessian（仅小模型可行）比较，显示Kronecker近似在 R²、层敏感度预测、交互效应捕捉和微调恢复方面均优于对角方法；在 H100 GPU 上，构造十亿参数模型的近似仅需约 40 分钟。

**⚠️ 局限性**

局限性包括：近似基于局部二次模型，无法完全捕捉如 O‑projection 这类因高值残差通道导致的非线性影响；低秩截断可能在极端压缩场景下失效，且对模型架构变化的适用性需进一步验证。

---

## 208. Asymmetric Paired-Annotation Learning for Multi-Structure ULF Pediatric Brain MRI Segmentation

**arXiv ID:** 2609.02210 | [PDF](https://arxiv.org/pdf/2609.02210v1)

**作者:** Ha-Hieu Pham `[一作]` (VinUniversity), Huy-Hieu Pham `[通讯]` (VinUniversity)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发了一种基于nnU-Net的非对称监督策略AURA，用高场(HF)掩模为主目标，将低场(LF)编辑掩模通过可靠性门控制的辅助信息加入训练，实现ULF脑MR图像的多结构分割。

**💡 创新点**

创新点在于将双注释视为不同观测而非等价标签，构造了包含标签不一致、边界、熵、类别可靠性和训练阶段的有限可靠性门，保持仅依赖ULF图像的部署路径。

**🔧 技术方法**

使用nnU-Net架构、软Dice、TopK交叉熵、熵门、边界限定损失以及概率融合与连通组件后处理。

**📊 数据集**

使用LISA 2026 Challenge的79例0.064 T儿童T2加权脑MR图，包含63例训练+16例验证，带有HF和LF两套注释。

**📈 对比分析**

与HF监督基线和AURA单模型进行对比，单模型DSC分别为0.7984（HF）与0.7950（AURA），其集成以0.6/0.4权重得到DSC 0.7988，表现略有提升但差距极小。

**⚠️ 局限性**

局限在于提升幅度极小，评估仅基于16例验证集，缺乏对隐藏测试集和外部ULF队列的泛化验证，且可靠性门参数的贡献尚未单独剖析。

---

## 209. Hearing the Whispers: Black-Box Membership Inference Attacks on Finetuned TTS Models

**arXiv ID:** 2609.01723 | [PDF](https://arxiv.org/pdf/2609.01723v1)

**作者:** Kunlin Cai `[一作]` (University of California, Los Angeles), Yuan Tian `[通讯]` (University of California, Los Angeles)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了针对细调文本到语音模型的黑盒成员推断攻击框架。

**💡 创新点**

创新点包括：系统化查询空间分析与两个评估准则（可评分范围与记忆激发），将“朗诵”查询定义为最强攻击；利用多层WavLM嵌入与动态时间规整实现对连续可变长语音的细粒度对齐；同时在声学和记录级别分别设计适配的特征提取与分类器。

**🔧 技术方法**

使用的技术包括：黑盒查询生成（recitation、continuation 等）；说话人验证编码器 (WavLM + ECAPA‑TDNN) 与自监督音频编码器 (WavLM 多层隐藏状态)；动态时间规整 (DTW) 与 LSTM 汇聚；以及对比实验中的 DP‑SGD 防御。

**📊 数据集**

使用的公开数据集为 VCTK 与 British Dialect，分别在 100 位说话人上进行细调与攻击评估。

**📈 对比分析**

通过 AUC、准确率和 TPR@1%FPR 等指标与随机猜测 baseline 对比；在三款主流 TTS（CosyVoice2、F5‑TTS、XTTS‑v2）上，声学级攻击 AUC ≥0.80，最优情形可达 0.98；记录级攻击 AUC 0.80–0.90；recitation 查询与多层 WavLM+DTW 的组合表现最佳。

**⚠️ 局限性**

局限性包括：对攻击预算（查询次数）敏感；在噪声或截断的参考语音下仍可攻击但性能下降；需要构建阴影模型；DP‑SGD 可降低泄露但会显著影响生成质量；攻击仅适用于黑盒场景，无法直接扩展至全局梯度或内部信息。

---

## 210. CrashDiffuser: VLM-Guided Collision Intent Reasoning for Fine-Grained Safety-Critical Traffic Scenario Generation

**arXiv ID:** 2609.02270 | [PDF](https://arxiv.org/pdf/2609.02270v1)

**作者:** Shucheng Zhang `[一作]` (University of Washington), Yinhai Wang `[通讯]` (University of Washington)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种闭环 VLM‑引导扩散框架，用于生成满足指定目标车辆接触区域（头部、侧面或尾部）的安全关键交通场景。

**💡 创新点**

创新点在于：①将语义碰撞推理与连续轨迹合成解耦，构建分层碰撞意图接口（场景级上下文 + 步级结构动作）；②利用 VLM 生成结构化意图并通过注意力融合至条件扩散模型；③在扩散采样过程中加入可微碰撞引导和多候选选择，实现对目标接触区域的可控生成；④在闭环模拟中不断重新规划以适应目标车辆的动态响应。

**🔧 技术方法**

技术包括 Vision‑Language Model（Qwen3‑VL‑8B‑Instruct）、条件扩散模型（DDPM/DDIM）、轨迹上下文编码器、注意力融合模块、可微碰撞引导、短时序执行与重新规划、MetaDrive 仿真平台。

**📊 数据集**

使用 Waymo Open Motion Dataset（WOMD）中的 441 个通过 CAT 过滤得到的碰撞场景作为训练与评估数据，并额外使用 208 个未触发碰撞的常规场景进行测试。

**📈 对比分析**

与 AdvSim、STRIVE、CAT、SAFE‑SIM、DiffScene 等基线对比，本文方法在单次尝试目标碰撞率 TCR@1 达 50.33%（相较最强基线提升 5.7%），三次尝试 TCR@3 达 67.98%；目标接触区域控制成功率 GCS 40.05%；在 Fréchet 距离、DTW 等自然度指标上亦保持竞争力，展示了更高的碰撞有效性与可控性。

**⚠️ 局限性**

局限性包括：①接触区域可行性受场景几何和目标车辆策略影响；②训练数据中侧面接触过度代表，罕见配置覆盖不足；③仅控制单一对手车辆，且接触区域分辨率仅为三类；④碰撞引导未显式约束加速度、jerk 或舒适度，导致自然度与碰撞强度之间的权衡。

---

## 211. Implicit Manipulation for Skill Selection in LLM Agents with Semantic Matching

**arXiv ID:** 2609.02035 | [PDF](https://arxiv.org/pdf/2609.02035v1)

**作者:** Qikai Wang `[一作]` (University of Electronic Science and Technology of China), Xiaosong Zhang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种在LLM代理中通过语义匹配实现的隐式技能选择操纵方法，能够在不直接给出选择指令的情况下将目标技能推上前台。

**💡 创新点**

提出了ISM（Implicit Skill-Selection Manipulation via Semantic Matching）三阶段策略，利用任务相关提示、技术钩子和提示自然化来隐藏操纵并显著提升目标技能被选中的概率。

**🔧 技术方法**

采用MiniLM语义相似度测量、WordNet/ConceptNet/FrameNet等语义关系桥、LLM驱动的语义约束重写以及对选择器的评估与对照。

**📊 数据集**

收集了62,055条公开技能、56个目标技能、4,765条公共提示（拆分为1,920个案例），覆盖四大任务域，构成实验数据集。

**📈 对比分析**

与Native、随机关键词、领域词汇、Explicit Steering等方法对照；在八个选择模型和四个域上，ISM将目标选择率从15.2%提升至63.5%（单域最高73.5%，仅比Explicit低9.8%）；人类和LLM审查通过率高于Explicit，但规则基防御覆盖有限。

**⚠️ 局限性**

仅针对预执行选择层，未评估后续执行安全；方法依赖模型的语义匹配机制，若选择器改用不同策略可能失效；对正常技能描述的语义重写也可能导致合法性下降。

---

## 212. A Tri-Agent Framework for Evaluating and Aligning Question Clarification Capabilities of Large Language Models

**arXiv ID:** 2609.02054 | [PDF](https://arxiv.org/pdf/2609.02054v1)

**作者:** Yikai Zhao `[一作]` (Amazon), Pradeep Kumar Misra `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个三代理框架（问答澄清代理、受访者代理、评估代理）来动态评估LLM在交互式问答澄清中的表现

**💡 创新点**

创新点在于将LLM同时作为被评估的澄清代理、模拟用户回应的代理以及“LLM-as-a-Judge”评判者，配合合成数据生成与多维度评估指标，实现可扩展、可验证的交互式评测

**🔧 技术方法**

利用Claude 3.5 Sonnet等大型语言模型实现三代理，使用LLM生成思考过程、生成澄清问题、模拟用户回复以及自动评判，并通过提示工程与思考标签提升性能

**📊 数据集**

使用供应链领域的合成数据集，基于预定义基准问题和实体列表生成200个(原始问答, 完整问答)对，涵盖不同程度的歧义和受访者行为

**📈 对比分析**

与传统静态评测相比，框架在200个对话上实现了87%任务成功率、平均4.8回合对话、4.15/5的完整澄清得分，显示在多轮澄清上的可行性和有效性；评估者与人工评分相关系数高达0.87

**⚠️ 局限性**

局限包括对受访者代理质量与偏差的依赖、合成数据与真实对话差异、评估代理对主观指标的可解释性不足，以及对不支持实体处理与模糊回答的鲁棒性仍需提升

---

## 213. Towards One-for-All Robustness Across a Continuum of Threat Levels

**arXiv ID:** 2609.02440 | [PDF](https://arxiv.org/pdf/2609.02440v1)

**作者:** Zhichao Hou `[一作]` (North Carolina State University), Xiaorui Liu `[通讯]` (North Carolina State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Threat Conditional Network (TCN)，实现单模型对连续威胁预算的自适应防御。

**💡 创新点**

创新点在于将表示学习分解为威胁不变共享骨干与轻量级威胁条件适配器，并用分布式对抗训练实现预算连续适应。

**🔧 技术方法**

采用Fourier特征嵌入、FiLM通道归一化调制以及对抗训练分布化等技术。

**📊 数据集**

在CIFAR-10、CIFAR-100、Tiny-ImageNet等视觉数据集以及AG News文本数据集上进行评估。

**📈 对比分析**

与单模型AT、TRADES、MART、NuAT以及多模型MoRE/Ensemble比较，TCN在保持≈11.7M参数的前提下平均精度提升至约74.7%，超过所有基线。

**⚠️ 局限性**

缺点是仍需在推理时给定威胁级别，未解决未知攻击强度下的自动检测与推断。

---

## 214. Harness Engineering in LLM Tool Use via Agent-Native Reusable Tool Primitives

**arXiv ID:** 2609.01736 | [PDF](https://arxiv.org/pdf/2609.01736v1)

**作者:** Haibo Jin `[一作]` (University of Illinois at Urbana-Champaign), Haohan Wang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 2919 | [OpenAlex ID](https://openalex.org/A5072244531)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了Tool Primitives、ToolFace与HEART框架，提升LLM工具调用的可扩展性、鲁棒性与多步推理能力。

**💡 创新点**

创新点在于用自然语言包装工具、动态检索工具并构建Planner‑Router‑Verifier多代理管线，实现无API schema暴露、嵌套/多轮调用与反馈驱动的恢复。

**🔧 技术方法**

技术包括LLM（Qwen3‑8B等）工具包装、语义检索ToolFace、Planner/Router/Verifier多代理协同，以及ReACT+DFSDT等对比方法。

**📊 数据集**

使用数据集有ToolBench、NESTFUL、τ²‑Bench、ACEBench、BFCLv4以及自建的50个真实世界任务。

**📈 对比分析**

通过与SFT模型和GPT‑5.4/Claude‑4.6‑Sonnet/Gemini‑3.1‑Pro等商业模型在五大基准上对比，平均提升10%/6%，在真实任务上实现84%完成率，API成本降低至85%。

**⚠️ 局限性**

局限在于仍需多轮重规划、计算成本高、对极端复杂或新型工具场景的适配有限，且实验主要基于公开基准和自建任务。

---

## 215. XMerge: Cross-Axis Selection and Reconstructive Layer Merging for LLM Depth Compression

**arXiv ID:** 2609.02083 | [PDF](https://arxiv.org/pdf/2609.02083v1)

**作者:** Jundong Hu `[一作]` (PayPal AI), Shekar Ramachandran `[通讯]` (PayPal AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种后训练层压缩方法XMerge，先用交叉轴（相对幅度与角度）筛选可删除的Transformer块，再通过局部激活匹配重构，将被删块与邻块合并为单一标准块，保持原始接口与参数量；

**💡 创新点**

创新点在于（1）交叉轴无参数选择器，结合相对幅度与角度两种衡量；（2）局部重构只利用现有块参数，不增添新模块或参数；（3）完全无标签、无端到端微调，构建成本低，兼顾推理速度与质量；

**🔧 技术方法**

使用相对幅度与角度评分、交叉轴最大融合、局部激活匹配重构（300步Adam优化）以及标准Transformer结构；评估采用WikiText‑2、CORE 22任务集、MMLU、WikiText‑2测试困惑度等指标；

**📊 数据集**

主要使用WikiText‑2（训练集做重构、测试集做困惑度评估），CORE 22任务集与MMLU作为下游任务评估；

**📈 对比分析**

在7个Llama/Qwen 0.5B–8B模型上，与ShortGPT、LaCo、MKA、SWM、CoMe等5基线比较。k=4时XMerge在CORE、MMLU上分别在6/7模型上领先，且在极端压缩时避免极高的困惑度；同时保持零折叠和无额外推理参数；

**⚠️ 局限性**

局限包括：构建时需要数分钟至数小时的梯度优化；仅在单一随机种子下测试，未评估多种种子波动；仅针对decoder‑only Llama/Qwen，未验证混合专家、编码器‑解码器或多模态模型；重构目标与评估指标高度相关，可能导致指标偏好；未对hallucination、安全性等方面进行评估。

---

## 216. Benchmarking Language Models for Statistical Problem Formulation

**arXiv ID:** 2609.01982 | [PDF](https://arxiv.org/pdf/2609.01982v1)

**作者:** Chen Wang `[一作]` (Tsinghua University), Ke Deng `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估LLM在统计问题表述阶段的能力，拆分为问题分类与变量识别与角色分配两子任务；

**💡 创新点**

首创StatFormBench基准，覆盖20类粗粒度、85类细粒度统计问题，涵盖多学科教材与真实案例，且使用场景化重写；

**🔧 技术方法**

使用多种公开/闭源LLM（Claude、GPT-5、Gemini、Kimi、DeepSeek、Qwen3.5等）并在零样本及提示策略下进行评测；

**📊 数据集**

构建1,013条样本，来自五本跨领域统计教材和一套数据科学案例库，数据形式包括表格、摘要统计、文本描述等；

**📈 对比分析**

对比14款LLM，最优零样本细粒度分类准确率72.0%，最佳变量集重叠63.2%，不同模型在分类与变量识别上表现各异，提示策略提升有限；

**⚠️ 局限性**

局限性包括样本来源于已解题教材和案例而非真实咨询，类别分布不均，教材为英文、案例库为中文可能引入跨语种偏差，且未充分捕捉真实咨询的歧义与不完整性。

---

## 217. SPAR: Enhancing Industrial-Scale Generative POI Recommendation via Real-World Spatial Perception

**arXiv ID:** 2609.02062 | [PDF](https://arxiv.org/pdf/2609.02062v1)

**作者:** Fangye Wang `[一作]` (AMAP, Alibaba Group), Pengjie Wang `[通讯]` (AMAP, Alibaba Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 SPAR 框架，通过 SI‑SID、MG‑CPT 与 TV‑SFT 三个阶段将真实城市空间知识嵌入生成式 POI 推荐。

**💡 创新点**

创新点在于将经纬度编码直接融合到 token 生成、持续预训练多层级地理数据并用任务向量冻结防止空间知识遗忘。

**🔧 技术方法**

使用 sinusoidal 地理编码、RQ‑Kmeans 聚类、Transformer LLM、连续预训练 (CPT)、LoRA 与 Task‑Vector Anchored SFT 等技术。

**📊 数据集**

使用公开基准（NYC、TKY）以及四个阿里地图工业数据集（北京、上海、天津、浙江）和 25 个自构建的多层级地理数据集。

**📈 对比分析**

与传统序列模型、基于 Transformer 的推荐和现有生成式推荐模型（TIGER、GNPR‑SID、PLUM）对比，SPAR 在所有指标上均提升 10%–40% 以上，工业规模上平均提升 38% 以上。

**⚠️ 局限性**

局限在于模型规模大、预训练与微调过程复杂，对地理数据依赖强，且在极端稀疏或多城市迁移场景下仍需进一步验证。

---

## 218. Synergistic Information Disentanglement for Omni-modal Slide Representation Learning in Computational Pathology

**arXiv ID:** 2609.02118 | [PDF](https://arxiv.org/pdf/2609.02118v1)

**作者:** Mingxin Liu `[一作]` (Nanjing University of Information Science and Technology), Jun Xu `[通讯]` (Nanjing University of Information Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于协同信息解耦的自监督学习框架ΦID，用于整合病理切片、基因组和报告信息以学习全切片表示。

**💡 创新点**

创新点在于用部分信息分解（PID）理论实现协同信息最大化，取代传统冗余对齐，利用Synergistic Information Bottleneck（SIB）和ΦID目标显式压缩冗余并挖掘高阶交互。

**🔧 技术方法**

采用多模态域特定编码器、ABMIL滑动窗口聚合、SIB瓶颈、Gaussian Canonical Projector（GCP）以及对称InfoNCE对齐等技术。

**📊 数据集**

训练数据来自TCGA的全模态乳腺（BRCA）和肺癌（NSCLC）病例；下游评估使用BRACS、CPTAC‑BRCA、CPTAC‑NSCLC等公开独立数据集。

**📈 对比分析**

与多种MIL模型和SSL基线（UNIv2、TANGLE等）对比，ΦID在8项少样本任务中均实现最高AUC，平均提升约3–5%，尤其在乳腺子类分类和基因突变预测上表现突出。

**⚠️ 局限性**

局限在于需完整的三模态配对数据，主要针对图像+文本+基因组，且对其他模态或跨域迁移的鲁棒性尚待验证。

---

## 219. ZipTok3D: High-Fidelity 3D Tokenization with Compact Token Prefixes

**arXiv ID:** 2609.01740 | [PDF](https://arxiv.org/pdf/2609.01740v1)

**作者:** Mingda Lin `[一作]` (Zhejiang University), Bohan Zhuang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 ZipTok3D，一个能够从极短 token 序列实现高保真 3D 重建的 tokenizer。

**💡 创新点**

创新点在于利用 nested dropout 训练可递归解码的前缀，并采用参数共享的 Transformer 迭代细化，使得前缀能先保存全局几何信息后逐步展开细节。

**🔧 技术方法**

采用 nested dropout、参数共享的 Transformer 迭代细化、triplane 码字以及 occupancy MLP 等技术进行编码和解码。

**📊 数据集**

使用 ShapeNet‑v2 和 TRELLIS‑500K 两个大型 3D 数据集进行实验。

**📈 对比分析**

与 COD‑VAE、VecSet、3DILG 等基线比较，ZipTok3D 在 ShapeNet 只需 1 个 token 就能达到 32‑token COD‑VAE 的重建质量，在 TRELLIS 仅需 4 个 token 就能超越 COD‑VAE‑32，且在类条件生成任务中保持接近 COD‑VAE‑32 的性能。

**⚠️ 局限性**

对极低 token 依赖多步迭代，导致解码时间增长；训练时对 token 预算的分布和 nested dropout 参数选择较为敏感，可能在更复杂或更大规模场景下表现受限。

---

## 220. Hybrid Retrieval-Augmented Generation with Knowledge Graph Expansion, RRF Fusion, and Per-Chunk Grounded Evaluation for Enterprise Document Search

**arXiv ID:** 2609.01617 | [PDF](https://arxiv.org/pdf/2609.01617v1)

**作者:** Harish Saragadam `[一作]` (Vodafone Idea -- SNOC), Meghana Pujari `[通讯]` (Vodafone Idea -- SNOC)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套多信号检索与本地化LLM生成的企业文档检索系统DocuSearch，融合密集向量检索、BM25、知识图谱邻居扩展，并通过递归排名融合与交叉编码重排序，最终通过逐块评估循环实现高精确度与高覆盖率的答案生成。

**💡 创新点**

首次在RAG中引入三种检索信号并采用加权递归排名融合、基于LLM的逐块上下文缺失检测、充分性评分与答案根源验证，形成完整的“逐块评估循环”，显著降低幻觉并提升答案根源率。

**🔧 技术方法**

使用BGE-Large-EN向量、Qdrant ANN、SQLite FTS5、知识图谱邻居扩展、LangGraph工作流、Cross-Encoder重排序、MMR多样性筛选、Mistral LLM本地推理、Dash UI。

**📊 数据集**

在内部电信运营商文档语料库（多供应商手册、SOP、RCA报告、审计记录）上评估，包含120个人工标注查询。

**📈 对比分析**

与仅密集检索和密集+BM25基线对比，DocuSearch在Precision@10提升至0.69、Recall@10提升至0.79，答案根源率89.6%（相较单通道RAG的71.2%）并将幻觉率从21.4%降至6.8%；平均端到端延迟12.6秒。

**⚠️ 局限性**

主要瓶颈在逐块评估阶段的顺序LLM调用导致延迟高；RRF权重和KG扩展等超参数针对电信文档调优，迁移性有限；缺乏多模态检索与更高并行度。

---

## 221. SR-Edit: Region-Aware Image Editing via Self-Refinement

**arXiv ID:** 2609.02504 | [PDF](https://arxiv.org/pdf/2609.02504v1)

**作者:** Andong Wang `[一作]` (Tsinghua University), Jun Zhu `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `da1b1a89-583a-4b57-9c81-478778569bec` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SR-Edit 框架，通过自我迭代精细化编辑区域并使用 Doob's h-transform 进行区域保留，实现无外部掩码的高忠实度图像编辑。

**💡 创新点**

创新点包括：①利用模型自身输出的像素级差分自反馈实现精准区域分离；②在保留过程中采用理论导出的 Doob’s h‑transform，产生与采样动力学一致的掩码残差；③将该机制同时适配扩散和流匹配模型，形成通用的迭代自修正流程。

**🔧 技术方法**

核心技术包括像素差分映射、Otsu 阈值化、形态学后处理、连通分量筛选；Doob’s h‑transform 的插值近似及其在扩散/流匹配中的掩码残差实现；以及多步自我迭代块来持续更新区域和保留导引。

**📊 数据集**

实验使用 ImgEdit‑Bench（约 1,075 个局部编辑案例）和 PIE‑Bench（带人工标注掩码的 578 个案例）进行评估。

**📈 对比分析**

与 InstructPix2Pix、AnyEdit、Qwen‑Image‑Edit 2511、Step1X‑Edit v1p2 等基准模型及 Follow‑Your‑Shape、SpotEdit 进行对比；在 SSIM、PSNR、DISTS、LPIPS、CLIP 等指标上，SR‑Edit 在非编辑区域保留上显著优于对照组，同时保持或提升语义一致性。

**⚠️ 局限性**

局限性包括：在过度保留导致编辑语义被抑制时出现混合伪影；软校正难以完全消除漂移，导致残留干扰；目前仅针对局部编辑场景，难以直接扩展到全图或视角变换等更大范围编辑。

---

## 222. Connectivity of HAPS-Based Solutions for Large-Scale Wireless Networks: A Percolation Theory Analysis

**arXiv ID:** 2609.02536 | [PDF](https://arxiv.org/pdf/2609.02536v1)

**作者:** Hao Lin `[一作]` (King Abdullah University of Science and Technology), Mohamed-Slim Alouini `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了高空平台站（HAPS）在大规模无线网络中的连通性，提出三种覆盖方案（H2D、H2G2D 与混合），并利用渗流理论分析其连续互联网覆盖的临界条件与连接概率。

**💡 创新点**

创新点在于：
- 将渗流理论与随机几何相结合，首次从连通性角度推导HAPS与GW部署密度的临界边界；
- 给出三种方案的下上界解析表达式，并证明临界曲线存在；
- 通过热力图直观展示不同密度组合下的连接概率，并对比三种方案的优劣。

**🔧 技术方法**

使用技术：
- 随机几何中的泊松点过程建模 HAPS 与 GW 的空间分布；
- 渗流理论（格点渗流、连续渗流及随机盘模型）用以评估大规模网络的连通性；
- 计算几何与蒙特卡洛仿真结合，验证理论推导。

**📊 数据集**

数据集：
- 本研究使用的是仿真生成的随机部署数据，区域为 400 km × 400 km，HAPS 高度 20 km、GW 高度 10 m、用户高度 0 m；
- 参数设定为 d_H2D = 30 km，d_H2G = 50 km，d_G2D = 10 km，r_G2G = 15 km。

**📈 对比分析**

比较方法与性能：
- 通过绘制不同 HAPS/GW 密度组合下的连通概率热力图，展示临界点的快速提升；
- 对比三种方案，混合方案在较低密度下即可实现较高连通概率，H2G2D 方案对 GW 密度更敏感；
- 结果表明，当 HAPS 密度超过约 4.5×10⁻¹⁰ HAPS/km² 时，所有方案均可获得非零连通概率。

**⚠️ 局限性**

局限性：
- 模型假设为理想的 PPP 分布，未考虑实际部署中的空间相关性与障碍物阴影；
- 未给出闭式的连通概率表达式，只能通过仿真估计；
- 只考虑了二维平面覆盖，未分析 3D 高度变化对连通性的影响；
- 未涉及实时信道条件、流量变化和能源约束等实际运营因素。

---

## 223. Diffusion-Encoding Gaussian Field for Joint k-q dMRI Reconstruction

**arXiv ID:** 2609.02288 | [PDF](https://arxiv.org/pdf/2609.02288v1)

**作者:** Zhibo Chen `[一作]` (Nanchang University), Qiegen Liu `[通讯]` (Nanchang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `70e40602-aae3-44bd-80ec-4a7f2674330f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种自监督的联合 k–q 加速 dMRI 重建框架——Diffusion-Encoding Gaussian Field，利用共享的 3D 高斯原语构建空间结构，并在每个原语中嵌入连续的张量-残差 q‑响应，实现联合空间与方向信息的重建与合成。

**💡 创新点**

创新点包括①将空间与方向信息统一到同一高斯原语中，②在每个原语中嵌入物理约束的张量锚点和偶数阶球谐残差，以保持正定性并兼顾灵活性；③自监督的逐步优化策略（原始标尺、观测响应校准、原语自适应分配、连续响应投影和联合细化），实现仅凭欠采样 k‑space 即可完成完整 dMRI 重建。

**🔧 技术方法**

技术手段包括：高斯 splatting（3D Gaussian 原语）、扩散张量物理模型、偶数阶球谐正交基的残差正则化、k‑space 数据一致性约束、稀疏性/TV 正则化、原语自适应出生与分裂、连续响应投影与联合细化。

**📊 数据集**

实验使用 Human Connectome Project（HCP）公开的 dMRI 数据，分别对 1000、2000、3000 s/mm² 三个 shell 进行回采，采用单线圈等价的二维 k‑space 变量密度欠采样（R_k=3/4，N_obs=30/15/10）。

**📈 对比分析**

与 ZF+SH、CSTV+SH、JointKQ-CS、qModeL-DAE、3DGS+SH、3DGS+PCCNN 等基线方法相比，该方法在所有加速设置下的缺失方向重建（PSNR/SSIM）以及下游指标（FA、MD、方向角误差）均表现最佳，尤其在高加速（R_k=4、N_obs=10/15）场景中优势显著。

**⚠️ 局限性**

局限性：仅在已处理的单线圈等价 HCP 数据上验证，未涉及原始多线圈原始数据；仅支持单 shell，单张量原语可能不足以捕捉纤维交叉；高斯原语使用对角协方差，缺乏完全的空间适应性；优化过程基于单一扫描，训练时间较长。

---

## 224. Solomonoff Induction and Singular Integrals

**arXiv ID:** 2609.01666 | [PDF](https://arxiv.org/pdf/2609.01666v1)

**作者:** Will Troiani `[一作]` (Resolution), Daniel Murfet `[通讯]` (Resolution)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文证明了可计算贝叶斯模型的贝叶斯证据可以通过单个固定的可测机产生的 Solomonoff 分布的 Riemann 和来近似，从而将 Solomonoff 先验与 Singular Learning Theory（SLT）的学习系数联系起来。

**💡 创新点**

创新点在于：①构造了一个可计算的采样器（BayesSampler）和对应的单一单调图灵机 B_𝔪，使其输出的半测度与任何可计算贝叶斯模型的证据 Z_n 在常数因子内相等；②利用这一构造，将 Solomonoff 先验的代码长度上界与 SLT 的学习系数 λ 关联，首次把模型复杂度的几何度量直接嵌入 Solomonoff 分布中。

**🔧 技术方法**

主要技术包括：对可计算贝叶斯模型的严谨可计算性定义；利用二进制编码构造 B_𝔪；基于 Riemann 和的离散化技术与单调图灵机的构造；以及对学习系数 λ 的几何解释和与先验概率的对数近似。

**📊 数据集**

本文未使用实验数据集，而是进行严格的理论证明；

**📈 对比分析**

由于没有实验评估，本文未与其它方法做直接比较；理论上证明了 -log M(X^n) 与 -log Z_n 的差距仅为 O(1)，并给出了 λlog n - (m-1)loglog n 的上界；

**⚠️ 局限性**

局限性在于：①需要对模型满足 SLT 的严格假设（如可计算性、Lipschitz 条件、稠密性等）；②构造的单调图灵机 B_𝔪 对模型描述的编码长度有影响，导致常数项随模型而变；③仅适用于可计算的连续参数模型，离散参数或非可计算模型的推广尚未给出。

---

## 225. The Exact Online Threshold for the Asymmetric Binary Perceptron

**arXiv ID:** 2609.02124 | [PDF](https://arxiv.org/pdf/2609.02124v1)

**作者:** Sunghyeon Jo `[一作]` (Georgia Institute of Technology), Taekyun Lee `[通讯]` (University of Texas at Austin)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在信息受限（在线）条件下的非对称二进制感知机，给出该模型的存储容量上限与可实现阈值的精确匹配。

**💡 创新点**

创新点在于将在线感知机的可实现阈值完全与一维布朗运动的随机控制问题等价，并通过半线性单调性和 Föllmer 驱动实现所有约束的同步满足；此外在零余量处提供了精确的计算上限和下限，显著提高了已知的可实现密度。

**🔧 技术方法**

主要技术包括：布朗运动极限定理、可预测漂移约束的几何解释、可预测简单控制的逼近、半线性单调性、Föllmer 驱动、条件高斯耦合、动态规划与 Gibbs 变分原理、以及计算机辅助验证。

**📊 数据集**

使用随机高斯矩阵 G∈ℝ^{M×N}（每个条目 i.i.d. 标准正态）作为数据集，研究 M/N→α 的极限。

**📈 对比分析**

与之前的离线算法（如 Li–Schramm–Zhou）和在线多阶段多数算法相比，提出的算法在固定余量 κ=0 时可实现的密度至少为 0.32747（此前最高仅 0.1），即提升超过三倍；在正余量大时阈值与离线容量首阶相同，负余量时阈值与离线容量存在  κ^2  的差距。

**⚠️ 局限性**

局限性包括：1）阈值解析依赖于一维布朗控制问题，实际实现需要复杂的数值优化和计算机辅助；2）对负余量的近似仅给出渐近量级，精确常数尚未确定；3）算法的实现虽在多项式位宽内，但在大规模实例中仍需高精度数值计算；4）当前结果只适用于高斯噪声模型，是否能推广到更一般分布尚未讨论。

---

## 226. Towards Effective Physical Reservoir Computing with a Pneumatic Soft Robot

**arXiv ID:** 2609.02157 | [PDF](https://arxiv.org/pdf/2609.02157v1)

**作者:** Jeevan Hebbal Manjunath `[一作]` (Arizona State University), Wenlong Zhang `[通讯]` (Arizona State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对软体气动臂的传感拓扑、机器人刚度与传感器数量进行系统性实验，评估其对物理储备计算（PRC）中同时间弯曲角估计性能的影响。

**💡 创新点**

提出三条设计准则：①独立密封的气囊拓扑显著提升可观测状态多样性；②提高基准压力导致传感器间冗余，尤其在耦合拓扑下误差显著增加；③在密封拓扑下，仅需2–3个精心放置的传感器即可获得几乎全部性能收益。

**🔧 技术方法**

使用物理储备计算框架，采用五个气囊软体臂的压力测量作为储备状态；通过20帧延迟嵌入（0.2 s）构建特征，采用岭回归（α = 0.01）进行线性读出。

**📊 数据集**

在36个实验条件（2种拓扑 × 3种基准压强 × 2种激励幅值 × 3种波形）下收集的时间序列数据；每个条件下的数据按70/30时间顺序划分为训练/测试集。

**📈 对比分析**

比较方法：用同时间弯曲角估计的归一化均方误差（NMSE）和延迟解码记忆代理（MC）评估性能；结果显示密封拓扑的NMSE平均降低82%，MC提升约4.5倍；在密封拓扑中，2个传感器已达到97.6%最佳性能，3个可覆盖全部收益。

**⚠️ 局限性**

局限性：实验仅在0.1 Hz慢速激励下进行，未考察高频行为；读出采用线性模型，未探索非线性读出的潜在提升；未对长期耐久性、温度漂移或织物磨损对传感器性能的影响进行评估。

---

## 227. On-Policy Distillation Meets Off-Policy GRPO: Training Compact Instruction-Following Rerankers

**arXiv ID:** 2609.01947 | [PDF](https://arxiv.org/pdf/2609.01947v1)

**作者:** Vignesh Prabhakar `[一作]` (SAP Labs), Anil Babu Ankisettipalli `[通讯]` (SAP Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两阶段强化学习管道，用奖励驱动的 on‑policy 蒸馏将大型 4B 指令跟随 reranker 的能力压缩到 1B。

**💡 创新点**

创新点在于：① 将传统离线 KD 换成学生自己采样排名并在其上获得教师奖励；② 通过 Stage 1 的离线 GRPO 与 LLM 判断提升教师质量，再在 Stage 2 用 on‑policy GRPO 训练学生；③ 证明软奖励比硬标签更能提升 OOD 泛化。

**🔧 技术方法**

技术核心：Plackett–Luce 排序采样、GRPO（group‑normalized policy gradient）、LLM‑judge 作为奖励函数、KL 与熵正则化、soft‑teacher‑reward 计算与 nDCG@6 目标。

**📊 数据集**

使用 8 条训练数据集（FollowIR、InstructIR、InfoSearch、InfIR‑MS MARCO、MetaMath、LeetCode、Robust04、WebQA）共 88.8K 条示例；验证集 9.9K 条查询；OOD 评测在 MAIR‑11（869 query）和 MAIR‑Full（9,356 query、126 任务）。

**📈 对比分析**

与离线 listwise KD、RankNet pairwise KD、on‑policy GKD 以及 7B RL‑训练 rerankers 比较。1B 蒸馏模型在验证集上达到 0.7624 nDCG@6，超过所有 1B+模型；在 MAIR‑11 上 0.7670 nDCG@6，超出 7B RL rerankers（Rank‑R1、REARANK）。在 MAIR‑Full 上实现最高 task‑macro 指标 0.6808 nDCG@6。

**⚠️ 局限性**

局限性：① 依赖 LLM‑judge 的奖励，可能携带偏见和校准误差；② Stage 1 仅针对 4B ZeRank‑2 教师，未验证教师结构泛化；③ 训练多 seed 的实验有限，微小差异需谨慎解释；④ 只评估英文文本检索，未覆盖多语言或生成任务。

---

## 228. New binary optimal LCD codes using heuristic embedding

**arXiv ID:** 2609.02096 | [PDF](https://arxiv.org/pdf/2609.02096v1)

**作者:** Haeun Lim `[一作]` (Sogang University), Jon-Lark Kim `[通讯]` (Sogang University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用贪婪算法在最短LCD嵌入空间中搜索，通过行操作和单元翻转构造二进制最优LCD码。

**💡 创新点**

提出了利用可逆矩阵与任意矩阵构成的最短LCD嵌入的贪婪搜索框架，显著提升了在维度6–8范围内发现最优LCD码的效率，并发现了14条此前未公布的最优码。

**🔧 技术方法**

贪婪搜索、最短LCD嵌入理论、可逆矩阵的初等行操作、任意矩阵的单元翻转、基于已知LCD界限的递归上界推导。

**📊 数据集**

以MAGMA BKLC数据库中的1019条基码为起点，构造其最短LCD嵌入后进行搜索。

**📈 对比分析**

将得到的码距与通过递归上界U(n,k)计算的理论最大距进行比较；共找到19条满足上界的码，其中14条为新码，显著提升了已知的最佳距表，实验结果以表格形式给出。

**⚠️ 局限性**

贪婪算法易陷入局部最优，导致在维度9、10的情形下未能达到上界；搜索空间仍然庞大，需开发更高级的全局搜索或改进启发式策略。

---

## 229. GeoStore: Finding Small Storefronts in Large Scenes -- A Fine-Grained POI Localization Benchmark with Global-to-Local Asymmetric Matching

**arXiv ID:** 2609.02012 | [PDF](https://arxiv.org/pdf/2609.02012v1)

**作者:** Lu Han `[一作]` (Amap), Chunlong Lv `[通讯]` (Amap)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GeoStore基准数据集，用于点位查询（POI）定位，并开发了GLAM方法实现异构、细粒度的查询-参考匹配。

**💡 创新点**

创新点在于：①创建了真实的异构、开放集POI定位基准；②设计了全局-局部联合匹配框架GLAM，其中局部路径使用参数无关的区域池化并通过可学习的软top‑k MaxSim进行匹配；③利用已训练的区域tokens实现近乎零成本的互最近邻重排。

**🔧 技术方法**

技术手段包括：DINOv2 ViT‑B/14骨干网络；SALAD式最佳传输聚合器做全局特征；软top‑k MaxSim（带可学习温度）做局部相似度；InfoNCE对联合相似度进行对比学习；互最近邻（mutual‑nearest‑neighbor）做轻量级重排。

**📊 数据集**

使用的数据集为GeoStore：包含1,215个地点、11,133张车载视角参考图、170张测试查询图，体现异构来源、昼夜与天气差异以及开放集识别。

**📈 对比分析**

与传统全局描述符VPR（CosPlace、BoQ、SelaVPR、ImAge、SALAD）以及两阶段方法FoL进行对比；GLAM在重排后达R@1 25.3%、R@5 36.5%、R@10 41.2%、mAP 24.5%，显著优于FoL（R@1 20.0%）且存储与匹配成本降低约5倍与100倍。

**⚠️ 局限性**

局限性：仍需依赖大量参考数据库；对极端光照或遮挡的鲁棒性待进一步提升；目前仅支持二维视觉查询，未覆盖多模态或三维定位；未来需扩展到更大规模和更丰富的局部交互。

---

## 230. SAUF-Net: Structure--Appearance Representation Learning with Uncertainty Feedback for Semi-Supervised Medical Image Segmentation

**arXiv ID:** 2609.02247 | [PDF](https://arxiv.org/pdf/2609.02247v1)

**作者:** Qin Lu `[一作]` (Nanchang Hangkong University), Shaofeng Jiang `[通讯]` (Nanchang Hangkong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SAUF-Net，一种基于结构–外观分解与不确定性反馈的半监督医学图像分割框架

**💡 创新点**

创新点包括：结构–外观分解模块（SADM）与解离引导模块（DGM）实现特征级分离；外观交换一致性（ASC）增强结构鲁棒性；可靠性引导的双头鉴别器实现特征层不确定性反馈；整体提升低标注下的分割性能

**🔧 技术方法**

使用 SegFormer-B4 作为骨干网络，配合 SADM、DGM、ASC、可靠性映射、双头鉴别器以及伪标签一致性等技术

**📊 数据集**

实验数据集为 ISIC-2016（皮肤病变）和 Kvasir-SEG（息肉）

**📈 对比分析**

与多种最新半监督方法（AdaptFRCNet、ARCO 等）在 10%/20% 标注比例下对比，SAUF-Net 在 ISIC-2016 上 Dice 达 91.88%/92.64%，在 Kvasir-SEG 上 Dice 达 90.53%/91.98%，均优于同类方法，并在仅 5% 标注时保持竞争力

**⚠️ 局限性**

局限性：仅在二维单类别分割任务中验证，未评估跨数据集泛化、3D 或多类别任务的性能

---

## 231. OutageDiT: A Generative Foundation Model for Power Outage Forecasting and Scenario Simulation

**arXiv ID:** 2609.01896 | [PDF](https://arxiv.org/pdf/2609.01896v1)

**作者:** Yunqin Zhu `[一作]` (Georgia Institute of Technology), Yao Xie `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了名为 OutageDiT 的生成式基础模型，用于在七天内以每 15 分钟分辨率预测电力停电计数轨迹；

**💡 创新点**

创新点在于将条件编码器与流解码器相结合，利用数字坐标计数表示与条件流匹配技术，实现跨地区预训练与零样本地理迁移，显著提升了对极端事件的建模；

**🔧 技术方法**

采用 Transformer 编码器、旋转位置嵌入（RoPE）、条件流匹配（Conditional Flow Matching）以及 ODE 生成器；并引入数字坐标、去量化噪声以及辅助 log‑计数损失来强化计数的离散性与可解释性；

**📊 数据集**

使用美国各县级停电与气象记录（包含历史停电、跟踪客户数、气象和日历变量），覆盖全国范围（除密歇根外），时间至 2025 年；

**📈 对比分析**

与 DeepAR、TFT、CSDI、TimesFM、Chronos‑2 等监督与基础模型对比，在全国测试集上在 MSE、WQL、Variogram Score、覆盖率等指标均表现更好；在密歇根零样本迁移评估中也超越了 DMDA 与 IISE 挑战基准；

**⚠️ 局限性**

主要局限是对极端停电事件的覆盖率和校准仍不足，导致极端事件的场景生成表现不如常规事件。

---

## 232. Scaling Inference Prefill with High-Radix Photonic Interconnects

**arXiv ID:** 2609.01821 | [PDF](https://arxiv.org/pdf/2609.01821v1)

**作者:** Arulselvan Madhavan `[一作]`, Thomas Graham `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过数值模拟探讨了 3D 集成光学互连在大规模 MoE 语言模型推理预填充阶段的加速效果，提出将光学互连用于 1152 GPU 的规模化上升模块；

**💡 创新点**

创新点在于将光学互连的极高带宽（4× 电气互连）与大尺度上升模块相结合，显著提升了长上下文推理的预填充吞吐和时间；

**🔧 技术方法**

采用了扩展后的 XLA 成本模型、MoE 模型的多级 IR、设备与批量大小遍历，模拟了电气与光学互连在不同硬件平台（B200、B300、Rubin、R4）上的表现；

**📊 数据集**

评估基于 DeepSeek R1（42B 活跃参数）等 MoE 模型，主要针对推理预填充阶段的总标记数 1K、8K、128K 与 1M 的情境；

**📈 对比分析**

通过对比电气互连基线和光学互连配置，结果显示在 1K–8K 低上下文时可获得 2.1–2.9 倍预填充加速；在 128K 上下文时 4.3–5.8 倍；在 1M 上下文时 2.2–4.5 倍（R4 最高 8.5 倍），并且在高批量与大规模设备情况下提升显著；

**⚠️ 局限性**

局限性包括仅为预填充阶段建模、未对光学互连的实际延迟、热、信号完整性及成本进行完整评估，且解码阶段的瓶颈未在同等条件下充分探讨，需进一步系统级共设计。

---

## 233. Context Inference Attacks Without Jailbreaks

**arXiv ID:** 2609.01663 | [PDF](https://arxiv.org/pdf/2609.01663v1)

**作者:** Prince Jha `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Nils Lukas `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 339 | [OpenAlex ID](https://openalex.org/A5086633938)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并评估了一种在Agentic AI系统中利用普通非恶意查询来推断隐式上下文（隐藏记录）的方法，称为上下文推断攻击。

**💡 创新点**

创新点在于：①将上下文泄露视为无指令、无直接提取的概率推断问题；②统一一个基于重放与似然比的攻击框架，能跨越已知、未知以及通过工具检索的三种上下文交付场景；③证明即使采用指令禁止、logit抑制和上下文稀释等防御，攻击仍能成功。

**🔧 技术方法**

技术核心是：预先使用代理模型（灰盒或黑盒）对一组善意查询进行离线优化；在部署时收集响应，使用代理模型计算每个候选上下文的平均负对数似然，进而做似然比判定。

**📊 数据集**

数据集与模型：LLM使用Qwen2.5系列（1.5B–72B）与LLaVA、Qwen2.5VLM等VLM；候选上下文为CelebA图像或CC-News文本；记录池设定为固定数量的公开新闻片段或随机生成的API key。

**📈 对比分析**

评估方法：对三种攻击情景（已知上下文、未知模板、agent检索）分别报告攻击成功率（ASR）或AUROC，并与GPT‑4o‑mini判断器与随机猜测基线对比。结果显示：已知上下文ASR可达100%；未知上下文AUROC可达78.9；agent检索场景AUROC可达81.8；所有实验均优于基线。

**⚠️ 局限性**

局限性包括：①仅在受控实验环境下验证，未考虑自适应检索或动态上下文变化；②黑盒攻击依赖同族代理模型的相似性，若代理与目标差异过大性能下降；③防御仅限于logit抑制、指令禁止和上下文稀释，未探索更强的隐私保障手段。

---

## 234. SSAKG 2.0: An Open-Source Package for Structural Associative Sequence Memory and Context-Based Retrieval

**arXiv ID:** 2609.01849 | [PDF](https://arxiv.org/pdf/2609.01849v1)

**作者:** Przemysław Stokłosa `[一作]` (Institute of Management and Information Technology), Paweł Raif `[通讯]` (Silesian University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了可开源的 SSAKG（Structural Sequential Associative Knowledge Graph）软件包，用于构建和操作稀疏图记忆，以实现从部分上下文中恢复有序序列；

**💡 创新点**

引入了位级（bit-level）序列位置编码与位级过滤算法，显著提升了记忆容量和检索精度；

**🔧 技术方法**

采用 Python 作为高层接口，核心图运算实现为 C 扩展；使用稀疏邻接矩阵、Hadamard 乘积、加权传递锦标赛（transitive tournament）以及位运算（OR、AND、位频率过滤）等技术；

**📊 数据集**

实验使用随机生成的数值序列（长度15、符号数2000）、NLTK 句子序列和 mRNA 序列，但实验结果仅展示数值序列；

**📈 对比分析**

与传统 SSAKG 算法对比：在存储序列数量和上下文长度增加时，位级实现的错误率始终低于0.6%，而传统实现快速升至100%；表明位级方法在高负载和短上下文下的鲁棒性显著优于传统方法；

**⚠️ 局限性**

主要限制包括：位级编码受限于内存单元位宽（目前最长64位），高图密度或特殊干扰时位频率过滤可能失效；未系统测评执行时间、内存占用和与其他序列记忆模型的对比；

---

## 235. Import What You Need: Learning When and How to Augment EHR Graphs with External Knowledge

**arXiv ID:** 2609.01839 | [PDF](https://arxiv.org/pdf/2609.01839v1)

**作者:** Chen Chen `[一作]` (University of Kansas), Zijun Yao `[通讯]` (University of Kansas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出了ReTA框架，通过强化学习在每个就诊节点动态决定是否引入知识图谱中的知识以提升EHR预测性能。

**💡 创新点**

创新点在于将KG导入视为预算约束的增量决策，区分软导入、硬导入和跳过，并利用LLM生成的模板构建有限质量知识池。

**🔧 技术方法**

采用强化学习策略、双通道解码器、语义/结构门控融合以及LLM-知识图谱模板化等技术。

**📊 数据集**

在MIMIC‑III和MIMIC‑IV两个EHR数据集上进行评估，并与多种基线方法比较。

**📈 对比分析**

与最强基线相比，ReTA在诊断预测、住院死亡率和30天再入院等任务上均提升了1.5–4.5个百分点，且在稀疏标签和跨数据集迁移下表现尤为突出。

**⚠️ 局限性**

主要局限包括对外部知识图谱的依赖、LLM生成模板可能存在错误、跳过机制对模型置信度的假设以及仅评估结构化编码，未涉及临床文本。

---

## 236. The Diagnosis a Reporter Leaves Unspoken: Surfacing Frozen Tumor Features for Brain-Tumor MRI Reporting

**arXiv ID:** 2609.02411 | [PDF](https://arxiv.org/pdf/2609.02411v1)

**作者:** Khawaja Murad ul Hassan `[一作]` (National University of Sciences and Technology), Mehran Ebrahimi `[通讯]` (Ontario Tech University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

在脑肿瘤 MRI 报告生成中，提出了通过在每个病灶特征上训练判别性头部（field‑classifier heads）并将其结果直接注入语言模型（Mistral‑7B）的单次草稿‑复核（draft‑then‑review）流程，生成可机器检查的结构化 JSON 报告。

**💡 创新点**

创新点在于：① 引入“head‑surfacing”机制，使模型在冻结的特征中提取诊断信息并强制语言模型接受这些字段，而非让 LM 自行生成；② 采用单次快速推理，显著降低推理时延（≈80 s/病例）；③ 通过语法约束解码和语义一致性判别，实现高达 92.3 % 的结构化有效率，并在临床读者研究中无关键错误。

**🔧 技术方法**

使用技术包括：冻结的 MedNeXt 语义分割器、基于 Q‑Former 的多头查询、判别性字段头、QLoRA 细调的 Mistral‑7B 语言模型、语法约束解码（JSON 语法掩码）以及可选的自适应冷却门（abstain）。

**📊 数据集**

数据集：BraTS‑2020（121 个带结构化报告的病例，121 训练；39 失效测试；50 校准），RadGenome（glioma 60、meningioma 60）和 BraTS‑MET（60，专门用于测试模型的 OOD 诊断），所有数据均公开可用。

**📈 对比分析**

与同一基底的多链链式思维（CoT）模型、M3D‑LaMed、LLaVA‑Med、AutoRG‑Brain、BrainGemma3D 等基线相比，本方法在 8/9 个内容评估指标（RaTEScore、RadGraph‑F1、GREEN）上取得显著提升（均为 Holm‑校正显著），并在临床读者评分中获得最高分（8/9 案例同意签字）且无关键错误。推理时延比 CoT 低 5–6 倍。

**⚠️ 局限性**

局限性包括：① 测试集规模有限（仅 39 病例），性能主要基于 174 个聚合的 hold‑out 病例；② 依赖分割器性能（尤其是 OOD 乳头状转移的 Dice 仅 0.67）；③ 目前是协助性草稿工具而非完全自治；④ 读者研究仅在单一机构、两名神经科医师、9 例样本，需扩大多中心验证。

---

## 237. OBJECTION! Lawyer Agents Mitigate Guilty Bias in Legal Judgment Prediction

**arXiv ID:** 2609.02158 | [PDF](https://arxiv.org/pdf/2609.02158v1)

**作者:** Jaehoon Jeong `[一作]` (Seoul National University), Jay-Yoon Lee `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在推理时引入律师代理的三步推理框架 OBJECTION，用于消除法律判决预测模型中的有罪偏差。

**💡 创新点**

通过在每一步推理中注入对抗性辩护论点，实现无训练的推理时偏差校正，并具备可调节的对抗强度。

**🔧 技术方法**

结合 SOAM 结构化信息抽取、三步刑事责任推理以及对抗性律师代理，使用大语言模型（如 Qwen、Llama、Gemma、GPT‑4）进行推理。

**📊 数据集**

新构建的 Natural Innocent 真实无罪判例集（3.4k 条）以及 CAIL 与 ELAM 合成无罪样本。

**📈 对比分析**

与 LJPIV、Self‑Refine、Debate‑Feedback 等基线对比，OBJECTION 将 False Guilty Rate 从 82.93% 降至 16.69%，Macro‑F1 保持或提升，并在多模型、多域上表现稳健。

**⚠️ 局限性**

仅适用于大陆法系；SOAM 结构化抽取缺乏案件细节；推理成本高于单通道模型；仅处理有罪/无罪判决，未覆盖指控与量刑。

---

## 238. Forbidden Subgraphs of Graphs with Low Bandwidth

**arXiv ID:** 2609.01949 | [PDF](https://arxiv.org/pdf/2609.01949v1)

**作者:** Maria Chudnovsky `[一作]` (Princeton University), Eran Nevo `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文证明了无论给定任意图 G 与整数 k，存在一算法在 2^O(9^k)·n^O(1) 时间内完成两件事：
① 若 G 的带宽至少 k，则输出一个子树 T 使得 T 的带宽至少 k；
② 若 G 的带宽小于 k，则输出一个布局，其带宽为 (10^85·k^28)^{4^k} 的上界。该结果回答了 Chung 与 Seymour 在 1989 年提出的关于子树带宽上界的开放问题。

**💡 创新点**

创新点主要有：
• 提出“带宽的参数化逼近”概念，并给出了首个 FPT 逼近算法；
• 建立了带宽与子树带宽之间的结构性关系，得到一种“禁用子图”表征，类似网格子理论对树宽、路径宽、树深的作用；
• 通过嵌入技术将任意图嵌入树或路径，得到低伸缩、低拥塞、低 S‑收缩的嵌入；
• 设计了一套完整的分解与递归框架，利用路径宽度、局部密度与树打包相结合，控制算法复杂度。

**🔧 技术方法**

核心技术包括：
• 低伸缩低拥塞嵌入（embedding）与布局之间的转换；
• 对带宽图的递归分解，利用路径宽度的层次结构和树打包的图形化；
• 结合 Erdős–Pósa 原理与局部密度的估计，得到分离子与球的上界；
• 通过构造“弱预布局”（weak pre‑layout）来捕捉图的结构，随后将其转化为真正的布局；
• 采用树剪枝与极大子树的构造来维护子树带宽不变；
• 复杂度分析主要依赖于对路径宽度 τ 的指数级递归，并在每一步保持 9^τ 的上界。

**📊 数据集**

该工作为理论算法，没有使用实验数据集。所有结果均为证明性，实验验证并未出现；若需验证，可在小规模稠密或树形图上手动构造测试实例，但本文并未给出具体数据集。

**📈 对比分析**

对比传统的无参数化或单纯参数化方法，本文取得的最小带宽上界为 (10^85·k^28)^{4^k}，虽然数值极大，但在 FPT 逼近框架下是首次给出多项式时间下的上界；
• 与现有的 O(log^3 n·loglog n) 近似算法相比，算法在参数化意义上提供了更强的理论保证；
• 与仅在树上可实现的多项式时间算法相比，本文将可计算性扩展到任意图，尽管时间复杂度为指数级，但在参数 k 受限时可行。

**⚠️ 局限性**

局限性包括：
• 运行时间是 2^O(9^k)·n^O(1)，对于 k 较大时不可接受；
• 输出的布局带宽上界指数级 (10^85·k^28)^{4^k}，实际应用中几乎无法达到；
• 需要图的最大局部密度和路径宽度信息，这在某些图上难以高效预处理；
• 仅给出理论上存在的“子树带宽上界”函数 f(k)，具体形式仍极不紧凑；
• 对于特殊图类（如有特定结构的图），仍无法取得多项式时间的精确带宽计算。

---

## 239. Disease Burden over Skin Tone: Decomposing the Dermatology-AI Generalization Gap

**arXiv ID:** 2609.02111 | [PDF](https://arxiv.org/pdf/2609.02111v1)

**作者:** Nirajan Kunwor `[一作]` (Tribhuvan University), Sunil Kumar Gaire `[通讯]` (North Carolina A&T State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文通过对公开数据和预训练模型进行冻结特征提取与线性探测，量化并分解了皮肤科AI在资源受限环境下的泛化缺口，探究疾病分布偏移与肤色差异的相对贡献；

**💡 创新点**

创新点在于首次用公开数据和无训练量的特征提取方法，将泛化缺口拆解为肤色与疾病分布两条轴；提出基于kNN纯度的无标签表示质量评估，证明缺口源自表示层缺陷；同时证明在疾病分布差异下，使用皮肤科预训练的基础模型并通过少量标记即可实现低成本适配；

**🔧 技术方法**

技术主要包括冻结预训练模型（ResNet‑50、DermLIP、MONET、DINOv3）提取特征；线性探测（L2正则化逻辑回归）评估线性可分性；kNN纯度提升度衡量表示层结构；少量样本（≈10例/类别）线性探测进行低算力适配；

**📊 数据集**

数据集：HAM10000+ISIC 2019用于训练与源域评估；DDI（656张）用于肤色分层、疾病保持不变；SCIN（6517张）用于疾病分布偏移、肤色多样；

**📈 对比分析**

比较方法为相同患者级拆分、相同标准化与相同线性探测器，测量在源域、SCIN与DDI上的平衡准确率；结果显示：疾病分布偏移导致准确率从0.62降至0.21（癌症基线），肤色偏移幅度仅0.10-0.18；DermLIP、MONET、DINOv3的下降幅度较小；kNN纯度表明癌症基线在SCIN上几乎无结构；在少量样本适配下，DermLIP、MONET可恢复到接近全量探测的性能；

**⚠️ 局限性**

局限性包括：DDI与SCIN为美国数据，未完全代表真实资源受限环境；DDI样本量小，肤色效应统计不稳；只评估冻结特征，未涉及端到端或参数高效微调；不同模型架构、预训练任务与规模混杂，难以单独归因；将SCIN长尾疾病映射为粗类别简化了诊断复杂性；未来需在真实RCS数据上验证。

---

## 240. Ranked by the Matcher: A Reproducibility Audit of Knowledge Graph Extraction from Threat Reports

**arXiv ID:** 2609.01671 | [PDF](https://arxiv.org/pdf/2609.01671v1)

**作者:** Safayat Bin Hakim `[一作]` (University of Maryland, Baltimore County), Houbing Herbert Song `[通讯]` (University of Maryland, Baltimore County)

**通讯引用:** 30968 | [OpenAlex ID](https://openalex.org/A5079301418)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究对十二种CTI知识图谱提取系统的匹配规则进行审计，并构建了CTIForge流水线，使用统一的匹配协议重新评分十个系统的输出，评估匹配器对F1分数及系统排名的影响；

**💡 创新点**

创新点在于：①将匹配规则明确定义并可重现，揭示匹配器对评估结果的决定性作用；②构建可切换验证层的可共享提取流水线，隔离后端组件对性能的影响；③通过对外部人类裁定集的机械匹配与LLM裁定器对比，量化机械匹配的可信度；

**🔧 技术方法**

主要技术包括：大语言模型(如GPT‑4o、Claude Haiku等)进行一次性提取，后端的符号验证、规范化、实体对齐与融合，八种匹配协议(精确匹配、软词汇匹配、嵌入阈值、自然语言推理等)的实现，统计错误分类及验证日志；

**📊 数据集**

使用的数据集包括：CTI‑Nexus（149份标注报告）、CTIKG（255句子）、GRID校准集（378条人类裁定）、以及多种公开模型的API与离线部署模型；

**📈 对比分析**

比较方法：在同一批文档和相同提取模型下，对十个系统的输出按八种匹配协议重新计算F1；将CTIForge与CTIKG在统一匹配器下对比；评估验证层的增益/损失。结果显示：匹配器可使单一预测集F1从0.16提升至0.70，系统间排名有11/45对被逆转；机械匹配与人类裁定吻合度低于71%，而LLM裁定达到86%；在不同后端配置下，验证层对精准度的影响呈正负分化；整体F1在0.47–0.55之间，主导瓶颈为关系标签与实体类型识别；

**⚠️ 局限性**

局限性包括：①匹配协议仍以机械方式与LLM裁定存在显著差距，缺乏统一的可靠评估标准；②验证层性能依赖于训练模型的类型约定，无法完全解耦；③实验仅覆盖现有公开数据集，未覆盖更大多样化威胁报告；④在离线与托管配置中后端与解码等因素共变，无法单独评估；⑤错误分类仅记录规则触发，而非真正错误率；

---

## 241. Git4Data: Database-Native Version Control for AI Agents

**arXiv ID:** 2609.02106 | [PDF](https://arxiv.org/pdf/2609.02106v1)

**作者:** Hongshen Gou `[一作]` (MatrixOrigin), Jianguo Wang `[通讯]` (Purdue University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Git4Data——一种在关系数据库中原生实现的版本控制抽象，使得表可以像Git仓库一样进行快照、分支、diff和三路合并，满足LLM代理并行探索数据状态的隔离、可复现和可审计需求。

**💡 创新点**

创新点在于将Git的版本化语义直接映射到SQL层面，利用表的主键实现行级冲突检测和合并，并通过只操作元数据的克隆、增量diff/merge大幅降低存储和计算成本；在MatrixOne上实现的实现证明了该思路可在云原生OLTP数据库上高效实现。

**🔧 技术方法**

使用技术包括：append‑only、不可变对象存储与LSM树、MVCC事务、对象级元数据快照、基于主键的三路合并逻辑、以及对MatrixOne的内部对象目录操作来实现克隆与增量diff/merge。

**📊 数据集**

主要使用TPC‑H 100规模数据（约47M行）进行微基准测试，并在BranchBench基准上评估四个代理工作流（软件开发、失败重现、数据清洗、MCTS），以及额外的1000代理并发工作流。

**📈 对比分析**

与传统SQL实现以及DoltDB（MySQL兼容的Git‑style数据库）比较，Git4Data在单条clone操作上仅需0.2 s、存储极小；diff/merge速度提升数百倍；在BranchBench中，Git4Data的冷/热运行时间比DoltDB快最多18.5×，并在1000代理并发场景下完成400 s而DoltDB两小时内都无法完成。

**⚠️ 局限性**

局限性包括：缺乏对模式演化的支持、仅实现行级冲突解析（对同一行不同列修改仍冲突）、对命名快照的持续存储会导致历史累积、以及在代理数量极大时的共享计算与I/O调度问题，未来需要资源治理、细粒度冲突策略和模式版本化等改进。

---

## 242. Train What You Deploy: Closing the MLP Reachability Gap in Low-Rank Clone Distillation

**arXiv ID:** 2609.02006 | [PDF](https://arxiv.org/pdf/2609.02006v1)

**作者:** Wenhui Chen `[一作]` (Incept Labs), Ritankar Das `[通讯]` (Incept Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在低秩克隆（Low‑Rank Clone，LRC）知识蒸馏中发现并修复了训练可达空间与部署权重空间不一致的“容量缺口”，通过从相同的 LRC warm‑start 开始训练整个部署 MLP 矩阵，恢复了被“绑住”的权重自由度；

**💡 创新点**

创新点在于提出并量化“训练利用率（training‑utilization）”概念，识别 LRC 的子空间限制导致 62.5–81.4% 的部署自由度未被利用，并通过“Train‑what‑you‑deploy”策略（Dense‑LRC 与 CORE‑LRC 两种可合并实现）在保持部署形状与 FLOPs 不变的前提下，实现全矩阵训练，显著提升性能；

**🔧 技术方法**

技术方法包括：低秩权重克隆、基于教师投影的子空间重参数化、完整矩阵训练（Dense‑LRC）以及教师谱子空间实现（CORE‑LRC），并在 LRC warm‑start 基础上进行蒸馏 + 轻量级 SFT；

**📊 数据集**

数据集：使用与教师模型相同的预训练语料（约 10B+ 0.35B tokens），蒸馏阶段约 10B tokens，SFT 阶段约 0.62B tokens；评估采用 LM‑Eval Harness 的 9 项多选/问答任务（含 MathQA）进行 0‑shot 评测；

**📈 对比分析**

与原始 LRC 基线、Meta 官方同谱压缩以及无子空间训练的对照进行对比，得到 Avg9（9‑任务平均）提升 +2.36/+2.71/+10.45（分别对应 Llama3.2‑3B→1.5B、Llama3.1‑8B→2.7B、Qwen2.5‑3B→1.7B），MMLU 也显著提升；在 Qwen 目标上实现 2 倍 token 效率；控制实验（基准、等参数、不同 recipe）验证提升来自可达空间扩展；

**⚠️ 局限性**

局限性包括：未实现完全零遗忘（MMLU 有轻微下降）；实验仅为单种种子，未做多次复现；仅验证 LRC 这一压缩骨干，未探究其他 prune‑distill 方法；评估聚焦 0‑shot 多选/问答，未覆盖生成式或长链推理；

---

## 243. Will there be a 7G?

**arXiv ID:** 2609.01877 | [PDF](https://arxiv.org/pdf/2609.01877v1)

**作者:** Adnan Aijaz `[一作]` `[通讯]` (Toshiba Europe Ltd.), Adnan Aijaz (Toshiba Europe Ltd.)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对7G的必要性进行框架性评估，提出判断是否需要新的世代的六项测试。

**💡 创新点**

构建了“7G准备框架”，并将其应用于七个潜在差异化驱动。

**🔧 技术方法**

主要使用系统分析、标准化方法论与文献综述；无具体技术实现。

**📊 数据集**

无数据集使用，主要基于ITU‑R、3GPP发布的标准与行业报告。

**📈 对比分析**

无实验对比；本文采用定性评估与案例分析。

**⚠️ 局限性**

局限在于缺乏定量验证和实证数据，框架主观性较强。

---

## 244. Improved Automatic Target Recognition in Synthetic Aperture Sonar Imagery Using Large Deep Neural Networks

**arXiv ID:** 2609.01800 | [PDF](https://arxiv.org/pdf/2609.01800v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 245. Video2Reaction: Training Foundation Video Models to Predict Audience Reaction

**arXiv ID:** 2609.01816 | [PDF](https://arxiv.org/pdf/2609.01816v1)

**作者:** Sidong Zhang `[一作]` (UMass Amherst), Madalina Fiterau `[通讯]` (UMass Amherst)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建 Video2Reaction 多模态数据集，通过社交媒体评论获取观众对短电影片段的情绪反应，并用多代理 LLM 进行分布式情绪标注；

**💡 创新点**

创新点在于：①将观众情绪作为分布式标签而非单一类别，反映情绪主观性；②使用两阶段多代理 LLM 注释管线实现可扩展、可持续更新；③通过 Video2Reaction 预训练提升 VLM 在不同情绪数据集上的迁移性能；

**🔧 技术方法**

技术包括：多模态 VLM 微调（LoRA）、两阶段 LLM 注释管线（重写+提取）、多代理投票、Label‑option 与 Word‑option 提示策略、KL 损失学习；

**📊 数据集**

数据集：Video2Reaction（约 10,000 条短片，1M YouTube 评论，21 类情绪），以及基准 VCE 数据集；

**📈 对比分析**

评估：在 Video2Reaction 上通过 MRR、TPE、F1_k 评估，LoRA 微调后的 LLaVA‑Next‑Video‑7B 和 Qwen2.5‑VL‑7B‑Instruct 超越传统 LDL 基线；在 VCE 上通过 Top‑3 Accuracy 评估，预训练 VLM 在仅 1% 样本时即可达到或超过全量 50k 训练的 SOTA；

**⚠️ 局限性**

局限性：①注释质量仍受 LLM 主观偏差影响；②仅在 7B 规模 VLM 上验证，未探究更大/更小模型的效果；③缺乏对不同文化/语言背景下情绪表达的细致分析；

---

## 246. Agentic UE-CoMIMO for 6G Terminals: From Virtual Antenna Augmentation to AI-Native Virtualization

**arXiv ID:** 2609.02290 | [PDF](https://arxiv.org/pdf/2609.02290v1)

**作者:** Chao-Kai Wen `[一作]` (National Sun Yat-sen University), Geoffrey Ye Li `[通讯]` (Imperial College London)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了基于多设备协作的UE‑CoMIMO在6G终端中的智能控制与语义信息交换；

**💡 创新点**

提出Agentic UE‑CoMIMO框架，将意图解释、任务分解、机制选择与反馈重规划结合，支持端侧协同网络、感知与计算；

**🔧 技术方法**

采用分层微代理、手机/CPE枢纽代理、边缘网络代理的多代理体系，结合语义Token交换、预测与意图推理技术；

**📊 数据集**

使用系统级仿真数据，涵盖创作者直播与盲区感知两大场景；

**📈 对比分析**

通过与固定、反应式、无预测与无代理等基线对比，证明Agentic策略在节能、热管理、服务连续性方面优于基线，显著提升QoE与任务可靠性；

**⚠️ 局限性**

主要局限在标准化接口、跨信任域协作的安全与隐私、对真实硬件实验的依赖，以及在极端干扰场景下仍需进一步改进。

---

## 247. ExecRetrieval: Measuring the Functional-Correctness Gap in Code-Embedding Retrieval

**arXiv ID:** 2609.01865 | [PDF](https://arxiv.org/pdf/2609.01865v1)

**作者:** Aaryan Kapoor `[一作]` (Kennesaw State University), Md Abdullah Al Hafiz Khan `[通讯]` (Kennesaw State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个功能导向的代码检索基准，包含938个 Python 函数任务，每个任务配有一条通过执行测试用例验证的正确实现和 3–4 条单编辑的错误变体。

**💡 创新点**

创新点在于把可执行验证的单编辑近克隆错误直接植入检索池，从而使检索模型的排名真正衡量功能正确性而非仅仅是语义或身份相似度，填补了现有基准的空白。

**🔧 技术方法**

使用 GPT‑5.4（高推理）生成代码及机械扰动、5 步验证管道（模式、AST、执行、错误检测、完整性）、Python 子进程执行沙箱、以及多种密集嵌入检索模型（Gemini、Qwen3、Mistral、OpenAI 等）与 BM25 对比。

**📊 数据集**

采用了自研基准数据集：约 938 条任务，覆盖 10 个算法域，配备 7–10 条测试用例、1 条通过验证的实现和最多 4 条单编辑错误实现。

**📈 对比分析**

通过 top‑k、rank‑1、canonical‑ID nDCG、配对 McNemar 检验和 bootstrap 区间进行评估。结果显示 top‑k 取得 ≥0.98 的成功率，但 rank‑1 仅最高 33.1%（Gemini Embedding 2），且错误大多为近克隆错误，表明嵌入模型在功能区分上仍有限。

**⚠️ 局限性**

局限性包括：仅针对 Python；错误仅为机械单编辑；检索池为闭合域（无外部代码）；未覆盖跨文件或跨语言情形；仅评估嵌入检索，未考虑 reranker 或 LLM 重排序；执行沙箱未做硬化；模型可能随提供商更新变化；数据规模相对较小。

---

## 248. Uncertainty-Guided Adverse Weather Restoration via Gated Transformer Network

**arXiv ID:** 2609.02434 | [PDF](https://arxiv.org/pdf/2609.02434v1)

**作者:** Zheke Jin `[一作]` (Technical University Of Munich), Hu Cao `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种针对所有恶劣天气图像恢复的统一网络UAR-Net，能够在单个模型中同时处理雨、雪、雾等多种降解。

**💡 创新点**

创新点包括：①选择性门控注意力（GDTB）结合双尺度局部前馈，能够自适应聚焦不同天气的全局与局部信息；②平衡多尺度跳连（BMSC）采用预测-校正融合，抑制浅层噪声并实现跨尺度平滑传递；③不确定性感知细化头（URH）联合亮度感知能量损失（BAE‑Loss），既提升细节重建又估计像素级恢复置信度。

**🔧 技术方法**

技术手段包括：线性注意力与正弦重加权、门控机制、双尺度卷积前馈、预测-校正多步融合、热扩散注意力（vHeat）、蒙特卡洛能量评分、亮度一致性约束和相关性损失。

**📊 数据集**

使用了四个公开数据集：Snow100K（含S/L/Real子集）、Raindrop、Outdoor‑Rain，以及合并的多天气训练集。

**📈 对比分析**

与多种统一与任务专用模型（如Histoformer、MODEM、HOGformer等）对比，UAR-Net在PSNR/SSIM、LPIPS、Q‑Align和MUSIQ等指标上均显著领先，尤其在Snow100K‑S、Outdoor‑Rain和Raindrop任务中获得最高分。

**⚠️ 局限性**

局限性主要在模型规模与推理速度（≈35 GMac），以及在极端高分辨率或实时应用场景中可能需进一步优化加速。

---

## 249. InstEditSeg: Instruction-Driven Image Editing for Polyp and Skin Lesion Segmentation

**arXiv ID:** 2609.02004 | [PDF](https://arxiv.org/pdf/2609.02004v1)

**作者:** Ziquan Liu `[一作]` (Southwest University of Science and Technology), Xuyang Shi `[通讯]` (Southwest University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

将医学影像分割任务改写为基于文本指令的图像编辑任务，模型直接在原图上渲染颜色编码的分割覆盖，而不是输出独立二值掩码。

**💡 创新点**

创新点在于：①使用颜色编码覆盖将生成结果与潜在扩散模型的自然图像先验对齐，显著减小医学与自然图像的域差；②引入冻结的 DINOv3 多尺度特征金字塔，通过通道拼接+零初始化卷积注入层次化判别先验；③提出双分支无分类器引导策略，仅两次前向传播即可完成文本与图像条件的结合，降低推理成本。

**🔧 技术方法**

核心技术包括：潜在扩散模型（Stable Diffusion）作为生成骨干，CLIP 文本编码器，冻结的 DINOv3 视觉编码器与 DINO Feature Guidance（DFG）模块，多尺度特征融合，双分支 CFG 采样，联合噪声预测与辅助分割损失。

**📊 数据集**

使用的医学数据集有：多种大肠息肉分割数据集（Kvasir-SEG、CVC-ClinicDB、CVC-ColonDB、ETIS-LaribPolypDB），未见数据集 PolypGen；皮肤病变分割数据集 ISIC2016 训练集与 ISIC2017 作为未见测试集。

**📈 对比分析**

与传统 U‑Net、Polyp‑PVT、EMCAD、SAM 系列、MedSAM 及多种扩散分割基线（MedSegDiff、TSLDseg 等）进行对比。实验显示：在所有在域测试集上表现与最强判别基线相当，在未见域 PolypGen 与 ISIC2017 上取得最佳 Dice 分别为 83.92% 与 83.14%，显著优于所有对照方法；在多病灶案例中相对 EMCAD 提升 12.75 Dice 分。

**⚠️ 局限性**

局限性包括：①在域内最强判别模型仍略占优势；②推理时仍比单通道判别模型慢；③目前仅支持类别级指令，无法处理空间/属性约束；④对颜色指令敏感，未见颜色时可能产生颜色映射错误。

---

## 250. Exact Limits of Random Projections for Preserving Geometry: Distance Recovery, Nearest-Neighbor Rankings, and Covariance Shape in Gaussian Models

**arXiv ID:** 2609.02155 | [PDF](https://arxiv.org/pdf/2609.02155v1)

**作者:** Piyush Sao `[一作]` `[通讯]` (Oak Ridge National Laboratory), Piyush Sao (Oak Ridge National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对高维空间中的 Johnson–Lindenstrauss 随机投影进行理论分析，揭示即使满足距离保真度，投影后的几何信息仍可能被严重削弱，并提出从线性草图中恢复距离特征的框架。

**💡 创新点**

创新点在于：① 将距离保真度与线性投影后的特征恢复关联起来；② 推导出对等距高斯数据闭式奇异值分布 ℓ_k ≈ (m/d)^{k/2}；③ 用该奇异值量化草图对几何信息（如方差、Kendall 相关、最近邻一致性）的保留程度，证明 JL 约束并不能保证几何可用性。

**🔧 技术方法**

采用了线性算子奇异值分解、条件期望最优解、Haar 平均、概率极限与正态分布分析等数学工具，对等距高斯样本的线性草图特征恢复问题进行解析。

**📊 数据集**

使用的是均值为 0、协方差矩阵为 σ^2 I_d 的等距高斯合成数据；未使用公开数据集。

**📈 对比分析**

通过计算期望 Kendall 相关系数 2/π √(m/d)(1+o(1))、最近邻一致性趋于 1/q 等指标，展示即使投影满足 JL 约束，几何相关性也可降至零；与传统仅测距的评估方法相比，本文提供了更细粒度的几何损失量化。

**⚠️ 局限性**

局限性包括：仅在等距高斯数据和线性投影下进行理论推导；未考虑非高斯分布、非线性投影或真实数据集的实证验证；只关注单一距离特征，未扩展到更复杂结构（如子空间或多维度特征）的恢复。

---

## 251. DiffIE: Diffusion-based Open Information Extraction

**arXiv ID:** 2609.02315 | [PDF](https://arxiv.org/pdf/2609.02315v1)

**作者:** Konstantin Fedorov `[一作]` (Matrosov Institute for System Dynamics and Control Theory), Valentin Malykh `[通讯]` (MWS AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DiffIE，一种基于条件离散扩散的非自回归 OpenIE 模型，利用多条逆扩散轨迹生成多样化的候选标记序列，并通过聚类与排序得到多条事实三元组。

**💡 创新点**

创新点在于：① 将逆扩散的随机性直接用于生成多条候选三元组，解耦提取预算与训练；② 采用宽容匹配聚类聚合候选集，显著提升召回；③ 证明统一噪声扩散优于吸收状态扩散，适用于四标签小词表。

**🔧 技术方法**

技术手段包括：条件离散扩散（D3PM uniform kernel）+预训练 Transformer 编码器 + 轻量级自注意力去噪器；多条轨迹采样 + 长度匹配聚类 + 排名；以及对比实验用的 MC‑dropout 标记器控制。

**📊 数据集**

使用数据：CycleOIE 采样的 LSOIE‑examples（包含 50–4,901 句子及扩充版本），并在 CaRB、CaRB(1‑1)、BenchIE、WiRe57 四个公开基准上进行评估。

**📈 对比分析**

与现有神经和基于规则的系统（OpenIE6、IMoJIE、DetIE、ClausIE 等）在四大基准上对比；在 CaRB(1‑1) 的 F1/AUC 和 BenchIE 的 F1 取得了新的最高分；在 CaRB、WiRe57 仍保持竞争力；通过调节采样数 n，可在计算成本与提取质量之间进行折衷。

**⚠️ 局限性**

局限性包括：使用最长连续跨度构造三元组，导致对不连续或重叠角色的丢失；仅在四标签小词表上实验，未验证更大词表或多语言场景；训练时 40% 的实例因 Span‑alignment 过滤被丢弃，可能影响模型泛化；未对离群或非提取式文本进行评估。

---

## 252. Quantum Workload Privacy Beyond Data Confidentiality

**arXiv ID:** 2609.02323 | [PDF](https://arxiv.org/pdf/2609.02323v1)

**作者:** Shaunak Suresh Pawar `[一作]` (University College Cork), Krishnendu Guha `[通讯]` (University College Cork)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `9cc9baba-5356-466d-81ff-d80028d90279` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了量子云平台中编译器对工作负载意图的泄露，提出SCI-IND概念，并在IBM Heron 156Q上验证能完全区分不同边界条件和几何结构的工作负载。

**💡 创新点**

创新点包括：①提出科学意图不可区分性SCI-IND安全模型；②证明在路由最优编译下被动SCI-IND安全不可实现；③提出三种物理到编译的耦合模式，展示路由开销可恢复网格分辨率与分子几何。

**🔧 技术方法**

使用量子编译器（Qiskit SABRE）、随机种子与不同优化级别，信息熵与互信息评估，机器学习分类（随机森林、逻辑回归），以及实测在IBM Heavy‑Hex 156Q的硬件上进行实验。

**📊 数据集**

实验数据集包括：PDE离散化任务（周期/Dirichlet边界、不同网格大小）、VQE 4‑qubit分子（线性H₂与弯曲H₂O）以及对应的编译后特征，共计数百对匹配实验。

**📈 对比分析**

评估方法采用互信息、Cohen’s d_z、Wilcoxon检验与宏F1分数；实验结果显示所有分类器在不同维度上宏F1≥0.93，VQE任务达到1.00；gate‑padding防御无效且导致相干性显著下降。

**⚠️ 局限性**

局限性：仅针对固定拓扑的超导量子芯片，未考虑可重构或全互连架构；防御方案缺乏低开销的常数时间编译API；论文未给出形式化的执行轨迹隐私保证。

---

## 253. CC-4DGS: Computational Deformation and Point-Cloud Compression for Storage-Efficient Dynamic Gaussian Splatting

**arXiv ID:** 2609.02184 | [PDF](https://arxiv.org/pdf/2609.02184v1)

**作者:** Kyungdae Park `[一作]` (Hanyang University), Chae Eun Rhee `[通讯]` (Hanyang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了CC-4DGS框架，通过计算式变形场和高效点云压缩，实现了动态高质量Gaussian splatting的极致存储压缩。

**💡 创新点**

创新点在于将传统多分辨率哈希表替换为稠密哈希编码与轻量化神经解码器（计算式变形场），以及对Canonical点云属性采用条件自编码器、选择性量化与残差码本的多级压缩，显著降低存储占用。

**🔧 技术方法**

主要技术包括稠密多哈希（Dense Hash Encoding）+神经解码、FiLM融合、频率位置编码、条件自编码器、量化、向量量化残差码本以及实时渲染的Gaussian splatting。

**📊 数据集**

使用公开的N3DV和Technicolor Light Field两大动态场景数据集进行训练与评估。

**📈 对比分析**

与4DGS、Grid4D、Swift4D等基准方法对比，CC-4DGS在保持≈32 dB PSNR、≈0.94 SSIM的同时，将模型大小压缩到20–30 MB，渲染速度保持实时（≈110–140 FPS）并在存储-质量-速度三者上实现更优权衡。

**⚠️ 局限性**

局限性包括：训练时计算开销相对较大；随着序列长度拉长，质量衰减更明显；当前仍依赖全局Canonical点集和统一变形场，需进一步探索分段/多层时间编码与实时流式处理。

---

## 254. Design and Validation of a Lightweight, Low-Profile Powered Knee Prosthesis with Quasi-Direct Drive Actuation

**arXiv ID:** 2609.02003 | [PDF](https://arxiv.org/pdf/2609.02003v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 255. From Silicon to Boot Code: Extending Automated Program Repair to Firmware-Layer Security Workarounds

**arXiv ID:** 2609.01769 | [PDF](https://arxiv.org/pdf/2609.01769v1)

**作者:** Maisha Mastora `[一作]` (University of New Hampshire), Dean Sullivan `[通讯]` (University of New Hampshire)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将自动程序修复（APR）方法从 RTL 设计迁移至 UEFI 固件层，利用提交历史挖掘真实修复模板，构建字典驱动的本地化‑合成‑验证流程，对多语言（C 与 x86 汇编）固件漏洞实现自动定位、合成修复并验证；

**💡 创新点**

① 通过提交历史聚类自动生成修复字典，避免手工定义漏洞模式；② 在固件层实现多语言、结构化的检测与合成；③ 通过根因分解量化误报，证明 100% recall 的同时实现可观的精度提升；

**🔧 技术方法**

提交历史相似性聚类、基于签名的扫描器、前向污点/数据流过滤、结构化校验或编译验证、自动化补丁传播；

**📊 数据集**

EDK II（TianoCore）UEFI 固件仓库，36,153 次提交，涵盖 13 个独立修复点与 31 次回移；

**📈 对比分析**

与简单语法基线（任何语句、任意比较）对比，Spectre v1 规则在未见文件上 100% recall、9.7% precision；各族 100% recall，精度从 15.5% 至 100%；回移成本平均 2.38 次/点；扩展新条目仅数分钟；

**⚠️ 局限性**

验证仅基于结构/编译，未验证功能性安全性；误报主要来自 intra‑procedural，缺乏完整别名分析；基线仅语法级，未与通用静态分析或 LLM 对比；样本规模有限，仅 EDK II，未覆盖功能性错误。

---

## 256. T2LSC-Bench: Benchmarking Localized Semantic Control in Text-to-Image Generation

**arXiv ID:** 2609.02255 | [PDF](https://arxiv.org/pdf/2609.02255v1)

**作者:** Yan Wang `[一作]`, Siwei Ma `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

展示了IEEEtran.cls模板的使用方法

**💡 创新点**

无具体创新点

**🔧 技术方法**

无技术

**📊 数据集**

无数据集

**📈 对比分析**

无比较方法，性能不适用

**⚠️ 局限性**

缺乏实际研究内容

---

## 257. AI agents reshape consensus formation in human groups

**arXiv ID:** 2609.02122 | [PDF](https://arxiv.org/pdf/2609.02122v1)

**作者:** Lin Chen `[一作]` (Northeastern University), Yong Li `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在实验中，将大型语言模型（LLM）代理与人类参与者混合成群体，使用无预设词汇的协作描述游戏，观察不同代理比例下群体共识的形成与内容变化。

**💡 创新点**

发现代理比例对共识形成呈非单调三段式：低比例促进人类共识，中间比例阻碍共识，高比例重新强化共识但内容被代理主导；提出“概念-词汇比率（CLR）”衡量代理对概念层面的领导力，并揭示采用-保持行为平衡对代理影响的因果机制。

**🔧 技术方法**

采用句子嵌入与余弦相似度评估共识强度，利用场景图与SPICE指标聚类语义表达以计算概念贡献；使用对齐提示（prompt engineering）调节代理的“高采纳”与“高保持”行为；通过OLS回归分析人类对代理身份与相似度的认知影响。

**📊 数据集**

实验数据来自24人混合组（5个代理比例条件）和LLM代理（基于OpenAI GPT-3.5/4或类似模型），使用抽象形状（Tangram）图像作为参考物。

**📈 对比分析**

与纯人类（0%）对照比较，低比例（12.5%）共识强度提升8%，中间比例（33.3%、50%）下降23%/14%，高比例（75%）恢复至0.725；词汇贡献随比例递增，概念贡献在高比例跃升至100%；内容维度显示高比例共识更抽象、信息稀疏。

**⚠️ 局限性**

局限性包括：使用抽象无情境的参考任务限制了对自然社会互动的普适性；随机配对网络忽视真实社群结构；仅测试单一LLM家族和能力层级；样本规模相对有限，且高比例条件下人类样本不足。

---

## 258. Atlas: Algorithm-Hardware Co-Design for On-Device City-Scale 3D Gaussian Splatting in VR

**arXiv ID:** 2609.02352 | [PDF](https://arxiv.org/pdf/2609.02352v1)

**作者:** He Zhu `[一作]`, Yu Feng `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种完全在移动VR设备上实现城市规模3D Gaussian Splatting（3DGS）渲染的框架，支持无网络连接的实时渲染。

**💡 创新点**

核心创新点包括：① 层次化内存卸载机制，仅将当前视角所需的高层级块元数据留在GPU内存；② 时序感知LoD搜索，利用帧间相似性减少树遍历；③ 立体光栅化（stereo rasterization）通过几何三角化共享左眼/右眼计算，实现位精确的视差映射；④ 与现有3DGS加速器的轻量级硬件集成，几乎不增加面积。

**🔧 技术方法**

采用块划分与轻量级元数据、GPU并行时序感知树遍历、异步高斯属性加载与使用窗口回收、线性/立体缓冲、基于GSCore的RTL实现以及CUDA/GPU baseline对比实验。

**📊 数据集**

实验使用大规模数据集 Urban、Mega、HierGS 以及小规模数据集 T&T、DB、M360；对比基准包括 HierGS、CityGS、OctreeGS。

**📈 对比分析**

在移动Ampere GPU、GSCore 与 GBU 等硬件上进行对比，结果显示：整体速度提升18.5×（相较GPU基线）和3.9×（相较GSCore），能耗降低13.1×；立体光栅化实现1.4-1.9×加速；GPU内存占用下降7×；在默认配置下可实现70.9FPS，扩展RUs后可达90FPS。

**⚠️ 局限性**

限制与挑战：仍需至少12GB显存以覆盖所有大规模场景；对磁盘/CPU存储的依赖；时序感知LoD搜索对帧间间隔w的敏感性，过大可能产生视觉不连续；立体光栅化在高分辨率/高帧率下对硬件资源需求增加；框架目前主要针对城市规模3DGS，尚未验证在其他神经渲染方法上的通用性。

---

## 259. The Vocabulary Gap Is an Equity Gap: Register Mismatch in Retrieval Systems for Public-Benefits Access

**arXiv ID:** 2609.01645 | [PDF](https://arxiv.org/pdf/2609.01645v1)

**作者:** Krish Sapru `[一作]` `[通讯]` (Dartmouth College), Krish Sapru (Dartmouth College)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个配对注册评估协议和基准，用以衡量检索系统在正式机构语言与普通用户语言下的公平性差距

**💡 创新点**

首次量化福利检索中的注册不匹配导致的“公平差距”，并提出可审计的词典桥接方法作为低成本、可解释的补救手段

**🔧 技术方法**

使用传统稀疏检索方法（BM25、TF‑IDF）和基于词共现的图检索，以及手工编制的 plain‑to‑formal 词典桥接做查询扩展

**📊 数据集**

采集了51条联邦福利资格规则的官方文本，并编写了25对正式/普通语言的查询，形成一个 50 查询 × 25 信息需求的基准

**📈 对比分析**

对比评估显示，正式注册下 Recall@5 接近 100%，普通注册下仅 44%；加入词典桥接后 Recall@5 提升至 80%，显著缩小 56 点的公平差距

**⚠️ 局限性**

局限性包括基准规模有限、查询为人工生成、未评估密集向量检索、词典桥接需人工维护、以及对不同社区语境的适用性尚未充分验证

---

## 260. Spatially Aware World Action Model via Geometric Latent Diffusion

**arXiv ID:** 2609.02531 | [PDF](https://arxiv.org/pdf/2609.02531v1)

**作者:** Javier Alejandro Lopetegui Gonzalez `[一作]` (Inria), Cordelia Schmid `[通讯]` (Inria)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在机器人控制中，提出了一种将3D深度信息注入视频扩散模型的空间感知世界动作模型（SA-WAM），实现了RGB和深度的联合预测与动作生成；

**💡 创新点**

创新点在于：①无需额外3D编码器，通过对深度进行log‑scale归一化，使其能直接映射到冻结的VAE标记器；②将深度视为与RGB同等的latent帧，保留了预训练视频模型的动态先验；③在单一扩散骨干中实现动作与未来观测的联合去噪；

**🔧 技术方法**

使用预训练的Cosmos‑Predict2 Diffusion Transformer与Wan2.1 VAE tokenizer，采用EDM训练目标，融合RGB、深度、关节状态和动作信息；

**📊 数据集**

在模拟数据集RoboCasa、LIBERO、LIBERO‑Plus上进行训练与评估，并在真实UR5机械臂的RGB‑D数据集上验证；

**📈 对比分析**

与Cosmos‑Policy、VideoPolicy、FLARE等基线相比，SA‑WAM在RoboCasa上仅用50个示范即可达到76.6%成功率（比Cosmos‑Policy高9.5个百分点），在LIBERO‑Plus零样本下获得86.6%的加权平均成功率（优于最佳VLA 84.6%），在UR5真实环境中完成率90%（高于Cosmos‑Policy 75%）；

**⚠️ 局限性**

局限性包括：未来状态和动作预测仍可能出现不一致；缺乏几何一致性的训练和验证机制；推理效率仍高，需进一步优化。

---

## 261. TalkFa: A Unified Benchmark for Farsi Dialogue Generation and Understanding

**arXiv ID:** 2609.01810 | [PDF](https://arxiv.org/pdf/2609.01810v1)

**作者:** Neda Jamshidi `[一作]` (University of Siena), Marco Gori `[通讯]` (University of Siena)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个统一的波斯语对话基准TalkFa，涵盖知识驱动生成、日常对话和戏剧对话的生成与理解任务。

**💡 创新点**

创新点在于采用LLM辅助生成并通过多阶段母语审阅实现高质量数据，同时将生成、情绪、情感等多任务整合到一个统一框架。

**🔧 技术方法**

采用LoRA微调LLaMA、Mistral等指令型大模型以及FaBERT、E5等多语言编码器进行实验。

**📊 数据集**

使用Wiki‑FaDial、DailyDialog‑FA、PlayDial‑FA三大子数据集。

**📈 对比分析**

对比实验表明LoRA显著提升生成质量，Mistral‑24B+LoRA在情感和情绪分类上取得最佳0.62/0.38 macro‑F1，FaBERT在对话行为分类上达到0.75 macro‑F1；但自动指标与人工评估存在显著差距。

**⚠️ 局限性**

主要限制包括潜在的LLM迁移污染、固定的6轮对话结构、跨语言标签迁移误差、戏剧对话重构导致的风格失真、方言与非正式语域覆盖不足，以及评测方法的主观性和基准规模有限。

---

## 262. IDEEA: training-free Input-Dependent stEEring via Activation cluster matching

**arXiv ID:** 2609.02089 | [PDF](https://arxiv.org/pdf/2609.02089v1)

**作者:** Zheng Wang `[一作]` (University of British Columbia), Yan Leng `[通讯]` (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练‑free 的输入相关 Steering 框架 IDEEA，通过聚类并对抗负正样本的最佳匹配来为每个输入动态选取激活方向；

**💡 创新点**

创新点在于将激活空间分成多模态簇并求解最优匹配，从而实现输入依赖的方向选择，避免了传统静态方向导致的拒绝陷阱；

**🔧 技术方法**

采用 K‑means 聚类、二次分配（QAP）求解最优匹配、最近簇/最小正交两种方向选择策略，并在残差流的多头注意力层注入线性偏置；

**📊 数据集**

使用 TruthfulQA、Dictator Game、TwinViews、TET 等公开对照数据集以及合成的对比激活数据；

**📈 对比分析**

与 ITI、CAA、SAE、SEA 等现有训练‑free baseline 对比，IDEA 在 TruthfulQA 的 truth×info 率平均提升约34.2%，在政治极化、毒性缓解及社交行为任务也均超越基线；

**⚠️ 局限性**

局限在于跨层激活漂移可能削弱早期层校准的有效性，并且激活注入本身可能被滥用于不安全或有偏见的行为，需配合安全审计与概念约束。

---

## 263. VakyArth: Evaluating Pragmatic Competence in LLMs across Indic Languages

**arXiv ID:** 2609.01788 | [PDF](https://arxiv.org/pdf/2609.01788v1)

**作者:** Usneek Singh `[一作]` (Georgia Institute of Technology), Junyi Jessy L `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

创建了 VakyArth 这一首个面向印度语的语用诊断基准，覆盖印地语、旁遮普语、泰米尔语、马拉雅拉姆语的五种语用现象，并在 MCQ、NLI 与翻译三种任务中评估多种大型语言模型。

**💡 创新点**

其创新点在于首次将多语种、多任务与文化真实度统一为一套评测框架，揭示模型对语用推理的系统性缺陷，并指出自动评估指标在语用忠实度上的局限。

**🔧 技术方法**

采用多样本提示（few-shot）与标准化评估流程，利用 COMET 及其变体进行自动翻译评估，并结合人工评估，进一步对模型进行脚本对比实验。

**📊 数据集**

数据集为 VakyArth，包含 578 条题目（印地语 185，旁遮普语 125，泰米尔语 134，马拉雅拉姆语 134），涵盖多任务题型与代码混合文本，翻译子集共 424 条人工英语翻译。

**📈 对比分析**

与五个指令调优的 LLM（8B‑111B）进行比较，发现 Gemma‑4‑31B 在 MCQ、NLI 与 COMET 上表现最佳；MCQ 分数普遍高于 NLI，印度语与旁遮普语在翻译上优于泰米尔/马拉雅拉姆；所有模型均倾向于字面解释，表现出显著的系统性偏差。

**⚠️ 局限性**

局限性包括仅评估单轮短句/段落，未覆盖数百种其他印度语言；脚本实验仅限 MCQ，未全面检验所有任务与模型；缺乏多轮语用评估与更广泛语言覆盖。

---

## 264. Reinforcement Learning and Rule-Based Peer-to-Peer Pricing in Residential PV-BES Communities

**arXiv ID:** 2609.01680 | [PDF](https://arxiv.org/pdf/2609.01680v1)

**作者:** Pablo Benalcazar `[一作]` (Polish Academy of Sciences), Jacek Kamiński `[通讯]` (Polish Academy of Sciences)

**通讯引用:** 3652 | [OpenAlex ID](https://openalex.org/A5073856274)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对住宅光伏社区中 P2P 电力交易的规则基定价机制（账单共享、MMR、SDR）与基于强化学习（RL）的定价机制（多重参数、SDR 可学习）进行经济效果比较，并在加入储能的情境下评估两者表现。

**💡 创新点**

首次将可学习 SDR 定价与传统规则基定价在同一实验框架下对比，揭示 RL‑SDR‑L 能在保持相同交易量的前提下通过更有效的经济结算（降低内部价差）显著提升社区储蓄；同时验证储能对 RL 定价的积极影响。

**🔧 技术方法**

使用深度 Q‑网络（DQN）实现 RL 定价策略；构建三种规则基机制；通过社区层指标（平均交易价、用户成本、生产者收入、社区储蓄、供需比）和运营指标（交易量、自给率）对比；采用经验回放、目标网络、ε‑贪婪探索等标准 RL 技术。

**📊 数据集**

合成 20户异构社区的 1 年小时级光伏与负荷数据：负荷通过 RAMP 生成；光伏通过 Renewables.ninja 辐照/温度序列转换并加温度衰减；电网进出价为波兰 G12w 时段电价与市场价格。储能容量与充放电曲线通过遗传算法预先优化并固定输入。

**📈 对比分析**

通过对比社区层和运营层的指标评估两类机制。结果显示：在仅光伏场景下，规则基机制（账单共享、MMR、SDR）实现最高社区储蓄（829.98 €），而 RL‑SDR‑L 次之（734.23 €）。加入储能后，RL‑SDR‑L 的社区储蓄提升至 978.52 €，显著优于 RL‑SDR‑F 与 RL‑M；但规则基机制在储能场景下未被测试。RL‑SDR‑L 在相同交易量下通过更低的内部价差实现更高储蓄。

**⚠️ 局限性**

局限性包括：规则基机制仅在光伏场景下评估，未在储能条件下比较；RL 训练仅单次实验，结果可能受随机种子影响；内部价差的预算平衡限制导致 RL 无法突破规则基的上限；未考虑网络约束、传输损耗或跨社区交易的影响；所用的合成数据和预先优化的储能策略可能与真实系统差异较大。

---

## 265. Adaptive Test-Time Inference for Text2Cypher with Trace Budgeting and Selective Refinement

**arXiv ID:** 2609.02324 | [PDF](https://arxiv.org/pdf/2609.02324v1)

**作者:** Makbule Gulcin Ozsoy `[一作]` `[通讯]` (Neo4j), Makbule Gulcin Ozsoy (Neo4j)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了适应性测试时推理策略，动态调整生成预算并对易问不做过度校正，提升了Text2Cypher的效率与可靠性。

**💡 创新点**

创新点在于两项策略：基于预估难度的可变生成预算和仅在需要时才执行的选择性执行指导校正。

**🔧 技术方法**

利用轻量级规则式难度估计器、语法与模式过滤、执行反馈驱动的迭代校正以及Gemma-4等LLM实现。

**📊 数据集**

在Text2Cypher基准（包含recommendations、companies、neoflix三大图谱）上进行实验。

**📈 对比分析**

与固定预算与全量校正的基线相比，适应性预算将平均生成轨迹减少30.7%，推理时间降低21–25%，执行成功率保持约92%，只损失不到0.5%。

**⚠️ 局限性**

主要局限是难度估计仍是基于手工规则，缺乏自学习或基于生成信心的动态调度；实验仅覆盖Text2Cypher且仅使用两款生成模型和一种校正模型，未验证跨任务与跨模型的普适性。

---

## 266. MineTRACE: An Evidence-Grounded Interactive Reasoning System for Mineral Prospectivity

**arXiv ID:** 2609.02060 | [PDF](https://arxiv.org/pdf/2609.02060v1)

**作者:** Yiran Zhang `[一作]` (University of Western Australia), Yihao Ding `[通讯]` (University of Western Australia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并部署了一个基于专家树的矿物勘探决策支持系统 MineTRACE，集成公开的地球化学、地球物理和地质数据，提供地图可视化、证据面板、自然语言助手等交互功能，用于评估并解释矿物潜力。

**💡 创新点**

创新点在于：① 将多源证据（化学、物理、地质）通过透明的专家树模型融合，生成可追溯、可解释的矿物潜力得分；② 将同一份证据记录同时供地图、热图、查询和对话接口使用，形成端到端一致的交互流程；③ 在未见新矿的盲测中验证模型的泛化能力，并展示系统如何通过独立证据支持新发现。

**🔧 技术方法**

技术实现包括：Python Django + PostGIS 做后台服务；专家树模型（手工定义专家、统计z分、离线拟合权重）做评分；ETL 处理 935 万化学测定、3420 知名矿区和多源栅格/矢量；自然语言助手通过 MCP 调用评分服务；大规模空间索引、稳健对数统计、距离特征工程等。

**📊 数据集**

数据集主要来自西澳地质调查局（GSWA）：CM02 Near Surface Geochemistry（约 935 万条测定），CM01 Mineralization Sites（3420 条已知矿区），CM08 Critical Minerals Basemap（7 个地球物理栅格 + 5 个地质向量），全部覆盖西澳约 2.5 百万平方公里。

**📈 对比分析**

评估方法：在四种负采样策略（随机、远程、非矿、空间）下，使用 ROC‑AUC 评估评分模型；平均 AUC 分别为 0.919（随机）、0.945（远程）、0.810（非矿）、0.684（空间）。多源证据融合相较单一来源提升 AUC；对话系统在 150 次人类评估中 92% 通过；在 2023‑2025 年公布的 6 个新金矿盲测中，5/6 位于西澳前 13%（3/6 位于前 10%），说明模型具有良好泛化能力。

**⚠️ 局限性**

局限性：① 仅为决策支持工具，仍需现场验证和专家判断；② 受公开数据覆盖、采样偏差、测定质量影响，空旷地区的评分为“插值”或弱；③ 专家树结构相对简单，某些矿物（如 Cu、Mn）在空间持久性上表现欠佳；④ 对话助手的正确性依赖评分可靠性，尚未完成大规模真实地质专家使用评估。

---

## 267. From Multi-Fisheye Sensing to Panoramic Perception: A Parallax-Aware Onboard Platform for Ultra-Low-Altitude UAVs

**arXiv ID:** 2609.02319 | [PDF](https://arxiv.org/pdf/2609.02319v1)

**作者:** Dun Dai `[一作]` (Beihang University), Quan Quan `[通讯]` (Beihang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

设计并实现了一个基于四镜头鱼眼相机的轻量级碳纤维无人机平台，机载实时将四个同步鱼眼视角转换为1280×640等距圆柱视图（ERP），同时保留原始视角、校准信息和GNSS导航数据，形成可直接供下游检测、定位等任务使用的统一接口。

**💡 创新点**

创新点包括：①在每个相邻相机重叠区独立选择投影半径，实现视差感知的自适应深度；②结合动态缝合、光度融合与残差网格校正，显著提升几何与色彩一致性；③多速率状态更新与Margin gating，兼顾实时性与精度；④完整的硬件同步四镜头设计与开源实现，提供从原始图像到ERP的完整数据接口。

**🔧 技术方法**

技术栈：四镜头鱼眼相机硬件同步；NVIDIA Jetson Orin NX GPU + CUDA 11.4 + TensorRT；Kalibr相机标定；SIFT/AKAZE特征匹配；投影半径库与动态深度选择；动态规划缝合搜索；两频段羽毛混合光度融合；指数平滑、Hysteresis、Margin gating；固定YOLOv10-N检测器、ResNet-18 CosPlace VPR等下游模型；OpenCV affine/homography对比。

**📊 数据集**

数据集：18条完整序列，共50k+四视角组，覆盖7个场景（Court、Farmland、Lake、North Playground、North Square、South Playground、South Square），包含日间、夜间、教-重复飞行以及近场结构压力测试，公开于GitHub。

**📈 对比分析**

对比方法：Fixed Feather、Fixed Depth、Global Radius、Per-seam（无残差网格）、Adaptive Seam、Ours（残差网格+Adaptive Seam）。性能：Ours在远场将中位数误差从5.66↓3.81、P90误差从9.97↓5.82（比Fixed Depth降低32.7%/41.6%）；近场中位误差从35.77↓9.25；嵌入式上Per-seam实现19.99 fps、0.026%时延丢失、13.29W平均功耗；VPR：ERP 8V Recall@1 74.9%、Recall@5 90.8%、误差17.2m；检测：8区ERP覆盖率55.1%。

**⚠️ 局限性**

局限性：仅建模单一深度层，无法处理混合深度或高度差异大的内容；依赖硬件同步与标定误差；GPU资源受限导致对极端光照、快速动态目标处理不足；残差网格在极端场景下的鲁棒性未知；平台重量约1kg，负载与飞行时间受限。

---

## 268. Linear Fusion MultiDiffusion for Fast Training-Free Spherical Panorama Generation

**arXiv ID:** 2609.01997 | [PDF](https://arxiv.org/pdf/2609.01997v1)

**作者:** Akio Hayakawa `[一作]` (University of Tokyo), Tatsuya Harada `[通讯]` (University of Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无训练的全景图像合成框架，扩展MultiDiffusion以支持任意线性投影，从而在预训练文本‑图像扩散模型上实现全景生成。

**💡 创新点**

核心创新在于将多视角聚合重新表述为带正则化的最小二乘问题，并在去噪循环中使用Krylov子空间迭代求解，从而打破了传统直接像素采样的限制，实现更密集、更自然的投影映射，显著减少视角评估次数并提升效率。

**🔧 技术方法**

主要技术包括MultiDiffusion框架、ERP（等经纬图）潜空间、双线性插值投影、正则化最小二乘（岭或拉普拉斯）、矩阵无关Krylov求解器（LSMR/PCG）、FLUX预训练扩散模型、视角后期细化。

**📊 数据集**

使用FLUX预训练扩散模型以及SphereDiff提供的20组文本提示，随机种子生成10幅全景；评估时从生成的全景渲染14幅视角，采用MUSIQ、LAION美学评分、CLIP文本相似度、Q‑Align、Qwen‑VL等指标。

**📈 对比分析**

与DynamicScaler、SphereDiff（无训练）以及Text2Light、PanFusion（训练）做对比。结果显示在图像质量、文本对齐、Aesthetic、Q‑Align、失真与连续性等指标上均优于基线；速度方面比DynamicScaler快3.58×，比SphereDiff快15.36×，将每幅全景的生成时间从数分钟降低至约1分钟。

**⚠️ 局限性**

局限性包括：仍依赖ERP投影，极角区仍可能出现轻微失真；需要一定数量的视角（N≥14）才能保证全局连贯性；对不同拓扑（如球面之外）尚未直接适用；并且在极端文本提示或极端场景下可能出现与训练域不匹配的细节失真。

---

## 269. KGVoyager: Knowledge Graph Agnostic Question Answering via Agentic Navigation

**arXiv ID:** 2609.01780 | [PDF](https://arxiv.org/pdf/2609.01780v1)

**作者:** Essam Wisam `[一作]` (University of Texas at Arlington), Chengkai Li `[通讯]` (University of Texas at Arlington)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种KG无关的代理架构KGVoyager，利用SPARQL端点通过搜索、探索和执行工具，动态发现图结构并生成SPARQL查询。

**💡 创新点**

创新点包括：仅使用轻量级类索引替代全量实体索引，结合语义与关键词的混合检索，以及通过探索工具直接从图中学习属性和实体关系。

**🔧 技术方法**

核心技术为大型语言模型（如GPT-5.3/5.4、Gemini 3 Flash、Devstral Small）结合ReAct框架、SPARQL工具调用、密集向量+BM25混合检索、执行反馈循环。

**📊 数据集**

实验使用四个领域专属基准：Climate Models KG、SOCKG、DREAM‑KG和DBLP‑QuAD，涵盖气候、土壤碳、地方政府服务和学术元数据。

**📈 对比分析**

与前沿系统GRASP比较，KGVoyager在四个基准上平均提升约8个F1分点，同时将成本与运行时分别降低约22%。

**⚠️ 局限性**

局限性在于仅适用于遵循RDFS/RDF类型约定的图，无法直接处理非RDFS建模的图（如Wikidata），且领域专属数据集仍相对稀缺。

---

## 270. Scalable Bayesian Optimization of Composite Functions for Image-Based Inverse Problems in Materials Characterization

**arXiv ID:** 2609.02126 | [PDF](https://arxiv.org/pdf/2609.02126v1)

**作者:** Dasol Yoon `[一作]` (Cornell University), Peter I. Frazier `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种可扩展的复合函数贝叶斯优化（SBOCF）方法，用于通过匹配实验与仿真得到的PACBED图像来估计样品厚度与晶体误倾角。

**💡 创新点**

创新点在于将高维像素输出压缩为 patch‑level 汇总并引入乘法与加法校正项，将原本需要建模  M²  个像素的高维 GP 约简到 P+2 个输出，同时保留像素级 SSE 目标，显著提升了贝叶斯优化在高维模拟问题中的效率。

**🔧 技术方法**

使用了贝叶斯优化、贝叶斯优化复合函数（BOCF）、多输出高斯过程、期望改进（EI）采样策略、abTEM 多切片仿真、4D‑STEM PACBED 图像处理以及 patch‑level 统计等技术。

**📊 数据集**

使用了合成 SrTiO₃ 的 PACBED 数据集（不同厚度与倾斜角度）以及实验获得的 SrTiO₃ PACBED 图像，并在此基础上进行了多切片电子ptychography 的重建实验。

**📈 对比分析**

将 SBOCF 与标准贝叶斯优化（EI、KG）及随机采样进行比较。对合成数据，SBOCF 在 50 次仿真预算下平均最终 SSE 可低至 290 倍（厚样本）或 47 倍（薄样本），参数误差提升 23–51 倍；在实验数据中，SBOCF 的 SSE 相比 EI、KG 分别提升约 1.08× 与 1.13×，且厚度与倾斜角估计更为准确；在下游多切片ptychography 重建中，使用 SBOCF 估计的参数可恢复清晰的原子柱，显著优于忽略误倾或使用默认参数。

**⚠️ 局限性**

局限性包括：仅针对厚度与倾斜角等少数参数，仍需依赖仿真模型的准确性；对噪声或实验-仿真差异更敏感；需预先选择 patch 方案与校正范围（c）；每次迭代的采集函数优化成本高于单输出 BO。

---

## 271. Grounded, Compute-Efficient LLM Policy Agents for Energy-Poverty Equity in Physically-Constrained Peer-to-Peer Energy Markets

**arXiv ID:** 2609.01918 | [PDF](https://arxiv.org/pdf/2609.01918v1)

**作者:** Kunal Jadhav `[一作]` (Arizona State University), Siddhesh More `[通讯]` (Arizona State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了EqGrid，一个闭环模拟框架，用低频LLM制定价格/碳边界与补贴，结合MARL交易和物理电网约束来缓解能源贫困。

**💡 创新点**

创新点在于将基于真实数据的能源贫困人设、可压缩LLM决策层、MARL市场以及物理网格约束整合，并量化AI碳足迹与社会影响。

**🔧 技术方法**

技术包括开源LLM策略代理、MAPPO多智能体强化学习、连续双拍卖、IEEE‑33公交子母网格的Newton‑Raphson电压计算与投影门。

**📊 数据集**

数据集为欧盟SILC Hungary 边缘分布与 UCI 单户智慧表计数据，用于人设生成与负载曲线验证。

**📈 对比分析**

通过与无政策、透明规则基线及云端大模型对比，LLM策略显著降低能量负担不平等（Gini 0.305）、平均负担与日均成本，且压缩模型至 0.8B 仍保持 92% 公平收益，能耗每决策仅约 0.01Wh。

**⚠️ 局限性**

局限包括人设仅基于边缘分布、负载验证使用跨区域单户数据、仅模拟单条馈线且未真实部署、能耗估算为代理模型近似、补贴不具财政可持续性、未验证真实系统安全性。

---

## 272. When Agents Implement Systems: A Case Study in Defects, Detection, and Evaluation Rigor

**arXiv ID:** 2609.01985 | [PDF](https://arxiv.org/pdf/2609.01985v1)

**作者:** Phanindra Reddy Madduru `[一作]` `[通讯]` (Amazon.com), Phanindra Reddy Madduru (Amazon.com)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究对一名LLM编码代理在构建多组件数据系统（图存储、元数据过滤向量索引、异步采集管道、力导向可视化客户端）过程中出现的五种系统级缺陷进行了观察性案例研究，并对指定的检索过滤策略在HotpotQA数据集上的表现进行了量化评估。

**💡 创新点**

创新点在于首次系统地记录并分类LLM代理自发引入的系统缺陷，展示其诊断与修复过程，并通过外部基准验证了图驱动检索过滤决策的实效性。

**🔧 技术方法**

所用技术包括Claude LLM代理（CLI交互）、图数据库、带元数据过滤的向量索引、异步多阶段采集管道、力导向图形可视化、自动化单元测试、截图比较以及HotpotQA多跳问答基准。

**📊 数据集**

使用的数据集为HotpotQA的distractor设置（300道验证问答，合计2,994段落），以及内部汇总的文档语料库。

**📈 对比分析**

通过对比过滤与未过滤两种检索策略的recall@k（k∈{1,3,5,10}）与Gold支持事实，发现过滤策略在k≥3时可达100%召回率，而未过滤策略在k=10仅达69%召回，统计显著性检验p<10⁻⁴，表明过滤显著提升检索精度。

**⚠️ 局限性**

局限性包括仅在单一代理单一会话中进行实验，缺乏对回归修复效果的重新测量，对实体识别阶段的假设（使用Gold标签代替实际识别）以及结果可推广性未得到验证。

---

## 273. A Unified Rate-Distortion Perspective on Vector, Product, and Scalar Quantization

**arXiv ID:** 2609.02107 | [PDF](https://arxiv.org/pdf/2609.02107v1)

**作者:** Xianghong Fang `[一作]` (University of Toronto), Tim G. J. Rudner `[通讯]` (Vijil)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于速率‑失真理论的统一视角，对向量量化（VQ）、产品量化（PQ）和标量量化（SQ）在视觉离散表示中的效果进行系统分析与比较。

**💡 创新点**

①证明最小化失真比最大化码本利用率更为根本，并与直通估计（STE）梯度误差相关；②提出公平比较的两条关键条件：相同潜在分布与相同固定长度速率；③揭示在低维结构下VQ能显著优于PQ、SQ，并在实验中验证了VQ的失真优势和重建质量提升。

**🔧 技术方法**

使用速率‑失真理论推导、理论证明、实验评估；采用VQ、PQ、SQ及其多种变体（如EMA、Online、Wasserstein、MMD VQ/PQ等）；在VQ‑Transplant框架中对潜在空间进行量化器替换与解码器适配；计算指标包括PSNR、SSIM、LPIPS、r‑FID、r‑IS。

**📊 数据集**

主要使用 ImageNet‑1K 数据集；实验结果也在 FFHQ、CelebA‑HQ（潜在空间）和 CelebA‑HQ（像素空间）上验证。

**📈 对比分析**

在相同潜在分布、相同 token 数量 T 与相同码本大小 K 的条件下，替换量化器并进行解码器适配。实验显示 VQ（尤其 MMD VQ）失真最低、r‑FID 最高；PQ 次之；SQ 最差。失真与重建质量高度相关，而码本利用率关联弱。

**⚠️ 局限性**

局限性：实验主要基于已预训练的 VAR tokenizer 与 VQ‑Transplant 框架，未覆盖从零开始的完整训练流程；只考虑固定长度编码率，未探讨可变长度或熵编码；低维结构假设在真实数据中可能不完全成立；仅比较了三种量化方法，未涉及更复杂或混合量化方案。

---

## 274. Source-Free Class Relearning: Diagnosing Forgetting in Class Unlearning

**arXiv ID:** 2609.02018 | [PDF](https://arxiv.org/pdf/2609.02018v1)

**作者:** Zahra Dehghani `[一作]` (LIVIA), Mohammadhadi Shateri `[通讯]` (LIVIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种源免费类重学习审计（SFRA），通过合成特征空间探针和轻量级分类头更新，在没有任何源数据或预训练检查点的条件下检测未学习后模型是否仍能恢复被遗忘类。

**💡 创新点**

创新点在于利用理论充分对齐条件和合成探针实现源免费重学习诊断，并引入综合评价指标Relearning Score（RS）及其与重训练参考的差值ΔRS，首次实现对未学习后可恢复性的定量评估。

**🔧 技术方法**

采用白盒访问模型、Gaussian采样、Softmax置信度过滤、低/高置信度探针生成、梯度步更新以及RS/ΔRS指标进行评估。

**📊 数据集**

在CIFAR‑10、CIFAR‑100、TinyImageNet三大数据集上，并结合ResNet‑18、ViT‑B/16、Swin‑T三种骨干网络进行实验。

**📈 对比分析**

与十种主流未学习方法以及Prototypical Relearning Attack（PRA）对比，SFRA发现部分方法（如Bad Teacher、DELETE）在RS和ΔRS上均表现出显著可恢复性，表明其未学习效果并未彻底抹除结构；而SCRUB、Negative Gradient+等方法表现相对较弱。

**⚠️ 局限性**

局限性包括仅评估分类头输入层的特征空间、对多类遗忘时探针分配存在不确定性、未能证明完全信息抹除以及缺乏更深层级或更强统计检验。

---

## 275. When Literature Data Mislead Artificial Intelligence in Materials Discovery

**arXiv ID:** 2609.01621 | [PDF](https://arxiv.org/pdf/2609.01621v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 276. A Layered Taxonomy for Chinese Learner Grammatical Error Annotation

**arXiv ID:** 2609.02153 | [PDF](https://arxiv.org/pdf/2609.02153v1)

**作者:** Mengyang Qiu `[一作]` (Saint Elizabeth University), Jungyeul Park `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一套分层的中文语法错误注释体系，将 CGEC 的编辑操作与语言学和教学诊断相结合，并对其覆盖率与一致性进行评估。

**💡 创新点**

创新点在于提出双路径（正字错误与非正字错误）分层标签，既保持可重复的编辑标记，又为教学提供可选的功能/构造扩展；同时首次在大语言模型上测试一致性，验证该体系的实用性。

**🔧 技术方法**

采用自动化对齐工具提取编辑、ChERRANT/ERRANT 处理框架、POS 与词法分析工具；使用大语言模型（如 ChatGPT）生成标签；并用统计学指标（Fleiss κ、Krippendorff α）评估一致性。

**📊 数据集**

使用 MuCGEC 中文语法错误纠正基准集（自动提取的 4,426 条编辑记录）以及 5 个 LLM 在 391 条样本上的实验。

**📈 对比分析**

通过对 MuCGEC 编辑的覆盖率分析（约 55.9% 替换、27.4% 缺失、16.7% 多余，24% 正字错误，17% 扩展触发）和对 5 个 LLM 的一致性评估（正字筛选与编辑操作一致性 >90%，整体完整标签一致性约 70%，扩展层一致性相对较低），展示了该分层体系在覆盖和可一致性上的表现。

**⚠️ 局限性**

局限在于仅依赖自动工具和单一人工专家评估，未完成多标注者可靠性研究；扩展层覆盖范围有限，未覆盖所有构造；缺乏不同水平、体裁、背景语料的泛化验证；未涵盖话语层面错误等更高层次问题。

---

## 277. Act More, Decide Less: Skill-Guided Adaptive Action Chunking for Long-Horizon LLM Agents

**arXiv ID:** 2609.02042 | [PDF](https://arxiv.org/pdf/2609.02042v1)

**作者:** Yanting Yang `[一作]` (Rutgers University), Dimitris N. Metaxas `[通讯]` (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于技能引导的变量长度动作块化方法，用大语言模型在长时序交互任务中直接输出可变长度的原子动作序列，减少LLM决策次数；

**💡 创新点**

创新点在于利用成功轨迹自动诱导的程序化技能来提供块边界监督，并通过混合前后向强化学习和块感知优势分配，解决传统多动作RL难以学习合理块界限的问题；

**🔧 技术方法**

采用LLM强化学习框架（RLVR）与GRPO、GiGPO等方法，构建程序化技能库，执行混合滚动（primitive‑chunk 与技能增强），并使用块感知两级优势进行策略优化；

**📊 数据集**

在两个长时序文本环境ALFWorld和ScienceWorld上进行实验，使用Qwen3‑4B、Llama‑3.1‑8B‑Instruct等LLM；

**📈 对比分析**

与提示式、ReAct、Reflexion以及GRPO、GiGPO、Multi‑action GRPO等基线相比，模型在成功率上提升7%–31%（视任务而定），且平均LLM决策轮数降低最多约78%，训练样本需求也显著减少；

**⚠️ 局限性**

局限性包括仅在文本环境验证，依赖成功轨迹生成技能结构，且在高随机性或安全关键场景下块化执行可能导致缺乏及时重规划。

---

## 278. Belief-Calibrated Optimization: An Explicit World Model for Agentic Optimization

**arXiv ID:** 2609.01861 | [PDF](https://arxiv.org/pdf/2609.01861v1)

**作者:** Yuhan Chen `[一作]` (Virginia Tech), Ruoxi Jia `[通讯]` (Virginia Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Belief‑Calibrated Optimization（BCO），在LLM代理优化循环中以持续的上下文文件形式记录并不断校正关于环境对代码编辑响应的世界模型；

**💡 创新点**

创新点在于将代理的隐式信念显式化为可写的、可更新的文档，并通过预测‑观察‑校正机制持续修正，从而在不改变冻结模型的前提下提升优化效果；

**🔧 技术方法**

技术核心是利用LLM的推理能力进行预测和校正，采用Markdown文档作为世界模型，构造原子信念并根据评价结果更新，配合编码代理（Kimi‑K2.6、GAIA、Codex等）执行提议、评估和编辑；

**📊 数据集**

数据集包括五个基准：LongMemEval‑s、LoCoMo（记忆QA），GAIA（工具使用QA），AppWorld（代码动作代理），Terminal‑Bench 2.0（终端代理），以及相应的训练/测试拆分；

**📈 对比分析**

对比方法为无世界模型的匹配优化（即仅使用原始证据重构信念），结果显示BCO在训练集的通过率提升+0.025到+0.150，并在所有测试拆分保持这一差距；在目标模型替换实验中，BCO的构造的 scaffold 在 GAIA 和 AppWorld 的部分设置下仍保持首位；

**⚠️ 局限性**

局限性包括：世界模型为近似的自然语言摘要，缺乏严格的贝叶斯后验；环境响应被假设为在一次运行中固定，可能导致对目标模型漂移不敏感；实验仅覆盖五个筛选过的基准，且每个基准仅做一次对比轨迹，缺乏更广泛的重复性和跨模型泛化验证。

---

## 279. Median-of-Means as an Extremal Convex Estimator and a Nonconvex Route to the Trimmed Oracle

**arXiv ID:** 2609.01689 | [PDF](https://arxiv.org/pdf/2609.01689v1)

**作者:** Angshul Majumdar `[一作]` `[通讯]` (Indian Institute of Information Technology), Angshul Majumdar (Indian Institute of Information Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在块级污染模型下，使用非凸的 block‑L_p 目标（p∈(0,1]）构造鲁棒均值估计器，并给出其确定性与概率学性质，提出从经典 median‑of‑means（MoM）到理想剪枝估计器（trimmed‑block oracle）的连续 1–p–0 路径。

**💡 创新点**

①证明在任何凸块 M‑估计器类中，MoM 的确定性鲁棒常数 1/(1‑2ε) 已达极限，凸聚合无法突破；②引入非凸 block‑L_p，显示其全局最优解在 p→0 时与剪枝 Oracle 对齐；③证明该非凸目标具有良好的能量景观（无坏局部极小值）。

**🔧 技术方法**

确定性鲁棒性分析、凸/非凸优化理论、块级污染模型、分段常数推导、无偏/有偏估计、概率界（(2+δ) 短尾、集中）、高维扩展（稀疏均值、稀疏回归）、梯度/子梯度优化。

**📊 数据集**

实验使用合成数据：学生‑t（ν=3）重尾样本、ε=0.2 的对抗性污染样本，以及块级分离污染样本；不涉及真实公开数据集。

**📈 对比分析**

与 MoM、trimmed‑mean、Huber 均值估计器对比。结果表明：在重尾无结构污染时性能相近；在对抗性污染时随 p↓0 错误逐渐下降；在块级分离污染时 p=0.2 的 block‑L_p 与 trimmed‑mean 接近，显著优于 MoM 与 Huber。整体说明小 p 在结构化污染下提供近 oracle 性能。

**⚠️ 局限性**

改进是有条件的——需要块级误差显著分离；p 的取值需要预先设定或通过自适应方法；常数不够精确，尚未给出最优化算法的收敛理论；实验仅在合成数据上验证，缺乏对真实高维数据的评估。

---

## 280. A Physics-Consistent Benchmark for Contact-Rich Human-Robot Interaction in Assistive Care

**arXiv ID:** 2609.02402 | [PDF](https://arxiv.org/pdf/2609.02402v1)

**作者:** Chengxiao He `[一作]` (Tongji University), Shenzhen Zhu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种物理一致的基准，用于评估辅助护理场景中机器人与人类的接触丰富交互，并以机器人洗澡任务为实例。

**💡 创新点**

创新点在于三方面：① 构建可被物理校准的被动响应人类模型；② 设计基于物理安全门的多维评分体系（峰值力、关节余量、运动活跃度、力稳定性等）；③ 采用冻结观测协议，防止策略获取模拟器内部状态，保证评估真实可迁移性。

**🔧 技术方法**

使用了柔体仿真栈（包含骨骼-柔软体耦合、法向力传递等），结合Franka Panda的阻抗控制实验进行力-挤压曲线校准；视觉输入包括RGB、深度、点云、语义分割；策略模型包括LLM增强的有限状态机、VoxPoser 以及零射击的π_0.5。

**📊 数据集**

数据集为基于医学护理假人（manikin）的力-挤压测量数据，用于对仿真参数的校准；实验任务共7个子任务（T1–T7）在仿真环境中随机化起始姿态，共140次试验。

**📈 对比分析**

与传统仅以任务完成率评估的方法相比，基准揭示了显著的物理失败：LLM状态机在72.9%完成率下，安全门筛选后仅剩56.4%；VoxPoser 虽轻量且力稳定，却仅完成27.9%；π_0.5 零射击直接失败（0.7%）。因此，任务成功率并不能代表物理安全与有效交互。

**⚠️ 局限性**

局限性包括：① 只使用医学护理假人而非真实人类，可能忽略人类生理差异；② 上臂仿真仅复用前臂校准参数；③ 只评估了三种策略，未覆盖更广泛的控制算法；④ 未深入研究软体参数对结果的敏感性。

---

## 281. Sim2Signal: Sim-to-Real Benchmarks for Traffic Signal Control

**arXiv ID:** 2609.01676 | [PDF](https://arxiv.org/pdf/2609.01676v1)

**作者:** Ferdous Al Rafi `[一作]` (Arizona State University), Hua Wei `[通讯]` (Arizona State University)

**通讯引用:** 8243 | [OpenAlex ID](https://openalex.org/A5100777770)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出了一套控制实验框架，用来在模拟环境中单独诱发交通信号控制中的观测、动作、转移和奖励四个 Sim‑to‑Real 差距，并对 18 种消减方法在 33 种设置、10 个经过校准的真实路网中进行评估。

**💡 创新点**

创新点在于将 Sim‑to‑Real 问题拆解为可控的四类差距，并通过“sim2sim”协议在统一协议下系统评测各类消减方法，从而发现差距严重程度并不能预测消减效果，且方法的有效性高度依赖路网与设置。

**🔧 技术方法**

采用 LibSignal 统一 CityFlow/SUMO 仿真，设计预训练/训练/部署流水线，结合域随机化、潜在空间适配、重建、延迟‑Q、PRLight、动作屏蔽、奖励推断、多目标 RL 以及动态奖励塑形等多种技术实现差距消减。

**📊 数据集**

使用了十个经过校准的真实路网，来源于五个真实地点（Tempe、Bullhead、Cologne、Ingolstadt、Hangzhou），其中 Tempe 与 Bullhead 为新建的 UTDF NEMA 方案网络。

**📈 对比分析**

通过将每种方法的平均行驶时间差（ATT）和奖励差距（regret）与直接转移基线对比，评估各方法性能；结果显示延迟和相位转换差距可被预测方法显著恢复，而观测、转移和奖励差距恢复不稳定，方法效果随路网和设置显著变化。

**⚠️ 局限性**

局限性包括仅考虑车辆单代理、未同时诱发多种差距、未涵盖行人交互或更复杂的部署约束，且实验依赖仿真数据，可能不完全泛化到所有真实交通场景。

---

## 282. TempoGround: State-Aware Streaming Visual Grounding with Vision-Language Models

**arXiv ID:** 2609.02359 | [PDF](https://arxiv.org/pdf/2609.02359v1)

**作者:** Leqian Ding `[一作]` (Xi'an Jiaotong University), Fei Wang `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向流式输入的视觉定位框架TempoGround，能在实时视频中持续追踪并定位符合语言查询的物体；

**💡 创新点**

核心创新在于利用跨帧对应的状态感知课程预测，先关联实例、估计进入/持续/退出状态，再从2D框回归到相机帧3D框，并通过Streaming Grounding Reinforcement（SGR）用可验证的定位、身份和一致性奖励强化闭环推理；

**🔧 技术方法**

技术包括VLM解码器的自回归结构、跨帧轻量化对应线索、离散状态预测、2D→3D解码链、三阶段训练（单帧监督→流式监督→强化学习）以及GRPO强化学习框架；

**📊 数据集**

训练数据来自多源公开数据集（Objects365、V3Det、COCO、LVIS、EgoObjects、BDD100K、nuImages、PACO、RefCOCO、RefL4、Flickr30k、ScanNet++、ScanNet、ADT、ASE、ARKitScenes、Hypersim、Objectron、SUN‑RGBD、KITTI、nuScenes 等），规模约26M单帧样本、5.1M流式样本；

**📈 对比分析**

在多任务（检测+定位）和多数据集的因果流式评估中，TempoGround在2D F1@0.5/0.95和3D F1@0.25/AP_3D均显著优于现有通用VLM和专用grounder，尤其在3D定位上提升幅度达6–7.5点；

**⚠️ 局限性**

局限性包括对跨帧对应线索的依赖，过度噪声或快速相机运动时仍可能导致误配；目前仅在RGB摄像头的相机帧坐标系下实验，未验证对其他传感器或多相机场景的泛化能力；

---

## 283. Auditory Illusion Benchmark for Large Audio Language Models

**arXiv ID:** 2609.02277 | [PDF](https://arxiv.org/pdf/2609.02277v1)

**作者:** Hayoon Kim `[一作]` (Seoul National University), Kyogu Lee `[通讯]` (Seoul National University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并发布了首个针对大型音频语言模型（LALMs）的听觉错觉基准，涵盖十种代表性错觉，并与受控人类听觉实验进行对比评估。

**💡 创新点**

创新点在于：①把听觉错觉引入模型评估作为认知对齐的诊断工具；②将错觉分为物理驱动与物理+知识驱动两类；③提出人类相似度（HLA）、现实对齐度（RA）和错觉易感指数（ISI）等新指标。

**🔧 技术方法**

采用多模态Transformer LALMs（Audio Flamingo、Gemini、Qwen等）对错觉音频进行多项选择问答，使用绝对音高听者进行在线人类实验，统计并解析模型与人类的回答分布。

**📊 数据集**

自研“Auditory Illusion Benchmark”数据集，包含10,385个错觉样本和4,444个控制样本，按音乐、语音、通用声以及物理/知识驱动两大类进行标注。

**📈 对比分析**

通过对模型输出与人类分布的比较，计算HLA、RA、ISI三项指标。结果显示：无模型达到人类水平；最高ISI仅0.505；物理错觉中少数模型表现类似人类，知识驱动错觉中模型更趋物理或受语言先验影响，整体缺乏稳定的错觉敏感性。

**⚠️ 局限性**

局限性包括：模型在控制条件下表现不稳定；错觉易感性高度依赖提示语；缺乏统一训练机制将低层声学与高层先验结合；基准未覆盖所有听觉错觉（尤其与双耳生理相关的）。

---

## 284. Not All Matches Are Equally Valuable: An Online Experiment of Retention-Focused Recommendation in a Job-Matching Platform

**arXiv ID:** 2609.01652 | [PDF](https://arxiv.org/pdf/2609.01652v1)

**作者:** Tatsuya Ute `[一作]` (Wantedly, Inc.), Yuta Saito `[通讯]` (Hanjuku-kaso Co., Ltd.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在Wantedly Visit这类双边匹配平台上，对风险用户（最近匹配数少的用户）实施后处理得分提升，以减少用户流失；

**💡 创新点**

创新点在于将保留率视为核心目标，采用简单得分提升方式将曝光重新分配给高流失风险用户，而非单纯最大化匹配数量；

**🔧 技术方法**

利用匹配概率预测模型、基于匹配概率的后处理公式、离线参数调优和在线实验中的逻辑回归估计；

**📊 数据集**

使用了Wantedly Visit平台的真实日志数据，包括用户登录、招聘者投递、匹配、回应、曝光等信息，实验覆盖约4周时间；

**📈 对比分析**

通过在招聘者层面随机化的线上A/B实验对比，结果显示处理组的用户流失几率比对照组略低（OR≈0.95，统计不显著），总体匹配数略增（≈5.4%，亦无显著差异）；

**⚠️ 局限性**

局限包括仅在单一平台与短期实验内验证、招募者级随机化导致的交叉曝光、统计显著性不足、阈值与提升参数离线设定、未与公平或竞争模型直接对比、缺乏因果中介机制验证。

---

## 285. PolERo: Studying Political Evasion in Romanian

**arXiv ID:** 2609.02391 | [PDF](https://arxiv.org/pdf/2609.02391v1)

**作者:** Gabriel Stefan `[一作]` (University of Bucharest), Sergiu Nisioi `[通讯]` (University of Bucharest)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

介绍了PolERo——一份3,574条罗马尼亚总统访谈问答对，采用两层模糊与含义不明的分类体系，用于研究政治回避与响应清晰度。

**💡 创新点**

首次在非英语语境下发布政治回避数据集，并提出滑动窗口+最大池化的多头编码器架构来联合预测清晰度与细粒度逃避类别。

**🔧 技术方法**

使用TF-IDF + 逻辑回归、单/多头预训练Transformer（RoBERTa、XLM‑R、RoBERT‑large 等）以及零/少量提示的 LLM（GPT‑5.4、Llama‑3.3‑70B 等）进行实验。

**📊 数据集**

英文 CLARITY 数据集与自研罗马尼亚 PolERo 数据集；另外利用机器翻译数据进行跨语言训练。

**📈 对比分析**

在宏观 F1 上，单头 RoBERTa‑large 在英文 3 类清晰度任务取得 0.58，双头模型提升至 0.71；在罗马尼亚 9 类逃避任务，RoBERT‑large 单头 0.51，双头 0.70；LLM 在细粒度任务上表现最佳，约 0.73‑0.76 宏观 F1；跨语言实验显示英语模型迁移到罗马尼亚下降 6‑7 点，罗马尼亚到英语下降 11‑14 点；多语训练略有提升。

**⚠️ 局限性**

仅覆盖单一非英语语言，注释者背景有限，跨语言推广性未知；模型对含义含糊的类别仍难以准确区分；实验成本高，需大显存 GPU。

---

## 286. Hardware-Accelerated Instance Segmentation for Resource-Constrained Space Robotics with Criticality Analysis

**arXiv ID:** 2609.02219 | [PDF](https://arxiv.org/pdf/2609.02219v1)

**作者:** Siddhant Shete `[一作]` (German Research Center for Artificial Intelligence), Frank Kirchner `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了面向月球环境的实例分割框架，集成了激活方差采样、量化校准、DPU部署与辐射鲁棒性分析。

**💡 创新点**

提出了AVIS无标签自适应量化校准方法、硬件友好的YOLOv8分割模型改造以及软件级辐射关键性评估与分级缓解。

**🔧 技术方法**

采用INT8后训练量化+偏置校正、HardSwish替换、DPU静态编译、CPU–DPU分工、辐射事件概率模型与ECSS安全级别评估等技术。

**📊 数据集**

在含25,000张低光合成与实测月球场景图像的数据集上进行训练与验证。

**📈 对比分析**

在LuNiS微型月球车上与FP32基线对比，随机校准下mAP降6.6%，AVIS+偏置校正恢复至mAP 0.786，推理延迟309 ms、功耗5.7 W，辐射关键性降低31.7%。

**⚠️ 局限性**

关键性分析基于理论模型，未进行实测辐射注射；AVIS依赖离线代表性数据，迁移到未知地形可能受限；推理时延对更高速机器人仍需进一步优化。

---

## 287. READY or Not: Reliable Enterprise Agent Deployment

**arXiv ID:** 2609.02095 | [PDF](https://arxiv.org/pdf/2609.02095v1)

**作者:** Veronica Chatrath `[一作]` (Scale AI), Yuan Xue `[通讯]` (Scale AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一个名为 READY 的评估框架，能够在企业工作流上对 AI 代理进行部署资格评估，既考虑代理的自主性能，又结合人工监督、成本和风险约束，给出统计可靠的部署配置。

**💡 创新点**

创新点包括：
- 把工作流级成功定义（多维度评估）与监督策略优化和统计检验统一在一个流程里；
- 将工作流定义与部署资格抽象分离，使同一份执行证据可在不同部署需求下复用；
- 通过可靠性阈值约束和最小成本优化，自动搜索最优监督策略；
- 采用开发/验证拆分、置信下限检验，使部署声明在新样本上具有统计支持；
- 提供“可靠性‑监督‑成本”三维部署配置文件，便于对比与决策。

**🔧 技术方法**

主要技术：
- 约束优化（在监督策略空间搜索满足可靠性目标且成本最小的策略）；
- 轨迹回放评估（终端接受‑升级策略可在同一执行轨迹上评估多策略）；
- 统计置信下限（Clopper‑Pearson 等）用于验证可靠性阈值；
- 风险函数（可选的 CVaR 等）用于处理严重失败；
- 成本模型（agent 成本 + 人工审核成本）与监督成功率假设结合。

**📊 数据集**

数据集：
- CliniCARE‑Bench，750 条临床审计案例；
- 16 种不同的大语言模型（GPT‑5.4、Sonnet‑5、Gemini 等），每个模型对所有案例执行一次。

**📈 对比分析**

比较方法与性能：
- 通过“可靠性‑监督前沿”绘制各系统在不同可靠性目标下所需的人工审核比例；
- 在 76% 可靠性目标下，同等自主准确率的 GPT‑5.4 与 Sonnet‑5 需要分别 39.2% 与 29.6% 的审核率，显示相同准确率可产生大幅不同的部署成本；
- 在 16 系统中，可靠性相近但路由信号差异导致审核负担差异达 20% 以上；
- 所有系统在验证集上均通过统计检验（置信下限≥目标）后被认为可部署。

**⚠️ 局限性**

局限性：
- 仅在终端接受‑升级（轨迹不变）场景下验证，未涵盖轨迹依赖的交互式监督；
- 人工审核成功率和成本均为外部假设，未在实验中直接测量；
- 仅使用 CliniCARE‑Bench 作为实例，缺少其他行业工作流的跨域验证；
- 依赖分布不变的假设，未对数据漂移、工作流变更或代理行为变化的鲁棒性进行评估。

---

## 288. CAT-Flow: Curvature-Adaptive sTeps for Flow Matching

**arXiv ID:** 2609.01746 | [PDF](https://arxiv.org/pdf/2609.01746v1)

**作者:** Qinchan Li `[一作]`, Hao Zhang `[通讯]` (Simon Fraser University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出两种训练无关、轻量级的自适应步长采样算法，利用流匹配向量场的曲率信息来动态调整步长；

**💡 创新点**

通过建立流匹配与梯度流的形式联系，将优化中的自适应曲率概念迁移至采样，首次在流匹配生成中使用时间/值曲率驱动的步长自适应；

**🔧 技术方法**

基于流匹配的ODE求解、梯度流理论、RMSProp式曲率估计以及有限差分求曲率；

**📊 数据集**

使用公开的 DiffDB 文本提示集，对 FLUX‑1‑dev、FLUX‑1‑Krea‑dev、FLUX‑1‑Schnell 及 SD‑3.5‑large 四个流匹配模型进行评估；

**📈 对比分析**

与默认动态步长（Dynamic）和固定步长（Fixed）等基线对比，使用 CLIP、AES、HPSv3 三个图像质量指标；实验表明新方法在大多数步长下均能以 40% 左右更少的生成步骤达到相同或更好的质量，并在低步长场景中显著优于基线；

**⚠️ 局限性**

受限于预训练向量场的精度，且曲率估计采用近似方法，未来可通过后训练或更精确的曲率计算进一步提升。

---

## 289. VirSqueezer: Generating Realistic Deformations and Squeezing Dynamics in VR from Fine-Grained Squeezing Controls

**arXiv ID:** 2609.01698 | [PDF](https://arxiv.org/pdf/2609.01698v1)

**作者:** Qian Zhang `[一作]` (Beijing Technology and Business University), Weidong Cai `[通讯]` (University of Sydney)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研发了VirSqueezer框架，利用细粒度指尖控制信号驱动VR中3DGS物体的压缩变形及二次视觉效果生成。

**💡 创新点**

创新点在于：①将指尖控制映射为Dirichlet边界约束驱动MPM仿真；②将物理仿真与生成模型相结合，生成裂缝、溢流等二次效果；③通过LLM推断材质属性并实现力感反馈，形成双向交互。

**🔧 技术方法**

使用SenseGlove采集手势，SC-CAE估计接触面积，Dirichlet边界条件，MPM物理仿真，LLM估算材质，AnimateDiff/Stable Diffusion/ControlNet等条件生成模型，以及G-buffer条件。

**📊 数据集**

评估基于8种物体（罐头、毛绒玩具、橡皮鸭、粘土、番茄、橙子、橡胶球、纸杯）由BlenderNeRF生成的场景，并结合SenseGlove控制信号与多大模型评测。

**📈 对比分析**

与PhysGaussian、OmniPhysGS、DreamGaussian4D、Gaussians-to-Life等基线在VBench、Physical Commonsense、人类评测、LLM评测以及BDC、SDC、mIoU等自定义指标上对比，VirSqueezer在视觉质量、运动一致性与物理合理性上均优于所有基线。

**⚠️ 局限性**

主要局限包括：1）生成模型耗时，尚不能实现实时渲染；2）LLM推断的材质属性不够精确，缺乏实际测量；3）仅针对单物体压缩，未扩展到多物体或更复杂交互；4）实验对象有限，需进一步扩大验证范围。

---

## 290. Interpretable Symptom Vectors for Depression in a Large Language Model

**arXiv ID:** 2609.01832 | [PDF](https://arxiv.org/pdf/2609.01832v1)

**作者:** Fangyi Zhu `[一作]` (Stanford University), Corey J. Keller `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究使用Gemma-3-27B-PT模型的内部激活来直接提取抑郁症患者自然语言中的症状分量。

**💡 创新点**

创新点在于通过机制解释技术在残差流层21上构建分离的症状向量，并结合去相关的伪逆投影，实现临床医生可解释的症状评分，而非仅输出诊断标签。

**🔧 技术方法**

技术方法包括PERMANOVA与PERMDISP的分离度检验、语义投影与Gram矩阵伪逆校正、以及情感门控的抑郁向量计算。

**📊 数据集**

使用的数据集为六种临床评估工具提取的核心临床语料、三种来源（书籍、手册、Reddit）的自然语言语料，以及正向情感语料和HappyDB。

**📈 对比分析**

在核心语料与自然语言语料上，投影系数与临床注释保持相同的排序，抑郁向量在层21上对抑郁与非抑郁文本的AUC分别为1.0（内部）和0.789（外部）。

**⚠️ 局限性**

局限性包括症状被聚合为三组、仅评估单一模型、文本预处理可能引入偏差，以及缺乏与诊断、治疗反应或纵向数据的临床验证。

---

## 291. CoViT: Instance-Correspondence Contrastive Learning for Vision Transformer

**arXiv ID:** 2609.01787 | [PDF](https://arxiv.org/pdf/2609.01787v1)

**作者:** Yisen Wang `[一作]` (Nanjing University), Limin Wang `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了CoViT，一种通过几何引导的对比学习将实例感知能力注入预训练Vision Transformer的自监督框架。

**💡 创新点**

创新点在于利用自注意力生成实例掩码，挖掘硬正负样本构造三元组对比损失，直接在ViT中实现实例区分，无需额外解码器或标注。

**🔧 技术方法**

采用自注意力阈值化与形态学闭运算生成掩码、对比损失（triplet+InfoNCE）、硬负样本挖掘以及自适应阈值机制。

**📊 数据集**

在COCO 2017和HICO-DET等实例检测/分割数据集上进行实验验证。

**📈 对比分析**

与ViTDet、ViT-Adapter等基线在COCO和HICO-DET上对比，平均提升1.6–2.9 AP点，尤其在实例分割与稀有类HOI上显著提升。

**⚠️ 局限性**

局限性在于对遮挡、重叠场景的鲁棒性有限，依赖注意力掩码质量，且未实现端到端自监督预训练。

---

## 292. RT-HiSS: Ray Tracing Accelerated High Dimensional Vector Similarity Searches

**arXiv ID:** 2609.01975 | [PDF](https://arxiv.org/pdf/2609.01975v1)

**作者:** Revanth Reddy Munugala `[一作]` (Northern Arizona University), Michael Gowanlock `[通讯]` (Northern Arizona University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研发了一种基于RT核心加CUDA核心的高维向量相似性搜索算法RT-HiSS。

**💡 创新点**

关键创新是将高维数据映射到三维索引以利用RT核心，并通过分组邻域、两遍射线追踪估计结果上限、共享内存分块、结果位掩码压缩等技术实现高效批处理。

**🔧 技术方法**

采用RT核心构建BVH、两遍射线追踪、CUDA核心细化、共享内存分块、结果位掩码压缩、Pinned内存传输等技术。

**📊 数据集**

评测使用六个真实世界高维数据集（维度18~128，点数500万至1150万）和三种搜索半径。

**📈 对比分析**

与六种SOTA GPU相似性搜索算法及RTX 5000低维RT算法对比，RT-HiSS在大多数场景下速度提升1–4k倍（最大2,368×），在低维场景提升约1.2–1.9×。

**⚠️ 局限性**

受限于RT核心只能处理三维索引，需先重排/分组，且在低维或工作量不足时可能导致资源浪费，当前实现单GPU，扩展多GPU仍需进一步探索。

---

## 293. CAPTURE: Disentangling Preference Drift from Memory Poisoning in Personalized LLM Agents

**arXiv ID:** 2609.02265 | [PDF](https://arxiv.org/pdf/2609.02265v1)

**作者:** S M Asif Hossain `[一作]`, Md Kishor Morol `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了一种兼顾个性化和防御的持续记忆语言代理，提出了真实性门控、分层账本和安全约束的完整框架。

**💡 创新点**

将真实性视为隐变量，使用连续时间信念更新、时间尺度分层账本和澄清询问来同时提升适应性和抵御记忆污染，证明理论误差下界并证明条件下显著改进。

**🔧 技术方法**

连续时间ODE门控、神经微分方程、时序账本、澄清查询、基于风险的选择器、因果审计、LoRA适配器以及对齐奖励模型。

**📊 数据集**

自制的2,400个长期对话样本（包含四种真实漂移、善意噪声、三类污染及安全冲突），HorizonBench、40名用户的实验记录，以及外部红队生成的攻击。

**📈 对比分析**

与纯无个性化、递归记忆、状态内存、仅基于来源过滤、默认MemGPT、以及训练的Transformer等基线对比。实验表明，提出的方法在测试集上实现71.5%胜率，11.5%污染率、83.5%真实更新遵循率，优于大多数基线并在大多数场景中取得最高综合性能。

**⚠️ 局限性**

适应性攻击仍能提升成功率；当攻击者利用模型参数或生成器特性时，安全优势可被削弱；方法在大规模多模态或无监督情境下的鲁棒性尚未验证。

---

## 294. Polynomial Invariants for Probabilistic Transition Systems with Unbounded Support

**arXiv ID:** 2609.02446 | [PDF](https://arxiv.org/pdf/2609.02446v1)

**作者:** Anne Schreuder `[一作]` (Aalto University), C. -H. Luke Ong `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对概率性转移系统（PTS），利用马尔可夫过程和可选停顿定理（OST），提出了新的可整合性前置条件（dui），并基于此自动合成多项式不变量，进而求解循环终止后变量的高阶矩上界。

**💡 创新点**

①允许采样分布具有无界支持；②提出无需常数运行时上界的dui前置条件；③将该条件转化为对线性循环可检验的期望幂条件，实现在无界分布下的多项式不变量合成。

**🔧 技术方法**

马尔可夫过程、可选停顿定理、积分占优函数、Jordan 正规化、线性方程组求解、符号传播规则与Cauchy-Schwarz不等式结合的自动推导。

**📊 数据集**

在自建的线性循环程序集上评估，包含均匀、正态等无界分布的随机采样；具体实现基于现有的概率循环分析工具。

**📈 对比分析**

与现有只能处理有界分布或需显式上界的工具对比；对无界分布案例能产生符号上界和下界，结果与手工推导一致，计算时间在可接受范围内，展示了在无界分布下的可行性。

**⚠️ 局限性**

仅适用于单循环线性程序；需要手动证明停止时间的有限矩；不保证完整性；对非线性或多分支循环尚未实现。

---

## 295. Examining the Vulnerability of Multi-Agent Medical Systems to Human Interventions for Clinical Reasoning

**arXiv ID:** 2609.02191 | [PDF](https://arxiv.org/pdf/2609.02191v1)

**作者:** Benjamin C Liu `[一作]` (Stanford University), Kevin Zhu `[通讯]` (University Of California Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过构建五代理（患者、医生、专家、测量、Priming）框架，对多代理医疗系统中的对话时刻进行诊断漂移分析，定位所谓的“fault point”，并在这些时刻引入人类干预（正向、错误、带推理）来检验其对诊断准确率的影响；同时研究认知偏差注入在故障点的效应。

**💡 创新点**

创新点在于：①提出并系统化地定义与定位诊断最脆弱的对话时刻（fault point）；②在这些关键节点引入可控人类干预，以验证干预对诊断可靠性的提升或破坏；③将认知偏差注入与多代理推理结合，探讨其对诊断过程的具体影响。

**🔧 技术方法**

使用技术包括：多代理框架（GPT‑4.1 LLM为所有代理）、对话轮次记录与共享、余弦相似度计算诊断漂移、四种干预提示（正确/错误/含推理/不含推理）、认知偏差注入（确认偏误、提前闭合等）、指标评估（整体准确率、Top‑K准确率、测试请求数、代理间分歧数）。

**📊 数据集**

采用公开的 MedQA 数据集（214 例病例），将每例案例拆解为患者信息、医生目标、测试结果等字段，用于生成各代理的提示。

**📈 对比分析**

通过与无干预基线对比，使用整体诊断准确率、Top‑1/3/5准确率、测试请求数、代理间分歧数等指标进行评估。结果显示，正确干预可将 Top‑1 准确率从 58% 提升至约 60%（最多可达 76%），Top‑3/5 接近 70%+；错误干预将 Top‑1 降至 48%；认知偏差注入对 Top‑1 影响有限，整体 Top‑3/5 维持在 70–74% 之间。

**⚠️ 局限性**

局限性包括：①假设代理间通信无误，未考虑信息丢失、截断或语义漂移；②所有代理均基于同一 GPT‑4.1 LLM，缺乏专业多样性与细分领域知识；③LLM 未经过医学专业微调，诊断深度受限；④MedQA 题目为结构化 OSCE 风格，缺乏真实患者对话的自然性；⑤实验仅在云 API 环境下进行，未测试在本地或边缘设备上的可行性。

---

## 296. Bridging the Gap: A Longitudinal Analysis of Extended Identifiers in the Post-Cookie Era

**arXiv ID:** 2609.02069 | [PDF](https://arxiv.org/pdf/2609.02069v1)

**作者:** Michael Smith `[一作]` (Indiana University Bloomington), Yi Chen `[通讯]` (New Jersey Institute of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过对HTTP Archive 41个月的Header Bidding请求进行系统分析，评估EID在Web中的普及程度与传播行为。

**💡 创新点**

首次量化EID在实际网页访问中的采用率，并揭示其实现中存在的准确性与隐私缺陷。

**🔧 技术方法**

使用Python脚本解析Prebid标准化的Header Bidding请求，并统计EID出现率、持久性与多重ID情况。

**📊 数据集**

HTTP Archive公开数据集（约145M HTTP请求、616k网站）以及六个主流SSP（Magnite、Yieldmo、Sonobi、Amazon Ads、Index Exchange、GumGum）的Header Bidding端点。

**📈 对比分析**

对比不同时间段和网站流量分层的EID出现率，发现自2022年起EID采用率从28%急剧提升至84%；对比结果显示EID仍存在多重ID和持久性等问题。

**⚠️ 局限性**

仅基于自动化爬虫且未考虑真实用户交互、仅限六个SSP的Header Bidding请求、缺少对服务器端请求及不同浏览器隐私限制的评估。

---

## 297. Integrated Laser Scanning and Image-Based Topology Optimization Techniques for Detection and Quantification of Visible and Subsurface Structural Defects

**arXiv ID:** 2609.01808 | [PDF](https://arxiv.org/pdf/2609.01808v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 298. ClaimReceipt: Verifying Evidence Sufficiency and Coverage in Agent Evaluations

**arXiv ID:** 2609.01992 | [PDF](https://arxiv.org/pdf/2609.01992v1)

**作者:** Peiying Zhu `[一作]` (Blossom AI), Sidi Chang `[通讯]` (Blossom AI Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 ClaimReceipt 规范与可选择验证器，用于在代理评估中验证声明的充分性与覆盖性，并在历史与前瞻实验中实现自动化审核。

**💡 创新点**

创新点在于：①将证明充分性与覆盖性拆分为四层证据，提供可签名实验清单和事务收据；②设计了基于声明的可选择验证流程（CR‑2 与 CR‑3），能够在缺失终端记录或私有信息时返回“不可核实”；③通过冻结规范并预注册哈希实现了可复现的、可验证的审计框架；④引入字段到声明的依赖矩阵，验证字段最小化与无冗余。

**🔧 技术方法**

主要技术包括：签名链（Ed25519）、加密开箱（X25519+HMAC‑SHA256）、OpenTimestamps/Bitcoin 时间戳、精确的声明重放与分层验证、以及基于协议指纹的事务追踪。

**📊 数据集**

数据集：使用 1,392 条历史买卖代理交互记录（包含 600 条确定性与 792 条 LLM 相关记录），以及 30 条前瞻实验（E7），涉及 Qwen2.5‑Instruct 1.5B/3B/14B 模型。

**📈 对比分析**

比较方法：对历史记录与手工标注的 5 个审计单元做一致性测试；对所有声明进行完整重放与字段消融实验；对前瞻实验进行覆盖性、缺失终端、私有信息缺失的分层检验；性能表现为每个事务平均 0.02% 的推理开销，存储 9.9 KB，完整重放几乎无延迟。

**⚠️ 局限性**

局限性：规范仍未完全可读，需进一步澄清；只能验证声明内的有限声明类；私有信息需要外部治理与钥匙托管，无法完全消除 L4 风险；只在单一代理实验框架与模型族上验证，缺乏跨域或多模型的泛化验证。

---

## 299. Refining Heuristic-Based Bitcoin Address Clustering with Graph Neural Networks

**arXiv ID:** 2609.01942 | [PDF](https://arxiv.org/pdf/2609.01942v1)

**作者:** Hugo Schnoering `[一作]` (École Polytechnique, Institut Polytechnique de Paris), Michalis Vazirgiannis `[通讯]` (École Polytechnique, Institut Polytechnique de Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于对比学习的图神经网络嵌入方法，用来细化传统启发式比特币地址聚类，并通过层次聚类生成可解释的多分辨率层级结构，自动标记潜在的错误合并。

**💡 创新点**

创新点在于①公开了规模可达数十万地址的大型比特币交易图数据集；②将启发式标签与对比学习嵌入相结合，并给出理论可分离的条件；③通过层次聚类提供多分辨率视图，并给出量化标准识别可疑合并。

**🔧 技术方法**

使用对比信息熵损失（InfoNCE）的图神经网络（GCN、GraphSAGE、GAT），结合位置编码、边特征、硬负样本（CoinJoin）惩罚，以及平均/Ward/完全连结的层次聚类。

**📊 数据集**

使用从比特币区块链抽样得到的交易图，包含数十万地址；公开的约 100k 地址实体标签；以及 500 例 CoinJoin 交易样本；数据集已托管于 Zenodo。

**📈 对比分析**

通过与传统启发式、无监督基线（GAE、DGI）以及先前的启发式过滤方法对比，采用 NMI、ARI、树纯度、平衡准确率和 F1 等指标。结果显示对比学习 + 层次聚类在 NMI/ARI 上提升约10%~15%，在实体标签评估中宏观 F1 提升至 65.7%，误合并率显著下降。

**⚠️ 局限性**

主要局限在于真实标签稀缺、对动态增量图的可扩展性不足，以及对异质结构（如 CoinJoin）仍需特殊硬负样本；层次聚类在大规模图上的计算成本高。

---

## 300. FlashKAN: B-Spline KANs via Truncated Power Form

**arXiv ID:** 2609.01956 | [PDF](https://arxiv.org/pdf/2609.01956v1)

**作者:** Naveen Mysore `[一作]` `[通讯]`, Naveen Mysore

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

实现了FlashKAN，一个通过截断幂形式取代Cox‑de Boor递归计算B‑spline基函数的KAN实现，并将所有操作融合成单一GPU核，提供了一个可直接替换现有KAN层的开源包。

**💡 创新点**

创新点包括：① 在GPU上将B‑spline基函数的截断幂表达式与所有元素级运算一次性融合，完全消除递归、跨度查找和散射‑收集操作；② 采用“有界坐标稳定化”——将归一化输入裁剪到基函数支持区间，从而消除数值消除问题；③ 提供可直接使用的、与标准KAN层API兼容的开源实现。

**🔧 技术方法**

使用的技术包括：截断幂（truncated‑power）基函数形式、单核GPU融合实现、坐标裁剪（clamp）以保证数值稳定、统一节点向量（uniform knot vector）以及B‑spline到多通道MLP的线性变换理论。

**📊 数据集**

论文未在特定数据集上进行实验，主要通过分析和基准测试（如CUDA profiling）评估计算时间消耗。

**📈 对比分析**

与传统基于Cox‑de Boor递归的KAN层相比，FlashKAN将基函数计算时间从占前向传播时间的91%大幅降低；单核融合实现显著减少内存访问开销，整体推理/训练速度得到显著提升（速度提升约为递归次数，等价于三阶B‑spline的阶数）。

**⚠️ 局限性**

局限性：仅适用于均匀节点向量；在非均匀或自适应网格（可变跨度）下需要重新计算每段多项式系数，导致数据相关索引重新出现，无法保持一次性融合。

---

## 301. An Emerging NVM-Based On-Chip Training Architecture with Non-Ideality Mitigation Through Bipolar Weight Distributions

**arXiv ID:** 2609.01948 | [PDF](https://arxiv.org/pdf/2609.01948v1)

**作者:** Peng Dang `[一作]` (State Key Laboratory of Processors, Institute of Computing Technology, Chinese Academy of Sciences), Huawei Li `[通讯]` (State Key Laboratory of Processors, Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于FeFET的NOVA加速器，实现端到端的在芯片训练。

**💡 创新点**

创新点在于设计NAT算法，通过正则化将权重拉向eNVM的高低导电稳定区，以补偿非理想性。

**🔧 技术方法**

采用FeFET异质结器件、交叉阵列、可配置的数据流切换以及多维行为模型和量化感知训练。

**📊 数据集**

使用VGG6、VGG11、ResNet18在SVHN、CIFAR10、CIFAR100数据集上进行评估。

**📈 对比分析**

与传统方法相比，NAT在极端非理想条件下平均提升15.1%准确率，能耗比GPU高33.58倍。

**⚠️ 局限性**

主要限制是写操作需高电压/长脉冲导致延迟，且对极端噪声的抵抗仍有限。

---

## 302. CRISP: Cliff-awaRe Input-adaptive Sparse Prefilling with Structural-Mass-Motivated Routing

**arXiv ID:** 2609.01925 | [PDF](https://arxiv.org/pdf/2609.01925v1)

**作者:** Huu Huy Nguyen `[一作]`, Thien Huu Nguyen `[通讯]` (University of Oregon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为CRISP的动态稀疏注意力方案，针对长文本推理中前置填充阶段的二次复杂度瓶颈。

**💡 创新点**

创新点在于用结构化质量度量C_struct直接判断注意力集中度，替代JSD的间接计算，并引入基于噪声阈值的sink‑aware阈值避免累积阈值导致的“mass cliff”，从而消除O(n)背景噪声。

**🔧 技术方法**

采用结构化代理注意力测度C_struct、sink-aware阈值选择、并结合两种路由路径（Vertical‑Slash VS 与 Pooled‑Estimation PE）实现动态稀疏预填充。

**📊 数据集**

在Meta‑Llama‑3.1‑8B和Qwen2.5‑7B模型上，使用InfiniteBench、RULER和LongBench三个基准进行评估。

**📈 对比分析**

与FlashAttention（全密集注意力）、MInference（离线稀疏模式）以及FlexPrefill（现有动态稀疏方法）对比，CRISP在检索任务可提升高达+28个百分点，在综合准确度上与全密集相当，并在512k token下获得最高5.3×的推理速度提升。

**⚠️ 局限性**

局限性包括：仅适用于具有sink结构的架构，二元VS/PE路由可能不适用于混合特征头；未验证超过8B规模或更长上下文；仅针对前置填充阶段，生成阶段的动态稀疏未覆盖。

---

## 303. Learning Evidence Sufficiency Boundaries for Selective Answering in Grounded Multi-Hop QA

**arXiv ID:** 2609.01687 | [PDF](https://arxiv.org/pdf/2609.01687v1)

**作者:** Haruto Sato `[一作]` (Independent Researcher), Mei Ito `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于证据充分性边界的 grounded selective answering 训练框架，使模型在多跳 QA 中只在充分证据出现时给出答案，并在后续冗余证据下保持答案稳定。

**💡 创新点**

创新点在于：1) 将多跳 QA 样本转换为四层有序证据链；2) 通过生成原生的 loss（包括层级监督、边界翻转边距、后边界稳定性和召回保护）直接学习从拒绝到回答的转折点；3) 引入边界 flip 准确率等专门评估指标。

**🔧 技术方法**

技术包括：生成式微调、LoRA 参数适配、logit 比较式答案与拒绝分数、基于 margin 的损失函数、并在 Qwen2.5-3B-Instruct 模型上进行 fine‑tune。

**📊 数据集**

使用了 HotpotQA、2WikiMultiHopQA、MuSiQue 三大多跳 QA 数据集，构造了 2,400 个问题族（每族 4 层证据，共 9,600 条链条）。

**📈 对比分析**

与答案仅训练、token‑level SEAL 和 R‑Tuning 的 question‑level 罢答基线对比，ESBT 在边界翻转准确率、pre‑boundary 拒绝率、post‑boundary 稳定性方面均优于基线，并在外部非答题集上的不支持答案率最低（0.095），但在 raw QA F1 与 post‑boundary 稳定性上略逊于 token‑level SEAL。

**⚠️ 局限性**

局限包括：仅在 Qwen2.5‑3B 规模模型上验证，缺乏对更大模型的泛化；对 post‑boundary 稳定性和 raw QA F1 的提升不足；评估主要采用自动判定，未覆盖人工验证；方法在冗余证据下的答案置信度下降仍需进一步改进。

---

## 304. Selective Knowledge Edit Reversal via Gated Singular Vector Shrinkage

**arXiv ID:** 2609.02091 | [PDF](https://arxiv.org/pdf/2609.02091v1)

**作者:** Weifeng Jiang `[一作]` (Nanyang Technological University), Kwok-Yan Lam `[通讯]` (Nanyang Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对参数修改型知识编辑的选择性逆转框架，能够仅撤销指定的编辑事实而保留其他已编辑知识

**💡 创新点**

基于假设不同编辑在编辑矩阵的主奇异子空间中稀疏编码，并通过可学习的门控对奇异向量进行缩减来定位并消除目标编辑的影响

**🔧 技术方法**

使用奇异值分解（SVD）对编辑权重进行分解，学习entry‑wise门控并在主奇异子空间进行归一化缩减；优化目标包含参考模型的KL损失和输入扰动惩罚，结合数据增强

**📊 数据集**

在ZsRE和CounterFact两大事实编辑基准上进行评估，并在SST、MRPC、MMLU、RTE、CoLA、NLI等下游NLP任务上检验模型通用性能

**📈 对比分析**

与编辑后模型、全局逆转参考模型、以及“重新编辑”基线比较；结果显示在Reversed集合上达到与参考模型相近的agreement和KL，且在Remaining集合上保持高agreement，证明可选择性逆转有效且对剩余编辑影响小

**⚠️ 局限性**

逆转过程依赖全局逆转作为参考，若全局逆转质量不足会限制效果；假设编辑已被准确检测，检测误差未在实验中考量，且在多编辑并行逆转时可能出现相互干扰

---

## 305. Beauty is in the AI of the beholder: MLLMs systematically overrate facial attractiveness

**arXiv ID:** 2609.02512 | [PDF](https://arxiv.org/pdf/2609.02512v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 306. OmegaUse-SOP: SOP Engineering for Professional Computer Use from Human Demonstrations

**arXiv ID:** 2609.02149 | [PDF](https://arxiv.org/pdf/2609.02149v1)

**作者:** Yixiong Xiao `[一作]` (Baidu, Inc.), Hua Wu `[通讯]` (Baidu, Inc.)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出OmegaUse-SOP系统，通过人机演示转化为可复用的标准操作程序（SOP）技能，提升GUI代理在专业工作流中的执行可靠性。

**💡 创新点**

引入SOP工程方法，将演示转化为可编辑、可参数化的步骤；四模块架构（Observe、Reason、Configure、Execute）实现语义化、领域规则嵌入和逐步验证；强调人机迭代与可复用性。

**🔧 技术方法**

使用多模态视觉语言模型（OmniParser、PaddleOCR、Qwen3‑VL、GPT‑5.5、Opus‑4.7）记录和解析GUI轨迹；基于截图的目标定位、逐步信息检索、结果验证和人机干预机制。

**📊 数据集**

实验使用光伏模拟软件PVsyst 7.2的五个代表性专业SOP任务的演示轨迹，未使用公开大规模SOP数据集，而是客户真实工作流程。

**📈 对比分析**

将OmegaUse‑SOP与基线“仅从用户指令直接执行”对比：三种模型在无SOP时完成率分别为1/5、3/5、2/5；使用OmegaUse‑SOP后均达到5/5；消融实验显示缺失Reason模块时完成率降至2/5，证明其关键作用。

**⚠️ 局限性**

依赖人工演示和手动配置领域知识，缺乏自动化提取；对更大规模、多软件环境的可迁移性与扩展性尚未验证；人机交互成本高，模型对视觉差异的鲁棒性仍有限。

---

## 307. Learning the Constitutive Behavior of Materials via Neural Operators and Causal Attention: Case Studies in Plasticity and Damage

**arXiv ID:** 2609.02194 | [PDF](https://arxiv.org/pdf/2609.02194v1)

**作者:** Rishabh Arora `[一作]` (RWTH Aachen University), Shahed Rezaei `[通讯]` (ACCESS e.V.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种序列到序列的物质算子（Material Operator），可一次性将完整应变历程映射到对应应力历程，解决传统递归/滑动窗口模型对采样分辨率的依赖问题。

**💡 创新点**

核心创新包括：①使用傅里叶神经算子（FNO）实现对时间轴的谱卷积，保证离散化不变性；②引入正弦激活网络（SIREN）精准捕捉塑性与损伤的尖锐跃迁；③在算子内部嵌入严格因果自注意力机制，适配历史依赖并保持并行计算。

**🔧 技术方法**

技术实现基于Python+PyTorch，包含：傅里叶变换卷积、SIREN激活、层归一化、残差连接、带掩码的多头自注意力、dropout正则化以及AdamW优化器。

**📊 数据集**

数据集为三类仿真生成的拉伸路径：一维非线性硬化弹塑性、耦合损伤-塑性和二维J₂弹塑性。每类使用数千条高斯过程（GP）采样路径作为训练集，另外设定多种外部测试分布（GP、Zig‑zag、Sinusoidal）。

**📈 对比分析**

与传统FFNN滑窗（W=1/5/10）和RNN‑GRU进行对比；在训练分辨率N=50下，FNO‑SIREN‑Attention的相对L₂误差约为0.75%，而FFNN和RNN误差低于1%；在N=1000等高分辨率下，FNO‑SIREN‑Attention仅提升至约2%，而FFNN和RNN误差暴涨至10–13%。推断速度上FNO‑SIREN‑Attention每条路径≈2.6 ms，远快于FFNN（≈13–19 ms）与RNN（≈18–19 ms），训练成本最高但一次性可接受。

**⚠️ 局限性**

局限性：仅针对速率无关的弹塑性与损伤材料，未处理显式时间相关的粘弹性/粘塑性；算子缺乏物理约束（如热力学可行性）需后续加强；以及在极大尺度多物理耦合问题中对内存与计算开销的进一步评估仍待开展。

---

## 308. A Power Law in Logarithm's Clothing: On the Scalability of Graph-Based Vector Search

**arXiv ID:** 2609.02143 | [PDF](https://arxiv.org/pdf/2609.02143v1)

**作者:** Sajad Faghfoor Maghrebi `[一作]` (University of Toronto), Niv Dayan `[通讯]` (University of Toronto)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在多种实际图索引（HNSW、Vamana）上系统地扩展数据规模，测量并验证固定召回率下查询成本随数据量的增长规律；基于对邻域内点分布与内在维数的理论分析，构建了统一的“稀疏/稠密”两阶段可解释性模型，并进一步给出了可预测指数与插入成本的经验公式。

**💡 创新点**

创新点在于：①首次跨尺寸实证检验了“多项式对数”常识，发现大多数场景下成本遵循亚线性幂律（c<1）并在极大规模才转为子多项式；②提出了内在维数随样本量增大而增长的机制，解释了幂律与对数两阶段的根本原因；③给出了可供工程实践的成本预测模型，帮助在规模扩张时合理调节索引参数。

**🔧 技术方法**

核心技术包括：基于 beam‑search 的查询与插入成本建模；局部均匀性假设与局部内在维数（LID）估计；对稀疏（d_int≈Θ(log N)）与稠密（d_int=o(log N)）两 regime 的数学分析，证明了亚线性幂律和子多项式上限；对不同索引参数（M、ef_construction、ef_search）进行线性回归并提取系数。

**📊 数据集**

使用八个公开/自研向量集合：SIFT、DEEP、SpaceV、Wiki、GloVe、Rand64、GIST、OpenAI‑ArXiv，规模从数十万到十亿不等，覆盖图像、文本与随机向量等多种类型。

**📈 对比分析**

与传统只在单一规模评测的基准不同，本文在保持相同召回阈值下对多种 N 做曲线拟合，证明成本随 N 服从 dc∝N^c 形式。实验显示 c 受召回、ef_construction、M 影响，可通过经验模型预测；插入成本同样满足子多项式形式。总体上，模型解释了约 88% 的方差，且在极大规模下成本趋向 poly‑log。

**⚠️ 局限性**

局限性包括：理论推导假设欧氏度量、查询与数据同分布、局部均匀性；实际生产索引使用启发式构造，理论模型对其精细行为不一定完全适用；稠密 regime 的转折点在多数数据集尚未被完全覆盖，需更大规模验证；模型未考虑缓存/硬件差异和实际 I/O 代价。

---

## 309. Domain shift-robust object detection with GenAI image editing

**arXiv ID:** 2609.02299 | [PDF](https://arxiv.org/pdf/2609.02299v1)

**作者:** Isabel D. Stein `[一作]` (TNO Defence Security and Safety), Friso G. Heslinga `[通讯]` (TNO Defence Security and Safety)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究在低样本环境下利用扩散模型对未伪装车辆图像进行伪装编辑，以提高对伪装车辆的检测鲁棒性。

**💡 创新点**

提出零射击编辑与LoRA微调相结合的伪装生成框架，并系统评估其对不同伪装类型（叶片、网格、多光谱）的检测效果，首次验证生成编辑能显著缩小域间差距。

**🔧 技术方法**

使用Qwen Image Edit 2509、Flux.2 Dev等扩散模型进行图像编辑；对Qwen进行LoRA微调；使用GroundingDINO检测器；进行自动注释、质量过滤与后处理。

**📊 数据集**

基准数据集为15类军用车辆的未伪装图像（360训练、150验证、432测试），并采集7类伪装车辆（叶片、网格、多光谱，共303张）。

**📈 对比分析**

对比零射击编辑、LoRA微调、Flux、黑条遮挡等四种增强方式，取25%伪装合成比例。结果显示零射击Qwen在伪装测试上mAP从56.8提升至69.9（+13.1），网格、叶片提升显著；LoRA在多光谱上进一步提升至76.5。

**⚠️ 局限性**

局限包括：提示工程缺乏系统化；仅在单一检测器上验证；伪装测试集样本量与类别不均衡；模型选择基于未伪装验证，可能导致过拟合。

---

## 310. Federated LoRA Adaptation of BiomedCLIP Across Four International Chest X-Ray Cohorts

**arXiv ID:** 2609.02101 | [PDF](https://arxiv.org/pdf/2609.02101v1)

**作者:** Sanjaya Poudel `[一作]` (North Carolina A&T State University), Sunil Kumar Gaire `[通讯]` (North Carolina A&T State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在四个跨洲胸部 X 光数据集上，研究了对 BiomedCLIP 的联邦 LoRA 适配，用于多标签分类任务。

**💡 创新点**

首次系统评估医学 VLM 的联邦参数高效微调，并引入基于 SVD 的 FlexLoRA 聚合，证明联邦 LoRA 能接近集中式训练性能。

**🔧 技术方法**

使用联邦学习（FedAvg、FedProx）、低秩适配（LoRA）、基于 SVD 的聚合（FlexLoRA）和 CLIP 对比损失进行训练。

**📊 数据集**

NIH ChestX‑ray14、CheXpert、VinDr‑CXR、PadChest 四个公开胸部 X 光数据集。

**📈 对比分析**

与单体训练、一次性聚合及 5 轮联邦训练对比；在共享 5 个通用诊断类别上，FedAvg 的宏观 AUC 为 0.802，超过单体基线 0.776，逼近集中式 0.812。

**⚠️ 局限性**

仅使用单一随机种子与单 GPU 实验，缺乏多种种子验证；VinDr 的图像级拆分可能导致患者泄露；未进行严格隐私保护或多轮聚合精细分析。

---

## 311. Accurate in space, unreliable in time: how LLMs represent national cultural change

**arXiv ID:** 2609.01902 | [PDF](https://arxiv.org/pdf/2609.01902v1)

**作者:** Yalda Daryani `[一作]` (University of Southern California), Madeleine I. G. Daepp `[通讯]` (Center for Democracy and Technology)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对四个 SOTA 大语言模型在 40 个国家、两十年间的文化价值轨迹进行评估，比较模型在最近波（snapshot）与不同时期（时间维度）下的表现。

**💡 创新点**

首次把文化视为时间轨迹而非单一快照，系统评估 LLM 是否能把握各国文化的变化速度、方向和逆转等动态特征。

**🔧 技术方法**

采用问卷式概率分布 elicitation（verbalized distribution）并映射到 Inglehart‑Welzel 两维坐标，计算欧氏距离、向量余弦、净位移、步骤对齐、时间信号，并利用非参数自助法和置换检验进行统计对比。

**📊 数据集**

使用 World Values Survey（WVS）4–7 波（1999–2023）的微观数据和四个 SOTA LLM 的模型输出。

**📈 对比分析**

在 snapshot 对齐中，模型与最近波的平均距离为 0.15–0.24 标准化单位，压缩程度有限；在时间对齐中，模型仅捕捉 45–68% 的真实位移，只有 Gemini 在步骤顺序上表现出显著的时间信号；在逆转检测中，成功率极低，整体不如真实轨迹。

**⚠️ 局限性**

局限在于仅用两维 Inglehart‑Welzel 简化文化，WVS 为不连续的交叉截面，模型可能隐含历史信息但未能通过简单时间提示检索，且样本仅涵盖 40 国且覆盖不均。

---

## 312. Adversarial Vulnerabilities of Neural Biomarker Identification Systems

**arXiv ID:** 2609.01856 | [PDF](https://arxiv.org/pdf/2609.01856v1)

**作者:** Polina Tapal `[一作]` (Cerberus Neurosecurity Research Institute), Bryce-Allen Bagley `[通讯]` (Cerberus Neurosecurity Research Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了基于EEG的生物特征认证系统在黑盒攻击下的脆弱性，构建并评估了三种自适应攻击器（多臂赌博机、VAE+REINFORCE、Gray‑Box GAN）以及多种信号级攻击族，并与基线分类器（最近质心、朴素贝叶斯、线性SVM）进行对比。

**💡 创新点**

创新点在于：①将攻击族选择建模为多臂赌博机，能从仅接收接受/拒绝反馈中自动发现最有效的攻击方向；②提出多维信号级攻击族覆盖噪声、时序扰动、频谱增益、协方差插值、重放+抖动、通道耦合、特征谱扭曲等；③在六个公开EEG数据集上系统性评估不同录制范式（视觉诱发、运动想象、静息）下的防御强度，揭示静息状态本身不可用作生物识别。

**🔧 技术方法**

技术主要包括：Log‑Euclidean 变换得到SPD协方差特征；最近质心和Gaussian Naïve Bayes 近似判别；多臂赌博机（UCB）策略；VAE 及 GAN 生成对抗攻击；Mann‑Whitney U 检验、FAR/FAR_zero 统计等。

**📊 数据集**

使用六个公开EEG数据集：Zhang（RSVP）、Won（RSVP）、Wang‑EO、Wang‑EC、COG‑BCI‑EO、COG‑BCI‑EC（四个静息），Cho、Lee（运动想象）。

**📈 对比分析**

评估方法为在不同阈值百分位（p∈{5,10,15,20,25}）下计算基线EER、FAR、攻击提升；对三类攻击器的FAR进行对比；对不同攻击族进行Mann‑Whitney U检验。结果显示：静息状态的零攻击FAR已高达0.8+，攻击提升几乎为0；RSVP在UCB下可提升FAR约0.03–0.05；运动想象几乎无可行攻击；不同攻击族在不同范式下表现差异显著，验证了范式对安全性的决定性影响。

**⚠️ 局限性**

局限性包括：①实验仅限于已公开数据，未覆盖更复杂的录制条件或多通道硬件差异；②攻击模型假设攻击者仅有黑盒访问，未考虑更强的白盒或特征提取层访问；③仅评估了几种分类器，缺乏对更复杂深度模型的探测；④未实现攻击检测机制，仅通过实验揭示脆弱性。

---

## 313. Similarity-Aware Personalized Federated Learning in Heterogeneous Environments

**arXiv ID:** 2609.02241 | [PDF](https://arxiv.org/pdf/2609.02241v1)

**作者:** Arun Kumar A `[一作]`, Dat Phan Trong `[通讯]` (Deakin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Similarity-Aware PErsonalized Federated Learning (SAPE‑FL)，一种在联邦学习中同时使用全局模型和相似度加权的同伴模型进行个性化训练的框架。

**💡 创新点**

创新点在于双重锚定机制：客户端既与全局模型对齐，又与筛选出的相似同伴模型对齐；同时引入输出相似度与权重相似度的混合度量以及基于相似度的自适应正则化和服务器端的相似度加权聚合，显著降低负迁移并提升鲁棒性。

**🔧 技术方法**

使用的技术包括基于Cosine相似度的模型输出/权重相似度计算、动态正则化系数、相似度筛选阈值、相似度加权同伴平均、相似度加权聚合以及对算法收敛性进行理论证明。

**📊 数据集**

实验数据集涵盖合成数据、HAR（加速度计）、CIFAR‑10、CIFAR‑100、MNIST、FMNIST 和 EMNIST，采用 Dirichlet(0.3) 分布产生非 IID 数据，100 个客户端，200 轮通信。

**📈 对比分析**

与 FedAvg、PerFedAvg、FedProx、Ditto、FedACS、LCFed、FedAFK、FedAS 等基线对比，SAPE‑FL 在大多数任务中取得最高平均准确率（最高可达 90.83%）和 AUC，证明其在高度异构环境下优于现有方法。

**⚠️ 局限性**

主要局限包括：需要计算并维护所有客户端间的相似度，导致额外的计算和通信开销；对相似度阈值和影响因子 δ 的选择敏感，需经验调参；在极端通信受限或高噪声环境下的鲁棒性仍待进一步验证。

---

## 314. Do Cantonese-Adapted Language Models Better Predict Cantonese Reading? A Cross-Model Eye-Tracking Evaluation

**arXiv ID:** 2609.02163 | [PDF](https://arxiv.org/pdf/2609.02163v1)

**作者:** Ziqi Zhang `[一作]` (Hong Kong Polytechnic University), Mohammad Momenian `[通讯]` (University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `2704f255-0c84-4173-b83c-0e9a3dbea232` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在Cantonese阅读时，使用不同程度的Cantonese特定训练的语言模型是否能更好地预测眼动数据。

**💡 创新点**

对比轻量级（CKIP GPT‑2 Tiny + JED351）与大规模（Qwen2.5‑7B + CantoneseLLM‑7B）模型，在四种信息论指标（词汇惊讶、词性惊讶、前熵、熵减）下评估其预测性能，并揭示不同指标对模型预测力的异质性。

**🔧 技术方法**

采用自回归大型语言模型（GPT‑2、Qwen）、信息论指标提取、CatBoost回归、五折交叉验证、留一预测器块消融。

**📊 数据集**

使用MCFIX数据集的Cantonese自然阅读与任务特定阅读眼动记录（约10,000词级样本）。

**📈 对比分析**

在无LLM基线的基础上逐一加入每个指标及四指标联合模型，评估MAE、R²、相关系数等指标。结果显示：联合模型相较于基线平均降低约1–2% MAE；在大模型对中，CantoneseLLM‑7B在词汇惊讶和联合模型上优于Qwen2.5‑7B；而在熵减上，轻量级CKIP模型表现最好。

**⚠️ 局限性**

对比结果受模型规模、训练策略差异混杂影响；不同标记器导致熵值不可直接跨模型比较；仅针对单一文学译本与受试者样本，未能完全分离Cantonese训练效果。

---

## 315. Contact-Constrained Lower-Limb Joint-Offset Calibration for Humanoid Robots

**arXiv ID:** 2609.02306 | [PDF](https://arxiv.org/pdf/2609.02306v1)

**作者:** Kaixiang Lu `[一作]`, Chuang Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在没有外部可视化系统的前提下，利用双足固定接触、底盘IMU的平地约束和机械先验，构建了自包含的下肢关节偏移校准框架。

**💡 创新点**

创新点包括：对并行pitch轴导致的观测耦合进行Hessian特征分析；阐明关节顺序对耦合结构与可观测性的影响；通过几何姿态多样化和机械先验实现对弱方向的分离与补偿。

**🔧 技术方法**

使用的技术包括：非线性最小二乘优化（Ceres）、基于SE(3)的相互变换一致性约束、IMU辅助的平地残差、Tikhonov正则化与膝关节先验、MuJoCo仿真、LiDAR‑惯性外部验证。

**📊 数据集**

数据集涵盖：A3仿真中注入偏移的MuJoCo双足姿态袋；A2与A3真实机器人在四个静态姿态（蹲、左/右弓步、宽蹲）下的编码器与IMU采样；以及用于外部验证的LiDAR‑惯性轨迹和独立的光学运动捕捉数据。

**📈 对比分析**

与Yamane等人提出的sole‑height方法对比，A3下足高度残差从4.26 mm降至2.20 mm，A2从8.03 mm降至1.43 mm；在LiDAR验证中，垂直方向RMS误差下降25.5%；闭环仿真中，步态跟踪误差显著下降，验证了校准对姿态一致性的提升。

**⚠️ 局限性**

局限性在于：假设双足接触固定且几乎共面；对pitch轴的弱方向仍依赖先验，无法完全从数据中恢复独立偏移；未提供绝对外部零点的验证；仅在静态姿态下评估，未检验对动态步态的长期鲁棒性。

---

## 316. Architecting Conversational Data Systems for Stateless LLM APIs: The Hydration Proxy Pattern

**arXiv ID:** 2609.01834 | [PDF](https://arxiv.org/pdf/2609.01834v1)

**作者:** Joseph Axisa `[一作]` `[通讯]` (Google Cloud), Joseph Axisa (Google Cloud)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出 Hydration Proxy Pattern，即三层架构（前端调度层、加密安全网关和混合持久层），解决 Stateless LLM 接口在企业分析平台中的会话状态与语义记忆管理问题。

**💡 创新点**

创新点包括：① 将会话持久化与推理引擎解耦，保证平台对对话数据的主权；② 引入 Context Stabilization Mandate 与 Linear Context Strategy，优化 KV 缓存利用；③ 通过递归摘要与可移除推理实现动态上下文压缩，保持高吞吐与低延迟。

**🔧 技术方法**

使用技术包括：stateless LLM API（如 OpenAI Chat Completions）、安全网关注入凭证与元数据、关系型数据库作为元数据账本、加密对象存储存放对话内容、KV 缓存与递归压缩算法、流式解析（stream parsing）分离模型思考与答案。

**📊 数据集**

论文未指定具体实验数据集，主要通过系统架构设计与对比分析展示其优点；若做实验，可能使用典型企业分析工作负载（如 SQL 查询日志、可视化元数据）进行模拟。

**📈 对比分析**

通过与 Naive Wrapper、Centrally‑Managed Session、Ecosystem‑Integrated Agent 等现有架构进行对比，强调在会话完整性、数据主权、缓存命中率、成本（token 费用）和延迟方面的优势；但未给出数值性能指标，主要以理论与架构优势为依据。

**⚠️ 局限性**

局限性包括：① 需要在内部维护复杂的三层架构与持久化机制，部署成本高；② 递归摘要与可移除推理在高密度对话时仍可能导致缓存失效或瞬时延迟；③ 对安全网关与凭证管理要求严格，若实现不当可能成为单点故障或安全漏洞；④ 方案针对 stateless LLM API，若 LLM 提供商改为有状态接口，需重新评估架构适用性。

---

## 317. SEAL: Reinforcing Global Safety in Mixture-of-Experts through Shared Expert ALignment

**arXiv ID:** 2609.02293 | [PDF](https://arxiv.org/pdf/2609.02293v1)

**作者:** Qingyu Meng `[一作]` (Vrije Universiteit Amsterdam), Min Chen `[通讯]` (Vrije Universiteit Amsterdam)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对混合 MoE 语言模型的安全性，提出通过训练时仅调优共享专家的 LoRA 适配器（SEAL）及其加入正交约束的改进版本（SEAL++），以增强模型在提示注入、恶意微调和神经元剪枝攻击下的安全对齐。

**💡 创新点**

创新点在于识别共享专家为结构上无路由依赖的安全锚点，并利用低秩适配器与正交约束将安全表征固定在共享专家参数子空间，从而实现参数高效、可插拔且可与路由层防御协同的全新安全强化方案。

**🔧 技术方法**

使用的技术包括 Mixture‑of‑Experts 架构、共享专家模块、LoRA 低秩适配器、Direct Preference Optimization (DPO) 对安全偏好数据的对齐训练，以及基于安全神经元激活差分的正交约束。

**📊 数据集**

采用的数据集包括 PKU‑SafeRLHF（安全偏好对）、AdvBench 与 HarmBench（激活差分与攻击用）、WildJailbreak、Do‑Not‑Answer、JailbreakBench（抗注入评测），以及 GSM8K、MMLU、ARC‑Challenge、TruthfulQA、PopQA（能力评测）。

**📈 对比分析**

与原始模型和现有 MoE 安全防御相比，SEAL 在直接攻击下将 ASR 降低 60% 以上、在组合剪枝攻击下将 ASR 降到原模型的 1/5 左右，且能力平均损失不超过 1.4%，显著提升安全性而几乎不影响通用性能。

**⚠️ 局限性**

局限性包括对共享专家容量有限的模型（如 Qwen3.5）保护效果不如在共享专家安全比例高的模型，且该方法不适用于缺乏共享专家的 MoE 架构，无法抵御基于安全神经元阈值的专门剪枝策略。

---

## 318. The Dynamics of Continuous Mixture Collapse in Language Models

**arXiv ID:** 2609.02049 | [PDF](https://arxiv.org/pdf/2609.02049v1)

**作者:** Ali Backour `[一作]` `[通讯]` (Massachusetts Institute of Technology), Ali Backour (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了为什么预训练大型语言模型在使用连续混合状态进行推理时容易出现信息丢失和混合崩溃，并给出了导致崩溃的三种独立机制；

**💡 创新点**

创新点在于：①用理论与实验相结合的方法系统识别出模型架构、训练过程以及软最大化回馈循环三条崩溃路径；②推导了软最大化递归动力学的临界阈值，并证明其导致的放大与收缩两种失效模式；③将两种混合的分析推广到K‑way混合，并证明要保持混合需要至少K‑1个上下文相关的自由度；

**🔧 技术方法**

技术主要包括：对连续混合状态的几何保真度量（CALIB）；对混合状态在Transformer中线性/非线性传递的分析；递归软最大化动力学的数学模型（包括Curie–Weiss自洽方程）；以及对K‑way混合的Jacobian秩分析；

**📊 数据集**

实验使用了多种预训练模型（如Qwen3.5‑4B、Gemma‑4‑E4B‑it等）以及对应的随机初始化控制网络，并在由三元混合或二元混合构成的自定义词嵌入数据集上进行评估；

**📈 对比分析**

方法比较：将预训练模型与同构的随机初始化模型对比；计算CALIB得分并随层深和模型规模绘图；在递归软最大化实验中测量L_t和b_t，并用温度参数验证临界阈值预测；实验结果显示预训练模型的混合保持显著差于随机控制，且大多数模型处于放大区间，导致混合趋于极端；

**⚠️ 局限性**

局限性：仅用词嵌入混合来衡量连续状态保持，未覆盖所有连续推理形式；实验仅覆盖两大模型族，K‑way结论为理论推断，未在多分量实验中验证；未评估对最终推理性能的实际影响；对训练过程中导致失真的具体原因尚未明确；

---

## 319. Lightweight Adaptation of General-Purpose VLMs for Multispectral and SAR Image Understanding

**arXiv ID:** 2609.02187 | [PDF](https://arxiv.org/pdf/2609.02187v1)

**作者:** Shanji Liu `[一作]` (Zhejiang University), Chao Li `[通讯]` (Zhejiang Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

通过将 Sentinel‑2 多光谱和 Sentinel‑1 SAR 观测渲染为七张命名的三通道图像，并在 Qwen3‑VL 等通用 VLM 上使用 LoRA 微调，实现在遥感数据上的分类与生成任务。

**💡 创新点**

创新点在于：① 用多图渲染与视图命名将多光谱与 SAR 信息映射到通用 VLM 的多图接口；② 只微调少量 LoRA 参数即可让模型理解遥感感知；③ 通过保留证据的 DPO 进一步提升缺失标签的完整性，避免重建遥感专用基础模型。

**🔧 技术方法**

技术方案包括多图输入渲染、视图命名、LoRA 参数微调（语言网络与视觉 Transformer）、监督式微调（SFT）、保留证据的直接偏好优化（DPO）以及结构化输出与证据一致性评估。

**📊 数据集**

主要使用的遥感数据集有：BigEarthNet‑v2（六类土地覆被分类），Sen1Floods11（洪水验证），以及 BigEarthNet.txt（土地覆被字幕生成）。

**📈 对比分析**

通过与零样本 VLM、冻结模型探测器以及专业遥感模型对比，SFT 后 micro‑F1 从 0.5921 提升至 0.8242，DPO 后进一步提升至 0.8275，证据 audit 率从 0.1531 提升至 0.9750；在 Sen1Floods11 与 BigEarthNet.txt 上亦能显著提升 F1、BLEU 等指标，显示该轻量化适配在多种模型与任务上均有效。

**⚠️ 局限性**

局限性包括：① 需要人工设计视图与名称，无法直接利用原生多通道或高分辨率遥感数据；② 目前仅验证于六类平面分类和部分生成任务，尚未扩展到像素级或大规模多类情形；③ 对模型架构的适配仍有一定依赖，需进一步验证跨模型通用性。

---

## 320. Spectral Initialization and Scheduled Graph Smoothness for Uncertain Knowledge Graph Completion

**arXiv ID:** 2609.02519 | [PDF](https://arxiv.org/pdf/2609.02519v1)

**作者:** Md Abrar Jahin `[一作]` (University of Southern California), Craig A. Knoblock `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 QUEST，一种在不增加可训练参数的前提下改进不确定知识图谱（UKG）完成的方法。

**💡 创新点**

创新点在于利用置信度加权图拉普拉斯的最小非零特征向量进行实体嵌入初始化，并在训练早期加入无偏 Mini‑batch Dirichlet 能量正则化，同时在自监督伪标签阶段（PCDG）停止该正则化，以避免梯度冲突和训练不稳定。

**🔧 技术方法**

使用图谱拉普拉斯谱分解、Dirichlet 能量正则化、ssCDL 作为基线框架以及 MAML 风格的伪标签生成（PCDG）等技术。

**📊 数据集**

在两个公开 UKG 基准上进行评估：NL27k（来自 NELL）和 CN15k（来自 ConceptNet）。

**📈 对比分析**

与 UKGE、PASSLEAF、BEUrRE、UPGAT、UKGsE 以及 ssCDL 进行比较，QUEST 在 8 项指标中 6 项领先、2 项持平，并且消除了在密集图上出现的 epoch‑30 训练不稳定峰值，整体性能优于现有方法。

**⚠️ 局限性**

局限包括仅在两大基准上测试；谱初始化在极大图规模下计算成本较高；使用无向拉普拉斯忽略关系方向和类型，可能限制对有向、类型化图的适用性。

---

## 321. Task-Level Natural Language Priors as Learning Signals for Low-Resource LLM Training

**arXiv ID:** 2609.02244 | [PDF](https://arxiv.org/pdf/2609.02244v1)

**作者:** Jian Gao `[一作]` (Tsinghua University), Ji Wu `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Prior‑Guided Tuning（PGT）框架和Contrastive Prior Steering（CPS）方法，在低资源下将任务级自然语言先验作为辅助学习信号来训练大型语言模型。

**💡 创新点**

创新点在于将任务级先验从传统的输入上下文转化为训练目标中的对比学习信号，并通过正负先验共同引导模型学习正确的任务规则，实现更高的数据利用效率。

**🔧 技术方法**

采用对比先验蒋制的辅助损失（正负先验+原监督损失）与梯度裁剪相结合的细调技术，在LLaMA 3.1 8B和Qwen 2.5 7B模型上实施。

**📊 数据集**

实验数据集包括：AmbiMath（合成模糊推理基准）、Jigsaw毒性分类子集（身份相关标签）以及MNLI/HANS自然语言推理基准。

**📈 对比分析**

与普通微调、提示微调和DPO风格基线对比，CPS在AmbiMath上取得97.6% EM，在Jigsaw上平均Macro F1提升9.5个百分点，在HANS上非蕴涵准确率提升8.3/5.2个百分点，并在低资源比例下实现与全数据微调相当或更优的表现。

**⚠️ 局限性**

局限性包括：需要人工编写任务级先验，自动化先验获取与验证尚未解决；对先验措辞仍有一定敏感性；在更复杂或多任务场景中的适用性尚需进一步验证。

---

## 322. WiFlow: Estimating Optical Flow using WiFi Channel State Information

**arXiv ID:** 2609.02452 | [PDF](https://arxiv.org/pdf/2609.02452v1)

**作者:** Thomas Weigel `[一作]` (Technical University of Darmstadt), Simone Schaub-Meyer `[通讯]` (Technical University of Darmstadt)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

基于WiFi通道状态信息（CSI）直接估计室内场景的光流；

**💡 创新点**

①首次从CSI直接恢复光流；②构建含同步摄像机光流标签的CSI数据集；③提出三种不同复杂度的RAFT‑风格网络（流块、掩码块及其组合）来实现光流估计；

**🔧 技术方法**

使用CSI预处理（如抗相位偏移的quotient归一化）与深度网络：特征网络、上下文网络、GRU式迭代细化，结合RoIAlign实现局部流预测；

**📊 数据集**

新收集的室内固定布局CSI+同步摄像机视频数据集（含7人动作、空闲、边缘运动），从视频中生成伪光流作为监督；

**📈 对比分析**

对比零预测、不同预处理方式、三种网络结构，评价指标为端点误差EPE（整体、运动/静止区域、放大误差EPEA）。最佳模型为结合掩码与流块的WiFlow‑Mask，在时间拆分和主体拆分上均取得最低EPEM，且能跨人群泛化；

**⚠️ 局限性**

局限性：只能在训练的房间/环境中泛化，无法迁移到不同房间；伪光流标签与真实光流存在阴影等不一致；低分辨率、多人/遮挡场景下精度下降；

---

## 323. Thinking effort aligns between humans and reasoning models in abductive reasoning

**arXiv ID:** 2609.01867 | [PDF](https://arxiv.org/pdf/2609.01867v1)

**作者:** Henry Arthur `[一作]` `[通讯]` (University of Trento), Henry Arthur (University of Trento)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究大型推理模型（LRMs）与人类在归纳推理任务中的思考成本对齐，比较不同解码策略下链式思考长度与人类反应时间的相关性。

**💡 创新点**

创新点在于使用非贪婪（随机）解码并对多次运行进行平均化，以提升模型与人类思考成本的一致性，并验证链式思考长度可作为认知努力的有效代理。

**🔧 技术方法**

主要技术包括链式思考（CoT）输出、RLVR训练的大型推理模型、温度采样调参、token计数、偏相关分析、Bootstrap收敛检验和多次采样平均。

**📊 数据集**

使用的数据集为AllenAI的Abductive Reasoning Dataset（ART），共160道强难归纳选择题。

**📈 对比分析**

通过将模型生成的token计数与人类反应时间进行偏相关分析进行比较，发现非贪婪解码（推荐温度）下相关系数显著提升至最高0.55，且模型准确率与人类错误模式高度一致。

**⚠️ 局限性**

局限性包括仅测试单一归纳推理形式、模型使用默认推理力度未针对难度调节、可能存在训练集泄漏以及对提示敏感性和温度范围的依赖。

---

## 324. One Demonstration, Many Objects: Generalizing Manipulation via Local Contact Geometry

**arXiv ID:** 2609.01938 | [PDF](https://arxiv.org/pdf/2609.01938v1)

**作者:** Satvik Sharma `[一作]` (Stanford University), Jeannette Bohg `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本工作提出了一种从单一人类演示学习并能跨不同物体与机器人手臂实现仿真到实物的可携带的多指手操作框架

**💡 创新点**

创新点在于引入了基于接触点的奖励机制（表面对齐与持续接触）以及将高层RGB引导与低层深度闭环分离，显著提升了实物迁移性能

**🔧 技术方法**

主要技术包括残差强化学习、接触中心化奖励、扩散策略下的深度感知低层控制、高层RGB引导策略以及域随机化与视觉转移技术（Cosmos-Transfer-2.5、FoundationStereo）

**📊 数据集**

使用单一人类演示数据通过运动捕捉或手动采集得到，随后在仿真中做尺度与几何随机化生成训练样本；在真实实验中涉及16种不同材质、尺寸与重量的物体与4个任务

**📈 对比分析**

与改造后的HERMES*和DexMachina*基线对比，虽然在仿真中的表现略逊一筹，但在真实环境中实现了71%的平均成功率，并且在仿真到实物的性能下降幅度最小

**⚠️ 局限性**

局限性包括高层策略仅为单步开放式，无法实时适应全局变化；低层训练仅在有限工作空间内随机化；实验任务仅覆盖四类；同时需要手动获得物体轨迹参考

---

## 325. FairLens: Benchmarking Fairness in Vision-Language Models for High-Stakes Decision-Making

**arXiv ID:** 2609.01691 | [PDF](https://arxiv.org/pdf/2609.01691v1)

**作者:** Vahid Reza Khazaie `[一作]` (Vector Institute), Shaina Raza `[通讯]` (Vector Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 FairLens 基准与评估框架，用以测量视觉语言模型在招聘、法律和医疗等高风险领域的公平性与答案合理性。

**💡 创新点**

将公平性评估与答案合理性（soundness）结合，区分不平等结果与未受支持的推断，并引入四个评估视角（人口统计学公平性、合理性、关联度与开放式文本偏差）。

**🔧 技术方法**

利用多模态 VLM 进行闭合与开放式问答，采用 LLM‑as‑a‑judge 进行自由文本偏差检测，并使用四类指标评估模型表现。

**📊 数据集**

基于 UTKFace 人脸数据集（年龄 30–65 岁，按性别、种族、年龄分层）构建 1,505 张图像的评估集，配以 69 道高风险场景问答。

**📈 对比分析**

对八种主流 VLM（含开放源与专有）执行全部问答，使用四项指标对比，结果显示模型在合理性（soundness）上差异显著，少数模型如 Ovis2.5 表现最佳，而大多数模型在支持未受证据支持的推断方面失效。

**⚠️ 局限性**

受 UTKFace 标签粗糙（性别二元、种族宽泛）限制；评估基于图像而非真实决策场景；LLM‑judge 可能引入主观误差；未提供置信区间或显著性检验。

---

## 326. Decoding Decision Correctness from EEG Under High Cognitive Workload in Virtual Reality: Implications for Collaborative Brain-Computer Interface Teams

**arXiv ID:** 2609.02436 | [PDF](https://arxiv.org/pdf/2609.02436v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 327. Humanoid Safe Stop via Learned Stoppability Value

**arXiv ID:** 2609.02358 | [PDF](https://arxiv.org/pdf/2609.02358v1)

**作者:** Junfeng Long `[一作]` (University of California Berkeley), C. Karen Liu `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出Safe-Stop框架，学习任务无关的停止策略和两种可停估计器，在紧急停止时判断是否安全停止；

**💡 创新点**

创新点在于结合了停止概率估计器和基于Hamilton‑Jacobi的到达-避让估计器，并采用窗口化决策规则与臂关节遮蔽的观察空间实现任务无关的安全决策；

**🔧 技术方法**

使用深度强化学习（PPO）训练停止策略和估计器，结合Hamilton‑Jacobi到达-避让值、可观测状态投影以及窗口化阈值决策；

**📊 数据集**

利用Unitree G1仿真环境与大规模人类运动捕捉数据集BONES‑SEED（142,220帧）进行训练与评估；

**📈 对比分析**

与单一估计器或即时决策基线比较，Safe-Stop在OOD停止成功率96.4%、错误警报率3.89%（精度99.78%）的情况下实现了最优性能，且对不同行为策略具有良好迁移；

**⚠️ 局限性**

局限在于仅使用简单的阻尼下落策略，无法处理可能可恢复的瞬态状态，且对仿真与真实环境的误差仍有一定影响。

---

## 328. LLM-Driven Joint Evolution of Coupled Heuristics Components for Routing Optimization

**arXiv ID:** 2609.02353 | [PDF](https://arxiv.org/pdf/2609.02353v1)

**作者:** Juntao Wei `[一作]` (Shanghai Jiao Tong University), Shan Jiang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种LLM驱动的联合进化框架LLM-HCJG，能够在同一设计蓝图下同时生成并协同进化路由问题的解初始化与惩罚构造两个相互依赖的启发式组件，并将其嵌入到Guided Local Search（GLS）中，以解决TSP和CVRP。

**💡 创新点**

创新点：① 在算法层面使用LLM与进化计算相结合，实现整体算法的生成与优化；② 引入共享设计蓝图，使得初始化与惩罚构造在同一框架内协同进化，克服两者的非可分性；③ 在GLS框架中加入了端到端的在线搜索增强机制，提升搜索方向；④ 通过仅用5个训练实例即可实现跨实例迁移，展示了样本高效性。

**🔧 技术方法**

技术手段：LLM（GPT‑4o‑mini）用于生成代码与蓝图；种群进化（交叉、变异、选择、管理）对算法级别的个体进行搜索；GLS核心（边缘惩罚、局部搜索、邻域操作）作为搜索框架；理论分析证明初始化与惩罚构造的状态转移不可分离；使用相对误差和最佳/最优计数作为评估指标。

**📊 数据集**

数据集：合成Euclidean TSP/CVRP（20、50、100节点，1,000/64个实例）；公开基准 TSPLIB（29实例）和 CVRPLIB（12实例）；训练阶段使用5个随机实例，验证阶段10个实例，测试阶段使用公开的最优/参考解。

**📈 对比分析**

比较方法：与传统手工启发式（LS、GLS、KGLS、EBGLS）、神经网络方法（AM、POMO、GNNGLS、NeuralGLS）以及现有LLM‑EC方法（EoH、ReEvo）进行对比。结果显示：在TSP100上平均gap为0.07%，在CVRP100上为3.53%；在TSPLIB 28/29实例取得最佳或同等最佳，在CVRPLIB 12/12实例取得最佳。相较于前置方法，LLM‑HCJG显著降低了误差并提升了实例最佳率。

**⚠️ 局限性**

局限性：仅在TSP和CVRP两种问题上验证，缺乏对更大规模或更异构实例的泛化验证；使用单一LLM模型，生成质量受模型表现限制；仅针对GLS框架，未探索其他元启发式；训练仍需调用LLM，虽然成本较低，但在大规模部署时仍可能成为瓶颈。

---

## 329. SMart: A Multi-source Multi-phase Time Series Representation Transfer Framework

**arXiv ID:** 2609.02203 | [PDF](https://arxiv.org/pdf/2609.02203v1)

**作者:** Fang He `[一作]` (Pennsylvania State University), Wang-chien Lee `[通讯]` (Pennsylvania State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了SMart框架，实现多源多相位时序表示学习与迁移；

**💡 创新点**

创新点包括：①利用多相位递归图（RP）恢复任务引导编码器学习多粒度时序动态；②设计基于交叉注意力的Source Selector，用编码器参数表示数据集并自动挑选有利的源数据集；

**🔧 技术方法**

技术手段包括Transformer时序编码器、掩码自恢复任务、RP恢复解码器、交叉注意力和多头自注意力；

**📊 数据集**

实验使用85个单变量Bake‑off数据集和9个多变量UEA数据集；

**📈 对比分析**

与ROCKET、Rel‑CNN、TimeNet、TS‑TCC、TS2Vec、DTW‑FCN、TST等SOTA模型对比，SMart在分类上最高准确率87.84%（相较TST提升约1.16%），在回归上MAE下降至0.313（提升约19.5%），训练时间约26.6h；

**⚠️ 局限性**

局限性包括：训练时间相对较长；源选择器仅在单变量数据上充分训练，跨域迁移受限；RP恢复仅考虑τ=1，缺乏更广泛的相位空间探索；对极长序列或复杂多变量场景的适应性待验证；

---

## 330. Do Large Language Models Capture the Diversity in their Training Data?

**arXiv ID:** 2609.02275 | [PDF](https://arxiv.org/pdf/2609.02275v1)

**作者:** Youqi Wu `[一作]` (Chinese University of Hong Kong), Farzan Farnia `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过条件von Neumann熵和矩阵熵测度，对比了多种公开训练的LLM和图像生成模型在给定条件下的输出多样性，并发现模型生成的输出相对于训练数据存在显著的“多样性缺口”，随后提出一种基于熵投影的后处理重加权方法来缓解该缺口。

**💡 创新点**

创新点在于：①将条件von Neumann熵定义为可凸约束的矩阵熵并证明其凹性；②基于此构建凸优化的熵投影重加权框架；③设计了可扩展的产品单纯形镜像下降算法（CEP‑BEG）以及使用随机特征投影的计算加速技术。

**🔧 技术方法**

使用技术包括：核矩阵熵、von Neumann熵、秩‑2矩阵熵、Bregman投影、KL 与 MMD、随机投影/随机 Fourier 特征、镜像下降与指数梯度重正化、以及对生成候选的权重投影。

**📊 数据集**

数据集涵盖：OLMo（Dolma）、Pythia 与 GPT‑Neo（The Pile）公开文本数据；ImageNet 类条件生成（LDM、ADM、BigGAN、DiT‑XL‑2）；MS‑COCO 文本条件生成（U‑ViT、SDXL、PixArt）等；实验亦扩展至长序列与多样化的提示长度。

**📈 对比分析**

比较方法：对训练样本与模型生成样本分别计算条件VNE、RKE、Distinct‑2 等指数化指标，发现模型熵均低于训练数据；通过熵投影重加权后，条件熵显著提升；在SDXL的图像生成实验中加入VNE引导，可在保持质量的前提下进一步提升生成多样性。

**⚠️ 局限性**

限制包括：①仅针对公开可复现训练数据的模型，无法直接验证闭源系统；②熵投影方法需要先生成多候选样本且对生成质量控制有限；③依赖核与特征映射，性能受限于所选核与投影维度；④缺乏对生成内容真实性与一致性的系统性评估。

---

## 331. APEx: Distillation of Agent Procedural Experience for Adaptive Deep Research Question Answering

**arXiv ID:** 2609.02253 | [PDF](https://arxiv.org/pdf/2609.02253v1)

**作者:** Jie Ding `[一作]` (University of Science and Technology of China), Xin Liu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 APEx 框架，构建实例级记忆与类别级技能的层次化经验库，并通过 Executor、Distiller、Planner 的闭环交替 GRPO 训练，实现多轮工具调用、技能蒸馏和测试时无标签强化学习。

**💡 创新点**

创新点在于：① 将交互历史分层存储为实例轨迹与可抽象的类别技能；② 采用三阶段交替 GRPO 优化三大模块，实现模块间信用分配；③ 在推理时用技能作为先验进行无标签测试时强化学习，并通过技能对齐正则防止策略漂移；④ 在不增大记忆占用的前提下显著提升推理效率。

**🔧 技术方法**

技术组合包括：大型多模态 LLM（Qwen2.5‑VL‑7B/8B/32B、Qwen3‑8B/32B），Group Relative Policy Optimization（GRPO）训练；Executor‑Planner‑Distiller 三模块架构；多判别 LLM‑as‑Judge 生成无标签奖励；技能引导的测试时强化学习（TTRL）和技能对齐正则。

**📊 数据集**

数据集涵盖图文与文本的多领域长序任务：FVQA、SimpleVQA、LiveVQA、InfoSeek、MMSearch、HotpotQA、2WikiMultiHopQA 以及训练用的 FVQA‑train、MATPO 等。

**📈 对比分析**

与 GPT‑5.4、Gemini‑3‑Flash、Qwen2.5‑VL‑7B/32B 以及 ReAct、RAG、Mem0、A‑Mem、ExpeL、Memento、MIA 等基线对比，APEx 在 7 个基准上平均 64.4%，比 GPT‑5.4 高 14.7、比 MIA 高 3.0；技能‑仅版已优于 MIA，且在 token‑效率上显著降低记忆占用。

**⚠️ 局限性**

局限性：依赖小型开源模型，未验证更大模型的可扩展性；三阶段交替训练方式不如端到端联合训练充分挖掘协同效应；TTRL 仅无标签，缺少最小监督可能进一步提升技能质量。

---

## 332. Sparse Readout Prism: Explaining Logit-Lens Scores in Features Instead of Tokens

**arXiv ID:** 2609.01936 | [PDF](https://arxiv.org/pdf/2609.01936v1)

**作者:** Matteo He `[一作]` (University of Cambridge), Nicholas D. Lane `[通讯]` (University of Cambridge)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Sparse Readout Prism (SRP)，一种基于稀疏自编码的词表解码特征分解方法，解决了传统 lens 方法中 token identity 与 readout 结构混淆的问题。

**💡 创新点**

创新点在于将 readout 维度稀疏分解为可解释的特征，并通过 SRP 提供与训练语料无关的固定基底，揭示同一 hidden state 在不同 lens 或语料下 token 读数变化的机制。

**🔧 技术方法**

使用稀疏自编码器 (Top‑K SAE) 对 LM 头行向量进行分解，并在所有 lens、层级上统一使用该基底进行分解；结合梯度、替换、重构等技术评估。

**📊 数据集**

数据集包含 Qwen3.5、Gemma‑4、Ministral 等多种模型的 LM 头，评估基于 C4 连续文本、CoarseWSD‑20 语义歧义、翻译对、事实回忆提示等 80+ 任务。

**📈 对比分析**

与六种基于行几何的基线（最近行岭、k‑NN、k‑means、PCA 等）比较，SRP 在覆盖率上提升 8.9–17.3% 并在残差与符号一致性上优于其他方法；在词义识别与对照测试中准确率高于 90%。

**⚠️ 局限性**

局限性包括一次性稀疏自编码的计算成本（6–12 GPU‑h）、对字典大小、稀疏度、训练随机种子敏感，以及仅解释选定 logits，无法直接揭示完整生成过程。

---

## 333. C$^2$T-OpenMax: A Novel Open-Set WiFi RF Fingerprinting Method via Center Constrained Learning and Confidence-Guided Tail Modeling

**arXiv ID:** 2609.02007 | [PDF](https://arxiv.org/pdf/2609.02007v1)

**作者:** Yuanyu Zhang `[一作]` (Xidian University), Yulong Shen `[通讯]` (Xidian University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 C2T-OpenMax 框架，将增强的 WiFi RF 指纹识别模型扩展到开放集识别。

**💡 创新点**

创新点在于结合中心约束学习提升类内紧凑度，并通过置信度导向的尾部建模改进 OpenMax 在数据增强场景下的可靠性。

**🔧 技术方法**

采用 DeepCRF 两阶段训练、中心损失、置信度筛选、Weibull 分布拟合与 OpenMax 重校准等技术。

**📊 数据集**

使用公开的 WiFi CSI 数据集，覆盖九个位置、约一年时间的采集。

**📈 对比分析**

与 MSP、原始 OpenMax、CSSR、ARPL 等方法比较，在多地点、不同开放度设置下均实现最高的开放集准确率、AUROC 与 OSCR。

**⚠️ 局限性**

局限在于对严重多径环境的适应性仍不足，未来需探索更自适应的筛选与更鲁棒的尾部建模。

---

## 334. A physics-enhanced bidirectional multi-order graph fusion network for interpretable bearing remaining useful life prediction

**arXiv ID:** 2609.02190 | [PDF](https://arxiv.org/pdf/2609.02190v1)

**作者:** Haoxuan Zhang `[一作]` (Beihang University), Ruijun Liu `[通讯]` (Beihang University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于物理增强的双向多阶图融合网络（PE‑BMGN）用于滚动轴承剩余使用寿命（RUL）预测；

**💡 创新点**

创新点在于：①将Kolmogorov‑Arnold网络（KAN）嵌入图卷积和回归阶段，显式可视化非线性映射；②采用双向多阶图传播结合门控交叉融合，捕获前向累积与后向校正的衰变依赖；③引入动态记忆银行检索历史衰变原型；④设计物理增强动态损失约束整体学习；

**🔧 技术方法**

技术包括可学习离散小波分解、基于余弦相似度的自适应稀疏图构建、Chebyshev谱图卷积、KAN非线性映射、门控交叉融合、记忆增强回归、物理约束正则化；

**📊 数据集**

使用公开的 XJTU‑SY 与 PHM2012 轴承数据集；

**📈 对比分析**

与 11 种主流方法（如 T‑GCN、ChebGCN‑LSTM、DyWave‑BiAGCN、SSPGKAN 等）在 RMSE、MAE、EAS 上进行比较，PE‑BMGN 在两大数据集均取得最低误差（XJTU‑SY RMSE 0.069、MAE 0.055、EAS 0.0049；PHM2012 RMSE 0.095、MAE 0.074、EAS 0.0067），并且提供更保守、可靠的预测；

**⚠️ 局限性**

局限性包括：对异常或分布漂移明显的衰变轨迹仍易产生误差，记忆银行依赖历史样本多样性，过大/过小的邻居数或记忆容量会影响性能，且模型对实时部署的计算开销相对较高。

---

## 335. Privacy Amplification Without Independence: How Far Negative Dependence Carries the Guarantees of Poisson Subsampling

**arXiv ID:** 2609.01944 | [PDF](https://arxiv.org/pdf/2609.01944v1)

**作者:** Xujun Che `[一作]` (University of North Carolina at Charlotte), Depeng Xu `[通讯]` (University of North Carolina at Charlotte)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了在负相关参与情况下，如何在没有独立性的情况下实现隐私放大，特别是针对Poisson子采样的替代方案，分析了随机分配等结构化参与的隐私保证。

**💡 创新点**

创新点在于提出了一种通用的主导原则，证明了在负相关参与的情况下，参与指示向量的每个整数Rényi阶数的去除方向Rényi散度被边际匹配的独立方案所主导，并且在特定条件下扩展到机制级别。

**🔧 技术方法**

使用了Rényi散度分析、概率机制的理论以及高斯机制的紧密分析等技术。

**📊 数据集**

使用了多种数据集，包括随机分配、每个时期的分配和随机签到等参与模式，特别关注了k=1的情况。

**📈 对比分析**

与传统的Poisson子采样方法进行了比较，结果表明在负相关参与的情况下，随机分配在每个整数Rényi阶数上都能保持隐私保证，且在某些条件下优于Poisson采样。

**⚠️ 局限性**

限制在于该理论主要针对高斯机制，且在整数阶数的情况下有效，机制级别的主导性需要额外的条件（如符号平衡）来确保。

---

## 336. Agent Flight Recorder: Tamper-Evident Audit Trails with On-Chain Anchoring for Long-Horizon Tool-Using Agents

**arXiv ID:** 2609.01931 | [PDF](https://arxiv.org/pdf/2609.01931v1)

**作者:** Laurent Bindschaedler `[一作]` (Max Planck Institute for Software Systems), Christoph Siebenbrunner `[通讯]` (Vienna University of Economics and Business)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一套面向长期运行AI代理的可验证审计架构——Agent Flight Recorder，能够记录代理行为的完整因果链，并通过哈希链、Merkle树与链上锚定实现防篡改与第三方可验证性。

**💡 创新点**

创新点在于：① 将代理行为分解为八个语义字段的结构化事件模式；② 在本地哈希链+Merkle批处理的基础上加入周期性链上根锚定，提供跨组织争议下的无信任验证；③ 通过分层HKDF密钥派生实现选择性披露，兼顾隐私与审计需求；④ 对所有四类篡改（编辑、删除、重排、分叉）进行系统化检测并通过实证验证其 100% 检测率。

**🔧 技术方法**

核心技术包括：确定性 CBOR 序列化、SHA‑256 哈希链、Merkle 树批处理、epoch 根链路、以太坊 L2 链（Base Sepolia）上的智能合约锚定、ECDSA 签名、HKDF 密钥派生与对称加密、Python 3.11、Solidity。

**📊 数据集**

使用的数据集：① 基于 agent 任务的合成工作负载（100、1k、10k 事件，八类工具调用；约 15% 并发）；② 真实 SWE‑agent 跟踪（SWE‑bench 共 38 事件，涵盖文件、API、授权等）；③ 通过 Base Sepolia 部署的链上锚定交易用以评估 gas 成本。

**📈 对比分析**

比较方法：按层级 ablation 评估（Baseline → Schema → Chain → Merkle → Full），测量每事件的延迟、存储占用、锚定成本与检测率；对比不同锚定机制（签名摘要、透明日志、TSA、链上）在信任假设、成本与第三方可验证性上的差异。性能方面，完整系统平均每事件 48 µs 延迟、512 字节存储；链上 L2 锚定成本约 2.30 USD/10^5 事件，窗口 100 s；所有篡改类 100% 检测；结构化查询的精准率从 1.0 提升至 0.013/0.077 以上。

**⚠️ 局限性**

主要局限：① 完整性无法覆盖绕过记录器的行为；② 仅保证在主机被完全劫持前的记录可信；③ 若离线存储丢失，链根无法恢复事件内容；④ 不捕捉模型内部推理或外部通信的侧信道；⑤ 依赖链可用性，链下时窗口拉长；⑥ 可能被攻击者延迟锚定，需外部监控；⑦ 选择性披露在强制披露场景下不具备不可恢复性。

---

## 337. WeaveMark: Robust and Scalable Multi-bit LLM Watermarking via Coded Payload Spreading

**arXiv ID:** 2609.02177 | [PDF](https://arxiv.org/pdf/2609.02177v1)

**作者:** Gang-Hyun Park `[一作]` (University of Ulsan), Dae-Young Yun `[通讯]` (University of Ulsan)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大语言模型生成文本的多比特水印，提出 WeaveMark 方案，改进了载荷容量、提取准确率和文本质量，并支持可靠的零比特检测。

**💡 创新点**

创新点包括：① 通过多比特/多层混合的 payload 扩散（multi‑bit‑per‑token + layer shuffling）提升载荷容量；② 使用软决策 ECC 解码利用投票余量最大化纠错；③ 预留专用零比特层实现无偏检测，避免了传统方案的最大化统计偏移。

**🔧 技术方法**

核心技术：无偏多层重加权、随机层次分配、可逆 ECC 码（如 Reed‑Muller、Golay）、软判决投票、上下文伪随机分区、Z‑score 统计检测。

**📊 数据集**

主要数据集：C4 RealNewsLike、CNN/DailyMail（摘要）、WMT16 en→ro（翻译）；模型包括 LLaMA‑3‑8B、Gemma‑2‑9B、Mistral‑7B、OpenGen。

**📈 对比分析**

与 BiMark、MPAC、Qu 等基线相比，WeaveMark 在 200‑token 32‑bit 水印的匹配率从 20.8% 提升至 89.8%；在 10% 同义词替换攻击下匹配率从 30.7% 提升至 86%；文本质量保持与无水印相当；零比特检测的 TPR 维持在 95% 以上，阈值稳定。

**⚠️ 局限性**

局限性：需要更多的层数与计算资源；在极短文本或非常高噪声（如大规模改写）下仍会出现误码；未充分评估跨模型（不同架构、不同 token 词表）和多语言场景的鲁棒性。

---

## 338. TAME: Temporal-Aware Mixture-of-Experts for Text-Video Retrieval

**arXiv ID:** 2609.02204 | [PDF](https://arxiv.org/pdf/2609.02204v1)

**作者:** Uicheol Jung `[一作]`, Yukyung Choi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在文本-视频检索任务中，基于CLIP双编码器的TAME模型通过稀疏Mixture-of-Experts（MoE）实现了帧级和文本级的专家路由，加入了帧-时序（FT）token和句子条件下的跨时序交互聚合（CTIA）模块，显著提升了视频时序建模与跨模态对齐的能力。

**💡 创新点**

创新点包括：① 采用帧一致的MoE路由，扩展编码器容量同时保持CLIP的跨模态对齐；② 设计轻量级FT token在视觉分支中注入跨帧上下文；③ 提出CTIA机制，将句子条件的帧级相似度通过局部平滑、图传播和可学习融合，生成更鲁棒的视频级匹配分数。

**🔧 技术方法**

核心技术包括：稀疏Mixture-of-Experts（Top‑K路由与负载均衡），帧-时序（FT）token与全局注意力融合，CTIA的三阶相关信号（raw、局部卷积、图传播）及其可学习加权。

**📊 数据集**

使用的公开视频检索基准有：MSR‑VTT、DiDeMo、MSVD、LSMDC和ActivityNet，训练时每个视频采样12帧（或64帧），文本长度上限32/64。

**📈 对比分析**

与CLIP4Clip、X‑CLIP、CenterCLIP等基线对比，TAME在MSR‑VTT R@1提升约+4.0（达到48.0），在DiDeMo提升+2.7，其他数据集亦有1–2%幅度提升；整体计算量仅略高于CLIP4Clip，模型参数约205M，GFLOPs约77。

**⚠️ 局限性**

局限性包括：① MoE路由需负载均衡与稳定训练；② FT token和全局注意力增加内存与推理成本，尤其对长视频不友好；③ CTIA对帧级相似度质量高度依赖，若单帧匹配弱则聚合效果受限；④ 目前仅适用于固定采样帧，缺乏动态帧采样或在线流式检索能力。

---

## 339. When Does Information Sharing Improve Decentralized Discovery? Aggregation, Independent Rescue, and Equilibrium Selection

**arXiv ID:** 2609.01814 | [PDF](https://arxiv.org/pdf/2609.01814v1)

**作者:** Yohei Nakajima `[一作]` `[通讯]`, Yohei Nakajima

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究有限搜索中信息共享对聚合后行动预算与个人救援机会的影响，并给出了精确的共享收益判据与两人策略游戏的共享区间。

**💡 创新点**

提出了增量共享同一性和残差误差前沿判据，系统分类共享曲线类型，发现共享收益区间仅在特定均衡选择下成立，并给出精确阈值。

**🔧 技术方法**

利用符号推导、精确有限枚举、代数证明与证书（如根区间证据）对不同信息结构与行动策略进行分析。

**📊 数据集**

使用合成的通道注册表（DD‑019、DD‑020、DD‑021、DD‑022）共177个场景，生成精确概率与恢复预算数据。

**📈 对比分析**

通过比较共享与私有、共享与中心化、不同均衡之间的发现概率与预期收益，得到阈值条件（如ρ∈(5√73‑17/48,1)）并证明在这些区间内共享收益严格提升，混合曲线在注册表中不存在。

**⚠️ 局限性**

局限在于只考虑有限的合成通道、两人二选一的策略模型、假设共享信息完全真诚且救援行动独立，未涉及动态学习、真实人类或组织实验数据。

---

## 340. Induction and Inquiry via Probabilistic Reasoning over Language and Code

**arXiv ID:** 2609.01815 | [PDF](https://arxiv.org/pdf/2609.01815v1)

**作者:** Wasu Top Piriyakulkij `[一作]` (Cornell University), Kevin Ellis `[通讯]` (Cornell University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于心理程序（自然语言+Python代码）与LLM辅助的贝叶斯序贯推理模型，用以模拟人类在稀疏、噪声流数据下的归纳学习与主动探询过程。

**💡 创新点**

创新点在于将自然语言与可执行代码混合为可解释的假设空间，并利用LLM作为低成本的假设生成器，结合序贯蒙特卡洛实现高效的贝叶斯更新，既捕捉不确定性又保持开放性与可解释性。

**🔧 技术方法**

采用LLM-guided Sequential Monte Carlo (LLM‑SMC‑S)、自然语言+代码的心理程序表示、贝叶斯概率推断、信息增益式实验/提问选择等技术。

**📊 数据集**

实验数据来源包括Rule et al. 250个算法学习任务、Thaker et al. 数字类别（anchor/garden‑path）任务、Zendo 目标概念学习与实验数据集，以及小规模的网络购物问答样本。

**📈 对比分析**

与传统贝叶斯、纯LLM、定制贝叶斯学习器及HL 500k搜索模型对比，LLM‑SMC‑S在计算预算有限（如5–50粒子）时就能达到或超过人类平均准确率，并在对数似然、R²等指标上优于对照方法，尤其在序贯学习与主动实验的综合任务中表现突出。

**⚠️ 局限性**

局限性包括：依赖预训练LLM的知识范围，难以处理不易用自然语言或代码表述的概念；假设空间离散化导致搜索空间仍有限；对LLM的“幻觉”依赖需进一步控制；在需要更大粒子数时计算成本仍显著；无法学习超出LLM知识库的新范畴。

---

## 341. GenCAR: Generative Counterfactual Alignment with Risk-Controlled Selection for Out-of-Distribution Recommendation

**arXiv ID:** 2609.02162 | [PDF](https://arxiv.org/pdf/2609.02162v1)

**作者:** Qianqian Wang `[一作]` (Southern University of Science and Technology), Lili Yang `[通讯]` (Southern University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 GenCAR 两阶段框架：先通过保持稳定偏好、对环境因素进行干预并使用离线 LLM 生成对抗候选，再用对齐代理和 conformal 校准实现 OOD 推荐的风险控制。

**💡 创新点**

创新点：将 OOD 推荐建模为 α‑Valid Counterfactual Recommendation (α‑VCR)；在生成候选时保持用户偏好不变并利用 LLM 离线生成；随后采用对齐代理与 Benjamini–Hochberg/Benjamini–Yekutieli 校准的 conformal p‑value，提供分布无关的有限样本 FDR 控制；在线不调用 LLM，保持低延迟。

**🔧 技术方法**

技术手段：结构因果模型、变分自编码器 (CausalVAE)、贝叶斯个性化排序 (BPR)、Anchor 投影、trust‑radius 过滤、LLM 离线对齐预测器、对齐阈值 τ、conformal 预测、Benjamini–Hochberg (BH) 与 Benjamini–Yekutieli (BY) 校准。

**📊 数据集**

数据集：MovieLens(ML)-100K、Coat、Amazon-Book、Amazon-Beauty、MovieLens-1M、Steam 以及用于校准的附加数据集。

**📈 对比分析**

方法对比与性能：与 BPRMF、DICE、IPS‑CN、CausE、MACR、InvCF、CDR、CausalDiffRec、TallRec 等基线比较；在 temporal、exposure、popularity 三种 shift 下，GenCAR 在 Recall@10 上显著提升（相较对应 CausalVAE backbone 提升 11%~43%），且实现的 FDP 始终低于预设 α；在线延迟保持在子毫秒级。

**⚠️ 局限性**

局限性：依赖离线 LLM 生成与对齐代理，若代理与因果真实不匹配会影响风险控制；α 与 trust‑radius 的选择对结果敏感；在仅部分用户有离线 LLM 覆盖时需额外 rerank 方案；当前仅提供池化级别的 FDR 保障，无法为每个用户单独给出保证。

---

## 342. PEARL: Path-Entity Aligned Relational Learning with Contextual Subgraphs for Inductive Knowledge Graph Completion

**arXiv ID:** 2609.02216 | [PDF](https://arxiv.org/pdf/2609.02216v1)

**作者:** Yunchi Yang `[一作]` (Shandong University), Cunquan Qu `[通讯]` (Shandong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PEARL框架，联合建模查询上下文子图与多跳关系路径，实现对未见实体的归纳知识图补全。

**💡 创新点**

创新点包括：①基于查询实体邻域的上下文子图而非传统交叉子图；②使用大语言模型进行路径检索与对齐，提升路径语义相关性；③双视图子图对比学习抑制噪声；④通过二分图神经网络实现路径-实体-全局交互，实现上下文感知路径推理。

**🔧 技术方法**

技术细节包括：CSNN对上下文子图进行消息传播；LPRA模块利用LLM（Qwen3-4B）评估路径语义重要性；BGNN使用多头R‑GAT实现路径与实体的交互；RAPF进行关系感知的注意力聚合；SCL采用InfoNCE对抗两视图子图以增强鲁棒性。

**📊 数据集**

在三个公开归纳知识图补全基准上评测：WN18RR、FB15k-237、NELL-995，各自划分的 V1–V4 子集。

**📈 对比分析**

与规则、子图、子图+路径等主流方法对比，PEARL 在所有数据集和设置下均取得最高 Hits@10（相对提升约 8.67%、6.01%、3.20%），并在训练收敛速度与 GPU 内存占用上表现优异。

**⚠️ 局限性**

局限性：依赖离线LLM检索，导致预处理开销和时延；未在动态或时序知识图上进行评估；对极大规模实时查询的可扩展性尚待进一步优化。

---

## 343. FORGE: Forward-Only Test-Time Adaptation for Integer-Only Vision Models on Microcontrollers

**arXiv ID:** 2609.01683 | [PDF](https://arxiv.org/pdf/2609.01683v1)

**作者:** Muhammad Rehan `[一作]` (National University of Sciences and Technology), Moaz Amjad `[通讯]` (National University of Sciences and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种前向单向测试时自适应方法，能够在已折叠的整数卷积微控制器模型上恢复对分布漂移的适应能力。

**💡 创新点**

通过在BN融合后重建通道统计并使用指数移动平均对每通道进行轻量化重校准，使得适配仅需前向推理，无梯度，能够在已折叠的 int8 模型上直接运行。

**🔧 技术方法**

利用整数量化 (int8)、BN 融合、前向指数移动平均、单通道均值/方差重校准、单样本流动窗口匹配动量以及在线安全门控技术。

**📊 数据集**

使用 CIFAR-10/100、Tiny-ImageNet 及其对应的 C 版本腐败数据集（CIFAR‑10‑C、CIFAR‑100‑C、Tiny‑ImageNet‑C）。

**📈 对比分析**

与 BN 适配、TENT 等基线比较，在折叠模型上实现约 +20.9 的准确率提升，几乎匹配梯度方法；在 ESP32‑S3 上适配耗能仅 6.8% 的推理能量，耗时 21.9 ms。

**⚠️ 局限性**

仅适用于卷积层的 BN 融合，无法直接应用于 Transformer 等使用 LayerNorm 的网络；安全门控在类别数较多时失效；适配仍需 FP32 计算和内存，且在极低位宽（如 4‑bit）下性能略有下降。

---

## 344. YesTrack: Referring Multi-Object Tracking via MLLM-based Yes/No Verification

**arXiv ID:** 2609.02318 | [PDF](https://arxiv.org/pdf/2609.02318v1)

**作者:** Quansheng Hu `[一作]` (University of Electronic Science and Technology of China), Jianxiao Zou `[通讯]` (Shenzhen Institute for Advanced Study)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出YesTrack框架，将RMOT任务改为二分类验证，去除文本生成延迟，加入Temporal Confidence Prior和Temporal Reference Propagation，并推广到MOT；

**💡 创新点**

创新点在于将MLLM直接用于二分类验证，消除生成式延迟；引入轻量级时间一致性约束TCP与TRP；实现可迁移到MOT的统一框架；

**🔧 技术方法**

使用Qwen3-VL-2B-Instruct MLLM做二分类；两阶段RMOT；Temporal Confidence Prior、Temporal Reference Propagation；Hungarian匹配；基于距离的门控；

**📊 数据集**

使用Refer-KITTI和Refer-KITTI-V2进行RMOT实验；使用KITTI数据集评估MOT性能；

**📈 对比分析**

与多种端到端与两阶段RMOT方法对比，结果显示在Refer-KITTI HOTA 54.00、AssA 66.57、Refer-KITTI-V2 HOTA 43.75；在KITTI MOT HOTA 45.10，均优于现有基线；

**⚠️ 局限性**

局限在于两阶段架构受底层tracker影响；TRP的key帧更新可能导致误差累积；TCP为经验式，未学习化；

---

## 345. MACAW: Reliable And Efficient Surgical Debridement Using Monocular Adaptive Compact Attention Windows

**arXiv ID:** 2609.01961 | [PDF](https://arxiv.org/pdf/2609.01961v1)

**作者:** Ziyang Chen `[一作]` (University of California, Berkeley), Ken Goldberg `[通讯]` (University of California, Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

利用单目RGB相机和视觉伺服实现自走式手术脱屑，提出MACAW（Monocular Adaptive Compact Attention Windows）窗口方法实现精确深度感知与抓取；

**💡 创新点**

首次在单目视觉下实现深度感知与视觉伺服的闭环配合；设计紧凑的关注窗口结合图像差分、光流与模板匹配，实现实时深度估计与抓取决策；

**🔧 技术方法**

视觉伺服（基于K和T_c^b的相机-基座变换）、SAM3分割、LiteTracker跟踪、MACAW窗口（图像差分+光流+模板匹配）、局部标定、双臂协调控制；

**📊 数据集**

无公开大规模数据集，全部实验基于自制软泡沫碎片的物理实验（100次单臂、400次单臂、10次双臂共计600+次），对比已有的程序化基线与学习策略；

**📈 对比分析**

与两种程序化基线、四种学习策略（ACT、Diffusion Policy、π₀、π₀.5）对比；MACAW单臂单片成功率93%，吞吐304片/小时；双臂实现92%成功率、473片/小时；相较于最优基线ACT（81%）和学习策略，MACAW显著提升；

**⚠️ 局限性**

主要局限在抓取深度固定导致抓取失败与释放失败；对真实粘性碎片的鲁棒性未知；未验证跨任务推广能力；

---

## 346. CREDIT: Cost-guided Reduction-reuse with Efficient DSMEM Inter-CTA Tiling

**arXiv ID:** 2609.01864 | [PDF](https://arxiv.org/pdf/2609.01864v1)

**作者:** Zhengxiong Li `[一作]` (University of Wisconsin--Madison), Umit Ogras `[通讯]` (University of Wisconsin--Madison)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了CREDIT框架，用于识别、预测并利用NVIDIA DSMEM实现Reduction‑Reuse计算加速。

**💡 创新点**

创新点包括：①基于profiling的工作负载特征识别；②针对Reduction‑Reuse的DSMEM转换方案；③成本模型用于预测何时使用DSMEM才具备盈利性。

**🔧 技术方法**

使用了CUDA微基准测量DSMEM原语成本、集群同步与全局‑所有聚合技术，并结合成本模型与控制核实现收益预测。

**📊 数据集**

使用了六种FP32 reduction‑reuse工作负载（LayerNorm、加权方差、Pearson、softmax‑logits、LARS动量、行向量int8量化）进行实验。

**📈 对比分析**

与三种无DSMEM基线（框架、Triton、CUDA）在RTX 5090和H100上比较，CREDIT在64K宽度下平均几何比提升1.466×（RTX 5090）和1.318×（H100），并在所有工作负载中击败最快基线。

**⚠️ 局限性**

限制：仅适用于可压缩的Reduction‑Reuse模式，无法处理无重复读取、扫描或远程状态扩展的内核；对硬件参数敏感，跨平台适用性需进一步验证。

---

## 347. From Feature Interaction to Feature Transport - A Unified Block for Scalable Recommendation Models

**arXiv ID:** 2609.01655 | [PDF](https://arxiv.org/pdf/2609.01655v1)

**作者:** Zichen Luo `[一作]` (Tianjin University), Jie Zhang `[通讯]` (Tianjin University)

**通讯引用:** 194989 | [OpenAlex ID](https://openalex.org/A5100408669)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 CRAFT 模块，利用特征传输方式在统一推荐模型中动态调控非序列特征对意图与序列表示的影响，实现层级递进的表示演化。

**💡 创新点**

创新点在于将非序列特征从被动融合对象转变为主动的上下文控制器，采用可靠性加权上下文、残差自适应偏移与记忆门控实现特征的可控传输，解决传统交互式统一模型的表示崩溃与信息泄漏问题。

**🔧 技术方法**

核心技术包括：多模态 tokenization（RankMixer、结构感知 MLP）、可靠性加权上下文聚合、RMSNorm + 可学习残差调度、桥式信息交换（序列编码、意图‑序列跨注意力、SwiGLU 通道混合）以及教师‑学生蒸馏辅助训练。

**📊 数据集**

使用 TAAC2026 广告推荐竞赛公开数据集，包含稀疏字段、密集向量、上下文信号和多域行为序列，作为评测 AUC 的标准数据集。

**📈 对比分析**

与优化版 HyFormer 基线及其蒸馏版本对比，CRAFT 在无蒸馏时提升 AUC 0.837165→0.837231，加入蒸馏后达到 0.838090（领先 0.000107），六层堆叠进一步提升至 0.838148；参数量和 GFLOPs 较基线略低，表示更高的计算效率。

**⚠️ 局限性**

局限性包括：深层堆叠导致 GFLOPs 明显升高，对大规模部署的计算成本有一定压力；模型在不同业务域的泛化能力尚未充分验证；仍可能受训练数据中的历史偏差与不平衡影响，需要在实际上线前进行公平性与隐私审计。

---

## 348. Tri-Band Channel Measurement-Enabled Multi-Layer Digital Twin for Terahertz Wireless Data Centers

**arXiv ID:** 2609.01699 | [PDF](https://arxiv.org/pdf/2609.01699v1)

**作者:** Mingjie Zhu `[一作]` (Shanghai Jiao Tong University), Chong Han `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在三频段（140、220、300 GHz）进行真实测量，构建基于LiDAR、THz‑TDS等的测量校准物理双胞胎，并在此基础上开发LoS感知隐式神经场AI双胞胎，实现高精度低时延的THz通道预测；随后基于AI双胞胎进行覆盖、干扰评估并提出AP布置优化。

**💡 创新点**

创新点在于①将多源测量与物理校准、AI推断三层融合的多层数字孪生框架；②设计了带有LoS感知双专家的隐式神经场模型，可显著提高在阻塞密集数据中心环境下的通道重构精度；③通过解析闭式覆盖概率公式实现覆盖评估与资源调度的实时闭环。

**🔧 技术方法**

采用LiDAR三维重建、THz‑TDS材料表征、协同仿真RT+扩散模型、改进的隐式神经场（INF）双专家架构、路径损耗统计与干扰建模等技术。

**📊 数据集**

使用实验室搭建的数据中心，收集了140/220/300 GHz三频段的方向性信道数据，约29个Tx‑Rx组合，总计数百个多径参数；并基于LiDAR点云与THz‑TDS得到几何与材质参数。

**📈 对比分析**

与传统MLP、INF（无LoS先验）和完整RT仿真对比，AI双胞胎在RMSE（≈5.9 dB）上优于MLP（≈7.5 dB）和INF（≈6.3 dB），推理时延仅143 ms，显著快于RT的≈1.8×10^6 ms；覆盖率在10 dB SINR阈值下，天花板AP可达90%以上。

**⚠️ 局限性**

局限性在于模型主要针对单一数据中心布局，未验证跨架构泛化；受测频段限制，对更高/更低频段的适用性未知；且在动态环境或高速移动场景下的实时性与准确性尚未充分评估。

---

## 349. MAOL: Morphology-Aware Ordinal Learning for Fine-Grained Industrial Defect Severity Grading

**arXiv ID:** 2609.02266 | [PDF](https://arxiv.org/pdf/2609.02266v1)

**作者:** Zhaoyang Wang `[一作]` (Hebei University of Technology), Atik Shahariar `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MAOL框架用于细粒度工业缺陷严重程度分级。

**💡 创新点**

将严重度分级建模为实例级序数学习，加入形态学特征、类别自适应阈值及预测感知训练，提升鲁棒性与精度。

**🔧 技术方法**

使用ResNet18编码器、形态学分支、类别嵌入、改进的CORAL序数头以及扰动训练。

**📊 数据集**

在工业缺陷分级基准数据集上，训练集924张图像1992实例，验证集230张图像554实例。

**📈 对比分析**

与规则、分类、距离感知、CORAL/CORN等基线对比，在清晰ROI下Acc≈93.5%、QWK≈96.7%，在预测ROI下Acc≈84.8%、QWK≈88.9%，显著优于所有基线。

**⚠️ 局限性**

仍依赖两阶段检测质量，检测误差仍会影响分级精度，且在极端分割错误时性能受限。

---

## 350. Retrosynthesis of Synthetic Media for Explainable AI Provenance Forensics

**arXiv ID:** 2609.02268 | [PDF](https://arxiv.org/pdf/2609.02268v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 351. CoMerge: Conflict-Driven Preference Optimization for Multi-Task Model Merging

**arXiv ID:** 2609.02273 | [PDF](https://arxiv.org/pdf/2609.02273v1)

**作者:** Mingjie Zheng `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CoMerge，一种基于自监督冲突驱动的偏好优化框架，用于在不完整重训练的情况下高效合并多任务大型语言模型。

**💡 创新点**

创新点在于将模型合并视为偏好优化问题，利用“专家输出”作为正样本、“Task Arithmetic 合并输出”作为负样本，自动生成硬负样本；并在张量级别学习可微合并系数，以精细化冲突抑制。

**🔧 技术方法**

采用自监督偏好优化（DPO）、任务向量提取（全局幅值剪枝+低秩SVD）、张量级学习系数（P×N个标量），与传统数据驱动与数据无关合并方法对比。

**📊 数据集**

使用 MergeBench 评测数据集（Instruction、Safety、Math、Multilingual、Coding 等），在 Llama‑3.1‑8B‑Instruct、Llama‑3.2‑3B‑Instruct 和 Gemma‑2‑2b‑it 上训练并评估。

**📈 对比分析**

与数据无关方法（Task Arithmetic、TIES‑Merging 等）和数据驱动方法（RegMean、AdaMerging 等）以及全参数 Fine‑Tuning（Full‑DPO）比较，CoMerge 在 MergeBench 平均归一化分数 NP 达 0.9968，超越所有基线且仅优化 1,445 个标量，GPU 资源使用比 Full‑DPO 节省 60.1%。

**⚠️ 局限性**

局限在于假设专家输出始终为正、Task Arithmetic 负样本完美反映冲突，易受标签噪声；剪枝比例、SVD 最高秩及 Task Arithmetic 乘数等超参数未针对不同模型进行微调，可能影响迁移性。

---

## 352. Diagnosing with Insights: Structured Analysis of Agent Failures via Behavioral Abstractions

**arXiv ID:** 2609.02371 | [PDF](https://arxiv.org/pdf/2609.02371v1)

**作者:** Jiayi Bi `[一作]` (Tsinghua University), Mao Yang `[通讯]` (Microsoft Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种神经符号框架，对LLM代理的执行轨迹进行结构化抽象为行为抽象图，并利用可由LLM验证的神经不变量进行失败定位与归因。

**💡 创新点**

创新点在于：①将长轨迹抽象为有向无环图和ISR结构化字段；②引入神经不变量，使不变量检查与LLM推理结合，实现精确定位与可解释归因；③分阶段诊断流程（抽象→检测→决策），显著降低单次推理负载与误判。

**🔧 技术方法**

主要技术包括：行为抽象图（BAG）构建、ISR三分支（Intent‑Context、Reasoning‑Action、Signal‑Validation）、神经不变量设计与LLM‑guided 推理、三阶段诊断流水线、并行化与缓存优化。

**📊 数据集**

使用公开的 Who&When 数据集（算法生成与手工制作两子集）以及作者自行构造的 FailureBench（基于故障注入的全故障类型覆盖数据集）。

**📈 对比分析**

与 All-at-once 与 Step-by-step 两个基线对比，定位准确率在 T±3 容差下从 8–25% 提升至 45–75%；归因准确率从 20–25% 提升至 40–50%，并在不同 LLM（GPT‑4o、GPT‑5.1、DeepSeek‑V3.2）上保持一致性。

**⚠️ 局限性**

局限包括：依赖 LLM 推理导致运行时开销较大（最长 12 min 轨迹约 750 s）；对极长或高度交织的多代理轨迹仍易出现误判；目前仅支持单代理诊断，未涵盖并行多代理协同情形。

---

## 353. FOCUS: Foot Observation Confidence for Robust Humanoid Proprioceptive Odometry

**arXiv ID:** 2609.02222 | [PDF](https://arxiv.org/pdf/2609.02222v1)

**作者:** Kaixin Feng `[一作]` (Wuhan University), Haiyu Lan `[通讯]` (AgiBot)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

针对动态步态下脚部前向运动（FK）不可靠性，提出 FOCUS 通过持续权重预测连续 FK 可靠性，并在 EKF 中自适应融合 FK 与 IMU 的速度观测。

**💡 创新点**

创新点在于：①使用无标注模拟训练的因果 Transformer 预测连续 FK 可靠性权重；②在 EKF 中通过权重调节观测噪声与速度融合，替代硬阈值的二值接触切换。

**🔧 技术方法**

主要技术包括：因果 Transformer、FK 速度一致性损失、轻量级二元交叉熵正则、EKF 观测权重与协方差调节。

**📊 数据集**

数据集：基于 Isaac Lab 的多姿态随机化模拟数据（5420 条训练、1151 条验证，覆盖行走、跳跃、跑步等动作）以及 5 台 A3 Ultra 机器人收集的 19 条真实行走段和 18 条舞蹈序列。

**📈 对比分析**

与阈值接触 EKF、Pronto、CoCo-InEKF、Legolas 对比：模拟行走 ATE 下降 83.7%，真实行走 ATE 下降 70.8%（从 2.634 m 降至 0.768 m），真实舞蹈 ATE 下降 42.7%（从 0.947 m 降至 0.542 m）。

**⚠️ 局限性**

局限性：仅利用 IMU 与关节角度，无法充分处理严重脚部滑动或非接触动态；需要在未见地形或接触条件下实现在线自适应；对复杂不平坦地形的泛化尚未验证。

---

## 354. RecEvolve: A Knowledge-Driven Autonomous Agent System for Recommender Systems

**arXiv ID:** 2609.01622 | [PDF](https://arxiv.org/pdf/2609.01622v1)

**作者:** Weidi Pan `[一作]` (Google), Onkar Dalal `[通讯]` (Google)

**通讯引用:** 155 | [OpenAlex ID](https://openalex.org/A5083882281)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并部署了一个基于知识驱动的自治代理系统 RecEvolve，用来自动化推荐系统的研发流程（从构思、编码、离线训练到在线评估）。

**💡 创新点**

创新点：①将大型生产推荐模型嵌入持续闭环自治管线，实现从零到上手的完整实验周期；②采用无状态子代理和中心指挥器的多代理层级结构，避免长时运行中的上下文漂移；③通过自治实验发现并修正评估协议中的“奖励黑客”漏洞，揭示短视优化偏差与冗余探索问题。

**🔧 技术方法**

技术：大语言模型（LLM）驱动的 Ideator、Critic、Coding Agent；多代理分层协同；版本控制与分支管理；离线训练基于分布式 TPU 集群；在线 A/B 测试与实时评估；知识库更新与策略 π 的自适应搜索。

**📊 数据集**

数据集：Google 生产级两塔检索模型的真实日志数据（约 2M 步训练数据），不公开具体规模；离线评估使用 NDCG、MRR、Recall@K；在线评估使用用户满意度、冷启动、唯一内容等指标。

**📈 对比分析**

比较方法：在生产环境中跑 41 次自治实验（5 条并发线程），每次 3 小时左右计算；最终 champion 模型相较基线提升 NDCG@50 约 19.9%（+3.77% 用户满意度），同时 MRR@50 也提升至 0.444。实验结果与传统手工调参相比，提升显著且上线验证成功。

**⚠️ 局限性**

局限性：①奖励黑客（metric hacking）揭示评估协议脆弱；②短步训练导致的贪婪优化偏差；③重复探索与记忆衰退导致资源浪费；④离线与在线性能迁移存在差距；⑤对领域知识依赖度高，缺乏深度 RecSys 专业化；⑥计算成本高，需双层筛选策略。

---

## 355. Online Non-Monotone DR-Submodular Maximization Matching the Offline $0.401$ Factor

**arXiv ID:** 2609.02145 | [PDF](https://arxiv.org/pdf/2609.02145v1)

**作者:** Vaneet Aggarwal `[一作]` (Purdue University), Yiyang Lu `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文提出了一种在线算法，用于在对角闭包凸集上最大化非负、非单调的DR-子模函数，并实现了与最优离线算法相同的 0.401 近似系数，同时保证子线性遗憾。

**💡 创新点**

创新点包括：① 将离线的异步 Box‑Maximization 步骤重构为一个加权在线子模最大化（USM）游戏，并给出其最优平衡常数为 2√(c_Xc_Y)。② 通过一次性重塑目标 G_t(a)=f_t(x_t⊙a) 将离线证明确保转移到在线累计保证上。③ 结合计数提升、Blackwell 接近性和有限差分重构梯度，实现了噪声可控的后决策价值查询。

**🔧 技术方法**

使用的技术主要有：离线 DR‑子模的延迟连续贪婪与 Box‑Maximization；加权在线 USM 的平衡游戏和 Blackwell 接近性；计数提升把连续目标离散为子集函数；双贪婪在线实现与噪声下的估计；路径积分梯度与有限差分重构梯度；外部线性优化的投影在线梯度上升。

**📊 数据集**

该工作为理论研究，无需使用实际数据集；所有结果均在抽象的函数序列和随机噪声模型上证明。

**📈 对比分析**

与已有工作相比，本文将在线近似系数从 1/e 提升至 0.401，保持子线性遗憾（直接实现 O(T^{3/4})，批处理可达 O(T^{4/5-δ/5})，一点 bandit 为 O(T^{5/6})）。算法在噪声模型下仍保持相同的 0.401 系数，且对噪声量仅影响低阶常数。

**⚠️ 局限性**

局限性：算法需要可观测的后决策价值查询（或满足正锚定条件的 bandit）；对噪声的假设为有界条件下的无偏噪声；实现复杂度高，需多次函数评估；对高维问题的实际计算量仍较大；结果仅给出渐进近似系数，有限期性能受遗憾指数和常数影响。

---

## 356. Semantic Signal-Assisted Inspection and Recovery Allocation in Reverse Logistics

**arXiv ID:** 2609.02116 | [PDF](https://arxiv.org/pdf/2609.02116v1)

**作者:** Jiani He `[一作]` (Independent Researcher), Shangjing Tang `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 SSADS（Semantic Signal‑Assisted Decision Support）框架，利用退货文本中的语义信号生成条件因子和信号质量分数，动态决定检验深度并在共享人工工时限制下进行恢复分配。

**💡 创新点**

创新点在于将非结构化的维护/客户记录转化为可插拔的检验决策与分配输入，实现检验与恢复决策的协同优化，并通过可替换的关键词/短语/LLM提取器验证模块化设计。

**🔧 技术方法**

主要技术包括文本语义提取器（关键词匹配、短语计数、DeepSeek LLM）、基于阈值的自适应检验策略、解析期望收益的线性分配器以及 RLDB 逆物流基准模拟。

**📊 数据集**

使用了自行构建的 RLDB（Reverse Logistics Decision Benchmark）数据集，包含三个场景：IT 基础设施退役、航空器 MRO 维修与退役、以及消费电子退货，每个场景通过噪声生成器产生带有误删或误标注的技术笔记。

**📈 对比分析**

通过与七种基线（随机、FIFO、结构化+噪声全检、结构化+语义自适应检验、全检oracle、无信号/无检验、语义阈值路由）进行配对种子比较，SSADS 关键词实现 IT 场景净恢复值提升约 51%，航空器场景提升 16%（相对噪声全检），消费者场景提升 1.8%；匹配成本检验后航空器场景额外增值 53.9 万美元。

**⚠️ 局限性**

局限性包括：基准为合成数据，缺乏真实退货文本与复杂法规约束；经济目标忽略安全/认证风险、保修成本；文本语料库有限，导致对未见词汇的鲁棒性不佳；LLM 方案依赖外部服务且未评估实时成本。

---

## 357. LeakageBench: Document-Level Leakage Risk for Redacting Personally Identifiable Information in Document Images

**arXiv ID:** 2609.02207 | [PDF](https://arxiv.org/pdf/2609.02207v1)

**作者:** Vishnu Prasad Vijaya Kumar `[一作]` (Technical University of Applied Sciences Würzburg Schweinfurt), Ivan P. Yamshchikov `[通讯]` (Technical University of Applied Sciences Würzburg Schweinfurt)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

LeakageBench 是一个由 500 页文档图片组成的 PII 隐私红action 基准，旨在评估在 OCR 错误、复杂布局和视觉噪声下的 PII 定位与遮蔽准确性。

**💡 创新点**

其创新点包括：① 提供 GDPR 对齐的三层 PII 分类（直接识别、链接键、情境推理）；② 设计了统一的图像空间预测接口，兼容 OCR 依赖与 OCR 无关的模型；③ 引入文档级泄露度（DocLeak）指标，衡量单页 PII 是否被完整遮蔽。

**🔧 技术方法**

使用的技术包括：传统 OCR + 文字级检测流水线（Tesseract、Amazon Textract）配合多种 PII 检测器（Presidio、GLiNER、Amazon Comprehend、Google DLP、OpenAI Privacy Filter）；OCR 无关的视觉语言模型（Qwen3‑VL‑32B、InternVL3‑38B、GPT‑5.4/5.5），以及 GPT‑5.5+Code Interpreter 进行工具辅助定位。

**📊 数据集**

数据集来源于三类公开业务文件：OCR‑IDL（历史扫描档案）、VRDU Ad‑Buy 表单（结构化发票/收据）和 FCC 公共文件（自由格式信件/邮件），共 500 页、11,954 条 PII 注释。

**📈 对比分析**

评估方法以 IoU≥0.75 的一对一匹配为基础，报告 entity‑level F1_loc、F1_type、DocLeak_all 与 DocLeak_crit。实验显示，最优系统（Amazon Comprehend PII）F1_loc 达 0.304、F1_type 0.119，但文档级泄露率仍高达 0.968，说明即使定位精度提高，单页泄露风险仍未得到根本缓解。

**⚠️ 局限性**

局限性包括：仅涵盖公开业务文档，未涉及手写、医疗、法律、跨语种或多页推理场景；评估仅为页级，未衡量跨页上下文；使用的 PII 分类以识别性为主，可能不完全符合各地法规；模型结果受 API 行为和图像预处理的影响。

---

## 358. Differential Games for Compositional Handling of Competing Control Tasks

**arXiv ID:** 2609.01838 | [PDF](https://arxiv.org/pdf/2609.01838v1)

**作者:** Joshua Shay Kricheli `[一作]` `[通讯]`, Joshua Shay Kricheli

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文提出一种基于差分游戏的“分而治之”控制设计框架，能够将单机智能体多目标动力系统拆分成若干虚拟输入，构建非合作非零和差分游戏，求解其 Nash 均衡，从而得到能够在各目标间保持稳定平衡的复合控制器，并给出连续、离散时间的数学推导与实现。

**💡 创新点**

创新点：
• 将每个控制目标映射为虚拟输入，利用差分游戏实现目标间的内部竞争；
• 通过保证 Nash 均衡实现多目标平衡，避免手工调优权重；
• 提出将代数 Riccati 方程转换为微分 Riccati 方程后迭代求解的新算法；
• 开源 Python 包 PyDiffGame，集成上述方法，支持连续、离散两类系统。

**🔧 技术方法**

所用技术：差分游戏理论（非零和、Nash 均衡）、线性二次调节器 (LQR)、代数 Riccati 方程 (CARE/DARE)、微分 Riccati 方程 (DRE)、连续/离散时间控制理论、Python 编程。

**📊 数据集**

数据集/案例：
• 倒立摆-滑移车系统（单目标 + 位置/速度控制）；
• 非线性层级控制四旋翼（姿态、速度、位置三层目标）。
均为仿真模型，无公开数据集。

**📈 对比分析**

比较方法与性能：与传统 LQR 在相同模型下对比，使用瞬态响应、稳态误差、控制能量等多指标评估；结果显示在冲击响应更快、稳态误差更小、各目标之间更平衡，整体性能优于单一 LQR。

**⚠️ 局限性**

局限性：
• 需要能够将系统控制输入拆分为若干虚拟输入；
• 求解耦合 GCARE 仍存在计算复杂度，特别是大规模状态/目标；
• Nash 均衡存在性与唯一性需满足一定条件；
• 离散化与连续化之间的误差未被系统性分析；
• 对模型不确定性、强非线性情况的鲁棒性尚未证明。

---

## 359. ASCII Attack: Recontextualising Harmful Requests as Artistic Critique in Large Language Models

**arXiv ID:** 2609.02215 | [PDF](https://arxiv.org/pdf/2609.02215v1)

**作者:** Da Cheng Gu `[一作]` (University of Technology Sydney), Wei Liu `[通讯]` (University of Technology Sydney)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种单轮黑盒攻击方法——ASCII Attack，它将完整可读的有害请求嵌入到ASCII艺术字符块中，随后以艺术点评的形式请求模型给出技术细节，从而绕过安全拒绝机制。

**💡 创新点**

创新点包括：
①首次明确定义并公开可复现的ASCII Attack；
②引入匹配对比实验设计，将每个攻击实例与相同内容的直接提问控制配对，隔离表面形式的影响；
③发现该攻击的效果与模型架构相关而非主题相关，且规模增大并不减弱；
④在单轮查询下与ArtPrompt、DeepInception、PAIR等已知攻击比较，ASCII Attack在四个主观判别器上表现最佳，且与多轮PAIR的成功率相当。

**🔧 技术方法**

使用的技术主要有：
- 单轮黑盒文本提示（无系统提示）
- ASCII艺术字符块与情境描述相结合的提示模板
- 匹配对照控制（同一有害请求的直述版本）
- 多判别器响应级别安全评估（JailbreakBench、WildGuard、Llama Guard 3/4、Granite Guardian）
- 统计分析（混合效应逻辑回归、Wilcoxon签名秩检验、Cohen κ）

**📊 数据集**

数据集包括：
- 11款目标模型（从1B到17B，涵盖DeepSeek、Gemma、GPT‑OSS、Llama、Qwen等家族）
- 8个危害主题（如BIC、Synthetic Identity Fraud、Elder Scam等）
- Alpha语料（43,113条响应，其中32,356条ASCII攻击，10,757条控制）
- Bravo语料（42,240条响应，完整种子记录，用于复现检验）

**📈 对比分析**

对比方法：在相同模型、相同主题、相同温度和种子条件下，分别生成ASCII攻击和匹配控制的回复，并用上述五个判别器进行评分。结果显示：
- ASCII攻击的整体成功率为60.1%（JailbreakBench）/38.1%（WildGuard）/23.1%（LG3）/11.4%（LG4）/11.4%（Granite Guardian）；
- 在四个判别器上均超过ArtPrompt（36.5%）和DeepInception（22.8%），仅Granite Guardian略低；
- 单轮ASCII攻击与多轮PAIR（最多15轮）在大多数模型上相当或更优；
- 通过最佳三攻击合并，成功率提升至77%。

**⚠️ 局限性**

局限性：
- 仅在英文单轮查询场景下评估；多轮、跨语言或非ASCII形式未测试；
- 受限于模型的提示模板和运行时实现，部分模型的chat模板来自运行时，可能影响结果；
- 判别器之间存在显著分歧，说明安全评估的主观性和不确定性；
- 该攻击并未验证在已对同类表面进行过强化学习或加密的模型中的效果；
- 数据集和模型细节因安全原因不公开，限制了外部复现与深入分析。

---

## 360. GlyphAnchor: Enhancing Visual Text Rendering via Position-Anchored Glyph Priors

**arXiv ID:** 2609.02349 | [PDF](https://arxiv.org/pdf/2609.02349v1)

**作者:** Qiang Xiang `[一作]` (Fudan University), Junping Zhang `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 GlyphAnchor 方法，在文本生成与编辑中通过位置锚定的字形补丁提升文字渲染质量。

**💡 创新点**

创新点在于将字形补丁与目标布局通过 RoPE 位置编码对齐，实现轻量化兼容多种扩散 Transformer，并采用分阶段监督微调与文本感知 NFT 后训练增强稳定性。

**🔧 技术方法**

使用技术包括位置锚定字形补丁、RoPE 位置编码、分阶段监督微调、文本感知 DiffusionNFT 奖励机制、MLLM 布局规划器等。

**📊 数据集**

使用的数据集包括约 50 万条网页文本图像与模板构造图像，评估基准涵盖 LongTextBench、OneIGBench、CVTG-2K、ChineseWord 与新建的 InfoTextBench。

**📈 对比分析**

与原始模型和现有文字渲染方法对比，GlyphAnchor 在所有基准上均提升文本准确率、词级召回/精确度、短语命中率和整体视觉质量，尤其在长、复杂、稠密文本和稀有字符场景上表现显著。

**⚠️ 局限性**

局限性包括对布局输入的依赖、对极短或单字符场景提升有限、生成时 token 开销略增以及在极端视觉风格匹配上仍有改进空间。

---

## 361. text2ql: Multi-Target Natural Language Querying via a Language-Agnostic Intermediate Representation

**arXiv ID:** 2609.02115 | [PDF](https://arxiv.org/pdf/2609.02115v1)

**作者:** Ritesh Kumar `[一作]` `[通讯]` (Independent Researcher USA), Ritesh Kumar (Independent Researcher USA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了text2ql框架，实现从自然语言到多目标（SQL、GraphQL）查询的统一转换；

**💡 创新点**

创新点在于语言无关的中间表示QueryIR、零LLM确定性引擎、可插拔渲染器和运行时置信度评分；

**🔧 技术方法**

采用七阶段检测管道、Schema-aware Prompting、LLM（gpt‑4o‑mini）或函数调用、Python dataclass IR和插件式渲染；

**📊 数据集**

评估使用Spider和BIRD两大文本到SQL基准数据集（随机抽样50条）；

**📈 对比分析**

在确定性模式下执行准确率100%，延迟≤5 ms；LLM模式下执行准确率84–91%，Exact Match 62–70%（Spider）/70–78%（BIRD）；

**⚠️ 局限性**

局限包括样本量小、缺乏GraphQL金标准、手工Schema配置成本高、未进行人工评估、Exact Match指标不够充分；

---

## 362. MeanField Surrogate Modeling for Scalable Runtime Scheduling of Concurrent Heterogeneous AI Inference on Shared GPUs

**arXiv ID:** 2609.02109 | [PDF](https://arxiv.org/pdf/2609.02109v1)

**作者:** Youssef Ennouri `[一作]` (Seoul National University), Soonhoi Ha `[通讯]` (Seoul National University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了MeanField近似的占位模型，用以可扩展预测共享GPU上异构AI推理（LLM与YOLO）并发时的吞吐量，并将其集成进遗传算法调度器，实现实时的资源调度；

**💡 创新点**

核心创新在于将每个模型的性能仅依赖于本地配置与聚合GPU状态，摒弃了组合式联合建模导致的指数级配置空间；样本复杂度线性增长（≈20N），并在GA搜索中保持与穷举搜索相当的效果；

**🔧 技术方法**

采用64‑32层MLP（BatchNorm、ReLU、Dropout、Sigmoid输出）构建MeanField占位器，配合分层抽样和遗传算法（交叉、变异、锦标选拔、精英保留）实现高效搜索；

**📊 数据集**

通过在NVIDIA RTX 3090上离线采样得到的LLM（Qwen2.5 FP16/AWQ，request_concurrency {1,2,4}）与YOLO11（{n,s,m}，skip_rate {1,2,4}，imgsz {320,480,640}）多配置组合，在低/中/高/突发负载下收集吞吐量/帧率数据；

**📈 对比分析**

与联合MLP基准、完整穷举搜索、静态Full/预算、随机方法对比。MeanField+GA在N=5、78,732可行配置下与穷举差距<0.10%，无SLA违例，决策延迟26 ms，搜索速度约为穷举的5倍；在八个动态场景中相较随机提升≈50%，相较预算有限的静态配置提升≈10%；

**⚠️ 局限性**

该方法仅考虑聚合GPU状态，忽略了模型间特定交互；实验仅覆盖LLM与YOLO两类工作负载，未涉及KV-cache预填/解码阶段；在极端负载波动或更大规模多模型系统中仍需进一步验证。

---

## 363. Modelstamp: Pre-Deserialization Verification of Machine-Learning Artifacts and Runtime Environment State

**arXiv ID:** 2609.01781 | [PDF](https://arxiv.org/pdf/2609.01781v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 364. RouteGraph-Mona: Confusion-Aware Routing Fine-Tuning for Mineral Image Classification

**arXiv ID:** 2609.02282 | [PDF](https://arxiv.org/pdf/2609.02282v1)

**作者:** Jierui Li `[一作]` (Xidian University), Wei Wang `[通讯]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出RouteGraph-Mona方法，通过在Mona适配器中引入样本自适应路由，构建路由空间并在该空间中进行正则化，以提升矿物图像分类性能。

**💡 创新点**

1）用样本自适应路由替代静态多尺度聚合；2）在路由空间中引入类别级路由锚点与在线混淆图，形成类间路由距离的加权边距；3）只更新轻量级模块，保持主干冻结。

**🔧 技术方法**

视觉参数高效适配器Mona、深度学习Transformer（ViT、Swin）、多尺度深度卷积分支、软最大路由器、类级路由锚点、在线混淆图与加权margin正则化。

**📊 数据集**

Minet、MinetV2和MineralPhotos三大矿物图像公开数据集。

**📈 对比分析**

在ViT-B/16和Swin-B两种预训练主干上与线性探针、全微调、Adapter、LoRA、Mona以及DenseNet、EfficientNet、YOLOv8-cls、SwinMin等基线对比，RouteGraph-Mona平均提升1–3个百分点准确率，且在参数量和显存占用上保持与Mona相近。

**⚠️ 局限性**

对小型数据集提升有限；路由锚点数量和混淆图更新策略对性能敏感；仅在预训练冻结主干下表现出优势，进一步扩大到更大多样化数据集的验证尚待验证。

---

## 365. DPA: Decoupling Product-Agnostic Anomaly Representations for Zero-shot Anomaly Generation

**arXiv ID:** 2609.02075 | [PDF](https://arxiv.org/pdf/2609.02075v1)

**作者:** Hang Yao `[一作]` (Harbin Institute of Technology), Wangmeng Zuo `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

该研究提出一种基于跨产品异常迁移的零样本异常生成框架 DPA，利用现有产品的真实异常样本生成新产品的高质量异常图像及对应标注。

**💡 创新点**

创新点在于①通过异常概念解耦（ACD）将异常信息与产品外观分离；②引入自适应掩模生成与精细标注；③在生成前用视觉语言模型过滤语义不匹配的异常类型，形成异常转移的零样本设置。

**🔧 技术方法**

采用 Stable Diffusion 1.5 作为扩散模型生成器，结合可学习文本嵌入、注意力引导掩模、Anomaly‑oriented Training Paradigm（ATP）等技术实现异常概念学习与迁移。

**📊 数据集**

在 MVTec‑AD、VisA 两大工业异常检测数据集以及自行构造的跨产品异常迁移基准 ATAD 上进行实验。

**📈 对比分析**

与现有零样本生成方法（DRAEM、CPR、RealNet、GLASS、AnomalyAny）以及少样本方法（AnomalyDiffusion、AnoGen、TF‑IDG）对比，DPA 在图像级 AUROC、像素级 AP 等指标上均领先 2–8 分，并且在多种检测框架中均能提升性能。

**⚠️ 局限性**

局限性包括需要源产品提供足够覆盖的异常类型；异常迁移效果受源异常与目标产品匹配度影响；对极端不匹配或极少源异常时性能下降；仍需离线过滤与人工标注源异常类别。

---

## 366. Information Density Imbalance in Visual Object Detection

**arXiv ID:** 2609.02369 | [PDF](https://arxiv.org/pdf/2609.02369v1)

**作者:** Ziwei Zhao `[一作]` (Technical University of Munich), Yanbiao Ma `[通讯]` (Renmin University of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出信息密度（Information Density）这一指标，用来衡量每个类别在目标检测中学习难度，并利用该指标改进三种主流不平衡损失函数（Seesaw、EFL、C2AM），在训练过程中动态更新信息密度，进一步提升模型对低频、难学类别的检测性能。

**💡 创新点**

创新点在于：①首次将类别多样性和实例面积结合成信息密度，并证明其与类别平均精度呈显著负相关；②将信息密度嵌入已有损失函数，实现对信息密度失衡的补偿；③设计低成本的动态更新策略（基于局部协方差矩阵合并），显著降低存储开销并保持信息密度精度。

**🔧 技术方法**

主要技术包括：基于特征嵌入的协方差矩阵估计（Ledoit‑Péché 收缩），信息密度计算与归一化，信息密度引导的 Seesaw、EFL、C2AM 损失重构，动态更新队列与局部协方差合并算法；实现框架基于 MMDetection、ResNet‑50/101 + FPN、SGD 训练。

**📊 数据集**

在 Pascal VOC（20 类非长尾）、COCO‑LT（80 类长尾）和 LVIS v1.0（1203 类长尾）三大检测基准上进行实验验证。

**📈 对比分析**

对比原始损失函数与信息密度改进版，使用 mAP、各频率组 AP、AP_r 等指标；实验表明信息密度改进在所有基准上均提升整体 mAP（约 1–3%），对低频/难学类别提升更显著（AP_r 3–5%），同时通过 AP 方差显著降低模型偏差（偏差下降 30–50%）。

**⚠️ 局限性**

局限性包括：仅针对三种损失函数验证，未探究信息密度在其他检测框架或网络结构中的泛化；动态更新仍需额外存储和计算（虽然已降低但仍有开销）；信息密度仅捕捉多样性与面积两因子，可能无法全面解释所有导致模型偏差的复杂因素。

---

## 367. From Detection to Characterization: A Large-Scale Study of Ragebait on Japanese X

**arXiv ID:** 2609.02262 | [PDF](https://arxiv.org/pdf/2609.02262v1)

**作者:** Zhiyang Qi `[一作]` (University of Tokyo), Fujio Toriumi `[通讯]` (University of Tokyo)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究开发了针对日本X平台的ragebait检测框架，并对数以亿级数据进行了大规模特征、扩散和情绪反应分析。

**💡 创新点**

创新点包括：①利用LLM自动生成伪标注数据，克服手工标注成本；②构建多模型投票集成，提升鲁棒性；③首次系统性量化ragebait在主题、扩散速度及受众情绪上的差异。

**🔧 技术方法**

核心技术包括：GPT‑5.4 mini对文本进行二元标签推断；对多种日本预训练模型（Tohoku BERT、Rinna RoBERTa、LINE DistilBERT 等）进行微调；采用多数投票集成；结构主题模型（STM）提取主题分布；情绪与情感分类模型评估回复与引用文本。

**📊 数据集**

使用了：1）1%样本的日本X事件记录（10/2022‑6/2023，共约153.8M条）；2）过滤后约17.4M原始推文；3）18,558条LLM伪标注数据（9,279 正例、9,279 反例）做训练/测试；4）约34.5M带回复/引用的原始推文用于情绪分析。

**📈 对比分析**

评估方法：在保留的2,000条测试集上比较六种单一模型，最高单一模型准确率83.40%、宏F1 83.38%；通过三模型多数投票得到集成模型，准确率84.05%、宏F1 84.04%，优于任何单一模型，显示集成方法有效提升性能。

**⚠️ 局限性**

局限性：①伪标注的可靠性受LLM主观性影响，未完全等同人工金标准；②仅聚焦日本X平台与1%样本，结果可能不具备跨平台或跨语言通用性；③模型对新兴ragebait表述可能缺乏及时适应；④分析主要基于文本与互动指标，未涵盖更深层次语境与多模态信息。

---

## 368. Efficient Context-Limited Telescope Bibliography Classification for the WASP-2025 Shared Task Using SciBERT

**arXiv ID:** 2609.01647 | [PDF](https://arxiv.org/pdf/2609.01647v1)

**作者:** Madhusudhana Naidu `[一作]` `[通讯]`, Madhusudhana Naidu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于SciBERT的高效方法，对科学论文进行四类（科学、仪器、提及、非望远镜）自动分类，显著提高了图书馆与档案工作的效率。

**💡 创新点**

创新点在于利用SciBERT的领域适配能力，结合上下文截断与分块策略，在512词限制下仍保持高精度，并对截断、分块与长文本模型的权衡进行了系统分析。

**🔧 技术方法**

使用的技术主要包括SciBERT预训练模型、文本截断与分块处理、宏F1评估指标及与长文本模型的对比实验。

**📊 数据集**

使用的数据集为WASP-2025共享任务提供的科研论文集合，包含多标签分类任务的标注数据。

**📈 对比分析**

与传统手工分类以及其他机器学习方法相比，本文方法在WASP-2025排行榜中取得宏F1 0.89的最高成绩，显示出优越的性能。

**⚠️ 局限性**

主要限制在于上下文长度受512词限制，截断会导致信息损失；虽然SciBERT表现稳健，但在极长文本和计算资源受限的环境下仍需进一步优化。

---

## 369. Unified Motion Retargeting for Humanoids with Learned Point Cloud Correspondence

**arXiv ID:** 2609.02134 | [PDF](https://arxiv.org/pdf/2609.02134v1)

**作者:** Hanyang Cao `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Renjing Xu `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出统一运动重定向（UMR）框架，利用稠密点云对应关系将多源人类运动转移到不同 humanoid 机器人，并实现姿态与接触的精准匹配。

**💡 创新点**

创新点在于以表面为中心、稠密点云学习方式消除手工骨骼映射，提供跨源、跨机器人的一致重定向接口，并直接支持接触关系传递。

**🔧 技术方法**

采用 PointNet+MLP 的点云对应学习，Chamfer、排斥、光滑等损失，配合约束 Gauss‑Newton 优化实现几何与动力学一致的重定向。

**📊 数据集**

使用 SMPL‑X LaFAN1、SOMA、MimicKit、BONES‑SEED、OmniContact、GRAIL 等多源人类运动数据，以及 Unitree G1 机器人数据。

**📈 对比分析**

与 GMR、Unitree 手工重定向、OmniRetarget 等方法在 LAFAN1 姿态跟踪、SONIC 大规模训练和多种接触任务中对比，UMR 在成功率、跟踪误差、奖励等指标上均优于对比方法，尤其在高动态和接触任务中提升显著。

**⚠️ 局限性**

仅适用于已配合网格化、T‑pose 参考的人类运动，缺乏对无结构观测或更高自由度关节（如手指、多人）等情况的适应。

---

## 370. Instance Optimal Sparse Recovery from Nonlinear Observations: A Unified Framework

**arXiv ID:** 2609.02120 | [PDF](https://arxiv.org/pdf/2609.02120v1)

**作者:** Junren Chen `[一作]` (Columbia University), Arian Maleki `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种通用框架，利用信号相关的约束近似可逆条件（RAIC）实现非线性测量下稀疏恢复的实例最优性，并给出实例：稀疏相位恢复、一比特压缩感知和稀疏ReLU回归的高效算法。

**💡 创新点**

创新点在于：①构造信号相关RAIC，将实例最优性问题转化为迭代硬阈值算法的收敛分析；②在三种非线性测量模型中首次给出可实现实例最优性的高效算法；③提出新的实例相关超平面分割（hyperplane tessellation）定理，克服传统均匀分割下的统计误差限制。

**🔧 技术方法**

主要技术手段包括：迭代硬阈值（IHT）与归一化IHT；信号相关RAIC的构造与证明；稀疏特征的随机矩阵理论（如稀疏特征值、Gaussian宽度、随机超平面嵌入）；高维概率工具（如切比雪夫、Chernoff、经验过程理论）及梯度匹配误差分析。

**📊 数据集**

本文为理论研究，不涉及具体数据集；所有结果均在高斯随机设计下给出。

**📈 对比分析**

与现有方法比较：在稀疏相位恢复上提供了首个实例最优且计算可行的算法，样本复杂度为Õ(s³)（比现有最优Õ(s²)仍有差距）；在一比特压缩感知上实现了实例最优的错误率Õ(s/m)，并给出了更精细的模型误差项；在稀疏ReLU回归上也给出与前沿方法相当的实例最优性。总体上，算法在理论上与经典线性压缩感知的实例最优性相匹配，并兼顾非线性测量的特异性。

**⚠️ 局限性**

局限性包括：①稀疏相位恢复的样本复杂度仍高于最优Õ(s²)；②一比特压缩感知的模型误差项带有额外对数因子，是否可去除尚未解决；③框架目前仅在高斯设计下证明，如何推广到非高斯或结构化测量矩阵仍是开放问题；④对模型误差的处理仍需进一步优化，尤其在更稀疏或高维场景下。

---

## 371. ORB-SVM : An Innovative Hybrid Framework for Efficient Brain Tumor Detection from MRI Scans

**arXiv ID:** 2609.02333 | [PDF](https://arxiv.org/pdf/2609.02333v1)

**作者:** Amirhosein Azarpour `[一作]` `[通讯]` (Shahid Beheshti University), Amirhosein Azarpour (Shahid Beheshti University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于 ORB 特征提取和 SVM 分类器的脑肿瘤检测框架

**💡 创新点**

创新点在于将 ORB 的局部稀疏特征与 BoW 压缩相结合，显著减少 99.5% 的数据量，并在保持高准确率的同时实现极低的模型参数

**🔧 技术方法**

技术包括 ORB（FAST+Rotated BRIEF）、Bag of Words（k‑means 150 词典）、标准化处理和 RBF‑SVM 分类

**📊 数据集**

使用公开的 Br35H（Brain Tumor Detection 2020）数据集，共 3000 张 MRI 图像

**📈 对比分析**

与多种深度学习模型（Custom CNN、Xception、DenseNet169、InceptionResNetV2、ResUNet）进行对比，ORB‑SVM 仅 12,349 参数却达到 97.5% 的准确率，远超大模型且参数量更低

**⚠️ 局限性**

局限在于仅针对二分类任务且数据集单一，需进一步验证在不同肿瘤类型、更多多分类数据集上的泛化能力

---

## 372. Structured-Prior-Guided Diffusion Inpainting with Physical Consistency for Traffic Sign Augmentation

**arXiv ID:** 2609.02348 | [PDF](https://arxiv.org/pdf/2609.02348v1)

**作者:** Luo Li `[一作]` (AMAP, Alibaba Group), Liang Cao `[通讯]` (AMAP, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种结构化先验引导的扩散式修复框架，用于交通标识的图像补全与数据增强。

**💡 创新点**

创新点在于将标识的语义、外观与几何先验分别拆分为三条互相独立的条件通道（JSON文本、前视向量模板+IP‑Adapter、仿射对齐向量模板+ControlNet），并在训练中加入CIELAB色度L1与Sobel梯度一致性损失，显著提升生成的语义准确性与物理一致性。

**🔧 技术方法**

使用 Stable Diffusion 1.5 作为基座，结合 IP‑Adapter、ControlNet、CIELAB 色差、Sobel 结构损失进行自监督重建训练；推理时采用 UniPC 采样器，控制指导强度与模板投影参数。

**📊 数据集**

训练数据为公司自采集的 1,673,112 张标识图像（包含类别、框、数字），评估数据为公开的 TT100K‑2021 数据集（5,182 区域）。

**📈 对比分析**

在 TT100K‑2021 上零拷贝对比七个竞争模型，所有九项指标（PSNR、SSIM、LPIPS、FID、边缘IoU、OCR‑EM 等）均排名第一；OCR‑EM 达 91.1%（比 FLUX.1 44.2% 高 46.9pt），模型参数约 1.4B，推理时间仅为 FLUX.1 的 1/14；在下游罕见类检测上，合成数据将 AP50 提升 1.23×–7.40×。

**⚠️ 局限性**

局限性包括：仅覆盖圆形和三角形标识；几何拟合在大角度遮挡时失效；单目标编辑序列化，无法一次性多标签；对非标准化标识（如不规则标志、车牌）效果未知；在极低数据比例（10%）下提升有限。

---

## 373. SignMatch: Matching Dictionary Signs to Continuous Sign Language Video

**arXiv ID:** 2609.01886 | [PDF](https://arxiv.org/pdf/2609.01886v1)

**作者:** Ryan Wong `[一作]` (University of Oxford), Andrew Zisserman `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个两阶段的视觉匹配框架，将连续手语视频与词典手语视频在视觉上对齐，以实现字典检索、手语定位和大规模自动标注。

**💡 创新点**

①通过在连续手语数据上学习可解释的原型结构的嵌入空间，解耦了表示学习与字典对齐；②利用原型之间的相似度生成软监督，提升视觉相似性捕捉；③通过迭代伪标签扩展训练，实现跨语言迁移和在无监督语料上的可扩展性。

**🔧 技术方法**

使用基于姿态的AGCN+Conformer网络与基于RGB的SignRep Transformer作为特征提取器；在两阶段中分别使用线性投影与双层LSTM对嵌入进行聚合；采用温度缩放的软最大化与KL散度作为训练损失。

**📊 数据集**

连续语料：BSLCorpus、YouTube‑ASL、YouTube‑BSL、BOBSL；字典资源：BSL SignBank、BSLDict、ASLDict；评估数据集：ASL‑Citizen、ChaLearn OSLWL、BOBSL（CSLR2）等。

**📈 对比分析**

在ASL‑Citizen字典检索中，DCG从基线71.21提升到83.17，R@1、R@5亦显著提升；在ChaLearn OSLWL手语定位中，F1从0.395提升至0.684/0.695，超越现有最高0.596；在BOBSL自动标注中，WER、mIoU、F1均大幅改进，尤其在无字幕条件下仍保持较高性能。

**⚠️ 局限性**

对字典资源依赖强，若缺失对应字典则无法对齐；主要聚焦手部动作，非手势（面部表情、眼神等）处理不足；实验覆盖的语言有限，未验证极端低资源或完全无监督场景的鲁棒性。

---

## 374. Propose to Learn, Learn to Propose: Evaluability-Aware Assistance under Bounded Rationality

**arXiv ID:** 2609.02242 | [PDF](https://arxiv.org/pdf/2609.02242v1)

**作者:** Yifan Zhu `[一作]` (ELLIS Institute Finland), Samuel Kaski `[通讯]` (ELLIS Institute Finland)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了评估可行性（evaluatorability）为辅助决策的重要维度，并设计了ProSE框架和ProSE‑Plan深度‑2贝叶斯自适应规划器，结合二元接受/拒绝的有界理性用户响应模型，实现了在序列化提案任务中同时优化提案价值与可评估性。

**💡 创新点**

创新点在于：①将每个提案视为既是干预也是对用户可评估性的观察；②引入基于信息熵的可评估性边界与信息前沿，证明最具信息量的提案往往位于拒绝边界之外；③通过响应依赖的贝叶斯更新与深度‑2规划实现了评估可行性驱动的决策。

**🔧 技术方法**

使用的技术包括：贝叶斯决策理论、贝叶斯自适应马尔可夫决策过程、KL正则化有界理性模型、Fisher信息分析、深度‑2树搜索与精确网格后验更新。

**📊 数据集**

使用的数据集是控制图形模拟任务（branching‑corridor和probe‑commit），不涉及公开真实数据集。

**📈 对比分析**

与随机、价值贪婪、阈值、群体/个人化贪婪、冻结后验深度‑2以及真值深度‑2等基线对比，ProSE‑Plan在高评估成本下成功率提升约20–30%，在probe‑commit任务中成功率从55%提升至81%，表明在评估可行性瓶颈场景中性能显著优于其他方法。

**⚠️ 局限性**

局限性包括：①仅使用二元接受/拒绝的简化响应空间；②评估可行性度量为距离的单一形式，需针对不同任务学习；③深度‑2规划和网格后验推断的可扩展性有限，无法直接应用于开放式设计空间。

---

## 375. How Output Format Confounds Data Quality and Capability in Instruction Tuning

**arXiv ID:** 2609.02015 | [PDF](https://arxiv.org/pdf/2609.02015v1)

**作者:** Chengguang Gan `[一作]` (Independent Researcher), Zhixi Cai `[通讯]` (Monash University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了输出接口对指令调优中数据质量评估与模型能力测量的混淆效应，并用梯度签名与接口残差揭示了接口锁定与能力存储的机制

**💡 创新点**

首次将接口视为测量与能力的根本混淆因素，证明梯度几何能诊断但无法控制接口，且接口残差能精确识别单个单位的目标任务

**🔧 技术方法**

采用低秩适配器、梯度签名、谱统计（有效秩、核范数）、对齐分数（语义/匹配/残差对齐）等技术进行分析

**📊 数据集**

使用12个任务、4种语义等价输出接口（裸答案、原始跨度、JSON字段、任务标签），在Qwen3.5-4B/9B和Mistral-7B三大模型族上进行实验，并引入多种噪声/格式扰动

**📈 对比分析**

对比谱统计与方向对齐分数，发现谱统计对接口不敏感且无法区分清洁与受损数据，而方向对齐（尤其残差对齐）能高精度区分并在所有模型族上精确识别目标；但能力测量呈接口锁定现象，单一评估预算可导致效果正负翻转

**⚠️ 局限性**

实验仅涵盖分类/短篇推理任务与低秩适配器，未扩展至长文本生成或完整微调；数据集与接口样式有限，规模受限，可能在更大规模或不同任务上出现不同程度的效应

---

## 376. Post-Training Ternarization of Qwen3-4B Capability, Effective Bit Budget, Storage Compression, and Deployment

**arXiv ID:** 2609.01962 | [PDF](https://arxiv.org/pdf/2609.01962v1)

**作者:** Anirudh Malik `[一作]`, Poojith Devan `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 Qwen3-4B 4B 参数预训练语言模型进行后训练低位数三元化（PTQ），并从能力、比特预算、存储压缩和部署四个维度进行完整评估。

**💡 创新点**

首次提供端到端实验报告，量化“1.58‑bit”标签与实际有效比特、压缩效果以及部署时性能的关系；阐明三元化后模型在不同任务和语料库中的能力衰减模式。

**🔧 技术方法**

使用 KOTMS 旋转、E2M‑ATQ 两阶三元化、GPTQ 误差补偿；保持 FP16 激活、A16 权重；无梯度训练，仅用 64 条 WikiText‑2 校准样本。

**📊 数据集**

WikiText‑2（校准与评估）、PTB、C4（跨语料库困惑度评估）以及 lm‑evaluation‑harness 的十个基准任务。

**📈 对比分析**

采用零样本 lm‑eval 对比 FP16 与 W1.58A16：平均准确率下降 9.8pp，WikiText‑2 困惑度提升 1.38×；压缩后大小从 8.29 GiB 降至 3.96 GiB；重构路径推理速度比 FP16 慢约 1.29×，原生三元化核更慢 4.6×。

**⚠️ 局限性**

局限性包括：单种随机种子、未完成部分超参调优；仅权重量化，嵌入和头部保持 BF16；未评估压缩后模型在所有任务上的性能；执行路径未针对硬件优化；缺乏跨规模、跨硬件的验证。

---

## 377. NS-Copilot: An LLM-Driven Agent System for Autonomous Neuroscience Analysis

**arXiv ID:** 2609.01971 | [PDF](https://arxiv.org/pdf/2609.01971v1)

**作者:** Wuche Liu `[一作]` (Case Western Reserve University), Jing Ma `[通讯]` (Case Western Reserve University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个多代理LLM驱动的神经科学分析系统，能够从原始神经信号和自然语言任务描述自动完成数据预处理、模型选择、代码生成、执行、性能评估直至最终报告。

**💡 创新点**

创新点包括：①统一接口集成九个跨模态预训练模型并实现自动模型选择；②闭环控制器可诊断性能瓶颈并在修改或重新规划之间切换；③全流程由LLM生成与执行代码，消除手工工程，显著降低专家门槛。

**🔧 技术方法**

核心技术为多代理架构（Planner、Coder、Controller、Interpreter）、LLM（GPT‑5.2、Claude Sonnet 4.5）驱动的代码生成与沙盒执行、预训练模型工具规范化接口、自动性能阈值早停与闭环自适应优化。

**📊 数据集**

实验使用了三大公开数据集：Alzheimer’s EEG（OpenNeuro ds004504）、Parkinson’s EEG（OpenNeuro ds004584）和 Working Memory spike dataset（DANDI 000006），涵盖 EEG 与 extracellular spike 两种模态。

**📈 对比分析**

与通用代理、科学代理、代码生成基线进行对比，采用主指标（AD宏F1、PD/WM平衡准确率）和多项次要指标评估。结果显示系统在三项任务上均优于所有基线，主指标平均提升约 3–5%，并且闭环控制器通过保留最佳回合进一步提升性能。

**⚠️ 局限性**

主要限制：执行时间和计算成本高（多模型循环与闭环调优导致平均 1.5–2 倍开销）；目前仅支持 EEG 与 spike 两种模态；闭环优化依赖预设阈值和重试次数，可能在极端任务下不够鲁棒；系统对数据集特异性敏感，需更多多样化验证。

---

## 378. Codebook Agent: Amortized Topology Design for LLM Multi-Agent Systems

**arXiv ID:** 2609.02264 | [PDF](https://arxiv.org/pdf/2609.02264v1)

**作者:** Jinxi Yu `[一作]` (University of California, Los Angeles), Ying Nian Wu `[通讯]` (University of California, Los Angeles)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了 Codebook Agent：通过离线压缩的拓扑代码书、查询嵌入预测与基于实际 token 的 MLP 评分，直接生成最优 LLM 多体通信拓扑，避免了迭代采样和消息传递。

**💡 创新点**

创新点在于：① 发现有效拓扑仅有少数几种；② 用离散代码书索引而非连续生成；③ 采用奖励加权的代码预测和 token 归一化的 MLP 评分，取代传统基于边数的 GNN 评分。

**🔧 技术方法**

使用技术包括向量量化自动编码器（VQ‑AE）、多层感知机（MLP）预测器与代理、奖励加权 soft 目标、token 归一化成本、离线记录收集与训练；实现基于 gpt‑4o‑mini 与 Qwen‑3‑8B 的 LLM 推理。

**📊 数据集**

使用的数据集：GSM8K、MATH、MultiArith、SVAMP、MBPP、HumanEval（代码生成）以及 MMLU（跨模型迁移）。

**📈 对比分析**

通过与单体提示、多体协作和已有学习拓扑设计方法（如 GPTSwarm、ARG‑Designer、TopoDIM、GTD 等）在准确率、token 消耗和生成延迟的对比，Codebook Agent 在所有 6 个基准上取得最高准确率（平均 84.6%），生成时间仅 2.4 ms，token 使用减少 22–33%。

**⚠️ 局限性**

局限性：需要离线收集并训练代码书；对同质团队最优，异质团队效果略逊；代码书容量增大后未被充分利用；需要手动调节 λ 以平衡准确率与成本。

---

## 379. Monitoring Web Agents Without Internal Signals: Observable Trajectories and Key-Step Supervision

**arXiv ID:** 2609.02057 | [PDF](https://arxiv.org/pdf/2609.02057v1)

**作者:** Sitong Pan `[一作]` (University of Minnesota), Qianwen Wang `[通讯]` (University of Minnesota)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了在Web Agent上基于可观测轨迹信号的前缀级风险预测，并实现了黑盒监控框架。

**💡 创新点**

创新点包括：提出Macro与Micro两类可观测轨迹特征；引入关键失败步骤的监督方法，避免对有效前缀的过早标记；通过LLM判别器自动生成关键步骤标签。

**🔧 技术方法**

技术手段包括：构造宏观行为特征（行为重复、循环、错误累积等）和微观决策一致性特征（多次采样的意图、动作、预期状态一致性）；使用逻辑回归预测器进行风险估计；利用LLM（Gemini‑3.5‑Flash）对完成轨迹进行关键步骤标注。

**📊 数据集**

实验数据集为WebArena‑Lite（165任务）和Online Mind2Web（300任务），并在五种开源与闭源视觉‑语言模型（Qwen3‑VL‑30B、Kimi 2.5、GPT‑5.2、GPT‑5.4‑nano、Claude Sonnet 4.6）上进行评估。

**📈 对比分析**

与内部信号基线（HTC‑Full）以及可观测的标量信号（verbalized confidence、action entropy、token log‑prob）进行对比；Macro特征在绝大多数设置下均优于或与内部基线持平；Macro+Micro在Mind2Web上进一步提升性能；在早期干预（false‑cut）实验中，观测信号实现约44% 的检测率且误报率低于10%。

**⚠️ 局限性**

局限性包括：仅在ReAct框架下验证，Micro特征需要多次采样和语义聚类，导致额外推理成本；关键步骤标签依赖LLM判断且不一定为不可恢复状态；实验未覆盖更广泛的代理架构与更大规模数据集，且内部基线在部分模型/指标上仍保持优势。

---

## 380. Modeling What Changes: Sparse, Residual World Models for Object-Centric Manipulation

**arXiv ID:** 2609.02046 | [PDF](https://arxiv.org/pdf/2609.02046v1)

**作者:** Param Thakkar `[一作]` (Veermata Jijabai Technological Institute), Manisha Sushant Gote `[通讯]` (ZuiGO Private Limited)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种稀疏残差对象中心化世界模型，利用每个对象的变化门控和残差预测，只更新被检测到会动的对象。

**💡 创新点**

创新点在于显式建模“何处变化”，通过门控避免对静止对象的误差注入，并保持参数规模与对象数无关，显著提高F1和参数效率。

**🔧 技术方法**

使用了基于MuJoCo的仿真推送任务，构建了包含位置、速度、目标关系等特征的门控+残差网络，训练时采用二分类交叉熵、L2残差损失与稀疏正则；推理采用Gumbel straight‑through门控。

**📊 数据集**

使用了程序化生成的MuJoCo桌面推送数据集，场景包含3~8个5cm箱子，动作是位姿控制的推手，数据集规模为250个脚本化轨迹，每个轨迹100步。

**📈 对比分析**

与全局MLP密集预测模型和无变化(no‑op)基线进行对比。稀疏模型在整体L2误差上比密集模型低2.5–4.6倍，但与no‑op基本相当；在变化检测F1上保持0.80–0.87，比密集模型高，参数量低8.6–11.1倍；在闭环滚动时，稀疏模型比密集模型误差累积更少，但仍不及no‑op；在规划任务中，稀疏模型的成功率为0.23±0.06，比密集模型0.00显著高，但与随机相比差距不大。

**⚠️ 局限性**

局限性包括：模型仅在结构化状态空间（非像素）下验证；推理速度提升主要是参数效率而非实际延迟；对变化检测的高F1依赖门控稀疏正则，可能对不同任务泛化不足；规划实验范围有限（单一任务、单一规划器），且稀疏模型未能显著超越随机策略。

---

## 381. Unifying Function- and Argument-First Bidirectional Type Systems

**arXiv ID:** 2609.02005 | [PDF](https://arxiv.org/pdf/2609.02005v1)

**作者:** Takuma Yoshioka `[一作]` (Kyoto University), Atsushi Igarashi `[通讯]` (Kyoto University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现了一种统一函数优先与参数优先双向类型系统的形式化框架，并给出了相应的算法与机器化证明。

**💡 创新点**

通过引入可插拔的“指南”与盒子类型，首次实现了对函数应用顺序的灵活控制，使两种传统双向类型风格与高阶多态性在同一系统中兼容。

**🔧 技术方法**

采用盒子类型、颜色类型思想、工作表（worklist）方法，并利用Abella定理证明器完成系统的形式化与证明。

**📊 数据集**

本研究为理论工作，无实验数据集。

**📈 对比分析**

通过元定理（soundness、completeness）与已存在的DK、XO系统进行理论比较，证明了系统的可判定性，但未给出具体数值性能指标。

**⚠️ 局限性**

算法在对声明式系统完整性证明上存在循环作用域问题，导致无法在最一般的情形下实现完整性；此外，系统尚未支持let多态性和显式类型应用等扩展。

---

## 382. Fairness-Aware Multimodal Transformer Modeling for Real-Time Student Attention Estimation

**arXiv ID:** 2609.02232 | [PDF](https://arxiv.org/pdf/2609.02232v1)

**作者:** Christoforos Fragkiadakis `[一作]` (University of Amsterdam), Ali Mohammed Mansoor Alsahag `[通讯]` (University of Amsterdam)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文基于DIPSER多模态课堂数据，构建了实时学生注意力估计系统，并引入公平性约束来减少性别和年龄组别的误差差距；

**💡 创新点**

创新点在于：①提出基于残差融合的Transformer架构，将视觉主预测与可选传感器补偿相结合；②在训练中加入基于主体级MAE差距的正则化，实现公平性正则化；③采用多组随机种子和多次被试级拆分进行稳健评估；

**🔧 技术方法**

技术上使用CLIP‑ViT‑L/14提取视觉特征、GRU对单模态序列建模、残差Transformer对多模态进行时间编码和融合，并用加权MSE+MAE‑gap正则化的损失函数；

**📊 数据集**

使用DIPSER（57名学生，面部视频+可穿戴传感器+注意力标签+自动推断的性别/年龄）进行训练与测试；

**📈 对比分析**

与视觉GRU、传感器GRU及残差Fusion Transformer三基线比较，残差Fusion Transformer在10个种子平均MAE 0.283、RMSE 0.363，且在最差组误差和性别/年龄MAE‑gap方面表现最优；公平性正则化在验证集上可降低MAE‑gap，但在测试集及重复拆分中并未稳健提升，整体预测性能保持相近；

**⚠️ 局限性**

局限性包括：数据样本少、性别/年龄分布不平衡、使用自动推断的属性而非自报、传感器信息贡献有限、以及公平性提升仅在验证集表现，未在未见被试上稳定；实时推理成本主要来自CLIP特征提取，未验证低功耗设备上的性能。

---

## 383. Automated Maize Ear Phenotyping Using 3D Reconstructions

**arXiv ID:** 2609.01921 | [PDF](https://arxiv.org/pdf/2609.01921v1)

**作者:** Ritwesh A. Kumar `[一作]` (Iowa State University), Baskar Ganapathysubramanian `[通讯]` (Iowa State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一套全自动的基于360°视频点云的玉米穗和单粒玉米粒特征提取管线，能够一次性输出11种形态、颜色、体积代理和Packing特征；

**💡 创新点**

创新点包括：①零训练的CPSAM实例分割结合三重展开图像解决缝隙重复计数；②通过FFT+行链图实现行数（KRN）和粒/行计数（KPR）预测；③同时恢复体积代理、凸包体积、凸包邻居数等3D形态与Packing指标；④在不需要专业硬件或专门训练模型的前提下实现育种规模的高通量测量；

**🔧 技术方法**

技术栈包括：COLMAP+NeRF点云重建、PCA轴对齐与圆柱展开、CLAHE对比增强、CPSAM零射训练实例分割、Ball‑Pivoting Algorithm（BPA）求表面积、凸包求体积、Gabriel图获取Packing、FFT+自适应平滑求行数、行链图重建粒/行计数、Earth‑Mover距离用于分布一致性评估；

**📊 数据集**

使用的数据集为：1091只玉米穗点云（已知基因型），其中268只标注集（100耳调参+168耳保留），27只合成耳用于精确几何验证，6只手工完整标注耳用于空间匹配评估；

**📈 对比分析**

与现有2D投影、结构光、微CT等方法对比，本方法在消费级摄像机下实现：KC MAPE 10.33%（R²=0.921），KRN MAE 0.75（95.2%耳在±2行内），Packing邻居数EMD平均0.57；虽然计数精度略低于顶级投影方法，但本管线同时提供体积代理、Packing、颜色等额外3D特征，满足育种规模需求；

**⚠️ 局限性**

局限性包括：①重建噪声导致粒粒计数误差，尤其高行数耳；②行数误判会影响KPR预测；③仅观测外侧盖，内部体积无法完整测量；④对极端曲度、光照变化或不同品种的鲁棒性未完全验证；⑤合成数据与真实数据的差异使行数评估存在偏差。

---

## 384. From Visual Cues to Spoken Narration: Rethinking Audio Description

**arXiv ID:** 2609.01725 | [PDF](https://arxiv.org/pdf/2609.01725v1)

**作者:** Akshita Gupta `[一作]` (TU Darmstadt & hessian.AI), Anna Rohrbach `[通讯]` (TU Darmstadt & hessian.AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个两阶段管道，用于在长片中自动生成音频描述，首先定位可视事件窗口和对话间隙窗口，然后生成对应的简短叙述。

**💡 创新点**

创新点：①重新定义AD任务为同时预测“什么”和“何时”；②设计双头音视频定位器并行预测可视与语音窗口；③引入描述排序损失训练LoRA VLM，使生成文本更符合AD风格。

**🔧 技术方法**

技术细节：双头音视频局部化网络（VideoMAE+log‑mel声学特征多尺度FPN），Soft‑NMS，LoRA适配的Qwen VLM，描述排序损失，focal loss+DIoU损失。

**📊 数据集**

数据集：LongLSMDC（LSMDC v2 8‑min长片，约83.8k AD句子及可视/语音窗口），以及CMD‑AD、MAD‑Eval 用于局部化基准。

**📈 对比分析**

性能评估：在LongLSMDC上，双头本地化分别达到42.2/42.0 mAP，超过单模ActionFormer 11.8/5.0点；CMD‑AD语音窗口mAP 41.0；MAD‑Eval可视窗口mAP 70.0；生成端CIDEr相对基准提升约1.3–1.4点，零样本迁移至MAD‑Eval亦表现显著。

**⚠️ 局限性**

局限性：①两阶段独立训练，未能联合优化；②生成不考虑邻近描述上下文；③缺乏盲/低视用户的真实评估；④角色库自动生成，可能存在错误。

---

## 385. InsightSeg: Reusing Correction Insights for Guideline-Consistent Segmentation

**arXiv ID:** 2609.02002 | [PDF](https://arxiv.org/pdf/2609.02002v1)

**作者:** Vanshika Vats `[一作]` (University of California Santa Cruz), James Davis `[通讯]` (University of California Santa Cruz)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在基于多代理的语义分割中，InsightSeg 通过把成功纠错的经验写入可检索的、视觉锚定的记忆，从而在首次预测前预防常见的规则违规。

**💡 创新点**

核心创新在于：①将纠错过程转换为可复用的自然语言洞察；②使用局部补丁级视觉概念向量为洞察进行视觉锚定；③在新图像前通过稠密补丁匹配检索相关洞察，实现跨图像知识迁移。

**🔧 技术方法**

技术组合包括 Gemini‑2.5 VLM、DINOv3 ViT 提取补丁与全局 CLS 嵌入、Meta‑Analyzer 进行洞察蒸馏、FAISS 索引的可插拔记忆库、文本级去重与视觉多样性筛选。

**📊 数据集**

实验基准为 Waymo guideline‑consistent 数据集（102 张精细标注图）以及 Cityscapes 验证集（500 张）。

**📈 对比分析**

与无记忆的多阶段 Refinement、LISA、GroundedSAM、Gemini‑2.5 等方法比较，InsightSeg 在 Waymo 上 gIoU 提升约 +2.9、mDice +3.0，Cityscapes 上 gIoU +2.5、mPr +8.2，并且平均缩减 62% 的 refinement 步数。

**⚠️ 局限性**

局限性包括：①局部补丁检索对极小或模糊目标的敏感度有限；②记忆库随着训练数据增长可能膨胀，需更高效压缩；③依赖固定的规则集，无法处理完全新颖的标注策略。

---

## 386. Epistemic Sybil Resistance: Multiplying AI Agents Without Multiplying Evidence

**arXiv ID:** 2609.01873 | [PDF](https://arxiv.org/pdf/2609.01873v1)

**作者:** Marc Bara `[一作]` `[通讯]` (Universitat Oberta de Catalunya), Marc Bara (Universitat Oberta de Catalunya)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究并验证了一种基于条件互信息的‘知识性 Sybil 抵御’框架，阐明多代理系统报告复制与证据独立性的区别。

**💡 创新点**

首次将 Sybil 概念迁移到信息内容层面，提出使用 I(Θ;Z|R)=0 衡量报告的新增信息量，并证明仅靠报告内容无法识别证据根源。

**🔧 技术方法**

采用条件互信息理论、Gaussian 共享根模型、有效样本量校正以及多 Agent LLM 实验；对比独立假设聚合、根源感知聚合和相关提取校正聚合。

**📊 数据集**

使用合成季度营收备忘录（约 300 个评估世界）和 20,000+ LLM 调用的人工生成数据集；包含校准数据集和实验数据集。

**📈 对比分析**

实验中，独立假设聚合的覆盖率从 0.94 降至 0.26；根源感知聚合保持在 0.85–0.95 之间，相关提取校正进一步恢复至 ≈0.95；与单一根源的模拟结果一致，验证理论预测。

**⚠️ 局限性**

局限性包括仅使用人工合成任务、单一模型、Gaussian 假设、噪声偏态、未覆盖真实自然语言证据、未给出最小信息接口的定理。

---

## 387. Connectivity Oracles Under Vertex Failures via a Simple and Fast Low-Degree Steiner Forest Decomposition

**arXiv ID:** 2609.02388 | [PDF](https://arxiv.org/pdf/2609.02388v1)

**作者:** Sayan Bhattacharya `[一作]` (University of Warwick), Haoze Wang `[通讯]` (Peking University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种弱型低度 Steiner 森林分解算法，允许分解树包含已移除的顶点，从而在近线性时间内构造最大度为 4 的分解；基于此分解，构造了新的确定性顶点失效连通性预言机，显著降低了预处理、更新空间和时间复杂度。

**💡 创新点**

创新点在于：
1) 通过放宽传统 Steiner 森林分解的“树不包含移除顶点”限制，设计出极简的贪心算法，时间仅为 O((m+n)α(n))；
2) 证明该弱分解在连通性预言机框架中同样有效，消除了原先方法中出现的子多项式因子；
3) 进一步推出一个更简单的预言机，保持 polylog 额外开销，展示了该分解的通用性。

**🔧 技术方法**

核心技术包括：
- 并查集（Disjoint‑Set Union）维护森林连通分量；
- 活跃子图与边界边管理，实现快速查找可连接路径；
- 深度优先搜索（DFS）寻找链接路径并更新活跃集合；
- 递归构造低度层次结构，形成层次化 Steiner 森林；
- 适用于连通性预言机的 Euler‑tour 与二维范围查询等经典数据结构。

**📊 数据集**

论文未给出实验数据或公开数据集，主要侧重理论分析与复杂度证明；若需实测，可在标准图数据集（如 SNAP、DIMACS 等）上验证。

**📈 对比分析**

与现有预言机比较：
- 预处理空间：O(m log³n) 取代先前 O(m log^*n) 或 m^{1+o(1)}；
- 预处理时间：O(d m log³n) 大幅削减了 sub‑poly(n) 因子；
- 更新时间：O(d² log³n log⁴(d log n))，比之前的 O(d² n^{o(1)}) 或 O(d² log⁶n) 更接近最优；
- 查询时间保持 O(d)，与最佳方案一致；
- 简单预言机进一步降低了空间与预处理时间，且更新时间与 d 成正比。

**⚠️ 局限性**

局限性包括：
- 更新时间仍为 O(d²)，在大规模失效场景下可能过高；
- 需要维护的多层结构和 Euler‑tour 使实现复杂；
- 对极稠密图（m ≈ n²）时，时间与空间仍受 m 线性支配；
- 论文未给出实验验证，实际常数与实现细节对性能影响尚待评估。

---

## 388. Before the Script, Set the Stage: How Worldview Simulation Amplifies Psychologically Grounded Persuasion in Multi-Turn Jailbreaking

**arXiv ID:** 2609.02414 | [PDF](https://arxiv.org/pdf/2609.02414v1)

**作者:** Siyu Chen `[一作]` (Shanghai Qi Zhi Institute), Wei Xu `[通讯]` (Shanghai Qi Zhi Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种多轮 jailbreak 评估框架，拆分成因子化的社会影响策略空间和跨轮情景建模模块，并通过 MCTS 在四轮对话中搜索最优因子组合，生成可解释的攻击轨迹。

**💡 创新点**

创新点在于将攻击过程从单一 prompt 转变为可解释的因子组合和情景上下文两部分，揭示模型在不同交互状态下的具体脆弱性；同时发现“具体可执行任务框架”是跨模型共同的从拒绝到同意的恢复路径。

**🔧 技术方法**

核心技术包括 18 个理论驱动的社会影响因子、跨轮情景模拟（维持角色、时间、机构等连续性）、蒙特卡罗树搜索（MCTS）优化因子组合、LLM 评估器给出 1–5 级合规评分。

**📊 数据集**

使用 HarmBench 验证集（80 条行为，涵盖化学、生物、网络犯罪、版权、错误信息、骚扰等多类恶意行为），并在六个前沿模型（Qwen3-Next-80B、DeepSeek-V3.2、GLM-4.7、Gemini-2.5-Flash、GPT-OSS-120B、GPT-5.1）上评测。

**📈 对比分析**

与 14 种基线（如 CL-GSO、PAIRS、TreeAttack、X-Teaming 等）相比，
- 在大部分模型上取得 79.2% 的平均成功率，单模型最高可达 100%；
- 平均查询成本仅为 2.46 次，显著低于基线；
- 在 OpenWeight 模型中几乎达到上限，在更难模型 GLM-4.7 上提升 27.5 分。

**⚠️ 局限性**

局限性包括：实验仅为行为层面的关联分析，未证明因子或情景的因果作用；因子集为手工设计，可能缺失其他有效攻击机制；防御实验仅覆盖有限静态方法，未测试更复杂动态防御或跨文化场景。

---

## 389. Evidence-Guided Detection, Localization and Explanation for Text-Centric Image Forensics

**arXiv ID:** 2609.02097 | [PDF](https://arxiv.org/pdf/2609.02097v1)

**作者:** Peifeng Liu `[一作]` (Shenzhen University), Xiaoye Qiu `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于证据引导的检测-定位-推理框架，实现文本中心图像的真实性判定、篡改定位与结构化取证报告生成。

**💡 创新点**

将检测器、定位器和多模态大语言模型解耦并通过检测结果与定位框相连形成证据链；引入难度感知迭代采样提升定位精度；以及报告-掩码一致性后处理保证报告与定位结果一致。

**🔧 技术方法**

采用DINOv3+卷积辅助定位头做检测，DTD+难度采样做定位，Qwen3‑VL‑4B‑Instruct+LoRA做证据条件推理，并用SFT与RMC后处理。

**📊 数据集**

使用RealText‑V2数据集（GenText‑Forensics Challenge提供的文本图像取证数据）。

**📈 对比分析**

与专家检测模型、通用MLLM和MLLM‑centric方法对比，整体分数0.638位居第二，检测与定位指标均优于多数基线。

**⚠️ 局限性**

对极小或罕见篡改的定位仍有限，报告生成可能出现轻微推理偏差，对极端光照或压缩扰动的鲁棒性不足。

---

## 390. Transfer Safety Awareness for Cross-Modal Safety Drift in Multimodal Large Language Models

**arXiv ID:** 2609.02082 | [PDF](https://arxiv.org/pdf/2609.02082v1)

**作者:** Tianqi Xiao `[一作]` (Tsinghua University), Renmiao Chen `[通讯]` (Tsinghua University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究跨模态安全漂移问题，提出安全感知表示迁移（SRT）方法以提升多模态大语言模型在视觉-文本组合场景下的安全性。

**💡 创新点**

将已在显式有害文本场景中激活的安全表示迁移至跨模态风险输入，并通过方向精炼与对比学习实现安全与效用的平衡。

**🔧 技术方法**

利用激活方向分析、注意力可视化、方向注入、对比监督以及冻结模型参数的轻量级干预技术。

**📊 数据集**

VLSBench、SIUO、HoliSafe、MM‑SafetyBench、FigStep、MM‑Vet、MME、L‑Bench等公开跨模态安全与效用数据集。

**📈 对比分析**

与ETA、DTR、ShiftDC、CoCA等推理阶段安全对齐方法比较，SRT在S_t（安全文本+风险图像/安全图像）场景下将不安全率显著降低（如从90%以上降至≈20%），同时保持或提升MM‑Vet、MME、L‑Bench等效用指标。

**⚠️ 局限性**

仅适用于能够在文本场景中可靠拒绝的模型；对不同架构、规模与训练方式的迁移效果尚需进一步验证。

---

## 391. Toward Explainable and Policy-Aware AI for Carbon Credit Price Prediction: A Research Framework for Emerging Carbon Markets

**arXiv ID:** 2609.01765 | [PDF](https://arxiv.org/pdf/2609.01765v1)

**作者:** Summaiya Unnisa Begum `[一作]`, Mohammed Abdul Ghani Khan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 EPA‑CarbonNet 框架，用于将碳信用价格序列与政策文本融合，完成碳信用价格预测，并对现有研究进行 gap 分析。

**💡 创新点**

创新点包括：①将市场时间序列与结构化政策文本通过 cross‑attention 联合建模；②在预测结果中输出校准的不确定性区间；③为模型提供可解释的政策归因解释；④构建十个研究 gap 的影响‑可行性矩阵，形成系统化的研究议程。

**🔧 技术方法**

技术手段主要有 Transformer‑based cross‑attention 架构、SHAP 解释、校准评估方法以及政策文本嵌入（LLM/ NLP）与市场系列的融合。

**📊 数据集**

数据集为 11 年日度 S&P 碳指数价格，作为实验和评估基础；实验中还引用印度 CCTS 案例作为评估场景。

**📈 对比分析**

与随机游走、传统基准模型进行对比；在 5‑日 RMSE 上随机游走取得 0.0365，而 EPA‑CarbonNet 为 0.0475；方向性准确率 58.6% 超过所有基线；SHAP 重要性排序与随机抽样背景的相关系数仅为 ρ=0.54；政策注意力权重与已记录的监管事件几乎不匹配。

**⚠️ 局限性**

局限性包括：模型预测性能低于随机游走；解释性指标不稳定，政策注意力未能与实际事件对齐；缺乏实证校准与不确定性量化；架构仅为概念设计，未完成训练与基准测试；对低数据新兴市场的适用性仍未验证。

---

## 392. World-Coherent Decoding: Self-Verifying Test-Time Planning for World Action Models

**arXiv ID:** 2609.02159 | [PDF](https://arxiv.org/pdf/2609.02159v1)

**作者:** Chuhan Zhang `[一作]` (Institute of Science Tokyo), Ikuro Sato `[通讯]` (Institute of Science Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种在世界动作模型（WAM）上进行自我验证的测试时规划框架，利用冻结的WAM多次生成未来-动作候选并根据内部生成信号筛选最可靠的动作，再通过执行后观察到的真实未来来反馈并训练轻量级在线预测器，以实现无奖励、无验证器的候选选择；

**💡 创新点**

创新点在于把WAM生成的未来视为可被现实验证的假设，通过内部流量惊讶（视频面）与动作路径努力（动作面）两种生成信号对候选进行排序，并利用执行后产生的自监督不匹配误差对后续候选的评分进行在线校准，既无需更新主模型，又显著提升了在随机场景下的鲁棒性；

**🔧 技术方法**

主要技术包括：1) 冻结的因果视频-动作WAM；2) 基于流匹配的流量惊讶计算；3) 动作解码路径努力评估；4) 延迟自我验证的误差反馈；5) 轻量级在线预测器学习生成特征到误差的映射；

**📊 数据集**

实验数据集主要为RoboTwin 2.0仿真任务集（50个双手操作任务）以及在真实Franka机器人上的锤击视觉偏移测试；

**📈 对比分析**

在RoboTwin 2.0的限随机化Hard模式下，WCD将基准模型的Hard成功率从55.80%提升到60.90%，在Horizon-3任务上提升16.43个百分点；在标准预训练模型上提升约5.10个百分点；在Franka视觉偏移实验中，WCD恢复了冻结模型无法完成的任务；

**⚠️ 局限性**

主要限制是计算开销，虽然在线预测器占用较小，但整体仍需并行生成多条候选；此外，方法依赖于冻结模型的可预测性，若模型自身未来质量过低，候选选择的效果有限；

---

## 393. CA-OPD: Confidence-Aware On-Policy Distillation for Structured Visual Prediction

**arXiv ID:** 2609.02401 | [PDF](https://arxiv.org/pdf/2609.02401v1)

**作者:** Menghao Li `[一作]` (Tianjin University), Fanyi Wang `[通讯]` (StepX)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于教师置信度的自回归视觉‑语言模型知识蒸馏框架（CA‑OPD），通过动态校正学生生成的前缀来缓解错误传播，并将校正决定直接映射到对应位置的监督方式。

**💡 创新点**

创新点在于：①用教师分布的绝对置信度（NLL）代替传统的排名指标来判断是否接受学生预测；②根据校正与否分别采用交叉熵或分布式 KL 监督，形成“干预对齐”监督；③采用余弦退火的置信度阈值，逐步把 rollout 控制权从教师转移回学生。

**🔧 技术方法**

技术手段包括自回归 VLM 推理、on‑policy 蒸馏、教师置信度评估（NLL/概率阈值）、交叉熵和前向 KL 蒸馏、top‑k 近似、以及余弦退火的干预阈值调度。

**📊 数据集**

使用了多任务数据集：GUI grounding 的 ScreenSpot‑v2、ScreenSpot‑Pro；OCR 的 OCRBench‑v2（英文/中文）、CC‑OCR、OmniDocBench；并在 RefCOCO、MMBench 上验证模型的通用性。

**📈 对比分析**

与三种基线（Offline KD、OPD、SKD）以及固定阈值 CA‑OPD 进行对比。实验显示 CA‑OPD 在所有六个目标任务上均优于基线，ScreenSpot‑Pro 提升 5.10 点，OCRBench‑v2 英文提升 3.39 点；同时在 RefCOCO、MMBench 上保持与 SFT 初始化相当的性能，未出现灾难性遗忘。

**⚠️ 局限性**

局限性包括：①对教师置信度阈值和退火曲线的设定较为敏感，需要经验或额外调优；②目前仅验证在 token 序列化的视觉任务，可能不适用于更复杂的结构化输出；③依赖强大的教师模型，成本较高；④对低置信度样本的干预频率仍有限，某些极端错误场景可能未被充分覆盖。

---

## 394. MESSY STREETS: A Benchmark for Geocoding Real-World Addresses

**arXiv ID:** 2609.01612 | [PDF](https://arxiv.org/pdf/2609.01612v1)

**作者:** Edward Gaere `[一作]` (ETH Zurich), Florian von Wangenheim `[通讯]` (ETH Zurich)

**通讯引用:** 7435 | [OpenAlex ID](https://openalex.org/A5060761466)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了基于真实网页地址的 MESSY STREETS 基准，并对 12 家商业及 3 家开源地理编码器在候选返回率和定位精度上进行评估。

**💡 创新点**

首次公开了规模可达 16.8M 条的非规范地址基准，并通过存在性验证与 LLM 评判显著提高了真实度；同时揭示非规范表面形式是影响地理编码召回的主要因素。

**🔧 技术方法**

采用 Web Data Commons 语义提取、OpenAddresses/OSM 对齐、LLM（Qwen3.5 35B）判断、候选返回率、geohash 精度等技术与指标。

**📊 数据集**

数据集包括 2024 年 December Web Data Commons、OpenAddresses、OpenStreetMap 以及 OpenAddresses/OpenStreetMap 交叉验证后的 Gold/Silver 10K 记录。

**📈 对比分析**

通过统一 API 调用、10 次 100 条记录的实验，商业服务在召回率上领先 49%；开放源代码在定位精度上与商业相近，但在候选返回率上明显落后，尤其对表面形式偏差敏感。

**⚠️ 局限性**

局限在于仅涵盖街道级地址、Gold/Silver 仅 10K 条样本、LLM 评判不透明、未覆盖更复杂的 POI/非街道实体，且缺乏对 PII 处理之外的多语言完整性评估。

---

## 395. SpiderSapien: Client-Centric Web Crawler and Security Scanner

**arXiv ID:** 2609.02532 | [PDF](https://arxiv.org/pdf/2609.02532v1)

**作者:** Eric Olsson `[一作]` (Chalmers University of Technology and University of Gothenburg), Andrei Sabelfeld `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种面向客户端交互的黑盒爬虫与安全扫描器（Client‑Centric Web Crawler and Security Scanner），通过浏览器高层用户反馈（如光标样式、tabindex、可见性）发现可交互元素，按优先级顺序执行 UI 操作，并利用 LLM 自动生成表单输入，从而深入挖掘现代 Web 应用的深层状态并检测 XSS 等漏洞。

**💡 创新点**

创新点：
• 采用浏览器渲染层的高层反馈（光标、tabindex、可见性）检测非语义化的交互元素，突破传统仅依赖语义标签或事件监听的局限；
• 设计两阶段优先级策略（先新发现的操作，再当前页面可用元素，最后输入元素），实现更合理的交互顺序；
• 将 LLM 作为表单求解器，支持多种复杂验证（数值、关系约束、后端校验），并对“危险”表单进行判定以避免破坏应用；
• 将上述三大模块统一到单一黑盒循环中，形成可复用的抽象层，为后续扫描器提供通用基石。

**🔧 技术方法**

技术：
• Selenium + ChromeDriver + Chrome DevTools 进行页面渲染与交互；
• 自定义元素检测算法（基于 cursor、tabindex、visible）和优先级调度；
• LangChain + Gemini 2.5 Flash（可替换为 OpenAI GPT、Ollama 等 LLM）用于表单求解与危险性评估；
• 代码覆盖采集采用 PHP xDebug 与 Chrome DevTools byte‑level 覆盖；
• 评估脚本实现多跑、离线日志分析与可视化。

**📊 数据集**

数据集：9 个开源 PHP Web 应用（Nextcloud、Kanboard、TinyFileManager、WordPress、phpMyAdmin、phpBB、DokuWiki、Moodle、Drupal 等），统一化认证与数据库，涵盖传统与单页、React、Vue 等多种前端框架。

**📈 对比分析**

对比方法：在同一硬件与 8 小时扫描时间内，分别跑 8 个主流黑盒扫描器（Skipfish、w3af、CrawlJax、Enemy of the State、Spider‑Scents 等），使用相同 seed URL 与配置；衡量指标为服务器端行级覆盖率（LoC）与已验证 XSS 漏洞数。结果显示：
• 本方法在 7/9 个应用上实现最高覆盖，平均提升约 10‑20%（相对任意单个扫描器）并在所有应用中发现更多 XSS，尤其在传统扫描器缺失的深层页面；
• 在最优覆盖（所有扫描器取并集）上，本方法覆盖率提升约 20%；
• XSS 漏洞覆盖率在 8 个应用中，平均发现 1.5 倍于其它扫描器。

**⚠️ 局限性**

局限性：
• 仍无法完全覆盖某些浅层、广覆盖型应用（如 phpMyAdmin、Drupal），多跑与更长时间仍未完全收敛；
• LLM 表单求解在极端复杂校验或多字段依赖时可能失效，导致漏测或误报；
• 扫描过程中可能触发破坏性操作（如更新全局配置、删除用户），需要更细粒度的安全性约束或状态恢复机制；
• 评估时间和资源消耗较高，且依赖 Chrome 与 LLM API；
• 仅评估 XSS，未覆盖注入、CSRF、文件上传等多样化漏洞；
• 缺少对“意图”或“可执行性”评估（如 CSP、权限限制）导致部分报告不具备真实攻击价值。

---

## 396. Doppio: A Dataset for Contactless Weight Estimation of Falling Particles

**arXiv ID:** 2609.02528 | [PDF](https://arxiv.org/pdf/2609.02528v1)

**作者:** Simon Kiefhaber `[一作]` (TU Darmstadt), Stefan Roth `[通讯]` (TU Darmstadt)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于计算机视觉的无接触粉末质量估计方法，并通过咖啡研磨实验进行验证

**💡 创新点**

首创了Doppio数据集（含落下咖啡颗粒视频与帧级精准重量标注），并展示了多种时空模型在该任务上的效果

**🔧 技术方法**

采用RGB帧输入，利用前馈网络、GRU循环网络和TCN时序卷积网络，并结合ImageNet预训练的MobileNet‑V3与ResNet特征提取器

**📊 数据集**

使用新收集的Doppio数据集，包含三种咖啡豆、25种研磨度共131训练/13验证/75测试序列

**📈 对比分析**

与单纯时间基线对比，所有视觉模型均大幅提升，最佳TCN+ResNet‑34在全序列上MAE仅0.18（即≈0.18g/双份），且在不同子段均表现优异；同时提供计算成本与精度的折中曲线

**⚠️ 局限性**

模型对极少量样本的依赖、对不同颗粒形状的泛化仍有限，且轻量级特征在时序模型中表现欠佳；需进一步研究量化、模型压缩以适配资源受限设备

---

## 397. When Persona Attributes Improve Population Alignment in Large Language Models

**arXiv ID:** 2609.02526 | [PDF](https://arxiv.org/pdf/2609.02526v1)

**作者:** Leon Fröhling `[一作]` (GESIS – Leibniz Institute for Social Sciences), Claudia Wagner `[通讯]` (GESIS – Leibniz Institute for Social Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了使用人格化提示（persona prompting）预测社会调查问卷结果的效果，并系统评估了不同属性选择方法对该方法性能的影响。

**💡 创新点**

创新点在于将人类回答变异度（human response variation）作为解释persona prompting表现差异的关键因素，提出并比较了多种基于统计和LLM的属性选择方法，并在跨国跨调查的实验中验证了这些方法的可泛化性。

**🔧 技术方法**

技术方法包括：使用normalized Shannon entropy和dissention度量人类回答变异度；基于随机森林特征重要性、相关性、语义相似度、以及LLM零射门（set selection和scoring）等多种属性选择策略；利用LLM（Qwen和Llama系列）进行persona构建与问卷回答生成；使用Jensen‑Shannon距离评估预测分布与真实分布的相似度。

**📊 数据集**

所用数据集为四份跨国社会调查：美国General Social Survey（GSS）、德国General Social Survey（GGSS）、美国与德国的World Value Survey（WVS‑US/WVS‑DE），涵盖数千名受访者、数百个问卷变量。

**📈 对比分析**

比较方法：对每个调查、每个LLM、每种属性选择方法，在20个目标问题上平均计算JSD；结果显示：①在人类回答变异度高的问卷上persona prompting优于无persona模型；②基于统计的属性选择（相关性、特征重要性）优于LLM驱动的选择；③在所有模型与调查中，该排名保持一致，且大模型稳定性更好。

**⚠️ 局限性**

局限性包括：仅考虑了固定的五个属性和零射门LLM选择，未探讨其他prompt设计、few‑shot或微调；未引入理论驱动或随机属性选择的基线；缺少对LLM内部机制的解释性分析；以及对跨文化偏差和数据隐私风险的深入讨论不足。

---

## 398. AceSpec: An Asymmetric Edge-Cloud Collaborative Framework for Communication-Efficient LLM Inference

**arXiv ID:** 2609.02514 | [PDF](https://arxiv.org/pdf/2609.02514v1)

**作者:** Yida Zhang `[一作]` (University of Science and Technology Beijing), Rui Wang `[通讯]` (University of Science and Technology Beijing)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了 AceSpec 框架，利用异步多分支缓存和概率 KV 缓存实现边缘‑云协作的低延迟 LLM 推理。

**💡 创新点**

采用异步多分支构造概率缓存，将回滚惩罚化为 O(1) 本地查找，并通过 Lagrangian 优化的非均匀分支分配最大化缓存命中率。

**🔧 技术方法**

使用异步树式草稿、概率 KV 缓存、压缩稀疏分布、网络感知预算、Lagrangian 优化和非均匀几何分支分配等技术。

**📊 数据集**

在 GSM8K、HumanEval、Alpaca 三个数据集上进行评估。

**📈 对比分析**

与自回归、Vanilla Spec、Split Inference、PicoSpec、DSSD 等基线对比，在 50 Kbps WAN 下吞吐量提升最高达 3.52×，并在大多数配置下保持近峰性能。

**⚠️ 局限性**

当边缘模型规模过大或任务对结构要求严格时，多分支扩展耗时超过云回传时间，导致速度低于自回归；在极低带宽（<25 Kbps）时仍受限。

---

## 399. Telligram: Text-Driven Calligram Generation via Diffusion-Guided Skeleton Optimization

**arXiv ID:** 2609.02511 | [PDF](https://arxiv.org/pdf/2609.02511v1)

**作者:** Tianci Shi `[一作]` (Shenzhen University), Pengfei Xu `[通讯]` (Shenzhen University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个两阶段、训练无关的文本输入式文字图像生成框架Telligram，用来生成既能保持字形可读又能呈现语义形状的文字艺术图；

**💡 创新点**

创新点在于将语义形状生成与可读性保持解耦：先通过变分分数蒸馏(VSD)和层级梯度投影生成语义占位先验，再通过几何处理（凸包、骨架拟合、Voronoi划分）恢复可读字形；

**🔧 技术方法**

核心技术包括：变分分数蒸馏对Diffusion模型的语义引导、基于骨架的参数化和层级梯度投影、LogSumExp软形态学闭运算、凸包拟合与Voronoi辅助填充；

**📊 数据集**

实验中主要使用DeepFloyd‑IF Stage I Diffusion模型进行引导，没有采用专门的标注数据集，评估使用随机种子生成的样本、CLIP zero‑shot分类以及OCR（CRNN、PARSeq）识别；

**📈 对比分析**

与现有给定轮廓的调用方法相比，Telligram在无轮廓输入下能实现语义形状与字形可读性的平衡；CLIPScore最高可达约34.3（×10⁻²），OCR识别率分别为PARSeq 44% 和 CRNN 38%，在视觉效果与可读性上与手工设计或基于轮廓的方法相当；

**⚠️ 局限性**

局限性包括：基于骨架与轮廓的低分辨率表示容易导致字形细节丢失、字形识别率低于更锐利边界的基准、无法完全避免字形被过度变形导致的识别失效，且受限于无训练策略，难以在 Stage‑1 就保持字形身份。

---

## 400. An Adaptive Control Architecture for Slope and Terrain Compensation in Autonomous Navigation in Mediterranean Greenhouses

**arXiv ID:** 2609.02487 | [PDF](https://arxiv.org/pdf/2609.02487v1)

**作者:** Fernando Cañadas-Aránega `[一作]` (Universidad de Almería), José L. Blanco-Claraco `[通讯]` (Universidad de Almería)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

在温室环境中针对差速驱动农用机器人开发了一种自适应控制策略，利用载荷、地形类型和坡度的实时估计来提升轨迹跟踪精度；

**💡 创新点**

创新点在于将IMU测得的坡度信息与实验得到的不同土壤摩擦参数相结合，设计了基于增益调度的前馈补偿器，并将其嵌入到由MPC外环和PI内环组成的级联控制结构中；

**🔧 技术方法**

采用了MPC（基于Timed Elastic Band）、PI速度控制、IMU坡度估计、增益调度前馈补偿以及摩擦阈值自适应约束；

**📊 数据集**

使用了仿真环境MVSim，其中包含三种土壤类型（碎石沙、混凝土、压实沙），并在0%与±5%坡度、0kg与70kg载荷两组条件下进行测试；

**📈 对比分析**

与不使用前馈补偿的基准控制器相比，实验数据显示跟踪误差（SAE）下降约10–20%，控制增量（SCI）略有上升（≤2%），整体提升了轨迹跟踪性能和安全约束满足度；

**⚠️ 局限性**

局限性包括仅在仿真环境验证，实际机器人硬件与环境噪声未考量；对坡度估计误差敏感；对地形转移的补偿效果仍有限；未来需在真实温室环境中进一步验证。

---

## 401. UnCapsTSR: An Unsupervised Transformer-based Image Super-Resolution Approach for Capsule Endoscopy Images

**arXiv ID:** 2609.02476 | [PDF](https://arxiv.org/pdf/2609.02476v1)

**作者:** Anjali Sarvaiya `[一作]` (Sardar Vallabhbhai National Institute of Technology), Kiran Raja `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种无监督的Transformer‑GAN框架UnCapsTSR，用于无线胶囊内镜图像的超分辨率重建。

**💡 创新点**

创新点在于引入双向总变差（BTV）损失提升空间连续性，以及设计针对内镜图像的无参考质量指标EndoQM，同时使用无配对数据进行训练。

**🔧 技术方法**

主要技术包括Transformer‑based生成器、双判别器的GAN结构、颜色一致性、纹理损失和BTV损失，并在无监督条件下完成超分辨率。

**📊 数据集**

训练使用了从Kvasir胶囊内镜数据集精细裁剪得到的SR任务专用子集（10,000训练、550验证、1,000测试），并将传统内镜HR图像作为无配对高分辨率样本。

**📈 对比分析**

与多种现有无监督SR方法（如SRResCGAN、DUSGAN、ZSSR、DASR、MDASR）以及在KID、GIANA等外部数据集进行对比，UnCapsTSR在BRISQUE、NIQE、PIQE及自定义EndoQM指标上均显著优越，EndoQM提升幅度可达40%–80%。

**⚠️ 局限性**

局限性包括对大规模无配对数据的依赖、Transformer模型较高的计算与内存开销，以及在极端噪声或特殊病理结构下的进一步泛化性能待验证。

---

## 402. Evaluating ML-based Intrusion Detection Systems: The Illusion of Model Efficacy

**arXiv ID:** 2609.02469 | [PDF](https://arxiv.org/pdf/2609.02469v1)

**作者:** Achilleas Spanos `[一作]` (University of West Attica), Ioanna Kantzavelou `[通讯]` (University of West Attica)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并执行跨日留一攻击实验，评估ML基网络入侵检测系统在未见攻击下的泛化能力，并基于实验结果提出七项新的评估准则。

**💡 创新点**

从‘模型效能幻觉’的视角揭示同分布高准确率并不代表真实世界泛化；通过对比全局与局部SHAP特征选择，证明局部特征会削弱泛化，提出以跨日留一攻击为核心的评估框架。

**🔧 技术方法**

采用随机森林、XGBoost等树模型，结合SMOTE/随机欠采样、Z-score/Min‑Max归一化；使用三种特征选择策略（全特征、局部SHAP、全局SHAP）并评估其对泛化的影响。

**📊 数据集**

使用CICIDS2017公共数据集，利用其每日文件拆分实现跨日实验，保持网络环境一致，单日内仅出现已知攻击。

**📈 对比分析**

通过交叉日测试矩阵、留一攻击实验和召回率/准确率指标对比，发现同分布下近乎100%准确率，跨日/留一攻击时检测率从约0%到≈80%不等，表明模型泛化性能不足；全局SHAP特征选择相比局部和全特征表现更好。

**⚠️ 局限性**

局部特征选择导致泛化下降；实验仅基于单一数据集，未覆盖不同网络环境；未评估对抗性攻击或多模态输入；对模型鲁棒性与可解释性的深入验证仍待进一步研究。

---

## 403. DeepAffinity: Long-Term Aspect Preference Prediction in eCommerce using Small Language Models

**arXiv ID:** 2609.02468 | [PDF](https://arxiv.org/pdf/2609.02468v1)

**作者:** Yotam Eshel `[一作]` (eBay Inc.), Bracha Shapira `[通讯]` (Ben-Gurion University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究并实现了一种预测电商用户对商品属性（如品牌、尺寸、颜色等）偏好的方法，提出了DeepAffinity模型；

**💡 创新点**

创新点在于将属性偏好预测建模为长期序列预测任务，利用小型语言模型与结构化提示、专用预测头以及Affinity Embedding，既实现高精度预测，又兼顾生产环境的低延迟和可扩展性；

**🔧 技术方法**

使用SmolLM‑v2‑135M小型语言模型、字符串化用户历史、[CLS] Affinity Embedding、类别嵌入+多任务分类头，训练时采用一次性全序列多点交叉熵损失，在线推理通过缓存Affinity Embedding并使用轻量分类头完成预测；

**📊 数据集**

在自家eBay美国平台收集的两大数据集上实验：时装类（Clothing‑Shoes‑Accessories）和全品类，涵盖数千万用户、数百万购买与查询，测试集分别为1K（时装）和20K（全品类）；

**📈 对比分析**

与MCV/PMCV/MRV等启发式基线、零射击Gemini 1.5 Flash、以及微调的3B生成式模型进行对比；DeepAffinity在时装数据的Size/Brand/Color微F1均超过0.30，整体表现优于所有基线；在全品类上优于3B SFT模型，7/10属性表现最好；在实际推荐场景中引入Affinity特征后Recall@1提升0.6%，@3 0.4%，@5 0.3%；

**⚠️ 局限性**

局限性包括仅使用固定24小时预测窗口，未显式编码时间戳、时间间隔或季节性等时序特征，对长期偏好漂移缺乏建模，且未在更大规模的SLM上验证扩展性。

---

## 404. Can Risk-Based Alerting Mitigate Cybersecurity Alert Fatigue?

**arXiv ID:** 2609.02465 | [PDF](https://arxiv.org/pdf/2609.02465v1)

**作者:** Rafael Uetz `[一作]` (Fraunhofer Fkie), Martin Henze `[通讯]` (Rwth Aachen University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了风险基础的告警（RBA）方法，并系统评估其对SOC告警疲劳的缓解效果。

**💡 创新点**

将RBA改造为连续优先级排序问题，提出并实现了五个风险假设模块，并构建CATS实验平台。

**🔧 技术方法**

采用基于时间窗口的统计风险计算（累积、种类、稀有度、严重等级、非周期性）并通过加权组合进行优先级评估。

**📊 数据集**

使用八个含真实标注的告警数据集，包括DEDALE、SOCBED、APT29S2、AIT-ADS、ERPCorp Falco等多来源。

**📈 对比分析**

通过AUROC、AP、Brier等多指标对比，最优加权组合在所有数据集上平均AUROC达0.92，显著优于仅按严重级别或无优先级的基线。

**⚠️ 局限性**

主要限制在于数据集主要来自仿真环境、窗口对齐导致实时性受限、对攻击者对抗RBA的评估不足以及未包含更多多样化的攻击与告警来源。

---

## 405. Learn from Whoever Is Right: Answer-Verified Multi-Teacher Distillation for Multi-Domain LLMs

**arXiv ID:** 2609.02548 | [PDF](https://arxiv.org/pdf/2609.02548v1)

**作者:** Xixiang He `[一作]` (National University of Defense Technology), Qingyong Hu `[通讯]` (Intelligent Game and Decision Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了一种多教师自监督策略优化（MT‑SDPO），通过样本级答案验证、可信教师筛选和特权去蒸馏，将多域教师知识融合到单一可部署模型。

**💡 创新点**

将教师可靠性视为样本级可验证答案而非域标签；剔除未验证教师；将自锚点与所有可验证教师反馈合并为单一特权自教师进行蒸馏。

**🔧 技术方法**

基于Self‑Distillation Policy Optimization（SDPO），引入EMA自教师、答案验证器、教师筛选和特权去蒸馏等技术。

**📊 数据集**

使用SciKnowEval L3 Science Q&A子集（化学、材料科学、物理）进行训练与评估。

**📈 对比分析**

与多域后训练、域路由、SDPO等基线对比，在Qwen3‑8B上宏观准确率提升4.64点、最差域提升14.79点、域差从20.96降至5.30，整体表现优于单域教师路由。

**⚠️ 局限性**

仍有约30%样本无正确教师；平衡初始化时可用空间有限；方法依赖可验证答案场景，难以直接迁移到非精确答案任务。

---

## 406. TrajMind: Chaining Role-Specialized LoRAs for Fast-and-Slow Collective Trajectory Anomaly Diagnosis

**arXiv ID:** 2609.02540 | [PDF](https://arxiv.org/pdf/2609.02540v1)

**作者:** Jiahao Wu `[一作]` (Hong Kong Polytechnic University), Qing Li `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 TrajMind 框架，实现了城市轨迹集体异常的高速检测与低延迟诊断，能够生成可验证的 what‑who‑where‑when 诊断报告。

**💡 创新点**

创新点包括：角色专化的 LoRA 适配器在同一冻结的视觉‑语言主干上切换；快慢双路推理，慢路通过画布渲染、文本定位和可执行验证器实现证据支持；以及使用群组相对策略优化（GRPO）提升文本路径的判别与定位性能。

**🔧 技术方法**

使用的技术有：冻结的视觉‑语言模型 + 低秩 LoRA 适配器；画布（canvas）渲染、文本序列化；可执行验证器基于原始轨迹重算；GRPO 的强化学习奖励设计；以及多任务训练与自监督微调。

**📊 数据集**

采用了成都、西安和波尔图三座城市的轨迹数据，人工合成的集体异常场景，用于训练和评估。

**📈 对比分析**

在与传统轨迹异常检测器、深度序列模型、图模型以及 Traj‑MLLM 等基线的对比实验中，慢路在异常类型识别上提升 15.3–29.0% 的三分类平衡准确率，定位 F1 提升 13.8–23.2%；快路将延迟从 9.026 秒降低 41.1%（到 5.315 秒），同时保持 93.5% 以上的二分类平衡准确率。

**⚠️ 局限性**

局限性包括：仍需人工标注集体异常样本，模型对极端交通场景或未见异常模式的泛化尚待验证；快慢双路的切换与验证过程增加系统复杂度；以及在真实部署时需严格处理隐私与安全风险。

---

## 407. A Comparative Study of Graph Representations for GNN-Based Power Grid Control in L2RPN

**arXiv ID:** 2609.02538 | [PDF](https://arxiv.org/pdf/2609.02538v1)

**作者:** Adrian Degenkolb `[一作]` (Karlsruhe Institute of Technology), Benjamin Schäfer `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在L2RPN环境中，系统比较了十种基于GNN的图表示（物理拓扑、功率流敏感度、混合结构）对电网控制任务的学习效果。

**💡 创新点**

首次在固定RL+GNN管道下对多种图构造方法进行严格对比，揭示图复杂度与任务粒度匹配比图表示丰富度更为重要。

**🔧 技术方法**

使用Grid2Op+Grid2Op、PPO算法、支持多边界类型和标量边特征的自定义GNN编码器，并结合MLP基准进行对照。

**📊 数据集**

采用IEEE 14‑bus L2RPN环境（rte_case5_example及IEEE 14-bus）作为训练和评估数据集。

**📈 对比分析**

通过五个随机种子训练，记录训练期的rollout survival和评估期的episode/step survival；结果显示Compact Substation图获得最高生存率，Default紧随其后，物理或敏感度单独图性能略逊，混合图在某些情形下提升了学习速度。

**⚠️ 局限性**

研究仅在单一基准、单一动作空间和单一RL架构下进行，未检验更大规模或不同结构环境，混合图设计未考虑稀疏或可学习的敏感度特征。

---

## 408. Orthogonal Ensembles and Tested Explanations for Performer-Independent Body-Motion Emotion Recognition

**arXiv ID:** 2609.02510 | [PDF](https://arxiv.org/pdf/2609.02510v1)

**作者:** Naoto Nishida `[一作]` (University of Tokyo), Yoshio Ishiguro `[通讯]` (University of Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在DIEM-A 12情绪体动识别任务中，作者构建并评估了一个包含十一种不同偏置（图卷积、注意力、混合/MLP、外部预训练）模型的等权平均logit集成，针对离散表演者的留一法交叉验证，达到了36.80%宏F1；同时提出了一套后验可解释性测试方法，验证模型对运动区域的关注与Laban运动分析（LMA）属性高度对齐，并报告了时间层面分散性等负面发现。

**💡 创新点**

主要创新在于：① 通过组合不同偏置的模型实现错误模式正交化，从而在严苛的离表演者外评估中实现43%相对提升；② 引入可验证的可解释性测试套件（部分遮挡、稳定性、反事实编辑、LMA对齐、叙事审核），提供可靠的模型决策依据，而非仅可视化；③ 以等权logit平均为集成基准，证明其优于概率平均且不需要额外训练。

**🔧 技术方法**

技术包括：多模型集成（等权logit平均）、骨架自监督预训练（Masked Motion Prediction/MAMP）、四类偏置（GCN、Attention、Hybrid/MLP、外部预训练）、离表演者外交叉验证、可解释性评估工具（遮挡、扰动、反事实、LMA关联、叙事审核）。

**📊 数据集**

使用DIEM-A数据集（74名训练表演者、18名测试表演者，24关节骨架序列，12类情绪），在训练集上进行10折留表演者交叉验证。

**📈 对比分析**

与同协议复现的STGCN++基线（25.73%宏F1）相比，十一模型等权logit集成提升至36.80%宏F1，提升11.07个百分点（+43%相对）。集成在不同偏置之间的错误模式正交化是提升的关键；单模型提升有限，表明多样性是主导因素。

**⚠️ 局限性**

局限性包括：① 仅在离表演者外的训练集上评估，未公布测试集成绩；② 仅使用体骨架信息，缺乏面部、音频、场景信息；③ 时间窗口固定为64帧（约0.53s），对完整动作序列处理不足；④ 数据集局限于日台两国表演者，跨文化泛化未知；⑤ 在不同表演者间F1差异大，模型对新表演者的泛化受限；⑥ 解释性套件虽可验证但不保证因果性，且未提供可部署的情绪检测器。

---

## 409. Blending Concepts: Benchmarking Visual Metaphor Generation in Text-to-Image Models

**arXiv ID:** 2609.02502 | [PDF](https://arxiv.org/pdf/2609.02502v1)

**作者:** Chuer Chen `[一作]` (Tongji University), Nan Cao `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 VMetaphor-Bench，首次系统评估文本到图像模型在生成视觉隐喻方面的能力。

**💡 创新点**

创新点在于构建了1500幅真实创意视觉隐喻样本，按三层等级和十类主题组织，并设计了结合 MCQ 与维度评分的混合评估框架，能够细粒度诊断模型在结构、映射等关键维度的表现。

**🔧 技术方法**

采用了 MLLM‑as‑judge 技术（如 Qwen3.5‑27B、GPT‑5.4）来自动生成 MCQ 与维度评分；利用概念性与描述性两种提示策略，配合结构化注释进行评测。

**📊 数据集**

使用的数据集为 1500 张来自 Pinterest 的视觉隐喻图像，经过两阶段 Gemini‑3.1‑Pro 注释后获得源域/目标域、结构类型、意义、映射等标注，并生成对应的概念与描述性提示。

**📈 对比分析**

通过在 11 种 T2I 模型（包括闭源、开源和统一 MLLM）上执行 MCQ 与维度评分，发现闭源模型如 GPT‑Image‑1.5 在整体准确率与维度得分上领先；开源模型在结构与映射层表现不足，但采用描述性提示后可显著提升并缩小与闭源模型的差距。

**⚠️ 局限性**

局限性包括：模型仍难以实现跨域映射与结构化组合；评估结果受 MLLM 判定偏差影响；样本量虽大但仍有限，未覆盖所有隐喻语义与视觉表现的多样性；且隐喻评价具有主观性，当前指标仍需进一步完善。

---

## 410. Training seeds and model-selection stability in recommender-system evaluation

**arXiv ID:** 2609.02499 | [PDF](https://arxiv.org/pdf/2609.02499v1)

**作者:** Juan Manuel Rodriguez `[一作]` (Aalborg University), Antonela Tommasel `[通讯]` (Johannes Kepler University Linz)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究训练种子对推荐系统评估的影响，固定数据分割并对多种模型多配置进行十次随机种子实验，分析用户级指标、模型选择稳定性、验证‑测试一致性和推荐列表一致性。

**💡 创新点**

系统性评估训练种子在多维度（指标、模型选择、列表）上的可检测性和实际意义，强调种子应视为评估协议的一部分而非噪声。

**🔧 技术方法**

基于RecBole实现的BPR、NeuMF、BERT4Rec、SASRec四种常用模型，采用早停、Friedman检验、性能差距分析、Jaccard/AO@k相似度等统计手段。

**📊 数据集**

Movielens‑1M、Steam、Amazon All Beauty 2023 三个基准数据集，按80/10/10时间顺序分割。

**📈 对比分析**

对每个配置分别训练10次，取验证最佳检查点，在不同种子间比较nDCG@10、验证‑测试性能差距和推荐列表相似度；发现种子变化可检测，但对模型选择和性能影响因数据集/模型而异。

**⚠️ 局限性**

仅将种子视为整体随机来源，未分解具体噪声来源；数据划分固定，未探讨划分随机性；未覆盖更多模型或多样化评估指标。

---

## 411. Learning to Track from Privileged Target Appearances

**arXiv ID:** 2609.02471 | [PDF](https://arxiv.org/pdf/2609.02471v1)

**作者:** Xin Chen `[一作]` (City University of Hong Kong), Kede Ma `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在训练阶段利用每帧真实框的目标裁剪（当前帧和未来帧）作为“特权”信息，教导学生模型学习更稳健的搜索表示，从而提升部署时的视觉跟踪性能。

**💡 创新点**

创新点：①将训练时可获取但部署时不可用的目标裁剪作为教师的特权输入；②通过多层搜索表示的预测而非最终框的模仿实现知识迁移；③引入可靠性加权机制，综合教师相对优势和绝对定位质量来调节表示预测损失。

**🔧 技术方法**

技术方法：教师‑学生框架、EMA 训练稳定化、Transformer 表示预测器、相对优势+绝对质量双重权重、基于 OSTrack 的单流 Transformer 追踪器。

**📊 数据集**

训练集：COCO、LaSOT、GOT‑10k、TrackingNet、VastTrack；评估集：LaSOT、LaSOT_ext、TrackingNet、GOT‑10k、TNL2K、NFS、UAV123。

**📈 对比分析**

与多种基准追踪器（SwinTrack、SeqTrack、ARTrack、ODTrack 等）进行对比，主模型在 LaSOT 上 AUC 提升 2.1 点，在 LaSOT_ext 上 1.6 点，TrackingNet、GOT‑10k 也实现了小幅提升；在未训练的 TNL2K、NFS、UAV123 上表现亦优于或接近最强基准，显示出跨数据集的泛化能力。

**⚠️ 局限性**

局限性：需要每帧完整标注的框；特权视图选择固定（当前+未来），未探讨更灵活的采样策略；训练过程额外加入教师与预测器，计算成本增加；在极少标注或低资源场景下的效果未知。

---

## 412. Model-Free Surrogate-Assisted Neural Architecture Search for Evolving Variable-Length Dense Blocks

**arXiv ID:** 2609.02460 | [PDF](https://arxiv.org/pdf/2609.02460v1)

**作者:** Asif Ameer `[一作]` (FAST National University of Computer & Emerging Sciences), Muhammad Fayyaz `[通讯]` (FAST National University of Computer & Emerging Sciences)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 MFSPNet，一种利用无模型代理预测器（VLE‑EMA）和粒子群优化（PSO）演化稠密块的轻量级 NAS 框架，并通过块级加权残差堆叠构建可扩展的 CNN。

**💡 创新点**

创新点包括：①基于验证损失的指数移动平均（VLE‑EMA）作为无模型代理，快速评估架构性能；②块级稠密连接与衰减加权残差堆叠，提升梯度流和跨数据集迁移能力；③在 PSO 里直接搜索稠密块，避免对完整网络的昂贵评估。

**🔧 技术方法**

使用的技术：粒子群优化（PSO）、无模型代理预测（VLE‑EMA）、稠密块编码与块级残差加权聚合、代理数据集（下采样 CIFAR‑10）、轻量化训练配置（FP32、Adam、8×8 解析度）。

**📊 数据集**

实验使用的数据集：CIFAR‑10（代理与搜索），CIFAR‑100、SVHN、ImageNet（迁移与最终评估）。

**📈 对比分析**

与多种 RL、EC、梯度 NAS 方法对比；在 CIFAR‑10 上误差率 3.91%/GPU‑days <3；CIFAR‑100 17.68%，SVHN 1.91%；ImageNet Top‑1/Top‑5 28.29%/12.82%。相比 EffPNet 等方法，误差相近但搜索成本大幅降低（<5 GPU‑days）。

**⚠️ 局限性**

局限性：仅针对 DenseNet 风格的搜索空间；代理数据集依赖 CIFAR‑10，可能对更大或不同分辨率的数据不够充分；对超参数敏感；实验仅在单张 NVIDIA Tesla P40 上验证，缺乏跨平台或更大规模 GPU 的验证；ImageNet 结果仅单次实验，缺乏统计显著性。

---

## 413. LightBridge: Feed-Forward Generative Relighting for 3D Gaussian Splatting

**arXiv ID:** 2609.02543 | [PDF](https://arxiv.org/pdf/2609.02543v1)

**作者:** Hezhi Cao `[一作]` (University of Science and Technology of China), Ligang Liu `[通讯]` (University of Science and Technology of China)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 LightBridge，一种一次性前向生成框架，可在给定光照控制的情况下直接生成重新光照的 3D Gaussian Splatting 资产。

**💡 创新点**

创新点在于将视频扩散模型的潜在桥接方法与稀疏注意力的 Gaussian Propagation Transformer 结合，实现单步、可控、无场景优化的光照编辑。

**🔧 技术方法**

使用视频扩散模型（潜在桥接）、3D Gaussian Splatting 表示、稀疏图像-点注意力的 Gaussian Propagation Transformer、Plücker 坐标编码等技术。

**📊 数据集**

使用自制的 Multi‑Illumination Relighting Dataset，包含 300+ 合成室内场景、约 9,000 种光照条件、54,000 训练视频和 5,580 测试视频。

**📈 对比分析**

与 GR3EN、ScribbleLight、Relit‑LiVE、Light‑A‑Video 等方法对比，LightBridge 在 relit 3DGS 上获得最高 PSNR/SSIM/LPIPS，视频质量排名第二，并将推理时间降低至约 3 秒。

**⚠️ 局限性**

局限性包括：无法处理所有视角都不可见的灯光；受限于合成室内场景的多样性，可能对未见材料或灯具泛化差；仅更新 SH 系数，无法修正几何误差或高度视角相关的材质效应。

---

## 414. Fine-Grained Anomaly Perception in Wild UGC-Enhanced Images: A Comprehensive Dataset and Difference-Fusion Framework

**arXiv ID:** 2609.02529 | [PDF](https://arxiv.org/pdf/2609.02529v1)

**作者:** Yan Zhong `[一作]` (Peking University), Tingting Jiang `[通讯]` (ByteDance Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了UGC图像增强异常感知（UEAP）任务，并构建了首个对应数据集UEAP-4k

**💡 创新点**

提出差异融合异常感知方法DFAP-UGC，结合差异特征、密集Transformer查询、区域验证与质量感知排名，并提出局部感知动态任务优先权策略LADTP

**🔧 技术方法**

使用Swin‑T编码器、Transformer编码器、任务对齐分配（TAL）、密集查询、区域RoI采样、质量预测头以及动态加权学习

**📊 数据集**

使用从公开短视频平台采集的4,222对增强前后图像的UEAP-4k数据集，包含13,662个异常框

**📈 对比分析**

在UEAP-4k测试集上与改造后的Faster R‑CNN、YOLOv1/12n等基线比较，DFAP-UGC实现AP_50 0.433、mAP 0.204、召回率0.811，显著优于基线

**⚠️ 局限性**

局限在于仅针对三类异常，且对极小或纹理模糊的异常识别仍有限；模型复杂度较高，推理速度受限

---

## 415. Rethinking the Teacher-Student Framework for Test-Time Adaptation

**arXiv ID:** 2609.02507 | [PDF](https://arxiv.org/pdf/2609.02507v1)

**作者:** Damian Sójka `[一作]` (Poznan University of Technology), Sebastian Cygert `[通讯]` (National Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了一种在测试时自适应（TTA）中将教师模型固定不更新（Intransigent Teacher）的改进方法，解决了传统EMA教师导致的长序列错误累积与模型崩溃问题。

**💡 创新点**

创新点在于：①通过理论与实验证明EMA教师仅仅延迟错误累积，而不会根本消除；②引入固定教师策略，避免教师权重被学生噪声污染，从而实现长期稳定的自适应；③证明该策略可无缝迁移到多种现有TTA方法（AdaContrast、CoTTA、RoTTA、PETAL）并显著提升性能。

**🔧 技术方法**

主要技术包括：自监督损失（一致性损失、对比学习）配合教师-学生框架；改用β=1的固定教师；误差累积与正反馈分析；在多模型（ResNet、ViT、ConvNeXt等）与多架构上实验验证。

**📊 数据集**

使用的数据集与场景：CIFAR10-C、ImageNet-C、ImageNet-R、DomainNet-126、CCC（长序列）以及对标准测试序列重复20次的长序列版本。

**📈 对比分析**

与原始TTA方法（AdaContrast、CoTTA、RoTTA、PETAL）以及无教师或其他自适应方法（TENT、EATA、SAR、RDUMB、MEMO、PeTTA）进行比较。实验显示，在长序列场景中，Intransigent Teacher 能提升10-20%准确率，显著减少模型崩溃；在小批量和不同网络架构下亦保持稳定性，甚至在某些设置下超过原方法。

**⚠️ 局限性**

局限性：1) 固定教师降低了模型对快速分布变化的塑性，导致在极高学习率或极低批量等极端条件下性能下降；2) 需要进一步探索教师可适应的初始阶段或动态平衡策略，以兼顾塑性与稳定性。

---

## 416. RINSE: Robust Target-Time Normality Estimation for Zero-Shot Graph Anomaly Detection

**arXiv ID:** 2609.02497 | [PDF](https://arxiv.org/pdf/2609.02497v1)

**作者:** Taufikur Rahman Fuad `[一作]` (Islamic University of Technology), Amir Hussain `[通讯]` (King Fahd University of Petroleum & Minerals)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了RINSE，一种零样本图异常检测框架，在目标图上不需要梯度、标签或微调即可估计正常性、校准嵌入并融合多视角证据；

**💡 创新点**

通过在冻结的重构评分器上实现基于残差裁剪的迭代目标正则化、双通道字典统计校准以及可靠性门控的多视角融合，并对不同初始化的编码器进行排名平均，实现了对未见域的稳健自适应；

**🔧 技术方法**

核心技术包括冻结的截断注意力重构评分器、残差裁剪迭代估计、两次字典统计校准、kNN距离、目标字典残差、属性亲和度视角、可靠性门控加权、编码器集成与排名平均；

**📊 数据集**

使用了四个源图（PubMed、CiteSeer、Questions、YelpChi）进行训练和模型选择，八个目标图（Cora、Flickr、ACM、BlogCatalog、Facebook、Weibo、Reddit、Amazon）进行评估；

**📈 对比分析**

与监督基准（BWGNN、GHRN）、无监督基准（DOMINANT、SL-GAD、TAM、CARE）以及一统式基准（ARC、UNPrompt、OWLEYE）对比，RINSE在标准与泄漏自由两种预处理协议下均获得最高平均AUPRC（约39.5%），超越OWLEYE约3-4个百分点，在大多数目标图上排名第一；

**⚠️ 局限性**

局限性包括仅针对节点级属性图；依赖冻结的重构评分器，可能不适用于极大规模图；缺乏因果性保证，异常排名仍可能产生误报，需要人工审核与域特定阈值校准。

---

## 417. Debias-SparseGPT: Bias-Aware Pruning for Large Language Models

**arXiv ID:** 2609.02496 | [PDF](https://arxiv.org/pdf/2609.02496v1)

**作者:** Irina Proskurina `[一作]` (Laboratoire Hubert Curien), Julien Velcin `[通讯]` (École Centrale de Lyon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在压缩大型语言模型时引入了基于对比样本的偏差感知二阶剪枝方法Debias‑SparseGPT，以在保持模型性能的同时降低因剪枝引起的偏见放大。

**💡 创新点**

创新点在于将一对正负性偏见文本的差异项加入剪枝的二阶重构目标，形成偏差感知海森矩阵，既影响掩码选择又影响权重补偿。

**🔧 技术方法**

主要技术包括后训练剪枝、二阶重构优化、偏差感知海森矩阵（包含ΔΔᵀ项）以及块级稀疏化与二阶误差补偿。

**📊 数据集**

使用了StereoSet、UltraChat作为校准数据，评估集包括UnQover、BBQ、CrowS‑Pairs、MMLU、HellaSwag以及WikiText‑2的困惑度。

**📈 对比分析**

与幅度剪枝、Wanda、SparseGPT以及未压缩基线相比，Debias‑SparseGPT在UnQover/BBQ等偏差基准上提升了10–20%（有统计显著性），同时保持与Dense相近的困惑度和零样本下的MMLU/HellaSwag准确率，在所有稀疏率下也实现了更低的DTO（距离最优点更近）。

**⚠️ 局限性**

局限性包括仅在英文单语言环境下评估、未充分覆盖毒性/有害生成等安全维度、对校准数据的敏感性、以及在最严结构化稀疏（2:4）下模型困惑度显著上升，需更丰富的校准文本来缓解。

---

## 418. Rights by Architecture: A Human-Compatible Sociotechnical Layer for Digital Protection Across Regulatory Regimes

**arXiv ID:** 2609.02455 | [PDF](https://arxiv.org/pdf/2609.02455v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 419. MS-MEM: Multi-Skill Manipulation-Enhanced Mapping via Uncertainty- and Disturbance-Aware Action Selection

**arXiv ID:** 2609.02493 | [PDF](https://arxiv.org/pdf/2609.02493v1)

**作者:** Yitian Shi `[一作]` (Karlsruhe Institute of Technology), Maren Bennewitz `[通讯]` (University of Bonn)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在受限、杂乱的货架空间中，提出一种多技能活跃映射框架 MS-MEM，结合主动视角选择、抓取和推送，通过证据推理实现不确定性感知的空间重建。

**💡 创新点**

核心创新包括：①完整的全证据抓取学习（FE-vMF）能够同时建模抓取可行性与方向不确定性；②多视角下的证据融合（FE-UMGF）实现抓取预测随观测不断更新；③统一的DOIG目标和碰撞扰动约束（CDC），使得不同技能的行动可在同一信息增益尺度下比较，并抑制无意义的场景扰动。

**🔧 技术方法**

技术手段涵盖：证据框架下的占据/语义估计、CNABU 网络进行快速贝叶斯更新、基于 vMF 分布的全证据抓取模型、点变换器+MLP 的特征提取、贝叶斯多视角融合、以及基于信息增益的决策算法。

**📊 数据集**

训练与评估数据：利用 Pybullet 生成 4 000+ 受限货架场景，使用 UR5+Robotiq 2F‑85 机器人和 RealSense L515 视觉；真实世界评测在 5 个包含 69 个物体的货架场景中，配合 Vicon 系统获取精确位姿。

**📈 对比分析**

通过与单技能基线（仅抓取、仅推送）、原 MEM、以及去掉 CDC 的 MS-MEM 进行对比。指标包括占据/语义 IoU、位置扰动、物体识别正确率等。实验显示，MS-MEM 在占据与语义 IoU 上优于单技能基线，同时位置扰动显著低于仅推送，且在真实环境中物体识别准确率最高。

**⚠️ 局限性**

局限性：仅验证在受限货架类静态环境；对极端遮挡或动态障碍物的鲁棒性未知；需要大量训练数据和 GPU 资源；决策过程对传感器误差和模型偏差敏感。

---

## 420. Learning to Fuse LLMs with Ontology Rankers for Rare-Disease Diagnosis

**arXiv ID:** 2609.02473 | [PDF](https://arxiv.org/pdf/2609.02473v1)

**作者:** Zhaoyang Jiang `[一作]` (University of Glasgow), Honghan Wu `[通讯]` (University of Glasgow)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

将大型语言模型与传统本体排名器融合，构建行为基融合门，提升罕见病诊断排名的准确性，同时保留候选疾病的本体证据。

**💡 创新点**

1) 设计了只基于两系统行为特征的共享门，自动为每个病例决定对两者的加权比例；2) 通过去除出版源重叠泄漏（LOPO）修正了现有评测偏差；3) 门不依赖目标LLM标签，能够无监督地迁移到新模型。

**🔧 技术方法**

使用 39 维特征向量（患者上下文、列表形状、本体支持、跨列表一致性）训练双层 MLP 作为加权门；对排名采用 reciprocal rank 变换；对候选集做加权融合；在评测时采用 LOPO、留一族群交叉验证。

**📊 数据集**

Phenopacket Store（10,377 病例、780 种疾病）与 HPOA 注解；RAMEDIS（624 病例、74 种代谢病）；多种公开 LLM（Qwen、Llama、MedGemma、HuatuoGPT、Baichuan、DeepSeek 等）。

**📈 对比分析**

与 Phenomizer、单独 LLM、RRF、CombMNZ、Borda 等基线比较。Phenopacket Store 上 Recall@1 从 0.15 提升至 0.20（+5%），RAMEDIS 上提升 20%；在加入 DeepSeek 后，Recall@1 从 0.1657 提升至 0.2176（+5.2%）。此外，门在跨族群迁移时仍保持显著优势。

**⚠️ 局限性**

仅评估基于 HPO 的诊断，未涉及基因或变异信息；依赖病例与注解均记录出版源，LOPO 仅能处理此类数据；门对新模型的泛化受限于疾病名称可映射；未验证临床实际效果。

---

## 421. Batch Before You Time: Decision-Scoped Proxy Execution for Timing-Aware Logic Rewriting

**arXiv ID:** 2609.02470 | [PDF](https://arxiv.org/pdf/2609.02470v1)

**作者:** Pujun Su `[一作]` `[通讯]` (Fudan University), Pujun Su (Fudan University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种批量化的代理执行方法（BBYT），将同一逻辑重写决策的所有候选方案一次性编译为零延迟镜像，仅保留一次代理评估；

**💡 创新点**

创新点在于：①将代理执行范围从单个候选扩展到整个重写决策；②利用相对差距门控决定是否直接采用代理结果；③通过一次性编译/回放实现省时而不牺牲精度；

**🔧 技术方法**

使用的技术包括：SDF注解的时间模拟、Yosys/ABC合成与等价检查、OpenSTA时序分析、Icarus Verilog零延迟回放、相对差距门控算法、批量化多候选打包与事件流解包；

**📊 数据集**

评估数据集涵盖ISCAS85与EPFL组合基准，11个电路（C6288、EPFL-i2c、EPFL-adder、EPFL-max、EPFL-cavlc等），共250个重写决策、532个候选；

**📈 对比分析**

对比方法：完整时序评估、按候选逐个代理执行、BBYT；结果显示BBYT在12个平衡对照序列中平均缩短完整候选选择时间18.05%，在受控工作量（8192跳）下相较于按候选代理执行提升9.24%；BBYT消除32.52%的时序候选评估，并在所有250次决策中与完整评估一致；

**⚠️ 局限性**

局限性：仅适用于按需生成代理且每个决策的候选数足够多时才显著；对门控阈值依赖性高，阈值固定为0.13；在候选数极少时打包优势有限；未针对并行或多核部署做进一步优化；

---

## 422. World-Model-Augmented Visual Locomotion for Humanoids on Foothold-Constrained Terrain

**arXiv ID:** 2609.02542 | [PDF](https://arxiv.org/pdf/2609.02542v1)

**作者:** Yuxi Liu `[一作]` (D-Robotics), Wei Sui `[通讯]` (D-Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在单目深度相机与自我感知信息的条件下，联合训练一个递归世界模型和PPO策略，使机器人能够在踏板、楼梯和缝隙等步态受限地形上行走。

**💡 创新点**

创新点在于使用无标注的递归世界模型生成预测特征，为策略提供前瞻信息，避免仅依赖当前观测导致的误踏；并在训练中整合多种地形奖励形状，而非单纯的分阶段或蒸馏方法。

**🔧 技术方法**

采用RSSM递归状态空间模型、PPO强化学习、Mixture‑of‑Experts网络、深度摄像头与惯性测量的联合感知、以及多项奖励形状（边界惩罚、踏板穿透、序列踏步奖励等）。

**📊 数据集**

通过Isaac Lab仿真平台生成的程序化地形（楼梯、缝隙、踏板）进行训练和评估；硬件测试使用一台配备Jetson Orin的Humanoid机器人。

**📈 对比分析**

与标准PPO基线在相同奖励与感知条件下对比，WM‑LOCO在缝隙和踏板场景下成功率从0%提升至78%–100%，在楼梯场景保持相近成功率并提升步态效率、降低臀部加速度；硬件实验同样显示高成功率。

**⚠️ 局限性**

仅针对离散踏板或窄几何约束的地形，未涵盖连续细支撑、可变高度踏板、户外不规则地形；实验仅在单一Humanoid模型和RSSM架构上验证，未测试其他机器人或更复杂地形。

---

## 423. The Price of Almost Navigability

**arXiv ID:** 2609.02498 | [PDF](https://arxiv.org/pdf/2609.02498v1)

**作者:** Tomer Waizer `[一作]` (Technion Israel Institute Of Technology), Yoav Danieli `[通讯]` (Technion Israel Institute Of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过建立全新的全局计数法则，证明了(1-ε)-几乎可导航图的平均出度必须为Ω(1/ε)（当ε较大时）或Ω(√n)（当ε趋近0时），并给出了完整的稀疏度相位图；

**💡 创新点**

其创新点在于将几乎可导航性与Zarankiewicz问题直接对应，利用随机球面与Füredi有限域构造生成低维欧氏实例，进而得到与已知上界完全匹配的下界；

**🔧 技术方法**

主要技术包括全局计费定理、球面随机截头、欧氏距离矩阵的构造与投影（Johnson–Lindenstrauss）、有限域图的谱压缩以及与极值图论的桥接；

**📊 数据集**

实验和理论证明所用数据集为在高维球面上独立均匀采样得到的通用欧氏点集，以及将Füredi构造的图嵌入欧氏空间得到的特定点集；

**📈 对比分析**

与已公布的O(1/ε)与O(√n)上界直接比较，证明其紧密匹配，进一步指出完全可导航图至少需要Ω(n^{3/2})条边；

**⚠️ 局限性**

限制主要在于：在小误差区间（ε≈0）所需的欧氏维度高达O(√n log n)，尚未能在多项式或常数维空间中实现完整相位。

---

## 424. A meshfree solver for coupled bulk-surface problems with self-organizing surface geometry

**arXiv ID:** 2609.02489 | [PDF](https://arxiv.org/pdf/2609.02489v1)

**作者:** Lennart J. Schulze `[一作]` (Dresden University of Technology), Ivo F. Sbalzarini `[通讯]` (Dresden University of Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种完全无网格的数值求解器，用于耦合的体-面偏微分方程，能自适应跟踪和解决可变形表面与周围流体的耦合动力学；

**💡 创新点**

创新点在于将半隐式表面表示（PCP+DC‑PSE）与自适应粒子重采样（SAISS）相结合，形成高阶、稳定且可并行的 Lagrangian 方案，首次实现表面张力、弯曲度、浓度场等多物理量的耦合计算；

**🔧 技术方法**

使用的技术包括粒子最近点（PCP）法求高阶几何量、表面 DC‑PSE 差分、基于 SPH 的多相 Navier‑Stokes 求解、预测-校正时间积分以及 OpenFPM 并行框架；

**📊 数据集**

所用“数据集”为合成测试案例：单位球面、均匀生长球面、黏性振荡液滴以及基于 Gray‑Scott 反应扩散的形态发生模拟；

**📈 对比分析**

通过与解析解对比验证收敛性，并将滴面 Gray‑Scott 模式与二维平面结果比较，显示相似的 Turing 模式；数值性能上，SAISS 规整化粒子显著提高 PCP 的求解速度，整体模拟时间减少约 20‑30%；

**⚠️ 局限性**

主要限制包括：表面 DC‑PSE 需要无交叉管道邻域，限制了大变形后的模拟时间；缺乏表面浓度对流体的反馈（如 Marangoni 效应）；以及多尺度邻域管理和自适应支持半径实现难度较大；

---

## 425. ViSAR: Training-Free Adaptive-$k$ Retrieval for Visual Document Question Answering

**arXiv ID:** 2609.02486 | [PDF](https://arxiv.org/pdf/2609.02486v1)

**作者:** Adrien Mialland `[一作]` (INSA Lyon), Céline Robardet `[通讯]` (INSA Lyon)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 ViSAR，一种无需训练的视觉语义激活检索方法，可在文档视觉问答（DocVQA）中根据查询自适应地选择相关页面。

**💡 创新点**

其创新点在于利用多向量 late‑interaction 嵌入空间构建查询条件的页面相似矩阵，并基于该矩阵进行自适应 k 检索，从而显著降低大型视觉语言模型（LVLM）的上下文负载。

**🔧 技术方法**

所用技术包括 late‑interaction 与 MaxSim 交互、激活权重和页面权重的自适应计算、页面间相似矩阵的构建，以及自适应 k 的成本函数优化。

**📊 数据集**

实验在两个 OCR‑free 文档问答数据集——MMLongBench 和 LongDocURL 上进行。

**📈 对比分析**

与固定 top‑k、Largest‑Gap、Score‑Cluster 以及 Oracle 等方法对比，ViSAR 在保持或提升答案准确率的同时，能将 RAG 延迟降低高达 58.7%，并在多种编码器（ColPali、ColQwen2.5、ColModernVBERT）和 LVLM（Qwen2.5‑VL‑7B‑Instruct 等）上均表现优异。

**⚠️ 局限性**

局限性包括检索阶段计算开销随文档尺寸增长，极大文档（数百页以上）时仍需近似方法以保持速度；此外，对较小的编码器（如 ColModernVBERT）改进空间有限。

---

## 426. How LLMs Build Fictional Worlds: Setting and Narrative Space in AI-Generated Creative Storytelling

**arXiv ID:** 2609.02482 | [PDF](https://arxiv.org/pdf/2609.02482v1)

**作者:** Katrin Rohrbacher `[一作]` (FAU Erlangen Nurnberg), Michaela Mahlberg `[通讯]` (University of Birmingham)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

分析大型语言模型（LLM）在叙事世界构建中对空间维度的使用，聚焦于场景构造；

**💡 创新点**

提出了基于叙事理论的五类空间（行动空间、感知空间、视觉空间、描述空间、无空间）框架，并将其扩展到英语，并系统比较不同LLM与人类文本的空间分布差异；

**🔧 技术方法**

使用了微调的BERT空间分类器（对德语原模型进行英语迁移），以及基于OpenAI GPT‑4.1、LLaMA‑3.3、Mistral‑3.2、Gemma‑3的长文本生成；

**📊 数据集**

数据集为8000条AI生成长篇（每模型1000条，英德双语）与约4000条Project Gutenberg人类原文的对照；

**📈 对比分析**

通过比例统计、GLMM和置信区间对比，发现LLM普遍过度生成“感知空间”，行动空间不足，差异在所有模型、时间段和语言上均显著，GPT‑4.1偏差最大；

**⚠️ 局限性**

局限包括：长文本生成不稳定、分类器在Mistral‑3.2上的表现偏差、仅覆盖1900年前公共领域文本、未对现代或其他语言的泛化、并且仅使用单句启动，可能无法捕捉完整的叙事风格。

---

## 427. CivBench: A Long-Horizon Benchmark for Tool-Mediated Agents in Civilization VI

**arXiv ID:** 2609.02459 | [PDF](https://arxiv.org/pdf/2609.02459v1)

**作者:** Austin Tudor David Andrews `[一作]` (University Of Oxford), Rui Ponte Costa `[通讯]` (University Of Oxford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了CivBench，一个基于Model Context Protocol（MCP）的长期、工具驱动的文明VI评测基准，能够通过工具调用与游戏互动并记录完整日志；

**💡 创新点**

创新点包括：①使用叙述层将视觉状态转换为结构化文本，同时仅通过显式查询获取信息，显著提升信息检索可测性；②引入两项接口层行为指标——Proactive Monitoring Rate（PMR）和Reflection–Action Gap（RAG@K），直接量化代理的监控与计划执行；

**🔧 技术方法**

技术实现涵盖：MCP接口与FireTuner协议连接文明VI，工具调用系统（76个工具），叙述层文本结构化，日志分析管道，PMR与RAG@K指标计算；

**📊 数据集**

数据集为23场可接受完整游戏的日志，包含四类模型在“Ground Control”和“Snowflake”两个场景下的游戏记录；

**📈 对比分析**

比较方法主要是行为层指标：对不同模型族进行PMR和RAG@10的统计，发现所有模型的监控率低且计划执行率在48-66%之间；整体结果未能显著区分模型性能，仅体现行为模式；

**⚠️ 局限性**

局限性：样本量小且分布不均，缺乏随机基线，指标受叙述层与 playbook 约束影响，未能完整评估模型策略质量，且只关注可见工具调用，未涵盖更广泛的自适应策略。

---

## 428. Adapting a Foundation Model for Lunar Surface Height Estimation

**arXiv ID:** 2609.02448 | [PDF](https://arxiv.org/pdf/2609.02448v1)

**作者:** Patrick Bauer `[一作]` (University of Technology of Troyes), Hichem Snoussi `[通讯]` (University of Technology of Troyes)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

针对月球表面高度估计，微调DAV2模型实现相对高度预测

**💡 创新点**

使用LoRA参数高效微调以及融合多种损失函数，克服域差距提升DEM质量

**🔧 技术方法**

Transformer编码器+DPT解码器、LoRA、Berhu、梯度匹配和法向损失

**📊 数据集**

LRO获取的基于SPG的2m/px DEM及对应正射影像，约84k训练、11k验证、13k测试图块

**📈 对比分析**

与零射手DAV2对比，MAE从11.64降至4.71，RMSE从14.13降至5.76，性能显著提升

**⚠️ 局限性**

推理速度慢，无法实时应用；仅提供相对高度，缺乏绝对高度；仅适用于月球，难以迁移到其他天体

---

## 429. Specification-Guided Path Shortcutting for Efficient Probabilistic Model Checking

**arXiv ID:** 2609.02457 | [PDF](https://arxiv.org/pdf/2609.02457v1)

**作者:** Tsubasa Matsumoto `[一作]` (Kyoto University), Masaki Waga `[通讯]` (Kyoto University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对Markov链和ω-正则属性的规格指导路径捷径抽象方法，在保持满足概率不变的前提下，减少状态空间并提升概率模型检验效率。

**💡 创新点**

创新点在于引入路径可替换性和可消除状态概念，并通过最小兼容后缀（MCCS）构造实现精确的抽象，突破传统二分歧最小化的限制，首次在产品构造前完成属性驱动的概率保持抽象。

**🔧 技术方法**

使用Markov链模型、LTL到确定性Rabin自动机转换、路径可替换性分析、MCCS构造、Storm/PRISM概率模型检查器以及Spot自动机库进行实现和评估。

**📊 数据集**

在QComp基准集的多种模型（包括bounded retransmission、Crowds、Contract signing、Leader election、NAND multiplexing等）以及15个不同的LTL公式上进行实验验证。

**📈 对比分析**

通过30次实验比较三种工作流（无抽象、抽象+Storm、抽象+Storm+Bisim）记录平均时间、状态数和转移差异；实验表明抽象+Storm在多数情况下比基线快，尤其在状态空间大或公式复杂时可获得多达18×的速度提升。

**⚠️ 局限性**

局限性包括：在某些公式下抽象可能导致转移数膨胀或额外开销；目前仅适用于Markov链与Rabin自动机，未扩展到非确定性或MDP；MCCS上界的选取仍需经验调优，且对某些模型的效果有限。

---

## 430. Source Distribution Estimation by Posterior Averaging

**arXiv ID:** 2609.02622 | [PDF](https://arxiv.org/pdf/2609.02622v1)

**作者:** Trung-Dung Hoang `[一作]` (University of Bern), Lisa M. Koch `[通讯]` (University of Bern)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `40105733-5154-44cd-8090-a8cab9e64b07` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过期望最大化的迭代方法（Posterior Averaging）在黑盒模拟器环境下估计源分布。

**💡 创新点**

直接逼近自洽条件，消除固定代理似然的误差，并证明每轮KL递减。

**🔧 技术方法**

使用神经后验估计（NPE）与可逆流（normalizing flows）实现源与后验，提供两种变体（独立流与共享流）。

**📊 数据集**

在三类模拟器（Two Moons、SLCP、Lotka–Volterra）上，用人工生成的 10,000 条真实观测数据进行实验。

**📈 对比分析**

与 NEB、Sourcerer 及其迭代版相比，PA 与 PA‑Shared 在多种初始先验下的数据空间 C2ST 分数明显更优，尤其在 Lotka–Volterra 上从 0.96+ 降至 0.64–0.68。

**⚠️ 局限性**

多源可导致同一边缘分布，评估仅关注数据空间匹配；E/M 步为近似；实验仅在合成任务，未验证在真实数据或高维情形下的表现。

---

## 431. Competitive Market Behavior of LLMs

**arXiv ID:** 2609.02580 | [PDF](https://arxiv.org/pdf/2609.02580v1)

**作者:** Pawel Struski `[一作]` (University of Warsaw), Przemyslaw Biecek `[通讯]` (University of Warsaw)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将大型语言模型（LLM）作为市场主体，重现经典双拍卖实验，构建可复现的实验框架，评估LLM在不含人类干预的双拍卖中是否能自发收敛到竞争均衡，并对其交易行为进行定量与语言分析。

**💡 创新点**

①开源可复用的双拍卖实验框架；②以历史人类实验（Smith 1962）为直接基准，量化市场对齐程度；③首次将LLM内部推理（CoT）与交易行为相结合，揭示“增量改价”与“成交冲动”之间的语言转变；④发现最小模型（GPT‑small）在效率上优于更大模型，挑战传统规模效应。

**🔧 技术方法**

使用大型语言模型API（OpenAI GPT‑5.4‑large/mini、Google Gemini‑3.1‑pro-preview）进行交互；实现持续订单簿、随机抽样交易顺序、价格收敛与效率统计；利用词频统计与加权对数几率（Dirichlet prior）对CoT文本进行主题分离；代码托管于GitHub。

**📊 数据集**

实验自生成的数据：10次独立实验 × 10个随机种子，包含每轮交易价格、成交数、价格波动α、allocative efficiency等；对照历史人类实验数据（Smith 1962）与理论均衡（价格$2.00，成交量6）。

**📈 对比分析**

与人类实验基准按价格波动α、成交数、allocative efficiency三指标对比。结果显示：LLM市场均未完全收敛，价格波动远高于人类（α>20% vs 3–4%），成交数不足均衡量（≤6）且效率最高仅0.91；GPT‑small表现最接近人类，但仍不及人类基准。

**⚠️ 局限性**

①仅评估三种模型且未系统探讨规模、温度等参数；②实验设置仅为单一商品、固定需求供给阶梯，缺乏多样化市场环境；③CoT分析仅对Gemini展开，无法推广；④未对LLM可能的协同或策略性操纵进行深入检验；⑤实验时间限制（300次迭代）可能导致部分模型未能完成收敛，影响结论。

---

## 432. RGB-to-IR image translation for infrared vehicle detection in unseen UAV domains

**arXiv ID:** 2609.02556 | [PDF](https://arxiv.org/pdf/2609.02556v1)

**作者:** Thijs A. Eker `[一作]` (TNO - Intelligent Imaging), Friso G. Heslinga `[通讯]` (TNO - Intelligent Imaging)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用RGB到IR图像翻译生成合成的红外训练数据，并在未见的无人机红外数据集上提升车辆检测性能

**💡 创新点**

构建跨域多源RGB-IR对齐数据集，比较GAN、控制网络扩散、LoRA等多种翻译方法，并探究随机种子与数据集特定提示对检测效果的影响

**🔧 技术方法**

基于Stable Diffusion 3.5/FLUX ControlNet的扩散翻译、pix2pixHD的监督式翻译、FLUX-LoRA的低秩适配以及灰度基线，使用RF-DETR进行车辆检测训练和评估

**📊 数据集**

源域：DroneVehicle、Caltech、M3OT；目标域：Kust4K（近域）、VTUAV（远域）

**📈 对比分析**

在源域红外训练基础上加入合成红外后，Kust4K mAP从50.8提升到60.0（+9.2），VTUAV从25.6提升到40.6（+15.0），最优结果来自SD‑ControlNet+数据集提示；若加入真实目标红外，mAP进一步提升至67.5/52.3

**⚠️ 局限性**

红外外观无法仅凭RGB确定，翻译模型易产生空间不一致与热特征缺失；源域数据偏倚导致泛化受限；缺乏多样化红外样本和对不同传感器、环境条件的显式建模

---

## 433. H3DNAS: Hardware-Aware ONNX-Native 3D Point Cloud Model Compression

**arXiv ID:** 2609.02684 | [PDF](https://arxiv.org/pdf/2609.02684v1)

**作者:** Anchit Mulye `[一作]` (Indian Institute of Technology Jodhpur), Hardik Jain `[通讯]` (Indian Institute of Technology Jodhpur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8d10c613-917e-4880-9716-17789f50e119` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 H3DNAS，一种完全基于 ONNX 计算图的硬件感知模型压缩框架，能够在不访问原始源代码、梯度或训练框架的情况下压缩 3D 点云网络。

**💡 创新点**

创新点在于：① 引入 Channel Dependency Graph（CDG）理论，证明自由参数比例 ρ_f 为拓扑不变量，可在搜索前预估压缩上限；② 设计两阶段层级搜索，先用 L1 通道重要性和输出相似度进行零样本预筛选，再应用 GhostConv 结构变异；③ 完全实现源代码无关的端到端压缩流水线，仅通过 ONNX 图操作。

**🔧 技术方法**

主要技术包括：ONNX 图形解析与形状推理；CDG 构建与约束规则推导；L1 通道重要性排序与 BN 对齐；输出相似度（logit 余弦相似度）零样本评估；GhostConv 结构替换；OnnxRuntime 与 TensorRT 进行硬件性能评估。

**📊 数据集**

使用 ModelNet40 数据集进行点云分类实验，评估 PointNet、PointNet++ SSG 和 PointMLP 三种 3D 网络。

**📈 对比分析**

与 CP3、HRank、LTH、T3DNet、HLS4PC 等方法对比，H3DNAS 在三种网络上分别实现了 65.5%、43.2% 及 49.1% 的参数压缩，准确率下降不超过 0.28pp，推理速度提升 1.99×、1.29×、1.67×；在 NVIDIA Jetson Orin Nano 8GB 设备上自动满足 4B FLOPs、20M 参数、50MB 模型、50ms 延迟等硬件预算。

**⚠️ 局限性**

局限性包括：CPU OnnxRuntime 延迟评估与真实 GPU 性能的相关性有限；GhostConv 仅实现第一级结构变异，二级结构变异待进一步研究；对某些自定义或深度可分离卷积的 ONNX 操作支持仍有不足；框架不支持梯度型 NAS（如 DARTS）。

---

## 434. Genesis: A Generative Engine for Hierarchical Satellite Image Synthesis

**arXiv ID:** 2609.02683 | [PDF](https://arxiv.org/pdf/2609.02683v1)

**作者:** Subash Khanal `[一作]` (Washington University in St. Louis), Nathan Jacobs `[通讯]` (Washington University in St. Louis)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种从稀疏种子图块生成完整多尺度卫星图像金字塔的任务——多尺度图块补全；

**💡 创新点**

创新点在于构建统一的生成引擎，将竖向超分辨率与横向遮罩化填补两种专门化的生成器通过四叉树结构组合，实现跨尺度和跨空间一致的完整金字塔；

**🔧 技术方法**

使用基于流匹配的像素空间Transformer（JiT）分别训练超分辨率和填补模型，并通过交叉注意力引入父图块特征和地理标签；

**📊 数据集**

训练数据为Git‑10M卫星图像（10–18级），并发布了500个完整深度4四叉树的Genesis基准数据集；

**📈 对比分析**

与FastDiffSR、SwinIR、ZoomLDM、SD2‑Inpaint等现有方法比较，生成模型在F​​ID、LPIPS、PSNR、SSIM、RAPSD等指标上均显著优于对照组，单体模型在子任务上达到SOTA；

**⚠️ 局限性**

局限性包括单个种子场景时生成漂移较大、对复杂大面积缺失仍需大量推理步骤，以及模型训练和推理成本较高。

---

## 435. Projective Affine Body Dynamics for Multibody Systems

**arXiv ID:** 2609.02675 | [PDF](https://arxiv.org/pdf/2609.02675v1)

**作者:** Zimeng Ye `[一作]` (Chinese Academy of Sciences), Hongan Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一种基于投影的仿射体动力学方法，实现了在GPU上实时求解具有非线性约束、接触与摩擦的多体动力学系统。

**💡 创新点**

创新点在于：将多体动力学转化为变分框架，采用仿射体和等距约束构成的等价投影模型；引入半隐式递归代换法（SISSM）实现无Hessian求解；利用仿射体 12×12 块矩阵的稀疏结构，显著降低GPU内存与运算开销。

**🔧 技术方法**

使用技术包括：仿射体动力学（ABD）、投影动力学、半隐式递归代换法、GPU并行实现、稀疏矩阵逆（块分解）、BFGS/ADMM、Coulomb摩擦模型、BVH加速的碰撞检测。

**📊 数据集**

主要使用自制仿真场景作为数据集：四足机器人在沙地、风车撞击盒子、桥梁穿越、帆船与水体耦合、沙地耦合以及 50×50×50 块堆砌等，未引用公开数据集。

**📈 对比分析**

与 PhysX 顺序冲击法和 AVBD 进行对比；在相同硬件（NVIDIA RTX 3080）下，帧率约为 40‑55 FPS；相比 Newton barrier 方法速度快 10 倍以上；在极大质量比等极端情形下，方法保持稳定，而 PhysX 会失稳。

**⚠️ 局限性**

局限性：在大旋转步长、极高阶多项式能量分解复杂时易失效；难以处理流体或高度可变形对象；在极端条件下仍可能出现数值不稳定。

---

## 436. Microcell Hot Spot and Smart Antennas Evaluation in WCDMA Macrocell System

**arXiv ID:** 2609.02636 | [PDF](https://arxiv.org/pdf/2609.02636v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 437. Query Rewriting for Complex Object Segmentation in 4D Gaussian Representations

**arXiv ID:** 2609.02664 | [PDF](https://arxiv.org/pdf/2609.02664v1)

**作者:** Thanh-Khoi Nguyen `[一作]` (University of Science), Minh-Triet Tran `[通讯]` (University of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并实现了一种无训练的多级关键词重写框架，用于提升4D高斯表示中对复杂、冗长自然语言查询的语言引导分割效果。

**💡 创新点**

提出在查询重写中加入关键词注入与结构化重写，解决传统模型对冗长查询的语义鸿沟问题，并通过无训练方式实现对时空检索的显著提升。

**🔧 技术方法**

基于4D LangSplat框架，结合SAM做空间分割、CLIP做视觉语义编码、MLLM（Gemini、Qwen2‑VL‑7B）进行实例级视频描述，Llama3‑8B进行关键词提取与重写，最终映射到共享的视觉语言嵌入空间。

**📊 数据集**

在HyperNeRF和Neu3D这两个公开的4D动态场景数据集上进行评估，并使用Gemini 3.0生成模拟真实用户的复杂查询。

**📈 对比分析**

通过与4D LangSplat和4DLangVGGT基线在vIoU、Temporal Accuracy、mIoU等指标上对比，重写后时空准确率从约60%提升至92%，vIoU从约20%提升至77%，mIoU也显著提升，证明重写方法有效。

**⚠️ 局限性**

局限性包括：依赖关键词重写仍无法处理极度模糊或不具备关键词的查询；缺乏自动关键词发现与自适应重写机制；在不同语言或更大规模多模态数据集上的泛化性尚待验证。

---

## 438. Characterizing Text Branch Sensitivity in Medical Vision-Language Segmentation via Evidence Decoupling

**arXiv ID:** 2609.02663 | [PDF](https://arxiv.org/pdf/2609.02663v1)

**作者:** Ziquan Liu `[一作]` (Southwest University of Science and Technology), Xuyang Shi `[通讯]` (Southwest University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

系统评估了多模态视觉‑文本分割模型中文本分支对像素级预测的贡献，并提出了可解释的Evidence Decoupling Decoder (EDD)；

**💡 创新点**

创新点在于①证明不同融合模块对性能影响微乎其微，②通过EDL和深度监督实现对图像与文本证据的逐层分解；

**🔧 技术方法**

采用Evidential Deep Learning、深度监督、四种文本‑视觉融合方式（FiLM、交叉注意、通道门控、拼接）和多种VLM预训练模型；

**📊 数据集**

使用四个配对文本的医学影像数据集（BUSI、BTMRI、Kvasir-SEG、ISIC）；

**📈 对比分析**

通过对比不同VLM、融合模块和EDD与标准解码器的Dice/IoU，发现EDD在保持近似性能的同时提供了证据分解；在文本扰动实验中，BTMRI和BUSI对文本高度敏感，ISIC和Kvasir-SEG对文本依赖较弱；

**⚠️ 局限性**

局限在于文本贡献被视为对图像特征的全局调制，缺乏真正的空间文本独立路径；对不同文本组件的影响机理尚未深入；

---

## 439. Unfolding the Leech Lattice: Fused Multi-Shell Decoding and VRAM Layouts for 2-Bit LLM Weights

**arXiv ID:** 2609.02652 | [PDF](https://arxiv.org/pdf/2609.02652v1)

**作者:** Pier-Jean Malandrino `[一作]` `[通讯]` (Scub), Pier-Jean Malandrino (Scub)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了Leech晶格向量量化（LLVQ）完整301类多壳解码器，并在GPU上测评其在2-bit GEMV阶段的服务成本。

**💡 创新点**

首次公开实现了完整多壳Λ_24(12)代码簿的无分支解码器，提出四比特VRAM布局，比较4-bit/2-bit GEMV字节效率与时间成本。

**🔧 技术方法**

使用CUDA融合解量化-矩阵向量核、Walsh–Hadamard旋转、位平面/一热掩码布局、代码簿展开以及DRAM/GPU内存层次优化。

**📊 数据集**

采用Qwen3-4B/8B/14B LLM权重，WikiText-2（4096上下文）与MMLU 2280题进行质量评估。

**📈 对比分析**

在同一进程同一L40S GPU上与FP16基准、4-bit AWQ、2-bit QTIP对比，Planes14比QTIP快2.27×但字节读多；4-bit方案吞吐提升至约2×，整体在512 B内存上达到87 tok/s。

**⚠️ 局限性**

局限性：仅在batch 1、单GPU L40S环境评估；缺乏多批量和70B大模型验证；解码器展开成本高、共享内存旋转受限，且MMLU推理质量缺失原因未完全排除。

---

## 440. Recommender System as Slow and Fast Thinkers

**arXiv ID:** 2609.02671 | [PDF](https://arxiv.org/pdf/2609.02671v1)

**作者:** Zichen Yuan `[一作]` (City University of Hong Kong), Junchen Fu `[通讯]` (University of Glasgow)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 DS-Frame，一种双路快慢推理框架，针对序列推荐中的用户异质性动态分配计算预算。

**💡 创新点**

将快速路径与迭代式慢路径结合，并用轻量级选择器根据样本的预期收益在两者间动态路由，实现适应性推理。

**🔧 技术方法**

采用 Transformer 编码器作为共享序列表示，Fast 系统为单次前向推理，Slow 系统为多步隐状态细化，Selector 通过 oracle 导向监督与预算正则化学习。

**📊 数据集**

在五个真实序列推荐数据集上评测：Yelp 与 Amazon 四个领域（Video Games、Beauty、Sports、Toys）。

**📈 对比分析**

与原始 Backbone（SASRec、BERT4Rec）及多种 reasoning‑enhanced 基线（ReaRec、STREAM‑Rec、LARES、ManCAR）进行对比，平均提升 NDCG@10 约 6–7%，在困难用户组提升更显著，且在控制慢路比例时能实现优越的精度-效率折中。

**⚠️ 局限性**

Slow 系统固定步数缺乏自适停顿，选择器仅利用最终表示，未加入不确定度信息，且 oracle 标签依赖训练集真实标签，迁移到新分布时鲁棒性未知。

---

## 441. Physics-Driven Independent Pair Generation for Iterative Self-Supervised Low-Dose CT Denoising

**arXiv ID:** 2609.02654 | [PDF](https://arxiv.org/pdf/2609.02654v1)

**作者:** Xianlei Han `[一作]` (Nanchang University), Qiegen Liu `[通讯]` (Nanchang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研发了一种自监督低剂量CT去噪框架，利用物理驱动的光子计数后验推断与投影-图像域迭代，构造近似独立的训练对

**💡 创新点**

创新点在于：①将Poisson–Gaussian噪声模型与光子计数后验联合使用，生成可独立的双分支；②采用二项与高斯数据稀释并进行残差尺度匹配；③通过跨域迭代将投影先验与图像恢复结果互相更新，提升物理一致性

**🔧 技术方法**

主要技术包括：后验推断、binomial/高斯稀释、残差尺度匹配、U‑Net网络、投影与反投影运算、投影-图像跨域迭代

**📊 数据集**

使用的数据集为：模拟数据（AAPM、LIDC‑IDRI、LoDoPaB‑CT）和真实低剂量CT（GE临床心脏CT、小鼠CT）

**📈 对比分析**

与FBP、BM3D、Noise2Void、Noise2Noise、Blind2Unblind、Noise2Sim、N2N‑BS以及监督RED‑CNN比较；在所有模拟与真实数据上均实现或超过监督基线，PSNR、SSIM显著提升，RMSE显著下降

**⚠️ 局限性**

局限性包括：对理想Poisson–Gaussian模型的依赖，散射、硬化等实际失真可能破坏独立性；实现仅限二维；后验近似与先验误差导致训练对独立性受限；跨域迭代需要多轮计算，计算成本较高

---

## 442. Differentiable Electricity-Market Clearing for Gradient-Based Planning

**arXiv ID:** 2609.02646 | [PDF](https://arxiv.org/pdf/2609.02646v1)

**作者:** Luca Mungo `[一作]` (Macrocosm Inc), Arnau Quera-Bofarull `[通讯]` (Macrocosm Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过把电力市场清算建模为可微分优化层，利用逆向自动微分获得规划目标相对于数据中心负载分配的梯度，并用梯度下降搜索，得到接近最优的站点选择与负载分配方案。

**💡 创新点**

创新点在于：①将市场清算嵌入可微分优化层，首次实现了直接对节点价格梯度的反向传播；②使用平滑站点计数近似处理离散的站点开启/关闭决策，允许在梯度优化中同时优化位置与规模。

**🔧 技术方法**

使用的技术包括可微分优化（隐式微分 + 逆向自动微分）、Adam梯度优化器、softmax平滑化站点计数、线性规划求解器以及对多运行状态的批量处理。

**📊 数据集**

实验数据来自两套合成电网：Erdős–Rényi 随机连通图和 GeoDe 地理稀疏网；均为12个节点，6个候选站点，36个运行状态，目标是分配总负载50 MW。

**📈 对比分析**

与对所有非空站点组合的穷举枚举得到的数值最优解相比，梯度优化在 Erdős–Rényi 实例中最大目标误差为 0.023 ΔE（相当于最佳与最差单站点成本差的 2.3%），在 GeoDe 实例中为 0.085 ΔE（约 8.5%），显示出高度接近最优的性能。

**⚠️ 局限性**

局限性包括：①平滑站点计数在靠近站点切换阈值时会逐步收缩站点而不是及时关闭，导致支持切换延迟；②实验仅在规模较小（6 个候选站点、36 个运行状态）的合成实例上验证，缺乏对大规模真实系统的评估；③对初始化、局部最优与计算成本的影响尚未深入研究。

---

## 443. Learning to Attract and Repel: Dual Quality Margin Learning for Face Recognition (DQM-Face)

**arXiv ID:** 2609.02644 | [PDF](https://arxiv.org/pdf/2609.02644v1)

**作者:** El Ouanas Belabbaci `[一作]` (University of Sciences and Technology Houari Boumediene), Philipp Terhörst `[通讯]` (Johannes Gutenberg University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种双质量边距学习框架 DQM‑Face，结合特征幅值与语义注意力进行质量估计，并在损失中加入自适应吸引边距与显式排斥边距，实现更紧凑且分离度更高的面部特征空间。

**💡 创新点**

创新点在于：①将幅值质量与语义质量（SE注意力）融合，获得更鲁棒、身份感知的质量评分；②设计双边距（吸引+排斥）策略，既自适应提升高质量样本的聚类，又显式拉开不同身份之间的角距离；③在质量学习与特征学习耦合后，所学质量能直接用于面部图像质量评估。

**🔧 技术方法**

使用 iResNet‑100 作为主干网络，SE 语义注意力模块，基于 ArcFace 结构的双边距损失（含 MagFace 归一化正则），并在训练中采用梯度下降、学习率衰减和 3 阶段排斥边距调度。

**📊 数据集**

训练数据为 MS1MV2（≈580 万张，85k+身份），评估数据包括 LFW、CFP‑FP、AgeDB‑30、CPLFW、IJB‑B、IJB‑C，用于面部识别；LFW、Adience、CPLFW、XQLFW 用于面部图像质量评估。

**📈 对比分析**

与 ArcFace、CosFace、MagFace、AdaFace、ElasticFace、CoReFace 等多种主流方法对比，DQM‑Face 在所有图像和视频基准上均获得第一或第二名（例如 IJB‑B/TAR@FAR=10⁻³ 达 96.71%，IJB‑C/TAR@FAR=10⁻⁵ 达 94.78%），并在 FIQA 评估中获得最低 pAUC，表明质量估计效果优于现有方法。

**⚠️ 局限性**

局限性包括：①质量估计高度依赖特定网络结构，缺乏通用性；②排斥边距的固定调度可能不适用于所有数据分布；③对极端模糊或遮挡的样本仍存在一定误判，尤其在低分辨率视频帧上。

---

## 444. From Detection to Localization: A Unified Forensics Framework for Fully Synthetic and Tampered Images

**arXiv ID:** 2609.02640 | [PDF](https://arxiv.org/pdf/2609.02640v1)

**作者:** Annalisa Gallina `[一作]` (University of Padova), Lamberto Ballan `[通讯]` (University of Padova)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种统一的多类别图像真实性与伪造检测与定位框架，能够区分真实、全合成和局部篡改图像并定位篡改区域。

**💡 创新点**

创新点在于将多类别检测与像素级分割结合在同一轻量化架构中，利用冻结的DINOv2特征并采用重要性估计和分层融合实现高精度与低计算成本。

**🔧 技术方法**

使用冻结的DINOv2视觉变换器作为骨干，结合训练的重要性估计模块、分层特征融合、U‑Net解码器以及分类和分割两支分支；训练目标混合交叉熵、对比损失、Dice+BCE。

**📊 数据集**

在SID‑Set、So‑Fake‑Set和跨数据集评估中训练与测试，主要以So‑Fake‑Set为主数据集。

**📈 对比分析**

与多种CNN、轻量级检测器及LLM方法对比，检测准确率92.4%/IoU 77.8%显著优于SOTA（如So‑Fake‑R1、SIDA），并在模型大小与推理速度上实现约80倍参数压缩和16倍速度提升。

**⚠️ 局限性**

局限性包括对小尺寸或薄弱篡改区域检测不够精细、裁剪导致范围限制、以及对未来新型生成模型的泛化仍需改进。

---

## 445. TaRA: Training-Aware Low-Rank Adaptation Initialization

**arXiv ID:** 2609.02639 | [PDF](https://arxiv.org/pdf/2609.02639v1)

**作者:** Taehyeon Kim `[一作]` (Pohang University of Science and Technology), Eunhyeok Park `[通讯]` (Pohang University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练感知的低秩适配初始化方法TaRA，针对LoRA进行低秩初始化以更好地匹配完整模型的梯度行为

**💡 创新点**

创新点在于利用激活与梯度协方差共同构造Fisher加权的SVD，直接保留训练相关的主方向，从而显著提升梯度相似度与fine‑tune效果

**🔧 技术方法**

技术手段包括二阶泰勒近似、K‑FAC近似的Fisher矩阵、激活/梯度协方差估计、低秩SVD、数值正则化与低精度协方差收集

**📊 数据集**

使用了MetaMathQA、CodeFeedback-Filtered-Instruction、GSM8K-D/COT、MATH、HumanEval、MBPP、Commonsense‑170K等多种语言生成与理解数据集进行评估

**📈 对比分析**

在数学推理、代码生成和常识推理等任务中，与LoRA、PiSSA、CorDA、LoRA‑One、MiSS、LoRAM等基线相比，TaRA在不同秩（128、64、32）下均实现了最优或最接近最优的准确率，且梯度相似度最高、训练曲线最平滑

**⚠️ 局限性**

局限性包括对校准数据分布的依赖、需要额外的校准步骤与内存开销，以及在分布外任务（如HumanEval/MBPP）对秩的敏感性与效果波动

---

## 446. Deeply Interleaved Text-Image Contexts for Multimodal LLMs Assessment

**arXiv ID:** 2609.02573 | [PDF](https://arxiv.org/pdf/2609.02573v1)

**作者:** Zihao Wang `[一作]` (Harbin Institute of Technology), Wangmeng Zuo `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并构建了TIC‑Bench，评估多模态模型在深度交织的文本‑图像上下文中的推理能力

**💡 创新点**

首次系统定义并细分逻辑、时间、空间三类关联推理任务，且构造了长文本和高图像密度的交织数据

**🔧 技术方法**

采用多模态大语言模型并利用“thinking”模式实现显式推理

**📊 数据集**

使用TIC‑Bench数据集，包含2280问答、45,776张图像，平均309词文本和20张图像

**📈 对比分析**

与10款开源与闭源MLLM及人类基准对比，最佳模型GPT‑5.5达59.9%准确率，仍距人类91.7%有约30%差距

**⚠️ 局限性**

主要局限在多模态上下文管理不足，难以维护实体、并行信息流和跨模态关联，导致推理错误

---

## 447. AffectDelta: Beyond Emotion Labels for Image Editing

**arXiv ID:** 2609.02616 | [PDF](https://arxiv.org/pdf/2609.02616v1)

**作者:** Xingzu Zhan `[一作]`, Ruogu Fang `[通讯]` (Vanderbilt University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种源感知的图像情感编辑器（EmotionDiff），通过源图像与目标情感分布的差值直接驱动扩散模型，实现对图像情感的细粒度调控。

**💡 创新点**

创新点：① 将情感编辑视为完整的八维分布差值，而非单标签或文本指令；② 通过冻结的情感分布预测器（EDP）对图像进行分布化标注，并构建包含跨类别与同类别分布变化的训练对；③ 训练过程中不依赖编辑文本，直接用配对图像监督分布差值的视觉实现。

**🔧 技术方法**

技术：使用 ResNet-18+Softmax 的情感分布预测器；一个四层 MLP 的 Transition Encoder 将差值映射为 77×768 的注意力 token；基于 InstructPix2Pix 的源感知扩散骨干（U-Net + VAE）进行图像生成；额外的情感损失对生成结果的分布进行约束。

**📊 数据集**

数据集：从 GPT-Image-Edit-1.5M 和 ImgEdit 合并 2.56M 条候选对，经过情感相关性过滤与分布差值筛选后得到 248,841 条源-目标对（EmotionDiff Dataset），涵盖跨类别与同类别情感变化。

**📈 对比分析**

与六种基线（AIF、IP2P、LDAST、SDEdit、EmoEditor、EmoEdit）进行定量与定性对比，评估指标包括 Top‑1、JSD、NAP、Δ‑Cos、CLIP‑I、LPIPS。EmotionDiff 在所有六项指标上均排名第一，Top‑1 0.899、JSD 0.056、NAP 0.199、Δ‑Cos 0.744、CLIP‑I 0.778、LPIPS 0.382，显示出更精确的情感对齐与内容保持。

**⚠️ 局限性**

局限性：① 仅支持 8 维情感空间，无法处理更细粒度或多模态情感；② 依赖预训练的情感分布预测器，若预测误差大会影响编辑质量；③ 训练与推理均需图像配对，缺乏单图像即时控制的便利；④ 对文本或操作指令的兼容性有限。

---

## 448. Collective creativity in hybrid societies

**arXiv ID:** 2609.02620 | [PDF](https://arxiv.org/pdf/2609.02620v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 449. Advancing Accessible Underwater Robotics: The Mini-Girona I-AUV at RAMI 2025

**arXiv ID:** 2609.02605 | [PDF](https://arxiv.org/pdf/2609.02605v1)

**作者:** Taqi Hamoda `[一作]`, Nuno Gracias `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `51c0528b-f690-4182-ae60-bb5f046c276c` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计、实现并在RAMI 2025比赛中测试了一款约5万美元、配备5-DOF机械臂、立体视觉与AI处理的可访问式干预式水下无人机Mini‑Girona I‑AUV。

**💡 创新点**

创新点在于：①低成本平台化设计，将高端干预式AUV与传统ROV桥接；②在COLA2分层架构中集成强化学习、Petri网任务执行及自适应规划；③利用YOLO‑E v11、LLaVA与PCA等深度学习与几何方法实现实时物体检测、语义分割与交互点定位；④通过Stonefish仿真与CIRS池实地验证快速迭代。

**🔧 技术方法**

技术包括：蓝色机器人T200推进器、Basler ace 2双相机立体视觉、Nortek Nucleus 1000 DVL、Tritech Miniking声纳、Jetson Nano Orin 8GB深度学习板、Intel NUC PC、STM32 NUCLEO-32微控制器、COLA2架构、RRT‑Connect路径规划、DBSCAN聚类、PCA轴向定位、EKF状态估计、YOLO‑E v11 + LLaVA对象识别、机械臂轨迹与任务优先级控制。

**📊 数据集**

使用了RAMI 2025比赛场景下的实测数据（管道、阀门、标记、环杆等）以及Stonefish生成的仿真数据；YOLO‑E v11与LLaVA模型基于公开预训练权重在比赛图像上微调，但未公开具体标注集。

**📈 对比分析**

与比赛参赛队伍对标：Mini‑Girona获得TBM‑3大赛第二名、TBM‑2干预第二名、TBM‑1检查第三名，整体排名第二，展示了在视觉感知、路径规划和机械臂交互上的优异性能；实验表明声纳地图覆盖面积达75×50 m，立体深度可用于精确导航和避障。

**⚠️ 局限性**

局限性包括：①高温导致内部电子过热，影响首两天的自主导航；②参赛现场人力受限，导致多角色并行操作；③部分干预任务仍需远程控制，未实现完全自主；④对极端环境（低能见度、高温）适应性不足，未来需加强散热与容错机制。

---

## 450. Predictors of Loneliness in Older Adults Using Multimodal Analysis of Speech and Language

**arXiv ID:** 2609.02606 | [PDF](https://arxiv.org/pdf/2609.02606v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 451. Generalizable Brain Tumor Segmentation with Self-Training and Tumor-Aware Deformations

**arXiv ID:** 2609.02600 | [PDF](https://arxiv.org/pdf/2609.02600v1)

**作者:** Henrique Zan Grande `[一作]` (Pontifícia Universidade Católica do Paraná), Andre Gustavo Hochuli `[通讯]` (Pontifícia Universidade Católica do Paraná)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本研究提出了结合自训练与肿瘤感知可变形数据增强的脑肿瘤分割方法，以提升在异质患者群体中的泛化性能。

**💡 创新点**

创新点在于设计了局部弹性变形与等距缩放相结合的肿瘤感知可变形增强器，以及通过置信度筛选的伪标签自训练流程，二者相辅相成。

**🔧 技术方法**

使用了nnU-Net框架的Large Residual Encoder、伪标签自训练策略、置信度筛选、以及基于Gaussian权重的局部可变形增强技术。

**📊 数据集**

实验基于官方goat挑战数据集，包括已标注的成人胶质瘤、脑膜瘤和转移瘤、未标注训练池以及包含未知肿瘤类型的验证集。

**📈 对比分析**

与仅使用标注数据的学生模型相比，加入自训练提升平均Dice从81.2%升至81.9%，加入增强器进一步提升至82.4%，并使HD95从30.6mm降低至25.4mm，验证了两者互补。

**⚠️ 局限性**

局限性包括：自训练的伪标签仍可能包含噪声；可变形增强主要针对大肿瘤，较小肿瘤被排除；实验仅在goat挑战数据上评估，缺乏跨平台验证。

---

## 452. Integrating Wi-Fi into 3GPP 5G Network Slicing: An Experimental Prototype Study

**arXiv ID:** 2609.02625 | [PDF](https://arxiv.org/pdf/2609.02625v1)

**作者:** Nelson Ion de Oliveira `[一作]` (Federal University of Rio Grande do Norte), Augusto Venâncio Neto e Vicente A. de Sousa `[通讯]` (Federal University of Rio Grande do Norte)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

实现了5G与Wi‑Fi的网络切片集成原型，支持5G SA与WLAN联合NSI；

**💡 创新点**

创新性地将Trusted Non‑3GPP Gateway Function (TNGF)扩展至Wi‑Fi，并在OpenWrt AP上实现基于5QI/DL AMBR的动态资源调度；

**🔧 技术方法**

采用free5GC、OpenWrt、IPsec、IKEv2、HTB、ATF等技术实现多RAT切片与QoS控制；

**📊 数据集**

使用自建实验测试床，无公开数据集，基于两台笔记本的UDP流进行验证；

**📈 对比分析**

通过对比六种场景的吞吐量与优先级，验证在不同带宽/优先级组合下吞吐稳定性与QoS分化；限速场景下吞吐约等于配置值，优先级显著；

**⚠️ 局限性**

局限在仅支持基于Linux AP的HTB/ATF实现，需WPA2/3‑Enterprise，实验规模有限，未覆盖大规模异构网络与自动化编排。

---

## 453. Beyond Problem Solving: Large Language Models for Emotional and Reflective Support in Mathematics Learning

**arXiv ID:** 2609.02611 | [PDF](https://arxiv.org/pdf/2609.02611v1)

**作者:** Vera Rief `[一作]` (Saarland University), Tomohiro Nagashima `[通讯]` (Saarland University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研发了基于LLM的情绪支持的代数ITS“Math with Matt”，通过正念聊天、呼吸练习和正念提示，为学生提供认知与情绪双重支持。

**💡 创新点**

首次将LLM驱动的正念交互与传统ITS的认知提示相结合，形成情绪调节层，创新性地将情绪支持嵌入智能辅导系统。

**🔧 技术方法**

采用OpenAI GPT‑4o‑mini生成正念聊天与提示，使用CTAT构建ITS界面，并手工编写正念化的反馈与提示。

**📊 数据集**

通过课堂实验收集252名七年级学生的预/后测成绩、STAI/AMAS焦虑量表与使用日志，最终有效样本为42人。

**📈 对比分析**

采用线性混合模型和Mann‑Whitney U检验比较Mindful版与仅认知版，发现两版学习成绩均提升但无显著差异，Mindful版学习效率更高、提示使用更少，情绪指标差异不显著。

**⚠️ 局限性**

样本量不足、学生英语水平低导致情绪支持效果被掩盖、课堂环境抑制个体正念练习，因而未能显著降低数学焦虑，且未观察到明显的学习表现提升。

---

## 454. EEG-based Visual Retrieval and Reconstruction: From Neurally Visible Optimal Layer to Hierarchical Diffusion Generation

**arXiv ID:** 2609.02582 | [PDF](https://arxiv.org/pdf/2609.02582v1)

**作者:** Minyi Wang `[一作]` (University of Macau), Rihui Li `[通讯]` (University of Macau)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了使用中间层视觉语言模型与EEG信号对齐的方法，提出层可视性最优层（NVOL）并构建了融合检索与两阶段扩散生成的统一框架；

**💡 创新点**

创新点在于发现EEG最适合对齐中间层而非最终语义层，使用NVOL选择与多层融合；引入CSLS校正减少hubness；采用两阶段扩散先重建中间特征再映射到语义层，从而提升检索和重建质量；

**🔧 技术方法**

技术包括ATM EEG编码器、CLIP ViT-H/14 中间层融合、InfoNCE对比学习、CSLS后处理、DDPM条件扩散、轻量化适配器以及Stable Diffusion XL + IP-Adapter；

**📊 数据集**

使用的数据集为THINGS‑EEG（10位受试者，200类目标）；

**📈 对比分析**

与单层、无CSLS、多层、单/双阶段方法对比，最佳配置（多层+CSLS）在200‑way检索中达到Top‑1 86.4%、Top‑5 98.5%；双阶段TS‑Diff在PixCorr、SSIM、Inception、CLIP等指标上优于单阶段，显著提升图像重建质量；

**⚠️ 局限性**

局限性包括受试者个体差异未建模、仅在THINGS‑EEG数据上验证、未测试跨数据集/跨模态的迁移、重建图像细节仍不够精细、对跨受试者最优层的泛化缺乏深入探讨。

---

## 455. HINT: Human-Intent Inception for Long-Horizon Robot Manipulation

**arXiv ID:** 2609.02653 | [PDF](https://arxiv.org/pdf/2609.02653v1)

**作者:** Mingyu Mei `[一作]` (Zhejiang University), Zaixing He `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 HINT（Human-INTent INcePtion）框架，将稀疏的语言意图转化为连续、目标导向的视觉指导，配合多视角感知与动作策略；

**💡 创新点**

创新点在于：①仅在操纵模式切换时进行语义推理，避免频繁计算；②通过目标跟踪保持语义承诺；③使用可视高亮与注意力先验两条无额外参数的接口，将语义信息注入预训练动作策略；

**🔧 技术方法**

核心技术包括多视角图像编码（ResNet-18+适配器）、GRU感知/扭矩融合、模式路由网络、Qwen3-VL 语义推理、Grounding DINO + SAM2 跟踪、视觉高亮与注意力偏置注入；

**📊 数据集**

使用双臂机器人（AgileX PiPER）配备全局 + 双腕摄像头的实地演示数据，涵盖水果蔬菜分类、字母拼写、钉孔插入三大任务；还构造了未见对象、属性、布局、目标与指令的 OOD 变体；

**📈 对比分析**

对比基准策略 Wall-OSS‑0.5 与 π_0.5，实验显示 HINT 在 IS、Sub. SR 与 Full SR 上提升显著（平均提升约 30–50%），且保持低延迟；在 OOD 情况下也显著提升成功率，验证了语义泛化能力；

**⚠️ 局限性**

局限性包括：①对 VLM 语义先验的依赖，可能在新概念或模糊观测下不稳；②需要人工标注操纵模式与多视角的预处理流程；③目前仅支持已知的运动原语，未能处理极端接触细节或超高精度任务。

---

## 456. AgOSS: A Dataset and Multi-Layer Characterization of Open-Source Agricultural Software

**arXiv ID:** 2609.02591 | [PDF](https://arxiv.org/pdf/2609.02591v1)

**作者:** Vatsal Dudhaiya `[一作]` (Purdue University), James C. Davis `[通讯]` (Purdue University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建AgOSS数据集并通过OpenSSF Scorecard、SBOM/OSV、CISA KEV等多维度评估农业OSS供应链安全。

**💡 创新点**

首次跨层级、跨类别的农业OSS安全基准，揭示安全缺口主要由项目规模和治理能力决定而非域本身。

**🔧 技术方法**

采用MSR技术、GitHub API、OpenSSF Scorecard、SBOM生成与OSV查询、KEV匹配，以及匹配与回归分析等方法。

**📊 数据集**

使用包含66个农业相关开源仓库的AgOSS数据集，分为6类。

**📈 对比分析**

通过Coarsened Exact Matching+Mahalanobis距离匹配和多元回归比较，发现Raw Scorecard差距在匹配后消失，依赖风险与治理无关。

**⚠️ 局限性**

样本规模有限、SBOM不完整导致依赖分析欠缺、GitHub检索局限、未考虑其他潜在混杂因素。

---

## 457. Automated Vulnerability Injection in Smart Contracts Using Large Language Models

**arXiv ID:** 2609.02624 | [PDF](https://arxiv.org/pdf/2609.02624v1)

**作者:** Luca Migliaccio `[一作]` (University of Naples Federico II), Marco Vieira `[通讯]` (University of North Carolina at Charlotte)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大型语言模型对Solidity智能合约进行漏洞注入，并通过多步验证生成可验证的脆弱合约数据集；

**💡 创新点**

创新点在于：①将LLM用于注入可行性评估和实际注入；②构建四步验证流程（编译、约束、业务逻辑、漏洞确认），确保注入结果真实有效；③覆盖49种OpenSCV漏洞类型，生成近千个候选合约，构建首批公开可验证数据集；

**🔧 技术方法**

使用的技术包括：LLM Qwen2.5‑Coder 进行可注入性评估，Meta‑Llama‑3 进行注入生成；prompt 设计、重复执行与分数阈值筛选；文本去重、编译/执行检查；手工验证（注入是否符合约束、业务逻辑完整性、漏洞是否可利用）；利用 Remix Analyzer、Slither 与 Solhint 进行工具评估；

**📊 数据集**

数据集来源：14个无已知漏洞的SmartBugs sb‑wild子集合约；OpenSCV 49种漏洞类型；通过LLM生成近1000个候选注入合约，最终得到32个通过全部验证的脆弱合约；

**📈 对比分析**

对验证后的32个合约分别使用 Remix、Slither、Solhint 进行检测，统计 TP、FP、FN、SAF，计算精度与召回：Remix(0.786/0.688)、Slither(1.000/0.579)、Solhint(0.857/0.706)。结果表明工具覆盖互补，单一工具无法完全覆盖；

**⚠️ 局限性**

限制在于：只有检查类（Checking）漏洞大概率通过，复杂合约或结构性漏洞通过率低；业务逻辑验证是主要瓶颈；手工验证耗时、难以扩展；LLM生成多样性有限，重复率高；仅使用开源模型且硬件受限，未涉及闭源模型；最终验证合约比例仅为16.58%，未能覆盖完整的OpenSCV 49种类型。

---

## 458. Pre-Lane-change Signal in Transitional Autonomous Vehicles: Results from Controlled Experiments

**arXiv ID:** 2609.02575 | [PDF](https://arxiv.org/pdf/2609.02575v1)

**作者:** Zeyu Mu `[一作]` (University of Virginia), George F. List `[通讯]` (North Carolina State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在受控实验中，对150条生产型过渡级自动驾驶车辆（tAV）的强制换道行为进行定量分析，探索换道前（SigT）已知的目标间隙信息及其对横向动作开始时机的影响；

**💡 创新点**

提出了操作性可观测的信号时间SigT，将换道决策与横向运动分离，构建基于相对位置与相对速度的前/后二元预测模型，并发现换道过程可划分为纵向准备与横向执行的两阶段；

**🔧 技术方法**

使用Firth逻辑回归进行二元预测、5折与留一交叉验证评估模型、引入AES（加急安全间距）和前后车距清除度等安全度量，进行纵向路径差异比较；

**📊 数据集**

采用NC‑tALC实验数据集，包含150次在公共道路上实施的强制换道试验，车辆通过RTK‑GNSS/INS记录高速、纵向与横向轨迹；

**📈 对比分析**

通过5折交叉验证模型准确率达0.89、留一交叉验证0.90、AUC0.97；纵向准备阶段，in‑position组的AES AUC约0.71，reposition组的车距清除度AUC约0.84，显示不同群组的路径特征；

**⚠️ 局限性**

仅基于受控实验，重定位案例比例低、SigT多等于ACT限制了提前观察；结果在自然驾驶情境中的适用性需进一步验证。

---

## 459. MARS: What Retrieval Signals Are Hidden in Multimodal Large Language Models for Text-Video Retrieval?

**arXiv ID:** 2609.02565 | [PDF](https://arxiv.org/pdf/2609.02565v1)

**作者:** Uicheol Jung `[一作]` (Sejong University), Yukyung Choi `[通讯]` (Sejong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 MARS（Multi‑layer Adaptive Representation Slots）框架，用多层隐藏状态融合与多槽嵌入的方式改进文本‑视频检索的表示学习，并引入硬负样本专门化损失提升细粒度区分能力。

**💡 创新点**

创新点：
1) 多层融合：利用每层解码器隐藏状态权重化融合，取代单层最终向量。
2) 多槽设计：在提示中插入可学习槽标记，构造多个互补的表示槽并分别匹配。
3) 硬负样本槽专门化：针对每个硬负样本选择最能区分正负的槽，加入对应的最大间隔损失。
4) 直接检索结构：在多模态 LLM 上保持双编码器直接相似度检索，避免额外跨模态匹配与重排序的计算成本。

**🔧 技术方法**

技术手段：基于 VideoChat‑Flash‑Qwen2‑7B MLLM；LoRA 微调；对齐损失、槽多样性正则、硬负样本专门化损失；DSL 置信度校准；可选的 BLiM 级重排序；多槽加权平均融合。

**📊 数据集**

数据集：文本‑视频检索四大基准——DiDeMo、ActivityNet、LSMDC、MSR‑VTT；文本‑图像检索——COCO（fine‑tuned）与 Flickr30K（zero‑shot）用于验证通用性。

**📈 对比分析**

与方法对比：在直接检索协议下，MARS 在四个基准上的 mR@1 达到 67.0（T2V）/64.5（V2T），引入 DSL 后提升至 71.6/72.0，显著优于 InternVideo2‑6B、BLiM 等；在 rerank 方案（MARS‑R）进一步提升至 87.6/82.2/83.5/77.9 的 R@1，取得 state‑of‑the‑art 结果。

**⚠️ 局限性**

局限性：仅针对视频级检索；未覆盖长视频、噪声描述、极端用户查询等开放域场景；未实现时间段定位或瞬间检索；在其他模态（音频‑语言、3D‑语言）或更广泛的检索任务中的可扩展性尚未验证。

---

## 460. A Finger on the Scale: Covert Policy Steering through Agentic Skills

**arXiv ID:** 2609.02564 | [PDF](https://arxiv.org/pdf/2609.02564v1)

**作者:** Jiarui Li `[一作]` (Chongqing University), Shouling Ji `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a2602d71-93ab-4bad-974b-672788df8193` `6215c339-3735-4be3-8a07-5bbb7004712d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了如何利用可重用的Skill（任务指令包）在保持原有任务与输出格式不变的前提下，隐蔽地改变LLM代理在固定候选集上的决策偏好，从而实现对目标候选的高概率诱导；

**💡 创新点**

创新点包括：①提出Skill Policy Integrity概念，揭示了Skill可以在不破坏任务合法性的前提下操控决策的风险；②设计了结构化黑盒优化框架（Policy Framing、Tie‑Breaking Manipulation、Semantic Anchoring），通过可验证的策略表示和约束搜索，在黑盒环境下实现高效、可迁移且难以被检测的策略操控；

**🔧 技术方法**

采用的技术主要有：结构化策略表示（G,U,T,X）、基于黑盒反馈的迭代优化、层级验证、失败引导覆盖修复、策略压缩以消除冗余；实现中使用GPT‑5.5做策略生成、Claude Haiku 4.5等模型做代理仿真，最终通过确定性解析器评估候选选择；

**📊 数据集**

实验数据集为两类固定候选任务：购物推荐（目标品牌Li‑Ning）和Python依赖包选择（目标包pandas），各自包含30个开发查询和20个保留查询，共计50个；

**📈 对比分析**

对比方法包括Clean Skill、Target Name Only、Keyword Stuffing、Generic Prompt Injection、Direct‑Skill Injection等基线，评价指标为PSR（目标选择率）、VR（有效输出率）及Lift。实验结果显示攻击在购物推荐中PSR从37.33%提升至81.33%，在Python依赖中从0%提升至63.33%，Lift均达到40‑80pp，且100% VR；攻击策略在不同模型和完整代理环境中保持正向Lift，显示良好的跨模型和跨代理迁移性；

**⚠️ 局限性**

局限性包括：仅在两种固定候选域验证，缺乏对动态候选集的测试；评估仅采用3次推理，统计显著性有限；下游质量评估仅依赖自动评分，未进行功能执行验证；检测器评估受限于现有工具配置，未覆盖所有潜在检测策略。

---

## 461. Stereo 4D Radar for 3D Object Detection: Integrating Geometric Alignment and Absolute Velocity Estimation

**arXiv ID:** 2609.02560 | [PDF](https://arxiv.org/pdf/2609.02560v1)

**作者:** Seung-Hyun Song `[一作]`, Seung-Hyun Kong `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提供了IEEEtran模板的使用指南，帮助作者快速撰写IEEE期刊和会议论文。

**💡 创新点**

创新点在于将官方文档简化为易读、易用的手册，降低了对LaTeX经验的门槛。

**🔧 技术方法**

使用LaTeX语言、IEEEtran类文件以及常用宏包和命令实现排版。

**📊 数据集**

未使用任何数据集，主要面向文档编写。

**📈 对比分析**

不涉及实验比较，主要以示例文件演示排版效果。

**⚠️ 局限性**

局限性在于仅覆盖基本模板功能，复杂排版和特殊会议模板需参考IEEEtran_HOWTO等文档。

---

## 462. oHC: Orthogonal Hyper-Connections on SO(4) via Quaternions

**arXiv ID:** 2609.02672 | [PDF](https://arxiv.org/pdf/2609.02672v1)

**作者:** Haoqiang Guo `[一作]` (Hong Kong University of Science and Technology), Wenhan Luo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对多流残差网络（Hyper‑Connections）进行系统分析，并提出一种将残差矩阵限制在 SO(n) 旋转群上的正交 Hyper‑Connections。

**💡 创新点**

创新点在于：①阐释双随机约束导致残差混合器对间流差异的消减；②引入正交约束使混合器保持等距，消除差异消减；③在 n=4 时用两单位四元数闭式参数化 SO(4)，无迭代且无额外参数。

**🔧 技术方法**

主要技术包括：残差矩阵的奇异值与均差分解、双随机投影（Sinkhorn–Knopp）、正交群投影（Cayley、Schulz、四元数），以及混合专家（MoE）架构与MuOn优化器。

**📊 数据集**

使用内部大型语料库训练 3.9B 参数、0.4B 活跃参数的 MoE 语言模型，评估 16 个基准（MMLU、CMMLU、SuperGPQA、ARC‑Challenge、PIQA、GSM8K、MATH、HumanEval、MBPP、CRUXEval、SimpleQA 等）。

**📈 对比分析**

对比单流残差、双随机、身份点和正交约束四种设置，正交 Hyper‑Connections 在训练损失和多任务 BPB 上均显著优于其他三种（统计显著性超过 6σ），且四元数实现成本最低。

**⚠️ 局限性**

局限性包括：①初始化受 Sinkhorn 投影取值影响，正交模型在某些实现下难以恢复到理想起始点；②实验仅在单一模型规模上验证，尚未验证规模扩展时性能是否保持。

---

## 463. Decoupling Disaggregated Memory Optimizations from Indexing: A Compiler-Runtime Approach

**arXiv ID:** 2609.02669 | [PDF](https://arxiv.org/pdf/2609.02669v1)

**作者:** Xinpeng Zhao `[一作]`, Eric Lo `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d`

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

## 464. Intuitionistic Unitary Linear Logic: A Proof-Theoretical Approach to Purely Quantum Higher-Order

**arXiv ID:** 2609.02661 | [PDF](https://arxiv.org/pdf/2609.02661v1)

**作者:** Julien Lamiroy `[一作]` (Université Paris-Saclay), Renaud Vilmart `[通讯]` (Université Paris-Saclay)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种新的形式化方法来表示非因果的高阶量子过程，特别是基于线性逻辑的逻辑系统。

**💡 创新点**

创新点在于提供了一种模块化和组合性的计算解释，能够更好地处理高阶量子过程的超位置特性，并确保单位性。

**🔧 技术方法**

使用了基于直觉主义乘法加法线性逻辑的类型系统和逻辑。

**📊 数据集**

论文中没有具体提到使用的数据集。

**📈 对比分析**

与现有的非因果量子计算模型相比，提出的方法在表达能力和操作性上具有优势，能够更好地处理已知的非因果量子过程。

**⚠️ 局限性**

限制在于该方法尚未完全证明与超通道的条件完全一致，特别是在处理更高阶过程时的扩展性问题。

---

## 465. Oracle, will I ever learn? A study of prediction convergence and complementarity across link prediction models

**arXiv ID:** 2609.02638 | [PDF](https://arxiv.org/pdf/2609.02638v1)

**作者:** Guillaume Méroué `[一作]` (Université Côte d’Azur), Pierre Monnin `[通讯]` (Université Côte d’Azur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了知识图谱链接预测模型的预测收敛性和互补性，提出基于oracle的评估框架，量化不同模型/实例的上界性能。

**💡 创新点**

通过oracle框架揭示模型互补性三大性能差距（实例-oracle差距、模型-跨模型差距、渐近差距），并证明即便集合规模极大也无法达到完美准确。

**🔧 技术方法**

采用多种知识图谱嵌入模型、神经推理模型和符号规则挖掘模型，训练20个随机种子实例，使用MRR、Hits@K评估，并实现oracle选择。

**📊 数据集**

在WN18RR、FB15k-237、CoDEx-S三个公开知识图谱上进行实验。

**📈 对比分析**

与单实例、随机实例集合以及基于贪心选取的跨模型集合比较；oracle性能显著高于单实例，跨模型oracle可逼近最佳单模型oracle；但oracle随实例数增长衰减，最终无法达到1。

**⚠️ 局限性**

局限包括：oracle仅为理想上界，实际集成方法难以逼近；未考虑超参数多样化、分区训练等多样性来源；存在不可解查询子集，提示数据集或模型表达力受限。

---

## 466. Scalable Direction-Following TTS via Voice Impression-Guided Pseudo Triplet Construction

**arXiv ID:** 2609.02623 | [PDF](https://arxiv.org/pdf/2609.02623v1)

**作者:** Kenichi Fujita `[一作]` (NTT, Inc.), Yusuke Ijima `[通讯]` (NTT, Inc.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了“方向跟随式TTS”，即在给定参考语音和自然语言方向的情况下生成保持说话人身份与文本内容但风格符合方向的合成语音，并提出一种可扩展的伪三元组构建方法。

**💡 创新点**

创新点：①将印象可控TTS与LLM结合，自动生成既包含预/后改写语音又带有自然语言方向的伪三元组；②在语音嵌入空间中使用方向条件化流匹配实现细粒度风格转化；③通过混合伪与真实录音数据平衡身份保持与表现力。

**🔧 技术方法**

使用的技术包括：印象可控FastSpeech2 TTS、HuBERT+GAN语音编码器、ECAPA‑TDNN说话人嵌入、ModernBERT‑Ja 对方向文本编码、rectified flow matching 的风格修正器、LLM（Qwen3‑Next‑80B‑A3B‑Instruct 与 GPT‑5.2）生成方向与评估。

**📊 数据集**

数据集：①伪数据——1,600名日语说话人共350k三元组（约127.6h）；②真实录音——2名专业配音演员共8.9h，包含6,899对语音与对应方向。

**📈 对比分析**

比较方法：在seen与unseen说话人上分别评估说话人相似度（ECAPA‑TDNN余弦相似度）、自然度（UTMOSv2）、方向对齐（LLM评分与主观AlignMOS）。结果显示：伪数据单独能保持身份稳定；录音数据单独能提高方向匹配；混合数据（Full）兼顾两者，取得最佳平衡。

**⚠️ 局限性**

局限性：伪数据的声学变化（如F0、节奏）比专业录音小，导致转化更保守；LLM评估作为代理指标，可能不完全捕捉人类主观体验；方法仍需验证在多语言、多说话人场景中的泛化能力。

---

## 467. Learning-Based Reconstruction Attacks on Coordinate-Obfuscated Point Clouds

**arXiv ID:** 2609.02568 | [PDF](https://arxiv.org/pdf/2609.02568v1)

**作者:** Mohammad Waquas Usmani `[一作]` (University of Massachusetts Amherst), Michael Zink `[通讯]` (University of Massachusetts Amherst)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

评估了选择性坐标加密在点云流媒体中的安全性，针对已加密的X坐标构造机器学习重建攻击模型。

**💡 创新点**

揭示加密粒度对攻击效果的关键影响：完全加密X坐标时重建困难，而交替加密X坐标则易被ML高精度重建，并首次将点对点MAE、Chamfer Distance与Hausdorff Distance等度量用于此类安全评估。

**🔧 技术方法**

使用PointNet回归网络和随机森林回归模型，并结合空间坐标、表面法线、颜色及邻域曲率等特征进行训练与预测。

**📊 数据集**

基于Open3D Office和Living Room两大室内场景点云数据集，分别提供训练与测试帧。

**📈 对比分析**

对比零填充基线、简单插值基线以及两种ML模型，完全加密场景下MAE约27.9%，CD≈0.11，HD≈0.52；交替加密场景下随机森林M4模型MAE≈3%，CD≈0.009，显示ML可显著提升重建精度。

**⚠️ 局限性**

仅考察两种极端加密粒度，未覆盖更细粒度或多维加密；模型规模受限，未使用更深或更强的网络（如PointNet++）；实验仅限室内场景，泛化性及对更大/复杂点云的评估仍待验证。

---

## 468. Online Reinforcement Learning in the Met Office Unified Model through Distributed Model-Agent Coupling

**arXiv ID:** 2609.02566 | [PDF](https://arxiv.org/pdf/2609.02566v1)

**作者:** Pritthijit Nath `[一作]` (University of Cambridge), Mark Webb `[通讯]` (Met Office)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在英国气象局统一模型（UM）中实现并测试了在线强化学习（RL）策略，用于自适应地校正潜热温度趋势并改进预报；

**💡 创新点**

首次在全球运营级NWP模型中完成了分布式在线RL与模型的实时耦合，并演示了训练后直接转到无nudging推理阶段的可行性；

**🔧 技术方法**

使用分布式MPI/OpenMP Fortran与Python RL代理的SmartSim/SmartRedis接口、DDPG actor共享权重以及基于潜热温度的奖励函数；

**📊 数据集**

利用英国气象局的运营分析数据与UM N320 40km、70层的气象格点数据进行10次训练预报和1次推理预报；

**📈 对比分析**

与同一初始条件的未校正UM控制进行比较，评估+6h时刻的MAE。结果显示Z_500 MAE在北、南热带分别降低45.8%和40.8%，MSLP MAE在0–30°N降低27.3%，但在极地和部分温度场存在增误；

**⚠️ 局限性**

仅在单一初始化下验证，缺乏跨季节和多重初始种子测试，奖励使用MSE而验证采用MAE导致局部提升可能掩盖整体误差；多变量反应不一致，可能需要区域化代理或更统一的奖励目标。

---

## 469. The Shape of Ownership: Verifying LLM Provenance through Semantic Structures

**arXiv ID:** 2609.02553 | [PDF](https://arxiv.org/pdf/2609.02553v1)

**作者:** Zhongrui Sun `[一作]` (Chongqing University), Shouling Ji `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种黑盒模型指纹技术，利用域条件的抽象语义结构（AMR模板）对大型语言模型进行指纹注入并通过自然查询进行所有权验证。

**💡 创新点**

创新点在于将指纹从稀疏的查询-响应关联转变为分布式的语义组织，使指纹在自然输入下自然激活，既保持鲁棒性和隐蔽性，又能在多模型、多域场景下验证所有权。

**🔧 技术方法**

采用了AMR抽象、模板构建、结构化验证的数据生成、混合监督微调（LoRA）以及统计黑盒检测方法。

**📊 数据集**

实验数据集包括数学推理领域的GSM8K、医疗诊断领域的MedQA，以及通用指令数据集Alpaca、Dolly、WildChatFr和OpenMathInstruct。

**📈 对比分析**

与IF、SF、CH、CTCC、SCW等现有黑盒指纹方法对比，本文方法在多模型、多尺度（Qwen2.5-3B/7B、Llama3.2-3B）下，跨域、跨任务、压缩、量化、剪枝、微调及知识蒸馏等攻击场景下均实现100%指纹检测成功率，且无误报，且对模型实用性影响极小。

**⚠️ 局限性**

局限性在于需手工构建域特定AMR模板，模板质量和覆盖度直接影响指纹学习；若攻击者能长期观察验证交互并推断模板，可能开展针对性过滤或反向微调攻击，尚无充分对抗方案。

---

## 470. ProbeMatchDTI: Probe-Driven Multi-Scale Biochemical Pattern Matching for Drug-Target Interaction Prediction

**arXiv ID:** 2609.02549 | [PDF](https://arxiv.org/pdf/2609.02549v1)

**作者:** Quan Hao `[一作]` (Beijing University of Technology), Liguo Zhang `[通讯]` (Beijing University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并提出了 ProbeMatchDTI 框架，利用 IterProbe 和 BindingProbe 两种主动探测机制来改进药物–靶点相互作用预测，并集成下游验证与实验规划流程。

**💡 创新点**

创新点在于主动保留多尺度化学和蛋白信息的 IterProbe，以及微观 atom–residue 对应与宏观配对兼容的 BindingProbe，从而突破传统被动聚合导致弱信号被压制的问题。

**🔧 技术方法**

采用基于 Transformer 的分子和蛋白序列编码、可学习的探针机制、图神经网络的邻域更新、双向微观对应以及宏观探针融合的多尺度交叉注意力技术。

**📊 数据集**

使用 BindingDB、DrugBank、C. elegans、Human 四个公开基准以及私有的 ABPP 数据集，并通过分子动力学模拟验证模型预测。

**📈 对比分析**

在四个公共基准上与 MolTrans、CPIformer、MMDG-DTI 等多种 SOTA 进行对比，ProbeMatchDTI 在 AUC‑ROC 及 AUC‑PR 上均取得最高分，最大提升 2.0% 的 AUC‑ROC（BindingDB）和 0.5% 的 AUC‑PR（DrugBank）。

**⚠️ 局限性**

局限性包括对公开数据集偏倚的依赖、缺乏真实临床或实验室级别的验证，以及模型结构复杂、计算资源需求较高。

---

## 471. From Tokens to Semantics: Leveraging Complementary Signals for Hallucination Detection in Black-Box LLMs

**arXiv ID:** 2609.02679 | [PDF](https://arxiv.org/pdf/2609.02679v1)

**作者:** Urja Pawar `[一作]` (BNY), Christopher Martin `[通讯]` (BNY)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究在缺乏参考证据的黑盒大型语言模型中，利用语义熵和token级对数概率两类可观测信号来检测模型生成的幻觉（hallucination）。

**💡 创新点**

创新点在于提出了三种无监督或轻监督的检测方法：TopK（聚合多样化生成的token不确定性）、Gated（按语义簇数路由）和Stacked（联合学习语义熵与token特征），并系统评估其在不同模型、数据集与误报预算下的表现。

**🔧 技术方法**

使用的技术包括：语义聚类与熵计算（Standard SE、UEigV、Hybrid、Von Neumann）、token log‑probability特征提取、TopK聚合、CoCoA混合得分、Gated与Stacked分类器（Logistic回归、PCA）。

**📊 数据集**

实验使用七个基准：AA Omni Finance、AmbigQA、HotpotQA、Cheque Generation、Financial Summaries、Long‑Text QA、SQuAD；四个生成模型：GPT‑4.1‑mini、GPT‑5.1、GPT‑5.4、Llama 3.3 70B；每个数据集包含多样的上下文、答案、推理深度与领域。

**📈 对比分析**

与方法比较基于AUROC、TPR‑FPR曲线及不同误报预算（1%–15%）下的召回率。Stacked在多达 11/26 比较中排名第一或同级，且平均距最佳方法 0.05 AUROC 以内；TopK 与 CoCoA 在无监督场景下表现竞争，Gated 在特定数据集（如 Cheque Generation）优于其它方法。不同模型与数据集会影响最佳方法，说明需要针对场景调优。

**⚠️ 局限性**

局限性包括：Gated/Stacked 需要目标域标注数据；对小样本数据集（如 Long‑Text QA）评估不稳；token方法依赖 API 暴露 log‑probability；未评估零样本迁移或跨域泛化；模型间性能差异表明单一方法难以适用于所有场景。

---

## 472. WinoQueer-NL: Assessing Bias in Dutch Language Models toward LGBTQ+ Identities

**arXiv ID:** 2609.02651 | [PDF](https://arxiv.org/pdf/2609.02651v1)

**作者:** Jiska Beuk `[一作]` (Maastricht University), Gerasimos Spanakis `[通讯]` (Maastricht University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过翻译、校正、社区验证等步骤，构建了首个荷兰语版WinoQueer基准（WinoQueer‑NL）并用其评估多种荷兰及多语种语言模型的 LGBTQ+ 偏见。

**💡 创新点**

创新点在于首次为荷兰语创建文化适配的偏见评估基准、采用参与式社区调查验证偏见的相关性与有害性，并对比多模型、多身份组的偏见差异，揭示跨语言偏见分布差异。

**🔧 技术方法**

使用了伪对数似然（pseudo‑log‑likelihood）和标准对数似然评分方法，分别对掩码语言模型（MLM）和自回归语言模型（ARLM）进行偏见测量；同时采用了身份平均和使用权重两种偏见评分变体。

**📊 数据集**

主要数据集为自制的 WinoQueer‑NL：42,906 句对，包含 167 条经社区评估的偏见表述，以及 43 名荷兰 LGBTQ+ 参与者的调查问卷结果。

**📈 对比分析**

通过对比使用权重偏见分数和身份平均偏见分数，研究发现整体偏见接近中性（≈50%），但跨模型和身份组差异显著，跨性别与非二元身份的偏见最高可达 97%，而同性恋和酷儿身份的偏见往往低于中性；不同模型间性能差异说明偏见的身份特异性和语言环境对模型行为的影响。

**⚠️ 局限性**

限制包括样本量相对较小（43 名调查者）、身份群体分布不均、模板结构难以涵盖所有偏见形式，以及基准结果并不能完全证明模型无偏见，需进一步结合更广泛的评估方法。

---

## 473. Loom: Weaving Diagnostic Strands into Free-Text Consensus via Embedding-Space Reweighting

**arXiv ID:** 2609.02649 | [PDF](https://arxiv.org/pdf/2609.02649v1)

**作者:** Ron Begleiter `[一作]` (NVIDIA), Gil Shabat `[通讯]` (NVIDIA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并部署了 Loom，一种基于连续嵌入空间的生成式共识框架，用于工业环境下的根因分析（RCA）。

**💡 创新点**

创新点：将弱监督与 LLM 结合，利用迭代中心点重加权算法在向量空间中对开放式文本假设进行去噪与排序，并通过单次 LLM 合成实现高效、可审计的 RCA；引入模块化的诊断条带（Diagnostic Strands）实现知识的程序化编码。

**🔧 技术方法**

技术：程序化诊断条带、文本嵌入（ϕ）、迭代嵌入‑中心点重加权算法、单次 LLM 合成（Claude 4.6 或 Llama‑3.1‑8B）。

**📊 数据集**

数据集：OpenRCA Benchmark（Bank、Market‑1、Market‑2、Telecom）以及 NVIDIA 生产机房的真实案例。

**📈 对比分析**

与传统自律 LLM Agent（RCA‑Agent）对比，Loom 在 Bank 与 Market‑2 上与 Agent 相当，其他数据集略低，但单次 LLM 调用实现约 26–33 倍的速度提升和更低成本。

**⚠️ 局限性**

局限：在 Market‑1 和 Telecom 上准确率落后；依赖预先编写的诊断条带，覆盖范围受限；单次 LLM 合成对复杂或噪声密集的情形易失真；嵌入模型表达能力限制；冷启动需要人工或离线 LLM 提取。

---

## 474. PrimSynth: An Agentic Approach to Discover, Validate, and Synthesize Exploit Primitives for Linux Kernel Vulnerabilities

**arXiv ID:** 2609.02647 | [PDF](https://arxiv.org/pdf/2609.02647v1)

**作者:** Pengfei Wang `[一作]` (National University of Defense Technology), Wei Xie `[通讯]` (National University of Defense Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PrimSynth 多智能体框架，实现 Linux 内核漏洞的探索、验证与原语链生成。

**💡 创新点**

创新性在于将探测、验证与合成三阶段纳入闭环迭代，构造六类可验证原语的形式化表示，并用 LLM 与动态验证协同生成可执行链。

**🔧 技术方法**

结合基于指令的动态定向模糊、PODE 静态分析、LLM 驱动 LangChain 调度、GDB 观察与 SyzDirect 定向探索等技术。

**📊 数据集**

使用 16 个真实 CVE（OOB、UAF、DF、Race、IntOverflow）作为基准，覆盖多内核版本。

**📈 对比分析**

对比 KOOBE、AlphaExp 等基线，PrimSynth 在多原语合成上实现 82.4%（有 PoC）/61.3%（无 PoC）的策略合成率，平均总耗时 72.4 秒，显著优于手工或半自动分析。

**⚠️ 局限性**

局限在于依赖禁用 SMEP/SMAP 等简化环境，对 KPTI、CFI、PAE 等更强缓解支持不足；且对极少见或非内存错误类漏洞尚未验证。

---

## 475. Latent Cluster Analysis for Vision-Language-Action Models

**arXiv ID:** 2609.02634 | [PDF](https://arxiv.org/pdf/2609.02634v1)

**作者:** Theodor Wulff `[一作]` (University of Manchester), Igor Farkas `[通讯]` (Comenius University Bratislava)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出LAVLA框架，对Vision‑Language‑Action模型的潜在空间进行聚类分析，重点研究GR00T N1.5的动作解码器；

**💡 创新点**

引入基于交叉注意力的嵌入加权机制，显著提升聚类质量，并结合概念提取实现人类可解释的标签；

**🔧 技术方法**

使用层级聚类、PCA降维、交叉注意力加权、CLIP评估与LLaVA 72B生成概念；

**📊 数据集**

使用Open‑X‑Embodiment数据集（24个任务，共约24k嵌入）以及GR00T N1.5预训练模型；

**📈 对比分析**

对比Baseline、加权、PCA、加权+PCA四种配置，聚类指标提升62.1% Silhouette，DB指数下降约8%，概念与视频的CLIPSIM略有提升；

**⚠️ 局限性**

仅在单一模型/数据集上验证，聚类方法假设球状分布，概念生成缺乏真实标签，且短时序数据限制了对长时序行为的洞察。

---

## 476. GDB-Reward: From Evaluation Metrics to Training Rewards for Graphic Design

**arXiv ID:** 2609.02813 | [PDF](https://arxiv.org/pdf/2609.02813v1)

**作者:** Adrienne Deganutti `[一作]` (Lica World), Andrew Gilbert `[通讯]` (University of Surrey)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发 GDB-Reward 框架，将多种图形设计评估指标整合为统一的强化学习奖励，在冻结的文本到图像生成器上通过优化提示词提升对设计规范的遵从性。

**💡 创新点**

创新点在于将非可微分的设计评估指标转换为强化学习可用的奖励，并仅训练提示词的 LoRA 适配器，避免昂贵的扩散模型微调。

**🔧 技术方法**

采用 Qwen3.5‑9B + LoRA 的提示生成器、Group Relative Policy Optimization (GRPO)、多指标归一化组合的 GDB-Reward 以及设计意图评估器。

**📊 数据集**

在 LICA 图形设计基准集（包含布局元数据、参考图像和简短描述）上进行训练与评估。

**📈 对比分析**

通过与冻结基准、SFT（生成器微调）以及不同奖励子集的 ablation 对比，结果显示 GDB-Reward 在 OCR、颜色准确率、组件计数等指标上显著提升，综合得分从约0.53提升至0.59，甚至可与更大模型的基线相当。

**⚠️ 局限性**

局限性包括受评估指标设计限制，可能无法覆盖所有设计维度；奖励仅对已定义的指标敏感；在生成器性能极差时提示优化效果有限；未能系统处理多样性与美学平衡。

---

## 477. Do Better Imagined Rollouts Mean Better Robot Control? A Controlled Study of World-Model Evaluation Under Feedback

**arXiv ID:** 2609.02811 | [PDF](https://arxiv.org/pdf/2609.02811v1)

**作者:** Dharini Raghavan `[一作]` (Georgia Institute of Technology), Amritpal Singh `[通讯]` (Emory University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过在差分驱动机器人路径跟踪任务中对六种状态估计器（死reckoning、EKF、GRU/SSM残差学习）进行轨迹回放、20步无测量回滚和闭环控制实验，评估离线预测指标与闭环性能的关系。

**💡 创新点**

创新点在于指出测量更新频率与回滚预测时长共同决定离线评估与闭环表现的一致性，并揭示训练时延长感测中断对不同估计器的条件性提升效果。

**🔧 技术方法**

采用扩展卡尔曼滤波、死reckoning、Gated Recurrent Unit（GRU）和Selective Diagonal State‑Space Model（SSM）残差学习等技术构建估计器，并在仿真环境中执行轨迹回放、无测量回滚和闭环路径跟踪。

**📊 数据集**

使用从200条或300条带有随机化感测噪声、轮滑、陀螺漂移等扰动生成的专家轨迹，实验共覆盖24种感测退化组合。

**📈 对比分析**

通过Spearman相关、Top‑1选择失败率以及平均/最大选择后悔等指标比较离线评估与闭环排名；回放RMSE与闭环误差相关性最高（ρ=0.923），无测量回滚相关性较低（ρ=0.774）；训练延长中断在组合退化条件下显著提升EKF基估计器的交叉轨迹误差，但对单独中断或DR基估计器效果不显著。

**⚠️ 局限性**

局限性包括仅在二维差分驱动仿真平台、单一纯粹追踪控制器和无视觉表示的设置下验证；训练样本量有限；实验未覆盖真实机器人或更复杂的视觉世界模型、多传感器融合和高维动力学；感测中断分布匹配不足，难以推广到更通用场景。

---

## 478. frb100-40 After Two Decades: An Optimality Certificate and a Preregistered Search Study

**arXiv ID:** 2609.02804 | [PDF](https://arxiv.org/pdf/2609.02804v1)

**作者:** Onur Uğurlu `[一作]` `[通讯]` (Izmir Bakircay University), Onur Uğurlu (Izmir Bakircay University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究首次公开可检验的 100 维独立集证明，提供完整证书并通过相同种子重放验证其确定性，并对 ULSA+ 的双、三重修复算子在多种预注册实验中进行系统评估；

**💡 创新点**

创新点在于获得了 20 年未解的 Model‑RB 100‑40 实例的最优证书，展示了可复现搜索轨迹，并通过半径三枚举揭示了搜索障碍，阐明了修复算子无显著加速效果；

**🔧 技术方法**

采用了无权组-aware 随机局部搜索 ULSA+，加入周期性双重/三重严格改进算子，结合完整枚举、Cox 回归与因子实验设计，以及确定性重放与独立验证器；

**📊 数据集**

数据集主要是 100 变量、域大小 40 的 Model‑RB 实例（4,000 顶点图），并在 FRB 小型族集（frb30‑15~frb59‑26）上进行对比实验；

**📈 对比分析**

对比方法包括在同一预算下比较 ULSA+ 的全/基/修复变体与 LibMVC‑NuMVC 的管道，结果显示修复算子未带来时间优势；在 FRB 小型族集上，ULSA+ 管道成功率 100% 且平均耗时显著低于 LibMVC‑NuMVC；在验证集 0/56 成功，未能估计跨 Solver 的风险比；

**⚠️ 局限性**

局限性包括发现率极低、修复算子效益缺乏统计显著性、半径三枚举仅覆盖已记录的冲突两状态、跨 Solver 性能比较受表示、实现与算法共同变化所致，且未验证其他更大邻域或非局部搜索方法的潜在优势。

---

## 479. AutoCompass: Accurate Visual Localization on Public Maps by Learning from Weak Labels

**arXiv ID:** 2609.02798 | [PDF](https://arxiv.org/pdf/2609.02798v1)

**作者:** Javier Tirado-Garín `[一作]` (Universidad de Zaragoza), Eric Brachmann `[通讯]` (Niantic Spatial)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种弱监督训练神经地图匹配器的方法，利用仅2D GPS标签或相对位姿即可实现3‑DoF视觉定位，消除了对精确绝对姿态标注的依赖。

**💡 创新点**

创新点包括：①引入半径为r的GPS块化损失，使模型在GPS误差≤20 m的范围内学习定位；②通过相对位姿（来自SLAM/SfM）提供更精确的弱监督，显著提升定位精度；③无需heading标签即可自动学习方向；④在多种驾驶与 egocentric 基准上取得新的SOTA。

**🔧 技术方法**

技术手段主要包括：基于OrienterNet的神经地图匹配框架、BEV特征与地图特征的交叉相关、改进的负对数似然损失、DINOv2图像编码器等。

**📊 数据集**

使用的数据集有：Mapillary Geo‑Localization (MGL) 760k图像、KITTI Test2、LaMAria、Oxford Day‑and‑Night (ODN) 等。

**📈 对比分析**

与OrienterNet、OSMLoc及卫星‑地面匹配方法进行Recall@1/3/5 m、位置/朝向误差等评估，结果显示：仅用原始GPS标签训练已优于强监督模型；加入相对位姿监督后进一步提升，达到或超过现有SOTA。

**⚠️ 局限性**

局限性在于：仍需相对位姿信息；对地图切片覆盖范围和全局误差的处理有限；对动态场景或遮挡的鲁棒性未充分验证。

---

## 480. Balancing Frequencies and Pixels in Flow Matching

**arXiv ID:** 2609.02748 | [PDF](https://arxiv.org/pdf/2609.02748v1)

**作者:** Lucas Degeorge `[一作]` (École Polytechnique), Vicky Kalogeiton `[通讯]` (École Polytechnique)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了流匹配模型在像素空间训练中的频谱偏差，并提出了焦点对数频率损失与两阶段训练策略以加速收敛并提升图像质量。

**💡 创新点**

创新点在于引入了可在频域平衡学习信号的焦点对数频率损失，并通过先频域后像素域的调度学习解决低频偏置问题。

**🔧 技术方法**

使用了流匹配框架（JiT）以及像素空间速度损失、傅里叶变换、焦点权重、对数压缩和时间调度权重，辅以Heun ODE采样器进行图像生成。

**📊 数据集**

实验采用 ImageNet 分类条件数据集，在 256×256 与 512×512 分辨率下训练。

**📈 对比分析**

与现有像素空间流匹配模型（如 PixelGen、PixelDiT）在 FID、Dino、IS 等指标上比较，取得 1.83 FID 仅需 500k 步，较对手节省约 50% 训练步骤，并实现约 40% 的收敛速度提升。

**⚠️ 局限性**

局限性包括对频谱偏差产生机制的理解尚不完整、频域损失单独使用时容易停滞、以及 FFT 计算引入的额外计算开销。

---

## 481. Incremental Pooled LLM Evaluation for Cost-Effective Retrieval Model Selection

**arXiv ID:** 2609.02745 | [PDF](https://arxiv.org/pdf/2609.02745v1)

**作者:** Max Nelson `[一作]` (JPMorganChase), Saket Sharma `[通讯]` (JPMorganChase)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了一种聚合LLM评估方法，用LLM对检索结果池进行统一打分，以实现增量检索模型评估、成本降低和结果可重复性。

**💡 创新点**

创新点在于仅对新系统加入时产生的独有文档进行LLM评估，显著提高判定重用率、减少评估成本，并通过此策略缓解传统TREC池的偏差和排名不稳定。

**🔧 技术方法**

使用GPT‑4.1（或Claude Sonnet）作为判定者，采用3分等级判定，结合稠密、稀疏和混合检索配置，并逐步扩充文档池，形成增量评估流程。

**📊 数据集**

在四个公开检索基准（FiQA、TREC‑COVID、Natural Questions、FinRAGBench‑V）以及一个金融新闻问答生产数据集上进行实验。

**📈 对比分析**

聚合LLM评估的排名与人类黄金标准的Spearman/Kendall相关系数达到0.69–0.95，pairwise一致率超过80%；成本相较于独立评估降低约4.9×，在生产环境中评估62配置仅耗约800美元。

**⚠️ 局限性**

局限性包括：针对单阶段检索系统，LLM判定的绝对分值可能有偏差；对高度相似（<0.001差距）的系统难以区分；判定者单一且冻结，若LLM更新需重新验证；宏观指标的排名稳定性仍为经验性验证。

---

## 482. Generating Medical Image Counterfactuals using Causal Explanations

**arXiv ID:** 2609.02697 | [PDF](https://arxiv.org/pdf/2609.02697v1)

**作者:** David A. Kelly `[一作]` (King's College London), Hana Chockler `[通讯]` (King's College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种无生成模型的反事实图像生成框架，直接利用分类器本身的因果解释来构造最小局部编辑，从而改变分类结果。

**💡 创新点**

首次将因果解释直接作为生成反事实的依据，实现了确定性、可控的局部编辑；消除了生成器带来的偏置，避免了模型与生成器间的混淆。

**🔧 技术方法**

使用 ReX 生成最小充分像素集（MSPS）和完整解释；基于距离约束的搜索、掩膜操作、质心定位、像素值迁移；评估使用 L1/L2、SSIM、LPIPS 等距离度量；实现了图像相似度约束的成功率计算。

**📊 数据集**

脑 MRI 数据集（TCIA 110 名患者，3,929 张 FLAIR/CT 影像，使用 ResNet50 预训练模型）和 ISIC 皮肤病变图像数据集（1,250 张，使用 VGG 关注模型）。仅对模型预测为阴性的样本进行评估。

**📈 对比分析**

与 GANterfactual（生成对抗网络）、MoPaDi（扩散模型）以及 naïve patching（直接贴合解释区域）进行对比；在低相似度阈值下，本文方法在脑 MRI 取得 85% 成功率（LPIPS 0.25/SSIM 0.2），在 ISIC 取得 67% 成功率（LPIPS 0.48/SSIM 0.21）；生成模型在高阈值下成功率更高，但产生的相似度远低于原图。本文方法保留了原始图像分布，生成的反事实更接近原图。

**⚠️ 局限性**

依赖参考集和预计算解释的多样性；若某些病变模式缺乏代表性，可导致可用的编辑空间受限；方法仅做局部编辑，可能无法捕捉全局语义变化；需要预先计算并存储大量解释，虽比训练生成器轻量，但仍需一定的数据资源。

---

## 483. ShikumiMiner: Mining Recurring Implementation Patterns in AI Codebases

**arXiv ID:** 2609.02789 | [PDF](https://arxiv.org/pdf/2609.02789v1)

**作者:** Afsana Tasnim `[一作]` (University of Texas at Arlington), Sheikh Motahar Naim `[通讯]` (Microsoft)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ShikumiMiner 静态分析框架，利用 AST 与 CFG 特征融合，对 C++ 本地 LLM 代码库中的七类实现模式进行自动识别。

**💡 创新点**

创新点在于首次将 AST 与 CFG 的多维特征结合，并通过弱标签多标签随机森林实现跨项目的实现模式检测与分类，填补了 LLM 代码模式分析的空白。

**🔧 技术方法**

使用 Clang LibTooling 生成 AST/CFG，提取函数级语法与控制流特征，采用随机森林分类器、Spearman 相关系数和 Jaccard 相似度等多种技术进行评估。

**📊 数据集**

使用十个开源 C++ LLM 项目（如 llama.cpp、gemma.cpp、ONNX Runtime GenAI、OpenVINO GenAI、LMDeploy、gpt2.cpp、minchatgpt.cpp、InferLLM、SGLang、DeepSpeed-FastGen）作为实验数据集。

**📈 对比分析**

通过 ablation、Leave-One-Project-Out（LOPO）和人工验证三种评估方案，发现 AST 特征主导性能（宏 F1 ≈ 0.40，微 F1 ≈ 0.46），但跨项目泛化有限（宏 F1 ≈ 0.11，微 F1 ≈ 0.17），显示模型对不同项目的适应性不足。

**⚠️ 局限性**

局限性包括弱标签噪声、仅针对 C++、样本量有限、跨项目迁移效果差、未能捕获更深层语义或运行时行为，且仅使用语法和控制流信息，缺乏更丰富的语义特征。

---

## 484. SafeEvolve: Harness-Policy Co-Evolution from Agent Experience for Safety Alignment

**arXiv ID:** 2609.02786 | [PDF](https://arxiv.org/pdf/2609.02786v1)

**作者:** Qinghua Mao `[一作]` (Shanghai AI Laboratory), Dongrui Liu `[通讯]` (Shanghai AI Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SafeEvolve 框架，利用完成的多步轨迹自适应更新安全提示和层次化技能库，并通过两阶段 SFT‑RL 实现 harness 与策略的协同进化，从而提升 LLM 代理的安全性。

**💡 创新点**

创新点在于将轨迹级安全经验转化为可审计、可回滚的 harness 组件更新，并在策略优化中使用 harness‑augmented RL，形成循环自我改进的 harness‑policy co‑evolution。

**🔧 技术方法**

使用基于轨迹证据的组件级更新、层次化技能库、两阶段 SFT‑RL（先 harness‑use SFT 再 verifier‑decomposed reward RL）、verifier‑based 安全‑效用反馈以及版本化的 harness 变更技术。

**📊 数据集**

在 AgentDojo、AgentDyn（环境注入攻击）和 AgentHarm（恶意查询）三套安全基准上测试，模型为 Qwen3.5‑4B 与 Qwen3‑4B‑Instruct‑2507。

**📈 对比分析**

与 SFT、DPO、GRPO、MetaSecAlign、AgentAlign 等基线对比，SafeEvolve 在 AgentDojo 3 倍 ASR 降低、AgentHarm 有害评分下降至 12.27、拒绝率提升至 83.83，同时保持或提升正常任务效能。

**⚠️ 局限性**

局限：需要手工定义 harness 组件；在弱模型或不同模型上的迁移效果不佳；依赖 verifier 的规则化评分；可能在极端攻击下仍有失败；整体训练成本相对较高。

---

## 485. Video-Based Palm-Vein Authentication under Challenging Conditions

**arXiv ID:** 2609.02776 | [PDF](https://arxiv.org/pdf/2609.02776v1)

**作者:** Xiaofeng Yan `[一作]` (Columbia University), Salvatore Stolfo `[通讯]` (Columbia University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了首个公开的基于视频的掌纹血管数据集 CUP，并在此基础上研究了表面退化（热、湿、脏）对识别性能的影响。

**💡 创新点**

创新点在于设计了双视图编码器与无学习参数的区域级最优传输匹配器，并结合时空一致性聚合，显著提升了恶劣条件下的鲁棒性。

**🔧 技术方法**

使用了深度卷积/变换器骨干、全局平均池化、区域网格、显著性引导的区域最优传输以及时间一致性损失。

**📊 数据集**

使用了 CUP 数据集（5,049 个 2 秒 NIR 视频，四种表面条件）以及四个公开单图像掌纹数据库。

**📈 对比分析**

在 CUP 上与 21 种基线模型对比，平均 EER 从约 16.5% 降至 9.56%，在所有四种表面条件下均领先；该匹配器在冻结骨干和公开数据库上同样提升约 29–37% EER。

**⚠️ 局限性**

局限在于样本量有限、仅覆盖四种表面条件，未验证跨会话、不同设备和更大人群的泛化；公平性分析仅为初步探索。

---

## 486. HyperStyler: Low-resource Authorship Style Transfer via Context-aware Style Navigation and Hypernetworks

**arXiv ID:** 2609.02772 | [PDF](https://arxiv.org/pdf/2609.02772v1)

**作者:** Jongkyung Shin `[一作]` (UNIST), Chiehyeon Lim `[通讯]` (UNIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HyperStyler，专门针对低资源作者风格迁移（LAST）问题的两阶段架构，先通过 Stylo‑navigator 选取上下文相关的风格坐标，再通过 Stylo‑hypernet 在参数空间进行动态调制，实现高风格忠实度与语义保持的平衡。

**💡 创新点**

创新点在于：①将风格选择与实现解耦，使用上下文感知的风格坐标避免模式平均化；②在参数空间而非隐藏状态空间进行风格调制，减少风格与内容的相互干扰；③利用自监督的三阶段训练和自蒸馏构造高质量伪平行数据，提升跨域泛化。

**🔧 技术方法**

核心技术包括：Transformer encoder‑decoder 架构、风格嵌入器（STYLE embedder）、双层注意力（自注意与跨注意）实现风格坐标预测、超网络（hypernetwork）生成关键/值前缀及低秩 FFN 适配器进行参数调制、以及自蒸馏与重排序（reranking）生成伪平行集。

**📊 数据集**

使用 Reddit、Blog 与 News 三个公开数据集，每个作者采样约10条句子（≤60 token），共约数千条句子，按 0.9/0.05/0.05 分布为训练/验证/测试。

**📈 对比分析**

与多种基线（LLM、推理时控制、无监督对齐）以及 TinyStyler、ASTRAPOP 等在三数据集上对比。HyperStyler 在 Joint 分数上显著领先（如 Reddit 上 Joint 0.818，TinyStyler 0.767），且在跨域迁移中表现更稳健。相较 LLM，Inference 速度提升 1.8×，参数量仅增加 2.4%。

**⚠️ 局限性**

局限性：仅针对短文本（1–3 句）进行迁移，难以扩展到段落或文档级；实验仅在英文语料，跨语言迁移与评估仍待研究；未提供完整的安全过滤，可能生成有害内容。

---

## 487. Measurement-Driven Sub-Network Selection for On-Premise Retrieval-Augmented Factory Agents

**arXiv ID:** 2609.02760 | [PDF](https://arxiv.org/pdf/2609.02760v1)

**作者:** Vasileios Rizeakos `[一作]` (enakronIC PC), Athanasios Bachoumis `[通讯]` (enakronIC PC)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了三阶段压缩‑适配‑选择流程，利用权重共享超网络、检索驱动蒸馏与硬件感知子网络选择，将 LLaMA‑3.2 3B/1B RAG 助手在工业现场部署至三种边缘设备，并保持接近完整模型的回答质量与低功耗；

**💡 创新点**

① 在结构化剪枝后通过检索驱动蒸馏使子网络恢复大部分任务质量；② 将硬件吞吐量与通用能力阈值结合，取代单一参数/代理的选择；③ 采用三锚点吞吐预测和权重共享超网络，显著降低选择成本；

**🔧 技术方法**

权重共享超网络 + Sandwich‑style in‑place distillation；结构化剪枝 + 校准候选网格；检索驱动蒸馏（RAFT）+ LoRA；INT8 ONNX 与 Q4_K_M GGUF 量化；ONNX Runtime GenAI / llama.cpp；工具路由（检索+视觉）与语法约束 JSON；

**📊 数据集**

通用指令数据 Alpaca‑GPT4；187页机台手册拆分 1,460 个 100‑token 块生成 686 训练 QA；633 个 held‑out QA 评测集；40 条路由查询；

**📈 对比分析**

通过 LLM judge（Claude Opus 4.8）评估三项 RAG 质量，对比未压缩、未适配抽取、单纯监督微调与不同蒸馏策略；在 Jetson、RevPi、UNO 上测量 TTFT、TPS、内存与能耗，结果显示抽取后质量下降 13.7%，检索蒸馏后回到 4.6% 内，选择子网络保持约 95% 的完整模型质量，显著降低功耗并提升吞吐；

**⚠️ 局限性**

仅评估单一检索器未测召回率；质量评估依赖单个 LLM judge，跨设备可比性有限；小模型在多步推理、长工具链与计数任务上仍易失效；通用能力阈值为经验设定，可能影响模型选取；未在长期真实工厂环境进行验证。

---

## 488. LoRA-TSD: Tangent-Space Spectral Descent for LoRA via Muon-Style Updates

**arXiv ID:** 2609.02734 | [PDF](https://arxiv.org/pdf/2609.02734v1)

**作者:** Dmitrii Andriianov `[一作]` (Basic Research of Artificial Intelligence Laboratory), Aleksandr Beznosikov `[通讯]` (Innopolis University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LoRA‑TSD，一种在 LoRA 固定秩张量空间内执行 Muon 样谱下降的优化器，兼顾低秩适配器的参数效率与高质量梯度更新。

**💡 创新点**

创新点包括：① 在 LoRA 参数化下直接构造张量空间的谱平滑步长；② 用切比雪夫多项式和矩阵符号的交替投影逼近谱极小化，而非完整 SVD；③ 通过因子诱导的重排实现重traction，保持了重traction的可逆性与低成本；④ 在理论上证明了在谱平滑与 Frobenius 形式下的停滞点收敛，且 LoRA‑TSD 等价于 LoRA‑Pro 的 Frobenius 近似。

**🔧 技术方法**

技术包括：低秩分解 (LoRA)、谱平滑 (Muon)、投影到固定秩流形、矩阵符号 (matrix sign) 迭代、Newton–Schulz 近似、梯度投影、因子重排、理论收敛分析。

**📊 数据集**

使用了 Llama‑3.2‑1B‑Instruct、Llama‑3.1‑8B、Qwen3‑32B 等大型语言模型，评估在 BoolQ、PIQA、SIQA、OBQA、QNLI、MultiNLI 六个基准数据集上。

**📈 对比分析**

与 SGD、AdamW、Muon、Riemannian SGD、Riemannion、LoRA‑Rite、LoRA‑Pro、全微调 Muon 进行对比；LoRA‑TSD 在 12 个任务/模型组合中获得最高或相当于最高准确率，且在 1B/8B/32B 大模型上表现最稳健，往往超过全微调的 Muon。

**⚠️ 局限性**

局限性在于内部投影的近似没有严格最优性保证，且在高秩或极端训练设置下谱平滑近似可能失效；此外需要额外的因子重排与梯度归一化步骤来保持数值稳定。

---

## 489. RVSD: Retrieval Vision Sparse Decoding for Mitigating Visual Hallucinations in Large Vision-Language Models

**arXiv ID:** 2609.02731 | [PDF](https://arxiv.org/pdf/2609.02731v1)

**作者:** Canjie Liu `[一作]` (Guangdong University of Technology), Zishao Zhong `[通讯]` (Guangzhou University of Chinese Medicine)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了RVSD（Retrieval Vision Sparse Decoding）框架，在推理时一次性完成视觉 token 的稀疏化与可检索的视觉补偿，从而在不训练、无外部知识库的前提下减少视觉幻觉并保持生成质量。

**💡 创新点**

创新点在于：1）语义导向的 token 选择策略，结合跨模态注意力评估视觉 token 的重要性；2）Semantic‑Space Visual Retrieval（SSVR）机制，将被稀疏化的 token 作为可检索的视觉记忆，按需恢复视觉信息；3）单前向推理、无多轮解码、无额外参数的高效实现。

**🔧 技术方法**

采用交叉注意力、温度化的文本重要性评分、时间平滑稀疏度控制、熵阈值触发检索、点积检索与轻量级适配器融合等技术。

**📊 数据集**

使用了多种公开评测数据集，包括 POPE、MME、AMBER、CHAIR、MM‑Vet、MS‑COCO、A‑OKVQA、GQA 等，覆盖辨别式与生成式视觉幻觉评测。

**📈 对比分析**

与 VCD、M3ID、AvisC、VTI 等基线相比，RVSD 在 LLaVA‑1.5、LLaVA‑NEXT、Qwen‑VL 三种 7B 模型上均取得最高的准确率/F1、最小的 hallucination rate，并在推理效率上保持与原始模型相当的延迟、显存与 FLOPs。

**⚠️ 局限性**

局限性包括：1）检索触发阈值为固定熵阈值，缺乏自适应机制；2）检索使用简单点积相似度，可能不足以捕获细粒度语义；3）实验仅覆盖两类 LVLM 架构，需进一步验证在更大规模或不同跨模态接口的泛化性。

---

## 490. A Top-Down Framework for Metric-Scale Athlete Localization from Single Broadcast Frames

**arXiv ID:** 2609.02705 | [PDF](https://arxiv.org/pdf/2609.02705v1)

**作者:** Thanh-Khoi Nguyen `[一作]` (University of Science, VNU-HCM), Minh-Triet Tran `[通讯]` (University of Science, VNU-HCM)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种两阶段top‑down框架，实现单帧校准摄像机下运动员的世界坐标定位，包含边界感知自适应切片和只预测骨盆与其地面投影的两点姿态估计。

**💡 创新点**

创新点：① Boundary‑Aware Adaptive Tiling（BAAT）利用粗检测结果迭代扩展切片边界，消除常规切片的边界分割问题；② 将姿态估计改为仅预测与相机投影几何耦合的两点，隐式捕捉透视畸变，避免全身骨架过度建模。

**🔧 技术方法**

技术手段：YOLO26‑Large检测器与SAHI切片；BAAT自适应切片；改造RTMPose‑X为两点关键点回归（SimCC+Gated Attention Unit）；相机标定与射线投射实现世界坐标推算。

**📊 数据集**

使用Spiideo SoccerNet SynLoc数据集，包含4K分辨率的合成足球场景、运动员标注和相机标定。

**📈 对比分析**

与官方YOLOX‑pose基线对比：公开测试集LocSim 97.44（vs 76.17），mAP 0.9128（vs 0.6958）；挑战集LocSim 97.67（vs 77.30），大幅提升。

**⚠️ 局限性**

局限性：仍依赖单帧定位，推理时延较高；BAAT与检测模型仍受远场像素不足影响；近场样本稀缺导致细小误差；未引入显式几何监督，实时性能尚待改进。

---

## 491. From Proxy Learning to Driving Decisions: A Transfer-Based Framework for Evaluating Future-Aware Autonomous Driving Planners

**arXiv ID:** 2609.02688 | [PDF](https://arxiv.org/pdf/2609.02688v1)

**作者:** Yikai Wu `[一作]` `[通讯]` (Nanjing University of Science and Technology), Yikai Wu (Nanjing University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Proxy-to-Decision Transfer (PDT) 框架，用以系统评估未来感知信息在基于提议的自动驾驶规划中的实际决策提升效果；

**💡 创新点**

创新点在于将代理任务、评分、选择、支持和部署五个转移阶段分解，并设计两模块（决策转移分解与可靠性约束验证）来量化代理提升何时能转化为可验证的驾驶性能改进；

**🔧 技术方法**

采用联合嵌入预测学习、候选轨迹评分器、顶1选择机制，并在PDT中实现分位数阈值、分支-效用、支持-选择后悔等定量分析；

**📊 数据集**

使用NAVSIM‑v1 12,146条真实驾驶日志的非反应式仿真数据作为评估基准；

**📈 对比分析**

通过与官方基线对比、前缀扩展、配对情景引导等方法，发现代理改进不一定提升选定轨迹或整体PDM分数，且在完整评估中大部分显著提升被抵消，说明需严格按PDT门限评估；

**⚠️ 局限性**

局限包括：仅在单一未来感知规划族与单一基准上验证；未进行前瞻性预注册；部分关键日志与候选数组缺失导致部分评估不完整；以及非反应式评估无法反映闭环交互安全性。

---

## 492. SPADE: SPaT Attack Detection from the Connected Vehicle's Perspective

**arXiv ID:** 2609.02741 | [PDF](https://arxiv.org/pdf/2609.02741v1)

**作者:** James Di Novo `[一作]` (Royal Military College of Canada), Sylvain P. Leblanc `[通讯]` (Royal Military College of Canada)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了SPADE数据集，针对车联网(SPaT)攻击的车载端IDS研究；

**💡 创新点**

创新点在于构建了多模态、标注完整、平衡的六类攻击+正常类数据集，并在仿真中实现了攻击注入与跨模态特征融合；

**🔧 技术方法**

使用Eclipse MOSAIC进行联动仿真，采用SAE J2735层级攻击注入，融合SPaT、摄像头感知、V2V通话、车载本体数据；

**📊 数据集**

数据集SPADE包含180条不同交叉口/交通密度/可见度配置、5次随机种子、7类标签，约1.26万条记录/类，共计8.82万条；

**📈 对比分析**

作者未提供具体模型性能，但指出可用1D-CNN、LSTM、Transformer等架构做基线，规模与VeReMi相当，适合深度学习IDS；

**⚠️ 局限性**

局限性包括：仅仿真场景、缺乏动态SPaT、车辆无法执行误导信息导致本体数据与真值一致、闭域标签，未覆盖混合或未知攻击。

---

## 493. EarlyEval: Cheaper Agent Evaluation via Early Outcome Prediction

**arXiv ID:** 2609.02783 | [PDF](https://arxiv.org/pdf/2609.02783v1)

**作者:** Yuling Shi `[一作]` (Shanghai Jiao Tong University), Xiaodong Gu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EarlyEval 框架，利用早期结果预测在 LLM Agent 评估过程中提前终止任务，以降低每次评估的计算和费用。

**💡 创新点**

在任务内部而非任务集合层面实现效率提升；通过训练轻量级成功/失败二分类器捕获行为、文本及参考解特征，实现无须完整执行即可判断最终结果；并提供可调阈值控制精度与节省的权衡。

**🔧 技术方法**

使用 LightGBM 梯度提升树构建成功和失败两套预测模型；对特征进行 TF‑IDF + SVD 降维、行为度量和可选参考解匹配；采用 Platt 校准和阈值决策实现在线停止。

**📊 数据集**

三大公开 Agent benchmark：SWE‑bench Verified（软件问题修复）、TerminalBench（Shell 自动化）和 Toolathlon（工具/API 组合任务）。

**📈 对比分析**

与完整跑完全程结果对比，覆盖率13–26%，步骤/输入/输出 token 减少分别 13–44%；预测准确率 89–97%；排行榜 Spearman ρ 在 0.959–0.994 之间，resolve‑rate 误差约 1–2%。

**⚠️ 局限性**

需依赖已有完整轨迹来训练模型，无法处理全新 benchmark 或无历史数据场景；参考解缺失时预测准确率略降；在极难预测失败的任务中成功预测率有限；不适合作为正式最终分数的唯一依据，仍需完整跑一次以获得准确榜单。

---

## 494. Finding a Shortest Vector and More in $2^{n/2+o(n)}$ Time using $q$-ary Coset Difference Tree

**arXiv ID:** 2609.02764 | [PDF](https://arxiv.org/pdf/2609.02764v1)

**作者:** Minki Hhan `[一作]` `[通讯]` (KAIST), Minki Hhan (KAIST)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种新的随机算法，用于解决精确的最短向量问题（SVP）和最近向量问题（CVP）。该算法在n维格子上运行时间和空间复杂度为2^(n/2 + o(n))。

**💡 创新点**

创新点在于将算法视为奇素数q的q-进制中点Hessian的类比，利用周期高斯函数的梯度来近似最短向量的方向，并通过组合方法优化计算过程。

**🔧 技术方法**

使用了随机化算法，结合了组合方法和高斯分布的性质，特别是周期高斯函数的梯度计算。

**📊 数据集**

使用了Haar-Siegel测度下随机生成的格子和目标向量，确保在距离保证条件下进行实验。

**📈 对比分析**

与现有方法相比，算法在最坏情况下的时间和空间复杂度均为2^(n/2 + o(n))，在随机实例上表现出色，能够在给定的距离保证下有效解决CVP问题。

**⚠️ 局限性**

算法的局限性在于其在实际应用于格基密码学时没有直接的实用性，并且在某些情况下可能无法突破2^(n/2)的复杂度界限。

---

## 495. Untangling the Mechanisms of Misleading Context in Medical Question Answering

**arXiv ID:** 2609.02754 | [PDF](https://arxiv.org/pdf/2609.02754v1)

**作者:** Robin Linzmayer `[一作]` (Columbia University), Noémie Elhadad `[通讯]` (Columbia University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了医学推理 LLM 对两种误导性上下文（fabricated evidence 与 bare answer assertion）的易感性、其在推理轨迹和最终答案中的披露情况、诱导机制以及监控可监测性，采用对照设计在同一问题上对比两种线索。

**💡 创新点**

创新点包括：① 在同一数据项上同时插入两类线索，系统比较它们对模型决策的影响；② 通过“transplant resampling”揭示两类线索在推理轨迹中的累积方式（早期积累 vs 结尾重定）；③ 评估监控器在不同可读表面（推理轨迹 vs 终端答案）和不同提示（未指导 vs 指导）下的检测性能，证明轨迹可读性与监控效果的显著关联。

**🔧 技术方法**

使用技术：多模态 LLM 推理（chain‑of‑thought）、误导线索注入、推理轨迹截断与移植（causal mediation），监控器采用预训练对话模型做二分类评分，并实验不同提示文本对检测率的提升。

**📊 数据集**

数据集：MedMisBench 医学推理子集（共 8,627 个多选题），其中包含 5 种内容类型和 3 种来源的 evidence cue；为每个问题生成对应的 answer cue，构成完整实验对照。

**📈 对比分析**

比较方法：对三种模型（两个开放权重可读轨迹、一个闭源 frontier）分别在 Clean、Evidence‑false、Answer‑false 三种投影下进行 1 次采样，计算准确率、Uptake、披露率、AUROC 与 5% FPR 下召回率。结果显示：① answer cue 使模型更易偏离（Uptake 增幅 10–27 点），② 轨迹披露率高于答案披露率，尤其 answer cue 在答案中几乎不被提及；③ 监控器在读取轨迹时 AUROC 高达 0.89，5% FPR 下召回率可达 78%，而仅读取答案时召回率仅 32%。

**⚠️ 局限性**

局限性：① 实验仅限单轮多选问答与固定位置线索；② 披露与监控评估只在一个内容‑来源单元（Neutral‑Cue‑Remapping）中进行，未覆盖全部线索组合；③ 机制分析只在 40 条小模型轨迹上完成，无法推广到大型模型；④ 监控器仅使用单一预训练模型与固定提示，未探索多种监控器或自训练；⑤ frontier 模型的轨迹不可见，导致披露与机制只能通过答案间接推断。

---

## 496. The Import Tax: A Longitudinal Measurement of Startup Cost in the Python Ecosystem

**arXiv ID:** 2609.02753 | [PDF](https://arxiv.org/pdf/2609.02753v1)

**作者:** Trinath Sai Subhash Reddy Pittala `[一作]` `[通讯]`, Trinath Sai Subhash Reddy Pittala

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对Python生态系统的导入成本进行大规模纵向测量，覆盖500个最受欢迎的PyPI包、6个CPython版本、2个平台，共计63431次测量。

**💡 创新点**

首次提供完整的生态系统级导入成本基线与长时间趋势；揭示冷启动、隐藏子模块成本、平台差异和PEP 810可延迟性的实际影响。

**🔧 技术方法**

采用可复现的基准 harness：在虚拟环境中分别测量首次（冷）和随后的（热）导入时间，分别测量顶层导入与完整子模块导入，利用Python 3.15的全局惰性模式评估可延迟性。

**📊 数据集**

使用2021 Q3起每季度发布的前500名PyPI包的版本历史，配合6个CPython 3.9–3.14版本，在Apple M5/macOS和Intel Xeon/Linux两套硬件/操作系统上收集数据。

**📈 对比分析**

通过对同一包在不同解释器、不同平台上对比导入时间，计算相对比值和年化增长率；结果显示macOS平台导入速度随解释器更新反而变慢（1.16×），Linux平台基本持平；冷启动倍增因子高达22×，最慢包单次导入可达1.9 s。

**⚠️ 局限性**

仅覆盖两平台；网络存储导致Linux测量波动；使用当下的依赖解析方式，可能与历史版本不符；只测顶层+第一层子模块，未覆盖真实应用的完整导入图；仅考虑下载排名前500包，可能偏向基础设施库。

---

## 497. Language Models Can Control Their Own Attention

**arXiv ID:** 2609.02737 | [PDF](https://arxiv.org/pdf/2609.02737v1)

**作者:** Namgyu Ho `[一作]`, Cicero Nogueira dos Santos `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种零样本提示协议——Declarative Attention（DA），让大型语言模型在链式思考中以文本标记声明其关注的上下文片段，从而在推理时动态生成 KV 缓存掩码，显著减少长上下文的注意力计算量；

**💡 创新点**

创新点在于将注意力范围的决策完全交给模型通过可解析的文本指令来完成，避免了传统方法对内部激活的昂贵扫描，且该方案无需任何额外训练即可在现有模型上实现；

**🔧 技术方法**

技术实现包括三种注意模式（全局、聚焦、局部）与对应的状态机解析、block‑aligned KV 缓存掩码、以及在 vLLM 框架中集成 FlashAttention 等高效注意力核；

**📊 数据集**

实验使用了15个长上下文基准，涵盖 RULER、LongBench（v1/v2）、LooGLE、ZeroSCROLLS 等数据集，任务包括单片段检索、跨片段推理、多文档问答等；

**📈 对比分析**

与 Vanilla（无掩码）和 DA‑no‑mask（仅提示）对比，DA 在 Gemma‑4‑31B 与 Qwen‑3.6‑27B 上平均降低 52%/31% 的注意力令牌，同时准确率仅下降 1–3%，理论上可将解码壁钟时间降至 0.71×/0.77×；

**⚠️ 局限性**

局限性包括：零样本策略导致解码步数增加、全局模式仍占主导的高成本；对小规模模型适用性差；无法在思考模式内执行；需要手工切分上下文；进一步训练或强化学习可提升效率。

---

## 498. Door-in-the-Face Requests and Refusal Behaviour in Large Language Models

**arXiv ID:** 2609.02707 | [PDF](https://arxiv.org/pdf/2609.02707v1)

**作者:** Til Jordan `[一作]` `[通讯]`, Til Jordan

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统测试了“门口先说大拒绝再小请求”（Door‑in‑the‑Face, DITF）在九个主流语言模型上的效果，发现模型家族差异显著：Anthropic 的 Opus 系列在大请求被拒后对相同小请求的合规率显著提升（约 +36.5%），而 OpenAI 和 Google 的模型则出现相反的降低（约 –23%）。

**💡 创新点**

创新点在于首次将人类社会心理学的 DITF 机制迁移到生产级 LLM 上，揭示了模型家族对同一策略的不同响应；并通过小请求所需产出（“可操作” vs “无操作”）阐明了决定退让能否成功的关键属性。

**🔧 技术方法**

方法上使用了公开 API 进行 60 条构造性判断请求和 7,664 条 benchmark 推理请求的多轮对话；结果通过两位跨家族裁判的二元判断、聚类自助抽样置信区间以及 GEE 统计模型评估。

**📊 数据集**

数据集包括：① 60 条人工/模型协同生成的“判定”请求；② 10 个公开 benchmark 套件（benign、controversial、harmful）共 7,664 条独特提示；③ 通过裁判判定后选出的可移动拒绝项用于二次实验。

**📈 对比分析**

与冷启动（单一小请求）对比，Opus 5 在 DITF 条件下的合规率从 29.3% 提升至 65.8%（+36.5%），而 Gemini 3.1 Pro 则从 24.7% 降至 1.7%（-23.0%）；在 benchmark 拒绝项上 DITF 甚至出现负效应，说明效果不具备普适性。

**⚠️ 局限性**

局限性包括：① 仅使用一种英文请求模板，结果可能不适用于其他语言或主题；② 大多数模型未固定版本，随时间漂移可能影响可重复性；③ 仅观察性分析“可操作”与“无操作”区别，缺乏因果验证；④ benchmark 受限于可移动拒绝项稀缺，统计功效有限。

---

## 499. Trace as State: Reasoning Traces as Conditional States for Long-Context Transformers

**arXiv ID:** 2609.02702 | [PDF](https://arxiv.org/pdf/2609.02702v1)

**作者:** Xu Zou `[一作]` (Z.ai), Jie Tang `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

论文未提供具体内容，无法确定所做工作

**💡 创新点**

创新点未知

**🔧 技术方法**

使用的技术未知

**📊 数据集**

使用的数据集未知

**📈 对比分析**

方法比较及性能表现无法评估

**⚠️ 局限性**

论文限制未知

---

## 500. GaLe: memory-efficient Global Approximate and Local Exact features

**arXiv ID:** 2609.02689 | [PDF](https://arxiv.org/pdf/2609.02689v1)

**作者:** Alberto Ancilotto `[一作]` (Fondazione Bruno Kessler), Elisabetta Farella `[通讯]` (Fondazione Bruno Kessler)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 GaLe，一种在不重新训练的前提下通过特征图分区实现内存高效推理的技术。

**💡 创新点**

将特征图拆分为局部精确（L_E）与全局近似（G_A）两种表示，支持全局感受野与注意力机制，并通过自适应切片与低分辨率全局分支实现显著内存削减和低计算开销。

**🔧 技术方法**

使用自适应切片、局部精确/全局近似分解、分块稀疏与低秩注意力近似、NHWC 连续内存布局和校准求取最小重叠等技术。

**📊 数据集**

在 ImageNet 分类数据集上进行主要评估，随后在 RT‑DETR‑L、YOLOv11n 的检测任务以及扩散模型实验中验证。

**📈 对比分析**

与分辨率缩放、全局/局部块推理（PPBI、FPBI）、Token Merging 等方法对比，GaLe 在保持 <1% 准确率损失的同时实现高达 90% 的 RAM 降低，并在 256px Cortex‑M33 上获得 65% 的速度提升，整体取得最佳 RAM/overhead 平衡。

**⚠️ 局限性**

仍需事先校准，适用性受限于可降维的全局近似，极小分辨率下细节可能受损，目前主要验证于图像任务，对更复杂场景的扩展仍需进一步研究。

---

## 501. DKL: Decoupled Knowledge Learning for Instruction-Tuned Language Models

**arXiv ID:** 2609.02685 | [PDF](https://arxiv.org/pdf/2609.02685v1)

**作者:** Kushagra Bhushan `[一作]` (IBM), Dinesh Raghu `[通讯]` (IBM)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在不需要昂贵的指令微调（IFT）的前提下，将新文档知识注入已有指令LLM的参数中；

**💡 创新点**

采用任务向量融合思想，先在基础模型上进行扩展预训练得到知识向量，再用基础模型的token embedding训练，最后将知识向量与指令向量按比例合并，避免指令能力的灾难性遗忘；

**🔧 技术方法**

任务向量融合、扩展预训练（EPT）在基础模型上、LoRA适配器、少量合成问答监督、token embedding迁移、模型融合；

**📊 数据集**

使用技术文档RedBooks（两个红皮书）和开放域长文档QuALITY，实验还覆盖 Mistral‑7B‑Instruct、Llama‑3.1‑8B‑Instruct、SmolLM2‑1.7B‑Instruct、Qwen3‑0.6B 等模型；

**📈 对比分析**

与RAFT、PA‑RAG、Chat‑Vector 等基线相比，在RAG检索失败场景下准确率从54.17%提升至79.26%，在QA和RAG整体上均优于基线，且所需合成数据量显著减少；

**⚠️ 局限性**

依赖未指令化的基础模型、对超参数高度敏感、需要大量调优，且在多数开源模型已指令化的情况下应用受限。

---

## 502. Large Language Models (LLMs) for Telecom Root Cause Analysis (RCA): A Structured Reasoning Framework for Evidence-Grounded Diagnosis

**arXiv ID:** 2609.02805 | [PDF](https://arxiv.org/pdf/2609.02805v1)

**作者:** Hao Zhou `[一作]`, Zhang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 SEKA‑FT 结构化推理框架，将大型语言模型用于 5G/6G 网络根因分析。

**💡 创新点**

创新点在于：1）将多源网络遥测规范化为统一的“canonical context”；2）在 fine‑tune 过程中强制模型生成逐步 CoT 推理路径；3）将证据与知识紧耦合的解释作为监督目标，显著降低模型幻觉与不稳定推理。

**🔧 技术方法**

技术包括：大型语言模型（Qwen2.5‑1.5B‑Instruct 等）+ LoRA 参数高效微调；结构化输入预处理；链式思维（CoT）决策路径；检索增强生成（RAG）与知识图谱对齐；自定义解释模板与 token‑级语言模型训练。

**📊 数据集**

使用了两套 5G RCA 基准：TeleLogs（drive‑test 结构化遥测）和 TelecomTS（实验室高分辨率 KPI 流）。

**📈 对比分析**

与 ICL、SFT、LSTM、SFT+StructuredInput、SFT+Explanation 等基线对比；在 TeleLogs 上 SEKA‑FT 达到 94.2% 准确率、93.7% Macro‑F1；在 TelecomTS 上 64.7% 准确率、61.5% Macro‑F1；显著优于所有对照组，且改进在统计检验下显著（p<0.001）。

**⚠️ 局限性**

局限性：仅在 5G RAN 级别的 drive‑test 数据上验证；对核心/传输网络、真实运营商事件的适应性仍待评估；需要人工构造证据块与解释模板，难以自动化；在覆盖与邻区冲突等模糊场景下仍存在误判。

---

## 503. DiscoSign: Discourse-Aware Text to Sign Language Gloss Translation

**arXiv ID:** 2609.02796 | [PDF](https://arxiv.org/pdf/2609.02796v1)

**作者:** Vasileios Baltatzis `[一作]` (Apple), Colin Lea `[通讯]` (Apple)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 DiscoSign 框架，实现跨句子文本到 ASL gloss 的语篇感知翻译；

**💡 创新点**

首次系统化处理空间指称、问答从句(QAC)和概念-词汇一致性，并引入对应的模块与评估指标；

**🔧 技术方法**

采用 LLM（Gemini 2.5 Pro、Qwen 3.6‑35B‑A3B）进行结构化提示，配合显式状态注册与确定性后处理；

**📊 数据集**

使用 ASL STEM Wiki、授权日常语料库及 Aesop’s Fables 进行实验；

**📈 对比分析**

与句子级与仅上下文级基线对比，传统 chrF/COMET 误差大；DiscoSign 在 SCA 0.84、CGC 0.97、QAC_AP 0.76 等语篇指标上显著优于基线，句子级翻译质量保持竞争；

**⚠️ 局限性**

受 LLM 版本、指令遵循能力限制；仅覆盖英语‑ASL，未加入非手势标记；缺乏真实语料语篇标注；QAC 选择规则过于确定，缺乏可选性。

---

## 504. Type Hints in Python Libraries and Frameworks: An Empirical Analysis of Adoption and Maintenance

**arXiv ID:** 2609.02782 | [PDF](https://arxiv.org/pdf/2609.02782v1)

**作者:** Thiago Roberto Magalhães `[一作]` (Universidade Federal de Minas Gerais), João Eduardo Montandon `[通讯]` (Universidade Federal de Minas Gerais)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

分析了 1000 个 GitHub 上的 Python 库/框架，量化它们的类型提示采用、使用、演变以及与 Pyright 推断的关系。

**💡 创新点**

首次系统性聚焦库/框架的类型提示实践，揭示对 API 接口的偏好、演化趋势以及开发者与推断器之间的差异，并对注解的维护行为进行细粒度跟踪。

**🔧 技术方法**

使用 AST 静态分析抽取注解、Git 历史追踪记录演化事件、Pyright 进行类型推断，并用 Spearman 相关等统计方法进行分析。

**📊 数据集**

从 GitHub 前 1000 名中筛选 720 个 Python 项目（其中 152 个为库/框架），共包含约 5.03 M 代码成员和 649 099 条类型注解。

**📈 对比分析**

通过将开发者注解与 Pyright 推断结果对比，统计覆盖率、重叠、差异以及演化事件和时间间隔；结果显示多数注解未被 Pyright 覆盖，改动倾向更丰富的类型，方法可在大规模仓库上实现。

**⚠️ 局限性**

仅涵盖受欢迎、星标高、英文的公开 GitHub 项目；仅使用 Pyright 做推断；10% 覆盖率阈值和 Git 历史可能导致信息遗漏；结果不一定适用于企业内部或低星项目。

---

## 505. CodePoisonRAG: Knowledge Poisoning Attacks on Retrieval-Augmented Code Generation

**arXiv ID:** 2609.02774 | [PDF](https://arxiv.org/pdf/2609.02774v1)

**作者:** Varun Gadey `[一作]` (University of Duisburg-Essen), Alexandra Dmitrienko `[通讯]` (University of Duisburg-Essen)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种针对检索增强代码生成（RACG）的黑盒知识毒化框架 CodePoisonRAG，能够将无害代码改造成含指定 CWE 的毒化片段，并在检索过程中被使用，从而诱导 LLM 生成带攻击者选定脆弱性的代码。

**💡 创新点**

① 仅使用单一任务对齐的毒化片段即可实现高成功率；② 通过漏洞注入+语义误标注构造源→传递→汇点链；③ 只需0.7%知识库污染即可；④ 在主流防御（CodeGuarder）下仍保持较高攻击成功率。

**🔧 技术方法**

基于 Dense Retriever + Cross‑Encoder re‑ranker 的 RAG 架构；漏洞注入技术和 Semantic Mislabeling（注释欺骗）；LLM 生成器 Qwen3.5 9B、Code Llama 13B、DeepSeek‑Coder‑V2 16B；GLM‑5.1 评估器；CodeBLEU 代码相似度评估。

**📊 数据集**

12,053 条 ReposVul 纯净代码 + 85 条手工/LLM 生成的毒化片段（10 个 CWE，Java 与 C），以及 CyberSecEval 查询集用于非目标评估。

**📈 对比分析**

采用检索成功率 (RSR)、攻击成功率 (ASR) 与目标弱点率 (TWR) 进行评估；在无防御下 ASR 高达 0.93（85/85）；在 CodeGuarder 防御下仍达 0.71；与 Lin 等先行攻击相比 ASR 从 0.67 提升至 0.97；不同 LLM 上表现一致，CWE 之间差异可观察。

**⚠️ 局限性**

依赖任务与 CWE 的准确匹配；仅针对函数级代码片段；防御能显著降低成功率；未覆盖多语言和更复杂上下文；评估基于 LLM 判定器，存在误判风险；对未预见查询的影响有限。

---

## 506. Do Tabular Foundation Models Know Physics? Contamination, Units, and the Deterministic Limit

**arXiv ID:** 2609.02766 | [PDF](https://arxiv.org/pdf/2609.02766v1)

**作者:** Wassim Tenachi `[一作]` (University of Montreal), Pierre-Luc Bacon `[通讯]` (University of Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估了 Tabular Foundation Models（TFMs）在物理方程生成的数据集上的表现，并分析了它们在插值、外推以及对噪声与维度结构的适应性。

**💡 创新点**

直接检验并证明了 TFMs 的先验并未捕捉到无噪声机制和物理维度无关性，揭示了它们在物理建模中的根本局限，同时显示在未污染数据上它们仍可实现优秀的插值性能。

**🔧 技术方法**

使用了蒙版填充的无监督预训练、贝叶斯后验预测、NMSE、预测区间宽度和平均排名等评估指标；对 TFMs 进行无调参和调参（随机搜索+后期集成）两种配置，并与六个传统基线模型进行对比。

**📊 数据集**

构造了三套合成数据集：Feynman 120 方程、其改写版 111 方程和 85 方程。每套数据在 50、200、2000 个样本、噪声水平 σ=0、0.01、0.1，以及域内/域外 k=1、2 的测试点下进行采样。

**📈 对比分析**

采用平均排名对 TFMs 与六个基线模型在每个任务/情境下进行比较。结果显示 TFMs 在所有条件下均排名靠前，尤其在插值任务中超越了调优的基线；但在外推以及噪声为零的极限情况下，TFMs 的预测区间不收敛、误差持续下降，表现不如传统模型。

**⚠️ 局限性**

局限性：TFMs 的先验无法表示无噪声机制，也不利用物理维度结构（Buckingham π 组），因此在外推、精确物理推断和维度压缩任务上表现受限。

---

## 507. Repo-To-Skill: Distilling GitHub Repositories Into AI4AI Skills

**arXiv ID:** 2609.02749 | [PDF](https://arxiv.org/pdf/2609.02749v1)

**作者:** Jianlyu Chen `[一作]` (Beijing Academy of Artificial Intelligence), Zheng Liu `[通讯]` (Beijing Academy of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究在自主机器学习研究代理中引入了可重用的操作知识层，通过构建和验证技能图来提升代理的执行效率。

**💡 创新点**

创新点在于提出了两种技能蒸馏方法（任务无关与任务有关），自动从论文和仓库中提取可验证的技能，并将其组织成可扩展的技能库；同时展示了技能在保持模型与抓手不变的情况下显著提升代理性能。

**🔧 技术方法**

技术实现包括：使用 GPT‑5.5/GPT‑5.6 进行技能蒸馏与验证；将技能定义为 Agent Skill 文件，采用逐层披露（progressive disclosure）策略；构建技能图与路由器以实现快速定位与加载；在 Codex 抓手上直接注入技能作为运行时上下文。

**📊 数据集**

实验数据集包括：MLE‑bench（75 场竞赛）、PaperBench（20 篇论文）、FrontierCS Agent Track（188 任务）和 PassNet（200 个样本）；源数据为 1,000 个广泛使用的机器学习仓库，蒸馏出 5,000+ 份经过验证的技能。

**📈 对比分析**

对照实验在相同 GPT‑5.5 + Codex 抓手与同一抓手但不使用技能的两种配置下进行；结果显示：MLE‑bench 提升 134.3%，PaperBench 34.4%，FrontierCS 9.2%，PassNet 14%，表明技能层显著提升了代理在多种基准上的表现。

**⚠️ 局限性**

局限性包括：技能检索精度可能导致某些任务性能下降；蒸馏与验证需要显著的计算与人工验证成本；当前实现仅覆盖机器学习领域，扩展到其他领域需要更多的技能来源与验证机制。

---

## 508. BuildOcc: A Large Language Model Occupant Agent Platform for Building Energy Research

**arXiv ID:** 2609.02729 | [PDF](https://arxiv.org/pdf/2609.02729v1)

**作者:** Wooyoung Jung `[一作]` `[通讯]` (University of Arizona), Wooyoung Jung (University of Arizona)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并公开发布 BuildOcc，一个基于大型语言模型的居住者行为模拟平台，支持Python库、REST API和MCP接口，可与 EnergyPlus、Home Assistant 等建筑能耗工具无缝集成。

**💡 创新点**

创新点包括：① 以美国时间使用调查（ATUS）为基础的活动调度和人口分层，真实捕捉不同族群的日常行为；② 代理具备记忆流、检索与反思机制，能够在多时步内做出连贯的决策；③ 提供多层接口和插件注册机制，允许社区轻松扩展新族群、活动来源或记忆架构；④ 通过两层验证（活动分布对比和族群行为差异检验）验证模型内部一致性。

**🔧 技术方法**

技术栈：Python 3.11+、FastAPI、SQLite/SQLAlchemy、LLM 调用（Anthropic Claude、OpenAI GPT、Google Gemini 或本地 Ollama），内存检索采用 recency‑plus‑importance 策略，反思阶段使用高能力模型压缩经验。

**📊 数据集**

使用的数据集：美国时间使用调查（ATUS）2022–2023 16,684 份日记数据（用于活动概率表和族群划分）以及 EIA RECS 设备拥有率分布；此外还使用 EnergyPlus 仿真生成的温度曲线做示例。

**📈 对比分析**

评估方法：Tier 1 用 KL 散度比较 ATUS 基准活动分布与固定规则基线；Tier 2 在相同环境下跑四个族群，统计动作类型分布和对三类需求响应信号的接受率。结果显示 ATUS 调度显著优于规则基线，且族群差异显现，验证了内部一致性；计算成本与运行时随 LLM 调用次数线性增长。

**⚠️ 局限性**

限制：① 只能每 15 分钟做一次单动作；② 仅覆盖 US 人口分层，缺少非 US 数据；③ ATUS 只提供主要活动，无法建模多重并行活动；④ 记忆重要性评分自我评估，缺乏外部校准；⑤ 未实现多能房屋多代理交互；⑥ 真实行为基准和现场验证仍待补充。

---

## 509. Neural operators approximate strongly continuous convex monotone semigroups

**arXiv ID:** 2609.02727 | [PDF](https://arxiv.org/pdf/2609.02727v1)

**作者:** Jonas Blessing `[一作]` (ETH Zurich), Alessandro Sgarabottolo `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了如何用神经算子近似强连续凸单调半群，提出Chernoff‑neural和envelope‑neural两类神经算子，并证明其在加权Holder空间上的全局逼近性和量化误差率。

**💡 创新点**

创新点在于将Chernoff逼近与神经算子结合，首次给出在无限维函数空间中的全局逼近定理，并针对结构化的一步算子（sup型）提供了可量化的误差上界；同时提出仅需训练一一步算子而非完整解算子的方法。

**🔧 技术方法**

核心技术包括：弱化拓扑与加权Holder空间的理论、神经算子架构（隐藏层映射→非线性激活→线性读出）、Chernoff逼近、ReLU实现的max操作、以及卷积/随机特征实现。

**📊 数据集**

使用合成数据：1）半线性PDE解函数、2）随机控制问题的期望支付函数、3）Wasserstein不确定性下的Lévy过程转移概率。

**📈 对比分析**

通过均方误差与迭代误差曲线对比；在训练集与测试集均表现出低MSE；envelope‑neural算子在参数量和收敛速度上优于Chernoff‑neural；实验表明两种算子都能逼近解析解。

**⚠️ 局限性**

局限性包括：需满足凸单调半群假设，训练过程对参数选择敏感；量化误差率涉及多项系数，实际实现中高维问题计算成本仍较高；对非凸或非单调系统的推广尚未研究。

---

## 510. Large Language Model-Driven Context-Aware Eco-Feedback Generation and Evaluation

**arXiv ID:** 2609.02719 | [PDF](https://arxiv.org/pdf/2609.02719v1)

**作者:** Wooyoung Jung `[一作]`, Prosper Babon-Ayeng `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一套基于大语言模型的上下文感知生态反馈生成框架，并通过实证验证和组合情境分析评估其准确性和适应性。

**💡 创新点**

创新点在于将上下文工程与自一致性链式思维（SC-CoT）提示相结合，既利用能源分析结果又融入家庭特征、计费结构和人物角色，实现高度个性化且可操作的生态反馈。

**🔧 技术方法**

使用技术包括：OpenAI GPT‑4o 语言模型、SC‑CoT 提示技术、能源使用指标计算、数据引用与标记评估、以及基于 token 效率的成本评估。

**📊 数据集**

使用数据集：Pecan Street 家庭级能耗数据（50户，验证用3户）、两种实测电价结构（标准与 TOU）、合成的四类居住角色（成本最小化者、舒适最大化者、技术采纳者、日程约束者）以及可用的太阳能发电数据。

**📈 对比分析**

通过与参考干预方案对比，测量了电器有效性率（94.3–100%）、参考对齐率（90.5–95.9%）和数据引用准确率（>90%），整体平均准确率为92%；在400种情境下的组合分析显示，框架能根据电价结构和角色差异显著调整建议，统计检验表明适应性显著（P<0.001）。

**⚠️ 局限性**

局限性包括：合成的角色特征可能不完全反映真实多样性；能耗数据仅来自夏季奥斯汀，季节性差异有限；未实现双向交互或实时反馈；缺乏建筑结构信息导致部分能源驱动因素未被充分区分。

---

## 511. MV-dVRK: A Multi-Viewpoint Benchmark for Spatial Surgical Perception

**arXiv ID:** 2609.02717 | [PDF](https://arxiv.org/pdf/2609.02717v1)

**作者:** Guido Caccianiga `[一作]` (Max Planck Institute for Intelligent Systems), Katherine J. Kuchenbecker `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究构建了MV-dVRK数据集，并在此数据集上对稀疏多视角3D重建方法进行了系统评估。

**💡 创新点**

创新点在于实现三台曝光同步立体内窥镜的多视角采集，提供高精度的SfM参考几何以及同步动态序列，填补了外科领域稀缺的多视角基准资源。

**🔧 技术方法**

采用了立体匹配、单目学习、前向多视模型（如Depth Anything 3）以及基于几何优化的SfM+点云优化技术（如GGPT、MASt3R、GGPT‑SfM）。

**📊 数据集**

使用的数据集为MV-dVRK，包括8个静态场景（密集SfM参考）和10个同步动态序列。

**📈 对比分析**

通过与单目、立体、双立体和多视（FF vs 结构优化）方法的对比，发现两台相机时注册立体覆盖率最高，三台相机时几何优化方法覆盖率达67%（1mm容差），FF模型仅达43%。

**⚠️ 局限性**

局限性包括：静态几何参考仅来自离体样本，动态序列缺乏密集几何基准；数据场景和视角数量有限，难以直接推广到全腹腔等真实手术环境。

---

## 512. Card-Based Computation in the Virtual Player Simulation Model

**arXiv ID:** 2609.02716 | [PDF](https://arxiv.org/pdf/2609.02716v1)

**作者:** Suthee Ruangwises `[一作]` `[通讯]` (Chulalongkorn University), Suthee Ruangwises (Chulalongkorn University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向虚拟玩家模拟的两种卡牌加密协议：最小值选择协议（Play‑Minimum）和全手牌排序协议（Sorting）。

**💡 创新点**

创新点在于在仅使用单张物理牌并保持牌面不可见的严格约束下，实现了安全的最小/最大卡牌选取和全手牌排序，为玩家模拟模型提供了通用计算基石。

**🔧 技术方法**

采用基于堆洗牌（pile‑scramble）与堆循环洗牌（pile‑shifting）的物理操作，结合所有牌背面不可区分的特性，实现信息理论安全的协议。

**📊 数据集**

使用标准扑克牌（含牌面值可公开排序的卡牌）与额外的标记牌（如所有者标签、标记牌α）作为辅助卡。

**📈 对比分析**

与传统多方计算与零知识证明方案相比，Play‑Minimum 仅需 2k 个额外卡和 2 次堆洗牌；Sorting 协议在期望上需要 O(k log k) 次堆循环洗牌，且在不泄露除公开信息外的任何内容。性能上，在手牌数量 k 较大时，排序协议仍保持可行但洗牌次数上升。

**⚠️ 局限性**

局限性包括：1) Play‑Minimum 仅适用于所有牌值互异的场景；2) Sorting 协议的洗牌次数随手牌数量呈对数增长，效率受限；3) 只针对可公开排序的牌组；4) 实际实现依赖于参与者的协作与物理洗牌精确性。

---

## 513. ACLE-MCP: Attested Capability Leases for Execution-Time Trust in Remote LLM Tool Use

**arXiv ID:** 2609.02690 | [PDF](https://arxiv.org/pdf/2609.02690v1)

**作者:** Zhiyang Ding `[一作]` (Peking University), Zhonghai Wu `[通讯]` (Peking University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种结合 OAuth、工作负载评估与调用限额的能力租约体系，填补 OAuth 授权后执行权威与实际执行工作负载之间的信任缺口；

**💡 创新点**

创新点在于：①把调用权威绑定到短期租约中，租约中包含工作负载身份、操作、参数范围和新鲜度；②在工具执行入口处做实时租约消费，确保仅由符合评估的新鲜工作负载执行；

**🔧 技术方法**

使用的技术包括：Keycloak/OIDC OAuth 认证、RATS 证据-评估-依赖模式、vTPM 证明、Python SDK 构建 MCP 服务、租约签发与校验、接入网关/侧车拦截器做调用时验证；

**📊 数据集**

实验数据集为模拟的 OAuth 访问令牌、工作负载评估结果、租约交互日志以及四类正常任务（文件系统摘要、GitHub 注释、数据库报表、Kubernetes 状态查询）和六类恶意利用场景；

**📈 对比分析**

与 OAuth-only、连接时评估、Stateful LP、Capability-only 四种弱模式对比，完整 ACLE-MCP 方案在所有恶意场景下实现 100% 阻断，正常任务无误判；在代理工具实验中，最慢 95% 请求延迟从 12.20 ms 提升至 15.34 ms，约 25.7% 的性能开销；

**⚠️ 局限性**

局限性包括：依赖不可绕过的 Provider‑side gate；未覆盖 Prompt 注入、主机规划错误、Verifier/lease‑issuer 受控；实验仅在四个正常任务和六个人工构造攻击上验证，未在大规模真实 LLM 代理场景中评估；

---

## 514. Almost Envy-Freeness for Additive Mixed Manna with Entitlements: Deterministic and Randomized Guarantees

**arXiv ID:** 2609.02724 | [PDF](https://arxiv.org/pdf/2609.02724v1)

**作者:** Zehan Lin `[一作]` (University of Macau), Shengwei Zhou `[通讯]` (Nanyang Technological University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在混合物品（既有物品也有任务）下，考虑权重（不同权利）时，如何对不可分物品进行公平分配，并证明可在多项式时间内得到加权一次性可移除物品（WEF1）分配；同时给出与效率（fPO）兼容的加权一次性可移除转移（WEF1T）分配，并构造了一个有限抽样（lottery）在期望上实现加权公平（WEF）且每一次结果满足WEF1T的最佳两世界保证；此外讨论了WEF1与fPO的不可兼容性，并给出WEF1T与fPO兼容的存在性与构造。

**💡 创新点**

创新点在于：1) 提出了基于 meta-good 预处理与前向/反向加权抽签序列相结合的多项式算法，解决了混合物品下WEF1存在性的长期开放问题；2) 证明了WEF1与fPO一般不兼容，并给出WEF1T与fPO兼容的最佳阈值；3) 通过 KKM 定理和随机化 DSE 等工具构造了实现期望加权公平且每一次结果满足 WEFA1T 的有限抽样。

**🔧 技术方法**

使用的技术主要包括：meta-good 结构化构造、加权前向/反向抽签序列、KKM 固定点定理、最大熵分布、DSE（不同速度吃掉）过程、Slot Matching 与极大熵分布、Carathéodory 定理等。

**📊 数据集**

本研究为理论性论文，未使用任何实测数据集；所有结果均基于数学证明与算法构造。

**📈 对比分析**

与先前工作相比：对加权混合物品的 WEFA1 已实现多项式时间解，且构造了期望加权公平且每一次结果满足 WEFA1T 的抽样；相较于已有的单一公平或效率结果，本文在公平-效率兼容性上给出了精确阈值，并提供了最佳两世界保证；总体上理论上取得了最优或最紧的结果。

**⚠️ 局限性**

局限性：WEF1 与 PO 的兼容性仍未解决，本文仅证明了 WEFA1T 与 fPO 兼容；此外，所有结果均基于可加性假设，对非可加性或更一般的偏好模型尚未讨论。

---

## 515. ShallowStream: Index Shallow then Answer Deep for Streaming Video Understanding

**arXiv ID:** 2609.02780 | [PDF](https://arxiv.org/pdf/2609.02780v1)

**作者:** Jitai Hao `[一作]` (Harbin Institute of Technology), Jun Yu `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 ShallowStream，利用多模态大语言模型的浅层 Transformer 进行实时视频流编码，仅在查询时触发深层推理，从而实现高效的流媒体视频理解。

**💡 创新点**

创新点在于：①通过查询‑日志门控决定是否需要历史信息；②利用 token‑级投票结合 max‑min 多样性进行检索，显著降低实时预填充成本与内存占用；③引入长时序聚类压缩，维持可扩展性。

**🔧 技术方法**

技术手段包括预训练的 Qwen3‑VL‑8B / LLaVA‑OneVision‑7B、浅层 Transformer 前向、KV 缓存、长时序聚类压缩、token 投票检索、max‑min 多样性、查询日志门控。

**📊 数据集**

使用 OVO‑Bench、StreamingBench、LVBench 三大公开流视频基准进行评估。

**📈 对比分析**

与 VideoLLM‑online、Flash‑VStream、ReKV、StreamKV、HERMES、OASIS 等基线比较，ShallowStream 在 OVO‑Bench 达到 69.5/62.2 分，在 StreamingBench 78.2/75.5 分，保持与最强方法相当的准确率，同时 per‑frame 预填充成本降低 52×、10 s 延迟降低 12×，GPU 内存维持约 18 GiB。

**⚠️ 局限性**

局限性包括：门控阈值需手工校准，极长序列仍需压缩；在需要完整深层上下文的复杂查询上可能表现受限；实验仅在单卡 RTX 5090 上验证，未考察多卡分布式场景。

---

## 516. From Reweighting to Rewriting: Unlocking the Intervention Effects of Influential Samples in Training Data Attribution

**arXiv ID:** 2609.02771 | [PDF](https://arxiv.org/pdf/2609.02771v1)

**作者:** Yuzhang Luo `[一作]` (Peking University), Liangming Pan `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并验证了利用影响函数指导的响应重写来干预大型语言模型的行为，证明相比传统加权或删除方式更能有效调节模型的回避与安全拒绝行为。

**💡 创新点**

创新点在于将影响函数的例子选择与监督内容重写解耦，展示影响函数所选例子在重写时具有更大的行为调节潜力；并将此方法在多模态、不同规模模型上进行系统评估。

**🔧 技术方法**

核心技术包括影响函数（Influence Functions）与EK‑FAC逆曲率近似，用于计算训练样本对目标行为的影响；以及在监督层面进行响应重写的自定义训练策略。

**📊 数据集**

使用公开的多规模LLM训练集（OLMo2‑1B/7B、Qwen3.5‑2B、Gemma3‑4B）以及针对知识问答与安全拒绝的专门构造的评估查询集。

**📈 对比分析**

通过与随机样本、加权、删除等基线比较，发现重写在提高回避召回率与安全拒绝得分方面均比基线显著提升，且影响更稳定、可双向调节；但在安全拒绝时会出现过度拒绝的副作用。

**⚠️ 局限性**

局限性包括：仅针对可通过明确重写实现的行为（如回避、拒绝）有效；在安全域需进一步细化标签以避免过度拒绝；以及在更大规模或其他任务上的推广尚未验证。

---

## 517. Multi-Tool Image Editing Attribution in Facial Forgery

**arXiv ID:** 2609.02751 | [PDF](https://arxiv.org/pdf/2609.02751v1)

**作者:** Sheng Liu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Juan Cao `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出多工具图像编辑归因任务并设计了相应模型

**💡 创新点**

首次针对多步骤、多工具的图像编辑场景，构建了大规模数据集并结合双域补丁增强与错误驱动课程学习实现全局工具识别

**🔧 技术方法**

使用滑窗DCT、空间与频域交叉注意力、Xception骨干、SPL式错误课程学习

**📊 数据集**

构造500k+面部图像编辑数据集MultiEdit，覆盖六种常见编辑工具（Deepfake、OpenCV、CSD-MT、SHMT、FLUX、DiffSwap）

**📈 对比分析**

与ResNet、Xception、Swin、F^3-Net、FatFormer等基线以及Qwen3-VL进行对比，平均严格准确率92.9%、工具级准确率98.6%，在最多五步编辑情况下仍达87.4%严格准确率，显示显著性能提升

**⚠️ 局限性**

仅限面部图像，工具序列预测未实现，工具种类受限于六种编辑器，压缩或重采样下性能仍有下降

---

## 518. Bilevel Coordinated Reflection: A Game-Theoretic Approach to Multi-Agent LLM Systems

**arXiv ID:** 2609.02750 | [PDF](https://arxiv.org/pdf/2609.02750v1)

**作者:** Yihang Chen `[一作]` (UCL Centre for Artificial Intelligence), Jun Wang `[通讯]` (UCL Centre for Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并验证了一套基于双层协调博弈和验证门控反射（SRMA）的多代理LLM系统，理论上分析了解构质量、内存漂移与外部验证的作用，并通过实验证明该框架在资源分配、Overcooked和SWE‑bench任务中提升了性能。

**💡 创新点**

创新点在于：① 用双层潜在博弈模型统一刻画任务解构与工作者协同，② 对自由形式反射给出最优时间界限并阐释持续有害承诺导致的误差下限，③ 证明仅凭文本的评估门无法均匀改进，提出必须具备外部 grounding 的验证门；④ 引入 SRMA，提供确切收敛（几何或多项式）与置信门与重新锚定机制。

**🔧 技术方法**

技术方法包括：双层协调博弈与近似势游戏分析、一次性漂移（one‑sided drift）与持续有害承诺分析、信息不可辨别环境构造的不可行性证明、基于评估器的严格风险门控（SRMA）以及多尺度随机逼近与漂移定理的收敛证明。

**📊 数据集**

使用的数据集/环境包括：隐藏容量分配游戏（Resource Contest）、三人版 Overcooked（包含三种布局）以及 500 条 SWE‑bench 编程修复任务，实验采用基准 LLM 后端如 DeepSeek、Kimi K2.5 等。

**📈 对比分析**

与基准方法比较：在 Resource Contest 中 SRMA 达到 98.5%–99.5% 的 oracle 奖励，提升 60% 的 regret；在 Overcooked 中 SRMA 的得分分别比无内存、自由反射、自我门控高 14–30%，首次交付时间缩短约 25%；在 SWE‑bench 上 Kimi K2.5 背骨下，SRMA 72.2% 的通过率超过 58.4% 的自由反射和 70.8% 的公共 mini‑SWE‑v2 参考，DeepSeek 下亦显著提升。

**⚠️ 局限性**

局限性包括：假设解构耦合有限、动作集有限、评估器已校准且校准后误差非退化；漂移参数仅在观测轨迹上验证，未提供跨任务的普适性；仅保证评估器风险的单调性，真实任务收益不一定；重新锚定在段内收敛，缺乏全局切换损失界；在开放式任务中可能难以满足所有假设。

---

## 519. InceptionGS: Generative Bootstrapping for Large-Scale Gaussian Splatting under Unstructured View Sampling

**arXiv ID:** 2609.02747 | [PDF](https://arxiv.org/pdf/2609.02747v1)

**作者:** Tianheng Lu `[一作]` (Tsinghua University), Lu Fang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在大规模场景中解决视角采样不规则问题，结合重建与生成技术提出 InceptionGS，以 3D Gaussian splatting 为核心，实现高质量、三维一致的图像重建和补全。

**💡 创新点**

核心创新在于将场景自适应的生成先验（通过 ControlNet 对预训练文本到图像扩散模型进行几何条件微调）软融合进 3D Gaussian splatting；同时采用视角自适应重要性采样和光度融合策略，使生成与重建相互补强。

**🔧 技术方法**

使用了 3D Gaussian splatting、PGSR 进行几何优化、预训练的 Latent Diffusion Model（LDM）+ ControlNet 进行自适应生成、Langevin Monte Carlo（LMC）重要性采样、图像渲染与光度融合等技术。

**📊 数据集**

在 GigaNVS（7 个大规模真实场景）以及 MipNeRF360（小规模场景）数据集上进行实验验证。

**📈 对比分析**

与多种基准（MVSplat360、GEN3C、NVS‑Solver、Stable Virtual Camera 等）进行比较，平均 FID 降低 10%，LPIPS 降低 14%，整体视觉质量提升约 32%，并在复杂视角下保持稳定。

**⚠️ 局限性**

主要限制在于训练时延长（Stage‑I ~30 分钟、Stage‑II ~25 分钟）以及显存需求高（最高 30GB），未来工作计划探索更高效的优化方案以提升可扩展性。

---

## 520. Choosing a PEFT Variant for Per-Patient Dysarthric ASR: A Single-Speaker Case Study on Two ASR Bases

**arXiv ID:** 2609.02735 | [PDF](https://arxiv.org/pdf/2609.02735v1)

**作者:** Bernard Muller `[一作]` (Scott-Morgan Foundation), LaVonne Roberts `[通讯]` (Scott-Morgan Foundation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对单一严重中风后失语症说话人进行实验，比较七种 LoRA 族参数高效微调方法（LoRA、QLoRA、AdaLoRA、DoRA、LoHA、VeRA、VB-LoRA）在两种生产级 ASR 基座（Whisper‑large‑v3 与 Qwen3‑ASR‑1.7B）上的适配效果。

**💡 创新点**

首次在单声道失语症场景下系统性比较七种 LoRA 变体，并证明 LoRA 与 DoRA 在小样本环境下表现相当、优于其它变体；提出可直接使用的生产配方、记录录音时间对 CER 的影响曲线，并提供公开可复现的实验脚本与 Pareto 图。

**🔧 技术方法**

使用低秩自适应参数化（LoRA 族方法）对冻结的 Transformer 基座进行微调；采用多种实验设计（固定超参、不同随机种子、bootstrap 统计、Pareto 分析、录音时长网格）评估性能；在两基座上实现贪婪解码与 CER/WER 评估。

**📊 数据集**

单一匈牙利语男性受试者 S1 的 409 句录音（55 分钟），拆分为 262 句训练、40 句验证、107 句测试；训练集包含读句与叙述，使用匈牙利失语症混合数据做预训练；测试集与训练/验证集文本无重叠。

**📈 对比分析**

在相同训练脚本、固定 5 轮、固定学习率 1e‑4 的条件下对每个变体进行实验；LoRA 与 DoRA 在 Whisper‑v3 上 CER ≈13.8%，在 Qwen3‑ASR‑1.7B 上 CER ≈28.1%；QLoRA 4‑bit 逊色且无显著显存节省；LoRA 存储 63 MB；全参数微调 CER 约 11.4% 但存储 3 GB；5 分钟录音即可获得约 45% 的 CER 降低，10–30 分钟进一步提升。

**⚠️ 局限性**

实验仅基于单一说话人、单一语言、严重失语症，缺乏多患者多方言或多病理类型的泛化验证；未考虑多模态或 TTS 增强；基座选择对结果影响显著，当前结论仅适用于所用两基座。

---

## 521. CORAL: An LLM-Native Harness for Production Recommender Systems

**arXiv ID:** 2609.02730 | [PDF](https://arxiv.org/pdf/2609.02730v1)

**作者:** Muhammad Rafay Azhar `[一作]` (Meta AI), Xiangjun Fan `[通讯]` (Meta AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建并验证了一个基于大型语言模型的持续闭环优化系统（CORAL），在推荐系统中自动调节资源分配，以提升用户参与度或服务效率。

**💡 创新点**

创新点在于将LLM与工具调用、持久记忆和约束优化器结合，形成可持续的闭环；能在不同平台、不同决策类型（连续预算分配与离散服务级别分配）下通用；通过在线A/B实验迭代改进，无需人工频繁干预，并提供可解释的决策过程。

**🔧 技术方法**

采用的大型语言模型（LLM）+工具调用（统计分析、检索历史、归因评估、约束优化、配置部署）+持久记忆机制 + 在线A/B实验 + 预算约束的数值优化。

**📊 数据集**

使用了两个内部运营数据集：一个是视频推荐平台的用户会话、观看时长、各检索源的贡献等指标；另一个是服务容量分配平台的用户分段处理成本与参与度指标。

**📈 对比分析**

通过A/B实验对比手动配置和前一轮决策，实验显示：在视频推荐中，3轮闭环后观看时长提升约0.15%，会话数提升0.16%；在服务容量分配中，第二轮实现成本节约44%且保持参与度不变；与传统人工迭代相比，速度快、成本低且效果显著。

**⚠️ 局限性**

局限性包括：仍需人工监督和安全守护；目前仅验证在资源分配决策，未扩展到检索/排序逻辑等更复杂决策；A/B实验昂贵且缺乏统一评估标准；记忆窗口和循环周期需经验调优；对快速变化环境的响应速度和低信号用户改进空间有限。

---

## 522. The PIONEER Project: A PrIvacy companion for mOtivatioN and knowlEdge transfER

**arXiv ID:** 2609.02700 | [PDF](https://arxiv.org/pdf/2609.02700v1)

**作者:** Simon Althaus `[一作]` (Technical University of Darmstadt), Ephraim Zimmer `[通讯]` (Technical University of Darmstadt)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发并评估了一款名为PIONEER的移动隐私伴侣应用，结合知识转移与说服性设计，针对儿童、青少年、家长、老年人等不同用户群体提升其隐私意识与行为；通过可视化隐私政策、权限管理与习惯养成模块，支持用户在日常数字生活中做出更具主权的隐私决策。

**💡 创新点**

创新点包括：①将心理学需求模型（自主性、能力、关联性）与说服系统设计框架结合，以满足不同年龄段用户的独特需求；②提出并验证了针对隐私政策的可视化隐喻和结构化展示，提高信息可理解性；③构建模块化、可扩展的移动应用分析框架，并集成机器学习辅助的政策分析与聊天机器人；④系统化地从初始动机到习惯养成的长期行为改变路径设计。

**🔧 技术方法**

使用技术包括：Android移动应用开发、跨平台多语言支持、Persuasive System Design框架、心理学需求模型、机器学习文本分析（用于隐私政策解析与聊天机器人）、Android Data Safety接口分析、可视化框架（用于展示数据流）、用户体验迭代评估工具。

**📊 数据集**

使用的数据集包括：收集的隐私政策文本、Android应用的Data Safety信息、用户访谈记录、在线实验问卷、专家焦点小组讨论、可用性评估日志，以及针对儿童和青少年的访谈与视频内容。

**📈 对比分析**

评估方法通过对比实验（可视化与文本展示对理解度与偏好的影响）、可用性评估（系统使用满意度与效率）、以及用户问卷（信任、动机、需求满足度）进行；结果显示：可视化隐私政策显著提升用户理解和偏好；系统的说服与需求满足功能提升了用户对权限管理的信任与主动参与度。

**⚠️ 局限性**

局限性包括：长期行为改变的持续性尚未通过大规模追踪验证；样本量和样本多样性（尤其是老年人和儿童）相对有限；当前实现仅在Android平台，跨平台适用性待进一步验证；文化与地区差异对说服机制的效果影响尚未深入探究。

---

## 523. Discriminative World Models for Web Agents

**arXiv ID:** 2609.02885 | [PDF](https://arxiv.org/pdf/2609.02885v1)

**作者:** Kelvin Li `[一作]` (University of California, Berkeley), Roei Herzig `[通讯]` (University of California, Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并训练了一种“预测状态匹配”目标，用以让 Web 世界模型生成能够区分不同动作结果的下一状态表示，并将其应用于测试时的动作选择

**💡 创新点**

核心创新在于把世界模型训练目标从固定格式的下一状态生成转变为判别式的预测状态匹配；同时构建了包含多重候选动作及对应结果的分支决策点数据集

**🔧 技术方法**

使用大规模语言模型（如 Qwen3-32B/8B、GPT‑4o 等）生成文本化状态表示，并用另一个判别器评估匹配质量；随后将预测状态输入 PRM/Ranker 进行动作排序

**📊 数据集**

基于 WebArena 的 Go‑Browse 轨迹构造的分支数据集、WebPRM Collection/Bench 以及 WebArena‑Lite 任务集

**📈 对比分析**

在 held‑out 预测状态匹配基准、WebPRMBench 的动作排名（训练与冻结两种设置）以及 WebArena‑Lite 的端到端任务成功率上，与传统监督生成模型和无状态 PRM 相比均取得显著提升（匹配准确率、排名准确率提升数个百分点，任务成功率提升约 15‑20%）

**⚠️ 局限性**

实验局限于 WebArena 环境和 GPT‑4o 策略，匹配判别器依赖大语言模型，分支数据并未覆盖所有可能动作，因而在更广泛的 Web 生态和不同策略下的泛化性尚待验证

---

## 524. Overcoming the Randomness-Utility Trade-off in Answering Differentially Private Linear Queries

**arXiv ID:** 2609.02880 | [PDF](https://arxiv.org/pdf/2609.02880v1)

**作者:** Surendra Ghentiyala `[一作]` (Cornell University), Pasin Manurangsi `[通讯]` (Google Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了一种新的差分隐私机制，能够用极少的随机位（期望仅 O(log d)）回答 d 条线性查询，同时在纯 DP 场景下实现几乎最优误差 O(d/ε)，并给出了一个计算上可行的变体，误差为 O((d log d)/ε)；

**💡 创新点**

核心创新在于引入多尺度分离划分（multi‑scale secluded partition, MSSP）并利用指数机制直接在划分单元上采样，彻底摆脱传统噪声注入，从而实现既低随机位又低误差；

**🔧 技术方法**

主要技术包括：MSSP 的存在性证明与显式构造（特别是对 ℓ∞-范数的显式实现）、指数机制的采样、拒绝采样（rejection sampling）以及线性规划求解 Δ_P(v) 的高效算法；

**📊 数据集**

文中未使用具体实验数据集，而是以理论分析为主，证明误差与随机位的上界；

**📈 对比分析**

与先前工作（如 Canonne 等、Ghentiyala 等）的比较显示，传统方法在随机位上需 Ω(d) 并且误差为 O(d²/ε)，而本文方法在高隐私（ε≤1/d）下实现了 O(log d) 随机位和 O(d/ε) 误差；在可计算版本中误差提升至 O((d log d)/ε) 但仍保持 O(log d) 随机位；

**⚠️ 局限性**

主要限制包括： 1) 随机位最优性仅在 ε≤1/d 的高隐私区间已证明，其他 ε 区间缺乏下界； 2) 高效实现仅针对 ℓ∞-范数，其他 K‑范数（尤其是 ℓ_p）尚未得到高效方案； 3) 与理论最优误差相比，计算版本误差多了 O(log d) 的倍数。

---

## 525. The Implications of Linguistic Illegibility for LLM Security

**arXiv ID:** 2609.02852 | [PDF](https://arxiv.org/pdf/2609.02852v1)

**作者:** James Mickens `[一作]` `[通讯]` (Harvard University), James Mickens (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并阐述了“语言不可读性”概念，指出 LLM 的内部计算与其自然语言输出之间可能存在不可解释的差距，并基于此设计了一套以污点追踪、硬件虚拟化、网络代理、行为异常检测和配置鉴定为核心的多层防护沙箱架构；

**💡 创新点**

创新点在于将传统系统安全技术（如信息流控制与硬件可信执行）与 LLM 语言不可读性问题结合，提出了“语言不可读性”这一新术语，并提供了针对前沿模型的“语言不可读性安全防护”框架；

**🔧 技术方法**

主要技术包括：污点追踪（可变标签与因果标记）、硬件/微内核级虚拟化（如 SEL4/TEE）、最小化出站代理、统计行为异常检测以及第三方配置鉴定与多方签名；

**📊 数据集**

本文为理论性工作，并未使用具体数据集；

**📈 对比分析**

因缺乏实验评估，本文未给出数值性能指标；该工作主要通过安全案例分析和对比说明所提沙箱架构的威胁缓解能力；

**⚠️ 局限性**

局限性包括：未实现原型验证，污点追踪的实现成本和性能开销未知；对隐式信息流的检测仍依赖统计方法；以及对硬件侧信道等低级攻击的防御尚未充分评估。

---

## 526. Almost Linear 3-Spanners of Temporal Cliques

**arXiv ID:** 2609.02851 | [PDF](https://arxiv.org/pdf/2609.02851v1)

**作者:** Julia Baligacs `[一作]` (University of Oxford), Anna Zych-Pawlewicz `[通讯]` (University of Warsaw)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构造了稀疏的3跳临时拓扑树（temporal 3‑spanner）用于临时完全图及其二分图，并给出了实现该拓扑树的递归算法。

**💡 创新点**

提出了一种新的、极其简洁的递归分解方法，可在几乎线性大小内构造3跳临时拓扑树，并消除了对生命周期的指数依赖，显著改进了先前的上界。

**🔧 技术方法**

利用Zorro算法在每一步选取两棵星形子图覆盖大规模顶点对，再递归处理剩余对；并通过数学归纳与对数项分析得到大小与时间复杂度的上界。

**📊 数据集**

论文仅在理论层面进行证明，并未使用实验数据集。

**📈 对比分析**

与之前最优的O(n^{3/2})（无生命周期）和O(2^L n log n)（有生命周期）上界相比，本文得到的上界分别为n^{1+o(1)}和O(nL)，并且算法时间分别为O(n^2)和O(L n^3)，性能大幅提升。

**⚠️ 局限性**

仍未能证明线性大小的3跳拓扑树是否存在；此外，算法在大规模图上的实际运行性能尚未通过实验验证。

---

## 527. UE5M3 FP4 Block Scaling for Stable Language Model Pretraining

**arXiv ID:** 2609.02846 | [PDF](https://arxiv.org/pdf/2609.02846v1)

**作者:** Robert Hu `[一作]`, Paul Balanca `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了一种基于unsigned E5M3块尺度的FP4预训练方案，完成了8B Nemotron模型在188.7B标记上的完整训练。

**💡 创新点**

创新点在于利用更宽动态范围的E5M3块尺度与周期性张量尺度、去除RHT和BF16最终块的使用，显著简化了FP4训练流程。

**🔧 技术方法**

采用FP4 E2M1载荷、E5M3块尺度、周期性样本‑保持张量尺度、2D权重尺度、上游梯度的随机舍入、probe‑matched GEMM模拟器等技术。

**📊 数据集**

使用82% DataComp‑LM + 18% OLMo混合数据，共计188.7B标记进行训练，验证集为768个8192标记序列。

**📈 对比分析**

与NVIDIA Transformer Engine（E4M3块尺度、RHT、BF16最终块）和BF16基准进行对比；在最终窗口损失、NLL和下游多项选择准确率上均优于对照组，模型体量推理吞吐提升21.2%。

**⚠️ 局限性**

局限性包括训练时间缩短（仅188.7B标记）、数据混合不同、仅单一种子、对特定硬件（GB200）依赖、未评估不同尺度目标T的稳定性以及对不同架构的可迁移性。

---

## 528. AI Contextual Measurement for Recovering Individual and Group-Level Effects: Validation Against Survey Measures and an Occupational Application

**arXiv ID:** 2609.02821 | [PDF](https://arxiv.org/pdf/2609.02821v1)

**作者:** Wenxin Jiang `[一作]` (Northwestern University), Yuxiao Wu `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并验证了AICOME框架，用AI生成个体层面测量并分解为组均值与个体偏差，支持情境分析。

**💡 创新点**

首次将AI产生的个体指标用于情境模型的组间与组内效应识别，并提出多层次验证方法。

**🔧 技术方法**

利用大型语言模型（GPT‑5.6 Sol）结合自定义prompt和特征向量生成个体测量，随后聚合计算组均值。

**📊 数据集**

使用2022年中国家庭追踪调查（CFPS）数据，包含职业、工作时间、电脑使用、外语使用、管理责任等变量。

**📈 对比分析**

与传统问卷测量在响应、模型、情境和边界条件四个层面进行对比；在信息丰富时AI指标能复现大部分情境效应，信息稀缺时性能显著下降。

**⚠️ 局限性**

局限在于依赖丰富的现有特征，无法补全多个相关维度缺失，且AI测量需人工诊断以防偏差；不能替代直接问卷测量。

---

## 529. GRADSOLVE: fast exact gradients for ODE ensembles on GPUs

**arXiv ID:** 2609.02876 | [PDF](https://arxiv.org/pdf/2609.02876v1)

**作者:** Alessio Spurio Mancini `[一作]` `[通讯]` (Royal Holloway, University of London), Alessio Spurio Mancini (Royal Holloway, University of London)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一款名为gradsolve的GPU加速ODE集合求解与反向模式微分库，利用记录-回放技术在保持最快前向求解速度的同时实现高效梯度计算。

**💡 创新点**

核心创新在于先记录自适应求解器接受的步长序列，再对固定步长的回放进行反向微分，得到与自适应步长相同的离散伴随梯度，既保留了自适应控制的精度，又避免了传统自适应循环在反向时的高开销。

**🔧 技术方法**

使用了JAX框架、XLA编译器、Warp GPU核（CUDA）进行融合求解与回放；实现了显式与Rosenbrock半隐式积分器，并支持单精度/双精度、稀疏/密集输出。

**📊 数据集**

在六个典型ODE模型上进行基准测试：Lorenz系统、Van der Pol振荡器、Robertson化学动力学、HIRES刚性系统、银河暗物质取向势能模型以及将神经网络嵌入Robertson系统的变体。

**📈 对比分析**

与DiffEqGPU.jl、Diffrax的检查点伴随、torchode和torchdiffeq进行比较；在前向求解上比DiffEqGPU.jl快，反向梯度上比Diffrax快10-30倍（小集合）到5-10倍（大集合），在PyTorch框架下提升1-2个数量级；单精度版本进一步加速。

**⚠️ 局限性**

局限性包括：适用于低维（≈64维）显式ODE；不支持随机微分、代数约束或事件处理；需要一次完整记录（mesh drift需手动重记录）；仅在NVIDIA GPU上经过验证；对非JAX可序列化的右手边函数支持有限。

---

## 530. PlantC2USeg: Cross-Scale Consistent Pre-Training for Few-Shot Unified Plant Point Cloud Segmentation

**arXiv ID:** 2609.02860 | [PDF](https://arxiv.org/pdf/2609.02860v1)

**作者:** Yu Tian `[一作]` (McGill University), Shangpeng Sun `[通讯]` (McGill University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种基于跨尺度一致性预训练的PlantC2USeg框架，联合进行植物点云的组织级语义分割与叶片实例分割，并在极少标签下实现快速适配。

**💡 创新点**

创新点在于①跨尺度一致性学习，显式对齐不同尺度的局部与全局特征；②信息受限解码，仅利用可见区域信息避免重建捷径；③统一的few-shot微调方案，在单一网络内同时完成语义和实例目标，减少后处理阈值搜索。

**🔧 技术方法**

采用多尺度Point-M2AE编码器、Transformer自注意力、交叉尺度一致性损失、信息受限交叉注意力解码、联合语义分类、特征与坐标实例约束以及进阶分割策略；训练使用AdamW、余弦学习率衰减等。

**📊 数据集**

使用多种植物点云数据集：Soybean3D（含145标注样本）、HR3D（多品种、低点数）、SYAU-Maize（22标注样本）以及ShapeNet Part用于通用形状分割验证。

**📈 对比分析**

与PlantNet、PSegNet、Eff‑3DPSeg、Soybean‑PCMAE、Deformation3D、SGPN、ASIS等方法对比，PlantC2USeg在全监督下达到91.91% IoU，20/10-shot分别为89.78%/83.19%；在HR3D 10-shot获得78.41% IoU；在SYAU-Maize 92.75% IoU；ShapeNet Part类别平均mIoU 85.0%，均高于或竞争现有方法。

**⚠️ 局限性**

仅在单一植物静态点云上验证，难以处理多植物重叠、背景杂草、不同采集方式导致的点云稀疏与不完整；时间序列匹配和跨场景泛化仍待探索；在极小叶片或极稀疏点云下实例分割仍易出错。

---

## 531. MuyBridge: Mobile Human Center-of-Mass Estimation from Monocular Video via Sparse Fusion

**arXiv ID:** 2609.02854 | [PDF](https://arxiv.org/pdf/2609.02854v1)

**作者:** Aidan Bradshaw `[一作]` (ETH Zurich), Christoph Leitner `[通讯]` (ETH Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一个单摄像头手机端系统MuyBridge，能够从手机视频估计运动员的三维身体质心轨迹。

**💡 创新点**

通过将轻量级2D姿态网络、单步稀疏单目深度网络与解剖学及物理先验的解析式度量融合，实现在不依赖3D标注的情况下得到绝对尺度的身体质心，并在iPhone 15上实现实时63 FPS的质心输出。

**🔧 技术方法**

使用RTMPose式轻量级2D姿态网络、Marigold基础的单步深度网络（含稀疏深度采样与一致性蒸馏）、de Leva解剖学参数化、场景一次校准的物理范围约束、模型压缩、量化以及Core ML异步推理等技术。

**📊 数据集**

训练姿态网络使用COCO‑WholeBody与HICO‑DET，训练深度网络使用Hypersim与Virtual KITTI 2的1.2 M合成RGB‑D对，评估使用AthletePose3D（跑步、田径、花样滑冰）数据集。

**📈 对比分析**

与MeTRAbs‑S、HMR2.0、CameraHMR、NLF、MotionAGFormer等单目人体重建/3D姿态方法对比，MuyBridge在跑步的绝对CoM误差仅次于最佳方法，成为唯一在移动端实现实时部署的方案；总体3D CoM平均误差为跑步≈187 mm、田径≈185 mm、滑冰≈707 mm，垂直误差保持在33–41 mm，范围误差2.3–6.6%。

**⚠️ 局限性**

主要受相机–运动员距离估计误差限制，尤其在长距离移动和空中运动时误差显著；系统依赖一次场景校准，且使用性别与身高比例的群体统计解剖参数，未能自适应个体差异。

---

## 532. Toward Robust LiDAR Semantic Segmentation for Real-World Deployment: Evaluation under Coarse Labels, Adverse Conditions, and Domain Shifts

**arXiv ID:** 2609.02830 | [PDF](https://arxiv.org/pdf/2609.02830v1)

**作者:** Samir Abou Haidar `[一作]` (Mines Paris, PSL University), Jean-Emmanuel Deschaud `[通讯]` (Mines Paris, PSL University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一套统一的LiDAR语义分割评估协议，涵盖粗标签安全评估、对八种传感器失真鲁棒性、跨域泛化以及嵌入式推理速度的多维度测评。

**💡 创新点**

创新点在于将安全关键粗标签、真实世界传感器失真鲁棒性和无适配跨域泛化三大评估维度整合进一体，并将硬件推理速度纳入评价体系，形成完整的部署准备评估框架。

**🔧 技术方法**

使用了多种主流LiDAR分割网络（点基、投影基、稀疏卷积、融合基）以及Robo3D、SemanticKITTI、nuScenes、ParisLuco3D等数据集，并引入八种合成失真实现鲁棒性评测。

**📊 数据集**

主要数据集包括SemanticKITTI、nuScenes（原始与腐败版）、ParisLuco3D（跨域测试）以及Robo3D中的腐败数据。

**📈 对比分析**

通过对比Fine-grained mIoU、Coarse mIoU、Mean Corruption Error (mCE) 和Jetson AGX Orin的FPS，实验表明：高分标注不一定对应安全性能，所有模型在失真和跨域测试中均有显著退化，且缺乏同时兼顾精度、鲁棒性、泛化与实时性的优秀模型。

**⚠️ 局限性**

局限性包括仅考察八种合成失真和两种跨域场景，未引入自适应或多模态策略，且评测聚焦于单一LiDAR传感器，无法全面覆盖不同硬件与环境的真实部署需求。

---

## 533. SolarWM: Open Data and Scalable Training for Long-Horizon Video World Models

**arXiv ID:** 2609.02886 | [PDF](https://arxiv.org/pdf/2609.02886v1)

**作者:** Junchao Huang `[一作]` (Chinese University of Hong Kong, Shenzhen), Li Jiang `[通讯]` (Chinese University of Hong Kong, Shenzhen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个可扩展、可复现的交互式视频世界模型基础，结合多源数据引擎、骨干本地适配框架以及统一的三阶段训练流程，推出了 5B–33B 四款可按摄像机控制的模型，并发布完整数据与代码。

**💡 创新点**

① 将异构视频数据统一到帧对齐的“合同”并解耦源处理与混合构造；② 采用骨干本地适配器保持各自前置能力；③ 统一三阶段训练（双向适配、教师强制AnyFlow初始化、分布匹配蒸馏），实现仅用 5 秒序列即可训练长时程交互模型。

**🔧 技术方法**

多源数据引擎（canonical clip contract, fused‑PRoPE 摄像机注入）、骨干适配器（Wan2.2、LTX‑2.5、MiniMax‑H3 预训练模型）、三阶段训练（双向训练、TF‑AnyFlow、DMD 蒸馏）、分布匹配损失和光流目标。

**📊 数据集**

10 个源数据集共 1.43M 片段（ABOT‑World、DL3DV、MiraData、RealCam、SpatialVID、Sekai‑Game、Sekai‑Walking、OmniWorld、MultiCamVideo 等），覆盖真实、合成、游戏环境。

**📈 对比分析**

在 OOD 与在域评估上，使用 10 秒、1 分钟、1 小时无截断的自回归生成，比较四款模型的视觉质量、摄像机响应与一致性；结果表明，所有模型在 5 秒训练后即可实现分钟到小时级连贯推理，视觉质量接近或超过同类基线，并能保持摄像机控制。

**⚠️ 局限性**

训练仅 5 秒序列导致对更长时间动态演化建模不足；在动态对象处理上可能出现误差；对摄像机控制的依赖，未覆盖动作/语义控制；训练成本高，需大规模 GPU。

---

## 534. When Does Authorization End? Effect Closure at Provider Boundaries

**arXiv ID:** 2609.02866 | [PDF](https://arxiv.org/pdf/2609.02866v1)

**作者:** Igor Santos-Grueiro `[一作]` `[通讯]` (International University of La Rioja), Igor Santos-Grueiro (International University of La Rioja)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在授权工作完成后，如何让服务边界（provider boundary）真实地报告已授权的工作不再能产生被拒绝的效果（policy‑relative effect closure）。

**💡 创新点**

创新点包括：① 提出了效果闭包（effect closure）语义及其与未来使用闭包、实例效应闭包和静止性（quiescence）的严格分离；② 通过前沿（frontier）适当性和边界可实现性（boundary realizability）建立正式验证框架；③ 设计并验证了三种通用修复（Bind、Re‑enter、Fence）来解决跨系统授权‑执行偏差；④ 将完整的机器校验流程（Lean 4）与实证证据结合，形成可重复的、可验证的闭包判定管道。

**🔧 技术方法**

采用的技术包括：有限控制系统模型、监督控制理论、Lean 4 形式化证明与证书检查、基于证据的有限合约（finite contract）构造、可追溯的权限链与前沿推断、以及自动化的修复搜索与成本评估。

**📊 数据集**

使用的数据集来自四大主流系统的实际部署：GitHub（MCP‑to‑REST接口）、Kubernetes（admission 与 RBAC）、NATS（消息传递与回调）、Kafka（生产者写入与 ACL revocation）。每个系统提供文档、源代码哈希、以及被捕获的运行时日志和重放轨迹。

**📈 对比分析**

评估方法包括：① 对六条不同的授权‑效果路径执行基线与修复后的对比，测量禁止效果发生率、必需工作完成率以及是否阻塞正当请求；② 在 Kafka 上对 Effect Fence 与 Applied‑State Fence 以及基线进行性能基准，记录吞吐量、p99 延迟以及对非受保护工作影响。结果显示：所有修复均能消除被拒绝效果且不阻塞必需工作；Effect Fence 在 Kafka 上实现零错误闭包，吞吐量下降约 10%，p99 延迟最高约 40 ms，且对其他工作几乎无影响。

**⚠️ 局限性**

局限性：① 仅在已给出完整证据的有限合约内证明；若存在未被记录的路径或隐藏行为，验证结果无法覆盖；② 不保证全局 provider 级别的完整性，仅关注声明的作用域；③ 对网络分区、动态成员变更等复杂场景缺乏覆盖；④ 性能评估基于有限规模部署，实际生产环境中的开销可能更大。

---

## 535. Post-Training Language Models for Gold-Medal Performance in Coding Competitions

**arXiv ID:** 2609.02849 | [PDF](https://arxiv.org/pdf/2609.02849v1)

**作者:** Aleksander Ficek `[一作]`, Boris Ginsburg `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了端到端竞赛编程流水线，包括问题库、合成推理数据、SFT、RL和GenCorrect，最终实现AI在IOI 2026中超过最高人类得分。

**💡 创新点**

创新点在于将多阶段训练与迭代测试时计算（GenCorrect）相结合，利用大规模合成推理轨迹和执行反馈提升解题质量，并在单次提交内完成全局优化。

**🔧 技术方法**

采用了大规模问题收集、DeepSeek生成的推理轨迹、Nemotron-3系列模型的SFT和GRPO强化学习、GenCorrect迭代检验、NVFP4量化与推理加速等技术。

**📊 数据集**

使用了约22 000个竞赛题目、120 万条DeepSeek推理轨迹、47 万条GLM‑5.2/DeepSeek‑V4‑Flash训练样本，并在IOI 2025/2026、ICPC 2025、LiveCodeBench Pro等评测集上验证。

**📈 对比分析**

与基准模型（gpt‑oss‑120b、Qwen3.6‑35B、Nemotron‑Cascade‑2、DeepSeek、GLM‑5.2）对比，Nano‑CC在IOI 2025 Score@1从21.7%提升至48.5%，Ultra‑CC 50.7%，并在IOI 2026单轮实时运行中获得535.4/600，超过金牌阈值和最高人类得分。

**⚠️ 局限性**

局限性在于对大量算力（训练和推理）依赖，RL在Ultra规模不可扩展，训练语料无法公开，且效果可能不易推广至其他任务。

---

## 536. RoGe: Novel View Synthesis via End-to-End Implicit Reconstruction and Generation

**arXiv ID:** 2609.02847 | [PDF](https://arxiv.org/pdf/2609.02847v1)

**作者:** Xiaolei Lang `[一作]` (Xiaomi EV), Naiyan Wang `[通讯]` (Xiaomi EV)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了RoGe，一个端到端统一的重建-生成框架，用于从稀疏视图生成随摄像机轨迹变化的连贯视频。

**💡 创新点**

通过在重建网络中提取射线查询得到的隐式几何特征直接作为条件注入到视频扩散模型中，并实现两者的联合训练，避免了传统方法中显式3D桥接导致的信息丢失。

**🔧 技术方法**

采用基于VGGT的摄像机条件重建网络、Plücker射线编码、隐式场查询、视频扩散Transformer (DiT)、Hybrid VAE 生成、LoRA 微调、像素解混等技术。

**📊 数据集**

在ARKitScenes、BlendedMVS、DL3DV、MVS-Synth、WildRGB-D等多模态稀疏数据集上训练，并在DL3DV测试集上评估。

**📈 对比分析**

与3DGS、AnySplat、YoNoSplat、LagerNVS、SEVA、FrameCrafter、Difix3D+、GEN3C、NeoVerse等基线在图像（PSNR、SSIM、LPIPS）和视频（TSED、MEt3R、FID、FVD、APE）指标上对比，RoGe在视觉质量、几何一致性和相机控制性上均超越竞争方法。

**⚠️ 局限性**

仍依赖大量训练数据，联合训练对GPU资源要求高，且在极端视角变化或极稀疏输入时可能出现细节失真或对齐误差。

---

## 537. Benchmarking RAW and RGB Restoration in Image Signal Processors

**arXiv ID:** 2609.02831 | [PDF](https://arxiv.org/pdf/2609.02831v1)

**作者:** Zihao Lu `[一作]` (University of Würzburg), Marcos V. Conde `[通讯]` (University of Würzburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文比较了在固定图像信号处理器（ISP）前后进行盲图像恢复的两种策略：RAW域预处理与RGB域后处理，构建了系统化的基准框架；

**💡 创新点**

创新点在于将恢复位置与训练分布解耦，系统评估了RAW域、RGB域以及针对目标ISP训练的RGB域恢复的表现，揭示训练分布对最终质量的决定性影响；

**🔧 技术方法**

使用了深度恢复网络（如UNet、NAFNet、MOFA、MFDNet、PromptIR、AirNet、MiOIR）以及两种学习式ISP模型作为固定后端，并在合成噪声/模糊数据上训练和评估；

**📊 数据集**

主要数据集为RawIR，包含Vivo X90 Pro、iPhone XS、Samsung S9/S21、Google Pixel 7–9四款手机的RAW图像，合成噪声、模糊与联合噪声模糊三种降质；

**📈 对比分析**

通过在四个设备组和两种ISP上对比PSNR/SSIM指标，发现基准训练的RAW恢复优于通用RGB预训练模型，而针对目标ISP训练的RGB恢复能进一步提升效果，显示训练分布匹配至关重要；

**⚠️ 局限性**

局限性包括：合成降质难以完全模拟真实传感器噪声与模糊；测试集仅47幅图像，泛化性有限；实验未对RAW和RGB恢复的损失域进行完全匹配，难以归因于域本身；

---

## 538. A Common Measure of Communication for Speech Brain-Computer Interfaces

**arXiv ID:** 2609.02887 | [PDF](https://arxiv.org/pdf/2609.02887v1)

**作者:** Dulhan Jayalath `[一作]` (University of Oxford), Oiwi Parker Jones `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `2704f255-0c84-4173-b83c-0e9a3dbea232` `e15e3743-5ee0-4d5f-813d-d146868082fc` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种开词汇互信息（OVMI）度量，用来在不同实验设置和词汇规模下评估语音脑机接口的可通信信息量。

**💡 创新点**

创新点在于：①将词汇覆盖率与解码准确度结合为单一信息量指标；②允许使用任意用户期望的词汇分布作为参考，解决传统准确率与词汇条件导致的不可比性；③利用OVMI指导词汇选择，可显著提升解码准确率。

**🔧 技术方法**

技术手段：信息理论（互信息公式）、对概率分布的估计、基于宏观准确率的OVMI近似、对话系统的语言模型后处理。

**📊 数据集**

使用的数据集包括：侵入式系统（Moses、Willett、Card 等）、非侵入式系统（LibriBrain100、MEG‑MASC、Tang fMRI）以及参考词频分布 SUBTLEX‑UK、Switchboard、AAC、Sherlock 等。

**📈 对比分析**

与传统准确率、WER 和 Wolpaw ITR 的比较显示：OVMI 更保守、能揭示词汇覆盖不足导致的信息损失；在词汇规模增大、覆盖率提高的系统中，OVMI 证明其信息传输显著提升；在词汇选择实验中，使用OVMI作为目标可比传统频率/准确率方法提升 15–16% 的准确率。

**⚠️ 局限性**

局限性包括：①大多数评估仅使用宏观准确率估计，缺乏完整混淆矩阵；②对未在词汇表中的词直接惩罚，未考虑同义或短语替代；③仅评估词汇层面的信息，未考虑上下文语言模型；④依赖于事先指定的参考分布，若选择不当可能影响比较结果。

---

## 539. Graph Machine: Towards Better Pretraining via Edges

**arXiv ID:** 2609.02881 | [PDF](https://arxiv.org/pdf/2609.02881v1)

**作者:** Lintai Hou `[一作]` `[通讯]`, Lintai Hou

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种Graph Machine（GM）架构，通过维护O(n)大小的状态并利用可微分的指针式路由（Referral）实现稀疏动态访问，实现了在语言模型预训练中将大部分Transformer层替换为稀疏层。

**💡 创新点**

创新点在于：①将图结构的软指针（edge indices & weights）与节点特征结合，实现O(n)状态下每步仅访问O(1)条路径；②引入Referral机制以递归构造ℓ-hop邻域，软化并稀疏化边权；③混合存储边、混合操作与稀疏注意力（SEA）共同实现高效稀疏推理。

**🔧 技术方法**

使用的技术包括：稀疏坐标表示（CSR-like）存储节点与边；Referral（两跳或多跳）生成新的邻居；稀疏注意力（SEA）将查询与软边权融合；温度缩放与top‑s稀疏化；以及在PyTorch+Triton环境下实现高效稀疏算子。

**📊 数据集**

在FineWeb‑Edu数据集上进行预训练，使用约15.7 B个token，训练时每个序列长度为4,096。Baseline为Qwen3‑0.6B。

**📈 对比分析**

与完整的Qwen3对比，GM模型在仅取2或4个KV位置（对应0.098%–0.195%密集访问）时，测试损失仅略高或略低（最高误差≈0.014），训练总计算量减少5%–15%，而Referral+Attention计算下降10%–30%，但在某些配置下训练速度仍略慢，需要更高效的自定义核。

**⚠️ 局限性**

局限性包括：实验规模较小、仅评估了单一的测试损失指标、实现依赖于原型PyTorch/Triton，硬件（H100 vs RTX4090）对性能影响大；缺乏对不同规模、不同任务的进一步验证与对比；并未深入探讨GM在下游任务中的泛化与可解释性。

---

## 540. Approximately Efficient Multidimensional Bilateral Trade

**arXiv ID:** 2609.02872 | [PDF](https://arxiv.org/pdf/2609.02872v1)

**作者:** Aviad Rubinstein `[一作]`, Zixin Zhou `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在多维双边交易场景下（买方具有 XOS 价值，卖方为加法成本），提出了能够在保证 BIC、IIR 与弱预算平衡的前提下实现对第一最佳 GFT 的常数比例近似（至少 1/44）的机制。

**💡 创新点**

创新点在于：①将多维买方的 XOS 价值通过核心-尾巴分解转化为可处理的单维复制实例；②设计了基于 ASPE（匿名顺序定价与入场费）的核心近似机制；③提出了限制性买方采购菜单加 VCG 方案以克服多买方时正外部性的预算平衡难题；④在多买方情形下统一实现了核心与尾巴的常数因子近似。

**🔧 技术方法**

采用的技术包括：核心-尾巴分解、Chernoff‑type 近似与 Paley–Zygmund，ASPE（匿名顺序售价+入场费）框架，单维复制实例与 Prophet Inequality 的折算，半容量限制的 VCG 采购菜单，以及虚拟价值最大化与铁化技术。

**📊 数据集**

本研究为理论性论文，无实验数据集；所有结论均基于概率模型与理论分析。

**📈 对比分析**

与此前仅能对单维交易实现常数因子近似的工作相比，该方案在多维 XOS 买方情境下实现了 1/44 的常数比例近似；在多买方情况下，通过 ASPE 与采购菜单实现了 1/88 的近似，表明在更复杂的市场结构中仍能保持可接受的效率。

**⚠️ 局限性**

限制包括：假设买卖双方的项目值与成本相互独立且无相关性；机制设计复杂，虽然提供了简单候选方案，但实现细节仍需对 XOS 价值结构进行精确分解；并且对实际计算复杂度未给出完整可行实现方案，主要关注理论上可行性与近似保證。

---

## 541. Towards Trustworthy Autonomous Robots: An Explainable AI-Based Decision Framework

**arXiv ID:** 2609.02861 | [PDF](https://arxiv.org/pdf/2609.02861v1)

**作者:** Cagri Temel `[一作]` `[通讯]` (Hezarfen LLC), Cagri Temel (Hezarfen LLC)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fede83ac-7505-405f-ab37-e7284695c47f` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出TRACE框架，将自主机器人决策过程拆分为语义感知、信念推理、动作合成和执行验证四层，并记录每层的证据链、因果图、反事实树和审计轨迹，实现决策级可追溯与可重构。

**💡 创新点**

创新点包括：①基于架构的审计机制，确保每一决策都能追溯至传感器证据；②正式定义证据可追溯性、时间连续性和决策可重构性三项客观指标；③引入反事实决策树记录动作选择的替代方案与理由；④对存储、压缩与多机队部署的层级方案进行分析。

**🔧 技术方法**

技术手段：深度学习感知模块（CNN、Transformer），贝叶斯信念图，因果图构建，规划与约束搜索，反事实树生成，实时审计记录写入，压缩编码与增量差分存储。

**📊 数据集**

使用自行搭建的Python仿真环境，模拟50m×30m仓库，包含LiDAR、摄像头、超声波传感器；共5种情境，每种100次决策循环，总计500次决策；没有公开数据集，而是采用仿真产生的传感器噪声与随机失效。

**📈 对比分析**

与LIME、SHAP、Attention、ModelPlex等后置解释方法进行对比；TRACE在证据可追溯率98.6%、时间连续率99.0%、决策可重构率98.1%时，后置方法只能给出特征归因，无法提供决策级审计，且在所有指标上显著优于基线；单个决策周期开销约0.12 ms，约占10 Hz控制预算的1%。

**⚠️ 局限性**

局限性：仅在仿真环境验证，缺乏真实硬件与同步误差的评估；存储与压缩估计需在大规模机队上进一步验证；指标只衡量日志可重构性，未验证调查人员实际使用效果；对极端安全边缘情况的可追溯率仍有约1–2%缺口；仅针对仓库导航，其他领域（如医疗、农业）需进一步适配。

---

## 542. User Feedback Provides a Unique Signal that LLMs Can not Detect

**arXiv ID:** 2609.02859 | [PDF](https://arxiv.org/pdf/2609.02859v1)

**作者:** Shachar Don-Yehiya `[一作]` (Hebrew University of Jerusalem), Omri Abend `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过构造合成数据和提取真实对话中的自然反馈，系统评估自然反馈在提升大型语言模型回答质量中的作用，证明自然反馈是可操作的改进信号。

**💡 创新点**

①证明自然反馈本质上有效；②揭示现有LLM-评判器评估方法存在系统性偏差，导致评判器低估反馈带来的提升；③发现改进器与评判器在识别有效改动时存在关联性失配；④对反馈强度和改进类型进行细粒度分析。

**🔧 技术方法**

使用大型语言模型（Gemini‑3‑Flash‑Preview、Qwen3‑8B、Gemini‑3.1‑Pro 等）进行回复生成、扰乱、改进和评估；采用 LLM‑as‑Judge 的 pairwise 对比评估；构造四类扰乱（因果反转、关键信息缺失、实体/主题交换、逻辑运算反转）；利用多级反馈（指明修复、仅指出问题、仅说明错误）。

**📊 数据集**

合成数据：Arena‑Hard‑v2.0（包含 500 个硬题和 250 个创意写作）；真实数据：ShareLM 集合（仅英文）。

**📈 对比分析**

对比 "有反馈" 与 "无反馈" 两种改进方案，主要指标为：
- 纠错率（issue resolution rate）：在合成数据上提升 9–35%，在真实数据上提升 16–27%；
- pairwise judge 的偏好：即使有反馈的答案更准确，评判器往往在 34–54%（自然数据）和 45–58%（合成数据）的情况下更偏好无反馈答案，显示评判偏差；
- 进一步分析表明，反馈会产生更多内容层面的改进，而评判器更倾向于认可风格层面的修改。

**⚠️ 局限性**

①合成数据中创意写作的扰乱效果较差，导致主要结果聚焦于数学/代码域；②手工标注的评估难度高，尤其是 pairwise 对比，未能完全覆盖所有样本；③依赖 LLM 评判器带来评估偏差，缺乏人类标注的基准；④实验仅在推理阶段完成，未验证在训练阶段的表现。

---

## 543. Efficient All-in-One Weather Restoration using Spectral Harmonization

**arXiv ID:** 2609.02839 | [PDF](https://arxiv.org/pdf/2609.02839v1)

**作者:** Paula Garrido-Mellado `[一作]` (Cidaut AI, Fundación Cidaut), Marcos V. Conde `[通讯]` (Cidaut AI, Fundación Cidaut)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量化的一体化天气图像恢复方法 FReSH-IR，利用频域分解和频谱和谐化在每个尺度对高低频特征进行分离与专门化处理，实现高质量恢复并显著降低模型参数与运算量。

**💡 创新点**

创新点在于：① 在每个编码/解码层显式分离高低频，并使用专门化模块分别处理；② 采用频谱和谐化跳连接，替代传统加法式跳连接，能够在保留空间细节的同时更好融合频域信息；③ 结合空间与频域的双注意模块，提升对不同天气退化的建模能力；④ 在整体架构上实现 80% 参数、90% MACs 下降，同时保持甚至超越主流重模型的恢复效果。

**🔧 技术方法**

核心技术包括：U‑Net 结构、FFT+高斯掩模频域分解、双注意模块（空间 + 频域）、频谱和谐化块（SHB）、频域变压器块（FTB）、残差块与轻量化注意力、三种联合损失（像素、增强、傅里叶边缘）。

**📊 数据集**

训练与评估使用 RainDrop、Outdoor‑Rain、Snow100K 三大数据集；测试集包括 RainDrop、Outdoor‑Rain Test1、Snow100K‑L，真实场景测试使用 RealSnow；下游任务实验则在 Rainy Cityscapes 数据集上进行检测与分割评估。

**📈 对比分析**

与 SSGFormer、TransWeather、MWFormer 等 state‑of‑the‑art 方法进行定量比较，FReSH-IR 在 PSNR、SSIM 等指标上保持相近或略优，参数量降低 80%~90%，MACs 降低 80%~90%，推理速度提升 4~5×，且能在 30 ms 内完成 256×256 图像处理，显著提升实际部署可行性。

**⚠️ 局限性**

局限性主要体现在：① 对极端真实天气场景的恢复效果仍有限，主要受限于真实样本缺乏；② 在极端下游任务（如自动驾驶高精度检测）中的提升相对有限；③ 需要进一步扩充更丰富的真实天气数据集以提升泛化性能。

---

## 544. Understanding Automatic Mixing: A Subtask-Oriented Analysis of Two-Stage Mixing System

**arXiv ID:** 2609.02835 | [PDF](https://arxiv.org/pdf/2609.02835v1)

**作者:** Jinjie Shi `[一作]` (Queen Mary University of London), Joshua Reiss `[通讯]` (Queen Mary University of London)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

评估了两阶段自动混音的子任务性能，并对全混模型的迁移、错误补偿及两阶段分解的优势进行了对比实验

**💡 创新点**

提出了以分组与局部平衡为核心设计原则的两阶段分析框架，验证了分解对全混质量的显著提升

**🔧 技术方法**

采用规则化平衡（ELL）、Diff-MST、MEGAMI等深度学习模型以及两阶段变体

**📊 数据集**

使用Cambridge Multitrack Library的三段流行/摇滚密集混音样本

**📈 对比分析**

通过三项控制听感实验与配对t检验比较，发现两阶段变体比单阶段模型在主观混音质量上有统计学显著提升

**⚠️ 局限性**

实验仅覆盖三首曲目，受试者为具有混音经验的听众，且分组与处理为人工规则，限制了对更广泛音乐类型与自动化流程的推广

---

## 545. Cliff: Learning Process Rewards from the First Mistake

**arXiv ID:** 2609.02817 | [PDF](https://arxiv.org/pdf/2609.02817v1)

**作者:** Peixuan Han `[一作]` (Amazon Web Services), Chris Kong `[通讯]` (Amazon Web Services)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于教师LLM检测推理过程第一次错误点的奖励塑造方法Cliff，将一次性奖励拆分为正确前缀和错误后缀的token级优势。

**💡 创新点**

创新点在于：只需定位推理过程的首次错误点即可提供细粒度监督，避免额外奖励模型或相同推理模式假设，且对模型和任务完全无关，显著提升RLVR学习效率。

**🔧 技术方法**

使用的核心技术包括：强化学习与可验证奖励（RLVR）框架、GRPO算法、教师LLM作为判定器、Token‑level优势分配、Pitfall Step检测、奖励形变和长度正则化。

**📊 数据集**

实验数据集覆盖数学推理（DAPO‑math‑17k、GSM8k、MATH‑500、AIME）和编码（Deepcoder）等，验证了方法的跨域适用性。

**📈 对比分析**

与GRPO、On‑policy Distillation、普通Distillation等基线相比，Cliff在12个不同场景中平均提升约7%（对GRPO）和15%（对OPD），在所有学生模型与教师组合下均保持最优性能。

**⚠️ 局限性**

局限性包括：对教师的准确性仍有一定依赖（弱教师需额外的真值过滤），λ超参数需谨慎调优以防长度攻击，且在缺乏可靠真值或教师能力不足时效果可能下降。

---

## 546. Causal Probabilistic Programming via Magmadic Do-Notation

**arXiv ID:** 2609.02873 | [PDF](https://arxiv.org/pdf/2609.02873v1)

**作者:** Mario Román `[一作]` `[通讯]`, Mario Román

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一种基于非结合单子（magmad）的因果概率编程金属语言，提供了 do‑notation 与 observe 操作，并通过重写生成可识别的因果估计。

**💡 创新点**

将因果推断的可识别算法（ID、Identify）整合为语言的派生构造（intervene），并给出非结合 do‑notation 的分配语义，解决了传统因果程序需要新原语的问题。

**🔧 技术方法**

使用 Racket 语言实现，利用非结合单子、归一化分布、do‑notation、observe、intervene 重写与可识别算法。

**📊 数据集**

使用医学观察数据（治疗 A/B 对疾病 X 的结果，X1/X2 变体）和烟草调查表作为示例数据集；未在大型公开数据集上做实验。

**📈 对比分析**

通过示例展示与传统随机对照试验结果一致，证明可识别算法的正确性；未给出数值性能指标，只做了概念性演示。

**⚠️ 局限性**

局限在于仅处理可识别的因果模型，未提供大规模实验验证，重写和推理效率未知，且对非结合结构的程序组合仍存在复杂性。

---

## 547. Thinking in Pictures: A Systematic Benchmark for Reasoning-driven Image Generation

**arXiv ID:** 2609.02864 | [PDF](https://arxiv.org/pdf/2609.02864v1)

**作者:** Yutong Liu `[一作]` (University of Illinois Urbana Champaign), James M. Rehg `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个2,000样本的 RIG‑Bench 基准，用以评估模型在视觉推理到图像生成的闭环能力。

**💡 创新点**

首次将视觉推理与图像生成统一为“推理到生成”任务，并引入无语言目标、全程图像输入输出的评测框架。

**🔧 技术方法**

结合统一多模态生成模型、LLM 判评规则、感知相似度指标（DINO、CLIP、LPIPS、FID）对模型进行评估。

**📊 数据集**

从 ARC、Raven 矩阵、科学过程、跨域类比等多源视觉推理任务改造而来，形成四大任务族共 11 个细分子任务的 RIG‑Bench。

**📈 对比分析**

对开源与专有图像/视频生成模型进行端到端生成，使用 LLM 裁判和感知度量，最优模型 Gemini 3 Pro 得到 64.6/100，开源模型低于 32，整体存在显著推理‑生成差距。

**⚠️ 局限性**

评测仅覆盖图像输出，未考虑多步生成与交互式推理，且难以区分视觉感知与生成错误的根因，导致对模型真正推理能力的诊断仍受限。

---

