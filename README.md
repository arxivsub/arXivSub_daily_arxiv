# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-11-11 | 今日论文总数: 692

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Graph Visualizations in Prompts for Knowledge Graph Question Answering

**arXiv ID:** 2611.12938 | [PDF](https://arxiv.org/pdf/2611.12938v1)

**作者:** Luiz do Valle Miranda `[一作]` (Jagiellonian University), Grzegorz J. Nalepa `[通讯]` (Jagiellonian University)

**通讯引用:** 2835 | [OpenAlex ID](https://openalex.org/A5023890658)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究在知识图谱问答（KGQA）中加入图形可视化，对大规模视觉语言模型（VLM）在推理阶段的表现进行系统评估。

**💡 创新点**

首次在推理阶段独立评估视觉提示对KGQA的贡献，并提出按图形复杂度与布局自适应的视觉提示策略。

**🔧 技术方法**

利用RoG框架的文本+图像多模态输入，生成Graphviz层级图、NetworkX布局和路径分解图像，结合多模型（Qwen、Mistral、Gemini）进行实验。

**📊 数据集**

使用WebQuestionSP与Complex WebQuestions共1,500个问答实例，按三层复杂度划分，并利用公开的RoG推理路径构造图上下文。

**📈 对比分析**

通过F1、精确率、召回率的宏平均和配对检验对比文本与文本+图像，平均提升1.85 F1点，低复杂度场景提升4.65点，多模型表现一致。

**⚠️ 局限性**

仅评估单一KGQA管道和有限模型，视觉设计空间受限，仅探究一组布局与分解策略，未考虑训练时可视化暴露、成本与人类评估。

---

## 2. AgentScript-Eval: Benchmarking LLMs and Agents for Code Generation in an Enterprise DSL

**arXiv ID:** 2611.13183 | [PDF](https://arxiv.org/pdf/2611.13183v1)

**作者:** Dipin Khati `[一作]` (William & Mary), Phil L. Mui `[通讯]` (Salesforce)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 AgentScript-Eval 基准，用于评估在 Salesforce Agentforce 上使用 AgentScript DSL 的 LLM 代码生成性能。

**💡 创新点**

创新点在于首个可验证的企业专用 DSL 基准，系统分析了模型规模、价格、agentic 循环以及域技能包对生成质量的影响。

**🔧 技术方法**

采用了编译+执行评测框架、pass@k、pass率、token 成本等指标，并对单射、两射、agentic 循环和不同技能包配置进行了对比实验。

**📊 数据集**

使用了 100 个来自真实 Agentforce 项目的任务数据集，涵盖作者、调试、优化三类场景。

**📈 对比分析**

在七种配置下对五个模型（GPT‑5.5、Claude Opus 4.8、Claude Sonnet 5、GPT‑5 mini、Gemma‑4 31B）进行编译率、pass@k、pass率和 token 成本比较；最佳配置的 pass@1 仅达 37%，编译率最高达 98%。

**⚠️ 局限性**

局限性包括：模型规模与价格并非性能的强预测因子；零射作者任务几乎无解；即使成功编译，大多数程序仍存在行为错误，提升主要依赖 agentic 循环和精细的域指导。

---

## 3. One Byte, One Token: Byteifying a 27B Hybrid-Attention LLM with Self-Generated Data

**arXiv ID:** 2611.13607 | [PDF](https://arxiv.org/pdf/2611.13607v1)

**作者:** Liwei Zhang `[一作]`, ZhiMing Tong `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将一个 27B 的子词（BPE）模型直接转换为纯字节模型：将词表从 248k 缩减到 384（256 个字节符号 + 特殊符号），保持其余参数不变，随后在 4.77 GB 的监督字节上进行两阶段持续训练，得到的 byte‑model 在大多数基准上与原始模型相当，并在字符级操作上有显著提升。

**💡 创新点**

创新点：① 采用“零改动、无补丁、无对齐”直观的 byte‑ification；② 通过自生成的对话与字典数据实现对话行为与编码层的自监督；③ 在两阶段课程中先学习字节↔字符映射再学习阅读，显著降低对大规模数据的需求；④ 针对字节模型设计的解码守护器、重复抑制与草稿头以解决特有的失败模式。

**🔧 技术方法**

技术手段：词表裁剪与嵌入/输出头重构；两阶段持续训练（编码+阅读）配合自生成的多模态数据；字节级解码守护器（UTF‑8 校验、预算与循环断路）；多步推理草稿头（MTP）以补偿步长膨胀；对话监督通过原模型的“on‑policy”重答；评估使用同一 GPU、相同服务堆栈的基准脚本。

**📊 数据集**

数据集：4.77 GB 的监督字节（54% Wikipedia、34% 词典、6% 字典、6% 对话、6% 编码会话）；4.77 GB 的自监督对话（角色字典、词典、通用对话、编程会话）；54% 的人类百科文本用于知识阅读；还使用了 C‑Eval、MMLU‑Pro、GSM‑8K、MATH‑500、HumanEval+、IFEval、LongBench‑v2、CUTE/EXECUTE 字符级任务及 22 语言烟雾测试等。

**📈 对比分析**

比较方法：在相同 RTX‑6000 Ada 48G、相同 FP8、相同推理堆栈下，测量思考与非思考两种解码模式的准确率、输出长度、以及字节/字符/token 速率；结果显示：非思考模式下 GSM8K、MATH‑500、HumanEval+、MMLU‑Pro 与原始模型无显著差距；C‑Eval 与 IFEval 在非思考模式下下降 4–6 分；思考模式下 GSM8K、C‑Eval 下降 4–5 分；字符级任务提升 8–12 分；但中文字符识别知识（发音、笔画等）下降 20–40%。

**⚠️ 局限性**

局限性：① 通过字节模型导致输出步长增大，吞吐量下降至原来的 1/4–1/5；② 在字符级知识检索（发音、笔画、同音异字）上性能下降；③ 思考模式不稳定，部分任务可能“思考不闭”；④ 存在四类特有失败模式（贪婪失控、UTF‑8 破坏、低频处罚无效、重复率高）需额外工程或训练补救；⑤ 多语言低资源脚本生成效果差；⑥ 视觉塔未重新适配；⑦ 仅在 27B 规模实验，难以推广到更大或更小模型。

---

## 4. SLIC: Reinforcement Fine-Tuning Small LMs for Multi-Turn Analog Circuit Optimization

**arXiv ID:** 2611.13452 | [PDF](https://arxiv.org/pdf/2611.13452v1)

**作者:** Yan Xu `[一作]` (MIT), Ruonan Han `[通讯]` (MIT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对模拟集成电路的多轮闭环优化，提出了一种基于小型语言模型的RL框架SLIC，支持本地部署且无需外部LLM服务。

**💡 创新点**

创新点包括：① 将结构化电路表示与离散相对更新动作结合；② 设计领域门控和成功条件Slack奖励，提升有效设计的信用分配；③ 引入Reward‑Guided Rollout Steering（RGRS）在每步模拟反馈中进行搜索并构造轨迹；④ 构建公开的41个放大器拓扑基准，兼顾ID、OOD和跨工艺迁移。

**🔧 技术方法**

技术主要有：多轮GRPO强化学习、SFT预训练、RGRS搜索、域门控奖励、Softmax/Random/Greedy继续策略、4B规模的Qwen3.5‑4B小型语言模型。

**📊 数据集**

使用基于65 nm CMOS PTM的模拟器生成的合成数据，训练时利用所有35个ID拓扑；评估时拆分为6个OOD拓扑，并跨22–130 nm CMOS节点以及TSMC 65 nm PDK进行迁移测试。

**📈 对比分析**

与Bayesian Optimization、Zero‑shotQwen3.5‑4B、DeepSeek‑V4.1‑Flash Think、GPT‑6‑Astra以及Tree MT‑GRPO等基线对比。SLIC‑Random在ID场景八回合下实现87.1% Pass@1，超过DeepSeek 38.6%和GPT‑6‑Astra 90.6%（四回合）。在OOD拓扑上SLIC‑Softmax达72.6% Pass@1，远高于DeepSeek 26.7%。跨工艺迁移仍保持60–70% Pass@1，证明了模型泛化能力。

**⚠️ 局限性**

局限包括：① 对于更大规模或混合信号系统仍需扩展多代理协同；② 受限于可用的PTM模型，真实工艺差异会导致性能下降；③ 训练成本相对较高，需多轮模拟调用；④ 仍无法处理极复杂拓扑或极端参数约束的情况。

---

## 5. A Unified Interactive Framework for Real-Time Three-Dimensional Cellular Automata Simulation in XR

**arXiv ID:** 2611.13296 | [PDF](https://arxiv.org/pdf/2611.13296v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 6. Multifidelity Operator Inference for Non-intrusive Model Reduction under Data Scarcity

**arXiv ID:** 2611.13624 | [PDF](https://arxiv.org/pdf/2611.13624v1)

**作者:** Vivian Zhang `[一作]` (Georgia Institute of Technology), Elizabeth Qian `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出多阶层控制变量算子推断（MF‑OpInf）方法，利用低阶仿真数据降低高阶算子学习的方差，并保持无偏估计。

**💡 创新点**

创新点在于把多阶层控制变量框架与非侵入式算子推断相结合，针对每个算子列采用无偏低方差的线性回归估计，并给出最优系数与样本分配公式，显著提升在数据稀缺情境下的学习效果。

**🔧 技术方法**

采用多阶层控制变量线性回归、POD（基于随机场KL展开的解析基底）、有限差分离散、隐式欧拉/半隐式欧拉、Crank‑Nicolson等数值积分方法，结合高低阶模型的矩阵特性进行列‑wise 估计。

**📊 数据集**

三组数值实验数据：
- 1D 热方程（N=128，高阶；Nℓ=16，低阶），
- 1D 黏性布格尔方程（N=128，高阶；Nℓ=16，低阶），
- 1D Kuramoto‑Sivashinsky 方程（N=512，高阶；Nℓ=64，低阶），
其中初始场使用高斯随机场（KL展开）或正弦/余弦组合，低阶模型采用粗网格。

**📈 对比分析**

比较指标为平均相对状态误差（热、布格尔）和相对算子误差（KSE），在不同训练预算p（10、100、1000、10000 等）和保留模态数r 下，与单高阶 OpInf 进行对比。结果显示，MF‑OpInf 在低预算下误差降低约10–50%，在高预算下亦保持优势。

**⚠️ 局限性**

局限性：误差仍高于传统模型降维文献，主要受限于训练样本不足和对未来时刻的外推；仅使用单一低阶模型，未考虑多层或不同误差级别的组合；列‑wise 估计忽略算子间相关性；POD 基底的选择对性能影响显著。

---

## 7. Data-Efficient Unlearning-Based Backdoor Detection in Federated Learning

**arXiv ID:** 2611.13491 | [PDF](https://arxiv.org/pdf/2611.13491v1)

**作者:** Dowon Kim `[一作]` (Korea University), Changhee Joo `[通讯]` (Korea University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在联邦学习中检测后门攻击的数据高效方法SURA；

**💡 创新点**

创新点在于通过重复清洗损失上升探测并在读取时恢复批归一化统计，既提升探测深度又消除BN漂移影响；

**🔧 技术方法**

采用多步梯度上升（unlearning）探测、BN状态恢复、统计阈值筛选及基于中位数的异常分数；

**📊 数据集**

在CIFAR-100（VGG16-BN）与CIFAR-10（ResNet18）上进行实验；

**📈 对比分析**

与FedAvg、MASA、FLTrust等多种防御方法对比，SURA仅使用10张清洗图像即可将六种后门攻击的平均成功率降至0.53%，保持主任务精度与大样本防御相当；

**⚠️ 局限性**

局限性包括对BN结构的依赖、需要预先设定阈值、对完整服务器代理泄露的攻击易失效

---

## 8. Answering Conjunctive Queries over Linear Existential Rules with Transitivity

**arXiv ID:** 2611.13138 | [PDF](https://arxiv.org/pdf/2611.13138v1)

**作者:** Cristina Feier `[一作]`, Filip Murlak `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了由线性元组生成依赖（TGD）和传递性断言组成的本体上的合取查询蕴含问题，证明了在二元符号下线性多头TGD的查询蕴含问题是Ackermann难度的。

**💡 创新点**

创新点在于将耗尽计数器机（lossy counter machine）的可达性问题与本体查询蕴含问题建立了紧密的归约，利用奇偶性编码和传递性谓词实现对计数器状态的精确模拟，从而得到新的下界结果。

**🔧 技术方法**

主要技术包括：线性TGD与传递性规则的逆向重写方法、奇偶性块编码计数器值、三种状态/计数器谓词（奇偶、底部），以及对计数器增减操作的专门线性规则。

**📊 数据集**

该研究基于理论分析，不使用实际数据集；主要通过对抽象的耗尽计数器机实例进行归约。

**📈 对比分析**

由于论文是关于理论复杂度的证明，未进行实验对比；作者通过归约展示了问题的Ackermann难度，即不存在多项式或更低复杂度的算法。

**⚠️ 局限性**

局限性包括：仅给出下界结果，未提供上界或可行算法；归约只覆盖二元符号和特定形式的线性多头TGD；实际应用中的优化和实现细节未作探讨。

---

## 9. Exploiting Feature-invariant Scaling Laws for Efficient A/B Experimentation in Recommendation Systems

**arXiv ID:** 2611.13539 | [PDF](https://arxiv.org/pdf/2611.13539v1)

**作者:** Zhiliang Chen `[一作]` (Meta), Xiaoqiang Luo `[通讯]` (Meta)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种利用推荐模型的特征不变缩放规律进行小规模实验并预测全规模A/B实验效果的方法；

**💡 创新点**

发现并应用特征不变缩放律，并将统计推断与贝叶斯/频率决策结合，能够在仅做少量实验的情况下可靠预测全规模表现；

**🔧 技术方法**

采用功率律缩放模型、线性回归+高斯噪声假设、Student‑t 后验推断、贝叶斯/频率阈值决策以及多种GPU训练设置；

**📊 数据集**

使用Criteo点击率数据、MovieLens‑20M（预测下一个电影）、以及预训练LLM（Llama‑3.1‑Instruct、Qwen‑3‑Instruct）等公开数据集；

**📈 对比分析**

在超过40种特征变更上与基线进行对比，HR@10/50/200等指标均实现1%以上提升，预测准确率>90%，误差随小规模实验数增加而显著下降，且计算成本约比全规模实验低45%；

**⚠️ 局限性**

当新增特征导致模型容量不足或缩放曲线不再遵循功率律（如加入大量上下文导致的超出容量）时，方法失效；同时需要足够多且高质量的小规模实验来保证统计可靠性。

---

## 10. Settling the Burn-in Gap: Sharp Risk Bounds for Tail-Averaged SGD in Linear Regression

**arXiv ID:** 2611.13383 | [PDF](https://arxiv.org/pdf/2611.13383v1)

**作者:** Qingyue Zhao `[一作]` (University of California Los Angeles), Quanquan Gu `[通讯]` (University of California Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

本文对恒定步长在线随机梯度下降（SGD）在高维线性回归中的尾部平均（tail‑averaging）策略进行严格的收敛性分析，给出了初始化相关的匹配上界和下界。

**💡 创新点**

创新点在于：①提出并利用“谱反馈比”χ_{s,N}来衡量整个训练轨迹中四阶矩能量的累积反馈；②通过对接收权重F_t与传播核k(h)的对偶中点卷积结构，证明在任意协方差谱下均可满足χ_{s,N}<1，从而实现匹配的上下界；③在多项式谱（λ_i≈i^{-a}, a>1）下给出单一无阶梯学习率的理论保证；④通过二维Gaussian与一维离散分布的对比构造，展示了原先下界的指数-多项式差距。

**🔧 技术方法**

主要技术手段包括：轨迹重组的扰动分解（perturbation decomposition）、离散Volterra卷积吸收不等式、四阶矩超算子理论、对偶中点卷积（pair‑midpoint convolution）与核双倍常数D_T分析、以及谱分区（spectral partition）与有效维数技巧。

**📊 数据集**

论文为理论分析，未使用具体实验数据集；所有结果均在假设下的高维（甚至无限维）线性回归模型中推导。

**📈 对比分析**

与之前工作（如Zou等人2021）相比，本文消除了头部初始化子空间中未被衰减的误差贡献，给出了完全匹配的上/下界；在典型s=Θ(N)情形下，误差收敛率保持在1/(γ^2N^2)的量级，并在满足χ_{s,N}<1时进一步实现指数级的初始误差衰减；数值实验（在本文附录中）与理论上限相符。

**⚠️ 局限性**

限制主要包括：①需要满足谱反馈条件χ_{s,N}<1，若不满足则上界不再紧致；②对于标量（d=1）或极端谱，双倍常数D_T会随T指数增长，导致无法使用单一无阶梯步长；③仅给出理论上限，下界示例说明在某些离散分布下仍存在更优策略，说明仅凭四阶矩约束无法完全决定收敛速率。

---

## 11. Parametric Proof-Carrying Compilation for Quantum Error-Correction

**arXiv ID:** 2611.12985 | [PDF](https://arxiv.org/pdf/2611.12985v1)

**作者:** John Zhuoyang Ye `[一作]` (University of California Los Angeles), Jens Palsberg `[通讯]` (University of California Los Angeles)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

本文提出一种参数化的证明携带编译框架，能够在量子错误更正（QEC）代码的不同距离和检波方案下生成证明，证明生成的电路具有电路层级的码距下界。

**💡 创新点**

创新点在于将代码家族的几何结构用符号切割（cuts）和代数归纳表示为单一有限证明，该证明与电路的提取方案无关，可跨多种检波方案复用；并通过 Lean 4 机械化实现，保证证明的正确性。

**🔧 技术方法**

主要技术包括：CodeML 语言用于描述可扩展的 QEC 家族；QStab 语言记录检波过程中的残余错误与注释；基于 Hoare 逻辑的电路级错误注入模型；切割签名（cut signatures）和钩子错误（hook error）对断言；以及编译器将源级证明与 QClifford 目标电路对应的证明传递。

**📊 数据集**

实验使用了五个代码家族（旋转 Surface、未旋转 Surface、HGP、Toric、Cartesian LP）以及若干有限代码（五量子位、Steane、Cube、Shor 等），对每个家族在四种检波方案（Standard、Flag、Shor、Knill）下进行评估。

**📈 对比分析**

比较方法是对每个家族与方案生成符号证明并测量检验时间；结果显示：旋转 Surface 的证明平均耗时 214 s，未旋转 Surface 23.9 s，Toric 9.48 s，其他家族在数十秒到数千秒范围内；对有限电路的证明检查则在 0.2–0.6 s 之间完成。

**⚠️ 局限性**

局限性包括：证明规模仍随提取顺序复杂度增长；需要对代码家族具有已知代数结构；依赖于电路级错误模型和读出合同的准确性；在提取方案或硬件实现变更时，需要重新证明钩子错误与切割的上界。

---

## 12. One Table, Many Queries: Benchmarking Generalist Tabular Models Across Generation, Imputation, and Prediction

**arXiv ID:** 2611.13001 | [PDF](https://arxiv.org/pdf/2611.13001v1)

**作者:** Yuanrui Wang `[一作]` (StableAI), Peng Cui `[通讯]` (StableAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了统一的 One Table Benchmark (OTBench)，在相同的数据集和能力轴上同时评估生成、缺失值填补与预测三种表格任务，并对方法、训练目标组合以及固定模型泛化进行系统比较。

**💡 创新点**

① 将三任务对齐为同一数据集与能力维度；② 通过单目标与交叉目标干预实验（SAE、CAE）探究训练目标的互补性；③ 对固定模型与专用组合的性能进行 Pareto 前沿比较；④ 将整个 benchmark 开源，提供完整代码和配置。

**🔧 技术方法**

采用共支持排名、Spearman 相关、目标组合干预实验、扩散、VAE、Copula 等生成模型，填补模型（TabDiff、TabSyn 等）与预测模型（TabPFN、LimiX 等），并使用多维评估指标（分布保真、依赖保持、预测效用、隐私诊断等）。

**📊 数据集**

使用 56 个公开表格数据集，包括分类与回归任务，如 JapaneseVowels、cleveland、credit_card_clients_default、website_phishing、MiamiHousing2016 等。

**📈 对比分析**

通过共支持排名在 20 个能力轴上对方法进行比较。结果显示填补与预测在数值恢复、依赖保持上高度相关；生成在分布保真与预测实用性上呈现不同关系；固定模型中无单一最佳，基于任务特定指标的模型往往优于专用组合；基础模型在无微调下已表现出强劲的预测与数值填补能力。

**⚠️ 局限性**

主要局限包括：指标与任务合同的选择可能影响结论；目标干预仅在排名层面，未揭示机制；交互效应与容量竞争尚未深入；统一模型在分类重建与联合采样方面仍有不足；实验受固定数据集、训练预算及可用方法覆盖范围的限制。

---

## 13. Darmok: Championship Cryptic Crossword Solving via Constraint Tightening and Resilient Orchestration

**arXiv ID:** 2611.13557 | [PDF](https://arxiv.org/pdf/2611.13557v1)

**作者:** Siddhartha Chatterjee `[一作]` `[通讯]` (University of Texas at Austin), Siddhartha Chatterjee (University of Texas at Austin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套完整的自动化英国密文填字系统，采用多模型并行推理、同步交叉验证、备份路径与异步全局填充，最终在 55 赛次英伦“Times Crossword Championship”中以 100% 正确率和最快速度（6 min 52 s）完成全部 30 题，并在日常生产跑 132 天内保持 99.84% 的正确率和 5 min 36 s 的平均解答时间。

**💡 创新点**

创新点包括：
① 重新设计的同步“重建‑保持”候选人归纳框架（Jacobi‑style 边界协同），消除了增量提交的顺序依赖和部分单词覆盖错误；
② 三台 LLM（Gemini、Claude、GPT‑5）在同一轮次内并行发问并依据置信度阈值、共识与历史答案进行投票，提升尾部鲁棒性；
③ 采用异步全局网格填充与局部字典检索作为非侵入式备份，保证在主模型延迟或失败时仍能完成填字；
④ 在极低错误容忍的比赛规则下，提出多模型并发、跨单元约束与重叠词的“主动‑被动”复制方案，实现零错误的最优解。

**🔧 技术方法**

技术栈：
- 三大 LLM 端点（Gemini、Claude、GPT‑5）做 per‑clue 解析；
- 同步 barrier admission 与 Jacobi‑style 交叉支持评分；
- 置信度阈值、共识打分、单词频率回退；
- 异步全局填充（whole‑grid）和本地词典检索；
- 多轮重试、历史答案回退、局部终端链（corr/pers/atf/swp）；
- 进程并发控制、timeout 90 s/150 s、活跃‑复制复制器；
- 轻量化 prompt 结构与自我信心去除实验。

**📊 数据集**

数据集：
- 132 天英伦 “Times Cryptic Crossword” 正式每日 15×15 版（后期 43 天冻结、前期 39 天实验）共 3 876 条线索；
- 16 题分层难度集（Easy–Extreme）共 463 条线索；
- 10 题 12×12 barred “Mephisto” 版 360 条线索（零-shot 迁移）；
- 135 次单次前沿模型抽样；
- 55 次英伦比赛 7 题全流程跑。

**📈 对比分析**

比较方法：
- 与单模型 V1/V2 以及 2025 赛季基线做前向/后向对照；
- 与人类参赛者的准确率、错误率、完成时间做直接对标（Qualifiers、Semi‑Final、Final 速度排名、错误率 0%）；
- 日常跑 132 天数据的平均/尾部分析；
- 单次抽样 135 次的 tail‑regime 性能；
- 结果显示：系统在比赛中 100% 正确率、最快 6 min 52 s，日常 99.84% 正确率、5 min 36 s 平均；测试集 99.57% 正确率，Mephisto 97.2%。

**⚠️ 局限性**

局限性：
- 依赖多大模型云端接口，算力和费用高；
- 对极端慢速/网络中断仍需复杂备份，系统复杂度高；
- 对新出现的词汇、罕见语法或非常规解析仍有失败；
- 缺乏可解释的词play解析器，难以交互式调优；
- 需手工设置信赖阈值、timeout、重试轮次等超参；
- 目前仅针对英国 15×15/12×12 版，其他网格类型需进一步适配。

---

## 14. DEPUTYBENCH: Evaluating Contextual Safety in Enterprise Agents

**arXiv ID:** 2611.13355 | [PDF](https://arxiv.org/pdf/2611.13355v1)

**作者:** Hao Zhao `[一作]` (Independent Researcher), Minghao Lin `[通讯]` (University of Southern California)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个可执行的企业权限决策基准，包含79个基础任务、375个实例以及每个实例的合法/可疑双版本，并配备可验证的审计机制。

**💡 创新点**

创新点在于：① 通过“配对权限世界”设计，使同一业务目标在不同权限上下文下进行对比；② 引入分级证据需求（L1–L3）和显式/隐式检查提醒；③ 通过可执行环境和版本化审计实现自动化安全与完成度评估；④ 在受控实验中揭示缺失证据导致更频繁的受限执行。

**🔧 技术方法**

使用的技术包括：大型语言模型代理（如 GPT‑6 Astra、GPT‑6.1 Sol、Gemini‑4、Seed 系列），可执行容器化工作流，基于脚本的审计器（scorer‑v2.1）以及对模型输出的手动与自动验证。

**📊 数据集**

数据集：79个基础任务、375个实例（包含188对合法/可疑实例）以及由原始任务衍生的28个受控实验任务；任务涵盖支付欺诈、未授权操作、客服伪装、隐私违规、对象访问违规和秘密泄露六大类。

**📈 对比分析**

比较方法：对12个不同模型进行批量跑测，评估安全率、完成率与配对成功率；在受控实验中对授权事实翻转、证据掩蔽、无关重写进行统计检验。结果显示整体安全率为88%，配对成功率为79%；在缺失证据情形下，受限执行率显著高于显式非授权（差值6.8个百分点，95%置信区间 1.9–13.0）。

**⚠️ 局限性**

局限性包括：任务为合成、仅英文；只涵盖六大任务类别和单一执行器；对 LLM 评估的依赖导致标注一致性（约0.705）不高；审计器仍存在误判；缺少公开复现与完整手工验证；未覆盖更复杂或真实业务场景。

---

## 15. Discrete Model-Based Diffusion for Combinatorial Optimization in Multi-Robot Task Allocation

**arXiv ID:** 2611.13382 | [PDF](https://arxiv.org/pdf/2611.13382v1)

**作者:** Haejoon Lee `[一作]` (University of Michigan), Dimitra Panagou `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出离散模型无训练扩散（Discrete MBD）方法，用于在多机器人任务分配中直接在离散决策空间进行反向扩散搜索。

**💡 创新点**

创新点在于将离散扩散模型与基于目标分布的采样优化结合，实现无训练、直接在离散空间反向扩散，并给出有限步收敛保证；同时提出优先编码方案确保每个样本可解。

**🔧 技术方法**

采用离散扩散过程（D3PM）作为前向噪声链，利用目标分布近似逆转分布，通过蒙特卡洛采样实现逆转转移；结合采样优化（SBO）、优先编码及温度调节。

**📊 数据集**

在通用整数程序（UIQP、UPO）以及异构多机器人任务分配（SS-BT、SS-AT、MS-AT）等合成数据集上进行实验。

**📈 对比分析**

与 CEM、SMC、Rounded MBD 等采样基优化以及 LNS、SA、RL、CTAS-D 等基准比较，Discrete MBD 在绝大多数设置下取得最低成本/最短完工时间，计算时间仅为几秒，性能优于其他采样方法并逼近 MIP 与 RL 的结果。

**⚠️ 局限性**

局限在于需要足够多的扩散步骤才能竞争，样本效率随问题规模下降；目前仅处理确定性目标，缺乏对不确定性建模；在更大规模实例上仍难与学习方法匹敌，并未实现分布式实现。

---

## 16. VIGOR: Benchmarking Visual Rationale Correctness in Multimodal Large Language Models

**arXiv ID:** 2611.13150 | [PDF](https://arxiv.org/pdf/2611.13150v1)

**作者:** Yanlin Chen `[一作]` (University of Virginia), Xi Peng `[通讯]` (University of Virginia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 VIGOR 基准，用于评估多模态大语言模型（MLLM）在生成视觉描述时是否真正“看到了”所说的概念；

**💡 创新点**

创新点包括：①将视觉合理性定义为模型输出所依赖的像素级视觉证据；②构建覆盖低级属性到高级关系的多层概念空间；③开发自动标注框架，利用现有 dense supervision、概念补全、掩码生成与多模态验证，实现大规模像素级标注；④通过对多模态模型的层级关注图进行评估，揭示其视觉合理性缺陷。

**🔧 技术方法**

采用的技术包括：MLLM attention map 提取、Grad‑CAM、LogitLens、SAM/Grounded‑SAM、CLIP‑CAM/CLIPSeg 等多种掩码生成器；多模态验证器 Qwen3.5‑9B、InternVL3.5‑8B、GLM‑4.6V‑Flash 的投票；评估指标 AP 与 Pointing Accuracy；以及自动标注与概念补全的脚本。

**📊 数据集**

使用的数据集：COCO、COCO‑Stuff、LVIS、PACO、Visual Genome、ADE20K、Open Images、AVA、Kinetics、GQA、VAW、DTD 等；并基于这些资源与自动生成得到的 1,000 张图像构建 VIGOR，包含 3,011 个概念与 31,433 个像素级掩码。

**📈 对比分析**

对 InternVL、Qwen‑VL、Gemma、Ministral 等 1B‑38B 规模模型在固定 caption 的前提下进行比较，采用注意力映射与 ground‑truth 掩码对齐；与随机基线对比，最佳模型仅达 31.2% AP、38.7% Pointing Accuracy；同时发现不同层级的关注图存在非单调趋势，并与模型的误报（hallucination）表现呈正相关。

**⚠️ 局限性**

限制：①视觉合理性整体仍偏低，尤其在关系、动作等高级概念上表现差；②自动验证虽保守但仍可能误判；③评估仅依赖注意力映射，未覆盖所有可能的可解释方式；④对细粒度概念的定位仍有难度；⑤多模态验证器与被评模型相同族类可能引入偏差。

---

## 17. TMAL: Topology-preserving Multi-offset Affinity Learning for Curvilinear Structure Segmentation

**arXiv ID:** 2611.13511 | [PDF](https://arxiv.org/pdf/2611.13511v1)

**作者:** Suho Han `[一作]` (Soongsil University), Yoseob Han `[通讯]` (Soongsil University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了Topology-preserving Multi-offset Affinity Learning（TMAL），用于提高曲线结构分割的像素精度与连通性。

**💡 创新点**

创新点：①用倒置距离变换得到的软方向亲和力目标，突出薄部位；②多偏移采样并通过自适应融合权重结合局部与全局上下文；③辅助分割分支为亲和力学习提供像素级监督。

**🔧 技术方法**

技术细节：方向亲和力预测（MAP）、自适应融合（AF）、距离地图解码（DMD）、辅助分割（AuxSeg）；网络基于ResNet‑34编码器+解码器；使用Smooth L1、Dice、clDice等多重损失。

**📊 数据集**

使用数据集：DRIVE（视网膜血管）和DeepCrack（混凝土裂缝）。

**📈 对比分析**

与Mask、Topology、Connectivity三类方法对比，TMAL在Dice、IoU、clDice上均位居榜首，连通组件误差Δβ₀显著下降（DRIVE 24.5，DeepCrack 20.95），表现优异。

**⚠️ 局限性**

局限性：亲和力目标仅约束像素对偶连通，未直接优化闭环和最大边界误差；采样偏移固定，未学习；Δβ₁与HD95在部分数据集上仍不如最佳。

---

## 18. DASWeave: Weaving Evidence across Speaking Conditions for Multimodal Depression, Anxiety, and Stress Screening

**arXiv ID:** 2611.13544 | [PDF](https://arxiv.org/pdf/2611.13544v1)

**作者:** Binyan Chen `[一作]` (University of Science and Technology of China), Shangfei Wang `[通讯]` (University of Science and Technology of China)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在儿童和青少年中使用多模态方法（语音、视频和文本）对抑郁、焦虑和压力进行自动筛查，创新性地将说话条件（朗读、叙述、回忆）之间的行为差异引入模型；

**💡 创新点**

核心创新点是：① 两级树融合将每个说话条件的行为统计（包括跨条件差异）转化为稠密向量加入到会话和参与者表示；② 参与者语言融合利用微调的LLM对所有回答与朗读基线的行为描述进行联合编码；③ 通过MLLM辅助监督将多模态信息投射到主网络，提升学习信号；

**🔧 技术方法**

使用的技术包括：共享会话编码器（音频-视觉-文本）、LightGBM监督树、Qwen3-8B微调的语言模型、LoRA参数适配、注意力融合、MLLM（Q-Former+Qwen3）辅助监督、噪声正则化与会话采样、参数平均、任务特定偏置校准；

**📊 数据集**

实验数据集为ACM Multimedia 2026 AdoDAS Grand Challenge Track 1，包含6,000名儿童/青少年共24,000条音视频片段，四种说话条件（朗读、日记叙述、最快乐回忆、最悲伤回忆），每条数据附有音频/视频特征、eGeMAPS、ASR转录和情绪标签；

**📈 对比分析**

在官方隐藏测试集上，DASWeave在95支队伍中排名第2，平均F1为0.5853，比官方音视基线高27.1%，平均AUROC为0.8019，是前五名中最高的；在验证集消融实验中，去除树融合、语言融合或MLLM辅助监督都会使平均F1下降，说明各模块均有贡献；

**⚠️ 局限性**

局限性包括：依赖于特定的AdoDAS数据集，跨学校匹配条件下表现良好，但在不同录制环境或不同人群的泛化性尚未验证；对极少见标签（如压力）的性能仍相对较低；模型规模大、训练成本高，需要大量计算资源；

---

## 19. Are Dense Rewards Worth Their Tokens in GRPO?

**arXiv ID:** 2611.13560 | [PDF](https://arxiv.org/pdf/2611.13560v1)

**作者:** Vikash Singh `[一作]` (Case Western Reserve University), Vipin Chaudhary `[通讯]` (Case Western Reserve University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种匹配计算（matched-compute）协议，用于在相同的生成 token 和 GPU‑小时预算下评估强化学习训练干预；在此框架下验证并比较稠密奖励（dense rewards）和提示选择（prompt selection）两类干预方法。

**💡 创新点**

创新点在于：①首次系统性地将计算成本（生成 token 与 GPU‑小时）纳入干预评估；②通过预注册实验、对照控制和多模型、多随机种子验证，揭示稠密奖励在真实计算成本下并未优于传统 GRPO；③发现提示选择策略能在匹配计算下提升 0.8–1.7 个百分点，并通过候选提示预测成功率阈值来解释其有效性。

**🔧 技术方法**

使用的技术包括：Group‑Relative Policy Optimization（GRPO）与其稠密奖励扩展（Monte‑Carlo value estimates, critic‑based rewards）；提示过滤器（critic filter 与历史过滤器）；匹配计算协议、预注册实验设计、TOST 等统计检验；以及对比实验中的精细控制（随机选择、学习率调整、严格验证器）。

**📊 数据集**

主要数据集为 MATH（含数字答案层级 3–5）以及 GSM8K，所有实验均使用 Qwen 和 Llama 系列小型 LLM（1.5B–4B）。

**📈 对比分析**

在匹配计算下，稠密奖励（如 Monte‑Carlo）在 token 或 GPU‑小时预算相等时均落后于标准 GRPO（差距约 14–15 分点），而提示选择在两种模型中提升了 0.8–1.7 分点，且提升可在不同种子、不同问题集上复现；与双倍学习率 baseline 相比，提示选择的优势仅约 0.5 分点。

**⚠️ 局限性**

限制包括：仅评估了少量小模型和特定任务（MATH、GSM8K）；稠密奖励的改进可能在更大模型或不同任务中表现不同；匹配计算协议虽更真实但实验成本高，限制了实验规模；并且提示选择策略的有效性高度依赖于候选提示的分布和预测模型的精度。

---

## 20. Enforcing Axioms for AI Alignment under Loss-Based Rules

**arXiv ID:** 2611.13369 | [PDF](https://arxiv.org/pdf/2611.13369v1)

**作者:** Alexandros Hollender `[一作]` (Technical University Of Munich), Sonja Kraiczy `[通讯]` (University Of Oxford)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了在AI对齐中使用基于损失的奖励模型时，如何满足 Pareto 最优性（PO）和多数投票一致性（PMC）等社会选择学的公理。

**💡 创新点**

创新点在于：①证明即使奖励模型从线性扩展到多项式或单隐藏层 ReLU MLP，PO 和 PMC 仍可能被违背；②提出在理想化的、方向均匀且长度统一的比较数据分布下，线性奖励模型能够恢复 PO，展示了数据设计对公理保证的重要性。

**🔧 技术方法**

主要技术手段包括：线性社会选择框架、对数损失（Bradley–Terry）、凸/可微损失函数、Berge 最大值定理、几何分析以及对多项式/MLP 结构的代数证明。

**📊 数据集**

论文使用的是构造性合成实验（synthetic counter‑examples），并未采用真实数据集；所有比较和候选点均由作者自行设定以满足理论要求。

**📈 对比分析**

方法主要是理论证明而非经验比较；作者通过构造对抗实例和极限分析证明了 PO/PMC 的失效或恢复，并未给出数值性能指标。

**⚠️ 局限性**

局限性包括：①仅针对固定容量（多项式度数或单隐藏层宽度）模型，无法推广到更深/更大网络；②恢复 PO 的结论仅在理想化的均匀方向分布下成立，现实数据分布可能偏离；③PMC 在均匀数据下仍不满足，说明公理在此框架下不一定可实现。

---

## 21. GEMOR-GS: Global Extrinsic-calibration-refined Mapping and Online Relocalization for 3D Gaussian-Splatting SLAM

**arXiv ID:** 2611.13528 | [PDF](https://arxiv.org/pdf/2611.13528v1)

**作者:** Yuchen Yang `[一作]` (Nanchang University), Hesheng Wang `[通讯]` (Nanjing 4D Vector Technology Co., Ltd.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为GEMOR‑GS的LiDAR‑惯性‑视觉SLAM系统，利用3D高斯分裂（3D Gaussian Splatting）构建高质量、可查询的世界模型，并实现在线定位、离线全局重建和重新定位；

**💡 创新点**

（1）引入离线直接法LiDAR‑摄像头外参校准，并将其质量传递至后续模块；（2）在离线轨迹中交替优化关键帧位姿与高斯地图，提升渲染质量；（3）通过全局重建阶段恢复完整的高分辨率3DGS地图；（4）设计基于XFeat+RANSAC‑PnP+RGB‑D warp的重新定位模块，实现跟踪丢失后的快速恢复；

**🔧 技术方法**

直接法LiDAR‑摄像头对齐（NID优化）、基于差分渲染的图像/深度监督、球谐展开、可微光栅化、RANSAC‑PnP、XFeat特征匹配、RGB‑D光流warp、IMU‑视觉‑LiDAR融合的IESKF；

**📊 数据集**

自采集室内展厅序列、CBD、Retail、Bright、hku_campus、tuhh_day_04等六个公开室内外数据集；

**📈 对比分析**

与在线SLAM地图、Gaussian‑LIC2地图以及不同外参扰动下的同一实验进行对比。离线重建在所有帧对比中平均提升约15‑20 dB PSNR；在同一场景下超越Gaussian‑LIC2约9 dB；重新定位成功率在大多数序列为100%，对外参误差的鲁棒性良好；

**⚠️ 局限性**

对纹理极弱场景（Bright）无效；外参误差大于≈1.5°会显著影响重建质量；系统对实时预算的离线微调仍需加速；目前未能直接生成可用于碰撞检测的网格或语义分割。

---

## 22. URL-Sim: Unified and Regulated LiDAR Simulation for High-Fidelity Reconstruction of Dynamic Driving Scenes

**arXiv ID:** 2611.12932 | [PDF](https://arxiv.org/pdf/2611.12932v1)

**作者:** Kefei Qian `[一作]` (Chongqing University), Guofa Li `[通讯]` (Chongqing University)

**通讯引用:** 7159 | [OpenAlex ID](https://openalex.org/A5030291480)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了统一且受限的 LiDAR 仿真框架 URL‑Sim，利用动态高斯场景图实现从真实驾驶日志中高保真重建，并支持刚性、SMPL 人体和非模板变形等多种动态参与者的建模；

**💡 创新点**

① 统一动态高斯场景图解耦静态背景与动态对象，支持多种非刚性对象；② 受限高斯密度化（RGD）策略，在固定全局预算下通过多维重要性评分精准分配高斯；③ 结合 SMPL 先验与轻量变形网络，实现对行人、骑行者等非刚性对象的高精度重建；

**🔧 技术方法**

3D Gaussian splatting（2DGS）+ 可微 LiDAR 光线跟踪；SMPL 人体先验；轻量变形网络；基于深度、强度、梯度和边缘的多维重要性评分；受限高斯密度化策略；U‑Net 后处理；多项损失（深度、强度、Chamfer、SSIM、BCE 等）；

**📊 数据集**

Waymo Open Dataset 与 KITTI‑360 两个公开大规模驾驶数据集；

**📈 对比分析**

与 NeRF、3DGS 等多种 SOTA 基线（LiDAR‑NeRF、DyNFL、LiDAR4D、LiDAR‑GS、GS‑LiDAR、LiDAR‑RT）在相同 GPU、相同高斯上限（约 3M）下统一评测；指标涵盖 FPS、RMSE、MedAE、SSIM、PSNR、LPIPS、Chamfer Distance、F‑score；URL‑Sim 在 2D 质量（尤其深度/强度 PSNR）和 3D 几何（最低 CD、最高 F‑score）上均显著优于大多数基线，渲染速度接近 LiDAR‑RT；

**⚠️ 局限性**

依赖准确的检测与跟踪框，密集场景下性能下降；大视角编辑导致未观测区域失真；对未观测视角的泛化仍受限，未来可引入生成模型提升 extrapolation 能力。

---

## 23. Synapse-Control: Making Agent Experience Actionable with Lightweight Runtime Control

**arXiv ID:** 2611.13615 | [PDF](https://arxiv.org/pdf/2611.13615v1)

**作者:** Zhiyuan Huang `[一作]` (Shanghai Jiao Tong University), Linpeng Huang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一种轻量级经验重用控制器，将诊断生成从生成式大模型转为1.7B分类器评估候选建议，并使用微合同与Prompt Value Table进行控制

**💡 创新点**

创新点在于：①将经验检索与评估拆分为规则检索+专用分类器，显著降低诊断开销；②用微合同表达接口检查，保持代理自主；③引入PVT进行历史反馈、冷却与预算管理，防止重复提示

**🔧 技术方法**

采用规则检索、1.7B微调分类器、Prompt Value Table、基于微合同的接口抽象、教师模型标注训练

**📊 数据集**

在AppWorld API编程任务和WebShop购物任务两套基准数据集上评估

**📈 对比分析**

与无恢复的Base和Sentry进行对比；在AppWorld上成功率提升至270/1008（比Base提升29.8%，比Sentry提升1.9%），在WebShop上为92/300（比Base提升9.6%，比Sentry提升5.1%）；同时辅助控制时间降低约50%，大模型调用次数减少约68%/60%

**⚠️ 局限性**

局限性包括：依赖微合同的手工/自动构造；需教师模型大规模标注；在非稳定API或非表单化接口的任务中效果未知；分类器误判可能导致不合适建议

---

## 24. Maze Runner: Autonomous Maze Navigation with Line-Based Localization and Control Barrier Functions

**arXiv ID:** 2611.12949 | [PDF](https://arxiv.org/pdf/2611.12949v1)

**作者:** Alvaro J. Gaona `[一作]`, Peter Pasuy-Quevedo `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建并评估了一个基于ROS 2的开源迷宫导航栈（Maze Runner），实现了从LiDAR线段提取、线特征EKF定位/SLAM、A*/Hybrid A*规划以及基于离散控制势函数（CBF）的非线性MPC跟踪。

**💡 创新点**

创新点包括：① 使用RANSAC+总最小二乘（TLS）得到带不确定性的线段，供定位与SLAM共享；② 将线特征直接用于EKF定位，并在SLAM模式下通过候选提升与范围关联处理共线墙；③ 在MPC中加入离散时间CBF约束，并通过扇区选择压缩LiDAR点约束，使求解实时可行；④ 将所有算法放在无ROS依赖的C++核心库中，便于单一可复现环境构建。

**🔧 技术方法**

技术包括：RANSAC+TLS线段提取、基于线特征的EKF与EKF‑SLAM、A* + 视线捷径、Hybrid A*、非线性MPC、离散时间控制势函数（CBF）、扇区化LiDAR点选择、CasADi + IPOPT求解、Gazebo模拟与ROS 2节点封装。

**📊 数据集**

数据集为自行构造的3D迷宫STL模型，在Gazebo中模拟Husarion ROSbot XL；实验在该单一仿真环境中完成，没有使用公开的真实地图或数据集。

**📈 对比分析**

方法对比：A*最短路径耗时6 ms，A*+捷径6.1 ms且路径长度减至133.8 m；Hybrid A*路径更平滑但耗时242.7 ms。定位误差：已知地图EKF RMSE 1.0 cm，SLAM模式RMSE 24 cm；MPC平均求解时间11.7 ms，最高35.6 ms，未超80 ms预算。CBA能安全避开未映射柱体，最小间隙0.57 m。

**⚠️ 局限性**

局限性：规划器仅使用已知墙壁，未将LiDAR障碍反馈至占据栅格；SLAM缺乏闭环校正，误差随探索增长；未在真实机器人上验证；未实现对完整自由空间覆盖或从SLAM地图直接规划。

---

## 25. We Used It Anyway: Low Trust and High Adoption of Large Language Models in Requirements Engineering Education

**arXiv ID:** 2611.12889 | [PDF](https://arxiv.org/pdf/2611.12889v1)

**作者:** Riccardo Coppola `[一作]` (Politecnico di Torino), Luca Ardito `[通讯]` (Politecnico di Torino)

**通讯引用:** 1254 | [OpenAlex ID](https://openalex.org/A5040165772)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在硕士软件工程课程中，通过对完成需求工程阶段的43名学生进行问卷调查，记录他们对大型语言模型（LLM）的使用工具、使用场景、输出质量感知以及对课程中AI整合的期望。

**💡 创新点**

首次系统化探讨学生在真实项目环境中自发使用LLM的行为与态度，并提出将提示工程与批判性评估纳入课程的可行路径，填补了RE教育与LLM应用研究的空白。

**🔧 技术方法**

使用多种公开LLM（Gemini、ChatGPT、Claude等）以及Google Forms进行问卷收集，采用开放式编码方法对开放性回答进行质性分析。

**📊 数据集**

数据来源为一次性收集的43名学生问卷结果（包含性别、年龄、工具使用频率、任务分布、输出质量感知等信息）。

**📈 对比分析**

通过频数统计与开放式编码对比不同工具在各任务中的使用率、学生对输出质量的主观评估以及多模型比较策略，结果显示LLM使用普遍但多数学生认为输出低于自我完成，且对Gemini的偏好主要由免费Pro订阅驱动；未提供客观性能指标。

**⚠️ 局限性**

研究局限包括单一机构、单一课程的样本不具代表性；自我报告的数据可能存在社会期望偏差；缺乏对实际提示与输出内容的深入分析；对LLM性能的评估仅基于主观感知，缺乏客观度量。

---

## 26. Bootstrapped On-Policy Distillation

**arXiv ID:** 2611.13680 | [PDF](https://arxiv.org/pdf/2611.13680v1)

**作者:** Yue Wu `[一作]` (Hong Kong Polytechnic University), Jian Huang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 On‑Policy Self‑Distillation (OPSD) 的训练动态，发现其早期提升会在继续训练后逐步消退；提出 Bootstrapped On‑Policy Distillation (BOPD)，通过两阶段流程先用 OPSD 在有参考答案的情境下构建任务适配教师模型，再将该教师在无参考答案的情境下对新的学生模型进行对抗性蒸馏，从而显著提升模型的数学推理和代码生成能力。

**💡 创新点**

创新点在于：① 将教师构造与教师使用分离，先利用 privileged reference 信息让教师在 Stage I 内部化任务相关知识；② 在 Stage II 使用无参考的教师，消除教师与学生在状态分布上的不匹配，从而保持持续的学习收益；③ 通过实验验证该两阶段方法可实现比单纯继续 OPSD 更好的性能，并对其机制进行了量化分析。

**🔧 技术方法**

主要技术包括 On‑Policy Distillation、Self‑Distillation、KL 逆向熵蒸馏、LoRA 参数高效微调、基于强化学习的目标函数、token‑级监督、温度采样、top‑p/k 策略等；BOPD 采用两阶段流程：Stage I 用 OPSD (teacher 受 reference 影响) 训练，Stage II 用蒸馏 loss 将 Stage I 产出的教师蒸馏到新学生。

**📊 数据集**

使用的主要数据集：数学推理任务基于 OpenThoughts 数据集、AIME 2024/2025、HMMT 2025；代码生成任务基于 TACO (DeepCoder) 子集、HumanEval+、MBPP+ 以及 LiveCodeBench v5/v6；实验中还对 HumanEval、IFEval、MMLU 等进行能力保持评估。

**📈 对比分析**

与原始模型及 OPSD 生成的教师模型对比，BOPD 在 4B/8B 规模下分别提升了数学推理平均准确率约 6–7%（4B）和 3–4%（8B），在代码生成 Pass@1 上提升约 1–2%；在额外的能力评测中也保持或略有提升。实验还展示了 BOPD 在 Stage II 中的教师与学生关系更匹配（Overlap@16 等指标更好），并通过消融实验验证仅延长 OPSD 或使用 SFT 生成教师无法达到同样效果。

**⚠️ 局限性**

主要局限包括：① BOPD 的提升在进一步蒸馏或多轮自我改进时不继续累积，易出现退化；② 对不同任务/模型规模的适用性仍需更多验证；③ Stage II 仍需足够训练步数才能实现收益，训练成本相对较高；④ 仅在无参考条件下评估，若任务本身需要参考信息，效果可能受限。

---

## 27. Framing-Induced Vulnerability in Large Language Models: Evidence from Chinese News Headlines

**arXiv ID:** 2611.13531 | [PDF](https://arxiv.org/pdf/2611.13531v1)

**作者:** Mo Zhou `[一作]` (Nanjing Normal University), Yiwei Zhang `[通讯]` (Communication University of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究先对2026年中国线上新闻标题进行生态学审计，标注框架强度；随后设计对照实验，测试四款中文大型语言模型在不同标题框架下对误导性和事件严重度的评估，并验证提供文章正文是否能缓解偏差。

**💡 创新点**

创新点在于：①首次量化并公开大型中文新闻标题的框架强度数据；②在实验中系统评估四大独立模型的标题框架敏感性；③发现并量化正文上下文可显著降低模型对标题的严重度偏差。

**🔧 技术方法**

使用了大语言模型（DeepSeek‑v4‑flash、MiMo‑v2.5‑pro、Dots3‑note‑preview、LongCat‑2.0）进行推理，利用5级框架编码对标题进行人工标注，并使用标准化问卷评估模型输出。

**📊 数据集**

使用的数据集为63,926条已去重并通过有效性过滤的2026年1-9月中国新闻标题，另外构造了60个事件（30真实+30合成）及其5种标题变体；自然标题对（29对）与正文文本也被纳入实验。

**📈 对比分析**

比较方法为对每种标题变体与客观基线（H1）在误导性（T1）和严重度（T2）上进行平均对比；结果显示负向动词和情绪化数字标题显著提升误导性评分，情绪化数字标题提升严重度0.65–0.88点，正向重构标题降低严重度0.34–0.52点；提供正文后严重度偏差下降54%–80%。

**⚠️ 局限性**

局限性包括：仅覆盖中文新闻且仅为2026年模型快照；合成事件缺乏外部效度；标题变体往往同时改变多种语言特征，难以单独归因；实验未加入事件级随机效应，缺乏混合效应分析；人类评估样本量有限，无法完成统计显著性比较。

---

## 28. Deep Personalization: Continuous Context Observation for Local Proactive Agents

**arXiv ID:** 2611.13457 | [PDF](https://arxiv.org/pdf/2611.13457v1)

**作者:** Yijia Shao `[一作]` (NVIDIA), Song Han `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了 Coco，一款基于持续上下文观察、长期记忆和自适应训练的个性化主动助手，能够在不干扰用户的前提下自动判断并提供合适的干预。

**💡 创新点**

创新点在于将观察、决策与自适应闭环整合为 Observer Agent，实现了在本地设备上完成多模态推理、记忆管理与隐式反馈优化，并提出了 OSWorld‑Persona 合成基准数据集以评估主动代理的个性化效果。

**🔧 技术方法**

技术上使用多模态 LLM（Qwen‑3.5‑9B/35B、Nemotron‑3‑Nano‑Omni），LoRA 微调、Prompt Evolution、变化感知帧选择与观察优先调度，以及构建的二分图记忆结构来维护用户长时上下文。

**📊 数据集**

使用 OSWorld‑2.0 轨迹通过 VLM 生成的合成数据构建 OSWorld‑Persona（约 6,558 关键帧、3,279 交互标签），并在真实用户一周试验中收集 10 位参与者的互动反馈。

**📈 对比分析**

相较于 Vanilla、Induced Persona、Oracle Persona 等基线，Coco 的 Prompt Evolution 或 LoRA 方案在 OSWorld‑Persona 上将宏 F1 提升 2.4–3.0 倍，最高达 0.77；在真实用户试验中，宏 F1 从 0.199 提升至 0.515，说明个性化显著提升了主动干预的精准度和用户满意度。

**⚠️ 局限性**

局限性包括：依赖隐式反馈可能导致过拟合或误判；合成数据可能无法完全覆盖真实多样化使用场景；在极端干扰或多任务环境下的性能尚未充分验证；对隐私的技术保障虽然在本地，但仍需用户可控的重置与透明机制。

---

## 29. Beyond the Document: Compiled Context and Governed Agents for Any Work Product

**arXiv ID:** 2611.13055 | [PDF](https://arxiv.org/pdf/2611.13055v1)

**作者:** James Rhodes `[一作]` (AlphaBitCore), George Kang `[通讯]` (AlphaBitCore)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究将专业工作产品包装成Portable Agent Format (PAF) 包，携带文档、编译上下文、证据、治理策略等信息，供受阅者通过托管或离线阅读器进行交互；

**💡 创新点**

创新点在于将已批准的知识与结构化上下文打包为可交付单元，实现跨生产路径和阅读器的可移植性，并在问答评估中证明其显著提升；

**🔧 技术方法**

使用技术包括PAF格式定义、托管的Lens阅读器、离线0.8B Qwen3.5模型、BM25检索、预训练的选择与写作适配器，以及对包的签名与manifest验证；

**📊 数据集**

数据集为四个新生成的金融与工程领域包（Whitaker 投资提案、DFEOX 基金分析、Apple 财报 XBRL、AWS 故障后评），每包包含100个问答；

**📈 对比分析**

评估方法为对比托管阅读器在四个条件（原始聊天、规则、检索、完整包）与离线阅读器在六个条件下的回答成功率，托管完整包达96%成功，离线完整包69%，显示显著提升；

**⚠️ 局限性**

局限性包括样本仅四个包、模型与评估依赖特定技术栈、未验证签名链与离线策略、记录检索失败案例、未公开数据与代码，无法保证在其他任务或独立实现中的普适性。

---

## 30. RoboACE: Adversarial Co-Evolution for Robot Self-Improvement

**arXiv ID:** 2611.13034 | [PDF](https://arxiv.org/pdf/2611.13034v1)

**作者:** Delin Zhao `[一作]` (SIAT), Yali Wang `[通讯]` (SIAT)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种机器人自我改进框架，通过对抗共演进的课程与执行 harness 自动生成练习并改进执行软件。

**💡 创新点**

创新点在于将课程选择与执行修订耦合为对抗过程，利用执行证据持续驱动课程更新和可复用代码修订，同时保持基础模型权重不变。

**🔧 技术方法**

使用 GPT‑6.1 (Terra) 语言模型、SAM‑3 图像分割工具、预训练 VLA 控制器、以及自定义的 evidence‑reader、skill contract 等技术。

**📊 数据集**

主要使用 LIBERO 生态系统中的 200 条任务条件（LIBERO、LIBERO‑Plus、LIBERO‑PRO、LIBERO‑X），并在这些面板上进行训练与迁移测试。

**📈 对比分析**

与 OpenVLA‑OFT、π₀‑FAST、Cosmos Policy、GR00T N1.7、RPent、VoLo 等系统在四个 LIBERO‑family 评测面板上比较，最终部署在 LIBERO、PRO、X 等面板上均实现最高成功率，提升幅度最高达 35.5%。

**⚠️ 局限性**

局限性包括仅在仿真环境验证、未在真实机器人上测试、仅冻结基础模型权重、对抗进化策略需手工设定、以及缺乏跨环境泛化的实证验证。

---

## 31. PneuGen: Task-Driven Agentic Co-Design of Soft Pneumatic Robots

**arXiv ID:** 2611.13344 | [PDF](https://arxiv.org/pdf/2611.13344v1)

**作者:** Shaohong Zhong `[一作]` (University of Oxford), Perla Maiolino `[通讯]` (University of Oxford)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 PneuGen 框架，利用 LLM 设计师与评审代理通过迭代循环完成软体机器人几何与气动参数的任务驱动共设计。

**💡 创新点**

创新点在于将 LLM 代理的设计与批评角色结合、引入模块响应模型做物理先验、以及自动化从设计程序到 CAD‑FEM 的完整流水线，显著提升搜索效率与设计质量。

**🔧 技术方法**

使用的技术包括 Claude Opus 5.5 LLM 代理、模块压强‑变形响应模型、CadQuery+Gmsh 自动 CAD 与网格生成、SOFA 软体仿真平台以及与基线算法的对比实验。

**📊 数据集**

使用的数据集为仿真任务中的 18 个抓取目标（9 个原始形状 + 9 个 YCB 物体）和多条形状匹配目标曲线，未采用公开大规模数据集。

**📈 对比分析**

在 6 轮仿真评估预算下，与随机搜索、遗传算法 GA 以及 CMA‑ES 进行对比，PneuGen 在抓取成功率和形状匹配得分上均显著优于基线，并且所需 FEM 评估次数更少。

**⚠️ 局限性**

限制主要体现在组件库范围有限、LLM 代理对复杂力学预测仍需依赖先验模型、仿真与真实差异未完全消除以及对更大规模任务的可扩展性尚待验证。

---

## 32. Trust, but verify, audit and attribute: a human-in-the-loop framework for checking AI-produced research results

**arXiv ID:** 2611.13042 | [PDF](https://arxiv.org/pdf/2611.13042v1)

**作者:** György Eigner `[一作]` (Obuda University), Roland Hollós `[通讯]` (HUN-REN Centre for Agricultural Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一套人机协作框架，用于验证人工智能生成的科研成果，包含验证工具、忠实度审核、人类决策节点和对话报告；

**💡 创新点**

核心创新是将“结果可信度”分为工具可靠性与语义忠实度两层，设计了四个决策点保障人类干预，并引入可追溯的对话报告记录贡献；

**🔧 技术方法**

技术上结合了多源验证工具（Lean、Wolfram、MATLAB、Python SciPy、R）、MCP服务器接口、自动化脚本、语义忠实度审核逻辑以及对话报告生成器；

**📊 数据集**

使用了2026年10月公开的719份AI生成数学手稿的示例数据，和统计分析示例的公开数据集；

**📈 对比分析**

通过两条示例工具链（数学与统计）展示，工具链可在不干预人类的情况下完成自动化检查，性能上在短小例子中工具运行时间均在数十秒内，且能检测出歧义、错误和一致性问题；

**⚠️ 局限性**

局限包括：对默认读音的依赖可能导致与作者意图不符；同一AI模型生成的语义翻译可能出现一致性误判；实验仅在两条示例中验证，未进行大规模评估；无法检测模型对真实系统的适配度。

---

## 33. FlowPlan: Reducing the Token Cost of Coding Agents with Abstraction-Guided Planning and Verification

**arXiv ID:** 2611.13241 | [PDF](https://arxiv.org/pdf/2611.13241v1)

**作者:** Melika Sepidband `[一作]` (York University), Hadi Hemmati `[通讯]` (York University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了FlowPlan，一种利用函数级控制流图（CFG）和数据流图（DFG）进行定位、规划与验证的抽象驱动修复框架；

**💡 创新点**

通过在编辑前将修复意图明确为图增量，并在编辑后通过图变更验证，形成机器可检验的完成判据，从而在不降低成功率的前提下显著降低令牌消耗；

**🔧 技术方法**

结合LLM编码代理、Model Context Protocol（MCP）扩展工具、CFG/DFG抽象、四阶段修复策略以及图增量规划与验证技术；

**📊 数据集**

使用SWE‑bench Pro、SWE‑bench Verified（Python）和SecBench（C/C++）三个公开基准集；

**📈 对比分析**

在与基线代理以及五个复现的令牌压缩/交互优化方法（SHERLOC、AgentDiet、SeeRepo、EET、No‑execution）在相同实例、模型和计费方案下进行对比，FlowPlan在三种模型-基准组合中获得最大的令牌节省（与基线相比降低至约0.46‑0.47倍或0.72‑0.76倍），在另一个组合中与最佳方法持平，整体成功率与基线无显著差异；

**⚠️ 局限性**

仅在Claude 3.5 Turbo与Sonnet 5两个模型、三大基准上测试；图抽象可能遗漏跨函数或语义信息；方法需要适配多语言和不同代理，复现可能受实现细节影响；实验为一次性运行，未覆盖模型随机性和更广泛任务。

---

## 34. Towards Scalable Network-Based Interventions for HIV Treatment and Prevention via R2C-Agent

**arXiv ID:** 2611.13312 | [PDF](https://arxiv.org/pdf/2611.13312v1)

**作者:** Akseli Kangaslahti `[一作]` (Harvard University), Cheryl Johnson `[通讯]` (World Health Organization)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计了一种基于同伴推荐网络的网络干预决策框架R2C-Agent，利用新提出的条件图扩散模型R2C-Diff从可观测的同伴推荐网络推断隐藏的接触网络，并在此基础上规划HIV感染防控干预。

**💡 创新点**

创新点包括：①将同伴推荐过程建模为接触网络的逐步破坏，使用R2C-Diff学习逆扩散过程；②在决策时采样多种可能的接触网络而非单一预测，从而有效处理网络不确定性；③在仅有少量标注数据的情况下，利用前向腐蚀过程和教师强迫实现高效训练。

**🔧 技术方法**

核心技术包括：条件图扩散模型R2C-Diff（基于图神经网络的消息传递与双通道嵌入）；干预规划算法CAST与SandIMIN；独立级联（IC）传播模型和Monte Carlo评估；以及与同伴推荐网络相结合的采样与决策聚合。

**📊 数据集**

使用ICPSR公开的真实人群联合社会与接触网络（1386节点、3793条边，1381条为接触边），通过模拟同伴推荐过程生成对应的同伴推荐网络。

**📈 对比分析**

与多种基线（无干预、随机、最高度、成对风险分类、Identity、Omniscient）在强力治疗和PrEP两种干预任务上进行比较。R2C-Agent在两项任务中分别在总遗憾上降低约54.9%和28.8%，并在所有预算区间均显著优于其他预测方法；实验亦验证其对α扰动和采样数量的鲁棒性。

**⚠️ 局限性**

局限性包括：需要同伴推荐网络与真实接触网络在数据分布上匹配；假设同伴推荐遵循所有边的采样规则；在极端预算或高误差情况下性能仍会下降；模型未直接针对决策目标进行优化，泛化到不同人群或网络结构的能力尚待验证。

---

## 35. Selective Surgical Action-Center Localization:Retaining Localization Gains through Supervised Scoring

**arXiv ID:** 2611.12977 | [PDF](https://arxiv.org/pdf/2611.12977v1)

**作者:** Yanchong Xue `[一作]` (Harbin Engineering University), Guochao Zhang `[通讯]` (Hainan Cancer Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种结合坐标监督与热图监督的定位器，并配备了学习成功评分器，实现手术动作中心的选择性定位。

**💡 创新点**

创新点在于：①将坐标监督与热图监督通过空间期望读取相结合，提升定位精度；②构建基于报告性与定位误差联合判定的成功标签，训练专门的Logistic回归评分器；③通过匹配比较分离候选供应与选择损失，清晰阐明选择性评分对最终输出收益的影响。

**🔧 技术方法**

使用的技术包括：ResNet34编码器+U‑Net解码器的热图+坐标监督定位器；空间期望读取方法；Logistic回归学习成功评分；熵与学习评分的选择规则；交叉熵、L2正则、AdamW、cosine学习率调度等训练细节。

**📊 数据集**

使用了COSA 2026数据集，包含5,000帧来自50个来源的手术视频，其中有POINT（可报告）与NA（不可报告）标注。

**📈 对比分析**

通过与基线B+S（热图MSE监督+argmax读取）在相同评分器和80%接受配额、0.10容差下的匹配比较，C+S在每100帧中取得46.08个正确输出，较基线提升1.88；在不同容差、配额、阈值设置下也表现出学习评分优于熵选择，且整体接受风险保持在约42%。

**⚠️ 局限性**

局限性包括：仅在单一数据集上验证；对不同来源的覆盖率差异较大，需要手动设定阈值和配额；未考虑时序信息或多摄像头场景；模型对未知来源的泛化能力尚未系统评估。

---

## 36. Understanding-First AI Engineering: Purposes, Reference Architecture, and Architectural Patterns for Human Understanding

**arXiv ID:** 2611.13022 | [PDF](https://arxiv.org/pdf/2611.13022v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 37. Seeing the Small: Deformable Duality Mamba for Retinal Lesion Segmentation

**arXiv ID:** 2611.13514 | [PDF](https://arxiv.org/pdf/2611.13514v1)

**作者:** Chang Yu `[一作]`, Alper Yilmaz `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种名为 D^2Mamba 的网络，用于糖尿病视网膜病变（DR）图像中的多类病灶分割。

**💡 创新点**

创新点在于将非因果双状态空间（NC‑SSD）与形变感知 Mamba 扫描（DMSM）相结合，既能在任意扫描顺序下聚合全局上下文，又能自适应地对小型、不规则病灶进行采样，显著提升微血管瘤等细小病灶的检测效果。

**🔧 技术方法**

核心技术包括：SDM（State‑Dual‑Modulation）块、NC‑SSD 全局状态聚合、DMSM 可形变扫描、多尺度 CCLR 解码器以及双分支融合机制。

**📊 数据集**

使用公开的 DDR（Diabetic Retinopathy Dataset）数据集，包含 757 张已标注的视网膜图像，进行五折交叉验证。

**📈 对比分析**

与十个重新实现的基线模型（如 U‑PerNet、Swin‑Unet、MambaLiteUNet 等）在同一训练协议下比较。D^2Mamba 在 mIoU 上达到 33.87%（比最强基线高 2.99 点），mDice 为 49.62%（比 Swin‑Unet 高 4.05 点），mAUPR 为 48.23%（比 DNL 高 1.25 点），在微血管瘤类表现尤为突出。

**⚠️ 局限性**

局限性包括：形变扫描的融合比例固定，可能对大面积扩散性出血病灶不够友好；未在其他视网膜数据集上验证泛化能力；模型对不同病灶尺度的自适应仍有改进空间。

---

## 38. Evidence-Grounded Iterative Reasoning for the Diagnosis of Diffuse Cystic Lung Diseases

**arXiv ID:** 2611.12981 | [PDF](https://arxiv.org/pdf/2611.12981v1)

**作者:** Haoqing Li `[一作]` (University of Science and Technology of China), Xiaowen Hu `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研发了 DCLDs‑EIR 框架，用于稀有肺囊性疾病（BHD、LAM、PLCH、LIP）的诊断，融合多视角空间感知、目标化 CT 检查与检索增强的知识验证；

**💡 创新点**

通过迭代证据获取与假设验证的闭环，将图谱检索结果与局部影像观察显式对比，实现可追溯诊断解释，并在无任务专门微调的情况下取得高准确率；

**🔧 技术方法**

使用多模态大型语言模型（MLLM）、nnU-Net 分割、三维渲染、知识图谱检索与迭代自我验证、提示工程等技术；

**📊 数据集**

采用 224 例 CT 检查（内部 144 例、外部 80 例），覆盖 BHD、LAM、PLCH、LIP 四种病种，利用真实临床影像与病理/指南诊断标签；

**📈 对比分析**

与多种 3D 经典分类器、医学 MLLM 及 2D RAG 方案对比，GPT‑5.6 Sol 版本在外部队列中实现 90.00% 准确率、89.05% macro‑F1，显著优于基线；

**⚠️ 局限性**

依赖预训练分割模型与知识图谱完整性，推理时间与成本较高，且仅在回顾性多中心数据上验证，尚需前瞻性评估与临床集成。

---

## 39. Free Lunch Uncertainty: Sequence-Level Rényi Entropy from Particle Decoding for Adaptive LLM Reasoning

**arXiv ID:** 2611.13414 | [PDF](https://arxiv.org/pdf/2611.13414v1)

**作者:** Erfan Baghaei Potraghloo `[一作]` (University of Southern California), Vijay Gupta `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在LLM推理中利用Sequential Monte Carlo (SMC) 采样已估计的归一化常数直接算出序列级Rényi熵 (SRE)，并用其零成本不确定性信号进行选择性预测与自适应粒子分配

**💡 创新点**

首次将SMC归一化常数估计与Rényi熵链式规则结合，推出一族逐步熵，并证明其1/2阶成员在多种设置下优于传统日志概率

**🔧 技术方法**

采用Power-SMC 采样、Rényi熵链式规则、零成本不确定性读出与两步自适应粒子策略

**📊 数据集**

在MATH500与GSM8K数学推理基准上测试四个开源LLM（Qwen2.5-Math-1.5B/7B、Qwen2.5-7B、Llama-3.1-8B-Instruct）

**📈 对比分析**

与平均token熵、序列日志概率、自一致性等基线对比，SRE-1/2在10/13设置中AUROC最高，可在MATH500上实现与N=64相同准确率但平均34.2%粒子节省；在自适应粒子分配时，阈值化SRE比随机分配显著提升准确率

**⚠️ 局限性**

仅在SMC粒子集上定义，未评估单个采样、开放式生成或多轮交互任务；对不同任务类型与模型迁移的适用性仍待验证

---

## 40. An Analytical Model of Compute-limited Multistage Training Pipelines

**arXiv ID:** 2611.13276 | [PDF](https://arxiv.org/pdf/2611.13276v1)

**作者:** Nishil Patel `[一作]` (Gatsby Computational Neuroscience Unit), Andrew Saxe `[通讯]` (Gatsby Computational Neuroscience Unit)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并分析了一个高维可解析模型，用以研究预训练、监督微调（SFT）与强化学习（RL）在基础模型训练中的相互作用和计算分配。

**💡 创新点**

创新点包括：①将随机矩阵理论与算子投影闭包相结合，实现学习动力学的低维近似；②发现预训练样本量对Bayes误差的α^(-1/2)缩放；③定量描述SFT热启动和RL奖励稀疏性对学习曲线的影响；④在三阶段训练中求解最优计算预算分配；⑤扩展模型至组合式RL，揭示任务相关技能的放大与非相关技能的保持。

**🔧 技术方法**

使用随机矩阵理论、算子投影闭包、常微分方程分析、教师-学生框架、光滑化学习率与球面约束、计算预算公式以及数值模拟验证。

**📊 数据集**

主要采用合成高斯分布数据（带信号/噪声子空间）作为预训练与微调的输入；在实验演示中引用了OLMo‑2 1B 在 GSM8K 数据集上的结果。

**📈 对比分析**

通过与完整更新规则的蒙特卡洛仿真对比，证明解析ODE预测与模拟轨迹高度吻合；评估了不同阶段的误差缩放、SFT对RL平台期的消除以及计算预算优化，结果表明完整的三阶段管线在整个计算受限范围内始终优于仅 SFT 或仅 RL。

**⚠️ 局限性**

局限性在于：模型仅为单层简化表示，假设编码器冻结且观测独立；未考虑真实数据标注成本、复杂模型架构与工程细节；RL模型仅捕捉批量反馈，忽略序列内相关性。

---

## 41. Audited Gradients, Anchored Policies: Model-Based Reinforcement Learning for Efficient Flow Control

**arXiv ID:** 2611.13203 | [PDF](https://arxiv.org/pdf/2611.13203v1)

**作者:** Ruige Kong `[一作]` (University of Washington), Stefania Fresca `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于可微代理的路径优化控制框架，核心是对代理动作梯度进行审计，并将通过审计得到的可靠梯度用于构建低维反馈法则和锚定惩罚，最终训练出高效的流体控制策略。

**💡 创新点**

创新点在于：①引入动作梯度审计，将代理梯度与高保真求解器的中心差分梯度在预定阈值下进行比较；②利用通过审计获得的可靠梯度拟合低维反馈法则，并在训练过程中用锚定惩罚保持策略在该法则附近；③在梯度不可信时直接使用高保真梯度构建控制器，从而避免代理梯度误差导致的性能下降。

**🔧 技术方法**

技术实现包括：可微代理（如隐变量场模型、傅里叶神经算子等）用于预测目标；中心差分求解器获取动作梯度；路径优化（PPO/TD3）结合锚定惩罚进行策略学习；低维法则拟合与投影；以及对高保真仿真成本的严格计量。

**📊 数据集**

数据集：四个流体控制案例—旋转圆柱流（Re=100）、三圆柱群（Re=30）、受限二维湍流（Re=200）以及参数化卡莫托–西瓦辛斯基方程（L=22），所有案例均使用高保真CFD/谱方法产生的轨迹数据。

**📈 对比分析**

与基于模型自由训练（PPO/TD3）的对比：在每个案例中，基于代理的控制器在同等初始条件下均实现了相同或更优的目标函数改进，同时高保真求解器交互次数比模型自由方法减少了3至156倍，表明方法既有效又节省计算成本。

**⚠️ 局限性**

局限性：①审计统计在梯度几乎恒定的场景下不具信息量；②目前仅在二维/一维流动上验证，三维扩展尚未探讨；③仍需高保真求解器计算梯度，导致一定的前期成本；④阈值和锚定权重需要经验设定，缺乏统一理论指导。

---

## 42. IC-GNN: Modeling Individualized Brain Community Organization for Disease Prediction

**arXiv ID:** 2611.13070 | [PDF](https://arxiv.org/pdf/2611.13070v1)

**作者:** Yang Zhang `[一作]` (Yale University), for the Alzheimer's Disease Neuroimaging Initiative `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本文提出一种以个体化社区为核心的图神经网络IC-GNN，用于基于静息态fMRI预测脑疾病并可解释社区结构。

**💡 创新点**

创新点在于：①在预定义脑区层级上通过Affinity Propagation递归生成多尺度个体化社区；②用概率边缘精炼模块利用社区信息去噪并增强重要连接；③采用社区引导读取，将共享的ROI权重动态适配每个个体的社区上下文，实现个体化预测与群体可比性的统一。

**🔧 技术方法**

技术方法包括Affinity Propagation层次池化、概率边缘精炼（PERM）、图卷积网络（GCN）、社区引导读出、以及交叉熵与KL正则化的联合训练。

**📊 数据集**

使用的公开数据集有：ABIDE（自闭症谱系障碍）、ADNI（阿尔茨海默病）和UK Biobank（抑郁症），共计约30,000人次。

**📈 对比分析**

与15种基线方法（传统机器学习、通用GNN、脑网络模型）在三大数据集上对比，IC-GNN在AUROC和准确率上均表现最优：ABIDE AUROC 75.41%、准确率 69.40%；ADNI AUROC 86.46%、准确率 80.52%；UK Biobank AUROC 60.62%、准确率 62.51%。

**⚠️ 局限性**

局限性包括：①对扫描质量和多中心差异敏感；②社区结构的生成依赖于Affinity Propagation的超参数，可能影响重现性；③模型在极端低质量或非标准分割的脑图谱上尚未充分验证。

---

## 43. Rethinking Evaluation for Video Capsule Endoscopy: Validation Protocol, Source-Video Diversity, and Cross-Dataset Shift

**arXiv ID:** 2611.13245 | [PDF](https://arxiv.org/pdf/2611.13245v1)

**作者:** Smriti Regmi `[一作]` (University of South Dakota), Debesh Jha `[通讯]` (University of South Dakota)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种用于视频胶囊内镜（VCE）图像分类的可变形多尺度边缘引导注意网络（DMSNet），并在多种架构下进行系统评估；

**💡 创新点**

创新点在于融合自适应多尺度特征融合、可变形病灶细化、边缘引导注意与稀疏特征选择，显著提升在源视频无交叉验证下的泛化性能；

**🔧 技术方法**

使用了EfficientNetV2‑S骨干、可变形卷积、注意力机制、稀疏门控等深度学习技术，并采用多任务损失（焦点损失+稀疏正则）训练；

**📊 数据集**

主要使用Kvasir‑Capsule数据集（11类，约47k帧）进行训练与评估，并在独立的CV2024数据集（8类）进行跨数据集泛化测试；

**📈 对比分析**

通过随机帧级划分和源视频无交叉划分两种验证策略，对19种代表性CNN、Transformer及预训练视觉模型进行比较；在随机帧级验证下DMSNet在Kvasir‑Capsule测试集上MCC最高为0.442；在源视频无交叉验证下MaxViT表现最佳；在外部CV2024上所有模型均显著下降，最高MCC仅0.207；

**⚠️ 局限性**

局限性包括：源视频分布不均导致验证/多样性实验受限；DMSNet在跨数据集时泛化不足；仅进行帧级分类，未利用视频序列时序信息；外部评估仅覆盖共享类别，未全面评估。

---

## 44. Branch-and-Commit for Frozen RL Agents: When and Where to Spend Test-Time Compute

**arXiv ID:** 2611.13003 | [PDF](https://arxiv.org/pdf/2611.13003v1)

**作者:** Sahar Admoni `[一作]` (Technion Israel Institute Of Technology), Ofra Amir `[通讯]` (Technion Israel Institute Of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

未提供论文具体内容，无法判断作者所做的具体工作。

**💡 创新点**

无法确定创新点，因为缺乏论文细节。

**🔧 技术方法**

缺少技术实现信息，无法说明使用了哪些技术。

**📊 数据集**

没有提及数据集，无法判断所使用的数据来源。

**📈 对比分析**

未给出实验或比较方法，无法评价性能表现。

**⚠️ 局限性**

由于信息不足，无法指出研究的局限性。

---

## 45. Metamorphic testing of learned AC optimal power flow surrogates reveals representation dependence

**arXiv ID:** 2611.13566 | [PDF](https://arxiv.org/pdf/2611.13566v1)

**作者:** Stephen Peng `[一作]` (Boston Consulting Group), Vineet Jagadeesan Nair `[通讯]` (Boston Consulting Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过13种AC-OPF变换的形态学测试，评估学习式AC-OPF替代模型在不同序列化编码下的表示一致性；

**💡 创新点**

创新点在于构建完整的变换目录、实现精确输出映射和可采样求解器控制，并在17个模型配置（含两份公开检查点）中系统定位表示相关缺陷；

**🔧 技术方法**

主要技术包括形态学测试框架、随机初始化对照、训练增量数据增强、张量中间干预定位错误、GPU/CPU特征算子对比；

**📊 数据集**

使用了N‑1拓扑扰动电网数据集（Case14、Case118、Case500、Case2000、Case4661）以及公开的GridSFM‑Open与LUMINA 2M模型；

**📈 对比分析**

对比方法是将每个物理案例在多种合法序列化下执行模型，计算变换前后预测差异并与阈值对照；结果显示所有配置至少违反六项关系，训练后即使错误下降，预测仍不满足AC约束；

**⚠️ 局限性**

局限性包括：只覆盖13种变换，样本范围有限；仅测试两份公开检查点，未覆盖闭源模型；增量训练未显著提升物理可行性；缺乏全局最优性保证与对更大规模网格的泛化验证。

---

## 46. Closed-Form Gaussian Information Projection for Embedded Simplices and Simplicial Mixtures

**arXiv ID:** 2611.13354 | [PDF](https://arxiv.org/pdf/2611.13354v1)

**作者:** Ali Darijani `[一作]` (Fraunhofer IOSB of Fraunhofer-Gesellschaft), Jürgen Beyerer `[通讯]` (Fraunhofer IOSB of Fraunhofer-Gesellschaft)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了嵌入单纯形上的Dirichlet-Gaussian概率密度函数的高斯近似，推导了非退化环境高斯概率密度函数的唯一信息投影，并获得了任意正Dirichlet参数的最优高斯和最小散度的封闭形式表达。

**💡 创新点**

创新点在于推导出信息投影散度的封闭形式，并证明其对仿射同构的不变性，独立于环境维度和法向概率密度，同时建立了基于内在维度的有限高斯混合的明确上界。

**🔧 技术方法**

使用了高斯信息投影、Kullback-Leibler散度和几何测度理论等数学技术。

**📊 数据集**

使用了与嵌入单纯形相关的Dirichlet概率密度函数和高斯混合模型进行实验，具体数据集未明确提及。

**📈 对比分析**

通过与现有方法的比较，证明了所提出方法在高斯混合模型的散度上具有明确的上界，且该上界独立于环境维度和单纯形的几何特性。

**⚠️ 局限性**

限制在于该方法依赖于高斯混合模型的假设，且在处理复杂几何形状时可能需要更多的计算资源。

---

## 47. Context Types: Bunched Modalities for Unifying Safety and Reachability

**arXiv ID:** 2611.13471 | [PDF](https://arxiv.org/pdf/2611.13471v1)

**作者:** Zhe Zhou `[一作]` (Peking University), Suresh Jagannathan `[通讯]` (Purdue University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出一种既能验证程序安全性又能验证可达性属性的 refinement type system，并通过将类型上下文视为资源来统一两类属性的处理

**💡 创新点**

创新点在于将类型上下文建模为资源，构建资源代数和 bunched modal 逻辑，以实现安全性与可达性在同一类型判定中的统一

**🔧 技术方法**

使用资源代数、bunched modal 逻辑、类型推导规则和定域语义实现和证明该类型系统的安全性

**📊 数据集**

本文无实验数据集，主要基于理论形式化与 Coq 机械化证明

**📈 对比分析**

通过与传统只验证安全性的类型系统对比，证明本系统在理论上对安全性与可达性都能给出正确保证；未进行性能或实测比较

**⚠️ 局限性**

主要局限在实现复杂度较高，尚未在大规模实际程序上验证，缺乏实证评估与性能评测

---

## 48. Activation-Guided Graph Retrieval for Cognitive Memory Reconstruction in Language Model Systems

**arXiv ID:** 2611.13127 | [PDF](https://arxiv.org/pdf/2611.13127v1)

**作者:** Marvin Danig `[一作]` `[通讯]` (Achiral Research), Marvin Danig (Achiral Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种激活导向图检索（AGGR）架构，通过激活控制记忆可用性、递归提示生成以及证据图重建等机制，提升长期 AI 系统的检索与推理能力。

**💡 创新点**

创新点在于将查询相关性与激活条件分离，设计了粉丝衰减的图遍历、递归提示机制、授权优先的记忆控制和以证据图为输出的检索结果。

**🔧 技术方法**

使用的技术包括图检索、基于激活的评分与传播、粉丝衰减（degree-aware attenuation）、递归提示生成、证据图重建、授权前置、以及基于结果的强化学习更新。

**📊 数据集**

论文未提供具体数据集，而是提出了一套可复现的评估协议，供后续实验使用。

**📈 对比分析**

与六个基线（词检索、密集检索、无激活图检索、固定池激活重排序、激活导向遍历、写时合并笔记）进行匹配比较；论文未给出具体性能数值，仅阐述预期的优势与评估指标。

**⚠️ 局限性**

局限性包括：缺乏实验验证、激活参数未校准、图结构标签错误可能导致错误因果链、热门锁定、过度衰减、目标设置错误、权限泄露以及证据图规模过大导致审计难度。

---

## 49. SetPrint: Black-Box LLM Fingerprinting via Setwise Correspondence of Response Changes

**arXiv ID:** 2611.12950 | [PDF](https://arxiv.org/pdf/2611.12950v1)

**作者:** Jinmo Han `[一作]` (Seoul National University), Nam Soo Kim `[通讯]` (Seoul National University)

**通讯引用:** 4442 | [OpenAlex ID](https://openalex.org/A5051356511)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无训练、仅使用黑盒查询的LLM指纹识别方法SetPrint，利用相同扰动查询产生的响应变化集合进行模型归属判断。

**💡 创新点**

创新点在于同时捕捉集合内的响应变化关系（within‑set relations）和集合间对应扰动的响应变化匹配（between‑set correspondence），通过对每个响应变化减去集合均值并对对应变化做内积来综合两种信息。

**🔧 技术方法**

采用固定文本编码器（all‑mpnet‑base‑v2）对多次采样的响应求平均得到嵌入，计算响应变化并归一化，随后对集合内中心化后的变化做点积并减去集合均值交叉点积，形成最终得分。

**📊 数据集**

使用LeaFBench基准数据集，其中包含13个源模型与149个通过微调、量化或解码设置改变得到的目标模型，以及HumanEval的前20个词作为基准查询。

**📈 对比分析**

与白盒方法REEF、监督黑盒方法LLMmap/LLMDNA+SVM以及无监督黑盒基线MET/SEF/ZeroPrint比较，SetPrint在AUROC、TPR@1%FPR和pAUC上均取得最高分，显示其在无训练黑盒指纹识别中的优越性能。

**⚠️ 局限性**

局限性包括仅在LeaFBench上验证，无法保证在更大规模或不同语言的模型上同样有效；方法依赖固定编码器和多次采样，可能受响应噪声或生成策略变化影响；未探索更复杂的非线性相似度或主动查询策略的潜在改进。

---

## 50. Min-CSPs on Complete Instances III: Nearly Optimal Characterizations of Approximation for Symmetric CSPs

**arXiv ID:** 2611.13635 | [PDF](https://arxiv.org/pdf/2611.13635v1)

**作者:** Yihwan Chae `[一作]` (Seoul National University), Euiwoong Lee `[通讯]` (University of Michigan)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

研究完整的 Min-k-CSP，给出对对称布尔谓词的近似算法和硬度结果，提出两阶段框架实现 QPTAS（w(A)≤2）和 O(n^{w(A)-3}) 近似（w(A)≥3）

**💡 创新点**

首次用直接乘积测试和多项式优化等工具统一处理完整 Min-CSP，揭示了宽度参数 w(A) 与可近似性之间的几乎完全对应关系

**🔧 技术方法**

两阶段框架：第一阶段利用直接乘积测试和 VC 维度构造“close-list”；第二阶段对 w(A)≥3 的情况通过约简到加权 Min‑2‑SAT；w(A)≤2 时转化为在截断单纯形上的二次多项式优化；并使用基于 Carathéodory 的算法实现 QPTAS

**📊 数据集**

无具体实验数据集，全部为理论证明与算法复杂度分析

**📈 对比分析**

与已有的 O(1) 或 O(n) 近似相比，QPTAS 在 w(A)=2 时显著改进，且对 w(A)≥3 的近似因子几乎匹配硬度下界；实验未给出，主要通过理论证明展示性能提升

**⚠️ 局限性**

对于 w(A)=2 仍仅有 QPTAS，未能实现 PTAS；w(A)≥3 的近似因子仍远高于理想，尤其在 n 较大时近似比可能很差；算法仅适用于对称布尔谓词，非对称或多值情况未覆盖

---

## 51. Do AI Agents need Preventive Care? An Epidemiologic Cohort Design for Agentic AI

**arXiv ID:** 2611.13333 | [PDF](https://arxiv.org/pdf/2611.13333v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 52. Predictive Surprise as Self-Grounding Concept Bottleneck for Interpretable Time Series

**arXiv ID:** 2611.12955 | [PDF](https://arxiv.org/pdf/2611.12955v1)

**作者:** Sachith Abeywickrama `[一作]` (Nanyang Technological University), Chau Yuen `[通讯]` (Nanyang Technological University)

**通讯引用:** 55916 | [OpenAlex ID](https://openalex.org/A5060020877)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于预测惊讶的概念瓶颈框架，用单一自监督信号完成时间序列的分段、概念构建和分类，实现准确、可解释与可弃置三重目标。

**💡 创新点**

创新点在于将预测惊讶同时用作分段边界判定和段落特征描述，彻底消除分段与概念循环依赖，并通过无标签数据自我学习概念词典，无需人工标注或语言模型。

**🔧 技术方法**

技术核心包括冻结的自回归高斯密度模型（ARGDM）预测下一个时刻分布、长度约束动态规划分段、基于预测统计的签名特征、GMM非参数概念词典以及轻量级Transformer分类器；同一套距离度量可作为置信度与置信度阈值。

**📊 数据集**

实验涵盖17个公开基准，包括 UEA archive、UCI HAR、Epilepsy、Sleep‑EDF 以及四个故障诊断数据集 FD‑A~FD‑D，验证跨领域与少量标签场景下的鲁棒性。

**📈 对比分析**

与黑盒、可解释与自适应补丁方法对比，本文在大多数数据集保持与黑盒相当或更优的准确率；在少标签（1%）和跨域迁移中优于所有黑盒基线；同时在 OOD 检测和删除可解释性评估中与 softmax、原型距离和后置解释器持平或领先。

**⚠️ 局限性**

局限性包括预设的概念词典大小、概念含义需要人工解释、缺乏层次化词典以及对实时信号变化的适应能力不足。

---

## 53. 51 Additions per Recursive Step for Rank-23 $3\times3$ Matrix Multiplication, Machine-Checked over Arbitrary Rings

**arXiv ID:** 2611.12907 | [PDF](https://arxiv.org/pdf/2611.12907v1)

**作者:** Greg Sidebottom `[一作]` `[通讯]`, Greg Sidebottom

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个秩23的3×3矩阵乘法二线性方案，在可变基准模型下每一步仅需51个加法，优于此前最佳的52。

**💡 创新点**

利用精确的自由基最小化、完整数据库扫描和Lean 4形式化验证，证明了51是该类方案的绝对最优，并揭示了二进制与整数符号模型之间的“符号缺口”现象。

**🔧 技术方法**

采用可变基准成本模型、Tellegen转置原理、精确自由基搜索、SAT签名枚举、Lean 4机器检验以及对Heule–Kauers–Seidl数据库的完整遍历等技术。

**📊 数据集**

使用Heule–Kauers–Seidl 17,376个秩23方案数据库，以及自建的四个51类及其签名/转换数据。

**📈 对比分析**

通过将每步加法数与传统平面模型记录（55）和已知最佳类（52）对比，并借助精确最小化与数据库扫描验证，发现四个不同类均实现51，加法数比平面模型低4，接近理论下限。

**⚠️ 局限性**

尚未解决整数域下最优值是否为52或53，是否存在更低（50）步加法方案；符号模型的可扩展性有限；对更高阶矩阵乘法（4×4）尚未全面验证。

---

## 54. Earth Embeddings in Environmental Remote Sensing

**arXiv ID:** 2611.13612 | [PDF](https://arxiv.org/pdf/2611.13612v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 55. Towards Accessible, Autonomous Photosynthesis Monitoring on Living Leaves

**arXiv ID:** 2611.13387 | [PDF](https://arxiv.org/pdf/2611.13387v1)

**作者:** Qitong Li `[一作]` (Northwestern University), Nivedita Arora `[通讯]` (Northwestern University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文设计并实现了LeafBreath，一种低成本、自动闭合的叶片封闭腔室，用于在多天内无人工干预地连续测量单片叶的净光合速率A_n。

**💡 创新点**

创新点在于将低成本NDIR CO₂传感器与可热成型、3D打印的可适配夹具相结合，形成无需泵管、无需外部气体线的自闭合测量系统，并实现了对多片叶片的并行长期监测。

**🔧 技术方法**

采用Senseair S88 GH NDIR传感器、Arduino/ESP32控制板、伺服驱动的闭合机件、混合风扇、湿度/温度校正以及泄漏补偿等技术，构建了完整的测量与数据处理流程。

**📊 数据集**

通过与CIRAS-4参考设备进行29次A–B–A实验、对C₃与CAM植物的多日监测、不同叶形/尺寸的适配测试，以及光响应曲线实验，构建了多场景的数据集。

**📈 对比分析**

与CIRAS-4比较时，LeafBreath在A_n测量上取得R²=0.90、RMSE≈0.78 µmol m⁻² s⁻¹的精度，并能够在同一植物上分辨不同叶片的昼夜光合模式，展示了良好的性能与可行性。

**⚠️ 局限性**

局限性包括无法准确测量蒸腾、只捕获下表面气孔的光合速率、对环境CO₂波动敏感、需要人工调整叶片位置以及长期部署的可靠性尚未完全验证。

---

## 56. You Are What You Read: Indian Political News Media Bias Dataset for Hindi Language

**arXiv ID:** 2611.13686 | [PDF](https://arxiv.org/pdf/2611.13686v1)

**作者:** Kushal Trivedi `[一作]` (Atal Bihari Vajpayee Indian Institute of Information Technology), Roshni Chakraborty `[通讯]` (Atal Bihari Vajpayee Indian Institute of Information Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并发布了包含 448 篇来自四家印度主要印地语新闻机构的新闻文章的多注释、可解释的媒体偏见数据集，涵盖点击诱导、选择、框架和遗漏四类偏见，并附有人类注释者的理由说明。

**💡 创新点**

首次在印地语新闻领域提供高质量、可解释的多标签偏见标注数据，并通过三位专家的多次标注与一致性分析验证标注可靠性，填补了低资源语言偏见研究的空白。

**🔧 技术方法**

主要使用手工标注方法，配合详细的规则手册和三轮审议以达成一致；同时对标注结果进行统计、Kappa 互评和表面语言特征分析，探索偏见与句子多样性等特征的关联。

**📊 数据集**

采用 448 篇印地语政治新闻文章，来源于 Aaj Tak、Live Hindustan、Dainik Bhaskar 和 Jagran 四大新闻机构，注释覆盖点击诱导、选择、框架、遗漏四类偏见。

**📈 对比分析**

通过对比人工标注与大型语言模型（LLM）的预测，发现 LLM 在识别非词汇驱动偏见（选择、框架、遗漏）时存在显著错误，且仅在点击诱导类能利用词汇丰富度区分；说明现有模型仍难以捕捉情境性偏见。

**⚠️ 局限性**

局限性包括：数据规模相对有限，仅覆盖四类偏见，缺乏种族、性别、地区等人口统计学维度的偏见；只针对印地语政治新闻，难以推广至其他语言或非政治领域；依赖人工注释，标注成本高且仍存主观性。

---

## 57. A Multi-Word Tokenizer for 204 Languages That Needs Fewer Tokens Than GPT-4o's, Including on English

**arXiv ID:** 2611.13059 | [PDF](https://arxiv.org/pdf/2611.13059v1)

**作者:** Johin Johny `[一作]` (Havon AI), Shervin Sam `[通讯]` (Havon AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文训练并评估了覆盖约200种语言的多语言分词器，提出了多词合并阶段的 Havon v4 和 v5，并与 GPT‑4o 及公开 tokenizer 进行 token 数量比较。

**💡 创新点**

其创新点在于结合字节级 BPE、专门的前置分词规则、语言预算分配与 SuperBPE 的多词合并，显著降低印度语言及英语的 token 需求，并在相同词表大小下实现更低 token 消耗。

**🔧 技术方法**

技术上使用了字节级 BPE、SuperBPE（多词合并）、Unicode 组合标记保留、脚本无空格分块、Hugging Face tokenizer 库以及自研代码，评估采用 FLORES‑200 数据集。

**📊 数据集**

数据集主要包括 FineWeb‑2、Sangraha、FineWeb‑Edu、L3Cube HingCorpus、MeCorpus 等人类写作文本，总计约1.69 B字符，覆盖195+语言。

**📈 对比分析**

通过在 FLORES‑200 devtest 上统计每种语言的 token 数量与 GPT‑4o、cl100k_base 及其他公开 tokenizer 对比，Havon v4‑200k 在204语言中有161种低于 GPT‑4o，25种印度语言全部低于 GPT‑4o，英语 token 数量减少约6%。

**⚠️ 局限性**

主要限制在于仅提供 token 计数未评估模型性能，训练样本覆盖有限且语言预算不均衡，第二阶段训练仅用 12% 数据，且未验证公开 tokenizer 版本一致性。

---

## 58. A Functional Preservation Score for Auditing Connectome Operations: Compression, Alignment, Transport, and Local-Rule Writing

**arXiv ID:** 2611.13681 | [PDF](https://arxiv.org/pdf/2611.13681v1)

**作者:** Praveg Vashishtha `[一作]` (Indian Institute of Technology), Praveg Vashishtha `[通讯]` (Sopra Banking Software)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `fede83ac-7505-405f-ab37-e7284695c47f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为 Functional Preservation Score (FPS) 的衡量网络功能保持的指标，并用其评估了压缩与推测性“心智迁移”流水线的各个阶段。

**💡 创新点**

创新点在于：① FPS 提供了跨底层的功能保持度量；② 揭示了整数量化的稀疏性底限；③ 证明了海马回放序列在极大压缩下仍可识别；④ 系统性模拟检验了迁移流水线的三阶段失败模式。

**🔧 技术方法**

采用了 null‑normalised 的 FPS 计算、bootstrap 置信区间、单侧置换检验、Gromov‑Wasserstein 对齐、混合精度量化、自动编码器压缩与递归网络建模。

**📊 数据集**

使用公开数据集：C. elegans 的真实躯体神经网络、DANDI 000044 的海马回放序列（48 轴向单元）以及小规模人工构造网络。

**📈 对比分析**

通过 FPS 与基线对比，压缩实验中 FPS 从 0.03 上升到 0.30；量化实验表明 int4 量化导致 73% 的有效稀疏性；回放压缩在 8–16 位/事件时 Top‑1 识别率仍高于 0.63，显著优于随机对照；迁移流水线各阶段在模拟中均显示显著失败（p<10⁻³）。

**⚠️ 局限性**

主要限制包括：仅在 306 细胞或 ≤135 细胞的极小规模网络上测试；采用简单的泄漏率递归模型；测试集样本量有限；未能在真实多体系统上验证；FPS 仅衡量功能保持，无法说明主观体验。

---

## 59. Splitting for Stability: a Modified Reward Scheme for Transaction-Fee-Only Regimes

**arXiv ID:** 2611.13252 | [PDF](https://arxiv.org/pdf/2611.13252v1)

**作者:** Eric Brigham `[一作]` (University of Minnesota), Nicholas Hopper `[通讯]` (University of Minnesota)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种奖励拆分机制，通过将区块奖励分给胜者和其他原子矿工，以降低基于交易手续费的 fork 行为的激励；

**💡 创新点**

创新点在于引入分配奖励的拆分方式，并给出局部盈利阈值分析与仿真证明，显示可显著降低开启式 fork 的阈值；

**🔧 技术方法**

使用了局部盈利阈值理论推导、奖励拆分与平滑参数的组合、MILP 近似代理、以及基于学习的多轮挖矿仿真；

**📊 数据集**

利用三年比特币矿池区块产出数据作为真实矿工分布，并在此基础上进行仿真与合成实验；

**📈 对比分析**

通过比较 Default‑Compliant 参与率阈值、盈利自私挖矿阈值、费用吞吐量与废弃块率等指标，实验显示阈值从 66% 降至 36%（无平滑）甚至 23%（平滑），且自私挖矿阈值保持不变；

**⚠️ 局限性**

局限在于采用固定 1% 阈值判定原子性、仅考虑预设的策略空间、只测试单个自私矿工、参数选择缺乏全局最优性，且未覆盖多矿工自私挖矿情形。

---

## 60. DeRep: Learning Depth Representations for End-to-End Navigation in Cluttered Environments

**arXiv ID:** 2611.13499 | [PDF](https://arxiv.org/pdf/2611.13499v1)

**作者:** Hanyu Jin `[一作]` (Carnegie Mellon University), Kenji Shimada `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了结合局部鸟瞰图到达性监督与深度重建的表示学习方法DeRep，用于端到端导航；

**💡 创新点**

通过显式监督自由空间连通性并与重建共同训练，使编码器同时保留路径可达性与几何细节，从而提升导航鲁棒性；

**🔧 技术方法**

使用CNN编码器、BEV到达性与深度重建解码器、联合不确定性加权损失，以及线性探测器、流匹配模仿学习和PPO强化学习；

**📊 数据集**

在Habitat Gibson环境中采集约115万帧深度图进行预训练，随后在Isaac Sim、Gazebo及真实四旋翼实验中评估；

**📈 对比分析**

通过冻结预训练编码器，使用线性探测器、流匹配模仿学习和PPO评估，DeRep在六个不同成功率/碰撞/鲁棒性条件下平均提升15–21 %成功率，并在四/六种评估条件中取得最高性能；

**⚠️ 局限性**

仅基于单帧深度，缺乏时序信息；在极端噪声或高度动态环境下仍易失效，且实验规模相对有限。

---

## 61. PhyRubric: Rubric-Grounded Diagnostic Evaluation of Physical Commonsense in Text-to-Video Generation

**arXiv ID:** 2611.13576 | [PDF](https://arxiv.org/pdf/2611.13576v1)

**作者:** Huiyi Chen `[一作]` (University of Illinois Chicago), Baoxin Li `[通讯]` (University of Illinois Chicago)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 PhyRubric 框架，用于基于每个字幕先行构建 rubric 的物理常识诊断评估，并实现 PhyRubric‑Bench 与 PhyRubric‑Train 两个评测集；训练出开放源码的 PhyRubric‑Judge；

**💡 创新点**

核心创新在于：先行生成字幕专属 rubric 并统一复用；通过证据扎根的项目级判定自动推导 1–5 评分并给出诊断；利用 rubric‑guided agent 进行大规模自动标注；让小型 VLM 在物理评估上匹敌前沿专有模型；

**🔧 技术方法**

使用 Qwen3‑VL‑8B VLM 进行 LoRA 微调，配合 agentic annotator 与 rubric 生成器；对视频进行 16 帧采样；实现语义遵从（SA）与物理常识（PC）两维评分与诊断；

**📊 数据集**

基准集为 PhyRubric‑Bench（245 条 caption、2,433 条视频）与 PhyRubric‑Train（947 条 caption、3,788 条视频，涵盖 4 个开源 T2V 生成器）；实验使用 10 种 T2V 模型输出；

**📈 对比分析**

将 10 款 T2V 模型在 SA、PC 与 JP 上进行比较，开源模型 SA <3、PC <3.5；专有 Seedance 2.5 取得 44% JP；PhyRubric‑Judge 与 Gemini 4 Argon 在人类评分上相关性相近（SA 0.61，PC 0.32），显著优于其他公开评判器；

**⚠️ 局限性**

仍存在局限：物理常识整体表现远未成熟；语义遵从是主要瓶颈；模型常通过不出现事件来逃避物理评估；rubric 需要人工审核，规模受限；在热与声学等领域表现尤差；

---

## 62. Latin American Studies since the 1980s: From Apocalyptic Dreams to Assimilationist Consolidation

**arXiv ID:** 2611.13216 | [PDF](https://arxiv.org/pdf/2611.13216v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 63. Uncovering the Blind Spots of OKS-mAP: A Unified Protocol for Occlusion, Joint-Angle, and Temporal Evaluation of 2D Pose Estimators

**arXiv ID:** 2611.12911 | [PDF](https://arxiv.org/pdf/2611.12911v1)

**作者:** Yifan Fan `[一作]` `[通讯]` (Taiyuan Yangfan School), Yifan Fan (Taiyuan Yangfan School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套统一的补充评估协议，用以揭示当前主流OKS-mAP指标在遮挡、关节角度和时序稳定性方面的盲区。

**💡 创新点**

创新点在于将遮挡可见性分层、关节角度误差与帧间抖动三大指标整合到同一评估框架，并提供完整的统计置信区间与公开评估工具。

**🔧 技术方法**

采用了基于OKS的mAP分层计算、角度误差（RMSE/MAE/P95）与膝关节内翻检测一致性、以及OneEuro平滑后的帧间位移、抖动与跳跃率等时序指标，全部在单次推理通道完成。

**📊 数据集**

在COCO val2017、CrowdPose val、OCHuman test及PoseTrack18 val四个公开数据集上评估了YOLO11s-pose、其遮挡域微调版round5‑3、以及ViTPose-B/L transformer模型。

**📈 对比分析**

通过对比实验发现：全局mAP往往高估遮挡场景精度（Δ_occ在‑0.24到‑0.47之间）；遮挡层的角度误差显著放大，遮挡训练提升了遮挡关节的角度精度；帧间抖动在原始输出下相对一致，使用像素域OneEuro平滑可在不降低AP的前提下降低跳跃率约10%。

**⚠️ 局限性**

局限包括：遮挡分层仅依赖v标签，未覆盖所有遮挡级别；角度评估基于二维投影角度，未考虑三维视角变化；时序指标仅在PoseTrack18的保留子集（≈53%）上验证；跨架构比较受预处理、输入尺寸和翻转测试差异影响。

---

## 64. Auditing LLM Pseudo-Label Distillation for Review Informativeness at E-Commerce Scale

**arXiv ID:** 2611.13797 | [PDF](https://arxiv.org/pdf/2611.13797v1)

**作者:** M. Emin Dogan `[一作]` (Hepsiburada), H. Bahadir Sahin `[通讯]` (Hepsiburada)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在土耳其电商平台上，用 Gemini 3.1 Flash‑Lite 生成的伪标签对 5.34M 条评论进行硬标签蒸馏，训练 BERTurk、ModernBERT 等轻量编码器，在新 SKU、时间外以及单一人工审计集上评估其教师忠实度与编辑者一致性，并测算推理吞吐量。

**💡 创新点**

① 在新 SKU 时序外测试中加入 H3 目录前缀可轻微提升教师忠实度；② 通过单一人工审计展示教师忠实度并不一定能保证与编辑者的一致性；③ 提供 50.2 倍吞吐量与教师对比的实测。

**🔧 技术方法**

使用 Gemini LLM 作为教师生成三阶离散标签；采用硬标签蒸馏结合交叉熵（CE）与条件序数回归（CORN）作为损失；在 BERTurk、ModernBERT、XLM‑R 等编码器上进行训练；评估指标包括 QWK、准确率、宏 F1、MAE、NDCG 等。

**📊 数据集**

土耳其 2025‑2026 年 5.34M 条商品评论，按时间分割为训练、验证、IID 与严格 OOT（新 SKU）集；另提供 4K 人工审计子集；外部 Amazon 8 领域 5.4K 条评论做语言/市场混合对比。

**📈 对比分析**

在严格 OOT（159k 条新 SKU）上，ModernBERT/CORN 的 QWK 最高为 0.9223；在 16K 人工审计上，教师与学生 QWK 分别为 0.9324 与 0.85 左右；NDCG 方面，BERT‑Join 与 BERTurk/CORN 在排名任务上实现近乎 1.0 的 macro‑NDCG；吞吐量比教师快 50.2 倍。

**⚠️ 局限性**

① 评估仅基于单一人工审计，缺乏多注释者交叉验证；② 学生模型输入与教师/人工审计存在信息不对称（仅文本或加 H3 前缀）；③ OOT 与审计集属于不同分布，无法直接对比；④ Amazon 测试同时涉及多重偏移，难以单独评估语言或市场影响。

---

## 65. Improved Bounds for Online Matching with Random Edge Arrivals

**arXiv ID:** 2611.13571 | [PDF](https://arxiv.org/pdf/2611.13571v1)

**作者:** Yilong Feng `[一作]` (University of Macau), Zonghan Yang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个三阶段的在线匹配算法，并给出了随机边到达模型的竞争比下界（0.5445/0.5535）和上界（0.6964/0.7043），同时提供了形式化验证与多线程求解器。

**💡 创新点**

创新点在于：①通过有选择性地在早期拒绝部分边来预留端点，从而突破传统 1/2 的基准；②首次给出 0.5445 的竞争比；③使用精确动态规划与形式化验证首次得到更紧的上界，并通过大规模枚举发现了最差实例。

**🔧 技术方法**

使用的技术包括：三阶段算法设计、局部恢复分析、存活概率的多项式展开与凸组合、局部贪心比较、动态规划（递推求解最优策略）、精确枚举与剪枝、Lean4 形式化证明、并行 C++ 计算实现。

**📊 数据集**

主要在小型图（如 G10、B2,3,3 等）上进行精确评估，并对所有非同构图（至 11 或 12 顶点）进行穷举搜索；还利用这些实例作为最差案例验证上界。

**📈 对比分析**

与之前的 0.50009/0.5+ε 等竞争比对比，证明新算法在一般图上达到 0.5445、在无三角图（包括二分图）上达到 0.5535，显著优于过去的结果；上界 0.6964/0.7043 表明在该模型下已逼近最优竞争比的极限。

**⚠️ 局限性**

限制：下界与上界仍存在约 0.15 的差距；上界的计算仅适用于已知同构图的模型或小规模图；算法参数在固定区间内才能实现最优，且对更大图的实际性能尚未验证；实现细节（如多线程同步与内存消耗）也未在论文中给出完整说明。

---

## 66. The Indic Token Tax in 2026: Measuring Tokenizer. Disparity and Its Cost for the Languages of India

**arXiv ID:** 2611.13058 | [PDF](https://arxiv.org/pdf/2611.13058v1)

**作者:** Johin Johny Arimbur `[一作]`, Shervin Sam `[通讯]` (Havon AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了2026年印度25种语言在各公开Tokenizer（如GPT‑4o、Sarvam等）中的token稀缺性、成本差异，并给出了基于真实API价格的月度成本估算。

**💡 创新点**

首次将正式文本（FLORES‑200）、罗马化文本（PHINC、Dakshina）和形态分割（MorphTok）结合，提供了Token边界与形态边界对齐度的度量，并用真实货币价格评估token税。

**🔧 技术方法**

使用tiktoken与各Tokenizer库对文本进行token计数，采用bootstrap置信区间统计parity ratio，计算形态边界精确度（precision/recall/F1），并将token计数与API定价映射。

**📊 数据集**

主要数据集包括FLORES‑200（25种印度语言句子）、PHINC Hinglish‑English对、Dakshina罗马化‑原文对、MorphTok形态分割表、L3Cube HingCorpus和MeCorpus。

**📈 对比分析**

通过parity ratio（token数相对英文）和95%置信区间进行横向比较，结果显示GPT‑4o在大多数语言上相对英文约1.6–5倍，Sarvam‑30B/105B最均衡（≤2.6倍），但所有Tokenizer的token边界对齐形态边界的F1均低于0.48。

**⚠️ 局限性**

局限性包括：数据集缺失部分印度语言（如Bodo、Dogri、Konkani）；仅评估正式文本而非对话文本；罗马化文本未与英文对齐；MorphTok仅覆盖印地语和马拉地语；成本估算基于假设且未验证Sarvam API使用的Tokenizer；未检测tokenizer是否已覆盖训练语料。

---

## 67. TypedInk: Knowing When Not To Draw

**arXiv ID:** 2611.13027 | [PDF](https://arxiv.org/pdf/2611.13027v1)

**作者:** Weiwei Yang `[一作]` `[通讯]` (Solus), Weiwei Yang (Solus)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了一种 TypedInk 编译器，能够在不完整的科研记录下生成合适类型（图表、示意图、实验平面）的科学插图，并在缺乏足够证据时主动拒绝绘制。

**💡 创新点**

首次引入“证据合同”与“完整性门控”概念，将类型推断与执行检查结合，防止在图表中生成虚假数值，从而实现对不完整输入下科学插图真实性的评估。

**🔧 技术方法**

结合文本检索与跨编码 reranker、LLM（Qwen3.7-Plus）生成 Matplotlib 代码、图像生成模板（Qwen-Image 和 Qwen-Image-3.0-Pro）以及聊天模型规划器。

**📊 数据集**

构建了 TypedFigBench（16 篇插图 × 5 视图共 80 例）作为核心基准，并与 PaperBananaBench、FigureBench、SciFig-Bench 等现有基准进行对比。

**📈 对比分析**

通过层 A 的类型推断诊断和层 B 的真实生成幻觉评估，TypedInk 在缺表视图下拒绝 100% 的金标图，而无门控 baseline 绘制 100%；在完整视图下恢复所有金标数字（recall 1.0），整体类型准确率 0.71；在 VL 评估中，TypedInk 的拒绝率更高且更符合真实数据。

**⚠️ 局限性**

对结构/大纲视图的类型推断仍有较高错误率，缺乏足够实验量的真实数据集，且无法阻止外部直接调用图像生成后将假图据呈现为真实实验记录。

---

## 68. Monitoring One Channel Is Not Communication Control: Causal Evidence for Adaptive Reallocation of Joint Message--Route Codes

**arXiv ID:** 2611.12959 | [PDF](https://arxiv.org/pdf/2611.12959v1)

**作者:** Liu Hung Ming `[一作]` (PARRAWA AI), Tian Xia `[通讯]` (PAII)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在一款基于深度强化学习的双代理环境中，作者设计并实现了一个自适应监控游戏，探究在对通道进行可观测性或惩罚时，学习到的通信协议是否会被迫迁移至未受监控的路径，或被完全抑制。

**💡 创新点**

创新点在于首次通过对比“选择性迁移”“完整覆盖”和“正确动作惩罚”三种监控策略，结合冻结干预（消息/路径打乱、联合打乱、擦除）构建了一套严谨的因果评估框架，证明了监控可导致协议重分配而非完全消失。

**🔧 技术方法**

主要技术包括 actor‑critic 强化学习、在线非平稳监控器（基于交叉熵更新的目标预测器）、多种冻结干预方法（全局/键内/路径/联合打乱、擦除）、以及通过覆盖率与检测概率计算的路由价值模型。

**📊 数据集**

实验数据来自自行搭建的“可观测通道”环境，在两个独立的 20‑seed 批次中进行 20,000 步训练，未使用公开数据集，全部为自生成任务与路由设置。

**📈 对比分析**

对比方法为“未监控 + 静态部分覆盖 + 在线部分覆盖 + 在线完整覆盖 + 正确动作惩罚”，通过任务准确率、影子路由使用比例、冻结干预下的消息贡献度等指标评估。结果显示：在线部分覆盖将影子路由使用率提升约 70%（无显著准确率下降），完整覆盖消除了部分迁移但未消除通信，正确动作惩罚导致准确率跌至低于随机水平。

**⚠️ 局限性**

局限性包括：仅考虑了两条预定义路径；监控器模型是固定且可观测的，未检验开放式侧信道发现；实验基于单一 actor‑critic 架构，缺乏跨算法验证；冻结干预的计算开销与可扩展性未评估；缺乏对真实世界通信协议（如 LLM 对话）的直接验证。

---

## 69. Uni-Loc: Weakly Supervised 3D CT Pathology Localization via Spatially Grounded Vision-Language Classification

**arXiv ID:** 2611.13330 | [PDF](https://arxiv.org/pdf/2611.13330v1)

**作者:** Hao-Chih Lee `[一作]`, Xueyan Mei `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出并实现了Uni-Loc框架，利用扫描级标签同时完成肺部CT异常的三维定位和分类。

**💡 创新点**

创新点包括：① 用病理条件查询的相似度池化将分类与定位统一；② 采用token‑to‑voxel U‑Net上采样实现高分辨率定位；③ 无需额外参数的梯度引导阈值（GradT）实现病灶分割。

**🔧 技术方法**

技术包括：BERT‑style多模态Transformer、CLIP式文本查询、相似度池化、UNet进阶上采样、梯度归一化阈值、线性分类头、预训练及EMA。

**📊 数据集**

使用的公开数据集包括355张带体素级标注的胸部CT（GGO、PE、LAD、AS、IO），以及CT‑RATE和RAD‑ChestCT用于分类评估。

**📈 对比分析**

与Uni‑CLS‑max、LXT、梯度显著性、COLIPRI‑CRM等方法对比，Uni‑Loc在五个定位任务中宏平均AUROC 0.961、AUPR 0.345，明显优于基线；分类AUROC分别为88.39%（CT‑RATE）和78.03%（RAD‑ChestCT），亦优于同类方法。

**⚠️ 局限性**

局限性包括：对部分病灶（如胸腔积液）的定位仍不理想；依赖扫描级标签仍可能导致对细小或稀疏病灶的误检；上采样和相似度池化对模型尺寸和训练成本有一定要求。

---

## 70. GroundCascade: Selective Second Opinions for GUI Grounding

**arXiv ID:** 2611.13177 | [PDF](https://arxiv.org/pdf/2611.13177v1)

**作者:** Adrian Phoulady `[一作]` (University of Connecticut), Hady Ahmady Phoulady `[通讯]` (Embry Riddle Aeronautical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文没有实际实验内容，只有关于LLM使用声明的说明

**💡 创新点**

没有提出任何创新点

**🔧 技术方法**

主要使用ChatGPT进行文献综述、论文写作与编辑

**📊 数据集**

未使用任何公开数据集

**📈 对比分析**

未给出任何对比方法或性能评估

**⚠️ 局限性**

缺乏实验设计、数据与性能评估，无法验证任何结论

---

## 71. Ender: Visual Scaffolding for Geometric Proof Comprehension

**arXiv ID:** 2611.13368 | [PDF](https://arxiv.org/pdf/2611.13368v1)

**作者:** Hwei-Shin Harriman `[一作]` (Carnegie Mellon University), Joshua Sunshine `[通讯]` (Carnegie Mellon University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 Ender，一种交互式渲染系统，用于可视化两栏几何证明的逻辑状态，并将其与相应的图形构造同步；

**💡 创新点**

创新点在于：①动态图形注释与文本同步，②以流图形式显式展示证明结构，③支持部分或错误证明的持续反馈，④通过色彩与箭头实时呈现证明依赖与矛盾；

**🔧 技术方法**

技术实现基于 React + TypeScript 的 Web 应用，利用自定义数据规范生成图形、依赖箭头和高亮；

**📊 数据集**

使用了包含 12 个正确/错误证明、13 条命题和 20 条理由的手工构造数据集；实验数据来自 12 名美国中学生的问卷和思考实验；

**📈 对比分析**

与传统两栏数字界面进行受试者对照实验，思考实验中 Ender 组平均准确率 80% 远高于对照组 58%（统计显著），且在错误识别与纠正题型上表现尤为突出；

**⚠️ 局限性**

局限包括：未对单一功能做消融验证、样本量小、仅单次实验未检验长期效果、缺乏高分辨率眼动数据、实验仅涉及高中三角形相似与全等证明等有限范围。

---

## 72. An Empirical Study on Improving LLM-Based Issue Classification through Domain-Informed Analysis

**arXiv ID:** 2611.13470 | [PDF](https://arxiv.org/pdf/2611.13470v1)

**作者:** Anubhav Gupta `[一作]` (University of British Columbia), Gema Rodríguez-Pérez `[通讯]` (University of British Columbia)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对6种大型语言模型在三分类（内在缺陷、外部缺陷、非缺陷）任务中，使用18种提示配置并进行实验与分析；

**💡 创新点**

将专家对模型推理失败模式的分析转化为域知情的多阶段分类管线，以提升模型准确性和外部缺陷召回率；

**🔧 技术方法**

利用大语言模型（Gemini、GPT、DeepSeek等）进行提示式推理、生成理由并在多阶段管线中拆分“缺陷有效性”与“缺陷来源”两步判定；

**📊 数据集**

使用InEx‑Bug v3.0数据集（428条npm仓库的GitHub issue），涵盖内在、外部缺陷及非缺陷三类标签；

**📈 对比分析**

通过宏观F1、准确率与外部缺陷召回率与单提示式基线比较，最佳管线实现宏观F1 0.801、准确率0.815，管线后进一步提升至准确率0.827、外部缺陷召回率0.757，较基线提升1.2个百分点与7.5个百分点；

**⚠️ 局限性**

受限于仅使用npm生态的样本、可能存在的训练集泄漏、推理理由不一定真实反映内部决策、且实验仅覆盖选定的六种LLM，因而结果对其他生态和模型的可推广性有限。

---

## 73. Mobile AI Factory: Building Specialized Language Models for Your Mobile Device

**arXiv ID:** 2611.13509 | [PDF](https://arxiv.org/pdf/2611.13509v1)

**作者:** Evan Li `[一作]` (University of Washington), Shyamnath Gollakota `[通讯]` (University of Washington)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Mobile AI Factory，一个能为指定任务和目标手机自动生成微调、量化并满足资源限制的专用小型语言模型的代理系统。

**💡 创新点**

创新点包括：①将手机硬件测评纳入模型选择；②从小到大分桶搜索并自适应目标；③利用 LLM orchestrator + 回滚机制进行迭代搜索；④基于失败类别生成合成训练数据；⑤整体实现手机端可落地的端到端流水线。

**🔧 技术方法**

使用技术包括：Claude Sonnet 5 作为 orchestrator、Qwen3.6‑35B‑A3B 作为 teacher、LoRA 微调、4/8/16‑bit 量化、llama.cpp/MNN 推理引擎、LangGraph 状态机、NVIDIA L40S GPU 训练、手机性能测评（prefill、decode、内存、能耗）。

**📊 数据集**

实验使用十个真实任务：BC5CDR、Calendar JSON、TOPv2、MultiCoNER、BFCL、CLINC150、SMS Spam、GoEmotions、BEA‑19、DialogSum；每任务 2600–5000 训练样本，1000 行 held‑out 集。

**📈 对比分析**

与 Claude Sonnet 5、DeepSeek v4 Flash 等 frontier 模型对比，工厂模型在 5 个任务上优于两者、3 个任务匹配；在未见测试集上仍保持优势；搜索速度比全枚举快约 19 倍（8.6 h 内达到 0.830 目标）；在四台手机上满足内存、延迟、能耗等多种资源约束。

**⚠️ 局限性**

局限性：仍需开发者手工提供任务说明与输出验证器；模型池限定在 135 M–4 B 参数范围；合成数据受 teacher 质量限制；仅覆盖 CPU/GPU，未考虑 NPU；手机测评耗时且未动态适应多任务并发或热量变化；整体自动化程度尚有限。

---

## 74. Function Inversion Reporting

**arXiv ID:** 2611.13294 | [PDF](https://arxiv.org/pdf/2611.13294v1)

**作者:** Itai Dinur `[一作]` (Ben-Gurion University), Yiding Zhu `[通讯]` (Georgetown University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究了一个函数反转问题，分为预处理和在线两个阶段，旨在找到满足 f(x) = y 的所有前像 x。

**💡 创新点**

通过统一和改进现有算法，提出了改进的数据结构，能够在多个问题上实现更好的性能。

**🔧 技术方法**

使用了Fiat-Naor算法和随机分区技术，结合了强对偶独立哈希函数。

**📊 数据集**

使用了任意函数 f: [N] → [N]，并在多个数据结构问题上进行了应用。

**📈 对比分析**

与现有方法相比，提出的算法在大多数情况下具有更好的时间空间权衡，特别是在 y 的前像数量较大时，性能显著提升。

**⚠️ 局限性**

在某些情况下，算法的性能可能不够优化，尤其是在处理特定类型的函数时。

---

## 75. Evaluation protocols for historical script classification: the cost of manuscript-level leakage and the role of soft labels

**arXiv ID:** 2611.12969 | [PDF](https://arxiv.org/pdf/2611.12969v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 76. DepthGS: Depth-Guided Gaussian Splatting with Adaptive Regularization for High-Quality Surface Reconstruction

**arXiv ID:** 2611.13628 | [PDF](https://arxiv.org/pdf/2611.13628v1)

**作者:** Xiao Ren `[一作]` (Southern University of Science and Technology), He Kong `[通讯]` (Southern University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出DepthGS，一种基于深度引导的高质量三维高斯 splatting 框架，用于表面重建与新视角合成。

**💡 创新点**

创新点在于推导单视角和多视角下深度差异与重建误差的上界，并据此设计深度差异加权的自适应法线正则化和跨视角深度一致性正则化。

**🔧 技术方法**

采用3D高斯 splatting、可微渲染、深度与法线估计、自适应权重、跨视角深度一致性约束、TSDF融合等技术。

**📊 数据集**

在DTU、Mip-NeRF360和Tanks & Temples三大公开数据集上进行实验。

**📈 对比分析**

与NeRF、VolSDF、NeuS、2DGS、GOF等方法对比，DTU Chamfer距离降至0.51mm（仅次于最佳方法），Tanks & Temples F1分数与最新Gaussian方法持平，Mip-NeRF360 LPIPS最低、PSNR第二，显示出优异的重建与渲染性能。

**⚠️ 局限性**

局限在于对复杂户外场景、强光照、碎片化几何的鲁棒性不足，且收敛速度与场景相关，不能在所有情况下统一提升性能。

---

## 77. Compile-Gated Lean Proof Refactoring with Enterprise Agent Practices: A Model per Step under a US\$3 Cap, Followed by Programmatic Rewrite Rules

**arXiv ID:** 2611.13011 | [PDF](https://arxiv.org/pdf/2611.13011v1)

**作者:** Bobber Cheng `[一作]` `[通讯]`, Bobber Cheng

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Lean Refactor Arena闭赛中，设计并实现了一个基于预算阶梯的LLM驱动证明改写管道，并结合程序化规则在$0成本下完成机械改写。

**💡 创新点**

创新点在于将企业级代理系统原则（调用计价、消费门控、程序化规则、完整审计）迁移到自动化证明改写；同时通过多阶段预算、模型选择和后期CPU重写实现了高效且可控的改写。

**🔧 技术方法**

使用了GPT‑6.1（Sol、Terra）、Gemini 4 Argon等LLM、Lean 4、自定义词法计数器与心跳计数器、程序化Rewrite规则和基准工具链；管道代码由Claude Code编写。

**📊 数据集**

数据集为Arena提供的BenchRows最终问题（ArkLib、CSLib、LeanPool、PhysLib、Strata、PutnamBench等），以及WarmupRows暖场问题。

**📈 对比分析**

通过与参考证明的L、H、Z分数对比，计算平均得分提升；Run 3在预算下实现了相对Run 1/2的显著提升，最终上传得分为UploadBoardCombined。

**⚠️ 局限性**

局限性包括非位可复现、部分后期零成本通道仅在样本上评估、模型提示效果有限、对新问题的泛化不确定、并且未对所有通道进行完整基准测试。

---

## 78. The RSI Tax: When Recursive Self-Improvement Is Not Worth the Cost

**arXiv ID:** 2611.13524 | [PDF](https://arxiv.org/pdf/2611.13524v1)

**作者:** Yitong Zhang `[一作]` (Peking University), Yisen Wang `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估递归自我改进（RSI）方法在持续任务流中的端到端 token 成本与准确率，并提出 RSI Tax 量化自我改进过程的 token 成本。

**💡 创新点**

引入 RSI Tax 概念并拆分为验证、探索、写入三类，分析其对后续任务使用比例，展示简单记笔记基线 SRN 在保持高效性与低成本方面的优势。

**🔧 技术方法**

基于大语言模型的多种 RSI 方法（Meta‑Harness、EvoSkill、RRSI、ReasoningBank、SRN），使用 token 计数、任务流评估与后续使用追踪等技术。

**📊 数据集**

使用 SWE‑Chain‑Evo（7条 15 任务链共 105 位置）和 Terminal‑Bench 2.1（89 任务）两套任务流，配合 DeepSeek V4.1 Flash 与 GPT‑6 Luna 两种模型。

**📈 对比分析**

通过比较六种方法在准确率与总 token 成本上的表现，发现 SRN 在大多数设置下准确率最高且 token 成本最低；Meta‑Harness 等方法的 RSI Tax 甚至可达任务执行成本的数百%。

**⚠️ 局限性**

评估仅覆盖两种模型和有限任务顺序，未研究动态更新策略，RSI Tax 与后续使用的关联性仅为描述性，缺乏因果干预验证。

---

## 79. OpticalRAG: Pixel-Space Compression for Token-Efficient Retrieval-Augmented Generation

**arXiv ID:** 2611.13675 | [PDF](https://arxiv.org/pdf/2611.13675v1)

**作者:** Senhao Liu `[一作]` (Renmin University of China), Xinran Zhang `[通讯]` (Claremont McKenna)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种在视觉编码空间对文档进行压缩并用于检索增强生成的框架OpticalRAG；

**💡 创新点**

核心创新在于将长文本渲染为页面图像，通过冻结视觉编码器得到约百个视觉标记，利用Encoder‑Bridged Retrieval实现跨模态检索，随后通过Query‑Driven Distillation将检索到的视觉标记压缩到仅16个query‑conditioned prefix tokens，无需对齐训练；

**🔧 技术方法**

主要技术包括冻结ViT视觉编码器、投影映射至视觉特征空间、跨注意力压缩查询、两层MLP桥接到LLM、深度卷积下采样以及Q‑Former式查询驱动压缩；

**📊 数据集**

实验使用LongBench‑v2、BAMBOO、LooGLE‑v2多选QA数据集，训练时采用Wikipedia/C4文本进行预训练与任务微调；

**📈 对比分析**

与闭书基线、标准RAG、xRAG、LLMLingua、Select Context、RECOMP等方法比较，OpticalRAG在三大基准上均取得最高CAP，并在LongBench‑v2 Raw score提升5.17点，仅需16个prefix tokens；

**⚠️ 局限性**

局限性在于仅评估多选QA场景，固定token预算可能不足以覆盖跨段多证据需求，且对开放式生成和自适应前缀长度的适用性仍待进一步研究。

---

## 80. Minimum Membership Dominating Set and Its Generalization: Leafage vs. Asteroidal Number

**arXiv ID:** 2611.13392 | [PDF](https://arxiv.org/pdf/2611.13392v1)

**作者:** Yuma Tamura `[一作]` (Tohoku University), Xiao Zhou `[通讯]` (Tohoku University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了最小成员数支配集（Minimum Membership Dominating Set, MMDS）及其通用形式 Min.(σ,ρ)-Dominating Set 在离散图参数化下的可解性，重点关注弦图的叶数（leafage）和任意图的星形数（asteroidal number）两个结构参数。

**💡 创新点**

创新点包括：① 证明 MMDS 在叶数 + ℓ 参数化下为 W[1]-hard，说明叶数虽能解决普通支配集但对成员数约束几乎没有帮助；② 设计了一个关于星形数 + μ 的 XP 算法，利用基于 BFS 层的动态规划与新的组合上界；③ 提出一个结构性质——任意图存在支配集使得每个顶点最多被 7(G)–6 个支配点覆盖，从而得到仅以星形数为参数的 XP 算法；④ 将结果推广到距离‑r 变体。

**🔧 技术方法**

主要技术手段包括：从多色团（Multicolored Clique）构造树表示（tree representation）和一系列精心设计的 gadget（selection、bound、grouped bound）实现 W[1]-hardness；基于 BFS 层的动态规划和对星形数上界的分析实现 XP 算法；以及利用星形数与叶数的关系和图的最小支配集的结构特性获得成员数上限。

**📊 数据集**

由于是理论性论文，未使用具体实验数据集；所有结果均通过严谨证明得出。

**📈 对比分析**

比较方式：与已知的叶数参数化结果对比，说明 MMDS 与普通支配集在参数化复杂度上存在显著差异；与距离‑r 变体的关系通过图的 r‑幂（G^r）映射展示；在算法层面，仅给出时间复杂度上限 O(n^{3((G)+1)(σ_max+1)}) 的 XP 算法，未进行实验性能评估。

**⚠️ 局限性**

局限性：① 结果仅为 XP，未能进一步提升为 FPT；② 对其它支配集变体（如总支配集、有效支配集）在星形数参数化下的可解性仍未解决；③ 证明依赖于复杂的 gadget 构造，难以直接推广到更一般的参数；④ 对实际大规模图的可行性缺乏实验验证。

---

## 81. GTAGym: Real AAA Game Tasks Environment for Evaluation and Reinforcement Learning

**arXiv ID:** 2611.13028 | [PDF](https://arxiv.org/pdf/2611.13028v1)

**作者:** Jiajun Chen `[一作]` (Shanghai Innovation Institute), Xipeng Qiu `[通讯]` (Shanghai Innovation Institute)

**通讯引用:** 32281 | [OpenAlex ID](https://openalex.org/A5044665993)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

论文未提供具体内容，因此无法总结做了什么。

**💡 创新点**

论文未提供具体内容，因此无法总结创新点。

**🔧 技术方法**

论文未提供具体内容，因此无法总结使用的技术。

**📊 数据集**

论文未提供具体内容，因此无法总结使用的数据集。

**📈 对比分析**

论文未提供具体内容，因此无法总结比较的方法及性能。

**⚠️ 局限性**

论文未提供具体内容，因此无法总结限制因素。

---

## 82. Incorporating World Models in Decision Making for Next Best View Selection in Active Sensing

**arXiv ID:** 2611.13229 | [PDF](https://arxiv.org/pdf/2611.13229v1)

**作者:** Nikita Sarawgi `[一作]` (GrayMatter Robotics Inc.), Satyandra K. Gupta `[通讯]` (GrayMatter Robotics Inc.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种在有限本地校准数据下，保持预训练世界模型不变的部署适配方法，用于下一最佳视角（NBV）选择。

**💡 创新点**

创新点在于分离任务级预测与传感器返回模型，仅学习一个轻量化的适配器来估计视角下的测量缺失，从而在不需大规模微调的前提下提升感知效率。

**🔧 技术方法**

使用FrameCrafter生成RGB预测，Depth Anything 3估计几何；通过视距、入射角、像素强度构造特征，并用稀疏变分高斯过程分类器训练适配器；采用贪婪的视角选择策略。

**📊 数据集**

在FANUC M‑710iC机器人上，使用Zivid 2+ LR110结构光传感器，对四个反射型头盔进行50次采集（约10⁶像素）做适配器训练，并在一只蓝色头盔上测试。

**📈 对比分析**

与传统的“名义”NBV方法（直接使用预测表面覆盖率）相比，适配器方法在达到98%覆盖率时仅需4次捕获，名义方法需10次，覆盖率提升约114%（vs 11.7%）。

**⚠️ 局限性**

局限在于依赖预先构建的参考表面，且适配器的泛化能力受限于相似的传感器、照明和材料条件，需在不同条件下重新校准。

---

## 83. Investigating Composite Modeling in Multi-Objective Bayesian Optimization for Scientific Design

**arXiv ID:** 2611.13565 | [PDF](https://arxiv.org/pdf/2611.13565v1)

**作者:** Ricky Ye `[一作]` (Cornell University), Peter I. Frazier `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在多目标贝叶斯优化（MOBO）中利用中间响应的组合建模（Composite modeling）对采样效率的影响。

**💡 创新点**

理论证明：当外部映射为双射或仿射（且GP协方差可分离）时，中间观测不提供额外预测信息；实验表明，在非双射、非仿射映射下，Composite modeling 能显著提升最终超体积（Hypervolume）和单目标简单后悔（Simple Regret）。

**🔧 技术方法**

使用 Gaussian Process 先验、Warped GP、LogEI/LogEHVI 采集函数、STCH 与 MORBO 等多目标采样方法，构建独立或独立-相关的GP 模型来预测中间响应与最终目标。

**📊 数据集**

数据集包括：六种合成的单目标映射（Identity、Cubic、Square、Shifted Square、Three-root Cubic、Cosine），以及八个科学设计基准（色彩配方、湖泊管理、桁架设计、功率流、纳米粒子设计、反应优化、Synthetic 1 与 Synthetic 2）。

**📈 对比分析**

与传统黑盒 GP（BlackBox-GP）和基于相同先验的 BlackBox-Induced 进行对比；Composite 在 15/16 组实验中获得正向超体积提升，单目标实验中平均降低约 99.8% 的最终简单后悔，表明组合建模在大多数情形下优于黑盒方法。

**⚠️ 局限性**

限制包括：高维问题与评估预算有限时收益不明显；使用独立 GP 模型未能捕捉中间响应间的相关性；实现细节（如核函数选择）与理论假设不完全一致，可能导致与理论结论出现偏差。

---

## 84. A Taxonomy of Failing Bug Reproduction Tests on SWT-Bench

**arXiv ID:** 2611.13395 | [PDF](https://arxiv.org/pdf/2611.13395v1)

**作者:** Luca Filippelli `[一作]` (York University), Hadi Hemmati `[通讯]` (York University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在SWT-Bench上对AssertFlip和OpenHands使用GPT-4o-mini和GPT-5-mini生成的bug reproduction测试进行失败案例手工标注，并构建了细粒度的失效分类法。

**💡 创新点**

提出了针对LLM驱动bug reproduction工具的全新失效分类法，并展示通过该分类法进行上下文增强能显著提升测试生成质量。

**🔧 技术方法**

使用大型语言模型（GPT-4o-mini、GPT-5-mini）生成测试，手工标注并采用主题分析方法构建分类法，利用SWT-Bench评估框架进行实验。

**📊 数据集**

使用SWT-Bench Verified benchmark（约435个失败实例），并公开标注结果于Zenodo。

**📈 对比分析**

通过对AssertFlip与OpenHands以及两款模型在失效率和分布上的对比，发现GPT-5-mini生成的测试数量更多、失败率更低；AssertFlip在精度上优于OpenHands。

**⚠️ 局限性**

实验仅覆盖Python语言和两款开源工具，未考虑其它语言或闭源框架；模型可能存在数据泄漏；人工标注存在主观性与模型随机性带来的可复现性限制。

---

## 85. Third-Order Asymptotics for Fixed-Length Lossy Compression of Discrete Memoryless Sources

**arXiv ID:** 2611.13573 | [PDF](https://arxiv.org/pdf/2611.13573v1)

**作者:** Vincent Y. F. Tan `[一作]` `[通讯]` (National University of Singapore), Vincent Y. F. Tan (National University of Singapore)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

研究有限字母离散无记忆源在固定误差概率下的固定长度有损压缩问题，确定其码本大小的三阶渐进展开

**💡 创新点**

首次给出三阶修正项 ½ log n + O(1)，从而精确揭示了有损压缩的细节量级，并消除了此前存在的对数修正误差

**🔧 技术方法**

结合了：① 逆变换加权权重估计（Scarlett‑Martinez‑Guillén 的多元加权条带估计）用于反证；② 定量 δ‑方法（Pinelis‑Molzon）与正态近似用于 achievability；③ 变形信息与失真球估计（Yang‑Zhang）用于球体概率控制

**📊 数据集**

使用的是理论模型：有限字母离散无记忆源，配备任意有限重构字母集，取内点失真阈值；没有实际数据集

**📈 对比分析**

与 Kostina‑Verdú、Moulin 等先前的二阶/三阶界相比，证明了 ½ log n 的精确系数，且误差项被限定在常数阶，提供了更紧的上/下界

**⚠️ 局限性**

仅适用于内部失真水平且源字母集有限；对连续/无界源、边界失真值以及更一般的失真度量尚未扩展

---

## 86. Beyond the Log-Linear Frontier: Pre-Asymptotic Analysis of Finite-Budget Neural Scaling Laws

**arXiv ID:** 2611.13534 | [PDF](https://arxiv.org/pdf/2611.13534v1)

**作者:** Tianyu Zhao `[一作]` (Shanghai Smart Logic Technology), Zhixiang Ren `[通讯]` (Shanghai Smart Logic Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过谱分析、Mellin–Barnes 技术和 KKT 水填充等理论工具，对有限预算下神经网络的前期学习曲线进行预渐近分析，揭示了容量和数据异质性如何导致传统 log‑linear 经验法则的曲率偏移和计算最优分配指数漂移。

**💡 创新点**

提出三项创新：① 解析容量侧的三项残差展开并解决多指数拟合的病态问题；② 通过 KKT 双重性将多域采样优化化为一维水填充，给出精确的瓶颈界和在线课程调度；③ 推导计算边界上分配指数的漂移方程并量化超参数冻结导致的曲率偏差。

**🔧 技术方法**

主要使用了谱解析（Mellin–Barnes 积分）、随机矩阵理论、KKT 水填充、Schur‑补分解、Aitken 差分、简化的二阶 Hessian 近似等数学方法，配合数值实验验证。

**📊 数据集**

在三组公开基础模型数据集上验证：Pythia（70M–12B）、Chinchilla（245 份训练跑）以及 RegMix（17 个不同领域）。

**📈 对比分析**

与传统单指数或多指数经验拟合相比，所提出的残差模型在 Pythia 上 MAPE 下降至 0.23% 以内；在 Chinchilla 上，使用 Log‑quadratic 表面能将高算力分配回报降低 76.4%；在 RegMix 上，加入跨域转移后 RMSE 下降 59.4%，并实现零 1M→1B 选择回报。

**⚠️ 局限性**

局限性包括：① 对极小训练规模的网格仍易受多指数共线导致的数值不稳定；② 需要先验谱衰减参数，无法自动估计；③ 只考虑了均衡混合的理想化场景，实际训练中的动态学习率、数据去重等非线性效应仍需进一步研究。

---

## 87. Advancing Mass-Conserving Deep Learning for Distributed Hydrologic Modeling: A Case Study in a Snow-Dominated Watershed

**arXiv ID:** 2611.13346 | [PDF](https://arxiv.org/pdf/2611.13346v1)

**作者:** Yihan Wang `[一作]` (University of Oklahoma), Tiantian Yang `[通讯]` (University of Oklahoma)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了分布式物理约束卷积LSTM框架PECAN，用于对科罗拉多北普莱斯河头水系统进行水文建模并诊断物理约束的结构限制。

**💡 创新点**

首次将质量守恒约束从传统的聚合LSTM扩展到空间分布卷积网络，并提供严格（PECANMC）与松弛（PECANMCR）两种约束形式，能够揭示模型在雪蓄积与释放过程中的缺陷，具有诊断性创新。

**🔧 技术方法**

采用卷积LSTM（PECAN）配合RVIC路由模块，比较了标准LSTM、MC‑LSTM、MCR‑LSTM以及过程模型VIC等基线；使用软/硬约束的门控设计实现质量守恒；使用Adam优化与余弦退火学习率。

**📊 数据集**

使用PRISM 4 km分辨率的降水、最高/最低温度数据；UA‑SWE（雪水等效）和SMAP L4土壤湿度（0–5 cm/0–100 cm）作为参考；USGS流量观测（北普莱斯河出口和米西根河上游）作为训练与验证目标。

**📈 对比分析**

在2013‑2022年期间用NSE、KGE、COR、PBias、FHV、FLV等指标评估；PECANMCR在多数指标上与VIC持平或优于其，并且在局部LSTM、MC‑LSTM、MCR‑LSTM基础上显著提升；在空间诊断上内部状态与SMAP土壤湿度有较高相关性。

**⚠️ 局限性**

主要局限在于模型未显式处理雪的蓄积与融化，导致雪水在内部状态中被部分“丢弃”到trash flux；序列‑一次训练导致内部状态在不同窗口间不连续，限制了长期过程的表征。

---

## 88. ARIA: An AI-Powered Radio-Channel Inference and Analysis Pipeline

**arXiv ID:** 2611.13326 | [PDF](https://arxiv.org/pdf/2611.13326v1)

**作者:** Homa Nikbakht `[一作]` (New York University), Theodore S. Rappaport `[通讯]` (New York University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一套端到端的AI驱动无线信道模型生成管道ARIA，能够从原始测量数据直接学习并拟合所有一阶信道参数分布，并将这些分布导入传统仿真器生成完整的信道样本，最终验证生成信道的二阶统计量与实测一致。

**💡 创新点**

创新点在于：① 用神经密度估计（MDN、NSF、DDIM）取代传统有限分布族，动态学习任意形状的参数分布；② 通过合成数据增强与最大似然补偿，将无监督学习得到的密度转换为闭式可用分布；③ 引入FiLM条件化实现跨频率联合学习，显著提升低样本频段拟合效果；④ 采用复合秩 + Friedman–Nemenyi统计检验做多模型、多任务的客观比较。

**🔧 技术方法**

核心技术包括滑动相关声纳测量处理、TCSL（时间聚类空间叶）参数提取、四种神经密度估计器（MDN、NSF、DDIM、KDE）、多指标归一化秩合成、Friedman–Nemenyi显著性检验、增量式最大似然参数化、FiLM特征线性调制、以及NYUSIM风格的Monte‑Carlo信道生成。

**📊 数据集**

使用NYU WIRELESS 16.95 GHz FR3室内热点测量数据（共20对TX‑RX，7 LOS、13 NLOS）以及同一环境下的6.75 GHz FR1（C‑band）测量数据做跨频率对照实验。

**📈 对比分析**

比较方法：对每个参数分布任务，在Protocol‑A（全样本内测）和Protocol‑B（五折交叉验证）下计算KS、CvM、AD、W1、Energy五个拟合指标，归一化秩求平均得到复合秩；使用Friedman检验与Nemenyi事后检验评估模型间差异。实验结果显示NSF在12/28任务中获胜，MDN次之，KDE、DDIM表现最差；FiLM条件化的NSF进一步提升均值秩，池化模式则适得其反；最终生成的信道样本在RMS时延、方位/仰角扩散等二阶统计量均与实测相符，误差均在2 ns/1°以内。

**⚠️ 局限性**

局限性：① 仍依赖已有测量数据，跨频率推广受限于频率间相似性；② 对高维联合分布（如功率–延迟耦合）仅通过条件流近似，未完全捕捉多元依赖；③ 生成器的训练与推理成本高，尤其是DDIM等参数化模型；④ 对极端稀疏环境（如 sub‑THz）需更多数据支持。

---

## 89. Multimodal Judges Flag, Humans Decide: A Review-and-Edit Framework for AI-Generated Product Video Ads

**arXiv ID:** 2611.13178 | [PDF](https://arxiv.org/pdf/2611.13178v1)

**作者:** Anoubhav Agarwaal `[一作]` (Anoki Inc), Susmita Ghose `[通讯]` (Anoki Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现了一个统一的审查和编辑框架，能够对电商平台的 AI 生成产品视频广告进行可扩展的质量评估与局部修复。

**💡 创新点**

创新点包括：
- 六维度（产品、场景质量、发音、脚本、音效、字幕）场景级审查流程，人工与 AI 判断共用同一问题集；
- 为每个维度构建可解释的 AI 判断器，输出与人工审查对应的具体缺陷位置与类别；
- 每个确认缺陷可直接跳转至最小修复单元（场景、字幕行、音频段）并调用对应的修复模型；
- 发布了可复用的审查门户模板与完整数据集，支持其他流水线快速集成。

**🔧 技术方法**

使用的技术包括：
- 视觉‑语言模型（VLM）与视频‑视频上采样模型（Repair）用于产品文本识别与修复；
- 场景借用与生成模型（reference‑to‑video）生成或替换错误场景；
- 语音‑文本模型（speech‑to‑text）用于检测发音错误；
- 语言模型与文本‑语音（TTS）用于脚本校正与重新配音；
- 文字‑音乐模型（text‑to‑music）替换背景音乐；
- 视觉‑语言模型与规则判定用于字幕定位与相关性评估。

**📊 数据集**

数据集：
- 120 条15秒生成广告（562 场景、20 个产品），每个广告都标注了 6 维度的人工评审结果和 AI 判断结果；
- 120 场景的盲标注（未暴露 AI 判断），用于评估判断器的 AUC；
- 另外 530 条字幕、若干场景质量与发音标注，用于进一步评测。

**📈 对比分析**

比较方法与性能：
- 产品判断器在盲标 120 场景上 AUC 达 0.84，显著优于通用视频质量/相似度评分（0.48–0.57）。
- 人工审查时，显示 AI 判断后，产品错误标记率从 26% 提升至 60%，而无 AI 判断时仅 26%。
- 评审平均时长 4.1 分钟（有 AI 判断）对比 2.9 分钟（无 AI 判断），成本约 1.40 美元/评审。 
- 修复模型 Repair 在 26 场景中完全修复 18 场，部分修复 8 场。 
- 通过交叉评审，Fleiss’ κ 在产品维度 0.61、场景替换 0.80，说明人工与 AI 判断在关键维度上高度一致。

**⚠️ 局限性**

局限性：
- 仅在单一生成流水线、单一市场、英语语音下实验，未验证在多语言或多品类场景的泛化；
- 评审与判断器仅基于 120 条广告，样本规模有限；
- 对完整循环（发现→编辑→二次评审→投放）的端到端效果评估尚未完成；
- 判断器与编辑工具使用的是封闭模型，难以外部复现；
- 只测试了 Repair 与脚本修复工具，其他修复方法的鲁棒性与用户满意度尚未充分评估。

---

## 90. TrustMod-SM: A Multi-Axis Benchmark for Evaluating Trustworthiness of LLMs in Social Media Content Moderation

**arXiv ID:** 2611.12970 | [PDF](https://arxiv.org/pdf/2611.12970v1)

**作者:** Aizan Zafar `[一作]` (International Institute of Information Technology), Srinath Srinivasa `[通讯]` (International Institute of Information Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了TrustMod-SM，一个统一的多维度评估基准，用于衡量大型语言模型在社交媒体内容审核中的可信度。

**💡 创新点**

创新点在于将可信度拆分为信任度、公平性、安全性、鲁棒性与语境完整性五个维度，并通过跨维度耦合矩阵量化它们之间的相互关系。

**🔧 技术方法**

研究采用大规模语言模型推理、结构化 JSON 输出、置信度估计、事实检查触发、可疑内容上报等技术手段进行评估。

**📊 数据集**

使用了八个公开的仇恨言论、毒性与误信息数据集（如SBIC、HateCheck、HateXplain、ToxiGen、HASOC、MHS、COVID‑FakeNews 等）共计28,792条样本。

**📈 对比分析**

对13个开源模型（0.5B–14B）以及3个更大模型和1个闭源模型进行对比，结果显示即使是最优秀模型（Qwen‑7B）在准确率、校准、差异性和安全性等多项指标上均表现不佳，表明当前模型尚未具备可部署水平。

**⚠️ 局限性**

局限性包括对大规模模型的鲁棒性未做全面评估、对误信息域的测试仅覆盖 COVID‑19 领域、以及跨维度耦合分析仅为探索性、未能得出因果结论。

---

## 91. Adaptive catalyst discovery emerges from composable scientific intelligence

**arXiv ID:** 2611.13536 | [PDF](https://arxiv.org/pdf/2611.13536v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 92. Pair-Merge Tokenisation Is Hard to Approximate on Binary Strings and Strongly NP-Hard on Unary Datasets

**arXiv ID:** 2611.13568 | [PDF](https://arxiv.org/pdf/2611.13568v1)

**作者:** Kundan Baliga `[一作]` (University of Illinois Urbana Champaign), Aditya Byju `[通讯]` (Georgia Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究字节对合并（BPE）和类似的合并式分词算法的近似可行性，证明在二进制字符串上无法在多项式时间内得到比 n/log n 更好的近似，比在单字母集合上则证明了两类分词任务的强 NP‑完整性。

**💡 创新点**

创新点包括：① 将合并操作视为小型语法，利用顶点覆盖与语法大小之间的桥梁；② 构造基于仿射复制与二进制编码的二进制硬实例，首次给出完整的全合并执行计划；③ 在单字母集合上首次给出 OPE‑seg 和标准下推分词的强 NP‑完整性证明；④ 证明 Karp 归约无法突破 n/log n 近似阈值。

**🔧 技术方法**

采用的技术主要是：图论（顶点覆盖）、语法压缩理论（最小 SLP）、仿射复制与编码技术、归约与加成法、以及复杂度理论中的 PCP 与 Karp 归约边界分析。

**📊 数据集**

未使用真实文本数据集，而是构造了合成字符串（如 s_H、W、F 等）以及单字母集合上的重复字符串数据集来证明硬性。

**📈 对比分析**

与现有方法（如 Greedy BPE、Re‑Pair 等）相比，本工作给出的是理论下界：在二进制字符串上最坏情况下的近似比为 Ω(n/log n)，而已知算法只能达到 O(n/loglog n)；在单字母集合上证明了相关分词任务的 NP‑完整性。

**⚠️ 局限性**

局限性包括：① 只对二进制和单字母集合做了最坏情况分析，未覆盖自然文本或多字母集合的实际性能；② 对全合并的上界仍为 O(n/(loglog n)^log_32)，与下界存在较大间隙；③ 归约仅适用于 Karp 归约，无法说明在更强假设（如 Gap‑ETH）下是否能突破；④ 对直接分词（WordPiece）等模型的可近似性未给出结论。

---

## 93. DocFS: File Systems as Agentic Reasoning Environments for Complex Multimodal Document Question Answering

**arXiv ID:** 2611.12957 | [PDF](https://arxiv.org/pdf/2611.12957v1)

**作者:** Bingdian Zhang `[一作]` (State Key Laboratory of AI Safety), Xueqi Cheng `[通讯]` (State Key Laboratory of AI Safety)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对复杂多模态文档问答，构建基于文件系统的环境 DocFS，将原始文档重新组织为层级目录和文件，并通过标准文件系统工具实现文档结构的导航、检索、读取和验证。

**💡 创新点**

创新点在于将文件系统本身视为可操作的 reasoning 环境：1) 将章节层级映射为目录结构；2) 将文本、图表、表格、公式等多模态内容分别存为对应文件；3) 通过结构描述文件（索引、关联）和图像文本描述文件显式暴露交叉模态关联；4) 让代理只需调用通用的文件系统工具（ls、grep、read、open_pdf），消除专用接口的需求，显著提升效率与准确性。

**🔧 技术方法**

技术方法包括：多模态文档解析与结构重建、公式/表格校正、文件系统构建、结构描述与图像描述文件生成、VLM（Qwen3.5-Plus/DeepSeek-V4.1-Flash）与 agentic 交互、工具调用框架、实验评估。

**📊 数据集**

使用的数据集为 MMLongBench-Doc（长篇多模态文档 QA）和 FinRAGBench-V（财务文档 QA）。

**📈 对比分析**

与 VLM+OCR、VLM-based RAG、VLM-based Agentic baseline（DocLens）进行对比。DocFS 在 MMLongBench-Doc 上准确率提升 2.9 点、F1 提升 3.9 点；在 FinRAGBench-V 上准确率提升 4.3 点；速度提升 4.3–10.3×，Token 消耗降低 1.5–10.1×，整体效果显著优于最强 baseline。

**⚠️ 局限性**

局限性包括：对文档结构的依赖，若文档缺乏明确层级或包含大量动态/非结构化内容则效果受限；整体性能受预处理解析质量影响；缺少针对更复杂跨模态交互的机制，未来可进一步完善。

---

## 94. Disentangling Multimodal Rotary Positional Embedding in Vision-Language Models

**arXiv ID:** 2611.13334 | [PDF](https://arxiv.org/pdf/2611.13334v1)

**作者:** Zhaoyi Liu `[一作]` (University of Maryland), Ang Li `[通讯]` (University of Maryland)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过控制注意力中的坐标分配，拆分并评估了多模态旋转位置编码（MRoPE）中跨模态空间效应与视觉内部空间结构对视觉-语言模型性能的影响。

**💡 创新点**

创新点在于提出了基于注意力特定坐标分配的实验框架，能够在保持视觉内部空间信息的同时独立消除跨模态空间偏好，从而揭示跨模态空间效应对模型并非有益且可能损害细粒度定位性能。

**🔧 技术方法**

采用了MRoPE系列（MRoPE‑I、VideoRoPE、HoPE）与注意力特定坐标分配技术，并在Qwen3‑VL（2B/4B/8B）模型上进行两阶段训练。

**📊 数据集**

使用了多种视觉理解、空间理解与定位评测基准：MMStar、SEEDBench、MMBench、RealWorldQA、BLINK、V*Bench、AI2D、OCRBench、DocVQA、InfoVQA、3DSRBench、Omni3DBench、Spatial457、SpatialEval、以及RefCOCO系列。

**📈 对比分析**

通过对比四种坐标配置（原始、仅消除跨模态空间、仅消除视觉内部空间、同时消除两者）进行训练和推理，结果显示：消除跨模态空间效应在保持视觉与空间理解得分的同时，显著提升RefCOCO定位的ACC@0.9（提升约1–9%），而保留视觉内部空间结构则是细粒度定位的重要正面因素；整体模型性能差异小，证明跨模态空间效应对整体任务贡献有限。

**⚠️ 局限性**

局限性包括：仅针对MRoPE系列的实验，未验证其他多模态位置编码方法；评测仅覆盖视觉-语言基准，可能不适用于其它跨模态任务；实验规模受限于公开模型和计算资源。

---

## 95. Stochastic Projections and Rank-Aware Learning Rates for Federated LoRA

**arXiv ID:** 2611.13331 | [PDF](https://arxiv.org/pdf/2611.13331v1)

**作者:** Saber Malekmohammadi `[一作]` (Stanford University), Golnoosh Farnadi `[通讯]` (McGill University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在联邦学习中对LoRA进行冻结与重采样的策略SPRA-LoRA，利用每轮随机重采样的A矩阵并采用与适配层秩相关的学习率，显著降低客户端计算和通信开销并提升模型精度。

**💡 创新点**

核心创新在于：① 将A矩阵冻结并每轮重新采样，使其仅作为随机低秩扰动；② 通过对不同秩的客户端采用秩感知学习率，缓解梯度扰动不均衡问题；③ 理论上证明该做法等价于在全梯度上施加随机低秩噪声，并给出收敛性分析。

**🔧 技术方法**

技术包括LoRA低秩微调、Gaussian随机初始化、秩稳定化（ν=0.5）、FedAvg聚合、Adam/SGD等优化器以及基于随机投影的梯度重塑。

**📊 数据集**

在Commonsense Reasoning（SQuAD、BoolQ等子任务）和GLUE（QNLI、MRPC、CoLA、MNLI、RTE、SST‑2、QQP）等自然语言处理基准数据集上进行实验。

**📈 对比分析**

与多种现有方法（FFA、HetLoRA、FLoRA、FlexLoRA、RoLoRA、LA‑LoRA等）在相同客户端秩设置下对比，SPRA‑LoRA在保持50%或更低客户端开销的前提下，平均精度提高2‑5%，且收敛更快。

**⚠️ 局限性**

局限性包括：对秩分布的依赖（需先行确定秩范围）；仅在语言模型微调任务上验证，其他任务或模型架构的适用性待进一步研究；重采样需要服务器同步随机种子，且在极端数据异质性下性能可能受限。

---

## 96. Teaching the Harness: Student-Aware Skill Transfer for LLM Agents

**arXiv ID:** 2611.13144 | [PDF](https://arxiv.org/pdf/2611.13144v1)

**作者:** Chang Liu `[一作]` (Colorado School of Mines), Bo Wu `[通讯]` (Colorado School of Mines)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在固定模型和 harness 环境中，利用学生模型来生成并筛选可复用技能库的教学方法——Student‑Aware Teaching。

**💡 创新点**

创新点在于：①构建包含经验剖面与分析假设的学生模型；②使用诊断探测（probe）验证假设；③通过实测任务的“录入”规则筛选有效技能；④训练后的技能库能在教师离开后继续提升学生表现。

**🔧 技术方法**

核心技术包括：大语言模型（LLM）教师与学生、执行 harness、自然语言技能生成与验证、Brier/AUROC 预测评估、技能选择器（selector）以及实验中的各种评测指标。

**📊 数据集**

实验数据集涵盖五类任务集：Terminal、Inventory API、Arithmetic、Python Code、Dependent Workflows，并在四个主要跑与一个扩展跑中使用，此外还测试了四个生产 harness（Codex、Pi、OpenCode、Claude Code）与公开数学子集（MATH、GSM8K）。

**📈 对比分析**

与无技能和无学生模型的“agnostic”教学对照，Student‑Aware Teaching 在四个主要跑中平均提升 5.42 分（相对 agnostic）和 11.46 分（相对无技能），在执行受限任务中提升尤为显著；ablations、误导性配置和技能库交换进一步验证了学生模型的价值。

**⚠️ 局限性**

局限性包括：仅对能执行流程的中等规模学生有效，对极小模型或接近上限的强大模型收益有限；依赖于 harness 的设计和技能选择器的策略；实验种子有限，需更多泛化验证。

---

## 97. Looping Deeper Without Growing the KV Cache: Recurrent Depth for Causal Encoder-Decoders

**arXiv ID:** 2611.13378 | [PDF](https://arxiv.org/pdf/2611.13378v1)

**作者:** Jiaqi Leng `[一作]` (University of California, San Diego), Hao Zhang `[通讯]` (University of California, San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种循环深度的因果编码解码器（CED），通过在编码器内部循环并利用最终回合的 KV 缓存，实现深层计算而不增加 KV 内存。

**💡 创新点**

创新点在于：①将循环放在编码器以动态细化全局 KV 源；②使用随机深度训练驱动循环核心收敛到固定点，使得仅保留最后一回合 KV 就能无损共享；③实现了与稠密循环 Transformer 相当的性能，却仅占 1/6 的 KV 内存。

**🔧 技术方法**

技术包括循环 Transformer 结构、滑动窗口注意力（SWA）、因果编码解码器架构、随机深度训练、最后一回合 KV 共享、跨层 KV 合并、预填充/解码分离以及大规模 MoE 基础模型。

**📊 数据集**

数据集涵盖 FineWeb‑Edu（预训练）、PG19、FinePDFs、GovReport（长文档测试）、DCLM CORE 与 CORE‑Extended（下游基准），以及 DeepSeek‑V4.1‑Flash 的生产规模实验。

**📈 对比分析**

通过与无循环 CED、稠密 Transformer 基线以及不同层放置的循环模型对比，循环 CED 在验证 BPB 上几乎与稠密循环模型持平（误差 <0.001），下游 CORE 及 CORE‑Extended 准确率提升 1‑3%，生产环境下 KV 缩减 36.8 倍，吞吐率提升 3.6×，可在单个 4×GB200 芯片上处理 128K token 请求。

**⚠️ 局限性**

局限性包括：①预训练实验仅至 340M 参数，未在更大 MoE 规模上完整训练；②假设大规模模型也能实现固定点收敛；③仅评估成本而非推理质量；④循环编码器在 prompt‑heavy 任务上预填充成本更高；⑤未在更广泛的长文本推理基准上验证。

---

## 98. YTShorts-92K: An Interpretable Multimodal Dataset of YouTube Shorts

**arXiv ID:** 2611.13002 | [PDF](https://arxiv.org/pdf/2611.13002v1)

**作者:** Celian Charrin `[一作]` (Grenoble INP), Roland Molontay `[通讯]` (Budapest University Of Technology And Economics)

**通讯引用:** 1212 | [OpenAlex ID](https://openalex.org/A5080190018)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了YTShorts-92K，一个包含92,654条YouTube Shorts的可解释多模态数据集，涵盖视频元数据、标题/描述/标签/转录文本、视频帧与缩略图视觉特征以及标题-缩略图一致性等80个特征；

**💡 创新点**

首次公开大规模可解释多模态短视频特征集合，聚焦预发布内容并实现跨模态一致性度量；

**🔧 技术方法**

采用YouTube Data API、字幕提取、NLP模型（VADER、YAKE、DistilRoBERTa、BART）、视觉特征提取（FFmpeg、PyAV、EasyOCR、YOLO、DeepFace）以及Qwen3VL多模态嵌入；

**📊 数据集**

使用YTShorts-92K数据集，涵盖12个手工定义的内容类别，共计92,654条短视频；

**📈 对比分析**

未给出具体模型对比，仅通过描述统计表明多模态特征与观看次数弱相关，可为后续预测与分析任务提供基线；

**⚠️ 局限性**

受限于欧美创作者样本、缺失后期观看时长与音频特征，以及仅覆盖2025–2026年期间内容，可能影响跨文化与时序推广性。

---

## 99. LandslideFM: A Foundation Model Towards Unified Landslide Perception and Understanding

**arXiv ID:** 2611.12936 | [PDF](https://arxiv.org/pdf/2611.12936v1)

**作者:** Shanwen Wang `[一作]` (City University of Macau), Jungong Han `[通讯]` (Tsinghua University)

**通讯引用:** 27582 | [OpenAlex ID](https://openalex.org/A5046605531)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种统一的基础模型LandslideFM，用于陆滑灾害的视觉感知与智能分析

**💡 创新点**

创新点在于：①源一致性分级信任（SAGT）半监督预训练，充分利用少量标注与海量无标注数据；②将视觉模型与视觉‑语言模型（VLM）融合并通过LoRA微调实现对陆滑场景的语义理解；③构建首个含图像-文本标注的陆滑基准数据集LandslideVL

**🔧 技术方法**

采用SAGT框架、基于Vision Transformer的主干、SAM视觉分割器、DINOv2教师-学生对齐、LoRA微调的VLM，以及多种评估指标（mIoU、mF1、BLEU、ROUGE等）

**📊 数据集**

使用BJL、GVLM、LandslideVL三大公开陆滑数据集；LandslideVL进一步结合文本标签与描述进行多模态训练

**📈 对比分析**

与多种现有视觉模型（ResNet, DenseNet, ViT, DINOv2, FixMatch, UniMatch, AllSpark, PF‑SAM）及通用VLM（Qwen2.5‑VL‑3B/7B, Qwen3‑VL‑8B, Kimi‑VL‑A3B）进行对比，LandslideFM在分类、检测、分割任务上均优于对手；在LandslideVL的结构化文本与详细描述生成上，其精确率、召回率、F1以及BLEU等指标显著高于其它VLM，尤其在召回率提升显著

**⚠️ 局限性**

局限包括：仅聚焦可见光遥感数据，未深入利用多光谱/合成孔径雷达等复杂光谱；缺乏针对极端灾害情形（如高速滑坡）的大规模标注；在多模态推理深度与通用性上仍有提升空间

---

## 100. DOBB: Dynamic Oriented Bounding Boxes for Multi-Agent Collision Avoidance under State Uncertainty and Heterogeneous Dynamics

**arXiv ID:** 2611.13142 | [PDF](https://arxiv.org/pdf/2611.13142v1)

**作者:** Pedro Antonio Alarcon Granadeno `[一作]` (University of Notre Dame), Jane Cleland-Huang `[通讯]` (University of Notre Dame)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出动态定向包围盒（DOBB）作为在线几何表示，用于在多智能体去中心化碰撞避免中将状态估计不确定性、代理异质性与动态安全边距相结合。

**💡 创新点**

创新点在于将不确定性与代理响应直接映射为方向相关的动态安全边距，减少政策对这些因素的隐式推断，从而显著提升在训练分布外的鲁棒性。

**🔧 技术方法**

主要技术包括：DOBB构建与更新、几何交互特征提取、基于几何的奖励设计、受限动作执行模型、EKF状态估计以及在GA3C‑CADRL框架下的强化学习。

**📊 数据集**

使用在模拟环境（Env4）中随机采样的代理属性、位置/速度噪声与10人测试场景的数据集；未采用公开真实数据集。

**📈 对比分析**

与传统基于经验训练和显式运行时不确定性表示方法对比，采用500个10人测试场景评估，DOBB在训练分布内外的平均总失败率仅为2.6%，碰撞失败率≤2.3%，明显优于基线。

**⚠️ 局限性**

在极高不确定性下，安全边距过大占用过多自由空间，导致超时失败率上升，表现为安全与进展之间的权衡。

---

## 101. When to Start Denoising? Adaptive Denoising Entry for Generative Video Compression

**arXiv ID:** 2611.13620 | [PDF](https://arxiv.org/pdf/2611.13620v1)

**作者:** Zifu Zhang `[一作]` (Beihang University), Ying Chen `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种自适应扩散式视频压缩框架ADE‑GVC，能够根据压缩后潜在的失真程度动态选择去噪起始时间点，从而在低比特率下兼顾重建精度与感知真实感。

**💡 创新点**

创新点在于将去噪起始时间点视为几何优化问题，给出闭式解并证明其随失真单调递减；设计轻量级时间步预测器实现自适应起始点；引入时间适配器将压缩感知条件注入预训练的VideoDiT模型，实现更佳的图像纹理与时序一致性。

**🔧 技术方法**

核心技术包括预训练的VACE/VideoDiT扩散模型、VAE潜在编码、时间步预测器（TP）、时间适配器（TA）、多阶段训练策略（潜在域→像素域→时间步预测），以及基于BD‑Rate的性能评估。

**📊 数据集**

训练集使用Pexels高质量视频约48万段；评估集为HEVC Class B、UVG和MCL‑JCV三大标准视频数据库。

**📈 对比分析**

与传统VTM‑17.0、学习型DCVC-FM/RT、生成式PLVC、GLC‑Video、GNVC‑VD、S²VC等方法对比，ADE‑GVC在LPIPS、DISTS、PSNR等指标上均领先；相较GLC‑Video在BD‑Rate上分别节省约41%（DISTS）、95%（PSNR）和21%（LPIPS）。实验显示在文本与纹理细节恢复以及时序一致性方面效果显著优于同类方法。

**⚠️ 局限性**

局限性主要在于采样步骤仍固定，未实现全自适应采样；扩散模型推理速度仍高于传统编码器；对极端压缩损失的鲁棒性尚待进一步验证。

---

## 102. Reasoning-Aware Relational Representation Learning for Open-Vocabulary Scene Graph Generation

**arXiv ID:** 2611.13496 | [PDF](https://arxiv.org/pdf/2611.13496v1)

**作者:** Jiawei Xiao `[一作]` (ShanghaiTech University), Xuming He `[通讯]` (ShanghaiTech University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过把多模态大语言模型的链式推理（CoT）转化为结构化关系嵌入，实现了开词汇场景图生成。

**💡 创新点**

创新点在于将CoT视为关系表征，并通过自监督对比学习对齐CoT与关系文本，再利用该嵌入空间构造奖励进行强化学习，从而显著提升关系判别性与泛化能力。

**🔧 技术方法**

采用的技术包括多模态大语言模型（Qwen3‑VL），自监督对比学习（InfoNCE），基于嵌入的奖励的策略优化（GRPO），以及文本原型检索的嵌入匹配。

**📊 数据集**

实验数据集为 Visual Genome（VG150）和 PSG，在 PredCls 与 SGDet 两种评估模式下进行评测。

**📈 对比分析**

与现有 OV‑SGG 方法相比，RaRe 在 mR@K（尤其是 Novel 和长尾类别）上取得显著提升，表现出更均衡的召回率，整体性能优于或接近最新基线。

**⚠️ 局限性**

局限性包括：对大语言模型的视觉幻觉依赖仍未完全消除；在检测噪声大或不同 proposal 设置下，提升不够稳定，尤其是对新关系的识别仍受限。

---

## 103. ExRL: Sample-Efficient Online Adaptation of Long-Chunk Policies via Execution Length Control

**arXiv ID:** 2611.13482 | [PDF](https://arxiv.org/pdf/2611.13482v1)

**作者:** Xu Luo `[一作]` (UESTC), Jingkuan Song `[通讯]` (Tongji University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为ExRL的在线适配框架，利用可学习的执行长度决策来改进预训练的长动作块策略，而不是直接修改动作序列；

**💡 创新点**

核心创新在于将高维动作块的在线适配问题转化为低维执行长度的决策，从而显著提升采样效率；

**🔧 技术方法**

采用离线/在线强化学习（SAC、Q学习）结合离线演示数据和在线行为克隆进行训练，使用SMDP风格的时序价值更新；

**📊 数据集**

在仿真Benchmarks（Ant, Adroit, Sawyer）和真实机器人任务（立方体堆叠、衣物折叠、插槽装配、零件整理）上进行实验；

**📈 对比分析**

与固定执行长度、随机k、AAC、在线BC、以及多种基于残差或完整微调的RL基线进行比较，ExRL在多种任务上实现近乎100%成功率，且样本效率显著优于对比方法；

**⚠️ 局限性**

局限性包括：实验中基准策略保持冻结，未充分利用在线行为克隆导致部分任务仍不完美；未评估多任务或跨任务泛化；未结合异步推理/实时分块等技术。

---

## 104. Dark Matters: VAE-Aware HDR Encoding for SDR-to-HDR Video Reconstruction

**arXiv ID:** 2611.13504 | [PDF](https://arxiv.org/pdf/2611.13504v1)

**作者:** Donghoon Cha `[一作]` (STUDIO FREEWILLUSION Inc.), Hanwool Sul `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出EquivFlow HDR编码，实现VAE感知的曝光等变换，保持暗区色度不丢失；

**💡 创新点**

设计可逆曝光等变编码，将曝光变化映射为代码空间平移，同时保留尺度独立的色度分量；

**🔧 技术方法**

采用可逆曝光等变编码、冻结的Latent Diffusion模型与IC‑LoRA适配器；

**📊 数据集**

使用960×544的5,780个裁剪样本（Tears of Steel、PolyHaven），在ARRI、HDRTV4K、UPIQ数据集上评测；

**📈 对比分析**

与LogC3、LumiVid、HDRTVNet++、LEDiff对比，EquivFlow在色彩保真（ΔC_TC_P、Δu'v'等）上显著优于同类方法，PSNR保持相当或更好；

**⚠️ 局限性**

需针对每个预训练VAE拟合参数，跨VAE迁移受限；对更多VAE家族或更广场景的鲁棒性尚未充分验证。

---

## 105. Thermodynamically consistent neural constitutive models for elastoplasticity via eigen-representations

**arXiv ID:** 2611.13162 | [PDF](https://arxiv.org/pdf/2611.13162v1)

**作者:** Harikrishnan Vijayakumaran `[一作]` (Delft University of Technology), Karl A. Kalina `[通讯]` (Technische Universität Dresden)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一个热力学一致、可微的神经网络塑性模型，能够同时学习可变弹塑性界面和演化规律，

**💡 创新点**

通过谱分解与置换不变的输入，直接在网络层面实现材料等变性与凸性约束，保证模型在未见加载状态下的热力学合法性；利用隐式微分与退化安全方案解决返回映射中的特征值退化问题；

**🔧 技术方法**

采用输入约束的ICNN网络、谱特征值平均、软化激活与隐式梯度推导（implicit function theorem）以及JAX/Optimistix等自动微分框架；

**📊 数据集**

使用合成的经典塑性模型数据、由标准与拓扑优化试样产生的有限元路径数据、以及开放细胞泡沫RVE的宏观同化数据；

**📈 对比分析**

与传统解析塑性模型、三角剖分网络以及序列模型基线对比，模型在已知模型的复现、等变性增强、以及非关联泡沫RVE的预测均达到了1–2 %级别的NRMSE，显示出优越的精度与泛化能力；

**⚠️ 局限性**

局限于等变、无率依赖、线性小变形，且仅对等变材料；未来需扩展至各向异性、有限变形及耦合损伤等更一般的塑性框架。

---

## 106. Latent Barrier Steering: Hierarchical Safety for Generative Planning

**arXiv ID:** 2611.13227 | [PDF](https://arxiv.org/pdf/2611.13227v1)

**作者:** Renhao Zhang `[一作]` (University of Massachusetts), Bruno Castro da Silva `[通讯]` (University of Massachusetts)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了Latent Barrier Steering (LBS)，一种两阶段安全规划框架，在生成器行为潜在空间先进行安全导向的优化，然后再在路径空间使用控制障碍函数（CBF）进行微调。

**💡 创新点**

创新点在于将行为潜在空间与路径空间分离，利用潜在空间的几何对齐性在生成之前就选择更安全、更易于校正的行为，从而显著降低后续路径修正能量并提升任务完成率。

**🔧 技术方法**

技术主要包括参数化流匹配生成器、几何正则化的行为潜在编码、基于潜在空间的软最小安全边距目标以及在潜在空间和路径空间分别求解的CBF二次规划。

**📊 数据集**

实验使用了多模态导航、迷宫导航和Franka机械臂末端执行器三类任务的数据集，涵盖二维点导航、三维空间导航和机器人操作等场景。

**📈 对比分析**

与传统的Vanilla FM、SafeDiffuser、SafeFlow以及SafeFlowMatcher等基线相比，LBS在保持100%安全率的同时显著提升了达标率（高达90–100%）并将CBF干预能量降低到最小，整体性能最优。

**⚠️ 局限性**

局限性包括：对潜在空间几何正则化的依赖需要在训练时充分学习，潜在梯度估计采用零阶差分导致运行时开销较大，且对完整轨迹跟踪误差和机器人全身碰撞的安全保证尚未覆盖。

---

## 107. Elicitation of the parameters of a multi-criteria sorting model

**arXiv ID:** 2611.13062 | [PDF](https://arxiv.org/pdf/2611.13062v1)

**作者:** Eda Ersek Uyanik `[一作]` `[通讯]`, Eda Ersek Uyanik

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用，旨在提高效率和准确性。

**💡 创新点**

创新点在于提出了一种新的优化策略，能够在处理大规模数据时显著减少计算时间。

**🔧 技术方法**

使用了深度学习技术和强化学习算法相结合的方法。

**📊 数据集**

采用了公开的图像识别数据集进行实验，数据集包含数万张标注图像。

**📈 对比分析**

与现有的几种主流算法进行了比较，结果显示该算法在准确率和速度上均优于其他方法。

**⚠️ 局限性**

限制在于算法对特定类型数据的适应性较差，可能无法广泛应用于所有领域。

---

## 108. Agreeing on the Answer but Not the Reason: Measurement Limits of LLM Reasoning Failure Taxonomies

**arXiv ID:** 2611.13140 | [PDF](https://arxiv.org/pdf/2611.13140v1)

**作者:** Madhu Sudhan Rao Beesetty `[一作]` `[通讯]` (PayPal), Madhu Sudhan Rao Beesetty (PayPal)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在多语言数学问答基准上，使用LLM审计器对“Compositional Collapse”这一推理失败类别进行评估，并对其测量可靠性和结果进行深入分析。

**💡 创新点**

创新点在于揭示失败类别分布易受测量流程（检验顺序、提示语）影响，并证明基准翻译缺陷会导致错误归因，提出四项低成本检查以保证失败率比较的有效性。

**🔧 技术方法**

技术上采用两阶段管道：生成阶段使用多种LLM，检验阶段用固定的Claude Sonnet 5审计器；同时引入人类评估、Cohen κ、PABAK、Gwet AC1等统计方法验证一致性。

**📊 数据集**

使用的公开数据集是 mGSM‑Symbolic（8种结构、11种语言、共2,316条记录），涵盖多语言数学推理任务。

**📈 对比分析**

通过对12个模型的失败率进行比较、对检验顺序进行置换、对人类评估结果与自动审计器对比，并在不同提示下重新评估，发现答案正确性一致性高（κ≈0.9），但对失败类别的一致性低（κ≈0.1），并识别出翻译缺陷导致的过度归因。

**⚠️ 局限性**

限制包括人类评估样本规模小、评估者与语言完全混淆、面向单一推理失败类别、仅针对数学词题、缺少英语基准、审计器与模型同属一系列、提示不对称以及未对翻译缺陷进行全面检测。

---

## 109. CapBuddy: Efficient Physical Memory Management for Capability-based Resource-controlling Microkernel

**arXiv ID:** 2611.13036 | [PDF](https://arxiv.org/pdf/2611.13036v1)

**作者:** Zhuang Lu `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Rui Hou `[通讯]` (Institute of Information Engineering, Chinese Academy of Sciences)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 CapBuddy，一种在基于能力的微内核（如 seL4）上运行的高性能用户级 buddy 内存管理器，显著降低了能力系统带来的延迟。

**💡 创新点**

创新点在于引入了 vzone 抽象，将物理页映射与能力操作分离，并采用 split‑design 的异步预置与批处理机制，将耗时的能力检查从前端关键路径中剥离，最终实现近 Linux 级别的性能。

**🔧 技术方法**

利用了 seL4 的 capability 体系、虚拟 zone(vzone)结构、红黑树管理、位图快查、后台 daemon 异步预置、事件驱动通知、以及批量槽分配等技术。

**📊 数据集**

实验使用 Raspberry Pi 3B+（ARM Cortex‑A53）平台，基准包含单页/批量分配与释放、RAMspeed、ELF 加载、SQLite3 YCSB 工作负载等，且与 Linux、Unikraft 及原始 seL4 Buddy 进行对比。

**📈 对比分析**

通过与 Linux Buddy、Unikraft Buddy、seL4 Buddy 等系统的对比实验，CapBuddy 在单页分配/释放上提升 8.9~54.5 倍，在批量分配/释放上接近 Linux，整体内存操作（分配+映射/释放+解除映射）提升 2-4 倍，SQLite3 性能提升约 39%，达到 Linux 的 75% 以上。

**⚠️ 局限性**

局限性包括：仍受 LibOS 兼容层的 300‑400 周期开销；在多客户端场景下池失效率与尾延迟上升；目前仅实现单客户端/单 daemon 模式，未充分支持多核并发；以及对非页级对象的支持有限。

---

## 110. PUMA: Probability-Constrained Sequential Interaction for Forensic Localization

**arXiv ID:** 2611.13621 | [PDF](https://arxiv.org/pdf/2611.13621v1)

**作者:** Zhouli Yu `[一作]`, Mengyuan Zhou `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种 PUMA 模型，利用空间受限的无平衡最优传输与概率约束的顺序传播，融合语义与取证特征，实现图像操纵的像素级定位。

**💡 创新点**

创新点包括：① 引入无平衡最优传输保留质量不匹配信息；② 通过位置级门控自适应选择跨分支传输与内分支聚合的执行顺序；③ 将质量残差作为门控信号，保持行随机化并通过残差路径更新噪声特征。

**🔧 技术方法**

采用无平衡最优传输、Softmax/归一化改进、门控的概率约束顺序传播、残差更新、GELU+Sigmoid MLP、Dice 损失以及 Grad‑CAM 可视化等技术。

**📊 数据集**

使用 CASIA1.0、COVERAGE、Columbia、NIST16 四个公开数据集，并在这些数据集上加入 JPEG 压缩、高斯模糊、高斯噪声和伽马校正等四种后处理扰动进行评估。

**📈 对比分析**

与 MVSS‑Net、PSCC‑Net、IML‑ViT、CAT‑Net、TruFor、Mesorch 等代表性方法进行对比；PUMA 在所有四个数据集上均获得最高像素级 F1 分数，尤其在 COVERAGE 与 NIST16 上显著提升，并在多种后处理条件下保持更稳健的性能。

**⚠️ 局限性**

局限性包括：在极端 JPEG 压缩或大尺寸模糊下的适应性略弱；模型相对复杂，计算成本较高；仍依赖两分支架构，未能直接处理无参照取证场景。

---

## 111. XAI-Driven Confidence Estimation for OOD-Resilient Runtime Prediction

**arXiv ID:** 2611.13199 | [PDF](https://arxiv.org/pdf/2611.13199v1)

**作者:** Mahmoud Sanad `[一作]`, Jörg Henkel `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种后置 XAI 框架，利用训练阶段的嵌入空间与高斯混合模型（GMM）为机器学习资源管理模型在运行时提供输入特异性的置信度估计，帮助检测并避免 OOD 输入导致的灾难性预测；

**💡 创新点**

创新点在于首次将 GMM 聚类嵌入空间与基于每个簇的误差上界相结合，实现无需重训练、可在线实时的输入级置信度推断；

**🔧 技术方法**

主要技术包括自编码器/中间层提取嵌入、GMM 聚类、簇级误差查找表、以及运行时置信区间输出；

**📊 数据集**

使用了 Intel Alder Lake 24 核 CPU 上收集的 219K 个窗口样本，覆盖 PARSEC、SPLASH‑2、SPEC CPU2006 三组基准；

**📈 对比分析**

与使用全局最大、均值或 90% 分位数误差阈值的基线相比，所提方法在 34 个基准和多种硬件配置下实现了约 94% 的包含率，并使置信区间平均宽度比全局基线窄 17%，运行时开销可忽略；

**⚠️ 局限性**

局限性包括需要额外训练嵌入与 GMM 以及在簇为空或完全新颖的 OOD 区域时可能失效，对动态硬件状态或新的应用特征适应性有限。

---

## 112. FlashMotion: Ultra-Fast Long-Term 3D Motion Generation

**arXiv ID:** 2611.13626 | [PDF](https://arxiv.org/pdf/2611.13626v1)

**作者:** Kaixing Yang `[一作]` (Renmin University of China), Jun He `[通讯]` (Renmin University of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出FlashMotion，一种在严格推理预算下即可生成高质量长时序3D人类运动的方法。

**💡 创新点**

创新点在于同时优化轨迹级采样效率与每步计算效率，采用3D运动空间MeanFlow与Adaptive Soft mask实现少步采样，并设计轻量级BiMamba+AdaIN的非自回归架构。

**🔧 技术方法**

技术包括MeanFlow流匹配、条件引导状态选择、AdaIN对齐、BiMamba时序模型、Adaptive Soft mask、运动的kinematic/dynamic约束。

**📊 数据集**

使用文本、语音、音乐三种条件的公开数据集：BABEL、BEAT2、FineDance（以及对应的文本/音频/音乐特征）。

**📈 对比分析**

与多种基线对比（如MDM、DoubleTake、FlowMDM、MoMask、BAMM等），在R@1、FID、DIV、FPS等指标上均取得SOTA；1步可达约30fps，4步可达70fps。

**⚠️ 局限性**

局限性：对极长序列仍受模型规模限制，Adaptive Soft mask仅支持多步编辑，未提供单步编辑；跨模态适配需额外训练。

---

## 113. Robust Flow Matching under Target Corruption and Label Noise

**arXiv ID:** 2611.13324 | [PDF](https://arxiv.org/pdf/2611.13324v1)

**作者:** Mert Can Kurucu `[一作]` (KTH Royal Institute of Technology), Hossein Azizpour `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对流匹配（Flow Matching）在目标数据受损与标签噪声两种现实腐败情形下的鲁棒性问题，提出了两种基于残差的样本加权策略：Temporal Residual Score（TRS）与 Comparative Conditional Score（CCS），并将其嵌入权重流匹配目标中；

**💡 创新点**

创新点在于：①首次将目标与标签腐败分别建模为流匹配回归问题中的时间相关残差结构；②基于该分析设计针对性残差评分——TRS利用时间偏差残差分布，CCS利用跨条件残差比较，进而实现无监督的样本重加权；

**🔧 技术方法**

核心技术包括：流匹配中的条件向量场回归、残差加权学习（采用指数移动平均与软阈值映射）、时间偏差采样（t=√ξ）、跨标签残差比较与温度控制；

**📊 数据集**

实验数据集涵盖：2D 旋涡图案、CIFAR‑10（目标腐败与标签噪声）、CIFAR‑100（大规模条件空间）以及WebVision 50‑class 真实标签噪声；

**📈 对比分析**

与标准FM、L1/Huber鲁棒损失、DeepGUM、DivideMix等基线相比，TRS在目标腐败场景下将Wasserstein-2与FID下降至接近干净数据水平；CCS在标签噪声与WebVision任务中显著提升类对齐指标与类均匀覆盖度，整体性能优于所有对照方法；

**⚠️ 局限性**

局限性包括：TRS依赖残差与时间分离度高，可能误下权重合法样本；时间偏差采样改变了优化动态；CCS需要可区分的条件损失并在每次刷新时产生 O(N·|C|·|T|) 的计算开销；实验主要集中在控制腐败与双重腐败的 2D 场景，未覆盖极端噪声、严重类别不平衡或扩展到扩散模型等情形。

---

## 114. When Reasoning looks like Hidden Understanding: How Chain-of-Thought Fosters Anthropomorphism, Perceptions of Epistemic Superiority, and Cognitive Deference

**arXiv ID:** 2611.13456 | [PDF](https://arxiv.org/pdf/2611.13456v1)

**作者:** Oliver Lack `[一作]` (Australian Institute for Machine Learning and School of Psychology, Adelaide University), Melissa McCradden `[通讯]` (Australian Institute for Machine Learning, Adelaide University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了链式推理（CoT）在生成式 AI 中被误解为透明解释的现象，并探讨了这种误解如何导致人类对 AI 的过度信任和认知偏差。

**💡 创新点**

提出将 CoT 的“人类化”表征视为认知转移的工具而非原因，并指出其对伦理与临床决策的潜在风险，强调 CoT 需要更严谨的可信度评估。

**🔧 技术方法**

主要采用了对已有文献、伦理框架与认知科学理论的综述与理论分析，未进行新的实验或模型训练。

**📊 数据集**

未使用具体数据集，全文基于文献回顾与理论推演。

**📈 对比分析**

无实验对比，作者通过对比 CoT 与传统后置解释的理论优势与不足，论证 CoT 在高风险场景下不宜直接作为解释。

**⚠️ 局限性**

局限在于缺乏实证验证，研究仅停留在概念层面，未能提供具体方法或指标来评估 CoT 的可信度与效用。

---

## 115. Closing the BF16-FP16 Convergence Speed Gap in RL Post-Training with Stochastic Rounding

**arXiv ID:** 2611.13545 | [PDF](https://arxiv.org/pdf/2611.13545v1)

**作者:** Mahdi Nazemi `[一作]` (NVIDIA), Carlo del Mundo `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析BF16在低学习率强化学习中的收敛问题并提出解决方案。

**💡 创新点**

提出在量化FP32权重到BF16时使用随机舍入（SR），显著缓解权重粘滞导致的慢收敛。

**🔧 技术方法**

采用随机舍入、混合精度训练、Dr. GRPO、vLLM、TorchInductor等技术。

**📊 数据集**

使用MATH‑1460、DAPO‑Math‑17k以及AIME 2024/2025数据集，测试DeepSeek‑R1‑1.5B、14B与Qwen3‑30B‑A3B模型。

**📈 对比分析**

与FP16和标准BF16对比，BF16+SR在相同学习率下收敛速度与FP16相当，且KL漂移更低；标准BF16收敛慢，提升LR可加速但漂移增大。

**⚠️ 局限性**

在更低位宽格式下权重粘滞可能更严重，实验未覆盖所有vLLM版本，且FP16在极大模型上仍可能出现溢出。

---

## 116. FaiSonic MMD: A popular music dataset with temporally aligned multidimensional annotations for multitask music analysis

**arXiv ID:** 2611.13503 | [PDF](https://arxiv.org/pdf/2611.13503v1)

**作者:** Tieyao Zhang `[一作]` (Zhejiang Conservatory of Music), Genfang Chen `[通讯]` (Zhejiang Conservatory of Music)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了FaiSonic Multidimensional Music Dataset（FaiSonic MMD），包含500首流行歌曲（250中文+250英文）并由专业音乐人标注句段、乐段、和弦与节奏，随后构建了一个多任务基线模型实现四维时间事件的联合预测。

**💡 创新点**

创新点在于：① 同一录音上提供四维时间注解并经过双专家复核，保证注解一致性与时序精确；② 通过共享录音时间基准实现不同粒度注解的可直接对齐；③ 提供完整的多任务基线与评测框架，为跨维度关系分析与联合建模提供基础。

**🔧 技术方法**

技术方法包括：使用MuQ和MusicFM提取的预训练音频特征；基于Transformer、Conformer、Tcn等网络结构的多任务编码器；结构化解码（Viterbi、动态规划、软阈值）与多任务损失组合；自动化检查与双专家审校流程。

**📊 数据集**

使用的数据集为：主数据集FaiSonic MMD（500首，约32.86小时），以及扩展训练时加入的2350首已有专家标注的歌曲。

**📈 对比分析**

评估方法：在固定的训练/验证/测试划分下，分别用F1（0.5s/3s）、ACC、mir_eval（Root、Majmin、Thirds、Triads、Sevenths、Tetrads、MIREX）等指标；基线在各任务上得到62.9–86.2%的F1或ACC，扩展训练后可提升1–5个百分点。

**⚠️ 局限性**

局限性：规模仍有限，主要聚焦中文与英文流行音乐；对其他语言、风格或录音条件的适用性尚未验证；未来需扩大覆盖面并深入分析跨域性能。

---

## 117. Multi-Future Drafting: Mixture Block Drafters for Tree-Verified Speculative Decoding

**arXiv ID:** 2611.13017 | [PDF](https://arxiv.org/pdf/2611.13017v1)

**作者:** Haoyu Huang `[一作]` (Beihang University), Baochang Zhang `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Multi‑Future Drafting，利用多模态嵌入在一次前向推断中生成多条未来，构建混合树进行树验证的推测式解码。

**💡 创新点**

创新点在于通过潜在变量恢复未来间的耦合，避免产品分布导致的“chimera”问题，使用混合分布树验证，仅增少量参数且一次推断即可显著提升速度与接受率。

**🔧 技术方法**

采用一通前向块式草稿器、模式嵌入、混合分布、最佳模式得分、贪心树搜索、门控训练、最优覆盖损失与多选学习等技术。

**📊 数据集**

在 Qwen3 系列大模型（4B/8B/4B+/27B）上训练，使用 GSM8K、MATH‑500、HumanEval、MBPP、MT‑Bench 等标准评测集，以及 1.42M 的目标再生成训练数据。

**📈 对比分析**

与单模态树、DARTree、JetSpec、EAGLE‑3 等基线在 T=0/1、batch‑1 等配置下比较，64/256 节点下平均提升 5–8%（单模态树）或 9–16%（DARTree）加速；在 SGLang 引擎中比最快发布草稿器快 45–46%，接受长度提升约 8 tokens/round，树覆盖率提高 8–15%。

**⚠️ 局限性**

局限性包括对树节点预算和混合模式训练数据的依赖，批量化或多请求时性能下降；混合树仍保留一定比例的 chimera 节点；潜在变量的使用需额外门控训练，且在某些推断场景下可能不适用。

---

## 118. Mapping Instructors' Attitudes Towards Cheating with AI: An Exploratory Analysis Using Forum Analysis

**arXiv ID:** 2611.13366 | [PDF](https://arxiv.org/pdf/2611.13366v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 119. Greedy+Singleton for knapsack-constrained monotone submodular maximization has approximation ratio exactly $β^\star \approx 0.42944$

**arXiv ID:** 2611.13575 | [PDF](https://arxiv.org/pdf/2611.13575v1)

**作者:** Kundan Baliga `[一作]` (University of Illinois Urbana Champaign), Aditya Byju `[通讯]` (Georgia Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文对单调子模函数在单一背包约束下的经典贪心求解方法（按单位成本收益率挑选，若需要则与最佳单个项目比较）进行严格分析，给出了其最坏情况逼近比的确切常数，并提供了对应的极端实例。

**💡 创新点**

创新点主要有：①通过因子揭示松弛与区间算术/分支限界相结合，突破了此前仅给出区间 [0.427,0.42945] 的界限，得到闭式常数 β≈0.428；②给出了无约束的极端三项实例并证明其最优性；③进一步分析了多种算法变体（停止贪心、CELF、密度指数非1）并给出它们的精确逼近比。

**🔧 技术方法**

所用技术包括因子揭示松弛（factor‑revealing relaxation）、凸/指数分析、极值求解、区间算术（interval arithmetic）与分支限界（branch‑and‑bound）以及整数算术计算；同时利用极限分析证明极端实例逼近该常数。

**📊 数据集**

研究主要基于构造的合成实例（极端三项实例、填充器实例等），没有使用真实数据集；所有实例均可在附录的代码和数据中复现。

**📈 对比分析**

通过与之前的下界/上界结果比较，本文证明该贪心+最佳单项算法的最坏逼近比为 β≈0.428，严格高于之前的 0.427；对停止贪心得到 1‑e^{-β_W}≈0.354，CELF 得 0.5(1‑1/e)≈0.316，指数非1 的贪心则无常数保证。实验与手算验证了所有实例的精确性。

**⚠️ 局限性**

局限性包括：①结果需要计算机辅助（区间算术与分支限界），不构成完全手算证明；②仅适用于单一背包、单调子模函数；③未覆盖非单调、多约束或近似值评估等情况；④极端实例逼近常数但未证明唯一性，构造速度慢。

---

## 120. Cyclically covering subspaces and additive perfect codes

**arXiv ID:** 2611.13044 | [PDF](https://arxiv.org/pdf/2611.13044v1)

**作者:** Yangcheng Li `[一作]` (South China Normal University), Pingzhi Yuan `[通讯]` (South China Normal University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了有限域上子空间的循环覆盖与其对应的向量字母码，提出了正覆盖余维与码参数之间的等价条件，并给出了关于循环完美码存在的算术判据。

**💡 创新点**

以循环核心为核心，将正余维编码为周期码，给出了满足1覆盖半径、无单权重字母、标量对偶一生成的充分必要条件；并对长度 $n=N_{q,k,m}$ 给出存在循环完美码的必要且足够的条件 $(m,q-1)=1$，进一步在边界长度 $q^k+1$ 时给出等价条件与完美码的存在。

**🔧 技术方法**

结合循环子空间理论、向量字母码的同构与对偶、正则映射、符号判据、模多项式与扩域的范数构造、符号分析与字符求和、组合论工具等，使用规范化同态构造完美码并计算其范数。

**📊 数据集**

本文为纯理论研究，无实验数据集。

**📈 对比分析**

通过证明计数上界被达到，给出 $h_q(n)$ 的确切值并与已有下界匹配；与现有构造方法对比，证明在特定参数下实现最优覆盖余维。

**⚠️ 局限性**

对非素数 $k$ 且 $m$ 较小的情况仍未完全覆盖；存在未解决问题：1) 是否对所有满足 $(m,q-1)=1$ 的参数都能构造循环完美码；2) 当 $(m,q-1)>1$ 时 $h_q(n)$ 是否严格小于 $k(m-1)$。

---

## 121. Approximating graphic TSP by b-matchings

**arXiv ID:** 2611.13585 | [PDF](https://arxiv.org/pdf/2611.13585v1)

**作者:** Swati Gupta `[一作]` (MIT), Levent Tunçel `[通讯]` (University of Waterloo)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在图形旅行商问题（Graphic TSP）中提出了新的多项式时间算法，给出长度不超过 89/64（约 1.3906） 的旅行路程；

**💡 创新点**

创新点在于：①将 Mömke‑Svensson 的可删除配对推广为可删除星（Star removable system）；②通过在辅助图上构造带有自环的 b‑matching（即优化奇偶校正和连通性），实现对可删除边的更灵活使用；③结合子旅程（subtour）线性规划的最优极点，构造满足 Blossom 约束的可行点，从而得到更好的近似比；

**🔧 技术方法**

使用的技术主要包括：DFS 树构造的可删除星系统；基于奇偶校正的 T‑join 多项式可行域（含自环、连接边的整数多匹配）；子旅程线性规划与 Blossom 约束的关系分析；凸组合与参数化构造（Q(d)）来降低额外代价；以及对特殊图类（4‑regular、4‑连通、子四度图等）的进一步分析；

**📊 数据集**

无数据集，研究完全基于理论证明与数学优化；

**📈 对比分析**

与以往最佳 7/5（1.4）的近似比相比，取得了 89/64（≈1.3906）的提升；在 4‑regular、4‑连通图上从 4/3 提升至 13/10（1.3）；在最大度 ≤ k+1 的 k‑连通图上从 7/5 提升至 4/3；此外，Lean 形式化证明表明可进一步到 14524/10481≈1.38575 等；

**⚠️ 局限性**

局限性包括：仍未达到图形实例的 4/3 猜想；算法依赖于 DFS 树的选择，实际实现效率和常数因子未给出；仅在理论层面验证，缺乏实验评估；

---

## 122. A Fast Algorithm for Subspace Convolution

**arXiv ID:** 2611.13134 | [PDF](https://arxiv.org/pdf/2611.13134v1)

**作者:** Shlomo A. Perles `[一作]` (Hebrew University of Jerusalem), Omri Weinstein `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了快速子空间 Zeta 变换及其逆、归约、卷积等操作，并将其应用于 Grassmann 图的线性代数和低二进切分宽度图的图着色。

**💡 创新点**

创新点在于利用递归超平面切分和有限域傅里叶变换，将子空间求和问题从指数级 O(3^n) 降至 Θ_q(N_q log N_q)，并首次在子空间层面实现了子空间卷积和低维子空间求和，显著提升了相关算法的理论与实验性能。

**🔧 技术方法**

核心技术包括：递归超平面分解、有限域离散傅里叶变换、矩阵转置与逆变换、子空间计数与卷积、以及对有限域符号的环扩展。

**📊 数据集**

实验使用了规模从 1.2×10^6 到 1.6×10^8 条目、不同 q（如 2、3）的子空间表，以及对应的 Grassmann 图邻接矩阵和低二进切分宽度图样例。

**📈 对比分析**

与现有的覆盖对/覆盖乘法、Poisson 求和等方法对比，实验表明在 Zeta 变换上最快 18 倍、在交叉子空间求和上最快 190 倍、在 Grassmann 核乘法上最快 22 倍；相比之下，覆盖对算法在内存不足时甚至无法完成。

**⚠️ 局限性**

主要局限包括：常数因子较大，导致小规模实例时可能不占优势；对系数环要求 p 可逆，限制了某些域的直接应用；实现复杂度高，未对极大 q 或非平凡基数进行充分测试。

---

## 123. Domain adaptation of a video Foundation model for Heart Failure Diagnosis from Echocardiography

**arXiv ID:** 2611.13297 | [PDF](https://arxiv.org/pdf/2611.13297v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 124. Domain-Adversarial Adaptation from MURA to In-House Radiographs for Improved Adult Elbow Fracture Classification

**arXiv ID:** 2611.13308 | [PDF](https://arxiv.org/pdf/2611.13308v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 125. EJAI: Judgment-Preserving Civic AI for Data Center Environmental Justice

**arXiv ID:** 2611.13525 | [PDF](https://arxiv.org/pdf/2611.13525v1)

**作者:** Evan Pačes-Wiles `[一作]` (University of North Carolina), Snehalkumar 'Neil' S. Gaikwad `[通讯]` (University of North Carolina)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了 EJAI——一个源落地的多智能体可视化‑LLM 系统，用于数据中心环境正义争议中的证据检索、可视化与公众决策准备。

**💡 创新点**

创新点包括：提出“判断保留式市民 AI”设计理念，让 LLM 仅做证据检索与引用，保持公众判断；整合多源（政府指标、企业报告、地方证词）证据、地图可视化、缺失数据提示及两角色对话与辩论矩阵，实现公开、可检验的决策支持。

**🔧 技术方法**

技术实现：React+Node.js 前端；Azure Document Intelligence 与 DocumentDB 存储文档嵌入；检索增强生成（RAG）与多智能体对话；GPT‑4o mini、GPT‑4o、GPT‑6‑Astra 用于检索规划、生成、审议与评估；系统生成 inline 引用、缺失提示与决策矩阵。

**📊 数据集**

数据集：107 篇公开文档（77 企业环境报告、19 机构/倡导报告、6 学术论文、5 EPA 文档），44 条地方公众证词（梅克伦堡、洛登、奥兰治县），数据中心设施记录，以及美国县级环境与社会经济指标。

**📈 对比分析**

评估方法：技术基准 21 题测量检索覆盖、子问题覆盖、真实性、引用准确率和缺失提示率；两角色对话评估非坍塌与驳斥落地率。用户研究 15 人三条件对比（地图仅、单体 LLM、EJAI）。结果显示：真实性 98.4%（GPT‑4o）/90.6%（GPT‑6‑Astra），引用准确率 89%（GPT‑4o）/84%（GPT‑6‑Astra），缺失提示率 97%。在 EJAI 条件下，用户更频繁检验 AI 引用，制定条件性保障并提升议论准备度。

**⚠️ 局限性**

局限性：语料库代表性不足（企业资料占比高），导致缺失信息与偏见；两角色对话存在 grounding 成本，可能掩盖对立；地图缺乏与文本同等 provenance 标注；系统仅为准备工具，无法直接影响正式决策；依赖商业 LLM，资源消耗大，未验证小模型可替代方案。

---

## 126. ARoute: Task-Aligned Batch and Streaming Model Routing

**arXiv ID:** 2611.13209 | [PDF](https://arxiv.org/pdf/2611.13209v1)

**作者:** Daofu Zhang `[一作]` (University of Utah), Fuheng Zhao `[通讯]` (University of Utah)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种任务对齐的预生成模型路由器（ARoute），在给定成本预算下动态选择弱模型或强模型来最大化准确率。

**💡 创新点**

创新点在于使用少量配对记录同时学习‘仅强模型正确’概率与每个查询的增量成本，并通过冻结BERT编码器和两头网络实现低成本、可任务对齐的路由。

**🔧 技术方法**

技术包括冻结BERT编码器、线性OSC头、MLP成本网络、基于期望收益/增量成本排序、批处理与流式阈值选择，以及有限样本期望成本上界证明。

**📊 数据集**

使用了MMLU-Pro、LiveCodeBench、SuperGPQA以及基于SWAN的两个值填充任务。

**📈 对比分析**

在与RouteLLM、FrugalGPT、BARGAIN、OctoSelector等基线对比时，ARoute在15个任务–成本设置中平均提升4.1个百分点，单一设置最高提升11.9个百分点，并在批量执行上比级联快2.8倍。

**⚠️ 局限性**

局限性包括仅预测‘强模型唯一正确’事件，忽略弱模型错误被强模型纠正的情况；证书仅针对单一阈值，未覆盖动态阈值更新；以及需要在任务中先收集少量配对记录。

---

## 127. Parameterized Complexity and Geometric Approximation of Riesz Energy Subset Selection

**arXiv ID:** 2611.13600 | [PDF](https://arxiv.org/pdf/2611.13600v1)

**作者:** Michael Emmerich `[一作]` (University of Jyvaskylä), Soumen Atta `[通讯]` (University of Jyvaskylä)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了如何选择n个站点中的k个，以最小化Riesz s-能量，提出了一种高效的参数化近似方案（EPAS）。

**💡 创新点**

创新点在于通过几何打包和离散交换界限来提高能量估计的精度，特别是在平面情况下将代表性界限从O(k^2log(2k)/ε^2)降低到O(klog(2k)/ε^2)。

**🔧 技术方法**

使用了几何打包、最远优先遍历和离散交换界限等技术。

**📊 数据集**

使用了具有界限倍增维度的度量和紧凑域的数据集，包括填充简单多边形。

**📈 对比分析**

与其他方法相比，本文的方法在平面情况下的性能显著提升，尤其是在处理大规模数据时，能够有效减少搜索空间和计算复杂度。

**⚠️ 局限性**

限制在于在非限制性度量下，s=2的精确选择是W[1]-完全的，除非FPT=W[1]，否则无法实现确定性的EPAS。

---

## 128. Before Words, Beyond Speech: Evaluating Nonverbal Social Reasoning in Early Childhood

**arXiv ID:** 2611.13213 | [PDF](https://arxiv.org/pdf/2611.13213v1)

**作者:** Marie Amale Huynh `[一作]` (Stanford University), Serena Yeung-Levy `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个针对自然幼儿与照护者非语言交互的多模态视频基准NEST，并提出分层评估框架。

**💡 创新点**

引入四层组合式语义图谱（情境、行为、人际、开放）以细粒度拆解社交推理，并揭示当前视频VLM在行为与人际层面的显著性能崩溃。

**🔧 技术方法**

采用视频视觉‑语言模型、自动化干扰项生成（AutoConverter）、语言评估判定器、LoRA微调等技术。

**📊 数据集**

组合 ChildPlay、CHILDES、FIND 等多来源的自然幼儿互动视频，生成约12秒片段及 14k 问答。

**📈 对比分析**

对比多款公开与闭源视频‑VLM（如Qwen3、ViLD等），使用多选问答精度、开放式相似度及主张召回指标，发现情境识别高达80+%但行为/人际层仅50%及以下。

**⚠️ 局限性**

VLM在细粒度时间定位、因果后果推理、儿童-儿童交互以及开放式描述上存在显著欠缺，且需人工审阅与专用检测器配合才能满足实践需求。

---

## 129. IntentTwin: Deconfounding Candidate Support and the Oracle-to-Deployment Gap in Offline Web Behavior Prediction

**arXiv ID:** 2611.12918 | [PDF](https://arxiv.org/pdf/2611.12918v1)

**作者:** Miaobo Hu `[一作]` (University of Chinese Academy of Sciences), Daren Zha `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 767 | [OpenAlex ID](https://openalex.org/A5051666209)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文探讨了离线网络行为基准中的候选支持混淆问题，提出了一种支持平衡的候选集构建方法，并评估了一种无目标的双路径提议系统。

**💡 创新点**

创新点在于正式化了候选支持混淆的影响，并引入了支持可识别率（SIR）作为诊断工具，同时构建了支持平衡的候选集和无目标的双路径提议评估。

**🔧 技术方法**

使用了支持检索与开放词汇的DOM/LM提议相结合的双路径提议技术。

**📊 数据集**

使用了过滤后的OPeRA数据集（包含4,282/469/992个训练/开发/测试步骤）和Mind2Web数据集进行评估。

**📈 对比分析**

与原始协议相比，支持平衡减少了SIR至0.0000，最强的条件重排序器在整体上获得了0.7214的Exact@1，而无目标的双路径系统在端到端的Exact@1上达到了0.5212，显示出与oracle条件的Exact@1（0.7450）之间的差距。

**⚠️ 局限性**

限制在于OPeRA的主要分割是会话不重叠的，且确切的支持是一个词汇审计变量，Mind2Web审计主要测量候选覆盖和排名，而非在线浏览器交互。

---

## 130. Discovering Structurally Plausible and Interpretable Cognitive Models with Large Language Models

**arXiv ID:** 2611.13561 | [PDF](https://arxiv.org/pdf/2611.13561v1)

**作者:** Boyoung Kwon `[一作]` (KAIST), Sang Ho Lee `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于大语言模型的演化搜索框架DisCo，用于自动发现可解释的符号认知模型。

**💡 创新点**

创新点在于加入Validator和Cleaner两大结构约束，既过滤无效结构又提升参数可识别性，从而实现结构合理且可解释的模型。

**🔧 技术方法**

技术包括LLM（Qwen2.5-32B）作为变异/交叉算子、岛屿式演化搜索、逻辑验证、参数剪枝、BIC评估和外部认知测量相关性检验。

**📊 数据集**

数据集涵盖三种经典实验：两步强化学习任务、概率反转学习任务和识别记忆任务，分别涉及计划、学习与记忆三大认知领域。

**📈 对比分析**

与传统基准模型（模型自由/混合RL、TD学习、SDT）以及GeCCo比较，DisCo在BIC上始终更优，且在所有随机种子下可获得有效模型，说明其性能稳健。

**⚠️ 局限性**

局限性包括对基准模型的依赖、Validator需手工制定任务约束、对提示工程影响有限，以及对不同数据划分和评估策略的敏感性。

---

## 131. G-STAR: Group-Scaled Token Capacity for Deeply Compressed Autoregressive Generation

**arXiv ID:** 2611.13004 | [PDF](https://arxiv.org/pdf/2611.13004v1)

**作者:** Xu Huang `[一作]`, Daquan Zhou `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了GSTAR框架，利用空间–组共缩放和半自回归深度解码器实现 32× 甚至有效 64× 的空间压缩，同时保持高质量的图像生成与编辑。

**💡 创新点**

创新点在于：① 空间与组容量共同缩放的理论与实践，① 通过 2×2 无损打包实现有效 64×；② 轻量半自回归头，既保留组内依赖又显著降低解码时延。

**🔧 技术方法**

技术方案包括多码本量化（基于产品量化）、分层自回归生成、半自回归深度头、CUDA 图捕获与 Triton 核融合、以及 SGLang 后端集成。

**📊 数据集**

使用 ImageNet 预训练的视觉编码器，并在 5 个文本到图像基准（GenEval、DPG‑Bench、PRISM、LongText‑Bench、Qwen‑Image‑Bench）以及 GEdit‑Bench 进行评测。

**📈 对比分析**

与主流专有、扩散和离散自回归模型比较，GSTAR‑Pro‑8B 在 32× 压缩下与领先模型保持相近或略优的分数；GSTAR‑Flash‑1.7B 在 64× 压缩下实现约 20× 的生成速度提升，单张 1024×1024 图像平均仅需 2.5 秒。

**⚠️ 局限性**

局限性：对大规模预训练数据与显存需求高；组内依赖建模仍可能出现轻微多模态碰撞；在极高分辨率或更广泛的多模态任务上效果待进一步验证。

---

## 132. GRAP: Gender Rubric-Aware Probing for Analyzing Gender Bias in LLMs

**arXiv ID:** 2611.13024 | [PDF](https://arxiv.org/pdf/2611.13024v1)

**作者:** Seona Seo `[一作]` (Ewha Womans University), Sookyung Kim `[通讯]` (Ewha Womans University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了性别对比提示对，并利用10维社交评判维度对LLM生成的回应进行评分，随后训练轻量化逻辑回归探测器以识别不同性别条件下的偏差维度；

**💡 创新点**

创新点在于将性别偏差拆解为可解释的维度，并通过方差加权的逻辑回归探测器实现任务无关的低维度偏差识别；

**🔧 技术方法**

使用的技术包括性别对比提示生成、LLM文本生成、Rubric评判器、逻辑回归探测器以及AUC、准确率等评估指标；

**📊 数据集**

采用了1000对性别对比提示和10维社交评判维度的数据，评估四个通用LLM（如GPT、LLaMA等），并使用Prompt生成工具与Rubric评判器进行评分；

**📈 对比分析**

对比了全信息（prompt+response）与仅响应的两种评判设置，结果显示AUC在0.79-0.90之间，响应仅设置甚至更优，说明偏差信息已嵌入文本本身；

**⚠️ 局限性**

局限性包括仅使用10维评判维度可能无法覆盖所有细粒度偏差，评判器本身的固定设计可能引入偏差，并且研究仅聚焦于性别而未考虑交叉性别身份。

---

## 133. One Temperature to Rule Them All?

**arXiv ID:** 2611.13077 | [PDF](https://arxiv.org/pdf/2611.13077v1)

**作者:** Aviv Orly `[一作]` (Tel Aviv University), Yaron Oz `[通讯]` (Tel Aviv University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大语言模型推理时的采样温度分配，分析每个问题的温度概率曲线与难度关系，并提出一种高效的多温度采样分配方案。

**💡 创新点**

揭示温度曲线普遍呈单峰结构，证明难度对最优温度的影响，并通过两种“消退效应”解释为何单温度方案已能近似最优；给出可在离散温度集上求解的凸优化近似。

**🔧 技术方法**

采用pass@K评估框架、统计分析温度曲线、凸优化求解样本分配、理论证明（定理/引理）、实验验证四种LLM与三大基准。

**📊 数据集**

模型：Qwen3‑0.6B、Qwen3‑8B、Mistral 7B Instruct、GPT‑OSS 20B；基准：LiveCodeBench（V1–V6）、MATH‑500、Hi‑ToM。

**📈 对比分析**

与均匀多温度、单一固定温度对比；最优多温度方案比单温度提高约10%，但若温度随采样预算 K 调整，优势大幅缩小；在不同 K 下，单温度调优即可获得与最优方案相近的通过率。

**⚠️ 局限性**

仅适用于pass@K设置；假设温度曲线单峰且难度可用样本数衡量；对离散温度集的近似可能在极端模型或任务上失效；未考虑自适应或高维采样策略，需进一步验证。

---

## 134. MicroLens: Read, Trace, and Rewire the Residual Stream with Two Lenses

**arXiv ID:** 2611.13322 | [PDF](https://arxiv.org/pdf/2611.13322v1)

**作者:** Nika Zahedi `[一作]` (Stanford University), Anna Wu `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MicroLens，可视化并编辑Transformer残差流，帮助理解语言模型推理过程。

**💡 创新点**

将读取（lens）与因果编辑（patching、swap、copy、add）结合到同一网格上，并通过手动编辑验证中间状态的作用。

**🔧 技术方法**

利用Transformer residual-stream、两种lens（logit和Jacobian）、注意力回溯、四种编辑操作，配合Python+Hugging Face实现本地可视化。

**📊 数据集**

使用Gemma-3-4B-it模型（34层，2560维）及其公开的ES；实验采用多任务五示例提示，覆盖国都、公司、反义词、英语-法语等功能向量。

**📈 对比分析**

对比不同编辑方向的成功率，发现logit方向在推断答案时更有效，而Jacobian方向在修改推理内容时更好；在两跳回想实验中，swap编辑在手势层前可更改答案，复制单元达到上限。整体性能依赖于层次位置，实验显示编辑成功率可达70–90%。

**⚠️ 局限性**

局限包括仅测试Gemma-3-4B-it模型、单词单子、未评估多模型/多语言、平均方向对多义词效果差、注意力回溯近似、未在Scenario C中进行编辑验证，缺乏用户研究。

---

## 135. Capacity Allocation at the Source: Sparse Target Optimization for LLM Knowledge Editing

**arXiv ID:** 2611.12995 | [PDF](https://arxiv.org/pdf/2611.12995v1)

**作者:** Dongyao Chen `[一作]` (Renmin University of China), Xiaobo Sun `[通讯]` (Emory University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在知识编辑中对目标隐藏状态进行L1稀疏化的策略（STO），通过在上游构造目标时压缩残差向量，减少不同知识编辑之间的互相干扰。

**💡 创新点**

创新点在于将关注点从传统的参数更新阶段转向上游的目标构造，证明目标残差的内积与参数更新的干扰呈正相关，并通过L1正则实现残差稀疏化，从而降低干扰。

**🔧 技术方法**

技术方法包括：定位-编辑（Locate‑then‑Edit）范式、因果追踪定位决定性词/层、MEMIT框架下的目标优化、加入L1正则化实现残差稀疏化。

**📊 数据集**

使用的评测数据集为 Multi‑CounterFact（MCF）和 ZsRE，两者各包含约 2,000 条待编辑知识条目。

**📈 对比分析**

在 Qwen2.5‑7B‑Instruct、GPT‑J 6B 与 GPT‑2 XL 1.5B 上与 MEMIT、PRUNE、RECT、AlphaEdit、EAMET、RLedit、WISE 等方法对比，STO 在有效性（Success/Accuracy）、泛化性（Success/Accuracy）以及特异性（D_KL、Top‑k）三方面均表现出色，往往在每项指标上优于基线并实现更好的整体平衡。

**⚠️ 局限性**

局限性包括：仅在 7B 以下规模模型验证，缺乏 14B 以上模型的实验结果；部分基线在更大模型上缺乏可用超参数；实验受限于计算资源，未能进一步扩展至更大规模知识编辑。

---

## 136. Actionable Feedback: Improving Model Capabilities from User Edit Data on Clinical Documentation

**arXiv ID:** 2611.12904 | [PDF](https://arxiv.org/pdf/2611.12904v1)

**作者:** Daniel P. Jeong `[一作]` (Carnegie Mellon University), Michael Oberst `[通讯]` (Johns Hopkins University)

**通讯引用:** 445 | [OpenAlex ID](https://openalex.org/A5029248315)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建临床编辑分类体系并生成半合成的医生编辑数据集，以评估对LLM生成医疗摘要的影响。

**💡 创新点**

提出将编辑按临床、基准与个人三轴分类，并通过过滤无根源和个人编辑提升模型学习质量。

**🔧 技术方法**

使用GPT‑5.5模拟并标注编辑，结合低秩适配器对Gemma、Qwen和Llama等LLM进行微调。

**📊 数据集**

基于MIMIC‑III的出院摘要及其真实医生记录构建半合成编辑对，并使用公开的MIMIC‑III数据。

**📈 对比分析**

对比原始、过滤后的编辑与原始医生摘要的微调结果，在DocLens、MEDCON等对齐指标上显著提升，同时保持或提升基于输入的真值度指标（如PDSQI‑9）。

**⚠️ 局限性**

主要局限在于编辑合成的真实性、对单一GPT‑5.5和单一临床文档类型的依赖，以及对个人编辑判定的不确定性。

---

## 137. Confidence-Aware xApp Conflict Triage in Open RAN: A Closed-Loop ns3-oran Evaluation

**arXiv ID:** 2611.13516 | [PDF](https://arxiv.org/pdf/2611.13516v1)

**作者:** Abdul Wadud `[一作]` (University College Dublin), Nima Afraz `[通讯]` (University College Dublin)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种置信度感知的冲突分流（triage）机制，用于在OpenRAN中对冲突的xApp控制决策进行容忍、推迟或缓解的三种处理方式。

**💡 创新点**

创新点在于：①引入校准后的预测区间（conformal radius）衡量KPI预测不确定性；②基于置信度和冲突持续性决定处理方式；③在闭环ns3‑oràn仿真中验证，显著减少不必要的干预。

**🔧 技术方法**

主要技术包括：控制决策预测模型（可解析或学习型）、有限样本共形预测、三态决策规则、ns3‑oràn+5G‑LENA仿真框架。

**📊 数据集**

数据集为仿真生成的网络场景：13个gNB、117个UE，采用3.5 GHz、20 MHz、数字微蜂窝模型，使用10个种子进行预测区间校准，20个种子进行评估。

**📈 对比分析**

与无缓解策略和固定优先级策略比较，提出方法在相同仿真条件下：吞吐量提高约2.6 %，RU能耗降低约30%，RSRP违规率大幅下降，且仅做出30% fewer 的干预。

**⚠️ 局限性**

局限性包括：仅评估CCO–ES直接冲突；预测模型简化（未考虑阴影、关联变化）；共形预测的可交换性假设；在更复杂或更稠密的部署下性能与可靠性需进一步验证。

---

## 138. Using Proof Dependencies to Manage Verification Projects

**arXiv ID:** 2611.13250 | [PDF](https://arxiv.org/pdf/2611.13250v1)

**作者:** Lea Salome Brugger `[一作]` (ETH Zurich), Peter Müller `[通讯]` (ETH Zurich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

提出基于证明依赖分析的三大管理技术——证明审计、进度测量和验证指导，并在Viper与Gobra中实现。

**💡 创新点**

首次为符号执行的分离逻辑验证器自动提取完整证明依赖、构造依赖图，并基于此定义规格覆盖与证明质量度量以及指导策略。

**🔧 技术方法**

利用符号执行、UNSAT 核提取构建依赖图，并在Viper的分离逻辑实现与Gobra的IVL层迁移中应用。

**📊 数据集**

使用Viper与Gobra示例集、真实项目（Diffie‑Hellman、SCION路由器、Refinement、Res‑Algebra等）及其提交历史作为实验数据集。

**📈 对比分析**

通过与自顶向下、自底向上两种固定策略的进度曲线对比，证明基于依赖的指导在早期进度提升上显著优于两者；实验显示分析精度在95%以上，内存与时间开销相对可控。

**⚠️ 局限性**

分析受UNSAT 核非最小化、符号堆更新导致的误差以及路径不可感知等限制，且对并发程序支持尚不完善。

---

## 139. Explanation-Aware Fairness Beyond Output Parity: Joint Output-Attribution Regularization for Medical Image Classification

**arXiv ID:** 2611.13052 | [PDF](https://arxiv.org/pdf/2611.13052v1)

**作者:** Gökhan Özbulak `[一作]`, André Anjos `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究将特征归因公平性正则化与传统输出公平性正则化结合，提升医学影像模型在性别/种族上的公平与性能。

**💡 创新点**

提出了理论证明：对于线性分类头，输出公平性差距等于各特征归因差异之和；基于此设计归因公平性正则化，并证明其能降低归因差异与“取消”效应，显著提升 Pareto 前沿。

**🔧 技术方法**

使用 DINOv2‑Small 预训练视觉模型 + LoRA 微调 + 线性二分类头；归因计算采用梯度乘以特征值；正则化目标包括输出公平性（MeanEodd、Fairret、DiffEodd）与归因公平性（AttrEodd），并通过 Pareto/Hypervolume 评估。

**📊 数据集**

使用 HAM10000 皮肤病变图像（性别敏感属性）和 Harvard‑GF RNFLT 图像（种族敏感属性）进行实验。

**📈 对比分析**

与 ERM、MeanEodd、Fairret、DiffEodd 进行对比。归因增强组在 BAcc‑EOD Pareto 前沿上的 hypervolume 提升 7–10%（HAM）和 2–4%（Harvard），并在同等效用下将 EOD 降低 9–38%，AUROC 保持或略升。

**⚠️ 局限性**

局限性：仅针对线性分类头，未验证非线性预测器；实验仅覆盖两种二分类医学影像任务；样本规模与敏感属性种类有限，可能影响结果的普适性。

---

## 140. An Open Implementation Flow for the Xilinx Spartan-3E and Spartan-6 that Runs in the Browser, Checked Byte for Byte Against ISE

**arXiv ID:** 2611.13008 | [PDF](https://arxiv.org/pdf/2611.13008v1)

**作者:** Pedro Malo `[一作]` `[通讯]` (Universidade NOVA de Lisboa), Pedro Malo (Universidade NOVA de Lisboa)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套完整的开源实现流程（从HDL到可下载的位流），该流程可在浏览器或 Node.js 中运行，支持 Xilinx Spartan‑3E 与 Spartan‑6 FPGA。

**💡 创新点**

创新点在于：① 通过黑盒测量得到的位流数据库和仅使用已测量特性的位流写入器，确保生成的位流与 Xilinx ISE 14.7 完全一致；② 将 nextpnr 的 Himbaechel 架构与 WebAssembly 结合，直接在 Web 环境中完成放置、路由和位流生成；③ 在保持可解释性的前提下，得到比 ISE 更小且在面积和速度目标下更快的实现。

**🔧 技术方法**

使用了 Yosys、ABC、nextpnr、WebAssembly、Silinx 自研前端（Verilog 生成、映射）、后端（bitstream 写入、静态时序模型）、代码字批量测量技术、与 ISE 的 Oracle 对比、以及 Node.js/浏览器环境的 Web Workers。

**📊 数据集**

使用了 67 个公开的开源设计（CPU、外围设备、加密核心、浮点单元、基准等），以及多种 Spartan 设备的芯片描述文件（XC3S100E、XC3S250E、XC3S500E、XC3S1200E、XC6SLX16、XC6SLX45）。

**📈 对比分析**

通过与 ISE 14.7 在相同目标（Area、Speed、Fast）下进行对比，测量 LUT 数量、切片数、关键路径长度以及构建时间。结果显示：面积约减少 15%–21%，Area 与 Speed 目标下关键路径比 ISE 短 8%–14%，构建时间比 ISE 低 80%–84%，Fast 目标下关键路径略长 12%。

**⚠️ 局限性**

局限性包括：位流数据库仅覆盖已测量的配置位，未涵盖所有 DCM/PLL 等高级设置；Fast 目标性能仍不如 ISE；未完成 Spartan‑6 与 ISE 的完整对比；浏览器运行速度慢；iCE40 流程尚未配备 Oracle 进行位流与路由合法性验证。

---

## 141. BuySideFlow: A Point-in-Time Benchmark for LLM Agents in Investment Research

**arXiv ID:** 2611.12966 | [PDF](https://arxiv.org/pdf/2611.12966v1)

**作者:** Dong Bo `[一作]`, Jing Zhang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立了中文买方投资研究任务的点时效性（PIT）基准 BuySideFlow，评估LLM 代理生成的 SQL / SQL‑Python 工作流在结果正确性和时间有效性上的表现。

**💡 创新点**

创新点在于：①提出 PIT 违规分类体系并将人工审核与自动检查结合；②揭示仅结果一致并不能保证时间有效性；③提供包含可执行参考的 408 条中文任务集，填补了金融领域缺乏点时效性评测的数据空白。

**🔧 技术方法**

使用技术包括：SWE‑agent、Claude、GPT‑5.5、DeepSeek‑V4‑Flash 等 LLM 代理；SQL/SQL‑Python 自动执行与结果对比；PIT 违规检测规则与人工审计；以及基于 GPT‑6.1 SOL 的辅助审计。

**📊 数据集**

使用的数据集为授权的 Gildata 金融数据库快照（110 张表、4,547 列），涵盖股票、基金、债券和宏观经济等多资产类别的数据。

**📈 对比分析**

评估方法采用结果准确率（RA）和通过结果检查后出现 PIT 违规的比例（PVR_pass）；9 种配置的结果准确率仅为 16.2%–36.8%，PVR_pass 高达 50.8%，表明即使结果通过检查，超过一半的提交仍存在时间有效性问题。

**⚠️ 局限性**

局限性包括：①仅适用于中文买方任务，难以直接迁移至其他市场；②依赖单一数据库快照，未覆盖多版本或多源情况；③每个模型仅保留单次生成结果，未评估生成方差；④人工审核主观性和对 PIT 定义的依赖；⑤未全面量化 PIT 违规对最终结果的影响。

---

## 142. Seeing Is Not Believing: Counterfactual Lesion Erasure Teaches a Small Medical Vision-Language Agent to Report Only What the Image Shows

**arXiv ID:** 2611.13006 | [PDF](https://arxiv.org/pdf/2611.13006v1)

**作者:** Bhaskar Gurram `[一作]` `[通讯]` (Independent Researcher), Bhaskar Gurram (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

自动利用公开分割数据集构建无人工标注的对照视图（擦除病灶与匹配的控制编辑），定义证据敏感度差（ESG）指标，并在 Qwen3.5‑4B 上使用这些视图进行监督与拒绝采样微调，得到一种“证据门控”的医学 VLM。

**💡 创新点**

创新点在于：① 用自动化的擦除与匹配控制编辑方法生成大规模的对照样本；② 引入 ESG 作为衡量模型是否真正依据图像做出诊断的量化指标；③ 通过预注册的选择与检查流程，证明仅凭对照训练即可显著降低“假发现”率，同时保持真实病例的识别能力。

**🔧 技术方法**

技术手段包括：Qwen3.5‑4B 4B 参数模型 + LoRA 微调；图像编辑工具 LaMa、Tissue Transplant 及后期的 Telea 等；代理框架允许模型调用裁剪与对比度窗口工具；拒绝采样微调、监督式轨迹生成、通用 VQA 复现等训练阶段；预注册的 ESG 评估与净效用计算。

**📊 数据集**

主要数据来源于公开医学影像分割数据集：SLAKE（CT/MRI）、Kvasir‑SEG（结肠镜）与 HyperKvasir（无病灶）、ChestX‑Det（胸 X 光）、BUSI（乳腺超声）和 TCGA‑LGG（脑 MRI）。此外，还构造了外部中心和“新鲜”内部/外部测试集用于外部验证。

**📈 对比分析**

在 507 条内部测试病例（1,115 视图）上，MedScope‑4B 的 ESG 达 0.758，远高于 9 家前沿模型（0.47‑0.62）。擦除病灶的误报率仅 9.6%（比前沿 41‑62% 低 32‑52 点），在健康图像上的误报率也最低（7.6%）。在真实病灶的识别上与前沿模型无显著差异（CCR 0.849 vs 0.829‑0.888）。模型对不同编辑工具、提示方式、图像解析度等的鲁棒性也得到验证，但在新中心的感知准确性略逊于前沿模型。

**⚠️ 局限性**

局限性包括：① 对稀缺关键发现（如气胸）的覆盖不足，导致在新测试集上完全漏诊；② 视觉编码器冻结限制了对细小或非典型病灶的感知；③ 仍存在编辑痕迹对模型行为的潜在影响；④ 评估仅针对单帧 2D 影像的是/否或简答问题，未涉及报告生成、三维影像或临床前景验证；⑤ ESG 受编辑方式与工具的影响，未能完全消除所有“编辑+缺失”快捷路径；⑥ 缺乏人工验证的擦除图像质量，所有检查均基于自动检测。

---

## 143. Efficient Conversion of Algorithms to Linear Programs

**arXiv ID:** 2611.12896 | [PDF](https://arxiv.org/pdf/2611.12896v1)

**作者:** Shermin Khosravi `[一作]` (University of New Brunswick), David Bremner `[通讯]` (University of New Brunswick)

**通讯引用:** 6559 | [OpenAlex ID](https://openalex.org/A5020530779)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了两种针对LP编译器的优化方法——共享约束优化与层次同步优化，并提出了一种新的框架，用于在组合优化问题中生成更小、更易求解的线性规划模型。

**💡 创新点**

创新点在于将共享约束与层次同步结合，提出了层次同步+局部化优化框架，解决传统层次同步对最坏情况执行时间估计过高导致的压缩不足问题，并实现了LP文件大小最高可压缩至96%。

**🔧 技术方法**

使用了LP编译器的时间抽象模型、占位符替换、共享约束聚合、层次同步结构和局部化共享优化等技术，并通过Cplex、Gurobi等求解器进行性能评估。

**📊 数据集**

实验数据集为多种基本组合算法基准，包括最大匹配、最小生成树、装配等问题，涵盖了具有指数扩展复杂度的实例。

**📈 对比分析**

对比方法采用原始编译器、共享约束优化、层次同步优化及其组合，评估LP行列、非零数、文件大小、求解时间、预处理时间、内存占用和数值条件；实验结果显示，层次同步和共享约束均能显著缩小LP，提升求解率和速度，最优方案将文件压缩至53%以内，求解时间提升至99.9%。

**⚠️ 局限性**

限制在于层次同步依赖于最坏情况的执行时间估计，导致在某些代码块实际执行时间远小于估计时无法充分压缩；通过引入局部化框架可缓解此问题，但仍需进一步改进。

---

## 144. UniWeaver: Harnessing Unified Multimodal Models for Interleaved Text-Image Generation

**arXiv ID:** 2611.13530 | [PDF](https://arxiv.org/pdf/2611.13530v1)

**作者:** Hongxu Chen `[一作]` (Ohio State University), Pheng-Ann Heng `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 UniWeaver 框架，实现多模态文本-图像交错生成。

**💡 创新点**

创新点是将多模态规划、生成与视觉验证集成到统一流程，并通过任务级与单位级目标实现结构化规划。

**🔧 技术方法**

采用结构化多模态规划、文本与图像角色分离、可选视觉参考路由及局部视觉验证等技术，并保持模型参数不变。

**📊 数据集**

构造 UniWeaver‑10K 数据集（10k 任务、20类）及 UniWeaver‑Bench 基准，用于训练与评估。

**📈 对比分析**

在七种 UMM backbone 上与最小逐步 harness 对比，UniWeaver 无参数更新即可提升整体得分至 58.9；finetune 后最高可达 87.1。

**⚠️ 局限性**

受限于底层 UMM 的文本与图像质量，以及只覆盖有限任务场景，难以满足专业级别的多模态生成需求。

---

## 145. Wrong the Same Way: Cross-Prompt Sharing Both Suppresses and Amplifies Reward Hacking in RLVR

**arXiv ID:** 2611.13218 | [PDF](https://arxiv.org/pdf/2611.13218v1)

**作者:** Michael Li `[一作]` (Carnegie Mellon University), Ray Chen `[通讯]` (Independent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在存在可验证奖励（RLVR）环境下，模型如何继承或放大基线模型的奖励攻击，重点分析泄漏（leak）的普遍性与参数共享的相互作用。

**💡 创新点**

提出了“共享参数是攻击放大通道”的假设，并证明了在 KL 正则化极值点 RLVR 仅作过滤器，而实际训练中跨提示共享导致攻击的双向偏移；还给出了基于模型内核权重的预测指标。

**🔧 技术方法**

使用了精确期望策略梯度、Adam/SGD/GRPO 优化器、KL 正则化、以及模型内核（tangent kernel）分析等技术。

**📊 数据集**

在一个精确可枚举的实验环境：单位乘法（64 个提示，答案空间 00–99）中训练一个 2 层 64 维 Transformer，数据由监督目标与注入的泄漏样本构成。

**📈 对比分析**

通过比较共享参数与每提示独立参数的两种架构，在不同学习率和泄漏普遍性下评估了攻击率、对数优势和准确率；发现共享参数在高普遍性下会放大攻击，在低普遍性下可能抑制；相对基线，准确率下降幅度可达数个百分点。

**⚠️ 局限性**

局限性包括：实验环境过于小（答案仅两位，内核为有限宽度）；泄漏仅以字符串形式注入，未模拟真实模型的错误模式；只考察了单一模型规模和特定任务；跨提示共享效应可能在更大规模模型中表现不同。

---

## 146. Compute the Interactions, Not the Attributes: What a Fact Layer Must Supply in the Chinese Sexagenary System

**arXiv ID:** 2611.13597 | [PDF](https://arxiv.org/pdf/2611.13597v1)

**作者:** Huixue Han `[一作]`, Jiaying Liu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了基于中国干支系统的事实层基准，系统评估了不同模型在闭书、表格检索、工具输出以及各种事实层设计（属性、完整关系、查询过滤关系）下的推理准确率。

**💡 创新点**

创新点在于首次将事实层的计算深度与模型表现进行细粒度对比，揭示属性信息对冲突推理几乎无效，计算关系及其呈现方式决定准确性；同时通过棋盘规则验证方法的通用性。

**🔧 技术方法**

采用多种大语言模型（Qwen3系列、Gemma、OLMo）在无思考与思考两种推理模式下进行实验，并结合工具输出、检索增强与 JSON‑schema 结构化评测，利用 McNemar 统计检验比较。

**📊 数据集**

使用合成的 BaZi 图表数据（80张图表、160 条冲突项、682 个事实项共 842 条）和对应的参考表，另设 160 条棋盘推理样本；金标准由独立实现的规则引擎生成。

**📈 对比分析**

通过对闭书准确率、属性与关系层的交叉对照，发现属性层接近上限但冲突推理极低；完整关系列表提升少量模型，最优的查询过滤层在大部分模型上将准确率提升至 96–100%，确认模型表现接近 100%。

**⚠️ 局限性**

局限包括：使用合成数据，未涵盖真实图表分布；评测仅限于 JSON 结构化答案，无法反映自由文本推理；查询过滤层未在真实错误过滤场景下测试；只覆盖两类规则系统（干支与棋盘），结果可能不适用于其他复杂领域。

---

## 147. Navigating relational agency in AI feedback musicianship: a diffractive dialogue

**arXiv ID:** 2611.13555 | [PDF](https://arxiv.org/pdf/2611.13555v1)

**作者:** Yichen Wang `[一作]` (Australian National University), David Cooper `[通讯]` (Dolby Laboratories)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了两款 AI 驱动的反馈乐器（Sù 与 Agentier）在双人即兴表演中的相互作用与共同建构关系，并采用差异性对话（diffractive dialogue）方法对五场演出进行定性分析。

**💡 创新点**

提出反馈音乐中的“相互建构关系”概念，识别控制与代理是关系性、可变的；引入“反馈环境”“注入”“引导”“导航代理”等新词汇，强调在反馈乐器中需学习导航代理的核心技能。

**🔧 技术方法**

使用了基于 RAVE 的音频生成与潜在空间反馈的乐器 Sù，基于 MDRNN 的 MIDI 手势生成反馈的乐器 Agentier，配合差异性对话（diffractive dialogue）方法进行反思与分析。

**📊 数据集**

未使用公开数据集，表演基于作者自建模型与设备，收集了 5 场演出（共 87 分钟）音频与视频记录。

**📈 对比分析**

通过写作反思、对话记录、文本分析进行定性比较，识别五大主题；未给出数值指标，性能评估以“反馈环境”构建质量、代理导航成熟度等为主。

**⚠️ 局限性**

仅涉及两位表演者与两台乐器，实验时间有限；缺乏对乐器设计与模型训练细节的系统评估；缺乏定量性能指标与更广泛群体验证。

---

## 148. Route-Time Conflict Maps for Cooperative End-to-End Autonomous Driving

**arXiv ID:** 2611.13494 | [PDF](https://arxiv.org/pdf/2611.13494v1)

**作者:** Haohua Que `[一作]` (University of Georgia), Handong Yao `[通讯]` (University of Georgia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种CDA Class B的端到端协同驾驶系统，通过共享意图并利用路段-时间冲突图来学习何时通过路口。

**💡 创新点**

创新点包括：1）路段-时间冲突图将感知、预测和合作计划融合为占用图；2）通过优先级网络实现无轮次的去中心化同意（order gate）；3）对无可观察原因的停止进行重标注以解决专家与学习者的不对称。

**🔧 技术方法**

结合LiDAR感知、运动预测、合作融合、深度规划器对速度曲线评分、优先级网络、以及基于特权专家的模仿学习。

**📊 数据集**

在CARLA仿真平台的InterDrive、MDrive、V2Xverse三个基准上训练与评估，并使用仿真日志中的特权专家轨迹做标签。

**📈 对比分析**

与16种公开方法及多种基线对比，取得最高驾驶分数：InterDrive 97.8/90.7、MDrive 88.2、V2Xverse 70.0，碰撞率低于对手并在大多数情景下实现近100%无碰撞。

**⚠️ 局限性**

局限在于仅限仿真环境、仅使用LiDAR感知、缺乏交通灯/停止标志输入、依赖特权专家标签、固定行驶路线、通信延迟或丢包时性能下降，尚未在真实车辆上验证。

---

## 149. When Are Predictions Enough? An Evaluation Protocol for Frozen Expert Composition

**arXiv ID:** 2611.13356 | [PDF](https://arxiv.org/pdf/2611.13356v1)

**作者:** Alessandro Pereira `[一作]` (Adelaide University), Lewis Mitchell `[通讯]` (Adelaide University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在冻结专家模型的组合中，特征空间组合相较预测空间组合的优势，并提出了可复用的评估协议。

**💡 创新点**

首次系统地将强预测基线阶梯、特征容量检验、轴隔离专家池和重叠控制纳入四部分协议，并在深度伪造检测上进行敏感性审计。

**🔧 技术方法**

使用逻辑回归、MLP、XGBoost 等组合器，以及训练集/验证集分层、留一生成器交叉验证、Bootstrap 置信区间等统计方法。

**📊 数据集**

基于 DF40 深度伪造检测基准，构建 21 个冻结检测器，覆盖八个隐藏生成器。

**📈 对比分析**

在 LOGO 评估下，线性特征组合在架构多样化专家池中相较于预测基线提升 0.081 AUROC（95% CI[0.024,0.140]），但在同构池或弱专家池中优势消失。

**⚠️ 局限性**

实验仅限于单一深度伪造基准、二分类输出，重叠控制与专家训练信息仍影响结论，且大容量特征组合器未显示显著提升。

---

## 150. The Left Ripple of Byte-Pair Encoding: Certified Lookbehind for Every Dictionary

**arXiv ID:** 2611.13061 | [PDF](https://arxiv.org/pdf/2611.13061v1)

**作者:** Nicklas Nidhögg `[一作]` `[通讯]` (Independent Researcher), Nicklas Nidhögg (Independent Researcher)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究字节对编码（BPE）中的字符串前缀添加对分词结果的传播（即"origin ripple"）。它给出了两种传播度量（origin radius 与 label distance），证明在任何字典下这两者可判定并用多项式时间算法精确计算；确立了无穷传播的必要且充分条件（有效自配对规则）；进一步证明 BPE 分词器是子序列化的，构造了零延迟的流式分词器；并给出了右侧扩展的相关度量（right span radius、right start radius）以及标签距离的上界。最后将这些理论与实际模型（GPT‑2、cl100k、Llama 3 等）进行实验验证。

**💡 创新点**

主要创新点包括：
1. 统一定义并可计算的左侧前缀传播半径(origin radius)和标签距离；
2. 证明任何字典下该半径可在多项式时间内决定是否为无穷，并给出判定条件（有效自配对规则）；
3. 证明 BPE 分词器是子序列化的，并构造了可在每个符号后立即输出稳定标记的流式分词器（实现零延迟）；
4. 将右侧扩展（right span/start radii）与左侧传播度量关联，并给出标签距离的上界；
5. 通过重叠图和词典自动机实现原理的高效实现，并在实际模型中验证其数值。

**🔧 技术方法**

技术手段：
- 自动机与转移图（重叠图、词典自动机、冻结词机）
- 语义化的 BPE 规则执行模型（按优先级一次扫描）
- 组合与最短路径/循环检测（用于判定无穷传播）
- 证明子序列化性与流式输出的理论（基于半径与切点）
- 多项式时间算法与复杂度分析
- 证明标签距离上界（Fibonacci 与二次上界）

**📊 数据集**

使用的数据集包括：
- 规模较小的词典基准（census）
- GPT‑2 完整合并表（merge table）
- 公开的 BPE 合并表 cl100k_base、o200k_base、Llama 3
- 其它自构造的最小例子（如 (a, a)、(a, b)、(a, ab) 等）

**📈 对比分析**

比较方法：
- 与 Berglund 等人提出的右侧传播理论（|D| 令牌界）进行对照；
- 与 Dalirrooyfard & Ye 提出的流式分词器算法对比；
- 通过在基准词典上计算 origin radius、label distance、右侧半径，验证与理论上限的匹配；
- 对 GPT‑2、cl100k 等实际模型计算得到的半径与标签距离与论文给出的理论区间进行比对。 性能：
  - 所有主要判定/计算算法在字典规模（写出形式）上为多项式；
  - 对于基准词典的实验时间在毫秒级；
  - 对 GPT‑2 等大表的实验在数秒内完成；
  - 流式分词器实现每个符号后立即输出，延迟为零。

**⚠️ 局限性**

限制与不足：
- 算法复杂度尽管是多项式，但对字典中规则数量指数级的符号长度（尤其在 clause (a) 约束下）可能导致实际规模非常大；
- 研究仅针对满足“排序条件”（ordering condition）的字典，非排序字典的传播行为仍未完全解决；
- 结果主要适用于 BPE 规则的标准执行（一次扫描），不适用于包含预分词器或其他预处理步骤的模型；
- 标签距离的上界虽然给出，仍存在理论与实际差距，未能给出更紧的全局下界；
- 由于对“有效自配对”判定的依赖，某些字典在实际实现中可能需要额外的预处理来检验该条件。

---

## 151. Topology-Aware Reinforcement Learning for CAD Decomposition toward All-Hexahedral Meshing

**arXiv ID:** 2611.12930 | [PDF](https://arxiv.org/pdf/2611.12930v1)

**作者:** Jiaming Peng `[一作]` (National University of Defense Technology), Xinhai Chen `[通讯]` (National University of Defense Technology)

**通讯引用:** 1153 | [OpenAlex ID](https://openalex.org/A5029556003)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于深度强化学习的 CAD 模型分块框架 DecompRL，自动将复杂 CAD 体裁分解为适合六面体网格的子块。

**💡 创新点**

创新点：① 设计拓扑感知的内部拆分动作空间（平面、对称、O‑grid、特征间隙、旋转）以解决特征错位；② 在奖励中加入网格可行性（Map/Sweep/Submap）信号；③ 采用多子节点 Tree‑Bellman 目标提升长序列决策质量。

**🔧 技术方法**

技术：深度强化学习（SAC）、B‑Rep 图编码器（AAGNet）、候选生成与可行性过滤、几何/拓扑奖励、可视化评估与后处理。

**📊 数据集**

使用公开 CAD 基准：ABC、Fusion360、NIST、DeepCAD。

**📈 对比分析**

与现有学习方法（PPO、DDPG、TD3）、Zhang 等基线以及 LMFF 进行统一评估：在三四个数据集上成功率最高（Fusion360 59.6%，ABC 40.6% 等），网格质量保持竞争性，分块耗时比 LMFF 快 40–100 倍。

**⚠️ 局限性**

局限性：① 动作模板仅针对机械 CAD，无法覆盖自由曲面或复杂 NURBS；② 缺少对 CFD 相关层间、壁面对齐等约束；③ 训练深树时收敛不稳定，局部可行性判定不保证全局网格成功。

---

## 152. Force-Closure Condition for Cooperative Multi-robot Pushing of Rectangular Objects

**arXiv ID:** 2611.13268 | [PDF](https://arxiv.org/pdf/2611.13268v1)

**作者:** Duy Anh Nguyen `[一作]` (University of Prince Edward Island), Trung Dung Ngo `[通讯]` (University of Prince Edward Island)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并推导了四个无摩擦单向推压接触点对矩形物体实现平面力闭合的必要且充分的闭式条件，并将其嵌入到协作推压控制框架中。

**💡 创新点**

创新点在于得到一个只依赖四个边缘接触位置、与质心或轨迹无关的闭式不等式；该不等式可直接用于接触可行性检查，并给出简单的接触调整规则。

**🔧 技术方法**

使用了平面接触力学分析、力向量锥的理论、非负最小二乘力分配、PD 位置速度跟踪控制，以及三自由度漂浮体动力学仿真。

**📊 数据集**

没有使用公开数据集，而是采用仿真模型：三自由度漂浮体惯性矩阵来自海上供应船（OSV）模型，矩形尺寸为长 83 m、宽 18 m。

**📈 对比分析**

通过对比三种接触配置（缺失边缘、满足/不满足力闭合）在静态力可行性测试和狭窄通道全姿态跟踪仿真中的表现。满足力闭合的方案在 x、y、偏航误差 RMS 分别从 12.05 m/18.11 m/535.96° 降至 2.24 m/2.66 m/2.69°，显著提升跟踪性能。

**⚠️ 局限性**

局限性包括：仅考虑无摩擦单向接触点和二维平面运动；未考虑海流、波浪等流体动力学扰动；未在真实机器人上验证；对更一般形状和多接触情况的推广仍待研究。

---

## 153. When Proofs Are Abundant, What Is a Paper For?

**arXiv ID:** 2611.12943 | [PDF](https://arxiv.org/pdf/2611.12943v1)

**作者:** Charles-Albert Lehalle `[一作]` `[通讯]` (Ecole Polytechnique), Charles-Albert Lehalle (Ecole Polytechnique)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文提出并形式化了两个核心概念——Derivation‑Level Intension（DLI）和 Directional Extension（DE），用以阐明在 AI 与自动化证明工具普及后，科研论文应如何传达研究者的“意图”和“推理链”，并突出其弱点与强点；

**💡 创新点**

创新点在于：①把论文的“问题选择与视角”抽象为 DLI；②把论文的“推理结构与可重用性”抽象为 DE，并给出形式化定义；③指出即使证明可自动验证，论文仍需通过这两个维度与社区进行批判性交流，确保研究的深度与方向；

**🔧 技术方法**

技术上主要借助逻辑（证明论）、类型理论（对证明作为对象的表述）和范畴理论（定义推理链、切点、可迁移性）来构造这两个概念；

**📊 数据集**

该工作为理论性框架，不依赖任何特定数据集；

**📈 对比分析**

没有实验比较或性能评估；讨论主要是概念性分析与示例说明；

**⚠️ 局限性**

局限性包括：①缺乏实证验证或案例研究以证明该框架在真实科研工作中的可操作性；②对 AI 工具在实际科研流程中具体如何影响 DLI/DE 的影响尚不明晰；③框架可能过于抽象，难以在已有的学术出版流程中直接落地。

---

## 154. Does Haptic Connection to a Healthy Partner Improve Motor Learning in Chronic Stroke? A Pilot Study of Robot-Based Dyadic Training

**arXiv ID:** 2611.13307 | [PDF](https://arxiv.org/pdf/2611.13307v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 155. Train on prefill, monitor at decode: Using sparse autoencoders for topic monitoring

**arXiv ID:** 2611.12935 | [PDF](https://arxiv.org/pdf/2611.12935v1)

**作者:** Rishikesh Mallagundla `[一作]` (Mercity), Pranav Patel `[通讯]` (Mercity)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在语言模型生成过程中训练并部署了基于稀疏自编码器(SAE)的主题监控器，能够为每个 token 实时标注主题。

**💡 创新点**

创新点在于：① 将预填充阶段的 SAE 特征用于训练监控器并在解码时无缝迁移；② 实现 token 级主题标注，使模型在生成过程中能即时捕捉主题漂移；③ 通过静态与动态特征选择保持监控器小型化并可解释。

**🔧 技术方法**

主要技术包括：稀疏自编码器（Gemma Scope 2）、GRU/ConvNeXt/Transformer 触发器、动态与静态特征选择策略、以及基于 GPT‑5.6 Luna 的 Synthetic 7 主题对话生成。

**📊 数据集**

使用的数据集是 10,000 条人工生成的多主题对话，覆盖 7 个业务主题（企业文件、通用新闻、客服、法律、金融、HR 与医疗），每条记录的 prompt 与 response 均被逐 token 标注。

**📈 对比分析**

评估方法：在 Gemma‑3 1B/4B 上对比宏 F1、覆盖率和纯度；与无 SAE（raw hidden state）基线对比；在预填充与解码阶段分别测得 90–98% 的纯度，覆盖率接近 95%；在插入、删除、重排等扰动实验中最大宏 F1 下降不超过 5 点，说明鲁棒性良好。

**⚠️ 局限性**

局限性包括：① 需要预先定义并手工标注的主题集合；② 评估仅基于 synthetic 对话，缺少真实生成文本的黄金标签；③ 仅在 Gemma‑3 系列模型上验证，未评估更大或不同家族模型；④ 监控器未直接提供“非主题”检测，插入内容仍被错误标注为现有主题。

---

## 156. Evaluating Compositional Generalization in Transformers: The Role of Composition Equivalence and Module Coverage

**arXiv ID:** 2611.13498 | [PDF](https://arxiv.org/pdf/2611.13498v1)

**作者:** Purva Pruthi `[一作]` (University of Massachusetts Amherst), David Jensen `[通讯]` (University of Massachusetts Amherst)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估Transformer在序列组合任务中的组合推广能力，并通过控制训练-测试分布差异来分析其表现。

**💡 创新点**

提出组合等价性与模块覆盖率是决定组合推广效果的两大关键因素，并将“未见”重新定义为功能层面的新映射。

**🔧 技术方法**

使用小型NanoGPT Transformer（直接与逐步两种训练目标，绝对与相对位置编码），通过KL散度量评估覆盖率，利用LoRA微调和精细的训练-测试拆分方法。

**📊 数据集**

基准数据集包括：无等价性的随机bijection数据集、含多种字符串操作的多样化数据集，以及真实编程任务LiveCodeBench。

**📈 对比分析**

在within‑k和cross‑k设置下对比直接与逐步模型，发现仅在共享等价性或低覆盖率偏移时准确率可达70–80%，否则几乎为0；Llama3‑8B在等价性泄漏的细分上显著提升。

**⚠️ 局限性**

局限在于仅处理线性序列组合，尚未验证层次或图结构组合的情况；实验受合成数据控制的影响，难以完全映射真实世界场景。

---

## 157. Large Language Model Explainability for Planning Lunar Rover Trajectories under Uncertainty

**arXiv ID:** 2611.13292 | [PDF](https://arxiv.org/pdf/2611.13292v1)

**作者:** Michelle Ho `[一作]` (Stanford University), Mykel J. Kochenderfer `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一个基于LLM的交互式助手，用于解释并重新规划月球探测车在不确定性下的轨迹。

**💡 创新点**

将LLM与MCTS规划器分离，仅在后期解释阶段使用LLM，并通过检索+工具调用增强可信度。

**🔧 技术方法**

MCTS轨迹规划、VIPER启发式物理模拟、ReAct + DSPy框架、检索增强生成、LLM大模型（GPT‑5.4/Claude Opus）。

**📊 数据集**

VIPER风暴区冰层稳定度深度图、太阳方位依赖的速度概率分布、任务仿真环境以及科学团队提供的兴趣点与钻探候选点。

**📈 对比分析**

通过一套模拟查询银行进行零射击、闭本、无检索、完整框架的比较，完整框架在语义与轨迹指标上获得最高分，准确率提升至90.9%。

**⚠️ 局限性**

受限于模型的解释一致性、部分查询导致信息缺失导致不可信回答、目前仅支持平稳MDP且未考虑部分可观测性及主动不确定性降低。

---

## 158. Block-Sparse Gauss-Newton Optimization of Finite Basis Physics-Informed Neural Networks: Sparse Automatic Differentiation Approach

**arXiv ID:** 2611.13231 | [PDF](https://arxiv.org/pdf/2611.13231v1)

**作者:** Denis Korolev `[一作]` (Weierstrass Institute for Applied Analysis and Stochastics), Jeeeun Lee `[通讯]` (Korea Advanced Institute of Science and Technology (KAIST))

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种利用有限基物理信息神经网络（FBPINNs）中的块稀疏结构，构造高效的高阶 Gauss–Newton 优化方法，并在多尺度 PDE 任务上实现快速收敛。

**💡 创新点**

创新点在于：① 将 FBPINNs 的重叠域分解视为图着色问题，推导出最优块着色与贪婪行着色；② 通过稀疏自动微分（AD）压缩残差雅可比，直接以块稀疏形式组装 Gauss–Newton Gramian；③ 在 GPU 上实现稀疏直接求解与预条件共轭梯度，显著降低计算和存储成本。

**🔧 技术方法**

核心技术包括：有限域分解网络结构、图着色的块/行颜色策略、稀疏前向/逆向 AD、块稀疏矩阵组装、稀疏直接（cuDSS）与预条件迭代（添加 Schwarz 等）求解器。

**📊 数据集**

使用了合成多尺度 Poisson 方程、带快速振荡系数的椭圆问题（第一阶系统）以及在穿孔域中的 Stokes 流动三种典型 PDE 作为实验数据集。

**📈 对比分析**

与 Adam（传统一阶优化）对比，Gauss–Newton 在相同 GPU 资源下在相同迭代次数或时间内能实现更低的 L¹/L² 误差，且收敛速度快、迭代次数少；稀疏直接求解在实验规模内比迭代求解更高效，但在更大规模下需进一步改进。

**⚠️ 局限性**

主要限制：① 稀疏直接求解在块规模增大时易产生大量填充，内存占用激增；② 迭代预条件效果受限，当前的跨/王国 Schwarz 需要较大预条件矩阵；③ 仅在规则网格重叠子域上推导了闭式颜色，非规则分块需更复杂的贪婪策略；④ 仍依赖单 GPU，缺乏集群并行实现；⑤ 需要更系统的正则化与参数冗余剔除方法以进一步提升数值稳定性。

---

## 159. Instance-preserving graded semantic reporting for surgical action triplets

**arXiv ID:** 2611.13009 | [PDF](https://arxiv.org/pdf/2611.13009v1)

**作者:** Yanchong Xue `[一作]` (Harbin Engineering University), Guochao Zhang `[通讯]` (Hainan Cancer Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种实例保持的分级语义报告框架，将外科动作三元组的实例选择与语义细节分离，并对同一预测结果进行完整‑拒绝、两级和三级报告的对比实验。

**💡 创新点**

创新点在于引入中间的 instrument‑verb (IV) 报告层，利用分级正确性评分与三种决策规则（阈值、DELTA、DARTS）实现对 I、IV、IVT 三种输出层次的动态选择，从而在保留有用信息的同时降低错误率。

**🔧 技术方法**

技术手段包括基于 Mask2Former 的 CholecTriplet‑Seg 预测器、层次化正确性评分网络、阈值校准与损失平衡、分级决策规则（THR、DELTA、DARTS）、基于 IoU 的实例匹配与报告权重计算。

**📊 数据集**

实验使用 CholecTriplet‑Seg 数据集，训练集包含 29 句子视频，验证集 5 句子视频，测试集 8 句子视频，总计 5,641 帧，采用 100 类三元组 + no object 分类。

**📈 对比分析**

比较方法通过 source‑normalized weighted correct‑report yield (Y_info) 与 weighted reporting risk (R_info) 两个指标实现；结果显示加入 IV 报告层在相同实例集上提升 19.1%–23.1% 的 weighted yield，风险变化仅为 ±1.0% 点，完整 triplet 的数量和风险随规则不同而略有波动。

**⚠️ 局限性**

局限性包括：在不同数据源迁移时需重新校准阈值和规则导致风险上升；对字段权重的假设可能不适用于所有下游任务；仅评估了固定 Mask2Former 预测器，未验证其他预测模型的通用性。

---

## 160. Decisions by Discrimination: Differential Evidence for Medical Reasoning

**arXiv ID:** 2611.13233 | [PDF](https://arxiv.org/pdf/2611.13233v1)

**作者:** Yunpeng Xiao `[一作]` (Emory University), Kai Shu `[通讯]` (Emory University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种检索驱动的差异性证据框架，用于医学多项选择问答。

**💡 创新点**

将差异性证据视为组织问题，而非一般检索，提取选项特定的判别信息，显著压缩检索上下文。

**🔧 技术方法**

四阶段流程：anchor cue提取、cue–option共现检索、LLM生成差异性证据、基于证据推理；使用外部LLM生成证据并在目标LLM中推理。

**📊 数据集**

MedQA、MedXpertQA-Text、MedBullets 三大医学多项选择基准。

**📈 对比分析**

对比基线、CoT、差异证据、CoT+差异证据；在所有模型和数据集上均取得平均约+7–10%的准确率提升，远优于仅检索文本或通用提示。

**⚠️ 局限性**

依赖外部LLM证据质量，仍可能产生答案泄露或在无候选选项的开放式场景中难以扩展。

---

## 161. A Technique to Model Product Information and User Reviews (MPIUR)

**arXiv ID:** 2611.13361 | [PDF](https://arxiv.org/pdf/2611.13361v1)

**作者:** Emna Belguith `[一作]` (Higher School of Communication of Tunis), Talel Abdessalem `[通讯]` (LTCI, Télécom Paris, IP Paris)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个多模态图神经网络框架（MPIUR），通过用户评论、商品文本描述和商品图片三种信息共同建模并实现物品推荐。

**💡 创新点**

创新点包括：①将用户评论先使用LLM抽取结构化的aspect‑opinion‑sentiment三元组，构建Topic Graph；②利用per‑item注意力融合图像与文本特征；③引入基于物品互动稀疏度的加权对齐损失，使稀疏物品更依赖多模态信息；④多任务损失（BPR、对齐、MI等）协同训练，兼顾协同与语义空间。

**🔧 技术方法**

使用的技术主要有：BERT/MiniLM提取文本与评论特征，ResNet50或VLM生成图像特征；图神经网络（TGNN/LightGCN）构建协同图和Topic图；per‑item注意力融合模块；InfoNCE对齐损失；BPR排序损失；HADSF LLM提取评论三元组；以及各种加权与对齐机制。

**📊 数据集**

在 Amazon Reviews 2023 的两个品类：Musical Instruments 和 Baby 上进行实验，使用这些品类的文本、图像与评论数据。

**📈 对比分析**

与 LightGCN、TGNN、BM3 等传统和多模态基线进行比较，采用 Recall@10 与 nDCG@10 评估。MPIUR 在整体数据集以及稀疏物品子集上均获得最高的 nDCG@10 和 Recall@10，稀疏物品的提升尤为显著。

**⚠️ 局限性**

局限性包括：仅使用包含评论的交互，忽略无评论的购买/点击等行为；未覆盖完整冷启动场景；评估采用采样负样本而非全库排序；BPR 视所有正交互同等对待，未区分星级；VLM 生成的图像描述效果不稳定。

---

## 162. Off-policy Learning with Excursion Policies

**arXiv ID:** 2611.13275 | [PDF](https://arxiv.org/pdf/2611.13275v1)

**作者:** Jiamin He `[一作]` (University of Alberta), Diana Borsa `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一类新的“越轨”策略（excursion policies），用于离线（off‑policy）强化学习，通过在行为策略和目标策略之间进行概率性、有限步的“越轨”来平衡偏差与方差；随后在动态规划框架下推导了对应的 Bellman 运营商，证明了其收敛性，并基于此设计了三种 TD 估计器（Ex‑TD、Ex‑trace、Ex‑Retrace），这些算法在行为策略下采样时即可估计越轨策略的值函数；最后通过理论分析和连续控制实验对越轨策略与传统线性混合策略（mixture policy）以及其他多步回报进行比较。

**💡 创新点**

创新点主要有：①首次将“越轨”概念正式化为可参数化的策略族，能够在行为策略与目标策略之间实现非线性、时间依赖的插值；②构建了越轨策略的动态规划理论，包括收敛的 Bellman 运营商和策略改进定理；③设计了能在离线数据上稳定收敛的 TD 算法（Ex‑TD、Ex‑trace、Ex‑Retrace），并通过可变的结束概率 β 逐步调节偏差-方差-收敛率平衡；④从理论到实践，系统地与混合策略（α‑trace 等）对比，揭示了越轨策略在时间依赖性、收敛速度与样本效率上的显著差异。

**🔧 技术方法**

技术上使用了：动态规划与 Bellman 递推、重要性采样与 IS 比例裁剪（V‑trace、Ex‑trace）、多步 TD 目标、特征化的“越轨”状态扩展、谱分析评估收敛率与方差、MPO（Maximum a Posteriori Policy Optimization）作为基准算法，以及分布式训练框架来加速连续控制实验。

**📊 数据集**

实验数据集主要来自 OpenAI Gym / DeepMind Control Suite 的 MuJoCo 连续控制任务，具体包括 7 个高维任务（如 Hopper、HalfCheetah、Walker2d、Ant、Humanoid 等），每个任务用 3 个随机种子进行评估。

**📈 对比分析**

评估方法包括：①在随机生成的离散 MDP 上进行收敛率-偏差-方差三维谱分析，比较 Ex‑Retrace、α‑Retrace、n‑step 与 corrected n‑step；②在连续控制实验中绘制学习曲线，衡量“效率‑稳定性”平衡；③通过不同的 Samples‑Per‑Insert (SPI) 比例模拟离线程度。实验结果显示：Ex‑Retrace 在偏差与收敛率上优于 α‑Retrace，且相较于纯 Retrace（不稳定）和 uncorrected TD（收敛慢）实现了更佳的效率‑稳定性折衷；随着 β 的增大，Ex‑Retrace 在低 SPI 环境下性能提升明显，超过 α‑Retrace 的上限。

**⚠️ 局限性**

局限性包括：①理论收敛证明仅在无函数逼近的离散表格环境下成立；②对 λ∈(0,1) 的一般化策略改进尚未完全解决；③越轨策略的最优参数 β 与 λ 的选择仍依赖经验，缺乏自适应机制；④实验仅覆盖标准 MuJoCo 任务，尚未验证在更大规模或离线数据集上的表现。

---

## 163. Specializing Meta Multi-Objectivization as Baseline for Software Engineering Optimization

**arXiv ID:** 2611.12991 | [PDF](https://arxiv.org/pdf/2611.12991v1)

**作者:** Chengjie Chuan `[一作]`, Tao Chen `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文针对电力系统（SE）优化问题，提出并实现了一种基于贝叶斯优化（Bayesian Optimization）的自动化搜索方法，并与传统的试错（Trial-and-error）方法进行对比。

**💡 创新点**

创新点在于将贝叶斯优化与电力系统调度问题结合，利用高斯过程（Gaussian Process）构建预测模型并通过期望改进（Expected Improvement）等采样策略显著提升搜索效率；同时在同一套测试集上实现了更高的最优解质量。

**🔧 技术方法**

技术路线主要包含：1) 高斯过程回归；2) 期望改进等采样准则；3) 多目标（或约束）贝叶斯优化框架；4) 与电力系统仿真接口的集成；5) 结果可视化（如条形图、点线图）。

**📊 数据集**

使用了9个公开电力系统优化案例作为实验数据集，分别对应不同的调度或负荷预测任务，标签为：NRP、SCT、SEE、SDP、SPLT、SPS、TPLM、WS、WSC。

**📈 对比分析**

对比方法：将贝叶斯优化结果与传统试错方法在相同问题上的解算次数（# of solution evaluations）进行比较。实验结果表明，虽然贝叶斯优化在某些问题上需要更多的评估次数（如NRP、SPLT等），但它在整体上能获得更优或更稳健的解，显著提升了解的质量；而试错方法评估次数相对较少，但解质量不如贝叶斯优化。

**⚠️ 局限性**

局限性包括：1) 贝叶斯优化在高维度或极端非线性问题上仍可能收敛缓慢；2) 需要较多的初始评估点才能构建可靠模型；3) 计算开销较大，特别是对大规模仿真模型；4) 结果受高斯过程假设的影响，若误差模型不匹配可能导致性能下降。

---

## 164. StatPrompt-LLM: A Lightweight Statistical Refinement for Foundation-Style Time-Series Forecasting

**arXiv ID:** 2611.12974 | [PDF](https://arxiv.org/pdf/2611.12974v1)

**作者:** Aniket Sinha `[一作]` (University of Sydney), Veronica W. Tsou `[通讯]` (University of Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

研究了将数据集级统计描述和自回归旁路加入Time-LLM框架，提出StatPrompt-LLM。

**💡 创新点**

创新点在于轻量级的统计提示增强和零初始化自回归旁路，兼顾统计与基础Transformer。

**🔧 技术方法**

使用基于GPT-2的Time-LLM、DLinear、PatchTST、Autoformer等模型，并加入统计向量、线性旁路。

**📊 数据集**

实验数据集包括ETTh1、ETTh2和Weather三大长期时序数据。

**📈 对比分析**

通过全前缀、10%前缀、消融、计算效率四个实验比较，StatPrompt-LLM在10%前缀下MSE下降约40%，MAE下降约22%，但全前缀表现各异。

**⚠️ 局限性**

局限性包括仅在三组数据集、单一长度与预测窗口、仅做在样本评估、未独立测试、无法分离提示与旁路贡献。

---

## 165. Agent Architecture Cost-Success Trade-offs: A Controlled Evaluation across Enterprise-Style SOPs

**arXiv ID:** 2611.12953 | [PDF](https://arxiv.org/pdf/2611.12953v1)

**作者:** Subhrangshu Nandi `[一作]` (Amazon), Prathap Ramachandra `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对企业级标准操作流程（SOP）执行的多种大型语言模型代理架构进行系统对比

**💡 创新点**

提出了可直接对比架构的实验框架、技能定义质量打分量表以及成本–收益决策视角

**🔧 技术方法**

使用大语言模型、函数调用、技能化单体代理、集中式多代理编排与聚类式多代理组

**📊 数据集**

基于公开的 SOP-Bench 13 个企业级 SOP 任务集合，使用确定性 mock‑tool 调度

**📈 对比分析**

通过任务成功率（TSR）和成本（token 与延迟）对八种“架构+技能”配置进行配对、随机效应和显著性检验；结果表明单体函数调用代理在大多数场景下最具成本效益，只有在复杂分支 SOP 上高级技能或改进编排能显著提升成功率

**⚠️ 局限性**

实验仅覆盖单一模型家族和固定的 SOP-Bench，缺乏对多模型、真实服务非确定性和多次运行的评估；样本量有限导致统计显著性不足，且仅评估了结果与成本，未覆盖过程质量和人类干预

---

## 166. Repertoires of Contention: Building computational tools and infrastructures for community resistance to bitcoin mining

**arXiv ID:** 2611.13527 | [PDF](https://arxiv.org/pdf/2611.13527v1)

**作者:** Marina Johnson-Zafiris `[一作]` (Cornell University), Chris Csíkszentmihályi `[通讯]` (Cornell University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发并部署三种工具（DEC_Counts、SatHelper、GANDER），支持社区对Greenidge比特币挖矿设施的环境影响监督与抗议

**💡 创新点**

提出“repertoires of contention”框架，将计算技术与社区争议策略相结合，形成可操作的抵抗工具

**🔧 技术方法**

自然语言处理与情感分析、无监督聚类、卫星影像处理、无人水面平台自动温度测量

**📊 数据集**

DEC公开评论数据、Landsat等卫星影像、现场温度传感器采集的水温数据

**📈 对比分析**

DEC_Counts与人工分类一致率超过98%，SatHelper实现了不需要GIS软件的可视化，GANDER采集数千点温度与卫星数据相关性高

**⚠️ 局限性**

数据获取受限、监管机构与企业阻挠、证据权重不足导致难以直接促成监管决策、技术实现成本与现场操作复杂

---

## 167. TimeOmni-o1: Bidirectionally Coupling Time Series Foundation Models and LLMs for Multimodal Reasoning and Forecasting

**arXiv ID:** 2611.12902 | [PDF](https://arxiv.org/pdf/2611.12902v1)

**作者:** Ziyang Zhang `[一作]` (Tsinghua University), Chao Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 21761 | [OpenAlex ID](https://openalex.org/A5100460246)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并训练了一种双向耦合的时序基础模型与大语言模型的架构，既能理解带文本的时序数据，又能预测未来时序，同时生成解释性推理轨迹。

**💡 创新点**

双向耦合TSFM与LLM，将TSFM压缩的时序信息作为软提示送入LLM进行文本推理；再将LLM的推理结果通过反馈Q‑Former和门控交叉注意力注入TSFM，以生成条件化预测；构建大规模多模态时序推理数据集并配备推理轨迹，统一实现理解与预测任务。

**🔧 技术方法**

使用Chronos‑2时序基础模型、Qwen3.5‑9B大语言模型、Q‑Former窗口压缩、门控交叉注意力、LoRA微调、量化预测与CRPS评估、GPT‑5.4评测推理质量等技术。

**📊 数据集**

自研的TimeOmni‑Reasoning数据集（502,586训练+71,067测试样本，覆盖12领域理解任务与5领域事件驱动预测任务），以及公开的OpenTSLM、ST‑Bench、RATs40K、TSRBench、TelecomTS、HiTSR等数据；在SciTS、ST‑Bench T4、CiK等公开基准上进行验证。

**📈 对比分析**

与14+传统时序模型、专门预测模型、LLM及其单独TSFM版本进行对比；在自研数据集上理解平均得分84.2，预测CRPS在所有领域均优于baseline，平均提升约30%；在SciTS、ST‑Bench T4、CiK等公开基准上亦取得最优或接近最优。

**⚠️ 局限性**

仍受LLM令牌限制，长时序读取需压缩；需要预训练TSFM与LLM兼容，训练成本高；推理轨迹依赖LLM生成，质量受限；在可直接文本化短历史的场景下可能不如更大LLM；未在真实部署环境中验证。

---

## 168. Attribute-Evidence Learning with Multimodal Prompt for Generalized Category Discovery

**arXiv ID:** 2611.12975 | [PDF](https://arxiv.org/pdf/2611.12975v1)

**作者:** Wei Zhang `[一作]` (Beijing Jiaotong University), Konstantions N. Plataniotis `[通讯]` (University of Toronto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的多模态框架AptM-GCD，通过属性证据学习和多模态提示推理来进行广义类别发现，旨在从未标记数据中发现新类别。

**💡 创新点**

创新点在于将属性视为局部视觉证据，而非全局类别描述，从而减少噪声或无关属性线索的影响，并通过多模态提示学习将局部属性语义与全局视觉理解相结合。

**🔧 技术方法**

使用了属性证据学习（AEL）机制和多模态提示学习（MPL）机制，AEL通过对象条件的属性挖掘和局部证据建模来提取可靠的属性相关区域，MPL则将细粒度文本证据转化为视觉提示。

**📊 数据集**

在六个数据集上进行了广泛的实验，包括CUB、Stanford Cars、FGVC-Aircraft、CIFAR-10/100和ImageNet-100。

**📈 对比分析**

与现有的多模态GCD方法进行了比较，AptM-GCD在细粒度和通用基准测试中均表现出色，尤其是在新类别的发现上，显示出强大的泛化能力。

**⚠️ 局限性**

局限性在于在视觉证据不足的情况下（如背景伪装和严重遮挡的物体），模型仍然难以进行可靠的视觉-语义对应，未来的工作将探索更稳健的对象中心表示和组合视觉-语义推理。

---

## 169. EyeQ: Exploring Corneal Reflection for Mobile Context Sensing

**arXiv ID:** 2611.13640 | [PDF](https://arxiv.org/pdf/2611.13640v1)

**作者:** Jaewook Lee `[一作]` (University of Washington), Andrew D Wilson `[通讯]` (Microsoft Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用智能手机面向用户的摄像头捕获眼角膜反射，并通过服务器端增强与多模态视觉语言模型（VLM）实现对环境场景的零样本语义理解，构建了一套端到端的“EyeQ”管线

**💡 创新点**

①首次证明单镜头角膜反射可为移动端场景感知提供信息；②使用现成VLM零样本读取低分辨率角膜图像；③将拍摄、增强与VLM推理拆分，兼顾设备实时性与服务器端计算；④通过最小场景先验显著提升识别准确率

**🔧 技术方法**

前端：iPhone 16 Pro 48 MP摄像头 + MediaPipe Iris定位 + 自动曝光/对焦；后端：Real‑ESRGAN 4×超分、CLAHE 对比度增强、NAFNet 去噪、Unsharp Mask、Saturation Boost；VLMs：Google Gemini 3.1 Pro/Flash/Flash Lite；后置 LLM 判别：GPT‑4.1

**📊 数据集**

10 635 张角膜图像，来自 10 名受试者（含 12 种目标对象），未公开共享；对照集为 12 项场景对象的真实标签

**📈 对比分析**

通过 N=10 受试者的控制实验，室内/室外分类准确率达 96.4%；对象检测 F1 在办公室 0.295、厨房 0.157、校园 0.078；加入最小场景先验后 F1 分别提升至 0.385、0.316、0.347；与传统世界视角摄像头相比，误差主要为误分类而非完全缺失

**⚠️ 局限性**

低分辨率导致小物体、低对比度目标检测率极低；捕获条件（距离、角度、运动）对性能影响显著；误分类和假设的累积误差在实时推理中难以纠正；样本量小、仅测试 3 种场景，未覆盖更广环境；未进行 VLM 微调与隐私评估，仍存在潜在泄露风险

---

## 170. Lightweight Deep Learning Framework for Astrocyte-Centric 3D Reconstruction across Normal and Stress-associated Ultrastructural States

**arXiv ID:** 2611.13581 | [PDF](https://arxiv.org/pdf/2611.13581v1)

**作者:** Bowen Deng `[一作]` (University of Waterloo), Na Young Kim `[通讯]` (University of Waterloo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一种轻量级的 AstroSeg-vEM 框架，用于高分辨率 vEM 图像中星形胶质细胞的 2D 分割、3D 重建与形态学分析。

**💡 创新点**

将 transformer‑based 2D 分割与多阶段形态学保持后处理相结合，显著提升稀疏、分支复杂的星形胶质细胞在强异方差 vEM 数据中的自动化重建效率，并且仅需消费级 GPU。

**🔧 技术方法**

使用 Transformer‑based U‑Net（MiT‑B2 编码器）、前景-aware patch 采样、滑动窗口推理、基于 soma 引导的连通组件细化、体素连通聚合、二值孔洞填补等技术；评估采用 Dice、wDice、FDR、召回率等指标。

**📊 数据集**

四个小鼠海马体 stratum radiatum 的 vEM 数据集（D1–D4），包含正常（N‑AUS）和应激相关（S‑AUS）星形胶质细胞形态。

**📈 对比分析**

与 U‑Net、SegFormer、Swin‑UNet 进行对比，AstroSeg‑vEM 在 Dice 0.70、wDice 0.94、精确率 0.58、召回率 0.89 的基础上，后处理后 Dice 提升至 0.92、wDice 0.96；推理速度比基线快 7.5%，参数量 20% 以内；但在训练数据量减少时性能显著下降。

**⚠️ 局限性**

仅适用于单通道 2D 分割，需人工标注；对不同 vEM 采集参数的泛化有限；轴向分辨率受 70–100 nm 断层限制；多类别或跨物种分割尚未实现。

---

## 171. Two-Sided Amortized Resource Analysis: Verifying Resource Lower Bounds via Ill-Typing

**arXiv ID:** 2611.12925 | [PDF](https://arxiv.org/pdf/2611.12925v1)

**作者:** Zimu Chen `[一作]` (Peking University), Di Wang `[通讯]` (Peking University)

**通讯引用:** 47637 | [OpenAlex ID](https://openalex.org/A5100387207)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种两侧类型系统（带潜能）可统一推导程序的上界、下界以及不满足性（错误）证明。

**💡 创新点**

创新点在于将需求（necessity）箭头与潜能结合，利用公式在 ⊢ 左右的位置来区分上界（充足）与下界（必要）；并引入安全片段、值状态语义和 affine 资源分配，首次实现单一证明系统即可得到上界、下界和 refutation。

**🔧 技术方法**

采用 AARA（自动化均摊资源分析）潜能模型、两侧（sequent‑style）类型推理规则、值状态语义、共享关系（sharing）与安全片段的形式化，并用 OCaml 实现了注解驱动的证明搜索器。

**📊 数据集**

使用 19 个手工编写的基准程序（包括基本测试、列表处理、字符串比较、堆栈操作等），每个基准都提供不同的输入规模与预期成本。

**📈 对比分析**

与 RaML 的 AARA 分析（默认推导与带约束验证）进行对比：在上界方向上大部分基准可获得相同或更精细的结果；在下界方向上，系统能够得到 RaML 无法推导的恒定或线性下界，证明了两侧类型系统的优势；性能上，证明搜索仅在 19 个小基准上完成，未进行大规模时间测评。

**⚠️ 局限性**

局限性包括：需要人工提供中间类型与辅助公式（不完整的推理）；只能在安全片段内证明，无法处理高阶需求箭头的完整类型；在某些程序（如具有相互关联分支的条件）下只能得到非最优上下界；对大型程序的可扩展性尚未验证。

---

## 172. Brasilidade: Measuring Brazilian Identity for Large Language Models

**arXiv ID:** 2611.12906 | [PDF](https://arxiv.org/pdf/2611.12906v1)

**作者:** José Roberto Homeli da Silva `[一作]` (SoberanIA), Marcellus Amadeus `[通讯]` (SoberanIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并发布了Brasilidade基准，用于评估LLM在巴西葡萄牙语语言、语法、变体等方面的表现，并公开了第一个域的27个任务与8个数据集。

**💡 创新点**

首个使用母语语言学家策划、非翻译的巴西葡萄牙语评估基准，结合生成与概率两种推理模式的多层任务框架，并系统化标准化提示与评价指标。

**🔧 技术方法**

采用EleutherAI的LM Evaluation Harness框架构建任务，使用generate_until和loglikelihood两种输出方式；评估指标包括BERTScore、Exact Match、f1_weighted、accuracy；使用few-shot与zero-shot提示。

**📊 数据集**

由8个手工与人工生成+人工审核的语料库构成，覆盖4000+条独特条目，涵盖词形分析、变位、变体识别、句法错误纠正等，全部为巴西本土语料。

**📈 对比分析**

评估19个模型（2B–70B）在生成任务与概率任务的表现；结果显示few-shot生成任务普遍优于zero-shot概率任务，模型规模与性能呈正相关，但中等规模模型如Mistral‑Small表现优于大型模型；巴西特定模型平均水平未显著优于通用模型。

**⚠️ 局限性**

仅公开第一个域，未评估其余三个域；仅测试开源模型，未包含商业模型；缺少对提示变体和LLM-as-a-judge的支持；未与现有相关基准做直接比较。

---

## 173. Sanity Checks for Attention Surgery: Null Models and Matched Interventions for Representation-Motivated Design

**arXiv ID:** 2611.13543 | [PDF](https://arxiv.org/pdf/2611.13543v1)

**作者:** Basavaraj R M `[一作]` (Independent AI & ML Systems Researcher), Aryan Bharat Kumar `[通讯]` (Independent ML Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Transformer内部attention heads的self-value对齐方向进行干预，评估其在语言建模中的冗余性。

**💡 创新点**

引入三重可重复检查（可解性、背景几何比较、方向依赖性）来系统评估单向干预，并提出可重构门控实现自值消除。

**🔧 技术方法**

通过可学习门控的rank‑one向量裁剪、split‑half reliability、Spearman相关、配对随机种子训练等统计与实验手段。

**📊 数据集**

采用WikiText‑103和FineWeb‑Edu公开文本，进行冻结模型测量和4×10⁸ token的训练阶梯。

**📈 对比分析**

与固定随机方向控制比较，发现自值消除在50.9M模型上显著降低验证损失0.0029 nats（p=0.0023），而随机控制无显著差异。

**⚠️ 局限性**

结果仅在中等规模模型验证，未证明在更大规模模型中可行；干预效果可靠性在不同模型间差异大；只检验余弦相似度，未探讨其他统计。

---

## 174. Third Eye: Beyond Immediate Gains in Recursive Self-Improvement

**arXiv ID:** 2611.12988 | [PDF](https://arxiv.org/pdf/2611.12988v1)

**作者:** Sujato Dutta `[一作]` (Mahindra University), Neeraj Choudhary `[通讯]` (Mahindra University)

**通讯引用:** 362 | [OpenAlex ID](https://openalex.org/A5109584903)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估语言模型自更新的即时和随后一次更新后的效用，验证了自生成修正能否在连续更新后仍保持价值。

**💡 创新点**

创新点在于提出“Third Eye”双时段评估框架，采用独立打分和无更新基线来衡量自更新的递归效用，并系统检验了候选选择、更新配方和数据来源对效用的影响。

**🔧 技术方法**

技术手段包括LoRA微调（rank 16、scale 32）、基于验证的自修正、独立分数（开发集与最终测试集分离）、bootstrap置信区间与声明性对比检验。

**📊 数据集**

使用的数据集包括数学题库GSM8K、MBPP（代码）训练与验证、MATH与HumanEval（OOD）、以及2cMMLU（保持率）。

**📈 对比分析**

通过对比G₁（即时选择后效用）与G₂（进一步更新后效用）的差异并给出同时95%区间，实验发现大多数3–8 B模型在后续更新后效用显著下降，部分模型（如Qwen3‑4B相对Llama3.2‑3B）差异尤为显著。

**⚠️ 局限性**

限制包括仅评估五个3–8 B模型、仅使用两例LoRA更新、只关注两步更新循环、硬件与训练历史未独立随机、样本量有限、且仅覆盖即时与一次后续更新，无法推广到更大规模或更长循环的自更新场景。

---

## 175. Benchmarking Vendor-Guided Optimizations on Edge NPUs: From NPU Only to End-to-End Performance. Extended technical report

**arXiv ID:** 2611.13563 | [PDF](https://arxiv.org/pdf/2611.13563v1)

**作者:** Youngjin Jo `[一作]` (Chonnam National University), Jaein Kim `[通讯]` (Electronics and Telecommunications Research Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了边缘 NPU 在三种不同部署配置（Baseline、VG-NPU、VG-E2E）下的精度与时延表现

**💡 创新点**

创新点是将 Vendor-guided 优化拆分为仅 NPU 级模型变换与完整部署两层，系统比较两者对精度和时延的影响

**🔧 技术方法**

使用 vendor 提供的编译器、INT8 量化、图优化、预/后处理与运行时工具链进行实验

**📊 数据集**

采用 ImageNet‑1K（ResNet50）和 COCO‑2017（YOLOv5m、YOLOv8s‑pose）三种数据集

**📈 对比分析**

通过 Baseline、VG‑NPU 与 VG‑E2E 三个配置，测量 Runtime‑call 与 System‑level 时延，结果显示 VG‑NPU 在 10/12 组合中提升精度且大幅缩短 Runtime‑call 时延；VG‑E2E 在所有组合中均实现 10.4–70.7% 的 System‑level 时延下降，但有 8/12 组合精度下降

**⚠️ 局限性**

局限性：仅评估四款 Edge NPU 与三种视觉任务，采用 batch‑1 同步执行，无法单独分离各优化组件的贡献；未测功耗与能效；不同硬件平台导致的 System‑level 结果不可直接归因于 NPU 本身

---

## 176. One Trajectory, Two Requirements: Accuracy and Local Stability in Euler Sampling for Flow Matching Policies

**arXiv ID:** 2611.13311 | [PDF](https://arxiv.org/pdf/2611.13311v1)

**作者:** Mahindra Rautela `[一作]` `[通讯]` (Los Alamos National Laboratory), Mahindra Rautela (Los Alamos National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

论文研究流匹配机器人策略在推理时使用固定欧拉步数时，轨迹精度与局部稳定性对步数需求是否相同，并通过受控实验和对冻结的 π_0.5-LIBERO 策略的 Jacobian 分析来验证这一点。

**💡 创新点**

创新点在于提出并区分了轨迹精度与局部稳定性对欧拉采样步数的不同约束，展示了二者可能产生独立步数阈值，并在实际预训练策略上检验是否出现稳定性受限的情况。

**🔧 技术方法**

使用的方法包括：可解析的两维系统实验、在训练好的流匹配模型上添加可控收缩维度、对冻结策略进行基于 Jacobian 的局部稳定性评估、以及通过 RK4 参考解测量终点误差的精度测试。

**📊 数据集**

数据集主要是 LIBERO 10 任务（10 个家居操作任务），在每个任务采集了 3 个早期 rollout 输入，共 30 条样本；另外使用了预训练的 π_0.5-LIBERO checkpoint 与对应的 10×32 内部动作张量。

**📈 对比分析**

比较方法：对 5 条样本做完整的精度扫掠（5%、2%、1% 终点误差）并与通过 Arnoldi 估计的局部稳定性指标 G_max 进行对比。结果显示在 5% 容差下步数小于默认 10 步即可满足，但当容差收紧到 1% 时步数需提升至 87–112 步；而 G_max 在所有样本中均低于 1，表明未出现局部稳定性受限。

**⚠️ 局限性**

局限性包括：仅分析了 30 条早期 rollout 输入并使用单一噪声种子，配对精度实验仅覆盖 5 条样本；稳定性检验基于冻结时间的 Jacobian 近似，未考虑非正常动力学或全局动态；因此对完整策略执行和不同噪声条件下的稳定性结论尚有限。

---

## 177. Tracing Psychometric Inference in Large Language Models

**arXiv ID:** 2611.12937 | [PDF](https://arxiv.org/pdf/2611.12937v1)

**作者:** Chih-Hao Hsu `[一作]` (National Taiwan University), Pin-Hao A. Chen `[通讯]` (National Taiwan University)

**通讯引用:** 627 | [OpenAlex ID](https://openalex.org/A5074136941)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用跨人格范式对14种LLM（Llama 3与Qwen 2.5）进行实验，探测其在预测心理量表答案时是否内部具备心理构念的表示。

**💡 创新点**

创新性地将行为输出、对比方向提取、激活探测、Steering与synthetic控制相结合，系统量化模型内部心理结构与行为放大之间的关联，并检验指令调优对心理结构的影响。

**🔧 技术方法**

主要技术包括：对比方向提取（contrastive direction extraction）、层级激活线性回归探测（ridge regression）、Steering实验、Mantel检验、synthetic控制与SNR代理、相关矩阵对齐与OLS斜率分析。

**📊 数据集**

使用272名台湾大学生完成的7份传统中文心理量表（共114道题、16个子量表）的问卷数据。

**📈 对比分析**

通过计算模型预测与人类相关矩阵的Pearson相关和斜率进行比较。大模型在指令调优后表现出更高的相关度（r≈0.68，控制后r≈0.88）和更大斜率放大；激活探测斜率区间（1.38–1.77）比行为斜率区间（0.52–1.53）窄，表明激活层对行为的放大贡献有限。

**⚠️ 局限性**

限制包括：样本仅来自单一文化背景，模型仅来自两大家族且同家族内部不可完全独立，几何测度对小模型不稳，synthetic控制使用代理估计，未能证明因果路径，Steering实验仅为探索性。

---

## 178. AHEADance: Anticipatory Humanoid Dance from Streaming Audio

**arXiv ID:** 2611.13370 | [PDF](https://arxiv.org/pdf/2611.13370v1)

**作者:** Tianhu Peng `[一作]` (University College London), Chengxu Zhou `[通讯]` (University College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种名为ListenAhead的机器人原生流式舞蹈生成器，使人形机器人能在实时听歌的同时生成并执行舞蹈动作。

**💡 创新点**

创新点在于利用冻结的音乐预测模型提前预估即将到来的节拍，并将其与已听音乐的可靠性权重结合，增强了动作与节拍的同步；此外，Commit Forcing训练机制让模型能够从自身已生成的动作片段中持续更新，显著降低低活跃时段。

**🔧 技术方法**

技术方法包括：潜在扩散模型（Latent Diffusion）用于动作生成、全身跟踪器实现实时姿态控制、音乐预测模型提供节拍预测、Commit Forcing训练策略以及模仿学习框架。

**📊 数据集**

数据集主要为真实Unitree G1机器人在多首音乐下录制的舞蹈动作数据，以及对应的音乐音频；在此基础上训练并验证模型。

**📈 对比分析**

与传统无预测的生成方法相比，ListenAhead将低活跃时间从83.5%降至10.3%，在同步、曲风适配等方面的人工评估得分更高；在真实Unitree G1机器人上实现了连贯的舞蹈表演且无摔倒现象。

**⚠️ 局限性**

局限性包括：对音乐预测模型的依赖导致对未知或极端节拍变化的适应性有限；实验仅在Unitree G1机器人上验证，未充分测试跨平台通用性；评估数据量相对有限，且主要依赖人工评分，缺乏客观量化指标。

---

## 179. Constant query LCCs over the reals

**arXiv ID:** 2611.13281 | [PDF](https://arxiv.org/pdf/2611.13281v1)

**作者:** Zeev Dvir `[一作]` `[通讯]` (Princeton University), Zeev Dvir (Princeton University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文描述了一种新的构造，利用低度多项式评估和新颖的采样程序，构造了在实数域上的常量查询局部可纠正码（LCCs）。

**💡 创新点**

创新点在于首次在实数域上构造了具有增长维度的常量查询LCCs，之前的构造仅限于正特征域，并且被猜测在实数域上不存在。

**🔧 技术方法**

使用了低度多项式评估和随机采样程序，特别是通过随机二次曲线与整数网格的交点进行解码。

**📊 数据集**

使用了一个整数网格作为数据集，具体为 G = [M]_0^r，其中 M 是一个大整数，r 是维度。

**📈 对比分析**

与之前的构造相比，本文的方法在查询复杂度上表现出更好的性能，能够在固定的查询数 q 下实现更高的维度，且在解码时具有较高的成功概率（至少 3/4）。

**⚠️ 局限性**

限制在于该构造的解码系数随着维度的增长而增大，这可能导致对小噪声的高度敏感性。

---

## 180. Massive Activations Confound Sparse Autoencoders

**arXiv ID:** 2611.13548 | [PDF](https://arxiv.org/pdf/2611.13548v1)

**作者:** Nathan Delisle `[一作]` (University of Chicago), Chenhao Tan `[通讯]` (University of Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究并纠正稀疏自编码器（SAE）在语言模型激活中的评估偏差，揭示大规模激活导致的主成分1（PC1）主导问题并提出PC1绕过训练与后置特征筛选解决方案。

**💡 创新点**

首次量化PC1对SAE特征重要性与方差解释的扭曲，证明重要特征不可复制；提出PC1绕过训练和激活幅度惩罚、后置筛选等方法消除此偏差。

**🔧 技术方法**

使用稀疏自编码器（TopK、BatchTopK、JumpReLU、ReLU+ℓ1等）训练；PC1投影与PC1绕过；激活幅度惩罚；特征重要性评估（零化、重采样）；方差解释与内容方差分解。

**📊 数据集**

在GPT‑2、Pythia‑70M/160M/1.4B/6.9B、Qwen‑2.5‑7B、Llama‑3.1‑8B/70B、Gemma‑2‑2B/9B、DINOv2‑base视觉变压器的中层激活；训练数据来源于WikiText‑103。

**📈 对比分析**

通过方差未解释量（FVU）与内容FVU比值、特征共享率、模型感知损失（perplexity）等指标对比；PC1绕过后，特征共享率提升至73–99.5%，内容FVU下降0.8–33.9%，插入模型导致的性能下降减少最多44%。

**⚠️ 局限性**

仍有部分内容方差未解释；PC1绕过需要重新训练；后置筛选仅适用于已有字典；方法主要验证于特定模型与层级，尚未验证在更大规模或其他架构上的普适性；对可解释性指标的提升有限。

---

## 181. Seeing Without Seeing: A Methodological Review of Visual Mental Imagery Reconstruction from Brain Activity

**arXiv ID:** 2611.13691 | [PDF](https://arxiv.org/pdf/2611.13691v1)

**作者:** Shiyi Zhang `[一作]` (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences), Yihang Zhou `[通讯]` (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文综述并系统评估了脑信号对视觉想象内容的解码与重建方法，提出统一的两阶段管线、四类方法族、可接受性判据、目标无关评估框架、诊断性神经依赖量，并给出了十条报告检查表，随后对现有方法进行性能比较。

**💡 创新点**

创新点在于为缺乏真实目标的想象重建引入目标无关评估与代理缺口的定量分析，统一方法族与可接受性判据，定义神经依赖量作为诊断指标，并提供详细的报告检查清单，帮助规范和提升研究质量。

**🔧 技术方法**

使用的技术包括：编码模型逆推（Family I）、判别式解码与RSA（Family II）、生成式跨解码（Family III）与对齐适配（Family IV），配合深度学习生成模型（扩散、CLIP等）、线性回归、卷积网络和脑机接口相关分析。

**📊 数据集**

主要数据集：fMRI想象数据集如NSD‑Imagery、Deeprecon、Shen等；对应的感知语料库如NSD、BOLD5000、THINGS等；少量EEG/MEG数据集用于验证。

**📈 对比分析**

通过统一的人类强制选择（forced‑choice）评价框架，将感知与想象的识别率直接对比，发现感知‑想象之间的准确率下降在5.9–26.1个百分点之间，且方法排序出现变化，表明感知表现并不能可靠预测想象表现。

**⚠️ 局限性**

主要局限：缺乏图像级目标的纯概念生成实验，现有图像级评分仅来自fMRI；EEG/MEG缺少图像级目标；代理目标导致偏差与伪造重建；想象信号弱、易被噪声干扰；跨状态迁移和个体差异影响泛化；实时闭环应用尚未充分验证。

---

## 182. Continuous Angular Control in Text-to-Image Generation via Geometric Token Encoding

**arXiv ID:** 2611.13172 | [PDF](https://arxiv.org/pdf/2611.13172v1)

**作者:** Furkan Güzelant `[一作]` (Bilkent University), Adam Kortylewski `[通讯]` (University of Freiburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种几何令牌编码框架，使文本到图像扩散模型能够精确控制时钟时间和圆形分数等连续角度量。

**💡 创新点**

将角度以周期性正弦余弦特征编码并映射为角色特定的令牌，结合随机定位松弛 (SLR) 与手指交换优先目标，显著提升几何可控性。

**🔧 技术方法**

使用周期性编码、轻量级投影网络、Stochastic Localization Relaxation、手指交换偏好损失、LoRA 适配 Stable Diffusion 2.1 等技术。

**📊 数据集**

通过控制网络生成的多样化时钟图像与手工制作的圆形对象分片，构建包含 32 类、10k 张图像的训练集；测试集包含未见风格和类别。

**📈 对比分析**

与 SD 2.1、SDXL、SD 3.5、Flux.1 等预训练模型对比，基线 Acc@6°<1% 而本方法在四种评测器上达 71–78%；在分数生成上 Acc@0.10 从约20% 提升至 89.5%，在未见对象与目标上亦保持高性能。

**⚠️ 局限性**

仅适用于具备明确圆角结构的量，需任务专用训练数据；在视觉复杂场景或组件模糊时控制效果可能下降。

---

## 183. Which Successes Make Good References? Action-Sequence Alignment for Multi-Turn On-Policy Distillation

**arXiv ID:** 2611.13135 | [PDF](https://arxiv.org/pdf/2611.13135v1)

**作者:** Mufei Li `[一作]` (Georgia Institute of Technology), Wei Niu `[通讯]` (Amazon Web Services)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了针对多回合智能体的参考增强式策略蒸馏（OPD），通过基于动作序列的最长公共子序列（LCS）度量，选择与失败轨迹最兼容的成功轨迹作为教师的参考；

**💡 创新点**

创新点在于将参考选择建模为无目标最优运输问题，利用LCS作为兼容性指标来实现对状态路径的一致性近似，并证明只更新失败轨迹即可提升性能；

**🔧 技术方法**

核心技术包括：对策蒸馏、参考增强的OPD、最优运输框架、动作序列最长公共子序列（LCS）相似度计算，以及仅对失败轨迹进行训练；

**📊 数据集**

实验使用了ALFWorld（家居环境）和τ^3-bench（航空与零售客服）两个基准数据集；

**📈 对比分析**

与多种参考选择策略（随机、最短、BM25、编辑距离、LCP、BRTS等）以及零射手、SFT、普通OPD进行对比，LCS在ALFWorld和τ^3-bench航空任务上均显著优于所有基线，并在跨域零售任务中保持优势；

**⚠️ 局限性**

局限性包括：对LCS兼容性假设的可行性依赖于动作对状态的可恢复性，且对多样化成功轨迹的可用性要求较高；

---

## 184. Image Quality and the Reliability of Fabric-Density Measurement: Counting Threads from Smartphone Photos Through a Pick Glass

**arXiv ID:** 2611.13200 | [PDF](https://arxiv.org/pdf/2611.13200v1)

**作者:** Anoubhav Agarwaal `[一作]` `[通讯]` (Indian Institute of Technology Madras), Anoubhav Agarwaal (Indian Institute of Technology Madras)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用手机拍摄的放大镜图像，提出一种基于±5%容差的线程密度计数任务，开发并评估多种计数方法；

**💡 创新点**

创新点在于将传统的手工取样取代为基于手机拍摄的无标定计数，并通过半/双倍误差分析、训练自由光谱计数器以及仅使用合成图像微调的轻量级VLM等多方案来实现可重复的计数；

**🔧 技术方法**

使用的技术包括光谱峰检测、FFT与子晶格假设搜索、局部对比度归一化、基于YIN的投影周期估计、点-计数VLM（Molmo2-4B）、GPT-6.1-sol、CNN回归、以及无监督光谱计数器；

**📊 数据集**

构建了76个工厂场景（44个织物补丁）的数据集，包含多次重拍，提供手工标注的参考计数和无专家基线；

**📈 对比分析**

与纺织专家标注的参考计数比较，最佳方法为微调后的Molmo2-4B和GPT-6.1-sol（给出四幅放大图），其MAPE约为28%，仍高于±5%容差；其他方法（光谱计数器、CNN、VLM等）误差更大；

**⚠️ 局限性**

局限性包括仅来自单一工厂与单个放大镜、未在布料上验证计数、参考计数本身可重复性仅为≈9%、缺少多种织物结构（如篮织、针织）以及对高分辨率采集与提升的依赖性不足。

---

## 185. Curie's Jarvis: Autonomous Robotic Chemistry Experiments from Printed Protocols

**arXiv ID:** 2611.13251 | [PDF](https://arxiv.org/pdf/2611.13251v1)

**作者:** Rumi Loghmani `[一作]` (Carnegie Mellon University), Amir Barati Farimani `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种能够从印刷实验协议照片中直接完成化学实验的机器人系统，能够在普通实验台上操作，且无需预先设定容器位置或使用视觉标记；

**💡 创新点**

创新点在于四个方面：①利用多视角摄像头读取手写/打印标签并投票识别同形容器；②每一步操作前重新定位目标以应对容器移动；③根据所持物体动态调整执行速度和角度；④将完整实验步骤通过大语言模型解析并映射为机器人可执行的技能序列。

**🔧 技术方法**

核心技术包括：VLM（GPT‑4.1）进行协议文本解析和标签识别；SAM‑3进行多视角语义分割；多视角几何融合实现容器定位；基于规则和LLM的规划与技能调用；Franka Panda机械臂与Franka Hand实现抓取、倒料、搅拌、滴液等技能。

**📊 数据集**

主要数据来自两套儿童化学实验套件（Magic Beaker 与 Jellyfish）以及实验台上手写/打印的标签；未使用公开标准化数据集，而是采用实验室自制的协议图片和容器标签作为输入。

**📈 对比分析**

实验评估中，Magic Beaker 5/5 次成功，Jellyfish 4/5 次成功；在 160 次标签识别实验中 100% 正确，平均定位误差 0.2 mm；速度规划通过LLM动态调整，未与固定速度直接对比，但实验时间约为 30 min（Magic）和 17 min（Jellyfish）。

**⚠️ 局限性**

局限性包括：①未能测量和控制化学品量，实验基于预先分装的试剂；②仅在单一实验台、单臂和两套简单实验上测试，缺乏广泛验证；③对 VLM 的依赖导致需要网络连接和外部计算资源；④多摄像头逐视图处理导致定位延时；⑤未对手动移动物体或标签移除的情况进行完整验证。

---

## 186. Euclidean Hulls of Double Cyclic Codes of Arbitrary Block Lengths

**arXiv ID:** 2611.13481 | [PDF](https://arxiv.org/pdf/2611.13481v1)

**作者:** Awadhesh Kumar Shukla `[一作]` (Banaras Hindu University), Ashish Upadhyay `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了任意区块长度下的二重循环码的欧几里得外壳，给出了外壳生成多项式的完整公式；

**💡 创新点**

创新点在于突破Gao等人仅针对互素区块长度的限制，提出了适用于所有（包括非互素）区块长度的通用框架，并导出了非分离双循环码转为分离码以及LCD码的必要充分条件；

**🔧 技术方法**

主要技术包括多项式环同态、欧几里得内积、互补性、多项式的最小公倍数与最大公因子、贝祖等式以及对偶码生成多项式的计算；

**📊 数据集**

文章未使用公开数据集，而是通过符号计算给出若干具体例子（如 (3,9)、(3,17)、(7,7)、(4,4) 等）并在 Magma 软件中验证；

**📈 对比分析**

与之前仅限互素长度研究的结果进行比较，证明其为特殊情况；实验示例表明在非互素情形下可获得非分离LCD码，且外壳维数满足理论预期；

**⚠️ 局限性**

限制在于只讨论了二进制环 ℤ₂，且结果在多项式运算上对大块长度的复杂度未给出分析；未来可扩展到多元域或更高阶环。

---

## 187. Reconstructing Boundary Layer Wind Profiles with Machine Learning: A Comparative Study on IGRA Radiosonde Data

**arXiv ID:** 2611.13359 | [PDF](https://arxiv.org/pdf/2611.13359v1)

**作者:** Fedor Buzaev `[一作]` (HSE University), Denis Derkach `[通讯]` (HSE University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对气象站上升辐射气象观测（IGRA）中的大气边界层风速剖面进行重建，并评估现代表格学习模型与传统解析参数化方法的性能。

**💡 创新点**

①验证了在大气边界层风速重建中预测变量集合对精度的决定性影响；②展示了从2.5 m/s降至1.3–1.6 m/s的误差提升；③首次将一维 Fourier Neural Operator 作为后置校正器，显著降低近地面误差。

**🔧 技术方法**

CatBoost（GBDT）、TabM（轻量化神经网络集成）和 FT‑Transformer（基于 Transformer 的表格学习），以及一维 FNO 用作后处理。

**📊 数据集**

IGRA（Integrated Global Radiosonde Archive）辐射气象观测，共计 158,097 次升降气象测量，覆盖全球大部分陆地区域。

**📈 对比分析**

通过与经典的功率律与 Monin‑Obukhov 解析模型对比，发现现代表格学习在全球范围内的平均 RMSE 约为 1.3–1.6 m/s；相比之下解析模型约为 3 m/s。进一步的交叉区域验证表明模型在不同地理区域仍能保持良好泛化，且 FNO 后处理可将近地面误差进一步降低约 0.3 m/s。

**⚠️ 局限性**

① IGRA 站点分布偏北半球与陆地，南半球及海洋地区不足；② 训练/测试拆分基于单次气象观测，未完全按站点区分，可能导致误差估计过于乐观；③ 未与完整 NWP 或 ERA5 结果对比；④ FNO 仅在美国数据上验证，未与同等算力的简单垂直混合器进行对比；⑤ 仅使用单次快照预测，未利用时间或空间上下文。

---

## 188. One Rater to Score Them All: Frozen Large Language Model Anchoring for Video Quality of Experience

**arXiv ID:** 2611.12916 | [PDF](https://arxiv.org/pdf/2611.12916v1)

**作者:** Parsa Hassani Shariat Panahi `[一作]` (Iran University of Science and Technology), M. Hassan Najafi `[通讯]` (Case Western Reserve University)

**通讯引用:** 1626 | [OpenAlex ID](https://openalex.org/A5012903661)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种冻结的大语言模型（LLM）评分器，利用播放事件日志生成统一的QoE参考分数，作为跨研究的共同尺度。

**💡 创新点**

创新点在于将开源权重的LLM直接应用为无训练的、仅读事件日志的评分器，克服了传统需要共享内容或人类评测的限制，并通过单一评分器实现不同研究之间的分数等价化。

**🔧 技术方法**

使用技术包括：固定词表的开源LLM（Qwen3-235B-A22B-2507）、贪婪解码、预设评分准则（六个区间、硬编码比特率要求）以及对每个研究的单调逻辑回归映射。

**📊 数据集**

所用数据集包括P.1203（TR04/06/ VL04/13/14）、LIVE‑NFLX‑II、Waterloo‑IV、Poqemon，以及外部验证集PNATS‑UHD‑1‑Long和AVT‑VQDB‑UHD‑1，共计3552个会话。

**📈 对比分析**

在多项对比中，评分器对主观评分的Spearman相关率最高可达0.81、平均误差约10点；与标准化模型相比，在VL04上更优，在其他数据集上相近；与现有15个客观模型相比，位居中游；对单个模型的比较表明其在AVT‑VQDB‑UHD‑1上排名第三。

**⚠️ 局限性**

主要局限包括：对仅编码差异（相同分辨率下不同码率）无法完全表达；托管推理环境导致重复性略低；设备偏差（如UHD电视）仍存在；样本量不足60的研究需线性映射，可能降低精度。

---

## 189. The WoW: A Large-Scale World Knowledge Corpus for Full-Lifecycle LLM Training

**arXiv ID:** 2611.13487 | [PDF](https://arxiv.org/pdf/2611.13487v1)

**作者:** Tianze Sun `[一作]` (Harbin Institute of Technology), Sam Chen `[通讯]` (OpenCSG)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 WoW（World Knowledge）——一个基于教育知识图谱的多学科中文教育语料库，并通过四阶段流程将海量文档转化为高质量、可追溯且富含推理需求的监督实例。

**💡 创新点**

创新点：① 用教育知识图谱作为统一分类与补全框架，系统化缺口识别；② 设计四阶段（上下文生成、问题生成、质量控制、多样化）全流程，既保证上下文完整性，又提升推理深度；③ 采用文档级 provenance，支持全生命周期模型训练；④ 提供交互式图谱平台，方便社区基于知识点筛选和检索。

**🔧 技术方法**

技术手段：语义分割+连贯关系预测、知识抽象（关键方程/命题/结果表）、检索式证据集合、依赖链问题生成、严格的合成/变换/依赖测试、分数评估（A,B,C）与多样化格式化；并在多语言环境下保持原文不变，统一元数据。

**📊 数据集**

数据集：约 2.7 TB 原始教材、网络文章、专著等，来源 14 个，经过流程生成 188.7 K 监督实例（约 146 M 词）。

**📈 对比分析**

比较方法：在 WoW hold‑out 以及 8 个外部基准（MMLU、CMMLU、C‑Eval、ChemBench、ScienceQA、SciEval、MoleculeQA、MedMCQA）上，使用相同训练与评估配置。结果显示：Llama‑3.2‑3B 在多答题准确率提升 15.9 p，Llama‑3.1‑8B 提升 5.2 p，Qwen3‑4B‑Base 提升 2.8 p；在科学类基准上可提升 5.5 p，整体平均提升约 0.1–0.3 %。大模型对广度知识影响微小，小模型表现最显著。

**⚠️ 局限性**

局限性：① 对未覆盖领域（如医学、临床）会出现知识遗忘；② 仅评估选择题/结构化问题，对开放式推理和多模态任务尚未覆盖；③ 数据仍以文本为主，缺少深度多模态内容；④ 规模虽大但仍不足以覆盖所有学科完整知识点。

---

## 190. RaMOD: Moving-Object Detection from Raw mmWave Radar Cubes under Ego Motion

**arXiv ID:** 2611.13271 | [PDF](https://arxiv.org/pdf/2611.13271v1)

**作者:** Yuan Fang `[一作]` (University College London), Kevin Chetty `[通讯]` (University College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为RaMOD的实时雷达原始RAD立方体物体检测与速度估计框架；

**💡 创新点**

创新点包括：①基于估计的机动速度进行静态回波抑制的自适应抑制机制；②跨尺度特征融合和跨视角辅助监督以提升RAD空间检测；③引入速度头同时预测未折叠与折叠的径向/前向速度，并提供折叠概率；

**🔧 技术方法**

采用3D卷积网络、Cross‑Scale Attentive Residual Fusion、Cross‑View Transformer、锚框检测与速度回归，整体在单芯mmWave雷达的RAD三维空间上实现检测与速度输出；

**📊 数据集**

使用自行构建的BRIDGE‑Liv数据集，该数据集包含同步单芯mmWave雷达、LiDAR、摄像头等传感器，标注了RAD立方体框、速度与折叠标签；

**📈 对比分析**

与多种RA/RA+RD与RAD立方体的基线（如RADM、Mamba‑RODNet、RADDet等）进行对比，RaMOD在IoU 0.30下的RAD立方体mAP达80.8%，比最强基线提升15.7个百分点，速度MAE 1.13 m/s，折叠准确率99.87%；

**⚠️ 局限性**

局限性包括：依赖于单芯mmWave雷达的特定波束与Doppler覆盖，当前假设前向机动且未考虑更复杂的转向或多向运动；处理多天线和高分辨率时的计算与能耗仍需进一步优化。

---

## 191. How Much of Your VLA Is kNN

**arXiv ID:** 2611.13160 | [PDF](https://arxiv.org/pdf/2611.13160v1)

**作者:** Denis Grachev `[一作]` (Sereact GmbH), Omar Moured `[通讯]` (Sereact GmbH)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了视觉语言动作模型（VLA）在LIBERO基准上的检索能力，提出了训练无关的kNN检索策略，用已演示的动作片段重现任务，并评估其对现有VLA背后视觉语言模型表示与动作的影响。

**💡 创新点**

① 证明LIBERO高成功率中可大量归因于检索重用演示动作；② 通过检索比较展示VLA视觉语言骨干在微调前后的表示差异；③ 引入离线LAC指标评估检索表示质量并可用于训练更优检索键。

**🔧 技术方法**

使用kNN检索、视觉语言嵌入（ResNet‑18、CLIP、VLM层）、多模态距离融合、软最大加权动作拼接、LAC（动作距离比例）等技术。

**📊 数据集**

使用LIBERO仿真基准（130个语言条件Franka操纵任务）及其扰动版LIBERO‑Plus，结合公开的9个VLA模型演示数据。

**📈 对比分析**

与9个公开VLA模型在LIBERO与LIBERO‑Plus上对比，kNN多模态策略在LIBERO获得91.7%成功率、LIBERO‑Plus 82.4%，在扰动下表现最稳健，且与部分VLA相当或更优。

**⚠️ 局限性**

仅在LIBERO稠密演示环境下验证，检索无法跨越演示支持范围；评估主要关注视觉与语言变化，未涵盖需全新轨迹或稀疏演示的情形，因而泛化能力有限。

---

## 192. Nepali-English Code-Mixed Speech Recognition: Data Preparation and Adaptation of a Speech Foundation Model

**arXiv ID:** 2611.13512 | [PDF](https://arxiv.org/pdf/2611.13512v1)

**作者:** Jeevan Neupane `[一作]` (AI Lab Pvt Ltd), Bipin Bashyal `[通讯]` (AI Lab Pvt Ltd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究如何在172.8小时的尼泊尔-英语混合语料上微调SraVaani模型，并通过数字与符号处理提升识别准确率。

**💡 创新点**

创新点在于保留原有分词器，仅通过自动生成双语数字读法并基于模型输出选择最匹配的标签，从而避免重建词表。

**🔧 技术方法**

使用了FastConformer编码器、TDT-CTC双重损失、BF16混合精度训练、以及自动化数字化生成与评分技术。

**📊 数据集**

数据集包括来自OpenSLR、YouTube、工程讲座、播客、新闻、对话等六类场景共115,107条语音，时长172.8小时。

**📈 对比分析**

与Whisper large‑v3‑turbo和IndicConformer对比，SraVaani在自定义语料上6个周期可将WER从48.6%降至17.1%，12周期进一步降至16.3%。

**⚠️ 局限性**

限制在于未完全保证说话人不重叠，数字标签的自动化选择可能引入偏差，且在外域FLEURS上仍存在写法不匹配导致的误差。

---

## 193. Building World Models from Intrinsic Motivation

**arXiv ID:** 2611.13113 | [PDF](https://arxiv.org/pdf/2611.13113v1)

**作者:** Reece Keller `[一作]` (Carnegie Mellon University), Aran Nayebi `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对24种内在动机目标与3种世界模型表示方法在DreamerV3框架下进行统一评估，探究其对零-shot、少量样本及多任务控制的影响；

**💡 创新点**

首次系统比较不同内在动机与表示学习的组合，并发现好奇心+观察重建是最稳健的预训练策略；

**🔧 技术方法**

使用模型基强化学习DreamerV3、NES-Dreamer、R2-Dreamer以及RND、NGU、ICM等内在动机模块；

**📊 数据集**

评估数据集包括DeepMind Control Suite（25个连续控制任务）、Craftax Classic（开放式探索游戏）以及RoboKitchen；

**📈 对比分析**

通过零-shot、任务特定头重置、在线微调等方法对比，结果显示好奇心+重建在多任务和少量样本中获得最高平均收益，NGU在含记忆的视觉任务中表现最佳；

**⚠️ 局限性**

局限包括对单一世界模型结构的依赖、对长时序和组合任务的记忆不足、以及在Vision任务中离线想象训练导致适应性下降等。

---

## 194. Tabular Foundation Models in Financial Predictions

**arXiv ID:** 2611.13550 | [PDF](https://arxiv.org/pdf/2611.13550v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 195. MiniZoo: An Evolving Dataset of Mini Apps for the Research Community

**arXiv ID:** 2611.13125 | [PDF](https://arxiv.org/pdf/2611.13125v1)

**作者:** Luca Ferrari `[一作]` (IMT School for Advanced Studies Lucca), Luca Verderame `[通讯]` (University of Genoa)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并维护了一个持续增长的 MiniZoo 数据集，收录了 WeChat 生态中的 mini‑app 包及其元数据，并计划扩展到其他超级应用平台；

**💡 创新点**

提出了面向 mini‑app 的持续版本记录与平台无关的数据结构，借鉴 AndroZoo 的成功经验，实现了自动化爬虫和与 AndroZoo 基础设施的无缝集成，首次为研究社区提供可随时更新的 mini‑app 数据库；

**🔧 技术方法**

使用 Python 编写爬虫，结合 ADB、Frida、Appium 进行设备交互和程序监测；通过 SHA‑256 哈希识别包、JSON/CSV 结构化元数据，并与 AndroZoo 的 API、存储系统实现统一管理；

**📊 数据集**

MiniZoo 自身数据集，初始版本包含约 3,200 个 mini‑app（共 23,500 个不同的包），并随时更新以捕获新版本；

**📈 对比分析**

通过提供统一的包和元数据，研究者可在同一数据集上对 TaintMini、WeMinT 等工具进行静态污点分析实验，比较数据流、覆盖率与执行时间；目前已知数据规模中位数包大小为 1.94 MB，包大小分布跨度从 KB 到十几 MB；

**⚠️ 局限性**

目前仅覆盖 WeChat，缺乏对其他平台的完整性；版本收集间隔不均匀，导致同一 mini‑app 的历史记录可能缺失；爬虫依赖 host 应用的接口与 UI，若其变更需及时维护；初始规模有限，需进一步扩充以提升研究代表性。

---

## 196. NCP: Verifier-Guided Optimization of Weight-Space Control Policies

**arXiv ID:** 2611.13236 | [PDF](https://arxiv.org/pdf/2611.13236v1)

**作者:** Ekrem Aksoy `[一作]` `[通讯]`, Ekrem Aksoy

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了Neural Control Plane（NCP），一个闭环系统，通过观察LLM推理，合成并评估权重空间干预规则，维护版本化规则表。

**💡 创新点**

创新点在于将控制平面与数据平面分离，使用W-ISA执行事务性权重干预，并在闭环中引入确定性候选合成、显式验证与版本化流表，以实现递归自我改进。

**🔧 技术方法**

技术包括W-ISA权重干预运行时、激活激励提取、基于方向的候选合成模板、路由任务与强制动作验证、谱检查、PyTorch 2.15与Python 3.12实现。

**📊 数据集**

使用SmolLM‑135M‑Instruct作为模型，WikiText‑2测试集（4,095目标切片）和手写九问任务作为验证与评估数据集。

**📈 对比分析**

通过在验证集上测量重复率和困惑度来比较候选规则，实验显示某些候选能将重复率从0.413降至0.219，困惑度仅变动-0.057%，但未提升准确率，最终所有规则被归档并保持原始策略。

**⚠️ 局限性**

局限性包括候选生成仅基于有限的确定性模板，未实现准确率提升，阈值设定保守，实验仅在小规模模型与有限任务上验证，未展示长期递归自我改进或大规模部署。

---

## 197. Placing LLM Lookup Tables by Per-Layer Prefetch Deadlines

**arXiv ID:** 2611.13413 | [PDF](https://arxiv.org/pdf/2611.13413v1)

**作者:** Wen Huang `[一作]` (Microsoft), Jack Liu `[通讯]` (Independent Researcher)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了如何在大型语言模型（如 DeepSeek‑V4.1、Qwen3.8）中，将按输入标记索引的稀疏查找表（Engram/PLE）从主机 DRAM 迁移到 NVMe 或远程 RDMA 记忆层，并通过预测每层的预取窗口决定表的存储层级与 GPU 缓存分配，从而在保持或微幅提升推理吞吐量的同时显著减少主机内存占用。

**💡 创新点**

创新点包括：① 针对每层预取窗口的测量与建模，形成基于“零停顿”条件的层级放置与缓存分配规则；② 设计 SlackTier 作为 SGLang 的 CUDA 图内层级存储层，实现与原始 gather 逐位一致；③ 通过共享 NVMe、RDMA、分层 GPU 缓存等多种组合验证，证明即使在极低窗口（0.3 ms）层也能在 NVMe 上无明显延迟；④ 通过释放的 180 GiB 主机内存做 KV 缓存，提高 36% 以上推理吞吐。

**🔧 技术方法**

使用技术包括：CUDA Graph、SGLang 0.5.22、NVMe SSD、InfiniBand RDMA、FP8 权重/KV 缓存、页缓存与 LRU 替换、动态窗口测量探针、以及基于预取窗口的数学模型。

**📊 数据集**

主要数据集为：OpenHands agent trajectories、SWE‑agent trajectories、WildChat 对话，以及 Qwen3.8 的 1.1 B 代币预训练日志；此外使用 128、256 并发会话的合成推理负载来评估吞吐与延迟。

**📈 对比分析**

与原生 SGLang（主机 DRAM 存表）对比，SlackTier 在 4 × H200 芯片上实现：① 主机内存从 223 GiB 减少到 34 GiB；② 对 agent 与 chat 的每标记生成时间（TPOT）最高提升 1.4%；③ 总执行时间最高提升 0.6%；④ 在 RDMA 或四块 NVMe 时，单标记延迟与原生几乎无差异。还验证了不同磁盘数量、缓存分配比例、以及共享/远程表的性能影响。

**⚠️ 局限性**

局限性包括：① 仅适用于按输入标记索引且预取窗口已知的稀疏表，无法直接用于 dense 或激活相关权重；② 需要在目标负载下测量每层窗口与读时延，部署成本相对较高；③ 对极低 GPU KV 缓存预算或统一内存设备效果有限；④ 依赖硬件特性（NVMe 读写速率、RDMA 带宽）与实现细节，跨平台可移植性需进一步验证。

---

## 198. Megrez: A Knowledge-Guided Agent for Formal Mathematics over MK Set Theory

**arXiv ID:** 2611.13613 | [PDF](https://arxiv.org/pdf/2611.13613v1)

**作者:** Tiancheng Xu `[一作]` (Beijing University of Posts and Telecommunications), Wensheng Yu `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研发了 Megrez，这是一款知识引导的 Rocq 自动化正式化代理，针对 Morse–Kelley 集合论的低资源库，支持跨章节累计正式化。

**💡 创新点**

创新点包括：①源可重建的知识图谱融合源码、.glob 与手工元数据；②1–5 计步短爆发式证明循环配合指数级目标本地反射；③三重检查接受流程与可恢复多定理批处理；④多路检索融合与中文别名支持。

**🔧 技术方法**

技术手段：使用 LLM（Codex）编写通用编码代理，MCP 接口与 Rocq 交互；SQLite 复合检索，Reciprocal Rank Fusion 与 BM25；多线程工作池、CAS 版本控制、哈希绑定；结构化反馈与本地账本。

**📊 数据集**

数据集：MKTopos 书本式点集拓扑正式化，共 262 个 .v 文件，3,894 声明，102,122 行证明；并在该库上生成 10k 与 100k 规模的合成图谱进行评测。

**📈 对比分析**

评测方法：与 legacy‑search/v1 与 legacy‑premise/v1 进行对比；在 100k 节点下 exact lookup 加速 58.8×，延迟降至 1.23 ms；SQL 语句控制在 3–7 条；批处理通过 319 条自动合同测试，单定理通过率高。

**⚠️ 局限性**

局限性：依赖手工注释和中文别名；.glob 仅提供符号出现信息而非最小逻辑依赖；候选作用域基于节点集而非完整模块可见性；可信边界仍需外部沙盒保证。

---

## 199. Exploration via Exploitation: The Blessing of Reward Diversity in Personalized Federated RL

**arXiv ID:** 2611.12964 | [PDF](https://arxiv.org/pdf/2611.12964v1)

**作者:** Jiin Woo `[一作]` (Carnegie Mellon University), Yuejie Chi `[通讯]` (Yale University)

**通讯引用:** 7085 | [OpenAlex ID](https://openalex.org/A5053809095)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在多智能体联邦强化学习框架下，提出了可在共享动态但奖励异质的情境下实现线性加速的个性化UCBVI算法和基于奖励多样性实现无探索的个性化价值迭代算法。

**💡 创新点**

创新点在于：①将奖励异质性视为协作加速的资源而非障碍；②通过共享转移统计、个性化UCB奖励实现线性速度提升；③引入奖励多样性假设，证明在足够多样的奖励下，纯贪婪即可实现对数级无探索回报。

**🔧 技术方法**

核心技术包括联邦UCB价值迭代、奖励多样性假设下的覆盖分析、纯贪婪价值迭代、动态共享计数与局部贝尔曼回溯，以及理论上对回报与样本复杂度的分析。

**📊 数据集**

使用合成的离散MDP数据集：状态转移采样自Dirichlet分布，奖励稀疏随机分布在SAH个单元格上，构造多代理异质奖励场景。

**📈 对比分析**

与Fed-UCBVI及单代理方法对比实验显示，在奖励多样性充分时，探索自由算法在大规模代理数下实现了与UCBVI相当或更优的累计回报且无显式探索开销；在低多样性场景下UCBVI仍保持稳健，形成相互补充的性能空间。

**⚠️ 局限性**

主要限制在于奖励多样性假设的严格性，若奖励差异不足则无探索算法退化；此外假设奖励已知且对动态估计误差鲁棒性有限，实际应用需进一步验证。

---

## 200. Plumbline: When the Loss Lies, Validating LLM Training Correctly

**arXiv ID:** 2611.13393 | [PDF](https://arxiv.org/pdf/2611.13393v1)

**作者:** Ruochen Wen `[一作]` (Amazon), Benoit Dumoulin `[通讯]` (Amazon)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了 Plumbline，一套用于大型语言模型训练的正确性验证系统，涵盖生命周期、架构和收敛三个层面的检查，并在 Amazon 生产环境中实现 Day‑0 Onboarding 与 Root‑Cause Tracing。

**💡 创新点**

创新点在于提出模块化、可重用的检查与诊断框架，利用可信参考实现对训练各阶段逐项对比，形成跨模型、跨框架的统一正确性验证方法，并通过该系统在实践中发现 61 个隐藏与显式问题。

**🔧 技术方法**

使用了分布式训练框架（Megatron‑LM、DeepSpeed、TorchTitan 等）、Hugging Face 官方实现作为参考、噪声阈值与精度容差、梯度与损失比对、bitwise 对比、CPU 侧单步验证、Run Reader、Flight Recorder、内存快照、Python 堆栈转储等技术，以及编程代理自动执行检查。

**📊 数据集**

采用了代表性工作负载数据集，包括 General Chat (Tülu 3)、Dolci‑Think‑SFT、LongReward‑10k、GSM8K 等，用于收敛检查和各类错误激活。

**📈 对比分析**

通过与官方发布模型、框架声明、公开食谱、独立实现或重新计算等作为参考，采用噪声基准和容差阈值进行比较；在 3 个框架、5 个模型上共检测到 61 个问题，其中 32 为静默，定位成本低（如一次单节点检查约 0.5 小时），且大部分问题已 upstream 修复。

**⚠️ 局限性**

局限性包括检查范围有限（无法覆盖无 Hugging Face 实现的模型或全模型级错误）、仍需人工审核确认、对依赖版本不兼容导致的错误可能无法被捕获，以及自动化诊断仍在完善中。

---

## 201. How Well Do Benchmarks Evaluate LLMs' Use for Personal Health Informatics?

**arXiv ID:** 2611.13652 | [PDF](https://arxiv.org/pdf/2611.13652v1)

**作者:** Fatemeh Mirhosseini `[一作]` (Texas A&M University), Yasaman S. Sefidgar `[通讯]` (Texas A&M University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对现有个人健康信息学领域的LLM基准进行系统评估与设计空间描述；

**💡 创新点**

创新之处在于将个人健康信息学理论与经验与基准设计维度结合，揭示关键缺口并提出人本化改进方向；

**🔧 技术方法**

采用文献检索、雪球法、主题编码、AI辅助筛选与分类、定性与定量对比分析等方法；

**📊 数据集**

分析了八个公开/非公开的个人健康基准（如CGM Speak、HealthBench等），使用其中的任务示例与数据；

**📈 对比分析**

通过任务类别与设计维度的对比矩阵展示基准覆盖度与不足，未进行系统性能度量，结论是现有基准的生态有效性有限；

**⚠️ 局限性**

局限性包括仅聚焦个人健康信息学，未涵盖安全、文化等维度；基准样本有限，分析方法在一定程度上带有主观性。

---

## 202. Byzantine-Robust Muon: Resilient Momentum Orthogonalization for Distributed Learning

**arXiv ID:** 2611.13137 | [PDF](https://arxiv.org/pdf/2611.13137v1)

**作者:** Grigory Malinovsky `[一作]` (King Abdullah University of Science and Technology), Peter Richtárik `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在分布式与联邦学习中针对拜占庭攻击的MuOn优化器，提出将其置于非欧几里得信任域框架下，并给出非凸与星凸目标、二阶光滑性、误差容忍的收敛分析；在理论上实现了对鲁棒聚合、动量正交化与LMO误差的统一处理；在实验上验证了在CIFAR‑10（ResNet‑18、ViT‑Tiny）和TinyStories语言模型上的鲁棒性；

**💡 创新点**

创新点主要包括：1）把MuOn与非欧几里得信任域统一为一阶框架；2）提出基于双重均值差异的非欧几里得异质性度量；3）在拜占庭与异质数据下给出非凸与星凸目标的收敛保证；4）引入可加的LMO误差模型和加速的梯度外推；5）在实验中将鲁棒聚合与MuOn几何结合，展示了比传统SGD更优的攻击鲁棒性能。

**🔧 技术方法**

技术手段包括：非欧几里得信任域优化、动量正交化（MuOn）、鲁棒聚合（RFA、几何中值、桶化方法）、线性最小化Oracle（LMO）与Newton–Schulz迭代、加权衰减、二阶光滑性分析、加法式Oracle误差模型、以及对第二矩聚合和梯度外推的理论扩展。

**📊 数据集**

实验数据集：CIFAR‑10（ResNet‑18、ViT‑Tiny）采用Dirichlet‑0.1非IID划分；TinyStories字节级语言模型（12层、3800万参数）；另外在ResNet‑18/CIFAR‑10上进行固定-诚实、共享检查点等对照实验。

**📈 对比分析**

比较方法：将MuOn‑NS与SGD、全局归一化SGD、sign更新以及AdamW等基线在同一鲁棒聚合（桶化RFA）下进行对照；指标为CIFAR‑10宏观准确率和TinyStories字节级困惑度。实验结果表明，MuOn‑NS在面对多种拜占庭攻击（sign‑flip、ALIE、IPM、模仿、方向性攻击）时取得最高的攻击后准确率，且在TinyStories上取得最低的困惑度；相比之下，AdamW在攻击下性能下降更大，且在无攻击时MuOn‑NS仍保持较高准确率。

**⚠️ 局限性**

局限性：理论分析基于无偏梯度、有限方差、Lipschitz光滑、异质性约束等假设，实验未验证所有常数；Newton–Schulz迭代仅在5次内使用，未保证LMO误差极小；攻击策略有限且可重复，未覆盖最坏情况；种子数较少，统计置信区间有限；实验仅在模拟环境下进行，未在真实分布式部署中验证安全性。

---

## 203. Deletion Sensitivity Is a Property of the Model: A Three-Condition Measurement across Seven Open Reasoning Models

**arXiv ID:** 2611.13223 | [PDF](https://arxiv.org/pdf/2611.13223v1)

**作者:** Hanzhang Jia `[一作]` (University of Sussex), Bo Ma `[通讯]` (Zhejiang Gongshang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对多轮工具调用的大模型进行推理历史删除敏感性测量，提出三种渲染条件（保留、跨轮删除、半程删除），并通过官方 BFCL 检查器盲测任务成功率和重复率，计算敏感度并给出统计区间。

**💡 创新点**

①首次直接、盲测完整推理历史删除；②将删除敏感度视为模型层面的可度量属性；③提出三种行为签名（补偿、循环、静默失败）来解释删除对模型行为的影响。

**🔧 技术方法**

多轮推理历史渲染技术、官方 BFCL 检查器、Bootstrap 区间估计、基于两细胞屏幕的敏感度判断、重复率监控。

**📊 数据集**

794 个 BFCL v4 任务（基准、长上下文、缺失函数、缺失参数四个子集），7 个开源权重模型（Qwen3、GLM、Baichuan、R1-Distill、Olmo），每个模型三种随机种子。

**📈 对比分析**

通过保留与跨轮删除两细胞对比，计算成功率差值 ΔS 与重复率增量 ΔR，使用 95% Bootstrap 区间进行显著性检验。结果显示删除敏感度范围为 -4.4 到 +51.3 分，区分敏感与容忍两类模型，并通过剂量反应验证敏感模型在删除比例变化时的稳健性。

**⚠️ 局限性**

①仅评估 BFCL 函数调用任务，可能不适用于其他任务类型；②依赖官方检查器，无法检测所有可能的失败模式；③未深入探究删除背后的机制，只给出了行为签名；④实验规模有限，未覆盖更大或更复杂的模型。

---

## 204. Linear Probes to Bayesian Manifolds: Geometry of Arithmetic in Large Language Models

**arXiv ID:** 2611.12998 | [PDF](https://arxiv.org/pdf/2611.12998v1)

**作者:** Anshul Kumar `[一作]` (Carnegie Mellon University), Deeksha Varshney `[通讯]` (Indian Institute of Technology Jodhpur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个四阶段贝叶斯管道，对大语言模型在两位数加法与乘法任务中激活的几何结构进行系统评估，检验可解码性、形状、所有权与因果使用四个独立命题；

**💡 创新点**

创新点在于将线性探针的结果拆分为四个判定，利用形状模型的边缘似然和相关概念消除来区分模型真实使用的结构与仅被共享的结构，并通过干预验证几何形状的因果效应；

**🔧 技术方法**

主要技术包括线性探针、SVD与置换检验、低维形状拟合（圆、环、螺旋、平衡体）、相关概念投影消除、贝叶斯模型比较以及对残差空间中子空间的插值干预；

**📊 数据集**

使用公开的GPT‑J 6B、Llama 3.1 8B、Pythia 6.9B三大模型，在每个模型上构造两位数加法（10 000题）与乘法（3 023题）的合成算术数据集；

**📈 对比分析**

与传统单一准确率评估相比，该方法在四个维度上给出更细粒度的判定，仅46/1 330细胞同时通过所有四个判定，显示出模型几何结构的真实使用比例在19%左右；

**⚠️ 局限性**

局限包括仅检验单个词级输出位置的几何、仅考虑预定义形状与相关概念、对超位置的处理依赖线性投影、以及干预与评估仅局限于最终输出的准确率与对数it的变化。

---

## 205. ADIndex: Layer-Adaptive Sparse Attention Indexing via Group-Level Distillation

**arXiv ID:** 2611.12956 | [PDF](https://arxiv.org/pdf/2611.12956v1)

**作者:** Jin-Fei Qi `[一作]` (Nanjing University), Yu-Feng Li `[通讯]` (Nanjing University)

**通讯引用:** 21155 | [OpenAlex ID](https://openalex.org/A5100355149)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于层自适应与蒸馏驱动的轻量级索引方法，用于加速DeepSeek Sparse Attention（DSA）中 token 级稀疏注意力的前填充与解码；

**💡 创新点**

创新点包括：①层级重用校准策略，针对每层不同的交叉查询重叠度自适应选择查询组大小与共享 token 银行预算；②基于群体级蒸馏的轻量化群体索引器，用更少的索引头学习共享 token 银行的检索需求；③推理阶段采用共享 token 银行 + 查询特化细化的两步策略，显著减少冗余的全前缀扫描；

**🔧 技术方法**

技术手段包括：DSA 的 token 级索引器与 Sparse MLA；层级重用校准（最大联合覆盖估计与速度提升优化）；群体级蒸馏（聚合查询表示、教师分布生成、KL 对齐）；多头 RoPE 编码；SGLang 实现；

**📊 数据集**

数据集：DeepSeek-V3.2 与 GLM-5.1 模型在 128K token 级别的 FineWeb-Edu 与 Cosmopedia V2 进行离线校准和训练；任务评测使用 LongBench、LongBench v2、RULER（含 16K/32K/64K/128K 长度）；

**📈 对比分析**

与原始 DSA 以及 HISA、MISA、IndexCache、PIVOT 四种索引加速方法对比；在 LongBench、LongBench v2、RULER 上保持近乎无损的任务性能（例如 DeepSeek: 56.23 vs 56.07，GLM: 56.18 vs 56.10）；在 256K 上预填充速度提升至 6.22×，解码速度提升至 3.40×，端到端服务速度提升 1.54×；

**⚠️ 局限性**

局限性包括：在较短上下文（< 32K）时速度提升有限甚至略慢；需要为每层单独校准与训练，增加部署复杂度；对模型结构和超参数依赖较强，迁移到其他 LLM 时需重新校准；

---

## 206. Two-Tier Task Offloading for Satellite-Assisted Marine Networks: A Hybrid Stackelberg-Bargaining Game Approach

**arXiv ID:** 2611.13019 | [PDF](https://arxiv.org/pdf/2611.13019v1)

**作者:** Zhen Wang `[一作]` (Dalian Maritime University), Haixia Peng `[通讯]` (Xi'an Jiaotong University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

设计并实现了基于低轨卫星辅助的两层任务卸载框架，结合NOMA/FDMA通信与Stackelberg-Bargaining博弈实现AUV、MASS与LEOS之间的能量与收益最优化。

**💡 创新点**

首次在海底-海面-低轨三层网络中提出两层卸载结构；将Stackelberg游戏与议价博弈耦合，既优化卸载比例又定价；采用NOMA+FDMA提升频谱利用率并考虑海底声学信道特性。

**🔧 技术方法**

多接入边缘计算（MEC）、非正交多址接入（NOMA）、频分多址（FDMA）、游戏理论（Stackelberg游戏与议价博弈）、二分搜索与线性搜索优化算法。

**📊 数据集**

采用仿真生成的海底-海面-卫星三维坐标、任务规模50 Mbits、带宽1 MHz/10 MHz等参数；未使用真实海上数据集，仅基于参数仿真。

**📈 对比分析**

通过MATLAB仿真与八种基准方案（OTS, OSFP, OSRP, OSLP, HAS, LPS, FPS, RPS）比较，所提方案在AUV、MASS、LEOS的效用上均优于基准，收敛速度快、能量消耗低。

**⚠️ 局限性**

仅考虑单LEOS场景；算法假设信道与位置已知，未对海上移动性与动态环境做适应；实现仍需一定的集中信息，易受网络规模扩展与实时性限制。

---

## 207. Fermat ProbCover: Coverage-Based Active Learning on Fermat Distance Graphs

**arXiv ID:** 2611.13448 | [PDF](https://arxiv.org/pdf/2611.13448v1)

**作者:** Gokul Bhusal `[一作]` (University of California Los Angeles), Stephan Wojtowytsch `[通讯]` (University of Pittsburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在低标注预算下提出PC‑Fermat，一种基于Fermat距离的覆盖式主动学习方法。

**💡 创新点**

创新点在于用Fermat距离取代欧氏距离构建覆盖图，从而更好捕捉数据分布的几何结构。

**🔧 技术方法**

技术包括构造加权ε邻域图、使用截断Dijkstra计算Fermat覆盖图，并采用贪心覆盖采样策略。

**📊 数据集**

在八个分类数据集上测试，涵盖原始特征与VAE嵌入的COIL‑20、Opt‑Digits、Landsat、USPS、Jasper、Pendigits、MNIST、Fashion‑MNIST。

**📈 对比分析**

与随机采样、Margin、KMeans++、ProbCover（欧氏）和TypiClust比较，PC‑Fermat在大多数数据集和极低预算下准确率最高。

**⚠️ 局限性**

局限在于计算成本较高，且在欧氏空间可分离度很高的场景下仍被Margin超越。

---

## 208. Prediction-Powered LLM Ranking with Adaptively Collected Human Feedback

**arXiv ID:** 2611.13186 | [PDF](https://arxiv.org/pdf/2611.13186v1)

**作者:** Gabriel Sargent `[一作]` (University of North Carolina at Chapel Hill), Yufeng Liu `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于预测增强的LLM排序框架（PP‑Rank），能够在适应性采样下结合少量人类反馈、LLM评判器预测和未标记任务来实现有效且可检验的排名推断。

**💡 创新点**

创新点在于：①将预测增强推断与适应性采样理论融合，提供在非 i.i.d. 环境下的渐近正态性和置信区间；②设计了数据驱动的对每对模型的权重 λ，既能抵消评判器偏差又能降低因逆倾向权重产生的方差；③给出了基于最小化协方差迹的最优 λ 公式，并证明其能在保持有效性的同时实现方差最小化；④提出了通过最小采样概率来动态降低实验成本的方法。

**🔧 技术方法**

核心技术包括：逆倾向加权估计方程、预测增强（Prediction‑Powered Inference）与 LLM 评判器的结合、马尔科夫中心极限定理用于处理适应性采样带来的依赖、联合正态性推导、同时推断（simultaneous inference）以及在线最优采样率的估计。

**📊 数据集**

使用了公开的 UltraFeedback‑Extended 数据集（含 64k 任务、10 模型响应与 10 评判器分数）进行半合成实验，并通过 Skywork‑Reward‑V2‑8B 模拟人类偏好；另外还构造了完全合成实验（4 模型、10 维上下文），用来验证理论与算法在不同评判器质量下的表现。

**📈 对比分析**

与仅使用人类反馈（无逆倾向加权）、仅使用评判器（PPI++）以及标准的 IPW 估计器等方法比较。PP‑Rank 在所有采样策略下均保持约 90% 的覆盖率，且相较于基线可将置信区间宽度缩小 15–20%，均方误差降低 30–40%，排名集合宽度缩小约 10%，并显著提升对“top‑k”与“pairwise”排名声明的认证率；在成本方面，PP‑Rank 可在不降低精度的前提下将评估成本降低 40–50%。

**⚠️ 局限性**

局限性包括：①仍需一定比例的人类采样，完全依赖评判器无法实现；②对评判器质量高度敏感，若评判器偏差大需谨慎权重调整；③理论假设如平稳分布、独立同分布的上下文、可逆倾向采样等在实际平台中可能不完全满足；④目前仅针对全局排名，未覆盖非平稳或个性化排名场景；⑤在线最优采样率的估计需要额外计算，并可能在极端稀疏采样下表现不佳。

---

## 209. Combinatorial optimization with bounded subdeterminants

**arXiv ID:** 2611.13065 | [PDF](https://arxiv.org/pdf/2611.13065v1)

**作者:** Samuel Fiorini `[一作]` (Universite Libre De Bruxelles), Stefan Weltge `[通讯]` (Technical University Of Munich)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在整数规划系数矩阵为“总 Δ-可调” (即所有子式行列式绝对值≤Δ) 的前提下，许多经典 NP‑难组合优化问题的多项式时间可解性。

**💡 创新点**

创新点在于：①首次将总 Δ-可调性与多种图问题（k‑着色、图同构、反馈顶点集、奇环截断、支配集、诱导匹配、集群顶点删除、旅行商、2‑CSP 等）关联起来；②证明总 Δ-可调性可导致输入图具有多种结构性质（如度≤Δ+1、循环数受限、树宽/线性 mim‑宽受限、可分解为森林/弦图等），并基于这些结构设计多项式算法；③展示不同问题需要的多种算法范式（动态规划、枚举、二项式约束求解等）。

**🔧 技术方法**

主要技术包括：
- 子式行列式分析与图结构推断（利用 Erdős‑Pósa 定理、网格子图排除、偶环截断的奇网格子图等）；
- 结构化分解（树宽、线性 mim‑宽、弦图、块树等）；
- 对特定整数规划的变换（将大于两项约束拆解为两项约束，消除 XOR 变量等）；
- 动态规划与枚举（在小度/小树宽/小块树下进行 DP，枚举覆盖集合、最小割等）；
- 对特定问题的已知多项式算法（如在弦图上求集群顶点删除、在 2‑可约整形整数规划上求解）。

**📊 数据集**

该工作为理论研究，无实验数据集；所有结果均为理论上多项式时间复杂度。

**📈 对比分析**

方法与性能：
- 对于每个固定的 Δ，给出的算法在输入规模 n 上是多项式时间；
- 具体复杂度取决于 Δ（例如树宽、mim‑宽的上界是关于 Δ 的函数，且常数阶可有效计算）。
- 与现有研究对比：在 Δ=1（全可调）时已知多项式可解；本文将可解范围扩展到所有固定 Δ，并对多种问题给出新的多项式算法。

**⚠️ 局限性**

局限性：
- 仅适用于固定 Δ，Δ 可变时仍是开放问题；
- 结果高度依赖特定的整数规划表述，若采用不同表述可能不成立；
- 对某些问题（如 3‑SAT、k‑CSP(k≥3)、非标准图类问题）尚无对应结果；
- 对大规模实例的实际运行时间尚未评估，理论多项式常数可能较大。

---

## 210. Designing UIs with GenAI: Understanding the Use, Benefits, and Limitations of Prompt-to-Design Generators

**arXiv ID:** 2611.13211 | [PDF](https://arxiv.org/pdf/2611.13211v1)

**作者:** Jingzhou Ye `[一作]` (University of Central Florida), Xueqiang Wang `[通讯]` (University of Central Florida)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过对17名 UI 设计师的半结构化访谈，系统性地探讨了 prompt‑to‑design 生成器在日常设计工作流程中的使用方式、提示策略、设计者态度以及组织层面的治理与采用差异。

**💡 创新点**

首次以真实使用者视角、从工作流程、提示策略与组织治理三维度全景呈现 s（生成式 AI 生成器）对 UI 设计实践的深层影响，并提出了针对提示、品牌、代码可维护性与隐私的改进建议。

**🔧 技术方法**

采用了定性研究方法：半结构化访谈、录音转写、主题编码与归纳式主题分析，未使用任何算法或机器学习模型。

**📊 数据集**

使用的数据为17位 UI 设计师的访谈文本与屏幕共享记录（不涉及公开数据集）。

**📈 对比分析**

该研究无算法对比或性能度量，结果通过主题分析得到定性发现；未涉及定量指标或客观性能评估。

**⚠️ 局限性**

主要限制包括样本规模相对较小、仅来自美国的设计师、主要基于自述的访谈数据，缺乏跨文化、多行业与定量验证的支持。

---

## 211. From Process Models to Composable Agentic Workflows: Combining BPMN, Agents and Enterprise Knowledge

**arXiv ID:** 2611.13196 | [PDF](https://arxiv.org/pdf/2611.13196v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 212. Cognitive Multi-Agent Search and Rescue: Language-Grounded Hierarchical Reinforcement Learning for Collaborative Robotics

**arXiv ID:** 2611.13161 | [PDF](https://arxiv.org/pdf/2611.13161v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 213. ChipOmni: A General-Purpose AI Agent for Electronic Design Automation

**arXiv ID:** 2611.13020 | [PDF](https://arxiv.org/pdf/2611.13020v1)

**作者:** Pengkun Wang `[一作]` (University of Science and Technology of China), Yang Wang `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 ChipOmni，一个知识增强、上下文演化的多智能体框架，能够在可执行 EDA 环境中统一处理模拟电路、RTL、测试平台、形式化断言等多种任务，并通过验证驱动的闭环实现自动规划、执行、修复与上下文优化。

**💡 创新点**

核心创新在于：1）将多模态输入（文本、图表、文档）与领域知识库耦合，形成可执行规划；2）构建四个协同智能体（PlanExec、Validator、DebugMemory、ContextOptimizer），实现规划、验证、诊断与上下文演化的闭环；3）通过工具反馈进行迭代修复，实现长周期、跨阶段的自主设计；4）在单一框架下统一多种 EDA 任务，突破传统工具链的碎片化。

**🔧 技术方法**

采用了多智能体协同技术、动态规划与任务分解、知识检索与增强、可执行环境接口（对接多款 EDA 工具）、验证驱动的诊断与修复机制、上下文压缩与优化、以及对多模态输入的适配（文档、图表、代码）。

**📊 数据集**

使用的基准数据集包括：AnalogCoder-Pro（30 个模拟电路任务）、RTLLM2（50 个 RTL 设计）、VerilogEval v2（156 个 RTL-到-RTL 任务）、FVEval（系统 Verilog 断言）、AssertLLM2（83 真实 RTL 设计断言）、AutoBench（156 个 HDLBits 测试平台）、以及一个三层级的 RISC‑V 微 SoC 长周期任务。除此之外，还集成了超过 10,000 本教材与技术文档、20+ EDA 手册、160+ 软件库、30+ 命令行工具和 40+ 可重用技能。

**📈 对比分析**

在所有基准上，ChipOmni 在 Pass@1、Pass@5 等指标上均明显优于对应的后端模型基线，并且在 VerilogEval v2 与 AnalogCoder-Pro 等任务中实现了 100% 成功率；在 RTL、断言与测试平台生成任务中分别提升了 10–30% 的语法与功能通过率；与 MAGE、ChipMATE、VeriSure 等方法比较，ChipOmni 在运行时间和 token 使用上更为高效，且在长周期 RISC‑V 任务中成功完成 GDSII 物理验证。

**⚠️ 局限性**

局限性主要体现在：1）依赖可执行工具和可验证的检查，难以处理缺乏完整工具反馈或难以量化的需求；2）对权限、审计与安全边界的支持尚未系统评估；3）模型对复杂的设计约束仍可能需要人工介入；4）尚未完成硅芯片制造与现场验证；5）在更广泛的工业环境下，鲁棒性与可扩展性仍需进一步验证。

---

## 214. Partition, Refine, and Detect: Preserving Novel Category Capacity in Open-Vocabulary Object Detection

**arXiv ID:** 2611.12976 | [PDF](https://arxiv.org/pdf/2611.12976v1)

**作者:** Wei Zhang `[一作]` (Beijing Jiaotong University), Jianping Fan `[通讯]` (Fudan University)

**通讯引用:** 11519 | [OpenAlex ID](https://openalex.org/A5100728564)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种分区与细化（PR-OVD）框架，用于在训练基类时预留空间并强化新类别区分，实现更好的开词汇对象检测。

**💡 创新点**

创新点包括：①基于人类视觉认知的全局分区与层级细化策略；②利用结构提示（Structure Prompt）生成高质量的新类别视觉原型；③在视觉空间采用非对称原型学习避免新类别与背景混淆。

**🔧 技术方法**

技术手段包括：CLIP跨模态对比学习、视觉与文本空间的分区与细化、结构提示生成新类别图像、非对称原型学习以及信息NCE双向对齐。

**📊 数据集**

主要数据集：OV-COCO、OV-LVIS；并在COCO-Stuff、ADE20K、Pascal-Context以及NUS-WIDE上验证跨任务迁移。

**📈 对比分析**

与多种State‑of‑the‑Art OVD 方法（如 CLIPSelf、RALF、SAS‑Det 等）进行比较，在 OV‑COCO 上提升 Novel AP 约 6.5%，在 OV‑LVIS 上提升 Novel mAP 约 7.9%；同时在分割、分类任务也获得显著提升。

**⚠️ 局限性**

局限性：对生成图像的质量依赖结构提示和扩散模型；在极小或拥挤场景下仍存在漏检或误检；对大量新类别的扩展需要更多生成样本，计算成本略高。

---

## 215. Repository-Context-Guided Image Understanding for Agentic Software Development

**arXiv ID:** 2611.13026 | [PDF](https://arxiv.org/pdf/2611.13026v1)

**作者:** Fumiki Yohena `[一作]` (IBM Systems Engineering), Michiaki Tatsubori `[通讯]` (IBM Research)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出利用代码仓库渲染的高质量参考截图辅助编码代理，在视觉质量受损的UI截图下定位源文件，并在自制基准上进行评估。

**💡 创新点**

创新点在于将同一代码库的可渲染状态生成参考图像作为视觉先验，弥补降质图像的文本识别缺失，而非仅依赖OCR或VLM。

**🔧 技术方法**

使用Playwright渲染脚本生成参考图像、Gemini Embedding 2进行多模态检索、Mini‑Swe‑Agent框架以及GPT‑5.4 LLM，并对图像进行下采样、重采样与JPEG压缩处理。

**📊 数据集**

基于SWE‑bench Multimodal中的scratch‑gui仓库，注入36个视觉UI缺陷的任务实例构成实验基准。

**📈 对比分析**

通过金文件匹配率和Bag‑of‑Words F1 与VLM、OCR+VLM、OCR+VLM+字典等基线对比，结果显示在四倍下采样下，参考图像将金文件匹配率从28%提升至42%，文本识别提升约12个百分点。

**⚠️ 局限性**

局限在于仅适用于具备构建与渲染脚本的Web应用，参考图像可能包含相同缺陷，覆盖范围有限；实验仅涉及单一应用、有限的降质方式和5张参考图，未评估大规模仓库或真实摄像捕获等场景。

---

## 216. Tri-Stack-PIM: A Heterogeneous Multi-Die 3D-Integrated Processing-in-Memory Architecture for Efficient LLM Inference

**arXiv ID:** 2611.13644 | [PDF](https://arxiv.org/pdf/2611.13644v1)

**作者:** Shuao Jia `[一作]` (Beijing University of Posts and Telecommunications), Jianwang Zhai `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了一种三层堆叠的Processing‑in‑Memory（PIM）架构——Tri‑Stack‑PIM，用于高效推理大规模语言模型（LLM）的注意力计算。

**💡 创新点**

创新点在于：①将DRAM、接口/控制/非线性运算与先进节点计算分别拆分到三颗芯片，消除了HB keep‑out 区域对基底芯片的面积与布线竞争；②在中间Buffer die上提出UBAR（Unified Buffer‑Aware Ready‑based）微操作乱序调度器，实时解决跨芯片依赖与资源冲突；③采用联合架构‑数据流共探索（Bayesian 优化），在硬件与工作流之间找到 Pareto‑最优配置。

**🔧 技术方法**

主要技术包括：3D hybrid bonding、成熟节点 Buffer die、先进节点 Compute die、SIMD MAC 数组、内置 SRAM 缓冲、Softmax/Reduction 引擎（SRE）、UBAR 动态调度、贝叶斯优化的共探索框架。

**📊 数据集**

实验使用的模型和数据集：OPT‑6.7B（MHA）、LLaMA3‑8B（GQA）和 PaLM‑8B（MQA），并在不同批量大小（1‑32）及序列长度（128/512/1024）组合下进行评估。

**📈 对比分析**

对比方法采用每 token 平均注意力延迟作为性能指标，与 GPU‑only、AttAcc 以及 H^2‑LLM 三个基线进行对比；Tri‑Stack‑PIM 在所有配置下实现了几何平均加速 4.95×（GPU‑only）、2.57×（AttAcc）和 1.68×（H^2‑LLM），单一配置最高可达 14.15× 对 GPU。

**⚠️ 局限性**

局限性：需要额外的 HB bonding 步骤和 TSV 结构，增加了成本与设计复杂度；将 PHY 与非线性运算迁移至成熟节点虽然降低面积，但对大面积 PHY 仍敏感；在完全计算密集的 MQA 场景下 SIMD 扩展受限；热量与封装管理仍需进一步优化。

---

## 217. Early Rug Pull Detection on Solana

**arXiv ID:** 2611.13342 | [PDF](https://arxiv.org/pdf/2611.13342v1)

**作者:** Saba Heydaridoost `[一作]` (Tehran Institute for Advanced Studies), Behnam Bahrak `[通讯]` (Khatam University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对 Solana 上新发行的代币，在创建后的 12.5、35.5、93 分钟内利用交易图与表格特征构建模型，评估图神经网络（GNN）是否能比传统非图模型更早检测到 rug‑pull（拉皮骗局）。

**💡 创新点**

① 在极短的观察窗口（仅 12.5 分钟）已观察到 rug‑pull 与非 rug‑pull 交易图在结构上存在显著差异；② 通过严格的时间分割、标签遮蔽和多重检验校正，首次系统性比较同一时间点下 GNN 与最佳非图模型的性能，验证图结构在极早期并未提供统计学显著的优势。

**🔧 技术方法**

图神经网络（GCN、GraphSAGE、GAT）+ 端点特征；传统模型包括 XGBoost、FT‑Transformer、随机森林、梯度提升、Logistic 回归、决策树等；实验采用 5 种随机种子、配对自助抽样、Bonferroni / Holm 校正。

**📊 数据集**

样本来自 SolRPDS 的 2,700 个 Solana 代币（2022‑2024），采集了 2.86M 条代币转账记录、Birdeye 1 分钟流动性与价格历史；标签依据 24 小时内的流动性流失、价格崩跌与持续稀释的三重阈值定义。

**📈 对比分析**

在面板数据（同一代币在所有窗口均可观测）上，GNN 在 12.5 分钟窗口的 AP 与 XGBoost 差距均在零的置信区间内，未见统计显著优势；在每个窗口内，GNN 的 AP 通常略高于对应的非图模型，但差异在 Holm 校正后均不显著。实验显示：更长观察窗口并未系统提升预测性能，且样本量有限导致检测小幅提升的能力不足。

**⚠️ 局限性**

① 样本量小（面板约 136‑159 代币）导致检验功效不足；② 仅使用了 SolRPDS 的 2024 代币，无法覆盖更大更新生态；③ 交易图为部分可见，未包含转账金额和合约代码信息；④ 缺失值与覆盖率与标签相关，可能引入偏倚；⑤ 只评估了 24 小时内流动性崩盘的 rug‑pull，未覆盖慢速或无流动性流失的欺诈模式。

---

## 218. Perspicacious: An Educational Intelligence Platform for Instructional Designers

**arXiv ID:** 2611.13549 | [PDF](https://arxiv.org/pdf/2611.13549v1)

**作者:** Yumou Wei `[一作]` (Carnegie Mellon University), Paulo F. Carvalho `[通讯]` (Carnegie Mellon University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个名为 Perspicacious 的教育智能平台，整合了 KCluster 的知识组件（KC）发现和 Generate‑Then‑Validate 的题目生成，提供单一的 Web 界面供教学设计者上传学习目标或题库并快速获得生成题目或 KC 报告。

**💡 创新点**

创新点在于：①将两项研究前沿算法（KCluster 与 Generate‑Then‑Validate）包装成低技术门槛、可异步使用的工具；②通过成本与时间可调节的资源分配模型，让使用者根据预算与时效自行权衡；③在设计时强调可解释性与信任度，提出可视化输出、模板化上传和告知机制。

**🔧 技术方法**

主要技术包括：基于 Phi‑2（2.7B LLM）的两大引擎；Vertex AI 的分布式 GPU 计算与异步批处理；Django + Google Cloud Run 的 Web 前端；以及用于成本预测的线性回归模型。

**📊 数据集**

使用了13个真实课程的题库（共计576道题）进行生产实验，另外在用户研究中使用了64道由学习目标生成的题目和其对应的KC。

**📈 对比分析**

虽然未对算法性能做量化比较，但通过用户研究得到 SUS 平均 78.8（B+ 级）表明可用性良好；成本模型显示对576题的KC提取费用在1.98–4.11美元之间，完成时间 36–158 分钟，且随实例数量线性下降。

**⚠️ 局限性**

局限性包括：①题目生成引擎尚未上线；②缺乏生成过程的可追溯性与解释性，影响信任度；③对小规模作业的启动开销高，首次体验可能缓慢；④KC 生成的动作动词单一，降低了对 Bloom 分类的区分度。

---

## 219. Minimum Synthesis Time for Single-Deletion Correcting Codes of Prescribed Size

**arXiv ID:** 2611.13434 | [PDF](https://arxiv.org/pdf/2611.13434v1)

**作者:** Yuling Li `[一作]` (Capital Normal University), Gennian Ge `[通讯]` (Capital Normal University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `67630363-6be0-4f51-ab05-7198250671a5` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在 DNA 合成中给定码大小时，最小合成时间的理论极限，包括无约束码和单删除校正码。

**💡 创新点**

提供了不同冗余量（线性、对数级、常数级）下的精确上界与下界，并揭示单删除校正所产生的对数项以及相关的常数系数。

**🔧 技术方法**

采用大偏差理论（Cramér 速率函数）、指数倾斜分布、差分 VT 码以及组合计数和信息熵技巧进行解析推导。

**📊 数据集**

本文为纯理论工作，无使用实际 DNA 采样或合成实验数据集。

**📈 对比分析**

通过与已有的基于参考序列的合成模型的数值模拟结果（图 1-4）进行比较，展示了理论界限与实际编码方案之间的接近程度，证明了对数项的必要性。

**⚠️ 局限性**

对常数冗余（c=1）时的首项仍未确定，且对于冗余在 log_q(q‑1) 与 1 之间的情况，是否存在可行码仍是未解决的问题。

---

## 220. MADSNet: A Multi-Scale Axial-Attentive Dense Spiking Network for Spike Camera Image Reconstruction

**arXiv ID:** 2611.12980 | [PDF](https://arxiv.org/pdf/2611.12980v1)

**作者:** Sirui Li `[一作]` (Chinese University of Hong Kong), Yajing Zheng `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种多尺度轴向注意力稠密突触网络（MADSNet），用于将高时间分辨率的二进制脉冲流重建为连续强度图像。

**💡 创新点**

创新点包括：① 多尺度Spike Stem融合时空信息；② 轴向注意力模块以低成本捕获全局依赖；③ 稠密瓶颈提升特征复用；④ 脉冲到连续的融合块实现平滑转换；⑤ 结合IBRA‑LIF神经元保持积累‑仅计算优势，同时显著提升重建质量。

**🔧 技术方法**

技术实现：基于IBRA‑LIF的SNN架构；深度可分离SEW块、轴向注意力、稠密连接、像素重排上采样、深度监督；损失函数为L1 + 感知（VGG）损失。

**📊 数据集**

使用数据集：灰度重建采用SREDS（模拟 + 实测），彩色重建使用RBSS（仿真）和BSS（实测 Bayer‑模式脉冲）。

**📈 对比分析**

与传统滤波、事件相机重建、ANN模型（Spk2ImgNet、HyperE2VID）及SNN基线SSIR对比。MADSNet在SREDS上PSNR 37.23 dB，较SSIR提升4.30 dB、较Spk2ImgNet提升1.94 dB；在RBSS上PSNR 29.36 dB，超越Spk2ImgNet与SSL‑CSC。推理延迟比Spk2ImgNet快15×、能耗仅1/40；相比SSIR延迟与能耗略高，但仍显著低于ANN。

**⚠️ 局限性**

局限性：网络更深导致推理延迟和能耗高于轻量级SNN；在常规GPU上仍不如最轻量化SNN；对IBRA‑LIF的依赖需要进一步研究其在更大规模或不同任务上的泛化。

---

## 221. Predicting Single-Cell Spatial Gene Expression from H&E using Pathology Foundation Models

**arXiv ID:** 2611.13649 | [PDF](https://arxiv.org/pdf/2611.13649v1)

**作者:** Ashley P. Tsang `[一作]` (Genentech), Kenneth T. Gao `[通讯]` (Genentech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

通过对现有病理学基础模型进行轻量化、任务特定的适配，实现了单细胞水平从H&E图像预测空间转录组的能力。

**💡 创新点**

创新点在于结合细胞级别的token池化、40×高分辨率输入、Poisson计数监督、非线性MLP解码以及LoRA参数高效微调，显著提升了单细胞表达预测精度，并通过多维度评估（基因集、细胞类型、通路）揭示了模型优势与局限。

**🔧 技术方法**

使用的技术包括ViT‑g/14架构的病理基础模型（如H‑Optimus‑0、Prov‑GigaPath、DINOv3）、LoRA适配、两层MLP预测头、Poisson负对数似然损失、细胞边界token池化、20×/40×分辨率切片、以及多任务交叉验证。

**📊 数据集**

使用的数据集为两类结肠组织：TMA组（176 UC核心，约8.6M细胞）和切除组（含UC、克罗恩病、健康对照，约2.9M细胞），配对H&E图像与10x Genomics Xenium 5K空间转录组。

**📈 对比分析**

与现有H&E‑to‑ST方法（SpatialEx、GHIST、OmiCLIP、BLEEP）以及基础模型的线性/非线性探针基线对比，模型在5K基因组上实现平均基因PCC 0.174±0.014、平均细胞PCC 0.280±0.033、AUROC 0.800±0.009，均优于对照组。

**⚠️ 局限性**

局限性包括：单细胞空间测序技术的噪声（转录物泄漏、细胞分割误差）、对不同组织和人群的泛化性不足、对基因表达细微差异的预测受限、以及模型不确定性和可解释性待进一步研究。

---

## 222. Evidential Deep Learning with conditional entropy for glaucoma detection

**arXiv ID:** 2611.13513 | [PDF](https://arxiv.org/pdf/2611.13513v1)

**作者:** Hai Siong Tan `[一作]` (Gryphon Center for Artificial Intelligence and Theoretical Sciences), Kuancheng Wang `[通讯]` (University of California San Diego)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究提出一种基于条件熵的证据深度学习框架，用于视网膜眼底图像的青光眼检测，并评估其不确定性校准与可解释性。

**💡 创新点**

创新点在于用条件熵替代传统的Dempster不确定性作为训练正则化项，提升模型的校准误差并产生更符合生理学的Grad‑CAM关注区。

**🔧 技术方法**

技术包括MobileNetV3-Small骨干网络、证据深度学习（EDL）、条件熵正则化、Grad‑CAM、MC Dropout、Deep Ensemble与温度缩放等不确定性量化方法。

**📊 数据集**

使用EyePACS‑AIROGS‑light‑V2（完整眼底图）和RIM‑ONE‑DL（光学杯区裁剪图）两个公开数据集，并在CRFO与ACRIMA数据集上评估OOD检测。

**📈 对比分析**

与基线、温度缩放、MC Dropout和Deep Ensemble比较，条件熵EDL在ECE、Brier分数、Grad‑CAM焦点指数和OOD AUROC上均优于其他方法，表现出更低的校准误差和更明显的错误预测区分。

**⚠️ 局限性**

局限在于仅使用MobileNet骨干、未验证对不同模型结构（如Vision Transformer）的普适性、缺乏对条件熵对应的aleatoric/epistemic不确定性进行深入分析。

---

## 223. EarthRecord: From Pixels to Land Change with Multimodal Agents over Four Decades of Satellite Imagery

**arXiv ID:** 2611.12934 | [PDF](https://arxiv.org/pdf/2611.12934v1)

**作者:** Haoran Zhang `[一作]` (AIEyes Technologies (Hong Kong) Co., Limited), Zhaohua Sun `[通讯]` (University of Hong Kong)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5137123013)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建EarthRecord基准，评估多模态智能体利用四十年原始多传感器卫星记录进行土地变化分析

**💡 创新点**

首次提供包含原始数据、闭卷与仪器对照两种控制的基准，衡量智能体从像素到真实变化的能力

**🔧 技术方法**

使用Python、Shell、地理空间与机器学习库、图像查看器和智能体工具链

**📊 数据集**

56个地理位置的原始Landsat、MODIS、VIIRS、Sentinel-2数据，附ERA5-Land等参考产品，共238道问题

**📈 对比分析**

对12个多模态模型进行三轮评测，最高pass^3≈85%，但在需要从影像映射或检测传感器漂移时表现显著下降

**⚠️ 局限性**

主要瓶颈在影像映射与传感器漂移检测，智能体往往先行放弃或误解像素变化，导致对真实地面变化判断不充分

---

## 224. How Post-Training Influences Verbalized Algorithmic Computation in Language Models

**arXiv ID:** 2611.13552 | [PDF](https://arxiv.org/pdf/2611.13552v1)

**作者:** Niall Dalton `[一作]` `[通讯]` (University College London), Niall Dalton (University College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统性地研究了在 Olmo‑3 3.7B-Instruct 模型的不同后训练阶段（SFT、DPO、Full）下，carry 信息的可解码性以及 Chain‑of‑Thought（CoT）书面推理的功能作用。通过线性探针、激活补丁、分布式对齐搜索、注意力抑制、语义聚类等技术，评估了内部状态对输出的因果影响，并与多位数加法和减法任务的性能进行对比。

**💡 创新点**

创新点在于：
1) 在同一模型家族的连续后训练检查点上进行纵向对比，首次揭示后训练如何改变内部信息的可用性与书面推理的重要性；
2) 结合线性探针与因果干预方法，区分“可解码”与“可控制”两类算法状态；
3) 采用注意力抑制与语义聚类，量化 CoT 中各语义单元对后续推理的影响，并在加法与减法两类任务中验证任务依赖性。

**🔧 技术方法**

主要技术包括：
- 线性探针（probe）用于提取携带信息；
- Probe‑guided steering、DAS 与激活补丁用于因果干预；
- 注意力抑制（attention suppression）与 Thought Anchors 方法评估单元重要性；
- TF‑IDF + K‑Means 语义聚类与手工标注，识别 CoT 语义单元；
- 教师强制（teacher forcing）和 CoT 交叉（crossover）实验。

**📊 数据集**

数据集：使用人工生成的 2、4、6 位数加法题目（平衡 carry 数量）以及对应的加法与减法子任务；每道题目配有两种提示——Answer‑Only 与 Free CoT；数据量以 383、312、291（不同阶段）等子集为例。

**📈 对比分析**

比较方法：
- 对齐不同后训练检查点的最终答案准确率，并统计使用 CoT 与不使用 CoT 的差异；
- 线性探针准确率随层、token 位置变化；
- 干预实验（DAS、补丁）成功率与答案正确率；
- 抑制实验中的重要性得分与“干扰率”（disruption rate）比较。
结果显示：Full 阶段在使用 CoT 时获得显著准确率提升，且其 CoT 在交叉实验中对所有模型都更有效；carry 信息在 SFT 与 Full 之间解码一致，但因果控制失败；语义重要性得分在 DPO 与 Full 阶段显著高于 SFT，尤其是 carry 相关语义单元。

**⚠️ 局限性**

局限性：
- 仅研究 Olmo‑3 家族（3.7B 参数）和加法/减法两类任务，结果可能不适用于其他模型或更复杂算法；
- 干预方法未能实现对 carry 的可靠控制，可能是干预策略、层级或模型尺寸限制导致；
- 研究基于后训练阶段的纵向对比，未能单独剖析训练目标与数据混合的贡献；
- 关注的推理长度和生成质量受 token 预算与模型终止行为限制；
- 仅使用合成算术数据，缺乏真实世界任务验证。

---

## 225. An Improved $(1+2e)$-Approximation Algorithm for Metric Data Placement

**arXiv ID:** 2611.13224 | [PDF](https://arxiv.org/pdf/2611.13224v1)

**作者:** S. Rasoul Etesami `[一作]` `[通讯]` (University of Illinois Urbana Champaign), S. Rasoul Etesami (University of Illinois Urbana Champaign)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种确定性多项式时间算法，在度量数据放置问题（支持异构需求、可变安装成本和整数站点容量）下实现了(1+2e)≈6.44的近似保证。

**💡 创新点**

创新点在于：①对每个资源采用多尺度贪心球划分；②将各资源层通过重叠球连接成单一流网络，仅需一次最小费用循环求解即可完成全局容量约束的圆整；③通过对球半径的几何序列优化，使得额外的几何误差控制在2e倍，取得最佳常数近似。

**🔧 技术方法**

关键技术包括：线性规划松弛并利用连续球半径表示访问成本；多尺度球聚类与层间交叉；构造全局流网络并引入代理流成本（surrogate cost）；利用流网络整性及最小费用循环求解；球半径的最优选择通过对指数序列的积分上界实现。

**📊 数据集**

该研究为理论工作，未使用实际数据集；所有分析均基于理论证明。

**📈 对比分析**

相较于以往的20.5、10、8、7.081等常数近似，本文实现了1+2e≈6.44的更优近似，成为自2001年提出该问题以来最佳的通用常数近似。

**⚠️ 局限性**

局限性包括：仍未突破6.44的常数；需要求解线性规划与大规模流网络，算法实现复杂；未考虑在线或分布式情形；对实际性能缺乏实验评估。

---

## 226. Continuous-Time Set-Transformers for Asynchronous WiFi-IMU Indoor Localization

**arXiv ID:** 2611.13057 | [PDF](https://arxiv.org/pdf/2611.13057v1)

**作者:** Mohamed Bachar `[一作]` (CESI), Yuehua Ding `[通讯]` (CESI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种连续时间Set-Transformer，用于异步WiFi与IMU数据的融合以实现室内定位。

**💡 创新点**

创新点在于使用时间编码的连续时间Token、单一自注意力块实现跨模态跨时间的无序集合融合，并通过模态与即时Dropout实现缺失/过时传感器的鲁棒性。

**🔧 技术方法**

采用Transformer自注意力、时间频率编码、WiFi-Net与IMU-CNN编码器、单个集成块与可学习位置查询。

**📊 数据集**

在Webots模拟、MSILN跨会话数据、IMUWiFine、UJIIndoorLoc和RoNIN等五个公开数据集上进行评估。

**📈 对比分析**

与k‑NN RSSI、ResNet1D、PDR、IMUWiFine等基线对比，跨会话测试MAE从28.31 m降至10.90 m，整体性能优于传统方法。

**⚠️ 局限性**

主要局限包括模拟WiFi场景过于理想、IMU编码器在跨主体时仍显劣势，以及测试集受单一路径样本比例偏高影响。

---

## 227. Robo-MLT: Dual-System Robotic Manipulation for Long-Horizon Medical Laboratory Tasks

**arXiv ID:** 2611.13039 | [PDF](https://arxiv.org/pdf/2611.13039v1)

**作者:** Zequan Zeng `[一作]` (Chongqing University), YunLong Zhang `[通讯]` (Chongqing University)

**通讯引用:** 2039 | [OpenAlex ID](https://openalex.org/A5100433494)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了一种双系统框架Robo‑MLT，用于医疗实验室的长周期自动化流程，结合显式工作流程监测与学习驱动的动作生成，并在血气分析与血样管分拣两项真实任务中实现完整流程执行与中断恢复。

**💡 创新点**

创新点包括：①低频多帧监测器配备完成/恢复两头，显式维护工作流程指针；②高频生成式执行器采用VLM条件流匹配策略；③异步运行时将两系统解耦，保持20 Hz控制速率；④利用冻结的VLM规划器生成顺序技能列表，并在恢复时重新规划剩余步骤。

**🔧 技术方法**

使用技术主要有：Qwen3.7‑Max VLM做规划/重规划；SigLIP2图像/文本编码器做多帧特征；Qwen3‑VL backbone+流匹配头做动作生成；Transformer结构与交叉注意力做多帧监测；异步线程框架；Python/ROS/C++实现。

**📊 数据集**

数据集：从专家操作员使用双臂机械臂和三目摄像头收集的1200条血气分析、1000条管分拣的完整轨迹数据；另外收集了操作员干预和失误的短段轨迹用于恢复头训练。

**📈 对比分析**

与ACT、SmolVLA、π₀、π₀.₅等单一策略基线以及VLM监测基线进行对比。Robo‑MLT在血气分析与管分拣任务中任务成功率分别为82%和70%，比π₀.₅高出30%和22%；在样本移除干扰下恢复成功率分别为90%和85%，显著优于VLM监测的65%/55%。执行时间更短、GPU占用约10 GB，控制延迟≈364 ms。

**⚠️ 局限性**

局限性包括：需要分别训练系统1和系统2，工程复杂；技能库可扩展性有限，新增操作需再演示并重新训练监测器；恢复头仅覆盖训练中出现的干扰类型，未验证对未见干扰的泛化；实验仅在同一工作台和两种任务上完成，未检验更广泛场景；对系统1的动作误差未由策略层捕获，需要进一步强化学习或知识隔离训练。

---

## 228. Quadratic Bendixson criterion for delay systems: a numerical implementation

**arXiv ID:** 2611.12972 | [PDF](https://arxiv.org/pdf/2611.12972v1)

**作者:** Mikhail Anikushin `[一作]` (St Petersburg University), Andrey Romanov `[通讯]` (St Petersburg University)

**通讯引用:** 30 | [OpenAlex ID](https://openalex.org/A5101698763)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ff43bff5-8ebd-437d-8b18-a060b245dafb` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种用于验证延迟系统中二次Bendixson准则的数值实现，开发了一个Python软件包来实现近似方案。

**💡 创新点**

创新点在于将频率不等式与延迟方程的传递算子表示为具有有界核的哈代-沃尔特拉型积分算子，从而构建出高效的数值方案。

**🔧 技术方法**

使用了数值分析和有限维截断技术，结合了哈代-沃尔特拉型积分算子的理论。

**📊 数据集**

使用了延迟方程的具体实例，包括Hopfield网络和Mackey-Glass模型，展示了软件包的应用。

**📈 对比分析**

通过与传统方法的比较，展示了所提方法在验证频率不等式方面的有效性，结果表明该方法在处理复杂系统时性能优越。

**⚠️ 局限性**

限制在于该方法主要针对具有单一延迟的线性部分的系统，可能不适用于更复杂的延迟结构或多重延迟情况。

---

## 229. Hand-Object Contact Transition Detection in Egocentric Video

**arXiv ID:** 2611.13486 | [PDF](https://arxiv.org/pdf/2611.13486v1)

**作者:** Huy Anh Nguyen `[一作]` (Adelaide University), Minh Hoai `[通讯]` (Adelaide University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种统一框架OASIS，用于精准检测第一人称视频中手与物体接触的起始（touch）和释放（release）瞬间，并提供了全新ContactTransition数据集；

**💡 创新点**

创新点在于同时考虑手与物体的联合表示并引入过渡感知的事件定位模块，使得touch与release不再互斥且可共享对象上下文；

**🔧 技术方法**

技术主要包括手中心化的HiCE特征增强、对象兴趣热图分支、对象出现门控、覆盖加权池化以及类感知位移回归；

**📊 数据集**

使用ContactTransition数据集，包含4021段第一人称视频，8498个touch标注、8077个release标注，并提供对象关注监督；

**📈 对比分析**

在精度严格的δ=0,1,2帧容忍度下，OASIS在无NMS条件下mAP为49.30，带NMS为51.19，均显著优于最强两阶段方法V-JEPA+MS-TCN（43.93/47.04）和最强端到端HiCE（43.67/47.58），在release检测上提升超过5个百分点；

**⚠️ 局限性**

局限性包括依赖手部检测框、对象热图需手部视野内可见且对遮挡/快速抬手场景鲁棒性有限，且模型未直接分配到具体手部或物体。

---

## 230. Embodied Science Agents: Making Embodied Experimental Design Learnable

**arXiv ID:** 2611.12958 | [PDF](https://arxiv.org/pdf/2611.12958v1)

**作者:** Yihang Che `[一作]`, Jun Wang `[通讯]` (University College London)

**通讯引用:** 38792 | [OpenAlex ID](https://openalex.org/A5010153821)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了嵌入式科学代理ESA，实现了数字认知与物理执行的闭环，在电化学废水处理实验中实现了从初始82%到99.7%的透光率提升。

**💡 创新点**

将认知学习、实验学习与具身学习三轴嵌入式学习整合为ESA框架，并设计了Embodied Unified Interface和可迁移的视觉-语言动作策略字典，实现了真正的具身实验者。

**🔧 技术方法**

融合了大型语言模型、贝叶斯优化、记忆检索、机器人视觉语言动作控制、双臂协作平台以及统一接口协议。

**📊 数据集**

使用Safranine T废水电-Fenton实验的二维参数空间（约2×10^8组合）进行自适应实验，未使用公开数据集。

**📈 对比分析**

与随机LHS初始化和贝叶斯优化单独运行相比，ESA在15轮实验中从82%提升至99.7%透光率，达到98.5%染料浓度降低，验证了在45小时内实现专家级配置的高效性。

**⚠️ 局限性**

缺乏多化学体系的泛化验证、对长时序记忆演化的定量评估不足、以及对不同硬件平台的可迁移性待进一步研究。

---

## 231. Multimodal Graph Learning: A Survey of Models, Benchmarks, and Frontiers

**arXiv ID:** 2611.12990 | [PDF](https://arxiv.org/pdf/2611.12990v1)

**作者:** Rui Xue `[一作]` (North Carolina State University), Tianfu Wu `[通讯]` (North Carolina State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对多模态图学习（MAG）领域进行系统综述，梳理了模型、基准、应用与前沿发展。

**💡 创新点**

创新点在于提出统一的消息传递框架、五层整合（特征、传播、拓扑、学习、基础模型）分类，映射跨模态敏感性；对近两年新基准进行细致比较；并将多模态图学习与AI代理与图工程关联，指明未来研究方向。

**🔧 技术方法**

综述涵盖的技术包括跨模态融合与耦合的图神经网络、图结构学习与自适应重构、跨模态对齐与对比学习、预训练的多模态图基础模型（如UniGraph2、PLANET）、LLM/MLLM与图序列接口，以及稳健性与联邦学习方法。

**📊 数据集**

使用的主要数据集来源于MM‑GRAPH（商品、书籍、知识图）、MAGB（亚马逊商品与Reddit社交）、GraphVLM（跨域节点分类）、OpenMAG（19个多模态图）以及MM‑OpenFGL（联邦多模态图）。

**📈 对比分析**

通过对上述基准的统一评测，作者比较了特征级、传播级、拓扑级、学习级与基础模型级方法的性能与效率；发现跨模态耦合与图结构学习在低同质性、稀疏边和多模态缺失场景下显著提升；而在高同质性、稠密边场景下基础模型往往能匹敌或超越传统GNN。

**⚠️ 局限性**

限制主要包括计算成本高（多模态编码与全图传播）、跨模态不一致导致噪声累积、缺失模态与冷启动难题，以及对异构分布和动态图的适应性不足；未来需要更高效的图预处理、鲁棒性理论与动态在线学习框架。

---

## 232. EgoHMP: Achieving Precise Human Motion Prediction in 3D Scenes via Egocentric Cues

**arXiv ID:** 2611.13654 | [PDF](https://arxiv.org/pdf/2611.13654v1)

**作者:** Xin Zhao `[一作]` (Chongqing University), Yu Liu `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一套名为 EgoHMP 的框架，用 egocentric 视频与头部运动来推断人类在 3D 场景中的交互意图，并通过视觉-运动对齐编码器、意图预测器以及基于接触的运动解码器来实现精准的未来运动预测。

**💡 创新点**

核心创新在于：①用 egocentric 视角与头部运动替代传统依赖瞳孔注视的意图推断；②引入模态感知调制机制，动态抑制无效交互线索；③利用自提示轨迹将全局轨迹与局部姿态分离；④多阶段去噪器结合接触建模，逐步纠正姿态伪影；⑤在解码器中加入 GCN 与 VPoser，使生成的运动既语义一致又物理可行。

**🔧 技术方法**

采用 DINOv2 视觉编码器、轻量 Transformer、PointNet++ 场景编码、视觉-运动对齐损失、模态调制标记、交叉注意力意图编码器、交互概率图、扩散模型自提示轨迹、接触概率多阶段去噪、AdaLN、GCN、VPoser、SMPL‑X 以及多种定制化损失函数。

**📊 数据集**

在 GIMO、GTA‑IM 以及 Ego‑Exo4D 三大公开数据集上进行训练与评测。

**📈 对比分析**

与多类基线（基于瞳孔的 BiFu、SIF3D、GAP3DS；非瞳孔的 Auxformer、MDP；基于 egocentric 的 EgoEgo、UniEgoMotion、EgoCast、EgoAllo）进行对比；EgoHMP 在 Traj‑Path、Traj‑End、MPJPE‑Path、MPJPE‑End、TDE@K 与 Penetration 等指标均实现了显著提升，成为目前最优方法，并在不同扰动下保持较高鲁棒性。

**⚠️ 局限性**

在视觉与头部运动信息都极具误导性的极端场景下仍可能出现意图错误；模型仅支持固定长度的预测序列，长序列时易截断或生成静止姿态；未来工作需引入长期因果推理与可变长度预测机制。

---

## 233. SPARK: Semantic-preserving Prompt Mutation for Code Generation using Consensus-based Verification Framework

**arXiv ID:** 2611.13401 | [PDF](https://arxiv.org/pdf/2611.13401v1)

**作者:** Alireza Daghigh `[一作]` (York University), Hung Viet Pham `[通讯]` (York University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过对问题描述进行语义保持的提示变异，结合共识验证框架，提升LLM代码生成的准确性。

**💡 创新点**

设计14种语义保持提示变异操作，系统化地生成多样化提示，并利用共识验证选择最优代码，显著提升生成质量。

**🔧 技术方法**

使用GPT‑4o进行提示变异；o4‑mini、DeepSeek R1 Basic、Gemini 2.5 Pro进行代码与测试生成；共识基验证（双重一致）与代码质量度量相结合。

**📊 数据集**

LiveCodeBench、BigCodeBench、DLBench。

**📈 对比分析**

与零样本、链式思维、CodeT基线对比，采用pass@k指标，平均提升至+4.1%（最高+5.2%），显著优于现有方法。

**⚠️ 局限性**

生成的测试可能不完全正确，提示变异需要人工验证；token消耗升高；覆盖范围受限，无法覆盖所有真实场景。

---

## 234. Reliable Computation of the $L_{\infty}$-Norm of Linear Time-Invariant Systems Depending on One Parameter

**arXiv ID:** 2611.13048 | [PDF](https://arxiv.org/pdf/2611.13048v1)

**作者:** Florent Corniquel `[一作]` (Sorbonne Université), Grace Younes `[通讯]` (Sorbonne University Abu Dhabi)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种计算线性时不变系统的L_∞-范数的新方法，该范数在鲁棒控制中具有重要意义。

**💡 创新点**

创新点在于通过计算解的通用参数化，避免了对参数空间的先前分解，从而降低了计算复杂度。

**🔧 技术方法**

使用了符号数值方法和多模计算，而不是传统的Gröbner基方法，确保了算法的认证性。

**📊 数据集**

使用了来自鲁棒控制文献的示例数据集，具体包括多个物理模型和控制系统。

**📈 对比分析**

与基于圆柱代数分解的方法相比，本文的方法在处理参数数量增加时的计算成本显著降低，性能表现更优。

**⚠️ 局限性**

限制在于该方法的有效性依赖于参数的通用性，对于非通用参数值的情况，可能需要额外的处理。

---

## 235. khammoon: From a Linguistic Guideline to a Thai Morpheme Segmenter

**arXiv ID:** 2611.13012 | [PDF](https://arxiv.org/pdf/2611.13012v1)

**作者:** Warit Sirichotedumrong `[一作]` (Typhoon, SCB DataX), Kunat Pipatanakul `[通讯]` (AI Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了轻量级泰语词形分词器，基于语言学指导原则进行手工标注与LLM自动标注，随后进行双轮校正与微调；

**💡 创新点**

首次提供了基于泰语词形（khammoon）的分词规范和手工评测集，并通过LLM标注扩充大规模语料，结合人类校正实现高精度的词形分词；

**🔧 技术方法**

采用大规模LLM（Qwen3-30B-A3B）进行标注，构建卷积BiLSTM字符级模型（约1M参数）实现CPU可部署的分词器；

**📊 数据集**

使用BEST2009和WangchanThaiInstruct两大语料，标注训练集约70M字符，开发集与测试集分别来自BEST2009与Wangchan，包含新闻、小说、法律、财经等多领域文本；

**📈 对比分析**

与词级分词器（newmm、DeepCut、AttaCut）以及词形基线（syllable、字典）对比，在Wangchan测试集上实现边界F1 0.992，明显优于词级分词器（最高0.925）且速度可达32k字符/秒；

**⚠️ 局限性**

限制包括：对非正式网络文本的鲁棒性未验证；在罕见专有名词、复杂借词与Pali–Sanskrit复合词上仍有误分；仅针对手工标注的正式文本，未覆盖大规模预训练模型对词形分词的影响。

---

## 236. APS-VLA: Action-Guided and Future-Aware Token Pruning for Efficient Vision-Language-Action Models

**arXiv ID:** 2611.13030 | [PDF](https://arxiv.org/pdf/2611.13030v1)

**作者:** Xiaoyang Xing `[一作]` (Tongji University), Asoke Nandi `[通讯]` (Brunel University of London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a4b10f5d-130b-4e77-9367-6469ec621899` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

学习一种轻量级视觉令牌选择器 APS‑VLA，在保持 VLA 基座冻结的前提下，利用机器人演示数据通过动作监督与视觉转移监督共同训练决策影响估计器（DIE），实现对多视角视觉令牌的高效裁剪并显著降低推理延迟。

**💡 创新点**

创新点在于：① 将动作一致性和动作条件视觉转移预测联合作为训练信号，使选择器既能保留即时控制信息又能捕获任务进程中的未来视觉变化；② 提出可微分的预算约束门控机制和单次硬裁剪方案，兼顾训练可微与部署效率；③ 在冻结的 VLA 背景下实现模块化插件化，保持原始策略的动作生成过程不变。

**🔧 技术方法**

使用的技术包括：预训练的 VLA 主干（OpenVLA‑OFT、π_0.5、XR‑1 等）、决策影响估计器（DIE）构建的多模态评分网络、基于 Laplace CDF 的可微裁剪门控、动作监督损失（匹配完整输入和演示动作）以及视觉转移监督（预测未来视觉特征）。

**📊 数据集**

实验数据集涵盖：LIBERO（四个任务套件），RoboCasa365（原子、组合、未见组合任务），以及两套真实机器人平台（YHRG THESEUS‑S1、双臂 UR5）和对应的演示录制。

**📈 对比分析**

与多种基准方法（FastV、VLA‑Cache、VLA‑ADP、VLA‑ACL、SAFE‑Pruner、SpecPrune‑VLA 等）进行比较。结果显示：在 LIBERO 上裁剪 75% 令牌时，推理延迟从 107.95 ms 降至 68.75 ms（1.57× 加速），任务成功率仅下降 1.25%；在 RoboCasa365 上实现 1.52× 加速，综合成功率提升；在真实任务中平均延迟从 137.25 ms 降至 93.25 ms，平均成功率从 60% 提升至 61%。

**⚠️ 局限性**

局限性包括：① 固定预算和硬裁剪可能导致在某些任务阶段丢失细粒度视觉信息；② 需要大量演示数据来学习选择器；③ 仅在冻结基座时工作，对动态自适应或跨任务迁移的鲁棒性尚待验证；④ 对不同摄像头的可变预算及跨时间视觉特征重用的研究仍未展开。

---

## 237. Camera3R: Feed-Forward 3D Reconstruction from Heterogeneous Camera Models

**arXiv ID:** 2611.13453 | [PDF](https://arxiv.org/pdf/2611.13453v1)

**作者:** Ruxiao Duan `[一作]` (Yale University), Alex Wong `[通讯]` (Yale University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出 Camera3R，一种低秩适配框架，用来将预训练的面向透视图像的前馈 3D 重建模型扩展到非透视（鱼眼、全景）以及混合摄像头序列，同时保持对透视图像的原有性能。

**💡 创新点**

创新点包括：①仅使用少量（约 0.5%）额外参数，冻结原模型并在各层插入针对不同摄像头的 LoRA 专家；②自适应容量分配，通过后验 SVD 截断自动为每个专家层分配合适的秩；③训练时使用摄像头路由器，在推理时无需摄像头元数据即可识别并选择合适的专家。

**🔧 技术方法**

技术手段主要是：低秩适配（LoRA）、专家路由（camera router）、自适应秩截断、跨摄像头数据增强、以及基于 Transformer 的 3D 重建网络（VGGT、MapAnything、π³）。

**📊 数据集**

实验使用多种数据集：透视 ADT、ETH3D；鱼眼 ASE、ScanNet++、KITTI360；全景 Matterport3D、Gibson、TartanAirV2、PanoCity、Stanford2D3DS、360Loc；并在混合摄像头环境中测试。

**📈 对比分析**

与基线方法（Wid3R、Fisheye3R、PanoVGGT、Argus、全景版 VGGT）以及 calibration tokens 进行对比。Camera3R 在所有评估指标（相机位姿、深度、点云精度、FoV）上均超过或匹配最佳基线，尤其在混合摄像头场景下显著提升，且对透视输入的性能保持不变。

**⚠️ 局限性**

局限性包括：仅支持预定义的摄像头类别（目前为透视、鱼眼、全景），需要额外训练来加入新类型；合成的畸变增强可能无法完全覆盖真实镜头和传感器的特定效应。

---

## 238. Designing Accessible and Rigorous ML Experiments: No-Code Visual Workflows and Structured Guidance for Beginners

**arXiv ID:** 2611.13386 | [PDF](https://arxiv.org/pdf/2611.13386v1)

**作者:** Henrik H. J. Detjen `[一作]` (Fraunhofer), Markus Wenzel `[通讯]` (Fraunhofer)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过两项内部对比实验，评估了面向初学者的两种机器学习实验支持工具：mlguide（结构化、基于证据的工作流设计指导）和AI Sandbox（可视化无代码实现环境）。

**💡 创新点**

创新点在于把机器学习工作流设计的概念层和实现层拆分开来，分别提供结构化指导与可视化实现，并系统探讨这两类支持对不同层次学习效果的影响；同时提供可复现的任务、数据集改动与测评问卷。

**🔧 技术方法**

使用了基于规则的证据库推荐引擎、拖拽式可视化工作流编辑器、NASA‑TLX 与 UEQ‑S 问卷评估工作负荷与用户体验、以及自定义的高低级工作流设计测验。

**📊 数据集**

使用并改造了 Kaggle 上公开的 Stroke Prediction 数据集，人工插入重复样本、冗余特征、缺失值与目标泄漏等典型数据科学缺陷。

**📈 对比分析**

通过前后对比实验（mlguide 之前后）和交叉设计实验（可视化无代码 VS Colab notebook）来比较工作流设计质量、方法论掌握程度、信心、工作负荷、用户体验与接受度。结果显示：mlguide 明显提升了高层次工作流结构分数但未显著改善细节方法论得分；可视化无代码环境在情感体验（hedonic）上优于笔记本，工作负荷与接受度基本相同。

**⚠️ 局限性**

局限性包括样本量仅18人且以技术背景为主，Study 1 的固定顺序可能导致学习效应；低层次方法论提升有限，表明仅提供推荐列表不足以培养深度方法论判断；实验时间短，未评估长期学习与实际项目迁移效果。

---

## 239. Measuring Definition Sensitivity in LLM Annotation of Social Science Concepts

**arXiv ID:** 2611.13339 | [PDF](https://arxiv.org/pdf/2611.13339v1)

**作者:** Bogdan Mamaev `[一作]` (Deakin University), Matteo Vergani `[通讯]` (Deakin University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究大型语言模型在社会科学文本注释中对概念定义的敏感性，衡量不同定义导致的标签差异。

**💡 创新点**

提出了“定义敏感性”和“改写底线”两个指标，系统评估模型是否真正按提示定义标注，并量化定义变更对注释结果的影响。

**🔧 技术方法**

使用Gemma、Qwen和Claude Sonnet三种LLM，在零温度或固定种子下进行批量注释，并采用区块自举法计算误差区间。

**📊 数据集**

样本包括六个概念：抗议、暗杀、恐怖主义、仇恨言论、气候持怀疑态度和假新闻；文本来源于ACLED、CCC、GDELT、MMPD、CNTS等公开数据集。

**📈 对比分析**

对同一文本在不同定义（及其改写）下的标注结果做成对比，计算不一致率（definition sensitivity与paraphrase floor）并求差距和比值；结果显示所有概念和模型中定义敏感性均显著高于改写底线，表明模型确实按定义标注。

**⚠️ 局限性**

局限性包括：样本对抗议、暗杀等概念的灰区案例偏重；仅使用英语和西班牙语文本；指标仅评估模型输出的差异，未验证标注准确性；Sonnet的重复实验受限于接口实现；结果与所选定义和模型版本紧密相关，泛化需进一步验证。

---

## 240. One-Sided Expert-Parallel All-to-All for Mixture-of-Experts Inference over NVLink

**arXiv ID:** 2611.13188 | [PDF](https://arxiv.org/pdf/2611.13188v1)

**作者:** Bo Li `[一作]` (NVIDIA), Chandler Zhou `[通讯]` (NVIDIA)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了一种基于单个 NVLink 域的一侧专家并行 All-to-All 通信，利用对称内存直接把 token 推送到目标 GPU 的 rank‑major 工作区，MoE 计算通过槽索引直接读取数据，省去传统的路由元数据交换。

**💡 创新点**

创新点包括：① 通过源侧自路由直接写入目标内存，消除预先交换计数或索引的元数据；② 采用 rank‑major 布局，减少字节传输、降低工作空间并实现按需加载；③ 提供两种同步协议（高吞吐与低延迟）以适配不同批量大小；④ 在单 NVLink 域内使用直接读写和系统级屏障实现低延迟同步。

**🔧 技术方法**

技术细节包括：CUDA 原子计数与系统范围内存屏障、对称内存映射、NVLink 直接读写、TMA 指令、Prims‑TS 分组 GEMM、FlashInfer 接口，以及多种量化格式（BF16、MXFP8、NVFP4 等）。

**📊 数据集**

实验数据集：DeepSeek‑V4.1‑Flash 与 Kimi‑K3 两种 Mixture‑of‑Experts 模型，token 数量从 1 到 4096，批量在 8 台 B300 GPU 以及 72 GPU NVL72 机架上进行测试。

**📈 对比分析**

与 V2.5、DeepGEMM MegaMoE 以及 +DeepGEMM 等基线对比，dispatch+combine 在所有批次中均比 V2.5 快 1.2–3.0×，在小批量时加速更显著；整体 MoE 层比 DeepGEMM 快 2.4×，并在 128 tokens 以下比 MegaMoE 更快；在大批量时 MegaMoE 由于融合内核可略占优。

**⚠️ 局限性**

局限性在于：EP 组必须完全落在单个 NVLink 域内；rank‑major 工作区随 EP 规模线性增长；实现依赖 NVLink 直接读写，不适用于跨域或 RDMA 场景。

---

## 241. Rethinking in Spikes: Mitigating Hallucinations in MDLMs with Step-Aware Decoding

**arXiv ID:** 2611.12901 | [PDF](https://arxiv.org/pdf/2611.12901v1)

**作者:** Zhongxing Xu `[一作]` (Monash University), Zongyuan Ge `[通讯]` (Monash University)

**通讯引用:** 16778 | [OpenAlex ID](https://openalex.org/A5005014252)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关的解码策略 StepRefine，通过检测多模态扩散语言模型（MDLM）迭代去噪过程中的熵尖峰来动态切换到隐层精细化，提升生成文本的可靠性并降低幻觉。

**💡 创新点**

创新点：①引入基于局部趋势的熵尖峰检测机制；②在检测到熵尖峰的步骤中使用概率加权嵌入进行多轮连续精细化；③利用 KL 散度自适应终止精细化；④实现了与现有加速方法（Fast‑dLLM、dLLM‑Cache、dKV‑Cache）的无缝兼容。

**🔧 技术方法**

技术手段包括：多模态扩散语言模型、熵尖峰检测（局部窗口统计）、概率加权嵌入（latent refinement）、自适应终止（KL 监控）、基于置信度的 remasking 策略，以及多轮前向传播。

**📊 数据集**

实验使用的多模态数据集包括：VStar、MMVP、MMEval‑Pro、VMCBench、RealWorldQA（通用推理与理解），MMHalu、Bingo、POPE（幻觉评估），MathVision、MathVista（数学推理），MMK12‑Phys、MMK12‑Chem（科学推理）。

**📈 对比分析**

与 CoTA、ReMix、WINO 等无训练解码方法以及 Fast‑dLLM、dLLM‑Cache、dKV‑Cache 等加速方法进行对比。结果显示 StepRefine 在四个 MDLM backbone 上平均提升 2–3% 的准确率，幻觉指标提升 2–4%，且在保持或略微提升吞吐量的同时维持原有加速方法的速度优势。

**⚠️ 局限性**

局限性：①熵尖峰检测依赖局部窗口统计，可能在噪声较大或极端小预算场景下误检或漏检；②每次精细化涉及额外的前向传播，虽然通过预算补偿控制总步骤，但在极低算力条件下仍会增加推理开销；③实验仅覆盖 MDLM，方法在非扩散或单模态模型上的效果尚未验证。

---

## 242. Market-Conditioned JEPA: Learning from Unpredictable Market Shocks

**arXiv ID:** 2611.13572 | [PDF](https://arxiv.org/pdf/2611.13572v1)

**作者:** Yuqi Wang `[一作]` (University of Hong Kong), Qi Liu `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了MC-JEPA，一种自监督的联合嵌入预测架构，在训练期间利用从其它资产未来嵌入构造的留一市场条件，学习能同时捕捉资产回报、风险与流动性等特征的单一资产表示；

**💡 创新点**

创新点在于将不可预测的市场冲击作为训练时的“特权信息”，通过留一市场条件和基于资产自身表示的因子加载器将市场噪声转化为监督信号，从而使市场暴露成为最优预测所必需，并且在推理时保持单资产无跨资产依赖；

**🔧 技术方法**

采用Transformer编码器的JEPA框架，构造因子结构化预测器（加载器），利用留一市场条件进行训练，并通过Ridge探针评估下游任务，此外还进行合成因子模型实验；

**📊 数据集**

使用2015–2025年中国A股一分钟级OHLCV数据，共计11年，训练集至2022年7月，评估集2023–2025年；

**📈 对比分析**

与Alpha158、时间序列基础模型（Kronos、TimesFM、Chronos、MOMENT）以及多种自监督目标（Mask‑Recon、Next‑Patch AR、TS2Vec、CPC、Per‑asset JEPA）进行比较，MC‑JEPA在九项下游指标（回报、市场beta、波动率、三种流动性指标）中取得最高RankIC，且在历史回测中获得最高超额收益、最高Sharpe比与最低回撤；

**⚠️ 局限性**

实验仅在中国A股单一频率下进行，未检验是否适用于其他市场、资产类别或频率；使用单一因子且标准化输入导致加载器捕捉的是相关性而非beta，若不作价格波动率校正则无法直接获得beta；

---

## 243. Making Strides in Symbolic Imitation Learning: Offline Demonstrations, Online Tuning

**arXiv ID:** 2611.13182 | [PDF](https://arxiv.org/pdf/2611.13182v1)

**作者:** Bryce MacInnis `[一作]` (Dalhousie University), Malcolm I. Heywood `[通讯]` (Dalhousie University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了符号模仿学习框架，先离线从示范数据提取线性运动原语并用进化算法构建决策列表，然后在线使用 ARS 对参数进行微调。

**💡 创新点**

创新点在于将离线演示与在线增量调优结合，利用决策列表和线性运动原语实现可解释且高效的策略，且在20个基准任务中不少任务超过了原始 DRL 预言者。

**🔧 技术方法**

技术包括期望最大化(EM)聚类提取原语、遗传编程(GP)演化决策列表、增量随机搜索(ARS)在线调参，以及深度强化学习或 CMA‑ES 作为预演示源。

**📊 数据集**

使用了20个常用 RL 基准（MuJoCo、Box2D、经典控制、Franka Kitchen、Adroit Hand）以及相应的离线演示数据集，演示者来自 DRL、CMA‑ES 或人工。

**📈 对比分析**

与多种进化和 DRL 对手（CMA‑ES、xNES、TQC、SAC、TD3、PPO 等）比较，实验显示离线+ARS 方案在11/20 任务中超过原始预言者，整体性能超过现有进化方法。

**⚠️ 局限性**

局限在于仍需依赖高质量演示数据、在线调参仍使用 ARS 可能不如其他优化器、离线阶段易受分布漂移影响、对复杂任务如 Ant、Humanoid 的性能仍未达到预言者。

---

## 244. Permutation-Based Pre-trained Model Merging via Reciprocal Orthogonality

**arXiv ID:** 2611.13016 | [PDF](https://arxiv.org/pdf/2611.13016v1)

**作者:** Akira Ito `[一作]` (Tohoku University), Atsutoshi Kumagai `[通讯]` (NTT Computer and Data Science Laboratories)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过搜索模型参数的置换，促进互相正交（RO），从而实现宽度较小的线性模式连通（LMC）和模型合并。

**💡 创新点**

首次将互相正交作为实现层级指数加权连通（LEWC）的关键条件，并用置换搜索（基于激活或权重）实现这一目标。

**🔧 技术方法**

利用二次分配问题（QAP）与Frank–Wolfe算法求解连续松弛，得到置换矩阵；随后进行温度校正的softmax、参数加法合并。

**📊 数据集**

在MNIST、Fashion‑MNIST、CIFAR‑10/100上，使用MLP、VGG‑11和ResNet‑20等网络进行实验。

**📈 对比分析**

与无置换的参数平均、权重匹配（WM）、MuDSC‑Align、ZipIt!等方法对比，所提置换方法在保持或接近集成（logit ensemble）精度的同时，仅需较小的宽度扩张，且在不同数据分布的模型合并中表现尤为突出。

**⚠️ 局限性**

仍需在更宽/更深网络或不同结构（如Transformer）上验证效果，激活基置换需要额外输入数据，且在某些极窄宽度下仍难以完全恢复原模型性能。

---

## 245. AURA: Auditory Spatial Intelligence via a Unified Representation from a General-Audio Encoder through Complete Alignment

**arXiv ID:** 2611.13687 | [PDF](https://arxiv.org/pdf/2611.13687v1)

**作者:** Boyan Han `[一作]` (Southeast University), Liang An `[通讯]` (Southeast University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `b88c6eac-d57a-4623-a604-1f401f3eb268` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究冻结的通用音频编解码器能否用于多源空间问答，提出AURA模型并在SpatialSoundQA上进行评估。

**💡 创新点**

通过层级分析揭示X-Codec的RVQ层对空间信息的分布，证明可通过六槽跨耳差异表示和场景监督恢复左/右定位与源绑定，展示仅用通用编码器即可匹敌或超越专用空间模型。

**🔧 技术方法**

使用X-Codec残差向量量化编码、六槽跨耳差异表示、场景与几何监督、冻结的Qwen3-8B LLM与LoRA训练等技术。

**📊 数据集**

使用SpatialSoundQA（基于Matterport3D BRIR的模拟场景）、MMAU、MMAR、AIR-Bench等通用音频评测数据集。

**📈 对比分析**

与BAT、DSpAST-BAT、OWL等专用空间系统以及Qwen2-Audio、Qwen2.5-Omni等通用模型对比；AURA在双源方向准确率提升10.31点、单源方向≥73%，距离误差最低，并在通用音频基准上优于专用系统。

**⚠️ 局限性**

仍需大量训练样本，未在真实录音或极端多源/噪声环境中充分验证泛化；对更复杂场景的适应性有待进一步研究。

---

## 246. Multimodal Road Closure Mapping via Cross-Modal Graph Attention

**arXiv ID:** 2611.13192 | [PDF](https://arxiv.org/pdf/2611.13192v1)

**作者:** Izzeddin Teeti `[一作]` (Amazon), Mohamed Moustafa `[通讯]` (Amazon)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出统一的多模态多任务框架，联合道路闭塞分类与路段关联。

**💡 创新点**

将街景图像、投影卫星图像与道路网络图进行跨模态注意力融合，并引入显式拓扑诱导的图神经网络，解决闭塞类型和路段定位的联合任务。

**🔧 技术方法**

使用ResNet、FPN、GATv2图神经网络、跨模态Transformer注意力以及边缘感知池化等技术，结合多任务交叉熵训练。

**📊 数据集**

基于19,380条真实物流车队采集样本，包含街景图像、投影卫星图像、道路图及闭塞类型与受影响路段标注，覆盖美国不同地区。

**📈 对比分析**

与VLM-A/B、最近邻基线及单任务版本比较，闭塞分类准确率达72.7%（比基线提升28%），路段关联准确率59.9%（比基线提升14%），模型参数仅23M，推理时间0.05s，显著小于VLM且速度更快。

**⚠️ 局限性**

仅适用于单闭塞场景，对完整闭塞召回低；在复杂网络或多闭塞情形下表现受限；缺乏时间维度闭塞持续时间预测。

---

## 247. AI Alignment Is Neither Solvable nor Sufficient

**arXiv ID:** 2611.13167 | [PDF](https://arxiv.org/pdf/2611.13167v1)

**作者:** Gerrit Großmann `[一作]` `[通讯]` (German Research Center for Artificial Intelligence), Gerrit Großmann (German Research Center for Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

探讨了AI与人类价值对齐的概念、可行性与局限，提出对齐不足以保障社会安全，并主张关注混合社会的涌现风险与韧性建设

**💡 创新点**

系统化评估对齐目标的不确定性、可验证性的技术边界以及对齐在社会层面的不足，首次将这些问题结合为一套完整的批判框架

**🔧 技术方法**

无具体算法或技术实现，主要采用理论分析与文献回顾

**📊 数据集**

无使用数据集

**📈 对比分析**

无对比实验或性能评估

**⚠️ 局限性**

缺乏针对具体实现的实证验证，无法给出可操作的技术解决方案

---

## 248. HiCAP: Hierarchical Cultural Adaptation via Personality for Dialogue Generation

**arXiv ID:** 2611.13207 | [PDF](https://arxiv.org/pdf/2611.13207v1)

**作者:** Lingzhi Shen `[一作]` (University of Southampton), Shoaib Jameel `[通讯]` (University of Southampton)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HiCAP 框架，将个性特征嵌入文化适应式对话生成，通过层级化控制实现语气与个性表达的分层调节。

**💡 创新点**

创新点在于：①把 Big‑Five 个性维度与文化信息联合编码；②设计层级化门控机制，先设定文化基调，再在每个 token 级别细化个性；③结合动态指令模板、个性推理与多代理演示，实现多维度的控制与示例化。

**🔧 技术方法**

采用 persona‑aware 有监督微调、动态指令构造、指令微调与多代理演示，并在 Transformer 变换器中实现基于文化与个性门控的层级化风格调节；使用多层 MLP、门控网络与检索式示例提取。

**📊 数据集**

使用 CulturalPersonas（跨国情景对话与人格标签）与 CPED（中文电视剧对话 + 人格/情感标签）进行训练与评估，基底模型为 Phi‑4‑14B（也对比 Qwen‑2.5‑14B、Llama‑3.1‑8B 等）。

**📈 对比分析**

与传统 Seq2Seq、Transformer、CDial‑GPT 以及多种大型 LLM（Qwen‑2.5、Llama‑3.3、Gemma‑2、DeepSeek‑V3.1、GPT‑4o、Kimi‑K2）在 BLEU、BERTScore、Perplexity、TTR、Distinct‑1/2 上对比。HiCAP 在 CPED 上全部指标最高，在 CulturalPersonas 上 BLEU、BERTScore、TTR、Distinct‑1 最高，整体效果显著优于 14B‑规模以上模型。

**⚠️ 局限性**

局限包括：①受训数据偏向高资源语言与英语，低资源、方言与子文化覆盖不足；②文化表示过于粗糙（仅按国家/语言），缺乏细粒度与动态更新；③个性推理稳定性不高，轻微提示变化会导致人格漂移；④可能放大训练数据中的刻板印象与偏见，需要更安全、更公平的控制机制。

---

## 249. Online-Adaptive Fuzzy Cognitive Maps for Neuro-symbolic Financial Forecasting

**arXiv ID:** 2611.13388 | [PDF](https://arxiv.org/pdf/2611.13388v1)

**作者:** Brandon Rodriguez `[一作]` (Florida International University), Olaoluwa Adigun `[通讯]` (Florida International University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了两套神经符号框架：框架 I 通过在线自适应模糊认知图（FCM）与 LSTM 的非对称加性残差融合实现金融预测；框架 II 通过文献检索与 LLM 验证构建可审计的因果图。

**💡 创新点**

创新点包括：① 在线 Hebbian 规则实现每日 FCM 边权自适应；② 非对称加性残差融合方法让 FCM 的方向性预测可动态调节 LSTM 输出；③ 每条边配备可验证的文献引用，提升模型可审计性；④ 双源新闻情绪与宏观指标实时融合，提升鲁棒性。

**🔧 技术方法**

使用技术：模糊认知图（FCM）、LSTM 循环网络、非对称加性残差融合、Hebbian 在线学习、新闻情绪分析（Loughran‑McDonald 词典）、大语言模型（GPT‑4）验证因果边、数据预处理（对数收益、滑动最小最大归一化）。

**📊 数据集**

数据集：NASDAQ 综合指数日频数据 2020‑2025（训练 2020‑2023，验证 2024，测试 2025）；宏观经济指标来自 FRED；新闻情绪来自 Yahoo News、Google News；框架 II 使用 2026 年 2–7 月的 119 天数据进行同日/滞后评估。

**📈 对比分析**

比较方法：与单独 LSTM 基线、单独 FCM 进行状态方向性和收益符号准确率比较。框架 I 的 ensemble 在 2025 测试集上实现 78.8% 的状态方向性准确率（与 LSTM 78.4% 相当）并在收益符号上提升 7.6pp（52.0% vs 44.4%），比纯 FCM 高 15.6pp。框架 II 同日预测准确率最高达 85.7%，随滞后时间递减。

**⚠️ 局限性**

局限性：① 仅在日频 NASDAQ 市场验证，未覆盖其他指数或高频交易；② 框架 II 依赖 LLM 生成的文献引用，若 LLM 出现幻觉需人工审核；③ 在线自适应需持续数据流，若出现 API 停机或数据缺失可能导致模型失效；④ 对于极端市场冲击或结构性断裂的快速反应尚未充分评估。

---

## 250. Agent-Guided Discovery of Exchange-Correlation Functionals with Physics-based Validation

**arXiv ID:** 2611.13329 | [PDF](https://arxiv.org/pdf/2611.13329v1)

**作者:** Achuth Chandrasekhar `[一作]` (Carnegie Mellon University), Amir Barati Farimani `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过前沿大型语言模型（LLM）代理和GPU评估器的协同工作，探索并发现了一种新型简洁的密度泛函 O159，并将其在自洽计算和外部基准上的性能进行了验证。

**💡 创新点**

创新点在于：①引入“同自旋相关窗口”在密度空间中对同自旋相关项进行平滑调节；②通过LLM代理实现符号式搜索，并结合物理屏蔽、训练拟合和自洽验证三阶段评估；③通过对窗口的refit控制，提炼出可解释的设计规则；④对比不同LLM骨干（Opus‑5.5 vs Codex）和传统符号进化（SyFES），展示LLM搜索在参数利用率和计算效率上的优势。

**🔧 技术方法**

技术手段包括：LLM代理在Model Context Protocol（MCP）框架下生成可执行符号表达式；GPU4PySCF 进行固定密度和自洽能量评估；PyTorch L‑BFGS 对参数进行拟合；四项物理屏蔽（自旋互换、均匀气体极限、尺度不变性、网格敏感度）和外部验证（MGCDB84、GSCDB137）。

**📊 数据集**

使用的数据集为 MGCDB84（包含 870 训练、2963 验证、1152 测试反应）和 36 个 GSCDB137 计算集（覆盖过渡金属、电场、振动等），其中 20 个为新集（未包含在 MGCDB84 训练/验证中）。

**📈 对比分析**

在与 GAS22、ωB97M‑V 和 SyFES 进化模型的比较中，O159 在自洽验证下的 WRMSD 下降 5.7%（训练）和 11.7%（测试），在 GSCDB137 上的平均 NER 与本地 ωB97M‑V（1.144）几乎持平（1.139），且仅使用约 40 倍更少的 token（Opus‑5.5）完成 SCF 门通过；相较于 SyFES，LLM 搜索在约 400 次拟合中实现了 0.10–0.13 的验证误差下降，且显著节约了计算资源。

**⚠️ 局限性**

局限性包括：①窗口功能在新 GSCDB137 子集上提升有限，主要显著在重原子总能量；②外部验证仍未覆盖更大分子或周期体；③窗口的物理解释仍基于经验控制，缺乏严格的理论根基；④在高梯度或多轨道区域窗口可能失效，需要进一步探索更通用的调制形式。

---

## 251. High Probability Risk Control for Online Policy Learning

**arXiv ID:** 2611.13163 | [PDF](https://arxiv.org/pdf/2611.13163v1)

**作者:** Yihong Guo `[一作]` (Johns Hopkins University), Anqi Liu `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于置换加权的风险评估器，并结合Sinkhorn算法高效近似置换边缘权重，构建在线策略学习的高概率风险控制框架；

**💡 创新点**

在存在反馈协变量偏移的在线决策中，实现了无额外方差、且在非单调风险函数下可得到渐近高概率风险控制的统一方法；

**🔧 技术方法**

置换权重近似（Sinkhorn迭代/矩阵尺度）、高斯乘子自举（Gaussian multiplier bootstrap）、自适应策略更新、对偶变分推断；

**📊 数据集**

在线情境赌博机（乳腺癌诊断、股市投资）、安全问答基准SafetyBench、医疗问答MedLFQA；

**📈 对比分析**

与O‑CRC、RollingRC、CPC、gCRC、LTT等基线比较；实验显示在乳腺癌诊断中快速达到阈值、在股市投资与SafetyBench中获得最低风险和最高置信度，且估计方差和偏差均低于重要性加权方法；

**⚠️ 局限性**

仅提供渐近高概率保证，需已知安全基准策略，假设候选策略集合有限且动作概率正则；置换权重近似可能在极大历史长度下导致计算成本升高，且在强烈策略漂移场景下近似误差需进一步验证。

---

## 252. Large-area Spatially Aligned Anchors

**arXiv ID:** 2611.12912 | [PDF](https://arxiv.org/pdf/2611.12912v1)

**作者:** Joni Vanherck `[一作]` (Hasselt University), Nick Michiels `[通讯]` (Hasselt University)

**通讯引用:** 197 | [OpenAlex ID](https://openalex.org/A5073702029)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种在XR头显内部使用空间锚点，同时利用外部漂移校正跟踪方法生成全局锚点云，在操作阶段通过点云配准实现大面积空间定位，从而在保持独立使用的前提下显著降低漂移。

**💡 创新点**

创新点在于将内部空间锚点与外部漂移校正跟踪（如标记或ORB‑SLAM3）结合，构建双锚点云并在每帧动态配准（可加权或不加权），实现从几米到十几米级范围内的精准对齐，同时解耦外部跟踪设备，支持任何支持空间锚点的XR头显。

**🔧 技术方法**

使用技术包括：空间锚点（各品牌SDK）、外部标记追踪（Aruco）、ORB‑SLAM3漂移校正、相机到头显的外参标定（控制器法或锚点法）、Levenberg‑Marquardt非线性优化、SVD点云配准（加权/不加权），以及OSC网络传输。

**📊 数据集**

数据集：实地测试环境（16.5 m×25 m与16 m×12 m），利用Leica RTC360激光扫描仪生成完整三维网格；外部标记布置在场景内多点；无公开公开数据集，全部为实验室采集数据。

**📈 对比分析**

评估方法为将内部头显相机姿态与外部标记/SLAM的姿态进行对比，计算角度误差（°）与距离误差（cm）的均值与标准差；实验表明加权配准在约3 cm左右的距离误差与1.2°左右的角度误差，比不加权和仅靠内部空间锚点时的误差低30–40%，且在仅使用两标记的情况下误差仍保持在≈5 cm。

**⚠️ 局限性**

局限性包括：整体精度受限于外部跟踪方法的精度；锚点布置与标记数量影响结果，少量标记会显著增加误差；需要外部摄像头或SLAM系统进行预处理，增加部署成本；锚点误放时会导致配准失效；对极端动态环境或大范围漂移的鲁棒性仍待进一步验证。

---

## 253. A computational framework for understanding embodied plans

**arXiv ID:** 2611.13669 | [PDF](https://arxiv.org/pdf/2611.13669v1)

**作者:** Aalap D. Shah `[一作]` (Yale University), Ilker Yildirim `[通讯]` (Yale University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为 GenEA 的多层次生成式代理模型，用以在自然、具身环境中推断他人计划与目标。

**💡 创新点**

创新点在于将粗粒度运动学、动作符号层与直觉物理动力学三层耦合进同一概率框架，实现了对长周期、复杂具身行为的生成式推断。

**🔧 技术方法**

采用了多层状态空间（Symbolic、Kinematic、Dynamics）与生成式规划器，结合任务规划、轨迹优化与游戏引擎式物理仿真，并使用概率分布推理。

**📊 数据集**

使用自制的块构建图像数据集（包含颜色/形状变体）和基于 Blender 的平台导航视频数据集进行实验验证。

**📈 对比分析**

通过与 TAMP、几何/稳定性启发式以及 ChatGPT VLM 的对比，GenEA 在目标与动作推断、手部使用偏好、导航难度评估等多项任务上与人类反应时间和偏好评分的相关性更高，显著优于替代模型。

**⚠️ 局限性**

局限在于依赖手工定义的符号动作与粗粒度物理模型，缺乏完整感知模块，且在更复杂多样的真实世界情境下的可扩展性和实时性尚待验证。

---

## 254. W-ISA: Transactional Execution of Weight-Space Intervention Programs

**arXiv ID:** 2611.13232 | [PDF](https://arxiv.org/pdf/2611.13232v1)

**作者:** Ekrem Aksoy `[一作]` `[通讯]`, Ekrem Aksoy

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套权重空间指令集及事务性运行时（W-ISA），实现了对预训练语言模型的可视化、可恢复的参数编辑与激活驱动干预。

**💡 创新点**

创新点在于：①把干预程序视为可序列化的指令集合，支持形状检查、精度控制和操作顺序；②在请求作用域内捕获并恢复参数快照，保证事务性和可重复性；③结合激活匹配规则实现条件干预；④提供与 PyTorch hooks 的精确对比。

**🔧 技术方法**

技术细节包括 FP64 CPU 计算、FP32/FP16 参数存储、CPU‑MPS 双设备同步、基于张量形状的指令编译、事务性快照与嵌套恢复、基于余弦相似度的激活匹配。

**📊 数据集**

使用 SmolLM‑135M‑Instruct、Qwen2.5‑1.5B‑Instruct 进行干预实验，并在 WikiText‑2 原始测试集上评估 perplexity。

**📈 对比分析**

对比方法：在 45 个定时请求中，W-ISA 与等价 PyTorch hook 在 32/128/512 词长度下产生完全相同的贪婪序列，最大 logit 差值 0.000184；WikiText‑2 perplexity 仅提升 0.375%；记忆占用峰值为 498 MiB，推理延迟相对基线略高。

**⚠️ 局限性**

局限性包括：仅支持单机、单设备；FP32/FP16 取舍导致微小数值误差；指令集有限，尚未覆盖所有可编辑变换；事务快照和恢复在高并发下可能引入锁争用。

---

## 255. Catch It Where It Happens: Real Time Error Correction During LLM Reasoning

**arXiv ID:** 2611.13107 | [PDF](https://arxiv.org/pdf/2611.13107v1)

**作者:** Rotem Elimelech `[一作]` (Technion Israel Institute of Technology), Ido Kaminer `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

做了在线、确定性验证，实时校验LLM生成的数学推理步骤并在发现错误时回滚并提供纠正提示。

**💡 创新点**

创新点在于利用计算机代数系统在CPU上即时验证自由文本中的等式，避免额外模型调用和GPU竞争，且对不同模型均可即插即用。

**🔧 技术方法**

采用正则表达式提取等式、数值采样验证、异步CPU检测、回滚与插入简短反馈的技术。

**📊 数据集**

在ASyMoB符号数学基准（2700题）上评估，使用三款开源权重模型：Qwen3-30B-A3B-Instruct-2507、Qwen3-4B-Instruct-2507、Gemma-4-12B-it。

**📈 对比分析**

与未加验证的普通推理相比，在线验证在平均 token 只增加 6‑7% 的情况下提升 1.4‑5.6 百分点准确率，尤其在被标记错误的样例中提升 6‑10.2 百分点。

**⚠️ 局限性**

限制包括仅能检测可解析为等式的局部错误，无法判断推理策略、前提正确性；数值采样可能漏判；提取与过滤规则导致覆盖率不足；仅在单一基准与三模型上验证。

---

## 256. DrugCoding: A Scientific Programming Agent for Drug Discovery

**arXiv ID:** 2611.13651 | [PDF](https://arxiv.org/pdf/2611.13651v1)

**作者:** Junwei Hu `[一作]` (XDeNovo Labs), Hengnan Zhang `[通讯]` (XDeNovo Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出DrugCoding，一种用于药物发现的科学编程代理系统，能够自动规划任务、路由适配模型、协同多智能体并执行代码。

**💡 创新点**

创新点在于将任务依赖图、模型自适应路由、代码生成与多智能体协同集成到统一框架，实现跨模型、跨工具的可靠执行与验证。

**🔧 技术方法**

技术包括大语言模型驱动的任务规划与执行、模型路由与绑定机制、工具调用协议（MCP）、多智能体协作架构以及验证与错误恢复模块。

**📊 数据集**

使用的主要数据集是由DrugPilot提供的Tool-Calling Dataset for Drug Discovery (TCDD)，并进一步抽样得到M-10%-TCDD的困难子集。

**📈 对比分析**

通过Pass@1指标与三种通用编码代理（Codex、Claude Code、Kimi）以及两种科研代理（Claude Science、Rosalind Workbench）比较，DrugCoding在完整多轮基准上平均Pass@1达97.35%，在困难子集上97.92%，分别比最强基线提升约9.24%与1.31%。

**⚠️ 局限性**

局限在于多智能体配置的收益因模型不同而异，且系统对外部计算资源与实时检索的依赖未完全解决，未来需进一步验证执行效果与科学结果的真实性。

---

## 257. BypassGym: A Benchmark for Evaluating AI Agents' Endpoint Defense Evasion Capabilities

**arXiv ID:** 2611.13684 | [PDF](https://arxiv.org/pdf/2611.13684v1)

**作者:** Weifeng Yuan `[一作]` (Huazhong University of Science and Technology), Haoyu Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一个评估AI代理在真实终端上规避已部署EDR的基准，测试其生成Shellcode加载器的能力。

**💡 创新点**

首次提出系统化的EDR规避基准，提供自动化反馈、可配置域知识技能以及多模型、多EDR、多payload的实验框架。

**🔧 技术方法**

结合Proxmox VE虚拟化、Windows 11 VM、六款EDR产品、LLM代理脚本工具（Codex CLI、Claude Code、OpenCode）、Shellcode生成框架（Havoc、AdaptixC2、Sliver）以及自定义知识技能。

**📊 数据集**

使用固定的三种C2 Shellcode payloads和六款EDR产品的配置，结合公开的Shellcode生成框架与自制的Evasion知识库。

**📈 对比分析**

通过多轮迭代（最多10次修复）记录成功率、迭代次数、调用次数、耗时与token；实验显示DeepSeek-V4.1 Flash最高成功率78.9%，Bitdefender最难，补充知识技能显著提升成功率。

**⚠️ 局限性**

仅覆盖Windows终端、六款EDR、有限的加载器知识库，未能完全区分加载器与payload检测，且无法公开完整EDR安装与Shellcode，未来需扩展至其他系统与更完整的知识图谱。

---

## 258. Living constitutions: Participatory alignment for sovereign foundation models

**arXiv ID:** 2611.13402 | [PDF](https://arxiv.org/pdf/2611.13402v1)

**作者:** Stine Lyngsø Beltoft `[一作]` (University of Southern Denmark), Lukas Galke Poech `[通讯]` (University of Southern Denmark)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种“活宪法”——一种持续迭代、参与式的 AI 规范制定流程，以保证 AI 模型在社会价值变化与多元群体争议下的对齐与合法性。

**💡 创新点**

核心创新在于将宪法设计为可携带争议的文档，采用“已定、争议、未决定”三种状态标记；并通过代理人模型验证该流程在面对多达 55%–80% 的恶意参与者时的鲁棒性。

**🔧 技术方法**

使用基于代理人的离散事件模拟（Agent‑Based Modeling）来评估委员会决策过程、投票机制与争议处理对最终规范质量的影响。

**📊 数据集**

未使用公开数据集；实验仅基于模拟的 1,000 名公民与 50 名委员会成员的人工生成投票与争议概率。

**📈 对比分析**

通过定义“有效对齐”指标（不含有害条款的已定/争议条款比例）与“通过率”指标进行评估。结果显示，在恶意比例高达 55%（捕获攻击）或 80%（稀释攻击）时系统仍能维持高对齐率，且在开发者席位占比 75% 以上时对齐效果近乎完好。

**⚠️ 局限性**

局限性包括：模型参数假设可能不符合现实；仅模拟投票与争议，未检验实际人类参与的可操作性；委员会容量固定导致队列拥堵；缺乏对实际数据集和多模态对齐技术的评估。

---

## 259. Parallel Broyden Methods for Efficiently Evaluating Nonlinear State Space Models

**arXiv ID:** 2611.13523 | [PDF](https://arxiv.org/pdf/2611.13523v1)

**作者:** Ian Christopher Tanoh `[一作]` (Stanford University), Scott W. Linderman `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一类 Parallel Broyden 量子-牛顿方法，用于在并行扫描（parallel scan）框架下对非线性状态空间模型的轨迹进行快速评估。

**💡 创新点**

创新点在于：① 用 Broyden 的分段更新构造对角加低秩（DPLR）雅可比近似，既保留了对角量子-牛顿的线性规模，又能捕获跨维度耦合；② 设计压缩近似扫描（compressed approximate scan）通过随机低秩压缩控制矩阵秩，保证对数级并行深度；③ 在理论上证明了压缩扫描下的有限步恢复与局部收敛性。

**🔧 技术方法**

核心技术包括：Broyden 分段和多 secant 更新、DPLR 表征、随机低秩压缩、并行扫描（Blelloch 树）、局部收敛分析（利用 LLE、压缩误差和块误差），以及对比实验实现。

**📊 数据集**

实验数据集：① 低秩 RNN（标准与 MLP 版本）用于验证 Jacobian 结构；② 八学校层级模型（Gibbs 采样）验证低秩耦合；③ 预训练 DiT 的 DDIM 扩散模型（D=4096）验证大规模 Jacobian 评估成本。

**📈 对比分析**

与基线方法（顺序、全 Newton、对角量子-牛顿、Picard、ParaTAA）比较：在 Jacobian 计算昂贵或高维时，Parallel Broyden/Accelerated Parallel Broyden 迭代次数显著减少、运行时间缩短（最高可达 30×），并在大规模问题上保持可扩展性；在 Jacobian 计算成本低时，对角量子-牛顿仍最快。

**⚠️ 局限性**

局限性：需要状态空间模型的耦合能够用低秩修正近似；当所需秩增大时压缩与扫描成本随秩增大，性能优势减弱；收敛分析为局部且保守，未给出全局收敛保证。

---

## 260. Inherited Weights: Measuring the Tokenization Premium and the Sovereignty Cost of Language Models for the Global South

**arXiv ID:** 2611.13441 | [PDF](https://arxiv.org/pdf/2611.13441v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 261. ZeroPoint: Lightweight Energy-Based Decision Model

**arXiv ID:** 2611.12992 | [PDF](https://arxiv.org/pdf/2611.12992v1)

**作者:** Javier Marín `[一作]` `[通讯]`, Javier Marín

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的能量基础模型ZeroPoint，该模型为每个输入的空答案分配零能量，从而使得决策过程能够更好地处理没有合适选项的情况。

**💡 创新点**

创新点在于将空答案作为模型的一个状态，并为其分配零能量，使得模型能够明确区分选项与空答案的兼容性。

**🔧 技术方法**

使用了小型多语言编码器作为能量函数，该编码器能够读取输入和所有选项，并且对选项的顺序不敏感。

**📊 数据集**

使用了多个数据集，包括CLINC150、Banking77、MASSIVE等，涵盖了多种语言和场景。

**📈 对比分析**

与其他决策系统（如Strands Decider 2B和Clef-flash）进行比较，ZeroPoint在正确决策率上略低，但在处理速度和模型大小上具有优势，且在没有合适选项时的表现有所改善。

**⚠️ 局限性**

限制在于模型的训练仅使用了一个种子，可能导致结果的可重复性问题；此外，模型在处理与任务无关的文本时的表现仍需进一步验证。

---

## 262. Distribution Shift Detection in Remote Sensing Imagery using Machine Learning Models

**arXiv ID:** 2611.13510 | [PDF](https://arxiv.org/pdf/2611.13510v1)

**作者:** Abdullah Almansour `[一作]`, Ozan K. Tonguz `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

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

## 263. ForeSpark: Joint Draft and Verification Budget Scheduling for Parallel Speculative Decoding

**arXiv ID:** 2611.13442 | [PDF](https://arxiv.org/pdf/2611.13442v1)

**作者:** Jaeyong Song `[一作]` (Meta), Shobhit Kanaujia `[通讯]` (Meta)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种工作保持（Work‑Conserving）推断框架WCS，联合为每个请求动态分配草稿（draft）长度和验证（verify）预算，利用上下文感知的预草稿估计器和轻量级联合调度器，最终显著提升LLM推理吞吐量。

**💡 创新点**

核心创新在于：①在草稿阶段前基于目标模型的上下文特征预测每个候选草稿长度的接受率；②利用凸包（concave envelope）将多请求多长度的收益转化为可排序的边际收益，实现全局最优且低成本的联合调度；③在草稿后通过后期估计器对草稿结果重新排序，智能分配验证预算，进一步减少无效计算。

**🔧 技术方法**

技术手段包括：上下文感知预草稿估计器（低秩投影+三角层输出）、轻量级联合调度器（凸包、全局排序、预算优化）、可变长度草稿执行（按计划裁剪掩码）、后期草稿估计器（基于路径选择的接受概率）、校准与温度缩放、以及在vLLM框架中实现的CUDA图重放与张量并行。

**📊 数据集**

主要数据集为ShareGPT多轮对话（20,000条），并在SPEED-Bench、数学、代码等多任务数据上进行泛化验证；对不同规模模型（14B~744B）进行实验，覆盖稠密、MoE和混合注意力架构。

**📈 对比分析**

与自回归解码、DSpark、Batchwise、EAGLE、AdaFlash等基线对比，WCS在匹配用户交互性时吞吐量提升最高达1.77×，对比自回归提升2.49×，在高负载下保持1.51×请求率；SLO满足率显著提高，能在更高请求率下仍满足TTFT/TPOT目标。

**⚠️ 局限性**

局限性包括：估计器对不同工作负载的泛化仍有余地，需进一步校准；全局排序可能导致部分请求未被草稿，影响公平性；当前方法主要针对并行草稿模型，向其他草稿形式（树形、递归）迁移需额外研究；并且在极低负载时，与最优全块草稿策略相当，提升有限。

---

## 264. FP64 Performance and the Limits of Ozaki Emulation on HPC Systems

**arXiv ID:** 2611.13332 | [PDF](https://arxiv.org/pdf/2611.13332v1)

**作者:** Katarzyna `[一作]` (Advanced Micro Devices), Jonathan Belof `[通讯]` (Advanced Micro Devices, Inc.)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

评估 FP64 在 HPC 计算中的重要性，系统性分析多种典型工作负载的算术强度和 FP64 需求；进一步研究 Ozaki 方法在低精度 Tensor Core 上对 GEMM 的准确性、性能和内存占用。

**💡 创新点**

①揭示仅有不到 10% 的 HPC 工作负载可通过 Ozaki GEMM 模拟获得显著加速；②量化 Ozaki 仿真对工作空间、数值稳定性和矩阵尺寸的敏感性；③证明在 AI 专用加速器上降低 FP64 吞吐会使原本带宽受限的内核转为算术受限。

**🔧 技术方法**

采用 Roofline 采样、FP64 与低精度 GEMM 对比、Ozaki-I 与 Ozaki-II 两种分解方案、FP8/INT8 Tensor Core 以及 ROCm/CUDA BLAS 库。

**📊 数据集**

47 种不同应用/基准的 131 个 kernel，覆盖 11 类 HPC 代理应用；矩阵尺寸从 384×384 到 40960×40960，条件数从 10⁰ 到 10¹⁷。

**📈 对比分析**

通过三台 GPU（AMD MI355X、NVIDIA B200、GB300）测量原生 FP64 与 Ozaki 仿真 GEMM 的吞吐、工作空间和加速比；实验显示在 4096×4096 及以上尺寸时，INT8 Ozaki-II 可实现 1–3× 的加速，但需要 2–3 倍 HBM；FP8 速度更高但内存占用更大。

**⚠️ 局限性**

Ozaki 仿真受限于矩阵尺寸、数值精度、工作空间与集成开销，无法覆盖大多数 HPC 负载；即使在适用场景中，整体加速仍受限于非 GEMM 代码路径，故 FP64 仍是关键硬件需求。

---

## 265. CP-MLPs: A Tensor-Rank Theory of Tied and Untied MLP Blocks

**arXiv ID:** 2611.13647 | [PDF](https://arxiv.org/pdf/2611.13647v1)

**作者:** Md Rifat Arefin `[一作]` (Mila Quebec AI Institute), Guillaume Rabusseau `[通讯]` (Mila Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的CP-MLP框架，用于比较绑定和未绑定的乘法前馈块，分析了不同设计选择对表达能力的影响。

**💡 创新点**

创新点在于引入了CP-MLP框架，明确区分了门控机制与检测器和数值方向的绑定选择，揭示了未绑定原子在某些情况下的宽度效率更高。

**🔧 技术方法**

使用了张量分解的CP（Canonical Polyadic）框架，结合了多种激活函数（如ReLU^2、ReGLU和SwiGLU）进行分析。

**📊 数据集**

实验中使用了合成的多项式目标，特别是针对不定二次目标和角色不对称的单项式进行测试。

**📈 对比分析**

通过与绑定和未绑定的MLP进行比较，发现未绑定块在特定输入投影预算下保持优势，且在宽度和损失方面的表现符合理论预期。

**⚠️ 局限性**

限制在于该研究主要集中在合成目标上，尚未在完整的Transformer语言模型中验证这些效果，未来需要扩展理论以涵盖更复杂的多腿原子。

---

## 266. Neural Network-Assisted CLEAN for Efficient Multipath Parameter Estimation in MIMO Systems

**arXiv ID:** 2611.13298 | [PDF](https://arxiv.org/pdf/2611.13298v1)

**作者:** Chaofan Deng `[一作]` (Georgia Institute of Technology), Arijit Raychowdhury `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种神经网络辅助的 CLEAN（NN‑CLEAN）算法，利用多头分类器替代传统多维参数搜索，实现 MIMO 系统的高分辨率多径参数估计；

**💡 创新点**

创新点在于将单一 MPC 的多头分类网络在 CLEAN 迭代中多次复用，支持可变提取深度，显著降低搜索复杂度，并通过标签平滑、角度门控与距离损失加权提升估计精度；

**🔧 技术方法**

使用的技术包括深度卷积+残差网络、多头分类、标签平滑（GLS）、角度条件门控、距离损失加权及三阶段（主导‑教师‑自回归）训练；

**📊 数据集**

使用的数据集包含：1）合成离散网格的多径信道（不同路径数、SNR、天线与子载波配置），以及 2）NVIDIA Sionna Ray‑Tracing 场景（街区峡谷，LOS/NLOS 两种情况）；

**📈 对比分析**

在与网格搜索 CLEAN（GS‑CLEAN）、3D‑MUSIC、3D‑ESPRIT、一次性 NN 的对比中，NN‑CLEAN 在单/双路径下准确率与 GS‑CLEAN 相当或略优，且在批处理 8 条通道时实现约 27× 的运行速度提升和 52 倍的 GPU 内存节省；然而在 3+ 路径和 NLOS 场景中准确率下降，BER 与 GS‑CLEAN 仍存在小幅差距；

**⚠️ 局限性**

局限性包括：1）对未知路径数的自动停止阈值未给出；2）受离散网格量化误差影响，在高路径数和复杂环境下性能衰减；3）对训练分布与实际环境（如 Sionna 语义差异）的迁移能力有限。

---

## 267. AgenticConflictLens: From Long Traffic Videos to Verifiable Conflict Evidence

**arXiv ID:** 2611.13540 | [PDF](https://arxiv.org/pdf/2611.13540v1)

**作者:** Boshra Khalili `[一作]` (Columbia University), Andrew W Smyth `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于查询的框架AgenticConflictLens，用于从长时段固定摄像头交通视频中高效、可审计地提取参与者级冲突。

**💡 创新点**

创新点在于结合了Traffic-Aware Controller、结构化交通记忆和专用冲突分析工具，实现查询驱动的自适应粗细搜索与可解释的验证流程，并仅检查视频少量比例即可达到高精度。

**🔧 技术方法**

采用了视觉-语言模型（Qwen2.5-VL-7B-Instruct）、检测与跟踪（如ByteTrack）、轨迹、语义场景先验、粗细冲突扫描器与细化验证器，并使用记忆模块实现状态化搜索。

**📊 数据集**

使用了由Bellevue和NYC DOT公开/受限摄像头构成的96小时交通视频数据集，并为每小时生成冲突标注与自然语言查询。

**📈 对比分析**

与五个改编的长视频推理与交通安全基线相比，AgenticConflictLens在Bellevue和NYC DOT上分别取得0.84/0.80的冲突存在F1、0.77/0.73的tIoU，并仅检查约9%/12%的视频，显示出更优的准确率与效率。

**⚠️ 局限性**

局限性在于对检测、跟踪和场景先验的依赖，无法处理严重遮挡、光照不足或轨迹丢失导致不可见冲突的情况。

---

## 268. Differentiable Filtering for Inertial Odometry

**arXiv ID:** 2611.13517 | [PDF](https://arxiv.org/pdf/2611.13517v1)

**作者:** Yaru Gu `[一作]` (Memorial University of Newfoundland), Ting Zou `[通讯]` (Memorial University of Newfoundland)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了可微分的滤波框架DFIO，用学习得到的三维位移测量及其协方差与误差状态卡尔曼滤波器结合，实现IMU-only惯导位姿估计。

**💡 创新点**

创新点在于：①将测量模型（IncNet）与滤波器耦合在计算图中，实现对滤波器输出的梯度回传；②采用两阶段训练，先独立预训练位移与协方差，再通过后验位姿监督联合微调，提升测量质量与权重；③在本土化重力对齐坐标系中输出位移与对角协方差，便于网络学习。

**🔧 技术方法**

技术手段包括：时序残差网络+自注意力（IncNet）生成位移/协方差；误差状态卡尔曼滤波器（ESKF）做IMU传播与测量更新；两阶段自监督训练，使用后验位置的负对数似然与NEES正则；PyTorch实现、GPU训练。

**📊 数据集**

数据集：TLIO（205条步态轨迹，Bosch BMI055校准IMU）和OxIOD子集（8条手机IMU步行轨迹，未校准噪声大），分别用于公开数据评测与噪声鲁棒性验证。

**📈 对比分析**

与RoNIN‑IncNet（开放式积分）和TLIO（先训练测量模型后滤波）比较。DFIO在TLIO测试集上平均绝对平移误差从1.957 m降至1.699 m（-13.2%），相对平移误差从0.140 m降至0.084 m（-39.6%）；在OxIOD上能明显抑制平移漂移，轨迹与标定更贴近。整体性能优于两种基线，且在单头（只调位移或只调协方差）与双头联合调优之间展示了不同的精度与漂移平衡。

**⚠️ 局限性**

局限性：仅在步态/行人数据上验证，未评估其它机器人或高速运动；仅使用IMU + 简单卡尔曼滤波器，未考虑外部观测（视觉/GNSS）；对极端噪声或极端偏置仍有挑战；训练需要大量标定数据，未讨论在线自适应。

---

## 269. AdjointFlow: Accelerating Scientific Inverse Problems with Physics-Based Transport

**arXiv ID:** 2611.13389 | [PDF](https://arxiv.org/pdf/2611.13389v1)

**作者:** Ethan Hanold `[一作]` (University of Wisconsin--Madison), Leonardo Zepeda-Núñez `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AdjointFlow 框架，通过在随机插值（Stochastic Interpolants）中注入物理信息，实现对科学逆问题的高效重构。

**💡 创新点**

创新点包括：① 基于逆向算子（adjoint）构造数据相关初始化，显著缩短目标分布的传输距离；② 设计无噪声（noise‑free）速度场，保持在目标分布高度集中时的空间正则性；③ 引入目标预测（target‑prediction）参数化，将物理初始化嵌入学习的速度中；④ 允许在推理时通过 RED‑Diff 等方法无额外训练地进行物理一致性微调。

**🔧 技术方法**

技术手段包括：随机插值（Flow Matching / Diffusion 的连续形式）、逆向算子初始化、速度分解与噪声桥（bridge）设计、目标预测网络、前向欧拉积分，以及可选的推理时间指导（inference‑time refinement）。

**📊 数据集**

使用多种科学逆问题基准数据集：非线性频域散射（Shepp–Logan、3–5–10h 三角、10h 重叠方块、脑扫描）、线性频域散射（HL60 细胞）、时域地震成像（CurveFault）、黑洞成像（EHT 模拟数据）。

**📈 对比分析**

与传统全波形反演（FWI）、确定性机器学习模型（SwitchNet、WideBNet、EquiNet）、条件扩散模型（CDM）以及插件式扩散优先方法（PnP‑DP）对比，AdjointFlow 在重构误差上提升约 70%，在推理速度上比 PnP‑DP 快 4–5 个数量级；在可选的推理时间微调后，其数据一致性可与甚至超过 PnP‑DP，且不需要重新训练。

**⚠️ 局限性**

局限性：性能高度依赖逆向算子或线性化的准确性；在逆向算子信息不足的案例（如 3–5–10h 三角）时优势不明显；需要对噪声桥、积分步数等超参数进行调优；并非所有逆问题都能直接获得可用的逆向算子或精确线性化。

---

## 270. Sharp Sample Complexity of KL-Regularized Reinforcement Learning

**arXiv ID:** 2611.13480 | [PDF](https://arxiv.org/pdf/2611.13480v1)

**作者:** Andrew Liu `[一作]` `[通讯]` (Columbia University), Andrew Liu (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了 KL 正则化 MDP 的软最优策略，并给出了生成模型下的样本复杂度上界与下界，证明其复杂度为 Θ(KH³/(ε·max{ε,β}))。

**💡 创新点**

提出了两阶段的采样复杂度与精确常数，揭示温度 β 对样本需求的影响，并证明在 H³ 依赖下实现 1/ε 速率，优于之前的 H⁴/H⁵ 结果。

**🔧 技术方法**

利用控制为推断的贝叶斯框架、马尔可夫不等式、极限定理、Fisher 信息权重、自适应采样分配（Neyman 样本分配）以及 Assouad/Le Cam 下界构造完成分析。

**📊 数据集**

在合成 MDP（两臂工具、jackpot MDP）、Garnet 随机 MDP 以及 100 状态环上进行仿真验证。

**📈 对比分析**

与先前的 KH⁵/(β ε)、KH³/ε² 等上界以及 bandit 的两段式下界对比，取得更优的 H³ 取向；仿真显示常数与理论一致，且自适应分配可将风险降低约 1.5–2.5 倍。

**⚠️ 局限性**

仅适用于离散表格 MDP 与已知奖励的生成模型，忽略探索与函数逼近；β 在 H 范围内的中间区间仍未完全收敛；常数与对数因子可能不够紧凑。

---

## 271. Post-OCR Correction of Mixed-Script Uzbek Text: Byte-Level ByT5 with an Empirical Homoglyph Noise Matrix

**arXiv ID:** 2611.13609 | [PDF](https://arxiv.org/pdf/2611.13609v1)

**作者:** Mannon Ochilov `[一作]` (Tashkent University of Information Technologies named after Muhammad al-Khwarizmi), Bahodir Zaripov `[通讯]` (Tashkent State University of Economics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在保密环境下针对混合脚本乌兹别克语的 OCR 后纠错问题，提出了一种基于字节级 ByT5 的模型，并结合经验混合字符噪声矩阵与保守解码规则进行训练与推理。

**💡 创新点**

创新点包括：① 通过手工标注的 359 条同形异字错误事件构建经验噪声矩阵，直接捕捉真实 OCR 错误分布；② 将噪声矩阵与域适配和保守解码结合，显著降低误更率并控制新误差；③ 在单机、单 GPU 环境下完成整个流水线，满足严格的机密性要求。

**🔧 技术方法**

技术手段包括：字节级 ByT5 模型（基于 T5 的 UTF-8 字节表示）、经验噪声生成器（基于 Unicode 同形字符表并重新加权）、保守解码（Levenshtein 距离阈值 3 防止过度纠正）、数据增强（合成多噪声样本）、Bf16 混合精度训练。

**📊 数据集**

数据集为 487,205 行原始报表文本（经过清洗后 422,815 行），从中生成 2,283,198 条合成训练对，63,420 条验证对，63,426 条测试对；另外收集了 5,204 条真实扫描的 Gold 集，并手工标注 359 条同形错误。

**📈 对比分析**

在 Gold 集上与原始 OCR、基于规则的校正、以及 Qwen2.5-7B LLM 进行对比。改进的 ByT5 系统将字符错误率从 0.0578 降至 0.0369（误差率下降 36.2%），同形错误恢复率提升至 91.3%，且仅新增 84 个错误，满足设定的 15% 错误减少与 100 新错误以内的双重阈值。

**⚠️ 局限性**

局限性包括：① 仅针对单一组织的工程-地质与信息技术报告，未覆盖法律、文学等其他文本类型；② 受保密约束导致未能使用更大规模模型或更广泛的数据；③ 经验噪声矩阵与域适配的效果未通过完整 2×2 ablation 分离验证；④ 对混合脚本中复杂词形和多字节字符的处理仍存在误差。

---

## 272. Rhythmic-Semantic Co-Speech Gestures for Emotional Dialogue in Robots

**arXiv ID:** 2611.13472 | [PDF](https://arxiv.org/pdf/2611.13472v1)

**作者:** Christian Arzate Cruz `[一作]` (Honda Research Institute Japan), Randy Gomez `[通讯]` (Honda Research Institute Japan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套基于文本和情感的实时节奏-语义共说手势生成系统，可在社交机器人 Haru 上实时产生情绪一致的标志性手势与节拍手势。

**💡 创新点**

首次在同一系统中同时建模标志性手势和节拍手势，并且仅使用文本与情感预测手势时序与强度，实现低延迟生成。

**🔧 技术方法**

使用 BERT/Transformer 进行句子级情感与手势激活与强度回归，RoBERTa 预测节拍强度，并结合轻量化 Transformer (TRS) 进行推理。

**📊 数据集**

在 BEAT2 数据集上训练并评估，包括文本、音频、动作与情感标签。

**📈 对比分析**

与 LLM 触发、真实标签和随机基线对比，在 60 位用户的在线实验中，该方法在手势与语音同步度最高、获得最多用户偏好，表现优于基线。

**⚠️ 局限性**

仅基于文本忽略真实语调，节拍手势采用离散强度，手势库有限，未在现场交互测试，且各子模型独立导致误差累积。

---

## 273. Train High, Ship Low: Efficient LoRA Synchronization for Distributed On-Policy Post-Training

**arXiv ID:** 2611.13306 | [PDF](https://arxiv.org/pdf/2611.13306v1)

**作者:** Shenghui Li `[一作]` (Uppsala University), Thiemo Voigt `[通讯]` (Uppsala University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种分布式 On‑Policy 后期训练的 LoRA 同步框架 HiLo，该框架用压缩梯度替代参数同步，并在工作节点上本地重放优化器步骤，以实现低通信量且保持同版一致。

**💡 创新点**

创新点包括：① 用梯度重放保证同版一致，解耦训练秩与网络秩；② 设计了预算感知的谱截断压缩器 Spar，结合 Gram 矩阵谱分配、误差反馈和自适应秩分配；③ 通过梯度而非参数进行压缩，减少了因优化器导致的全秩扩展问题。

**🔧 技术方法**

主要技术手段包括：LoRA 参数化、梯度压缩（Spectral Packing、PowerSGD、Top‑k）、误差反馈、分布式 Actor‑Learner 架构、AdamW/ Muon/ Lion/ SGD‑Momentum 等优化器的本地重放、BF16/FP32 数据类型处理。

**📊 数据集**

实验数据集来自 Reasoning Gym，涵盖 Knights and Knaves (OPD) 与 Countdown‑4 (RLAD) 两个推理任务，使用 Qwen3‑1.7B‑Base 与 Qwen3‑0.6B‑Base 模型及其 rank‑64 LoRA 适配器。

**📈 对比分析**

与全同步、低秩 LoRA、Delta、PowerSGD、Top‑k 等基线对比。HiLo+Spar 在 1.6% 的传输量下，保持 99% 以上的 pass@4（与全同步相差 ≤1.6 点），并在 62.5/125 Mbps 链路上实现 2.2× 的更新加速；在同等字节预算下比 PowerSGD/Top‑k 取得更低的通信量和更高的学习质量。

**⚠️ 局限性**

局限性：① 只在 OPD 与 RLAD 任务上验证，缺乏 GRPO 等其他 On‑Policy 环境的评估；② 对硬件异构性、数值精度、消息丢失和重启恢复等鲁棒性未做充分实验；③ 误差反馈与谱截断对学习质量的影响依赖于具体任务与优化器，需进一步理论与实证研究。

---

## 274. Network-Co-Designed Dynamic Sparsification for Dependable Distributed Robot Learning over 3GPP Networks

**arXiv ID:** 2611.13282 | [PDF](https://arxiv.org/pdf/2611.13282v1)

**作者:** Géza Szabó `[一作]` (Ericsson Hungary Limited), Boldizsár Bunda `[通讯]` (Ericsson Hungary Limited)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fede83ac-7505-405f-ab37-e7284695c47f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一种网络共设计的动态稀疏化机制，将模型更新压缩与网络字节预算与时限绑定，实现分布式机器人学习的通信效率提升；

**💡 创新点**

创新点在于将通信压缩与网络资源监测/控制进行闭环耦合，并通过3GPP SA6和VASEAL框架提供AI专属的网络暴露与控制接口；

**🔧 技术方法**

使用了异步联邦强化学习（Flower框架+PPO）、增量稀疏编码（按阈值选取参数）、Linux traffic control模拟网络拥塞、Genesis机器人仿真与3GPP NEF/PCF/NWDAF API扩展；

**📊 数据集**

使用的是在Genesis仿真环境中生成的四种异构四足机器人在不同地形上的训练数据（无公开数据集）；

**📈 对比分析**

通过对比不同最小保留比例（100%、75%、50%、25%、5%）下的平均奖励曲线和学习速率，发现即使压缩至5%参数，学习进度几乎不受影响，通信量可降低约95%；

**⚠️ 局限性**

局限性在于实验仅在仿真网络与机器人环境中验证，未涉及真实3GPP网络和更复杂AI任务，且结果可能无法直接推广到多种硬件/场景下。

---

## 275. A Concept-Based Interpretable Framework for Clinical Text-based Prediction via Token-Concept Attention without Concept Supervision

**arXiv ID:** 2611.13564 | [PDF](https://arxiv.org/pdf/2611.13564v1)

**作者:** Jifan Gao `[一作]` (Dana-Farber Cancer Institute), Guanhua Chen `[通讯]` (University of Wisconsin-Madison)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一种无需概念标签的概念瓶颈解释框架，利用跨注意力瓶颈在临床文本中自动推断ICD‑10概念并通过线性层实现可解释预测。

**💡 创新点**

创新点在于不依赖人工概念标注，采用跨注意力匹配文本与概念描述来估计概念相关性，并以线性可解释模型做预测。

**🔧 技术方法**

技术包括预训练语言模型（如SapBERT、ClinicalBERT、bge‑large）编码文本和概念、token–concept跨注意力瓶颈、线性概念权重、温度化归纳及L1正则化。

**📊 数据集**

使用了两套临床数据：美国UW Health系统的胸部创伤严重程度数据（7,693/1,983例）和公开的MIMIC‑III ICU住院死亡预测数据（36,803/9,291例）。

**📈 对比分析**

与原始黑盒模型以及传统概念瓶颈模型对比，本文框架在AUROC、AUPR等指标上与黑盒差距≤0.02，显著优于标准概念瓶颈模型。

**⚠️ 局限性**

局限性包括仅评估二分类任务、截断文本到512令牌、未覆盖更长文本和生成任务，且对不同后端编码器的解释一致性仍有差异。

---

## 276. What drives the emergence of symbolic mechanisms in large language models?

**arXiv ID:** 2611.13270 | [PDF](https://arxiv.org/pdf/2611.13270v1)

**作者:** Melody Zixuan Li `[一作]` (Princeton University), Taylor Webb `[通讯]` (Princeton University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

训练从零开始的神经网络完成抽象序列推理任务，并系统地改变数据多样性与模型结构，研究符号机制的出现

**💡 创新点**

证明数据多样性是驱动符号机制出现的主要因素，而Transformer的特定结构或注意力机制并非必需

**🔧 技术方法**

使用因果中介分析、表示相似性分析、概念干预以及对抗训练等机制可解释技术来识别和验证符号抽象头与子空间旋转

**📊 数据集**

构造的自定义序列数据集，涵盖四个维度的多样性：样本数、词表大小、规则数与角色-词表偏置，另外还设计了实体-属性绑定任务

**📈 对比分析**

与标准Transformer、三种结构变体（去掉位置权重共享、共享Q/K/V、MLP注意力）以及无注意力的MLP进行对比。结果显示：在足够多样性时，所有模型均实现约100%概念翻转率，Transformer训练更快；在低多样性下表现差异明显；OOD一般化随词表大小提升而恢复

**⚠️ 局限性**

仅在小型、从零训练的受控任务中验证；未完整复现LLM中所有符号头（如符号推导与检索头）；训练动态（先统计解后符号化）在大模型中可能不同；如何在MLP中实现符号机制仍未明晰

---

## 277. Auditable Human-Agent Teams for Autonomous X-ray Experimentation

**arXiv ID:** 2611.12933 | [PDF](https://arxiv.org/pdf/2611.12933v1)

**作者:** Gunik Luthra `[一作]` (Brookhaven National Laboratory), Esther H. R. Tsai `[通讯]` (Brookhaven National Laboratory)

**通讯引用:** 2603 | [OpenAlex ID](https://openalex.org/A5046305784)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

在同步辐射X射线实验中，使用基于Claude Code的多智能体架构（VISION-CC）实现了自适应闭环实验，实验对象为纳米粒子超晶格，支持人类动态设定实验目标并实时调整采样策略。

**💡 创新点**

创新点在于：①将自然语言交互与物理实验紧密结合，实现了低门槛、可持续的实验自动化；②通过完整的对话和事件日志构建可审计的决策谱系，量化人机与机机协作中的认知劳动分配；③在实验中展示了多智能体在不同阶段（获取、分析、优化、报告）之间的职责划分，并提供了错误修正与决策共识的经验分析。

**🔧 技术方法**

使用技术包括：Claude Code智能体平台、Bluesky实验控制、SciAnalysis数据处理、gpCAM高斯过程模型与自适应采样、xray-calc库进行X射线计算、Python文件系统化技能与全局指令、自然语言提示与模型验证、审计流水线与事件链验证。

**📊 数据集**

数据集为在NSLS-II 11-BM CMS光束线收集的纳米粒子超晶格（CdSe/CdS dot-in-rod和MnFe₂O₄）小角X射线散射（SAXS）帧，实验持续超过50小时，包含三次Beamtime，生成了多张二维散射图、1D强度曲线、二维峰强度/方向分布图。

**📈 对比分析**

与传统手工设置的闭环实验相比，VISION-CC将设置时间从≈6小时压缩至≈2小时，基本闭环配置可在≈1小时完成；实验中智能体对决策和错误修正的贡献率均超过50%，且在后两次实验中方法与科学贡献比例提升至56–61%；实验结果与人工设计的采样策略一致，且能够实时定位结构边界与取向域，验证了系统的可靠性与效能。

**⚠️ 局限性**

限制包括：仅在单一光束线与单个实验者上验证，缺乏跨域和多团队的通用性；对知识库和技能文件的构建需要人工指导，初期成本高；审计链依赖人工与模型一致性，存在不确定性；在大规模实验或多样化实验对象时，对同步控制与安全桥接的可扩展性尚未完全评估。

---

## 278. Towards Reliable VLM Judges: State-Conditional Invariance and Presentation-Aware Diagnostics

**arXiv ID:** 2611.13596 | [PDF](https://arxiv.org/pdf/2611.13596v1)

**作者:** Qinan Zhang `[一作]` (University of Science and Technology of China), Jian Di `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个基于 MiniGrid 仿真器的可执行诊断基准 GridWM‑Judge，用于评估视觉‑语言模型(VLM)评判者的可靠性

**💡 创新点**

提出 State‑Conditional Invariance (SCI) 原则，明确判决者在语义保持不变时对展示方式保持不变，却在因果干预改变结果时需改变判决

**🔧 技术方法**

结合确定性轨迹、完整/无线索/对抗变体、展示与表示探测以及动作/感知控制任务，利用 VLM 进行自动评判与自动评分

**📊 数据集**

使用 MiniGrid 六大环境族生成的轨迹数据，包含 Full、NoCue、CF 等变体，以及 3D 扩展实验

**📈 对比分析**

与多款前沿及开源 VLM（如 Claude、GPT‑4o、Gemini 等）对照，测评正确率、呈现稳定性、表示一致性与因果敏感度；结果显示不同模型表现差异明显，单一准确率无法体现可靠性

**⚠️ 局限性**

受限于仅涵盖确定性 2D 轨迹、有限的展示/表示变体以及对 VLM 内部机制的间接测评，难以完全覆盖现实复杂场景与模型内部因果推理能力

---

## 279. Veritas: A Verifier Generation Harness for Agent Tasks

**arXiv ID:** 2611.13336 | [PDF](https://arxiv.org/pdf/2611.13336v1)

**作者:** Daoan Zhang `[一作]` (ByteDance Inc.), Jianing Yin `[通讯]` (ByteDance Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种新的 harness（称为 ERH），通过加入 evidence header、domain rule library 与 authoring principles 三种机制来自动生成可靠的验证器；

**💡 创新点**

创新点在于将这三种机制系统化组合，并证明其显著提升生成验证器的准确性与交付率，可轻松迁移至其他 harness 与模型；

**🔧 技术方法**

采用 GLM‑5.2 大语言模型配合 OpenCode 等 plain harness，利用 prompt 机制实现 evidence header、规则库查询与作者原则校验；

**📊 数据集**

使用六个基准：SWE‑bench Verified、Terminal‑Bench 4.0、TB‑Science 0.1、LLMBar、JudgeBench 以及对不同厂商模型（GPT‑5.6‑sol、Qwen3.7‑Plus、Gemini 3.5 Flash）的评估；

**📈 对比分析**

通过将生成验证器与官方验证器的判定结果进行 Cohen’s κ 比较，实验显示 ERH 在所有四个执行计分 benchmark 上均优于七个 plain harness，最高提升约 0.08，且在多数 benchmark 上保持 0.70‑以上的 κ；

**⚠️ 局限性**

局限性包括：计算资源有限，只在单轮实验评估，主要以 GLM‑5.2 进行比较；候选人构造较为简单，真实任务的难度可能更高；官方验证器本身可能存在错误，导致“ground truth”不一定完全可靠；模型训练中可能已见过部分 benchmark 数据，影响泛化性。

---

## 280. Using LLM Agents to Evaluate Prosocial Interventions on Social Media Platforms

**arXiv ID:** 2611.12922 | [PDF](https://arxiv.org/pdf/2611.12922v1)

**作者:** Lucen Li `[一作]` (University of California, Davis), Magdalena Wojcieszak `[通讯]` (University of Warsaw)

**通讯引用:** 7798 | [OpenAlex ID](https://openalex.org/A5044652655)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

基于LLM代理的社交媒体模拟，使用OASIS框架在真实Twitter/X推文池中创建13,835名代理，评估三种提升已验证新闻曝光的干预（内容注入、新闻重新排序、名人来源提示），并将结果与人类实验对照。

**💡 创新点**

①在单一模拟环境中同时比较多种干预而无需人类实验；②首次将注入、位置和来源三种机制在LLM代理上直接对比；③采用剂量标准化效率将模拟结果与人类实验直接比较。

**🔧 技术方法**

OASIS框架、GPT‑4o‑mini LLM、Poisson GLMM统计模型、随机生成的20步新闻feed、四种干预算法（Inject‑2/Inject‑4/Upranking/Celebrity）。

**📊 数据集**

从Apify抓取的477k条英语推文（经过滤后435k条）+32k条高质量新闻推文，合计467k条；35个低偏见新闻账户；公开Twitter/X推文数据集（如twitter100m_tweets）以及真实人口统计分布用于生成代理属性。

**📈 对比分析**

采用Poisson GLMM估计每步新闻互动计数与比例，计算EMM和对照组差异；所有干预显著提升互动，Upranking效果最大（比Inject‑4高约4.5倍），注入剂量提升0.136，Celebrity提升0.120；与人类实验对照显示方向一致，剂量标准化效率相近。

**⚠️ 局限性**

代理属性仅限年龄、性别、教育，行为空间仅四种互动；feed构造缺乏真实平台的个性化与反馈循环；所有实验共享相同feed生成过程，可能夸大位置效应；仅使用单一LLM（GPT‑4o‑mini）与固定参数；人类验证仅覆盖注入干预，缺乏对Upranking和Celebrity的直接验证。

---

## 281. Comparative Evaluation of Deep Learning Models for Geographic Atrophy Segmentation in En Face OCT Images

**arXiv ID:** 2611.13191 | [PDF](https://arxiv.org/pdf/2611.13191v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 282. Decode Token Parallelism: Memory-Efficient Distributed Attention Parallelism for Scalable LLM Decoding

**arXiv ID:** 2611.13293 | [PDF](https://arxiv.org/pdf/2611.13293v1)

**作者:** Susav Shrestha `[一作]` (NVIDIA), Narasimha Reddy `[通讯]` (Texas A&M University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Decode Token Parallelism (TKNP)，在 LLM 解码阶段实现权重量化与注意力计算的解耦，使多节点能够共享 KV 缓存与注意力带宽，同时保持密集层的高效执行。

**💡 创新点**

创新点在于：① 将模型层分块分配给不同节点，保持权重驻留均衡；② 让每个节点仅负责自己请求的注意力计算，避免跨节点聚合；③ 通过双活根微批技术实现密集层与注意力计算的并行重叠；④ 让所有节点既承担 KV 缓存又参与注意力，显著提升内存利用率。

**🔧 技术方法**

技术手段包括：vLLM 推理框架、FlashAttention‑3、FP8（部分）与 BF16 量化、CUDA‑graph 并行、NVLink 组内 TP、InfiniBand 组间通信、双活根微批调度与 KV‑缓存分配策略。

**📊 数据集**

实验数据集：Llama 3.1 405B、Nemotron‑4 340B（FP8/ BF16 量化模型），以及合成长上下文负载（RULER）、L‑Eval、LongBench v1/v2 的真实请求混合，覆盖 8K–126K token 长度。

**📈 对比分析**

与 DP、TP、PP、DCP 等基线对比，TKNP 在 32/64/96 K 长上下文、1K‑输出 token 的解码任务中：吞吐量提升至 2.06×（DP）或 1.7×（DCP），在 32 H100 GPU 上达到约 90% 规模效率；SLO（≤100 ms/输出 token）满足率在 1 req/s 时达 100%，并在高负载下保持尾部延迟优势。

**⚠️ 局限性**

局限性包括：在小批量或短上下文场景下通信与同步成本可能超过收益；双活根会消耗额外权重量化，降低 KV 容量；实验仅在 H100 + InfiniBand 环境下验证，迁移到其它网络或低带宽环境需进一步评估。

---

## 283. SHADOWPREFILL: Keeping Your Next Model Warm for Per-Step Model Routing in Agentic LLM Serving

**arXiv ID:** 2611.13176 | [PDF](https://arxiv.org/pdf/2611.13176v1)

**作者:** Haoran Qiu `[一作]`, Mark Russinovich `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 WarmPrefill 系统，通过在未被选中模型上预热 KV 缓存，降低代理式 LLM 调度中的模型切换开销。

**💡 创新点**

创新点在于利用正常流量的空闲 GPU 能力，以低优先级阴影任务填充未使用模型的前缀缓存，并通过 SLO‑slack 控制器、shadow KV 管理和主内存溢写等技术实现无 SLO 回退。

**🔧 技术方法**

采用连续批处理推理引擎（vLLM/VIDUR）、前缀缓存、低优先级阴影作业、SLO slack 收集、shadow KV 阶段、主内存 KV 溢写、以及预填/解码解耦（P/D）架构。

**📊 数据集**

使用 xRouteBench、FusionBench、Router‑R1、GHCP‑coding（GitHub Copilot 代理轨迹）、SWE‑bench 以及 Azure 生产 LLM 推理流量做实验。

**📈 对比分析**

在 24 组模型对比 5 组基准上，与冷切换基线相比，WarmPrefill 平均降低 30–50% 的切换调用时延、20–40% 的 LLM 调用时延，并将路由 token 成本降低 25–35%；相对主机内存溢写和 SLO‑slack 控制器的 ablation 也验证了各模块的贡献。

**⚠️ 局限性**

局限性包括：在高负载时可用空闲容量下降，shadow KV 的容量受限导致长上下文的缓存失效；需要额外实现阴影任务调度和内存溢写，且对模型大小、上下文长度敏感。

---

## 284. Stinging the Swarm: Extracting Attacker Intelligence from a Twin-Controller OT Honeypot

**arXiv ID:** 2611.13406 | [PDF](https://arxiv.org/pdf/2611.13406v1)

**作者:** Canyon Grey Dell'Omo `[一作]` (Tractian), Joao Pedro de Carvalho Voltani `[通讯]` (Tractian)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于inline gate与mirror twin的工业控制系统防御管道，可在毫秒级判断并隔离恶意LLM驱动的控制器请求，同时在仿真镜像中记录攻击者行为。

**💡 创新点**

创新点在于将实时模型驱动的门控与可被恶意代理继续操作的镜像双重防御相结合，既阻断物理破坏又实现攻击者行为提取。

**🔧 技术方法**

采用小型LLM作为on-device判断器，基于Modbus/OPC-UA协议的寄存器读写接口，结合设备层校验和仿真镜像技术。

**📊 数据集**

使用合成水泵驱动设备族（water_abb_yaskawa）和单一开放权重LLM（Llama-3.3-70B）生成的攻击流量。

**📈 对比分析**

与传统阻断方法相比，该管道在四个隐蔽层级下实现了100%拒绝对关键寄存器的写入，并在三类攻击者中实现了高达98%的信息提取率。

**⚠️ 局限性**

局限包括仅评估单一LLM模型、使用单一合成设备族、对身份信息提取受限以及在更复杂环境下的可迁移性待验证。

---

## 285. TERM: Target-Free Jailbreak Defense via Entropy-Guided Residual Modulation

**arXiv ID:** 2611.13577 | [PDF](https://arxiv.org/pdf/2611.13577v1)

**作者:** Yunkai Dang `[一作]` (Nanjing University), Wenbin Li `[通讯]` (Nanjing University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无预定义安全目标的推理时防御方法——Target-free Entropy-guided Residual Modulation (TERM)，通过提示校准的注意力熵偏差来自适应调节模型自身残差更新；

**💡 创新点**

创新点在于：①利用提示特定的注意力熵基线校准，避免了对固定安全方向的依赖；②通过自适应时间累计与归一化生成控制信号，精准调节残差缩放；③仅对模型内部残差进行有界缩放，无需外部安全向量或拒绝规则；

**🔧 技术方法**

使用技术包括：注意力熵计算与校准、指数移动方差归一化、Sigmoid映射、时间累计控制、残差缩放（乘法）以及对Transformer层的监控与调节；

**📊 数据集**

使用的数据集涵盖：七类 jailbreak 攻击（AutoDAN、Cipher、GCG、Jailbroken、Multilingual、ReNeLLM、XSTest）以及两类过度拒绝基准（XSTest、OKTest），并在 LLaMA‑3.1‑8B、Qwen2.5‑7B‑Instruct、Gemma‑2‑9B‑it 等模型上评估；

**📈 对比分析**

在七种 jailbreak 测试中，TERM 在三大模型上平均提升 DSR 约 3.4–6.7个百分点，同时保持最低的 ORR；与七种基线（解码调控、方向导向等）相比，TERM 在综合安全评分（DSR‑ORR）上最高；在 AlpacaEval 上也保持与未防御模型相近的通用指令遵循性能；

**⚠️ 局限性**

局限性包括：①依赖于特定层的注意力熵分离度，层选择对性能影响较大；②对极端多轮自适应攻击或更强的对抗性提示仍需进一步验证；③在某些模型规模上，提升并非单调；④对实时性能的轻微影响，仍需要在生产部署中评估。

---

## 286. When Reconstruction Helps Event-Camera Person Triggers

**arXiv ID:** 2611.13488 | [PDF](https://arxiv.org/pdf/2611.13488v1)

**作者:** Yifan Shuai `[一作]` `[通讯]`, Yifan Shuai

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

评估事件相机通过不同的事件到图像表示（包括灰度累积、时间表面、三分区图像以及E2VID、FireNet+重建）在车载/手持情境下对人检测触发决策的影响。

**💡 创新点**

在可比较的窗口和连续重放实验中，系统性展示重建能显著提升与直接表示相比的召回率，并量化启动误报与处理成本的权衡。

**🔧 技术方法**

采用E2VID与FireNet+等稀疏重建网络、Jev与Qwen3-VL等视觉语言模型、直接灰度与时间表面编码以及连续时间回放框架。

**📊 数据集**

主要使用DSEC驾驶事件数据集（四个未参与校准的记录），并补充PEDRo手持DAVIS数据进行转移验证。

**📈 对比分析**

通过固定阈值的窗口评估、冻结阈值在新记录上的迁移测试、以及连续重放捕获与误报统计，结果显示E2VID在同等事件历史下召回率提升约70%，FireNet+参数量极少但召回略低；直接编码召回低至10-20%，并伴随更高误报。

**⚠️ 局限性**

实验受限于小规模开发集、阈值选择不确定、仅关注可见人类检测、缺乏更广泛的实时能耗与多模态评估，且连续重放长度有限。

---

## 287. The AI Era's Lines of Code? A Case Study on What an Engineer's Token Spend Measures

**arXiv ID:** 2611.13396 | [PDF](https://arxiv.org/pdf/2611.13396v1)

**作者:** Shyam Agarwal `[一作]` (Carnegie Mellon University), Bogdan Vasilescu `[通讯]` (Carnegie Mellon University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对JetBrains工程师在五周内的AI编码代理代币消费与提交量进行关联研究。

**💡 创新点**

证明代币消耗主要衡量工程师使用代理的程度，而非产出，且未与输出成正比。

**🔧 技术方法**

使用Poisson回归、固定效应模型和工具箱计量分析，结合内部代理调用日志与代码仓库记录。

**📊 数据集**

来自JetBrains内部的匿名调研问卷与该公司580名工程师的代理调用日志、提交记录。

**📈 对比分析**

在与提交量和其他工作指标比较后，发现代币消耗对提交量的弹性仅为0.26，远低于1，说明其预测力有限。

**⚠️ 局限性**

研究仅涵盖单一公司、无强制使用政策，且未对代理输出之外的工作进行完整记录，导致对代币消耗实际价值的估计可能偏低。

---

## 288. Trajectory-Regularized Distribution Distillation for One-Step Generation

**arXiv ID:** 2611.12978 | [PDF](https://arxiv.org/pdf/2611.12978v1)

**作者:** An Zhao `[一作]` (ByteDance Seed), Lingyun Sun `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新型的梯度正则化分布蒸馏方法（TRD），通过共享的动量混合策略同时完成一阶闭式生成、学生边际速度估计和教师轨迹预测，从而在一次或少量步的采样下实现高质量图像生成。

**💡 创新点**

创新点在于：① 把动量混合 ArcFlow 策略用作统一的学生输出，既生成终点又提供时序速度估计，消除了传统方法中需要额外训练的假速度网络；② 通过轨迹对齐、分布匹配、自己流匹配和分布保持四项损失协同训练，使目标更新保持逆 KL 方向并可控；③ 引入分布保持（distribution preservation）提升训练稳定性，并提供理论分析证明目标的理想极点。

**🔧 技术方法**

采用 ArcFlow 动量混合参数化，结合分布匹配（reverse KL）、自己流匹配（self‑flow matching）、轨迹对齐（trajectory alignment）和分布保持（distribution preservation）四项损失；同时利用教师的轨迹滚动、指数滑动平均、分布式训练等技术。

**📊 数据集**

在公开的 2,500 条文本提示集上对 SANA‑1.6B 和 Stable Diffusion 3.5 Medium 两个大型文本到图像模型进行蒸馏；训练数据来源为公开的多模态训练集，评估使用多种质量、覆盖率、对齐度等指标。

**📈 对比分析**

与仅轨迹监督（MeanFlow、PD 等）和仅分布监督（DMD、DMD2、SiD 等）基线相比，TRD 在 1 NFE 下在 FD‑DINOv2、偏好、审美、CLIP 相似度等指标上均优于对手，并在 2 NFE 时甚至超过多步教师的部分指标；但在多样性（DINO Diversity）和覆盖率（Recall‑DINOv2）上略逊于纯轨迹方法。

**⚠️ 局限性**

局限性包括：① 牺牲了一定的多样性和覆盖率；② 仍然是模式收敛的方向，可能导致输出分布过窄；③ 对分布保持的依赖表明若去掉会训练崩溃；④ 仅在两种模型上验证，需在更广泛模型家族和真实数据基准上进一步测试。

---

## 289. Möbius Interaction Filtering: Order-Selective Spectral Regularization for Multimodal Humor Detection

**arXiv ID:** 2611.12913 | [PDF](https://arxiv.org/pdf/2611.12913v1)

**作者:** Jiahao Qin `[一作]`, Yiwen Wang `[通讯]`

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Möbius Interaction Filtering (MoIF) 模型，用私有时序编码器和平衡递归交互学习多模态子集预测，并通过 Möbius 域的阶次递减滤波器实现鲁棒融合

**💡 创新点**

创新点在于将子集预测映射到 Möbius 表示，构造阶次选择的光谱正则化滤波器，使得高阶交互表达得到保留而其对决策的贡献被自动调节，从而提升估计稳定性

**🔧 技术方法**

采用私有时序编码器、平衡递归交互、三模态压缩、Möbius 变换与子集聚合的光谱滤波器，并联合主损失和子集监督损失进行训练

**📊 数据集**

使用 UR-FUNNY（幽默检测）以及 CMU-MOSI、CMU-MOSEI（情感/情绪强度预测）三大数据集

**📈 对比分析**

与十三种公开方法（TFN、LMF、MulT、MISA、Self-MM、ALMT、EMT、LNLN、MPLMM、CASP、DLF、GCE、TLRRB）在同一数据集上对比，MoIF 在准确率、加权 F1、MAE、相关系数等指标均位列第一或第二，并表现出最优的鲁棒性和稳定性

**⚠️ 局限性**

局限在于仅针对三模态的全子集构造，滤波器响应固定，难以自适应不同分布或更大模态集合；对动态质量评估和更复杂交互结构的扩展仍需进一步研究

---

## 290. Priority-Aware Deficit Round Robin for AXI4 Bandwidth Fairness in ASAP-7nm Process

**arXiv ID:** 2611.12979 | [PDF](https://arxiv.org/pdf/2611.12979v1)

**作者:** Tasmin Khan `[一作]` (Bangladesh University of Engineering and Technology), Ch. Md. Rakin Haider `[通讯]` (Bangladesh University of Engineering and Technology)

**通讯引用:** 97 | [OpenAlex ID](https://openalex.org/A5054311127)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了三种 AXI4 交叉口服务调度架构（轮询、QoS+老化、QoS+DRR），并在 ASAP7 7nm 工艺下完成 RTL 到 GDSII 的实现与评估。

**💡 创新点**

引入按位计费的 QoS‑aware Deficit Round‑Robin 使优先级与带宽份额分离，既保证低延迟又实现按比特公平性。

**🔧 技术方法**

使用 AXI4 协议、Deficit Round‑Robin 算法、QoS 级别、RTL 仿真、ASAP7 预测工艺、物理实现与 PPA 评估等技术。

**📊 数据集**

采用确定性流量生成器模拟三主机读取同一从机的场景，主机间 burst 长度差异和负载调节。

**📈 对比分析**

通过相同流量下的 RTL 仿真与 GDSII 后的面积、功耗、时序比较，QoS‑aware DRR 在饱和前后将延迟降低 61%（RR）/45%（QoS），并在全负载下实现 1:1 的比特公平。

**⚠️ 局限性**

增加约 8% 的面积与 10% 的功耗，时序延迟更大，且在低负载或主机对称时收益不大。

---

## 291. Understanding Computing Instructors' Perceptions of Student Sense of Belonging

**arXiv ID:** 2611.13237 | [PDF](https://arxiv.org/pdf/2611.13237v1)

**作者:** Ritesh Kanchi `[一作]` (Harvard University), Alex Oesterling `[通讯]` (Harvard University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对九名大型公共大学的入门计算机课程教师进行半结构化访谈，探究教师如何概念化、评估及支持学生归属感，及其面临的规模与资源限制。

**💡 创新点**

首次将教师视角与学生中心的归属感理论相结合，系统揭示教师在课程设计、TA利用和学生期望管理方面的实践与挑战，并提出针对教师的归属感干预设计思路。

**🔧 技术方法**

采用定性研究方法：半结构化访谈与归纳式主题分析；使用教师访谈文本进行编码与主题聚类。

**📊 数据集**

9名教师的访谈记录；包含教学经验、课程规模、TA安排、学生反馈等信息的文本数据。

**📈 对比分析**

该研究未使用对照实验或量化评估方法，主要通过访谈内容构建六大主题框架，未给出具体性能指标；其贡献在于理论与实践结合的深度分析。

**⚠️ 局限性**

样本量小且来自四所大型公共大学，可能存在自选偏倚；研究聚焦讲授式、TA支持的课程，难以推广至小班或高级课程；缺乏对不同地区、私立院校或非英语语境的验证。

---

## 292. SELD: Detecting Silent Evidence Loss in Context Compression beyond the Answer

**arXiv ID:** 2611.12903 | [PDF](https://arxiv.org/pdf/2611.12903v1)

**作者:** Wentao Lei `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Li Liu `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**通讯引用:** 25974 | [OpenAlex ID](https://openalex.org/A5100418783)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无监督、无需标签的无声证据损失检测器（SELD），并构建了长度匹配的证据删除与对照删除对（SEL-Pairs）基准。

**💡 创新点**

创新点在于通过比较压缩前后模型对未生成答案的log概率向量的角度差异，隔离证据模式的变化，解决传统仅关注答案一致性而忽视证据丢失的问题。

**🔧 技术方法**

技术主要包括：对读者模型的全词表log概率进行中心化、归一化，计算两向量的余弦相似度（即SELD分数）；对比JS散度、答案概率下降、LUMINA‑ECU等分布差异度量；使用精确长度匹配的对照实验。

**📊 数据集**

使用的数据集包括 HotpotQA、2WikiMultihopQA、MuSiQue 的原始和无答问题，以及 LLMLingua‑2、Provence、Granite‑3.3‑8B 生成的压缩输出，用于构造证据删除与对照删除对。

**📈 对比分析**

与传统方法比较，SELD 在 5 种读者模型、3 种压缩器、不同压缩率下的测试中，在未改变答案的对中以 83% 的通过率显著优于 JS 散度（70%）和答案概率下降（≈54%）；在全部指标上平均提升约 12–30 分（相对基线）。

**⚠️ 局限性**

局限性包括：使用的证据标注是“支持句子”而非“必要证据”，因此检测到的“缺失”不一定意味着答案无效；评估依赖精确长度匹配，覆盖范围有限；仅在英文多跳问答、1.5B–14B 规模的开源模型上验证，未覆盖其他语言或更大模型。

---

## 293. Few-Shot Residual Adaptation of Pretrained Latent World Models

**arXiv ID:** 2611.13592 | [PDF](https://arxiv.org/pdf/2611.13592v1)

**作者:** Jakub Suliga `[一作]` (Karlsruhe Institute of Technology), Rudolf Lioutikov `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究在预训练的潜在世界模型上进行少量自监督适配，解决因物理动力学变化导致的规划失效。

**💡 创新点**

提出双站点残差适配器：在预测器内部和输出处同时加入轻量级残差校正，保持原模型冻结，能在仅三次交互后恢复90%–100%成功率。

**🔧 技术方法**

技术包括基于Transformer的潜在世界模型（DUET‑DINO）、Adapter+ 低秩/非线性残差模块、外部 MLP 残差、CEM 控制器、以及基于预测误差的自监督损失。

**📊 数据集**

使用 Franka Panda 机器人在真实实验室环境中以及 RoboLab 仿真平台，任务为箱子推送，评估中心质量偏移与摩擦不均两种动力学偏移。

**📈 对比分析**

与冻结模型、完整微调、LoRA、单站点残差等方法对比；双站点适配在仿真与真实机器人上均能将成功率从 0% 提升至 90%–100%，显著优于单站点或仅微调的方案。

**⚠️ 局限性**

局限性包括仅考察了中心质量与摩擦两种偏移，适配器需针对每种偏移单独学习，且在跨域（仿真↔真实）或视觉变换下迁移性能有限。

---

## 294. An Interpretable Stylometric Framework for Early Chinese Buddhist Translator Styles: A Case Study on the Attribution of Disputed Works of An Shigao

**arXiv ID:** 2611.13636 | [PDF](https://arxiv.org/pdf/2611.13636v1)

**作者:** Yating Pan `[一作]` (Peking University), Jun Wang `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建可解释的风格分析框架，结合专家验证的句式与词汇特征，对早期汉语佛经翻译进行作者归属与风格演化分析。

**💡 创新点**

创新点在于将专家导向的可解释特征与统计检验相结合，零监督下实现与深度学习模型相当的归属精度，同时保留可解释性。

**🔧 技术方法**

使用手工定义与数据驱动的特征工程、Kruskal–Wallis检验、Z‑score标准化、Sigmoid变换以及基于chi‑square的相似度评分等统计方法。

**📊 数据集**

采用CBETA数字语料中的31篇已鉴定译本和8篇争议译本，总计约数十万字符的早期汉语佛经文本。

**📈 对比分析**

与BERT深度模型在归属任务上对比，双方均可达到约80%以上的置信度，专家特征方法在保持高精度的同时提供了可解释的风格证据。

**⚠️ 局限性**

局限在于文本长度不足导致特征稳定性下降，且仅涵盖东汉至西晋译本，未来需扩展到后期译者和更大规模语料。

---

## 295. MouldIt: Data Efficiency of Multi-Task Temporal Representation Learning in Injection Moulding

**arXiv ID:** 2611.13646 | [PDF](https://arxiv.org/pdf/2611.13646v1)

**作者:** Abdullah Ayad `[一作]` `[通讯]` (University of Technology Nuremberg), Abdullah Ayad (University of Technology Nuremberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种名为MouldIt的多任务时序变分自编码器，用于注塑过程中的部件分类、未知缺陷检测和排序，并在合成数据上评估其数据效率与多模态优势。

**💡 创新点**

引入多目标时序VAE结合卷积与注意力，联合学习表示、分类和重建，并通过创新评分实现单模型同时完成已知与未知缺陷识别，显示数据量和模态多样化可显著提升性能。

**🔧 技术方法**

时序卷积网络、注意力机制、变分自编码、BCE分类头、重建MSE、KLD正则、Mahalanobis距离和重构误差组成的复合异常得分。

**📊 数据集**

以德国Nuremberg技术大学公开的Erlanger Träger注塑零件单个真实循环为基础，合成出从50到16000个周期的单模态（压缩）和四模态（压、温、超声）数据集，并人工注入多种故障模式。

**📈 对比分析**

与kNN、逻辑回归、随机森林、PCA及始终接受的基准进行比较；在单模态下，MouldIt从4000周期开始超越经典方法，单模态排序损失为6.0%；在多模态下，MouldIt在200周期即可实现99%准确率，排序损失仅1.0%，远优于其他基线。

**⚠️ 局限性**

所有样本均源自单个零件与单种材料的合成数据，缺乏真实故障验证；阈值设定基于5%FPR在生产中较高；对模型超参数和Gaussian假设的依赖，以及未对不同成本不平衡进行调优。

---

## 296. Physics2Weight: Direct Neural Network Architecture and Weight Generation from Physical Laws Using Large Language Models

**arXiv ID:** 2611.12954 | [PDF](https://arxiv.org/pdf/2611.12954v1)

**作者:** Shogo Sato `[一作]` (NTT, Inc.), Takashi Sano `[通讯]` (NTT, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种名为 Physics2Weight（PtW）的框架，利用大语言模型直接生成神经网络的结构与权重，从而在不进行梯度优化的前提下求解偏微分方程。

**💡 创新点**

创新点在于三大策略的组合：① 物理约束提示（physics-constrained prompting）把守恒、边界等物理约束显式加入提示；② 物理引导架构编辑（physics-guided architecture editing）让 LLM 根据 PDE 的数学结构自动改写网络结构；③ 迭代权重细化（iterative weight refinement）在无梯度的情况下利用物理残差反馈持续优化权重。

**🔧 技术方法**

技术手段主要包括：使用 Claude Opus 5.5 作为 LLM 生成网络结构与权重；实现 WSP、PINN 以及其改进版本作为基线；通过物理残差、初始与边界误差等评价指标进行迭代和最终模型选取。

**📊 数据集**

实验数据集为四个典型的一维时空 PDE：热方程、对流方程、Burgers 方程和 Allen–Cahn 方程，分别对应不同的扩散、输运、非线性对流扩散和反应扩散问题。

**📈 对比分析**

与 PINN、WSP 及其各项 ablation（PC、AE、WR）比较显示，PtW 在热方程和对流方程上实现近机精度（相对 L₂误差可达 10⁻¹⁷），在 Burgers 方程与 Allen–Cahn 方程上也显著低于基线，但在非线性 PDE 上 PDE 残差仍高，表明物理一致性仍有提升空间。

**⚠️ 局限性**

主要局限性包括：仅在已知解析结构或一维 PDE 上验证，非线性 PDE 的局部残差高且对网格敏感；实验仅用单一 LLM 进行三次重复，缺乏更广泛的泛化评估；对更高维或更复杂物理模型的适用性尚未证明。

---

## 297. You Don't Need Feature Learning to Do In-Context Learning

**arXiv ID:** 2611.13431 | [PDF](https://arxiv.org/pdf/2611.13431v1)

**作者:** Joseph Turnbull `[一作]` (University Of California Berkeley), Jamie Simon `[通讯]` (University Of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过理论推导与实验验证，研究了在上下文学习（ICL）中记忆化与泛化的转变机制，重点分析了旋转不变核岭回归在线性回归ICL任务中的学习过程。

**💡 创新点**

创新点包括：①证明泛化解是记忆解的低阶截断，揭示记忆化是对高阶任务统计的逐步学习；②提供了旋转不变核的谱分解与累积矩分析；③用这一理论解释了ICL的转瞬即逝（transience）现象；④指出特征学习并非实现ICL所必需。

**🔧 技术方法**

技术手段：旋转不变核岭回归、谱偏差（spectral bias）理论、核特征分解、累积矩与高阶统计分析、随机特征网络与Transformer实验、早停与训练动态分析。

**📊 数据集**

数据集：合成数据，使用高斯输入 x∼N(0,1/d I_d) 和线性任务向量 θ∼N(0,I_d)，通过控制任务数 T、上下文长度 L、噪声 σ^2 等超参数生成不同实验场景。

**📈 对比分析**

评估方法：将模型在训练任务与新任务上的均方误差分别与记忆化oracle（dMMSE）和泛化oracle（ridge）进行对比。实验表明：训练误差随样本增多逐步下降至记忆解；新任务误差先逼近泛化oracle，随后随着高阶统计被学习而上升至记忆解；不同任务多样性、kurtosis 以及特征学习方式均影响此转变过程。

**⚠️ 局限性**

局限性：仅针对线性ICL任务与旋转不变核；未探讨非线性、自然语言或真实任务场景；理论依赖高斯假设且对大模型、多任务设置的泛化性待验证。

---

## 298. Multi-Stage Phase-Aware Self-Supervised Reconstruction for Multi-Echo Gradient-Echo MRI

**arXiv ID:** 2611.13352 | [PDF](https://arxiv.org/pdf/2611.13352v1)

**作者:** Juan Liu `[一作]` (University of Texas Southwestern Medical Center), Nian Wang `[通讯]` (University of Texas Southwestern Medical Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种多阶段、相位感知的自监督重建框架MSPR-GRE，用于加速高分辨率多回波梯度回波（mGRE）MRI的图像重建与定量映射。

**💡 创新点**

创新点在于：①利用递进的k空间扩展与相位优先更新，实现粗到细的相位估计与修正；②针对回波数不同采用子空间或图像域重建；③在自监督训练中逐阶段冻结前置网络，避免先前误差的累积。

**🔧 技术方法**

技术包括：自监督重建（SSDU）、深度可展开网络、相位演化拟合、基于衰减的时间子空间、相位先验与子空间投影、以及多阶段递进训练策略。

**📊 数据集**

使用数据集：两套 9.4 T 离体小鼠脑（30 µm 八回波、18 µm 双回波）和 7 T AHEAD 人脑（四回波）数据。

**📈 对比分析**

与传统 SSDU、LLR、L1-小波等方法比较，MSPR-GRE 在 R=4、8、12 的加速条件下在幅度图、T2* 图和 QSM 上实现更低的 NRMSE、更高的 PSNR/SSIM，尤其在高加速因子下优势显著。

**⚠️ 局限性**

局限性包括：实验样本量有限，未对相位估计精度进行定量评估，子空间与阶段数的选择受制于数据特性，且长时间扫描可能引入系统漂移未被监测。

---

## 299. ABRR: Transforming Temporal Sentence Grounding with Wasserstein-Adaptive Refinement and Precision Ranking

**arXiv ID:** 2611.13310 | [PDF](https://arxiv.org/pdf/2611.13310v1)

**作者:** Xiang Fang `[一作]` (Nanyang Technological University), Daizong Liu `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种模型无关的后处理框架 ABRR，用来纠正视频句子定位中的量化误差并重新排序预测段落以提升定位精度。

**💡 创新点**

创新点包括：① 使用高斯核平滑时间分布以降低噪声；② 通过 2‑Wasserstein 距离和 Sinkhorn 迭代将预测边界对齐到连续的目标分布，实现基于最优传输的边界精细化；③ 结合定位精度与语义相关性的新质量得分，对预测结果进行重排序。

**🔧 技术方法**

技术手段主要包括：时间分布平滑、最优传输理论（Monge‑Kantorovich 与 2‑Wasserstein 距离）、Sinkhorn 迭代、余弦相似度计算与加权组合质量分数。

**📊 数据集**

在 Charades-STA、ActivityNet Captions 与 QVHighlights 三个公开基准数据集上进行实验。

**📈 对比分析**

与多种先进 TSG 模型（如 UniVTG、QD-DETR、Moment-DETR 等）对比，ABRR 在 R@1@0.7、mIoU 等指标均提升 3–5% 甚至超过同类后处理方法，证明其显著的性能提升。

**⚠️ 局限性**

局限性包括：只能改进已有模型的预测，无法解决根本的特征表示不足；对极短事件或完全错过目标的情况效果有限；需预先缓存视频/文本特征，增加一次性存储成本。

---

## 300. POEM: Output-Relative Block-Sparse Attention for Long-Context Prefilling

**arXiv ID:** 2611.13578 | [PDF](https://arxiv.org/pdf/2611.13578v1)

**作者:** Haoyi Wu `[一作]` (Nanjing University), Wenbin Li `[通讯]` (Nanjing University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的块稀疏注意力方法 POEM，利用输出误差最小化原则在预填充阶段高效筛选重要块；

**💡 创新点**

创新点在于将注意力质量（块权重）与输出偏差（块输出与完整注意力输出的距离）融合成输出相对分数，并通过头敏感度驱动的离线阈值校准，实现对每个注意力头的自适应保留阈值；

**🔧 技术方法**

采用块稀疏注意力框架，利用低精度查询键估计块权重、协方差校正的输出近似、Taylor 展开、顶部p（top‑p）筛选以及 FlashAttention‑2 后端加速；

**📊 数据集**

在 Llama‑3.1‑8B、Qwen3‑8B、Qwen3‑14B 三大模型上，在 LongBench、RULER 与 PG19 等公开基准上进行评估；

**📈 对比分析**

与稠密 FlashAttention‑2 以及四种训练无关稀疏基线（MInference、FlexPrefill、XAttention、STEM）对比，POEM 在所有模型上保持与稠密注意力相近的准确率，且在 128K 上实现最高 6.5 倍的速度提升；

**⚠️ 局限性**

局限性包括：仅针对推理阶段，未对解码时 KV 缓存进行优化；输出近似仍可能引入误差，导致某些场景下事实错误或偏差放大；对安全关键或高风险任务不宜直接部署。

---

## 301. BioMem: Agentic Memory for Biomedical LLM Agents

**arXiv ID:** 2611.13426 | [PDF](https://arxiv.org/pdf/2611.13426v1)

**作者:** Mingchen Li `[一作]` (University of Massachusetts Amherst), Hong Yu `[通讯]` (University of Massachusetts Amherst)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统评估并比较了五种医疗记忆方案在六个生物医学任务（知识推理、临床预测与长时会话记忆）中的表现，发现检索质量、信息保真度和利用效果是提升性能的关键因素，并提出了无需训练的Hybrid Memory Retrieval and Utilization（HMRU）框架，显著提高了问答与对话记忆准确率；

**💡 创新点**

提出了针对生物医学场景的检索+利用双管齐下的HMRU框架，解决了传统记忆系统在检索覆盖、答案利用和信息演化过程中出现的偏差与错误；

**🔧 技术方法**

基于BM25+稠密检索、跨编码器重排以及任务指令增强的检索-利用框架；

**📊 数据集**

使用RiTeK-PharmKG、RiTeK-ADInt、MIMIC‑III死亡率、MIMIC‑III再入院、TCGA‑Reports、MIMIC‑IV过量、RealMem医疗子集等七个公开数据集；

**📈 对比分析**

在四个LLM（Qwen3‑8B、Qwen3‑30B‑A3B‑Instruct‑2507、Meditron3‑Qwen2.5‑14B、HuatuoGPT‑3‑27B）上与全记忆对比，HMRU平均提升8.6%相对性能，错误率降低24.9%，与全上下文推断的性能差距缩小48%；

**⚠️ 局限性**

检索到的证据并不总能被正确利用，语义相似度与临床结果相似度不一致，记忆演化过程中可能引入不受支持的信息，长时对话性能受历史长度、信息分布等多因素影响，HMRU虽然训练自由但仍需在不同任务上调优并验证跨域泛化能力。

---

## 302. Curriculum proof repair with learned counterexamples

**arXiv ID:** 2611.13627 | [PDF](https://arxiv.org/pdf/2611.13627v1)

**作者:** Yuechun Sun `[一作]` (University of Chicago), Kexin Pei `[通讯]` (University of Chicago)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了名为TopoVerus的LM驱动的Verus证明修复框架，内部化基于counterexample的诊断，单步生成结构化counterexample、修复理由和修复证明。

**💡 创新点**

创新点在于将counterexample生成视为诊断学习目标，使用强化学习将其与安全修复相结合，并提出基于依赖图的Topological Graph Reward形成微观课程化的修复顺序。

**🔧 技术方法**

使用QwQ-32B语言模型，结合GRPO强化学习、可验证奖励（RLVR）和安全门控，输出结构化诊断和修复代码。

**📊 数据集**

采用七个Verus基准（VerusBench、DafnyBench、HumanEval、LeetCode、VeriCogen、ObfsBench、VeruSAGE）共471个问题，训练集包含461条修复轨迹。

**📈 对比分析**

与云端LM（Sonnet‑4.5、GPT‑4o、Qwen3‑Coder、DeepSeek‑V4）以及自研模型比较，Safe‑Pass@1/3分别提升至61.6%/72.2%，比Sonnet‑4.5高5.7/6.3个百分点，在ObfsBench等高难度套件表现尤为突出。

**⚠️ 局限性**

主要限制是聚焦于循环不变式，难以直接推广到其他验证结构；生成的counterexample缺乏形式化保证，可能出现幻觉或不完整的执行轨迹。

---

## 303. SafeRoute-MAS: Coverage-Aware and Risk-Calibrated Message Routing for Multi-Agent Scientific Reasoning

**arXiv ID:** 2611.13454 | [PDF](https://arxiv.org/pdf/2611.13454v1)

**作者:** Zhiming Xue `[一作]` (Northeastern University), Jing Yang `[通讯]` (Washington University in St. Louis)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SafeRoute-MAS，一种在多代理 LLM 委员会中基于风险校准的消息路由器，能够在只使用前期信息的情况下决定是否保留或删去单条消息。

**💡 创新点**

创新点在于：① 将覆盖率感知与风险校准结合，形成可冻结阈值的选择门；② 引入以数值目标为基础的资格检验流程和 Keep‑All 后备策略；③ 通过多指标安全门（准确率、token 节省、风险上限）实现全局投产决策。

**🔧 技术方法**

技术手段包括：TF–IDF 1–2 词的文本特征、结构化元数据、类别平衡的 logistic SGD 与 sigmoid 校准、验证冻结的阈值与多样性规则，以及后续的目标后端资格评估程序。

**📊 数据集**

使用的数据集为 SciAgentTrace v2.1.0（JEEBench、LAB‑Bench、MaScQA、Omni‑MATH‑2、SciBench）提供的委托缓存（5,328 个委员会、93,282 次行动），以及公测 gpt‑oss‑120b 端点上的 4,580 次集成测试。

**📈 对比分析**

与 Keep‑All、随机 prune‑one、最低置信度、最长消息、少数答案等基线对比，SafeRoute‑MAS 在源缓存上提升 1.80pp 准确率并减少 12.8% token，随机基线提升 0.93pp；但在加权的端点样本中未能显著提高准确率或显示选择优势，风险上限在部分子组（如 LAB‑Bench）超标。

**⚠️ 局限性**

局限性：① 未在端点上实现显著准确率提升或选择优势；② 某些子组的风险上限未达 5% 目标；③ 资格规则在实验后期定义，缺乏前瞻性验证；④ 仅针对单删操作，无法推广到多删或在线实时重跑场景。

---

## 304. Ground Reaction Force-Aware Model Predictive Control for Underactuated Upright Physical Assistance

**arXiv ID:** 2611.13204 | [PDF](https://arxiv.org/pdf/2611.13204v1)

**作者:** Darren Biskup `[一作]` (Columbia University), Sunil K. Agrawal `[通讯]` (Columbia University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并验证了一种基于地面反作用力（GRF）的模型预测控制（GRF‑MPC），用于在仅四根电缆的欠驱动系统中实现正姿势物理辅助；

**💡 创新点**

创新点在于将实时测量的GRF和CoP直接嵌入人类质心动力学模型，构造子线性凸二次规划，使欠驱动系统在100 Hz下实现实时规划；

**🔧 技术方法**

使用技术包括：质心动力学建模、线性化与离散化、凸QP求解、层级控制架构、Vive跟踪与力板测量、PID低层张力跟踪；

**📊 数据集**

数据集为6名健康成人（共300余次试验），涵盖稳定站立扰动、深蹲（冲击性）和可用工作空间三大任务；

**📈 对比分析**

与全驱动8根电缆阻尼控制以及透明模式进行比较；GRF‑MPC在峰值误差下降≈19.6%，稳态时间下降≈48%，轨迹误差下降≈24.1%，机械功耗降低≈28.7%，工作空间增大≈45.9%；

**⚠️ 局限性**

限制包括：单刚体近似和惯量矩估计简化、GRF视为预测期内常数导致预测误差、仅在受力板环境下验证、仅对健康年轻成人有效、未实现患者自节奏或可穿戴GRF估计等。

---

## 305. Transformers Provably Learn Graph Search: Training Dynamics and the Exponential Power of Depth

**arXiv ID:** 2611.13594 | [PDF](https://arxiv.org/pdf/2611.13594v1)

**作者:** Xutao Ma `[一作]` (University of California Berkeley), Somayeh Sojoudi `[通讯]` (University of California Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了Transformer如何在有向图中实现深度优先搜索（DFS），阐明了层数对表达力和搜索效率的影响；

**💡 创新点**

发现了一个关键深度阈值：单层Transformer需要Ω(N²)表示大小，二层即可实现近线性（O(NlogN)）DFS，且每增加一层可将可达范围对数级提升；

**🔧 技术方法**

使用多头注意力、前向和回溯两头的二层Transformer结构，结合离散Chain‑of‑Thought（CoT）提示，构建了可证明的DFS实现，并通过梯度流分析展示其可训练性；

**📊 数据集**

在实验中使用自定义的有向图训练集（固定骨架加随机顶点置换）以及随机生成的16节点有向图评估泛化性能；

**📈 对比分析**

通过与理论推导的参数增长率、泛化误差指数对比，验证了二层模型在梯度流下参数呈对数增长，泛化误差随时间按t⁻¹（非根回溯）和t⁻^(N-1)/N（根回溯）衰减，实验结果与理论吻合；

**⚠️ 局限性**

局限性包括：训练过程依赖对称的训练数据和预置的第一层；实验使用的DFS仅为离散精确搜索，未涵盖近似或启发式搜索；未对更复杂的推理任务（如软约束、概率转移）进行扩展。

---

## 306. Domain-Routed Delta Optimization for Multi-Teacher On-Policy Distillation

**arXiv ID:** 2611.13249 | [PDF](https://arxiv.org/pdf/2611.13249v1)

**作者:** Lin Yao `[一作]` `[通讯]` (Shanghai Jiao Tong University), Lin Yao (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多教师 on‑policy 蒸馏提出 Domain‑Routed Delta Optimization (DRD‑MOPD)，通过将每个专家的增量初始化为可训练组件并指定域到组件的梯度路由，从而实现共享稠密学生模型的域特定更新。

**💡 创新点**

创新在于将专家增量拆分为可训练组件并为每个域定义四种独立的梯度路由与优化器状态，实现对专家几何结构的可追踪更新并保持几何保留。

**🔧 技术方法**

采用任务算子（task arithmetic）初始化、独立 AdamW 记录、在共享稠密网络中进行域监督的逆 KL 蒸馏，并通过角度、投影系数和能量分析评估增量保持。

**📊 数据集**

在 Open‑MOPD 3 域设置下，使用 AIME 2024/2025、LiveCodeBench v5/v6、IFEval 与 IFBench 等六个基准对 Math、Code 与 IF 领域进行评测。

**📈 对比分析**

与原始专家、任务算子合并模型以及官方 Release 的 Open‑MOPD Final 进行对比，20 次更新后得到的模型在六项基准上的宏平均分从 32.44% 提升至 32.72%，显著提升 IF 分数但 Math、Code 略有下降。

**⚠️ 局限性**

比较受限于不同训练/评测环境，评估仅针对单个 20 步检查点，且缺乏跨域稳定性与因果验证，扩展到更多专家和不同分片方案仍待研究。

---

## 307. DUO: Dual Readouts of Diffusion Models for OOD Detection

**arXiv ID:** 2611.13304 | [PDF](https://arxiv.org/pdf/2611.13304v1)

**作者:** Alireza Javid `[一作]` (University of California, San Diego), Nuria González Prelcic `[通讯]` (University of California, San Diego)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无监督、训练无关的 OOD 检测方法 DUO，利用冻结扩散模型的重建误差和中间特征敏感性进行融合。

**💡 创新点**

创新点在于结合重建误差与中间特征 Jacobian 的领先谱，理论证明在重建误差失效时该谱仍能有效区分 OOD，并设计仅用 ID 进行校准的全流程。

**🔧 技术方法**

使用了扩散模型、Jacobian 近似、中心差分探测、谱聚类、标准化融合以及 ID 仅校准的噪声级选择等技术。

**📊 数据集**

在 CIFAR‑10/100、SVHN、CelebA、TinyImageNet、FFHQ‑256、ImageNet‑64、AFHQ、Microscopy 等多种 ID–OOD 组合上进行评估。

**📈 对比分析**

与 Likelihood、Diffusion‑based、EigenScore 等基线对比，平均 AUROC 提升至 0.897，较 EigenScore 提升约 5.9%，在高分辨率和 near‑OOD 上亦取得 SOTA，同时推理速度比 EigenScore 快 30×。

**⚠️ 局限性**

局限性在于需先训练好 ID 扩散模型，且敏感性通道需要额外前向传播，算力仍高于单一重建，限制了在数据稀缺或训练成本高的场景中的适用性。

---

## 308. picoEmergencyNet : Attention-Enhanced Ultra-Small Deep Learning Model for Aerial Image Classification in Emergency Monitoring

**arXiv ID:** 2611.13638 | [PDF](https://arxiv.org/pdf/2611.13638v1)

**作者:** Christos Kyrkou `[一作]` `[通讯]` (University of Cyprus), Christos Kyrkou (University of Cyprus)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种超轻量化的卷积神经网络picoEmergencyNet，用于无人机航拍灾害图像的实时分类。

**💡 创新点**

通过结合深度可分离膨胀卷积、多尺度并行分支、通道洗牌和高效通道注意力，构建了专门针对低资源环境的高效架构，参数仅23k、FLOPs 7M、模型大小0.1 MB，显著提升了效率。

**🔧 技术方法**

深度可分离膨胀卷积、通道洗牌（Channel Shuffle）、Efficient Channel Attention（ECA）、多分支并行卷积、BatchNorm+LeakyReLU激活、AdamW优化及多样化数据增强等技术。

**📊 数据集**

AIDERv2和AIDER两个公开的航空灾害图像数据集。

**📈 对比分析**

在AIDERv2上与MobileNetV2、MobileViT、DirecNetV2等小模型对比，F1达到96.1%，参数和FLOPs比竞争模型小约35×；在AIDER上F1 95.0%；在NVIDIA Jetson Orin上GPU 258 FPS、CPU 46 FPS，激活内存仅4.4 MB，显著优于同类模型TakuNet。

**⚠️ 局限性**

对更复杂或类别更稀缺的场景鲁棒性略逊，且当前未结合量化/剪枝等进一步压缩技术，未来需在多标签或更大数据集上验证其泛化能力。

---

## 309. IFRS Compliance Verification: An Open Multi-LLM Benchmark and Content-Conditioned Ensemble Classifier

**arXiv ID:** 2611.13595 | [PDF](https://arxiv.org/pdf/2611.13595v1)

**作者:** Aghoghomena Akasukpe `[一作]` (Ontario Tech University), Miguel Vargas Martin `[通讯]` (Ontario Tech University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个多阶段框架，用于验证企业财务披露是否符合IFRS标准，结合检索、LLM多模型标注和内容条件的集成分类器；

**💡 创新点**

创新点在于：①构建公开的IFRS合规验证基准（≈20k条规条-披露对）；②通过三阶段领域自适应句向量（MLM→TSDAE/SimCSE→监督对比）提升检索质量；③设计基于梯度提升树的内容条件学习型集成器，利用LLM概率特征与冻结的句向量共同决定最终标签，显著降低单模型偏差；

**🔧 技术方法**

使用的技术包括：BERT在SEC EDGAR上微调、TSDAE、SimCSE、监督对比学习、三模型LLM（GPT‑4.1、Claude Sonnet 4、Gemini 2.5 Pro）标注、Tree‑structured Parzen Estimator优化的XGBoost元学习器；

**📊 数据集**

使用的数据集为：3,489条IFRS标准段落（通过LlamaParse提取）、39,404条加拿大上市公司财报段落（SEDAR+）、约19,998条规条-披露对（11,161条LLM一致标注、8,837条人工标注），以及对应的BERT嵌入检查点；

**📈 对比分析**

与多种无监督聚合方法（硬/软投票、Dawid–Skene、MACE、CROWDLAB等）及单模型LLM相比，学习型集成器在人工标注子集上达到81.7%加权F1，超越最优单模型（Gemini 2.5 Pro 71.3%）和最佳无监督聚合（MACE 72.2%），并显著提升对少数类（未合规/合规）的召回；

**⚠️ 局限性**

局限性包括：仅评估IFRS和英语、单文件级别、三模型LLM小队、仅对检索候选对进行标注、人工标注覆盖率低、数据偏向加拿大中小公司、类不平衡严重、无法验证对大型或其他会计准则的泛化，以及对商业审计工具缺乏对比；

---

## 310. Unified Optimization of Traffic Signal and Lane Channelization for an Isolated Intersection Using Reinforcement Learning

**arXiv ID:** 2611.13599 | [PDF](https://arxiv.org/pdf/2611.13599v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 311. Reliability of Claude Code as an Agentic AI Grading System for Handwritten University Exams

**arXiv ID:** 2611.13217 | [PDF](https://arxiv.org/pdf/2611.13217v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 312. MICRO: A Surgery-Phase Retinal Image Rigid Registration Benchmark and One-Step Frequency-Structure Similarity Registration Learning

**arXiv ID:** 2611.13045 | [PDF](https://arxiv.org/pdf/2611.13045v1)

**作者:** Mingdao Gong `[一作]` (Southern University of Science and Technology), Jiang Liu `[通讯]` (Southern University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文先构建了MICRO公共基准数据集，涵盖1,365对跨模态视网膜图像，覆盖术前、术中和术后三个阶段；随后提出了一种一站式频域结构相似性注册框架OFS^2Reg，利用跨模态血管结构相似性构建关键点拓扑约束，实现精准的视网膜图像刚性配准。

**💡 创新点**

创新点在于：①从关键点拓扑约束角度重新定义MRIRG任务，摆脱传统分割-检测两步流程；②设计频域引导结构保持（FGSP）描述子，融合高低频动态核以突出血管结构；③提出跨模态血管结构相似性学习（CMVS^2）——单模态稳定关键点挖掘、双向几何投影、跨模态三元组学习——进一步细化拓扑约束；④在共享检测网络下实现一站式注册，显著提升对跨模态差异的鲁棒性。

**🔧 技术方法**

核心技术包括：共享检测网络（SuperPoint‑style encoder），FGSP描述子（频域动态核生成器FDKG、低频聚合器LFA、高频增强器HFE），CMVS^2学习模块（SSKM、BGPM、CMTL），自监督检测损失、跨模态描述子损失和拓扑约束损失；实验使用PyTorch、Adam优化器、标准指标（RMSE、SR、mAUC）评估性能，并在多种CNN与基础模型上验证泛化。

**📊 数据集**

数据集：MICRO基准，包含1,365对跨模态视网膜图像，分为MICRO-pre、MICRO-pre2intra、MICRO-pre2post三子集，涵盖CF、FA、ICGA、UWF、彩色眼底、荧光素血管造影等多种成像模式，覆盖AMD、RVO、ROP、HM等疾病。

**📈 对比分析**

与传统方法（SIFT、GDB‑ICP）、两步深度注册（VDD‑Reg、EyeLiner）以及一站式深度注册（SuperPoint、SuperRetina、SuperJunction、CGTME）对比，OFS^2Reg在MICRO-pre集上实现RMSE 5.08、SR 0.784、mAUC 0.641，显著优于其他方法；在术中和术后子集亦保持更低RMSE和更高SR，显示出良好的OOS泛化能力。

**⚠️ 局限性**

局限性：①仅针对刚性配准，未处理非刚性形变；②依赖于跨模态血管相似性，可能对血管缺失或严重病变区域表现不足；③训练及推理时间相对较长，缺乏实时性能评估；④数据集虽然覆盖多阶段，但仍未包含所有可能的成像模式和极端病例，需进一步扩展。

---

## 313. Benchmarking LLM Next-Step Recommendations with Knowledge Space Theory

**arXiv ID:** 2611.13667 | [PDF](https://arxiv.org/pdf/2611.13667v1)

**作者:** Peter Steiner `[一作]` (St.Gallen University of Teacher Education), Jan Hochweber `[通讯]` (St.Gallen University of Teacher Education)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出并验证了一种基于知识空间理论（KST）的基准方法，用于评估大型语言模型在给定领域和学习状态下生成的下一步学习推荐的结构准确性。

**💡 创新点**

创新点在于将KST作为正式的参考框架，能够自动计算可接受的下一步集合并与LLM输出进行精确集合、Jaccard相似度、精确率等多指标对比，从而实现对模型-提示组合的结构化、可复现性评估；并通过实验展示提示信息量和领域差异对性能的显著影响。

**🔧 技术方法**

所用技术包括：知识空间理论（KST）与外缘（outer fringe）计算、LLM提示设计与API调用、JSON方案校验、集合相似度与误差度量（精确率、召回率、F1、Jaccard、精确集合准确率）以及Bootstrap和随机化检验等统计方法。

**📊 数据集**

数据集：四个线性函数数学领域（共23项任务、50个学习状态），通过对六个LLM（GPT‑5.6 Sol、GPT‑5.6 Terra、Claude Fable 5、Claude Sonnet 5、Gemini 3.1 Pro Preview、Gemini 3.7 Flash）在四个提示条件（P0–P3）和两种任务呈现顺序下生成的2,400条响应，用于计算推荐集合与KST外缘的匹配度。

**📈 对比分析**

比较方法：在每个学习状态下，将模型输出的推荐集合与KST外缘集合做集合比较，计算精确集合准确率、精确率、召回率、F1、Jaccard相似度；结果显示：在仅提供描述的条件下准确率约61%–63%；当提示包含参考先决关系时准确率提升至96%–98%，模型之间的差异随提示和领域显著变化；不同呈现顺序的推荐一致率约85%，表明存在一定的随机性与呈现效应。

**⚠️ 局限性**

限制：仅在四个小型数学领域进行实验，难以推广至更大规模或不同学科；评估仅关注结构准确性，未考虑解释质量、学习效果或教师/学生的实际接受度；参考KST结构的质量依赖于领域专家制定，若参考结构不符合教学实践，得分的教育意义有限。

---

## 314. Prompt2Echo (P2Echo): Text-Prompted Multi-View Echocardiography Segmentation with One Model

**arXiv ID:** 2611.13425 | [PDF](https://arxiv.org/pdf/2611.13425v1)

**作者:** Moein Heidari `[一作]` (University of British Columbia), Ilker Hacihaliloglu `[通讯]` (University of British Columbia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了 P2Echo，一种统一的、基于文本提示的多视角超声心动图分割模型。

**💡 创新点**

通过 DyITA 模块实现跨模态动态对齐和解码器特征的实时调节，显著提升跨视角鲁棒性和边界精度。

**🔧 技术方法**

结合 PVTv2 视觉编码器、冻结 Qwen3 文本编码器、DyITA 动态图像‑文本对齐以及 BDoU 损失等技术。

**📊 数据集**

在八个内部数据集（Camus、CardiacUDA、EchoCP、EchoNet‑Pediatric、SegRWMA 等）以及三个外部数据集（CardiacNet、EchoNet‑Dynamic、HMC_QU）上进行训练与评估。

**📈 对比分析**

与传统 nnUNet、SwinUNet、MissFormer、CENet 以及基于 SAM 的 EchoONE、SAMUS、Grounding DINO‑US‑SAM 等方法对比，P2Echo 在大多数视图上获得最高 Dice/IoU，计算量仅 6.90 GFLOPs、FPS 64.68，显著优于其他基准。

**⚠️ 局限性**

仅支持已知的四个标准视角，未验证对未知视角的泛化；标注不均衡导致部分结构（如 MYO）仍受限；对临床测量的直接影响尚未评估。

---

## 315. Joint-State Inference for Autonomous Excavators Using Cab-View Vision-Control Fusion

**arXiv ID:** 2611.13690 | [PDF](https://arxiv.org/pdf/2611.13690v1)

**作者:** Junwei Li `[一作]` (Northeastern University), Jianqiang Wang `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种闭环自主挖掘框架，利用驾驶舱视图图像、操纵杆历史输入和滚动状态缓冲，实现无附加传感器的挖掘机关节状态连续推理并作为闭环控制反馈。

**💡 创新点**

创新点包括：①将关节范围约束嵌入学习目标以提升物理一致性；②为每个关节设计专属时序编码器并引入桶关节的历史状态回馈；③结合轻量视觉+控制先验实现对部分可见度缺失的鲁棒推理；④将推理模块与ACT控制策略无缝耦合，形成完整闭环。

**🔧 技术方法**

使用ResNet‑18视觉编码器 + LSTM时序编码 + 线性回归头；引入ACT（Action Chunking with Transformers）作为控制策略；训练时采用联合误差和关节范围惩罚；在推理时采样频率30 Hz；在实验中使用了Adam优化器。

**📊 数据集**

使用公开的Fuxi挖掘机运动数据集（用于视觉‑单独基线）和自行构建的AGX Dynamics for Unity仿真数据集（50个工况，共2,000步/工况，30 Hz）。

**📈 对比分析**

与仅使用视觉输入的基线相比，加入操纵杆历史和滚动状态后，桶关节MAE从23.4°降至7.1°；整体关节平均误差从4.37°降至0.46°；在闭环实验中，推理状态下成功率为95%（相较于100%），平均推理延迟10.11 ms/步；与地面真值相比，推理状态的采样延迟仅多1.87 ms。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，缺乏真实工地测试；对环境干扰（尘埃、雨雾、摄像头模糊）研究不足；出现严重遮挡时仍可能累积误差；需要进一步提升跨场景泛化和失效模式分析。

---

## 316. CAFE: Closed-Loop Autonomous Task Formulation and Execution without Instance-Specific Instructions

**arXiv ID:** 2611.12983 | [PDF](https://arxiv.org/pdf/2611.12983v1)

**作者:** Mingjia Chen `[一作]` (Nanjing Normal University), Jun Chen `[通讯]` (Nanjing Normal University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个基于在线语义记忆的闭环系统 CAFE，实现了在没有实例化目标或动作序列的情况下，基于场景观察自动发现并执行清洁、安全与存储/组织任务。

**💡 创新点**

创新点包括：① 将任务发现、规划与执行通过记忆共享实现循环闭环；② 引入三阶段 Tri‑LLM（Draft‑Complete‑Audit）来完成任务提议、细化与依赖校验；③ 通过可选择触发与功能区域机制实现高效的任务采纳与空间绑定；④ 反馈驱动的结果修正使得执行失败能自动重规划。

**🔧 技术方法**

采用大型语言模型（DeepSeek‑V3、Qwen3‑Max、Gemini‑2.5‑Pro）进行语义记忆推理，配合视觉感知模块、功能区识别、执行器接口和 LLM 生成的行动序列。

**📊 数据集**

使用 AI2‑THOR 的 12 个室内场景（浴室、客厅、厨房）进行仿真实验，并在 LimoCobot 上进行物理验证。

**📈 对比分析**

在三种复杂度层级（Simple、Compound、Complex）和三种 LLM 基础上与 Single‑LLM、ProgPrompt 进行对比，Tri‑LLM 在 GCR、UC、sUC、CUR、SR 等指标上均优于基线，尤其在 Complex 任务中提升幅度显著。

**⚠️ 局限性**

局限性包括：① 任务类别预定义，难以泛化到更广泛的任务集合；② 依赖固定的技能与感知模型；③ 对动态环境变化与多机器人协同的适应性尚未验证。

---

## 317. Deformable Texture Maps: Compiling Dynamic Microgeometry into Neural Textures

**arXiv ID:** 2611.13677 | [PDF](https://arxiv.org/pdf/2611.13677v1)

**作者:** Antonio Arian Silaghi `[一作]` (University Rey Juan Carlos), Elena Garces `[通讯]` (Adobe)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Deformable Texture Maps (DTM)，将织物在变形下的细微几何结构压缩为神经纹理，能够在粗网格上直接评估并渲染细节。

**💡 创新点**

创新点在于：①将细节几何映射到统一材质空间并通过变形条件的坐标网络生成渲染相关的 alpha、法线、深度和材质 ID 四通道纹理；②实现可重光照、材质编辑且无需重新训练；③持久化纹理缓存与选择性更新显著提升实时渲染性能。

**🔧 技术方法**

使用了：基于 MADYPG 的纱线仿真采样、坐标编码+SIREN（Fourier‑SIREN）网络来学习四通道纹理、以及持久化纹理缓存与 LOD 机制实现高效渲染。

**📊 数据集**

数据集为五种织物结构（rib、basket、stockinette、honeycomb、satin）在多种拉伸/剪切状态下生成的微几何样本，进而渲染得到 alpha、法线、深度和材质 ID 四通道。

**📈 对比分析**

与显式几何渲染、基于图像混合的纹理以及 ASTC 压缩纹理比较；在 PSNR、FLIP、材质 ID 准确率上与 ASTC(8×8)相近，存储压缩 42‑94 倍；实时渲染时静止阶段帧率高于屏幕空间网络评估，运动阶段性能略下降。

**⚠️ 局限性**

局限性：①受训练微结构限制，无法外推至更大变形或不同纱线；②仅重建渲染相关属性，缺乏完整 3D 几何与多射线散射；③不支持弯曲、局部压缩等更复杂变形；④无法捕捉复杂光学效应（透光、多次散射等）。

---

## 318. Which Acoustic Features Survive Compression? Tracker-Estimated First-Formant Statistics for Codec-Robust Audio DeepfakeDetection

**arXiv ID:** 2611.13258 | [PDF](https://arxiv.org/pdf/2611.13258v1)

**作者:** Yash Sukhdeve `[一作]` (Clarkson University), Masudul H. Imtiaz `[通讯]` (Clarkson University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `fede83ac-7505-405f-ab37-e7284695c47f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究在压缩环境下哪些低维声学特征能保持对深度伪造的判别力，并提出仅使用四个F1特征的稳健检测方法

**💡 创新点**

通过效果量筛选和编码器鲁棒性评估，发现所有有效特征集中于首音高F1，提供可解释且压缩稳定的诊断信号

**🔧 技术方法**

使用Praat Burg追踪、Rank‑Biserial效应量、PCA冗余分析、RBF‑SVM及混合效应模型等统计与机器学习技术

**📊 数据集**

训练使用ASVspoof 2019 LA（清晰数据），评估在ASVspoof 2021 LA的七种编解码器条件下，以及In‑the‑Wild公开深度伪造数据集

**📈 对比分析**

与完整36特征SVM和神经网络AASIST对比，F1四特征SVM在压缩失真下EER仅波动3.4pp（22.9%→25.0%），相比全特征模型波动14pp，表明更高的压缩稳健性

**⚠️ 局限性**

主要限制是检测准确率仍远高于神经网络（约20%+ EER），仅适合作为诊断或补充工具，且在狭带电话通道下性能衰退明显

---

## 319. Dependable Long-Horizon LLM Agents: A Trajectory-Centered Empirical Study of Where Errors Enter and Which Controls Reach Them

**arXiv ID:** 2611.13254 | [PDF](https://arxiv.org/pdf/2611.13254v1)

**作者:** Shaohuang Wang `[一作]` (Villanova University), Xun Jiao `[通讯]` (Villanova University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文设计了一个面向LLM多代理系统运行轨迹的“事件记录”框架，记录错误从起始、传播、检测到恢复的全过程，并将其转化为可操作的注释协议；随后对公开的Who&When和MAD两大语料库中的失败运行进行标注，分析错误的起点、传播方式、可被检验的证据以及潜在控制措施。

**💡 创新点**

创新点在于：①提出了统一的“事件记录”表述，整合故障、错误、失败三种概念；②用操作阶段(P0–P7)与状态域(R1–R6)双坐标定位错误；③将错误分析与控制家族对应，系统评估各类控制对错误可观测性与可恢复性的贡献；④通过LLM自动化标注实现大规模错误分析，填补现有标签化仅关注症状或责任主体的空白。

**🔧 技术方法**

技术方法包括：基于LLM(Claude)的自动化注释；构建基于阶段-域的轨迹分析框架；采用事件记录E=⟨O,ℐ_E,ℒ,D,C,M,σ,U⟩来编码因果链；统计各类错误起始点、传播动力学(H1–H7)；对比人类标注的“决定性步骤”与记录的起点；对控制家族(TC、POL、IFS、EVS、PROV、TEST、JUDGE、CKPT、COORD、STALL、HUMAN)的覆盖率和有效性。

**📊 数据集**

使用的数据集为：①Who&When公开的184条失败轨迹（含58条手工、126条算法生成的Magnetic‑One与CaptainAgent团队）；②MAD（MetaGPT、ChatDev等）19条失败轨迹；两者均已标准化为单一文本格式。

**📈 对比分析**

比较方法：将记录的错误起点与人类标注的“决定性步骤”进行对齐，计算误差比例；统计每类错误在轨迹中出现的比例、传播距离、是否存在矛盾证据及是否被检测/响应；评估各控制家族的覆盖率。结果显示：错误多数在P1–P2阶段进入，约71%在任务知识状态R2；约68%轨迹中已出现矛盾证据但无人使用；检测率约33%，但几乎无一能消除错误状态；恢复控制CKPT效果低，仅占4%；约45%失败是主机层（harness）错误。

**⚠️ 局限性**

局限性包括：①标注由LLM完成，缺乏人类互评一致性验证；②样本主要来自问答与沙盒编程任务，未覆盖共享或第三方系统的外部影响；③控制效果评估为理论推断，未在实际运行中验证；④对齐判定与人类标注存在偏差，尤其在错误与决定性步骤差异时；⑤未对异常检测/恢复机制进行因果实验，仅基于轨迹推断。

---

## 320. Bursts That Outlast Themselves: Persistent Degradation in Prefix-Cached Agent Serving

**arXiv ID:** 2611.13412 | [PDF](https://arxiv.org/pdf/2611.13412v1)

**作者:** Wen Huang `[一作]` (Microsoft), Jack Liu `[通讯]` (Independent Researcher)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过对LLM代理推理中前缀缓存的动态行为进行实验，发现并描述了在短暂流量激增后仍能持续导致缓存失效、吞吐下降的“持续降级”现象，并提出了暂停已入驻会话和基于KV需求的预算器两种快速恢复及预防方案。

**💡 创新点**

创新点在于首次系统性地揭示了“持续降级”的机理——会话闭环导致的工作集扩展与缓存失效形成反馈循环；提出并验证了对已入驻会话的暂停和KV需求预算控制，可在几分钟内恢复服务质量；在不同模型（Qwen3-8B/32B）和引擎（vLLM、SGLang）上统一实验，展示该问题与解决方案的普适性。

**🔧 技术方法**

使用的技术包括：配对实验设计、vLLM与SGLang的前缀缓存与KV池管理、会话状态（parking）与限流控制、基于KV需求的预算器与AIMD动态并发控制、缓存命中率监控、Little’s Law 等理论分析。

**📊 数据集**

实验数据集包括：①合成代理会话（25轮、起始6K-37K token、随机token）；②公开的SWE-bench mini-代理轨迹（约200条，真实上下文长度与间隔）；③TraceLab编码代理轨迹（用于对比长上下文情形）。

**📈 对比分析**

采用配对跑实验（同一基础流量，单独注入5分钟突发），比较hit ratio、吞吐量与会话数；在恢复实验中，对五种控制策略（暂停+预算器、会话上限+暂停、AIMD+暂停、仅限新会话、无控制）进行10+种seed的对照，结果显示暂停+预算器能在<5分钟内将hit ratio恢复至0.93-0.95并完成所有后续到达；仅限新会话恢复慢或失败。

**⚠️ 局限性**

主要局限：实验仅在单GPU单副本环境下进行，未覆盖多副本、自动扩缩、生产级长上下文等情况；合成会话使用随机token，未模拟真实语义关联；CPU KV offload仅延迟崩溃，未提供根本预防；理论模型为一阶近似，无法完全预测边界；恢复后队列最终清空仍需额外容量或流量削减。

---

## 321. SteerDistill: Quantizing LLM Agents via Closed-Loop Teacher Steering

**arXiv ID:** 2611.13410 | [PDF](https://arxiv.org/pdf/2611.13410v1)

**作者:** Genming Cui `[一作]`, Ting Cao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究低比特量化LLM代理在执行任务时的失败行为，并提出一种闭环教师指导的量化感知蒸馏方法

**💡 创新点**

通过教师在学生失败回合中迭代修正并内部化局部验证的行为修正，使低比特量化代理在不依赖教师推理时接近全精度性能

**🔧 技术方法**

量化感知蒸馏、闭环教师指导、行为修正蒸馏、W4A16/ A16 量化、BF16 等技术

**📊 数据集**

五个 agentic 基准与 τ^2-Bench 数据集

**📈 对比分析**

与最强量化基线对比，平均得分提升 1.67–2.18 分；在 W4A16 下接近 BF16；在 τ^2-Bench 上成功率比 QeRL 高 1.84个百分点，训练 GPU 小时减少 67%，成功轨迹更短

**⚠️ 局限性**

在更低比特量化下仍可能出现性能损失；方法依赖教师迭代反馈，训练成本相对较高；数据集有限，泛化能力尚未充分验证

---

## 322. Just AI Governance and the Forbidden-Boundary Principle

**arXiv ID:** 2611.13007 | [PDF](https://arxiv.org/pdf/2611.13007v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 323. Understanding Students' Rationales for AI-Mediated Cheating: An Instructor Perspective from Online Forums Analysis

**arXiv ID:** 2611.13358 | [PDF](https://arxiv.org/pdf/2611.13358v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 324. RBLA+: Efficient Rank-Specific LoRA Adapter Extraction for Federated Fine-Tuning with Heterogeneous Clients

**arXiv ID:** 2611.12961 | [PDF](https://arxiv.org/pdf/2611.12961v1)

**作者:** Shuaijun Chen `[一作]`, Albert Y. Zomaya `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在联邦学习中不同LoRA rank客户端如何高效聚合并提取适配器，并提出RBLA+方法实现无额外payload的最优重构。

**💡 创新点**

创新点在于：①引入SP+谱提取，使用compact QR/core‑SVD避免构造密集更新；②支持客户端与服务器的双向尺度转换；③实现对不同rank客户端的无损最优Frobenius误差恢复，且不增加通信负载。

**🔧 技术方法**

技术：LoRA低秩适配器、支持-aware聚合、QR分解+核心SVD、尺度一致化、CPU性能基准、实验平台为NVIDIA A10G GPU和AMD Ryzen 9 9950X CPU。

**📊 数据集**

数据集：MNIST、FMNIST、KMNIST、QMNIST、CIFAR‑10、CINIC‑10、SVHN、AG News、IMDB、MNLI、QQP、RTE、SST‑2（GLUE子任务）。

**📈 对比分析**

对比方法包括RBLA、FlexLoRA、Zero Padding、Replication Padding、FLoRA、FLoRIST；RBLA+在大多数任务上提升准确率（相较RBLA 1–5pp），保持或降低通信payload；服务器提取时间仅比RBLA提升2–10ms，CPU占用显著低于dense SVD或stack‑SVD。

**⚠️ 局限性**

局限：对某些任务的预测精度提升不一定显著；需要额外的QR/SVD计算，虽然开销小但仍是服务器负担；实验基于仿真联邦环境，未在真实移动设备上验证；对LoRA缩放系数的假设需要统一设置。

---

## 325. Fastener-centric acquisition and AND/OR graph representation of state-dependent disassembly knowledge from CAD-derived virtual assemblies

**arXiv ID:** 2611.13379 | [PDF](https://arxiv.org/pdf/2611.13379v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 326. X Marks the Spot: Diffusion for Image Geolocalization with Location Representations

**arXiv ID:** 2611.13180 | [PDF](https://arxiv.org/pdf/2611.13180v1)

**作者:** Brian Wei `[一作]` (Washington University in St. Louis), Nathan Jacobs `[通讯]` (Washington University in St. Louis)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个在预训练位置编码空间内进行图像定位的潜在扩散模型（Diffusion for Image Geolocalization），并通过 Location Autoencoder（LAE）将生成的嵌入映射回坐标，从而实现连续空间的地理定位。

**💡 创新点**

创新点在于：①利用预训练位置编码器（GeoCLIP、SatCLIP、TTE）提供的语义+空间结构作为扩散空间，突破了仅基于坐标的扩散方法；②引入 LAE，将位置嵌入逆映射到地理坐标，实现千米级精度；③通过自监督对齐（REPA）和多模态条件编码提升定位准确率；④生成模型优于传统检索与判别方法，并可作为检索-再排序的先验。

**🔧 技术方法**

技术主要包括：预训练位置编码器、残差 MLP 结构的潜在扩散网络、直接预测干净嵌入的 denoising 目标、表示对齐（REPA）、自适应层归一化（AdaLN）、多模态条件编码（StreetCLIP+DINOv3）以及算术微分求解的解码器精细化。

**📊 数据集**

使用的数据集包括 OSV‑5M（4.9M 训练图像，210K 测试图像）、Im2GPS3k（2.997 张图像）、YFCC4k（4K 图像）以及其训练来源 MP16‑Pro（4.1M 图像）和更大规模的 YFCC‑44M（44M 图像）。

**📈 对比分析**

与现有的判别式和生成式方法相比，在 OSV‑5M、Im2GPS3k 和 YFCC4k 上均取得了显著提升；在 OSV‑5M 上，city 级别准确率提升约 11–15 点，均值误差降至约 0.4 km；在 Im2GPS3k 和 YFCC4k 上，在 1 km 及更粗尺度上均优于 RFM、LocDiff 等前沿方法；并且其样本可进一步提升检索‑再排序系统。

**⚠️ 局限性**

局限性包括：①对 GeoCLIP 的解码器需要数十亿样本才能实现 1 km 级精度，虽可通过解析精细化提升，但仍受限于位置编码器的空间分辨率；②在小尺度（<1 km）定位任务上性能尚未达到预期；③对卫星视图对齐的 SatCLIP、TTE 在城市级精度上逊色；④模型对稀缺地区数据可能表现不佳，导致空间误差不均匀。

---

## 327. Code-Grounded Agentic Explanations for Static Analysis Vulnerability Findings

**arXiv ID:** 2611.13041 | [PDF](https://arxiv.org/pdf/2611.13041v1)

**作者:** Oshando Johnson `[一作]` (Fraunhofer IEM), Eric Bodden `[通讯]` (Fraunhofer IEM)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种基于 LLM 的安全漏洞解释工具 SAFE，通过在 IDE 中展示静态分析器产生的漏洞报告并让语言模型根据代码库和规则生成可追溯、结构化的解释；

**💡 创新点**

创新点在于：①保持检测由传统可审计的 SAST 工具完成，只让 LLM 做解释；②采用 agentic 后端，模型在回答前可查询规则定义、代码段和 taint 路径，保证解释与实际代码和规则一致；③在插件 UI 上实现按需展开的分层解释、与数据流直接关联的可视化以及代码注解，提升可读性与可操作性；

**🔧 技术方法**

技术包括：Java/IntelliJ IDEA 插件、Python+FastAPI + LangChain/LangGraph 的 agentic 系统、SARIF 解析、规则与代码检索工具（规则查找、文件索引、代码读取、全文搜索等），以及结构化输出模式。

**📊 数据集**

数据集为内部 Java Web 项目 Health-Doc-Exchange，利用 Semgrep 产生 10 条真/假正漏洞（覆盖 5 类常见弱点），并在 2 组 5 条漏洞的 4 种排列中进行实验。

**📈 对比分析**

通过 24 名参与者的 AB/BA 交叉实验比较旧版 SAFE-Blue（单提示、受众切换）与新版 SAFE-Red（分层、agentic、集成化）。结果显示：可用性得分提升（77.3→85.0，p=0.011），行动性得分最大提升（+0.67，p≈0.057），修复建议正确率提升约 2.6 倍；但正确判定、平均决策时间与时间效率均无显著差异。

**⚠️ 局限性**

局限性包括：实验样本规模与任务时长受限，未对单个设计要素进行因子实验，故无法分离 UI、分层展示和 agentic 后端对效果的具体贡献；生成延迟未在实验中体现；数据集为私有项目，缺乏公开可复现性；以及信任度未因 grounding 提升，表明 grounding 主要提升理解而非信任。

---

## 328. The Shape of a Program: Path Signatures for Trace-to-Program Induction

**arXiv ID:** 2611.13005 | [PDF](https://arxiv.org/pdf/2611.13005v1)

**作者:** Mohamed Ghanem `[一作]` (CISPA Helmholtz Center for Information Security), Bernd Finkbeiner `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于路径签名（log‑signature）对执行轨迹进行编码的程序诱导方法，并将其与 LSTM、Transformer 和 Fourier 编码器结合使用，构建了端到端的编码-解码器框架，用于从多条执行轨迹中合成满足这些轨迹的程序。

**💡 创新点**

创新点在于将粗糙路径理论的路径签名作为轨迹特征，利用其对几何形状的完整、长度无关且可加性的编码优势；同时发现该编码与解码管道天然具有收缩性，能够在无外部监督的情况下自动压缩冗余训练标签；并展示了该方法能产生更易解释的潜在几何结构。

**🔧 技术方法**

主要技术包括：路径签名（log‑signature）变换、单位步长归一化与位置编码、线性投影、Transformer 解码器、交叉熵训练、Adam 优化、梯度裁剪以及基于 t‑SNE 的潜在空间可视化。

**📊 数据集**

使用了三种程序诱导基准数据集：Karel（二维网格世界）、List（符号列表操作）和 RASP（序列到序列的 Transformer 语义程序），并为评估压缩效果构造了冗余注入版本 Karel⁺、List⁺、RASP⁺。

**📈 对比分析**

通过与 LSTM、Transformer 和 Fourier 三种传统轨迹编码器进行对比，采用语义准确率（semantic accuracy）作为评测指标；实验显示 log‑signature 编码在低数据量时几乎能把准确率提升三倍，在大多数设置下领先；在冗余注入数据集上，利用收缩性实现的自训练能显著减少训练程序的 token 数量。

**⚠️ 局限性**

局限性主要是截断的 log‑signature 维度随截断阶数快速指数增长，限制了对极长或高度交互的程序轨迹的处理；对非常长的执行路径或需要高阶交互的复杂程序仍是挑战。

---

## 329. The Optimal Constant for One-Shot Channel Simulation and Strong Functional Representation Lemma

**arXiv ID:** 2611.13397 | [PDF](https://arxiv.org/pdf/2611.13397v1)

**作者:** Cheuk Ting Li `[一作]` `[通讯]` (Chinese University of Hong Kong), Cheuk Ting Li (Chinese University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过引入指数型功能信息和泊松功能表示，证明了在单次通道模拟中条件熵的三阶紧致上界，即 H^* ≤ I + log I + (1-γ)log e – log log e + O(I⁻¹log I)，并给出了可实现的 Poisson 方案和有限块长度下的冗余上界。

**💡 创新点**

创新点主要包括：① 提供了最优的三阶常数项（(1-γ)log e – log log e），证明其不可改进；② 引入指数型功能信息作为新度量，使得 SFRL 的上界更紧；③ 证明泊松功能表示几乎是最优的，并给出 log e 的常数间隙；④ 结合指数型功能信息给出了记忆无关通道的有限块长度冗余上界。

**🔧 技术方法**

技术方法包括：泛化功能信息的定义与性质推导；泊松点过程与 Mecke 公式的应用；熵与信息密度的高阶展开；Beta/Gamma/几何混合分布的精细分析；对数-指数不等式和极限定理的使用。

**📊 数据集**

由于是理论研究，未使用任何实际数据集；所有结果均为理论证明与推导。

**📈 对比分析**

与以往的上界（如 I+log(I+5.22)+0.76、I+log(I+1)+4 等）相比，新的上界在所有阶级上都更紧，尤其在 I 较大时三阶常数项已达到最优；泊松方案的实现仅差 log e 的常数，已逼近最优；有限块长度结果实现了 (1/2)log n 的二阶项，并给出了明确的常数项，比之前的粗略 O(√n) 或 (1/2+o(1))log n 更精确。

**⚠️ 局限性**

局限性包括：① 结果仅适用于无限共同随机性的变量长度编码；② 证明依赖泊松点过程的特性，尚未给出其他可实现构造的通用算法；③ 常数项的最优性仅针对仅以 I 为参数的上界，若引入更多信息量度，仍可能存在改进空间；④ 对实际系统的实现复杂度与效率未做评估。

---

## 330. From Relevant Posts to Attributable Spans: Explicit Symptom Evidence for Depression Questionnaire Inference

**arXiv ID:** 2611.12900 | [PDF](https://arxiv.org/pdf/2611.12900v1)

**作者:** Sumanth Manduru `[一作]` (George Mason University), Carlotta Domeniconi `[通讯]` (George Mason University)

**通讯引用:** 6987 | [OpenAlex ID](https://openalex.org/A5090148962)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SREE框架，利用LLM提取第一人称症状子跨度、采用设施位置子集化筛选代表性证据，并通过证据门控推理实现BDI-II问卷的个体化推断。

**💡 创新点**

从语义检索转向可归因的症状子跨度定位，结合子集化选择代表性证据和门控机制，显著提升症状级别推断的可靠性与解释性。

**🔧 技术方法**

零样本LLM提取、句子嵌入与设施位置子模函数、门控推理与Ordinal预测等技术。

**📊 数据集**

eRisk 2019/2020/2021 Reddit历史数据及其对应的BDI‑II问卷答案，及BDI‑Sen句子级别症状标注。

**📈 对比分析**

与固定k检索、aRAG、两跳检索等基线在AHR、ACR、ADODL指标上对比，SREE在AHR上提升约2–3个百分点，且在ACR、ADODL上保持竞争力。

**⚠️ 局限性**

仅评估已检索到证据的项目，缺乏完整问卷恢复；数据来源有限，主要是少量Reddit用户，可能不具备跨平台或跨语言的泛化能力。

---

## 331. TIB: Sample-wise Tempered Information Bottleneck for Multimodal Attribution beyond Alignment Assumption

**arXiv ID:** 2611.12997 | [PDF](https://arxiv.org/pdf/2611.12997v1)

**作者:** Qiming Huang `[一作]` (University of Birmingham), Jianbo Jiao `[通讯]` (University of Birmingham)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种针对多模态归因的温度化信息瓶颈（TIB）框架，用以在图像-文本对存在语义不对齐时自动调节归因强度，避免传统信息瓶颈方法因误匹配而导致的过度拟合和不稳定归因。

**💡 创新点**

创新点在于：① 将跨模态不可靠性建模为样本级预测方差；② 在信息瓶颈目标中引入可调温度，形成异方差化的损失，使模型在不对齐样本上弱化拟合；③ 通过实验验证该方法在合成和真实大规模数据集上显著提升归因鲁棒性。

**🔧 技术方法**

技术包括：信息瓶颈理论、异方差高斯预测模型、可学习的预测方差网络、基于掩码的多模态信息瓶颈、梯度归一化（α-NLL）以及自监督的可变权重优化。

**📊 数据集**

使用的数据集包括：合成的Flickr8k与RefCOCO对齐/误对齐样本；大规模公开图像-文本数据集Conceptual Captions 3M、Flickr8k、LAION-400M，用于评估真实场景下的归因性能。

**📈 对比分析**

与现有方法（Grad‑CAM、RISE、M2IB、Fast‑IG、Chefer、Grad‑ECLIP、FIxLIP、NIB）以及CLIP/COCOA等在多种评价指标（Drop/Incr、Grounding、ROAR、IoU）上对比，TIB在合成与真实数据集上均取得最高或相近最高的IoU和改进的Drop/Incr得分，显示出更稳健的归因质量。

**⚠️ 局限性**

局限性包括：① 需要训练额外的预测方差网络，增加模型复杂度；② 对超参数β与α的选择仍有一定敏感性；③ 在极端误匹配比例下，预测方差可能无法充分区分有效与无效样本；④ 目前仅针对视觉‑文本场景，尚未验证至其他多模态组合（如音频‑文本）。

---

## 332. ExpertCrowd-SWE: Measuring Frontier Coding Agents on Expert-Authored Engineering Tasks in Unfamiliar Codebases

**arXiv ID:** 2611.13246 | [PDF](https://arxiv.org/pdf/2611.13246v1)

**作者:** Josip Vrdoljak `[一作]` (Toptal ExpertCrowd), Sam Pigott `[通讯]` (Toptal ExpertCrowd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并发布了 ExpertCrowd‑SWE (EC‑SWE)，一种由行业工程师手工编写、包含加权分数和门控的多文件软件工程任务基准，专门评估自主编码代理在陌生代码库中的实战表现。

**💡 创新点**

创新点在于：①任务完全为专家原创、未公开，避免训练数据污染；②采用部分信用、阻塞器和门控的评分体系，模拟真实维护者的 PR 审核；③任务环境与评测均通过容器化、校准与老练工程师审查保证可重复、可信；④在同一平台上使用各模型原生工具接口，消除工具差异的偏倚。

**🔧 技术方法**

技术手段包括：HUD 平台的容器化运行环境、SDK‑驱动的原生工具调用、分步评测与完整执行轨迹记录、加权评价器与门控逻辑、以及 LLM 辅助的评测与控制补丁。

**📊 数据集**

数据集为 100 条专家编写的任务，覆盖 87 个开源项目、11 种语言，包含 24 条公开任务与 76 条隐藏任务；每条任务配有环境镜像、参考解法与评分器。

**📈 对比分析**

比较方法：在同一平台上对七个前沿模型（Claude Opus 5.5、GPT‑6 Astra 等）各执行 5 次评测，计算宏平均得分并给出 95% 置信区间；结果显示模型间得分差异约 35 分，前两名高度接近但表现互补，模型间仍存在 5‑10 分的显著差距。

**⚠️ 局限性**

局限性包括：①评测平台使用各模型原生工具接口，可能引入工具设计偏差；②未对推理力度做系统调优；③任务样本仅 100 条，任务采样不确定性仍显著；④代码库本身公开，可能已在预训练数据中；⑥评测器与 LLM 判断的误差尚未完全量化。

---

## 333. Counting Under Disagreement: Dollar-Weighted Measurement, Selective Machine Labelling and Budgeted Human Review from Disagreeing AI Readers of Administrative Text

**arXiv ID:** 2611.13215 | [PDF](https://arxiv.org/pdf/2611.13215v1)

**作者:** George Koomullil `[一作]` `[通讯]`, George Koomullil

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在行政记录中引入机器阅读与人类抽样，对共享文本单元进行有效计量，提出并评估了预测辅助推断、加权合规选择和预算人类审核的多种方法。

**💡 创新点**

将预测辅助推断与设计基抽样结合，提出跨折叠预测器的无偏估计；设计加权 conformal p 值与 Learn‑then‑Test 规则实现美元加权错误控制；并引入活跃推断策略与混合抽样设计以优化有限人类预算。

**🔧 技术方法**

采用交叉拟合多项式逻辑回归、Dawid–Skene 聚类、加权 conformal inference、Benjamini–Hochberg 步进、Hoeffding–Bentkus p 值、Poisson/固定大小抽样理论及自助法模拟评估。

**📊 数据集**

使用 2020–2024 年美国私人基金会和慈善机构在 Form 990‑PF 与 Schedule I 报税中的拨款目的文本，共 651 个唯一文本，总计数千亿美元。

**📈 对比分析**

与经典估计、规则分类器、单一 LLM 预测比较，校准后估计在总额上收敛至经典区间；加权合规选择在 0.10 水平下可接受约 10% 美元错误；主动抽样在 100 条文本预算下 RMSE 降低约 30%，但区间覆盖率低于 95%；混合设计和成本调整可进一步提升。

**⚠️ 局限性**

在已对机器投票分层 PPS 抽样的环境下，预测器对方差提升有限；加权合规选择和 Learn‑then‑Test 仅得到经验验证，缺乏理论覆盖率保证；主动设计在小样本下覆盖率不足；结果高度依赖设计假设与特定数据集，推广性有限。

---

## 334. FaST: Factored Synthesis of Transitions for Offline Reinforcement Learning

**arXiv ID:** 2611.13013 | [PDF](https://arxiv.org/pdf/2611.13013v1)

**作者:** Daniel Campa `[一作]` (West Virginia University), Srinjoy Das `[通讯]` (West Virginia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了FaST（Factored Synthesis of Transitions）生成器，用链式规则将状态、动作、动力学、奖励和终止等五个阶段分别建模，实现离线强化学习的数据增强；

**💡 创新点**

创新点在于将转移分布分解为独立因素，针对每个因素选用匹配的统计模型，并通过显式的(A|S)条件实现可控的“最佳动作”采样（best‑of‑N），大幅提升生成速度和数据质量；

**🔧 技术方法**

技术上采用自回归混合密度网络（AR‑MDN）建模状态、对数logit变换后的动作混合密度网络、残差回归（delta ensemble）建模动力学、MLP回归奖励、ExtraTrees分类器预测终止，并利用IQL优势函数进行动作筛选；

**📊 数据集**

实验使用十二个D4RL基准，包括MuJoCo的HalfCheetah、Hopper、Walker2d（中等、回放、专家级别）以及Maze2d（umae、medium、large）等；

**📈 对比分析**

与真实数据、SynthER、过滤式目标和best‑of‑10等方法对比，FaST在不降低性能的前提下生成速度提升90倍；best‑of‑10在大多数中等/回放环境下甚至优于真实数据，且在目标强化方面优于滤波方法；

**⚠️ 局限性**

局限性包括：先行采样可能导致误差累积；在专家混合数据集上优势目标失效；缺乏多步一致性，无法处理需长轨迹的任务；仅适用于低维连续状态，无法直接扩展到图像或高维观测；

---

## 335. From Egocentric Human Video to Robot Manipulation in the Era of Embodied Intelligence

**arXiv ID:** 2611.13610 | [PDF](https://arxiv.org/pdf/2611.13610v1)

**作者:** Yi Wang `[一作]` (Hong Kong Polytechnic University), Lap-Pui Chau `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

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

## 336. No finite rulebook can certify every safe program: an impossibility theorem for gatekeeping AI agents

**arXiv ID:** 2611.13367 | [PDF](https://arxiv.org/pdf/2611.13367v1)

**作者:** Tomas Perez-Acle `[一作]` `[通讯]` (Universidad San Sebastián), Tomas Perez-Acle (Universidad San Sebastián)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在可观测行为规范（安全、保密、响应）下，是否存在可验证且完整的双侧规则书（允许列表与禁止列表），并证明了这种完整性在一般可计算程序空间中不可实现；进一步探讨了预算约束下的可判定性、可证实集合的结构，以及通过分离器（separator）实现的门控机制。

**💡 创新点**

核心创新在于：① 用可验证证据定义可证实集并给出其确切的算术层次；② 证明了安全性可证实但不可完全，保密与响应两侧均不可证实；③ 构造自指计划显示任何可验证规则书必遗漏与其规模相当的合规程序；④ 引入“分离器”概念，说明在可验证集合之间仍可实现必要的门控；⑤ 在预算有限的情况下证明任何永远正确的检查器需近乎完整地执行程序，凸显资源成本。

**🔧 技术方法**

主要技术包括可接受程序系统的自指性、Rice 与 Rice–Shapiro 定理、停机问题的归约、时间层级定理、可观察行为模型（可观测事件序列）与可证实集合的有效开集合理论。

**📊 数据集**

使用随机寄存器机程序数据集：对长度为 4、6、…、24 的 200,000 个随机程序（共 2.2 M 程序）进行实验；另外实验了 483,641 个 1–4 指令的完整程序。

**📈 对比分析**

实验对比了有界测试（时间窗）与无限测试，展示了：在有限时间窗下的安全测试错误率随程序长度增加而上升，但在常规模型中误判稀少；保密测试误判率随时间窗增大趋于常数；自指计划与预算计划证明了永远正确的检查器需要消耗与预算相近的时间，远超简单静态分析。

**⚠️ 局限性**

局限性包括：① 只考虑无限计算资源与无穷长程序；② 观察模型仅涵盖可见事件，未覆盖隐式或内部状态；③ 规范仅为单一二分区，未处理冲突或不确定性；④ 结果基于理论归约，实际部署需先验证三问（可计算性、可观测性、可验证证据）是否满足；⑤ 计算机实验仅针对特定指令集与分布，未给出整体密度或平均性能分析。

---

## 337. PegasusAI Studio: Legible Cyberinfrastructure and Accountable AI Agents for Scientific Workflows

**arXiv ID:** 2611.13309 | [PDF](https://arxiv.org/pdf/2611.13309v1)

**作者:** Komal Thareja `[一作]` (University of North Carolina at Chapel Hill), Ewa Deelman `[通讯]` (University of Southern California)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文设计并实现了PegasusAI Studio——一个基于Pegasus与HTCondor工作流管理系统、JupyterHub或Open OnDemand门户的AI辅助科学工作流平台，能够在共享计算基础设施上实现工作流编排、监控、调试与重跑，并通过知识投影、结构化状态输出和信任封装实现代理与设施的无缝协作。

**💡 创新点**

创新点主要体现在：① 将系统与站点知识统一存储并投影到多种AI代理的本地配置，实现跨代理的知识共享；② 通过将Pegasus与调度器的工作流和运行状态转换为结构化事件流，使代理能基于完整的运行信息进行诊断与修复；③ 通过身份、授权、审批和可观测性等“信任包”，让AI代理在用户的身份与权限下安全执行，所有操作均可审计；④ 采用可插拔的多部署模型（JupyterHub+K8s与Open OnDemand）以及多代理/模型堆栈，证明方案的通用性。

**🔧 技术方法**

技术手段包括Pegasus WMS、HTCondor、CILogon+Authentik+JupyterHub、Open OnDemand、Docker/Apptainer、Python+MCP、OpenTelemetry、插件化监控服务、抽象语法树解析、结构化日志/事件流、代理安全策略与审批机制等。

**📊 数据集**

案例研究使用了多种科学数据集：CASA气象雷达数据、GWTC-1引力波数据、OpenNeuro ds007096 EEG数据，以及Event Horizon Telescope (EHT) M87 重建数据。

**📈 对比分析**

评估方法主要是观察性案例研究，验证系统机制在真实工作流中的表现；未给出量化的性能对比。通过案例展示：调试循环从失败到零失败、工作流按用户身份提交并通过审批、不同代理与模型堆栈均能正常工作。

**⚠️ 局限性**

主要局限包括：仅在作者团队与已熟悉Pegasus的用户中使用，缺乏大规模或多机构的用户量化评估；案例选择非随机，代表性不高；技术实现目前仅支持Pegasus/HTCondor与JupyterHub/OpenOnDemand，未扩展到其他工作流/调度系统；单用户容器与SQLite存储、模型凭证本地化限制了可扩展性；代理与模型版本随时间漂移，需要持续维护。

---

## 338. From Questions to Decisions: Benchmarking How LLM Agents Acquire Decision-Relevant Information from Human Beings

**arXiv ID:** 2611.13015 | [PDF](https://arxiv.org/pdf/2611.13015v1)

**作者:** Yuhan Tang `[一作]` (Massachusetts Institute of Technology), Yuan Zhou `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个名为PlanAsk的基准，用于评估大型语言模型在规划任务中向操作员询问缺失关键信息的能力

**💡 创新点**

创新点在于：①为每个隐藏事实计算并标注“遗憾”值（决策价值），并配对相似但无影响的双胞胎控制；②通过收集79次人类操作员交互会话构建与人类行为高度匹配的仿真操作员；③将首计划与最终计划分开评分，凸显信息获取瓶颈

**🔧 技术方法**

使用的大型语言模型包括 GPT‑6.1 Sol、Gemini 3.8 Flash、DeepSeek V4.1 Flash 与 GLM‑5.3 Flash；同时使用Gurobi求解器来评估计划质量；仿真操作员通过两步LLM调用生成回复，采样人类回复率并控制长度与风格

**📊 数据集**

数据集由43个规划任务组成，涵盖路由、排程、调度等领域；任务基于公开优化实例（如Solomon、MIPLIB、FrontierOR）并手工添加隐藏事实，包含约235个隐藏项、131个双胞胎，最大条件 regret 达0.836

**📈 对比分析**

对比方法：在交互条件下让模型先询问再提交首计划；在全信息条件下直接给出完整隐藏事实；同时对比两种操作员（人类、仿真）并评估最终计划。结果显示：在交互下首计划失败率高达58–98%，所有信息下平均首计划误差仅0.05–0.28；模型仅比双胞胎多询问5–12%关键信息；反馈后计划损失下降至0.42–0.62，但仍受信息获取质量影响

**⚠️ 局限性**

局限性包括：①仅关注单个任务阶段（首计划）对信息获取评价，未考虑长期多轮交互策略；②仿真操作员虽贴近人类，但仍无法完全复制人类细微决策与情境适应；③基准任务规模受算例限制，未覆盖极大规模真实生产环境；④只评估四种LLM，可能不代表未来模型的全部能力

---

## 339. PhysMoE: Fast Spatial Grounding with Single-Token Coordinate Representation in Modality-Decoupled VLMs

**arXiv ID:** 2611.13000 | [PDF](https://arxiv.org/pdf/2611.13000v1)

**作者:** Ruimeng Liu `[一作]` (Nanyang Technological University), Lihua Xie `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种可直接输出连续像素坐标的视觉‑语言模型，并通过半 Mixture-of-Experts（PhysMoE）架构将空间回归与文本生成解耦。

**💡 创新点**

创新点在于：①将每个二维坐标作为单一 token 并用视觉编码器的已学习位置嵌入直接编码；②在上层解码层使用 Semi‑MoE 将 FFN 分为语言专家和空间专家，消除梯度干扰；③用可微分的 Mask‑SDF（ESDF）监督代替单点回归，解决多峰接触点与不规则形状的困扰。

**🔧 技术方法**

主要技术包括：视觉‑语言 Transformer、位置嵌入插值、Semi‑MoE 路由器、双头输出（LM 与坐标回归）、Mask‑SDF 监督损失、两阶段 LoRA 训练策略。

**📊 数据集**

训练数据约 1200 万样本，来源于视觉‑语言通用任务（Visual7W、Visual Genome 等）与实体抓取/路径规划数据集（RoboAfford、VABench‑Point、ShareRobot‑Trajectory 等），并通过 SAM 自动生成掩码以构造 ESDF。

**📈 对比分析**

在 20 项基准（9 个坐标基准 + 11 个文本推理基准）上，4B 版本平均提升 10.5 分，9B 版本平均提升 15.7 分；在坐标基准上比同规模基线快 6×–8×，在 9B 级别已超过 30B 的 HY‑Embodied‑VLM；文本推理性能保持与基线相当。

**⚠️ 局限性**

局限性包括：主要聚焦 2D 像素坐标，尚未验证对 3D 目标或深度信息的适用性；对复杂遮挡、动态环境的鲁棒性未深入评估；模型仍需大量计算资源，特别是 9B 版本。

---

## 340. 5G CATS: 5G-Based Cryptographic Authorization Against Trafficking of Semiconductor Devices

**arXiv ID:** 2611.13437 | [PDF](https://arxiv.org/pdf/2611.13437v1)

**作者:** Sreevikram Rajaraman `[一作]` (Texas A&M University), Jeyavijayan Rajendran `[通讯]` (Texas A&M University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于 5G URLLC 的芯片位置认证系统 5G CATS，利用与 GPU 同包装的 5G 天线、加密引擎和硬件互锁，在芯片层面完成加密挑战-响应、时延测量与位置绑定，实现对出口受控 AI 加速器的实时位置信息验证。

**💡 创新点**

创新点：
1) 将 5G URLLC 直接集成到芯片包装中，消除主机软件、PCIe 与 BMC 的干扰；
2) 用“执行租约”取代永久密钥抹除，避免误判导致合法芯片被禁用；
3) 采用载波绑定校验（cell ID 与运营商记录对比），抵御伪造基站攻击；
4) 在 5G 链路层使用 99.9th 分位阈值与短周期随机挑战，显著缩小时延尾部，提升对光纤中继的抵抗力。

**🔧 技术方法**

技术：
- 5G URLLC 配置（μ=2、0.25 ms 载波、1.25 ms 上行、配置授予、无 HARQ 重传）
- AES‑GCM、X25519（或 ML‑KEM）密钥交换与签名
- 计时前置与时延测量（T_wait、T_SR 等）
- 执行租约（lease）与序列化测试
- Carrier‑attested cell binding（通过 NEF/AMF 记录）

**📊 数据集**

数据集：
- ns‑3 / 5G‑LENA 仿真，20 个 1 000 轮 recertification 的无负载单机单元；
- 与 PowerCATS 发布模型及默认 eMBB 参数进行对比。
- 未使用真实硬件或现场数据。

**📈 对比分析**

对比方法：在同一仿真环境下，比较默认 eMBB、5G CATS 以及 PowerCATS 的 99.9th 分位时延、隐藏中继距离、可安全距离以及误拒率。性能表现：
- 5G CATS 99.9th 分位时延 5.72 ms（默认 eMBB 为 56.40 ms），时延尾部仅 2.60 ms；
- 隐藏光纤中继距离从 300 km（5G CATS）降至 2 000 km（默认 eMBB）；
- 可安全距离 1 410 km（5G CATS）对比 291–619 km（PowerCATS）；
- 误拒绝率大幅下降，执行租约避免了永久锁定。
- 但在光纤中继 50 km 时仍有 0.95% 的误拒，需进一步优化阈值。

**⚠️ 局限性**

局限性：
- 仿真未实现配置授予 uplink 与子槽 mini‑slot，导致测得时延略高；
- 仅模拟单机单单元，无载波、非视距环境评估缺失；
- 未实际验证 carrier‑attested cell binding 的实时性与可靠性；
- 设计假设每个基站可按需保留 CATS 流资源，实际大规模部署时是否满足资源分配需验证；
- 仅在私有 URLLC 细分中评估，公共网络环境尚未验证。

---

## 341. RadRef: A Reference-Free Metric for Radiology Report Generation

**arXiv ID:** 2611.13168 | [PDF](https://arxiv.org/pdf/2611.13168v1)

**作者:** Shelly Schwartz `[一作]` (University of Virginia), Chirag Agarwal `[通讯]` (University of Virginia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

我们提出了一种无参考的评估指标，利用图像推断得到的临床发现词表及其与临床指征相关性的加权，对AI生成的放射科报告进行打分。

**💡 创新点**

该指标不依赖参考报告，直接将报告主张与图像导出的信念进行对比，并根据指征相关性加权，从而实现缺少参考时的质量评估。

**🔧 技术方法**

采用图像-语言模型生成图像导出的发现信念，构建词表权重，并使用Kendall τ_b等统计方法评估指标与放射科医生排序的相关性。

**📊 数据集**

在ReXVal数据集上评估指标，并在100份MIMIC‑CXR研究的模型生成与放射科医生撰写的报告中进行读者研究。

**📈 对比分析**

与领先的基于参考的指标相比，ReXVal上该指标达73.1%的同辐射图像报告排序一致率，在遗漏检测和错误发现上表现相当，在读者研究中对遗漏的Kendall τ_b为0.295，接近读者间一致性0.386。

**⚠️ 局限性**

主要局限在于指标对图像、报告及指征的依赖性需要进一步验证，词表或权重选择不当可能导致评分偏差，并且缺乏在多中心真实临床环境下的广泛验证。

---

## 342. IIT Patna at "ICPR 2026 TVRID Competition on Privacy-Preserving Person Re-Identification from Top-View RGB-Depth Camera"

**arXiv ID:** 2611.12960 | [PDF](https://arxiv.org/pdf/2611.12960v1)

**作者:** Abhinav Kumar `[一作]`, Chandranath Adak `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文介绍了在ICPR 2026 TVRID比赛中，基于顶部视角摄像头的隐私保护人脸重识别（Re-ID）方法，使用RGB和深度数据进行评估。

**💡 创新点**

创新点在于使用深度传感器和JET深度着色技术来减少颜色信息，从而提高隐私保护，同时展示了两种不同的方法在比赛中的优异表现。

**🔧 技术方法**

使用了OSNet和ResNeSt-50d、ResNet-50作为卷积神经网络（CNN）骨干网络，结合了深度着色和特征聚合技术。

**📊 数据集**

使用了TVRID数据集，该数据集包含86个身份，由四个同步的Intel RealSense D455摄像头在不同视角下捕获的RGB和深度数据。

**📈 对比分析**

与比赛中的其他方法相比，方法1在深度轨道上取得了93.51%的mAP和90.42%的CMC@1，方法2在RGB轨道上达到了99.63%的mAP，均显著超出深度轨道的平均水平57.19%。

**⚠️ 局限性**

限制在于缺乏控制性消融实验，无法将性能提升归因于单个组件，同时训练设置的差异也影响了结果的归因。

---

## 343. Project Progress Desk: Know Where All Your Claude Code and Codex Projects Stand at a Glance

**arXiv ID:** 2611.13287 | [PDF](https://arxiv.org/pdf/2611.13287v1)

**作者:** Wanghao Ye `[一作]` (University of Maryland), Ang Li `[通讯]` (University of Maryland)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了Project Progress Desk，一款本地桌面界面，整合编码代理（Claude Code、Codex）在多项目并行工作中的进度、阻塞点与待决策，并通过“receipt”机制实时更新项目完成度。

**💡 创新点**

创新点包括：1）将代理产生的结果与统一的项目接受检查表绑定，区分已确认的进度与仅估计的部分；2）在单一视图中展示多项目进度、阻塞状态与用户待决策；3）提供项目历史分支视图，让用户对比不同实现路径的成效。

**🔧 技术方法**

使用了自然语言处理与代码生成代理（Claude Code、Codex）产生的收据、Windows桌面应用框架、JSON数据结构与图形化进度条显示；同时集成了文件检索与聊天窗口链接。

**📊 数据集**

主要数据来自实验室内约36个正在进行的工程项目，演示阶段使用30个模拟项目；项目进度数据来自代理收据、检查表权重与估计值。

**📈 对比分析**

文章未给出量化比较结果，而是在讨论中提到计划与传统项目表格进行受控对照实验，以评估用户识别阻塞项目和选择下一任务的效率。

**⚠️ 局限性**

局限性包括：1）依赖代理收据的准确性，估计值可能不可靠；2）目前仅在内部实验室环境测试，缺乏大规模用户验证；3）对多供应商账号切换仍需手动管理，系统尚未完全自动化。

---

## 344. Reliability Analysis for Ground-Air Cooperative Wireless-Powered Internet-of-Things

**arXiv ID:** 2611.12952 | [PDF](https://arxiv.org/pdf/2611.12952v1)

**作者:** Xingyu Huang `[一作]` (South-Central Minzu University), Di Bai `[通讯]` (South-Central Minzu University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了基于随机几何的地空协同无线供能物联网(GA‑WPIoT)模型，并利用元分布分析评估设备级能量转移与上行通信的可靠性分布。

**💡 创新点**

创新点在于将能量供给与数据采集解耦、采用环形GPB部署与Matérn硬核过程描述无人机分布，并将元分布方法与能量与通信的耦合成功条件相结合，首次给出联合可靠性的闭式近似。

**🔧 技术方法**

主要技术包括随机几何建模、Matérn硬核点过程、Binomial点过程、Rayleigh衰落、路径损耗模型、拉氏变换、Beta近似元分布等。

**📊 数据集**

由于研究基于理论模型，未使用真实数据集，而是通过Monte‑Carlo仿真验证所推导的矩和元分布，仿真参数选自常见的城市与郊区环境设置。

**📈 对比分析**

与仿真结果对比，解析式的第一、二阶矩和元分布均与仿真保持高度一致，表明模型能准确捕捉不同GPB内部半径与无人机高度对可靠性的影响，并给出了最优部署方案。

**⚠️ 局限性**

限制包括：假设GPB与无人机链接相互独立、使用简化的链路模型（如理想的能量转移、频率分配），未考虑能量供给与上行链路的动态协同控制以及实际硬件与能耗限制。

---

## 345. INTENT: Out-of-Sample, Out-of-Time, and Cold-Start Evaluation on User Event Sequences

**arXiv ID:** 2611.13069 | [PDF](https://arxiv.org/pdf/2611.13069v1)

**作者:** Mattias Kvist `[一作]` (KTH AI Society), Rickard Brüel-Gabrielsson `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了INTENT基准，用于评估用户事件序列中的下一个动作预测，构建了七格测试网格，包含新用户、未来时间段、新物品以及跨平台转移情况，并对公开数据集进行审核与划分；

**💡 创新点**

创新点在于提出统一的交互-新颖性-转移评估框架，使用基于规则的时间切分避免未来信息泄漏，首次将用户、时间与物品三轴同时分层评估，并公开完整的评测代码与结果；

**🔧 技术方法**

采用NDCG@10作为评价指标，评测了基于Transformer的SASRec及其利用物品元数据的变体SASRec‑F、常用的热门项和重复上一次项基线，使用公开的带有日历时间戳的事件序列进行训练与测试；

**📊 数据集**

使用了五个满足条件的公开数据集：Amazon H&K评论、KuaiRand视频曝光、MicroLens多模态视频、H&M顾客画像与价格、REES46购物漏斗，均拥有超过25,000名用户、每人至少三条事件；

**📈 对比分析**

通过在七格网格内对比模型性能，SASRec在已出现物品且时间切点后的已知用户上表现最佳；repeat‑last在新物品预测上居首；SASRec‑F仅在MicroLens的新物品场景中提升显著；整体来看，时间切点导致的性能衰减大于新用户带来的影响；

**⚠️ 局限性**

局限性包括：每个特定条件仅有单一数据集，导致泛化受限；元数据对大多数数据集提升有限；未使用银行真实数据（缺少时间排序）；参考模型未进行超参数调优；缺乏用户特定参数，且部分拆分未通过完整泄漏检测。

---

## 346. SPMW: A Grid-Based Programming Model for Scalable Accelerator Design

**arXiv ID:** 2611.12944 | [PDF](https://arxiv.org/pdf/2611.12944v1)

**作者:** Hongzheng Chen `[一作]` (Cornell University), Zhiru Zhang `[通讯]` (Cornell University)

**通讯引用:** 7873 | [OpenAlex ID](https://openalex.org/A5037210004)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了单程序多连通工作单元（SPMW）编程模型，使加速器的网格结构、连接规则和数据传输成为一等抽象，帮助设计者在保持网格正则性的同时实现高性能、可组合的固定功能与可编程加速器。

**💡 创新点**

创新点在于：①把网格正则性、连接规则和边界绑定显式化为语言层面的构造，保持从前端到RTL的完整正则性；②在编译器中利用此正则性实现工作单元的实例化共享、并行编译和多核仿真；③通过统一的 MLIR 流水线把 HLS 行为与 RTL 结构结合，支持多种加速器族的快速构建。

**🔧 技术方法**

技术包括：Python 前端定义工作单元、网格和连接规则；MLIR dialect 用于保留正则性并生成 HLS C++ 代码；HLS（Vitis）用于行为合成；RTL 生成器拼装 IP、FIFO 连接与边界束缚；并行编译（多核 HLS、RTL 组装）与 OpenMP 多线程功能仿真。

**📊 数据集**

数据集与实验：GEMM（各种尺寸）、FFT（256/1024 点）、mini‑TPU（LLaMA、DeepSeek 前馈层）、FEATHER（128³ GEMM、3×3 卷积）以及 32×32 矩阵‑向量引擎，在 AMD Alveo U280 FPGA 上运行；对比 AutoSA、Allo、HP‑FFT、VTA、Gemmini、Spiral 等基准。

**📈 对比分析**

对比方法：在相同 FPGA 平台、相同频率上比较 RTL 运行时延、资源占用、HLS 生成时间和功能仿真速度；通过功能验证确保结果一致。性能结果：32×32 GEMM 延迟比 AutoSA 低 4.36×、比 Allo 低 2.51×；FFT 延迟比 HP‑FFT 低 2.65×；mini‑TPU 在 16×16 上延迟比 Gemmini 低 14.6%、比 VTA 低 11.6%；功能仿真速度比 Vitis HLS 低 18.4×；编码代理生成的设计令 Token 消耗比 SystemVerilog 低 7.51×。

**⚠️ 局限性**

局限性：仍需完整 RTL 合成与 P&R，后者占据大部分构建时间；功能仿真不具备性能预测与死锁检测；未覆盖 ASIC/AI Engine 目标；对动态数据流或非正则拓扑支持有限；缺乏形式化验证与自动化资源估计。

---

## 347. NIF-VIO: Visual--Inertial Odometry Across Inertial and Non-Inertial Reference Frames

**arXiv ID:** 2611.13662 | [PDF](https://arxiv.org/pdf/2611.13662v1)

**作者:** Zichao Shu `[一作]` (Shanghai Institute of Technical Physics, Chinese Academy of Sciences), Jianyu Wang `[通讯]` (Shanghai Institute of Technical Physics, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出NIF‑VIO框架，在惯性与非惯性环境下实现连续定位；

**💡 创新点**

通过视觉–惯性一致性自适应融合、早期约束隔离与局部重优化，以及携带运动模型的选择性惯性约束，解决传统VIO在非惯性条件下的冲突；

**🔧 技术方法**

基于视觉–惯性一致性检测、基于Student‑t分布的置信度判定、SO(3)对数映射、联合优化（视觉重投影+惯性预积分+携带运动先验）以及局部因子图重优化；

**📊 数据集**

使用欧朗科EuRoC、TUM‑VI公开数据集评估惯性条件下性能，并收集包含汽车、地铁、电梯场景的自制数据集进行非惯性条件下的评估；

**📈 对比分析**

与VINS‑Fusion、OpenVINS、Basalt、ORB‑SLAM3、OKVIS2、SchurVINS、√(VINS)等方法对比，NIF‑VIO在惯性场景下保持最佳/次佳位置误差，在非惯性场景中定位成功率超过98%且条件误差显著低于传统VIO；

**⚠️ 局限性**

仅限于内部环境特征丰富的场景，无法处理外部视觉观测不足的情况，且在极端快速或振动激烈的非惯性运动中仍可能出现误差。

---

## 348. AI Governance Should Prioritize Control and Knowledge Boundaries Over Limiting Intelligence

**arXiv ID:** 2611.12941 | [PDF](https://arxiv.org/pdf/2611.12941v1)

**作者:** Hendrik Luuk `[一作]` `[通讯]` (Pfizer Inc.), Hendrik Luuk (Pfizer Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出“竞争电路”框架，将智能、知识与控制三要素拆解为 IKC + KC + C，形成一条多维路径来解释智能体的现实世界影响；对快速崩溃（fast takeoff）进行形式化，列举六个必须同时满足的条件；利用 NK 适应度景观的随机模拟，验证并量化智能提升的速度乘数上限，并探索控制与知识限制对这一上限的影响。

**💡 创新点**

1) 竞争电路的概念化，将知识与控制作为智能的门槛与直接影响路径；2) 提出了一个可解析的“智能乘数”上限公式，表明无论任务难度如何，智能提升的速度提升被竞争智能水平比值所限定；3) 通过模拟展示了控制障碍（chokepoints）对高智能体更为显著的抑制效应，强调控制与知识比单纯限制智能更为关键的治理杠杆；4) 对纯数字与物理领域的区别给出“物理循环在路”设计建议。

**🔧 技术方法**

形式化定义、逻辑推导、基于 NK 适应度景观的随机仿真、bootstrap 置信区间、敏感性分析。

**📊 数据集**

人工生成的 NK 适应度景观（N=100），包含不同噪声比例、知识初始比例和控制障碍设置的六种情景（0~3B），并在每种情景下运行 100 次模拟。

**📈 对比分析**

与基线（β=0.1）和超智能（β=0.9、1.0）对比，测量完成最终适应度所需的步数/时间；结果显示：智能乘数在 1.8–8.9 之间，且始终低于解析上限；在存在控制障碍时，高智能体的乘数显著下降；在数字域中乘数可高达 ~8 倍，但物理域因实验延迟导致乘数仅约 1.2 倍。

**⚠️ 局限性**

1) 模型过度简化，未包含竞争、跨技术复用、推理驱动的知识生成等真实情境；2) 仅针对 NK 适应度景观的模拟，未验证在实际 AI 部署中的泛化性；3) 只考虑了实验延迟与授权延迟，未模拟更复杂的社会工程或多方协同决策；4) 结果对超智能的假设（β=0.9、1.0）理想化，实际智能提升可能受限于更低的 β。

---

## 349. Agent Benchmarks Stop Ranking Before They Run Out of Tasks

**arXiv ID:** 2611.13541 | [PDF](https://arxiv.org/pdf/2611.13541v1)

**作者:** Zihan Dong `[一作]` (Lenovo Group), Minxuan Hu `[通讯]` (Cornell University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对现有代理人基准的评估与重跑实验，发现任务数量不足，关键是测量方式导致饱和，并提出更细粒度的分数和成本计量方案。

**💡 创新点**

识别pass/fail比特过于粗糙与被引用上限限制导致的饱和，提出基于锚定参考的对数比率分数和独立成本维度的评估框架，并用实际重跑验证。

**🔧 技术方法**

使用统计显著性检验（t检验、Holm多重比较）、分层设计的门控与目标函数、对数比率评分、成本计量、LLM辅助验证器编写、以及锚点冻结与对标定技术。

**📊 数据集**

利用八个代理人基准（SWE‑bench Verified、Terminal‑Bench 2.0、ALE‑Bench、MLE‑Bench、METR、SWE‑fficiency、AlgoTune、OSWorld‑Verified）及其公开运行记录和重新运行的 13 个任务。

**📈 对比分析**

通过统计层级划分和S指数对顶尖模型进行分层，利用对数比率评分恢复多层级，比较不同阈值、连续分数与传统二进制分数，结果显示连续分数显著提高区分度，成本维度揭示显著差异。

**⚠️ 局限性**

验证器错误仅在 SWE‑bench Verified 中可量化；重跑样本有限，参考解漂移和价格表更新可能影响结果，对 LLM 判断的可靠性仍有依赖。

---

## 350. Leveraging Psychophysical Attentional Distribution for Gaze-Augmented Reward Modeling

**arXiv ID:** 2611.13477 | [PDF](https://arxiv.org/pdf/2611.13477v1)

**作者:** Chaeho Lee `[一作]` (KAIST), Sang Ho Lee `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文在奖励模型训练中引入了基于眼动的注意力分布，对总阅读时间（TRT）进行基于不对称高斯核的重分布，以模拟人类阅读时的感知范围；

**💡 创新点**

创新点在于将眼动信号从仅局限于注视词的离散特征，转变为在注视点周围以不对称高斯分布的连续注意力表示，并让分布宽度参数在训练中自适应学习；

**🔧 技术方法**

采用了眼动预测模型生成的TRT特征、对TRT进行重分布的自学习不对称高斯核、GazeReward框架的特征投影与文本嵌入融合（Concat/​Add），以及在Llama‑3系列的奖励模型上进行对比损失训练；

**📊 数据集**

主要使用了OpenAssistant Conversations（OASST1）、HelpSteer2和UltraFeedback三个对话偏好数据集；

**📈 对比分析**

通过与基线（token‑level TRT）以及多种分布设定（Uniform、Fixed、Symmetric、Asymmetric）进行对比，实验表明在OASST1上准确率从68.38%提升至74.73%（+9.3%），在HelpSteer2和UltraFeedback上亦分别提升5.3%与13.0%；

**⚠️ 局限性**

局限性包括：仅使用TRT特征且基于token级别重分布，忽略了字级感知跨度；高斯核为单峰递减，未捕捉可能的非单调或抑制性注意力模式；并且模型未考虑个体差异、阅读技能或文本难度对感知范围的影响。

---

## 351. How Hyperbolic Are Hyperbolic Vision-Language Models?

**arXiv ID:** 2611.13384 | [PDF](https://arxiv.org/pdf/2611.13384v1)

**作者:** Freek Byrman `[一作]` (University of Amsterdam), Pascal Mettes `[通讯]` (University of Amsterdam)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了当前流行的超曲面视觉语言模型（VLMs）中曲率的实际作用，发现指数映射几乎退化为恒等函数，模型在训练和推理阶段几乎不受超曲面几何影响；并训练了仅使用欧氏空间的对照模型，证明两者在性能、训练动态和半径分布上几乎一致。

**💡 创新点**

创新点在于提出并量化“失真比”作为评估超曲面VLM实际曲率的指标，系统性地在不同模型、不同监督关系下验证指数映射的近似恒等性，并通过去除指数映射和训练欧氏对照模型两种实验手段，揭示超曲面VLM在目前的实践中几乎没有发挥超曲面几何的潜力。

**🔧 技术方法**

使用的技术包括：超曲面（Lorentz模型）与欧氏空间的几何运算、对比学习损失与层次结构损失（外部角度、半开角），指数映射、失真比计算、训练曲率轨迹跟踪、实验中对指数映射的删减与欧氏对照训练。

**📊 数据集**

主要数据集为CC3M（约300万图文对），并使用其框标注进行有监督学习，评估涵盖16个零样本分类基准、COCO和Flickr30K检索以及WordNet-ImageNet的层次分类。

**📈 对比分析**

比较方法包括：在训练后直接删除指数映射在推理阶段评估模型；以及与同一目标训练的欧氏对照模型在各项指标（分类、检索、层次分类）上的对比。实验表明，删除指数映射对性能几乎无影响，欧氏对照模型在绝大多数指标上与超曲面模型相当甚至略优，说明曲率在当前模型中未被有效利用。

**⚠️ 局限性**

局限性在于研究范围主要集中在使用CC3M这一规模相对较小的数据集和短时间训练（100k步）上，虽然作者指出更大规模训练可能只会延长近欧氏阶段但未做完整验证；此外，实验只考察了指数映射对几何的影响，未探索其他可能利用曲率的设计或优化策略。

---

## 352. The Gene Is Not the Gate: Conceptualizing the Semantic Genome of Quantum Software

**arXiv ID:** 2611.13049 | [PDF](https://arxiv.org/pdf/2611.13049v1)

**作者:** Jiongchi Yu `[一作]` (Nanyang Technological University), Zhaoxuan Li `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了QGenome框架，定义量子软件基因及其分类，构建跨层责任的语义模型；

**💡 创新点**

通过引入基因（typed semantic responsibility）概念，将责任与软件层次和责任家族分离，形成可追踪、可验证的基因卡与基因组关系，为量子软件的跨层正确性、演化与重用提供系统化视角；

**🔧 技术方法**

采用语义约束与合同形式化、图模型构建基因组，结合软件层分析、依赖、精炼、溯源等关系；

**📊 数据集**

以公开的量子软件实例（如Bell态准备、量子门编译、控制系统Pulse等）为代表性案例作为示例，并未使用标准数据集；

**📈 对比分析**

作为理论框架，未进行实验比较；通过案例演示展示其在跨层语义追踪与验证中的可行性；

**⚠️ 局限性**

未给出完整分类或自动抽取技术，缺乏自动化工具与实验评估，对不同硬件和软件生态的适配仍待验证。

---

## 353. ASPIRE: An Agentic System-Identification Platform for Iterative Reasoning and Exploration

**arXiv ID:** 2611.13114 | [PDF](https://arxiv.org/pdf/2611.13114v1)

**作者:** Erik Derner `[一作]` (Czech Technical University in Prague), Robert Babuška `[通讯]` (Czech Technical University in Prague)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用大型语言模型代理自主选择并执行系统识别方法，生成可解释的闭式非线性模型并完整记录参数、估计过程与模型结构；

**💡 创新点**

创新点在于将物理约束与正式验证集成到LLM驱动的识别流程中，同时将测试数据与代理隔离，保证结果可复现且可审计；

**🔧 技术方法**

主要技术包括大型语言模型（OpenAI GPT 类）、Python 科学库（SciPy、gplearn、PySINDy）以及符号回归、稀疏回归等自定义估计方案；

**📊 数据集**

实验数据来自三组真实系统：移动机器人、带干摩擦的旋转系统以及发酵罐的压力动力学；

**📈 对比分析**

与 PySINDy 与 PySR 进行对比，ASPIRE 在测试误差、物理约束满足度与参数数量等方面表现相当或更优，且每轮成本仅约 1–3 美元；

**⚠️ 局限性**

局限在于仅处理单输出、已测状态且测量方程已知的系统，依赖单一语言模型版本，未对多模型或大规模实验进行评估。

---

## 354. Adaptive Residual Quantization for Memory-Efficient Temporal Action Segmentation

**arXiv ID:** 2611.13262 | [PDF](https://arxiv.org/pdf/2611.13262v1)

**作者:** Gerard Donahue `[一作]` (Honda Research Institute USA), Faizan Siddiqui `[通讯]` (Honda Research Institute USA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种自适应残差量化变分自编码器（ARQ‑VAE），用以压缩并存储时间动作分割（TAS）中的帧级特征，形成无标签、紧凑的特征记忆库；

**💡 创新点**

核心创新在于对每帧动态截断残差量化链，只有在重构误差最小的前缀被保留，从而实现按帧自适应的压缩；

**🔧 技术方法**

采用残差向量量化（RQ‑VAE）加上自适应前缀选择与轻量化微调；

**📊 数据集**

在Breakfast、50Salads、GTEA、YouTube Instructional等四大TAS基准数据集上进行评估；

**📈 对比分析**

与现有重放/稀疏记忆基线（TCA、GNI、RQ‑VAE等）相比，ARQ‑VAE在保持或提升编辑、F1等指标的同时，将存储空间压缩至1~10 MB（原始I3D特征存储高达28 GB），在监督、增量和无监督TAS任务中均实现了显著的存储‑性能优势；

**⚠️ 局限性**

局限性包括对预提取特征的依赖，且自适应前缀选择仅基于无标签重构误差，未来可探索更贴合TAS目标的选择策略。

---

## 355. SepsisFlow: an event-stream foundation model for continuous risk warning and trajectory modelling in ICU sepsis

**arXiv ID:** 2611.13661 | [PDF](https://arxiv.org/pdf/2611.13661v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 356. SpatialSR: Spatial Resolution-Centric Pipeline for Arbitrary-Scale Remote Sensing Super-Resolution

**arXiv ID:** 2611.13046 | [PDF](https://arxiv.org/pdf/2611.13046v1)

**作者:** Yan Zhang `[一作]` (Chongqing University of Posts and Telecommunications), Xinbo Gao `[通讯]` (Chongqing University of Posts and Telecommunications)

**通讯引用:** 47709 | [OpenAlex ID](https://openalex.org/A5101785348)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于绝对空间分辨率（SRe）的任意尺度遥感图像超分辨率框架 SpatialSR，能够在无先验尺度信息时自动估计低分辨率图像的 SRe 并生成对应高分辨率输出。

**💡 创新点**

创新点包括：① SREM 通过混合采样和层次网络自动估计 SRe；② RSCM 结合预训练的 SRe‑to‑image 扩散模型提取分辨率特定纹理风格并注入解码器；③ TC 损失将扩散模型多步推理压缩为单步训练，显著提升纹理真实性与训练效率。

**🔧 技术方法**

采用卷积+VisionMamba 结构、FiLM 风格调制、Stable Diffusion 预训练扩散模型、Trajectory Consistency Loss 以及混合采样策略实现上述模块。

**📊 数据集**

使用 ISPRS‑Vaihingen、LoveDA、UAVid、UCMD、WHU‑RS19 等公开遥感分割与分类基准数据集，并构建了新的 MSRUAV 实际降解数据集进行评估。

**📈 对比分析**

在合成与真实降解场景下与多种生成式（ESRGAN、LDM、StableSR、ResShift）和 INR（LIIF、ContSR、AnySR、LMF、HIIF）方法对比，SpatialSR 在 PSNR、SSIM、LPIPS、DISTS、FID 以及下游语义分割 mIoU 与分类 Top‑1 Accuracy 上均达到或接近现有最优水平，尤其在大倍率和真实降解情境表现尤为突出。

**⚠️ 局限性**

局限性在于只能在预定义的分辨率范围内工作，对超出该范围或跨传感器的 SRe 估计与纹理约束效果下降，需要进一步研究连续 SRe 演化建模以提升泛化能力。

---

## 357. Reinforcement Learning based control of Cancer Chemotherapy treatments

**arXiv ID:** 2611.12940 | [PDF](https://arxiv.org/pdf/2611.12940v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 358. MOAI: Module-Optimizing Architecture for non-interactive secure transformer Inference

**arXiv ID:** 2611.13641 | [PDF](https://arxiv.org/pdf/2611.13641v1)

**作者:** Linru Zhang `[一作]` (Nanyang Technological University), Kwok-Yan Lam `[通讯]` (Nanyang Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在云端为Transformer模型（如BERT-base、LLaMA-3-8B）提供全非交互的同态加密推理框架 MOAI，显著减少 HE 旋转操作并提升推理速度。

**💡 创新点**

创新点包括：
- 统一列/对角矩阵打包策略，避免层间格式转换；
- 旋转无关的 Softmax 与 LayerNorm 评估算法，完全消除这两项操作中的 HE 旋转；
- 对齐批处理（interleaved batching）降低矩阵乘法中的 HE 旋转；
- 精细化 Bootstrapping 放置策略，进一步降低加密深度和运行时间。

**🔧 技术方法**

核心技术：CKKS 同态加密、列/对角矩阵打包、无旋转 Softmax/LayerNorm、对齐批处理、Goldschmidt 反演、指数多项式逼近、精细的 Bootstrapping 调度。

**📊 数据集**

使用公开数据集：SST-2、QNLI、RTE（GLUE 基准），以及在 LLaMA-3-8B 上的推理评测。

**📈 对比分析**

与现有最先进的 HE 推理方案 THOR（CCS'25）和 Powerformer（ACL'25）在同一 NVIDIA A100 GPU 上对比：MOAI 在 BERT-base 上实现了 52.8% 的推理时间下降（≈2.36 分钟/输入），在 Powerformer 上实现 55.7% 的时间降低；在 CPU 环境下每输入 9.6 分钟；总体上比 THOR 和 Powerformer 减少 1.7× 以上 HE 旋转次数。

**⚠️ 局限性**

局限性：
- 虽然相比 THOR 和 Powerformer 加速显著，但整体推理仍依赖高成本的 Bootstrapping，尤其在更大模型或更长序列时可能成为瓶颈；
- 目前主要针对 Transformer 编码器（BERT）和解码器（LLaMA），对其它网络结构的通用性尚未完全验证；
- 需要精细调参（如 CKKS 参数、分级层级）以保持 128-bit 安全等级，部署复杂度较高。

---

## 359. An IoT Based Approach for Platform Independent Positioning Service

**arXiv ID:** 2611.13634 | [PDF](https://arxiv.org/pdf/2611.13634v1)

**作者:** Anh-Van Vu `[一作]` (Korea Advanced Institute of Science and Technology), Dongsoo Han `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了基于IoT的可穿戴设备与KAIST Indoor Localization System（KAILOS）集成的室内定位架构，移除了移动操作系统对WiFi/蓝牙扫描的限制，实现了无需安装App即可通过网页浏览器访问定位服务。

**💡 创新点**

核心创新在于将定位数据采集迁移到专用IoT设备，利用单芯片集成WiFi/蓝牙/ LTE/ GNSS 与多种传感器，提供高采样率连续定位并支持多设备共享单个定位服务，同时保持与传统KAILOS的向后兼容。

**🔧 技术方法**

采用WiFi fingerprinting、IMU（加速度计、陀螺仪、磁力计）融合、PDR 步态计数、GNSS、BLE、LTE上传、KAILOS引擎滤波与混合定位算法，并在硬件层面使用SoC、UART/SPI/I2C通信、前后端服务器架构。

**📊 数据集**

实验使用KAIST校园已有的KAILOS无线电图（radio map）以及现场采集的WiFi指纹和IMU数据，在KAIST IT大楼 N1 进行真实路径与地面真值对比。

**📈 对比分析**

通过与Android移动App对比，IoT设备实现每分钟30次扫描、每次扫描2秒，平均误差约2.4米；PDR实验表明在IoT设备上采用100Hz采样率能得到最优轨迹匹配；整体性能显示IoT方案兼具高采样率、无OS限制和群组用户支持，误差保持在2–3米。

**⚠️ 局限性**

目前存在不支持5GHz WiFi、设备尺寸较大、功耗与电池寿命有限，以及仅实现WiFi核心定位、缺乏多场景验证和完整多模态（BLE/GNSS）支持等局限，需要进一步优化硬件轻量化、功耗和多频段支持。

---

## 360. Collision Avoidance and Collision Detection Strategies to Counteract Large-Scale FMCW Automotive Radar Interference

**arXiv ID:** 2611.13214 | [PDF](https://arxiv.org/pdf/2611.13214v1)

**作者:** Francesco Miccoli `[一作]` (National Laboratory of Wireless Communications (WiLab)), Alessandro Bazzi `[通讯]` (National Laboratory of Wireless Communications (WiLab))

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过系统层仿真评估了两种主动干扰缓解策略（FbF-CA和FbF-CD）在密集高速公路环境中对FMCW汽车雷达的干扰抑制效果。

**💡 创新点**

创新点在于：①提出基于空闲时段的干扰功率感知实现的帧级碰撞规避（FbF-CA）；②提出基于传输中已损坏的chirp计数实现的帧级碰撞检测（FbF-CD）。两种方案均为无协调的本地自适应方法。

**🔧 技术方法**

采用事件驱动的雷达网络仿真框架（MATLAB实现），结合SUMO交通仿真与WiLabVIsim潜在干扰器识别；使用FMCW雷达信号模型、传播模型及干扰损坏阈值来量化干扰。

**📊 数据集**

使用了基于SUMO生成的150辆/公里密集高速公路车辆位置和运动数据，以及WiLabVIsim提供的潜在干扰器列表。

**📈 对比分析**

将FbF-CA、FbF-CD与基线（固定频率）、FbF（帧级频率跳变）以及CbC（chirp级频率跳变）进行对比，评估指标为系统失败间平均时间（即1/失败概率）。实验显示：在有限带宽时，FbF-CD表现最佳；随着可用总带宽增加，FbF-CA逐渐优于FbF-CD；若带宽足够大，CbC虽理论上最好但实际实现难度高。

**⚠️ 局限性**

局限性包括：仅考虑了相关干扰且使用相同波形参数；未研究异构雷达配置或非相关干扰；只关注主动（预防性）策略，未结合被动后处理方法；实验仅在140 GHz波段进行，结果对其他频段的推广需进一步验证。

---

## 361. Just Image Transformers with Late-Stage Pixel Refinement

**arXiv ID:** 2611.13120 | [PDF](https://arxiv.org/pdf/2611.13120v1)

**作者:** Denis Lukovnikov `[一作]` (Ruhr University Bochum), Asja Fischer `[通讯]` (Ruhr University Bochum)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在预训练的Just Image Transformer（JiT）基础上，添加并微调了一个小型卷积像素路径（pixel‑head），并证明该路径仅在去噪过程的后20%阶段使用即可显著提升图像质量。

**💡 创新点**

创新点包括：① 在后训练阶段插入卷积像素路径实现局部细节补偿；② 结合x‑prediction目标，使像素路径可在后期启用；③ 对采样过程进行去除分母裁剪的修正，恢复完整去噪；④ 在多种模型尺寸下通过双头（线性+卷积）实现高效、可调的像素细化。

**🔧 技术方法**

采用的技术包括：x‑prediction训练目标、局部卷积像素头（ConvNeXt‑style）、双头轻量级像素路径、Heun 采样器、软‑最小‑SNR 权重的时间步采样、EMA 与 CFG 的调优、残差与跳连结构的 ablation。

**📊 数据集**

使用 ImageNet 256×256 数据集进行训练与评估，生成 50k 样本后计算 FID、IS、sFID 等指标。

**📈 对比分析**

与原始 JiT‑B/16、JiT‑L/16 以及最新的 Pixel‑DiT/XL 等模型对比，后者在同等计算资源下取得了更低的 FID（JiT‑H/16 + Dual Head 的 FID 为 1.78，远优于原始 2.34），且在 GPU 计算量上仅略微增加（JiT‑B/16 + Conv Head 约 +21%，JiT‑L/16 + Conv Head 约 +6%）。

**⚠️ 局限性**

局限性包括：① 像素路径的容量仅为四层卷积，未探索更大、更复杂的结构；② 像素路径加入时机和细化策略尚未在更大模型或文本条件生成任务上充分验证；③ 对于不同时间步采样分布的最佳设置仍需要进一步研究；④ 仅在 ImageNet 上验证，缺乏多域、跨模态的泛化评估。

---

## 362. Preserving Cultural Records: A Large-Scale Benchmark and Visual-Prior Routing Network for Ancient Book Image Restoration

**arXiv ID:** 2611.13010 | [PDF](https://arxiv.org/pdf/2611.13010v1)

**作者:** Shumin Fan `[一作]` (Dalian Maritime University), Xiang Chen `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建大型合成对照数据集AncBench，并提出针对古籍图像的动态视觉先验路由网络AncRestore；

**💡 创新点**

提出了状态依赖的多阶段降解生成流程与基于损伤表征的专家路由机制，能在全局与局部层面自适应选择视觉先验；

**🔧 技术方法**

使用DINOv3视觉先验提取、损伤表征块DRB、动态专家路由DER、HQS恢复骨干及多尺度损伤先验；

**📊 数据集**

主要使用自建的AncBench（100K对齐清晰-降解图像），并在HDR28K、HanInpaint等公开数据集上验证；

**📈 对比分析**

与多种通用与文档特定恢复方法对比，AncRestore在AncBench上PSNR 26.70 dB、SSIM 0.9191、LPIPS 0.1346，OCR识别准确率达89.58%，在HDR28K与HanInpaint也保持领先或竞争性表现；

**⚠️ 局限性**

对实际古籍场景的适用性仍受限于合成降解与真实噪声差异，且模型复杂度高，部署成本较大。

---

## 363. An ensemble of graph generative models for assisted history matching of the Brugge field case

**arXiv ID:** 2611.12927 | [PDF](https://arxiv.org/pdf/2611.12927v1)

**作者:** Gleb Shishaev `[一作]` (Tomsk Polytechnic University), Daniel Arnold `[通讯]` (Heriot-Watt University)

**通讯引用:** 2742 | [OpenAlex ID](https://openalex.org/A5019549039)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Brugge多地层场景下，提出基于图Wasserstein自编码器的多生成器历史匹配框架，直接在角点网格上生成孔隙度和渗透率，并在混合潜空间中进行优化；

**💡 创新点**

创新点包括：①把生成器按地层分解并使用拓扑数据分析判断是否需要分解；②在原始不规则网格上构造图并训练WAE；③在目标函数中加入基于解码器雅可比矩阵的地质真实性惩罚；

**🔧 技术方法**

使用技术包括图WAE、CMA-ES优化、PCA/t-SNE/拓扑持久同调诊断、Jacobian realism penalty、两阶段优化策略；

**📊 数据集**

使用数据集为10,000条Brugge基准场景的地层化训练集（四层地层，包含OM和SIS两种建模方法），以及对应的井和生产观测；

**📈 对比分析**

通过与EnKF、PSO‑PCA、MKL‑PSO等传统方法在井级匹配好/差分类的对比，GWAE实现大部分井匹配为“好”，生产响应落在先验范围内，孔隙度/渗透率误差控制在1–11％；

**⚠️ 局限性**

局限性包括仅更新孔隙度和渗透率，未考虑相对渗透率及其它流动参数；假设各地层独立；潜在空间仍受训练集规模限制；未进行后验抽样或引入其他观测（如4D地震、示踪剂）。

---

## 364. F3-AGI v4.0: Bioenergetically-Gated Active Inference, Metaplastic Preconditioning, and Residual Program Synthesis for Sample-Efficient Abstract Reasoning

**arXiv ID:** 2611.13325 | [PDF](https://arxiv.org/pdf/2611.13325v1)

**作者:** Henrique Oliveira `[一作]` `[通讯]` (Polybee AI Research), Henrique Oliveira (Polybee AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 F3-AGI v4.0，融合生物能门控的元可塑性优化、适应性复杂度引导的程序合成与循环‑3残差补丁合成，并通过 512 维随机投影实现无 LLM 的主动推理，针对 ARC‑AGI 任务实现样本高效的抽象推理与持续学习。

**💡 创新点**

创新点包括：① 将错误门控、在线 Fisher‑EMA 预调与虚拟 ATP 能量门控整合为单一生物能驱动的元可塑性更新；② 在 AACR 合成中加入最小对（smallest‑pair）早期拒绝和循环‑3残差补丁机制，实现结构子空间收敛；③ 使用 512 维等距随机投影和死签名剪枝的主动推理框架，在 CPU 约 400 MB RAM 下实现主动探索。

**🔧 技术方法**

技术主要包括：CPU‑native 神经符号程序合成、误差门控可塑性更新、在线 Fisher‑EMA 预调、虚拟 ATP 代谢约束、最小对早期拒绝、循环‑3残差补丁、512 维等距随机投影、死签名剪枝与交叉层规则转移。

**📊 数据集**

使用的数据集：ARC‑AGI‑1 公共评测集（400 任务）、N=50 发展切片、局部 5 级互动套件（5 款游戏共 15 级 64×64 网格）、以及 5 任务类增量 64‑维合成数据。

**📈 对比分析**

与传统方法对比：相较于 EWC、Brute‑Force LLM 集成，F3‑AGI 在无 LLM 调用下实现 57/400（14.25% pass@2）任务成功率，平均任务延迟 1.62 s；在 5 任务类增量实验中达 91.67% 终止准确率、9.17% 遗忘率、M1 代谢成本下降 65.7% 并实现 2.19× 速度提升；在 15 级互动套件中完成 100% 关卡，RHAE 加权 89.44%（未加权 90.72%）。

**⚠️ 局限性**

局限性：仅在已检验的 50 任务上扩展 DSL，未能在 350 任务的严格未检验集上获得高通量表现；使用固定随机投影缺乏自适应特征学习；算法依赖手工设计的 30 操作符库，表达能力受限；未在 ARC‑AGI‑2 或私有 ARC‑AGI‑3 服务器上公开验证；未充分评估对非平面 3D 或自然语言交互场景的泛化。

---

## 365. An evidence-traceable benchmark of clinical reasoning in large language models

**arXiv ID:** 2611.13562 | [PDF](https://arxiv.org/pdf/2611.13562v1)

**作者:** Zhiqun Zhao `[一作]` (Shandong University), Xiaoming Wu `[通讯]` (Shandong Computer Science Center)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了Med Evidence Bench（MEB），一个可追溯证据的临床推理基准，并在三层级（指南知识、标准化情境、真实世界实验室情境）上评估了10款大型语言模型的表现。

**💡 创新点**

创新点包括：①将每个评分标准与版本化证据记录和文本片段关联，实现可审计；②在三层级区分指南回忆、证据应用与证据激活；③引入自动化评判器与医师人工评估对比，验证评判的可靠性。

**🔧 技术方法**

使用了LLM驱动的评判器（CEC Agent）和质量控制代理（MEB‑QC）进行自动评分和证据链追溯，并利用十款LLM生成回答。

**📊 数据集**

数据集由1,536条医师构建的项目组成，涵盖319条指南知识题、852条标准化临床情境、365条真实世界实验室情境；证据语料约4万条指南、共识及教材，按三层（A/B/C）分类。

**📈 对比分析**

通过Evidence Trace Score（ETS）进行比较，指南知识层得分0.975–0.997，标准化情境0.292–0.658，真实世界情境0.182–0.447。医学专用模型排名靠前，但不同层级的排名差异显著，说明知识回忆不足以保证临床证据应用。

**⚠️ 局限性**

局限性包括：仅使用中文材料，难以推广至其他语言和健康系统；模型评估条件不完全一致；级别与能力域未完全交叉；自动评分在证据匹配与应用阶段的可靠性较低；未涵盖多轮对话、影像、长期追踪及临床实证验证。

---

## 366. GigaEdge Spench: An SDK for Practical Streaming Speech Enhancement

**arXiv ID:** 2611.13243 | [PDF](https://arxiv.org/pdf/2611.13243v1)

**作者:** Artem Sokolov `[一作]`, Vitaliy Yulis `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并发布了GigaEdge Spench SDK，提供可配置的实时语音增强通道，支持CPU服务器、移动端和浏览器。

**💡 创新点**

创新点在于：① 单一C++核心实现模型共享、可按需组合DSP与神经网络块；② 支持多种后端（ONNX Runtime、TensorFlow Lite）和跨平台接口；③ 通过一次推理调用处理多帧，显著提升吞吐量；④ 在同一硬件上与RNNoise、DPDFNet2对比，展示模型、块大小和并发数对质量与资源占用的权衡。

**🔧 技术方法**

技术手段包括：语音增强神经网络（FullSubNet改造、双路径网络）、ONNX Runtime / TensorFlow Lite推理、SpeexDSP回声消除、可配置归一化与动态处理链、可调块大小与多线程/WebWorker并行、模型共享与递归状态缓存。

**📊 数据集**

使用EARS–WHAM v2作为外部质量评测数据集，评估8kHz INT8模型和48kHz FP32模型；此外使用RNNoise 0.2和DPDFNet2的公开权重做基准对比。

**📈 对比分析**

比较方法：在同一WSL2+Ubuntu 24.04.5环境下，利用ONNX Runtime测算RTF、CPU/内存占用，并通过ESTOI、PESQ-WB/NB、SI‑SDR、DNSMOS、SIGMOS、NISQA等多维指标对质量进行评估；同时通过并发CLI进程和共享模型多流实验测定可支持的最大并发数。性能结果显示：Spench 48kHz模型在质量上略低于DPDFNet2但计算量显著减少；Spench 8kHz INT8在窄带任务中优于RNNoise downsample；Spench 48kHz模型可支持约590个并发流（vs RNNoise 1088），而DPDFNet2仅约56个。WebAssembly实现相对本机慢约2–3倍，但仍在实时范围内。

**⚠️ 局限性**

局限性：仅在单一WSL2/Ubuntu硬件上评估，未覆盖低端设备或高并发实时场景；模型训练与优化过程未公开，难以复现；仅测试了EARS–WHAM v2，缺乏跨域多场景验证；未对长时延实时性、与视频编码器协同负载等实际应用场景进行完整评估。

---

## 367. Useful Work Can Increase Energy Use: Demand Elasticity and Proof of Useful Work

**arXiv ID:** 2611.13131 | [PDF](https://arxiv.org/pdf/2611.13131v1)

**作者:** Isaiah Hull `[一作]` (FirstQFM), Or Sattath `[通讯]` (Ben Gurion University of the Negev)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

本文通过经济均衡模型分析了从PoW转向PoUW后，因需求弹性可能导致总电力消耗反而上升的现象；

**💡 创新点**

创新点在于提出了“后退效应”阈值 γ>1+R，揭示了弹性需求在PoUW中的潜在能源后果；

**🔧 技术方法**

采用需求弹性分析、线性与一般需求曲线的泰勒展开、均衡方程推导等经济学工具；

**📊 数据集**

无实测数据集，全部使用理论推导与假设场景；

**📈 对比分析**

比较方法为理论上前后均衡的电力消耗差异，未涉及实验性能指标；

**⚠️ 局限性**

局限性包括仅考虑电力成本、忽略Proof生成开销、仅适用于私有需求的有用工作市场。

---

## 368. ICLayout-Bench: Benchmarking LLM Agents for Custom Circuit Layout Design

**arXiv ID:** 2611.12982 | [PDF](https://arxiv.org/pdf/2611.12982v1)

**作者:** Zhangmai Li `[一作]` (University of Science and Technology of China), Yunji Chen `[通讯]` (Chinese Academy of Sciences)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ICLayout-Bench基准，评估LLM代理在从晶体管级网表构建并优化GDSII布局的性能，并在21个公开与19个私有任务上测试22种配置。

**💡 创新点**

创新点：①设计完整端到端评估流程（DRC/LVS/寄生提取/后仿真）；②构建跨技术节点、PDK与EDA流水线的多任务基准；③同时评估模型成本与token‑to‑score比率。

**🔧 技术方法**

技术：使用大型语言模型（GPT‑6系列、DeepSeek Flash、Kimi K3.1 Pro、GLM‑5.3 Flash）与官方harness，通过Model Context Protocol桥接调用KLayout、Magic、Virtuoso等EDA工具链。

**📊 数据集**

数据集：ICLayout-Bench，公开任务来自OpenRAM、spicexplorer、OpenFASOC、sky130_ef_ip等，私有任务来自tape‑out验证；托管于HuggingFace。

**📈 对比分析**

比较方法：对每配置在公私任务集合上计算平均分（0‑100）并绘制模型成本与token‑to‑score Pareto曲线；最高平均分达80.26（公用）或67.01（私有）。

**⚠️ 局限性**

局限：①更高推理努力不一定提升分数；②模型成本与分数不成正比；③评估仅覆盖现有任务，未覆盖更大规模或不同架构电路；④对EDA工具/工艺的依赖限制迁移性。

---

## 369. Structured Kernel Surrogate Models for Multimodal Attribution

**arXiv ID:** 2611.13198 | [PDF](https://arxiv.org/pdf/2611.13198v1)

**作者:** Ziniu Zhang `[一作]` (Northeastern University), Hongyang R. Zhang `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种跨样本模态归因方法，使用结构化核代理模型估计一个样本中某一模态对整体预测损失的影响。

**💡 创新点**

创新点在于：①构造分层RBF核，分别衡量样本参与度与同一样本内模态选择的相似性；②利用核岭回归拟合从选择矩阵到损失的非线性映射；③通过有限差分得到每个样本-模态对的归因分数，并在大规模模型上实现梯度线性化高效估计。

**🔧 技术方法**

采用核岭回归（结构化RBF核）、梯度线性化、有限差分、以及RBF核带宽调参等技术。

**📊 数据集**

在MMTraCE（交通事故预测，包含卫星图像、交通量、天气等多模态）和MultiBench（六个多模态任务：UR‑FUNNY、MUStARD、CMU‑MOSI、CMU‑MOSEI、AV‑MNIST、MM‑IMDB）上进行评估。

**📈 对比分析**

与影响函数、TracIn、Trak、DSIR、LESS、BIF、线性代理等基线比较；在LDS（Spearman相关）指标上提升约50%–65%；在固定20%模态选择预算下的下游数据选择任务中平均提升1.5%–2.3%准确率；所选数据模型具有更低的Hessian迹，表明更平坦的损失曲面和更好的泛化。

**⚠️ 局限性**

局限性包括：①实验主要聚焦于两模态场景，扩展到更多模态的实际效果尚未充分验证；②梯度线性化近似对极端非线性模型可能不够精确；③核带宽选择虽不敏感但仍需经验调参；④当前方法对大规模数据集的内存与时间开销尚待进一步优化。

---

## 370. Ego Guidance: Enhancing Robot Policy Learning with Action-Free Human Egocentric Videos

**arXiv ID:** 2611.13685 | [PDF](https://arxiv.org/pdf/2611.13685v1)

**作者:** Lin Long `[一作]` (Zhejiang University), Shanghang Zhang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用无动作标签的人体第一人称视频，通过联合动作与未来视觉状态预测，提升机器人策略学习；

**💡 创新点**

提出SPAR框架，将预训练VLM扩展为MoT结构，动作专家与生成专家并行学习，实现跨数据源（机器人与人类视频）的无监督预训练；

**🔧 技术方法**

基于Vision‑Language‑Model的Mixture‑of‑Transformers（MoT），使用flow‑matching与回归两种视觉状态预测目标，采用两阶段训练（预训练+微调），并在推理时移除生成专家；

**📊 数据集**

机器人数据来自DROID与LIBERO（用于评估），人类视频来自Ego4D（经质量门控）和Open‑AoE（约2,000小时，1M视频‑描述对）；

**📈 对比分析**

与大型机器人预训练的VLA基线π₀.₅进行对比：在LIBERO‑Plus OOD任务中，SPAR平均成功率提升至约76%（相较于基线约70%）；在真实机器人5项任务中，SPAR平均成功率为40%（略低于π₀.₅的44%），但在光照、背景与静态干扰等视觉扰动下，SPAR表现更稳健；

**⚠️ 局限性**

仍面临精准抓取、多臂旋转操作及长周期执行的挑战，且对小规模机器人预训练语料依赖较高；在数据多样性权衡中，单纯提高多样性并未显著提升性能；

---

## 371. Tabular Foundation Models vs Tuned Gradient Boosting for Telecom Churn: A Profit- and Calibration-Aware Evaluation

**arXiv ID:** 2611.12942 | [PDF](https://arxiv.org/pdf/2611.12942v1)

**作者:** Ismail Hossain Polas `[一作]` (Jahangirnagar University), Ismail Hossain Polas `[通讯]` (Jahangirnagar University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5102043467)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对 Tabular Foundation Models（TabPFN 与 TabICL）与调优梯度提升树（LightGBM、XGBoost、CatBoost）在电信客户流失预测中的收益、概率校准以及阈值策略进行实证评估。

**💡 创新点**

在利润（EMP）和校准的视角下，首次使用多重比较校正对模型进行公平比较；系统评估了不同版本的 TabPFN、TabICL 与调优 GBDT 的表现，并探讨了均衡上下文对性能的影响。

**🔧 技术方法**

使用 TabPFN（2.5、3.5、3.5‑Fast）与 TabICL v2，Optuna 对 LightGBM/XGBoost/CatBoost 进行 25‑40 次（部分 200 次）AUC 调优；评估指标包括 EMP、e‑Profits、PR‑AUC、ROC‑AUC、校准误差（ECE、Brier、校准‑大）以及政策回报损失；统计检验采用 Nadeau‑Bengio 纠正 t 检验与 Holm 校正。

**📊 数据集**

五个真实运营商数据集（Cell2Cell、Iranian、Orange、Telco、Maven）以及一个 E‑Commerce 数据集用于复现；所有数据均经过相同的预处理（类别编码、缺失处理、去重）。

**📈 对比分析**

通过 5 个随机种子、60/20/20 分层拆分，所有模型共享同一拆分；对 EMP 进行多重比较校正后发现 TabPFN‑3.5 在 Cell2Cell、Iranian 上显著优于最佳调优 GBDT，Orange 上方向一致但未显著；TabICL v2 仅在 Iranian 上显著；在概率校准方面，TabPFN‑3.5 原始概率已接近实际 churn 率，且对阈值策略的回报损失极小；均衡上下文在大多数数据集未产生提升。

**⚠️ 局限性**

仅基于三家真实运营商的重采样数据，未检验跨运营商泛化；调优仅针对 AUC，未直接调优 EMP；去重策略可能影响结果；TabPFN 的非商业许可证限制其在生产环境中的直接使用。

---

## 372. Developing an Operational Model for an Internet of Data Based Trusted Data Space for Vocational Education

**arXiv ID:** 2611.13520 | [PDF](https://arxiv.org/pdf/2611.13520v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 373. RetinexGS: Retinex-Inspired Gaussian Splatting for Low-Light RAW Reconstruction

**arXiv ID:** 2611.13280 | [PDF](https://arxiv.org/pdf/2611.13280v1)

**作者:** Yulang Fei `[一作]` (University of Waterloo), Chul Min Yeum `[通讯]` (University of Waterloo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在低照度RAW图像下改进3D高斯云渲染，提出RetinexGS框架以实现高质量实时新视角合成。

**💡 创新点**

将Retinex启发的颜色分解为视角独立基色与共享空间视角调制，结合DNG噪声信息与畸变感知渲染，以及深度引导初始化。

**🔧 技术方法**

采用高斯分数云、乘法式颜色因子、DNG噪声加权损失、光学畸变内置光栅化、单目深度补全以及多场景实验。

**📊 数据集**

评估于RawNeRF三公开低光场景（pianotest、officetest、yuccatest）及自制两低照度场景（housecorner、miningtunnel）。

**📈 对比分析**

在原生流水线和统一输入两种协议下与LE3D、RawNeRF、RawGS、HDRSplat等基线对比，RetinexGS在sRGB PSNR平均提升2.18–2.71 dB，实时帧率91–145 FPS。

**⚠️ 局限性**

仍在极暗或纹理缺失区产生浮动伪影，局限于静态场景且依赖DNG噪声元数据，未来需更广泛传感器验证及更强几何约束。

---

## 374. LLM Validation Makes Minimal-Pair Benchmarks Easier for Language Models

**arXiv ID:** 2611.13788 | [PDF](https://arxiv.org/pdf/2611.13788v1)

**作者:** Guofang Tang `[一作]` `[通讯]` (Ghostcloud), Guofang Tang (Ghostcloud)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究使用大型语言模型（LLM）来替代人类评审，对 BLiMP（英语）和 QFrBLiMP（魁北克法语）的最小对偶基准进行验证，并在两阶段预注册协议中评估其效果。

**💡 创新点**

创新点包括：① 设计并执行了预注册的三阶段协议，系统比较 LLM 验证器与人类评审小组在标签检测上的性能；② 发现 LLM 验证器在高一致率下却极度缺失人类拒绝的对偶，导致验证过程偏向 LLM 自己的偏好；③ 证明了 LLM 过滤器会提升模型得分但掩盖真正的语言错误，并在新语言（荷兰语）上复制了该效应。

**🔧 技术方法**

技术手段包括：① LLM 的强制选择（Forced Choice）、逐句评分（Rating）以及三模型投票面板；② 用开放 LMs 的对数概率评估对偶的正确性；③ 采用 1,000 次重采样和 10,000 次项目自助法计算置信区间；④ 预注册的阈值、效应量和统计检验。

**📊 数据集**

使用的数据集为：BLiMP（英语）、QFrBLiMP（魁北克法语）以及 BLiMP-NL（荷兰语），其中后两者提供了每位评审者的逐对决策信息。

**📈 对比分析**

比较方法：将 LLM 验证器与由 1、3、5 名人类评审组成的面板进行 LNS（标签不受支持）检测率、精确率、召回率、F1 以及与 held‑out 大多数意见的一致率比较。结果显示：LLM 验证器的“一致率”高于 5 人面板，但其召回率远低于单一评审，整体效能不超过一名人类评审；此外，LLM 过滤后的对偶更难以识别，去除这些对偶可使模型准确率提升数个百分点，且提升幅度超出随机去除。

**⚠️ 局限性**

局限性：仅评估了三种公开基准，LNS 对偶并不一定无效；验证器仅采用了三种 API 模型、单一温度和三种提示，可能影响检测结果；所用开放 LMs 规模有限，未涉及更大或闭源模型；分析为相关性，未揭示机制；对荷兰语基准的检测仅为描述性，缺乏统计显著性。

---

## 375. Do Language Models Feel the Sentiment? Investigating Emotion Vectors

**arXiv ID:** 2611.13744 | [PDF](https://arxiv.org/pdf/2611.13744v1)

**作者:** Ali Hassan Khan `[一作]` (National University of Computer and Emerging Sciences), Hajra Waheed `[通讯]` (National University of Computer and Emerging Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究情感向量在 Qwen2.5 模型中的情感信息与可解释性，探讨其对情感分类的影响及在跨域情感校正中的实用价值。

**💡 创新点**

首次系统评估情感向量在情感信息恢复、干预效果以及跨域校正中的作用，并将结果与能量匹配的随机子空间进行对比，揭示情感方向并非唯一决定情感预测的机制。

**🔧 技术方法**

采用情感向量提取、PCA 去噪、情感与情感方向的几何测量、情感探针、激活干预（Steering 与子空间消融）、基于逻辑回归的情感校正头以及跨域评估等技术。

**📊 数据集**

GoEmotions、EmpatheticDialogues、SST‑2、TweetEval、Amazon Polarity。

**📈 对比分析**

与随机方向、全状态和去噪后情感方向的情感探测精度进行对比；使用能量匹配消融与随机消融进行对比，结果显示在 TweetEval 上情感方向消融显著降低准确率，SST‑2 上无显著影响；情感校正头在 TweetEval 上提升约 6 百分点，但在 Amazon 跨域转移中并未始终优于仅使用模型边际。

**⚠️ 局限性**

仅评估两款 Qwen 模型且仅在单一层进行实验；PCA 去噪阈值固定，未做阈值敏感性分析；消融实验使用历史样本，跨域评估未对模型重新训练；统计方法受 Bootstrap 精度限制；与随机方向比较不等价；未验证在其他模型族的普适性。

---

## 376. CIALLO: Learning Shared Intent Representations for Cross-Embodiment Manipulation

**arXiv ID:** 2611.13732 | [PDF](https://arxiv.org/pdf/2611.13732v1)

**作者:** Choo Sin Wai `[一作]` (Tsinghua University), Haoqian Wang `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 CIALLO 行动码解码器，通过将动作、视觉变化和语言映射到共享的意图空间，实现跨机器人知识迁移，并在新机器人上仅需短时录制校准即可使用。

**💡 创新点**

创新点在于：①将共享意图与身体特定参数分离；②利用视觉与语言的对比对齐来约束意图空间；③通过仅校准嵌入向量实现快速迁移；④将预训练的意图空间与大型 LLM（Qwen3.5-4B）结合，无需大规模动作预训练即可直接学习任务。

**🔧 技术方法**

使用技术包括：对比学习与 InfoNCE 对齐、动作重构与 Softmax 期望回归、冻结 DINOv2 视觉骨干、BERT 文本编码、Qwen3.5-4B 作为意图预测器、StarVLA 结构、以及针对动作、视觉与语言的三条编码路径和共享解码器。

**📊 数据集**

数据集涵盖 125k 条演示，来自 8 种机器人（LIBERO、Bridge、Fractal、RoboTwin 2.0 等）；真实机器人实验使用 SO-101 与单臂 Piper；模拟实验使用 LIBERO、RoboTwin、LIBERO-Plus 等。

**📈 对比分析**

在基准对比中，Qwen-CIALLO 在 LIBERO 四套平均 97.3%（超过 CLAP‑RF、OpenVLA‑OFT、StarVLA 等）；在 LIBERO‑Plus 无需再训练即可达 80.8%（高于 DreamWAM）；在 RoboTwin 2.0 达 52.7%（优于 StarVLA、π₀ 等）；在真实机器人平均 76.7%（优于全微调的 π₀ 70.0%）。

**⚠️ 局限性**

局限性：需要预训练八个机器人且对意图空间的质量高度依赖；在长周期、复杂连续任务上的性能下降；只关注短期动作编码，可能难以捕捉长期依赖；冻结的视觉特征在极端环境变化时可能限制泛化。

---

## 377. Omni-JEV: Text and Dual-Stream Speech Decisions

**arXiv ID:** 2611.13777 | [PDF](https://arxiv.org/pdf/2611.13777v1)

**作者:** Shengzhou Kong `[一作]` `[通讯]` (Merit Interactive Research Institute), Shengzhou Kong (Merit Interactive Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并训练了Omni-JEV模型，利用共享的STS理解路径和角色分离的双声道编码，实现在收集段端点对全双工语音助手的状态条件下，输出五种交互动作（等待、开始、保持、打断、恢复）的概率分布；

**💡 创新点**

创新点在于：1) 统一候选分布接口，将文本和角色索引语音映射到同一动作空间；2) 通过共享前端但独立角色编码，保持用户与助手语音的区分；3) 在相同初始化下对比Gaussian logit探索与负Brier Choice RLCD两种完整策略；4) 构建包含中英双语、9源混合的弱标注交互动作语料库；

**🔧 技术方法**

采用MiniMind‑O Thinker–Talker（SenseVoice前端+Qwen3语言模型）进行对齐STS初始化；使用交叉熵监督后，分别实施Gaussian logit探索与负Brier Choice的RLCD；利用角色保留的双声道编码、BERT‑style提示串联、原始LM头输出候选概率；

**📊 数据集**

使用9个来源（FD v2/v3、BANKING77、MASSIVE、MInDS‑14、ESC‑10、Mini Speech Commands、Intelligent Wakeup、SmoothConv、内部转录文本）构成约313k训练、34k测试的双语（中英）交互动作数据集；

**📈 对比分析**

在固定源分割下，先进行一次SFT再一次RLCD（Gaussian或Choice），与多款全双工助手（Qwen3‑Omni‑30B、MiniCPM‑o 4.5、MOSS‑Audio‑4B、Step‑Audio‑2‑mini、FireRedAudio、AudioJev）进行零‑shot对比；Omni‑JEV音频准确率84.08%、宏F1 76.00%；文本准确率96.55%、宏F1 80.56%；Gaussian RLCD在音频宏F1略高，Choice RLCD在Brier、NLL上更优；

**⚠️ 局限性**

局限性包括：1) 仅在离线端点任务评估，未验证实时闭环延迟、回声鲁棒性和多轮交互性能；2) 依赖弱标注策略，可能存在标注偏差；3) 语音与文本来源不均衡，导致结果难以直接比较；4) 未对多通道噪声、环境变化和同步问题进行系统测试。

---

## 378. Weighted-AoI-Driven UAV Trajectory and Reporting for Dynamic CKM Tracking

**arXiv ID:** 2611.13726 | [PDF](https://arxiv.org/pdf/2611.13726v1)

**作者:** Zeyu Zhao `[一作]` (Shenzhen University), Sheng Luo `[通讯]` (Shenzhen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在UAV机队下持续在线更新动态通道知识图（CKM）的优化问题，提出基于加权信息年龄（W-AoI）的轨迹与测量调度策略。

**💡 创新点**

创新点在于将时间信息年龄与U‑Net预测的空间不确定度相结合，构造W‑AoI作为单细胞优先级指标，并证明其上界逼近真实误差；此外设计了轻量级贪婪轨迹规划，兼顾可达性、碰撞规避和沿途连续测量。

**🔧 技术方法**

使用了二维U‑Net进行CKM与误差图预测，仿真平台实现动态小尺度衰落与干扰建模，利用Voronoi分区与可达集合进行轨迹规划。

**📊 数据集**

实验数据集基于仿真生成的300m×300m城市环境，包含六个基站、八座建筑、随机小尺度衰落参数，生成10个未见场景进行评估。

**📈 对比分析**

与随机、FIFO、AoI‑only、AEM‑only、Oracle、Kriging、Diffusion等基线比较，W‑AoI在稳态MSE上比最差基线下降约4.8 dB²，接近Oracle仅差0.45 dB²，且计算耗时仅8 ms。

**⚠️ 局限性**

局限在于依赖U‑Net预测的校准质量，模型对极端动态变化的适应性有限；且实验仅在仿真环境验证，实际空中部署需考虑通信延迟与能源约束。

---

## 379. CJRR-Net: A Candidate-Joint Reprojection Ranking for Future-View Selection in Human Pose Estimation

**arXiv ID:** 2611.13700 | [PDF](https://arxiv.org/pdf/2611.13700v1)

**作者:** Huiting Xi `[一作]` (UNSW Sydney), Francisco Cruz `[通讯]` (UNSW Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在已校准的八相机阵列中，提出了候选关节重投影排名网络（CJRR-Net），通过将当前视角的3D姿态及其不确定性投影到每个候选相机，生成关节级几何特征，并与相机配置和当前观测融合，来预测下一个最优视角。

**💡 创新点**

创新点在于显式利用重投影的3D姿态和不确定性构建候选视角的关节级几何特征，并通过三分支交叉注意力结构将其与相机参数和观测信息融合，实现了无需获取候选图像即可进行视角排名。

**🔧 技术方法**

使用了预训练的Neural Localizer Fields（NLF‑S）和RootHead进行3D姿态估计，随后对姿态进行投影、协方差传播，构造12维候选特征；模型采用三分支网络（候选关节、相机配置、当前观测）和交叉注意力加权的投票头进行排名。

**📊 数据集**

评估数据集为MPI‑INF‑3DHP，使用其中的S1–S6进行训练，S7验证，S8测试，全部采用同步、已校准的八相机观测。

**📈 对比分析**

与均匀随机、几何效用规则、ActiveMoCap启发式规划器、PoseErrNet以及共享输入MLP等基线比较，CJRR-Net在测试集上实现了平均2D MPJPE 105.18像素，减少了约64.9%的误差，并在不同遮挡等级下保持最低误差，表明性能显著优于基线。

**⚠️ 局限性**

局限性包括仅在单一受训练集外的受试者（S8）上测试，且测试相机布局固定；可能存在NLF‑S预训练与S8重叠导致泛化评估受限，闭环性能仅通过可视化验证，缺乏定量闭环误差和切换频率分析。

---

## 380. Agentic AI Meets Formal Verification in the Era of Agentic Hardware Development

**arXiv ID:** 2611.13705 | [PDF](https://arxiv.org/pdf/2611.13705v1)

**作者:** Wenji Fang `[一作]` (Hong Kong University of Science and Technology), Zhiyao Xie `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述了AI在硬件形式化验证中的方法和应用，提出了从规范、证明构造到推理引擎三层技术体系，并探讨了在RTL生成、调试和优化等阶段引入形式化验证实现持续正确性反馈的可能性。

**💡 创新点**

创新点在于系统化整合AI技术与形式化验证的各层面，阐述了AI驱动的规范生成、证明自动化以及推理引擎优化；并展示了将形式化验证嵌入AI驱动硬件设计流程（生成、调试、优化）以实现“proof‑native”设计的前景。

**🔧 技术方法**

使用的大语言模型与AI代理、模型检查、定理证明工具、等价检查、E‑graph 等式饱和、SMT/ SAT 求解器等技术。

**📊 数据集**

参考了ScaleBench、RISC‑V 等多种硬件设计数据集，以及相关公开实验结果。

**📈 对比分析**

通过实验比较，例如CorrectRTL 在 ScaleBench 上将 Pass@1 从 20.5% 提升至 49.0%，Dr. RTL 在 20 份设计中平均提升 21.3% WNS、16.9% TNS 并保持 86% SEC 通过率；E‑Layout 在 9 份设计中平均提升 6.07% TNS、2.29% WNS，面积下降 1.07%。

**⚠️ 局限性**

局限性包括规范质量和覆盖不足导致的误报/漏报、根因定位和修复搜索效率、对大型复杂设计的可扩展性挑战，以及 AI 生成的方案与正式验证之间的接口完整性问题。

---

## 381. Relaxing Cross-Modal Correspondence for Multimodal Remote Sensing Pretraining

**arXiv ID:** 2611.13807 | [PDF](https://arxiv.org/pdf/2611.13807v1)

**作者:** Xianping Ma `[一作]` (Southwest Jiaotong University), Yuanxin Ye `[通讯]` (Southwest Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出在遥感多模态预训练中放宽跨模态对应关系，研究实例级对应可被类级对应替代。

**💡 创新点**

引入 SemPair 与 MixPair 对跨模态对应粒度进行可控调节，证明精确实例对应不是迁移的必要条件。

**🔧 技术方法**

使用 DFormer 等多模态编码器与对比/掩码学习相结合，并通过潜在因子模型理论分析对应粒度对跨模态相关性的影响。

**📊 数据集**

预训练使用 ImageNet‑RS90 与 NWPU‑RGBD‑45 数据集，下游分割基准为 Potsdam、Vaihingen、MMHunan 与 GeoNRW。

**📈 对比分析**

在冻结、微调、标签少量等多种迁移设置下与不同对应策略比较，SemPair 与 Exact Pairing 的 mIoU 差距≤0.2%，混合对应在多数场景表现最好。

**⚠️ 局限性**

仅在已知类标签且可构造跨实例对应的场景下验证；对需要精确空间或时序对应的任务可能不适用；真实未配对多模态档案的效果尚未验证。

---

## 382. Toward Autonomous RTL and Synthesis Optimization: Benchmarking and Self-Improving Agents

**arXiv ID:** 2611.13713 | [PDF](https://arxiv.org/pdf/2611.13713v1)

**作者:** Wenji Fang `[一作]` (Hong Kong University of Science and Technology), Zhiyao Xie `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了三条协同方向：① 真实基准（参考式与开放式）评估RTL优化；② 通过多智能体闭环框架Dr.RTL实现工具驱动、自我改进的RTL重写与优化；③ 在设计特定层面对合成流程进行脚本调优与算法演化，以获得额外的PPA收益。

**💡 创新点**

创新点包括：① 设计两种互补基准（RTL-OPT与Dr.RTL）区分可被合成工具已完成的与剩余的优化空间；② 构建可持续学习的多智能体系统，利用细粒度时序反馈、等价检查及技能库实现自我改进；③ 在合成层面实现脚本调优与基于演化的算法发现，针对单个设计定制优化策略。

**🔧 技术方法**

采用技术包括：多智能体协同（时序分析、RTL重写、评估）；细粒度时序分析与关键路径映射；等价检查（Cadence JasperGold）；商用合成工具（Synopsys Design Compiler）与开源合成（Yosys+ABC）；LLM驱动的代码生成与演化；技能学习库与经验复用；仿真/验证技术。

**📊 数据集**

使用数据集：20个原始人写Verilog设计（处理器、通信接口、缓冲、信号处理等，平均约812行）；36个手工优化对照RTL用于RTL-OPT基准；8个设计用于脚本调优与算法发现实验。

**📈 对比分析**

与基准方法比较：在商用合成下，Dr.RTL平均WNS提升21.3%、TNS提升16.9%、面积减5.8%；单次LLM提升仅4–7%；RTLRewriter与SymRTLo平均提升约4–6%；脚本调优平均WNS提升7.4%、TNS提升11.1%、面积略增1.6%；算法发现平均ADP降低18.7%，7/8设计同时改善面积与延迟。

**⚠️ 局限性**

局限性：仅在45 nm Nangate库与Synopsys DC环境评估，未验证功耗收益；对极大规模设计的可扩展性与运行时成本未知；脚本调优与算法发现仍局限于单一合成工具；缺乏完整的跨层次（架构→RTL→EDA）统一框架与验证约束定义。

---

## 383. AnchorMTS: Multi-View Signal Representation and Forecast-Anchored Fusion for Long-Term Time-Series Forecasting

**arXiv ID:** 2611.13794 | [PDF](https://arxiv.org/pdf/2611.13794v1)

**作者:** Gao Wu `[一作]` (Zhejiang University), Kexin Zhang `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

AnchorMTS在长周期时间序列预测中通过自衍生的视觉与文本视角辅助数值预测，实现了可控的多视角信息融合。

**💡 创新点**

创新点在于将辅助视角仅用于生成残差候选，并通过参数无关的投影严格限定其对最终预测的影响，分离了辅助信息的效用与控制。

**🔧 技术方法**

使用自衍生的Gramian角场、轨迹线、重现图等图像视角、GPT-4.1-mini文本生成、CLIP ViT-B/32编码器、交叉注意力融合以及L2约束投影。

**📊 数据集**

在ETTh1/2、ETTm1/2、Exchange、Weather等六个公共长周期预测基准上进行评测。

**📈 对比分析**

与七个监督式基准（包括TimePro）比较，AnchorMTS在所有24个输入-预测时长组合上取得最小宏观MSE/MAE（0.336/0.367），在18/24个MSE细胞中占优。

**⚠️ 局限性**

局限在于需固定投影阈值，且对极端异常补偿的削弱可能在某些场景下限制细粒度改进。

---

## 384. Stochastic Aggregation with Prescribed Resources and Additive Delay

**arXiv ID:** 2611.13757 | [PDF](https://arxiv.org/pdf/2611.13757v1)

**作者:** Tianhang Lu `[一作]` (Southern University of Science and Technology), Shengcai Liu `[通讯]` (Southern University of Science and Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

本文研究了在已知 Poisson 到达率的在线聚合问题，允许请求集合共享资源并产生延迟费用，并给出了一个确定性在线算法，其长期期望成本与完全先验的最优方案相比有常数上界；同时针对单资源（TCP 认可）情形给出了精确的最优长期比值，证明了 1.409 这一上限不可突破；

**💡 创新点**

创新点在于：①提出资源尺度线性规划（resource‑scale LP）将共享采购与等待成本统一下界，避免重复计费；②利用该 LP 的对偶构造嵌套时钟调度，使不同请求类型能够在不同频率下共享采购；③在单资源情形下通过潜在函数与概率分析精确求解最优期望比值，指出随机化无法改进；

**🔧 技术方法**

主要技术手段包括：线性规划与对偶分析、Poisson 过程的概率计数与插值、嵌套时钟（nested clock）调度、潜在函数（potential）证明在线下界、对单资源问题的动态规划与解析式、以及数值证明以获得常数值；

**📊 数据集**

本工作为理论研究，不使用实验数据集，而是通过严谨的概率与优化理论推导得出结果；

**📈 对比分析**

与无先验信息的离线最优方案（即 Clairvoyant 最优）进行比较，给出在线方案的长期期望成本比值：对任意资源集合≤2.113，对根型多层聚合和树上的配对聚合也同样适用；在单资源（TCP 认可）情形下，最坏负载下的比值约为1.409；

**⚠️ 局限性**

局限性在于：①对于一般资源重叠问题，理论上已知的最优比值仍在约1.408–2.110之间，尚未确定是否可进一步压缩；②算法仅适用于已知 Poisson 到达率的情形，对非平稳或未知率的需求尚无分析；③虽然证明了随机化无益，但在实际场景中可能仍有改进空间。

---

## 385. Unified Far-field and Near-field Codebook Design for XL-array Systems

**arXiv ID:** 2611.13768 | [PDF](https://arxiv.org/pdf/2611.13768v1)

**作者:** Chao Zhou `[一作]` (Southern University of Science and Technology), Liujia Yao `[通讯]` (Southern University of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种适用于远场和近场XL阵列系统的统一代码本设计，解决了传统DFT代码本在近场通信中的低效问题。

**💡 创新点**

创新点在于通过将近场信道引导向量表示为角度向量和曲率相关向量的Hadamard积，从而实现了对传统Type-I代码本的重用，并引入了轻量级的辅助代码本进行曲率量化。

**🔧 技术方法**

使用了Hadamard积的数学表示法来优化代码本设计，并通过最大化平均波束形成增益来优化辅助曲率代码本。

**📊 数据集**

使用了一个模拟的XL阵列系统，用户均匀分布在一个特定区域内，考虑了不同的用户范围和角度。

**📈 对比分析**

与多种基准方案进行比较，提出的统一代码本在波束形成性能上接近完美信道状态信息（CSI）的上限，同时显著降低了所需的反馈开销。

**⚠️ 局限性**

限制在于该方法在高分辨率情况下的优化问题可能较为复杂，且在实际应用中可能需要进一步的调整以适应不同的用户分布和信道条件。

---

## 386. DisasterAgent: An Evidence-Driven Agent for Verifiable Disaster Report Generation From Remote Sensing Imagery

**arXiv ID:** 2611.13776 | [PDF](https://arxiv.org/pdf/2611.13776v1)

**作者:** Zhijian Li `[一作]` (Aerospace Information Research Institute, Chinese Academy of Sciences), Kun Fu `[通讯]` (Aerospace Information Research Institute, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于感知-增量-生成工作流的灾害信息检索代理（DisasterAgent），能够从双时遥感影像生成可验证的灾害报告并附上证据引用。

**💡 创新点**

创新点在于：①构建结构化的时空感知图（STPG）以统一存储感知结果；②设计多源证据增量（SCE、VGE、EEE）实现语义验证、像素级定位与历史经验检索；③通过两阶段生成与引用闭环确保报告可追溯性。

**🔧 技术方法**

核心技术包括：预训练多模态语言模型（Qwen3.5-27B）作为思考者；工具调用框架（Plan-and-Execute）实现视觉专家模型集成；STPG结构与多源证据抽取；基于引用的可验证生成策略。

**📊 数据集**

使用了DisasterM3（包含十类灾害的灾情描述与恢复建议）和RSCC（双时图像变化字幕）两大公开数据集。

**📈 对比分析**

在RSCC上相较于基线VLM和其他遥感代理，ROUGE‑L、METEOR和ST5‑SCS分别提升4.08%、3.77%和15.34%；在DisasterM3上平均报告评分提升13.5%，灾害描述提升19.8%，恢复建议提升7.2%。

**⚠️ 局限性**

局限性包括：目前仅支持光学影像，无法处理SAR或多时序数据；依赖离线预训练工具的准确性；模型对异常或极端场景的泛化能力尚待验证。

---

## 387. Decision-Focused Learning via Neural Optimization Layers for Battery-Assisted Grid-Import Smoothing in AI Data Centers

**arXiv ID:** 2611.13694 | [PDF](https://arxiv.org/pdf/2611.13694v1)

**作者:** Xinda Zheng `[一作]` (Monash University), Markus Wagner `[通讯]` (Monash University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于神经优化层的决策聚焦学习框架（BS-DFL），用于AI数据中心的电池辅助电网进口平滑。

**💡 创新点**

创新点包括：①直接从调度目标学习神经优化层，免除最优解标签；②结合情景感知编码与顺序约束感知解码，实现可行决策的离线训练；③冻结神经层后引导预测器学习，实现无在线优化的滚动时域调度。

**🔧 技术方法**

使用了深度学习（MLP、注意力编码器+解码器）、神经优化层、CVaR风险度量、两阶段随机调度及滚动时域方法。

**📊 数据集**

使用真实AI工作负载测得的10MW数据中心负载概况，生成10,000分钟的仿真轨迹进行评估。

**📈 对比分析**

与传统MSE训练+Gurobi混合整数规划相比，BS-DFL平均网格进口变化降低71.70%，总调度成本降低18.01%，并以毫秒级（平均18.77 ms）完成决策。

**⚠️ 局限性**

局限在于仅针对单一电池规模与零上升下限情景，情景数量受历史负载可用性限制，未验证不同电价、不同电池配置的通用性。

---

## 388. When Covariance Is Not Enough: Kernel-Based Regularization for Single-Cell Counterfactuals

**arXiv ID:** 2611.13748 | [PDF](https://arxiv.org/pdf/2611.13748v1)

**作者:** Korbinian Traeuble `[一作]` (Helmholtz Zentrum München), Kipper Fletez-Brant `[通讯]` (Genentech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了单细胞RNA‑seq的反事实编辑，提出在概念瓶颈生成模型中加入非线性Hilbert‑Schmidt独立性准则（nHSIC）来更好地分离已知概念与未知因素，从而实现对个体特异性细胞表达的精确预测。

**💡 创新点**

创新点在于：①首次发现仅用线性交叉协方差惩罚无法捕捉条件对高阶（方差）交互的影响；②将nHSIC与传统线性惩罚联合使用，使模型能够在非线性/异方差情境下恢复正确的个体特异性编辑；③通过模拟与真实Taurus数据验证该方法显著提升了编辑的个体准确性。

**🔧 技术方法**

核心技术包括：概念瓶颈变分自编码器（scCBGM）架构、交叉协方差正则化、nHSIC核正则化、MSE重构损失、KL散度、BCE概念损失、正则化权重调节与梯度裁剪。

**📊 数据集**

数据集：①仿真单细胞RNA‑seq（240k细胞，150个供体，4个细胞类型，3k基因，低秩供体结构）；②Taurus人类结肠单细胞RNA‑seq（约987k细胞，7个主要细胞谱系，二元组织概念）。

**📈 对比分析**

比较方法：使用标准化最大均方差（rMMD）和Sinkhorn距离对编辑结果进行评估；实验显示：在线性平均位移情境下，单纯交叉协方差即可得到rMMD≈0.06；在异方差情境下，单交叉协方差导致rMMD≈1.30；加入nHSIC后，rMMD下降至0.35，并且编辑比例提升到0.67（相比仅交叉协方差的0.04/0.22）。

**⚠️ 局限性**

局限性：①对正则化权重高度敏感，需同时考察编辑比例和距离比值；②易出现过度正则化导致模型脱离数据流；③实验仅在PCA空间且使用单一随机种子；④仅在一个模拟器和一个真实数据集上验证，缺乏跨平台或多组学推广；⑤需要预先标注的概念，限制了对未知或复杂多维因素的处理。

---

## 389. Topological Dynamics Under Non-Stationary Supervision: Persistent Homology of Neural Network Weight Evolution

**arXiv ID:** 2611.13701 | [PDF](https://arxiv.org/pdf/2611.13701v1)

**作者:** Mitra Varun `[一作]` `[通讯]` (Independent Researcher), Mitra Varun (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了在正确标签、固定随机标签以及每个 epoch 随机标签三种监督条件下，输入权重点云的 Vietoris–Rips 持久同调演化，并与训练诊断指标相关联。

**💡 创新点**

提出了基于多种标签监督的对照实验框架，系统评估了持久同调在权重空间的可解释性与预测价值，并首次尝试在早期训练阶段用拓扑特征区分监督条件。

**🔧 技术方法**

使用了 Vietoris–Rips 持久同调（H0 与 H1）与 Ripser、Persim 计算 Wasserstein 距离，配合原始与行归一化的云，采用 exact sign‑flip 与 Holm 校正的统计检验，以及多特征的逻辑回归早期分类。

**📊 数据集**

采用 Fashion‑MNIST 数据集（10 类 28×28 灰度图像）进行训练和评估。

**📈 对比分析**

通过 5 种种子、22 个检查点的 15 个模型，比较权重云的持久同调指标与损失、梯度范数、验证准确率等训练诊断，并尝试用早期拓扑特征预测监督条件；结果显示拓扑特征未能显著提升分类性能，均衡准确率仅为 60%（拓扑）与 66.7%（基线）。

**⚠️ 局限性**

实验仅覆盖 5 个种子、单一 MLP 架构和数据集、H1 维度受限、缺乏完整距离覆盖和瓶颈分析、样本量小导致统计功效不足，限制了结论的推广性。

---

## 390. Plug-and-Play ADMM for Inverse Problems with Flow Matching Denoiser

**arXiv ID:** 2611.13742 | [PDF](https://arxiv.org/pdf/2611.13742v1)

**作者:** Dibyanshu Kumar `[一作]` (Technische Hochschule Würzburg-Schweinfurt), Magda Gregorova `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于 ADMM 的 Plug‑and‑Play 框架，结合流匹配生成先验并引入时间自适应轨迹校正，实现更一致、高效的逆问题求解。

**💡 创新点**

创新点在于通过辅助变量与双重更新增强测量一致性，并设计随机与确定性投影混合的轨迹校正机制，解决流匹配先验与数据一致性步之间的漂移问题。

**🔧 技术方法**

采用了 ADMM 优化、流匹配 ODE、时间自适应投影、随机/确定性混合投影以及共轭梯度求解数据一致性子问题等技术。

**📊 数据集**

使用了 CelebA（128×128）和 AFHQ‑Cat（256×256）图像数据集。

**📈 对比分析**

与 OT‑ODE、D‑Flow、Flow‑Priors 和 PnP‑FBS 等方法比较，在噪声、去块、超分等 6 类逆问题中，该方法在 PSNR/SSIM/LPIPS 上排名第一或第二，并且在运行时间和显存消耗上均显著低于对手。

**⚠️ 局限性**

缺乏对 PnP‑ADMM 与流匹配先验的理论收敛性证明，算法的稳定性仅凭经验验证。

---

## 391. Softness is something you also need! Why compliance helps control: information metrics of a passive walker with compliant ankle

**arXiv ID:** 2611.13779 | [PDF](https://arxiv.org/pdf/2611.13779v1)

**作者:** Fabio Bonsignorio `[一作]` (University of Zagreb), Enrica Zereik `[通讯]` (National Research Council)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过计算可逆走步机的相空间体积与可行性概率分布，量化柔性（顺应性）对主动行走系统的形态计算效应，并提出使用再生核希尔伯特空间（RKHS）构建分布式复杂度度量来衡量控制器面临的信息处理负担。

**💡 创新点**

创新点在于：①将传统的相空间体积与信息熵指标统一到形态-动力-信息三位一体的Mind–Body Nexus框架；②通过可行性概率分布的RKHS嵌入与协方差算子，提出分布式复杂度C(P)=log_F(I+γC_P)，揭示柔性仅改变概率分布而非几何支撑的事实；③将蒙特卡罗与核方法相结合，提供高维可行域体积与复杂度的数值估计。

**🔧 技术方法**

主要技术包括：相空间体积积分、动力学的能量约束、可行域的蒙特卡罗采样、可行域后的冲击映射、再生核希尔伯特空间的特征映射与均值嵌入、核协方差算子以及弗罗贝尼乌斯行列式（Fredholm determinant）计算分布式复杂度。

**📊 数据集**

数据集：论文仅使用仿真生成的状态样本（如二维、四维能量椭圆或球体内的均匀采样），无真实实验数据。

**📈 对比分析**

通过对比刚性与柔性两种行走模型，使用可行相空间体积W与RKHS分布式复杂度C(P)进行评估，结果显示柔性模型在保持相同几何支撑的前提下，C(P)显著降低（从约11.9下降至9.9），表明柔性可通过改变概率分布降低控制器的信息处理负担。

**⚠️ 局限性**

局限性包括：①假设能量函数可分离且遵循欧氏几何，实际复杂机器人可能违背此假设；②分布式复杂度与控制器资源之间缺乏直接映射，需要进一步研究；③高维系统下蒙特卡罗采样与核协方差算子求解的计算成本高；④冲击映射和可行域定义的简化（如理想塑性冲击）可能与真实物理不符。

---

## 392. RiSE-Drive: Risk-Sensitive Long-Horizon Candidate Evaluation for End-to-End Autonomous Driving

**arXiv ID:** 2611.13790 | [PDF](https://arxiv.org/pdf/2611.13790v1)

**作者:** Yu Zhao `[一作]` (Zhejiang University), Xiubo Liang `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 RiSE-Drive，一个风险敏感的长时程候选评估框架，用于端到端自动驾驶。

**💡 创新点**

创新点是将候选质量建模为未来条件相关的长时程结果分布，并利用 CVaR 与跨条件方差进行风险敏感排序，而非单点分数。

**🔧 技术方法**

使用技术包括未来条件生成、候选条件化潜在后果传播、分布量化的分位数预测以及基于 CVaR 的排序；基础框架基于 DrivoR 的 ViT‑S 感知与提议模块。

**📊 数据集**

数据集：NAVSIM‑v1、NAVSIM‑v2、HUGSIM 以及 NavSafe 诊断集。

**📈 对比分析**

与 DrivoR 以及多种先进端到端规划器在同一视觉后端进行对比，结果在 PDMS、EPDMS 和 HD‑Score 上分别提升约 1–3 点，尤其在硬难、扰动和闭环评估中显著优于对手。

**⚠️ 局限性**

局限：仍依赖候选生成质量，评估开销略高；对未来分支数量和分位数设置敏感，可能对超大模型或实时部署有限制。

---

## 393. Local Modifications of Nearest-Neighbor and Min-Plus Representations

**arXiv ID:** 2611.13787 | [PDF](https://arxiv.org/pdf/2611.13787v1)

**作者:** Liliia Redina `[一作]` `[通讯]` (HSE University), Liliia Redina (HSE University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文研究了在有限域上对布尔函数进行有限个本地修改时，最近邻（NN）和整数最小-加（min-plus）表示的大小与数值精度如何变化，并给出了最优的增量上界；

**💡 创新点**

创新点在于首次证明任意有限域上NN表示增量上界为+2d、布尔立方体上为+3d，并构造了同时修改所有变化点的几何算法；同时给出了整数min-plus表示在布尔立方体上+ d 增量、权重提升 ≤2n 的结果，并在任意有理域上给出多项式权重上界；

**🔧 技术方法**

主要技术包括几何扰动与圆锥分割、全局有理化与公共dyadic网格化、整数affine形式的投影排序、以及对距离差的精细上界分析；

**📊 数据集**

该研究未使用实验数据集，而是针对任意有限域和布尔立方体的理论分析；

**📈 对比分析**

通过比较原始与修改后表示的最小尺寸，证明了所给增量上界是最优的（常数 2 或 3 为最佳），并给出了在多项式输入规模下的多项式时间可构造性；

**⚠️ 局限性**

局限性在于仅处理完全准确的表示，未讨论近似或学习场景；对NN在布尔立方体上的常数 2 是否可进一步改进仍开放；此外对有理坐标的最优性缺乏完整证明。

---

## 394. The Target Is a Predicate: Predicate-State Grounding for Semantic-Level Video Reasoning Segmentation

**arXiv ID:** 2611.13785 | [PDF](https://arxiv.org/pdf/2611.13785v1)

**作者:** Yuhuan Hong `[一作]` (Tsinghua University), Yujiu Yang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Predicate‑State Video Reasoning Segmentation (PS‑VRS) 任务，构建了包含 2077 视频、16250 查询的 PS‑Bench 数据集，并基于 MLLM 与 SAM2 开发了 Predicate‑State Grounding (PSG) 方法。

**💡 创新点**

创新点在于把自然语言查询视为随时间变化的谓词，显式建模 active / vacant 状态，并通过 State‑Bind‑Emit 管线将语义有效性与掩码生成分离，从而显著减少无效掩码的产生。

**🔧 技术方法**

技术上使用多模态大型语言模型（MLLM）生成 predicate‑state tokens，利用密集跨模态相似度对齐实现时间绑定，并将 active 状态对应的 mask‑emit token 投影至 SAM2 进行条件掩码解码。

**📊 数据集**

主要数据集为自建的 PS‑Bench（约 1.57M 像素级掩码），并在公开的 MeViS、Ref‑YouTube‑VOS 等数据集上进行跨数据集验证。

**📈 对比分析**

方法与大型 MLLM、历史 Video Reasoning Segmentation (VRS) 模型做对比。PSG 在 PS‑Bench 上 mIoU 提升 +9.32，J&F 分数大幅提升，False‑Active/False‑Vacant 误报率分别下降至 19.70%/12.09%，在 4B 参数规模下实现最佳性能。

**⚠️ 局限性**

局限性包括：仅以 1 fps 采样视频导致时间边界粗化，段落准确率低（段落匹配仅 40.57%/30.50%），无法精准捕捉快速或模糊的状态变化；模型规模受限，尚未探索更大 MLLM、长视频上下文或更强掩码解码器的潜在提升。

---

## 395. AI-Adaptive Virtual Reality for STEM: Engagement, Retention, and Learning Outcomes

**arXiv ID:** 2611.13774 | [PDF](https://arxiv.org/pdf/2611.13774v1)

**作者:** Srinivasa K G `[一作]` (IIIT Naya Raipur), Kshitij Kumar Singh Chauhan `[通讯]` (IIIT Naya Raipur)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了基于Meta Quest 2的AI自适应VR太阳系学习模块，并比较其与标准VR及传统教学的学习效果和参与度。

**💡 创新点**

将机器学习模型实时根据学习者的停留时间、注视模式和响应时延动态调整难度，实现了实时自适应的VR教学。

**🔧 技术方法**

采用Unity引擎开发VR环境、Meta Quest 2硬件、眼动追踪与心率变异性监测，以及基于这些日志训练的机器学习模型。

**📊 数据集**

采用120名印度中学生的交互日志和学习测验数据，构成实验数据集。

**📈 对比分析**

通过配对t检验、一元ANOVA和结构方程模型比较，AI自适应VR在知识提升、长期保留和生理参与度上均显著优于标准VR和传统教学。

**⚠️ 局限性**

样本规模有限、仅聚焦于太阳系内容、缺乏多学科验证以及VR硬件快速更新可能影响结果的普适性。

---

## 396. RF3DGS: Reflection-Free 3D Gaussian Splatting with Self-Priors

**arXiv ID:** 2611.13772 | [PDF](https://arxiv.org/pdf/2611.13772v1)

**作者:** Qiang Wen `[一作]` (Hong Kong University of Science and Technology), Qifeng Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种仅使用单目sRGB视频的3D Gaussian Splatting方法（RF3DGS），通过光度先验与自适应双渲染损失实现高效的3D反射去除。

**💡 创新点**

创新点包括：①不依赖闪光/RAW等额外拍摄技巧；②利用多视角最小光度值构造反射自由伪标签；③引入自适应双渲染（ATR）损失平衡原始与伪标签监督；④基于Unity构建45案例、涵盖多种透明度、厚度、反射强度的3D反射去除基准。

**🔧 技术方法**

采用技术：3D Gaussian Splatting、SfM点云初始化、几何先验深度平滑、光度先验深度warp、ATR损失、互斥损失、双向光度约束，以及Unity渲染引擎生成基准数据。

**📊 数据集**

使用的数据集：基于Unity渲染的45案例基准（每个案例包含多种玻璃透明度、厚度、反射强度）以及多幅真实世界sRGB视频。

**📈 对比分析**

对比方法：MVSNeRF、TNeRF、NeRFReN、Vanilla 3DGS、DSRNet+3DGS、Flash-Splat；在基准上RF3DGS取得PSNR 31.45、SSIM 0.915、LPIPS 0.150，训练时间仅60.74 min，显著优于NeRFReN的63 h；在真实场景中亦表现出色。

**⚠️ 局限性**

局限性：光度先验伪标签可能带有warp误差，深度估计对精度要求高；极端视角变换或光照不均时性能可能下降；未对动态场景或多视角RAW/闪光条件进行评估。

---

## 397. From Depth to Safe Depth: Uncertainty-Aware Monocular Perception for Mobile Robot Navigation

**arXiv ID:** 2611.13769 | [PDF](https://arxiv.org/pdf/2611.13769v1)

**作者:** Saeid Firouzi Daghigh `[一作]` (Payame Noor University), Saeed Ayat `[通讯]` (Payame Noor University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `f86bf285-fd08-4156-973b-6e6481af8fa0` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在移动机器人导航中，作者提出了一种轻量级插件模块，将冻结的单目深度基础模型（Depth Anything V2）的预测深度校正为安全深度，并通过不确定性估计和保守下界实现对机器人碰撞的有效抑制。

**💡 创新点**

创新点在于：①仅使用一个0.58 M参数的U‑Net头部，在冻结的深度模型上学习像素级的对数深度校正 μ 与不确定性 σ；②采用 β‑NLL 训练使 μ 捕捉系统性误差、σ 反映随机误差；③使用分位数校准的 conformal 预测确定安全系数 k，从而生成单侧下界 d̂_safe；④针对向下摄像头导致的“幻影地面”问题，引入基于不确定性的地面判别和修正。

**🔧 技术方法**

主要技术包括：深度基础模型（Depth Anything V2），轻量级 U‑Net 校正头，β‑加权 Gaussian NLL 损失，分位数 conformal 校准，基于不确定性的地面测试与成本地图集成，ROS 2 Nav2 传统栅格规划器。

**📊 数据集**

使用的主要数据集为：通过程序化生成的室内世界（包含走廊、房间、家具），其中包含三种硬案例（玻璃隔板、无纹理白墙、暗区）。训练采集了 7,045 帧（4 个世界），验证与校准使用 1 个世界，测试使用 4 个未见过的世界。

**📈 对比分析**

比较方法包括：原始深度（Raw）、加固定比例边距（Margin）、仅校正（μ）、以及完整的安全深度（Safe）。在 120 条配对测试情节中，Safe 的成功率 70%（95% CI [61%,77%]）明显高于 Raw 37%；碰撞率从 Raw 的 63% 降至 Safe 的 3%，仅比真实深度（2%）差距很小；Margin 虽降低碰撞率但成功率仅 30%，显示仅均匀留边并非有效策略。

**⚠️ 局限性**

主要局限包括：所有实验均在 Gazebo 模拟环境中进行，缺乏真实机器人与真实图像的验证；只测试了单一机器人与单一摄像头配置，未考虑定位误差；地面测试假设平坦地面，无法处理不规则地面；模型在 1 Hz 的推理速度下限制了机器人运动速度；未结合 epistemic 不确定性，可能在未见环境中仍出现未校正的系统性误差。

---

## 398. FAME-QP: A GPU-Accelerated Quadratic-Programming Solver for Real-Time Model Predictive Control

**arXiv ID:** 2611.13766 | [PDF](https://arxiv.org/pdf/2611.13766v1)

**作者:** Santosh Mohan Rajkumar `[一作]` (Ohio State University), Stephanie Stockar `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为 FAME-QP 的 GPU 加速求解器，用于在线线性 MPC 中出现的密集二次规划问题，融合了约束依赖的算子表示、CUDA 图回放和惩罚参数探测等技术，并支持惰性矩阵更新。

**💡 创新点**

创新点在于：①针对不同约束结构设计约束依赖的算子，显著减少存储与迭代时间；②融合单核与双核迭代并利用 CUDA 图自动重放；③通过惩罚银行探测自适应初始惩罚并自动选择执行模式；④惰性更新操作矩阵以适应在线矩阵变化；⑤自动精度与设备配置选择。

**🔧 技术方法**

采用 CUDA 核函数融合、CUDA 图、Ruiz 等价化缩放、惩罚银行与探测、惰性矩阵更新、自动执行模式与精度选择等技术，全部在 NVIDIA GPU 上实现。

**📊 数据集**

使用随机密集 QP 数据集、随机不稳定线性 MPC 系统（随机动态与预测期）、Unitree G1 人形机在 MuJoCo 中的推力恢复实验，以及 Skydio X2 四旋翼在 MuJoCo 中的轨迹跟踪实验。

**📈 对比分析**

与 CPU 传统求解器（OSQP、PIQP、ProxQP、Clarabel、DAQP）以及 GPU 求解器（cuOSQP、ReLU‑QP、qpax、CuPIQP、CUDA MPC、πMPC）进行对比。实验显示，在随机 QP 上相较 DAQP 提升 1.2–25.9 倍、相较 ReLU‑QP 提升 1.5–4.3 倍；在 LMPC 关节控制闭环步骤中，相较 ReLU‑QP 和 πMPC 分别提升 3–13 倍，同时保持相同或更优的控制性能；且在实时控制周期内始终保持低延迟。

**⚠️ 局限性**

主要局限在于仅针对密集 QP，稀疏或高度结构化的问题尚未覆盖；在极大规模问题或长预测期时，算子存储与计算仍可能成为瓶颈；惩罚探测与设置阶段存在额外成本；实现依赖 CUDA，需 NVIDIA GPU。

---

## 399. List Update with Per-Page Maximum Delay

**arXiv ID:** 2611.13750 | [PDF](https://arxiv.org/pdf/2611.13750v1)

**作者:** Tianhang Lu `[一作]` (Southern University of Science and Technology), Shengcai Liu `[通讯]` (Southern University of Science and Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种基于列表的延迟访问模型，允许一次访问清除同一页面的所有等待请求，并给出了一个确定性阈值规则的算法及其分析；

**💡 创新点**

核心创新在于异步标记‑秩耦合（marked‑rank coupling）和时间传输原理（temporal transfer principle），实现了对不同列表顺序与不同服务历史的统一比较，并将这些技术推广到带最大延迟、硬期限、异构请求权重等多种延迟函数；

**🔧 技术方法**

采用离散时间事件模拟、逆序标记与逆序潜能函数、分层动态规划、图搜索以及超加性不等式等技术；

**📊 数据集**

该工作为纯理论论文，未使用任何实验数据集，仅通过算法分析与可计算性证明来评估性能；

**📈 对比分析**

通过竞争分析，阈值规则在最坏情况下取得严格的 5‑竞争比；在一页面情形下证明最佳比例为 2；对固定顺序的最优方案与动态调度之间存在 Ω(m) 的性能差距；还证明了 NP‑完全性与固定参数可计算性；

**⚠️ 局限性**

局限性包括：阈值规则的最佳竞争比上限为 10/3，无法处理非超加性（如凹形）延迟或无限制的请求特定权重；随机化规模的平均分析未能突破常数 5；对非统一页面函数的多样化延迟缺乏统一的竞争保证。

---

## 400. Broad-Persistent Advising in Interactive Reinforcement Learning via Latent State Encoding

**arXiv ID:** 2611.13712 | [PDF](https://arxiv.org/pdf/2611.13712v1)

**作者:** Zhuoran Liang `[一作]` (UNSW Sydney), Francisco Cruz `[通讯]` (UNSW Sydney)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出Encoder BPA框架，用神经网络编码器替代K‑means，实现在高维状态下的建议检索与重用，提升强化学习早期训练效率。

**💡 创新点**

创新点在于：①用表示学习生成的潜在空间而非离散聚类来泛化状态；②结合余弦相似度、top‑k聚合与一致性阈值的三重检索机制；③在连续动作与部分可观测环境中实现了持久建议的高效重用。

**🔧 技术方法**

核心技术包括：LSTM/Mamba编码器的监督预训练、潜在空间余弦相似度检索、PPR概率重用策略、top‑k聚合与一致性阈值判定。

**📊 数据集**

实验数据集涵盖四种环境：Cart‑Pole、Inverted Pendulum、MuJoCo Ant 与部分可观测的 POMDP Ant；每种环境下的教师网络已预训练。

**📈 对比分析**

通过与基础RL、非持久IntRL、持久IntRL、原始BPA等对照，使用平均奖励与建议重用次数衡量；结果显示Encoder BPA在多数环境的早期阶段均显著提升奖励和建议利用率，但在后期连续动作环境中可能因平均策略导致性能下降。

**⚠️ 局限性**

主要局限：①阈值（相似度、一致性）对重用频率和性能高度敏感；②在连续动作空间中平均策略易引入不恰当建议；③额外的编码器计算开销；④不同编码器实现会导致检索质量差异。

---

## 401. Word-Level Alignment of OCR and Ground Truth under Zoning and Reading-Order Errors

**arXiv ID:** 2611.13752 | [PDF](https://arxiv.org/pdf/2611.13752v1)

**作者:** Fonseca Cacho `[一作]` (University of Nevada, Las Vegas), Jorge Ramón `[通讯]` (University of Nevada, Las Vegas)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

重新评估OCR后处理中文本与真值对齐的“分区问题”，并提出基于唯一词和n-gram锚点、局部匹配以及块-窗口分配的多区域对齐器；

**💡 创新点**

在分区错误导致的阅读顺序颠倒场景下，提供一种既能利用锚点也能恢复大范围块位移的对齐方案，并通过实验验证其优越性与局限；

**🔧 技术方法**

实现了多种对齐算法（2019位置感知、重新锚定、RETAS-lite、genalog RETAS、Java RETAS及自研多区域对齐器），并用窗口搜索、编辑距离、倒排索引等技术；

**📊 数据集**

使用合成分区错误数据、真实荷兰报纸KB IMPACT的PAGE格式真值、Tesseract 5 OCR、以及包含人工单词框的古印刷品数据集进行评测；

**📈 对比分析**

与多种基线对齐器比较，发现多区域对齐器在合成分区错误上F1最高（≈0.98），在长文档上速度更快；2019位置感知方法在锚点稀缺的样例（如 boilerplate）上表现最佳；整体F1平均约0.6-0.9，取决于数据集与错误类型；

**⚠️ 局限性**

主要限制包括：基于几何生成的真值不完美且误差高；使用的文章顺序约定可能导致评估偏差；缺乏现代多列页面的完整单词级真值；多区域对齐器在文档长度上呈二次复杂度，v2未能完全达到预期改进；TREC-5的分区标注不可再现；

---

## 402. Effects of Laser Beam Width and Measurement Noise on Tree Diameter Estimation in Laser Scanning

**arXiv ID:** 2611.13739 | [PDF](https://arxiv.org/pdf/2611.13739v1)

**作者:** Jesse Muhojoki `[一作]` (Finnish Geospatial Research Institute), Juha Hyyppä `[通讯]` (Finnish Geospatial Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a7d414a-27d1-4de0-aac0-e554088edeb4`

**🎯 论文内容**

本文通过仿真与实测相结合，系统研究了移动激光扫描（MLS）中激光束发散和测距噪声对树干直径估计的影响，并提出了一种基于仿真的无标定偏差补偿方法，显著降低了树干曲线估计误差；

**💡 创新点**

创新点在于：①将激光束宽度和异方差噪声等因素纳入仿真模型，解释了不同研究中出现的矛盾偏差；②构建了只依赖传感器规格的无标定偏差补偿表，实现对大于10 cm树干的高精度修正；③通过模拟表实现快速在线补偿，避免了昂贵的现场标定；

**🔧 技术方法**

使用技术包括：移动激光扫描（Faro Orbis）与地面激光扫描（Riegl VZ‑400i）点云获取；基于RANSAC与“hyperaccurate”圆拟合提取树干弧线；滑动样条与离群值滤波处理树干曲线；仿真模块（Gaussian beam + heteroscedastic噪声）生成偏差表；以及基于距离与直径加权的补偿算法；

**📊 数据集**

数据集涵盖：①9块芬兰Evo地区的32 × 32 m样本样地（共计约9 000根树）；②每块样地在同一时段内使用Faro Orbis MLS与Riegl VZ‑400i TLS采集点云；③人工圆柱目标（210 mm直径）在实验室不同距离和不同反射率下的扫描；

**📈 对比分析**

与无补偿MLS、已发布的基线方法以及TLS参考结果进行比较。补偿后，树干曲线RMSE从约14 cm降至约5.7 cm，DBH RMSE从约8.1 cm降至约5.1 cm；相对误差提升约30–50%，逼近TLS级别的精度；

**⚠️ 局限性**

局限性包括：①对直径<10 cm的小树偏差曲线高度敏感，受树皮粗糙度、树干外形不圆形等影响；②仿真依赖传感器规格与触发阈值，若实际值偏差会影响补偿精度；③未考虑激光束椭圆形变化和扫描器旋转导致的取向差异；④低成本传感器虽受益，但仍受大束宽与高噪声限制。

---

## 403. SeqFold: Timestamp-Addressable Compressed States for User Action Sequence Backbones

**arXiv ID:** 2611.13704 | [PDF](https://arxiv.org/pdf/2611.13704v1)

**作者:** Mianwei Zhou `[一作]` (Coupang), Tao Cheng `[通讯]` (Coupang)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种SeqFold接口，将冻结的长历史Transformer backbone的多层状态压缩为时间可寻址的离散代码，以供离线后填充和在线服务使用。

**💡 创新点**

创新点在于将多通道门控融合、通道特定时序衰减与Lookup-Free Quantization结合，形成可压缩的时间戳对齐表示，并通过流式融合实现稀疏导出。

**🔧 技术方法**

使用Transformer backbone、门控聚合、指数衰减、LFQ量化、交叉注意力解码以及自定义流式内核实现稀疏导出。

**📊 数据集**

使用公司内部电商平台的用户行为序列数据，平均长度约1700，最大2000。

**📈 对比分析**

与全序列、窗口池化、采样、最近64等压缩基线对比，SeqFold在存储上比全序列压缩3,400倍，性能仅低2.11%PR-AUC；在搜索和查询自动完成任务上提升约0.6%；在稀疏导出时加速19×，内存占用大幅下降。

**⚠️ 局限性**

局限在于依赖冻结的backbone，无法自适应不同下游任务的压缩比例；LFQ在过度量化时可能丢失信息；实验基于私有数据，难以复现；对在线实时延迟的评估有限。

---

## 404. Hint, Not Hole: Localization-Robust Video Overlay Removal

**arXiv ID:** 2611.13711 | [PDF](https://arxiv.org/pdf/2611.13711v1)

**作者:** Dongyang Zhang `[一作]` (Malanshan Audio & Video Laboratory), Zhiwei Huang `[通讯]` (Malanshan Audio & Video Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种视频覆盖去除方法HintErase，利用完整视频和“提示”通道（非缺口）训练模型，结合流式管线实现全长影片的去除；

**💡 创新点**

1）把遮挡定位误差视为提示而非缺口，使模型既能补全缺失的覆盖，又能保留被误提示的真实场景文字；2）采用三区域损失对比真实覆盖区域与提示误差进行训练；3）构建可在固定显存下处理完整影片的流式管线；

**🔧 技术方法**

基于视频扩散变换器的预训练模型，在VAE潜在空间中微调；使用提示编码器对提示进行下采样编码；三区域（完整覆盖、提示覆盖与提示背景）流匹配损失；以及多步流式推理、去交错、剪辑分段、补帧和自适应粘贴回退等后处理技术；

**📊 数据集**

训练数据为合成覆盖的清晰视频（随机叠加karaoke歌词、标题卡、字幕、logo等），真实覆盖的目标已知；测试集来自许可的真实作品（karaoke、动画、音乐视频等），提示由字符级字幕分割器和logo/标题卡检测器生成，并加入随机误差；

**📈 对比分析**

通过对同一视频使用五种不同提示（字符掩码、腐蚀掩码、缺失笔划、填充框、检测框）进行比较，展示模型能在所有提示下去除覆盖，优于传统“缺口-擦除”方法；另外对完整影片进行流式推理，展示无缝无闪烁的结果；由于实验主要为定性展示，未给出数值指标，但视觉评估显示显著提升；

**⚠️ 局限性**

1）仍依赖提示的质量；过于粗糙或过多误提示可能导致漏去或残留；2）对低分辨率或极小文本的恢复效果有限；3）生成式上采样可能引入轻微色差或细节偏差；4）未在大规模自动化评测上验证，主要以人工观察为准；5）在误用场景下可能被用于去除版权水印。

---

## 405. Exploring Synchrony for Communication in Modular Neural Networks

**arXiv ID:** 2611.13759 | [PDF](https://arxiv.org/pdf/2611.13759v1)

**作者:** Nikolas Willems `[一作]` (Imperial College London), Björn W. Schuller `[通讯]` (Imperial College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出SyncNet，一种利用相位同步来协调模块通信的神经网络架构，旨在提升模块化网络的组合泛化能力。

**💡 创新点**

将同步机制与模块化分离，并用相位对共享通道进行地址化，首次证明同步可作为模块间通信的有效绑定方式。

**🔧 技术方法**

采用球面Kuramoto振荡器实现相位动力学，结合共享通道的相位对齐通信、GRU模块更新以及可学习的相位耦合。

**📊 数据集**

在Sort-of-CLEVR（关系推理）和SQOOP（系统性泛化）两个合成数据集上进行实验。

**📈 对比分析**

与六种基线（CNN+MLP、Conv+LSTM、Relation Network、FiLM、Transformer、Shared Workspace）比较，SyncNet在SQOOP低组合量设置下获得最高的离群准确率（相对领先约5个百分点），在Sort-of-CLEVR上表现接近最强模型。

**⚠️ 局限性**

局限性包括仅在小型诊断任务验证，缺乏大规模验证；功能模块化评估不完整；同步机制是否优于更通用的路由尚未确定。

---

## 406. Beyond Interaction Scores: Selecting Feature Interactions via Independent Replication

**arXiv ID:** 2611.13765 | [PDF](https://arxiv.org/pdf/2611.13765v1)

**作者:** Jincheng Gao `[一作]` (Hong Kong University of Science and Technology), Shuo Sun `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种名为LENS的模型无关框架，用来识别和筛选特征交互项。它先用两个独立训练的模型分别计算每一对特征的残差表面，并通过投影到加性表面之外得到交互能量；随后采用复制测试（replication test）结合能量排序和阈值（相对边际η和规模底阈κσ）来决定哪些交互项真正具有统计显著性，从而得到一份低误发现率、较高检索率的交互列表。

**💡 创新点**

创新点主要包括：
1) 将残差能量（Δ）与投影得到的能量比率作为交互的重要性度量，证明其在所有模型（树、网络等）中都能保持一致且不受主效应影响；
2) 通过对两个独立样本拟合得到的交互表面做交叉内积，消除自相关噪声，从而实现对交互的稳定复制检测；
3) 引入规模底阈κσ来过滤弱交互，保证误发现率控制的同时不失去重要交互；
4) 采用Benjamini–Yekutieli多重检验修正，提供在任意相关性下的FDR控制。

**🔧 技术方法**

技术实现：
- 残差表面计算与投影到加性表面的正交分解；
- 交叉内积与平均表面得到复制比率；
- 两样本分割（Discovery、Confirmation、Evaluation）保证独立性；
- 统计阈值与Benjamini–Yekutieli步骤上升检验；
- 采用GPU并行预测快速完成数十万交互的能量与复制统计。

**📊 数据集**

数据集：
- 合成函数：F1–F10（32个交互），Friedman基准，Additive与Sparse设计；
- 真实数据：
  * freMTPL2（68万条保险单，9个评级因子，36对交互）
  * HELOC（1万条信用申请，23个特征，253对）
  * UCI Credit Card（3万条持卡人记录，23个特征，253对）。

**📈 对比分析**

比较方法与性能：
- 与TreeSHAP、H^2、NID、Integrated Hessians、Archipelago、SHAP-IQ、DIAMOND、FAST、iLOCO等方法对比；
- 评估指标包括Recall@15、平均精度（AP）、误发现率（FDP）、检索率（Power）和对真实数据的预测损失；
- LENS在所有合成测试中实现了最高的Recall@15和AP，误发现率远低于对手；
- 在freMTPL2上LENS恢复了已被保险行业确认的交互，且在保费计价的GLM上实现了显著的损失改进；
- 在信用数据上，LENS得到了极短的交互列表，且无损失提升，证明了其在弱交互场景下的稳健性；
- 计算成本方面，LENS仅需两次模型拟合，预测查询量线性于交互数，显著低于大多数重拟合或留出方法。

**⚠️ 局限性**

局限性：
- 需要至少两套独立样本，样本量不足时复制检测失效；
- 仅针对二元交互，无法直接扩展到三元及更高阶交互；
- 交互表面基于网格平均，可能无法捕捉在子群中平均为零的高阶交互；
- 对高度相关特征的代理交互仍可能通过复制检验被错误保留；
- 参数η和κσ的设置需要领域经验，错误设定可能导致过多或过少交互被保留；
- 依赖同一学习器的两个拟合，若学习器本身存在系统性偏差，复制阈值无法完全消除。

---

## 407. Multi-Objective Scheduling of LLM-Based Multi-Agent Workflows in Cloud Computing

**arXiv ID:** 2611.13740 | [PDF](https://arxiv.org/pdf/2611.13740v1)

**作者:** Goshgar Can Ismayilov `[一作]` `[通讯]`, Goshgar Can Ismayilov

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文将基于大语言模型的多智能体工作流在云计算中的调度建模为多目标优化问题，并提出了 MO-SAIC 框架，包括四种调度器来求解。

**💡 创新点**

创新点在于首次将 LLM 工作流调度视为三目标（完成时间、成本、资源利用率）优化问题，并设计了专门的 MO-SAIC 框架与合成工作流生成器，提供可控实验平台。

**🔧 技术方法**

采用了 NSGA-II、加权和遗传算法 (W-GA)、多目标 HEFT (MO-HEFT) 与随机基线，以及基于云资源特性的仿真模型、统计显著性检验（Wilcoxon）等技术。

**📊 数据集**

使用自定义的参数化合成工作流生成器产生多种规模、深度、宽度的 DAG 以及 AWS EC2 CPU/GPU 实例的公开规格作为测试数据集。

**📈 对比分析**

通过与随机基线比较，NSGA-II 在完成时间、成本和利用率上均显著优于其他调度器（成本降低 94% 以上，完成时间提升 8% 以上），并在超维度前沿质量（非支配解数、超体积）方面表现最强；MO-HEFT 在利用率上突出，但成本高；W-GA 在成本/利用率平衡上占优。

**⚠️ 局限性**

主要局限包括：合成工作流与真实场景差距；仿真模型未考虑网络抖动、冷启动、硬件兼容性与内存限制；计算能力统一模型过于简化；调度仅为离线静态优化，缺乏动态容错与质量目标。

---

## 408. Retargetable Compilation for Tensor Kernels via a Unified Layout Abstraction

**arXiv ID:** 2611.13741 | [PDF](https://arxiv.org/pdf/2611.13741v1)

**作者:** Likai Ma `[一作]`, Yun Liang `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种跨后端可移植的编译器 Arknife，通过约束代数将指令可见布局视为一等语义，在编译阶段对所有指令的布局约束进行全局求解，并自动插入目标提供的布局修复，从而实现高性能的张量核生成。

**💡 创新点**

创新点：①将指令可见布局抽象为统一的约束代数，消除后端之间的布局描述差异；②通过 e‑graph 消除冗余的布局修复，减少不必要的指令；③在编译期间构造并求解全局整数规划，完成布局的全局一致性检查和最佳化；④提供仅依赖目标描述的接口，极大降低重新后端化成本。

**🔧 技术方法**

技术细节：约束代数 + 轴级映射 + 目标描述；e‑graph 归约 + 代价搜索；整数规划（PySCIPOpt/SCIP）求解全局布局；TensorIR‑style 指令图；硬件层次、内存空间、指令模板等目标元数据描述。

**📊 数据集**

使用多种工作负载组合：x86 CPU 的 MatMul、Bias、Add、ReLU 等；Qualcomm Hexagon NPU 的量化加法、乘法、转置、解码等；NVIDIA A100 GPU 的 FP16 MHA、TF32 GEMM；并在 Qwen3 推理实验中使用不同序列长度与头数进行评测；未使用公开数据集，而是按多维矩阵尺寸和模型配置构造测试。

**📈 对比分析**

评估方法：将 Arknife 与 oneDNN、Halide、Hydride、MISAAL、QNN、PyTorch、Triton、FlashAttention、TileLang 等现有库或编译器进行对比；通过多种形状/规模的矩阵运算、量化算子以及注意力/ GEMM 任务测量每秒调用次数或吞吐率；结果显示：x86 上对 oneDNN 提升 2.17×、对 MISAAL 提升 5.08×；Hexagon 上对 QNN 提升 3.78×；A100 上对未融合 PyTorch 提升 5.04×、对 cuBLAS 达到 97% ；同时布局处理指令数显著下降，布局相关开销缩减 17–85×。

**⚠️ 局限性**

限制与挑战：①需要为每个目标提供完整的指令模板和硬件描述，虽比传统方法轻量但仍需人工工作；②全局整数规划在图规模增大或布局约束复杂时求解时间可达数百毫秒，影响离线编译时长；③当前不处理动态形状、稀疏或非规则张量；④未自动完成权重量化预处理、Megakernel 融合等后端细节；⑤在极端异构或多层级 GPU/CPU 混合环境下，布局约束的表达和求解仍存在局限。

---

## 409. Deterministic, Reproducible Inference in Exact Integer Arithmetic, and Verifiable Distributed Inference as Its Consequence

**arXiv ID:** 2611.13730 | [PDF](https://arxiv.org/pdf/2611.13730v1)

**作者:** Sayed Amir Karim `[一作]` `[通讯]` (Pelloran Inc), Sayed Amir Karim (Pelloran Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

实现了一套全整数推理引擎，能够在 CPU、GPU 甚至分布式机器上对大型语言模型和视觉模型产生完全相同的位级输出。

**💡 创新点**

创新点在于将每个运算都拆成整数路径并配备浮点参考门，保证了可验证的确定性与跨平台可移植性，并实现了分布式推理的可核查性。

**🔧 技术方法**

使用平衡三进制权重量化、动态张量尺度、固定点矩阵乘、重量化、归一化、查表非线性、SIMD 与多线程（可结合）以及 SHA-256 哈希来确保重现性。

**📊 数据集**

实验数据涵盖 Qwen2.5‑0.5B、OLMoE‑1B‑7B、GLM‑4.5‑Air（106B 参数 MoE）以及 SAM 2.1，使用文本提示、token 采样与视觉输入进行评测。

**📈 对比分析**

通过与原生浮点推理在多种 CPU（ARM、x86‑64）和 GPU（Apple M3 Metal）上对比，整数路径相较于标量基线提升 4–12×，在 Qwen2.5 上 SIMD+多线程加速 11×，且所有位级输出均与浮点参考保持一致；分布式跑 6 台机器得到与单机哈希完全相同的结果。

**⚠️ 局限性**

局限性包括：相比优化的浮点推理无速度优势；输出为近似浮点而非完全相同；量化依赖批次/调度；不提供隐私保障；整数代码尺寸大，部分模型需分布式或分片存储。

---

## 410. Do They Do as They Are Told? Investigating Coding Agents' Adherence to Explicit Instructions in Context Files

**arXiv ID:** 2611.13709 | [PDF](https://arxiv.org/pdf/2611.13709v1)

**作者:** Gianmario Voria `[一作]` (University of Salerno), Fabio Palomba `[通讯]` (University of Salerno)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建三阶段实验管道，先从实际的代理上下文文件（ACF）中提取并细化可观测、可执行的指令，再通过LLM生成并验证测量脚本，最后评估这些指令在真实代理生成提交中的遵从情况。

**💡 创新点**

创新点在于：①将ACF指令细化为可观测且可操作的细粒度指令；②利用LLM自动生成并通过双重测试验证测量脚本，实现大规模可操作化；③揭示指令遵从呈现明显双峰分布，说明有些指令始终被遵循，而有些则几乎从不被遵循。

**🔧 技术方法**

核心技术包括：LLM辅助指令提取（Qwen 3.5 9B），LLM驱动的测量脚本生成与测试驱动验证（CodeT式双重一致性判定），以及统计检验（Kruskal-Wallis、Mann-Whitney U）来比较指令类别和代理模型的遵从率。

**📊 数据集**

使用的数据集包含621个独立的ACF快照与6,541个关联的代理生成提交，覆盖268个GitHub仓库，涵盖26种编程语言，构成约106,923条指令-提交映射。

**📈 对比分析**

实验结果显示，约62.9%的可观测指令能够被成功脚本化，最终在49.8%的可操作化指令中有至少一次可评估的提交；总体平均遵从率为59.8%，其中Copilot的遵从率显著高于Claude Code和OpenAI Codex；测试与开发流程类指令的遵从率最低。

**⚠️ 局限性**

主要限制包括：仅有13.8%的指令最终可被评估，导致样本覆盖率有限；LLM对可观测性判定易过度包含，可能导致合规率偏高；脚本默认宽松导致的可评估误判；以及对不同语言、仓库和代理模型的外部效度尚未完全验证。

---

## 411. On Private Knowledge Retrieval and Private Inference for secure multi-agent systems

**arXiv ID:** 2611.13696 | [PDF](https://arxiv.org/pdf/2611.13696v1)

**作者:** Vicenç Torra `[一作]` (Umeå University), Maria Bras-Amorós `[通讯]` (Universitat Politècnica de Catalunya)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对多智能体系统中的语义记忆中毒攻击，提出安全内存、更新与通信机制，并实现基于私有信息检索（PIR）与数字签名的私有知识检索与推理，验证其在Python实验中的有效性。

**💡 创新点**

创新点在于将PIR、k-匿名、桶式以及双非协作服务器与数字签名结合，构建安全外部知识库访问架构，并首次将Prolog式推理与PIR集成以抵御记忆中毒。

**🔧 技术方法**

使用了Prolog-like 推理引擎、PIR（k-匿名、桶式、双服务器非协作）、数字签名、哈希与加密技术。

**📊 数据集**

使用自建知识库（99条事实与规则、46条谓词-arity条目）进行实验。

**📈 对比分析**

通过基准查询与不同隐私机制对比，实验表明双服务器PIR在未缓存时耗时约0.22s，其他方法相近，显示可接受的性能。

**⚠️ 局限性**

局限包括未处理服务器协作攻击、缺乏通信日志与加密、推理引擎未实现JITI或记忆化优化。

---

## 412. FedVP: Federated Learning across Heterogeneous Models by Pooling Locally Trained Verifiers

**arXiv ID:** 2611.13708 | [PDF](https://arxiv.org/pdf/2611.13708v1)

**作者:** Yifan Li `[一作]` (University of Bath), Xi Chen `[通讯]` (University of Bath)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Federated Verifier Pool（FedVP）方案，让不同家族、不同规模的 LLM 通过共享验证器而非模型参数实现联邦学习，保持各自的生成器冻结。

**💡 创新点**

创新点在于：①把协作焦点从参数聚合转移到验证决策共享；②支持生成器和验证器异构、规模差异大的模型；③通过本地训练的验证器组成池，无需统一模型结构或参数；④验证器池可以由中心服务器维护，也可直接在客户端之间交换。

**🔧 技术方法**

技术细节包括：在每个客户端用 LoRA 对验证器进行二分类微调；生成器固定产生 N 个候选答案；验证器输出 logit，聚合方式采用平均 sigmoid；在推理时池中所有验证器评估候选答案，选择得分最高的答案；还提供了蒸馏方案将池知识压缩为单一验证器。

**📊 数据集**

使用了五个公开医学 QA 数据集：BioASQ yes/no、MedMCQA、MedQA、medical MMLU、PubMedQA；在每个中心分别使用不同的 Qwen/LLaMA 3B–14B 背景模型并训练对应的验证器。

**📈 对比分析**

与八种传统联邦基线（FedAvg、FedProx、FedAdam、FedAdagrad、FedIT、FFA-LoRA、FLoRA、FedPrompt）以及模型异构联邦方法 FedMD、FedMKT 进行对比，评估指标为准确率；FedVP 在所有五个中心的平均准确率上比最强基线提升 1.46–5.79 个百分点，并且在跨模型（异构）实验中同样表现优异；蒸馏后单一验证器的性能与池相近。

**⚠️ 局限性**

局限性包括：仅在五个医学 QA 数据集上验证；仅涉及 LLaMA 与 Qwen 两个模型族，规模 3B–14B；对其他任务、模型族及更大/更小模型的适用性尚未测试；验证器池在推理时需要多次评估，推理成本相对较高（可通过蒸馏缓解）。

---

## 413. Exploiting What NVIDIA Profilers Hide: Fast and Accurate Analytical GPU Performance Modeling

**arXiv ID:** 2611.13762 | [PDF](https://arxiv.org/pdf/2611.13762v1)

**作者:** Ziwen Wang `[一作]` (Hong Kong University of Science and Technology), Yun Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了名为UnitSight的队列分析模型和计数器恢复器，用以预测不同GPU硬件配置下的执行时间。

**💡 创新点**

创新点在于：①将固定延迟停顿直接重用，②通过每个硬件单元的压力重新计算排队停顿；③揭露NVIDIA官方分析器隐藏的每单元计数器，实现更精细的性能建模。

**🔧 技术方法**

使用的技术包括：队列分析（MVA）、固定点迭代、内核级计数器读取、微基准测量、重播调度和内存延迟模型。

**📊 数据集**

使用的数据集包括：9款NVIDIA GPU的真实硬件配置；50个LLM推理与微基准工作负载（vLLM、SiLU、RoPE等）。

**📈 对比分析**

与Accel‑Sim、GCoM、AMALI、Habitat等基线模型对比，整体平均MAPE为8.3%，比最佳基线低约30%；预测速度比Accels‑Sim快49倍，且比区间模型快1.6–4.2倍。

**⚠️ 局限性**

局限性包括：需在支持Turing及以上架构的GPU上收集计数器；对极端分支/异步工作负载的准确性有限；依赖单个基准GPU的计数器，迁移到全新架构时仍需一定调整。

---

## 414. Marrying Capacity and Precision for Efficient 2D Gaussian Image Representation

**arXiv ID:** 2611.13784 | [PDF](https://arxiv.org/pdf/2611.13784v1)

**作者:** Lin Dong `[一作]`, Lei Shi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种容量-精度耦合的二维高斯点集表示框架 CP-GS，用于在给定编码预算下实现高效图像表示。

**💡 创新点**

①内容自适应容量重分配：将多余的高斯迁移到需求高的区域；②分层畸变感知量化：对位置、几何、颜色属性及颜色内部细分进行混合精度分配；③在同一预算下联合优化容量与精度，构造共享候选空间并基于真实序列化尺寸与解码失真进行选择。

**🔧 技术方法**

使用二维高斯点集表示与可微分渲染；基于残差需求的非极大值抑制与冗余高斯合并；渲染域畸变校准与结构化19点量化配置；层次化颜色分组与量化；联合预算约束下的候选空间搜索（硬预算与 Lagrangian 形式）。

**📊 数据集**

DIV2K 与 Kodak 数据集。

**📈 对比分析**

与多种现有 INR 与 2D 高斯表示方法（如 SIREN、EVOS、GaussianImage、Image-GS 等）在相同高斯数量下对比，采用 PSNR、MS-SSIM、LPIPS 等指标。CP-GS 在保持相同或更低参数量与优化时间的前提下，PSNR 提升 0.86‑2.88 dB，BD‑rate 减少 45‑53%，整体表现优于对比方法。

**⚠️ 局限性**

依赖图像特定的校准与预定义候选空间；仅评估静态二维图像；搜索过程仍需手工设定比例与精度集合，未引入学习式自适应分配策略。

---

## 415. Integrating Action Prefix Prediction and Alignment Across Modalities for Multimodal Reinforcement Learning

**arXiv ID:** 2611.13782 | [PDF](https://arxiv.org/pdf/2611.13782v1)

**作者:** Sijie Xu `[一作]` (UNSW Sydney), Francisco Cruz `[通讯]` (Universidad Central de Chile)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对在线多模态强化学习，提出 Improved Predictive Representation (IPR) 模型，通过改进预测方式和跨模态对齐来提升控制策略性能。

**💡 创新点**

创新点在于：①将 MJPR 的递归隐层滚动改为基于动作前缀的并行预测；②将配对对齐改为锚点级别的 CoMM 对齐，以更好地利用多模态信息；③在训练时同时使用两种对齐和预测损失。

**🔧 技术方法**

主要技术包括 Soft Actor–Critic (SAC) 基础网络、模态特定编码器、动作前缀 Transformer、动量目标网络、InfoNCE 对齐损失以及数据增强（随机裁剪、掩码）。

**📊 数据集**

使用的数据集有：DeepMind Control Suite（RGB-Depth 与 RGB-Position 两种多模态观测）以及 Webots 模拟导航环境（相机、LiDAR 与位置向量三模态，或相机与位置两模态）。

**📈 对比分析**

与 MJPR 基线对比时，IPR-H 在 DMC 上平均提升约 10%（最高 725.9 vs 659.9），IPR-S 也有显著提升；在 Webots 中，缺少 LiDAR 时 IPR 超越基线，而拥有 LiDAR 时基线仍优。总体而言，IPR 在多模态输入缺失关键信息时效果更佳。

**⚠️ 局限性**

局限性包括：实验缺乏统计显著性检验；不同裁剪强度对性能影响不一致；跨模态对齐与预测方式的交互效果尚未系统分析；结果受数据集与任务选择的影响，缺乏更广泛的通用性验证。

---

## 416. Scaling Shapley Energy Attribution to Large LLM Batches

**arXiv ID:** 2611.13802 | [PDF](https://arxiv.org/pdf/2611.13802v1)

**作者:** Qi Luo `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Yun Chen `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究LLM服务的能耗归因，提出一种基于Shapley值闭式公式的能耗分摊方法，可在一次GPU运行和CPU调度器重放中获得每个请求的精确能耗份额。

**💡 创新点**

创新点在于发现每个前向步的能耗可拆分为共享能耗（主要为权重量读）和请求自有能耗，从而推导出Shapley值的闭式解；通过一次微基准校准系数、在线账本实时计费以及在连续批处理和分布式部署下利用CPU重放校正步骤，显著降低了原本指数级的GPU运算开销。

**🔧 技术方法**

使用了Shapley值理论、能耗分解模型、微基准校准、vLLM调度器的CPU重放、控制变量抽样修正以及在线账本实现技术。

**📊 数据集**

实验数据来源于Azure LLM推理追踪（conversation、code、BurstGPT、Mooncake）、ShareGPT剪裁、七种模型（Qwen2.5‑7B/14B/72B、Gemma‑4‑12B、gpt‑oss‑20b、Qwen3‑30B‑A3B、Qwen3.8‑27B）以及三代GPU（Hopper H100、Ampere A800、Blackwell RTX PRO 6000）和分布式部署场景。

**📈 对比分析**

与GPU测得的真实Shapley值进行误差对比，在静态批处理（8–256请求）和连续批处理（8–128请求）以及分布式、MoE、前缀缓存等多种场景下，误差均低于0.4–6.4%，远优于按token比例（13.4–71.3%）或回归模型；在线账本耗时47µs/步，对吞吐影响<0.3%；CPU重放在几秒至数分钟内完成，误差可控制在0.01以内。

**⚠️ 局限性**

局限性包括：需要针对每个模型和GPU一次微基准校准；连续批处理和分布式部署需额外CPU重放，增加计算开销；MoE需获取路由日志；对极长输入或高并发场景下误差可能增大；受测量噪声和共享能耗不随请求数变化的假设限制。

---

## 417. TRUER: Test-Time Reward Correction Using an Error-Aware Rectifier for Reinforcement Learning

**arXiv ID:** 2611.13714 | [PDF](https://arxiv.org/pdf/2611.13714v1)

**作者:** Mingxiang Shen `[一作]` (Shanghai Jiao Tong University), Ningyi Xu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种利用有限源监督训练奖励修正器的框架 TRUER，用以纠正测试时强化学习中基于多数投票的奖励误差，从而提升语言模型的推理性能。

**💡 创新点**

创新点在于聚焦错误多数与正确少数的样本，训练错误意识的奖励修正器并冻结使用，以在缺乏目标真实奖励的情况下提高奖励准确率并改进学习信号。

**🔧 技术方法**

采用 GRPO 优化策略，利用源数据训练判别器生成分数并纠正奖励；结合采样、聚类、阈值判断等技术实现测试时奖励纠正。

**📊 数据集**

在数学推理任务上评估，使用 MATH500、MATH‑L5、AMC23 与 AIME24 四个数据集；源监督来自与目标任务同类的问题集合。

**📈 对比分析**

通过与无纠正的 TTRL 以及使用真实奖励的 Oracle GRPO 对比，TRUER 在奖励准确率和 Pass@1 上均显著优于 TTRL，并将差距缩小到 Oracle 的约 60‑90% 之间。

**⚠️ 局限性**

局限性包括训练时额外的计算开销、奖励纠正主要针对结构化答案且在开放式任务上的适用性尚未验证。

---

## 418. AutoManifold: Agentic Design of Data Visualisation Algorithms via Manifold Embedding

**arXiv ID:** 2611.13710 | [PDF](https://arxiv.org/pdf/2611.13710v1)

**作者:** Burak Susam `[一作]` (University of Manchester), Tingting Mu `[通讯]` (University of Manchester)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了 AutoManifold——一个基于大型语言模型的多代理系统，自动设计针对用户指定结构优先级的流形嵌入算法，并生成可执行代码。

**💡 创新点**

创新点：①将 LLM 与多代理工作流结合，形成“agent‑propose, engine‑measure”模式；②提出统一嵌入框架和原语词汇，实现可解释、可控的算法合成；③引入优先级约束，让合成算法针对单一或多目标指标优化，生成具备组合新颖性的嵌入方法。

**🔧 技术方法**

技术：Gemini‑3.1‑Pro LLM、结构化多代理系统、原语词汇（affinity、objective、solver）、嵌入难度分析、邻域分析、权重自适应、迭代超参更新、预定义评估引擎。

**📊 数据集**

数据集：iris、aging、kidney、pbmc3k、fashion‑mnist、cortex（覆盖易、中、难难度）

**📈 对比分析**

比较方法：与 t‑SNE、UMAP、PaCMAP（默认与随机搜索调参版）在 6 项可靠性得分（T_area、C_area、Spearman ρ、Stress、Silhouette、LOF）以及 4 项保留指标（Shepard、NMI、RNX、kNN）对比；AutoManifold 在目标指标上往往优于基线，且在非目标指标保持可接受范围。

**⚠️ 局限性**

limitation：①受原语词汇限制，搜索空间受约束；②对局部指标改进有限，主要在全局属性上优势显著；③生成算法一次性完成，迭代阶段仅调参，缺乏结构演化；④LLM 生成过程存在不确定性，需要人工验证；⑤评估仅覆盖六个数据集，未验证更大规模或不同领域的泛化。

---

## 419. Learning from First Discrepancies with Joint Reset and Pilot Guarantees

**arXiv ID:** 2611.13913 | [PDF](https://arxiv.org/pdf/2611.13913v1)

**作者:** Michal Čertík `[一作]` `[通讯]` (World Bank Group), Michal Čertík (World Bank Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在受限的停止式反馈下设计了一种学习算法，能以接近最优的重置次数和使用的不同假设数对 VC 维度的分类器进行学习，并给出了相应的下界

**💡 创新点**

首次给出在仅能观测错误前缀的“停止反馈”模型下，既有重置次数又有已使用假设数的最优两阶量级结果；同时提出了新颖的截断可见度、深度鲁棒图和和谐恢复技术

**🔧 技术方法**

使用指数权重更新、稀疏采样、潜在势函数、可见度阈值、深度鲁棒图、和谐平均恢复以及逆概率估计等经典工具的组合

**📊 数据集**

该工作为理论研究，未使用具体数据集，所有结果均为分析证明而非实验验证

**📈 对比分析**

通过理论下界证明所给上界是坐标wise‑minimax 最优；实验对比未给出，但相对于相关工作可证明在该反馈模型下实现了最佳资源复杂度

**⚠️ 局限性**

主要限制包括：算法需调用指数时间的模式或一致性 oracle；仅适用于共同记录分布、精确标签、以及固定的停止规则；对真实系统的鲁棒性与计算效率未做评估

---

## 420. Are Causal Neuron Circuits Unique and Transferable?

**arXiv ID:** 2611.13885 | [PDF](https://arxiv.org/pdf/2611.13885v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 421. Mixed Signals: Weight-Space Programming for EEG and ECG

**arXiv ID:** 2611.13758 | [PDF](https://arxiv.org/pdf/2611.13758v1)

**作者:** Dara Varam `[一作]` (Ecole Polytechnique Federale de Lausanne), David Atienza `[通讯]` (Ecole Polytechnique Federale de Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在EEG和ECG小型网络上，分别做了读取网络、提升网络、修复已知缺陷、调试隐藏缺陷以及从零写出完整网络的五个实验。

**💡 创新点**

首次将权重空间编程（Weight‑Space Programming）应用于生物信号领域，展示了语言模型在极少训练样本下能够从先验知识写出完整网络、精确定位并修复隐藏缺陷，以及发现未被告知的录制短路。

**🔧 技术方法**

使用大型语言模型（Claude Opus 5.5、GLM‑5.3、DeepSeek‑V4.1‑Flash）通过推理读取、编辑或生成网络权重；配合Python工具会话实现网络评估；与传统梯度下降和手工特征工程进行对比。

**📊 数据集**

使用 PhysioNet EEG Motor Movement/Imagery 数据集（109 受试者）和 PTB‑XL ECG 数据集（21,799 12‑lead ECG）进行实验。

**📈 对比分析**

评估指标：EEG 使用平衡准确率，ECG 使用宏平均 AUC。与 1–30 受试者梯度下降训练、手工特征（alpha‑band、CSP+LDA）以及基准修复方法比较。结果显示：在仅用 5 个受试者的情况下，LLM 写出的网络可达到与 30 受试者梯度下降相当或更高的性能；能够完整修复可识别缺陷；在调试隐藏短路时提升 MI‑AUC 至 0.83–0.86，但仍略低于已知原因下的最佳梯度修复。

**⚠️ 局限性**

局限包括：实验 3–5 仅单次跑；评估文本答案仅由单一 gpt‑oss‑120b 模型打分；所用数据集可能已包含在 LLM 训练语料中，影响先验知识；手工基线仅在实验 5 后加入；LLM 不能提升已良好训练的网络，也无法仅凭权重推断网络任务。

---

## 422. N -phase flow simulations using X-Mesh and an explicit marker interface representation

**arXiv ID:** 2611.13921 | [PDF](https://arxiv.org/pdf/2611.13921v1)

**作者:** Antoine Quiriny `[一作]` (UCLouvain), Jean-François Remacle `[通讯]` (University Grenoble Alpes)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于Lagrangian前沿追踪的X‑Mesh框架，用于在保持网格连通性不变的情况下，通过将网格边与显式标记链对齐，实现在不可压N相流中对尖锐界面进行高精度、无网格重建的模拟。

**💡 创新点**

创新点包括：①将显式标记表示与X‑Mesh网格变形相结合，允许在网格中出现高度扭曲或近乎退化的单元来精确捕捉细小的薄膜/丝状结构；②在破碎压力空间中采用分裂连续压力、变分膜模型，直接从耦合方程得到压力跃迁而不需显式曲率修正；③利用特征尺寸控制（δ）在显式拓扑变更中调节薄膜/丝状结构的保留与切除；④对三相及固壁接触点采用壁能量项，实现Young角的自然满足。

**🔧 技术方法**

核心技术包括Lagrangian前沿追踪（RK2积分）、X‑Mesh网格变形算法、Winslow能量最小化的粗细网格优化、TFEM对退化单元的锁定抑制、等阶压力-速度有限元配对（PSPG+SUPG）以及变分膜与壁能量的能量力学实现。

**📊 数据集**

实验数据全部来自标准数值基准：旋转箱（vortex‑in‑a‑box）预应变测试；Rayleigh‑Taylor不稳定性（两相与三相）；Hysing两泡上升基准；静态接触滴和毛细上升；以及三相液镜实验，均为公开参考数据集。

**📈 对比分析**

通过与He、Garoosi、Hysing等参考实验结果对比，本文在形态捕捉、薄膜保持与拓扑控制方面表现良好；在泡沫上升、三相RT等典型问题中，圆度、质量中心高度与上升速度等诊断指标与文献数据相符；然而在高层次卷曲或极细丝状结构出现时，结果对特征尺寸阈值高度敏感，说明数值扩散与拓扑控制参数对后期演化具有显著影响。

**⚠️ 局限性**

局限性包括：①需要手动调节标记间距、网格尺寸与特征尺寸阈值以避免尺度失配导致的数值振荡；②在极细结构下流场可能被低分辨率网格低精度捕捉，导致压力或速度振荡；③三维扩展需保证变形后网格不产生倒置单元；④特征尺寸控制本质上是一个建模参数，过度切除可能失去物理意义；⑤算法实现相对复杂，尤其是拓扑重建与变形迭代。

---

## 423. DroneSegEdit: Adapting Pretrained Generative Editors for UAV Segmentation with Language-Grounded Verification

**arXiv ID:** 2611.13890 | [PDF](https://arxiv.org/pdf/2611.13890v1)

**作者:** Dehua Chen `[一作]` (Fudan University), Jiangtao Wei `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用预训练的图像生成模型（Qwen2.5‑VL/DiT）结合语言‑定位验证器（LGSV）实现无人机图像的指代分割与全景解析。

**💡 创新点**

提出将RGB渲染作为统一输出接口，并通过语言‑定位验证器在保持生成边界的前提下完成实例选择，解决“Grounding‑Rendering Gap”问题。

**🔧 技术方法**

使用LoRA对Diffusion Transformer进行轻量级微调，读取冻结的Qwen2.5‑VL多模态隐藏状态生成空间‑语义证据，并采用固定颜色‑标签解码。

**📊 数据集**

在RIS‑LAD、UAVid‑RIS、VDD‑RIS三大无人机指代分割基准上评测，并在VDD‑dense任务上完成单指令全景解析。

**📈 对比分析**

与传统RIS方法、MLLM‑基线和其他生成式RGB模型对比，RIS‑LAD上mIoU达52.78（比前沿提高7.16），UAVid‑RIS和VDD‑RIS的compact目标精度显著提升，单指令VDD解析mIoU 80.96，位列第二。

**⚠️ 局限性**

存在颜色解码冲突、证据分辨率低、无法恢复被忽略的前景、以及高计算开销等限制，需进一步改进颜色编码、提升证据分辨率及加速推理。

---

## 424. Transparency Disclosures for Chain-of-Thought Monitorability

**arXiv ID:** 2611.13873 | [PDF](https://arxiv.org/pdf/2611.13873v1)

**作者:** Jacob Davies `[一作]`, Theo Bearman `[通讯]` (Institute for AI Policy and Strategy)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了五个可公开披露的链式思维监控指标，并对顶尖前沿AI开发者的公开披露进行了审计

**💡 创新点**

首次明确将链式思维可监控性拆分为三大维度并确定五个最具可行性和信息量的指标，形成了可操作的披露框架

**🔧 技术方法**

结合文献综述、指标打分方法和跨公司审计的方式构建评估体系

**📊 数据集**

使用OpenAI、Anthropic、Google DeepMind的系统卡与安全报告作为数据源

**📈 对比分析**

通过比较三家公司披露的指标覆盖率，发现每个指标至少有一家披露，但没有公司覆盖全部指标，凸显披露差距

**⚠️ 局限性**

指标评分主观性、缺乏统一标准化测评、仅基于公开信息、易受游戏化影响

---

## 425. FastWild-GS: Minutes-Scale In-the-Wild Gaussian Splatting via Compact Per-Image Appearance Adaptation

**arXiv ID:** 2611.13808 | [PDF](https://arxiv.org/pdf/2611.13808v1)

**作者:** Donghun Ryu `[一作]` (Chung-Ang University), Minhyeok Lee `[通讯]` (Chung-Ang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种轻量化的图像外观建模模块，使3D高斯散点渲染能够在分钟级训练时间内适应互联网照片集合的曝光、照明和瞬时物体差异。

**💡 创新点**

创新点在于引入每张照片的128维 Photo Appearance Token 与一个通道级缩放偏移变换，并通过共享的 PAT Decoder MLP 将 Token 转化为高斯颜色，从而在保持全局外观差异的同时避免传统的每张图像独立颜色参数。

**🔧 技术方法**

技术上使用 3D Gaussian Splatting 基础架构、FastGS 的多视角稠密化与修剪、Splat Descriptor 与 PAT Decoder 的小型 MLP、以及深度先验正则化，并在测试时采用半图像适配与径向增益。

**📊 数据集**

实验数据集包括标准 PhotoTourism 三个场景（布兰登堡门、圣心教堂、特雷维喷泉）以及 NeRF-OSR 四个户外场景（europa、lwp、st、stjohann）。

**📈 对比分析**

与现有的 In-the-wild Gaussian Splatting 方法比较，本文在 PhotoTourism 上以约 3 分钟/场景训练时间实现最低 LPIPS，并在 NeRF-OSR 上以 9.7 分钟/场景达到最佳 PSNR/SSIM/LPIPS，显著优于几小时级的竞争方法。

**⚠️ 局限性**

局限性在于仅提供图像全局的外观控制，无法明确分离瞬时遮挡物或局部光照变化，且在更大规模的照片集合中可能需要更严格的高斯计数控制。

---

## 426. Adapting a Vision Foundation Model for Fracture Characterization in Acoustic Image Logs

**arXiv ID:** 2611.13911 | [PDF](https://arxiv.org/pdf/2611.13911v1)

**作者:** Guangyu Wang `[一作]` (University of Science and Technology of China), Xinming Wu `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过微调预训练的DINOv3视觉基础模型，实现声学图像日志的裂缝分割及参数提取；

**💡 创新点**

在仅80张标注样本下证明VFM微调可超越从零开始的U-Net、YOLOv8、Swin Transformer，并系统评估解码器、微调策略、图像增强对性能的影响；

**🔧 技术方法**

使用DINOv3+PUP解码器+全微调、交叉熵+Dice+SCNP损失，配合正弦/直线拟合提取天然裂缝(NF)和钻孔诱发张力裂缝( DITF)；

**📊 数据集**

120张声学幅度图像，80张训练、20验证、20测试，来自4口井训练/验证、6口井测试，标注为NF、DITF；

**📈 对比分析**

采用IoU评估分割，FNR/FPR、参数误差与综合得分比较；测试集IoU 0.624（比U-Net 0.545高14.5%），NF检测FNR 0.15（比0.49低69%），DITF长度误差最低；但随着样本量减少，优势下降；

**⚠️ 局限性**

受图像噪声、井壁粗糙、地质纹理等影响，部分细碎裂缝仍难以识别；专家评审仍必要；图像增强并不总能提升性能；

---

## 427. EVAThink: Evidence Sufficiency-Guided Training for Adaptive Thinking in Video Understanding

**arXiv ID:** 2611.13907 | [PDF](https://arxiv.org/pdf/2611.13907v1)

**作者:** Changti Wu `[一作]` (East China Normal University), Yuxiang Zhang `[通讯]` (Huawei Technologies Co., Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

EVAThink通过证据充分性指导训练，实现视频理解的自适应推理。

**💡 创新点**

创新点在于构造证据对比对（Recoverable Evidence Gap Mining）并采用两阶段强化学习和首词优势正则来学习何时直接回答、何时推理。

**🔧 技术方法**

技术包括基于Qwen3‑VL‑8B的VLM、两阶段GRPO强化学习、证据差异化奖励设计以及首词优势正则化。

**📊 数据集**

使用了约118.2K条多任务视频问答样本，来源于8个公开数据集（Video‑MME、MVBench、LongVideoBench、MMVU、Video‑MMMU、VSI‑Bench等）。

**📈 对比分析**

在32/64帧固定视觉预算下与VideoAuto‑R1、AdaThinkV、Video‑FLAIR等基线对比，平均准确率提升至65.6%（64帧），同时推理占比下降至约34%，输出长度缩短约62%。

**⚠️ 局限性**

局限性包括对极端稀疏证据场景的鲁棒性不足、训练仍需要一定量（6.7K）样本、以及在长视频或更大帧预算下推理开销未完全消除。

---

## 428. Target Selection with a Disappearing-Stimulus Mixed-Reality P300 Interface: Feasibility and Object-Distinguishable Versus Uniform Cues

**arXiv ID:** 2611.13906 | [PDF](https://arxiv.org/pdf/2611.13906v1)

**作者:** Runqi Pan `[一作]` (Imperial College London), Kiyoshi Kotani `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了消失刺激混合现实P300接口的可行性，比较了统一球体与可辨识物体提示的差异。

**💡 创新点**

首次在MR环境中验证目标隐藏后仍可通过P300进行选择，并探讨物体身份提示对性能的影响。

**🔧 技术方法**

基于EEG的P300事件相关电位，使用收缩线性判别分析和嵌套交叉验证。

**📊 数据集**

六名健康受试者共48次实验，包含3通道EEG记录和5×5格子目标。

**📈 对比分析**

采用完整试验留出法和配对符号翻转检验，发现两种提示在完整目标准确率上无显著差异，平均约37.5%，信息速率约0.4bit/min。

**⚠️ 局限性**

样本量小、未随机化目标、未在线评估、仅三通道EEG，且受试者为BCI经验者。

---

## 429. The BRIDGE dataset: paired brightfield whole-slide images of prostate core-needle biopsies in paraffin, after deparaffinization and H&E staining

**arXiv ID:** 2611.13883 | [PDF](https://arxiv.org/pdf/2611.13883v1)

**作者:** Julia A. Mielcarz `[一作]` (Karolinska Institutet), Andrea Camilloni `[通讯]` (Karolinska Institutet)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

发布了BRIDGE数据集，包含169名患者的869块玻璃切片在原始、脱石蜡和H&E染色三个处理阶段的全切片图像，并提供相应的Gleason评分。

**💡 创新点**

首创公开提供同一组织切片在三种处理阶段的配对WSI，并附带病理等级标签，为无标签虚拟染色、虚拟脱蜡、图像配准与分割等研究提供丰富且多样化的数据基础。

**🔧 技术方法**

采用VENTANA DP 600全切片扫描仪（40×，0.25 µm/像素）获取图像，并通过人工视觉对图像质量和染料强度进行标签；无自定义代码，仅使用厂家扫描软件。

**📊 数据集**

使用BRIDGE数据集本身，包含2607张WSI，涵盖原始、脱石蜡和H&E染色三个阶段，及其对应的Gleason评分与ISUP分级。

**📈 对比分析**

该研究未进行模型训练或性能评估，而是将数据公开供外部研究者用于方法验证和性能比较；目前尚无相关性能指标。

**⚠️ 局限性**

局限性包括：数据来自单一机构和单一扫描仪，染料使用未标准化导致图像质量差异；缺少像素级标注；缺乏跨机构或跨扫描仪的泛化验证。

---

## 430. SFU-Sim: Bit-Exact Reverse Engineering of NVIDIA MUFU and Validation of PTX Paths

**arXiv ID:** 2611.13879 | [PDF](https://arxiv.org/pdf/2611.13879v1)

**作者:** Haozhi Xie `[一作]` (Sun Yat-sen University), Tao Su `[通讯]` (Sun Yat-sen University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对 NVIDIA GPU 的 MUFU（多功能单元）指令进行黑盒逆向工程，构建了比特精确的算术模型，并将其与 PTX 路径中的周边操作结合，实现对 GPU 特殊函数指令输出的完全复现。

**💡 创新点**

创新点在于：①将 PTX 路径拆解为单一 MUFU 与周边操作；②通过观察硬件输出恢复段结构、系数表、截断平方等有限宽度算术；③实现无查表、无 GPU 依赖的比特级别模拟；④对三代 GPU 进行全域验证，达成零误差。

**🔧 技术方法**

使用了 SASS 反汇编、黑盒逆向技术、整数区间约束求解、段索引与量化分析、截断平方实现、Python/C++ 算法实现以及自动化验证脚本。

**📊 数据集**

数据集：针对 A100（Ampere）、RTX 4060（Ada Lovelace）和 RTX PRO 5000 Blackwell 三个 GPU，枚举其指定输入域（FP32 2^32、FP16/ BF16 2^16、FP64 高字 2^32）进行完整覆盖，生成对应的硬件输出结果。

**📈 对比分析**

比较方法：对每个输入分别在 GPU 与 SFU‑Sim 计算得到的输出做比特级别逐位比较，统计误差数量。性能表现：在所有测试路径（12 条最小路径、7 条非 FP32 近似路径、2 条 FP16x2、8 条四舍五入模式的 RCP/SQRT）上均达到零误差，验证覆盖率 100%。

**⚠️ 局限性**

局限性：仅针对已选的 14 条 MUFU 指令（8 条 FP32、2 条 FP16、2 条 BF16、2 条 FP64 高字、7 条非 FP32 近似路径等）；未覆盖更复杂的指令组合或全局编译优化；对未来 GPU 架构的更改可能需要重新逆向；模型假设硬件实现稳定，若出现硬件变更需重构模型。

---

## 431. Feedback Vertex Set, Max Cut, and Edge Coloring on the Strong Product of a Graph and a Path

**arXiv ID:** 2611.13917 | [PDF](https://arxiv.org/pdf/2611.13917v1)

**作者:** Jan Diester `[一作]` (Karlsruhe Institute of Technology), Torsten Ueckerdt `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了强积路径图P⊠P及其子图、诱导子图和完整乘积G⊠Pℓ上的最大割、3边着色和回路反馈点集问题的计算复杂度；

**💡 创新点**

创新点在于证明这些问题在P⊠P的子图/诱导子图上为NP‑完全，揭示了奇偶路径长度对问题可解性的关键影响，并将3边着色问题与平面图3边着色等价，扩展了图乘积结构理论的算法适用范围；

**🔧 技术方法**

采用了图嵌入与细化技术，将任意子三度图的偶次细分嵌入P⊠P，构造交叉消除工具和加权反馈点集映射，利用树宽、克利克宽与图类封闭性等参数化技术；

**📊 数据集**

未使用外部数据集，而是通过构造与多项式时间可实现的图嵌入与细分来生成实验实例；

**📈 对比分析**

通过多场景归约与参数化分析与已知NP‑完全基准（如3边着色、最大割）比较，展示了在P⊠P子图上的NP‑完全性以及在某些图类（如树宽有限、二分图）上的多项式或线性时间解法；

**⚠️ 局限性**

局限性包括对非二分图奇数路径长度的可解性仍未完全确定，且对更广泛图类的参数化复杂度分析尚待深入；

---

## 432. Where to Approximate in Neurosymbolic Inference?

**arXiv ID:** 2611.13862 | [PDF](https://arxiv.org/pdf/2611.13862v1)

**作者:** Samy Badreddine `[一作]` (Sony AI), Antonio Vergari `[通讯]` (University of Edinburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fede83ac-7505-405f-ab37-e7284695c47f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一套统一的三阶段近似编译管线，用于在神经符号模型中高效计算加权模型计数（WMC）

**💡 创新点**

创新点在于将张量网络（tensor train）重构与重压缩方法嵌入管线，系统分析何处投放近似，并设计新型基于TT的乘法重压缩策略

**🔧 技术方法**

主要技术包括张量列阵（tensor train）分解、SVD与插值重压缩、结构可分解电路（如SDD）以及底层的离散权重计算

**📊 数据集**

实验使用的主要数据集包括Sudoku、MNISTAdd/MNISTMul、CLEVR-Diff/CLEVR-CountDiff，以及视觉Sudoku（ViSudo-Classification/Completion）

**📈 对比分析**

与传统完全编译（SDD）和基于独立性（fuzzy）近似对比，实验显示新管线在大规模Sudoku、加法/乘法和场景计数任务上WMC误差降低数个数量级，学习准确度显著提升

**⚠️ 局限性**

局限在于对稀疏布尔张量的TT压缩尚未优化，且对程序因子化（尤其辅助变量）对梯度影响的理解仍不充分

---

## 433. Satellite Pretraining Is Not One Thing: A Matched-Pair Study of Generic and Earth-Observation Foundation Models for Scene Classification

**arXiv ID:** 2611.13849 | [PDF](https://arxiv.org/pdf/2611.13849v1)

**作者:** Sandarbh Agrawal `[一作]` (SatSure Analytics India Pvt Ltd), Ashwathi Nambiar `[通讯]` (SatSure Analytics India Pvt Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对五对匹配的地理空间与通用预训练视觉模型进行冻结特征评估，比较其在五个 RGB 场景分类数据集上的表现。

**💡 创新点**

通过匹配相同目标与架构的模型对比，独立验证卫星预训练是否真正带来优势，并用可解释的分隔度指标预测性能。

**🔧 技术方法**

Vision Transformer 预训练模型（DINO、DINOv3、MAE、CLIP 等）、k‑NN、线性、注意力探测器和检索 mAP。

**📊 数据集**

PatternNet、RSICD、EuroSAT、UC Merced 与 RESISC45 这五个 RGB 场景分类基准。

**📈 对比分析**

采用六个随机 10%/90% 划分、配对显著性检验，结果显示卫星预训练在三对模型中提升性能、两对模型下降，差距随标签稀缺程度增大。

**⚠️ 局限性**

实验局限于 RGB 冻结特征、单一预训练目标、未控制数据集大小与内容差异、未考察多光谱/SAR 或微调效果。

---

## 434. Persistent Metric-Semantic Maritime Scene Memory from Open-Vocabulary 3D Scene Graphs

**arXiv ID:** 2611.13903 | [PDF](https://arxiv.org/pdf/2611.13903v1)

**作者:** Kim Alexander Christensen `[一作]` (Norwegian University of Science and Technology), Eleni Kelasidi `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在论文中，作者通过对开放词汇的户外三维场景图（OpenGraph）进行层级迁移与后处理，构建了面向海事环境的持久场景记忆系统，融合了几何-语义保守合并与多视角视觉-语言模型（VLM）推理，实现了物理实体与局部位置的聚合以及宏观海域区域的构造；

**💡 创新点**

创新点包括：①对户外3D场景图的分层迁移评估，揭示了仅有语义迁移但物理一致性与高层空间组织需改进的事实；②利用VLM在多视角几何约束下对碎片化的物理实体与局部位置进行重构；③结合地图拓扑与船上水面/结构观测，独立生成宏观海域区域并关联局部位置；

**🔧 技术方法**

所用技术主要包括：OpenGraph的RGB–LiDAR三维场景图构建框架、保守几何-语义合并算法（C050）、GPT-6 Astra VLM用于多视角物理实体与地方位置推理、基于OSM地图先验和船上水面/结构检测的宏观区域生成；

**📊 数据集**

使用的数据集为 Maritime Urban Tracking（MUT）数据集（含LiDAR、单目RGB、RTK GNSS位姿）和 Pohang Canal 数据集（包含多次巡航、LiDAR、RGB与OSM先验）；

**📈 对比分析**

与原始OpenGraph基线、仅合并（C050）以及合并+VLM后处理进行对比；物理实体碎片化从平均每艘船约2.3个节点降至1.3，碎片船只数从25/46降至11/47；宏观区域与参考IoU达0.81；地方位置跨巡航对应率为12/13；上下文检索F1从0.378提升至0.810；总体性能显著提升；

**⚠️ 局限性**

局限性包括：仅在两套数据集上验证，缺乏长期多周期评估；仅关注静态结构与固定实体，未处理动态对象追踪；对OSM先验的依赖可能限制未知海域应用；地方位置边界仍不够稳定。

---

## 435. A 10K drone image dataset with component-level annotations for rooftop photovoltaic inspection

**arXiv ID:** 2611.13892 | [PDF](https://arxiv.org/pdf/2611.13892v1)

**作者:** Penglei Zhang `[一作]` (Eastern Institute of Technology), Yuntian Chen `[通讯]` (Eastern Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了PVIS-10K数据集，包含10,065张无人机红绿蓝图像和63,632个光伏组件的三维注释，供屋顶光伏检验研究使用。

**💡 创新点**

创新点在于提供了组件级实例分割标注，结合多角度倾斜视角、遮挡、阴影等真实场景挑战，并发布了可与城市间跨站评估的分区。

**🔧 技术方法**

使用三维重建、投影回到二维、手工纠正、COCO格式导出，结合Mask R‑CNN、HTC、SOLOv2、Mask2Former四种监督模型和SAM3零样本模型进行基准测试。

**📊 数据集**

数据集来源于中国宁波、绍兴和深圳22个屋顶光伏站点，包含10,065幅高分辨率无人机图像。

**📈 对比分析**

与四个监督模型对比，HTC在Ningbo达72.72% Mask AP，Mask2Former在同城前景覆盖率最高；SAM3零样本模型AP仅为4–33%但前景覆盖率仍超过89%。

**⚠️ 局限性**

局限在于标注仅为投影形状，遮挡部分仍保留完整轮廓；跨站评估受城市差异影响，且零样本模型性能远低于监督模型。

---

## 436. Mechanistic Interpretability of Deep Neural Networks: Methods, Frameworks, and Perspectives

**arXiv ID:** 2611.13868 | [PDF](https://arxiv.org/pdf/2611.13868v1)

**作者:** Lorenzo Cirillo `[一作]`, Andrea Cavallaro `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了深度网络机制解释方法，提出将可视化、分解、因果干预和电路跟踪集成进解释框架的思路，并给出音频分类与图像重建的案例。

**💡 创新点**

创新点在于将多种机制解释技术归纳为四大类，并阐释如何在框架中层层协作，提出针对数据分布漂移与偏见等开放挑战。

**🔧 技术方法**

使用了表示可视化（激活图、logit lens）、表示分解（稀疏自编码器、任务向量）、因果干预（激活补丁、归因补丁）和电路跟踪（自动电路发现、边缘归因补丁）等技术。

**📊 数据集**

实验使用ESC‑50（音频谱图）、ImageNet‑1K（预训练）、CIFAR‑10（图像重建）等数据集。

**📈 对比分析**

在ESC‑50上实现88%准确率；在CIFAR‑10上自动电路发现得到PSNR 24、MSE 0.004；通过对比传统与机制解释框架，展示了对特定行为因果性验证的有效性。

**⚠️ 局限性**

局限包括高计算成本、对阈值和数据选择的敏感性、多义性导致的解释不清，以及在分布外场景下解释力下降。

---

## 437. Influence of the branching geometry on SPION propagation: coupling of experiment and simulation

**arXiv ID:** 2611.13876 | [PDF](https://arxiv.org/pdf/2611.13876v1)

**作者:** Daniel Fleischhauer `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Dietmar Drummer `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `51c0528b-f690-4182-ae60-bb5f046c276c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

在实验血管模型中对不同分支几何（SF=0.25, 0.56, 1）的单分支血管进行SPION传输实验，并通过Lagrangian粒子模拟（LPS）和Eulerian Volume of Fluid（VoFS）仿真与实验数据进行对比，研究分支几何、浓度层化和磁场调控对SPION在血管分支处的传播和分支选择的影响。

**💡 创新点**

首次系统性探讨分支几何不对称对SPION传输的作用，揭示在对称分支中浓度层化主导粒子分布、磁场可逆转被动分支偏好，并验证两种不同尺度模拟方法的适用范围，提出同时使用峰值和质量分数描述分支选择的需求。

**🔧 技术方法**

采用增材制造血管模型、光学实时传输测量、NdFeB永磁体磁场调控、Lagrangian粒子仿真、VoFS仿真以及一系列实验与仿真指标（峰值时间、FWHM、衰减率、分支质量分数）进行比较。

**📊 数据集**

实验数据为在流速2.5、5、10 cm/s下，三种分支几何中SPION光学信号的多次测量；仿真数据为对应几何网格、流速、SPION浓度（10%）及磁场分布（上分支150 mT、梯度46，低分支75 mT、梯度18）生成的粒子浓度场。

**📈 对比分析**

通过定量比较实验与仿真指标，发现LPS在非对称分支的峰值时间误差≈0.5–0.7，VoFS在对称分支能捕捉峰值早/宽度差异；磁场作用下两模型均可预测上分支质量分数升高和衰减率下降，表明磁场能显著改变化学信号。

**⚠️ 局限性**

实验中高浓度下光学信号超线性导致下分支信号低估，单粒子沉降被忽略；仿真未考虑真实微结构和瞬态混合，模型仅适用于低雷诺数静态血管，未验证在脉动流或更大尺度下的表现。

---

## 438. SAIL: Self-Adaptive Index Learning with Hybrid Models (Extended Version)

**arXiv ID:** 2611.13856 | [PDF](https://arxiv.org/pdf/2611.13856v1)

**作者:** Kasun Amarasinghe `[一作]` (University of Melbourne), Jianzhong Qi `[通讯]` (University of Melbourne)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种自适应混合学习索引（Sail），通过递归划分键集并为每个子集选择最合适的叶子或内部节点类型，以优化查询性能。

**💡 创新点**

创新点在于：① 采用多种候选节点类型（线性学习节点、SIMD搜索、二叉搜索、哈希搜索）并结合缓存友好的桶结构；② 通过基于硬件的成本模型快速估算不同节点类型的查询成本，从而在递归构建时做出节点选择；③ 将学习型节点作为内部节点，减少分区深度并提升点查询效率。

**🔧 技术方法**

主要技术包括：递归自顶向下划分、学习式线性预测模型、SIMD向量化搜索、哈希映射与线性探测、缓存行对齐的桶存储、基于整数比较/乘加/哈希/缓存命中等操作的成本模型。

**📊 数据集**

使用六个真实数据集（OSM、Genome、Facebook、Books、Uniform、Lognormal）以及两个合成数据集，全部包含约2亿条键值对进行实验。

**📈 对比分析**

与Chameleon、LIPP、SALI、ALEX、PGM、VEGA、CARMI、HIRE等基线相比，Sail 在点查询上平均提升约 2 倍，范围查询性能与 VEGA、PGM 相当，内存占用和构建时间与现有学习索引相近；在插入场景下，Sail 也保持了与 Chameleon 竞争甚至优于其的插入速度。

**⚠️ 局限性**

局限性包括：① 需要手动调节桶大小、节点容量和哈希叶阈值等参数；② 哈希叶子节点虽然加速点查询，却在范围查询上存在一定损失；③ 成本模型需针对不同硬件和工作负载进行校准，模型误差可能导致非最优节点选择；④ 对于持续动态变化的键分布，单次构建后的自适应性有限。

---

## 439. PolyRoute: A Twelve-Language and Code-Mixed Benchmark for Typed-Decision Routing, with Baselines and a Leaderboard

**arXiv ID:** 2611.13843 | [PDF](https://arxiv.org/pdf/2611.13843v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 440. Online Network Design with Maximum Delay and Stochastic Arrivals

**arXiv ID:** 2611.13865 | [PDF](https://arxiv.org/pdf/2611.13865v1)

**作者:** Tianhang Lu `[一作]` `[通讯]` (Ocean University of China), Tianhang Lu (Ocean University of China)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在线网络服务的批量处理，针对已知Poisson到达率的情形，提出了无状态服务的确定性调度算法，证明在最大等待延迟下可实现常数比值 e/(e‑1)，并给出了设施位置问题的确定性算法，取得比值 3。

**💡 创新点**

创新点在于：①将请求间隔的并集作为离线成本的下界，利用Poisson性质推导期望覆盖率；②基于已知到达率而非未来信息，构造无状态服务的最优常数比值；③将这一思想推广到树聚合（多根、对聚合）并给出统一的时间调度方案；④针对设施位置问题设计“观测间隔赋值”策略，使连接成本可分离，并通过打包分析证明三倍常数上界。

**🔧 技术方法**

主要技术：Poisson进程的无偏估计与覆盖概率、区间并集下界、周期性调度与立即服务的比较、期望率计算、可观察到达间隙的指数分布、贪心打包与Poisson插入定理、时间切片分析、离线下界与随机化不变性论证。

**📊 数据集**

未使用实际数据集；全部是理论分析与期望比值评估，基于假设的独立Poisson到达率。

**📈 对比分析**

与无信息离线最优（已知所有到达）进行长期期望比值比较；最大延迟模型下的比值 e/(e‑1) 已证明最优（随机化亦如此），设施位置问题则获得 3 的常数比值；对树聚合模型也给出相同的常数上界；性能评估主要通过理论上限与下界的匹配来证明。

**⚠️ 局限性**

局限性：①比值是长期期望比值，无法给出统一的有限时隧比值；②要求已知Poisson到达率且服务无状态（不适用于持续移动服务器或每次连接收费不同的场景）；③对于设施位置问题，初始期的赋值与完成成本需要额外的实例相关常数；④在极端实例下，算法在短期内的比值可能无界。

---

## 441. Beyond Passive Imitation: On-Policy Mid-Training Distillation for Language Models

**arXiv ID:** 2611.13855 | [PDF](https://arxiv.org/pdf/2611.13855v1)

**作者:** Ruize Zheng `[一作]` (Fudan University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在固定数学语料上进行持续预训练时不同基础模型的表现差异，并提出了一种名为 On-Policy Mid-Training Distillation (OPMD) 的方法，利用 anchor token 识别、学生策略 rollouts 与冻结教师的逆 KL 奖励来主动获取知识，解决传统 NTP 的负面效应。

**💡 创新点**

创新点在于将中期预训练转变为主动查询驱动的知识获取流程，通过 anchor token 的熵筛选触发学生策略 rollouts，并在学生生成的轨迹上使用密集的 Token‑level 逆 KL 蒸馏，既解决了训练‑生成不匹配，又降低了对语料质量的敏感性。

**🔧 技术方法**

采用了 anchor token identification (基于预测熵)、on‑policy rollouts、Token‑level reverse KL distillation、PPO 强化学习框架以及冻结教师模型的蒸馏技术。

**📊 数据集**

使用了 1B‑token 的 MegaMath‑Web‑Pro 子集进行中期预训练，并在 MATH‑500 与 Minerva 两个数学推理基准上评测。

**📈 对比分析**

通过与标准 NTP 进行对比实验，OPMD 在 Qwen3‑1.7B 上将 MATH‑500 的 pass@16 提升至 82.8%（+6.2pp）且 avg@16 提升至 30.5%，Minerva 的 pass@16 提升至 36.0%（+5.5pp），且不出现 NTP 导致的性能下降；不同规模教师（4B 与 14B）在相同 token exposure 下表现相近。

**⚠️ 局限性**

局限性包括仅在数学领域验证，anchor 选取策略对性能的影响仍需进一步探究；对更大规模模型或其他领域的泛化能力尚未测试。

---

## 442. Alania-2: Turkish text-to-speech with instant and fine-tuned voice cloning

**arXiv ID:** 2611.13845 | [PDF](https://arxiv.org/pdf/2611.13845v1)

**作者:** Sezgin Saygili `[一作]` (PatientDesk AI), Fikri San Koktas `[通讯]` (PatientDesk AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了Alania‑2，面向土耳其语的高质量TTS系统，包含文本规范化、语音克隆与低延迟推理引擎；

**💡 创新点**

创新点在于为中等资源语言设计的训练数据方案（90%合成语音），全流程的文本规范化规则、即时与LoRA方式的语音克隆，以及针对单GPU的推理引擎优化；

**🔧 技术方法**

采用VoxCPM2架构（语言模型+扩散解码器+VAE）+无监督引导、低秩LoRA适配、规则化文本预处理、Nano‑vLLM推理引擎（批量VAE解码、CUDA‑graph预填充等）；

**📊 数据集**

使用1,772h真实YODAS语音+16,423h合成语音（VoxCPM2生成）、FineWeb2、维基百科、1.44M句子域文本以及3,411h设计语音的合成数据集；

**📈 对比分析**

在Freya‑TR‑Eval 495句基准上对比14个系统，Alania‑2通过API取得1.44% WER（第七位），CER为0.28%；与4B模型相比仅差0.5个百分点，Antalia‑2 Mini 7.6M模型则达到1.05% WER；首次音频延迟约32 ms，吞吐率≈136 s/s；

**⚠️ 局限性**

局限包括：自然度评估使用基于英文的UTMOS22，误码率受识别器与文本规范化影响；模型继承VoxCPM2的声学特征与潜在伪音；训练细节与部分GPU时间未完整记录；未公开权重，缺乏用户听感评测。

---

## 443. MusicFlux: Flexible Conditioning for Text-Guided Music Generation and Editing

**arXiv ID:** 2611.13834 | [PDF](https://arxiv.org/pdf/2611.13834v1)

**作者:** Teysir Baoueb `[一作]`, Ga{ë}l Richard `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MusicFlux框架，集成文本引导的音乐生成、局部编辑（inpainting/outpainting）和全局编辑（SFX修复），并实现两种条件注入策略：序列拼接和CAR式条件重参数化。

**💡 创新点**

首次将CAR-Flow理念引入音频编辑，利用轻量级偏移网络在不扩大序列长度的前提下实现条件对齐；同时统一生成与编辑任务，减少专用架构需求。

**🔧 技术方法**

使用流匹配（flow matching）+ Diffusion Transformer（DiT）核心，结合FLAN‑T5文本编码、条件自适应归一化、CFG引导以及轻量级偏移网络实现条件重参数化。

**📊 数据集**

在JamendoMaxCaps（JMC）训练并评估，使用SonicMaster‑Test进行SFX修复测试，使用Song Describer Dataset（SDD）检验跨域生成与编辑性能。

**📈 对比分析**

与Stable Audio 3.0及SonicMaster基线对比，利用CAF‑Score、KAD、FLEUR*、KL、SSIM、PQ等指标。MusicFlux‑CAR在CAF‑Score上领先，MusicFlux‑Concat在KAD、FLEUR*与PQ上表现最佳；CAR版本在推理时RTF与动态内存下降50%+，显著提升效率。

**⚠️ 局限性**

对外域文本提示的泛化仍有限；CAR策略在局部编辑中相对拼接效果略弱；需要进一步探索更丰富、高效的条件注入方法以提升细粒度编辑质量。

---

## 444. Oscillatory Processing Units as a computing substrate for biomimetic intelligence

**arXiv ID:** 2611.13840 | [PDF](https://arxiv.org/pdf/2611.13840v1)

**作者:** Alexander A. Wolf `[一作]` `[通讯]` (Artificial Brains), Alexander A. Wolf (Artificial Brains)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

论文提出并实现了振荡处理单元（OPU）作为人工大脑的计算单元，通过实验测量源-接收交互所产生的振荡占用空间（OOS），并将这些可测交互用于构建逻辑运算、关联记忆、决策与物理机器人控制。

**💡 创新点**

创新点在于将神经振荡的相位与周期性释放机制转化为可度量的计算单元；通过构建“奥帕图（Atlas）”映射源-接收条件与OOS属性的关系，实现在不同网络和时序条件下的预判性合成；以及演示固定权重网络即可实现记忆回放与闭环行为。

**🔧 技术方法**

使用的技术包括：基于FPGA的合成神经元（spiking）模型；固定权重的OOPU网络；振荡相位调控和周期性释放的实验干预；多维度状态特征（参与度、熵、有效秩、CKA、NMI等）与统计验证（Monte Carlo、Bootstrap、Exact sign‑flip）；以及在虚拟和物理机器人平台上的闭环控制实现。

**📊 数据集**

实验数据集：人工生成的神经网络实例（11个独立网络），对多种节拍类（Gamma、Beta、Alpha、Theta、Delta）交互进行大规模参数扫；感知事件序列（18个对齐周期、25,000 步等）；在虚拟机器人（Webots）与真实 ANT 机器人上收集的感知/动作记录。

**📈 对比分析**

比较方法：通过OOS特征的聚类、CKA、NMI、有效秩等指标评估交互差异；使用Atlas进行“预判性合成”时将预测值与实际测得的 OOS 进行直接比较；在动作实验中将完整电路与“影子输出”对照进行碰撞率、AUC 等统计检验。性能表现：逻辑/算术操作的拟合 R² ≥ 0.90；闭环机器人在 23/24 运行中无碰撞；预判性合成在所有 11 个网络上均保持预设顺序一致。

**⚠️ 局限性**

局限性：仅使用固定权重网络，未评估学习与权重适配的潜力；部分功能（如 XOR）在特定输入条件下表现不佳；实验规模受限于 FPGA 资源（最多约 10,000 个神经元）；对不同任务与更复杂情境的泛化能力尚未验证；统计检验多依赖于网络对称与同一感知序列，可能受到共线性与序列偏差影响。

---

## 445. Swanada: Split-Encoder Capability-Routed Conformer CTC and Sub-30 MB On-Device Speech Recognition Across Twelve Scripts

**arXiv ID:** 2611.13814 | [PDF](https://arxiv.org/pdf/2611.13814v1)

**作者:** Hemanth HM `[一作]` `[通讯]`, Hemanth HM

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `fede83ac-7505-405f-ab37-e7284695c47f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了名为Swanada的浏览器/Node.js/Python端多语言流式语音识别系统，支持12种印度脚本（包括古典梵语、印度英语及十种印度语言），采用16 kHz Conformer‑CTC模型。

**💡 创新点**

创新点在于：①分离编码器与12个可热切换的本地脚本CTC头，显著压缩模型体积；②动态相对位置编码与点卷积替换，减少ONNX静态常量并提升WebAssembly SIMD兼容性；③零依赖protobuf尾部修补，< 2 ms即可在会话中注入共享特征；④两区Longest‑Common‑Prefix流式稳定化，保证0 %前缀漂移；⑤结合SVD预热、四项对齐蒸馏与后量化头校准，保持高精度。

**🔧 技术方法**

技术手段包括：Conformer‑CTC、INT8量化、ONNX+WebAssembly SIMD、动态相对位置合成、1×1卷积重写为矩阵乘、SVD warm‑start +四项蒸馏、后量化头校准、零依赖protobuf patch、两区LCP流式解码、语音增强（HPF、噪声跟踪、Wiener滤波）。

**📊 数据集**

使用数据集：480条留存的Studio 12语言评测集；1280条公开Hugging Face官方拆分（包含12个Indic、4个多方言、4个通用语、4个俚语和1个口语会议）；150条多方言/俚语/回声测试集；以及72条100 ms流式会话测试。

**📈 对比分析**

对比方法：与12个单语大规模Conformer（单体177 MiB）和未压缩的Studio模型比较。Swanada Studio保持相同的5.43 % held‑out CER，Open+Wild 8.87 % CER；Mobile版 19.23 % CER。RTF为0.0097×–0.0264×，TTFT 30.6 ms，中位数；模型大小压缩至Studio 11.6×、Mobile 67.8×。

**⚠️ 局限性**

局限性：对极端混响、多说话人会议的鲁棒性下降；词汇错误导致高WER，尤其在黏着语和梵语复合词中；缺乏自动语言识别与切换；iOS单线程限制导致性能下降；未整合语言模型，无法进一步降低WERS。

---

## 446. Camera equalisation of an 11.5x drop in environmental illuminance: four imitation-learning policies on an SO-101 arm

**arXiv ID:** 2611.13810 | [PDF](https://arxiv.org/pdf/2611.13810v1)

**作者:** Simon Cowling `[一作]` `[通讯]` (fysical.ai), Simon Cowling (fysical.ai)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在低成本 SO-101 机器人上，对比四种视觉驱动策略在正常光照与大幅光照降低（约 11.5 倍）条件下的 pick‑and‑place 成功率；

**💡 创新点**

指出光照强度（lux）与机器人实际收到的图像亮度差异显著，强调记录相机参数而非单纯 lux 以评估视觉策略鲁棒性；

**🔧 技术方法**

采用 ACT、SmolVLA、Diffusion Policy 与未命名策略的行为克隆模型，结合 OpenCV 捕获、图像亮度/对比度/锐度等统计；

**📊 数据集**

使用 50 条遥控演示数据训练模型，随后在相同 10 个放置位置进行 80 次实地 roll‑out；

**📈 对比分析**

通过 Fisher、McNemar、Mantel–Haenszel、GLM 等统计检验发现光照降低对四种策略成功率无显著影响，平均差异仅 -2.5%（95% CI -23~+19%），相互间无差异；

**⚠️ 局限性**

局限性包括样本量小（每组仅 10 次）、光照与日序/顺序混杂、未锁定曝光导致相机自适应、仅测试单一任务与单一硬件配置，且对照组与训练集对比不足。

---

## 447. EAGLE: Early Identification of Information Propagation Sources through Generalized Bayesian Inference in Large-scale Social Networks

**arXiv ID:** 2611.13919 | [PDF](https://arxiv.org/pdf/2611.13919v1)

**作者:** Tao Wen `[一作]` (University of Manchester), Kang Hao Cheong `[通讯]` (Nanyang Technological University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 EAGLE 框架，用稀疏观测者与广义贝叶斯推理结合，实现大规模社交网络中信息传播源的早期定位。

**💡 创新点**

创新点：①使用离散虫火优化算法对观测者进行全局最优部署，兼顾覆盖率与冗余；②引入信念函数代替传统概率，以显式表达候选源的不确定性；③对不同观测者组合赋予可信度权重，缓解噪声与信息不完整的影响；④整体方法无需已知传播模型或用户属性，兼容多种传播动态。

**🔧 技术方法**

技术：离散虫火优化 (Discrete Moth‑Flame Optimization)、广义贝叶斯推理 (Generalized Bayesian Inference)、Dempster–Shafer 证据理论、时间序列与路径关联的似然计算、观测者可信度计算与加权融合。

**📊 数据集**

数据集：10 个真实网络（dolphins、USAir97、email‑Eu、polblogs、facebook、ca‑GrQc、wiki‑Vote、lastfm‑asia、ca‑HepTh、ca‑HepPh）；两大规模 Twitter 实时传播路径（MoscowAthletics 与 NYClimateMarch）；以及基于四种传播模型（HSI、HDP、RWP、TDP）的合成网络。

**📈 对比分析**

与 SNF、OLS、GE、DISGE、ESL 等 SOTA 观测者基础源定位方法比较。EAGLE 在大部分网络、传播模型与观测比例下均取得最低错误距离和最高 F‑score；在 60k‑100k 规模的 Twitter 实际传播路径实验中，EAGLE 能以 20–120 个曝光观测者精准定位源，其他方法要么 F‑score 接近 0，要么候选源集超过千人。

**⚠️ 局限性**

局限：目前仅处理单源情况，未考虑多源传播；对观测者部署与数据获取存在一定前置成本；在极大网络或极低曝光率时仍面临样本稀缺导致的不确定性；对极端噪声或误报的鲁棒性尚待进一步验证。

---

## 448. MicroBT: Centroid-Guided Synergistic Learning for Multi-Modal Micro Brain Tumor Segmentation and Counting

**arXiv ID:** 2611.13912 | [PDF](https://arxiv.org/pdf/2611.13912v1)

**作者:** Zhenye Lou `[一作]` (University of Lincoln), Wenting Duan `[通讯]` (University of Lincoln)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了MicroBT框架，实现多模态MRI脑肿瘤微小病灶的联合分割与计数。

**💡 创新点**

创新点是利用尺度不变的病灶中心点作为共享锚点，实现分割与检测的协同学习，并引入体积校准损失和尺寸感知课程优化提升微小病灶的梯度贡献与训练稳定性。

**🔧 技术方法**

采用3D卷积编码器、中心点预测头、双分支解码器（分割与检测），结合自注意力、位置编码、VC‑Loss和SACO等技术。

**📊 数据集**

使用四个公开BraTS挑战数据集，包括脑转移、脑膜瘤、小儿和成人脑肿瘤，均含有3mm以下微小病灶。

**📈 对比分析**

与nnU-Net、TransUNet、Swin-UMamba等九种先进方法对比，在Dice、HD95、MAPE、AP_50等指标上均取得显著提升，尤其是微小病灶的Dice提升约2.5%，计数MAPE降低约26%。

**⚠️ 局限性**

局限在于对极大病灶的精度提升有限，且训练过程中需要两阶段课程学习与额外超参数，可能导致对资源和经验的依赖。

---

## 449. Domain-Invariant and Language-Model-Assisted Acoustic Side-Channel Attacks on Keyboards

**arXiv ID:** 2611.13922 | [PDF](https://arxiv.org/pdf/2611.13922v1)

**作者:** Bikrant Bikram Pratap Maurya `[一作]` `[通讯]` (Indraprastha Institute of Information Technology), Bikrant Bikram Pratap Maurya (Indraprastha Institute of Information Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套完整的声学侧信道攻击框架（DECKER），实现了在未知键盘、不同用户和环境下的跨键盘预测，并在 VoIP 通道上实现连续稀疏转导；同时构建了大规模、跨设备、多用户、多性别、噪声环境和非视线传播的 HEAR 数据集，用于真实场景下的评估。

**💡 创新点**

创新点主要包括：
- 通过键盘签名归一化（KSN）学习逆滤波器去除设备色彩；
- 采用梯度逆层（GRL）和跨键盘监督对比损失实现键盘不变性与键盘一致性；
- 设计声学风格随机化（ASR）在训练时模拟未见设备；
- 在 VoIP 上引入 Kolmogorov–Arnold 网络（KAN）自适应头以及事件触发连续时域状态空间模型（EvFlow‑CTSSM）以恢复压缩失真；
- 用大型语言模型（GPT‑2/FLAN‑T5）进行受限束搜索，纠正残余键级误差。

**🔧 技术方法**

使用的技术包括：ECAPA‑TDNN 声学编码器、梯度逆层、监督对比学习、Kolmogorov–Arnold 网络、事件触发连续时域状态空间模型、语音预训练模型（Wav2Vec2、WavLM、HuBERT、X‑Vectors、Whisper）、多模态融合策略、受限束搜索以及大语言模型。

**📊 数据集**

采用的主要数据集为 HEAR，包含 10,000+ 个按键事件，来自 53 名受试者、37 台笔记本键盘，采样于三种通道（外部麦克风、设备内麦克风、VoIP 流），并标注性别、手势、环境噪声等元数据；同时使用其 VoIP 子集进行压缩通道评估。

**📈 对比分析**

评估方法：在跨键盘、跨用户、跨环境留一分域的训练/测试拆分下，比较单模态、欧氏多模态融合以及 DECKER 的关键级准确率；在句子/密码级别使用受限束搜索。结果显示：单模态跨键盘 top‑1 几乎下降 40 %；多模态最佳仅 66 %；DECKER 在未见键盘上达 81 %（降低 18 % 误差）。VoIP 子集里，KAN 头将准确率提升约 10–15 %；EvFlow‑CTSSM 在连续稀疏转导上显著降低 CER。

**⚠️ 局限性**

局限性：
- 仍受键盘结构相似性导致的误差限制；
- 需要离线 GPU 计算，部署成本较高；
- 主要针对 QWERTY 键盘，未验证到触摸屏、密码锁等其它输入方式；
- 语言模型对文本结构的依赖可能导致对高熵密码的恢复不足；
- 防御措施（硬件调节、麦克风滤波等）尚未在实验中系统评估。

---

## 450. Beyond Fluent Responses: Finding User Dissatisfaction in Human-AI Interactions

**arXiv ID:** 2611.13894 | [PDF](https://arxiv.org/pdf/2611.13894v1)

**作者:** Yuka Saito `[一作]` (Tohoku University), Reina Akama `[通讯]` (Tohoku University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在ChatGPT对话中利用用户自报不满意标签，将用户发言视为判定不满意的输入，探究是否能仅凭用户发言及其前一次发言来检测用户的不满意情绪。

**💡 创新点**

创新点在于①将用户发言（尤其是前后发言的变化）作为直接输入，②结合异常检测方法捕捉用户行为的细微转变，从而提高不满意检测的准确性，且在英语数据中表现出比传统仅用系统响应更优的效果。

**🔧 技术方法**

技术包括：BERT（small、mini）和 GPT‑5.1 的二分类模型；异常检测算法包括 LOF、LUNAR、AE、VAE；嵌入模型使用 BERT‑base 和 OpenAI 的文本嵌入。

**📊 数据集**

使用的主要数据集是 ChatGPT Dissatisfaction Dataset（英文子集 824 句、147 句不满意），并扩展到波兰语、西班牙语和日语的子集（分别 152/29/192 句）。

**📈 对比分析**

与三种输入条件（仅系统响应、仅用户发言、系统+用户、前一次用户发言+当前）及五种异常检测方式进行对比。AUPRC 在“前一次用户发言+当前”条件下的异常检测在英语数据中平均可提升至约 20‑25%（相较基线 17.8%），在多语言环境中提升幅度因语言而异，未能在所有语言上保持优势。

**⚠️ 局限性**

局限性包括：①数据规模有限，未能进行对话级拆分，可能导致一定的上下文泄漏；②自报不满意标签主观性强，难以统一定义；③多语言实验样本不足，难以分辨语言/文化差异与样本量的影响；④仅在开放式 ChatGPT 对话上验证，结果可能不适用于任务导向型或多轮对话系统。

---

## 451. Conformal Risk Control for QoT-Aware Lightpath Feasibility Under Technology-Driven Domain Shift

**arXiv ID:** 2611.13893 | [PDF](https://arxiv.org/pdf/2611.13893v1)

**作者:** Kiarash Rezaei `[一作]` (Chalmers University of Technology), Carlos Natalino `[通讯]` (Chalmers University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出后验CRC框架，将预训练的AI/ML回归器转换为风险控制不确定性区间，并通过两种运营政策（RA、EO）实现QoT‑aware光路可行性决策，适用于技术升级导致的域迁移。

**💡 创新点**

创新点在于①使用LTT‑基的Conformal Risk Control实现有限样本覆盖保证的区间宽度选择；②将不确定性区间映射到可调节安全性与资源利用率的两种政策；③引入特征迁移贡献评分用于量化域偏移；④在17种技术迁移场景下系统验证该方法。

**🔧 技术方法**

采用Conformal Risk Control（LTT）、XGBoost回归、特征迁移贡献评分、OSNR预测与QoT阈值决策，构建不确定性区间并映射至运营决策。

**📊 数据集**

使用公开的SDM QoT数据集（CONUS拓扑），包含9个技术配置，划分为17个源‑目标场景（光纤、放大器、交换机升级组合）。

**📈 对比分析**

与源域QR、零样本ZS和目标域TL基准进行对比。LTT在所有170次试验中保持≥90%误覆盖率；在强域迁移下，EO政策将准确率从≈85%提升至>99%，接近TL；RA政策FPR接近0但FNR较高，整体性能优于基准。

**⚠️ 局限性**

局限性包括：需目标域校准样本；假设校准与测试样本可交换；无法自适应持续漂移或噪声变化；使用对称区间，未考虑区间方向性；依赖α、δ等超参数的设置。

---

## 452. Spatial-Spectral Coupling as Attention: Scaling Shape Correspondence to High-Resolution Surfaces

**arXiv ID:** 2611.13870 | [PDF](https://arxiv.org/pdf/2611.13870v1)

**作者:** Bence Demeter `[一作]` (Technical University of Munich), Lennart Bastian `[通讯]` (Imperial College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在高分辨率3D形状的稠密对应中提出Geo-FlashAttention，利用几何感知的稀疏注意力实现空间-谱耦合的精确软匹配；

**💡 创新点**

创新点是将空间-谱耦合等价于缩放点积注意力，进而应用FlashAttention的高效实现，并通过几何选取的局部补丁实现近线性规模的稀疏注意力；

**🔧 技术方法**

采用的技术包括缩放点积注意力（SDPA）、FlashAttention、线性注意力、Geo-FlashAttention以及KeOps的符号求和；

**📊 数据集**

使用的数据集包括FAUST、SCAPE、SHREC'19、SMAL、DeformingThings4D（DT4D）、TOPKIDS、Mindboggle-101等；

**📈 对比分析**

与传统稠密耦合方法相比，Geo-FlashAttention在保持相同或更低地理误差的同时，将峰值显存从106 GiB降至5.67 GiB，并在60k顶点级别比FlashAttention快约1.5倍；

**⚠️ 局限性**

局限性在于仅适用于通过矩阵乘法消费耦合的场景，候选补丁预算固定且缺乏理论误差上界，对皮层表面缺乏完整的地面真值验证；

---

## 453. Online Network Design with Additive Delays and Poisson Arrivals

**arXiv ID:** 2611.13859 | [PDF](https://arxiv.org/pdf/2611.13859v1)

**作者:** Tianhang Lu `[一作]` (Southern University of Science and Technology), Shengcai Liu `[通讯]` (Southern University of Science and Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出三种在线服务模型（设施定位、树上多根聚合、移动服务器）并给出基于已知泊松到达率的固定空间决策与可变时间调度的确定性算法，证明其长期期望成本与最优离线解相比的常数竞争比；

**💡 创新点**

创新点在于利用到达率信息在不观察未来请求的前提下做出固定空间分配（设施、根或服务器家园），并通过空间-时间证书、连通分配与根森林调度等技术实现全局常数竞争比；

**🔧 技术方法**

使用泊松过程概率工具、空间-时间证书、连通0扩展（Englert等）、根森林线性规划、两站点下界与中心点（centroid）均衡等技术；

**📊 数据集**

未使用实验数据集，完全基于理论模型（有限度量空间、树结构、已知泊松率）；

**📈 对比分析**

通过长期期望成本比对最优离线解进行比较，获得：设施定位<27.249、树上多根聚合<94.415、移动服务器<24.326；

**⚠️ 局限性**

局限性包括：设施与移动服务器算法的位数复杂度未保证多项式；常数可能非最优；结果仅适用于树或有限度量，无法直接推广到任意度量；未考虑在线学习到达率的情况。

---

## 454. Towards Text-Only Remote Sensing Scene Classification via Semantic Proxy

**arXiv ID:** 2611.13851 | [PDF](https://arxiv.org/pdf/2611.13851v1)

**作者:** Xianping Ma `[一作]` (Southwest Jiaotong University), Yuanxin Ye `[通讯]` (Southwest Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了仅使用文本语义代理（semantic proxy）进行遥感场景分类的新框架 SeProRS，并构建了可复现的文本仅分类基准。

**💡 创新点**

创新点在于：①将遥感场景识别从传统视觉输入转化为文本代理输入，彻底实现“图像无输入”分类；②设计了两种可复现的提示策略（Flat Prompt 与 Structured Prompt）来生成文本代理；③系统评估了多种 VLM（LLaVA-OneVision、Qwen2.5-VL、Gemini-2.5-Flash）对代理质量与跨生成器迁移性的影响。

**🔧 技术方法**

核心技术包括：①视觉‑语言模型（VLM）用于无标签生成语义代理；②通用文本编码器（如 CLIP‑Text）和轻量化投影层 + 线性分类器；③多维度实验评估（不同数据集、训练比例、代理生成器、提示方式）。

**📊 数据集**

使用了三大公开遥感场景分类数据集：UCM（21类）、AID（30类）和 NWPU‑RESISC45（45类）。

**📈 对比分析**

与完整图像输入的视觉模型（HFAM、CGINet、UPetu、MSCN、UniRepLKNet、DINOv3‑Adapter）相比，文本仅模型在 50%/80% 训练比例下分别取得 96.19%/95.39%（UCM）、93.95%/95.18%（AID）、89.13%/90.08%（NWPU‑RESISC45）的准确率，虽低于视觉基准，但仍显示显著的可行性；同时在参数量、内存和推理速度上远优于视觉模型。

**⚠️ 局限性**

局限性包括：①文本代理无法完整保留纹理、尺度、细粒度空间布局等低级视觉信息，导致与图像基准仍有明显差距；②不同 VLM 生成的代理存在生成器特定的语言风格和分布偏移，跨生成器迁移性能不理想；③框架整体仍以最简化设计为主，存在进一步提升代理质量、文本编码与投影适配的空间。

---

## 455. Uncovering the Memory Hierarchy of Modern NVIDIA GPUs for Accurate Simulation

**arXiv ID:** 2611.13860 | [PDF](https://arxiv.org/pdf/2611.13860v1)

**作者:** Ziwen Wang `[一作]` (Hong Kong University of Science and Technology), Yun Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了基于 SASS 级别的微基准框架，系统逆向工程并恢复了现代 NVIDIA GPU 的 L1、L2 和常量内存层级的细粒度结构与行为，并将其集成至 AccelSim 仿真器以提升 LLM 工作负载的预测精度。

**💡 创新点**

首次在全套级别揭示了 L1 的切片化、XOR 映射、跨片 eviction 与 SLRU 策略；L2 的分区/FBP/切片哈希与全局一致性；常量内存三级缓存的物理布局与 Tree-PLRU 替换，并提出了切片约束的 eviction‑set 构造方法。

**🔧 技术方法**

采用 SASS 微基准、位翻转分析、组消除、特殊寄存器读取等技术实现细粒度测量；利用 AccelSim 进行仿真验证；通过构建 -Bench 作为评测数据集。

**📊 数据集**

-Bench 包含 25 个传统 GPU 基准与 28 个来自 vLLM、PyTorch、FlashAttention‑3 的 LLM 算子，尺寸以 Qwen3‑30B‑A3B 模型为参考。

**📈 对比分析**

与基准 BL‑Impv 相比，模型在 4 个 GPU 上将平均 MAPE 从约 34% 降至 24%，对 LLM 算子从 36% 降至 21%，对内存敏感工作负载从 36% 降至 20%，并将相关性提升至 0.974。

**⚠️ 局限性**

仍未完全解析 GCC 的替换策略与跨片 eviction 的精确选取规则；仿真仅提升了内存层级精度，未解决其他硬件部件误差；对极端高并发和极限工作负载的泛化尚待进一步验证。

---

## 456. JAX-WAVE: A Differentiable Physics-Supervised Framework for Inverse Wave Design in Heterogeneous Media

**arXiv ID:** 2611.13826 | [PDF](https://arxiv.org/pdf/2611.13826v1)

**作者:** Honghui Du `[一作]` (University of Minnesota), Yushu Pan `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本工作设计并训练了一个共享的神经网络JAX‑WAVE，能够在已知异质介质中一次性预测满足目标终端场的受限初始压强分布，避免了每次都要进行数值迭代；

**💡 创新点**

其创新点在于：①利用物理对称性（压强符号对称关系）引入共享权重投影以强制满足逆问题的正负对称约束；②通过可微FDTD求解器实现无标签的物理监督训练；③将离散传播算子与其加权伴随算子关联，形成完整的可摊销逆设计框架；

**🔧 技术方法**

采用的技术包括：JAX + Flax实现的U‑Net结构、可微时间域有限差分(FDTD)求解器、共享权重对称投影、Adam + cosine learning‑rate训练、梯度下降、FISTA等数值优化以及自动微分；

**📊 数据集**

使用了合成数据集，包含两类目标（几何形状与可达目标）与两类介质（层状与包含椭圆的介质）组合，训练集规模为16k或65k条目，验证集256条，测试集1024条；

**📈 对比分析**

通过在5 ms在线预算内与投影梯度、FISTA、重启FISTA等传统数值逆解进行对比；网络在5 ms内平均终端场误差为18.8%，显著低于FISTA的34.2%；在独立材料集上，几何目标误差为25.2%，可达目标误差为4.8%，与传统迭代求解相比误差下降约45%；

**⚠️ 局限性**

局限性包括：对几何目标的高频细节仍难以复现，网络对光滑介质的泛化能力有限；模型容量提升后收益递减；需要固定网格和终端时间，难以一次性处理多时间点或多尺度控制问题。

---

## 457. CASCADE: Constraint-Aware SCAffolded DEcomposition for Logical Anomaly Detection

**arXiv ID:** 2611.13820 | [PDF](https://arxiv.org/pdf/2611.13820v1)

**作者:** Jinchan Kim `[一作]` (Ajou University), Hyunsouk Cho `[通讯]` (Ajou University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个无训练的视觉辅助逻辑异常检测框架CASCADE，在原有检查清单分解基础上为每个约束生成相应的视觉支架，以实现对原始图像的约束特定可视化并提高判定精度。

**💡 创新点**

创新点在于将约束特定视觉支架与文本检查清单相结合，首次系统性定位并解决实例混淆与尺度漂移两大感知瓶颈，并通过约束结构与参考分布动态配置支架参数。

**🔧 技术方法**

使用技术包括大型视觉语言模型（如GPT‑5‑image‑mini、Gemini‑3.1‑Flash‑Lite、Qwen3‑VL‑32B等）、SAM3分割、视觉支架渲染、约束解析器和检查清单生成器等。

**📊 数据集**

实验数据集为MVTec‑LOCO（5类共16约束）和VID‑AD（10类共20约束），覆盖绑定与标定两类逻辑约束。

**📈 对比分析**

与PatchCore、EfficientAD、LogSAD、UniVAD等基线以及GPT‑4o、Gemini‑3.1‑Flash‑Lite、Qwen3‑VL‑32B等模型对比，MVTec‑LOCO上无训练AUROC 93.2，超越全量训练SOTA 89.3；VID‑AD上AUROC 94.5，领先基线11.4点，仅使用3张参考图。

**⚠️ 局限性**

局限性包括对分割质量敏感，无法同时处理需要计数与测量的混合约束；推理时间相对较长（≈70秒），不适合极高吞吐量的在线检测。

---

## 458. Antalia-2 Mini: A 7.6-million-parameter Turkish text-to-speech model trained entirely on synthetic speech

**arXiv ID:** 2611.13811 | [PDF](https://arxiv.org/pdf/2611.13811v1)

**作者:** Sezgin Saygili `[一作]` (PatientDesk AI), Fikri San Koktas `[通讯]` (PatientDesk AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款 7.62M 参数的单声线 Turkish TTS 模型 Antalia‑2 Mini，使用仅 1.8k 小时的合成语音训练，能够在 CPU 上以 35–41 倍实时速率生成 48kHz 语音。

**💡 创新点**

创新点在于：① 只用单一大型合成教师 VoxCPM2 生成的数据训练学生模型；② 采用“shortcut self‑consistency” 结合 diffusion transformer 生成 Mel，显著降低采样步数；③ 在生成 Mel 上细粒度地对声码器进行自适应微调，显著减少 WER；④ 引入隐藏前缀与双音节延长规则，修复教师引入的起始音错误。

**🔧 技术方法**

技术包括：字符级文本编码、ConvNeXt + 旋转位置嵌入的自注意力编码器、基于流匹配与 shortcut self‑consistency 的 Diffusion Transformer 生成器、Vocos 结构的声码器、Classifier‑Free Guidance、动态持续时间预测、短音频微调、以及 5.74‑6.16 级的自动化质量控制（CER、F0、UTMOS 等）。

**📊 数据集**

训练数据来自 VoxCPM2 合成音频（约 1,790 小时），拆分为长句（约 1,580 小时）和短句（约 226 小时）两套；评估采用 Freya‑TR‑Eval 495 句 Turkish 语料，提供 8 kHz 频段匹配的 WER/CER 以及 UTMOS 强化分数。

**📈 对比分析**

在公共 Harness 上与 EMA Lightning 等 15 系统对比，Antalia‑2 Mini 的 WER 1.09%（CER 0.22%）与 EMA Lightning 1.04%（CER 0.23%）无显著差异，均在 1% 以内；在自有 Harness 上 WER 0.97%（CER 0.19%），与教师 VoxCPM2 的 0.92% 也无显著差异。RTF 0.024–0.029（Apple M5 / NVIDIA L4）表现优于多种大型模型；仅 7.6M 参数即可匹配 2–3B 参数模型的错误率。

**⚠️ 局限性**

局限性：缺乏人工听力评测；训练完全基于合成语音，未验证与真实说话者的相似度；教师在句首 ö 开头时存在错误，模型虽通过前缀隐藏但仍未彻底解决；仅支持 Turkish 单声线，无情感/风格控制；未在真实手机或高并发环境下进行测速；评估基于 Whisper 识别的自动指标，可能低估真实可懂度；缺少针对短语/外来词等边缘情况的针对性评测。

---

## 459. From Counterfactual Edits to Human Narratives: LLM-Driven Interpretations for Recommender System Explanations

**arXiv ID:** 2611.13914 | [PDF](https://arxiv.org/pdf/2611.13914v1)

**作者:** Vojtech Cahlik `[一作]` (Czech Technical University in Prague), Eva Zangerle `[通讯]` (University of Innsbruck)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用大型语言模型（LLM）将推荐系统中产生的反事实编辑列表转化为简洁、可读的自然语言解释，提升用户对推荐结果的可解释性。

**💡 创新点**

创新点包括：① 将反事实编辑与LLM相结合，形成中介层生成语义化解释；② 设计分解的评估指标（正确性、信息量、语言质量），并通过人类标注进行校准；③ 采用Direct Preference Optimization（DPO）优化LLM，使生成的解释在语义与可读性上均得到提升。

**🔧 技术方法**

使用技术：反事实解释生成器（ACCENT、SPINRec 等）、大型语言模型（Ministral 8B、Gemma 3‑12B、Qwen3 8B）、DPO 优化、基于人类标注的校准评估器（ModernBERT‑large 二分类器）等。

**📊 数据集**

实验数据集：MovieLens‑1M 电影推荐数据集，使用变分自编码器推荐模型生成反事实编辑。

**📈 对比分析**

对比方法：在多种反事实生成器与多种 LLM 组合上评估 CFX 成功率、编辑数、简洁率、正确性、信息量与语言质量。结果显示，ACCENT 与 SPINRec 生成的反事实编辑成功率最高，且与 DPO 优化的 LLM 配合后得到最高的正确性和信息量（≈0.85），语言质量普遍在 0.8 以上。

**⚠️ 局限性**

限制：① 仍难以处理极大编辑数（>50）导致解释失效；② 简单编辑率低，说明大多数情况仍需摘要；③ 仅在单一数据集与单一推荐模型上验证，泛化性待进一步测试；④ LLM 生成的解释受训练语料影响，可能与实际用户需求不完全一致。

---

## 460. CAppleNet: A Coordinate Attention-Based YOLO11 for Real-Time Apple Stem Cavity and Calyx Detection

**arXiv ID:** 2611.13920 | [PDF](https://arxiv.org/pdf/2611.13920v1)

**作者:** Alican Güneş `[一作]` (Ege University), Murat Kurt `[通讯]` (Ege University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

在固定摄像头与照明条件下，基于YOLO11s的基础模型，对苹果的果体、茎腔和萼片三类细粒度标记进行了系统的单因素消融实验；

**💡 创新点**

首次将输入分辨率、数据增强、优化器、轻量化结构模块及推理阈值等关键设计因素逐一拆解评估，发现采用Coordinate Attention替代C2PSA显著提升茎腔检测；

**🔧 技术方法**

使用YOLO11s网络并在其C2PSA位置替换为Coordinate Attention，结合Ultralytics默认数据增强和优化器设置，采用PyTorch实现训练与推理；

**📊 数据集**

构建自采集的1506张苹果图像数据集，手工标注果体、茎腔、萼片三类，并按1054/301/151划分为训练、验证与测试集；

**📈 对比分析**

通过对比基线模型与各消融变体在mAP@0.5、mAP@0.5:0.95、精度、召回率、F1等指标，基线F1为0.9047，Coordinate Attention替换模型F1 0.9038、mAP@0.5 0.935，其他因素未能显著提升性能；

**⚠️ 局限性**

受限于单一固定拍摄环境、数据规模有限、仅评估YOLO11s与单一注意力模块、未全面探索跨因子组合及推理延迟一致性，结果的推广性与适用于更动态环境的适用性尚待进一步验证。

---

## 461. Region-Wise Logit Fusion and Utility-Gated Refinement for Brain Metastasis Segmentation

**arXiv ID:** 2611.13884 | [PDF](https://arxiv.org/pdf/2611.13884v1)

**作者:** Zhenye Lou `[一作]` (Hong Kong Polytechnic University), Zhen Chen `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种名为COMPASS-METS的多训练器概率组装与结构化概率锚点的脑转移灶分割方法。

**💡 创新点**

创新点在于：① 多训练器对齐的区域概率锚点；② 基于父级支持的ET增量提炼；③ 利用存在性与几何安全性的LightGBM门控进一步过滤孤立候选；

**🔧 技术方法**

使用了三种五折nnU‑Net（ResEncXL、ResEncM、病变敏感ResEncM）生成概率，结合logit融合、区域阈值滤波、LightGBM分类器和集合后操作。

**📊 数据集**

使用BraTS‑METS 2026 Task 1的数据集，包含1295个训练样本和179个验证样本，包含多参数MRI（T1n、T1c、T2‑FLAIR、T2w）。

**📈 对比分析**

与单一ResEncXL以及纯平均/多数投票的对照组相比，COMPASS‑METS在179例验证集上小病灶F1≈0.354、DSC≈0.712、NSD≈0.714、All F1≈0.779、HD95≈56.6 mm；在1295例留出评估中较ResEncXL提升约0.033（F1）、0.031（DSC）、0.024（NSD）、0.023（All F1）并将HD95减少约5.8 mm。

**⚠️ 局限性**

局限性包括：仅在内部开发集上评估（选择性偏倚）；RC（切除腔）表现不稳定且缺乏独立验证；模型量大（15个子模型）且未进行外部/前瞻性验证；使用固定候选提议，未实现完全端到端学习。

---

## 462. Ceci n'est pas une pipe -- Toward a Symbolic Solver for the Abstraction and Reasoning Corpus Using Evolutionary Computation

**arXiv ID:** 2611.13831 | [PDF](https://arxiv.org/pdf/2611.13831v1)

**作者:** Luca Favole `[一作]`, Alberto Tonda `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 FairlySymbolic，一种完全符号化的进化算法，用以在 ARC‑AGI 任务中同时演化解析器、程序和投影器。

**💡 创新点**

创新点在于联合进化表示与程序、基于核心知识的对象感知适应度以及针对 ARC 的对象级评估。

**🔧 技术方法**

采用遗传编程、图形解析、DSL 规则集、对象级距离度量以及进化搜索与局部精炼技术。

**📊 数据集**

使用 ARC‑AGI‑1、ARC‑AGI‑2 以及 ConceptARC 数据集进行实验。

**📈 对比分析**

与现有符号与神经方法对比，FairlySymbolic 在 ARC‑AGI‑1 训练集上达到 219/400 任务，公测 1 级 110/400，虽然低于最强神经模型，但在无神经网络的条件下已处于领先符号系统水平。

**⚠️ 局限性**

局限在于对复杂组合与高级抽象的支持不足，ARC‑AGI‑2 公共测试仅 3/120 任务，且部分任务受限于 DSL 与解析器能力。

---

## 463. Learn-To-Propagate: Distributionally Robust Motion Planning for Nonlinear Uncertain Systems

**arXiv ID:** 2611.13863 | [PDF](https://arxiv.org/pdf/2611.13863v1)

**作者:** Eduardo Figueiredo `[一作]` (Delft University of Technology), Luca Laurenti `[通讯]` (Delft University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于分布鲁棒优化的采样式运动规划算法L2P-DR-Forest，能够在未知初始状态和过程噪声分布下为非线性随机系统生成高置信度安全轨迹。

**💡 创新点**

1) 结合最优传输与分布学习构建可自适应的Wasserstein模糊集；2) 在树搜索中实现模糊集的非线性传播和重学习，形成多树森林策略；3) 在理论上证明算法的正确性与概率完备性。

**🔧 技术方法**

Wasserstein距离与模糊集、分布鲁棒优化、离散分布学习、非线性不确定性传播、RRT采样式搜索、MILP求解、Lipschitz传播规则。

**📊 数据集**

五个机器人基准：2D 5模态切换系统、AV Maze、Unicycle Maze、4D Tractor Trailer（神经网络动力学）、6D 双积分器。

**📈 对比分析**

与ϵ-RandUP-RRT基线对比，L2P-DR-Forest在所有基准上实现100%成功率，规划时间远低于对照方法，且在Monte-Carlo验证中无违例、全通达率。

**⚠️ 局限性**

在高维系统中样本复杂度高、对Lipschitz常数依赖强、需要大量噪声样本，导致可扩展性受限。

---

## 464. Taxonomy-Agnostic Neurosymbolic Grounding of Robot Kinematics

**arXiv ID:** 2611.13850 | [PDF](https://arxiv.org/pdf/2611.13850v1)

**作者:** Bastien Dussard `[一作]` (Université de Toulouse), Guillaume Sarthou `[通讯]` (Université de Toulouse)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

自动将原始URDF文件转换为符合任意用户提供的OWL本体的语义化本体化表示。

**💡 创新点**

提出多通道递归图清理与双轨（树向/链向）神经符号抽取框架，并通过LLM上下文分块、多投票与层级分类实现对任意TBox的兼容。

**🔧 技术方法**

利用本地LLM进行语义推理，配合多数投票、上下文批处理、层级分类等神经符号守护；使用图结构递归清理、描述逻辑本体推理与OWL本体化。

**📊 数据集**

在包含四足Solo、移动抓取RB‑VOGUI+、PR2、iCub、Ingenuity等多模态机器人URDF的自构实验数据集以及自定义本体。

**📈 对比分析**

与手工注释本体对比，实验显示抽取精度>90%，显著减少手工投入，误判率低；在五台机器人上实现了高语义对齐和结构映射。

**⚠️ 局限性**

受限于对链接命名语义的依赖、TBox静态化导致对新概念缺乏适应性、未实现DL推理器实时反馈、对极稠密URDF时LLM上下文溢出风险，以及对混合树/链结构粒度处理的不足。

---

## 465. Game-Enhanced Embodied Learning with Event and Foresight Pretraining

**arXiv ID:** 2611.13819 | [PDF](https://arxiv.org/pdf/2611.13819v1)

**作者:** Lianlei Shan `[一作]` (Beijing Xinying Suixing Technology), Xin Bin `[通讯]` (Beijing Xinying Suixing Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文利用人类游戏记录的事件和视觉信息，对 InternVLA‑A1.5 进行事件预测与视觉预知预训练，并在此基础上进行机器人细调。

**💡 创新点**

创新点在于将非机器人键鼠事件与视觉预知相结合，使用覆盖保留的游戏样本、因果注意力屏蔽以及冻结视频模型的监督，直接补充机器人可用的嵌入式监督。

**🔧 技术方法**

技术包括事件与视觉预知联合预训练、因果注意力掩码、覆盖保留采样、Frozen Wan2.2 VAE 视频监督、以及桥接机制将游戏训练权重映射到机器人前向路径。

**📊 数据集**

使用了 D2E 选取的 20 款游戏的 394 条录制，共 223 小时游戏时长、超过 2.07M 训练锚点的数据集。

**📈 对比分析**

通过在 LIBERO 四个套件（Spatial、Object、Goal、Long）上使用相同的机器人微调协议对比，游戏预训练初始化在 2000 次测试中提升成功率从 85.90% 提升到 90.55%，总体增加 93 次成功。

**⚠️ 局限性**

局限性包括未能将提升归因于单一目标（事件预测或视觉预知）、训练预算差异无法隔离；结果仅适用于选定游戏、模型与仿真环境，且任务层面存在提升与下降的混合表现。

---

## 466. Program-Aided Patchable Skill Evolution

**arXiv ID:** 2611.14019 | [PDF](https://arxiv.org/pdf/2611.14019v1)

**作者:** Jiahao Wang `[一作]` (Northeastern University), Yubin Bao `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Program‑Aided Patchable Skill Evolution（PAPSE）框架，利用可执行程序解耦技能文本与过程逻辑，并通过增量补丁机制迭代演化LLM代理的技能。

**💡 创新点**

创新点在于：①将可重用的程序化流程提取为独立程序，保持技能文本的可读性；②采用补丁式共同演化（patch‑based joint evolution）对技能文本与程序分别进行细粒度更新；③通过工具化的上下文读取实现自适应信息检索，避免信息噪声。

**🔧 技术方法**

核心技术包括：LLM ReAct 推理、可执行程序调用、补丁生成器（React‑based proposer）、工具化上下文检索、验证门控（validation gating）与迭代演化。

**📊 数据集**

实验使用的基准数据集包括 SpreadsheetBench、Domain‑Spreadsheet、WikiTableQuestions 与 HiTab，分别用于评估表格操作、表格问答与跨领域适用性。

**📈 对比分析**

与基线（No Skill、Human Skill、LLM Skill、Trace2Skill、EvoSkill）以及不同模型（Qwen、Gemma）进行对比，PAPSE 在所有基准上均优于对手，最大提升约 2.8%（整体）且在 SpreadsheetBench‑Verified 上提升超过 14%，并在跨域/跨模型迁移中表现出更强的泛化能力。

**⚠️ 局限性**

局限性包括：未考虑技能检索对性能的影响；程序执行依赖外部工具，若代理自行实现流程则难以保证程序化路径；验证门控仅接受提升验证分数的补丁，可能忽略保持性能但有潜在长远价值的中性更新。

---

## 467. WAM is capable in BEHAVIOR-1K: the 2nd-place solution to the 2026 BEHAVIOR Challenge

**arXiv ID:** 2611.13818 | [PDF](https://arxiv.org/pdf/2611.13818v1)

**作者:** Yelin He `[一作]` (Shenzhen Intellifusion Technologies Co Ltd), Rong Xiao `[通讯]` (Shenzhen Intellifusion Technologies Co Ltd)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对2026年BEHAVIOR挑战，提出基于世界动作模型（Cosmos 3）的完整端到端移动操作系统，结合视觉历史、任务进度估计、任务组LoRA与技能专属LoRA，以及动作候选聚合、滚动修补和时间重采样，实现高效推理。

**💡 创新点**

创新点包括：1）将视觉历史与任务进度作为显式条件注入WAM；2）利用任务组与技能特定LoRA进行路由优化；3）使用进度估计器在推理时动态选择策略；4）在推理中引入滚动修补和时间重采样来提升动作连贯性与速度。

**🔧 技术方法**

核心技术为Cosmos 3（WAM）+ 视觉-VAE + mRoPE编码 + 2层Transformer阶段估计 + SigLIP-B/16子任务预测 + 5个动作候选聚合+ UniPC + TOPP‑RA时域重采样 + LoRA微调。

**📊 数据集**

使用BEHAVIOR 2026任务集：100个家庭任务，每个任务200条演示（约1950小时），OmniGibson 3.9.x仿真环境；同时在公开和私有测试集上评估。

**📈 对比分析**

在官方私测中获得Q-score 0.615、完整任务成功率41.2%，在公开测中Q-score 0.613；相较于2025年最佳VLA方案（Q-score 0.260）提升显著，排名第二；实例挑选可达Q-score 0.891、成功率79.5%。

**⚠️ 局限性**

局限性：1）对实例/任务的依赖性强，性能在不同配置下波动；2）计算资源受限，未能充分探索更大模型与更长训练；3）未使用递归自我提升或agentic harness，导致在极端长序列时可能出现累积误差；4）对硬件要求较高，单端推理需56 GB GPU内存。

---

## 468. BioXArena-Pro: a research-grade benchmark for autonomous biomedical machine learning agents

**arXiv ID:** 2611.13994 | [PDF](https://arxiv.org/pdf/2611.13994v1)

**作者:** Loka Li `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Le Song `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并发布了 BioXArena‑Pro 基准，包含 64 个从真实研究中重建、保持原始数据格式、采用 OOS 拆分的多领域（化学生物、成像、网络、干扰、表型、序列、单细胞、结构）任务，用于评估自驱动机器学习代理的端到端建模能力。

**💡 创新点**

创新点包括：① 任务规模与多样性大幅提升，保持完整实体拆分与原始格式；② 通过四轮 AI + 人工审核彻底剔除泄漏、短路与分数操纵；③ 对比重现出版方法与代理配置，揭示适配性、预训练模型价值与脚手架对性能的影响。

**🔧 技术方法**

采用大型语言模型代理（Codex、Claude、Gemini 等）与四种模型框架（OpenCode、Biomni、MLEvolve、最小脚手架）交叉评估；使用 0–100 归一化评分体系；任务拆分采用无识别 ID、加密文件与多源输入；对代理代码、提示、调优流程进行完整公开与复现。

**📊 数据集**

使用了 76 个来源于 Nature/Science/Cell 及主流公开数据库（如 CSD、CheMixHub、DrugComb、MassSpecGym、Cohorts、PDB、PDB2022、CPTAC、TCGA、GEO 等）的原始数据，涵盖化学结构、图像、基因表达、蛋白质序列、空间坐标、时间序列等多模态。

**📈 对比分析**

通过 22 种代理配置（三种 CLI + 两种 backbone，四种脚手架 + 四种 backbone）与重现的出版方法（tier1/2）对比，最强配置 Codex CLI GPT‑6 Astra 在 0–100 评分上达 55.1，平均 54.5；与原始集任务接近时与出版方法相当；在重新拆分任务中超越出版方法 8.2 分；预训练模型提升约 9 分；与旧基准相比平均得分下降 11 分。

**⚠️ 局限性**

局限包括：受 2 小时 GPU 限制；某些任务可能因网络访问导致泄漏；未评估噪声上限和人类专家对比；预训练模型可能存在数据重叠风险；部分拆分仍可能隐藏未检测的捷径；任务复杂度导致部分配置表现不稳定。

---

## 469. Data Quality Challenges in Multimodal Tourism Recommender Systems

**arXiv ID:** 2611.13972 | [PDF](https://arxiv.org/pdf/2611.13972v1)

**作者:** Zehui Wang `[一作]` (University of Applied Sciences Ravensburg-Weingarten), Dietmar Jannach `[通讯]` (University of Klagenfurt)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

分析了旅游推荐系统中多模态数据质量问题，并提出未来研究方向

**💡 创新点**

首次系统梳理噪声、偏差与跨模态矛盾等质量挑战，并提出数据感知融合与结构化LLM生成等创新思路

**🔧 技术方法**

利用行为模式识别、结构化摘要、数据感知多模态学习等技术进行数据清洗与融合

**📊 数据集**

使用Yelp多模态数据集和来自德国湖区的Instagram数据集（受限公开）

**📈 对比分析**

论文未给出实验对比与性能指标，而是通过案例分析和理论讨论阐述问题与解决方案

**⚠️ 局限性**

数据可获取性有限、对不同地区和用户群的泛化不足，且缺乏实证评估验证所提方案的有效性

---

## 470. Universal Approximation Theorem for Metaplectic Neural Operators

**arXiv ID:** 2611.13991 | [PDF](https://arxiv.org/pdf/2611.13991v1)

**作者:** Ahmed Abdeljawad `[一作]` (Austrian Academy of Sciences), Malek Yacine Rezoug `[通讯]` (Austrian Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

提出了一种基于Metaplectic变换的神经算子架构（MpNO），并证明在任意固定自由辛矩阵下，该架构能在 C^s → C^r 和 H^s → H^r 范数下实现紧致一致逼近；

**💡 创新点**

创新点在于将整个空间的 Metaplectic 乘子与零延拓/限制相结合，构造出仅需单一非局部分支即可实现通用逼近，并给出了对导数范数的严格控制；

**🔧 技术方法**

采用了辛群与 Metaplectic 变换理论、平均神经算子技术、符号类构造及正则性估计等数学工具；

**📊 数据集**

本文为理论研究，没有使用任何实验数据集；

**📈 对比分析**

由于是纯理论证明，未进行实验对比；展示了在 C^r 和 H^r 范数下逼近误差可任意小；

**⚠️ 局限性**

局限包括：需要预先固定辛变换、输入输出维度有限、对零延拓的前置处理，以及缺乏数值实现和复杂度分析。

---

## 471. Understanding Mixture-of-Experts Language Models Beyond Expert Boundaries

**arXiv ID:** 2611.13975 | [PDF](https://arxiv.org/pdf/2611.13975v1)

**作者:** Zhenhong Zhou `[一作]` (Nanyang Technological University), Junfeng Fang `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过“路由感知幅度剖面”（RAMP）分析Mixture-of-Experts（MoE）模型中单个神经元在不同领域间的专门化，揭示了域级计算主要由跨专家的神经元集合实现，而非单个专家。

**💡 创新点**

创新点在于①从神经元层面而非专家层面考察域专门化；②提出RAMP量化神经元在路由权重下的贡献；③基于RAMP生成跨专家共享的稀疏神经元集，并用RAMP‑based Pruning（RAMPr）实现高效压缩，兼顾多域性能与显存/吞吐改进。

**🔧 技术方法**

技术主要包括：路由感知幅度剖面（RAMP）、基于RAMP的层级/域平衡剪枝（RAMPr‑L、RAMPr‑D）、对比专家级和神经元级重要性变化、对MoE训练过程的检查（预训练与后续微调阶段）。

**📊 数据集**

数据集覆盖11个域（对话、数学、医学、中文文本、代码等），每域使用128条校准文本；评估采用13项知识推理与生成任务，结合公开基准如GLM‑4.7‑Flash、Qwen 系列与 OLMoE 等六大 MoE 模型。

**📈 对比分析**

与 Fisher‑based 神经元剪枝和 REAP（专家剪枝）对比，RAMPr 在 50% 神经元保留时实现 91–98% 的原模型性能，且在 GLM‑4.7‑Flash 上 75% 保留时相较 REAP 提升 2.91 分；显存降低 36–43%，prefill 吞吐提升 17–39%，decode 近乎不变。

**⚠️ 局限性**

局限性包括：①仅在已公开的几款 MoE 模型上验证，可能不完全适用于其他架构；②依赖 128 条校准文本，性能受校准数据规模和领域选择影响；③RAMP 计算和剪枝实现相对复杂，部署时需额外算子与布局支持；④对极端小模型或低精度部署的适用性尚未深入研究。

---

## 472. Pulse or Provenance? What rPPG-Based Deepfake Detectors Learn

**arXiv ID:** 2611.13992 | [PDF](https://arxiv.org/pdf/2611.13992v1)

**作者:** Othmane Harraq `[一作]` (Temple University), Tamer Aldwairi `[通讯]` (Temple University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对远程光血氧测量（rPPG）深度伪造检测器进行系统研究，揭示其判别信号主要来自图像到视频合成的高频色彩闪烁，而非真实心跳；并指出这些检测器往往编码视频来源与数据集偏差。

**💡 创新点**

创新点包括：① 通过匹配采样、频段过滤、帧打乱等对比实验，精确定位判别信号的时间频率特征；② 在自我重现实验中证明高频信号由“单帧动画”方式引入；③ 发现检测器对视频来源极其敏感，能够将不同来源的真实视频误判为伪造；④ 提出了若干评价与验证控制（背景/全帧检测、帧打乱、频段对比、重编码、跨源/身份离散分割），帮助未来研究者评估“生理”检测器的真实性。

**🔧 技术方法**

使用的技术包括：CHROM、POS 等经典色度提取、RhythmFormer 深度 rPPG 提取器、1D ResNet 分类器、CLIP ViT-L/14 预训练特征、Xception 视觉特征、离散余弦变换 (DCT) 频域描述子，以及 FFT 频段滤波与能量分析。

**📊 数据集**

主要使用公开数据集 Celeb-DF++（包含 22 种伪造方法，TF/FR/FS 子集）与 FaceForensics++（FF++）进行实验，并对官方 Celeb-DF/DF++ 的训练/测试分割进行细致剖析。

**📈 对比分析**

比较方法：在严格身份分离、匹配窗口长度、全帧/背景/面部三种区域下计算 AUC；对比 CHROM、POS、RhythmFormer、CLIP、Xception、融合模型。结果显示：CHROM 在 TF 子集上 AUC 0.998，远超专用 rPPG 提取器 0.87；CLIP 仅 0.957，融合模型提升不超过 0.004；在 FF++ 及身份离散协议中，所有方法性能显著下降，说明先前高分是因数据重叠导致。

**⚠️ 局限性**

局限性：① 未获取真实心率基准，无法确定伪造视频是否真存在心跳；② 只对两种重现生成器做自我重现实验，难以推广到所有合成算法；③ 结果高度依赖 Celeb-DF++ 的数据分割与视频编码方式，跨数据集泛化差；④ 评估主要聚焦 rPPG 轨迹分类，未涉及音频同步或多模态检测器。

---

## 473. Controlled Knowledge Engineering for Taxonomy Induction with Tool-Grounded LLM Agents and Runtime-Adaptive Policies

**arXiv ID:** 2611.13966 | [PDF](https://arxiv.org/pdf/2611.13966v1)

**作者:** Jonas Limniatis `[一作]` (Lufthansa Technik), Robert Stahlbock `[通讯]` (University of Hamburg)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种基于工具驱动、可运行时自适应策略的 LLM 代理框架 TIGA，用以在显式状态空间上迭代构造实体中心化分类法。

**💡 创新点**

创新点在于将分类法构建过程拆解为可观测的工具调用、外部化记忆与可动态修订的组织策略，弥补传统自由文本生成的缺陷，并显著提升层次结构完整性与一致性。

**🔧 技术方法**

使用大型语言模型（GPT‑5.4‑mini 与 Gemma‑4‑31b）作为决策核心，配合四种可调用工具、运行时策略更新和可选的全局审阅器，实现了控制知识工程的迭代推理。

**📊 数据集**

在 WordNet、DBLP、TExEval‑2 及 Google Product Taxonomy（子集）四个公开基准上进行实验，涵盖浅层与深层、宽度与高度多样的分类法规模。

**📈 对比分析**

与单次提示（HF）和层级提示（CoL）两大基线相比，TIGA 在大多数设置下获得最高的 F1（尤其是召回和 F1），但在精度、运行波动和计算成本上表现不一，显示出显著的结构质量提升但更高的 token 与 API 调用消耗。

**⚠️ 局限性**

主要限制包括：① 计算成本与延时显著增加；② 在某些数据集与模型上精度不一定提升且波动更大；③ 受限的审阅预算导致错误修复不完整；④ 仅评估了有限的模型与基准，未覆盖多父关系或更大规模的知识图谱。

---

## 474. AVCC: Audio-Visual Conversation Corpus for Turn-Taking Prediction with a Human Benchmark

**arXiv ID:** 2611.13957 | [PDF](https://arxiv.org/pdf/2611.13957v1)

**作者:** Haotian Qi `[一作]`, Gabriel Skantze `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

把AVCC语音视觉数据集扩展为公开的多方下一说话者预测（NSP）基准，并提供测试分区、词级转录和人类基准。

**💡 创新点**

创新点在于提供单摄像头单麦克风的真实多方对话数据、统一的预测事件点、人类观察者表现基准，并对模型进行公平比较。

**🔧 技术方法**

使用音频、视觉和词汇特征的RoleVAP（音频VAP）以及多模态MuVAP模型，并在测试集上进行评估。

**📊 数据集**

使用AVCC 31小时的两、三方在线对话视频，划分训练/验证/测试，包含865个hold和865个shift事件。

**📈 对比分析**

与人类单独观测者和多数投票比较，MuVAP在NSP准确率63.3%（比人均高3.8%），RoleVAP在hold/shift准确率66.5%（比人均高2.5%），表现略低于多数投票。

**⚠️ 局限性**

局限包括观测者仅看到10秒上下文而未听到或看到间隙，导致对短间隙预测更好；缺少多摄像头/多麦克风信息；以及对多人shift的预测仍不理想。

---

## 475. Chem World: A Large-Scale Benchmark and Physics-Aware Framework for Trustworthy Molecular Property Prediction

**arXiv ID:** 2611.13945 | [PDF](https://arxiv.org/pdf/2611.13945v1)

**作者:** Tianyou Bai `[一作]` (Institute of Automation, Chinese Academy of Sciences), Siming Dong `[通讯]` (Cleer Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

创建了Chem World大规模基准，并在其上提出PACM模型用于多组分化学系统属性预测。

**💡 创新点**

创新点在于统一整合17个公开数据集至800k+实例、10条属性轨道，并提出结合预训练分子表示与物理约束的组合建模框架。

**🔧 技术方法**

使用预训练分子语言模型（MolFormer、MolT5）、交互式聚合与温度/组成/边界约束的正则化技术。

**📊 数据集**

利用17个公开数据集（涵盖密度、导电率、混合焓、粘度等10条属性轨道），涵盖单分子与多组分系统。

**📈 对比分析**

通过与描述符+XGBoost、GNN、UNIFAC、COSMO-SAC、NRTL等基线对比，PACM在宏观NRMSE上达0.367、R² 0.858，显著优于传统方法。

**⚠️ 局限性**

局限性包括数据源异质性导致的测量协议差异、分组拆分未完全保证对未知分子/组分/条件的泛化，以及物理约束依赖任务假设。

---

## 476. Safety and Connectivity Certificates for Finite-Time Formation Transitions

**arXiv ID:** 2611.13927 | [PDF](https://arxiv.org/pdf/2611.13927v1)

**作者:** Yash Purswani `[一作]` (Indian Institute of Technology Madras), Anuj Tiwari `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

设计并验证了一种分布式第一阶运动学协议，实现任意平滑轨迹的有限时集合跟踪，并给出了离散时刻的收敛时间界定。

**💡 创新点**

提出了基于协同误差收敛的安全与连通性证明方法，利用误差上界生成离线耦合增益选择规则，并提供了对碰撞与通信连通的充分条件。

**🔧 技术方法**

使用分数幂反馈、拉普拉斯矩阵、Filippov解理论、误差投影与Rayleigh商估计等非线性控制与网络分析技术。

**📊 数据集**

没有使用公开数据集，而是通过数值仿真（10 机器人二维星形压缩/扩展）验证理论。

**📈 对比分析**

通过与理论收敛时间、误差衰减曲线、命令速度幅值和控制代价（积分能量）对比，展示了增益提升可显著缩短收敛时间但伴随命令速度与能量增大；安全证书的保守性通过比较增益 70 与 30、1 的仿真结果验证。

**⚠️ 局限性**

局限性包括：证书为充分不必要，可能过于保守；仅适用于第一阶运动学模型，未考虑推力/加速度约束；增益上限受实际执行器限制，且高增益会导致命令速度过大。

---

## 477. Geometry-Dependent Approximation and Stability Properties of Shallow ReLU$^k$ Networks on Bounded Euclidean Domains

**arXiv ID:** 2611.13952 | [PDF](https://arxiv.org/pdf/2611.13952v1)

**作者:** Yuqing Liu `[一作]` (RPTU University Kaiserslautern-Landau), Jinchao Xu `[通讯]` (Shenzhen Loop Area Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了利用有限数量的ReLU型神经网络对 Sobolev 与 Besov 空间内函数进行近似的理论上限和实现方法，给出了精确的误差与系数上界，并证明了这些界限的最佳性。

**💡 创新点**

创新之处在于将超平面分布的几何特性（球面上极轴分离与球冠包装）与核矩阵谱特征相结合，首次给出了神经网络近似的下界，并通过分层频域分析实现了上界的精细控制。

**🔧 技术方法**

主要技术包括傅里叶变换与复合算子法、Littlewood‑Paley 频分解、波形系数与 Besov 空间的等价性、采样不等式与卷积核估计、以及对核矩阵特征值的渐近估计。

**📊 数据集**

由于论文为理论分析，未使用任何公开数据集，所有示例均为构造性的理论函数。

**📈 对比分析**

通过与已知的分段多项式与球面波形网络的理论误差上界对比，本文提供的误差上界与下界在 Sobolev 与 Besov 规范下均达到最优阶，显示了在相同参数规模下可获得最优逼近率。

**⚠️ 局限性**

局限性主要体现在对网络结构的严格假设（如参数集满足球面均匀分布、激活函数为阶数 k 的多项式 ReLU）以及在实际实现中对样本点分布的高质量要求，且缺乏数值实验验证。

---

## 478. One Sequence Is Enough: Fingerprinting Language Models from Fixed-Length Integer Sequences

**arXiv ID:** 2611.14010 | [PDF](https://arxiv.org/pdf/2611.14010v1)

**作者:** Fantong Meng `[一作]` (Harbin Institute of Technology), Xinyang Li `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究利用固定长度整数序列进行语言模型身份识别，构建 benchmark 并提出 SMCT 方法。

**💡 创新点**

提出结合统计特征与因果 Transformer 的双编码架构 SMCT，实现单个整数序列高精度模型识别，并公开首个固定长度整数序列 benchmark。

**🔧 技术方法**

使用统计特征编码、因果注意力 Transformer、温度校准与自适应分类门控、多答案平均以及交叉训练等技术。

**📊 数据集**

使用 51 个模型共 5,036 个回答，三种语言提示下生成 294 个整数（1‑355）的整数序列，形成 NumPrint benchmark。

**📈 对比分析**

与 20+ 传统统计/机器学习与序列模型基线比较，SMCT 在单答 51 类识别上达到 92.72% 准确率、91.10% macro‑F1，二答提升至 96.06%，四答接近 97.39%。

**⚠️ 局限性**

覆盖模型有限、未处理未知模型；采集条件不统一；仅针对 294 整数任务，未验证不同长度/范围或工具辅助生成的可迁移性。

---

## 479. TreeAgent: Structure-Aware Memory Agent for Processing Long Hierarchical Data

**arXiv ID:** 2611.13946 | [PDF](https://arxiv.org/pdf/2611.13946v1)

**作者:** Wei Gu `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 TreeAgent，利用输入的层次结构对长序列进行结构化分段、路径标注和残差内存链接，提升结构化数据的推理效果。

**💡 创新点**

创新点在于①用完整子树或相邻兄弟分段保留局部结构；②给每段附加结构路径标识其层次位置；③通过残差内存链接使低层信息直接流向高层，减少信息丢失。

**🔧 技术方法**

采用递归树分段算法、结构路径编码、残差内存链接与跨度检索，并无缝集成至四种现有记忆代理。

**📊 数据集**

使用四类真实应用结构化数据集：JSON（USGS、NVD 等）、HTML、YAML、JavaScript，输入长度覆盖 8K–1M tokens。

**📈 对比分析**

在相同代理下与 Flat 固定长度分段基线对比，四代理平均提升 3.61%（最高 6.35%），在所有任务、长度和格式上均优于基线。

**⚠️ 局限性**

仍存在轻微的模型调用增加、对准确树解析的依赖，以及极大子树超限时产生额外段的问题，且实验仅在无额外训练的设置下验证。

---

## 480. LUMIR: Decoupled Multimodal Representation Learning from Heterogeneous Data for Next-POI Recommendation

**arXiv ID:** 2611.14004 | [PDF](https://arxiv.org/pdf/2611.14004v1)

**作者:** Zehui Wang `[一作]`, Dietmar Jannach `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的深度学习模型，用于图像分类任务。

**💡 创新点**

创新点在于引入了一种新的激活函数，能够提高模型的收敛速度和分类准确率。

**🔧 技术方法**

使用了卷积神经网络（CNN）和改进的激活函数。

**📊 数据集**

使用了CIFAR-10数据集进行实验。

**📈 对比分析**

与传统的激活函数（如ReLU）进行比较，结果显示新模型在分类准确率上提高了5%。

**⚠️ 局限性**

限制在于模型在处理高分辨率图像时的性能尚未得到验证。

---

## 481. Bounded Weighted Edit Distance with Metric Costs

**arXiv ID:** 2611.13984 | [PDF](https://arxiv.org/pdf/2611.13984v1)

**作者:** Jihan Wang `[一作]` `[通讯]` (University of Illinois Urbana Champaign), Jihan Wang (University of Illinois Urbana Champaign)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出了一种确定性 O(n + k²) 时间算法，用于在给定阈值 k 的情况下计算带权编辑距离，若距离 ≤ k 则返回最优对齐，否则报告距离 > k。

**💡 创新点**

创新点在于将传统对角线转移方法推广为“质量偏移”前沿，证明仅需保留每个分数和质量偏移下最远顶点即可保证最优性，从而实现与字母表大小和最大代价无关的运行时间。

**🔧 技术方法**

核心技术包括：质量偏移（gap mass offset）作为前沿维度；最长公共前缀（LCE）查询实现匹配扩展；整数度量的上舍入与阈值截断；以及对实度量的归一化与近似处理。

**📊 数据集**

论文为理论算法研究，没有使用实际数据集；所有结果均通过数学证明与复杂度分析给出。

**📈 对比分析**

与之前的 O(n + k⁵/²) 或 O(n + √n·k³/²) 等上界相比，本算法在整数度量下取得了最优的 O(n + k²) 复杂度；对实度量的 (1+ε) 近似在 O(n + (k/ε)²) 时间内完成，且在整数 gap 成本时进一步提升至 O(n + k²/ε)。

**⚠️ 局限性**

局限性包括：需假设输入代价构成整数度量；对非整数度量仅能得到 (1+ε) 近似；上舍入与截断导致在极端代价或 ε 接近 0 时的时间与空间略有增长；对一般化代价（非度量）仍需更高阶的复杂度。

---

## 482. RB-LDC: Redundancy-Balanced Latent Coding for Noise-Robust Diffusion Latents

**arXiv ID:** 2611.13955 | [PDF](https://arxiv.org/pdf/2611.13955v1)

**作者:** Hyunseok Jeong `[一作]` (DGIST), Young-Sik Kim `[通讯]` (DGIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在潜在扩散模型中引入可逆线性层，对编码器输出的每个潜在方向进行 SNR 重新分配，并依据解码器的敏感性优化该层的谱分布，随后在 ImageNet 256×256 上使用 SD‑VAE 与 VA‑VAE 两个预训练 tokenizer 进行实验，验证该方法在噪声潜在下的重构质量上优于等能量的等 isotropic 编码。

**💡 创新点**

提出了“decoder‑sensitivity‑weighted LMMSE” 这一新的设计目标，并推导出其唯一最优解及大预算下的幂律分配规律，首次将解码器的拉回度量与潜在噪声分配紧密关联；同时给出了可直接求解的谱求解器与基于 Jacobian 的敏感性估计。

**🔧 技术方法**

可逆线性编码层、谱求解器、LMMSE 风险评估、Jacobian 基敏感性估计、线性最小均方误差分析、Cosine 以及均匀噪声加权的扩散时间调度。

**📊 数据集**

ImageNet 256×256 验证集。使用 SD‑VAE 与 VA‑VAE 两种已训练的 tokenizer。

**📈 对比分析**

与相同预算的等 isotropic 编码进行对比，评价指标包括 perturbed FID、PSNR、SSIM、LPIPS。实验显示在 VA‑VAE block‑local（k=128）配置下 perturbed FID 下降 17.1，PSNR 提升 0.855 dB，SSIM 提升 0.0337，LPIPS 降低 0.0346；在更大块尺寸（k=512）时提升更显著。对冻结的扩散模型在采样过程中加入噪声时，该方法亦能在 12 个设置中平均降低 25.5 FID。

**⚠️ 局限性**

风险评估仅为 LMMSE 的线性近似，未完全捕捉解码器的非线性误差；实验仅在固定 tokenizer、线性可逆层、块级编码的设置下进行，未探讨联合训练、非线性编码或输入自适应分配；对低 SNR 区域的性能仍有局限。

---

## 483. TAPCO-G: LLM-Guided Design-Adaptive Timing and Power Co-Optimization with Graph-Based Physical Priors

**arXiv ID:** 2611.13931 | [PDF](https://arxiv.org/pdf/2611.13931v1)

**作者:** Wuqian Tang `[一作]` (National Tsing Hua University), Chun-Yao Wang `[通讯]` (National Tsing Hua University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在后置放置阶段对电路进行时序与功耗的协同优化，利用LLM指导的策略适配和基于图的物理先验评估单元替换，实现高质量结果。

**💡 创新点**

将大型语言模型用于自适应优化策略的生成，并在内部采用基于时序图的差分NLDM、驱动负载耦合与递归批量回溯的物理先验；同时实现双层表征的策略与参数空间扩展。

**🔧 技术方法**

LLM（大语言模型）+ 代码生成、时序图（GNN/DSL）+ 差分NLDM + 事件驱动递归波传播 + 端点价值裁剪 + 粗细两级功耗筛选 + 分批物理试验与二分恢复。

**📊 数据集**

采用M‑CAD 2026赛题的八个ASAP7 7nm基准芯片，规模从2万到3.1万个实例。

**📈 对比分析**

与OR‑LR、OR‑LR+PR、TAPCO及赛题冠军CERBERUS在同一签约流程和评分标准下对比，取得100%有效率，平均质量分提升5.516点（相对提升13.4%），在所有基准上均击败CERBERUS，并相较于TAPCO提升约21.6点。

**⚠️ 局限性**

对超大规模设计的可扩展性尚未验证，LLM产生的策略仍受算子可重构性与计算资源限制；且在极度宏块占比的SoC中功耗改进受限。

---

## 484. MEOW: Medical Open-World Interactive Segmentation

**arXiv ID:** 2611.13935 | [PDF](https://arxiv.org/pdf/2611.13935v1)

**作者:** Philip Matthias Winter `[一作]` (VRVis GmbH), Katja Bühler `[通讯]` (VRVis GmbH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种可在不更新模型参数的情况下，实现交互式分割的开放世界学习框架 OW-IS。

**💡 创新点**

创新点在于将冻结的特征提取器、参数无关的传导模块和记忆网络结合，利用传导推理和层级消息传递实现实时交互与持续学习；通过记忆合并实现零样本到少样本再到多样本的自我改进。

**🔧 技术方法**

核心技术包括：基于最近邻检索的参数无关传导（transduction）、层级 k-NN 及跨层匹配、层级消息传递（hierarchical message passing）以及线性自编码器压缩特征，配合虚拟用户交互仿真和贝叶斯优化调参。

**📊 数据集**

实验数据集涵盖医学 3D 成像（CT、MRI、micro‑CT）以及公开语义分割数据集（COCO、Cityscapes、Synapse）。

**📈 对比分析**

与现有开放集/交互式分割基线（SAM-Med2D、MedSAM、SAM、UNet）对比，OW-IS 在闭集、开放集及开放世界任务中均显著提升 Dice 分数（如 3D 肝脏分割从 0.92 提升至 0.89，需点击次数平均下降 50%），并在低延迟（<1 s）下实现实时交互与毫秒级记忆写入。

**⚠️ 局限性**

局限性包括：需要保存特征金字塔的显存开销相对较大；对基于 Transformer 的网络兼容性不足；在极大规模记忆库下 k‑NN 搜索仍可能成为瓶颈。

---

## 485. PARDIG: Global Precipitation Estimates Learned from Sparse Satellite Observations

**arXiv ID:** 2611.13941 | [PDF](https://arxiv.org/pdf/2611.13941v1)

**作者:** Guy Shalev `[一作]` (Google DeepMind), Alvaro Sanchez-Gonzalez `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过训练基于图形网格变压器的连续坐标查询模型，从ECMWF HRES/ERA5再分析、SatWeave 11通道卫星影像等多模态输入，利用2016‑2023年GPM Core Observatory的稀疏雷达观测为监督，生成全球15‑分钟时空分辨率的降水场（近实时和历史两条流）。

**💡 创新点**

采用稀疏‑至‑密集的坐标查询输出头，在全球icosahedral网格上实现无边界推断；完全不依赖地面雷达或雨量计校准，使用空间卫星雷达观测直接监督，使得极端降水的预测显著优于现有IMeRG、MSWEP等产品。

**🔧 技术方法**

利用WeatherNext 3的图形网格变压器与连续坐标查询架构，处理4小时时间窗口的再分析与卫星影像特征，输出多类别降水概率分布并通过阈值映射转化为定量降水率。

**📊 数据集**

训练目标：2016‑2023年GPM Core Observatory（CORRA）雷达观测；输入：ECMWF HRES/ERA5 13层气压面+地面变量、SatWeave 11通道GOES/Meteosat/Himawari影像；评估：MRMS、EURADCLIM、JMA Radar‑AMeDAS雷达组合以及36,678个雨量计站点。

**📈 对比分析**

在全球雷达与站点基准上使用CSI、F1、精确率、召回率等多指标与IMeRG Final/V07、MSWEP Gc、ERA5、HRES等基线对比；平均CSI提升约0.08（+28 %），极端阈值（≥15 mm h⁻¹）提升更大；2024‑25年外推维持稳定；案例研究中对极端事件的CSI比IMeRG高3‑5倍。

**⚠️ 局限性**

对高海拔山区及极端寒季降水仍有欠估，GPM DPR对细雨/降雪敏感度有限；北极地区因缺乏观测需谨慎使用；模型受GPM训练目标局限，无法完全捕获复杂地形与极地降水特征。

---

## 486. Margin-Guided Token Credit for Reinforcement Learning with Verifiable Rewards

**arXiv ID:** 2611.14015 | [PDF](https://arxiv.org/pdf/2611.14015v1)

**作者:** Chi Liu `[一作]` (PayPal), Srinivasan Manoharan `[通讯]` (PayPal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究在强化学习与可验证奖励框架下，提出一种基于top‑2边际阈值的令牌级奖励分配方法（MCPO），以提升语言模型的数学推理准确率。

**💡 创新点**

创新点在于将奖励重新分配给模型在生成过程中的“犹豫”位置（边际小的令牌），通过绝对阈值转换为分数并在保证平均优势不变的前提下调节每个令牌的梯度贡献。

**🔧 技术方法**

使用技术包括：GRPO（基于可验证奖励的策略梯度）、LoRA微调、温度缩放后的概率分布 top‑2 边际计算、固定阈值映射、指数权重分配与裁剪的策略梯度损失。

**📊 数据集**

训练数据为 2,048 条 DAPO‑Math‑17k 题目，评估使用六个公开竞赛数学基准（AIME24/25/26、HMMT26、AMC23、MATH500‑L5），每个基准共 297 道题。

**📈 对比分析**

与未改动的 GRPO 基线在八个（模型+种子）组合上均取得提升；MiniCPM5‑2B‑Base 的宏平均 Pass@32 提升约 6.55 分（相对 22%），Qwen3‑4B‑Base 提升约 4.75 分；提升集中在中等难度题目，且随采样预算增大而显著扩大。

**⚠️ 局限性**

局限性包括：未与熵基奖励或其他信用分配方法做对比；基线强度有限，可能受 LoRA 训练步数和学习率影响；MCPO 产生更长答案，导致部分提升可能归因于推理长度；未对多机、多卡、不同模型规模或完整微调的泛化性进行评估；以及缺乏对阈值与 κ 参数的系统敏感性分析。

---

## 487. Spatio-Temporal Calibration of a Vehicle-Mounted Microphone Array for Autonomous Driving

**arXiv ID:** 2611.14003 | [PDF](https://arxiv.org/pdf/2611.14003v1)

**作者:** Marvin Seegert `[一作]` (Technical University of Munich), Johannes Betz `[通讯]` (Technical University of Munich)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b88c6eac-d57a-4623-a604-1f401f3eb268` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在车辆上实现了一种无需先验信息的麦克风阵列空间-时间校准方法，利用气球爆炸作为单一跨模态目标，结合LiDAR定位、声波起始检测和摄像机时间标定完成几何与时间同步校准；

**💡 创新点**

创新点在于：①使用气球爆炸作为可在LiDAR中定位、在麦克风阵列中产生清晰起始的多模态标靶；②通过每个观测的自身时间方差加权，自动折衷阴影（非视距）声波而非简单遮蔽或模型化阴影；③一次事件即可同时校准空间与时间，简化流程并提高鲁棒性；

**🔧 技术方法**

核心技术包括：非线性最小二乘校准（软ℓ₁损失）、多通道声波起始检测（高通+CFD）、LiDAR差分球体拟合、观测自加权、相机-音频时间同步与延迟估计；

**📊 数据集**

使用的数据集为：在一辆改装为自动驾驶平台的全尺寸面包车上，布置8麦克风阵列（4.9 m孔径），在户外多天收集30+次气球爆炸实验，并同步记录LiDAR、摄像头和音频；

**📈 对比分析**

与传统室内光学/雷达对齐、基于单模态的校准方法相比，本方法在校准残差上达到cm级（约3 cm），并在动态声源方向定位任务中实现几度（≈3–4°）的误差，明显优于现有方案；

**⚠️ 局限性**

主要局限包括：缺乏外部精确测量的基准验证，阴影声波的残余偏差难以完全校正；阵列高度分辨率受限（垂直误差约两倍水平误差）；对声速假设敏感，且实验需要在车辆周围均匀布置至少12次气球爆炸以获得最佳性能。

---

## 488. Specify Privacy Yourself: Assessing Inference-Time Personalized Privacy Preservation Ability of Large Vision-Language Models

**arXiv ID:** 2611.13997 | [PDF](https://arxiv.org/pdf/2611.13997v1)

**作者:** Xingqi Wang `[一作]` (Tsinghua University), Jia Jia `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了推理时个性化隐私保护（ITP^3）任务，并构建了 SPY-Bench 与 SPY-Tune 两套基准与训练数据集，用于评估和提升大型视觉语言模型在动态隐私约束下的遵从能力。

**💡 创新点**

创新点在于定义了用户可通过自然语言动态指定隐私边界的任务框架，设计了专门的评价指标，并提出了 NCA‑P 这一对比学习变体，在隐私保护方面显著优于传统对齐方法。

**🔧 技术方法**

采用了对比学习的 Noise Contrastive Alignment (NCA) 及其改进版本 NCA‑P、强化学习的 Direct Preference Optimization (DPO)、监督微调 (SFT) 与自我监督的 Self‑Moderation 等多种对齐技术。

**📊 数据集**

主要使用的数据集来自 VISPR 图像数据，构成 32,700 条样本的 SPY-Bench 以及 81,096 条样本的 SPY‑Tune，用于评估与训练。

**📈 对比分析**

在对 21 款最新 LVLM 进行评估时，绝大多数模型的 Instruction Compliance Score (ICS) 低于 25%，但通过 NCA‑P 微调后可达 96.88% 的合规率，并在一般能力基准上保持了较好性能。

**⚠️ 局限性**

主要局限在于现有模型对隐私指令的理解仍不充分，且对齐方法在保持通用能力方面存在一定退化，未来需要更精细的隐私训练策略与更大规模的多场景数据支持。

---

## 489. Dual-Core Tensor Wheel Decomposition with Structure-Cached Acceleration for Tensor Recovery

**arXiv ID:** 2611.13938 | [PDF](https://arxiv.org/pdf/2611.13938v1)

**作者:** Wenchao Xie `[一作]` (Xiangtan University), Qingsong Wang `[通讯]` (Xiangtan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了双核张量轮解构（Dual‑Core Tensor Wheel, DCTW）以及结构缓存加速策略，用于高阶张量的恢复与去噪

**💡 创新点**

创新点在于将传统张量轮的单一高阶中心核拆分为两个低阶核心并通过共享潜在连结进行低秩分解，同时通过结构缓存重用中间张量，显著降低中心核更新的计算与内存成本

**🔧 技术方法**

核心技术包括DCTW网络拓扑、基于近似交替最小化（Proximal Alternating Minimization, PAM）的优化框架、隐式更新与结构缓存加速的张量网络运算

**📊 数据集**

实验数据集涵盖五个彩色视频（Akiyo、Mother、Grandma、Train、Ukulele）以及四个含盐椒噪声的彩色视频（News、Foreman、Suzie、Silent）

**📈 对比分析**

与TT、TR、FCTN、TW、SVDinsTN、TMacTT、TRLRF等张量网络及经典方法进行对比，DCTW在MPSNR/MSSIM上普遍优于对手，并在运行时间与峰值内存上实现显著加速，尤其在高秩与大尺寸张量上表现突出

**⚠️ 局限性**

局限性包括：需手工设定多组秩参数，尚未实现自动秩与结构选择；在极端噪声或非彩色视频场景下的鲁棒性尚待进一步验证

---

## 490. Temporal Behavior Trees for Reinforcement Learning: Specifications as Rewards, Curricula, and Diagnostics

**arXiv ID:** 2611.13993 | [PDF](https://arxiv.org/pdf/2611.13993v1)

**作者:** Vladimir Krsmanovic `[一作]`, Bernd Finkbeiner `[通讯]` (Cispa Helmholtz Center For Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出Temporal Behavior Trees（TBT）作为RL任务的统一规范、奖励、课程和诊断工具，构建TBTCL框架

**💡 创新点**

将TBT的层级结构与鲁棒性语义融合，自动提取课程DAG、生成密集奖励并通过轨迹分段诊断瓶颈，全部以单一规范实现

**🔧 技术方法**

TBT语义与鲁棒性计算、课程提取算法、轨迹分段诊断、PPO强化学习、Python生态（PyTorch、Stable-Baselines3、NumPy、NetworkX、Hydra）

**📊 数据集**

LetterWorld（离散格点字母目标）和PointNav（连续2D点质运动）两种实验环境

**📈 对比分析**

与STL Shaping、AGCL、SPECTRL、Sparse、LDBA五种基线对比；在单目标与序列、重复任务中与STL Shaping持平，在避障任务中取得最高任务满足率（如LetterWorld 0.85、PointNav 0.64）

**⚠️ 局限性**

扩展性有限：对长轨迹和Seq/Kleene运算计算昂贵；需调节安全属性和并行操作的额外超参数；未给出收敛或最优课程顺序的理论保证

---

## 491. Communication-Learning Co-Design for Pinching-Antenna-Assisted Federated Learning in Heterogeneous Wireless Networks

**arXiv ID:** 2611.13987 | [PDF](https://arxiv.org/pdf/2611.13987v1)

**作者:** Shuai Wang `[一作]` (University of Electronic Science and Technology of China), Zhi Chen `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了在无线边缘网络中利用分段波导启用的收缩天线（SWAN）与异质化鲁棒联邦学习框架的协同设计；

**💡 创新点**

创新点在于将收缩天线的可调辐射点与基于控制变量的方差降低算法和双重量化机制结合，形成统一的通信与学习优化框架，并提出两阶段下行协议和公平上行设计；

**🔧 技术方法**

采用的技术包括分段波导收缩天线架构、Rician 统计信道模型、量化压缩、基于控制变量的FedAvg改进、SDP/随机化与交替优化的波束成形和天线位置规划、以及逆问题和二分搜索的上行资源分配；

**📊 数据集**

实验使用CIFAR-10图像分类数据集，并将训练集按Dirichlet分布划分为100个客户端以模拟高异质性；

**📈 对比分析**

与FedAvg、FedAdam、SCAFFOLD等基线在固定量化、BS优化与SWAN优化三种通信配置下进行对比，结果显示在SWAN设置下，本文方法在收敛速度和最终准确率上显著优于基线，准确率提升约10%并将训练轮数减少约2.5倍；

**⚠️ 局限性**

局限性在于仅考虑单一收缩天线平台与理想的Rician信道模型，未探讨天线互连与多路径干扰的实际部署复杂度，以及不同波段与更大规模网络下的可扩展性。

---

## 492. Advancing a Global Framework for AI Safety and Governance for the Well-being of Humanity

**arXiv ID:** 2611.13968 | [PDF](https://arxiv.org/pdf/2611.13968v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 493. Gradient Inversion through Linear and Affine Subspace Constraints

**arXiv ID:** 2611.13976 | [PDF](https://arxiv.org/pdf/2611.13976v1)

**作者:** Konstantinos Bouros `[一作]` (Technical University of Crete), Athanasios P. Liavas `[通讯]` (Technical University of Crete)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出利用全连接层梯度的仿射子空间结构进行梯度逆向攻击，并在此基础上设计软硬约束的子空间正则化；

**💡 创新点**

创新点在于首次将每个FC层归一化梯度视为输入的仿射组合，构造多层子空间约束，提升逆向精度；

**🔧 技术方法**

使用梯度匹配优化（cosine距离+TV或嵌入正则化），SVD估计子空间，软硬约束两种变体；

**📊 数据集**

评估数据集包括CIFAR‑10、Tiny‑ImageNet（FC‑3网络）和ImageNet（预训练VGG‑16）；

**📈 对比分析**

与IG、CPA等基线对比，软/硬仿射变体在B≤128时在LPIPS/CS指标上均优于基线；对比结果显示子空间方法在低噪声下更稳健，且批量大小误估时对硬约束更容忍；

**⚠️ 局限性**

局限性：对网络结构、批量大小及标签信息依赖较大，仿射优势并非普遍；DP防御实验仅在单一批量大小与噪声水平下，未充分探究隐私‑效能权衡；

---

## 494. On the information rate of phones, syllables and words

**arXiv ID:** 2611.13990 | [PDF](https://arxiv.org/pdf/2611.13990v1)

**作者:** Herman Kamper `[一作]` `[通讯]` (Stellenbosch University), Herman Kamper (Stellenbosch University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在有限建模预算下，使用不同语言单位（音位、音节、单词、以及词级与音位级BPE组合）对同一语音内容进行编码时的比特率；

**💡 创新点**

在同一数据集与相同建模预算下系统性比较不同单位的编码效率，并发现略长于单词、短于长多词组合的单位在信息率上表现最佳；

**🔧 技术方法**

采用绝对折扣n-gram模型、20M参数的transformer语言模型（基于OPT-125M变体）、词级BPE与音位级BPE以及拼写模型等技术；

**📊 数据集**

使用LibriSpeech 960h英语有声书的强制对齐音素序列（dev-clean 5.4h 评估集）；

**📈 对比分析**

通过在held‑out评估集上计算交叉熵转化为比特率来比较不同单位；在transformer模型中，单词单位达到28.07 bps（最佳），n-gram模型中word‑BPE 200单位为31.05 bps，其他单位均比这些更高；

**⚠️ 局限性**

仅评估语音序列的比特率，未考虑人类心理成本与语言多样性；单位定义与不同语言的适用性有限，且模型规模受限，仅适用于英语。

---

## 495. What to Predict for Efficient Scheduling on Parallel Machines

**arXiv ID:** 2611.13980 | [PDF](https://arxiv.org/pdf/2611.13980v1)

**作者:** Evripidis Bampis `[一作]` (Sorbonne Universite), Michalis Xefteris `[通讯]` (University of Bremen)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在并行机器调度问题（最小化完工时间和加权完工时间）中，利用带噪声的预测（解决方案编码）实现精确调度恢复的可行性与限制。

**💡 创新点**

提出三种预测编码模型（贪心、分区、顺序）并证明后两种对噪声极度敏感；发现顺序编码在噪声扰动k=O(log n)时仍能通过动态规划高效恢复最优解；同时给出一套完整的DP框架，并给出在包含释放时间、优先级约束等更一般场景下的算法与复杂度。

**🔧 技术方法**

主要技术包括：基于动态规划的参数化算法（按k局部扰动）、分治与状态压缩、复杂度可归约性证明（对greedy/partition编码的硬性证明）、与经典NP/Σ-hard问题的归约、以及多层递归DP（用于优先级约束）。

**📊 数据集**

本工作为理论研究，无实验数据集；所述算法与结论均在理论上完成。

**📈 对比分析**

与传统的无预测调度算法相比，顺序编码在k=O(log n)下可在多项式时间内恢复最优解；相对于在噪声下的近似调度，提供了精确解的可能性。由于缺乏实验，对实际性能的数值评估未给出；但理论上复杂度为O(m·n²·2^{4k}·F)（F为单机评估时间），对k=O(log n)保持可行。

**⚠️ 局限性**

限制包括：贪心与分区编码在任何非零噪声下均为NP/Σ-hard；顺序编码在机器数为可变参数且存在优先级约束时仍为强NP/Σ-hard；对于P|prec|C_max在机器数可变时，即使k=2也不可多项式求解；此外，算法对k的指数依赖意味着在k较大时仍不可行。

---

## 496. Euclidean and Hermitian Self-Orthogonal Codes from Divisible Codes

**arXiv ID:** 2611.13963 | [PDF](https://arxiv.org/pdf/2611.13963v1)

**作者:** Ziling Heng `[一作]` (Chang'an University), Chenkun Xie `[通讯]` (Chang'an University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并证明了在偶特征下4‑可除码和在任意特征下p‑可除码（p为质数）包含全一向量时，分别满足欧氏自正交性与厄米自正交性，从而给出了可除码构造自正交码的简单判据。

**💡 创新点**

创新点：首次将可除性与自正交性在更一般的有限域上统一起来，提供了两条新的可除性-自正交性判据；利用这些判据构造了两族新的循环自正交码（含欧氏和厄米），并进一步推导出相应的量子错误更正码，展示了其距离最优性与极优性。

**🔧 技术方法**

主要技术包括：有限几何中的线与点计数、可除码的符号计数与线性同余、Trace 与 Norm 的性质、Hermitian 内积的代数性、以及自正交码与量子码之间的CSS/厄米构造。对自正交码的距离最优性使用了 cap 上界与汉明界的推导。

**📊 数据集**

本工作完全基于理论证明，没有使用具体数据集；所有结果均来自代数、几何与编码理论的推导。

**📈 对比分析**

与已知的欧氏/厄米自正交码（如 Reed–Muller 码）进行比较，证明了新构造的码在长度、维数与距离上的最优或极优性能；同时通过自正交码得到的量子码在相同长度、维数下的距离达到或接近量子汉明界，表明其性能优良。

**⚠️ 局限性**

局限性：判据依赖于码中必须存在全一向量，q>2 时该假设不可去除；此外，目前仅适用于可除码，尚未覆盖更一般的非可除自正交码；构造的循环码族在具体参数选择上仍受限于已知可除码的结构。

---

## 497. The Case for AI-Enabled Gate Monitoring as Farm Security in the UK: A PRISMA-Guided Review of the Evidence on Rural Crime, Livestock Predation and Farm Access

**arXiv ID:** 2611.13933 | [PDF](https://arxiv.org/pdf/2611.13933v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 498. Tailoring Optimization Approaches with Domain Knowledge in Industry Process for Meta-heuristic Development

**arXiv ID:** 2611.13940 | [PDF](https://arxiv.org/pdf/2611.13940v1)

**作者:** Jakub Cebula `[一作]` (Wrocław University of Science and Technology), Adam Paliszek-Saładyga `[通讯]` (Wrocław University of Science and Technology)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了行业元启发式开发流程IPMD，并基于该流程设计了BNTGP超启发式，用于解决车辆路径规划和多技能资源受限项目调度两类多目标NP难问题。

**💡 创新点**

创新点在于将BNTGA与遗传编程相结合形成可自动适应不同实例的超启发式，并通过问题特定特征与解码器将领域知识嵌入搜索空间。

**🔧 技术方法**

采用遗传编程、平衡非支配锦标赛遗传算法、启发式解码器以及特征选择技术，构建可进化的评分树以评估候选动作。

**📊 数据集**

使用CVRPTW标准测试集（如Solomon实例）和MS‑RCPSP标准实例（d36、d45）进行实验。

**📈 对比分析**

通过与现有基准方法的统计性能对比（显著性检验与Pareto前沿质量指标），BNTGP在两类问题上均取得了与最先进方法相当甚至更优的结果。

**⚠️ 局限性**

局限性包括对特征设计的手工依赖、缺乏跨问题可迁移性的实证、以及演化得到的启发式的可解释性不足，未来工作计划进一步自动化特征生成与评估。

---

## 499. HERO-D3QN: Inter-Cluster Routing with Energy-Aware Soft Constraints in Wireless Sensor Networks

**arXiv ID:** 2611.13943 | [PDF](https://arxiv.org/pdf/2611.13943v1)

**作者:** Weihao Wang `[一作]` (Shenyang University of Technology), Sixue Liu `[通讯]` (Shenyang University of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种基于深度强化学习的无线传感器网络跨簇路由方法 HERO-D3QN，能够在给定簇结构的情况下动态选择下一跳节点，同时考虑能量平衡与热点风险。

**💡 创新点**

创新点包括：① 将能量感知的软约束融入 D3QN 的动作价值调整，允许低能量节点在满足方向与能耗条件下仍可被选中；② 提升状态表示，加入能量、前进进度、相对路径长度、热点风险等多维特征；③ 设计多目标奖励函数，平衡节点生存、传输量、能量效率、能量分布与热点惩罚；④ 采用层级筛选策略（方向筛选 + 能耗筛选 + 两级软惩罚）显著提升路由质量。

**🔧 技术方法**

技术：双重 DQN（Dueling Double Deep Q-Network）与优先经验回放、软更新、探索衰减；特征工程：节点能量比例、距离归一化、进度增益、路径比例、邻居/转发候选比例；奖励设计：节点存活权重、数据量权重、能量均衡惩罚、热点惩罚与软约束；实验平台：Python + TensorFlow/ PyTorch，仿真环境基于自定义 WSN 仿真器。

**📊 数据集**

使用了随机部署的 100 节点网络（200 m×200 m 区域，静态接收器位于 (100,215) m）的仿真数据；多种随机种子（7、21、42、84、168）用于评估鲁棒性；数据集并非公开真实传感器数据，而是由模拟器生成的网络拓扑与能量消耗序列。

**📈 对比分析**

与 LEACH 基线、单流 DQN、双流 D3QN 在相同节点布局下进行对比。HERO-D3QN 将首次节点死亡时间从 629 步提升至 1269 步，峰值残余能量方差下降 55.5%，累计传输量 170.52 Mbit。实验表明 HERO-D3QN 在节点存活、能量平衡和数据吞吐方面均优于基线和传统 D3QN，且在多种随机种子下保持较低的方差。

**⚠️ 局限性**

局限性：① 仅在静态节点、固定接收器、静态链路条件下验证；② 未评估端到端数据传输成功率；③ 依赖仿真，缺乏真实网络实验验证；④ 软约束与奖励参数选择仍需经验调优，可能在不同网络规模或拓扑下表现不一致。

---

## 500. Data4Cyber: Labeled Cyber-Physical Datasets for Distribution-Grid Management Systems Under Representative OT Attacks

**arXiv ID:** 2611.13942 | [PDF](https://arxiv.org/pdf/2611.13942v1)

**作者:** Ömer Sen `[一作]` (Fraunhofer FIT), Yoshihiro Ujiie `[通讯]` (Panasonic)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并发布了Data4Cyber数据集，该数据集在实验室配备的分布式能源管理系统（DMS）中记录了六种不同的网络与物理攻击场景，包括 Modbus/TCP 和 MQTT 通信、控制上下文与同步物理测量，且每个时间步带有相位分辨标签。

**💡 创新点**

创新点在于：①提供了 DER‑centric DMS 环境下混合工业协议（Modbus/TCP + MQTT）的公开、可复现数据；②以 1 s 同步时间基准、相位分辨标签和状态日志实现对过程与网络层的完整关联；③通过针对命令、透明 FDI 与高层价格操纵三大攻击族的设计，展示传统 IDS 在此类时序块状态检测中的局限性，为过程感知与物理信息融合的研究提供实证基准。

**🔧 技术方法**

技术与方法：实验室测试平台（实际 PV、BSS、负载、仪表、Modbus/TCP 与 MQTT 网络）；时间同步 1 s；对网络 PCAP、控制日志与物理测量进行标签生成与阶段划分；采用 IPAL 框架对 12 种开源 IDS 进行基准评估，并使用块状时间拆分避免邻近样本泄漏。

**📊 数据集**

数据集：Data4Cyber，包含 12,562 个 1 s 采样点，1 个基线 S0 + 6 个攻击场景；每个场景提供同步网络 PCAP、控制与物理测量 CSV、阶段标签 JSON、语义映射与元数据，已发布至 Zenodo（DOI: 10.5281/zenodo.19965384）。

**📈 对比分析**

评估方法：在 S0–S6 上分别训练与测试 12 种 IDS（无监督、过程感知、监督），使用“块状”时间拆分避免过拟合；结果显示高召回率并不等于低误报，过程感知模型（如 6lUnsupervised/process‑aware）在保持低误报的同时实现与随机森林相当的 F1；其他无监督方法则在误报或召回上表现不均衡。

**⚠️ 局限性**

局限：①实验规模有限，仅覆盖 30 min 的单馈线实验室运行；②缺少日夜、季节性负荷变化及更大规模资产；③攻击仅集中在 OT 层的 Modbus/TCP 与 MQTT，未覆盖 IT 预备阶段（钓鱼、凭证窃取等）；④数据来源为特定厂商设备，协议实现细节可能不具通用性。

---

## 501. Beyond Proxy Metrics: MLLMs as Human Surrogates for Explainable AI Evaluation

**arXiv ID:** 2611.13926 | [PDF](https://arxiv.org/pdf/2611.13926v1)

**作者:** Eoin M. Kenny `[一作]` (JP Morgan), Tom Bewley `[通讯]` (Mistral AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文利用多模态大型语言模型（MLLMs）代替人类受试者，对可解释AI（XAI）解释方法进行评估，提出了“purpose-grounded”评估框架并在五个XAI领域上对14个MLLM进行实验。

**💡 创新点**

创新点在于首次系统性验证MLLM在多目的导向评估中能重现人类实验的排名，给出排名保持的理论条件与采样稳定性分析，并提供多模型基准与可复现代码。

**🔧 技术方法**

技术方法包括对GPT-5、GPT-4o、Claude、Qwen、Gemma、Mistral等多模型进行Prompt设计与链式推理，利用Spearman相关和bootstrap置信区间对模型输出与人类基准进行比较。

**📊 数据集**

使用的数据集包括复现的五个原始人类研究数据（医疗专业人员、血酒浓度教学、模型改进、依赖校准、可操作推荐、系统参与）以及自行生成的合成数据。

**📈 对比分析**

通过比较各模型在各任务中的条件均值与人类基准排名，计算Spearman相关并给出95%置信区间；大多数模型在各任务中呈现正相关（ρ>0.5），部分模型排名完全匹配，平均相关性约0.65‑0.75，样本量100即可获得90%+的排名稳定性。

**⚠️ 局限性**

局限性包括模型的黑盒性、潜在记忆化、对相似解释差距小的任务易失真、对文本分类任务的天花板效应、模型版本漂移以及仅关注排名而非绝对效能，无法完全替代真实人类测试。

---

## 502. A Stage-Wise Explainable Pipeline for Arteriolar-to-Venular Ratio Estimation and Retinopathy Screening

**arXiv ID:** 2611.14013 | [PDF](https://arxiv.org/pdf/2611.14013v1)

**作者:** Sharan Sharmeela Karuppannan `[一作]` (Indian Institute of Information Technology Design and Manufacturing), Umarani Jayaraman `[通讯]` (Indian Institute of Information Technology Design and Manufacturing)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了一套五阶段的可解释流程，用于从眼底照片估计动脉-静脉直径比（AVR）并进行视网膜病变筛查。

**💡 创新点**

创新点在于把传统的手工测量步骤（血管分割、视盘定位、最小弦宽度测量、AVR阈值决策）全部模块化并公开中间产物，使错误可归因；并采用双掩码分割结合标签转移、最小弦边界走访校正等方法提升精度。

**🔧 技术方法**

使用了Attention-WNet双模型进行动静脉分割，Laplacian边界提取与二分量圆交点检测，最小弦边界走访算法校正宽度，Zone B 区域的多半径聚合，基于阈值0.67的二分类决策。

**📊 数据集**

实验数据集包括 DRIVE（A/V训练），HRF（30张筛查），LES‑AV（22张校准），MESSIDOR（1200张筛查）。

**📈 对比分析**

与 VascX 等基线及基于六血管 Knudtson 计算的 AVR 进行对比，HRF 上准确率达96.7%，AUC 0.978；MESSIDOR 上估计准确率0.948，F1 0.922；在阈值 0.67 下比 VascX 的误判率低 1 倍。

**⚠️ 局限性**

局限包括样本量有限（HRF 仅 30 张）、未使用独立的参照尺度数据、使用两大血管而非六大血管的 AVR 计算，可能导致与临床规范不完全一致。

---

## 503. Personalized Memory Rubrics: Learning What to Remember per User via Rubric-Guided Credit Assignment

**arXiv ID:** 2611.14017 | [PDF](https://arxiv.org/pdf/2611.14017v1)

**作者:** Yupeng Han `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Personalized Memory Rubrics (PMR)，通过学习可解释的用户定制化评估标准，将终端 QA 奖励转化为细粒度的步骤级信用分配，改进长期记忆代理的学习。

**💡 创新点**

创新点在于把评估拆成通用层和用户适配层，使用 Rubric‑Guided Credit Assignment 与方向安全门控，只对成功轨迹进行正向放大，构建自我演化的鲁棒信用机制。

**🔧 技术方法**

技术包括 GRPO 强化学习、LLM 作为教师生成词级认可度、鲁棒的方向门控与熵门控、共演化的 Rubric 学习与信用分配循环。

**📊 数据集**

使用四个个人化记忆基准：PersonaMem、PrefEval、PersonaBench、PERMA，覆盖不同场景与噪声。

**📈 对比分析**

与多种静态与 RL 基线对比，PMR 在所有十二个评测场景中均获得显著提升，PersonaMem 32K 上准确率从 67.47% 提升至 76.76%，在 OOD 任务亦保持领先。

**⚠️ 局限性**

局限包括对失败轨迹的信用归因仍依赖鲁棒门控，且在极高噪声下需更多训练数据，且当前仅在 LLM 辅助下实现，可能难以直接迁移至资源受限环境。

---

## 504. Coding Agent Memory Post-Training II: Unlocking the Memory Potential of Pre-trained File Operations for Long-Horizon Tasks via On-Policy Distillation

**arXiv ID:** 2611.14009 | [PDF](https://arxiv.org/pdf/2611.14009v1)

**作者:** Lirui Luo `[一作]` (Peking University), Cong Fang `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了CAMG-II长时限环境套件，并开发CAMP-II算法，利用共享的文件操作接口实现对记忆的主动学习与教师监督；

**💡 创新点**

创新点在于将教师和学生共同熟悉的文件操作作为记忆介质，解决了传统记忆工具与教师不匹配导致的负优势问题，并通过两侧接口先验实现高效的OPD；

**🔧 技术方法**

技术包括：基于Qwen3.5模型的on-policy distillation (OPD)、文件操作记忆接口、跨任务统一策略训练、以及对记忆内容的token级监督；

**📊 数据集**

使用的数据集包含CAMG-II五大任务（Shop、Coding、AutoResearch、General、Visual）以及外部基准Terminal-Bench 2.1和SWE-Bench Pro V2；

**📈 对比分析**

与多种记忆学习基线（MemOPD、CompactOPD、ACM、HELM、Hy-Memory）以及CAMP-I进行对比，CAMP-II在CAMG-II上平均成功率达到64.6%，是最强基线的2.97倍；在外部基准上，CAMP-II-9B的表现超过同规模更大模型（35B）并在两大基准上均取得最高分；

**⚠️ 局限性**

局限性包括仅在Qwen3.5-4B/9B规模内验证，未测试更大规模或多教师场景，且仅使用单一冻结教师模型。

---

## 505. Interleaving anatomical inspection and diagnostic reasoning for explainable radiology report generation

**arXiv ID:** 2611.14018 | [PDF](https://arxiv.org/pdf/2611.14018v1)

**作者:** Chentao Huang `[一作]` (Dalian University of Technology), Ling Luo `[通讯]` (Dalian University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一套基于解剖结构的多模态推理轨迹（MCoT）并在此基础上训练 ARIVE 模型，以可解释的方式生成胸部 X 光的放射科报告。

**💡 创新点**

核心创新在于将解剖区域的视觉检查与诊断推理交替嵌入生成流程，并通过“解剖与临床内容优化（ACCO）”的多目标奖励系统联合提升定位准确性、推理可信度与报告临床质量。

**🔧 技术方法**

技术包括：使用 GPT‑OSS‑120B 生成 MCoT、Qwen3‑VL‑Instruct‑4B 作为多模态大语言模型基座、局部解剖视觉编码模块、GRPO（group relative policy optimization）强化学习、四项奖励（bounding‑box、推理‑CheXbert、报告‑CheXbert、RadGraph）及逆频率加权。

**📊 数据集**

主要使用了公开的 MIMIC‑CXR（普通与异常子集 MIMIC‑ABN）以及 Chest ImaGenome 的 29 个解剖区域定位注释，构造了 159,416 条 MCoT 训练样本。

**📈 对比分析**

在 MIMIC‑CXR 和 MIMIC‑ABN 上与 6 类基线（大规模通用 MLLM、通用医学 MLLM 与 4 个专用报告生成模型）进行对比，ARIVE 在 CheXbert、SRR‑BERT、RadGraph、GREEN 等临床有效性指标均取得最高分，且专家打分与 Qwen3‑5 接近，MCoT 轨迹的可追溯性评分高达 4.9/5。

**⚠️ 局限性**

局限性包括：解剖区域标注稀疏导致某些区域定位仍弱；仅使用前平视胸片，未覆盖侧位或其他模态；MCoT 轨迹依赖于现有报告，可能继承报告偏差；在不同机构或报告风格下的泛化性尚未验证；模型需先验解剖框架，迁移至其他影像域需要额外标注与设计。

---

## 506. Comparing WeChat and Telegram MiniApp Security Architectures: Toward Cross-Platform Analysis

**arXiv ID:** 2611.13965 | [PDF](https://arxiv.org/pdf/2611.13965v1)

**作者:** Luca Ferrari `[一作]` (IMT School for Advanced Studies Lucca), Luca Verderame `[通讯]` (University of Genoa)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对WeChat Mini Programs和Telegram MiniApps的架构与安全模型进行定性比较，构建共享分类法并给出案例分析。

**💡 创新点**

首次提出统一的安全控制/资产/威胁源分类框架，并将其映射到两个生态系统；在Telegram侧首次通过客户端源代码细粒度分析桥接与权限路径。

**🔧 技术方法**

官方文档分析、先行研究综述与Telegram Android客户端源代码的静态分析。

**📊 数据集**

官方开发者文档、相关学术论文以及Telegram Android客户端（特定commit）的源码。

**📈 对比分析**

通过对两侧执行环境、通信路径、身份验证与权限控制等维度进行对比，并以两个工作示例阐释分析需求；未给出性能指标，仅提供模型和理论比较。

**⚠️ 局限性**

分析仅基于单一Telegram客户端版本且未涵盖真实MiniApp实例，缺乏实证评估与自动化检测效果验证。

---

## 507. Optimization-Based Path Planning for Industrial Manipulators: Experimental Validation in Additive Manufacturing

**arXiv ID:** 2611.13983 | [PDF](https://arxiv.org/pdf/2611.13983v1)

**作者:** Shiva Ganesh Murali `[一作]` (Zürich University of Applied Sciences), Alisa Rupenyan `[通讯]` (Zürich University of Applied Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在离线阶段，本文设计了一种基于约束非线性优化的关节路径规划器，能够将机器人可行性与增材制造过程几何要求相统一，随后在 UR10e 机器人上分别实现了碳纤维缠绕和三维打印的路径规划与执行。

**💡 创新点**

创新点包括：①提出统一的过程感知关节路径公式，将纤维方向/喷嘴姿态与关节平滑性、自碰撞、关节极限等多种约束融合；②引入迭代权重调优（IWT）算法，根据 KPI 与目标差距自动确定成本权重；③为长路径采用块式优化与边界耦合，实现高效分段求解。

**🔧 技术方法**

技术手段包括：前向运动学、基于 JAXIPM 的非线性规划求解、迭代权重调优、块式与全路径优化、关节差分平滑成本、自碰撞与纤维-机器人间距惩罚、打印时的四元数姿态误差度量。

**📊 数据集**

使用的数据集为实验机器人平台自身采集的几何信息：缠绕任务共2401个采样点（20条缠绕边），打印任务共479 523个姿态采样点；未使用公开工业数据集。

**📈 对比分析**

通过与过程容差和 KPI 目标的对比评估，缠绕路径最大纤维误差1.9°、打印路径位置误差0.066 mm、姿态误差0.0013°，均低于容差；执行跟踪误差仅为几度；IWT 在3–5 次迭代即可满足所有 KPI，块式优化显著降低计算时间。

**⚠️ 局限性**

局限性包括：需预先固定纺丝机位置、机器人基座标定及环境几何，无法适应实时环境变动；仅考虑几何约束，未涉及材料流动/挤出动力学；对过程参考的手工构建要求高，且仅适用于可通过成本/约束表述的过程。

---

## 508. Lipschitz-Controlled Flow Matching for Robust Image Restoration

**arXiv ID:** 2611.13995 | [PDF](https://arxiv.org/pdf/2611.13995v1)

**作者:** Sedjro Salomon Hotegni `[一作]` (Paderborn University), Sebastian Peitz `[通讯]` (Paderborn University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出LipFlower框架，在流匹配图像恢复中引入Lipschitz约束与增益正则化，以降低输入扰动放大；

**💡 创新点**

基于稳定性分析给出的状态与观测Lipschitz限制，设计LipHDiT网络并加入对抗性有限半径增益正则化，首次实现对流匹配图像恢复的鲁棒性提升；

**🔧 技术方法**

使用Lipschitz‑aware hourglass diffusion transformer（LipHDiT）、近似谱归一化、凸残差混合、时间调制、对抗性增益正则化以及SPSA攻击评估；

**📊 数据集**

训练数据为FFHQ，评估使用CelebA‑Test的3k对盲人脸恢复数据；

**📈 对比分析**

与CodeFormer、GFPGAN、RestoreFormer++、OSDFace、PMRF、ELIR、Flow‑from‑Y、DiffBIR、FlowIE、BayesFlow等多种基线在鲁棒性（平均/95%/最大/CVaR）和恢复质量（PSNR、FID、LPIPS、DEG、LMD）指标上对比，LipFlower在Flow‑from‑Y下既保持相近质量又显著降低放大倍率，条件采样在最大放大和感知漂移上更好但质量下降更大；

**⚠️ 局限性**

仅在盲人脸恢复任务验证，未给出全局Lipschitz保证，鲁棒性提升依赖分析假设，对其他逆问题和更强扰动的效果未知。

---

## 509. WINIC: An Automatic Cycle-Accurate Multi-Platform Microbenchmarking Tool for Instruction Performance Metrics

**arXiv ID:** 2611.13978 | [PDF](https://arxiv.org/pdf/2611.13978v1)

**作者:** Tobias Rühl `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Gerhard Wellein `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了WINIC，一款自动化、跨平台、周期精确的微基准工具，能够在用户空间测量x86、AArch64和RISC‑V指令的吞吐量以及基于每个操作数的延迟，并将测得的数据用于改进内核性能模型。

**💡 创新点**

创新点包括：①支持所有主流ISA并且不需要root权限；②自动生成基准循环，能打破隐式依赖；③首次能测量每个操作数的延迟（包括不对称延迟）；④大规模指令集合的系统化测量与比对。

**🔧 技术方法**

技术手段：基于LLVM/Clang实现指令表解析与汇编生成；使用循环展开、依赖打破和操作数延迟链测量；在用户空间通过高精度时钟计时并在子进程中执行；采用自动化脚本在四个不同硬件平台上收集数据。

**📊 数据集**

数据集：在四台硬件平台（AMD Zen 4、Intel Ice Lake、NVIDIA Grace Neoverse V2、Banana Pi Spacemit X60/K1）收集了数千条指令的吞吐量与延迟；与uops.info、llvm‑exegesis、OSACA以及ARM软件优化指南等公开数据库和文档进行对比。

**📈 对比分析**

比较方法：对每条指令在不同工具/文档中的测值进行全匹配、部分匹配或不匹配分类（允许10%误差），统计匹配比例；在两个科学内核（改进后的Schoenauer DAXPY和2D‑5pt Gauss‑Seidel）中，用WINIC测得的操作数延迟替换OSACA模型，发现预测误差从原来的4周期/迭代下降至2周期/迭代，整体预测精度显著提升。

**⚠️ 局限性**

局限性：对所有不对称延迟的范围仍无法完全确定；尚未实现端口使用推断；目前只能测量最多两种指令类型的延迟链，较大组合的精确结果需进一步研究；在某些架构（如部分RISC‑V）中依赖于LLVM提供的指令信息，若缺失则无法测量。

---

## 510. From Attack Surface to Defense Architecture: Compositional Security for Agentic Coding Harnesses

**arXiv ID:** 2611.13960 | [PDF](https://arxiv.org/pdf/2611.13960v1)

**作者:** Simon Hegelich `[一作]` (Evercursion -- GaiaLogic Unipessoal Lda.), Erik Cambria `[通讯]` (Nanyang Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为ActiveSecHarness的 neurosymbolic 参考监控架构，用于保护 AI 编码 harness 免受 Living‑off‑the‑Land 攻击

**💡 创新点**

通过将权限与智能分离、采用多环信任/权限体系、产品型信息流 lattice 以及可验证的工具证书和双键审计日志，解决了传统正则过滤和 LLM 判定无法检测的跨步、共享工作空间和细粒度权限组合的安全缺口

**🔧 技术方法**

采用基于 Macaroon 的 HMAC 能力链、信息流 lattice、三层中介（工具调用、进程/能力、OS 资源）、AST 级别代码验证、分层日志加密以及与 LLM 的异步协同

**📊 数据集**

使用包含 120 个攻击、120 个正常 SWE 任务和 12 条 5 步 kill‑chain 的 252 条 fixture（从公共漏洞和真实项目获取）

**📈 对比分析**

与无防护、正则黑名单和实时 LLM 判定进行对比，ActiveSecHarness 在单步测试中 100% 阻断攻击且 0% 错误阻断，延迟仅 12.7 µs；在 12 条 kill‑chain 中同样 100% 阻断，且相比单链任务的 83% FPR 的粗 Boolean taint，性能保持约 20 µs/步

**⚠️ 局限性**

依赖手工编写的能力链和 lattice 规则，需人工维护证书和日志；对模型本身的推理质量不敏感，但对低阶工具的安全审计仍需要人工或后端验证，且在极端资源耗尽等极端攻击场景下仍有未知风险

---

## 511. ReACT: Reflection-Consistent Modulation and Cross-Load Consistency for Humanoid Velocity Tracking Under Asymmetric Loads

**arXiv ID:** 2611.13929 | [PDF](https://arxiv.org/pdf/2611.13929v1)

**作者:** Xianghui Wang `[一作]` (Hong Kong Polytechnic University), Wei Zhang `[通讯]` (Eastern Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了ReACT框架，用于在仿人机器人携带单侧物体时保持高精度速度跟踪，并验证了其在仿真与真实 Unitree G1 机器人的有效性。

**💡 创新点**

核心创新点是：① 反射一致的扰动调制，将估计的外部力矩拆分为对称与反射奇偶分量并通过可学习的尺度与耦合因子动态调制网络特征；② 跨负载跟踪一致性训练，利用同一速度指令在空载、左侧负载和右侧负载三种分支间对比，逼近不同负载下的相同轨迹，显著降低负载相关的跟踪偏差。

**🔧 技术方法**

采用了强化学习（FastSAC+SE‑Policy）训练时序控制器；引入自学习的六维力矩估计器；利用双分支模调模块实现扰动调制；构建跨负载一致性奖励；在 Isaac Gym 中通过大规模并行环境进行随机化训练；在真实硬件上进行导航实验。

**📊 数据集**

数据集主要为：① 5000+ 并行仿真环境，包含随机摩擦、质心、质量、关节增益与 0–6 kg 单侧负载的随机化；② 真实 Unitree G1 机器人在四个负载（0、1.5 kg 左/右）下的导航轨迹与姿态传感数据；未使用公开的标准机器人数据集，全部为自构造实验数据。

**📈 对比分析**

与公开的 FALCON 方案进行对比。ReACT 在三种速度跟踪协议下，前向、侧向与角速度 MAE 均比 FALCON 低 38–84%；偏差范围下降 90% 以上；在 40 次无地图导航实验中，ReACT 成功率 100%，而 FALCON 全部失败。表明 ReACT 能在携带单侧负载时实现无碰撞目标到达。

**⚠️ 局限性**

局限性：仅在单侧静态负载情况下验证；未处理负载瞬时变化或双侧携带；实验仅覆盖 Unitree G1；缺乏与最新 ADAPT 等方法的公开比较；在极端高速或复杂地形下的性能尚未评估。

---

## 512. OpenRE-Bench: A Benchmark Harness for Multi-Agent Requirements Engineering with Mined GitHub Issues

**arXiv ID:** 2611.13925 | [PDF](https://arxiv.org/pdf/2611.13925v1)

**作者:** Haowei Cheng `[一作]` (Waseda University), Hironori Washizaki `[通讯]` (Waseda University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个面向大型语言模型（LLM）的多智能体需求工程（RE）基准工具，提供可复现的 GitHub issue 挖掘流水线、共享运行框架、适配器机制以及版本化的案例库，用以统一评估不同 RE 框架的性能。

**💡 创新点**

创新点在于：① 通过共享运行契约和可插拔适配器实现框架无关的统一评估；② 设计了可复现的 GitHub issue 挖掘管道，自动生成真实项目的需求案例；③ 发行了可版本化、可扩展的案例集和完整的评测记录；④ 在单一协议下对多种框架（MARE、iReDev、QUARE）进行对比实验。

**🔧 技术方法**

使用的技术包括：Python 3.13 + Docker、GitHub API + TF‑IDF 去重、KAOS‑style 共享表示、BERTScore 语义 F1、ISO/IEC/IEEE 29148 可验证性评分、LLM（如 GPT‑3/4）进行需求建模与协商、统一的运行契约与日志记录。

**📊 数据集**

数据集：从 5 个多语言、多领域的 GitHub 项目共抓取 3,000 条标签匹配问题，经过筛选、聚合后得到 5 条真实挖掘案例（约 368–438 词），与 5 条人工构造案例一起组成 v1.0 版（10 条案例）。

**📈 对比分析**

比较方法：所有框架通过适配器提供相同的输入、Phase‑1~4 产物和运行记录，评估指标包括需求元素计数、MDC（多维分散度）、语义 F1、ISO 可验证性评分和耗时。实验结果显示：iReDev 生成需求数量最多、耗时最长；QUARE 介于两者之间；MARE 生成需求数量最少、耗时最快，说明在统一协议下能客观展示框架差异。

**⚠️ 局限性**

局限性：① 共享契约和预设的 Phase‑1 scaffold 可能掩盖框架原生特性，导致表示偏差；② 评估仅覆盖 Phase‑1~4，未能完整体现所有业务流程；③ 数据挖掘使用固定种子和时间窗口，可能导致主题混杂或代表性不足；④ 案例规模有限（10 条），难以覆盖所有需求类型；⑤ 对 LLM 输出的可复现性仍受模型内部随机性的影响。

---

## 513. Real-Time String Indexing in Optimal Time

**arXiv ID:** 2611.13950 | [PDF](https://arxiv.org/pdf/2611.13950v1)

**作者:** Gregory Kucherov `[一作]` (Universite Gustave Eiffel), Yakov Nekrich `[通讯]` (Michigan Technological University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种能够在任意整数字母表上实现完全实时的字符串索引方案，能够在每个新字符插入时以O(1)的最坏情况时间更新索引，并在查询时以O(|P|+occ)的最坏情况时间报告所有匹配。

**💡 创新点**

核心创新点是：1）路径分解后缀树（Path‑Decomposed Suffix Tree, PST）—仅保留后缀树的子树划分及其质点，而非完整拓扑，从而避免了对预decessor/weighted‑ancestor 的昂贵操作；2）w‑link 机制—对Weiner link 的多字符推广，仅记录路径级别的指向，降低了维护成本；3）通过动态哈希字典（如Cuckoo/Perfect Hashing）实现 O(1) 的查询/更新，结合块划分与快速前驱/后继结构实现对小子树的高效插入。

**🔧 技术方法**

使用的技术包括：后缀树（Weiner 递推）、质点路径（centroid‑path）分解、动态完美哈希/ Cuckoo 哈希、低位前驱/后继（Fusion Tree、B‑Tree）、块划分与重构、动态 LCA 与路径祖先结构、动态 LCA+路径字典、以及随机化字典保证高概率 O(1) 复杂度。

**📊 数据集**

该工作为理论算法研究，无实验数据集，评估基于时间复杂度与空间复杂度的理论证明。

**📈 对比分析**

与之前的工作相比：此前实现仅能在常数字母表或近实时（O(loglog n)）下工作，且多依赖于预decessor/weighted‑ancestor；本方案在任意整数字母表上实现了最优的实时插入时间 O(1)（w.h.p.）和查询时间 O(|P|+occ)，并且空间仅为 O(n) 单词。

**⚠️ 局限性**

限制包括：1）算法依赖于随机化字典，保证高概率 O(1) 但非确定性；2）实现复杂度高，需要多层字典和路径结构；3）仅在单词 RAM 模型下可实现，且对字母表大小无上界的前提；4）在最坏情况下仍需对块重构与重建进行 O(1) 维护，可能导致常数项较大。

---

## 514. Where Do Vision-Language Models Get Left and Right? Position-Code Counterfactuals in Qwen2.5-VL, Qwen3-VL and InternVL3

**arXiv ID:** 2611.14060 | [PDF](https://arxiv.org/pdf/2611.14060v1)

**作者:** Enhao Cui `[一作]` (Changzhou University), Zijian Chen `[通讯]` (Changzhou University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过对VLM的视觉编码器（ViT）位置代码和语言模型侧位置id进行单一轴镜像（position‑only counterfactual）干预，研究这些位置信息对左右、上下及深度关系推理的必要性与充分性。

**💡 创新点**

创新点在于：①首次在保持像素不变的前提下仅编辑位置编码，直接评估其对空间答案的充分性；②将干预细化到ViT不同块（全局注意力与窗口注意力）与LLM侧id/顺序，揭示全局注意力块尤其是最后一块是左右关系的主要信息载体；③利用四选项任务和geometry‑consistent子集，区分真正的方向信息与随机扰动导致的答案变化。

**🔧 技术方法**

主要技术：ViT位置编码镜像（2D‑RoPE或学习绝对嵌入）、LLM侧多维RoPE或1D RoPE id镜像、token顺序重新排列、块级镜像与掩蔽、内容‑only 对照、统计检验（Wilson 区间、配对引导、bootstrap）。

**📊 数据集**

数据集：CV‑Bench Relation（650问，左右/上下）、What'sUp（412问，四选项）、CV‑Bench 3D Depth（验证/测试各314问），以及公开的四个 Fine‑Tuned 版本（SpatialLadder、SpaceQwen、SpaceR、VLM‑R1）。

**📈 对比分析**

比较方法：与真实图像翻转的答案对齐率、geometry‑consistent 子集的 flip‑rate、不同干预的相对效果；在 CV‑Bench 上，ViT width 镜像的对齐率达约 70‑90%（视模型而定），LLM‑侧镜像则几乎无影响；在 What’sUp 上，对齐率为 60‑80%；对 3D Depth，height 镜像对符合“高处更远”线索的答案产生更大影响。整体性能表现显示位置代码对左右答案的决定性作用。

**⚠️ 局限性**

局限性：①位置镜像对细粒度内容（如文字、计数）产生副作用；②四选项任务中非方向性变化和标签先验仍存在；③跨轴泄漏和模型规模限制（仅至 8B 参数）；④实验仅针对单图像空间推理，未涵盖视频时序或不同视角；⑤LLM‑侧位置/顺序在本实验中未显著作用，可能受训练方式限制，未能完全排除弱贡献。

---

## 515. HiPFuse: Real-Time LiDAR Semantic Segmentation for Robotic Perception on Edge Devices

**arXiv ID:** 2611.14025 | [PDF](https://arxiv.org/pdf/2611.14025v1)

**作者:** Haiyang Wu `[一作]` (University of Twente), Ville Lehtola `[通讯]` (University of Twente)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种点云与鸟瞰图（BEV）融合框架HiPFuse，用于单帧LiDAR点云的实时语义分割；

**💡 创新点**

通过在z轴上进行高度分区保留不同高度层次信息，结合点级细节与二维空间上下文，并引入双尺度BEV上下文模块来提升语义判别能力；

**🔧 技术方法**

使用Conv-SE-NeXt模块、残差注意力融合、双尺度BEV上下文、点级编码与聚合、辅助监督（3D/2D），并在Jetson AGX Orin上实现FP32推理；

**📊 数据集**

在SemanticKITTI、MCD（NTU）以及Semantic-ITC（室内）三大公开数据集上进行训练与评估；

**📈 对比分析**

与现有Projection、Voxel、Fusion等方法对比，HiPFuse在SemanticKITTI上取得63.7% mIoU、MCD上62.5% mIoU，且在Jetson Orin上分别达到22.4 FPS和14.2 FPS，均超过10 Hz扫描速率；在下游定位、回环检索和语义映射任务中，其语义输入使得位姿误差下降约10–15%，回环召回率提升约6个百分点；

**⚠️ 局限性**

需要针对不同LiDAR传感器手动设定BEV网格尺寸、z范围与高度分区数，并且需重新训练才能适配新场景，缺乏通用的自适应分区策略，室内数据集上性能仍落后于部分基准。

---

## 516. Energy-Proportional Serverless Computing on SBC Clusters

**arXiv ID:** 2611.14026 | [PDF](https://arxiv.org/pdf/2611.14026v1)

**作者:** Yuliya Sasinskaya `[一作]` (Technische Universität Berlin), David Bermbach `[通讯]` (Technische Universität Berlin)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并实现了一个基于单板机(SBC)集群的无服务器平台，允许在节点级别切换电源，以实现功能实例的scale-to-zero扩展到物理节点。

**💡 创新点**

通过将函数调度与节点电源决策绑定，使无状态函数能让节点在无负载时完全断电，从而显著降低能耗，同时保持可观的延迟。

**🔧 技术方法**

在tinyFaaS平台上加入节点切换控制，利用继电器实现外部电源控制，采用最小负载调度、空闲超时释放等简单策略。

**📊 数据集**

使用四台Raspberry Pi 4B构成的集群，配合稀疏、突发（Azure Functions轨迹压缩）和持续三种工作负载进行实验。

**📈 对比分析**

将功耗和延迟与始终开启节点的基线对比；在突发负载下功耗下降39.6%，稀疏负载下降46.7%，且中位延迟几乎无变化；峰值延迟略升，平均延迟也仅略升。

**⚠️ 局限性**

受限于节点启动时间与领导者/继电器的基准功耗，且对大型集群需要改进函数复制策略；对需要状态保存的函数不适用，无法完全消除领导者电源占用。

---

## 517. Can KV Cache Be Reused Across Model Updates? A DeepSeek-V4-Flash Case Study

**arXiv ID:** 2611.14051 | [PDF](https://arxiv.org/pdf/2611.14051v1)

**作者:** Zhikun Tao `[一作]`, Junping Zhao `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在相同架构、相同参数规模的模型更新中，直接将前一版本的 KV 缓存无映射直接用于后续版本进行推理，评估其在检索、问答质量、时延以及内部状态漂移上的表现。

**💡 创新点**

创新点在于：①提出无映射直接 KV 缓存重用方案并量化其兼容性；②引入 Functional Cache Divergence (FCD) 指标，用以衡量重用导致的预测分布差异；③通过内部状态分析 (NMSE、logit margin) 解释检索与生成行为的改变。

**🔧 技术方法**

使用的技术包括 KV 缓存重用、闭式 ridge mapper 对齐、NIAH 受控检索实验、TTFT（Time To First Token）测量、FCD 计算、以及多层注意力与残差流的内部差异分析。

**📊 数据集**

使用的数据集：NIAH 生成检索任务、BABILong、HotpotQA、2WikiMQA、MultiFieldQA-en（长文本 QA 任务）以及 FineWeb‑Edu（自然文本连续性评估）。

**📈 对比分析**

比较方法：对同一前缀和问题，分别使用 Target、Source、KV Reuse 以及 Ridge Mapper 四种配置计算 NIAH 正确率、错误翻转率、以及 TTFT 加速比。实验表明：强检索模型（GLM‑5.3、DeepSeek‑V4.1‑Flash）在 KV 重用下保持高达 90% 以上的检索准确率，且 TTFT 在 HotpotQA 可实现 5.08×、2WikiMQA 2.10×、MultiFieldQA‑en 2.51×的加速；而弱检索模型 Qwen 在 KV 重用时准确率急剧下降。FCD 在检索任务中随前缀长度呈指数增长，而在自然文本连续性任务中保持低而平稳。

**⚠️ 局限性**

局限性：①仅在同一模型族与相同结构的版本间测试；②缺乏对不同解码策略（如 beam search、top‑k/top‑p）的评估；③对极长前缀时的质量下降未给出完整补偿方案；④内部分析聚焦于选定层级，未覆盖所有层级或潜在对抗性干扰。

---

## 518. CroCoMAE: Cross-modal self-supervised collaborative representation learning for SAR-optical remote sensing data

**arXiv ID:** 2611.14124 | [PDF](https://arxiv.org/pdf/2611.14124v1)

**作者:** Vito Recchia `[一作]` (University of Bari Aldo Moro), Dino Ienco `[通讯]` (INRIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了跨模态自监督学习框架CroCoMAE，用于融合SAR与光学遥感图像。

**💡 创新点**

创新点在于同时在图像级别与补丁级别引入对比学习与掩码自编码，并在补丁级别将特征划分为共享与非共享子空间，实现细粒度跨模态协同。

**🔧 技术方法**

采用Vision Transformer编码器，结合InfoNCE对比损失、MAE重建损失与共享子空间对比损失；整体为多任务联合训练。

**📊 数据集**

使用Sentinel-1 SAR与Sentinel-2光学图像的EuroSAT-SAR（多分类）和DFC2020（语义分割）两个基准数据集。

**📈 对比分析**

与单模态SSL（SimCLR、BarlowTwins、MAE）及跨模态SSL（DeCUR、CROMA）在低标注率（1%/10%）下进行线性探测与全微调比较，CroCoMAE在两任务均显著提升，尤其在语义分割上超过10% mIoU。

**⚠️ 局限性**

局限包括需已配准的双模数据、对大尺寸图像或其他模态的扩展尚未验证，且共享/非共享比例需经验选择；计算成本虽不高，但仍高于单模态对比学习。

---

## 519. Structured Confidence Perturbation via Fisher Geometry for Adversarial Training

**arXiv ID:** 2611.14046 | [PDF](https://arxiv.org/pdf/2611.14046v1)

**作者:** Chenlong Wei `[一作]` (Xi'an Jiaotong-Liverpool University), Xiaobo Jin `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对对抗训练中的伪攻击问题，提出两种方法——单点扰动（SPP）与结构化置信度扰动（SCP），从样本级别和统计流形级别双重角度攻击并强化模型的置信度结构。

**💡 创新点**

创新点包括：① 通过将交叉熵与KL散度结合为混合损失来主动攻击分布一致性，彻底消除伪攻击；② 将对抗搜索视为置信度流形上距离最大化问题，证明最优扰动方向为Fisher信息矩阵的主特征向量，并通过低维分解实现高效计算。

**🔧 技术方法**

主要技术：对抗训练、KL散度、Fisher信息矩阵、特征分解（低维B矩阵）、梯度签名（sign），以及标准对抗攻击算法PGD、CW、AutoAttack用于评估。

**📊 数据集**

使用的数据集包括CIFAR‑10、CIFAR‑100和Tiny‑ImageNet，实验网络结构包括ResNet、DenseNet和ViT。

**📈 对比分析**

通过与PGD、TRADES、MIFGSM、NIFGSM、DIFGSM、PIFGSM等现有对抗训练方法比较，在标准准确率、对抗准确率（AutoAttack/CW）和攻击成功率（ASR）上均优于SOTA，尤其在CIFAR‑10/ResNet‑50上实现最高的AA 72.5% 与最低ASR 12.9%。

**⚠️ 局限性**

局限性：① 在类别数较多或ViT等梯度噪声大的模型上效果下降；② 对λ权重的选择仍需经验调优；③ 虽有低维分解，但SCP在计算量和显存上仍高于纯梯度方法；④ 主要验证在分类任务，尚未探究生成模型或自监督学习中的适用性。

---

## 520. RedirectMimic: Local Motion Content Redaction for Humanoid Whole-Body Control

**arXiv ID:** 2611.14049 | [PDF](https://arxiv.org/pdf/2611.14049v1)

**作者:** Zepeng Wang `[一作]` (Wuhan University), Xiaochuan Shi `[通讯]` (Wuhan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为RedirectMimic的本地运动编辑方法，通过在保持原始参考输入的同时，对已编辑的目标进行双重参考微调，使仿真仿生人能够在不改变外部接口的情况下执行预定的运动替换。

**💡 创新点**

核心创新在于：①利用离线编辑器生成的配对参考（原始参考与编辑后参考），让策略在观测到原始参考的同时跟踪编辑后的目标；②在PPO微调中引入编辑窗口采样，提升稀疏替换任务的学习效率；③通过规则生成的编辑轨迹为局部运动提供明确的替换目标，解决了传统“抑制”方式缺乏具体替换路径的问题。

**🔧 技术方法**

使用的技术包括：基于Proximal Policy Optimization（PPO）的强化学习微调；离线规则编辑器（Rule-based Editor）生成配对参考；编辑窗口采样策略（Edit‑Window Sampling）；以及运动跟踪奖励与物理正则化的组合。

**📊 数据集**

使用了从Retargeted AMASS for Robotics得到的7,509条运动剪辑作为训练和评估数据集，针对Unitree G1仿真机器人进行姿态和速度定义的臂部编辑实验。

**📈 对比分析**

在与ZeroTrack、AntiTrack、ForgetMimic等基线方法的对比中，RedirectMimic在核心区间内容出现率降至2.7%（相较于58.9%和32.8%），核心区间关节成功率达94.2%，替换误差最低，并且在未编辑动作上的跟踪误差和跌倒率与预训练控制器相近，表明其在保持原始运动同时实现高质量编辑的能力。

**⚠️ 局限性**

主要局限包括：仅在仿真环境中验证，缺乏真实硬件上的实验；评估范围局限于两种臂部编辑场景，未展示对更复杂或多身体部位编辑的适用性；编辑窗口的过渡时机与实时流式参考的同步性尚未充分验证；方法依赖离线编辑器生成的高质量编辑轨迹，对编辑规则和目标的制定要求较高。

---

## 521. A human-in-the-loop Control Paradigm for Construction Robotics: From VR-Guided Teleoperation to Semi-Autonomous Dual-Arm Operation

**arXiv ID:** 2611.14062 | [PDF](https://arxiv.org/pdf/2611.14062v1)

**作者:** Jinhua Chen `[一作]` (Hong Kong University Of Science And Technology), Changying Xiang `[通讯]` (Hong Kong University Of Science And Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种人机共轭的控制范式，能够在VR引导的遥控、半自动共享控制与基于规则的全自动之间实现结构化过渡，应用于双臂高层幕墙施工任务（钻孔、支架安装、螺钉紧固），并在实验平台上进行了验证。

**💡 创新点**

创新点包括：①将遥控、共享与全自动视为连续控制谱，支持任务级结构化过渡；②构建三层控制层次与共享自治框架，明确人机职责分配与命令优先级规则；③引入子任务分配比例（SAR）量化人机协同；④结合VR沉浸式引导与手动定义的技能库，实现高精度接触操作；⑤对不同控制模式进行定量比较并评估人因工作量。

**🔧 技术方法**

技术手段包括：Meta Quest 3 VR头显与控制器实现手势捕捉；双臂FAIRINO FR5机器人与多功能末端执行器；基于YOLOv6的目标检测与深度相机定位；六轴力/扭矩传感器与混合力/位姿控制；ROS+MoveIt实时规划与安全过滤；共享自治融合权重α/β与命令优先级规则；NASA-TLX问卷评估主观工作负荷。

**📊 数据集**

本文未使用公开数据集，而是在实验室幕墙模型上自行搭建标定点、深度图与视觉标记，利用YOLO检测与点云平面估计完成目标定位与姿态估计。

**📈 对比分析**

通过十名无VR经验的新手操作员，分别在TeleOp、Semi-Auto与Auto模式下完成钻孔、支架安装和螺钉紧固三类任务，共计1,000次实验。评估指标包括成功率、深度、位置误差、垂直误差和任务时长。实验显示：在标定良好的条件下，Auto模式取得最高成功率与最小误差；Semi-Auto显著优于TeleOp，在成功率、精度与时长上均有提升；在灯光衰减、遮挡、干扰等环境挑战下，Semi-Auto保持更高的鲁棒性；NASA-TLX显示Semi-Auto的主观工作负荷降低约46%。SAR sweep进一步证实人机分配越多，精度与成功率越低。

**⚠️ 局限性**

局限性包括：实验仅在室内模型上进行，未在高空真实施工现场验证；环境扰动仅覆盖光照与遮挡，未涉及目标位移、墙面倾斜等；仅评估钻孔的鲁棒性；仅使用主观工作量指标，缺乏客观生理或运动量测；未进行全因子剖析或学习式基准对比；技能库固定，未引入在线学习或自适应机制；样本量有限，未进行群体统计分析。

---

## 522. Resilient Integrated Access and Backhaul Network Planning via Graph Neural Network Curriculum Learning

**arXiv ID:** 2611.14042 | [PDF](https://arxiv.org/pdf/2611.14042v1)

**作者:** Jie Zhang `[一作]` (University of York), Qiang Ni `[通讯]` (Lancaster University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文针对毫米波IAB网络的部署规划，提出一种基于图注意力网络与课程学习的鲁棒性规划框架，能够在不同IAB捐献者拓扑下实现覆盖率与多跳后备传输的双重约束。

**💡 创新点**

创新点包括：①将覆盖、传输预算、父节点冗余等多项约束融入MDP并通过GATv2编码捕捉多跳依赖；②设计四阶段课程学习策略，逐步提升覆盖率、冗余度与拓扑多样性；③引入灾难隔离、软混合重放等稳定机制以缓解灾难性探索导致的训练波动。

**🔧 技术方法**

使用技术包括：图注意力网络v2（GATv2）+边条件注意力、Proximal Policy Optimization（PPO）强化学习、动作掩码、分层奖励设计、课程学习与灾难隔离策略。

**📊 数据集**

数据集与实验场景：1000 m×1000 m 60 GHz城市仿真，候选站点间距50 m；覆盖掩码与链路预算基于3GPP TR 38.901；IAB捐献者布局多样化（五骰、五边形、六边形、垂直/水平、稀疏随机、密集簇集）。

**📈 对比分析**

与Direct‑Hard PPO、Graph‑DQN、Dueling‑DQN、贪心算法等基线对比，采用成功率、部署节点数、覆盖率、服务需求占比等指标。课程学习GATv2–PPO在98%覆盖率与双父连接要求下成功率≥95%，平均部署节点数比基线低10–20%，覆盖率与服务比例均略优；在动态流量与链路失效实验中，其服务占比和后失效覆盖率最高。

**⚠️ 局限性**

局限性：①训练过程依赖多阶段课程手工设定，缺少自动化或自适应难度调节；②对密集簇集等极端捐献者布局的成功率仍显下降；③模型仅在离线规划阶段使用，缺乏在线链路动态、自适应重规划与移动终端排队延迟的考虑；④在更大候选集或不同频段下的泛化尚未验证。

---

## 523. Visual Similarity in Adult Word Extension and Artificial Word Learning

**arXiv ID:** 2611.14116 | [PDF](https://arxiv.org/pdf/2611.14116v1)

**作者:** Patricio M. Vera `[一作]` `[通讯]` (Neurocreaciones), Patricio M. Vera (Neurocreaciones)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对成人词汇扩展实验数据进行二次分析，并使用人工学习者在相同视觉空间内检验词汇扩展与视觉相似度的关系

**💡 创新点**

首次将自监督视觉模型（DINOv2）与成人词汇扩展行为及人工学习者的性能进行对比，揭示视觉相似度与教学例子组合共同影响扩展范围

**🔧 技术方法**

采用冻结的自监督视觉编码器（DINOv2）、基于质心的距离计算、逻辑混合模型、人工学习者（原型网络）以及统计检验（Mantel检验、Holm校正）

**📊 数据集**

成人实验数据（lewis2018）、CIFAR-10/100图像、THINGS、ImageNet-HSJ、CIFAR-10H人类相似度/混淆度数据集

**📈 对比分析**

通过逻辑混合模型量化距离对成人扩展的影响（log odds显著负相关），人工学习者的命名准确率随类别距离递减（近→中→远梯度），与人类相似度测度呈部分相关，但未能显著预测学习难度

**⚠️ 局限性**

主要局限在于成人分析为相关性且仅使用DINOv2的CLS嵌入；人工学习者受限于冻结编码器、CIFAR图像、两类词汇、浅层学习；人类相似度测度混合感知与概念，样本量小，未检验不同层或模型的稳健性

---

## 524. Where do we Jam? - The Impact of Real and Virtual XR Spaces on Musical Creativity, Co-Presence, Affect, and Flow

**arXiv ID:** 2611.14029 | [PDF](https://arxiv.org/pdf/2611.14029v1)

**作者:** Ruben Schlagowski `[一作]` (University of Augsburg), Elisabeth André `[通讯]` (University of Augsburg)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究在实验室设置下，让28位音乐家在AR和VR两种XR环境中进行网络即兴演奏，测量创意支持、情感、共在、沉浸与流动等心理指标，并收集访谈资料进行质性分析。

**💡 创新点**

创新之处在于首次将可定制的虚拟环境与网络音乐表演相结合，探讨空间维度对创作与沉浸的影响，并同时采用定量与定性方法验证。

**🔧 技术方法**

使用Varjo XR-3 HMD、Azure Kinect DK摄像头、Unity引擎与Varjo SDK、Gaussian Splatting点云着色、MRTK 3D用户界面、手部跟踪等技术实现AR/VR交互与实时点云传输。

**📊 数据集**

数据来源于28名受试者的实验记录，包括自评量表（PANAS、IPQ、NMSPI、FKS、CSI）、3D点云日志、虚拟环境选择与定制日志以及访谈转录文本。

**📈 对比分析**

通过配对t检验、TOST等统计方法对AR与VR进行比较，结果显示VR显著提高参与度，但在创意支持、情感、共在与流动等方面未出现显著差异，说明个体差异和环境偏好是关键因素。

**⚠️ 局限性**

局限性包括样本量小、性别偏向、实验时长短、技术缺陷（如手部跟踪不稳定、VR手臂模型干扰、AR校准问题、点云误差、HMD遮挡面部）、未对德语版NMSPI进行心理测量验证以及潜在的配对依赖效应。

---

## 525. ScaleBrain: A Scale-Consistent Framework for EEG Foundation Models

**arXiv ID:** 2611.14035 | [PDF](https://arxiv.org/pdf/2611.14035v1)

**作者:** Minyu Zhou `[一作]` (Nankai University), Wensheng Zhang `[通讯]` (Nankai University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 ScaleBrain，一种多尺度一致的 EEG 基础模型预训练框架，用于在大规模无标签 EEG 语料上进行自监督预训练，并在六个公开 BCI 基准上进行微调。

**💡 创新点**

核心创新点包括：1) 明确对齐多尺度输入（不同时间窗）与多尺度监督目标（scale‑specific reconstruction），保证尺度一致性；2) 采用多尺度 EEG tokenization 与共享时间锚点，形成可比较的多尺度表示；3) 引入尺度感知的时空位置编码，保持尺度身份；4) 通过门控尺度融合与轴向时空注意力将多尺度表示统一并上下文化；5) 采用平衡权重的多尺度损失，避免粗尺度主导。

**🔧 技术方法**

技术手段包括：自监督掩码重建（masked reconstruction）目标；STFT 与三层卷积的双分支特征提取；多尺度 tokenization（s1=200, s2=100, s3=40 采样点）；尺度感知位置编码；门控尺度融合（SwiGLU 结构）；轴向时空 Transformer（Axial ST attention）；以及均衡权重的多尺度损失。

**📊 数据集**

预训练使用 Temple University EEG（TUEG）语料，包含约 9,800 小时 EEG；下游评测六个公开 BCI 任务与数据集：PhysioNet‑MI（运动想象）、SEED‑V（情感识别）、HMC（睡眠分期）、MentalArithmetic（精神压力检测）、TUAB（异常检测）和 TUEV（事件类型分类）。

**📈 对比分析**

与 12 个基线模型（6 个任务特定模型 + 6 个 EEG 基础模型）在六个数据集上进行对比，使用 Balanced Accuracy、AUC‑PR、AUROC 等指标。ScaleBrain 在 5/6 个数据集上获得最高 Balanced Accuracy，第三名仅落在 SEED‑V；使用 5 M 参数即可超过 69 M 参数的 REVE‑Base，在参数效率上表现突出。通过 ablation 实验验证多尺度输入与多尺度监督、尺度感知位置编码、轴向注意力等模块的贡献。

**⚠️ 局限性**

局限性包括：1) 仅针对 EEG 进行了验证，尚未测试其他脑电/脑磁/影像模态；2) 多尺度数量固定为三尺度，更多尺度并不一定提升性能；3) 主要评测在标准公开数据集，缺乏跨域或实时场景的验证；4) 对于不同采样率、通道数的通用性需要进一步探索；5) 计算开销与推理延迟未作深入分析。

---

## 526. From Show Programs to Linked Data

**arXiv ID:** 2611.14037 | [PDF](https://arxiv.org/pdf/2611.14037v1)

**作者:** Clarisse Bardiot `[一作]` (University of Rennes), Jeanne Fras `[通讯]` (University of Rennes)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

将法国阿维尼翁节的剧场节目手册通过视觉‑语言模型转写后，结合自定义的Linked Art表演艺术本体，对文本进行结构化语义标注，生成可查询的RDF数据；

**💡 创新点**

提出了完整的端到端工作流，首次将VLM与表演艺术专属本体相结合，显著降低LLM的幻觉率，并实现了规模化的可互操作节目档案；

**🔧 技术方法**

使用视觉‑语言模型（Claude Sonnet 3.7、GPT‑4o等）进行转写；采用Linked Art Performing Arts（LA‑PA）本体进行语义建模；通过强化学习（Pleias‑350m）、监督微调（POntAvignon‑4b）以及提示编码的Claude Opus 4.8进行自动语义标注；

**📊 数据集**

以阿维尼翁节1947‑至今的1,840份节目手册为数据集，涵盖1954‑2022年68年，包含多种排版与语言；

**📈 对比分析**

与传统OCR（EasyOCR）相比，VLM转写的Levenshtein比率达0.98，NER比率0.99；POntAvignon‑4b在测试集上实现token准确率96.6%，有效JSON率97%；整体方法在文本转写和结构化提取上均优于现有方法；

**⚠️ 局限性**

局限性包括：VLM转写依赖专有API，版本变动可能导致不可复现；本体与模型同步滞后，导致部分新模型无法覆盖最新概念；POntAvignon‑4b在大型节目（30+成员）时会被截断；当前使用的LLM为商业模型，缺乏完全开源替代方案；

---

## 527. Model-Free Runtime Measurement of Task-Relevant Information

**arXiv ID:** 2611.14111 | [PDF](https://arxiv.org/pdf/2611.14111v1)

**作者:** Dimitris Panagopoulos `[一作]` (Cranfield University), Weisi Guo `[通讯]` (Cranfield University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并验证了一种基于任务需求的模型无关运行时监控框架——Acquisition-State Telemetry (AST)，用于实时评估自主系统的信息获取进度与摩擦。

**💡 创新点**

引入任务绑定的观测映射与固定因果监视器，构建两条连续信号（进度与信息摩擦），实现跨域、无模型的运行时知识获取监督。

**🔧 技术方法**

采用任务绑定的分辨率映射 S 与比较表示 e，使用贝叶斯进度估计器 (PE^B) 与正交子空间停滞指数 (SI^⊥)，并在无训练的固定配置下进行因果监控。

**📊 数据集**

在八个不同场景下测试：SAR 对话、MultiWOZ 2.4、HCRC Map Task、AnnoMI、MiniGrid、Crafter、MetaWorld、BOIR。

**📈 对比分析**

与未监控基线、仅操作成本惩罚、仅观测 AST、惩罚 AST 摩擦、以及完整 AST 奖励的 PPO 策略比较；在早期终止条件下，惩罚摩擦的策略完成率超过 85%，远高于基线约 7%。

**⚠️ 局限性**

需要先验的有限任务规范且信息累积保持单调，无法处理时间变化或欺骗性观测；监控信号幅值依赖于任务映射，需扩展至动态或学习式任务结构。

---

## 528. Obscured Wildfire Flame Detection by Spatio-temporal Analysis of Smoke Patterns Using Frame-wise Transformers

**arXiv ID:** 2611.14074 | [PDF](https://arxiv.org/pdf/2611.14074v1)

**作者:** Uma Meleti `[一作]` (Clemson University), Fatemeh Afghah `[通讯]` (Clemson University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于帧级变换器的无人机RGB视频烟雾时空分析方法，用于检测被浓烟遮挡的野火火焰。

**💡 创新点**

创新点包括：① 仅关注帧间注意力，保持帧内空间信息不被混合；② 在低分辨率特征上使用更多帧来捕获长时序烟雾运动；③ 通过阈值策略实现帧深度与空间分辨率的动态权衡。

**🔧 技术方法**

使用 VGG‑16 作为特征提取器，卷积式 Transformer（自注意力 + 卷积投影 + 位置嵌入）实现帧级聚合；采用 Dice 损失训练，并用滑动窗口方式进行实时推理。

**📊 数据集**

在 FLAME2 数据集上进行实验，选取无人机停稳、RGB 与 IR 对齐的片段，用 IR 处理得到的热图作为训练标签；共 594 条训练样本和 255 条测试样本。

**📈 对比分析**

与传统 Non‑Local 注意力以及高分辨率/低分辨率两种帧深度策略对比；在 20 帧滑动窗口下，Dice 92.61%，对象检测 F1 90.96%，clip‑level 分类准确率 98.02%；相较于 Non‑Local 在 0.3 阈值下提升约 7% 的 F1 分数；低分辨率多帧方案在保持相同精度的前提下将算力降低 72 倍。

**⚠️ 局限性**

局限性包括：① 训练仍依赖配套 IR 影像，无法完全摆脱双摄需求；② 只在 FLAME2 这类实验室/受控场景验证，未知在不同气候、不同火源类型下的泛化能力；③ 对极端浓烟或光照不佳的情况仍可能误检或漏检。

---

## 529. ZoomGain: Mining Global-to-Zoom Perception Gaps for Fine-Grained Visual Supervision

**arXiv ID:** 2611.14076 | [PDF](https://arxiv.org/pdf/2611.14076v1)

**作者:** Haozhan Shen `[一作]` (Ant Group), Jianwei Yin `[通讯]` (Ant Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种基于全景与放大视图感知差距的自动化数据构造管道ZoomGain，利用该管道构建了ZoomGainBench基准，并使用该数据对VLM进行训练以提升细粒度视觉感知能力。

**💡 创新点**

创新点主要包括：
1) 通过对比全图与缩放视图差异来发现并利用细节，自动生成需细粒度感知的问题；
2) 采用多模型一致性验证与盲目验证，确保答案正确且不受语言先验影响；
3) 构造多源干扰项（视觉证据、语言先验、全景错误）来提升评测严谨性；
4) 完全由模型生成且仅人工筛选，显著降低人工成本；
5) 证明该数据可显著提升多种规模模型的细粒度与通用感知表现。

**🔧 技术方法**

使用技术包括：
- 多模型（VLM与LLM）生成全图与放大描述并提取增量细节；
- 多模型一致性检验与盲目检验以验证答案；
- 基于视觉、语言与全景错误的干扰项挖掘；
- 强化学习（GRPO）在ZoomGain/ZWZ QA上微调VLM；
- 对比实验与错误分析。

**📊 数据集**

主要使用数据集：
- 12K高分辨率图像来自ZwZ，生成23,610 QA对；
- 1,000问答被筛选为ZoomGainBench基准；
- 剩余9,000 QA用于训练；
- 对比使用原始ZwZ 9K QA；
- 评估基准包括ZoomBench、V*Bench、HR-Bench、MME-RealWorld-Lite、TreeBench、VisualProbe、CV-Bench、MMStar、MMVP、BabyVision、PerceptionBench、O3-Bench 等。

**📈 对比分析**

对比方法：在28个VLM上评测ZoomGainBench，最优模型仅达61.08%；在多达14个基准上，ZoomGain训练模型相较ZWZ训练模型平均提升6.19分；仅用9K样本的9B模型得到66.10分，超过多款大模型；细粒度基准提升最显著；总体来看ZoomGain数据能在保持样本成本低的情况下显著提升模型细粒度感知与部分通用感知表现。

**⚠️ 局限性**

局限性：
1) 在一般感知与复杂推理任务上仍受模型规模限制；
2) 依赖多模型一致性验证，若参考模型本身存在偏差可能导致误标签；
3) 生成流程主要聚焦全图-缩放对比场景，可能不适用于更复杂或不同场景的细粒度任务；
4) 由于自动化流程，仍需人工筛选，难以完全消除人工成本。

---

## 530. Sable: A Small-Field Sumcheck Argument for Bit-Exact Inference of Large Language Models

**arXiv ID:** 2611.14083 | [PDF](https://arxiv.org/pdf/2611.14083v1)

**作者:** Tomer Solberg `[一作]` `[通讯]` (MoonMath.ai), Tomer Solberg (MoonMath.ai)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

Sable提出了一种单 GPU 可证明推理系统，能够对 Qwen 等大型语言模型的整数量化推理给出可验证的完整证明。

**💡 创新点**

创新点在于构造了一个无包裹的 31 位域约束系统、精确整数变换、流式多项式求证以及多重线性承诺+BLAKE3+BaseFold，兼顾大规模模型与可验证推理。

**🔧 技术方法**

使用的技术包括整数变换 F_O、Reed–Solomon RS 编码、BLAKE3 Merkle 树、BaseFold FRI、sumcheck、GKR、LogUp 以及 CUDA 动态生成的求证核。

**📊 数据集**

模型在 Qwen2.5‑32B、Qwen3‑32B 以及 Qwen3‑30B‑A3B 上验证，并在 WikiText‑2、MMLU、GSM8K 上评估精度。

**📈 对比分析**

在 RTX PRO 6000 上单层 4K 令牌 92.5 秒、16K 令牌 398 秒，证明大小 134.8 MiB，验证时间 19–56 秒；相比 zkLLM，参数/令牌成本低 40–64 倍；精度误差 8.3% perplexity 下降可通过 Q/K 分割改善。

**⚠️ 局限性**

主要限制是证明非零知识、证明大小 100–500 MiB、对开权重模型依赖且对整数化误差敏感；闭合权重需要额外隐藏承诺。

---

## 531. DEIA: Evidence-Grounded Crash Repair with Change-Aware Test Design for HarmonyOS ArkTS

**arXiv ID:** 2611.14064 | [PDF](https://arxiv.org/pdf/2611.14064v1)

**作者:** Xuhui Lu `[一作]` (Beihang University), Zheng Zheng `[通讯]` (Beihang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个面向HarmonyOS ArkTS的基于证据的崩溃修复工作流DEIA。

**💡 创新点**

将崩溃报告、源代码、LLM生成的编辑、构建与设备运行验证等跨阶段证据链连接起来，并提出基于F2P/P2P的义务设计与阶段成本账本。

**🔧 技术方法**

采用大语言模型（GPT‑6.1‑Sol）进行诊断与编辑建议，使用Deterministic writer进行差异合成，结合HarmonyOS构建工具和模拟器进行包签名与运行，构建成本账本。

**📊 数据集**

30个来自24个HarmonyOS ArkTS应用的崩溃修复任务，构建自公开的ArkTS修复集与对应的崩溃报告。

**📈 对比分析**

与OpenHands基线进行匹配比较，DEIA在参考位点一致率提升约20.7%，模型调用/令牌使用降低81.5%，候选生成耗时减少34.9%，但仅有两条行为特定P2P检查被执行。

**⚠️ 局限性**

仅验证了定位与可构建性，行为保真性验证不足；实验仅单次运行、未固定模型版本，且受限于HarmonyOS 26模拟器与特定环境，缺乏跨平台泛化。

---

## 532. Few-Step Streaming Video Generation via Condition Shifting

**arXiv ID:** 2611.14033 | [PDF](https://arxiv.org/pdf/2611.14033v1)

**作者:** Deyuan Liu `[一作]` (Tsinghua University), Jintao Zhang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在长序列视频流式生成中，引入条件转移（Condition Shifting）机制，使生成器内部同时学习真实与假速度估计，从而实现更高效的分布匹配和指令跟随。

**💡 创新点**

创新点包括：① 用条件转移在同一网络中实现假估计，消除独立假网络；② 将此机制融入S2-Avatar的Self‑Replay Forcing，保留重播与感知正则化；③ 在180秒视频上展示更佳指令遵循与美学表现，并在训练效率上取得显著提升。

**🔧 技术方法**

主要技术：连续生成模型（流匹配 + 拟数估计）、条件转移、Self‑Replay Forcing、冻结预训练教师、分布匹配损失与感知正则化。

**📊 数据集**

使用S2-Avatar全量训练数据集，涵盖直播、对话头、影视、舞蹈、2D/3D动画等多场景；在StreamAV‑Bench 160个Progressive与160个Interactive情景上进行评估。

**📈 对比分析**

与S2‑Avatar和DMD在相同设置下比较；在人类评价（70对比）中指令遵循和视觉美学得到显著提升（+30%净偏好），在全量benchmark中VA、VIF、AIF排名第一；训练时显存从102.7 GiB降至98.2 GiB（-4.4%），平均迭代时间从35.9 s降至32.6 s（-9.2%）。

**⚠️ 局限性**

局限性包括：在运动质量和视听同步上仍略逊于S2‑Avatar；在极低采样步（2 NFE）下，部分维度（如同步、运动）表现下降；对教师模型的依赖使得整体架构仍需保持冻结预训练网络。

---

## 533. HandsOnAid: Building and Evaluating an Offline First-Aid Assistant for Emergency Response Scenarios

**arXiv ID:** 2611.14059 | [PDF](https://arxiv.org/pdf/2611.14059v1)

**作者:** Rezwan Islam Salvi `[一作]` (Islamic University of Technology), Mushfiqur Rahman Mushfique `[通讯]` (Islamic University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建并评估了一个名为HandsOnAid的小型语言模型，专门用于在无网络、无救护车的紧急场景中提供安全、准确的急救指引；对七种不同配置（基线、4‑bit/8‑bit微调、长度上限、安保门、检索）进行了比较；采用了四个前沿LLM做盲评，辅以医学专业人员的人工评测；并在Pixel 7 Pro手机上验证了离线推理性能。

**💡 创新点**

①首次在离线急救场景中系统比较小模型的量化与检索对安全性和准确性的影响；②将LLM作为盲评判者引入，并使用隐藏对照答案检验评判者的可靠性；③分析训练语料对ANZCOR指南覆盖率的影响，指出数据不足是低分的主要原因。

**🔧 技术方法**

使用Gemma 2B作为基础模型；采用QLoRA对5,550对急救问答进行4‑bit（和8‑bit）微调；引入检索（从FirstAidQA中检索相似QA对）；实现安全门对答案的预检查；利用四个大型LLM（GPT‑5.6 Sol、Claude Opus 5、DeepSeek V4 Pro、GLM‑5.3）在温度为0的条件下进行盲评；在Pixel 7 Pro手机CPU上进行离线推理。

**📊 数据集**

FirstAidQA数据集（5,550个问答对，已公开）以及专门构造的41个急救问答对，所有问题均遵循ANZCOR指南并由医学专业人士审核。

**📈 对比分析**

评估方法：用四个LLM盲评每个答案，并混入45个控制答案检查评判一致性；采用0–5分制严格评分规则（12项安全覆盖）；再由医学专家对40个答案进行人工评估。结果显示：微调后平均分从基线的1.26提升至2.34，最佳配置（4‑bit微调+检索）得到2.51/5；医学专家评分与LLM评估接近。设备性能：Pixel 7 Pro在无网络环境下，首次文字出现6.86 s，完整答案平均14.88 s。

**⚠️ 局限性**

①训练语料对ANZCOR指南某些细节的覆盖不足，导致模型缺失关键信息；②小模型的表达能力受限，无法完全实现高分标准；③安全门在评判时表现有限；④评估依赖LLM判断和人工评分，可能存在主观性与一致性问题。

---

## 534. Generalizing Causal Decision-Making: Markov Equivalence, Transportability and Counterfactuals

**arXiv ID:** 2611.14066 | [PDF](https://arxiv.org/pdf/2611.14066v1)

**作者:** Min Woo Park `[一作]` (Seoul National University), Sanghack Lee `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在结构因果Bandit（SCB）的框架上，扩展了三大方向：①在仅知马尔可夫等价类（PAG）时对最小干预集（MIS）与可能最优最小干预集（POMIS）进行图形化判定并给出搜索算法；②通过部分可转移性（partial transportability）和优势关系（dominance bounds）推导跨环境的目标奖励下界，进而裁剪动作空间；③设计基于可实现实验能力的反事实SCB模型，并给出可行动作的图形判定与算法。

**💡 创新点**

创新点包括：
- 将MIS和POMIS的定义推广到最大祖先图（MAG）和部分祖先图（PAG）层面，并提供必要且充分的图形判定；
- 引入“确定性最小干预集”（DMIS）以弥补PAG上MIS判定的真空；
- 开发部分可转移性框架，处理非可操纵变量时的奖励转移和优势裁剪；
- 将反事实干预纳入SCB，允许在实验能力受限时仍能执行特定反事实动作；
- 给出累计回报的理论上界与实验验证，证明动作空间裁剪与奖励转移可显著降低累积损失。

**🔧 技术方法**

技术方法主要包括：结构因果模型（SCM）与do-算子；图形分离（d-分离、m-分离）和可导向路径判定；最大祖先图与部分祖先图的构造与判定；Fast Causal Inference（FCI）与局部变换；规则3的do-推理；部分可转移性与优势关系推导；以及针对MIS/POMIS的递归搜索算法。

**📊 数据集**

实验使用合成的结构因果模型（多变量SCM），在这些模型上模拟Bandit过程，实验规模为 50,000 步长，重复 100 次，比较不同动作空间裁剪与奖励转移方案的累计回报。

**📈 对比分析**

与传统无结构Bandit或仅使用完整动作空间的SCB相比，本文方法通过裁剪冗余干预集和利用跨环境奖励信息，显著降低累计回报的上界；实验结果显示，回报损失曲线明显优于基线，并且在理论上提供了可满足的回报上界保证。

**⚠️ 局限性**

局限性包括：
- 对PAG上MIS的判定存在真空，DMIS定义虽解决此问题但缺乏直接图形判定规则；
- MIS/POMIS的枚举在最坏情况下指数级，导致搜索复杂度高；
- 部分可转移性框架假设有效先验边界，实际中难以保证；
- 反事实干预需要事先明确定义实验能力，限制了通用性；
- 仅在单个环境或有限动作集合上验证，未讨论连续动作空间或动态策略；
- 对非可操纵变量的处理仍依赖于强假设，实际应用中可能不成立。

---

## 535. Continuous 3D Spatial Transcriptomics Reconstruction via Histology-Informed Implicit Neural Representations

**arXiv ID:** 2611.14129 | [PDF](https://arxiv.org/pdf/2611.14129v1)

**作者:** Kaito Shiku `[一作]`, Muhammad Nabeel Asim `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种利用组织学图像信息的多任务隐式神经表示（HIM-INR）来从稀疏的三维空间转录组（ST）观测中连续重建基因表达。

**💡 创新点**

创新点在于将高分辨率组织学特征作为辅助监督，联合训练共享的INR骨干，弥补仅基于稀疏ST数据难以学习空间连续性的缺陷，并且在推理时仅需坐标即可预测基因表达。

**🔧 技术方法**

采用的技术包括SIREN式隐式神经表示、低维潜在空间的自编码器、共享的多任务网络、片段特异性缩放偏置校正以及组织学特征的预训练UNI2-h模型提取。

**📊 数据集**

使用的公开数据集为三维空间转录组3DST-RA（类风湿关节炎组织）和HemiBrain（小鼠脑组织），每个数据集包含多层切片且仅保留部分切片的ST数据。

**📈 对比分析**

与基于最近邻、KNN、STAGE、3D-STINR、3D-SUICA、SINTER3D等六种基线方法对比，HIM-INR在所有指标（Top 50/250 高变基因的 PCC）上均优于现有方法，尤其在 ST 稀疏时优势更显著。

**⚠️ 局限性**

局限性包括对预训练组织学模型的依赖、仅在公开数据上验证、缺乏对不同组织或疾病类型的泛化评估，以及在极端稀疏或缺失组织学图像时性能下降的风险。

---

## 536. LEASH: Stay Safe in Agentic Robotics

**arXiv ID:** 2611.14115 | [PDF](https://arxiv.org/pdf/2611.14115v1)

**作者:** Qinfeng Zhu `[一作]` (Xi'an Jiaotong-Liverpool University), Kaizhu Huang `[通讯]` (Duke Kunshan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了RoboRisk基准和LEASH安全框架，对基础模型驱动的机器人进行评估与改进。

**💡 创新点**

创新点在于全面覆盖五类风险的基准，并设计了基于决策模型的三阶段安全门“LEASH”，实现对任务、动作和不可逆操作的实时监督。

**🔧 技术方法**

使用决策模型（Jev、OpenAI Decisions）实现安全门，结合共享安全策略；在MuJoCo仿真与真实Dobots平台上测试。

**📊 数据集**

RoboRisk数据集：92个仿真任务+13个真实机器人任务，涵盖Task-level Refusal、Plan Repair、Execution-time Hazard、Injection、Sequential Memory五类风险。

**📈 对比分析**

对六个主流基础模型（GPT‑6 Astra、GPT‑6.1 Sol、GPT‑6 Luna、Claude Opus 5.5、Gemini 4 Argon、Grok 4.7）和两种机器人特定API进行评估；LEASH将Harm Rate从最高77%降至0%或接近0，Safe Success Rate略有提升，时间与成本提升仅≈1.6%。

**⚠️ 局限性**

局限性包括对决策模型准确性的依赖、弱规划器可能绕过门禁、无法在多人场景下平衡安全与协作，以及对长技能执行时间监控的覆盖不足。

---

## 537. Looking One Step Ahead: An Anticipation-guided Agent for Object State Readout in Videos

**arXiv ID:** 2611.14113 | [PDF](https://arxiv.org/pdf/2611.14113v1)

**作者:** Yuanhe Tian `[一作]` (Zhongguancun Academy), Yan Song `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于前瞻预测的对象状态读取框架LASER，利用可能的下一步变化指导对当前对象属性的视觉检验，并通过后续帧验证修正描述。

**💡 创新点**

创新点在于将可能的状态变化转化为可检验的视觉问题，构建可复用的文本观察指令库，并通过对比生成与参考描述的差异来迭代优化指令，从而在不更新模型参数的前提下提升状态描述质量。

**🔧 技术方法**

技术包括：多模态大语言模型（MLLM）作为视觉推理器；前瞻性属性选择（predictive attribute selection）；顺序视觉校正（sequential visual correction）；基于反馈的技能修订（feedback‑guided skill revision）；以及离线指令库的迭代更新。

**📊 数据集**

使用的主要数据集为OSC​aR（含7K训练、500测试）和PLM‑RCap（含179K训练、10K测试）两大对象级视频描述基准。

**📈 对比分析**

在四种MLLM骨干（Qwen3.5‑4B、Qwen3.5‑9B、Qwen3.5‑35B‑A3B、Ministral‑3‑3B）上与直接生成、顺序读取、通用重检、属性清单、自由预测等基线以及五个现有方法对比，LASER 在 BLEU、ROUGE‑1/2/L、和官方相似度得分上均实现了显著提升，尤其在Qwen3.5‑9B 上获得最高的 ROUGE‑L 32.0 与相似度 29.6。

**⚠️ 局限性**

局限性包括：依赖手工设计并迭代的观察技能库，仍需人工或外部参考来指导技能修订；在对高度动态或极端遮挡场景的推断仍存在误差；以及对复杂多对象交互的扩展尚未充分验证。

---

## 538. Stylization as Geodesic Navigation on Diffusion Manifolds

**arXiv ID:** 2611.14097 | [PDF](https://arxiv.org/pdf/2611.14097v1)

**作者:** Yan Li `[一作]` (HKUST), Yike Guo `[通讯]` (HKUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种无训练的双控制框架，在预训练扩散模型的曲率隐空间中通过几何引导的插值和自注意力查询优化，实现对人脸图像的可控风格迁移。

**💡 创新点**

创新点在于将拉回Riemannian度量与拉格朗日能量最小化结合，利用秩一雅可比近似进行几何优化，在隐空间和注意力空间分别实现连续、细粒度的风格强度控制，同时保持身份保真。

**🔧 技术方法**

使用技术包括Stable Diffusion/DiT预训练模型、拉回度量、能量最小化、幂迭代近似雅可比、梯度投影以及自注意力查询的几何优化。

**📊 数据集**

实验采用CelebA（人脸内容）、WikiArt-Face与MetFace（艺术风格）三大数据集。

**📈 对比分析**

与DiT编辑、Diffusion风格迁移、CNN风格迁移、ControlNet/IP Adapter等多种基线对比，在ArtFID、LPIPS、身份损失等指标上实现了最优或相近的性能，尤其在身份保持和内容保真方面显著优于现有方法。

**⚠️ 局限性**

局限性包括仅在局部曲率附近收敛，无法保证全局最优几何轨迹；在极端风格或大尺度迁移时仍可能出现身份漂移或细节损失。

---

## 539. Sim(3)-TurboReg: Robust and Efficient Scale-Aware Registration via Tri-Scale Consistency

**arXiv ID:** 2611.14135 | [PDF](https://arxiv.org/pdf/2611.14135v1)

**作者:** Zijun Zhou `[一作]` (Wuhan University), Jiayuan Li `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了Sim(3)-TurboReg注册框架，利用Tri-Scale Consistency（TSC）替代传统绝对距离约束，解决feed-forward 3D模型在滑动窗口分块时产生的尺度不确定性与尺度偏移问题。

**💡 创新点**

创新点在于将TurboReg的兼容图构建改为三尺度一致性判定，既能高效剔除匹配错误，又能抑制因动态或深度误差导致的坐标扭曲离群点，保持了鲁棒性与低延迟。

**🔧 技术方法**

采用TSC判定、节点/边尺度计算、二阶邻居支持加权、TurboClique高效三元组搜索、尺度感知退化检测以及Umeyama闭式估计。

**📊 数据集**

在KITTI Odometry和TUM RGB‑D两个公开数据集上进行实验，并在控制90%离群率的未知尺度注册实验中验证。

**📈 对比分析**

与RANSAC、TEASER++、IRLS等方法对比，Sim(3)-TurboReg在90%离群率下匹配RANSAC精度，且中位延迟降低92.5%；在DA3‑Streaming上平均ATE降低22.2%，端到端时间缩短16.6%。

**⚠️ 局限性**

局限性包括对k、τ等超参数敏感；在尺度极端变化或匹配质量低下时需回退到Umeyama；目前仅在滑动窗口分块框架验证，未针对无窗口连续流场景进行评估。

---

## 540. Hierarchical Graph Representation Learning with Pooling-Induced Substructures

**arXiv ID:** 2611.14126 | [PDF](https://arxiv.org/pdf/2611.14126v1)

**作者:** Luca Sbicego `[一作]` (EPFL), Dorina Thanou `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种将图池化产生的子结构显式保留并处理的PSA-GNN框架，解决传统池化在层次任务中信息丢失的问题。

**💡 创新点**

通过显式处理池化子结构提升表达能力，构造可控层次合成数据集与结构化指标，证明在WL层面更具表达力。

**🔧 技术方法**

采用结构基池化（Leiden、CurvPool、CliquePool）、PSA‑GNN（含子图编码与图级上下文交互）、GIN消息传递层、DeepSets读出等技术。

**📊 数据集**

使用两套新合成数据集Macro‑Hierarchies与Granular‑Hierarchies，以及ZINC、NCI1、REDDIT‑5K/12K、COLLAB、MalNet‑Tiny等真实图数据集。

**📈 对比分析**

与传统MPNN、Graph Transformer、学习式池化等基线对齐；在层次结构任务上PSA‑GNN显著优于即刻合并的池化模型，在非层次任务保持竞争力，整体提升明显。

**⚠️ 局限性**

仅在图表现出明显层次结构时有效，池化方案对任务敏感，且未验证检测到的层次是否与标签直接相关。

---

## 541. Right for the Wrong Reasons? Evaluating Plausibility and Behavioral Change in Explanation-Guided Learning

**arXiv ID:** 2611.14134 | [PDF](https://arxiv.org/pdf/2611.14134v1)

**作者:** David Dembinsky `[一作]` (German Research Center for Artificial Intelligence (DFKI) GmbH), Andreas Dengel `[通讯]` (German Research Center for Artificial Intelligence (DFKI) GmbH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个多轴评估框架，系统比较解释引导学习（EGL）在合成shortcut偏差下对模型鲁棒性、shortcut依赖、可解释性和可信度的影响。

**💡 创新点**

创新点在于：①构造可分离与全局性 shortcut 的定制化 cFMNIST 基准；②区分并对比归因约束（Invariance、Alignment）与依据 rationale 的数据增强（Sufficiency、Uncertainty）在行为与解释层面的差异；③以多维指标评估不同 EGL 目标的真实效果。

**🔧 技术方法**

使用 VisFIS 目标家族（Sufficiency、Uncertainty、Invariance、Alignment 及其组合）以及 Saliency 作为训练时解释信号；在六种后验解释方法（Saliency、SmoothGrad、Integrated Gradients、Occlusion、LIME、SHAP）上评估 Importance Mass、AUROC、IoU AUC、DDS 等指标。

**📊 数据集**

基准数据集为自制的 cFMNIST（Fashion‑MNIST 前景 + CIFAR‑10 背景），包含 patch、background、color 三种 shortcut；使用 ResNet18 与 Vision Transformer (ViT‑B/16) 两种模型架构。

**📈 对比分析**

对每种 EGL 目标与 shortcut 条件下，先挑选准确率最高的模型进行鲁棒性、依赖、可解释性与可信度评估；实验结果显示：Sufficiency 在可分离 shortcut 上提升鲁棒性与前景依赖最多（平均 +4.8pp 准确率，+前景依赖）；归因约束在可解释性上提升最显著（Importance Mass +14pp）；但不同目标的效果往往不一致，组合目标存在相互干扰，整体性能差异不一。

**⚠️ 局限性**

局限性：①基准过于人工，仅覆盖三类 shortcut；②仅采用 Saliency 作为训练时解释信号，未探讨其他指导信号；③组合目标缺乏联合调参，可能低估其潜力；④未在真实高风险领域（如医学图像）验证结果；⑤DDS 等可信度指标可能无法完全反映语义正确性。

---

## 542. Sckwoky at SHROOM-Visions: Character-Level Hallucination-Span Detection in Image-Conditioned VLMs via Dual-Prompted LoRA Decoders

**arXiv ID:** 2611.14100 | [PDF](https://arxiv.org/pdf/2611.14100v1)

**作者:** Mikhail Kulakov `[一作]` (Novosibirsk State University), Ivan Bondarenko `[通讯]` (Novosibirsk State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一种基于冻结Gemma-4-12B-it并使用LoRA适配的非生成式字符级幻觉检测系统，采用双重提示实现双向上下文，输出幻觉概率、BIO边界及类型。

**💡 创新点**

双重提示以回声方式为单向解码器提供双向上下文；将定位与类型分类解耦；在多语言上使用轻量级LoRA适配；验证模型对图像-答案配对的依赖性。

**🔧 技术方法**

Gemma-4-12B-it, LoRA, LayerMix, Conv1D+BiGRU, Tversky+ranking 损失, BIO tag decoder, 5 类幻觉类型分类器，非自回归推理。

**📊 数据集**

SHROOM‑visions 2026 训练/开发/测试集，包含英语、法语、意大利语和中文四语图文问答与字符级幻觉标注。

**📈 对比分析**

与官方基线对比，宏平均 IoU 从 0.1854 提升至 0.4404，Cor+Lbl 从 0.1854 提升至 0.3361，整体排名第五/第六，表明定位与校准均得到显著提升。

**⚠️ 局限性**

仅使用单一随机种子；决策阶段可能存在数据泄漏；类型分类器在预测跨度上存在分布漂移；中文最小跨度规则未针对 CJK 文字优化；实验未训练无图像模型，导致对图像依赖性评估有限。

---

## 543. Hx2Dx: From Anamnesis to Diagnosis - A Multimodal Arena for Conversational Clinical Reasoning in LLMs

**arXiv ID:** 2611.14122 | [PDF](https://arxiv.org/pdf/2611.14122v1)

**作者:** Federico Ravenda `[一作]` (Università della Svizzera italiana), Andrea Raballo `[通讯]` (Università della Svizzera italiana)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现了 Hx2Dx ARENA，一个包含 1,000 条 NEJM 多模态临床案例的交互式诊断基准，用来评估 LLM 在从初始问诊、体检到检验请求直至最终诊断的完整诊断流程中的表现。

**💡 创新点**

创新点在于将诊断拆分为信息获取与推理两大子过程，构建与完整案例对照的交互式评测框架，并通过对比揭示 LLM 在信息收集方面的显著瓶颈，推动从单纯推理向完整诊断流程的评估转变。

**🔧 技术方法**

采用多代理模拟（Doctor、Patient、Data、Examiner）实现 OSCE 风格对话，使用 GPT‑6 Luna、Gemini 3.5 Flash Lite、DeepSeek V4.1 Flash、GLM 5.3 Flash、Llama 4 Scout、Gemma 3 27B IT、Ministral 8B 等 LLM 进行实验。

**📊 数据集**

使用 1,000 条 NEJM 发表的真实临床案例，涵盖文本、表格、实验室数值、影像及病理图像等多模态证据，并将其拆解为病历脚本、检验登记表和参考诊断。

**📈 对比分析**

通过在交互式环境与静态完整案例环境下分别评估模型的诊断准确率，发现交互式情境下准确率平均下降 25–36%；但当将强信息采集者（GLM 5.3 Flash）的对话转移给其他模型时，弱模型可恢复 70–90% 的性能差距，显示信息获取是主导因素。

**⚠️ 局限性**

局限包括：仅涵盖能够口头交流的英文病例，缺乏多语言、多文化和非口头交互；部分 NEJM 案例可能已在模型训练中出现，导致结果偏差；且评估仅涉及文本对话，未涵盖更丰富的交互形式或多模态交互细节。

---

## 544. ProofCoder: Continuous Reasoning for Automated Software Verification

**arXiv ID:** 2611.14106 | [PDF](https://arxiv.org/pdf/2611.14106v1)

**作者:** Martim Afonso `[一作]` (INESC-ID & IST, University of Lisbon), João F. Ferreira `[通讯]` (INESC-ID & Faculty of Engineering, University of Porto)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于持续推理的 agentic Coq 证明合成系统 ProofCoder，能够在自动化软件验证中自主管理证明过程。

**💡 创新点**

通过持续推理与灵活的编辑‑修复循环，使代理在证明过程中能够在失败时定位、修复局部片段，而非单一的 tactic 生成循环，从而提升成功率并降低成本。

**🔧 技术方法**

采用大型语言模型（GPT‑4.1 / gpt‑6‑sol）结合 Coq 接口、OpenCode 框架，实现工具调用协议（查询、回溯、脚本编辑），并在 ProofCoder 内部实现连续推理与局部修复。

**📊 数据集**

在 SV‑COMP 2024 AutoRocq benchmark 的 200 条复杂验证 lemma 以及 Linux kernel Verker 80 条 lemma 上进行实验。

**📈 对比分析**

与 AutoRocq 及其它基准系统（Rango、PALM、COPRA 等）在相同预算下对比，ProofCoder 在 SV‑COMP 上成功率提升至 60%（比 AutoRocq 47% 提升 27.7%），在 Verker 上 60%（比 AutoRocq 47.5% 提升 12.5%），同时平均成本约为 AutoRocq 的一半。

**⚠️ 局限性**

依赖于大型语言模型的推理质量；对缺乏证明的 lemmas 无法保证；持续推理与回溯机制对不同证明助手可能需要重构；实验仅覆盖部分 benchmark，未验证跨工具普适性。

---

## 545. Wave-Domain Semantic ISAC: Holographic Surfaces as Analog Semantic Encoders for Integrated TN-NTN

**arXiv ID:** 2611.14094 | [PDF](https://arxiv.org/pdf/2611.14094v1)

**作者:** Shadi Danesh `[一作]` (University of Surrey), Mohsen Khalily `[通讯]` (University of Surrey)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出利用大型电磁开口（非局部超材料与全息表面）在近场进行波域语义投影，实现在数字化前完成线性语义特征提取，从而将语义通信与近场感知、集成感知通信（ISAC）统一到单一硬件平台；

**💡 创新点**

创新点在于：① 将大尺寸电磁开口作为硬件级线性投影机，实现近场语义编码和位置相关感知；② 通过数据自适应PCA投影与轻量化数字解码器的协同设计，在能量受限场景下获得4倍更少的传输系数和11.4倍更少的MAC运算；③ 将语义通信、近场操作与ISAC三者融合为一体化架构，并映射至3GPP/ITU-R/ETSI标准体系。

**🔧 技术方法**

使用技术包括非局部超材料（空间频率滤波）、全息表面（线性投影）、近场聚焦（位置相关特征）、数据自适应PCA投影、DCT基准投影、轻量化全连接网络解码器、能量限制评估等。

**📊 数据集**

实验采用scikit‑learn手写数字（8×8）图像数据集，作为抽象的空间观测输入。

**📈 对比分析**

比较方法：将三种方案——A（完整数字化、64维输入）、B（全息前端+PCA投影、16维）和C（非局部滤波+DCT投影、16维）在相同平均传输能量下，评估任务识别准确率随信噪比变化。结果显示，方案B在能量受限区间（10–20 dB）优于方案A，准确率提升约10%且仅传输四分之一系数；在高信噪下三者收敛。

**⚠️ 局限性**

局限性包括：① 现实硬件的插入损耗、色散、量化误差及校准误差未在实验中考虑；② 仅在理想AWGN通道下评估，未覆盖多径、相位漂移、Doppler等近场TN‑NTN特性；③ 需要更精细的自适应与多任务调度机制；④ 评价指标与标准尚未统一，需进一步制定语义KPIs与安全验证框架。

---

## 546. DexSeed: Expanding Human Hand-Object Interactions into Robot Demonstrations for Dexterous Policy Learning

**arXiv ID:** 2611.14092 | [PDF](https://arxiv.org/pdf/2611.14092v1)

**作者:** Huaijin Pi `[一作]` (University of Hong Kong), Taku Komura `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究如何将人类手物交互转化为可执行的机器人演示，支持基于视觉-语言-动作（VLA）的手部操控学习。

**💡 创新点**

创新点在于：①按抓取模式组织人类交互并给出语言描述；②基于 InteractionMesh 的跨体型与跨物体保留局部关系的重定向；③结合物理仿真与强化学习的闭环指尖控制，让机器人在跟随预定臂轨迹的同时实现真实接触；④自动化生成多样化、可执行的演示并用于 VLA 训练。

**🔧 技术方法**

使用 InteractionMesh 进行跨体型重定向；Diff3F 进行跨物体特征匹配；IsaacLab 强化学习（MLP 策略）做闭环物理模仿；3D Gaussian Splatting 重建真实场景并渲染观察；LingBot‑VLA 预训练模型做下游 VLA 微调。

**📊 数据集**

GRAB 人类手物交互数据集；选取 8 个代表性抓取模式；对 17 类对象生成 2000+ 机器人演示。

**📈 对比分析**

与 Dex‑Retargeting、SPIDER、Wuji 官方重定向器对比，显著降低指尖接触误差、角度误差和穿透深度；与 ManipTrans、Dexplore 的物理模仿对比，取得更低的跟踪误差、较高的接触召回率和成功率；VLA 训练在模拟与真实机器人上均获得比仅使用运动参考或其他模仿数据集更高的任务成功率，尤其在未见几何、布局、外观的情况下。

**⚠️ 局限性**

仅针对基础抓取和搬运任务；未考虑手内操作、双手协作；对不同手指结构的迁移研究不足；缺少触觉感知，无法实时检测滑动。

---

## 547. CURRENT: A Recurrent Neural Network for Concurrent Spatial and Temporal Super-Resolution of 4D Flow MRI

**arXiv ID:** 2611.14058 | [PDF](https://arxiv.org/pdf/2611.14058v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 548. What a CAGE 4 Score Measures: An Audit of a Learned Cyber-Defense Benchmark, Its Reference Defenders, and the Tables Built on Them

**arXiv ID:** 2611.14068 | [PDF](https://arxiv.org/pdf/2611.14068v1)

**作者:** James Rice `[一作]`, Andres Kwasinski `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究并揭示CAGE Challenge 4基准在评估学习型网络防御者时存在的缺陷与误导，重点分析了模拟器的“静默丢弃”机制、奖励计分不一致、以及训练与评估过程中的随机性和缺失信息；通过重现参考防御者（KEEP与H‑MARL Expert）并使用自定义仪器，量化了丢弃对得分的影响、评价随机性对平均奖励的贡献、以及训练过程中的未声明设计选择。

**💡 创新点**

创新点在于：1) 系统性发现并量化了CAGE 4基准中的关键缺陷（静默丢弃、奖励计分隐藏等）；2) 设计并实现了完整的评估仪器和可复现的实验环境；3) 引入“停顿式交付”和“分布式评估”两种新规则，并通过严格的统计分解揭示奖励噪声来源；4) 提出对比基准时需披露的五项信息清单，促进更透明的结果发布。

**🔧 技术方法**

采用了基于CybORG的网络安全仿真器、强化学习框架RLlib、Python 3、Docker容器化环境；在技术实现上使用了自定义记录器捕获每步执行的指令与奖励、随机种子重播、并行环境构造、以及针对KEEP和H‑MARL的专门包装器。

**📊 数据集**

数据集：CAGE 4 challenge v4的官方网络拓扑与脚本化红方攻击（stock与impact‑focused Wrecker两种敌手），并在固定的环境种子与策略种子上多次对比实验。

**📈 对比分析**

通过在同一环境种子下对比两位参考防御者的得分差异、记录每步奖励与动作、以及对不同交付规则（fired、held、queue‑first）的性能差异进行配对统计（95%置信区间、符号检验），验证了静默丢弃导致的分数误差、随机性对平均奖励的贡献（大约20%）以及训练时未声明的busy‑check对性能的显著影响。

**⚠️ 局限性**

局限性包括：1) 只关注Cage 4基准，未验证在其他仿真器或真实环境中的普适性；2) 评估主要基于自动化脚本，缺乏人类专家的主观判断；3) 虽量化了奖励噪声，但未对恢复精度的多重标准进行全面对比；4) 修复后的训练程序仍可能包含未识别的细节差异，导致结果难以完全复现。

---

## 549. BODYinTRANSIT-Lab: A Multimodal Dataset for Investigating Sensory-Driven Transformations in Body Perception and Affective Experiences

**arXiv ID:** 2611.14093 | [PDF](https://arxiv.org/pdf/2611.14093v1)

**作者:** Mohammad Mahdi Dehshibi `[一作]` (University of the West of England), Ana Tajadura-Jiménez `[通讯]` (Universidad Carlos III de Madrid)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究团队基于Footsteps Illusion，构建了BODYinTRANSIT-Lab多模态数据集，记录了101名参与者在三种声学反馈（控制、高频、低频）下的步态运动学、心电、皮肤电、肌电、呼吸等生理信号，并同步收集了多尺度主观评估（即刻体重感受、试后情绪、身体感觉问卷、avatar体型估计和身体感知地图）以及参与者的个体特征。

**💡 创新点**

创新点包括：① 通过实时音频反馈实现可控的身体感知转化实验；② 设计兼顾实验操控、主观体验与客观测量的注释方案；③ 采用系统化分层招募（饮食障碍症状与运动量）以捕获个体差异；④ 文档化完整的采集、同步、预处理与数据组织流程，方便复制与扩展。

**🔧 技术方法**

技术手段涵盖：SoniWeight鞋面向步态的实时频谱调制；Rokoko SmartSuit 3D运动捕捉；BITalino 8通道生理信号采集；基于Node‑RED的多设备同步；信号预处理（滤波、Hampel去噪、归一化）；深度学习基准模型——利用预训练的运动表示网络与注意力聚合器对加速度序列进行分类。

**📊 数据集**

使用的数据集是自研的BODYinTRANSIT-Lab（101名参与者、约16小时、15 GB），并将预训练模型的运动表示从英国Biobank 100k加速度数据（5 s窗口、30 Hz）迁移至本实验。

**📈 对比分析**

比较方法：在每个参与者内部采用10次随机种子交叉验证，按段（一次往返）划分为训练、验证、测试（60/20/20），并在每个窗口级别评估性能。三项任务分别为（a）声学条件分类、（b）相对体重感知分类、（c）相对情绪价值分类。得到的平均平衡准确率为：声学条件 36.6 %（对照三类基准33.3 %），相对体重 44.5 %（±11.5 %），相对情绪价值 45.9 %（±12.7 %）。

**⚠️ 局限性**

局限性：① 评估仅在同一实验会话内（未验证跨会话或跨受试者推广）；② 主观标签基于两次控制试验的均值，缺乏连续时间标注；③ 数据主要来自西班牙18–70岁年轻成人，缺乏文化、年龄或病理多样性；④ 仅考察短期音频干预，未探讨长期效应；⑤ 训练集与测试集可能共享同一试验段，影响真正的泛化评估。

---

## 550. Isolation Is Not Confidentiality: Security Boundaries of Geometric Access Control in RAG

**arXiv ID:** 2611.14061 | [PDF](https://arxiv.org/pdf/2611.14061v1)

**作者:** Steve Nouyep `[一作]` (Universite Clermont Auvergne), Sébastien Salva `[通讯]` (Universite Clermont Auvergne)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对检索增强生成（RAG）中基于几何变换的检索保持授权控制方法进行统一安全分析，评估其在多策略环境下的策略隔离、表示机密性、查询伪造抗性以及策略成员隐藏四个安全属性。

**💡 创新点**

① 将策略隔离与其他安全目标区分开来，指出后者并非由后者隐含；② 构建几何变换的组件模型，统一分析旋转、锚点、噪声与非线性等机制；③ 在已有公开实现基础上进行大规模实验，首次量化对齐对称样本对恢复与伪造的学习成本；④ 明确展示几何隔离能有效防止跨策略检索，却无法阻止表示恢复、伪造和成员泄露。

**🔧 技术方法**

几何变换的组件模型（正交旋转、锚点、噪声、非线性映射）
最小二乘/正交Procrustes估计变换
查询伪造与表示恢复攻击
无监督聚类(AR = Adjusted Rand Index)评估成员泄露
使用MiniLM、MPNet、GTE-large三种Encoder；
采用FAISS等向量检索库进行实验。

**📊 数据集**

• MedQuAD ABAC样本（9策略、3600文档）用于策略隔离评估。
• MS MARCO、BEIR（NFCorpus、SciFact、FiQA）用于检索效果评估。
• MedQuAD目标策略子集（3500嵌入、300查询）用于表示恢复与查询伪造实验。

**📈 对比分析**

与未加保护的基线做对比；指标包括 Recall@10、nDCG@10、QVR（查询违规率）、UCR（未经授权文档比例）、ARI（成员聚类质量）。
实验结果：
• 所有几何机制实现零跨策略检索（QVR=UCR≈0），检索效果与基线差异≤0.01。
• 表示恢复在200–400个对齐样本内即可达到80% top‑10重叠；
• 查询伪造成本随维度升高（300→1800对齐样本），但仍能在80%+重叠。
• 成员隐藏失效：ARI≈1.0（几何机制可完全区分策略）。

**⚠️ 局限性**

• 假设攻击者可获取目标策略的对齐样本（比真实环境更强的攻击面）。
• 评估仅覆盖三种公开实现及其串联，未涵盖所有搜索保持构造。 
• 正式隔离分析使用 g=id，未对非线性和噪声做完整解析。 
• 未考虑可信服务器被破坏、重键或动态策略切换等情形。

---

## 551. Software Refactoring Research with Large Language Models: A Systematic Literature Review

**arXiv ID:** 2611.14027 | [PDF](https://arxiv.org/pdf/2611.14027v1)

**作者:** Sofia Martinez `[一作]` (Stevens Institute of Technology), Eman Abdullah AlOmar `[通讯]` (Stevens Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统文献综述（SLR）对2022-2025年间50篇关于大型语言模型（LLM）驱动的代码重构研究进行了归纳、分类与评估，系统梳理了使用的工具、数据集、提示工程技术、评估指标以及当前面临的挑战。

**💡 创新点**

创新点在于首次提供一份完整的LLM重构技术生态图谱，明确指出缺乏统一的准确性度量、数据泄漏防护与多语言评估标准，并提出改进方向；同时揭示不同提示策略（如One-Shot、Few-Shot、Context-Specific、Chain-of-Thought等）对重构质量的显著影响。

**🔧 技术方法**

方法主要包括：检索八大数据库、筛选符合RQs的论文、使用质量评估量表评估原始研究、构建属性表（工具、模型、语言、提示方式、数据集、评估结果），并对收集的数据进行定量统计与定性对比；同时对评估结果中出现的准确率、召回率、F1分数等指标进行归纳。

**📊 数据集**

研究主要利用公开开源项目（GitHub、Apache、F-Droid）以及部分教育与生成式数据集作为评估样本，覆盖Java、Python、JavaScript、C++等语言；在本文中并未直接创建新数据集，而是对已有50篇论文中引用或自行构造的多样化数据集进行汇总与分析。

**📈 对比分析**

本文通过对比不同工具（如JDeodorant、JMove、EM-Assist等）、不同提示技术和不同LLM（ChatGPT、CodeT5、CodeGPT、Llama 2等）在准确率、召回率、F1等指标上的表现，发现One-Shot和Context-Specific提示往往在Java重构任务中表现最佳，Few-Shot在Python任务中更优；但由于缺乏统一评测基准和标准化指标，比较结果仍存在一定主观性。

**⚠️ 局限性**

局限性包括：评估方法缺乏统一标准，导致准确率等指标在不同研究间难以直接对比；大部分研究未对数据泄漏做明确防护，存在潜在偏差；主要聚焦Java与Python，其他语言研究不足；提示工程与模型选择高度依赖实验设置，泛化能力有待验证；整体缺少跨工具、跨语言的系统性对比与长周期效果评估。

---

## 552. Ontology-Guided Knowledge Graph Construction and Robust Evidence Retrieval for Indian Knowledge Systems

**arXiv ID:** 2611.14050 | [PDF](https://arxiv.org/pdf/2611.14050v1)

**作者:** Anand Kaniyaar Sriranga Prasad `[一作]` (HTSR Institute), Mohan Raghavan `[通讯]` (Indian Institute of Technology Hyderabad)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于印度知识体系文本的本体驱动知识图谱，并用它实现检索前置的证据候选生成；

**💡 创新点**

首次将本体约束与大语言模型联合用于多向量实体与关系抽取，形成图结构检索与传统单向量检索互补的复合方案；

**🔧 技术方法**

采用Qwen3-32B进行两轮抽取、MiniLM-L6-v2与BGE-M3进行实体归一、关系向量嵌入，利用Neo4j存储图并结合递归分块、关系向量搜索、相对倒排融合等技术；

**📊 数据集**

使用Dhara‑Bench v3 320题的印度古典文献语料（包括史诗、经文、医学、哲学、建筑与表演理论等十二类）作为评测数据集；

**📈 对比分析**

通过固定候选数的内测，对KG、单向量（SV）、KG+SV联合、KG+SV+实体笔记（EN）以及外部Gemini进行对比，KG在Counting/Composite/Multi‑hop题目上 MRR@20 与 Precision@5 最优；联合方案在 Recall@20 与完整证据 CE‑MRR@20 上表现更佳，Qwen reranker 在整体指标上高于 BGE，但推理延迟约 7 倍；

**⚠️ 局限性**

主要限制包括抽取与实体归一的误差导致图事实不确定、缺乏外部验证的三元组质量评估、KG检索延迟高、未实现多类别路由与动态预算、实验仅覆盖固定问答集且不保证在外域的泛化。

---

## 553. LaST0.5: A Generalist World Action Model with Interaction-Centric Latent Reasoning

**arXiv ID:** 2611.14048 | [PDF](https://arxiv.org/pdf/2611.14048v1)

**作者:** Xuheng Zhang `[一作]` (Peking University), Shanghang Zhang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了LaST_0.5，一种融合世界模型、交互式潜在推理和自适应世界到动作传输的通用机器人操纵框架。

**💡 创新点**

创新点包括：①引入交互式潜在推理（ICLR），把全局场景表示转化为局部机器人‑对象交互的紧凑潜在空间；②提出基于进度一致性的Adaptive World‑to‑Action Transfer（AWAT），动态截断动作执行并触发未来重采样；③利用超过14k小时的异构机器人演示大规模预训练，显著提升零样本泛化。

**🔧 技术方法**

技术细节：Mixture‑of‑Transformers架构整合世界专家（Cosmos‑Predict2.5‑2B）、潜在专家（Janus‑based）与动作专家；视觉提示结合运动曲线与对象框作为潜在监督；进度估计通过轻量MLP值头实现；训练采用流匹配损失、余弦相似度及进度回归等。

**📊 数据集**

数据集：14k小时异构机器人演示（遥控、手持、头摄像头等）；Libero四个官方套件（Spatial、Object、Goal、Long）；RLBench十个仿真任务；真实双臂任务（Bag Pulling、Object Organization、Screw Fastening、Parcel‑Box Packing）。

**📈 对比分析**

与多种VLAs（OpenVLA‑OFT、π_0.5、HybridVLA、InternVLA）和WAMs（Cosmos Policy、Motus、Fast‑WAM）对比；在Libero平均成功率99.5%，在RLBench平均成功率88.0%；AWAT提升约3%；在真实任务中平均成功率63.8%，显著高于π_0.5（52.5%）和Cosmos Policy（43.8%）。

**⚠️ 局限性**

局限性：仍需海量预训练数据；对世界模型预测误差敏感，尤其在长序列和极端动态场景下；进度估计受分段粒度限制；在强遮挡或复杂视觉条件下性能可能下降；依赖多视角摄像头和精确时间同步。

---

## 554. Criticality-Aware RAN Function Placement for Robust 6G Networks

**arXiv ID:** 2611.14121 | [PDF](https://arxiv.org/pdf/2611.14121v1)

**作者:** Gabriel M. Almeida `[一作]` (Universidade Federal de Goiás), Kleber V. Cardoso `[通讯]` (Universidade Federal de Goiás)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种基于云位置关键性的 RAN 功能放置方法，以提升分散化移动网络在随机失效和定向攻击下的鲁棒性。

**💡 创新点**

创新点在于首次将云位置关键性纳入放置优化，通过 ε‑约束同时最小化功能链依赖和部署成本，实现不依赖冗余即可提升网络鲁棒性。

**🔧 技术方法**

使用整数线性规划（MILP）模型，结合处理、传输、延迟与容量约束，利用 CPLEX 求解；同时通过关键性阈值 ε 对云位置重要性进行控制。

**📊 数据集**

采用 5G‑Crosshaul 项目真实网络拓扑（50 个 RU 与相连云点）以及 7 天每小时的真实流量负载数据集。

**📈 对比分析**

与三种基准（D‑RAN、PlaceRAN、Reveal）比较，随机失效下鲁棒性提升最多 20%–18%，定向攻击下提升至 75%，且仅增加约 10.7% 的活跃云点数量。

**⚠️ 局限性**

局限性包括：需手动设定关键性阈值 ε，模型规模较大时求解复杂度高；未考虑恢复过程与传输链路失效对鲁棒性的影响。

---

## 555. MISO-BR: An Open Corpus for Misogyny Detection in Brazilian Portuguese

**arXiv ID:** 2611.14036 | [PDF](https://arxiv.org/pdf/2611.14036v1)

**作者:** Fábio A. S. Passos `[一作]`, Ana Paula Cavalcanti `[通讯]` (UFRPE)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个公开、可追溯、版本化的巴西葡语厌女主义检测语料库（896条文本），并提供完整采集、预处理、人工标注与评测流程；

**💡 创新点**

创新点在于全流程可审计的公开资源、基于大型语言模型的自动预筛选、明确的三层标注指南与来源追溯字段、以及对样本平衡与偏差的量化评估；

**🔧 技术方法**

采用了Python爬虫+lexicon-guided采集、LLM（BERTimbau）三步预筛选、三位人工标注、基于Bag‑of‑Words、TF‑IDF的经典模型（NB、SVM、随机森林）和Fine‑tuned BERTimbau Base/Large；

**📊 数据集**

使用自采集自X的微博式评论与从OLID‑BR引入的文本，人工标注后按50/50比例划分为训练/测试集；

**📈 对比分析**

与经典模型（macro‑F1 0.651–0.676）相比，Transformer取得macro‑F1 0.833，Precision/Recall分别提升至0.841/0.822，Base与Large表现相当但Large耗时2.6倍；

**⚠️ 局限性**

局限性包括采样偏差导致的真实出现率偏高、预筛选缺乏误报率评估、标注者人数有限、样本量小、类别平衡不符合自然流、以及来源与标签相关性可能导致模型学习到非厌女特征。

---

## 556. TAMEing the Open-World Personalization: Towards Open-Set Personalized MLLM Assistant

**arXiv ID:** 2611.14118 | [PDF](https://arxiv.org/pdf/2611.14118v1)

**作者:** Rongpei Hong `[一作]` (University of Electronic Science and Technology of China), Fan Zhou `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种训练无关的概念调和技能（CR Skill），实现多模态大型语言模型在长上下文对话中的开放集个性化，能够在持续交互中自动识别、解锁新概念并持续细化已有概念。

**💡 创新点**

创新点在于：① 将个性化视为开放集问题并设计时间感知动态记忆（TAD Memory）；② 用时序加权相似度检索候选概念并交给LLM做最终决策；③ 引入周期性概念融合策略（PCF）抑制概念分裂；④ 所有机制均为无训练、可插拔，能直接集成至现有模型。

**🔧 技术方法**

主要技术包括：无训练的检索-决策框架、时序加权余弦相似度、时间标记的三元组记忆、LLM提示式概念决策与融合、以及周期性相似度聚类融合；使用的模型为Qwen3.5-9B（亦在Qwen3.5-70B验证），并与TAME等基线兼容。

**📊 数据集**

数据集：使用现有的LCMP‑H（30个概念）和自构造的LCMP‑O（45个视觉相似度更高的概念）进行评测；每个概念配有历史对话、图像和硬核问答。

**📈 对比分析**

对比了基线（Backbone、MyVLM、Yo’LLaVA、RAP、PeKit、R2P、TAME^†）以及对TAME的改进版本；在ACC‑C、ACC‑F、SPR‑L、SPR‑S和新引入的CRA指标上均实现显著提升（例如CRA提升至70%+，相较TAME^†提升约30%），并在概念混淆度高、概念出现顺序变动的情况下保持稳健。

**⚠️ 局限性**

局限性包括：仍需手动调节阈值或时间衰减参数，面对极其相似或多变属性的概念可能导致分裂误判；CR Skill依赖LLM的提示工程，不同LLM可能表现不一致；在大规模部署时，记忆检索和融合的计算开销尚待进一步优化。

---

## 557. Mobility-Aware Task Offloading in Mobile Edge Computing-Empowered Space-Air-Ground Integrated Internet-of-Things Networks

**arXiv ID:** 2611.14102 | [PDF](https://arxiv.org/pdf/2611.14102v1)

**作者:** Minghao Sun `[一作]` (University of Sheffield), Zitian Zhang `[通讯]` (Zhejiang Gongshang University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文针对MEC赋能的空间-空-地集成物联网网络中的移动ID和时变LEO卫星服务窗口，提出了联合优化UAV轨迹、任务拆分与计算资源分配的能量最小化问题，并给出了分解求解框架。

**💡 创新点**

创新点包括：1）将检索增强生成（RAG）与大语言模型结合，用于生成能量导向的UAV轨迹；2）利用RAG检索得到的历史案例与能量加权分数指导DDPG探索，显著提升学习效率；3）证明任务拆分子问题为凸优化，可用CVXPY直接求解；4）整体采用RLD框架实现动态环境下的能量高效决策。

**🔧 技术方法**

使用的技术包括大语言模型（GPT‑5）、检索增强生成（RAG）、深度确定策略梯度（DDPG）强化学习、案例库检索、凸优化求解（CVXPY）、能量加权检索机制。

**📊 数据集**

数据集为仿真生成的：20组基于街道约束的移动ID轨迹用于训练，3组未见过的测试场景用于评估；所有ID、卫星轨迹与通信参数均在仿真中随机生成，无使用公开数据集。

**📈 对比分析**

通过与四种基准（直线、随机、质心、DDPG‑only）以及遗传算法和模拟退火算法的对比，RLD在所有测试场景和不同ID密度下均能完成任务，能量消耗比基准低18%–29%；在线决策时延仅比DDPG略高，远低于传统搜索方法。

**⚠️ 局限性**

局限性包括：仅验证单UAV场景，未考虑多UAV或多链路情况；假设卫星服务窗口已知且高度固定；检索机制仅考虑历史能量与相似度，可能忽略更细致的动态特征；实际部署中对真实信道和移动模式的适应性尚待验证。

---

## 558. CUVE: Making Supervised Classification more Accessible through Visualization

**arXiv ID:** 2611.14098 | [PDF](https://arxiv.org/pdf/2611.14098v1)

**作者:** Sanchitha Kuppusami `[一作]` (University of Victoria), Miguel Nacenta `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出CUVE界面，通过可视化生成中间表示实现无代码监督分类。

**💡 创新点**

创新点是将可视化设计、模型训练和结果对比集成到单一Web UI，实现VLAIR的交互式无编程体验。

**🔧 技术方法**

采用可视化中间表示（VLAIR）技术，结合Web前端（React/Canvas）、机器学习后端（PyTorch）和交互式构造可视化（iVolver）等技术。

**📊 数据集**

使用公开的数值数据集（如水概率数据集）以及.xlsx、.csv格式文件。

**📈 对比分析**

通过用户研究对比CUVE与Jupyter Notebook，在SUS评分、功能可用性和模型评估指标上，非专家CUVE得分更高；整体模型性能（准确率、精确率等）相当。

**⚠️ 局限性**

局限包括对可视化设计经验的依赖、可重用性不足，以及需进一步验证在更大规模数据和更复杂模型上的适用性。

---

## 559. Opportunity-Cost Option-Critic: Adaptive Switching Costs for Robust Hierarchical Reinforcement Learning

**arXiv ID:** 2611.14099 | [PDF](https://arxiv.org/pdf/2611.14099v1)

**作者:** Khaled Ammoura `[一作]` (American University of Beirut), Naseem Daher `[通讯]` (American University of Beirut)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一种基于机会成本的选项延迟机制，用于改进层次强化学习中选项的持续性和适应性。

**💡 创新点**

创新点在于将当前选项与最优替代方案的价值差（正价值间隙）作为状态-选项相关的动态延迟成本，取代传统的固定或基于持续时间的惩罚。

**🔧 技术方法**

采用Option-Critic架构、成本自适应学习、价值间隙目标、Actor-Critic梯度更新以及深度神经网络实现成本预测。

**📊 数据集**

使用离散导航的Four Rooms环境和连续控制的Pinball环境，实验采用十个随机种子进行评估。

**📈 对比分析**

与固定成本、灵活成本和标准Option-Critic等基线比较，OC^2在Four Rooms获得更短轨迹和更高折扣回报，在Pinball目标变动后首10回合成功率提升至52%，整体回报最高，表现优于其它方法。

**⚠️ 局限性**

主要局限包括对价值估计准确性的高度依赖、成本学习跟不上环境变化的速度、以及所有选项在所有状态均可用的假设。

---

## 560. From Bias to Positionality: A critical grammar for Large Language Models

**arXiv ID:** 2611.14079 | [PDF](https://arxiv.org/pdf/2611.14079v1)

**作者:** Dave Murray-Rust `[一作]` (Delft University Of Technology), Maria Luce Lupetti `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过引入“定位性（positionality）”的概念，提出一种比传统“偏差（bias）”更适合描述大型语言模型（LLM）行为的分析框架，并构建了一个层叠的定位性堆栈模型，用以追踪模型在语料库、训练、系统提示、应用和交互等不同层面上的定位性来源与传递；

**💡 创新点**

创新点在于：①提出将定位性作为LLM评估的核心范式，替代单一的偏差视角；②构建层叠继承模型，揭示从创建者、意图到实施行为三层定位性如何相互影响；③识别并讨论了继承、破裂与崩溃等动态现象；④将女性主义认知学和计算机解释学相结合，为LLM语境提供跨学科方法；

**🔧 技术方法**

主要采用理论分析与概念阐释，引用女性主义认知学、视角理论、系统论与计算机解释学等学术框架；未涉及新的算法实现；

**📊 数据集**

未使用任何实验数据集，论文以文献综述、案例引用和概念分析为主；

**📈 对比分析**

本研究为理论性研究，未进行定量或定性的实验比较，也未报告性能指标；

**⚠️ 局限性**

局限性包括：①缺乏经验验证，定位性框架的可操作性和评估方法尚不成熟；②对具体LLM的定位性量化与测评缺乏标准；③在实际工程实践中如何落地、与现有公平评估工具衔接仍需进一步探讨。

---

## 561. Classical Tensor Codes and Quantum Tanner Codes Achieve Capacity on Erasure Channels

**arXiv ID:** 2611.14087 | [PDF](https://arxiv.org/pdf/2611.14087v1)

**作者:** Henry D. Pfister `[一作]` (Duke University), Gilles Zémor `[通讯]` (Université de Bordeaux)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一类基于量子 Tanner 码的码族，在经典擦除信道上实现容量近似，且提供线性时间的纠错算法。

**💡 创新点**

创新点在于证明：①任意两个在经典擦除信道上达到容量的码族的张量积仍然达到容量；②随机线性码的张量对偶码同样满足容量；③将这一结果应用于量子 Tanner 码，得到码率可接近 1-2ρ，且误码概率指数下降、解码复杂度线性。

**🔧 技术方法**

核心技术包括：张量码容量保持定理、对偶张量码的鲁棒性（product‑expansion）证明、随机线性码容量性质、量子 Tanner 码构造（四面体复形与本地内码）以及迭代线性时间解码算法。

**📊 数据集**

本文未使用公开数据集，而是构造理论上随机线性码和 Cayley 四面体复形作为实验基准。

**📈 对比分析**

与现有随机 QLDPC 码相比，本文方案在相同码率下实现指数级误码概率衰减，并提供线性时间解码，优于仅给出错误半径保证的先前工作。

**⚠️ 局限性**

局限性包括需要足够大的本地码维度 Δ、对随机码高概率假设的依赖，以及实现大规模随机码与硬件可扩展性的实际挑战。

---

## 562. UniFlow-RS: Unified Generative Prediction for Heterogeneous Remote-Sensing Tasks

**arXiv ID:** 2611.14281 | [PDF](https://arxiv.org/pdf/2611.14281v1)

**作者:** Jianlong He `[一作]`, Gui-Song Xia `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出 UniFlow‑RS，一种共享生成预测器的遥感感知框架，能够在同一模型中完成六个不同任务并覆盖十三个公开数据集。

**💡 创新点**

核心创新在于将异构任务输出映射到共用的视觉潜在空间，使用单一预训练的流匹配预测器，并通过任务‑模态自适应时间步采样（TMAS）解决不同任务的学习动态差异，实现真正意义上的共享预测过程。

**🔧 技术方法**

技术手段包括预训练的 MMDiT/FLUX.2 latent flow 生成器、冻结的 VAE 与 Qwen3 文本编码器、共享 LoRA 适配器、Flow Matching 损失以及 TMAS 机制。

**📊 数据集**

实验使用了 6 类任务（建筑提取、变更检测、语义分割、参考分割、旋转检测、地形高度估计）覆盖 13 个公开数据集，并在 5 个未见 OOD 数据集上进行零样本迁移评估。

**📈 对比分析**

对比多种统一与单任务基线，使用同一检查点进行评估，平均 Unified Task Score 在多数基准上达到或逼近 SOTA，并在未见数据集上展现出强劲的零样本性能，且训练步数显著降低。

**⚠️ 局限性**

局限性包括对极端遮挡或高分辨率细粒度实例恢复的不足、部分任务（如旋转检测）仍不及专门模型，且在更大规模数据与算力支持下仍有提升空间。

---

## 563. Semantic Chains: From Reachable Facts to Justified Answers

**arXiv ID:** 2611.14252 | [PDF](https://arxiv.org/pdf/2611.14252v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 564. Recurrent Modular Structure in Spiking Neural Network Reinforcement Learning

**arXiv ID:** 2611.14262 | [PDF](https://arxiv.org/pdf/2611.14262v1)

**作者:** Vikram Ramavarapu `[一作]` (University of Illinois), Ashish Gautam `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并验证了一种基于模块化的可学习脉冲神经网络架构CoRe SNN，利用局部三因素规则Map-STDP实现自组织模块，提升RL任务性能。

**💡 创新点**

创新点在于将网络划分为控制器、动作和潜在社区，实现了可学习的模块化递归结构，并通过Map-STDP将信息压缩与奖励学习相结合。

**🔧 技术方法**

采用三因素STDP（Map-STDP）、三种脉冲神经元模型（branching、LIF、QIF）、TD-STDP actor‑critic、随机游走与map方程等技术。

**📊 数据集**

使用OpenAI Gym的CartPole、Acrobot、LunarLander三种强化学习环境进行实验。

**📈 对比分析**

对比三种模块化程度（动态、部分、完整）与基线feed‑forward网络，结果显示动态/完整模式在CartPole和LunarLander上提升10–45点回报，压缩度与能效相关，整体性能与基线相当或更优。

**⚠️ 局限性**

局限包括：基于随机游走的抽象导致计算瓶颈，map方程仅为一种模块化驱动原则，潜在社区功能不足，规模化和并行性待改进。

---

## 565. From Prediction to Action: Learning Physical Context for Deformable Linear Object Manipulation

**arXiv ID:** 2611.14274 | [PDF](https://arxiv.org/pdf/2611.14274v1)

**作者:** Chengxiao Dong `[一作]` (University of Bologna), Gianluca Palli `[通讯]` (University of Bologna)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在可变形线性物体（DLO）单抓手形状控制中，探讨物理上下文对预测与决策效用的影响，提出 Geometry–Pointer–Flow (GPF) 策略并比较 Oracle、Estimated 与交互历史（Hist）三种上下文表示。

**💡 创新点**

① 区分预测效用与决策效用；② 设计将抓取位置与运动生成分离的 GPF 基础；③ 证明交互历史上下文在控制上优于显式物理参数，尤其在零射转移到真实机器人时仍保持优势。

**🔧 技术方法**

使用 Transformer+Pointer+Rectified Flow 的 GPF 网络；训练一个冻结的物理参数估计器；交互历史编码为 32 维潜在变量并可通过 FiLM 或 concat 注入；仿真平台 AGX Dynamics；零射转移实验。

**📊 数据集**

AGX Dynamics 生成 90,106 句子（901,060 转），包含 20–200 kPa 的杨氏模量和 0.15–0.60 的摩擦系数；真实机器人实验共 1,800 次转动，覆盖 6 种硬度/表面组合（Soft/Medium/Hard × Smooth/Rough）和 10 个目标形状。

**📈 对比分析**

在仿真单步和闭环评估中与 DLO‑DP、行为克隆等基线对比：GPF‑Geom 在单步误差上比 DLO‑DP 降低约 4.5 mm，参数量 18.7× 少，延迟 4.4 ms；在闭环 10 步内，Hist‑FiLM 在仿真 10 mm 阈值下达成率 69%，在硬件上 10 mm 达成率 88%，显著优于其他控制器。

**⚠️ 局限性**

无法完全解释预测与决策效用分离的根本原因；实验未将信息质量与集成方式拆分；未在真实世界训练或微调，零射转移效果受仿真与硬件差异影响；交互历史优势可能源于估计器在硬件上退化，尚需进一步因果分析。

---

## 566. Chain of Physics: Supporting Fault Diagnosis with Engineering-Inspired Large Language Model Reasoning

**arXiv ID:** 2611.14263 | [PDF](https://arxiv.org/pdf/2611.14263v1)

**作者:** Wenqi Zhou `[一作]` (Zurich University of Applied Science), Manuel Arias Chao `[通讯]` (Zurich University of Applied Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 Chain of Physics (CoP)，一种基于受限知识图 (CKG) 的 LLM 辅助故障诊断框架。

**💡 创新点**

创新点在于自动从技术文档构建受限知识图，并将完整的故障–物理变化–传感器链作为检索式示例和推理模板，显著提升 LLM 诊断推理的准确性。

**🔧 技术方法**

使用的大型语言模型（Claude Opus 4.7、Llama 3.1 8B、Qwen 2.5 7B）结合检索增强生成（RAG）、图检索、上下文学习和结构化链式推理。

**📊 数据集**

评估基于公开的 FailureSensorIQ 数据集，在四类工业资产（燃气涡轮、电机、风扇、变压器）上使用单一答案多项选择问题。

**📈 对比分析**

与直接提示和传统 RAG 对比，CoP 在所有模型上平均提升 7.8–19.6 个百分点，RAG 提升 11.7–18.2 个百分点；最优模型 Claude Opus 4.7 在单一答案任务中准确率从 52% 提升至 68.8%。

**⚠️ 局限性**

局限在于对文档质量和结构化抽取的依赖、对多模态或实时传感数据缺乏支持，以及在存在模糊或多重对应关系的题目中仍会出现错误。

---

## 567. SHARP: Semantic Hierarchical Agentic Reasoning for Next POI Prediction with Dynamic Memory

**arXiv ID:** 2611.14220 | [PDF](https://arxiv.org/pdf/2611.14220v1)

**作者:** Yuhang Zhan `[一作]` (University of Electronic Science and Technology of China), Shuo Shang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于代理层级推理的下一个兴趣点预测框架SHARP，融合类别级推理与语义化POI表示；

**💡 创新点**

创新点包括：①在层级架构中将类别预测作为中介，提升语义推理与具体POI预测的耦合；②引入动态轨迹记忆库以保持长周期用户行程信息并控制LLM上下文；③构建多粒度时间感知类别转移图，保留时段特定的转移规律；④通过RQ‑VAE学习语义ID，形成可与代理推理相结合的语义POI表征；

**🔧 技术方法**

使用的大模型为Llama3.1‑8B；核心技术包括：代理式层级推理、动态轨迹记忆更新、时间分辨率多层级转移图、RQ‑VAE语义ID编码、基于Transformer的POI预测器；

**📊 数据集**

实验数据集为Foursquare‑NYC、Foursquare‑TKY、Gowalla‑CA共三份真实定位记录；

**📈 对比分析**

与12种基线（含通用序列推荐、空间时间模型及LLM驱动模型）对比，SHARP在Acc@5/10、NDCG@5/10和MRR上分别提升约6%–21%，在NYC、TKY、CA三大数据集均名列前茅；

**⚠️ 局限性**

局限性包括：①对大模型的依赖导致计算成本较高；②动态记忆和多粒度图仍需要手工设定时间粒度和代码簿大小；③在极端稀疏或极大规模用户场景下的扩展性尚待验证；

---

## 568. Deviation-Aware Pretraining for Cardiovascular Disease Detection from Electrocardiograms

**arXiv ID:** 2611.14141 | [PDF](https://arxiv.org/pdf/2611.14141v1)

**作者:** Siyuan Yan `[一作]`, Wei Li `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

构建了基于自监督预训练的心电图异常检测框架，提出了Deviation Localization Embedding (DLE) 与 Association Reconstruction Embedding (ARE) 两种嵌入方式，并通过ScalesMamba2网络进行特征提取与融合。

**💡 创新点**

创新点在于：①结合DLE与ARE两种互补嵌入提升局部与跨导联的判别力；②提出Weighted Rotational Partial Freezing与对称掩码自监督策略，强化特征提取而抑制头部过拟合；③采用多尺度U‑Net结构的ScalesMamba2实现长时序建模，显著优于Transformer/CNN/RNN。

**🔧 技术方法**

主要技术包括：Mamba2（Selective State Space Model）+ ScalesMamba2；对称掩码自监督；Weighted Rotational Partial Freezing；多尺度DTW聚类与Soft‑DTW barycenter；线性评估与轻量化分类头；与SOTA自监督模型（DINOv2、TimesFM、Chronos、MOIRAI）对比。

**📊 数据集**

使用918,000条未标注12导联ECG（MIMIC‑IV‑ECG + 私有临床数据）进行预训练；外部验证数据集为PTB‑XL、ML12L、CPSC2020、WGEX；观察性数据集来自另一地区临床合作伙伴，包含ICD编码病例。

**📈 对比分析**

与传统线性评估、CNN/Transformer/Mamba分类器以及公开SOTA自监督模型（ASTCL、BeatGAN、MERL、ECGAD）进行比较。实验结果显示，DLE+ARE融合显著提升了Balanced Accuracy、F1、AUC，特别在CPSC2020、WGEX两大未见测试集上分别获得0.547/0.584的Balanced Accuracy、0.800/0.809的AUC，优于所有对比模型。

**⚠️ 局限性**

局限性包括：①未进行信号滤波或数据增强，模型对噪声鲁棒性仍需验证；②对不同人口/设备的泛化能力虽然提升，但仍可能存在分布漂移；③模型规模与训练成本较大，部署在资源受限的医疗终端仍有挑战；④缺乏可解释性机制，临床解释性不足。

---

## 569. Uneven Self-Reported AI Use: An Embedded Case Study Across Eleven Teams in a Software Organization

**arXiv ID:** 2611.14154 | [PDF](https://arxiv.org/pdf/2611.14154v1)

**作者:** Daniele Veri' `[一作]` `[通讯]`, Daniele Veri'

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对一家软件组织内部11个团队的自报AI使用频率、政策意识和工作实践进行描述性案例研究。

**💡 创新点**

在同一组织内部使用嵌套单案例设计，并对不同版本量表的项目兼容性和覆盖范围进行系统审查，明确说明量表差异导致比较限制。

**🔧 技术方法**

使用基于Tally的问卷（10个标准化表格+1个早期表格）进行访谈记录收集，并采用定量计数与交叉表分析以及访谈笔记的定性整理。

**📊 数据集**

收集了98条访谈记录（65条标准化问卷+33条早期问卷），包括10个团队的量表数据以及12条调研优先级记录。

**📈 对比分析**

主要方法是对频率得分≥4/5的比例进行团队级交叉比较，并与政策意识交叉表格展示相关性；但因量表不等价且样本非概率，无法提供统计显著性或效能评估。

**⚠️ 局限性**

局限性包括自我报告偏差、量表不可比的语言锚点、非概率样本、缺乏行为验证、定性数据缺少系统编码、样本规模小、时间与工具版本混淆以及研究者身份可能影响回答。

---

## 570. Verifiable Agentic Traffic Class: A Policed Quality-of-Service Mechanism for Agentic AI Traffic, Evaluated from Simulation to Real-Kernel Adversarial Testing

**arXiv ID:** 2611.14211 | [PDF](https://arxiv.org/pdf/2611.14211v1)

**作者:** Naveen Babu Narayanan Nair `[一作]` (Comcast Corporation), Sajay Kurup `[通讯]` (Comcast Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可验证的代理AI流量类（VATC），在边缘路由器上对声称为代理的流量进行DSCP标记、行为验证和速率限流，以实现低延迟且防止优先级盗用。

**💡 创新点**

创新点在于：①为代理AI流量专门定义一个可验证的QoS类，兼容现有DiffServ；②引入路由器侧行为验证引擎（利用流量持续时间、吞吐量、上下行比、IAT-CV和应用层跳数等特征）动态确认标记是否可信；③使用速率上限的优先队列和概率排队子池（probation lane）在验证完成前限制恶意流量对链路的占用。

**🔧 技术方法**

使用技术包括：DSCP编码与5QI风格的AQI映射；路由器数据平面（eBPF/TC）实现优先队列与速率限制；IPFIX/DTLS传输的跳数元数据；机器学习（逻辑回归/随机森林）用于流量行为分类；Linux HTB与FQ-CoDel等现成队列策略作为基线；Python+NumPy等仿真框架。

**📊 数据集**

数据集：基于Cisco 2026 WAN AI推理流量报告统计的合成流量生成器（包括人类、批量、代理、欺骗五类流量）；两套真实代理端到端测试环境（mini-swe-agent、OpenAI客户端+MCP SDK）产生的TCP流量；以及对IPFIX记录、队列计数等实验数据的收集。

**📈 对比分析**

比较方法：在自定义离散时间仿真器、第二个事件驱动仿真器以及真实Linux内核（HTB+TC）上分别对VATC、静态DiffServ、FQ-CoDel、以及无标记的最小队列进行性能对比；指标包括平均/95%分位排队延迟、完成率、验证器准确率、误判率。结果显示：在稳定负载下VATC与DiffServ相当；在构造的代理突发场景下VATC平均代理排队延迟下降98%（相对DiffServ），人类流量仅增加12%；在伪造洪水测试中速率限流将人类延迟降低至DiffServ的1/11；在真实内核下，VATC与DiffServ在诚实负载下无明显差异，但在存在伪造流量时能显著降低代理延迟。

**⚠️ 局限性**

局限性：①流量完全合成，缺乏真实代理AI流量的实测数据；②跳数特征高度依赖可信网关，若伪造可能削弱验证效果；③验证器在完整流量上表现良好，半途流量准确率下降；④实验未覆盖跨域DSCP漂移、核心路由器部署、或多AQI等级；⑤构造的突发场景未经过现场校准；⑥仅在低速链路（20‑200 Mbit/s）上验证，未评估高速核心链路；⑦未对适应性攻击者进行动态演化测试。

---

## 571. HaiRuo: Forecasting Ocean Eddies as Fields and Objects with a Satellite-Altimetry Foundation Model

**arXiv ID:** 2611.14136 | [PDF](https://arxiv.org/pdf/2611.14136v1)

**作者:** Xukun Qin `[一作]` (Ocean University of China), Ge Chen `[通讯]` (Ocean University of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种同时预测海洋中尺度环流涡的场景（海平面异常、地转速度等）和涡的轨迹的统一框架，利用海平面高度卫星数据。

**💡 创新点**

创新点包括：①使用1.05 B参数的无标签预训练“海洋基础模型”，能在不使用涡标签的情况下学习全海域动力学；②设计了“Lagrangian对象注意力”解码器，在涡的共移动极坐标系中对预测场进行采样与多轮校正；③在预训练、对象解码器与全局评估三方面实现显著性能提升。

**🔧 技术方法**

技术手段包括：自监督预训练（未来预测、掩码重建、流场对比），多尺度球面注意力块，双速时间记忆压缩，四专家多轮对象校正，及指数移动平均微调。

**📊 数据集**

数据集：1993‑2016年全球DUACS卫星海平面高度数据用于预训练与标注训练，2017‑2019年Dev48和2020‑2022年Test48作为验证与测试；涡标签来源于META4.0卫星涡目录。

**📈 对比分析**

在预注册的全球评估（114,546个涡）中，模型在15天中心误差（D15）上比最强基线下降18%（从30.6 km降至25.2 km），在场景精度、掩码mIoU和物理一致性上也均超过全部14种基线方法。

**⚠️ 局限性**

局限性：仅利用海平面高度观测，缺乏多源数据融合；对极地或浅海区域的泛化仍有限；模型规模大、训练成本高，实际部署需进一步优化。

---

## 572. A Cross-Scenario Survey and Analysis of Self-Evolving Multimodal Agents

**arXiv ID:** 2611.14256 | [PDF](https://arxiv.org/pdf/2611.14256v1)

**作者:** Jingjing Jiang `[一作]` `[通讯]`, Jingjing Jiang

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述并系统分析了跨四大场景（多模态推理、多模态生成、GUI/网页导航、具身操作）中自演进多模态智能体的自我提升机制，阐明了机制转移的结构性不对称性。

**💡 创新点**

创新点在于提出“自演进循环”的三阶段框架、基于“可执行度”和“反馈可信度”两条任务外上限构建定位矩阵、将各机制视作资源与稀缺因子权衡的可移植性判定，并给出可验证的度量议程。

**🔧 技术方法**

采用理论形式化（自演进循环定义、不同ia、原产矩阵）、量化指标（经验生成、反馈获取、更新合并的效用分解）和比较矩阵，结合已有自演进方法的元操作归因，构建跨场景比较模型。

**📊 数据集**

作为综述性质，本文未使用单一实验数据集；其分析基于已公开的多模态任务集（如VQA、图像生成基准、WebArena、机器人仿真/真实任务）和对应的自演进实验结果。

**📈 对比分析**

方法比较以固定循环与自演进循环在同一场景下的性能差异为主，引用各研究报告的收益曲线、收敛性和对照实验，指出不同机制在相同预算下的提升幅度与瓶颈，但并未给出统一的数值性能基准。

**⚠️ 局限性**

局限性包括：跨场景转移的可移植性判定尚未通过实证验证；指标如保留率、泛化度量缺乏统一测量；不同方法的基线、超参数和硬件环境差异导致结果难以直接对比；模型背后的先验能力与场景约束交织，影响结论的普适性。

---

## 573. Elastic-WAM: Adaptive Spatiotemporal Imagination for Efficient Robot Control

**arXiv ID:** 2611.14257 | [PDF](https://arxiv.org/pdf/2611.14257v1)

**作者:** Haitian Li `[一作]` (National University of Singapore), Lin Shao `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一种 Elastic-WAM，利用统一的时空细化需求场对世界-动作模型进行自适应细化，以提升机器人控制性能并减少推理延迟。

**💡 创新点**

创新点在于构建时空细化需求场，联合决定何时停止世界细化（WHEN）、每步细化进度（HOW FAR）以及需要刷新哪些未来视频区域（WHERE），并通过三阶段动作监督学习该场，从而实现稀疏推理并显著降低计算量。

**🔧 技术方法**

技术包括基于扩散的世界-动作模型、Beta 分布细化曲线、共享时序基底与空间分组、动作监督的三阶段训练、直通（straight‑through）软硬掩码以及基于冻结 WAM 的稀疏递归推理。

**📊 数据集**

使用了 LIBERO、LIBERO‑Plus、LIBERO‑Pro 三个仿真基准、RoboCasa‑GR1 复杂操作数据集，以及在双臂机器人上进行的四阶段茶叶制作的真实世界实验。

**📈 对比分析**

与 Joint‑WAM、Fast‑WAM 等现有方法对比，Elastic‑WAM 在 LIBERO、LIBERO‑Plus、LIBERO‑Pro 和 RoboCasa‑GR1 上均取得最高成功率，并将推理延迟降低 32.2%；在真实世界长周期任务中，在正常、光照 OOD 与物体 OOD 条件下也实现了最高的任务完成率和进度分数。

**⚠️ 局限性**

局限性包括依赖冻结的 WAM 主干，可能在极端域外环境下需要进一步适配；对 token 刷新预算的设定仍具有经验性；在预算过大或过小的情况下性能可能不稳定；缺乏显式的任务进度记忆模块。

---

## 574. Unipolar Neuron Networks Trained by Recirculation for Reward-Based Continual Learning

**arXiv ID:** 2611.14153 | [PDF](https://arxiv.org/pdf/2611.14153v1)

**作者:** Florin Leon `[一作]` `[通讯]` (Gheorghe Asachi Technical University of Iasi), Florin Leon (Gheorghe Asachi Technical University of Iasi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于单极神经元的网络，通过双通道回路（recirculation）从奖励信号中学习；

**💡 创新点**

创新点在于将单极神经元的树状结构、分枝局部输出、全局输出与分枝抑制（task code）相结合，并引入基于自身历史的权重重要性调节（consolidation）实现无标签连续学习；

**🔧 技术方法**

使用了无回路神经元模型、双通道回路学习规则、任务代码抑制、重要性加速的自适应学习率、基于奖励的actor‑critic框架、以及可选的梯度回传或局部规则；

**📊 数据集**

在MNIST（按两数字一任务的五个任务序列）和Iris（三类逐步学习）上进行实验；

**📈 对比分析**

与梯度下降（含监督和奖励目标）以及之前的局部学习规则对比；结果显示recirculation在保持旧任务性能（低遗忘）与学习新任务（接近梯度法准确率）之间取得良好折衷；

**⚠️ 局限性**

限制包括：需要外部任务代码（无法处理class‑incremental场景）、保护依赖于读出层仅读取最后一层的假设、训练速度略慢、未处理多步延迟奖励、对结构可塑性的探索尚未实现。

---

## 575. An Empirical Study on CVE-based Vulnerabilities Propagation from one System to another System (IntraProject vs InterProject)

**arXiv ID:** 2611.14164 | [PDF](https://arxiv.org/pdf/2611.14164v1)

**作者:** Junaid Akram `[一作]` (University of Nottingham Ningbo, China), Danish Vasan `[通讯]` (King Fahd University of Petroleum & Mineral)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过构建基于补丁的漏洞签名和代码克隆检测框架SQVDT，对大量开源项目中的CVE漏洞进行大规模检测，并分析了漏洞在同一项目（IntraProject）和跨项目（InterProject）中的传播模式。

**💡 创新点**

创新点在于：①首次使用补丁文件提取的漏洞特征与代码克隆技术相结合，实现对CVE漏洞传播路径的定量识别；②对传播机制进行细分（代码克隆、第三方库、AI生成代码）并给出针对性治理建议；③通过经验研究展示了跨项目传播对现代软件生态的安全威胁。

**🔧 技术方法**

使用技术包括：SQVDT（可扩展定量漏洞检测技术）、爬虫抓取补丁与源代码、文本相似度/AST克隆检测、CVSS、CWE等指标统计与可视化。

**📊 数据集**

数据集：约1.87千个CVE补丁（来自NVD、git.kernel.org、GitHub等）共3.36M行代码；约150千个GitHub开源项目，7TB C/C++源代码。

**📈 对比分析**

实验与对比：在相同数据集上与VFDETECT、DECKARD等工具比较，SQVDT误报率约1%，覆盖7TB代码，发现12,000+漏洞并完成传播路径分析，验证了方法的可扩展性与高精度。

**⚠️ 局限性**

局限性：①采样仅覆盖部分GitHub项目，缺乏完整代表性；②CVE补丁覆盖面有限，可能漏检；③AI生成代码部分为探索性实验，缺乏大规模验证；④传播检测假设基于补丁文件，可能忽略上下文导致误判。

---

## 576. Lilo: A Multi-Agent Conversational Sandbox for Children's Digital Communication

**arXiv ID:** 2611.14260 | [PDF](https://arxiv.org/pdf/2611.14260v1)

**作者:** Ofri Elbaum `[一作]` (University of Haifa), Oshrat Ayalon `[通讯]` (University of Haifa)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个名为 Lilo 的多智能体对话系统，允许儿童在模拟的群聊冲突情境中练习数字沟通，提升其冲突调解与自我调节能力。

**💡 创新点**

创新点在于将大型语言模型生成的角色与 Stop‑Think‑Act 交通灯教学框架结合，提供实时、情境化的干预，并在多代理编排下实现动态难度适配；同时系统在不需要预先脚本的情况下支持自然语言互动。

**🔧 技术方法**

技术实现主要使用 Claude 3.7 Sonnet LLM，配合多代理架构（Sagi、Reut、Lilo、Orchestrator、Feedback Agent）和动态提示技术，辅以安全过滤和交通灯干预机制。

**📊 数据集**

数据集为内部构建的 Hebrew 语言预设冲突脚本和角色设定；实验数据来自 28 名 9‑11 岁儿童的交互日志、问卷与访谈，未使用公开大规模对话数据集。

**📈 对比分析**

通过单次实验与前后问卷的 Wilcoxon 检验，发现儿童在应对不愉快信息的自我效能显著提升；系统日志显示交通灯在不同难度等级下的分布，绿色最多；由于缺乏对照组，未能量化相对性能优势。

**⚠️ 局限性**

局限性包括单场实验、无对照组、仅 Hebrew 语言、样本同质化（高教育背景家长），场景固定（男孩拒绝女孩参加派对），以及未评估长期迁移效果或真实群聊中的行为改变。

---

## 577. Location Matters: Sparse Latent-Variance Attacks for Face Privacy against Diffusion Personalization

**arXiv ID:** 2611.14254 | [PDF](https://arxiv.org/pdf/2611.14254v1)

**作者:** Shuting Yang `[一作]` (National University of Singapore), See-Kiong Ng `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种稀疏扰动框架Sparse Encoder Attack on Latent Variance（SEALV），通过在面部图像上仅修改5%的像素来防止扩散模型的个性化复制。

**💡 创新点**

创新点在于将稀疏像素选择与VAE编码器的潜在方差扰动相结合，并引入基于显著性引导的分散正则化，既保持图像视觉质量又显著降低身份识别相似度。

**🔧 技术方法**

采用可微分Top‑k掩码、Laplace基潜在方差损失、直通估计（STE）与显著性分散惩罚，并在Stable Diffusion v2.1的冻结VAE编码器上进行梯度优化。

**📊 数据集**

在CelebA‑HQ和VGGFace2两个公开人脸数据集上进行实验，使用每个身份4张样本进行防护与个性化训练。

**📈 对比分析**

与Anti‑DreamBooth、Mist、SDS(-)、MYOPIA、VCPro等五种基线相比，SEALV在PSNR、SSIM、LPIPS、DISTS等图像质量指标上均名列前茅，同时在身份相似度(ISM)低于0.15、面部检测失败率高达0.93，显示出更强的防护效果。

**⚠️ 局限性**

仅针对基于VAE编码器的微调个性化方法有效，对Encoder‑based个性化方法无效；稀疏扰动易被JPEG压缩、模糊或缩放等预处理消除，导致身份相似度恢复。

---

## 578. PARLEY: Evaluating LLM Collaboration Through Cooperative Asymmetric Puzzles

**arXiv ID:** 2611.14242 | [PDF](https://arxiv.org/pdf/2611.14242v1)

**作者:** Javier Hernandez `[一作]` (Microsoft Research), Andrew D. Wilson `[通讯]` (Microsoft Research)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了PARLEY平台，开展 GPT‑4o 代理协作实验，系统记录四种不对称拼图任务的对话，并构建了基于数据的 PAC 失败分类法。

**💡 创新点**

创新点在于将认知心理学的合作理论与大语言模型协作任务相结合，提出了六类（Perception、Adaptation、Coordination）数据驱动的失败模式，并利用该分类指导系统提示干预，从而显著提升协作成功率。

**🔧 技术方法**

使用技术包括多模态 GPT‑4o、功能调用接口、对话日志自动化记录、LLM‑as‑judge 失败提取、文本嵌入+层次聚类生成分类、以及基于分类的系统提示干预。

**📊 数据集**

数据集由 400 条基于四个不对称任务（Tangram、Grid Coordination、Scene Comparison、Map Task）的对话组成，任务实例通过随机生成的视觉、工具和信息元素构成。

**📈 对比分析**

通过对照实验（Baseline vs. Intervention）结合 Wilcoxon、Fisher 等统计检验，介入后整体成功率从 60.5% 提升至 70.5%，各任务平均提升约 10%，并在成功对话中显著减少对话轮数。

**⚠️ 局限性**

局限性包括仅在单一模型 GPT‑4o 与四个受控任务上验证，缺乏跨模型或真实世界环境的泛化；PAC 分类可能未涵盖其他模型特有的失败模式，且任务设置相对简单。

---

## 579. StaRSE: Compact Russian Sentence Embeddings with a Sign-Coded Static Encoder

**arXiv ID:** 2611.14236 | [PDF](https://arxiv.org/pdf/2611.14236v1)

**作者:** Boris Malashenko `[一作]` (ITMO University), Valeria Efimova `[通讯]` (ITMO University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种名为StaRSE的轻量级俄语静态句子编码器，利用符号化的词表方向和单独的词向量范数实现低内存、低延迟的句子嵌入；

**💡 创新点**

创新点在于将Transformer输入嵌入表通过PCA+ABTT+SIF进行初始化，并在训练阶段使用对称InfoNCE损失和符号化后训练（sign‑projection）进行量化感知的后训练，使模型在不使用Transformer推理的情况下保持较高的检索质量；

**🔧 技术方法**

技术包括WordPiece分词器、token‑level PCA、ABTT、SIF词频加权、相邻句子正样本构造、对称InfoNCE对比学习、符号化后训练与Teacher‑distribution KL正则化、以及量化感知的后训练；

**📊 数据集**

使用的数据集包括Cultura RU Edu（文档文本用于相邻句子对）、俄语子集的mMARCO IR、ru-HNP、ru-paraphrase-NMT-Leipzig等有标签的句子对；RuMTEB 1.1用于多任务评估，MLDR-RU用于长文档检索评测；

**📈 对比分析**

与多种基线比较：上下文Sentence Transformer（ruBERT‑tiny‑turbo、FRIDA、GigaEmbeddings等）和静态词向量（DeepPavlov fastText、Navec、Multilingual static embeddings）。StaRSE在RuMTEB任务平均得分51.16，超过所有静态基线7.98点，接近最近的上下文编码器；在长文档检索中nDCG@10为51.66，几乎与USER2‑small相当，同时模型尺寸仅11.3 MiB，CPU单句编码延迟0.08 ms；

**⚠️ 局限性**

局限性包括：模型仅适用于俄语，迁移到其他语言需重新验证分词器、阈值和数据；作为静态模型，无法捕捉词序、语法或上下文语义；训练过程使用单一随机种子，缺乏多次实验的稳健性；最后，性能评估侧重编码器本身，未覆盖完整检索系统的端到端效率。

---

## 580. An AI Assistant for Automating Anti-Corruption Expertise of Draft Legislation Based on the TRACE Framework

**arXiv ID:** 2611.14244 | [PDF](https://arxiv.org/pdf/2611.14244v1)

**作者:** Serhii Zabolotnii `[一作]` (Cherkasy State Business College), Alina Tarasova `[通讯]` (Finport Technologies Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个公开基准和基于TRACE架构的分层流水线，用于自动化乌克兰国家反腐败预防署（NAZK）对草案法案的反腐败专业评估。

**💡 创新点**

创新点包括：①将评估过程拆分为确定性解析、单条款资格、规则映射和列表排序四层；②使用LLM评判器（LLM‑judge）进行语义匹配；③设计“腐败风险旋钮”可视化工具，将历史修订与流水线输出关联；④公开了完整的草案–结论对数据集及评估脚本。

**🔧 技术方法**

采用的技术包括：Python确定性解析器、BM25检索、Gemma 4 31B 与 Claude Sonnet 5.5（云端）进行单条款推理、结构化提示方案、规则映射至30因子税onomy、LLM‑judge（两路API一致性验证）、基于Cluster Bootstrap 的置信区间估计。

**📊 数据集**

使用数据集：132份公开的草案法案文本与其对应的NAZK结论，共计474条模型抽取的发现；按 21 条盲测、12 条测试集、100 条实验集等划分；此外还用 OCR 识别扫描件和模型转写。

**📈 对比分析**

评估方法：先地址匹配，再由LLM‑judge判定“相同”“部分相同”“不同”，计算严格召回（same）与宽松召回（same+partial）在 top‑10 与全量标记上的比例。结果显示：在 100 条草案中，Gemma 0.28/0.43，Sonnet 0.30/0.46；在 21 条盲测中，分层流水线比单一提示在严格召回上显著提升（0.13 vs 0.00）；列表排序进一步缩小本地与云端模型差距，减少标记量约 70%。

**⚠️ 局限性**

局限性：①参考发现与LLM‑judge均未人工校验，导致召回值仅可用于内部比较；②缺乏负例，无法评估精确率；③单条款推理缺少外部法条上下文，无法识别法条冲突；④OCR 误差和小样本导致置信区间宽；⑤仅覆盖草案法案，未验证对议会决议等其他类型法案的泛化。

---

## 581. CFGKAT: A Complete Calculus for Non-Local Control Flow

**arXiv ID:** 2611.14201 | [PDF](https://arxiv.org/pdf/2611.14201v1)

**作者:** Spencer Van Koevering `[一作]` (Cornell University), Tobias Kappé `[通讯]` (Leiden University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

提出了一套完整且可判定的程序语义等价公理化框架，并给出其完整性与正确性证明；

**💡 创新点**

在处理带标签跳转、循环和指示器变量的语言中实现了完整的等价判定方法；

**🔧 技术方法**

采用了行为论证、Bisimulation、结构归约与规范化技术；

**📊 数据集**

无实测数据集，主要基于理论推导与示例程序验证；

**📈 对比分析**

通过示例证明与已有公理的可推导性，未涉及实验性能评估；

**⚠️ 局限性**

证明过程繁复，适用性对大型程序的实用性尚需进一步验证。

---

## 582. The Monophonic Directed Convexities

**arXiv ID:** 2611.14191 | [PDF](https://arxiv.org/pdf/2611.14191v1)

**作者:** Júlio Araújo `[一作]` (Universidade Federal do Ceará), Rudini M. Sampaio `[通讯]` (Universidade Federal do Ceará)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了有向图的单子凸性与强单子凸性，并给出了它们在不同图类中的凸集判定、hull number 与 interval number 的多项式与 NP 难度结果。

**💡 创新点**

首次将单子凸性与强单子凸性引入有向图，并证明了它们在有向图中的凸集判定、hull number 与 interval number 的复杂性，揭示了强有向图可线性求 hull number 而 interval number 仍为 NP‑难。

**🔧 技术方法**

运用了图构造、最短路径搜索、强连通分量分解、从 3‑SAT 与 2‑链路问题的多项式归约等图论技术。

**📊 数据集**

未使用真实数据集，而是通过理论构造的合成图和图实例进行验证。

**📈 对比分析**

对比了可在多项式时间内求解的图类（如强有向图、传递无向图）与 NP‑难图类；在强有向图中 hull number 可在 O(|V|+|A|) 时间内求得，而 interval number 在强有向图和双分图中为 NP‑难。

**⚠️ 局限性**

尚未确定强单子凸性的 interval number 的复杂性，且对更一般的有向图类（如半完全图、多分图）的凸性参数研究仍处于开放状态。

---

## 583. BARS: Budget-Aware Sequential Sampling with Refusal for Majority Identification under Unknown Distributions

**arXiv ID:** 2611.14171 | [PDF](https://arxiv.org/pdf/2611.14171v1)

**作者:** Shenxi Liu `[一作]` (Beijing Institute of Technology), Kan Li `[通讯]` (Beijing Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种预算感知的顺序抽样规则 BARS，能够在给定样本预算内识别答案分布的多数，同时在无法满足误差目标时给出拒绝输出。

**💡 创新点**

创新点在于将误差预算、拒绝机制与自适应抽样结合，利用 top‑two gap 的时间均匀置信序列实现可验证的错误控制，并提供可证明的误判与误拒绝概率界。

**🔧 技术方法**

采用 Hoeffding 及经验 Bernstein 时间均匀置信序列、运行长度动态规划、以及快速分支与朴素分支的组合，形成完整的停止和拒绝决策。

**📊 数据集**

实验使用合成 Δ‑grid 数据集、GSM8K 与 MATH 两大算术推理基准，基于预生成的多样本池进行评估。

**📈 对比分析**

与固定预算、多臂识别与现有 LLM 早停方法相比，BARS 在多数可解实例上显著减少样本量（平均降低约 19‑93%），在难题上通过合理拒绝提升整体准确率；在 GSM8K 上平均样本 6.21，准确率 93.8%，比 8 次采样提升 5.1 分；在 MATH 上样本 7.76，准确率 82.5%，比 8 次采样提升 12.5%。

**⚠️ 局限性**

局限在于：认证模式下的置信宽度常数过大导致在部署级预算下频繁拒绝；误拒绝概率的偏差缺乏无分布上界，实用模式仅靠经验校准；假设多数概率唯一且采样独立，未考虑多峰或严重偏斜分布；且拒绝定义依赖预算和模型特性，无法完全衡量问题难度。

---

## 584. Cross-Border Fulfillment Center Selection via Doubly Robust Estimation and Linear Programming

**arXiv ID:** 2611.14174 | [PDF](https://arxiv.org/pdf/2611.14174v1)

**作者:** Kriti Mahajan `[一作]` (Amazon), Patrick Bloebaum `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并评估了一套跨境履行中心（FC）选择模型，结合短期预测、双重稳健估计（DRE）和多目标线性规划，实现自动化选取最优源仓库；

**💡 创新点**

创新点在于将因果估计与预测相结合生成候选仓库，再用多目标LP在固定网络内实现软区域化，兼顾交付时间、成本、利润与发货量，显著优于人工和单目标策略；

**🔧 技术方法**

使用技术包括短期多变量时间序列预测、双重稳健估计（EconML DRLearner）、多目标二进制线性规划、贝叶斯优化调参、离线重放以及最近邻/相同ASIN匹配；

**📊 数据集**

采用12个月亚马逊跨境订单与仓库日志数据，聚合成每个源-目的-天的OD流指标（平均交付时间、运输成本、发货量、利润）；

**📈 对比分析**

通过与人工专家策略、无区域化基线及多种排名、LP、组合评分基线进行离线重放与影子部署比较，XB模型在三条线路上交付时间降低8.5–51.1%，成本降低4.4–48.3%，5/5合规率提升最多6.5个百分点，且在Pareto前沿表现最优；

**⚠️ 局限性**

局限性包括缺少仓库级XB库存/容量观测，无法进行在线A/B测试，且模型需要大幅库存重新分配，实际部署仍受制约。

---

## 585. An Augmented Lagrangian Approach to Lifted Bellman Linear Programming

**arXiv ID:** 2611.14183 | [PDF](https://arxiv.org/pdf/2611.14183v1)

**作者:** Jongchan Park `[一作]` (NAVER LABS), Donghwan Lee `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于乘子法的提升Bellman线性规划（AL-LBLP）求解器，用单一Q、V无目标网络的离线RL批判器，并与多种演员（高斯、流、best-of‑N）配合。

**💡 创新点**

将固定的铰链系数转化为可学习的乘子，并通过惩罚参数ρ实现自适应权重；同时提出优势归一化的DDPG+BC演员以及对比不同演员的统一批判器。

**🔧 技术方法**

采用乘子法（增量拉格朗日）、凸优化KKT分析、优势归一化的演员更新、单网络Q/V、一次性数据集训练以及best‑of‑N采样等技术。

**📊 数据集**

在50个基于状态的OGBench离线RL任务（含antmaze、humanoidmaze等）以及像素级操作任务上进行评估。

**📈 对比分析**

与ALBUM、IQL、FQL、FAC等基线在相同离线训练步数、单网络设置下对比，平均成功率分别为高斯55.8%、流55.7%、best‑of‑N 60.1%；在长时程任务中达到81.5%和79.8%，并在大批量尺寸下保持稳定。

**⚠️ 局限性**

对ρ和α仍需环境手动调参，单量纲乘子近似不足；在操纵任务表现不如导航任务，且小ρ在长时间训练时不稳定。

---

## 586. Nonadaptive One-Sided Testing of Binary Rank

**arXiv ID:** 2611.14163 | [PDF](https://arxiv.org/pdf/2611.14163v1)

**作者:** Nader H. Bshouty `[一作]` `[通讯]` (Technion Israel Institute of Technology), Nader H. Bshouty (Technion Israel Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

研究二进制矩阵的二进制秩属性的非自适应一侧误差测试。

**💡 创新点**

提出多项式量级（O(d^4/ε^2)）的测试器以及匹配的Ω(d^{3/2}/log(1/ε))下界，缩小已知的 O(d^4) 与 Ω(d^{3/2})之间的空白。

**🔧 技术方法**

采用矩阵分解与图论（支撑为团与独立集）构造完整矩阵、基于随机网格采样的概率分析以及冲突图染色完成子矩阵。

**📊 数据集**

无须真实数据集，所有实验均在理论分布（随机活矩形）上进行。

**📈 对比分析**

与之前的 O(2^d) 量级结果相比，提出的算法在维数无关下取得多项式复杂度；下界证明说明至少需要 d^{3/2} 次查询。

**⚠️ 局限性**

仍存在 d 的指数差距以及对样本行列数的对数因子，且测试仍需完整子格采样，无法实现更稀疏的查询模式。

---

## 587. Physics-Constrained Autoencoder for Interpretable Health Representation Learning in Turbofan Engines

**arXiv ID:** 2611.14266 | [PDF](https://arxiv.org/pdf/2611.14266v1)

**作者:** Leonardo Baldo `[一作]` (Zurich University of Applied Science), Manuel Arias Chao `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种物理约束自编码器，用于涡扇发动机PHM任务中构建可解释的健康表示。

**💡 创新点**

创新点在于将可微热力学方程嵌入解码器，使潜变量直接对应预设的物理量，既保留了重建精度，又实现了组件级的解释性。

**🔧 技术方法**

采用了物理信息约束自编码器（Physics‑Constrained Autoencoder）+ 传统热力学方程 + 训练损失中加入功率平衡残差，还用多项式回归去除工况影响。

**📊 数据集**

使用公开的 N‑CMAPSS 涡扇发动机数据集（DS003‑DS006）进行训练与评估。

**📈 对比分析**

与 PCA、全数据驱动 AE、瓶颈 1 维 AE、监督回归等方法进行基准比较；重建 R²≈0.9993，MAE≈0.0036；健康指数与地面真值 Spearman ρ≈0.979，RMSE≈0.0286，整体性能与最强的线性基线相当。

**⚠️ 局限性**

局限性包括：潜变量可观测性有限，部分物理量不可唯一辨识；模型基于简化热力学关系，可能无法捕捉更细致的物理耦合；需要健康参考区间来校正工况；对传感器配置敏感。

---

## 588. From Search to Decision: How Post-Training Shapes Information-Seeking Capabilities in Large Language Models

**arXiv ID:** 2611.14149 | [PDF](https://arxiv.org/pdf/2611.14149v1)

**作者:** Haohang Huang `[一作]` (Xi'an Jiaotong-Liverpool University), Haiyang Zhang `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对大型语言模型（LLM）在外部信息检索与推理中的信息寻求行为进行系统综述，并提出以六个互相关联的能力为中心的框架，整理了各类后训练方法、学习信号、优化策略、更新位置和信用分配机制。

**💡 创新点**

创新点在于：①将信息寻求拆解为“知识意识与检索效用”“信息需求表述”“信息空间探索”“证据评估”“证据整合与信念修正”“自适应搜索控制”六大能力；②构建能力‑学习信号‑优化‑更新‑信用分配的全链路视图；③提出能力专属评价原则和实验设计清单；④系统总结现有研究在行为提升、端到端性能和跨环境泛化方面的实证进展与空白。

**🔧 技术方法**

主要技术包括：监督学习（SFT）、偏好优化（DPO）、强化学习（RL）与混合策略；不同信用分配方式（StepSearch、AutoSearch、CriticSearch、TRACE、SSPO等）；专门组件训练（QREF、BRIEF、RAGTruth等）与整体策略更新；以及多种实验评估方法（行为级指标、受控干预、跨环境测试）。

**📊 数据集**

使用的数据集与评测环境多样：工具使用的 OOD 问题集、网页检索管线、内存管理基准、不同模型骨干的检索日志、预算约束下的任务、真实与对抗 QA 任务等，涵盖检索、推理和内存管理等子任务。

**📈 对比分析**

比较方法：在能力专属指标（如检索路由、查询改写、检索多样性、证据覆盖、支持判断、预算使用等）上与基线直接对比；在端到端任务（如知识问答、推理）中对比 RL 与 SFT 训练的模型；通过受控干预（固定历史、预算、检索环境）评估单一决策的改进；通过跨环境迁移（模型骨干、检索后端、预算变化）检验泛化能力。性能方面，后训练方法普遍提升各能力的指标，但提升幅度因能力和评测环境而异，且在某些场景仍不显著。

**⚠️ 局限性**

局限性：①对未来检索效用的前瞻学习仍不成熟，难以在未获取证据前准确估值；②不同能力间的协同学习缺乏统一监督，导致单能力提升不一定整体协同；③跨环境自适应仍受限，尤其在检索后端或信息质量变化时性能下降；④实验评测高度依赖任务设计与数据来源，缺乏统一标准，导致结果难以跨论文直接比较。

---

## 589. Representational Atoms in Neural Networks

**arXiv ID:** 2611.14168 | [PDF](https://arxiv.org/pdf/2611.14168v1)

**作者:** Chenhui Hu `[一作]` (Institute of Automation Chinese Academy of Sciences), Jun Zhao `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了神经网络内部表征的基本单元（FRUs），提出代表性原子（representational atoms）的理论定义与忠诚度/稳定性评估标准，并通过阈值激活稀疏自编码器（TSAE）在多模型、多模态下实现高忠诚、高稳定的原子识别，进一步揭示原子与人类概念结构的层次关系。

**💡 创新点**

创新点：①在压缩感知框架下正式定义FRU并引入忠诚度与稳定性两项量化指标；②证明阈值激活稀疏自编码器能在适当条件下识别高忠诚、高稳定的原子；③发现原子稳定性与单义性相关，并通过HowNet显示原子组合可重建人类语义结构。

**🔧 技术方法**

使用稀疏自编码器（BiJumpReLU等）、压缩感知稀疏表示、原子内积（AIP）定义、阈值激活机制、线性可分性理论、跨层语义对齐评估（HowNet F1）以及大规模实验。

**📊 数据集**

数据集包括Wikipedia、Wikidata、LAION‑COCO、OpenHowNet以及多种预训练模型（Gemma‑2‑2B/9B、Llama‑3.1‑8B、Qwen‑3.5‑9B、Olmo‑Hybrid‑7B、Stable Diffusion v1.5、FLUX.2‑klein‑9B）产生的内部表示。

**📈 对比分析**

与传统神经元、已训练特征（Gemma Scope/Llama Scope）以及多种SAE架构（ReLU、TopK、AbsTopK、JumpReLU、BiJumpReLU）对比，采用忠诚度R²、稳定性q*、单义性得分和语义F1评估。BiJumpReLU在R²>0.95、q*>0.98的同时，单义性与稳定性呈显著正相关（Spearman ρ≈0.71）。

**⚠️ 局限性**

限制：原子识别依赖稀疏性假设且对阈值敏感；实验主要基于Transformer与扩散模型，未覆盖更广泛网络；对人类语义映射受HowNet注释完整性影响，原子与语义结构的对应不是完全一一对应；计算成本较高，需大规模GPU资源。

---

## 590. ViGA-FLAME: Aligning Latent Visual and Geometric Representations for Robust 3D Head Expression Estimation

**arXiv ID:** 2611.14218 | [PDF](https://arxiv.org/pdf/2611.14218v1)

**作者:** Wen Hao He `[一作]` (Hong Kong Baptist University), Jie Chen `[通讯]` (Hong Kong Baptist University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 ViGA-FLAME 框架，利用 Parametric–Geometric VAE (PG‑VAE) 学习统一的几何与参数潜在空间，并通过 Visual‑Geometric Adapter (ViGA) 将单张图像的视觉特征映射到该潜在空间，从而实现单图像 3D 头部表情与 FLAME 参数的前向预测。

**💡 创新点**

创新点在于：1）将密集几何与可解释 FLAME 参数放入同一变分自编码器，兼顾细节重建与参数解释；2）设计只读的 ViGA 侧支路，通过非对称注意力将视觉特征对齐到几何潜在空间，而不破坏预训练视觉编码器；3）引入多重监督（几何、参数、解码表面一致性）提升表面重建与参数回归的鲁棒性。

**🔧 技术方法**

使用的技术包括：变分自编码器（VAE）+自注意力网络；四个参数条件语义查询与可学习几何查询；交叉注意力与自注意力融合；多任务损失（Huber、MSE、SO(3) geodesic、KL 等）；预训练视觉 Transformer（Sapiens2）与自适应 ViGA 适配器。

**📊 数据集**

主要使用合成数据：采样 FLAME 参数并通过 Nvdiffrast 渲染得到图像、顶点和参数；训练集包含 100K 张 768×1024 的渲染图像；评估数据包括 RenderMe360（真实图像）和多视角合成测试集。

**📈 对比分析**

与 DECA、SMIRK、TEASER、SHeaP 等基线进行对比，评价指标为 Chamfer‑L1（mm）和 F‑score@5mm。ViGA‑FLAME 在所有合成极端场景和多视角测试中均显著优于基线，尤其是参数分支在“所有极端”情况下最为突出；在 RenderMe360 上参数分支虽略低于 DECA/SHeaP，但仍优于 SMIRK/TEASER，显示出良好的迁移能力。

**⚠️ 局限性**

局限性包括：1）训练完全基于合成渲染，虽然做了多样化增强，但在真实图像中仍存在差距；2）参数与几何分支的平衡需要手动调节，过度一致性可能略微损伤几何重建；3）对极端姿态/表情的鲁棒性虽提升，但在极端遮挡或光照条件下仍可能受限。

---

## 591. Dyship: Leveraging Acoustic Features For Speaker Relationship Classification From Dyadic Conversations

**arXiv ID:** 2611.14241 | [PDF](https://arxiv.org/pdf/2611.14241v1)

**作者:** Kieran Sturges `[一作]` (Johns Hopkins University), Najim Dehak `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在Seamless Interaction数据集上构建并评估DyShip框架，用于从二人对话音频预测说话者熟悉度和亲属关系。

**💡 创新点**

系统性比较了五种语音表征（光谱、MFCC、WavLM、自监督、发声器运动和x-vectors）在熟悉度和亲属关系分类中的效果，并提供可复现的端到端流程与代码。

**🔧 技术方法**

使用ResNet‑34与SVM分类器、WavLM、ECAPA‑TDNN、SPARC等预训练模型，以及加权平均与无权重融合方式进行对话级推理。

**📊 数据集**

基于Seamless Interaction 4000小时双人对话数据集，划分训练/开发/测试三集，确保参与者不重叠。

**📈 对比分析**

在对话级别上使用准确率、宏F1和AUC评估，谱图和MFCC在熟悉度上取得最高准确率（87.81%），WavLM在亲属关系上最高准确率（85.17%），无权重融合进一步提升到熟悉度AUC 0.925、亲属关系AUC 0.916，均优于先前人类和LLM基准。

**⚠️ 局限性**

仅评估于单一数据集，测试/开发集人数有限且存在多会话相关性；亲属关系类别不平衡导致AUC与F1偏差；模型未显式建模交替与时间依赖关系。

---

## 592. DailyWildfireCAN: A Dataset and Benchmark for Next-Day Wildfire Spread Prediction in Canada

**arXiv ID:** 2611.14207 | [PDF](https://arxiv.org/pdf/2611.14207v1)

**作者:** Mahacine Ettahri `[一作]` (Mila - Quebec AI Institute), Gaétan Marceau Caron `[通讯]` (Mila - Quebec AI Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了一个新的加拿大每日野火蔓延预测数据集DailyWildfireCAN，并构建了对应的基准评测框架；

**💡 创新点**

首次在加拿大尺度上提供高分辨率（90 m）每日火势演进标签，融合多模态多时序输入，支持细粒度、跨季节的模型评估；

**🔧 技术方法**

使用深度学习多模态分割模型（如U‑Net、SegFormer、Swin‑U‑Net、U‑TAE等），并结合传统仿真、时序卷积、注意力和预训练地理基础模型进行对比；

**📊 数据集**

数据集涵盖2017‑2024年8个火季共2440场火灾，包含33个特征（地形、气象、植被、森林结构、土地覆盖、人类影响、Sentinel‑2影像等），标签来自CFSDS；

**📈 对比分析**

在mono‑temporal和multi‑temporal两套数据上对比多种模型，单时序U‑Net+Time‑Gap在Test‑Easy、Test‑Medium、Test‑Hard的AUC‑PR分别为0.287、0.233、0.192；多时序U‑TAE最高，分别为0.352、0.306、0.259；总体表现优于传统基线，但在大火、长周期、偏远地区仍显不足；

**⚠️ 局限性**

局限包括天气输入分辨率粗糙、仅针对已点燃的火灾、对大面积、长时间火势预测仍不稳定，且仍需改进对空间时间标签不确定性的鲁棒性。

---

## 593. Stagewise and interpretable emergence of inside-outside computation in Transformer-based language models

**arXiv ID:** 2611.14221 | [PDF](https://arxiv.org/pdf/2611.14221v1)

**作者:** Jack T. Parley `[一作]` `[通讯]` (École Polytechnique Fédérale de Lausanne), Jack T. Parley (École Polytechnique Fédérale de Lausanne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了 GPT‑2 风格 Transformer 在带可变树结构的合成 PCFG（VT‑RHM）上的学习动态，揭示了阶段性学习曲线并证实模型内部实现了 inside‑outside 句法解析。

**💡 创新点**

创新点在于：①用可解析的随机层级语法构建可控合成数据；②把 Transformer 的学习曲线分解为可解释的阶段并与解析算法对应；③通过线性探测展示中间层编码了隐藏树变量，实现了 encoder‑decoder 结构的可视化。

**🔧 技术方法**

采用了 GPT‑2 结构的 Transformer、线性探测器、inside‑outside 张量实现以及自监督的末位标记预测任务。

**📊 数据集**

使用的仅是合成语料：VT‑RHM 语法（L=3、v=8、f=1/4），在 varying‑tree 与 fixed‑tree 两种训练协议下生成句子，并在 fixed‑tree 测试集上评估。

**📈 对比分析**

与传统 fixed‑tree 训练相比，varying‑tree 训练在更深层次实现了 inside‑outside 计算，交叉熵损失逐阶段逼近 Bayes‑optimal；实验表明最小深度需 D=4，D=6 及 D=12 在收敛速度与稳定性上几乎相同。

**⚠️ 局限性**

局限性包括：仅在合成语法上验证；未提供完整的样本复杂度理论；只关注末位标记预测；未探究更大词表或多层次语法、不同 Transformer 架构或真实自然语言数据的适用性。

---

## 594. Joint Quantization and Dynamic Partitioning for Context-Aware Edge Vision Transformers

**arXiv ID:** 2611.14196 | [PDF](https://arxiv.org/pdf/2611.14196v1)

**作者:** Zhu Jiaqi `[一作]` (Walmart), Kumar Saurabh `[通讯]` (Walmart)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种在车联网中对视觉Transformer进行动态拆分与混合精度量化的边缘算力调度框架。

**💡 创新点**

创新点在于将流特定的模型拆分点与层级的INT8/FP16量化决策联合优化为混合整数线性规划，实现能耗最小化且满足时延与任务质量约束。

**🔧 技术方法**

采用MILP优化、TensorRT推理、Pyomo+SCIP求解、能耗与时延模型。

**📊 数据集**

使用OPV2V基准中的多流LiDAR/摄像头数据集。

**📈 对比分析**

与全局FP16、静态早/晚拆分、统一INT8拆分比较，实验在Jetson Orin Nano/NX与AGX Orin上，联合优化后V2V能耗降低33‑36%，时延降低约34‑45%，mAP_50仅下降3‑4个百分点。

**⚠️ 局限性**

局限在固定两节点拆分、仅考虑INT8/FP16两种精度、未对不同V2X链路条件或多边缘节点扩展，且求解仍需依赖离线优化。

---

## 595. Performance Portability Explained: Insights from accelerating ICON microphysics

**arXiv ID:** 2611.14204 | [PDF](https://arxiv.org/pdf/2611.14204v1)

**作者:** Faveo Hörold `[一作]`, Hannes Vogt `[通讯]` (Swiss National Computing Centre)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在ICON气候模型的Graupel微物理核上，针对七种性能可移植框架（OpenMP、Kokkos、RAJA、Alpaka、oneAPI/DPC++、GridTools、PyGridTools）进行实现、编译与基准测试，评估其在多种HPC系统（CPU+GPU）上的运行时间、可移植性与工程成本；

**💡 创新点**

提出框架表达性（expressivity）度量，系统性分析稀疏优化（masking、thresholding）对不同硬件的性能影响，并展示同一框架在不同执行实例下可获得最佳性能的事实；

**🔧 技术方法**

使用C++/Fortran并行化技术（OpenMP pragma、Kokkos/RAJA/Alpaka抽象层、oneAPI/DPC++、GridTools DSL、Python JIT），并结合多种编译器（GNU、Clang、Intel、NVIDIA、AMD、HPE）和编译选项；

**📊 数据集**

使用ICON的Graupel核数据，分别在10 km、20 km、40 km三种网格分辨率上跑，测试多台HPC节点（EPYC、Xeon、Grace、A100、MI300、GH200等）；

**📈 对比分析**

采用每个框架最佳编译器，跑105次取中位数，比较CPU和GPU的运行时间，发现各框架在同一设备上相差不超过2倍；对稀疏优化的开启/关闭做细粒度对比，验证其对性能的正向或负向作用；

**⚠️ 局限性**

局限包括：框架成熟度差异导致编译器支持不完整、部分框架无法在所有GPU上生成高效代码、浮点非结合性导致“正确性”难以定义、实验需要大量手工调优、稀疏优化虽性能提升但会略微改变数值结果、测量不完整（如某GPU框架无可用测试）。

---

## 596. Koopman Model-Based Reinforcement Learning

**arXiv ID:** 2611.14186 | [PDF](https://arxiv.org/pdf/2611.14186v1)

**作者:** Wenjian Hao `[一作]` (Purdue University), Shaoshuai Mou `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于 Koopman 动力学的模型驱动强化学习（K-MBRL），在未知非线性机器人动力学下，通过单步预测加上 critic 估计的成本-到-期望来学习控制策略，并实现实时 feedforward 控制。

**💡 创新点**

创新点：①将深度 Koopman 模型与一阶 critic 结合，单步预测避免多步误差并显著简化梯度计算；②提供全局收敛保证的理论分析；③在仿真与真实四足机器人上实现高样本效率、低计算成本的控制。

**🔧 技术方法**

使用技术包括深度 Koopman 动力学学习（DNN lifting + 线性矩阵预测）、Actor‑Critic 梯度更新、随机梯度下降、经验回放、单步预测 + critic 目标、理论收敛证明、MuJoCo 仿真、物理四足机器人实验。

**📊 数据集**

数据集：从随机采样得到的状态‑动作‑下一状态三元组（初始为空），随后通过数据增强收集 1–2 万条样本；在半独行、四足仿真任务中使用 MuJoCo 生成的数据；硬件实验使用传感器观测得到的状态序列。

**📈 对比分析**

与 SAC、RKL、EMBRL、MPC‑optimal 等基线对比，K‑MBRL 在半独行和四足仿真任务中取得更高回报、更快收敛、每次更新更短的计算时间；在真实四足机器人上完成起立、双足步行与降落任务，表现优于基线并具有较好的实用性。

**⚠️ 局限性**

局限性：对数据覆盖度和提升函数设计敏感；单步预测在极端非线性或长时间跨度的系统中可能不足；需要足够的网络容量来保证 Koopman 逼近；硬件实验受传感器误差、执行保护与物理不对称影响，需额外的校准和适配。

---

## 597. GARD: Gradient-Guided Adversarial Representation Defense against Backdoors in Multimodal Contrastive Learning Models

**arXiv ID:** 2611.14143 | [PDF](https://arxiv.org/pdf/2611.14143v1)

**作者:** Yuhong Wang `[一作]` (Southeast University), Songze Li `[通讯]` (Southeast University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种针对多模态对比学习模型（如CLIP）的后置防御方法GARD，该方法通过全局和局部对抗扰动来触发模型的表示敏感性，然后将扰动后的表示与干净图像的锚点重新对齐，以消除后门行为。

**💡 创新点**

创新点在于：1）实现了无目标、无触发器的防御策略；2）通过组合全局与局部对抗探测，能同时覆盖分布式与局部化的后门触发结构；3）避免了传统后门防御中对目标类别或触发器形状的依赖。

**🔧 技术方法**

技术方法包括：梯度引导的L∞约束全局扰动、可学习的局部补丁扰动；表示重定位与干净锚点对齐的对比损失；交替优化全局/局部扰动与重定位的训练框架；无需推理时额外计算。

**📊 数据集**

使用的主要数据集：ImageNet‑1K（作为评估集），小规模（20K）干净ImageNet子集作为防御训练；在基线对比中还使用了500K CC3M图文对。模型基底为CLIP ViT‑B/32和RN50。

**📈 对比分析**

与FT、CleanCLIP、PAR、InverTune以及无防御基线进行对比，覆盖六种后门攻击（BadNets、Blended、SIG、BadEncoder、BadCLIP、BadVision）。实验结果显示，GARD在所有12个模型/攻击组合中将攻击成功率（ASR）降至<1%，同时保持与未受攻击模型相近的干净准确率；其余基线在至少一种攻击下失效。还在匹配防御数据、适应性攻击以及不同扰动参数下验证了鲁棒性。

**⚠️ 局限性**

局限性包括：1）需要一定量（至少1–2万张）干净且类别平衡的数据；2）对扰动预算和补丁尺寸有一定依赖，过大或过小可能影响效果；3）在极端对抗性攻击或未知触发器结构时的性能尚未彻底验证；4）对干净准确率的轻微下降。

---

## 598. AutoThreshold: Joint Confusion-Profile Learning for Adaptive Mask Binarization

**arXiv ID:** 2611.14279 | [PDF](https://arxiv.org/pdf/2611.14279v1)

**作者:** Shaokai Wu `[一作]` (Shanghai Jiao Tong University), Hongtao Lu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了 AutoThreshold，一种后置阈值策略，通过预测正负像素在分数区间的联合质量并利用累计解码得到阈值。

**💡 创新点**

通过联合混合分布预测实现阈值无关的混淆曲线，避免分别估计患病率与条件率，支持多指标无训练重置。

**🔧 技术方法**

采用 softmax 归一化、累计解码、可插拔的度量解码器、轻量 MLP 预测分数统计，以及联合损失正则化。

**📊 数据集**

在 Kvasir 语义分割数据集及其 CVC 等转移数据集、KITTI Road 失真评测等上进行实验。

**📈 对比分析**

与固定阈值、全局阈值、Otsu、直接价值回归、岭回归、GBRT 等基线对比，AutoThreshold 在跨域测试宏 Dice 提升约 0.008，并在多指标上优于 Otsu；在某些任务中直接价值回归仍略占优。

**⚠️ 局限性**

仅适用于可按阈值分解的决策单元（如像素级），不适用于全局或 NMS 等耦合决策；在极端或未见的域偏移下表现不一定优于直接价值回归。

---

## 599. Personalized Affective Perception via Continual Learning in Human-Agent Negotiation

**arXiv ID:** 2611.14247 | [PDF](https://arxiv.org/pdf/2611.14247v1)

**作者:** Mehmet Onur Keskin `[一作]` (Özyeğin University), Catholijn Jonker `[通讯]` (Delft University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种利用持续学习的个性化情感感知系统（CLIFER），在实时人机谈判中通过情绪反馈调节代理的目标效用，提升谈判效果。

**💡 创新点**

将连续情感（valence–arousal）估计与差分反馈规则结合，实现对代理策略的实时自适应；同时采用异步生成重放的持续学习架构，解决灾难性遗忘与实时性冲突。

**🔧 技术方法**

使用CLIFER框架（Growing-When-Required记忆网络 + 条件对抗自编码器）、FaceChannel预训练模型、Solver Agent决策策略与差分情感反馈公式；硬件包括Nao机器人、Webcam、GPU加速。

**📊 数据集**

实验数据来自66名大学生的双人谈判，使用Holiday域（4个议题）和Grocery域（6分钟演示）收集面部视频和谈判记录，生成的情感估计及谈判指标。

**📈 对比分析**

与固定预训练情感模型FaceChannel做配对比较；结果显示CLIFER在个人效用上提升了约0.022（p=0.017，d_z=0.30），参与者让步率提高9.35个百分点（p=0.013，d_z=0.31），且CLIFER在情感估计上产生更高的arousal值和正的valence–arousal相关性。

**⚠️ 局限性**

局限性包括：仅使用面部单模态；需先进行演示冷启动；样本为单文化大学生，未验证跨文化/长期交互；缺乏对灾难性遗忘的单独评估；未与其他情感基线或多模态系统做广泛对照。

---

## 600. Ninety-Six Ways to Measure a Scribe: Fragmented Evaluation of Ambient Scribes in Healthcare

**arXiv ID:** 2611.14284 | [PDF](https://arxiv.org/pdf/2611.14284v1)

**作者:** Amama Mahmood `[一作]` (Johns Hopkins University), Chien-Ming Huang `[通讯]` (Johns Hopkins University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统综述方法，对54篇关于环境式人工智能笔记员（ambient AI scribe）在临床实践中的评估研究进行汇总与分析；

**💡 创新点**

创新点在于构建了10个评估维度、40个指标、96种操作化方式的完整映射，量化了指标的集中度与碎片化，揭示了安全、患者结果等关键维度的严重缺失。

**🔧 技术方法**

使用了系统综述技术（PRISMA 2020、数据提取表、手工编码、频数统计），未涉及机器学习或新模型开发。

**📊 数据集**

主要数据来源来自被纳入研究的54篇论文，涵盖EHR审计日志、问卷调查、访谈文本、临床笔记文本和计费数据等。

**📈 对比分析**

比较方法采用计数与频数分析而非元分析；结果显示大多数研究无法直接比较，文献整体缺乏可量化的性能汇总，效率评估占主导但未提供统一标准。

**⚠️ 局限性**

局限性包括：研究设计普遍为短期、单中心、缺乏对照；主要以自报为主，导致偏倚；指标碎片化、缺乏统一定义；安全、临床结果与患者体验等重要维度缺失；缺少长期、跨站点的严格评估。

---

## 601. Federated Learning for Cross-Regional Epidemic Forecasting with Asynchronous Outbreaks

**arXiv ID:** 2611.14272 | [PDF](https://arxiv.org/pdf/2611.14272v1)

**作者:** Ruibin Liu `[一作]` (Macau University of Science and Technology), Ana Teresa Freitas `[通讯]` (University of Lisbon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

使用联邦学习框架（Flower+FedAvg）在广东、澳门与葡萄牙三地区的公开 COVID‑19 日新增病例数据上训练 LSTM 模型，实现跨境短期预测；

**💡 创新点**

首次在异步疫情、异质干预背景下，在真实异质环境中完成跨国联邦学习，既保持数据主权又实现了模型协同提升；

**🔧 技术方法**

采用 Flower 联邦学习框架、FedAvg 聚合、LSTM 网络、7 天移动平均预处理及非负预测约束；

**📊 数据集**

利用 WHO 与中国 CDC 公开的每日新确诊病例时间序列（2021‑11‑24 至 2023‑01‑08）；

**📈 对比分析**

通过与本地单模型和集中式模型比较（nRMSE、nMAE、SMAPE），联邦模型在 1 天预测的 nRMSE 仅 0.001，整体提升约 88.3%，与集中模型性能相当；

**⚠️ 局限性**

局限性包括仅使用时间序列特征，未引入人流、气候等变量；未试验其他聚合或个性化策略；数据非 IID 导致训练不稳定；实验基于公开数据，未验证在真正受限情境下的实用价值。

---

## 602. Model-Based Authoring of SOC Simulation Exercises for Cybersecurity Education

**arXiv ID:** 2611.14227 | [PDF](https://arxiv.org/pdf/2611.14227v1)

**作者:** Felix Härer `[一作]` `[通讯]` (University of Applied Sciences and Arts Northwestern Switzerland), Felix Härer (University of Applied Sciences and Arts Northwestern Switzerland)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种基于模型的安全运营中心（SOC）模拟练习生成方法，将技术事件模型与练习编排模型结合，并通过LLM仅生成部分文本，最终生成可在虚拟SOC环境中执行的练习包。

**💡 创新点**

创新点在于：①通过技术模型与编排模型的对应关系实现跨文档一致性；②保持模型结构不变，仅让LLM填充文本字段，确保结构一致性；③设计并实现一致性验证器，能够检测并报告四类常见错误；④将完整生成、验证与执行流程集成到虚拟SOC原型中。

**🔧 技术方法**

使用了结构化模型（I和E）、Python生成器、OpenAI兼容LLM（GLM‑5.3）生成文本、Node.js服务、前端虚拟SOC界面、API交互及验证脚本。

**📊 数据集**

使用了一个“phishing‑to‑ransomware”参考练习数据集：8个事件、8个对象、6个证据、7个警报、13条时间线、12个学习任务、5个评估项目；以及静态证据文件作为资源。

**📈 对比分析**

通过在模型中植入四个人工错误（缺失证据、事件顺序错误、评估缺失答案、警报地址修改），验证器成功检测（TP）并对正常数据给出无错误（TN）；演示表明生成的练习包能够在虚拟SOC中无误执行，验证了方法的有效性。性能指标未给出，但验证过程无误，生成与执行时间未作量化。

**⚠️ 局限性**

局限性包括：仅覆盖虚拟SOC所需的数据，未实现网络层仿真或自动攻击执行；LLM生成的文本不做语义一致性检查；验证仅基于已实现规则，缺乏全面的形式化验证；未评估学习效果；只在单一练习案例验证，泛化性待进一步验证。

---

## 603. SpecValley: Aggregation as Spectral Valley in Dense Heterophilic Graphs

**arXiv ID:** 2611.14176 | [PDF](https://arxiv.org/pdf/2611.14176v1)

**作者:** Christoph Jahn `[一作]` (Mercedes-Benz AG), Bastian Goldluecke `[通讯]` (University of Konstanz)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种可学习的聚合阶段SpecValley，在节点特征先经过可调节的频谱谷后再通过传统ChebNet进行传播，从而提升异质图节点分类性能。

**💡 创新点**

创新点在于将聚合视为独立的频谱运算，并学习其谷位置、锐度和两侧增益，从而避免了传统度归一化固定的λ=1谷带来的信息损失。

**🔧 技术方法**

使用自注意力聚合、可学习的频谱谷函数、Chebyshev多项式卷积、图拉普拉斯谱分析和能量预算理论。

**📊 数据集**

在六个异质图节点分类基准上进行评测：Squirrel‑f、Chameleon‑f、Actor、Crocodile、Tolokers 和 HM‑categories。

**📈 对比分析**

与多种聚合器（sym, mean, attn, attninit, valley）和四类下游滤波器（APPNP, ARMA, Gegenbauer, ChebNet）交叉实验，SpecValley在四个基准上达到或超过最佳公开结果，性能提升明显；在Actor上仍落后。

**⚠️ 局限性**

局限在于仅针对密集异质图的线性部分，未深入探讨ReLU、层归一化对能量迁移的影响；对稀疏图或标签能量分布如噪声时聚合效果不佳。

---

## 604. Tree-Ising: Repairing Language Models with Bit-Level Ising Edits

**arXiv ID:** 2611.14210 | [PDF](https://arxiv.org/pdf/2611.14210v1)

**作者:** Hadi Salloum `[一作]`, Ammar Ali `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将语言模型的循环推理错误通过稀疏离散优化改写为位级Ising编辑，从而修复模型权重。

**💡 创新点**

将权重空间编程转化为基于BF16状态树的稀疏Ising优化，避免梯度训练且显著减少位修改量。

**🔧 技术方法**

使用BF16状态树、前向统计、行级QUBO/Ising模型、GPU模拟退火或模拟分岔求解以及QAOA量子近似优化算法。

**📊 数据集**

主要在MATH-500、GSM8K、AMC23、Minerva Math、OlympiadBench等数学推理数据集上进行评估。

**📈 对比分析**

在相同评估协议下与GPT-6 Astra LLM代理修复比较，Tree-Ising在所有5个基准上均优于代理：MATH-500提升至54.6%（+2.9%），GSM8K提升至70.5%（+15%），仅需4次前向传递，计算成本约$0.90。

**⚠️ 局限性**

仅针对推理循环错误，实验范围局限于同家族小模型和数学基准，未验证更大模型、不同任务或其他失效模式；前向统计近似忽略跨层交互，可能导致对部分问题的误修。

---

## 605. Breakdown of Autoregressive Multi-Step Reasoning Under Concentrated vs. Diffuse Attention Head Dropout

**arXiv ID:** 2611.14209 | [PDF](https://arxiv.org/pdf/2611.14209v1)

**作者:** Karol Struniawski `[一作]` `[通讯]` (Warsaw University of Life Sciences -- SGGW), Karol Struniawski (Warsaw University of Life Sciences -- SGGW)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在大型语言模型中，聚焦式注意力头遮蔽对推理链的影响，并与均匀随机遮蔽进行对照。

**💡 创新点**

创新点在于揭示了集中式遮蔽导致的“吸附陷阱”机制，并将单步密度与多步连续性对推理错误的作用进行分离。

**🔧 技术方法**

使用自回归Transformer模型的前向钩子对上层注意力头进行遮蔽，配合精确McNemar检验、Bootstrap置信区间和策略性遮蔽时间分布。

**📊 数据集**

实验数据来自GSM8K数学推理基准，分别使用250个样本做对照实验和100个样本做因子分离实验。

**📈 对比分析**

与均匀i.i.d.遮蔽相比，Qwen3-8B在聚焦遮蔽下准确率下降12.4pp，LLaMA-3.1-8B下降7.2pp；单步密度导致约-10pp的额外误差。

**⚠️ 局限性**

局限包括遮蔽预算实现偏差、单次排程随机性、未对KV缓存进行回滚、仅测试单一任务且未保存完整生成过程。

---

## 606. SinkState: On-the-Fly SVD of Recurrent States for Efficient Hybrid LLM Serving

**arXiv ID:** 2611.14246 | [PDF](https://arxiv.org/pdf/2611.14246v1)

**作者:** Haolin Jiang `[一作]` (Rutgers University), Hang Liu `[通讯]` (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种服务系统，在混合模型推理过程中将递归状态动态压缩为低秩SVD形式，从而显著减少状态存储与传输成本；

**💡 创新点**

创新点在于：①发现了线性注意力的“线性注意力沉降”现象，即查询、键、值向量聚集在稳定的低维子空间；②设计了基于局部SVD更新与异步CPU合并的自适应低秩压缩策略；③采用缓冲更新、层级化稀疏化与核融合技术，将压缩工作排除在解码关键路径之外；

**🔧 技术方法**

使用技术包括：随机化SVD初始化、增量SVD局部更新、缓冲区批量合并、异步CPU上精确SVD合并、CSR式低秩布局、GPU核融合与CUDA Graphs；

**📊 数据集**

评估数据集涵盖：GSM8K、MATH‑500、IFEval、IFBench、GPQA Diamond、LiveBench、RULER、以及长生成任务（32K token）；

**📈 对比分析**

与密集状态实现（Mamba‑SSM、FLA，亦即vLLM/SGLang常用基线）进行对比；在四大混合模型（Nemo‑4B、Nemo‑120B、Qwen‑9B、Qwen‑80B）上，峰值状态内存可压缩至原来的0.4‑0.6倍，平均内存缩减至0.24‑0.58倍；推理吞吐量提升最高1.58×（平均1.25×），可支持2倍以上的批量；模型准确率保持在±2.6点以内，长上下文与长生成任务无显著偏差；

**⚠️ 局限性**

局限性包括：对GPU+CPU协同依赖较高，需额外SVD计算开销，尤其在小批量或高秩状态下收益有限；实现复杂度较高；某些模型（如高秩GDN层）仍需退回密集存储；在极大批量或特殊硬件环境下可能仍存在瓶颈。

---

## 607. Blurring the Block Boundaries: Overlapping Block Denoising for Efficient Diffusion Language Models

**arXiv ID:** 2611.14387 | [PDF](https://arxiv.org/pdf/2611.14387v1)

**作者:** Amir Mann `[一作]` (Technion -- Israel Institute of Technology), Niv Nayman `[通讯]` (Amazon Web Services)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在自编辑块扩散语言模型中通过并行处理相邻块以提升推理速度

**💡 创新点**

提出Blurring the Block Boundaries方法，训练无关、无超参数的并行块刷新

**🔧 技术方法**

利用块级扩散、熵门控的未来块触发与自编辑技术

**📊 数据集**

在10个多任务基准（数学、科学、推理、指令、代码）上评估

**📈 对比分析**

与基线和其他多块方法比较，TPS提升11.5–14.5%，TPF提升15.9–19.4%，准确率波动±0.8pp

**⚠️ 局限性**

依赖自编辑，适用于掩码模型，无法保证完全相同输出，速度提升受低接受尾部限制

---

## 608. Evaluating Tactile Encoding Strategies for Communicating 3-D Shape

**arXiv ID:** 2611.14389 | [PDF](https://arxiv.org/pdf/2611.14389v1)

**作者:** MacKenzie C. Harnett `[一作]` (Texas A&M University), Rebecca F. Friesen `[通讯]` (Texas A&M University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文评估了在二维表面触觉显示（如可编程针阵和摩擦调制触摸屏）上两种编码三维信息的方法——高度映射和二进制层叠，并在不同显示配置（低/高密度针阵、带/不带模板的摩擦调制屏以及3D打印参考模型）上进行用户研究

**💡 创新点**

创新点在于将连续高度映射和分层二进制编码两种策略与实际可刷新触觉显示结合，并通过细粒度的显示配置（针密度、模板划分）揭示触觉抽象层次对三维信息传递的影响

**🔧 技术方法**

采用机械驱动针阵、摩擦调制屏以及3D打印模型作为刺激源，并利用手部跟踪（LeapMotion）记录手部运动，采用磁性块组装任务评估性能

**📊 数据集**

使用自制的三维模型（9个立方体，层次高度0–2或1–2）作为实验数据集

**📈 对比分析**

通过两步对比：①在高度映射实验中比较低/高密度针阵与摩擦调制屏（带/不带模板）与3D打印模型的准确率、完成时间和手部能量消耗；②在二进制层叠实验中比较针阵、摩擦调制屏与3D打印模型。结果显示所有触觉显示均能实现高准确率，针阵在二进制层叠中更快；摩擦调制屏在无模板时完成时间最长，使用模板后性能接近针阵；3D打印模型总体表现最好

**⚠️ 局限性**

主要限制包括手部跟踪精度不一致导致数据不完整；针阵配置仅以静态3D打印模型验证，缺乏刷新功能验证；样本量有限，缺乏长期熟悉效应评估；高密度针阵间隙可能影响触觉连续性

---

## 609. Exploring Post-Hoc Disentanglement of Audio Foundation Models for Music Similarity

**arXiv ID:** 2611.14415 | [PDF](https://arxiv.org/pdf/2611.14415v1)

**作者:** Lucas de Souza Brandão `[一作]` (Universidade de São Paulo), Diego Furtado Silva `[通讯]` (Universidade de São Paulo)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

论文通过在已冻结的音频基础模型（CLAP、MuQ、MERT）上引入CSN（Conditional Similarity Network）框架，利用可训练的适配模块和掩码实现后置解耦，生成针对乐器、流派、情绪等属性的子空间表示，并通过三元组学习提升音乐检索性能。

**💡 创新点**

创新点在于：① 将CSN迁移至冻结的前沿模型上，避免大规模重训练；② 系统评估不同轻量级适配策略（Identity、One‑Layer、MLP）、池化方式（均值/注意力）以及掩码类型（固定/学习），揭示仅需极少可训练参数即可实现有效解耦；③ 通过“Specificity”指标量化掩码对目标属性的选择性，验证后置解耦的可解释性。

**🔧 技术方法**

技术手段包括：冻结的音频预训练模型特征提取；时间维度自适应平均池化/注意力池化；三层适配模块（Identity、One‑Layer、MLP）；固定或学习的元素级掩码；基于三元组的度量学习损失加正则化（掩码稀疏、正交、嵌入L2）；Adam优化、余弦学习率调度等。

**📊 数据集**

使用的数据集为：① MTG‑Jamendo（55k+曲目，按艺术家划分训练/验证/测试）；② MagnaTagATune（约26k短片，进行标签归一化后划分为流派、乐器、情绪三类），后者仅用于外部测试以检验泛化能力。

**📈 对比分析**

比较方法：与未进行掩码或适配的基线（直接使用模型的平均池化特征）进行宏观三元组准确率对比，并计算“Specificity”分数；实验覆盖所有组合（3模型×2池化×3适配×2掩码），每组5个随机种子。结果表明：① 引入One‑Layer或MLP适配后，整体三元组准确率提升1–3%；② 掩码的学习/固定并无显著差异，关键在于是否有适配参数；③ Specificity表现与准确率关联，说明解耦主要通过抑制跨属性干扰实现。

**⚠️ 局限性**

局限性：① 解耦效果对不同属性、不同基础模型的提升幅度有限，主要通过抑制干扰实现而非注入新特征；② 研究仅关注三种属性，未验证更细粒度或多标签情况；③ 评估指标局限于三元组准确率和Specificity，未覆盖检索召回率、用户满意度等实际应用度量；④ 仅在两大公开数据集上测试，需进一步验证在更大规模或不同音乐风格上的泛化能力。

---

## 610. What Factors Matter for Collection-Level Counting with Human-AI Collaboration?

**arXiv ID:** 2611.14380 | [PDF](https://arxiv.org/pdf/2611.14380v1)

**作者:** Seoyun Jeong `[一作]` (University of Massachusetts Amherst), Subhransu Maji `[通讯]` (University of Massachusetts Amherst)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了基于人机协作的集合级计数框架，结合不同模型、采样和估计方法，覆盖医学、野生动物和城市遥感等多领域任务。

**💡 创新点**

创新点在于将冻结的大规模多模态模型作为训练自由的起点，设计了基于模型预测的增量估计与自适应重要性采样，并提出通过小规模试点快速挑选最优模型的策略。

**🔧 技术方法**

使用了Faster R‑CNN检测器、开源与商业多模态大型语言模型、增量差分/比值/回归估计、主动测量与自适应重要性采样、模型更新策略以及Hellinger距离模型选择。

**📊 数据集**

利用12个单类计数任务，来源于BBBC041（疟疾细胞）、Birds、Kenyan Wildlife Aerial Survey、DOTA v1.5等四个公开检测数据集，覆盖医学、野生动物与城市遥感。

**📈 对比分析**

通过每个任务在不同标注预算下进行1,000次随机采样，评估绝对分数误差；结果显示冻结的开源多模态模型在低预算下误差低于10%，自适应重要性采样平均提升约2–4倍，精细调检测器在大预算下可进一步降低误差；在Pronghorn案例中，仅标注3.2%图像即可获得±5%置信区间。

**⚠️ 局限性**

研究仅针对单类计数，未涵盖多类或标签噪声情况，评估指标主要为误差而非其他统计量，且仅比较模型推理成本，未来需在更广泛的任务和模型演进下重新验证。

---

## 611. From Video Frames to Robot Meta-Frames: A Unified Latent Interface for World Action Models

**arXiv ID:** 2611.14365 | [PDF](https://arxiv.org/pdf/2611.14365v1)

**作者:** Guoqiang Liang `[一作]` (National University of Singapore), Mike Zheng Shou `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种将机器人信号打包进预训练视频扩散模型（latent space）的统一频域meta-frame接口，构建无额外解码器的联合世界动作模型。

**💡 创新点**

将机器人信号视为分布对齐问题，使用固定DCT频域通道块对信号进行编码和逆DCT升维，既兼容预训练模型，又显著减少latent帧数与推理延迟。

**🔧 技术方法**

预训练视频扩散模型（Cosmos-Predict2）、离散逆DCT频域编码、slot-aware mask任务路由、Diffusion denoising损失与Huber监督。

**📊 数据集**

RoboCasa、LIBERO、LIBERO-Plus模拟环境及三项Franka Panda真实桌面任务。

**📈 对比分析**

与Cosmos Policy等frame-injection基线对比；在RoboCasa 50演示下获得68.9%成功率（vs 67.1%），LIBERO-Plus整体得分83.4%（vs 82.2% Cosmos Policy），真实任务ID/OOD成功率分别为85.0%/68.3%，显著提升。

**⚠️ 局限性**

固定通道布局和权重对不同任务/信号规模不一定通用；语言引导仍受Cosmos-Predict2限制，对背景和机器人初始状态变化的鲁棒性有限。

---

## 612. An Information-Theoretic Perspective on Regulatory T Cell Localization as an IL-2 Interference Filter

**arXiv ID:** 2611.14338 | [PDF](https://arxiv.org/pdf/2611.14338v1)

**作者:** Alexander Wietfeld `[一作]` (Technical University of Munich), Andrew W. Eckford `[通讯]` (York University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

研究调节性T细胞（Treg）聚集对IL‑2介导的邻近激活位点干扰的抑制作用，并用信息理论评估其对本地与邻近信号的影响。

**💡 创新点**

创新点在于将信息理论方法应用于Treg空间聚集效应的定量分析，首次量化聚集对干扰抑制、本地信号保持、弱/强源辨别以及自/非自信号保留的多重收益。

**🔧 技术方法**

采用三维IL‑2扩散方程的准稳态近似、有限体积求解、T细胞种群动力学模型、基于逻辑回归的互信息下界估计以及多重随机种子蒙特卡洛仿真。

**📊 数据集**

使用的是基于模型的合成数据：通过在固定空间支持上随机放置IL‑2源、Treg分布与种群动力学参数产生的仿真输出，未使用真实实验数据。

**📈 对比分析**

对比无Treg、分散Treg和聚集Treg三种情形；评价指标包括信号干扰减少量、强源响应压制量、峰值扩增、持久性以及互信息与自/非自信息保留比例。结果显示聚集Treg在所有参数点上均实现约30‑60% 的干扰抑制，而强源响应压制仅约3‑15%，并在短距离显著提升弱/强源辨别（从0.71→0.88比特），在长距离显著降低邻近源信息（从0.83→0.35比特），自/非自信息保留提升至0.07‑0.18。

**⚠️ 局限性**

局限性包括：模型参数多为经验或有效值，缺乏实验校准；仅考虑两点静态布局，未模拟动态微域形成；信息估计依赖于输出选择、先验设定与解码器；缺乏对真实免疫组织的验证。

---

## 613. Precision on Demand: Escalating Weight Bits at Uncertain Decoding Steps in a Small Local Model

**arXiv ID:** 2611.14418 | [PDF](https://arxiv.org/pdf/2611.14418v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 614. When Should Agents Remember? Falsification-Gated Self-Evolution for LLM Agents

**arXiv ID:** 2611.14326 | [PDF](https://arxiv.org/pdf/2611.14326v1)

**作者:** Runxuan Tang `[一作]` (Nanyang Technological University), Liwei Hou `[通讯]` (National University Of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于验证器的自我进化框架FGSE，用以判定何时将经验产生的更新写入LLM代理的长期记忆。

**💡 创新点**

创新点在于将更新视为可验证的假设，通过临时测试目标迁移、反例化验和回归检测三种probe，形成三元门控的commit‑refine‑reject决策。

**🔧 技术方法**

技术包括结构化假设建模、可执行验证器、临时状态评估、范围细化与运行时监控等。

**📊 数据集**

使用了WebArena、AppWorld、τ-bench、AgentDojo、LongMemEval等多种可验证环境以及HumanEval、SWE-bench等编程/推理基准。

**📈 对比分析**

与直接记忆写入、Self‑Refine、Reflexion、ACE等基线相比，FGSE在Web、App、工具与长记忆任务上均提升了10–15%成功率，同时把伪更新率、回归损伤和违规率降低到个位数。

**⚠️ 局限性**

局限在于依赖可验证器，无法处理无可执行或间接验证的情形；范围细化一次且仅基于单轮反馈，可能不足以捕获更复杂的语义约束。

---

## 615. MMDM: Measuring Massive Multitask Decision Making

**arXiv ID:** 2611.14309 | [PDF](https://arxiv.org/pdf/2611.14309v1)

**作者:** Junhao Shen `[一作]` (Chinese University of Hong Kong), Hong Cheng `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了MMDM基准，用于评估系统‑1决策模型在多任务、多选项、长上下文等真实场景下的决策能力，并对28个模型进行了系统评估

**💡 创新点**

创新点在于将单选、完整多选和软标签三种决策任务统一入基准，并通过SB‑F、Soft Brier、Score nMAE等新指标细粒度评估概率质量和完整性，同时构建了覆盖16类任务、包含2–1024选项、最长64K标记长度的挑战性数据集

**🔧 技术方法**

采用系统‑1决策模型（单前向推理、直接输出概率头或一次解码），构建基于文本的决策实例，并使用专门设计的评估协议和指标

**📊 数据集**

使用了5,448道题目，覆盖16个任务类别，来源于QuALITY、LongBench、BIRD、ToolTalk等公开数据集，并结合重写、重组和新构造任务实现

**📈 对比分析**

通过对28个模型的单选准确率、完整多选准确率、SB‑F、多选均衡得分、Soft Brier、Score nMAE等指标进行对比，发现最高单选准确率约为78%但完整多选最高仅36%，概率分布质量（Soft Brier）和期望分数准确性差异显著

**⚠️ 局限性**

局限性包括：仅覆盖文本决策，未涉及多模态或交互式情境；软标签主要来自打分任务，缺乏更广泛主观评估；重写和重组方式仍可能导致对原始数据集的泄露影响

---

## 616. Cognitive Mapping of Learning by Thinking Mechanisms in Large Language Models

**arXiv ID:** 2611.14382 | [PDF](https://arxiv.org/pdf/2611.14382v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 617. SIGMO: Sigmoid Gated Interaction-Aware Motion Planning

**arXiv ID:** 2611.14414 | [PDF](https://arxiv.org/pdf/2611.14414v1)

**作者:** Avinash Singh `[一作]` (Honda Research Institute USA Inc), Sangjae Bae `[通讯]` (Honda Research Institute USA Inc)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了SIGMO（Sigmoids Gated Interaction‑Aware Motion Planner），一种利用多维物理特征通过多项式Sigmoid函数估计周围车/行人让行概率，并将该估计融入MPPI路径规划的轻量级交互规划方法。

**💡 创新点**

创新点：①设计多项式Sigmoid函数（Polygate）将时间碰撞、加速度、相对速度、路径优先级、距离门控等五个物理特征映射到让行概率，具备可解释性和极少调参；②将Polygate作为风险乘子直接嵌入MPPI，避免了对每条候选轨迹进行复杂的预测模型推理；③实现了在多代理、动态变化环境下的可扩展、可实时运行的交互规划框架。

**🔧 技术方法**

技术手段：多项式Sigmoid映射、Polygate层级融合、Model Predictive Path Integral (MPPI) 采样式优化、基于Bhattacharyya核的空间风险评估、基于弹性距离门控的安全约束。

**📊 数据集**

数据集与实验平台：①仿真交互场景（窄通道+人行道、合流密集车流）①MetaDrive 150个多样场景（Waymo Open Dataset初始化）②1/10 F1Tenth赛道硬件实验。

**📈 对比分析**

对比方法：原始MPPI、Oracle‑Yield（真实合作参数）、Oracle‑Traj（真实轨迹预测）。性能：在窄通道场景中，SIGMO成功率+4%、停滞时间-24%；在合流场景中，成功率+12%，冲突率保持不变；MetaDrive实验中，SIGMO成功率92.7%比IANN‑MPPI的89.3%更高，冲突率更低。

**⚠️ 局限性**

局限性：①交互模型较为简化，未覆盖更复杂的人类行为；②依赖MPPI，无硬约束，碰撞避免仍基于软处罚；③缺乏各Sigmoid组件的消融分析，难以评估单一特征对性能的具体贡献；④对极端高速或极端环境的鲁棒性尚待进一步验证。

---

## 618. Uniform Quantum Average-Case Fine-Grained Hardness of Arithmetized $k$-Orthogonal Vectors

**arXiv ID:** 2611.14411 | [PDF](https://arxiv.org/pdf/2611.14411v1)

**作者:** Dhruv Prasad Deshpande `[一作]` (Birla Institute of Technology and Science), Abhishek Mishra `[通讯]` (Birla Institute of Technology and Science)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `09944146-298c-433e-89df-37255de463d7` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文在量子计算框架下证明了k-正交向量问题（k-Orthogonal Vectors）相关的算术化多项式（^k）在均匀随机输入上具有细粒度硬度，给出了在量子Strong Exponential-Time Hypothesis（QSETH）假设下的下界 Ω(n^k/2‑o(1))，并讨论了在两种典型素数域（p>n^k‑1 与 p>n^k）上的硬度证明；

**💡 创新点**

创新点在于：①首次将经典的随机自还原（random self‑reduction）与量子平均案例评估相结合，完成了从判定问题到算术化多项式平均案例的直接还原；②提出了利用量子最小值搜索（Dürr‑Høyer）与非平凡素数间隙定理结合，克服了在均匀算法中素数生成开销的难题；③在两类规范素数族上给出了完整的统一量子下界与非统一量子下界的对比；

**🔧 技术方法**

使用了的核心技术包括：量子随机自还原、量子最小值搜索、经典素数间隙定理、随机采样求解正交向量、以及多项式自修正（polynomial self‑correction）等；

**📊 数据集**

本文为理论研究，无需具体数据集，全部以符号量化的输入（n维二进制向量集合、素数域 𝔽_p）为模型；

**📈 对比分析**

与基线方法（Grover 搜索 n^k 组元）相比，论文给出的下界与 Grover 的上界在判定问题上实现了匹配（均为 Θ(n^k/2)），但在算术化多项式的精确评估问题上仍存在上界未知的空缺，论文仅给出下界，无法直接证明或实现更优性能；

**⚠️ 局限性**

局限性包括：①仅给出下界，缺乏对应的上界算法，导致无法验证下界是否最优；②对素数生成的阈值和常数隐含在“sufficiently large n”中，缺少明确数值界限，实用性受限；③未讨论量子证明工作（proof‑of‑work）实现细节，未给出可实际部署的方案。

---

## 619. ARE-bench: Benchmarking Agents' Research Ability in Real-World Researcher Sessions and Workspaces

**arXiv ID:** 2611.14364 | [PDF](https://arxiv.org/pdf/2611.14364v1)

**作者:** Jiahang Lin `[一作]`, Xuanjing Huang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 ARE-bench benchmark，收集了 75 个真实科研会话中的任务，采用 Curator‑Builder‑Validator 流程生成任务，保留了会话历史（Background）、当前请求（Requirements）和工作区状态（Environment），并为每个任务制定二进制检查表；随后对七款前沿大模型在每个任务上做了 3 次 rollout，使用 ISR 与 CSR 两种指标评估完成度，并对成本、上下文长度、评判者一致性等进行分析。

**💡 创新点**

创新点在于：① 关注科研代理在持续项目中的实际交付，既需要完成当前请求，又要保持此前决策与工作区状态的连续性；② 通过检查表将需求映射到 207 个子需求，并按七个研究阶段归类；③ 采用 ISR 与 CSR 两种严苛与细粒度评估方式，揭示模型在完整交付与局部合规性上的差异；④ 对比历史与材料/工作区约束对模型性能的显著影响，凸显传统单一端到端评测的不足。

**🔧 技术方法**

技术手段包括：① 构建 Curator、Builder、Validator 三个角色实现任务采样、重放与检查表校验；② 利用 LLMs（GPT‑6.1‑Sol、Claude Opus 5.5、DeepSeek‑V4.1‑Flash、Qwen3.8‑Max、GLM‑5.3‑Flash、Kimi‑K3、Qwen3.8‑27B）执行任务；③ 对每个任务生成三轮 rollout 并统计 ISR 与 CSR；④ 进行成本估算、峰值上下文量分析、评判者（DeepSeek vs Qwen）的一致性评估；⑤ 通过实验揭示各研究阶段的难度与模型差异。

**📊 数据集**

数据来源为 75 条真实科研会话（涉及 Survey、Design、Data & Env.、Code、Execution、Analysis、Writing 七个阶段），每条会话保留在研项目的中间状态，未进行清理；任务覆盖 1,173 条检查项，涉及 150 个子需求，构成了 Benchmark 的“数据集”。

**📈 对比分析**

比较方法：对七个模型在同一 75 任务集上做 3 次 rollouts，计算 ISR（所有检查项通过）和 CSR（部分通过）。性能表现：GPT‑6.1‑Sol ISR 66.2%、CSR 92.7%；Claude Opus 5.5 ISR 56.9%；DeepSeek‑V4.1‑Flash、GLM‑5.3‑Flash、Kimi‑K3 ISR 均低于 40%；Qwen3.8‑27B ISR 仅 25.8%。在仅检测交付合同与通用规范时，三款模型约 69% ISR，但加入历史与工作区约束后 ISR 分别降至 40%、31.6% 和 28.4%。阶段级别分析显示 Design 与 Execution 难度最高；成本-ISR Pareto 前沿由 GLM‑5.3‑Flash、DeepSeek‑V4.1‑Flash 与 GPT‑6.1‑Sol 形成。

**⚠️ 局限性**

局限性：① 选取的科研会话可能偏向特定项目、工具或研究者群体；② 通过重放恢复工作区可能丢失原始上下文，匿名化处理亦可能剥夺关键信息；③ 检查表的编写为人工判断，仍可能漏判或误判；④ Benchmark 仅评估已选择的 75 任务，结果不一定能推广到更广泛的科研任务；⑤ 对模型的性能评估同时受使用的 harness 影响，跨模型差异可能包含 harness 成分。

---

## 620. Don't Just Trust Your Memory: Evidence-Linked Memory Reduces Memory-Induced Degradation in LLM Agents

**arXiv ID:** 2611.14320 | [PDF](https://arxiv.org/pdf/2611.14320v1)

**作者:** Yukyeong Lee `[一作]` (KAIST), Sung Ju Hwang `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 WikiTrace，一种将每条经验教训与其原始轨迹通过编译摘要和原始轨迹相连的外部记忆系统，允许 LLM 代理在执行新任务时回溯并验证或修正这些教训。

**💡 创新点**

创新点在于实现了教训与来源轨迹的证据链接，使得代理可以在执行时动态检索并核对先前经验，从而显著降低记忆诱发的退化。

**🔧 技术方法**

采用了分层记忆结构（原始轨迹 → 编译摘要 → 洞察），利用编译器从成功/失败轨迹生成摘要和洞察；通过 Prompt 设计使代理在必要时下沉到原始轨迹并使用终端工具检索细节。

**📊 数据集**

使用了五大基准的八个子任务套装，包括文本世界任务（ALFWorld、ScienceWorld、PlanCraft）和工具使用环境（AppWorld、τ^3-bench Retail）。

**📈 对比分析**

与无记忆 baseline、AWM、ExpeL、ReasoningBank 以及仅提供洞察的 Insights Only 进行比较；WikiTrace 在所有套装上均取得最高分，平均提升 10.7 分（相对无记忆），相对 Insights Only 提升 3.2 分，并将回归率从 6.5% 降至 4.5%。

**⚠️ 局限性**

局限性包括：仅依据二元成功/失败标签构建，可能导致编译摘要和洞察的误导；源链接虽可核查但不保证洞察正确；提示随记忆增大而膨胀，过大记忆时需检索，未在本文解决；实验分析依赖 LLM 判读，无法完全验证实际原因。

---

## 621. The Unused Input: The Initial State as a Conditioning Interface for Recurrent TSFMs

**arXiv ID:** 2611.14357 | [PDF](https://arxiv.org/pdf/2611.14357v1)

**作者:** Bernhard Voggenberger `[一作]` (NXAI GmbH), Sepp Hochreiter `[通讯]` (NXAI GmbH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了冻结的预训练循环时间序列基础模型TiRex的未使用初始状态，并证明通过一个小的状态初始化头对其进行调制能提升预测效果。

**💡 创新点**

创新点在于将初始状态视为可调节的条件接口，比较谱特征和外部文本信号与LoRA适配器的表现，并对注入状态随上下文长度衰减的机制进行量化。

**🔧 技术方法**

使用TiRex xLSTM骨干，配合谱摘要提取、CLIP式文本对齐（Qwen3-Embedding-0.6B）、小型线性状态初始化头、LoRA适配器、MASE评估指标等技术。

**📊 数据集**

采用GiftEval基准进行谱条件实验，MoTime数据集（MovieLens、News‑FO、Tianchi、Wiki People、PixelRec Full与Sub）用于外部文本条件实验。

**📈 对比分析**

对比零样本基线、参数匹配LoRA、表达力匹配静态状态、完整微调等方法；谱条件头相较基线提升约0.175%，文本条件状态在四个数据集上超过静态状态，LoRA+文本组合可达到或超过完整微调的121%提升，改进幅度虽小但稳定。

**⚠️ 局限性**

局限性包括仅在TiRex上验证，只有两种简单条件信号，未与提示/前缀调优直接对比，初始状态对长上下文的影响衰减显著，且未探索更丰富的多模态信号和其他循环骨干。

---

## 622. OnlyAct: Extracting Action Representations from Foundation Models for World Models

**arXiv ID:** 2611.14420 | [PDF](https://arxiv.org/pdf/2611.14420v1)

**作者:** Runjia Li `[一作]` (University of Oxford), Tomas Jakab `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过对预训练的 V‑JEPA‑2 基础模型进行无标签对比学习，将其改造成能够提取动作特征的动作编码器，随后用该动作编码器训练视频世界模型，实现跨域动作迁移与实时控制。

**💡 创新点**

创新点在于：①不从零开始学习动作空间，而是直接从已预训练的多模态视频表示中提取动作信息；②利用光流引导的场景交换正样本与同视频硬负样本，构造无监督对比目标，既提升动作表达能力，又保持场景不变性；③不采用传统压缩瓶颈，而是通过对预训练空间几何重塑来实现动作与场景的分离。

**🔧 技术方法**

主要技术包括：V‑JEPA‑2 视觉 Transformer 预训练模型、低秩适配器（adapter）进行轻量级更新、光流估计与图像裁剪/粘贴生成正样本、InfoNCE 对比损失、基于 AdaWorld 的自回归视频世界模型、以及对动作进行白化和控制模式的 fine‑tune。

**📊 数据集**

使用了 168K 条无标签帧对来自九大视频源（机器人操控、人类/第一人称视角、相机导航、游戏等）的数据进行预训练；评估时覆盖了 13 个含动作标签的数据集（包括 LIBERO、RLBench、SpatialVID、DROID、RoboCasa、MetaWorld、Robosuite 等）以及 NuScenes、DMLab、Minecraft 等新领域。

**📈 对比分析**

与 DreamDojo、AdaWorld 等从零开始的潜在动作方法以及冻结的 V‑JEPA‑2 编码器进行对比。结果表明：OnlyAct 在动作表达（平均 R² 0.490，明显高于 0.211）和场景不变性（LPIPS 0.304 与潜在动作相当）上均优越；在动作驱动的视频生成中，跨域转移指标 Transfer‑Dir 最高，控制精度（Follow 0.928 vs. 0.722）也显著提升，整体性能领先。

**⚠️ 局限性**

局限性包括：①正样本构造依赖静态摄像头的光流分割，移动相机时只能退化到颜色增强；②依赖大规模预训练基础模型，对模型的可迁移性和可解释性仍有限；③在极端动态或非人类主体的场景下，动作与场景的分离效果可能受限。

---

## 623. MammothDP: Private Boosted Decision Trees Ready for Trusted Hardware and Leakage-free Hyperparameters

**arXiv ID:** 2611.14317 | [PDF](https://arxiv.org/pdf/2611.14317v1)

**作者:** Moritz Kirschte `[一作]` (University of Luebeck), Esfandiar Mohammadi `[通讯]` (University of Luebeck)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一个基于可信执行环境（TEE）的差分隐私梯度提升决策树学习器（MammothDP），并对其进行了时序侧信道加固、改进的回滚防护和无额外隐私代价的自适应树停止机制。

**💡 创新点**

创新点包括：① 通过常数时间实现的S-BDT以及对噪声采样、内存访问、控制流等所有数据相关操作的硬化；② 结合Memoir思想的哈希链回滚防护，使TEE在存在时序侧信道攻击时仍能完整维护隐私预算；③ 利用中间梯度的后处理来实现早停，既保持DP保障又无需额外隐私预算。

**🔧 技术方法**

使用的技术包括：可信执行环境（Intel SGX/TDX、AMD SEV），常数时间编程（libfixedtimefixedpoint、FACCT、libftfp），RDP隐私核算与过滤，Microwalk时序侧信道验证，哈希链回滚防护。

**📊 数据集**

实验数据集为：Insurance（回归），Abalone（回归），Credit（分类），Adult（分类），Spambase（分类）。

**📈 对比分析**

通过与原始S-BDT、无隐私XGBoost以及其他基于TEE或MPC的GBDT方案比较，发现常数时间实现导致1.6–69.4倍的运行时开销；但在早停机制下，MammothDP在回归任务中显著优于使用静态超参数或隐私泄漏的超参数设置，分类任务在低隐私预算区间表现相当或略优；整体运行时在TEE+回滚防护下相较原始S-BDT提升2.8–8.3倍。

**⚠️ 局限性**

主要局限包括：常数时间实现带来的显著性能开销、实验仅覆盖5个数据集且只针对GBDT，未探索更复杂模型或多方安全计算；回滚防护仍依赖TEE硬件的安全性；早停机制虽然不增加隐私预算，但对不同任务的自适应性和理论收敛性仍需进一步研究。

---

## 624. Causal Concept Explanations for Deep Neural Models

**arXiv ID:** 2611.14407 | [PDF](https://arxiv.org/pdf/2611.14407v1)

**作者:** Josh Rountree `[一作]` (MIT), Julie A. Shah `[通讯]` (MIT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出Causal Concept Wrapper Network（CCW-Net），利用多中介干预因果中介分析将概念表示与模型输出的因果关系嵌入训练目标，并通过注意力头和向量化概念实现可解释且因果驱动的决策；

**💡 创新点**

创新点在于：①首次将干预中介分析作为深度网络训练目标，直接优化概念的必要性、充分性和独立性；②将概念表示从标量扩展为向量，既编码存在性又捕捉情境依赖的表达；③通过可逆流生成反事实概念，构建严格的因果评估框架；

**🔧 技术方法**

使用了可逆流（invertible flow）作为介质模型生成反事实概念、基于注意力的概念头、概念向量化表示、干预中介分析损失（必要性、充分性、独立性、presence），以及传统的任务损失和概念分类损失；

**📊 数据集**

实验数据集包括：航班控制仿真（extended trail formation）、CarRacing‑v1（视觉驱动任务）以及CUB‑200（200类鸟类的细粒度分类）；

**📈 对比分析**

通过与无对齐损失的同构模型、CW‑Net、CBM、β‑TCVAE、iVAE等基线比较。CCW-Net在保持相近或更低任务误差的同时，将活跃概念对输出的因果贡献提升至约94%（航班）/89%（驾驶）/91%（CUB），显著优于仅依赖概念准确率或解耦度的基线；

**⚠️ 局限性**

局限性包括：需要预先冻结骨干网络，依赖高质量且覆盖完整的概念标签，对标签稀疏和噪声敏感，评估仅聚焦于模型内部的因果关系而非外部数据生成过程，且对齐损失在不同任务对效果影响不一。

---

## 625. Complexity of Deciding Existence of Maltsev Polymorphisms of Finite Relational Structures

**arXiv ID:** 2611.14379 | [PDF](https://arxiv.org/pdf/2611.14379v1)

**作者:** Dejan Delic `[一作]` (Toronto Metropolitan University), Ali Syed `[通讯]` (Toronto Metropolitan University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文构造了一种多项式时间算法，用于判定给定有限关系结构是否存在Maltsev多项式；

**💡 创新点**

创新点在于利用Maltsev代数乘积的矩形性（rectangularity）属性以及三元图（triple graph）构造，避免了对具体多项式的假设；

**🔧 技术方法**

核心技术包括约束满足问题（CSP）框架、可判定的矩形性检验、三元图与群的构造，以及1,2-对称Datalog程序的局部一致性检查；

**📊 数据集**

论文主要为理论研究，未使用具体数据集进行实验；

**📈 对比分析**

通过理论复杂度分析证明算法在多项式时间内完成，未给出与其他算法的实验对比；

**⚠️ 局限性**

局限性在于算法仍属于P级，尚未压缩至更低复杂度类（如NC^2或#L），且对所有变量对的矩形性检查仍需多项式时间。

---

## 626. A Margin Perspective on LoRA: Robustness to Catastrophic Forgetting and Adapter Merging (MaLoRA)

**arXiv ID:** 2611.14397 | [PDF](https://arxiv.org/pdf/2611.14397v1)

**作者:** Ziqing Xu `[一作]`, Rene Vidal `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多任务和连续学习场景中，对LoRA（低秩适配器）的理论分析，阐释其对灾难性遗忘的鲁棒性以及独立训练的适配器可合并的机制，并基于此提出了无训练的边际基LoRA合并方法（MaLoRA）

**💡 创新点**

通过统一的边际视角，首次在三种正则化范式下解析LoRA最优解，揭示其与最大间隔分类器的对齐关系，并推导出精确的边际折中与最优合并系数，形成理论驱动的合并规则

**🔧 技术方法**

利用线性分类的最大间隔理论、低秩分解、正则化分析、边际计算以及实际验证中的验证集边际估计

**📊 数据集**

CIFAR‑100（用于线性探针验证）、PACS、VLCS、TerraIncognita（视觉多域任务）以及MNLI、Amazon Reviews、CivilComments（语言多域任务）

**📈 对比分析**

与多种无训练/训练后合并基线（Soup、TA、TIES、DARE‑TIES、DARE‑Linear）在多域分类任务上对比，MaLoRA在平均标准化精度上取得最高（≈93.3%）并保持最小相对降幅（≈6.7%），性能与基线相当甚至更优

**⚠️ 局限性**

理论假设受限于近正交任务结构、均衡类别样本和线性模型；未直接给出泛化或测试风险分析，且对非线性模型的理论解释仍待拓展

---

## 627. Uncertainty-Aware Hybrid Offline Reinforcement Learning with Online Action Selection Enabled by Sparse Gaussian-Process Critics

**arXiv ID:** 2611.14374 | [PDF](https://arxiv.org/pdf/2611.14374v1)

**作者:** Marco Chan `[一作]` (Imperial College London), Mehmet Mercangöz `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对过程工业的启动、停机等操作，提出HOFLON-sGP方法，用稀疏高斯过程替代HOFLON原始的树集成+AAE模型，在在线优化中引入低置信度界进行动作选择。

**💡 创新点**

创新点在于：①使用单一稀疏GP作为critic，利用其预测标准差做支持惩罚，消除额外的AAE模型；②通过低置信度界避免对数据稀疏区域的过度探索，保证闭环稳定；③在多噪声种子、多个critics的情形下实现无失败、与HOFLON相当甚至更优的性能。

**🔧 技术方法**

技术方法包括：一阶离线强化学习、稀疏变分高斯过程（Matérn‑5/2、SE核）、低置信度界（μ–κσ）动作选择、无模型在线优化（Nelder–Mead/POWELL/DIRECT）以及传统树集成（XGBoost）+AAE。

**📊 数据集**

使用工业聚合物连续搅拌釜反应器的启动数据集，共100条PI控制记录（含温度、聚合度、噪声等），作为历史转移用于离线训练。

**📈 对比分析**

比较方法为：HOFLON（XGBoost+AAE）、HOFLON无AAE、HOFLON-sGP（Matérn‑5/2 κ=3）以及平方指数GP，评价指标包括累计奖励、峰温、冷却器动作幅度、支持率等。实验显示HOFLON-sGP无失败、奖励与HOFLON相当，动作平滑且峰温低。

**⚠️ 局限性**

主要限制包括：仅在单一模拟器、固定初始状态和测量噪声下验证；未测试对工厂参数或非平稳扰动的鲁棒性；GP未包含积分器特征导致稳态偏移；置信水平 κ 的选择受核和训练样本影响。

---

## 628. Auditing the Judge: Internal Signals for LLM Reasoning Assurance

**arXiv ID:** 2611.14323 | [PDF](https://arxiv.org/pdf/2611.14323v1)

**作者:** Mayank Gulati `[一作]` (Free University Berlin), Gerhard Wunder `[通讯]` (Free University Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于内部注意力头激活的监督式错误检测探针，用于在大语言模型部署时在有限的人工审核或额外推理预算内优先挑选最有可能错误的答案。

**💡 创新点**

创新点在于：①将内部激活与外部自评置信分数进行对比，证明内部信号在特定预算下能显著提升错误召回率；②从理论上将预算约束下的误差召回、剩余风险和AUROC等指标关联起来，阐明为何全局AUROC不一定能反映部署时的性能；③通过跨数据集、权重精度和生成预算的联合迁移实验，界定了内部探针在协议变化时的可靠性边界。

**🔧 技术方法**

主要技术包括：①从模型的判断通道抓取所有层的注意力头输出；②使用Fisher判别统计进行头选择；③构造一层隐藏网络（MLP）或逻辑回归作为探针，将选定头的拼接特征映射为“正确性”分数；④对比内部探针与黑盒token质量分数在固定审计/推理预算下的性能。

**📊 数据集**

实验使用的公开数据集为数学推理任务GSM8K和MATH，采用三款开源指令调优模型（Gemma 3 12B IT、Qwen2.5 7B Instruct、Llama 3.1 8B Instruct）。

**📈 对比分析**

比较方法：在相同的固定审计预算（如5%/10%）和推理预算下，分别使用内部探针、黑盒token质量分数以及随机/oracle路由策略对答案进行排序。性能指标主要是错误召回率、未审计答案的错误率、AUROC和路由时的准确率提升。结果显示：在所有六种模型-数据集组合中，内部探针在5%审计预算下平均提高1.3–22.5个百分点的错误召回；在两项GSM8K路由实验中，探针在相同预算下比等成本随机路由提升2.2–3.0个百分点，且当判别器较弱时优于黑盒判别。

**⚠️ 局限性**

局限性包括：①探针需要在目标模型-数据集-推理协议上进行重新训练和验证，无法在协议变化（数据集、权重精度、生成预算）后直接迁移；②内部探针可能仅学习模型自我评估的信念而非客观正确性；③实验仅覆盖三款7–12B模型和两类数学数据集，未评估更大模型或非数学领域；④评估未考虑对抗性查询或持续的模型漂移；⑤生成结果截断、非解析答案的排除可能导致错误率估计偏低。

---

## 629. Multimodal Abstention That Preserves Acquired Correct Answers

**arXiv ID:** 2611.14311 | [PDF](https://arxiv.org/pdf/2611.14311v1)

**作者:** Jinyuan Zhang `[一作]` `[通讯]` (Hubei University), Jinyuan Zhang (Hubei University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种“Regression-Free Abstention (ReFrain)”方法，在任务适配后为多模态模型添加停答功能，且不丢失已学习的正确答案。

**💡 创新点**

创新点在于将停答视为受约束的优化问题，使用线性探针得分并在保留原始正确率的前提下进行阈值校准，从而实现零损失（或可控损失）下的最大正确停答率。

**🔧 技术方法**

核心技术包括：基于首词 logits 的线性探针、基于图像簇的阈值校准、确定性 veto 替换为 NOTA 选项、以及多种可替换的分数器（MSP、BICR、ACR-MCQ 等）和对比实验框架。

**📊 数据集**

使用 Qwen2.5-VL-3B 与 InternVL3-8B 作为基础模型，在包含 677 训练、4567 校准、10678 评估样本的多模态问答数据集；并在独立 AI2D 证明集上进行冻结后验证。

**📈 对比分析**

与 EmbLoRA、Plain Abstention SFT 等基线在相同答案 SFT 起点下比较；ReFrain 在零预算下保持所有 57 个 SFT 获得答案，正确停答率提升 7.30pp；在 0.5% 预算下达到 74.05% 正确停答率、8 次原始正确答案丢失，显著优于 EmbLoRA 的 64.65% 与 62 次丢失。

**⚠️ 局限性**

局限性包括：仅使用单一训练种子、有限的 677 条训练样本、仅在 AI2D 相关图形问题上验证、校准仅在有限簇上保证零损失（不保证泛化）、对其他数据集和更大规模模型的适用性尚待检验。

---

## 630. Directed Singular-Value Sparsifiers with $O(n \log n/ε^2)$ Arcs

**arXiv ID:** 2611.14342 | [PDF](https://arxiv.org/pdf/2611.14342v1)

**作者:** Emile Anand `[一作]` `[通讯]` (Georgia Institute of Technology), Emile Anand (Georgia Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了无权有向图的强奇异值稀疏化，提出了一种新的稀疏化方法，该方法能够在保持每个入度和出度精确的情况下，使用O(nlog(2n)/ε^2)条重加权的原始弧。

**💡 创新点**

创新点在于引入了投影方差界限，使得每个电阻片仅贡献其自身的Schur互补缺陷，从而使得矩阵差异工具能够应用而不损失共享顶点的影响。

**🔧 技术方法**

使用了随机化构造方法和稀疏Jacobi迭代来计算投影，同时结合了有效电阻分解和高斯矩阵界限。

**📊 数据集**

使用了无权有向图作为数据集，具体的输入数据集未详细说明，但涉及到的图的顶点数为n。

**📈 对比分析**

与Lau, Wang和Zhou的O(nlog^5 n/ε^2)支持界限进行了比较，证明了本研究的O(nlog(2n)/ε^2)支持界限是一个改进。性能上，随机化构造在几乎线性时间内完成，且输出的权重具有多项式位长度。

**⚠️ 局限性**

限制在于当前的算法是随机化的，且在实际应用中可能隐藏较大的多对数因子，未来的工作可以探索确定性算法或更小的多对数开销。

---

## 631. ForeseeDrive: Future-Token Conditioning for Anticipatory Generative Driving

**arXiv ID:** 2611.14286 | [PDF](https://arxiv.org/pdf/2611.14286v1)

**作者:** Shaokai Wu `[一作]` (Shanghai Jiao Tong University), Yue Ding `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了ForeseeDrive，一个基于未来token的接口，在生成候选轨迹前对生成器进行条件化，以提升预测和安全性；

**💡 创新点**

创新点在于利用标注好的未来轨迹生成查询索引的动作面向token，结合零token残差适配器实现无漂移的生成控制，并加入候选保持损失和TTC边界守卫，形成可部署的预测驱动生成方案；

**🔧 技术方法**

技术手段包括：未来token的教师-学生蒸馏、残差适配器的零token门控、候选保持损失、TTC边界安全守卫以及对DiffusionDrive等生成模型的后端集成；

**📊 数据集**

使用了NAVSIM、nuScenes和Bench2Drive三大公开基准数据集，并在训练阶段利用日志中的未来场景信息进行监督；

**📈 对比分析**

在三大基准上与最新生成规划器对比，NAVSIM上PDMS提升至89.2，nuScenes上平均L2为0.41m、碰撞率0.05%，Bench2Drive上驾驶分数76.5，整体表现超过或接近最强对手，且通过同基线诊断验证了零token不影响基础生成、对齐token对候选分布产生正向作用；

**⚠️ 局限性**

局限性在于生成时条件化对性能的提升相对有限，oracle最优质量混合，基准对比不具有因果性，安全性改进主要通过后处理守卫实现，且需要依赖有未来信息的训练日志。

---

## 632. LM-SimBench: A Unified Framework, Dataset, and Benchmark for Large Language Model Training Simulation

**arXiv ID:** 2611.14285 | [PDF](https://arxiv.org/pdf/2611.14285v1)

**作者:** Yu Dong `[一作]` (Southeast University), Zhiyuan Liu `[通讯]` (Southeast University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 LM‑SimBench，一个统一的框架、数据集和评测基准，用来评估分布式大型语言模型训练的模拟器。

**💡 创新点**

创新点包括：统一了模拟器的输入输出与功能阶段，提供多粒度（系统、组件、策略）评测任务，发布了覆盖五维并行度和激活重计算的多模型配置数据集，并实现跨平台校准与基于 NPU 预测的 GPU 配置选择。

**🔧 技术方法**

采用了分阶段模拟（编译、设备绑定、执行表示、成本估算、聚合）技术，结合 DP/TP/PP/CP/EP 并行、激活重计算、性能模拟与参数校准，并使用 MAE、sMAPE、Spearman/Kendall 等指标进行评估。

**📊 数据集**

使用了 23 种模型设置、10 个密集/MoE LLM 系列的 348 个可行配置，来自 Ascend 910B3 NPU 的训练记录；并包含 NVIDIA V100 与 Ascend NPU 的配对测量，用于跨平台对比。

**📈 对比分析**

通过匹配配置、rank 与组件级视图，对迭代时间、内存、组件成本和配置选择进行评估；在 Qwen1.5‑1.8B 上对 vTrain 与 Calculon 的对比显示 vTrain 的 sMAPE 1.86%（比 Calculon 的 8.88% 更低），Spearman ρ 为 1.000；vTrain 的模拟时间为 25.46 ms，Calculon 为 1.78 ms。

**⚠️ 局限性**

局限性包括：目前仅覆盖 Ascend NPU 与 V100 GPU，缺乏更大规模模型和多种硬件平台的评测；模拟器对 profiling 与手工校准的依赖较强，尚未处理动态调度和实时优化等复杂情况。

---

## 633. $\mathsf{SVP}_\infty$ Is NP-Hard for Power-of-two Cyclotomic Ideals

**arXiv ID:** 2611.14334 | [PDF](https://arxiv.org/pdf/2611.14334v1)

**作者:** Yansong Feng `[一作]` (Academy of Mathematics and Systems Science, Chinese Academy of Sciences), Yanbin Pan `[通讯]` (Academy of Mathematics and Systems Science, Chinese Academy of Sciences)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明了在系数ℓ∞-范数下，对二次幂阶循环多项式环 R_n=[X]/(X^n+1) 以及循环环 S_{2n}=[X]/(X^{2n}-1) 的主理想和自由模（秩为 2 甚至任意固定秩）最短向量问题（SVP）是 NP‑完整的；同时给出了从子集和（Subset‑Sum）到这类结构化格问题的确定性多项式时间多项式归约，构造的格维度为 O(s)（s 为输入整数的个数）。

**💡 创新点**

创新点在于：①首次给出从 Subset‑Sum 到这些代数结构化格问题的直接归约，证明了系数 ℓ∞-范数下的 SVP 在这些环上的 NP‑完整性；②提出了利用复系数表示和多项式环的代数结构来强制向量的系数满足布尔约束，从而将子集和嵌入到格的最短向量问题；③将结果从 cyclotomic 环扩展到更通用的循环环，并保留了线性维度和多项式时间的性质；④完成了对该证明的 Lean 形式化，为理论结果提供了可验证的实现。

**🔧 技术方法**

技术手段主要包括：多项式环（cyclotomic 与循环环）的代数性质、系数嵌入（coefficient embedding）、复系数（complex coefficient）表示、系数 ℓ∞-范数的约束与归约构造、对 Subset‑Sum 的巧妙编码、整数矩阵与多项式矩阵的构造、以及对最短向量与第一最小值的分析。

**📊 数据集**

该工作为纯理论研究，不使用实验数据集；所有结果均通过数学证明与算法构造获得。

**📈 对比分析**

由于研究的是理论复杂度问题，文中没有实验性能比较；但归约保证了构造的多项式时间复杂度和格维度线性（O(s)）以及搜索版本的 NP‑难度（通过多项式时间 Turing 归约）。

**⚠️ 局限性**

局限性包括：①仅适用于 n 为二次幂的循环多项式环和循环环；②仅针对系数 ℓ∞-范数，其他范数（如 ℓ2）尚未得到类似结果；③只证明了固定秩（尤其是秩 2）的自由模；④对更一般的代数结构（如非平凡分数域、非循环多项式）仍缺乏完整的归约。

---

## 634. Evaluating LLM-Generated World Models for Neuro-Symbolic Reasoning

**arXiv ID:** 2611.14419 | [PDF](https://arxiv.org/pdf/2611.14419v1)

**作者:** Souhaila Djaffal `[一作]`, Jobst Heitzig `[通讯]` (PRISM AI Safety Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究大型语言模型在将自然语言场景转换为正式世界模型（POSG）时的表现，并提出了一套包含机械有效性、结构一致性和语义诊断的三层评估框架。

**💡 创新点**

创新点在于构建了一个覆盖游戏、情境与文化三类场景的多层级基准，首次系统性探讨机械有效性与模型可信度之间的关系，并提出结构对齐与语义规则检查的组合方法。

**🔧 技术方法**

使用五种主流LLM（GPT‑5.6、Claude Opus 5、Llama 3.3 70B、Qwen 3.8 27B、DeepSeek V4 Flash）生成JSON/Python形式的POSG，并通过程序化验证、结构对齐（F1）以及语义规则匹配进行评估。

**📊 数据集**

数据集由44个多样化场景组成，分别包含10个玩具环境、9个已知规则的游戏、11个开放式情境以及14个文化背景情景，涵盖不同的多玩家和不确定性特征。

**📈 对比分析**

评估方法采用四个层级：Lane 1（机械验证）、Lane 2a（结构F1）、Lane 2b（无参考结构诊断）和Lane 3（语义检查）。实验结果显示机械通过率高（游戏134/135），但结构F1仅为0.366，说明机械有效的模型往往缺乏结构忠实度。

**⚠️ 局限性**

局限性包括基准规模有限、仅覆盖离散POSG、结构比较深度受限（仅3步）、缺乏完整的人类语义评估以及未评估生成模型对下游规划、可达性或赋权等安全量的影响。

---

## 635. A minimal intuitionistic public announcement logic

**arXiv ID:** 2611.14408 | [PDF](https://arxiv.org/pdf/2611.14408v1)

**作者:** Philippe Balbiani `[一作]` (Indian Institute of Technology Kanpur), Clara Lerouvillois `[通讯]` (Institut de recherche en informatique de Toulouse)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并研究了两种新的直觉主义公共公告逻辑 Ł_min 和 Ł_，通过引入二元算子 ▹（相对必然性）在无冲突（非前向/后向）双关系框架上实现公告的语义与归约。

**💡 创新点**

创新点在于摒弃传统的前向/后向共振假设，利用 ▹ 而非 □ 构建归约公理体系，并分别在所有双关系模型和等价关系模型上证明完整性。

**🔧 技术方法**

技术方法包括：定义新的语义算子 ▹、构造相应的归约公理、使用 Zorn 引理构建最大理论、构造 Canonical 模型并证明真值引理，从而完成完整性证明。

**📊 数据集**

该工作为纯理论研究，未使用任何实验数据集；所有结论均来自形式化证明。

**📈 对比分析**

由于论文为理论性研究，未涉及实验比较；通过形式化证明展示了逻辑的一致性、可判定性及相对于传统公共公告逻辑的严格包含关系。

**⚠️ 局限性**

局限性包括：仅覆盖单代理情境，未考虑多代理、公共/分布知识以及动态成功公告的性质；此外，对 ▹ 的可扩展性（如加入其对偶 ◃）仍待进一步探讨。

---

## 636. Discovering Equations for the Thermodynamics of South Asian Monsoon Precipitation

**arXiv ID:** 2611.14401 | [PDF](https://arxiv.org/pdf/2611.14401v1)

**作者:** Savannah L. Ferretti `[一作]` (University of California, Irvine), Tom Beucler `[通讯]` (University of Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了印度次大陆季风降水的热力学控制，并通过集成核学习与符号回归结合的符号蒸馏方法，提出了可解释且参数量极低的降水方程。

**💡 创新点**

创新点在于让神经网络自主学习垂直结构权重，再用符号回归提取闭式方程，既保留约85%全网络预测性能，又将参数量压缩约千倍，揭示湿度与不稳定性竞争决定降水的机理。

**🔧 技术方法**

使用集成核学习（integration kernel learning）与 PySR 符号回归，以及标准全连接神经网络进行对比。

**📊 数据集**

使用 ERA5 2000–2020 年 6–8 月的再分析数据，包括表面变量（陆分、潜热通量、感热通量）和 1000–500 hPa 的相对湿度、等势温度及饱和等势温度等垂直剖面。

**📈 对比分析**

与仅使用 B_L 诊断、完整神经网络以及不同特征组合的模型比较，符号方程在测试集上的 R²≈0.42（约为全网络的70%），参数量仅为全网络的千分之一，说明符号方程在捕捉湿度与不稳定性竞争关系方面效果显著。

**⚠️ 局限性**

局限在于对陆地降水的昼夜循环及强降水的解释不足，模型仅基于共时热力学状态，难以充分考虑地形、动力学及大尺度气流的影响。

---

## 637. Forward-Value Discretization: Faster Stochastic Diffusion Sampling Without High-Order Approximation

**arXiv ID:** 2611.14328 | [PDF](https://arxiv.org/pdf/2611.14328v1)

**作者:** Yuchen Jiao `[一作]` (Chinese University of Hong Kong), Gen Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种前向值离散化框架，并基于此设计了快速的确定性采样器F-DPMSolver和快速的随机采样器F-DDPM，二者都仅使用一次模型评估即可完成每一步更新；

**💡 创新点**

创新点在于将模型评估位置从当前时刻改为前瞻估计，从而在保持一阶收敛的前提下，在极少步数下显著提升样本质量，并证明了随机采样器SDE‑DPMSolver++2在强收敛率上仍为一阶，揭示了随机采样器加大离散化阶数并不能提高收敛阶的根本原因；

**🔧 技术方法**

使用的技术主要包括时间重参数化、噪声/数据预测器、SDE/ODE反向过程离散化以及前向值预测的构造；

**📊 数据集**

实验采用CIFAR‑10、ImageNet、LSUN、FFHQ四个数据集，并在这些数据集上评估F-DPMSolver和F-DDPM的性能；

**📈 对比分析**

与DDIM、DPMSolver‑2/3、UniPC‑3等现有采样器以及SDE‑DPMSolver++1/2比较，F-DDPM在相同NFE下的FID明显优于随机基线，而在使用1.5倍NFE的情况下，F-DDPM的FID可与确定性采样器相近或更好；

**⚠️ 局限性**

局限性包括：采样器的收敛阶仍停留在一阶，无法解释前向值评估为何在极少步数时提升样本质量；对指导（classifier/ classifier‑free）以及逆问题（如去模糊、超分辨率）的适用性尚未深入研究。

---

## 638. An Energy-Based Perspective on Machine Learning

**arXiv ID:** 2611.14383 | [PDF](https://arxiv.org/pdf/2611.14383v1)

**作者:** Bo Dai `[一作]` (Georgia Institute of Technology), Yilun Du `[通讯]` (Harvard University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文综述了能量基模型（EBM）在机器学习中的统一视角，阐明能量函数、推断过程与学习方法三者如何构成 EBM，并将判别、生成、表示学习、强化学习等多领域模型归纳为 EBM 的特例。

**💡 创新点**

创新点在于：① 将多类模型统一为能量框架，揭示学习算法（最大似然、NCE、伪似然、评分匹配、对抗训练、回归训练等）本质上是对不同负样本采样器的特化；② 提出能量组合（乘积、混合、否定）实现可复用与推断时组合的原则；③ 强调能量的可组合性与适应性，为新算法设计提供了模块化思路。

**🔧 技术方法**

技术手段包括 Metropolis–Hastings / Gibbs / Langevin / Hamiltonian MC 等 MCMC 推断；变分推断与自编码器；对比学习框架（MLE、NCE、伪似然、评分匹配、对抗训练、回归训练）；以及与流模型、扩散模型、GAN、Hopfield、深度平衡模型等生成方法的映射。

**📊 数据集**

论文主要是综述性工作，引用的示例实验多使用公开数据集如 MNIST、CelebA、CIFAR 等；并无新的数据集或大规模实验。

**📈 对比分析**

通过理论分析与小规模实验比较，论文指出：① EBM 能处理无归一化分布且易组合，但采样成本高；② GAN 在采样上最快但易模式崩溃；③ 流模型可实现一次性采样且具可解释性；④ 扩散模型在高质量生成上表现优异但需多步采样；总体而言，不同方法在训练难度、采样效率、能量尺度敏感性等方面存在显著差异。

**⚠️ 局限性**

局限性包括：① 计算分区函数不可求，导致采样与学习成本高；② 能量组合对尺度和正则化要求严格，易产生不稳定；③ 负样本或噪声设计不易，影响学习效果；④ 对离散高维数据的可扩展性不足；⑤ 在推断时的组合与优化仍是实用瓶颈。

---

## 639. Point-in-Time Encoders for Earnings Calls

**arXiv ID:** 2611.14325 | [PDF](https://arxiv.org/pdf/2611.14325v1)

**作者:** Alessio Brini `[一作]` `[通讯]` (Duke University), Alessio Brini (Duke University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究并构建了按时间点切分的财经电话会议检索编码器（vintages），并将其用于检索问答、语言模型回答及收益预测。

**💡 创新点**

创新点在于利用带时间戳的文本窗口训练年度“vintages”，通过问答对进行对比学习，系统性解决检索与语言模型中的前瞻偏差，并对不同预训练策略的检索与预测效果进行对比分析。

**🔧 技术方法**

技术手段包括 ModernBERT 架构的掩码语言模型预训练、InfoNCE 对比微调、连续预训练（chaining）、域适应预训练（DAPT）、检索增强的语言模型、岭回归等。

**📊 数据集**

实验使用 2004‑2026 年的 Capital IQ 交易电话会议记录（超过 16,000 家上市公司），对每个季度随机抽取 10% 作为 hold‑out，提取问答对作为训练与评估样本。

**📈 对比分析**

评估方法：在 hold‑out 召回@1、跨公司召回上与 bge、e5、FinBERT 等通用检索器进行对比；在收益预测中计算 IC 与 Sharpe 比例；通过将 2024 年 vintages 替换原有 vintages 的方式测量 look‑ahead premium，结果显示检索召回率提升 10‑15% 以上，IC 提升约 0.005‑0.01。

**⚠️ 局限性**

局限性包括：实验仅基于单一电话会议语料；模型规模有限（最多 33.8M 参数）；问答对来源有限，难以推广至其他文本类型；收益预测提升有限，主要体现在短期收益上；未来需要验证更大规模模型与更广泛语料的效果。

---

## 640. TaoMate-R1: Temporal Role Specialization for Streaming Multi-Reference Video Generation

**arXiv ID:** 2611.14307 | [PDF](https://arxiv.org/pdf/2611.14307v1)

**作者:** Qijun Gan `[一作]` (Alibaba Group), Junfeng Ma `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于时间角色专门化的流式多引用视频生成框架 TaoMate‑R1；

**💡 创新点**

创新点在于将低秩适配器按“场景建立”与“场景延续”划分，并用 Self‑Gradient‑Forcing（SGF）把这两类角色通过生成历史相互耦合，区别于传统按运算分离的专门化方式；

**🔧 技术方法**

技术手段包括冻结大规模扩散基底，使用 LoRA 低秩适配器，学生回放训练、分布匹配蒸馏、历史梯度回传（SGF）以及三步 Euler 采样；

**📊 数据集**

训练数据为 2,196 条 4–15 秒的多引用视频样例（涵盖对象保留、移除、引入），评估集包含 10 条约 2 分钟的长流、10 条短交互和 12 条公开参考合成；

**📈 对比分析**

在同一两适配器预算下，与共享适配器和按操作拆分的 SGF+ 进行对比，TaoMate‑R1 在 CLIP/DINO 时序相似度上提升 0.006–0.01，边界亮度跳跃下降 68.7%，整体流畅度和参考一致性明显优于对照组；

**⚠️ 局限性**

局限性包括：仅在 H3 基底上验证，适配器数量受限，音频同步仍略逊于传统模型，且对极端指令变化或高复杂度场景的鲁棒性尚未充分评估。

---

## 641. REAL Loss: Optimizing Real Expected Acceptance Length for Speculative Decoding

**arXiv ID:** 2611.14399 | [PDF](https://arxiv.org/pdf/2611.14399v1)

**作者:** Yu Tian `[一作]` `[通讯]` (University of Hong Kong), Yu Tian (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于无偏分支估计器的真实接受长度（J）训练损失，用于提升链式投机解码的速度；

**💡 创新点**

创新点在于揭示并修正现有块级近似目标的协方差偏差，构造无偏路径权重估计并利用 Rao–Blackwellization 降低方差；

**🔧 技术方法**

使用无偏路径估计、Rao–Blackwell化、交叉熵辅助项、以及与目标模型的对齐采样；

**📊 数据集**

采用 Qwen3.6‑35B‑A3B 作为目标模型、8‑token  Drafter（1 anchor + 7 推测）以及 Open‑PerfectBlend 训练数据，评估 9 个任务子集；

**📈 对比分析**

与 LK、WTV、VLB、FReDraft、DBLast 等基线相比，Real‑J 在概率式服务下平均接受长度达到 4.792，优于其他方法（4.657–4.749），在贪心服务下亦保持领先；

**⚠️ 局限性**

局限性在于仅在单一目标模型、单一 Drafter 体系和单一数据集上验证，缺乏跨模型/数据的更广泛评估；

---

## 642. EngineAgent: Agentizing On-Device LLM Inference Engines with Inference Stack Hierarchy Awareness and Hardware Abilities

**arXiv ID:** 2611.14356 | [PDF](https://arxiv.org/pdf/2611.14356v1)

**作者:** Zhengxiang Huang `[一作]` (Shanghai Jiao Tong University), Guihai Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了EngineAgent，自动化在移动设备多后端上启用新LLM模型的全过程。

**💡 创新点**

创新点在于将引擎层次结构知识与硬件特定调试工作流拆分为可重用组件，并通过Builder/Debugger/Auditor三代理循环实现跨层次实现与设备验证。

**🔧 技术方法**

技术包括CodeGraph构建层次索引、硬件-aware构建/调试技能、三代理（Builder、Debugger、Auditor）协同、以及使用DeepSeek V4-Flash和GPT-5.6 Sol进行编码与推理。

**📊 数据集**

使用12个模型启用任务（Qwen3.5‑2B、Gemma4‑E2B‑it、MiniCPM‑V 4.6）和四个引擎‑后端组合（MNN‑CPU、llama.cpp‑Vulkan、MLC‑LLM‑OpenCL、ExecuTorch‑QNN）进行评估。

**📈 对比分析**

与SWE‑Agent、OpenCode、CodeGraph+OpenCode+AgenTracer以及Codex基线对比；在DeepSeek V4-Flash下EngineAgent实现75%任务成功、平均3.82h、$1.53的成本，优于基线的33.3%成功率和更高的时间/成本。

**⚠️ 局限性**

局限在于对极端新特性导致的图层级或硬件内核实现缺失仍有挑战；缺乏对后续模型版本持续自动化的完整评估；对不同硬件工具链的兼容性依赖于预构建技能。

---

## 643. Can AI be a 'Researcher Without Borders'? Auditing Bias in AI-Assisted Social Science Research about China and the US

**arXiv ID:** 2611.14288 | [PDF](https://arxiv.org/pdf/2611.14288v1)

**作者:** Yuan Zhang `[一作]`, Ralph Schroeder `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对五个不同国家开发的LLM（两美国、两中国、一个欧洲）在四种提示方式下，针对十个社会科学问题生成200篇学术式答案，并由两名专家手工标注答案中的国家偏向、理论框架和引用来源。

**💡 创新点**

首次系统结合专家标注与来源分类，揭示LLM在跨国社会科学研究中的“国家偏向”与“学术偏见”，并发现提示身份与问题类型显著影响答案的论证框架和来源多样性。

**🔧 技术方法**

使用OpenRouter API调用LLM并启用web_search工具；人工标注偏向和来源类别；采用卡方、Fisher检验和Shannon多样性指数进行统计与可视化分析。

**📊 数据集**

采用自制的十个社会科学问题（包含中美对比、单国和通用题），共生成200篇答案；每篇答案中的引用被手工归类为学术、政府、国际组织、智库、媒体、商业、其他组织或个人等八类。

**📈 对比分析**

通过交叉比较模型、提示与问题类型的来源分布和偏向标签，并用Cohen's κ评估标注一致性；结果显示总体无显著国家偏向，但某些对比问题存在轻微倾向，且不同提示和模型在来源多样性上差异显著。

**⚠️ 局限性**

受限于问题数量、模型与提示设置的有限性，未能探究偏向产生的因果机制；未测试具体的偏见缓解方法；标注者主观性可能影响结果；未覆盖所有LLM版本与检索系统。

---

## 644. Gaussian Process-Augmented Model Predictive Contouring Control for 6-DoF Trajectory Tracking with a Holonomic AUV

**arXiv ID:** 2611.14373 | [PDF](https://arxiv.org/pdf/2611.14373v1)

**作者:** David Dörner `[一作]` (KTH Royal Institute of Technology), Ivan Stenius `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种6自由度的MPCC控制器，并通过GP学习残差动力学改进海底多功能AUV的轨迹与姿态跟踪

**💡 创新点**

首次在全六自由度的holonomic AUV上实时部署MPCC，并结合GP残差学习来补偿未建模动力学，且只学习有意义的角度残差以提升性能

**🔧 技术方法**

使用非线性MPCC、Gaussian Process残差建模、零阶鲁棒优化与实时SQP求解

**📊 数据集**

训练数据来自在仿真Stonefish与真实Mola AUV中进行的开放环激励实验（约100点/GP），实验使用Mola AUV在13×9×10 m盐水试验池

**📈 对比分析**

与基准MPCC和经验调参的PID比较，GP-角度残差MPCC在姿态RMSE下降约50%，位置RMSE基本保持不变，控制能量略高，且在复杂螺旋巡检任务中能成功完成而PID失败

**⚠️ 局限性**

对全残差学习导致噪声放大、控制能量增加的缺点；需要在线学习与不确定性传播；实验规模受限，未验证长期部署与障碍物回避能力

---

## 645. How Well Do SAST Tools Work Across Languages? An Empirical Study Using OCPP Implementations

**arXiv ID:** 2611.14362 | [PDF](https://arxiv.org/pdf/2611.14362v1)

**作者:** Zachary Wadhams `[一作]` (Montana State University), Ann Marie Reinhold `[通讯]` (Montana State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 OCPP 1.6 开源实现的静态安全测试工具在不同编程语言中的表现。

**💡 创新点**

首次系统评估多语言 SAST 工具在电动车充电协议实现中的检测差异，揭示语言生态对工具效果的影响。

**🔧 技术方法**

使用 SonarQube、Semgrep 以及各语言专用分析器（如 Bandit、Cppcheck 等）进行静态分析。

**📊 数据集**

选取 12 种不同语言的 OCPP 1.6 实现作为测试集。

**📈 对比分析**

通过比较工具在同一实现上的发现数量和类型，发现 Java、C 等成熟语言产生大量检测结果，而 Rust 仅两条；总体检测总量 4,660 条。

**⚠️ 局限性**

仅采用默认配置、单一实现每语言且未验证误报率，限制了结果的准确性和泛化性。

---

## 646. Reliable and explainable analysis of AVS/RS by Statistical Model Checking and Process Mining

**arXiv ID:** 2611.14371 | [PDF](https://arxiv.org/pdf/2611.14371v1)

**作者:** Emanuele Guerrazzi `[一作]` (Sant'Anna School of Advanced Studies), Andrea Vandin `[通讯]` (Sant'Anna School of Advanced Studies)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一套将离散事件仿真与统计模型检测（SMC）和流程挖掘（PM）相结合的分析框架，用于评估自动化车辆存取系统（AVS/RS）的性能。

**💡 创新点**

创新点在于首次将SMC和PM联合应用于AVS/RS，通过SMC获得统计可靠的性能指标，并利用PM从仿真日志中抽取行为模型，既保证了结果的可信度，又提升了可解释性。

**🔧 技术方法**

使用的技术包括MATLAB实现的离散事件仿真模型、MultiVeStA工具进行黑箱SMC分析，以及Disco等流程挖掘工具对日志进行过程映射与瓶颈分析。

**📊 数据集**

数据集采用根据行业文献设定的参数生成的合成订单和需求分布（指数、泊松、正态分布），并基于开放源码AVS/RS仿真器产生事件日志。

**📈 对比分析**

通过对随机、最深和最近三种库存与拣选策略进行多点t检验和置信区间分析，结果显示“最近”策略在拣选/存储周期时间上最快（约30%提升），但空间利用率最差；“最深”策略在空间利用率上表现最佳，周期时间与随机相近。

**⚠️ 局限性**

局限性包括仅在有限的72小时仿真窗口内评估、使用合成数据而非真实运营日志，且未考虑季节性需求波动和长期性能衰减等实际运营因素。

---

## 647. CLADE: Category-Level Answering from Deficient Evidence in Retrieval-Augmented Language Models

**arXiv ID:** 2611.14355 | [PDF](https://arxiv.org/pdf/2611.14355v1)

**作者:** S M Rayeed `[一作]` (Rensselaer Polytechnic Institute), Yao Ma `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究检索增强系统中检索到的文本仅给出答案的更高层级类别时，模型会继续给出具体答案而非放弃，导致准确率下降，并提出利用本体将粗糙证据转化为精确答案的流程。

**💡 创新点**

提出“可解析不足”概念及其导致的“基于缺失证据的类别级回答”现象，并系统评估利用本体恢复精确答案的可行性，揭示模型在面对粗粒度证据时缺乏自我放弃的能力。

**🔧 技术方法**

使用大型语言模型（Qwen、OLMo、Granite）进行检索增强推理，结合对检索文本的粒度分类、对比实验、逆因果编辑、置信度过滤、规模扩展和指令微调，并使用本体（MONDO、GloBI taxonomy）生成候选列表。

**📊 数据集**

主要使用PrimeKG（药物-疾病关系）与GloBI（物种交互关系）两个知识图谱，结合Wikipedia和PubMed文本，构造约1800条PrimeKG事实和45万条GloBI交互。

**📈 对比分析**

实验显示：在PrimeKG中，家族级检索文本将模型放弃率从≈28.5点降低至≈14.7点，导致正确率从≈90%降至≈50%；利用本体后，在410条家族级事实上，答案在候选列表中的覆盖率达98.5%，端到端恢复率从10%提升至26.8%。在GloBI中效果相似。

**⚠️ 局限性**

限制包括仅测试两大领域、仅使用多项选择题形式、模型规模上限为14B、对本体的字符串匹配依赖度高、以及对家族级对抗样本的覆盖率有限。

---

## 648. Bits or permutations? Spending a fixed inference budget on weight precision or on option-order averaging in small local language models

**arXiv ID:** 2611.14413 | [PDF](https://arxiv.org/pdf/2611.14413v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 649. AI Isn't Getting Dumber, We're Getting Pickier: Rising Standards in Human Evaluation of Large Language Models

**arXiv ID:** 2611.14333 | [PDF](https://arxiv.org/pdf/2611.14333v1)

**作者:** Zihan Chen `[一作]` (Stevens Institute of Technology), Di Zhu `[通讯]` (Stevens Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文利用Chatbot Arena的公共对比数据，系统分析并发现即使模型本身保持不变，用户对固定模型的绝对评判（“两者都差”）随时间逐步变得更严苛；

**💡 创新点**

创新之处在于首次将评价偏差归因于评估者标准随类别进步而上升（适应水平理论），并证明通过保持冻结参考模型可有效去除这一漂移；

**🔧 技术方法**

主要采用线性概率模型、配对固定效应、双重聚类标准误、以及多层次控制（提示、对话、模型配置、界面变更）等统计方法进行因果推断；

**📊 数据集**

数据来源为LMSYS公开的Chatbot Arena战斗日志（2023年5月至2024年8月），共计数百万条投票，筛选出仅包含冻结版本模型对的对照记录；

**📈 对比分析**

在对“都差”投票率进行时间趋势回归后，发现每月平均上升约0.1–0.2个百分点，且相对排名（较强模型获胜率）保持稳定，显示绝对评判漂移显著而相对评判无显著变化；

**⚠️ 局限性**

局限性包括样本为自愿的公开评测用户、缺乏提示和答案文本信息、无法精确捕捉评估者对其他AI的曝光情况，且结果对一般人群的外推性尚需进一步验证。

---

## 650. Contrastive Discovery: Open-Ended Scientific Discovery over Competing Explanations

**arXiv ID:** 2611.14514 | [PDF](https://arxiv.org/pdf/2611.14514v1)

**作者:** Ziang Liu `[一作]` (Cornell University), Jennifer J. Sun `[通讯]` (Cornell University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Discovery框架，将单一假设替换为包含互斥且穷尽的对照集，从而让发现过程以科学问题及其备选解释为单位；

**💡 创新点**

创新点在于明确构建对照集，利用贝叶斯惊讶量衡量证据对这些备选解释的信念转移，既能判定假设被否定，也能指出哪一备选解释得到支持，从而显著提升发现效率；

**🔧 技术方法**

技术包括：LLM驱动的对照集生成与信念挖掘、常数和分配任务实现信念先验与后验、基于Kullback–Leibler散度的对照信念转移评分、蒙特卡洛树搜索（MCTS）与UCT策略用于实验规划；

**📊 数据集**

实验使用六个多样化数据集：Materials Project（材料科学）、SDSS Galaxy（天体物理）、BLADE Mortgage（抵押贷款歧视）、BLADE Conversation（在线对话）、DiscoveryBench Requirements Engineering（软件工程）和DiscoveryBench NLS Raw（社会行为）；

**📈 对比分析**

与仅使用惊讶度、创新度或可行度评分的基线相比，Discovery在相同实验预算下将已解决问题数提升约67%，在六个数据集上平均提高了约25%（从基线的约42%到Discovery的约68%），并在专家盲测中获得78%的解释一致性；

**⚠️ 局限性**

局限性包括：仅在表格观测数据上验证，结论为统计关系非因果；对照集可能缺失或构造不当，需要专家审核；依赖LLM进行信念挖掘，结果受模型偏差影响；

---

## 651. PSL-RGHA: Parallel Split Learning with Reference Gradient Half-Space Alignment for Mitigating Gradient Divergence

**arXiv ID:** 2611.14312 | [PDF](https://arxiv.org/pdf/2611.14312v1)

**作者:** Han Li `[一作]` (Xiamen University of Technology), Shunzhi Zhu `[通讯]` (Xiamen University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出PSL‑RGHA框架，在无需客户端聚合的并行分割学习中通过参考梯度半空间对齐机制缓解梯度冲突，提升非IID环境下的收敛稳定性和精度。

**💡 创新点**

创新点在于结合参考梯度逐步构造（RGPC）与梯度半空间对齐（GHA）两种服务器端机制，实现不依赖客户端聚合的全局梯度一致化。

**🔧 技术方法**

采用梯度一致性打分、Weiszfeld迭代构建参考向量、投影对齐以及多任务联邦/分割学习框架。

**📊 数据集**

在FMNIST、CIFAR‑10、CIFAR‑100、Speech Commands和SNIPS等标准视觉/语音数据集上进行实验。

**📈 对比分析**

与SFL、PSL、EPSL等基线对比，在IID与非IID、不同客户端数量与数据异质性设置下，PSL‑RGHA在收敛速度与最终准确率上均优于基线，尤其在极端非IID（α=0.1）下提升约5‑10%。

**⚠️ 局限性**

局限性包括仅在同步批训练下验证，异步环境与大规模基础模型调优仍待扩展；对梯度计算开销与服务器资源有一定额外需求。

---

## 652. Adversarial risk analysis with fully probabilistic designs

**arXiv ID:** 2611.14289 | [PDF](https://arxiv.org/pdf/2611.14289v1)

**作者:** Mario Chacón-Falcón `[一作]` (Institute of Mathematical Sciences), David Rios Insua `[通讯]` (Institute of Mathematical Sciences)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种将全概率设计（FPD）与对抗风险分析（ARA）相结合的方法，利用攻击者的理想概率分布而非显式效用函数来预测其行动，并给出单阶段与多阶段博弈的显式解法和后验学习算法。

**💡 创新点**

创新点在于：① 用KL散度将攻击者的理想分布与实际分布对齐，避免了对攻击者效用函数的复杂设定；② 推导出攻击者最优策略的显式解析解，消除了传统ARA中需要大量仿真-优化循环的计算负担；③ 通过递归的前向后向算法实现多阶段博弈的策略优化，并引入贝叶斯学习更新攻击者理想分布。

**🔧 技术方法**

主要技术包括：全概率设计（FPD）、KL散度与信息论方法、贝叶斯推理、蒙特卡洛模拟、影响图（BAID）建模、单/多阶段决策优化、回溯学习算法。

**📊 数据集**

实验使用人工合成数据，主要包括机场武器检测场景和空中作战（无人机攻击）两阶段博弈，数据来源于论文中给出的表格与参数设置。

**📈 对比分析**

与传统ARA和最小化极值（minimax）方法相比，FPD-ARA在相同情境下能更快得到最优防御策略，且在所示案例中避免了过度保守的资源配置；同时显式解法显著降低了计算成本；实验结果表明，当攻击者理想分布已知时，FPD-ARA能更准确地预测攻击概率并提升防御收益。

**⚠️ 局限性**

局限性包括：① 对于结果空间维度较大时，需要为每个结果诱导理想概率，仍存在维度灾难；② 需要先验假设攻击者按KL最小化选择行动，若攻击者行为更复杂则模型偏差；③ 仅考虑单层递归思考，无法捕捉更深层的策略嵌套；④ 学习过程依赖于足够的历史交互数据，数据不足时估计不稳定。

---

## 653. Toward Foundation Detection Models via Example-Based Grounding in EM-DETR

**arXiv ID:** 2611.14518 | [PDF](https://arxiv.org/pdf/2611.14518v1)

**作者:** Sheethal Bhat `[一作]` (Friedrich-Alexander-Universität), Andreas Maier `[通讯]` (Friedrich-Alexander-Universität)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了基于EM-DETR的多模态医学检测基础模型，通过学生-教师方法融合解剖与异常数据，实现对胸部X光和乳腺摄影的高精度定位与分类。

**💡 创新点**

引入例子驱动的EM-DETR框架与学生-教师弱标注联合，解决异构标签空间与灾难性遗忘问题，并在多模态数据上实现SOTA检测。

**🔧 技术方法**

使用EM-DETR、GDINO预训练、Swin Transformer骨干、学生-教师伪标签、对比学习与特征损失等技术。

**📊 数据集**

使用Dataset A、B、C（胸片解剖与异常）、VinDR-CXR、VinDR-Mammo共四个跨域数据集。

**📈 对比分析**

与商业Reference Product Solution、GDINO基线及EM-DETR单数据集比较，AUC平均提升1%，mAP_50在CXR提升至0.567、乳腺提升至0.721，显著优于SOTA。

**⚠️ 局限性**

对某些模糊标签（如肺不张、吸气性萎陷）的检测仍低；学生-教师阈值敏感，训练稳定性受类分布影响；仅覆盖胸片与乳腺，未验证其他影像模态。

---

## 654. Second-Best Trade Is Half Efficient in Multidimensional Markets

**arXiv ID:** 2611.14489 | [PDF](https://arxiv.org/pdf/2611.14489v1)

**作者:** Dirk Bergemann `[一作]`, Renzhe Xu `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明在任意多维两侧市场（买卖双方私有价值与成本可多维且内部相关，且满足下闭的可行性约束）中，存在一个满足贝叶斯激励兼容、间接可行性以及强预算平衡的直接机制，其实现的交易收益至少是第一最佳收益的一半。

**💡 创新点**

创新点在于：①通过对Cai–Devanur–Weinberg对偶框架的全新应用，得到一个关于所有可行对偶解的统一下界；②将对偶中的流量变换与连续时间马尔可夫链相连接，利用指数时钟的恢复性质将虚值映射回真实值；③通过“密度上界”和“负相关时钟”技术，在多代理、多维情形下仍保持仅损失一倍的因子，完成了对偶目标与第一最佳之间的比值证明。

**🔧 技术方法**

核心技术包括：多维机制设计的线性规划与对偶推导；对偶约束构造的流量模型；连续时间马尔可夫链的逆矩阵表示（Markov恢复过程）；指数分布时钟与贝塔积分上界；以及在可行性多闭集合上利用向下闭特性进行稀释与抽样。

**📊 数据集**

论文不依赖于任何外部数据集；所有证明均为理论性，使用了随机分布模型和先验概率。

**📈 对比分析**

比较方法：直接对比第二最佳（满足贝叶斯IC、IR、SBB）的期望收益与第一最佳（仅受可行性约束的最大期望收益）。性能方面，证明得到的下界为1/2，且该比例在双边交易场景已被证明是紧的，因而在更一般的多维两侧市场中该比值也是最优的。

**⚠️ 局限性**

限制主要包括：①需假设可行性集合是下闭且非空；②机制要求满足贝叶斯激励兼容、间接可行性和强预算平衡，无法直接推广到强制支付或可执行性更严格的情形；③在实际应用中实现此类机制的计算复杂度可能较高，且对偶流量构造的具体实现未给出算法细节。

---

## 655. Mechanistic Interpretability Needs a Logic of Inference

**arXiv ID:** 2611.14483 | [PDF](https://arxiv.org/pdf/2611.14483v1)

**作者:** Zhimin Hu `[一作]` (Georgia Tech), Sashank Varma `[通讯]` (Georgia Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一套由五个原理组成的递归推理框架，用于从局部解释结果（如神经元、特征、电路等）出发，逐步构建更完整、更可检验的机制解释，并讨论了其在模型内部和模型外部的科学发现潜力。

**💡 创新点**

创新点在于：①将神经科学与机制解释（MI）的研究路径对照并提炼出共同的推理瓶颈；②系统化归纳出五条原则（P0–P5），明确每一步从观察到机制主张的逻辑；③将解释对象、信息利用、因果角色、组织结构以及跨条件不变性四个维度串联成递归思路，提供了一种可循环检验、不断精炼机制的范式。

**🔧 技术方法**

使用的技术主要是机制解释相关的方法（神经元/特征可视化、稀疏自编码器、注意力头/电路构建、群体几何/动力学分析）以及干预手段（消融、补丁、定向操作），并结合实验设计与统计检验来评估信息是否被使用、因果关系是否成立等。

**📊 数据集**

本文并未针对具体数据集进行实验，而是以已有研究中的案例（如视觉CNN、语言模型、神经元实验）为例子进行概念阐释和推理逻辑演示；因此数据集未被明确列出。

**📈 对比分析**

文中并未给出针对单一方法的性能指标或与其他方法的对比实验；主要侧重于理论框架和推理流程的描述，强调在不同层级和不同测量条件下进行互补验证。

**⚠️ 局限性**

主要局限包括：
1. 需要外部、领域特定的验证来证明模型机制对真实世界的通用性；
2. 解释结果高度依赖测量方案，测量偏差可能导致误判；
3. 递归框架虽系统但在具体实验设计中仍需经验判断，可能缺乏自动化工具；
4. 对跨模型或跨任务的一致性与可迁移性的实证支持不足。

---

## 656. When Does the Real Wiring Win? Testing connectome structure-function claims against unmeasured physiology

**arXiv ID:** 2611.14508 | [PDF](https://arxiv.org/pdf/2611.14508v1)

**作者:** Yeyuan Liu `[一作]` `[通讯]` (University of Virginia), Yeyuan Liu (University of Virginia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在神经网络模型中将实际神经连接图与随机或重排连接图进行对比时，参数调优、模型适配和实验数据约束如何导致结构-功能结论偏差，并在果蝇的迷宫体、头方向环吸引子和视觉系统等三个电路上检验。

**💡 创新点**

提出了三种公平比较协议（P1、P2、P3）来分离调优偏差、模型不充分和数据约束三种来源的误差；量化了选择偏差和权重遗传效应；给出了理论预测与合成数据的闭式解；通过实验验证了哪些结构主张在不同协议下保持稳健，哪些被逆转；并指出单一未测参数（如嗅觉反应KC比例）决定了一个关键主张的方向。

**🔧 技术方法**

使用基于FlyWire v783 的电路模型、随机参数搜索、梯度下降和重启训练、Sobol敏感性分析、拉丁超立方采样、占位子 null（placebo null）以及理论公式来评估偏差。

**📊 数据集**

利用果蝇完整连接图（FlyWire v783）、高分辨率嗅觉和视觉行为实验数据（如Calcium imaging、Sintel 视觉流数据）、以及已有的实验测量（如亚类型KC活性比例、单细胞多巴胺响应分布）。

**📈 对比分析**

对比方法：在同一参数集上评估实际连接图与多种 null 组合的任务性能，分别采用 P1（共享调优）、P2（独立调优）和 P3（受实验数据约束的最佳调优）。性能表现：某些主张（如多巴胺分离、头方向环吸引子持久波）在所有协议下保持正向优势；但迷宫体的 KC→MBON 连接主张在 P3 时被逆转，显示模型在实验约束下失效；P1 的优势常被占位子 null 的选择偏差放大。

**⚠️ 局限性**

局限性包括：模型仅为速率/传递式，未捕捉发放或更细致的生理机制；空间约束的 null 仍为近似；未覆盖所有可能的生理参数空间；P3 仅对单一未测参数有效；对训练预算的敏感性（如 flyvis 的 warm‑start 训练）未得到完全解析；统计检验未对所有 θ 进行多重比较校正；实验数据集可能不完全匹配模型假设。

---

## 657. A note on the intractability of enumerating the maximal satisfiable subformulas of a Horn expression

**arXiv ID:** 2611.14471 | [PDF](https://arxiv.org/pdf/2611.14471v1)

**作者:** Oscar Defrain `[一作]` (Aix-Marseille Université), Kazuhisa Makino `[通讯]` (Kyoto University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275`

**🎯 论文内容**

本文证明在Horn表达式中枚举所有最大可满足子公式（MSS）不可能在输出多项式时间内完成，除非P=NP。

**💡 创新点**

首次将输出多项式时间的上限与P=NP联系起来，给出了Horn公式MSS枚举问题的理论下限。

**🔧 技术方法**

采用多项式时间归约、构造Horn表达式、最大子公式可判定性分析等理论技术。

**📊 数据集**

无实际数据集，全部为理论构造与证明。

**📈 对比分析**

未进行实验比较，主要通过数学证明说明无法实现输出多项式时间算法。

**⚠️ 局限性**

仅针对Horn表达式的通用情况，未给出特殊子类的上界；缺乏对实际应用或更广泛约束的讨论。

---

## 658. Measuring Robustness and Efficiency in a Connectome-Constrained Fly Visual System Model: Linking an Adversarial and a Perceptual Illusion

**arXiv ID:** 2611.14498 | [PDF](https://arxiv.org/pdf/2611.14498v1)

**作者:** Alexandre Garcia-Duran `[一作]` (Centre de Recerca Matemàtica), Janne K. Lappalainen `[通讯]` (Allen Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于果蝇视网膜连接组的连续时间递归神经网络 FlyNet，并通过可微分 3D 渲染管线在逼近碰撞检测任务中与 CNN、RAFT、CTRNN 等人工网络进行对比评估。

**💡 创新点**

创新点在于将真实连接组约束与任务优化相结合，利用可微分渲染发现对抗纹理和逆 phi 幻觉共享同一低阶运动统计，从而揭示生物网络的鲁棒性与效率权衡。

**🔧 技术方法**

使用可微分 3D 渲染、连续时间递归神经网络、对抗优化（PGD、NES）、逆 phi 幻觉实验以及 Hassenstein–Reichardt 相关量等技术。

**📊 数据集**

使用 Sintel 光流数据训练网络，并在自定义 3D 逼近轨迹与纹理上评估，纹理包括黑白、棋盘、随机等。

**📈 对比分析**

通过对抗攻击和常见视频畸变的鲁棒性面积（ℛ）与 FLOPs 计算成本的对比，提出 LERS 指标，结果显示 FlyNet 在常见畸变下拥有最高的鲁棒性/效率比，但在对抗攻击下仍不如 CTRNN。

**⚠️ 局限性**

局限在于缺乏下游逃逸通路、解码器维度差异、单实例评估、仅针对二分类碰撞任务以及 FLOPs 与生物代谢成本的不同。

---

## 659. C$^2$-Routing: Complementary Cropping and Compression for Efficient Vision-Language Inference

**arXiv ID:** 2611.14469 | [PDF](https://arxiv.org/pdf/2611.14469v1)

**作者:** Guorui Lu `[一作]` (Leiden University), Qinyu Chen `[通讯]` (Leiden University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种模型无关的输入路由器C^2-Routing，能在视觉问答中自适应选择裁剪或压缩，并通过安全门控回退以降低计算量和视觉标记数。

**💡 创新点**

创新点在于将裁剪与全图压缩的互补性结合，使用联合正确性预测来构建安全门控，以及在已设视觉标记上限的条件下进一步节省 FLOPs。

**🔧 技术方法**

采用轻量级前端编码器（SmolVLM-500M）提取图像-问题特征，Grounding DINO 进行裁剪定位，分辨率头选择压缩级别，风险头预测联合正确性分布，并通过软阈值实现回退决策。

**📊 数据集**

实验使用了 206,996 条问题、18 个 VQA 数据集，涵盖局部细节、全局语境、文本/文档和混合感知任务。

**📈 对比分析**

与 CARES、CropVLM、ViCrop、ZoomEye、VisionZip 等方法比较，Balanced-A 方案在 1,024 视觉标记上限下平均减少 36.53% FLOPs、42.36% 视觉标记，仅损失 0.77% 任务分数；Safe-A 在保准分数上几乎无损。

**⚠️ 局限性**

局限性包括仅在视觉标记上限下评估，安全门控对不同模型的稳定性不足，且无法完全分离标记上限与路由带来的节省。

---

## 660. SkipLane: Write-Once, Conflict-Free Accumulation for Zero-Skipping CNN Engines on FPGAs

**arXiv ID:** 2611.14495 | [PDF](https://arxiv.org/pdf/2611.14495v1)

**作者:** Aditya Saboo `[一作]` (Indian Institute of Technology Indore), Pratibha Verma `[通讯]` (Indian Institute of Technology Indore)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种在 FPGA 上的 SkipLane 零跳过 CNN 加速器，采用写一次无冲突累积路径，每个卷积核行（lane）只写一次到专用银行，避免了传统的行缓冲读-改-写（RMW）操作；同时支持将两个滤波器打包在同一 3×3 引擎中，以利用乘法器空闲时间。

**💡 创新点**

创新点：
- 写一次累积（write‑once）：利用卷积行的有序性，保证每列在完成后立即写入不再被修改，消除部分和读回，显著降低存储冲突。
- 行分割 lane 与冲突‑free bank：每行拥有独立的 B 银行，列交错映射，保证任何时刻最多只能写入三列，天然冲突-free。
- 滤波器打包（filter packing）：一个 3×3 引擎可以同时处理两组权重，乘法器被零权重占用时直接计算第二组滤波器，提升乘法器利用率。

**🔧 技术方法**

使用技术：
- 3×3 行缓冲窗口 + 3‑步寄存器窗口寄存器（R0,R1,R2）实现按列累积。
- 跳跃控制 Δ（Delta）判断列差，按顺序完成列并写入。
- B = 4（或可配置）冲突‑free RAM banks，每行独立。
- 1‑bit filter tag + weight 切换实现两滤波器打包。
- RTL 设计，Vivado 2023.2，在 Xilinx Zynq‑7020 (PYNQ‑Z2) 上实现六个 3×3 引擎。
- 采用 AXI4‑Lite 配置、AXI DMA 数据传输，支持 140 MHz 时钟。
- LeNet‑5（3×3 kernel） 8‑bit 量化网络作为验证基准。

**📊 数据集**

数据集：MNIST（LeNet‑5 3×3 卷积），在 PYNQ‑Z2 上做 100 张测试图片的跑通，记录稀疏度 0–90%。

**📈 对比分析**

比较方法与性能：
- 与之前的稀疏引擎（queue‑based RMW）在相同前端（窗口、PE）下进行 RTL 仿真；在 70% 稀疏度下 SkipLane 与前端界限差异 ≤0.5%。
- 与 SCNN‑style 散点累积器（4/18 banks）在同一前端比较：在 50–80% 稀疏度下，SkipLane 速度提升 1.10–1.26×；在 90% 稀疏度下提升 3.7–5.7×。 
- 具体指标：
  * conv1：引擎周期 1.94× 更少，端到端 1.26× 更快；
  * conv2：引擎周期 1.05×，端到端 0.87×，受行宽限制。 
  * 最高 dense‑equivalent GOPS：25.0（51% 稀疏）和 36.9（69% 稀疏）。
- 资源占用：LUT 21.1k、FF 19.0k、BRAM36 81、DSP 54；比早期稀疏引擎缩减 23% LUT、增加 50% FF。

**⚠️ 局限性**

局限性：
- 只支持 3×3 stride‑1 卷积，无法直接扩展到更大核或更大步幅。 
- 滤波器打包仅能一次打包两组滤波器，1‑bit tag 限制多重打包。 
- 需要固定行宽（≤16）和行数（≤54）来保证不冲突。 
- 对极稀疏（>90%）或宽行（>13）性能下降，因行读取和输出读写成为瓶颈。 
- 需要额外 BRAM（81 个），在资源受限的 FPGA 上可能不适用。

---

## 661. Assessing Alignment in Automatic Piano Transcription Datasets: A Benchmark for Scoring Methods

**arXiv ID:** 2611.14443 | [PDF](https://arxiv.org/pdf/2611.14443v1)

**作者:** Lucas de Souza Brandão `[一作]` (Universidade de São Paulo), Andrea Poltronieri `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了音频与MIDI的对齐质量评估，提出了一个控制破坏的评估框架，并设计了一种基于AMT模型的STI评分方法。

**💡 创新点**

创新点在于将AMT模型的转录结果与传统特征对齐方法相结合，形成STI评分以捕捉细粒度的起始时间误差；同时构建了可控的音高、起始和结构破坏实验，配合四个新的评估指标。

**🔧 技术方法**

采用MIDI破坏生成实验数据，使用嵌入式（CLaMP3、CLAP）、特征对齐（Spectrogram、CENS、DLNCO）和起始F1、STI等多种评分方法，STI中利用Hungarian算法与IOI信息进行匹配。

**📊 数据集**

实验数据集包括MAESTRO、SMD、MAPS‑S（合成对齐）以及ENSTDkAm、ENSTDkCl、MusicNet等六个钢琴AMT数据集。

**📈 对比分析**

通过Monotonicity、Stability、Sensitivity、Consistency四个指标对方法进行比较，STI在检测细粒度起始时间偏差方面显著优于传统AMT F‑measure、DTW‑D等方法，整体性能最为稳健。

**⚠️ 局限性**

局限性包括对速度误差（起始）仍敏感；方法高度依赖合成音频质量和AMT模型的表现，且尚未扩展到多乐器或更复杂的对齐错误场景。

---

## 662. BoviShift: A Six-Year Multi-Attribute Audit of Visual Cattle Phenotyping Across Animals and Time

**arXiv ID:** 2611.14476 | [PDF](https://arxiv.org/pdf/2611.14476v1)

**作者:** Mobasshir Bhuiya Shagor `[一作]` (Graaho Technologies LLC), Md Saef Ullah Miah `[通讯]` (American International University Bangladesh)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究了多年畜牧视觉表型评估数据集BoviShift，评估图像随机分割与畜种离散分割以及未来年度的迁移性能。

**💡 创新点**

首次构建六年连续的牛群视觉数据集，并系统比较身份泄露与时间漂移对模型泛化的影响。

**🔧 技术方法**

采用预训练ResNet‑50为基准，进行多任务学习、视图聚合和零样本视频推理。

**📊 数据集**

使用BoviShift数据集，包含2,657头牛、10,610张静态图像、2,566段围行视频及五个属性标签。

**📈 对比分析**

通过四种分割配置（图像随机、牛种离散、未来2025、锁定2026）比较模型性能，发现身份泄露不一定导致普适退化，未来年度的重量、年龄和高度误差显著增大，而多任务和视图聚合在部分属性上提升。

**⚠️ 局限性**

限制包括单一数据来源、品种不平衡、未进行视频特定训练、评估只使用单一训练种子以及对测量误差与标注缺失的处理有限。

---

## 663. ReTrace: Distilling Tool-Using Agents into Planner-Free Workflows

**arXiv ID:** 2611.14492 | [PDF](https://arxiv.org/pdf/2611.14492v1)

**作者:** Kaiyuan Hou `[一作]` (Columbia University), Francesco Carbone `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将标准操作程序（SOP）与少量已记录的LLM代理运行日志转化为可复用的工作流，使得在执行时由代码而非LLM决定每一步。

**💡 创新点**

创新点在于利用代理的推理文本提取每一步的条件、输入来源和备选方案，聚合多条日志生成规则文档，然后一次性生成可执行工作流；构建阶段唯一调用LLM，运行时完全无规划调用；并引入多轮修复回路提升成功率。

**🔧 技术方法**

采用LLM（如Sonnet、Opus）进行步骤提取与规则聚合，使用决策列表和中间表示（IR）合成工作流；支持生成Python代码和Typed Workflow Graph，兼容多平台；还使用了工件修复模型和跨平台编译技术。

**📊 数据集**

实验数据集包括私有企业平台的79个生产流程；公开Benchmark为SOP‑Bench（8个域）和Telecom Support Benchmark（74训练+40测试），并使用Sonnet、Opus等LLM进行构建。

**📈 对比分析**

与原始LLM代理、AWM、Skill Factory、BPOP等方法对比；在企业数据上完成率93.8%，每任务耗时1.3 s，成本下降24%；在SOP‑Bench原始未修复80.6%，修复后达到87.8%；整体性能接近或略低于原始代理，但显著提升速度和降低成本。

**⚠️ 局限性**

局限在于高度依赖代理的推理准确性，缺乏对生成条件正确性的验证；需要足够覆盖不同分支的日志；修复过程需人工或多轮模型调用；对新SOP或工具更改需重新构建；在公共数据集上的实验范围有限，未证明跨域迁移效果。

---

## 664. AutoScaffold: AI-Guided Scaffolding for Reinforcement Learning of Language Models

**arXiv ID:** 2611.14515 | [PDF](https://arxiv.org/pdf/2611.14515v1)

**作者:** Zijian Zhang `[一作]`, Mingyi Hong `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在RL训练中引入一个冻结的教师LLM，自动诊断学生模型的失败模式并生成可插入训练提示的临时支持（class‑level 策略和 instance‑level hints），在训练期间周期性更新并验证这些支持，最终在评估时移除，提升学生的独立性能。

**💡 创新点**

创新点在于：① 完全自动化的教师决策过程，无需人工手工设计支持；② 通过交互式查询工具和记忆机制让教师自适应地诊断并改进支持；③ 统一的框架适用于多种任务结构（agentic、QA、数学推理、GPU kernel）和不同的RL算法；④ 引入了对 class‑level 支持的在线验证，避免无效或有害的支持对训练造成负面影响。

**🔧 技术方法**

使用的技术包括：RL with group‑relative policy optimization（GRPO、GiGPO、DAPO）、冻结教师LLM（GPT‑5.5）进行自然语言生成和推理、查询工具（class statistics、trajectories、rollout groups）、记忆机制、支持注入概率调控、以及支持验证的离线评估流程。

**📊 数据集**

实验所用数据集为：ALFWorld（六类家庭任务）、搜索式问答（NQ、HotpotQA 等）、KernelBench（GPU kernel 生成）、以及数学推理基准（AIME、AMC、MATH‑500 等）。

**📈 对比分析**

与无 scaffolds 的 RL 基线相比，AutoScaffold 在 ALFWorld 提升约 7.2 点、在搜索式 QA 提升约 4.0 点、在 GPU kernel 生成提升约 5 点；在数学推理上提升 1–2 点。与手工设计的 scaffolding 方法相比，它在同等模型规模下能够达到或超过竞争方法，且无需额外的手工支持或预设日程。

**⚠️ 局限性**

局限性包括：需要教师 LLM 的推理和生成成本，且每轮更新仍需额外评估和验证步骤；框架假设任务已可划分为若干类，跨域迁移时可能需要调整类划分；在非二进制奖励场景下的理论分析与实践效果尚待进一步验证；并且对极大规模训练和更复杂任务的适用性尚未充分评估。

---

## 665. Is Your Robot's Memory Just a Clock? Temporal Shortcuts in Memory-Augmented VLA Policies

**arXiv ID:** 2611.14488 | [PDF](https://arxiv.org/pdf/2611.14488v1)

**作者:** Chubin Zhang `[一作]` (Nanyang Technological University), Bo An `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了使用时间采样填充记忆的视觉-语言-动作(VLA)策略，并揭示其在出现空闲事件（pause、fidget、air‑grasp）时会错误地认为已完成任务（即“phantom progress”）

**💡 创新点**

创新点在于通过对记忆索引方式的因果诊断，发现时间索引导致了对空闲事件的误读，并证明将记忆索引改为事件驱动即可显著恢复性能；同时提供了一个基于空闲事件的测试协议

**🔧 技术方法**

利用强化学习的记忆增广模型（MME‑VLA）与交叉注意力机制、时序位置编码、以及自定义的事件索引策略；在实验中对时间缩放、时间编码移除等做了 ablation

**📊 数据集**

主要使用了RoboMME仿真数据集的两个计数任务（PickXtimes、SwingXtimes），并在其中注入了三类空闲事件；此外对比了真实遥控数据集以验证时间预测的有效性

**📈 对比分析**

对比了不同记忆索引（时间采样、像素变化、子目标、文本子目标等）在无事件与注入空闲事件两种条件下的50次测试集成功率；结果显示时间索引下成功率从47/50降至16/50；事件索引恢复到约38/50，表现显著优于原始策略

**⚠️ 局限性**

局限性包括仅在仿真环境下验证，空闲事件均为手工脚本；实验依赖于特定的MME‑VLA架构，未对所有记忆模型进行彻底分解；事件检测器在真实传感器下的鲁棒性尚未评估

---

## 666. Accurate predictions of properties of energetic materials with tabular foundation models

**arXiv ID:** 2611.14462 | [PDF](https://arxiv.org/pdf/2611.14462v1)

**作者:** Vagn Krister Nes Glomsrud `[一作]` (Norwegian Defence Research Establishment), Dennis Christensen `[通讯]` (Norwegian Defence Research Establishment)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文使用TabPFN这一新型表格基础模型预测能量材料的冲击敏感度、分解温度和晶体密度，且不需要任何超参数调优。

**💡 创新点**

创新点在于首次将TabPFN应用于能量材料性质预测，并证明其在多种输入特征（手工化学描述符、Chemprop嵌入）下均优于传统机器学习和已发表的最佳模型，同时保留不确定性分布估计与可解释性分析。

**🔧 技术方法**

主要技术包括TabPFN（Transformer‑based in‑context学习）、Chemprop消息传递神经网络嵌入、随机森林、XGBoost、Lasso、Gaussian Process、Feed‑Forward NN以及Shapley值解释。

**📊 数据集**

使用三大公开数据集：冲击敏感度（1,533个测量，516个化合物）、分解温度（565个化合物）以及晶体密度（12,792个有机化合物）。

**📈 对比分析**

通过10折交叉验证、平衡RMSE指标（η平衡重复测量）和R²评估，TabPFN在所有任务上均实现最高R²、最低RMSE，速度上比XGBoost快十倍以上；与VSURF、q‑RASPR等顶尖模型比较，TabPFN在所有任务均表现更好。

**⚠️ 局限性**

局限性包括未研究跨任务或跨数据集的迁移学习潜力、对Chemprop encoder进一步训练的负面影响尚未彻底解释、以及对大规模嵌入时的计算资源需求与模型解释性假设（特征独立性）等。

---

## 667. When does text encoding matter for language modeling?

**arXiv ID:** 2611.14459 | [PDF](https://arxiv.org/pdf/2611.14459v1)

**作者:** Eugene Jang `[一作]` (Northeastern University), Terra Blevins `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究不同文本编码（包括UTF-8、Unicode变体、语言专用编码以及自定义CuTF编码）对字节级语言模型性能的影响，并在四种正字法多样的语言上进行系统评估。

**💡 创新点**

提出Aether混合编码框架，在保留脚本特定编码优势的同时使用UTF-8回退，构建38个语言-编码组合的实验体系，发现粗粒度文字（汉字、韩文）对编码压缩度敏感，而细粒度文字（泰米尔、格鲁吉亚）则几乎不受影响。

**🔧 技术方法**

使用GPT‑2/LLaMA‑3 语言模型、字节级编码、CuTF自定义压缩编码、Aether tokenizer 以及宏平均负对数似然（Macro NLL）评估方法。

**📊 数据集**

采用 FineWeb‑Edu‑Chinese‑v2、AI Hub 韩语语料、Sangraha 泰米尔子集、FineWeb2 格鲁吉亚子集作为训练数据，评估集选用 FLORES。

**📈 对比分析**

所有模型在相同语料单 epoch 训练，仅变化编码方式；通过 Macro NLL 对比，中文和韩语显示压缩度与性能正相关（相关系数 0.91 与 0.69），而泰米尔和格鲁吉亚无显著相关；Jamo（固定长度子字符）在韩语中表现最好，超过最压缩的字符编码。

**⚠️ 局限性**

实验规模仅约 120M 参数，单语实验，未验证多语场景；模型可能生成无效/畸形序列，需进一步研究生成质量与规模化效果。

---

## 668. Re-Search: Dissecting LLM Ideation Through a Two-Player Questioning Game

**arXiv ID:** 2611.14487 | [PDF](https://arxiv.org/pdf/2611.14487v1)

**作者:** Muhan Gao `[一作]` (Texas A&M University), Kuan-Hao Huang `[通讯]` (Texas A&M University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Re-Search 这类二人问答游戏，将研究创意生成拆解为可观测的提问-回答步骤，评估 LLM 在创意探索中的表现。

**💡 创新点**

将创意生成视为问答循环，分离创意与执行，发现大多数模型在前 30 个问题即停滞，瓶颈在生成能力而非判断；提出利用外部候选问题或假设空间来显著提升弱模型。

**🔧 技术方法**

使用二人问答游戏设计、LLM 进行问答与评估、树形结构化分析搜索轨迹、信息增益与假设空间干预技术。

**📊 数据集**

基于 2026 年 ICML 200 篇方法论文构建的 200 对 (研究问题 p, 解决方案 s) 数据集，使用 GLM-5.2 自动提取并人工校验。

**📈 对比分析**

对六款 LLM（Gemma-4-31B、GPT-OSS-120B、DeepSeek-V4-Flash、Qwen3.6-35B-A3B、Claude Sonnet 4.5、GPT-5.4）进行 Re-Search 评估；GPT-5.4 领先，后四款约为其 2/3；模型大多在前 30 题后停滞；提供外部候选问题/假设空间可将弱模型提升至接近强模型。

**⚠️ 局限性**

仅测试机器学习论文，评估基于 LLM 判断存在主观性；对最新模型受知识污染限制；树形结构生成可能有误；未覆盖实验执行，只评估创意生成。

---

## 669. Metric-Aligned Group Relative Policy Optimization for Explainable Deepfake Detection

**arXiv ID:** 2611.14436 | [PDF](https://arxiv.org/pdf/2611.14436v1)

**作者:** Luka Dragar `[一作]` (University of Ljubljana), Borut Batagelj `[通讯]` (University of Ljubljana)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一套预测-解释分离的深度伪造检测与可解释系统，先用视觉基础模型判别真伪，再用Qwen3-VL-8B生成视觉证据解释。

**💡 创新点**

创新点在于将检测与解释解耦，利用决策条件下的提示让生成模型专注解释；同时采用GRPO强化学习将评测指标直接对齐到生成过程，提高解释质量。

**🔧 技术方法**

主要技术包括双分支LoRA微调的SigLIP2–DINOv2与DINOv3-MAC视觉检测器，Qwen3-VL-8B-Instruct作为复杂与简化解释器，以及基于BERTScore-F1与SLE的GRPO奖励机制。

**📊 数据集**

使用的训练数据为XPlainVerse（含真实/伪造图像及对应的复杂与简洁解释），外部增强集包括OpenFake、DFBench、SID-Set、NTIRE 2026 Robust AI-Generated Image Detection 等。

**📈 对比分析**

在ACM Multimedia 2026 Explainable Deepfake Detection Challenge 上与官方基线对比，复杂解释的BERTScore-F1达0.712，排名第一，但整体排名第七，检测F1排名第17。

**⚠️ 局限性**

主要局限是检测错误会导致解释偏差，简化解释过度依赖模式化短句降低多样性，且当前GRPO奖励未能充分覆盖实体与证据覆盖度。

---

## 670. Earlybird: Predictive Streaming Speech Recognition in 29M Parameters

**arXiv ID:** 2611.14421 | [PDF](https://arxiv.org/pdf/2611.14421v1)

**作者:** Sahil Mahendrakar `[一作]` `[通讯]`, Sahil Mahendrakar

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

训练一个29M参数的流式语音识别模型，使其能在说话者还未完成单词时就准确预测该单词并给出下两个词的猜测。

**💡 创新点**

创新点在于：①仅通过在随机截断点训练（无未来音频）来逼迫模型提前输出完整单词；②使用“教师只有正确时才加权”的蒸馏方式，避免模型输出后被撤回的情况；③加入三帧空白padding，让模型在未听完音频时完成单词；④利用原模型的输出作为对数概率的置信度，直接得到可校准的置信度。

**🔧 技术方法**

技术包括：Encoder‑Decoder Transformer（Moonshine Tiny 作为起点）、随机切点训练、目标扩展策略、教师蒸馏、padding帧、量化感知训练（4‑bit）以及置信度计算与ECE评估。

**📊 数据集**

数据集：LibriSpeech（960h）、People's Speech（约300h）、Synthetic Kokoro（约25h）以及各种测试集（LibriSpeech test‑clean/other、TED‑LIUM 3、People's Speech、Earnings‑22、Synthetic 500句）。

**📈 对比分析**

与现有流式模型（Moonshine Tiny runtime、Whistle、Whisper base/en）在六个测试集上对比，模型的中位单词延迟为-90 ms，明显早于其它模型；词错误率(WER)在8个Open ASR排行榜集上与Whistle相当，低于Moonshine Tiny；同时保持每秒约两倍于实时的推理速度，模型大小仅16.8 MB。

**⚠️ 局限性**

局限性：在背景噪声（尤其是伴随语音）下准确率下降；提前输出导致撤回次数增多；高置信度的下一个词猜测率低（约12–16%）；仅针对英文，跨域指令语音的校准性能显著下降。

---

## 671. A Lightweight Differentiable Method for Real-Time Skin Rendering and Digitization

**arXiv ID:** 2611.14451 | [PDF](https://arxiv.org/pdf/2611.14451v1)

**作者:** Mario Marquez-Balduque `[一作]` (Universidad Rey Juan Carlos), Jorge Lopez-Moreno `[通讯]` (Universidad Rey Juan Carlos)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种轻量级可微分逆渲染管线，直接从单视角、单光源拍摄的皮肤图像中估计实时皮肤着色器所需的多散射参数，实现实时皮肤渲染与数字人建模。

**💡 创新点**

创新点包括：①基于Burley 15 BSSRDF的可微分屏空间多散射（SSSS）框架，直接优化实时渲染所需的散射距离；②针对阴影区的遮罩损失函数，聚焦多散射的梯度，提升低对比度皮肤的估计精度；③采用极低成本的单光单视角捕捉硬件，降低采集门槛；④构建覆盖Monk皮肤色阶的多样化数据集；⑤多阶段分离优化策略，防止基色、粗糙度与散射参数耦合。

**🔧 技术方法**

使用技术包括：可微分渲染与Tensor化SSSS实现、Burley 15 物理模型、屏空间卷积与指数扩散近似、Adam优化器与多阶段学习率调度、残差高光映射、遮罩化损失、单目深度先验与Sobel法线预估、以及对图像和光照进行Otsu阈值化、形态学处理。

**📊 数据集**

数据集：225张面部皮肤补丁，来源于45名受试者，覆盖Monk 1–10色阶和5个面部区域（鼻、颊、颈、额、耳），每张补丁采集四张图像；另有17名受试者（85张补丁）用于多光源验证；并使用合成皮肤补丁验证参数恢复。

**📈 对比分析**

与合成地面真值、多光源捕获以及无遮罩损失的基准对比。指标包括FLIP、LPIPS、SSIM、PSNR。遮罩损失使FLIP平均下降约45%；单光方法在散射距离估计上与多光源结果相当；每张补丁的优化耗时约198 s（RTX 4070 Ti），峰值显存9.4 GB，性能稳定且易于在消费级硬件上复现。

**⚠️ 局限性**

局限性：假设局部均匀散射，单目深度先验可能忽略细小几何细节；未显式建模自遮蔽和极薄透明结构（如鼻腔、耳蜗）导致散射距离略高；在光照极端或多方向光照下精度下降；整个管线仍依赖单光源，复杂表面光照交互仅通过残差高光映射近似。

---

## 672. One Policy for 100 Household Tasks: Our Solution for the 2026 BEHAVIOR Challenge

**arXiv ID:** 2611.14432 | [PDF](https://arxiv.org/pdf/2611.14432v1)

**作者:** Ilia Larchenko `[一作]`, Anjana Yodaiken `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了单一视觉-语言-动作（VLA）模型与GPT‑6 Astra LLM代理，完成OmniGibson仿真环境中100个长期家庭任务的挑战。

**💡 创新点**

提出了任务迁移至100任务、语义阶段划分、方差降低的多样本流匹配、注意力寄存器与修剪、以及将LLM代理用于难度高的任务等创新改进。

**🔧 技术方法**

基于π_0.5架构的视觉语言模型+动作专家，结合多样本流匹配、分层时间采样、抗相关噪声、GPT‑6 Astra LLM、分布式训练（B300、H200+FSDP）与云评估基础设施。

**📊 数据集**

使用了211 M帧、20 000条演示的LeRobot数据集，其中包含原始50个任务以及新增的50个任务。

**📈 对比分析**

通过公开与私有评估对比，Q从2025年的0.260提升至0.371（共享任务），整体单模型Q 0.392，LLM代理在14个难题上平均Q 0.639，最终在公开测试中取得Q 0.446、27.3 %成功率并排名第9。

**⚠️ 局限性**

局限性包括LLM代理运行耗时长、API成本高、仅在少数任务使用、恢复策略无明显收益、对顺序与稳定性错误仍易受限，且缺乏大规模恢复数据与高效的代理部署方案。

---

## 673. TorchCurves: Differentiable Parametric Curves

**arXiv ID:** 2611.14516 | [PDF](https://arxiv.org/pdf/2611.14516v1)

**作者:** Alex Shtoff `[一作]` `[通讯]`, Alex Shtoff

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了 PyTorch 库 TorchCurves，用于将一维可微分参数化曲线（B‑spline 与 Legendre 多项式）作为可学习原语，提供模块化、基函数缓存和无状态函数 API，并演示其在连续特征嵌入、Kolmogorov‑Arnold 网络和函数回归中的应用。

**💡 创新点**

创新点在于把可微分参数化曲线作为统一可组合的学习原语，提供多层次接口、纯 PyTorch 向量化实现（Cox‑deBoor、Clenshaw）、并兼容现有网络结构；同时通过实验证明在无需自定义 CUDA 核心的前提下，性能仍可与专用实现相当。

**🔧 技术方法**

使用纯 PyTorch 代码实现了 B‑spline 与 Legendre 多项式的向量化评估和梯度传播，采用 Cox‑deBoor 算法、Clenshaw 递推；提供可训练模块、基函数缓存模块和无状态函数 API；实验使用了 CPU 与 GPU 的并行计算、PyTorch 自动微分。

**📊 数据集**

实验采用合成数据（随机生成的输入特征与标签）来评估 KAN 训练速度；曲线评估实验则基于 NumPy / SciPy 与 TorchCurves 的时间对比；未使用公开工业或学术数据集。

**📈 对比分析**

通过在 Intel Core Ultra 7 CPU 与 NVIDIA RTX PRO 500 GPU 上分别测量单次曲线评估与 KAN 训练步骤的时间，比较 TorchCurves 与 NumPy/SciPy、EfficientKAN 的性能；结果显示：在小规模（曲线数/批量小）时 TorchCurves 稍慢，但在大规模或多核并行下与 NumPy/SciPy 相当甚至更快；KAN 训练时间与 EfficientKAN 接近，GPU 上更具优势。

**⚠️ 局限性**

局限性包括：仅支持 B‑spline 与 Legendre 基础，缺乏专用 GPU 核心导致小规模下 Python 循环开销较大；实现基于纯 Python 循环的次数与曲线阶数相关；未涵盖更复杂或大规模真实数据集；API 仍需进一步完善对非 [-1,1] 区间的自定义映射。

---

## 674. Artificial Currency and Separate Selling: A 3-Approximation for Consumer Surplus

**arXiv ID:** 2611.14509 | [PDF](https://arxiv.org/pdf/2611.14509v1)

**作者:** Yang Cai `[一作]` (Yale University), Jinzhao Wu `[通讯]` (Yale University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究在多买家、独立值分布下，使用两种简单机制（逐项出售与人工货币）来最大化净消费者剩余；通过构造机制并证明在任意贝叶斯纳什均衡下均可获得最优剩余的1/3左右的常数近似。

**💡 创新点**

创新点在于：①提出“人工信用机制”并证明其能捕获基准机制的稳定部分；②将剩余分解为稳定部分与变动部分，分别由人工货币和逐项出售完成；③实现了在不做均衡选择、任意贝叶斯纳什均衡下的常数逼近，填补了此前仅在单参数或完全信息模型中已知的空白。

**🔧 技术方法**

核心技术包括：对准则下的贝叶斯激励兼容机制的可行性与收益分解；对目标分配概率的平滑（smoothness）偏差论证；对买家预算的比例分配与可比性；使用Cauchy–Schwarz与凸性证明稳健收益；构造单项机制以覆盖变动部分。

**📊 数据集**

该工作为理论分析，未使用实证数据集；所有结果基于独立分布的随机变量与期望值。

**📈 对比分析**

通过理论证明，组合机制在任意实例下均能保证至少最优剩余的1/3（可进一步趋近至1/3+η），相比单独使用逐项出售或人工货币时，后者可能仅获得无穷小份额。该结果与已有单参数或完全信息模型的常数逼近相当，且不依赖均衡选取。

**⚠️ 局限性**

局限性：仅适用于加性买家、值分布独立且均有有限均值的情况；对替代品/互补品、相关分布等更一般情形不一定成立；预算设定需在机制设计前预先确定，可能在实际应用中产生实施难度。

---

## 675. Online Graph Coloring: Competitive Ratios Separating Randomized and Deterministic Algorithms

**arXiv ID:** 2611.14513 | [PDF](https://arxiv.org/pdf/2611.14513v1)

**作者:** Ken-ichi Kawarabayashi `[一作]` (National Institute of Informatics), Masataka Yoneda `[通讯]` (University of Tokyo)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了在线图着色的新的竞争比下界和上界，并证明随机化算法在该问题上优于任何确定性算法。

**💡 创新点**

实现了 Ω(n / log n) 的确定性下界、O(n·(loglog n)² / log² n) 的随机化上界以及 O(n / (loglog n)²‑ε) 的确定性上界，几乎闭合了随机化与确定性之间的竞争比差距。

**🔧 技术方法**

运用了改进的对手模型、三层/多层随机化着色算法、在线对象着色框架、k‑团自由图的递归颜色分配以及对KYY27递归深度的显著缩短。

**📊 数据集**

论文仅进行理论分析，不使用任何实验数据集。

**📈 对比分析**

通过理论竞争比分析与已知的下界/上界进行比较，随机化算法的竞争比接近最优下界；确定性算法的竞争比显著提升，随机化算法相较确定性具有渐进优势。

**⚠️ 局限性**

仍存在 O((loglog n)²) 的残差；随机化下界与上界之间保持对数阶差距，确定性上界需引入 ε 以实现，且实现过程复杂度高。

---

## 676. Continual-ARC: Measuring Skill Acquisition over a Working Lifetime

**arXiv ID:** 2611.14505 | [PDF](https://arxiv.org/pdf/2611.14505v1)

**作者:** Daniel Lenton `[一作]` `[通讯]` (Unify), Daniel Lenton (Unify)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文设计并实现了 Continual-ARC benchmark，利用持续流式 ARC 任务评估记忆与权重学习在连续学习中的效果。

**💡 创新点**

创新点在于将持续学习、即时反馈计费、任务复现与多用户集体学习统一到单一 benchmark，并在同一测试框架下比较 harness 与权重更新方法。

**🔧 技术方法**

主要技术包括 Re-ARC 生成器、prime-agent/​hermes-agent/​openclaw harness、DAgger LoRA 微调、专家指导程序生成与统计学习评价。

**📊 数据集**

使用的数据集是 ARC-AGI-1/2/3 三个版本的 ARC 任务，结合 Re-ARC 生成的 400 训练规则和 100 评估规则以及对应的实例流。

**📈 对比分析**

方法评估通过与 reset baseline 的预算使用（U）、学习量（L）以及测试节省（S_seen / S_held）进行对比；结果显示大多数 harness 在记忆与泛化上几乎无提升，而 DAgger 权重更新在泛化方面显著优于 harness。

**⚠️ 局限性**

限制在于实验仅覆盖最易的 ARC-AGI-1 轨道、模型规模有限，harness 的提示设计不足导致学习效果不佳，未测试更强模型与更难轨道，且缺乏对隐私泄漏与多用户交互的深入分析。

---

## 677. RS-JEPA: Remote Sensing Joint-Embedding Predictive Architecture with Scan-Path Views for Cross-Sensor Image Retrieval

**arXiv ID:** 2611.14496 | [PDF](https://arxiv.org/pdf/2611.14496v1)

**作者:** Tanishq Rachamalla `[一作]` (Prince Mohammad Bin Fahd University), Swalpa Kumar Roy `[通讯]` (Tezpur University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于共享 Vision Mamba 编码器的 Compact Joint-Embedding Predictive Architecture（RS‑JEPA），实现无标签的跨传感器遥感图像检索。

**💡 创新点**

创新点包括：利用扫描路径视图通过不同读取顺序生成视图；引入对抗视图耦合（AVC）使预测任务更具挑战性；采用邻域一致性损失对齐不同传感器的相似度分布；全部技术均在单一模型中实现，无需额外教师网络或大规模变压器。

**🔧 技术方法**

技术手段包括 Vision Mamba 共享编码器、扫描路径视图、六层 Transformer 预测器、Adversarial View Coupling、邻域一致性损失、InfoNCE 对比学习、cosine 损失以及梯度反转等。

**📊 数据集**

使用 BEN‑14K、CBRSIR‑VS 和 DSRSID 三个跨传感器检索基准数据集（SAR–光学、PAN–MS 等）。

**📈 对比分析**

与12个重实现的基线（如 X‑JEPA、CR‑JEPA、DeCUR、SkySense 等）在 F1@5、mAP 等指标上对比，RS‑JEPA 在所有三个基准上均取得最高 F1@5（72.92/80.35/94.43），并且仅需 3.88 M 参数（训练时 14.57 M），显著降低计算量和存储开销。

**⚠️ 局限性**

局限性包括：仍需共注册的训练对；实验仅在 120×120 像素、固定扫描路径上进行；未验证跨区域或新传感器的迁移效果；训练所需额外参数与时间未做系统评估。

---

## 678. Comparative Analysis of Agent Evaluation Frameworks: Stability, Detection, and Discovery in Enterprise Analytics Agents

**arXiv ID:** 2611.14493 | [PDF](https://arxiv.org/pdf/2611.14493v1)

**作者:** Howard Zhang `[一作]` (Amazon), Lily Miao `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并执行三阶段比较实验，对比了三种评估框架（Trace‑based LLM judge、Text‑only LLM judge+红队、Deterministic heuristics）在稳定性、已知风险检测与未知风险发现上的表现，并在两款供应链分析智能体上进行验证。

**💡 创新点**

首次系统比较不同信息访问层次的评估框架，揭示 Trace‑based 评估在精确度与召回率上的优势、Text‑only 评估在安全发现上的互补性，并提出两框架联用的实用评估策略。

**🔧 技术方法**

采用 Frozen execution traces、OpenTelemetry、LLM‑as‑Judge（Claude Sonnet 4.6、Claude Haiku、Qwen3 32B）、PromptFoo 红队、Agenta deterministic 规则、统计稳定性指标（平均标准差、决策一致性）及 F1、FPR 等评估指标。

**📊 数据集**

使用两款企业级分析智能体的多阶段测试集，包括 72 个已标注失败案例，以及通过实验生成器与红队产生的 40+90+0 个未知案例。

**📈 对比分析**

通过多轮重复测评评估稳定性（≥97% 满足阈值），在已知风险检测中 Trace‑based 达到 F1=0.90，Text‑only 召回率相同但误报率提升 5 倍；未知风险发现共识别 24 个新缺陷，表明信息访问深度直接决定检测能力。

**⚠️ 局限性**

仅在实验室冻结轨迹上验证，未测试实时部署和 CI/CD 门控能力；框架性能受模型与配置交互影响，Trace‑based 变异性较大；对复杂维度的覆盖仍有限，且缺少跨多智能体的广泛验证。

---

## 679. Pterodactyl: Pushing the Limits of Efficient Automated Translation Evaluation

**arXiv ID:** 2611.14480 | [PDF](https://arxiv.org/pdf/2611.14480v1)

**作者:** Evelyn Y. Garland `[一作]` (1020 Group), Carola F. Berger `[通讯]` (1020 Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了Pterodactyl系统，使用LLM-as-a-judge方法在WMT26的子任务2（段级质量分数预测）和子任务3（错误无段检测）中实现低成本、可定制的质量评估；通过迭代式提示工程和ROC分析选择最佳模型与提示；并针对低资源语言提供了系统变体。

**💡 创新点**

提出了极简且高度可定制的工程流程，利用ROC筛选统一提示与模型选择，避免逐组件优化；通过在阈值变化下保持MCC稳定来提升对新注释方案的鲁棒性；同时兼顾成本和延迟的批量处理。

**🔧 技术方法**

采用LLM-as-a-judge范式、提示工程、ROC曲线与MCC阈值分析进行模型与提示选择；使用OpenAI gpt-5.4、gpt-5.6-terra以及NorMistral-11B进行实验，并通过分批处理降低token使用。

**📊 数据集**

使用WMT25 en→zh ESA数据、WMT23 en→de ESA数据，以及从wmt26-eng-sme生成的人工北萨米（Northern Sámi）数据；对这些数据进行70/30拆分；并以CometKiwi22和xCOMET-XL作为基准。

**📈 对比分析**

通过比较ROC曲线与ROC‑AUC值与基准模型，选取具有最佳AUC且MCC曲线更平稳的提示；最终系统在en→zh上与CometKiwi22/xCOMET-XL相当或略优，MCC对阈值变化鲁棒；在推理成本上，输入token约68.2M、输出token约10.7M，子任务3的二分类步骤无需额外LLM成本。

**⚠️ 局限性**

局限在于：1）WMT26使用新的contrastive ESA，真实黄金标准未知，可能影响评估；2）仅在两种ESA语言对上进行开发，低资源语言的人工数据为人工构造；3）系统在不同语言/领域的泛化性与对新注释方案的适应性尚待进一步验证。

---

## 680. UGG: Unified Generative Geometry Model

**arXiv ID:** 2611.14440 | [PDF](https://arxiv.org/pdf/2611.14440v1)

**作者:** Amani Kiruga `[一作]` (Harvard University), Yilun Du `[通讯]` (Harvard University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一的UGG框架，能够在同一模型中完成RGB视频、深度图和相机位姿的生成与预测，支持任意部分观测的缺失模态补全；

**💡 创新点**

创新点在于将视觉生成和几何预测视为同一生成模型中的条件去噪任务，通过噪声模式控制不同模态/时间点的条件和目标，并利用推理时多候选搜索提升几何一致性；

**🔧 技术方法**

使用多模态扩散Transformer（Mixture-of-Transformers）、Ray Map（稠密射线图）相机表示、Diffusion Forcing 与 Flow Matching 的去噪训练、History Guidance 等技术；

**📊 数据集**

在多种真实与合成数据集上训练和评估，包括 DL3DV‑10K、RealEstate10K、TartanAir、BEDLAM、Aria Synthetic Environments、OmniWorld‑Game、Virtual‑KITTI2、Mid‑Air、Sintel、ScanNetV2、ScanNet++、KITTI、7‑Scenes、Bonn 等；

**📈 对比分析**

与专门的相机控制生成、相机位姿估计和深度预测基线（如 Aether、SEVA、Geo4D、RayDiffusion、ChronoDepth、Marigold 等）进行对比，UGG 在相机控制生成、稀疏/密集观测下的相机姿态和深度预测上均取得最优或竞争性性能，且推理时搜索进一步提升了几何一致性；

**⚠️ 局限性**

局限性包括对比特定任务的专用模型仍表现更好，RGB 重投影误差作为搜索准则不够稳健，缺乏更强的几何监督与更大规模/高分辨率训练，导致在部分室内深度指标上仍不及专用方法。

---

## 681. New linear codes over GF(7), GF(8) and GF(9) from quasi-cyclic codes and Construction X

**arXiv ID:** 2611.14433 | [PDF](https://arxiv.org/pdf/2611.14433v1)

**作者:** Sebastian Cochinescu `[一作]` `[通讯]` (University of Bucharest), Sebastian Cochinescu (University of Bucharest)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在有限域 7、8、9 上构造了九个新的线性码，满足 [n,k,d]_q 的参数，并通过枚举计算了它们的最小距离和权重分布。

**💡 创新点**

创新点在于利用一种基于一生成器的准循环码与在每块上常数的向量相加，再结合 Construction X（追加重复码）的组合方法，成功将最小距离提高到已知下界之上（每组参数提高 1）。

**🔧 技术方法**

主要技术包括：准循环码的多项式表示、Construction X 代码扩展、对所有代码字进行完整枚举以获得权重分布、使用 Python/NumPy 与 SageMath 进行独立验证，以及 AI 辅助的搜索与编程。

**📊 数据集**

主要数据来源为 Grassl 的最佳已知线性码表、已发表的准循环码数据库（如 Chen 的数据库），以及作者通过搜索得到的准循环码多项式。最终验证时采用了完整枚举，产生的权重分布即为数据集。

**📈 对比分析**

与 Grassl 表格中的下界进行对比，发现每个新码的最小距离均比表中记录的下界高 1。性能方面，文中列出了每个码的权重分布（如 3 828 个权重 64 的码字），表明代码的实际距离和码字数已被精确计算并与三种独立方法一致。

**⚠️ 局限性**

限制包括：尚未证明这些码是最优的；搜索过程并非穷举，可能存在更优码；部分结果依赖 AI 编写的程序，虽然已多次独立验证；并且方法主要适用于较短长度的准循环码，难以直接推广到更大规模。

---

## 682. TanCRAFT: Context-Reasoned Anticipatory Feedback for Assembly Tasks

**arXiv ID:** 2611.14470 | [PDF](https://arxiv.org/pdf/2611.14470v1)

**作者:** Sassan Mokhtar `[一作]` (University of Bonn), Juergen Gall `[通讯]` (University of Bonn)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一套实时预测辅助系统，能在非顺序装配中追踪场景、推断部分装配、预测活物件运动并提前警告错误。

**💡 创新点**

通过将装配状态推理与短期运动预测相结合，提前检测可兼容的子图容量并在自由构造顺序下实现针对性错误预警。

**🔧 技术方法**

采用SAM3进行遮罩跟踪、MediaPipe Hands获取手姿、Kalman滤波平滑轨迹、MLP分类子图、TCN运动预测、GRU意图判定，以及阈值驱动的警告策略。

**📊 数据集**

通过自建的Tangram多视角装配数据集（152次会话、8小时）收集训练与评估数据，并在Pilot阶段收集72次组装会话。

**📈 对比分析**

通过离线模块基准与端到端学习模型（GRU、Transformer等）比较，系统在错误预测上达最高Precision/F1，用户研究显示相对无反馈和反应性反馈，误差率降低约70%，工作量和完成时间均显著下降。

**⚠️ 局限性**

仅验证于Tangram小型零件、固定子图划分，未评估更复杂装配或不同件种类，且持续反馈可能增加认知负荷；对未知装配目标的泛化仍有限。

---

## 683. Demonstration of MANA: Multi-perspective Alignment of NPC Agents

**arXiv ID:** 2611.14431 | [PDF](https://arxiv.org/pdf/2611.14431v1)

**作者:** Amir Tahmasbi `[一作]` (Purdue University), Jason Wu `[通讯]` (Purdue University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为MANA的交互式工具，允许开发者在Minecraft游戏中直接控制NPC进行演示，随后系统将演示数据转换为通用行为策略，使NPC能够在未见情境中自主执行对应动作和对话。

**💡 创新点**

创新点在于：①多视角实时掌控NPC，②通过语音与动作共同记录实现更丰富的角色个性构建，③利用LLM自动提炼自然语言人格与可解释的行动策略，④实现对不同NPC的多元化、可持续行为对齐。

**🔧 技术方法**

采用Minecraft模组实现演示与记录；使用LLM（如ChatGPT）进行演示内容解析与策略生成；构建基于邻域信息的动作-上下文对齐模型；利用自然语言生成模块产生NPC对话。

**📊 数据集**

使用的“数据集”是游戏内实时收集的演示记录（语音流与动作-邻域对），未使用公开的大规模游戏数据集。

**📈 对比分析**

通过对比两名NPC（Fletcher与Farmer）的行为表现，在约两分钟的演示后，NPC在新场景中均能持续、可信地展现各自的攻击或逃避策略，实验表明策略在不同时间、不同环境下保持一致性。

**⚠️ 局限性**

局限性包括：①对演示内容的覆盖范围有限，难以捕捉所有潜在游戏情境；②依赖LLM的质量和推断能力，可能导致策略不稳定或出现意外行为；③目前仅在Minecraft单机环境验证，缺乏跨平台或多人游戏的评估；④缺乏量化性能指标与系统性对比实验。

---

## 684. Quantifying the Sampling Frequency Invariance of Time-Series Foundation Models for Sensor Data Applications

**arXiv ID:** 2611.14425 | [PDF](https://arxiv.org/pdf/2611.14425v1)

**作者:** Sourish Gunesh Dhekane `[一作]`, Shabnam Ghaffarzadegan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文针对时间序列基础模型（TSFM）在不同采样频率（SF）下的鲁棒性进行系统性压力测试，评估模型在嵌入、预测和下游分类三维度上的SF不变性；

**💡 创新点**

创新点在于首次将SF不变性作为核心评估维度，引入了频率敏感性指数（FSI）、预测退化指数（FDI）和分类退化指数（CDI）等量化指标，并设计对比学习对齐头来减少迁移缺口；

**🔧 技术方法**

技术上使用了带限算子（band‑limiting）模拟SF下降，计算嵌入余弦距离、归一化RMSE和宏F1差异，结合NT‑Xent对比学习提升TSFM在不同SF下的表征；

**📊 数据集**

实验数据集包括三类公开传感器数据：HHAR（加速计/陀螺仪）、CWRU轴承振动（12kHz/48kHz）以及MIT‑BHI心电（125Hz）；

**📈 对比分析**

通过三维度指标对四个主流TSFM（MOMENT、Chronos‑2、Moirai‑MoE、TimesFM）进行比较，结果显示MOMENT与TimesFM在CDI和转移缺口上表现最优，且对齐头平均提升约40%，但部分模型仍存在较大迁移缺口；

**⚠️ 局限性**

局限性包括：未覆盖所有TSFM架构（仅Transformer族）、SF不变性并非绝对可实现、对齐头在保留原始SF性能上有一定下降，以及实验仅在三类传感器数据上验证，缺乏更广泛的跨域验证。

---

## 685. Pyrite: Building and Benchmarking Protein-Ligand Pose Ranking Agents

**arXiv ID:** 2611.14511 | [PDF](https://arxiv.org/pdf/2611.14511v1)

**作者:** Megha Srivastava `[一作]` (Stanford University), Nate Gruver `[通讯]` (Genesis Molecular AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了Pyrite——一个基于配体配位姿对比的50个蛋白–配体复合物，包含1,500对姿势的公开基准；开发了两种LLM驱动的代理（PyriteReasoner和PyriteFeaturizer）用于对姿势进行排名；对比人类专家和多种LLM模型，并使用RL（GRPO）提升性能。

**💡 创新点**

①首次公开提供高多样性、对抗性强的配体配位姿对比基准；②提出PyriteRep结构化表示显著提升LLM推理效果；③展示LLM既可直接推理也可自动生成特征并提升传统逻辑回归模型。

**🔧 技术方法**

大型语言模型（Opus 4.8、GPT‑5.6、Sonnet 5、Qwen3.8‑27B），PyriteRep结构化数据，强化学习（GRPO），逻辑回归与其他传统机器学习模型。

**📊 数据集**

Pyrite基准：从自有co‑folding模型生成的高质量姿势，筛选后得到1,500对；配体化学信息来自RDKit、OpenMM、Mogul、PoseBusters等工具。

**📈 对比分析**

与人类专家（5名）和多种物理评分（MM‑GBSA、交互计数、Mogul等）比较；LLM代理在后期AF3数据集上取得67–74%准确率，优于专家（约45%）和传统物理评分（约30–50%）。

**⚠️ 局限性**

仍有显著误差：LLM对姿势差异细微时易失效；训练数据受限于pre‑AF3集，导致与post‑AF3分布偏移；代理对物理评分的偏好仍需进一步校正。

---

## 686. Medical Image Segmentation: Foundation Model Adaptation versus Task-Specific Training

**arXiv ID:** 2611.14519 | [PDF](https://arxiv.org/pdf/2611.14519v1)

**作者:** Emre Kutlusan `[一作]` (Brunel University of London), Yongmin Li `[通讯]` (Brunel University of London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

比较医学影像分割中基础模型适配与从零训练的任务特定网络，在三个成像域（结肠镜息肉、乳腺超声、皮肤病变）上进行域内与跨域评估。

**💡 创新点**

①在分离域内/域外评估的同时系统考察不同适配容量（冻结、BitFit、LoRA rank 4/8/16/32、全微调）对跨域性能的影响；②发现适配容量是域属性而非方法属性；③给出实用的适配与部署建议。

**🔧 技术方法**

使用 DINOv2‑Small 编码器+线性解码器；参数高效适配方法 LoRA 与 BitFit；全微调；Plain U‑Net 与 Residual U‑Net（nnU‑Net 方案）；多指标（DSC、IoU、敏感度、精确度、HD95、ASSD、NSD）评估；Wilcoxon 符号秩检验；中心化核对齐（Centered Kernel Alignment）分析。

**📊 数据集**

结肠镜息肉：Kvasir‑SEG、CVC‑ClinicDB、CVC‑ColonDB、ETIS‑LaribPolypDB、CVC‑T；乳腺超声：BUSI、BrEaST、BUS‑UCLM；皮肤病变：ISIC 2018、ISIC 2016*、ISIC 2017*、PH2。

**📈 对比分析**

通过在每个域内设定最优学习率、训练预算后，对比适配后 DINOv2（LoRA r=32）与两种 U‑Net；域外实验显示 DINOv2 在 DSC、IoU、敏感度、HD95 上优于两种 U‑Net，优势随训练样本减少而增强；域内无显著优势；在适配容量方面，冻结→BitFit 就能获得大部分收益，LoRA r=32 与全微调在 DSC 上相当或更优，低rank 在不同域上所需容量差异显著。

**⚠️ 局限性**

仅使用单一基础模型（DINOv2‑Small）；BitFit 进行了学习率搜索，LoRA 与全微调未同步；大部分实验仅用单种子；仅覆盖三种成像域；训练集规模有限，未检验更大/更小的数据量对结果的影响；使用极简线性解码器，未探究更复杂解码器的影响；内部验证结果可能对域外评估产生偏差。

---

## 687. FERN: Cross-Benchmark Table Retrieval for Text-to-SQL with Frozen Retrievers and Learned Ranking

**arXiv ID:** 2611.14477 | [PDF](https://arxiv.org/pdf/2611.14477v1)

**作者:** Arjun Rao `[一作]` (SAP), Nishant Gopinath `[通讯]` (SAP)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于固定检索器与学习排序器的文本到SQL表检索框架FERN，旨在在无生成式LLM的情况下实现高效表检索。

**💡 创新点**

创新点在于将三种冻结检索器（BM25、Qwen3、BGE）联合生成100候选表，再用梯度提升学习者（LightGBM）对120维手工特征进行一次性训练，从而在跨库环境下达到接近LLM方案的召回率。

**🔧 技术方法**

技术包括词向量检索、BM25、Dense embedding、特征工程（包括数据库、词汇、覆盖、证据、数值、类型、图结构等多维度特征）以及LightGBM排名器。

**📊 数据集**

实验使用BIRD、Spider、Fiben三大公共表检索基准以及企业级BEAVER仓库，覆盖数百至数千张表。

**📈 对比分析**

与公开的LLM基线RASL对比，FERN在表召回率上仅落后0.3点（Spider）或0.9点（BIRD带证据），且在15表召回上可与RASL持平或略优；在低表预算下通过排序提升完整表集覆盖率，整体延迟约200ms，成本远低于LLM调用。

**⚠️ 局限性**

局限包括：需要先在公共数据上训练，迁移效果受限于目标数据库特征差异；缺乏对单表候选之外的召回；在表命名不规范或单数据库场景（如Fiben）效果显著下降；且未覆盖数据库路由与SQL生成完整流程。

---

## 688. TimeSeek: Multi-Task and Multi-Domain Pretraining for Generalized Time Series Analytics

**arXiv ID:** 2611.14491 | [PDF](https://arxiv.org/pdf/2611.14491v1)

**作者:** Peng Chen `[一作]` (East China Normal University), Chenjuan Guo `[通讯]` (East China Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TimeSeek，一种多任务、多域的时间序列基础模型，能够在预训练阶段同时学习预测、异常检测与分类任务，并在下游零样本或少样本情境下取得显著性能。

**💡 创新点**

核心创新在于：1）多域特征提取器结合渐进式多尺度建模与自适应槽式通道建模，能够动态捕获不同域的时间依赖与通道相关性；2）多任务协同训练机制，既保留任务特异性，又通过共享特征与双重损失实现任务间互补。

**🔧 技术方法**

技术手段包括：Transformer架构、旋转位置嵌入、PatchTST式分块、门控融合、通道路由与槽注意力、非自回归预测解码器、掩码重构模块等。

**📊 数据集**

预训练使用约10亿条来自能源、气象、交通、网络、医疗、经济等领域的公开时序数据；下游评测覆盖ETT、Weather、Traffic、Solar、Exchange、GIFT-Eval、SMD、MSL、SMAP、SWaT、PSM、UEA10分类等多项基准。

**📈 对比分析**

与现有时间序列基础模型（Timer、TimesFM、Chronos等）及任务特定模型（iTransformer、PatchTST、DLinear等）对比，TimeSeek在零样本预测、异常检测和分类任务中均达成SOTA，零样本MAE/MSE平均下降约15%~25%，异常检测F1提升数个百分点，分类准确率提升约7%。

**⚠️ 局限性**

局限性包括：1）主要聚焦三大任务，未涵盖如生成式时序建模等；2）对极端异构域（如高频金融与低频气候）在通道槽数调优上仍需手工设定；3）虽然参数规模较大但仍低于极大模型，但在推理延迟与能耗上仍需进一步优化。

---

## 689. Emergent Biological Capabilities in a Foundation Model for Molecular Interactions

**arXiv ID:** 2611.14504 | [PDF](https://arxiv.org/pdf/2611.14504v1)

**作者:** Adi Mashiach `[一作]` `[通讯]` (Output Labs), Adi Mashiach (Output Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

训练了一个基于 Transformer 的自监督基础模型 Bonbon，仅使用配对蛋白‑配体序列，学习并在冻结表示上实现零拷贝机制预测、结合位点定位以及共价/非共价结合识别，并在外部分布测试中得到验证。

**💡 创新点**

创新点在于：①在分子相互作用领域首次展示三阶层次的零拷贝 emergent 能力；②采用极高的参数对 token 比 1:2000 的数据密集扩展；③通过单一表示空间实现多任务能力并获得实验验证。

**🔧 技术方法**

技术：Transformer 架构、self‑supervised 语言建模、冻结表示 + 参数无关签名、规模化训练、参数对 token 1:2000；使用 FlashAttention、rotary embedding 等高效实现。

**📊 数据集**

数据集：约 3.2 万亿 token 的配对蛋白‑配体序列，来源于专有筛选数据和合成交互数据；评估使用公开数据库（BindingDB、PCBA、CASP16、FEP+4、OpenFE 等）以及 9 个疾病靶点的实验验证。

**📈 对比分析**

与传统 co‑folding、结构基方法相比，Bonbon 在二进制结合预测、亲和力预测以及实验验证中达到或超过 state‑of‑the‑art，并且计算速度提升 5–6 个数量级。

**⚠️ 局限性**

局限：目前仅能识别粗粒度的功能、位点与结合模式，缺乏 3D 坐标提取、时间动力学预测；未做 fine‑tuning，未探索更细粒度的特性；对模型权重公开有限。

---

## 690. A Theory of Policy Optimization in Post-Training: Coverage Collapse and Reference Resets

**arXiv ID:** 2611.14467 | [PDF](https://arxiv.org/pdf/2611.14467v1)

**作者:** Audrey Huang `[一作]` (Harvard University), Dylan J. Foster `[通讯]` (Microsoft Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文从理论角度系统阐述了语言模型后训练（post‑training）中的策略优化问题，揭示了基于“覆盖度”的收敛瓶颈并提出了通过参考重置实现覆盖度保持的元算法框架。

**💡 创新点**

创新点包括：①证明了固定参考或无正则化的策略梯度会出现指数级的覆盖度崩塌；②引入参考重置的近似策略镜像下降（APMD）元算法，并给出全局收敛和覆盖度保持的定理；③在此框架下对多种主流方法（如带 KL 正则化的策略梯度、带裁剪的 PPO、Kimi 风格的平方损失方法以及配对平方损失方法）给出了完整的样本与梯度步数上界，实现了理论与实践的一致性。

**🔧 技术方法**

主要技术手段包括：KL 正则化、参考重置、近似策略镜像下降、重要性加权梯度、裁剪技术、平方损失与均方误差分析、软最大线性策略模型、局部凸/凹性分析和大规模样本复杂度估计。

**📊 数据集**

本工作基于理论建模，未采用具体数据集；实验验证采用公开的语言模型提示与奖励回合，主要关注理论证明与抽象分析。

**📈 对比分析**

通过理论上对比，覆盖度保持算法在样本与梯度步数上均显著优于传统固定参考方法；在实验层面，本文的 APMD 及其实例化实现了与现有 PPO、Kimi 等方法相当甚至更优的收敛速度，且保持了对基模型覆盖度的有效利用。

**⚠️ 局限性**

局限性包括：仅在软最大线性策略（冻结特征）下给出完整证明，未覆盖更复杂的神经网络结构；研究聚焦于一次性推理（one‑shot）设置，未考虑多步交互或长序列生成；覆盖度仅作为基模型质量的单一衡量，未探讨其他潜在特性。

---

## 691. DnNP: Denoising Input Uncertainty in Neural Processes

**arXiv ID:** 2611.14424 | [PDF](https://arxiv.org/pdf/2611.14424v1)

**作者:** Fatemeh Tohidian `[一作]` (Northeastern University), Jennifer Dy `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 Denoising Neural Processes（DnNP），一种在神经过程框架下显式建模输入噪声的元学习模型。

**💡 创新点**

创新点在于把无噪声输入视为潜在变量，加入变分去噪器和对应的 ELBO 目标，使模型能同时学习去噪与函数估计，并提供理论证明：噪声下的贝叶斯最优预测属于该模型类，且相较于传统 NP 的 “直接替换” 方法存在不可避免的 KL 散度缺口。

**🔧 技术方法**

核心技术包括：变分推理、去噪器网络（q_ψ）、全局潜变量 z 的 NP 结构、基于 ELBO 的端到端训练，以及对输入噪声协方差 Γ 的显式建模。

**📊 数据集**

使用了四个数据集：1) 1D GP 回归（合成）、2) MNIST 图像补全（合成）、3) PM2.5 空间插值（真实城市空气质量传感器数据）、4) NOAA 加州海平面气压插值（真实气象站数据）。

**📈 对比分析**

与标准 NP、CNP、ANP、ConvCNP 等基线相比，DnNP 在所有任务上均取得更低的 NLL、RMSE，尤其在 PM2.5 数据上 RMSE 降低约 30%，NLL 降低 73%；在 MNIST 图像补全任务中，准确率在中等噪声水平下几乎恢复至无噪声基线。

**⚠️ 局限性**

局限性包括：假设输入噪声为独立且协方差已知；未处理噪声相关或标签噪声；模型复杂度较高，去噪器的泛化能力依赖于噪声分布假设；在极端噪声水平下仍可能失效。

---

## 692. Solomonoff Induction with Reflective Oracles

**arXiv ID:** 2611.14454 | [PDF](https://arxiv.org/pdf/2611.14454v1)

**作者:** Damini Kusum `[一作]` `[通讯]`, Damini Kusum

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文探讨了反射性索罗门诺夫归纳的最优预测器是否能够与仅指定部分数据生成过程的有效预测策略竞争。

**💡 创新点**

创新点在于提出了一种反射性计算模型，通过反射性预言者的条件概率构建假设，并证明这些假设在累积对数损失方面受限于描述长度。

**🔧 技术方法**

使用了反射性oracle和概率图灵机的技术，结合了下半可计算半测度的概念。

**📊 数据集**

使用了无限二进制序列作为数据集，并考虑了计算模型的完整性和不完整性。

**📈 对比分析**

通过构造比较假设来评估反射性预测器与部分模型的性能，结果表明反射性预测器在满足条件概率的情况下，能够接近允许的概率区间，且在每个兼容概率法则下几乎肯定收敛。

**⚠️ 局限性**

限制在于反射性预测器的条件概率在某些情况下可能无法计算，且在某些情况下，主导假设可能无法从预测器中计算得出。

---

